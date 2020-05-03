from typing import List, Tuple
import torch
from torchsummary import summary

from vgg import VGG19Features


class GeneratorBlock(torch.nn.Module):

    def __init__(self, prev_channels: int, vgg_channels: int, out_channels: int, block_size: int, dropout_rate: float = 0.2, initial: bool = False) -> None:
        """ GeneratorBlock: Single block of cascaded generator. Each of these blocks receives three inputs (previous block output, vgg feature tensor, and downsampled copy of main input)
            and concatenates all of them before two convolutional layers are applied. Therefore the previous block activations are bilinearly resized to the block_size of the current GeneratorBlock.
            The initial GeneratorBlock in the cascade does not receive a previous activation tensor.

        Parameters
        ----------
        prev_channels : int
            Number of incoming channels from the previous GeneratorBlock. This is irrelevant if initial is True.
        vgg_channels : int
            Number of incoming channels from the vgg feature tensor of this GeneratorBlock.
        out_channels : int
            Number of outgoing channels of this GeneratorBlock.
        block_size : int
            Spatial size of the tensors in this GeneratorBlock. The main input tensor and the output of the previous GeneratorBlock are bilinearly resized
            to this block_size. The vgg feature tensor is NOT resized.
        initial : bool
            Flag that indicates whether or not this GeneratorBlock is the start of the cascade.

        """
        super(GeneratorBlock, self).__init__()

        # store flag
        self.initial = initial

        # define bilinear resizing
        self.bilinear_resizer = torch.nn.Upsample(size=block_size, mode='bilinear')

        #self.vgg_mapping = torch.nn.Sequential(
        #    torch.nn.Conv2d(in_channels=vgg_channels, out_channels=vgg_channels, kernel_size=3, padding=1, bias=True),
        #    torch.nn.LeakyReLU(),
        #    torch.nn.InstanceNorm2d(num_features=vgg_channels)
        #)

        self.layer_norm = torch.nn.LayerNorm(normalized_shape=(vgg_channels, block_size, block_size))

        # define this blocks processing layers
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=vgg_channels+1+(prev_channels if not initial else 0), out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(normalized_shape=(out_channels, block_size, block_size)),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.LayerNorm(normalized_shape=(out_channels, block_size, block_size)),
        )

    def forward(self, input: torch.Tensor, prev: torch.Tensor, vgg_feature: torch.Tensor) -> torch.Tensor:
        """ Forwards the three input tensors of the GeneratorBlock through this block by bilinearly downsampling the main input, bilinearly upsampling the
            output of the previous GeneratorBlock, concatenation of these with the given vgg feature tensor, and feeding it through two convolutional channels
            each followed by batch normalization and a leaky relu activation.

        Parameters
        ----------
        input : torch.Tensor
            Main input tensor of all GeneratorBlocks. This input gets bilinearly resized to the size of the respective GeneratorBlock.
        prev : torch.Tensor
            Incoming output tensor of the previous GeneratorBlock. This input gets bilinearly resized to the size of this GeneratorBlock.
        vgg_feature : torch.Tensor
            Incoming vgg feature tensor. This input will NOT be bilinearly resized.

        Returns
        -------
        torch.Tensor
            Output tensor of this GeneratorBlock. This tensor is either used as the input to the next GeneratorBlock in the cascade or to the
            final block that reduces it to two channels.

        """

        # resize the input to the GeneratorBlock's block_size
        input = self.bilinear_resizer(input)

        # normalize the vgg feature tensor
        vgg_feature = self.layer_norm(vgg_feature)

        # concatenate all available inputs
        if self.initial:
            x =  torch.cat([input, vgg_feature], dim=1)
        else:
            prev = self.bilinear_resizer(prev)
            x = torch.cat([input, prev, vgg_feature], dim=1)

        # feed the concatenated inputs through the block
        x = self.layers(x)

        return x


class Generator(torch.nn.Module):

    def __init__(self, block_sizes: Tuple[int, ...] = (256, 128, 64, 32, 16),
                       block_out_channels: Tuple[int, ...] = (32, 64, 128, 256, 512),
                       vgg_layer_idxs: Tuple[int, ...] = (3, 8, 17, 26, 35),
                       vgg_layer_channels: Tuple[int, ...] = (64, 128, 256, 512, 512)) -> None:
        """ Generator: Cascaded generator consisting of a cascade of GeneratorBlocks followed by a final block that reduces the last GeneratorBlock's output to two channels.
            This type of generator has been designed to map a single lightness (grayscale) L input channel onto its two AB color channels in the LAB color space. The
            architecture can be understood as a modified cascaded refinement network that additionally utilizes the intermediate feature layer activations of a pre-pretrained
            VGG19 model and incorporates them during the recursive computation through the cascaded generator.

        Parameters
        ----------
        block_sizes : Tuple[int, ...]
            Tuple of spatial sizes of the tensors in all GeneratorBlocks. Currently restricted to square shapes. The first block in this Tuple refers to the last block in the cascade.
        block_out_channels : Tuple[int, ...]
            Tuple of number of outgoing channels for each of the GeneratorBlocks. The first block in this Tuple refers to the last block in the cascade.
        vgg_layer_idxs : Tuple[int, ...]
            Tuple of VGG feature layer indices. The first block in this Tuple refers to the last block in the cascade. The block sizes given in the block_sizes Tuple
            must match the spatial sizes of the vgg layers since they are not bilinearly resized.
        vgg_layer_channels : Tuple[int, ...]
            Tuple of number of channels of each VGG feature layer. The first block in this Tuple refers to the last block in the cascade. The layer channels in this Tuple
            must match the number of channels in the hidden VGG layers referred to the vgg_layer_idxs Tuple.

        """
        super(Generator, self).__init__()

        # assertion
        B = len(block_sizes)
        assert len(block_out_channels) == B and len(vgg_layer_idxs) == B and len(vgg_layer_channels) == B

        # define VGG19 feature extraction module
        self.vgg_features = VGG19Features(vgg_layer_idxs)

        # build cascade of generator blocks
        self.blocks = torch.nn.ModuleList()
        for b in range(B):
            # derive block hyperparameters
            prev_channels = 0 if b == 0 else block_out_channels[-b]
            vgg_channels = vgg_layer_channels[-(b+1)]
            out_channels = block_out_channels[-(b+1)]
            block_size = block_sizes[-(b+1)]
            initial = b == 0

            # append block to cascade
            self.blocks.append(
                GeneratorBlock(prev_channels=prev_channels, vgg_channels=vgg_channels, out_channels=out_channels, block_size=block_size, initial=initial)
            )

        # build final block that reduces all channels of the last GeneratorBlock to two channels and applies a tanh activation
        self.final_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=block_out_channels[0], out_channels=2, kernel_size=3, padding=1, bias=True),
            torch.nn.Tanh()
        )


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # feed through vgg and get intermediate layer activations
        vgg_features = self.vgg_features(torch.cat([input]*3, dim=1))

        # feed through the entire cascade of GeneratorBlocks
        x = None
        for b, block in enumerate(self.blocks):

            x = block(input, x, vgg_features[-(b+1)])

        # feed through the final GeneratorBlock
        x = self.final_block(x)

        return x
