from typing import List, Tuple
import torch
from torchsummary import summary

from vgg import VGG19Features


class GeneratorBlock(torch.nn.Module):

    def __init__(self, prev_channels: int, vgg_channels: int, out_channels: int, block_size: int, initial: bool = False) -> None:
        """ Constructor method.

        Parameters
        ----------
        prev_channels : int
            Number of incoming channels from the previous GeneratorBlock. This is irrelevant if initial is True
        vgg_channels : int
            Number of incoming channels from the vgg feature tensor of this GeneratorBlock
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
        self.bilinear_resizer = torch.nn.Upsample(size=block_size, mode='bilinear', align_corners=True)
        # define batch normalization
        self.batch_norm = torch.nn.BatchNorm2d(vgg_channels)

        # define this blocks processing layers
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=vgg_channels+1+(prev_channels if not initial else 0), out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU()
        )

    def forward(self, input: torch.Tensor, prev: torch.Tensor, vgg_feature: torch.Tensor) -> torch.Tensor:
        """ Forwards the three input tensors of the GeneratorBlock through it by bilinearly downsampling the main input, bilinearly upsampling the
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
            Output tensor of this GeneratorBlock. This tensor is either used as the input to the next GeneratorBlock in the cascade or to The
            final block.

        """

        # resize the input to the GeneratorBlock's block_size
        input = self.bilinear_resizer(input)
        # batch normalize the vgg feature tensor
        vgg_feature = self.batch_norm(vgg_feature)

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
                       block_out_channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
                       vgg_layer_idxs: Tuple[int, ...] = (3, 8, 17, 26, 35),
                       vgg_layer_channels: Tuple[int, ...] = (64, 128, 256, 512, 512)) -> None:
        """ Constructor method.

        Parameters
        ----------
        block_sizes : Tuple[int, ...]
            Tuple of spatial sizes of the tensors in all GeneratorBlocks. Currently restricted to square shapes. The first block in this Tuple refers to the last block in the cascade.
        block_out_channels : Tuple[int, ...]
            Tuple of number of outgoing channels for each of the GeneratorBlocks. The first block in this Tuple refers to the last block in the cascade.
        vgg_layer_idxs : Tuple[int, ...]
            Tuple of VGG feature layer indices. The first block in this Tuple refers to the last block in the cascade. The block sizes given in the block_sizes Tuple
            must match the spatial sizes of vgg layer since they are not bilinearly resized.
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
            prev_channels = 0 if b == 0 else block_out_channels[-b]
            vgg_channels = vgg_layer_channels[-(b+1)]
            out_channels = block_out_channels[-(b+1)]
            block_size = block_sizes[-(b+1)]
            initial = b == 0

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


if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    g = Generator()
    g.to(DEVICE)
    print(g)
    summary(g, (1, 256, 256))
