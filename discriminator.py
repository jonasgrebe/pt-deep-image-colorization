from typing import Tuple
import torch

from torchsummary import summary


class DiscriminatorBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float=0.1) -> None:
        """Short summary.

        Parameters
        ----------
        in_channels : int
            Number of incoming channels.
        out_channels : int
            Number of outgoing channels.
        dropout_rate : float
            Probability for each of the outgoing channels to get dropped during training.

        """
        super(DiscriminatorBlock, self).__init__()

        # define layers of this block
        self.layers = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU()
        )

        # define residual mapping
        self.residual_mapping = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)),
        )

        # define pooling and dropout
        self.pooling = torch.nn.AvgPool2d(kernel_size=2)
        self.dropout = torch.nn.Dropout2d(p=dropout_rate)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Short summary.

        Parameters
        ----------
        input : torch.Tensor
            Description of parameter `input`.

        Returns
        -------
        torch.Tensor
            Description of returned object.

        """

        x = self.layers(input)
        x = x + self.residual_mapping(input)
        x = self.pooling(x)
        x = self.dropout(x)

        return x


class Discriminator(torch.nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int] = (3, 256, 256), block_channels: Tuple[Tuple[int, int]] = ((3, 32), (32, 64), (64, 128), (128, 256), (256, 512))) -> None:
        super(Discriminator, self).__init__()

        self.blocks = torch.nn.ModuleList()
        for in_channels, out_channels in block_channels:
            self.blocks.append(DiscriminatorBlock(in_channels=in_channels, out_channels=out_channels))

        last_block_size = input_shape[-1] // 2 ** len(block_channels)
        self.final_linear = torch.nn.Linear(in_features=last_block_size*last_block_size*block_channels[-1][-1], out_features=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x.flatten(1))

        return x


if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    d = Discriminator(input_shape=(3, 256, 256))
    d.to(DEVICE)
    print(d)
    summary(d, (3, 256, 256))
