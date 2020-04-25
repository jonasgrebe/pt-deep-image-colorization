from typing import Tuple, List
import torch
import torchvision.models as models

# highly inspired by https://github.com/ChristophReich1996/Deep_Fovea_Architecture_for_Video_Super_Resolution/blob/master/vgg_19.py

class VGG19Features(torch.nn.Module):

    def __init__(self, vgg_layer_idxs: Tuple[int] = (3, 8, 17, 26, 35), frozen: bool = True) -> None:
        """ VGG19Features module: This module enables the utilization of (pretrained) VGG19 vgg_features.

        Parameters
        ----------
        vgg_layer_idxs : Tuple[int]
            Tuple of VGG feature layer indices. Defaults to last activation layer before MaxPooling in each of the five blocks.
        frozen : bool
            Flag that indicates whether or not the weights of the VGG19 network will be frozen.

        """
        super(VGG19Features, self).__init__()

        # store feature layer indices
        self.vgg_layer_idxs = vgg_layer_idxs

        # load pretrained VGG19 model (and discard the fully-connected classifier layers)
        self.vgg19 = models.vgg19(pretrained=True).features

        # freeze all weights if frozen is True
        if frozen:
            for param in self.vgg19.parameters():
                param.requires_grad = False


    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Short summary.

        Parameters
        ----------
        input : torch.Tensor
            Input image tensor with three channels.

        Returns
        -------
        List[torch.Tensor]
            List of extracted VGG19 feature tensors.

        """

        # feed the input through the entire model
        features = []
        for idx, layer in enumerate(self.vgg19):
            input = layer(input)

            # store the hidden feature layer output for all layers in the Tuple of indices.
            if idx in self.vgg_layer_idxs:
                features.append(input)
        return features
