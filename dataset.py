from typing import Tuple, Callable

import torch
import torchvision

import os
import numpy as np
import cv2


class ImageDatasetLAB(torch.utils.data.Dataset):

    def __init__(self, data_dir: str, transform: Callable = None) -> None:
        """ ImageDatasetL2AB: Dataset class to handle a flat image directory.

        Parameters
        ----------
        data_dir : str
            Path to the flat image directory. This directory should not contain any other subdirectory or unsupported file types.
        transform : Callable
            Callable that takes in a PIL image and returns the transformation result.

        """
        super(ImageDatasetLAB, self).__init__()

        self.data_dir = data_dir
        self.img_files = os.listdir(self.data_dir)
        self.transform = transform


    def __len__(self) -> int:
        return len(self.img_files)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self):
            raise IndexError

        # load the image at the given index
        load_path = os.path.join(self.data_dir, self.img_files[idx])
        img = cv2.imread(load_path)

        # translate the image to the LAB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # convert image to PIL image for pytorch transformations
        img = torchvision.transforms.functional.to_pil_image(img)

        # apply transformation if necessary
        if self.transform:
            img = self.transform(img)

        # convert image back to numpy and translate into LAB color space

        if torch.is_tensor(img):
            img = np.array(img)
            img = img.transpose(1, 2, 0)
        else:
            img = np.array(img)


        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # convert numpy.ndarray to torch.Tensor
        img = torchvision.transforms.functional.to_tensor(img)

        return img
