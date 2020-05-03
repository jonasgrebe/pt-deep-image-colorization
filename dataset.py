from typing import Tuple, Callable

import torch
import torchvision

import os
import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == '__main__':

    dataset = ImageDatasetLAB('data/square-custom-unsplash-10K', )

    n_val = 100

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (len(dataset)-n_val, n_val))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    for step, sample in enumerate(dataloader):

        # EXAMPLE FOR VALIDATION ROUTINE
        # get image from batch
        img_batch = sample['img']
        file_batch = sample['img_file']

        # transform to (0, 1)
        img_batch = transform_input(img_batch)

        # extract L and AB channels
        L_batch, AB_batch = img_batch[:,:1], img_batch[:,1:]

        print(L_batch.size(), AB_batch.size())

        # fill the other one with zeros for each of them
        L_batch, AB_batch = np.concatenate([L_batch, np.zeros_like(AB_batch)], axis=1), np.concatenate([np.zeros_like(L_batch), AB_batch], axis=1)

        print(L_batch.shape, AB_batch.shape)

        for img, L, AB, file in zip(img_batch, L_batch, AB_batch, file_batch):

            # show all of them together with the image
            x = np.concatenate([img, L, AB], axis=2).transpose(1, 2, 0)
            x = transform_output(x)

            x = x.astype('uint8')
            x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR)
            cv2.imwrite(save_path, x)
