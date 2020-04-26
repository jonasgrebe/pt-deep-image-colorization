from typing import Tuple, Callable

import torch
import torchvision

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_single_image_lab(load_path):
    img = cv2.imread(load_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def save_single_image_lab(save_path, img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    cv2.imwrite(save_path, img)


class ImageLABDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: str, pil_transform: Callable = None, pt_transform: Callable = None) -> None:
        super(ImageLABDataset, self).__init__()

        self.data_dir = data_dir
        self.img_files = os.listdir(self.data_dir)
        self.pil_transform = pil_transform
        self.pt_transform = pt_transform


    def __len__(self) -> int:
        return len(self.img_files)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self):
            raise IndexError

        load_path = os.path.join(self.data_dir, self.img_files[idx])
        img = load_single_image_lab(load_path)

        img = torchvision.transforms.functional.to_pil_image(img)
        if self.pil_transform:
            img = self.pil_transform(img)

        img = torchvision.transforms.functional.to_tensor(img)
        if self.pt_transform:
            img = self.pt_transform(img)

        return img[0:1], img[1:]


if __name__ == '__main__':
    dataset = ImageLABDataset('data/square-custom-unsplash-10K',
                              pil_transform=torchvision.transforms.Compose([
                                #torchvision.transforms.RandomCrop(384),
                                torchvision.transforms.Resize(32),
                              ]),
                              pt_transform=torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=0)

    for step, batch in enumerate(dataloader):
        L = batch[0]
        AB = batch[1]

        LAB = torch.cat([L, AB], dim=1)

        grid = torchvision.utils.make_grid(LAB, nrow=200)
        grid = grid * 127.5 + 127.5
        save_single_image_lab('test.png', grid.numpy().transpose((1, 2, 0)))

        exit(1)
