from typing import Dict, Any
import os
import numpy as np
import torch

import cv2

class Logger():

    def __init__(self, name: str) -> None:
        self.log_dir = os.path.join('logs', name)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        self.epoch = 0
        self.global_step = 0

    def new_epoch(self, epoch: int) -> None:
        raise NotImplementedError


    def new_step(self) -> None:
        self.global_step += 1


    def log_loss(self, loss_dict: Dict[str, torch.Tensor], validation=False) -> None:
        raise NotImplementedError


    def log_image(self, filename: str, img: torch.Tensor, validation=False) -> None:
        raise NotImplementedError


    def log_hypers(self, hypers_dict: Dict[str, Any]) -> None:
        print(hypers_dict)


    def log_graph(self, model, model_input):
        raise NotImplementedError


class DirectoryLogger(Logger):

    def __init__(self, name: str) -> None:

        super(DirectoryLogger, self).__init__(name)
        self.step_losses = {}


    def new_epoch(self, epoch: int) -> None:

        for loss_name in self.step_losses:
            array = np.array(self.step_losses[loss_name])

            loss_dir = os.path.join(self.log_dir, 'training', 'loss', loss_name)

            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)

            np.save(os.path.join(loss_dir, str(self.epoch)), array)

        self.epoch = epoch
        self.step_losses = {}


    def log_loss(self, loss_dict: Dict[str, torch.Tensor], validation=False) -> None:
        for loss_name, value in loss_dict.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()

            if loss_name not in self.step_losses:
                self.step_losses[loss_name] = []

            self.step_losses[loss_name].append(value)


    def log_image(self, filename: str, img = None, format: str ='bgr', validation=False) -> None:
        if torch.is_tensor(img):
            img = img.detach().numpy().transpose(1, 2, 0)

        if format.lower() == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        epoch_dir = os.path.join(self.log_dir, 'images', str(self.epoch))
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)
        cv2.imwrite(os.path.join(epoch_dir, filename), img)


    def log_graph(self, model, model_input):
            raise NotImplementedError


class TensorboardLogger(Logger):

    def __init__(self, name: str) -> None:
        super(TensorboardLogger, self).__init__(name)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))


    def new_epoch(self, epoch: int) -> None:
        self.epoch = epoch


    def log_loss(self, loss_dict: Dict[str, torch.Tensor], validation=False) -> None:
        for loss_name in loss_dict:
            if torch.is_tensor(loss_dict[loss_name]):
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().numpy()

        if not validation:
            self.writer.add_scalars('train', loss_dict, self.global_step)
        else:
            self.writer.add_scalars('validation', loss_dict, self.epoch)


    def log_image(self, filename: str, img = None, format: str ='bgr', validation=False) -> None:

        if torch.is_tensor(img):
            img = img.detach().numpy().transpose(1, 2, 0)

        if format.lower() == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not validation:
            self.writer.add_image(f'training/{self.epoch}/{filename}', img, self.global_step, dataformats='HWC')
        else:
            self.writer.add_image(f'validation/{self.epoch}/{filename}', img, self.epoch, dataformats='HWC')


    def log_graph(self, model, model_input):
        self.writer.add_graph(model, model_input)
