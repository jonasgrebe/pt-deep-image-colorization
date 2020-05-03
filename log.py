from typing import List

import torch
import numpy as np
import os
import cv2

class Logger():

    def __init__(self, run_name) -> None:
        self.log_dir = 'logs'
        self.run_name = run_name
        self.mode = 'training'

        self.epoch = 0
        self.global_step = 0


    def set_mode(self, mode: str):
        self.mode = mode


    def new_epoch(self):
        self.epoch += 1


    def new_step(self):
        self.global_step += 1


    def log_losses(self, loss_dict) -> None:
        raise NotImplementedError


    def log_images(self, img_batch, name, dataformats='NCHW') -> None:
        raise NotImplementedError


class DirectoryLogger(Logger):

    def __init__(self, run_name) -> None:
        super(DirectoryLogger, self).__init__(run_name)


    def log_losses(self, loss_dict) -> None:
        pass


    def log_images(self, img_batch, name, dataformats='NCHW') -> None:
        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()

        if dataformats =='NCHW':
            img_batch = img_batch.transpose(0, 2, 3, 1)

        for img in img_batch:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        epoch_dir = os.path.join(self.log_dir, self.run_name, 'validation', 'images', str(self.epoch))
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)
        cv2.imwrite(os.path.join(epoch_dir, f'{name}.png'), img)



class TensorboardLogger(Logger):

    def __init__(self, run_name) -> None:
        super(TensorboardLogger, self).__init__(run_name)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.run_name))


    def log_losses(self, loss_dict) -> None:
        for loss_name in loss_dict:
            if torch.is_tensor(loss_dict[loss_name]):
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().numpy()

        self.writer.add_scalars(self.mode, loss_dict, self.global_step)


    def log_images(self, img_batch, name, dataformats='NCHW') -> None:

        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()

        self.writer.add_images(f'{self.mode}/{self.epoch}/{name}', img_batch, self.global_step, dataformats=dataformats)


class MergedLogger(Logger):

    def __init__(self, run_name, logger_types: = None) -> None:
        super(MergedLogger, self).__init__(run_name)
        self.loggers = [logger_type(run_name) for logger_type in logger_types]


    def log_losses(self, loss_dict) -> None:
        for logger in self.loggers:
            logger.log_losses(loss_dict)


    def log_images(self, img_batch, name, dataformats='NCHW') -> None:
        for logger in self.loggers:
            logger.log_images(img_batch, name, dataformats)
