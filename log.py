from typing import List, Dict

import torch
import numpy as np
import os
import cv2

class Logger():

    def __init__(self, run_name: str) -> None:
        """ Logger: Base class for all training loggers. Such a logger handles all the logging of loss values, images etc.

        Parameters
        ----------
        run_name : str
            Name of the current experiment run.

        """
        self.log_dir = 'logs'
        self.run_name = run_name
        self.mode = 'training'

        self.epoch = 0
        self.global_step = 0


    def set_mode(self, mode: str) -> None:
        self.mode = mode


    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


    def new_epoch(self) -> None:
        self.epoch += 1


    def new_step(self) -> None:
        self.global_step += 1


    def log_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError


    def log_images(self, img_batch: torch.Tensor, name: str, dataformats: str = 'NCHW') -> None:
        raise NotImplementedError


class DirectoryLogger(Logger):

    def __init__(self, run_name) -> None:
        super(DirectoryLogger, self).__init__(run_name)


    def log_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        pass


    def log_images(self, img_batch: torch.Tensor, name: str, dataformats: str = 'NCHW') -> None:
        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()

        if dataformats =='NCHW':
            img_batch = img_batch.transpose(0, 2, 3, 1)

        for img in img_batch:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        epoch_dir = os.path.join(self.log_dir, self.run_name, self.mode, 'images', str(self.epoch))
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)
        cv2.imwrite(os.path.join(epoch_dir, f'{name}.png'), img)



class TensorboardLogger(Logger):

    def __init__(self, run_name) -> None:
        super(TensorboardLogger, self).__init__(run_name)

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.run_name))


    def log_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        for loss_name in loss_dict:
            if torch.is_tensor(loss_dict[loss_name]):
                loss_dict[loss_name] = loss_dict[loss_name].detach().cpu().numpy()

        self.writer.add_scalars(self.mode, loss_dict, self.global_step)


    def log_images(self, img_batch: torch.Tensor, name: str, dataformats: str = 'NCHW') -> None:

        if torch.is_tensor(img_batch):
            img_batch = img_batch.detach().cpu().numpy()

        self.writer.add_images(f'{self.mode}/{self.epoch}/{name}', img_batch, self.global_step, dataformats=dataformats)


class MergedLogger(Logger):

    def __init__(self, run_name, logger_types) -> None:
        super(MergedLogger, self).__init__(run_name)
        self.loggers = [logger_type(run_name) for logger_type in logger_types]


    def set_mode(self, mode: str):
        for logger in self.loggers:
            logger.set_mode(mode)


    def set_epoch(self, epoch: int) -> None:
        for logger in self.loggers:
            logger.set_epoch(epoch)


    def new_epoch(self):
        for logger in self.loggers:
            logger.new_epoch()


    def new_step(self):
        for logger in self.loggers:
            logger.new_step()


    def log_losses(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        for logger in self.loggers:
            logger.log_losses(loss_dict)


    def log_images(self, img_batch: torch.Tensor, name: str, dataformats: str = 'NCHW') -> None:
        for logger in self.loggers:
            logger.log_images(img_batch, name, dataformats)
