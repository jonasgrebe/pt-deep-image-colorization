import os
import numpy as np
import torch

import cv2

class Logger():

    def __init__(self, trainer_name):
        self.log_dir = os.path.join('logs', trainer_name)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)


    def new_epoch(self, epoch):
        raise NotImplementedError


    def log_loss(self, loss_dict, validation=False):
        raise NotImplementedError


    def log_image(self, filename, img, color_space='lab'):
        raise NotImplementedError


class DirectoryLogger(Logger):

    def __init__(self, trainer_name) -> None:

        super(DirectoryLogger, self).__init__(trainer_name)

        self.epoch = None
        self.step = None
        self.step_losses = {}


    def new_epoch(self, epoch) -> None:

        for loss_name in self.step_losses:
            array = np.array(self.step_losses[loss_name])

            loss_dir = os.path.join(self.log_dir, 'training', 'loss', loss_name)

            if not os.path.isdir(loss_dir):
                os.makedirs(loss_dir)

            np.save(os.path.join(loss_dir, str(self.epoch)), array)

        self.epoch = epoch
        self.step = 0
        self.step_losses = {}


    def log_loss(self, loss_dict, validation=False):

        for loss_name, value in loss_dict.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()

            if loss_name not in self.step_losses:
                self.step_losses[loss_name] = []

            self.step_losses[loss_name].append(value)


    def log_image(self, filename, img, color_space='lab'):
        epoch_dir = os.path.join(self.log_dir, 'images', str(self.epoch))
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)

        if color_space == 'lab':
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(epoch_dir, filename), img)
