from typing import Tuple, List, Callable, Dict, Any
import torch
import numpy as np
import os

import cv2
import log

class Trainer():

    def __init__(self, logger: log.Logger,
                       generator: torch.nn.Module, discriminator: torch.nn.Module,
                       g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer,
                       pixel_loss: torch.nn.Module, adversarial_loss: torch.nn.Module,
                       transform_input: Callable = lambda x: x, transform_output: Callable = lambda x: x,
                       hypers: Dict[str, Any] ={}, device: torch.device ='cuda:0') -> None:
        """ Trainer: Specific helper class for training a generative adversarial network for the task of
                     image colorization.

        Parameters
        ----------
        logger : logger.Logger
            Logger instance that handles the logging of the given hyperparameters, the losses, the validation results, etc.
        generator : torch.nn.Module
            Generator module that expects a 1-channel image tensor and returns a 3-channel image tensor. Note that this module
            should already be transferred to the given device.
        discriminator : torch.nn.Module
            Discriminator module that expects a 3-channel image tensor and returns a single scalar tensor. Note that this module
            should already be transferred to the given device.
        g_optimizer : torch.optim.Optimizer
            Optimizer over the learnable parameters of the generator.
        d_optimizer : torch.optim.Optimizer
            Optimizer over the learnable parameters of the discriminator.
        pixel_loss : torch.nn.Module
            Per-pixel loss module that compares the input and target images component-wise.
        adversarial_loss : torch.nn.Module
            Adversarial loss module that judges the discriminator's output.
        transform_input : Callable
            Input transformation that is applied to each of the images before feeding it through the network.
        transform_output : Callable
            Output transformation that reverses the input transformation.
        hypers : Dict[str, Any]
            Dictionary of hyperparameters.
        device : torch.device
            PyTorch device of the generator and discriminator.

        """
        # initialize epoch
        self.epoch = 0

        # set generator and discriminator
        self.generator = generator
        self.discriminator = discriminator

        # set optimizers
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        # set pixel and adversarial loss
        self.pxl_loss = pixel_loss
        self.adv_loss = adversarial_loss

        # set logger
        self.logger = logger

        # set input and output transformations
        self.transform_input = transform_input
        self.transform_output = transform_output

        # set hyperparameter dictionary and device
        self.hypers = hypers
        self.device = device


    def training_step(self, img_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Trains the generator and the discriminator on the given image batch.

        Parameters
        ----------
        img_batch : torch.Tensor
            Input image tensor.

        Returns
        -------
        Dict[str, torch.Tensor]
            .Tuple with a dictionary that holds all the loss_name's together with their losss values

        """
        # reset the gradients of the discriminator
        self.discriminator.zero_grad()

        # let discriminator judge the real images
        d_real_batch = self.discriminator(img_batch)

        # define adversarial loss targets
        real_target = torch.ones_like(d_real_batch)
        fake_target = torch.zeros_like(d_real_batch)

        # compute loss and backward it
        adv_d_real_loss = self.adv_loss(d_real_batch, real_target * 0.9) * self.hypers['adv_d_loss_weight']
        adv_d_real_loss.backward()

        # split images into L and AB channels
        L_batch, AB_batch = img_batch[:,:1], img_batch[:,1:]
        # generate colors based on L
        g_AB_batch = self.generator(L_batch)

        # construct fake images by concatenation
        fake_batch = torch.cat([L_batch, g_AB_batch], dim=1)
        # let discriminator judge the fake images (without a gradient flow through the Generator)
        d_fake_batch = self.discriminator(fake_batch.detach())

        # compute loss and backward it
        adv_d_fake_loss = self.adv_loss(d_fake_batch, fake_target) * self.hypers['adv_d_loss_weight']
        adv_d_fake_loss.backward()

        # add adversarial losses for logging
        adv_d_loss = adv_d_real_loss + adv_d_fake_loss

        # optimize the discriminator's parameters based on the computed gradients
        self.d_optimizer.step()

        # reset the gradients of the generator
        self.generator.zero_grad()

        # let the discriminator judge the fake images
        d_batch = self.discriminator(fake_batch)

        # compute loss and backward it (but keeping the forward information inside the generator)
        adv_g_loss = self.adv_loss(d_batch, real_target) * self.hypers['adv_g_loss_weight']
        adv_g_loss.backward(retain_graph=True)

        # compute per-pixel loss and backward it
        pxl_loss = self.pxl_loss(AB_batch, g_AB_batch) * self.hypers['pxl_loss_weight']
        pxl_loss.backward()

        # optimize the generator's parameters based on the computed gradients
        self.g_optimizer.step()

        # put all of the losses in a dictionary
        loss_dict = {'pxl_loss': pxl_loss, 'adv_g_loss': adv_g_loss, 'adv_d_loss': adv_d_loss}

        return loss_dict


    def forward(self, img_batch: torch.Tensor) -> List[torch.Tensor]:
        """ Forwards a given batch of images through the network and returns a list of relevant output batches for visualization .

        Parameters
        ----------
        img_batch : torch.Tensor

        Returns
        -------
        List[torch.Tensor]
            List of relevant output batches for visualization

        """

        # feed L channel through the generator and create the fake images afterwards
        L_batch, AB_batch = img_batch[:,:1], img_batch[:,1:]
        g_AB_batch = self.generator(L_batch)

        # build the fake images
        fake_batch = torch.cat([L_batch, g_AB_batch], dim=1)

        # ask the discriminator for its opinion
        d_real_batch = self.discriminator(img_batch)
        d_fake_batch_g = self.discriminator(fake_batch)

        # put all of the relevant batches in a list and return it
        batches = [img_batch, AB_batch, L_batch, g_AB_batch, fake_batch]

        return batches


    def fit(self, train_dataset: torch.utils.data.Dataset,
                  val_dataset: torch.utils.data.Dataset = None,
                  batch_size: int = 1, epochs: int = 10) -> None:
        """ Fits the models of this trainer to the given training dataset. If no validation dataset is provided,
            no validation will be performed.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Dataset with training data.
        val_dataset : torch.utils.data.Dataset
            Dataset with validation data.
        batch_size : int
            Number of data samples in a batch during training.
        epochs : int
            Number of epochs for training.

        """
        # create dataloader for the training data
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # initial validation
        if self.epoch == 0 and val_dataset is not None:
            self.validate(val_dataset)
            self.save_checkpoint()

        # for each of the epochs:
        for self.epoch in range(self.epoch+1, self.epoch+epochs+1):
            self.logger.set_mode('training')
            self.logger.new_epoch()

            # for each of the batches in this epoch:
            for step, batch in enumerate(dataloader):
                self.logger.new_step()

                # set mode of both networks to training
                self.generator.train()
                self.discriminator.train()

                img_batch = self.transform_input(batch).to(self.device)
                loss_dict = self.training_step(img_batch)

                # log and print losses
                self.logger.log_losses(loss_dict)
                status = f'[{self.epoch}: {step}/{len(dataloader)}] ' + ' | '.join([f'{loss_name}: {value:.6f}' for loss_name,value in loss_dict.items()])
                print(status)

            # if validation data is given
            if val_dataset is not None:
                # validate the adversarial network
                self.validate(val_dataset)

            # save models and optimizers in checkpoint
            self.save_checkpoint()


    def validate(self, dataset: torch.utils.data.Dataset) -> None:
        """ Validates the network on a given dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset with validation data.

        """
        # create dataloader for the validation data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # set mode of both networks to evaluation
        self.generator.eval()
        self.discriminator.eval()
        self.logger.set_mode('validation')

        # for each validation sample:
        for step, batch in enumerate(dataloader):
            # transform image values to the range (-1, 1)
            img_batch = self.transform_input(batch).to(self.device)

            # forward input batch through the adversarial network
            batches = self.forward(img_batch)
            conc_images = self.visualize_prediction(batches)

            self.logger.log_images(np.array(conc_images), step, dataformats='NHWC')


    def visualize_prediction(self, batches) -> List[np.ndarray]:
        # detach all batches
        img_batch, AB_batch, L_batch, g_AB_batch, fake_batch = map(lambda x: x.detach(), batches)

        # fill missing channels
        L_batch, AB_batch, g_AB_batch = torch.cat([L_batch, torch.zeros_like(AB_batch)], dim=1), torch.cat([torch.zeros_like(L_batch), AB_batch], dim=1), torch.cat([torch.zeros_like(L_batch), g_AB_batch], dim=1)

        # concatenate all batches, move the result to the cpu and transform it to numpy
        conc_batch = torch.cat([img_batch, AB_batch, L_batch, g_AB_batch, fake_batch], dim=3)
        conc_batch = conc_batch.cpu().numpy()

        # convert all images to RGB
        conc_batch = self.transform_output(conc_batch).astype('uint8')
        conc_images = [cv2.cvtColor(conc.transpose(1, 2, 0), cv2.COLOR_LAB2RGB) for conc in conc_batch]

        return conc_images


    def test(self, dataset: torch.utils.data.Dataset) -> None:
        """ Tests the network on a given testing dataset.

        Parameters
        ----------
        dataset : type
            Dataset with testing data.

        """
        # create dataloader for the testing data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # set mode of both networks to evaluation
        self.generator.eval()
        self.discriminator.eval()
        self.logger.set_mode('testing')

        # for each validation sample:
        for step, batch in enumerate(dataloader):
            batches = self.forward(batch)
            img_batch, AB_batch, L_batch, g_AB_batch, fake_batch = map(lambda x: x.detach(), batches)
            g_AB_batch = torch.cat([torch.zeros_like(L_batch), g_AB_batch], dim=1)
            conc_batch = torch.cat([img_batch, g_AB_batch, fake_batch], dim=3)

            conc_batch = conc_batch.cpu().numpy()
            conc_batch = self.transform_output(conc_batch).astype('uint8')
            conc_images = [cv2.cvtColor(conc.transpose(1, 2, 0), cv2.COLOR_LAB2RGB) for conc in conc_batch]

            self.logger.log_images(np.array(conc_images), step, dataformats='NHWC')


    def save_checkpoint(self) -> None:
        """ Saves the parameters and states of the models and optimizers in a single checkpoint file.

        """
        # create the checkpoint directory if it does not exist
        checkpoint_dir = os.path.join(self.logger.log_dir, self.logger.run_name, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # create the checkpoint
        checkpoint = {
            'epoch': self.epoch,
            'generator': self.generator,
            'discriminator': self.discriminator,
            'g_optimizer': self.g_optimizer,
            'd_optimizer': self.d_optimizer
        }
        # save the checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'{self.epoch}.ckpt'))


    def load_checkpoint(self, epoch: int) -> None:
        """ Loads the checkpoint of a given epoch and restores the parameters and states
            of all models and optimizers.

        Parameters
        ----------
        epoch : int
            Number of epoch to restore.

        """
        # load the checkpoint of the given epoch
        checkpoint_dir = os.path.join(self.logger.log_dir, self.logger.run_name, 'checkpoints')
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{epoch}.ckpt'))

        # restore the information from the checkpoint
        self.epoch = checkpoint['epoch']
        self.generator = checkpoint['generator']
        self.discriminator = checkpoint['discriminator']
        self.g_optimizer = checkpoint['g_optimizer']
        self.d_optimizer = checkpoint['d_optimizer']

        # inform the logger about the restored epoch
        self.logger.set_epoch(checkpoint['epoch'])
