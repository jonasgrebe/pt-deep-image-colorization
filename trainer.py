import torch
import numpy as np
import os

from logger import DirectoryLogger

class Trainer():

    def __init__(self, trainer_name,
                       generator, discriminator,
                       g_optimizer, d_optimizer,
                       pixel_loss, adversarial_loss,
                       transform_input, transform_output,
                       logger=None, hypers={}, device='cuda:0') -> None:

        self.epoch = 0

        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.pxl_loss = pixel_loss
        self.adv_loss = adversarial_loss

        if logger is None:
            self.logger = DirectoryLogger(trainer_name)
        else:
            self.logger = logger

        self.transform_input = transform_input
        self.transform_output = transform_output

        self.hypers = hypers
        self.device = device


    def fit(self, train_dataset=None, val_dataset=None, batch_size=1, epochs=10):
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.generator.train()
        self.discriminator.train()

        for self.epoch in range(self.epoch+epochs+1):
            self.logger.new_epoch(self.epoch)

            for step, batch in enumerate(dataloader):
                g_loss, d_loss, loss_dict, _ = self.forward(batch)

                g_loss.backward(retain_graph=True)
                d_loss.backward()

                self.g_optimizer.step()
                self.d_optimizer.step()

                self.logger.log_loss(loss_dict)
                status = f'[{step}/{len(dataloader)}] ' + ' | '.join([f'{loss_name}: {value}' for loss_name,value in loss_dict.items()])
                print(status)

            if val_dataset is not None:
                self.validate(val_dataset)

            self.save_checkpoint()


    def forward(self, batch):
        img_batch = self.transform_input(batch).to(self.device)

        L_batch, AB_batch = img_batch[:,:1], img_batch[:,1:]

        g_AB_batch = self.generator(L_batch)

        fake_batch = torch.cat([L_batch, g_AB_batch], dim=1)

        d_real_batch = self.discriminator(img_batch)
        d_fake_batch = self.discriminator(fake_batch)

        pxl_loss = self.pxl_loss(g_AB_batch, AB_batch)

        real_target = torch.ones_like(d_real_batch) * 0.9
        fake_target = torch.zeros_like(d_fake_batch)

        adv_g_loss = self.adv_loss(d_fake_batch, real_target)
        adv_d_loss = self.adv_loss(d_real_batch, real_target) + self.adv_loss(d_fake_batch, fake_target)

        pxl_loss *= self.hypers['pxl_loss_weight']
        adv_g_loss *=  self.hypers['adv_g_loss_weight']
        adv_d_loss *= self.hypers['adv_d_loss_weight']

        g_loss = pxl_loss + adv_g_loss
        d_loss = adv_d_loss

        loss_dict = {'pxl_loss': pxl_loss, 'adv_g_loss': adv_g_loss, 'adv_d_loss': adv_d_loss}
        batches = [img_batch, AB_batch, L_batch, g_AB_batch, fake_batch]

        return g_loss, d_loss, loss_dict, batches


    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = {
            'epoch': self.epoch,
            'generator': self.generator,
            'discriminator': self.discriminator,
            'g_optimizer': self.g_optimizer,
            'd_optimizer': self.d_optimizer
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'{self.epoch}.ckpt'))
        self.logger.epoch = checkpoint['epoch']


    def load_checkpoint(self, epoch):
        checkpoint_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{epoch}.ckpt'))
        self.epoch = checkpoint['epoch']
        self.generator = checkpoint['generator']
        self.discriminator = checkpoint['discriminator']
        self.g_optimizer = checkpoint['g_optimizer']
        self.d_optimizer = checkpoint['d_optimizer']


    def validate(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        self.generator.eval()
        self.discriminator.eval()

        for step, batch in enumerate(dataloader):
            g_loss, d_loss, loss_dict, batches = self.forward(batch)

            # logger.log_loss(loss_dict, validation=True)

            img_batch, AB_batch, L_batch, g_AB_batch, fake_batch = map(lambda x: x.detach().cpu().numpy(), batches)

            L_batch, AB_batch, g_AB_batch = np.concatenate([L_batch, np.zeros_like(AB_batch)], axis=1), np.concatenate([np.zeros_like(L_batch), AB_batch], axis=1), np.concatenate([np.zeros_like(L_batch), g_AB_batch], axis=1)

            for img, AB, L, g_AB, fake in zip(img_batch, AB_batch, L_batch, g_AB_batch, fake_batch):
                x = np.concatenate([img, AB, L, g_AB, fake], axis=2).transpose(1, 2, 0)
                x = self.transform_output(x)

                self.logger.log_image(f'{step}.png', x)


    def test(self, dataset):
        # test the trained generator
        self.generator.eval()
        self.discriminator.eval()
