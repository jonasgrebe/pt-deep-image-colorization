import argparse

parser = argparse.ArgumentParser(description='Train deep image colorization network')

parser.add_argument('--data', default='data/')
parser.add_argument('--input_shape', default=(3, 256, 256))
parser.add_argument('--g_lr', default=1e-5)
parser.add_argument('--d_lr', default=1e-5)
parser.add_argument('--batch_size', default=1)
parser.add_argument('--epochs', default=50)
parser.add_argument('--n_val', default=64)
parser.add_argument('--cuda', default=True)

parser.add_argument('--aug_flip_horizontal', default=True)
parser.add_argument('--aug_jitter', default=0)
parser.add_argument('--aug_rotation', default=0)

args = parser.parse_args()


import torch
import torchvision
from dataset import ImageDatasetLAB
from trainer import Trainer
from generator import Generator
from discriminator import Discriminator

dataset = ImageDatasetLAB('data/square-custom-unsplash-10K', transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(args.input_shape[1:], scale=(0.7, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5)
]))

transform_input = torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2)
transform_output = torchvision.transforms.Lambda(lambda x: (x + 1) * 127.5)

train_dataset, val_dataset = torch.utils.data.random_split(dataset, (len(dataset)-args.n_val, args.n_val))
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, (10, len(dataset)-10))

g = Generator()
d = Discriminator(input_shape=args.input_shape)

if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda:0")
elif not args.cuda:
    device = torch.device("cpu")
else:
    raise ValueError("No cuda available!")

if args.cuda:
    g = g.to(device)
    d = d.to(device)

g_optimizer = torch.optim.Adam(params=g.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(params=d.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

pxl_loss = torch.nn.MSELoss()
adv_loss = torch.nn.BCEWithLogitsLoss()

hypers = {
    'pxl_loss_weight': 1.0,
    'adv_g_loss_weight': 0.1,
    'adv_d_loss_weight': 1.0
}

trainer = Trainer(trainer_name='exp_1',
                  generator=g, discriminator=d,
                  g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                  pixel_loss=pxl_loss, adversarial_loss=adv_loss,
                  transform_input=transform_input, transform_output=transform_output,
                  logger=None, hypers=hypers, device=device)

trainer.fit(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=1, epochs=10)
