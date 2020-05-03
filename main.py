import argparse
from typing import Tuple

# create argument parser
parser = argparse.ArgumentParser(description='Train deep image colorization network')

# add specific arguments for training and testing
parser.add_argument('--name', type=str, default='exp_layernorm')
parser.add_argument('--data', type=str, default='data/square-custom-unsplash-10K')
parser.add_argument('--input_shape', type=Tuple[int, int, int], default=(3, 256, 256))
parser.add_argument('--g_lr', type=float, default=1e-5)
parser.add_argument('--d_lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--n_val', type=int, default=32)

# parse arguments
args = parser.parse_args()

# import relevant modules
import torch
import torchvision
from dataset import ImageDatasetLAB
from trainer import Trainer
from generator import Generator
from discriminator import Discriminator
from log import DirectoryLogger, TensorboardLogger, MergedLogger


# specify the training and validation data
dataset = ImageDatasetLAB(args.data, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(300),
    torchvision.transforms.RandomCrop(args.input_shape[1:]),
    torchvision.transforms.RandomHorizontalFlip(p=0.5)
]))

train_dataset, val_dataset = torch.utils.data.random_split(dataset, (len(dataset)-args.n_val, args.n_val))

transform_input = torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2)
transform_output = torchvision.transforms.Lambda(lambda x: (x + 1) * 127.5)

# instantiate the generator and the discriminator
g = Generator()
d = Discriminator(input_shape=args.input_shape)

# get the available device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# transfer models to the device
g = g.to(device)
d = d.to(device)

# define the optimization
g_optimizer = torch.optim.Adam(params=g.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(params=d.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

# specify the loss functions
pxl_loss = torch.nn.MSELoss()
adv_loss = torch.nn.BCEWithLogitsLoss()

# specify some fixed hyperparameters
hypers = {
    'pxl_loss_weight': 1.0,
    'adv_g_loss_weight': 0.1,
    'adv_d_loss_weight': 0.1,
}
hypers.update(vars(args))

# instantiate the logger
logger = MergedLogger(args.name, [DirectoryLogger, TensorboardLogger])

# build the trainer helper class
trainer = Trainer(logger=logger,
                  generator=g, discriminator=d,
                  g_optimizer=g_optimizer, d_optimizer=d_optimizer,
                  pixel_loss=pxl_loss, adversarial_loss=adv_loss,
                  transform_input=transform_input, transform_output=transform_output,
                  hypers=hypers, device=device)

# fit the adversarial network to the data
trainer.fit(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=args.batch_size, epochs=args.epochs)
