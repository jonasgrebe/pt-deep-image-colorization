
import argparse

parser = argparse.ArgumentParser(description='Train deep image colorization network')

parser.add_argument('--data', default='data/')
parser.add_argument('--input_shape', default=(256, 256, 3))
parser.add_argument('--include_vgg', default=True)
parser.add_argument('--g_lr', default=1e-5)
parser.add_argument('--d_lr', default=1e-5)
parser.add_argument('--batch_size', default=1)
parser.add_argument('--epochs', default=50)
parser.add_argument('--n_val', default=64)

parser.add_argument('--aug_flip_horizontal', default=True)
parser.add_argument('--aug_jitter', default=0)
parser.add_argument('--aug_rotation', default=0)


args = parser.parse_args()

print(args)
