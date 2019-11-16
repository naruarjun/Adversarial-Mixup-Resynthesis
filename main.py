import torch
from torchvision import datasets, transforms
from models import *
import argparse
from model_handler import ModelHandler
import sys

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--lamb', type=float, default=0.01)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--update_g_freq', type=int, default=5)

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.test_batch_size, shuffle=True)
# TODO
args1 = {
        'width': 32,
        'latent_width': 4,
        'depth': 16, # nfg?
        'advdepth': 16, #nfd?
        'latent': 2, #4x4x2 = 32
    }
n_channels = 1
args = vars(args)
print(args)
scales = int(round(math.log(args1['width'] // args1['latent_width'], 2)))
generator = Autoencoder(scales,n_channels,args1['depth'],args1['latent'])
discriminator = Discriminator(scales,args1['depth'],args1['latent'],n_channels)
mixer = 'mixup' 
handler = ModelHandler(args, train_loader, test_loader, generator, discriminator, mixer)
handler.train()