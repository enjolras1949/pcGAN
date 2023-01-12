#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:37:43 2023

@author: lg809
"""

import argparse
from torch.utils.data import DataLoader
from models import *
from datasets import *
from utils import *
import torch
print(torch.__version__)

parser = argparse.ArgumentParser(description="3ppGAN")

parser.add_argument("-network_name", type=str, default="3ppGAN", help="name of the network")
parser.add_argument("--training_dataset", type=str, default="ex-vivo", help="name of the dataset")
parser.add_argument("--testing_dataset", type=str, default="ex-vivo", help="name of the testing dataset")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs oef training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=25, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=200, help="size of image height")
parser.add_argument("--img_width", type=int, default=200, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model checkpoints")


parser.add_argument("--lambda_merging", type=float, default=10, help="scaling factor for the new loss")
parser.add_argument("--textfile_training_results_interval", type=int, default=50, help="textfile_training_results_interval")

opt = parser.parse_args()
print(opt)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
input_shape = (opt.channels, opt.img_height, opt.img_width)
# Initialize generator and discriminator
G_AB = GeneratorUNet()

if cuda:
    G_AB = G_AB.cuda()

G_AB.load_state_dict(torch.load("saved_models/%s-%s-G_AB-%dep.pth" % (opt.network_name, opt.testing_dataset, opt.epoch)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_AB_buffer = ReplayBuffer()


transforms_testing_non_fliped_ = [
    transforms.ToTensor(),
]

# Test data loader - non flipped
val_dataloader_non_flipped = DataLoader(
    ImageDataset("../data/Testing/%s-testing" % opt.testing_dataset, transforms_=transforms_testing_non_fliped_, unaligned=False),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

def testing():
    os.makedirs("images/%s-Est-Depths" % (opt.network_name), exist_ok=True)
    G_AB.eval()

    for i, batch in enumerate(val_dataloader_non_flipped):
        real_A = Variable(batch["A"].type(Tensor))
        fake_AB = G_AB(real_A)
        save_image(fake_AB, "images/%s-Est-Depths/%s-Est-Depths-%s.png" % (opt.network_name,opt.network_name, i),normalize=False, scale_each=False) #range= (0,128)
testing()
