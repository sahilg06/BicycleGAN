import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="CIFAR10", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=3200, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=25, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Loss functions
mae_loss = torch.nn.L1Loss()

# Initialize generator, encoder and discriminators
generator = Generator(opt.latent_dim, input_shape)
encoder = Encoder(opt.latent_dim, input_shape)
D_VAE = MultiDiscriminator(input_shape)
D_LR = MultiDiscriminator(input_shape)

if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    mae_loss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_LR.load_state_dict(torch.load("saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
    D_LR.apply(weights_init_normal)

# Optimizers
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor



#adding Gaussian NOISE 
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Dataset
train_dataset = datasets.CIFAR10(root="../data/%s" % opt.dataset_name, 
                                           train=True, 
                                           transform=transforms.Compose([
                                                    transforms.Resize((opt.img_height,opt.img_width), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                           ]),
                                           download=True)
  


test_dataset = datasets.CIFAR10(root="../data/%s" % opt.dataset_name, 
                                            train=False, 
                                            transform=transforms.Compose([
                                                      transforms.Resize((opt.img_height,opt.img_width), Image.BICUBIC),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                            ]),
                                            download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = opt.batch_size,
                                           num_workers = 8,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size = opt.batch_size,
                                          num_workers = 1,
                                          shuffle=False)



#noised-dataset
train_dataset_noised = datasets.CIFAR10(root="../data/%s" % opt.dataset_name, 
                                           train=True, 
                                           transform=transforms.Compose([
                                                    transforms.Resize((opt.img_height,opt.img_width), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                                    AddGaussianNoise(0.,2.)
                                           ]),
                                           download=True)
  


test_dataset_noised = datasets.CIFAR10(root="../data/%s" % opt.dataset_name, 
                                            train=False, 
                                            transform=transforms.Compose([
                                                      transforms.Resize((opt.img_height,opt.img_width), Image.BICUBIC),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                                      AddGaussianNoise(0.,2.)
                                            ]),
                                            download=True)
# Noised-Data loader
train_loader_noised = torch.utils.data.DataLoader(dataset=train_dataset_noised,
                                           batch_size = opt.batch_size,
                                           num_workers = 8,
                                           shuffle=True)

test_loader_noised = torch.utils.data.DataLoader(dataset=test_dataset_noised,
                                          batch_size = opt.batch_size,
                                          num_workers = 1,
                                          shuffle=False)



def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    generator.eval()
    img_noised = next(iter(test_loader_noised))
    img=next(iter(test_loader))
    img_samples = None
    for i in range(len(img_noised[0])):

        # Repeat input image by number of desired columns
        real_A = img_noised[0][i].view(1, *img_noised[0][i].shape).repeat(opt.latent_dim, 1, 1, 1)
        real_A = Variable(real_A.type(Tensor))
        
        # Sample latent representations
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (opt.latent_dim,opt.latent_dim))))

        # Generate samples
        fake_B = generator(real_A, sampled_z)

        # Concatenate samples horizontally
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        img_sample = torch.cat((img[0][i], fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)
    generator.train()


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z


# ----------
#  Training
# ----------

# Adversarial loss
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.epoch,opt.n_epochs):
    i=0
    for batch_real, batch_noised in zip(train_loader, train_loader_noised):

        # Set model input
        real_A = Variable(batch_noised[0].type(Tensor))
        real_B = Variable(batch_real[0].type(Tensor))

        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # ----------
        # cVAE-GAN
        # ----------

        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)

        # Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        # ---------
        # cLR-GAN
        # ---------

        # Produce output using sampled z (cLR-GAN)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
        _fake_B = generator(real_A, sampled_z)
        # cLR Loss: Adversarial loss
        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------

        loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl

        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        # ---------------------
        # Generator Only Loss
        # ---------------------

        # Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = opt.lambda_latent * mae_loss(_mu, sampled_z)

        loss_latent.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------

        optimizer_D_VAE.zero_grad()

        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)

        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------

        optimizer_D_LR.zero_grad()

        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)

        loss_D_LR.backward()
        optimizer_D_LR.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i ; i=i+1
        batches_left = opt.n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader),
                loss_D_VAE.item(),
                loss_D_LR.item(),
                loss_GE.item(),
                loss_pixel.item(),
                loss_kl.item(),
                loss_latent.item(),
                time_left,
            )
        )

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_LR.state_dict(), "saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, epoch))
