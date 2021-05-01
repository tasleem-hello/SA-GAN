#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
import torch.nn.modules
import numpy as np
from models import Generator
from datasets_utils import ImageDataset
import Visualizer
viz = Visualizer.Visualizer("test")

parser = argparse.ArgumentParser()
parser.adD_Srgument('--batchSize', type=int, default=1, help='size of the batches')
parser.adD_Srgument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.adD_Srgument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.adD_Srgument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.adD_Srgument('--size', type=int, default=512, help='size of the data (squared assumed)')
parser.adD_Srgument('--cuda', action='store_true', help='use GPU computation')
parser.adD_Srgument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.adD_Srgument('--generator_S2T', type=str, default='output/netG_S2T.pth', help='S2T generator checkpoint file')
opt = parser.parse_Srgs()
print(opt)

os.makedirs('output/source', exist_ok=True)
os.makedirs('output/target', exist_ok=True)
os.makedirs('output/S2T', exist_ok=True)


def tensor2image(tensor):
    image = 0.5 * (tensor + 1.0)
  
    return image

if torch.cuda.is_Svailable() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
netG_S2T = Generator(opt.input_nc, opt.output_nc)
netG_T2S = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_S2T.cuda()
    netG_T2S.cuda()


temp_S2T = torch.load('output/netG_S2T.pth')



new_temp_S2T = OrderedDict()
for k, v in temp_S2T.items():
    name = k[7:]  # remove `module.`
    new_temp_S2T[name] = v

s


netG_S2T.load_state_dict(new_temp_S2T)

netG_S2T.eval()


Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_ = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_S = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)


transforms_ = [transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),
              
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

###################################

###### Testing######




for i, batch in enumerate(dataloader):
    

    real_S = Variable(input_S.copy_(batch['S']))
    real_T = Variable(input_T.copy_(batch['T']))
    S_name = batch['S_name']
   

    fake_T = netG_S2T(real_S).data



    # #
    viz.img("S",tensor2image(real_S))
    viz.img("T",tensor2image(real_T))
    viz.img("S2T",tensor2image(fake_T))
  

    save_image(tensor2image(real_S), 'output/source/%s' % (S_name[0]))
    save_image(tensor2image(real_T), 'output/target/%04d.png' % (i + 1))



    save_image(tensor2image(fake_T), 'output/S2T/{}'.format(S_name[0]) )

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
