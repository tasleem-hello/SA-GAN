#!/usr/bin/python3

import argparse
import itertools
import pdb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import os
from networks import Generator
from networks import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from data_utils import ImageDataset
from Model_VGG19 import gram_matrix
from vis import Visualizer
VIS = Visualizer("SA-GAN")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.adD_Srgument('--epoch', type=int, default=0, help='starting epoch')

parser.adD_Srgument('--n_epochs', type=int, default=12, help='number of epochs of training')
parser.adD_Srgument('--batchSize', type=int, default=1, help='size of the batches')
parser.adD_Srgument('--Time', type=str, default='Time_2', help='size of the batches')

parser.adD_Srgument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.adD_Srgument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.adD_Srgument('--decay_epoch', type=int, default=11, help='epoch to start linearly decaying the learning rate to 0')
parser.adD_Srgument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.adD_Srgument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.adD_Srgument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.adD_Srgument('--cuda', action='store_true',default=True, help='use GPU computation')
parser.adD_Srgument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.adD_Srgument('--transfer', type=bool, default=False, help='Restore parameters of this network')
parser.adD_Srgument('--GAN_weight', type=float, default=0.001, help='GAN weight')
parser.adD_Srgument('--threshold', type=int, default=1, help='Adjust by yourself')


opt = parser.parse_Srgs()
print(opt)

if torch.cuda.is_Svailable() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# Networks
netG_S2T = Generator(opt.input_nc, opt.output_nc)

netD_S = Discriminator(opt.input_nc)

netD_S_self = Discriminator(opt.input_nc)




####### Restore Parameters #########
if opt.transfer == True:
    netG_S2T.load_state_dict('output/netG_S2T.pth')

    netD_S.load_state_dict('output/netD_S.pth')



if opt.cuda:
    netG_S2T.cuda()
    netG_S2T = torch.nn.DataParallel(netG_S2T, device_ids=range(torch.cuda.device_count()))

    netD_S.cuda()
    netD_S = torch.nn.DataParallel(netD_S, device_ids=range(torch.cuda.device_count()))



    netD_S_self.cuda()
    netD_S_self = torch.nn.DataParallel(netD_S, device_ids=range(torch.cuda.device_count()))





netG_S2T.apply(weights_init_normal)

netD_S.apply(weights_init_normal)


netD_S_self.apply(weights_init_normal)



# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_S2T.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_S = torch.optim.Adam(netD_S.parameters(), lr=opt.lr, betas=(0.5, 0.999))

optimizer_D_S_self = torch.optim.Adam(netD_S_self.parameters(), lr=opt.lr, betas=(0.5, 0.999))




lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_S = torch.optim.lr_scheduler.LambdaLR(optimizer_D_S, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_S = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_T = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_S_Tuffer = ReplayBuffer()
fake_T_Tuffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),
                
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,drop_last=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))

# heat map
loader = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Resize((512, 512)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor




def image_loader(image):

    # fake batch dimension required to fit network's input dimensions
    image = loader(image)

    print(image.shape)

    return image

###################################
A_dis = 0.998
early_stop = 0

loss_Test = 10000
###################################
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):

    # if epoch>3:
    #     input("Press Enter to continue...")
    print(epoch)

    for i, batch in enumerate(dataloader):

        # Set model input
        real_S = Variable(input_S.copy_(batch['S']))
        real_T = Variable(input_T.copy_(batch['T']))

        ###### Generators S2T  ######
        optimizer_G.zero_grad()
        # GAN loss##############################################
        fake_T = netG_S2T(real_S)


        pred_fake, _ = netD_S(fake_T.detach())

        if pred_fake.size() != target_real.size():

            target_real = torch.unsqueeze(target_real, 1)

        loss_GAN_S2T = criterion_GAN(pred_fake, target_real)



        
   

        pred_fake, _ = netD_S_self(fake_T.detach())

        if pred_fake.size() != target_fake.size():
            target_fake = torch.unsqueeze(target_fake, 1)

        loss_D_fake_self_S = criterion_GAN(pred_fake, target_real)


        loss_D_S_self = loss_D_fake_self_S

       
        # Texturalloss
        _, feature_RS = netD_S(real_S)



        _, feature_RT = netD_S(real_T)
        _, feature_FT = netD_S(fake_T)



        _, content_feature_RS = netD_S_self(real_S)

        _, content_feature_RT = netD_S_self(real_T)
        _, content_feature_FT = netD_S_self(fake_T)

        Textural_loss = criterion_cycle(content_feature_FT[3], content_feature_RS[3].detach())
        


        A_dis = A_dis*0.999
        if epoch < 20:
            loss_G = (loss_GAN_S2T + loss_D_S_self) * opt.GAN_weight + Textural_loss

        else:
            loss_G = (loss_GAN_S2T + loss_D_S_self) * (opt.GAN_weight * A_dis) + Textural_loss



        loss_G.backward(retain_graph=True)
        
        optimizer_G.step()
        ###################################

        optimizer_G.zero_grad()


        ##################################################################################################
        #                       Discriminator
        ##################################################################################################


        ###### Discriminator A ######
        optimizer_D_S.zero_grad()

        # Real loss

        pred_real,_ = netD_S(real_T)
        loss_D_real = criterion_GAN(pred_real, target_real)

        #Fake loss

        pred_fake,_ = netD_S(fake_T.detach())

        # # print(pred_fake.size())
        if pred_fake.size() != target_fake.size():
            target_fake = torch.unsqueeze(target_fake, 1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)


        loss_D_S = loss_D_fake+loss_D_real
        loss_D_S.backward(retain_graph=True)

        optimizer_D_S.step()

        ###### Discriminator A_self ######
        optimizer_D_S_self.zero_grad()

        # # Real loss
        pred_real, _ = netD_S_self(real_S)
        loss_D_real_self_S = criterion_GAN(pred_real, target_real)


        
        pred_fake, _ = netD_S_self(fake_T.detach())

        loss_D_fake_self_S = criterion_GAN(pred_fake, target_fake)


        loss_D_S_self = loss_D_fake_self_S + loss_D_real_self_S

        loss_D_S_self.backward()

        optimizer_D_S_self.step()



        real_S = real_S * 0.5 + 0.5
        real_T = real_T * 0.5 + 0.5
        fake_T = fake_T * 0.5 + 0.5


        if i%10 == 0:
           
            VIS.img(name="real_S", img_=real_S)
            VIS.img(name="real_T", img_=real_T)
            VIS.img(name="fake", img_=fake_T)

    # Save  checkpoints and early stop

    if loss_GAN_S2T < loss_Test:
        loss_Test = loss_GAN_S2T
        early_stop = 0
    elif loss_GAN_S2T > loss_Test:
        early_stop += 1

    if early_stop >= opt.threshold:
        break

    torch.save(netG_S2T.state_dict(), 'output/netG_S2T_{}.pth'.format(opt.Time))

    torch.save(netD_S.state_dict(), 'output/netD_S_{}.pth'.format(opt.Time))

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_S.step()



###################################
