import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
# import cv2
import scipy.misc as misc

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_S = sorted(glob.glob(os.path.join(root, '%s/Train' % mode) + '/*.*'))
        self.files_T = sorted(glob.glob(os.path.join(root, '%s/Test' % mode) + '/*.*'))

    def __getitem__(self, index):


        item_S = self.transform(Image.open(self.files_S[index % len(self.files_S)]).convert("L"))

        temp_name = self.files_S[index % len(self.files_S)]
        Temp_NAME = temp_name.split('/')
        temp_name = Temp_NAME[-1]

        S_name = temp_name

        if self.unaligned:

            item_T = self.transform(Image.open(self.files_T[random.randint(0, len(self.files_T) - 1)]).convert("L"))
        else:

            item_T = self.transform(Image.open(self.files_T[index % len(self.files_T)]).convert("L"))

        return {'S': item_S, 'T': item_T, 'S_name': S_name}

    def __len__(self):
        return max(len(self.files_S), len(self.files_T))