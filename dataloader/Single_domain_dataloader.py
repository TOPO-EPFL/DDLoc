import os, random, time, copy, sys 
import numpy as np
import os.path as path
import PIL.Image

import blosc, struct

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import InterpolationMode

from torchvision import datasets, models, transforms

class Single_domain_dataloader(Dataset):
    def __init__(self, root_dir, set_name='test', img_normalize='pure'):
        # this dataloader is only used for test
        self.root_dir = root_dir 
        self.set_name = set_name
        self.img_normalize = img_normalize
        self.current_set_len = 0
        self.path2files = []
        
        curfilenamelist = os.listdir(path.join(self.root_dir, self.set_name, 'rgb'))
        self.path2files += [path.join(self.root_dir, self.set_name, 'rgb')+'/'+ curfilename for curfilename in curfilenamelist]
        self.current_set_len = len(self.path2files)   
        
        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        if self.img_normalize == 'pure':
            self.image_transform = transforms.Compose([transforms.ToTensor()])
        elif self.img_normalize == 'urban':
            self.image_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.4245, 0.4375, 0.3836],std=[0.1823, 0.1701, 0.1854])])
        elif self.img_normalize == 'nature':
            self.image_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.3636, 0.4331, 0.2956],std=[0.1383, 0.1457, 0.1147])])
        else:
            # feel free to add normalize for your own dataset
            raise NotImplementedError('normalization type [%s] is not found' % self.img_normalize)

        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        filename = self.path2files[idx]

        image = PIL.Image.open(filename).convert('RGB')
        image = self.image_transform(image)

        coord = torch.load(filename.replace('rgb','init').replace('png','dat'))
        gt_poses = np.loadtxt(filename.replace('rgb','poses').replace('png','txt')) #camera to world
        gt_poses = torch.from_numpy(gt_poses).float()
        focal_length = float(np.loadtxt(filename.replace('rgb','calibration').replace('png','txt')))

        return image, coord, gt_poses, focal_length









    