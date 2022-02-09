import os, sys, random, time, copy
import numpy as np
import math
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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.bin', '.dat'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Joint_coord_dataLoader(Dataset):
    def __init__(self, real_root_dir, syn_root_dir, img_normalize='pure', paired_data=False, augment=False):
        self.real_root_dir = real_root_dir
        self.syn_root_dir = syn_root_dir
        self.img_normalize = img_normalize
        self.current_set_len = 0
        self.real_path2files = []
        self.syn_path2files = []
        self.paired_data = paired_data # whether 1 to 1 matching
        self.augment = augment # whether to augment each batch data

        self.aug_scale_min = 2/3.0 # defalut range of scaling
        self.aug_scale_max = 3/2.0
        self.aug_rotation = 30.0 # defalut range of rotation
        self.SUBSAMPLE = 8.0 # defalut downsampling factor

        self.set_name = 'train' # Joint_coord_dataLoader is only used in training phase
        
        real_curfilenamelist = os.listdir(os.path.join(self.real_root_dir, self.set_name, 'rgb'))
        for fname in sorted(real_curfilenamelist):
            if is_image_file(fname):
                path = os.path.join(self.real_root_dir, self.set_name, 'rgb', fname)
                self.real_path2files.append(path)

        self.real_set_len = len(self.real_path2files)

        syn_curfilenamelist = os.listdir(os.path.join(self.syn_root_dir, self.set_name, 'rgb'))
        for fname in sorted(syn_curfilenamelist):
            if is_image_file(fname):
                path = os.path.join(self.syn_root_dir, self.set_name, 'rgb', fname)
                self.syn_path2files.append(path)

        self.syn_set_len = len(self.syn_path2files)

        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()
        if self.augment:
            image_transform = [transforms.ColorJitter(brightness=True, contrast=True)]
        else:
            image_transform = []
        if self.img_normalize == 'pure':
            image_transform.append(transforms.ToTensor())
        elif self.img_normalize == 'urban':
            image_transform.append(transforms.ToTensor())
            image_transform.append(transforms.Normalize(mean=[0.4245, 0.4375, 0.3836],std=[0.1823, 0.1701, 0.1854]))
        elif self.img_normalize == 'nature':
            image_transform.append(transforms.ToTensor())
            image_transform.append(transforms.Normalize(mean=[0.3636, 0.4331, 0.2956],std=[0.1383, 0.1457, 0.1147]))
        else:
            # feel free to add normalize for your own dataset
            raise NotImplementedError('normalization type [%s] is not found' % self.img_normalize)
        
        self.image_transform = transforms.Compose(image_transform)
        
    def __len__(self):
        # looping over real dataset
        return self.real_set_len
    
    def __getitem__(self, idx):
        real_filename = self.real_path2files[idx % self.real_set_len]
        if self.paired_data:
            syn_filename = real_filename.replace(self.real_root_dir,self.syn_root_dir)
        else:
            rand_idx = random.randint(0, self.syn_set_len - 1)
            syn_filename = self.syn_path2files[rand_idx]


        image_real, coord_real, gt_poses_real, focal_length_real = self.fetch_img_coord(real_filename)
        image_syn, coord_syn, gt_poses_syn, focal_length_syn = self.fetch_img_coord(syn_filename)
        return_dict = {'real': [image_real, coord_real, gt_poses_real, focal_length_real],
                       'syn': [image_syn, coord_syn, gt_poses_syn, focal_length_syn]}

        return return_dict

    def fetch_img_coord(self, filename):

        image = PIL.Image.open(filename).convert('RGB')
        image = self.image_transform(image)

        coord = torch.load(filename.replace('rgb','init').replace('png','dat'))
        gt_poses = np.loadtxt(filename.replace('rgb','poses').replace('png','txt')) #camera to world
        gt_poses = torch.from_numpy(gt_poses).float()
        focal_length = float(np.loadtxt(filename.replace('rgb','calibration').replace('png','txt')))

        return image, coord, gt_poses, focal_length

    def batch_resize(self, batch):
        """
        Backbone collate_fn to resize data (images & coords) using a common scale factor.
        Usage: torch.utils.data.DataLoader(..., collate_fn=YOUR_DATASET.batch_resize, ...)
        """

        b_image = [item['real'][0] for item in batch]
        b_geo_labels = [item['real'][1] for item in batch]
        b_pose = [item['real'][2] for item in batch]
        b_focal_length = [item['real'][3] for item in batch]

        b_image_real, b_geo_labels_real, b_pose_real, b_focal_length_real, image_h_real, image_w_real, coords_h_real, coords_w_real = self.data_aug(b_image, b_geo_labels, b_pose, b_focal_length)

        b_image = [item['syn'][0] for item in batch]
        b_geo_labels = [item['syn'][1] for item in batch]
        b_pose = [item['syn'][2] for item in batch]
        b_focal_length = [item['syn'][3] for item in batch]

        b_image_syn, b_geo_labels_syn, b_pose_syn, b_focal_length_syn, image_h_syn, image_w_syn, coords_h_syn, coords_w_syn = self.data_aug(b_image, b_geo_labels, b_pose, b_focal_length)

        return_dict = {'real': [b_image_real, b_geo_labels_real, b_pose_real, b_focal_length_real,image_h_real,image_w_real, coords_h_real, coords_w_real],
                       'syn': [b_image_syn, b_geo_labels_syn, b_pose_syn, b_focal_length_syn, image_h_syn, image_w_syn, coords_h_syn, coords_w_syn]}

        return return_dict



    def data_aug(self, b_image, b_geo_labels, b_pose, b_focal_length):
        scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
        angle = random.uniform(-self.aug_rotation, self.aug_rotation)

        image_h = 4*math.ceil(b_image[0].size(1) * scale_factor/4) #force to be divisible by 4 because two downsample is included before U-net
        image_w = 4*math.ceil(b_image[0].size(2) * scale_factor/4)

        b_image_tensor = torch.stack(b_image, dim=0)
        b_image_tensor = F.interpolate(b_image_tensor, size=(image_h, image_w), mode='bilinear',
                                        align_corners=False)
        b_image_tensor = transforms.functional.rotate(b_image_tensor, angle, fill=-1)

        b_focal_length = [item * scale_factor for item in b_focal_length]

        # scale coordinates
        coords_w = math.ceil(b_image_tensor[0].size(2) / self.SUBSAMPLE)
        coords_h = math.ceil(b_image_tensor[0].size(1) / self.SUBSAMPLE)

        # single 3d geometric label, such as coords/depth/normal
        trim_b_geo_labels = torch.stack(b_geo_labels, dim=0)
        trim_b_geo_labels = F.interpolate(trim_b_geo_labels, size=(coords_h, coords_w), mode='nearest')
        trim_b_geo_labels = transforms.functional.rotate(trim_b_geo_labels, angle, fill=-1.0)

        return b_image_tensor, trim_b_geo_labels, torch.stack(b_pose), \
                torch.tensor(b_focal_length, dtype=torch.float64), image_h, image_w, coords_h, coords_w
        
    