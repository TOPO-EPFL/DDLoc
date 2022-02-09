import os, sys
import random, time, copy
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader.Joint_coord_dataLoader import Joint_coord_dataLoader


import random
import numpy as np

# step 1: Train Initial Coordinate Regressor C (pre-train C)
# from training.train_initial_coord_regressor_C import train_initial_coord_regressor_C as train_model

# step 2: Train Style Translator T
# from training.train_style_translator_T import train_style_translator_T as train_model

# step 3: Train Initial Attention Module A (pre-train A)
# from training.train_initial_attention_module_A import train_initial_attention_module_A as train_model

# step 4: Train Inpainting Module I 
# from training.train_inpainting_module_I import train_inpainting_module_I as train_model

# step 5: Jointly Train Coordinate Regressor C and Attention Module A (pre-train C)
# from training.jointly_train_coord_regressor_C_and_attention_module_A import jointly_train_coord_regressor_C_and_attention_module_A as train_model

# step 6: Finetune the Coordinate Regressor C with translated image
from training.finetune_coord_regressor_C import finetune_coord_regressor_C as train_model

import warnings # ignore warnings
warnings.filterwarnings("ignore")

print(sys.version)
print(torch.__version__)

################## set attributes for this project/experiment ##################

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=os.path.join(os.getcwd(), 'experiments'),
							 	help='place to store all experiments')
parser.add_argument('--project_name', type=str, help='Test Project')
parser.add_argument('--path_to_real', type=str, default='your absolute path to real data',
								 help='absolute dir of real dataset')
parser.add_argument('--path_to_syn', type=str, default='your absolute path to synthetic data',
							 	help='absolute dir of synthetic dataset')
parser.add_argument('--isTrain', action='store_true', help='whether this is training phase')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cropSize', type=list, default=[480, 720] , help='size of samples in experiments')
parser.add_argument('--total_epoch_num', type=int, default=50, help='total number of epoch')
parser.add_argument('--device', type=str, default='cpu', help='whether running on gpu')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataLoaders')

parser.add_argument('--softclamp', type=float, default=100, help='robust square root loss after this threshold, applied to reprojection error in pixels')
parser.add_argument('--hardclamp', type=float, default=1000, help='clamp loss with this threshold, applied to reprojection error in pixels')
parser.add_argument('--isdownsample', action='store_true', help='whether using downsampled dataset')
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch to train')
parser.add_argument('--data_augment', action='store_true', help='whether to implement data augmentation')
parser.add_argument('--data_ispaired', action='store_true', help='whether to use 1 to 1 matching data')
parser.add_argument('--img_normalize', type=str, default='pure',
							 	help='image normalization type, specific for each dataset')

args = parser.parse_args()


# fix random seed for reproducibility
torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.cuda.manual_seed_all(2021)


if torch.cuda.is_available(): 
	args.device='cuda'
	torch.cuda.empty_cache()

if args.isdownsample:
	print("training on downsampled dataset")
else:
	print("training on fullsize dataset")

datasets_xLabels_joint = Joint_coord_dataLoader(real_root_dir=args.path_to_real, syn_root_dir=args.path_to_syn,
								img_normalize=args.img_normalize, paired_data=args.data_ispaired, augment=args.data_augment)

if args.data_augment:
	dataloaders_xLabels_joint = DataLoader(datasets_xLabels_joint,
									batch_size=args.batch_size,
									shuffle=True,
									num_workers=args.num_workers,
									pin_memory=True, collate_fn=datasets_xLabels_joint.batch_resize)
else:
	dataloaders_xLabels_joint = DataLoader(datasets_xLabels_joint,
									batch_size=args.batch_size,
									shuffle=True,
									pin_memory=True,
									num_workers=args.num_workers)


model = train_model(args, dataloaders_xLabels_joint)

model.train()