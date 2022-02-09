import os, sys
import random, time, copy
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import functools
import torch.nn as nn
from PIL import Image

import random
import numpy as np

from models.depth_generator_networks import _UNetGenerator, init_weights, _ResGenerator_Upsample, _UNet_coord_down_8_skip_layer
from models.discriminator_networks import Discriminator80x80InstNorm
from models.attention_networks import _Attention_FullRes
from models.cyclegan_networks import ResnetGenerator, NLayerDiscriminator


import warnings # ignore warnings
warnings.filterwarnings("ignore")

print(sys.version)
print(torch.__version__)

def compute_spare_attention(confident_score, t):
	# t is the temperature --> scalar
	confident_score = confident_score / t

	confident_score = F.sigmoid(confident_score)

	return confident_score

################## set attributes for generating translated images ##################

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=os.path.join(os.getcwd(), 'experiments'),
							 	help='place to load checkpoint')
parser.add_argument('--path_to_real', type=str, default='your absolute path to real data',
								 help='absolute dir of real dataset')
parser.add_argument('--path_to_translate', type=str, default='your absolute path to store translated data',
								 help='absolute dir for storing translated dataset')
args = parser.parse_args()

norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
attModule = _Attention_FullRes(input_nc = 3, output_nc = 1, n_blocks=9, norm='instance')
inpaintNet = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)
styleTranslator = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)

preTrain_path = os.path.join(os.getcwd(), args.exp_dir, 'jointly_train_coord_regressor_C_and_attention_module_A/best_attModule.pth')
state_dict = torch.load(preTrain_path)
attModule.load_state_dict(state_dict)
attModule.to('cuda').eval()
print('***********************************************************************************************************************\n')
print('Successfully loaded pre-trained {} model from {}'.format('attModule', preTrain_path))

preTrain_path = os.path.join(os.getcwd(), args.exp_dir, 'train_inpainting_module_I/best_inpaintNet.pth')
state_dict = torch.load(preTrain_path)
inpaintNet.load_state_dict(state_dict)
inpaintNet.to('cuda').eval()
print('***********************************************************************************************************************\n')
print('Successfully loaded pre-trained {} model from {}'.format('inpaintNet', preTrain_path))

preTrain_path = os.path.join(os.getcwd(), args.exp_dir, 'train_style_translator_T/best_styleTranslator.pth')
state_dict = torch.load(preTrain_path)
styleTranslator.load_state_dict(state_dict)
styleTranslator.to('cuda').eval()
print('***********************************************************************************************************************\n')
print('Successfully loaded pre-trained {} model from {}'.format('styleTranslator', preTrain_path))

start_time = time.time()
print('start generate translated image')

TF2tensor = transforms.ToTensor()
images = os.listdir(os.path.join(args.path_to_real,'train/rgb'))
save_dir = os.path.join(args.path_to_translate,'train/rgb') # remember to add other data ('init','poses','calibration') to this dir if you want to generate train data for step 6
if not os.path.exists(save_dir): os.makedirs(save_dir)

for image_name in images:
	image = Image.open(os.path.join(args.path_to_real,'train/rgb',image_name)).convert('RGB')
	image = TF2tensor(image)
	image = image.unsqueeze(0)
	image = image.to('cuda')

	with torch.no_grad():
		r2s_img = styleTranslator(image)
		confident_score = attModule(image)[-1]
		# convert to sparse confident score
		confident_score = compute_spare_attention(confident_score, t=0.5)
		# hard threshold
		confident_score[confident_score < 0.5] = 0.
		confident_score[confident_score >= 0.5] = 1.
		masked_r2s_img = r2s_img * confident_score
		inpainted_r2s = inpaintNet(masked_r2s_img)

		reconst_img = inpainted_r2s * (1. - confident_score) + confident_score * r2s_img

	img = reconst_img.data
	image_numpy = img[0].cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0 
	image_pil = Image.fromarray(image_numpy.astype(np.uint8))
	image_pil.save(os.path.join(save_dir,image_name))