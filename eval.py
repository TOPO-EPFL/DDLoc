import os, sys
import random, time, copy
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import functools
import torch.nn as nn

from dataloader.Single_domain_dataloader import Single_domain_dataloader

import random
import numpy as np

from models.depth_generator_networks import _UNet_coord_down_8_skip_layer
from models.attention_networks import _Attention_FullRes
from models.cyclegan_networks import ResnetGenerator

from utils.coord_eval import scene_coords_eval

import warnings # ignore warnings
warnings.filterwarnings("ignore")

print(sys.version)
print(torch.__version__)

def compute_spare_attention(confident_score, t):
	# t is the temperature --> scalar
	confident_score = confident_score / t

	confident_score = F.sigmoid(confident_score)

	return confident_score

################## set attributes for evaluation ##################

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=os.path.join(os.getcwd(), 'experiments'),
							 	help='place to load checkpoint')
parser.add_argument('--path_to_data', type=str, default='your absolute path to data',
								 help='absolute dir of dataset, you should include a folder named test under this dir')
parser.add_argument('--img_normalize', type=str, default='urban',
							 	help='image normalization type, specific for each dataset')
args = parser.parse_args()

Eval_log = os.path.join(args.exp_dir, 'Eval.txt')
fn = open(Eval_log,'a')
fn.write('\nEvaluating dataset from {}\n'.format(args.path_to_data))
fn.write('--'*5+'\n')
fn.close()

dataset = Single_domain_dataloader(root_dir=args.path_to_data,set_name='test', img_normalize='pure')
dataloaders = DataLoader(dataset, batch_size = 1,shuffle = False, drop_last=False,num_workers=4)

if args.img_normalize == 'urban':
	TFNormalize = transforms.Normalize(mean=[0.4245, 0.4375, 0.3836],std=[0.1823, 0.1701, 0.1854]) #urban statistics
elif args.img_normalize == 'nature':
	TFNormalize = transforms.Normalize(mean=[0.3636, 0.4331, 0.2956],std=[0.1383, 0.1457, 0.1147]) #nature statistics
else:
	# feel free to add normalize for your own dataset
	raise NotImplementedError('normalization type [%s] is not found' % args.img_normalize)

norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
coordRegressor = _UNet_coord_down_8_skip_layer(input_nc = 3, output_nc = 3, norm='instance')
attModule = _Attention_FullRes(input_nc = 3, output_nc = 1, n_blocks=9, norm='instance')
inpaintNet = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)
styleTranslator = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)

preTrain_path = os.path.join(os.getcwd(), args.exp_dir, 'finetune_coord_regressor_C/best_coordRegressor.pth')
state_dict = torch.load(preTrain_path)
coordRegressor.load_state_dict(state_dict)
coordRegressor.to('cuda').eval()
print('***********************************************************************************************************************\n')
print('Successfully loaded pre-trained {} model from {}'.format('coordRegressor', preTrain_path))

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
print('start evaluating checkpoint')

acc_count = [0, 0, 0, 0]
t_err_list = []
r_err_list = []
est_xyz_list = []
coords_err_list = []

for sample in dataloaders:
	image, coord, gt_poses, focal_length = sample

	image = image.to('cuda')
	coord = coord.to('cuda')
	gt_poses = gt_poses.to('cuda')

	focal_length = float(focal_length.view(-1)[0])

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
		reconst_img = TFNormalize(reconst_img)

		coord_pred = coordRegressor(reconst_img).detach().to('cpu')

	coord = coord.to('cpu')
	gt_poses = gt_poses.to('cpu')

	t_err, r_err, est_xyz, coords_error = scene_coords_eval(coord_pred, coord, gt_poses, -1,
															 focal_length, image.size(2),image.size(3),output_subsample=8)

	if t_err < 5 and r_err < 5:
		acc_count[0] += 1

	if t_err < 10 and r_err < 7:
		acc_count[1] += 1

	if t_err < 20 and r_err < 10:
		acc_count[2] += 1

	if t_err < 30 and r_err < 10:
		acc_count[3] += 1

	t_err_list.append(t_err)
	r_err_list.append(r_err)
	est_xyz_list.append(est_xyz)
	coords_err_list.extend(coords_error)
	

Trans_err = 'Norm-2 of translate error: median={:.2f}[m] mean={:.2f}[m], std={:.2f}[m]'.format(np.median(t_err_list),np.mean(t_err_list),np.std(t_err_list))
Rot_err = 'Norm-2 of rotation angle error: median={:.2f}[deg] mean={:.2f}[deg], std={:.2f}[deg]'.format(np.median(r_err_list),np.mean(r_err_list),np.std(r_err_list))
Pixel_err = 'Norm-2 of pixel-wise coord error: median={:.2f}[m] mean={:.2f}[m], std={:.2f}[m]'.format(np.median(coords_err_list),np.mean(coords_err_list),np.std(coords_err_list))
Accuracy1 = 'percentage of point where translate_err < 5m and rotation err < 5 degree is {:.3f}'.format(float(acc_count[0]/len(dataloaders)))
Accuracy2 = 'percentage of point where translate_err < 10m and rotation err < 7 degree is {:.3f}'.format(float(acc_count[1]/len(dataloaders)))
Accuracy3 = 'percentage of point where translate_err < 20m and rotation err < 10 degree is {:.3f}'.format(float(acc_count[2]/len(dataloaders)))
Accuracy4 = 'percentage of point where translate_err < 30m and rotation err < 10 degree is {:.3f}'.format(float(acc_count[3]/len(dataloaders)))
time_elapsed = time.time() - start_time
time_log = 'time taken: {:.2f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60)

print(Trans_err)
print(Rot_err)
print(Pixel_err)
print(Accuracy1)
print(Accuracy2)
print(Accuracy3)
print(Accuracy4)
print('valid coord pixel rate is {:.2f}%'.format(100*len(coords_err_list)/(60.0*90.0*len(dataloaders))))
print(time_log)

fn = open(Eval_log,'a')
fn.write(Trans_err + '\n')
fn.write(Rot_err + '\n')
fn.write(Pixel_err + '\n')
fn.write(Accuracy1 + '\n')
fn.write(Accuracy2 + '\n')
fn.write(Accuracy3 + '\n')
fn.write(Accuracy4 + '\n')
fn.write(time_log + '\n')
fn.close()