import os, time, sys
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import functools

import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from models.depth_generator_networks import _UNetGenerator, init_weights, _ResGenerator_Upsample, _UNet_coord_down_8_skip_layer, _UNet_coord_down_8_skip_layer_ft
from models.discriminator_networks import Discriminator80x80InstNorm
from models.cyclegan_networks import ResnetGenerator, NLayerDiscriminator
from models.attention_networks import _Attention_FullRes

from utils.metrics import *
from utils.image_pool import ImagePool

from training.base_model import set_requires_grad, base_model

import warnings # ignore warnings
warnings.filterwarnings("ignore")

def get_pixel_grid(SUBSAMPLE):
    """
    Generate grid of target reprojection pixel positions (tensor)
    """
    pixel_grid = torch.zeros((2,
                              math.ceil(1080 / SUBSAMPLE),
                              # 1200px is max limit of image size, increase if needed
                              math.ceil(1080 / SUBSAMPLE)))

    for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
            pixel_grid[0, y, x] = x * SUBSAMPLE + SUBSAMPLE / 2
            pixel_grid[1, y, x] = y * SUBSAMPLE + SUBSAMPLE / 2

    pixel_grid = pixel_grid.cuda()
    return pixel_grid

def get_cam_mat(width, height, focal_length):
    """
    Get intrinsic camera matrix (tensor)
    """
    cam_mat = torch.eye(3)
    cam_mat[0, 0] = focal_length
    cam_mat[1, 1] = focal_length
    cam_mat[0, 2] = width / 2
    cam_mat[1, 2] = height / 2
    cam_mat = cam_mat.cuda()
    return cam_mat

def coords_world_to_cam(scene_coords, gt_coords, gt_poses):
    """
    Transform the scene coordinates to camera coordinates.
    @param scene_coords           [B, 3, N] Predicted scene coords tensor.
    @param gt_coords              [B, 3, N] Ground-truth scene coords tensor.
    @param gt_poses               [B, 4, 4] cam-to-world matrix.
    @return camera_coords         [B, 3, N] camera coords tensor corresponding to scene_coords.
    @return target_camera_coords  [B, 3, N] camera coords tensor corresponding to gt_coords.
    """
    gt_pose_inv = gt_poses.inverse()[:, 0:3, :]  # [B, 3, 4], world to camera matrix
    ones = torch.ones((scene_coords.size(0), 1, scene_coords.size(2))).cuda()

    scene_coords_ = torch.cat([scene_coords, ones], dim=1)  # [B, 4, N]
    gt_coords_ = torch.cat([gt_coords, ones], dim=1)  # [B, 4, N]

    camera_coords = torch.bmm(gt_pose_inv, scene_coords_)  # [B, 3, N] = [B, 3, 4] * [B, 4, N]
    target_camera_coords = torch.bmm(gt_pose_inv, gt_coords_)  # [B, 3, N] = [B, 3, 4] * [B, 4, N]

    return camera_coords, target_camera_coords

def pick_valid_points(coord_input, nodata_value, boolean=False):
    """
    Pick valid 3d points from provided ground-truth labels.
    @param   coord_input   [B, C, N] or [C, N] tensor for 3D labels such as scene coordinates or depth.
    @param   nodata_value  Scalar to indicate NODATA element of ground truth 3D labels.
    @param   boolean       Return boolean variable or explicit index.
    @return  val_points    [B, N] or [N, ] Boolean tensor or valid points index.
    """
    batch_mode = True
    if len(coord_input.shape) == 2:
        # coord_input shape is [C, N], let's make it compatible
        batch_mode = False
        coord_input = coord_input.unsqueeze(0)  # [B, C, N], with B = 1

    val_points = torch.sum(coord_input == nodata_value, dim=1) == 0  # [B, N]
    val_points = val_points.to(coord_input.device)
    if not batch_mode:
        val_points = val_points.squeeze(0)  # [N, ]
    if boolean:
        pass
    else:
        val_points = torch.nonzero(val_points, as_tuple=True)  # a tuple for rows and columns indices
    return val_points

def check_constraints(camera_coords, reproj_error, cam_coords_reg_error, mask_gt_coords_nodata,
                      min_depth = 0.1, max_reproj_error = 10., max_coords_reg_error = 50.0):
	"""
	Check constraints on network prediction.
	@param  camera_coords          [B, 3, N] tensor for camera coordinates.
	@param  reproj_error           [B, N] tensor for reprojection errors.
	@param  cam_coords_reg_error   [B, N] tensor for scene coordinate regression raw errors including invalid points.
	@param  mask_gt_coords_nodata  [B, N] tensor indicating points w/o valid scene coords labels.
	@param  min_depth              Scalar, threshold of minimum depth before camera panel in meter.
	@param  max_reproj_error       Scalar, threshold of maximum reprojection error in pixel.
	@param  max_coords_reg_error   Scalar, threshold of maximum scene coords regression error in meter.
	@return valid_sc               [B, N] Pixels w/ valid scene coords prediction, goes for reprojection error.
	"""
	# check predicted scene coordinate for various constraints
	invalid_min_depth = camera_coords[:, 2] < min_depth  # [B, N], behind or too close to camera plane
	invalid_repro = reproj_error > max_reproj_error      # [B, N], very large reprojection errors

	# check for additional constraints regarding ground truth scene coordinates
	invalid_gt_distance = cam_coords_reg_error > max_coords_reg_error  # [B, N] too far from ground truth scene coordinates
	invalid_gt_distance[mask_gt_coords_nodata] = 0  # [B, N], filter out unknown ground truth scene coordinates

	# print('mean reprojection error',reproj_error.mean())
	# print('1+2',torch.sum((invalid_min_depth + invalid_repro) == 0,dim=1))
	# print('2+3',torch.sum((invalid_repro + invalid_gt_distance) == 0,dim=1))
	# print('1+3',torch.sum((invalid_min_depth + invalid_gt_distance) == 0,dim=1))
	# print('1+2+3',torch.sum((invalid_min_depth + invalid_repro + invalid_gt_distance) == 0,dim=1))

	# combine all constraints
	valid_sc = (invalid_min_depth + invalid_repro + invalid_gt_distance) == 0  # [B, N]

	return valid_sc

def get_repro_err(camera_coords, cam_mat, pixel_grid_crop, min_depth=0.1):
    """
    Get reprojection error for each pixel.
    @param camera_coords        [B, 3, N] tensor for camera coordinates.
    @param cam_mat              [3, 3] tensor for intrinsic camera matrix.
    @param pixel_grid_crop      [2, N] tensor for pixel grid.
    @param min_depth            Scalar for minimum reprojected depth.
    @return reprojection_error  [B, N] tensor for reprojection error in pixel.
    """
    batch_size = camera_coords.size(0)
    reprojection_error = torch.bmm(cam_mat.expand(batch_size, -1, -1), camera_coords)  # [B, 3, H_ds*W_ds]
    reprojection_error[:, 2].clamp_(min=min_depth)  # avoid division by zero
    reprojection_error = reprojection_error[:, 0:2] / reprojection_error[:, 2:]  # [B, 2, H_ds*W_ds]

    reprojection_error = reprojection_error - pixel_grid_crop[None, :, :]
    reprojection_error = reprojection_error.norm(p=2, dim=1).clamp(min=1.e-7)  # [B, H_ds*W*ds]
    return reprojection_error

class jointly_train_coord_regressor_C_and_attention_module_A(base_model):
	def __init__(self, args, dataloaders_xLabels_joint):
		super(jointly_train_coord_regressor_C_and_attention_module_A, self).__init__(args)
		self._initialize_training()

		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

		self.attModule = _Attention_FullRes(input_nc = 3, output_nc = 1, n_blocks=9, norm='instance')
		self.inpaintNet = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)
		self.styleTranslator = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)
		self.netD = NLayerDiscriminator(3, norm_layer=norm_layer)
		if args.data_augment:
			self.coordRegressor = _UNet_coord_down_8_skip_layer_ft(input_nc = 3, output_nc = 3, norm='instance')
		else:
			self.coordRegressor = _UNet_coord_down_8_skip_layer(input_nc = 3, output_nc = 3, norm='instance')

		self.tau_min = 0.05
		self.KL_loss_weight = 1.0
		self.dis_weight = 1.0
		self.fake_loss_weight = 1e-3

		self.tensorboard_num_display_per_epoch = 1
		self.model_name = ['attModule', 'inpaintNet', 'styleTranslator', 'netD', 'coordRegressor']
		self.L1loss = nn.L1Loss()

		# ***********************
		if args.img_normalize == 'urban':
			self.TFNormalize = transforms.Normalize(mean=[0.4245, 0.4375, 0.3836],std=[0.1823, 0.1701, 0.1854]) #urban statistics
		elif args.img_normalize == 'nature':
			self.TFNormalize = transforms.Normalize(mean=[0.3636, 0.4331, 0.2956],std=[0.1383, 0.1457, 0.1147]) #nature statistics
		else:
			# feel free to add normalize for your own dataset
			raise NotImplementedError('normalization type [%s] is not found' % self.img_normalize)
		self.softclamp = args.softclamp
		self.hardclamp = args.hardclamp
		self.isdownsample = args.isdownsample
		if self.isdownsample:
			self.pixel_grid = get_pixel_grid(8)
		else:
			self.pixel_grid = get_pixel_grid(1)
		self.start_epoch = args.start_epoch
		self.rho = args.rho
		self.data_augment = args.data_augment
		print("rho is {:.7f}".format(self.rho))

		if self.isTrain:
			self.optim_netD = optim.Adam(self.netD.parameters(), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_coord = optim.Adam(list(self.coordRegressor.parameters()) + list(self.attModule.parameters()), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_name = ['optim_coord', 'optim_netD']
			self._get_scheduler()
			self.loss_BCE = nn.BCEWithLogitsLoss()

			self._initialize_networks(['netD'])

			if self.start_epoch > 0:
				self._load_models(self.model_name, self.start_epoch, isTrain=True,model_path=self.save_dir)
				# take step in optimizer
				for scheduler in self.scheduler_list:
					for _ in range(self.start_epoch):
						scheduler.step()
				for optimizer in self.optim_name:				
					lr = getattr(self, optimizer).param_groups[0]['lr']
					lr_update = 'Start with epoch {}/{} Optimizaer: {} learning rate = {:.7f} '.format(
						self.start_epoch+1, self.total_epoch_num, optimizer, lr)
					print(lr_update)
			else:

				# load the "best" coordRegressor C (from step 1)
				preTrain_path = os.path.join(os.getcwd(), self.exp_dir, 'train_initial_coord_regressor_C')
				self._load_models(model_list=['coordRegressor'], mode='best', isTrain=True, model_path=preTrain_path)
				print('Successfully loaded pre-trained {} model from {}'.format('coordRegressor', preTrain_path))

				# load the "best" style translator T (from step 2)
				preTrain_path = os.path.join(os.getcwd(), self.exp_dir, 'train_style_translator_T')
				self._load_models(model_list=['styleTranslator'], mode='best', isTrain=True, model_path=preTrain_path)
				print('Successfully loaded pre-trained {} model from {}'.format('styleTranslator', preTrain_path))

				# load the "best" attention module A (from step 3)
				preTrain_path = os.path.join(os.getcwd(), self.exp_dir, 'train_initial_attention_module_A')
				self._load_models(model_list=['attModule'], mode='best', isTrain=True,model_path=preTrain_path)
				print('Successfully loaded pre-trained {} model from {}'.format('attModule', preTrain_path))

				# load the "best" inpainting module I (from step 4)
				preTrain_path = os.path.join(os.getcwd(), self.exp_dir, 'train_inpainting_module_I')
				self._load_models(model_list=['inpaintNet'], mode='best', isTrain=True, model_path=preTrain_path)
				print('Successfully loaded pre-trained {} model from {}'.format('inpaintNet', preTrain_path))

		self.EVAL_best_loss = float('inf')
		self.EVAL_best_model_epoch = 0
		self.EVAL_all_results = {}

		self._check_parallel()

	def _get_project_name(self):
		return 'jointly_train_coord_regressor_C_and_attention_module_A'

	def _initialize_networks(self, model_name):
		for name in model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

	def compute_D_loss(self, real_sample, fake_sample, netD):
		loss = 0
		syn_acc = 0
		real_acc = 0

		output = netD(fake_sample)
		label = torch.full((output.size()), self.syn_label, device=self.device)
		predSyn = (output > 0.5).to(self.device, dtype=torch.float32)
		total_num = torch.numel(output)
		syn_acc += (predSyn==label).type(torch.float32).sum().item()/total_num

		loss += self.loss_BCE(output, label)

		output = netD(real_sample)
		label = torch.full((output.size()), self.real_label, device=self.device)                    
		predReal = (output > 0.5).to(self.device, dtype=torch.float32)
		real_acc += (predReal==label).type(torch.float32).sum().item()/total_num

		loss += self.loss_BCE(output, label)

		return loss, syn_acc, real_acc

	def compute_coord_loss(self, image, coord_gt, gt_poses, focal_length, size, size_adapt=False, nodata_value=-1.):
		
		if size_adapt:
			# feed with size information when data augmentation is activated
			# size = [[img_h, img_w], [coords_h, coords_w]]
			coord_pred = self.coordRegressor(image, size[0][0], size[0][1], size[1][0], size[1][1])
		else:
			coord_pred = self.coordRegressor(image)
		
		cam_mat = get_cam_mat(image.size(3), image.size(2), focal_length)
		pixel_grid_crop = self.pixel_grid[:, 0:coord_gt.size(2), 0:coord_gt.size(3)].clone().view(2, -1)

		coord_pred = coord_pred.view(coord_pred.size(0), 3, -1)  # [B, 3, H_ds*W_ds]
		coord_gt = coord_gt.view(coord_gt.size(0), 3, -1)  # [B, 3, H_ds*W_ds]

		camera_coords, target_camera_coords = coords_world_to_cam(coord_pred, coord_gt, gt_poses)  # [B, 3, H_ds*W_ds]*2
		camera_coords_reg_error = torch.norm(camera_coords - target_camera_coords, dim=1, p=2)  # [B, H_ds*W_ds]

		reprojection_error = get_repro_err(camera_coords, cam_mat, pixel_grid_crop)  # [B, H_ds*W_ds]

		# check for invalid/unknown ground truth scene coordinates
		mask_gt_coords_valdata = pick_valid_points(coord_gt[:, :3, :], nodata_value, boolean=True)  # [B, H_ds*W_ds]
		mask_gt_coords_nodata = torch.logical_not(mask_gt_coords_valdata)  # [B, H_ds*W_ds]

		# [B, H_ds*W_ds], warning: it is not coupled with mask_gt_coords_valdata!
		valid_scene_coordinates = check_constraints(camera_coords, reprojection_error, camera_coords_reg_error,
													mask_gt_coords_nodata, max_reproj_error = self.hardclamp)  
		
		invalid_scene_coordinates = torch.logical_not(valid_scene_coordinates)  # [B, H_ds*W_ds]
		num_valid_sc = valid_scene_coordinates.sum(dim=1).cpu().numpy()  # [B]
		num_pixels_batch = valid_scene_coordinates.numel()  # number of all pixels in the batch
		num_pixels_instance = valid_scene_coordinates[0].numel()  # number of pixels in one data point

		valid_pred_rate = float(num_valid_sc.sum() / num_pixels_batch)  # scalar
		# assemble loss
		loss = 0

		"""Reprojection error for all valid scene coordinates"""
		if num_valid_sc.sum() > 0:
			# calculate soft clamped l1 loss of reprojection error
			reprojection_error = reprojection_error * valid_scene_coordinates  # [B, H_ds*W_ds]
			loss_l1 = torch.sum(reprojection_error * (reprojection_error <= self.softclamp), dim=1).clamp(min=1.e-7)  # [B]
			loss_sqrt = reprojection_error * (reprojection_error > self.softclamp)  # [B, H_ds*W_ds]
			loss_sqrt = torch.sum(torch.sqrt(self.softclamp * loss_sqrt + 1.e-7), dim=1).clamp(min=1.e-7)  # [B]
			loss += loss_l1 + loss_sqrt  # [B]

		"""3D distance loss for all invalid scene coordinates where the ground truth is known"""
		invalid_scene_coordinates[mask_gt_coords_nodata] = 0  # filter out pixels w/o valid labels

		loss_3d = torch.sum(camera_coords_reg_error * invalid_scene_coordinates,
							dim=1)  # [B], applied to invalid pixels w/ valid labels
		loss += loss_3d
		
		loss = loss.sum()  # scalar, mean over each pixels within the batch	
		loss /= num_pixels_batch

		loss_3d = loss_3d.sum()
		loss_3d /= num_pixels_batch

		return loss, valid_pred_rate

	def compute_spare_attention(self, confident_score, t, isTrain=True):
		# t is the temperature --> scalar
		if isTrain:
			noise = torch.rand(confident_score.size(), requires_grad=False).to(self.device)
			noise = (noise + 0.00001) / 1.001
			noise = - torch.log(- torch.log(noise))

			confident_score = (confident_score + 0.00001) / 1.001
			confident_score = (confident_score + noise) / t
		else:
			confident_score = confident_score / t

		confident_score = F.sigmoid(confident_score)

		return confident_score

	def compute_KL_div(self, cf, target=0.5):
		g = cf.mean()
		g = (g + 0.00001) / 1.001 # prevent g = 0. or 1.
		y = target*torch.log(target/g) + (1-target)*torch.log((1-target)/(1-g))
		return y

	def compute_real_fake_loss(self, scores, loss_type, datasrc = 'real', loss_for='discr'):
		if loss_for == 'discr':
			if datasrc == 'real':
				if loss_type == 'lsgan':
					# The Loss for least-square gan
					d_loss = torch.pow(scores - 1., 2).mean()
				elif loss_type == 'hinge':
					# Hinge loss used in the spectral GAN paper
					d_loss = - torch.mean(torch.clamp(scores-1.,max=0.))
				elif loss_type == 'wgan':
					# The Loss for Wgan
					d_loss = - torch.mean(scores)
				else:
					scores = scores.view(scores.size(0),-1).mean(dim=1)
					d_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores).detach())
			else:
				if loss_type == 'lsgan':
					# The Loss for least-square gan
					d_loss = torch.pow((scores),2).mean()
				elif loss_type == 'hinge':
					# Hinge loss used in the spectral GAN paper
					d_loss = -torch.mean(torch.clamp(-scores-1.,max=0.))
				elif loss_type == 'wgan':
					# The Loss for Wgan
					d_loss = torch.mean(scores)
				else:
					scores = scores.view(scores.size(0),-1).mean(dim=1)
					d_loss = F.binary_cross_entropy_with_logits(scores, torch.zeros_like(scores).detach())

			return d_loss
		else:
			if loss_type == 'lsgan':
				# The Loss for least-square gan
				g_loss = torch.pow(scores - 1., 2).mean()
			elif (loss_type == 'wgan') or (loss_type == 'hinge') :
				g_loss = - torch.mean(scores)
			else:
				scores = scores.view(scores.size(0),-1).mean(dim=1)
				g_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores).detach())
			return g_loss

	def train(self):
		phase = 'train'
		since = time.time()

		set_requires_grad(self.styleTranslator, requires_grad=False) # freeze style translator T
		set_requires_grad(self.inpaintNet, requires_grad=False) # freeze inpainting module I

		tensorboardX_iter_count = 0
		for epoch in range(self.start_epoch,self.total_epoch_num):			
			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			if epoch <= self.total_epoch_num*2/3.:
				print("Adversial loss included")
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()

			self._set_models_train(['attModule', 'inpaintNet', 'styleTranslator', 'coordRegressor'])
			iterCount = 0

			sta_list = []
			start_time = time.time()
			for sample_dict in self.dataloaders_xLabels_joint:
				if self.data_augment:
					image_real, coord_real, gt_poses_real, focal_length_real, img_h_real, img_w_real, coords_h_real, coords_w_real = sample_dict['real']
					image_syn, coord_syn, gt_poses_syn, focal_length_syn, img_h_syn, img_w_syn, coords_h_syn, coords_w_syn = sample_dict['syn']
					size_real = [[img_h_real, img_w_real], [coords_h_real, coords_w_real]]
					size_syn = [[img_h_syn, img_w_syn], [coords_h_syn, coords_w_syn]]
				else:
					image_real, coord_real, gt_poses_real, focal_length_real = sample_dict['real']
					image_syn, coord_syn, gt_poses_syn, focal_length_syn = sample_dict['syn']
					size_real = [[0,0],[0,0]] #useless if no data_aug
					size_syn = [[0,0],[0,0]]

				focal_length_real = float(focal_length_real.view(-1)[0])
				focal_length_syn = float(focal_length_syn.view(-1)[0])

				image_real = image_real.to(self.device)
				coord_real = coord_real.to(self.device)
				gt_poses_real = gt_poses_real.to(self.device)

				image_syn = image_syn.to(self.device)
				coord_syn = coord_syn.to(self.device)
				gt_poses_syn = gt_poses_syn.to(self.device)

				B, C, H, W = image_real.size()[0], image_real.size()[1], image_real.size()[2], image_real.size()[3]

				with torch.set_grad_enabled(phase=='train'):
					r2s_img = self.styleTranslator(image_real)
					confident_score = self.attModule(image_real)[-1]
					# convert to sparse confident score
					confident_score = self.compute_spare_attention(confident_score, t=self.tau_min, isTrain=True)

					masked_r2s_img = r2s_img * confident_score
					inpainted_r2s = self.inpaintNet(masked_r2s_img)

					reconst_img = inpainted_r2s * (1. - confident_score) + confident_score * r2s_img

					reconst_img = self.TFNormalize(reconst_img)
					image_syn = self.TFNormalize(image_syn)

					# update coordinate regressor and attention module
					self.optim_coord.zero_grad()
					total_loss = 0.
					real_coord_loss, valid_rate_real = self.compute_coord_loss(reconst_img, coord_real, gt_poses_real, focal_length_real,size_real,size_adapt=self.data_augment)
					syn_coord_loss, valid_rate_syn = self.compute_coord_loss(image_syn, coord_syn, gt_poses_syn, focal_length_syn,size_syn,size_adapt=self.data_augment)
					KL_loss = self.compute_KL_div(confident_score, target=self.rho) * self.KL_loss_weight

					fake_pred = self.netD(reconst_img) # TODO: change "inpainted_r2s" into "reconst_img" here.
					fake_label = torch.full(fake_pred.size(), self.real_label, device=self.device)
					fake_loss = self.loss_BCE(fake_pred, fake_label) * self.fake_loss_weight

					total_loss += (real_coord_loss + syn_coord_loss + KL_loss + fake_loss)
					total_loss.backward()

					self.optim_coord.step()

					# stop adding adversaial loss after stable
					if epoch <= self.total_epoch_num*2/3.:
						self.optim_netD.zero_grad()
						netD_loss = 0.
						netD_loss, _, _ = self.compute_D_loss(image_syn, inpainted_r2s.detach(), self.netD)

						netD_loss.backward()

						self.optim_netD.step()
					else:
						netD_loss = 0.
						set_requires_grad(self.netD, requires_grad=False)

				iterCount += 1

				sta_list.append([float(total_loss),float(real_coord_loss),float(valid_rate_real),
									float(syn_coord_loss),float(valid_rate_syn),float(KL_loss),float(fake_loss),float(netD_loss)])

				if self.use_tensorboardX:
					nrow = image_real.size()[0]
					self.train_display_freq = len(self.dataloaders_xLabels_joint) # feel free to adjust the display frequency
					if tensorboardX_iter_count % self.train_display_freq == 0:
						img_concat = torch.cat((image_real, r2s_img, masked_r2s_img, inpainted_r2s, reconst_img), dim=0)
						self.write_2_tensorboardX(self.train_SummaryWriter, img_concat, name='real, r2s, r2sMasked, inpaintedR2s, reconst', mode='image',
							count=tensorboardX_iter_count, nrow=nrow)

						self.write_2_tensorboardX(self.train_SummaryWriter, confident_score, name='Attention', mode='image',
							count=tensorboardX_iter_count, nrow=nrow, value_range=(0., 1.0))

					# add loss values
					loss_val_list = [total_loss, real_coord_loss, syn_coord_loss, KL_loss, fake_loss, netD_loss]
					loss_name_list = ['total_loss', 'real_coord_loss', 'syn_coord_loss', 'KL_loss', 'fake_loss', 'netD_loss']
					self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

					tensorboardX_iter_count += 1

				if iterCount % 5 == 0:
					loss_summary = '\t{}/{}, total_loss: {:.7f}, netD_loss: {:.7f}'.format(iterCount, len(self.dataloaders_xLabels_joint), total_loss, netD_loss)
					G_loss_summary = '\t\t G loss summary: real_coord_loss: {:.7f}, syn_coord_loss: {:.7f}, KL_loss: {:.7f} fake_loss: {:.7f}'.format(real_coord_loss, syn_coord_loss, KL_loss, fake_loss)

					print(loss_summary)
					print(G_loss_summary)

					fn = open(self.train_log,'a')
					fn.write(loss_summary + '\n')
					fn.write(G_loss_summary + '\n')
					fn.close()

			sta_list = np.mean(sta_list,axis=0)
			print('\t average total loss: {:.7f}, real_coord_loss: {:.7f}, valid_rate_real: {:.7f}, syn_coord_loss: {:.7f}, valid_rate_syn: {:.7f}'.format(
				sta_list[0],sta_list[1],sta_list[2],sta_list[3],sta_list[4]))
			print('\t average KL_loss: {:.7f}, fake_loss: {:.7f}, netD_loss: {:.7f}'.format(
				sta_list[5],sta_list[6],sta_list[7]))

			# take step in optimizer
			for scheduler in self.scheduler_list:
				scheduler.step()
			for optim in self.optim_name:				
				lr = getattr(self, optim).param_groups[0]['lr']
				time_elapsed = time.time() - start_time
				lr_update = 'Epoch {}/{} finished: {} learning rate = {:.7f} time taken: {:.0f}m {:.0f}s'.format(
					epoch+1, self.total_epoch_num, optim, lr, time_elapsed // 60, time_elapsed % 60)
				print(lr_update)
				fn = open(self.train_log,'a')
				fn.write(lr_update)
				fn.close()

			if (epoch+1) % self.save_steps == 0:
				self.save_models(self.model_name, mode=epoch+1)

		time_elapsed = time.time() - since
		print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		
		fn = open(self.train_log,'a')
		fn.write('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
		fn.close()