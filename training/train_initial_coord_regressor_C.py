import os, time, sys
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from models.depth_generator_networks import _UNetGenerator, init_weights, _UNet_coord_down_8_skip_layer, _UNet_coord_down_8_skip_layer_ft

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

class train_initial_coord_regressor_C(base_model):
	def __init__(self, args, dataloaders_xLabels_joint):
		super(train_initial_coord_regressor_C, self).__init__(args)
		self._initialize_training()
		
		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		if args.data_augment:
			self.coordRegressor = _UNet_coord_down_8_skip_layer_ft(input_nc = 3, output_nc = 3, norm='instance')
		else:
			self.coordRegressor = _UNet_coord_down_8_skip_layer(input_nc = 3, output_nc = 3, norm='instance')
		
		self.model_name = ['coordRegressor']
		self.L1loss = nn.L1Loss()

		######
		self.softclamp = args.softclamp
		self.hardclamp = args.hardclamp
		self.isdownsample = args.isdownsample # True
		if self.isdownsample:
			self.pixel_grid = get_pixel_grid(8)
		else:
			self.pixel_grid = get_pixel_grid(1)
		self.start_epoch = args.start_epoch
		self.data_augment = args.data_augment


		if self.isTrain:
			self.coord_optimizer = optim.Adam(self.coordRegressor.parameters(), lr=self.task_lr)
			self.optim_name = ['coord_optimizer']
			self._get_scheduler(optim_type='linear',constant_ratio = 0.4)
			self._initialize_networks()

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

		self.EVAL_best_loss = float('inf')
		self.EVAL_best_model_epoch = 0
		self.EVAL_all_results = {}

		self._check_parallel()

	def _get_project_name(self):
		return 'train_initial_coord_regressor_C'

	def _initialize_networks(self):
		for name in self.model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

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

	def train(self):
		phase = 'train'
		since = time.time()

		tensorboardX_iter_count = 0
		for epoch in range(self.start_epoch,self.total_epoch_num):
			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()

			self._set_models_train(['coordRegressor'])
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


				with torch.set_grad_enabled(phase=='train'):
					total_loss = 0.
					self.coordRegressor.zero_grad()
					real_coord_loss, valid_rate_real = self.compute_coord_loss(image_real, coord_real, gt_poses_real, focal_length_real,size_real,size_adapt=self.data_augment)
					syn_coord_loss, valid_rate_syn = self.compute_coord_loss(image_syn, coord_syn, gt_poses_syn, focal_length_syn,size_syn,size_adapt=self.data_augment)
					total_loss += (real_coord_loss + syn_coord_loss)

					total_loss.backward()

					self.coord_optimizer.step()

				sta_list.append([float(total_loss),float(real_coord_loss),float(valid_rate_real),float(syn_coord_loss),float(valid_rate_syn)])
				iterCount += 1

				if iterCount % 5 == 0:
					loss_summary = '\t{}/{} total_loss: {:.7f}, real_coord_loss: {:.7f}, syn_coord_loss: {:.7f}'.format(
						iterCount, len(self.dataloaders_xLabels_joint), total_loss, real_coord_loss, syn_coord_loss)

					print(loss_summary)
					print('valid prediction rate for real data is {:.7f}, valid prediction rate for syn data is {:.7f}'.format(valid_rate_real, valid_rate_syn))
					fn = open(self.train_log,'a')
					fn.write(loss_summary)
					fn.close()

			if self.use_tensorboardX:	
				sta_list = np.mean(sta_list,axis=0)
				print(sta_list)
				# add loss values
				loss_val_list = [sta_list[0],sta_list[1],sta_list[2],sta_list[3],sta_list[4]]
				loss_name_list = ['total_loss', 'real_coord_loss', 'valid_rate_real', 'syn_coord_loss','valid_rate_syn']
				self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

				tensorboardX_iter_count += 1
			print('\t average total loss: {:.7f}, real_coord_loss: {:.7f}, valid_rate_real: {:.7f}, syn_coord_loss: {:.7f}, valid_rate_syn: {:.7f}'.format(
				sta_list[0],sta_list[1],sta_list[2],sta_list[3],sta_list[4]))
			fn = open(self.train_log,'a')
			fn.write('\t average total loss: {:.7f}, real_coord_loss: {:.7f}, valid_rate_real: {:.7f}, syn_coord_loss: {:.7f}, valid_rate_syn: {:.7f}'.format(
				sta_list[0],sta_list[1],sta_list[2],sta_list[3],sta_list[4]))
			fn.close()
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
