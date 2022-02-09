import os, time, sys
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

from models.depth_generator_networks import _UNetGenerator, init_weights, _ResGenerator_Upsample
from models.discriminator_networks import Discriminator80x80InstNorm
from models.attention_networks import _Attention_FullRes
from models.cyclegan_networks import ResnetGenerator, NLayerDiscriminator

from utils.metrics import *
from utils.image_pool import ImagePool

from training.base_model import set_requires_grad, base_model

import warnings # ignore warnings
warnings.filterwarnings("ignore")

def value_scheduler(start, total_num_epoch, end=None, ratio=None, step_size=None, multiple=None, mode='linear'):
	if mode == 'linear':
		return np.linspace(start, end, total_num_epoch)
	elif mode == 'linear_ratio':
		assert ratio is not None
		linear = np.linspace(start, end, total_num_epoch * ratio)
		stable = np.repeat(end, total_num_epoch * (1 - ratio))
		return np.concatenate((linear, stable))

	elif mode == 'step_wise':
		assert step_size is not None
		times, res = divmod(total_num_epoch, step_size)
		for i in range(0, times):
			value = np.repeat(start * (multiple**i), step_size)
			if i == 0:
				final = value
			else:
				final = np.concatenate((final, value))

		if res != 0:
			final = np.concatenate((final, np.repeat(start * (multiple**(times)), res)))
		return final

class train_initial_attention_module_A(base_model):
	def __init__(self, args, dataloaders_xLabels_joint):
		super(train_initial_attention_module_A, self).__init__(args)
		self._initialize_training()

		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		# define loss weights
		self.lambda_identity = 0.5 # coefficient of identity mapping score
		self.lambda_real = 10.0
		self.lambda_synthetic = 10.0
		self.lambda_GAN = 1.0

		self.KL_loss_weight_max = 1.
		self.tau_min = 0.05
		self.tau_max = 0.9

		self.pool_size = 50
		self.generated_syn_pool = ImagePool(self.pool_size)
		self.generated_real_pool = ImagePool(self.pool_size)

		self.attModule = _Attention_FullRes(input_nc = 3, output_nc = 1, n_blocks=9, norm='instance')
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
		self.netD_s = NLayerDiscriminator(3, norm_layer=norm_layer)
		self.netD_r = NLayerDiscriminator(3, norm_layer=norm_layer)
		self.netG_s2r = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)
		self.netG_r2s = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, n_blocks=9)

		self.model_name = ['netD_s', 'netD_r', 'netG_s2r', 'netG_r2s', 'attModule']
		self.L1loss = nn.L1Loss()

		# ***********************
		self.save_steps = args.save_steps
		self.isbuffer = args.isbuffer
		self.start_epoch = args.start_epoch
		self.rho = args.rho

		if self.isTrain:
			self.netD_optimizer = optim.Adam(list(self.netD_s.parameters()) + list(self.netD_r.parameters()), lr=self.D_lr, betas=(0.5, 0.999))
			self.netG_optimizer = optim.Adam(list(self.netG_r2s.parameters()) + list(self.netG_s2r.parameters()) + list(self.attModule.parameters()), lr=self.G_lr, betas=(0.5, 0.999))
			self.optim_name = ['netD_optimizer', 'netG_optimizer']
			self._get_scheduler()
			# self._get_scheduler(constant_ratio = 1.0)  # finetune with constant lr
			self.loss_BCE = nn.BCEWithLogitsLoss()
			self.loss_MSE = nn.MSELoss() 
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

		self._check_parallel()

	def _get_project_name(self):
		return 'train_initial_attention_module_A'

	def _initialize_networks(self):
		for name in self.model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

	def compute_D_loss(self, real_sample, fake_sample, netD):
		syn_acc = 0
		real_acc = 0

		output = netD(fake_sample.detach())
		label = torch.full((output.size()), self.syn_label, device=self.device)

		predSyn = (output > 0.5).to(self.device, dtype=torch.float32)
		total_num = torch.numel(output)
		syn_acc += (predSyn==label).type(torch.float32).sum().item()/total_num
		loss_D_fake = self.loss_MSE(output, label.float())

		output = netD(real_sample)
		label = torch.full((output.size()), self.real_label, device=self.device)                    

		predReal = (output > 0.5).to(self.device, dtype=torch.float32)
		real_acc += (predReal==label).type(torch.float32).sum().item()/total_num
		loss_D_real = self.loss_MSE(output, label.float())
		loss = (loss_D_real + loss_D_fake) * 0.5

		return loss, syn_acc, real_acc

	def compute_G_loss(self, real_sample, synthetic_sample, r2s_rgb, s2r_rgb, rct_real, rct_syn, cs_imageListReal):
		'''
		real_sample: [batch_size, 4, 240, 320] real rgb
		synthetic_sample: [batch_size, 4, 240, 320] synthetic rgb
		r2s_rgb: netG_r2s(real)
		s2r_rgb: netG_s2r(synthetic)
		'''
		non_reduction_L1loss = nn.L1Loss(reduction='none')
		loss = 0

		# identity loss if applicable
		if self.lambda_identity > 0:
			idt_real = self.netG_s2r(real_sample)
			idt_synthetic = self.netG_r2s(synthetic_sample)
			idt_loss = (self.L1loss(idt_real, real_sample) * self.lambda_real + 
				self.L1loss(idt_synthetic, synthetic_sample) * self.lambda_synthetic) * self.lambda_identity
		else:
			idt_loss = 0

		# GAN loss
		real_pred = self.netD_r(s2r_rgb)
		real_label = torch.full(real_pred.size(), self.real_label, device=self.device)
		GAN_loss_real = self.loss_MSE(real_pred, real_label.float())

		syn_pred = self.netD_s(r2s_rgb)
		syn_label = torch.full(syn_pred.size(), self.real_label, device=self.device)
		GAN_loss_syn = self.loss_MSE(syn_pred, syn_label.float())

		GAN_loss = (GAN_loss_real + GAN_loss_syn) * self.lambda_GAN

		# cycle consist loss
		rec_real_loss = cs_imageListReal * non_reduction_L1loss(rct_real, real_sample)
		rec_real_loss = rec_real_loss.mean() * self.lambda_real

		rec_syn_loss = self.L1loss(rct_syn, synthetic_sample) * self.lambda_synthetic
		rec_loss = rec_real_loss + rec_syn_loss

		loss += (idt_loss + GAN_loss + rec_loss)

		return loss, idt_loss, GAN_loss, rec_loss

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
		y = target * torch.log(target/g) + (1-target) * torch.log((1-target)/(1-g))
		return y

	def train(self):
		phase = 'train'
		since = time.time()
		best_loss = float('inf')

		self.train_display_freq = len(self.dataloaders_xLabels_joint) // self.tensorboard_num_display_per_epoch

		tau_value_scheduler = value_scheduler(self.tau_max, self.total_epoch_num*10, end=self.tau_min, mode='linear') 

		tensorboardX_iter_count = 0
		for epoch in range(self.start_epoch,self.total_epoch_num):
			start_time = time.time()

			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()

			iterCount = 0

			for sample_dict in self.dataloaders_xLabels_joint:
				imageListReal, coord_real, gt_poses_real, focal_length_real = sample_dict['real']
				imageListSyn, coord_syn, gt_poses_syn, focal_length_syn = sample_dict['syn']

				imageListSyn = imageListSyn.to(self.device)
				imageListReal = imageListReal.to(self.device)

				with torch.set_grad_enabled(phase=='train'):
					s2r_rgb = self.netG_s2r(imageListSyn)
					rct_syn = self.netG_r2s(s2r_rgb)

					cs_imageListReal = self.attModule(imageListReal)[-1]
					cs_imageListReal = self.compute_spare_attention(cs_imageListReal, t=tau_value_scheduler[(10*iterCount)//len(self.dataloaders_xLabels_joint)+epoch*10], isTrain=True)
					mod_imageListReal = imageListReal * cs_imageListReal		
					r2s_rgb = self.netG_r2s(mod_imageListReal)

					rct_real = self.netG_s2r(r2s_rgb)

					#############  update generator
					set_requires_grad([self.netD_r, self.netD_s], False)
					netG_loss = 0.
					self.netG_optimizer.zero_grad()
					netG_loss, G_idt_loss, G_GAN_loss, G_rec_loss = self.compute_G_loss(imageListReal, imageListSyn,
						r2s_rgb, s2r_rgb, rct_real, rct_syn, cs_imageListReal)

					KL_loss = 0.
					KL_loss += self.compute_KL_div(cs_imageListReal, target=self.rho) * self.KL_loss_weight_max
					netG_loss += KL_loss

					netG_loss.backward()

					self.netG_optimizer.step()

					#############  update discriminator
					set_requires_grad([self.netD_r, self.netD_s], True)

					self.netD_optimizer.zero_grad()

					r2s_rgb_pool = self.generated_syn_pool.query(r2s_rgb)
					s2r_rgb_pool = self.generated_real_pool.query(s2r_rgb)

					if self.isbuffer:
						netD_s_loss, netD_s_syn_acc, netD_s_real_acc  = self.compute_D_loss(imageListSyn, r2s_rgb_pool, self.netD_s)
						netD_r_loss, netD_r_syn_acc, netD_r_real_acc = self.compute_D_loss(imageListReal, s2r_rgb_pool, self.netD_r)
					else:
						netD_s_loss, netD_s_syn_acc, netD_s_real_acc  = self.compute_D_loss(imageListSyn, r2s_rgb, self.netD_s)
						netD_r_loss, netD_r_syn_acc, netD_r_real_acc = self.compute_D_loss(imageListReal, s2r_rgb, self.netD_r)		

					netD_loss = netD_s_loss + netD_r_loss

					netD_loss.backward()

					self.netD_optimizer.step()

				iterCount += 1

				if self.use_tensorboardX:
					self.train_display_freq = len(self.dataloaders_xLabels_joint) # feel free to adjust the display frequency
					nrow = imageListReal.size()[0]
					if tensorboardX_iter_count % self.train_display_freq == 0:
						s2r_rgb_concat = torch.cat((imageListSyn, s2r_rgb, imageListReal, rct_syn), dim=0)
						self.write_2_tensorboardX(self.train_SummaryWriter, s2r_rgb_concat, name='RGB: syn, s2r, real, reconstruct syn', mode='image', 
							count=tensorboardX_iter_count, nrow=nrow)

						r2s_rgb_concat = torch.cat((imageListReal, r2s_rgb, imageListSyn, rct_real), dim=0)
						self.write_2_tensorboardX(self.train_SummaryWriter, r2s_rgb_concat, name='RGB: real, r2s, synthetic, reconstruct real', mode='image', 
							count=tensorboardX_iter_count, nrow=nrow)

						self.write_2_tensorboardX(self.train_SummaryWriter, cs_imageListReal, name='Atten: real', mode='image', 
							count=tensorboardX_iter_count, nrow=nrow, value_range=(0.0, 1.0))

					loss_val_list = [netD_loss, netG_loss, KL_loss]
					loss_name_list = ['netD_loss', 'netG_loss', 'KL_loss']
					self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

					tensorboardX_iter_count += 1

				if iterCount % 100 == 0:
					loss_summary = '\t{}/{} netD: {:.7f}, netG: {:.7f}'.format(iterCount, len(self.dataloaders_xLabels_joint), netD_loss, netG_loss)
					G_loss_summary = '\t\tG loss summary: netG: {:.7f}, idt_loss: {:.7f}, GAN_loss: {:.7f}, rec_loss: {:.7f}, KL_loss: {:.7f}'.format(
						netG_loss, G_idt_loss, G_GAN_loss, G_rec_loss, KL_loss)

					print(loss_summary)
					print(G_loss_summary)

					fn = open(self.train_log,'a')
					fn.write(loss_summary + '\n')
					fn.write(G_loss_summary + '\n')
					fn.close()


			if (epoch+1) % self.save_steps == 0:
				print('***********************************************************')
				self.save_models(self.model_name, mode=epoch+1)
				print('Model of Epoch {} is saved'.format(epoch+1))
				print('***********************************************************')

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
				fn.write(lr_update + '\n')
				fn.close()

		time_elapsed = time.time() - since
		print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		
		fn = open(self.train_log,'a')
		fn.write('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
		fn.close()
