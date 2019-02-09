'''
	this main needs to be cleaned up and is not currently working
	1. config.py and pull in arguments to file
	2. make decision about whether model contains model params and latent vars 
'''


from __future__ import division
import time
import sys
import os
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import display, embed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD, Adam
from torch.distributions import Normal, Bernoulli, MultivariateNormal
import models
from models import LDS, LogReg_LDS
import inference
from inference import Map, MeanFieldVI, StructuredVITriDiagonal
from config import get_args
import psutil
process = psutil.Process(os.getpid())


# PARAMETERS
args = get_args()
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
sim_bool = args.simulation
T = args.sim_time_steps
grad_model_params = args.grad_model_params
grad_latents = args.grad_latents
model_type = args.model
inference_method = args.inference_method
lr = args.learning_rate
print_every = args.print_every

if __name__ == '__main__':
	print args
	model = None
	if model_type == 'LDS':
		model = LDS(grad_model_params)
	elif model_type == 'LogReg_LDS':
		model = LogReg_LDS()
	elif model_type == 'LearningDynamicsModel':
		model = LearningDynamicsModel()
	else:
		print 'choose valid model type.'
		sys.exit()

	inference = None
	if inference_method == 'map':
		inference = Map()
	elif inference_method == 'mfvi':
		inference = MeanFieldVI(model)
	elif inference_method == 'tri_diag_vi':
		inference = StructuredVITriDiagonal()
	elif inference_method == 'iwae':
		inference = IWAE()
	elif inference_method == 'fivo':
		inference = FIVO()
	else:
		print 'choose valid inference type.'
		sys.exit()

	# sample latents, and observations from model
	x, z_true = model.sample(T)

	# cast as torch tensors
	x = torch.tensor(x, dtype=dtype, device=device)
	z_true = torch.tensor(z_true, requires_grad=False, dtype=dtype, device=device)
	z_mean = torch.rand(T, requires_grad=grad_latents, dtype=dtype, device=device)
	z_log_scale = torch.tensor(torch.log(torch.rand(T)), requires_grad=grad_latents, 
		dtype=dtype, device=device)
	# if different variational family
	# z_log_scale = torch.tensor(torch.cat([torch.log(torch.ones(T)), 
	#	torch.log(.7*torch.ones(T-1))]), requires_grad=True, dtype=dtype, device=device)

	variational_params = [z_mean, z_log_scale]
	# create list of params to optimize
	opt_params = []
	if grad_latents:
		opt_params.extend(variational_params)
	if grad_model_params:
		opt_params.extend(model.return_model_params())


	# specify optimization objective
	optimizer = torch.optim.Adam(opt_params, lr = lr)
	outputs = []
	for t in range(50000):
		old_mean = z_mean.detach().cpu().numpy()
		optimizer.zero_grad()
		output = -inference.forward(x, variational_params)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % print_every == 0:
			print 'iter: ', t, ' output: ', output.item(), ' norm: ', \
				np.linalg.norm(np.abs(z_mean.detach().cpu().numpy() - z_true.cpu().numpy())), \
				model.return_model_params()
	 			#'transition scale: ', np.exp(transition_log_scale.detach().cpu().numpy()), \
	 			#'obs scale: ', np.exp(obs_log_scale.detach().cpu().numpy())


	plt.plot(outputs)
	# get date
	plt.savefig('../output/map_loss.png')

	fig = plt.figure()
	plt.plot(np.array(z_true.cpu().numpy()), label='true')
	plt.plot(z_mean.detach().cpu().numpy(), label='model')
	plt.legend(loc='upper right')
	plt.savefig('../output/latent_traj.png')
	embed()
