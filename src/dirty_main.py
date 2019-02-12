from __future__ import division
import time
import sys
import os
import numpy as np
import math
import matplotlib
# matplotlib.use('Agg')
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
from models import LDS, LogReg_LDS, LinearRegression
import inference
from inference import Map, MeanFieldVI, StructuredVITriDiagonal
import smc
from smc import IS, SMC
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
dtype = torch.float32

if __name__ == '__main__':

	grad_latents = True
	grad_model_params = False

	inference_types = ['map', 'mfvi', 'is', 'smc']
	inference_type = inference_types[3]
	T = 10
	num_particles = 100
	# time-series model
	if inference_type == 'smc':
		model = LDS()
		x, z_true = model.sample(T=T)
	else:
		model = LinearRegression(num_samples=10)
		x, y, z_true = model.sample(T=T)

	inference = None
	if inference_type == 'map':
		inference = Map(model)
	elif inference_type == 'mfvi':
		inference = MeanFieldVI(model)
	elif inference_type == 'is':
		inference = IS(model)
	elif inference_type == 'smc':
		inference = SMC(model, num_particles=num_particles, T=T)
	else:
		print 'error: select valid inference.'
		sys.exit()
	# dim = x.size(1)
	dim = len(x)
	# cast as torch tensors
	x = torch.tensor(x, dtype=dtype, device=device)
	data = x
	if inference_type != 'smc':
		data = [x,y]
	mean, var = inference.estimate(data)

	z_true = torch.tensor(z_true, requires_grad=False, dtype=dtype, device=device)
	z_mean = torch.rand(torch.tensor([dim]), requires_grad=grad_latents, dtype=dtype, device=device)
	z_log_scale = torch.tensor(torch.log(torch.rand(dim)), requires_grad=grad_latents, 
	 	dtype=dtype, device=device)
	
	plt.plot(z_true.detach().cpu().numpy(), label='true')
	plt.plot(mean.detach().cpu().numpy(), label='smc')
	plt.legend()
	plt.show()

	#opt_params = z_mean

	if inference_type == 'map':
		variational_params = [z_mean]
	else:
		variational_params = [z_mean, z_log_scale]
	# create list of params to optimize
	opt_params = []
	if grad_latents:
		opt_params.extend(variational_params)
	if grad_model_params:
		opt_params.extend(model.return_model_params())
	data = [x, y]
	print_every = 10
	# specify optimization objective
	optimizer = torch.optim.Adam(opt_params, lr = .01)
	outputs = []
	for t in range(1000):
		old_mean = z_mean.detach().cpu().numpy()
		optimizer.zero_grad()
		output = -inference.forward(data, opt_params)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % print_every == 0:
			print 'iter: ', t, ' output: ', output.item(), ' norm: ', \
				np.linalg.norm(np.abs(z_mean.detach().cpu().numpy() - z_true.cpu().numpy()))
	embed()
	plt.plot(outputs)


	# map_est = Map(lds)
	# lr = 0.01
	# optimizer = torch.optim.Adam(diff_params, lr = lr)
	# outputs = []
	# for t in range(10000):
	# 	old_mean = latent_mean.detach().cpu().numpy()
	# 	optimizer.zero_grad()
	# 	output = -map_est.forward(obs, params)
	# 	outputs.append(output.item())
	# 	output.backward()
	# 	optimizer.step()
	# 	if t % 2000 == 0:
	# 		print 'iter: ', t, ' output: ', output.item(), ' norm: ', \
	# 			np.linalg.norm(np.abs(latent_mean.detach().cpu().numpy() - latents)), \
	# 			'transition scale: ', np.exp(transition_log_scale.detach().cpu().numpy()), \
	# 			'obs scale: ', np.exp(obs_log_scale.detach().cpu().numpy())
	# plt.plot(outputs[2500:])
	# plt.savefig('../output/map_loss.png')


