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
from torch.distributions import Normal, Bernoulli
import models
from models import LDS, LogReg_LDS
import inference
from inference import MeanFieldVI, Map
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

	# lds = LogReg_LDS()
	lds = LDS()
	T = 2000
	latents, obs = lds.sample(T)

	## params
	obs = torch.tensor(obs, device=device)
	# latent_mean = torch.rand(T, requires_grad = True, device=device)
	latent_mean = torch.tensor(latents, requires_grad=False, device=device)
	scale = 1.0 # 0.01 # torch.rand(1)
	transition_log_scale = torch.tensor([math.log(scale)], 
		requires_grad=True, device=device)
	obs_log_scale = torch.tensor([math.log(scale)], 
		requires_grad=True, device=device)

	params = [latent_mean, transition_log_scale, obs_log_scale]
	diff_params = [transition_log_scale, obs_log_scale]
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

	# fig = plt.figure()
	# plt.plot(np.array(latents), label='true')
	# plt.plot(latent_mean.detach().cpu().numpy(), label='map')
	# plt.legend(loc='upper right')
	# plt.savefig('../output/latent_traj.png')
	# embed()


	# variational inference
	vi = MeanFieldVI(lds, num_samples=1)
	vi_params_grad = True
	model_params_grad = True
	diff_params = []
	if vi_params_grad:
		var_mean = torch.randn(T, requires_grad=True, device=device)
		var_log_scale = torch.tensor(torch.log(torch.rand(T)), requires_grad=True, device=device)
		diff_params.extend([var_mean, var_log_scale])
	else:
		var_mean = torch.tensor(np.load('../data/var_mean.npy'), requires_grad=False, device=device)
		var_log_scale = torch.tensor(np.load('../data/var_log_scale.npy'), requires_grad=False, device=device)

	var_params = [var_mean, var_log_scale]
	
	# model params
	init_scale = .001 # 0.01 # torch.rand(1)
	transition_log_scale = torch.tensor([math.log(init_scale)], 
		requires_grad=model_params_grad, device=device)
	obs_log_scale = torch.tensor([math.log(init_scale)], 
		requires_grad=model_params_grad, device=device)
	model_params = [transition_log_scale, obs_log_scale]
	if model_params_grad:
		diff_params.extend(model_params)

	optimizer = torch.optim.Adam(diff_params, lr = .01)
	outputs = []
	for t in range(80000):
		old_mean = var_mean.detach().cpu().numpy()
		optimizer.zero_grad()
		output = -vi.forward(obs, var_params, model_params)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % 2000 == 0:
			print 'iter: ', t, ' output: ', output.item(), ' norm: ', \
				np.linalg.norm(np.abs(var_mean.detach().cpu().numpy() - latents)), \
				'transition scale: ', np.exp(transition_log_scale.detach().cpu().numpy()), \
				'obs scale: ', np.exp(obs_log_scale.detach().cpu().numpy())

	fig = plt.figure()
	plt.plot(outputs[2500:])
	plt.savefig('../output/elbo_loss.png')

	fig = plt.figure()
	plt.plot(np.array(latents), label='true')
	plt.plot(latent_mean.detach().cpu().numpy(), label='map')
	plt.plot(var_mean.detach().cpu().numpy(), label='meanfield vi')
	plt.legend(loc='upper right')
	plt.savefig('../output/latent_traj.png')
	embed()