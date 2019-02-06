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
from inference import MeanFieldVI
import psutil
process = psutil.Process(os.getpid())



if __name__ == '__main__':
	lds = LogReg_LDS()
	T = 2000
	latents, obs = lds.sample(T)
	obs = torch.tensor(obs, device=device)
	latent_mean = torch.rand(T, requires_grad = True, device=device)
	lr = 0.001
	optimizer = torch.optim.Adam([latent_mean], lr = lr)
	outputs = []
	for t in range(40000):
		old_mean = latent_mean.detach().cpu().numpy()
		optimizer.zero_grad()
		output = -lds.forward(obs, latent_mean)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % 2000 == 0:
			print 'iter: ', t, ' output: ', output.item(), ' norm: ', np.linalg.norm(np.abs(latent_mean.detach().cpu().numpy() - latents)), 'delta norm: ', np.linalg.norm(np.abs(latent_mean.detach().cpu().numpy() - old_mean))
	plt.plot(outputs)
	plt.savefig('map_loss.png')

	# variational inference
	vi = MeanFieldVI(lds, num_samples=1)
	var_mean = torch.randn(T, requires_grad=True, device=device)
	var_log_scale = torch.tensor(torch.log(torch.rand(T)), requires_grad=True, device=device)
	var_params = [var_mean, var_log_scale]
	optimizer = torch.optim.Adam(var_params, lr = lr)
	outputs = []
	for t in range(1200000):
		old_mean = var_mean.detach().cpu().numpy()
		optimizer.zero_grad()
		output = -vi.forward(obs, var_params)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % 2000 == 0:
			print 'iter: ', t, ' output: ', output.item(), ' norm: ', np.linalg.norm(np.abs(var_mean.detach().cpu().numpy() - latents)), 'delta norm: ', np.linalg.norm(np.abs(var_mean.detach().cpu().numpy() - old_mean))

	

	fig = plt.figure()
	plt.plot(outputs[2500:])
	plt.savefig('elbo_loss.png')


	fig = plt.figure()
	plt.plot(np.array(latents))
	plt.plot(latent_mean.detach().cpu().numpy())
	plt.plot(var_mean.detach().cpu().numpy())
	plt.savefig('latent_traj.png')
	embed()