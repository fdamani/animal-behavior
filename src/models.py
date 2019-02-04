'''
	model classes
	each class inherits model abstract class
'''
from __future__ import division
import time
import sys
import os
import numpy as np
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

import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def univar_normal(loc, scale):
	return Normal(loc, scale)
def bernoulli(param):
	return Bernoulli(param)

class LDS(object):
	def __init__(self):
		# initialize parameters
		self.init_latent_loc = 0.0
		self.init_latent_scale = 0.1
		self.transition_scale = 0.1
		self.obs_scale = 0.1

	def sample(self, T):
		'''
			sample latent variables and observations
		'''
		latents, obs = [], []
		log_prior_cpd = Normal(self.init_latent_loc, self.init_latent_scale)
		latents.append(log_prior_cpd.sample())
		obs_cpd = Normal(latents[0], self.obs_scale)
		obs.append(obs_cpd.sample())
		for i in range(1,T):
			transition_cpd = Normal(latents[i-1], self.transition_scale)
			latents.append(transition_cpd.sample())
			obs_cpd = Normal(latents[i], self.obs_scale)
			obs.append(obs_cpd.sample())
		return latents, obs

	def logjoint(self, x, latent_mean):
		'''
		input: x (observations T x D)
		input: latent_mean
		return logpdf under the model parameters
		'''
		T = x.size(0)
		# init log prior
		init_latent_logpdf = Normal(self.init_latent_loc, self.init_latent_scale)
		# transitions
		transition_logpdf = Normal(latent_mean[:-1], self.transition_scale)
		# observations
		obs_logpdf = Normal(latent_mean, self.obs_scale)

		# compute log probs
		logprob = init_latent_logpdf.log_prob(latent_mean[0])
		logprob += torch.sum(transition_logpdf.log_prob(latent_mean[1:]))
		logprob += torch.sum(obs_logpdf.log_prob(x))

		return logprob

	def forward(self, x, latent_mean):
		'''
		forward should define whatever objective function we want to take gradients with respect to
		examples:
		1) MAP/MLE: logpdf
		2) VI: elbo
		'''
		return self.logjoint(x, latent_mean)


class LogReg_LDS(object):
	def __init__(self):
		# initialize parameters
		self.init_latent_loc = 0.0
		self.init_latent_scale = 0.1
		self.transition_scale = 0.1
		self.sigmoid = nn.Sigmoid()

	def sample(self, T):
		'''
			sample latent variables and observations
		'''
		latents, obs = [], []
		log_prior_cpd = Normal(self.init_latent_loc, self.init_latent_scale)
		latents.append(log_prior_cpd.sample())
		obs_cpd = Bernoulli(self.sigmoid(latents[0]))
		obs.append(obs_cpd.sample())
		for i in range(1,T):
			transition_cpd = Normal(latents[i-1], self.transition_scale)
			latents.append(transition_cpd.sample())

			# pass latents[i] through a sigmoid
			obs_cpd = Bernoulli(self.sigmoid(latents[i]))
			obs.append(obs_cpd.sample())
		return latents, obs

	def logjoint(self, x, latent_mean):
		'''
		input: x (observations T x D)
		input: latent_mean
		return logpdf under the model parameters
		'''
		T = x.size(0)
		# init log prior
		init_latent_logpdf = Normal(self.init_latent_loc, self.init_latent_scale)
		# transitions
		transition_logpdf = Normal(latent_mean[:-1], self.transition_scale)
		# observations
		obs_logpdf = Bernoulli(self.sigmoid(latent_mean))

		# compute log probs
		logprob = init_latent_logpdf.log_prob(latent_mean[0])
		logprob += torch.sum(transition_logpdf.log_prob(latent_mean[1:]))
		logprob += torch.sum(obs_logpdf.log_prob(x))

		return logprob

	def forward(self, x, latent_mean):
		'''
		forward should define whatever objective function we want to take gradients with respect to
		examples:
		1) MAP/MLE: logpdf
		2) VI: elbo
		'''
		return self.logjoint(x, latent_mean)

class VI(object):
	def __init__(self, model, num_samples):
		self.model = model
		self.num_samples = num_samples

	def unpack_params(params, T):
		loc, log_scale = params[0:T], params[T:]
		return loc, log_scale

	def q_sample(mean, log_scale):
		T = mean.size(0)
		Z = torch.randn(self.num_samples, T)
		samples = Z * torch.exp(log_scale) + mean
		return samples

	def gaussian_entropy(log_scale):
		D = log_scale.size(0)
		return 0.5 * D * (1.0 + np.log(2*np.pi)) + torch.sum(log_scale)

	def forward(self, x, var_params):
		T = x.size(0)
		loc, log_scale = unpack_params(var_params, T)
		samples = self.q_samples(loc, log_scale)
		data_terms = []
		for sx in samples:
			data_terms.append(self.model.logjoint(x, sx))
		data_term = np.mean(data_term)
		entropy = self.gaussian_entropy(log_scale)

		return (data_term + entropy)

def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

if __name__ == '__main__':

	lds = LogReg_LDS()
	T = 2000
	latents, obs = lds.sample(T)
	obs = torch.tensor(obs, device=device)
	latent_mean = torch.rand(T, requires_grad = True, device=device)
	lr = 0.001
	optimizer = torch.optim.Adam([latent_mean], lr = lr)
	outputs = []
	for t in range(10):
		optimizer.zero_grad()
		output = -lds.forward(obs, latent_mean)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % 500 == 0:
			print 'iter: ', t, ' output: ', output.item(), ' norm: ', np.linalg.norm(np.abs(latent_mean.detach().cpu().numpy() - latents))
	plt.plot(outputs)
	plt.savefig('map_loss.png')

	# variational inference
	vi = VI(lds, num_samples=10)
	var_mean = torch.rand(T, requires_grad=True, device=device)
	embed()
	################## fix initializaition -> cant optimize non leaf node
	var_log_scale = torch.log(torch.exp(torch.rand(T, requires_grad=True, device=device)))
	var_params = [var_mean, var_log_scale]
	optimizer = torch.optim.Adam(var_params, lr = lr)
	outputs = []
	for t in range(10):
		optimizer.zero_grad()
		output = -vi.forward(obs, var_params)
		outputs.append(output.item())
		output.backward()
		optimizer.step()
		if t % 500 == 0:
			print 'iter: ', t, ' output: ', output.item() #, ' norm: ', np.linalg.norm(np.abs(latent_mean.detach().cpu().numpy() - latents))
	

	fig = plt.figure()
	plt.plot(outputs)
	plt.savefig('elbo_loss.png')


	fig = plt.figure()
	plt.plot(np.array(latents))
	plt.plot(latent_mean.detach().cpu().numpy())
	plt.plot(var_mean.detach().cpu().numpy())
	plt.savefig('latent_traj.png')
	embed()