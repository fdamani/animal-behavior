'''
	model classes
	each class inherits model abstract class
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

	def unpack_params(self, params, T):
		loc, log_scale = params[0], params[1]
		return loc, log_scale

	def q_sample(self, mean, log_scale):
		T = mean.size(0)
		Z = torch.randn(self.num_samples, T, device=device)
		samples = Z * torch.exp(log_scale) + mean
		return samples

	def gaussian_entropy(self, log_scale):
		D = log_scale.size(0)
		return 0.5 * D * (1.0 + math.log(2*math.pi)) + torch.sum(log_scale)

	def forward(self, x, var_params):
		T = x.size(0)
		loc, log_scale = self.unpack_params(var_params, T)
		samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i])
		data_term = torch.mean(data_terms)
		entropy = self.gaussian_entropy(log_scale)
		return (data_term + entropy)

def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

