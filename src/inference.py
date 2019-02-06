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

import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Map(object):
	def __init__(self, model):
		self.model = model

	def forward(self, x, params):
		return self.model.logjoint(x, params)

class MeanFieldVI(object):
	'''
	Mean field fully factorized variational inference.
	'''
	def __init__(self, model, num_samples):
		self.model = model
		self.num_samples = num_samples

	def unpack_var_params(self, params, T):
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

	def forwardOLD(self, x, var_params, model_params):
		T = x.size(0)
		loc, log_scale = self.unpack_var_params(var_params, T)
		samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = self.gaussian_entropy(log_scale)
		return (data_term + entropy)

	def forward(self, x, var_params, model_params):
		T = x.size(0)
		loc, log_scale = self.unpack_var_params(var_params, T)
		var_dist = Normal(loc, torch.exp(log_scale))
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)

class StructuredVIFullCovariance(object):
	'''
	Structured variational inference.
	- captures different variational families
	- AR-1 model on q(z)
	- low-rank structure on covariance matrix

	'''
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

	def forward(self, x, var_params, model_params):
		T = x.size(0)
		loc, log_scale = self.unpack_var_params(var_params, T)
		var_dist = MultivariateNormal(loc, torch.exp(log_scale))
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)