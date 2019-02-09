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
from torch.distributions import constraints, transform_to
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class Map(object):
	def __init__(self, model):
		self.model = model
	def unpack_params(self, params):
		return params[0]
	def forward(self, x, params):
		params = self.unpack_params(params)
		return self.model.logjoint(x, params)

class MeanFieldVI(object):
	'''
	Mean field fully factorized variational inference.
	'''
	def __init__(self, model, num_samples=1):
		self.model = model
		self.num_samples = num_samples

	def unpack_var_params(self, params):
		loc, log_scale = params[0], params[1]
		return loc, log_scale

	def forward(self, x, var_params):
		'''
			useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
		'''
		loc, log_scale = self.unpack_var_params(var_params)
		cov = torch.diag(torch.exp(log_scale))**2
		scale_tril = cov.tril()
		var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i])
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)

class StructuredVITriDiagonal(object):
	'''
	Structured variational inference.
	- captures different variational families
	- block tridiagonal
	- lower triangular parameterization means we only need band below diag
	'''
	def __init__(self):
		self.model = None
		self.num_samples = 0

	def initialize(self, model, num_samples=1):
		self.model = model
		self.num_samples = num_samples

	def unpack_var_params(self, params, T):
		loc, log_scale = params[0], params[1]
		cov = self.convert_log_scale_to_cov(log_scale, T)
		return loc, cov

	def convert_log_scale_to_cov(self, log_scale, T):
		a = torch.diag(torch.exp(log_scale[0:T])**2, diagonal=0)
		b = torch.diag(torch.exp(log_scale[T:T + T-1])**2, diagonal=-1)
		cov = a+b
		return cov

	def forward(self, x, var_params, model_params):
		T = x.size(0)
		loc, cov = self.unpack_var_params(var_params, T)
		scale_tril = cov.tril()
		var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, dtype=dtype, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)

class IWAE(object):
	'''
	importance weighted autoencoder
	special case of FIVO (no smc resampling)
	'''
	def __init__(self, model, num_particles=10):
		self.model = model
		self.num_particles = num_particles

	def unpack_var_params(self):
		return 1

	def forward(self):
		'''objective func'''

		return 1

class FIVO(object):
	'''
	Filtered variational objectives. 
	IWAE + smc
	should inherit an SMC object with resampling, etc
	'''
	def __init__(self):
		self.model = None
		self.num_particles = 0

	def initialize(self, model, num_particle=10):
		self.model = model
		self.num_particles = num_particles

	def unpack_var_params(self):
		return 1

	def forward(self):
		return 1

