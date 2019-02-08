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
		var_dist = Normal(loc, torch.exp(log_scale))
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)

	def forward(self, x, var_params, model_params):
		T = x.size(0)
		loc, log_scale = self.unpack_var_params(var_params, T)
		cov = torch.diag(torch.exp(log_scale))**2
		scale_tril = cov.tril()
		#scale_tril = transform_to(constraints.lower_cholesky)(cov)
		var_dist = MultivariateNormal(loc, scale_tril=scale_tril)#scale_tril=scale_tril)
		
		#var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
		#var_dist = Normal(loc, torch.exp(log_scale))
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)


class StructuredVITriDiagonal(object):
	'''
	Structured variational inference.
	- captures different variational families
	- AR-1 model on q(z)
	- low-rank structure on covariance matrix

	'''
	def __init__(self, model, num_samples):
		self.model = model
		self.num_samples = num_samples

	def unpack_var_params(self, params, T):
		loc, log_scale = params[0], params[1]
		cov = self.convert_log_scale_to_cov(log_scale, T)
		return loc, cov

	def convert_log_scale_to_cov(self, log_scale, T):
		a = torch.diag(torch.exp(log_scale[0:T])**2, diagonal=0)
		b = torch.diag(torch.exp(log_scale[T:T + T-1])**2, diagonal=-1)
		#c = torch.diag(torch.exp(log_scale[T+T-1:])**2, diagonal=-2)
		# c = torch.diag(torch.exp(log_scale[T + T-1:])**2, diagonal=1)
		cov = a+b#+c
		# cov = torch.pinverse(torch.pinverse(cov))
		# embed()
		return cov

	def forward(self, x, var_params, model_params):
		T = x.size(0)
		loc, cov = self.unpack_var_params(var_params, T)
		# add jitter to cov
		#  cov = cov #+ torch.tensor(1e-6*torch.diag(torch.ones(T)), device=device)
		#cov = cov # + (torch.eye(T, out=cov.new_empty(T,T)) * 1e-6)
		scale_tril = cov.tril()
		# scale_tril = transform_to(constraints.lower_cholesky)(cov)
		var_dist = MultivariateNormal(loc, scale_tril=scale_tril)# scale_tril=scale_tril)
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, dtype=dtype, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		return (data_term + entropy)

class StructuredVIAR1(object):
	'''
	AR1 variational family
	z1 ~ N(loc, scale)
	z2 ~ N(z1, scale)
	Structured variational inference.
	- captures different variational families
	- AR-1 model on q(z)
	- low-rank structure on covariance matrix

	'''
	def __init__(self, model, num_samples):
		self.model = model
		self.num_samples = num_samples

	def unpack_var_params(self, params, T):
		loc, log_scale = params[0], params[1]
		cov = self.convert_log_scale_to_cov(log_scale, T)
		return loc, cov

	def convert_log_scale_to_cov(self, log_scale, T):
		a = torch.diag(torch.exp(log_scale[0:T])**2, diagonal=0)
		b = torch.diag(torch.exp(log_scale[T:T + T-1])**2, diagonal=-1)
		c = torch.diag(torch.exp(log_scale[T + T-1:])**2, diagonal=1)
		cov = a+b+c
		return cov

	def forward(self, x, var_params, model_params):
		T = x.size(0)
		loc, cov = self.unpack_var_params(var_params, T)
		# add jitter to cov
		cov = cov + torch.tensor(1e-6*torch.diag(torch.ones(T)), dtype=dtype, device=device)
		try:
			var_dist = MultivariateNormal(loc, cov)
		except:
			embed()
		samples = var_dist.rsample(torch.Size((self.num_samples,)))
		# samples = self.q_sample(loc, log_scale)
		data_terms = torch.empty(self.num_samples, dtype=dtype, device=device)
		for i in range(len(samples)):
			data_terms[i] = self.model.logjoint(x, samples[i], model_params)
		data_term = torch.mean(data_terms)
		entropy = torch.sum(var_dist.entropy())
		print (data_term+entropy)
		return (data_term + entropy)