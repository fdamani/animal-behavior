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


class IS(object):
	'''importance sampling
	'''
	def __init__(self, model, num_particles=10, proposal_loc = 0.0, proposal_scale = 2.0):
		self.model = model
		self.num_particles = num_particles
		self.proposal_loc = torch.tensor(proposal_loc, requires_grad=False, device=device)
		self.proposal_scale = torch.tensor(proposal_scale, requires_grad=False, device=device)
		self.proposal_dist = Normal(self.proposal_loc, self.proposal_scale)

	def estimate(self, data):
		'''importance sample 
		'''
		# sample from proposal distribution
		samples = self.proposal_dist.sample(torch.Size((10,1)))
		# compute weights
		weights = torch.zeros(self.num_particles, requires_grad=False, device=device)
		for i,sx in enumerate(samples):
			p_x_z = self.model.logjoint(data, sx)
			q = self.proposal_dist.log_prob(sx)
			wx = p_x_z / q
			weights[i] = wx
		# normalize weights
		weights = self.normalize_weights(weights)
		exp_value = self.compute_expected_value(weights, samples)
		var = self.compute_variance(weights, samples)

		return exp_value, var

	def normalize_weights(self, weights):
		return weights / torch.sum(weights)

	def compute_expected_value(self, weights, samples):
		return torch.dot(weights.flatten(), samples.flatten())

	def compute_variance(self, weights, samples):
		return torch.var(weights.flatten() * samples.flatten())

class SMC(object):
	'''
	sequential monte carlo
	try this on simple LDS model where log prior is (0,1)
	transition and obs_scale set to 0.1
	'''
	def __init__(self, model):
		self.model = model


	def effective_sample_size(self, weights):
		'''compute ESS to decide when to resample
		inverse of sum of squares
		(sum (w_i)**2)**(-1)
		if ESS < k -> resample
		k = N/2 

		intuitively: ESS describes how many samples from the target
		would be equivalent to importance sampling with the weights 
		for this particle.

		if weights are evenly distributed, high entropy distribution
		effective sample size is high. if highly unbalanced, then
		low entropy and low ESS.
		'''
		return 1

	def multinomial_resampling(self, weights):
		'''
			given logits or normalized weights
			sample from multinomial/categorical
			***make sure its resampling with replacement.

			*note we are sampling full particle trajectories
			x_1:t given weights at time point t.
		'''



	def forward(self, x, var_params):
		'''
			useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
		'''
		return 1