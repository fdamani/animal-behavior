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


class MeanFieldVI(object):
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