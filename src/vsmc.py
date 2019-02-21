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
from torch.distributions import constraints, transform_to
import smc
from smc import SMCOpt
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class VSMC(object):
	'''
	Variational SMC
		- write this assuming we can access particle filter samples 

	posit a variational family -> q(z0; mean, var)
		z1 | z0 ~ N(z0, log_scale) with differentiable params
	- pass this in to proposal distribution as variational params lambda to optimize
	- optimize log marginal likelihood of particle filter
	- objective is expectation log p^hat_N(x_1:T)
		- 
	'''
	def __init__(self,
				 model,
				 proposal_params,
				 num_particles=100,
				 T=50):
		# declare smc proposal params as differentiable params
		# declare variational params and pass into SMCOpt as requires_grad=True
		#self.variational_params = variational_params
		self.num_particles = num_particles

		self.T = T
		self.smc = SMCOpt(model,
					      proposal_params=proposal_params,
						  num_particles=self.num_particles,
						  T=self.T)
		self.model = model

	def forward(self, data):
		'''
			useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
		'''

		return self.smc.particle_filter(data)
		#self.smc.particle_filter(x)
		#log_marginal_ll = self.smc.compute_log_marginal_likelihood()
		#return -log_marginal_ll