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
from torch.distributions import Normal, Bernoulli, MultivariateNormal, Categorical
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
	def __init__(self, model, num_particles=10000, proposal_loc = 0.0, proposal_scale = 2.0):
		self.model = model
		self.num_particles = num_particles
		self.proposal_loc = torch.tensor(proposal_loc, requires_grad=False, device=device)
		self.proposal_scale = torch.tensor(proposal_scale, requires_grad=False, device=device)
		self.proposal_dist = Normal(self.proposal_loc, self.proposal_scale)

	def estimate(self, data):
		'''importance sample 
		'''
		# sample from proposal distribution
		samples = self.proposal_dist.sample(torch.Size((self.num_particles,1)))
		# compute weights
		log_weights = torch.zeros(self.num_particles, requires_grad=False, device=device)
		for i,sx in enumerate(samples):
			log_p_x_z = self.model.logjoint(data, sx)
			log_q = self.proposal_dist.log_prob(sx)
			log_wx = log_p_x_z - log_q
			log_weights[i] = log_wx
		# normalize weights
		softmax = F.softmax
		# probabilities on simplex
		norm_weights = softmax(log_weights)
		exp_value = self.compute_expected_value(norm_weights, samples)
		var = self.compute_variance(norm_weights, samples)
		return exp_value, var

	def normalize_weights(self, weights):
		return weights / torch.sum(weights)

	def compute_expected_value(self, weights, samples):
		return torch.dot(weights.flatten(), samples.flatten())

	def compute_variance(self, weights, samples):
		'''this might be the wrong variance
		E_q [w * f]
		variance of w*f evaluated under samples from q
		'''
		return torch.var(weights.flatten() * samples.flatten())

class SMC(object):
	'''
	sequential monte carlo
	try this on simple LDS model where log prior is (0,1)
	transition and obs_scale set to 0.1
	'''
	def __init__(self, model, 
				num_particles=1000,
				init_prior = (0.0, 0.5),
				transition_scale = 0.5, T=100, isLearned=False):
		self.model = model
		self.q_init_latent_loc = torch.tensor([init_prior[0]], 
			requires_grad=isLearned, device=device)
		self.q_init_latent_log_scale = torch.tensor([math.log(init_prior[1])], 
			requires_grad=isLearned, device=device)
		self.q_transition_log_scale = torch.tensor([math.log(transition_scale)], 
			requires_grad=isLearned, device=device)

		self.num_particles = num_particles
		self.T = T
		self.weights = torch.ones(self.num_particles, device=device)
		self.particles = torch.zeros((self.num_particles, self.T))

	def q_init_sample(self):
		q_t = Normal(self.q_init_latent_loc, torch.exp(self.q_transition_log_scale))
		return q_t.sample()

	def q_sample(self, x, z_past):
		'''
			z_t ~ q_t(z_t | x_1:t, z_1:t-1)
		'''
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return q_t.sample()

	def q_init_logprob(self, z_sample):
		q_t = Normal(self.q_init_latent_loc, torch.exp(self.q_transition_log_scale))
		return q_t.log_prob(z_sample)

	def q_logprob(self, z_t, x, z_past):
		'''
			q_t(z_t | x_1:t, z_1:t-1)
		'''
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return q_t.log_prob(z_t)

	def compute_incremental_weight(self, z_0_to_t, x_0_to_t):
		'''
			for particle i
			alpha_t (z_{1:t}^i) = p_t(x_t, z_t^i | x_{1:t-1}, z_{1:t-1}^i) / q_t(z_t^i | x_{1:t}, z_{1:t-1}^i)
	
			input: z_sample is z_t
			particle is z_1:t-1
			x is a vector x_{1:t}
		'''
		z_sample = z_0_to_t[-1]
		particle = z_0_to_t[:-1]
		z_0_to_t = torch.cat([particle, z_sample])
		log_p_x_z_t = self.model.logjoint_t(x_0_to_t, z_0_to_t)
		log_q = self.q_logprob(z_sample, x_0_to_t, particle)
		log_weight = log_p_x_z - log_q
		return log_weight

	def update_weights(self, alpha):
		'''
			multiply weights at time t-1 with incremental weights alpha for time t
		'''
		self.weights = self.weights * alpha

	def normalize_weights(self, log_weights):
		'''
			input: N particle log_weights
			return normalized exponentiated weights
		'''
		softmax = F.softmax
		norm_weights = softmax(log_weights)
		return norm_weights

	def reset_weights(self):
		self.weights = torch.ones(num_particles, device=device)

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
		return 1.0 / torch.sum(weights ** 2)
		
	def multinomial_resampling(self):
		'''
			given logits or normalized weights
			sample from multinomial/categorical
			***make sure its resampling with replacement.

			*note we are sampling full particle trajectories
			x_1:t given weights at time point t.
		'''
		# categorical over normalized weight vector for N particles
		sampler = Categorical(self.weights)
		# sample indices over particles
		samples = sampler.sample(torch.Size((self.num_particles)))
		# identify particles corresponding to indices
		self.particles = self.particles[samples]


	def resample(self, type):
		self.multinomial_resampling()
		self.reset_weights()


	def compute_weights(self, t, x):
		'''
			compute weights for time point t: w_t
			input: x
		'''
		x_0_to_t = x[0:t]
		# compute weight for each particle
		log_weights = torch.zeros(self.num_particles, requires_grad=False, device=device)
		for i in range(0, self.num_particles):
			z_sample = self.particles[i, t]
			# access ith particle trajectory z_0:t^i
			z_0_to_t = self.particles[i, 0:t]
			log_w = self.compute_incremental_weight(z_0_to_t, x_0_to_t)
			log_weights[i] = log_w
		# normalize weights
		alpha = self.normalize_weights(log_weights)
		self.update_weights(alpha)

	def estimate(self, x):
		'''
			# sample from q_t for N particle trajectories
			# compute incremental weights for samples
			# update weights by multiplying by incremental then normalizing
			# compute ESS of weights
			# if less than threshold:
				# resample particle trajectories from categorical dist over new weights
				# save new trajectories
				# set weights to one
		'''
		# compute particles and weights for t = 0
		for i in range(0, self.num_particles):
			# z_0^i = q(z_0)
			self.particles[i, 0] = self.q_init_sample()
		t = 0
		self.compute_weights(0, t + 1)
		# compute weights



		# for each time point
		for t in range(0, T):
			# for each particle
			for i in range(0, self.num_particles):
				# sample from q_t


			





		'''

		return 1