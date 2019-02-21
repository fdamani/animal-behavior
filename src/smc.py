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

class SMCOpt(object):
	'''
	sequential monte carlo
	try this on simple LDS model where log prior is (0,1)
	transition and obs_scale set to 0.1
	'''
	def __init__(self, 
				 model,
				 proposal_params,
				 num_particles=100,
				 T=50):
		isLearned=False
		self.q_init_latent_loc = proposal_params[0]
		self.q_init_latent_log_scale = proposal_params[1]
		self.q_transition_log_scale = proposal_params[2]

		self.dim = self.q_init_latent_loc.size(1)
		self.model = model
		
		self.num_particles = num_particles
		self.resampling_criteria = num_particles / 2.0
		self.T = T

		self.init_params()
	
	def init_params(self):
		self.particles_list = [torch.zeros((self.T, self.num_particles, self.dim))]
		
		#self.particles_list = [torch.zeros((self.num_particles, self.T))]
		self.inter_marginal = torch.zeros(self.T)
		self.weights = []

	def q_init_sample(self, num_samples):
		q_t = Normal(self.q_init_latent_loc.flatten(), torch.exp(self.q_init_latent_log_scale).flatten())
		return q_t.sample((torch.Size((num_samples,)))) ## might want rsample() to get reparameterized gradients

	def q_sample(self, x, z_past):
		'''
			z_t ~ q_t(z_t | x_1:t, z_1:t-1)

			specifically, z_t ~ q_t(z_t| z_t-1)
		'''
		samples = []
		for i in range(self.num_particles):
			# -1 indexes last time point t-1, i indexes particle
			z_mean = z_past[-1][i]
			#z_mean = z_past[i][-1]
			q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
			samples.append(q_t.rsample())
		return samples

	def q_sample_vec(self, x, z_past):
		'''
			z_t ~ q_t(z_t | x_1:t, z_1:t-1)

			specifically, z_t ~ q_t(z_t| z_t-1)
		'''
		samples = []
		# mean is last time point: particles x dimension
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return q_t.rsample()

	def q_init_logprob(self, z_sample):
		q_t = Normal(self.q_init_latent_loc, torch.exp(self.q_init_latent_log_scale))
		return torch.sum(q_t.log_prob(z_sample), dim=-1)
	def q_init_logprob_serial(self, z_sample):
		q_t = Normal(self.q_init_latent_loc, torch.exp(self.q_init_latent_log_scale))
		return torch.sum(q_t.log_prob(z_sample))

	def q_logprob(self, z_t, y_t, x, z_past):
		'''
			q_t(z_t | x_1:t, z_1:t-1)
		'''
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return torch.sum(q_t.log_prob(z_t),dim=-1)

	def q_logprob_serial(self, z_t, y_t, x, z_past):
		'''
			q_t(z_t | x_1:t, z_1:t-1)
		'''
		z_mean = z_past[-1]
		q_t = Normal(z_mean, torch.exp(self.q_transition_log_scale))
		return torch.sum(q_t.log_prob(z_t))

	def compute_incremental_weight(self, z_0_to_t, y_0_to_t, x_0_to_t):
		'''
			for particle i
			alpha_t (z_{1:t}^i) = p_t(x_t, z_t^i | x_{1:t-1}, z_{1:t-1}^i) / q_t(z_t^i | x_{1:t}, z_{1:t-1}^i)
	
			input: z_sample is z_t
			particle is z_1:t-1
			x is a vector x_{1:t}

			input: z_0_to_t: time x num_particles x dim
			vectorize over particles.

		'''
		log_wts = []
		z_sample = z_0_to_t[-1] # particles x dimension
		particles = z_0_to_t[0:-1] # time x particles x dim
		if particles.nelement() == 0:
			log_q = self.q_init_logprob(z_sample) # this is correct.
		else:
			log_q = self.q_logprob(z_sample, y_0_to_t, x_0_to_t, particles) # this is correct.

		log_lh, log_prior = self.model.logjoint_t(y_0_to_t, x_0_to_t, z_0_to_t)

		log_p_x_z_t = log_lh + log_prior
		log_wts = log_p_x_z_t - log_q
		return log_wts

		'''
		log_qs = []
		for i in range(self.num_particles):
			z_sample = z_0_to_t[-1][i] # of length dim
			#z_sample = z_0_to_t[i][-1] # confirm this is the right sample
			# all past points z_1:t-1
			particle = z_0_to_t[0:-1]
			if particle.nelement() == 0:
				log_q = self.q_init_logprob(z_sample)
			else:
				particle = z_0_to_t[0:-1, i]
				#particle = z_0_to_t[0:-1][i]
				log_q = self.q_logprob(z_sample, y_0_to_t, x_0_to_t, particle)
				log_qs.append(log_q)
			#particle = z_0_to_t[i][:-1] 
			# log p_t(x_t, z_t | x_1:t-1, z_1:t-1)
			log_lh, log_prior = self.model.logjoint_t(y_0_to_t, x_0_to_t, z_0_to_t[:, i])#.unsqueeze(dim=1))
			log_p_x_z_t = log_lh + log_prior
			#log_p_x_z_t = self.model.logjoint_t(x_0_to_t, z_0_to_t[i])
			# log q_t(z_t | x_1:t, z_1:t-1 )
			# if particle.nelement() == 0:
			# 	log_q = self.q_init_logprob(z_sample)
			# else:
			# 	log_q = self.q_logprob(z_sample, y_0_to_t, x_0_to_t, particle)
			# # log alpha = log p - log qself.
			log_weight = log_p_x_z_t - log_q
			log_wts.append(log_weight)
		embed()
		return torch.cat(log_wts)
		'''

	def compute_incremental_weight_serial(self, z_0_to_t, y_0_to_t, x_0_to_t):
		'''
			for particle i
			alpha_t (z_{1:t}^i) = p_t(x_t, z_t^i | x_{1:t-1}, z_{1:t-1}^i) / q_t(z_t^i | x_{1:t}, z_{1:t-1}^i)
	
			input: z_sample is z_t
			particle is z_1:t-1
			x is a vector x_{1:t}

			input: z_0_to_t: time x num_particles x dim

		'''
		log_wts = []
		for i in range(self.num_particles):
			z_sample = z_0_to_t[-1][i] # of length dim
			#z_sample = z_0_to_t[i][-1] # confirm this is the right sample
			# all past points z_1:t-1
			particle = z_0_to_t[0:-1]
			if particle.nelement() == 0:
				log_q = self.q_init_logprob_serial(z_sample)
			else:
				particle = z_0_to_t[0:-1, i]
				#particle = z_0_to_t[0:-1][i]
				log_q = self.q_logprob_serial(z_sample, y_0_to_t, x_0_to_t, particle)
			#particle = z_0_to_t[i][:-1] 
			# log p_t(x_t, z_t | x_1:t-1, z_1:t-1)
			log_lh, log_prior = self.model.logjoint_t_serial(y_0_to_t, x_0_to_t, z_0_to_t[:, i])#.unsqueeze(dim=1))
			log_p_x_z_t = log_lh + log_prior
			#log_p_x_z_t = self.model.logjoint_t(x_0_to_t, z_0_to_t[i])
			# log q_t(z_t | x_1:t, z_1:t-1 )
			# if particle.nelement() == 0:
			# 	log_q = self.q_init_logprob(z_sample)
			# else:
			# 	log_q = self.q_logprob(z_sample, y_0_to_t, x_0_to_t, particle)
			# # log alpha = log p - log qself.
			log_weight = log_p_x_z_t - log_q
			log_wts.append(log_weight)
		return torch.cat(log_wts)


	def normalize_log_weights(self, log_weights):
		'''
			input: N particle log_weights
			return normalized exponentiated weights
		'''
		logsoftmax = F.log_softmax
		log_norm_weights = logsoftmax(log_weights, dim=0)
		return log_norm_weights

	def reset_weights(self):
		return torch.ones(self.num_particles, device=device)
		#self.weights = torch.ones(self.num_particles, device=device)
		
	def multinomial_resampling(self, wx):
		'''
			wx is logits

			given logits or normalized weights
			sample from multinomial/categorical
			***make sure its resampling with replacement.

			*note we are sampling full particle trajectories
			x_1:t given weights at time point t.
		
			****ancestor sampling not reparameterizable. 
			****check if rsample() gives score function gradients
			****pass in argument grad = bool. to compare with and w/o score
				func estimator.

		'''
		# categorical over normalized weight vector for N particles
		# p is on simplex
		sampler = Categorical(logits=wx)
		# sample indices over particles
		ancestor_samples = sampler.sample(torch.Size((self.num_particles,)))
		return ancestor_samples

	def new_particles_from_ancestors(self, ancestors):
		return self.particles_list[-1][:, ancestors]
	
		# identify particles corresponding to indices
		# self.particles = self.particles[samples]
	def compute_expected_value(self, weights, particles):
		#return torch.sum(weights.unsqueeze(dim=-1) * particles, dim=0)
		return torch.sum(particles * weights[None, :, None], dim=1)

	def compute_scale(self, weights, particles):
		#return torch.var(weights.unsqueeze(dim=-1) * particles, dim=0)
		x_bar = self.compute_expected_value(weights, particles)
		sos = (particles - x_bar[:, None, :]) ** 2
		var = torch.sum(sos * weights[None, :, None], dim=1) / self.num_particles
		#sos = (particles - x_bar)**2
		#var = torch.sum(sos * weights[:, None], dim = 0) / self.num_particles
		return torch.sqrt(var)

	def estimate(self, data):
		y, x = self.model.unpack_data(data)
		exp_value = self.compute_expected_value(torch.exp(self.weights[-1]), self.particles_list[-1])
		var = self.compute_scale(torch.exp(self.weights[-1]), self.particles_list[-1])
		return exp_value, var

	def forward(self, data):
		#return self.particle_filter_serial(data)
		return self.particle_filter(data)

	def particle_filter(self, data):
		'''
			# time-step t=0: standard IS
				# sample z_0^i ~ q_0 for i = 1,...,N
				# compute normalized importance weights w_0^i, for i = 1,...,N
			# time-step t > 0
				# sample ancestor indices a_t-1^i ~ Cat(w_t-1)
				# update particles according to ancestor indices
				# sample z_t^i ~ q_t(z_t | ... )
				# append new z's to particle trajectories
				# compute new weights p/q and normalize.


				particles are T x particles x dim
		'''
		y, x = self.model.unpack_data(data)
		self.init_params()
		log_marginal_ll = []
		#weights = []
		
		# importance sample init
		t=0
		init_log_weights = torch.zeros(self.num_particles, device=device)
		init_samples = self.q_init_sample(self.num_particles)
		self.particles_list[-1][0] = self.particles_list[-1][0] + init_samples
		#self.particles_list[-1][:,0] = self.particles_list[-1][:,0] + init_samples.flatten()
		init_log_wts = self.compute_incremental_weight(self.particles_list[-1][0:t+1], y[0:t+1], x[0:t+1])

		#init_log_wts = self.compute_incremental_weight(self.particles_list[-1][:,0:t+1], y[0:t+1], x[0:t+1])

		# add to log marginal likelihood
		# normalize first
		log_marginal_ll.append(self.compute_inter_log_marginal_likelihood(init_log_wts))
		
		# normalize weights and save
		self.weights.append(self.normalize_log_weights(init_log_wts))

		for t in range(1, self.T):
			ancestor_inds = self.multinomial_resampling(self.weights[-1])
			ancestors = self.new_particles_from_ancestors(ancestor_inds)
			self.particles_list.append(ancestors)
			# sample z
			samples = torch.cat(self.q_sample(x[0:t+1], self.particles_list[-1][0:t]))
			# samples = torch.cat(self.q_sample(y[0:t+1], x[0:t+1], self.particles_list[-1][:, 0:t]))
			
			# append to particle trajectory
			self.particles_list[-1][t] = self.particles_list[-1][t] + samples
			#self.particles_list[-1][:, t] = self.particles_list[-1][:, t] + samples.flatten()
		
			log_weight_t = self.compute_incremental_weight(self.particles_list[-1][0:t+1], y[0:t+1], x[0:t+1])
			log_marginal_ll.append(self.compute_inter_log_marginal_likelihood(log_weight_t))

			self.weights.append(self.normalize_log_weights(log_weight_t))

			if t % 10 == 0:
				print t
		marginal_ll = torch.sum(torch.cat(log_marginal_ll))

		return marginal_ll
	

	def particle_filter_serial(self, data):
		'''
			# time-step t=0: standard IS
				# sample z_0^i ~ q_0 for i = 1,...,N
				# compute normalized importance weights w_0^i, for i = 1,...,N
			# time-step t > 0
				# sample ancestor indices a_t-1^i ~ Cat(w_t-1)
				# update particles according to ancestor indices
				# sample z_t^i ~ q_t(z_t | ... )
				# append new z's to particle trajectories
				# compute new weights p/q and normalize.


				particles are T x particles x dim
		'''
		y, x = self.model.unpack_data(data)
		self.init_params()
		log_marginal_ll = []
		#weights = []
		
		# importance sample init
		t=0
		init_log_weights = torch.zeros(self.num_particles, device=device)
		init_samples = self.q_init_sample(self.num_particles)
		self.particles_list[-1][0] = self.particles_list[-1][0] + init_samples
		#self.particles_list[-1][:,0] = self.particles_list[-1][:,0] + init_samples.flatten()
		init_log_wts = self.compute_incremental_weight_serial(self.particles_list[-1][0:t+1], y[0:t+1], x[0:t+1])

		#init_log_wts = self.compute_incremental_weight(self.particles_list[-1][:,0:t+1], y[0:t+1], x[0:t+1])

		# add to log marginal likelihood
		# normalize first
		log_marginal_ll.append(self.compute_inter_log_marginal_likelihood(init_log_wts))
		
		# normalize weights and save
		self.weights.append(self.normalize_log_weights(init_log_wts))

		for t in range(1, self.T):
			ancestor_inds = self.multinomial_resampling(self.weights[-1])
			ancestors = self.new_particles_from_ancestors(ancestor_inds)
			self.particles_list.append(ancestors)
			# sample z
			samples = torch.cat(self.q_sample(x[0:t+1], self.particles_list[-1][0:t]))
			# samples = torch.cat(self.q_sample(y[0:t+1], x[0:t+1], self.particles_list[-1][:, 0:t]))
			
			# append to particle trajectory
			self.particles_list[-1][t] = self.particles_list[-1][t] + samples
			#self.particles_list[-1][:, t] = self.particles_list[-1][:, t] + samples.flatten()
		
			log_weight_t = self.compute_incremental_weight_serial(self.particles_list[-1][0:t+1], y[0:t+1], x[0:t+1])
			log_marginal_ll.append(self.compute_inter_log_marginal_likelihood(log_weight_t))

			self.weights.append(self.normalize_log_weights(log_weight_t))
			
			if t % 10 == 0:
				print t
		return torch.sum(torch.cat(log_marginal_ll))
	def compute_inter_log_marginal_likelihood(self, log_wx):
		'''
			log 1/n sum_i w_t^i
		'''
		return (math.log(1.0) - math.log(self.num_particles) + torch.logsumexp(log_wx, dim=0)).unsqueeze(dim=0)

		# return torch.mean(torch.exp(torch.logsumexp(log_wx, dim=0))).unsqueeze(dim=0)
		# return torch.mean(wx)
