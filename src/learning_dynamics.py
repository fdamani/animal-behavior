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
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import utils
from utils import get_gpu_memory_map
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class LearningDynamicsModel(object):
	def __init__(self,
				 init_prior=(0.0, 1.0),
				 transition_log_scale=math.log(0.01),
				 beta=4., 
				 log_alpha= -2.,
				 dim=3, grad=False):
		# initialize parameters

		grad_model_params=False
		self.init_latent_loc = torch.tensor([init_prior[0]], 
			requires_grad=grad_model_params, device=device)
		self.init_latent_log_scale = torch.tensor([init_prior[1]], 
			requires_grad=grad_model_params, device=device)
		self.transition_log_scale = torch.tensor([transition_log_scale], 
			requires_grad=grad, device=device)
		self.beta = torch.tensor([beta], requires_grad=grad, device=device)
		self.log_alpha = torch.tensor([log_alpha], requires_grad=grad,device=device)
		self.sigmoid = nn.Sigmoid()

	def init_grad_vbles(self):
		self.transition_log_scale.requires_grad_(True)
		self.beta.requires_grad_(True)
		self.log_alpha.requires_grad_(True)

	def init_no_grad_vbles(self):
		self.transition_log_scale.requires_grad_(False)
		self.beta.requires_grad_(False)
		self.log_alpha.requires_grad_(False)

	def sample(self, T, num_obs_samples=10, dim=3):
		'''
			sample latent variables and observations
		'''
		# generate 1D x from standard normal
		intercept = torch.ones(T, num_obs_samples, 1, device=device)
		x = torch.randn(T, num_obs_samples, dim-1, device=device)
		x = torch.cat([intercept, x], dim=2)
		z = [self.sample_init_prior()]
		z = [self.sample_init_prior()]

		y = [self.sample_likelihood(x[0], z[0], num_obs_samples)]
		for i in range(1, T):
			# sample and append a new z
			z.append(self.sample_prior(z[i-1], y[-1], x[-1]))
			# sample an observation
			y.append(self.sample_likelihood(x[i], z[i], num_obs_samples))

		y = torch.t(torch.cat(y, dim = 1))
		z = torch.cat(z)
		return y, x, z
	def basis(self, x):
		'''
		given input x, learn a basis with a useful hidden representation
		op 1: 1D convolution on x outputs high-d hidden representation per time point
		op 2: rnn outputting hidden rep at each time point
		might want to switch to a class if multiple options we want to try
		'''
		return 1

	def unpack_data(self, data):
		y, x = data[0], data[1]
		return y, x

	def log_joint_notvec(self, y, x, z):
		'''
		input: x (observations T x D)
		input: latent_mean
		return logpdf under the model parameters

		y is T x num obs
		x is T x num obs x dim
		z is T x dim
		'''
		T = y.size(0)
		logprob = 0
		logprob += self.log_init_prior(z[0][None])
		logprob += self.log_likelihood_t(y[0][:, None], x[0][None, :], z[0][None, :])
		for i in range(1, T):
			logprob += self.log_prior_t(z[0:i+1], y[0:i+1], x[0:i+1])
			logprob += self.log_likelihood_t(y[0:i+1], x[0:i+1], z[0:i+1])

		return logprob

#####################################################################################################
## vectorized functions ##
	def log_joint(self, y, x, z):
		'''
		input: x (observations T x D)
		input: latent_mean
		return logpdf under the model parameters
		'''
		T = y.size(0)
		logprob = 0

		logprob += self.log_init_prior(z[0][None])
		logprob += self.log_prior_vec(z, y, x)
		logprob += self.log_likelihood_vec(y, x, z)

		return logprob

	def log_joint_batch(self, y, x, z):
		''' vectorize over particles and time.
		input: x (observations T x D)
		input: latent_mean
		return logpdf under the model parameters
		'''
		T = y.size(0)
		z = z.transpose(1,0)
		# vector of length num particles
		logprob = self.log_init_prior_batch(z[0])
		z = z.transpose(1,0)
		# add elementwise
		logprob += self.log_prior_batch_compl(z, y, x)
		logprob += self.log_likelihood_compl_batch(y, x, z)
		return logprob


	def log_joint_batch_bootstrap(self, y1, x1, z, z1):
		''' vectorize over particles and time.
		input: x (observations T x D)
		input: latent_mean
		return logpdf under the model parameters
		'''
		T = y1.size(0)
		logprob = self.log_prior_batch_compl_bootstrap(z, y1, x1, z1)
		return logprob

	def log_likelihood_vec(self, y, x, z):
		'''
			p(y_t | y_1:t-1, x_1:t, z_1:t)
		'''
		logits = torch.sum(x * z[:, None, :], dim=2)
		obs = Bernoulli(logits=logits)
		return torch.sum(obs.log_prob(y))

	def log_prior_vec(self, z, y, x):
		'''
			input: z_1:t
			parameterize p(z_t | z_t-1, theta)
		'''
		z_prev = z[0:-1]
		z_curr = z[1:]

		# properly vectorized
		grad_rat_obj = self.grad_rat_obj_score_vec(y, x, z)[0:-1]
		# l2 = self.sigmoid(self.beta) * z_prev
		learning = torch.exp(self.log_alpha) * grad_rat_obj
		# l1 = torch.sign(z_prev) * torch.exp(self.log_sparsity)

		penalty = self.sigmoid(self.beta)
		regularization = penalty * z_prev + (1.0 - penalty) * -torch.sign(z_prev)
		# sparsity = self.compute_sparsity_vec(z_prev)
		mean = learning + regularization
		# mean = l2 + learning - l1
		scale = torch.exp(self.transition_log_scale)
		prior = Normal(mean, scale)
		return torch.sum(prior.log_prob(z_curr))

	def grad_rat_obj_score_vec(self, y, x, z):
		'''
			many time points with one particle
			grad log p(y|x, z)
			grad rat objective function using score function estimator

			**************edit this function to be vectorized.


		'''
		prob_y_1_given_x_z = self.rat_policy_vec(x, z) # T x obs
		prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

		# r(action=1, x)
		r_y_1_x = self.rat_reward_vec(torch.tensor([1.], device=device), x)
		r_y_0_x = 1.0 - r_y_1_x

		# grad of logistic regression: x_n(y_n - sigmoid(z^t x))
		y_1 = torch.ones(y.size(0), y.size(1), device=device)
		grad_log_policy_y1 = self.grad_rat_policy_vec(y_1, x, z)

		y_0 = torch.zeros(y.size(0), y.size(1))
		grad_log_policy_y0 = self.grad_rat_policy_vec(y_0, x, z)

		per_sample_gradient = prob_y_1_given_x_z[:,:,None] * grad_log_policy_y1 * r_y_1_x[:,:,None] + \
			prob_y_0_given_x_z[:,:,None] * grad_log_policy_y0 * r_y_0_x[:,:,None]

		avg_gradient = torch.mean(per_sample_gradient, dim=1)
		return avg_gradient

	def grad_rat_obj_score_batch(self, y, x, z):
		'''
			single time point with batch of particles

			grad log p(y|x, z)
			grad rat objective function using score function estimator

			x is time x obs x dim
			y is time x obs x 1
			z is time x particles x dim

		'''
		if len(z.size()) == 2:
			z = z[None]
			x = x[None]
			y = y[None]
			T = 1
		else:
			T = z.size(0)
			assert T == x.size(0)
		num_particles = z.size(1)
		prob_y_1_given_x_z = self.rat_policy_batch(x, z) # T x obs
		prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

		# r(action=1, x)
		# T x 250
		r_y_1_x = self.rat_reward_vec(torch.tensor([1.], device=device), x)
		r_y_0_x = 1.0 - r_y_1_x

		# grad of logistic regression: x_n(y_n - sigmoid(z^t x))
		y_1 = torch.ones(T, num_particles, x.size(1), device=device)
		#assert x.size(1) == 250
		grad_log_policy_y1 = self.grad_rat_policy_batch(y_1, x, z)

		y_0 = torch.zeros(T, num_particles, x.size(1), device=device)
		grad_log_policy_y0 = self.grad_rat_policy_batch(y_0, x, z)

		per_sample_gradient = prob_y_1_given_x_z[:, :, :, None] * grad_log_policy_y1 * \
				r_y_1_x[:, None, :, None] + prob_y_0_given_x_z[:, :, :, None] * \
				grad_log_policy_y0 * r_y_0_x[:, None, :, None]
		# average over observations (within a single time-step)
		avg_gradient = torch.mean(per_sample_gradient, dim=2)
		return avg_gradient

	def rat_policy_batch(self, x, z):
		''' vectorize over time and particles
			single time point many particles
			x is time x obs x dim
			z is time x particles x dim

		p(y = 1 | x, z)
			z is 1 x 3
			x is num samples x dimension

			x is time x num samples x dimension
			z = time x dimension
		'''
		return self.sigmoid(torch.bmm(z, x.transpose(2, 1))) # t x particles x observations

		# prob = self.sigmoid(torch.sum(x * z[:, None, :], dim=2))

		# #prob = torch.t(prob)
		# assert prob.size(0) == x.size(0)
		# assert prob.size(1) == x.size(1)
		# return prob



	def rat_policy_vec(self, x, z):
		'''
		p(y = 1 | x, z)
			z is 1 x 3
			x is num samples x dimension

			x is time x num samples x dimension
			z = time x dimension
		'''
		prob = self.sigmoid(torch.sum(x * z[:, None, :], dim=2))

		#prob = torch.t(prob)
		assert prob.size(0) == x.size(0)
		assert prob.size(1) == x.size(1)
		return prob
	
	def rat_reward_vec(self, action, x):
		'''
		rat's reward func
		action is {0,1}
		x is T x 2 
		assume this is preprocessing. we compute rewards ahead of time.
		'''
		stim1 = x[:, :, 1]#.unsqueeze(dim=1)
		stim2 = x[:, :, 2]#.unsqueeze(dim=1)
		rewards = ((stim1 > stim2)*(action==1) + (stim1 < stim2)*(action==0))
		return rewards.float()

	def grad_rat_policy_vec(self, y, x, z):
		'''gradient of logistic regression
			grad log p(y|x, z)
			y: T x 1
			x: T x obs x dim
			z: T x dim
		'''
		prob = self.rat_policy_vec(x, z) # T x obs
		assert prob.size(0) == x.size(0)
		assert prob.size(1) == x.size(1)
		# this is the gradient: weight error by input features
		error = (y - prob)
		assert error.size(0) == x.size(0)
		assert error.size(1) == x.size(1)
		grad_log_policy = x * error[:, :, None] # T x num samples x dimension
		assert grad_log_policy.size(0) == x.size(0)
		assert grad_log_policy.size(1) == x.size(1)
		assert grad_log_policy.size(2) == x.size(2)
		return grad_log_policy
		# grad_log_policy_avg = torch.mean(grad_log_policy, dim=0)[None] # 1 x dimension
		# assert grad_log_policy_avg.size(-1) == z.size(-1)
		# return grad_log_policy_avg

	def grad_rat_policy_batch(self, y, x, z):
		'''gradient of logistic regression
			grad log p(y|x, z)
			y: time x obs x 1
			x: time x obs x dim
			z: time x particles x dim
		'''
		prob = self.rat_policy_batch(x, z) # T x obs
		# assert prob.size(0) == x.size(0)
		# assert prob.size(1) == x.size(1)
		# this is the gradient: weight error by input features
		error = (y - prob) # time x particles x obs
		# x is time x 1 x obs x dim, error is timex particles x obs x 1
		grad_log_policy = x[:, None, :, :] * error[:, :, :, None]
		return grad_log_policy

		# assert error.size(0) == x.size(0)
		# assert error.size(1) == x.size(1)
		# grad_log_policy = x * error[:, :, None] # T x num samples x dimension
		# assert grad_log_policy.size(0) == x.size(0)
		# assert grad_log_policy.size(1) == x.size(1)
		# assert grad_log_policy.size(2) == x.size(2)
		# return grad_log_policy
		# grad_log_policy_avg = torch.mean(grad_log_policy, dim=0)[None] # 1 x dimension
		# assert grad_log_policy_avg.size(-1) == z.size(-1)
		# return grad_log_policy_avg

	def log_likelihood_t_test(self, y, x, z):
		'''
			p(y_t | y_1:t-1, x_1:t, z_1:t)
		'''
		x_t = x[-1]
		z_t = z[-1][None, :]
		y_t = y[-1][:, None]
		logits = torch.matmul(x_t, torch.t(z_t))
		#logits = torch.dot(x_t, z_t)
		obs = Bernoulli(self.sigmoid(logits))
		logprobs = obs.log_prob(y_t)
		return logprobs

#####################################################################################################

	def log_likelihood_t(self, y, x, z):
		'''
			p(y_t | y_1:t-1, x_1:t, z_1:t)
		'''
		x_t = x[-1]
		z_t = z[-1][None, :]
		y_t = y[-1][:, None]
		logits = torch.matmul(x_t, torch.t(z_t))
		#logits = torch.dot(x_t, z_t)
		obs = Bernoulli(self.sigmoid(logits))
		logprobs = obs.log_prob(y_t)
		assert logprobs.size(0) == x_t.size(0)
		assert logprobs.size(1) == 1
		return torch.sum(obs.log_prob(y_t))

	def complete_data_log_likelihood_bootstrap(self, y1, x1, z, z1, weights):
		'''vecotrized computation over particles and time.
			E_q[log p(y, x, z)]
			q is approximate posterior
			we want gradients with respect to model parameters theta (not does not depend on q)
			compute monte carlo approximation.


			**need to vectorize this so its operations on particles x time x num observations x dimension
			**start with removing time for loop
		'''
		num_particles = weights.size(0)
		expected_value = 0
		z = z.transpose(1, 0)
		log_prob_vec = self.log_joint_batch_bootstrap(y1, x1, z, z1)

		expected_value = torch.dot(log_prob_vec, weights)
		return expected_value

	def complete_data_log_likelihood(self, y, x, particles, weights):
		'''vecotrized computation over particles and time.
			E_q[log p(y, x, z)]
			q is approximate posterior
			we want gradients with respect to model parameters theta (not does not depend on q)
			compute monte carlo approximation.


			**need to vectorize this so its operations on particles x time x num observations x dimension
			**start with removing time for loop
		'''
		num_particles = weights.size(0)
		expected_value = 0
		particles = particles.transpose(1, 0)
		log_prob_vec = self.log_joint_batch(y, x, particles)

		expected_value = torch.dot(log_prob_vec, weights)
		return expected_value

	def complete_data_log_likelihood_serial(self, y, x, particles, weights):
		'''
			E_q[log p(y, x, z)]
			q is approximate posterior
			we want gradients with respect to model parameters theta (not does not depend on q)
			compute monte carlo approximation.


			**need to vectorize this so its operations on particles x time x num observations x dimension
			**start with removing time for loop
		'''
		num_particles = weights.size(0)
		expected_value = 0
		for i in range(num_particles):
			z = particles[:, i, :]
			log_prob = self.log_joint(y, x, z)
			expected_value += weights[i] * log_prob
		return expected_value

	def logjoint_t(self, y, x, z):
		'''
		z is time x particles x dimension
		'''
		# transpose z to be particles x time x dimension--particles is "batch" dim
		z = z.transpose(0, 1) 
		z_t = z[:, -1]
		z_prev = z[:, :-1]

		if z_prev.nelement() == 0:
			log_prior = self.log_init_prior_batch(z_t)
		else:
			log_prior = self.log_prior_t_batch(z, y, x).squeeze(dim=0)
		log_lh = self.log_likelihood_t_batch(y, x, z)
		return log_lh, log_prior

		# z_t = z[-1]
		# z_prev = z[:-1]
		# if z_prev.nelement() == 0:
		# 	log_prior = self.log_init_prior(z_t)
		# else:
		# 	log_prior = self.log_prior_t(z, y, x)
		
		# log_lh = self.log_likelihood_t(y, x, z)
		# return log_lh.unsqueeze(dim=0), log_prior.unsqueeze(dim=0)

	def logjoint_t_serial(self, y, x, z):
		z_t = z[-1]
		z_prev = z[:-1]
		if z_prev.nelement() == 0:
			log_prior = self.log_init_prior(z_t)
		else:
			log_prior = self.log_prior_t(z, y, x)
		
		log_lh = self.log_likelihood_t(y, x, z)
		return log_lh.unsqueeze(dim=0), log_prior.unsqueeze(dim=0)

	def log_prior_t_batch(self, z, y, x):
		'''
			input: z_1:t
			parameterize p(z_t | z_t-1, theta)
		'''
		z_t = z[:, -1] # particles x dimension
		z_prev = z[:, :-1]
		y_prev =  y[-2][:, None]
		x_prev = x[-2]#[None]
		grad_rat_obj = self.grad_rat_obj_score_batch(y_prev, x_prev, z_prev[:, -1])
		# 1 x particles x dim
		learning = torch.exp(self.log_alpha) * grad_rat_obj

		penalty = self.sigmoid(self.beta)
		# particles x dimension
		regularization = penalty * z_prev[:, -1] + (1.0 - penalty) * -torch.sign(z_prev[:, -1])
		mean = learning + regularization[None]
		scale = torch.exp(self.transition_log_scale)
		prior = Normal(mean, scale)
		return torch.sum(prior.log_prob(z_t), dim=-1)


	def log_prior_batch_compl_bootstrap(self, z, y1, x1, z1):
		'''
			input: z_1:t
			parameterize p(z_t | z_t-1, theta)
		'''
		z = z.transpose(1,0)
		grad_rat_obj = self.grad_rat_obj_score_batch(y1, x1, z1)
		learning = torch.exp(self.log_alpha) * grad_rat_obj

		penalty = self.sigmoid(self.beta)
		#regularization = penalty * z_prev[:, -1] + (1.0 - penalty) * -torch.sign(z_prev[:, -1])
		regularization = (penalty * z1 + (1.0 - penalty) * -torch.sign(z1))
		mean = learning + regularization
		scale = torch.exp(self.transition_log_scale)
		prior = Normal(mean, scale)
		log_prob = prior.log_prob(z)
		# sum over dimension
		log_prob = torch.sum(log_prob, dim=(0, 2)) # sum over time and dimension
		return log_prob

	def log_prior_batch_compl(self, z, y, x):
		'''
			input: z_1:t
			parameterize p(z_t | z_t-1, theta)
		'''
		# time points 1:T
		z_curr = z[:, 1:]
		# time points 0:T-1
		z_prev = z[:, 0:-1]
		x_prev = x[:, 0:-1]
		z = z.transpose(1,0)
		grad_rat_obj = self.grad_rat_obj_score_batch(y, x, z)[0:-1]
		learning = torch.exp(self.log_alpha) * grad_rat_obj

		penalty = self.sigmoid(self.beta)
		#regularization = penalty * z_prev[:, -1] + (1.0 - penalty) * -torch.sign(z_prev[:, -1])
		regularization = (penalty * z + (1.0 - penalty) * -torch.sign(z))[0:-1]
		mean = learning + regularization
		scale = torch.exp(self.transition_log_scale)
		prior = Normal(mean, scale)
		log_prob = prior.log_prob(z[1:])
		# sum over dimension
		log_prob = torch.sum(log_prob, dim=(0, 2)) # sum over time and dimension
		return log_prob
	
	def log_prior_t(self, z, y, x):
		'''
			input: z_1:t
			parameterize p(z_t | z_t-1, theta)
		'''
		z_t = z[-1][None]
		z_prev = z[:-1]#[None]
		y_prev =  y[-2][:, None]
		x_prev = x[-2]#[None]

		grad_rat_obj = self.grad_rat_obj_score(y_prev, x_prev, z_prev[-1][None])
		
		# l2 = self.sigmoid(self.beta) * z_prev[-1]
		learning = torch.exp(self.log_alpha) * grad_rat_obj
		# l1 = torch.sign(z_prev[-1]) * torch.exp(self.log_sparsity)
		#sparsity = self.compute_sparsity(z_prev[-1])

		penalty = self.sigmoid(self.beta)
		regularization = penalty * z_prev[-1] + (1.0 - penalty) * -torch.sign(z_prev[-1])
		mean = learning + regularization

		# mean = l2 + learning - l1
		scale = torch.exp(self.transition_log_scale)

		prior = Normal(mean, scale)
		return torch.sum(prior.log_prob(z_t))

	def log_prior_t_test(self, z, y, x):
		'''
			input: z_1:t
			parameterize p(z_t | z_t-1, theta)
		'''
		z_t = z[-1][None]
		z_prev = z[:-1]#[None]
		y_prev =  y[-2][:, None]
		x_prev = x[-2]#[None]

		grad_rat_obj = self.grad_rat_obj_score(y_prev, x_prev, z_prev[-1][None])

		mean = self.sigmoid(self.beta) * z_prev[-1] + torch.exp(self.log_alpha) * grad_rat_obj
		scale = torch.exp(self.transition_log_scale)

		prior = Normal(mean, scale)
		return torch.sum(prior.log_prob(z_t))#, mean, grad_rat_obj


	def sample_prior(self, z_prev, y_prev=None, x_prev=None):
		'''sample from p(z_t | z_t-1, y_t-1, x_t-1)

		z_t+1 = beta * z_t + alpha * grad_rat_obj - sgn(z_t)*C
		'''
		# add learning component
		grad_rat_obj = self.grad_rat_obj_score(y_prev, x_prev, z_prev)
		
		penalty = self.sigmoid(self.beta)
		regularization = penalty * z_prev + (1.0 - penalty) * -torch.sign(z_prev)
		# l2 = self.sigmoid(self.beta) * z_prev
		learning = torch.exp(self.log_alpha) * grad_rat_obj
		# l1 = torch.sign(z_prev) * torch.exp(self.log_sparsity)
		# sparsity = self.compute_sparsity(z_prev)
		mean = learning + regularization
		# mean = l2 + learning - l1
		scale = torch.exp(self.transition_log_scale)
		prior = Normal(mean, scale)
		return prior.sample()

	def log_init_prior(self, z):
		'''evaluate log pdf of z0 under the init prior
		'''
		prior = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale)) 
		return torch.sum(prior.log_prob(z))

	def log_init_prior_batch(self, z):
		'''evaluate log pdf of z0 under the init prior
		z0 is particles x dimension
		return log probs for each particle
		'''
		prior = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale)) 
		return torch.sum(prior.log_prob(z), dim=-1)

	def sample_init_prior(self):
		prior = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
		return prior.sample()
	
	def log_likelihood_t(self, y, x, z):
		'''
			p(y_t | y_1:t-1, x_1:t, z_1:t)
		'''
		x_t = x[-1]
		z_t = z[-1][None, :]
		y_t = y[-1][:, None]
		logits = torch.matmul(x_t, torch.t(z_t))
		#logits = torch.dot(x_t, z_t)
		obs = Bernoulli(self.sigmoid(logits))
		logprobs = obs.log_prob(y_t)
		assert logprobs.size(0) == x_t.size(0)
		assert logprobs.size(1) == 1
		return torch.sum(obs.log_prob(y_t))

	def log_likelihood_compl_batch(self, y, x, z):
		'''log likelihood summed over time batch across particles
			p(y_t | y_1:t-1, x_1:t, z_1:t)
		'''
		z = z.transpose(1,0) # time x particles x dim
		x = x.transpose(2, 1)
		logits = torch.bmm(z, x) # time x particles x obs
		logits = logits.transpose(1, 0) # particles x time x obs
		obs = Bernoulli(self.sigmoid(logits))
		return torch.sum(obs.log_prob(y), dim=(1, 2))


		# print 'lh compl batch'
		# embed()
		# x_t = x[-1]
		# z_t = z[:, -1] # particles x dimension
		# y_t = y[-1][None, :]
		# logits = torch.t(torch.matmul(x_t, torch.t(z_t))) # batch x dimension
		# obs = Bernoulli(self.sigmoid(logits))
		# logprobs = obs.log_prob(y_t)

		# return torch.sum(logprobs, dim=-1)

	def log_likelihood_t_batch(self, y, x, z):
		'''
			p(y_t | y_1:t-1, x_1:t, z_1:t)
		'''
		x_t = x[-1]
		z_t = z[:, -1] # particles x dimension
		y_t = y[-1][None, :]
		logits = torch.t(torch.matmul(x_t, torch.t(z_t))) # batch x dimension
		obs = Bernoulli(self.sigmoid(logits))
		logprobs = obs.log_prob(y_t)

		return torch.sum(logprobs, dim=-1)

	def sample_likelihood(self, x_t, z_t, num_obs_samples):
		''' z_t is 1 x D
			x_t 

			x_t is num_samples x dimension
		'''
		#logits = torch.matmul(z_t, x_t).flatten()
		logits = torch.matmul(x_t, torch.t(z_t))
		obs = Bernoulli(self.sigmoid(logits))
		return obs.sample()

	def rat_reward(self, action, x):
		'''
		rat's reward func
		action is {0,1}
		x is T x 2 
		assume this is preprocessing. we compute rewards ahead of time.
		'''
		stim1 = x[:, 1].unsqueeze(dim=1)
		stim2 = x[:, 2].unsqueeze(dim=1)
		rewards = ((stim1 > stim2)*(action==1) + (stim1 < stim2)*(action==0))
		return rewards.float()

	def rat_policy(self, x, z):
		'''
		p(y = 1 | x, z)
			z is 1 x 3
			x is num samples x dimension

			x is time x num samples x dimension
			z = time x dimension
		'''
		prob = self.sigmoid(torch.sum(x * z[:, None, :], dim=2))
		prob = torch.t(prob)
		assert prob.size(0) == x.size(0)
		assert prob.size(1) == 1
	   
		return prob

	def grad_rat_policy(self, y, x, z):
		'''gradient of logistic regression
			grad log p(y|x, z)
		'''
		prob = self.rat_policy(x, z)
		assert prob.size(0) == x.size(0)
		assert prob.size(1) == 1
		# this is the gradient: weight error by input features
		error = (y - prob)
		assert error.size(0) == x.size(0)
		assert error.size(1) == 1
		grad_log_policy = x * error # T x num samples x dimension
		assert grad_log_policy.size(0) == x.size(0)
		assert grad_log_policy.size(1) == x.size(1)
		return grad_log_policy
		# grad_log_policy_avg = torch.mean(grad_log_policy, dim=0)[None] # 1 x dimension
		# assert grad_log_policy_avg.size(-1) == z.size(-1)
		# return grad_log_policy_avg

	def rat_obj_func(self, x, z):
		'''expectation_policy [r]
			
			p(y=1|x,z) * r(y=1, x) + p(y=0|x,z)*r(y=0, x)
		'''
		prob_y_1_given_x_z = self.rat_policy(x, z)
		prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

		# r(action=1, x)
		r_y_1_x = self.rat_reward(torch.tensor([1], device=device), x)
		r_y_0_x = 1.0 - r_y_1_x

		return prob_y_1_given_x_z * r_y_1_x + prob_y_0_given_x_z * r_y_0_x

	def grad_rat_obj_score(self, y, x, z):
		'''
			grad log p(y|x, z)
			grad rat objective function using score function estimator
		'''
		prob_y_1_given_x_z = self.rat_policy(x, z)
		prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

		# r(action=1, x)
		r_y_1_x = self.rat_reward(torch.tensor([1.], device=device), x)
		r_y_0_x = 1.0 - r_y_1_x

		# grad of logistic regression: x_n(y_n - sigmoid(z^t x))
		y_1 = torch.ones(y.size(0), 1, device=device)
		grad_log_policy_y1 = self.grad_rat_policy(y_1, x, z)
		y_0 = torch.zeros(y.size(0), 1)
		grad_log_policy_y0 = self.grad_rat_policy(y_0, x, z)
		per_sample_gradient = prob_y_1_given_x_z * grad_log_policy_y1 * r_y_1_x + \
			prob_y_0_given_x_z * grad_log_policy_y0 * r_y_0_x

		avg_gradient = torch.mean(per_sample_gradient, dim=0)[None]

		return avg_gradient

	def grad_warping(self):
		'''takes in gradient and warps through a nonlinear function'''
		return 1


def print_memory():
	print("memory usage: ", (process.memory_info().rss)/(1e9))

