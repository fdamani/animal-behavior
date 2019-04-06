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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython import display, embed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD, Adam
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        #self.sigmoid = nn.LogSigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        #output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size, device=device)


class LearningDynamicsModel(object):
    def __init__(self,
                 init_prior=(0.0, 1.0),
                 transition_log_scale=math.log(0.01),
                 dim=3, 
                 grad=False):
        # initialize parameters

        isGrad=True
        self.init_latent_loc = torch.tensor([init_prior[0]], 
            requires_grad=False, device=device)
        self.init_latent_log_scale = torch.tensor([init_prior[1]], 
            requires_grad=False, device=device)
        self.transition_log_scale = torch.tensor([transition_log_scale], 
            requires_grad=isGrad, device=device)
        self.sigmoid = nn.Sigmoid()


    def sample(self, T, num_obs_samples=10, dim=3, x=None):
        '''
            sample latent variables and observations
        '''
        # generate 1D x from standard normal
        intercept = torch.ones(T, num_obs_samples, 1, device=device)
        if x is None: 
            x = torch.randn(T, num_obs_samples, dim-1, device=device)
            x = torch.cat([intercept, x], dim=2)
        
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

    def sample_prior(self, z_prev, y_prev=None, x_prev=None):
        '''sample from p(z_t | z_t-1, y_t-1, x_t-1)
        simple AR-1 prior

        z_t+1 = beta * z_t + alpha * grad_rat_obj - sgn(z_t)*C
        '''
        mean = z_prev
        scale = torch.exp(self.transition_log_scale)
        prior = Normal(mean, scale)
        return prior.sample()
    
    def sample_init_prior(self):
        prior = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        return prior.sample()
    
    def sample_likelihood(self, x_t, z_t, num_obs_samples):
        ''' z_t is 1 x D
            x_t 

            x_t is num_samples x dimension
        '''
        #logits = torch.matmul(z_t, x_t).flatten()
        logits = torch.matmul(x_t, torch.t(z_t))
        obs = Bernoulli(self.sigmoid(logits))
        return obs.sample()

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

    def return_train_ind(self, y):
        return y[:,0] != -1


    def log_likelihood_vec(self, y, x, z):
        '''
            p(y_t | y_1:t-1, x_1:t, z_1:t)
            y will contain -1's denoting unobserved data
            only compute log probs for observed y's.
            identify indices where y does not equal -1
            compute log probs for those and then sum accordingly.
        '''
        logits = torch.sum(x * z[:, None, :], dim=2)
        train_inds = self.return_train_ind(y)
        logits_train = logits[train_inds]
        # limit logits to observed y's
        obs = Bernoulli(logits=logits_train)
        return torch.sum(obs.log_prob(y[train_inds]))

    def log_prior_vec(self, z, y, x):
        '''
            input: z_1:t
            parameterize p(z_t | z_t-1, theta)
        '''
        z_prev = z[0:-1]
        z_curr = z[1:]

        # properly vectorized
        mean = z_prev

        scale = torch.exp(self.transition_log_scale)
        prior = Normal(mean, scale)
        return torch.sum(prior.log_prob(z_curr))

    def log_likelihood_test(self, y_train, y_test, test_inds, x, z):
        logits = torch.sum(x * z[:, None, :], dim=2)
        train_inds = self.return_train_ind(y_train)
        logits_train = logits[train_inds]
        obs = Bernoulli(logits=logits_train)
        train_ll = torch.sum(obs.log_prob(y_train[train_inds]))
        logits_test = logits[test_inds]
        obs = Bernoulli(logits=logits_test)
        test_ll = torch.sum(obs.log_prob(y_test))
        train_probs = self.sigmoid(logits_train)
        test_probs = self.sigmoid(logits_test)
        return train_ll, test_ll, train_probs, test_probs

    def sample_forward(self, y_train, y_test, test_inds, x, z, x_future, num_obs, num_future_steps):
        '''
            sample forward from last time point of z.
        '''
        z_future = []
        y_future = []
        for i in range(num_future_steps):
            y_prev = y_train[-1] if y_train[-1][0] != -1 else y_test[-1]
            z_i = self.sample_prior(z[-1], y_prev, x[-1])
            y_i = self.sample_likelihood(x[i], z_i, num_obs)

            z_future.append(z_i)
            y_future.append(y_i)
        
        y_sampled_future = torch.t(torch.cat(y_future, dim=1))
        z_future = torch.cat(z_future)

        return y_sampled_future, z_future


def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

