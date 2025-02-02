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
from torch.distributions import Normal, Bernoulli

import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def univar_normal(loc, scale):
    return Normal(loc, scale)
def bernoulli(param):
    return Bernoulli(param)

class LinearRegression(object):
    def __init__(self,
                 grad_model_params=False,
                 init_prior=(0.0, 0.1),
                 obs_scale=0.001,
                 num_samples=100):
        # initialize parameters
        self.prior_loc = torch.tensor([init_prior[0]], 
            requires_grad=grad_model_params, device=device)
        self.prior_log_scale = torch.tensor([math.log(init_prior[1])], 
            requires_grad=grad_model_params, device=device)
        self.obs_log_scale = torch.tensor([math.log(obs_scale)], 
            requires_grad=grad_model_params, device=device)

        self.num_samples = num_samples

    def return_model_params(self):
        '''returns list of model params'''
        return [self.init_latent_loc, self.init_latent_log_scale,
            self.transition_log_scale, self.obs_log_scale]

    def sample(self):
        '''
            sample latent variables and observations
        '''
        prior_z = Normal(self.prior_loc, torch.exp(self.prior_log_scale))
        z = prior_z.sample().reshape(-1,1)
        prior_x = Normal(self.prior_loc, torch.exp(self.prior_log_scale))
        X = prior_x.sample(torch.Size((self.num_samples,)))
        mean = torch.matmul(X, z)
        observation_model = Normal(mean, torch.exp(self.obs_log_scale))
        y = observation_model.sample()
        return X, y, z


    def logjoint(self, data, z):
        '''
        input: x (observations T x D)
        input: latent_mean
        return logpdf under the model parameters
        '''
        #transition_log_scale, obs_log_scale = model_params[0], model_params[1]
        #self.transition_scale = torch.exp(transition_log_scale)
        #self.obs_scale = torch.exp(obs_log_scale)
        x, y = data[0], data[1]
        prior = Normal(self.prior_loc, torch.exp(self.prior_log_scale))
        mean = torch.matmul(x, z).reshape(-1,1)
        likelihood = Normal(mean, torch.exp(self.obs_log_scale))
        logprob = torch.sum(prior.log_prob(z))
        logprob += torch.sum(likelihood.log_prob(y))
        return logprob


class LDS(object):
    def __init__(self,
                 grad_model_params=False, 
                 init_prior=(0.0, 0.1),
                 transition_scale=0.1,
                 obs_scale=0.1):
        # initialize parameters
        self.init_latent_loc = torch.tensor([init_prior[0]], 
            requires_grad=grad_model_params, device=device)
        self.init_latent_log_scale = torch.tensor([math.log(init_prior[1])], 
            requires_grad=grad_model_params, device=device)
        self.transition_log_scale = torch.tensor([math.log(transition_scale)], 
            requires_grad=grad_model_params, device=device)
        self.obs_log_scale = torch.tensor([math.log(obs_scale)], 
            requires_grad=grad_model_params, device=device)

    def return_model_params(self):
        '''returns list of model params'''
        return [self.init_latent_loc, self.init_latent_log_scale,
            self.transition_log_scale, self.obs_log_scale]

    def sample(self, T):
        '''
            sample latent variables and observations
        '''
        latents, obs = [], []
        log_prior_cpd = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        latents.append(log_prior_cpd.sample())
        obs_cpd = Normal(latents[0], torch.exp(self.obs_log_scale))
        obs.append(obs_cpd.sample())
        for i in range(1,T):
            transition_cpd = Normal(latents[i-1], torch.exp(self.transition_log_scale))
            latents.append(transition_cpd.sample())
            obs_cpd = Normal(latents[i], torch.exp(self.obs_log_scale))
            obs.append(obs_cpd.sample())
        return obs, latents

    def log_likelihood_t(self, x_t, x_past, z):
        '''
            generic likelihood:
            p(x_t | x_{1:t-1}, z_{1:t})
        '''
        obs_logpdf = Normal(z[-1], torch.exp(self.obs_log_scale))
        return obs_logpdf.log_prob(x_t)


    def init_log_prior(self, z_t):
        init_latent_logpdf = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        return init_latent_logpdf.log_prob(z_t)

    def log_prior_t(self, z_t, z_past, x_past):
        '''
            generic prior
            p(z_t | x_{1:t-1}, z_{1:t-1})
        '''
        transition_logpdf = Normal(z_past[-1], torch.exp(self.transition_log_scale))
        return transition_logpdf.log_prob(z_t)

    def logjoint_t(self, x, z):
        '''
            p(x_t, z_t | x_1:t-1, z_1:t-1)
        '''
        x_t = x[-1]
        x_past = x[:-1]
        z_t = z[-1]
        z_past = z[:-1]
        if z_past.nelement() == 0:
            log_prior = self.init_log_prior(z_t)
        else:
            log_prior = self.log_prior_t(z_t, z_past, x_past)
        log_lh = self.log_likelihood_t(x_t, x_past, z)
        return log_lh, log_prior
        #return log_lh + log_prior

    def logjoint(self, x, latent_mean):
        '''
        input: x (observations T x D)
        input: latent_mean
        return logpdf under the model parameters
        '''
        #transition_log_scale, obs_log_scale = model_params[0], model_params[1]
        #self.transition_scale = torch.exp(transition_log_scale)
        #self.obs_scale = torch.exp(obs_log_scale)

        T = x.size(0)
        # init log prior
        init_latent_logpdf = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        # transitions
        transition_logpdf = Normal(latent_mean[:-1], torch.exp(self.transition_log_scale))
        # observations
        obs_logpdf = Normal(latent_mean, torch.exp(self.obs_log_scale))

        # compute log probs
        logprob = init_latent_logpdf.log_prob(latent_mean[0])
        logprob += torch.sum(transition_logpdf.log_prob(latent_mean[1:]))
        logprob += torch.sum(obs_logpdf.log_prob(x))

        return logprob

    def log_marginal_likelihood(self, T, y_true):
        mu0 = self.init_latent_loc
        Sigma0 = torch.exp(self.init_latent_log_scale)
        A = np.array([[1.0]])
        Q = np.exp(self.transition_log_scale.detach().cpu().numpy())[:,None] ** 2
        C = np.array([[1.0]])
        R = np.exp(self.obs_log_scale.detach().cpu().numpy())[:,None] **2
        y_true = y_true.detach().cpu().numpy()[:,None]

        # mu0, Sigma0, A, Q, C, R = model_params
        Dx = mu0.shape[0]
        
        log_likelihood = 0.
        xfilt = np.zeros(Dx)
        Pfilt = np.zeros((Dx,Dx))
        xpred = mu0
        Ppred = Sigma0

        for t in range(T):
            if t > 0:
                # Predict
                xpred = np.dot(A,xfilt)
                Ppred = np.dot(A,np.dot(Pfilt,A.T)) + Q

            # Update
            yt = y_true[t,:] - np.dot(C,xpred)
            S = np.dot(C,np.dot(Ppred,C.T)) + R
            K = np.linalg.solve(S,np.dot(C,Ppred)).T
            xfilt = xpred + np.dot(K,yt)
            Pfilt = Ppred - np.dot(K,np.dot(C,Ppred))

            sign, logdet = np.linalg.slogdet(S)
            log_likelihood += -0.5*(np.sum(yt*np.linalg.solve(S,yt))+logdet+Dx*np.log(2.*np.pi))
            
        return log_likelihood

class LogReg_LDS(object):
    def __init__(self,
                init_prior=(0.0, 0.1),
                transition_scale=0.1):
        # initialize parameters
        grad_model_params=False
        self.init_latent_loc = torch.tensor([init_prior[0]], 
            requires_grad=grad_model_params, device=device)
        self.init_latent_log_scale = torch.tensor([math.log(init_prior[1])], 
            requires_grad=grad_model_params, device=device)
        self.transition_log_scale = torch.tensor([math.log(transition_scale)], 
            requires_grad=grad_model_params, device=device)
        self.sigmoid = nn.Sigmoid()

    def sample(self, T):
        '''
            sample latent variables and observations
        '''
        latents, obs = [], []
        log_prior_cpd = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        latents.append(log_prior_cpd.sample())
        obs_cpd = Bernoulli(self.sigmoid(latents[0]))
        obs.append(obs_cpd.sample())
        for i in range(1,T):
            transition_cpd = Normal(latents[i-1], torch.exp(self.transition_log_scale))
            latents.append(transition_cpd.sample())

            # pass latents[i] through a sigmoid
            obs_cpd = Bernoulli(self.sigmoid(latents[i]))
            obs.append(obs_cpd.sample())
        return obs, latents

    def logjoint(self, x, z, model_params):
        '''
        input: x (observations T x D)
        input: latent_mean
        return logpdf under the model parameters
        '''
        T = x.size(0)
        transition_log_scale = model_params[0]
        self.transition_scale = torch.exp(self.transition_log_scale)
        # init log prior
        init_latent_logpdf = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        # transitions
        transition_logpdf = Normal(z[:-1], torch.exp(self.transition_log_scale))
        # observations
        obs_logpdf = Bernoulli(self.sigmoid(z))

        # compute log probs
        logprob = init_latent_logpdf.log_prob(z[0])
        logprob += torch.sum(transition_logpdf.log_prob(z[1:]))
        logprob += torch.sum(obs_logpdf.log_prob(x))

        return logprob

    def log_likelihood_t(self, x_t, x_past, z):
        '''
            p(x_t | x_1:t-1, z_1:t)
        '''
        obs_logpdf = Bernoulli(self.sigmoid(z[-1]))
        return obs_logpdf.log_prob(x_t)

    def init_log_prior(self, z_t):
        init_latent_logpdf = Normal(self.init_latent_loc, torch.exp(self.init_latent_log_scale))
        return init_latent_logpdf.log_prob(z_t)

    def log_prior_t(self, z_t, z_past, x_past):
        '''
            p(z_t | x_1:t-1, z_1:t-1)
        '''
        transition_logpdf = Normal(z_past[-1], torch.exp(self.transition_log_scale))
        return transition_logpdf.log_prob(z_t)

    def logjoint_t(self, x, z):
        '''
            p(x_t, z_t | x_1:t-1, z_1:t-1)
        '''
        x_t = x[-1]
        x_past = x[:-1]
        z_t = z[-1]
        z_past = z[:-1]
        if z_past.nelement() == 0:
            log_prior = self.init_log_prior(z_t)
        else:
            log_prior = self.log_prior_t(z_t, z_past, x_past)
        log_lh = self.log_likelihood_t(x_t, x_past, z)
        return log_lh, log_prior
        #return log_lh + log_prior

def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

