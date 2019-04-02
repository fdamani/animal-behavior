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
import learning_dynamics
from learning_dynamics import LearningDynamicsModel
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

global_params = []


def to_numpy(tx):
    return tx.detach().cpu().numpy()

def plot(mean, scale, savedir):

    plt.cla()
    # plot means
    plt.plot(to_numpy(mean)[:,0], label='bias')
    plt.plot(to_numpy(mean)[:,1], label='x1')
    plt.plot(to_numpy(mean)[:,2], label='x2')
    plt.plot(to_numpy(mean)[:,3], label='choice hist')
    plt.plot(to_numpy(mean)[:,4], label='rw side hist')
    plt.plot(to_numpy(mean)[:,5], label='sensory hist1')
    plt.plot(to_numpy(mean)[:,6], label='senory hist2')

    # plot scales
    # plt.fill_between(np.arange(mean[:,0].size(0)), to_numpy(mean)[:,0] - to_numpy(scale)[:,0], 
    #    to_numpy(mean)[:,0] + to_numpy(scale)[:,0], alpha=.5)

    plt.legend(loc = 'upper left')
    plt.savefig(savedir+'/latent_trajectory.png')

def plot_loss(loss, savedir):
    plt.cla()
    plt.plot(np.array(loss).flatten())
    plt.savefig(savedir+'/loss.png')

def plot_model_params(model_params, savedir):
    beta = model_params[:, 0].flatten()
    alpha = model_params[:, 0].flatten()
    transition_log_scale = model_params[:, 1].flatten()
    # log_gamma = model_params[:, 3].flatten()

    # plt.cla()
    # plt.plot(beta)
    # plt.savefig(savedir+'/beta.png')

    plt.cla()
    plt.plot(alpha)
    plt.savefig(savedir+'/alpha.png')

    plt.cla()
    plt.plot(transition_log_scale)
    plt.savefig(savedir+'/transition_log_scale.png')

    # plt.cla()
    # plt.plot(log_gamma)
    # plt.savefig(savedir+'/log_gamma.png')

def save(loss_l, particles_l, weights_l, mean_l, scale_l, model_params, savedir):
    np.save(savedir+'/loss_l.npy', loss_l)
    np.save(savedir+'/particles_l.npy', particles_l)
    np.save(savedir+'/weights_l.npy', weights_l)
    np.save(savedir+'/mean_l.npy', mean_l)
    np.save(savedir+'/scale_l.npy', scale_l)
    #np.save(savedir+'/model_params.npy', model_params)

class Map(object):
    def __init__(self, model):
        self.model = model
    def unpack_params(self, params):
        return params[0]
    def forward(self, x, params):
        params = self.unpack_params(params)
        return self.model.logjoint(x, params)


class Inference(object):
    def __init__(self, data, model, savedir, num_obs):
        self.data = data
        self.dim = self.data[1].size(2)
        self.T = self.data[1].size(0)
        self.model = model
        self.savedir = savedir
        self.num_obs = num_obs

        self.vi = MeanFieldVI(self.model, self.savedir)
        self.var_params = self.vi.init_var_params(self.T, self.dim)

        self.iters = 10000
        lr = 1e-2
        self.opt_params = self.var_params
        self.optimizer = torch.optim.Adam(self.opt_params, lr = lr)

    def optimize(self):
        #print 'gpu usage: ', torch.cuda.memory_allocated(device) /1e9
        #print 'cpu usage: ', print_memory()
        print 'optimizing...'
        #loss_l, weights_l, mean_l, scale_l, model_params = [], [], [], [], [], []
        #outfile = open(self.savedir+'/params.txt', 'wb')

        outputs = []
        for t in range(self.iters):
            # e-step
            output = -self.vi.forward(self.data, self.var_params)
            outputs.append(output.item())
            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()
            print 'iter: ', t, 'loss: ', output.item()
            if t == 8000:
                embed()
        zx = self.var_params[0]
        plt.plot(to_numpy(zx))
        plt.savefig('learned_z.png')
        return self.opt_params

class MeanFieldVI(object):
    '''
    Mean field fully factorized variational inference.
    '''
    def __init__(self, model, savedir, num_samples=1):
        self.model = model
        self.savedir = savedir
        self.num_samples = num_samples

    def init_var_params(self, T, dim):
        mean = torch.ones(T, dim, requires_grad=True, device=device)
        log_scale = torch.tensor(-5 * torch.ones(T, dim), requires_grad=True, device=device)
        return (mean, log_scale)

    def unpack_var_params(self, params):
        loc, log_scale = params[0], params[1]
        return loc, log_scale
    def unpack_data(self, data):
        y = data[0]
        x = data[1]
        return y, x

    def forward(self, data, var_params):
        '''
            useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
        '''
        y, x = self.unpack_data(data)
        loc, log_scale = self.unpack_var_params(var_params)
        var_dist = Normal(loc, torch.exp(log_scale))
        #cov = torch.diag(torch.exp(log_scale))**2
        #scale_tril = cov.tril()
        #var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
        samples = var_dist.rsample(torch.Size((self.num_samples,)))
        data_terms = torch.empty(self.num_samples, device=device)
        for i in range(len(samples)):
            data_terms[i] = self.model.log_joint(y, x, samples[i])
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


def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

