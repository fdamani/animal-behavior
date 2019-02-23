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
import psutil
import learning_dynamics
from learning_dynamics import LearningDynamicsModel
import smc
from smc import SMCOpt

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
    alpha = model_params[:, 1].flatten()
    transition_log_scale = model_params[:, 2].flatten()
    plt.cla()
    plt.plot(beta)
    plt.savefig(savedir+'/beta.png')

    plt.cla()
    plt.plot(alpha)
    plt.savefig(savedir+'/alpha.png')

    plt.cla()
    plt.plot(transition_log_scale)
    plt.savefig(savedir+'/transition_log_scale.png')


def save(loss_l, particles_l, weights_l, mean_l, scale_l, model_params, savedir):
    np.save(savedir+'/loss_l.npy', loss_l)
    np.save(savedir+'/particles_l.npy', particles_l)
    np.save(savedir+'/weights_l.npy', weights_l)
    np.save(savedir+'/mean_l.npy', mean_l)
    np.save(savedir+'/scale_l.npy', scale_l)
    np.save(savedir+'/model_params.npy', model_params)

class EM(object):
    def __init__(self, data, savedir):
        self.savedir = savedir
        self.data = data
        self.dim = self.data[1].size(2)
        self.T = self.data[1].size(0)
        self.em_iters = 1
        global_params.append('em iters: ' + str(self.em_iters))

        self.model = None
        self.init_model()
        print 'initialized model...'

        self.e_step = E_Step(self.model,
                             self.T)

        print 'initialized e-step...'
        self.m_step = M_Step(self.model)
        print 'initialized m-step...'

    def init_model(self,
                   init_prior_loc = 0.0,
                   init_prior_log_scale = 0.0,
                   transition_log_scale = math.log(1e-1),
                   beta = 4.0,
                   log_alpha = math.log(1e-3)):
        global_params.append('init_prior_loc: ' + str(init_prior_loc))
        global_params.append('init_prior_log_scale: ' + str(init_prior_log_scale))
        global_params.append('init_transition_log_scale: ' + str(transition_log_scale))
        global_params.append('init_beta: ' + str(beta))
        global_params.append('init_log_alpha: ' + str(log_alpha))

        init_prior = ([init_prior_loc]*self.dim, [init_prior_log_scale]*self.dim)
        transition_log_scale = [transition_log_scale]#*self.dim
        # log_alpha = [log_alpha] * self.dim
        self.model = LearningDynamicsModel(init_prior=init_prior, 
                                           transition_log_scale=transition_log_scale, 
                                           beta=beta,
                                           log_alpha=log_alpha, 
                                           dim=self.dim)

    def update_model(self, opt_params):
        self.model.beta = opt_params[0]
        self.model.log_alpha = opt_params[1]
        self.model.transition_log_scale = opt_params[2]

    def optimize(self):
        self.init_model()
        print 'optimizing...'
        loss_l, particles_l, weights_l, mean_l, scale_l, model_params = [], [], [], [], [], []
        for i in range(self.em_iters):
            # e-step
            particles, weights, mean, scale, marginal_ll = self.e_step.forward(self.data, self.model)

            # plot
            plot(mean, scale, self.savedir)

            # m-step
            opt_params, expected_likelihood = self.m_step.optimize(self.data, particles, weights)

            # update model parameters
            self.update_model(opt_params)

            # append new params to lists
            loss_l.append(expected_likelihood.detach().cpu().numpy())
            model_params.append([sx.detach().cpu().numpy() for sx in opt_params])
            particles_l.append(particles.detach().cpu().numpy()), weights_l.append(weights.detach().cpu().numpy())
            mean_l.append(mean.detach().cpu().numpy()), scale_l.append(scale.detach().cpu().numpy())

            # print and plot
            print 'EM iter: ', i, '\n'
            plot(mean, scale, self.savedir)
            plot_loss(loss_l, self.savedir)
            plot_model_params(np.array(model_params), self.savedir)

        # save
        save(np.array(loss_l), np.array(particles_l), np.array(weights_l), np.array(mean_l), 
            np.array(scale_l), model_params, self.savedir)

        outfile = open(self.savedir+'/params.txt', 'wb')
        outfile.write("\n".join(global_params))

class E_Step(object):
    '''
        posterior over latents
    '''
    def __init__(self, 
                 model,
                 T,
                 num_particles=100):

        self.model = model
        self.inference = None
        self.T = T
        self.num_particles = num_particles
        self.init_smc()
        global_params.append('num_particles: ' + str(num_particles))

    def init_smc(self):
        # set smc proposal params to model prior
        self.proposal_params = [self.model.init_latent_loc.detach(), 
                                self.model.init_latent_log_scale.detach(),
                                self.model.transition_log_scale.detach()]
        self.inference = SMCOpt(self.model,
                                proposal_params=self.proposal_params,
                                num_particles=self.num_particles,
                                T=self.T)

    def forward(self, data, model):
        self.model = model
        # update smc params
        self.init_smc()

        # e-step
        marginal_ll = self.inference.forward(data)
        mean, scale = self.inference.estimate(data)
        log_weights = self.inference.weights[-1]
        weights = torch.exp(log_weights)
        particles = self.inference.particles_list[-1]

        return particles, weights, mean, scale, marginal_ll

class M_Step(object):
    def __init__(self, 
                 model,
                 lr = 0.001):
        self.model = model
        self.opt_params = [self.model.beta, 
                           self.model.log_alpha, 
                           self.model.transition_log_scale]

        self.optimizer = torch.optim.Adam(self.opt_params, lr = lr)
        self.num_iters = 1000
        global_params.append('adam learning rate: ' + str(lr))
        global_params.append('m step num iters: ', + str(self.num_iters))

    def unpack_params(self, params):
        return params[0]
    def forward(self, y, x, particles, weights):
        return self.model.complete_data_log_likelihood(y, x, particles, weights)

    def optimize(self, data, particles, weights):
        y, x = self.model.unpack_data(data)
        # m-step
        outputs = []
        for t in range(self.num_iters):
            self.optimizer.zero_grad()
            output = -self.forward(y, x, particles, weights)#, opt_params)
            output.backward(retain_graph=True)
            if t % 100 == 0:
                print 'iter: ', t, \
                      'loss: ', output.item(), \
                      'sparsity: ', self.model.beta.detach().item(), \
                      'alpha: ', self.model.log_alpha.detach().item(), \
                      'scale: ', self.model.transition_log_scale.detach().item()
            self.optimizer.step()
            outputs.append(output)

        return self.opt_params, outputs[-1]

class Map(object):
    def __init__(self, model):
        self.model = model
    def unpack_params(self, params):
        return params[0]
    def forward(self, x, params):
        params = self.unpack_params(params)
        return self.model.logjoint(x, params)

class MeanFieldVI(object):
    '''
    Mean field fully factorized variational inference.
    '''
    def __init__(self, model, num_samples=1):
        self.model = model
        self.num_samples = num_samples

    def unpack_var_params(self, params):
        loc, log_scale = params[0], params[1]
        return loc, log_scale

    def forward(self, x, var_params):
        '''
            useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
        '''
        loc, log_scale = self.unpack_var_params(var_params)
        cov = torch.diag(torch.exp(log_scale))**2
        scale_tril = cov.tril()
        var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
        samples = var_dist.rsample(torch.Size((self.num_samples,)))
        data_terms = torch.empty(self.num_samples, device=device)
        for i in range(len(samples)):
            data_terms[i] = self.model.logjoint(x, samples[i])
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

class FIVO(object):
    '''
    Filtered variational objectives. 
    IWAE + smc
    should inherit an SMC object with resampling, etc
    '''
    def __init__(self):
        self.model = None
        self.num_particles = 0

    def initialize(self, model, num_particle=10):
        self.model = model
        self.num_particles = num_particles

    def unpack_var_params(self):
        return 1

    def forward(self):
        return 1

def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

