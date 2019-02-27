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
import utils
from utils import get_gpu_memory_map
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# we want to bootstrap estimate model parameters on the expected complete data likelihood

# randomly sample with replacement the z's
# for each z_t sampled we want log probs for that z_t given its actual z_t-1, y_t-1, x_t-1
# create a dataset that is z_t, z_t-1, y_t-1, x_t-1
# randomly sample rows.

# load model parameters
f = '/tigress/fdamani/neuro_output/diffusion_l_s/'
files = os.listdir(f)
f += files[int(sys.argv[1])] + '/'

print f
#f = '/tigress/fdamani/neuro_output/2019-02-23 16:41:23.358163__obs75__W066'
model_params = np.load(f+'model_params.npy')
weights = np.load(f+'weights_l.npy')
particles = np.load(f+'particles_l.npy')
x = np.load(f+'/data/x.npy')
y = np.load(f+'/data/y.npy')

init_prior = (0.0, 1.0)
beta = model_params[-1][0]
log_alpha = model_params[-1][1]
transition_log_scale = model_params[-1][2]
log_gamma = model_params[-1][3]
#log_gamma = np.array([math.log(1e-10)], dtype=np.float32)
# log_gamma = np.array([math.log(1e-10)], dtype=np.float32)
#beta = np.array([0.0], dtype=np.float32)
#log_alpha = np.array([math.log(1e-10)], dtype=np.float32)

x = torch.tensor(x, device=device)
y = torch.tensor(y, device=device)
particles = torch.tensor(particles[-1], device=device)
weights = torch.tensor(weights[-1], device=device)
dim = x.size(-1)

def bootstrap(x, y, z):
    '''residual bootstrap estimator for AR1 process
    s_t = z_t - z_t-1

    # sample w/ replacement numbers from 1-T T times.
    # our z_ts are z[inds]
    # we want y_t-1s for each z_t so y[inds-1]
    # we want x_t-1s for each x_t so x[inds-1]
    # we also want our z_t-s which is z[inds-1]
    '''
    datasets = []
    num_datasets = 100
    t = x.size(0)
    a = np.arange(1, t)
    for i in range(num_datasets):
        inds = np.random.choice(a, size=t)
        x1 = x[inds-1]
        y1 = y[inds-1]
        z1 = z[inds-1]
        z_true = z[inds]
        datasets.append((y1, x1, z_true, z1))
    return datasets

datasets = bootstrap(x, y, particles)
og = [(y[0:-1], x[0:-1], particles[1:], particles[0:-1])]

bootstrapped_params = []

for data in datasets:
    y1, x1, z, z1 = data
    model = LearningDynamicsModel(init_prior=init_prior, 
                                       transition_log_scale=transition_log_scale, 
                                       beta=beta,
                                       log_alpha=log_alpha,
                                       log_gamma = log_gamma, 
                                       dim=dim, grad=False)
    model.init_grad_vbles()
    lr = 5e-2
    opt_params = [model.beta, 
                  model.log_alpha, 
                  model.transition_log_scale,
                  model.log_gamma]

    optimizer = torch.optim.Adam(opt_params, lr = lr)
    num_iters = 3000
    outputs = []
    for t in range(num_iters):
        output = -model.complete_data_log_likelihood_bootstrap(y1, x1, z, z1, weights)
        # compute loss
        outputs.append(output.item())
        # zero all of the gradients
        optimizer.zero_grad()
        # backward pass: compute gradient of loss w.r.t. to model parameters
        output.backward()
        # call step function
        optimizer.step()
        if t % 500 == 0:
            print 'iter: ', t, \
                  'loss: ', output.item(), \
                  'sparsity: ', model.beta.item(), \
                  'alpha: ', model.log_alpha.item(), \
                  'scale: ', model.transition_log_scale.item(), \
                  'regularization: ', model.log_gamma.item()
    model.init_no_grad_vbles()

    bootstrapped_params.append((model.beta.item(), model.log_alpha.item(), 
        model.transition_log_scale.item(), model.log_gamma.item()))
np.save(f+'bootstrap_params.npy', bootstrapped_params)