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

import inference
from inference import EM

import psutil
import learning_dynamics
from learning_dynamics import LearningDynamicsModel
import sim
from sim import generateSim 
import read_data
from read_data import read_and_process
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(10)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32

def to_numpy(tx):
    return tx.detach().cpu().numpy()

if __name__ == '__main__':

    grad_latents = True
    grad_model_params = False

    inference_types = ['map', 'mfvi', 'is', 'smc', 'vsmc']
    inference_type = inference_types[4]

    sim = False
    if sim:
        # T = 200 # 100
        T = 100
        num_particles = 100# 200
        # time-series model
        # sim model parameters
        dim = 3
        init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
        transition_scale = [math.log(.01)] * dim
        log_sparsity = math.log(1e-2)
        embed()
        beta = 4.5 # sigmoid(4.) = .9820
        log_alpha = math.log(1e-3)
        model = LearningDynamicsModel(init_prior, transition_scale, beta, log_alpha, dim=3, log_sparsity=log_sparsity)
        #model = LogReg_LDS(init_prior=(0.0, 0.02), transition_scale=1e-3)
        num_obs_samples = 250
        y, x, z_true = model.sample(T=T, num_obs_samples=num_obs_samples)

        # plt.plot(to_numpy(z_true))
        # plt.show()
        # model params
    else:
        x, y, rw = read_and_process(num_obs=25)
        #x = torch.tensor(x, device=device)
        x = torch.tensor(x, dtype=dtype, device=device)
        y = torch.tensor(y, dtype=dtype, device=device)
        rw = torch.tensor(rw, dtype=dtype, device=device)
    #plt.plot(to_numpy(z_true))
    #plt.show()
    #x = x[:, :, 0:3]
    data = [y, x]
    sx = EM(data)
    sx.optimize()

  