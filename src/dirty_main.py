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
import models
from models import LDS, LogReg_LDS, LinearRegression
import inference
from inference import EM, Map, MeanFieldVI, StructuredVITriDiagonal
import smc
from smc import IS, SMC
import vsmc
from vsmc import VSMC
import psutil
import learning_dynamics
from learning_dynamics import LearningDynamicsModel
import sim
from sim import generateSim 
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(10)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
dtype = torch.float32

def to_numpy(tx):
    return tx.detach().cpu().numpy()

if __name__ == '__main__':

    grad_latents = True
    grad_model_params = False

    inference_types = ['map', 'mfvi', 'is', 'smc', 'vsmc']
    inference_type = inference_types[4]
    # T = 200 # 100
    T = 100
    num_particles = 20# 200
    # time-series model
    # sim model parameters
    dim = 3
    init_prior = ([0.0]*dim, [math.log(0.1)]*dim)
    transition_scale = [math.log(0.1)] * dim
    beta = 4. # sigmoid(4.) = .9820
    log_alpha = -1.
    model = LearningDynamicsModel(init_prior, transition_scale, beta, log_alpha, dim=3)
    #model = LogReg_LDS(init_prior=(0.0, 0.02), transition_scale=1e-3)
    num_obs_samples = 2
    y, x, z_true = model.sample(T=T, num_obs_samples=num_obs_samples)

    plt.plot(to_numpy(z_true))
    # plt.show()
    # model params
    init_prior = ([0.0]*dim, [math.log(0.1)]*dim)
    transition_scale = [math.log(0.1)] * dim
    model = LearningDynamicsModel(init_prior, transition_scale, beta, dim=3)
    # proposal params
    smc_init_prior = ([0.0]*dim, [math.log(0.1)]*dim)
    smc_transition_scale = [math.log(0.1)] * dim
    q_init_latent_loc = torch.tensor([smc_init_prior[0]], 
        requires_grad=False, device=device)
    q_init_latent_log_scale = torch.tensor([smc_init_prior[1]], 
        requires_grad=False, device=device)
    q_transition_log_scale = torch.tensor([smc_transition_scale], 
        requires_grad=False, device=device)

    variational_params = [q_init_latent_loc,
                          q_init_latent_log_scale,
                          q_transition_log_scale]
    inference = VSMC(model,
                     variational_params, 
                     num_particles=num_particles,
                     T=T)
    # dim = x.size(1)
    dim = len(x)
    # cast as torch tensors
    x = torch.tensor(x, dtype=dtype, device=device)
    data = [y, x]

    smcopt_marginal_ll = inference.forward(data)
    mean, scale = inference.smc.estimate(data)
    log_weights = inference.smc.weights[-1]
    weights = torch.exp(log_weights)
    particles = inference.smc.particles_list[-1] # T x particles x dim
    print smcopt_marginal_ll#, exact_marginal_ll
    plt.plot(to_numpy(z_true), label="true")
    plt.plot(np.arange(mean.size(0)), to_numpy(mean), 'k-', label='smc')
    #plt.fill_between(np.arange(mean.size(0)), to_numpy(mean) - to_numpy(scale), 
    #    to_numpy(mean) + to_numpy(scale), alpha=.5)
    plt.legend(loc='lower right')
    # plt.show()
    # embed()
    

    # given nonparametric posterior learn model parameters
    beta_init = 2. # sigmoid(-4.) = .0180
    log_alpha_init = -2.
    model = LearningDynamicsModel(init_prior, transition_scale, beta_init, log_alpha_init, dim=3)
    #opt_params = [model.beta]
    opt_params = [model.beta, model.log_alpha]
    print_every = 1
    em = EM(model)
    # specify optimization objective
    optimizer = torch.optim.Adam(opt_params, lr = .1)
    outputs = []
    for t in range(300):
        #old_lambda = lx.detach().cpu().numpy()
        optimizer.zero_grad()
        output = -em.forward(y, x, particles, weights)#, opt_params)
        print t, output, opt_params[0].detach().cpu().numpy(), opt_params[1].detach().cpu().numpy()
        # mean, var = inference.smc.estimate(data)
        # print np.linalg.norm(z_true.detach().cpu().numpy() - mean.detach().cpu().numpy())
        # outputs.append(output.item())
        # # if t % print_every == 0:
        # print 'iter: ', t, ' output: ', output.item(), \
        #     q_init_latent_loc.detach().cpu().numpy(), \
        #     np.exp(q_init_latent_log_scale.detach().cpu().numpy()), \
        #     np.exp(q_transition_log_scale.detach().cpu().numpy())
        output.backward(retain_graph=True)
        optimizer.step()
        outputs.append(output)
    embed()
    plt.plot(outputs)

    '''
    z_true = torch.tensor(z_true, requires_grad=False, dtype=dtype, device=device)
    z_mean = torch.rand(torch.tensor([dim]), requires_grad=grad_latents, dtype=dtype, device=device)
    z_log_scale = torch.tensor(torch.log(torch.rand(dim)), requires_grad=grad_latents, 
        dtype=dtype, device=device)
    
    smc_marginal_ll = inference.compute_log_marginal_likelihood()
    exact_marginal_ll = model.log_marginal_likelihood(T, x)
    print 'exact_marginal_ll: ', exact_marginal_ll, ' smc: ', smc_marginal_ll,
    embed()
    plt.plot(z_true.detach().cpu().numpy(), label='true')
    plt.plot(mean.detach().cpu().numpy(), label='smc')
    # plt.legend()
    # plt.show()

    #opt_params = z_mean

    if inference_type == 'map':
        variational_params = [z_mean]
    else:
        variational_params = [z_mean, z_log_scale]
    # create list of params to optimize
    opt_params = []
    if grad_latents:
        opt_params.extend(variational_params)
    if grad_model_params:
        opt_params.extend(model.return_model_params())
    data = [x, y]
    print_every = 10
    # specify optimization objective
    optimizer = torch.optim.Adam(opt_params, lr = .01)
    outputs = []
    for t in range(1000):
        old_mean = z_mean.detach().cpu().numpy()
        optimizer.zero_grad()
        output = -inference.forward(data, opt_params)
        outputs.append(output.item())
        output.backward()
        optimizer.step()
        if t % print_every == 0:
            print 'iter: ', t, ' output: ', output.item(), ' norm: ', \
                np.linalg.norm(np.abs(z_mean.detach().cpu().numpy() - z_true.cpu().numpy()))
    embed()
    plt.plot(outputs)


    # map_est = Map(lds)
    # lr = 0.01
    # optimizer = torch.optim.Adam(diff_params, lr = lr)
    # outputs = []
    # for t in range(10000):
    #   old_mean = latent_mean.detach().cpu().numpy()
    #   optimizer.zero_grad()
    #   output = -map_est.forward(obs, params)
    #   outputs.append(output.item())
    #   output.backward()
    #   optimizer.step()
    #   if t % 2000 == 0:
    #       print 'iter: ', t, ' output: ', output.item(), ' norm: ', \
    #           np.linalg.norm(np.abs(latent_mean.detach().cpu().numpy() - latents)), \
    #           'transition scale: ', np.exp(transition_log_scale.detach().cpu().numpy()), \
    #           'obs scale: ', np.exp(obs_log_scale.detach().cpu().numpy())
    # plt.plot(outputs[2500:])
    # plt.savefig('../output/map_loss.png')
    '''

