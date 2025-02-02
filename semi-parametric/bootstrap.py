from __future__ import division
import time
import sys
import os
import numpy as np
import math
import matplotlib
from IPython import display, embed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD, Adam
from torch.distributions import Normal, Bernoulli, MultivariateNormal

import inference
from inference import Inference, MeanFieldVI

import psutil
import learning_dynamics
from learning_dynamics import LearningDynamicsModel
from evaluation import Evaluate
import read_data
from read_data import read_and_process, train_test_split, train_future_split
process = psutil.Process(os.getpid())
# set random seed
torch.manual_seed(10)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

#dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32

def to_numpy(tx):
    return tx.detach().cpu().numpy()


def simulate_dataset(model_params, T, dim, num_obs_samples):
    transition_log_scale = model_params['transition_log_scale']
    init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
    model = LearningDynamicsModel(init_prior, transition_log_scale, dim=dim)
    num_obs = 50
    y, x, z_true = model.sample(T=T, num_obs_samples=num_obs_samples)
    return (y, x, z_true)


def estimation(dataset, num_obs_samples, num_future_steps, num_mc_samples):
    y, x, z_true = dataset
    init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
    init_transition_log_scale = [math.log(1.)]
    model = LearningDynamicsModel(init_prior, init_transition_log_scale, dim=3)
    inference = Inference(data, model, savedir='', num_obs_samples=num_obs_samples, num_future_steps=num_future_steps, num_mc_samples=num_mc_samples, z_true=z_true)
    opt_params = inference.optimize()
    return opt_params


def compute_mean_and_std(x):
    return x.mean(), x.std()


if __name__ == '__main__':

    grad_latents = True
    grad_model_params = False
    inference_types = ['map', 'mfvi', 'is', 'smc', 'vsmc']
    inference_type = inference_types[4]
    sim = True
    file_path = '/tigress/fdamani/neuro_output/'
    # file_path = 'output/'
    savedir = file_path
    import datetime
    savedir += str(datetime.datetime.now())

    datafiles = ['W065.csv', 'W066.csv', 'W068.csv', 'W072.csv', 'W073.csv', 'W074.csv', 'W075.csv', 'W078.csv',
                 'W080.csv', 'W081.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv', 'W094.csv']
    if sim:
        # T = 200 # 100
        T = 200
        # time-series model
        # sim model parameters
        dim = 3
        init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
        transition_scale = [math.log(.2)]# * dim
        log_gamma = math.log(1e-0)
        beta = 10. # sigmoid(4.) = .9820
        log_alpha = math.log(1e-2)
        model = LearningDynamicsModel(init_prior, transition_scale, dim=3)
        #model = LogReg_LDS(init_prior=(0.0, 0.02), transition_scale=1e-3)
        num_obs_samples = 50
        y, x, z_true = model.sample(T=T, num_obs_samples=num_obs_samples)
        y = y.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        z_true = z_true.detach().cpu().numpy()

        plt.cla()
        plt.plot(z_true)
        #plt.show()
        plt.savefig('sim_z.png')
        # embed()
        # model params
    else:
        num_obs_samples = 200
        #f = '/tigress/fdamani/neuro_data/data/clean/LearningData_W066_minmaxnorm.txt'
        # data file
        index = 0# sys.argv[1]
        rat = datafiles[int(index)]
        print rat
        f = '/tigress/fdamani/neuro_data/data/raw/allrats_withmissing_limitedtrials/csv/'
        f += rat
        rat = f.split('/')[-1].split('.csv')[0]
        
        # add to dir name
        savedir += '__obs'+str(num_obs_samples)
        savedir += '__'+rat

        # os.mkdir(savedir)
        # f = '../data/W066_short.csv'
   
        x, y, rw = read_and_process(num_obs_samples, f, savedir=savedir)
        rw = torch.tensor(rw, dtype=dtype, device=device)

        dim = x.shape[2]
        T = x.shape[0]
        init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
        transition_scale = [math.log(1.0)] #* dim  
    num_future_steps = 1
    category_tt_split = 'single'
    num_mc_samples = 10
    y, x, y_future, x_future = train_future_split(y, x, num_future_steps)
    y_train, y_test, test_inds = train_test_split(y, x, cat=category_tt_split)
    x = torch.tensor(x, dtype=dtype, device=device)
    y_train = torch.tensor(y_train, dtype=dtype, device=device)
    y_test = torch.tensor(y_test, dtype=dtype, device=device)
    test_inds = torch.tensor(test_inds, dtype=torch.long, device=device)
    y_future = torch.tensor(y_future, dtype=dtype, device=device)
    x_future = torch.tensor(x_future, dtype=dtype, device=device)
    #data = [y_train, x]
    data = [y_train, x, y_test, test_inds, y_future, x_future]
    # declare model here
    init_transition_log_scale = [math.log(1.)]# * dim
    model = LearningDynamicsModel(init_prior, init_transition_log_scale, dim=3)
    inference = Inference(data, model, savedir='', num_obs_samples=num_obs_samples, num_future_steps=num_future_steps, num_mc_samples=num_mc_samples, z_true=z_true)
    opt_params = inference.optimize()

    num_bootstraps = 25
    b_params = []
    for i in range(num_bootstraps):
        dataset = simulate_dataset(opt_params, T, dim, num_obs_samples)
        opt_params = estimation(dataset, num_obs_samples, num_future_steps, num_mc_samples)
        b_params.append(opt_params)

    transition_log_scales = []
    for i in range(num_bootstraps):
        val = b_params[i]['transition_log_scale']
        val = np.log2(np.exp(val))
        transition_log_scales.append(val)
    transition_log_scales = np.array(transition_log_scales)

    mu, std = compute_mean_and_std(transition_log_scales)
    plt.cla()
    plt.errorbar(x=np.arange(1), y=mu, yerr=2*std, fmt='o')
    plt.show()
    embed()



    # y_future, z_future, avg_future_marginal_lh = inference.ev.sample_future_trajectory(inference.var_params, num_future_steps)
    # if sim:
    #     plt.plot(z_true[-num_future_steps:])
    # plt.plot(to_numpy(z_future))
    # plt.show()
    #plt.show()
    # ev = Evaluate(data, model, savedir='', num_obs=num_obs)
    # train_ll, test_ll = ev.valid_loss()
    # then test!

  