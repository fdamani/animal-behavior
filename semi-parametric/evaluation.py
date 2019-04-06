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
from torch.distributions import constraints, transform_to
import psutil
import learning_dynamics
from learning_dynamics import LearningDynamicsModel
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

class Evaluate(object):
    def __init__(self, data, model, savedir, num_obs):
        self.data = data
        self.model = model
        self.num_obs = num_obs
    def unpack_data(self, data):
        y_train = data[0]
        x = data[1]
        y_test = data[2]
        test_inds = data[3]
        y_future = data[4]
        x_future = data[5]
        return y_train, x, y_test, test_inds, y_future, x_future

    def unpack_var_params(self, params):
        var_loc, var_log_scale = params[0], params[1]
        return var_loc, var_log_scale

    def return_train_ind(self, y):
        return y[:,0] != -1

    def valid_loss(self, opt_params):
        '''
            TODO:
            1. monte carlo approx by averaging over multiple trajectories
            sampled from q(z) (e.g. the posterior predictive)
            2. ROC/AUC as well as accuracy.
        '''
        y_train, x, y_test, test_inds, y_future, x_future = self.unpack_data(self.data)
        num_test = y_test.shape[0]
        num_train = y_train.shape[0] - num_test
        var_loc, var_log_scale = self.unpack_var_params(opt_params)
        z = var_loc # for now, just take mean of q(z) as your z sample
        train_ll, test_ll, train_probs, test_probs = \
            self.model.log_likelihood_test(y_train, y_test, test_inds, x, z)
        train_inds = self.return_train_ind(y_train)
        train_preds = torch.tensor(train_probs.detach() > .5, device=device, dtype=torch.float)
        train_accuracy = torch.mean(torch.tensor(train_preds == y_train[train_inds], device=device, dtype=torch.float))
        
        test_preds = torch.tensor(test_probs.detach() > .5, device=device, dtype=torch.float)
        test_accuracy = torch.mean(torch.tensor(test_preds == y_test, device=device, dtype=torch.float))

        avg_train_ll = train_ll.detach() / float(num_train)
        avg_test_ll = test_ll.detach() / float(num_test)

        return avg_train_ll, avg_test_ll, train_accuracy, test_accuracy, train_probs, test_probs


    def sample_future_trajectory(self, opt_params, num_future_steps):
        '''
        given a posterior p(z|x) 
        - sample a trajectory
        - forward sample from last time point for num_future_steps

        return predicted y's and z's
        '''
        y_train, x, y_test, test_inds, y_true_future, x_future = self.unpack_data(self.data)
        num_test = y_test.shape[0]
        num_train = y_train.shape[0] - num_test
        num_future_samples = y_true_future.shape[0]
        var_loc, var_log_scale = self.unpack_var_params(opt_params)
        z = var_loc

        # sample forward once
        y_future, z_future = self.model.sample_forward(y_train, y_test, test_inds, x, z, 
            x_future, self.num_obs, num_future_steps)


        # compute log prob
        future_marginal_lh = self.model.log_likelihood_vec(y_true_future, x_future, z_future)
        avg_future_marginal_lh = future_marginal_lh.detach() / float(num_future_samples)


        return y_future, z_future, avg_future_marginal_lh

    def loss_on_future_trajectory(self, y_future, z_future):
        y_train, x, y_test, test_inds, y_true_future, x_future = self.unpack_data(self.data)


   