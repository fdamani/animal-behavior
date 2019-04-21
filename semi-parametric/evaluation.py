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
    def __init__(self, data, model, savedir, num_obs_samples):
        self.data = data
        self.model = model
        self.num_obs_samples = num_obs_samples
    def unpack_data(self, data):
        y_train = data[0]
        x = data[1]
        y_test = data[2]
        test_inds = data[3]
        y_future = data[4]
        x_future = data[5]
        return y_train, x, y_test, test_inds, y_future, x_future

    def unpack_var_params(self, params):
        var_loc, var_log_scale = params['var_mu'], params['var_log_scale']
        return var_loc, var_log_scale

    def return_train_ind(self, y):
        return y[:,0] != -1

    def valid_loss(self, opt_params, num_mc_samples=25):
        '''
            TODO:
            1. monte carlo approx by averaging over multiple trajectories
            sampled from q(z) (e.g. the posterior predictive)
            2. ROC/AUC as well as accuracy.
        need to use log sum exp trick

        log sum_{mc samples} prod p(y|x,z)

        - test marginal likelihood
        - evaluate p(y_test|x_test, z_test) under many samples from q(z_test)
        - average probabilities
        - take log.


        get S monte carlo samples from posterior
        for each sample
            compute the log likelihood of the test data under the sample
        use the log sum exp trick to get the average probability (e.g. the posterior predictive)
            - log(1/N) + log sum exp (log p(y_test | x, z'))
        average the log marginal by number of observations to get the average lh.

        '''
        y_train, x, y_test, test_inds, y_future, x_future = self.unpack_data(self.data)
        num_test = y_test.shape[0]
        num_train = y_train.shape[0] - num_test
        num_obs = y_test.shape[1]
        var_loc, var_log_scale = self.unpack_var_params(opt_params)
        var_loc = var_loc.clone().detach()
        var_log_scale = var_log_scale.clone().detach()
        posterior_dist = Normal(var_loc, torch.exp(var_log_scale))
        # expected_test_likelihood = 0
        # expected_train_probs = 0
        # expected_test_probs = 0
        test_log_lh = []
        for i in range(num_mc_samples):
            z = posterior_dist.sample()
            test_log_lh.append(self.model.log_likelihood_test(y_train, y_test, test_inds, x, z))
            # expected_test_likelihood += test_likelihood
            # expected_train_probs += train_p_y_1_given_x
            # expected_test_probs += test_p_y_1_given_x
        test_log_lh = torch.tensor(test_log_lh, device=device)
        test_marginal_log_ll = -torch.log(torch.tensor(float(num_mc_samples), device=device)) + \
            torch.logsumexp(test_log_lh, dim=0)

        test_marginal_log_ll = test_marginal_log_ll / float(num_test)
        test_marginal_log_ll = test_marginal_log_ll / float(num_obs)


        # compute posterior predictive per sample
        # test_marginal_log_ll = torch.sum(torch.log(expected_test_likelihood))
        # test_posterior_predictive = test_marginal_log_ll / float(num_test)
        # test_posterior_predictive = test_posterior_predictive / float(num_obs)


        # train_inds = self.return_train_ind(y_train)

        # compute train accuracy
        # train_preds = torch.tensor(expected_train_probs.detach() > .5, device=device, dtype=torch.float)
        # train_accuracy = torch.mean(torch.tensor(train_preds == y_train[train_inds], device=device, dtype=torch.float))
        
        # # compute test accuracy
        # test_preds = torch.tensor(expected_test_probs.detach() > .5, device=device, dtype=torch.float)
        # test_accuracy = torch.mean(torch.tensor(test_preds == y_test, device=device, dtype=torch.float))

        # avg_train_log_ll = train_log_ll.detach() / float(num_train)
        # avg_test_log_ll = test_log_ll.detach() / float(num_test)

        return test_marginal_log_ll #, train_accuracy, test_accuracy
        #return avg_train_log_ll, avg_test_log_ll, train_accuracy, test_accuracy, train_probs, test_probs


    def accuracy(self, opt_params):
        y_train, x, y_test, test_inds, y_future, x_future = self.unpack_data(self.data)
        num_test = y_test.shape[0]
        num_train = y_train.shape[0] - num_test
        num_obs = y_test.shape[1]
        var_loc, var_log_scale = self.unpack_var_params(opt_params)
        z = var_loc.clone().detach()
        
        logits = torch.sum(x * z[:, None, :], dim=2)

        # compute train accuracy
        train_inds = self.model.return_train_ind(y_train)
        logits_train = logits[train_inds]
        train_probs = torch.sigmoid(logits_train)
        train_preds = torch.tensor(train_probs.detach() > .5, device=device, dtype=torch.float)
        train_accuracy = torch.mean(torch.tensor(train_preds == y_train[train_inds], device=device, dtype=torch.float))
        
        # compute validation accuracy
        logits_test = logits[test_inds]
        test_probs = torch.sigmoid(logits_test)
        test_preds = torch.tensor(test_probs.detach() > .5, device=device, dtype=torch.float)
        test_accuracy = torch.mean(torch.tensor(test_preds == y_test, device=device, dtype=torch.float))


        return train_accuracy, test_accuracy

    def sample_future_trajectory(self, opt_params, num_future_steps):
        '''
        given a posterior p(z|x) 
        - sample a trajectory
        - forward sample from last time point for num_future_steps

        return predicted y's and z's

        - compute a set of plausible future trajectories
        - evaluate p(y_test | x_test, z_test) under each trajectory
        - average probabilities
        - take log and return value.
        '''
        y_train, x, y_test, test_inds, y_true_future, x_future = self.unpack_data(self.data)
        num_test = y_test.shape[0]
        num_train = y_train.shape[0] - num_test
        num_future_samples = y_true_future.shape[0]
        num_obs = y_test.shape[1]
        var_loc, var_log_scale = self.unpack_var_params(opt_params)
        z = var_loc.clone().detach()
        assert num_future_samples == num_future_steps

        # sample forward once
        #### sample forward multiple times?
        num_mc_samples = 50
        log_lh = []
        future_trajectories = []
        for i in range(num_mc_samples):
            y_future, z_future = self.model.sample_forward(y_train, y_test, test_inds, x, z, 
                x_future, self.num_obs_samples, num_future_steps)
            future_trajectories.append(z_future)
            # compute log prob
            log_lh.append(self.model.log_likelihood(y_true_future, x_future, z_future))

        log_lh = torch.tensor(log_lh, device=device)
        marginal_lh = -torch.log(torch.tensor(float(num_mc_samples), device=device)) + \
            torch.logsumexp(log_lh, dim=0)

        marginal_lh = marginal_lh / float(num_future_samples)
        marginal_lh = marginal_lh / float(num_obs)


        return y_future, future_trajectories, marginal_lh, 

    def ppc_reward(self, y_true, x_true, T, num_obs_samples, dim, window, num_samples=25):
        ''' posterior predictive check
        given model parameters, simulate trajectories and data.
        - compute smoothed reward over time
        - compare to true.
        '''
        rw_true = self.model.rat_reward_vec(y_true, x_true)
        rw_true = torch.mean(rw_true, dim=1)
        rw_true_avg = np.convolve(rw_true, np.ones(window))/ float(window)
        rw_true_avg = rw_true_avg[window:-window]

        rw_avg_list = []
        for i in range(num_samples):
            y, x, z = self.model.sample(T, num_obs_samples, dim)
            rw = self.model.rat_reward_vec(y, x)
            rw = torch.mean(rw, dim=1)
            rw_avg = np.convolve(rw, np.ones(window))/ float(window)
            rw_avg = rw_avg[window:-window]
            rw_avg_list.append(rw_avg)
        rw_avg = np.mean(np.array(rw_avg_list), axis=0)

        return rw_avg, rw_true_avg

    def loss_on_future_trajectory(self, y_future, z_future):
        y_train, x, y_test, test_inds, y_true_future, x_future = self.unpack_data(self.data)


   