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
import evaluation
from evaluation import Evaluate
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
if torch.cuda.is_available():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
from utils import sigmoid

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
    def __init__(self, 
                 data, 
                 model, 
                 model_params_grad, 
                 savedir, 
                 num_obs_samples, 
                 num_future_steps, 
                 num_mc_samples,
                 ppc_window,
                 z_true=None,
                 true_model_params=None):
        self.data = data
        self.dim = self.data[1].size(2)
        self.T = self.data[1].size(0)

        self.train_data = self.data[0:2]
        self.y_future = self.data[4]
        self.x_future = self.data[5]
        self.y_complete = self.data[6]

        self.num_future_steps = self.y_future.shape[0]
        self.model = model
        self.savedir = savedir
        self.num_obs_samples = num_obs_samples
        self.num_future_steps = num_future_steps
        self.num_mc_samples = 1
        self.model_params_grad = model_params_grad
        self.true_model_params = true_model_params
        self.vi = MeanFieldVI(self.model, self.savedir, self.num_mc_samples)

        self.ppc_window = ppc_window
        self.isPPC = False


        init = 'map' # 'true'
        if init == 'map':
            init_z = self.map_estimate()
            self.var_params = self.vi.init_var_params(self.T, self.dim, init_z)
        elif init == 'true':
            self.var_params = self.vi.init_var_params(self.T, self.dim, z_true[0:-1])
        else:
            print 'specify valid init option.'
        self.iters = 30000
        #lr = 1e-4

        self.opt_params = {'var_mu': self.var_params[0], 
                   'var_log_scale': self.var_params[1]}

        # find model params with grad signal=True
        for k,v in self.model_params_grad.items():
            if v == True:
                self.opt_params[k] = self.model.params[k]
        #self.opt_params = [self.var_params[0], self.var_params[1], self.model.transition_log_scale]
        #self.optimizer =  torch.optim.SGD(self.opt_params.values(), lr=1e-2, momentum=.9)
        self.optimizer =  torch.optim.Adam(self.opt_params.values(), lr=1e-3)

        self.ev = Evaluate(self.data, self.model, savedir='', num_obs_samples=self.num_obs_samples)
        self.num_test = self.data[2].shape[0]
        self.num_train = self.data[0].shape[0] - self.num_test
    def unpack_data(self, data):
        y = data[0]
        x = data[1]
        return y, x
    def map_estimate(self):
        # initialize to all ones = smooth.
        z = torch.tensor(torch.ones(self.T, self.dim, device=device), requires_grad=True, device=device)
        y, x = self.unpack_data(self.data)
        self.map_iters = 4000
        self.opt_params = [z]
        self.map_optimizer =  torch.optim.Adam(self.opt_params, lr=1e-2)
        for t in range(self.map_iters):
            output = -self.model.log_joint(y, x, z)
            self.map_optimizer.zero_grad()
            output.backward()
            self.map_optimizer.step()
            if t % 250 == 0:
                print t, output.item()
            if t % 1000 == 0:
                plt.cla()
                plt.plot(to_numpy(z))
                figure = plt.gcf() # get current figure
                figure.set_size_inches(8, 6)
                plt.savefig(self.savedir+'/plots/curr_map_z.png')
        return self.opt_params[0].clone().detach()


    def optimize(self):
        #init_z = self.map_estimate()
        # self.opt_params = {'var_mu': self.var_params[0], 
        #                    'var_log_scale': self.var_params[1],
        #                    'transition_log_scale': self.model.transition_log_scale}
        #self.opt_params = [self.var_params[0], self.var_params[1], self.model.transition_log_scale]
        #self.optimizer =  torch.optim.SGD(self.opt_params.values(), lr=1e-1, momentum=.9)
        #self.optimizer = torch.optim.Adagrad(self.opt_params, lr=1e-2, momentum=0.9)
        #self.optimizer = torch.optim.Adam(self.opt_params, lr=1e-3)
        y, x = self.train_data[0], self.train_data[1]

        print 'optimizing...'
        outputs = []
        clip = 5.
        curr_model_params = {}
        for k,v in self.model_params_grad.items():
            if v == True:
                curr_model_params[k] = []
        #var_clip = 5.
        #model_param_clip = 500.
        for t in range(self.iters):
            # if t == 10000:
            #     self.optimizer =  torch.optim.SGD(self.opt_params.values(), lr=1e-2, momentum=.9)
            # e-step
            output = -self.vi.forward(self.train_data, self.var_params, t) / float(self.num_train)
            outputs.append(output.item())
            self.optimizer.zero_grad()
            output.backward()
          
            torch.nn.utils.clip_grad_norm(self.opt_params.values(), clip)

            self.optimizer.step()
            for k, v in curr_model_params.items():
                curr_model_params[k].append([el.item() for el in self.opt_params[k].flatten()])
            if t % 500 == 0:
                # printing
                print 'iter: ', t, 'loss: %.2f ' % output.item(), 
                for k,v in self.model_params_grad.items():
                    if v == True:
                        for el in self.opt_params[k].flatten():
                            print k, '%.3f ' % el.item(),
                test_post_predictive = self.ev.valid_loss(self.opt_params)
                y_future, future_trajectories, avg_future_marginal_lh = self.ev.sample_future_trajectory(self.opt_params, self.num_future_steps)
                train_acc, test_acc = self.ev.accuracy(self.opt_params)
                print 'train acc: %.3f ' % train_acc.item(), 'test acc: %.3f ' % test_acc.item(), \
                     'post pred: %.3f ' % test_post_predictive, 'future marginal lh: %.3f' % avg_future_marginal_lh.item()

                # plotting
                plt.cla()
                plt.plot(outputs)
                figure = plt.gcf() # get current figure
                figure.set_size_inches(8, 6)
                plt.savefig(self.savedir+'/loss.png')
                plt.cla()
                for k, v in curr_model_params.items():
                    plt.cla()
                    if k == 'beta':
                        plt.plot(sigmoid(np.array(v)))
                        if self.true_model_params:
                            plt.axhline(y=sigmoid(self.true_model_params[k]), color='r', linestyle='-')
                    else:
                        plt.plot(v)
                        # if self.true_model_params:
                        #     for el in self.true_model_params[k]:
                        #         plt.axhline(y=el, color='r', linestyle='-')
                    
                    figure = plt.gcf() # get current figure
                    figure.set_size_inches(8, 6)
                    plt.savefig(self.savedir+'/plots/'+k+'.png')

                zx = self.var_params[0]
                plt.cla()
                plt.plot(to_numpy(zx))
                figure = plt.gcf() # get current figure
                figure.set_size_inches(8, 6)
                plt.savefig(self.savedir+'/plots/curr_est_z.png')
            if t == 4000:
                # plt.cla()
                # plt.plot(to_numpy(future_trajectories[0]))
                # plt.show()


                if self.isPPC:
                    rw_avg, rw_true_avg = self.ev.ppc_reward(self.y_complete, x, self.T, self.num_obs_samples, self.dim, self.ppc_window)
                    plt.cla()
                    plt.plot(rw_avg, label='model')
                    plt.plot(rw_true_avg, label='true')
                    plt.legend(loc='lower right')
                    figure = plt.gcf() # get current figure
                    figure.set_size_inches(8, 6)
                    plt.savefig(self.savedir+'/plots/ppc_reward.png')

        # detach and clone all params
        for k in self.opt_params.keys():
            self.opt_params[k] = self.opt_params[k].clone().detach()
        

        # access learning and regularization components
        learning, regularization = self.model.log_prior_relative_contrib(self.var_params[0], y, x)
        torch.save(learning.clone().detach(), self.savedir+'/model_structs/learning_after_training.pth')
        torch.save(regularization.clone().detach(), self.savedir+'/model_structs/regularization_after_training.pth')
        plt.cla()
        plt.plot(to_numpy(learning.clone().detach()))
        plt.savefig(self.savedir+'/plots/learning_after_training.png')
        plt.cla()
        plt.plot(to_numpy(regularization.clone().detach()))
        plt.savefig(self.savedir+'/plots/regularization_after_training.png')
        
        return self.opt_params

class MeanFieldVI(object):
    '''
    Mean field fully factorized variational inference.
    '''
    def __init__(self, model, savedir, num_samples=5):
        self.model = model
        self.savedir = savedir
        self.num_samples = num_samples

    def init_var_params(self, T, dim, init_mean=None):
        mean = torch.tensor(init_mean, device=device, requires_grad=True)
        #mean = torch.tensor(5*torch.rand(T, dim, device=device), requires_grad=True, device=device)
        log_scale = torch.tensor(-5 * torch.ones(T, dim, device=device), requires_grad=True, device=device)
        return (mean, log_scale)

    def unpack_var_params(self, params):
        loc, log_scale = params[0], params[1]
        return loc, log_scale
    def unpack_data(self, data):
        y = data[0]
        x = data[1]
        return y, x

    def forward(self, data, var_params, itr):
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
        data_term = self.model.log_joint(y, x, samples[0])
        # data_terms = torch.empty(self.num_samples, device=device)
        # for i in range(len(samples)):
        #     data_terms[i] = self.model.log_joint(y, x, samples[i])
        # data_term = torch.mean(data_terms)
        entropy = torch.sum(var_dist.entropy())
        return (data_term + entropy)

def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

