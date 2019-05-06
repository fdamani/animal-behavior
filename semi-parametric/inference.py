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

#from LBFGS import LBFGS, FullBatchLBFGS


# set random seed
torch.manual_seed(7)
np.random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.float32
dtype = torch.double

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
                 model_params,
                 model_params_grad, 
                 savedir, 
                 num_obs_samples, 
                 num_future_steps, 
                 num_mc_samples,
                 ppc_window,
                 z_true=None,
                 true_model_params=None,
                 iters=1000):
        self.data = data
        self.dim = self.data[1].size(2)
        self.T = self.data[1].size(0)
        self.model_params = model_params
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
        self.init_z = self.map_estimate()
        if init == 'map':
            self.var_params = self.vi.init_var_params(self.T, self.dim, self.init_z, grad=True)
        elif init == 'true':
            self.var_params = self.vi.init_var_params(self.T, self.dim, z_true, grad=True)
        else:
            print 'specify valid init option.'
        self.iters = iters

        self.opt_params = {'var_mu': self.var_params[0], 
                   'var_log_scale': self.var_params[1]}
        for k,v in self.model_params_grad.items():
            if v == True:
                self.opt_params[k] = self.model_params[k]
        # self.var_params_model = self.vi.init_var_params_model()
        # self.opt_params['model_mu'] = self.var_params_model[0]
        # self.opt_params['model_log_scale'] = self.var_params_model[1]


        self.ev = Evaluate(self.data, self.model, savedir='', num_obs_samples=self.num_obs_samples)
        self.num_test = self.data[2].shape[0]
        self.num_train = self.data[0].shape[0] - self.num_test
    def unpack_data(self, data):
        y = data[0]
        x = data[1]
        return y, x
    def map_estimate(self):
        # initialize to all ones = smooth.
        z = torch.tensor(torch.rand(self.T, self.dim, dtype=dtype, device=device), requires_grad=True, dtype=dtype, device=device)
        y, x = self.unpack_data(self.data)
        self.map_iters = 100
        self.opt_params = [z]
        #self.map_optimizer =  torch.optim.Adam(self.opt_params, lr=1e-3)
        self.map_optimizer = torch.optim.LBFGS(self.opt_params)
        lbfgs = True
        for t in range(self.map_iters):
            def closure():
                self.map_optimizer.zero_grad()
                output = -self.model.log_joint(self.model_params, y, x, z)
                output.backward()
                return output
            if lbfgs:
                self.map_optimizer.step(closure)
                with torch.no_grad():
                    output = -self.model.log_joint(self.model_params, y, x, z)
            else:
                output = -self.model.log_joint(self.model_params, y, x, z)
                self.map_optimizer.zero_grad()
                output.backward()
                self.map_optimizer.step()
            if t % 5 == 0:
                print t, output.item()
            if t % 5 == 0:
                plt.cla()
                plt.plot(to_numpy(z))
                figure = plt.gcf() # get current figure
                figure.set_size_inches(8, 6)
                plt.savefig(self.savedir+'/plots/curr_map_z.png')
        return self.opt_params[0].clone().detach()

    def run(self):
        self.optimizer = torch.optim.SGD(self.opt_params.values(), momentum=0.99, lr=1e-6) # .99, 1e-6
        return self.optimize(300000, False, 1000)

    def optimize(self, iters, lbfgs, print_every):
        y, x = self.train_data[0], self.train_data[1]
        print 'optimizing...'
        outputs = []
        clip = 5.
        curr_model_params = {}
        for k,v in self.model_params_grad.items():
            if v == True:
                curr_model_params[k] = []

        self.iters = iters
        for t in range(self.iters):

            #torch.nn.utils.clip_grad_norm(self.opt_params.values(), clip)
            self.optimizer.zero_grad()
            output = -self.vi.forward(self.model_params, self.train_data, self.var_params, t) #/ float(self.num_train)
            #output = -self.vi.forward_with_model_param_post(self.model_params, self.train_data, self.opt_params, t) #/ float(self.num_train)

            outputs.append((output.item()/float(self.num_train)))
            output.backward()
            self.optimizer.step()

            for k, v in curr_model_params.items():
                if k in self.opt_params:
                    curr_model_params[k].append([el.item() for el in self.opt_params[k].flatten()])

            if t % print_every == 0:
                # printing
                ox = output.item() / float(self.num_train)
                print 'iter: ', t, 'loss: %.2f ' % ox, 'scale: ',
                if 'var_log_scale' in self.opt_params:
                    print torch.mean(self.opt_params['var_log_scale'].clone().detach()).item(),
                if 'model_mu' in self.opt_params:
                    print self.opt_params['model_mu'].item(), self.opt_params['model_log_scale'].item()
                for k,v in self.model_params_grad.items():
                    if v == True:
                        if k in self.opt_params:
                            for el in self.opt_params[k].flatten():
                                print k, '%.3f ' % el.item(),
                print '\n'
                test_marginal = self.ev.valid_loss(self.opt_params)
                #y_future, future_trajectories, avg_future_marginal_lh = self.ev.sample_future_trajectory(self.opt_params, self.num_future_steps)
                train_acc, test_acc = self.ev.accuracy(self.opt_params)
                print 'train acc: %.3f ' % train_acc.item(), 'test acc: %.3f ' % test_acc.item(), \
                     'test marginal likelihood: %.3f ' % test_marginal#, 'future marginal lh: %.3f' % avg_future_marginal_lh.item()

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
                    else:
                        plt.plot(v)
                        # if self.true_model_params:
                        #     for el in self.true_model_params[k]:
                        #         plt.axhline(y=el, color='r', linestyle='-')
                    
                    figure = plt.gcf() # get current figure
                    figure.set_size_inches(8, 6)
                    plt.savefig(self.savedir+'/plots/'+k+'.png')

                zx = self.var_params[0]
                zx = to_numpy(zx)
                zx_scale = np.exp(to_numpy(self.var_params[1]))
                plt.cla()
                labels = ['Bias', 'X1', 'X2', 'Choice t-1', 'RW Side t-1', 'X1 t-1', 'X2 t-1']
                for j in range(zx_scale.shape[1]):
                    #plt.plot(zx[:,j], label=labels[j], linewidth=.5)
                    plt.plot(zx[:,j], linewidth=.5)
                    
                    # plt.fill_between(np.arange(zx.shape[0]), zx[:,j] - zx_scale[:,j],  zx[:,j] + zx_scale[:,j])
                figure = plt.gcf() # get current figure
                figure.set_size_inches(12, 8)
                # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.savefig(self.savedir+'/plots/curr_est_z.png')
                test_inds = self.data[-4].cpu().numpy()
                zx_test = zx[test_inds]
                plt.cla()
                for j in range(zx_scale.shape[1]):
                    #plt.plot(zx_test[:,j], label=labels[j], linewidth=.5)
                    plt.plot(zx_test[:,j], linewidth=.5)

                plt.savefig(self.savedir+'/plots/curr_est_test_z.png')


        test_marginal = self.ev.valid_loss(self.opt_params)
        np.savetxt(self.savedir+'/test_marginal.txt', np.array([test_marginal.item()]))
        print 'final test marginal: ', test_marginal.item()
            # detach and clone all params
        for k in self.opt_params.keys():
            self.opt_params[k] = self.opt_params[k].clone().detach()
        


        # access learning and regularization components
        #learning, regularization = self.model.log_prior_relative_contrib(self.var_params[0], y, x)
        #torch.save(learning.clone().detach(), self.savedir+'/model_structs/learning_after_training.pth')
        #torch.save(regularization.clone().detach(), self.savedir+'/model_structs/regularization_after_training.pth')
        #plt.cla()
        #plt.plot(to_numpy(learning.clone().detach()))
        #plt.savefig(self.savedir+'/plots/learning_after_training.png')
        #plt.cla()
        #plt.plot(to_numpy(regularization.clone().detach()))
        #plt.savefig(self.savedir+'/plots/regularization_after_training.png')
        return self.opt_params

class MeanFieldVI(object):
    '''
    Mean field fully factorized variational inference.
    '''
    def __init__(self, model, savedir, num_samples=5):
        self.model = model
        self.savedir = savedir
        self.num_samples = num_samples

    def init_var_params(self, T, dim, init_mean=None, grad=True):
        mean = torch.tensor(init_mean, device=device, dtype=dtype, requires_grad=grad)
        #mean = torch.tensor(torch.rand(T, dim, device=device), requires_grad=grad, device=device)
        log_scale = torch.tensor(-5 * torch.ones(T, dim, dtype=dtype, device=device), requires_grad=grad, dtype=dtype, device=device)
        return (mean, log_scale)

    def init_var_params_model(self, grad=True):
        mean = torch.tensor([-3.], device=device, dtype=dtype, requires_grad=grad)
        #mean = torch.tensor(torch.rand(T, dim, device=device), requires_grad=grad, device=device)
        log_scale = torch.tensor(-5 * torch.ones(1, dtype=dtype, device=device), requires_grad=grad, dtype=dtype, device=device)
        return (mean, log_scale)

    def unpack_var_params(self, params):
        loc, log_scale = params[0], params[1]
        return loc, log_scale
    def unpack_data(self, data):
        y = data[0]
        x = data[1]
        return y, x

    def forward(self, model_params, data, var_params, itr, num_samples=1):
        '''
            useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
        '''
        y, x = self.unpack_data(data)
        loc, log_scale = self.unpack_var_params(var_params)
        var_dist = Normal(loc, torch.exp(log_scale))
        #cov = torch.diag(torch.exp(log_scale))**2
        #scale_tril = cov.tril()
        #var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
        samples = var_dist.rsample(torch.Size((num_samples,)))
        data_term = self.model.log_joint(model_params, y, x, samples[0])
        entropy = torch.sum(var_dist.entropy())
        return (data_term + entropy)

    def forward_with_model_param_post(self, data, var_params, itr, num_samples=1):
        '''
            useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
        '''
        y, x = self.unpack_data(data)
        loc, log_scale, loc_mod, log_scale_mod = var_params['var_mu'], var_params['var_log_scale'], \
            var_params['model_mu'], var_params['model_log_scale']
        #loc, log_scale = self.unpack_var_params(var_params)
        var_dist = Normal(loc, torch.exp(log_scale))
        var_dist_model = Normal(loc_mod, torch.exp(log_scale_mod))
        #cov = torch.diag(torch.exp(log_scale))**2
        #scale_tril = cov.tril()
        #var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
        samples = var_dist.rsample(torch.Size((num_samples,)))
        mod_samples = var_dist_model.rsample(torch.Size((num_samples,)))
        # data_terms = torch.empty(num_samples, device=device)
        # for i in range(len(samples)):
        #     data_terms[i] = self.model.log_joint(y, x, samples[i], mod_samples[i])
        # data_term = torch.mean(data_terms)

        data_term = self.model.log_joint(y, x, samples[0], mod_samples[0])
        entropy = torch.sum(var_dist.entropy())
        entropy_mod = torch.sum(var_dist_model.entropy())
        return (data_term + entropy + entropy_mod)

    def forward_multiple_mcs(self, model_params, data, var_params, itr, num_samples=5):
        '''
            useful for analytic kl  kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
        '''
        y, x = self.unpack_data(data)
        loc, log_scale = self.unpack_var_params(var_params)
        var_dist = Normal(loc, torch.exp(log_scale))
        #cov = torch.diag(torch.exp(log_scale))**2
        #scale_tril = cov.tril()
        #var_dist = MultivariateNormal(loc, scale_tril=scale_tril)
        samples = var_dist.rsample(torch.Size((num_samples,)))
        #data_term = self.model.log_joint(y, x, samples[0])
        data_terms = torch.empty(num_samples, device=device)
        for i in range(len(samples)):
            data_terms[i] = self.model.log_joint(model_params, y, x, samples[i])
        data_term = torch.mean(data_terms)
        entropy = torch.sum(var_dist.entropy())
        return (data_term + entropy)
def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

