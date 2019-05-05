'''
    model classes
    each class inherits model abstract class
'''
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
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import psutil
process = psutil.Process(os.getpid())

# set random seed
torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.float32
dtype = torch.double
if torch.cuda.is_available():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn as nn


class LearningDynamicsModel(object):
    def __init__(self,
                 dim):
        self.dim = dim
        self.sigmoid = nn.Sigmoid()

    def sample(self, T, model_params, num_obs_samples=10, dim=3, x=None):
        '''
            sample latent variables and observations
        '''
        # generate 1D x from standard normal
        intercept = torch.ones(T, num_obs_samples, 1, dtype=dtype, device=device)
        if x is None: 
            x = torch.randn(T, num_obs_samples, dim-1, dtype=dtype, device=device)
            x = torch.cat([intercept, x], dim=2)
        
        z = [self.sample_init_prior(model_params)]
        # z[0][0][1] = torch.tensor(-1., dtype=dtype, device=device)
        # z[0][0][2] = torch.tensor(1., dtype=dtype, device=device)
        # set second value to -1.
        # set 3rd value to +1
        y = [self.sample_likelihood(x[0], z[0], num_obs_samples)]
        for i in range(1, T):
            # sample and append a new z
            z.append(self.sample_prior(model_params, z[i-1], y[-1], x[-1]))
            # sample an observation
            y.append(self.sample_likelihood(x[i], z[i], num_obs_samples))

        y = torch.t(torch.cat(y, dim = 1))
        z = torch.cat(z)
        return y, x, z

    def sample_prior(self, model_params, z_prev, y_prev=None, x_prev=None):
        '''sample from p(z_t | z_t-1, y_t-1, x_t-1)
        simple AR-1 prior

        z_t+1 = beta * z_t + alpha * grad_rat_obj - sgn(z_t)*C
        z is 1 x dim
        y is num_obs x 1
        x is num_obs x dim
        '''

        # compute policy gradient update
        grad_rat_obj = self.grad_rat_obj_score(y_prev, x_prev, z_prev)

        # grad_loss = -grad_rat_obj + torch.exp(self.log_gamma) * z_prev 
        # grad_loss = -grad_rat_obj + torch.exp(self.log_gamma) * (.5 * z_prev + .5 * torch.sign(z_prev))
        regularization_comp = torch.exp(model_params['log_gamma']) * (self.sigmoid(model_params['beta']) * z_prev + \
            (1.0 - self.sigmoid(model_params['beta']))* torch.sign(z_prev))

        grad_loss = -grad_rat_obj + regularization_comp

        mean = z_prev - torch.exp(model_params['log_alpha']) * grad_loss
        scale = torch.exp(model_params['transition_log_scale'])
        prior = Normal(mean, scale)
        return prior.sample()
    
    def sample_init_prior(self, model_params):
        prior = Normal(model_params['init_latent_loc'], torch.exp(model_params['init_latent_log_scale']))
        return prior.sample()[None]
    
    def sample_likelihood(self, x_t, z_t, num_obs_samples):
        ''' z_t is 1 x D
            x_t 

            x_t is num_samples x dimension
        '''
        #logits = torch.matmul(z_t, x_t).flatten()
        logits = torch.matmul(x_t, torch.t(z_t))
        obs = Bernoulli(self.sigmoid(logits))
        return obs.sample()

    def log_init_prior(self, model_params, z):
        '''evaluate log pdf of z0 under the init prior
        '''
        prior = Normal(model_params['init_latent_loc'], torch.exp(model_params['init_latent_log_scale'])) 
        return torch.sum(prior.log_prob(z))

    def log_init_prior_batch(self, z):
        '''evaluate log pdf of z0 under the init prior
        z0 is particles x dimension
        return log probs for each particle
        '''
        prior = Normal(model_params['init_latent_loc'], torch.exp(model_params['init_latent_log_scale'])) 
        return torch.sum(prior.log_prob(z), dim=-1)
## vectorized functions ##
    def log_joint(self, model_params, y, x, z, scale_loc=None):
        '''
        input: x (observations T x D)
        input: latent_mean
        return logpdf under the model parameters
        '''
        T = y.size(0)
        logprob = 0

        logprob += self.log_init_prior(model_params, z[0][None])
        #print 'init prior: ', self.log_init_prior(z[0][None]).item()
        logprob += self.log_prior_vec(model_params, z, y, x)
        #print 'prior: ', self.log_prior_vec(z, y, x).item()
        logprob += self.log_likelihood_vec(y, x, z)
        #print 'lh: ', self.log_likelihood_vec(y, x, z).item()
        return logprob
    
    def return_train_ind(self, y):
        return y[:,0] != -1

    def log_likelihood_vec(self, y, x, z):
        '''
            p(y_t | y_1:t-1, x_1:t, z_1:t)
            y will contain -1's denoting unobserved data
            only compute log probs for observed y's.
            identify indices where y does not equal -1
            compute log probs for those and then sum accordingly.
        '''
        logits = torch.sum(x * z[:, None, :], dim=2)
        train_inds = self.return_train_ind(y)
        logits_train = logits[train_inds]
        # limit logits to observed y's
        obs = Bernoulli(logits=logits_train)
        return torch.sum(obs.log_prob(y[train_inds]))

    def log_prior_vec(self, model_params, z, y, x):
        '''
            input: z_1:t
            parameterize p(z_t | z_t-1, theta)
        '''
        z_prev = z[0:-1]
        z_curr = z[1:]

        grad_rat_obj = self.grad_rat_obj_score_vec(y, x, z)[0:-1]
        # grad_loss = -torch.exp(self.log_alpha) * grad_rat_obj + torch.exp(self.log_gamma) * z_prev
        # # properly vectorized
        # mean = z_prev - grad_loss

        # grad_loss = -grad_rat_obj + torch.exp(self.log_gamma) * z_prev
        #grad_loss = -grad_rat_obj + torch.exp(self.log_gamma) * (.5 * z_prev + .5 * torch.sign(z_prev))
        regularization_comp = torch.exp(model_params['log_gamma']) * (self.sigmoid(model_params['beta']) * z_prev + \
            (1.0 - self.sigmoid(model_params['beta']))* torch.sign(z_prev))

        grad_loss = -grad_rat_obj + regularization_comp
        #grad_loss = -torch.exp(self.log_alpha) * grad_rat_obj + self.sparsity_dims * regularization_comp

        mean = z_prev - ((torch.exp(model_params['log_alpha'])[0] * grad_loss * self.rat_reward_vec(y, x)[0:-1]) + \
            (torch.exp(model_params['log_alpha'])[1] * grad_loss * (1.0 - self.rat_reward_vec(y, x)[0:-1])))

        #mean = z_prev - (torch.exp(model_params['log_alpha']) * grad_loss * self.rat_reward_vec(y, x)[0:-1])
        #mean = z_prev - torch.exp(model_params['log_alpha']) * grad_loss

        prior_scale = Normal(-3., .01)
        scale = torch.exp(model_params['transition_log_scale'])
        prior = Normal(mean, scale)
        return torch.sum(prior.log_prob(z_curr)) + torch.sum(prior_scale.log_prob(model_params['transition_log_scale']))

    def log_likelihood(self, y, x, z):
        logits = torch.sum(x * z[:, None, :], dim=2)
        obs = Bernoulli(logits=logits)
        log_lh = torch.sum(obs.log_prob(y))
        return log_lh


    def sample_forward(self, y_train, y_test, test_inds, x, z, x_future, num_obs, num_future_steps):
        '''
            sample forward from last time point of z.
        '''
        z_future = []
        y_future = []
        y_prev = y_train[-1] if y_train[-1][0] != -1 else y_test[-1]
        z_prev = z[-1][None]
        x_prev = x[-1]
        for i in range(num_future_steps):
            z_i = self.sample_prior(z_prev, y_prev[:,None], x_prev)
            y_i = self.sample_likelihood(x_future[i], z_i, num_obs)
            y_prev = y_i
            z_prev = z_i
            x_prev = x_future[i]

            z_future.append(z_i)
            y_future.append(y_i)
        
        y_sampled_future = torch.t(torch.cat(y_future, dim=1))
        z_future = torch.cat(z_future)

        return y_sampled_future, z_future

    ###################################################
    # learning
    def rat_obj_func(self, x, z):
        '''expectation_policy [r]
            
            p(y=1|x,z) * r(y=1, x) + p(y=0|x,z)*r(y=0, x)
        '''
        prob_y_1_given_x_z = self.rat_policy(x, z)
        prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

        # r(action=1, x)
        r_y_1_x = self.rat_reward(torch.tensor([1], dtype=dtype, device=device), x)
        r_y_0_x = 1.0 - r_y_1_x

        return prob_y_1_given_x_z * r_y_1_x + prob_y_0_given_x_z * r_y_0_x

    def grad_rat_obj_score(self, y, x, z):
        '''
            grad log p(y|x, z)
            grad rat objective function using score function estimator
        '''
        prob_y_1_given_x_z = self.rat_policy(x, z)
        prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

        # r(action=1, x)
        r_y_1_x = self.rat_reward(torch.tensor([1.], dtype=dtype, device=device), x)
        r_y_0_x = 1.0 - r_y_1_x

        # grad of logistic regression: x_n(y_n - sigmoid(z^t x))
        y_1 = torch.ones(y.size(0), 1, dtype=dtype, device=device)
        grad_log_policy_y1 = self.grad_rat_policy(y_1, x, z)
        y_0 = torch.zeros(y.size(0), 1, dtype=dtype, device=device)
        grad_log_policy_y0 = self.grad_rat_policy(y_0, x, z)
        per_sample_gradient = prob_y_1_given_x_z * grad_log_policy_y1 * r_y_1_x + \
            prob_y_0_given_x_z * grad_log_policy_y0 * r_y_0_x

        avg_gradient = torch.mean(per_sample_gradient, dim=0)[None]

        return avg_gradient

    def grad_rat_obj_score_vec(self, y, x, z, smoothed_reward=False):
        '''
            many time points with one particle
            grad log p(y|x, z)
            grad rat objective function using score function estimator

            **************edit this function to be vectorized.


        '''
        prob_y_1_given_x_z = self.rat_policy_vec(x, z) # T x obs
        prob_y_0_given_x_z = 1.0 - prob_y_1_given_x_z

        # r(action=1, x)
        r_y_1_x = self.rat_reward_vec(torch.tensor([1.], dtype=dtype, device=device), x)
        r_y_0_x = 1.0 - r_y_1_x

        if smoothed_reward:
            rw = self.rat_reward_vec(y, x)
            k = 6
            weights = np.array([.1, .5, 1., 2., 4., 6])
            convolved_rw = (np.convolve(rw.detach().cpu().numpy().flatten(), weights)[:-k+1]) / float(np.sum(weights))

            #convolved_rw = (np.convolve(rw.detach().cpu().numpy().flatten(), np.ones(k))[:-k+1]) / float(k)
            prev_conv_rw = torch.ones(rw.shape[0], dtype=dtype, device=device)
            prev_conv_rw[1:] = torch.tensor(convolved_rw[:-1], dtype=dtype, device=device)
            r_y_1_x = r_y_1_x * prev_conv_rw[:, None]
            r_y_0_x = r_y_0_x * prev_conv_rw[:, None]

        # actual_rewards = self.rat_reward_vec(y, x)
        # rw_avg = np.convolve(actual_rewards, np.ones(10))/ 10.0

        # grad of logistic regression: x_n(y_n - sigmoid(z^t x))
        y_1 = torch.ones(y.size(0), y.size(1), dtype=dtype, device=device)
        grad_log_policy_y1 = self.grad_rat_policy_vec(y_1, x, z)

        y_0 = torch.zeros(y.size(0), y.size(1), dtype=dtype, device=device)
        grad_log_policy_y0 = self.grad_rat_policy_vec(y_0, x, z)

        per_sample_gradient = prob_y_1_given_x_z[:,:,None] * grad_log_policy_y1 * r_y_1_x[:,:,None] + \
            prob_y_0_given_x_z[:,:,None] * grad_log_policy_y0 * r_y_0_x[:,:,None]

        avg_gradient = torch.mean(per_sample_gradient, dim=1)
        return avg_gradient

    def rat_policy(self, x, z):
        '''
        p(y = 1 | x, z)
            z is 1 x 3
            x is num samples x dimension

            x is time x num samples x dimension
            z = time x dimension
        '''
        prob = self.sigmoid(torch.sum(x * z[:, None, :], dim=2))
        prob = torch.t(prob)
        assert prob.size(0) == x.size(0)
        assert prob.size(1) == 1
       
        return prob

    def rat_policy_vec(self, x, z):
        '''
        p(y = 1 | x, z)
            z is 1 x 3
            x is num samples x dimension

            x is time x num samples x dimension
            z = time x dimension
        '''
        prob = self.sigmoid(torch.sum(x * z[:, None, :], dim=2))

        #prob = torch.t(prob)
        assert prob.size(0) == x.size(0)
        assert prob.size(1) == x.size(1)
        return prob

    def rat_reward(self, action, x):
        '''
        rat's reward func
        action is {0,1}
        x is T x 2 
        assume this is preprocessing. we compute rewards ahead of time.
        '''
        stim1 = x[:, 1].unsqueeze(dim=1)
        stim2 = x[:, 2].unsqueeze(dim=1)
        rewards = ((stim1 > stim2)*(action==1) + (stim1 < stim2)*(action==0))
        return rewards.double()

    def rat_reward_vec(self, action, x):
        '''
        rat's reward func
        action is {0,1}
        x is T x 2 
        assume this is preprocessing. we compute rewards ahead of time.
        '''
        stim1 = x[:, :, 1]#.unsqueeze(dim=1)
        stim2 = x[:, :, 2]#.unsqueeze(dim=1)
        rewards = ((stim1 > stim2)*(action==1) + (stim1 < stim2)*(action==0))
        return rewards.double()

    def grad_rat_policy_vec(self, y, x, z):
        '''gradient of logistic regression
            grad log p(y|x, z)
            y: T x 1
            x: T x obs x dim
            z: T x dim
        '''
        prob = self.rat_policy_vec(x, z) # T x obs
        assert prob.size(0) == x.size(0)
        assert prob.size(1) == x.size(1)
        # this is the gradient: weight error by input features
        error = (y - prob)
        assert error.size(0) == x.size(0)
        assert error.size(1) == x.size(1)
        grad_log_policy = x * error[:, :, None] # T x num samples x dimension
        assert grad_log_policy.size(0) == x.size(0)
        assert grad_log_policy.size(1) == x.size(1)
        assert grad_log_policy.size(2) == x.size(2)
        return grad_log_policy
        # grad_log_policy_avg = torch.mean(grad_log_policy, dim=0)[None] # 1 x dimension
        # assert grad_log_policy_avg.size(-1) == z.size(-1)
        # return grad_log_policy_avg
    
    def grad_rat_policy(self, y, x, z):
        '''gradient of logistic regression
            grad log p(y|x, z)
        '''
        prob = self.rat_policy(x, z)
        assert prob.size(0) == x.size(0)
        assert prob.size(1) == 1
        # this is the gradient: weight error by input features
        error = (y - prob)
        assert error.size(0) == x.size(0)
        assert error.size(1) == 1
        grad_log_policy = x * error # T x num samples x dimension
        assert grad_log_policy.size(0) == x.size(0)
        assert grad_log_policy.size(1) == x.size(1)
        return grad_log_policy
        # grad_log_policy_avg = torch.mean(grad_log_policy, dim=0)[None] # 1 x dimension
        # assert grad_log_policy_avg.size(-1) == z.size(-1)
        # return grad_log_policy_avg



    def log_likelihood_test(self, y_train, y_test, test_inds, x, z):
        '''
        make sure we are taking the log of averages not average of logs
        e.g. for posterior predictive its the expectation of hte likelihood under posterior
        not expectation of log likelihood under posterior
        '''
        logits = torch.sum(x * z[:, None, :], dim=2)
        train_inds = self.return_train_ind(y_train)
        
        logits_train = logits[train_inds]
        obs = Bernoulli(logits=logits_train)
        train_log_ll = torch.sum(obs.log_prob(y_train[train_inds]))
        
        logits_test = logits[test_inds]
        obs = Bernoulli(logits=logits_test)
        test_log_ll = torch.sum(obs.log_prob(y_test))
        return test_log_ll


def print_memory():
    print("memory usage: ", (process.memory_info().rss)/(1e9))

