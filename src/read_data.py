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
import models
from models import LDS, LogReg_LDS, LinearRegression
import inference
from inference import EM

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


def read_and_process(num_obs, f):
	raw_data = np.loadtxt(f, skiprows=1)
	header = open(f).readline()[:-1].split(",")
	raw_data = raw_data[11026:]

	num_trials = 100
	trials_per_session = np.bincount(raw_data[:,7].astype(int))[1:]
	sessions = np.where(trials_per_session > num_trials)[0] + 1
	data = raw_data[np.isin(raw_data[:, -1].astype(int), sessions)]
	data = data[~np.isnan(data[:,3])]

	data[np.where(data == 108.)], data[np.where(data == 114.)] = 0, 1

	data = np.concatenate([data[:,0:2], data[:,3:6]], axis=1)
	#data = data[0:1000]
	#data = data[0:10000]
	#data = data[1000:2000]
	x = data[:, 0:2]
	# log transform data 
	x = np.log(x)
	# standardize data
	x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

	# normalize x
	# x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
	x = np.concatenate([np.ones((data.shape[0],1)), x],axis=1)
	y = data[:,4]
	rw = data[:,2]

	# plot smoothed reward
	rw_avg = np.convolve(rw, np.ones(500))/ 500.0
	rw_avg = rw_avg[500:-500]
	# plt.plot(rw_avg)
	# plt.show()
	true_side = data[:,3]

	# create observations by binning over time
	x = featurize_x(x, y, rw, true_side)
	y = y[2:]
	rw = rw[2:]

	dim = x.shape[1]
	num_time_points = int(x.shape[0] / num_obs)
	total_obs = num_time_points * num_obs

	x_cut = x[0:total_obs]
	binned_x = x_cut.reshape(-1, num_obs, dim)
	y_cut = y[0:total_obs]
	binned_y = y_cut.reshape(-1, num_obs)

	rw_cut = rw[0:total_obs]
	binned_rw = rw_cut.reshape(-1, num_obs)

	return binned_x, binned_y, binned_rw

def featurize_x(x, y, rw, true_side):
	'''feature engineering.
		- previous 3 stimuli (x_t-3, x_t-2, x_t-1)
		-  previous 3 choices (x_t-3, x_t-2, x_t-1)
		-  rewards on left
		- rewards on right
	'''
	# choice history: -1 if choice was 0 (left), +1 if choice was 1 (right)
	choice = np.copy(y)
	choice[np.where(choice==0)[0]] = -1
	choice_m1 = np.roll(choice, 1)[2:][:, None]
	choice_m2 = np.roll(choice, 2)[2:][:, None]

	# reward side history: -1 if reward side was 0 (left), +1 if reward side was 1 (right)
	true_side[np.where(true_side==0)[0]] = -1
	true_side_m1 = np.roll(true_side, 1)[2:][:, None]
	true_side_m2 = np.roll(true_side, 2)[2:][:, None]

	# sensory history (last trial)
	sensory_hist = np.copy(x)[:, 1:]
	sensory_hist_m1 = np.roll(sensory_hist, shift=1, axis=0)[2:]
	sensory_hist_m2 = np.roll(sensory_hist, shift=2, axis=0)[2:]

	x  = x[2:]
	design_mat = np.concatenate([x, choice_m1, true_side_m1, sensory_hist_m1],axis=1)
	# design_mat = np.concatenate([x, true_side_m1],axis=1)
	# design_mat = np.concatenate([x],axis=1)

	#design_mat = np.concatenate([x, choice_m1, choice_m2, true_side_m1, true_side_m2],axis=1)
	return design_mat

