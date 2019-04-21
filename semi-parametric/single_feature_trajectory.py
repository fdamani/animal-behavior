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
import datetime
import utils
from utils import sigmoid
#dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
dtype = torch.float32
input_file = '/tigress/fdamani/neuro_output/exp1/'

import os
boots = []
rats = []
vble = 'var_mu'
#vble = 'log_alpha'
for file in os.listdir(input_file):
	try:
		x = torch.load(input_file+file+'/model_structs/opt_params.pth')
		rats.append(file)
	except:
		continue 
	x = x[vble].cpu().numpy()
	boots.append(x)

cutoff = 1000
num_features = boots[0].shape[1]
for i in range(num_features):
	trajectories = []
	for ds in boots:
		# if ds[:,i].shape[0] < cutoff:
		# 	continue
		trajectories.append(ds[:cutoff,i])
	plt.cla()
	for j,tj in enumerate(trajectories):
		plt.plot(tj[0:cutoff], color='blue', lw=.1)
	for_mean_trajectory = []
	for ds in boots:
		if ds[:,i].shape[0] < cutoff:
			continue
		for_mean_trajectory.append(ds[:cutoff,i])
	plt.plot(np.mean(np.stack(for_mean_trajectory), axis=0), color='blue', lw=1.)
		#if j == 5: break

	figure = plt.gcf() # get current figure
	figure.set_size_inches(8, 6)
	plt.xlabel('Trials')
	plt.ylabel('Posterior Mean')
	plt.savefig(input_file+vble+'_'+str(i)+'_'+'trajectory.png', dpi=200)


sys.exit(0)
if len(boots.shape) == 3:
	boots = boots.T
	for i in range(boots.shape[0]):
		embed()
		fix, ax = plt.subplots()
		ax.boxplot(np.exp(boots[i]))
		ax.set_axisbelow(True)
		ax.set_xlabel('Rats')
		if vble == 'log_gamma':
			ax.set_ylabel('Gamma')
		else:
			ax.set_ylabel('Alpha')
		figure = plt.gcf() # get current figure
		figure.set_size_inches(8, 6)
		plt.savefig(input_file+'/'+vble+'_'+str(i)+'_'+'bootstrap.png', dpi=200)

else:
	fix, ax = plt.subplots()
	ax.boxplot(np.exp(np.transpose(boots)))
	ax.set_axisbelow(True)
	ax.set_xlabel('Rats')
	if vble == 'log_gamma':
		ax.set_ylabel('Gamma')
	else:
		ax.set_ylabel('Alpha')
	figure = plt.gcf() # get current figure
	figure.set_size_inches(8, 6)
	plt.savefig(input_file+'/'+vble+'_'+'bootstrap.png', dpi=200)


