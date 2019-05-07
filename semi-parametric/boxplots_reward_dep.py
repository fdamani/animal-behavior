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
#input_file = '/tigress/fdamani/neuro_output/5.5/switching_alpha/'
input_file = '../output/5.5/switching_alpha/'

import os
boots = []
data_stat = []
rats = []
#vble = 'log_gamma'
vble = 'log_alpha'
for file in os.listdir(input_file):
	rats.append(file)
	try:
		if torch.cuda.is_available():
			x = torch.load(input_file+file+'/model_structs/bootstrapped_params.npy')
			y = torch.load(input_file+file+'/model_structs/opt_params.pth')
		else:
			x = torch.load(input_file+file+'/model_structs/bootstrapped_params.npy', map_location='cpu')
			y = torch.load(input_file+file+'/model_structs/opt_params.pth', map_location='cpu')
	except:
		continue 
	x = torch.stack(x[vble]).cpu().numpy()
	y = y[vble].cpu().numpy()
	boots.append(x)
	data_stat.append(y)
plt.cla()
boots = np.array(boots)
data_stat = np.array(data_stat)
standard_errors = []

# for each rat
for i in range(boots.shape[0]):
	# compute standard error
	standard_errors.append(np.std(boots[i],axis=0)) # of length number of parameters
standard_errors = np.array(standard_errors)
plt.errorbar(np.arange(data_stat.shape[0]), data_stat[:,0], 2*standard_errors[:,0], linestyle='None', fmt='-o')
plt.savefig(input_file+'/'+vble+'_feature0_'+'confidence_intervals.png', dpi=200)
plt.cla()
plt.errorbar(np.arange(data_stat.shape[0]), data_stat[:,1], 2*standard_errors[:,1], linestyle='None', fmt='-o')
plt.savefig(input_file+'/'+vble+'_feature1'+'confidence_intervals.png', dpi=200)
sys.exit(0)
plt.show()
embed()
#plt.errorbar([0,1], data_stat[0], 2*sx, linestyle='None', fmt='-o')
if len(boots.shape) == 3:
	boots = boots.T
	for i in range(boots.shape[0]):
		fix, ax = plt.subplots()
		ax.boxplot(boots[i], showfliers=False)
		ax.set_axisbelow(True)
		ax.set_xlabel('Rats')
		if vble == 'log_gamma':
			ax.set_ylabel('Gamma')
		else:
			ax.set_ylabel('Log Alpha')
		figure = plt.gcf() # get current figure
		figure.set_size_inches(8, 6)
		plt.savefig(input_file+'/'+vble+'_'+str(i)+'_'+'bootstrap.png', dpi=200)

else:
	fix, ax = plt.subplots()
	ax.boxplot(np.exp(np.transpose(boots.squeeze())), showfliers=False)
	ax.set_axisbelow(True)
	ax.set_xlabel('Rats')
	if vble == 'log_gamma':
		ax.set_ylabel('LOG Gamma')
	else:
		ax.set_ylabel('Log Alpha')
	figure = plt.gcf() # get current figure
	figure.set_size_inches(8, 6)
	plt.savefig(input_file+'/'+vble+'_'+'bootstrap.png', dpi=200)


