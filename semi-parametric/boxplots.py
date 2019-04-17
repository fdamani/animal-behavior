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
#vble = 'log_gamma'
vble = 'log_alpha'
for file in os.listdir(input_file):
	rats.append(file)
	try:
		x = torch.load(input_file+file+'/model_structs/bootstrapped_params.npy')
	except:
		continue 
	x = torch.stack(x[vble]).cpu().numpy()
	boots.append(x)
plt.cla()
boots = np.array(boots)
if len(boots.shape) == 3:
	boots = boots.T
	for i in range(boots.shape[0]):
		embed()
		fix, ax = plt.subplots()
		ax.boxplot(np.exp(boots[i]))
		ax.set_axisbelow(True)
		ax.set_xlabel('Rats')
		ax.set_ylabel('Alpha')
		figure = plt.gcf() # get current figure
		figure.set_size_inches(8, 6)
		plt.savefig(input_file+'/'+vble+'_'+str(i)+'_'+'bootstrap.png', dpi=200)

else:
	fix, ax = plt.subplots()
	ax.boxplot(np.exp(np.transpose(boots)))
	ax.set_axisbelow(True)
	ax.set_xlabel('Rats')
	ax.set_ylabel('Alpha')
	figure = plt.gcf() # get current figure
	figure.set_size_inches(8, 6)
	plt.savefig(input_file+'/'+vble+'_'+'bootstrap.png', dpi=200)


