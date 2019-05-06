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
input_file = '/tigress/fdamani/neuro_output/5.5/switching_alpha/'

import os
boots = []
rats = []
#vble = 'log_gamma'
vble = 'log_alpha'
#vble = 'transition_log_scale'
for file in os.listdir(input_file):
	rats.append(file)
	try:
		x = torch.load(input_file+file+'/model_structs/opt_params.pth')
	except:
		continue 
	if vble == 'transition_log_scale':
		x = x[vble]
	else:
		x = x[vble].cpu().numpy()
	boots.append(x)
boots = np.array(boots)
fix, ax = plt.subplots()
if vble == 'log_gamma':
	ax.boxplot(boots.squeeze(), showfliers=False)
else:
	ax.boxplot(boots.squeeze(), showfliers=False)
ax.set_axisbelow(True)
ax.set_xlabel('Features')
if vble == 'log_gamma':
	ax.set_ylabel('Log Gamma')
elif vble == 'log_alpha':
	ax.set_ylabel('Log Alpha')
else:
	ax.set_ylabel('Scale')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig(input_file+'/'+vble+'_'+'boxplot_mle_across_rats.png', dpi=200)

