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
from evaluation import Evaluate, HeldOutRat
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
dtype = torch.double
#output_file = '/tigress/fdamani/neuro_output/5.5/test2'
output_file = '../output'
# lets compare a switching alpha model to a single alpha
f = '../data/'
#f = '/tigress/fdamani/neuro_data/data/raw/allrats_withmissing_limitedtrials/csv/'
rat = 'W074.csv'
f += rat
rat = f.split('/')[-1].split('.csv')[0]
num_obs_samples=1
output_file += '/'+rat
output_file += '_'+str(datetime.datetime.now())
os.makedirs(output_file)
x, y, rw = read_and_process(num_obs_samples, f, savedir=output_file)

T = 1000
num_particles = 500
x = x[0:T]
y = y[0:T]

x = torch.tensor(x, dtype=dtype, device=device)
y = torch.tensor(y, dtype=dtype, device=device)

dim = x.shape[-1]
md = LearningDynamicsModel(dim)
ev = HeldOutRat([y, x], md)


model_params_file = 'single_alpha_opt_params.pth'
if torch.cuda.is_available():
	model_params = torch.load(model_params_file)
else:
	model_params = torch.load(model_params_file, map_location='cpu')

single_alpha = ev.eval_particle_filter(model_params, T, num_particles)
print 'single alpha marginal lh: ', single_alpha
##########################

model_params_file = 'switching_alpha_opt_params.pth'
if torch.cuda.is_available():
	model_params = torch.load(model_params_file)
else:
	model_params = torch.load(model_params_file, map_location='cpu')

switching_alpha = ev.eval_particle_filter(model_params, T, num_particles, switching=True)
print 'switching_alpha marginal lh: ', switching_alpha
##########################


model_params_file = 'multiple_gamma_single_alpha_opt_params.pth'
if torch.cuda.is_available():
	model_params = torch.load(model_params_file)
else:
	model_params = torch.load(model_params_file, map_location='cpu')

mult_gamma = ev.eval_particle_filter(model_params, T, num_particles)
print 'mult gamma marginal lh: ', mult_gamma
#########################
embed()




mult_gamma_single_alpha_marginal_lh = ev.eval(model_params, T, 100, switching=False)

print 'mult gamma: ', single_marginal_lh
embed()



model_params = torch.load('/tigress/fdamani/neuro_output/block_residual_bootstrap/multiple_gamma_single_alpha/W065/model_structs/opt_params.pth')
mult_gamma_single_alpha_marginal_lh = ev.eval(model_params, T, 50, switching=False)
print 'mult gamma: ', single_marginal_lh
embed()


model_params = torch.load('/tigress/fdamani/neuro_output/5.5/single_alpha/W065/model_structs/opt_params.pth')
single_marginal_lh = ev.eval(model_params, T, 500, switching=False)
print 'single alpha: ', single_marginal_lh
embed()
model_params = torch.load('/tigress/fdamani/neuro_output/5.5/switching_alpha/W065_2019-05-05 13:14:57.854528/model_structs/opt_params.pth')
switching_marginal_lh = ev.eval(model_params, T, 500, switching=True)
print 'switching alpha: ', switching_marginal_lh

embed()


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
		fix, ax = plt.subplots()
		ax.boxplot(np.exp(boots[i]), showfliers=False)
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
	ax.boxplot(np.exp(np.transpose(boots.squeeze())), showfliers=False)
	ax.set_axisbelow(True)
	ax.set_xlabel('Rats')
	if vble == 'log_gamma':
		ax.set_ylabel('LOG Gamma')
	else:
		ax.set_ylabel('Alpha')
	figure = plt.gcf() # get current figure
	figure.set_size_inches(8, 6)
	plt.savefig(input_file+'/'+vble+'_'+'bootstrap.png', dpi=200)


