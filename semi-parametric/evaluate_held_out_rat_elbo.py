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
datafiles = ['W065.csv', 'W066.csv', 'W068.csv', 'W072.csv', 'W073.csv', 'W074.csv', 'W075.csv', 'W078.csv',
              'W080.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv']
# f = '/tigress/fdamani/neuro_data/data/raw/allrats_withmissing_limitedtrials/csv/'
# rat = 'W065.csv'
# f += rat
# rat = f.split('/')[-1].split('.csv')[0]
# num_obs_samples=1
#output_file += '/'+rat
output_file += '/kfold_single_alpha_results_2k'
os.makedirs(output_file)

folds = np.arange(1, len(datafiles))
held_out_marginal_lhs = []
for fd in folds:
	rat = datafiles[fd]
	f = '/tigress/fdamani/neuro_data/data/raw/allrats_withmissing_limitedtrials/csv/'
	#f = '../data/'
	#rat = 'W073.csv'
	f += rat
	num_obs_samples=1
	x, y, rw = read_and_process(num_obs_samples, f, savedir=output_file)
	T = 2000
	#T = 50
	#num_particles = 100
	num_particles = 1000
	x = x[0:T]
	y = y[0:T]

	x = torch.tensor(x, dtype=dtype, device=device)
	y = torch.tensor(y, dtype=dtype, device=device)
	data = [y, x, y, x, y, x, y]

	dim = x.shape[-1]
	md = LearningDynamicsModel(dim)


	ev = HeldOutRat([y, x], md)

	model_params_file = '/tigress/fdamani/neuro_output/5.5/single_alpha_shared_model_kfold_leave_out_'+str(int(fd))+'/model_structs/opt_params.pth'
	
	num_obs_samples=1
	
	if torch.cuda.is_available():
		try:
			model_params = torch.load(model_params_file)
		except:
			continue
	else:
		model_params = torch.load(model_params_file, map_location='cpu')



    inference = Inference(data,
                          md,
                          model_params, 
                          model_params_grad, 
                          savedir=output_file, 
                          num_obs_samples=num_obs_samples, 
                          num_future_steps=10, 
                          num_mc_samples=1,
                          ppc_window=50, 
                          z_true=None,
                          true_model_params=None) # pass in just for figures
    opt_params = inference.run()
    embed()










	switching_alpha = ev.eval_particle_filter(model_params, T, num_particles, False, output_file, fd) # -263.3014, -.526
	held_out_marginal_lhs.append(switching_alpha.item())
	print 'single alpha marginal lh: ', switching_alpha.item(), fd
	np.savetxt(output_file+'/held_out_rat_marginal_lh.txt', np.array([held_out_marginal_lhs]))

