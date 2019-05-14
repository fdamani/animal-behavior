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

import inference_multiple_rats
from inference_multiple_rats import Inference, MeanFieldVI

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
#dtype = torch.float32
dtype = torch.double

f = '../output/switching_alpha_shared_model_kfold_leave_out_6/model_structs/opt_params.pth'
#f = '../output/single_alpha_shared_model_kfold_leave_out_6/model_structs/opt_params.pth'
model_params = torch.load(f, map_location='cpu')
f = '../output/switching_alpha_shared_model_kfold_leave_out_6/data/data.pth'
#f = '../output/single_alpha_shared_model_kfold_leave_out_6/data/data.pth'
data = torch.load(f, map_location='cpu')
y, x = data[0], data[1]
model = LearningDynamicsModel(dim=7)
z_switch = []
z_switch.append(model.sample_init_prior(model_params))
# set diffusion noise to small value
model_params['transition_log_scale'] = torch.tensor([-20.], dtype=dtype, device=device)
for i in range(1,10000):
	z_switch.append(model.sample_prior(model_params, z_switch[i-1], y[i-1], x[i-1], switching=True))
z_switch = torch.stack(z_switch).numpy().squeeze()

torch.manual_seed(10)
np.random.seed(7)
#f = '../output/switching_alpha_shared_model_kfold_leave_out_6/model_structs/opt_params.pth'
f = '../output/single_alpha_shared_model_kfold_leave_out_6/model_structs/opt_params.pth'
model_params = torch.load(f, map_location='cpu')
#f = '../output/switching_alpha_shared_model_kfold_leave_out_6/data/data.pth'
f = '../output/single_alpha_shared_model_kfold_leave_out_6/data/data.pth'
data = torch.load(f, map_location='cpu')
y, x = data[0], data[1]
model = LearningDynamicsModel(dim=7)
z = []
z.append(model.sample_init_prior(model_params))
# set diffusion noise to small value
model_params['transition_log_scale'] = torch.tensor([-8.], dtype=dtype, device=device)
for i in range(1,10000):
	z.append(model.sample_prior(model_params, z[i-1], y[i-1], x[i-1], switching=False))
z = torch.stack(z).numpy().squeeze()

plt.plot(z_switch[:,0], color='blue')
plt.plot(z[:,0], color='blue', linestyle='--')

plt.plot(z_switch[:,1], color='red')
plt.plot(z[:,1], color='red', linestyle='--')

plt.plot(z_switch[:,2], color='black')
plt.plot(z[:,2], color='black', linestyle='--')

plt.show()
embed()