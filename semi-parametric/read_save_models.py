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

def to_numpy(tx):
    return tx.detach().cpu().numpy()
x = torch.load('/tigress/fdamani/neuro_output/W073/model_structs/bootstrapped_params.npy')
x_log_gamma = x['log_gamma']
v = torch.stack(x_log_gamma)


plt.cla()
vx = v
#vx = [to_numpy(el) for el in v]
fix, ax = plt.subplots()
ax.boxplot(to_numpy(vx))
ax.set_axisbelow(True)
ax.set_xlabel('Feature')
figure = plt.gcf() # get current figure
figure.set_size_inches(8, 6)
plt.savefig('/tigress/fdamani/neuro_output/W073/log_gammaconfidence.png')