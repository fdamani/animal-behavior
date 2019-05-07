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

f = '/tigress/fdamani/neuro_output/5.5/switching_alpha_shared_model_kfold_leave_out_6/model_structs/opt_params.pth'
model_params = torch.load(f)
f = '/tigress/fdamani/neuro_output/5.5/switching_alpha_shared_model_kfold_leave_out_6/data/data.pth'
data = torch.load(f)
embed()