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

#output_file = sys.argv[1]
first_half = True
server = False
if server:
    output_file = '/tigress/fdamani/neuro_output/5.5/multiple_gamma_single_alpha_taketwo'
else:
    output_file = '../output/'
    # if first_half:
    #     output_file = '../output/l2_per_param_first_half/'
    # else:
    #     output_file = '../output/l2_per_param_second_half/'

def to_numpy(tx):
    return tx.detach().cpu().numpy()
def compute_mean_and_std(x):
    return x.mean(), x.std()

def simulate_datasets(model_params,
                      model_params_grad,
                      dim, 
                      num_obs_samples, 
                      num_datasets, datasets):
    '''
        sample rats with replacement
    '''
    num_samples = len(datasets)
    bootstrapped_datasets = [] # each bootstrapped dataset will be a collection of rats
    for i in range(num_datasets):
        sample = np.random.choice(np.arange(num_samples), size=num_samples)
        d = []
        for sx in sample:
            d.append(datasets[sx])
        bootstrapped_datasets.append(d)
    return bootstrapped_datasets

def estimation(datasets,
               boot_index,
               model_params,
               model_params_grad, 
               num_obs_samples, 
               num_future_steps, 
               category_tt_split,
               num_mc_samples,
               output_file,
               true_model_params=None):
    #proc_datasets = []
    # for dataset in datasets:
    #     y, x = dataset
    #     y_complete = y.clone().detach()
    #     y_complete = y_complete[0:-num_future_steps]
    #     category_tt_split = 'session'
    #     y, x, y_future, x_future = train_future_split(y, x, num_future_steps)
    #     y_train, y_test, test_inds = train_test_split(y.cpu(), x.cpu(), cat=category_tt_split)
    #     x = x.clone().detach() #torch.tensor(x, dtype=dtype, device=device)
    #     y_train = y_train.clone().detach() #torch.tensor(y_train, dtype=dtype, device=device)
    #     y_test = torch.tensor(y_test, dtype=dtype, device=device)
    #     test_inds = torch.tensor(test_inds, dtype=torch.long, device=device)
    #     y_future = y_future.clone().detach() #torch.tensor(y_future, dtype=dtype, device=device)
    #     x_future = x_future.clone().detach() #torch.tensor(x_future, dtype=dtype, device=device)

    #     y_train = torch.tensor(y, device=device)
    #     data = [y_train, x, y_test, test_inds, y_future, x_future, y_complete]
    #     proc_datasets.append(data)

    model = LearningDynamicsModel(dim=dim)

    boot_output_file = output_file+'/'+str(boot_index)
    os.makedirs(boot_output_file)
    os.makedirs(boot_output_file+'/model_structs')
    os.makedirs(boot_output_file+'/data')
    os.makedirs(boot_output_file+'/plots')
    
    inference = Inference(datasets=datasets, 
                          model=model,
                          model_params=model_params,
                          model_params_grad=model_params_grad,
                          savedir=boot_output_file,
                          num_obs_samples=num_obs_samples, 
                          num_future_steps=num_future_steps, 
                          num_mc_samples=num_mc_samples,
                          ppc_window=50,
                          z_true=z_true,
                          true_model_params=true_model_params) # pass in just for figures

    opt_params = inference.run()
    torch.save(opt_params, boot_output_file+'/model_structs/opt_params.npy')
    torch.save(dataset, boot_output_file+'/data/dataset.npy')
    torch.save(model_params, boot_output_file+'/model_structs/model_params.npy')
    return opt_params

if __name__ == '__main__':

    grad_latents = True
    grad_model_params = False
    inference_types = ['map', 'mfvi', 'is', 'smc', 'vsmc']
    inference_type = inference_types[4]
    sim = False
    file_path = '/tigress/fdamani/neuro_output/'
    # file_path = 'output/'
    savedir = file_path
    import datetime
    savedir += str(datetime.datetime.now())

    datafiles = ['W065.csv', 'W066.csv', 'W068.csv', 'W072.csv', 'W073.csv', 'W074.csv', 'W075.csv', 'W078.csv',
                  'W080.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv']
    # datafiles = ['W065.csv', 'W066.csv', 'W072.csv', 'W083.csv']
    # datafiles = ['W065.csv', 'W066.csv', 'W068.csv', 'W072.csv', 'W073.csv', 'W074.csv', 'W075.csv', 'W078.csv',
    #               'W080.csv', 'W081.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv', 'W094.csv']

    # datafiles = ['W074.csv', 'W075.csv', 'W078.csv', 'W080.csv', 'W081.csv', 'W082.csv', 
    #                 'W083.csv', 'W088.csv', 'W089.csv', 'W094.csv']

    # datafiles = ['W065.csv', 'W066.csv', 'W072.csv', 'W073.csv', 'W074.csv', 'W075.csv', 'W078.csv',
    #              'W080.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv']
    inds = np.arange(len(datafiles))
    leave_out_ind = int(sys.argv[1])
    inds = inds[inds!=leave_out_ind]
    num_obs_samples = 1
    output_file += 'shared_model_kfold_leave_out_'+str(leave_out_ind)
    #output_file += '_'+str(datetime.datetime.now())
    os.makedirs(output_file)
    os.makedirs(output_file+'/model_structs')
    os.makedirs(output_file+'/data')
    os.makedirs(output_file+'/plots')
    savedir = output_file
    raw_datasets = []
    for ind in inds:
        index = ind
        rat = datafiles[int(index)]
        print rat
        if torch.cuda.is_available():
            f = '/tigress/fdamani/neuro_data/data/raw/allrats_withmissing_limitedtrials/csv/'
        else:
            f = '../data/'
        f += rat
        rat = f.split('/')[-1].split('.csv')[0]
        

        x, y, rw = read_and_process(num_obs_samples, f, savedir=savedir, proficient=False)
        # x = x[0:500]
        # y = y[0:500]
        # rw = rw[0:500]
        rw = torch.tensor(rw, dtype=dtype, device=device)
        z_true = None
        true_model_params=None


        dim = x.shape[2]
        T = x.shape[0]
        raw_datasets.append((x, y, rw, z_true, true_model_params, dim, T, rat))
    

    datasets = []
    for dataset in raw_datasets:
        x, y, rw, z_true, true_model_params, dim, T, rat = dataset
        # split data into train/test
        ############ initial estimation 
        num_future_steps = 1
        category_tt_split = 'session'
        num_mc_samples = 10
        ppc_window = 50
        percent_test = .2
        features = ['Bias', 'X1', 'X2', 'Choice t-1', 'RW Side t-1', 'X1 t-1', 'X2 t-1']

        y_complete = torch.tensor(y.copy(), device=device)
        y_complete = y_complete[0:-num_future_steps]

        #y, x, y_future, x_future = train_future_split(y, x, num_future_steps)
        y_future = y
        x_future = x
        y_train, y_test, test_inds = train_test_split(y, x, category_tt_split, percent_test)
        x = torch.tensor(x, dtype=dtype, device=device)
        y_train = torch.tensor(y_train, dtype=dtype, device=device)
        y_test = torch.tensor(y_test, dtype=dtype, device=device)
        test_inds = torch.tensor(test_inds, dtype=torch.long, device=device)
        y_future = torch.tensor(y_future, dtype=dtype, device=device)
        x_future = torch.tensor(x_future, dtype=dtype, device=device)
        #data = [y_train, x]
        data = [y_train, x, y_test, test_inds, y_future, x_future, y_complete]
        datasets.append(data)
    
    model_params_grad = {'init_latent_loc': False,
                    'init_latent_log_scale': False,
                    'transition_log_scale': False,
                    'log_gamma': False,
                    'beta': False,
                    'log_alpha': True}
    model_params = {'init_latent_loc': torch.tensor([0.0]*dim, dtype=dtype,  device=device, requires_grad=model_params_grad['init_latent_loc']),
                    'init_latent_log_scale': torch.tensor([math.log(1.0)]*dim, dtype=dtype, device=device, requires_grad=model_params_grad['init_latent_log_scale']),
                    'transition_log_scale': torch.tensor([math.log(.05)], dtype=dtype, device=device, requires_grad=model_params_grad['transition_log_scale']),
                    'log_gamma': torch.tensor([math.log(.05)]*dim, dtype=dtype, device=device, requires_grad=model_params_grad['log_gamma']),
                    'beta': torch.tensor([100.], dtype=dtype, device=device, requires_grad=model_params_grad['beta']),
                    'log_alpha': torch.tensor([math.log(.05)], dtype=dtype, device=device, requires_grad=model_params_grad['log_alpha'])}
    # [math.log(.05)]*2
    torch.save(model_params_grad, output_file+'/model_structs/model_params_grad.pth')
    torch.save(model_params, output_file+'/model_structs/init_model_params.pth')

    model = LearningDynamicsModel(dim)
    inference = Inference(datasets,
                          model,
                          model_params, 
                          model_params_grad, 
                          savedir=output_file, 
                          num_obs_samples=num_obs_samples, 
                          num_future_steps=num_future_steps, 
                          num_mc_samples=num_mc_samples,
                          ppc_window=ppc_window, 
                          z_true=z_true,
                          true_model_params=None) # pass in just for figures
    
    opt_params = inference.run()

    final_loss = 0
    for d in range(len(inference.datasets)):
        y, x = inference.datasets[d][0], inference.datasets[d][1]
        num_train = inference.datasets[d][0].shape[0] - inference.datasets[d][2].shape[0]
        final_loss += -inference.vi.forward_multiple_mcs(model_params, (y,x), inference.var_params[d], 50, num_samples=100) / float(num_train)

    #final_loss = -inference.vi.forward_multiple_mcs(model_params, inference.datasets, inference.var_params, 50, num_samples=100) #/ float(inference.num_train)
    
    k = float(dim+1)
    bic = (2 * final_loss.item() + k * np.log(T * num_obs_samples)) / float(T * num_obs_samples)

    np.savetxt(output_file+'/training_bic.txt', np.array([bic]))
    np.savetxt(output_file+'/training_elbo.txt', np.array([final_loss.item()]))
    print final_loss.item(), bic

    for k,v in model_params_grad.items():
        if v == False:
            opt_params[k] = model_params[k]

    torch.save(opt_params, output_file+'/model_structs/opt_params.pth')
    torch.save(data, output_file+'/data/data.pth')

    ################### bootstrap ################################################
    num_datasets = 3
    sim_datasets = simulate_datasets(opt_params, model_params_grad, dim, num_obs_samples, num_datasets, datasets)
    bootstrapped_params = {'init_latent_loc': [],
                            'init_latent_log_scale': [],
                           'transition_log_scale': [],
                           'log_gamma': [],
                           'beta': [],
                           'log_alpha': []}
    for ind in range(num_datasets):
        print 'bootstrap: ', ind
        estimated_params = estimation(datasets=sim_datasets[ind], 
                                      boot_index=ind, 
                                      model_params=model_params, 
                                      model_params_grad=model_params_grad, 
                                      num_obs_samples=num_obs_samples, 
                                      num_future_steps=num_future_steps,
                                      category_tt_split=category_tt_split,
                                      num_mc_samples=num_mc_samples,
                                      output_file=output_file,
                                      true_model_params=opt_params) # true is initial params fit to data.
        for k,v in estimated_params.items():
            if k in bootstrapped_params:
                bootstrapped_params[k].append(v)

    for k,v in bootstrapped_params.items():
        # if param was estimated in model
        if model_params_grad[k]:
            plt.cla()
            vx = torch.stack(v).squeeze(dim=-1)
            #vx = [to_numpy(el) for el in v]
            fix, ax = plt.subplots()
            ax.boxplot(to_numpy(vx))
            ax.set_axisbelow(True)
            ax.set_xlabel('Feature')
            ax.set_ylabel(k)
            # plot theta hat
            if len(opt_params[k]) != len(features):
                ax.scatter(np.arange(to_numpy(opt_params[k]).shape[0]), to_numpy(opt_params[k]), color='b')
            else:
                ax.set_xticklabels(features)
                plt.scatter(features, to_numpy(opt_params[k]), color='b')
            # mu, std = compute_mean_and_std(np.array(v))
            # plt.errorbar(x=np.arange(1), y=mu, yerr=2*std, fmt='o')
            # plot theta hat
           #  plt.axhline(y=opt_params[k], color='r', linestyle='--')
            # plot theta star if exists
            # if sim:
            #     plt.axhline(y=true_model_params[k], color='b', linestyle='--')
            figure = plt.gcf() # get current figure
            figure.set_size_inches(8, 6)
            plt.savefig(output_file+'/plots/'+k+'confidence.png')
    torch.save(bootstrapped_params, output_file+'/model_structs/bootstrapped_params.npy')
  