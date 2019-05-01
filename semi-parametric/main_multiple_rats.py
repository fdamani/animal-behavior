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
dtype = torch.float32

#output_file = sys.argv[1]
first_half = True
server = True
if server:
    output_file = '/tigress/fdamani/neuro_output/multiple_rats/test'
else:
    output_file = ''
    if first_half:
        output_file = '../output/l2_per_param_first_half/'
    else:
        output_file = '../output/l2_per_param_second_half/'
# output_file = output_file + '_'+str(datetime.datetime.now())
# os.makedirs(output_file)
# os.makedirs(output_file+'/model_structs')
# os.makedirs(output_file+'/data')
# os.makedirs(output_file+'/plots')

def to_numpy(tx):
    return tx.detach().cpu().numpy()
def compute_mean_and_std(x):
    return x.mean(), x.std()
def simulate_datasets(model_params,
                      model_params_grad,
                      dim, 
                      num_obs_samples, 
                      num_datasets, data):
    '''
        block residual bootstrap.
    '''
    # instantiate model
    y_train, x, y_test, test_inds, y_future, x_future, y_complete = data
    T = y_train.shape[0]
    num_blocks = 100.
    num_obs_per_block = int(T / num_blocks)
    block_inds = np.arange(0, T, step=num_obs_per_block)[:-1]
    datasets = []
    for i in range(num_datasets):
        sample = np.random.choice(block_inds, size=int(num_blocks))
        y_b = torch.stack([y_train[sam:sam+num_obs_per_block] for sam in sample]).reshape(-1,num_obs_samples)
        x_b = torch.stack([x[sam:sam+num_obs_per_block] for sam in sample]).reshape(-1, num_obs_samples, dim)
        datasets.append((y_b, x_b))
    return datasets

    # data = data
    # model = LearningDynamicsModel(model_params, model_params_grad, dim=dim)
    # num_obs = 50
    # datasets = []
    # for i in range(num_datasets):
    #     y, x, z_true = model.sample(T=T, num_obs_samples=num_obs_samples, dim=dim)
    #     datasets.append((y, x, z_true))
    # return datasets

def estimation(dataset,
               boot_index,
               model_params,
               model_params_grad, 
               num_obs_samples, 
               num_future_steps, 
               category_tt_split,
               num_mc_samples,
               output_file,
               true_model_params=None):
    y, x = dataset
    y_complete = y.clone().detach()
    y_complete = y_complete[0:-num_future_steps]
    category_tt_split = 'session'
    y, x, y_future, x_future = train_future_split(y, x, num_future_steps)
    y_train, y_test, test_inds = train_test_split(y.cpu(), x.cpu(), cat=category_tt_split)
    x = x.clone().detach() #torch.tensor(x, dtype=dtype, device=device)
    y_train = y_train.clone().detach() #torch.tensor(y_train, dtype=dtype, device=device)
    y_test = torch.tensor(y_test, dtype=dtype, device=device)
    test_inds = torch.tensor(test_inds, dtype=torch.long, device=device)
    y_future = y_future.clone().detach() #torch.tensor(y_future, dtype=dtype, device=device)
    x_future = x_future.clone().detach() #torch.tensor(x_future, dtype=dtype, device=device)

    y_train = torch.tensor(y, device=device)
    data = [y_train, x, y_test, test_inds, y_future, x_future, y_complete]


    model = LearningDynamicsModel(model_params, model_params_grad, dim)

    boot_output_file = output_file+'/'+str(boot_index)
    os.makedirs(boot_output_file)
    os.makedirs(boot_output_file+'/model_structs')
    os.makedirs(boot_output_file+'/data')
    os.makedirs(boot_output_file+'/plots')
    
    inference = Inference(data=data, 
                          model=model, 
                          model_params_grad=model_params_grad,
                          savedir=boot_output_file,
                          num_obs_samples=num_obs_samples, 
                          num_future_steps=num_future_steps, 
                          num_mc_samples=num_mc_samples,
                          ppc_window=50,
                          z_true=z_true,
                          true_model_params=true_model_params) # pass in just for figures

    opt_params = inference.em()
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
                  'W080.csv', 'W081.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv', 'W094.csv']

    # datafiles = ['W074.csv', 'W075.csv', 'W078.csv', 'W080.csv', 'W081.csv', 'W082.csv', 
    #                 'W083.csv', 'W088.csv', 'W089.csv', 'W094.csv']

    # datafiles = ['W065.csv', 'W066.csv', 'W072.csv', 'W073.csv', 'W074.csv', 'W075.csv', 'W078.csv',
    #              'W080.csv', 'W082.csv', 'W083.csv', 'W088.csv', 'W089.csv']
    # 78 is not bad, 82 is not bad, 88 is a maybe
    if sim:
        output_file = output_file + '_'+str(datetime.datetime.now())
        os.makedirs(output_file)
        os.makedirs(output_file+'/model_structs')
        os.makedirs(output_file+'/data')
        os.makedirs(output_file+'/plots')
        # T = 200 # 100
        T = 1000
        # time-series model
        # sim model parameters
        dim = 3
        init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
        transition_log_scale = [math.log(5e-2)]# * dim
        log_gamma = math.log(1e-10) # 0.08
        #log_gamma = math.log(0.00000004)
        
        beta = 100. # try -3, 0.
        log_alpha = math.log(.1) # .1
        
        true_model_params = {'init_prior': init_prior,
                        'transition_log_scale': transition_log_scale,
                        'log_gamma': log_gamma,
                        'beta': beta,
                        'log_alpha': log_alpha}
        torch.save(true_model_params, output_file+'/model_structs/true_model_params.pth')
        model_params_grad = {'init_prior': False,
                'transition_log_scale': False,
                'log_gamma': True,
                'beta': False,
                'log_alpha': False}
        model = LearningDynamicsModel(true_model_params, model_params_grad, dim=3)
        num_obs_samples = 500
        y, x, z_true = model.sample(T=T, num_obs_samples=num_obs_samples)
        
        rw = torch.mean(model.rat_reward_vec(y, x), dim=1)
        # window=100
        # rw_avg = np.convolve(rw, np.ones(window))/ float(window)
        #rw_avg = rw_avg[window:-window]
        # plt.plot(rw_avg)
        # plt.show()

        y = y.detach().cpu().numpy()
        x = x.detach().cpu().numpy()
        z_true = z_true.detach().cpu().numpy()
        plt.cla()
        plt.plot(z_true)
        plt.savefig(output_file+'/plots/sim_z.png')

        # rw = torch.mean(model.rat_reward_vec(torch.tensor(y, device=device), torch.tensor(x,device=device)), dim=1)
        # ppc_window=100
        # rw_avg = np.convolve(to_numpy(rw), np.ones(window))/ float(window)
        # rw_avg = rw_avg[window:-window]
        # plt.cla()
        # plt.plot(rw_avg)
        # plt.show()

        # model params
    else:
        num_obs_samples = 100
        #f = '/tigress/fdamani/neuro_data/data/clean/LearningData_W066_minmaxnorm.txt'
        # data file
        inds = [1, 2, 3, 4, 5]
        output_file += '_'+str(datetime.datetime.now())
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
            
            # add to dir name
            #output_file += '/'+rat
            # output_file += '__obs'+str(num_obs_samples)

            #output_file += '_'+str(datetime.datetime.now())
            # os.mkdir(savedir)
            #f = '../data/W066_short.csv'
            #savedir = output_file
       
            x, y, rw = read_and_process(num_obs_samples, f, savedir=savedir)
            # half = int(x.shape[0]/2)
            # if first_half:
            #     x = x[:half]
            #     y = y[:half]
            #     rw = y[:half]
            # else:
            #     x = x[half:]
            #     y = y[half:]
            #     rw = y[half:]
            rw = torch.tensor(rw, dtype=dtype, device=device)
            z_true = None
            true_model_params=None

            #x = x[0:10000]
            #y = y[0:10000]

            # rnn_hiddens = torch.load('hiddens_rnn_7_features.pt')
            # rnn_hiddens = rnn_hiddens.detach().to(device)
            # x = rnn_hiddens[:, None, :]
            # x = x[:, 0, :][:, None, :]
            # y = y[:, 0][:, None]

            dim = x.shape[2]
            T = x.shape[0]

            raw_datasets.append((x, y, rw, z_true, true_model_params, dim, T, rat))

    datasets = []
    for dataset in raw_datasets:
        x, y, rw, z_true, true_model_params, dim, T, rat = dataset

        # split data into train/test
        ############ initial estimation 
        num_future_steps = 1
        category_tt_split = 'single'
        num_mc_samples = 10
        ppc_window = 50
        percent_test = .01
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
    cv_bool = False
    if cv_bool:
        # parameter sweep over scale
        transition_log_scales = [1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 7e-2, 1e-1]
        #transition_log_scales = [3e-2, 5e-2, 7e-2]

        test_marginals = []
        for tls in transition_log_scales:
            print tls
            # model params
            init_transition_log_scale = [math.log(tls)]# * dim
            #init_transition_log_scale = []
            init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
            log_gamma = [math.log(1e-20)]#*dim# .08 1e-10
            beta = 100. # sigmoid(4.) = .9820
            log_alpha = [math.log(1e-20)]#*dim # .1

            model_params = {'init_prior': init_prior,
                            'transition_log_scale': init_transition_log_scale,
                            'log_gamma': log_gamma,
                            'beta': beta,
                            'log_alpha': log_alpha}
            
            model_params_grad = {'init_prior': False,
                            'transition_log_scale': False,
                            'log_gamma': False,
                            'beta': False,
                            'log_alpha': False}

            torch.save(model_params_grad, output_file+'/model_structs/model_params_grad.pth')
            torch.save(model_params, output_file+'/model_structs/init_model_params.pth')

            model = LearningDynamicsModel(model_params, model_params_grad, dim)
            inference = Inference(datasets, 
                                  model, 
                                  model_params_grad, 
                                  savedir=output_file, 
                                  num_obs_samples=num_obs_samples, 
                                  num_future_steps=num_future_steps, 
                                  num_mc_samples=num_mc_samples,
                                  ppc_window=ppc_window, 
                                  z_true=z_true,
                                  true_model_params=true_model_params) # pass in just for figures
            
            map_params = inference.init_z
            ev = Evaluate(data, model, savedir='', num_obs_samples=num_obs_samples)
            test_marginal = ev.valid_loss_map(map_params)
            print test_marginal
            test_marginals.append(test_marginal)
        init_transition_log_scale = [math.log(transition_log_scales[np.argmax(test_marginals)])]
    else:
        init_transition_log_scale = [math.log(.05)]# * dim

    #init_transition_log_scale = []
    init_prior = ([0.0]*dim, [math.log(1.0)]*dim)
    log_gamma = [math.log(.01)]*dim# .08 1e-10
    beta = 100. # sigmoid(4.) = .9820
    log_alpha = [math.log(.01)]#*dim # .1

    model_params = {'init_prior': init_prior,
                    'transition_log_scale': init_transition_log_scale,
                    'log_gamma': log_gamma,
                    'beta': beta,
                    'log_alpha': log_alpha}
    
    model_params_grad = {'init_prior': False,
                    'transition_log_scale': False,
                    'log_gamma': False,
                    'beta': False,
                    'log_alpha': True}

    torch.save(model_params_grad, output_file+'/model_structs/model_params_grad.pth')
    torch.save(model_params, output_file+'/model_structs/init_model_params.pth')

    model = LearningDynamicsModel(model_params, model_params_grad, dim)
    inference = Inference(datasets, 
                          model, 
                          model_params_grad, 
                          savedir=output_file, 
                          num_obs_samples=num_obs_samples, 
                          num_future_steps=num_future_steps, 
                          num_mc_samples=num_mc_samples,
                          ppc_window=ppc_window, 
                          z_true=z_true,
                          true_model_params=true_model_params) # pass in just for figures
    
    opt_params = inference.em()

    final_loss = -inference.vi.forward_multiple_mcs(inference.train_data, inference.var_params, 50, num_samples=100) #/ float(inference.num_train)
    
    k = float(2*dim)
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
    num_datasets = 25
    sim_datasets = simulate_datasets(opt_params, model_params_grad, dim, num_obs_samples, num_datasets, data)
    bootstrapped_params = {'init_prior': [],
                           'transition_log_scale': [],
                           'log_gamma': [],
                           'beta': [],
                           'log_alpha': []}
    for ind in range(num_datasets):
        print 'bootstrap: ', ind
        estimated_params = estimation(dataset=sim_datasets[ind], 
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
            embed()
            ax.boxplot(to_numpy(vx))
            ax.set_axisbelow(True)
            ax.set_xlabel('Feature')
            ax.set_ylabel(k)
            # plot theta hat
            if len(opt_params[k]) == 1:
                ax.scatter(1, to_numpy(opt_params[k]), color='b')
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


    # y_future, z_future, avg_future_marginal_lh = inference.ev.sample_future_trajectory(inference.var_params, num_future_steps)
    # if sim:
    #     plt.plot(z_true[-num_future_steps:])
    # plt.plot(to_numpy(z_future))
    # plt.show()
    #plt.show()
    # ev = Evaluate(data, model, savedir='', num_obs_samples=num_obs_samples)
    # train_ll, test_ll = ev.valid_loss()
    # then test!

  