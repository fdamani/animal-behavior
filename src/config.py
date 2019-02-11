import os
import argparse
from IPython import embed
parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=bool, default=True)
#parser.add_argument('--data_file', type=str, default='../data/data.txt')
parser.add_argument('--sim_time_steps', type=int, default=1000)
parser.add_argument('--grad_model_params', type=bool, default=False)
parser.add_argument('--grad_latents', type=bool, default=True)
parser.add_argument('--model', type=str, default='LDS',
    choices = ['LDS', 'LogReg_LDS', 'LearningDynamicsModel'])


parser.add_argument('--inference_method', type=str, default='mfvi',  # need to have option to include list here--we might want to run multiple inf methods
    choices=['map', 'mfvi', 'tri_diag_vi', 'iwae', 'fivo', 'smc'])

parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--print_every', type=int, default=250)


def get_args():
    args = parser.parse_args() 
    return args