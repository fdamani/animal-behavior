import os
import argparse

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
parser.add_argument('--print_every', type=int, default=500)


def get_args():
    args = parser.parse_args()
    embed()
    def cstr(arg, arg_name, default, custom_str=False):
        """ Get config str for arg, ignoring if set to default. """
        not_default = arg != default
        if not custom_str:
            custom_str = f'_{arg_name}{arg}'
        return custom_str if not_default else ''

    args.exp_name = (f'm{args.mean_num}_k{args.importance_num}'
                     f'{cstr(args.dataset, "", "stochmnist")}{cstr(args.arch, "", "bernoulli")}'
                     f'{cstr(args.seed, "seed", 42)}{cstr(args.batch_size, "bs", 20)}'
                     f'{cstr(args.h_dim, "h", 200)}{cstr(args.z_dim, "z", 50)}'
                     f'{cstr(args.learning_rate, "lr", 1e-3)}{cstr(args.analytic_kl, None, False, "_analytic")}'
                     f'{cstr(args.no_iwae_lr, None, False, "_noiwae")}{cstr(args.epochs, "epoch", 3280)}')

    args.figs_dir = os.path.join('figs', args.exp_name)
    args.out_dir = os.path.join('result', args.exp_name)
    args.best_model_file = os.path.join('result', args.exp_name, 'best_model.pt')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.figs_dir):
        os.makedirs(args.figs_dir)

    args.log_likelihood_k = 100 if args.dataset == 'cifar10' else 5000
    args.img_shape = (32, 32) if args.dataset == 'cifar10' else (28, 28)
    return args