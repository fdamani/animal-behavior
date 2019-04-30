#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --mem 30G
#SBATCH --mail-user=fdamani@princeton.edu
#SBATCH -t 10:00:00
#SBATCH -o /tigress/fdamani/neuro_output/sbatch/console_output.txt
cd /home/fdamani/animal-behavior/semi-parametric/
source activate neuro
python main.py ind