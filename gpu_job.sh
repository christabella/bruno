#!/bin/bash

# What works:
# conda activate bruno_v2
# srun --gres=gpu:1 --mem=4G python3 -m config_conditional.train  --config_name m1_shapenet_glow --nr_gpu 1

#SBATCH --time=30:00:00
#SBATCH --gres=gpu:2        ## one K80 requested
#SBATCH --mem=3G            ## one K80 requested

# srun python3 -m config_conditional.train  --config_name m1_shapenet_glow --nr_gpu 2
# srun --time=0:10:00 --gres=gpu:1 --mem=2G guild run -y conditional_bruno:train config_name=m1_shapenet_glow
srun --time=30:00:00 --gres=gpu:2 --mem=2G guild run -y conditional_bruno:train \
     config_name=m1_shapenet_glow nr_gpu=2 learning_rate=1e-5
# srun --gres=gpu:1 guild run -y simple:train
