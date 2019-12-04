#!/bin/bash

# What works:
# conda activate bruno_v2
# srun --gres=gpu:1 --mem=4G python3 -m config_conditional.train  --config_name m1_shapenet_glow --nr_gpu 1


#SBATCH --time=12:00:00          ## wallclock time hh:mm:ss
#SBATCH --gres=gpu:2        ## one K80 requested
#SBATCH --mem=4G            ## one K80 requested

# srun python3 -m config_conditional.train  --config_name m1_shapenet_glow --nr_gpu 2
# srun --time=0:10:00 --gres=gpu:1 --mem=2G guild run -y conditional_bruno:train config_name=m1_shapenet_glow
srun --time=0:10:00 --gres=gpu:1 --mem=2G guild run -y conditional_bruno:train config_name=m1_shapenet
# srun --gres=gpu:1 guild run -y simple:train
