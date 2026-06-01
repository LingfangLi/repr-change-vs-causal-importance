#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o llama2-eval-%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a-lowsmall
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH --mem=64G

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

hostname
nvidia-smi
python "$EVAL_SCRIPT"
