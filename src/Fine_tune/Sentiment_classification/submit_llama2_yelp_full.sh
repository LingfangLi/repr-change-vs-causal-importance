#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o llama2_yelp_full_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-l40s,gpu-a100-lowbig
#SBATCH -N 1
#SBATCH -t 1-00:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "=== SLURM start $(date) on $(hostname) ==="
PROJECT_ROOT=<PROJECT_ROOT>

python ${PROJECT_ROOT}/src/Fine_tune/Sentiment_classification/Llama2-7b-yelp-full.py

echo "Exit code: $?"
echo "=== SLURM end $(date) ==="
