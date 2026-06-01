#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o yelp_sft_train_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s
#SBATCH -N 1
#SBATCH -t 8:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "=== SLURM start $(date) on $(hostname) ==="
PROJECT_ROOT=<PROJECT_ROOT>

echo "--- Training gpt2-small yelp (SFT) ---"
python ${PROJECT_ROOT}/src/Fine_tune/Sentiment_classification/gpt2-yelp-data.py
rc_gpt2=$?
echo "gpt2 yelp exit code: ${rc_gpt2}"

echo "--- Training llama3.2-1b yelp (SFT) ---"
python ${PROJECT_ROOT}/src/Fine_tune/Sentiment_classification/llama3.2-yelp-data.py
rc_llama3=$?
echo "llama3.2 yelp exit code: ${rc_llama3}"

echo "=== SLURM end $(date) ==="
