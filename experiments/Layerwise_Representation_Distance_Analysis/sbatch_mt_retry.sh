#!/bin/bash -l
# Retry MT task for Llama-2 full-FT + Qwen2 + Llama-3.2 after the KDE4
# dataset loading fix (added name="en-fr" + trust_remote_code=True).
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_mt_retry_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --job-name=mt_retry_3models

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$DIR"

echo; echo "--- Qwen2 MT ---"
CURRENT_TASK=MT python -u qwen2_PCA_distance.py 2>&1 | tail -80

echo; echo "--- Llama-3.2 MT ---"
CURRENT_TASK=MT python -u llama3_PCA_distance.py 2>&1 | tail -80

echo; echo "--- Llama-2 full-FT MT ---"
CURRENT_TASK=MT python -u llama2_PCA_distance_full_ft.py 2>&1 | tail -80

echo; echo "All MT retries done at $(date)"
