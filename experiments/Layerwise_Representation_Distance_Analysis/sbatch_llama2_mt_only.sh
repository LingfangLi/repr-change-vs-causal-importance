#!/bin/bash -l
# Re-run ONLY Llama-2 full-FT on MT (KDE4) — the previous combined sbatch
# 3618407 hit the 2h time limit during Llama-2.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_llama2_mt_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --job-name=llama2_mt

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$DIR"

echo; echo "--- Llama-2 full-FT MT ---"
CURRENT_TASK=MT python -u llama2_PCA_distance_full_ft.py 2>&1

echo; echo "Done at $(date)"
