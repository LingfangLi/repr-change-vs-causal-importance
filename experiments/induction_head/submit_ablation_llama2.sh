#!/bin/bash -l
# Run the canonical induction head ablation test on llama-2 full-FT models.
# Sequential across 3 tasks in a single job (model is reloaded per task).
#
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o ablate_llama2_induction_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 2:00:00

set -euo pipefail

PROJECT_ROOT="<PROJECT_ROOT>"
SCRIPT="${PROJECT_ROOT}/experiments/induction_head/ablate_canonical_induction_heads.py"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID started at $(date)"
echo "Node = $(hostname)"
echo "========================================================="

date_start=$(date +%s)

python -u "$SCRIPT"

date_end=$(date +%s)
sec=$((date_end - date_start))
hh=$((sec / 3600)); mm=$(((sec % 3600) / 60)); ss=$((sec % 60))

echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${hh}h ${mm}m ${ss}s"
echo "========================================================="
