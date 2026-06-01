#!/bin/bash -l
# Run Llama-2-7B full-FT PCA layer-wise distance analysis for all 3 main tasks.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_llama2_PCA_full_ft_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --job-name=llama2_pca_full_ft

set -uo pipefail
SCRIPT_DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID on $(hostname) started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================================="

cd "$SCRIPT_DIR"

for TASK in Sentiment QA MT; do
    echo; echo "=========================================="
    echo "=== TASK=$TASK ==="
    echo "=========================================="
    CURRENT_TASK="$TASK" python -u llama2_PCA_distance_full_ft.py \
        2>&1 | tail -200
done

echo; echo "All 3 tasks done at $(date)"
echo "Results under: $SCRIPT_DIR/Results/{Sentiment,QA,MT}/Llama2_full_ft/"
