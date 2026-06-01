#!/bin/bash -l
# Qwen2-0.5B + Llama-3.2-1B PCA layer-wise distance analysis,
# source = <DATA_ROOT>/fine_tuned_models/ (canonical best).
# 3 tasks each = 6 runs sequential on gpu-l40s-low.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_qwen2_llama3_PCA_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s-low
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH --job-name=qwen2_llama3_pca

set -uo pipefail
SCRIPT_DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "=== $SLURM_JOB_ID on $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$SCRIPT_DIR"

for SCRIPT_MODEL in "qwen2_PCA_distance.py" "llama3_PCA_distance.py"; do
    for TASK in Sentiment QA MT; do
        echo; echo "=========================================="
        echo "=== $SCRIPT_MODEL | TASK=$TASK ==="
        echo "=========================================="
        CURRENT_TASK="$TASK" python -u "$SCRIPT_MODEL" 2>&1 | tail -120
    done
done

echo; echo "All done at $(date)"
echo "Results under Results/{Sentiment,QA,MT}/{Qwen2_hfdir,Llama3_hfdir}/"
