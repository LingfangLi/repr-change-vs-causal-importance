#!/bin/bash -l
# QA CoQA AR probe: 4 models x 30 random samples (seed=42), greedy.
# Per-model max_new_tokens matches each FT eval script:
#   gpt2=30, llama3.2=30, qwen2=50, llama2=50
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_ar_coqa_all_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --job-name=arcoqa_all

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export PYTHONPATH=<DATA_ROOT>/pylibs

cd "$DIR"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Order: smallest first to surface errors quickly.
for SPEC in "gpt2:30" "llama3.2:30" "qwen2:50" "llama2:50"; do
    MODEL="${SPEC%%:*}"
    MAX="${SPEC##*:}"
    echo
    echo "--- $MODEL / coqa   max_new=$MAX  N=30  seed=42  greedy ---"
    MODEL_NAME=$MODEL TASK=coqa NUM_SAMPLES=30 RANDOM_SEED=42 MAX_GEN_LEN=$MAX \
        python -u probe_autoreg_bertscore.py 2>&1 | tail -80
done
echo; echo "Done at $(date)"
