#!/bin/bash -l
# Remaining 3 MT AR BERTScore cells: qwen2/tatoeba, llama3.2/kde4, llama3.2/tatoeba.
# Greedy + max_new_tokens=64, eval-aligned (see probe_autoreg_bertscore.py).
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_ar_bert_remaining_mt_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --job-name=arbert_remaining

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export PYTHONPATH=<DATA_ROOT>/pylibs

cd "$DIR"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Order: do qwen2/tatoeba first (smallest left), then llama3.2 x 2.
for SPEC in "qwen2:tatoeba" "llama3.2:kde4" "llama3.2:tatoeba"; do
    MODEL="${SPEC%%:*}"
    TASK="${SPEC##*:}"
    echo
    echo "--- $MODEL / $TASK (AR + BERTScore, greedy, max_new=64) ---"
    MODEL_NAME=$MODEL TASK=$TASK NUM_SAMPLES=30 python -u probe_autoreg_bertscore.py 2>&1 | tail -80
done
echo; echo "Done at $(date)"
