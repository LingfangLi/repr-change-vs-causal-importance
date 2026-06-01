#!/bin/bash -l
# Llama-2-7B AR sampling generation + BERTScore on KDE4+Tatoeba.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_ar_bert_llama2_mt_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s-low
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --job-name=arbert_l2mt

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export PYTHONPATH=<DATA_ROOT>/pylibs

cd "$DIR"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for TASK in kde4 tatoeba; do
    echo
    echo "--- Llama-2 / $TASK (AR + BERTScore, eval params) ---"
    MODEL_NAME=llama2 TASK=$TASK NUM_SAMPLES=30 SEED=42 python -u probe_autoreg_bertscore.py 2>&1 | tail -50
done
echo; echo "Done at $(date)"
