#!/bin/bash -l
# Re-score 8 MT cells with rescaled BERTScore + per-sample matrices.
# Reads texts_*.json saved by the prior probe runs.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_rescore_mt_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 1:30:00
#SBATCH --job-name=rescore_mt

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export PYTHONPATH=<DATA_ROOT>/pylibs

cd "$DIR"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUNS=(
    "20260516_122453"   # llama2/kde4
    "20260516_122603"   # gpt2/kde4
    "20260516_123354"   # gpt2/tatoeba
    "20260516_124103"   # qwen2/kde4
    "20260516_131603"   # llama2/tatoeba
    "20260516_133431"   # qwen2/tatoeba
    "20260516_143413"   # llama3.2/kde4
    "20260516_145517"   # llama3.2/tatoeba
)

for RUN in "${RUNS[@]}"; do
    echo
    echo "==== $RUN ===="
    META="Results/probe_autoreg_bertscore/$RUN/summary.json"
    MODEL_NAME=$(python -c "import json; print(json.load(open('$META'))['model'])")
    TASK=$(python -c "import json; print(json.load(open('$META'))['task'])")
    echo "model=$MODEL_NAME task=$TASK"
    # MT cells: just rescore BERTScore with rescale. No squad_f1 needed for MT.
    MODEL_NAME=$MODEL_NAME TASK=$TASK \
        python -u score_texts.py "$RUN" \
            --metrics bert \
            --bert-rescale \
            --update-csv 2>&1 | tail -80
done
echo; echo "Done at $(date)"
