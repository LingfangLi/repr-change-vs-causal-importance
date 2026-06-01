#!/bin/bash -l
# SLURM worker for one Llama-2-7B full fine-tuning run on a
# lexical-complexity-filtered subset (text_complexity experiment).
#
# Required env vars (set by submit_tc_full_ft_train.sh):
#   TASK            yelp | squad | tatoeba
#   EXPERIMENT_TYPE simple | complex
# Optional:
#   OUTPUT_BASE_DIR  override default model save dir
#
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 1-00:00:00

set -euo pipefail

if [ -z "${TASK:-}" ] || [ -z "${EXPERIMENT_TYPE:-}" ]; then
    echo "[ERROR] TASK and EXPERIMENT_TYPE env vars are required."
    exit 1
fi

case "$TASK" in
    yelp)    SCRIPT="llama2_sentiment_train_full.py" ;;
    squad)   SCRIPT="llama2_qa_train_full.py" ;;
    tatoeba) SCRIPT="llama2_mt_train_full.py" ;;
    *)       echo "[ERROR] Unknown TASK: $TASK"; exit 1 ;;
esac

PROJECT_ROOT="<PROJECT_ROOT>"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/text_complexity/fine_tune/${SCRIPT}"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID started at $(date)"
echo "TASK            = $TASK"
echo "EXPERIMENT_TYPE = $EXPERIMENT_TYPE"
echo "SCRIPT          = $SCRIPT_PATH"
echo "Node            = $(hostname)"
echo "========================================================="

date_start=$(date +%s)

python -u "$SCRIPT_PATH"

date_end=$(date +%s)
sec=$((date_end - date_start))
hh=$((sec / 3600)); mm=$(((sec % 3600) / 60)); ss=$((sec % 60))

echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${hh}h ${mm}m ${ss}s"
echo "========================================================="
