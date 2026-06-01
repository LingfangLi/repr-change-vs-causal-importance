#!/bin/bash -l
# SLURM worker for evaluating one Llama-2-7B full FT (text_complexity) model
# on BOTH simple and complex test subsets in a single job (saves on model
# loading overhead).
#
# Required env vars (set by submit_tc_full_ft_eval.sh):
#   TASK             yelp | squad | tatoeba
# And EXACTLY ONE of:
#   TRAIN_SUBSET     simple | complex   (resolve latest matching FT model dir)
#   IS_BASE_MODEL=1                     (evaluate base pre-trained Llama-2-7B)
#
# Optional:
#   MODEL_BASE_DIR   override default scratch dir
#   RESULTS_DIR      override output dir
#
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 6:00:00

set -euo pipefail

if [ -z "${TASK:-}" ]; then
    echo "[ERROR] TASK env var is required."
    exit 1
fi

case "$TASK" in
    yelp)    SCRIPT="llama2_sentiment_eval_full.py" ;;
    squad)   SCRIPT="llama2_qa_eval_full.py" ;;
    tatoeba) SCRIPT="llama2_mt_eval_full.py" ;;
    *)       echo "[ERROR] Unknown TASK: $TASK"; exit 1 ;;
esac

PROJECT_ROOT="<PROJECT_ROOT>"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/text_complexity/fine_tune/${SCRIPT}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/experiments/text_complexity/fine_tune/results_full}"
MODEL_BASE_DIR="${MODEL_BASE_DIR:-<DATA_ROOT>/fine_tuned_model/text_complexity}"
mkdir -p "$RESULTS_DIR"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID started at $(date)"
echo "TASK             = $TASK"
echo "Mode             = $([ "${IS_BASE_MODEL:-0}" = "1" ] && echo BASE || echo "FT (train=$TRAIN_SUBSET)")"
echo "Node             = $(hostname)"
echo "RESULTS_DIR      = $RESULTS_DIR"
echo "========================================================="

# Resolve model path for FT mode
RESOLVED_MODEL_PATH=""
if [ "${IS_BASE_MODEL:-0}" != "1" ]; then
    if [ -z "${TRAIN_SUBSET:-}" ]; then
        echo "[ERROR] TRAIN_SUBSET required when IS_BASE_MODEL is not 1"
        exit 1
    fi
    pattern="${MODEL_BASE_DIR}/llama2-7b-${TASK}-${TRAIN_SUBSET}-full-*"
    RESOLVED_MODEL_PATH=$(ls -d $pattern 2>/dev/null | sort -r | head -1)
    if [ -z "$RESOLVED_MODEL_PATH" ] || [ ! -d "$RESOLVED_MODEL_PATH" ]; then
        echo "[ERROR] No FT model dir found matching: $pattern"
        exit 1
    fi
    echo "Resolved MODEL_PATH = $RESOLVED_MODEL_PATH"
fi

date_start=$(date +%s)

for test_subset in simple complex; do
    echo
    echo "---------------------------------------------------------"
    echo "Running eval: TEST_SUBSET=$test_subset"
    echo "---------------------------------------------------------"
    if [ "${IS_BASE_MODEL:-0}" = "1" ]; then
        IS_BASE_MODEL=1 \
        TEST_SUBSET="$test_subset" \
        OUTPUT_DIR="$RESULTS_DIR" \
        python -u "$SCRIPT_PATH"
    else
        IS_BASE_MODEL=0 \
        MODEL_PATH="$RESOLVED_MODEL_PATH" \
        TRAIN_SUBSET="$TRAIN_SUBSET" \
        TEST_SUBSET="$test_subset" \
        OUTPUT_DIR="$RESULTS_DIR" \
        python -u "$SCRIPT_PATH"
    fi
done

date_end=$(date +%s)
sec=$((date_end - date_start))
hh=$((sec / 3600)); mm=$(((sec % 3600) / 60)); ss=$((sec % 60))

echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${hh}h ${mm}m ${ss}s"
echo "========================================================="
