#!/bin/bash
# Wrapper to fan out cross-task EAP for Llama2-7B full FT into 6 parallel sbatch jobs.
# Each job loads ONE source-task model and runs EAP on the 5 other tasks' corrupted data.
#
# Usage:
#   bash submit_cross_task_llama2_full.sh
#
# Each sbatch job calls run_cross_task_generation.sh with MODEL_TASK=<src_task> set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/run_cross_task_generation.sh"

ALL_TASKS=("yelp" "sst2" "coqa" "squad" "kde4" "tatoeba")

for src_task in "${ALL_TASKS[@]}"; do
    echo "Submitting job for source task: ${src_task}"
    sbatch \
        --export=ALL,MODEL_TASK="${src_task}" \
        --job-name="ct_llama2_${src_task}" \
        --output="cross_task_llama2_full_${src_task}_%j.out" \
        "${WORKER_SCRIPT}"
done

echo
echo "Submitted 6 jobs. Check status with:  squeue -u \$USER"
