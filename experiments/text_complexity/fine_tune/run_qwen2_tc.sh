#!/bin/bash
# Qwen2-0.5B lexical-complexity full-FT: 6 trains + 18 evals + aggregate.
# Designed to run interactively on a single GPU. Resumable: skips existing
# outputs.
set -uo pipefail

PROJECT_ROOT="<PROJECT_ROOT>"
DIR="${PROJECT_ROOT}/experiments/text_complexity/fine_tune"
MODEL_BASE_DIR="<DATA_ROOT>/fine_tuned_model/text_complexity_qwen2"
RESULTS_DIR="${DIR}/results_full_qwen2"
mkdir -p "$MODEL_BASE_DIR" "$RESULTS_DIR"
export OUTPUT_BASE_DIR="$MODEL_BASE_DIR"

TASKS=(yelp squad tatoeba)
SUBSETS=(simple complex)
declare -A TASK_SCRIPT
TASK_SCRIPT[yelp]="qwen2_sentiment"
TASK_SCRIPT[squad]="qwen2_qa"
TASK_SCRIPT[tatoeba]="qwen2_mt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- 1) Train: 6 models ----------
for task in "${TASKS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        # Skip if any matching model dir exists
        if ls -d "${MODEL_BASE_DIR}/qwen2-0.5b-${task}-${subset}-full-"* 2>/dev/null | head -1 | grep -q .; then
            log "[skip-train] ${task}-${subset} already trained"
            continue
        fi
        log "===== TRAIN task=${task} subset=${subset} ====="
        EXPERIMENT_TYPE="$subset" python -u "${DIR}/${TASK_SCRIPT[$task]}_train_full.py"
        rc=$?
        log "train rc=$rc"
        if [ $rc -ne 0 ]; then
            log "[ERROR] training failed; continuing"
        fi
    done
done

# ---------- 2) Eval: 6 FT + 3 base = 9 jobs × 2 test subsets = 18 JSONs ----------
# FT evals
for task in "${TASKS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        pattern="${MODEL_BASE_DIR}/qwen2-0.5b-${task}-${subset}-full-*"
        model_path=$(ls -d $pattern 2>/dev/null | sort -r | head -1)
        if [ -z "$model_path" ] || [ ! -d "$model_path" ]; then
            log "[ERROR] no model dir for ${task}-${subset}"; continue
        fi
        script="${DIR}/${TASK_SCRIPT[$task]}_eval_full.py"
        suffix="_full.json"; [ "$task" = "tatoeba" ] && suffix="_full_nltk.json"
        for test_subset in simple complex; do
            out="${RESULTS_DIR}/eval_results_${task}_train_${subset}_test_${test_subset}${suffix}"
            if [ -f "$out" ]; then
                log "[skip-eval] $(basename $out)"; continue
            fi
            log "----- EVAL task=${task} train=${subset} test=${test_subset} -----"
            IS_BASE_MODEL=0 \
            MODEL_PATH="$model_path" \
            TRAIN_SUBSET="$subset" \
            TEST_SUBSET="$test_subset" \
            OUTPUT_DIR="$RESULTS_DIR" \
            python -u "$script"
        done
    done
done

# Base evals
for task in "${TASKS[@]}"; do
    script="${DIR}/${TASK_SCRIPT[$task]}_eval_full.py"
    suffix="_full.json"; [ "$task" = "tatoeba" ] && suffix="_full_nltk.json"
    for test_subset in simple complex; do
        out="${RESULTS_DIR}/eval_results_${task}_test_${test_subset}_base${suffix}"
        if [ -f "$out" ]; then
            log "[skip-eval] $(basename $out)"; continue
        fi
        log "----- EVAL BASE task=${task} test=${test_subset} -----"
        IS_BASE_MODEL=1 \
        TEST_SUBSET="$test_subset" \
        OUTPUT_DIR="$RESULTS_DIR" \
        python -u "$script"
    done
done

# ---------- 3) Aggregate ----------
log "===== AGGREGATE ====="
python -u "${DIR}/aggregate_eval_results.py" "$RESULTS_DIR" || log "[WARN] aggregate failed"

log "ALL DONE. Model dir: $MODEL_BASE_DIR  |  Results: $RESULTS_DIR"
