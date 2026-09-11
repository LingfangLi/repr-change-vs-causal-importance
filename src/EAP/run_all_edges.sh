#!/bin/bash
# EAP edge generation for one model: 6 pretrained + 6 own-task + 30 cross-task
# runs (top_k=-1, all edges). The cross-task CSVs feed the Figure-4 overlap
# (compute_same_ft_cross_data_overlap.py).
#
#   bash src/EAP/run_all_edges.sh <gpt2|qwen2|llama3|llama2>
set -uo pipefail

MODEL="${1:?usage: run_all_edges.sh <gpt2|qwen2|llama3|llama2>}"
PROJECT_ROOT="<PROJECT_ROOT>"
MODEL_DIR="<DATA_ROOT>/fine_tuned_model"
DATA_DIR="${PROJECT_ROOT}/output/corrupted_data"
SCRIPT="${PROJECT_ROOT}/src/EAP/eap_unified.py"
TASKS=(yelp sst2 squad coqa kde4 tatoeba)
declare -A FT

case "$MODEL" in
  gpt2)
    MODEL_NAME=gpt2; BASE_MODEL=gpt2
    FT[yelp]="${MODEL_DIR}/gpt2-small-yelp-full-ft-20260415-232443"
    FT[sst2]="${MODEL_DIR}/gpt2-sst2-full-ft-20251205-172809"
    FT[squad]="${MODEL_DIR}/gpt2-small-squad-full-ft-20260105-230037"
    FT[coqa]="${MODEL_DIR}/gpt2-small-COQA-full-ft-20260105-230716"
    FT[kde4]="${MODEL_DIR}/gpt2-small-kde4-full-ft-20260106-204426/gpt2-small-kde4-full-ft-20260106-204426"
    FT[tatoeba]="${MODEL_DIR}/gpt2-small-tatoeba-full-ft-20260106-224923"
    ;;
  qwen2)
    MODEL_NAME=qwen2; BASE_MODEL="Qwen/Qwen2-0.5B"
    FT[yelp]="${MODEL_DIR}/qwen2-0.5b-yelp-full-ft-20251124-204027"
    FT[sst2]="${MODEL_DIR}/qwen2-0.5b-sst2-full-20251209-1054"
    FT[squad]="${MODEL_DIR}/qwen2-0.5b-squad-full-20251125-165024"
    FT[coqa]="${MODEL_DIR}/qwen2-0.5b-coqa-full-20251125-182058"
    FT[kde4]="${MODEL_DIR}/qwen2-kde4-tech-trans-full-20251125-165755"
    FT[tatoeba]="${MODEL_DIR}/qwen2-0.5b-tatoeba-en-fr-20251125-165129"
    ;;
  llama3)
    MODEL_NAME=llama3.2; BASE_MODEL="meta-llama/Llama-3.2-1B"
    FT[yelp]="${MODEL_DIR}/llama3.2-1b-yelp-full-ft-20260415-233127"
    FT[sst2]="${MODEL_DIR}/llama3.2-1b-sst2-full-20251209-1554"
    FT[squad]="${MODEL_DIR}/llama3.2-1b-SQUAD-full-ft-20260106-222423"
    FT[coqa]="${MODEL_DIR}/llama3.2-1b-COQA-full-ft-20260105-230514"
    FT[kde4]="${MODEL_DIR}/llama3.2-1b-kde4-full-ft-20260106-221031"
    FT[tatoeba]="${MODEL_DIR}/llama3.2-1b-tatoeba-full-ft-20260106-225023"
    ;;
  llama2)
    MODEL_NAME=llama2; BASE_MODEL="meta-llama/Llama-2-7b-hf"
    FT[yelp]="${MODEL_DIR}/llama2-7b-yelp-full"
    FT[sst2]="${MODEL_DIR}/llama2-7b-sst2-full"
    FT[squad]="${MODEL_DIR}/llama2-7b-squad-full"
    FT[coqa]="${MODEL_DIR}/llama2-7b-coqa-full"
    FT[kde4]="${MODEL_DIR}/llama2-7b-kde4-full"
    FT[tatoeba]="${MODEL_DIR}/llama2-7b-tatoeba-full"
    ;;
  *) echo "unknown model '$MODEL' (use gpt2|qwen2|llama3|llama2)"; exit 1 ;;
esac

OUT_DIR="${PROJECT_ROOT}/output/EAP_edges/${MODEL}_all_edges"
mkdir -p "$OUT_DIR"

run_one() {
    local label="$1"; shift
    echo; echo "==== [$(date '+%H:%M:%S')] $label ===="
    "$@"; echo "[$(date '+%H:%M:%S')] done rc=$?"
}

# 1) Pretrained base on each task's corrupted data
for t in "${TASKS[@]}"; do
    out="${OUT_DIR}/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$out" ] && { echo "[skip] $(basename "$out")"; continue; }
    run_one "pretrained | data=$t" \
        python "$SCRIPT" --task "$t" --model_name "$MODEL_NAME" --mode pretrained \
            --base_model_name "$BASE_MODEL" --data_path "${DATA_DIR}/${t}_corrupted.csv" \
            --top_k -1 --batch_size 1 --output_dir "$OUT_DIR"
    src="${OUT_DIR}/pretrained/${MODEL_NAME}_${t}_pretrained_edges.csv"
    [ -f "$src" ] && mv "$src" "$out"
done
rmdir "${OUT_DIR}/pretrained" 2>/dev/null || true

# 2) Fine-tuned: ft_task x test_task (own-task when equal, else cross-task)
TMP_DIR="${OUT_DIR}/_tmp"; mkdir -p "$TMP_DIR"
for ft_task in "${TASKS[@]}"; do
    ft_path="${FT[$ft_task]}"
    [ ! -d "$ft_path" ] && { echo "[ERROR] missing FT dir: $ft_path"; continue; }
    for test_task in "${TASKS[@]}"; do
        if [ "$ft_task" = "$test_task" ]; then
            out="${OUT_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        else
            out="${OUT_DIR}/${MODEL_NAME}_Finetuned-${ft_task}_Corrupted-Data_${test_task}_finetuned_edges.csv"
        fi
        [ -f "$out" ] && { echo "[skip] $(basename "$out")"; continue; }
        run_one "FT=$ft_task | data=$test_task" \
            python "$SCRIPT" --task "$test_task" --model_name "$MODEL_NAME" --mode finetuned \
                --base_model_name "$BASE_MODEL" --ft_model_path "$ft_path" \
                --data_path "${DATA_DIR}/${test_task}_corrupted.csv" \
                --top_k -1 --batch_size 1 --output_dir "$TMP_DIR"
        src="${TMP_DIR}/${MODEL_NAME}_${test_task}_finetuned_edges.csv"
        [ -f "$src" ] && mv "$src" "$out" || echo "[ERROR] missing: $src"
    done
done
rmdir "$TMP_DIR" 2>/dev/null || true

echo "All done. Files under $OUT_DIR: $(ls "$OUT_DIR" | wc -l)"
