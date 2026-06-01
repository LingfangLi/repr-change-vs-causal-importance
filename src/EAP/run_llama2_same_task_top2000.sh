#!/bin/bash -l
# Re-run same-task EAP for Llama-2-7B full FT with --top_k 2000 so we can
# answer whether induction heads appear in the top 401-2000 range (the
# existing finetuned CSVs are hard-truncated at 400 at attribution time).
#
# Parameterized by env var TASK (one of: sst2, kde4, tatoeba).
# Writes to output/EAP_edges/finetuned_top2000/ — does NOT clobber the
# existing top-400 CSVs in output/EAP_edges/finetuned/.
#
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 4:00:00

set -euo pipefail

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

PROJECT_ROOT="<PROJECT_ROOT>"
MODEL_DIR="<DATA_ROOT>/fine_tuned_model"
DATA_DIR="${PROJECT_ROOT}/output/corrupted_data"
SCRIPT_PATH="${PROJECT_ROOT}/src/EAP/eap_unified.py"
OUTPUT_DIR="${PROJECT_ROOT}/output/EAP_edges/finetuned_top2000"

mkdir -p "$OUTPUT_DIR"

TASK="${TASK:?TASK env var must be set (sst2|kde4|tatoeba)}"

ft_path="${MODEL_DIR}/llama2-7b-${TASK}-full"
data_path="${DATA_DIR}/${TASK}_corrupted.csv"

if [ ! -d "$ft_path" ]; then
    echo "[Error] Model dir not found: $ft_path"
    exit 1
fi
if [ ! -f "$data_path" ]; then
    echo "[Error] Data not found: $data_path"
    exit 1
fi

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID  task=$TASK  started $(date)"
echo "Node = $(hostname)"
echo "ft_path  = $ft_path"
echo "data     = $data_path"
echo "out_dir  = $OUTPUT_DIR"
echo "========================================================="

date_start=$(date +%s)

python -u "$SCRIPT_PATH" \
    --mode finetuned \
    --task "$TASK" \
    --base_model_name "meta-llama/Llama-2-7b-hf" \
    --model_name "llama2" \
    --data_path "$data_path" \
    --ft_model_path "$ft_path" \
    --output_dir "$OUTPUT_DIR" \
    --top_k 2000 \
    --batch_size 1

date_end=$(date +%s)
sec=$((date_end - date_start))
hh=$((sec / 3600)); mm=$(((sec % 3600) / 60)); ss=$((sec % 60))

echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${hh}h ${mm}m ${ss}s"
echo "========================================================="
