#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o cross_task_llama2_full_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 2-00:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
pip install tabulate
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export CUDA_LAUNCH_BLOCKING=1

# Paths
# Fine-tuned model directory (contains llama2-7b-<task>-full subdirs)
MODEL_DIR="<DATA_ROOT>/fine_tuned_model"
# Corrupted data directory
DATA_DIR="<PROJECT_ROOT>/output/corrupted_data"
# Script path
SCRIPT_PATH="<PROJECT_ROOT>/src/EAP/eap_unified.py"
# Output directory for cross-task edges
OUTPUT_DIR="<PROJECT_ROOT>/output/EAP_edges/cross_task_edges"

# All task names (twitter excluded)
ALL_TASKS=("yelp" "sst2" "coqa" "squad" "kde4" "tatoeba")

# If MODEL_TASK env var is set, restrict the outer loop to just that one source task
# (used by submit_cross_task_llama2_full.sh to fan out across multiple sbatch jobs).
# Otherwise, iterate all 6.
if [ -n "$MODEL_TASK" ]; then
    SRC_TASKS=("$MODEL_TASK")
else
    SRC_TASKS=("${ALL_TASKS[@]}")
fi

# Fixed for this run: Llama-2-7B full fine-tuning
base_model="meta-llama/Llama-2-7b-hf"
model_short="llama2"

mkdir -p "$OUTPUT_DIR"

# Outer loop: 6 llama2 full FT models, one per task
for model_train_task in "${SRC_TASKS[@]}"; do
    ft_path="${MODEL_DIR}/llama2-7b-${model_train_task}-full"

    if [ ! -d "$ft_path" ]; then
        echo "[Error] Model dir not found: $ft_path"
        continue
    fi

    # --------------------------------

    echo "=================================================="
    echo "Processing Model: $model_short | Trained on: $model_train_task"
    echo "=================================================="

    # Inner loop: run on all other tasks' data
    for data_task in "${ALL_TASKS[@]}"; do

        # Skip if data task matches model training task
        if [[ "$data_task" == "$model_train_task" ]]; then
            echo "  [Skip] Data task ($data_task) matches model task. Skipping."
            continue
        fi
        
        #if [[ "$model_short" == "llama2" ]]; then
        #    if [[ "$data_task" != "squad" && "$data_task" != "coqa" ]]; then
                 # echo "  [Skip] Llama2 filter: skipping $data_task" 
        #         continue
         #   fi
        #fi
        # ---------------------------------------

        # Set data path
        current_data_path="${DATA_DIR}/${data_task}_corrupted.csv"
        if [ ! -f "$current_data_path" ]; then
            echo "  [Error] Data not found: $current_data_path"
            continue
        fi

        # Custom model name for descriptive output filenames
        custom_model_name="${model_short}_Finetuned-${model_train_task}_Corrupted-Data"

        echo "  -> Running on Data: $data_task"
        
        python "$SCRIPT_PATH" \
            --mode finetuned \
            --task "$data_task" \
            --base_model_name "$base_model" \
            --model_name "$custom_model_name" \
            --data_path "$current_data_path" \
            --ft_model_path "$ft_path" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 1

    done
done

echo "Cross-task generation finished."