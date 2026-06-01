#!/bin/bash -l
# Llama-2-7B full-FT logit lens across all 6 tasks (pretrained + FT) at 1000 samples each.
# Llama-2 is the slow one (~8 samples/min), so 12h time and A100 80GB partition.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_lens_llama2_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --job-name=lens_llama2

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

cd "$DIR"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for TASK in yelp sst2 kde4 tatoeba squad coqa; do
    echo
    echo "--- Llama-2 / $TASK ---"
    MODEL_NAME=llama2 TASK=$TASK NUM_SAMPLES=1000 python -u logit_lens_analysis.py 2>&1 | tail -30
done

echo; echo "Done at $(date)"
