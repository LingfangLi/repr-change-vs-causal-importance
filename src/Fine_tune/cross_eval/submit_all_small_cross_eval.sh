#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o all_small_cross_eval_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-a100-lowbig,gpu-a-lowsmall,gpu-l40s
#SBATCH -N 1
#SBATCH -t 1-00:00:00

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "=== SLURM start $(date) on $(hostname) ==="
PROJECT_ROOT=<PROJECT_ROOT>

for MODEL in gpt2 llama3.2 qwen2; do
  echo ""
  echo "===================================="
  echo "Cross-eval: ${MODEL}"
  echo "===================================="
  python ${PROJECT_ROOT}/src/Fine_tune/cross_eval/gpt2_llama3_cross_eval.py --model ${MODEL}
  echo "Exit code for ${MODEL}: $?"
done

echo ""
echo "=== SLURM end $(date) ==="
