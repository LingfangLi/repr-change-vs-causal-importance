#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o llama2-full-ft-coqa-%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a-lowsmall
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export OUTPUT_DIR="<DATA_ROOT>/fine_tuned_model"

echo =========================================================
echo "SLURM job: submitted date = $(date)"
date_start=$(date +%s)
echo =========================================================
hostname
nvidia-smi

python <PROJECT_ROOT>/src/Fine_tune/Question_answering/LlaMA2-7b-coqa-full.py

echo =========================================================
echo "SLURM job: finished date = $(date)"
date_end=$(date +%s)
seconds=$((date_end-date_start))
echo "Total run time : $((seconds/3600))h $(((seconds%3600)/60))m $((seconds%60))s"
echo =========================================================
