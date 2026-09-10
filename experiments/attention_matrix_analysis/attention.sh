#!/bin/bash -l
# Use the current working directory and current environment for this job.
#SBATCH -D ./
#SBATCH --export=ALL

#SBATCH -o attention_kl_all_plot_%j.out
##SBATCH -o attention_figures_all_in_one%j.out

# Request 40 cores on 1 node
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs,gpu-h100,gpu-l40s,gpu-a100-lowbig,gpu-a-lowsmall
#SBATCH -N 1
#SBATCH -t 1-00:00:00


module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
#pip install cmapy  
#conda install -c conda-forge graphviz pygraphviz
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

#pip install pandas seaborn matplotlib 
#install eap pack for mt eap
#git clone https://github.com/hannamw/EAP-positional.git
#cd EAP-positional
#git checkout tutorial
#pip install -e .

echo ========================================================= 
echo SLURM job: submitted date = date 
date_start=$(date +%s)
echo ========================================================= 
echo Job output begins 
echo ----------------- 
echo
hostname
# $SLURM_NTASKS is defined automatically as the number of processes in the
# parallel environment.
export CUDA_LAUNCH_BLOCKING=1

PROJECT_ROOT=<PROJECT_ROOT>
python ${PROJECT_ROOT}/experiments/attention_matrix_analysis/measure_attention_kl.py
# Figure-2 downstream (run after the top-400 EAP edge CSVs from src/EAP/ exist):
#   python ${PROJECT_ROOT}/experiments/attention_matrix_analysis/build_layer_kl_vs_eap.py \
#       --model-prefix <model> --kl-csv <attn_kl.csv> --eap-dir <edges_dir> --out-dir <layer_csv_dir>
#   python ${PROJECT_ROOT}/experiments/attention_matrix_analysis/compute_layer_entropy.py \
#       --layer-csv-dir <layer_csv_dir>
echo --------------- 
echo Job output ends 
date_end=$(date +%s)
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo ========================================================= 
echo SLURM job: finished date =$(date) 
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================