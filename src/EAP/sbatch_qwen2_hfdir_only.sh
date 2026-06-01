#!/bin/bash -l
# Run ONLY Qwen2 cross-task EAP v2 (<DATA_ROOT>/fine_tuned_models/ source),
# to parallelize with the Llama-3.2+Qwen2 sequential job on gpu-a100-cs.
# Output goes to the same target dir; the runner's skip-if-exists logic
# prevents collisions if the other job also reaches the Qwen2 stage.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/output/EAP_edges/sbatch_qwen2_hfdir_only_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s-low
#SBATCH -N 1
#SBATCH -t 1:30:00
#SBATCH --job-name=qwen2_eap_v2

set -uo pipefail
PROJECT_ROOT="<PROJECT_ROOT>"

module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'

echo "========================================================="
echo "SLURM job $SLURM_JOB_ID on $(hostname) started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "========================================================="

start_ts=$(date +%s)

bash "${PROJECT_ROOT}/src/EAP/run_qwen2_all_edges_v2.sh" 2>&1 \
    | tee -a "${PROJECT_ROOT}/output/EAP_edges/qwen2_cross_task_hfdir/_run.log" \
    | grep -E "^\[|===|ERROR" | tail -80

end_ts=$(date +%s)
sec=$((end_ts-start_ts)); mm=$((sec/60)); ss=$((sec%60))
echo; echo "========================================================="
echo "Finished at $(date)  |  wall clock: ${mm}m ${ss}s"
echo "  qwen2_cross_task_hfdir/: $(ls ${PROJECT_ROOT}/output/EAP_edges/qwen2_cross_task_hfdir 2>/dev/null | wc -l) files"
echo "========================================================="
