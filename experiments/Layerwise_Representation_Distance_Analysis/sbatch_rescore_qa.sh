#!/bin/bash -l
# Re-score 8 QA cells:
#   - bert_score rescale_with_baseline=True (adds ar_bert_*_resc cols + matrix)
#   - squad_f1 multi-ref (adds per-sample matrix for CI)
# Reads texts_*.json that were already saved; no probe regeneration.
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -o <PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis/sbatch_rescore_qa_%j.out
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100-cs
#SBATCH -N 1
#SBATCH -t 1:30:00
#SBATCH --job-name=rescore_qa

set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
module load miniforge3/25.3.0-python3.12.10
source activate MI-FineTune
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export PYTHONPATH=<DATA_ROOT>/pylibs

cd "$DIR"
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 8 runs identified from prior session output.
RUNS=(
    # squad
    "20260517_111432"   # gpt2/squad
    "20260517_111650"   # llama3.2/squad
    "20260517_112134"   # qwen2/squad
    "20260517_113756"   # llama2/squad
    # coqa
    "20260517_131237"   # gpt2/coqa
    "20260517_131538"   # llama3.2/coqa
    "20260517_132114"   # qwen2/coqa
    "20260517_133824"   # llama2/coqa
)

for RUN in "${RUNS[@]}"; do
    echo
    echo "==== $RUN ===="
    # We need MODEL_NAME/TASK set so score_texts can import logit_lens_analysis
    # cleanly if refs.json is missing. Read them from summary.json.
    META="Results/probe_autoreg_bertscore/$RUN/summary.json"
    MODEL_NAME=$(python -c "import json; print(json.load(open('$META'))['model'])")
    TASK=$(python -c "import json; print(json.load(open('$META'))['task'])")
    echo "model=$MODEL_NAME task=$TASK"
    MODEL_NAME=$MODEL_NAME TASK=$TASK \
        python -u score_texts.py "$RUN" \
            --metrics bert,squad_f1 \
            --bert-rescale \
            --update-csv 2>&1 | tail -120
done
echo; echo "Done at $(date)"
