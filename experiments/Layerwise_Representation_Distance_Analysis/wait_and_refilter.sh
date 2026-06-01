#!/bin/bash
# Wait for the Llama-2 / Qwen2+Llama3 PCA SLURM jobs to finish, then run
# refilter_samples.py to regenerate filtered_layer_averages_*.csv using the
# original paper's filter (accuracy_improved AND >= 50% layers decreased).
set -uo pipefail

PROJECT_ROOT="<PROJECT_ROOT>"
SCRIPT_DIR="${PROJECT_ROOT}/experiments/Layerwise_Representation_Distance_Analysis"
LOG="${SCRIPT_DIR}/wait_and_refilter.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "waiting for my PCA SLURM jobs to exit (3605917, 3606492)..."
until ! squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -qE "^(3605917|3606492)$"; do
    sleep 60
    n=$(squeue -u "$USER" -h 2>/dev/null | grep -cE "3605917|3606492" || true)
    log "  still running: $n PCA jobs"
done
log "PCA jobs finished."

module load miniforge3/25.3.0-python3.12.10 2>/dev/null || true
source activate MI-FineTune

log "running refilter_samples.py ..."
python -u "${SCRIPT_DIR}/refilter_samples.py" 2>&1 | tee -a "$LOG"

log "done."
