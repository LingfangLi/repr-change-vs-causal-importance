#!/bin/bash
# Wait for SLURM job 3618407 (MT retry) to finish, then refilter everything.
set -uo pipefail
DIR="<PROJECT_ROOT>/experiments/Layerwise_Representation_Distance_Analysis"
LOG="$DIR/wait_mt_and_refilter.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "waiting for SLURM 3618407 (MT retry) ..."
until ! squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -q "^3618407$"; do
    sleep 60
    n=$(squeue -u "$USER" -h 2>/dev/null | grep -c "3618407" || true)
    log "  still running ($n job)"
done
log "3618407 finished."

module load miniforge3/25.3.0-python3.12.10 2>/dev/null || true
source activate MI-FineTune

log "running refilter_samples.py ..."
python -u "${DIR}/refilter_samples.py" 2>&1 | tee -a "$LOG"
log "done."
