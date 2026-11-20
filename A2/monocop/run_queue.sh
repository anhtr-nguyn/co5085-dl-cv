#!/usr/bin/env bash
# Run MonoCoP training jobs one after another on a single GPU.
# Usage: bash run_queue.sh GPU [WAIT_PID] CONFIG:SEED [CONFIG:SEED ...]
#   WAIT_PID: optional PID of a job already on this GPU; the queue starts after it exits.
# Run from the MonoCoP root. Logs go to outputs/logs/<model_name>_s<seed>.out.
set -u
GPU=$1; shift
if [[ $1 =~ ^[0-9]+$ ]]; then
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    shift
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate monocop
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
mkdir -p outputs/logs

for job in "$@"; do
    cfg=${job%%:*}; seed=${job##*:}
    name=$(basename "$cfg" .yaml)
    echo "$(date '+%F %T') start $cfg seed $seed on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU python train_val.py --config "$cfg" --seed "$seed" \
        > "outputs/logs/${name}_s${seed}.out" 2>&1 < /dev/null
    status=$?
    echo "$(date '+%F %T') done $cfg seed $seed (exit $status)"
done
