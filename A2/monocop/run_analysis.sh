#!/usr/bin/env bash
# Run analyze.py (branch ablation + error by distance + KITTI-C + speed) on a list of runs, one GPU.
# Usage: bash run_analysis.sh GPU ITEM [ITEM ...]
#   ITEM = wait:<pid>               block until that process exits (e.g. the training job on this GPU)
#        | <config>@<run_dir_glob>  analyze <run_dir>/checkpoint_best.pth; the glob is resolved when reached
# Run from the MonoCoP root. Results: A2/outputs/analysis/<run_dir>.json (finished sets are skipped).
set -u
GPU=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../outputs/analysis
mkdir -p "$OUT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate monocop
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

for item in "$@"; do
    if [[ $item == wait:* ]]; then
        pid=${item#wait:}
        echo "$(date '+%F %T') waiting for pid $pid"
        while kill -0 "$pid" 2>/dev/null; do sleep 60; done
        continue
    fi
    cfg=${item%%@*}
    run=$(ls -d ${item#*@} 2>/dev/null | head -1)
    if [[ -z $run || ! -f $run/checkpoint_best.pth ]]; then
        echo "$(date '+%F %T') no checkpoint for ${item#*@}, skipped"
        continue
    fi
    name=$(basename "$run")
    echo "$(date '+%F %T') start $name"
    CUDA_VISIBLE_DEVICES=$GPU python "$HERE/analyze.py" --config "$cfg" --ckpt "$run/checkpoint_best.pth" \
        --out "$OUT/$name.json" --sets all --speed > "$OUT/$name.log" 2>&1
    status=$?
    echo "$(date '+%F %T') done $name (exit $status)"
done
