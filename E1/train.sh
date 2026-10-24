#!/usr/bin/env bash
# Full grid: {softmax, mlp, cnn} x {no aug, aug} x 5 seeds = 30 runs.
# Configs that already have a finished run in outputs/runs are skipped.
# Usage: bash train.sh            (sequential, one GPU)
#        GPUS="0 1" JOBS=6 bash train.sh   (parallel, round-robin over GPUs)
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(912 42 100 200 300)
MODELS=(softmax mlp cnn)
AUGS=(false true)
GPUS=(${GPUS:-0})
JOBS=${JOBS:-1}
mkdir -p outputs/logs

i=0
for aug in "${AUGS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${model}_aug$([[ $aug == true ]] && echo 1 || echo 0)_s${seed}"
      if compgen -G "outputs/runs/${tag}_*/log.json" > /dev/null; then
        echo "skip $tag (already done)"; continue
      fi
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      log="outputs/logs/${model}_aug${aug}_s${seed}.log"
      CUDA_VISIBLE_DEVICES=$gpu python train.py --config "config/${model}.yaml" \
        --set data.augment=$aug train.seed=$seed > "$log" 2>&1 &
      i=$((i + 1))
      while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    done
  done
done
wait
