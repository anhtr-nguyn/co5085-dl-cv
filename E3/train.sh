#!/usr/bin/env bash
# Main grid: cell {lstm, gru} x sequence {row, col, patch} x 5 seeds = 30 runs (config/rnn.yaml).
# Configs that already have a finished run in outputs/runs are skipped.
# Usage: bash train.sh                     (sequential, GPU 0)
#        GPUS="0 1" JOBS=6 bash train.sh   (parallel, round-robin over GPUs)
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(912 42 100 200 300)
CELLS=(lstm gru)
SEQS=(row col patch)
GPUS=(${GPUS:-0})
JOBS=${JOBS:-1}
mkdir -p outputs/logs

# tải CIFAR-10 một lần trước, tránh nhiều job song song cùng tải
python -c "from data import get_dataloaders; get_dataloaders(num_workers=0, split_path=None)"

i=0
for cell in "${CELLS[@]}"; do
  for seq in "${SEQS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${cell}_${seq}_aug1_s${seed}"
      if compgen -G "outputs/runs/${tag}_*/log.json" > /dev/null; then
        echo "skip $tag (already done)"; continue
      fi
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      echo "run  $tag on GPU $gpu"
      CUDA_VISIBLE_DEVICES=$gpu python train.py --config config/rnn.yaml \
        --set model.params.cell=$cell model.params.seq_type=$seq train.seed=$seed \
        > "outputs/logs/${tag}.log" 2>&1 &
      i=$((i + 1))
      while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    done
  done
done
wait
