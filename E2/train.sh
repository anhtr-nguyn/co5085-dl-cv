#!/usr/bin/env bash
# Main grid: tokenizer {patch, row, cnn} x MSA {my, torch} x 5 seeds = 30 runs,
# all with the baseline config/tinyvit.yaml (d128, 4 layers, mlp_ratio 2, 100 epochs).
# A config is skipped if a finished run with the same tag and baseline architecture exists.
# Usage: bash train.sh                     (sequential, GPU 0)
#        GPUS="0 1" JOBS=6 bash train.sh   (parallel, round-robin over GPUs)
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(912 42 100 200 300)
TOKENIZERS=(patch row cnn)
MSAS=(my torch)
GPUS=(${GPUS:-0})
JOBS=${JOBS:-1}
mkdir -p outputs/logs

is_done() {
  for d in outputs/runs/"$1"_*/; do
    [[ -f "$d/log.json" ]] || continue
    grep -q "embed_dim: 128" "$d/config.yaml" && grep -q "num_layers: 4" "$d/config.yaml" \
      && grep -q "mlp_ratio: 2.0" "$d/config.yaml" && grep -q "epochs: 100" "$d/config.yaml" \
      && return 0
  done
  return 1
}

i=0
for tok in "${TOKENIZERS[@]}"; do
  for msa in "${MSAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${tok}_${msa}_aug1_s${seed}"
      if is_done "$tag"; then echo "skip $tag (already done)"; continue; fi
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      echo "run  $tag on GPU $gpu"
      CUDA_VISIBLE_DEVICES=$gpu python train.py --config config/tinyvit.yaml \
        --set model.params.tokenizer_type=$tok model.params.msa_type=$msa train.seed=$seed \
        > "outputs/logs/${tag}.log" 2>&1 &
      i=$((i + 1))
      while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    done
  done
done
wait
