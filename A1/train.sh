#!/usr/bin/env bash
# PHASE=sweep: dò learning rate trên val cho 3 chế độ pretrained (không chạm test)
#   head: lịch đầy đủ 15 epoch; partial/full: lịch rút gọn 10 epoch. -> outputs/sweep/
# PHASE=main: 2 arch x 4 chế độ với learning rate trong config/lr.txt, theo SEEDS. -> outputs/runs/
# Bỏ qua cấu hình đã có run hoàn tất (log.json). Log stdout ở outputs/logs/.
# Usage: PY=<python của env> GPUS="0 1" JOBS=2 PHASE=sweep bash train.sh
#        PY=<python của env> GPUS="0 1" JOBS=2 PHASE=main SEEDS="912 42 100" bash train.sh
# Lọc: ARCHS="resnet50", MODES="head partial" ...
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
PHASE=${PHASE:-main}
GPUS=(${GPUS:-0})
JOBS=${JOBS:-1}
SEEDS=(${SEEDS:-912})
ARCHS=(${ARCHS:-resnet50 deit_s})
MODES=(${MODES:-scratch full partial head})
mkdir -p outputs/logs
export TQDM_DISABLE=1

declare -A EPOCHS=([head]=15 [partial]=30 [full]=30 [scratch]=100)
declare -A SWEEP_EPOCHS=([head]=15 [partial]=10 [full]=10)
declare -A SWEEP_LRS=(
  [resnet50_head]="3e-4 1e-3 3e-3 1e-2" [deit_s_head]="3e-4 1e-3 3e-3 1e-2"
  [resnet50_partial]="1e-4 3e-4 1e-3 3e-3 1e-2" [deit_s_partial]="3e-5 1e-4 3e-4 1e-3 3e-3"
  [resnet50_full]="1e-4 3e-4 1e-3 3e-3" [deit_s_full]="3e-5 1e-4 3e-4 1e-3"
)

i=0
launch() {   # launch <tag> <root> <args...>
  local tag=$1 root=$2; shift 2
  for d in "$root/${tag%_lr*}"_*/; do
    [[ -f "$d/log.json" && -f "$d/.tag_$tag" ]] && { echo "skip $tag"; return; }
  done
  local gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  echo "run  $tag on GPU $gpu"
  ( CUDA_VISIBLE_DEVICES=$gpu $PY train.py --config config/a1.yaml --set output.root=$root "$@" \
      > "outputs/logs/$tag.log" 2>&1 || true
    d=$(grep -o -- "-> .*" "outputs/logs/$tag.log" | head -1 | sed 's/.*-> //' || true)
    if [[ -n "$d" && -f "$d/log.json" ]]; then touch "$d/.tag_$tag"; else echo "FAIL $tag"; fi ) &
  i=$((i + 1))
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || true; done
}

if [[ $PHASE == sweep ]]; then
  for mode in "${MODES[@]}"; do
    [[ $mode == scratch ]] && continue
    for arch in "${ARCHS[@]}"; do
      for lr in ${SWEEP_LRS[${arch}_${mode}]}; do
        launch "${arch}_${mode}_s912_lr${lr}" outputs/sweep \
          model.arch=$arch model.mode=$mode train.lr=$lr \
          train.epochs=${SWEEP_EPOCHS[$mode]} train.eval_test=false
      done
    done
  done
else
  # config/lr.txt: mỗi dòng "<arch> <mode> <lr>"
  declare -A LR
  while read -r a m v; do [[ -z "$a" || $a == \#* ]] || LR[${a}_${m}]=$v; done < config/lr.txt
  for mode in "${MODES[@]}"; do
    for arch in "${ARCHS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        launch "${arch}_${mode}_s${seed}_lr${LR[${arch}_${mode}]}" outputs/runs \
          model.arch=$arch model.mode=$mode train.lr=${LR[${arch}_${mode}]} \
          train.epochs=${EPOCHS[$mode]} train.seed=$seed
      done
    done
  done
fi
wait || true
