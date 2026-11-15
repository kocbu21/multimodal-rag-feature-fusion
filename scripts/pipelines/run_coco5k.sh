#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
export HF_HUB_DISABLE_XET=1
export MMRAG_VRAM_CAP_GB=${MMRAG_VRAM_CAP_GB:-3.3}
ENCODERS=${*:-google/siglip-base-patch16-224}

wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ $((total - used)) -ge "$1" ] && return
    echo "$(date +%T) waiting for GPU ($((total - used)) MiB free, need $1)"; sleep 60
  done
}

for ENC in $ENCODERS; do
  SLUG=$(echo "${ENC##*/}" | sed 's/[^a-zA-Z0-9]\+/-/g')
  OUT=indexes/coco5k_multi_$SLUG
  wait_gpu 3500
  echo "== $(date +%T) build $OUT"
  [ -f "$OUT/store.json" ] || $PY scripts/build_index.py --source coco5k --encoder "$ENC" --out "$OUT" --caption-batch-size "${CAPTION_BATCH:-2}"
  wait_gpu 3500
  echo "== $(date +%T) online eval $OUT"
  $PY scripts/eval_online.py --dataset coco5k --store "$OUT" --n 1000 --sleep 0.2 --suffix _blip2
done
echo "== $(date +%T) COCO5K-DONE"
