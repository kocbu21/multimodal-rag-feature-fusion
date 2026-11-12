#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
export HF_HUB_DISABLE_XET=1

for PID in "$@"; do
  while tasklist //FI "PID eq $PID" 2>/dev/null | grep -q " $PID "; do
    echo "$(date +%T) waiting for PID $PID (other GPU job)"; sleep 120
  done
done
wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ $((total - used)) -ge "$1" ] && return
    echo "$(date +%T) waiting for GPU ($((total - used)) MiB free, need $1)"; sleep 60
  done
}

for ENC in openai/clip-vit-base-patch32 google/siglip-base-patch16-224 openai/clip-vit-large-patch14; do
  SLUG=$(echo "${ENC##*/}" | sed 's/[^a-zA-Z0-9]\+/-/g')
  OUT=indexes/flickr1k_multi_$SLUG
  wait_gpu 4000
  echo "== $(date +%T) build $OUT"
  [ -f "$OUT/store.json" ] || $PY scripts/build_index.py --source flickr1k --encoder "$ENC" --out "$OUT"
  wait_gpu 4000
  echo "== $(date +%T) online eval $OUT"
  $PY scripts/eval_online.py --store "$OUT" --n 1000 --sleep 0.2
done
echo "== $(date +%T) ONLINE-DONE"
