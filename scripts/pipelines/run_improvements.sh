#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
export HF_HUB_DISABLE_XET=1

wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ $((total - used)) -ge "$1" ] && return
    echo "$(date +%T) waiting for GPU ($((total - used)) MiB free, need $1)"; sleep 60
  done
}

for CAP in Salesforce/blip-image-captioning-large \
           "florence-community/Florence-2-large#<DETAILED_CAPTION>" \
           "florence-community/Florence-2-large#<MORE_DETAILED_CAPTION>"; do
  echo "== $(date +%T) captions: $CAP"
  wait_gpu 4000
  $PY scripts/caption_images.py --captioner "$CAP" --device cuda
done

for S in clip-vit-base-patch32 clip-vit-large-patch14 siglip-base-patch16-224; do
  echo "== $(date +%T) experiments $S"
  wait_gpu 4000
  $PY scripts/experiments.py --store indexes/flickr1k_$S
done
echo "== $(date +%T) done"
