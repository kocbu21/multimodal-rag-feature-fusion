#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe
for ENC in openai/clip-vit-base-patch32 google/siglip-base-patch16-224 openai/clip-vit-large-patch14; do
  SLUG=$(echo "${ENC##*/}" | sed 's/[^a-zA-Z0-9]\+/-/g')
  echo "=== $ENC ==="
  [ -f "indexes/flickr1k_$SLUG/store.json" ] || $PY scripts/build_index.py --source flickr1k --encoder "$ENC" --captioner Salesforce/blip-image-captioning-base --single-caption
  $PY scripts/evaluate.py --store "indexes/flickr1k_$SLUG"
done
