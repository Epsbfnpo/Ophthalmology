#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "concepts.pth" ]; then
  echo "Building Concept Bank..."
  python tools/build_concepts.py --output concepts.pth
fi

python main_hc.py \
  --data_root . \
  --source_domains FGADR_processed \
  --target_domains IDRID_processed DDR_processed \
  --epochs 50 \
  --batch_size 128 \
  --concept_bank concepts.pth
