#!/usr/bin/env bash
set -euo pipefail

python main_hc.py \
  --data_root . \
  --source_domains IDRID_processed DDR_processed FGADR_processed \
  --target_domains APTOS MESSIDOR \
  --epochs 50 \
  --batch_size 8 \
  --concept_bank concepts.pth
