#!/usr/bin/env bash
set -euo pipefail

python -m new.main_hc \
  --algorithm HC_MT_LG_GDRNet \
  --data_root ./data \
  --source_domains IDRiD DDR FGADR \
  --target_domains APTOS MESSIDOR \
  --lambda_seg 10.0 \
  --lambda_reg 1.0
