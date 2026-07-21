#!/usr/bin/env bash
# Forward vs. inverse self-models on a grokked modular-addition MLP.
# Runs locally in the `glp` conda env (torch 2.7.0, MPS). Tiny compute.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/grokking_fwd_vs_inv_${STAMP}.log"

echo "Logging to $LOG"
conda run -n glp python grokking_fwd_vs_inv.py \
    --p 97 \
    --n-epochs 40000 \
    --weight-decay 1.0 \
    --sm-steps 4000 \
    --widths 8,32,128,512,2048 \
    --n-hidden 1 \
    --log-interval 500 \
    2>&1 | tee "$LOG"

echo "Done. Log at $LOG"
