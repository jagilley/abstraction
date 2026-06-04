#!/bin/bash
cd ~/Code/research/experiments

echo "=== 1% forward model (controlled retrain) ==="
modal run --detach a2a_forward/mirror_test_geometry_control.py \
  --n-tokens 10000000 \
  --predict-from post_block0 \
  --predict-to post_block3 \
  --fwd-n-layer 2 \
  --inject-after-block 1

echo ""
echo "=== 10% forward model (extended training) ==="
modal run --detach a2a_forward/mirror_test_geometry_control.py \
  --n-tokens 10000000 \
  --n-steps 14999 \
  --predict-from post_block0 \
  --predict-to post_block3 \
  --fwd-n-layer 3 \
  --fwd-d-head 128 \
  --fwd-n-head 4 \
  --fwd-mlp-mult 4 \
  --inject-after-block 1
