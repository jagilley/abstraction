#!/bin/bash
cd ~/Code/research/experiments
modal run --detach a2a_forward/mirror_test.py \
  --n-tokens 10000000 \
  --n-steps 14999 \
  --predict-from post_block0 \
  --predict-to post_block3 \
  --fwd-n-layer 3 \
  --fwd-d-head 128 \
  --fwd-n-head 4 \
  --fwd-mlp-mult 4 \
  --inject-after-block 1
