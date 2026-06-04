#!/bin/bash
cd ~/Code/research/experiments
modal run --detach a2a_forward/mirror_test.py \
  --n-tokens 10000000 \
  --predict-from post_block0 \
  --predict-to post_block3 \
  --fwd-n-layer 2 \
  --inject-after-block 1
