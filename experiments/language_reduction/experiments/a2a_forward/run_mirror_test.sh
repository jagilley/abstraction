#!/bin/bash
cd ~/Code/research/experiments
modal run --detach language_reduction/modal_app.py \
  --stage a2a-mirror-test \
  --n-tokens 10000000 \
  --predict-from post_block0 \
  --predict-to post_block3 \
  --fwd-n-layer 2 \
  --inject-after-block 1
