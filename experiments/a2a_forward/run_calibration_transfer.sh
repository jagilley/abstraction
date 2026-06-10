#!/bin/bash
# Calibration transfer: competence probes (activations → own per-token loss)
# trained ID, evaluated frozen on OOD corpora. All 6 conditions from the
# baseline battery checkpoints. Run from experiments/.
#
# Prereq: modal run a2a_forward/calibration_transfer.py::cache_ood_tokens
set -e
cd "$(dirname "$0")/.."
mkdir -p a2a_forward/logs
modal run --detach a2a_forward/calibration_transfer.py::a2a_calibration_transfer \
    2>&1 | tee a2a_forward/logs/calibration_transfer_$(date +%Y%m%d_%H%M%S).log
