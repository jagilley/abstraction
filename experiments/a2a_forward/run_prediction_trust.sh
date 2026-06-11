#!/bin/bash
# Prediction trust tests on the controlled retrain checkpoints:
#   Test A — Wiener gain analysis: do the gate's and downstream layers'
#            per-direction usage gains track the forward model's
#            per-direction reliability (closed-form Wiener gains)?
#   Test B — counterfactual error injection: corrupt the prediction along
#            habitual-error vs trusted vs random directions at matched norm;
#            differential output movement = differential trust.
# Run from experiments/.
set -e
cd "$(dirname "$0")/.."
mkdir -p a2a_forward/logs
modal run --detach a2a_forward/prediction_trust.py::a2a_prediction_trust \
    2>&1 | tee a2a_forward/logs/prediction_trust_$(date +%Y%m%d_%H%M%S).log
