#!/usr/bin/env bash
# verify the mirror, then write results/tables.md and figs/.
set -eu
M=${1:?usage: reduce.sh <local-mirror> [tag]}
TAG=${2:-}
HERE=$(cd "$(dirname "$0")/../../../../.." && pwd)     # -> experiments/
python "$(dirname "$0")/verify.py" "$M"
cd "$HERE"
python -m rhm.logit_reading.orbitofrontal.regime.analyze --root "$M" ${TAG:+--tag "$TAG"} \
    --out rhm/logit_reading/orbitofrontal/regime/results \
    --figs rhm/logit_reading/orbitofrontal/regime/figs
