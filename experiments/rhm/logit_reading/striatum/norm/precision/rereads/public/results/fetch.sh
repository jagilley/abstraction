#!/usr/bin/env bash
# Fetch this node's cells (and, for s42, projection's) off the `rhm-scaling-data` volume
# (`chromatic` workspace) into <pubmirror>/traj_a1_s4X/.  Sequential on purpose: parallel
# `modal volume get` calls dropped files in this session.  Existing files are skipped; the
# reduction's section G touches every member of every npz it reads.
#   bash rhm/logit_reading/striatum/norm/precision/rereads/public/results/fetch.sh <pubmirror>
set -u
OUT=${1:?usage: fetch.sh <pubmirror>}
D=/v16_s2_L6_m4_distinct/logit_reading
for s in 42 43 44; do
  mkdir -p "$OUT/traj_a1_s$s"
  for st in 000000 008000 064000; do
    for tg in a1 swap65k; do
      fs="norm_${tg}_public.json norm_${tg}_public.npz norm_${tg}_public_twins.npz readout_basis.npz"
      if [ "$s" = 42 ]; then
        fs="$fs proj_${tg}.json proj_${tg}_rows_tv.npz proj_${tg}_rows_fd.npz proj_${tg}_twins.npz"
      fi
      for f in $fs; do
        dst="$OUT/traj_a1_s$s/step${st}_$f"
        [ -s "$dst" ] && continue
        modal volume get rhm-scaling-data "$D/traj_a1_s$s/step${st}_$f" "$dst" >/dev/null 2>&1 \
          || echo "FAILED $dst"
      done
    done
  done
done
ls "$OUT"/traj_a1_s4*/ | grep -c public
