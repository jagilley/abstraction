#!/bin/sh
# [rb_r1] GATE R-1 — saved at c, restored in a FRESH container, continued to the stop: the same
# arm in every logged quantity, every table and every weight. On `preflight`'s toy substrate with
# its admission twin `st_pf_gn` (every organ `st_gn_yk` carries is live), so the gate costs
# minutes, not an arm. A CPU coordinator (`rb_gate_r1`) runs, each in its OWN container
# (`single_use_containers`):
#   P   the yoke source `voi3b_pf_src`, once (the plan every other run replays)  -- reused from
#       the machinery smoke (`rb_r1_P`, same code path) with --reuse-plan
#   A   uninterrupted to the stop, saving EVERY cycle and every era boundary
#   A2  uninterrupted to the stop in another container, saving only at the stop: the
#       cross-container floor and the save's inertness; plus the DONOR's st_pf_gn on A2's
#       substrate (fidelity on the arm of record itself)
#   B1  A's state at an era boundary, restored fresh, continued
#   B2  A's state mid-era, restored fresh, continued
#   F1  B1 with the global torch CPU stream left unrestored   (must be RED)
#   F2  B1 with the plant's numpy stream `grng` left unrestored (must be RED)
# The coordinator writes `rhm_practice_rubato/rb_r1/r1_gate.json`; `reduce_rubato.py --r1` tables it.
cd "$(dirname "$0")/../../../../../../.."        # -> experiments/
export MODAL_PROFILE=chromatic
python3 rhm/practice/voicing/sotto_voce/aliquot/rubato/launch_detached.py --fn rb_gate_r1 \
    --tag rb_r1 --reuse-plan
