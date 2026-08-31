"""tuning/transparency_g1 — the Gate-1 fidelity gate.

FORK NOTICE: `transparency.py`'s `compare` is imported unmodified; only the twin map and the
prefix logic are new.

WHICH PREFIX IS GATE-CHECKABLE, AND WHY IT IS TWO DIFFERENT NUMBERS.

Gate 1's stream carries three event types. Its first divergence from each reference is the
first step at which its stream stops being that reference's stream:

  vs the DONORS (fwlm0, fwlm1, tsdB, ey0)      steps < 5000
      They have no burst and no drift, so the shared prefix ends at Gate 1's first burst.
  vs GATE 0 (`tn0`)                            steps < 6000
      `tn0` and Gate 1 share the burst schedule, the burst seed and the FIRST window's rate
      (rho = 0.0242 at step 5000), so the streams stay identical THROUGH that burst; they part
      at Gate 1's first drift, where the grammar itself changes.
  vs EACH OTHER (arm vs arm, in-tag)           steps < the arm's first fired op
      Every arm consumes the same stream until its policy acts, so any two arms are
      bit-identical up to the earlier of their first ops. That is not asserted — a treatment
      arm firing before the first event is a FALSE FIRE, a result, and the gate prints the
      first-op step per arm instead of hiding it.

`pair_parse` is the `no_wall` twin (it never acts); `track` is the `wall` twin.

Usage (from experiments/):
    python3 rhm/practice/tuning/transparency_g1.py --tag g1a --fetch
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transparency import FIVE, HARD, compare, dig            # noqa: E402

FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_tuning_g1"

# (donor remote, tag, {our arm: their arm}, max shared step)
REFS = [
    ("rhm_practice_fourwall_lm", "fwlm0",
     {"track": "wall", "pair_parse": "no_wall"}, 5000),
    ("rhm_practice_fourwall_lm", "fwlm1",
     {"track": "merge_8000", "pair_parse": "no_wall"}, 5000),
    ("rhm_practice_teacher_slot_decision", "tsdB", {"track": "outer_task"}, 1125),
    ("rhm_practice_teacher_slot_endo_yield", "ey0", {"pair_parse": "no_wall"}, 5000),
    ("rhm_practice_tuning", "tn0", {"track": "wall", "pair_parse": "no_wall"}, 6000),
]
LOCAL = {
    "fwlm0": "../fourwall/lm/figures/fwlm0",
    "fwlm1": "../fourwall/lm/figures/fwlm1",
    "tsdB": "../teacher_slot/decision/figures/tsdB",
    "ey0": "../teacher_slot/endo_yield/figures/ey0",
    "tn0": "figures/tn0",
}


def fetch(remote, tag, dest):
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{remote}/{tag}", dest],
                   check=True)


def first_op(root, arm):
    """The step at which this arm's policy first changed the stream (None = never)."""
    p = os.path.join(root, f"{arm}.json")
    if not os.path.exists(p):
        return None, None
    obj = json.load(open(p))
    for o in obj.get("ops", []):
        if o.get("op") in ("merge", "skip"):
            return o["step"], o.get("type")
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g1a")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    root = os.path.join(FIG, a.tag)
    if a.fetch:
        fetch(REMOTE, a.tag, FIG)
    print("=" * 78)
    print(f"GATE 1 FIDELITY  —  tag {a.tag}")
    print("=" * 78)

    arms = sorted(x[:-5] for x in os.listdir(root)
                  if x.endswith(".json") and x not in ("setup.json",)
                  and not x.startswith("panel_"))
    print("\n  FIRST FIRED OP per arm (a fire before the first event at 5000 is a FALSE "
          "FIRE, reported not asserted):")
    firsts = {}
    for arm in arms:
        st, typ = first_op(root, arm)
        firsts[arm] = st
        flag = ""
        if st is not None and st < 5000:
            flag = "   <-- BEFORE THE FIRST EVENT"
        print(f"    {arm:14s} {('s' + str(st) + '  (' + str(typ) + ')') if st else 'never':>24s}{flag}")

    ok = True
    for remote, dtag, mapping, cap in REFS:
        droot = os.path.join(FIG, "_donors", dtag)
        if a.fetch and not os.path.exists(droot):
            try:
                fetch(remote, dtag, os.path.join(FIG, "_donors"))
            except subprocess.CalledProcessError:
                pass
        if not os.path.exists(droot):
            cand = os.path.normpath(os.path.join(HERE, LOCAL.get(dtag, "")))
            if os.path.exists(cand):
                droot = cand
            else:
                print(f"\n    [{dtag}] not available — skipped")
                continue
        pairs = [(o, t) for o, t in mapping.items()
                 if os.path.exists(os.path.join(root, f"{o}.json"))]
        if not pairs:
            continue
        # an arm that fired before the cap shortens its own comparable prefix
        eff = min([cap] + [firsts[o] for o, _ in pairs
                           if firsts.get(o) is not None and firsts[o] < cap])
        ok &= compare(root, droot, pairs, FIVE, max_step=eff,
                      label=f"vs {dtag} (steps < {eff})")

    # in-tag: every arm against `track` up to the earlier of their first ops
    others = [x for x in arms if x not in ("track", "pair_parse")]
    print()
    for arm in others:
        cap = min([x for x in (firsts.get(arm), firsts.get("track"), 10 ** 9)
                   if x is not None])
        compare(root, root, [(arm, "track")], FIVE, max_step=cap,
                label=f"in-tag: {arm} vs track (steps < {cap})")

    print(f"\n  GATE 1 FIDELITY: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
