"""FORK FIDELITY GATE (SPEC.md Phase A#1) -- `committee_head`'s `value` arm at K=1, rpf_beta=0 must
reproduce E3's `value` arm BIT-IDENTICALLY.

Why this has to be a check and not an intention. `committee_head.py` adds a lot to E3's loop: a
committee, a disagreement read, an online benchmark net b(s), an arity-1 net for the agency gap, and
a head that is trained on EVERY arm. Each of those is a chance to consume a shared RNG draw at a
different moment and silently shift every subsequent sample -- the exact failure mode
`on_policy/verify_backcompat.py` was written to catch for `collection_mode`. If the fork's incumbent
is not the incumbent, the whole ladder is measuring two things at once.

What is compared: every per-round quantity the loop logs on the `value` arm (allocation, in-region
collection share, survey error, per-region grader error, visits, lprog, metered step counts,
ballistic and reactive control), plus the setup-level artifacts (region centers, matched-FM ceiling).
`array_equal`, not `allclose` -- an approximate match here would mean an RNG stream diverged and
happened to land somewhere similar.

MEASURED, 2026-09-03. Every per-round quantity matched across all 14 rounds. ONE region center
differed by 1.110e-16 (0.5 ulp -- one bit of the float64 mantissa). Running E3 against *itself* on a
second container reproduces the SAME deviation, 1.110e-16, exactly -- so it is host-level round-off
in the pure-numpy geometry (reduction kernels differ with the host's CPU features), not a fork
divergence, and the geometry block is byte-identical between the two files besides. That self-check
is the reason `--e3`/`--ch` accept two E3 runs; re-run it before ever believing a geometry FAIL here.

Run (both sides, then compare):
    cd experiments/            # MODAL_PROFILE=chromatic
    modal run mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy \
        --quick --policies value --tag fid_e3
    modal run mjc/committee_head/committee_head.py::committee_head \
        --quick --policies value --k-members 1 --rpf-beta 0 --tag fid_ch
    python3 mjc/committee_head/check_fidelity.py

Both entrypoints mirror their results to `figures/<node>_<tag>/results.json` locally, which is what
this script reads; `--e3` / `--ch` override the paths.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_E3 = os.path.join(_HERE, "..", "on_policy", "directed_on_policy", "figures",
                   "directed_on_policy_fid_e3", "results.json")
_CH = os.path.join(_HERE, "figures", "committee_head_fid_ch", "results.json")

# per-round keys that must match exactly. `reactive` is present only on some rounds, so it is
# compared where both sides have it.
ROUND_KEYS = ["alloc", "in_share", "per_err_mon", "reg_err", "visits", "lprog", "stale_mask",
              "b_state", "ballistic_cem", "mon_steps", "coll_steps"]
SETUP_KEYS = ["region_names", "ceil_err", "P0", "G"]


def _eq(a, b):
    """array_equal with NaN treated as equal to NaN (per-region grader error is NaN where a region
    has no probe transitions -- E3's off-reach measurement hole, which must reproduce too)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", default=_E3)
    ap.add_argument("--ch", default=_CH)
    ap.add_argument("--policy", default="value")
    a = ap.parse_args()

    for p in (a.e3, a.ch):
        if not os.path.exists(p):
            print(f"[FAIL] missing {p}\n       run both sides first (see this file's docstring).")
            return 1
    E = json.load(open(a.e3)); C = json.load(open(a.ch))

    fails, checks = [], 0

    def check(name, ok, detail=""):
        nonlocal checks
        checks += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print(f"\n=== fork fidelity: committee_head(K=1, beta=0) vs E3, policy={a.policy!r} ===")
    kc = C["config"]
    # The `--ch` side is normally a committee_head run, but pointing both sides at E3 runs turns
    # this into an E3-AGAINST-ITSELF self-consistency check -- which is how the region-center
    # round-off below is told apart from a fork divergence. So these two are skipped, not failed,
    # when the second side is not a committee run.
    if "k_members" in kc:
        check("K=1", int(kc["k_members"]) == 1, f"k_members={kc['k_members']}")
        check("rpf_beta=0", float(kc["rpf_beta"]) == 0.0, f"rpf_beta={kc['rpf_beta']}")
    else:
        print("  [self-check mode] both sides are E3 runs; committee knobs not applicable")
    check("shared-substrate config matches",
          all(E["config"].get(k) == kc.get(k) for k in E["config"] if k in kc and k not in
              ("tag", "policies", "render_rounds", "render_policies", "render_n")),
          "every E3 knob present here holds the same value")

    for k in SETUP_KEYS:
        if k == "region_names":
            check(f"setup:{k}", E[k] == C[k])
        else:
            check(f"setup:{k}", _eq(E[k], C[k]))
    # Region centers are reported WITH their deviation rather than as a bare pass/fail. They are
    # produced by pure numpy geometry (no RNG, no torch) that is byte-identical between the two
    # files, so any difference here is float64 round-off from the host's reduction kernels, not a
    # fork divergence -- and the way to tell those apart is the magnitude, next to the same
    # comparison run E3-against-E3 (`--e3 <run A> --ch <run B>`). Anything above round-off, or any
    # deviation at all accompanied by a per-round failure, is a real divergence.
    dev = max(float(np.max(np.abs(np.asarray(E["region_centers"][n], float)
                                  - np.asarray(C["region_centers"][n], float))))
              for n in E["region_names"])
    scale = max(float(np.max(np.abs(np.asarray(list(E["region_centers"].values()), float)))), 1e-12)
    check("setup:region_centers (float64 round-off tolerated; see note)",
          dev <= 64 * np.finfo(np.float64).eps * scale,
          f"max abs deviation {dev:.3e}  ({dev / scale / np.finfo(np.float64).eps:.1f} ulp)")

    pe, pc = E["results"].get(a.policy), C["results"].get(a.policy)
    if pe is None or pc is None:
        print(f"[FAIL] policy {a.policy!r} missing from one side")
        return 1
    check("round count", len(pe) == len(pc), f"E3={len(pe)} fork={len(pc)}")
    nr = min(len(pe), len(pc))
    for key in ROUND_KEYS:
        bad = [t for t in range(nr) if not _eq(pe[t][key], pc[t][key])]
        d = ""
        if bad:
            t = bad[0]
            d = f"first divergence round {t}: E3={pe[t][key]} fork={pc[t][key]}"
        check(f"per-round:{key}", not bad, d or f"{nr} rounds identical")
    rb = [t for t in range(nr) if ("reactive" in pe[t]) != ("reactive" in pc[t])
          or ("reactive" in pe[t] and not _eq(pe[t]["reactive"], pc[t]["reactive"]))]
    check("per-round:reactive", not rb)

    print(f"\n{'PASS' if not fails else 'FAIL'}: {checks - len(fails)}/{checks} checks"
          + (f"  failures={fails}" if fails else "  -- the fork's incumbent IS E3's incumbent."))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
