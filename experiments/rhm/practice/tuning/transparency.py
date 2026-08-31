"""tuning/transparency — the three gates that license everything Gate 0 reports.

FORK NOTICE: `../teacher_slot/endo_yield/transparency.py`'s idiom, extended with the FM
gate and the donor-fidelity gate.

  (a) THE PANEL MOVES NOTHING.  `--shadow` vs `--no-shadow`. The panel adds forward passes
      under `no_grad` at every checkpoint; none of them may touch the model, the donor's
      instruments, or the trajectory.
  (b) THE ATTACHED FM MOVES NOTHING.  `--fm` vs `--no-fm` (both with `--no-shadow`, so the
      panel is out of the picture). The FM reads activations detached, owns its optimiser,
      and its construction is wrapped in a global-RNG save/restore, so an arm with an
      uncharged FM must be bit-identical to the same arm without one. The one thing this
      gate can genuinely catch is a kernel-level effect (the FM's allocations changing a
      cuBLAS algorithm choice) — which is exactly why it is measured rather than argued.
  (c) DONOR FIDELITY.  Our `wall` / `no_wall` against every stored twin that shares a
      prefix with them — fwlm0, fwlm1, tsdB, ey0 — over every checkpoint strictly BEFORE
      the first burst, on the donor's own five instruments.

WHAT MUST BE EXACTLY ZERO: the per-position NLL trajectory (`nll_pos_true`) and every
model-derived instrument. Reference-derived quantities (`excess.*`) carry a ~4e-16 BP
recomputation term when the references were recomputed rather than loaded from the donor's
`setup.json`; in a full run they are loaded, so that term is identically zero.

Usage (from experiments/):
    python3 rhm/practice/tuning/transparency.py --fetch                 # gates a + b
    python3 rhm/practice/tuning/transparency.py --fidelity --fetch      # gate c
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_tuning"

# model-derived only: nothing here touches the BP references
HARD = ["nll.true.idx", "nll.true.out", "nll.true.all", "nll.none.idx", "nll.rand.idx",
        "nll.perm.idx", "nll.old.idx", "binding.idx", "binding.misleading_idx",
        "binding.stale_idx", "binding.vs_none_idx",
        "index.transfer_gap", "index.jsd_mean", "index.map_match_cur", "index.e_cur"]
SOFT = ["excess.true.all", "excess.true.idx", "excess_by_level.1"]
# the donor's own gate-9a five
FIVE = ["nll.true.idx", "nll.true.out", "binding.idx",
        "index.transfer_gap", "index.jsd_mean"]

# every stored twin whose trajectory must coincide with ours before the first burst
DONORS = [
    ("rhm_practice_fourwall_lm", "fwlm0", {"wall": "wall", "no_wall": "no_wall"}),
    ("rhm_practice_fourwall_lm", "fwlm1", {"wall": "merge_8000", "no_wall": "no_wall"}),
    ("rhm_practice_teacher_slot_decision", "tsdB", {"wall": "outer_task"}),
    ("rhm_practice_teacher_slot_endo_yield", "ey0", {"no_wall": "no_wall"}),
]
# tsdB's `outer_task` is a plain `wall` arm only until its first ABBA collapse QUARTER,
# which begins at step 1000 (its s1000 checkpoint already reports consumed == "none"); the
# weights at s1000 are still pure-wall, so s1125 is the first non-comparable checkpoint.
DONOR_MAX_STEP = {"tsdB": 1125}


def fetch(remote, tag, dest=None):
    dest = dest or FIG
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{remote}/{tag}", dest],
                   check=True)


def dig(d, parts):
    for p in parts:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _load(root, arm):
    p = os.path.join(root, f"{arm}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def compare(a_root, b_root, arms, paths, max_step=None, label=""):
    """max|delta| on `paths` over shared checkpoints. Returns (ok, table rows)."""
    rows, ok = [], True
    for arm_a, arm_b in arms:
        A, Bd = _load(a_root, arm_a), _load(b_root, arm_b)
        if A is None or Bd is None:
            rows.append((arm_a, arm_b, 0, None, None, None))
            continue
        bm = {r["step"]: r for r in Bd["log"]}
        traj = hard = soft = 0.0
        n = 0
        for r in A["log"]:
            if max_step is not None and r["step"] >= max_step:
                continue
            q = bm.get(r["step"])
            if q is None:
                continue
            n += 1
            if "nll_pos_true" in r and "nll_pos_true" in q:
                traj = max(traj, max(abs(x - y) for x, y in
                                     zip(r["nll_pos_true"], q["nll_pos_true"])))
            for grp, which in ((paths, "h"), (SOFT, "s")):
                for p in grp:
                    x, y = dig(r, p.split(".")), dig(q, p.split("."))
                    if x is None or y is None:
                        continue
                    d = abs(float(x) - float(y))
                    if which == "h":
                        hard = max(hard, d)
                    else:
                        soft = max(soft, d)
        good = n > 0 and traj == 0.0 and hard == 0.0
        ok &= good
        rows.append((arm_a, arm_b, n, traj, hard, soft))
    print(f"\n  {label}")
    print(f"    {'ours':14s} {'twin':16s} {'ckpts':>6s} {'traj max|d|':>13s} "
          f"{'hard max|d|':>13s} {'soft max|d|':>13s}")
    for a, b, n, t, h, s in rows:
        if t is None:
            print(f"    {a:14s} {b:16s} {'-':>6s}  (absent)")
            continue
        print(f"    {a:14s} {b:16s} {n:6d} {t:13.3e} {h:13.3e} {s:13.3e}")
    return ok


def gates_ab(a_tag, ns_tag, nf_tag):
    arms = [("wall", "wall"), ("no_wall", "no_wall"), ("wall_skip", "wall_skip")]
    ok_a = compare(os.path.join(FIG, a_tag), os.path.join(FIG, ns_tag), arms, HARD,
                   label=f"(a) THE PANEL MOVES NOTHING   {a_tag} (shadow) vs "
                         f"{ns_tag} (no-shadow)")
    ok_b = compare(os.path.join(FIG, ns_tag), os.path.join(FIG, nf_tag), arms, HARD,
                   label=f"(b) THE FM MOVES NOTHING      {ns_tag} (fm) vs "
                         f"{nf_tag} (no-fm)")
    ok_ab = compare(os.path.join(FIG, a_tag), os.path.join(FIG, nf_tag), arms, HARD,
                    label=f"(a+b) BOTH TOGETHER           {a_tag} vs {nf_tag} "
                          f"(= the donor's code path)")
    print(f"\n  GATE (a) panel: {'PASS' if ok_a else 'FAIL'}"
          f"   GATE (b) FM: {'PASS' if ok_b else 'FAIL'}"
          f"   combined: {'PASS' if ok_ab else 'FAIL'}")
    return ok_a and ok_b and ok_ab


def gate_c(tag, first_burst, do_fetch=False):
    root = os.path.join(FIG, tag)
    ok = True
    # internal: the skip arm is a `wall` arm until its first skipped step, and it lives in
    # a DIFFERENT worker — so this doubles as the cross-worker reproducibility check.
    if os.path.exists(os.path.join(root, "wall_skip.json")):
        ok &= compare(root, root, [("wall_skip", "wall")], FIVE, max_step=first_burst,
                      label=f"(c0) INTERNAL: wall_skip vs wall, different workers "
                            f"(steps < {first_burst})")
    for remote, dtag, mapping in DONORS:
        dest = os.path.join(FIG, "_donors")
        droot = os.path.join(dest, dtag)
        if do_fetch and not os.path.exists(droot):
            try:
                fetch(remote, dtag, dest)
            except subprocess.CalledProcessError:
                print(f"    [{dtag}] not on the volume — skipped")
                continue
        if not os.path.exists(droot):
            # fall back to the donor's locally committed figures/
            local = {
                "fwlm0": "../fourwall/lm/figures/fwlm0",
                "fwlm1": "../fourwall/lm/figures/fwlm1",
                "tsdB": "../teacher_slot/decision/figures/tsdB",
                "ey0": "../teacher_slot/endo_yield/figures/ey0",
            }.get(dtag)
            cand = os.path.normpath(os.path.join(HERE, local)) if local else None
            if cand and os.path.exists(cand):
                droot = cand
            else:
                print(f"    [{dtag}] absent locally and not fetched — skipped")
                continue
        cap = min(first_burst, DONOR_MAX_STEP.get(dtag, 10 ** 9))
        ok &= compare(root, droot, list(mapping.items()), FIVE, max_step=cap,
                      label=f"(c) DONOR FIDELITY vs {dtag} (steps < {cap})")
    print(f"\n  GATE (c) donor fidelity: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tn_smoke")
    ap.add_argument("--ns-tag", default="tn_smoke_ns")
    ap.add_argument("--nf-tag", default="tn_smoke_nf")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--fidelity", action="store_true",
                    help="run gate (c) against the donor twins for --tag")
    ap.add_argument("--first-burst", type=int, default=5000)
    a = ap.parse_args()
    if a.fetch:
        for t in ([a.tag] if a.fidelity else [a.tag, a.ns_tag, a.nf_tag]):
            fetch(REMOTE, t)
    print("=" * 78)
    print(f"TRANSPARENCY GATES  ({'fidelity' if a.fidelity else 'panel + FM'})")
    print("=" * 78)
    ok = gate_c(a.tag, a.first_burst, a.fetch) if a.fidelity \
        else gates_ab(a.tag, a.ns_tag, a.nf_tag)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
