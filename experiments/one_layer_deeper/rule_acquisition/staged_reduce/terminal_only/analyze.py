"""Aggregate `terminal_only` results against the donor's `staged_reduce` cells.

Four tables:

  1. **The 2x2.** `chain_heldout_y` and `chain_heldout_n` for {8, 142 moduli} x
     {monolithic, staged-supervised, terminal-only}. The donor cells are read straight out
     of `../results/` — the pools are bit-identical, so the numbers sit in one table.
  2. **The stage probe.** The tied stage map applied *once* to a bounded-quotient query
     `(N, y')`, `y' ~ U[0, R*N)` — the donor's *training* distribution, which a terminal-only
     arm never sees. `sr3_r10` reads 0.9978 there having trained on it.
  3. **Per-stage diagnostics.** Whether each intermediate is the true partial remainder, a
     relabelled one (purity above its within-modulus permutation null), a copy of its
     predecessor, or nothing.
  4. **Budget.** Tail power-law slope of each headline curve, with a decay flag, so
     "still closing" and "asymptoted" are distinguishable rather than asserted.

  python3 one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/analyze.py \\
      --tags to3a,to3b,to3c,to3d
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DONOR = HERE.parent / "results"
HARD_EXAMPLES = 768

# Donor cells this node reads against. `(tag, arm)` under `../results/`.
DONOR_CELLS = [
    ("sr3b", "sr3_mono", "8", "monolithic, terminal label"),
    ("sr3b", "sr3_mono_inner6", "8", "monolithic + 6 inner ops (compute control)"),
    ("sr3a", "sr3_r10", "8", "staged, single-stage supervision (test-time chain)"),
    ("sr3m8", "sr3_r10_m8", "8*", "staged, single-stage supervision, family-matched"),
    ("srb1", "sr3_mono_many", "142", "monolithic, terminal label"),
    ("srb1", "sr3_r10_many", "142", "staged, single-stage supervision (test-time chain)"),
]


def certifiable_T(eps: float) -> float:
    return float("inf") if eps <= 0 else math.log(2.0) / (HARD_EXAMPLES * eps)


def _load(root: Path, tags: list[str], seed: int) -> dict:
    arms = {}
    for tag in tags:
        p = root / tag / f"results_seed{seed}.json"
        if not p.exists():
            print(f"[miss] {p}")
            continue
        d = json.loads(p.read_text())
        for name, rec in d["arms"].items():
            rec["_tag"] = tag
            rec["_spec"] = d.get("arm_specs", {}).get(name, {})
            arms[f"{name}@{tag}"] = rec
    return arms


def _tail_slope(log: list[dict], key: str, frac: float = 0.5) -> tuple[float, bool]:
    """Power-law slope of `eps` against step over the last `frac` of the curve."""
    pts = [(r["step"], r[key]) for r in log if key in r and r[key] is not None and r[key] > 0]
    if len(pts) < 4:
        return float("nan"), False
    pts = pts[max(0, int(len(pts) * (1 - frac))):]
    xs = [math.log(s) for s, _ in pts]
    ys = [math.log(e) for _, e in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = float("nan") if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    best = min(e for _, e in pts)
    return slope, pts[-1][1] > best * 1.05  # decayed from its own best


def _cell(rec: dict, key: str) -> dict | None:
    fc = rec.get("final_chain") or {}
    return fc.get(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    mine = _load(RES, tags, args.seed)
    donor = {}
    for tag, arm, _, _ in DONOR_CELLS:
        d = _load(DONOR, [tag], 0)
        if f"{arm}@{tag}" in d:
            donor[f"{arm}@{tag}"] = d[f"{arm}@{tag}"]

    # ---------------------------------------------------------------- table 1
    print("\n" + "=" * 108)
    print("1. THE 2x2 — chain accuracy on the donor's (bit-identical) chain pools")
    print("=" * 108)
    hdr = (f"{'arm':<26} {'mods':>5} {'steps':>7} {'held-out y':>11} {'floor':>7} "
           f"{'held-out N':>11} {'floor':>7} {'eps':>9} {'cert_T':>7}  what")
    print(hdr)
    print("-" * 108)

    def row(name, rec, what, key_y="chain_heldout_y", key_n="chain_heldout_n"):
        fl = rec.get("floor_no_reduction", {})
        y = _cell(rec, key_y) or {}
        n = _cell(rec, key_n) or {}
        steps = rec["train_log"][-1]["step"] if rec.get("train_log") else 0
        flag = "" if rec.get("complete", True) else "  [PARTIAL]"
        print(f"{name:<26} {rec.get('n_train_moduli', '?'):>5} {steps:>7} "
              f"{y.get('acc', float('nan')):>11.4f} "
              f"{fl.get('chain_heldout_y', float('nan')):>7.4f} "
              f"{n.get('acc', float('nan')):>11.4f} "
              f"{fl.get('chain_heldout_n', float('nan')):>7.4f} "
              f"{y.get('eps', float('nan')):>9.2e} "
              f"{certifiable_T(y.get('eps', 1.0)):>7.3f}  {what}{flag}")

    for tag, arm, _, what in DONOR_CELLS:
        k = f"{arm}@{tag}"
        if k in donor:
            row(f"[donor] {arm}", donor[k], what)
    print("-" * 108)
    for k, rec in sorted(mine.items()):
        sm = rec.get("stage_mode", "?")
        row(k.split("@")[0], rec, f"terminal-only staged forward ({sm})")
        if "chainhard_heldout_y" in (rec.get("final_chain") or {}):
            row(f"  ^ hard chain", rec, "same weights, argmax re-ground",
                key_y="chainhard_heldout_y", key_n="chainhard_heldout_n")

    # ---------------------------------------------------------------- table 2
    print("\n" + "=" * 108)
    print("2. THE STAGE PROBE — one application of the tied map to `(N, y')`, y' ~ U[0, R*N)")
    print("   (the donor's *training* distribution; a terminal-only arm has never seen it)")
    print("=" * 108)
    print(f"{'arm':<26} {'stage_all':>10} {'floor':>7} {'heldout_y':>10} {'heldout_N':>10} "
          f"{'floor':>7} {'teacher stage_acc':>18}")
    print("-" * 108)
    for tag, arm, _, _ in DONOR_CELLS:
        k = f"{arm}@{tag}"
        if k not in donor:
            continue
        rec, f_ = donor[k], donor[k].get("floor_no_reduction", {})
        fin = rec.get("final", {})
        t = (rec.get("final_chain_teacher") or {}).get("chain_heldout_y", {})
        print(f"{'[donor] ' + arm:<26} {'-':>10} {'-':>7} "
              f"{fin.get('heldout_y', {}).get('acc', float('nan')):>10.4f} "
              f"{fin.get('heldout_n', {}).get('acc', float('nan')):>10.4f} "
              f"{f_.get('heldout_n', float('nan')):>7.4f} "
              f"{t.get('stage_acc', float('nan')):>18.4f}")
    print("-" * 108)
    for k, rec in sorted(mine.items()):
        fin, f_ = rec.get("final", {}), rec.get("floor_no_reduction", {})
        t = (rec.get("final_chain_teacher") or {}).get("chain_heldout_y", {})
        print(f"{k.split('@')[0]:<26} "
              f"{fin.get('stage_all', {}).get('acc', float('nan')):>10.4f} "
              f"{f_.get('stage_all', float('nan')):>7.4f} "
              f"{fin.get('heldout_y', {}).get('acc', float('nan')):>10.4f} "
              f"{fin.get('heldout_n', {}).get('acc', float('nan')):>10.4f} "
              f"{f_.get('heldout_n', float('nan')):>7.4f} "
              f"{t.get('stage_acc', float('nan')):>18.4f}")

    # ---------------------------------------------------------------- table 3
    print("\n" + "=" * 108)
    print("3. PER-STAGE DIAGNOSTICS — what the intermediate remainders actually are")
    print("   true   : decoded == (y // R^i) mod N          identity: == its predecessor")
    print("   purity : decoded is the group mode given (N, true partial remainder),")
    print("            against a within-modulus permutation null — a *relabelled* residue")
    print("            reads purity >> null with true ~ 0")
    print("=" * 108)
    for k, rec in sorted(mine.items()):
        probe = rec.get("final_stage_probe") or {}
        for pool, p in probe.items():
            print(f"\n{k.split('@')[0]}  |  {pool}  (n={p['n']}, mode={p['mode']})")
            print(f"  {'st':>2} {'dig':>3} {'true':>7} {'in_rng':>7} {'ident':>7} {'modal':>7} "
                  f"{'purity':>7} {'null':>7} {'inv_pur':>8} {'null':>7} {'distinct':>8}")
            for r in p["stages"]:
                ident = "     -" if r["identity"] is None else f"{r['identity']:>7.4f}"
                print(f"  {r['stage']:>2} {r['digit_index']:>3} {r['true_match']:>7.4f} "
                      f"{r['in_range']:>7.4f} {ident} {r['modal_share']:>7.4f} "
                      f"{r['purity']:>7.4f} {r.get('purity_null', float('nan')):>7.4f} "
                      f"{r['inv_purity']:>8.4f} {r.get('inv_purity_null', float('nan')):>7.4f} "
                      f"{r['n_distinct']:>8}")

    # ---------------------------------------------------------------- table 4
    print("\n" + "=" * 108)
    print("4. BUDGET — tail power-law slope of eps vs step (last half of the curve)")
    print("=" * 108)
    print(f"{'arm':<26} {'curve':<26} {'end eps':>10} {'best eps':>10} {'@step':>8} "
          f"{'slope':>8}  decayed")
    print("-" * 108)
    for k, rec in sorted(mine.items()):
        log = rec.get("train_log") or []
        for key in ("chain_heldout_y.eps", "chain_heldout_n.eps", "stage_all.eps"):
            pts = [(r["step"], r[key]) for r in log if key in r and r[key] is not None]
            if not pts:
                continue
            slope, decayed = _tail_slope(log, key)
            best = min(pts, key=lambda t: t[1])
            print(f"{k.split('@')[0]:<26} {key:<26} {pts[-1][1]:>10.3e} {best[1]:>10.3e} "
                  f"{best[0]:>8} {slope:>8.2f}  {'YES' if decayed else '-'}")


if __name__ == "__main__":
    main()
