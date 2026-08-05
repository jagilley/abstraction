"""Merge the rule_acquisition sweep's per-family result files into one table.

The sweep is split across tags so the four jobs can run in parallel without clobbering a
shared output path; this reassembles them. Every readout is reported against the *floor
computed on that arm's own eval pool*, because the floors differ by pool and by task and a
raw accuracy number means nothing without one.

Usage:
  python3 one_layer_deeper/rule_acquisition/analyze.py --tags cut1_task,cut1_repr,cut1_compute,cut1_stack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY_OF = {
    "sq": "T baseline",
    "mul2": "T reduction only",
    "sqnomod": "T multiply only",
    "redmod": "T reduce only",
    "redmod_q8": "T reduce q<=8",
    "redmod_q64": "T reduce q<=64",
    "redmod_qfull": "T reduce q<N",
    "redmod_q64_many": "T reduce q<=64 x142",
    "redmod_qfull_many": "T reduce q<N x142",
    "auxprod": "T staged supervision",
    "binary": "R base-2",
    "abacus": "R place-value",
    "enc8": "C depth",
    "inner8": "C inner steps",
    "wide4k": "C width",
    "manymod": "D pressure",
    "stack": "all",
}
ORDER = list(FAMILY_OF)


def load(tags: list[str], seed: int) -> dict:
    arms = {}
    for tag in tags:
        path = HERE / "results" / tag / f"results_seed{seed}.json"
        if not path.exists():
            print(f"[missing] {path}")
            continue
        blob = json.loads(path.read_text())
        for name, res in blob["arms"].items():
            res["_spec"] = blob.get("arm_specs", {}).get(name, {})
            arms[name] = res
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cut1_task,cut1_repr,cut1_compute,cut1_stack")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = load([t.strip() for t in args.tags.split(",") if t.strip()], args.seed)
    if not arms:
        raise SystemExit("no results found — fetch them from the volume first")

    names = [n for n in ORDER if n in arms] + [n for n in arms if n not in ORDER]

    print(f"\n{'arm':9s} {'family':17s} {'mods':>4s} {'params':>10s} {'train':>6s} "
          f"{'ID@1':>6s} {'heldx':>6s} {'floor':>6s} {'lift':>6s} {'heldN':>6s} {'ID@2':>6s}")
    print("-" * 96)
    for n in names:
        r = arms[n]
        ex, fl = r["exact"], r["floor_no_reduction"]
        heldx = ex["seen_n_heldout_x"]["1"]
        floor = fl["seen_n_heldout_x"]
        id2 = ex["seen_n_seen_x"].get("2")
        train = r["train_log"][-1]["train_exact"] if r["train_log"] else float("nan")
        print(
            f"{n:9s} {FAMILY_OF.get(n, ''):17s} {r['family']['n_train_moduli']:>4d} "
            f"{r['n_params']:>10,} {train:>6.3f} {ex['seen_n_seen_x']['1']:>6.3f} "
            f"{heldx:>6.3f} {floor:>6.3f} {heldx - floor:>+6.3f} "
            f"{ex['heldout_n']['1']:>6.3f} {'—' if id2 is None else format(id2, '>6.3f')}"
        )

    # Group structure, for the arms where the coordinate was computable.
    print(f"\n{'arm':9s} {'fourier':>8s} {'(null)':>8s} {'transR2':>8s} {'(null)':>8s}  n_moduli")
    print("-" * 60)
    for n in names:
        g = arms[n].get("group_structure_mean")
        if not g:
            continue
        print(
            f"{n:9s} {g['fourier']:>8.4f} {g['fourier_null']:>8.4f} "
            f"{g['translation_r2']:>8.4f} {g['translation_r2_null']:>8.4f}"
            f"  {len(arms[n]['group_structure'])}"
        )

    # Held-out-x against training step: the readout that separates "never generalised" from
    # "generalised late". A flat trace over the whole budget is the negative result; a rising
    # tail is a reason to extend that arm rather than to conclude anything.
    print("\nheld-out x @1 over training (grokking check)")
    for n in names:
        log = arms[n]["train_log"]
        if not log:
            continue
        pts = [log[i] for i in range(0, len(log), max(1, len(log) // 8))][:9]
        trace = " ".join(f"{p['step']//1000}k:{p['seen_n_heldout_x@1']:.3f}" for p in pts)
        print(f"  {n:9s} {trace}")


if __name__ == "__main__":
    main()
