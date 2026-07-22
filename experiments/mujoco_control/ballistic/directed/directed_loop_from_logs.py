"""Recover `directed_loop.py` results from the printed run LOGS rather than from `results.json`.

WHY THIS EXISTS. All three `loopD` clients were killed at ~91% of the run (through the five
value-signal arms, partway into `lprog-only`). `directed_loop.py` writes `results.json` and
commits the Modal volume only at the very END of the function, so nothing was persisted — but
every per-round line was already printed, and the per-round BALLISTIC goal-distance plus the
drift schedule is all the headline metric (excess damage) needs. So loopD's numbers come from
here, not from a results.json. Provenance is recorded in the README; re-run loopD if you want a
clean artifact.

SPLICING. loopD's `lprog-only` arm is incomplete (31-39 of 72 rounds). loopC ran the same arm
to completion under an IDENTICAL configuration, and the two are directly comparable because
every stochastic input is keyed on (seed, round) and never on the policy: the drift schedule is
`seed+77`, the base FM is shared, the eval geometry is `seed+7`, and the per-round collection
RNG is `seed + 11000 + 97*t`. Each policy therefore runs an independent trajectory that does not
depend on which OTHER policies were in the run. `--verify-splice` checks this empirically on the
overlapping rounds (observed max |diff| = 5e-5, i.e. the 4-decimal print rounding).

Run:
    cd experiments/
    python3 mujoco_control/ballistic/directed/directed_loop_from_logs.py \
        --tags loopD_s0 loopD_s1 loopD_s2 --splice-from loopC --splice-policy lprog-only --verify-splice
"""

import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(__file__)


def parse_log(tag):
    """-> (drift schedule [(round, region_name)], {policy: {round: ballistic_dist}})"""
    path = os.path.join(HERE, "logs", f"directed_loop_{tag}.log")
    with open(path) as fh:
        txt = fh.read()
    m = re.search(r"drift schedule: (.+)", txt)
    if not m:
        raise ValueError(f"no drift schedule line in {path}")
    sched = []
    for part in m.group(1).split(";"):
        mm = re.search(r"r(\d+): (\S+) phi", part)
        if mm:
            sched.append((int(mm.group(1)), mm.group(2)))
    out = {}
    for line in txt.splitlines():
        mm = re.match(r"\[\s*([a-z-]+) r(\d+)\] .*? ball=([0-9.]+)", line)
        if mm:
            out.setdefault(mm.group(1), {})[int(mm.group(2))] = float(mm.group(3))
    return sched, out


def load_json(tag):
    path = os.path.join(HERE, "figures", f"directed_loop_{tag}", "results.json")
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="log tags, one per seed")
    ap.add_argument("--splice-from", default=None,
                    help="tag PREFIX of a completed run to source an incomplete policy from")
    ap.add_argument("--splice-policy", default=None)
    ap.add_argument("--verify-splice", action="store_true")
    ap.add_argument("--rounds", type=int, default=72)
    ap.add_argument("--drift-every", type=int, default=12)
    args = ap.parse_args()

    per_seed, arms = [], None
    for tag in args.tags:
        sched, D = parse_log(tag)
        seed_suffix = tag.split("_")[-1]
        if args.splice_from and args.splice_policy:
            src = f"{args.splice_from}_{seed_suffix}"
            C = {r["round"]: r["ballistic_cem"]
                 for r in load_json(src)["results"][args.splice_policy]}
            if args.verify_splice and args.splice_policy in D:
                common = sorted(set(D[args.splice_policy]) & set(C))
                diff = np.array([abs(D[args.splice_policy][r] - C[r]) for r in common])
                print(f"[splice-check {tag}<-{src}] {len(common)} overlapping rounds, "
                      f"max|diff|={diff.max():.2e} mean={diff.mean():.2e}"
                      + ("  OK" if diff.max() < 1e-3 else "  *** MISMATCH ***"))
            D[args.splice_policy] = C
        complete = {p: v for p, v in D.items() if len(v) == args.rounds}
        dropped = sorted(set(D) - set(complete))
        if dropped:
            print(f"[{tag}] dropping incomplete arms: "
                  + ", ".join(f"{p}({len(D[p])}/{args.rounds})" for p in dropped))
        per_seed.append((sched, complete))
        arms = [p for p in complete] if arms is None else [p for p in arms if p in complete]

    agg = {a: [] for a in arms}
    for sched, D in per_seed:
        events = [t for t, n in sched if n.startswith("A-")]      # ON-reach drifts only
        series = {a: np.array([D[a][r] for r in range(args.rounds)]) for a in arms}
        best = min(float(v.min()) for v in series.values())        # shared absolute floor
        for a in arms:
            for t0 in events:
                agg[a].append(float((series[a][t0:t0 + args.drift_every] - best).sum()))

    n = len(agg[arms[0]])
    ref = "lprog-only" if "lprog-only" in arms else arms[0]
    order = sorted(arms, key=lambda a: np.mean(agg[a]))
    print(f"\n=== excess damage after an ON-REACH drift (n={n} events, "
          f"{len(args.tags)} seeds; lower=better) ===")
    print(f"{'policy':>14s}  {'pooled':>20s}   {'vs ' + ref:>14s}   per-seed")
    for a in order:
        e = np.array(agg[a])
        ps = "  ".join(f"s{i}={np.mean(agg[a][i * 4:(i + 1) * 4]):.3f}" for i in range(len(args.tags)))
        print(f"{a:>14s}  {e.mean():>8.3f} ± {e.std() / np.sqrt(len(e)):<9.3f}  "
              f"{np.mean(agg[a]) / np.mean(agg[ref]):>13.2f}x   {ps}")


if __name__ == "__main__":
    main()
