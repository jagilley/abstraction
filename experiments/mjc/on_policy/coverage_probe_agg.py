"""Aggregate E0 (`coverage_probe.py`) across seeds, reading the per-run LOGS.

Reads logs rather than `results.json` for the reason [`../ballistic/directed/`](../ballistic/directed/README.md)
had to write `directed_loop_from_logs.py`: launching several detached clients from one shell is
fragile, and a client killed before the Modal function's final `volume.commit()` loses the
results.json entirely while the per-config log lines survive. Seed 2 of the first E0 run died that
way at 19/20 configurations.

Usage:
    python3 mjc/on_policy/coverage_probe_agg.py --tags e0_s0 e0_s1 e0_s2
"""

import argparse
import os
import re
import statistics as st

ROW = re.compile(
    r"^\[(B\w+) n=(\d+)\] task=([\d.]+) \(excess ([+-][\d.]+)\)\s+broad=([\d.]+) \(excess ([+-][\d.]+)\)"
    r"(?:\s+reactive=([\d.]+) ballistic=([\d.]+))?")
DIAG = re.compile(
    r"^\s+\[diag\] max\|corr\(u,s\)\|=([\d.nan]+) speed_mean=([\d.nan]+) p95=([\d.nan]+) "
    r"cover\(ref->pool\)=([\d.nan]+) reach\(pool->ref\)=([\d.nan]+) eps=(\d+) forced_resets=(\d+)")
REF = re.compile(r"^\[reference:(\S+)[^\]]*\] task=([\d.]+) broad=([\d.]+)"
                 r"(?: reactive=([\d.]+) ballistic=([\d.]+))?")


def parse(path):
    rows, refs, cur = {}, {}, None
    with open(path) as fh:
        for ln in fh:
            m = REF.match(ln)
            if m:
                refs[m.group(1)] = dict(task=float(m.group(2)), broad=float(m.group(3)),
                                        reactive=float(m.group(4)) if m.group(4) else None,
                                        ballistic=float(m.group(5)) if m.group(5) else None)
                continue
            m = ROW.match(ln)
            if m:
                cur = (m.group(1), int(m.group(2)))
                rows[cur] = dict(task=float(m.group(3)), broad=float(m.group(5)),
                                 reactive=float(m.group(7)) if m.group(7) else None,
                                 ballistic=float(m.group(8)) if m.group(8) else None)
                continue
            m = DIAG.match(ln)
            if m and cur:
                rows[cur].update(corr=float(m.group(1)), speed=float(m.group(2)),
                                 speed_p95=float(m.group(3)), cover=float(m.group(4)),
                                 reach=float(m.group(5)), eps=int(m.group(6)),
                                 forced=int(m.group(7)))
    return rows, refs


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], float("nan"), 1
    return st.mean(vals), st.stdev(vals) / len(vals) ** 0.5, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--logdir", default=os.path.join(os.path.dirname(__file__), "logs"))
    a = ap.parse_args()

    # Tags are grouped by their trailing `_s<N>` so a follow-up run that adds RUNGS to an existing
    # seed merges into it rather than being counted as another seed. Valid because every
    # experimenter instrument (reference pool, norm, both probes, eval geometry, FM init) is
    # derived from `seed` alone and built before the rung loop, so it is bit-identical across runs
    # of the same seed with different `--rungs`. `B0` is re-run in the follow-up as the check that
    # this actually held -- see the reproducibility block at the end.
    bysd, allrefs, dupes = {}, [], []
    for t in a.tags:
        p = os.path.join(a.logdir, f"coverage_probe_{t}.log")
        if not os.path.exists(p):
            print(f"[warn] missing {p}"); continue
        r, rf = parse(p)
        m = re.search(r"_s(\d+)$", t)
        sd = m.group(1) if m else t
        allrefs.append(rf)
        prev = bysd.setdefault(sd, {})
        for k, v in r.items():
            if k in prev:
                dupes.append((sd, k, prev[k]["task"], v["task"]))
            prev[k] = v
        print(f"[read] {t}: {len(r)} configurations -> seed {sd}")
    runs = sorted(bysd.items())

    keys = sorted({k for _, r in runs for k in r}, key=lambda k: (k[0], k[1]))
    print("\n=== E0: the price of the collection flag "
          f"({len(runs)} seeds; mean ± SEM) ===")
    print(f"{'rung':>5s} {'n':>6s} {'sd':>3s} {'task_err':>16s} {'broad_err':>16s} "
          f"{'|corr|':>8s} {'speed':>7s} {'cover':>7s} {'ballistic':>16s}")
    for nm in ("teleport", "on-policy(B2)"):
        vals = [rf[k] for rf in allrefs for k in rf if k.startswith(nm.split("(")[0])]
        if vals:
            mt, et, _ = agg([v["task"] for v in vals])
            mb, eb, _ = agg([v["broad"] for v in vals])
            mba, eba, _ = agg([v["ballistic"] for v in vals])
            print(f"{'ref':>5s} {nm:>6s} {'':>3s} {mt:>9.4f}±{et:<6.4f} {mb:>9.4f}±{eb:<6.4f} "
                  f"{'':>8s} {'':>7s} {'':>7s} {mba:>9.4f}±{eba:<6.4f}")
    for k in keys:
        got = [r[k] for _, r in runs if k in r]
        mt, et, ns = agg([g["task"] for g in got])
        mb, eb, _ = agg([g["broad"] for g in got])
        mc, _, _ = agg([g.get("corr") for g in got])
        ms, _, _ = agg([g.get("speed") for g in got])
        mv, _, _ = agg([g.get("cover") for g in got])
        mba, eba, _ = agg([g.get("ballistic") for g in got])
        print(f"{k[0]:>5s} {k[1]:>6d} {ns:>3d} {mt:>9.4f}±{et:<6.4f} {mb:>9.4f}±{eb:<6.4f} "
              f"{mc:>8.3f} {ms:>7.2f} {mv:>7.3f} {mba:>9.4f}±{eba:<6.4f}")

    # the headline: on-policy vs teleport on each probe, at the largest matched budget
    nmax = max(k[1] for k in keys)
    print(f"\n[P1] error ratio vs teleport at n={nmax} (>1 = worse than teleport):")
    for rg in ("B0m", "B0r", "B1", "B2", "B3"):
        got = [(r[(rg, nmax)], r[("B0", nmax)]) for _, r in runs
               if (rg, nmax) in r and ("B0", nmax) in r]
        if not got:
            continue
        mt, et, _ = agg([g["task"] / b["task"] for g, b in got])
        mb, eb, _ = agg([g["broad"] / b["broad"] for g, b in got])
        print(f"    {rg}: task {mt:.2f}x ± {et:.2f}    broad {mb:.2f}x ± {eb:.2f}")

    # the sample-efficiency crossing: smallest n at which each rung beats B0's BEST task error
    print("\n[P1b] sample-efficiency on the TASK probe: n needed to beat teleport's best")
    for _, r in runs[:1]:
        pass
    b0_best = agg([min(r[k]["task"] for k in r if k[0] == "B0") for _, r in runs])[0]
    print(f"    teleport's best task error over all n: {b0_best:.4f}")
    for rg in ("B0m", "B0r", "B1", "B2", "B3"):
        ns = []
        for _, r in runs:
            best = min((r[k]["task"] for k in r if k[0] == "B0"), default=None)
            hit = [k[1] for k in sorted(r, key=lambda x: x[1])
                   if k[0] == rg and best is not None and r[k]["task"] < best]
            ns.append(min(hit) if hit else None)
        got = [x for x in ns if x is not None]
        if any(k[0] == rg for k in keys):
            print(f"    {rg}: n = {got if got else 'never'}"
                  + (f"  (vs teleport's {max(k[1] for k in keys)})" if got else ""))

    # Reproducibility across merged runs: any (rung, n) seen in TWO log files for the same seed
    # must agree. `B0` is deliberately re-run in the follow-up so this block has something to
    # check; a mismatch means the two runs did not share their instruments and the merge is void.
    if dupes:
        worst = max(abs(x[2] - x[3]) for x in dupes)
        print(f"\n[merge check] {len(dupes)} configurations present in two logs for the same seed; "
              f"max task-error disagreement = {worst:.2e} "
              f"({'OK -- instruments shared' if worst < 1e-9 else 'MISMATCH -- do not merge'})")
        for sd, k, x, y in dupes[:4]:
            print(f"    seed {sd} {k}: {x:.6f} vs {y:.6f}")


if __name__ == "__main__":
    main()
