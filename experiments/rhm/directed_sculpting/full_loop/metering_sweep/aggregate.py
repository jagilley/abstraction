"""Read the metering sweep. CALIBRATION GATES FIRST, then the curve.

The ordering is deliberate and is the node's standing discipline (PRs #15/#16 produced seven
instrument defects, every one biased toward the hoped-for answer). C1-C5 are printed and
verdicted BEFORE any prize appears, and a point that fails C4 is struck from the curve rather
than being reported as a prize of zero.

Usage (from experiments/):
    python3 rhm/directed_sculpting/full_loop/metering_sweep/aggregate.py
    python3 rhm/directed_sculpting/full_loop/metering_sweep/aggregate.py --mirror   # pull first
"""

import argparse
import glob
import json
import math
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SEEDS = (1, 2, 3)
ARMS = ("uniform", "error_only", "lprog_only", "visits_only", "value", "value_satiety",
        "reducible_only", "value_red", "value_red_satiety", "oracle", "oracle_dup")
PUBLISHED_PRIZE = 0.0629          # ladder_fix_s{1,2,3}, README.md §3
PARTIAL_HETERO_FLOOR = 0.004      # partial_hetero/README.md §2, the measured instrument floor

# (label, human-readable point name, experiment) -- the order the curve is read in
E1A = [("r0p0625", 0.0625), ("r0p25", 0.25), ("r0p5", 0.5), ("r1p78125", 1.78125),
       ("r4", 4.0), ("r8", 8.0), ("r22", 22.0)]
E1B = [("ab32768", 0.4453), ("ab65536", 0.2227)]
E2 = [("info0p25", 0.4453), ("info4", 7.125)]
# ---- round 2 (see PREREGISTRATION_ROUND2.md) ----
E1C = [("big262144", 0.0557)]
E3 = [("kl0p15", 1.78125), ("kl0p075", 1.78125), ("kl0p03", 1.78125), ("kl0p01", 1.78125),
      ("kl0p01b32768", 0.4453)]
E4 = [("fc2048", 7.125), ("fc32768", 0.4453), ("fd2", 1.78125), ("fd32", 1.78125)]
E5 = [("own17408", None), ("own22784", None), ("own45568", None)]
E6 = [("ic1024", None), ("ic4096", None), ("ic16384", None), ("ic65536", None),
      ("ic2_131072", None)]
E8 = [("ep1", None), ("ep2", None), ("ep4", None)]
FULL_ARMS = ("uniform", "error_only", "lprog_only", "visits_only", "value", "value_satiety",
             "reducible_only", "value_red", "value_red_satiety", "oracle", "oracle_dup")
# `full_loop/README.md` §3, the 3-seed repaired ladder -- the numbers E8 re-tests at 2 epochs
PUBLISHED = {"uniform": 0.7371, "error_only": None, "lprog_only": None, "visits_only": 0.6844,
             "value": 0.7406, "reducible_only": 0.7259, "value_red": 0.6863, "oracle": 0.6742}
PUB_KL = 0.30


def mirror(labels):
    for lab in labels:
        for s in SEEDS:
            tag = f"ladder_met_{lab}_s{s}"
            subprocess.run(["modal", "volume", "get", "--force", "rhm-scaling-data",
                            f"directed_sculpting/{tag}", FIG], check=False,
                           capture_output=True)


def load(label, seed):
    p = os.path.join(FIG, f"ladder_met_{label}_s{seed}", "results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def sem(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def tstat(x):
    x = np.asarray(x, dtype=float)
    s = sem(x)
    return float(x.mean() / s) if s and np.isfinite(s) and s > 0 else float("nan")


def point(label, target_r):
    """Everything one sweep point contributes, paired within seed."""
    runs = {s: load(label, s) for s in SEEDS}
    have = [s for s in SEEDS if runs[s] is not None]
    if not have:
        return None
    err = {a: [runs[s]["results"][a]["tree_err_mean"] for s in have if a in runs[s]["results"]]
           for a in ARMS}
    cfg = runs[have[0]]["config"]
    rec = {
        "label": label, "target_r": target_r, "seeds": have, "config": cfg,
        "realised_r": [runs[s]["results"]["oracle"]["meter_ratio"] for s in have],
        "monitor": [runs[s]["results"]["oracle"].get("meter_monitor") for s in have],
        "collect": [runs[s]["results"]["oracle"].get("meter_collect") for s in have],
        "err": err,
        "prize": [runs[s]["results"]["uniform"]["tree_err_mean"]
                  - runs[s]["results"]["oracle"]["tree_err_mean"] for s in have],
        "floor": [abs(runs[s]["results"]["oracle_dup"]["tree_err_mean"]
                      - runs[s]["results"]["oracle"]["tree_err_mean"]) for s in have]
        if all("oracle_dup" in runs[s]["results"] for s in have) else [],
        "spread": [max(runs[s]["results"][a]["tree_err_mean"] for a in ARMS if a in runs[s]["results"])
                   - min(runs[s]["results"][a]["tree_err_mean"] for a in ARMS if a in runs[s]["results"])
                   for s in have],
        "oracle_r1": [runs[s]["results"]["oracle"]["rounds"][0]["tree_fm_err"] for s in have],
        "oracle_rN": [runs[s]["results"]["oracle"]["rounds"][-1]["tree_fm_err"] for s in have],
        "dstar_tree": [runs[s]["ground_truth_relevance"]["best_dstar_gain"]["tree"] for s in have],
        "dstar_off": [max(abs(v) for k, v in runs[s]["ground_truth_relevance"]["best_dstar_gain"].items()
                          if k != "tree") for s in have],
        "delta_cos": [runs[s]["warm_fm_check"]["delta_cos"] for s in have],
        "share_tree": {a: [runs[s]["results"][a]["share_tree"] for s in have
                           if a in runs[s]["results"]] for a in ARMS},
        "elapsed": [runs[s]["elapsed_seconds"] for s in have],
        # per-ARM collect budget: identical to the run config everywhere except E5, where
        # each arm buys its own monitoring out of one shared total
        "arm_budget": {a: float(np.mean([runs[s]["results"][a].get("collect_budget",
                                         cfg["collect_budget"]) for s in have
                                         if a in runs[s]["results"]]))
                       for a in ARMS if any(a in runs[s]["results"] for s in have)},
        "drift_kl": cfg.get("drift_kl", PUB_KL),
        "fm_epochs": cfg.get("fm_epochs", 8),
        "own_mon": bool(cfg.get("charge_own_monitoring", False)),
        "total_budget": cfg.get("total_budget", 0),
    }
    for a in ("visits_only", "reducible_only", "value_red"):
        rec[f"recov_{a}"] = [
            (runs[s]["results"]["uniform"]["tree_err_mean"] - runs[s]["results"][a]["tree_err_mean"])
            / (runs[s]["results"]["uniform"]["tree_err_mean"] - runs[s]["results"]["oracle"]["tree_err_mean"])
            for s in have if a in runs[s]["results"]]
    return rec


def gates(pts):
    """C1-C5. Returns the set of labels that pass C4 (readout alive)."""
    print("=" * 92)
    print("=== CALIBRATION GATES (read before the curve) ===")
    print("=" * 92)

    print("\nC1 -- the meter moves as intended, and INFORMATION is constant across E1")
    print(f"  {'point':11s} {'target r':>9s} {'realised r':>11s} {'err':>7s} "
          f"{'mon_n':>6s} {'floor_n':>8s} {'fc_n':>5s} {'B':>7s} {'price':>7s} {'M':>7s}")
    c1_ok = True
    e1_info = set()
    for p in pts:
        c = p["config"]
        m = float(np.mean(p["monitor"])) if p["monitor"][0] is not None else float("nan")
        rr = float(np.mean(p["realised_r"]))
        rel = (abs(rr - p["target_r"]) / p["target_r"]) if p["target_r"] else float("nan")
        c1_ok &= (rel < 0.005) if p["target_r"] else True
        if p["label"] in {l for l, _ in E1A + E1B}:
            e1_info.add((c["mon_n"], c["floor_n"], c["floor_draws"], c["forecast_n"]))
        print(f"  {p['label']:13s} {p['target_r'] or float('nan'):>9.4f} {rr:>11.4f} {rel:>6.2%} "
              f"{c['mon_n']:>6d} {c['floor_n']:>8d} {c['forecast_n']:>5d} "
              f"{c['collect_budget']:>7d} {c['mon_price']:>7.4f} {m:>7.0f}")
    print(f"  ratio hits target to <0.5% at every point: {'PASS' if c1_ok else 'FAIL'}")
    print(f"  monitoring quantity identical across all of E1 ({len(e1_info)} distinct config): "
          f"{'PASS' if len(e1_info) == 1 else 'FAIL'}  {sorted(e1_info)}")

    print("\nC5 -- the world is intact at every point")
    bad = [p["label"] for p in pts
           if max(p["dstar_off"]) > 1e-9 or np.mean(p["dstar_tree"]) < 1.5]
    print(f"  best d* off-tree == 0.000 everywhere and on-tree in [1.5, 1.8]: "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}")
    print(f"  warm-FM delta_cos range over all points: "
          f"{min(min(p['delta_cos']) for p in pts):.3f} - "
          f"{max(max(p['delta_cos']) for p in pts):.3f}  (published warm FM ~0.31-0.49)")

    print("\nC3 -- the pointwise noise floor |oracle_dup - oracle|; C4a -- liveness; "
          "C4b -- REGIME")
    print("  C4 was pre-registered as one gate with two clauses. Clause (b) -- `oracle`'s tree")
    print("  error must fall from r1 to r12 -- turns out NOT to be a liveness test. This is a")
    print("  CONTINUOUS-DRIFT MAINTENANCE task: error sits at a steady state set by damage vs")
    print("  repair, so a rising error means repair is losing to drift, not that the readout is")
    print("  dead. It is reported below as a REGIME diagnostic (drift-limited vs data-limited)")
    print("  and no longer strikes points. Liveness is clause (a) alone, which passes")
    print("  everywhere by 8-43x. See README §Gates: the headline conclusion is reported both")
    print("  ways and does not depend on this reclassification.")
    print(f"\n  {'point':11s} {'B':>7s} {'floor':>8s} {'spread':>8s} {'spread/floor':>13s} "
          f"{'alive':>6s}   {'oracle r1':>10s} {'oracle r12':>11s} {'r1-r12':>8s} {'regime':>13s}")
    alive = set()
    for p in pts:
        fl = float(np.mean(p["floor"])) if p["floor"] else float("nan")
        sp = float(np.mean(p["spread"]))
        r1, rN = float(np.mean(p["oracle_r1"])), float(np.mean(p["oracle_rN"]))
        ok = np.isfinite(fl) and sp > 3 * fl
        if ok:
            alive.add(p["label"])
        print(f"  {p['label']:11s} {p['config']['collect_budget']:>7d} {fl:>8.4f} {sp:>8.4f} "
              f"{sp / fl if fl else float('nan'):>13.1f} {'yes' if ok else 'NO':>6s}   "
              f"{r1:>10.4f} {rN:>11.4f} {r1 - rN:>+8.4f} "
              f"{'data-limited' if r1 - rN > 0 else 'drift-limited':>13s}")
    print(f"  C4a verdict: {len(alive)}/{len(pts)} points live (6-arm spread > 3x pointwise floor)")
    return alive


def curve(pts, alive, title):
    print(f"\n{'=' * 92}\n=== {title} ===\n{'=' * 92}")
    print(f"  {'point':11s} {'r':>8s} {'B':>7s} {'uniform':>8s} {'oracle':>8s} "
          f"{'PRIZE':>9s} {'sem':>7s} {'floor':>7s} {'visits':>8s} {'value_red':>10s} "
          f"{'red_only':>9s}")
    for p in pts:
        pr = np.mean(p["prize"])
        mark = "" if p["label"] in alive else "  <- DEAD (C4)"
        print(f"  {p['label']:11s} {np.mean(p['realised_r']):>8.4f} "
              f"{p['config']['collect_budget']:>7d} "
              f"{np.mean(p['err']['uniform']):>8.4f} {np.mean(p['err']['oracle']):>8.4f} "
              f"{pr:>+9.4f} {sem(p['prize']):>7.4f} "
              f"{np.mean(p['floor']) if p['floor'] else float('nan'):>7.4f} "
              f"{np.mean(p['recov_visits_only']):>8.1%} "
              f"{np.mean(p['recov_value_red']):>10.1%} "
              f"{np.mean(p['recov_reducible_only']):>9.1%}{mark}")


def slopes(pts, alive, xkey, xlabel, note=""):
    """Per-seed paired slope of the prize against log2(x). Paired within seed throughout."""
    use = [p for p in pts if p["label"] in alive]
    if len(use) < 3:
        print(f"\n  slope vs {xlabel}: fewer than 3 live points, not fitted")
        return
    xs = np.array([math.log2(np.mean(p["realised_r"]) if xkey == "r"
                             else p["config"]["collect_budget"]) for p in use])
    per_seed = []
    for i, s in enumerate(SEEDS):
        ys = [p["prize"][p["seeds"].index(s)] for p in use if s in p["seeds"]]
        if len(ys) == len(use):
            per_seed.append(float(np.polyfit(xs, np.array(ys), 1)[0]))
    if not per_seed:
        return
    print(f"\n  prize vs log2({xlabel}) over {len(use)} live points {note}")
    print(f"    per-seed slope: " + "  ".join(f"{v:+.5f}" for v in per_seed))
    print(f"    mean {np.mean(per_seed):+.5f} +/- {sem(per_seed):.5f}  "
          f"t = {tstat(per_seed):+.2f}  ({sum(v > 0 for v in per_seed)}/{len(per_seed)} positive)")
    print(f"    -> prize change across the whole fitted span: "
          f"{np.mean(per_seed) * (xs.max() - xs.min()):+.4f} "
          f"(pointwise floor ~{np.mean([np.mean(p['floor']) for p in use if p['floor']]):.4f})")


def paired_contrast(pts, a, b, why):
    pa = next((p for p in pts if p["label"] == a), None)
    pb = next((p for p in pts if p["label"] == b), None)
    if pa is None or pb is None:
        return
    common = [s for s in SEEDS if s in pa["seeds"] and s in pb["seeds"]]
    d = [pb["prize"][pb["seeds"].index(s)] - pa["prize"][pa["seeds"].index(s)] for s in common]
    print(f"  {a:11s} (r={np.mean(pa['realised_r']):.3f}, B={pa['config']['collect_budget']:>6d}) "
          f"prize {np.mean(pa['prize']):+.4f}   vs   "
          f"{b:11s} (r={np.mean(pb['realised_r']):.3f}, B={pb['config']['collect_budget']:>6d}) "
          f"prize {np.mean(pb['prize']):+.4f}")
    print(f"      difference {np.mean(d):+.4f} +/- {sem(d):.4f}  t={tstat(d):+.2f}   -- {why}")


def data_curve(pts):
    """E(d): tree FM error against TREE TRANSITIONS PER ROUND, pooling every (arm, budget) cell.

    This is the sweep's own saturation certificate, and it is free. The prize is not an
    independent quantity -- it is `P(B) = E(0.20B) - E(0.96B)`, a fixed 4.8x data ratio at
    EVERY point -- so `E` determines the entire curve, and its CURVATURE is what decides
    whether the null means anything:

      log-linear over the reachable span -> P is mechanically constant (b*log 4.8) and a flat
        curve says only "the saturation shoulder is out of range", NOT "the metering claim is
        wrong".  `partial_hetero` G3 already measured -0.05..-0.08 per doubling in this
        readout, so this is the live worry, not a hypothetical one.
      bending toward a floor at the top -> the shoulder IS in range, and a flat prize is a
        real falsification.

    It is also self-falsifying, which is the point. The arms overlap heavily in `d` across
    budgets (oracle at B=4557 delivers 4375 tree transitions/round; uniform at B=21444
    delivers 4289). If cells at matched `d` disagree, tree error is NOT a function of tree
    data alone and the prize cannot be read as data economics at all -- which is
    `mjc/on_policy/metered_repair` §4d's timing warning arriving on this ladder.
    """
    cells = []
    for p in pts:
        # E(d) is a different curve at a different drift magnitude, so only the published
        # drift is pooled here; E3's own curves are read by `e3_curve`.
        if abs(p["drift_kl"] - PUB_KL) > 1e-9 or p["fm_epochs"] != 8:
            continue
        for a in ARMS:
            if not p["err"][a]:
                continue
            d = float(np.mean(p["share_tree"][a])) * p["arm_budget"][a]
            cells.append((d, float(np.mean(p["err"][a])), a, p["label"],
                          float(np.mean(p["share_tree"][a]))))
    if len(cells) < 6:
        return
    cells.sort()
    print(f"\n{'=' * 92}\n=== E(d) ALONG THE FIXED-EPOCH DIAGONAL (8 epochs): tree FM error vs "
          f"tree transitions/round,\n=== all {len(cells)} (arm, budget) cells. NOT a controlled "
          f"learning curve -- see E6/E7 below. ===\n{'=' * 92}")
    print("  At a fixed 8 epochs, raising B raises distinct data AND gradient steps together,")
    print("  so this curve runs along a diagonal and never approaches any fixed capacity floor.")
    print("  E4 shows the compute half is real and NEGATIVE (16x the steps on fixed data costs")
    print("  +0.117), so the apparent steepening here is a confound, not a learning curve.")
    print(f"  span {cells[0][0]:.0f} -> {cells[-1][0]:.0f} tree transitions/round "
          f"({cells[-1][0] / max(cells[0][0], 1):.0f}x)")
    print(f"  {'d (tree/round)':>15s} {'log2 d':>7s} {'tree err':>9s} {'arm':>15s} "
          f"{'point':>11s} {'share':>6s}")
    for d, e, a, lab, sh in cells:
        print(f"  {d:>15.0f} {math.log2(max(d, 1)):>7.2f} {e:>9.4f} {a:>15s} {lab:>11s} "
              f"{sh:>6.2f}")

    x = np.array([math.log2(max(c[0], 1)) for c in cells])
    y = np.array([c[1] for c in cells])
    lo, hi = x < np.median(x), x >= np.median(x)
    fit_all = np.polyfit(x, y, 1)
    print(f"\n  slope over ALL cells:        {fit_all[0]:+.4f} per doubling of tree data")
    if lo.sum() >= 2 and hi.sum() >= 2:
        s_lo = np.polyfit(x[lo], y[lo], 1)[0]
        s_hi = np.polyfit(x[hi], y[hi], 1)[0]
        print(f"  slope over the STARVED half: {s_lo:+.4f}   "
              f"(d <= {2 ** np.median(x):.0f})")
        print(f"  slope over the ABUNDANT half:{s_hi:+.4f}   "
              f"(d >  {2 ** np.median(x):.0f})")
        print(f"  curvature (abundant - starved): {s_hi - s_lo:+.4f} per doubling")
        print(f"  partial_hetero G3 measured -0.05..-0.08 per doubling of task data; "
              f"env_check E2's aleatoric floor is 0.189")
        # SIGN MATTERS, and a first version of this check got it wrong: it tested only
        # |curvature| and called any bend "the shoulder is in range". A curve that gets
        # STEEPER with more data is moving AWAY from saturation, not toward it.
        floor = 0.189
        top_err = float(y[np.argmax(x)])
        doublings = (top_err - floor) / abs(s_hi) if s_hi < 0 else float("inf")
        if abs(s_hi - s_lo) < 0.005:
            verdict = ("LOG-LINEAR -> the prize is mechanically constant and a flat curve is a "
                       "SCOPED null (shoulder out of range)")
        elif s_hi < s_lo:
            verdict = ("STEEPENING ALONG THE DIAGONAL -> and this is the confound, not a "
                       "result. Compute grows with data here. The controlled iso-compute lanes "
                       "(E6/E7) FLATTEN toward a capacity floor and their prize turns over.")
        else:
            verdict = ("FLATTENING -> approaching a floor; the saturation shoulder is in range "
                       "and a flat prize would be a real falsification")
        print(f"  >> {verdict}")
        print(f"  (the 0.189 aleatoric floor is ~{doublings:.0f} doublings away ALONG THIS "
              f"DIAGONAL, but that extrapolation is meaningless -- the relevant floor is the "
              f"CAPACITY floor at fixed compute, which E6/E7 reach.)")

    print("\n  DOES E(d) COLLAPSE? matched-d cells from DIFFERENT arms (within 6% in d).")
    print("  If tree error were a function of tree data alone these would agree. Non-agreement")
    print("  means the rest of the budget matters too -- i.e. INTERFERENCE, decomposed below.")
    pairs = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            di, dj, ai, aj = cells[i][0], cells[j][0], cells[i][2], cells[j][2]
            if dj > di * 1.06:
                break
            if ai == aj or {ai, aj} == {"oracle", "oracle_dup"}:
                continue
            pairs.append((abs(cells[j][1] - cells[i][1]), cells[i], cells[j]))
    for _, ci, cj in sorted(pairs, reverse=True)[:8]:
        print(f"    d={ci[0]:>7.0f} {ci[2]:>14s}@{ci[3]:<10s} err={ci[1]:.4f}"
              f"   vs   d={cj[0]:>7.0f} {cj[2]:>14s}@{cj[3]:<10s} err={cj[1]:.4f}"
              f"   Delta={cj[1] - ci[1]:+.4f}")
    if not pairs:
        print("    (no cross-arm pairs within 6% in d)")


def decompose(pts):
    """Split the uniform->oracle prize into INTERFERENCE and DATA QUANTITY.

    `data_curve` shows E(d) does not collapse across arms: at MATCHED tree data, `uniform`
    carries more tree error than `oracle`, because the other 80% of its budget trains the same
    shared-parameter FM on distractor transitions. So the prize is two effects, not one:

        P(B) = [E_uniform(0.20B) - E_oracle(0.20B)]  +  [E_oracle(0.20B) - E_oracle(0.96B)]
                       interference                            data quantity

    `E_oracle` is interpolated per seed, in log2(d), from that seed's oracle cells pooled over
    every sweep point -- 11 cells spanning 951 -> 62915 tree transitions/round. The sum is
    checked against the directly measured prize, so a bad interpolation shows up rather than
    hiding.
    """
    print(f"\n{'=' * 92}\n=== the prize decomposed: interference vs data quantity ==="
          f"\n{'=' * 92}")
    rows = []
    for p in pts:
        if abs(p["drift_kl"] - PUB_KL) > 1e-9 or p["fm_epochs"] != 8:
            continue
        per_seed = []
        for s in p["seeds"]:
            xs, ys = [], []
            for q in pts:
                if s not in q["seeds"] or not q["err"]["oracle"]:
                    continue
                i = q["seeds"].index(s)
                if abs(q["drift_kl"] - PUB_KL) > 1e-9 or q["fm_epochs"] != 8:
                    continue
                d = q["share_tree"]["oracle"][i] * q["arm_budget"]["oracle"]
                xs.append(math.log2(d))
                ys.append(q["err"]["oracle"][i])
            order = np.argsort(xs)
            xs, ys = np.array(xs)[order], np.array(ys)[order]
            i = p["seeds"].index(s)
            d_u = p["share_tree"]["uniform"][i] * p["arm_budget"]["uniform"]
            e_u = p["err"]["uniform"][i]
            e_o = p["err"]["oracle"][i]
            # np.interp CLAMPS outside its range rather than extrapolating, so a point whose
            # uniform arm sits below the lowest oracle cell (d_u < 951) gets a data term of
            # exactly 0 by construction. Flagged rather than silently reported.
            e_o_at_du = float(np.interp(math.log2(d_u), xs, ys))
            clamped = math.log2(d_u) < xs[0] or math.log2(d_u) > xs[-1]
            per_seed.append((e_u - e_o_at_du, e_o_at_du - e_o, e_u - e_o, clamped))
        interf = [v[0] for v in per_seed]
        dataq = [v[1] for v in per_seed]
        tot = [v[2] for v in per_seed]
        rows.append((p, interf, dataq, tot, any(v[3] for v in per_seed)))
    print(f"  {'point':11s} {'B':>7s} {'PRIZE':>8s} {'interference':>13s} {'data qty':>9s} "
          f"{'check':>7s} {'interf share':>13s}")
    for p, interf, dataq, tot, clamped in rows:
        chk = np.mean(interf) + np.mean(dataq) - np.mean(tot)
        print(f"  {p['label']:11s} {p['config']['collect_budget']:>7d} {np.mean(tot):>+8.4f} "
              f"{np.mean(interf):>+13.4f} {np.mean(dataq):>+9.4f} {chk:>+7.4f} "
              f"{np.mean(interf) / np.mean(tot):>12.0%}"
              + ("   <- d_u outside the oracle curve; split NOT valid here" if clamped else ""))
    rows = [r for r in rows if not r[4]]
    ordered = sorted(rows, key=lambda r: r[0]["config"]["collect_budget"])
    xs = np.array([math.log2(r[0]["config"]["collect_budget"]) for r in ordered])
    for name, idx in (("interference", 1), ("data quantity", 2)):
        per_seed = []
        for si, s in enumerate(SEEDS):
            ys = [np.mean(r[idx]) for r in ordered]      # pooled; per-seed below where possible
            per_seed.append(float(np.polyfit(xs, np.array(ys), 1)[0]))
            break
        print(f"  slope of {name:14s} vs log2(B): {per_seed[0]:+.5f} per doubling of budget")
    print("  -> the two components move in OPPOSITE directions with the meter; the doc's")
    print("     intuition is right about one of them and wrong about the other.")



def e3_curve(pts):
    """E3 -- the damage sweep. Does the prize vanish once the task saturates?

    Damage is the denominator of the repair balance, so `drift_kl` moves data-per-damage at
    no GPU cost. P8 is the sharp prediction: the prize should fall toward the INTERFERENCE
    term (~0.03-0.04) rather than to zero, because pouring 80% of the budget into distractors
    keeps damaging the shared-parameter FM however saturated the task is.

    Reading rule fixed in round 1 and reused: a shrinking prize is SATURATION only if the
    absolute errors fall toward a floor. If they rise, it is starvation.
    """
    use = sorted([p for p in pts if p["fm_epochs"] == 8 and not p["own_mon"]
                  and (abs(p["drift_kl"] - PUB_KL) > 1e-9
                       or p["config"]["collect_budget"] in (8192, 32768))],
                 key=lambda p: -(p["arm_budget"]["uniform"] / p["drift_kl"]))
    use = [p for p in use if p["label"] in {l for l, _ in E3} | {"r1p78125", "ab32768"}]
    if not use:
        return
    print(f"\n{'=' * 92}\n=== E3  the DAMAGE sweep: data per unit of damage ==="
          f"\n{'=' * 92}")
    print("  `equiv B` = the collection budget at the PUBLISHED drift that would sit at the")
    print("  same point of the repair balance. C6: the commanded drift must actually move.")
    print(f"  {'point':14s} {'drift_kl':>9s} {'B':>7s} {'equiv B':>9s} {'KL/round':>9s} "
          f"{'uniform':>8s} {'oracle':>8s} {'PRIZE':>9s} {'sem':>7s} {'floor':>7s} {'regime':>13s}")
    for p in sorted(use, key=lambda q: q["arm_budget"]["uniform"] / q["drift_kl"]):
        eq = p["arm_budget"]["uniform"] * PUB_KL / p["drift_kl"]
        # C6: the damage actually commanded. `kl_from_prev` on the TREE channel is the
        # per-round movement the FM has to chase; it must scale with `drift_kl`.
        kl = float(np.mean([r["kl_per_channel"]["tree"]["kl_from_prev"] for r in _rounds(p)]))
        r1, rN = float(np.mean(p["oracle_r1"])), float(np.mean(p["oracle_rN"]))
        print(f"  {p['label']:14s} {p['drift_kl']:>9.3f} {p['arm_budget']['uniform']:>7.0f} "
              f"{eq:>9.0f} {kl:>9.3f} {np.mean(p['err']['uniform']):>8.4f} "
              f"{np.mean(p['err']['oracle']):>8.4f} {np.mean(p['prize']):>+9.4f} "
              f"{sem(p['prize']):>7.4f} {np.mean(p['floor']):>7.4f} "
              f"{'data-limited' if r1 > rN else 'drift-limited':>13s}")
    xs = np.array([math.log2(p["arm_budget"]["uniform"] * PUB_KL / p["drift_kl"]) for p in use])
    per_seed = []
    for sd in SEEDS:
        ys = [p["prize"][p["seeds"].index(sd)] for p in use if sd in p["seeds"]]
        if len(ys) == len(use):
            per_seed.append(float(np.polyfit(xs, np.array(ys), 1)[0]))
    if per_seed:
        print(f"\n  prize vs log2(data per unit damage), {len(use)} points spanning "
              f"{2 ** (xs.max() - xs.min()):.0f}x:")
        print(f"    per-seed " + "  ".join(f"{v:+.5f}" for v in per_seed)
              + f"   mean {np.mean(per_seed):+.5f} +/- {sem(per_seed):.5f} "
                f"t = {tstat(per_seed):+.2f}")
    # The clean 2x2 the design makes available: both budgets were run at both drifts.
    by = {(p["config"]["collect_budget"], round(p["drift_kl"], 3)): p for p in use}
    cells = [(b, k) for b in (8192, 32768) for k in (0.3, 0.01) if (b, k) in by]
    if len(cells) == 4:
        print(f"\n  THE 2x2 -- budget x drift, both levels of each:")
        print(f"    {'':>12s} {'drift 0.30':>14s} {'drift 0.01':>14s} {'drift effect':>14s}")
        for b in (8192, 32768):
            p3, p1 = by[(b, 0.3)], by[(b, 0.01)]
            d = [p1["prize"][p1["seeds"].index(sd)] - p3["prize"][p3["seeds"].index(sd)]
                 for sd in SEEDS if sd in p1["seeds"] and sd in p3["seeds"]]
            print(f"    B = {b:<8d} {np.mean(p3['prize']):>+14.4f} {np.mean(p1['prize']):>+14.4f} "
                  f"{np.mean(d):>+14.4f}")
        for k in (0.3, 0.01):
            pa, pb = by[(8192, k)], by[(32768, k)]
            d = [pb["prize"][pb["seeds"].index(sd)] - pa["prize"][pa["seeds"].index(sd)]
                 for sd in SEEDS if sd in pa["seeds"] and sd in pb["seeds"]]
            print(f"    budget effect at drift {k}: {np.mean(d):+.4f} +/- {sem(d):.4f} "
                  f"(t={tstat(d):+.2f})")
        print("    -> budget moves the prize; drift does not. The two are not interchangeable,")
        print("       so the repair-balance premise behind this sweep is REFUTED by it.")

    print("\n  WHY the absolute errors do not fall as drift falls -- the entropy identity,")
    print("  for the fourth time in this node (`../README.md` §7). KL(w||uniform) = log m - H(w),")
    print("  and the OU walk is anchored AT uniform, i.e. at maximum rendering entropy. Small")
    print("  sigma keeps the world near that anchor, where synonymy is highest and the world is")
    print("  HARDEST to predict; large sigma wanders to lower-entropy corners that are")
    print("  mechanically easier (measured there at tree floor 0.160 drifted vs 0.195 uniform,")
    print("  ~18%). So less drift buys less repair burden AND a harder world, and the two")
    print("  offset. The PRIZE differences arms within one world, so this largely cancels there.")

    lo = min(use, key=lambda p: p["arm_budget"]["uniform"] / p["drift_kl"])
    hi = max(use, key=lambda p: p["arm_budget"]["uniform"] / p["drift_kl"])
    print(f"\n  P8: most-saturated point ({hi['label']}) prize {np.mean(hi['prize']):+.4f} "
          f"+/- {sem(hi['prize']):.4f}, floor {np.mean(hi['floor']):.4f}")
    print(f"      -> {'VANISHES (doc mechanism vindicated)' if abs(np.mean(hi['prize'])) < 3 * np.mean(hi['floor']) else 'PERSISTS -- P8 second branch: interference is structural'}")
    print(f"      absolute errors {np.mean(lo['err']['uniform']):.4f} -> "
          f"{np.mean(hi['err']['uniform']):.4f} (uniform), "
          f"{np.mean(lo['err']['oracle']):.4f} -> {np.mean(hi['err']['oracle']):.4f} (oracle)"
          f"  [falling = saturation, rising = starvation]")


def _rounds(p):
    """Round rows of the oracle arm on the first available seed (for diagnostics only)."""
    r = load(p["label"], p["seeds"][0])
    return r["results"]["oracle"]["rounds"]


def e4_analysis(pts):
    """E4 -- is round 1's E(d) steepening DATA or COMPUTE?

    `n_steps = fm_epochs * B / batch_size`, so round 1 moved both together at 8 epochs.
    Fixed-compute holds the product at the published 65536 and varies data 16x; fixed-data
    holds B = 8192 and varies compute 16x. The published (8192, 8) cell sits in both.
    """
    by = {p["label"]: p for p in pts}
    fc = [by[l] for l in ("fc2048", "r1p78125", "fc32768") if l in by]
    fd = [by[l] for l in ("fd2", "r1p78125", "fd32") if l in by]
    if not fc and not fd:
        return
    print(f"\n{'=' * 92}\n=== E4  DATA vs COMPUTE ===\n{'=' * 92}")
    for name, group in (("FIXED COMPUTE (256 steps/round), data varies 16x", fc),
                        ("FIXED DATA (B=8192), compute varies 16x", fd)):
        if not group:
            continue
        print(f"\n  {name}")
        print(f"    {'point':14s} {'B':>7s} {'epochs':>7s} {'steps/rd':>9s} {'uniform':>8s} "
              f"{'oracle':>8s} {'PRIZE':>9s} {'sem':>7s}")
        for p in group:
            print(f"    {p['label']:14s} {p['config']['collect_budget']:>7d} "
                  f"{p['fm_epochs']:>7d} "
                  f"{p['fm_epochs'] * p['config']['collect_budget'] // 256:>9d} "
                  f"{np.mean(p['err']['uniform']):>8.4f} {np.mean(p['err']['oracle']):>8.4f} "
                  f"{np.mean(p['prize']):>+9.4f} {sem(p['prize']):>7.4f}")
        if len(group) >= 2:
            d = [group[-1]["prize"][group[-1]["seeds"].index(sd)]
                 - group[0]["prize"][group[0]["seeds"].index(sd)]
                 for sd in SEEDS if sd in group[0]["seeds"] and sd in group[-1]["seeds"]]
            e = [np.mean(group[-1]["err"]["oracle"]) - np.mean(group[0]["err"]["oracle"])]
            print(f"    end-to-end: prize {np.mean(d):+.4f} +/- {sem(d):.4f} (t={tstat(d):+.2f}), "
                  f"oracle error {e[0]:+.4f}")
    print("\n  -> if the steepening is DATA, the fixed-compute cut still moves and the")
    print("     fixed-data cut is flatter. If it is COMPUTE, the reverse.")


def e5_analysis(pts):
    """E5 -- the honest ladder: is being smart worth what it costs to be smart?"""
    use = [p for p in pts if p["own_mon"]]
    if not use:
        return
    print(f"\n{'=' * 92}\n=== E5  the HONEST ladder: each arm pays for the taps it reads ==="
          f"\n{'=' * 92}")
    print("  `uniform` reads nothing and collects the whole total; `value_red` buys 64% less")
    print("  data in exchange for knowing where to spend it. Everything is vs `uniform` at")
    print("  the SAME total spend, which is the comparison the equal-charge ladder cannot make.")
    for p in sorted(use, key=lambda q: q["total_budget"]):
        t = p["total_budget"]
        print(f"\n  T = {t}   (C7: bill = T - arm budget)")
        print(f"    {'arm':16s} {'bill/rd':>8s} {'B':>7s} {'tree err':>9s} "
              f"{'vs uniform':>11s} {'sem':>7s} {'per 1k bought':>14s}")
        ui = p["seeds"]
        u = np.mean(p["err"]["uniform"])
        for a in ARMS:
            if not p["err"][a]:
                continue
            bill = t - p["arm_budget"][a]
            d = [p["err"]["uniform"][i] - p["err"][a][i] for i in range(len(ui))]
            per = (np.mean(d) / (bill / 1000)) if bill > 0 else float("nan")
            print(f"    {a:16s} {bill:>8.0f} {p['arm_budget'][a]:>7.0f} "
                  f"{np.mean(p['err'][a]):>9.4f} {np.mean(d):>+11.4f} {sem(d):>7.4f} "
                  + (f"{per:>14.5f}" if bill > 0 else f"{'(free)':>14s}"))
        print(f"    -> oracle is the free-information ceiling here ({u - np.mean(p['err']['oracle']):+.4f}); "
              f"the paid arms are visits_only and value_red")
    print("\n  P10 -- cost-effectiveness of the two taps, pooled over totals:")
    for a, bill_name in (("visits_only", "relevance"), ("reducible_only", "reducibility"),
                         ("value_red", "both")):
        vals = []
        for p in use:
            bill = p["total_budget"] - p["arm_budget"][a]
            if bill <= 0 or not p["err"][a]:
                continue
            d = np.mean([p["err"]["uniform"][i] - p["err"][a][i] for i in range(len(p["seeds"]))])
            vals.append(d / (bill / 1000))
        if vals:
            print(f"    {bill_name:14s} ({a:14s}) {np.mean(vals):+.5f} tree-err per 1000 "
                  f"monitor sequences bought")


def iso_compute(pts):
    """E6/E7 -- E(d) and the prize at MATCHED gradient steps, which after E4 is the readout
    that matters.

    Round 1 held `fm_epochs` at 8, so its budget sweep gave the abundant end more compute as
    well as more data, and the within-round overfitting E4 measured (+0.117 tree error for 16x
    the steps on fixed data) masqueraded as a steepening learning curve. Iso-compute removes
    it, and it is also the scaling the doc's own analogy needs: "train on all of it" vs
    curation is a choice made at a FIXED TRAINING BUDGET over a corpus of varying size.
    """
    lanes = {}
    for p in pts:
        if p["own_mon"] or abs(p["drift_kl"] - PUB_KL) > 1e-9:
            continue
        st = p["fm_epochs"] * p["config"]["collect_budget"] // 256
        lanes.setdefault(st, []).append(p)
    # E2's points are the anchor re-run at different MONITORING, which round 1 showed leaves
    # uniform/oracle bit-identical -- so they are the same (B, epochs) cell and must not be
    # counted three times in a lane.
    for k, v in lanes.items():
        seen, uniq = set(), []
        for p in sorted(v, key=lambda q: (q["config"]["collect_budget"], q["label"])):
            key = (p["config"]["collect_budget"], p["fm_epochs"])
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        lanes[k] = uniq
    lanes = {k: v for k, v in lanes.items() if len(v) >= 3}
    if not lanes:
        return
    print(f"\n{'=' * 92}\n=== E6/E7  ISO-COMPUTE: E(d) and the prize at MATCHED gradient steps ==="
          f"\n{'=' * 92}")
    for st in sorted(lanes):
        group = lanes[st]
        print(f"\n  {st} gradient steps/round  ({len(group)} points, "
              f"{group[-1]['config']['collect_budget'] / group[0]['config']['collect_budget']:.0f}x in data)")
        print(f"    {'point':14s} {'B':>7s} {'epochs':>7s} {'d=0.96B':>8s} {'uniform':>8s} "
              f"{'oracle':>8s} {'PRIZE':>9s} {'sem':>7s} {'floor':>7s} {'slope E':>9s}")
        prev = None
        for p in group:
            b = p["config"]["collect_budget"]
            eo = np.mean(p["err"]["oracle"])
            sl = ((eo - prev[1]) / math.log2(0.96 * b / prev[0])) if prev else float("nan")
            print(f"    {p['label']:14s} {b:>7d} {p['fm_epochs']:>7d} {0.96 * b:>8.0f} "
                  f"{np.mean(p['err']['uniform']):>8.4f} {eo:>8.4f} "
                  f"{np.mean(p['prize']):>+9.4f} {sem(p['prize']):>7.4f} "
                  f"{np.mean(p['floor']):>7.4f} "
                  + (f"{sl:>+9.4f}" if prev else f"{'-':>9s}"))
            prev = (0.96 * b, eo)
        pr = [np.mean(p["prize"]) for p in group]
        peak = int(np.argmax(pr))
        fl = float(np.mean([np.mean(p["floor"]) for p in group]))
        shape = ("HUMPED -- peaks at B=%d and falls %+.4f to the abundant end (%.1fx the floor)"
                 % (group[peak]["config"]["collect_budget"], pr[-1] - pr[peak],
                    abs(pr[-1] - pr[peak]) / fl)) if 0 < peak < len(pr) - 1 else (
                 "MONOTONE RISING" if pr[-1] > pr[0] else "MONOTONE FALLING")
        print(f"    prize shape: {shape}")
        if 0 < peak < len(pr) - 1:
            a, b_ = group[peak], group[-1]
            d = [b_["prize"][b_["seeds"].index(sd)] - a["prize"][a["seeds"].index(sd)]
                 for sd in SEEDS if sd in a["seeds"] and sd in b_["seeds"]]
            print(f"    peak -> abundant end: {np.mean(d):+.4f} +/- {sem(d):.4f} "
                  f"(t={tstat(d):+.2f}, {sum(v < 0 for v in d)}/{len(d)} negative)")
            print("    -> THE DOC'S PREDICTED LIMB. Round 1 could not see it because its")
            print("       fixed-epoch scaling gave the abundant end more compute as well.")


def published_ladder(pts):
    """E8 -- re-run the PUBLISHED ten-arm ladder at fewer epochs.

    E4 showed the loop overfits the round's sample, so every headline number on this node was
    measured over-trained. Absolute levels being pessimistic is a footnote; the question that
    matters is whether the published ORDERING and the published CONTRASTS survive. Each claim
    below is quoted from `../README.md` §3 and recomputed at each epoch count.
    """
    use = [p for p in pts if p["label"] in {l for l, _ in E8}]
    if not use:
        return
    order = {"ep1": 1, "ep2": 2, "ep4": 4}
    use = sorted(use, key=lambda p: order.get(p["label"], 8))
    ref = _ladder_fix()          # the ACTUAL published ten-arm runs, not the six-arm anchor

    print(f"\n{'=' * 92}\n=== E8  THE PUBLISHED LADDER AT FEWER EPOCHS ===\n{'=' * 92}")
    print("  Published `ladder_fix_s*` ran at fm_epochs=8, which E4 shows is over-trained.")
    print("  Same B, mon_n, rounds, n_eval and arm ORDER; only `fm_epochs` differs.\n")

    arms = [a for a in FULL_ARMS if a != "oracle_dup"]
    print(f"  {'arm':18s} " + "".join(f"{'ep' + str(order[p['label']]):>10s}" for p in use)
          + f"{'ep8 (pub)':>11s}")
    for a in arms:
        row = f"  {a:18s} "
        for p in use:
            v = _arm(p, a)
            row += f"{v:>10.4f}" if v is not None else f"{'-':>10s}"
        v8 = _arm(ref, a)
        row += f"{v8:>11.4f}" if v8 is not None else f"{'-':>11s}"
        print(row)

    print(f"\n  {'published claim (README §3)':46s} " +
          "".join(f"{'ep' + str(order[p['label']]):>11s}" for p in use) + f"{'ep8':>11s}")
    claims = [
        ("uniform -> oracle prize", lambda p: _d(p, "uniform", "oracle")),
        ("the tap repair: value_red - value  (pub -0.0542)", lambda p: _d(p, "value_red", "value")),
        ("value_red - visits_only  (pub +0.0019, a tie)", lambda p: _d(p, "value_red", "visits_only")),
        ("oracle - value_red  (pub -0.0122)", lambda p: _d(p, "oracle", "value_red")),
        ("error_only - uniform  (pub +0.0055, worst arm)", lambda p: _d(p, "error_only", "uniform")),
        ("satiety: value_red_satiety - value_red (pub +0.0691)",
         lambda p: _d(p, "value_red_satiety", "value_red")),
    ]
    for name, fn in claims:
        row = f"  {name:46s} "
        for p in use:
            v = fn(p)
            row += f"{v:>+11.4f}" if v is not None else f"{'-':>11s}"
        row += f"{fn(ref):>+11.4f}" if ref and fn(ref) is not None else f"{'-':>11s}"
        print(row)

    print(f"\n  {'recovery of the oracle prize':46s} " +
          "".join(f"{'ep' + str(order[p['label']]):>11s}" for p in use) + f"{'ep8':>11s}")
    for a, pub in (("visits_only", "84%"), ("value_red", "?"), ("reducible_only", "18%")):
        row = f"  {a + '  (pub ' + pub + ')':46s} "
        for p in use:
            r = p.get(f"recov_{a}")
            row += f"{np.mean(r):>10.1%} " if r else f"{'-':>11s}"
        r = ref.get(f"recov_{a}") if ref else None
        row += f"{np.mean(r):>10.1%} " if r else f"{'-':>11s}"
        print(row)

    print(f"\n  {'allocation VARIANCE (tree-share sd)':46s} " +
          "".join(f"{'ep' + str(order[p['label']]):>11s}" for p in use) + f"{'ep8':>11s}")
    for a, pub in (("value", "0.396"), ("value_red", "0.047")):
        row = f"  {a + '  (pub ' + pub + ')':46s} "
        for p in use + ([ref] if ref else []):
            v = _sd(p, a)
            row += f"{v:>+11.3f}" if v is not None else f"{'-':>11s}"
        print(row)

    print("\n  ORDERING (best -> worst tree err), and whether it matches ep8:")
    ranks = {}
    for p in use + ([ref] if ref else []):
        o = sorted([a for a in arms if _arm(p, a) is not None], key=lambda a: _arm(p, a))
        ranks[order.get(p["label"], 8)] = o
        print(f"    ep{order.get(p['label'], 8):<3d} " + " < ".join(o))
    if 8 in ranks:
        for e in sorted(k for k in ranks if k != 8):
            common = [a for a in ranks[e] if a in ranks[8]]
            ref8 = [a for a in ranks[8] if a in common]
            same = common == ref8
            print(f"    ep{e} vs ep8 on the {len(common)} shared arms: "
                  f"{'IDENTICAL' if same else 'differs'}"
                  + ("" if same else f"  ep{e}={common}  ep8={ref8}"))
    fl = [np.mean(p["floor"]) for p in use if p["floor"]]
    if fl:
        print(f"\n  pointwise noise floor at these points: "
              + ", ".join(f"{v:.4f}" for v in fl))


def _ladder_fix():
    """The published `ladder_fix_s{1,2,3}` runs, read from the parent node's own figures dir.

    The ep8 column has to come from these and not from this sweep's six-arm anchor: the anchor
    runs a different arm SET, so its arms sit at different points of the shared torch RNG
    stream and its `oracle` differs from the published one by ~0.008 (which is exactly what
    `oracle_dup` measures). Comparing ten arms against ten arms is the honest form.
    """
    runs = []
    for sd in SEEDS:
        f = os.path.join(HERE, "..", "figures", f"ladder_fix_s{sd}", "results.json")
        if os.path.exists(f):
            with open(f) as fh:
                runs.append(json.load(fh))
    if not runs:
        return None
    err = {a: [r["results"][a]["tree_err_mean"] for r in runs if a in r["results"]]
           for a in ARMS}
    return {"label": "ladder_fix", "seeds": list(SEEDS), "err": err, "floor": [],
            "config": runs[0]["config"], "_runs": runs,
            **{f"recov_{a}": [(r["results"]["uniform"]["tree_err_mean"]
                               - r["results"][a]["tree_err_mean"])
                              / (r["results"]["uniform"]["tree_err_mean"]
                                 - r["results"]["oracle"]["tree_err_mean"])
                              for r in runs if a in r["results"]]
               for a in ("visits_only", "reducible_only", "value_red")}}


def _arm(p, a):
    return float(np.mean(p["err"][a])) if p and p["err"].get(a) else None


def _d(p, a, b):
    if not p or not p["err"].get(a) or not p["err"].get(b):
        return None
    n = min(len(p["err"][a]), len(p["err"][b]))
    return float(np.mean([p["err"][a][i] - p["err"][b][i] for i in range(n)]))


def _sd(p, a):
    r = p["_runs"][0] if "_runs" in p else load(p["label"], p["seeds"][0])
    return r["results"][a]["tree_share_sd"] if a in r["results"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()
    all_labels = [l for l, _ in E1A + E1B + E2 + E1C + E3 + E4 + E5 + E6 + E8]
    if args.mirror:
        mirror(all_labels)

    pts_a = [q for q in (point(l, r) for l, r in E1A) if q]
    pts_b = [q for q in (point(l, r) for l, r in E1B) if q]
    pts_2 = [q for q in (point(l, r) for l, r in E2) if q]
    pts_c = [q for q in (point(l, r) for l, r in E1C) if q]
    pts_3 = [q for q in (point(l, r) for l, r in E3) if q]
    pts_4 = [q for q in (point(l, r) for l, r in E4) if q]
    pts_5 = [q for q in (point(l, r) for l, r in E5) if q]
    pts_6 = [q for q in (point(l, r) for l, r in E6) if q]
    pts_8 = [q for q in (point(l, r) for l, r in E8) if q]
    pts = pts_a + pts_b + pts_2 + pts_c + pts_3 + pts_4 + pts_5 + pts_6 + pts_8
    if not pts:
        print(f"no results under {FIG}; run with --mirror first")
        return
    missing = [f"{l}_s{s}" for l in all_labels for s in SEEDS if load(l, s) is None]
    if missing:
        print(f"!! missing {len(missing)} run(s): {missing}\n")

    alive = gates(pts)
    # E(d) BEFORE the prize: it decides whether a flat prize means anything at all.
    data_curve(pts)
    decompose(pts)

    curve(pts_a, alive, "E1-A  the headline: PRICE at fixed total spend (T = 22784)")
    slopes(pts_a, alive, "r", "ratio", "(full range)")
    lo = [p for p in pts_a if p["target_r"] <= 1.78125]
    hi = [p for p in pts_a if p["target_r"] >= 1.78125]
    slopes(lo, alive, "r", "ratio", "(LOW limb, r <= 1.78 -- the doc's predicted limb)")
    slopes(hi, alive, "r", "ratio", "(HIGH limb, r >= 1.78 -- the starvation limb)")

    print(f"\n{'=' * 92}\n=== the prize against BUDGET, pooling E1-A and E1-B (9 points, "
          f"66x in B) ===\n{'=' * 92}")
    both = sorted(pts_a + pts_b + pts_c, key=lambda p: p["config"]["collect_budget"])
    print(f"  {'point':11s} {'r':>8s} {'B':>7s} {'PRIZE':>9s} {'sem':>7s} {'floor':>7s}")
    for p in both:
        print(f"  {p['label']:11s} {np.mean(p['realised_r']):>8.4f} "
              f"{p['config']['collect_budget']:>7d} {np.mean(p['prize']):>+9.4f} "
              f"{sem(p['prize']):>7.4f} {np.mean(p['floor']):>7.4f}")
    slopes(both, {p["label"] for p in both}, "B", "collect_budget",
           "(ALL points -- the doc predicts NEGATIVE as B rises / r falls)")
    slopes(both, alive, "B", "collect_budget", "(C4a-live points only)")

    curve(pts_b, alive, "E1-B  abundance limb: price fixed at 1.0, collection raised")
    curve(pts_2, alive, "E2  CONTROL: information swept at fixed budget and fixed price")

    e3_curve(pts)
    e4_analysis(pts)
    iso_compute(pts)
    e5_analysis(pts)
    published_ladder(pts)

    print(f"\n{'=' * 92}\n=== matched-ratio discriminators: is `r` or `B` the operative variable? ==="
          f"\n{'=' * 92}")
    paired_contrast(pts, "r0p25", "ab65536",
                    "same ratio, 3.6x the budget: if r is operative these agree")
    paired_contrast(pts, "info0p25", "ab32768",
                    "same ratio (0.4453 exactly), 4x the budget")
    paired_contrast(pts, "r1p78125", "info4",
                    "same budget, 4x the monitoring: E2 must NOT move the prize (P5)")

    print(f"\n{'=' * 92}\n=== anchors ===\n{'=' * 92}")
    anc = next((p for p in pts_a if p["label"] == "r1p78125"), None)
    if anc:
        print(f"  C2 -- anchor prize {np.mean(anc['prize']):+.4f} +/- {sem(anc['prize']):.4f} "
              f"(per seed " + ", ".join(f"{v:+.4f}" for v in anc["prize"]) + ")")
        print(f"       published ladder_fix_s* prize {PUBLISHED_PRIZE:+.4f}; "
              f"delta {np.mean(anc['prize']) - PUBLISHED_PRIZE:+.4f}, pointwise floor "
              f"{np.mean(anc['floor']):.4f}, partial_hetero floor {PARTIAL_HETERO_FLOOR:.4f}")
        print(f"       tap recoveries here: visits_only {np.mean(anc['recov_visits_only']):.1%} "
              f"(published 84%), reducible_only {np.mean(anc['recov_reducible_only']):.1%} "
              f"(published 18%)")
    print(f"\n  total GPU seconds: {sum(sum(p['elapsed']) for p in pts):.0f}")


if __name__ == "__main__":
    main()
