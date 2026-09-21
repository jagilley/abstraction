"""[tessitura] THE RE-RUN'S REDUCTION — the three objects, read as they were trained.

`reduce_logged.py` shows the banked record carries the world and not the judge's level.
`reduce_rows.py` recovers the level statically, with the final critic, on the ring-buffer tail.
This file reads the tag that logs it live (`results/RUN_ts_s0.sh`):

  [N] THE NORM'S TIME COURSE. Per slot per cycle, the judge's mean predicted P(solve) beside
      the world's base rate on the identical rows — the calibration of `reduce_rows.py`'s [N]
      with the time axis put back. Pooled per era, and as a lag: how many cycles the judge's
      expectation trails the outcome distribution it is being fed, against the ring buffer's
      own lag from `reduce_logged.py` [B], which is the floor.
  [P] THE FIXED PANELS. One panel per (slot, era), frozen once and scored every cycle after.
      Its base rate is a constant by construction, so every movement is the reader's. THE TRUNK
      MOVES UNDER IT: on a live learner the reader is critic + trunk and only the two together
      could be frozen, which is exactly what `logit_reading/striatum/norm/` could freeze and
      this substrate cannot. Said at the table, not only here.
  [C] COST VERSUS STRUCTURE AT THE WRITE. The judge and the prior against the verdict and
      against the repair-set label on the identical rows, each with the other held fixed —
      striatum §2's question with the DP prior where the model's surprisal stood.
  [T] the gates the tag carries: TS-1 (bit-identity against the banked arm) is read by
      `analyze_voicing.py`'s section [R] and quoted here; the instrument tally is checked to
      have MOVED, which is this round's liveness half.

Usage (from experiments/):
    python3 rhm/practice/voicing/tessitura/reduce_rerun.py --tag ts_s0
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICING = os.path.dirname(HERE)
FIG = os.path.join(VOICING, "figures")


def f(x, nd=3):
    return ("  -  " if x is None or (isinstance(x, float) and not np.isfinite(x))
            else f"{x:.{nd}f}")


def wmean(vals, wts):
    vals = np.asarray(vals, np.float64)
    wts = np.asarray(wts, np.float64)
    m = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    return float((vals[m] * wts[m]).sum() / wts[m].sum()) if m.any() else None


def ema_lag(inst, trail, taus=(1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)):
    """The trailing series' effective window, in cycles, as the best-fitting EMA of the
    instantaneous one. `reduce_logged.py`'s own helper, repeated here so the two reductions
    stay independent files."""
    m = np.isfinite(inst) & np.isfinite(trail)
    if int(m.sum()) < 20:
        return None
    best = None
    for tau in taus:
        a = 1.0 / float(tau)
        run, pred = None, np.full(len(inst), np.nan)
        for i in range(len(inst)):
            if not np.isfinite(inst[i]):
                continue
            run = inst[i] if run is None else (1 - a) * run + a * inst[i]
            pred[i] = run
        mm = m & np.isfinite(pred)
        if int(mm.sum()) < 20:
            continue
        e = float(np.sqrt(np.mean((pred[mm] - trail[mm]) ** 2)))
        if best is None or e < best[1]:
            best = (tau, e, int(mm.sum()))
    return best


def _load(tag, arm):
    return json.load(open(os.path.join(FIG, tag, arm, "results.json")))


def seed_table(tags, arm, o):
    """[tessitura] THE SIGN TABLE. §8's two headlines are RATE-LIKE — a per-era signed gap and a
    per-era level on frozen rows — and a rate-like claim is a statement about a trajectory, which
    is exactly what a seed moves. So each is restated here as a SIGN that either agrees across
    the seeds or does not. **The seeds are never averaged**, this lineage's standing rule."""
    o("=" * 104)
    o(f"[X] THE TWO SEEDS, AS SIGNS — {', '.join(tags)}")
    o("=" * 104)
    o("  Each of §8's headlines restated as a sign. Magnitudes are printed so the reader can see")
    o("  how far apart they are, but the CLAIM is the sign, and a magnitude that moves between")
    o("  seeds is reported as moving rather than averaged away.")
    o("")
    D = {}
    for t in tags:
        d = _load(t, arm)
        D[t] = {"v": d["log"]["vo"], "eras": np.asarray(d["log"]["era"], np.int64),
                "seed": d["config"].get("seed")}

    # ---- headline 1: the norm's gap to the world, per era ------------------------------- #
    o("  AN EXCLUSION CRITERION, STATED BEFORE THE TABLES AND APPLIED MECHANICALLY. A critic")
    o("  begins governing only once a slot holds `vo_critic_min = 256` filed rows, so an era")
    o("  early enough can contain little but the readout's own INITIALISATION. `build_critic`")
    o("  zeroes every bias and draws the weights at N(0, 0.02), so an untrained critic emits a")
    o("  ~0 logit and reads 0.500 at EVERY candidate — which makes the signature exact and")
    o("  checkable rather than a matter of judgement: a cell is flagged `init` iff")
    o("  |V_w - 0.5| < 0.01 AND V_max - V_tab < 0.005, i.e. the judge cannot tell its")
    o("  candidates apart at all. The separator is not marginal: at seed 2's era 1 that gap is")
    o("  0.000292 on 671 rows, and at its era 2 it is 0.106531 on 309,334. Flagged cells are")
    o("  printed with a `*` and the sign patterns are given BOTH WAYS, with and without them.")
    o("  Nothing is dropped.")
    o("")
    o("  HEADLINE 1 — the norm undershoots a rising world and converges. `V_w - base` per era,")
    o("  on the rows the base rate is read from (filed):")
    o(f"    {'tag (seed)':16} " + "  ".join(f"{'e'+str(e):>9}" for e in range(1, 6))
      + "   sign pattern")
    pat, pat_ok, band = {}, {}, {}
    for t in tags:
        v, eras = D[t]["v"], D[t]["eras"]
        cells, signs, live = [], [], []
        for e in range(1, 6):
            gs, ws, vw, vmx, vtb = [], [], [], [], []
            for i, c in enumerate(v):
                if eras[i] != e:
                    continue
                for q in ((c or {}).get("critic") or {}).values():
                    n = float(q.get("n") or 0)
                    ts = q.get("ts") or {}
                    if not n or q.get("base_rate") is None or ts.get("mean_p") is None:
                        continue
                    gs.append(float(ts["mean_p"]) - float(q["base_rate"]))
                    ws.append(n)
                    vw.append(float(ts["mean_p"]))
                    vmx.append(float(ts.get("mean_pmax", float("nan"))))
                    vtb.append(float(ts.get("mean_ptab", float("nan"))))
            g = wmean(gs, ws)
            mp, mx, tb = wmean(vw, ws), wmean(vmx, ws), wmean(vtb, ws)
            # the separator is not marginal: at seed 2's era 1 the gap V_max - V_tab is
            # 0.000292 on 671 rows, and at its era 2 it is 0.106531 on 309,334. 0.005 sits
            # two orders of magnitude from either side.
            is_init = (mp is not None and abs(mp - 0.5) < 0.01
                       and mx is not None and tb is not None and (mx - tb) < 0.005)
            cells.append((f"{g:>+8.4f}*" if is_init else f"{g:>+9.4f}")
                         if g is not None else f"{'-':>9}")
            signs.append("*" if is_init else
                         ("+" if (g or 0) > 0 else ("-" if (g or 0) < 0 else "0")))
            live.append(None if (is_init or g is None) else g)
        pat[t] = "".join(signs)
        pat_ok[t] = "".join(x for x in signs if x != "*")
        band[t] = max((abs(x) for x in live if x is not None), default=None)
        o(f"    {t + ' (s' + str(D[t]['seed']) + ')':16} " + "  ".join(cells)
          + f"   {pat[t]}")
    o(f"    -> the full per-era sign pattern agrees across the seeds: "
      f"{len(set(pat.values())) == 1}")
    o(f"    -> with `init` cells excluded, the sign pattern agrees: "
      f"{len(set(pat_ok.values())) == 1}   {pat_ok}")
    o(f"    -> the STATE-like form instead — max |V_w - base| over the non-init eras, per "
      f"seed: { {t: (round(b, 4) if b is not None else None) for t, b in band.items()} }")
    o(f"    -> every seed calibrated to within 0.04 at every non-init era: "
      f"{all((b is not None and b <= 0.04) for b in band.values())}")
    o("")

    # ---- headline 2: the panels' drift on identical rows -------------------------------- #
    o("  HEADLINE 2 — on identical rows the judge's level rises with the era it is READ in.")
    o("  Per seed, the panels frozen in era 1 (and in era 2), read across the eras:")
    o(f"    {'tag (seed)':16} {'frozen':>7} {'base':>7} "
      + "  ".join(f"{'e'+str(e):>9}" for e in range(1, 6))
      + "   drift all / non-init")
    drift = collections.defaultdict(dict)
    drift_ok = collections.defaultdict(dict)
    for t in tags:
        v, eras = D[t]["v"], D[t]["eras"]
        pan = collections.defaultdict(dict)
        for i, c in enumerate(v):
            for pk, q in ((c or {}).get("ts_panel") or {}).items():
                pan[pk][i] = q
        for fe in (1, 2):
            ks = [k for k in pan if int(k.split("|e")[1]) == fe]
            if not ks:
                continue
            base = float(np.mean([next(iter(pan[k].values()))["base"] for k in ks]))
            cells, vals = [], []
            for e in range(1, 6):
                vv = [pan[k][i]["mean_p"] for k in ks for i in pan[k] if eras[i] == e]
                cells.append(f"{np.mean(vv):>9.4f}" if vv else f"{'-':>9}")
                vals.append(float(np.mean(vv)) if vv else None)
            got = [x for x in vals if x is not None]
            # the SAME init criterion: a panel read in an era where the critic is still at its
            # initialisation reads ~0.5 on everything, so that cell is flagged and the drift is
            # reported both over every era and over the non-init ones.
            live = [x for x in vals if x is not None and abs(x - 0.5) >= 0.01]
            dr = (got[-1] - got[0]) if len(got) > 1 else None
            dr_ok = (live[-1] - live[0]) if len(live) > 1 else None
            drift[t][fe] = dr
            drift_ok[t][fe] = dr_ok
            o(f"    {t + ' (s' + str(D[t]['seed']) + ')':16} {fe:>7} {base:>7.3f} "
              + "  ".join(cells)
              + f"   {('%+.4f' % dr) if dr is not None else '-'}"
              + f" / {('%+.4f' % dr_ok) if dr_ok is not None else '-'}")

    def _sgn(dd):
        return {t: ("+" if all((q or 0) > 0 for q in dd[t].values()) else
                    "-" if all((q or 0) < 0 for q in dd[t].values()) else "mixed")
                for t in tags}
    s2a, s2b = _sgn(drift), _sgn(drift_ok)
    o(f"    -> sign over EVERY era: {s2a}   agrees: {len(set(s2a.values())) == 1}")
    o(f"    -> sign over the NON-INIT eras: {s2b}   agrees: {len(set(s2b.values())) == 1}")
    o("")

    # ---- headline 3: cost versus structure, as a sign ------------------------------------ #
    o("  HEADLINE 3 — at matched structure the judge is near chance on the cost; at matched")
    o("  cost it reads the structure. Printed as the DIFFERENCE (structure|cost) - (cost|struct)")
    o("  per era, whose sign is the claim:")
    o(f"    {'tag (seed)':16} {'rows':7} "
      + "  ".join(f"{'e'+str(e):>9}" for e in range(1, 6)) + "   sign pattern")
    pat3 = {}
    for t in tags:
        v, eras = D[t]["v"], D[t]["eras"]
        for which in ("filed", "probe"):
            cells, signs = [], []
            for e in range(1, 6):
                rows = [q for i, c in enumerate(v) if eras[i] == e
                        for k, q in ((c or {}).get("ts_struct") or {}).items()
                        if k.startswith(which + ":")]
                num1 = den1 = num2 = den2 = 0.0
                for q in rows:
                    if q.get("crit_st_g_y") is not None:
                        num1 += q["crit_st_g_y"] * q["n"]; den1 += q["n"]
                    if q.get("crit_y_g_st") is not None:
                        num2 += q["crit_y_g_st"] * q["n"]; den2 += q["n"]
                if den1 and den2:
                    dd = num1 / den1 - num2 / den2
                    cells.append(f"{dd:>+9.4f}")
                    signs.append("+" if dd > 0 else "-")
                else:
                    cells.append(f"{'-':>9}")
                    signs.append(".")
            pat3[(t, which)] = "".join(signs)
            o(f"    {t + ' (s' + str(D[t]['seed']) + ')':16} {which:7} "
              + "  ".join(cells) + f"   {pat3[(t, which)]}")
    for which in ("filed", "probe"):
        ps = {pat3[(t, which)] for t in tags if (t, which) in pat3}
        tail_ = {pat3[(t, which)][2:] for t in tags if (t, which) in pat3}
        o(f"    -> {which}: patterns {sorted(ps)}; agrees over all five eras: {len(ps) == 1}; "
          f"agrees over eras 3-5 (where the frontier is live and the critic is not warming "
          f"up): {len(tail_) == 1}  {sorted(tail_)}")
    o("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ts_s0")
    ap.add_argument("--arm", default="ovt_comp_pr_sh")
    ap.add_argument("--seeds", default="",
                    help="tag,tag — emit ONLY the cross-seed sign table")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.seeds:
        tags = [x for x in a.seeds.split(",") if x]
        lines = []

        def o(t=""):
            lines.append(t)
            print(t, flush=True)
        seed_table(tags, a.arm, o)
        out = a.out or os.path.join(HERE, "results", "seed_signs.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n[wrote] {out}")
        return
    root = os.path.join(FIG, a.tag, a.arm)
    d = json.load(open(os.path.join(root, "results.json")))
    log, cfg = d["log"], d["config"]
    v = log["vo"]
    eras = np.asarray(log["era"], np.int64)
    lines = []

    def o(t=""):
        lines.append(t)
        print(t, flush=True)

    o("=" * 104)
    o(f"[tessitura] {a.tag}:{a.arm} — the judge's level as it was trained")
    o("=" * 104)
    o(f"  knobs: ts_norm={cfg.get('ts_norm')}  ts_panel={cfg.get('ts_panel')}  "
      f"ts_struct={cfg.get('ts_struct')}  seed={cfg.get('seed')}  "
      f"cycles={len(v)}  eras={[int((eras == e).sum()) for e in range(1, 6)]}")
    o("")

    # ------------------------------------------------------------------ [N] the norm --- #
    o("-" * 104)
    o("[N] THE NORM'S TIME COURSE — the judge's expectation against the world it is being fed")
    o("-" * 104)
    o("  `base` is the mean verdict over the audited held-out rows; `V_w` the judge's mean")
    o("  predicted P(solve) at the candidate those rows carry; `V_max` and `V_tab` its max and")
    o("  mean over the operative table at the same contexts. All four on the IDENTICAL rows,")
    o("  n-weighted across slots inside each era.")
    o("")
    for which, kn, kb in (("FILED", "n", "base_rate"), ("PROBE", "probe_n", "probe_base_rate")):
        o(f"  {which}")
        o(f"    {'era':>4} {'cycles':>7} {'n':>9} {'base':>7} {'V_w':>7} {'V_w-base':>9} "
          f"{'V_max':>7} {'V_tab':>7} {'AUC':>7}")
        for e in range(1, 6):
            acc = collections.defaultdict(list)
            wts = []
            for i, c in enumerate(v):
                if eras[i] != e:
                    continue
                for _k, q in ((c or {}).get("critic") or {}).items():
                    n = float(q.get(kn) or 0)
                    ts = (q.get("ts") if which == "FILED"
                          else ((q.get("probe_x") or {}).get("ts")))
                    if not n or q.get(kb) is None:
                        continue
                    wts.append(n)
                    acc["base"].append(float(q[kb]))
                    acc["auc"].append(q.get("auc" if which == "FILED" else "probe_auc"))
                    for kk in ("mean_p", "mean_pmax", "mean_ptab"):
                        acc[kk].append((ts or {}).get(kk))
            if not wts:
                continue
            vw = wmean(acc["mean_p"], wts)
            bs = wmean(acc["base"], wts)
            o(f"    {e:>4} {int((eras == e).sum()):>7} {int(sum(wts)):>9} {f(bs):>7} "
              f"{f(vw):>7} "
              + (f"{vw - bs:>+9.3f} " if (vw is not None and bs is not None) else f"{'-':>9} ")
              + f"{f(wmean(acc['mean_pmax'], wts)):>7} "
                f"{f(wmean(acc['mean_ptab'], wts)):>7} {f(wmean(acc['auc'], wts)):>7}")
        o("")
    # the lag: the judge's level against the world's trailing and instantaneous rates
    o("  THE LAG. `base` is already a trailing estimate (the ring buffer, whose own window")
    o("  `reduce_logged.py` [B] measures); the cycle's own solve rate is the instantaneous")
    o("  world. Fitting an EMA of each to the judge's level gives the judge's window in cycles.")
    nf = np.array([(c or {}).get("n_filed", 0) for c in v], np.float64)
    ns = np.array([(c or {}).get("n_solved", 0) for c in v], np.float64)
    inst = np.where(nf > 0, ns / np.maximum(nf, 1), np.nan)
    ser = {k: np.full(len(v), np.nan) for k in ("base", "vw")}
    for i, c in enumerate(v):
        cr = (c or {}).get("critic") or {}
        if not cr:
            continue
        w = [float(q.get("n") or 0) for q in cr.values()]
        ser["base"][i] = wmean([q.get("base_rate") for q in cr.values()], w) or np.nan
        ser["vw"][i] = wmean([(q.get("ts") or {}).get("mean_p") for q in cr.values()],
                             w) or np.nan
    taus = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)
    o("    THE WHOLE RMSE CURVE, not only its argmin: a shallow curve is a weak identification")
    o("    and saying so is the difference between a measurement and a number.")
    o(f"      {'against':28} " + "  ".join(f"{t:>6}" for t in taus))
    for nm, src in (("the cycle's own rate", inst), ("the buffer's trailing rate", ser["base"])):
        cells = []
        for t in taus:
            b = ema_lag(src, ser["vw"], taus=(t,))
            cells.append(f"{b[1]:.4f}" if b else "  -  ")
        b = ema_lag(src, ser["vw"], taus=taus)
        o(f"      {nm:28} " + "  ".join(f"{c:>6}" for c in cells))
        o(f"      {'-> argmin':28} tau = {b[0]} cycles, rmse {b[1]:.4f}, n {b[2]}" if b
          else f"      {'-> argmin':28} -")
    m = np.isfinite(ser["vw"]) & np.isfinite(ser["base"])
    if int(m.sum()) > 10:
        o(f"    corr(level, trailing base) = {np.corrcoef(ser['vw'][m], ser['base'][m])[0,1]:.3f}"
          f"   mean signed bias = {float((ser['vw'] - ser['base'])[m].mean()):+.4f}")
    o("")

    # ------------------------------------------------------------------ [P] panels --- #
    o("-" * 104)
    o("[P] THE FIXED PANELS — identical rows, a moving reader")
    o("-" * 104)
    o("  One panel per (slot, era), frozen the first cycle in that era at which the slot's")
    o("  buffer held enough rows, and scored by the LIVE critic every cycle after. The base")
    o("  rate is a constant, so `V_w` moving is the reader and not the world.")
    o("  THE TRUNK MOVES UNDER THE PANEL. The critic reads the trunk's pooled hiddens and the")
    o("  trunk is fine-tuned every cycle, so what drifts here is the whole reader. norm could")
    o("  freeze its representation; a live learner cannot, and no reduction can separate them.")
    o("")
    pan = collections.defaultdict(dict)       # panel key -> cycle -> record
    for i, c in enumerate(v):
        for pk, q in ((c or {}).get("ts_panel") or {}).items():
            pan[pk][i] = q
    if not pan:
        o("  (no panels in this tag — `ts_panel` was 0)")
    else:
        o(f"    {'panel':12} {'frozen':>7} {'n':>5} {'base':>7} " +
          "  ".join(f"{'e'+str(e):>13}" for e in range(1, 6)))
        for pk in sorted(pan, key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1].split("|")[0]),
                                             k)):
            rows = pan[pk]
            any_ = next(iter(rows.values()))
            cells = []
            for e in range(1, 6):
                idx = [i for i in rows if eras[i] == e]
                if not idx:
                    cells.append(f"{'-':>13}")
                    continue
                mp = float(np.mean([rows[i]["mean_p"] for i in idx]))
                au = [rows[i]["auc"] for i in idx if rows[i]["auc"] is not None]
                cells.append((f"{mp:.3f}/{np.mean(au):.3f}" if au else f"{mp:.3f}/  -  ").rjust(13))
            o(f"    {pk:12} {any_['frozen_cycle']:>7} {any_['n']:>5} {any_['base']:>7.3f} "
              + "  ".join(cells))
        o("")
        o("    POOLED by the era a panel was FROZEN in (mean over its panels, per reading era):")
        o(f"      {'frozen in':>10} {'panels':>7} {'base':>7} " +
          "  ".join(f"{'read in e'+str(e):>14}" for e in range(1, 6)))
        for fe in range(1, 6):
            ks = [k for k in pan if int(k.split("|e")[1]) == fe]
            if not ks:
                continue
            base = float(np.mean([next(iter(pan[k].values()))["base"] for k in ks]))
            cells = []
            for e in range(1, 6):
                vals = [pan[k][i]["mean_p"] for k in ks for i in pan[k] if eras[i] == e]
                cells.append((f"{np.mean(vals):.4f} (n{len(vals)})" if vals else "-").rjust(14))
            o(f"      {fe:>10} {len(ks):>7} {base:>7.3f} " + "  ".join(cells))
        o("")
        o("    THE SAME PANELS' RANKING (held-out AUC on the frozen rows), which is a different")
        o("    question from the level and moves differently:")
        o(f"      {'frozen in':>10} {'panels':>7} " +
          "  ".join(f"{'read in e'+str(e):>14}" for e in range(1, 6)))
        for fe in range(1, 6):
            ks = [k for k in pan if int(k.split("|e")[1]) == fe]
            if not ks:
                continue
            cells = []
            for e in range(1, 6):
                vals = [pan[k][i]["auc"] for k in ks for i in pan[k]
                        if eras[i] == e and pan[k][i]["auc"] is not None]
                cells.append((f"{np.mean(vals):.4f} (n{len(vals)})" if vals else "-").rjust(14))
            o(f"      {fe:>10} {len(ks):>7} " + "  ".join(cells))
        o("")

    # --------------------------------------------------------- [C] cost vs structure --- #
    o("-" * 104)
    o("[C] COST VERSUS STRUCTURE AT THE WRITE")
    o("-" * 104)
    o("  Two labels on the identical rows: the COST label is the verdict the judge is trained")
    o("  on; the STRUCTURAL label is whether the written (or substituted) class holds a feature")
    o("  in the repair set at that slot, taken under the TRUE root. `crit` is the judge's")
    o("  logit and `dp` the surface model's own score of the same candidate. `| st` and `| y`")
    o("  hold the other label fixed, which is the only form in which 'reads the cost, not the")
    o("  structure' is a claim rather than a correlation. AUCs pooled by pair count.")
    o("")
    for which in ("filed", "probe"):
        o(f"  {which.upper()}")
        o(f"    {'era':>4} {'rows':>8} {'base_y':>7} {'base_st':>8} {'P(y|st)':>8} "
          f"{'P(y|~st)':>9} {'crit~y':>7} {'crit~st':>8} {'dp~y':>7} {'dp~st':>7} "
          f"{'crit~y|st':>10} {'crit~st|y':>10} {'repset':>7}")
        for e in range(1, 6):
            rows = [q for i, c in enumerate(v) if eras[i] == e
                    for k, q in ((c or {}).get("ts_struct") or {}).items()
                    if k.startswith(which + ":")]
            if not rows:
                continue
            n = sum(q["n"] for q in rows)
            cell = np.sum([q["cell"] for q in rows], 0)      # [~st&~y, ~st&y, st&~y, st&y]

            def pooled(key):
                num = den = 0.0
                for q in rows:
                    if q.get(key) is None:
                        continue
                    num += q[key] * float(q["n"])
                    den += float(q["n"])
                return (num / den) if den else None
            p_y_st = (cell[3] / max(cell[2] + cell[3], 1))
            p_y_nst = (cell[1] / max(cell[0] + cell[1], 1))
            o(f"    {e:>4} {n:>8} {np.mean([q['base_y'] for q in rows]):>7.3f} "
              f"{np.mean([q['base_struct'] for q in rows]):>8.3f} {p_y_st:>8.3f} "
              f"{p_y_nst:>9.3f} {f(pooled('crit_y')):>7} {f(pooled('crit_st')):>8} "
              f"{f(pooled('dp_y')):>7} {f(pooled('dp_st')):>7} "
              f"{f(pooled('crit_y_g_st')):>10} {f(pooled('crit_st_g_y')):>10} "
              f"{np.mean([q['rep_set'] for q in rows]):>7.2f}")
        o("")
    o("  PER SLOT LEVEL, pooled over the whole run (the frontier reads differently by design):")
    o(f"    {'rows':8} {'lvl':>4} {'n':>8} {'base_y':>7} {'base_st':>8} {'crit~y':>7} "
      f"{'crit~st':>8} {'crit~y|st':>10} {'crit~st|y':>10}")
    for which in ("filed", "probe"):
        by = collections.defaultdict(list)
        for c in v:
            for k, q in ((c or {}).get("ts_struct") or {}).items():
                if not k.startswith(which + ":"):
                    continue
                by[int(k.split(":")[1])].append(q)
        for lvl in sorted(by):
            rows = by[lvl]
            n = sum(q["n"] for q in rows)

            def pooled(key):
                num = den = 0.0
                for q in rows:
                    if q.get(key) is None:
                        continue
                    num += q[key] * float(q["n"])
                    den += float(q["n"])
                return (num / den) if den else None
            o(f"    {which:8} {lvl:>4} {n:>8} "
              f"{np.mean([q['base_y'] for q in rows]):>7.3f} "
              f"{np.mean([q['base_struct'] for q in rows]):>8.3f} "
              f"{f(pooled('crit_y')):>7} {f(pooled('crit_st')):>8} "
              f"{f(pooled('crit_y_g_st')):>10} {f(pooled('crit_st_g_y')):>10}")
    o("")

    # ------------------------------------------------------------------- [T] gates --- #
    o("-" * 104)
    o("[T] THE TAG'S OWN GATES")
    o("-" * 104)
    o("  TS-1 (bit-identity against banked `vo_s3b:voi3b_comp_pr_yk`) is section [R] of")
    o("  `analyze_voicing.py` and is read there, not here.")
    o(f"  LIVENESS: the instruments' oracle reads must have MOVED the unpriced tally. "
      f"exp_reads = {d.get('exp_reads')}")
    o(f"  panels frozen: {sum(1 for _ in pan)}  ·  struct records: "
      f"{sum(len((c or {}).get('ts_struct') or {}) for c in v)}  ·  "
      f"cycles with a level: {sum(1 for c in v if any((q.get('ts') is not None) for q in ((c or {}).get('critic') or {}).values()))}")

    out = a.out or os.path.join(HERE, "results", f"{a.tag}_rerun.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
