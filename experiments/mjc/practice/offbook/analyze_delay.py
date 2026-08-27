"""Reduce a delay-gate run (`d0`, round 4) and its round-5 content ladder (`d1`).

WHAT THIS PRINTS, AND WHY IT IS ORDERED THIS WAY.

1. The two ceilings, exactly as round 4 reported them: `ref_stale` (ballistic-per-segment under the
   STALE forward model, undelayed -- the arm-neutral playability anchor the pre-fixed criterion
   uses) and the ORIGINAL DRAFT ceiling (2x the best of the three at Delta = 0), which was revised
   pre-run on smoke evidence and is kept on the record so the revision can be audited.
2. The sanity block BEFORE any table, because two of these are load-bearing controls:
     * the ladder's `d0` arm must be bit-identical to the round-4 sweep (in-run assertion, repeated
       here from the record);
     * `reactive` and `live_seg` must be IDENTICAL across every arm -- they are library-independent
       by construction, and if they ever differ, an arm changed something it should not have.
3. The strategy x Delta table per arm, then the pre-fixed verdict per arm, then per-segment medians
   (round 4 found the chain's error concentrating on the closing leg; whether keying or cross-state
   selection moves that is a per-segment question).
4. The ladder read: how much of the gap between round 4's stored content and the playability anchor
   each rung closes, with legato's own measured frozen-unit numbers on this piece as the external
   reference. Ranks and signs are the claim; single seed.

Usage (from experiments/):
    python3 mjc/practice/offbook/analyze_delay.py --tag d1 --fetch
    python3 mjc/practice/offbook/analyze_delay.py --tag d1 --figs      # -> results/d1/*.png
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# legato's measured frozen units on THIS piece, read off `legato/results/l1/l1/<arm>/results.json`
# rather than off its README -- because the two report DIFFERENT STATISTICS and the gate's criterion
# is stated on the MEAN. legato's headline table is `e_perf`, a MEDIAN over its 48 eval geometries;
# this gate reports the mean over its 24. Both are carried here so no comparison is made across
# statistics by accident. (This corrects the round-5 framing, which compared d1's means against
# legato's medians and so overstated the gap by ~25%.)
LEGATO = {
    #                        median@c75  mean@c75  median@c90  mean@c90   by_seg@c90 (medians)
    "seg_frozen":    dict(med75=0.0940, mean75=0.1311, med90=0.1206, mean90=0.1495,
                          by_seg=[0.0550, 0.1108, 0.1861]),
    "phrase_frozen": dict(med75=0.1011, mean75=0.1173, med90=0.1117, mean90=0.1213,
                          by_seg=[0.0964, 0.1198, 0.1103]),
    "seg_plan_launch": dict(med75=0.0951, mean75=0.0985, med90=None, mean90=None,
                            by_seg=[0.0401, 0.1410, 0.0972]),
    "never":         dict(med75=0.0098, mean75=0.0099, med90=None, mean90=None,
                          by_seg=[0.0102, 0.0187, 0.0013]),
}
# legato `L1`'s own commit events -- the content-fidelity reference for the nesting port (round 6).
# `per_state_oracle` is the best ANY candidate achieves per launch state, so it measures the POOL
# before selection; `n_reactive_traces` of `n_traces` says how much of that pool was recorded in the
# already-committed configuration.
LEGATO_COMMITS = {
    "1:0": dict(cycle=25, n_pool=144, n_traces=6, n_reactive_traces=6, n_cand=64, n_score=96,
                score_med=0.115749, best_fixed=0.046003, per_state_oracle=0.013707,
                oracle_gain=3.356, chosen_score=0.042584, launch_tip_spread=0.022945),
    "1:1": dict(cycle=32, n_pool=144, n_traces=6, n_reactive_traces=2, n_cand=64, n_score=96,
                score_med=0.313635, best_fixed=0.156786, per_state_oracle=0.048146,
                oracle_gain=3.257, chosen_score=0.127076, launch_tip_spread=0.053673),
    "1:2": dict(cycle=39, n_pool=144, n_traces=6, n_reactive_traces=2, n_cand=64, n_score=96,
                score_med=0.454634, best_fixed=0.318173, per_state_oracle=0.055356,
                oracle_gain=5.748, chosen_score=0.199343, launch_tip_spread=0.195942),
    "3:0": dict(cycle=55, n_pool=144, n_traces=6, n_reactive_traces=2, n_cand=48, n_score=96,
                score_med=0.286461, best_fixed=0.109157, per_state_oracle=0.058571,
                oracle_gain=1.864, chosen_score=0.100488, launch_tip_spread=0.021889),
}
STRATS = ("reactive", "live_seg", "seg_tape", "chain")


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_offbook/{tag}", dst], check=False,
                   env={**os.environ,
                        "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "chromatic")})


def load(tag):
    root = os.path.join(HERE, "results", tag)
    for p in (os.path.join(root, tag, "delay_gate", "results.json"),
              os.path.join(root, "delay_gate", "results.json"),
              os.path.join(root, "delay_gate.json")):
        if os.path.isfile(p):
            return json.load(open(p)), p
    raise SystemExit(f"no delay-gate results under {root}; run with --fetch")


def f(x, p=4):
    if x is None:
        return "   -  "
    if isinstance(x, bool):
        return "yes" if x else "NO"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def rs_(d):
    return d.get("ref_stale") or 1.0


def cell_e(sw, D, nm):
    c = sw.get(str(D), {}).get(nm, {})
    return c.get("e")


def report(d, tag, refd=None, ref_tag=""):
    cfg = d["config"]
    delays = [int(x) for x in cfg["delays"]]
    dt_ms = 1000.0 * cfg["frame_skip"] * cfg["timestep"]
    L = ["=" * 96,
         f"delay gate `{tag}`  seed {cfg['seed']}  complete={d.get('complete')}  "
         f"Delta in {delays} ({[int(x * dt_ms) for x in delays]} ms)",
         "=" * 96]

    # ---------------------------------------------------------------- 1. the two ceilings
    v0 = d["verdict"]
    L += ["", "## Ceilings (both, as round 4 reported them)",
          f"  ref_stale (ballistic-per-segment, STALE FM, Delta=0)  = {f(d.get('ref_stale'))}"
          "   <- the pre-fixed playability guard",
          f"  original draft (2x the best of the three at Delta=0)  = "
          f"{f(v0.get('guard_ceiling_original_draft'))}   (best at Delta=0 = "
          f"{f(v0.get('best_at_D0'))})",
          "  legato `L1` on this piece (NOTE: its headline is a MEDIAN over 48 geometries; this",
          "  gate's criterion is a MEAN over 24 — both are given so nothing is compared across",
          "  statistics by accident):"]
    for nm, v in LEGATO.items():
        L.append(f"      {nm:<16} c75 med {f(v['med75'])} / mean {f(v['mean75'])}"
                 + (f"   c90 med {f(v['med90'])} / mean {f(v['mean90'])}"
                    if v["med90"] is not None else "")
                 + f"   by-seg(c90, medians) {[round(x, 4) for x in v['by_seg']]}")

    lad = d.get("ladder") or {}
    if "ref_stale_warm" in lad or "ref_stale_adapted" in lad:
        L += ["", "  The SAME anchor recomputed under the run's own models -- FLAGGED, and NOT used",
              "  by the criterion, which keeps the stale-FM `ref_stale` above:",
              f"      under the WARM (pre-nesting) FM    : {f(lad.get('ref_stale_warm'))}",
              f"      under the ADAPTED (post-nesting) FM: {f(lad.get('ref_stale_adapted'))}"]

    # ---------------------------------------------------------------- 2. sanity
    L += ["", "## Sanity"]
    if lad:
        arms = list(lad["arms"])
        # Under `--nest-adapt` the forward model has moved ON PURPOSE, so neither of these two
        # references is round 4's any more: the `d0` delta IS the model effect on an `audit` read,
        # and the cross-arm reference is the pair recomputed under the adapted model.
        adapted = lad.get("base_sweep_adapted") is not None
        base_ref = lad.get("base_sweep_adapted") or d["sweep"]
        idn = lad["arms"].get("d0", {}).get("identity_vs_round4")
        if adapted:
            L.append(f"  ladder `d0` arm vs the round-4 sweep : max|delta| = "
                     f"{'n/a' if idn is None else f'{idn:.3e}'}"
                     f"   EXPECTED NONZERO (the FM adapted; `d0` is an audit read)")
        else:
            L.append(f"  ladder `d0` arm vs the round-4 sweep : max|delta| = "
                     f"{'n/a' if idn is None else f'{idn:.3e}'}"
                     f"   {'OK' if idn == 0.0 else 'MISMATCH'}")
        worst, n = 0.0, 0
        for nm in ("reactive", "live_seg"):
            for D in delays:
                ref = cell_e(base_ref, D, nm)
                for a in arms:
                    x = cell_e(lad["arms"][a]["sweep"], D, nm)
                    if x is not None and ref is not None:
                        worst = max(worst, abs(x - ref)); n += 1
        L.append(f"  library-independent strategies identical across all {len(arms)} arms : "
                 f"max|delta| = {worst:.3e} over {n} values   {'OK' if worst == 0.0 else 'BAD'}"
                 + ("   (reference: the pair recomputed under the ADAPTED model)" if adapted else ""))
        if adapted:
            dr = max(abs(cell_e(base_ref, D, nm) - cell_e(d["sweep"], D, nm))
                     for nm in ("reactive", "live_seg") for D in delays)
            L.append(f"  the adapted pair vs round 4's : max|delta| = {dr:.3e}  "
                     f"(this IS the model effect, reported not asserted)")
        for name, meta in lad.get("libraries", {}).items():
            pool = meta.get("pool") or {}
            dcount = ", ".join(f"{c}:{v['n_distinct']}/{v['n_drawn']}" for c, v in pool.items())
            L.append(f"  lib `{name}` sizes={meta['sizes']}")
            L.append(f"      harvest   : {meta.get('harvest')}")
            L.append(f"      selection : {meta.get('selection')}")
            if dcount:
                L.append(f"      distinct tapes drawn per cell: {dcount}")
            if meta.get("build_cost"):
                bc = meta["build_cost"]
                L.append(f"      build cost (agent): t_priced {bc['t_priced']:.1f} s  "
                         f"steps {bc['steps']}  fb {bc['fb']}  plans {bc['plans']}")
            if meta.get("audition"):
                for c, a_ in sorted(meta["audition"].items()):
                    L.append(f"      audition {c:>4}: n_cand {a_['n_cand']} x n_score "
                             f"{a_['n_score']} -> {a_['n_distinct']}/{a_['n_pick']} distinct  "
                             f"chosen {a_['chosen_score']:.4f}  best_fixed {a_['best_fixed']:.4f}  "
                             f"per-state oracle {a_['per_state_oracle']:.4f} "
                             f"({a_['oracle_gain']:.2f}x)")
        for k_, c_ in (lad.get("harvest_cost") or {}).items():
            L.append(f"  harvest `{k_}` cost (agent): t_priced {c_['t_priced']:.1f} s  "
                     f"steps {c_['steps']}  fb {c_['fb']}  plans {c_['plans']}")
    led = d.get("ledger", {})
    if led:
        L.append(f"  ledger: agent t_priced {led['t_priced']:.1f} s  steps {led['steps_agent']}  "
                 f"fb {led['fb_agent']}  | instrument steps {led['steps_instrument']}")

    # ---------------------------------------------------------------- 2b. the FM through nesting
    fml = ((lad.get("libraries", {}).get("nest") or {}).get("fm_ladder")) if lad else None
    if fml:
        ad = (lad["libraries"]["nest"] or {}).get("fm_adapted")
        L += ["", f"## The forward model through the nesting (FM adapting = {ad})",
              "   legato's own `e_react` ran ~0.010 at its c25-39 commits.",
              "   stage      | nest cycle | e_react med | e_react mean | by-segment medians"]
        for r in fml:
            L.append(f"   {r['stage']:<10} | {r['nest_cycle']:10d} | {r['e_react_med']:11.4f} | "
                     f"{r['e_react_mean']:12.4f} | "
                     + " ".join(f"{x:.4f}" for x in r["by_seg"]))

    # ---------------------------------------------------------------- 3. tables
    def table(sw, title):
        rows = ["", f"### {title}",
                "  D |    ms | " + " | ".join(f"{s:>9}" for s in STRATS) + " |  fb(react/live/seg/chain)"]
        for D in delays:
            e = [cell_e(sw, D, s) for s in STRATS]
            best = min([x for x in e if x is not None], default=None)
            cells = []
            for x in e:
                s = f(x)
                cells.append(f"*{s}" if (x is not None and x == best) else f" {s}")
            fbs = "/".join(str(int(sw[str(D)][s]["fb"])) if "e" in sw[str(D)][s] else "-"
                           for s in STRATS)
            rows.append(f" {D:2d} | {int(D * dt_ms):5d} | " + " | ".join(f"{c:>9}" for c in cells)
                        + f" |  {fbs}")
        deg = []
        for s in STRATS:
            a, b = cell_e(sw, delays[0], s), cell_e(sw, delays[-1], s)
            deg.append(f"{s} {b / a:.2f}x" if (a and b) else f"{s} -")
        rows.append("  degradation Delta_min -> Delta_max:  " + "   ".join(deg))
        return rows

    L += ["", "## Strategy x Delta (mean piece error over the shared eval geometries; * = best)"]
    L += table(d["sweep"], "round-4 sweep (`d0` configuration)")
    if lad:
        for a in lad["arms"]:
            r = lad["arms"][a]
            L += table(r["sweep"], f"arm `{a}`  (library `{r['library']}`, seam `{r['seam_mode']}`)")

    # ---------------------------------------------------------------- 4. verdicts
    L += ["", "## The pre-fixed criterion, per arm",
          "   Delta* = smallest Delta with e_chain <= e_live_seg <= e_reactive, subject to",
          "   min(the three) <= ref_stale.  Nothing below is re-fitted.", "",
          "  arm      | Delta* | rows (D: ordered/playable -> verdict)"]

    def vrow(name, v):
        bits = []
        for r in v["rows"]:
            bits.append(f"{r['delay']}:{'o' if r['ordered'] else '.'}"
                        f"{'p' if r['playable'] else '.'}"
                        f"{'->PASS' if r['passes'] else ''}")
        return f"  {name:<8} | {str(v['delta_star']):>6} | " + "  ".join(bits)

    L.append(vrow("(round4)", v0))
    if lad:
        for a in lad["arms"]:
            L.append(vrow(a, lad["arms"][a]["verdict"]))

    # ---------------------------------------------------------------- 5. per-segment medians
    L += ["", "## Per-segment medians (round 4: the chain's error concentrates on the closing leg)"]
    nseg = len(d["sweep"][str(delays[0])]["reactive"]["by_seg"])
    L.append("  arm      | strat     |  D | " + " | ".join(f"seg{k}" for k in range(nseg)))
    for src_name, sw in ([("(round4)", d["sweep"])]
                         + ([(a, lad["arms"][a]["sweep"]) for a in lad["arms"]] if lad else [])):
        for s in (STRATS if src_name == "(round4)" else ("seg_tape", "chain")):
            for D in delays:
                c = sw[str(D)].get(s, {})
                if "by_seg" not in c:
                    continue
                L.append(f"  {src_name:<8} | {s:<9} | {D:2d} | "
                         + " | ".join(f"{x:.4f}" for x in c["by_seg"]))

    # ---------------------------------------------------------------- 4b. the strong incumbent
    if lad and any("sweep_predict" in r for r in lad.get("arms", {}).values()):
        bn = lad.get("base_sweep_predict")
        L += ["", "## The strong incumbent: efference copy through the delay (`obs_predict`)",
              "   s_hat(t) = FM-rollout(s(t-D), raw commands issued since). Ported from `../accompanist/presto/`.",
              "   EVERY strategy is re-swept under it, not just the incumbent -- scoring a",
              "   predictor-equipped `reactive` against predictor-less stored content would be the",
              "   mirror strawman. `ref_stale` is untouched in both.",
              f"   no-op check at Delta = 0: max|delta| = "
              f"{lad.get('obs_predict_noop_at_D0', float('nan')):.3e}", "",
              "  library-independent strategies, naive -> efference copy:",
              "   D |    ms |      reactive      |      live_seg"]
        base_n = lad.get("base_sweep_adapted") or d["sweep"]
        for D in delays:
            a_, b_ = base_n[str(D)], (bn or {}).get(str(D), {})
            L.append(f" {D:2d} | {int(D * dt_ms):5d} | {a_['reactive']['e']:.4f} -> "
                     f"{b_.get('reactive', {}).get('e', float('nan')):.4f} | "
                     f"{a_['live_seg']['e']:.4f} -> "
                     f"{b_.get('live_seg', {}).get('e', float('nan')):.4f}")
        L += ["", "  Delta* under both incumbents, with the playability MARGIN "
                  "(min of the three / ref_stale):",
              "  arm        | naive Delta* | efference Delta* | per-D  margin naive/efference + "
              "ordered flags"]
        for a in lad["arms"]:
            r = lad["arms"][a]
            vn, vp = r["verdict"], r.get("verdict_predict", {})
            bits = []
            for rn, rp in zip(vn["rows"], vp.get("rows", vn["rows"])):
                mn = min(rn["e_reactive"], rn["e_live_seg"], rn["e_chain"]) / rs_(d)
                mp = min(rp["e_reactive"], rp["e_live_seg"], rp["e_chain"]) / rs_(d)
                bits.append(f"{rn['delay']}:{mn:.2f}/{mp:.2f}"
                            + ("o" if rn["ordered"] else ".")
                            + ("o" if rp.get("ordered") else "."))
            L.append(f"  {a:<10} | {str(vn['delta_star']):>12} | "
                     f"{str(vp.get('delta_star')):>16} | " + "  ".join(bits))
        L.append("   (margin < 1.00 = inside playability; the two letters are `ordered` under the "
                 "naive / efference operator.)")
        for a in lad["arms"]:
            if "sweep_predict" in lad["arms"][a]:
                L += table(lad["arms"][a]["sweep_predict"],
                           f"arm `{a}` UNDER EFFERENCE COPY (library "
                           f"`{lad['arms'][a]['library']}`, seam "
                           f"`{lad['arms'][a]['seam_mode']}`)")

    # ---------------------------------------------------------------- 4c. cross-run per-arm deltas
    if refd is not None and lad:
        rl = refd.get("ladder", {}).get("arms", {})
        shared = [a for a in lad.get("arms", {}) if a in rl]
        if shared:
            L += ["", f"## Per-arm deltas vs `{ref_tag}` (this run minus that one)",
                  "  arm        | strat     | " + " | ".join(f"  D={D}  " for D in delays)]
            for a in shared:
                for st in ("seg_tape", "chain"):
                    row = []
                    for D in delays:
                        x = cell_e(lad["arms"][a]["sweep"], D, st)
                        y = cell_e(rl[a]["sweep"], D, st)
                        row.append("    -   " if (x is None or y is None) else f"{x - y:+.4f}")
                    L.append(f"  {a:<10} | {st:<9} | " + " | ".join(row))
            L.append("  (0.0000 everywhere on a `key`-read arm is the model-independence check: a "
                     "key lookup never touches the FM.)")

    # ---------------------------------------------------------------- 5b. legato content fidelity
    if lad:
        auds = {n: m["audition"] for n, m in lad.get("libraries", {}).items() if m.get("audition")}
        if auds:
            L += ["", "## Content fidelity vs legato `L1`'s own commit events",
                  "   `per_state_oracle` = best ANY candidate achieves per launch state, so it is a",
                  "   property of the POOL before selection. `chosen_score` is what the committed",
                  "   library scores on the held-out launch distribution. Ratios are ours ÷ legato's."]
            for nm, aud in auds.items():
                L.append(f"\n  library `{nm}`")
                L.append("   cell |   pool (traces, reactive) |  chosen (ratio) | "
                         "per-state oracle (ratio) | best_fixed | gain | spread")
                for c in ("1:0", "1:1", "1:2", "3:0", "2:1"):
                    a_ = aud.get(c)
                    if not a_:
                        continue
                    g = LEGATO_COMMITS.get(c)
                    pool = (f"{a_.get('n_pool', '-')} ({a_.get('n_traces', '-')}, "
                            f"{a_.get('n_reactive_traces', '-')} react)"
                            if "n_pool" in a_ else f"n_cand {a_['n_cand']} only")
                    rc = f"{a_['chosen_score'] / g['chosen_score']:.2f}x" if g else "  -  "
                    ro = (f"{a_['per_state_oracle'] / g['per_state_oracle']:.2f}x" if g else "  -  ")
                    L.append(f"   {c:>4} | {pool:>25} | {a_['chosen_score']:.4f} ({rc:>6}) | "
                             f"{a_['per_state_oracle']:.4f} ({ro:>6})       | "
                             f"{a_['best_fixed']:.4f} | {a_['oracle_gain']:.2f} | "
                             f"{a_.get('launch_tip_spread', float('nan')):.4f}")
                    if g:
                        L.append(f"        | legato c{g['cycle']:<3}: pool {g['n_pool']} "
                                 f"({g['n_traces']}, {g['n_reactive_traces']} react)"
                                 f"      | {g['chosen_score']:.4f}          | "
                                 f"{g['per_state_oracle']:.4f}                | "
                                 f"{g['best_fixed']:.4f} | {g['oracle_gain']:.2f} | "
                                 f"{g['launch_tip_spread']:.4f}")
            L += ["", "  Undelayed by-segment medians against legato's plateau (medians, c90):",
                  "   source                  | seg0   | seg1   | seg2",
                  "   legato seg_frozen       | "
                  + " | ".join(f"{x:.4f}" for x in LEGATO['seg_frozen']['by_seg']),
                  "   legato phrase_frozen    | "
                  + " | ".join(f"{x:.4f}" for x in LEGATO['phrase_frozen']['by_seg'])]
            for a in lad["arms"]:
                for s in ("seg_tape", "chain"):
                    c = lad["arms"][a]["sweep"][str(delays[0])].get(s, {})
                    if "by_seg" in c:
                        L.append(f"   {(a + ' ' + s):<23} | "
                                 + " | ".join(f"{x:.4f}" for x in c["by_seg"]))

    # ---------------------------------------------------------------- 6. the ladder read
    if lad:
        rs = d.get("ref_stale")
        base = min(x for x in (cell_e(d["sweep"], delays[0], "seg_tape"),
                               cell_e(d["sweep"], delays[0], "chain")) if x is not None)
        L += ["", "## The ladder: best STORED error at Delta = 0, against the anchors",
              f"   round-4 baseline (best of seg_tape/chain at Delta=0) = {f(base)};  "
              f"ref_stale = {f(rs)};  legato seg_frozen mean@c75 = "
              f"{LEGATO['seg_frozen']['mean75']};  phrase_frozen mean@c75 = "
              f"{LEGATO['phrase_frozen']['mean75']}",
              "  arm      |  chain |seg_tape| best  | gap closed to ref_stale | best stored, ANY D"]
        for a in lad["arms"]:
            sw = lad["arms"][a]["sweep"]
            ec, es = cell_e(sw, delays[0], "chain"), cell_e(sw, delays[0], "seg_tape")
            b = min([x for x in (ec, es) if x is not None], default=None)
            closed = ("-" if (b is None or base <= rs) else
                      f"{100.0 * (base - b) / (base - rs):6.1f}%")
            anyd = min([cell_e(sw, D, s) for D in delays for s in ("seg_tape", "chain")
                        if cell_e(sw, D, s) is not None], default=None)
            L.append(f"  {a:<8} | {f(ec)} | {f(es)} | {f(b)} | {closed:>23} | {f(anyd)}")
        L.append("   ('gap closed' = fraction of the distance from the round-4 baseline down to "
                 "the playability anchor; >100% means the arm is under the anchor.)")
    return "\n".join(L)


def figures(d, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = d["config"]
    delays = [int(x) for x in cfg["delays"]]
    dt_ms = 1000.0 * cfg["frame_skip"] * cfg["timestep"]
    ms = [D * dt_ms for D in delays]
    rs = d.get("ref_stale")
    lad = d.get("ladder")
    out = os.path.join(HERE, "results", tag)
    os.makedirs(out, exist_ok=True)
    made = []
    col = {"reactive": "#c1440e", "live_seg": "#e08214", "seg_tape": "#2166ac",
           "chain": "#1b7837"}

    # (1) error vs delay, one panel per arm
    panels = ([("round-4 (`d0` config)", d["sweep"])]
              + ([(a, lad["arms"][a]["sweep"]) for a in lad["arms"]] if lad else []))
    nc = min(4, len(panels))
    nr = (len(panels) + nc - 1) // nc
    fig, ax = plt.subplots(nr, nc, figsize=(4.1 * nc, 3.3 * nr), squeeze=False)
    for i, (name, sw) in enumerate(panels):
        a_ = ax[i // nc][i % nc]
        for s in STRATS:
            y = [cell_e(sw, D, s) for D in delays]
            if all(v is None for v in y):
                continue
            a_.plot(ms, y, "o-", ms=3.5, lw=1.4, color=col[s], label=s)
        if rs:
            a_.axhline(rs, color="k", ls="--", lw=1.0)
            a_.text(ms[-1], rs, " ref_stale", va="bottom", ha="right", fontsize=7)
        a_.axhline(LEGATO["seg_frozen"]["mean75"], color="#7b3294", ls=":", lw=1.0)
        a_.set_title(name, fontsize=9)
        a_.set_xlabel("observation delay (ms)", fontsize=8)
        a_.set_ylabel("mean piece error", fontsize=8)
        a_.tick_params(labelsize=7)
        a_.set_yscale("log")
        if i == 0:
            a_.legend(fontsize=7, loc="lower right")
    for j in range(len(panels), nr * nc):
        ax[j // nc][j % nc].axis("off")
    fig.suptitle(f"{tag}: delay sensitivity by strategy, per content arm "
                 f"(dashed = ref_stale, dotted = legato seg_frozen mean)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, f"{tag}_delay_by_arm.png")
    fig.savefig(p, dpi=140); plt.close(fig); made.append(p)

    if lad:
        arms = list(lad["arms"])
        # (2) the ladder at delay 0
        fig, a_ = plt.subplots(figsize=(1.05 * len(arms) + 3.0, 4.0))
        x = range(len(arms))
        for s, dx in (("chain", -0.19), ("seg_tape", 0.19)):
            y = [cell_e(lad["arms"][a]["sweep"], delays[0], s) for a in arms]
            a_.bar([i + dx for i in x], [v if v else 0 for v in y], width=0.36,
                   color=col[s], label=s)
        if rs:
            a_.axhline(rs, color="k", ls="--", lw=1.2, label="ref_stale (playability)")
        a_.axhline(LEGATO["seg_frozen"]["mean75"], color="#7b3294", ls=":", lw=1.2,
                   label="legato seg_frozen (mean)")
        a_.axhline(LEGATO["phrase_frozen"]["mean75"], color="#7b3294", ls="-.", lw=1.0,
                   label="legato phrase_frozen (mean)")
        a_.set_xticks(list(x)); a_.set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
        a_.set_ylabel("mean piece error at Delta = 0")
        a_.set_title(f"{tag}: stored-content quality by build rung (undelayed)", fontsize=10)
        a_.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out, f"{tag}_ladder_delay0.png")
        fig.savefig(p, dpi=140); plt.close(fig); made.append(p)

        # (3) per-segment medians of the chain, per arm, at every delay
        nseg = len(d["sweep"][str(delays[0])]["reactive"]["by_seg"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), squeeze=False)
        for c_, s in enumerate(("chain", "seg_tape")):
            a_ = ax[0][c_]
            for a in arms:
                sw = lad["arms"][a]["sweep"]
                bs = sw[str(delays[0])].get(s, {}).get("by_seg")
                if bs:
                    a_.plot(range(nseg), bs, "o-", ms=3.5, lw=1.2, label=a)
            a_.set_title(f"{s} per-segment median at Delta = 0", fontsize=9)
            a_.set_xlabel("drilled segment"); a_.set_ylabel("median error")
            a_.set_xticks(range(nseg)); a_.tick_params(labelsize=7)
            if c_ == 1:
                a_.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        p = os.path.join(out, f"{tag}_by_segment.png")
        fig.savefig(p, dpi=140); plt.close(fig); made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--vs", default="", help="another tag to diff per-arm results against")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    d, path = load(a.tag)
    ref = None
    if a.vs:
        try:
            ref, _ = load(a.vs)
        except SystemExit:
            print(f"[warn] --vs {a.vs}: no results on disk; skipping the delta table")
    txt = report(d, a.tag, ref, a.vs)
    print(txt)
    dst = os.path.join(HERE, "results", a.tag, f"{a.tag}_report.txt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        fh.write(txt + "\n")
    print(f"\n[report] {dst}   (source: {path})")
    if a.figs:
        for p in figures(d, a.tag):
            print(f"[fig] {p}")


if __name__ == "__main__":
    main()
