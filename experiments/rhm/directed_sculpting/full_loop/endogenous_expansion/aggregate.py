"""Aggregate the target-source ladder: does an endogenous evaluative grader expand?

Reads the mirrored `endo_expansion_*/results.json` under `figures/` and prints the four
readouts together, per seed and paired. PR is never printed alone -- the source docs warn
repeatedly that it is not certified as "the frontier", so every table carries depth, transfer
and control alongside it.

Three panels:

  EXPANSION   the headline 2x2 extended: PR / d4 / ballistic / fresh-FM top-1, each as a PAIRED
              per-seed delta against the same seed's `frozen` arm at the same round. Paired,
              because the anchors exist to be differenced against, not admired.
  BRACKET     `critic` against the two fidelity-matched external controls. This is what makes a
              null a claim: if both controls expand and the critic does not, the deficit is
              structural, not accuracy.
  WIREHEAD    the pre-registered discriminator. `plan_acc` (agreement with the arm's OWN target)
              and `belief_success` (what the loop thinks it achieved) against
              `policy_gain_frac` (realised Dd* on the frozen probe) and `ballistic` (what it
              actually achieved). Rising left column with a flat right column is the
              edit-control signature.

Run from experiments/:
  python3 rhm/directed_sculpting/full_loop/endogenous_expansion/aggregate.py
  python3 rhm/directed_sculpting/full_loop/endogenous_expansion/aggregate.py --figures <dir>
"""

import argparse
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))

# The published static rows of `../expansion.py`'s 2x2, at 3 seeds -- what `frozen`, `dense`
# and `evaluative` have to reproduce for anything else in the run to be readable.
PUBLISHED = {
    "frozen": {"PR_r1": 6.99, "PR_rN": 6.93, "d4_r1": 0.660, "d4_rN": 0.619,
               "ballistic": 0.024, "top1": 0.173},
    "dense": {"PR_r1": 6.31, "PR_rN": 6.61, "d4_r1": 0.634, "d4_rN": 0.626,
              "ballistic": 0.007, "top1": 0.20},
    "evaluative": {"PR_r1": 9.03, "PR_rN": 11.56, "d4_r1": 0.747, "d4_rN": 0.812,
                   "ballistic": 0.369, "top1": 0.554},
}
ORDER = ["frozen", "dense", "evaluative", "dp_noised_g", "dp_noised_a", "critic", "mirror"]


def _ms(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def _t(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size)))


def _fmt(m, sd, n, w=6, p=3):
    if not np.isfinite(m):
        return " " * (w + 9) + "--"
    return f"{m:>{w}.{p}f} +- {sd:<5.{p}f}(n={n})"


def load(figdir):
    """{seed: results-dict}, quick runs skipped.

    A sibling session runs a DIFFERENT design under the same `endo_expansion_*` tag prefix on
    the shared volume (arms `endo_value`/`endo_rollout`/`alloc_kstar`, readouts
    `disagree`/`meter`/`teacher`). Those files glob identically and must never be folded in
    here, so acceptance is by schema, not by filename: a run of THIS script always writes
    `config.conditions` and a top-level `instrument_ceiling`.
    """
    out, foreign = {}, []
    for path in sorted(glob.glob(os.path.join(figdir, "endo_expansion_*", "results.json"))):
        d = json.load(open(path))
        if d["config"].get("quick"):
            continue
        if "conditions" not in d["config"] or "instrument_ceiling" not in d:
            foreign.append(os.path.basename(os.path.dirname(path)))
            continue
        out[int(d["config"]["seed"])] = d
    if foreign:
        print(f"  [skipped {len(foreign)} run(s) written by a different script: "
              f"{', '.join(foreign)}]")
    return out


def last_graded(rows):
    g = [r for r in rows if "ballistic_w16" in r]
    return g[-1] if g else None


def series(d, cond, key, world="base"):
    """Per-round series for one arm; `world` selects the grading venue for graded quantities."""
    rows = d["results"][cond]["rounds"]
    if key in ("PR", "d1", "d2", "d3", "d4"):
        return [r["depth"][key] for r in rows]
    if key in ("kstar_agree", "gain_frac", "tree_share"):
        return [r["policy"][key] for r in rows]
    if key in ("plan_acc", "root_ce", "root_acc"):
        return [r.get(key) for r in rows]
    if key == "target_kstar":
        return [r["target_diag"]["kstar_agree"] for r in rows if "target_diag" in r]
    if key == "target_gain":
        return [r["target_diag"]["gain_frac"] for r in rows if "target_diag" in r]
    if key == "top1":
        return [r["xworld"][world]["fm"]["value_top1_agree"] for r in rows if "xworld" in r]
    return [r["xworld"][world][key] for r in rows if "xworld" in r]


def endpoint(d, cond, key, world="base"):
    s = series(d, cond, key, world)
    return s[-1] if s else float("nan")


def panel_expansion(runs, conds):
    seeds = sorted(runs)
    print(f"\n{'=' * 108}\n=== PANEL 1 -- EXPANSION: all four readouts, paired against the "
          f"same seed's no-loop floor ===\n{'=' * 108}")
    r0 = _ms([runs[s]["belief_round0"]["PR"] for s in seeds])
    d0 = _ms([runs[s]["belief_round0"]["d4"] for s in seeds])
    print(f"  round-0 belief: PR {r0[0]:.2f} +- {r0[1]:.2f}   d4 {d0[0]:.3f} +- {d0[1]:.3f}"
          f"   ({len(seeds)} seeds: {seeds})")
    print(f"\n  ABSOLUTE (base world, last graded round)")
    print(f"  {'arm':<13}{'PR r1':>18}{'PR rN':>18}{'d4 rN':>18}"
          f"{'ballistic':>18}{'freshTop1':>18}")
    for c in conds:
        pr = [series(runs[s], c, "PR") for s in seeds]
        row = [_fmt(*_ms([x[0] for x in pr]), w=5, p=2),
               _fmt(*_ms([x[-1] for x in pr]), w=5, p=2),
               _fmt(*_ms([endpoint(runs[s], c, "d4") for s in seeds]), w=5, p=3),
               _fmt(*_ms([endpoint(runs[s], c, "ballistic_w16") for s in seeds]), w=5, p=3),
               _fmt(*_ms([endpoint(runs[s], c, "top1") for s in seeds]), w=5, p=3)]
        print(f"  {c:<13}" + "".join(x.rjust(18) for x in row))

    print(f"\n  PAIRED vs `frozen`, same seed, same round  (this is the effect)")
    print(f"  {'arm':<13}{'dPR':>20}{'dd4':>20}{'dballistic':>20}{'dtop1':>20}"
          f"{'t(dPR)':>9}")
    base = {s: runs[s] for s in seeds}
    for c in conds:
        if c == "frozen":
            continue
        dpr = [series(base[s], c, "PR")[-1] - series(base[s], "frozen", "PR")[-1]
               for s in seeds]
        dd4 = [endpoint(base[s], c, "d4") - endpoint(base[s], "frozen", "d4") for s in seeds]
        dba = [endpoint(base[s], c, "ballistic_w16")
               - endpoint(base[s], "frozen", "ballistic_w16") for s in seeds]
        dt1 = [endpoint(base[s], c, "top1") - endpoint(base[s], "frozen", "top1")
               for s in seeds]
        print(f"  {c:<13}" + "".join(x.rjust(20) for x in
                                     (_fmt(*_ms(dpr), w=6, p=2), _fmt(*_ms(dd4), w=6, p=3),
                                      _fmt(*_ms(dba), w=6, p=3), _fmt(*_ms(dt1), w=6, p=3)))
              + f"{_t(dpr):>9.2f}")

    print(f"\n  REPRODUCTION of the published 2x2 static rows (3 seeds, `../README.md` §5)")
    print(f"  {'arm':<13}{'PR r1':>16}{'PR rN':>16}{'d4 rN':>16}{'ballistic':>16}"
          f"{'freshTop1':>16}")
    for c, ref in PUBLISHED.items():
        if c not in conds:
            continue
        pr = [series(runs[s], c, "PR") for s in seeds]
        got = (np.mean([x[0] for x in pr]), np.mean([x[-1] for x in pr]),
               np.mean([endpoint(runs[s], c, "d4") for s in seeds]),
               np.mean([endpoint(runs[s], c, "ballistic_w16") for s in seeds]),
               np.mean([endpoint(runs[s], c, "top1") for s in seeds]))
        exp = (ref["PR_r1"], ref["PR_rN"], ref["d4_rN"], ref["ballistic"], ref["top1"])
        print(f"  {c:<13}" + "".join(f"{g:>7.3f}/{e:<8.3f}".rjust(16)
                                     for g, e in zip(got, exp)))
    print("       (cell = this run / published)")


def panel_bracket(runs, conds):
    seeds = sorted(runs)
    print(f"\n{'=' * 108}\n=== PANEL 2 -- THE BRACKET: is the critic's deficit fidelity, or "
          f"structure? ===\n{'=' * 108}")
    ceil = _ms([runs[s]["instrument_ceiling"]["kstar_agree"] for s in seeds])
    print(f"  instrument ceiling (exact DP re-drawn vs the frozen probe): "
          f"k*agree {ceil[0]:.3f} +- {ceil[1]:.3f}  -- read every k*agree against THIS, not 1.0")
    cr = _ms([runs[s]["critic_round0"]["kstar_agree"] for s in seeds])
    cg = _ms([runs[s]["critic_round0"]["gain_frac"] for s in seeds])
    print(f"  round-0 critic target : k*agree {cr[0]:.3f} +- {cr[1]:.3f}   "
          f"gain_frac {cg[0]:.3f} +- {cg[1]:.3f}")
    print(f"  corruption rates      : p_agree "
          f"{_ms([runs[s]['config']['p_agree'] for s in seeds])[0]:.3f}   p_gain "
          f"{_ms([runs[s]['config']['p_gain'] for s in seeds])[0]:.3f}")

    print(f"\n  TARGET FIDELITY as delivered, averaged over rounds (the match, verified)")
    print(f"  {'arm':<13}{'target k*agree':>24}{'target gain_frac':>24}")
    for c in conds:
        if not runs[seeds[0]]["results"][c].get("target"):
            continue
        ka = [np.mean(series(runs[s], c, "target_kstar")) for s in seeds]
        gf = [np.mean(series(runs[s], c, "target_gain")) for s in seeds]
        print(f"  {c:<13}" + _fmt(*_ms(ka), w=6, p=3).rjust(24)
              + _fmt(*_ms(gf), w=6, p=3).rjust(24))

    print(f"\n  EXPANSION vs FIDELITY -- the bracket. dPR/dd4 are paired vs `frozen`.")
    print(f"  {'arm':<13}{'tgt gain_frac':>16}{'dPR':>20}{'dd4':>20}{'dballistic':>20}")
    for c in [x for x in ("evaluative", "dp_noised_g", "critic", "dp_noised_a", "mirror")
              if x in conds]:
        gf = [np.mean(series(runs[s], c, "target_gain")) for s in seeds]
        dpr = [series(runs[s], c, "PR")[-1] - series(runs[s], "frozen", "PR")[-1]
               for s in seeds]
        dd4 = [endpoint(runs[s], c, "d4") - endpoint(runs[s], "frozen", "d4") for s in seeds]
        dba = [endpoint(runs[s], c, "ballistic_w16")
               - endpoint(runs[s], "frozen", "ballistic_w16") for s in seeds]
        print(f"  {c:<13}{_ms(gf)[0]:>16.3f}" + "".join(
            x.rjust(20) for x in (_fmt(*_ms(dpr), w=6, p=2), _fmt(*_ms(dd4), w=6, p=3),
                                  _fmt(*_ms(dba), w=6, p=3))))
    print("\n  reading: if `critic` sits BELOW both controls at bracketed fidelity, the deficit"
          "\n  is structural. If it tracks the fidelity column, the deficit is accuracy.")


def panel_wirehead(runs, conds):
    seeds = sorted(runs)
    print(f"\n{'=' * 108}\n=== PANEL 3 -- WIREHEAD: what the loop believes vs what it "
          f"achieves ===\n{'=' * 108}")
    print("  Pre-registered: rising SELF columns with flat/falling WORLD columns is the"
          "\n  edit-control signature (P(r*) -> 1 while ~99% of sequences go off-grammar).")
    print(f"\n  {'arm':<13}| {'plan_acc':>10}{'believed':>10}{'wire_idx':>10} | "
          f"{'pol k*agr':>11}{'pol gain':>10}{'ballistic':>11}{'freshTop1':>11}{'rootCE':>9}")
    print(f"  {'':13}| {'-- SELF (the loop grading itself) --':^30} | "
          f"{'-- WORLD (held-out instruments) --':^52}")
    for c in conds:
        pa = [series(runs[s], c, "plan_acc")[-1] for s in seeds]
        bs = [endpoint(runs[s], c, "belief_success") for s in seeds]
        wi = [endpoint(runs[s], c, "wirehead_index") for s in seeds]
        pk = [series(runs[s], c, "kstar_agree")[-1] for s in seeds]
        pg = [series(runs[s], c, "gain_frac")[-1] for s in seeds]
        ba = [endpoint(runs[s], c, "ballistic_w16") for s in seeds]
        t1 = [endpoint(runs[s], c, "top1") for s in seeds]
        rc = [series(runs[s], c, "root_ce")[-1] for s in seeds]
        pam = _ms(pa)[0]
        print(f"  {c:<13}| {(f'{pam:.3f}' if np.isfinite(pam) else '--'):>10}"
              f"{_ms(bs)[0]:>10.3f}{_ms(wi)[0]:>10.3f} | "
              f"{_ms(pk)[0]:>11.3f}{_ms(pg)[0]:>10.3f}{_ms(ba)[0]:>11.3f}"
              f"{_ms(t1)[0]:>11.3f}{_ms(rc)[0]:>9.4f}")

    print(f"\n  TRAJECTORY of the discriminator (mean over seeds, per round)")
    for c in conds:
        pg = np.array([series(runs[s], c, "gain_frac") for s in seeds], dtype=float)
        pk = np.array([series(runs[s], c, "kstar_agree") for s in seeds], dtype=float)
        print(f"    [{c}]")
        print("      pol gain : " + " ".join(f"{x:.3f}" for x in pg.mean(0)))
        print("      pol k*   : " + " ".join(f"{x:.3f}" for x in pk.mean(0)))
        pa = [series(runs[s], c, "plan_acc") for s in seeds]
        if pa[0][0] is not None:
            print("      plan_acc : " + " ".join(
                f"{x:.3f}" for x in np.array(pa, dtype=float).mean(0)))

    print(f"\n  CROSS-WORLD (ballistic at the last graded round; `own` == `base` by "
          f"construction in a static run)")
    wl = [w for w in runs[seeds[0]]["results"][conds[0]]["rounds"][-1].get("xworld", {})]
    print(f"  {'arm':<13}" + "".join(w.rjust(16) for w in wl))
    for c in conds:
        print(f"  {c:<13}" + "".join(
            f"{_ms([endpoint(runs[s], c, 'ballistic_w16', w) for s in seeds])[0]:>16.3f}"
            for w in wl))

    print(f"\n  METERING -- what each arm paid, per run (the currency question)")
    print(f"  {'arm':<13}{'rollout transitions':>22}{'DP state-move evals':>22}"
          f"{'materialisations':>20}")
    for c in conds:
        p = runs[seeds[0]]["results"][c]["paid"]
        print(f"  {c:<13}{p['rollout_transitions']:>22,}{p['dp_state_move_evals']:>22,}"
              f"{p['materialisations']:>20,}")


def _spearman(x, y):
    """Rank correlation, no scipy in this env."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def panel_readouts(runs, conds):
    """PANEL 4 -- the readout bake-off.

    PR anti-correlated with target fidelity in the first pass, so the question is whether ANY
    representation-side number tracks function. The ladder is the calibration: its arms come
    pre-ordered by target fidelity and by three functional readouts that never enter a loss.
    A candidate readout is scored by rank-correlating it, ACROSS ARMS, with each functional
    readout -- per seed, so a correlation that only exists on the cross-seed mean is visible as
    such. `R_act` is PR under `residual_decomposition`'s name and is the incumbent.
    """
    seeds = sorted(runs)
    if "frontier" not in runs[seeds[0]]["results"][conds[0]]["rounds"][-1]:
        print(f"\n{'=' * 108}\n=== PANEL 4 -- readout bake-off: not available (run predates "
              f"`frontier_probe`) ===\n{'=' * 108}")
        return
    print(f"\n{'=' * 108}\n=== PANEL 4 -- READOUT BAKE-OFF: does any representation-side number "
          f"track function? ===\n{'=' * 108}")

    def cand(d, c, view):
        r = [x for x in d["results"][c]["rounds"] if "frontier" in x][-1]["frontier"][view]
        return {"R_act": r["repaired"]["R_act_pr"],
                "R_res_participation": r["repaired"]["R_res_participation"],
                "R_comp_participation": r["repaired"]["R_comp_participation"],
                "frontier_mass": r["repaired"]["frontier_mass"],
                "naive_R_res": r["naive"].get("R_res"),
                "rel_residual": r["basic"]["relative_residual"],
                "beta": r["beta"], "beta_r2": r["beta_r2"],
                "alignment_index": r["alignment_index"]}

    # THE DECISIVE TEST, and a correction to the obvious one. Rank-correlating across ALL seven
    # arms is dominated by the frozen/dense floor -- those arms have both low PR and low
    # function, which manufactures a positive correlation that HIDES the inversion. The
    # inversion lives among the four arms that carry a target, which is where fidelity actually
    # varies. Restricting to them is what makes the ladder a calibration rather than a
    # floor-vs-ceiling contrast.
    TARGETS = [c for c in ("evaluative", "dp_noised_g", "dp_noised_a", "critic") if c in conds]
    if len(TARGETS) == 4:
        print(f"\n  DECISIVE -- restricted to the 4 target arms (fidelity varies; the floor does"
              f" not drown the signal)")
        print(f"  {'readout':<22}" + "".join(f"{t:>24}" for t in
                                             ("vs ballistic", "vs d4", "vs target fidelity")))
        fid = {(s, c): np.mean(series(runs[s], c, "target_gain")) for s in seeds
               for c in TARGETS}
        keys = list(cand(runs[seeds[0]], TARGETS[0], "tree")) + ["belief_PR"]
        for k in keys:
            cells = []
            for tgt in ("ballistic_w16", "d4", "fidelity"):
                rr = []
                for s in seeds:
                    xs = ([series(runs[s], c, "PR")[-1] for c in TARGETS] if k == "belief_PR"
                          else [cand(runs[s], c, "tree")[k] for c in TARGETS])
                    ys = ([fid[(s, c)] for c in TARGETS] if tgt == "fidelity"
                          else [endpoint(runs[s], c, tgt) for c in TARGETS])
                    rr.append(_spearman(xs, ys))
                cells.append(f"{np.mean(rr):+.2f} [" + " ".join(f"{v:+.1f}" for v in rr) + "]")
            print(f"  {k:<22}" + "".join(c.rjust(24) for c in cells))
        print(f"\n  ordering of the 4 target arms, best-function FIRST")
        f_ball = lambda c: -np.mean([endpoint(runs[s], c, "ballistic_w16") for s in seeds])
        print(f"    by BALLISTIC (truth)  : {' > '.join(sorted(TARGETS, key=f_ball))}")
        for k in ("belief_PR", "R_act", "R_res_participation", "frontier_mass"):
            g = ((lambda c: -np.mean([series(runs[s], c, "PR")[-1] for s in seeds]))
                 if k == "belief_PR"
                 else (lambda c: -np.mean([cand(runs[s], c, "tree")[k] for s in seeds])))
            print(f"    by {k:<19}: {' > '.join(sorted(TARGETS, key=g))}")
        print(f"\n  is R_res_participation independent of R_act, or a scaled copy?")
        for c in conds:
            ra = np.mean([cand(runs[s], c, "tree")["R_act"] for s in seeds])
            rp = np.mean([cand(runs[s], c, "tree")["R_res_participation"] for s in seeds])
            print(f"    {c:<13} R_act={ra:6.2f}  R_res_part={rp:6.2f}  ratio={rp / ra:.3f}")

    for view in ("tree", "all"):
        keys = list(cand(runs[seeds[0]], conds[0], view))
        print(f"\n  [{view} blocks]  absolute values at the last graded round (mean over seeds)")
        print(f"  {'arm':<13}" + "".join(k[:13].rjust(15) for k in keys) + "   belief-PR")
        for c in conds:
            vals = [np.mean([cand(runs[s], c, view)[k] for s in seeds]) for k in keys]
            bpr = np.mean([series(runs[s], c, "PR")[-1] for s in seeds])
            print(f"  {c:<13}" + "".join(f"{v:>15.3f}" for v in vals) + f"{bpr:>12.2f}")

        print(f"\n  [{view} blocks]  rank-correlation ACROSS ARMS with the functional readouts"
              f"  (per seed; want |rho| high and POSITIVE)")
        print(f"  {'readout':<22}" + "".join(f"{t:>26}" for t in
                                             ("vs ballistic", "vs d4", "vs freshTop1")))
        for k in keys + ["belief_PR"]:
            cells = []
            for tgt in ("ballistic_w16", "d4", "top1"):
                rr = []
                for s in seeds:
                    xs = ([series(runs[s], c, "PR")[-1] for c in conds] if k == "belief_PR"
                          else [cand(runs[s], c, view)[k] for c in conds])
                    ys = [endpoint(runs[s], c, tgt) for c in conds]
                    rr.append(_spearman(xs, ys))
                cells.append(f"{np.mean(rr):+.2f} [" + " ".join(f"{v:+.2f}" for v in rr) + "]")
            print(f"  {k:<22}" + "".join(c.rjust(26) for c in cells))
    print("\n  reading: the incumbent `belief_PR` / `R_act` is expected NEGATIVE (the first pass"
          "\n  found dPR anti-correlates with target fidelity). A candidate is only an improvement"
          "\n  if it is positive and consistent ACROSS SEEDS -- `R_res_participation` is on record"
          "\n  as non-monotone in model quality, so a sign flip between seeds is the failure mode"
          "\n  to watch for, not a surprise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    runs = load(args.figures)
    if not runs:
        raise SystemExit(f"no non-quick results under {args.figures}")
    conds = [c for c in ORDER if c in runs[sorted(runs)[0]]["results"]]
    panel_expansion(runs, conds)
    panel_bracket(runs, conds)
    panel_wirehead(runs, conds)
    panel_readouts(runs, conds)


if __name__ == "__main__":
    main()
