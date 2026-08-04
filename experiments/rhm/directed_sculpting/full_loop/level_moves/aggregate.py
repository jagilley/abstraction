"""Aggregate the level-indexed action-space probe across seeds.

Prints the four things the experiment exists to decide, in the order they have to be read:

  1. THE RAW PROFILE -- |ΔV| by level. This is the number the naive question asks for ("higher
     levels explain more, so the value should rate them higher"), and it is the one that cannot
     be read on its own.
  2. THE SPAN NULL -- |ΔV| by level in the DISTRACTOR channels. Their ground-truth Δd* is exactly
     0.000 at EVERY level (P1), so any rise with level there is the branching factor alone: a
     level-`ell` move rewrites `s**ell` tokens and perturbs the latent more whether or not it
     buys anything. This is the control the raw profile must be read against, and it costs
     nothing because the channels are already in the action space.
  3. WHAT THE LEVEL ACTUALLY BUYS -- best Δd* by level from the exact DP, and the DP's own top-1
     distribution over levels. If deep moves buy nothing, a value that rates them low is right.
  4. THE CALIBRATED ANSWER -- `residual`, mean ΔV after regressing ΔV on the move's true Δd*
     over all tree moves pooled. Positive = the value OVER-rates that level for what it buys.
     Reported with a per-seed slope against level, so "the value systematically over-rates depth"
     is a testable claim rather than an eyeballed one.

Run from experiments/:
  python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py
  python3 rhm/directed_sculpting/full_loop/level_moves/aggregate.py --figures <dir>
"""

import argparse
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def _mean_sd(xs):
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size)


def _slope(xs, ys):
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    ok = np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else float("nan")


def _t_stat(xs):
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size)))


def load(figdir, pattern="level_*"):
    """{arm: {seed: result}} from matching result files, skipping --quick smokes.

    Keyed by seed, so DIFFERENT sweeps must not be aggregated together -- `level_lv_s1` and
    `level_null4_s1` are the same seed on different layouts and would silently overwrite each
    other. Use `--pattern` to aggregate one sweep at a time; the layout of whatever was read is
    echoed by the caller so a mixed directory cannot be mistaken for a single sweep.
    """
    out, cfgs = {}, []
    for path in sorted(glob.glob(os.path.join(figdir, pattern, "results.json"))):
        d = json.load(open(path))
        if d["config"].get("quick"):
            continue
        cfgs.append(d["config"])
        for arm, res in d["arms"].items():
            out.setdefault(arm, {})[d["config"]["seed"]] = res
    return out, cfgs


def report_arm(arm, byseed):
    seeds = sorted(byseed)
    cells = list(byseed[seeds[0]]["cells"])
    tree_levels = sorted(byseed[seeds[0]]["tree_by_level"], key=int)

    def cell(c, field):
        return _mean_sd([byseed[s]["cells"][c][field] for s in seeds])

    print("=" * 104)
    print(f"ARM {arm} -- value trained on the {arm} action space, probed on every level move")
    print(f"       (mean +- sd over seeds {seeds})")
    print("=" * 104)
    print(f"{'cell':14s} {'span':>4s} {'|ΔV|':>16s} {'ΔV':>16s} {'best Δd*':>16s} "
          f"{'residual':>16s} {'top1':>13s} {'DP top1':>8s}")
    for c in cells:
        span = byseed[seeds[0]]["cells"][c]["span_blocks"]
        row = [cell(c, f) for f in ("abs_dvalue", "mean_dvalue", "best_dstar_gain", "residual")]
        t1, t1sd, _ = cell(c, "top1_share")
        dp1, _, _ = cell(c, "dp_top1_share")
        print(f"{c:14s} {span:>4d} " + " ".join(f"{m:>+10.4f}+-{sd:.4f}" for m, sd, _ in row)
              + f" {t1:>+7.3f}+-{t1sd:.3f} {dp1:>8.3f}")

    # (1) the raw profile, and (2) the span null it has to be read against
    print(f"\n  (1) RAW |ΔV| BY LEVEL on the tree, and (2) THE SPAN NULL: the same profile in the")
    print(f"      distractor channels, whose ground-truth Δd* is exactly 0.000 at every level.")
    print(f"      A rise in the null column is the branching factor, not relevance.")
    # committed distractor cells only -- a lazy twin is a control, not a member of the null average
    null_cells = [c for c in cells
                  if not c.startswith("tree|") and not byseed[seeds[0]]["cells"][c].get("lazy")]
    print(f"{'level':>5s} {'tree |ΔV|':>18s} {'distractor |ΔV| (null)':>26s} {'ratio':>8s}")
    for ell in tree_levels:
        tm, tsd, _ = _mean_sd([byseed[s]["tree_by_level"][ell]["abs_dvalue"] for s in seeds])
        same = [c for c in null_cells if byseed[seeds[0]]["cells"][c]["level"] == int(ell)]
        if same:
            nm, nsd, _ = _mean_sd([np.mean([byseed[s]["cells"][c]["abs_dvalue"] for c in same])
                                   for s in seeds])
            print(f"{ell:>5s} {tm:>11.4f}+-{tsd:.4f} {nm:>19.4f}+-{nsd:.4f} {tm / nm:>8.1f}x")
        else:
            print(f"{ell:>5s} {tm:>11.4f}+-{tsd:.4f} {'(no distractor at this level)':>26s}")

    # (3) what the level actually buys, and (4) the calibrated answer
    print(f"\n  (3) WHAT THE LEVEL BUYS and (4) THE CALIBRATED ANSWER (tree moves only).")
    print(f"      residual > 0 means the value OVER-rates that level for what it actually buys.")
    print(f"{'level':>5s} {'best Δd*':>18s} {'value top1':>18s} {'DP top1':>18s} "
          f"{'residual':>18s}")
    for ell in tree_levels:
        row = [_mean_sd([byseed[s]["tree_by_level"][ell][f] for s in seeds])
               for f in ("best_dstar_gain", "top1_share", "dp_top1_share", "residual")]
        print(f"{ell:>5s} " + " ".join(f"{m:>+11.4f}+-{sd:.4f}" for m, sd, _ in row))

    lv = [int(x) for x in tree_levels]
    print(f"\n  per-seed slope against LEVEL (per level of the hierarchy):")
    for field, label in (("abs_dvalue", "raw |ΔV|"),
                         ("best_dstar_gain", "best Δd* (what it buys)"),
                         ("top1_share", "value top-1 share"),
                         ("dp_top1_share", "DP top-1 share"),
                         ("residual", "residual (calibrated)")):
        sl = [_slope(lv, [byseed[s]["tree_by_level"][e][field] for e in tree_levels])
              for s in seeds]
        m, sd, n = _mean_sd(sl)
        print(f"    {label:26s} {m:>+9.5f} +- {sd:.5f}  (n={n}, t={_t_stat(sl):+.2f}, "
              f"per-seed " + " ".join(f"{x:+.4f}" for x in sl) + ")")

    rc, rcsd, _ = _mean_sd([byseed[s]["value_vs_dp_rank_corr"] for s in seeds])
    tt, ttsd, _ = _mean_sd([byseed[s]["tree_top1_share"] for s in seeds])
    dt, _, _ = _mean_sd([byseed[s]["dp_tree_top1_share"] for s in seeds])
    ts, tssd, _ = _mean_sd([byseed[s].get("terminal_success") for s in seeds])
    sl, _, _ = _mean_sd([byseed[s]["calibration"]["slope"] for s in seeds])
    print(f"\n  value-vs-DP rank corr over ALL moves {rc:+.3f}+-{rcsd:.3f} | top-1 is a tree move: "
          f"value {tt:.3f}+-{ttsd:.3f} vs DP {dt:.3f}")
    print(f"  behaviour-policy terminal success {ts:.3f}+-{tssd:.3f} | calibration slope "
          f"ΔV per unit Δd* = {sl:+.4f}\n")


def report_matched_span(arm, byseed):
    """(5) THE MATCHED-SPAN CONTROL -- abstraction or span?

    Each committed level-ℓ move is paired with a LAZY twin at the same node: identical tokens
    rewritten, so the branching-factor/pooling displacement cancels INSIDE the pair. The only
    remaining difference is whether the span is one legal abstract commitment or `s**(ell-1)`
    independent level-1 guesses. `pref` is the fraction of (node, state) pairs where the value
    ranks the commitment above the guesses; 0.5 is indifference, and the confound is removed by
    construction rather than by regression.

    The DISTRACTOR rows are a second, harder control and the one that decides the reading: there a
    commitment is worth exactly nothing to the task (Δd* is 0.000 either way), so a value that
    still prefers it is responding to GRAMMATICALITY -- the legal span sits on the manifold the
    encoder was trained on -- and not to task value. Tree preference ABOVE distractor preference
    at matched level is the only pattern that supports the headline.
    """
    seeds = sorted(byseed)
    if not byseed[seeds[0]].get("matched_span"):
        return
    keys = list(byseed[seeds[0]]["matched_span"])
    print(f"  (5) THE MATCHED-SPAN CONTROL on arm {arm}: committed level-ℓ move vs its LAZY twin")
    print(f"      at the same node. Identical tokens rewritten, so span/pooling cancels in-pair.")
    print(f"      `pref` = P(value ranks the commitment above the guesses); 0.5 = indifferent.")
    print(f"      Distractor rows are the GRAMMATICALITY null: a commitment there buys Δd* = 0.")
    print(f"      `d*` is integer-valued so many pairs TIE on ground truth; the UNTIED columns put")
    print(f"      the value's rate and the DP's on the same subset, which is the fair comparison.")
    print(f"{'cell':13s} {'span':>4s} {'ΔV premium':>17s} {'Δd* premium':>17s} {'DPtie':>6s} | "
          f"{'value(untied)':>17s} {'DP(untied)':>17s} {'agree':>15s}")
    for k in keys:
        c0 = byseed[seeds[0]]["matched_span"][k]
        row = [_mean_sd([byseed[s]["matched_span"][k][f] for s in seeds])
               for f in ("value_premium", "dp_premium")]
        tie, _, _ = _mean_sd([byseed[s]["matched_span"][k]["tie_rate_dp"] for s in seeds])
        line = (f"{k:13s} {c0['span_blocks']:>4d} "
                + " ".join(f"{m:>+11.4f}+-{sd:.4f}" for m, sd, _ in row) + f" {tie:>6.3f} | ")
        if c0.get("pref_rate_untied") is None:
            print(line + "   (all pairs tied on d*)")
            continue
        u = [_mean_sd([byseed[s]["matched_span"][k][f] for s in seeds])
             for f in ("pref_rate_untied", "dp_pref_rate_untied", "agree_untied")]
        print(line + " ".join(f"{m:>10.3f}+-{sd:.3f}" for m, sd, _ in u[:2])
              + f" {u[2][0]:>8.3f}+-{u[2][1]:.3f}")

    tree_keys = [k for k in keys if k.startswith("tree|")]
    null_keys = [k for k in keys if not k.startswith("tree|")]
    if tree_keys:
        lv = [byseed[seeds[0]]["matched_span"][k]["level"] for k in tree_keys]
        for field, label in (("pref_rate_untied", "tree pref slope vs level (untied)"),
                             ("pref_rate", "tree pref slope vs level (all pairs)")):
            sl = [_slope(lv, [byseed[s]["matched_span"][k][field] for k in tree_keys])
                  for s in seeds]
            m, sd, n = _mean_sd(sl)
            print(f"\n    {label:36s} {m:>+9.5f} +- {sd:.5f} (n={n}, t={_t_stat(sl):+.2f}, "
                  f"per-seed " + " ".join(f"{x:+.4f}" for x in sl) + ")")
    # The grammaticality null. In a distractor EVERY pair ties on d*, so there is no untied rate to
    # compare -- and a sign rate on a distribution centred at ~0 is a skew statistic, not a
    # preference. The honest contrast is therefore the PREMIUM's magnitude: how much value does a
    # legal commitment carry where legality buys the task exactly nothing?
    for ell in sorted({byseed[seeds[0]]["matched_span"][k]["level"] for k in null_keys}):
        tk = [k for k in tree_keys if byseed[seeds[0]]["matched_span"][k]["level"] == ell]
        nk = [k for k in null_keys if byseed[seeds[0]]["matched_span"][k]["level"] == ell]
        if not (tk and nk):
            continue
        tm, tsd, _ = _mean_sd([np.mean([byseed[s]["matched_span"][k]["value_premium"] for k in tk])
                               for s in seeds])
        nm, nsd, _ = _mean_sd([np.mean([byseed[s]["matched_span"][k]["value_premium"] for k in nk])
                               for s in seeds])
        print(f"    L{ell} GRAMMATICALITY NULL: ΔV premium tree {tm:+.4f}+-{tsd:.4f} vs "
              f"distractor {nm:+.4f}+-{nsd:.4f}  ({abs(tm / nm) if nm else float('nan'):.0f}x)")
        print(f"       -> a legal commitment where legality buys Δd*=0 carries "
              f"{abs(nm / tm) * 100 if tm else float('nan'):.1f}% of the tree's premium")
    print()


def compare_arms(data):
    """`level` minus `flat`, paired by seed: does giving the value the axis change what it does?"""
    if not {"flat", "level"} <= set(data):
        return
    seeds = sorted(set(data["flat"]) & set(data["level"]))
    if not seeds:
        return
    tree_levels = sorted(data["flat"][seeds[0]]["tree_by_level"], key=int)
    print("=" * 104)
    print("LEVEL minus FLAT, paired by seed -- the effect of putting the level axis in the "
          "action space the")
    print("value was TRAINED on. Both arms are probed on the identical move set and the "
          "identical states.")
    print("=" * 104)
    print(f"{'level':>5s} {'Δ raw |ΔV|':>18s} {'Δ value top1':>18s} {'Δ residual':>18s}")
    for ell in tree_levels:
        row = []
        for f in ("abs_dvalue", "top1_share", "residual"):
            row.append(_mean_sd([data["level"][s]["tree_by_level"][ell][f]
                                 - data["flat"][s]["tree_by_level"][ell][f] for s in seeds]))
        print(f"{ell:>5s} " + " ".join(f"{m:>+11.4f}+-{sd:.4f}" for m, sd, _ in row))
    for f, label in (("value_vs_dp_rank_corr", "value-vs-DP rank corr"),
                     ("tree_top1_share", "top-1 is a tree move"),
                     ("terminal_success", "behaviour terminal success")):
        d = [data["level"][s].get(f, np.nan) - data["flat"][s].get(f, np.nan) for s in seeds]
        m, sd, n = _mean_sd(d)
        print(f"  Δ {label:28s} {m:>+9.4f} +- {sd:.4f}  (n={n}, t={_t_stat(d):+.2f})")
    print()


def report_damage_sweep(figdir, patterns=("level_dmg*", "level_lz2_*")):
    """The damage-level sweep: does the value's abstraction premium track WHERE the error lives?

    Read this table as the answer to "is the value sensitive to the hierarchy". The DGP knob is
    `damage_level` -- the depth at which the error is planted, with every block left on-grammar so
    the error is invisible block-locally (see `certify_damage`). A move at level ell can only
    repair an error at level <= ell, so the ORACLE's abstraction premium at L2 must collapse as
    the damage moves to level 3, while L3's holds. The question is whether the value follows.

    L4 IS EXCLUDED AND MUST BE. A root move masks every tree token, so the generator sees only the
    (undamaged, identically seeded) non-tree context and emits the same span whatever the damage
    is; `d_cur` then cancels out of the premium. Measured: the L4 Δd* premium is bit-identical
    across all three damage levels. The root cell is structurally blind to this knob, which is
    also why it never resolved -- it needs a partial-mask deep move, not more seeds.
    """
    runs = {}
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(figdir, pat, "results.json"))):
            d = json.load(open(path))
            if d["config"].get("quick") or "flat" not in d["arms"]:
                continue
            dl = d["config"].get("damage_level", 0)
            runs.setdefault(dl, {})[d["config"]["seed"]] = d
    if not runs:
        return
    print("=" * 104)
    print("THE DAMAGE-LEVEL SWEEP -- does the abstraction premium track where the error lives?")
    print("  damage 0 = published random-symbol damage (off-grammar, block-locally visible)")
    print("  damage k>=1 = a level-k subtree swapped for a legal derivation of a feature it")
    print("                cannot produce: 100% on-grammar, so only a level->=k commitment sees it")
    print("  L4 excluded: a root move masks the whole tree, so its premium cannot respond (see doc)")
    print("=" * 104)
    print(f"{'damage':>6s} {'seeds':>6s} {'on-gram':>8s} {'d*':>6s} {'rankcorr':>16s} "
          f"{'L2 DP':>15s} {'L2 value':>15s} {'L3 DP':>15s} {'L3 value':>15s}")
    for dl in sorted(runs):
        seeds = sorted(runs[dl])
        byseed = {s: runs[dl][s]["arms"]["flat"] for s in seeds}

        def ms(level, field):
            return _mean_sd([byseed[s]["matched_span"].get(f"tree|L{level}", {}).get(field)
                             for s in seeds])
        og = _mean_sd([runs[dl][s].get("damage_certification", {}).get("on_grammar")
                       for s in seeds])[0]
        ds = _mean_sd([byseed[s]["dstar_mean"] for s in seeds])[0]
        rc = _mean_sd([byseed[s]["value_vs_dp_rank_corr"] for s in seeds])
        cols = [ms(2, "dp_pref_rate_untied"), ms(2, "pref_rate_untied"),
                ms(3, "dp_pref_rate_untied"), ms(3, "pref_rate_untied")]
        print(f"{dl:>6d} {len(seeds):>6d} {og:>8.3f} {ds:>6.2f} "
              f"{rc[0]:>+9.3f}+-{rc[1]:.3f} "
              + " ".join(f"{m:>9.3f}+-{sd:.3f}" for m, sd, _ in cols))

    # the decisive within-level contrast: does the ORACLE's premium at L2 fall as damage deepens,
    # and does the value's fall with it?
    print(f"\n  tracking error |value - DP| on the untied subset, per cell:")
    for dl in sorted(runs):
        seeds = sorted(runs[dl])
        byseed = {s: runs[dl][s]["arms"]["flat"] for s in seeds}
        errs = []
        for level in (2, 3):
            d = [byseed[s]["matched_span"][f"tree|L{level}"]["pref_rate_untied"]
                 - byseed[s]["matched_span"][f"tree|L{level}"]["dp_pref_rate_untied"]
                 for s in seeds if f"tree|L{level}" in byseed[s]["matched_span"]]
            m, sd, n = _mean_sd(d)
            errs.append(f"L{level} {m:+.3f}+-{sd:.3f}")
        print(f"    damage {dl}: " + "   ".join(errs))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    ap.add_argument("--pattern", default="level_*",
                    help="glob for ONE sweep, e.g. level_lv_* / level_null4_* / level_lz_*. "
                         "Results are keyed by seed, so mixing sweeps overwrites cells.")
    ap.add_argument("--damage-sweep", action="store_true",
                    help="print the damage-level x move-level table instead (groups by "
                         "damage_level, so it may span sweeps safely)")
    args = ap.parse_args()
    if args.damage_sweep:
        report_damage_sweep(args.figures)
        return
    print(f"reading {args.figures} [{args.pattern}]\n")
    data, cfgs = load(args.figures, args.pattern)
    if not data:
        print(f"(no non-quick {args.pattern} results found)")
        return
    layouts = sorted({(c["struct_depths"], c["struct_ms"], bool(c.get("lazy_twins")))
                      for c in cfgs})
    for sd, sm, lz in layouts:
        print(f"  layout struct_depths={sd} struct_ms={sm} lazy_twins={lz}")
    if len(layouts) > 1:
        print("  !! MORE THAN ONE LAYOUT MATCHED -- seeds collide across sweeps. Narrow --pattern.")
    print()
    for arm in sorted(data, key=lambda a: a != "flat"):     # the published action space first
        report_arm(arm, data[arm])
        report_matched_span(arm, data[arm])
    compare_arms(data)


if __name__ == "__main__":
    main()
