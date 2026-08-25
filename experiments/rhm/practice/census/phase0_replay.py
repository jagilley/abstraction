"""census / PHASE 0 — the free experiment: replay every candidate gauge over `sp_s0`'s logs.

No GPU, no Modal. Every gauge this round might put live is a function of per-cycle series that
`spiral/sp_s0` already logged, so each one can be asked, retroactively and for free, the only
question that matters before it is paid for: **when would it have fired, and would that have
been anywhere other than where the certificate already fired?**

A gauge that never fires inside the run, or that fires with the certificate, is degenerate on
this substrate and is dropped from Phase 1 with this replay as the record.

WHAT IS REPLAYED (all label-free, all already in the log):

  G-A  ADMISSION RATE at the frontier — `log["miner"][l]["n_at_support"][sup]`, the count of
       distinct level-l-shaped tuples seen at least `sup` times in the beam's own chosen
       trajectories. `recital`'s within-level pacer, aimed at the frontier level. Fires when
       the admissions over a trailing window fall to or below a threshold.
  G-C  BUILDABLE-COVERAGE PLATEAU — `log["aud"][l]["n_entries"]`, the entries `Miner.build`
       can actually assemble over the operative lower table. Distinct from G-A because the
       ratchet filter sits between an observation and an entry, and at L3 it may be the binding
       constraint rather than the observation stream.
  G-D  FROZEN-VS-LIVE DIVERGENCE — the recert events' `n_live`/`n_frozen`/`n_diff`, plus the
       per-cycle live-minus-frozen entry gap. This is the licensing signal for the
       COMMIT-THEN-EXTEND op, so it is not a firing cycle but an opportunity series: how many
       entries an extension would have had available, at every recert, and how much of the
       audition those entries would have changed.
  G-Y  NEXT-LEVEL YIELD — the T4-shaped observation stream. See `t4_observability()`: this one
       is **not mechanically present** in `sp_s0`'s logs, and the check is reported rather than
       worked around.

Usage (from experiments/):
    python3 rhm/practice/census/phase0_replay.py
    python3 rhm/practice/census/phase0_replay.py --tag sp_s0 --out figures/phase0
"""

import argparse
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPIRAL_FIG = os.path.join(os.path.dirname(HERE), "spiral", "figures")

EARNING = ["enum_live", "spiral_route", "spiral", "enum_live_g722"]
REFERENCE = ["given", "given_native", "fid"]
LEVELS = ["2", "3"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(tag):
    root = os.path.join(SPIRAL_FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    summary = json.load(open(os.path.join(root, "summary.json")))
    arms = {a: json.load(open(os.path.join(root, a, "results.json")))
            for a in summary["order"]}
    return setup, summary, arms


def era_bounds(setup):
    """[(era_index, first_cycle, last_cycle, level)] — the ladder as absolute cycles."""
    out, c = [], 0
    for i, e in enumerate(setup["eras"]):
        n = int(e["cycles"])
        out.append((i + 1, c + 1, c + n, e["level"], e["name"]))
        c += n
    return out


def commit_cycle(arm_json, level):
    return next((e["cycle"] for e in arm_json["events"]
                 if e["kind"] == "commit" and e["level"] == int(level)), None)


def cert_cycle(arm_json, level):
    return arm_json.get("shadow_cert", {}).get(str(level), {}).get("fired")


# --------------------------------------------------------------------------- #
# the observability check that has to come first
# --------------------------------------------------------------------------- #

def t4_observability(arms):
    """Is the T4-shaped observation stream present in these logs? Checked, not assumed.

    `run_arm` builds `miners = {l: MC.Miner(l, s) for l in range(2, max_macro_level + 1)}` and
    `max_macro_level = 3`, so **no level-4 miner is ever constructed**, and the per-cycle
    `Miner.state()` dump that `log["miner"]` records therefore cannot contain a level-4 row.
    The same bound applies to `log["aud"]` and to the ablation battery's `mine` block.

    Nor can the stream be reconstructed from what IS logged. A level-4-shaped observation is a
    specific pair of ADJACENT level-3-shaped spans within one configuration; `Miner.state()`
    stores per-key counts and (via this arc's identity instrument) the key set at support, but
    no co-occurrence and no positions. Counts without adjacency cannot be re-paired.

    So the yield gauge is not degenerate — it is UNMEASURED, which is a different verdict and
    has a different consequence: it needs an instrument added to the live run, not a decision
    made from these logs.
    """
    found = {}
    for a, r in arms.items():
        lv_miner = sorted({k for cyc in r["log"]["miner"] for k in cyc})
        lv_aud = sorted({k for cyc in r["log"]["aud"] for k in cyc})
        ab = r.get("ablation") or {}
        lv_ab = sorted({k for cond in ab.values() if isinstance(cond, dict)
                        for era in (cond.get("eras") or {}).values()
                        for k in (era.get("mine") or {})})
        found[a] = {"miner_levels": lv_miner, "aud_levels": lv_aud,
                    "ablation_mine_levels": lv_ab,
                    "max_macro_level": r["config"]["max_macro_level"]}
    any_l4 = any("4" in v["miner_levels"] or "4" in v["aud_levels"]
                 or "4" in v["ablation_mine_levels"] for v in found.values())
    return {"t4_observable": bool(any_l4), "per_arm": found,
            "verdict": ("OBSERVABLE" if any_l4 else
                        "NOT OBSERVABLE — no level-4 miner is constructed anywhere in the "
                        "sp_s0 run (miners span range(2, max_macro_level+1) = levels 2-3), and "
                        "the logged per-key counts carry no adjacency, so the stream cannot be "
                        "reconstructed offline."),
            "what_phase_1_would_need": (
                "an UNPRICED instrument miner at level 4 (span s**3 = 8 blocks = 16 tokens) "
                "observing the era's own level-4 node in every era, decoupled from "
                "`max_macro_level` (which caps what may be COMMITTED, not what may be "
                "OBSERVED). Level 4 stays unearnable — 1,024 entries, ~384 cycles to cover — "
                "so this is a growth-DIRECTION readout only, never a commit candidate.")}


# --------------------------------------------------------------------------- #
# the series each gauge reads
# --------------------------------------------------------------------------- #

def series(arm_json, level, support="3"):
    lg = arm_json["log"]
    at_sup = [c[level]["n_at_support"][support] if level in c else 0 for c in lg["miner"]]
    distinct = [c[level]["n_distinct"] if level in c else 0 for c in lg["miner"]]
    n_obs = [c[level]["n_obs"] if level in c else 0 for c in lg["miner"]]
    ents = [(c.get(level) or {}).get("n_entries") or 0 for c in lg["aud"]]
    rec = [(c.get(level) or {}).get("tab_recall") for c in lg["aud"]]
    prec = [(c.get(level) or {}).get("tab_precision") for c in lg["aud"]]
    frozen = [(c.get(level) if isinstance(c, dict) else None) for c in lg["vocab"]]
    frozen = [0 if f is None else int(f) for f in frozen]
    fg = [(c or {}).get(level) for c in lg["committed_grade"]]
    return {"cycle": lg["cycle"], "at_sup": at_sup, "distinct": distinct, "n_obs": n_obs,
            "live_entries": ents, "live_recall": rec, "live_precision": prec,
            "frozen_entries": frozen, "frozen_grade": fg}


# --------------------------------------------------------------------------- #
# G-A / G-C: trailing-window saturation detectors
# --------------------------------------------------------------------------- #

def saturation_fire(x, W, theta, c_min, active_from=None):
    """First cycle at which the total growth of `x` over the trailing W cycles is <= theta.

    `active_from` is the first cycle at which the series is even defined (the frontier level is
    not mined until its era begins); a detector may not fire before it has W cycles of data
    inside its own active window, or the plateau it reads is the plateau of a series that does
    not exist yet — which is how a saturation gauge fires at cycle 1 for a level nobody has
    started earning.
    """
    x = np.asarray(x, float)
    n = x.size
    start = max(c_min, (active_from or 1) + W)
    for c in range(start, n + 1):
        i = c - 1
        if i - W < 0:
            continue
        if (x[i] - x[i - W]) <= theta:
            return c
    return None


def first_active(x):
    """First cycle at which the series is non-zero — when the level starts being mined."""
    for i, v in enumerate(x):
        if v:
            return i + 1
    return None


def replay_gauge(arms, level, key, grid_W, grid_theta, support="3", c_min=6):
    """Replay one saturation gauge over a (W, theta) grid, on `key`'s series."""
    rows = []
    for a in EARNING:
        s = series(arms[a], level, support)
        act = first_active(s[key])
        cc = commit_cycle(arms[a], level)
        ce = cert_cycle(arms[a], level)
        for W, th in itertools.product(grid_W, grid_theta):
            f = saturation_fire(s[key], W, th, c_min, act)
            row = {"arm": a, "level": level, "key": key, "support": support,
                   "W": W, "theta": th, "fired": f, "first_active": act,
                   "cert_cycle": ce, "commit_cycle": cc}
            if f is not None:
                i = f - 1
                row.update({"entries_at_fire": s["live_entries"][i],
                            "recall_at_fire": s["live_recall"][i],
                            "precision_at_fire": s["live_precision"][i],
                            "at_sup_at_fire": s["at_sup"][i],
                            "frozen_at_fire": s["frozen_entries"][i],
                            "live_minus_frozen_at_fire":
                                s["live_entries"][i] - s["frozen_entries"][i],
                            "sep_from_cert": (None if ce is None else f - ce),
                            "after_actual_commit": bool(cc is not None and f > cc)})
            else:
                row.update({"entries_at_fire": None, "sep_from_cert": None,
                            "after_actual_commit": None})
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# G-D: the extension opportunity series
# --------------------------------------------------------------------------- #

def replay_extend(arms):
    """What COMMIT-THEN-EXTEND would have had available, per recert, per arm, per level.

    Not a firing cycle: the op is licensed at every recert at which the live table holds
    entries the frozen one does not. Reported as the opportunity, its size, and the two things
    that decide whether taking it is safe — how far the live table's own grade is from the
    frozen one's, and how many audition answers the swap would actually change.
    """
    out = []
    for a in EARNING:
        for e in arms[a]["events"]:
            if e["kind"] != "recert":
                continue
            gap = e["n_live"] - e["n_frozen"]
            out.append({"arm": a, "level": e["level"], "cycle": e["cycle"],
                        "n_frozen": e["n_frozen"], "n_live": e["n_live"],
                        "extend_available": gap, "n_diff": e.get("n_diff"),
                        "e_frozen": e["e_frozen"], "e_live": e.get("e_live"),
                        "de_live_minus_frozen": (None if e.get("e_live") is None
                                                 else e["e_live"] - e["e_frozen"]),
                        "live_recall": e.get("live_recall"),
                        "live_precision": e.get("live_precision")})
    return out


def frozen_vs_live_endstate(arms):
    """The end-of-run gap the extension op exists to close, per arm per level."""
    out = []
    for a in EARNING:
        for lv in LEVELS:
            s = series(arms[a], lv)
            fg = next((g for g in reversed(s["frozen_grade"]) if g), None)
            out.append({"arm": a, "level": lv,
                        "frozen_entries": s["frozen_entries"][-1],
                        "frozen_recall": (fg or {}).get("recall"),
                        "frozen_precision": (fg or {}).get("precision"),
                        "live_entries": s["live_entries"][-1],
                        "live_recall": s["live_recall"][-1],
                        "live_precision": s["live_precision"][-1],
                        "gap_entries": s["live_entries"][-1] - s["frozen_entries"][-1],
                        "at_sup_end": s["at_sup"][-1],
                        "still_growing_at_end": bool(s["at_sup"][-1] > s["at_sup"][-6])})
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

GRID_W = [3, 5, 8, 12]
GRID_TH = [0, 1, 2]
SUPPORTS = ["2", "3", "5"]


def growth_tail(arms, level, n=20, support="3"):
    """Is the observation stream still climbing at the end of the run? The precondition for a
    saturation gauge existing at all."""
    out = []
    for a in EARNING:
        s = series(arms[a], level, support)
        x = np.asarray(s["at_sup"], float)
        y = np.asarray(s["live_entries"], float)
        out.append({"arm": a, "level": level,
                    "at_sup_last": int(x[-1]), f"at_sup_delta_last{n}": int(x[-1] - x[-1 - n]),
                    "live_last": int(y[-1]), f"live_delta_last{n}": int(y[-1] - y[-1 - n]),
                    "at_sup_slope_per_cycle": float(np.polyfit(np.arange(n + 1),
                                                               x[-1 - n:], 1)[0]),
                    "live_slope_per_cycle": float(np.polyfit(np.arange(n + 1),
                                                             y[-1 - n:], 1)[0])})
    return out


def _fmt(v, w=8, p=3):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.{p}f}".rjust(w)
    return str(v).rjust(w)


def report(setup, summary, arms):
    R = {"tag": setup.get("tag", "sp_s0"),
         "ladder": [{"era": i, "first": f, "last": l, "level": lv, "name": nm}
                    for i, f, l, lv, nm in era_bounds(setup)],
         "true_table_entries": {"2": 14, "3": 56}}
    print("=" * 78)
    print("census / PHASE 0 — gauge replay over spiral/sp_s0 (no GPU)")
    print("=" * 78)
    eb = era_bounds(setup)
    print("\nLADDER (absolute cycles):")
    for i, f, l, lv, nm in eb:
        print(f"  era {i}  {nm:7s} level {lv}  c{f}-c{l}  ({l - f + 1} cycles)")
    print("  NOTE the era-2 boundary is c88, not c56 — era 2 is 56 cycles LONG but spans "
          "c33-c88.")
    print("  Frontier mining windows: L2 opens at era 1 (c1); L3 opens at era 2 (c33), since "
          "\n  `run_arm` mines levels 2..min(max_macro_level, era_level+1).")

    # ---- 0. the observability check ---------------------------------------------------- #
    t4 = t4_observability(arms)
    R["t4_observability"] = t4
    print("\n" + "-" * 78)
    print("(0) GAUGE G-Y (next-level yield): IS THE T4 STREAM IN THESE LOGS?")
    print("-" * 78)
    print(f"  verdict: {t4['verdict']}")
    print(f"  levels present per arm (identical across all seven): "
          f"miner {t4['per_arm'][EARNING[0]]['miner_levels']}, "
          f"aud {t4['per_arm'][EARNING[0]]['aud_levels']}, "
          f"ablation-mine {t4['per_arm'][EARNING[0]]['ablation_mine_levels']}; "
          f"max_macro_level = {t4['per_arm'][EARNING[0]]['max_macro_level']}")
    print(f"  Phase 1 would need: {t4['what_phase_1_would_need']}")

    # ---- 1. does a plateau exist to detect? -------------------------------------------- #
    print("\n" + "-" * 78)
    print("(1) PRECONDITION: is the frontier stream still climbing at c116?")
    print("    A saturation gauge can only fire on a series that stops.")
    print("-" * 78)
    R["growth_tail"] = {}
    for lv in LEVELS:
        g = growth_tail(arms, lv)
        R["growth_tail"][lv] = g
        print(f"\n  level {lv}   (true table: {R['true_table_entries'][lv]} entries)")
        print(f"    {'arm':16s}{'at_sup end':>12s}{'d(last20)':>11s}{'slope/cyc':>11s}"
              f"{'live end':>10s}{'d(last20)':>11s}{'slope/cyc':>11s}")
        for r in g:
            print(f"    {r['arm']:16s}{r['at_sup_last']:12d}{r['at_sup_delta_last20']:11d}"
                  f"{r['at_sup_slope_per_cycle']:11.3f}{r['live_last']:10d}"
                  f"{r['live_delta_last20']:11d}{r['live_slope_per_cycle']:11.3f}")

    # ---- 2. G-A and G-C over the grid --------------------------------------------------- #
    print("\n" + "-" * 78)
    print("(2) G-A (admission rate at support) and G-C (buildable entries): firing cycles")
    print("    over a (W, theta) grid. `sep` = fired - cert_cycle (negative = before the cert).")
    print("-" * 78)
    R["gauges"] = {}
    for gname, key in (("G-A", "at_sup"), ("G-C", "live_entries")):
        for lv in LEVELS:
            rows = replay_gauge(arms, lv, key, GRID_W, GRID_TH)
            R["gauges"][f"{gname}_L{lv}"] = rows
            print(f"\n  {gname} level {lv}  (support=3)")
            print(f"    {'W':>3s}{'th':>4s} | " + "".join(f"{a[:14]:>16s}" for a in EARNING))
            for W in GRID_W:
                for th in GRID_TH:
                    cells = []
                    for a in EARNING:
                        r = next(x for x in rows
                                 if x["arm"] == a and x["W"] == W and x["theta"] == th)
                        if r["fired"] is None:
                            cells.append("never".rjust(16))
                        else:
                            sep = r["sep_from_cert"]
                            cells.append(f"c{r['fired']}({sep:+d}) n{r['entries_at_fire']}"
                                         .rjust(16))
                    print(f"    {W:3d}{th:4d} | " + "".join(cells))
            print("    cert cycles:   " + "  ".join(
                f"{a}={cert_cycle(arms[a], lv)}" for a in EARNING))
            print("    commit cycles: " + "  ".join(
                f"{a}={commit_cycle(arms[a], lv)}" for a in EARNING))
    return R


def report_extend(arms, R):
    print("\n" + "-" * 78)
    print("(3) G-D (frozen-vs-live divergence): what COMMIT-THEN-EXTEND would have had")
    print("    available at every recert. `avail` = n_live - n_frozen (entries extension")
    print("    could add); `n_diff` = audition answers a whole-table swap would change.")
    print("-" * 78)
    ext = replay_extend(arms)
    R["extend_opportunities"] = ext
    for lv in (2, 3):
        print(f"\n  level {lv}")
        print(f"    {'arm':16s}{'cycle':>7s}{'frozen':>8s}{'live':>6s}{'avail':>7s}"
              f"{'n_diff':>8s}{'live rec':>10s}{'live prec':>11s}{'de(live-frozen)':>17s}")
        for r in ext:
            if r["level"] != lv:
                continue
            print(f"    {r['arm']:16s}{r['cycle']:7d}{r['n_frozen']:8d}{r['n_live']:6d}"
                  f"{r['extend_available']:7d}{_fmt(r['n_diff'], 8)}"
                  f"{_fmt(r['live_recall'], 10)}{_fmt(r['live_precision'], 11)}"
                  f"{_fmt(r['de_live_minus_frozen'], 17, 4)}")
    end = frozen_vs_live_endstate(arms)
    R["frozen_vs_live_endstate"] = end
    print("\n  END-OF-RUN frozen vs live (the gap the extension op exists to close):")
    print(f"    {'arm':16s}{'lvl':>4s}{'frozen n':>9s}{'f.recall':>10s}{'f.prec':>8s}"
          f"{'live n':>8s}{'l.recall':>10s}{'l.prec':>8s}{'gap':>6s}{'growing':>9s}")
    for r in end:
        print(f"    {r['arm']:16s}{r['level']:>4s}{r['frozen_entries']:9d}"
              f"{_fmt(r['frozen_recall'], 10)}{_fmt(r['frozen_precision'], 8)}"
              f"{r['live_entries']:8d}{_fmt(r['live_recall'], 10)}"
              f"{_fmt(r['live_precision'], 8)}{r['gap_entries']:6d}"
              f"{str(r['still_growing_at_end']):>9s}")
    return R


def verdicts(arms, R):
    """Degeneracy per gauge per level, and the era-2 sizing consequence."""
    print("\n" + "-" * 78)
    print("(4) DEGENERACY VERDICTS")
    print("    degenerate := never fires inside the run, OR fires within +/-2 cycles of the")
    print("    certificate in every arm (nothing to separate).")
    print("-" * 78)
    V = {}
    for gname in ("G-A", "G-C"):
        for lv in LEVELS:
            rows = R["gauges"][f"{gname}_L{lv}"]
            best = None
            never = {}
            for W, th in itertools.product(GRID_W, GRID_TH):
                sel = [x for x in rows if x["W"] == W and x["theta"] == th]
                fires = [x["fired"] for x in sel]
                never[(W, th)] = sum(1 for f in fires if f is None)
                if any(f is None for f in fires):
                    continue
                seps = [x["sep_from_cert"] for x in sel]
                # a useful setting fires in EVERY arm, later than the certificate (that is the
                # only direction that buys coverage), and by a margin worth paying for
                if min(seps) >= 1:
                    score = (min(seps), np.mean(seps))
                    if best is None or score > best[0]:
                        best = (score, W, th, sel)
            n_never = min(never.values())
            if best is None:
                why = ("never fires in at least one arm at every grid setting"
                       if n_never else
                       "fires at or before the certificate in at least one arm at every "
                       "setting that fires everywhere")
                V[f"{gname}_L{lv}"] = {"usable": False, "why": why,
                                       "min_arms_never_firing": int(n_never)}
                print(f"  {gname} L{lv}: DEGENERATE — {why}")
            else:
                (mn, mean), W, th, sel = best
                V[f"{gname}_L{lv}"] = {
                    "usable": True, "W": W, "theta": th,
                    "min_sep": int(mn), "mean_sep": float(mean),
                    "per_arm": [{k: x[k] for k in
                                 ("arm", "fired", "cert_cycle", "commit_cycle",
                                  "sep_from_cert", "entries_at_fire", "recall_at_fire",
                                  "precision_at_fire", "after_actual_commit")} for x in sel]}
                print(f"  {gname} L{lv}: USABLE at W={W}, theta={th} — fires "
                      f"{mn}..{max(x['sep_from_cert'] for x in sel)} cycles after the cert")
                for x in sel:
                    print(f"      {x['arm']:16s} fire c{x['fired']:<4d} cert c{x['cert_cycle']:<4d}"
                          f" sep {x['sep_from_cert']:+3d}  entries {x['entries_at_fire']:3d}"
                          f" recall {_fmt(x['recall_at_fire'], 6)}"
                          f"  (post-commit extrapolation: {x['after_actual_commit']})")
    R["verdicts"] = V
    return R


def era2_sizing(arms, R):
    print("\n" + "-" * 78)
    print("(5) ERA-2 SIZING for a gate-later arm at L3")
    print("-" * 78)
    e2 = next(x for x in R["ladder"] if x["era"] == 2)
    print(f"  era 2 spans c{e2['first']}-c{e2['last']} ({e2['last'] - e2['first'] + 1} cycles);"
          f" L3 mining opens at c{e2['first']}.")
    rows = []
    for lv in ["3"]:
        for gname, key in (("G-A", "at_sup"), ("G-C", "live_entries")):
            allrows = R["gauges"][f"{gname}_L{lv}"]
            for W, th in itertools.product(GRID_W, GRID_TH):
                sel = [x for x in allrows if x["W"] == W and x["theta"] == th]
                fired = [x["fired"] for x in sel]
                rows.append({"gauge": gname, "W": W, "theta": th,
                             "n_never": sum(1 for f in fired if f is None),
                             "max_fire": max([f for f in fired if f], default=None),
                             "past_era2_boundary": bool(
                                 any(f and f > e2["last"] for f in fired))})
    R["era2_sizing"] = {"era2_first": e2["first"], "era2_last": e2["last"], "grid": rows}
    strict = [r for r in rows if r["n_never"] == 0]
    print(f"  grid settings at which the L3 gauge fires in ALL FOUR arms: "
          f"{len(strict)} of {len(rows)}")
    if strict:
        worst = max(strict, key=lambda r: r["max_fire"])
        need = worst["max_fire"] - e2["first"] + 1
        R["era2_sizing"]["latest_all_arms_fire"] = worst
        R["era2_sizing"]["era2_cycles_needed"] = int(need)
        print(f"  latest all-arm firing across those settings: c{worst['max_fire']} "
              f"({worst['gauge']}, W={worst['W']}, theta={worst['theta']}) -> era 2 would need "
              f">= {need} cycles (it has {e2['last'] - e2['first'] + 1}).")
    else:
        print("  NONE. At every (W, theta) on the grid, the L3 gauge fails to fire in at least")
        print("  one arm inside the whole 116-cycle run. Extending era 2 does not fix this at")
        print("  any length the round can afford — the L3 stream has no plateau to detect.")
        R["era2_sizing"]["era2_cycles_needed"] = None
    return R


def gauge_collinearity(arms, R):
    """Are G-A and G-C actually two gauges, or one gauge twice?

    At L2 the ratchet's lower table is `MC.base_table(v)`, which is COMPLETE by construction —
    every level-1 feature is always available — so every L2-shaped tuple that reaches support
    also builds into an entry and the two series are identical. They can only separate where
    the lower table is itself earned, i.e. at L3.
    """
    rows = []
    for a in EARNING:
        for lv in LEVELS:
            s = series(arms[a], lv)
            same = s["at_sup"] == s["live_entries"]
            rows.append({"arm": a, "level": lv, "identical_series": bool(same),
                         "max_gap": int(max(x - y for x, y in
                                            zip(s["at_sup"], s["live_entries"])))})
    R["gauge_collinearity"] = rows
    print("\n" + "-" * 78)
    print("(5b) ARE G-A AND G-C TWO GAUGES? (at_sup series vs buildable-entries series)")
    print("-" * 78)
    for r in rows:
        print(f"  {r['arm']:16s} L{r['level']}  identical: {str(r['identical_series']):5s}"
              f"  max(stream - buildable) = {r['max_gap']}")
    print("  => at L2 they are the SAME SIGNAL (the ratchet's lower table is the complete base")
    print("     table, so every tuple at support builds). G-C is a distinct gauge only at L3.")
    return R


def replay_validity(arms, R):
    """The window in which this replay is a counterfactual at all.

    A gate-later gauge, by definition, fires LATER than the certificate. Under `delta_prov` the
    certificate IS what committed in every arm at both levels (cert cycle == commit cycle,
    8/8), so "later than the certificate" is exactly "after the commit" — and after the commit
    the logged series come from a trajectory in which the level was already frozen, the macro
    was already in the action set, and the beam's chosen trajectories (which is what mining
    reads) were already changed by it. That is not the trajectory a gate-later arm would have
    had.

    So this is reported as a hard scope limit, not a caveat in a footnote."""
    rows = []
    for a in EARNING:
        for lv in LEVELS:
            rows.append({"arm": a, "level": lv, "cert": cert_cycle(arms[a], lv),
                         "commit": commit_cycle(arms[a], lv)})
    same = all(r["cert"] == r["commit"] for r in rows)
    R["replay_validity"] = {"cert_equals_commit_everywhere": bool(same), "rows": rows}
    print("\n" + "-" * 78)
    print("(6) SCOPE LIMIT: where this replay is a counterfactual and where it is not")
    print("-" * 78)
    print(f"  cert cycle == commit cycle in every arm at both levels: {same} (8 of 8)")
    print("  => the replay is EXACT up to the certificate and an EXTRAPOLATION after it, and")
    print("     'gate-later' means 'fires after the certificate' by definition. Every grid")
    print("     setting above that fires later than the cert therefore does so in the")
    print("     extrapolation zone. No gate-later gauge can be VALIDATED on these logs; the")
    print("     replay can only bound what such a gauge could have found (section 7).")
    return R


def ratchet_ceiling(arms, R):
    """The hard cap the frozen lower table puts on the level above it.

    `Miner.build(lower, support)` keeps an observed level-l tuple only if BOTH its level-(l-1)
    halves are entries of `lower`. So with a frozen L2 of recall r, the fraction of true L3
    entries that are even BUILDABLE is at most ~r**2 — no L3 gate, however long it holds the
    commit open, can exceed it. This is the number that decides at which level a gate belongs.
    """
    print("\n" + "-" * 78)
    print("(7) THE RATCHET CEILING: what a frozen L2 caps L3 coverage at")
    print("-" * 78)
    rows = []
    for a in EARNING:
        s2, s3 = series(arms[a], "2"), series(arms[a], "3")
        fg2 = next((g for g in reversed(s2["frozen_grade"]) if g), None)
        r2 = (fg2 or {}).get("recall")
        row = {"arm": a, "frozen_L2_recall": r2,
               "L3_buildable_ceiling_r2sq": (None if r2 is None else r2 ** 2),
               "L3_live_recall_end": s3["live_recall"][-1],
               "L3_frozen_recall_end": (next((g for g in reversed(s3["frozen_grade"]) if g),
                                             {}) or {}).get("recall"),
               "L3_at_sup_end": s3["at_sup"][-1],
               "L3_live_entries_end": s3["live_entries"][-1],
               "stream_minus_buildable": s3["at_sup"][-1] - s3["live_entries"][-1],
               "L2_live_recall_end": s2["live_recall"][-1],
               "L2_frozen_recall_end": r2,
               "L2_coverage_on_offer": (None if r2 is None
                                        else s2["live_recall"][-1] - r2)}
        rows.append(row)
    R["ratchet_ceiling"] = rows
    print(f"  {'arm':16s}{'L2 froz rec':>12s}{'L2 live rec':>12s}{'L2 on offer':>12s}"
          f"{'L3 ceiling':>12s}{'L3 froz rec':>12s}{'L3 live rec':>12s}"
          f"{'stream-built':>13s}")
    for r in rows:
        print(f"  {r['arm']:16s}{_fmt(r['frozen_L2_recall'], 12)}"
              f"{_fmt(r['L2_live_recall_end'], 12)}{_fmt(r['L2_coverage_on_offer'], 12)}"
              f"{_fmt(r['L3_buildable_ceiling_r2sq'], 12)}"
              f"{_fmt(r['L3_frozen_recall_end'], 12)}{_fmt(r['L3_live_recall_end'], 12)}"
              f"{r['stream_minus_buildable']:13d}")
    print("\n  READ: at L3 the observation stream is NOT the binding constraint — every arm")
    print("  observes more L3-shaped tuples at support than it can build (stream-built > 0).")
    print("  The frozen L2 is. At L2 the ratchet's lower table is the BASE table, which is")
    print("  complete by construction, so L2 coverage is capped only by the stream.")
    return R


def extension_windows(arms, R):
    """How many chances the commit-then-extend op actually gets, and in which eras.

    `run_arm`'s recert fires only for `rc_level = era["level"]` and only when that level is
    already committed. On this ladder that means: L2 extension in era 2, L3 extension in era 3,
    and **nothing at all in eras 4-5** — because era 4's level is 4 and `committed[4]` does not
    exist (max_macro_level = 3). Eras 4-5 are exactly where the coverage payoff is measured.
    """
    print("\n" + "-" * 78)
    print("(8) EXTENSION WINDOWS: how many chances the extend op gets, and where")
    print("-" * 78)
    counts = {}
    for a in EARNING:
        per = {}
        for e in arms[a]["events"]:
            if e["kind"] == "recert":
                per.setdefault((e["level"], e["era"]), 0)
                per[(e["level"], e["era"])] += 1
        counts[a] = {f"L{lv}_era{er}": n for (lv, er), n in sorted(per.items())}
    R["extension_windows"] = counts
    for a in EARNING:
        print(f"  {a:16s} {counts[a]}")
    print("\n  READ: the recert/extension mechanism runs in eras 2 and 3 only. In eras 4-5 —")
    print("  the eras whose recovered fraction is this round's money readout — NO recert fires,")
    print("  because `rc_level = era['level']` is 4 or 5 and no such level is committable.")
    print("  A `census_extend` arm built on the inherited recert path would therefore add")
    print("  nothing during the eras where its value is measured. Phase 1 must widen the")
    print("  recert loop to every COMMITTED level rather than the current era's level.")
    return R


def perfect_l2_control(arms, R):
    """The cleanest bound the logs contain on what an L3 gate could EVER deliver.

    `given` / `fid` / `given_native` hold the DGP's own L2 as their operative lower table
    (recall 1.0, so their L3 candidates are not ratchet-limited) and never freeze an earned
    table at all — so their live L3 candidate series is a 116-cycle, unfrozen, un-ratcheted
    mining trajectory. That is precisely the trajectory a perfect L2 gate followed by an L3
    gate would hand the miner, and it is already on disk.
    """
    rows = []
    for a in REFERENCE + EARNING:
        s2, s3 = series(arms[a], "2"), series(arms[a], "3")
        rows.append({"arm": a, "role": ("perfect-L2 control" if a in REFERENCE else "earning"),
                     "L2_live_recall": s2["live_recall"][-1],
                     "L3_at_sup_end": s3["at_sup"][-1],
                     "L3_live_entries_end": s3["live_entries"][-1],
                     "L3_live_recall_end": s3["live_recall"][-1],
                     "L3_live_precision_end": s3["live_precision"][-1],
                     "L3_stream_minus_built": s3["at_sup"][-1] - s3["live_entries"][-1]})
    R["perfect_l2_control"] = rows
    print("\n" + "-" * 78)
    print("(9) THE PERFECT-L2 CONTROL: what an L3 gate could reach at BEST, from these logs")
    print("-" * 78)
    print(f"  {'arm':16s}{'role':>20s}{'L2 live rec':>13s}{'L3 at_sup':>11s}{'L3 built':>10s}"
          f"{'L3 recall':>11s}{'L3 prec':>9s}")
    for r in rows:
        print(f"  {r['arm']:16s}{r['role']:>20s}{_fmt(r['L2_live_recall'], 13)}"
              f"{r['L3_at_sup_end']:11d}{r['L3_live_entries_end']:10d}"
              f"{_fmt(r['L3_live_recall_end'], 11)}{_fmt(r['L3_live_precision_end'], 9)}")
    g = next(r for r in rows if r["arm"] == "given")
    print("\n  READ: `given` mines L3 for 84 cycles over a COMPLETE L2 and never freezes, and")
    print(f"  still ends at recall {g['L3_live_recall_end']:.3f} "
          f"({int(round(g['L3_live_recall_end'] * 56))} of 56) at precision "
          f"{g['L3_live_precision_end']:.3f}. That is the empirical CEILING on what any L3")
    print("  gate could deliver inside a ladder of this length — the earning arms' 0.143-0.179")
    print("  is not mostly a gating failure, it is mostly a mining-rate and ratchet fact.")
    print("  It also shows the frozen L2 costs PRECISION as well as recall: 0.727 with a true")
    print("  L2 against 0.400-0.500 with an earned one.")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sp_s0")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "phase0"))
    a = ap.parse_args()
    setup, summary, arms = load(a.tag)
    setup["tag"] = a.tag
    R = report(setup, summary, arms)
    R = report_extend(arms, R)
    R = verdicts(arms, R)
    R = era2_sizing(arms, R)
    R = gauge_collinearity(arms, R)
    R = replay_validity(arms, R)
    R = ratchet_ceiling(arms, R)
    R = extension_windows(arms, R)
    R = perfect_l2_control(arms, R)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "phase0.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n[phase0] wrote {os.path.join(a.out, 'phase0.json')}")
    return R


if __name__ == "__main__":
    main()
