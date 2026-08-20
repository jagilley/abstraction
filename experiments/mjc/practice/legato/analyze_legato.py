"""Reduce a round-2b run into the arm tables.

Fetches `/data/practice_legato/<tag>/<arm>/results.json` and prints the headline (piece error at
performance tempo x priced time), the per-segment breakdown, the commit record, the fusion
comparison, and the **d_fb x d_plan economics surface** -- which is the round's quantitative claim:
phrase commitment should overtake segment commitment as the sensorimotor delay rises, because it
pays two fewer feedback events per traversal, and the crossing point is the tempo knob made
numerical.

THREE READING RULES.

  * **Window-mean, not final.** Round-0 CAL-D found the closed-loop descent oscillates with an
    amplitude comparable to the whole usable range, and that it is NOT metering noise (the metering
    seed is fixed, so the run-through is deterministic given the model) -- it is a ballistic grader
    amplifying a model bootstrapping on its own on-policy data. Headline = mean over the last
    `--window` probes; the final value is printed beside it.
  * **Ranks are the currency.** Single seed. `recital`'s methodology export: recovery fractions at
    one RNG stream position carry +-0.15; the ordering is the claim.
  * **Two deliberation prices, not one.** `d_plan` per PLAN EVENT is what a nervous system pays to
    interrupt itself; `d_delib` per CEM ROLLOUT-STEP is what it pays to think. A phrase plan is one
    event but three segments' worth of work, so the two prices rank the arms differently and both
    are reported. Round 1 could ignore this because every plan had the same shape.

Usage (from experiments/):
    python3 mjc/practice/legato/analyze_legato.py --tag L1 --fetch
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["never", "seg_plan_launch", "seg_frozen", "phrase_plan_launch", "phrase_frozen",
         "phrase_chain_fixed"]


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_legato/{tag}", dst], check=False,
                   env={**os.environ,
                        "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "chromatic")})


def load(tag):
    root = os.path.join(HERE, "results", tag)
    out = {}
    for base in (os.path.join(root, tag), root):
        if os.path.isdir(base):
            for a in sorted(os.listdir(base)):
                p = os.path.join(base, a, "results.json")
                if os.path.isfile(p):
                    d = json.load(open(p))
                    if isinstance(d, dict) and "arm" in d:     # skip gate worlds (see note above)
                        out[a] = d
    if os.path.isdir(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".json") and not f.startswith("launch"):
                d = json.load(open(os.path.join(root, f)))
                if isinstance(d, dict) and "arm" in d:
                    out.setdefault(d["arm"], d)
    return ({a: out[a] for a in ORDER if a in out}
            | {a: v for a, v in out.items() if a not in ORDER})


def fmt(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{p}f}"
    if isinstance(x, (list, tuple)):
        return "[" + " ".join(fmt(v, p) for v in x) + "]"
    return str(x)


def row(vals, w=18):
    return "".join(str(v).ljust(w) for v in vals)


def reprice(t_cum, steps_agent, dt, d_fb_old, d_fb_new):
    """Re-price a logged (t_cum, steps_agent) pair under a different sensorimotor delay.

    Pricing here is pure bookkeeping -- fixed cycle counts, and no in-run decision ever reads the
    ledger -- so the whole commitment-pays question is a POST-HOC sweep rather than a family of
    runs. Since `t = steps*dt + fb*d_fb`, the feedback-event count is recovered exactly:
    `fb = (t_cum - steps_agent*dt) / d_fb_old`.
    """
    fb = (t_cum - steps_agent * dt) / d_fb_old
    return steps_agent * dt + fb * d_fb_new


def interp_xy(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
        if x0 <= x <= x1:
            f = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + f * (y1 - y0)
    return ys[-1]


def interp(ladder, t):
    return interp_xy([r["t_cum"] for r in ladder], [r["e_perf"] for r in ladder], t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--window", type=int, default=4)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    R = load(a.tag)
    if not R:
        raise SystemExit(f"no results for tag {a.tag}")
    arms = list(R)
    cfg0 = R[arms[0]]["config"]
    # COMMON HORIZON. If any arm is short (here: `never` hit the Modal wall-clock timeout at c75
    # of 90), every arm is compared at the last cycle they ALL reached. Comparing a truncated arm
    # against others' full run would be a budget mismatch dressed as an accuracy result. The
    # committed arms' full-c90 numbers are printed separately so nothing is hidden.
    HZ = min(R[x]["ladder"][-1]["cycle"] for x in arms)
    FULL = max(R[x]["ladder"][-1]["cycle"] for x in arms)
    truncated = [x for x in arms if R[x]["ladder"][-1]["cycle"] > HZ]
    incomplete = [x for x in arms if not R[x].get("complete")]

    def lad(arm, hz=None):
        return [r for r in R[arm]["ladder"] if r["cycle"] <= (HZ if hz is None else hz)]

    NS = len(cfg0["h_seg"])
    print(f"\n=== legato round 2b: tag {a.tag}, seed {cfg0['seed']}, curl_b={cfg0['curl_b']} ===")
    print(f"    piece H_app={cfg0['h_app']} H_seg={cfg0['h_seg']} (phrase {sum(cfg0['h_seg'])} "
          f"steps), patch on segment {cfg0['patch_seg']}")
    print(f"    n_cycles={cfg0['n_cycles']} commit_seg={cfg0['commit_seg']} "
          f"commit_phrase={cfg0['commit_phrase']} interleave={cfg0['interleave_period']} "
          f"calp={cfg0['calp']}")
    print(f"    cost shaping: vel_pen={cfg0['vel_pen']} vel_pen_mid={cfg0['vel_pen_mid']} "
          f"commit_lookahead={cfg0['commit_lookahead']}  (all from the gate's CAL-C)")
    s0 = R[arms[0]]["setup"]
    print(f"    refs (piece @tempo): stale {s0['ref_stale']:.4f}  ceiling {s0['ref_ceiling']:.4f}  "
          f"reactive {s0['ref_reactive']:.4f}  (range {s0['usable_range']:.4f})")
    for arm in arms:
        if not R[arm].get("complete"):
            print(f"    !! {arm} INCOMPLETE -- last logged cycle "
                  f"{R[arm].get('log', {}).get('cycle', [None])[-1]} "
                  f"(no done.txt; Modal wall-clock timeout)")
    if truncated:
        print(f"\n    *** COMMON HORIZON c{HZ}: {', '.join(truncated)} ran to c{FULL} and are "
              f"TRUNCATED to c{HZ} for every\n        comparison below, because "
              f"{', '.join(incomplete) or 'an arm'} stopped there. Full-c{FULL} numbers for the\n"
              f"        complete arms are printed separately. Etude precedent (eg_s0, lost at "
              f"c42).")

    def win(arm, key="e_perf", hz=None):
        L = lad(arm, hz)
        v = [r[key] for r in L[-a.window:]]
        return sum(v) / len(v)

    print(f"\n--- HEADLINE: piece error at performance tempo x priced time (window={a.window})")
    print(row(["arm", "e_piece(win)", "e_piece(fin)", "fb/piece", "t_priced", "never@t", "margin",
               "rank"], 15))
    order = sorted(arms, key=lambda x: win(x))
    for arm in arms:
        L = lad(arm)[-1]
        t = L["t_cum"]
        nv = interp(lad("never"), t) if "never" in R else None
        w = win(arm)
        print(row([arm, fmt(w), fmt(L["e_perf"]), L.get("n_fb_perf", "-"), fmt(t, 1),
                   fmt(nv), fmt(None if nv is None else nv - w),
                   f"{order.index(arm) + 1}/{len(arms)}"], 15))

    if truncated:
        print(f"\n--- the same arms at their FULL horizon c{FULL} (NOT comparable to `never`, "
              f"which stopped at c{HZ})")
        print(row(["arm", "e_piece(win)", "e_piece(fin)", "t_priced", f"vs its own c{HZ}"], 16))
        for arm in truncated:
            wf, wh = win(arm, hz=FULL), win(arm)
            print(row([arm, fmt(wf), fmt(R[arm]["ladder"][-1]["e_perf"]),
                       fmt(R[arm]["ladder"][-1]["t_cum"], 1), fmt(wf - wh)], 16))

    print("\n--- per-segment breakdown (segment 1 carries the patch; errors compound downstream)")
    print(row(["arm"] + [f"seg{k}" for k in range(NS)] + ["react", "ball_seg", "ball_phrase"], 15))
    for arm in arms:
        L = lad(arm)[-1]
        print(row([arm] + [fmt(v) for v in L["e_perf_by_seg"]]
                  + [fmt(L.get("e_react")), fmt(L.get("e_ball_seg")),
                     fmt(L.get("e_ball_phrase"))], 15))

    print("\n--- the commit record (every frozen arm reads its own seam-matched audition matrix)")
    print(row(["arm", "cycle", "span", "pool", "chosen", "best_fixed", "oracle", "gain",
               "cells", "distinct"], 13))
    degenerate = []
    for arm in arms:
        evs = [e for e in R[arm]["events"] if e["kind"] == "commit"]
        if not evs:
            print(row([arm] + ["-"] * 9, 13)); continue
        for e in evs:
            lib = e.get("library", {})
            nd = lib.get("n_distinct")
            if nd == 1:
                degenerate.append((arm, e["cycle"], e.get("span")))
            print(row([arm, e["cycle"], e.get("span", "-"), e.get("n_pool", "-"),
                       fmt(e.get("chosen_score")), fmt(e.get("best_fixed")),
                       fmt(e.get("per_state_oracle")), fmt(e.get("oracle_gain"), 2),
                       str(lib.get("cell_sizes", "-")), nd if nd is not None else "-"], 13))
    if degenerate:
        print("    !! LIBRARY COLLAPSED to one distinct pick at: "
              + ", ".join(f"{a} c{c} span{s}" for a, c, s in degenerate))
        print("       Those commits are `fixed` units wearing a library's name -- the "
              "state-conditioning\n       is inert there and any fusion margin at those cells is "
              "not about conditioning.")

    print("\n--- THE FUSION COMPARISON (the round's own control: these two differ ONLY by fusing)")
    for pair in (("seg_frozen", "phrase_frozen"), ("seg_plan_launch", "phrase_plan_launch")):
        if pair[0] in R and pair[1] in R:
            a0, a1 = win(pair[0]), win(pair[1])
            t0 = lad(pair[0])[-1]["t_cum"]; t1 = lad(pair[1])[-1]["t_cum"]
            f0 = lad(pair[0])[-1].get("n_fb_perf")
            f1 = lad(pair[1])[-1].get("n_fb_perf")
            print(f"  {pair[0]:20s} {a0:.4f} @ {t0:9.1f}s ({f0} fb/piece)")
            print(f"  {pair[1]:20s} {a1:.4f} @ {t1:9.1f}s ({f1} fb/piece)   "
                  f"-> fusion costs {a1 - a0:+.4f} error, saves {t0 - t1:+.1f}s")
    if "phrase_frozen" in R and "phrase_chain_fixed" in R:
        print(f"  state-conditioning at the phrase launch buys "
              f"{win('phrase_chain_fixed') - win('phrase_frozen'):+.4f} "
              f"(phrase_chain_fixed - phrase_frozen); <= 0 means the fused arm's one feedback event "
              f"is buying nothing, which is the etude's diagnosis one level up")

    print("\n--- post-commit behaviour")
    print(row(["arm", "drift", "drift sd", "commits", "cycles", "interleaved", "reselects"], 14))
    for arm in arms:
        s = R[arm].get("summary")
        if s is None:                      # arm killed before the summary block was written
            print(row([arm, "-", "-", "-", "(incomplete)", "-", "-"], 14)); continue
        print(row([arm, fmt(s.get("post_commit_drift")), fmt(s.get("post_commit_sd")),
                   s.get("n_commits"), str(s.get("commit_cycles")), s.get("n_interleaved"),
                   s.get("n_reselect")], 14))

    print("\n--- priced time, itemised (the price vector can be re-applied post-hoc)")
    kinds = sorted({k for arm in arms for k in R[arm]["ledger"]["t_by_kind"]})
    print(row(["arm"] + kinds + ["total", "steps", "plans", "delib"], 13))
    for arm in arms:
        L = R[arm]["ledger"]
        print(row([arm] + [fmt(L["t_by_kind"].get(k, 0.0), 1) for k in kinds]
                  + [fmt(L["t_priced"], 1), f"{L['steps_agent']:,}", f"{L['plans_agent']:,}",
                     f"{L.get('delib_agent', 0):,}"], 13))

    # ---------------------------------------------------------------- the economics surface
    dt = cfg0["frame_skip"] * cfg0["timestep"]
    d0 = cfg0["d_fb"]
    nv = R.get("never")
    if nv is None:
        print("\n(no `never` arm -- skipping the economics surface)")
        print(); return

    # ---------------------------------------------------------------- the definitive clock
    print("\n--- MASTERY CLOCK, read off `never`'s own ladder (round 1b's rule, verbatim):")
    print("    trailing-3 mean of the held-out BALLISTIC curve, plateau = mean +- sd over the last")
    print("    5 probes, clock = first probe from which the smoothed curve stays within 1 sd.")
    print("    The gate's clocks (l1/l2) were LOWER BOUNDS -- 20 warmup cycles was not enough to")
    print("    plateau. This is the measurement the commit schedule should have been read from.")

    def clock_of(series, cycles, w=3, tail=5):
        import numpy as _np
        v = _np.asarray(series, float)
        if len(v) < w + tail:
            return None, None, None, None
        sm = _np.array([v[max(0, i - w + 1):i + 1].mean() for i in range(len(v))])
        mu, sd = sm[-tail:].mean(), sm[-tail:].std()
        prev = sm[-2 * tail:-tail].mean() if len(sm) >= 2 * tail else float("nan")
        plateaued = bool(_np.isfinite(prev) and (prev - mu) <= sd)
        for i in range(len(sm)):
            if (sm[i:] <= mu + sd).all():
                return cycles[i], float(mu), float(sd), plateaued
        return None, float(mu), float(sd), plateaued

    nl = lad("never")
    cyc = [r["cycle"] for r in nl]
    print(row(["curve", "clock", "plateau", "plateau sd", "plateaued?", "schedule used"], 15))
    sched = {"e_ball_seg": cfg0["commit_seg"], "e_ball_phrase": [cfg0["commit_phrase"]]}
    for key in ("e_ball_seg", "e_ball_phrase", "e_react", "e_perf"):
        if key not in nl[0]:
            continue
        c, mu, sd, pl = clock_of([r[key] for r in nl], cyc)
        print(row([key, f"c{c}" if c else "-", fmt(mu), fmt(sd),
                   "yes" if pl else "NO", str(sched.get(key, "-"))], 15))
    cbs, _, _, plbs = clock_of([r["e_ball_seg"] for r in nl], cyc)
    if cbs is not None:
        segs, ph = cfg0["commit_seg"], cfg0["commit_phrase"]
        verdict = ("VALIDATED: every commit fired at or after the measured clock"
                   if min(segs) >= cbs else
                   f"NOT validated: segments committed at {segs} but the ballistic clock is c{cbs}"
                   " -- those commits fired MID-DESCENT")
        print(f"    -> schedule check (segments {segs}, phrase c{ph} vs ball_seg clock c{cbs}): "
              f"{verdict}")
        if not plbs:
            print("       (never's own ballistic curve had NOT plateaued by the common horizon, so"
                  " this clock is itself a LOWER BOUND)")

    print("\n--- d_fb SWEEP: does commitment pay at a different sensorimotor delay?")
    print("    The round's prediction is that phrase commitment OVERTAKES segment commitment as\n"
          "    d_fb rises, because it pays two fewer feedback events per traversal. The crossing\n"
          "    is the tempo knob made quantitative.")
    grid = [0.05, 0.10, 0.30, 1.00, 3.00]
    others = [x for x in arms if x != "never"]
    print(row(["d_fb"] + others, 15))
    for d in grid:
        nx = [reprice(r["t_cum"], r["steps_agent"], dt, d0, d) for r in lad("never")]
        ny = [r["e_perf"] for r in lad("never")]
        cells = []
        for arm in others:
            L = lad(arm)[-1]
            ta = reprice(L["t_cum"], L["steps_agent"], dt, d0, d)
            e_arm = win(arm)
            cells.append(f"{nx[-1] / ta:.2f}x {'PAYS' if e_arm < interp_xy(nx, ny, ta) else 'no'}")
        print(row([f"{d:.2f}"] + cells, 15))
    print("    (cell = never_time/arm_time, then whether the arm's window-mean error beats never's\n"
          "     own trajectory interpolated to the arm's repriced budget. `no` everywhere means the\n"
          "     comparison is ERROR-dominated: no price vector can rescue it.)")

    print("\n    priced-time ranking at each d_fb (the ordering among committed arms):")
    for d in grid:
        rk = sorted(others, key=lambda x: reprice(lad(x)[-1]["t_cum"],
                                                  lad(x)[-1]["steps_agent"], dt, d0, d))
        print(f"      d_fb={d:<5} " + " < ".join(
            f"{x}({reprice(lad(x)[-1]['t_cum'], lad(x)[-1]['steps_agent'], dt, d0, d):.0f}s)"
            for x in rk))

    have = all("plans_agent" in lad(x)[-1] for x in arms)
    if have:
        import numpy as _np
        for axis, key, label in (("d_plan", "plans_agent", "per PLAN EVENT (the cost of "
                                  "interrupting yourself)"),
                                 ("d_delib", "delib_agent", "per CEM ROLLOUT-STEP (the cost of "
                                  "thinking)")):
            if not all(key in R[x]["ladder"][-1] for x in arms):
                continue
            print(f"\n--- {axis} SURFACE: smallest {axis} at which each arm beats `never` -- {label}")
            print("    `-` = wins already at 0; `never` = no price up to the grid's top flips it.")
            pg = [0.0] + list(_np.logspace(-6 if axis == "d_delib" else -3, 1, 61))
            print(row(["d_fb"] + others, 15))
            for d in [0.05, 0.10, 0.30, 1.00]:
                cells = []
                for arm in others:
                    La = lad(arm)[-1]
                    e_arm = win(arm)
                    hit = "never"
                    for dp in pg:
                        nx = [reprice(r["t_cum"], r["steps_agent"], dt, d0, d) + r[key] * dp
                              for r in lad("never")]
                        ny = [r["e_perf"] for r in lad("never")]
                        ta = reprice(La["t_cum"], La["steps_agent"], dt, d0, d) + La[key] * dp
                        if e_arm < interp_xy(nx, ny, ta):
                            hit = "-" if dp == 0.0 else f"{dp:.3g}"
                            break
                    cells.append(hit)
                print(row([f"{d:.2f}"] + cells, 15))
            print(row([key] + [f"{lad(x)[-1][key]:,}" for x in others], 15))
            print(f"    (never: {lad('never')[-1][key]:,})")

    print("\n--- payback: first probe at which a committed arm beats `never` at MATCHED priced time")
    for arm in others:
        hit = None
        for r in lad(arm):
            if r["n_committed"] and r["e_perf"] < interp(lad("never"), r["t_cum"]):
                hit = (r["cycle"], r["t_cum"], r["e_perf"], interp(lad("never"), r["t_cum"]))
                break
        print(f"  {arm:22s} " + ("never" if hit is None else
                                 f"cycle {hit[0]:3d}  t={hit[1]:9.1f}s  {hit[2]:.4f} vs never "
                                 f"{hit[3]:.4f}"))
    print()


if __name__ == "__main__":
    main()
