"""Reduce a round-1 run into the arm tables.

Fetches `/data/practice_fingering/<tag>/<arm>/results.json` off `mujoco-control-data` (`--fetch`)
and prints the headline (drilled boundary error at performance tempo x priced time), the commit
record, the seam-law check, post-commit drift, the candidate-pool geometry, and the priced-time
itemisation.

TWO READING RULES, both forced by round 0.

  * **Window-mean, not final.** Round-0 CAL-D found the closed-loop descent oscillates with an
    amplitude comparable to the whole usable range, and that this is NOT metering noise (the
    metering seed is fixed, so the run-through is deterministic given the model) -- it is a
    ballistic grader amplifying a model bootstrapping on its own on-policy data. The oscillation
    survives post-commit only in `never`, whose performance still runs through the live model;
    every committed arm's performance is a frozen object. So a single final probe is a biased
    comparison in `never`'s favour or against it depending on phase. Headline = mean over the last
    `--window` probes; the final value is printed beside it.
  * **Ranks are the currency.** Single seed. `recital`'s methodology export: recovery fractions at
    one RNG stream position carry +-0.15; the ordering is the claim.

Usage (from experiments/):
    python3 mjc/practice/fingering/analyze_fingering.py --tag f0 --fetch
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["never", "fixed", "library_kmeans", "library_proj", "regress", "mean"]


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_fingering/{tag}", dst], check=False,
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
                    out[a] = json.load(open(p))
    if os.path.isdir(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".json") and not f.startswith("launch"):
                d = json.load(open(os.path.join(root, f)))
                if isinstance(d, dict) and "arm" in d:
                    out.setdefault(d["arm"], d)
    return {a: out[a] for a in ORDER if a in out} | {a: v for a, v in out.items()
                                                     if a not in ORDER}


def fmt(x, p=4):
    if x is None:
        return "-"
    return f"{x:.{p}f}" if isinstance(x, float) else str(x)


def row(vals, w=16):
    return "".join(str(v).ljust(w) for v in vals)


def reprice(t_cum, steps_agent, dt, d_fb_old, d_fb_new):
    """Re-price a logged (t_cum, steps_agent) pair under a different sensorimotor delay.

    Pricing in this design is pure bookkeeping -- fixed cycle counts, and no in-run decision ever
    reads the ledger -- so the whole commitment-pays question is a post-hoc sweep rather than a
    family of runs. Every probe logs both the priced time and the cumulative agent step count, and
    `t = steps*dt + fb*d_fb`, so the feedback-event count is recovered EXACTLY:

        fb = (t_cum - steps_agent * dt) / d_fb_old

    (Verified against the analytic schedule on round 1: `never` -> 90,720 events = 60 cycles x
    (24 practice + 48 metering) performers x 21 events; a committed arm -> 45,312 = 12,000 practice
    + 24,000 metering + 9,216 audition + 96 score-set. Exact, both arms.)
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
    """`never`'s own trajectory evaluated at another arm's priced time -- the matched-budget
    comparison the etude used, rather than comparing two different points on two curves."""
    pts = [(r["t_cum"], r["e_perf"]) for r in ladder]
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, e0), (t1, e1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return e0 + f * (e1 - e0)
    return pts[-1][1]


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
    print(f"\n=== fingering round 1: tag {a.tag}, seed {cfg0['seed']}, curl_b={cfg0['curl_b']}, "
          f"push_a={cfg0['push_a']} ===")
    print(f"    n_cycles={cfg0['n_cycles']} commit@{cfg0['commit_cycle']} "
          f"n_cand={cfg0['n_cand']} n_score={cfg0['n_score']} n_lib={cfg0['n_lib']} "
          f"trace_window={cfg0['trace_window']}")
    print(f"    commit rationale: {cfg0.get('commit_rationale')}")
    s0 = R[arms[0]]["setup"]
    print(f"    refs: stale {s0['ref_stale']:.4f}  ceiling {s0['ref_ceiling']:.4f}  "
          f"reactive {s0['ref_reactive']:.4f}  (range {s0['usable_range']:.4f})")
    for arm in arms:
        if not R[arm].get("complete"):
            print(f"    !! {arm} incomplete -- last cycle "
                  f"{R[arm].get('log', {}).get('cycle', [None])[-1]}")

    def win(arm, key="e_perf"):
        L = R[arm]["ladder"]
        v = [r[key] for r in L[-a.window:]]
        return sum(v) / len(v)

    print(f"\n--- HEADLINE: drilled boundary error at performance tempo x priced time "
          f"(window={a.window} probes)")
    print(row(["arm", "e_perf(win)", "e_perf(fin)", "t_priced", "never@t", "margin", "rank"]))
    order = sorted(arms, key=lambda x: win(x))
    for arm in arms:
        t = R[arm]["ladder"][-1]["t_cum"]
        nv = interp(R["never"]["ladder"], t) if "never" in R else None
        w = win(arm)
        print(row([arm, fmt(w), fmt(R[arm]["ladder"][-1]["e_perf"]), fmt(t, 1),
                   fmt(nv), fmt(None if nv is None else nv - w),
                   f"{order.index(arm) + 1}/{len(arms)}"]))

    print("\n--- matched control mode on the same held-out geometry (all arms, free instrument)")
    print(row(["arm", "e_perf", "e_reactive", "e_ballistic", "e_approach"]))
    for arm in arms:
        L = R[arm]["ladder"][-1]
        print(row([arm, fmt(L["e_perf"]), fmt(L["e_react"]), fmt(L["e_ball"]),
                   fmt(L.get("e_app"))]))

    print("\n--- the commit event (every selection arm reads the SAME audition matrix)")
    print(row(["arm", "cycle", "chosen_score", "best_fixed", "state_oracle", "oracle_gain",
               "cells", "distinct"], 15))
    for arm in arms:
        ev = [e for e in R[arm]["events"] if e["kind"] == "commit"]
        if not ev:
            print(row([arm, "-", "-", "-", "-", "-", "-", "-"], 15)); continue
        e = ev[0]
        lib = e.get("library", {})
        og = (None if e.get("best_fixed") is None or e.get("per_state_oracle") is None
              else e["best_fixed"] / max(e["per_state_oracle"], 1e-9))
        print(row([arm, e["cycle"], fmt(e.get("chosen_score")), fmt(e["best_fixed"]),
                   fmt(e.get("per_state_oracle")), fmt(og, 2),
                   str(lib.get("cell_sizes", "-")), str(lib.get("n_distinct", "-"))], 15))
        if "r2_of_key" in lib:
            print(f"    {arm}: supervised key R^2 = {lib['r2_of_key']:.3f}  "
                  f"edges={[round(v, 3) for v in lib['edges']]}  picks={lib['picks']}")

    print("\n--- seam law + post-commit behaviour")
    print(row(["arm", "optimism gap", "drift", "drift sd", "recerts", "cert would fire"], 17))
    for arm in arms:
        s = R[arm]["summary"]
        print(row([arm, fmt(s.get("optimism_gap"), 3), fmt(s.get("post_commit_drift")),
                   fmt(s.get("post_commit_sd")), s.get("recerts"),
                   s.get("certificate_would_fire")], 17))

    print("\n--- candidate-pool geometry (logged for the averaging-geometry question; not acted on)")
    print(row(["arm", "coherence", "coh(top1/3)", "disp", "pair corr", "corr p10"], 15))
    for arm in arms:
        ev = [e for e in R[arm]["events"] if e["kind"] == "commit"]
        if not ev or "pool_geometry" not in ev[0]:
            print(row([arm, "-", "-", "-", "-", "-"], 15)); continue
        g, gs = ev[0]["pool_geometry"], ev[0]["pool_geometry_successful"]
        print(row([arm, fmt(g["coherence"], 3), fmt(gs["coherence"], 3), fmt(g["disp_mean"], 3),
                   fmt(g["pairwise_corr_mean"], 3), fmt(g["pairwise_corr_p10"], 3)], 15))

    print("\n--- priced time, itemised (the price vector can be re-applied post-hoc)")
    kinds = sorted({k for arm in arms for k in R[arm]["ledger"]["t_by_kind"]})
    print(row(["arm"] + kinds + ["total", "steps_agent", "plans_agent", "steps_instr"], 14))
    for arm in arms:
        L = R[arm]["ledger"]
        print(row([arm] + [fmt(L["t_by_kind"].get(k, 0.0), 1) for k in kinds]
                  + [fmt(L["t_priced"], 1), L["steps_agent"], L.get("plans_agent", "-"),
                     L["steps_instrument"]], 14))

    # ---------------------------------------------------------------- per-cell pool geometry
    print("\n--- per-cell pool geometry: does the conditioning variable separate the modes?")
    any_cg = False
    for arm in arms:
        ev = [e for e in R[arm]["events"] if e["kind"] == "commit" and "cell_geometry" in e]
        if not ev:
            continue
        any_cg = True
        cg = ev[0]["cell_geometry"]
        g = cg["global"]
        print(f"  {arm} (commit c{ev[0]['cycle']}, top {cg['frac']:.0%} of candidates)")
        print(f"    {'global':>10s}  coherence {g['coherence']:.3f}  pair-corr "
              f"{g['pairwise_corr_mean']:.3f}  mean-replay {cg['global_mean_replay']:.4f}  "
              f"best {cg['global_best']:.4f}  mean/best "
              f"{cg['global_mean_replay'] / max(cg['global_best'], 1e-9):.2f}x")
        for c in cg["cells"]:
            gg = c["geometry"]
            print(f"    {'cell ' + str(c['cell']):>10s}  coherence {gg['coherence']:.3f}  pair-corr "
                  f"{gg['pairwise_corr_mean']:.3f}  mean-replay {c['mean_replay']:.4f}  "
                  f"best {c['best_in_cell']:.4f}  mean/best {c['mean_over_best']:.2f}x  "
                  f"(n_states={c['n_states']})")
        coh = [c["geometry"]["coherence"] for c in cg["cells"]]
        mob = [c["mean_over_best"] for c in cg["cells"]]
        if coh:
            print(f"    -> cell coherence {min(coh):.3f}-{max(coh):.3f} vs global "
                  f"{g['coherence']:.3f}; within-cell mean/best "
                  f"{min(mob):.2f}-{max(mob):.2f}x vs global "
                  f"{cg['global_mean_replay'] / max(cg['global_best'], 1e-9):.2f}x")
    if not any_cg:
        print("  (no arm logged cell_geometry -- pre-1b run)")

    # ---------------------------------------------------------------- d_fb price sweep
    print("\n--- d_fb price sweep: does commitment pay at a different sensorimotor delay?")
    dt = cfg0["frame_skip"] * cfg0["timestep"]
    d0 = cfg0["d_fb"]
    grid = [0.05, 0.10, 0.30, 1.00, 3.00]
    nv = R.get("never")
    print(row(["d_fb"] + [a for a in arms if a != "never"], 15))
    if nv is not None:
        nvt = [reprice(r["t_cum"], r["steps_agent"], dt, d0, d0) for r in nv["ladder"]]
        print(row(["never t"] + ["" for a in arms if a != "never"], 15))
        for d in grid:
            nx = [reprice(r["t_cum"], r["steps_agent"], dt, d0, d) for r in nv["ladder"]]
            ny = [r["e_perf"] for r in nv["ladder"]]
            cells = []
            for arm in arms:
                if arm == "never":
                    continue
                L = R[arm]["ladder"][-1]
                ta = reprice(L["t_cum"], L["steps_agent"], dt, d0, d)
                e_arm = sum(r["e_perf"] for r in R[arm]["ladder"][-a.window:]) / a.window
                e_nv = interp_xy(nx, ny, ta)
                cells.append(f"{nx[-1] / ta:.2f}x {'PAYS' if e_arm < e_nv else 'no'}")
            print(row([f"{d:.2f}"] + cells, 15))
        print("    (cell = never_total_time / arm_total_time, then whether the arm's window-mean "
              "error beats never's own trajectory interpolated to the arm's repriced budget)")
        print("    (a `no` at every d_fb means the comparison is ERROR-dominated, not "
              "time-dominated: no price vector can rescue it)")
        # the d_fb at which committing first becomes TIME-positive: solve
        # steps_n*dt + fb_n*d = steps_a*dt + fb_a*d for d
        Ln = nv["ladder"][-1]
        sn, fn = Ln["steps_agent"], (Ln["t_cum"] - Ln["steps_agent"] * dt) / d0
        for arm in arms:
            if arm == "never":
                continue
            La = R[arm]["ladder"][-1]
            sa = La["steps_agent"]
            fa = (La["t_cum"] - La["steps_agent"] * dt) / d0
            den = fn - fa
            be = (sa - sn) * dt / den if abs(den) > 1e-9 else float("nan")
            print(f"    {arm:28s} time-positive above d_fb = {be:.4f} s "
                  f"(steps {sa:,} vs never {sn:,}; events {fa:,.0f} vs {fn:,.0f})")

    have_plans = all("plans_agent" in R[a]["ladder"][-1] for a in arms)
    if nv is not None and have_plans:
        print("\n--- deliberation price surface: smallest d_plan (s) at which each arm beats "
              "`never`")
        print("    A reactive traversal deliberates H_drill times, a plan-at-launch unit once, a "
              "frozen unit\n    not at all -- so the ops separate on an axis that a feedback-only "
              "price cannot see.\n    `-` = wins already at d_plan=0; `never` = no d_plan up to "
              "10 s flips it.")
        import numpy as _np
        pg = [0.0] + list(_np.logspace(-3, 1, 61))
        print(row(["d_fb"] + [a for a in arms if a != "never"], 15))
        for d in [0.05, 0.10, 0.30, 1.00]:
            cells = []
            for arm in arms:
                if arm == "never":
                    continue
                La = R[arm]["ladder"][-1]
                e_arm = sum(r["e_perf"] for r in R[arm]["ladder"][-a.window:]) / a.window
                hit = "never"
                for dp in pg:
                    nx = [reprice(r["t_cum"], r["steps_agent"], dt, d0, d)
                          + r["plans_agent"] * dp for r in nv["ladder"]]
                    ny = [r["e_perf"] for r in nv["ladder"]]
                    ta = reprice(La["t_cum"], La["steps_agent"], dt, d0, d) + La["plans_agent"] * dp
                    if e_arm < interp_xy(nx, ny, ta):
                        hit = "-" if dp == 0.0 else f"{dp:.4g}"
                        break
                cells.append(hit)
            print(row([f"{d:.2f}"] + cells, 15))
        print(row(["plans"] + [f"{R[a]['ladder'][-1]['plans_agent']:,}" for a in arms
                               if a != "never"], 15))
        print(f"    (never deliberates {nv['ladder'][-1]['plans_agent']:,} times)")

    if "never" in R:
        print("\n--- payback: first probe at which a committed arm beats `never` at MATCHED priced time")
        for arm in arms:
            if arm == "never":
                continue
            hit = None
            for r in R[arm]["ladder"]:
                if r["committed"] and r["e_perf"] < interp(R["never"]["ladder"], r["t_cum"]):
                    hit = (r["cycle"], r["t_cum"], r["e_perf"],
                           interp(R["never"]["ladder"], r["t_cum"]))
                    break
            print(f"  {arm:16s} " + ("never" if hit is None else
                                     f"cycle {hit[0]:3d}  t={hit[1]:8.1f}s  "
                                     f"{hit[2]:.4f} vs never {hit[3]:.4f}"))
    print()


if __name__ == "__main__":
    main()
