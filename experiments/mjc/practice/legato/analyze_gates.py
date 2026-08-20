"""Fetch the round-2a gate results off the volume and print the gate report.

One table per gate across worlds, plus the calibration record (CAL-C cost shaping, CAL-P planner
sizing per span, CAL-D mastery clocks) that the main run's knobs are read from.

READING RULE, inherited from round 1: **ranks and shapes are the currency, not absolute values.**
Single seed. What a gate has to establish is that an axis EXISTS and is above the noise floor, and
what a calibration has to establish is which setting a measurement picks -- neither is a claim about
the third decimal place.

Usage (from experiments/):
    python3 mjc/practice/legato/analyze_gates.py --tag l0 --fetch
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


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
            for w in sorted(os.listdir(base)):
                p = os.path.join(base, w, "results.json")
                if os.path.isfile(p):
                    out[w] = json.load(open(p))
    if os.path.isdir(root):
        for f in sorted(os.listdir(root)):
            if f.endswith(".json"):
                d = json.load(open(os.path.join(root, f)))
                if isinstance(d, dict) and "world" in d:
                    out.setdefault(d["world"], d)
    return out


def fmt(x, p=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{p}f}"
    if isinstance(x, (list, tuple)):
        return "[" + " ".join(fmt(v, p) for v in x) + "]"
    return str(x)


def row(vals, w=17):
    return "".join(str(v).ljust(w) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    R = load(a.tag)
    if not R:
        raise SystemExit(f"no results for tag {a.tag}")
    ws = list(R)
    cfg0 = R[ws[0]]["config"]
    NS = len(cfg0["h_seg"])
    print(f"\n=== legato round 2a gates: tag {a.tag}, seed {cfg0['seed']}, worlds {ws} ===")
    print(f"    piece: H_app={cfg0['h_app']} H_seg={cfg0['h_seg']} (phrase {sum(cfg0['h_seg'])} "
          f"steps) patch on drilled segment {cfg0['patch_seg']}")
    for w in ws:
        if not R[w].get("complete"):
            print(f"    !! {w} incomplete")

    print("\n--- C0: the piece (geometry is a precondition, not a result)")
    print(row(["world", "leg lengths", "m/step", "turn cos", "radius", "gate@wps"], 22))
    for w in ws:
        c = R[w]["c0"]
        print(row([w, fmt(c["leg_lengths"], 3), fmt(c["m_per_step"], 4), fmt(c["turn_cos"], 2),
                   fmt(c["radius_range"], 3), fmt(c["gate_at_waypoints"], 3)], 22))

    print("\n--- CAL-C: cost shaping. CRITERION is lexicographic: (1) competence guard on the")
    print("    REACTIVE arm (the reference opponent must be strong), then (2) max FM-ATTRIBUTABLE")
    print("    FRACTION on BALLISTIC-per-segment -- the mode that actually transmits model quality")
    print("    (P5: ~4.9x reactive). NOT min raw error: that was l0's mistake, and it picked a")
    print("    setting where only 3% of the reactive error was FM-attributable.")
    for w in ws:
        cc = R[w].get("cal_cost")
        if not cc:
            continue
        ch = cc["chosen"]
        print(f"  {w}: chosen w_waypoint={ch.get('w_waypoint')} vel_pen_mid={ch.get('vel_pen_mid')} "
              f"vel_pen_terminal={ch.get('vel_pen_terminal')} "
              f"lookahead_gamma={ch.get('lookahead_gamma', ch.get('commit_lookahead'))}")
        print("    " + row(["cell", "react", "react by-seg", "ceiling", "FM-attrib", "seg",
                            "phrase"], 15))
        for k, c in cc.get("cost", {}).items():
            print("    " + row([k, fmt(c["react"]), fmt(c.get("react_by_seg"), 3),
                                fmt(c["react_ceiling"]),
                                f"{100 * c['fm_attributable']:.0f}%", fmt(c["seg"]),
                                fmt(c["phrase"])], 15))
        if cc.get("lookahead"):
            print("    " + row(["lookahead", "seg piece", "by segment", "delib"], 15))
            for v, r in cc["lookahead"].items():
                print("    " + row([v, fmt(r["seg"]), fmt(r["by_seg"]), f"{r['delib']:,}"], 15))
        seam = cc.get("seam")
        if seam:
            print(f"    STAGE 2 (vel_pen terminal x lookahead gamma). Criterion = min NEUTRAL mean")
            print(f"    piece error across reactive/seg/phrase, among cells whose FM axis survives.")
            print(f"    metering convention: {cc['chosen'].get('metering_convention')}")
            print("    " + row(["cell", "react", "seg", "phrase", "NEUTRAL", "FM-attrib"], 13))
            for k, c in seam.items():
                print("    " + row([k, fmt(c["react"]), fmt(c["seg"]), fmt(c["phrase"]),
                                    fmt(c["neutral"]),
                                    f"{100 * c['fm_attributable']:.0f}%"], 13))
            print("\n    G5b ordering AT EVERY COST CELL (stale / ceiling piece error) -- this is")
            print("    where the cost-dependence of the span effect is visible:")
            keys = list(next(iter(seam.values()))["g5b"].keys())
            print("    " + row(["cell"] + keys, 15))
            for nm in ("stale", "ceiling"):
                print(f"      [{nm}]")
                for k, c in seam.items():
                    print("    " + row([k] + [fmt(c["g5b"][g][nm]["piece"]) for g in keys], 15))
            print("    (span degradation is PRESENT at a cell if piece error rises as the number")
            print("     of committed groups falls, i.e. 1+1+1 < ... < 3)")

    print("\n--- G5a: FM COMPOSITION DIVERGENCE (no planner in it -- this is the mechanism)")
    print(row(["world/src", "model", "h>0.05", "h>0.02"]
              + [f"err@{s}" for s in ["seg0", "seg1", "seg2"][:NS]], 13))
    for w in ws:
        for src, r in R[w]["g5a"].items():
            for nm in ("stale", "ceiling"):
                print(row([f"{w}/{src}", nm, r[nm]["h_005"], r[nm]["h_002"]]
                          + [fmt(v) for v in r[nm]["err_at"]], 13))
    print("    (h = first step at which median predicted-vs-true tip error crosses the threshold;\n"
          "     arm_substrate P4 measured 20-23 for this plant at n=3, independently)")

    print("\n--- CAL-P: planner sizing per span. A span whose error is STILL FALLING at the top of\n"
          "    the grid is STARVED, and any degradation G5b reports at that span is not attributable")
    for w in ws:
        cp = R[w]["cal_planner"]
        print(f"  {w}")
        keys = [k for k in cp["span1"] if k.startswith("k")]
        print("    " + row(["span (dims)"] + keys + ["starved?"], 15))
        for s in range(1, NS + 1):
            r = cp[f"span{s}"]
            dims = s * cfg0["h_seg"][0] * cfg0["n_links"]
            print("    " + row([f"{s} ({dims}d)"]
                               + [f"{r[k]['stale']['e_end']:.4f}/{r[k]['ceiling']['e_end']:.4f}"
                                  for k in keys]
                               + [f"stale {fmt(r['still_improving_stale'])}, "
                                  f"ceil {fmt(r['still_improving_ceiling'])}"], 15))
        print(f"    chosen: {R[w]['cal_planner_chosen']}")
    print("    (cells are stale/ceiling end-of-span median error)")

    print("\n--- G5b: LIVE PLAN-AT-LAUNCH vs RE-GROUNDING COUNT (same segments, same endpoint,\n"
          "    each span given its CAL-P planner -- the only difference is how often it re-grounds)")
    for w in ws:
        print(f"  {w}")
        print("    " + row(["grouping", "fb", "piece", "final", "by segment", "delib"], 15))
        for g, r in R[w]["g5b"].items():
            s = r["stale"]
            print("    " + row([g, s.get("n_fb", "-"), fmt(s["e_piece"]), fmt(s["e_final"]),
                                fmt(s["e_by_seg"]), f"{s.get('delib', 0):,}"], 15))
        print("    (stale FM; ceiling rows are in the JSON)")

    print("\n--- G6: seam information at PHRASE level, and the price of fusing")
    print(row(["world", "launch tip sd", "/noise", "oracle gain", "gain by seg", "opt(noisy)"], 16))
    for w in ws:
        g = R[w]["g6"]
        print(row([w, fmt(g["launch_tip_sd"]), fmt(g["spread_over_noise"], 1),
                   fmt(g["oracle_gain_phrase"], 2), fmt(g["oracle_gain_by_seg"], 2),
                   fmt(g.get("audition_optimism"), 2)], 16))
    print("\n    R^2 of a fixed chain's realised per-segment error on what each granularity can see:")
    print("    " + row(["world", "seg", "from phrase launch", "from own seam", "fusion loses"], 20))
    for w in ws:
        for r in R[w]["g6"]["r2"]:
            print("    " + row([w, r["seg"], fmt(r["r2_from_phrase_launch"], 3),
                                fmt(r["r2_from_seam"], 3),
                                fmt(r["fusion_information_loss"], 3)], 20))

    print("\n--- G7: usable range + noise floor at phrase granularity")
    for w in ws:
        print(f"  {w}")
        print("    " + row(["mode", "fb/piece", "stale", "ceiling", "range", "noise sd",
                            "range/noise"], 13))
        for mn, c in R[w]["g7"]["modes"].items():
            print("    " + row([mn, c["stale"]["n_fb"], fmt(c["stale"]["piece"]),
                                fmt(c["ceiling"]["piece"]), fmt(c["usable_range"]),
                                fmt(c["noise_sd"]), fmt(c["range_over_noise"], 1)], 13))
        print(f"    ballistic/reactive {R[w]['g7']['ballistic_over_reactive']:.2f}x   "
              f"phrase/segment {R[w]['g7']['phrase_over_seg']:.2f}x")

    print("\n--- CAL-D: mastery clocks (round 1b's rule, on the held-out BALLISTIC curve)")
    for w in ws:
        cd = R[w]["cal_descent"]
        print(f"  {w}")
        print("    " + row(["series", "clock", "plateau", "plateau sd", "plateaued?"], 18))
        for k, v in cd["clocks"].items():
            print("    " + row([k, f"c{v['clock']}" if v["clock"] else "-", fmt(v["plateau"]),
                                fmt(v["plateau_sd"]),
                                fmt(v.get("plateaued"))], 18))
        if any(v["clock"] and not v.get("plateaued", True) for v in cd["clocks"].values()):
            print("    !! a curve that had not plateaued makes its clock a LOWER BOUND, not a "
                  "mastery cycle")
        print(f"    ball_seg by cycle:    {[round(v, 4) for v in cd['series']['ball_seg_piece']]}")
        print(f"    ball_phrase by cycle: {[round(v, 4) for v in cd['series']['ball_phrase_piece']]}")
        print(f"    reactive (cycle, err): {[(c, round(v, 4)) for c, v in cd['series']['react_piece']]}")
        fc = cd["series"].get("fm_corridor") or []
        if fc:
            print(f"    FM corridor err (cycle, clean, in-region): "
                  f"{[(c, round(a, 4), None if b is None else round(b, 4)) for c, a, b in fc]}")
            d0, d1 = fc[0][1], fc[-1][1]
            print(f"    -> the MODEL itself went {d0:.4f} -> {d1:.4f} on the clean corridor "
                  f"({'improved' if d1 < d0 else 'DEGRADED'}); compare against ball_seg above to "
                  f"separate a diet problem from a controller problem")

    print("\n--- VERDICT")
    for w in ws:
        print(f"  {w}: {json.dumps(R[w]['verdict'], default=str)}")
    print("\n--- priced time spent by the gate itself (agent vs instrument)")
    print(row(["world", "t_priced", "steps agent", "steps instr", "plans", "delib"], 16))
    for w in ws:
        L = R[w]["ledger"]
        print(row([w, fmt(L["t_priced"], 1), f"{L['steps_agent']:,}",
                   f"{L['steps_instrument']:,}", f"{L['plans_agent']:,}",
                   f"{L.get('delib_agent', 0):,}"], 16))
    print()


if __name__ == "__main__":
    main()
