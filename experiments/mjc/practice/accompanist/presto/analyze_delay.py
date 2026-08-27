"""Reduce a presto Phase B (`delay_gate.py`) run.

Prints the setup and both anchors, the Delta sweep for the two incumbents side by side, the
degradation factors (offbook `d0`'s headline instrument: is delay sensitivity still ordered by
feedback consumption?), the pre-fixed criterion's verdict, and every crossover with the delay at
which it first lands and whether that delay is inside playability.

    python3 mjc/practice/accompanist/presto/analyze_delay.py --tag p0 --fetch
    modal run mjc/practice/accompanist/presto/analyze_delay.py::figs --tag p0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import modal                                                            # noqa: E402

from mjc.shared import app, volume, DATA_DIR                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STRATS = ("reactive", "live_seg", "live_chain", "seg_tape", "seg_fixed", "chain",
          "chain_fixed", "chain_react")


def fetch(tag):
    import subprocess

    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    src = f"/practice_presto/{tag}/delay_gate/results.json"
    r = subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data", src,
                        os.path.join(dst, "delay_gate.json")], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"fetch failed for {src}")
    return os.path.join(dst, "delay_gate.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()

    path = os.path.join(HERE, "results", a.tag, "delay_gate.json")
    if a.fetch or not os.path.exists(path):
        path = fetch(a.tag)
    R = json.load(open(path))
    sw = R.get("sweep", {})
    ds = sorted((int(d) for d in sw), key=int)
    cfg = R.get("config", {})
    lines = []

    def W(*s):
        t = " ".join(str(x) for x in s)
        print(t)
        lines.append(t)

    W(f"\n=== presto Phase B ({a.tag}) — complete={R.get('complete')} "
      f"wall={R.get('wall_s', 0) / 60:.0f} min ===")
    W(f"piece: R={cfg.get('R')} damping={cfg.get('joint_damping')} h_seg={cfg.get('h_seg')} "
      f"mean drilled leg {R.get('mean_leg', 0):.4f} m -> nominal tip speed "
      f"{R.get('nominal_speed', 0):.2f} m/s")
    W(f"knobs from Phase A: lookahead_gamma={cfg.get('lookahead_gamma')} "
      f"vel_pen_mid={cfg.get('vel_pen_mid')} react_look grid={cfg.get('look_grid')} "
      f"calp={cfg.get('calp')}")
    W(f"anchors: ref_stale {R.get('ref_stale', 0):.4f} | half_leg {R.get('half_leg', 0):.4f} "
      f"-> ref_play {R.get('ref_play', 0):.4f}   |   do-nothing floor "
      f"{R.get('hold', {}).get('e', float('nan')):.4f}")

    if sw:
        c0 = sw[str(ds[0])]
        W("\nfeedback events per traversal: "
          + ", ".join(f"{n}={c0[n]['fb']:.0f}" for n in STRATS if n in c0))

    for sfx, nm in (("p", "PREDICTING incumbent (efference copy through the delay) — PRIMARY"),
                    ("", "NAIVE incumbent (act on where you were) — offbook d0's operator")):
        W(f"\n=== the Delta sweep, {nm} ===")
        hdr = f"{'D':>3} {'ms':>5} " + " ".join(f"{n[:9]:>9}" for n in STRATS)
        W(hdr); W("-" * len(hdr))
        for d in ds:
            c = sw[str(d)]
            W(f"{d:>3} {d * 20:>5} " + " ".join(
                f"{c[n + sfx]['e']:>9.4f}" if n + sfx in c else f"{'n/a':>9}" for n in STRATS))
        if len(ds) > 1:
            lo, hi = sw[str(ds[0])], sw[str(ds[-1])]
            W(f"degradation D={ds[0]} -> D={ds[-1]}: " + "  ".join(
                f"{n} {hi[n + sfx]['e'] / max(lo[n + sfx]['e'], 1e-9):.2f}x"
                for n in STRATS if n + sfx in lo))

    # ---------------------------------------------------------------- the criterion
    v = R.get("verdict", {})
    if v:
        W(f"\n=== the pre-fixed criterion (fixed before the run) ===")
        W("Delta* = smallest Delta with e_chain <= e_live_seg <= e_reactive on mean piece error, "
          "subject to min(of the three) <= ref_play = max(ref_stale, 0.5 x mean drilled leg).")
        W("MARGINS, not just verdicts: `ch<live` and `live<rct` are the ordering margins (positive "
          "= the ordering holds), `guard mgn` is ref_play minus the best of the three and "
          "`chain mgn` is ref_play minus the chain alone (positive = inside playability). The "
          "sibling d2 round missed this guard by 3.6% on offbook's piece, which a pass/fail "
          "column alone would have reported as an indistinguishable 'no'.")
        for nm in ("predict", "naive"):
            W(f"\n  --- {nm} ---")
            W(f"  {'D':>3} {'ms':>5} {'chain':>8} {'live_seg':>9} {'reactive':>9} "
              f"{'ord':>4} {'ch<live':>8} {'live<rct':>9} {'play':>5} {'guard mgn':>10} "
              f"{'(%)':>7} {'chain mgn':>10} {'(%)':>7} {'passes':>7}")
            for r in v.get("rows", []):
                x = r.get(nm, {})
                if not x:
                    continue
                W(f"  {r['delay']:>3} {r['ms']:>5.0f} {x['e_chain']:>8.4f} "
                  f"{x['e_live_seg']:>9.4f} {x['e_reactive']:>9.4f} {str(x['ordered'])[0]:>4} "
                  f"{x.get('order_margin_chain_live', float('nan')):>+8.4f} "
                  f"{x.get('order_margin_live_react', float('nan')):>+9.4f} "
                  f"{str(x['playable'])[0]:>5} "
                  f"{x.get('guard_margin', float('nan')):>+10.4f} "
                  f"{100 * x.get('guard_margin_f', float('nan')):>+7.1f} "
                  f"{x.get('chain_vs_guard', float('nan')):>+10.4f} "
                  f"{100 * x.get('chain_vs_guard_f', float('nan')):>+7.1f} "
                  f"{('PASS' if x['passes'] else 'no'):>7}")
        W(f"\n  Delta* (PRIMARY, predicting) = {v.get('delta_star')}   |   "
          f"Delta* (naive, offbook's operator) = {v.get('delta_star_naive')}")
        if v.get("delta_star") is None:
            W("  -> no delay inverts the ordering under the pre-fixed rule. The round stops at the "
              "gate; Delta is not extended and the criterion is not re-fitted (legato F2).")

    # ---------------------------------------------------------------- crossovers
    W("\n=== crossovers (first delay at which each holds, and whether it is inside playability) ===")
    play = R.get("ref_play", float("inf"))
    pairs = [("chain", "reactive"), ("seg_tape", "reactive"), ("chain", "live_seg"),
             ("seg_tape", "live_seg"), ("seg_fixed", "seg_tape"), ("chain_fixed", "chain"),
             ("chain", "seg_tape")]
    for sfx, nm in (("p", "predict"), ("", "naive")):
        for x, y in pairs:
            hit = next((d for d in ds
                        if x + sfx in sw[str(d)] and y + sfx in sw[str(d)]
                        and sw[str(d)][x + sfx]["e"] < sw[str(d)][y + sfx]["e"]), None)
            if hit is None:
                W(f"  [{nm:7s}] {x} < {y}: never in the sweep")
            else:
                c = sw[str(hit)]
                inside = min(c[n + sfx]["e"] for n in ("reactive", "live_seg", "chain")
                             if n + sfx in c) <= play
                W(f"  [{nm:7s}] {x} < {y}: first at Delta = {hit} ({hit * 20} ms), "
                  f"{'INSIDE' if inside else 'outside'} playability "
                  f"({c[x + sfx]['e']:.4f} vs {c[y + sfx]['e']:.4f})")

    # ---------------------------------------------------------------- content, and the incumbent's
    clock = R.get("clock", [])
    if clock:
        W("\n=== the forward model's own competence at each commit (legato CAL-D's curve) ===")
        W("Content frozen against a model that is still improving is legato's measured failure "
          "mode (`L1` committed at c25-c39 against a ballistic clock not plateaued by c72). The "
          "model trains through the between-commit practice and is frozen only for the sweep.")
        W(f"  {'commit':>7} {'cycles':>7} {'e_react':>9} {'e_ball_seg':>11}")
        for c in clock:
            W(f"  {str(c['seam']):>7} {c['cycles']:>7} {c['e_react']:>9.4f} "
              f"{c.get('e_ball_seg', float('nan')):>11.4f}")

    ct = R.get("content", {})
    if ct:
        W("\n=== the stored content, as auditioned (plant replay, cross-state, NESTED build: "
          "candidates AND score set from the configuration that deploys the unit) ===")
        W("  `chosen` is what the committed library actually scores on its own score set; "
          "`per-state oracle` is the ceiling a perfect key could reach on the same pool -- if the "
          "ORACLE is bad the pool is wrong, which no selection can repair (the sibling d1's "
          "seam-1/2 finding).")
        rows = []
        for lvl in ("key", "fix"):
            for n, x in (ct.get(lvl) or {}).items():
                rows.append((f"{lvl}/{n}", x))
        rows += [(n, ct[n]) for n in ("chain", "chain_react", "chain_fixed") if n in ct]
        for n, x in rows:
            W(f"  {n:>12}: {x['n_cand']:>4} cand x {x.get('n_score', 0):>3} states | "
              f"chosen {x.get('chosen', float('nan')):.4f} | best single tape "
              f"{x['best_fixed']:.4f} | per-state oracle {x['per_state_oracle']:.4f} "
              f"({x['oracle_gain']:.2f}x) | {x['n_distinct']}/{len(x['cell_sizes'])} distinct")
    if sw:
        W("\n=== the incumbent's lookahead, per delay (it is given its best at each) ===")
        for sfx, nm in (("p", "predict"), ("", "naive")):
            for d in ds:
                g = sw[str(d)].get("reactive" + sfx, {}).get("look_grid")
                if g:
                    W(f"  [{nm:7s}] D={d:>2}: " + "  ".join(f"{k}:{x:.4f}" for k, x in g.items())
                      + f"   -> chose {sw[str(d)]['reactive' + sfx].get('look')}")

    # ---------------------------------------------------------------- by-segment, and the ledger
    if sw:
        W("\n=== by-segment medians (where in the figure each strategy loses it) ===")
        for sfx, nm in (("p", "predict"), ("", "naive")):
            for d in (ds[0], ds[len(ds) // 2], ds[-1]):
                W(f"  [{nm:7s}] D={d:>2} ({d * 20:>3} ms)")
                for n in STRATS:
                    if n + sfx in sw[str(d)]:
                        bs = sw[str(d)][n + sfx]["by_seg"]
                        W(f"      {n:>12}  " + "  ".join(f"{x:.4f}" for x in bs)
                          + f"   | v_mean {sw[str(d)][n + sfx]['v_mean']:.2f} "
                            f"v_max {sw[str(d)][n + sfx]['v_max']:.2f}")

    lg = R.get("ledger", {})
    if lg:
        W("\n=== the ledger (agent-side is practice + the nested build; every sweep traversal and "
          "every audition is instrument-side) ===")
        for k in ("steps_agent", "steps_instrument", "fb_agent", "fb_instrument",
                  "plans_agent", "plans_instrument", "delib_agent", "aud_agent",
                  "aud_instrument", "t_priced"):
            if k in lg:
                W(f"  {k:>18}: {lg[k]:,.0f}" if isinstance(lg[k], (int, float))
                  else f"  {k:>18}: {lg[k]}")
        nb = lg.get("n_by_kind", {})
        if nb:
            top = sorted(nb.items(), key=lambda kv: -kv[1].get("steps", 0))[:8]
            W("  largest kinds by env steps: "
              + ", ".join(f"{k}={v['steps']:,}" for k, v in top))

    dst = os.path.join(HERE, "results", a.tag)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {dst}/report.txt")


@app.function(memory=16384, timeout=1800, volumes={DATA_DIR: volume})
def _figs(tag: str):
    """Figures are drawn REMOTELY: matplotlib is not installed on the launching client."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    src = os.path.join(DATA_DIR, "practice_presto", tag, "delay_gate", "results.json")
    R = json.load(open(src))
    sw = R["sweep"]
    ds = sorted((int(d) for d in sw), key=int)
    outdir = os.path.join(DATA_DIR, "practice_presto", tag, "figs")
    os.makedirs(outdir, exist_ok=True)
    col = {"reactive": "#c0392b", "live_seg": "#e67e22", "live_chain": "#f1c40f",
           "seg_tape": "#2980b9", "seg_fixed": "#7f8c8d", "chain": "#27ae60",
           "chain_fixed": "#16a085", "chain_react": "#8e44ad"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (sfx, nm) in zip(axes, (("p", "predicting incumbent (efference copy) — PRIMARY"),
                                    ("", "naive incumbent — offbook d0's operator"))):
        for n, c in col.items():
            if n + sfx not in sw[str(ds[0])]:
                continue
            ax.plot([d * 20 for d in ds], [sw[str(d)][n + sfx]["e"] for d in ds], "o-",
                    color=c, lw=1.8, label=f"{n} ({sw[str(ds[0])][n + sfx]['fb']:.0f} fb)")
        ax.axhline(R["ref_play"], color="k", ls="--", lw=1.2, label="ref_play (guard)")
        ax.axhline(R["hold"]["e"], color="k", ls=":", lw=1.0, label="do-nothing floor")
        ax.set_xlabel("observation delay (ms)")
        ax.set_title(nm, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean piece error (m)")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"presto {tag}: piece error vs reflex delay, by feedback consumption")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_sweep.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(7, 5))
    names = [n for n in col if n + "p" in sw[str(ds[0])]]
    fb = [sw[str(ds[0])][n + "p"]["fb"] for n in names]
    deg = [sw[str(ds[-1])][n + "p"]["e"] / max(sw[str(ds[0])][n + "p"]["e"], 1e-9) for n in names]
    ax.scatter(fb, deg, c=[col[n] for n in names], s=70)
    for n, x, y in zip(names, fb, deg):
        ax.annotate(n, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("feedback events per traversal")
    ax.set_ylabel(f"degradation factor, D=0 -> D={ds[-1]}")
    ax.set_title("Is delay sensitivity still ordered by feedback consumption?")
    ax.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(os.path.join(outdir, "fig2_ordering.png"), dpi=150)
    volume.commit()
    return f"wrote {outdir}"


@app.local_entrypoint()
def figs(tag: str = "p0"):
    print(_figs.remote(tag))


if __name__ == "__main__":
    main()
