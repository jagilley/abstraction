"""Reduce an S2 run: which of the two suspects S1's behavioural negative belongs to.

Fetches `/data/practice_span/<tag>/results.json` and prints

  * FIDELITY -- the regenerated reference set and phrase plans against S1's stored values. These
    must be zero; the whole run is a re-measurement of S1's models, not a second run.
  * THE THREE DIVERGENCE SOURCES -- `fixed` (frozen cycle-0 reference commands, S1's number),
    `current` (a reactive traversal under this snapshot, same launch states), `plan` (the
    snapshot's own phrase plan). Suspect (1), corridor narrowing, predicts `plan` decaying while
    `fixed`/`current` improve.
  * THE EXPLOITATION GAP -- true minus predicted arrival at each seam, under the plan the CEM
    actually chose. How much of what the model sold did not survive contact with the plant.
  * ONE-STEP FM ERROR on-corridor (matched motor noise) vs off-corridor, and the corridor's own
    geometry.
  * CAL-P PER SNAPSHOT -- true and predicted piece error against planner size. Suspect (2),
    starvation, predicts the far-end degradation vanishing as the search grows; suspect (1)
    predicts the model's BELIEF falling with search while the truth does not.

Usage (from experiments/):
    python3 mjc/practice/span/analyze_s2.py --tag S2 --fetch
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ("fixed", "current", "plan")


def fetch(tag):
    dst = os.path.join(HERE, "results", tag)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                    f"practice_span/{tag}", dst], check=False,
                   env={**os.environ,
                        "MODAL_PROFILE": os.environ.get("MODAL_PROFILE", "chromatic")})


def load(tag):
    root = os.path.join(HERE, "results", tag)
    for p in (os.path.join(root, tag, "results.json"), os.path.join(root, "results.json")):
        if os.path.isfile(p):
            return json.load(open(p))
    raise SystemExit(f"no results.json under {root}")


def rho(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(np.asarray(x, float), np.asarray(y, float))
    return (float(r), float(p)) if np.isfinite(r) else (float("nan"), float("nan"))


def hv(row, t="0.05"):
    return float(row["h"][t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="S2")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    d = load(a.tag)
    S = d["snapshots"]
    c = np.array([s["cycle"] for s in S], float)
    print(f"[load] tag={a.tag} complete={d.get('complete')} snapshots={len(S)} "
          f"calp_cycles={len(d.get('calp', []))}")

    # ---------------------------------------------------------------- fidelity
    fid = [s["fidelity_e_seam"] for s in S if s.get("fidelity_e_seam")]
    mx = max((max(abs(v) for v in f) for f in fid), default=float("nan"))
    print(f"\n=== FIDELITY === reference launch tips vs S1: "
          f"{d['reference']['launch_fidelity']:.2e} m | regenerated phrase plans reproduce S1's "
          f"e_seam at {len(fid)} cycles, max |diff| {mx:.2e} m | launch-state match across "
          f"snapshots {max(s['launch_match'] for s in S):.2e}")

    # ---------------------------------------------------------------- the three sources
    print("\n=== THREE DIVERGENCE SOURCES (composition horizon @0.05 m, steps) ===")
    print(f"{'cyc':>4} {'fixed':>6} {'current':>8} {'plan':>6} {'plan-fixed':>11} | "
          f"{'1step corr':>10} {'1step plan':>10} {'ratio':>6} | {'corr logdet':>11} "
          f"{'plan-nn':>8} | {'e_react':>8} {'e_piece':>8}")
    for s in S:
        print(f"{s['cycle']:4.0f} {hv(s['div_fixed']):6.0f} {hv(s['div_current']):8.0f} "
              f"{hv(s['div_plan']):6.0f} {hv(s['div_plan']) - hv(s['div_fixed']):11.0f} | "
              f"{s['fm_err_corridor_perf']:10.4f} {s['fm_err_plan']:10.4f} "
              f"{s['fm_err_ratio_perf']:6.2f} | {s['corridor']['logdet']:11.2f} "
              f"{s['corridor']['plan_nn_med']:8.3f} | {s['e_react']:8.4f} {s['e_piece']:8.4f}")
    print("\n  trends (Spearman vs cycle):")
    for nm in SRC:
        for t in ("0.02", "0.05"):
            y = [hv(s[f"div_{nm}"], t) for s in S]
            r, p = rho(c, y)
            print(f"    horizon@{t} {nm:>8}: {np.median(y[:6]):5.1f} -> {np.median(y[-6:]):5.1f}"
                  f"   rho={r:+.3f} p={p:.2e}")
    for k, lbl in (("fm_err_corridor_perf", "1-step err, corridor (perf noise)"),
                   ("fm_err_plan", "1-step err, plan trajectory"),
                   ("fm_err_ratio_perf", "ratio plan/corridor"),
                   ("e_piece", "phrase-plan piece error")):
        y = [s[k] for s in S]
        r, p = rho(c, y)
        print(f"    {lbl:34s}: {np.median(y[:6]):.4f} -> {np.median(y[-6:]):.4f}   "
              f"rho={r:+.3f} p={p:.2e}")
    for k in ("logdet", "trace", "tip_area", "speed", "plan_nn_med", "plan_nn_p90"):
        y = [s["corridor"][k] for s in S]
        r, p = rho(c, y)
        print(f"    corridor {k:<12s}: {np.median(y[:6]):+.4f} -> {np.median(y[-6:]):+.4f}   "
              f"rho={r:+.3f} p={p:.2e}")

    # ---------------------------------------------------------------- exploitation gap
    print("\n=== THE EXPLOITATION GAP (true minus predicted arrival, m) ===")
    print(f"{'cyc':>4} | {'W2 true':>8} {'W2 pred':>8} {'gap':>7} | {'W3 true':>8} {'W3 pred':>8} "
          f"{'gap':>7} | {'W0 true':>8} {'W0 pred':>8} {'gap':>7}")
    for s in S:
        t_, p_, g_ = s["e_seam"], s["e_seam_pred"], s["exploit_gap"]
        print(f"{s['cycle']:4.0f} | " + " | ".join(
            f"{t_[k]:8.3f} {p_[k]:8.3f} {g_[k]:7.3f}" for k in range(3)))
    for k, nm in enumerate(("W2", "W3", "W0")):
        for fld, lbl in (("e_seam", "true"), ("e_seam_pred", "predicted"), ("exploit_gap", "gap")):
            y = [s[fld][k] for s in S]
            r, p = rho(c, y)
            print(f"    {nm} {lbl:<10s}: {np.median(y[:6]):.3f} -> {np.median(y[-6:]):.3f}   "
                  f"rho={r:+.3f} p={p:.2e}")

    # ---------------------------------------------------------------- CAL-P
    if d.get("calp"):
        print("\n=== CAL-P PER SNAPSHOT (true piece error / what the model believed) ===")
        g0 = d["calp"][0]["grid"]
        hdr = "  ".join(f"{g['k_shoot']}x{g['cem_iters']}".rjust(15) for g in g0)
        print(f"{'cyc':>4}  {hdr}   | still improving? | W0 at top")
        for r_ in d["calp"]:
            cells = "  ".join(f"{g['e_piece']:6.3f}/{g['e_piece_pred']:<8.3f}".rjust(15)
                              for g in r_["grid"])
            print(f"{r_['cycle']:4d}  {cells}   | {str(r_['still_improving']):>5} "
                  f"(gain {r_['last_step_gain']:+.4f}) | {r_['w0_at_top']:.3f}")
        print("\n  per-cycle: does more search help the TRUTH, and does it help the BELIEF?")
        for r_ in d["calp"]:
            v = [g["e_piece"] for g in r_["grid"]]
            vp = [g["e_piece_pred"] for g in r_["grid"]]
            w0 = [g["e_seam"][-1] for g in r_["grid"]]
            print(f"    c{r_['cycle']:<3d} true {v[0]:.3f} -> {v[-1]:.3f} "
                  f"({'better' if v[-1] < v[0] else 'WORSE'})   believed {vp[0]:.3f} -> "
                  f"{vp[-1]:.3f} ({'better' if vp[-1] < vp[0] else 'worse'})   "
                  f"W0 {w0[0]:.3f} -> {w0[-1]:.3f}   best {min(v):.3f} at {r_['best_at']}")

    if not a.no_figs:
        figures(d, os.path.join(HERE, "results", a.tag, "figs"))


def figures(d, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    S = d["snapshots"]
    c = np.array([s["cycle"] for s in S], float)

    # ---- fig 1: the three sources ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for nm, col in zip(SRC, ("#1f4e79", "#4a8", "#a33")):
        axes[0].plot(c, [hv(s[f"div_{nm}"]) for s in S], color=col, lw=1.8, marker="o", ms=3,
                     label=f"{nm} commands @0.05 m")
        axes[0].plot(c, [hv(s[f"div_{nm}"], "0.02") for s in S], color=col, lw=1.0, ls="--",
                     alpha=0.7, label=f"{nm} @0.02 m")
    axes[0].set_xlabel("practice cycle"); axes[0].set_ylabel("composition horizon (steps)")
    axes[0].set_title("Where the commands came from decides the horizon")
    axes[0].legend(fontsize=7.5, ncol=2)
    ax = axes[1]
    ax.plot(c, [s["fm_err_corridor_perf"] for s in S], color="#4a8", lw=1.8, marker="o", ms=3,
            label="1-step FM error, on corridor")
    ax.plot(c, [s["fm_err_plan"] for s in S], color="#a33", lw=1.8, marker="s", ms=3,
            label="1-step FM error, on the plan's trajectory")
    ax.set_yscale("log"); ax.set_xlabel("practice cycle"); ax.set_ylabel("median |Δ̂ − Δ| (state)")
    ax.set_title("One-step accuracy, on and off the corridor")
    ax2 = ax.twinx()
    ax2.plot(c, [s["corridor"]["plan_nn_med"] for s in S], color="#888", lw=1.2, ls=":",
             label="plan's distance to nearest corridor state")
    ax2.set_ylabel("normalised state distance", color="#888")
    ax2.tick_params(axis="y", colors="#888")
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "s2fig1_sources.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: the exploitation gap -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for k, (nm, col) in enumerate(zip(("W2 (step 20)", "W3 (step 40)", "W0 (step 60)"),
                                      ("#1f4e79", "#c60", "#a33"))):
        axes[0].plot(c, [s["e_seam"][k] for s in S], color=col, lw=1.8, marker="o", ms=3,
                     label=f"{nm} true")
        axes[0].plot(c, [s["e_seam_pred"][k] for s in S], color=col, lw=1.2, ls="--", alpha=0.8,
                     label=f"{nm} model's prediction")
        axes[1].plot(c, [s["exploit_gap"][k] for s in S], color=col, lw=1.8, marker="o", ms=3,
                     label=nm)
    axes[0].set_yscale("log"); axes[0].set_title("What the plan delivered vs what it promised")
    axes[0].set_ylabel("arrival error at the waypoint (m, log)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Exploitation gap (true − predicted)")
    axes[1].set_ylabel("m")
    for x in axes:
        x.set_xlabel("practice cycle"); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "s2fig2_exploitation.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: CAL-P ----------------------------------------------------------------------
    if d.get("calp"):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
        cm = plt.get_cmap("viridis")
        n = len(d["calp"])
        for j, r_ in enumerate(d["calp"]):
            x = [g["k_shoot"] * g["cem_iters"] for g in r_["grid"]]
            col = cm(j / max(1, n - 1))
            axes[0].plot(x, [g["e_piece"] for g in r_["grid"]], color=col, lw=1.7, marker="o",
                         ms=4, label=f"c{r_['cycle']}")
            axes[0].plot(x, [g["e_piece_pred"] for g in r_["grid"]], color=col, lw=1.1, ls="--",
                         alpha=0.7)
            axes[1].plot(x, [g["e_seam"][-1] for g in r_["grid"]], color=col, lw=1.7, marker="o",
                         ms=4, label=f"c{r_['cycle']}")
        chosen = d["src_config"]["calp"]["3"]
        for x_ in axes:
            x_.axvline(chosen[0] * chosen[1], color="k", lw=0.9, ls=":")
            x_.set_xscale("log"); x_.set_xlabel("CEM budget (k_shoot × cem_iters)")
            x_.legend(fontsize=7.5, ncol=2)
        axes[0].set_ylabel("phrase piece error (m)")
        axes[0].set_title("solid = true, dashed = what the model believed\n"
                          "(dotted line = CAL-P's chosen sizing, used throughout S1)")
        axes[1].set_ylabel("W0 arrival error (m)")
        axes[1].set_title("The far end, against search budget")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "s2fig3_calp.png"), dpi=150)
        plt.close(fig)
    print(f"\n[figs] wrote to {outdir}")


if __name__ == "__main__":
    main()
