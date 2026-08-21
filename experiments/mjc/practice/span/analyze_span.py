"""Reduce a `span` run: the two horizon trajectories against the ladder, and the per-step curves.

Fetches `/data/practice_span/<tag>/results.json` and prints

  * THE HORIZON LADDER -- imagination horizon (fixed executed reference, and the on-policy
    variant) and executed span, per cycle, beside `e_react` (the task metric) and `e_ball_seg`.
  * THE ANCHORS -- the same two quantities for `legato` G5a's two models (stale `fm0`,
    corridor-matched ceiling) measured on THIS reference set, so the numbers are comparable to
    the gate's 21 / 23 / 44 / 58.
  * THE FLOORS -- what a MEASURED chain re-executed open-loop costs (motor noise only), what the
    reference traversal's own curves look like, and the plant's replay-chaos floor.
  * THRESHOLD SENSITIVITY -- every horizon at 0.01 / 0.02 / 0.05 / 0.10 / 0.20 m, because a
    horizon is a level crossing and the level was chosen by `legato`, not by nature.

and writes figures to `results/<tag>/figs/`.

READING RULES.
  * A horizon equal to (probe length + 1) is a MEASUREMENT CEILING, not a result. The imagination
    probe is `H_ref` steps (two laps, 134); the executed span is one phrase (60). Ceilings are
    printed as `>N` and drawn clipped.
  * Single seed. The shape of the trajectory is the claim; individual cycles wobble.
  * `e_react` is at plateau from cycle 0 on this piece by construction (`legato` L1), so "flat"
    is the premise of the measurement, not a finding of it.

Usage (from experiments/):
    python3 mjc/practice/span/analyze_span.py --tag S1 --fetch
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THRS = ["0.01", "0.02", "0.05", "0.1", "0.2"]


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
            return json.load(open(p)), os.path.dirname(p)
    raise SystemExit(f"no results.json under {root}")


def h(row, thr, cap=None):
    """A horizon, printed as `>N` when it is the probe's own length + 1 (a measurement ceiling)."""
    if row is None:
        return "-"
    v = row["h"][thr]
    n = len(row["med"])
    return f">{n}" if v > n else str(v)


def hv(row, thr):
    return float(row["h"][thr])


def ceiled(row, thr):
    return row["h"][thr] > len(row["med"])


# --------------------------------------------------------------------------- tables

def table_ladder(d):
    snaps = d["snapshots"]
    lad = {r["cycle"]: r for r in d["ladder"]}
    print("\n=== HORIZON LADDER "
          "(imag = FM composition horizon along the fixed executed reference, 134 steps; "
          "span = one live phrase plan flown open-loop, 60 steps) ===")
    print(f"{'cyc':>4} | {'imag@.05':>8} {'imag@.02':>8} {'imag@.01':>8} {'onpol@.05':>9} | "
          f"{'exc@.05':>7} {'exc@.02':>7} {'path@.05':>8} {'ref@.05':>7} {'e_phrase':>8} | "
          f"{'e_react':>8} {'e_ball_seg':>10} {'s/cyc':>6}")
    for s in snaps:
        c = s["cycle"]
        im, on, sp = s["imagination"], s.get("imagination_onpolicy"), s.get("span")
        L = lad.get(c, {})
        row = (f"{c:4d} | {h(im,'0.05'):>8} {h(im,'0.02'):>8} {h(im,'0.01'):>8} "
               f"{(h(on,'0.05') if on else '-'):>9} | ")
        if sp:
            row += (f"{h(sp.get('goal_excess'),'0.05'):>7} {h(sp.get('goal_excess'),'0.02'):>7} "
                    f"{h(sp['path'],'0.05'):>8} {h(sp['ref'],'0.05'):>7} {sp['e_piece']:8.4f} | ")
        else:
            row += f"{'-':>7} {'-':>7} {'-':>8} {'-':>7} {'-':>8} | "
        row += (f"{L.get('e_react', float('nan')):8.4f} "
                f"{L.get('e_ball_seg', float('nan')):10.4f} "
                f"{s.get('wall', float('nan')):6.1f}")
        print(row)


def table_anchors(d):
    print("\n=== ANCHORS (the two models `legato` G5a measured, on THIS reference set) ===")
    print(f"{'model':>10} | " + " ".join(f"imag@{t:>4}" for t in THRS) + " | "
          + " ".join(f"exc@{t:>4}" for t in THRS) + " | e_phrase")
    for nm, a in d["anchors"].items():
        print(f"{nm:>10} | " + " ".join(f"{h(a['imagination'], t):>9}" for t in THRS) + " | "
              + " ".join(f"{h(a['span'].get('goal_excess'), t):>8}" for t in THRS)
              + f" | {a['span']['e_piece']:.4f}")
    print("  (`legato` gate G5a, executed source: stale 21 (l0) / 23 (l2) at 0.05 m, "
          "ceiling 44 (l0) / 58 (l2); its probe stopped at 60 steps)")


def table_floors(d):
    print("\n=== FLOORS (measured once; every span curve is competing against these) ===")
    F = d["floors"]
    for nm in ("frozen_noisy", "frozen_clean", "replay", "reference"):
        if nm not in F:
            continue
        r = F[nm]
        print(f"  {nm:14s} vs-ref " + " ".join(f"{t}:{h(r['ref'], t):>5}" for t in THRS))
        print(f"  {'':14s} excess " + " ".join(f"{t}:{h(r.get('goal_excess'), t):>5}" for t in THRS)
              + "   path " + " ".join(f"{t}:{h(r['path'], t):>5}" for t in THRS))
    ref = d["reference"]
    print(f"  reference traversal: e_piece lap1 {ref['e_piece_lap1']:.4f} lap2 "
          f"{ref['e_piece_lap2']:.4f} | launch tip spread {ref['launch_tip_spread']:.4f} | "
          f"replay drift at step {ref['H_ref']} = {ref['replay_repro']:.2e} m")


def table_drift(d):
    print("\n=== LAUNCH-STATE DRIFT (the reference launch set is frozen at cycle 0) ===")
    print(f"{'cyc':>4} {'drift(m)':>9} {'spread(m)':>10} {'e_react':>8} {'e_ball_seg':>10}")
    for r in d["ladder"]:
        print(f"{r['cycle']:4d} {r.get('launch_drift', float('nan')):9.4f} "
              f"{r.get('launch_tip_spread', float('nan')):10.4f} {r['e_react']:8.4f} "
              f"{r['e_ball_seg']:10.4f}")


def summarise(d):
    snaps, lad = d["snapshots"], d["ladder"]
    cyc = np.array([s["cycle"] for s in snaps])
    im05 = np.array([hv(s["imagination"], "0.05") for s in snaps])
    im02 = np.array([hv(s["imagination"], "0.02") for s in snaps])
    sp = [(s["cycle"], hv(s["span"]["goal_excess"], "0.05"), hv(s["span"]["path"], "0.05"))
          for s in snaps if "span" in s and "goal_excess" in s["span"]]
    H_seg = d["config"]["h_seg"][0]
    print("\n=== SUMMARY ===")
    k = max(3, len(cyc) // 5)
    print(f"  imagination h@0.05: first {k} cycles median {np.median(im05[:k]):.1f}  "
          f"last {k} median {np.median(im05[-k:]):.1f}  (min {im05.min():.0f} max {im05.max():.0f})")
    print(f"  imagination h@0.02: first {k} median {np.median(im02[:k]):.1f}  "
          f"last {k} median {np.median(im02[-k:]):.1f}")
    if sp:
        a = np.array(sp)
        first = [c for c, e, _ in sp if e > H_seg]
        print(f"  executed span (goal-excess h@0.05): first {max(2, len(sp)//5)} median "
              f"{np.median(a[:max(2, len(sp)//5), 1]):.1f}  last median "
              f"{np.median(a[-max(2, len(sp)//5):, 1]):.1f}")
        print(f"  first cycle whose executed span exceeds ONE SEGMENT ({H_seg} steps): "
              f"{first[0] if first else 'never'}")
        seam = np.array([s["span"]["e_seam"] for s in snaps if "span" in s])
        scy2 = np.array([s["cycle"] for s in snaps if "span" in s])
        for t in (0.05, 0.10, 0.15, 0.20):
            nd = np.array([int(np.argmax(np.append(e > t, True))) for e in seam])
            hit = scy2[nd >= 2]
            print(f"    waypoints delivered within {t:.2f} m: first8 median {np.median(nd[:8]):.1f}"
                  f"  last8 median {np.median(nd[-8:]):.1f}  max {nd.max()}"
                  f"  first cycle delivering >=2: {hit[0] if len(hit) else 'never'}")
    er = np.array([r["e_react"] for r in lad])
    print(f"  e_react: median {np.median(er):.4f}  min {er.min():.4f}  max {er.max():.4f}  "
          f"(sd {er.std():.4f}) -- the task metric, flat by construction on this piece")
    eb = np.array([r["e_ball_seg"] for r in lad])
    print(f"  e_ball_seg: first {np.median(eb[:3]):.4f} -> last {np.median(eb[-3:]):.4f}")
    if "wall" in snaps[-1]:
        w = np.array([s["wall"] for s in snaps if s.get("wall")])
        print(f"  wall clock: {w.mean():.1f} s/cycle, {w.sum() / 3600:.2f} h of practice loop")


# --------------------------------------------------------------------------- figures

def figures(d, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    snaps, lad = d["snapshots"], d["ladder"]
    H_seg = d["config"]["h_seg"][0]
    H_phr = d["reference"]["h_phrase"]
    H_ref = d["reference"]["H_ref"]
    F, A = d["floors"], d["anchors"]

    # ---- fig 1: the two horizons against the task metric ----------------------------------
    cyc = np.array([s["cycle"] for s in snaps])
    im05 = np.array([hv(s["imagination"], "0.05") for s in snaps])
    im02 = np.array([hv(s["imagination"], "0.02") for s in snaps])
    on05 = np.array([hv(s["imagination_onpolicy"], "0.05") if "imagination_onpolicy" in s
                     else np.nan for s in snaps])
    scy = np.array([s["cycle"] for s in snaps if "span" in s])
    spans_ = [s for s in snaps if "span" in s and "goal_excess" in s["span"]]
    scy = np.array([s["cycle"] for s in spans_])
    sx05 = np.array([hv(s["span"]["goal_excess"], "0.05") for s in spans_])
    spa05 = np.array([hv(s["span"]["path"], "0.05") for s in spans_])

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(cyc, im05, lw=1.8, color="#1f4e79", label="imagination horizon @0.05 m")
    ax.plot(cyc, im02, lw=1.2, color="#5b8fc9", ls="--", label="imagination horizon @0.02 m")
    ax.plot(cyc, on05, lw=0.9, color="#9dc3e6", alpha=0.8, label="imagination, on-policy cmds")
    ax.plot(scy, sx05, lw=1.8, color="#a33", marker="o", ms=3,
            label="executed span @0.05 m (goal-excess)")
    ax.plot(scy, spa05, lw=1.2, color="#d98", ls="--", label="executed span @0.05 m (cross-track)")
    for nm, col in (("stale", "#888"), ("ceiling", "#3a3")):
        ax.axhline(hv(A[nm]["imagination"], "0.05"), color=col, lw=1.0, ls=":",
                   label=f"anchor: {nm} FM imagination")
    ax.axhline(H_seg, color="k", lw=0.7, alpha=0.4)
    ax.axhline(H_phr, color="k", lw=0.7, alpha=0.4)
    ax.text(cyc[-1], H_seg, " one segment", va="bottom", ha="right", fontsize=8, alpha=0.6)
    ax.text(cyc[-1], H_phr, " one phrase (span probe ceiling)", va="bottom", ha="right",
            fontsize=8, alpha=0.6)
    fz = F["frozen_noisy"]["goal_excess"]["h"]["0.05"]
    ax.axhline(fz, color="#a33", lw=1.0, ls=":", alpha=0.7,
               label=f"frozen measured chain reaches {fz} (executed-span ceiling)")
    ax.set_xlabel("practice cycle"); ax.set_ylabel("horizon (control steps)")
    ax.set_ylim(0, min(H_ref + 6, max(im05.max(), H_phr) * 1.25))
    ax.legend(fontsize=7.5, loc="upper center", ncol=3, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.13))
    ax2 = ax.twinx()
    lc = np.array([r["cycle"] for r in lad])
    ax2.plot(lc, [r["e_react"] for r in lad], color="#c60", lw=1.4, marker="s", ms=3,
             label="e_react (task metric)")
    ax2.plot(lc, [r["e_ball_seg"] for r in lad], color="#c60", lw=1.0, ls="--", alpha=0.6,
             label="e_ball_seg")
    ax2.set_yscale("log"); ax2.set_ylabel("piece error (m, log)", color="#c60")
    ax2.tick_params(axis="y", colors="#c60")
    ax2.legend(fontsize=8, loc="lower right")
    ax.set_title("Composition horizon vs practice, against the reactive task metric")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_horizons.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: the imagination divergence curves at early / mid / late ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    picks = [snaps[i] for i in np.unique(np.linspace(0, len(snaps) - 1, 6).astype(int))]
    cm = plt.get_cmap("viridis")
    for j, s in enumerate(picks):
        axes[0].plot(np.arange(1, H_ref + 1), s["imagination"]["med"],
                     color=cm(j / max(1, len(picks) - 1)), lw=1.4, label=f"c{s['cycle']}")
    for nm, col in (("stale", "#888"), ("ceiling", "#3a3")):
        axes[0].plot(np.arange(1, H_ref + 1), A[nm]["imagination"]["med"], color=col, ls=":",
                     lw=1.6, label=f"{nm} FM")
    axes[0].plot(np.arange(1, H_ref + 1), F["replay"]["ref"]["med"], color="#bbb", lw=0.8,
                 label="plant replay chaos floor")
    axes[0].set_yscale("log"); axes[0].axhline(0.05, color="k", lw=0.6, alpha=0.4)
    axes[0].axhline(0.02, color="k", lw=0.6, alpha=0.2)
    axes[0].axvline(H_phr, color="k", lw=0.6, alpha=0.3)
    axes[0].set_xlabel("rollout step"); axes[0].set_ylabel("median |FM tip - true tip| (m)")
    axes[0].set_title("Imagination: open-loop FM rollout vs the plant")
    axes[0].legend(fontsize=7, ncol=2)

    spans = spans_
    if not spans:
        print("[figs] no executed-span probes carry `goal_excess` (pre-2026-08-20 run); "
              "fig2/fig4 skipped")
        return
    picks = [spans[i] for i in np.unique(np.linspace(0, len(spans) - 1, 6).astype(int))]
    for j, s in enumerate(picks):
        axes[1].plot(np.arange(1, H_phr + 1), s["span"]["goal_excess"]["med"],
                     color=cm(j / max(1, len(picks) - 1)), lw=1.4, label=f"c{s['cycle']}")
    axes[1].plot(np.arange(1, H_phr + 1), F["frozen_noisy"]["goal_excess"]["med"][:H_phr],
                 color="#000", lw=1.4, ls="--", label="frozen measured chain (noise floor)")
    axes[1].axhline(0.05, color="k", lw=0.6, alpha=0.4)
    for x in d["reference"]["seg_hi"]:
        axes[1].axvline(x, color="k", lw=0.6, alpha=0.25)
    axes[1].set_xlabel("step within the phrase")
    axes[1].set_ylabel("excess distance to the step's waypoint (m)")
    axes[1].set_title("Executed span: one live phrase plan, flown open-loop")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_curves.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: threshold sensitivity ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for t, col in zip(THRS, plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(THRS)))):
        axes[0].plot(cyc, [hv(s["imagination"], t) for s in snaps], color=col, lw=1.3,
                     label=f"{t} m")
        axes[1].plot(scy, [hv(s["span"]["goal_excess"], t) for s in spans_],
                     color=col, lw=1.3, marker="o", ms=2.5, label=f"{t} m")
    axes[0].axhline(H_ref, color="k", ls=":", lw=0.8)
    axes[0].set_title("imagination horizon, by threshold"); axes[0].set_ylabel("steps")
    axes[1].axhline(H_phr, color="k", ls=":", lw=0.8)
    axes[1].set_title("executed span (goal-excess), by threshold")
    for a in axes:
        a.set_xlabel("practice cycle"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_thresholds.png"), dpi=150)
    plt.close(fig)

    # ---- fig 4: the raw within-phrase profile (the paper's within-trial figure's analogue) --
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, s in enumerate(picks):
        axes[0].plot(np.arange(1, H_phr + 1), s["span"]["goal"]["med"],
                     color=cm(j / max(1, len(picks) - 1)), lw=1.3, label=f"c{s['cycle']}")
        axes[1].plot(np.arange(1, H_phr + 1), s["span"]["ref"]["med"],
                     color=cm(j / max(1, len(picks) - 1)), lw=1.3, label=f"c{s['cycle']}")
    axes[0].plot(np.arange(1, H_phr + 1), F["reference"]["goal"]["med"][:H_phr], color="#000",
                 lw=1.5, ls="--", label="reference reactive traversal")
    axes[1].plot(np.arange(1, H_phr + 1), F["frozen_noisy"]["ref"]["med"][:H_phr], color="#000",
                 lw=1.5, ls="--", label="frozen measured chain")
    for a, ttl, yl in ((axes[0], "distance to the step's waypoint (raw sawtooth)",
                        "|tip - goal| (m)"),
                       (axes[1], "distance from the competent reference trajectory",
                        "|tip - reference tip| (m)")):
        for x in d["reference"]["seg_hi"]:
            a.axvline(x, color="k", lw=0.6, alpha=0.25)
        a.set_xlabel("step within the phrase"); a.set_ylabel(yl); a.set_title(ttl)
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig4_profile.png"), dpi=150)
    plt.close(fig)

    # ---- fig 5: the executed span as WAYPOINTS DELIVERED -------------------------------------
    # A `goal_excess` crossing at step 4 is a stereotyped mid-leg route difference (the open-loop
    # plan does not rush to the waypoint the way the reactive controller does), NOT a failure --
    # the unit still arrives at W2 within 0.05 m. What the committed unit actually delivers is
    # waypoints, so read the span off the seam errors.
    scy2 = np.array([s["cycle"] for s in spans])
    seam = np.array([s["span"]["e_seam"] for s in spans])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for k, (col, nm) in enumerate(zip(["#1f4e79", "#c60", "#a33"],
                                      ["W2 (step 20)", "W3 (step 40)", "W0 (step 60)"])):
        axes[0].plot(scy2, seam[:, k], color=col, lw=1.5, marker="o", ms=3, label=nm)
    for t, ls in ((0.05, ":"), (0.15, "--")):
        axes[0].axhline(t, color="k", lw=0.8, ls=ls, alpha=0.5)
    axes[0].set_yscale("log"); axes[0].set_xlabel("practice cycle")
    axes[0].set_ylabel("open-loop arrival error at the waypoint (m, log)")
    axes[0].set_title("What one live phrase plan actually delivers")
    axes[0].legend(fontsize=8)
    for t, col in ((0.05, "#1f4e79"), (0.10, "#4a8"), (0.15, "#c60"), (0.20, "#a33")):
        nd = [int(np.argmax(np.append(e > t, True))) for e in seam]
        axes[1].step(scy2, nd, where="mid", color=col, lw=1.4, label=f"within {t} m")
    axes[1].set_ylim(-0.2, 3.2); axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_xlabel("practice cycle"); axes[1].set_ylabel("consecutive waypoints delivered")
    axes[1].set_title("Executed span, in units of the piece")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig5_seams.png"), dpi=150)
    plt.close(fig)
    print(f"\n[figs] wrote 5 figures to {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="S1")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    d, root = load(a.tag)
    print(f"[load] tag={a.tag} complete={d.get('complete')} snapshots={len(d['snapshots'])} "
          f"ladder={len(d['ladder'])} cycles_configured={d['config']['n_cycles']}")
    table_anchors(d)
    table_floors(d)
    table_ladder(d)
    table_drift(d)
    summarise(d)
    if not a.no_figs:
        figures(d, os.path.join(HERE, "results", a.tag, "figs"))


if __name__ == "__main__":
    main()
