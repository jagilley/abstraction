"""Local post-processing for the étude / E-gate runs (no compute).

Reads `results/<tag>/<arm>.json` (the local mirror written by the entrypoint) or, with `--fetch`,
pulls `/practice_etude/<tag>/` off the Modal volume first. Prints:

  * the instrument checks — stale vs ceiling reference ladders per segment, so the usable range on
    the drilled segment is visible before anything is read off the arms;
  * the metering trace — per-segment e_k (at-tempo run-through boundary error), b_k, δ_k;
  * the counterfactual δ-silence panel: for a grid of (c, c_v, W) the cycle at which the gate WOULD
    have fired, computed off any arm whose routing never changed (normally `never`);
  * the priced-time grade — (cumulative practice traversal time, waypoint error at performance
    tempo) per milestone per arm, plus time-to-criterion, plus the accuracy-only readout that
    Cut 4b makes reactive win by construction.

Figures under `figures/<tag>/`.

Usage (from experiments/):
    python3 mjc/practice/etude/analyze_etude.py --tag eg_s0 [--fetch] [--figures]
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ORDER = ["never", "sched_late", "delta_gate", "sched_early", "sloppy",
             "gate_select", "gate_plan", "gate_over_select", "sloppy_select", "sched_early_select",
             "gate_select_x", "gate_plan_eval", "gate_over_select_x", "sloppy_select_x",
             "seq_seam", "seq_practice", "seam_drill", "seam_drill_fix"]
COL = {"never": "#495057", "delta_gate": "#2f9e44", "sched_early": "#e8590c",
       "sched_late": "#7048e8", "sloppy": "#c92a2a",
       "gate_select": "#2f9e44", "gate_plan": "#1971c2", "gate_over_select": "#e8590c",
       "sloppy_select": "#c92a2a", "sched_early_select": "#868e96",
       "gate_select_x": "#2f9e44", "gate_plan_eval": "#1971c2",
       "gate_over_select_x": "#e8590c", "sloppy_select_x": "#c92a2a",
       "seq_seam": "#2f9e44", "seq_practice": "#c92a2a", "seam_drill": "#1971c2",
       "seam_drill_fix": "#1971c2"}


def load(tag, fetch=False):
    d = os.path.join(HERE, "results", tag)
    if fetch or not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "--force", "mujoco-control-data",
                        f"/practice_etude/{tag}", d], check=False)
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            if not f.endswith(".json"):
                continue
            p = os.path.join(root, f)
            try:
                R = json.load(open(p))
            except Exception:
                continue
            if "arm" in R:
                out[R["arm"]] = R
    return out


def perf_R(R):
    return R["config"]["ladder"][-1]


def seg_err(rec, R):
    return np.array(rec[f"R{R}"]["wp_err"], float)


def summarize(res):
    any_arm = next(iter(res.values()))
    cfg = any_arm["config"]
    K = len(cfg["waypoints"]) - 1
    drill = cfg["drill_seg"]
    Rp = perf_R(any_arm)
    lad = cfg["ladder"]

    print(f"\n=== piece: K={K} segments, H={cfg['seg_H']}, drill={drill}, "
          f"regions={cfg['regions']} | tempo ladder R={lad} (R{Rp} = performance tempo) ===")
    print(f"    time model: dt_ctrl={cfg['frame_skip']*0.002:.3f}s/step, d_fb={cfg['d_fb']}s/feedback event")
    print(f"    gate: |b-e| < {cfg['sil_c']}*scale and sd < {cfg['sil_cv']}*scale for W={cfg['sil_W']}; "
          f"bench EWMA alpha={cfg['bench_alpha']}")

    print("\n--- instrument: reference ladders (median over eval performers, per segment) ---")
    for nm in ("ref_stale", "ref_ceiling"):
        if nm not in any_arm:
            continue
        for R in lad:
            v = np.median(np.array(any_arm[nm][f"R{R}"], float), 0)
            print(f"  {nm:12s} R{R:<3d} " + "  ".join(f"s{k}={v[k]:.4f}" for k in range(K)))
    fs_ = any_arm["setup"]
    print(f"  FM probe   stale  region={fs_['fm_stale']['region']:.4f} clean={fs_['fm_stale']['clean']:.4f}")
    print(f"  FM probe   ceil   region={fs_['fm_ceiling']['region']:.4f} clean={fs_['fm_ceiling']['clean']:.4f}")
    st = np.median(np.array(any_arm["ref_stale"][f"R{Rp}"], float), 0)[drill]
    ce = np.median(np.array(any_arm["ref_ceiling"][f"R{Rp}"], float), 0)[drill]
    print(f"  >> drilled segment at performance tempo: stale {st:.4f} -> ceiling {ce:.4f} "
          f"(range {st-ce:+.4f})")
    return cfg, K, drill, Rp, lad, st, ce


def metering_table(res, K, drill):
    print("\n--- metering trace on the drilled segment (e = at-tempo run-through boundary error) ---")
    for arm in [a for a in ARM_ORDER if a in res]:
        R = res[arm]; lg = R["log"]
        e = np.array(lg["e_rt"], float)[:, drill]
        b = np.array(lg["b"], float)[:, drill]
        d = np.array(lg["delta"], float)[:, drill]
        sil = np.array(lg["sil_run"], float)[:, drill]
        ev = [x for x in R["events"] if x["kind"] == "compile"]
        cc = ev[0]["cycle"] if ev else None
        print(f"  [{arm:11s}] compile@cycle={cc}  e: {e[0]:.4f} -> {e[-1]:.4f} (min {e.min():.4f})  "
              f"max sil_run={int(sil.max())}")
    R0 = res.get("never") or next(iter(res.values()))
    lg = R0["log"]
    e = np.array(lg["e_rt"], float); b = np.array(lg["b"], float); d = np.array(lg["delta"], float)
    print(f"\n  per-cycle detail ({'never' if 'never' in res else R0['arm']}), all {K} segments:")
    print("   cyc | " + " | ".join(f"s{k}: e      b      d     " for k in range(K)))
    for i, c in enumerate(lg["cycle"]):
        if c % 2 and len(lg["cycle"]) > 20:
            continue
        row = " | ".join(f"    {e[i,k]:.4f} {b[i,k]:.4f} {d[i,k]:+.4f}" for k in range(K))
        print(f"   {c:3d} | {row}")


def silence_panel(res, drill, cfg, cs=(0.02, 0.035, 0.05, 0.08, 0.12), ws=(3, 5, 8)):
    """Counterfactual: on an arm whose routing never changed, when would each (c, W) have fired?

    Mirrors the runner's rule exactly — scale from the SETUP stale reference at performance tempo,
    window-MEAN delta, window sd, held `sil_hold` cycles."""
    arm = "never" if "never" in res else next(iter(res))
    lg = res[arm]["log"]
    e = np.array(lg["e_rt"], float)[:, drill]
    Rp = cfg["ladder"][-1]
    stale = float(np.median(np.array(res[arm]["ref_stale"][f"R{Rp}"], float), 0)[drill])
    alpha = cfg["bench_alpha"]; hold = cfg.get("sil_hold", 2)
    print(f"\n--- δ-silence counterfactual panel (off arm `{arm}`, drilled segment; stale ref "
          f"{stale:.4f}, c_v={cfg['sil_cv']}, alpha={alpha}, hold={hold}) ---")
    print("     c \\ W   " + "  ".join(f"W={w:<4d}" for w in ws))
    for c in cs:
        row = []
        for w in ws:
            b = None; emin = None; run = 0; fire = None; dh = []
            for i, ev in enumerate(e):
                if b is None:
                    b = ev; emin = ev
                d = b - ev; emin = min(emin, ev); dh.append(d)
                scale = max(stale - emin, b, 1e-6)
                we = e[max(0, i - w + 1):i + 1]; wd = np.array(dh[max(0, i - w + 1):i + 1])
                ok = len(we) >= w and abs(float(wd.mean())) < c * scale \
                    and float(we.std()) < cfg["sil_cv"] * scale
                run = run + 1 if ok else 0
                b = b + alpha * (ev - b)
                if run >= hold and fire is None and (i + 1) >= cfg["sil_min_cycle"]:
                    fire = i + 1
            row.append(f"{fire if fire else '-':<6}")
        print(f"     {c:<8.3f} " + "  ".join(row))


def compile_step(res, drill):
    """The load-bearing readout: what the compiled unit achieves versus the CEM-ballistic
    controller it REPLACED, measured at the same tempo on the same segment."""
    print("\n--- the compile step: committed unit vs the controller it replaced (drilled segment) ---")
    print(f"     {'arm':12s} {'cycle':>5s} {'tuples':>7s} {'CEM-ballistic':>14s} {'compiled (next 8 cyc)':>22s} {'ratio':>7s}")
    for arm in [a for a in ARM_ORDER if a in res]:
        ev = [e for e in res[arm]["events"] if e["kind"] == "compile"]
        if not ev:
            print(f"     {arm:12s} {'-':>5s} {'-':>7s} {'-':>14s} {'-':>22s} {'-':>7s}")
            continue
        e0 = ev[0]; c = e0["cycle"]
        post = np.array(res[arm]["log"]["e_rt"], float)[c:c + 8, drill]
        if len(post) == 0:
            print(f"     {arm:12s} {c:5d} {int(e0['n_tuples']):7d} {e0['e_rt']:14.4f} "
                  f"{'(no post-compile cycles)':>22s} {'-':>7s}")
            continue
        print(f"     {arm:12s} {c:5d} {int(e0['n_tuples']):7d} {e0['e_rt']:14.4f} "
              f"{post.mean():15.4f}+-{post.std():.4f} {post.mean()/e0['e_rt']:7.2f}x")


def grade(res, drill, Rp, lad, ceiling_perf, crit_mult=1.25, crit_on="drill", ceiling_drill=None):
    print("\n--- priced grade: cumulative practice traversal time x waypoint error at performance tempo ---")
    base = ceiling_drill if (crit_on == "drill" and ceiling_drill is not None) else ceiling_perf
    eps = base * crit_mult
    print(f"    criterion eps = {crit_mult}x ceiling {crit_on}-error at R{Rp} = {eps:.4f}")
    print("    (the drilled segment is the primary readout: its stale->ceiling range is the whole"
          " instrument; the piece mean dilutes it 4x with the clean segments' open-loop floor)")
    for arm in [a for a in ARM_ORDER if a in res]:
        R = res[arm]; L = R["ladder"]
        t = np.array([x["t_cum"] for x in L], float)
        Ep = np.array([np.mean(seg_err(x, Rp)) for x in L], float)
        Ed = np.array([seg_err(x, Rp)[drill] for x in L], float)
        Er = np.array([np.mean(seg_err(x, lad[0])) for x in L], float)
        hit = np.where((Ed if crit_on == "drill" else Ep) <= eps)[0]
        ttc = f"{t[hit[0]]:.0f}s @c{L[hit[0]]['cycle']}" if len(hit) else "never"
        ev = [x for x in R["events"] if x["kind"] == "compile"]
        cc = ev[0]["cycle"] if ev else None
        print(f"  [{arm:11s}] compile@c{str(cc):<5s} t_final={t[-1]:7.0f}s  "
              f"E_perf={Ep[-1]:.4f} (drill {Ed[-1]:.4f})  E_react={Er[-1]:.4f}  t_to_crit={ttc}")
    print("\n    per-milestone (cycle, t_cum s, E_perf):")
    for arm in [a for a in ARM_ORDER if a in res]:
        L = res[arm]["ladder"]
        s_ = "  ".join(f"c{x['cycle']}:{x['t_cum']:.0f}s/{np.mean(seg_err(x, Rp)):.3f}" for x in L)
        print(f"      {arm:11s} {s_}")
    print("\n    per-milestone drilled segment at performance tempo (the primary readout):")
    for arm in [a for a in ARM_ORDER if a in res]:
        L = res[arm]["ladder"]
        s_ = "  ".join(f"c{x['cycle']}:{seg_err(x, Rp)[drill]:.4f}" for x in L)
        print(f"      {arm:11s} {s_}")


def skew_factor(R, cycle):
    """Median/mean ratio of the run-through's boundary-error distribution, taken from the ladder
    milestone nearest `cycle`. Open-loop boundary error is right-skewed, so this is ~0.82-0.87 for
    committed arms and ~1.0 for uncommitted ones."""
    L = R["ladder"]; Rp = R["config"]["ladder"][-1]
    x = min(L, key=lambda z: abs(z["cycle"] - cycle))[f"R{Rp}"]
    m = float(x["wp_err_mean"])
    return (float(np.mean(x["wp_err"])) / m) if m > 0 else 1.0


def gap_table(res):
    """The optimism gap, raw and baseline-corrected.

    The audition score is a MEAN over score states; the anchored level is a MEDIAN over run-through
    performers. That is not like-for-like, and it puts a ~0.85 floor under every gap. The runner
    logs only per-segment medians, so a per-segment mean is not recoverable post hoc; the correction
    below divides by the arm's own piece-level median/mean skew, which makes 1.0 neutral. Analyzer
    side only — the runner is untouched, so all tags stay bit-compatible for pooling.
    """
    print("\n--- optimism gap: raw and baseline-corrected (1.0 = as auditioned) ---")
    print(f"     {'arm':16s} {'seg':>3s} {'commit':>6s} {'score':>7s} {'anchored':>8s} "
          f"{'raw':>5s} {'skew':>5s} {'corrected':>9s}")
    for arm in [a for a in ARM_ORDER if a in res]:
        ev = res[arm]["events"]
        ancs = {x["seg"]: x for x in ev if x["kind"] == "anchor"}
        comps = [x for x in ev if x["kind"] == "compile"]
        # `eg_s0`..`e4_s0` wrote the compile event's `seg` against the fixed drill index rather than
        # the moving commit target, so every commit there is labelled seg 1. Under the sequential
        # rule the i-th commit IS segment i, and the anchor events carry the true segment — so
        # recover the attribution rather than silently mis-pairing.
        if len(comps) > 1 and len({x["seg"] for x in comps}) == 1 and len(ancs) > 1:
            for i, x in enumerate(comps):
                x = dict(x); x["seg"] = i; comps[i] = x
        for e in comps:
            an = ancs.get(e["seg"])
            if an is None or "chosen_score" not in e:
                continue
            raw = an["level"] / e["chosen_score"]
            sk = skew_factor(res[arm], an["cycle"])
            print(f"     {arm:16s} {e['seg']:3d} {e['cycle']:6d} {e['chosen_score']:7.4f} "
                  f"{an['level']:8.4f} {raw:5.2f} {sk:5.2f} {raw/sk:9.2f}")


def aggregate(tags, fetch=False):
    """Seed-aggregated verdicts across tags."""
    runs = {t: load(t, fetch) for t in tags}
    arms = [a for a in ARM_ORDER if all(a in runs[t] for t in tags)]
    print(f"\n=== seed-aggregated over {tags} (arms present in all: {arms}) ===")
    print(f"\n{'arm':16s} {'t_cum (mean+-sd)':>22s} {'E_piece (mean+-sd)':>22s}   per-seed E_piece")
    agg = {}
    for a in arms:
        t = np.array([runs[x][a]["ladder"][-1]["t_cum"] for x in tags], float)
        Rp = runs[tags[0]][a]["config"]["ladder"][-1]
        E = np.array([np.mean(runs[x][a]["ladder"][-1][f"R{Rp}"]["wp_err"]) for x in tags], float)
        agg[a] = (t, E)
        print(f"{a:16s} {t.mean():10.0f} +- {t.std(ddof=0):<8.0f} {E.mean():10.4f} +- {E.std(ddof=0):<8.4f}   "
              + "  ".join(f"{v:.4f}" for v in E))
    if "never" in agg and "seq_seam" in agg:
        d = agg["never"][1] - agg["seq_seam"][1]
        print(f"\n  headline margin (never - seq_seam) per seed: {np.round(d,4).tolist()}"
              f"   mean {d.mean():+.4f}  all-same-sign={bool(np.all(np.sign(d)==np.sign(d[0])))}")
    if "seam_drill_fix" in agg and "seq_seam" in agg:
        d = agg["seam_drill_fix"][1] - agg["seq_seam"][1]
        dt = agg["seam_drill_fix"][0] - agg["seq_seam"][0]
        print(f"  knitting deficit (seam_drill_fix - seq_seam) per seed: {np.round(d,4).tolist()}"
              f"   mean {d.mean():+.4f}  all-same-sign={bool(np.all(np.sign(d)==np.sign(d[0])))}")
        print(f"  knitting time price per seed: {np.round(dt,0).tolist()}   mean {dt.mean():+.0f}s")
    print("\n  commit cycles per segment (certification-delay reproducibility):")
    for a in arms:
        for x in tags:
            cs = [(e["seg"], e["cycle"]) for e in runs[x][a]["events"] if e["kind"] == "compile"]
            an = {e["seg"]: e["level"] for e in runs[x][a]["events"] if e["kind"] == "anchor"}
            print(f"    {a:16s} {x:8s} commits {cs}   anchored "
                  + " ".join(f"s{k}:{v:.4f}" for k, v in sorted(an.items())))


def clean_table(res, K, drill, Rp, lad):
    """The clean segments are a retention / collateral-damage readout, not just a floor: at the
    calibration the ceiling FM was NOT better than the stale FM on them at performance tempo, so
    adapting to the rotation region can cost open-loop accuracy elsewhere. Report their
    trajectories per arm, both per-cycle (at-tempo run-through) and per-milestone (held-out ladder).
    """
    clean = [k for k in range(K) if k != drill]
    print(f"\n--- clean segments {clean}: interference / retention readout ---")
    print("    per-milestone mean error at performance tempo R%d:" % Rp)
    for arm in [a for a in ARM_ORDER if a in res]:
        L = res[arm]["ladder"]
        s = "  ".join(f"c{x['cycle']}:{np.mean(seg_err(x, Rp)[clean]):.4f}" for x in L)
        print(f"      {arm:11s} {s}")
    print("    per-milestone, per clean segment (first -> last milestone):")
    for arm in [a for a in ARM_ORDER if a in res]:
        L = res[arm]["ladder"]
        s = "  ".join(f"s{k}:{seg_err(L[0], Rp)[k]:.4f}->{seg_err(L[-1], Rp)[k]:.4f}" for k in clean)
        print(f"      {arm:11s} {s}")
    print("    per-cycle at-tempo run-through, clean-segment mean (every 4th cycle):")
    for arm in [a for a in ARM_ORDER if a in res]:
        lg = res[arm]["log"]; e = np.array(lg["e_rt"], float)[:, clean].mean(1)
        idx = [i for i, c in enumerate(lg["cycle"]) if c % 4 == 0]
        s = "  ".join(f"c{lg['cycle'][i]}:{e[i]:.4f}" for i in idx)
        print(f"      {arm:11s} {s}")
    print("    FM probes (region / clean) over the run, every 4th cycle:")
    for arm in [a for a in ARM_ORDER if a in res]:
        lg = res[arm]["log"]
        idx = [i for i, c in enumerate(lg["cycle"]) if c % 4 == 0]
        s = "  ".join(f"c{lg['cycle'][i]}:{lg['fm_region'][i]:.4f}/{lg['fm_clean'][i]:.4f}" for i in idx)
        print(f"      {arm:11s} {s}")


def ladder_table(res, Rp, lad, drill):
    print("\n--- final tempo ladder (piece-mean waypoint error; R1 = fully reactive) ---")
    hdr = "  ".join(f"R{R:<3d}" for R in lad)
    print(f"     {'arm':12s} {hdr}      compiled")
    for arm in [a for a in ARM_ORDER if a in res]:
        L = res[arm]["ladder"][-1]
        row = "  ".join(f"{np.mean(seg_err(L, R)):.4f}" for R in lad)
        print(f"     {arm:12s} {row}   {L['compiled']}")


def figures(res, tag, drill, Rp, lad, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    outd = os.path.join(HERE, "figures", tag); os.makedirs(outd, exist_ok=True)
    arms = [a for a in ARM_ORDER if a in res]

    # fig1: the metering trace on the drilled segment
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for arm in arms:
        lg = res[arm]["log"]; c = np.array(lg["cycle"])
        e = np.array(lg["e_rt"], float)[:, drill]; d = np.array(lg["delta"], float)[:, drill]
        axes[0].plot(c, e, lw=1.8, color=COL[arm], label=arm)
        axes[1].plot(c, d, lw=1.4, color=COL[arm], label=arm)
        for ev in res[arm]["events"]:
            axes[0].axvline(ev["cycle"], color=COL[arm], ls=":", lw=1.2, alpha=0.8)
    axes[0].set_xlabel("practice cycle"); axes[0].set_ylabel(f"e (seg {drill} boundary err, at tempo)")
    axes[0].set_title("metering: at-tempo boundary error on the drilled segment\n(dotted = compile event)")
    axes[1].axhline(0, color="#adb5bd", lw=1)
    axes[1].set_xlabel("practice cycle"); axes[1].set_ylabel("δ = b − e")
    axes[1].set_title("δ on the drilled segment (silence = |δ|≈0, low variance)")
    axes[0].legend(fontsize=7.5); fig.tight_layout()
    fig.savefig(os.path.join(outd, "fig1_metering.png"), bbox_inches="tight"); plt.close(fig)

    # fig2: the priced grade — cumulative time vs error at performance tempo
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for arm in arms:
        L = res[arm]["ladder"]
        t = np.array([x["t_cum"] for x in L]); E = np.array([np.mean(seg_err(x, Rp)) for x in L])
        ax.plot(t, E, "o-", ms=4, lw=1.8, color=COL[arm], label=arm)
        ax.plot(t[-1], E[-1], "*", ms=14, color=COL[arm])
    ax.set_xlabel("cumulative practice traversal time (s, priced at d_fb per feedback event)")
    ax.set_ylabel(f"piece waypoint error at performance tempo (R{Rp})")
    ax.set_title("the priced grade: time spent practising vs competence at tempo\n(down-left is better; ★ = end of run)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outd, "fig2_priced.png"), bbox_inches="tight"); plt.close(fig)

    # fig3: final tempo ladder
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm in arms:
        L = res[arm]["ladder"][-1]
        ax.plot(lad, [np.mean(seg_err(L, R)) for R in lad], "o-", lw=1.8, color=COL[arm], label=arm)
    any_arm = res[arms[0]]
    for nm, ls in (("ref_stale", ":"), ("ref_ceiling", "--")):
        ax.plot(lad, [np.mean(np.median(np.array(any_arm[nm][f"R{R}"], float), 0)) for R in lad],
                ls, lw=1.4, color="#868e96", label=nm)
    ax.set_xscale("log"); ax.set_xticks(lad); ax.set_xticklabels([str(R) for R in lad])
    ax.set_xlabel("replan_every (tempo: right = fewer corrections per segment = faster)")
    ax.set_ylabel("piece waypoint error")
    ax.set_title("final tempo ladder")
    ax.legend(fontsize=7.5); fig.tight_layout()
    fig.savefig(os.path.join(outd, "fig3_ladder.png"), bbox_inches="tight"); plt.close(fig)

    # fig4: clean-segment interference / retention, plus the FM probes that drive it
    K = len(cfg["waypoints"]) - 1
    clean = [k for k in range(K) if k != drill]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for arm in arms:
        lg = res[arm]["log"]; c = np.array(lg["cycle"])
        axes[0].plot(c, np.array(lg["e_rt"], float)[:, clean].mean(1), lw=1.7, color=COL[arm], label=arm)
        axes[1].plot(c, lg["fm_region"], lw=1.7, color=COL[arm], label=arm)
        axes[1].plot(c, lg["fm_clean"], lw=1.1, ls="--", color=COL[arm], alpha=0.7)
        L = res[arm]["ladder"]
        axes[2].plot([x["cycle"] for x in L], [np.mean(seg_err(x, Rp)[clean]) for x in L],
                     "o-", ms=4, lw=1.7, color=COL[arm], label=arm)
        for ev in res[arm]["events"]:
            for ax_ in axes:
                ax_.axvline(ev["cycle"], color=COL[arm], ls=":", lw=1.1, alpha=0.7)
    axes[0].set_title("clean segments, at-tempo run-through (per cycle)")
    axes[1].set_title("FM prediction error: region (solid) / clean (dashed)")
    axes[2].set_title(f"clean segments, held-out ladder at R{Rp} (per milestone)")
    for ax_ in axes:
        ax_.set_xlabel("practice cycle")
    axes[0].set_ylabel("mean boundary error"); axes[1].set_ylabel("FM error")
    axes[2].set_ylabel("mean boundary error"); axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outd, "fig4_clean.png"), bbox_inches="tight"); plt.close(fig)
    print(f"\n[figures] wrote 4 figures to {outd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--tags", help="comma-separated tags -> seed-aggregated verdicts")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--crit-mult", type=float, default=1.25)
    ap.add_argument("--crit-on", choices=["drill", "piece"], default="drill")
    a = ap.parse_args()
    if a.tags:
        aggregate([t for t in a.tags.split(",") if t], a.fetch)
        return
    if not a.tag:
        raise SystemExit("need --tag or --tags")
    res = load(a.tag, a.fetch)
    if not res:
        raise SystemExit(f"no arm results found for tag {a.tag}")
    print(f"loaded arms: {sorted(res)}")
    cfg, K, drill, Rp, lad, st, ce = summarize(res)
    metering_table(res, K, drill)
    silence_panel(res, drill, cfg)
    compile_step(res, drill)
    gap_table(res)
    any_arm = next(iter(res.values()))
    ceiling_perf = float(np.mean(np.median(np.array(any_arm["ref_ceiling"][f"R{Rp}"], float), 0)))
    ceiling_drill = float(np.median(np.array(any_arm["ref_ceiling"][f"R{Rp}"], float), 0)[drill])
    grade(res, drill, Rp, lad, ceiling_perf, a.crit_mult, a.crit_on, ceiling_drill)
    clean_table(res, K, drill, Rp, lad)
    ladder_table(res, Rp, lad, drill)
    if a.figures:
        figures(res, a.tag, drill, Rp, lad, cfg)


if __name__ == "__main__":
    main()
