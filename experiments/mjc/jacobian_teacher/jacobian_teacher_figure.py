"""Cross-seed aggregation + figures for jacobian_teacher Phase B. Local, read-only.

Reports the readouts in the order of how much weight they can bear:

  1. THE ADAPTATION LADDER — final endpoint error per (FM rung × teacher), with both FM
     readings printed alongside every rung: the record's forecast error and this node's
     Jacobian direction accuracy. `arm_substrate` P5/P7's standing rule ("report damage/gain on
     this substrate, never a slope against `fm_err`") is why the association between outcome
     and each reading is reported as a rank correlation across rungs, never as a regression
     slope on a noisy x-axis.
  2. SPEED AND VARIABILITY — reaches-to-90%-of-own-gain, the plateau standard deviation, the
     plateau jitter of the learned mean, and `grad_snr` (the cosine between two half-batch
     action gradients). The last is Garibbo's own claim about why RBL and EBL differ — "the
     RBL action gradient is known to have a higher variance (i.e., noisier) than the equivalent
     EBL gradient" — measured directly rather than inferred from behaviour.
  3. GENERALIZATION — endpoint error against direction offset from the trained band.
  4. THE AFTEREFFECT — the learned program run back in the field-free world, signed against the
     curl. Reported as a signed pair, because an unsigned miss is equally consistent with
     "the policy just got worse".
  5. THE DYSMETRIA TABLE — per-component sign flips, read on the SIGNED radial (hypermetria vs
     hypometria) and lateral (displacement) deviations.

Run (from `experiments/`):
    python3 mjc/jacobian_teacher/jacobian_teacher_figure.py --tags jt_s0
    python3 mjc/jacobian_teacher/jacobian_teacher_figure.py --tags jt_s0 jt_s1 jt_s2
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "figures", f"jt_{tag}", "results.json")) as fh:
        return json.load(fh)


def ms_sem(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    if not len(a):
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std() / max(np.sqrt(len(a)), 1))


def plateau_mean(curve, key, k=4):
    """Mean of the last `k` readouts. The primary summary, in place of the single final value.

    These arms do not settle: Adam at a fixed learning rate on a gradient that shrinks as the
    endpoint error does takes a roughly constant-size step forever, so the policy oscillates
    around its optimum instead of converging, and the LAST readout samples the phase of that
    oscillation. In the jt_s0 run `ebl_sensory` reads 0.0209, 0.0168, 0.0224, 0.0212, 0.0407,
    0.1126 over its final six readouts at one rung — a five-fold spread that says nothing about
    the FM it was taught through. The mean over the plateau window is the stable statistic;
    `divergence` below reports the instability itself rather than letting it contaminate this.
    """
    v = [c[key] for c in curve if key in c][-k:]
    return float(np.mean(v)) if v else float("nan")


def divergence(curve, key, k=4):
    """Plateau mean divided by the best value the arm ever reached. > 1 means the arm walked
    back away from its own optimum — the late instability, reported rather than averaged over."""
    v = [c[key] for c in curve if key in c]
    if not v:
        return float("nan")
    return float(np.mean(v[-k:]) / max(np.min(v), 1e-12))


def reaches_to(curve, key, thresh):
    """Executed reaches until the arm FIRST reaches a FIXED error threshold, linearly
    interpolated between readouts.

    This is the data-efficiency readout "does it teach faster" actually asks for.
    `reaches_to_90` (90% of the arm's OWN gain) cannot answer it: an arm that plateaus high
    hits 90% of its own smaller gain early and scores well, and on a log-spaced grid it
    quantises to whichever readout happens to straddle the crossing. A shared threshold puts
    every arm on one axis.
    """
    rows = [c for c in curve if key in c]
    for a, b in zip(rows, rows[1:]):
        if b[key] <= thresh:
            if a[key] <= thresh:
                return float(a["reaches"])
            f = (a[key] - thresh) / max(a[key] - b[key], 1e-12)
            return float(a["reaches"] + f * (b["reaches"] - a["reaches"]))
    return float("nan")


def grad_snr_early(curve, frac=0.25):
    """Mean `grad_snr` over the EARLY readouts only.

    The half-batch gradient cosine measures teacher noise, and it does so honestly only while
    there is a signal to be noisy about. At convergence the batch-mean action gradient is near
    zero, so the cosine between two half-batch means is dominated by whatever is left and swings
    freely — Phase A's `ebl_sensory` runs +0.996, +0.999, +0.998 early and then -0.842, +0.983,
    -0.932 at plateau. Averaging the whole run mixes a measurement with an artifact; the early
    window is the measurement.
    """
    rows = [c for c in curve if "grad_snr" in c]
    if not rows:
        return float("nan")
    cut = max(rows[-1]["reaches"] * frac, rows[0]["reaches"])
    early = [c["grad_snr"] for c in rows if c["reaches"] <= cut]
    return float(np.mean(early)) if early else float(rows[0]["grad_snr"])


def spearman(x, y):
    """Rank correlation. Rank-based on purpose: the x-axes here (`fm_err`, Jacobian cosine)
    are the ones `arm_substrate` P5 found non-monotonic in absolute value on this substrate,
    so only their ORDER is trustworthy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        # A constant series has no ordering. argsort's index tie-break would happily invent
        # one — it reported rho = 0.976 for `rbl`, whose outcome cannot vary with the FM by
        # construction. Refuse instead.
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="agg")
    args = ap.parse_args()

    R = [load(t) for t in args.tags]
    cfg = R[0]["config"]
    rungs = [l["name"] for l in R[0]["ladder"]]
    labels = sorted({k for r in R for l in r["ladder"] for k in l["arms"]})
    W = max(13, max(len(l) for l in labels) + 1)

    print(f"\n{'=' * 110}\njacobian_teacher Phase B — the forward model in backward mode")
    print(f"tags: {', '.join(args.tags)}   n_links={cfg['n_links']} H={cfg['plan_H']} "
          f"k_shoot={cfg['k_shoot']} cem_iters={cfg['cem_iters']} n_eval={cfg['n_eval']}")
    print(f"budget {cfg['reach_budget']} executed reaches/arm  batch {cfg['train_batch']}  "
          f"sigma={cfg['sigma']}  lr_rbl={cfg['lr_rbl']} lr_ebl={cfg['lr_ebl']}")
    ip = ms_sem([r["initial_policy"]["train_band"] for r in R])
    print(f"shared initial motor program (BC from the pre-drift FM), in the curl world: "
          f"{ip[0]:.4f} ± {ip[1]:.4f}")

    PLATEAU_K = cfg.get("plateau_evals", 4)

    def arm(ri, lab, key):
        rows = [r for r in R if lab in r["ladder"][ri]["arms"]]
        if key == "grad_snr_early":
            return [grad_snr_early(r["ladder"][ri]["arms"][lab]["curve"]) for r in rows]
        if key.endswith("_plateau_mean"):
            return [plateau_mean(r["ladder"][ri]["arms"][lab]["curve"],
                                 key[: -len("_plateau_mean")], PLATEAU_K) for r in rows]
        if key.endswith("_divergence"):
            return [divergence(r["ladder"][ri]["arms"][lab]["curve"],
                               key[: -len("_divergence")], PLATEAU_K) for r in rows]
        return [r["ladder"][ri]["arms"][lab]["summary"].get(key) for r in rows]

    # ---------------------------------------------------------------- 1. the ladder
    print(f"\n--- 1. PLATEAU-MEAN train-band endpoint error, per FM rung x teacher "
          f"(seed mean; mean of the last {cfg.get('plateau_evals', 4)} readouts — see "
          f"plateau_mean() for why not the final value) ---")
    print(f"{'rung':>16s} {'fm_err':>8s} {'jac1cos':>8s} {'jacEcos':>8s} {'vjpcos':>7s} "
          f"{'signW':>6s} " + "".join(f"{l:>{W}s}" for l in labels) +
          f"{'bal_cem':>{W}s}{'reactive':>{W}s}")
    for ri, name in enumerate(rungs):
        rd = [r["ladder"][ri]["readings"] for r in R]
        row = (f"{name:>16s} {np.mean([d['fm_err'] for d in rd]):8.4f} "
               f"{np.mean([d['jac1']['cos'] for d in rd]):8.4f} "
               f"{np.mean([d['jacE']['cos'] for d in rd]):8.4f} "
               f"{np.mean([d['jacE']['vjp_cos'] for d in rd]):7.4f} "
               f"{np.mean([d['jacE']['sign_agree_w'] for d in rd]):6.3f} ")
        for l in labels:
            v = arm(ri, l, "train_band_plateau_mean")
            row += f"{ms_sem(v)[0]:{W}.4f}" if v else f"{'—':>{W}s}"
        inc = [r["ladder"][ri]["incumbents"] for r in R]
        row += f"{np.mean([i['ballistic_cem'] for i in inc]):{W}.4f}"
        row += f"{np.mean([i['reactive'] for i in inc]):{W}.4f}"
        print(row)

    print(f"\n--- 1b. which FM reading does the outcome follow? (Spearman across rungs; "
          f"|rho|=1 is a perfect ordering) ---")
    print(f"{'teacher':>{W}s} {'vs fm_err':>10s} {'vs jac1 cos':>12s} {'vs jacE cos':>12s} "
          f"{'vs vjp_cos':>11s}")
    for l in labels:
        outs, fe, j1, jE, vj = [], [], [], [], []
        for ri in range(len(rungs)):
            v = arm(ri, l, "train_band_plateau_mean")
            if not v:
                continue
            outs.append(np.mean(v))
            rd = [r["ladder"][ri]["readings"] for r in R]
            fe.append(np.mean([d["fm_err"] for d in rd]))
            j1.append(np.mean([d["jac1"]["cos"] for d in rd]))
            jE.append(np.mean([d["jacE"]["cos"] for d in rd]))
            vj.append(np.mean([d["jacE"]["vjp_cos"] for d in rd]))
        print(f"{l:>{W}s} {spearman(fe, outs):10.3f} {spearman(j1, outs):12.3f} "
              f"{spearman(jE, outs):12.3f} {spearman(vj, outs):11.3f}")
    print("   (outcome is an ERROR, so a NEGATIVE rho against a direction cosine and a "
          "POSITIVE rho against fm_err both mean 'better model -> better learning')")

    # ---------------------------------------------------------------- 2. speed + variability
    print(f"\n--- 1c. DATA EFFICIENCY: executed reaches to first reach a FIXED endpoint error "
          f"(interpolated; '—' = never reached) ---")
    for th in cfg.get("thresholds", [0.10, 0.05, 0.03, 0.02]):
        print(f"  threshold {th:.3f} m")
        print(f"{'rung':>16s} " + "".join(f"{l:>{W}s}" for l in labels))
        for ri, name in enumerate(rungs):
            row = f"{name:>16s} "
            for l in labels:
                v = [reaches_to(r["ladder"][ri]["arms"][l]["curve"], "train_band", th)
                     for r in R if l in r["ladder"][ri]["arms"]]
                v = [x for x in v if np.isfinite(x)]
                row += f"{int(np.mean(v)):{W}d}" if v else f"{'—':>{W}s}"
            print(row)

    ref = cfg.get("speedup_ref", "rbl_cont")
    if ref in labels:
        print(f"\n--- 1d. SPEEDUP over `{ref}`: its reaches-to-threshold divided by this "
              f"arm's, computed WITHIN each seed and then averaged (>1 = faster) ---")
        print(f"   The per-seed ratio, not the ratio of per-seed means: the seeds' shared "
              f"initial motor programs differ (BC quality is seed-dependent — 0.2066 and "
              f"0.1439 in jt_s0/jt_s1), so each seed has a different distance to travel to a "
              f"fixed threshold and averaging the raw counts mixes that in.")
        for th in cfg.get("thresholds", [0.10, 0.05, 0.03, 0.02]):
            print(f"  threshold {th:.3f} m")
            print(f"{'rung':>16s} " + "".join(f"{l:>{W}s}" for l in labels))
            for ri, name in enumerate(rungs):
                row = f"{name:>16s} "
                for l in labels:
                    rat = []
                    for r in R:
                        if l not in r["ladder"][ri]["arms"] or ref not in r["ladder"][ri]["arms"]:
                            continue
                        a = reaches_to(r["ladder"][ri]["arms"][l]["curve"], "train_band", th)
                        b = reaches_to(r["ladder"][ri]["arms"][ref]["curve"], "train_band", th)
                        if np.isfinite(a) and np.isfinite(b) and a > 0:
                            rat.append(b / a)
                    row += f"{np.mean(rat):{W}.2f}" if rat else f"{'—':>{W}s}"
                print(row)

    print(f"\n--- 2. speed and variability (seed mean; '—' = arm absent at that rung) ---")
    for key, cap in (("train_band_reaches_to_90", "reaches to 90% of own gain"),
                     ("train_band_best", "best endpoint error the arm ever reached"),
                     ("train_band_divergence", "divergence: plateau mean / best (>1 = the arm "
                                               "walked back away from its own optimum)"),
                     ("train_band_plateau_sd", "plateau sd of endpoint error"),
                     ("mu_jitter_plateau", "plateau jitter of the learned mean"),
                     ("grad_snr_early", "grad_snr, EARLY (half-batch cosine; 1 = noiseless "
                                        "teacher). See grad_snr_early(): the whole-run mean "
                                        "mixes this measurement with a plateau artifact."),
                     ("g_norm_mean", "mean action-gradient norm")):
        print(f"  {cap}")
        print(f"{'rung':>16s} " + "".join(f"{l:>{W}s}" for l in labels))
        for ri, name in enumerate(rungs):
            row = f"{name:>16s} "
            for l in labels:
                v = arm(ri, l, key)
                m = ms_sem(v)[0] if v else float("nan")
                row += (f"{'—':>{W}s}" if not np.isfinite(m)
                        else (f"{int(m):{W}d}" if "reaches" in key else f"{m:{W}.4f}"))
            print(row)

    # ---------------------------------------------------------------- 3. generalization
    offs = [o for o in cfg["gen_offsets"]]
    print(f"\n--- 3. generalization: endpoint error vs direction offset from the trained band "
          f"(trained at {cfg['train_dir']:+.0f}±{cfg['train_dir_halfwidth']:.0f}°) ---")
    for ri, name in enumerate(rungs):
        if not any(l in R[0]["ladder"][ri]["arms"] for l in labels):
            continue
        print(f"  rung {name}")
        print(f"{'teacher':>{W}s} " + "".join(f"{int(o):>9d}°" for o in offs))
        for l in labels:
            if l not in R[0]["ladder"][ri]["arms"]:
                continue
            row = f"{l:>{W}s} "
            for o in offs:
                v = arm(ri, l, f"gen@{int(o)}")
                row += f"{ms_sem(v)[0]:10.4f}"
            print(row)

    # ---------------------------------------------------------------- 4. aftereffect
    print(f"\n--- 4. the aftereffect: the learned program run back in the FIELD-FREE world "
          f"(+lat = the direction the curl pushes the hand) ---")
    ip = R[0]["initial_policy"]
    print(f"{'initial policy':>{W}s}  in-field lat={ms_sem([r['initial_policy']['train_band_lat'] for r in R])[0]:+.4f}"
          f"   field-free lat={ms_sem([r['initial_policy']['ae_lat'] for r in R])[0]:+.4f}"
          f"   field-free dist={ms_sem([r['initial_policy']['ae_dist'] for r in R])[0]:.4f}")
    for ri, name in enumerate(rungs):
        for l in labels:
            if l not in R[0]["ladder"][ri]["arms"]:
                continue
            lat = ms_sem(arm(ri, l, "train_band_lat_final"))[0]
            ae = ms_sem(arm(ri, l, "ae_lat"))[0]
            aed = ms_sem(arm(ri, l, "ae_dist"))[0]
            print(f"{name + '|' + l:>{max(W, 28)}s}  in-field lat={lat:+.4f}   "
                  f"field-free lat={ae:+.4f}   field-free dist={aed:.4f}")

    # ---------------------------------------------------------------- 5. dysmetria
    dys = [d for r in R for d in r.get("dysmetria", [])]
    if dys:
        print(f"\n--- 5. the dysmetria control: one component of the FM's believed ∂y_k/∂u_j "
              f"sign-flipped, the FM otherwise untouched ---")
        print("    (+rad = overshoot / hypermetria, −rad = undershoot / hypometria, "
              "|lat| = displacement)")
        names = []
        for d in dys:
            if (d["rung"], d["mask"]) not in names:
                names.append((d["rung"], d["mask"]))
        print(f"{'rung':>14s} {'mask':>16s} {'dist':>9s} {'Δdist':>9s} {'radial':>9s} "
              f"{'lateral':>9s} {'grad_snr':>9s}")
        base = {}
        for rung, mask in names:
            rows = [d["summary"] for d in dys if d["rung"] == rung and d["mask"] == mask]
            m = {k: ms_sem([x.get(k) for x in rows])[0] for k in
                 ("train_band_final", "train_band_rad_final", "train_band_lat_final",
                  "grad_snr_mean")}
            if mask == "healthy":
                base[rung] = m["train_band_final"]
            print(f"{rung:>14s} {mask:>16s} {m['train_band_final']:9.4f} "
                  f"{m['train_band_final'] - base.get(rung, np.nan):+9.4f} "
                  f"{m['train_band_rad_final']:+9.4f} {m['train_band_lat_final']:+9.4f} "
                  f"{m['grad_snr_mean']:9.3f}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                          # pragma: no cover
        print(f"\n[plot] skipped ({e})")
        return

    outdir = os.path.join(HERE, "figures", args.out)
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ri_last = len(rungs) - 1
    for l in labels:
        if l not in R[0]["ladder"][0]["arms"]:
            continue
        cur = R[0]["ladder"][0]["arms"][l]["curve"]
        ax[0].plot([c["reaches"] for c in cur], [c["train_band"] for c in cur],
                   marker="o", ms=3, label=l)
        cur = R[0]["ladder"][ri_last]["arms"][l]["curve"]
        ax[1].plot([c["reaches"] for c in cur], [c["train_band"] for c in cur],
                   marker="o", ms=3, label=l)
    for a, t in ((ax[0], f"stale FM ({rungs[0]})"), (ax[1], f"matched FM ({rungs[ri_last]})")):
        a.set_xlabel("executed reaches"); a.set_ylabel("endpoint error (m)")
        a.set_title(t); a.grid(alpha=0.3); a.legend(fontsize=7)
    for ri, name in enumerate(rungs):
        for l in labels:
            if l not in R[0]["ladder"][ri]["arms"]:
                continue
            v = [ms_sem(arm(ri, l, f"gen@{int(o)}"))[0] for o in offs]
            ax[2].plot(offs, v, marker="o", ms=3, alpha=0.4 + 0.6 * ri / max(len(rungs) - 1, 1),
                       label=f"{name}|{l}" if ri == ri_last else None)
    ax[2].set_xlabel("goal-direction offset from the trained band (deg)")
    ax[2].set_ylabel("endpoint error (m)"); ax[2].set_title("generalization")
    ax[2].grid(alpha=0.3); ax[2].legend(fontsize=6)
    fig.tight_layout()
    p = os.path.join(outdir, "jacobian_teacher.png")
    fig.savefig(p, dpi=150)
    print(f"\n[plot] wrote {p}")


if __name__ == "__main__":
    main()
