"""Aggregation for aleatoric_flip (local, no compute).

Forked from `../priced_plasticity/analyze_priced.py` (the frontier machinery is unchanged
and deliberately so: a modulated-plasticity rule is graded against the WHOLE one-parameter
family of ungated uniform learning rates, not against a matched-budget point). Three things
are added, all of them about the manipulation this node introduces:

  * `--damage`   the damage-channel ladder. Retention is read off the flip region's CLEAN
                 probe (the same query set stepped in a world where the flip region's noise
                 is off, so its Delta s IS the conditional mean); the ladder crosses flip
                 amplitude with uniform learning rate, and the `f0` column is the no-flip
                 control that separates OVER-WRITING damage from interference-from-elsewhere.
  * `--condmean` the load-bearing assumption, measured: is E[s'|s,u] in the flip region
                 unchanged by switching the noise on? chi2/dof is 1.0 at the null; the
                 `base` row is a region where the two worlds are literally identical, so it
                 is a null draw from the same instrument.
  * `--disag`    the shadow ensemble's per-class disagreement profile, in both variants
                 (`shared` batches vs per-member `boot`strap), by quartile of the run.

A `cell` here is (capacity, flip amplitude): every arm inside one cell sees the same stream,
the same pretrained FM and the same world, so the uniform-lr family inside it is the right
control for everything else inside it.

Usage (from experiments/):
    python3 mjc/practice/aleatoric_flip/analyze_flip.py --tags afcal_s0 --pull --damage
    python3 mjc/practice/aleatoric_flip/analyze_flip.py --tags af_s0 --pull --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
VOLUME = "mujoco-control-data"


# ------------------------------------------------------------------ io
def pull(tag):
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, f"{tag}.json")
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME,
                        f"aleatoric_flip/{tag}/results.json", dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[pull] {tag}: {r.stderr.strip().splitlines()[-1] if r.stderr else 'failed'}")
    return r.returncode == 0


def load(tag, do_pull=False, allow_partial=False):
    p = os.path.join(DATA, f"{tag}.json")
    if do_pull or not os.path.exists(p):
        pull(tag)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        R = json.load(fh)
    if not R.get("complete") and not allow_partial:
        print(f"[load] {tag}: incomplete (mid-run checkpoint); use --partial to read anyway")
        return None
    return R


# ------------------------------------------------------------------ readouts
AGG_MODES = ("half", "q3", "last4", "final")


def late_probe(arm, key, mode="half"):
    t = np.asarray(arm["trace"]["t"], float)
    k = f"probe_{key}"
    if k not in arm["trace"]:
        return np.nan
    y = np.asarray(arm["trace"][k], float)
    if len(t) == 0:
        return np.nan
    if mode == "final":
        return float(y[-1])
    if mode == "last4":
        return float(y[-4:].mean())
    return float(y[t >= (0.75 if mode == "q3" else 0.5) * t.max()].mean())


def final_probe(arm, key):
    k = f"probe_{key}"
    return float(arm["trace"][k][-1]) if k in arm["trace"] else np.nan


def late_ctrl(arm, key, window):
    lad = arm.get("ladder")
    if not lad:
        return np.nan
    v = [r[key] for r in lad if r.get("transitions", 0) >= window and key in r]
    return float(np.mean(v)) if v else np.nan


def flip_names(R):
    return [r["name"] for r in R["config"]["regions"] if r["noise_pre"] != r["noise"]]


def drift_key(R):
    for r in R["config"]["regions"]:
        if r["noise"] == 0 and not r["pre"] and r["phi"] != 0.0:
            return r["name"]
    return None


def mastered_key(R):
    """The retention readout. When the mastered region is a FLIP region its noisy probe is
    dominated by the irreducible floor, so retention is read off the paired CLEAN probe."""
    for r in R["config"]["regions"]:
        if r["pre"]:
            return r["name"] + ("@clean" if r["noise_pre"] != r["noise"] else "")
    return None


def noise_key(R):
    """The always-aleatoric decoy (aleatoric from the start, off the eval path)."""
    for r in R["config"]["regions"]:
        if r["noise"] > 0 and r["noise_pre"] == r["noise"]:
            return r["name"]
    return None


# ------------------------------------------------------------------ frontier
def interp_on(curve_x, curve_y, x):
    o = np.argsort(curve_x)
    cx, cy = np.asarray(curve_x)[o], np.asarray(curve_y)[o]
    if x < cx[0] or x > cx[-1]:
        return np.nan
    return float(np.interp(x, cx, cy))


def pareto(points):
    keep = []
    for i, (a, r, lb) in enumerate(points):
        dom = any((aa <= a and rr <= r and (aa < a or rr < r))
                  for j, (aa, rr, _) in enumerate(points) if j != i)
        if not dom:
            keep.append((a, r, lb))
    return sorted(keep, key=lambda p: p[0]), [p[2] for p in points if p not in keep]


def frontier_stats(uni_a, uni_r, pa, pr):
    r_at = interp_on(uni_a, uni_r, pa)
    a_at = interp_on(uni_r, uni_a, pr)
    span_r = float(np.nanmax(uni_r) - np.nanmin(uni_r))
    span_a = float(np.nanmax(uni_a) - np.nanmin(uni_a))
    return dict(ret_gain=r_at - pr if np.isfinite(r_at) else np.nan,
                adapt_gain=a_at - pa if np.isfinite(a_at) else np.nan,
                ret_gain_pct=100 * (r_at - pr) / span_r if (np.isfinite(r_at) and span_r > 0) else np.nan,
                adapt_gain_pct=100 * (a_at - pa) / span_a if (np.isfinite(a_at) and span_a > 0) else np.nan,
                uni_ret_at_matched_adapt=r_at, uni_adapt_at_matched_ret=a_at)


def curve_gap(ref_a, ref_r, q_a, q_r, n=64):
    if len(q_a) < 2:
        return np.nan, np.nan
    lo = max(min(ref_a), min(q_a)); hi = min(max(ref_a), max(q_a))
    if not (hi > lo):
        return np.nan, np.nan
    xs = np.linspace(lo, hi, n)
    d = [interp_on(ref_a, ref_r, x) - interp_on(q_a, q_r, x) for x in xs]
    span_r = float(np.nanmax(ref_r) - np.nanmin(ref_r))
    return float(np.nanmean(d)), (100 * float(np.nanmean(d)) / span_r if span_r > 0 else np.nan)


# ------------------------------------------------------------------ tables
def arm_rows(R, ctrl_window, agg="half"):
    dk, mk, nk = drift_key(R), mastered_key(R), noise_key(R)
    fk = flip_names(R)
    rows = []
    for name, arm in R["arms"].items():
        sp = arm["spec"]; bud = arm["budget"]
        row = dict(name=name, kind=sp["kind"], lr=sp["lr"], spend_mode=sp["spend_mode"],
                   norm_mode=sp["norm_mode"], n_replay=sp["n_replay"],
                   h=sp["fm_hidden"], L=sp["fm_layers"], w_mult=sp["w_mult"],
                   flip_amp=sp["flip_amp"], tau_mode=sp.get("tau_mode", "pretrain"),
                   ens_mode=sp.get("ens_mode", "shared"),
                   disag_law=sp.get("disag_law", "lin"), mask_w=sp.get("mask_w", 0.0),
                   stack=arm["stack"], cell=arm["cell"],
                   mean_w=bud["mean_w"], mean_spend=bud["mean_spend"],
                   eff_lr=bud["eff_lr"], clip_frac=bud["clip_frac"],
                   tau_used=bud.get("tau_used", np.nan),
                   share=bud["cum_w_share"], sample_share=bud["sample_share"],
                   adapt_fm=late_probe(arm, dk, agg), retain_fm=late_probe(arm, mk, agg),
                   base_fm=late_probe(arm, "base", agg),
                   noise_fm=late_probe(arm, nk, agg) if nk else np.nan,
                   glob_fm=late_probe(arm, "global", agg),
                   adapt_fm_final=final_probe(arm, dk),
                   retain_fm_final=final_probe(arm, mk),
                   flip_noisy_fm=late_probe(arm, fk[0], agg) if fk else np.nan)
        for c in R["config"]["controllers"]:
            row[f"adapt_{c}"] = late_ctrl(arm, f"drift/{c}", ctrl_window)
            row[f"retain_{c}"] = late_ctrl(arm, f"mastered/{c}", ctrl_window)
            row[f"retain_noisy_{c}"] = late_ctrl(arm, f"mastered_noisy/{c}", ctrl_window)
            row[f"agg_{c}"] = late_ctrl(arm, f"agg/{c}", ctrl_window)
        rows.append(row)
    return rows


def cell_key(r):
    return (r["stack"], r["flip_amp"])


def short(name):
    return name.split("@")[0] + (":" + ":".join(p for p in name.split(":")[1:]
                                                if not p.startswith("f")) if ":" in name else "")


# ------------------------------------------------------------------ the manipulation
def print_condmean(R):
    cm = R.get("cond_mean") or {}
    if not cm:
        return
    print("\n=== the load-bearing assumption: is E[s'|s,u] unchanged by the flip? ===")
    print("  (chi2/dof = 1.0 at the null 'the noise is mean-preserving'; `base` is a region")
    print("   where the two worlds are LITERALLY identical, i.e. a null draw from the same")
    print("   instrument. `bias` is the per-sample MC deviation, `se` its expected size.)")
    print(f"{'amp':>6s} {'region':12s} {'n':>5s} {'reps':>5s} {'|d_det|':>9s} {'noise_sd':>9s} "
          f"{'bias':>9s} {'se':>9s} {'bias/se':>8s} {'chi2/dof':>12s} {'pooled':>9s} "
          f"{'+/-':>9s} {'b/(split/2)':>12s}")
    for amp, d in sorted(cm.items(), key=lambda kv: float(kv[0])):
        for nm, v in d.items():
            print(f"{amp:>6s} {nm:12s} {v['n']:5d} {v['reps']:5d} {v.get('det', np.nan):9.5f} "
                  f"{v['sd']:9.5f} {v['bias']:9.5f} {v['se']:9.5f} {v['bias_over_se']:8.2f} "
                  f"{v['chi2_dof']:9.3f}±{v['chi2_dof_res']:.3f} "
                  f"{v['pooled']:9.5f} {v['pooled_se']:9.5f} "
                  f"{v.get('split_ratio', float('nan')):12.3f}")


def print_disag(R):
    cells = R.get("cells") or {}
    if not cells:
        return
    print("\n=== shadow-ensemble disagreement by region class (q1 -> q4 of the run) ===")
    for ck, c in sorted(cells.items()):
        dp = c.get("disag_profile") or {}
        for variant, prof in dp.items():
            print(f"  [{ck} / {variant}]")
            print(f"    {'class':12s} {'mean':>9s} {'q1':>9s} {'q2':>9s} {'q3':>9s} {'q4':>9s}")
            for nm, v in prof.items():
                print(f"    {nm:12s} {v['mean']:9.5f} {v['q1']:9.5f} {v['q2']:9.5f} "
                      f"{v['q3']:9.5f} {v['q4']:9.5f}")


def print_damage(R, rows, agg):
    """The damage-channel ladder. Retention = the flip region's CLEAN probe; the f0 column
    is the no-flip control, so `flip - noflip` at matched lr isolates OVER-WRITING damage
    from interference-from-elsewhere (which the parent showed is what erodes this corridor
    in every prior geometry)."""
    mk, dk = mastered_key(R), drift_key(R)
    fk = flip_names(R)
    print("\n=== the damage channel: does uniform plasticity damage the mastered region, "
          "and does the damage scale with the flip? ===")
    print(f"  retention readout = probe `{mk}`   adaptation readout = probe `{dk}`")
    amps = sorted({r["flip_amp"] for r in rows})
    for ck, c in sorted((R.get("cells") or {}).items()):
        sp = c["stale_probe"]
        print(f"  [{ck}] stale: " + " ".join(f"{k}={v:.4f}" for k, v in sp.items()))
    uni = [r for r in rows if r["kind"] == "fixed" and r["w_mult"] == 1.0]
    lrs = sorted({r["lr"] for r in uni})
    print(f"\n  {'uniform lr':>11s} | " +
          " | ".join(f"amp={a:g}".rjust(19) for a in amps))
    print(f"  {'':>11s} | " + " | ".join(f"{'retain':>9s} {'adapt':>9s}" for _ in amps))
    for lr in lrs:
        cells_ = []
        for a in amps:
            m = [r for r in uni if r["lr"] == lr and r["flip_amp"] == a]
            cells_.append(f"{m[0]['retain_fm']:9.4f} {m[0]['adapt_fm']:9.4f}" if m
                          else " " * 19)
        print(f"  {lr:11.1e} | " + " | ".join(cells_))
    base_amp = min(amps)
    if len(amps) > 1:
        print(f"\n  over-writing damage = retention(amp) - retention(amp={base_amp:g}) "
              f"at matched lr  (>0 = the flip damages)")
        print(f"  {'uniform lr':>11s} | " + " | ".join(f"amp={a:g}".rjust(10)
                                                       for a in amps if a != base_amp))
        for lr in lrs:
            b = [r for r in uni if r["lr"] == lr and r["flip_amp"] == base_amp]
            if not b:
                continue
            out = []
            for a in amps:
                if a == base_amp:
                    continue
                m = [r for r in uni if r["lr"] == lr and r["flip_amp"] == a]
                out.append(f"{m[0]['retain_fm'] - b[0]['retain_fm']:+10.4f}" if m else " " * 10)
            print(f"  {lr:11.1e} | " + " | ".join(out))
    # the same contrast for every modulated arm present at both amps
    mods = [r for r in rows if not (r["kind"] == "fixed" and r["w_mult"] == 1.0)]
    if mods:
        print(f"\n  {'arm':28s} {'amp':>5s} {'spend':>7s} {'retain':>9s} {'adapt':>9s} "
              f"{'noisy-flip':>11s} " +
              " ".join(f"r[{k}]".rjust(9) for k in (mods[0]["share"] or {})))
        for r in sorted(mods, key=lambda r: (r["kind"], r["flip_amp"], r["lr"])):
            rel = " ".join(f"{r['share'][k] / max(r['sample_share'][k], 1e-9):9.3f}"
                           for k in r["share"])
            print(f"  {r['name']:28s} {r['flip_amp']:5.0f} {r['mean_spend']:7.3f} "
                  f"{r['retain_fm']:9.4f} {r['adapt_fm']:9.4f} {r['flip_noisy_fm']:11.4f} {rel}")


def print_protection(R, rows):
    """The 2x2 the node turns on: (flip on/off) x (oracle protection on/off), at matched
    learning rate. `damage` = what the flip costs an unprotected uniform learner;
    `protection value` = what withholding plasticity from the flipped region buys back;
    `protection cost` = what the same withholding costs when the region is NOT flipped."""
    om = [r for r in rows if r["kind"] == "omask"]
    if not om:
        return
    print("\n=== the value of protection, with an ORACLE mask (no signal involved) ===")
    print(f"  {'lr':>9s} {'amp':>5s} {'mask_w':>7s} {'retain':>9s} {'adapt':>9s} "
          f"{'vs uniform (retain)':>21s} {'(adapt)':>10s}")
    uni = {(r["lr"], r["flip_amp"]): r for r in rows
           if r["kind"] == "fixed" and r["w_mult"] == 1.0}
    for r in sorted(om, key=lambda r: (r["lr"], r["flip_amp"], r["mask_w"])):
        u = uni.get((r["lr"], r["flip_amp"]))
        dr = r["retain_fm"] - u["retain_fm"] if u else float("nan")
        da = r["adapt_fm"] - u["adapt_fm"] if u else float("nan")
        print(f"  {r['lr']:9.1e} {r['flip_amp']:5.0f} {r['mask_w']:7.2f} {r['retain_fm']:9.4f} "
              f"{r['adapt_fm']:9.4f} {dr:21.4f} {da:10.4f}   "
              f"({'protection HELPS' if dr < 0 else 'protection HURTS'} retention)")
    amps = sorted({r["flip_amp"] for r in rows})
    if len(amps) > 1:
        lo, hi = min(amps), max(amps)
        print(f"\n  {'lr':>9s} {'damage(uniform)':>16s} {'protected damage':>17s} "
              f"{'protection cost @noflip':>24s}")
        for lr in sorted({r["lr"] for r in om}):
            u0, u1 = uni.get((lr, lo)), uni.get((lr, hi))
            m0 = next((r for r in om if r["lr"] == lr and r["flip_amp"] == lo
                       and r["mask_w"] == 0.0), None)
            m1 = next((r for r in om if r["lr"] == lr and r["flip_amp"] == hi
                       and r["mask_w"] == 0.0), None)
            if not (u0 and u1 and m1):
                continue
            dmg = u1["retain_fm"] - u0["retain_fm"]
            pdmg = (m1["retain_fm"] - m0["retain_fm"]) if m0 else float("nan")
            cost = (m0["retain_fm"] - u0["retain_fm"]) if m0 else float("nan")
            print(f"  {lr:9.1e} {dmg:16.4f} {pdmg:17.4f} {cost:24.4f}")


def print_calibration(R, rows):
    print("\n=== stacks (pretrained on the PRE-drift world; identical across flip amps) ===")
    for k, s in R["stacks"].items():
        print(f"  {k}: tau={s['tau']:.5f} clean_err={s['pretrain_fm_err_clean']:.4f} "
              f"gateAUROC={s['gate_auroc_train']:.4f} d_ref={s.get('d_ref', np.nan):.5f} "
              f"(K={s.get('ens_k')}, rpf_beta={s.get('rpf_beta')})")
    print("\n=== cells (stale probes + the online tau the `:tauon` arm uses) ===")
    for k, c in sorted((R.get("cells") or {}).items()):
        print(f"  {k}: tau_online={c['tau_online']:.5f}  " +
              " ".join(f"{n}={v:.4f}" for n, v in c["stale_probe"].items()))
        if "stale" in c:
            print("      stale ctrl:   " + "  ".join(
                f"{kk}={vv:.4f}" for kk, vv in c["stale"].items()
                if kk not in ("fm", "transitions")))
            print("      ceiling ctrl: " + "  ".join(
                f"{kk}={vv:.4f}" for kk, vv in c["ceiling"].items()
                if kk not in ("fm", "transitions")))


def print_frontier(R, rows, axis_a, axis_r, label):
    out = {}
    for ck in sorted({cell_key(r) for r in rows}):
        cr = [r for r in rows if cell_key(r) == ck]
        uni = sorted([r for r in cr if r["kind"] == "fixed" and r["w_mult"] == 1.0],
                     key=lambda r: r["lr"])
        oth = [r for r in cr if not (r["kind"] == "fixed" and r["w_mult"] == 1.0)]
        pts = [(r[axis_a], r[axis_r], r) for r in uni]
        if not all(np.isfinite(p[0]) and np.isfinite(p[1]) for p in pts) or len(pts) < 2:
            print(f"\n[{label} | {ck}] readout unavailable"); continue
        front, dominated = pareto(pts)
        ua = [p[0] for p in front]; ur = [p[1] for p in front]
        print(f"\n=== frontier: {label}   cell {ck[0]}, flip_amp={ck[1]:g} ===")
        print(f"{'uniform lr':>12s} {'adapt':>9s} {'retain':>9s}   on-frontier")
        for a_, r_, rr in pts:
            print(f"{rr['lr']:12.1e} {a_:9.4f} {r_:9.4f}   "
                  f"{'yes' if (a_, r_, rr) in front else 'DOMINATED'}")
        print(f"  frontier spans: adaptation {max(ua)-min(ua):.4f}   "
              f"retention {max(ur)-min(ur):.4f}   ({len(front)}/{len(pts)} lrs on frontier)")
        if not oth:
            out[str(ck)] = dict(frontier=[(a, r) for a, r, _ in front], arms=[])
            continue
        print(f"\n{'arm':28s} {'eff_lr':>9s} {'adapt':>9s} {'retain':>9s} "
              f"{'retGain':>9s} {'%span':>7s} {'adaptGain':>10s} {'%span':>7s} {'outside':>8s}")
        recs = []
        for r in sorted(oth, key=lambda r: (r["kind"], r["lr"])):
            fs = frontier_stats(ua, ur, r[axis_a], r[axis_r])
            outside = (np.isfinite(fs["ret_gain"]) and fs["ret_gain"] > 0) and \
                      (np.isfinite(fs["adapt_gain"]) and fs["adapt_gain"] > 0)
            print(f"{r['name']:28s} {r['eff_lr']:9.1e} {r[axis_a]:9.4f} {r[axis_r]:9.4f} "
                  f"{fs['ret_gain']:9.4f} {fs['ret_gain_pct']:7.1f} {fs['adapt_gain']:10.4f} "
                  f"{fs['adapt_gain_pct']:7.1f} {str(outside):>8s}")
            recs.append(dict(arm=r["name"], **fs))
        fams = {}
        for r in oth:
            if r["spend_mode"] == "unit":
                continue
            fams.setdefault((r["kind"] + (f"@m{r['mask_w']:g}" if r["kind"] == "omask" else ""),
                             r["tau_mode"], r["ens_mode"], r["disag_law"]), []).append(
                (r[axis_a], r[axis_r]))
        print(f"\n  {'family':30s} {'n':>3s} {'mean retention gap vs uniform frontier':>40s}")
        for k, ps in fams.items():
            ps = sorted(ps)
            g, gp = curve_gap(ua, ur, [p[0] for p in ps], [p[1] for p in ps])
            print(f"  {'/'.join(k):30s} {len(ps):3d} {g:20.5f}  ({gp:+.1f}% of frontier span)")
        out[str(ck)] = dict(frontier=[(a, r) for a, r, _ in front],
                            dominated=[d["lr"] for d in dominated], arms=recs)
    return out


# ------------------------------------------------------------------ figures
COL = {"delta": "#7048e8", "raw_err": "#e8590c", "fixed": "#495057",
       "disag": "#0ca678", "conj": "#1971c2", "omask": "#c2255c"}


def make_figures(R, rows, tag, ctrl_window, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    mk, dk = mastered_key(R), drift_key(R)

    axes_sets = [("adapt_fm", "retain_fm", f"FM probe ({dk} vs {mk})")]
    for c in R["config"]["controllers"]:
        if np.isfinite(rows[0].get(f"adapt_{c}", np.nan)):
            axes_sets.append((f"adapt_{c}", f"retain_{c}", f"control, {c}"))
    cells = sorted({cell_key(r) for r in rows})

    fig, axs = plt.subplots(len(cells), len(axes_sets),
                            figsize=(5.0 * len(axes_sets), 4.4 * len(cells)), squeeze=False)
    for ci, ck in enumerate(cells):
        cr = [r for r in rows if cell_key(r) == ck]
        for ai, (ka, kr, ttl) in enumerate(axes_sets):
            ax = axs[ci][ai]
            uni = sorted([r for r in cr if r["kind"] == "fixed" and r["w_mult"] == 1.0],
                         key=lambda r: r["lr"])
            if len(uni) >= 2:
                front, _ = pareto([(r[ka], r[kr], r) for r in uni])
                ax.plot([r[ka] for r in uni], [r[kr] for r in uni], ":", color=COL["fixed"],
                        lw=0.9, zorder=1)
                ax.plot([p[0] for p in front], [p[1] for p in front], "-o", color=COL["fixed"],
                        ms=4, lw=1.6, label="uniform lr (the frontier)", zorder=2)
                for r in uni:
                    ax.annotate(f"{r['lr']:.0e}", (r[ka], r[kr]), fontsize=6,
                                textcoords="offset points", xytext=(4, -8), color=COL["fixed"])
            for kind in COL:
                if kind == "fixed":
                    continue
                fam = sorted([r for r in cr if r["kind"] == kind
                              and r["spend_mode"] != "unit"], key=lambda r: r[ka])
                if len(fam) > 1:
                    ax.plot([r[ka] for r in fam], [r[kr] for r in fam], "-",
                            color=COL[kind], lw=1.3, alpha=0.8, zorder=2)
            for r in cr:
                if r["kind"] == "fixed" and r["w_mult"] == 1.0:
                    continue
                c = COL.get(r["kind"], "#0c8599")
                m = "D" if r["spend_mode"] == "unit" else "*"
                ax.plot([r[ka]], [r[kr]], m, color=c, ms=13 if m == "*" else 7,
                        mec="k", mew=0.5, label=r["name"], zorder=3)
            ax.set_xlabel(f"adaptation: {ka}  (lower = better)")
            ax.set_ylabel(f"retention: {kr}  (lower = better)")
            ax.set_title(f"{ttl}\ncell {ck[0]}, flip_amp={ck[1]:g}", fontsize=8.5)
            ax.grid(alpha=0.25, lw=0.5)
            if ai == 0:
                ax.legend(fontsize=6.0, loc="best")
    fig.suptitle(f"aleatoric_flip {tag}: with a direct over-writing damage channel, does "
                 f"any gain sit outside the uniform-lr frontier?", fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIGS, f"fig1_frontier_{tag}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[figures] wrote {p}")

    # fig2: the retention trace (the damage channel, over time) + allocation
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for r in sorted(rows, key=lambda r: (r["flip_amp"], r["kind"], r["lr"])):
        arm = R["arms"][r["name"]]
        k = f"probe_{mk}"
        if k not in arm["trace"]:
            continue
        ls = "-" if r["flip_amp"] > 0 else "--"
        axs[0].plot(arm["trace"]["t"], arm["trace"][k], ls, lw=1.1,
                    color=COL.get(r["kind"], "#0c8599"),
                    alpha=0.35 + 0.65 * (r["kind"] != "fixed"), label=r["name"])
    axs[0].set_xlabel("transitions"); axs[0].set_ylabel(f"probe {mk}  (retention)")
    axs[0].set_title("the damage channel over time\n(solid = flipped, dashed = no-flip control)",
                     fontsize=8.5)
    axs[0].legend(fontsize=5.5, ncol=2)
    mod = [r for r in rows if r["kind"] != "fixed"]
    if mod:
        cls = list(rows[0]["share"].keys())
        wid = 0.8 / max(len(mod), 1)
        for i, r in enumerate(mod):
            vals = [r["share"][c] / max(r["sample_share"][c], 1e-9) for c in cls]
            axs[1].bar(np.arange(len(cls)) + i * wid, vals, wid, label=r["name"],
                       color=COL.get(r["kind"], "#0c8599"),
                       alpha=0.5 + 0.5 * (r["flip_amp"] > 0))
        axs[1].axhline(1.0, color="k", lw=0.8, ls="--")
        axs[1].set_xticks(np.arange(len(cls)) + 0.4 - wid / 2)
        axs[1].set_xticklabels(cls, fontsize=7)
        axs[1].set_ylabel("realized weight share / sample share")
        axs[1].set_title("allocation (1.0 = uniform)", fontsize=8.5)
        axs[1].legend(fontsize=5.5, ncol=2)
        for r in mod:
            arm = R["arms"][r["name"]]
            axs[2].plot(arm["blog"]["t"], np.convolve(
                np.asarray(arm["blog"]["spend"], float), np.ones(21) / 21, mode="same"),
                lw=1.1, color=COL.get(r["kind"], "#0c8599"),
                alpha=0.5 + 0.5 * (r["flip_amp"] > 0), label=r["name"])
        axs[2].axhline(1.0, color="k", lw=0.8, ls="--")
        axs[2].set_xlabel("transitions"); axs[2].set_ylabel("mean batch spend (= lr multiplier)")
        axs[2].set_title("total plasticity spend", fontsize=8.5)
        axs[2].legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    p = os.path.join(FIGS, f"fig2_damage_{tag}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[figures] wrote {p}")

    # fig3: per-class weight traces for the modulated arms in the flipped cell
    fk = flip_names(R)
    mods = [r for r in rows if r["kind"] != "fixed" and r["flip_amp"] > 0]
    if mods and fk:
        cls = list(rows[0]["share"].keys())
        fig, axs = plt.subplots(1, len(cls), figsize=(3.6 * len(cls), 3.4), squeeze=False)
        for j, cnm in enumerate(cls):
            ax = axs[0][j]
            for r in mods:
                b = R["arms"][r["name"]]["blog"]
                y = np.asarray(b[f"w_{cnm}"], float)
                m = np.isfinite(y)
                if m.sum() > 21:
                    ax.plot(np.asarray(b["t"], float)[m],
                            np.convolve(y[m], np.ones(21) / 21, mode="same"),
                            lw=1.1, color=COL.get(r["kind"], "#0c8599"), label=r["name"])
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.set_title(cnm, fontsize=8.5); ax.set_xlabel("transitions")
            if j == 0:
                ax.set_ylabel("mean plasticity weight"); ax.legend(fontsize=5.5)
        fig.suptitle("where each gain sends its plasticity, over time (flipped cell)",
                     fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIGS, f"fig3_weights_{tag}.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"[figures] wrote {p}")


def merge(Rs):
    base = json.loads(json.dumps(Rs[0]))
    for R in Rs[1:]:
        assert R["config"]["seed"] == base["config"]["seed"], "merge across seeds"
        assert R["config"]["regions"] == base["config"]["regions"], "merge across geometries"
        for k, v in R["stacks"].items():
            base["stacks"].setdefault(k, v)
        for k, v in (R.get("cells") or {}).items():
            base.setdefault("cells", {}).setdefault(k, v)
        for k, v in (R.get("cond_mean") or {}).items():
            base.setdefault("cond_mean", {}).setdefault(k, v)
        for k, v in R["arms"].items():
            if k in base["arms"]:
                print(f"[merge] duplicate arm {k}: keeping the first")
                continue
            base["arms"][k] = v
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="af_s0")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--agg", default="half", choices=list(AGG_MODES))
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--damage", action="store_true", help="the damage-channel ladder")
    ap.add_argument("--condmean", action="store_true", help="the mean-preservation check")
    ap.add_argument("--disag", action="store_true", help="the disagreement profile")
    ap.add_argument("--all", action="store_true", help="every table")
    ap.add_argument("--ctrl-window", type=int, default=2048)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    summary = {}
    tags = [t for t in a.tags.split(",") if t]
    if a.merge:
        Rs = [r for r in (load(t, a.pull, a.partial) for t in tags) if r is not None]
        if not Rs:
            print("[skip] nothing to merge"); return
        groups = [("+".join(tags), merge(Rs))]
    else:
        groups = [(t, load(t, a.pull, a.partial)) for t in tags]
    for tag, R in groups:
        if R is None:
            print(f"[skip] {tag}"); continue
        rows = arm_rows(R, a.ctrl_window, a.agg)
        print(f"\n################ {tag}  (mode={R['config']['mode']}, "
              f"{len(rows)} arms, seed {R['config']['seed']}, agg={a.agg}) ################")
        print_calibration(R, rows)
        if a.condmean or a.all:
            print_condmean(R)
        if a.disag or a.all:
            print_disag(R)
        if a.damage or a.all or R["config"]["mode"] == "calibrate":
            print_damage(R, rows, a.agg)
            print_protection(R, rows)
        if R["config"]["mode"] != "calibrate":
            fr = {"fm": print_frontier(R, rows, "adapt_fm", "retain_fm", "FM probe")}
            for c in R["config"]["controllers"]:
                fr[c] = print_frontier(R, rows, f"adapt_{c}", f"retain_{c}", f"control/{c}")
            summary[tag] = {"rows": rows, "frontier": fr}
            if a.figures:
                make_figures(R, rows, tag.replace("+", "_"), a.ctrl_window, a.agg)
    if a.json and summary:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(f"[json] wrote {a.json}")


if __name__ == "__main__":
    main()
