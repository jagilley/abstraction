"""Aggregation for priced_plasticity (local, no compute).

The grading instrument of this node is a FRONTIER, not a matched-budget point. The
`fixed` arm swept across a grid of uniform learning rates traces a curve in the
(adaptation, retention) plane; delta / delta:unit / raw_err are single points on the same
plane. The question is whether any of them sits OUTSIDE the curve the scalar knob draws.

Readouts
  * FM level (the tight, seed-stable one, per this arc's convention): late-window mean of
    the per-region probe errors -- `A-drift` = adaptation, `B-mastered` = retention,
    `base` = the untouched-space canary.
  * control level: late-window (m >= `--ctrl-window`) mean of `drift/<ctrl>` (adaptation)
    and `mastered/<ctrl>` (retention) goal distance.

Frontier statistics for each non-`fixed` arm P:
    ret_gain  = R_uniform(A_P) - R_P     (retention saved at MATCHED adaptation; >0 = P
                                          is better than any scalar lr at its own
                                          adaptation level)
    adapt_gain= A_uniform(R_P) - A_P     (the transpose)
Both by linear interpolation along the uniform curve, reported in absolute units and as a
fraction of the uniform curve's own span on that axis. `outside` = both > 0.

Usage (from experiments/):
    python3 mjc/practice/priced_plasticity/analyze_priced.py --tags cal_s0 --pull
    python3 mjc/practice/priced_plasticity/analyze_priced.py --tags pp_s0 --figures
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
                        f"priced_plasticity/{tag}/results.json", dst],
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
    """The aggregation that DEFINES the frontier. Every quantitative claim in this node is
    relative to this choice, so it is a named, switchable parameter rather than a default
    buried in a helper:
      half  -- mean over t >= T/2   (35 of 69 probe points; the primary readout)
      q3    -- mean over t >= 3T/4
      last4 -- mean of the last four probe points (~ end state, still averaged)
      final -- the single last probe (lowest bias, highest variance)
    They disagree at the top of the learning-rate grid; see README `Method / aggregation`."""
    t = np.asarray(arm["trace"]["t"], float)
    y = np.asarray(arm["trace"][f"probe_{key}"], float)
    if len(t) == 0:
        return np.nan
    if mode == "final":
        return float(y[-1])
    if mode == "last4":
        return float(y[-4:].mean())
    return float(y[t >= (0.75 if mode == "q3" else 0.5) * t.max()].mean())


def final_probe(arm, key):
    return float(arm["trace"][f"probe_{key}"][-1])


def late_ctrl(arm, key, window):
    lad = arm.get("ladder")
    if not lad:
        return np.nan
    v = [r[key] for r in lad if r.get("transitions", 0) >= window and key in r]
    return float(np.mean(v)) if v else np.nan


def region_names(R):
    return [r["name"] for r in R["config"]["regions"]]


def drift_key(R):
    for r in R["config"]["regions"]:
        if r["noise"] == 0 and not r["pre"] and r["phi"] != 0.0:
            return r["name"]
    return None


def mastered_key(R):
    for r in R["config"]["regions"]:
        if r["pre"]:
            return r["name"]
    return None


def noise_key(R):
    for r in R["config"]["regions"]:
        if r["noise"] > 0:
            return r["name"]
    return None


# ------------------------------------------------------------------ frontier
def interp_on(curve_x, curve_y, x):
    """Linear interpolation of y at x along a curve sorted by x. NaN outside the span."""
    o = np.argsort(curve_x)
    cx, cy = np.asarray(curve_x)[o], np.asarray(curve_y)[o]
    if x < cx[0] or x > cx[-1]:
        return np.nan
    return float(np.interp(x, cx, cy))


def pareto(points):
    """points: list of (a, r, label). Keep only the non-dominated ones (both axes: lower
    is better).  A scalar knob that runs past its own knee produces dominated settings --
    those are not part of the frontier it can offer and must not be interpolated through."""
    keep = []
    for i, (a, r, lb) in enumerate(points):
        dom = any((aa <= a and rr <= r and (aa < a or rr < r))
                  for j, (aa, rr, _) in enumerate(points) if j != i)
        if not dom:
            keep.append((a, r, lb))
    return sorted(keep, key=lambda p: p[0]), [p[2] for p in points if p not in keep]


def frontier_stats(uni_a, uni_r, pa, pr):
    """uni_* : the reference curve (adaptation, retention). p* : the query point."""
    r_at = interp_on(uni_a, uni_r, pa)          # reference retention at P's adaptation
    a_at = interp_on(uni_r, uni_a, pr)          # reference adaptation at P's retention
    span_r = float(np.nanmax(uni_r) - np.nanmin(uni_r))
    span_a = float(np.nanmax(uni_a) - np.nanmin(uni_a))
    return dict(ret_gain=r_at - pr if np.isfinite(r_at) else np.nan,
                adapt_gain=a_at - pa if np.isfinite(a_at) else np.nan,
                ret_gain_pct=100 * (r_at - pr) / span_r if (np.isfinite(r_at) and span_r > 0) else np.nan,
                adapt_gain_pct=100 * (a_at - pa) / span_a if (np.isfinite(a_at) and span_a > 0) else np.nan,
                uni_ret_at_matched_adapt=r_at, uni_adapt_at_matched_ret=a_at)


def curve_gap(ref_a, ref_r, q_a, q_r, n=64):
    """Mean vertical gap (reference retention - query retention) over the adaptation range
    where the two curves overlap. >0 = the query curve lies BELOW the reference, i.e. it
    offers a strictly better tradeoff over that range. NaN if fewer than 2 query points
    or no overlap."""
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
    rows = []
    for name, arm in R["arms"].items():
        sp = arm["spec"]; bud = arm["budget"]
        row = dict(name=name, kind=sp["kind"], lr=sp["lr"], spend_mode=sp["spend_mode"],
                   norm_mode=sp["norm_mode"], n_replay=sp["n_replay"],
                   h=sp["fm_hidden"], L=sp["fm_layers"], w_mult=sp["w_mult"],
                   stack=arm["stack"],
                   mean_w=bud["mean_w"], mean_spend=bud["mean_spend"],
                   eff_lr=bud["eff_lr"], clip_frac=bud["clip_frac"],
                   share=bud["cum_w_share"], sample_share=bud["sample_share"],
                   adapt_fm=late_probe(arm, dk, agg), retain_fm=late_probe(arm, mk, agg),
                   base_fm=late_probe(arm, "base", agg), noise_fm=late_probe(arm, nk, agg),
                   glob_fm=late_probe(arm, "global", agg),
                   adapt_fm_final=final_probe(arm, dk),
                   retain_fm_final=final_probe(arm, mk))
        for c in R["config"]["controllers"]:
            row[f"adapt_{c}"] = late_ctrl(arm, f"drift/{c}", ctrl_window)
            row[f"retain_{c}"] = late_ctrl(arm, f"mastered/{c}", ctrl_window)
            row[f"agg_{c}"] = late_ctrl(arm, f"agg/{c}", ctrl_window)
        rows.append(row)
    return rows


def print_calibration(R, rows):
    stacks = R["stacks"]
    print("\n=== stacks (pretrained FM: is the mastered corridor actually mastered?) ===")
    print(f"{'stack':10s} {'tau':>9s} {'clean_err':>10s} {'gateAUROC':>10s}  stale probes")
    for k, s in stacks.items():
        sp = s["stale_probe"]
        print(f"{k:10s} {s['tau']:9.5f} {s['pretrain_fm_err_clean']:10.4f} "
              f"{s['gate_auroc_train']:10.4f}  " +
              " ".join(f"{n}={v:.4f}" for n, v in sp.items()))

    print("\n=== calibration cells (late-window FM probes; ratio = final/stale) ===")
    hdr = (f"{'arm':34s} {'h':>4s} {'rep':>4s} {'lr':>8s} {'spend':>7s} "
           f"{'adaptA':>8s} {'retB':>8s} {'base':>8s} {'B/stale':>8s} {'A/stale':>8s}")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["h"], r["n_replay"], r["lr"], r["name"])):
        st = R["stacks"][r["stack"]]["stale_probe"]
        dk, mk = drift_key(R), mastered_key(R)
        print(f"{r['name']:34s} {r['h']:4d} {r['n_replay']:4d} {r['lr']:8.1e} "
              f"{r['mean_spend']:7.3f} {r['adapt_fm']:8.4f} {r['retain_fm']:8.4f} "
              f"{r['base_fm']:8.4f} {r['retain_fm']/st[mk]:8.2f} {r['adapt_fm']/st[dk]:8.2f}")


def cell_key(r):
    """A `cell` is a regime, not an arm setting: capacity x replay. Every arm inside one
    cell sees the same data, the same pretrained FM and the same cost structure, so the
    uniform-lr family inside it is the right control for everything else inside it."""
    return (r["stack"], r["n_replay"])


def short(name):
    return name.split(":")[0]


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
        print(f"\n=== frontier: {label}   cell {ck[0]}, n_replay={ck[1]} ===")
        print(f"{'uniform lr':>12s} {'adapt':>9s} {'retain':>9s}   on-frontier")
        for a_, r_, rr in pts:
            print(f"{rr['lr']:12.1e} {a_:9.4f} {r_:9.4f}   "
                  f"{'yes' if (a_, r_, rr) in front else 'DOMINATED'}")
        print(f"  frontier spans: adaptation {max(ua)-min(ua):.4f}   "
              f"retention {max(ur)-min(ur):.4f}   ({len(front)}/{len(pts)} lrs on frontier)")
        print(f"\n{'arm':26s} {'eff_lr':>9s} {'adapt':>9s} {'retain':>9s} "
              f"{'retGain':>9s} {'%span':>7s} {'adaptGain':>10s} {'%span':>7s} {'outside':>8s}")
        recs = []
        for r in oth:
            fs = frontier_stats(ua, ur, r[axis_a], r[axis_r])
            outside = (np.isfinite(fs["ret_gain"]) and fs["ret_gain"] > 0) and \
                      (np.isfinite(fs["adapt_gain"]) and fs["adapt_gain"] > 0)
            print(f"{short(r['name']):26s} {r['eff_lr']:9.1e} {r[axis_a]:9.4f} {r[axis_r]:9.4f} "
                  f"{fs['ret_gain']:9.4f} {fs['ret_gain_pct']:7.1f} {fs['adapt_gain']:10.4f} "
                  f"{fs['adapt_gain_pct']:7.1f} {str(outside):>8s}")
            recs.append(dict(arm=r["name"], **fs))
        # curve-vs-curve: each modulated family against the uniform frontier, and delta
        # against raw_err (the hygiene claim, priced)
        fams = {}
        for r in oth:
            if r["spend_mode"] == "unit":
                continue                      # a single decomposition point, not a family
            fams.setdefault(r["kind"], []).append((r[axis_a], r[axis_r]))
        print(f"\n  {'family':16s} {'n':>3s} {'mean retention gap vs uniform frontier':>40s}")
        for k, ps in fams.items():
            ps = sorted(ps)
            g, gp = curve_gap(ua, ur, [p[0] for p in ps], [p[1] for p in ps])
            print(f"  {k:16s} {len(ps):3d} {g:20.5f}  ({gp:+.1f}% of frontier span)")
        if "delta" in fams and "raw_err" in fams:
            dp = sorted(fams["delta"]); rp = sorted(fams["raw_err"])
            g, gp = curve_gap([p[0] for p in rp], [p[1] for p in rp],
                              [p[0] for p in dp], [p[1] for p in dp])
            print(f"  {'delta vs raw_err':16s}     {g:20.5f}  "
                  f"(>0 = delta's tradeoff beats raw's)")
        out[str(ck)] = dict(frontier=[(a, r) for a, r, _ in front],
                            dominated=[d["lr"] for d in dominated], arms=recs)
    return out


# ------------------------------------------------------------------ figures
def make_figures(R, rows, tag, ctrl_window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"delta": "#7048e8", "raw_err": "#e8590c", "fixed": "#495057"}

    axes_sets = [("adapt_fm", "retain_fm", "FM probe (A-drift vs B-mastered)")]
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
            front, _ = pareto([(r[ka], r[kr], r) for r in uni])
            ax.plot([r[ka] for r in uni], [r[kr] for r in uni], ":", color=COL["fixed"],
                    lw=0.9, zorder=1)
            ax.plot([p[0] for p in front], [p[1] for p in front], "-o", color=COL["fixed"],
                    ms=4, lw=1.6, label="uniform lr (the frontier)", zorder=2)
            for r in uni:
                ax.annotate(f"{r['lr']:.0e}", (r[ka], r[kr]), fontsize=6,
                            textcoords="offset points", xytext=(4, -8), color=COL["fixed"])
            for kind in ("delta", "raw_err"):
                fam = sorted([r for r in cr if r["kind"] == kind
                              and r["spend_mode"] != "unit"], key=lambda r: r[ka])
                if len(fam) > 1:
                    ax.plot([r[ka] for r in fam], [r[kr] for r in fam], "-",
                            color=COL[kind], lw=1.3, alpha=0.8, zorder=2)
            for r in cr:
                if r["kind"] == "fixed" and r["w_mult"] == 1.0:
                    continue
                c = COL.get(r["kind"], "#0c8599")
                mk = "D" if r["spend_mode"] == "unit" else "*"
                ax.plot([r[ka]], [r[kr]], mk, color=c, ms=13 if mk == "*" else 7,
                        mec="k", mew=0.5,
                        label=short(r["name"]) + (":unit" if mk == "D" else ""), zorder=3)
            ax.set_xlabel(f"adaptation: {ka}  (lower = better)")
            ax.set_ylabel(f"retention: {kr}  (lower = better)")
            ax.set_title(f"{ttl}\ncell {ck[0]}, n_replay={ck[1]}", fontsize=8.5)
            ax.grid(alpha=0.25, lw=0.5)
            if ai == 0:
                ax.legend(fontsize=6.0, loc="best")
    fig.suptitle(f"priced_plasticity {tag}: does delta sit outside the uniform-lr frontier?",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(FIGS, f"fig1_frontier_{tag}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[figures] wrote {p}")

    # fig2: allocation + spend traces for the modulated arms
    mod = [r for r in rows if r["kind"] != "fixed"]
    if mod:
        cls = list(rows[0]["share"].keys())
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
        wid = 0.8 / max(len(mod), 1)
        for i, r in enumerate(mod):
            vals = [r["share"][c] / max(r["sample_share"][c], 1e-9) for c in cls]
            axs[0].bar(np.arange(len(cls)) + i * wid, vals, wid,
                       label=r["name"], color=COL.get(r["kind"], "#0c8599"),
                       alpha=0.55 + 0.45 * (r["spend_mode"] == "free"))
        axs[0].axhline(1.0, color="k", lw=0.8, ls="--")
        axs[0].set_xticks(np.arange(len(cls)) + 0.4 - wid / 2); axs[0].set_xticklabels(cls, fontsize=7)
        axs[0].set_ylabel("realized weight share / sample share")
        axs[0].set_title("allocation (1.0 = uniform)"); axs[0].legend(fontsize=6.5)
        for r in mod:
            arm = R["arms"][r["name"]]
            axs[1].plot(arm["blog"]["t"], np.convolve(
                np.asarray(arm["blog"]["spend"], float),
                np.ones(21) / 21, mode="same"), lw=1.2, label=r["name"])
        axs[1].axhline(1.0, color="k", lw=0.8, ls="--")
        axs[1].set_xlabel("transitions"); axs[1].set_ylabel("mean batch spend  (= lr multiplier)")
        axs[1].set_title("total plasticity spend"); axs[1].legend(fontsize=6.5)
        fig.tight_layout()
        p = os.path.join(FIGS, f"fig2_allocation_{tag}.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"[figures] wrote {p}")


def merge(Rs):
    """Union the arms of several runs that share a seed and a geometry (a top-up run adds
    learning-rate grid points to an existing seed).  The runner is deterministic given
    (seed, config), and arms do not interact, so a top-up is exactly equivalent to having
    swept the extra points in the original job."""
    base = json.loads(json.dumps(Rs[0]))
    for R in Rs[1:]:
        assert R["config"]["seed"] == base["config"]["seed"], "merge across seeds"
        assert R["config"]["regions"] == base["config"]["regions"], "merge across geometries"
        for k, v in R["stacks"].items():
            base["stacks"].setdefault(k, v)
        for k, v in R["arms"].items():
            if k in base["arms"]:
                print(f"[merge] duplicate arm {k}: keeping the first")
                continue
            base["arms"][k] = v
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="pp_s0")
    ap.add_argument("--merge", action="store_true",
                    help="treat --tags as one run (same seed, top-up grid points)")
    ap.add_argument("--agg", default="half", choices=list(AGG_MODES),
                    help="the aggregation that defines the frontier (see README)")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--ctrl-window", type=int, default=2048)
    ap.add_argument("--json", default="")
    ap.add_argument("--seeds", default="",
                    help="comma-separated groups (';'-separated tags per group) to aggregate "
                         "across seeds, e.g. 'pp_s0,pp_int_s0;pp_s1;pp_s2'")
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
        if R["config"]["mode"] == "calibrate":
            print_calibration(R, rows)
        else:
            print_calibration(R, rows)
            fr = {"fm": print_frontier(R, rows, "adapt_fm", "retain_fm", "FM probe")}
            for c in R["config"]["controllers"]:
                fr[c] = print_frontier(R, rows, f"adapt_{c}", f"retain_{c}", f"control/{c}")
            summary[tag] = {"rows": rows, "frontier": fr}
            if a.figures:
                make_figures(R, rows, tag.replace("+", "_"), a.ctrl_window)
    if a.seeds:
        seed_table([g.split(",") for g in a.seeds.split(";") if g], a.ctrl_window, a.agg)
    if a.json and summary:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(f"[json] wrote {a.json}")


def seed_table(groups, ctrl_window, agg):
    """Per-seed frontier statistics, aggregated. Each group is one seed (optionally several
    tags merged). Reported as mean +/- sd over seeds, which is the convention this arc uses
    for anything it leans on."""
    per = []
    for g in groups:
        Rs = [r for r in (load(t) for t in g) if r is not None]
        if not Rs:
            print(f"[seeds] missing {g}"); continue
        R = merge(Rs) if len(Rs) > 1 else Rs[0]
        rows = arm_rows(R, ctrl_window, agg)
        cr = [r for r in rows if r["n_replay"] == 0]
        if not cr:
            cr = rows
        uni = [r for r in cr if r["kind"] == "fixed" and r["w_mult"] == 1.0]
        front, _ = pareto([(r["adapt_fm"], r["retain_fm"], r) for r in uni])
        ua = [p[0] for p in front]; ur = [p[1] for p in front]
        rec = {"seed": R["config"]["seed"], "n_uni": len(uni), "n_front": len(front)}
        for kind in ("delta", "raw_err"):
            ps = sorted([(r["adapt_fm"], r["retain_fm"]) for r in cr
                         if r["kind"] == kind and r["spend_mode"] != "unit"])
            rec[kind] = curve_gap(ua, ur, [p[0] for p in ps], [p[1] for p in ps])[1]
        dp = sorted([(r["adapt_fm"], r["retain_fm"]) for r in cr
                     if r["kind"] == "delta" and r["spend_mode"] != "unit"])
        rp = sorted([(r["adapt_fm"], r["retain_fm"]) for r in cr
                     if r["kind"] == "raw_err" and r["spend_mode"] != "unit"])
        rec["delta_vs_raw"] = curve_gap([p[0] for p in rp], [p[1] for p in rp],
                                        [p[0] for p in dp], [p[1] for p in dp])[0]
        u = next((r for r in cr if r["spend_mode"] == "unit"), None)
        f = next((r for r in cr if r["kind"] == "fixed" and abs(r["lr"] - (u["lr"] if u else 0)) < 1e-12
                  and r["w_mult"] == 1.0), None)
        if u is not None and f is not None:
            rec["unit_B"] = u["retain_fm"]; rec["fixed_B"] = f["retain_fm"]
            rec["unit_A"] = u["adapt_fm"]; rec["fixed_A"] = f["adapt_fm"]
            rec["unit_wB"] = u["share"]["B-mastered"] / u["sample_share"]["B-mastered"]
        rec["delta_spend"] = float(np.mean([r["mean_spend"] for r in cr if r["kind"] == "delta"]))
        per.append(rec)
    if not per:
        return
    print("\n=== across seeds (priced cell, FM probe) ===")
    print(f"{'seed':>5s} {'hull':>7s} {'delta %span':>12s} {'raw %span':>10s} "
          f"{'d-vs-raw':>9s} {'unit B':>8s} {'fixed B':>8s} {'w_B(unit)':>10s} {'spend':>6s}")
    for r in per:
        print(f"{r['seed']:5d} {r['n_front']:3d}/{r['n_uni']:<3d} {r['delta']:12.1f} "
              f"{r['raw_err']:10.1f} {r['delta_vs_raw']:9.5f} {r.get('unit_B', np.nan):8.4f} "
              f"{r.get('fixed_B', np.nan):8.4f} {r.get('unit_wB', np.nan):10.3f} "
              f"{r['delta_spend']:6.3f}")
    def ms(k):
        v = np.array([r[k] for r in per if np.isfinite(r.get(k, np.nan))], float)
        return f"{v.mean():+.4g} +/- {v.std(ddof=1) if len(v) > 1 else 0:.4g}  (n={len(v)})"
    print(f"  delta vs frontier (% of span): {ms('delta')}")
    print(f"  raw_err vs frontier (% of span): {ms('raw_err')}")
    print(f"  delta - raw (retention, abs):   {ms('delta_vs_raw')}")
    print(f"  delta:unit B - fixed B:         "
          f"{ms('unit_B') if len(per) else ''}  vs fixed {ms('fixed_B')}")
    n_worse = sum(1 for r in per if r.get("unit_B", np.nan) > r.get("fixed_B", np.nan))
    print(f"  seeds where delta:unit retains WORSE than fixed at matched spend: {n_worse}/{len(per)}")


if __name__ == "__main__":
    main()
