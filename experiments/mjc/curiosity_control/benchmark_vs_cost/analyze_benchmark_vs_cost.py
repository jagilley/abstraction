"""
benchmark_vs_cost — Experiment 3 of ideas/performance_error_is_the_bridge.md §8:
noisy-TV discrimination between two candidate forms of the intrinsic value signal,
re-analyzed on logged error sequences from runs we already have. No new training.

The two formulations
--------------------
  benchmark model (performance error):   δ_t = b_t − e_t,   b_{t+1} = (1−a_b)·b_t + a_b·e_t
      (the doc's δ = (b−e)·σ(g/θ); the agency gate g is NOT computable from these logs —
       no arity-1/arity-2 prediction pair was recorded — so we report the ungated b−e.
       σ(g/θ) is a positive gain, so it cannot change any sign below.)
  cost model (two_timescale open-Q #2):  c_t = −α·e_t + β·(−d‖e‖/dt)
      operationalized with the repo-canonical two-EMA derivative proxy
      LPsigned_t = EMA_slow(e)_t − EMA_fast(e)_t  (fast=0.1, slow=0.02 — the deployed
      constants in curiosity_reaching/curiosity_drift), so c_t = −e_t + κ·LPsigned_t
      with κ calibrated *in the cost model's favor*: κ* makes the Phase-1 struct
      frontier read exactly 0 on average; we evaluate at κ = 2κ*.
  pure-LP drive (the deployed allocation signal): LPrelu_t = relu(LPsigned_t).

Data (all logged; pulled from Modal volumes, mirrored under ./data/)
-------------------------------------------------------------------
  A. a2a_forward curiosity_reaching (Phase 1, full run, 1000 iters, single seed):
     per-region err_series (struct/noise/blank) = the fast-EMA (rate 0.1) of the
     per-iteration mean executed glimpse error. The EMA recursion is exact and
     inverted here to recover the RAW per-iteration regional mean error:
         e_raw[t] = (f[t] − 0.9·f[t−1]) / 0.1
     (random arm = visitation-unconfounded primary; surprise arm's noise region
      = densely-sampled secondary check.)
  B. a2a_forward curiosity_scarcity swap150 full run: needleerr_series = unbiased
     (eval-based, no EMA) current-needle error per iteration; swaps every 150 iters
     → 6 abrupt re-opening events.
  C. a2a_forward curiosity_drift quick runs (swap + morph, dp=120, 360 iters):
     structerr_series, unbiased, raw. Secondary re-opening + steady-tracking check.
  D. mjc curiosity_control local mirrors: records[].frontier_err — 16 eval points
     (every 4 rounds) of unbiased frontier-cell error. Coarse; frontier/stationary
     cells only. NOTE: no per-round noise-cell error was logged, so the mjc noisy-TV
     cell is structurally unavailable from logs (only occupancy fractions exist,
     which speak to what the *original* drives sampled, not what δ would read).

Outputs: figures/*.png + summary.json + printed tables.
Run:  python3 analyze_benchmark_vs_cost.py   (pure local re-analysis, no Modal)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# palette (validated; dataviz reference instance)
C_BENCH = "#2a78d6"   # slot 1 blue  — benchmark model b−e
C_COST = "#eb6834"    # slot 2 orange — cost model
C_LP = "#1baf7a"      # slot 3 aqua  — pure-LP (deployed drive)
C_ERR = "#0b0b0b"     # primary ink  — the raw error e
C_MUT = "#898781"     # muted
C_GRID = "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False,
})

EMA_FAST, EMA_SLOW = 0.10, 0.02   # repo-canonical derivative constants
AB_CANON = 0.02                    # canonical benchmark EWMA rate (≈35-iter half-life)
AB_SWEEP = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]


# ------------------------------------------------------------------ signals
def bench_delta(e, ab):
    """δ_t = b_t − e_t with b the EWMA of past e (b_0 = e_0 → δ_0 = 0)."""
    b = np.empty_like(e)
    b[0] = e[0]
    for t in range(1, len(e)):
        b[t] = (1 - ab) * b[t - 1] + ab * e[t - 1]
    return b - e, b


def lp_signed(e, af=EMA_FAST, asl=EMA_SLOW):
    """slow−fast EMA difference = the repo's −d‖e‖/dt proxy (signed)."""
    f = np.empty_like(e); s = np.empty_like(e)
    f[0] = s[0] = e[0]
    for t in range(1, len(e)):
        f[t] = (1 - af) * f[t - 1] + af * e[t]
        s[t] = (1 - asl) * s[t - 1] + asl * e[t]
    return s - f


def stats(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p_pos": float((x > 0).mean()), "e_relu": float(np.maximum(x, 0).mean()),
            "p95": float(np.percentile(x, 95))}


def max_pos_run(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v > 0 else 0
        best = max(best, cur)
    return int(best)


# ------------------------------------------------------------------ load
def load(p):
    with open(os.path.join(DATA, p, "results.json")) as f:
        return json.load(f)


reach = load("curiosity_reaching_mnist_g3_K6_4H128D")
scar_swap = load("curiosity_scarcity_mnist_swap150_a1.0_bl1_K4_g3_4H128D")
scar_morph = load("curiosity_scarcity_mnist_morph150_a1.0_bl1_K4_g3_4H128D")
drift_swap = load("curiosity_drift_mnist_g3_K6_4H128D_quick")
drift_morph = load("curiosity_drift_mnist_morph_g3_K6_4H128D_quick")


def raw_from_logged_ema(f, alpha=0.10):
    f = np.asarray(f, float)
    raw = np.empty_like(f)
    raw[0] = f[0]
    raw[1:] = (f[1:] - (1 - alpha) * f[:-1]) / alpha
    return raw


# Phase-1 raw per-region errors, random arm (primary) + surprise-arm noise (dense)
e_reach = {}   # region -> raw e (random arm)
err_log = np.array(reach["results"]["random"]["err_series"])       # (1000, 3) EMA(0.1)
for i, name in enumerate(["struct", "noise", "blank"]):
    e_reach[name] = raw_from_logged_ema(err_log[:, i])
e_noise_dense = raw_from_logged_ema(
    np.array(reach["results"]["surprise"]["err_series"])[:, 1])

# analysis windows (from the trajectory itself; see report)
W_FRONTIER = slice(1, 163)     # struct still descending (reaches final+10%·range at 163)
W_MASTERED = slice(400, 1000)  # struct flat at ~0.005
W_NOISYTV = slice(300, 1000)   # noise flat at floor (raw slope ~3e-7/iter)
W_DARK = slice(200, 1000)      # blank flat at ~0.004

# ------------------------------------------------------------------ cost-model κ calibration
lps_struct = lp_signed(e_reach["struct"])
kappa_star = float(e_reach["struct"][W_FRONTIER].mean()
                   / lps_struct[W_FRONTIER].mean())
KAPPA = 2.0 * kappa_star


def cost_signal(e):
    return -e + KAPPA * lp_signed(e)


# ------------------------------------------------------------------ Phase-1 regime table
def regime_cells():
    cells = {}
    for regime, (region, win) in {
        "frontier": ("struct", W_FRONTIER), "noisy_tv": ("noise", W_NOISYTV),
        "dark_room": ("blank", W_DARK), "mastered": ("struct", W_MASTERED),
    }.items():
        e = e_reach[region]
        d, _ = bench_delta(e, AB_CANON)
        lps = lp_signed(e)
        cells[regime] = {
            "e": stats(e[win]),
            "bench": {**stats(d[win]), "max_pos_run": max_pos_run(d[win])},
            "cost": stats(cost_signal(e)[win]),
            "lp_relu": stats(np.maximum(lps, 0)[win]),
            "lp_signed": stats(lps[win]),
        }
    return cells


CELLS = regime_cells()

# first-contact readings (t=0..2 mean)
first_contact = {
    "struct": {"e0": float(e_reach["struct"][0]),
               "bench0": 0.0, "cost0": float(-e_reach["struct"][0])},
    "noise": {"e0": float(e_reach["noise"][0]),
              "bench0": 0.0, "cost0": float(-e_reach["noise"][0])},
}

# dense-sampling noisy-TV check (surprise arm, occ≈0.89 of 768 glimpses/iter)
d_dense, _ = bench_delta(e_noise_dense, AB_CANON)
NOISE_DENSE = {**stats(d_dense[W_NOISYTV]), "max_pos_run": max_pos_run(d_dense[W_NOISYTV])}

# ------------------------------------------------------------------ EWMA-timescale sensitivity
SENS = []
for ab in AB_SWEEP:
    d_s, _ = bench_delta(e_reach["struct"], ab)
    d_n, _ = bench_delta(e_reach["noise"], ab)
    row = {"ab": ab, "half_life": float(np.log(2) / ab),
           "frontier_mean": float(d_s[W_FRONTIER].mean()),
           "noise_std": float(d_n[W_NOISYTV].std()),
           "noise_e_relu": float(np.maximum(d_n[W_NOISYTV], 0).mean()),
           "noise_p95": float(np.percentile(d_n[W_NOISYTV], 95)),
           "noise_max_run": max_pos_run(d_n[W_NOISYTV])}
    row["discrim"] = row["frontier_mean"] / row["noise_std"]
    row["fake_frac"] = row["noise_e_relu"] / max(row["frontier_mean"], 1e-12)
    SENS.append(row)

# ------------------------------------------------------------------ re-opening (abrupt drift)
def event_aligned(e, swaps, pre=20, post=145):
    """stack windows around each swap; also per-event recovery metrics per signal."""
    E = []
    for s in swaps:
        if s - pre >= 0 and s + post <= len(e):
            E.append(e[s - pre:s + post])
    return np.array(E)  # (n_events, pre+post)


def reopening_metrics(e, swaps, ab):
    """per-event: bench dip depth/duration, positive hump; LPrelu blind window."""
    out = []
    for s in swaps:
        seg = slice(s - 20, min(s + 145, len(e)))
        d, _ = bench_delta(e, ab)          # computed on the full series (state carries over)
        lpr = np.maximum(lp_signed(e), 0)
        dw = d[s:s + 100]
        below = dw < 0
        dip_dur = int(np.argmax(~below)) if below[0] else 0
        # first index after the dip where δ>0
        pos_after = np.where(dw > 0)[0]
        out.append({
            "dip_depth": float(dw.min()),
            "dip_dur": dip_dur,
            "first_pos": int(pos_after[0]) if len(pos_after) else None,
            "hump_peak": float(dw.max()),
            "lp_blind": int(np.argmax(lpr[s:s + 100] > 0.05 * lpr[s:s + 100].max()))
            if lpr[s:s + 100].max() > 0 else None,
        })
    return out


SWAPS_SCAR = [150 * k for k in range(1, 7)]
e_scar = {arm: np.array(scar_swap["results"]["4"][arm]["needleerr_series"])
          for arm in ["disagree", "random"]}
REOPEN = {arm: reopening_metrics(e_scar[arm], SWAPS_SCAR, AB_CANON)
          for arm in ["disagree", "random"]}
REOPEN_AB = {}   # dip duration vs ab (disagree arm, mean over events)
for ab in AB_SWEEP:
    m = reopening_metrics(e_scar["disagree"], SWAPS_SCAR, ab)
    REOPEN_AB[ab] = {"dip_dur_mean": float(np.mean([x["dip_dur"] for x in m])),
                     "dip_depth_mean": float(np.mean([x["dip_depth"] for x in m]))}

# drift quick (secondary, 2 events at 120/240)
e_dq = np.array(drift_swap["results"]["120"]["random"]["structerr_series"])
REOPEN_DQ = reopening_metrics(e_dq, [120, 240], AB_CANON)

# steady gradual tracking (morph): does either signal stay engaged?
e_morph = np.array(scar_morph["results"]["4"]["disagree"]["needleerr_series"])
W_STEADY = slice(300, 1000)
d_m, _ = bench_delta(e_morph, AB_CANON)
STEADY = {"e": stats(e_morph[W_STEADY]), "bench": stats(d_m[W_STEADY]),
          "cost": stats(cost_signal(e_morph)[W_STEADY]),
          "lp_relu": stats(np.maximum(lp_signed(e_morph), 0)[W_STEADY])}

# ------------------------------------------------------------------ mjc frontier_err (coarse)
MJC_DIR = os.path.join(HERE, "..", "figures")
MJC = {}
for tag in ["ft_drift_s0", "ft_drift_s1", "ft_drift_s2",
            "ft_stat_s0", "ft_stat_s1", "ft_stat_s2"]:
    p = os.path.join(MJC_DIR, f"curiosity_control_{tag}", "results.json")
    with open(p) as f:
        d = json.load(f)
    MJC[tag] = {}
    for arm in ["reducible", "random"]:
        fe = np.array([r["frontier_err"] for r in d["results"][arm]["records"]])
        db, _ = bench_delta(fe, 0.3)       # 16 pts, Δ=4 rounds → faster EWMA
        MJC[tag][arm] = {"frontier_err": fe.tolist(),
                         "bench_mean": float(db[1:].mean()),
                         "bench_p_pos": float((db[1:] > 0).mean())}
        # coarse signed derivative for the cost term: finite difference per record
        dd = -(np.diff(fe))                # −Δe per 4 rounds (positive = improving)
        MJC[tag][arm]["neg_e_mean"] = float(-fe[1:].mean())
        MJC[tag][arm]["lp_fd_mean"] = float(dd.mean())

# ------------------------------------------------------------------ print tables
def fmt(v, w=9, p=4):
    return f"{v:+{w}.{p}f}" if isinstance(v, float) else f"{v!s:>{w}}"


print("=" * 100)
print("PHASE-1 REGIME TABLE  (a2a active vision, random arm, raw reconstructed errors;"
      f" a_b={AB_CANON}, κ=2κ*={KAPPA:.1f}, κ*={kappa_star:.1f})")
print("=" * 100)
hdr = f"{'regime':<10} {'window':<12} {'e mean':>9} | {'bench mean':>10} {'P(δ>0)':>7} {'E[relu δ]':>10} {'maxrun+':>8} | {'cost mean':>10} {'P(c>0)':>7} | {'LPrelu mean':>11}"
print(hdr); print("-" * len(hdr))
wins = {"frontier": "1–163", "noisy_tv": "300–1000", "dark_room": "200–1000", "mastered": "400–1000"}
for rg, c in CELLS.items():
    print(f"{rg:<10} {wins[rg]:<12} {c['e']['mean']:>9.4f} | "
          f"{c['bench']['mean']:>+10.4f} {c['bench']['p_pos']:>7.2f} {c['bench']['e_relu']:>10.4f} "
          f"{c['bench'].get('max_pos_run', ''):>8} | "
          f"{c['cost']['mean']:>+10.4f} {c['cost']['p_pos']:>7.2f} | {c['lp_relu']['mean']:>11.5f}")

print(f"\nFirst contact (t=0): struct e0={first_contact['struct']['e0']:.3f} → cost {first_contact['struct']['cost0']:+.3f}, bench 0.000 ;  "
      f"noise e0={first_contact['noise']['e0']:.3f} → cost {first_contact['noise']['cost0']:+.3f}, bench 0.000")
print(f"Noisy-TV densely-sampled check (surprise arm): bench mean {NOISE_DENSE['mean']:+.5f}, "
      f"std {NOISE_DENSE['std']:.5f}, E[relu] {NOISE_DENSE['e_relu']:.5f}, max +run {NOISE_DENSE['max_pos_run']}")

print("\n" + "=" * 100)
print("EWMA-TIMESCALE SENSITIVITY (benchmark model; noisy-TV vs frontier, Phase-1)")
print("=" * 100)
print(f"{'a_b':>7} {'t½ (it)':>8} {'frontier mean δ':>16} {'noise std δ':>12} {'noise E[relu δ]':>16} {'noise p95':>10} {'max +run':>9} {'discrim':>8} {'fake frac':>10}")
for r in SENS:
    print(f"{r['ab']:>7} {r['half_life']:>8.1f} {r['frontier_mean']:>+16.4f} {r['noise_std']:>12.4f} "
          f"{r['noise_e_relu']:>16.4f} {r['noise_p95']:>+10.4f} {r['noise_max_run']:>9} "
          f"{r['discrim']:>8.2f} {r['fake_frac']:>10.3f}")

print("\n" + "=" * 100)
print("RE-OPENING AFTER ABRUPT DRIFT (scarcity swap150, 6 events; needle-err, per-event metrics)")
print("=" * 100)
for arm in ["disagree", "random"]:
    m = REOPEN[arm]
    print(f"  arm={arm:9s}  bench(a_b={AB_CANON}): dip depth {np.mean([x['dip_depth'] for x in m]):+.4f} "
          f"(events: {[round(x['dip_depth'],3) for x in m]})")
    print(f"                dip duration (iters δ<0): mean {np.mean([x['dip_dur'] for x in m]):.1f} "
          f"{[x['dip_dur'] for x in m]}")
    print(f"                hump peak {np.mean([x['hump_peak'] for x in m]):+.4f}; "
          f"LPrelu blind window (iters to 5% of post-swap max): {[x['lp_blind'] for x in m]}")
print(f"  drift-quick dp120 (random): {REOPEN_DQ}")
print(f"  bench dip-duration vs a_b (disagree): "
      f"{ {ab: round(v['dip_dur_mean'],1) for ab, v in REOPEN_AB.items()} }")

print("\nSTEADY GRADUAL TRACKING (morph150, iters 300–1000): "
      f"e={STEADY['e']['mean']:.3f}±{STEADY['e']['std']:.3f}  bench δ={STEADY['bench']['mean']:+.4f} "
      f"(P>0 {STEADY['bench']['p_pos']:.2f})  cost={STEADY['cost']['mean']:+.4f}  LPrelu={STEADY['lp_relu']['mean']:.5f}")

print("\n" + "=" * 100)
print("MJC curiosity_control frontier_err (coarse: 16 eval pts, Δ=4 rounds; bench a_b=0.3)")
print("=" * 100)
for tag, arms in MJC.items():
    for arm, v in arms.items():
        print(f"  {tag:12s} {arm:9s} bench mean {v['bench_mean']:+.4f} (P>0 {v['bench_p_pos']:.2f}) "
              f"  −e mean {v['neg_e_mean']:+.4f}  −Δe/4rnd mean {v['lp_fd_mean']:+.4f}")

# ------------------------------------------------------------------ figures
it = np.arange(1000)

# ---- fig 1: three regimes
fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.4), sharex=True)
titles = {"struct": "struct — learnable frontier → mastered",
          "noise": "noise — noisy TV (irreducible)",
          "blank": "blank — dark room"}
for j, region in enumerate(["struct", "noise", "blank"]):
    e = e_reach[region]
    d, b = bench_delta(e, AB_CANON)
    ax = axes[0, j]
    ax.plot(it, e, color=C_ERR, lw=0.7, alpha=0.55, label="raw e (reconstructed)")
    ax.plot(it, b, color=C_MUT, lw=1.8, label=f"benchmark b (a_b={AB_CANON})")
    ax.set_title(titles[region])
    if j == 0:
        ax.set_ylabel("error ‖e‖")
        ax.legend(loc="upper right", fontsize=8)
    ax2 = axes[1, j]
    ax2.axhline(0, color="#c3c2b7", lw=0.8)
    ax2.plot(it, d, color=C_BENCH, lw=1.2, label="benchmark  b−e")
    ax2.plot(it, cost_signal(e), color=C_COST, lw=1.2, label=f"cost  −e+κ·LP (κ=2κ*)")
    ax2.plot(it, np.maximum(lp_signed(e), 0), color=C_LP, lw=1.2, label="pure LP (relu)")
    ax2.set_xlabel("iteration")
    if j == 0:
        ax2.set_ylabel("signal")
        ax2.legend(loc="lower right", fontsize=8)
fig.suptitle("Phase-1 active vision (random arm): the three regimes under each formulation",
             fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig1_three_regimes.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- fig 2: noisy-TV zoom
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
e_n = e_reach["noise"]
d_n, _ = bench_delta(e_n, AB_CANON)
ax = axes[0]
ax.axhline(0, color="#c3c2b7", lw=0.8)
ax.plot(np.arange(300, 1000), d_n[300:1000], color=C_BENCH, lw=0.9)
ax.set_title("noisy TV: benchmark δ = b−e (iters 300–1000)")
ax.set_xlabel("iteration"); ax.set_ylabel("δ")
ax = axes[1]
d_s, _ = bench_delta(e_reach["struct"], AB_CANON)
bins = np.linspace(-0.02, 0.05, 60)
ax.hist(d_n[W_NOISYTV], bins=bins, color=C_BENCH, alpha=0.65, label="noise (TV)", density=True)
ax.hist(d_s[W_FRONTIER], bins=bins, color=C_COST, alpha=0.65, label="struct (frontier)", density=True)
ax.axvline(0, color="#c3c2b7", lw=0.8)
ax.set_title("δ distribution: frontier vs noisy TV"); ax.set_xlabel("δ"); ax.legend(fontsize=8)
ax = axes[2]
labels = ["bench δ", "pure LP", "cost (relu)"]
frontier_vals = [np.maximum(d_s[W_FRONTIER], 0).mean(),
                 np.maximum(lp_signed(e_reach["struct"])[W_FRONTIER], 0).mean(),
                 np.maximum(cost_signal(e_reach["struct"])[W_FRONTIER], 0).mean()]
noise_vals = [np.maximum(d_n[W_NOISYTV], 0).mean(),
              np.maximum(lp_signed(e_n)[W_NOISYTV], 0).mean(),
              np.maximum(cost_signal(e_n)[W_NOISYTV], 0).mean()]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, frontier_vals, w, color=C_COST, label="frontier")
ax.bar(x + w / 2, noise_vals, w, color=C_BENCH, label="noisy TV")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("E[positive part] — what a softmax drive sees")
ax.legend(fontsize=8)
for xi, (fv, nv) in enumerate(zip(frontier_vals, noise_vals)):
    ax.text(xi + w / 2, nv, f"{nv:.4f}", ha="center", va="bottom", fontsize=7.5, color="#52514e")
    ax.text(xi - w / 2, fv, f"{fv:.4f}", ha="center", va="bottom", fontsize=7.5, color="#52514e")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig2_noisytv_zoom.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- fig 3: re-opening, event-aligned (scarcity swap150, disagree arm)
pre, post = 20, 145
tt = np.arange(-pre, post)
e_full = e_scar["disagree"]
E = event_aligned(e_full, SWAPS_SCAR, pre, post)
d_full, _ = bench_delta(e_full, AB_CANON)
d_fast, _ = bench_delta(e_full, 0.1)
d_slow, _ = bench_delta(e_full, 0.005)
lps_full = lp_signed(e_full)
D = event_aligned(d_full, SWAPS_SCAR, pre, post)
Df = event_aligned(d_fast, SWAPS_SCAR, pre, post)
Ds = event_aligned(d_slow, SWAPS_SCAR, pre, post)
L = event_aligned(lps_full, SWAPS_SCAR, pre, post)
Cs = event_aligned(cost_signal(e_full), SWAPS_SCAR, pre, post)
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharex=True)
ax = axes[0]
ax.plot(tt, E.mean(0), color=C_ERR, lw=1.5)
ax.fill_between(tt, E.min(0), E.max(0), color=C_ERR, alpha=0.12, lw=0)
ax.axvline(0, color=C_MUT, lw=0.8, ls="--")
ax.set_title("needle error e (mean ± range, 6 swaps)")
ax.set_xlabel("iters since swap"); ax.set_ylabel("‖e‖")
ax = axes[1]
ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color=C_MUT, lw=0.8, ls="--")
ax.plot(tt, Df.mean(0), color="#86b6ef", lw=1.2, label="b−e, a_b=0.1 (t½ 7)")
ax.plot(tt, D.mean(0), color=C_BENCH, lw=1.8, label="b−e, a_b=0.02 (t½ 35)")
ax.plot(tt, Ds.mean(0), color="#104281", lw=1.2, label="b−e, a_b=0.005 (t½ 139)")
ax.set_title("benchmark model at re-opening")
ax.set_xlabel("iters since swap"); ax.legend(fontsize=8)
ax = axes[2]
ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color=C_MUT, lw=0.8, ls="--")
ax.plot(tt, np.maximum(L, 0).mean(0), color=C_LP, lw=1.8, label="pure LP (relu)")
ax.plot(tt, Cs.mean(0), color=C_COST, lw=1.8, label="cost −e+κ·LP")
ax.set_title("derivative-family signals at re-opening")
ax.set_xlabel("iters since swap"); ax.legend(fontsize=8)
fig.suptitle("Abrupt re-opening (scarcity swap150, disagree arm): frustration-transient vs blindness",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig3_reopening.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- fig 4: EWMA sensitivity
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
hl = [r["half_life"] for r in SENS]
ax = axes[0]
ax.plot(hl, [r["frontier_mean"] for r in SENS], "o-", color=C_BENCH, ms=4)
ax.set_xscale("log"); ax.set_title("frontier: mean δ vs benchmark half-life")
ax.set_xlabel("b half-life (iters)"); ax.set_ylabel("mean δ (frontier)")
ax = axes[1]
ax.plot(hl, [r["noise_e_relu"] for r in SENS], "o-", color=C_BENCH, ms=4, label="noise E[relu δ]")
ax.plot(hl, [r["noise_std"] for r in SENS], "o-", color=C_MUT, ms=4, label="noise std δ")
ax.set_xscale("log"); ax.set_title("noisy TV: spurious positive signal")
ax.set_xlabel("b half-life (iters)"); ax.legend(fontsize=8)
ax = axes[2]
ax.plot(hl, [r["fake_frac"] for r in SENS], "o-", color=C_COST, ms=4, label="fake frac (noise/frontier)")
ax.plot(hl, [REOPEN_AB[r["ab"]]["dip_dur_mean"] / 100 for r in SENS], "o-",
        color=C_BENCH, ms=4, label="re-open dip dur /100 it")
ax.set_xscale("log"); ax.set_title("the timescale tradeoff")
ax.set_xlabel("b half-life (iters)"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig4_ewma_sensitivity.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ---- fig 5: mjc coarse frontier signals
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
for k, (kind, tags) in enumerate([("drift", ["ft_drift_s0", "ft_drift_s1", "ft_drift_s2"]),
                                  ("stationary", ["ft_stat_s0", "ft_stat_s1", "ft_stat_s2"])]):
    ax = axes[k]
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    for tag in tags:
        fe = np.array(MJC[tag]["reducible"]["frontier_err"])
        db, _ = bench_delta(fe, 0.3)
        rounds = np.arange(len(fe)) * 4
        ax.plot(rounds, db, "o-", color=C_BENCH, ms=3, lw=1,
                alpha=0.8, label="b−e (reducible)" if tag.endswith("s0") else None)
        ax.plot(rounds, -fe, "s--", color=C_COST, ms=3, lw=1,
                alpha=0.6, label="−e (cost level term)" if tag.endswith("s0") else None)
    ax.set_title(f"mjc {kind} (3 seeds, reducible arm)")
    ax.set_xlabel("round")
    if k == 0:
        ax.set_ylabel("signal"); ax.legend(fontsize=8)
fig.suptitle("mjc curiosity_control frontier cell (coarse: 16 eval pts)", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig5_mjc_frontier.png"), dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ save summary
summary = {
    "kappa_star": kappa_star, "kappa_used": KAPPA, "ab_canonical": AB_CANON,
    "phase1_cells": CELLS, "first_contact": first_contact,
    "noise_dense_check": NOISE_DENSE, "ewma_sensitivity": SENS,
    "reopening_swap150": REOPEN, "reopening_ab_sweep": REOPEN_AB,
    "reopening_driftquick_dp120": REOPEN_DQ, "steady_morph": STEADY,
    "mjc_coarse": MJC,
}
with open(os.path.join(HERE, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(f"\nSaved figures to {FIGS} and summary.json")
