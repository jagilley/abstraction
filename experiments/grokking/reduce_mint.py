"""[grokking] Reduce a `mint.py::mint` run to the table of record and figures.

    python3 grokking/reduce_mint.py --tag m1 --fetch      # from experiments/

--fetch pulls /data/<tag>/mint.json off the `grokking-mint-data` volume into
results/<tag>/mint.json (git-ignored; the volume copy is the artifact). Writes:
    figures/mint_<tag>_table.txt      per-snapshot table (each rule), final-net section, falsifiers
    figures/mint_<tag>_curve.png      held-out accuracy of each compiled form vs the net, and kept k
    figures/mint_<tag>_read.png       producer score over training (heatmap) + final-net spectra
    figures/mint_<tag>_svd.png        SVD rank sweep vs the compiled forms (accuracy vs parameters)
    results/mint_<tag>_summary.json   compact per-snapshot numbers (the plotted series)
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = ("A", "ALS", "B", "C")
RULES = ("le0", "lt1", "m3")
P, H = 97, 128
ORIG_PARAMS = 53985

# reference categorical palette (dataviz skill, light mode), fixed order
COL = {"net": "#52514e", "A": "#2a78d6", "ALS": "#eb6834", "B": "#1baf7a", "C": "#e87ba4"}
NAME = {"A": "A closed form", "ALS": "A-LS amplitudes", "B": "B unit-pruned net",
        "C": "C frequency-filtered net", "net": "net (test)"}


def fetch(tag):
    d = os.path.join(HERE, "results", tag)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, MODAL_PROFILE="chromatic")
    subprocess.run(["modal", "volume", "get", "--force", "grokking-mint-data", f"{tag}/mint.json",
                    os.path.join(d, "mint.json")], check=True, env=env)


def params_B(units):
    return units * (2 * P + 1) + H * units + H + H * P + P


def params_C(k):
    return H * 2 * (1 + 2 * k) + H + H * H + H + H * P + P


def fmt_acc(x):
    return f"{x:.3f}"


def fmtK(K, n=14):
    s = ",".join(str(w) for w in K[:n])
    return s + (",..." if len(K) > n else "")


def load(tag):
    with open(os.path.join(HERE, "results", tag, "mint.json")) as f:
        return json.load(f)


SHAPES = ((128, 194), (128, 128), (97, 128))


def svd_params(r):
    """Rank-r truncation of all three layers, each stored factored or dense, whichever is smaller."""
    return sum(min(min(r, o, i) * (o + i), o * i) + o for o, i in SHAPES)


def svd_rows(d):
    return [dict(x, params=svd_params(x["rank"])) for x in d["extras"]["svd"]]


def by_run(d, run):
    R = [r for r in d["records"].values() if r["run"] == run]
    return sorted(R, key=lambda r: (r["epoch"], r["label"]))


def table(d, tag):
    L = []
    w = L.append
    true = by_run(d, "true")
    shuf = by_run(d, "shuffled")
    init = by_run(d, "init")
    rules = d["rules"]
    w(f"# grokking / mint -- tag {tag}   (p={d['p']}, n_train={d['n_tr']}, seed 42, "
      f"{d['train']['true']['n_epochs']} epochs)")
    w(f"# gate rules (tol in census_walk's `e_x <= best_e + tol`): " +
      ", ".join(f"{k}={v:+.6f}" for k, v in rules.items()))
    w("# every compiled form is gated by its OWN agreement with the net's own argmax on the train pairs;")
    w("# 'held' = that form's label accuracy on the 6587 held-out pairs at its kept K (logged, never consumed).")
    w("# A = closed form sum cos, ALS = cos amplitudes LS-fit to the net's train logits,")
    w("# B = net with units whose dominant w not in K zeroed, C = net with layer-0 rows filtered to DC+K.")
    w("")
    ft = d["train"]["true"]["first_hit"]
    w(f"net (true labels): first eval epoch with test acc >= 0.5/0.9/0.95/0.99/1.0: "
      f"{ft['0.5']}/{ft['0.9']}/{ft['0.95']}/{ft['0.99']}/{ft['1.0']}  (eval every 50 epochs)")
    w(f"training wall-clock: true {d['train']['true']['seconds']:.0f}s, shuffled "
      f"{d['train']['shuffled']['seconds']:.0f}s (8-CPU container)")
    w("")
    for rn in RULES:
        w(f"## Per-snapshot table, rule {rn}  (true-label run)")
        w(f"{'epoch':>6} {'net_tr':>6} {'net_te':>6} {'n90':>4} {'PR':>5} | "
          f"{'kA':>3} {'heldA':>6} | {'kALS':>4} {'hALS':>6} | {'kB':>3} {'units':>5} {'heldB':>6} | "
          f"{'kC':>3} {'heldC':>6} | K_B")
        for r in true:
            W = r["walks"]
            w(f"{r['epoch']:>6} {fmt_acc(r['net_train_acc_fit']):>6} {fmt_acc(r['net_test_acc']):>6} "
              f"{r['score_spread']['n90']:>4} {r['score_spread']['participation']:>5.1f} | "
              f"{W['A'][rn]['k']:>3} {fmt_acc(W['A'][rn]['heldout']):>6} | "
              f"{W['ALS'][rn]['k']:>4} {fmt_acc(W['ALS'][rn]['heldout']):>6} | "
              f"{W['B'][rn]['k']:>3} {W['B'][rn]['units']:>5} {fmt_acc(W['B'][rn]['heldout']):>6} | "
              f"{W['C'][rn]['k']:>3} {fmt_acc(W['C'][rn]['heldout']):>6} | {fmtK(W['B'][rn]['K'])}")
        w("")
    # random-order controls
    w("## Random-order controls (5 permutations) vs producer order, k and held-out at snapshots with controls")
    w(f"{'epoch':>6} {'rule':>4} | " + " | ".join(f"{f}: k_prod k_rand[min..max] held_prod held_rand_mean"
                                                 for f in ("B", "C")))
    for r in true:
        if not r["walks"]["B"]["lt1"]["rand"]:
            continue
        for rn in RULES:
            parts = []
            for f in ("B", "C"):
                W = r["walks"][f][rn]
                ks = [x["k"] for x in W["rand"]]
                hs = [x["heldout"] for x in W["rand"]]
                parts.append(f"{W['k']:>3} [{min(ks):>2}..{max(ks):>2}] {W['heldout']:.3f} {np.mean(hs):.3f}")
            w(f"{r['epoch']:>6} {rn:>4} | " + " | ".join(parts))
    w("")
    # final net
    fin = true[-1]
    w(f"## Final net (epoch {fin['epoch']}): net train {fin['net_train_acc_fit']:.4f}, "
      f"test {fin['net_test_acc']:.4f}")
    sc = np.asarray(fin["score"])
    order = np.argsort(-sc)
    tot = sc.sum()
    sup = fin["support"]
    w("READ (layer-0 weights): top 16 frequencies by producer score  w:score_frac(support units)")
    w("  " + "  ".join(f"{i + 1}:{sc[i] / tot:.3f}({sup[i]})" for i in order[:16]))
    s = fin["score_spread"]
    w(f"  frequencies with >=1 dominant unit: {fin['n_support']}; participation ratio {s['participation']:.1f}; "
      f"n for 90%/99% of score: {s['n90']}/{s['n99']}; top-5 score frac {s['top5_frac']:.3f}")
    w(f"  units whose a-row and b-row dominant w agree: {fin['dom_a_eq_b']}/128; median unit concentration "
      f"(energy at dominant w / non-DC energy): {np.median(fin.get('unit_conc', [np.nan])):.3f}")
    ld = np.asarray(fin["logit_diag"])
    lo = np.argsort(-ld)
    w("LOGIT TABLE (net outputs on all p^2 inputs): top 12 diagonal frequencies  w:energy_frac")
    w("  " + "  ".join(f"{i + 1}:{ld[i]:.4f}" for i in lo[:12]))
    w(f"  diagonal total {fin['logit_diag_total']:.4f}; DC {fin['logit_dc_frac']:.4f}; donor measure "
      f"(top-5 UNFOLDED diag + DC, cnb README's 0.374 on its net) = {fin['logit_old_top5']:.4f}; "
      f"logit-diag n90 {fin['logit_diag_spread']['n90']}, PR {fin['logit_diag_spread']['participation']:.1f}")
    w("")
    w("KEPT K per gate (producer order):")
    for f in FORMS:
        for rn in RULES:
            W = fin["walks"][f][rn]
            ks = [x["k"] for x in W["rand"]]
            kd = sum(ld[x - 1] for x in W["K"]) if W["K"] else 0.0
            w(f"  {f:>3} {rn:>4}: k={W['k']:>2} held={W['heldout']:.4f} e_final={W['e_final']:.4f} "
              f"units={W['units']:>3} rand k=[{','.join(map(str, ks))}] logit-diag energy at K={kd:.3f}  "
              f"K={W['K']}")
    w("")
    for rn in ("lt1", "m3"):
        w(f"CROSS table, rule {rn}: held-out accuracy of each form (columns) at each gate's kept K (rows)")
        w(f"  {'gate':>5} {'k':>3} | " + " ".join(f"{f:>6}" for f in FORMS))
        for g in FORMS:
            c = fin["cross"][rn][g]
            w(f"  {g:>5} {fin['walks'][g][rn]['k']:>3} | " + " ".join(f"{c[f]:>6.3f}" for f in FORMS))
    w("")
    # SVD baseline
    svd = svd_rows(d)
    w("SVD baseline (all three layers truncated to rank r, biases kept; cnb precompute_svd.py recipe)")
    for thr in (0.5, 0.9, 0.95, 0.99, 1.0):
        hit = next((x for x in svd if x["heldout_acc"] >= thr - 1e-12), None)
        if hit:
            w(f"  held-out >= {thr:.2f}: min rank {hit['rank']:>3}, params {hit['params']:>6} "
              f"(held {hit['heldout_acc']:.4f}, train {hit['train_acc']:.4f})")
        else:
            w(f"  held-out >= {thr:.2f}: never")
    w("  ranks 8,12,16,20,24,32,48,64: " + "  ".join(
        f"r{x['rank']}:{x['heldout_acc']:.3f}" for x in svd if x["rank"] in (8, 12, 16, 20, 24, 32, 48, 64)))
    w("")
    w("SIZE TABLE (final net)")
    w(f"  original MLP: {ORIG_PARAMS} params, held {fin['net_test_acc']:.4f}")
    for rn in RULES:
        kA = fin["walks"]["A"][rn]["k"]
        kL = fin["walks"]["ALS"][rn]["k"]
        B = fin["walks"]["B"][rn]
        C = fin["walks"]["C"][rn]
        w(f"  rule {rn}: A closed form k={kA} ints, 0 floats (lookup form 4pk={4 * P * kA}) held "
          f"{fin['walks']['A'][rn]['heldout']:.4f} | ALS k={kL} + {kL} amplitudes held "
          f"{fin['walks']['ALS'][rn]['heldout']:.4f} | B {B['units']} units = {params_B(B['units'])} params held "
          f"{B['heldout']:.4f} | C k={C['k']} -> {params_C(C['k'])} params held {C['heldout']:.4f}")
    hitB = {}
    for rn in RULES:
        tgt = fin["walks"]["B"][rn]["heldout"]
        hit = next((x for x in svd if x["heldout_acc"] >= tgt - 1e-12), None)
        hitB[rn] = hit
        if hit:
            w(f"  SVD matched to B ({rn}) held {tgt:.4f}: rank {hit['rank']}, {hit['params']} params")
    w("")
    # sanity
    allr = list(d["records"].values())
    w("## Instrument checks")
    w(f"  form B at K=all bit-exact vs the net: {sum(r['B_all_bitexact'] for r in allr)}/{len(allr)} snapshots")
    w(f"  form C at K=all: max |dlogit| over snapshots {max(r['C_all_maxabs'] for r in allr):.2e}, "
      f"min argmax agreement {min(r['C_all_argmax_agree'] for r in allr):.4f}")
    w(f"  units assigned a dominant w in 1..48 at K=all: min {min(r['units_all'] for r in allr)}/128")
    w("")
    # falsifiers
    w("## Falsifiers: shuffled-label net (train labels permuted; held = TRUE-label held-out accuracy)")
    w(f"{'label':>8} {'net_tr':>6} {'net_te':>6} | " + " | ".join(
        f"{f}: k le0/lt1/m3 held(lt1) held(m3)" for f in FORMS))
    for r in shuf + init:
        parts = []
        for f in FORMS:
            W = r["walks"][f]
            parts.append(f"{W['le0']['k']:>2}/{W['lt1']['k']:>2}/{W['m3']['k']:>2} "
                         f"{W['lt1']['heldout']:.3f} {W['m3']['heldout']:.3f}")
        lab = r["label"] if r["run"] != "init" else r["label"].replace("init_", "rinit_")
        w(f"{lab:>8} {r['net_train_acc_fit']:>6.3f} {r['net_test_acc']:>6.3f} | " + " | ".join(parts))
    w("  (rows rinit_s* = untrained nets, seeds 0..3; the seed-42 init is the 'init' row of each run)")
    w("")
    # plateau summary: first epoch at which each form's held-out reaches thresholds
    w("## First snapshot epoch at which held-out accuracy >= threshold (true run, producer order)")
    w(f"{'series':>12} " + " ".join(f"{t:>6}" for t in ("0.5", "0.9", "0.99", "1.0")))

    def first(xs, ep, t):
        for x, e in zip(xs, ep):
            if x >= t - 1e-12:
                return e
        return None
    ep = [r["epoch"] for r in true]
    w(f"{'net':>12} " + " ".join(f"{str(first([r['net_test_acc'] for r in true], ep, t)):>6}"
                                 for t in (0.5, 0.9, 0.99, 1.0)))
    for f in FORMS:
        for rn in RULES:
            xs = [r["walks"][f][rn]["heldout"] for r in true]
            w(f"{f + ' ' + rn:>12} " + " ".join(f"{str(first(xs, ep, t)):>6}" for t in (0.5, 0.9, 0.99, 1.0)))
    return "\n".join(L)


def figures(d, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e6e5e0"})
    true = [r for r in by_run(d, "true") if r["epoch"] >= 0]
    ep = np.array([r["epoch"] for r in true])
    summary = {"epoch": ep.tolist(), "net_test": [r["net_test_acc"] for r in true],
               "net_train": [r["net_train_acc_fit"] for r in true]}

    # --- curve: 2 rows (held-out acc, kept k) x 3 cols (rule) ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for j, rn in enumerate(RULES):
        ax, axk = axes[0, j], axes[1, j]
        ax.plot(ep, summary["net_test"], color=COL["net"], lw=2.0, label=NAME["net"], zorder=5)
        for f in FORMS:
            y = [r["walks"][f][rn]["heldout"] for r in true]
            k = [r["walks"][f][rn]["k"] for r in true]
            summary[f"held_{f}_{rn}"] = y
            summary[f"k_{f}_{rn}"] = k
            lw, ls = (3.2, "-") if f == "A" else ((1.4, (0, (3, 2))) if f == "ALS" else (1.6, "-"))
            ax.plot(ep, y, color=COL[f], lw=lw, ls=ls, label=NAME[f])
            axk.plot(ep, k, color=COL[f], lw=lw, ls=ls, label=NAME[f])
        ax.set_title(f"gate rule {rn}  (tol {d['rules'][rn]:+.4f})", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", lw=0.6)
        axk.set_ylim(-1, 49)
        axk.grid(True, axis="y", lw=0.6)
        axk.set_xlabel("epoch")
        if j == 0:
            ax.set_ylabel("held-out accuracy")
            axk.set_ylabel("kept k (of 48)")
    axes[0, 0].legend(loc="center right", frameon=False, fontsize=8)
    fig.suptitle(f"mint {tag}: each compiled form, gated by its own agreement with the net's train argmax "
                 f"(A and A-LS coincide: thick blue under dashed orange)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"mint_{tag}_curve.png"), dpi=140)
    plt.close(fig)

    # --- read: heatmap of normalized producer score over epochs; final spectra ---
    S = np.array([np.asarray(r["score"]) / np.sum(r["score"]) for r in true])
    fin = true[-1]
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(S.T, aspect="auto", origin="lower", cmap="Blues",
                   extent=[ep[0], ep[-1], 0.5, 48.5], interpolation="nearest")
    ax.set_xlabel("epoch")
    ax.set_ylabel("frequency w")
    ax.set_title("producer score (layer-0 energy at w, fraction of snapshot total)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    wv = np.arange(1, 49)
    KB = set(fin["walks"]["B"]["lt1"]["K"])
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(wv, S[-1], color=["#2a78d6" if x in KB else "#b9b8b2" for x in wv], width=0.8)
    ax1.set_title(f"final net: producer score (blue = kept by gate B, lt1; k={len(KB)})", fontsize=10)
    ax1.set_ylabel("score fraction")
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.bar(wv, fin["logit_diag"], color="#1baf7a", width=0.8)
    ax2.set_title("final net: logit-table energy on diagonal (w,w), fraction of total", fontsize=10)
    ax2.set_xlabel("frequency w")
    ax2.set_ylabel("energy fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"mint_{tag}_read.png"), dpi=140)
    plt.close(fig)
    summary["score_frac"] = S.round(5).tolist()

    # --- svd: held-out acc vs params (full range, and a zoom on the top) ---
    svd = svd_rows(d)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    for j, ax in enumerate(axes):
        ax.plot([x["params"] for x in svd], [x["heldout_acc"] for x in svd], color=COL["net"], lw=1.6,
                marker="o", ms=3, label="SVD rank-r truncation (r = 1..128)")
        for rn, mk in zip(RULES, ("s", "o", "D")):
            B = fin["walks"]["B"][rn]
            C = fin["walks"]["C"][rn]
            ax.scatter([params_B(B["units"])], [B["heldout"]], color=COL["B"], marker=mk, s=60, zorder=5,
                       edgecolor="white", linewidth=1.5, label=f"B {rn} ({B['units']} units)")
            ax.scatter([params_C(C["k"])], [C["heldout"]], color=COL["C"], marker=mk, s=60, zorder=5,
                       edgecolor="white", linewidth=1.5, label=f"C {rn} (k={C['k']})")
            kL = fin["walks"]["ALS"][rn]["k"]
            if kL and j == 0:
                ax.scatter([2 * kL], [fin["walks"]["ALS"][rn]["heldout"]], color=COL["ALS"], marker="^",
                           s=60, zorder=5, edgecolor="white", linewidth=1.5,
                           label=f"ALS {rn} (k={kL}: {kL} freqs + {kL} amps)")
        ax.axvline(ORIG_PARAMS, color="#b9b8b2", lw=1, ls="--")
        ax.grid(True, axis="y", lw=0.6)
        ax.set_xlabel("parameters" + (" (log)" if j == 0 else ""))
    axes[0].set_xscale("log")
    axes[0].set_ylabel("held-out accuracy")
    axes[0].set_title("final net: SVD baseline vs compiled forms (dashed = original); "
                      "A has 0 float params, not drawn", fontsize=9)
    axes[1].set_xlim(15000, 60000)
    axes[1].set_ylim(0.9, 1.004)
    axes[1].set_title("zoom: held-out >= 0.9", fontsize=9)
    hl = axes[0].get_legend_handles_labels()
    axes[1].legend(*hl, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"mint_{tag}_svd.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(HERE, "results", f"mint_{tag}_summary.json"), "w") as f:
        json.dump(summary, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    d = load(a.tag)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    txt = table(d, a.tag)
    with open(os.path.join(HERE, "figures", f"mint_{a.tag}_table.txt"), "w") as f:
        f.write(txt + "\n")
    figures(d, a.tag)
    print(txt)


if __name__ == "__main__":
    main()
