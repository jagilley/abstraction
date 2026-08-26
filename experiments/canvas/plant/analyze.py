"""Fetch the plant node's results off the Modal volume and reduce them to figures.

    cd experiments      # MODAL_PROFILE=chromatic, conda glp
    python3 canvas/plant/analyze.py --tag pl0 --qtag q0 --fetch --figures
"""

import argparse
import json
import os
import subprocess

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # experiments/, for the canvas package
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
VOL = "canvas-data"

PARTY = ("girih_stars", "quilt_patch", "circuit_traces", "tumbling_blocks",
         "truchet_meander", "stained_glass_cells")
TEXTUREY = ("agate_bands", "brain_coral", "wood_grain", "lichen_crust")
CGRP = {"party": "#c0392b", "texturey": "#2471a3", "other": "#a6acaf"}


def grp(s):
    return "party" if s in PARTY else ("texturey" if s in TEXTUREY else "other")


def fetch(tag, qtag, k, strip=True):
    os.makedirs(RES, exist_ok=True)
    want = [(f"plant/{qtag}/quant_stats.json", f"quant_{qtag}.json"),
            (f"plant/{tag}/run.json", f"run_{tag}.json")]
    if strip:
        want += [(f"plant/{tag}/strip.npz", f"strip_{tag}.npz"),
                 (f"plant/{qtag}/quant_K{k}.npz", f"quant_K{k}_{qtag}.npz")]
    for remote, local in want:
        dst = os.path.join(RES, local)
        if os.path.exists(dst):
            os.remove(dst)
        r = subprocess.run(["modal", "volume", "get", VOL, remote, dst],
                           capture_output=True, text=True,
                           env=dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic")))
        print(("ok  " if r.returncode == 0 else "MISS") + f" {remote}")


# --------------------------------------------------------------------------- #

def fig_quant(q, out):
    import matplotlib.pyplot as plt
    ks = sorted(q["per_k"], key=int)
    styles = sorted(q["per_k"][ks[0]]["per_style"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for s in styles:
        y = [q["per_k"][k]["per_style"][s]["nmse_all"] for k in ks]
        ax[0].plot([int(k) for k in ks], y, "-o", ms=3, lw=1, color=CGRP[grp(s)], alpha=0.75)
    for g, c in CGRP.items():
        ax[0].plot([], [], color=c, label=g)
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("codebook size K"); ax[0].set_ylabel("patch NMSE (1.0 = the K=1 floor)")
    ax[0].set_title("quantizer floor per style"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    kk = ks[-1]
    order = sorted(styles, key=lambda s: q["per_k"][kk]["per_style"][s]["nmse_all"])
    ax[1].barh(range(len(order)),
               [q["per_k"][kk]["per_style"][s]["nmse_all"] for s in order],
               color=[CGRP[grp(s)] for s in order])
    ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(order, fontsize=6)
    ax[1].set_xlabel(f"patch NMSE at K={kk}"); ax[1].set_title("hardest styles to quantize")
    ax[1].grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_wave(R, out):
    import matplotlib.pyplot as plt
    steps = np.array(R["wave"]["steps"]); lad = R["wave"]["ladder"]
    nll = np.array(R["wave"]["nll"])                       # (ckpt, style, side)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    m = nll.mean(1)
    for j, side in enumerate(lad):
        ax[0].plot(steps, m[:, j], "-o", ms=3, label=f"{side}x{side} ({side*side})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("plant training step")
    ax[0].set_ylabel("held-out NLL / masked token (nats)")
    ax[0].set_title("mask-size ladder (mean over 36 styles)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    frac = (m - m[-1]) / np.maximum(m[0] - m[-1], 1e-9)
    for j, side in enumerate(lad):
        ax[1].plot(steps, frac[:, j], "-o", ms=3, label=f"{side*side}")
    ax[1].axhline(0.5, color="k", lw=.6, ls=":")
    ax[1].set_xscale("log"); ax[1].set_xlabel("plant training step")
    ax[1].set_ylabel("fraction of this size's total drop remaining")
    ax[1].set_title("is coarse structure learned after fine?"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    # The per-size normalization above hides the structure: what actually happens is a
    # SHARED phase in which every mask size improves together, then a divergence in which only
    # the small holes keep improving. Panel 3 decomposes the total drop into those two.
    knee = int(np.argmin(np.abs(np.log(steps) - np.log(1403)))) if len(steps) > 3 else 0
    area = np.array([s * s for s in lad])
    shared = m[0] - m[knee]
    after = m[knee] - m[-1]
    ax[2].plot(area, shared, "-o", label=f"drop by step {steps[knee]} (shared phase)")
    ax[2].plot(area, after, "-o", label=f"drop after step {steps[knee]}")
    ax[2].plot(area, m[-1], "-o", color="k", label="final held-out NLL")
    best = m.min(0)
    ax[2].plot(area, best, "--", color="gray", lw=1, label="best over checkpoints")
    ax[2].set_xscale("log"); ax[2].set_xlabel("mask area (cells)")
    ax[2].set_ylabel("nats / masked token")
    ax[2].set_title("two phases: shared, then only the small holes")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_recur(R, out, q=None, k=512):
    import matplotlib.pyplot as plt
    rec = R["recurrence"]; styles = list(rec)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    x = [rec[s]["T2"]["distinct_per_obs"] for s in styles]
    y = [rec[s]["T2"]["mass_at_support"] for s in styles]
    z = [rec[s]["T3"]["frac_obs_all_at_support"] for s in styles]
    for i, s in enumerate(styles):
        ax[0].scatter(x[i], y[i], color=CGRP[grp(s)], s=26)
        if grp(s) != "other":
            ax[0].annotate(s, (x[i], y[i]), fontsize=5.5, alpha=.8)
    ax[0].set_xlabel("distinct 2x2 code blocks / occurrence"); ax[0].set_ylabel("mass at support (>=8)")
    ax[0].set_title("T[2]: do code tuples recur?"); ax[0].grid(alpha=.3)
    for i, s in enumerate(styles):
        ax[1].scatter(y[i], z[i], color=CGRP[grp(s)], s=26)
        if grp(s) != "other":
            ax[1].annotate(s, (y[i], z[i]), fontsize=5.5, alpha=.8)
    ax[1].set_xlabel("T[2] mass at support"); ax[1].set_ylabel("T[3] occurrences with ALL halves at support")
    ax[1].set_title("T[3]: does the ratchet constraint survive one level up?"); ax[1].grid(alpha=.3)
    order = sorted(styles, key=lambda s: -rec[s]["T2"]["mass_at_support"])
    ax[2].barh(range(len(order)), [rec[s]["T2"]["mass_at_support"] for s in order],
               color=[CGRP[grp(s)] for s in order])
    ax[2].barh(range(len(order)), [rec[s]["T3"]["frac_obs_all_at_support"] for s in order],
               color="k", height=.35)
    ax[2].set_yticks(range(len(order))); ax[2].set_yticklabels(order, fontsize=6)
    ax[2].set_xlabel("bar = T[2] mass at support, black = T[3] pure fraction")
    ax[2].grid(alpha=.3, axis="x")
    if q is not None:
        ps = q["per_k"][str(k)]["per_style"]
        pp = [ps[s]["code_perplexity"] for s in styles]
        for i, s in enumerate(styles):
            ax[3].scatter(pp[i], y[i], color=CGRP[grp(s)], s=26)
            if grp(s) != "other":
                ax[3].annotate(s, (pp[i], y[i]), fontsize=5.5, alpha=.8)
        ra, rb = np.argsort(np.argsort(pp)), np.argsort(np.argsort(y))
        r = float(np.corrcoef(ra, rb)[0, 1])
        ax[3].set_xscale("log"); ax[3].set_xlabel(f"per-style code perplexity at K={k}")
        ax[3].set_ylabel("T[2] mass at support")
        ax[3].set_title(f"recurrence tracks code entropy, not parts (Spearman {r:+.2f})")
        ax[3].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_c2d(R, out):
    import matplotlib.pyplot as plt
    c2d = R["cost_to_depth"]
    sides = sorted({v["side"] for v in c2d.values()})
    samplers = ["argmax1", "iter8", "beam8x4"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for nm in samplers:
        cost = c2d[f"{sides[0]}_{nm}"]["cost"]
        a = [c2d[f"{s}_{nm}"]["code_acc"] for s in sides]
        n = [c2d[f"{s}_{nm}"]["gB_nll"] for s in sides]
        p = [c2d[f"{s}_{nm}"]["gB_pass"] for s in sides]
        ax[0].plot([s * s for s in sides], a, "-o", ms=4, label=f"{nm} (cost {cost})")
        ax[1].plot([s * s for s in sides], n, "-o", ms=4, label=f"{nm} (cost {cost})")
        ax[2].plot([s * s for s in sides], p, "-o", ms=4, label=f"{nm} (cost {cost})")
    for a_, t, yl in zip(ax, ["exact code recovery", "reporting grader NLL",
                              "reporting grader pass rate"],
                         ["fraction of masked cells exactly right", "NLL / token (nats)",
                          "pass rate at q=0.90"]):
        a_.set_xscale("log"); a_.set_xlabel("mask area (cells)"); a_.set_ylabel(yl)
        a_.set_title(t); a_.legend(fontsize=8); a_.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_grader(R, out):
    import matplotlib.pyplot as plt
    G = R["grader"]; names = [g for g in G if g.startswith("gA")] + ["gB"]
    sides = sorted({int(k.split("_")[0]) for k in G[names[0]] if not k.startswith("_")})
    classes = ["clean", "same_style_other", "roll", "shuffle", "other_style", "marginal",
               "uniform", "plant_argmax", "plant_iter8", "plant_beam4"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    n_ex = [int(g[2:]) for g in names if g.startswith("gA")]
    for c in classes:
        y = [np.mean([G[f"gA{n}"][f"{s}_{c}"]["pass"] for s in sides]) for n in n_ex]
        ax[0].plot(n_ex, y, "-o", ms=4, label=c, lw=2 if c in ("clean", "shuffle") else 1)
    ax[0].axhline(0.90, color="k", ls=":", lw=.8); ax[0].axhline(0.10, color="k", ls=":", lw=.8)
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("gA corpus (swatches per style)")
    ax[0].set_ylabel("pass rate at q=0.90"); ax[0].set_title("the corpus-size flip")
    ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)
    sel = R["selected_grader"]
    for g in [sel, "gB"]:
        r = np.array(R["roc"][g])
        ax[1].plot(r[:, 2], r[:, 1], "-", label=f"{g}")
        i = int(np.argmin(np.abs(r[:, 0] - 0.90)))
        ax[1].scatter([r[i, 2]], [r[i, 1]], zorder=5)
    ax[1].plot([0, 1], [0, 1], "k:", lw=.6)
    ax[1].set_xlabel("pass rate on self-manufactured damage")
    ax[1].set_ylabel("pass rate on clean held-out exemplars")
    ax[1].set_title("ROC over q (dot = the q=0.90 rule)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    w = 0.8 / len(sides)
    for j, s in enumerate(sides):
        ax[2].bar(np.arange(len(classes)) + j * w, [G[sel][f"{s}_{c}"]["pass"] for c in classes],
                  width=w, label=f"mask {s}x{s}")
    ax[2].set_xticks(np.arange(len(classes)) + 0.4); ax[2].set_xticklabels(classes, rotation=40,
                                                                          ha="right", fontsize=7)
    ax[2].axhline(0.90, color="k", ls=":", lw=.8); ax[2].axhline(0.10, color="k", ls=":", lw=.8)
    ax[2].set_ylabel("pass rate"); ax[2].set_title(f"{sel} by candidate class")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_strip(R, tag, qtag, k, out, n_styles=8):
    import matplotlib.pyplot as plt
    from canvas.plant.codebook import from_patches
    sp = np.load(os.path.join(RES, f"strip_{tag}.npz"))
    qz = np.load(os.path.join(RES, f"quant_K{k}_{qtag}.npz"))
    cent = qz["centroids"].astype(np.float32)
    styles = [s for s in R["styles"] if grp(s) != "other"][:n_styles]
    sides = sorted({int(kk.split("|")[1]) for kk in sp.files})
    cols = ["truth", "argmax1", "iter8", "beam8x4"]
    fig, ax = plt.subplots(len(styles) * len(sides), len(cols) + 1,
                           figsize=(2.0 * (len(cols) + 1), 2.0 * len(styles) * len(sides)))
    r = 0
    for s in styles:
        for side in sides:
            m = sp[f"{s}|{side}|mask"].reshape(16, 16)
            for c, name in enumerate(["masked"] + cols):
                key = "truth" if name == "masked" else name
                img = from_patches(cent[sp[f"{s}|{side}|{key}"].astype(np.int64)][None])[0]
                if name == "masked":
                    img = img.copy()
                    img[np.kron(m, np.ones((16, 16), bool))] = 1.0
                ax[r, c].imshow(np.clip(img, 0, 1)); ax[r, c].axis("off")
                if r == 0:
                    ax[r, c].set_title(name, fontsize=8)
            ax[r, 0].set_ylabel(f"{s}\n{side}x{side}", fontsize=6)
            ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
            r += 1
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pl0")
    ap.add_argument("--qtag", default="q0")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag, a.qtag, a.k)
    os.makedirs(FIG, exist_ok=True)
    q = json.load(open(os.path.join(RES, f"quant_{a.qtag}.json")))
    fig_quant(q, os.path.join(FIG, "quant_floor.png"))
    print("quantizer floor (mean over styles):")
    for kk in sorted(q["per_k"], key=int):
        v = q["per_k"][kk]
        print(f"  K={kk:>5}  mse {v['mse']:.5f}  psnr {v['psnr']:.2f}  nmse {v['nmse']:.4f}  "
              f"used {v['codes_used']}")
    rp = os.path.join(RES, f"run_{a.tag}.json")
    if a.figures and os.path.exists(rp):
        R = json.load(open(rp))
        fig_wave(R, os.path.join(FIG, "wave.png"))
        fig_recur(R, os.path.join(FIG, "recurrence.png"), q=q, k=a.k)
        fig_c2d(R, os.path.join(FIG, "cost_to_depth.png"))
        fig_grader(R, os.path.join(FIG, "grader_calibration.png"))
        try:
            fig_strip(R, a.tag, a.qtag, a.k, os.path.join(FIG, "completions.png"))
        except Exception as e:
            print("strip figure skipped:", e)
        print("selected grader:", R["selected_grader"])
        print("figures ->", FIG)


if __name__ == "__main__":
    main()
