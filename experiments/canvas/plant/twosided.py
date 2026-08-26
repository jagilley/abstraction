"""Two-sided typical-set re-analysis of a panel dump. Offline; retrains nothing.

`critic/`'s rule is ONE-SIDED: pass iff the verdict's NLL is at most the q = 0.90 quantile of
the grader's NLL on genuine exemplars -- "at least as typical as 90% of real exemplars". On
RHM that was the right shape, because at uniform demand the reader's log-likelihood is FLAT on
the valid set (`critic/SPEC.md` §1: every derivation has probability m^-15), so there is no
such thing as a completion that is *too* typical.

`pl0` showed that property does not survive the port. The plant's own fills score 2.06-2.60
nats against genuine held-out exemplars' 3.27-3.46 and pass at 0.997-1.000 -- MORE typical
than the truth, because mean per-token NLL rewards the marginal mode (the blank fill). A
one-sided rule cannot express "too bland"; a two-sided one can:

    pass iff  tau_lo(style, size) <= nll <= tau_hi(style, size)

with both quantiles taken on the SAME oracle-free object the one-sided rule uses -- the
grader's own NLL on a disjoint validation slice of genuine exemplars of that style at that
mask size. Nothing new is measured; only the decision rule changes.

    cd experiments
    python3 canvas/plant/twosided.py --tag pl0
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

CLASSES = ("clean", "same_style_other", "roll", "shuffle", "other_style", "marginal",
           "uniform", "plant_argmax", "plant_iter8", "plant_beam4")
SIDES = (2, 4, 8, 11)
DAMAGE = ("shuffle", "other_style", "marginal")


def buckets(style, side):
    """Bucket key as a single int: style * 1000 + mask side. Vectorised throughout, because
    the calibration slice is 18k items and the panel 8.6k."""
    return np.asarray(style, np.int64) * 1000 + np.asarray(side, np.int64)


def group(nll, keys, min_n=8):
    out = {}
    for kk in np.unique(keys):
        v = nll[keys == kk]
        if len(v) >= min_n:
            out[int(kk)] = v
    return out


def taus(nll, keys, q, min_n=8):
    return {kk: float(np.quantile(v, q)) for kk, v in group(nll, keys, min_n).items()}


def apply2(nll, keys, lo, hi):
    fb_lo = float(np.median(list(lo.values()))) if lo else -np.inf
    fb_hi = float(np.median(list(hi.values()))) if hi else np.inf
    a = np.array([lo.get(int(kk), fb_lo) for kk in keys])
    b = np.array([hi.get(int(kk), fb_hi) for kk in keys])
    return (nll >= a) & (nll <= b)


def load(tag, col=""):
    z = np.load(os.path.join(RES, f"panel_dump_{tag}.npz"))
    out = {}
    for g in ("gA24", "gB"):
        cal = z[f"cal{col}|{g}"]
        ck = buckets(z[f"calstyle|{g}"], z[f"calside|{g}"])
        panel = {}
        for s in SIDES:
            st = z[f"pstyle|{s}"]
            for c in CLASSES:
                panel[(s, c)] = (z[f"p{col}|{g}|{s}|{c}"], buckets(st, [s] * len(st)))
        out[g] = (cal, ck, panel)
    return out, z


def sweep(cal, ck, panel, q_hi=0.90, q_los=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)):
    hi = taus(cal, ck, q_hi)
    rows = []
    for ql in q_los:
        lo = taus(cal, ck, ql) if ql > 0 else {}
        r = {"q_lo": ql}
        for c in CLASSES:
            v = [apply2(panel[(s, c)][0], panel[(s, c)][1], lo, hi).mean() for s in SIDES]
            r[c] = float(np.mean(v))
        r["damage"] = float(np.mean([r[c] for c in DAMAGE]))
        r["plant"] = float(np.mean([r[c] for c in ("plant_argmax", "plant_iter8", "plant_beam4")]))
        rows.append(r)
    return rows


def percentiles(cal, ck, panel):
    """Where each candidate's NLL sits inside its own bucket's genuine-exemplar distribution."""
    ref = {kk: np.sort(v) for kk, v in group(cal, ck).items()}
    out = {}
    for c in CLASSES:
        pcs, below = [], []
        for s in SIDES:
            nll, keys = panel[(s, c)]
            for kk in np.unique(keys):
                r = ref.get(int(kk))
                if r is None:
                    continue
                v = nll[keys == kk]
                pcs.append(np.searchsorted(r, v) / len(r))
                below.append((v < r[0]).astype(float))
        out[c] = (float(np.mean(np.concatenate(pcs))), float(np.mean(np.concatenate(below))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pl0")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    res = {}
    for col, nm in (("", "8-forward chain rule"), ("1", "1-forward independent")):
        D, _ = load(a.tag, col)
        for g, (cal, ck, panel) in D.items():
            res[(g, col)] = sweep(cal, ck, panel)
            if col == "":
                res[("pct", g)] = percentiles(cal, ck, panel)
        print(f"\n=== {nm} ===")
        for g in ("gA24", "gB"):
            print(f"-- {g} (q_hi = 0.90) --")
            hdr = ["q_lo"] + [c[:9] for c in CLASSES] + ["DAMAGE", "PLANT"]
            print("".join(f"{h:>11}" for h in hdr))
            for r in res[(g, col)]:
                print(f"{r['q_lo']:>11.2f}" + "".join(f"{r[c]:11.3f}" for c in CLASSES)
                      + f"{r['damage']:11.3f}{r['plant']:11.3f}")
    print("\n=== where the candidates sit in the genuine-exemplar distribution (gA24, 8-fwd) ===")
    print(f"{'class':>18}{'mean pctile':>14}{'frac below min':>16}")
    for c, (p, b) in res[("pct", "gA24")].items():
        print(f"{c:>18}{p:14.3f}{b:16.3f}")

    tag = a.tag
    # ---- figure ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    rows = res[("gA24", "")]
    ql = [r["q_lo"] for r in rows]
    for c in CLASSES:
        style = "-" if not c.startswith("plant") else "--"
        lw = 2.2 if c in ("clean", "same_style_other") or c.startswith("plant_argmax") else 1.1
        ax[0].plot(ql, [r[c] for r in rows], style, lw=lw, marker="o", ms=3, label=c)
    ax[0].axhline(0.90, color="k", ls=":", lw=.8); ax[0].axhline(0.10, color="k", ls=":", lw=.8)
    ax[0].set_xlabel("lower quantile q_lo (0 = the one-sided rule)")
    ax[0].set_ylabel("pass rate (q_hi = 0.90)")
    ax[0].set_title("gA24: two-sided typical set"); ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=.3)
    ax[1].plot([r["clean"] for r in rows], [r["plant"] for r in rows], "-o", label="plant fills")
    ax[1].plot([r["clean"] for r in rows], [r["same_style_other"] for r in rows], "-o",
               label="same_style_other (valid alt)")
    ax[1].plot([r["clean"] for r in rows], [r["damage"] for r in rows], "-o", label="damage set")
    for r in rows:
        ax[1].annotate(f"{r['q_lo']:.2f}", (r["clean"], r["plant"]), fontsize=6)
    ax[1].plot([0, 1], [0, 1], "k:", lw=.6)
    ax[1].set_xlabel("pass rate on clean held-out exemplars"); ax[1].set_ylabel("pass rate")
    ax[1].set_title("what the lower cut costs and buys"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    pc = res[("pct", "gA24")]
    names = list(pc)
    ax[2].barh(range(len(names)), [pc[c][0] for c in names], color="#2471a3", label="mean percentile")
    ax[2].barh(range(len(names)), [pc[c][1] for c in names], color="k", height=.35,
               label="fraction below the bucket minimum")
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=7)
    ax[2].axvline(0.5, color="gray", ls=":", lw=.8)
    ax[2].set_xlabel("position inside the genuine-exemplar NLL distribution")
    ax[2].set_title("the plant is off the bottom of the scale"); ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"two_sided_{tag}.png"), dpi=140); plt.close(fig)
    print("\nfigure ->", os.path.join(FIG, f"two_sided_{tag}.png"))


if __name__ == "__main__":
    main()
