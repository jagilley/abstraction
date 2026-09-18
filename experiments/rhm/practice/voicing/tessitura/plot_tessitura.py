"""[tessitura] The figures. Three panels per pass, and nothing that is not in a reduction.

  `figures/ts_world.png`   the world each arm was fed, per era, across seeds/diets/graders,
                           with the probe rate beside it — `reduce_logged.py` [W].
  `figures/ts_norm.png`    the norm against the world: per-slot calibration at both banked
                           arms, the reliability curve, and the two critics cross-scored —
                           `reduce_rows.py` [N] and [R].
  `figures/ts_twins.png`   the twin contrast: the judge's probability difference against the
                           world's, per slot and by quintile of the prior's own difference —
                           `reduce_rows.py` [X].
  `figures/ts_rerun{,_s2}.png`  each paid arm: the norm's time course, the fixed panels' drift on
                           identical rows, and what the judge reads at the write with the other
                           label held fixed — `reduce_rerun.py` [N] [P] [C].
  `figures/ts_refit.png`   the diets: one trunk, seven diets, one set of held-out rows; the
                           shift-vs-rescaling regression on both trunks; and the filed-only
                           diet's blindness to counterfactuals read on the LEVEL — `refit.py`.

PNGs are gitignored under `experiments/`; regenerate with this file.

Usage (from experiments/):
    python3 rhm/practice/voicing/tessitura/plot_tessitura.py
"""
import collections
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VOICING = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(VOICING, "..", "..", "..")))
FIGD = os.path.join(HERE, "figures")
FIG = os.path.join(VOICING, "figures")

INK = "#1b1b1b"
C = {"s0": "#1f4e79", "s2": "#b5651d", "world": "#7a7a7a", "judge": "#1f4e79",
     "probe": "#b5651d", "filed": "#1f4e79"}


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK)
    ax.grid(alpha=0.15, lw=0.6)


# ------------------------------------------------------------------- the world ------- #

def fig_world():
    from rhm.practice.voicing.tessitura.reduce_logged import (
        find_arms, era_of_cycle, per_cycle_world, DIET)
    rows = []
    for tag, arm, p in find_arms():
        d = json.load(open(p))
        v = d["log"].get("vo") or []
        if not any((c or {}).get("critic") for c in v):
            continue
        eras = era_of_cycle(d["log"])
        inst, pinst, nf, npb = per_cycle_world(v)
        diet, grader, _ = DIET.get(arm, ("?", "?", "?"))
        rows.append((f"{tag}:{arm}", int(d["config"].get("seed", -1)), diet, grader,
                     eras, inst, pinst, nf, npb))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, which, ttl in ((axes[0], 5, "filed writes — what the learner wrote"),
                           (axes[1], 6, "probe substitutions — the road not taken")):
        for nm, seed, diet, grader, eras, inst, pinst, nf, npb in rows:
            src = inst if which == 5 else pinst
            w = nf if which == 5 else npb
            ys = []
            for e in range(1, 6):
                m = eras == e
                ys.append(np.nansum(src[m] * w[m]) / w[m].sum() if w[m].sum() else np.nan)
            col = C["s0"] if seed == 0 else (C["s2"] if seed == 2 else "#4a7c59")
            ls = {"world": "-", "mirror": "--", "committee": ":", "hybrid": "-.",
                  "-": (0, (5, 1))}.get(grader, "-")
            ax.plot(list(range(1, 6)), ys, color=col, lw=1.3, alpha=0.8, marker="o", ms=3,
                    linestyle=ls)
        _style(ax)
        ax.set_xlabel("era  (damage depth = era index)", fontsize=9, color=INK)
        ax.set_ylabel("solve rate", fontsize=9, color=INK)
        ax.set_title(ttl, fontsize=10, color=INK)
        ax.set_xticks(range(1, 6))
    axes[0].text(0.02, 0.97, "seed 0 blue · seed 2 orange · seed 1/3 green\n"
                             "world grader solid · mirror dashed · committee dotted",
                 transform=axes[0].transAxes, fontsize=7.5, va="top", color=INK)
    fig.suptitle("[tessitura] the world each judge was fed — 24 arms, one era ladder",
                 fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(os.path.join(FIGD, "ts_world.png"), dpi=150)
    plt.close(fig)
    return len(rows)


# -------------------------------------------------------------------- the norm ------- #

def _parse_norm(path):
    """Read the per-slot norm table straight out of `results/rows.txt`, so the figure and the
    reduction cannot disagree — the reduction is the record and this only draws it."""
    arms = collections.OrderedDict()
    cur_arm = cur_which = None
    for ln in open(path):
        if ln.strip().startswith("ov_") and "(seed" in ln:
            cur_arm = ln.strip().split(" ")[0]
            arms.setdefault(cur_arm, {"filed": [], "probe": [], "rel": {}})
        elif ln.strip() in ("filed", "probe") and cur_arm:
            cur_which = ln.strip()
        elif cur_arm and cur_which and ln.startswith("      ") and ":" in ln.split()[0]:
            t = ln.split()
            try:
                arms[cur_arm][cur_which].append((t[0], int(t[1]), float(t[2]), float(t[3])))
            except (ValueError, IndexError):
                pass
        elif cur_arm and ln.strip().startswith(("filed  (pred/obs)", "probe  (pred/obs)")):
            w = ln.strip().split("(pred/obs)")[0].strip()
            arms[cur_arm]["rel"][w] = [tuple(float(z) for z in c.split("/"))
                                       for c in ln.strip().split("(pred/obs)")[1].split()]
    return arms


def fig_norm():
    path = os.path.join(HERE, "results", "rows.txt")
    arms = _parse_norm(path)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax = axes[0]
    for i, (nm, d) in enumerate(arms.items()):
        col = C["s0"] if "s0b" in nm else C["s2"]
        for which, mk in (("filed", "o"), ("probe", "^")):
            xs = [q[2] for q in d[which]]
            ys = [q[3] for q in d[which]]
            ax.scatter(xs, ys, s=16, marker=mk, color=col, alpha=0.75,
                       label=f"{nm.split(':')[0]} {which}")
    lim = [0, 0.62]
    ax.plot(lim, lim, color=INK, lw=0.8, ls="--", alpha=0.6)
    _style(ax)
    ax.set_xlabel("the world: base rate on the rows", fontsize=9)
    ax.set_ylabel("the norm: judge's mean P(solve)", fontsize=9)
    ax.set_title("the norm calibrates in LEVEL, per slot", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    ax = axes[1]
    for nm, d in arms.items():
        col = C["s0"] if "s0b" in nm else C["s2"]
        for which, ls in (("filed", "-"), ("probe", "--")):
            r = d["rel"].get(which)
            if not r:
                continue
            ax.plot([q[0] for q in r], [q[1] for q in r], color=col, lw=1.4,
                    marker="o", ms=3, linestyle=ls)
    ax.plot([0, 0.75], [0, 0.75], color=INK, lw=0.8, ls=":", alpha=0.6)
    _style(ax)
    ax.set_xlabel("predicted P(solve), decile mean", fontsize=9)
    ax.set_ylabel("observed solve rate", fontsize=9)
    ax.set_title("reliability — rows pooled over slots", fontsize=10)

    ax = axes[2]
    txt = []
    for ln in open(path):
        if "REGRESSION of the norm" in ln or ln.strip().startswith("POOLED: base"):
            txt.append(ln.strip())
    ax.axis("off")
    ax.text(0.0, 1.0, "the shift-versus-rescaling regression\n(norm §1's form, across slots)\n\n"
            + "\n".join(t[:64] for t in txt[:8]), fontsize=6.6, va="top", family="monospace",
            color=INK)
    fig.suptitle("[tessitura] the judge's norm against the world — two banked arms, "
                 "final critic, buffer tail", fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(os.path.join(FIGD, "ts_norm.png"), dpi=150)
    plt.close(fig)
    return len(arms)


# ------------------------------------------------------------------- the twins ------- #

def fig_twins():
    path = os.path.join(HERE, "results", "rows.txt")
    blocks, cur, inq = {}, None, False
    for ln in open(path):
        if ln.strip().startswith("ov_") and "(seed" in ln and inq:
            cur = ln.strip().split(" ")[0]
            blocks[cur] = {"rows": [], "quint": None}
        if "[X] THE TWINS" in ln:
            inq = True
        if "[R] ACROSS ARMS" in ln:
            inq = False
        if cur and inq and ln.startswith("    ") and ":" in ln.split()[0] if ln.split() else False:
            t = ln.split()
            try:
                blocks[cur]["rows"].append((t[0], float(t[4]), float(t[5]), float(t[9])))
            except (ValueError, IndexError):
                pass
        if cur and inq and "BY QUINTILE OF dDP" in ln:
            cells = ln.split("agreement):")[1].split()
            q = []
            for c in cells:
                a, b = c.split("->")
                bq, wq, sg = b.split("/")
                q.append((float(a), float(bq), float(wq), float(sg)))
            blocks[cur]["quint"] = q
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax = axes[0]
    for nm, d in blocks.items():
        col = C["s0"] if "s0b" in nm else C["s2"]
        ax.scatter([q[2] - q[1] for q in d["rows"]], [q[3] for q in d["rows"]],
                   s=18, color=col, alpha=0.8, label=nm.split(":")[0])
    lim = [-0.45, 0.0]
    ax.plot(lim, lim, color=INK, lw=0.8, ls="--", alpha=0.6)
    _style(ax)
    ax.set_xlabel("the world:  y(substituted) - y(written)", fontsize=9)
    ax.set_ylabel("the judge:  P(substituted) - P(written)", fontsize=9)
    ax.set_title("the twin contrast, per slot", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for nm, d in blocks.items():
        if not d["quint"]:
            continue
        col = C["s0"] if "s0b" in nm else C["s2"]
        q = d["quint"]
        ax.plot([x[0] for x in q], [x[1] for x in q], "-o", color=col, ms=4, lw=1.3)
    _style(ax)
    ax.set_xlabel("quintile mean of the PRIOR's own difference  (dp_sub - dp_written)",
                  fontsize=8)
    ax.set_ylabel("judge's logit difference", fontsize=9)
    ax.set_title("the twin contrast at matched prior", fontsize=10)

    ax = axes[2]
    for nm, d in blocks.items():
        if not d["quint"]:
            continue
        col = C["s0"] if "s0b" in nm else C["s2"]
        q = d["quint"]
        ax.plot([x[0] for x in q], [x[3] for x in q], "-o", color=col, ms=4, lw=1.3)
    ax.axhline(0.5, color=INK, lw=0.8, ls=":", alpha=0.6)
    _style(ax)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("quintile mean of the prior's own difference", fontsize=8)
    ax.set_ylabel("share of pairs the judge signs correctly", fontsize=9)
    ax.set_title("does the judge get the world's sign right", fontsize=10)
    fig.suptitle("[tessitura] the twins — a substituted class against the written one at an "
                 "identical context", fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(os.path.join(FIGD, "ts_twins.png"), dpi=150)
    plt.close(fig)
    return len(blocks)





# ------------------------------------------------------------------ the re-run ------- #

def fig_rerun(tag="ts_s0", arm="ovt_comp_pr_sh"):
    """The paid arm's three objects. Unlike the panels above this one recomputes from the
    tag's `results.json` rather than parsing the reduction, because the series are per-cycle
    and the reduction prints them pooled; the arithmetic is `reduce_rerun.py`'s, repeated."""
    root = os.path.join(FIG, tag, arm)
    d = json.load(open(os.path.join(root, "results.json")))
    v, eras = d["log"]["vo"], np.asarray(d["log"]["era"], np.int64)
    bnd = [int(np.nonzero(eras == e)[0][0]) for e in range(2, 6) if (eras == e).any()]

    def w_mean(vals, wts):
        vals, wts = np.asarray(vals, float), np.asarray(wts, float)
        m = np.isfinite(vals) & (wts > 0)
        return float((vals[m] * wts[m]).sum() / wts[m].sum()) if m.any() else np.nan

    base = np.full(len(v), np.nan)
    lvl = np.full(len(v), np.nan)
    pbase = np.full(len(v), np.nan)
    plvl = np.full(len(v), np.nan)
    for i, c in enumerate(v):
        cr = (c or {}).get("critic") or {}
        if not cr:
            continue
        w = [float(q.get("n") or 0) for q in cr.values()]
        base[i] = w_mean([q.get("base_rate") for q in cr.values()], w)
        lvl[i] = w_mean([(q.get("ts") or {}).get("mean_p") for q in cr.values()], w)
        pw = [float(q.get("probe_n") or 0) for q in cr.values()]
        pbase[i] = w_mean([q.get("probe_base_rate") for q in cr.values()], pw)
        plvl[i] = w_mean([((q.get("probe_x") or {}).get("ts") or {}).get("mean_p")
                          for q in cr.values()], pw)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax = axes[0]
    x = np.arange(len(v))
    ax.plot(x, base, color=C["world"], lw=1.6, label="the world (filed base rate)")
    ax.plot(x, lvl, color=C["judge"], lw=1.6, label="the norm (judge's mean P)")
    ax.plot(x, pbase, color=C["world"], lw=1.1, ls="--", label="the world (probe)")
    ax.plot(x, plvl, color=C["probe"], lw=1.1, ls="--", label="the norm (probe)")
    for b in bnd:
        ax.axvline(b, color=INK, lw=0.7, ls=":", alpha=0.45)
    _style(ax)
    ax.set_xlabel("cycle  (dotted lines are era boundaries; damage depth 1 -> 5)", fontsize=8)
    ax.set_ylabel("P(solve)", fontsize=9)
    ax.set_title("the norm's time course", fontsize=10)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")

    ax = axes[1]
    pan = collections.defaultdict(dict)
    for i, c in enumerate(v):
        for pk, q in ((c or {}).get("ts_panel") or {}).items():
            pan[pk][i] = q
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, 5))
    for fe in range(1, 6):
        ks = [k for k in pan if int(k.split("|e")[1]) == fe]
        if not ks:
            continue
        b = float(np.mean([next(iter(pan[k].values()))["base"] for k in ks]))
        xs, ys = [], []
        for e in range(1, 6):
            vals = [pan[k][i]["mean_p"] for k in ks for i in pan[k] if eras[i] == e]
            if vals:
                xs.append(e)
                ys.append(float(np.mean(vals)))
        ax.plot(xs, ys, "-o", color=cols[fe - 1], ms=4, lw=1.5,
                label=f"frozen in era {fe} (n={len(ks)})")
        ax.plot([xs[0] - 0.18, xs[0] + 0.18], [b, b], color=cols[fe - 1], lw=2.4, alpha=0.55)
    _style(ax)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("era the panel is READ in", fontsize=9)
    ax.set_ylabel("judge's mean P(solve) on the frozen rows", fontsize=9)
    ax.set_title("identical rows, a moving reader\n(short bars: each panel set's own base rate)",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[2]
    for which, ls in (("filed", "-"), ("probe", "--")):
        for key, col, nm in (("crit_y_g_st", C["s0"], "cost at matched structure"),
                             ("crit_st_g_y", C["s2"], "structure at matched cost")):
            xs, ys = [], []
            for e in range(1, 6):
                rows = [q for i, c in enumerate(v) if eras[i] == e
                        for k, q in ((c or {}).get("ts_struct") or {}).items()
                        if k.startswith(which + ":") and q.get(key) is not None]
                if not rows:
                    continue
                num = sum(q[key] * q["n"] for q in rows)
                den = sum(q["n"] for q in rows)
                xs.append(e)
                ys.append(num / den)
            ax.plot(xs, ys, color=col, lw=1.5, ls=ls, marker="o", ms=4,
                    label=f"{nm} ({which})")
    ax.axhline(0.5, color=INK, lw=0.8, ls=":", alpha=0.6)
    _style(ax)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("era  (damage depth)", fontsize=9)
    ax.set_ylabel("AUC, the other label held fixed", fontsize=9)
    ax.set_title("what the judge reads at the write", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle(f"[tessitura] {tag}:{arm} — the judge's level as it was trained "
                 "(gate TS-1: bit-identical to the banked arm)", fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(os.path.join(FIGD,
                             "ts_rerun.png" if tag == "ts_s0" else f"ts_rerun_{tag[-2:]}.png"),
                dpi=150)
    plt.close(fig)
    return len(pan)


if __name__ == "__main__":
    print("world  :", fig_world(), "arms ->", os.path.join(FIGD, "ts_world.png"))
    print("norm   :", fig_norm(), "arms ->", os.path.join(FIGD, "ts_norm.png"))
    print("twins  :", fig_twins(), "arms ->", os.path.join(FIGD, "ts_twins.png"))


# ------------------------------------------------------------------- the diets -------- #

def fig_refit(tag="ov_s0b"):
    """norm §1's two panels: the norm against the world it was fed, and the shift-vs-rescaling
    regression on both trunks. Parsed from `results/refit_<tag>.txt` so the figure cannot
    disagree with the reduction."""
    path = os.path.join(HERE, "results", f"refit_{tag}.txt")
    rows, reg, trunk, sec = {}, {}, None, None
    for ln in open(path):
        t = ln.split()
        if "TRUNK =" in ln:
            trunk = ln.strip().split("=")[1].strip()
        if "[N] THE NORM PER DIET" in ln:
            sec = "N"
        if "[S] SHIFT OR RESCALING" in ln:
            sec = "S"
        if not t or t[0] not in ("full", "filed", "probe", "both",
                                 "out_lo", "out_mid", "out_hi"):
            continue
        try:
            if sec == "N" and len(t) >= 9:
                rows[(trunk, t[0])] = tuple(float(x) for x in (t[1], t[2], t[4], t[5],
                                                               t[6], t[7]))
            elif sec == "S" and len(t) >= 6:
                reg[(trunk, t[0])] = tuple(float(x) for x in (t[1], t[2], t[3], t[4]))
        except ValueError:
            pass
    order = ["probe", "out_lo", "both", "full", "out_mid", "out_hi", "filed"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

    ax = axes[0]
    for trk, col, mk in (("trained", C["s0"], "o"), ("random", C["s2"], "^")):
        xs = [rows[(trk, d)][0] for d in order if (trk, d) in rows]
        ys = [rows[(trk, d)][1] for d in order if (trk, d) in rows]
        ax.scatter(xs, ys, s=44, color=col, marker=mk, alpha=0.85, label=f"{trk} trunk")
        for d in order:
            if (trk, d) in rows and trk == "trained":
                ax.annotate(d, (rows[(trk, d)][0], rows[(trk, d)][1]), fontsize=6.2,
                            xytext=(3, -8), textcoords="offset points", color=INK)
    ax.plot([0, 0.36], [0, 0.36], color=INK, lw=0.8, ls="--", alpha=0.6)
    _style(ax)
    ax.set_xlabel("the diet's E[outcome]", fontsize=9)
    ax.set_ylabel("the norm on the IDENTICAL held-out rows", fontsize=9)
    ax.set_title("one trunk, seven diets, one set of rows", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    fam = ["out_lo", "out_mid", "out_hi"]
    for trk, col, ls in (("trained", C["s0"], "-"), ("random", C["s2"], "--")):
        xs = [rows[(trk, d)][0] for d in fam if (trk, d) in rows]
        ax.plot(xs, [reg[(trk, d)][0] for d in fam if (trk, d) in reg], color=col,
                lw=1.6, marker="o", ms=5, linestyle=ls, label=f"slope ({trk})")
        ax.plot(xs, [reg[(trk, d)][3] for d in fam if (trk, d) in reg], color=col,
                lw=1.0, marker="s", ms=4, linestyle=":", alpha=0.7,
                label=f"sd ratio ({trk})")
    ax.axhline(1.0, color=INK, lw=0.8, ls=":", alpha=0.6)
    _style(ax)
    ax.set_xlabel("the diet's E[outcome]", fontsize=9)
    ax.set_ylabel("regression on `full`, identical rows", fontsize=9)
    ax.set_title("shift or rescaling\n(1.0 = a pure level shift)", fontsize=9.5)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")

    ax = axes[2]
    w = 0.36
    xs = np.arange(len(order))
    for k, (lab, col) in enumerate((("held-out FILED rows", C["s0"]),
                                    ("held-out PROBE rows", C["s2"]))):
        ax.bar(xs + (k - 0.5) * w, [rows[("trained", d)][2 + k] for d in order],
               width=w, color=col, alpha=0.85, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=7, rotation=30)
    _style(ax)
    ax.set_ylabel("the refit's norm", fontsize=9)
    ax.set_title("a filed-only diet cannot tell a counterfactual\nfrom a real write, in LEVEL",
                 fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle("[tessitura] the diets — norm §1's between-subjects design, "
                 "voicing's trunk held fixed", fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(FIGD, exist_ok=True)
    fig.savefig(os.path.join(FIGD, "ts_refit.png"), dpi=150)
    plt.close(fig)
    return len(rows)


if __name__ == "__main__":
    for _t in ("ts_s0", "ts_s2"):
        if os.path.isfile(os.path.join(FIG, _t, "ovt_comp_pr_sh", "results.json")):
            print(f"re-run {_t}:", fig_rerun(_t), "panels ->",
                  os.path.join(FIGD, f"ts_rerun{'' if _t == 'ts_s0' else '_s2'}.png"))
    if os.path.isfile(os.path.join(HERE, "results", "refit_ov_s0b.txt")):
        print("diets  :", fig_refit(), "cells ->", os.path.join(FIGD, "ts_refit.png"))
