"""Reduce a reread/lm run.

    python3 rhm/practice/reread/lm/analyze_lm.py --tag lm0 --fetch --figures

Readouts, in the order the question asks for them:
  0. THE FIDELITY GATE — `frozen_200000` at step 12000 IS `conditional_revision`'s Gate-0
     base (same DGP, pool, sampler, optimiser, steps), so its per-level recovery must
     reproduce the published d1 0.979 / d3 0.836 / d6 0.088.
  1. PER-LEVEL RECOVERY vs TOKENS CONSUMED, per arm, against the exact BP ceiling.
     The reread signature lives here: deep levels still climbing on the frozen corpus after
     val loss and shallow levels have saturated, tracking `fresh`.
  2. THE MEMORISATION GAP — val NLL on held-out fresh windows vs train NLL on the arm's own
     corpus. What "the archive is consumed" looks like if it happens.
  3. BP-REFERENCED EXCESS LOSS per arrival level — model NLL minus exact Bayes surprisal.
  4. THE ORDERING READOUT — tokens-to-reach-level-ℓ (first crossing of `thresh_frac` of that
     level's BP ceiling). Monotonicity in ℓ IS the vocabulary-gating mechanism; the
     frozen/fresh ratio per level is the "effectively unlimited data" quantification.
  5. SATURATION — where each arm's curve stops moving, per level, and whether the frozen
     arms stall before or after `fresh`.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_reread_lm"
LEVELS = ["d1", "d2", "d3", "d4", "d5", "d6"]           # d1 shallowest, d6 root


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or name == "setup.json":
            continue
        r = json.load(open(os.path.join(root, name)))
        arms[r["arm"]] = r
    order = [a for a in setup["arms"] if a in arms]
    return setup, {a: arms[a] for a in order}


def curve(rec, key, sub=None):
    if sub is None:
        return [r[key] for r in rec["log"]]
    return [r[key].get(sub) for r in rec["log"]]


def first_cross(toks, ys, thr):
    """Tokens at the first checkpoint whose value is >= thr (linear interp in log-tokens)."""
    for i, y in enumerate(ys):
        if y is not None and y >= thr:
            if i == 0:
                return float(toks[0])
            y0, y1 = ys[i - 1], y
            if y1 == y0:
                return float(toks[i])
            f = (thr - y0) / (y1 - y0)
            lo, hi = np.log(max(toks[i - 1], 1)), np.log(toks[i])
            return float(np.exp(lo + f * (hi - lo)))
    return None


# --------------------------------------------------------------------------- #

def report(tag):
    setup, arms = load(tag)
    cfg, ceil = setup["config"], setup["ceiling"]
    thr_frac = cfg["thresh_frac"]
    print(f"\n=== {tag} ===")
    print(f"{cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}D on v{cfg['v']}_s{cfg['s']}"
          f"_L{cfg['L']}_m{cfg['m']}  T={cfg['T']}  batch={cfg['batch_size']}  "
          f"max_steps={cfg['max_steps']} ({cfg['max_steps'] * cfg['batch_size'] * cfg['T']:,} "
          f"tokens consumed)")
    print("BP ceiling P(z_l | x_1..T): " + "  ".join(f"{k}={ceil[k]:.3f}" for k in LEVELS))
    print("Bayes surprisal per arrival level: "
          + "  ".join(f"{k}={x:.3f}" for k, x in sorted(setup["bayes_by_level"].items(),
                                                        key=lambda z: int(z[0]))))
    print("\ncorpora: " + " | ".join(
        f"{a}: {r['corpus_tokens']:,} tok"
        + (f" sha={r['corpus_sha']} ({cfg['max_steps'] * cfg['batch_size'] * cfg['T'] / max(r['corpus_tokens'], 1):.0f}x re-read)"
           if r["kind"] == "frozen" else " (unbounded)")
        for a, r in arms.items()))

    # ---- 0. the fidelity gate ----------------------------------------------------------
    print("\n-- 0. FIDELITY GATE --")
    ref, rstep = setup["cr_ref"], setup["cr_ref_step"]
    fr = arms.get("frozen_200000")
    if fr is None:
        print("   frozen_200000 not in this run — gate not evaluable")
    else:
        row = next((r for r in fr["log"] if r["step"] == rstep), None)
        if row is None:
            print(f"   no checkpoint at step {rstep}")
        else:
            print(f"   conditional_revision Gate-0 base (published): "
                  + "  ".join(f"{k}={v:.3f}" for k, v in ref.items()))
            print(f"   frozen_200000 @ step {rstep} (this run):      "
                  + "  ".join(f"{k}={row['levels'][k]:.3f}" for k in ref))
            print("   delta:                                        "
                  + "  ".join(f"{k}={row['levels'][k] - v:+.3f}" for k, v in ref.items()))
            print(f"   (full best-over-all-blocks probe: {row['full_probe']})")

    # ---- 1. per-level recovery ----------------------------------------------------------
    print("\n-- 1. PER-LEVEL RECOVERY vs TOKENS (BP ceiling in brackets) --")
    toks = [r["tokens"] for r in list(arms.values())[0]["log"]]
    print("   tokens:  " + " ".join(f"{t/1e6:7.2f}M" for t in toks))
    for lvl in LEVELS:
        print(f"   {lvl}  [ceiling {ceil[lvl]:.3f}]")
        for a, r in arms.items():
            ys = curve(r, "levels", lvl)
            print(f"     {a:16s} " + " ".join(f"{y:8.3f}" for y in ys)
                  + f"   final {ys[-1]:.3f}  ({ys[-1]/ceil[lvl]:.2f} of ceiling)")

    # ---- 2. loss and the memorisation gap ----------------------------------------------
    print("\n-- 2. VAL NLL (held-out fresh windows) / TRAIN NLL / gap --")
    print("   tokens:  " + " ".join(f"{t/1e6:7.2f}M" for t in toks))
    for a, r in arms.items():
        v_ = curve(r, "val_nll")
        print(f"   {a:16s} val   " + " ".join(f"{y:8.4f}" for y in v_))
    for a, r in arms.items():
        t_ = curve(r, "train_nll")
        if t_[0] is None:
            continue
        v_ = curve(r, "val_nll")
        print(f"   {a:16s} gap   " + " ".join(f"{x-y:8.4f}" for x, y in zip(v_, t_))
              + f"   final gap {v_[-1]-t_[-1]:+.4f}")

    # ---- 3. BP-referenced excess loss ---------------------------------------------------
    print("\n-- 3. EXCESS NLL OVER THE EXACT BAYES FLOOR, by arrival level "
          "(0 = closes the root, 6 = closes only a leaf pair) --")
    keys = sorted(setup["bayes_by_level"], key=lambda z: int(z))
    for k in keys:
        print(f"   arrival level {k} (Bayes {setup['bayes_by_level'][k]:.3f}):")
        for a, r in arms.items():
            ys = [x["excess_by_level"].get(k) for x in r["log"]]
            print(f"     {a:16s} " + " ".join(f"{y:7.3f}" for y in ys)
                  + f"   final {ys[-1]:.3f}")

    # ---- 4. the ordering readout --------------------------------------------------------
    print(f"\n-- 4. TOKENS TO REACH {thr_frac:.0%} OF EACH LEVEL'S BP CEILING "
          "(the ordering readout, and the frozen/fresh ratio) --")
    cross = {}
    for a, r in arms.items():
        tk = [x["tokens"] for x in r["log"]]
        cross[a] = {lvl: first_cross(tk, curve(r, "levels", lvl), thr_frac * ceil[lvl])
                    for lvl in LEVELS}
    hdr = "   " + f"{'arm':16s} " + " ".join(f"{l:>11s}" for l in LEVELS)
    print(hdr)
    for a in arms:
        print(f"   {a:16s} " + " ".join(
            ("      never" if cross[a][l] is None else f"{cross[a][l]/1e6:10.2f}M")
            for l in LEVELS))
    print("\n   ORDERED? (is tokens-to-reach monotone increasing from d1 to d6 — "
          "level l+1 only after level l)")
    for a in arms:
        seq = [cross[a][l] for l in LEVELS]
        got = [x for x in seq if x is not None]
        mono = all(got[i] <= got[i + 1] + 1e-9 for i in range(len(got) - 1))
        print(f"     {a:16s} reached {len(got)}/6 levels, monotone={mono}, "
              f"order " + " < ".join(l for l in LEVELS if cross[a][l] is not None))
    if "fresh" in cross:
        print("\n   FROZEN / FRESH token ratio to the same level "
              "(>1 = the frozen corpus needs more tokens; 'never' = it never got there)")
        for a in arms:
            if a == "fresh":
                continue
            row = []
            for l in LEVELS:
                f_, c_ = cross["fresh"].get(l), cross[a].get(l)
                row.append("     never" if c_ is None else
                           ("      n/a" if not f_ else f"{c_/f_:9.2f}"))
            print(f"     {a:16s} " + " ".join(row))

    # ---- 5. saturation ------------------------------------------------------------------
    # PROTOCOL CAUTION: most checkpoints probe 5 blocks with a linear head; the two marked
    # `full_probe` use all 9 blocks and max(linear, MLP), which reads systematically HIGHER.
    # A late-movement number spanning the two protocols is a protocol change, not learning.
    # So this section compares only `full_probe` checkpoints with each other.
    any_arm = list(arms.values())[0]
    fi = [i for i, r in enumerate(any_arm["log"]) if r["full_probe"]]
    print("\n-- 5. LATE MOVEMENT, PROTOCOL-CLEAN: change in recovery between the two "
          "FULL-protocol checkpoints (all 9 blocks, max(linear, MLP)) --")
    if len(fi) >= 2:
        i0, i1 = fi[-2], fi[-1]
        print(f"   (from {toks[i0]/1e6:.2f}M to {toks[i1]/1e6:.2f}M tokens; "
              f"reduced-protocol checkpoints in between are NOT comparable)")
        print("   " + f"{'arm':16s} " + " ".join(f"{l:>8s}" for l in LEVELS) + "     val")
        for a, r in arms.items():
            d = [curve(r, "levels", l)[i1] - curve(r, "levels", l)[i0] for l in LEVELS]
            dv = curve(r, "val_nll")[i1] - curve(r, "val_nll")[i0]
            print(f"   {a:16s} " + " ".join(f"{x:+8.3f}" for x in d) + f"  {dv:+7.4f}")
    print("\n   PROTOCOL OFFSET (the size of the artefact): full-protocol checkpoint minus "
          "the reduced-protocol checkpoint nearest after it")
    for a, r in arms.items():
        row = []
        for l in LEVELS:
            ys = curve(r, "levels", l)
            nxt = [i for i in range(fi[-2] + 1, len(ys)) if not r["log"][i]["full_probe"]]
            row.append(f"{ys[fi[-2]] - ys[nxt[0]]:+7.3f}" if nxt else "   n/a ")
        print(f"     {a:16s} " + " ".join(row))
    return setup, arms


# --------------------------------------------------------------------------- #

def figures(tag, setup, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    ceil = setup["ceiling"]
    colors = {a: c for a, c in zip(arms, plt.cm.viridis(np.linspace(0, 0.9, len(arms))))}

    # fig1 — per-level recovery vs tokens, one panel per level
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, lvl in enumerate(LEVELS):
        ax = axes[i // 3][i % 3]
        for a, r in arms.items():
            tk = [x["tokens"] for x in r["log"]]
            ax.plot(tk, curve(r, "levels", lvl), marker="o", ms=3,
                    color=colors[a], label=a)
        ax.axhline(ceil[lvl], color="black", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_title(f"{lvl} recovery (BP ceiling {ceil[lvl]:.3f})")
        ax.set_xlabel("tokens consumed"); ax.set_ylabel("probe accuracy")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_levels.png"), dpi=130)

    # fig2 — loss and the memorisation gap
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, r in arms.items():
        tk = [x["tokens"] for x in r["log"]]
        axes[0].plot(tk, curve(r, "val_nll"), marker="o", ms=3, color=colors[a], label=a)
        t_ = curve(r, "train_nll")
        if t_[0] is not None:
            axes[1].plot(tk, [x - y for x, y in zip(curve(r, "val_nll"), t_)],
                         marker="o", ms=3, color=colors[a], label=a)
    for ax, ttl, yl in ((axes[0], "val NLL (held-out fresh windows)", "nats"),
                        (axes[1], "memorisation gap (val − train)", "nats")):
        ax.set_xscale("log"); ax.set_xlabel("tokens consumed"); ax.set_ylabel(yl)
        ax.set_title(ttl)
    axes[0].axhline(setup.get("bayes_mean", np.nan), color="black", ls="--", lw=1)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_loss.png"), dpi=130)

    # fig3 — BP-referenced excess loss per arrival level
    keys = sorted(setup["bayes_by_level"], key=lambda z: int(z))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.6), squeeze=False)
    for j, k in enumerate(keys):
        ax = axes[0][j]
        for a, r in arms.items():
            tk = [x["tokens"] for x in r["log"]]
            ax.plot(tk, [x["excess_by_level"].get(k) for x in r["log"]],
                    marker="o", ms=3, color=colors[a], label=a)
        ax.set_xscale("log"); ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"arrival level {k}"); ax.set_xlabel("tokens")
        if j == 0:
            ax.set_ylabel("NLL − exact Bayes (nats)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_excess.png"), dpi=130)

    # fig4 — tokens-to-reach each level (the ordering readout)
    thr_frac = setup["config"]["thresh_frac"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for a, r in arms.items():
        tk = [x["tokens"] for x in r["log"]]
        ys = [first_cross(tk, curve(r, "levels", l), thr_frac * ceil[l]) for l in LEVELS]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[i] for i in xs], marker="o", color=colors[a], label=a)
    ax.set_yscale("log"); ax.set_xticks(range(6)); ax.set_xticklabels(LEVELS)
    ax.set_xlabel("level (d1 shallowest → d6 root)")
    ax.set_ylabel(f"tokens to reach {thr_frac:.0%} of the BP ceiling")
    ax.set_title("the ordering readout: is extraction level-by-level, and does the "
                 "frozen corpus pay more?")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_ordering.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lm0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    setup, arms = report(a.tag)
    if a.figures:
        figures(a.tag, setup, arms)


if __name__ == "__main__":
    main()
