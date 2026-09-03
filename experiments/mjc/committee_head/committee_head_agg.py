"""Aggregator for `committee_head` runs -- merges tags (the ladder may be split across several
Modal jobs at one seed, since the drift trajectory and the base committee are pure functions of the
seed) and seeds, and prints the four tables this cut is for:

  1. THE LADDER          -- region-A FM error (the sighted grader), ballistic control, budget shares
                            to A / off-reach-reducible / noise, leak, monitor:collect.
  2. SPLIT QUALITY       -- did the head learn the three-way split? Predicted reducibility vs
                            realised next-round `lprog`, predicted relevance vs realised visits,
                            and the AUROC of the head's own reducibility prediction for
                            reducible-vs-noise. Measured on EVERY arm, because a shadow head runs
                            everywhere -- so "learned the split" is separable from "allocated well".
  3. THE ABLATION TABLE  -- which channel carries the split. This is the product of the cut.
  4. PHASE A             -- raw-channel separability (AUROC per channel), committee health
                            (ensemble cosine, in-data vs in-region disagreement), benchmark
                            estimability (`b-e` by region class), and heterogeneous_graders §9's
                            discriminator (does disagreement CLOSE when you collect there).

Also writes `fig_ladder.png` (region-A error + ballistic vs round) and `fig_signature.png` (the
per-region signature over rounds, coloured by region class -- SPEC Phase A#4's read-only plot).

Run:
    python3 mjc/committee_head/committee_head_agg.py --tags ch_s0
    python3 mjc/committee_head/committee_head_agg.py --tags ch_core_s0 ch_heads_s0 ch_abl_s0
    python3 mjc/committee_head/committee_head_agg.py --tags ch_s0 ch_s1 ch_s2 --by-seed
"""

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_COL = {"A": "#d1603d", "offred": "#3d6fd1", "noise": "#868e96"}


def _load(tag):
    for p in (os.path.join(_HERE, "figures", "committee_head_" + tag, "results.json"),
              os.path.join(_HERE, "data", tag, "results.json")):
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(f"no results.json for tag {tag!r} (looked in figures/ and data/)")


def _cls(d, j):
    if d["region_names"][j] == "A":
        return "A"
    return "offred" if d["region_reducible"][j] else "noise"


def _mstd(v):
    v = np.asarray([x for x in v if x == x], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / max(len(v) ** 0.5, 1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--by-seed", action="store_true", help="treat tags as seeds (report mean±sem)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    runs = [_load(t) for t in a.tags]
    ref = runs[0]
    names = ref["region_names"]
    K = len(names)
    a_idx = names.index("A") if "A" in names else 0
    noise_j = [j for j in range(K) if not ref["region_reducible"][j]]
    offred_j = [j for j in range(K) if ref["region_reducible"][j] and j != a_idx]

    # merge: policy -> list of (run, hist). Tags at one seed contribute disjoint policies; tags at
    # different seeds contribute repeats of the same policy.
    pol_runs = {}
    for r in runs:
        for p, h in r["results"].items():
            pol_runs.setdefault(p, []).append((r, h))
    order = [p for p in ("uniform", "error-only", "disagree-only", "lprog-only", "visits-only",
                         "steady", "burst", "value", "oracle", "head_sup", "head_hetero",
                         "head_rl") if p in pol_runs]
    order += sorted(p for p in pol_runs if p not in order)

    def agg(p, fn):
        return _mstd([fn(r, h) for r, h in pol_runs[p]])

    def regA(r, h):
        return float(np.nanmean([x["reg_err"][a_idx] for x in h]))

    def ball(r, h):
        return float(np.nanmean([x["ballistic_cem"] for x in h]))

    def share(js):
        def f(r, h):
            s = np.mean(np.stack([np.asarray(x["alloc"]) / max(sum(x["alloc"]), 1.0) for x in h]), 0)
            return float(s[js].sum())
        return f

    def mcr(r, h):
        return sum(x["mon_steps"] for x in h) / max(sum(x["coll_steps"] for x in h), 1)

    print(f"\n{'='*100}\nTABLE 1 -- THE LADDER   (tags: {' '.join(a.tags)};  "
          f"regions {names};  n per policy = seeds/tags contributing)\n{'='*100}")
    print(f"{'policy':>16s} {'A-err ↓':>14s} {'ballistic ↓':>13s} {'→A':>6s} {'→offred':>8s} "
          f"{'→noise':>7s} {'mon:coll':>9s} {'n':>3s}")
    for p in order:
        m1, s1 = agg(p, regA); m2, s2 = agg(p, ball)
        sa, _ = agg(p, share([a_idx])); so, _ = agg(p, share(offred_j)); sn, _ = agg(p, share(noise_j))
        mc, _ = agg(p, mcr)
        print(f"{p:>16s} {m1:8.4f}±{s1:<5.3f} {m2:8.4f}±{s2:<4.3f} {sa:6.2f} {so:8.2f} "
              f"{sn:7.2f} {mc:9.2f} {len(pol_runs[p]):3d}")
    if "uniform" in pol_runs and "oracle" in pol_runs:
        u = agg("uniform", regA)[0]; o = agg("oracle", regA)[0]
        print(f"\n  gap-closed on the sighted grader (uniform={u:.4f} -> oracle={o:.4f}):")
        for p in order:
            g = (u - agg(p, regA)[0]) / (u - o) if abs(u - o) > 1e-9 else float("nan")
            print(f"      {p:>16s}  {g:+.2f}")

    # ---------------- 2. split quality ----------------
    print(f"\n{'='*100}\nTABLE 2 -- SPLIT QUALITY (does the head read the three-way split?)\n"
          f"  rho(pred_red, realised lprog@t+1) and rho(pred_rel, realised visits@t+1), per region\n"
          f"  per round; AUROC = the head's own reducibility prediction separating REDUCIBLE from\n"
          f"  NOISE regions. 0.5 = blind. Measured on every arm: a shadow head runs everywhere, so\n"
          f"  a head under `value`'s allocation is the split read at a FIXED, known-good policy.\n{'='*100}")
    print(f"{'policy':>16s} {'rho(red,lprog)':>15s} {'rho(rel,visits)':>16s} {'AUROC red:r-v-n':>16s} "
          f"{'pred_red(A)':>12s} {'pred_red(noise)':>16s}")
    for p in order:
        sq = [r["split_quality"].get(p) for r, _ in pol_runs[p]]
        sq = [s for s in sq if s]
        if not sq:
            print(f"{p:>16s} {'--':>15s}"); continue
        c1, _ = _mstd([s["rho_red_lprog"] for s in sq])
        c2, _ = _mstd([s["rho_rel_visits"] for s in sq])
        au, _ = _mstd([s["auroc_red"] for s in sq])
        pA, pN = [], []
        for r, h in pol_runs[p]:
            for x in h:
                if "pred_red" in x:
                    pA.append(x["pred_red"][a_idx]); pN += [x["pred_red"][j] for j in noise_j]
        mA, _ = _mstd(pA); mN, _ = _mstd(pN)
        print(f"{p:>16s} {c1:15.3f} {c2:16.3f} {au:16.3f} {mA:12.4f} {mN:16.4f}")

    # ---------------- 3. ablations ----------------
    abl = [p for p in pol_runs if p.startswith("head_sup")]
    if len(abl) > 1:
        print(f"\n{'='*100}\nTABLE 3 -- ABLATIONS: which reader carries the split\n"
              f"  Δ is against the full head (`head_sup`). A channel is load-bearing if dropping it\n"
              f"  hurts A-err OR collapses the AUROC. `-lp` is the honest ceiling: if the head only\n"
              f"  works when handed the survey's own counterfactual, it adds nothing beyond it.\n{'='*100}")
        base = agg("head_sup", regA)[0] if "head_sup" in pol_runs else float("nan")
        print(f"{'arm':>18s} {'A-err ↓':>9s} {'Δ vs full':>10s} {'AUROC':>8s} {'→noise':>8s}")
        for p in sorted(abl):
            m, _ = agg(p, regA)
            sq = [r["split_quality"].get(p) for r, _ in pol_runs[p]]
            au, _ = _mstd([s["auroc_red"] for s in sq if s])
            sn, _ = agg(p, share(noise_j))
            print(f"{p:>18s} {m:9.4f} {m - base:+10.4f} {au:8.3f} {sn:8.2f}")

    # ---------------- 4. Phase A ----------------
    print(f"\n{'='*100}\nTABLE 4 -- PHASE A: raw-channel separability, committee health, benchmark\n{'='*100}")
    chans = ["e", "d", "b-e", "g", "v", "lprog", "excess*"]
    print("  (a) AUROC of each RAW channel, pooled over rounds. Read this BEFORE any head result:")
    print(f"      {'policy':>16s} " + " ".join(f"{c:>8s}" for c in chans) + "     <- reducible vs noise")
    for p in order:
        vals = []
        for c in chans:
            v, _ = _mstd([r["separability"][p][c]["red_vs_noise"] for r, _ in pol_runs[p]
                          if p in r["separability"]])
            vals.append(v)
        print(f"      {p:>16s} " + " ".join(f"{v:8.3f}" for v in vals))
    print(f"      {'policy':>16s} " + " ".join(f"{c:>8s}" for c in chans) + "     <- A vs rest")
    for p in order:
        vals = []
        for c in chans:
            v, _ = _mstd([r["separability"][p][c]["A_vs_rest"] for r, _ in pol_runs[p]
                          if p in r["separability"]])
            vals.append(v)
        print(f"      {p:>16s} " + " ".join(f"{v:8.3f}" for v in vals))

    print("\n  (b) committee health + benchmark estimability "
          "(ens_cos high = members leave the SAME residual, i.e. not a committee):")
    print(f"      {'policy':>16s} {'ens_cos':>8s} {'d(replay)':>10s} {'d(red)':>10s} {'d(noise)':>10s} "
          f"{'b-e(red)':>10s} {'b-e(noise)':>11s}")
    for p in order:
        H = [r["committee_health"][p] for r, _ in pol_runs[p] if p in r["committee_health"]]
        if not H:
            continue
        f = lambda k1, k2=None: _mstd([(h[k1][k2] if k2 else h[k1]) for h in H])[0]
        print(f"      {p:>16s} {f('ens_cos_mean'):8.3f} {f('var_replay_mean'):10.2e} "
              f"{f('d_by_class','reducible'):10.2e} {f('d_by_class','noise'):10.2e} "
              f"{f('bme_by_class','reducible'):10.4f} {f('bme_by_class','noise'):11.4f}")

    inv = [r["invariance_gate"] for r in runs if r.get("invariance_gate")]
    if inv:
        print("\n  (c) allocation-size invariance gate -- are allocation and YIELD separable?\n"
              "      in-region share vs allocation size, under both denominations. Episode must be\n"
              "      FLAT (every size is a whole number of completed reaches); transition rises\n"
              "      with size, which is the confound.")
        for mode in ("episode", "transition"):
            rows = [g[mode] for g in inv if mode in g]
            if not rows:
                continue
            sizes = [k for k in rows[0] if k.isdigit()]
            sh = [_mstd([r[k]["in_share"] for r in rows])[0] for k in sizes]
            sp, _ = _mstd([r["spread"] for r in rows])
            rho, _ = _mstd([r.get("rho_size_yield", float("nan")) for r in rows])
            print(f"      {mode:>11s}  sizes {sizes} -> in-share "
                  + " ".join(f"{x:.3f}" for x in sh)
                  + f"   rho(size,yield) {rho:+.3f}  spread {sp:.3f}"
                  # the verdict is the TREND. A max-minus-min spread confuses per-reach
                  # sampling scatter with a size effect, which is what the first cut of this
                  # gate did; |rho| is the criterion the run itself applies.
                  + ("   FLAT" if abs(rho) < 0.35 else "   SIZE-DEPENDENT"))

    print("\n  (d) heterogeneous_graders §9 discriminator -- does disagreement CLOSE when you\n"
          "      collect there? (negative rho = it closes; the named separator between a genuine\n"
          "      frontier and a noisy TV, pooled over all arms in each run)")
    for cl in ("reducible", "noise"):
        rho, _ = _mstd([r["hetero_discriminator"][cl]["rho_alloc_dDisagree"] for r in runs])
        slp, _ = _mstd([r["hetero_discriminator"][cl]["slope"] for r in runs])
        print(f"      {cl:>10s}: rho={rho:+.3f}  slope={slp:+.3e}")

    # ---------------- figures ----------------
    outdir = a.out or os.path.join(_HERE, "figures", "agg_" + "_".join(a.tags))
    os.makedirs(outdir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[agg] matplotlib unavailable; tables only")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    cmap = plt.get_cmap("tab10")
    for i, p in enumerate(order):
        r, h = pol_runs[p][0]
        rr = range(len(h))
        axes[0].plot(rr, [x["reg_err"][a_idx] for x in h], label=p, color=cmap(i % 10), lw=1.6)
        axes[1].plot(rr, [x["ballistic_cem"] for x in h], label=p, color=cmap(i % 10), lw=1.6)
    axes[0].axhline(ref["ceil_err"][a_idx], ls="--", c="k", lw=1, label="matched-FM ceiling")
    axes[0].set_title("region-A FM error (the sighted grader)"); axes[0].set_xlabel("round")
    axes[1].set_title("ballistic reach error (behavioural cash-out)"); axes[1].set_xlabel("round")
    axes[0].legend(fontsize=6, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_ladder.png"), dpi=140); plt.close()

    # SPEC Phase A#4: the per-region signature over rounds, by region class, read-only
    pa = "uniform" if "uniform" in pol_runs else order[0]
    r, h = pol_runs[pa][0]
    keys = [("per_err_mon", "committee error e"), ("disagree", "disagreement d"),
            ("bme", "benchmarked error b(s)−e"), ("agap", "agency gap g"),
            ("visits", "plan occupancy v"), ("lprog", "survey lprog (label)")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True)
    for ax, (k, ttl) in zip(axes.ravel(), keys):
        seen = set()
        for j in range(K):
            c = _cls(r, j)
            ax.plot(range(len(h)), [x[k][j] for x in h], color=CLASS_COL[c], lw=1.4, alpha=0.85,
                    label=c if c not in seen else None)
            seen.add(c)
        ax.set_title(ttl, fontsize=10); ax.axhline(0, c="k", lw=0.5, alpha=0.4)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Phase A#4 -- per-region signature over rounds, by region class "
                 f"(arm: {pa}; red=A, blue=off-reach reducible, grey=noise)", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fig_signature.png"), dpi=140); plt.close()
    print(f"\n[agg] figures -> {outdir}")


if __name__ == "__main__":
    main()
