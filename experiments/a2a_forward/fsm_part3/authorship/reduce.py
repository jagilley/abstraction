"""Reduce an `authorship_test` run to a table and figures.

Runs locally (no Modal). Fetch the artifacts first:

    cd experiments/a2a_forward/fsm_part3/authorship
    modal volume get language-reduction-data \
        /a2a_forward/confabulation/authorship/main /tmp/authorship_main
    python reduce.py --dir /tmp/authorship_main --tag main

Writes `figures/<tag>_reduction.txt` and the PNGs beside it. Reduction only: no
interpretation, no claims -- the numbers and the curves.
"""

import argparse
import json
import os

import numpy as np


CONTESTANTS = ["SELF_mlp", "SELF_seq", "O_input", "O_lik", "O_io", "O_io_nolik",
               "O_act", "LIK_pos"]
THIRD_PARTY = ("O_input", "O_lik", "O_io", "O_io_nolik", "LIK_pos")


def auc(scores, labels):
    from scipy.stats import rankdata
    pos = labels > 0.5
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def span_offset(seq, pos, lab):
    """Tokens of the current span already seen, recomputed from the saved labels.

    NB this replaces the job's saved `_offset`, which counts runs from position 0 and so
    does NOT reset at the prefix boundary: a corpus span that begins right after the
    (always-corpus) prefix inherits the prefix's run length, which makes offset partly
    predict the class. Recomputing inside the scored region -- which starts at the prefix
    boundary for every sequence -- removes that.
    """
    order = np.lexsort((pos, seq))
    out = np.zeros(len(seq), dtype=np.int64)
    run = 0
    for i, k in enumerate(order):
        if i == 0 or seq[k] != seq[order[i - 1]] or lab[k] != lab[order[i - 1]]:
            run = 0
        else:
            run += 1
        out[k] = run
    return out


def best(obs, prefix, exclude_half=True):
    v = [x for k, x in obs.items()
         if k.startswith(prefix) and not (exclude_half and "half" in k)]
    return max(v) if v else float("nan")


def best_third_party(obs):
    v = [x for k, x in obs.items()
         if k.startswith(THIRD_PARTY) and "half" not in k]
    return max(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding results.json / predictions.npz")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.dir, "results.json")) as f:
        res = json.load(f)
    npz_path = os.path.join(a.dir, "predictions.npz")
    P = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}
    cfg, sets = res.get("config", {}), res["sets"]

    L = []
    w = L.append
    w("=" * 100)
    w(f"AUTHORSHIP  --  reduction of tag '{a.tag}'")
    w("=" * 100)
    w(f"  M: {cfg.get('n_layer')}L/{cfg.get('n_head')}H/{cfg.get('n_embd')}D  "
      f"arm={cfg.get('cond')}  {cfg.get('n_steps')} wake steps  "
      f"val(standalone)={res.get('val_standalone', float('nan')):.4f}")
    w(f"  report site {cfg.get('report_block')}   N={cfg.get('n_report_sequences')} seqs"
      f"   prefix={cfg.get('prefix_len')}   spans={cfg.get('span_lens')}"
      f"   observers={cfg.get('observer_caps')} ({cfg.get('obs_steps')} steps)")
    w(f"  seed={cfg.get('seed')}  instrument={cfg.get('inst_cap')}  "
      f"ens_n={cfg.get('ens_n')}  impl_k={cfg.get('impl_k')}")

    # ---------------- descriptive: what the public signal looks like ------------
    w("")
    w("-" * 100)
    w("SET DESCRIPTIVES  (the public signal, before anyone is trained)")
    w("-" * 100)
    w(f"  {'set':14s} {'mode':6s} {'temp':>5s} {'chance':>7s} {'mine_frac':>10s} "
      f"{'logp(mine)':>11s} {'logp(corpus)':>13s} {'AUC(surprisal)':>15s} {'topk_mass':>10s}")
    for s, r in sets.items():
        w(f"  {s:14s} {r['mode']:6s} {r['temp']:5.2f} {r['chance']:7.3f} "
          f"{r['mine_frac_test']:10.3f} {r['logp_mine']:11.3f} {r['logp_corpus']:13.3f} "
          f"{r['auc_raw_surprisal']:15.3f} {r['topk_mass']:10.3f}")

    # ---------------- the battery ----------------
    for s, r in sets.items():
        w("")
        w("=" * 100)
        w(f"SET {s}   (mode={r['mode']}  temperature={r['temp']})")
        w("=" * 100)
        for tn in ("MINE", "IMPL", "IMPL_COS"):
            if tn not in r["report"]:
                continue
            obs, rep = r["observers"][tn], r["report"][tn]
            chance = r["baselines"].get(tn, float("nan"))
            sm, ss = rep["full"], best(obs, "SELF_seq")
            bself = max(sm, ss) if ss == ss else sm
            bio, b3p, oact = best(obs, "O_io@"), best_third_party(obs), best(obs, "O_act")
            w("")
            w(f"  --- target {tn}   (chance {chance:.3f}) ---")
            w(f"      SELF_mlp (per-position MLP on the report site) = {sm:.3f}")
            for k in sorted(obs):
                if k.startswith("SELF_seq"):
                    w(f"      SELF_seq {k.split('@')[1]:>8s}                       = {obs[k]:.3f}")
            w(f"      {'rung':22s} " + " ".join(f"{c[0]}L{c[1]}D".rjust(9)
                                                for c in _caps(cfg))
              + "     (blank = not run at that capacity)")
            for pre, lbl in (("O_input@", "O_input  (tokens)"),
                             ("O_lik@", "O_lik    (likelihood only)"),
                             ("O_io@", "O_io     (tokens+dist+lik)")):
                cells = []
                for c in _caps(cfg):
                    k = f"{pre}{c[0]}L{c[1]}D"
                    cells.append(f"{obs[k]:9.3f}" if k in obs else " " * 9)
                w(f"      {lbl:22s} " + " ".join(cells))
            for pre, lbl in (("O_io_nolik@", "O_io_nolik (parent's)"),
                             ("O_act@", "O_act    (ceiling)")):
                v = best(obs, pre)
                if v == v:
                    w(f"      {lbl:22s} {v:.3f}   [top capacity only]")
            hd = [v for k, v in obs.items() if "half" in k]
            if hd:
                w(f"      {'O_io half data':22s} {hd[0]:.3f}")
            if "LIK_pos" in obs:
                w(f"      {'LIK_pos (no aggreg.)':22s} {obs['LIK_pos']:.3f}")
            w(f"      >> best self = {bself:.3f}   best O_io = {bio:.3f}   "
              f"best third party (any rung) = {b3p:.3f}   O_act = {oact:.3f}")
            w(f"      >> ADVANTAGE vs O_io = {bself - bio:+.3f}"
              f"      ADVANTAGE vs best third party = {bself - b3p:+.3f}")
            abl = {k[5:]: v for k, v in rep.items() if k.startswith("eval_")}
            if abl:
                w(f"      ablations (SELF_mlp): full={rep['full']:.3f}  "
                  + "  ".join(f"{k}={v:.3f}" for k, v in abl.items())
                  + (f"   confab={rep['confab']:.3f} "
                     f"margin={rep['full'] - rep['confab']:+.3f}"
                     if "confab" in rep else ""))
            if "auc" in rep:
                w(f"      SELF_mlp AUC = {rep['auc']:.3f}")
        if "auc" in r:
            w("")
            w("  AUC (position-level, test split):  "
              + "  ".join(f"{k}={v:.3f}" for k, v in r["auc"].items()))
        if "instrument" in r:
            i = r["instrument"]
            rs = i["residual_structure"]
            pub = {"1": -0.85, "2": -0.62, "3": +0.62, "4": +0.84}
            nm = {"1": "sent_start", "2": "after_punct", "3": "after_opener",
                  "4": "before_closer"}
            w("")
            w(f"  INSTRUMENT GUARDS  ({i['cap']}, {i['fm_params']/1e3:.0f}K = "
              f"{i['pct_of_predicted']:.1f}% of the predicted blocks)")
            w(f"    fwd_cosine={i['fwd_cosine']:.4f}  |r|={i['res_norm']:.3f}  "
              f"ens_cos={i['ens_cos']:.3f}   <-- ens_cos is the junk-residual gate")
            w(f"    eta2_norm={rs['eta2_norm']:.4f}  eta2_dir={rs['eta2_dir']:.4f}  "
              f"eta2_vec={rs['eta2_vec']:.4f}")
            w("    |r| Cohen's d vs the published table:  "
              + "  ".join(f"{nm[k]}={v:+.2f}(pub {pub[k]:+.2f})"
                          for k, v in sorted(rs["cohens_d"].items()) if k in pub))
            if "cluster_quality" in r:
                cq = r["cluster_quality"]
                w(f"    residual cluster quality: within={cq['within']:.3f} "
                  f"between={cq['between']:.3f} separation={cq['separation']:.3f}")

    # ---------------- from the saved predictions ----------------
    if P:
        w("")
        w("=" * 100)
        w("FROM THE SAVED PER-POSITION PREDICTIONS")
        w("=" * 100)
        for s in sets:
            lab = P.get(f"{s}|_label")
            if lab is None:
                continue
            seq, pos = P[f"{s}|_seq"], P[f"{s}|_pos"]
            off = span_offset(seq, pos, lab)
            names = [c for c in CONTESTANTS if f"{s}|{c}" in P]
            # group = the whole sequence when it has one author, else the contiguous span
            whole = sets[s]["mode"] == "whole"
            grp = seq if whole else (np.cumsum(np.concatenate(
                [[1], (off[np.lexsort((pos, seq))][1:] == 0).astype(int)]))
                [np.argsort(np.lexsort((pos, seq)))])
            w("")
            w(f"  {s}:  {'sequence' if whole else 'span'}-level AUC "
              f"(mean predicted probability per {'sequence' if whole else 'span'})")
            ug = np.unique(grp)
            for c in names:
                sc = P[f"{s}|{c}"]
                m = np.array([sc[grp == u].mean() for u in ug])
                y = np.array([lab[grp == u].mean() > 0.5 for u in ug]).astype(float)
                w(f"      {c:10s} group-AUC={auc(m, y):.3f}   pos-AUC={auc(sc, lab):.3f}"
                  f"   (n_groups={len(ug)})")
            # accuracy against how much of the span has been seen
            bins = [(0, 0), (1, 3), (4, 7), (8, 15), (16, 31), (32, 200)]
            w(f"  {s}:  position-level AUC by offset within span "
              f"(0 = first token of a span)")
            w("      " + f"{'contestant':11s}" + "".join(
                f"{('%d-%d' % b) if b[0] != b[1] else str(b[0]):>9s}" for b in bins)
              + f"{'n':>9s}")
            counts = ["" for _ in bins]
            for c in names:
                sc, cells = P[f"{s}|{c}"], []
                for k, b in enumerate(bins):
                    m = (off >= b[0]) & (off <= b[1])
                    v = auc(sc[m], lab[m]) if m.sum() > 50 else float("nan")
                    cells.append(f"{v:9.3f}" if v == v else " " * 9)
                    counts[k] = str(int(m.sum()))
                w(f"      {c:11s}" + "".join(cells))
            w(f"      {'n':11s}" + "".join(f"{x:>9s}" for x in counts))

    txt = "\n".join(L)
    with open(os.path.join(a.out, f"{a.tag}_reduction.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    _figures(res, P, a.out, a.tag)
    print(f"\nwrote {a.out}/{a.tag}_reduction.txt and figures")


def _caps(cfg):
    return [tuple(int(z) for z in c.split(":"))
            for c in cfg.get("observer_caps", "1:64,2:128,4:256").split(",")]


def _figures(res, P, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = res["sets"]
    C = {"SELF_mlp": "#1b3b6f", "SELF_seq": "#2f6fb5", "O_input": "#9aa5b1",
         "O_lik": "#c0392b", "O_io": "#e07b39", "O_io_nolik": "#e8c07d",
         "O_act": "#5b8c5a", "LIK_pos": "#8e5aa8"}

    # --- fig 1: the battery, one panel per set ---------------------------------
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.6), squeeze=False)
    for ax, (s, r) in zip(axes[0], sets.items()):
        obs, rep = r["observers"]["MINE"], r["report"]["MINE"]
        vals, names = [], []
        for c in CONTESTANTS:
            v = (rep["full"] if c == "SELF_mlp"
                 else obs["LIK_pos"] if c == "LIK_pos"
                 else best(obs, c + "@"))
            if v == v:
                names.append(c); vals.append(v)
        ax.bar(range(len(vals)), vals,
               color=[C[n] for n in names], edgecolor="black", linewidth=.5)
        ax.axhline(r["chance"], color="k", ls="--", lw=1, label=f"chance {r['chance']:.2f}")
        for i, v in enumerate(vals):
            ax.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
        ax.set_ylim(min(vals + [r["chance"]]) - .04, max(vals) + .05)
        ax.set_title(f"{s}\n(mode={r['mode']}, T={r['temp']})", fontsize=10)
        ax.set_ylabel("accuracy on MINE")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Who can tell which tokens M wrote?  (accuracy, held-out positions)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/{tag}_battery.png", dpi=160)
    plt.close(fig)

    # --- fig 2: MINE vs IMPL, the dissociation on the same positions ------------
    main = next((s for s, r in sets.items() if "IMPL" in r["report"]), None)
    if main:
        r = sets[main]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, tn in zip(axes, ("MINE", "IMPL", "IMPL_COS")):
            if tn not in r["report"]:
                ax.axis("off"); continue
            obs, rep = r["observers"][tn], r["report"][tn]
            ss = best(obs, "SELF_seq")
            rows = [("SELF_mlp", rep["full"]), ("SELF_seq", ss),
                    ("O_input", best(obs, "O_input@")), ("O_lik", best(obs, "O_lik@")),
                    ("O_io", best(obs, "O_io@")),
                    ("O_io_nolik", best(obs, "O_io_nolik@")),
                    ("O_act", best(obs, "O_act"))]
            rows = [(n, v) for n, v in rows if v == v]
            ax.barh(range(len(rows)), [v for _, v in rows],
                    color=[C[n] for n, _ in rows], edgecolor="black", linewidth=.5)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([n for n, _ in rows], fontsize=8)
            ax.invert_yaxis()
            ch = r["baselines"].get(tn, 0.0)
            if ch:
                ax.axvline(ch, color="k", ls="--", lw=1)
            for i, (_, v) in enumerate(rows):
                ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=7.5)
            ax.set_title(f"{tn}   (chance {ch:.3f})", fontsize=10)
            ax.set_xlabel("accuracy" if tn != "IMPL_COS" else "test cosine")
        fig.suptitle(f"Same model, same sequences, same positions ({main}): "
                     "an implementation fact and an authorship fact", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{out}/{tag}_dissociation.png", dpi=160)
        plt.close(fig)

    if not P:
        return

    # --- fig 3: AUC vs offset within span --------------------------------------
    bins = [(0, 0), (1, 3), (4, 7), (8, 15), (16, 31), (32, 200)]
    xs = [0, 2, 5.5, 11.5, 23.5, 45]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 4.2), squeeze=False)
    for ax, (s, r) in zip(axes[0], sets.items()):
        lab = P.get(f"{s}|_label")
        if lab is None:
            ax.axis("off"); continue
        off = span_offset(P[f"{s}|_seq"], P[f"{s}|_pos"], lab)
        for c in CONTESTANTS:
            k = f"{s}|{c}"
            if k not in P:
                continue
            ys, xx = [], []
            for b, x in zip(bins, xs):
                m = (off >= b[0]) & (off <= b[1])
                if m.sum() > 50:
                    ys.append(auc(P[k][m], lab[m])); xx.append(x)
            ax.plot(xx, ys, "o-", color=C[c], label=c, lw=1.6, ms=4)
        ax.axhline(.5, color="k", ls="--", lw=1)
        ax.set_xlabel("tokens of the span already seen")
        ax.set_ylabel("AUC on MINE")
        ax.set_title(f"{s}", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("How much of a span does each contestant need?", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/{tag}_by_offset.png", dpi=160)
    plt.close(fig)

    # --- fig 4: log p under M, by class ----------------------------------------
    fig, axes = plt.subplots(1, len(sets), figsize=(4.3 * len(sets), 3.6), squeeze=False)
    for ax, (s, r) in zip(axes[0], sets.items()):
        lab, lp = P.get(f"{s}|_label"), P.get(f"{s}|_logp")
        if lp is None:
            ax.axis("off"); continue
        bs = np.linspace(np.percentile(lp, .5), 0, 60)
        ax.hist(lp[lab > .5], bins=bs, alpha=.55, density=True, label="M wrote it",
                color="#c0392b")
        ax.hist(lp[lab < .5], bins=bs, alpha=.55, density=True, label="corpus",
                color="#1b3b6f")
        ax.set_title(f"{s}  (AUC {r['auc_raw_surprisal']:.3f})", fontsize=10)
        ax.set_xlabel(r"$\log p_M(x_t \mid x_{<t})$")
        ax.legend(fontsize=8)
    fig.suptitle("The public quantity: M's own log-probability of the token, by author",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}/{tag}_logp.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
