"""Reduce a `merge` run.

    python3 rhm/practice/merge/analyze_merge.py --tag mg_s0 --fetch --figures

Readouts, in the spec's priority order:

  0. THE GATES -- the confound must exist (I(venue; latent) > 0), paced rotation must pay it
     down (cumulative MI falls while instantaneous does not), the world must not get harder
     (d0, on-grammar, DP floor), and the fifth wall must be admissible (its demanded level-2
     mass has to be covered by the training nodes' support, or nothing could transfer).
  1. IN-DISTRIBUTION PARITY -- E1 (own turf, own demand) and E2 (the varied-set envelope,
     uniform demand = the unmetered arm's home distribution). The unmetered arm should match or
     beat everything here; if it does not, the setup is miscalibrated.
  2. THE FIFTH WALL -- E3 (held-out node x held-out venue: a genuine cache miss for a keyed
     index) and E3b (held-out node, training venue). Read as the TRANSFER COST E3 - E2, which
     nets out how good the arm is in the first place, plus the cache-miss range for keyed arms.
  3. NEXT-LEVEL MINABILITY -- |T2| -> |T3| under the ratchet's nesting, per reading of "the
     arm's representation": its fallback cell, the union of its cells (the generous bound), and
     a post-hoc mine of its own solved repairs (the only reading available to a dense arm).
  4. SEC-8 INSTRUMENTS -- the scaffold's early value (keyed vs unkeyed over the first cycles),
     key granularity over time, the never-merge arm's carried storage, and the merge log.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_merge"
ORDER = ["never_base", "given", "global", "track", "merge", "merge_pool", "dense", "dense_wd",
         "dense_glob"]


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True, env=dict(os.environ,
                                        MODAL_PROFILE=os.environ.get("MODAL_PROFILE",
                                                                     "chromatic")))


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if os.path.isfile(p):
            arms[arm] = json.load(open(p))
    keyed = {a: arms[a] for a in ORDER if a in arms}
    keyed.update({a: r for a, r in arms.items() if a not in keyed})
    return setup, keyed


def _f(x, n=4, w=7):
    return " " * (w - 1) + "." if x is None else f"{x:{w}.{n}f}"


def ranks(d):
    """Rank ordering (1 = best = lowest error). The arc's currency: absolute levels do not
    carry across rule draws, orderings do."""
    items = sorted(((v, k) for k, v in d.items() if v is not None))
    return {k: i + 1 for i, (_v, k) in enumerate(items)}


# --------------------------------------------------------------------------- #

def report(setup, arms):
    cfg, gate, refs = setup["config"], setup["gate"], setup["refs"]
    print("=" * 100)
    print(f"merge  --  {cfg['n_venue']} venues (+1 held out) x nodes {cfg['train_nodes']} "
          f"(+{cfg['hold_node']} held out) = {len(setup['keys'])} training keys")
    print(f"sigma={cfg['sigma']} kappa={cfg['kappa']} rotate every {cfg['rotate_every']} "
          f"from c{cfg['rotate_start']}; {cfg['cycles']} cycles; dense_mult={cfg['dense_mult']}; "
          f"merge tau={cfg['merge_tau']} every {cfg['merge_every']} from c{cfg['merge_start']}")
    print("=" * 100)

    print("\n0. GATES")
    print(f"  M1 fidelity (sigma=0 sampler == the parent's): d0 {gate['M1']['d0_parent']:.3f} "
          f"vs {gate['M1']['d0_demand']:.3f}, on-grammar {gate['M1']['og_parent']:.3f}/"
          f"{gate['M1']['og_demand']:.3f}, hist KL {gate['M1']['hist_kl']:.4f}  "
          f"-> {'PASS' if gate['M1']['pass'] else 'FAIL'}")
    print(f"  M2 nothing becomes false: true tables identical={gate['M2']['truth_identical']}, "
          f"d0 spread {gate['M2']['d0_spread']:.3f}, on-grammar min "
          f"{gate['M2']['on_grammar_min']:.3f}  -> "
          f"{'PASS' if gate['M2']['pass'] else 'FAIL'}")
    print(f"  M3 the confound exists: I(venue;latent) = {gate['M3']['i_sigma']:.4f} nats "
          f"(sigma=0 control {gate['M3']['i_zero']:.4f})  -> "
          f"{'PASS' if gate['M3']['pass'] else 'FAIL'}")
    print(f"  M4 rotation decorrelates: I_inst {[round(x, 3) for x in gate['M4']['i_inst']]}")
    print(f"                            I_cum  {[round(x, 3) for x in gate['M4']['i_cum']]}"
          f"  -> {'PASS' if gate['M4']['pass'] else 'FAIL'}")
    print(f"  M5 fifth wall admissible: {gate['M5']['holdout_mass_covered_by_train_support']:.3f}"
          f" of the held-out node's demanded mass is inside the training nodes' support "
          f"-> {'PASS' if gate['M5']['pass'] else 'FAIL'}")
    print(f"  M6 merge is index-only -> {'PASS' if gate['M6']['pass'] else 'FAIL'}")
    print(f"  refs: d0={refs['d0']:.3f} e0={refs['e0']:.3f} floor={refs['floor']:.3f} "
          f"on_grammar={refs['on_grammar']:.3f}")

    ex = {a: r["exam"] for a, r in arms.items()}
    e1 = {a: x["E1"]["e"] for a, x in ex.items()}
    e2 = {a: x["E2"]["e"] for a, x in ex.items()}
    e3 = {a: x["E3"]["e"] for a, x in ex.items()}
    e3b = {a: x["E3b"]["e"] for a, x in ex.items()}
    r1, r2, r3 = ranks(e1), ranks(e2), ranks(e3)

    print("\n1-2. THE EXAMS  (error; lower is better; rank in brackets)")
    print(f"  {'arm':<12} {'E1 home':>10} {'E2 envel':>10} {'E3 wall5':>10} "
          f"{'E3b node':>10} | {'E3-E2':>8} {'E3b-E2':>8} | {'miss':>5} {'cells':>6}")
    for a in arms:
        x = ex[a]
        miss = x["E3"].get("n_miss", 0)
        cells = x.get("n_cells", 0)
        print(f"  {a:<12} {e1[a]:10.4f}[{r1[a]}] {e2[a]:10.4f}[{r2[a]}] "
              f"{e3[a]:10.4f}[{r3[a]}] {e3b[a]:10.4f} | {e3[a] - e2[a]:8.4f} "
              f"{e3b[a] - e2[a]:8.4f} | {miss:5d} {cells:6d}")
    best_metered = min((v for a, v in e2.items() if not a.startswith("dense")), default=None)
    for a in arms:
        if a.startswith("dense") and best_metered is not None:
            verdict = "PASS" if e2[a] <= best_metered + 1e-9 else "FAIL"
            print(f"  [parity gate] {a} E2={e2[a]:.4f} vs best metered {best_metered:.4f} "
                  f"-> {verdict}")

    print("\n  cache-miss range on E3 (keyed arms: what the best / worst available cell scores)")
    for a in arms:
        x = ex[a]["E3"]
        if "e_best_cell" in x:
            print(f"    {a:<12} fallback {x['e']:.4f}   best cell {x['e_best_cell']:.4f}   "
                  f"worst cell {x['e_worst_cell']:.4f}")

    print("\n3. NEXT-LEVEL MINABILITY  (T[3] built over T[2]; the ratchet's nesting)")
    print("   |T3| own    = mined from the spans THIS arm solved and observed (realistic, but "
          "conflates representability with how much it solved)")
    print("   |T3| shared = the SAME oracle observation set for every arm, so only the lower "
          "table varies -- a pure statement about the nesting constraint")
    print(f"  {'arm':<12} {'reading':<10} {'|T2|':>5} {'|T3|own':>8} {'rec':>6} "
          f"{'|T3|shr':>8} {'recS':>6} {'e(T3own)':>9} {'e(T3shr)':>9} {'e(trueT3)':>10} "
          f"{'e L3solve':>10}")
    for a in arms:
        for name, rec in ex[a]["minability"].items():
            print(f"  {a:<12} {name:<10} {rec['n_t2']:5d} {rec['n_t3']:8d} "
                  f"{rec['t3_recall']:6.3f} {rec.get('n_t3_shared', -1):8d} "
                  f"{rec.get('t3s_recall', float('nan')):6.3f} "
                  f"{_f(rec.get('e_t3_macro'), 4, 9)} {_f(rec.get('e_t3s_macro'), 4, 9)} "
                  f"{rec['e_t3_true']:10.4f} {rec['e_l3_solve']:10.4f}")
    print("\n  post-hoc mined T2 (the only reading a dense arm has):")
    for a in arms:
        p = arms[a]["post_mine"] if "post_mine" in arms[a] else ex[a].get("post_mine", {})
        p = arms[a]["exam"].get("post_mine", p)
        if p:
            print(f"    {a:<12} |T2|={p['n_t2']:3d} recall={p['t2_recall']:.3f} "
                  f"prec={_f(p['t2_precision'], 3, 6)} obs={p['n_obs']:5d} "
                  f"e(mine pool)={p['e_mine_pool']:.4f}")

    print("\n4. SEC-8 INSTRUMENTS")
    print("  (a) the scaffold's early value: mean metered error over the first / last cycles")
    k = max(3, cfg["cycles"] // 6)
    for a in arms:
        e = np.asarray(arms[a]["log"]["e"], float)
        print(f"    {a:<12} first{k}={e[:k].mean():.4f}  last{k}={e[-k:].mean():.4f}  "
              f"all={e.mean():.4f}")
    print("  (b) key granularity over time (cells) and carried storage (entries)")
    for a in arms:
        nc = arms[a]["log"]["n_cells"]
        st = arms[a]["log"]["storage"]
        if max(nc) == 0:
            continue
        idx = list(range(0, len(nc), max(1, len(nc) // 8))) + [len(nc) - 1]
        print(f"    {a:<12} cells " + " ".join(f"{nc[i]:3d}" for i in sorted(set(idx)))
              + "   | storage " + " ".join(f"{st[i]:4d}" for i in sorted(set(idx))))
    print("  (c) merge events")
    for a in arms:
        evs = [e for e in arms[a]["events"] if e["kind"] == "merge"]
        if evs:
            print(f"    {a:<12} " + "; ".join(
                f"c{e['cycle']}: {e['n_cells_before']}->{e['n_cells_after']}" for e in evs))
    print("  (d) alias audit (route a) vs forced transfer (route b): mean Jaccard between the "
          "surviving cells' entry sets")
    for a in arms:
        jc = ex[a].get("jaccard")
        if jc:
            vals = [x for x in jc.values() if x is not None]
            if vals:
                print(f"    {a:<12} mean {np.mean(vals):.3f}  min {min(vals):.3f}  "
                      f"max {max(vals):.3f}  (n={len(vals)} pairs)")
    print("  (e) the per-cell tables that survive")
    for a in arms:
        cells = ex[a].get("cells")
        if not cells:
            continue
        ent = [c["n_entries"] for c in cells]
        rec = [c.get("tab_recall") for c in cells if c.get("tab_recall") is not None]
        print(f"    {a:<12} {len(cells)} cells, entries/cell mean {np.mean(ent):.1f} "
              f"(min {min(ent)} max {max(ent)}), recall vs truth mean "
              f"{np.mean(rec) if rec else float('nan'):.3f}")

    print("\n5. WALL CLOCK")
    for a in arms:
        print(f"    {a:<12} {arms[a]['elapsed']:7.0f}s  mult={arms[a]['mult']}  "
              f"wd={arms[a]['wd']}")
    print()


# --------------------------------------------------------------------------- #

def figures(tag, setup, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    cfg = setup["config"]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    # fig1: competence + key granularity
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, a in enumerate(arms):
        log = arms[a]["log"]
        ax[0].plot(log["cycle"], log["e"], label=a, color=cols[i % 10], lw=1.4)
        if max(log["n_cells"]) > 0:
            ax[1].plot(log["cycle"], log["n_cells"], label=a, color=cols[i % 10], lw=1.6)
    for x in range(cfg["rotate_start"], cfg["cycles"] + 1, max(1, cfg["rotate_every"])):
        ax[0].axvline(x, color="0.85", lw=0.6, zorder=0)
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("metered error (own venue)")
    ax[0].set_title("competence; grey lines = rotation events"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("cells in the index")
    ax[1].set_title("key granularity over time (merge vs track)"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_competence_granularity.png"),
                                    dpi=130); plt.close(fig)

    # fig2: the exams
    names = list(arms)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, (lab, off) in enumerate((("E1", -0.27), ("E2", -0.09), ("E3", 0.09),
                                    ("E3b", 0.27))):
        ax[0].bar(x + off, [arms[a]["exam"][lab]["e"] for a in names], width=0.17, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("error"); ax[0].legend(fontsize=8)
    ax[0].set_title("E1 own turf / E2 envelope / E3 fifth wall / E3b held-out node")
    ax[1].bar(x - 0.19, [arms[a]["exam"]["E3"]["e"] - arms[a]["exam"]["E2"]["e"]
                         for a in names], width=0.36, label="E3 - E2")
    ax[1].bar(x + 0.19, [arms[a]["exam"]["E3b"]["e"] - arms[a]["exam"]["E2"]["e"]
                         for a in names], width=0.36, label="E3b - E2")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax[1].set_ylabel("transfer cost (error)"); ax[1].legend(fontsize=8)
    ax[1].set_title("the cost of a transform outside the varied set")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_exams.png"), dpi=130); plt.close(fig)

    # fig3: next-level minability
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
    readings = sorted({r for a in names for r in arms[a]["exam"]["minability"]})
    w = 0.8 / max(1, len(readings))
    for k, rd in enumerate(readings):
        off = -0.4 + w * (k + 0.5)
        for p, field in ((0, "n_t2"), (1, "n_t3"), (2, "n_t3_shared")):
            ax[p].bar(x + off, [arms[a]["exam"]["minability"].get(rd, {}).get(field, 0)
                                for a in names], width=w, label=rd)
    for k, lab, ylab in ((0, "|T2| the reading holds", "|T2|"),
                         (1, "|T3| mined from what the arm solved", "|T3| own"),
                         (2, "|T3| over a SHARED observation set (nesting only)",
                          "|T3| shared")):
        ax[k].set_xticks(x); ax[k].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax[k].set_title(lab, fontsize=9); ax[k].set_ylabel(ylab); ax[k].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_minability.png"), dpi=130)
    plt.close(fig)

    # fig4: the last transfer matrix of each merge arm
    mats = [(a, m) for a in names
            for m in [next((x for x in reversed(arms[a]["log"]["merge"]) if x and x["S"]),
                           None)] if m]
    if mats:
        fig, ax = plt.subplots(1, len(mats), figsize=(5.5 * len(mats), 4.6), squeeze=False)
        for k, (a, m) in enumerate(mats):
            ids = sorted(m["S"], key=int)
            M = np.array([[m["S"][i][j] for j in ids] for i in ids])
            im = ax[0][k].imshow(M, vmin=0, vmax=1, cmap="viridis")
            ax[0][k].set_title(f"{a}: forced-transfer matrix\nS[a][b] = success of a's table "
                               f"on b's demand", fontsize=9)
            ax[0][k].set_xlabel("demand cell b"); ax[0][k].set_ylabel("table cell a")
            fig.colorbar(im, ax=ax[0][k])
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_transfer.png"), dpi=130)
        plt.close(fig)
    print(f"figures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    setup, arms = load(a.tag)
    report(setup, arms)
    if a.figures:
        figures(a.tag, setup, arms)


if __name__ == "__main__":
    main()
