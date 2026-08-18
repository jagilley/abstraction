"""Reduce a `selfplay` run.

    python3 rhm/practice/merge/analyze_selfplay.py --tag sp_s0 --fetch --figures

The question: **is free data free coverage?** The world is uniform demand -- free i.i.d.
variation -- and the only difference between arms is which of the instances the world freely
offers they choose to practise on. Readouts:

  0. GATES. The inherited world gates, the entry-frequency and difficulty profiles, and the two
     mirror gates: **X-2 inclusion** (the exogenous full-support arm must realize a support
     covering every entry -- the re-run of `mg_s0`'s M-5) and **X-1 exclusion** (the self-play
     arm must leave entries its realized support genuinely starves, or there is no hole).
  1. THE THREE METRICS. Every arm's exam error under (a) its OWN footprint, (b) the WORLD's
     i.i.d. distribution -- which is what any test set drawn from the same generator measures --
     and (c) FLAT over entries, which is coverage. `own - world` is self-grading inflation;
     `world - flat` is the part of a hole an i.i.d. benchmark structurally cannot see, because
     the starved entries are the rare ones.
  2. THE HOLES, per entry, as excess over `given` (which holds the DGP's own table, sees every
     entry, and therefore measures intrinsic difficulty directly). Split head vs tail by world
     mass.
  3. SELF-SELECTED vs EXOGENOUS. corr(s0, visitation) and corr(s0, excess), plus the same
     correlations after regressing out log world mass -- because world frequency drives both
     and is the confound that has to be partialled out before "holes at the policy's own
     weaknesses" means anything.
  4. NEXT-LEVEL MINABILITY, carried over from `mg_s0`.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures_sp")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_selfplay"
ORDER = ["never_base", "given", "metered_glob", "dense_full", "dense_visit", "dense_selfplay",
         "dense_sp_hard", "dense_narrow", "dense_sp_nolib"]


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True, env=dict(os.environ,
                                        MODAL_PROFILE=os.environ.get("MODAL_PROFILE",
                                                                     "chromatic")))


def _load_dir(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for a in sorted(os.listdir(root)):
        p = os.path.join(root, a, "results.json")
        if os.path.isfile(p):
            arms[a] = json.load(open(p))
    return setup, arms


def _same_world(base, other, btag, otag):
    """A later launch that adds ARMS to an existing round rebuilds the world and the setup
    policy from the same seeds rather than overwriting the first tag. That is only a legitimate
    overlay if the rebuild is deterministic, so state the check instead of assuming it: same
    entry list, same world masses, same `s0` (the shared setup policy's per-entry error)."""
    ok = base["entries"] == other["entries"]
    dfr = max(abs(base["freq"][e] - other["freq"][e]) for e in base["entries"]) if ok else -1
    ds0 = max(abs(base["s0"][e] - other["s0"][e]) for e in base["entries"]) if ok else -1
    keys = ("cycles", "dense_mult", "n_pr", "n_ex", "n_probe", "pool_per_node", "sp_beta",
            "sp_eps", "sp_oversample", "seed", "train_seed", "rule_seed", "demand_seed")
    dcfg = [k for k in keys if base["config"].get(k) != other["config"].get(k)]
    print(f"  overlay {otag} onto {btag}: entries {'MATCH' if ok else 'DIFFER'}; "
          f"max |dfreq| {dfr:.2e}; max |ds0| {ds0:.4f}; config diffs {dcfg or 'none'}")
    if not ok or dcfg or ds0 > 1e-9:
        print(f"  !! the overlaid run's world is NOT bit-identical to {btag}'s -- cross-tag "
              f"per-entry comparisons carry that as an extra source of variance")
    return ok


def load(tag, extra=()):
    setup, arms = _load_dir(tag)
    for t in extra:
        s2, a2 = _load_dir(t)
        _same_world(setup, s2, tag, t)
        arms.update({k: v for k, v in a2.items() if k not in arms})
    keyed = {a: arms[a] for a in ORDER if a in arms}
    keyed.update({a: r for a, r in arms.items() if a not in keyed})
    return setup, keyed


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _partial(a, b, z):
    """corr(a, b) with z regressed out of both -- here z = log world mass, the common driver."""
    a, b, z = (np.asarray(x, float) for x in (a, b, z))
    Z = np.vstack([np.ones_like(z), z]).T
    ra = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    rb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    return _corr(ra, rb)


def report(setup, arms):
    cfg = setup["config"]
    gate = setup["gate"]
    ents = setup["entries"]
    E = len(ents)
    fr = np.array([setup["freq"][e] for e in ents])
    s0v = np.array([setup["s0"][e] for e in ents])
    lg = np.log(fr)

    print("=" * 112)
    print(f"selfplay -- UNIFORM demand (free i.i.d. variation); support axis = the level-2 "
          f"entry ({E} entries)")
    print(f"{cfg['cycles']} cycles, dense_mult={cfg['dense_mult']}, beta={cfg['sp_beta']} "
          f"(hard arm 2.0), eps={cfg['sp_eps']}, oversample={cfg['sp_oversample']}")
    print("=" * 112)

    print("\n0. GATES")
    print("  " + "   ".join(f"{k} -> {'PASS' if gate[k]['pass'] else 'FAIL'}"
                            for k in ("M1", "M2", "M3", "M4", "M5", "M6")))
    print(f"  world entry mass {fr.min():.4f}-{fr.max():.4f} (perplexity "
          f"{np.exp(-(fr * np.log(fr)).sum()):.2f} of {E}); exam cells "
          f"{min(setup['n_avail'].values())}-{max(setup['n_avail'].values())} instances")
    print(f"  s0 (shared setup-policy error per entry) {s0v.min():.3f}-{s0v.max():.3f}; "
          f"corr(world mass, s0) = {_corr(fr, s0v):+.3f}")
    gv = arms.get("given")
    gper = np.array([gv["per_entry"][e] for e in ents]) if gv else np.zeros(E)
    if gv:
        print(f"  `given` per-entry error (the DIFFICULTY PROFILE everything is netted "
              f"against) {gper.min():.3f}-{gper.max():.3f}; corr(world mass, given) = "
              f"{_corr(fr, gper):+.3f}")

    print("\n  X-2 INCLUSION (mirror of mg_s0's M-5) / X-1 EXCLUSION")
    print(f"  {'arm':<16} {'perplex':>8} {'unseen':>7} {'min n_seen':>11} "
          f"{'n<0.2% of draws':>16} {'held |T2|':>10} {'entries missing from T2':>24}")
    for a, r in arms.items():
        ns = np.array(r["n_seen"], float)
        q = ns / max(1.0, ns.sum())
        held = set(r.get("held_entries", []))
        miss = [e for e in ents if e not in held] if held else []
        print(f"  {a:<16} {r['perplexity']:8.2f} {r['n_zero_seen']:7d} {int(ns.min()):11d} "
              f"{int((q < 0.002).sum()):16d} {len(held):10d} "
              f"{(str(len(miss)) + ': ' + ','.join(miss[:4])):>24}")
    ff, sp = arms.get("dense_full"), arms.get("dense_selfplay")
    if ff and sp:
        nf = np.array(ff["n_seen"], float); nsz = np.array(sp["n_seen"], float)
        qs = nsz / max(1.0, nsz.sum())
        print(f"  -> X-2 {'PASS' if nf.min() > 0 else 'FAIL'} (dense_full min n_seen "
              f"{int(nf.min())});  X-1 "
              f"{'PASS' if (qs < 0.002).sum() >= 2 else 'FAIL'} (dense_selfplay starves "
              f"{(qs < 0.002).sum()} entries below 0.2% of its draws; min n_seen "
              f"{int(nsz.min())})")

    print("\n1. THE THREE METRICS  (exam error; lower better)")
    print(f"  {'arm':<16} {'own footprint':>14} {'WORLD iid':>10} {'FLAT (coverage)':>16} "
          f"{'own-world':>10} {'world-flat':>11} {'flat vs given':>14}")
    for a, r in arms.items():
        e = np.array([r["per_entry"][x] for x in ents])
        ns = np.array(r["n_seen"], float)
        q = ns / max(1.0, ns.sum())
        own, wld, flt = float((q * e).sum()), float((fr * e).sum()), float(e.mean())
        print(f"  {a:<16} {own:14.4f} {wld:10.4f} {flt:16.4f} {own - wld:+10.4f} "
              f"{wld - flt:+11.4f} {flt - gper.mean():+14.4f}")

    print("\n2. THE HOLES  (per-entry EXCESS over `given`; entries ordered by world mass, "
          "commonest first)")
    print(f"  {'entry':<10} {'mass':>7} {'s0':>6} " +
          " ".join(f"{a[:9]:>9}" for a in arms))
    order = np.argsort(-fr)
    for i in order:
        row = " ".join(f"{arms[a]['per_entry'][ents[i]] - gper[i]:+9.3f}" for a in arms)
        print(f"  {ents[i]:<10} {fr[i]:7.4f} {s0v[i]:6.3f} " + row)
    print(f"\n  head/tail split (head = entries carrying the top 80% of world mass):")
    cum = np.cumsum(fr[order]); head = set(order[cum <= 0.80]); tail = set(order) - head
    hi, ti = sorted(head), sorted(tail)
    print(f"  {'arm':<16} {'head excess':>12} {'tail excess':>12} {'tail-head':>11} "
          f"{'head n_seen%':>13} {'tail n_seen%':>13}")
    for a, r in arms.items():
        e = np.array([r["per_entry"][x] for x in ents]) - gper
        ns = np.array(r["n_seen"], float); q = ns / max(1.0, ns.sum())
        print(f"  {a:<16} {e[hi].mean():12.4f} {e[ti].mean():12.4f} "
              f"{e[ti].mean() - e[hi].mean():+11.4f} {100 * q[hi].sum():13.2f} "
              f"{100 * q[ti].sum():13.2f}")

    print("\n3. SELF-SELECTED vs EXOGENOUS HOLES")
    print("  s0_e = the SHARED setup policy's error on entry e (higher = initially WORSE there)")
    print(f"  {'arm':<16} {'corr(s0,visit)':>15} {'corr(s0,excess)':>16} "
          f"{'PARTIAL(s0,visit)':>18} {'PARTIAL(s0,excess)':>19}")
    for a, r in arms.items():
        e = np.array([r["per_entry"][x] for x in ents]) - gper
        ns = np.array(r["n_seen"], float); q = ns / max(1.0, ns.sum())
        lq = np.log(q + 1e-9)
        print(f"  {a:<16} {_corr(s0v, lq):15.3f} {_corr(s0v, e):16.3f} "
              f"{_partial(s0v, lq, lg):18.3f} {_partial(s0v, e, lg):19.3f}")
    print("  (PARTIAL columns regress out log world mass, the common driver of both. The "
          "self-play signature is a")
    print("   NEGATIVE partial corr(s0, visit) -- it practises where it was already strong "
          "beyond what the world's")
    print("   own skew explains -- and a POSITIVE partial corr(s0, excess). dense_narrow's "
          "holes are exogenous, so both ~0.)")

    print("\n4. NEXT-LEVEL MINABILITY")
    print(f"  {'arm':<16} {'reading':<8} {'|T2|':>5} {'|T3|shr':>8} {'recS':>6} "
          f"{'|T3|own':>8} {'e(T3shr)':>9} {'e(trueT3)':>10}")
    for a, r in arms.items():
        for nm, rec in r["minability"].items():
            et = rec.get("e_t3s_macro")
            print(f"  {a:<16} {nm:<8} {rec['n_t2']:5d} {rec.get('n_t3_shared', -1):8d} "
                  f"{rec.get('t3s_recall', float('nan')):6.3f} {rec['n_t3']:8d} "
                  f"{('        .' if et is None else f'{et:9.4f}')} {rec['e_t3_true']:10.4f}")

    print("\n5. WALL CLOCK")
    for a, r in arms.items():
        ht = r.get("held_table", {})
        print(f"  {a:<16} {r['elapsed']:6.0f}s  mult={r['mult']}  held |T2|="
              f"{ht.get('n_entries', 0):3d} recall={ht.get('tab_recall', float('nan')):.3f} "
              f"prec={ht.get('tab_precision')}")
    print()


def figures(tag, setup, arms, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = out or os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    ents = setup["entries"]
    E = len(ents)
    fr = np.array([setup["freq"][e] for e in ents])
    s0v = np.array([setup["s0"][e] for e in ents])
    gper = (np.array([arms["given"]["per_entry"][e] for e in ents])
            if "given" in arms else np.zeros(E))
    names = [a for a in arms if a != "given"]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    order = np.argsort(-fr)
    x = np.arange(E)

    fig, ax = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True)
    for i, a in enumerate(names):
        e = np.array([arms[a]["per_entry"][z] for z in ents]) - gper
        ax[0].plot(x, e[order], "o-", color=cols[i % 10], label=a, lw=1.4, ms=4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("excess error over `given`")
    ax[0].set_title("per-entry holes (intrinsic difficulty netted out); "
                    "entries ordered by world mass, commonest left")
    ax[0].legend(fontsize=7)
    for i, a in enumerate(names):
        ns = np.array(arms[a]["n_seen"], float)
        q = ns / max(1.0, ns.sum())
        ax[1].semilogy(x, q[order] + 1e-6, "o-", color=cols[i % 10], label=a, lw=1.4, ms=4)
    ax[1].semilogy(x, fr[order], "k--", lw=1.6, label="the world's own mass")
    ax[1].set_xticks(x); ax[1].set_xticklabels([ents[i] for i in order], rotation=60,
                                               fontsize=7)
    ax[1].set_ylabel("share of practice instances"); ax[1].legend(fontsize=7)
    ax[1].set_title("realized support: what each arm actually practised on")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_holes.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, a in enumerate(names):
        ns = np.array(arms[a]["n_seen"], float); q = ns / max(1.0, ns.sum())
        e = np.array([arms[a]["per_entry"][z] for z in ents]) - gper
        ax[0].scatter(s0v, np.log10(q + 1e-6), color=cols[i % 10], label=a, s=32)
        ax[1].scatter(s0v, e, color=cols[i % 10], label=a, s=32)
    ax[0].set_xlabel("s0: shared setup-policy error on entry e (initial weakness)")
    ax[0].set_ylabel("log10 share of practice"); ax[0].legend(fontsize=7)
    ax[0].set_title("does the arm practise where it was already strong?")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("s0: initial weakness"); ax[1].set_ylabel("final excess over `given`")
    ax[1].legend(fontsize=7)
    ax[1].set_title("do the residual holes sit at the initial weaknesses?")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_selfselection.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    nm = list(arms)
    xa = np.arange(len(nm))
    for k, (lab, f) in enumerate((("own footprint", lambda r, e: (np.array(r["n_seen"], float)
                                                                 / max(1.0, sum(r["n_seen"])) * e).sum()),
                                  ("world i.i.d.", lambda r, e: (fr * e).sum()),
                                  ("flat / coverage", lambda r, e: e.mean()))):
        vals = [f(arms[a], np.array([arms[a]["per_entry"][z] for z in ents])) for a in nm]
        ax[0].bar(xa - 0.27 + 0.27 * k, vals, width=0.25, label=lab)
    ax[0].set_xticks(xa); ax[0].set_xticklabels(nm, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("exam error"); ax[0].legend(fontsize=8)
    ax[0].set_title("the objective gap: what you measure depends on whose distribution you use")
    ax[1].bar(xa - 0.2, [max((r["minability"][k].get("n_t3_shared", 0)
                              for k in r["minability"]), default=0) for r in arms.values()],
              width=0.4, label="|T3| shared (best reading)")
    ax[1].bar(xa + 0.2, [max((r["minability"][k]["n_t2"] for k in r["minability"]), default=0)
                         for r in arms.values()], width=0.4, label="|T2|")
    ax[1].set_xticks(xa); ax[1].set_xticklabels(nm, rotation=30, ha="right", fontsize=8)
    ax[1].legend(fontsize=8); ax[1].set_title("next-level minability")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_objective_minability.png"), dpi=130)
    plt.close(fig)
    print(f"figures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--extra", default="",
                    help="comma-separated ADDITIONAL tags whose arm directories are overlaid "
                         "onto --tag's (for arms added to a round in a later launch, so the "
                         "original tag is never rewritten). --tag supplies setup.json.")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    extra = [t for t in a.extra.split(",") if t.strip()]
    if a.fetch:
        fetch(a.tag)
        for t in extra:
            fetch(t)
    setup, arms = load(a.tag, extra)
    report(setup, arms)
    if a.figures:
        # an overlaid report writes its own figure directory so --tag's originals stand
        figures(a.tag, setup, arms,
                out=os.path.join(FIG, "+".join([a.tag] + extra)) if extra else None)


if __name__ == "__main__":
    main()
