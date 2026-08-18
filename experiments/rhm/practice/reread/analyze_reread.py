"""Reduce a reread run.

    python3 rhm/practice/reread/analyze_reread.py --tag rr_s0 --fetch --figures

Readouts, in the order the question asks for them:
  0. THE FIDELITY GATE — `fid_ratchet` (ratchet's era-1 configuration on this loop) against
     ratchet's published era-1 numbers. Nothing below is trustworthy if this fails.
  1. MINING YIELD PER PASS, PER LEVEL, per (mix, arm): raw entries, new entries, new entries
     the DP actually SELECTS (`n_new_used` — the non-double-counting count), the audition,
     and the pass-over-pass audition change the count is checked against.
  2. THE EXTRACTED FRACTION against ground truth: what share of the archive's own level-l
     grammatical content the vocabulary holds at pass k.
  3. THE PAIRED NOVELTY PROBE: at each pass, the same agent re-solves the frozen archive and
     a fresh matched draw, mining each with a throwaway miner. reread-yield vs fresh-yield.
  4. THE MONOLITH BASELINE: the dense arms' per-pass metering and archive (practice) error —
     what matched extra passes buy a learner with no vocabulary.
  5. CONCENTRATION VS COVERAGE: the mined table's audition against a matched-size random
     subset of the TRUE table (`rand_k`) and of the COMPOSITION CLOSURE (`comp_k`), plus the
     full closure — the "re-derivable from T[l-1] alone" control.
  6. THE DEPTH COORDINATE: everything above, split by archive mix (shallow / mixed / deep).
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_reread"
ARM_ORDER = ["reread", "fresh", "dense_reread", "dense_fresh", "given_reread", "fid_ratchet"]
MIX_ORDER = ["shallow", "mixed", "deep"]
# ratchet's published era-1 (L1n6) reference, README table + certificate table
RATCHET_REF = {"e_L1n6_never_base": 0.349, "tab_recall": 0.500, "tab_precision": 0.875}


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    cells = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "results.json")
        if os.path.isfile(p):
            cells[name] = json.load(open(p))

    def key(n):
        mix, _, arm = n.partition("__")
        return (MIX_ORDER.index(mix) if mix in MIX_ORDER else 9,
                ARM_ORDER.index(arm) if arm in ARM_ORDER else 9)
    return setup, {k: cells[k] for k in sorted(cells, key=key)}


def V(rec, ell, field, default=None):
    return rec["vocab"].get(str(ell), {}).get(field, default)


def fmt(x, n=3):
    return "  -  " if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


# --------------------------------------------------------------------------- #

def report(tag):
    setup, cells = load(tag)
    cfg = setup["config"]
    print(f"\n=== {tag} ===")
    print(f"n_arch={cfg['n_arch']} chunk={cfg['chunk']} n_passes={cfg['n_passes']} "
          f"commit_pass={cfg['commit_pass']} mine_support={cfg['mine_support']} "
          f"mine_cap={cfg['mine_cap']} seed={cfg['seed']}")
    print("archives: " + " | ".join(
        f"{k}: sha={a['sha']} content " +
        " ".join(f"L{e}={c['n_true']}/{setup['truth_sizes'][e]}"
                 for e, c in sorted(setup["content"][k].items()))
        for k, a in sorted(setup["archives"].items())))

    # ---- 0. the fidelity gate ----------------------------------------------------------
    print("\n-- 0. FIDELITY GATE (fid_ratchet = ratchet's era-1 config on this loop) --")
    fid = cells.get("shallow__fid_ratchet")
    if fid is None:
        print("   MISSING")
    else:
        ps = fid["passes"]
        e = [p["meter"]["L1n6"] for p in ps]
        last = ps[-1]
        rec = V(last, 2, "n_correct"), V(last, 2, "n_entries")
        recall = rec[0] / setup["truth_sizes"]["2"] if rec[1] else 0.0
        prec = (rec[0] / rec[1]) if rec[1] else None
        print(f"   L1n6 metering e per pass: " + " ".join(f"{x:.3f}" for x in e))
        print(f"   final e={e[-1]:.3f}   ratchet never_base era-1 = "
              f"{RATCHET_REF['e_L1n6_never_base']:.3f}   "
              f"delta={e[-1] - RATCHET_REF['e_L1n6_never_base']:+.3f}")
        print(f"   final L2 table: {rec[1]} entries, recall={recall:.3f} "
              f"(ratchet {RATCHET_REF['tab_recall']:.3f}), precision={fmt(prec)} "
              f"(ratchet {RATCHET_REF['tab_precision']:.3f})")
        print(f"   L2 audition per pass: "
              + " ".join(fmt(V(p, 2, 'aud')) for p in ps))

    # ---- 1-2. yield per pass, per level -------------------------------------------------
    for ell in (2, 3):
        print(f"\n-- 1. LEVEL-{ell} YIELD PER PASS  "
              f"[entries / new / new_used | aud | extracted frac of archive content] --")
        for name, r in cells.items():
            if name.endswith("fid_ratchet") and ell == 3:
                continue
            tot = r["content"][str(ell)]["n_true"]
            rows = []
            for p in r["passes"]:
                rows.append(f"{fmt(V(p, ell, 'n_entries'))}/{fmt(V(p, ell, 'n_new'))}/"
                            f"{fmt(V(p, ell, 'n_new_used'))}"
                            f"|{fmt(V(p, ell, 'aud'))}"
                            f"|{(V(p, ell, 'n_content_true') or 0) / max(tot, 1):.2f}")
            print(f"   {name:24s} " + "  ".join(f"p{i+1}:{x}" for i, x in enumerate(rows)))
        print(f"   (denominator = the archive's own grammatical level-{ell} content)")

    # ---- 2b. the two channels, separated ------------------------------------------------
    print("\n-- 2b. WHERE TABLE GROWTH COMES FROM: `n_distinct` (support-1: spans the agent "
          "actually OBSERVED — the solvability channel) vs `n_at_support[3]` (what is "
          "buildable — which also grows when an OLD span crosses support by repetition) --")
    for ell in (2, 3):
        print(f"   level {ell}:")
        for name, r in cells.items():
            mn = [V(p, ell, "miner") for p in r["passes"]]
            if not mn or mn[0] is None:
                continue
            d = [x["n_distinct"] for x in mn]
            a = [x["n_at_support"]["3"] for x in mn]
            print(f"     {name:24s} distinct " + " ".join(f"{x:2d}" for x in d)
                  + f"  (+{d[-1]-d[0]:2d})   at_sup3 " + " ".join(f"{x:2d}" for x in a)
                  + f"  (+{a[-1]-a[0]:2d}; >={max(a[-1]-a[0]-(d[-1]-d[0]), 0)} from repetition)")

    # ---- 3. the paired novelty probe ----------------------------------------------------
    print("\n-- 3. PAIRED NOVELTY PROBE (same agent, same pass: frozen archive vs a fresh "
          "matched draw; throwaway miner both sides) --")
    for ell in (2, 3):
        print(f"   level {ell}:  reread_entries -> fresh_entries  (and correct/correct)")
        for name, r in cells.items():
            if not r["passes"] or r["passes"][0].get("probe") is None:
                continue
            row = []
            wins = {"rr": 0, "fr": 0, "tie": 0}
            for p in r["passes"]:
                pr = p["probe"]["reread"].get(f"L{ell}")
                pf = p["probe"]["fresh"].get(f"L{ell}")
                if pr is None or pf is None:
                    row.append("  -  ")
                    continue
                a, b = pr["n_entries"], pf["n_entries"]
                wins["rr" if a > b else ("fr" if b > a else "tie")] += 1
                row.append(f"{a:2d}>{b:2d}" if a > b else
                           (f"{a:2d}<{b:2d}" if a < b else f"{a:2d}={b:2d}"))
            print(f"     {name:24s} " + " ".join(row)
                  + f"   [reread wins {wins['rr']}, fresh {wins['fr']}, tie {wins['tie']}]")

    print("\n   correct-entry version (mined entries that are real grammar):")
    for ell in (2, 3):
        for name, r in cells.items():
            if not r["passes"] or r["passes"][0].get("probe") is None:
                continue
            rr = [p["probe"]["reread"].get(f"L{ell}", {}).get("n_correct") for p in r["passes"]]
            fr = [p["probe"]["fresh"].get(f"L{ell}", {}).get("n_correct") for p in r["passes"]]
            rr = [x for x in rr if x is not None]
            fr = [x for x in fr if x is not None]
            if not rr:
                continue
            print(f"     L{ell} {name:24s} reread mean={np.mean(rr):5.2f} "
                  f"fresh mean={np.mean(fr):5.2f}  ratio={np.mean(rr)/max(np.mean(fr),1e-9):.3f}")

    print("\n   solved-fraction of each side (the solvability channel):")
    for name, r in cells.items():
        if not r["passes"] or r["passes"][0].get("probe") is None:
            continue
        rr = [p["probe"]["reread"]["solved"] / p["probe"]["reread"]["n"] for p in r["passes"]]
        fr = [p["probe"]["fresh"]["solved"] / p["probe"]["fresh"]["n"] for p in r["passes"]]
        print(f"     {name:24s} reread " + " ".join(f"{x:.2f}" for x in rr)
              + " | fresh " + " ".join(f"{x:.2f}" for x in fr))

    # ---- 4. the monolith baseline -------------------------------------------------------
    print("\n-- 4. METERING PER PASS, PER DAMAGE DEPTH (e = 1 - success @ declared budget) --")
    for cellname in ("L1n6", "L2n3", "L3n1"):
        print(f"   {cellname}:")
        for name, r in cells.items():
            print(f"     {name:24s} "
                  + " ".join(f"{p['meter'][cellname]:.3f}" for p in r["passes"]))
    print("\n   archive (practice) error per pass — the epoch curve on the presented data:")
    for name, r in cells.items():
        byp = {}
        for c in r["cycles"]:
            byp.setdefault(c["pass"], []).append(c["e_practice"])
        print(f"     {name:24s} "
              + " ".join(f"{np.mean(byp[k]):.3f}" for k in sorted(byp)))

    # ---- 5. concentration vs coverage ---------------------------------------------------
    print("\n-- 5. CONCENTRATION VS COVERAGE (audition e; lower is better). "
          "mined vs matched-size random subset of the TRUE table (rand_k) vs of the "
          "COMPOSITION CLOSURE (comp_k) vs the full closure vs the true table --")
    for ell in (2, 3):
        print(f"   level {ell}  [mined | rand_k | comp_k | closure | true]  "
              "(mean over passes where the mined table is non-empty)")
        for name, r in cells.items():
            cols = {k: [] for k in ("aud", "rand_k", "comp_k", "closure", "true")}
            for p in r["passes"]:
                if V(p, ell, "aud") is None:
                    continue
                for k in cols:
                    x = V(p, ell, k)
                    if x is not None:
                        cols[k].append(x)
            if not cols["aud"]:
                continue
            print(f"     {name:24s} " + " | ".join(
                f"{k}={np.mean(v):.3f}" if v else f"{k}=  -  " for k, v in cols.items()))

    # ---- 6. commits and the plant guard -------------------------------------------------
    print("\n-- 6. COMMITS --")
    for name, r in cells.items():
        for ev in r["events"]:
            print(f"   {name:24s} L{ev['level']} pass {ev['pass_']:2d}  "
                  f"entries={ev['n_entries']:3d} recall={ev['tab_recall']:.3f} "
                  f"precision={fmt(ev['tab_precision'])} moves={ev['n_moves_after']}")
    print("\n-- plant guard (clean held-out configs; oracle readout, never consumed) --")
    for name, r in cells.items():
        pl = [p["plant"] for p in r["passes"]]
        print(f"   {name:24s} parse {pl[0]['parse_acc']:.3f}->{pl[-1]['parse_acc']:.3f}  "
              f"infill {pl[0]['infill_acc']:.3f}->{pl[-1]['infill_acc']:.3f}  "
              f"read {pl[0]['read_acc']:.3f}->{pl[-1]['read_acc']:.3f}")
    return setup, cells


# --------------------------------------------------------------------------- #

def figures(tag, setup, cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    mixes = [m for m in MIX_ORDER if any(k.startswith(m + "__") for k in cells)]
    arms = [a for a in ARM_ORDER if a != "fid_ratchet"
            and any(k.endswith("__" + a) for k in cells)]
    colors = {a: c for a, c in zip(arms, plt.cm.tab10.colors)}

    def series(name, fn):
        return [fn(p) for p in cells[name]["passes"]]

    # fig1 — mining yield per pass, per level, per mix
    fig, axes = plt.subplots(2, len(mixes), figsize=(5 * len(mixes), 8), squeeze=False)
    for j, mix in enumerate(mixes):
        for i, ell in enumerate((2, 3)):
            ax = axes[i][j]
            for a in arms:
                nm = f"{mix}__{a}"
                if nm not in cells:
                    continue
                y = series(nm, lambda p: V(p, ell, "n_entries"))
                ax.plot(range(1, len(y) + 1), [np.nan if v is None else v for v in y],
                        marker="o", ms=3, color=colors[a], label=a)
                yu = series(nm, lambda p: V(p, ell, "n_used"))
                ax.plot(range(1, len(yu) + 1), [np.nan if v is None else v for v in yu],
                        ls="--", lw=1, color=colors[a])
            tot = cells[f"{mix}__{arms[0]}"]["content"][str(ell)]["n_true"]
            ax.axhline(tot, color="grey", lw=0.8, ls=":")
            for pp in (setup["config"]["commit_pass"].get(str(ell)),):
                if pp:
                    ax.axvline(pp + 0.5, color="black", lw=0.8, alpha=0.5)
            ax.set_title(f"{mix}: level-{ell} table size (solid) / entries USED (dashed)")
            ax.set_xlabel("pass"); ax.set_ylabel("entries")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_yield.png"), dpi=130)

    # fig2 — extracted fraction of the archive's own grammatical content
    fig, axes = plt.subplots(2, len(mixes), figsize=(5 * len(mixes), 8), squeeze=False)
    for j, mix in enumerate(mixes):
        for i, ell in enumerate((2, 3)):
            ax = axes[i][j]
            for a in arms:
                nm = f"{mix}__{a}"
                if nm not in cells:
                    continue
                tot = max(cells[nm]["content"][str(ell)]["n_true"], 1)
                y = series(nm, lambda p: (V(p, ell, "n_content_true") or 0) / tot)
                ax.plot(range(1, len(y) + 1), y, marker="o", ms=3, color=colors[a], label=a)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{mix}: fraction of the archive's level-{ell} content extracted")
            ax.set_xlabel("pass"); ax.set_ylabel("extracted / content")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_extracted.png"), dpi=130)

    # fig3 — the paired novelty probe
    fig, axes = plt.subplots(2, len(mixes), figsize=(5 * len(mixes), 8), squeeze=False)
    for j, mix in enumerate(mixes):
        for i, ell in enumerate((2, 3)):
            ax = axes[i][j]
            for a in arms:
                nm = f"{mix}__{a}"
                if nm not in cells or cells[nm]["passes"][0].get("probe") is None:
                    continue
                rr = series(nm, lambda p: p["probe"]["reread"].get(f"L{ell}", {}).get("n_entries"))
                fr = series(nm, lambda p: p["probe"]["fresh"].get(f"L{ell}", {}).get("n_entries"))
                ax.plot(range(1, len(rr) + 1), [np.nan if v is None else v for v in rr],
                        marker="o", ms=3, color=colors[a], label=f"{a} reread")
                ax.plot(range(1, len(fr) + 1), [np.nan if v is None else v for v in fr],
                        marker="x", ms=4, ls="--", color=colors[a], label=f"{a} fresh")
            ax.set_title(f"{mix}: L{ell} paired probe — frozen (o) vs fresh (x) yield")
            ax.set_xlabel("pass"); ax.set_ylabel("entries mined this pass")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_novelty.png"), dpi=130)

    # fig4 — metering per damage depth
    cellnames = ("L1n6", "L2n3", "L3n1")
    fig, axes = plt.subplots(len(mixes), 3, figsize=(15, 4 * len(mixes)), squeeze=False)
    for j, mix in enumerate(mixes):
        for i, cn in enumerate(cellnames):
            ax = axes[j][i]
            for a in arms:
                nm = f"{mix}__{a}"
                if nm not in cells:
                    continue
                y = series(nm, lambda p: p["meter"][cn])
                ax.plot(range(1, len(y) + 1), y, marker="o", ms=3, color=colors[a], label=a)
            ax.set_title(f"{mix}: metered e on {cn}")
            ax.set_xlabel("pass"); ax.set_ylabel("e")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_competence.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rr_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    setup, cells = report(a.tag)
    if a.figures:
        figures(a.tag, setup, cells)


if __name__ == "__main__":
    main()
