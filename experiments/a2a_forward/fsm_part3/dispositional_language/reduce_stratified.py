"""Reduce a stratified re-analysis (`stratified.py`) to the repo: tables and one figure.

Run locally (CPU):

    modal volume get language-reduction-data \
        /a2a_forward/confabulation/dispositional/main/stratified/stratified.json .
    python3 -m a2a_forward.fsm_part3.dispositional_language.reduce_stratified \
        --results stratified.json

Writes `figures/stratified_reduction.txt` and `figures/stratified_advantage.png`.
Reduction only -- no interpretation.
"""

import argparse
import json
import os

FAM_ORDER = {"IMPL": 0, "BEHAV": 1}
DIR_ORDER = {"prosp": 0, "retro": 1}
STRAT_ORDER = ["syntactic", "entropy_q", "position", "resnorm_q"]
STRAT_NOTE = {
    "syntactic": "the parent harness's 5-way taxonomy, a deterministic function of the "
                 "input tokens. The residual's\n             published behavioural "
                 "signature is largest before closing delimiters and smallest at "
                 "sentence starts,\n             which is what makes this the proxy with "
                 "a prior.",
    "entropy_q": "M's own output-entropy quartile at that checkpoint, computed from the "
                 "logits. Boundaries fit per\n             checkpoint on train sequences, "
                 "matching how the targets themselves were binned.",
    "position":  "position-in-sequence bins: the crudest context-depth proxy.",
    "resnorm_q": "the OCCURRENT residual magnitude |r_c(p)| quartile at the default "
                 "instrument. An outside observer\n             cannot compute this, so "
                 "it is a comparison only and never an equal member of the set.",
}
HEADLINE = ["IMPL_PROSP", "BEHAV_PROSP", "IMPL_RETRO", "BEHAV_RETRO"]


def short(nm):
    """IMPL_PROSP_k1_h16m0.5 -> IMPL_PROSP; IMPL_PROSPEXC_k1_h16m0.5 -> IMPL_PROSPEXC."""
    return nm.split("_k")[0].split("_w")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="stratified.json")
    ap.add_argument("--tag", default="stratified")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "figures"))
    a = ap.parse_args()
    R = json.load(open(a.results))
    L = []
    P = L.append

    P("=" * 104)
    P(f"DISPOSITIONAL SELF-KNOWLEDGE -- stratified re-analysis  [tag={R['config']['tag']}]")
    P("=" * 104)
    P("Does the pooled dispositional result hide structure, as the grammar sibling's")
    P("hierarchy-level cut does? Language has no level labels, so the same cut is run with")
    P("proxies for computational depth: three public, one private and for comparison only.")
    P("")
    P("All numbers are the STRICT evaluation set: held-out sequences at the held-out")
    P("checkpoint block, scored WITHIN stratum.")
    P("  ADVio  = self - best O_io, the max over the three observer capacities evaluated")
    P("           IN THAT STRATUM (which lets the observer pick its best capacity per")
    P("           stratum while the self cannot -- generous to the third party).")
    P("  ADVpub = self - max(O_io, O_hist): the conservative statistic for a dispositional")
    P("           target, since O_hist reads only M's outputs, just over several snapshots.")
    P("  O_act  = handed M's early activations. A ceiling, never an advantage comparator.")
    P("  chance = the WITHIN-STRATUM majority-class share. No pooled head or observer may")
    P("           condition on the stratum label, so it is not a baseline any of them could")
    P("           have used -- but a large advantage in a stratum where nobody reaches")
    P("           chance is a different object from one where they all do. Read it.")
    P("Strata with fewer than 200 evaluation positions are dropped.")
    P("")

    # ---------------- reproduction check ----------------
    chk = R["reproduction_check"]
    diffs = [abs(c["refit"] - c["stored"]) for c in chk
             if c["stored"] == c["stored"]]
    P("-" * 104)
    P("0. REPRODUCTION CHECK")
    P("   The parent run saved aggregate scores, not per-position predictions, so the heads")
    P("   and observers had to be RE-FIT to be interrogated. Nothing else was recomputed:")
    P("   M's checkpoints, the instrument FMs and the per-checkpoint parts were loaded from")
    P("   the volume. Every seed on the head/observer path is explicit, so the re-fit models")
    P("   should be the same models. `n/a` = a capacity the parent run did not train (the")
    P("   two light targets got only the 4L256D O_io; here they get the full sweep, which")
    P("   makes their observer stronger).")
    P("-" * 104)
    P(f"  {'target':32s} {'predictor':16s} {'re-fit':>9s} {'stored':>9s} {'diff':>9s}")
    for c in chk:
        st = c["stored"]
        if st != st:
            P(f"  {c['target']:32s} {c['predictor']:16s} {c['refit']:9.4f} "
              f"{'n/a':>9s} {'n/a':>9s}")
        else:
            P(f"  {c['target']:32s} {c['predictor']:16s} {c['refit']:9.4f} "
              f"{st:9.4f} {c['refit'] - st:+9.4f}")
    P(f"  max |difference| over the {len(diffs)} comparable pairs = {max(diffs):.4f}")
    P("")

    # ---------------- the four stratifiers ----------------
    for i, sname in enumerate(STRAT_ORDER, start=1):
        s = R["strata"][sname]
        P("-" * 104)
        P(f"{i}. STRATIFIER: {sname}   [{s['kind']}]")
        P(f"   {STRAT_NOTE[sname]}")
        P("-" * 104)
        P(f"  {'target':30s} {'stratum':16s} {'n':>8s} {'share':>6s} {'chance':>7s} "
          f"{'self':>7s} {'O_io':>7s} {'O_hist':>7s} {'O_act':>7s} {'ADVio':>7s} "
          f"{'ADVpub':>7s}")
        rows = sorted(s["rows"], key=lambda r: (FAM_ORDER.get(r["family"], 9),
                                                DIR_ORDER.get(r["direction"], 9),
                                                r["target"]))
        last = None
        for r in rows:
            if last is not None and r["target"] != last:
                P("")
            last = r["target"]
            P(f"  {r['target']:30s} {r['stratum']:16s} {r['n']:8d} {r['share']:6.3f} "
              f"{r['chance']:7.3f} {r['self']:7.3f} {r['O_io']:7.3f} {r['O_hist']:7.3f} "
              f"{r['O_act']:7.3f} {r['adv_io']:+7.3f} {r['adv_pub']:+7.3f}")
        P("")

    # ---------------- headline cross-stratifier summary ----------------
    P("-" * 104)
    P("5. THE HEADLINE ROWS ACROSS ALL FOUR STRATIFIERS (ADVpub), one line per stratum")
    P("-" * 104)
    for sname in STRAT_ORDER:
        s = R["strata"][sname]
        strata = []
        for r in s["rows"]:
            if r["stratum"] not in strata:
                strata.append(r["stratum"])
        P(f"  {sname}  [{s['kind']}]")
        P(f"    {'row':14s} " + " ".join(f"{st:>16s}" for st in strata))
        P(f"    {'share':14s} " + " ".join(
            f"{next((r['share'] for r in s['rows'] if r['stratum'] == st), float('nan')):16.3f}"
            for st in strata))
        for h in HEADLINE:
            cells = []
            for st in strata:
                v = next((r["adv_pub"] for r in s["rows"]
                          if r["stratum"] == st and short(r["target"]) == h), None)
                cells.append(f"{v:+16.3f}" if v is not None else f"{'--':>16s}")
            P(f"    {h:14s} " + " ".join(cells))
        P("")
    P("=" * 104)

    os.makedirs(a.outdir, exist_ok=True)
    txt = "\n".join(L)
    with open(os.path.join(a.outdir, f"{a.tag}_reduction.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    make_figure(R, a.outdir, a.tag)
    print(f"\nwrote {a.outdir}/{a.tag}_reduction.txt and "
          f"{a.outdir}/{a.tag}_advantage.png")


def make_figure(R, outdir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    COL = {"IMPL_PROSP": "#2b6cb0", "IMPL_RETRO": "#1a365d",
           "BEHAV_PROSP": "#a0aec0", "BEHAV_RETRO": "#718096"}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    for ax, sname in zip(axes, STRAT_ORDER):
        s = R["strata"][sname]
        strata, shares = [], []
        for r in s["rows"]:
            if r["stratum"] not in strata:
                strata.append(r["stratum"]); shares.append(r["share"])
        x = np.arange(len(strata))
        w = 0.2
        for j, h in enumerate(HEADLINE):
            vals = [next((r["adv_pub"] for r in s["rows"]
                          if r["stratum"] == st and short(r["target"]) == h), np.nan)
                    for st in strata]
            ax.bar(x + (j - 1.5) * w, vals, w, color=COL[h],
                   label=h.replace("_", " ").lower() if sname == STRAT_ORDER[0] else None)
        ax.axhline(0, color="#333", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{st}\n{sh * 100:.1f}%" if sh < 0.02 else f"{st}\n{sh:.0%}"
             for st, sh in zip(strata, shares)], fontsize=7, rotation=20, ha="right")
        ax.set_title(f"{sname}  [{s['kind']}]"
                     + ("\n(not available to an observer)"
                        if s["kind"] == "PRIVATE" else ""), fontsize=10)
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel(r"ADVpub = self $-$ max($O_{io}$, $O_{hist}$)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Dispositional advantage within stratum, strict evaluation set "
                 "(x-axis labels carry each stratum's share of positions)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{outdir}/{tag}_advantage.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
