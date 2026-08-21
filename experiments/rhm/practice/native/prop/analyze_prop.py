"""Reduction for PORT 1 (routing / proposal).

Reuses `ratchet/analyze_ratchet.py` verbatim for every ratchet-standard readout (per-era
competence and priced time, the earned-vs-given fraction, matched-priced-time margins, commit
events, unit-LP trajectories, the plant guard) by retargeting its module-level volume prefix and
figure root — nothing in `ratchet/` is modified.

On top of it, the readouts the spec commits Port 1 to:

  (0) THE GATES. Twinning: every treated arm must be bit-identical to its untreated twin until
      the filter switches on (`prop_warmup`), max|de| = 0.0. Fidelity: `*_prop_kN` (k = n_moves,
      the filter an exact no-op) must reproduce its twin for the WHOLE run.
  (1) THE LEDGER. Groundings, materialisations and proposal reads per solve at the declared
      budget, per era, with the effective branching factor and the width it buys; plus the
      cost of reaching the twin's own success, read off the width ladder (the "banked" reading
      of the freed budget, against the run's "reinvested" one).
  (2) THE CAN'T-DECOMPOSE SIGNATURE. On states where pi proposes a macro, how much mass does it
      keep on that macro's own primitive decomposition? Against the `never_base_prop_*` control,
      where the same primitives are the only way to address the same span.
  (3) NEXT-LEVEL CURRENCY. |T_{k+1}| minable at `mine_support` and the level-k+1 shadow
      audition, under native routing vs exogenous enumeration.
  (4) PLANT GUARD, every probe (handle/'s instrument, like-for-like).

Usage (from experiments/):
    python3 rhm/practice/native/prop/analyze_prop.py --tag np_s0 --fetch --figures
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RATCHET = os.path.join(os.path.dirname(os.path.dirname(HERE)), "ratchet")
sys.path.insert(0, RATCHET)
import analyze_ratchet as AR                                          # noqa: E402

AR.FIG = FIG
AR.REMOTE = "rhm_practice_native_prop"
ORDER = ["never_base", "never_base_prop_k1", "never_base_prop_k2", "never_base_prop_k4",
         "given", "given_prop_kN", "given_prop_k1", "given_prop_k2", "given_prop_k4",
         "practice_late", "practice_late_prop_k2", "practice_late_prop_k4"]
AR.ORDER = ORDER

TWIN = {"given_prop_kN": "given", "given_prop_k1": "given", "given_prop_k2": "given",
        "given_prop_k4": "given",
        "never_base_prop_k1": "never_base", "never_base_prop_k2": "never_base",
        "never_base_prop_k4": "never_base",
        "practice_late_prop_k2": "practice_late", "practice_late_prop_k4": "practice_late"}

SLOT_NAME = ["L1n0", "L1n1", "L1n2", "L1n3", "L1n4", "L1n5", "L1n6", "L1n7",
             "L2n0", "L2n1", "L2n2", "L2n3", "L3n0", "L3n1"]


# --------------------------------------------------------------------------- #
# (0) the gates
# --------------------------------------------------------------------------- #

def gates(arms, setup):
    warm = int(setup["config"].get("prop_warmup", 4))
    print("\n== (0a) TWINNING GATE — bit-identical to the untreated twin until the filter "
          f"switches on (cycle {warm + 1}) ==")
    print(f"{'arm vs twin':50s} {'max|de| pre':>12s} {'first diverge':>14s} "
          f"{'max|de| all':>12s} {'dt_cum':>12s}")
    out = {}
    for arm, tw in TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        a = np.asarray(arms[arm]["log"]["e"], float)
        b = np.asarray(arms[tw]["log"]["e"], float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        pre = np.abs(a[:warm] - b[:warm])
        d = np.nonzero(np.abs(a - b) > 0)[0]
        ta = np.asarray(arms[arm]["log"]["t_cum"], float)[:n]
        tb = np.asarray(arms[tw]["log"]["t_cum"], float)[:n]
        out[arm] = {"pre": float(pre.max()) if pre.size else 0.0,
                    "first": int(d[0]) + 1 if len(d) else None,
                    "all": float(np.abs(a - b).max()),
                    "dt": float(ta[-1] - tb[-1])}
        print(f"{arm + ' - ' + tw:50s} {out[arm]['pre']:12.6f} "
              f"{('c' + str(out[arm]['first'])) if out[arm]['first'] else '(never)':>14s} "
              f"{out[arm]['all']:12.6f} {out[arm]['dt']:+12.0f}")

    print("\n== (0b) FIDELITY GATE — k = n_moves must reproduce enumeration bit-for-bit ==")
    for arm, tw in TWIN.items():
        if not arm.endswith("_kN") or arm not in arms or tw not in arms:
            continue
        ok = out[arm]["all"] == 0.0
        print(f"  {arm} vs {tw}: max|de| over the whole run = {out[arm]['all']:.8f}  "
              f"-> {'PASS' if ok else 'FAIL'}   (priced-time delta {out[arm]['dt']:+.0f}, "
              "the declared root encode)")
    return out


# --------------------------------------------------------------------------- #
# (1) the ledger
# --------------------------------------------------------------------------- #

def ledger(arms, setup, tail=5):
    """Per era: e over the last `tail` cycles, the era's priced time, and the PRICED (metering)
    beam's own cost per solve — groundings, materialisations and proposal reads — together with
    the effective branching factor and the width it buys at the declared G."""
    print(f"\n== (1) THE LEDGER — e (last {tail} cycles of the era), priced time, and the "
          "metering beam's cost per solve ==")
    print(f"{'arm':26s} {'era':>3s} {'e':>7s} {'t_era':>10s} {'g/solve':>8s} "
          f"{'mat/solve':>9s} {'pi/solve':>9s} {'k_eff':>6s} {'w':>5s} {'nm':>4s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = AR.era_slices(log)
        t = np.asarray(log["t_cum"], float)
        e = np.asarray(log["e"], float)
        pr = log.get("prop") or []
        out[arm] = {}
        for k, idx in sl.items():
            sel = idx[-tail:]
            mt = [pr[i]["meter"] for i in sel if i < len(pr) and pr[i].get("meter")]
            f = lambda key, dflt=float("nan"): (float(np.mean([c[key] for c in mt]))
                                                if mt else dflt)
            row = {"e": float(e[sel].mean()),
                   "t_era": float(t[idx[-1]] - (t[idx[0] - 1] if idx[0] else 0.0)),
                   "t_end": float(t[idx[-1]]),
                   "g_solve": f("g", float(np.asarray(log["g_per_solve"], float)[sel].mean())),
                   "mat_solve": f("mat"), "pi_solve": f("prop", 0.0),
                   "k_eff": f("k_eff"), "w": f("w", float(np.asarray(log["width"], int)[sel].mean())),
                   "n_moves": int(log["n_moves"][idx[-1]])}
            out[arm][k] = row
            print(f"{arm:26s} {k:3d} {row['e']:7.4f} {row['t_era']:10.0f} "
                  f"{row['g_solve']:8.1f} {row['mat_solve']:9.1f} {row['pi_solve']:9.1f} "
                  f"{row['k_eff']:6.1f} {row['w']:5.1f} {row['n_moves']:4d}")
    return out


def matched_success(arms, per, tail=5):
    """The freed budget, read the OTHER way. The run reinvests it (each arm takes the widest
    beam its branching lets it afford at G); the width ladder logged at every probe recovers
    the banked reading — the cheapest width at which the treated arm already matches its
    enum twin's error, and what that costs in groundings."""
    print("\n== (1b) COST TO MATCH THE TWIN'S SUCCESS (from the probe width ladder, "
          "unpriced instrument) ==")
    print(f"{'arm vs twin':50s} {'era':>3s} {'twin e':>7s} {'twin g':>7s} "
          f"{'w*':>4s} {'e@w*':>7s} {'g@w*':>7s} {'ratio':>6s}")
    out = {}
    for arm, tw in TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        pa = [p for p in arms[arm]["log"]["probe"]]
        pb = [p for p in arms[tw]["log"]["probe"]]
        for era in sorted({p["era"] for p in pa}):
            la = [p for p in pa if p["era"] == era][-tail:]
            lb = [p for p in pb if p["era"] == era][-tail:]
            if not la or not lb:
                continue
            widths = sorted(la[0]["ladder"], key=int)
            e_tw = per[tw][era]["e"]           # the twin at its own operating point
            g_tw = per[tw][era]["g_solve"]
            best = None
            for w in widths:
                ew = float(np.mean([q["ladder"][w]["e"] for q in la]))
                gw = float(np.mean([q["ladder"][w]["g"] for q in la]))
                if ew <= e_tw and (best is None or gw < best[2]):
                    best = (w, ew, gw)
            if best is None:
                w = widths[-1]
                best = (w + "*", float(np.mean([q["ladder"][w]["e"] for q in la])),
                        float(np.mean([q["ladder"][w]["g"] for q in la])))
            out[(arm, era)] = {"twin_e": e_tw, "twin_g": g_tw, "w": best[0],
                               "e": best[1], "g": best[2], "ratio": best[2] / max(g_tw, 1e-9)}
            print(f"{arm + ' - ' + tw:50s} {era:3d} {e_tw:7.4f} {g_tw:7.1f} "
                  f"{str(best[0]):>4s} {best[1]:7.4f} {best[2]:7.1f} "
                  f"{out[(arm, era)]['ratio']:6.2f}")
    print("  (w* with a trailing * = the ladder never reached the twin's error; the widest "
          "rung is shown instead)")
    return out


# --------------------------------------------------------------------------- #
# (2) the can't-decompose signature
# --------------------------------------------------------------------------- #

def decompose(arms, tail=3):
    """An expert cannot decompose its chunks. On the states where pi's argmax IS a macro, how
    much mass does it still hold on that macro's own primitive decomposition?"""
    print("\n== (2) THE CAN'T-DECOMPOSE SIGNATURE (mean over the last "
          f"{tail} probes of each era) ==")
    print(f"{'arm':26s} {'era':>3s} {'top1':>6s} {'H':>6s} {'macro mass':>10s} "
          f"{'macro argmax':>12s} {'p(macro)':>9s} {'p(dec L1)':>10s} {'ratio':>7s} "
          f"{'p(dec L2)':>10s}")
    out = {}
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "pi" in p]
        if not pr:
            continue
        out[arm] = {}
        for era in sorted({p["era"] for p in pr}):
            sel = [p for p in pr if p["era"] == era][-tail:]
            row = {"top1": float(np.mean([p["pi"]["top1"] for p in sel])),
                   "entropy": float(np.mean([p["pi"]["entropy"] for p in sel])),
                   "macro_mass": float(np.mean([p["pi"]["macro_mass"] for p in sel]))}
            fr, pm, d1, d2, r1 = [], [], [], [], []
            for p in sel:
                for c in p["pi"]["macros"]:
                    fr.append(c["frac_argmax"])
                    if c.get("n_argmax"):
                        pm.append(c["p_macro"])
                        if "p_dec1" in c:
                            d1.append(c["p_dec1"]); r1.append(c["ratio1"])
                        if "p_dec2" in c:
                            d2.append(c["p_dec2"])
            row.update({"macro_argmax": float(np.sum(fr) / max(len(sel), 1)),
                        "p_macro": float(np.mean(pm)) if pm else float("nan"),
                        "p_dec1": float(np.mean(d1)) if d1 else float("nan"),
                        "ratio1": float(np.mean(r1)) if r1 else float("nan"),
                        "p_dec2": float(np.mean(d2)) if d2 else float("nan")})
            out[arm][era] = row
            print(f"{arm:26s} {era:3d} {row['top1']:6.3f} {row['entropy']:6.3f} "
                  f"{row['macro_mass']:10.4f} {row['macro_argmax']:12.3f} "
                  f"{row['p_macro']:9.4f} {row['p_dec1']:10.4f} {row['ratio1']:7.3f} "
                  f"{row['p_dec2']:10.4f}")
    print("  ratio = p(the macro's primitive decomposition) / p(the macro), on states where "
          "the macro IS the argmax.\n  The `never_base_prop_*` rows have no macros: their "
          "top1/H are the control for how peaked the head gets at all.")
    return out


def slot_profile(arms, tail=3):
    """Where the head's mass actually sits, per slot, at the end of each era."""
    print("\n== (2b) pi's mean mass per slot at each era's end ==")
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "pi" in p]
        if not pr:
            continue
        for era in sorted({p["era"] for p in pr}):
            sel = [p for p in pr if p["era"] == era][-tail:]
            mm = np.mean([p["pi"]["mean_mass"] for p in sel], axis=0)
            top = np.argsort(-mm)[:5]
            print(f"{arm:26s} era{era}  " + "  ".join(
                f"{SLOT_NAME[j]}:{mm[j]:.3f}" for j in top))


# --------------------------------------------------------------------------- #
# (3) next-level currency
# --------------------------------------------------------------------------- #

def yield_table(arms, tail=5):
    print("\n== (3) NEXT-LEVEL CURRENCY — |T_{k+1}| at mine_support and the level-k+1 "
          "shadow audition, at the end of each era ==")
    print(f"{'arm':26s} {'era':>3s} {'lvl':>3s} {'|T|':>6s} {'A':>7s} {'A_true':>7s} "
          f"{'recall':>7s} {'prec':>6s} {'rand_k':>7s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = AR.era_slices(log)
        out[arm] = {}
        for k, idx in sl.items():
            lvl = int(log["level"][idx[0]]) + 1
            cells = [log["aud"][i].get(str(lvl), {}) for i in idx[-tail:]]
            f = lambda key: float(np.mean([c[key] for c in cells if c.get(key) is not None])) \
                if any(c.get(key) is not None for c in cells) else float("nan")
            row = {"level": lvl, "n_entries": f("n_entries"), "A": f("cand"),
                   "A_true": f("true"), "recall": f("tab_recall"),
                   "precision": f("tab_precision"), "rand_k": f("rand_k")}
            out[arm][k] = row
            print(f"{arm:26s} {k:3d} {lvl:3d} {row['n_entries']:6.1f} {row['A']:7.4f} "
                  f"{row['A_true']:7.4f} {row['recall']:7.3f} {row['precision']:6.3f} "
                  f"{row['rand_k']:7.4f}")
    print("\n-- native routing minus exogenous enumeration (treated arm minus its twin) --")
    for arm, tw in TWIN.items():
        if arm not in out or tw not in out:
            continue
        for k in sorted(out[arm]):
            a, b = out[arm][k], out[tw][k]
            print(f"{arm + ' - ' + tw:50s} era{k} L{a['level']}  "
                  f"d|T|={a['n_entries'] - b['n_entries']:+7.1f}  dA={a['A'] - b['A']:+.4f}  "
                  f"drecall={a['recall'] - b['recall']:+.3f}  "
                  f"dprec={a['precision'] - b['precision']:+.3f}")
    return out


# --------------------------------------------------------------------------- #
# (4) the plant guard
# --------------------------------------------------------------------------- #

def plant_table(arms, tail=3):
    print("\n== (4) PLANT GUARD — parse / infill accuracy on clean held-out configurations ==")
    print(f"{'arm':26s} {'parse c1':>9s} {'parse end':>9s} {'d':>8s} "
          f"{'infill c1':>10s} {'infill end':>10s} {'d':>8s} {'read':>7s}")
    rows = {}
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "plant" in p]
        if not pr:
            continue
        cell = {}
        for k in ("parse_acc", "infill_acc", "read_acc"):
            vals = [p["plant"][k] for p in pr if k in p["plant"]]
            if vals:
                cell[k] = {"first": float(vals[0]), "last": float(np.mean(vals[-tail:])),
                           "series": [float(x) for x in vals],
                           "cycles": [p["cycle"] for p in pr][-len(vals):]}
        rows[arm] = cell
        pa, ia = cell["parse_acc"], cell["infill_acc"]
        print(f"{arm:26s} {pa['first']:9.4f} {pa['last']:9.4f} {pa['last']-pa['first']:+8.4f} "
              f"{ia['first']:10.4f} {ia['last']:10.4f} {ia['last']-ia['first']:+8.4f} "
              f"{cell.get('read_acc', {}).get('last', float('nan')):7.4f}")
    print("\n-- treated minus twin (end-of-run means) --")
    for arm, tw in TWIN.items():
        if arm not in rows or tw not in rows:
            continue
        a, b = rows[arm], rows[tw]
        print(f"{arm + ' - ' + tw:50s} dparse={a['parse_acc']['last']-b['parse_acc']['last']:+.4f}"
              f"  dinfill={a['infill_acc']['last']-b['infill_acc']['last']:+.4f}")
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

COLOUR = {"never_base": "0.55", "never_base_prop_k1": "0.75", "never_base_prop_k2": "0.35",
          "never_base_prop_k4": "0.15",
          "given": "tab:green", "given_prop_kN": "black", "given_prop_k1": "tab:olive",
          "given_prop_k2": "tab:red", "given_prop_k4": "tab:orange",
          "practice_late": "tab:blue", "practice_late_prop_k2": "tab:purple",
          "practice_late_prop_k4": "tab:cyan"}


def figures(tag, setup, arms, per, plant, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(FIG, tag)
    os.makedirs(root, exist_ok=True)
    style = {a: ("--" if a in TWIN else "-") for a in arms}
    ec = setup["config"]["era_cycles"]
    bounds = [ec * i for i in range(1, len(setup["eras"]))]

    # fig1 — competence per cycle, and against priced time
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for arm, r in arms.items():
        ax[0].plot(r["log"]["cycle"], r["log"]["e"], style[arm], lw=1.3,
                   color=COLOUR.get(arm), label=arm)
        ax[1].plot(r["log"]["t_cum"], r["log"]["e"], style[arm], lw=1.3,
                   color=COLOUR.get(arm), label=arm)
    for b_ in bounds:
        ax[0].axvline(b_, color="0.85", lw=0.8, zorder=0)
    warm = setup["config"].get("prop_warmup", 4)
    ax[0].axvline(warm + 0.5, color="firebrick", lw=0.8, ls=":", zorder=0)
    ax[0].set_xlabel("cycle"); ax[1].set_xlabel("priced time")
    for a_ in ax:
        a_.set_ylabel("e (1 - success), metering set"); a_.legend(fontsize=6)
    ax[0].set_title("competence per cycle (dotted red = the filter switches on)")
    ax[1].set_title("competence against priced time")
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig1_prop_competence.png"), dpi=150)
    plt.close(fig)

    # fig2 — the twin contrast and the ledger
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for arm, tw in TWIN.items():
        if arm not in arms or tw not in arms:
            continue
        a = np.asarray(arms[arm]["log"]["e"], float)
        b = np.asarray(arms[tw]["log"]["e"], float)
        n = min(len(a), len(b))
        ax[0].plot(arms[arm]["log"]["cycle"][:n], (a - b)[:n], lw=1.3,
                   color=COLOUR.get(arm), label=f"{arm} - {tw}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axvline(warm + 0.5, color="firebrick", lw=0.8, ls=":")
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("e(prop) - e(twin)")
    ax[0].set_title("the routing contrast (0 before the filter = matched pair)")
    for arm, r in arms.items():
        ax[1].plot(r["log"]["cycle"], r["log"]["g_per_solve"], style[arm], lw=1.3,
                   color=COLOUR.get(arm), label=arm)
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("groundings per solve (metering beam)")
    ax[1].set_title("the declared budget, spent")
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "pi" in p]
        if not pr:
            continue
        ax[2].plot([p["cycle"] for p in pr], [p["pi"]["macro_mass"] for p in pr], "-",
                   lw=1.3, color=COLOUR.get(arm), label=arm)
    ax[2].set_xlabel("cycle"); ax[2].set_ylabel("pi mass on macro slots")
    ax[2].set_title("does the head route to the chunk?")
    for a_ in ax:
        a_.legend(fontsize=6)
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig2_prop_ledger.png"), dpi=150)
    plt.close(fig)

    # fig3 — the can't-decompose signature, and the next-level currency
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for arm, r in arms.items():
        pr = [p for p in r["log"]["probe"] if "pi" in p]
        if not pr:
            continue
        xs, ys = [], []
        for p in pr:
            cells = [c for c in p["pi"]["macros"] if c.get("n_argmax")]
            if cells:
                xs.append(p["cycle"])
                ys.append(float(np.mean([c["ratio1"] for c in cells if "ratio1" in c])))
        if xs:
            ax[0].plot(xs, ys, "-", lw=1.3, color=COLOUR.get(arm), label=arm)
        ax[1].plot([p["cycle"] for p in pr], [p["pi"]["top1"] for p in pr], "-", lw=1.3,
                   color=COLOUR.get(arm), label=arm)
    ax[0].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("p(primitive decomposition) / p(macro)")
    ax[0].set_title("the can't-decompose signature\n(low = the chunk is not enumerated)")
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("mean top-1 proposal mass")
    ax[1].set_title("how peaked the policy gets")
    for arm, r in arms.items():
        log = r["log"]
        xs, ys = [], []
        for i, cyc in enumerate(log["cycle"]):
            lvl = int(log["level"][i]) + 1
            cell = log["aud"][i].get(str(lvl), {})
            if cell.get("n_entries") is not None:
                xs.append(cyc); ys.append(cell["n_entries"])
        if xs:
            ax[2].plot(xs, ys, style[arm], lw=1.2, color=COLOUR.get(arm), label=arm)
    ax[2].set_xlabel("cycle"); ax[2].set_ylabel("|T_{k+1}| minable entries")
    ax[2].set_title("next-level currency")
    for a_ in ax:
        a_.legend(fontsize=6)
        for b_ in bounds:
            a_.axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig3_prop_chunk.png"), dpi=150)
    plt.close(fig)

    # fig4 — the plant guard
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for j, key in enumerate(("parse_acc", "infill_acc")):
        for arm, cell in plant.items():
            if key not in cell:
                continue
            ax[j].plot(cell[key]["cycles"], cell[key]["series"], style.get(arm, "-"), lw=1.2,
                       color=COLOUR.get(arm), label=arm)
        ax[j].set_xlabel("cycle"); ax[j].set_ylabel(key)
        ax[j].set_title(f"plant guard: {key} on clean held-out configs")
        ax[j].legend(fontsize=6)
        for b_ in bounds:
            ax[j].axvline(b_, color="0.85", lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig4_prop_plant.png"), dpi=150)
    plt.close(fig)
    print(f"\nfigures -> {root}/fig1_prop_competence.png, fig2_prop_ledger.png, "
          f"fig3_prop_chunk.png, fig4_prop_plant.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="np_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        AR.fetch(a.tag)
    setup, arms = AR.load(a.tag)
    _, _, per_ar = AR.report(a.tag, tail=a.tail)
    g = gates(arms, setup)
    per = ledger(arms, setup, tail=a.tail)
    ms_ = matched_success(arms, per, tail=min(a.tail, 3))
    dec = decompose(arms)
    slot_profile(arms)
    yl = yield_table(arms, tail=a.tail)
    plant = plant_table(arms)
    os.makedirs(os.path.join(FIG, a.tag), exist_ok=True)
    if a.figures:
        figures(a.tag, setup, arms, per, plant, dec)
    with open(os.path.join(FIG, a.tag, "summary.json"), "w") as fh:
        json.dump({"tag": a.tag, "gates": g, "ledger": per, "decompose": dec, "yield": yl,
                   "matched_success": {f"{k[0]}|era{k[1]}": val for k, val in ms_.items()}},
                  fh, indent=2)
    print(f"\nsummary -> {os.path.join(FIG, a.tag, 'summary.json')}")


if __name__ == "__main__":
    main()
