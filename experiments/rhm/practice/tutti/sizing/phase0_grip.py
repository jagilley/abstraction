"""PHASE 0 — the DEEP-ERA QUESTION GRIP, sized offline.

`antiphon` made question-choice an outer-loop action and measured its delivered dose as
**era-bounded**: 0.37-0.54 in era 1, ~0 in eras 4-5, "because there the damage cell swallows the
whole span". The README's own caveat says question-construction machinery for the deep eras does
not exist. Jasper's hypothesis on that: *"I suspect that deep question-construction may be a
tough problem; it's possible that question-asking is a primarily local/one-level-up type of
notion. But I could be misunderstanding, so feel free to run whatever empirical determinators
you like."*

THIS FILE IS THE DETERMINATOR. It measures, per era and per level, what fraction of the mined
key a selector could control — the GRIP — under each candidate parameterization of a question,
on the DGP's own geometry and the banked key streams. Nothing here decides in advance whether
grip falls with depth; the table decides it.

THE TWO GRIPS, SEPARATED. `antiphon`'s lever has two parts and the round's headline conflates
them:

  * STRUCTURAL grip — the blocks of the mined key that the posed instance fixes and the repair
    CANNOT touch (`questions.py::half_keys`'s `clean_offsets`). Guaranteed, not probabilistic.
  * DELIVERED dose — the probability that the WHOLE mined key equals the key the question
    designed (`log["q"]["dose_hit"]`). This includes the half the repair authors.

They have different laws, and the difference is the finding.

WHAT IT MEASURES.
  [0] the structural grip law: `target_geometry` swept over era x `max_macro_level`.
  [1] grip in blocks vs grip in bits, per target level.
  [2] the delivered dose, measured per era per arm on `an_s0`'s own question log, and the law
      it obeys.
  [3] GRIP x ERA x PARAMETERIZATION — the table the determinator exists to produce.
  [4] node choice, priced: what moving the demand support would actually buy (it violates the
      arc's one hard norm, so it is measured, not proposed).
  [5] menu reach vs K per level (`antiphon` Phase 0(d) extended to L5/L6).
  [6] the damage-schedule counterfactual: if the deep-era cell were narrower, does grip return,
      and what does it cost in demand coverage?

No GPU, no Modal, no substrate. Writes `phase0_grip.json` beside this file.

Run from experiments/:  PYTHONPATH=. python3 rhm/practice/tutti/sizing/phase0_grip.py
"""

import collections
import json
import math
import os

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import sample_derivations
from rhm.practice.ratchet import macros as MC
from rhm.practice.antiphon.questions import target_geometry

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(os.path.dirname(HERE))

V, S, DEPTH, M = 8, 2, 6, 2
RULE_SEED = 0
SUPPORT = 3
MINE_CAP = 8
K_MENU = 2048                     # `antiphon`'s budget-matched menu size

LADDER = [(1, 25, 60), (2, 12, 50), (3, 6, 70), (4, 3, 12), (5, 1, 9)]
AN_S0 = os.path.join(PRACTICE, "antiphon", "figures", "an_s0")
N_MARG = 1_000_000


def enc(a, v=V):
    p = np.int64(v) ** np.arange(a.shape[1], dtype=np.int64)
    return (a.astype(np.int64) * p).sum(1)


def flats_of(table):
    return {tuple(int(x) for x in r) for r in table["flat"]}


def pge(lam, k=SUPPORT):
    tot, term = 0.0, math.exp(-lam)
    for i in range(k):
        tot += term
        term *= lam / (i + 1)
    return 1.0 - tot


# --------------------------------------------------------------------------- #
def main():
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    ib = build_inverse_maps(rules)[-1]
    truth_tab = MC.true_tables(rules, DEPTH, S, V, M, 5)
    sizes = {1: V}
    sizes.update({ell: len(flats_of(truth_tab[ell])) for ell in (2, 3, 4, 5)})
    sizes[6] = 13_056_344_064          # from phase0_l5.py [0], by inclusion-exclusion
    out = {"world": {"v": V, "s": S, "depth": DEPTH, "m": M},
           "level_sizes": {str(k): int(v) for k, v in sizes.items()}}

    print("=" * 92)
    print("PHASE 0 — THE DEEP-ERA QUESTION GRIP")
    print("=" * 92)

    # ---- [0] the structural grip law --------------------------------------------------- #
    print("\n[0] STRUCTURAL GRIP = |clean_offsets| / span, from `questions.py::target_geometry`.")
    print("    The mined level is min(max_macro_level, era_level + 1); the damage cell covers")
    print("    s^(era_level-1) of its s^(level-1) blocks. Swept over the level cap:")
    print(f"    {'era':<8} " + "  ".join(f"{'maxl='+str(x):>22}" for x in (3, 4, 5, 6)))
    grid = []
    for lv, nd, _c in LADDER:
        cells, row = [], {"era": f"L{lv}n{nd}", "era_level": lv, "era_node": nd}
        for maxl in (3, 4, 5, 6):
            g = target_geometry(lv, nd, maxl, S)
            grip = len(g["clean_offsets"]) / g["span"]
            row[f"maxl{maxl}"] = {"level": g["level"], "span": g["span"], "node": g["node"],
                                  "n_clean": len(g["clean_offsets"]), "grip": grip}
            cells.append(f"L{g['level']} {len(g['clean_offsets'])}/{g['span']} = {grip:.3f}")
        grid.append(row)
        print(f"    L{lv}n{nd:<5} " + "  ".join(f"{c:>22}" for c in cells))
    print("\n    THE LAW. With level = era_level + 1 the clean block count is")
    print("    s^(era_level) - s^(era_level-1), so grip = (s-1)/s = 0.500 EXACTLY, at every")
    print("    depth. With level = maxl < era_level + 1 it is 1 - s^(era_level - maxl), which is")
    print("    <= 0 the moment era_level >= maxl. So the eras-4/5 collapse in `antiphon` is a")
    print("    property of the LEVEL CAP (max_macro_level = 4 against era levels 4 and 5), not")
    print("    of depth: at maxl = 6 the structural grip is 0.500 in all five eras.")
    out["grip_grid"] = grid

    # ---- [1] blocks vs bits ------------------------------------------------------------- #
    print("\n[1] the same grip in BITS — the clean part of a level-l key is one level-(l-1) flat,")
    print("    so the question pins log2|T_(l-1)| of the key's log2|T_l| bits:")
    print(f"    {'level':>5} {'|T_l|':>16} {'|T_(l-1)|':>14} {'grip (blocks)':>14} "
          f"{'grip (bits)':>12} {'residual keys':>14}")
    bits = []
    for ell in range(2, 7):
        gb = 0.5
        gbit = math.log2(sizes[ell - 1]) / math.log2(sizes[ell])
        resid = sizes[ell] / sizes[ell - 1]
        bits.append({"level": ell, "size": sizes[ell], "lower": sizes[ell - 1],
                     "grip_blocks": gb, "grip_bits": gbit, "residual_keys": resid})
        print(f"    {ell:>5} {sizes[ell]:>16,} {sizes[ell-1]:>14,} {gb:>14.3f} {gbit:>12.3f} "
              f"{resid:>14,.0f}")
    print("    Grip in blocks is flat at 1/2. Grip in bits falls 0.79 -> 0.53 and converges to")
    print("    1/2 — a slow decline, not a collapse. But the RESIDUAL — the key space the ask")
    print("    does not pin, which the repair must author — grows as |T_l|/|T_(l-1)|, i.e.")
    print("    m^(s^(l-2)): 4, 15, 252, 63,435. That is the quantity [2] measures directly.")
    out["grip_bits"] = bits

    # ---- [2] the delivered dose, measured ----------------------------------------------- #
    print("\n[2] DELIVERED DOSE — measured on `an_s0`'s own question log "
          "(`log['q']['dose_hit']`:")
    print("    the fraction of a cycle's mined observations whose key equals the DESIGNED key)")
    dose = collections.defaultdict(list)
    hasclean = collections.defaultdict(list)
    dmean = collections.defaultdict(list)
    tgt = {}
    arms = []
    if os.path.isdir(AN_S0):
        for arm in sorted(os.listdir(AN_S0)):
            p = os.path.join(AN_S0, arm, "results.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                res = json.load(f)
            q, era = res["log"]["q"], res["log"]["era"]
            per = collections.defaultdict(list)
            for i, row in enumerate(q):
                e = era[i]
                if row.get("dose_hit") is not None:
                    per[e].append(float(row["dose_hit"]))
                    dose[e].append(float(row["dose_hit"]))
                hasclean[e].append(bool(row.get("has_clean")))
                dmean[e].append(float(row.get("d_mean") or 0))
                tgt[e] = (row.get("target_level"), row.get("target_node"))
            arms.append({"arm": arm,
                         "dose_by_era": {str(e): float(np.mean(v)) for e, v in per.items()}})
            print(f"    {arm:<16} " + "  ".join(
                f"e{e}:{np.mean(per[e]):.3f}" if per.get(e) else f"e{e}: -"
                for e, _, _ in LADDER))
    print(f"\n    {'era':<8} {'target':>8} {'has_clean':>10} {'dose (mean over 7 arms)':>24} "
          f"{'1/|T_(l-1)|':>12} {'kappa = dose*|T_(l-1)|':>23}")
    law = []
    for lv, nd, _c in LADDER:
        e = lv
        tl = tgt.get(e, (None, None))[0]
        d = float(np.mean(dose[e])) if dose[e] else float("nan")
        lower = sizes[tl - 1] if tl else None
        kap = d * lower if lower else float("nan")
        law.append({"era": f"L{lv}n{nd}", "target_level": tl,
                    "has_clean": bool(np.mean(hasclean[e]) > 0.5), "dose": d,
                    "one_over_lower": (1 / lower if lower else None), "kappa": kap,
                    "d_star_mean": float(np.mean(dmean[e]))})
        print(f"    L{lv}n{nd:<5} {'L'+str(tl):>8} {str(bool(np.mean(hasclean[e])>0.5)):>10} "
              f"{d:>24.4f} {1/lower:>12.4f} {kap:>23.2f}")
    print("    THE DOSE LAW. Where the question has a clean half (eras 1-3), the dose tracks")
    print("    kappa / |T_(l-1)| with kappa ~ 2-4: the ask fixes its half exactly and the repair")
    print("    reproduces the other half at 2-4x chance. So the WHOLE-KEY grip falls at the rate")
    print("    the level ladder grows — doubly exponentially — while the guaranteed half does")
    print("    not move at all. Extrapolated with kappa = 2.5:")
    for ell, nm in [(5, "era 4 at maxl>=5"), (6, "era 5 at maxl=6")]:
        print(f"        L{ell} ({nm}): dose ~ 2.5 / {sizes[ell-1]:,} = "
              f"{2.5/sizes[ell-1]:.2e}")
    out["dose"] = {"per_arm": arms, "law": law,
                   "extrapolated": {str(e): 2.5 / sizes[e - 1] for e in (5, 6)}}

    # ---- [4] node choice, priced (computed before [3] so the table can cite it) ---------- #
    print("\n[4] NODE CHOICE, PRICED. It violates the arc's one hard norm (the demand support")
    print("    must not move), so it is measured, not proposed: per-node true-key marginals and")
    print("    the at-support yield each node would give at the run's own N = 1,600.")
    rng = np.random.default_rng(99)
    roots = rng.integers(0, V, size=N_MARG)
    pf = MC.exact_features(sample_derivations(rules, roots, S, rng), ib, V, S)
    codes_true = {ell: set(enc(np.unique(truth_tab[ell]["flat"], axis=0)).tolist())
                  for ell in (2, 3, 4, 5)}
    era_node_of = {2: 12, 3: 6, 4: 3, 5: 1}
    node_tab, marg_era = [], {}
    print(f"    {'level':>5} {'#nodes':>7} {'era node':>9} {'E@1600 (era node)':>18} "
          f"{'best node':>10} {'E@1600 (best)':>14} {'gain':>7}")
    for ell in (2, 3, 4, 5):
        span, nn = S ** (ell - 1), 32 // S ** (ell - 1)
        ee = []
        for node in range(nn):
            cnt = collections.Counter(enc(pf[:, node * span:(node + 1) * span]).tolist())
            pt = np.array([c / N_MARG for k, c in cnt.items() if k in codes_true[ell]])
            ee.append((node, float(sum(pge(1600 * p) for p in pt)), float(pt.sum())))
            if node == era_node_of[ell]:
                marg_era[ell] = {k / 1.0: c / N_MARG for k, c in cnt.items()
                                 if k in codes_true[ell]}
        en = era_node_of[ell]
        best = max(ee, key=lambda r: r[1])
        node_tab.append({"level": ell, "n_nodes": nn, "era_node": en,
                         "E_era_node": ee[en][1], "best_node": best[0], "E_best": best[1],
                         "gain": best[1] / ee[en][1] if ee[en][1] else float("nan"),
                         "per_node": [{"node": r[0], "E_at_1600": r[1], "mass_true": r[2]}
                                      for r in ee]})
        g = best[1] / ee[en][1] if ee[en][1] > 1e-9 else float("nan")
        print(f"    {ell:>5} {nn:>7} {en:>9} {ee[en][1]:>18.3g} {best[0]:>10} {best[1]:>14.3g} "
              f"{g:>6.2f}x")
    print("    Node choice buys ~1.17x at L3, ~1.04x at L4, and at L5 it moves a quantity that is")
    print("    ~1e-5 keys either way — there are only 2 L5 nodes and both are empty at this")
    print("    budget. The op that costs the arc its one hard norm is worth ~4% at the rung it")
    print("    was proposed for and nothing at the rung the second extension needs.")
    out["node_choice"] = node_tab
    del pf

    # ---- [5] menu reach vs K ------------------------------------------------------------ #
    print(f"\n[5] MENU REACH vs K (`antiphon` Phase 0(d), extended): a key is reachable if a menu")
    print(f"    of K candidates contains it with p > 1/2, i.e. 1 - (1-p_k)^K > 1/2.")
    print(f"    {'K':>7} " + " ".join(f"{'L'+str(e)+' (of '+format(sizes[e],',')+')':>22}"
                                      for e in (2, 3, 4, 5)))
    reach = {}
    for K in (256, 1024, 2048, 4096, 16384, 65536):
        cells = []
        for ell in (2, 3, 4, 5):
            pv = list(marg_era[ell].values())
            n = int(sum(1 for p in pv if 1 - (1 - p) ** K > 0.5))
            reach.setdefault(str(K), {})[str(ell)] = n
            cells.append(f"{n:>22,}")
        print(f"    {K:>7} " + " ".join(cells))
    print("    At K = 2048 (the round's budget-matched choice) reach saturates at L2/L3 and is")
    print("    ~150 of 816 at L4; at L5 it is ZERO — not one L5 key is more likely than not to")
    print("    appear in a menu of 2048. L5 reach first becomes non-zero at K ~ 65,000 (4 keys),")
    print("    and the era's whole observation budget can only drive ~186 keys to support anyway,")
    print("    so at L5 the menu is not the binding constraint either.")
    out["reach_vs_k"] = reach

    # ---- [3] the parameterization table ------------------------------------------------- #
    print("\n[3] GRIP x ERA x PARAMETERIZATION — the determinator's table.")
    print("    grip = the fraction of the mined key's level-1 blocks the selector's choice fixes")
    print("    EXACTLY, before the repair. 'dose' = P(the whole mined key is the designed key),")
    print("    measured in eras 1-3 and extrapolated (kappa/|T_(l-1)|, [2]) beyond. 'norm' = does")
    print("    it move the demand support (the SPEC's one hard norm)?")
    dose_meas = {lv: (float(np.mean(dose[lv])) if dose[lv] else float("nan"))
                 for lv, _, _ in LADDER}

    def dose_at(level):
        return 2.5 / sizes[level - 1]

    PARAMS = [
        ("cell_instance (current)", "n",
         "choose which instance of the era's own damage cell to pose"),
        ("cell_instance, cap lifted", "n",
         "the same selector at max_macro_level = era_level + 1"),
        ("span_half", "y (within level)",
         "choose which of the s halves of the mined span the cell occupies"),
        ("probe_completion", "n",
         "the SPEC's bisection proper: the poser fixes the whole clean derivation"),
        ("node_choice", "Y",
         "choose the damage node among the level's s^(depth-l) siblings"),
        ("rehearsal_episode", "n",
         "re-pose a previously solved instance (`woodshed`'s form)"),
        ("archive_slice", "n",
         "choose which slice of a fixed archive to re-read (`reread`'s form)"),
        ("narrowed_deep_cell", "n",
         "keep the cap, shrink the deep-era cell to level maxl - 1"),
    ]
    rows = []
    print(f"\n    {'parameterization':<28} {'norm':>16} " +
          " ".join(f"{'era'+str(e):>13}" for e in range(1, 6)))
    for name, norm, _why in PARAMS:
        cells, vals = [], {}
        for lv, nd, _c in LADDER:
            if name == "cell_instance (current)":
                g = target_geometry(lv, nd, 4, S)
                grip = len(g["clean_offsets"]) / g["span"]
                d = dose_meas[lv]
            elif name == "cell_instance, cap lifted":
                g = target_geometry(lv, nd, lv + 1, S)
                grip = len(g["clean_offsets"]) / g["span"]
                d = dose_meas[lv] if lv <= 3 else dose_at(g["level"])
            elif name == "span_half":
                g = target_geometry(lv, nd, lv + 1, S)
                grip = len(g["clean_offsets"]) / g["span"]
                d = dose_meas[lv] if lv <= 3 else dose_at(g["level"])
            elif name == "probe_completion":
                g = target_geometry(lv, nd, lv + 1, S)
                grip = 1.0                                  # the whole key is DESIGNED
                d = dose_meas[lv] if lv <= 3 else dose_at(g["level"])
            elif name == "node_choice":
                g = target_geometry(lv, nd, 4, S)
                grip = len(g["clean_offsets"]) / g["span"]
                d = dose_meas[lv]
            elif name in ("rehearsal_episode", "archive_slice"):
                grip = 1.0                                  # a stored episode fixes every block
                d = 1.0
            else:                                           # narrowed_deep_cell
                eff = min(lv, 3)                            # cell shrunk to maxl - 1 = 3
                g = target_geometry(eff, (nd * S ** (lv - 1)) // S ** (eff - 1), 4, S)
                grip = len(g["clean_offsets"]) / g["span"]
                d = dose_meas[min(lv, 3)]
            vals[f"era{lv}"] = {"grip": grip, "dose": d}
            cells.append(f"{grip:.2f} / {d:.1e}")
        rows.append({"parameterization": name, "moves_support": norm, **vals})
        print(f"    {name:<28} {norm:>16} " + " ".join(f"{c:>13}" for c in cells))
    print("\n    each cell is  grip / dose.  Reach and price, which the grip number does not")
    print("    carry, per parameterization:")
    notes = {
        "cell_instance (current)": (
            "reach = the level's key set restricted to the era's node; grip 0 in eras 4-5 is the "
            "CAP, not depth ([0]). Price: 0 (it is the incumbent)."),
        "cell_instance, cap lifted": (
            "restores grip 0.500 in every era; needs an L5/L6 miner, whose arrival cost "
            "phase0_l5.py [7] prices at 32-135x the run's observation budget."),
        "span_half": (
            "grip unchanged; doubles the reachable half-key set per rung by letting either half "
            "be the fixed one. Moves the damage node WITHIN the level — a weaker violation than "
            "node choice, and the only parameterization that widens reach without leaving the "
            "era's own cell."),
        "probe_completion": (
            "grip 1.0 by construction (the poser writes the clean derivation) but it is the "
            "DESIGNED key, not the delivered one; the dose column is the same measured number, "
            "so this parameterization renames the lever rather than strengthening it. This is "
            "what `q_bisect` already does."),
        "node_choice": (
            "priced in [4]: 1.16x at L3, 1.04x at L4, 1.00x at L5. Costs the arc's one hard norm "
            "for ~4% at the rung it was proposed for."),
        "rehearsal_episode": (
            "grip 1.0 and dose 1.0 — a stored episode reproduces its own key — but reach is "
            "closed: only keys ALREADY observed. Its lever is not arrival but SUPPORT: it "
            "converts seen-once keys into at-support keys. Sized in [6b]."),
        "archive_slice": (
            "same shape as rehearsal at coarser grain; `reread`'s measured law caps it — the "
            "archive is renewable in competence, not vocabulary, and yield decays to zero with "
            "no re-arm at commits."),
        "narrowed_deep_cell": (
            "restores grip 0.500 in eras 4-5 with no new miner, at the cost of the eras' own "
            "demand depth — priced in [6]."),
    }
    for k, v in notes.items():
        print(f"      - {k}: {v}")
    out["parameterizations"] = rows
    out["parameterization_notes"] = notes

    # ---- [6] the damage-schedule counterfactual ----------------------------------------- #
    print("\n[6] THE DAMAGE-SCHEDULE COUNTERFACTUAL — if the deep-era cell were narrower, does")
    print("    grip return, and what does it cost in demand coverage?")
    print(f"    {'era':<8} {'cell blocks':>12} {'mined span':>11} {'grip':>6} {'d* mean':>9} "
          f"{'blocks the learner must rebuild':>32}")
    cf = []
    for lv, nd, _c in LADDER:
        g = target_geometry(lv, nd, 4, S)
        cell = S ** (lv - 1)
        cf.append({"era": f"L{lv}n{nd}", "variant": "as run", "cell_blocks": cell,
                   "span": g["span"], "grip": len(g["clean_offsets"]) / g["span"],
                   "d_star": float(np.mean(dmean[lv])), "rebuild_blocks": min(cell, g["span"])})
        print(f"    L{lv}n{nd:<5} {cell:>12} {g['span']:>11} "
              f"{len(g['clean_offsets'])/g['span']:>6.3f} {np.mean(dmean[lv]):>9.2f} "
              f"{min(cell, g['span']):>32}")
    print("    narrowed (eras 4 and 5 re-cut to a level-3 cell inside the same L4 span):")
    for lv, nd, _c in LADDER[3:]:
        eff_node = (nd * S ** (lv - 1)) // S ** 2
        g = target_geometry(3, eff_node, 4, S)
        cell = S ** 2
        cf.append({"era": f"L{lv}n{nd}", "variant": "narrowed to L3 cell", "cell_blocks": cell,
                   "span": g["span"], "grip": len(g["clean_offsets"]) / g["span"],
                   "d_star": float(np.mean(dmean[3])), "rebuild_blocks": cell})
        print(f"    L{lv}n{nd:<5} {cell:>12} {g['span']:>11} "
              f"{len(g['clean_offsets'])/g['span']:>6.3f} {np.mean(dmean[3]):>9.2f} "
              f"{cell:>32}  <- grip returns")
    print("    Grip returns exactly (0.000 -> 0.500) and costs nothing in machinery. What it")
    print(f"    costs is the eras' own demand: d* falls {np.mean(dmean[4]):.2f} -> "
          f"{np.mean(dmean[3]):.2f} in era 4 and {np.mean(dmean[5]):.2f} -> "
          f"{np.mean(dmean[3]):.2f} in era 5, and the number of blocks the learner must rebuild")
    print("    halves (8 -> 4) in era 4 and quarters (16 -> 4) in era 5. A narrowed era 4 IS")
    print("    era 3: the consumption eras exist to price coverage PAST what was earned")
    print("    (`census`/`assay` finding 3), and narrowing the cell deletes exactly that.")
    out["counterfactual"] = cf

    # ---- [6b] the support lever, sized -------------------------------------------------- #
    print("\n[6b] THE SUPPORT LEVER (what a rehearsal-shaped question would attack). The runs'")
    print("     own G-Y miner logs the at-support histogram at thresholds 1/2/3/5/10:")
    print(f"     {'tag/arm':<26} {'level':>5} {'n_obs':>7} {'>=1':>7} {'>=2':>7} {'>=3':>7} "
          f"{'>=5':>6} {'convertible (2 -> 3)':>21}")
    sup = []
    roots_ = [("cr3_s0", os.path.join(PRACTICE, "crescendo", "figures", "cr3_s0")),
              ("an_s0", AN_S0)]
    for tag, root in roots_:
        if not os.path.isdir(root):
            continue
        for arm in sorted(os.listdir(root)):
            p = os.path.join(root, arm, "results.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                res = json.load(f)
            gf = res.get("gy_final")
            if not gf:
                continue
            h = gf["n_at_support"]
            conv = int(h["2"]) - int(h["3"])
            sup.append({"tag": tag, "arm": arm, "level": int(gf["level"]),
                        "n_obs": int(gf["n_obs"]), "at1": int(h["1"]), "at2": int(h["2"]),
                        "at3": int(h["3"]), "at5": int(h["5"]), "convertible": conv})
            print(f"     {tag+'/'+arm:<26} {gf['level']:>5} {gf['n_obs']:>7} {h['1']:>7} "
                  f"{h['2']:>7} {h['3']:>7} {h['5']:>6} {conv:>21}")
    if sup:
        print(f"     At L4 a rehearsal selector re-posing the seen-twice keys would convert a")
        print(f"     median of {int(np.median([r['convertible'] for r in sup]))} keys into "
              f"at-support keys — against a median at-support-3 count of "
              f"{int(np.median([r['at3'] for r in sup]))}.")
        print("     That is the one lever that acts on SUPPORT rather than on arrival, which is")
        print("     the wall phase0_l5.py [8] measures as binding at L5. It is bounded by the")
        print("     junk mass (0.82 at L4, 0.97 at L5): most convertible keys are junk.")
    out["support_lever"] = sup

    with open(os.path.join(HERE, "phase0_grip.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'phase0_grip.json')}")
    return out


if __name__ == "__main__":
    main()
