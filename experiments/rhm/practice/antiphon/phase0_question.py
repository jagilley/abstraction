"""PHASE 0 — size the QUESTION PORT offline, before any GPU.

`antiphon` asks whether question-choice can be an outer-loop action: does question *quality*
move climbing speed at matched priced budget, and can the learner's own value signal grade
questions. The design step (SPEC) leaves the question parameterization open. This file sizes
the one parameterization that has a computable lever on this substrate — **the posed instance
as the learner's action** — against the banked donor logs and the DGP's own arithmetic, and
says how much headroom exists before a GPU is spent (`ostinato`'s precedent; `crescendo`'s
`phase0_l4.py` is the direct model, and its `buildable()` set arithmetic is reused verbatim).

THE CHAIN THE QUESTION KNOB ACTS ON. Every cycle the arm poses `n_pr=64` fresh instances of the
era's damage cell, solves them, and mines up to `mine_cap=8` of the SOLVED answers: the reader
parses each answer and the level-l span containing the era's cell is handed to `miners[l]`. A
key reaching `mine_support=3` becomes a candidate table entry; `Miner.build` then ratchets it
against the operative lower table. So:

    posed instance -> its clean derivation's level-l constituent at the mining node
                   -> (after repair + parse) an observation of that key
                   -> at support 3, a candidate entry
                   -> buildable iff both halves are rows of the lower table  (the r^2 wall)

which makes "which instance do I pose" a *direct* lever on which rules get mined — and the
lever is exact on a known DGP, so the ceiling arm is computable.

WHAT THIS FILE MEASURES.
  [0] the DGP's own PARSE AMBIGUITY at each era's mining node — the fraction of clean
      derivations whose exact level-1-feature tuple at that span is NOT a true table row.
      This is a property of the grammar, not of the learner, and it is the aleatoric channel
      the SPEC's A3' wanted to install: it is already installed.
  [1] the realized coverage/precision from the banked logs (cr3_s0 / ma_s0 / cd_s0).
  [2] a random-draw model of the observation stream, calibrated against [1].
  [3] menu reach vs K — how deep into the key distribution a menu of K candidates lets a
      selector reach — and the budget-matched choice of K.
  [4] the r^2 payoff curve: buildable true L4 as a function of L3 recall, by the substrate's
      own `buildable()` arithmetic.
  [5] the predicted windows for the round, under a stated delivery-fidelity band.
  [6] the difficulty alphabet (d* strata per era) that the arms' quota pins, and the menu's
      wall-clock cost.

No GPU, no Modal, no substrate. Run from experiments/:
    PYTHONPATH=. python3 rhm/practice/antiphon/phase0_question.py
"""

import collections
import json
import math
import os
import time

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, sample_derivations
from rhm.practice.ratchet import macros as MC
from rhm.practice.crystallize.units import corrupt_hier

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)

V, S, DEPTH, M = 8, 2, 6, 2
RULE_SEED = 0
SUPPORT = 3                       # cfg["mine_support"]
MINE_CAP = 8                      # cfg["mine_cap"] — observations per cycle, the volume pin
N_PR = 64                         # cfg["n_pr"]     — posed instances per cycle

# `crescendo`'s ladder: era -> (damage level, damage node, cycles)
LADDER = [(1, 25, 60), (2, 12, 50), (3, 6, 70), (4, 3, 12), (5, 1, 9)]

# the offline corpus: fetched copies of the donor runs
CORPUS = [
    ("cr3_s0", os.path.join(PRACTICE, "crescendo", "figures", "cr3_s0")),
    ("ma_s0", os.path.join(PRACTICE, "maestro", "figures", "ma_s0")),
    ("cd_s0", os.path.join(PRACTICE, "conductor", "figures", "cd_s0")),
]

N_MARGINAL = 600_000              # derivations sampled to estimate the key marginals


# --------------------------------------------------------------------------- #
# the substrate's own arithmetic
# --------------------------------------------------------------------------- #

def flats_of(table):
    return {tuple(int(x) for x in r) for r in table["flat"]}


def buildable(keys, lower_flats, s=S):
    """`Miner.build`'s ratchet test as a set operation (crescendo/phase0_l4.py, verbatim)."""
    if not keys:
        return []
    half = len(next(iter(keys))) // s
    out = []
    for k in keys:
        k = tuple(int(x) for x in k)
        if all(k[i * half:(i + 1) * half] in lower_flats for i in range(s)):
            out.append(k)
    return out


def mining_node(era_level, era_node, level):
    """The span the arm loop observes at `level` for an era damaging (era_level, era_node) —
    `run_arm`'s own arithmetic: node = (era_node * s**(era_level-1)) // s**(level-1)."""
    span = S ** (level - 1)
    return (era_node * S ** (era_level - 1)) // span


def pge(lam, k=SUPPORT):
    """P(Poisson(lam) >= k)."""
    tot, term = 0.0, math.exp(-lam)
    for i in range(k):
        tot += term
        term *= lam / (i + 1)
    return 1.0 - tot


# --------------------------------------------------------------------------- #
def key_marginals(rules, ib, truth, nodes_by_level, n=N_MARGINAL, seed=0):
    """The distribution over level-1-feature tuples at each mining node, under the DGP's own
    draw (`_sample_pool`'s: uniform root, uniform rules). Split into true / junk by the true
    table. The junk mass is the grammar's PARSE AMBIGUITY: `build_inverse_maps` is last-writer-
    wins, so distinct level-1 features can share a leaf tuple and a legal derivation can parse
    to a tuple that is no table row at all."""
    rng = np.random.default_rng(seed + 4242)
    roots = rng.integers(0, V, size=n)
    lv = sample_derivations(rules, roots, S, rng)
    pf = MC.exact_features(lv, ib, V, S)
    out = {}
    for level, node in nodes_by_level.items():
        span = S ** (level - 1)
        keys = [tuple(int(x) for x in r) for r in pf[:, node * span:(node + 1) * span]]
        cnt = collections.Counter(keys)
        tl = flats_of(truth[level])
        p_true = {k: c / n for k, c in cnt.items() if k in tl}
        p_junk = {k: c / n for k, c in cnt.items() if k not in tl}
        out[level] = {"node": node, "p_true": p_true, "p_junk": p_junk,
                      "n_true_seen": len(p_true), "n_true_total": len(tl),
                      "n_junk_seen": len(p_junk),
                      "mass_true": float(sum(p_true.values()))}
    return out


def realized_from_logs(truth):
    """Per-cycle observation state at each level from the donors' own miner logs, plus the
    truth overlap of the at-support key sets (`keys_at_support` is logged at support 3)."""
    arms = []
    for tag, root in CORPUS:
        if not os.path.isdir(root):
            continue
        for arm in sorted(os.listdir(root)):
            d = os.path.join(root, arm)
            if not os.path.isfile(os.path.join(d, "results.json")):
                continue
            with open(os.path.join(d, "results.json")) as f:
                res = json.load(f)
            log = res["log"]
            cycles, eras = log["cycle"], log["era"]
            mn, gy = log["miner"], log.get("gy")
            commits = {int(e["level"]): int(e["cycle"])
                       for e in res["events"] if e["kind"] == "commit"}
            traj = []
            for i, c in enumerate(cycles):
                row = {"cycle": c, "era": eras[i]}
                for level in (2, 3, 4):
                    st = ((gy[i] or {}) if level == 4 and gy
                          else ((mn[i] or {}).get(str(level)) or {}))
                    if not st:
                        continue
                    ks = {tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])}
                    tl = flats_of(truth[level])
                    row[f"L{level}"] = {
                        "n_obs": int(st["n_obs"]), "n_distinct": int(st["n_distinct"]),
                        "at_sup": int((st.get("n_at_support") or {}).get(str(SUPPORT), 0)),
                        "true_at_sup": len(ks & tl),
                        "recall": len(ks & tl) / len(tl),
                        "precision": (len(ks & tl) / len(ks)) if ks else None,
                    }
                traj.append(row)
            arms.append({"tag": tag, "arm": arm, "commits": commits,
                         "n_cycles": len(cycles), "traj": traj})
    return arms


def model_vs_realized(marg, arms, level):
    """Calibration: the random-draw model's E[keys at support 3] after N observations against
    what the donors actually logged at the same N. The model treats each mined observation as
    an independent draw from the marginal, which ignores the repair and the reader — so the
    ratio it misses by IS the round's delivery-fidelity prior."""
    p_true = marg[level]["p_true"]
    p_junk = marg[level]["p_junk"]
    rows = []
    for a in arms:
        for t in a["traj"]:
            r = t.get(f"L{level}")
            if not r or r["n_obs"] < 100:
                continue
            n = r["n_obs"]
            rows.append({
                "tag": a["tag"], "arm": a["arm"], "cycle": t["cycle"], "n_obs": n,
                "true_model": sum(pge(n * v) for v in p_true.values()),
                "true_real": r["true_at_sup"],
                "junk_model": sum(pge(n * v) for v in p_junk.values()),
                "junk_real": r["at_sup"] - r["true_at_sup"],
            })
    return rows


def reach_vs_k(marg, level, ks=(64, 256, 512, 1024, 2048, 4096, 8192, 16384)):
    """How many true keys a menu of K candidates puts within reach — a key k appears in a menu
    of K with probability 1-(1-p_k)**K, and 'reachable' is that probability exceeding 1/2."""
    p_true = marg[level]["p_true"]
    return {int(k): int(sum(1 for v in p_true.values() if 1 - (1 - v) ** k > 0.5)) for k in ks}


def r2_curve(truth, seed=0, draws=40):
    """buildable TRUE L4 as a function of L3 recall, by the substrate's own `buildable()`."""
    t3 = sorted(flats_of(truth[3]))
    t4 = sorted(flats_of(truth[4]))
    rng = np.random.default_rng(seed)
    out = []
    for k in (5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 56):
        vals = []
        for _ in range(draws):
            sub = {t3[i] for i in rng.permutation(len(t3))[:k]}
            vals.append(len(buildable(t4, sub)))
        out.append({"k": k, "recall": k / len(t3), "mean": float(np.mean(vals)),
                    "sd": float(np.std(vals)), "r2_816": (k / len(t3)) ** 2 * len(t4)})
    return out


def difficulty_alphabet(rules, n=2048, seed=1):
    """The d* histogram per era cell — the alphabet the arms' difficulty quota pins — and the
    wall-clock cost of drawing a menu of that size."""
    out = []
    for level, node, _cyc in LADDER:
        rng = np.random.default_rng(seed + level)
        t0 = time.time()
        r = rng.integers(0, V, size=n)
        lv = sample_derivations(rules, r, S, rng)
        xd = corrupt_hier(lv, rules, DEPTH, V, M, S, level, [node], rng)
        d = nearest_derivation_cost(rules, xd, r, S)
        dt = time.time() - t0
        cnt = collections.Counter(int(x) for x in d)
        out.append({"era_level": level, "era_node": node, "n": n,
                    "d_hist": {str(k): int(v) for k, v in sorted(cnt.items())},
                    "broken_frac": float((d > 0).mean()),
                    "seconds_for_n": dt})
    return out


# --------------------------------------------------------------------------- #
def main():
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    truth = MC.true_tables(rules, DEPTH, S, V, M, 5)
    ib = build_inverse_maps(rules)[-1]
    sizes = {str(l): {"rows": int(truth[l]["child"].shape[0]),
                      "distinct_flats": len(flats_of(truth[l]))} for l in (2, 3, 4, 5)}

    print("=" * 82)
    print("PHASE 0 — the question port, sized offline")
    print("=" * 82)
    print("\n[0a] the DGP's own levels")
    for l in (2, 3, 4, 5):
        print(f"     L{l}: {sizes[str(l)]['rows']:>7d} rows   "
              f"{sizes[str(l)]['distinct_flats']:>7d} distinct flat tuples")

    # the mining nodes the arm loop actually uses, per era
    per_era_nodes = {f"L{lv}n{nd}": {ell: mining_node(lv, nd, ell) for ell in range(2, 5)}
                     for lv, nd, _c in LADDER}
    print("\n[0b] mining nodes per era (run_arm's own span arithmetic)")
    for k, vv in per_era_nodes.items():
        print(f"     era {k:<8} " + "  ".join(f"L{e}->node {n}" for e, n in sorted(vv.items())))
    # eras 1-4 share one node set; era 5 (the last consumption era, 9 cycles) does not. The
    # earning eras are 1-3, so the sizing below uses THEIR nodes and states the exception.
    nodes = {ell: mining_node(*LADDER[1][:2], ell) for ell in range(2, 5)}
    same = [k for k, vv in per_era_nodes.items() if vv == nodes]
    print(f"     sizing uses the earning eras' node set {nodes} (shared by {', '.join(same)}); "
          f"era L5n1 mines a different set and is 9 consumption cycles")

    print("\n[0c] PARSE AMBIGUITY at the mining node — the aleatoric channel, already installed")
    marg = key_marginals(rules, ib, truth, nodes)
    for l in (2, 3, 4):
        m = marg[l]
        print(f"     L{l} node{m['node']}: true keys reachable {m['n_true_seen']}/"
              f"{m['n_true_total']}   junk keys {m['n_junk_seen']}   "
              f"mass on true {m['mass_true']:.3f}  ->  junk mass {1 - m['mass_true']:.3f}")
        fr = np.array(sorted(m["p_true"].values()))
        print(f"        true-key freq: min {fr[0]:.6f}  q25 {np.quantile(fr, .25):.6f}  "
              f"med {np.median(fr):.6f}  q75 {np.quantile(fr, .75):.6f}  max {fr[-1]:.6f}   "
              f"(uniform would be {1 / m['n_true_total']:.6f})")

    print("\n[1] REALIZED, from the banked logs — at each level's commit and at end of run")
    arms = realized_from_logs(truth)
    print(f"     {'tag/arm':<26} {'lvl':>3} {'cyc':>4} {'N_obs':>6} {'@sup':>5} {'true':>5} "
          f"{'recall':>7} {'prec':>6}")
    for a in arms:
        if a["arm"].startswith(("yoked", "given", "ceiling")):
            continue
        marks = sorted(set(list(a["commits"].values()) + [a["traj"][-1]["cycle"]]))
        for t in a["traj"]:
            if t["cycle"] not in marks:
                continue
            for l in (3, 4):
                r = t.get(f"L{l}")
                if not r or r["n_obs"] == 0:
                    continue
                print(f"     {a['tag'] + '/' + a['arm']:<26} {l:>3} {t['cycle']:>4} "
                      f"{r['n_obs']:>6} {r['at_sup']:>5} {r['true_at_sup']:>5} "
                      f"{r['recall']:>7.3f} "
                      f"{(r['precision'] if r['precision'] is not None else float('nan')):>6.3f}")

    print("\n[2] CALIBRATION — random-draw model vs the logs (the delivery-fidelity prior)")
    calib = {}
    for l in (2, 3, 4):
        rows = model_vs_realized(marg, arms, l)
        if not rows:
            continue
        rt = np.array([r["true_real"] / max(r["true_model"], 1e-9) for r in rows])
        rj = np.array([r["junk_real"] / max(r["junk_model"], 1e-9) for r in rows])
        calib[str(l)] = {"n": len(rows), "true_ratio_median": float(np.median(rt)),
                         "true_ratio_iqr": [float(np.quantile(rt, .25)),
                                            float(np.quantile(rt, .75))],
                         "junk_ratio_median": float(np.median(rj))}
        print(f"     L{l}: realized/model true-keys-at-support  median "
              f"{np.median(rt):.2f}  IQR [{np.quantile(rt, .25):.2f}, {np.quantile(rt, .75):.2f}]"
              f"   (junk median {np.median(rj):.2f})   n={len(rows)} cycle-snapshots")
    print("     A ratio below 1 is the gap between the DGP's clean draw and what the repair +")
    print("     the reader actually deliver. It is this round's delivery-fidelity prior and the")
    print("     smoke measures it directly (designed key vs delivered key).")

    print("\n[3] MENU REACH vs K — true keys a menu of K puts within reach (P(present) > 1/2)")
    reach = {str(l): reach_vs_k(marg, l) for l in (2, 3, 4)}
    ks = sorted(int(k) for k in reach["3"])
    print(f"     {'K':>7} " + " ".join(f"{'L' + str(l):>10}" for l in (2, 3, 4)))
    for k in ks:
        print(f"     {k:>7} " + " ".join(f"{reach[str(l)][k]:>10}" for l in (2, 3, 4)))
    budget = {2: LADDER[0][2] * MINE_CAP, 3: LADDER[1][2] * MINE_CAP, 4: LADDER[2][2] * MINE_CAP}
    print("     observation budget for the era that earns each level (cycles x mine_cap=8):")
    for l in (2, 3, 4):
        print(f"        L{l}: {budget[l]} observations -> at most {budget[l] // SUPPORT} keys "
              f"can be driven to support {SUPPORT}")

    print("\n[4] THE r^2 PAYOFF CURVE — buildable TRUE L4 vs L3 recall (substrate arithmetic)")
    curve = r2_curve(truth)
    print(f"     {'|T3 sub|':>9} {'recall':>7} {'buildable true L4':>19} {'r^2*816':>9}")
    for c in curve:
        print(f"     {c['k']:>9} {c['recall']:>7.3f} {c['mean']:>13.1f} +-{c['sd']:>4.1f} "
              f"{c['r2_816']:>9.1f}")

    print("\n[5] PREDICTED WINDOWS for the round")
    print("     delta = DELIVERY FIDELITY: the chance a posed instance's designed key is what")
    print("     the miner actually records, after the repair and the reader. It bites twice —")
    print("     a key delivered with probability delta needs SUPPORT/delta observations AND is")
    print("     only worth targeting at all if it lands sometimes, so reach scales with it too.")
    print("     The band's anchor is [2]'s L3 calibration ratio "
          f"{calib.get('3', {}).get('true_ratio_median', float('nan')):.2f} — the factor by "
          "which the donors' own")
    print("     realized coverage falls short of the DGP's clean draw at the same n_obs.")
    d_prior = calib.get("3", {}).get("true_ratio_median", 0.5)
    pred = {"delivery_band": [0.3, round(float(d_prior), 3), 0.7, 1.0], "rows": []}
    n_t3, n_t4 = len(flats_of(truth[3])), len(flats_of(truth[4]))
    for K in (1024, 2048, 4096):
        r3, r4 = reach_vs_k(marg, 3, (K,))[K], reach_vs_k(marg, 4, (K,))[K]
        for delta in pred["delivery_band"]:
            l3 = min(r3 * delta, budget[3] * delta / SUPPORT)
            l4 = min(r4 * delta, budget[4] * delta / SUPPORT)
            rec3 = l3 / n_t3
            pred["rows"].append({"K": K, "delta": delta, "L3_at_sup": l3,
                                 "L3_recall": rec3, "L4_at_sup": l4,
                                 "buildable_true_L4": rec3 ** 2 * n_t4})
    print(f"     {'K':>6} {'delta':>6} {'L3@sup':>7} {'L3 recall':>10} {'L4@sup':>7} "
          f"{'buildable true L4':>18}")
    for r in pred["rows"]:
        print(f"     {r['K']:>6} {r['delta']:>6.2f} {r['L3_at_sup']:>7.1f} "
              f"{r['L3_recall']:>10.3f} {r['L4_at_sup']:>7.1f} "
              f"{r['buildable_true_L4']:>18.1f}")
    print("     incumbent, measured (cr3_s0/outer_yield_m4 at its L3 commit c92): "
          "L3@sup 28, true 10, recall 0.179 -> buildable true L4 ~23; the committed L4 book "
          "was 5 entries / 2 true.")

    print("\n[6] THE DIFFICULTY ALPHABET (d* per era cell) and the menu's cost")
    diff = difficulty_alphabet(rules)
    for d in diff:
        h = d["d_hist"]
        print(f"     era L{d['era_level']}n{d['era_node']}: strata "
              f"{sorted(int(k) for k in h)}   broken {d['broken_frac']:.3f}   "
              f"draw of {d['n']} took {d['seconds_for_n']:.3f}s")

    out = {"sizes": sizes, "support": SUPPORT, "mine_cap": MINE_CAP, "n_pr": N_PR,
           "ladder": LADDER, "mining_nodes": nodes,
           "per_era_nodes": {k: {str(a): b for a, b in v.items()} for k, v in per_era_nodes.items()},
           "ambiguity": {str(l): {k: v for k, v in marg[l].items()
                                  if k not in ("p_true", "p_junk")} for l in (2, 3, 4)},
           "true_key_freqs": {str(l): sorted(marg[l]["p_true"].values()) for l in (2, 3, 4)},
           "realized": arms, "calibration": calib, "reach": reach,
           "obs_budget": {str(k): v for k, v in budget.items()},
           "r2_curve": curve, "predicted": pred, "difficulty": diff}
    with open(os.path.join(HERE, "phase0.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'phase0.json')}")
    return out


if __name__ == "__main__":
    main()
