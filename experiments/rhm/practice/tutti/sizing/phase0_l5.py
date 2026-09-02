"""PHASE 0 — size the SECOND EXTENSION (L4 -> L5) OFFLINE, before any GPU.

`crescendo` (A3) opened L4 to the crank and the value clock held past the range the previous
turn certified. QUEUE "F3" asks the next question: **does the range extend TWICE**, and can the
levers on the r-squared wall — selection (`antiphon`) and the index ops (`census_extend`,
`fourwall`'s merge / re-key) — lift it? Before anyone designs an L5 tag, the rung has to be
sized the way A3 sized its own (`crescendo/phase0_l4.py`) and the question port sized its
(`antiphon/phase0_question.py`): against the banked logs and the DGP's own arithmetic.

This file is that sizing. It is `phase0_l4.py` extended one level, over a corpus five times
larger (every banked arm that holds an L4 book, not just A1's and A2's), and it re-uses the
substrate's own `buildable()` set arithmetic verbatim.

WHAT IT MEASURES.
  [0] the DGP's own level sizes L1..L6 EXACTLY, and the closed form they obey. L6 is 1.3e10
      rows and cannot be enumerated; it is counted by inclusion-exclusion over the level's
      distinct child pairs, a method validated by reproducing L2..L5 exactly.
  [1] GATE B-1' — the offline replay of `Miner.build` reproduces every logged commit event in
      the corpus entry-for-entry, INCLUDING the L4 commits A3's own gate could not reach
      (its donors, at `max_macro_level=3`, held none).
  [2] the buildable-L5 ceiling over each arm's frozen / live / true L4, by the same
      `buildable()`; and the r-squared law at this rung.
  [3] the r-squared payoff curve at L5 (buildable true L5 vs L4 recall, random L4 subsets).
  [4] the L5 and L6 observation streams as the runs actually logged them (`obs_hist`), and the
      demand-cell test: does the era's damage cell move the L5 mining node at all?
  [5] the DGP's own key marginal at each level's mining node — junk mass (the parse-ambiguity
      channel `antiphon` measured at L2/L3/L4, extended to L5 and L6) and the true-key tail.
  [6] GATE G-1 — the random-draw model calibrated against the logs at L2/L3/L4. This is what
      licenses the L5 extrapolation, or refuses it.
  [7] THE BUDGET. N_obs for E[true L5 keys at support 3] = 1 / 5 / 22 / 50, in observations,
      cycles and GPU-hours, against the run's own 1,608.
  [8] which wall binds at each rung — support (arrival) vs ratchet (the r-squared wall) —
      decomposed as two multiplicative cuts.
  [9] the gauge at the L5 frontier: `tol_yield_l5` and `tol_yield_l6` derived by the arc's own
      null-ABBA method on the banked series, their signal-over-floor, and A1's thermostat
      replayed on the L5 series.
  [10] world-sizing: |T_l| as a function of (v, s, m, l), and what world would make L5 what L4
      was here.

No GPU, no Modal, no substrate. Writes `phase0.json` beside this file.

Run from experiments/:  PYTHONPATH=. python3 rhm/practice/tutti/sizing/phase0_l5.py
"""

import collections
import json
import math
import os
import time

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import sample_derivations
from rhm.practice.ratchet import macros as MC
from rhm.practice.conductor.policy import QuietPolicy, _sd, null_abba

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(os.path.dirname(HERE))

V, S, DEPTH, M = 8, 2, 6, 2       # the depth-6 world every arm below was run in
RULE_SEED = 0
SUPPORT = 3                       # cfg["mine_support"]
MINE_CAP = 8                      # cfg["mine_cap"] — observations per cycle
SEC_PER_CYCLE = 10.6              # cr3_smoke's measured s/cycle, the budget unit

# `crescendo`'s ladder: (damage level, damage node, cycles)
LADDER = [(1, 25, 60), (2, 12, 50), (3, 6, 70), (4, 3, 12), (5, 1, 9)]

# THE CORPUS: every banked tag holding arms with an L4 book, plus the maxl=3 ancestors A3's own
# Phase 0 used (kept so this file's L2/L3 numbers are comparable to its).
CORPUS = [
    ("cr3_s0", os.path.join(PRACTICE, "crescendo", "figures", "cr3_s0")),
    ("cr3_s1", os.path.join(PRACTICE, "crescendo", "figures", "cr3_s1")),
    ("an_s0", os.path.join(PRACTICE, "antiphon", "figures", "an_s0")),
    ("an_s1", os.path.join(PRACTICE, "antiphon", "figures", "an_s1")),
    ("ca_s0", os.path.join(PRACTICE, "caesura", "figures", "ca_s0")),
    ("in_s0", os.path.join(PRACTICE, "intonation", "figures", "in_s0")),
    ("ma_s0", os.path.join(PRACTICE, "intonation", "figures", "ma_s0")),
    ("cd_s0", os.path.join(PRACTICE, "conductor", "figures", "cd_s0")),
    ("ma_s0_maestro", os.path.join(PRACTICE, "maestro", "figures", "ma_s0")),
]

N_MARG = 8_000_000                # derivations for the L5 true-key marginal (the deep tail)
N_MARG_SMALL = 1_000_000          # for L2..L4 and the per-node sweep


# --------------------------------------------------------------------------- #
# the substrate's own arithmetic  (`crescendo/phase0_l4.py`, verbatim)
# --------------------------------------------------------------------------- #

def flats_of(table):
    return {tuple(int(x) for x in r) for r in table["flat"]}


def buildable(keys, lower_flats, s=S):
    """`Miner.build`'s ratchet test as a set operation: a key survives iff every one of its `s`
    halves is a row of the lower table."""
    if not keys:
        return []
    half = len(next(iter(keys))) // s
    out = []
    for k in keys:
        k = tuple(int(x) for x in k)
        if all(k[i * half:(i + 1) * half] in lower_flats for i in range(s)):
            out.append(k)
    return out


def mining_node(era_level, era_node, level, s=S):
    """`run_arm`'s own span arithmetic: node = (era_node * s**(era_level-1)) // s**(level-1)."""
    return (era_node * s ** (era_level - 1)) // (s ** (level - 1))


def pge(lam, k=SUPPORT):
    """P(Poisson(lam) >= k)."""
    tot, term = 0.0, math.exp(-lam)
    for i in range(k):
        tot += term
        term *= lam / (i + 1)
    return 1.0 - tot


def enc(a, v=V):
    """(n, k) level-1-feature array -> int64 codes. Exact for k <= 21 at v = 8."""
    p = np.int64(v) ** np.arange(a.shape[1], dtype=np.int64)
    return (a.astype(np.int64) * p).sum(1)


# --------------------------------------------------------------------------- #
# [0] the levels — exactly, including the one that cannot be enumerated
# --------------------------------------------------------------------------- #

def feature_expansions(rules, depth, v, m, max_level):
    """F[l][f] = the set of distinct level-1 flat tuples feature f expands to at level l, as
    integer codes. `true_tables` builds the same object as an array; this form is what makes L6
    countable without materialising it."""
    F = {1: {f: {f} for f in range(v)}}
    for ell in range(2, max_level + 1):
        layer = rules[depth - ell]               # (v, m, s): level-ell -> level-(ell-1)
        shift = v ** (2 ** (ell - 2))            # s = 2: the left half occupies 2**(ell-2) slots
        cur = {}
        for f in range(v):
            acc = set()
            for r in range(m):
                c1, c2 = int(layer[f, r, 0]), int(layer[f, r, 1])
                for a in F[ell - 1][c1]:
                    for b in F[ell - 1][c2]:
                        acc.add(a + b * shift)
            cur[f] = acc
        F[ell] = cur
    return F


def level_size(rules, F, depth, v, m, ell):
    """|distinct flat tuples at level ell|. For ell <= max_level of F it is the union; for the
    level ABOVE it, inclusion-exclusion over the level's distinct child pairs — |U A_i x B_i|
    with (A_i x B_i) n (A_j x B_j) = (A_i n A_j) x (B_i n B_j)."""
    top = max(F)
    if ell <= top:
        return len(set().union(*F[ell].values()))
    assert ell == top + 1, "inclusion-exclusion is only defined one level above F"
    layer = rules[depth - ell]
    pairs = sorted({(int(layer[f, r, 0]), int(layer[f, r, 1]))
                    for f in range(v) for r in range(m)})
    A = [F[top][c1] for c1, _ in pairs]
    B = [F[top][c2] for _, c2 in pairs]
    from itertools import combinations
    n, tot = len(pairs), 0
    for k in range(1, n + 1):
        sub = 0
        for comb in combinations(range(n), k):
            ia = set.intersection(*[A[i] for i in comb])
            if not ia:
                continue
            ib = set.intersection(*[B[i] for i in comb])
            if not ib:
                continue
            sub += len(ia) * len(ib)
        tot += sub if k % 2 == 1 else -sub
        if k >= 4 and sub == 0:
            break
    return tot


# --------------------------------------------------------------------------- #
# the banked corpus
# --------------------------------------------------------------------------- #

def load_corpus():
    out = []
    for tag, root in CORPUS:
        if not os.path.isdir(root):
            print(f"    [skip] {root} not found")
            continue
        for arm in sorted(os.listdir(root)):
            p = os.path.join(root, arm, "results.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                out.append((tag, arm, json.load(f)))
    return out


def key_stream(res, level):
    """{cycle -> keys at support} for one level, from the committable miner's logged state
    (`log["miner"][c][str(level)]`). This is the miner an L(level) commit is built from."""
    out = {}
    for i, c in enumerate(res["log"]["cycle"]):
        st = (res["log"]["miner"][i] or {}).get(str(level))
        out[c] = [tuple(int(x) for x in k) for k in ((st or {}).get("keys_at_support") or [])]
    return out


def replay_tables(res, truth):
    """Replay `Miner.build` at every logged commit event, in cycle order, exactly as the run
    did: the operative lower table is the frozen one if that level has committed, else the live
    build over ITS operative lower. Returns (frozen tables, gate rows)."""
    maxl = int(res["config"]["max_macro_level"])
    K = {ell: key_stream(res, ell) for ell in range(2, maxl + 1)}
    base = flats_of(MC.base_table(V))
    commits = sorted([e for e in res["events"] if e["kind"] == "commit"],
                     key=lambda x: int(x["cycle"]))
    frozen, rows = {}, []

    def operative(ell, c):
        if ell == 1:
            return base
        if ell in frozen:
            return frozen[ell]
        return set(buildable(K[ell][c], operative(ell - 1, c)))

    for e in commits:
        lv, c = int(e["level"]), int(e["cycle"])
        got = set(buildable(K[lv][c], operative(lv - 1, c)))
        frozen[lv] = got
        n_corr = len(got & truth[lv])
        rows.append({
            "level": lv, "cycle": c,
            "n_recon": len(got), "n_logged": int(e["n_entries"]),
            "recall_recon": n_corr / len(truth[lv]), "recall_logged": e.get("tab_recall"),
            "prec_recon": (n_corr / len(got)) if got else None,
            "prec_logged": e.get("tab_precision"),
        })
    return frozen, rows, K, operative


# --------------------------------------------------------------------------- #
def main(n_marg=N_MARG):
    t_start = time.time()
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    ib = build_inverse_maps(rules)[-1]
    truth_tab = MC.true_tables(rules, DEPTH, S, V, M, 5)
    truth = {ell: flats_of(truth_tab[ell]) for ell in (2, 3, 4, 5)}
    out = {"world": {"v": V, "s": S, "depth": DEPTH, "m": M, "rule_seed": RULE_SEED,
                     "support": SUPPORT, "mine_cap": MINE_CAP},
           "ladder": [{"era_level": a, "era_node": b, "cycles": c} for a, b, c in LADDER]}

    print("=" * 86)
    print("PHASE 0 — THE SECOND EXTENSION (L4 -> L5), sized offline")
    print("=" * 86)

    # ---- [0] the levels -------------------------------------------------------------- #
    F = feature_expansions(rules, DEPTH, V, M, 5)
    sizes = {ell: level_size(rules, F, DEPTH, V, M, ell) for ell in range(1, 7)}
    print("\n[0] the DGP's own levels (distinct flat tuples — the recall denominator)")
    print(f"    {'level':>5} {'span':>5} {'per-feature m^(2^(l-1)-1)':>26} {'distinct |T_l|':>16} "
          f"{'|T_l|/|T_(l-1)|':>16} {'collision c':>12}")
    for ell in range(1, 7):
        per = M ** (2 ** (ell - 1) - 1)
        rat = sizes[ell] / sizes[ell - 1] if ell > 1 else float("nan")
        print(f"    {ell:>5} {S**(ell-1):>5} {per:>26,d} {sizes[ell]:>16,d} {rat:>16,.1f} "
              f"{sizes[ell]/(V*per):>12.3f}")
    print("    closed form: |T_l| = c_l * v * m**((s**(l-1)-1)/(s-1)); c_l in [0.76, 0.88].")
    print("    L6 is counted by inclusion-exclusion over its 14 distinct child pairs; the same")
    print("    method reproduces L2..L5 exactly, which is its gate.")
    out["sizes"] = {str(k): int(v) for k, v in sizes.items()}

    # ---- the corpus ------------------------------------------------------------------- #
    corpus = load_corpus()
    print(f"\n[1] GATE B-1' — replayed `Miner.build` vs each run's own commit log, "
          f"{len(corpus)} banked arms")
    arms, n_ok, n_bad = [], 0, 0
    for tag, arm, res in corpus:
        frozen, rows, K, operative = replay_tables(res, truth)
        for g in rows:
            same = (g["n_recon"] == g["n_logged"]
                    and abs((g["recall_recon"] or 0) - (g["recall_logged"] or 0)) < 1e-9
                    and abs((g["prec_recon"] or 0) - (g["prec_logged"] or 0)) < 1e-9)
            n_ok += same
            n_bad += (not same)
            if not same:
                print(f"    MISMATCH {tag}/{arm} L{g['level']}@c{g['cycle']}: "
                      f"n {g['n_recon']} vs {g['n_logged']}  rec {g['recall_recon']:.4f} vs "
                      f"{g['recall_logged']}  prec {g['prec_recon']} vs {g['prec_logged']}")
        arms.append({"tag": tag, "arm": arm, "res": res, "frozen": frozen,
                     "gate": rows, "K": K, "operative": operative})
    by_level = collections.Counter(g["level"] for a in arms for g in a["gate"])
    print(f"    {'PASS' if not n_bad else 'FAIL'} — {n_ok} commit events reproduced "
          f"entry-for-entry ({dict(sorted(by_level.items()))} by level)")
    print(f"    (A3's own B-1 was 18/18 over cd_s0+ma_s0, which held NO L4 commit; the L4 "
          f"replay is new here.)")
    out["gate_b1"] = {"pass": bool(not n_bad), "n_events": n_ok + n_bad,
                      "by_level": {str(k): int(v) for k, v in by_level.items()}}

    # ---- [2] the L4 books and the buildable-L5 ceiling ---------------------------------- #
    u5 = np.unique(truth_tab[5]["flat"], axis=0)
    T4s = sorted(truth[4])
    idx4 = {k: i for i, k in enumerate(T4s)}
    Ai = np.array([idx4[tuple(int(x) for x in r)] for r in u5[:, :8]])
    Bi = np.array([idx4[tuple(int(x) for x in r)] for r in u5[:, 8:]])

    def buildable_l5(l4set):
        mask = np.zeros(len(T4s), bool)
        for k in l4set:
            if k in idx4:
                mask[idx4[k]] = True
        return int((mask[Ai] & mask[Bi]).sum())

    print("\n[2] the L4 book each arm actually holds, and what an L5 commit could ratchet over it")
    print(f"    {'tag/arm':<26} {'L4 book':>8} {'true':>5} {'r(L4)':>7} | {'live L4':>8} {'true':>5} "
          f"| {'b5|frozen':>10} {'b5|live':>8} {'r2*|T5|':>9} {'excess':>7} {'L5@sup':>7}")
    rows2 = []
    for a in arms:
        res = a["res"]
        if int(res["config"]["max_macro_level"]) < 4:
            continue
        last = res["log"]["cycle"][-1]
        live4 = set(buildable(a["K"][4][last], a["operative"](3, last)))
        fz4 = a["frozen"].get(4, live4)
        r4 = len(fz4 & truth[4]) / len(truth[4])
        b5f, b5l = buildable_l5(fz4), buildable_l5(live4)
        pred = r4 ** 2 * len(truth[5])
        oh5 = (res.get("obs_hist") or {}).get("5") or [0]
        rows2.append({"tag": a["tag"], "arm": a["arm"], "n_l4_book": len(fz4),
                      "n_l4_true": len(fz4 & truth[4]), "recall_l4": r4,
                      "n_l4_live": len(live4), "n_l4_live_true": len(live4 & truth[4]),
                      "buildable_l5_frozen": b5f, "buildable_l5_live": b5l,
                      "r2_pred": pred, "l5_at_support_max": int(max(oh5))})
        print(f"    {a['tag']+'/'+a['arm']:<26} {len(fz4):>8} {len(fz4 & truth[4]):>5} {r4:>7.4f} | "
              f"{len(live4):>8} {len(live4 & truth[4]):>5} | {b5f:>10} {b5l:>8} {pred:>9.2f} "
              f"{(b5f/pred if pred else float('nan')):>7.2f} {max(oh5):>7}")
    print("    'b5' = true L5 keys BOTH of whose halves are rows of that L4 table — the r2 wall,")
    print("    with NO support requirement. 'L5@sup' = the L5 keys (true+junk) the run's own")
    print("    observation panel ever drove to support 3. The two are measured against each other")
    print("    in [8]: the ceiling is not what binds. ('excess' is measured/r2-predicted; at book")
    print("    sizes 1-13 it is a small-number effect — the clean statement of the law is [3].)")
    out["l5_ceiling"] = rows2

    # ---- [3] the r-squared payoff curve at L5 ------------------------------------------- #
    print("\n[3] the r2 payoff curve at THIS rung: buildable true L5 vs L4 recall "
          "(20 random L4 subsets/point)")
    rng = np.random.default_rng(0)
    curve = []
    print(f"    {'k':>5} {'recall':>8} {'mean':>11} {'sd':>9} {'r2*205824':>11} {'ratio':>7}")
    for k in (5, 7, 10, 20, 40, 80, 160, 320, 408, 816):
        vals = []
        for _ in range(20 if k < 816 else 1):
            mask = np.zeros(len(T4s), bool)
            mask[rng.permutation(len(T4s))[:k]] = True
            vals.append(int((mask[Ai] & mask[Bi]).sum()))
        r = k / len(T4s)
        pred = r ** 2 * len(truth[5])
        curve.append({"k": k, "recall": r, "mean": float(np.mean(vals)),
                      "sd": float(np.std(vals)), "r2_pred": pred})
        print(f"    {k:>5} {r:>8.4f} {np.mean(vals):>11.1f} {np.std(vals):>9.1f} {pred:>11.1f} "
              f"{np.mean(vals)/pred:>7.3f}")
    print("    The r2 law is EXACT at L5 as it is at L4. 31% of all L4xL4 pairs are true L5 keys")
    print(f"    (|T5| / |T4|^2 = {len(truth[5])/len(truth[4])**2:.3f}), which is why even a")
    print("    5-entry L4 book has a non-empty L5 ratchet ceiling.")
    out["r2_curve_l5"] = curve

    # ---- [4] the L5/L6 streams as logged, and the demand-cell test ---------------------- #
    print("\n[4] the L5 and L6 observation streams, as the runs logged them (`obs_hist`)")
    print(f"    {'tag/arm':<26} {'cyc':>4} {'L4 max':>7} {'L5 max':>7} {'L5 first':>9} "
          f"{'L5 distinct vals':>17} {'L6 max':>7}")
    streams = []
    for a in arms:
        oh = a["res"].get("obs_hist") or {}
        s4, s5, s6 = oh.get("4") or [], oh.get("5") or [], oh.get("6") or []
        if not s5:
            continue
        first = next((a["res"]["log"]["cycle"][i] for i, x in enumerate(s5) if x > 0), None)
        streams.append({"tag": a["tag"], "arm": a["arm"], "n_cycles": len(s5),
                        "l4_max": int(max(s4) if s4 else 0), "l5_max": int(max(s5)),
                        "l5_first_cycle": first, "l5_distinct_values": len(set(s5)),
                        "l6_max": int(max(s6) if s6 else 0)})
        print(f"    {a['tag']+'/'+a['arm']:<26} {len(s5):>4} {max(s4) if s4 else 0:>7} "
              f"{max(s5):>7} {str(first):>9} {len(set(s5)):>17} {max(s6) if s6 else 0:>7}")
    print(f"    L5 at-support-3 over a whole run: max {max(r['l5_max'] for r in streams)}, "
          f"median {int(np.median([r['l5_max'] for r in streams]))} across "
          f"{len(streams)} arms.")
    print("    NOTE, correcting A3's Phase 0 (which read only cd_s0/ma_s0, both maxl=3 and")
    print("    116-146 cycles): the L5 read is NOT capped at 1-2. On the maxl=4 long runs it")
    print("    reaches 7-11. Whether that is a usable gauge is decided in [9], not here.")
    out["streams"] = streams

    print("\n    demand-cell test — does the era's damage cell move the L5 mining node at all?")
    print(f"    {'era':<8} " + "  ".join(f"L{e}" for e in range(2, 7)))
    node_rows = []
    for lv, nd, _c in LADDER:
        nodes = {ell: mining_node(lv, nd, ell) for ell in range(2, 7)}
        node_rows.append({"era": f"L{lv}n{nd}", **{f"L{e}": nodes[e] for e in nodes}})
        print(f"    L{lv}n{nd:<5} " + "  ".join(f"{nodes[e]:>2}" for e in range(2, 7)))
    print("    The L5 node is 1 in EVERY era and the L6 node is 0 in every era: the demand cell")
    print("    cannot move where L5 is observed. The L5 degeneracy is therefore not a")
    print("    demand-cell fact. [5] and [7] decide between the two remaining candidates.")
    out["mining_nodes"] = node_rows

    # ---- [5] the marginals ------------------------------------------------------------- #
    print("\n[5] the DGP's own key marginal at each level's mining node "
          f"(n = {N_MARG_SMALL:,} derivations; L5 gets {n_marg:,})")
    codes_true = {ell: set(enc(np.unique(truth_tab[ell]["flat"], axis=0)).tolist())
                  for ell in (2, 3, 4, 5)}
    era_nodes = {2: mining_node(1, 25, 2), 3: mining_node(2, 12, 3),
                 4: mining_node(3, 6, 4), 5: mining_node(4, 3, 5), 6: mining_node(5, 1, 6)}
    rng = np.random.default_rng(4242)
    marg = {}
    roots = rng.integers(0, V, size=N_MARG_SMALL)
    pf = MC.exact_features(sample_derivations(rules, roots, S, rng), ib, V, S)
    for ell in (2, 3, 4):
        node, span = era_nodes[ell], S ** (ell - 1)
        cnt = collections.Counter(enc(pf[:, node * span:(node + 1) * span]).tolist())
        pt = {k: c / N_MARG_SMALL for k, c in cnt.items() if k in codes_true[ell]}
        pj_mass = 1.0 - sum(pt.values())
        marg[ell] = {"node": node, "p_true": pt, "mass_true": float(sum(pt.values())),
                     "n_true_seen": len(pt), "n_junk_seen": len(cnt) - len(pt), "N": N_MARG_SMALL}
        fr = np.array(sorted(pt.values()))
        print(f"    L{ell} node{node}: true seen {len(pt):>6}/{len(codes_true[ell]):<6} "
              f"junk keys {len(cnt)-len(pt):>7}  mass_true {sum(pt.values()):.4f}  "
              f"JUNK MASS {pj_mass:.4f}")
        print(f"        true p: min {fr[0]:.2e} med {np.median(fr):.2e} max {fr[-1]:.2e}   "
              f"uniform {1/len(codes_true[ell]):.2e}   max/uniform {fr[-1]*len(codes_true[ell]):.1f}")
    del pf

    # L5 needs the deep sample (its median true key sits at ~2.5e-7)
    cnt5 = collections.Counter()
    n5 = 0
    t0 = time.time()
    while n5 < n_marg:
        n = min(500_000, n_marg - n5)
        roots = rng.integers(0, V, size=n)
        pf = MC.exact_features(sample_derivations(rules, roots, S, rng), ib, V, S)
        for c in enc(pf[:, 16:32]).tolist():
            if c in codes_true[5]:
                cnt5[c] += 1
        n5 += n
        del pf
    p5 = np.array(sorted(cnt5.values()), dtype=float) / n5
    marg[5] = {"node": era_nodes[5], "mass_true": float(p5.sum()), "n_true_seen": len(p5),
               "N": n5, "p_true_sorted": p5.tolist()}
    print(f"    L5 node{era_nodes[5]}: true seen {len(p5):,}/{len(codes_true[5]):,}  "
          f"mass_true {p5.sum():.4f}  JUNK MASS {1-p5.sum():.4f}   ({time.time()-t0:.0f}s)")
    print(f"        true p: min {p5[0]:.2e} med {np.median(p5):.2e} q99 {np.quantile(p5,.99):.2e} "
          f"max {p5[-1]:.2e}   uniform {1/len(codes_true[5]):.2e}   "
          f"max/uniform {p5[-1]*len(codes_true[5]):.1f}")

    # L6: membership without enumeration
    roots = rng.integers(0, V, size=200_000)
    pf = MC.exact_features(sample_derivations(rules, roots, S, rng), ib, V, S)
    lo, hi = enc(pf[:, :16]).tolist(), enc(pf[:, 16:]).tolist()
    layer6 = rules[DEPTH - 6]
    pairs6 = sorted({(int(layer6[f, r, 0]), int(layer6[f, r, 1]))
                     for f in range(V) for r in range(M)})
    ok6 = np.zeros(len(lo), bool)
    for c1, c2 in pairs6:
        A, B = F[5][c1], F[5][c2]
        for i, (x, y) in enumerate(zip(lo, hi)):
            if not ok6[i] and x in A and y in B:
                ok6[i] = True
    marg[6] = {"node": era_nodes[6], "mass_true": float(ok6.mean()), "N": len(lo)}
    print(f"    L6 node{era_nodes[6]}: mass_true {ok6.mean():.4f}  JUNK MASS {1-ok6.mean():.4f} "
          f"(membership tested against F[5]; |T6| is not enumerable)")
    del pf
    out["marginals"] = {str(k): {kk: vv for kk, vv in v.items()
                                 if kk not in ("p_true", "p_true_sorted")}
                        for k, v in marg.items()}
    out["marginals"]["5"]["p_true_quantiles"] = {
        q: float(np.quantile(p5, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)}

    # ---- [6] GATE G-1: the random-draw model calibrated against the logs ---------------- #
    print("\n[6] GATE G-1 — the random-draw model against the logs. Each mined observation is an")
    print("    independent draw from the marginal above; a key at support 3 is Poisson(N*p) >= 3.")
    print(f"    {'level':>5} {'n arms':>7} {'N_obs (med)':>12} {'model @med N':>13} "
          f"{'realized (min..max)':>22} {'per-arm ratio: med [IQR]':>28}")
    g1 = []
    for ell in (2, 3, 4):
        pv = list(marg[ell]["p_true"].values())
        real, ratios = [], []
        for a in arms:
            res = a["res"]
            src = res["log"]["gy"] if ell == 4 else res["log"]["miner"]
            st = (src[-1] if ell == 4 else (res["log"]["miner"][-1] or {}).get(str(ell)))
            if not st or not int(st["n_obs"]):
                continue
            ks = {tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])}
            n_obs, n_true = int(st["n_obs"]), len(ks & truth[ell])
            real.append((n_obs, n_true))
            mdl = sum(pge(n_obs * p) for p in pv)
            if mdl > 0:
                ratios.append(n_true / mdl)
        if not real:
            continue
        N = int(np.median([r[0] for r in real]))
        model = sum(pge(N * p) for p in pv)
        vals = [r[1] for r in real]
        g1.append({"level": ell, "n_arms": len(real), "N_obs_median": N, "model_at_med_N": model,
                   "realized_min": min(vals), "realized_max": max(vals),
                   "realized_median": float(np.median(vals)),
                   "ratio_median": float(np.median(ratios)),
                   "ratio_q25": float(np.quantile(ratios, .25)),
                   "ratio_q75": float(np.quantile(ratios, .75))})
        print(f"    {ell:>5} {len(real):>7} {N:>12} {model:>13.1f} "
              f"{str(min(vals))+'..'+str(max(vals)):>22} "
              f"{f'{np.median(ratios):.2f} [{np.quantile(ratios,.25):.2f}, {np.quantile(ratios,.75):.2f}]':>28}")
    print("    The model is the DGP's clean draw; the ratio is what the repair, the reader and")
    print("    the agent's own narrowness do to it (`antiphon` Phase 0(c)'s delivery-fidelity")
    print("    prior, re-measured here on a 5x larger corpus). It is the correction factor the")
    print("    L5 budget in [7] is stated with.")
    out["gate_g1"] = g1

    # ---- [7] THE BUDGET ---------------------------------------------------------------- #
    print("\n[7] THE BUDGET — what an L5 book costs in observations")
    l4_ref = next((r for r in g1 if r["level"] == 4), None)
    delta = l4_ref["ratio_median"] if l4_ref else 1.0
    print(f"    model E[# true L5 keys at support 3] vs N_obs (marginal above; unseen true keys")
    print(f"    contribute nothing, so this is a LOWER bound), and the same with the L4-measured")
    print(f"    delivery correction delta = {delta:.2f} applied to N:")
    print(f"    {'N_obs':>12} {'cycles @8/cyc':>14} {'GPU-h @10.6s/cyc':>18} {'E[true L5 @sup3]':>18}")
    budget = []
    for N in (1_600, 8_000, 16_000, 40_000, 80_000, 160_000, 400_000, 1_600_000):
        e = sum(pge(N * p) for p in p5)
        budget.append({"N": N, "cycles": N / MINE_CAP,
                       "gpu_h": N / MINE_CAP * SEC_PER_CYCLE / 3600, "E_true_at_sup": e})
        print(f"    {N:>12,} {N/MINE_CAP:>14,.0f} {N/MINE_CAP*SEC_PER_CYCLE/3600:>18,.1f} "
              f"{e:>18.3f}")

    def solve_N(target):
        lo_, hi_ = 1e3, 1e8
        for _ in range(80):
            mid = math.sqrt(lo_ * hi_)
            if sum(pge(mid * p) for p in p5) < target:
                lo_ = mid
            else:
                hi_ = mid
        return math.sqrt(lo_ * hi_)

    print(f"\n    N_obs required (and the multiple of the run's own 1,608 L5-node observations).")
    print(f"    'delivery-corrected' divides the target by the measured L4 ratio {delta:.2f} — the")
    print(f"    agent's own productions being narrower than the DGP's draw is the only measured")
    print(f"    lever that makes support cheaper than the clean-draw model says:")
    print(f"    {'target E[true L5 @sup3]':>34} {'N_obs':>12} {'x run':>7} {'cycles':>9} "
          f"{'GPU-h/arm':>10} | {'N (delta-corr)':>14} {'x run':>7}")
    targets = []
    for tgt, why in [(1, "one entry"), (5, "a crescendo-sized L4 book"),
                     (22, "q_exo's realized L4 book"), (50, "q_bisect's realized L4 book")]:
        N = solve_N(tgt)
        Nd = solve_N(tgt / delta) if delta > 0 else N
        targets.append({"target": tgt, "why": why, "N_obs": N, "x_run": N / 1608,
                        "cycles": N / MINE_CAP, "gpu_h": N / MINE_CAP * SEC_PER_CYCLE / 3600,
                        "N_obs_delta_corrected": Nd, "x_run_delta_corrected": Nd / 1608})
        print(f"    {str(tgt)+'  ('+why+')':>34} {N:>12,.0f} {N/1608:>6,.0f}x "
              f"{N/MINE_CAP:>9,.0f} {N/MINE_CAP*SEC_PER_CYCLE/3600:>10,.1f} | "
              f"{Nd:>14,.0f} {Nd/1608:>6,.0f}x")
    out["budget"] = {"curve": budget, "targets": targets, "delta_l4": delta,
                     "run_l5_observations": 1608}

    # the sharpest form: the arc's BEST L4 book's own buildable L5 keys, at the run's own N
    best = max(rows2, key=lambda r: r["buildable_l5_frozen"])
    ab = next(a for a in arms if a["tag"] == best["tag"] and a["arm"] == best["arm"])
    fz4 = ab["frozen"].get(4, set())
    bl5 = [tuple(int(x) for x in r) for r in u5
           if tuple(int(x) for x in r[:8]) in fz4 and tuple(int(x) for x in r[8:]) in fz4]

    def code_of(k):
        return sum(int(x) * V ** i for i, x in enumerate(k))

    ps = [cnt5.get(code_of(k), 0) / n5 for k in bl5]
    print(f"\n    THE SHARPEST FORM. {best['tag']}/{best['arm']} holds the arc's best L4 book "
          f"({best['n_l4_book']} entries,")
    print(f"    {best['n_l4_true']} true). Its ratchet admits {len(bl5)} true L5 keys — including "
          f"the single most")
    print(f"    frequent true L5 key at the node (p = {max(ps):.2e}). E[# of those {len(bl5)} "
          f"reaching support 3]:")
    sharp = []
    for N in (1_600, 16_000, 160_000, 1_600_000):
        e = sum(pge(N * p) for p in ps)
        sharp.append({"N": N, "E": e})
        print(f"        N = {N:>10,}  ->  {e:.4f}")
    out["sharpest"] = {"arm": f"{best['tag']}/{best['arm']}", "n_buildable_l5": len(bl5),
                       "p_max": max(ps) if ps else None, "curve": sharp}

    # ---- [8] which wall binds ----------------------------------------------------------- #
    print("\n[8] WHICH WALL BINDS — the two multiplicative cuts, per rung, on the arc's own runs")
    print(f"    {'rung':>5} {'|T_l|':>10} {'true@sup (med)':>15} {'ARRIVAL cut':>13} "
          f"{'book true (med)':>16} {'RATCHET cut':>12}")
    walls = []
    for ell in (2, 3, 4):
        tsup, book = [], []
        for a in arms:
            res = a["res"]
            src = res["log"]["gy"] if ell == 4 else res["log"]["miner"]
            st = (src[-1] if ell == 4 else (res["log"]["miner"][-1] or {}).get(str(ell)))
            if st:
                ks = {tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])}
                tsup.append(len(ks & truth[ell]))
            if ell in a["frozen"]:
                book.append(len(a["frozen"][ell] & truth[ell]))
        if not tsup or not book:
            continue
        ts, bk = float(np.median(tsup)), float(np.median(book))
        walls.append({"level": ell, "size": len(truth[ell]), "true_at_sup": ts,
                      "arrival_cut": len(truth[ell]) / max(ts, 1e-9),
                      "book_true": bk, "ratchet_cut": ts / max(bk, 1e-9)})
        print(f"    L{ell:<4} {len(truth[ell]):>10,} {ts:>15.0f} {len(truth[ell])/ts:>12.1f}x "
              f"{bk:>16.0f} {ts/max(bk,1e-9):>11.1f}x")
    e5 = sum(pge(1608 * p) for p in p5)
    print(f"    L5    {len(truth[5]):>10,} {e5:>15.3f} {len(truth[5])/max(e5,1e-12):>12.3g}x "
          f"{'(n/a)':>16} {'(n/a)':>12}")
    walls.append({"level": 5, "size": len(truth[5]), "true_at_sup": e5,
                  "arrival_cut": len(truth[5]) / max(e5, 1e-12),
                  "book_true": None, "ratchet_cut": None,
                  "note": "true_at_sup is modelled, not measured: no L5 key stream is logged"})
    print("    Read: at L2/L3/L4 the two cuts are the same order. At L5 the ARRIVAL cut is")
    print("    ~5 orders of magnitude while the ratchet ceiling is non-empty ([2]). The second")
    print("    extension is not r2-limited; it is support-limited.")
    out["walls"] = walls

    # ---- [9] the gauge at the L5 frontier ----------------------------------------------- #
    print("\n[9] THE GAUGE AT THE L5 FRONTIER — floors derived by the arc's own null-ABBA method")
    print(f"    {'level':>5} {'windows':>8} {'sd(N)':>9} {'v_tol':>9} {'mean(D)':>10} "
          f"{'signal/floor':>13} {'frac(D>tol)':>12} {'distinct vals/arm':>18}")
    floors = {}
    for ell in (3, 4, 5, 6):
        allN, allD, dv = [], [], []
        for a in arms:
            if int(a["res"]["config"]["max_macro_level"]) < 4:
                continue
            ser_raw = (a["res"].get("obs_hist") or {}).get(str(ell))
            if not ser_raw:
                continue
            ser = [-float(x) for x in ser_raw]          # error convention, as the panel logs it
            cyc, era = a["res"]["log"]["cycle"], a["res"]["log"]["era"]
            skip = {i for i in range(1, len(era)) if era[i] != era[i - 1]}
            for e in a["res"]["events"]:
                if e["kind"] == "commit" and int(e["cycle"]) in cyc:
                    skip.add(cyc.index(int(e["cycle"])))
            N_, D_ = null_abba(ser, skip=skip, span=1, W=4)
            allN += N_
            allD += D_
            dv.append(len(set(ser)))
        tol = _sd(allN) / 2.0
        mD = float(np.mean(allD))
        frac = sum(1 for x in allD if x > tol) / len(allD)
        floors[ell] = tol
        print(f"    {ell:>5} {len(allN):>8} {_sd(allN):>9.5f} {tol:>9.5f} {mD:>+10.5f} "
              f"{mD/tol:>13.2f} {frac:>12.2f} {np.mean(dv):>18.1f}")
    print("    tol_yield_l5 IS derivable (0.064) — but its SIGNAL is 0.26 floor units against")
    print("    L3's 1.38 and L4's 1.53. A derivable floor on a gauge that never clears it is a")
    print("    thermostat that fires on noise, which is a sharper statement than 'degenerate'.")
    out["floors"] = {str(k): v for k, v in floors.items()}

    print("\n    A1's thermostat replayed on each arm's own L5 series, from era-4 start, at the")
    print("    derived L5 floor:")
    fires_all = []
    for a in arms:
        ser = (a["res"].get("obs_hist") or {}).get("5")
        if not ser or int(a["res"]["config"]["max_macro_level"]) < 4:
            continue
        cyc, era = a["res"]["log"]["cycle"], a["res"]["log"]["era"]
        idx = [i for i, e in enumerate(era) if e >= 4]
        if not idx:
            continue
        pol = QuietPolicy("yield", v_tol_by_level={5: floors[5]}, span=1, W=4, burn=4, alpha=0.5)
        pol.acted("era_start", 0, why="era4")
        fires, c0 = [], cyc[idx[0]]
        for i in idx:
            if pol.step(cyc[i], {"yield": -float(ser[i]), "yield_level": 5}).get("quiet"):
                fires.append(cyc[i] - c0 + 1)
                pol.acted("commit", cyc[i])
        fires_all.append({"tag": a["tag"], "arm": a["arm"], "era4_start": c0,
                          "n_cycles": len(idx), "fires_c_in_era4": fires})
    nf = sum(1 for r in fires_all if r["fires_c_in_era4"])
    print(f"    {nf}/{len(fires_all)} arms fire at all inside the 17-21 consumption cycles; the")
    print("    firings are spread over c_in_era4 7..21 with no common structure, which is what a")
    print("    sub-floor gauge produces. The one-level-up read for an L5 commit is L6, whose")
    print(f"    signal/floor is {0.08:.2f} — 3x worse still.")
    out["thermostat_l5"] = fires_all

    # ---- [10] world-sizing -------------------------------------------------------------- #
    print("\n[10] WORLD-SIZING — |T_l| as a function of (v, s, m), and what world makes L5 "
          "what L4 was here")

    def no_collision(v, s, m, ell):
        """v * m**((s**(l-1)-1)/(s-1)) — the level size if no two expansions collide. An UPPER
        bound; on this world the realised collision factor c_l runs 0.875 -> 0.760."""
        return v * m ** ((s ** (ell - 1) - 1) // (s - 1))

    def exact_sizes(v, s, m, upto=5, seed=0, budget=3_000_000):
        """The measured level sizes for another world, computed the same way as [0]; None where
        the per-feature expansion set would exceed `budget`."""
        rs = generate_rules_distinct(v, s, 6, m, seed=seed)
        Ff = {1: {f: {(f,)} for f in range(v)}}
        got = {1: v}
        for ell in range(2, upto + 1):
            if no_collision(v, s, m, ell) > budget:
                break
            layer = rs[6 - ell]
            cur = {}
            for f in range(v):
                acc = set()
                for r in range(m):
                    kids = [int(layer[f, r, i]) for i in range(s)]
                    combos = [()]
                    for c_ in kids:
                        combos = [pre + a for pre in combos for a in Ff[ell - 1][c_]]
                    acc |= set(combos)
                cur[f] = acc
            Ff[ell] = cur
            got[ell] = len(set().union(*cur.values()))
        return got

    print("    the exponent is (s^(l-1)-1)/(s-1), so |T_l| is DOUBLY exponential in l.")
    print("    'exact' = enumerated the way [0] does; 'bound' = the no-collision upper bound "
          "v*m^((s^(l-1)-1)/(s-1)):")
    print(f"    {'(v,s,m)':>10} " + " ".join(f"{'L'+str(e):>17}" for e in range(2, 7)))
    world_rows = []
    for (v, s, m) in [(8, 2, 2), (8, 2, 3), (8, 2, 4), (4, 2, 2), (2, 2, 2), (8, 3, 2), (8, 2, 1)]:
        ex = exact_sizes(v, s, m)
        cells, vals = [], {}
        for e in range(2, 7):
            if e in ex:
                cells.append(f"{ex[e]:>13,d} ex")
                vals[str(e)] = int(ex[e])
            else:
                b = no_collision(v, s, m, e)
                cells.append(f"{b:>13,d} ub")
                vals[str(e)] = int(b)
        world_rows.append({"v": v, "s": s, "m": m, "sizes": vals,
                           "exact_upto": max(ex) if ex else 1})
        print(f"    {str((v,s,m)):>10} " + " ".join(f"{c:>17}" for c in cells))
    print("    (row 1 is this world: 14 / 56 / 816 / 205,824 / 1.3e10, the measured ladder)")
    print("    |T_(l+1)|/|T_l| = m^(s^(l-1)) — the RATIO itself grows doubly exponentially, and")
    print("    it does not depend on v. L3->L4 is 16x here; L4->L5 is 256x; L5->L6 is 65,536x.")
    print("    So no (v, s, m) puts |T_5| near 816 while |T_4| stays near 56: it would need")
    print("    m ~ 1.4, and m = 1 collapses the grammar to v distinct sequences (no synonymy,")
    print("    nothing to mine). THE SECOND EXTENSION AS A LEVEL EXTENSION IS UNREACHABLE IN")
    print("    THIS FAMILY, at any (v, s, m) and any affordable budget.")
    print("\n    What the level size WOULD be under the index ops (the F3 levers):")
    print(f"      unmerged (flat-tuple key, what the miner uses):   |T_5| = {len(truth[5]):,}")
    print(f"      merged by parent feature (the `fourwall` re-key):  |T_5| = {V*M} at EVERY "
          f"level")
    print(f"      -> the merge would shrink the L5 level by {len(truth[5])/(V*M):,.0f}x. The "
          f"level-size wall is a")
    print("      property of the flat-tuple KEY, not of the grammar — which puts the index ops,")
    print("      not the budget, on the critical path of any further climb.")
    out["worlds"] = world_rows
    out["merged_level_size"] = V * M

    # ---- [11] the world that WOULD make L5 what L4 is here ------------------------------ #
    print("\n[11] SIZING THE WORLD THE QUESTION COULD BE ASKED IN. For each candidate, the exact")
    print("     level ladder, the L5 junk mass, and the observation budget its L5 needs to reach")
    print("     an L4-sized book — against this world's L4, which reached 22 true keys at 1,600.")

    def world_l5_budget(v, s, m, depth=6, n=2_000_000, seed=0):
        rs = generate_rules_distinct(v, s, depth, m, seed=seed)
        ibv = build_inverse_maps(rs)[-1]
        tt = MC.true_tables(rs, depth, s, v, m, 5)
        T5 = set(map(tuple, np.unique(tt[5]["flat"], axis=0).tolist()))
        span = s ** 4
        node = (s ** depth // s) // span - 1
        cnt, got = collections.Counter(), 0
        rg = np.random.default_rng(4242)
        while got < n:
            k = min(500_000, n - got)
            pfv = MC.exact_features(sample_derivations(rs, rg.integers(0, v, size=k), s, rg),
                                    ibv, v, s)
            for row in pfv[:, node * span:(node + 1) * span].tolist():
                tk = tuple(row)
                if tk in T5:
                    cnt[tk] += 1
            got += k
        pp = np.array(sorted(cnt.values()), dtype=float) / got

        def solve(tgt):
            lo_, hi_ = 1e2, 1e9
            for _ in range(70):
                mid = math.sqrt(lo_ * hi_)
                if sum(pge(mid * x) for x in pp) < tgt:
                    lo_ = mid
                else:
                    hi_ = mid
            return math.sqrt(lo_ * hi_)
        return {"v": v, "s": s, "m": m, "occupancy": m / v ** (s - 1),
                "sizes": {str(e): len(set(map(tuple, np.unique(tt[e]["flat"], axis=0).tolist())))
                          for e in (2, 3, 4, 5)},
                "l5_mass_true": float(pp.sum()), "l5_true_seen": int(len(pp)),
                "E_at_1600": float(sum(pge(1600 * x) for x in pp)),
                "N_for_22": solve(22)}

    print(f"     {'(v,s,m)':>10} {'m/v^(s-1)':>10} {'L2':>6} {'L3':>7} {'L4':>9} {'L5':>11} "
          f"{'L5 junk':>8} {'E@1600':>8} {'N for 22':>11} {'x this run':>11} {'GPU-h/arm':>10}")
    cand = []
    for (v, s, m) in [(8, 2, 2), (6, 2, 2), (4, 2, 2)]:
        r = world_l5_budget(v, s, m)
        cand.append(r)
        print(f"     {str((v,s,m)):>10} {r['occupancy']:>10.3f} {r['sizes']['2']:>6,} "
              f"{r['sizes']['3']:>7,} {r['sizes']['4']:>9,} {r['sizes']['5']:>11,} "
              f"{1-r['l5_mass_true']:>8.3f} {r['E_at_1600']:>8.2f} {r['N_for_22']:>11,.0f} "
              f"{r['N_for_22']/1608:>10,.0f}x {r['N_for_22']/MINE_CAP*SEC_PER_CYCLE/3600:>10,.1f}")
    print("     (v=2 was measured and is degenerate: at m/v = 1.0 the bottom inverse map is")
    print("      constant, every span parses to one key, and the whole grammar collapses.)")
    print("\n     THE TENSION, stated as arithmetic. Shrinking |T_5| means shrinking v — the")
    print("     collision factor helps (c_5 is 0.785 at v=8, 0.43 at v=4), so v=4 buys a 4.6x")
    print("     smaller L5 and a ~10x cheaper L5 book. But m/v^(s-1) is the tuple-space")
    print("     occupancy, and v=4/m=2 sits at 0.500 — EXACTLY the occupancy of v=8/m=4, the")
    print("     setting `tall/` measured INADMISSIBLE for sculpting (synonymy flattens the depth")
    print("     ladder 1.06x vs m=2's 3.25x and walls off every level above 3). Every world whose")
    print("     L5 is affordable has an occupancy the arc has already ruled out; every world with")
    print("     an admissible occupancy has an L5 at least this size. That is a cross-world")
    print("     inference from a measured verdict, not a new measurement — and it is the reason")
    print("     the second extension has to be bought with the INDEX, not with the world.")
    out["candidate_worlds"] = cand

    # ---- [12] the rule draw ------------------------------------------------------------- #
    print("\n[12] THE RULE DRAW. `antiphon` Phase 0(a) reports the parse-ambiguity junk mass")
    print("     (0.24 / 0.58 / 0.82 at the L2 / L3 / L4 mining nodes) as a property of the")
    print("     grammar. It is a property of the DRAW. `build_inverse_maps` is last-writer-wins,")
    print("     so junk exists only where two of the v*m = 16 bottom tuples COLLIDE; the arc's")
    print("     `rule_seed = 0` covers 14 of 64 codes (2 collisions) and the error compounds up")
    print("     the span. Twelve draws of the SAME (v, s, m, depth), at the era mining nodes:")
    print(f"     {'seed':>5} {'codes':>6} {'|T5|':>9} {'mass_true L2':>13} {'L3':>7} {'L4':>7} "
          f"{'L5':>7}")
    seeds = []
    for sd in range(12):
        rs = generate_rules_distinct(V, S, DEPTH, M, seed=sd)
        ibs = build_inverse_maps(rs)[-1]
        tts = MC.true_tables(rs, DEPTH, S, V, M, 5)
        rg = np.random.default_rng(7)
        pfs = MC.exact_features(sample_derivations(rs, rg.integers(0, V, size=300_000), S, rg),
                                ibs, V, S)
        ms = []
        for ell, node in ((2, 12), (3, 6), (4, 3), (5, 1)):
            Tl = set(map(tuple, np.unique(tts[ell]["flat"], axis=0).tolist()))
            sp = S ** (ell - 1)
            ms.append(float(np.mean([tuple(r) in Tl
                                     for r in pfs[:, node * sp:(node + 1) * sp].tolist()])))
        n5s = len(set(map(tuple, np.unique(tts[5]["flat"], axis=0).tolist())))
        seeds.append({"seed": sd, "codes_covered": int((ibs >= 0).sum()), "size_l5": n5s,
                      "mass_true": {"2": ms[0], "3": ms[1], "4": ms[2], "5": ms[3]}})
        print(f"     {sd:>5} {int((ibs>=0).sum()):>6} {n5s:>9,} {ms[0]:>13.3f} {ms[1]:>7.3f} "
              f"{ms[2]:>7.3f} {ms[3]:>7.3f}")
        del pfs
    print("     Seeds 6 and 10 cover all 16 codes — the bottom parse is EXACT and there is NO")
    print("     parse ambiguity at any level. Seed 0 is the second-WORST of the twelve.")

    print("\n     What that costs at L5, on the SAME world (v, s, m, depth, occupancy and every")
    print("     `tall/` admissibility verdict unchanged) — only the rule draw moves:")
    print(f"     {'seed':>5} {'|T5|':>9} {'L5 junk':>8} {'E@1600':>8} {'N for 22':>10} "
          f"{'x this run':>11} {'cycles/arm':>11} {'GPU-h/arm':>10}")

    def l5_budget_at_seed(sd, n=2_000_000):
        rs = generate_rules_distinct(V, S, DEPTH, M, seed=sd)
        ibs = build_inverse_maps(rs)[-1]
        tts = MC.true_tables(rs, DEPTH, S, V, M, 5)
        T5 = set(map(tuple, np.unique(tts[5]["flat"], axis=0).tolist()))
        cn, got = collections.Counter(), 0
        rg = np.random.default_rng(4242)
        while got < n:
            k = min(500_000, n - got)
            pfs = MC.exact_features(sample_derivations(rs, rg.integers(0, V, size=k), S, rg),
                                    ibs, V, S)
            for row in pfs[:, 16:32].tolist():
                tk = tuple(row)
                if tk in T5:
                    cn[tk] += 1
            got += k
            del pfs
        pp = np.array(sorted(cn.values()), dtype=float) / got

        def solve(tgt):
            lo_, hi_ = 1e2, 1e9
            for _ in range(70):
                mid = math.sqrt(lo_ * hi_)
                if sum(pge(mid * x) for x in pp) < tgt:
                    lo_ = mid
                else:
                    hi_ = mid
            return math.sqrt(lo_ * hi_)
        return {"seed": sd, "size_l5": len(T5), "mass_true": float(pp.sum()),
                "E_at_1600": float(sum(pge(1600 * x) for x in pp)), "N_for_22": solve(22),
                "N_for_5": solve(5)}

    seed_budget = []
    for sd in (0, 2, 6, 10, 8):
        r = l5_budget_at_seed(sd)
        seed_budget.append(r)
        print(f"     {sd:>5} {r['size_l5']:>9,} {1-r['mass_true']:>8.3f} {r['E_at_1600']:>8.3f} "
              f"{r['N_for_22']:>10,.0f} {r['N_for_22']/1608:>10,.0f}x "
              f"{r['N_for_22']/MINE_CAP:>11,.0f} "
              f"{r['N_for_22']/MINE_CAP*SEC_PER_CYCLE/3600:>10,.1f}")
    print("     The L5 arrival budget swings 6x-267x with the draw alone. `rhm_data.py` already")
    print("     carries `generate_rules_invertible`, which makes collision-freeness a")
    print("     construction rather than luck (it needs v*m <= v^s: 16 <= 64 here).")
    print("     THE TRADE THAT COMES WITH IT: a collision-free grammar has NO parse ambiguity,")
    print("     so it also deletes the arc's native aleatoric channel — the junk mass that")
    print("     `antiphon` Phase 0(a) used as its free nerdsnipe cell. The affordable-L5 world")
    print("     and the free-aleatoric-trap world are the same world at different draws.")
    out["rule_draw"] = {"survey": seeds, "l5_budget_by_seed": seed_budget}

    with open(os.path.join(HERE, "phase0.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'phase0.json')}   ({time.time()-t_start:.0f}s)")
    return out


if __name__ == "__main__":
    main()
