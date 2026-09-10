"""PHASE 0 — size the QUOTIENT offline, before anything is forked or a GPU is spent.

`enharmonic/SPEC.md` Q0. The parent node asks whether the practice learner can re-key its
vocabulary by CATEGORY rather than by flat spelling; `temperament/SPEC.md` asks whether it can
license those merges itself. Both need four things sized first, and all four are answerable
against the banked logs and the DGP's own arithmetic:

  [1] the COVER. `MC.true_tables`'s `feature` column is each true row's parent feature, and the
      same flat tuple can carry several. So "the true partition" the merge is graded against is
      not a partition. Made exact per level, by rows and by observation MASS.
  [2] the CATEGORY COORDINATES. Two of them, and they are different objects:
        (A) the PARENT-FEATURE key (`fourwall`'s re-key basis, the parent spec's fact 4):
            v*m = 16 pairs at every rung. Set-valued from below under the cover, so an
            observation needs a counting rule; three are priced.
        (B) the TOKEN CLASS — the set of level-l features a written program is a legal
            derivation of, i.e. what the exact grader can see. Single-valued, closed under
            composition (so the ratchet restates as C[l+1] subset C[l] x C[l]), and the sizes
            are computed by closure and validated against enumeration of the true tables.
      Plus the L5/L6 arrival budget under each key, by the sizing lane's own `pge` model.
  [3] the ALIAS AUDIT, on what is actually banked. The beam use record is a per-cycle,
      level-wide count vector over the committed table's rows — no slot and no context
      resolution — so "used interchangeably in the same slot" is not readable. What IS
      readable is each row's use TIME SERIES; three similarity statistics on it are tested
      within-class against between-class, with a per-arm label permutation as the control, on
      the exact-DP tags (the live-executor tags record on ~8-20 of ~130 cycles and are
      excluded). The sharpest cell is the use-share ratio of BIT-IDENTICAL programs.
  [4] FORCED TRANSFER, offline and exact: `fourwall.entry_profile` verbatim over each arm's
      replayed committed table, on DGP-drawn damaged instances at the level's own node. The
      probe budget, the partition it recovers, the solve tax, the loss margin on
      multiply-parented rows, and what the junk rows do.

GATES. B-1' (the offline replay of `Miner.build` reproduces every logged commit event
entry-for-entry) is re-run from this file's own loader before anything downstream reads a
replayed table, and is joined by an ORDER gate: the replayed row ORDER must reproduce the
logged `log["entry"][c]["true_mask"]` elementwise, which is the precondition for keying the
beam use-record vector by row.

No GPU, no Modal, no substrate. Writes `phase0_cat.json` beside this file.

Run from experiments/:  PYTHONPATH=. python3 rhm/practice/enharmonic/sizing/phase0_cat.py
"""

import collections
import json
import math
import os
import time

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import possible_sets, sample_derivations
from rhm.practice.ratchet import macros as MC
from rhm.practice.fourwall import wall as W
from rhm.practice.tutti.tutti import context_instances
from rhm.practice.tutti.sizing.phase0_l5 import (
    feature_expansions, level_size, pge, enc, mining_node, SUPPORT, MINE_CAP, SEC_PER_CYCLE)

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(os.path.dirname(HERE))

V, S, DEPTH, M = 8, 2, 6, 2
RULE_SEED = 0
LADDER = {1: 25, 2: 12, 3: 6, 4: 3, 5: 1}     # SPIRAL_ERAS "1:25,2:12,3:6,4:3,5:1"

# THE CORPUS. `tutti/sizing/phase0_l5.py`'s nine roots, plus `tu_s0` and `tr_s0` (which post-date
# it) and `tc_s0`. `exact` marks the EXACT-DP lineage: `span_tau_fire` unset, so every macro
# execution goes through `MC.macro_features` where the entry recorder lives. On the live-executor
# tags (`span_tau_fire = 0.50`) `SpanExecutor.apply` realises an open slot with the head and takes
# its intention from `SN.dp_features`, so the beam phase records on ~8-20 of ~130 cycles; those
# tags are kept for the replay gates and the transfer probe and excluded from the alias audit.
CORPUS = [
    ("cr3_s0", "crescendo/figures/cr3_s0", True),
    ("cr3_s1", "crescendo/figures/cr3_s1", True),
    ("an_s0", "antiphon/figures/an_s0", True),
    ("an_s1", "antiphon/figures/an_s1", True),
    ("tr_s0", "antiphon/figures/tr_s0", True),
    ("tc_s0", "tacet/figures/tc_s0", True),
    ("cd_s0", "conductor/figures/cd_s0", True),
    ("ma_s0_maestro", "maestro/figures/ma_s0", True),
    ("tu_s0", "tutti/figures/tu_s0", False),
    ("ca_s0", "caesura/figures/ca_s0", False),
    ("in_s0", "intonation/figures/in_s0", False),
    ("ma_s0", "intonation/figures/ma_s0", False),
]

N_POOL = 4096            # forced-transfer instances per (level, era) cell
N_MARG = 2_000_000       # derivations for the class-key marginals
N_PERM = 2000            # label permutations in the alias audit


# --------------------------------------------------------------------------- #
# [0] the loader and the two replay gates
# --------------------------------------------------------------------------- #

def flats_of(t):
    return [tuple(int(x) for x in r) for r in t["flat"]]


def buildable_ordered(keys, lower_flats, s=S):
    """`Miner.build`'s ratchet test as a set operation, ORDER-PRESERVING. `keys` arrives in
    `sorted()` order (that is what `_install_identity_miner` logs), which is the order
    `Miner.build` iterates, so the surviving list is the committed table's row order."""
    if not keys:
        return []
    half = len(keys[0]) // s
    return [k for k in keys
            if all(k[i * half:(i + 1) * half] in lower_flats for i in range(s))]


def key_stream(res, level):
    out = {}
    for i, c in enumerate(res["log"]["cycle"]):
        st = (res["log"]["miner"][i] or {}).get(str(level))
        out[c] = [tuple(int(x) for x in k) for k in ((st or {}).get("keys_at_support") or [])]
    return out


def replay(res, truth_sets):
    """Replay `Miner.build` at every logged commit, in cycle order, exactly as the run did.
    Returns (frozen tables as ordered row lists, gate rows)."""
    maxl = int(res["config"]["max_macro_level"])
    K = {ell: key_stream(res, ell) for ell in range(2, maxl + 1)}
    base = set(flats_of(MC.base_table(V)))
    commits = sorted([e for e in res["events"] if e["kind"] == "commit"],
                     key=lambda x: int(x["cycle"]))
    frozen, rows = {}, []

    def operative(ell, c):
        if ell == 1:
            return base
        if ell in frozen:
            return set(frozen[ell])
        return set(buildable_ordered(K[ell][c], operative(ell - 1, c)))

    for e in commits:
        lv, c = int(e["level"]), int(e["cycle"])
        got = buildable_ordered(K[lv][c], operative(lv - 1, c))
        frozen[lv] = got
        ncorr = len(set(got) & truth_sets[lv])
        ci = res["log"]["cycle"].index(c)
        tm_log = ((res["log"]["entry"][ci] or {}).get("true_mask") or {}).get(str(lv))
        rows.append({
            "level": lv, "cycle": c, "n_recon": len(got), "n_logged": int(e["n_entries"]),
            "recall_recon": ncorr / len(truth_sets[lv]), "recall_logged": e.get("tab_recall"),
            "prec_recon": (ncorr / len(got)) if got else None,
            "prec_logged": e.get("tab_precision"),
            "order_ok": (None if tm_log is None
                         else tm_log == [int(k in truth_sets[lv]) for k in got])})
    return frozen, rows


def load_arms(truth_sets):
    out = []
    for tag, rel, exact in CORPUS:
        root = os.path.join(PRACTICE, rel)
        if not os.path.isdir(root):
            print(f"    [skip] {root} not found")
            continue
        for arm in sorted(os.listdir(root)):
            p = os.path.join(root, arm, "results.json")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                res = json.load(f)
            fr, gate = replay(res, truth_sets)
            out.append({"tag": tag, "arm": arm, "exact": exact, "res": res,
                        "books": fr, "gate": gate})
    return out


# --------------------------------------------------------------------------- #
# the token class: what the exact grader can see
# --------------------------------------------------------------------------- #

def entry_leaf_rows(keys, canon_np):
    return np.stack([W.entry_leaves(k, canon_np) for k in keys])


def token_classes(keys, ell, rules, canon_np):
    """For each level-`ell` flat tuple, the set of level-`ell` features its CANONICAL leaf
    rendering is a legal derivation of. This is the class the world's own DP assigns to the
    program `MC.apply_any` would write, so it is exactly what a forced-transfer probe can
    resolve. Empty = the program is off-grammar (TOKEN junk)."""
    if not keys:
        return []
    lv = entry_leaf_rows(keys, canon_np)
    P = possible_sets(rules[DEPTH - ell:], lv, S)[-1][:, 0, :]
    return [frozenset(np.nonzero(r)[0].tolist()) for r in P]


def class_step(c1, c2, ell, rules):
    """`possible_sets`' composition step, on classes: the level-`ell` features some rule of
    which sends its left child into `c1` and its right into `c2`."""
    layer = rules[DEPTH - ell]
    out = set()
    for f in range(V):
        for r in range(M):
            if int(layer[f, r, 0]) in c1 and int(layer[f, r, 1]) in c2:
                out.add(f)
                break
    return frozenset(out)


def class_closure(rules, canon_np, max_level=6):
    """The token-class alphabet at every level, and the number of legal class PAIRS (the
    class-keyed level size). Level 1's alphabet is over the WRITABLE blocks — the v canonical
    renderings, of which two coincide wherever the bottom rule draw collides."""
    bottom = rules[DEPTH - 1]

    def blk(t):
        return frozenset(f for f in range(V) for r in range(M)
                         if tuple(int(z) for z in bottom[f, r]) == t)
    writable = sorted({tuple(int(z) for z in canon_np[f]) for f in range(V)})
    cur = sorted({blk(t) for t in writable}, key=lambda c: sorted(c))
    out = {"n_writable_blocks": len(writable), "1": {"n_classes": len(cur),
                                                     "classes": [sorted(c) for c in cur]}}
    for ell in range(2, max_level + 1):
        pairs = {}
        for c1 in cur:
            for c2 in cur:
                c = class_step(c1, c2, ell, rules)
                if c:
                    pairs[(c1, c2)] = c
        nxt = sorted(set(pairs.values()), key=lambda c: sorted(c))
        out[str(ell)] = {"n_classes": len(nxt), "n_legal_class_pairs": len(pairs),
                         "classes": [sorted(c) for c in nxt]}
        cur = nxt
    return out


# --------------------------------------------------------------------------- #
# derivation sampling that keeps the generative feature at every level
# --------------------------------------------------------------------------- #

def sample_with_latents(rules, roots, s, rng):
    """`sample_derivations`, keeping the feature array at every level. `lat[l]` is (B, n_nodes)
    level-`l` features — the label the DGP actually used, which no evidence can recover where
    the cover is many-to-one."""
    L = len(rules)
    _v, m, _s = rules[0].shape
    cur = roots[:, None]
    lat = {L: cur}
    for ell in range(L):
        w = cur.shape[1]
        ch = rng.integers(0, m, size=(cur.shape[0], w))
        nxt = np.empty((cur.shape[0], w * s), dtype=np.int64)
        for j in range(w):
            nxt[:, j * s:(j + 1) * s] = rules[ell][cur[:, j], ch[:, j]]
        cur = nxt
        lat[L - ell - 1] = cur
    return cur, lat        # leaves, {level -> (B, n_nodes) features}; level 0 == leaf tokens


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    rng = np.random.default_rng(4242)
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    canon = np.ascontiguousarray(rules[DEPTH - 1][:, 0, :])
    ib = build_inverse_maps(rules)[-1]
    tt = MC.true_tables(rules, DEPTH, S, V, M, 5)
    truth = {e: set(flats_of(tt[e])) for e in (2, 3, 4, 5)}
    parents = {}
    for ell in (2, 3, 4, 5):
        d = {}
        for row, f in zip(tt[ell]["flat"], tt[ell]["feature"]):
            d.setdefault(tuple(int(x) for x in row), set()).add(int(f))
        parents[ell] = d
    F = feature_expansions(rules, DEPTH, V, M, 5)
    out = {"world": {"v": V, "s": S, "depth": DEPTH, "m": M, "rule_seed": RULE_SEED,
                     "support": SUPPORT, "mine_cap": MINE_CAP}}

    print("=" * 92)
    print("PHASE 0 — THE QUOTIENT, sized offline   (enharmonic Q0)")
    print("=" * 92)

    # ---- [0] the corpus and the two replay gates --------------------------------------- #
    print("\n[0] THE CORPUS AND THE REPLAY GATES")
    arms = load_arms(truth)
    nok = nbad = nord = nord_bad = 0
    for a in arms:
        for r in a["gate"]:
            ok = (r["n_recon"] == r["n_logged"]
                  and abs(r["recall_recon"] - (r["recall_logged"] or 0.0)) < 1e-9
                  and (r["prec_logged"] is None
                       or abs((r["prec_recon"] or 0.0) - r["prec_logged"]) < 1e-9))
            nok += ok
            nbad += (not ok)
            if not ok:
                print(f"    B-1' FAIL {a['tag']}/{a['arm']} {r}")
            if r["order_ok"] is not None:
                nord += 1
                nord_bad += (not r["order_ok"])
    print(f"    arms {len(arms)} ({sum(1 for a in arms if a['exact'])} exact-DP)   "
          f"commit events {nok + nbad}")
    print(f"    GATE B-1'  (n_entries / tab_recall / tab_precision) : {nok}/{nok + nbad}")
    print(f"    GATE ORDER (replayed row order == logged true_mask) : {nord - nord_bad}/{nord}")
    lv_ct = collections.Counter(r["level"] for a in arms for r in a["gate"])
    print(f"    by level: {dict(sorted(lv_ct.items()))}")
    out["gates"] = {"n_arms": len(arms), "b1_pass": nok, "b1_total": nok + nbad,
                    "order_pass": nord - nord_bad, "order_total": nord,
                    "commits_by_level": {str(k): v for k, v in sorted(lv_ct.items())}}

    # ---- [1] the cover, exact ----------------------------------------------------------- #
    print("\n[1] THE COVER — the true 'partition' is not one")
    print(f"    {'level':>5} {'true rows':>10} {'distinct |T_l|':>15} {'multi-parented':>15} "
          f"{'parent-count histogram':>34}")
    cover = {}
    for ell in (2, 3, 4, 5):
        h = collections.Counter(len(v) for v in parents[ell].values())
        multi = sum(n for k, n in h.items() if k > 1)
        cover[ell] = {"n_true_rows": int(tt[ell]["flat"].shape[0]),
                      "n_distinct": len(parents[ell]), "n_multi_parented": multi,
                      "hist": {str(k): int(n) for k, n in sorted(h.items())}}
        print(f"    {ell:>5} {tt[ell]['flat'].shape[0]:>10,} {len(parents[ell]):>15,} "
              f"{multi:>15,} {str(dict(sorted(h.items()))):>34}")
    print("    (the two L1 rule collisions propagate: |T_l| < v*m*(entries per child)^s at every")
    print("     rung, and the shortfall IS the multiply-parented set)")

    # mass-weighted: the DGP marginal at each level's own mining node
    print("\n    the same by observation MASS, at each level's own mining node "
          f"(n = {N_MARG:,} derivations):")
    print(f"    {'level':>5} {'node':>5} {'mass on true keys':>18} {'mass multi-parented':>20} "
          f"{'| of the true mass':>19}")
    era_node = {ell: mining_node(ell - 1, LADDER[ell - 1], ell) for ell in (2, 3, 4, 5)}
    codes_true = {ell: {tuple(int(x) for x in r): parents[ell][tuple(int(x) for x in r)]
                        for r in np.unique(tt[ell]["flat"], axis=0)} for ell in (2, 3, 4)}
    mass = {}
    roots = rng.integers(0, V, size=N_MARG)
    pf = MC.exact_features(sample_derivations(rules, roots, S, rng), ib, V, S)
    for ell in (2, 3, 4):
        node, span = era_node[ell], S ** (ell - 1)
        blk = pf[:, node * span:(node + 1) * span]
        cnt = collections.Counter(tuple(int(x) for x in r) for r in blk)
        m_true = sum(c for k, c in cnt.items() if k in codes_true[ell]) / N_MARG
        m_multi = sum(c for k, c in cnt.items()
                      if len(codes_true[ell].get(k, ())) > 1) / N_MARG
        mass[ell] = {"node": node, "mass_true": m_true, "mass_multi": m_multi}
        print(f"    {ell:>5} {node:>5} {m_true:>18.4f} {m_multi:>20.4f} "
              f"{m_multi / max(m_true, 1e-12):>19.4f}")
    del pf
    out["cover"] = {"by_rows": {str(k): v for k, v in cover.items()},
                    "by_mass": {str(k): v for k, v in mass.items()}}

    # ---- [2] the category coordinates ---------------------------------------------------- #
    print("\n[2] THE CATEGORY COORDINATES")
    print("\n    (B) THE TOKEN CLASS — the set of level-l features a written program derives.")
    print("        Computed by closure of `possible_sets`' own composition step over the")
    print("        WRITABLE blocks, and validated against enumeration of the true tables.")
    clos = class_closure(rules, canon, 6)
    enum_ok = {}
    for ell in (2, 3, 4, 5):
        ks = sorted(parents[ell])
        tc = token_classes(ks, ell, rules, canon)
        enum_ok[ell] = (len(set(tc)) == clos[str(ell)]["n_classes"])
        sup = sum(1 for k, c in zip(ks, tc) if not parents[ell][k] <= c)
        clos[str(ell)]["n_classes_enumerated"] = len(set(tc))
        clos[str(ell)]["superset_violations"] = int(sup)
    print(f"        {'level':>5} {'|C_l|  (closure)':>17} {'(enumerated)':>14} "
          f"{'legal class-PAIRS':>19} {'flat |T_l|':>14} {'shrink':>12}")
    flat_sizes = {2: 14, 3: 56, 4: 816, 5: len(truth[5])}
    flat_sizes[6] = level_size(rules, F, DEPTH, V, M, 6)
    for ell in range(2, 7):
        c = clos[str(ell)]
        pr = c["n_legal_class_pairs"]
        print(f"        {ell:>5} {c['n_classes']:>17} "
              f"{str(c.get('n_classes_enumerated', '-')):>14} {pr:>19} "
              f"{flat_sizes[ell]:>14,} {flat_sizes[ell] / pr:>11,.0f}x")
    print(f"        L1 alphabet: {clos['1']['n_classes']} classes over "
          f"{clos['n_writable_blocks']} writable blocks "
          f"(v = {V}: the bottom rule collision fuses two)")
    print(f"        enumeration agrees with closure at every level: "
          f"{all(enum_ok.values())};  parent-set subset of token class always: "
          f"{all(clos[str(e)]['superset_violations'] == 0 for e in (2, 3, 4, 5))}")
    print("        the class map is CLOSED under composition (the closure is a fixed point of")
    print("        the DP's own step), so the ratchet restates as C[l+1] subset C[l] x C[l] and")
    print("        the key of a level-(l+1) row is a pair of level-l CLASS ids.")

    print("\n    (A) THE PARENT-FEATURE key (`fourwall`'s re-key basis, the parent spec's fact 4):")
    par_pairs = {}
    for ell in range(2, 7):
        layer = rules[DEPTH - ell]
        par_pairs[ell] = len({(int(layer[f, r, 0]), int(layer[f, r, 1]))
                              for f in range(V) for r in range(M)})
    print(f"        distinct legal (parent, parent) pairs per level: "
          f"{[par_pairs[e] for e in range(2, 7)]}  (v*m = {V * M} is the ceiling)")
    print("        single-valued only if the GENERATIVE label is supplied; read off a flat")
    print("        tuple it is set-valued wherever the cover is (see [1]).")

    print("\n    the expansion set a class must carry (the executor's choice at execution time)")
    print("    — measured |F[l][f=0]|; the collision-free bound is m^((s^(l-1)-1)/(s-1)):")
    print("       " + "  ".join(
        f"L{e}: {len(F[e][0]):,} (bound {M ** ((S ** (e - 1) - 1) // (S - 1)):,})"
        for e in range(1, 6)))
    out["classes"] = {"closure": clos, "parent_feature_pairs":
                      {str(k): v for k, v in par_pairs.items()},
                      "flat_sizes": {str(k): int(v) for k, v in flat_sizes.items()},
                      "expansion_per_class": {str(e): len(F[e][0]) for e in range(1, 6)}}

    # ---- [2b] arrival under each key ----------------------------------------------------- #
    print("\n[2b] ARRIVAL — E[# keys at support 3] vs observations, under each key, at each")
    print("     level's own mining node. Same Poisson model as `tutti/sizing` gate G-1.")
    print("     Three counting rules for the set-valued parent key: ALL (every pair in the")
    print("     cross-product gets a count), UNI (skip an observation whose halves are not")
    print("     both unambiguous), GEN (the generative label — unrecoverable, the `given_cat`")
    print("     ceiling). The token key is a FUNCTION of the tokens: no rule is needed.")
    cls_index = {ell: {c: i for i, c in enumerate(
        [frozenset(x) for x in clos[str(ell)]["classes"]])} for ell in range(2, 6)}
    lvl1_cls = {}
    bottom = rules[DEPTH - 1]
    for a in range(V):
        for b in range(V):
            c = frozenset(f for f in range(V) for r in range(M)
                          if int(bottom[f, r, 0]) == a and int(bottom[f, r, 1]) == b)
            if c:
                lvl1_cls[(a, b)] = c

    def half_class(tok_half, ell_half):
        """token class of a raw token half of level `ell_half` (>=1)."""
        P = possible_sets(rules[DEPTH - ell_half:], tok_half, S)[-1][:, 0, :]
        return P

    arrival = {}
    N_GRID = (1_600, 16_000, 160_000, 1_600_000)
    print(f"\n     {'level':>5} {'key':>16} {'#legal keys':>12} "
          + " ".join(f"{'E@' + f'{n:,}':>14}" for n in N_GRID) + f" {'N for 22 keys':>14}")
    for ell in (4, 5, 6):
        node = mining_node(min(ell - 1, 5), LADDER[min(ell - 1, 5)], ell)
        span = S ** (ell - 1)
        r2 = rng.integers(0, V, size=N_MARG)
        leaves, lat = sample_with_latents(rules, r2, S, rng)
        tok = leaves[:, node * span * S:(node + 1) * span * S]
        half = tok.shape[1] // 2
        # token key
        cl = half_class(tok[:, :half], ell - 1), half_class(tok[:, half:], ell - 1)
        keyt = collections.Counter(
            (frozenset(np.nonzero(u)[0].tolist()), frozenset(np.nonzero(w)[0].tolist()))
            for u, w in zip(cl[0], cl[1]))
        # parent key, three rules
        gl = lat[ell - 1][:, node * S], lat[ell - 1][:, node * S + 1]
        keyg = collections.Counter(zip(gl[0].tolist(), gl[1].tolist()))
        pset = [[frozenset(np.nonzero(u)[0].tolist()) for u in cl[i]] for i in (0, 1)]
        keya, keyu = collections.Counter(), collections.Counter()
        for u, w in zip(pset[0], pset[1]):
            for x in u:
                for y in w:
                    keya[(x, y)] += 1
            if len(u) == 1 and len(w) == 1:
                keyu[(next(iter(u)), next(iter(w)))] += 1
        # flat key
        pfe = MC.exact_features(leaves, ib, V, S)
        keyf = collections.Counter(tuple(int(x) for x in r)
                                   for r in pfe[:, node * span:(node + 1) * span])
        legal_par = {(int(rules[DEPTH - ell][f, r, 0]), int(rules[DEPTH - ell][f, r, 1]))
                     for f in range(V) for r in range(M)}
        rows_a = []
        for name, cnt, legal in (("token", keyt, None), ("parent-GEN", keyg, legal_par),
                                 ("parent-ALL", keya, legal_par),
                                 ("parent-UNI", keyu, legal_par),
                                 ("flat", keyf, truth.get(ell))):
            tot = sum(cnt.values())
            if legal is None:
                ps = [c / tot for c in cnt.values()]
            else:
                ps = [c / tot for k, c in cnt.items() if k in legal]
            es = [sum(pge(n * p) for p in ps) for n in N_GRID]

            def solve(target, ps=ps):
                lo_, hi_ = 1e2, 1e9
                for _ in range(70):
                    md = (lo_ + hi_) / 2
                    if sum(pge(md * p) for p in ps) < target:
                        lo_ = md
                    else:
                        hi_ = md
                return hi_
            n22 = solve(min(22, len(ps))) if ps else float("nan")
            rows_a.append({"level": ell, "key": name, "n_legal_seen": len(ps),
                           "E": dict(zip(map(str, N_GRID), es)), "N_for_22": n22})
            print(f"     {ell:>5} {name:>16} {len(ps):>12,} "
                  + " ".join(f"{e:>14.3f}" for e in es) + f" {n22:>14,.0f}")
        arrival[ell] = rows_a
        del leaves, lat, pfe
    out["arrival"] = {str(k): v for k, v in arrival.items()}
    print("\n     (the flat row at L5 is the `tutti/sizing` result re-derived on a 2M marginal;")
    print("      its absolute N is ~20% optimistic against that lane's 8M estimate. |T6| is not")
    print("      enumerable, so the L6 flat row counts every observed key, true and junk alike —")
    print("      an upper bound on E and a lower bound on N.)")

    # ---- [2c] the ratchet in class coordinates ------------------------------------------- #
    print("\n[2c] THE RATCHET, RE-MEASURED IN CLASS COORDINATES — the parent spec's fact 5 as a")
    print("     measurement rather than an argument. Over each arm's OWN committed level-(l-1)")
    print("     book, the fraction of DGP-drawn level-l observations at the mining node whose")
    print("     two halves the book can look up: by FLAT tuple (what `Miner.build` does) versus")
    print("     by TOKEN CLASS (what the same lookup does once the book is quotiented).")
    N_GEN = 200_000
    gen = {}
    for ell in (3, 4, 5):
        node, span = mining_node(ell - 1, LADDER[ell - 1], ell), S ** (ell - 1)
        r3 = rng.integers(0, V, size=N_GEN)
        lv3 = sample_derivations(rules, r3, S, rng)
        tok = lv3[:, node * span * S:(node + 1) * span * S]
        h = tok.shape[1] // 2
        cl = [possible_sets(rules[DEPTH - (ell - 1):], tok[:, :h], S)[-1][:, 0, :],
              possible_sets(rules[DEPTH - (ell - 1):], tok[:, h:], S)[-1][:, 0, :]]
        clh = [[frozenset(np.nonzero(u)[0].tolist()) for u in c] for c in cl]
        fe = MC.exact_features(lv3, ib, V, S)[:, node * span:(node + 1) * span]
        fl = [[tuple(int(z) for z in r[:span // 2]) for r in fe],
              [tuple(int(z) for z in r[span // 2:]) for r in fe]]
        gen[ell] = (clh, fl)
        del lv3, fe
    ratch = []
    for a in arms:
        for ell in (3, 4, 5):
            lo = a["books"].get(ell - 1)
            if not lo:
                continue
            clh, fl = gen[ell]
            flat_ok = set(lo)
            cls_ok = set(token_classes(lo, ell - 1, rules, canon)) - {frozenset()}
            nf = sum(1 for i in range(N_GEN) if fl[0][i] in flat_ok and fl[1][i] in flat_ok)
            nc = sum(1 for i in range(N_GEN) if clh[0][i] in cls_ok and clh[1][i] in cls_ok)
            ratch.append({"tag": a["tag"], "arm": a["arm"], "level": ell,
                          "lower_rows": len(lo), "lower_classes": len(cls_ok),
                          "buildable_flat": nf / N_GEN, "buildable_class": nc / N_GEN})
    print(f"     {'level':>5} {'arms':>5} {'lower rows':>11} {'lower classes':>14} "
          f"{'P(build | flat)':>16} {'P(build | class)':>17} {'lift':>9}")
    for ell in (3, 4, 5):
        rr = [r for r in ratch if r["level"] == ell]
        if not rr:
            continue
        mf = np.median([r["buildable_flat"] for r in rr])
        mc = np.median([r["buildable_class"] for r in rr])
        print(f"     {ell:>5} {len(rr):>5} {np.median([r['lower_rows'] for r in rr]):>11.0f} "
              f"{np.median([r['lower_classes'] for r in rr]):>14.0f} {mf:>16.4f} "
              f"{mc:>17.4f} {mc / max(mf, 1e-9):>8.1f}x")
    out["ratchet_in_class"] = ratch

    # ---- [3] the alias audit ------------------------------------------------------------- #
    print("\n[3] THE ALIAS AUDIT — what the banked use record can and cannot say")
    units = []
    for a in arms:
        if not a["exact"]:
            continue
        res = a["res"]
        for ell, rows in sorted(a["books"].items()):
            U, skip = [], False
            for i in range(len(res["log"]["cycle"])):
                vec = (((res["log"]["entry"][i] or {}).get("hist") or {})
                       .get("beam") or {}).get(str(ell))
                if vec is None:
                    continue
                if len(vec) != len(rows):      # a table that was swapped mid-run (census_extend)
                    skip = True
                    break
                if sum(vec) > 0:
                    U.append(vec)
            if skip or len(U) < 10 or len(rows) < 3:
                continue
            units.append({"tag": a["tag"], "arm": a["arm"], "level": ell,
                          "U": np.array(U, float), "rows": rows,
                          "fcls": [frozenset(parents[ell].get(k, set())) for k in rows],
                          "tcls": token_classes(rows, ell, rules, canon),
                          "leaf": [tuple(int(z) for z in r)
                                   for r in entry_leaf_rows(rows, canon)]})
    print(f"    units (arm x level with a stable, >=10-cycle record): {len(units)}  "
          f"{dict(sorted(collections.Counter(u['level'] for u in units).items()))}")
    n_skip = sum(1 for a in arms if a["exact"] for ell in a["books"]) - len(units)
    print(f"    ({n_skip} arm-levels dropped: table swapped mid-run, <10 recorded cycles, or "
          f"<3 rows)")

    def sim_mats(U):
        sh = U / np.maximum(U.sum(1, keepdims=True), 1e-12)
        rk = np.argsort(np.argsort(sh, axis=0), axis=0).astype(float)
        rk = (rk - rk.mean(0)) / (rk.std(0) + 1e-12)
        prof = U / (U.sum(0, keepdims=True) + 1e-12)
        cos = prof.T @ prof / np.sqrt((prof ** 2).sum(0)[:, None]
                                      * (prof ** 2).sum(0)[None, :] + 1e-24)
        ms = np.log10(sh.mean(0) + 1e-9)
        return {"spear": (rk.T @ rk) / U.shape[0], "cos": cos,
                "lev": -np.abs(ms[:, None] - ms[None, :])}

    def T_of(mat, lab):
        idx = [i for i, c in enumerate(lab) if len(c) > 0]
        w, b = [], []
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                (w if lab[i] == lab[j] else b).append(mat[i, j])
        if not w or not b:
            return None
        return float(np.mean(w) - np.mean(b))

    prng = np.random.default_rng(7)
    audit = []
    print(f"\n    {'labels':>7} {'statistic':>10} {'level':>5} {'arms':>5} {'T (within-between)':>19} "
          f"{'null mean+-sd':>20} {'z':>8} {'p(one-sided)':>13}")
    for labname in ("fcls", "tcls"):
        for stat in ("spear", "cos", "lev"):
            for lvl in (2, 3, 4):
                us = [u for u in units if u["level"] == lvl]
                obs, perm = [], []
                for u in us:
                    mat = sim_mats(u["U"])[stat]
                    lab = u[labname]
                    idx = [i for i, c in enumerate(lab) if len(c) > 0]
                    t = T_of(mat, lab)
                    if t is None:
                        continue
                    obs.append(t)
                    sub = [lab[i] for i in idx]
                    ps = []
                    for _ in range(N_PERM):
                        pl = list(lab)
                        sh = prng.permutation(len(idx))
                        for x, i in enumerate(idx):
                            pl[i] = sub[sh[x]]
                        ps.append(T_of(mat, pl))
                    perm.append(np.array(ps, float))
                if not obs:
                    continue
                O = float(np.mean(obs))
                Pm = np.mean(np.stack(perm), 0)
                z = (O - Pm.mean()) / (Pm.std() + 1e-12)
                p = (1 + int((Pm >= O).sum())) / (len(Pm) + 1)
                audit.append({"labels": labname, "stat": stat, "level": lvl, "n_arms": len(obs),
                              "T": O, "null_mean": float(Pm.mean()), "null_sd": float(Pm.std()),
                              "z": float(z), "p_greater": p})
                print(f"    {labname:>7} {stat:>10} {lvl:>5} {len(obs):>5} {O:>+19.4f} "
                      f"{f'{Pm.mean():+.4f} +- {Pm.std():.4f}':>20} {z:>+8.2f} {p:>13.4f}")
    out["alias_audit"] = audit

    print("\n    two stricter controls on the same statistics, token labels only. `strat` permutes")
    print("    labels WITHIN strata of |class|, so 'ambiguous rows are used more' cannot produce")
    print("    a hit; `drop-top` deletes each unit's single highest-mass row first.")
    audit2 = []
    print(f"    {'statistic':>10} {'control':>10} {'level':>5} {'arms':>5} {'T':>10} "
          f"{'null mean+-sd':>20} {'z':>8} {'p(one-sided)':>13}")
    for stat in ("spear", "lev"):
        for mode in ("strat", "drop-top"):
            for lvl in (2, 3, 4):
                obs, perm = [], []
                for u in units:
                    if u["level"] != lvl:
                        continue
                    U, lab = u["U"], list(u["tcls"])
                    if mode == "drop-top":
                        k = int(np.argmax(U.sum(0)))
                        keep = [i for i in range(U.shape[1]) if i != k]
                        U, lab = U[:, keep], [lab[i] for i in keep]
                        if U.shape[1] < 3 or U.sum() == 0 or (U.sum(1) == 0).any():
                            continue
                    mat = sim_mats(U)[stat]
                    if not np.isfinite(mat).all():
                        continue
                    t = T_of(mat, lab)
                    if t is None:
                        continue
                    obs.append(t)
                    idx = [i for i, c in enumerate(lab) if len(c) > 0]
                    if mode == "strat":
                        strat = collections.defaultdict(list)
                        for i in idx:
                            strat[len(lab[i])].append(i)
                    else:
                        strat = {0: idx}
                    ps = []
                    for _ in range(N_PERM):
                        pl = list(lab)
                        for _k, gg in strat.items():
                            vals = [lab[i] for i in gg]
                            sh = prng.permutation(len(gg))
                            for x, i in enumerate(gg):
                                pl[i] = vals[sh[x]]
                        r = T_of(mat, pl)
                        ps.append(t if r is None else r)
                    perm.append(np.array(ps, float))
                if not obs:
                    continue
                O = float(np.mean(obs))
                Pm = np.mean(np.stack(perm), 0)
                z = (O - Pm.mean()) / (Pm.std() + 1e-12)
                p = (1 + int((Pm >= O).sum())) / (len(Pm) + 1)
                audit2.append({"stat": stat, "control": mode, "level": lvl, "n_arms": len(obs),
                               "T": O, "null_mean": float(Pm.mean()),
                               "null_sd": float(Pm.std()), "z": float(z), "p_greater": p})
                print(f"    {stat:>10} {mode:>10} {lvl:>5} {len(obs):>5} {O:>+10.4f} "
                      f"{f'{Pm.mean():+.4f} +- {Pm.std():.4f}':>20} {z:>+8.2f} {p:>13.4f}")
    out["alias_audit_controls"] = audit2

    print("\n    the sharpest cell — the use-share ratio min/max WITHIN a pair of rows, split by")
    print("    how interchangeable the pair actually is:")
    buck = {"identical program": [], "same token class": [], "different token class": []}
    for u in units:
        tot = u["U"].sum(0)
        share = tot / tot.sum()
        for x in range(len(u["rows"])):
            for y in range(x + 1, len(u["rows"])):
                r = (min(share[x], share[y]) + 1e-12) / (max(share[x], share[y]) + 1e-12)
                if u["leaf"][x] == u["leaf"][y]:
                    buck["identical program"].append(r)
                elif u["tcls"][x] == u["tcls"][y] and len(u["tcls"][x]) > 0:
                    buck["same token class"].append(r)
                else:
                    buck["different token class"].append(r)
    print(f"    {'pair kind':>22} {'n':>7} {'median':>9} {'q25':>9} {'q75':>9} "
          f"{'frac < 0.01':>12}")
    ratio = {}
    for k, vv in buck.items():
        x = np.array(vv)
        ratio[k] = {"n": len(x), "median": float(np.median(x)),
                    "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
                    "frac_lt_0.01": float(np.mean(x < 0.01))}
        print(f"    {k:>22} {len(x):>7} {np.median(x):>9.4f} {np.quantile(x, .25):>9.4f} "
              f"{np.quantile(x, .75):>9.4f} {np.mean(x < 0.01):>12.2f}")
    out["use_share_ratio"] = ratio

    # ---- [4] forced transfer ------------------------------------------------------------- #
    print("\n[4] FORCED TRANSFER — `fourwall.entry_profile` verbatim, offline and exact")
    pools = {}

    def pool(era_level, n=N_POOL, seed=31337):
        if era_level not in pools:
            ctx = {"name": f"L{era_level}n{LADDER[era_level]}", "level": era_level,
                   "nodes": [LADDER[era_level]]}
            pools[era_level] = context_instances(rules, ctx, n, S, DEPTH, V, M, seed=seed)
        return pools[era_level]

    print("\n    (a) THE RESOLUTION CEILING at each level's own damage cell: how many of the")
    print("        level's token classes any forced-transfer probe can tell apart, one")
    print("        representative program per class, at two instance counts.")
    ceil_rows = []
    for ell in (2, 3, 4):
        keys = sorted(parents[ell])
        tc = token_classes(keys, ell, rules, canon)
        reps = {}
        for k, c in zip(keys, tc):
            reps.setdefault(c, k)
        cls = sorted(reps, key=lambda c: sorted(c))
        ent = [reps[c] for c in cls]
        for era in (ell, ell - 1, ell + 1):
            if era not in LADDER:
                continue
            node = mining_node(era, LADDER[era], ell)
            roots_, x_ = pool(era)
            for n in (512, N_POOL):
                P, g = W.entry_profile(rules, x_[:n], roots_[:n], ent, node, ell, S, canon)
                byp = collections.defaultdict(list)
                for i, c in enumerate(cls):
                    byp[P[i].tobytes()].append(sorted(c))
                dead = [sorted(cls[i]) for i in range(len(cls)) if not P[i].any()]
                coll = [v for k_, v in byp.items() if len(v) > 1 and any(v)]
                nres = len(byp) - (1 if dead else 0)
                ceil_rows.append({"level": ell, "era": era, "node": node, "n": n,
                                  "n_token_classes": len(cls), "n_resolvable": nres,
                                  "dead": dead, "collapsed": coll, "gradings": g})
                print(f"        L{ell} era{era} node{node} N={n:>5}: |C|={len(cls):>3} -> "
                      f"resolvable {nres:>3}   dead {dead}   collapsed {coll}")
    out["resolution_ceiling"] = ceil_rows
    print("        the collapse is STRUCTURAL, not sample-limited: N=512 and N=4096 give the")
    print("        same partition everywhere. It is the demand at the node, not the probe count.")

    print("\n    (b) THE ARMS' BOOKS — token-space vs feature-space precision, and what a probe")
    print("        over the arm's own committed rows recovers.")
    tr_rows = []
    prof_cache = {}
    tc_groups = tc_bad = 0        # GATE: same token class => bit-identical success profile
    for a in arms:
        for ell, rows in sorted(a["books"].items()):
            if not rows:
                continue
            era = ell
            node = mining_node(era, LADDER[era], ell)
            roots_, x_ = pool(era)
            key = (a["tag"], a["arm"], ell)
            P, g = W.entry_profile(rules, x_, roots_, rows, node, ell, S, canon)
            prof_cache[key] = P
            fcls = [frozenset(parents[ell].get(k, set())) for k in rows]
            tcls = token_classes(rows, ell, rules, canon)
            leaf = [tuple(int(z) for z in r) for r in entry_leaf_rows(rows, canon)]
            n_f = sum(1 for c in fcls if c)
            n_t = sum(1 for c in tcls if c)
            # the demand partition at N_POOL: rows grouped by identical success profile
            grp = collections.defaultdict(list)
            for i in range(len(rows)):
                grp[P[i].tobytes()].append(i)
            live = [i for i in range(len(rows)) if P[i].any()]
            bycls = collections.defaultdict(list)
            for i in range(len(rows)):
                bycls[tcls[i]].append(i)
            for c, ix in bycls.items():
                if len(ix) < 2:
                    continue
                tc_groups += 1
                tc_bad += (not all((P[ix[0]] == P[j]).all() for j in ix[1:]))
            tr_rows.append({
                "tag": a["tag"], "arm": a["arm"], "level": ell, "n_rows": len(rows),
                "n_true_feature": n_f, "n_true_token": n_t,
                "prec_feature": n_f / len(rows), "prec_token": n_t / len(rows),
                "n_live": len(live), "coverage": float(P.any(0).mean()),
                "n_probe_groups": len({P[i].tobytes() for i in live}),
                "n_token_classes": len({tcls[i] for i in live}),
                "n_distinct_programs": len({leaf[i] for i in live}),
                "gradings": g})
    for ell in (2, 3, 4):
        rr = [r for r in tr_rows if r["level"] == ell]
        if not rr:
            continue
        print(f"        L{ell}: arms {len(rr):>2}  rows {np.median([r['n_rows'] for r in rr]):>4.0f} "
              f"(med)   precision FEATURE {np.median([r['prec_feature'] for r in rr]):.2f} "
              f"-> TOKEN {np.median([r['prec_token'] for r in rr]):.2f}   "
              f"coverage {np.median([r['coverage'] for r in rr]):.2f}   "
              f"probe groups {np.median([r['n_probe_groups'] for r in rr]):.0f} of "
              f"{np.median([r['n_live'] for r in rr]):.0f} live rows")
    print(f"        GATE (same token class => bit-identical success profile over "
          f"{N_POOL} instances): {tc_groups - tc_bad}/{tc_groups} groups")
    out["transfer_by_arm"] = tr_rows
    out["gates"]["token_class_profile_groups"] = tc_groups
    out["gates"]["token_class_profile_pass"] = tc_groups - tc_bad

    print("\n    (c) THE PROBE BUDGET — instances needed for the probe partition to reach its own")
    print("        N=4096 fixed point, and the gradings that costs. `wgt` restricts the rows")
    print("        probed to those carrying >=90% of the beam's cumulative use mass.")
    budget = []
    NS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    for a in arms:
        res = a["res"]
        for ell, rows in sorted(a["books"].items()):
            key = (a["tag"], a["arm"], ell)
            if key not in prof_cache or len(rows) < 2:
                continue
            P = prof_cache[key]
            tot = np.zeros(len(rows))
            for i in range(len(res["log"]["cycle"])):
                vec = (((res["log"]["entry"][i] or {}).get("hist") or {})
                       .get("beam") or {}).get(str(ell))
                if vec is not None and len(vec) == len(rows):
                    tot += np.array(vec, float)
            order = np.argsort(-tot)
            keep = []
            acc = 0.0
            for i in order:
                keep.append(int(i))
                acc += tot[i]
                if tot.sum() > 0 and acc / tot.sum() >= 0.90:
                    break
            for tagn, sel in (("all", list(range(len(rows)))), ("wgt", keep)):
                ref = collections.defaultdict(list)
                for i in sel:
                    ref[P[i].tobytes()].append(i)
                refp = {frozenset(v) for v in ref.values()}
                hit = None
                for n in NS:
                    g2 = collections.defaultdict(list)
                    for i in sel:
                        g2[P[i][:n].tobytes()].append(i)
                    if {frozenset(v) for v in g2.values()} == refp:
                        hit = n
                        break
                budget.append({"tag": a["tag"], "arm": a["arm"], "level": ell, "sel": tagn,
                               "n_rows": len(sel), "n_min": hit,
                               "gradings": (None if hit is None else hit * len(sel)),
                               "n_groups": len(refp)})
    print(f"        {'level':>5} {'sel':>5} {'arms':>5} {'rows (med)':>11} {'N* (med)':>9} "
          f"{'N* (max)':>9} {'gradings (med)':>15} {'gradings (max)':>15}")
    for ell in (2, 3, 4):
        for sel in ("all", "wgt"):
            bb = [b for b in budget if b["level"] == ell and b["sel"] == sel and b["n_min"]]
            if not bb:
                continue
            print(f"        {ell:>5} {sel:>5} {len(bb):>5} "
                  f"{np.median([b['n_rows'] for b in bb]):>11.0f} "
                  f"{np.median([b['n_min'] for b in bb]):>9.0f} "
                  f"{max(b['n_min'] for b in bb):>9} "
                  f"{np.median([b['gradings'] for b in bb]):>15.0f} "
                  f"{max(b['gradings'] for b in bb):>15}")
    out["probe_budget"] = budget

    print("\n    (d) THE SOLVE TAX — probes that cannot pay. A grading is wasted when the")
    print("        instance is repaired by no row in the book (nothing to transfer FROM), and a")
    print("        row is unprobeable when it repairs nothing at the cell.")
    tax = []
    for r in tr_rows:
        tax.append({"level": r["level"], "uncovered": 1 - r["coverage"],
                    "dead_rows": (r["n_rows"] - r["n_live"]) / r["n_rows"]})
    print(f"        {'level':>5} {'instances no row repairs':>26} {'rows that repair nothing':>26}")
    for ell in (2, 3, 4):
        tt_ = [t for t in tax if t["level"] == ell]
        print(f"        {ell:>5} {np.median([t['uncovered'] for t in tt_]):>26.3f} "
              f"{np.median([t['dead_rows'] for t in tt_]):>26.3f}")
    print("        (medians over arms; the exact grader's own cost is one grading per")
    print("         (row, instance) pair, so `wgt` above is the lever that matters)")
    out["solve_tax"] = tax

    print("\n    (e) THE LOSS MARGIN — `fourwall.merge_candidates`' two-directional transfer")
    print("        loss, split by the true relation between the pair's token classes.")
    margin = collections.defaultdict(list)
    for a in arms:
        for ell, rows in sorted(a["books"].items()):
            key = (a["tag"], a["arm"], ell)
            if key not in prof_cache:
                continue
            P = prof_cache[key]
            tc = token_classes(rows, ell, rules, canon)
            live = [i for i in range(len(rows)) if P[i].any()]
            for x in range(len(live)):
                for y in range(x + 1, len(live)):
                    i, j = live[x], live[y]
                    inter = int((P[i] & P[j]).sum())
                    loss = max(1 - inter / P[j].sum(), 1 - inter / P[i].sum())
                    if tc[i] == tc[j]:
                        k = "equal"
                    elif tc[i] < tc[j] or tc[j] < tc[i]:
                        k = "subset (one multiply-parented)"
                    elif tc[i] & tc[j]:
                        k = "overlapping"
                    else:
                        k = "disjoint"
                    margin[k].append(float(loss))
    print(f"        {'relation':>32} {'n pairs':>9} {'min':>8} {'median':>8} {'max':>8}")
    mg = {}
    for k in ("equal", "subset (one multiply-parented)", "overlapping", "disjoint"):
        x = np.array(margin.get(k, []))
        if not len(x):
            continue
        mg[k] = {"n": len(x), "min": float(x.min()), "median": float(np.median(x)),
                 "max": float(x.max())}
        print(f"        {k:>32} {len(x):>9} {x.min():>8.4f} {np.median(x):>8.4f} "
              f"{x.max():>8.4f}")
    out["loss_margin"] = mg

    # ---- [5] junk under transfer --------------------------------------------------------- #
    print("\n[5] JUNK UNDER TRANSFER — do the `true_mask`-false rows sit alone?")
    jrows = []
    for a in arms:
        for ell, rows in sorted(a["books"].items()):
            key = (a["tag"], a["arm"], ell)
            if key not in prof_cache:
                continue
            P = prof_cache[key]
            fcls = [frozenset(parents[ell].get(k, set())) for k in rows]
            tcls = token_classes(rows, ell, rules, canon)
            leaf = [tuple(int(z) for z in r) for r in entry_leaf_rows(rows, canon)]
            true_leaf = {tuple(int(z) for z in r)
                         for r in entry_leaf_rows(sorted(parents[ell]), canon)}
            for i in range(len(rows)):
                if fcls[i]:
                    continue
                bestj = 0.0
                for j in range(len(rows)):
                    if j == i or not fcls[j] or not (P[i].any() or P[j].any()):
                        continue
                    u = int((P[i] | P[j]).sum())
                    if u:
                        bestj = max(bestj, int((P[i] & P[j]).sum()) / u)
                jrows.append({"level": ell, "token_true": bool(tcls[i]),
                              "leaf_is_a_true_program": leaf[i] in true_leaf,
                              "repairs": int(P[i].sum()), "n_inst": int(P.shape[1]),
                              "max_jaccard_with_a_true_row": bestj})
    print(f"    {'level':>5} {'junk rows':>10} {'token-legal':>12} {'leaf == a true key':>19} "
          f"{'repairs > 0':>12} {'max Jaccard w/ true row (med)':>30}")
    for ell in (2, 3, 4):
        jj = [j for j in jrows if j["level"] == ell]
        if not jj:
            continue
        print(f"    {ell:>5} {len(jj):>10} "
              f"{sum(j['token_true'] for j in jj) / len(jj):>12.3f} "
              f"{sum(j['leaf_is_a_true_program'] for j in jj) / len(jj):>19.3f} "
              f"{sum(j['repairs'] > 0 for j in jj) / len(jj):>12.3f} "
              f"{np.median([j['max_jaccard_with_a_true_row'] for j in jj]):>30.3f}")
    out["junk"] = jrows

    out["elapsed_s"] = time.time() - t_start
    with open(os.path.join(HERE, "phase0_cat.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(HERE, 'phase0_cat.json')}   ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
