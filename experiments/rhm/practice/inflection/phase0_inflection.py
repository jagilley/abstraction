"""PHASE 0 — size the INFLECTION RULE FAMILIES OFFLINE, before any GPU.

[`SPEC.md`](SPEC.md) makes one change below the tables: the synonym index at a level-1 node
stops being a uniform draw over the m spellings and becomes a function of a context variable.
The SPEC leaves *which* rule open (decision 2) and says "Q0 settles it; do not guess". This
file is that settlement, in `tutti/sizing`'s idiom: CPU-only numpy against the data-generating
process's own arithmetic, facts and tables, no interpretation, no substrate, no Modal.

THE CONSTRAINT THAT MAKES IT HARD. The world is v=8, s=2, m=2, depth 6 at `rule_seed=0`
(`generate_rules_distinct`). The per-(feature, context) choice is therefore ONE BIT, and the
obvious rule "the register selects the synonym index" gives only m = 2 contexts, where the
curriculum ladder (contexts practiced in {1,2,3}, one always held out) needs at least four.

WHAT IT MEASURES.
  [0] the world: the bottom table, its two collisions, and the fork invariant — the flat-key
      sizes at L2..L5 are functions of `rules[0..depth-2]` ONLY, so no bottom rule can move
      them. Shown by rebuilding the level tables over a randomised bottom.
  [1] the rule families, defined and counted: contexts, free parameters, distinct spelling
      patterns, and the sharing ratio against a plain (feature, context) table.
  [2] THE IDENTIFIABILITY LADDER, exact. For each family and each hypothesis class a renderer
      might carry, the number of practiced contexts that pins the held-out context — counted
      by enumerating every completion of the held-out column that the class admits, never by
      fitting anything.
  [3] lexical invisibility of the held-out context: does holding a context out remove any leaf
      tuple from the practiced corpus? (What the frozen reader would have to parse.)
  [4] parse ambiguity under the rule: the two colliding bottom tuples at `rule_seed=0`, whether
      the rule keeps both alive, and whether the context is ever NEEDED to parse a block to its
      feature — or ever able to.
  [5] the register's own readability: how many blocks pin the context, for a reader that knows
      the rule.
  [6] the strict grader: the fraction of a canon-writing executor's meaning-successes a strict
      spelling grader rejects, by damage level and by blocks written, on rule-spelled instances.
  [7] the d* ladder — `tall/`'s admissibility instrument — reproduced at (v=8, m=2) and (v=8,
      m=4), then re-measured at the widened worlds that would give "register = synonym index"
      four contexts, and at a structured (stem, ending) bottom layer.
  [8] what widening costs above level 1: |T_l| and the macro-table row counts by world; the
      SPEC's "arrival above level 1 unchanged" check is [0] and [7d].
  [9] agreement's realised (feature, context) cells, and whether a held-out cell is a
      held-out rule or a held-out production.
  [10] what this file cannot decide, and the measurement that would.

No GPU, no Modal, no substrate. Writes `phase0.json` beside this file.

Run from experiments/:  PYTHONPATH=. python3 rhm/practice/inflection/phase0_inflection.py
"""

import itertools
import json
import math
import os
import time

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.practice.ratchet import macros as MC
from rhm.practice.crystallize.units import corrupt_hier
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, possible_sets

HERE = os.path.dirname(os.path.abspath(__file__))

V, S, DEPTH, M = 8, 2, 6, 2        # the world every practice arm runs in
RULE_SEED = 0
PARAM_SEED = 11                    # the draw of the rule PARAMETERS (a, w, theta, ...)
LADDER = [(1, 25), (2, 12), (3, 6), (4, 3), (5, 1)]   # the arc's damage ladder (level, node)
# `tall/FILES.md` section 1, "the oracle's repair distance d* over the nested ladder,
# broken-only instances, measured on CPU" — the gate this file's harness must reproduce.
TALL_DSTAR = {(4, 2): [1.58, 2.25, 2.67], (4, 4): [1.13, 1.15, 1.17],
              (6, 2): [1.71, 2.37, 3.23, 4.52, 5.54], (6, 3): [1.33, 1.42, 1.33, 1.52, 1.73],
              (6, 4): [1.09, 1.12, 1.12, 1.17, 1.15]}
N_DSTAR = 800                      # instances per (world, ladder rung) for the d* ladder
N_SIM = 4000                       # instances for the spelling / parse simulations


# --------------------------------------------------------------------------- #
# the world's own arithmetic
# --------------------------------------------------------------------------- #

def bottom_codes(rules, v=V, s=S):
    """rules[-1] as leaf-tuple codes: (v, m) integers in [0, v**s)."""
    b = rules[-1]
    powers = v ** np.arange(s)
    return (b * powers).sum(-1)


def flats_of(table):
    return {tuple(int(x) for x in r) for r in table["flat"]}


def level_sizes(rules, depth, s, v, m, max_level=5):
    """(rows, distinct) per level of `true_tables` — the learner's table and the recall
    denominator. Depends on rules[0 .. depth-2] only; rules[depth-1] is the BOTTOM
    (level-1 feature -> leaf tuple) and is never read."""
    tt = MC.true_tables(rules, depth, s, v, m, max_level)
    out = {}
    for ell in range(2, max_level + 1):
        fl = tt[ell]["flat"]
        out[ell] = (int(fl.shape[0]), int(len(np.unique(fl, axis=0))))
    return out, tt


def macro_rows(v, m, ell):
    """`tall/FILES.md`'s doubly-exponential law: entries(l) = v*e(l), e(l) = m*e(l-1)**s."""
    e = 1
    for _ in range(2, ell + 1):
        e = m * e ** 2
    return v * e


# --------------------------------------------------------------------------- #
# the rule families.  A rule is a table K of shape (v, R): K[f, c] = the synonym index
# feature f takes in context c.
# --------------------------------------------------------------------------- #

def ctx_bits(c, d):
    return np.array([(c >> i) & 1 for i in range(d)], dtype=np.int64)


def fam_affine(v, d, seed, shared_w=False):
    """A: k_f(r) = a_f XOR <w_f, r> over GF(2)^d.  R = 2**d contexts."""
    rng = np.random.default_rng(seed)
    R = 2 ** d
    a = rng.integers(0, 2, size=v)
    if shared_w:
        ws = rng.integers(0, 2, size=d)
        while not ws.any():                       # w = 0 is the contextless rule
            ws = rng.integers(0, 2, size=d)
        w = np.tile(ws, (v, 1))
    else:
        w = rng.integers(0, 2, size=(v, d))
    K = np.zeros((v, R), dtype=np.int64)
    for c in range(R):
        K[:, c] = (a ^ (w * ctx_bits(c, d)[None, :]).sum(1) % 2)
    return K, {"a": a.tolist(), "w": w.tolist(), "d": d, "shared_w": shared_w}


def fam_threshold(v, R, seed, n_classes=None):
    """E: k_f(rho) = 1{rho >= theta_f} on an ORDERED register rho in {0..R-1}.
    theta_f in {0..R}: theta=0 always spells 1, theta=R always spells 0.
    `n_classes` restricts the thetas to g shared paradigms (inflection classes)."""
    rng = np.random.default_rng(seed)
    if n_classes is None:
        theta = rng.integers(1, R, size=v)            # 1..R-1: every feature inflects
    else:
        pool = rng.choice(np.arange(1, R), size=n_classes, replace=False)
        theta = pool[rng.integers(0, n_classes, size=v)]
    K = (np.arange(R)[None, :] >= theta[:, None]).astype(np.int64)
    return K, {"theta": theta.tolist(), "R": R, "n_classes": n_classes}


def fam_two_class(v, seed):
    """A_2class: k_f(r) = <w_{c(f)}, r>, with TWO shared response vectors w_0=(1,0), w_1=(0,1).
    Two spelling paradigms, four distinguishable contexts — the most shared 4-context rule
    a one-bit spelling admits (see the paradigm/context bound in [1])."""
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 2, size=v)
    W = np.array([[1, 0], [0, 1]])
    K = np.zeros((v, 4), dtype=np.int64)
    for c in range(4):
        K[:, c] = (W[cls] * ctx_bits(c, 2)[None, :]).sum(1) % 2
    return K, {"class": cls.tolist(), "W": W.tolist()}


def fam_agreement(v, seed):
    """B: k_f(f') = a_f XOR b_{f'} — context is a neighbouring level-1 feature."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=v)
    b = rng.integers(0, 2, size=v)
    return (a[:, None] ^ b[None, :]).astype(np.int64), {"a": a.tolist(), "b": b.tolist()}


# --------------------------------------------------------------------------- #
# hypothesis classes a renderer might carry.  Each is expressed as the SET OF ROWS it
# admits (row-separable classes) or as a predicate on the whole column block.
# --------------------------------------------------------------------------- #

def rows_table(R):
    return [tuple(bits) for bits in itertools.product((0, 1), repeat=R)]


def rows_affine(d):
    R = 2 ** d
    out = set()
    for a in (0, 1):
        for w in itertools.product((0, 1), repeat=d):
            out.add(tuple(a ^ int(np.dot(w, ctx_bits(c, d)) % 2) for c in range(R)))
    return sorted(out)


def rows_monotone(R):
    return [tuple(1 if c >= t else 0 for c in range(R)) for t in range(R + 1)]


def xorfree(mat):
    """Is `mat` (v, k) expressible as 1{alpha_f + gamma_c > 0} for reals — i.e. can a
    real-valued ADDITIVE renderer over (one-hot feature, one-hot context) express it?
    Equivalent to: no 2x2 submatrix equal to [[1,0],[0,1]] or [[0,1],[1,0]]."""
    v, k = mat.shape
    for i in range(v):
        for j in range(i + 1, v):
            di = mat[i] - mat[j]
            if (di > 0).any() and (di < 0).any():
                return False
    return True


def ladder_rowsep(K, admissible_rows, practiced, held):
    """Row-separable class: per feature, which admissible rows agree on the practiced
    contexts, and do they all agree at the held-out one?  Returns (frac determined,
    expected accuracy of a uniform-over-consistent predictor)."""
    v = K.shape[0]
    det, acc = 0.0, 0.0
    for f in range(v):
        obs = tuple(int(K[f, c]) for c in practiced)
        cons = [r for r in admissible_rows if tuple(r[c] for c in practiced) == obs]
        vals = {r[held] for r in cons}
        det += (len(vals) == 1)
        n_ok = sum(1 for r in cons if r[held] == int(K[f, held]))
        acc += n_ok / len(cons)
    return det / v, acc / v


def ladder_xorfree(K, practiced, held):
    """Not row-separable: enumerate every completion of the held-out column and keep the
    ones that keep the practiced+held block additive-real-expressible."""
    v = K.shape[0]
    P = K[:, list(practiced)]
    cons = []
    for comp in itertools.product((0, 1), repeat=v):
        mat = np.concatenate([P, np.array(comp, dtype=np.int64)[:, None]], axis=1)
        if xorfree(mat):
            cons.append(np.array(comp, dtype=np.int64))
    if not cons:
        return float("nan"), float("nan"), 0
    C = np.stack(cons)                                   # (n_cons, v)
    det = float((C.min(0) == C.max(0)).mean())
    acc = float((C == K[:, held][None, :]).mean())
    return det, acc, len(cons)


def ladder_shared_w(K, d, practiced, held):
    """A_shared: one w for all features, per-feature offset a_f."""
    v, R = K.shape
    cands = []
    for w in itertools.product((0, 1), repeat=d):
        g = np.array([int(np.dot(w, ctx_bits(c, d)) % 2) for c in range(R)])
        a = None
        ok = True
        for f in range(v):
            vals = {int(K[f, c]) ^ int(g[c]) for c in practiced}
            if len(vals) != 1:
                ok = False
                break
        if ok:
            a = np.array([int(K[f, practiced[0]]) ^ int(g[practiced[0]]) for f in range(v)])
            cands.append(a ^ g[held])
    if not cands:
        return float("nan"), float("nan"), 0
    C = np.stack(cands)
    return float((C.min(0) == C.max(0)).mean()), float((C == K[:, held][None, :]).mean()), len(cands)


def ladder_paradigm(K, R, g, practiced, held):
    """E_classed: the rows are drawn from g SHARED monotone paradigms (inflection classes);
    both the paradigms and each feature's class must be identified."""
    v = K.shape[0]
    mono = rows_monotone(R)
    dets, accs, n_c = np.zeros(v), np.zeros(v), 0
    poss = [set() for _ in range(v)]
    for sub in itertools.combinations(range(len(mono)), g):
        rows = [mono[i] for i in sub]
        ok = True
        for f in range(v):
            obs = tuple(int(K[f, c]) for c in practiced)
            if not any(tuple(r[c] for c in practiced) == obs for r in rows):
                ok = False
                break
        if not ok:
            continue
        n_c += 1
        for f in range(v):
            obs = tuple(int(K[f, c]) for c in practiced)
            for r in rows:
                if tuple(r[c] for c in practiced) == obs:
                    poss[f].add(r[held])
    for f in range(v):
        dets[f] = (len(poss[f]) == 1)
        accs[f] = (1.0 / len(poss[f])) if int(K[f, held]) in poss[f] else 0.0
    return float(dets.mean()), float(accs.mean()), n_c


# --------------------------------------------------------------------------- #
# rule-spelled sampling (local; the donors are untouched)
# --------------------------------------------------------------------------- #

def sample_spelled(rules, roots, K, ctx, s=S):
    """Sample one derivation per root, spelling every level-1 node by the rule:
    the bottom synonym is K[feature, ctx[i]] instead of a uniform draw.
    `ctx` is (B,) — one register per sequence.  Returns (leaves, level1_feats)."""
    rng = sample_spelled.rng
    L = len(rules)
    cur = roots[:, None]
    for ell in range(L - 1):
        w = cur.shape[1]
        ch = rng.integers(0, rules[ell].shape[1], size=(cur.shape[0], w))
        nxt = np.empty((cur.shape[0], w * s), dtype=np.int64)
        for j in range(w):
            nxt[:, j * s:(j + 1) * s] = rules[ell][cur[:, j], ch[:, j]]
        cur = nxt
    feats = cur                                            # (B, n_blocks) level-1 features
    k = K[feats, ctx[:, None]]                             # (B, n_blocks) synonym index
    leaves = rules[L - 1][feats, k]                        # (B, n_blocks, s)
    return leaves.reshape(feats.shape[0], -1), feats


sample_spelled.rng = np.random.default_rng(0)


def corrupt_spelled(leaves, rules, depth, v, m, s, level, node, K, ctx, rng):
    """`crystallize/units.corrupt_hier`, with the bottom rendered by the rule instead of a
    uniform draw (so spelling itself does not leak where the damage is).  Returns
    (damaged leaves, the level-1 features written into the damaged span)."""
    batch = leaves.shape[0]
    nodes = np.asarray([node], dtype=np.int64)
    levels = possible_sets(rules, leaves, s)
    poss = levels[level - 1][:, nodes, :]
    scores = rng.random((batch, 1, v))
    scores[poss] = -1.0
    feats = scores.argmax(-1)[:, :, None]
    for lv in range(level, 1, -1):
        table = rules[depth - lv]
        r = rng.integers(0, table.shape[1], size=feats.shape)
        feats = table[feats, r].reshape(batch, 1, -1)
    bottom = rules[depth - 1]
    k = K[feats, ctx[:, None, None]]
    tup = bottom[feats, k]                                  # (B, 1, span, s)
    span = feats.shape[-1]
    blk0 = nodes * span
    blocks = blk0[None, :, None] + np.arange(span)[None, None, :]
    blocks = np.broadcast_to(blocks, (batch, 1, span))
    pos = blocks[..., None] * s + np.arange(s)[None, None, None, :]
    out = leaves.copy()
    np.put_along_axis(out, pos.reshape(batch, -1), tup.reshape(batch, -1), axis=1)
    return out, feats.reshape(batch, span)


# --------------------------------------------------------------------------- #
# the d* ladder (`tall/`'s admissibility instrument)
# --------------------------------------------------------------------------- #

def dstar_ladder(v, m, depth=DEPTH, s=S, n=N_DSTAR, rule_seed=RULE_SEED, seed=17,
                 rules=None, ladder=None, K=None, R=None):
    """Oracle repair distance over the nested damage ladder, broken-only instances.
    `rules` overrides the draw (for the structured-bottom world); K/R make the clean and
    damaged renderings rule-spelled instead of uniform."""
    if rules is None:
        rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    if ladder is None:
        ladder = ([(l, int(0.75 * s ** (depth - l))) for l in range(1, depth)]
                  if depth != DEPTH else LADDER)
    rng = np.random.default_rng(seed)
    out = []
    for level, node in ladder:
        roots = rng.integers(0, v, size=n)
        if K is None:
            cur = roots[:, None]
            for ell in range(len(rules)):
                ch = rng.integers(0, rules[ell].shape[1], size=(n, cur.shape[1]))
                nxt = np.empty((n, cur.shape[1] * s), dtype=np.int64)
                for j in range(cur.shape[1]):
                    nxt[:, j * s:(j + 1) * s] = rules[ell][cur[:, j], ch[:, j]]
                cur = nxt
            clean = cur
            dmg = corrupt_hier(clean, rules, depth, v, m, s, level, [node], rng)
        else:
            ctx = rng.integers(0, R, size=n)
            sample_spelled.rng = rng
            clean, _ = sample_spelled(rules, roots, K, ctx, s)
            dmg, _ = corrupt_spelled(clean, rules, depth, v, m, s, level, node, K, ctx, rng)
        lv = possible_sets(rules, clean, s)
        nochoice = float(lv[level - 1][:, node, :].all(-1).mean())
        d = nearest_derivation_cost(rules, dmg, roots, s)
        br = d > 0
        out.append({"level": level, "node": node, "d_mean": float(d.mean()),
                    "d_sem_broken": (float(d[br].std(ddof=1) / math.sqrt(br.sum()))
                                     if br.sum() > 1 else 0.0),
                    "d_mean_broken": float(d[br].mean()) if br.any() else 0.0,
                    "frac_broken": float(br.mean()), "frac_no_offgrammar_choice": nochoice,
                    "d_max": int(d.max())})
    g = (out[-1]["d_mean_broken"] / out[0]["d_mean_broken"]
         if out[0]["d_mean_broken"] else float("nan"))
    return out, float(g)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    out = {"world": {"v": V, "s": S, "depth": DEPTH, "m": M, "rule_seed": RULE_SEED,
                     "param_seed": PARAM_SEED}}
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    codes = bottom_codes(rules)
    ib = build_inverse_maps(rules)[-1]

    print("=" * 92)
    print("PHASE 0 — INFLECTION: the rendering rule below the tables, sized offline")
    print("=" * 92)

    # ---- [0] the world and the fork invariant ---------------------------------------- #
    print("\n[0] the world at rule_seed=0: the bottom table, its collisions, and what a bottom")
    print("    rule can and cannot move")
    print(f"    v={V} s={S} m={M} depth={DEPTH}; bottom table rules[{DEPTH-1}] is (v, m, s) = "
          f"{tuple(rules[-1].shape)}")
    print(f"    {'feature':>8} {'synonym 0':>12} {'synonym 1':>12}   (leaf tuple, and its code)")
    for f in range(V):
        t0s = tuple(int(x) for x in rules[-1][f, 0])
        t1s = tuple(int(x) for x in rules[-1][f, 1])
        print(f"    {f:>8} {str(t0s)+' c'+str(int(codes[f,0])):>12} "
              f"{str(t1s)+' c'+str(int(codes[f,1])):>12}")
    coll = {}
    for f in range(V):
        for k in range(M):
            coll.setdefault(int(codes[f, k]), []).append((f, k))
    collisions = {c: p for c, p in coll.items() if len(p) > 1}
    print(f"    distinct codes {len(coll)} of v*m={V*M} produced, of v**s={V**S} possible; "
          f"{len(collisions)} collisions")
    for c, p in sorted(collisions.items()):
        print(f"      code {c:>3}: {p}  -> `build_inverse_maps` last-writer-wins keeps "
              f"f={int(ib[c])}")
    print("    NOTE both collisions are between the SAME synonym index of the two features"
          if all(len({k for _, k in p}) == 1 for p in collisions.values()) else "")
    out["bottom"] = {"codes": codes.tolist(), "n_distinct": len(coll),
                     "collisions": {str(c): p for c, p in collisions.items()}}

    sizes, tt = level_sizes(rules, DEPTH, S, V, M)
    rng0 = np.random.default_rng(123)
    rules_rb = [r.copy() for r in rules]
    rules_rb[-1] = rng0.integers(0, V, size=rules[-1].shape)      # randomise the BOTTOM only
    sizes_rb, tt_rb = level_sizes(rules_rb, DEPTH, S, V, M)
    same = all(flats_of(tt[l]) == flats_of(tt_rb[l]) for l in (2, 3, 4, 5))
    print("\n    the fork invariant, checked: `true_tables` reads rules[0..depth-2] only, so the")
    print("    flat keys at L2..L5 cannot move when the BOTTOM rule changes.")
    print(f"    {'level':>6} {'rows':>10} {'distinct |T_l|':>16} | {'rows (bottom randomised)':>26} "
          f"{'distinct':>10}")
    for l in (2, 3, 4, 5):
        print(f"    {l:>6} {sizes[l][0]:>10,} {sizes[l][1]:>16,} | {sizes_rb[l][0]:>26,} "
              f"{sizes_rb[l][1]:>10,}")
    print(f"    flat-key SETS identical at every level: {same}   "
          f"(so decision 2's rule leaves L2..L5 untouched by construction)")
    out["levels"] = {str(l): {"rows": sizes[l][0], "distinct": sizes[l][1]} for l in (2, 3, 4, 5)}
    out["fork_invariant_flatsets_identical"] = bool(same)

    # ---- [1] the families ------------------------------------------------------------- #
    print("\n[1] the rule families.  A rule is a table K (v x R): K[f,c] = the synonym index")
    print("    feature f takes in context c.  A plain (feature, context) table costs v*R bits.")
    fams = {}
    KA2, mA2 = fam_affine(V, 2, PARAM_SEED)
    KA2s, mA2s = fam_affine(V, 2, PARAM_SEED, shared_w=True)
    KA3, mA3 = fam_affine(V, 3, PARAM_SEED)
    KE4, mE4 = fam_threshold(V, 4, PARAM_SEED)
    KE4g, mE4g = fam_threshold(V, 4, PARAM_SEED, n_classes=2)
    KE8, mE8 = fam_threshold(V, 8, PARAM_SEED)
    K2c, m2c = fam_two_class(V, PARAM_SEED)
    KB, mB = fam_agreement(V, PARAM_SEED)
    fams["A_d2"] = dict(K=KA2, meta=mA2, R=4, label="GF(2)-affine register, d=2, per-feature w",
                        kind="affine", d=2, bits=3 * V, patterns=len(rows_affine(2)))
    fams["A_d2_sharedw"] = dict(K=KA2s, meta=mA2s, R=4, label="GF(2)-affine, d=2, w SHARED",
                                kind="affine_shared", d=2, bits=V + 2, patterns=2)
    fams["A_d3"] = dict(K=KA3, meta=mA3, R=8, label="GF(2)-affine register, d=3, per-feature w",
                        kind="affine", d=3, bits=4 * V, patterns=len(rows_affine(3)))
    fams["E_R4"] = dict(K=KE4, meta=mE4, R=4, label="ordered register, per-feature threshold, R=4",
                        kind="threshold", bits=V * math.log2(5), patterns=5)
    fams["E_R4_g2"] = dict(K=KE4g, meta=mE4g, R=4,
                           label="ordered register, 2 shared paradigms, R=4",
                           kind="threshold_g", g=2, bits=V * 1 + 2 * math.log2(5), patterns=2)
    fams["E_R8"] = dict(K=KE8, meta=mE8, R=8, label="ordered register, per-feature threshold, R=8",
                        kind="threshold", bits=V * math.log2(9), patterns=9)
    fams["A_2class"] = dict(K=K2c, meta=m2c, R=4, label="two crossing paradigms, one per bit",
                            kind="affine", d=2, bits=V + 4, patterns=2)
    fams["B_agree"] = dict(K=KB, meta=mB, R=V, label="agreement: k_f(f') = a_f XOR b_f'",
                           kind="agreement", bits=2 * V - 1, patterns=2)
    print(f"    {'family':<14} {'R':>4} {'R_eff':>6} {'free bits':>10} {'table bits':>11} "
          f"{'share':>7} {'paradigms':>10} {'additive-real?':>15}")
    for name, F in fams.items():
        K = F["K"]
        realised = len({tuple(int(x) for x in K[f]) for f in range(V)})
        reff = len({tuple(int(K[f, c]) for f in range(V)) for c in range(F["R"])})
        F["realised_rows"] = realised
        F["R_eff"] = reff
        F["xorfree"] = bool(xorfree(K))
        print(f"    {name:<14} {F['R']:>4} {reff:>6} {F['bits']:>10.1f} {V*F['R']:>11} "
              f"{F['bits']/(V*F['R']):>7.2f} {realised:>10} {str(F['xorfree']):>15}")
    print("    R_eff = distinct COLUMNS of K: two contexts with the same column are the same")
    print("    context for every observer, so R_eff and not R is the number of contexts the")
    print("    curriculum can hold out.  `paradigms` = distinct ROWS of K.")
    print("    'additive-real?' = is the rule expressible as 1{alpha_f + gamma_c > 0} — i.e. can a")
    print("    real-valued ADDITIVE renderer over (one-hot feature, context) represent it at all?")
    print("    Every GF(2) rule with two features that disagree in opposite directions is a 2x2")
    print("    XOR pattern and is NOT; a monotone (threshold) rule always is.")
    for name, F in fams.items():
        print(f"      {name:<14} K rows = " + " ".join("".join(str(int(x)) for x in F["K"][f])
                                                       for f in range(V)))
    print("\n    the paradigm / context bound at m=2, by exhaustion over all row sets of size g")
    print("    in {0,1}^4 (R=4 contexts): how many contexts can g spelling paradigms distinguish?")
    print(f"      {'paradigms g':>12} {'max R_eff (any rows)':>22} {'max R_eff (NESTED rows —':>26}")
    print(f"      {'':>12} {'':>22} {'additive-representable)':>26}")
    bound = {}
    allrows = [tuple(b) for b in itertools.product((0, 1), repeat=4)]
    for g in range(1, 5):
        best, best_n = 0, 0
        for sub in itertools.combinations(allrows, g):
            cols = len({tuple(r[c] for r in sub) for c in range(4)})
            best = max(best, cols)
            mat = np.array(sub, dtype=np.int64)
            if xorfree(mat):
                best_n = max(best_n, cols)
        bound[g] = (best, best_n)
        print(f"      {g:>12} {best:>22} {best_n:>26}")
    print("      -> four contexts need only TWO paradigms if they may CROSS (parity), and at")
    print("         least THREE if the renderer is to be additive in a scalar register.")
    out["paradigm_context_bound"] = {str(k): list(v) for k, v in bound.items()}
    out["families"] = {n: {"label": F["label"], "R": F["R"], "R_eff": F["R_eff"], "bits": F["bits"],
                           "table_bits": V * F["R"], "patterns": F["patterns"],
                           "realised_rows": F["realised_rows"], "xorfree": F["xorfree"],
                           "K": F["K"].tolist(), "params": F["meta"]} for n, F in fams.items()}

    # ---- [2] the identifiability ladder ------------------------------------------------ #
    print("\n[2] THE IDENTIFIABILITY LADDER — exact, by enumerating every completion of the")
    print("    held-out context that a hypothesis class admits.  `det` = fraction of features")
    print("    whose held-out spelling is PINNED; `acc` = expected accuracy of a renderer that")
    print("    picks uniformly among the completions its class still allows.  `--` = the class")
    print("    cannot even fit the practiced contexts (no completion is admissible).")
    print("    The renderer classes, from most to least generic:")
    print("      table            one-hot(feature) x one-hot(context)  — the plain lookup")
    print("      additive_1hot    1{alpha_f + gamma_c > 0}, context ONE-HOT — a linear renderer")
    print("      additive_scalar  1{alpha_f + beta*rho > 0}, context an ORDERED SCALAR (monotone")
    print("                       rows); the same linear renderer given the register as a number")
    print("      affine_gf2       a_f XOR <w_f, r> — parity over the register bits")
    print("      affine_sharedw   as affine_gf2 with one w for all features")
    print("      paradigm_g       rows drawn from g shared monotone paradigms")
    ladders = {}
    for name, F in fams.items():
        K, R = F["K"], F["R"]
        classes = {"table": ("rowsep", rows_table(R))}
        if F["kind"] in ("affine", "affine_shared"):
            classes["affine_gf2"] = ("rowsep", rows_affine(F["d"]))
            classes["affine_sharedw"] = ("sharedw", F["d"])
        elif F["kind"] == "threshold":
            classes["additive_scalar"] = ("rowsep", rows_monotone(R))
        elif F["kind"] == "threshold_g":
            classes["additive_scalar"] = ("rowsep", rows_monotone(R))
            classes["paradigm_g2"] = ("paradigm", (R, F["g"]))
        classes["additive_1hot"] = ("xorfree", None)
        combos_all = []
        for h in range(R):
            rest = [c for c in range(R) if c != h]
            for C in range(1, R):
                subs = list(itertools.combinations(rest, C))
                if len(subs) > 12:
                    subs = subs[::max(1, len(subs) // 12)][:12]
                for P in subs:
                    combos_all.append((C, list(P), h))
        rows, brk = [], {}
        for cname, (kind, arg) in classes.items():
            acc_by_C, det_by_C, feas_by_C = {}, {}, {}
            for C, P, h in combos_all:
                if kind == "rowsep":
                    d_, a_ = ladder_rowsep(K, arg, P, h)
                elif kind == "sharedw":
                    d_, a_, n_ = ladder_shared_w(K, arg, P, h)
                elif kind == "paradigm":
                    d_, a_, n_ = ladder_paradigm(K, arg[0], arg[1], P, h)
                else:
                    d_, a_, n_ = ladder_xorfree(K, P, h)
                feas_by_C.setdefault(C, []).append(not (isinstance(d_, float) and math.isnan(d_)))
                if not (isinstance(d_, float) and math.isnan(d_)):
                    det_by_C.setdefault(C, []).append(d_)
                    acc_by_C.setdefault(C, []).append(a_)
                    if F["kind"] in ("threshold", "threshold_g") and cname == "additive_scalar":
                        key = (C, bool(min(P) < h < max(P)))
                        brk.setdefault(key, []).append(a_)
            rows.append((cname, {C: (float(np.mean(det_by_C[C])) if C in det_by_C else float("nan"),
                                     float(np.mean(acc_by_C[C])) if C in acc_by_C else float("nan"),
                                     float(np.mean(feas_by_C[C])))
                                 for C in sorted(feas_by_C)}))
        ladders[name] = {c: {str(k): list(v) for k, v in d.items()} for c, d in rows}
        Cs = sorted(rows[0][1])
        print(f"\n    {name}  ({F['label']}; R={F['R']})")
        print("      " + f"{'renderer class':<17}" + "".join(f"{'C='+str(c):>15}" for c in Cs))
        for cname, d in rows:
            cells = []
            for c in Cs:
                if d[c][2] == 0.0:
                    cells.append(f"{'--':>15}")
                else:
                    cells.append(f"{d[c][0]:>6.2f}/{d[c][1]:>6.2f}" + ("*" if d[c][2] < 1 else " "))
            print("      " + f"{cname:<17}" + "".join(cells))
        if brk:
            print("      coverage split (additive_scalar, acc): is the held-out register INSIDE"
                  " the practiced range (interpolated) or outside it (extrapolated)?")
            for C in sorted({c for c, _ in brk}):
                a = brk.get((C, True)); b = brk.get((C, False))
                print(f"        C={C}: interpolated {np.mean(a):.2f} ({len(a)} cases)  "
                      f"| extrapolated {np.mean(b):.2f} ({len(b)} cases)"
                      if a and b else
                      f"        C={C}: interpolated {np.mean(a):.2f}" if a else
                      f"        C={C}: extrapolated {np.mean(b):.2f}")
            ladders[name]["bracket_split"] = {f"C{c}_{'in' if k else 'out'}": float(np.mean(v))
                                              for (c, k), v in brk.items()}
    print("\n    (each cell: det / acc; `*` = the class was infeasible for some practiced sets)")
    out["ladder"] = ladders

    # ---- [3] lexical invisibility of the held-out context ------------------------------ #
    print("\n[3] LEXICAL INVISIBILITY — does holding a context out remove any WORD from the")
    print("    practiced corpus?  `new spellings` = (feature, synonym) pairs first seen at the")
    print("    held-out context; `new codes` = leaf tuples first seen there (what the frozen")
    print("    reader would meet for the first time).  Averaged over held-out context and over")
    print("    practiced subsets of size C = R-1.")
    print(f"    {'family':<14} {'C=R-1':>7} {'new spellings /v':>17} {'new codes':>10} "
          f"{'worst-case held-out':>20}")
    lex = {}
    for name, F in fams.items():
        K, R = F["K"], F["R"]
        tot_sp, tot_cd, worst = [], [], 0
        for h in range(R):
            P = [c for c in range(R) if c != h]
            newsp = sum(1 for f in range(V) if int(K[f, h]) not in {int(K[f, c]) for c in P})
            seen = {int(codes[f, K[f, c]]) for f in range(V) for c in P}
            newcd = len({int(codes[f, K[f, h]]) for f in range(V)} - seen)
            tot_sp.append(newsp / V)
            tot_cd.append(newcd)
            worst = max(worst, newcd)
        lex[name] = {"new_spellings_frac": float(np.mean(tot_sp)),
                     "new_codes_mean": float(np.mean(tot_cd)), "new_codes_worst": int(worst)}
        print(f"    {name:<14} {R-1:>7} {np.mean(tot_sp):>17.3f} {np.mean(tot_cd):>10.2f} "
              f"{worst:>20}")
    # the same question one rung down the curriculum
    print(f"\n    the curriculum rungs (practiced C of R, one held out) — new codes at the held-out:")
    print(f"    {'family':<14}" + "".join(f"{'C='+str(c):>9}" for c in range(1, 8)))
    for name, F in fams.items():
        K, R = F["K"], F["R"]
        cells = []
        for C in range(1, 8):
            if C >= R:
                cells.append(None)
                continue
            vals = []
            for h in range(R):
                rest = [c for c in range(R) if c != h]
                for P in itertools.combinations(rest, C):
                    seen = {int(codes[f, K[f, c]]) for f in range(V) for c in P}
                    vals.append(len({int(codes[f, K[f, h]]) for f in range(V)} - seen))
            cells.append(float(np.mean(vals)))
        lex[name]["new_codes_by_C"] = cells
        print(f"    {name:<14}" + "".join("      -  " if x is None else f"{x:>9.2f}" for x in cells))
    print("    (a held-out context is lexically INVISIBLE where this is 0: the reader meets no")
    print("     new leaf tuple, so held-out transfer tests the RULE and not the vocabulary.)")
    print("\n    for the ordered register, new codes by WHICH context is held out (C = R-1):")
    for name in ("E_R4", "E_R8"):
        K, R = fams[name]["K"], fams[name]["R"]
        per = []
        for h in range(R):
            P = [c for c in range(R) if c != h]
            seen = {int(codes[f, K[f, c]]) for f in range(V) for c in P}
            per.append(len({int(codes[f, K[f, h]]) for f in range(V)} - seen))
        lex[name]["new_codes_by_heldout"] = per
        print(f"      {name:<8} " + " ".join(f"rho={h}:{per[h]}" for h in range(R)) +
              "   (the two ENDS of the scale are where a spelling can be unique)")
    out["lexical"] = lex

    # ---- [4] parse ambiguity under the rule -------------------------------------------- #
    print("\n[4] PARSE AMBIGUITY under the rule.  Unconditional block->feature sets come from the")
    print("    full legal bottom table (any synonym, as `possible_sets` reads it); rule-conditioned")
    print("    sets keep only features whose RULE spelling produces the observed tuple.")
    amb = {}
    uncond = {}
    for c in range(V ** S):
        fs = {f for f in range(V) for k in range(M) if int(codes[f, k]) == c}
        if fs:
            uncond[c] = fs

    def coin_leaves(n, seed=5):
        """the CURRENT substrate: an independent uniform synonym draw at every level-1 node."""
        rg = np.random.default_rng(seed)
        sample_spelled.rng = rg
        rts = rg.integers(0, V, size=n)
        _ = rg.integers(0, 4, size=n)                       # keep the tree draw aligned
        cur = rts[:, None]
        for ell in range(len(rules) - 1):
            ch = rg.integers(0, M, size=(n, cur.shape[1]))
            nxt = np.empty((n, cur.shape[1] * S), dtype=np.int64)
            for j in range(cur.shape[1]):
                nxt[:, j * S:(j + 1) * S] = rules[ell][cur[:, j], ch[:, j]]
            cur = nxt
        k = rg.integers(0, M, size=cur.shape)
        return rules[-1][cur, k].reshape(n, -1), cur

    print(f"    {'arm':<14} {'ambiguous blocks':>17} {'ctx reduces':>12} {'ctx resolves':>13} "
          f"{'root-set mean':>14} {'root ambiguous':>15}")
    lv_u, _ = coin_leaves(N_SIM)
    blk_u = (lv_u.reshape(N_SIM, -1, S) * (V ** np.arange(S))).sum(-1)
    base_amb = float(np.mean([len(uncond[int(c)]) > 1 for row in blk_u for c in row]))
    ps_u = possible_sets(rules, lv_u[:2000], S)
    rs_u = ps_u[-1][:, 0, :].sum(-1)
    print(f"    {'uniform coin':<14} {base_amb:>17.4f} {'-':>12} {'-':>13} "
          f"{float(rs_u.mean()):>14.4f} {float((rs_u > 1).mean()):>15.4f}")
    amb["uniform_coin"] = {"frac_block_ambiguous": base_amb,
                           "root_set_mean": float(rs_u.mean()),
                           "frac_root_ambiguous": float((rs_u > 1).mean())}
    for name, F in fams.items():
        K, R = F["K"], F["R"]
        rng = np.random.default_rng(5)
        sample_spelled.rng = rng
        roots = rng.integers(0, V, size=N_SIM)
        ctx = rng.integers(0, R, size=N_SIM)
        leaves, feats = sample_spelled(rules, roots, K, ctx)
        blk = (leaves.reshape(N_SIM, -1, S) * (V ** np.arange(S))).sum(-1)
        alive = {c: {cd: {f for f in range(V) if int(codes[f, K[f, c]]) == cd} for cd in uncond}
                 for c in range(R)}
        n_amb = n_res = n_red = 0
        tot = blk.size
        for i in range(N_SIM):
            cc = int(ctx[i])
            for cd in blk[i]:
                u = uncond[int(cd)]
                if len(u) > 1:
                    n_amb += 1
                    r_ = alive[cc][int(cd)]
                    if len(r_) < len(u):
                        n_red += 1
                    if len(r_) == 1:
                        n_res += 1
        ps = possible_sets(rules, leaves[:2000], S)
        rootsz = ps[-1][:, 0, :].sum(-1)
        amb[name] = {"blocks": int(tot), "frac_block_ambiguous": n_amb / tot,
                     "frac_context_reduces": n_red / tot, "frac_context_resolves": n_res / tot,
                     "root_set_mean": float(rootsz.mean()),
                     "frac_root_ambiguous": float((rootsz > 1).mean()),
                     "root_sets_equal_to_coin": bool(np.array_equal(
                         ps[-1][:, 0, :], possible_sets(rules, coin_leaves(2000)[0], S)[-1][:, 0, :]))}
        print(f"    {name:<14} {n_amb/tot:>17.4f} {n_red/tot:>12.4f} {n_res/tot:>13.4f} "
              f"{float(rootsz.mean()):>14.4f} {float((rootsz>1).mean()):>15.4f}")

    print("\n    where the bottom ambiguity dies: mean possible-set size per level (level 1 =")
    print("    blocks, level 6 = root), same 2,000 trees, three spellings of them:")
    print(f"      {'arm':<14}" + "".join(f"{'L'+str(l):>9}" for l in range(1, DEPTH + 1)))
    lvl_tab = {}
    trees_ref = None
    for name in ("uniform coin", "A_d2", "E_R4"):
        if name == "uniform coin":
            lv2, _ = coin_leaves(2000)
        else:
            K, R = fams[name]["K"], fams[name]["R"]
            rng = np.random.default_rng(5)
            sample_spelled.rng = rng
            rts = rng.integers(0, V, size=2000)
            cx = rng.integers(0, R, size=2000)
            lv2, _ = sample_spelled(rules, rts, K, cx)
        ps = possible_sets(rules, lv2, S)          # ps[0] = level 1 ... ps[-1] = root
        row = [float(ps[i].sum(-1).mean()) for i in range(len(ps))]
        lvl_tab[name] = row
        print(f"      {name:<14}" + "".join(f"{x:>9.4f}" for x in row))
    print("      the spellings differ at level 1 and are IDENTICAL from level 2 up: at")
    print("      `rule_seed=0` neither colliding pair survives its level-2 composition, so the")
    print("      context can sharpen a block read but is never NEEDED to parse a sequence.")
    amb["per_level_setsize"] = lvl_tab
    out["ambiguity"] = amb

    # ---- [5] the register's own readability -------------------------------------------- #
    print("\n[5] IS THE CONTEXT READABLE FROM THE OBSERVATION?  For a reader that knows the rule,")
    print("    each block's code rules out the contexts that could not have produced it.  Blocks")
    print("    are read left to right until one context remains.")
    readab = {}
    for name, F in fams.items():
        if F["kind"] == "agreement":
            continue
        K, R = F["K"], F["R"]
        prod = {c: {int(codes[f, K[f, c]]) for f in range(V)} for c in range(R)}
        rng = np.random.default_rng(9)
        sample_spelled.rng = rng
        roots = rng.integers(0, V, size=1500)
        ctx = rng.integers(0, R, size=1500)
        leaves, _ = sample_spelled(rules, roots, K, ctx)
        blk = (leaves.reshape(1500, -1, S) * (V ** np.arange(S))).sum(-1)
        need, left = [], []
        for i in range(1500):
            cand = set(range(R))
            n, hit = 0, -1
            for cd in blk[i]:
                cand &= {c for c in cand if int(cd) in prod[c]}
                n += 1
                if len(cand) == 1 and hit < 0:
                    hit = n
            need.append(hit)
            left.append(len(cand))
        need = np.array(need)
        left = np.array(left)
        pinned = need > 0
        readab[name] = {"frac_pinned": float(pinned.mean()),
                        "blocks_mean": float(need[pinned].mean()) if pinned.any() else None,
                        "frac_by_1": float((need == 1).mean()),
                        "frac_by_3": float(((need > 0) & (need <= 3)).mean()),
                        "candidates_left_mean": float(left.mean()), "R": R}
        print(f"    {name:<14} R={R} | pinned within the 32 blocks {pinned.mean():>6.3f} | mean "
              f"blocks {need[pinned].mean() if pinned.any() else float('nan'):>6.2f} | one block "
              f"is enough {float((need==1).mean()):>6.3f} | candidates left after 32 blocks "
              f"{left.mean():>5.2f} of {R}")
    out["readability"] = readab

    # ---- [6] the strict grader --------------------------------------------------------- #
    print("\n[6] THE STRICT GRADER.  A canon-writing executor (`canon = rules[depth-1][:,0,:]`,")
    print("    synonym 0 always) solves a rule-spelled instance and writes B = s^(level-1) blocks.")
    print("    A strict grader accepts only if every written block carries the synonym the rule")
    print("    called for.  Reported: the fraction of MEANING-successes it would reject.")
    strict = {}
    for name, F in fams.items():
        if F["kind"] == "agreement":
            continue
        K, R = F["K"], F["R"]
        rng = np.random.default_rng(21)
        sample_spelled.rng = rng
        roots = rng.integers(0, V, size=N_SIM)
        ctx = rng.integers(0, R, size=N_SIM)
        leaves, feats = sample_spelled(rules, roots, K, ctx)
        rows = []
        for level, node in LADDER:
            span = S ** (level - 1)
            b0 = node * span
            fb = feats[:, b0:b0 + span]                       # (N, span) features written
            ok = (K[fb, ctx[:, None]] == 0).all(1)
            q = float((K[fb, ctx[:, None]] == 0).mean())
            rows.append({"level": level, "blocks": int(span), "p_strict_accept": float(ok.mean()),
                         "per_block_q": q, "q_pow_B": q ** span, "half_pow_B": 0.5 ** span})
        strict[name] = rows
        print(f"\n    {name}  ({F['label']})")
        print(f"      {'damage level':>12} {'blocks B':>9} {'per-block q':>12} "
              f"{'P(strict accept)':>17} {'q^B (independent)':>18} {'2^-B':>9} {'rejected':>9}")
        for r in rows:
            print(f"      {r['level']:>12} {r['blocks']:>9} {r['per_block_q']:>12.3f} "
                  f"{r['p_strict_accept']:>17.4f} {r['q_pow_B']:>18.4f} {r['half_pow_B']:>9.4f} "
                  f"{1-r['p_strict_accept']:>9.4f}")
    print("\n    the register is shared by every block of an instance, so spelling errors are")
    print("    CORRELATED: P(strict accept) is not q^B.  For the ordered register the canon")
    print("    executor is exactly right at the bottom of the scale and wrong at the top:")
    for name in ("E_R4", "E_R8"):
        K, R = fams[name]["K"], fams[name]["R"]
        print(f"      {name}: P(K[f,rho]=0) by register rho = " +
              " ".join(f"{float((K[:, c] == 0).mean()):.2f}" for c in range(R)))
    out["strict_grader"] = strict

    # ---- [7] the d* ladder -------------------------------------------------------------- #
    print("\n[7] THE d* LADDER — `tall/`'s admissibility instrument, recomputed.  Oracle repair")
    print("    distance over the nested damage ladder, broken-only instances, CPU.")
    print("\n    (a) GATE I-1 — `tall/FILES.md` section 1 reproduced from this offline harness")
    print("        (its published d* means in brackets):")
    dst = {}
    rep = []
    gate_dev = 0.0
    for (dep, m_) in [(4, 2), (4, 4), (6, 2), (6, 3), (6, 4)]:
        rr, g = dstar_ladder(V, m_, depth=dep)
        rep.append({"depth": dep, "m": m_, "rungs": rr, "gradient": g})
        cells = " ".join(f"{x['d_mean_broken']:>6.2f}" for x in rr)
        ref = TALL_DSTAR[(dep, m_)]
        dev = max(abs(x["d_mean_broken"] - r) for x, r in zip(rr, ref))
        gate_dev = max(gate_dev, dev)
        print(f"      depth {dep}, v={V}, m={m_}: {cells:<36} gradient {g:>5.2f}x  "
              f"tall [{' '.join('%.2f' % r for r in ref)}]  max dev {dev:.2f}")
    dst["tall_replication"] = rep
    dst["gate_i1_max_dev"] = gate_dev
    print(f"      GATE I-1: {'PASS' if gate_dev < 0.25 else 'FAIL'} — largest deviation from "
          f"`tall`'s published d* is {gate_dev:.2f} tokens across all 21 rungs.  (`tall` ran on")
    print("      the substrate with its own damage-acceptance filter; this harness calls the same")
    print("      `corrupt_hier` and the same exact DP with no filter, and reports d* > 0 only.)")

    print("\n    (b) the widened worlds — where 'the register selects the synonym index' would")
    print("        have >= 4 contexts.  occupancy = m / v^(s-1).")
    print("        d* is BROKEN-ONLY (d* > 0), as `tall/` reports it; `frac broken` is the share")
    print("        of damaged instances that need any repair at all, and `no-choice` the share")
    print("        where the node's possible-set already covers every feature so `corrupt_hier`")
    print("        cannot pick an underivable one — the mechanism behind a collapsed ladder.")
    print(f"      {'(v,m)':>10} {'occupancy':>10} " + "".join(f"{'L'+str(l):>7}" for l in range(1, 6))
          + f" {'gradient':>9} {'frac broken (L1..L5)':>26} {'no-choice L5':>13}")
    wide = []
    for (v_, m_) in [(4, 2), (8, 2), (8, 4), (16, 2), (16, 4), (16, 8), (32, 4)]:
        rr, g = dstar_ladder(v_, m_)
        wide.append({"v": v_, "m": m_, "occupancy": m_ / v_ ** (S - 1), "rungs": rr, "gradient": g})
        print(f"      {'('+str(v_)+','+str(m_)+')':>10} {m_/v_**(S-1):>10.3f} " +
              "".join(f"{x['d_mean_broken']:>7.2f}" for x in rr) + f" {g:>9.2f}x  " +
              " ".join(f"{x['frac_broken']:>4.2f}" for x in rr) +
              f" {rr[-1]['frac_no_offgrammar_choice']:>13.3f}")
    dst["widened_worlds"] = wide

    print("\n    (c) the register families change the SAMPLING, not the tables — the DP is over")
    print("        the same rule tables, so this is a distributional check, not a new world:")
    same_world = []
    for nm in ("uniform_coin", "A_d2", "E_R4"):
        if nm == "uniform_coin":
            rr, g = dstar_ladder(V, M)
        else:
            rr, g = dstar_ladder(V, M, rules=rules, K=fams[nm]["K"], R=fams[nm]["R"])
        same_world.append({"arm": nm, "rungs": rr, "gradient": g})
        print(f"      {nm:<14} " + "".join(f"{x['d_mean_broken']:>7.2f}" for x in rr) +
              f" {g:>9.2f}x")
    dst["same_world_spellings"] = same_world

    print("\n    (d) the structured bottom (stem token, ending token; 4 endings shared across")
    print("        features).  Upper rules untouched; only rules[depth-1] is rewritten, and its")
    print("        m goes 2 -> 4:")
    bot_D = np.empty((V, 4, S), dtype=np.int64)
    for f in range(V):
        for c in range(4):
            bot_D[f, c] = (f, c)
    rules_D = [r.copy() for r in rules[:-1]] + [bot_D]
    K_id = np.tile(np.arange(4), (V, 1))
    rr, g = dstar_ladder(V, M, rules=rules_D, K=K_id, R=4)
    dst["structured_bottom"] = {"rungs": rr, "gradient": g}
    print(f"      stem+ending    " + "".join(f"{x['d_mean_broken']:>7.2f}" for x in rr) +
          f" {g:>9.2f}x")
    cD = bottom_codes(rules_D)
    print(f"      bottom codes distinct {len(set(cD.reshape(-1).tolist()))} of v*m'={V*4} "
          f"produced, of {V**S} possible -> parse ambiguity "
          f"{'zero' if len(set(cD.reshape(-1).tolist()))==V*4 else 'nonzero'}; the ending token")
    print("      IS the context, so one block names it exactly and a held-out context is a")
    print("      held-out TOKEN.")
    szD, ttD = level_sizes(rules_D, DEPTH, S, V, M)
    print(f"      flat keys L2..L5 under the structured bottom: " +
          " ".join(f"L{l} {szD[l][1]:,}" for l in (2, 3, 4, 5)) +
          f"   identical to the original: {all(flats_of(ttD[l])==flats_of(tt[l]) for l in (2,3,4,5))}")
    dst["structured_bottom"]["codes_distinct"] = len(set(cD.reshape(-1).tolist()))
    dst["structured_bottom"]["flatsets_identical"] = bool(
        all(flats_of(ttD[l]) == flats_of(tt[l]) for l in (2, 3, 4, 5)))
    out["dstar"] = dst

    # ---- [8] what widening costs above level 1 ------------------------------------------ #
    print("\n[8] WHAT WIDENING COSTS ABOVE LEVEL 1.  `true_tables` rows (the learner's own")
    print("    vocabulary; entries(l) = v*e(l), e(l) = m*e(l-1)^s) and the distinct flat keys.")
    print(f"      {'(v,m)':>10} " + "".join(f"{'L'+str(l)+' rows':>22}" for l in range(2, 6)) +
          "   distinct |T_l| (measured where enumerable)")
    wcost = []
    for (v_, m_) in [(8, 2), (16, 2), (16, 4), (16, 8), (32, 4)]:
        rws = [macro_rows(v_, m_, l) for l in range(2, 6)]
        dis = {}
        if max(rws[:3]) <= 500_000:
            rl = generate_rules_distinct(v_, S, DEPTH, m_, seed=RULE_SEED)
            top = 4 if rws[2] <= 500_000 else 3
            szz, _ = level_sizes(rl, DEPTH, S, v_, m_, max_level=top)
            dis = {l: szz[l][1] for l in range(2, top + 1)}
        wcost.append({"v": v_, "m": m_, "rows": rws, "distinct": {str(k): v for k, v in dis.items()}})
        print(f"      {'('+str(v_)+','+str(m_)+')':>10} " + "".join(f"{r:>22,}" for r in rws) +
              "   " + " ".join(f"L{l} {dis[l]:,}" for l in sorted(dis)))
    print("      (the arc earns L2 and L3 and mines toward L4; `tall/FILES.md` section 1 read the")
    print("       same law as 'at m=4 nothing above level 3 is representable'.)")
    out["widening_cost"] = wcost
    print(f"    the d* rungs carry a standard error of "
          f"{np.mean([r['d_sem_broken'] for w in wide for r in w['rungs']]):.3f} on average "
          f"(n={N_DSTAR} per rung), so the gradients above are separated by many SEs.")

    # ---- [9] agreement: are its contexts free? ------------------------------------------ #
    print("\n[9] AGREEMENT — which (feature, context) cells the grammar actually realises, and")
    print("    whether a held-out cell is a held-out RULE or a held-out PRODUCTION.")
    sib, par = set(), set()
    lay2 = rules[DEPTH - 2]                                   # level-2 -> level-1
    for p in range(V):
        for r in range(lay2.shape[1]):
            c0, c1 = int(lay2[p, r, 0]), int(lay2[p, r, 1])
            sib.add((c0, c1)); sib.add((c1, c0))
            par.add((c0, p)); par.add((c1, p))
    _, fe = coin_leaves(20000, seed=3)
    adj = set()
    for row in fe:
        for j in range(row.shape[0] - 1):
            adj.add((int(row[j + 1]), int(row[j])))
    print(f"    realised (feature, context) cells of the v*v = {V*V} possible:")
    print(f"      sibling's level-1 feature      {len(sib):>4}   (a cell is a level-2 rule half — "
          f"holding one out holds out a PRODUCTION)")
    print(f"      parent's level-2 feature       {len(par):>4}   (same: a cell is a level-2 rule)")
    print(f"      left-neighbour block's feature {len(adj):>4}   (observed in 20,000 sequences)")
    # additive-GF(2) identifiability over a held-out column vs held-out cells
    KBt = fams["B_agree"]["K"]
    colnew = []
    for h in range(V):
        P = [c for c in range(V) if c != h]
        d_, a_ = ladder_rowsep(KBt, rows_table(V), P, h)
        colnew.append(a_)
    # cell-level: an additive GF(2) model pins cell (f, c) iff f and c are connected in the
    # bipartite graph of practiced cells
    def cell_ladder(cells, frac, reps=200, seed=4):
        rg = np.random.default_rng(seed)
        cells = sorted(cells)
        det = []
        for _ in range(reps):
            keep = [cells[i] for i in range(len(cells)) if rg.random() < frac]
            uf = {}

            def find(x):
                while uf.setdefault(x, x) != x:
                    uf[x] = uf[uf[x]]
                    x = uf[x]
                return x
            for (f_, c_) in keep:
                a_, b_ = find(("f", f_)), find(("c", c_))
                if a_ != b_:
                    uf[a_] = b_
            held = [x for x in cells if x not in set(keep)]
            if held:
                det.append(np.mean([find(("f", f_)) == find(("c", c_)) for f_, c_ in held]))
        return float(np.mean(det)) if det else float("nan")

    print(f"\n    held-out CELLS (additive GF(2) over a bipartite graph of practiced cells):")
    print(f"      {'context':<12}" + "".join(f"{'p='+str(x):>9}" for x in (0.2, 0.4, 0.6, 0.8)))
    cellres = {}
    for cname, cs in (("sibling", sib), ("parent", par), ("neighbour", adj)):
        vals = [cell_ladder(cs, x) for x in (0.2, 0.4, 0.6, 0.8)]
        cellres[cname] = vals
        print(f"      {cname:<12}" + "".join(f"{x:>9.2f}" for x in vals))
    print("      (fraction of held-out cells whose spelling is PINNED by the practiced ones)")
    print("    held-out CONTEXT (a whole column): b_context is unconstrained, so a renderer that")
    print(f"      knows the family is still at chance ({float(np.mean(colnew)):.2f}) — additivity")
    print("      buys generalisation to unseen (feature, context) CELLS, never to an unseen context.")
    out["agreement"] = {"cells_sibling": len(sib), "cells_parent": len(par),
                        "cells_neighbour": len(adj), "heldout_column_acc": float(np.mean(colnew)),
                        "cell_ladder": cellres}

    # ---- [10] what this file cannot decide ---------------------------------------------- #
    print("\n[10] WHAT IS NOT DECIDABLE OFFLINE (left for Q1 on GPU).")
    print("    (i)  Does the frozen generator's block head parse rule-spelled blocks, and does it")
    print("         parse the HELD-OUT context?  Sections [3] and [4] settle the vocabulary half")
    print("         — under the register families no leaf tuple is new at a held-out context, and")
    print("         the block codes are the same 14 the reader was pretrained on.  What is not")
    print("         settled is the DISTRIBUTION: the reader was pretrained by masked infilling on")
    print("         uniform-coin data, and a rule makes the per-feature spelling frequencies")
    print("         context-dependent.  The measurement that decides it: `parse_features` block")
    print("         accuracy against `exact_features`, per context, on rule-spelled clean")
    print("         sequences, for a reader pretrained (a) on uniform-coin data and (b) on")
    print("         rule-spelled data with one context held out.")
    print("    (ii) Whether a renderer FITTED from the learner's own solved productions reaches")
    print("         the ceilings in [2].  Those ceilings assume noise-free observations of every")
    print("         practiced (feature, context) cell; a learner sees only the cells its own")
    print("         commits visited, at its own frequencies, with grading noise.")
    print("    (iii) Whether the executor can infer the register inside a DAMAGED instance.  The")
    print("         counts in [5] read clean blocks with the rule known; in an instance the")
    print("         damaged span is being rewritten and the rule is what is being learned.")
    print("    (iv) Whether the reader and the renderer share parameters (decision 4) — that is a")
    print("         fit, not an arithmetic.")

    with open(os.path.join(HERE, "phase0.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.join(HERE, 'phase0.json')}   ({time.time()-t0:.0f}s)")
    return out


if __name__ == "__main__":
    main()
