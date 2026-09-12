"""phase0_q2_lexicon — Q2.0 for `embouchure`: how much homophony the world tolerates.

Offline, CPU, no Modal, no substrate. The SPEC's Q2 (revised, 2026-09-11) replaces `bottom`
ALONE with a constructed lexicon carrying a controlled number of shared forms — two
(feature, synonym) pairs on one leaf tuple — leaving every level above untouched, so mining
dynamics stay `tutti`'s and only the lexicon moves. This lane sizes that construction before a
GPU is bought:

  (1) THE CANDIDATE PAIRS. Which merges make the world's block genuinely ambiguous UNDER THE
      RULE — both owners admissible at the same register — and at how many registers, split
      practised / held out. A merge that is never ambiguous under the rule changes the surface
      and nothing else, and is not what Q2 is about.
  (2) THE LADDER. For H = 1, 2, 3, ... merged pairs: is every instance still parseable to the
      root, what happens to the level-1 and root possible-set sizes, to the root-ambiguous
      fraction, to `d*` at every era's damage cell, and to the on-grammar rate.
  (3) LAST-WRITER-WINS. What `build_inverse_maps` does at each homophone — which owner the
      reader's own table keeps, and therefore which meaning the arc's exact parse silently
      picks.
  (4) THE WRITER'S SIDE. At how many (code, register) cells the learner can still RECALL what
      it wrote (exactly one owner admissible) against the cells where it cannot — the
      tip-of-the-tongue cells, which are a property of the lexicon and not of the learner.

Facts only. Interpretation is the orchestrator's.

NOT COVERED HERE, and it needs a GPU: what the frozen neural reader does at the homophonous
cells at production `reader_steps`. That is a measurement on a trained reader, i.e. one
`build_shared` (~610 s), and the coordinator's instruction is no GPU for Q2 until this ladder
and the two Q1 reductions are read. `readback_flip_check`-style machinery for it is already in
`embouchure.py`; the call is one line in `preflight`.

Run (from experiments/):
    PYTHONPATH=. python3 rhm/practice/embouchure/phase0_q2_lexicon.py
"""

import itertools
import json
import os

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, possible_sets
from rhm.practice.crystallize.units import on_grammar_rate
from rhm.practice.embouchure import embouchure as E

HERE = os.path.dirname(os.path.abspath(__file__))

V, S, DEPTH, M, N_CTX = 8, 2, 6, 2, 8
RULE_SEED = 6                      # the collision-free draw every clean `inflection` tag used
PRACTICED = [0, 3, 7]
ERAS = [("L1n25", 1, 25), ("L2n12", 2, 12), ("L3n6", 3, 6)]
N_POOL = 512


def code_of(tup):
    return int(tup[0] + V * tup[1])


def lexicon(rules, K):
    """The bottom map as a lexicon: form -> the (meaning, word) pairs that spell it, and which
    of them the RULE admits at each register."""
    bot = rules[-1]
    owners = {}
    for f in range(V):
        for k in range(M):
            owners.setdefault(code_of(bot[f, k]), []).append((f, k))
    adm = {}
    for c_, own in owners.items():
        adm[c_] = [[f for f, k in own if K[f, r] == k] for r in range(N_CTX)]
    return owners, adm


def candidate_pairs(rules, K, theta):
    """Every merge that is legal and what it would cost.

    LEGAL: the two (feature, synonym) pairs must belong to DIFFERENT features —
    `generate_rules_distinct` guarantees a feature's m tuples are distinct and the grader,
    the miner and `k_of` all rely on it.

    AMBIGUOUS UNDER THE RULE at register r iff the rule admits BOTH owners there, i.e.
    `K[f1, r] == k1 and K[f2, r] == k2`. For the ordered-register family that count is closed
    form: `min(t1, t2)` when both words are synonym 0, `8 - max(t1, t2)` when both are synonym
    1, and `max(0, t1 - t2)` for (k1, k2) = (0, 1).
    """
    out = []
    for (f1, k1), (f2, k2) in itertools.combinations(
            [(f, k) for f in range(V) for k in range(M)], 2):
        if f1 == f2:
            continue
        amb = [r for r in range(N_CTX) if K[f1, r] == k1 and K[f2, r] == k2]
        closed = (min(theta[f1], theta[f2]) if (k1 == 0 and k2 == 0) else
                  N_CTX - max(theta[f1], theta[f2]) if (k1 == 1 and k2 == 1) else
                  max(0, theta[f1] - theta[f2]) if (k1 == 0 and k2 == 1) else
                  max(0, theta[f2] - theta[f1]))
        out.append({"pair": [[f1, k1], [f2, k2]],
                    "theta": [int(theta[f1]), int(theta[f2])],
                    "amb_registers": amb,
                    "n_amb": len(amb),
                    "n_amb_practiced": len([r for r in amb if r in PRACTICED]),
                    "n_amb_heldout": len([r for r in amb if r not in PRACTICED]),
                    "closed_form_agrees": bool(len(amb) == closed)})
    return out


def merged_rules(rules, pairs):
    """`bottom` with each pair's second word RESPELLED onto the first's form. Levels 0..L-2 are
    the same objects, so `true_tables`, every flat key at L2-L5 and every level-size fact above
    L1 are untouched by construction (the sizing lane's gate I-2)."""
    bot = rules[-1].copy()
    for (f1, k1), (f2, k2) in pairs:
        bot[f2, k2] = bot[f1, k1]
    return list(rules[:-1]) + [bot]


def world_stats(rules_h, rule, seed=4242):
    """Is the world still a world: parseable to the root, on-grammar, and what the repair
    distance does at every era's damage cell."""
    roots, leaves, ctx = E._sample_pool_r(rules_h, N_POOL, S, seed, rule=rule,
                                          n_ctx=N_CTX, practiced=None)
    rp, lp, _ = E._sample_pool_r(rules_h, N_POOL, S, seed + 1, rule=rule,
                                 n_ctx=N_CTX, practiced=PRACTICED)
    inv = build_inverse_maps(rules_h)
    lev = possible_sets(rules_h, leaves, S)
    l1, root = lev[0], lev[-1]
    hit = root[np.arange(root.shape[0]), 0, roots]
    lvp = possible_sets(rules_h, lp, S)
    out = {"n": int(N_POOL),
           # the same block ambiguity restricted to the PRACTISED registers — the only ones
           # the priced beam ever draws, so this is the number the learner actually meets.
           "l1_ambiguous_practiced": float((lvp[0].sum(-1) > 1).mean()),
           "root_ambiguous_practiced": float((lvp[-1].sum(-1) > 1).mean()),
           "root_reachable": float(hit.mean()),
           "root_set_mean": float(root.sum(-1).mean()),
           "root_ambiguous": float((root.sum(-1) > 1).mean()),
           "l1_set_mean": float(l1.sum(-1).mean()),
           "l1_ambiguous_blocks": float((l1.sum(-1) > 1).mean()),
           "on_grammar": float(on_grammar_rate(leaves, inv[-1], V, S)),
           "dstar": {}}
    for name, level, node in ERAS:
        dmg = E.corrupt_hier_r(leaves.copy(), rules_h, DEPTH, V, M, S, level, [node],
                               np.random.default_rng(seed + 7), rule=rule, ctx=ctx)
        d = nearest_derivation_cost(rules_h, dmg, roots, S)
        out["dstar"][name] = {"mean": float(d.mean()), "zero_frac": float((d == 0).mean()),
                              "on_grammar": float(on_grammar_rate(dmg, inv[-1], V, S))}
    return out


def writer_side(rules_h, K):
    """Can the learner RECALL what it wrote? It emitted `bottom[f, k]`; the form identifies
    (f, k) only where the form has one owner, and where it has two the register may still
    separate them — `adm[code][r]` with a single entry is a recallable cell.

    A cell with two admissible owners is the tip-of-the-tongue cell: the learner has an intent
    and cannot retrieve which word it used, and neither can its reader. Counted, not argued."""
    owners, adm = lexicon(rules_h, K)
    n_cells = n_uniq = n_amb = n_dead = 0
    by_reg = {r: {"unique": 0, "ambiguous": 0} for r in range(N_CTX)}
    for c_, own in owners.items():
        for r in range(N_CTX):
            a = adm[c_][r]
            if not a:
                n_dead += 1
                continue
            n_cells += 1
            if len(a) == 1:
                n_uniq += 1
                by_reg[r]["unique"] += 1
            else:
                n_amb += 1
                by_reg[r]["ambiguous"] += 1
    return {"n_forms": len(owners), "n_forms_shared": sum(1 for o in owners.values() if len(o) > 1),
            "n_live_cells": n_cells, "n_recallable": n_uniq, "n_ambiguous": n_amb,
            "n_unadmitted": n_dead,
            "recallable_frac": (n_uniq / max(n_cells, 1)),
            "by_register": {str(r): by_reg[r] for r in range(N_CTX)}}


def last_writer(rules_h, pairs):
    """What `build_inverse_maps` keeps at each homophone — the owner the arc's exact parse
    silently picks, and therefore the meaning every instrument that goes through
    `inverse_maps[-1]` will report."""
    inv = build_inverse_maps(rules_h)[-1]
    rows = []
    for (f1, k1), (f2, k2) in pairs:
        c_ = code_of(rules_h[-1][f1, k1])
        rows.append({"code": int(c_), "owners": [[f1, k1], [f2, k2]],
                     "last_writer_keeps": int(inv[c_])})
    return rows


def main():
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    rule = E.make_rule("E_R8", V, M, N_CTX)
    K = np.asarray(rule.K)
    theta = [int(np.argmax(K[f] > 0)) if K[f].any() else N_CTX for f in range(V)]
    out = {"world": {"rule_seed": RULE_SEED, "theta": theta, "practiced": PRACTICED}}

    print("=" * 78)
    print("Q2.0 (1) — THE CANDIDATE MERGES: which shared form is ambiguous UNDER THE RULE")
    print("=" * 78)
    cands = candidate_pairs(rules, K, theta)
    out["candidates"] = cands
    print(f"  theta = {theta}; practised {PRACTICED}; "
          f"{len(cands)} legal merges of {V * M} (feature, synonym) pairs")
    print(f"  closed form agrees with the enumeration on all {len(cands)}: "
          f"{all(c['closed_form_agrees'] for c in cands)}")
    bad = [c for c in cands if c["n_amb"] == 0]
    print(f"  merges that are NEVER ambiguous under the rule (surface-only, not what Q2 asks "
          f"for): {len(bad)} of {len(cands)}")
    top = sorted(cands, key=lambda c: (-c["n_amb_practiced"], -c["n_amb"]))[:12]
    print(f"\n  the twelve richest, by ambiguous PRACTISED registers:")
    print(f"    {'pair':>22s} {'theta':>9s} {'n_amb':>5s} {'prac':>4s} {'held':>4s}  registers")
    for c in top:
        (f1, k1), (f2, k2) = c["pair"]
        print(f"    (f{f1},k{k1}) = (f{f2},k{k2})".rjust(26)
              + f" {str(c['theta']):>9s} {c['n_amb']:5d} {c['n_amb_practiced']:4d} "
                f"{c['n_amb_heldout']:4d}  {c['amb_registers']}")

    # the ladder's merges: greedily take the richest DISJOINT pairs (each form owned by exactly
    # two meanings), so H merges leave 16 - H distinct forms and no chain of three.
    chosen, used = [], set()
    for c in sorted(cands, key=lambda c: (-c["n_amb_practiced"], -c["n_amb"])):
        (f1, k1), (f2, k2) = c["pair"]
        if (f1, k1) in used or (f2, k2) in used:
            continue
        chosen.append(c)
        used.update({(f1, k1), (f2, k2)})
    out["ladder_order"] = chosen
    print(f"\n  the disjoint greedy order used for the ladder ({len(chosen)} available):")
    for i, c in enumerate(chosen, 1):
        (f1, k1), (f2, k2) = c["pair"]
        print(f"    H={i:2d}  (f{f1},k{k1})=(f{f2},k{k2})  n_amb {c['n_amb']} "
              f"(prac {c['n_amb_practiced']}, held {c['n_amb_heldout']}) at {c['amb_registers']}")

    print()
    print("=" * 78)
    print("Q2.0 (2)+(3)+(4) — THE LADDER")
    print("=" * 78)
    base = world_stats(rules, rule)
    out["ladder"] = [{"H": 0, "world": base, "writer": writer_side(rules, K),
                      "last_writer": [], "pairs": []}]
    print(f"  {'H':>2s} {'forms':>5s} {'shared':>6s} {'root_ok':>7s} {'root_set':>8s} "
          f"{'root_amb':>8s} {'L1_set':>7s} {'L1_amb':>7s} {'L1ambP':>7s} {'on_gram':>7s} "
          f"{'recall':>7s} {'d*L1':>6s} {'d*L2':>6s} {'d*L3':>6s}")

    def row(H, w, wr):
        print(f"  {H:2d} {wr['n_forms']:5d} {wr['n_forms_shared']:6d} "
              f"{w['root_reachable']:7.4f} {w['root_set_mean']:8.4f} "
              f"{w['root_ambiguous']:8.4f} {w['l1_set_mean']:7.4f} "
              f"{w['l1_ambiguous_blocks']:7.4f} {w['l1_ambiguous_practiced']:7.4f} "
              f"{w['on_grammar']:7.4f} "
              f"{wr['recallable_frac']:7.4f} "
              + " ".join(f"{w['dstar'][n]['mean']:6.3f}" for n, _, _ in ERAS))

    row(0, base, out["ladder"][0]["writer"])
    for H in range(1, len(chosen) + 1):
        pairs = [tuple(map(tuple, c["pair"])) for c in chosen[:H]]
        rh = merged_rules(rules, pairs)
        w = world_stats(rh, rule)
        wr = writer_side(rh, K)
        lw = last_writer(rh, pairs)
        out["ladder"].append({"H": H, "world": w, "writer": wr, "last_writer": lw,
                              "pairs": [list(map(list, p)) for p in pairs]})
        row(H, w, wr)

    print("\n  LAST-WRITER-WINS at each homophone (the owner the arc's exact parse keeps):")
    for r in out["ladder"][-1]["last_writer"]:
        (f1, k1), (f2, k2) = r["owners"]
        print(f"    code {r['code']:3d}: (f{f1},k{k1}) and (f{f2},k{k2})  ->  keeps "
              f"f={r['last_writer_keeps']}")

    print("\n  THE WRITER'S SIDE by register, at the densest rung "
          f"(H={out['ladder'][-1]['H']}):")
    wr = out["ladder"][-1]["writer"]
    print(f"    live (form, register) cells {wr['n_live_cells']}, recallable "
          f"{wr['n_recallable']} ({wr['recallable_frac']:.4f}), ambiguous "
          f"{wr['n_ambiguous']}")
    print(f"    {'rho':>3s} {'practised':>9s} {'recallable':>10s} {'ambiguous':>9s}")
    for r in range(N_CTX):
        b = wr["by_register"][str(r)]
        print(f"    {r:3d} {str(r in PRACTICED):>9s} {b['unique']:10d} {b['ambiguous']:9d}")

    p = os.path.join(HERE, "phase0_q2.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n[q2.0] wrote {p}")


if __name__ == "__main__":
    main()
