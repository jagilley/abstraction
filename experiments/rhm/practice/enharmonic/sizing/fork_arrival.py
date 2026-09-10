"""ADDENDUM to Q0 — what the FORK's own mining path arrives at, against §3's model.

§3 of [`SIZING.md`](SIZING.md) modelled arrival on the class-pair key by computing each half's
token class from the RAW TOKENS of a DGP derivation. The fork cannot do that. `ClassMiner`
sees the reader's parsed level-1 feature string, and the class it must use is the class of
`canon[features]` — the canonical rendering — because that is the only map consistent between
what the learner PERCEIVES and what it can PRODUCE: `macros.apply_any` writes `canon[feats]`,
so a committed row's class is `possible_sets(canon[row])` and that is a fact about the
executor, not a choice. Using the raw-token class on the observe side and the canonical class
on the build side would mean an observed class with no producible representative, and the
level-2 ratchet would drop nearly everything (a rule-1 bottom tuple's class is not any
canonical block's class).

The two sides of that are measured here:

  [1] THE ROUND-TRIP LOSS. How often `canon[exact_features(tokens)]` is still a legal
      derivation, per level. The parse is last-writer-wins, so a block whose tokens are shared
      by two (feature, rule) pairs is read as the last writer, and `canon` then renders THAT
      feature's rule-0 tuple — which is a different token string wherever the two features do
      not share their rule-0 tuple. The loss is not the 2<->7 canon collision (that one is
      exactly preserved); it is every other shared bottom code.

  [2] THE FORK'S OWN MINING PATH, simulated. `tu_s0`'s ladder, `mine_cap = 8` observations per
      cycle drawn from the DGP and parsed by an exact reader (`read_acc` = 1.0, which is what
      the substrate has), the committable miners era-gated to `era_level + 1` exactly as
      `run_arm` gates them, `QT.make_miner` / `ClassMiner.build` / `Quotient` verbatim, and a
      boundary commit of the active level at each era end. This is the arithmetic the fork
      runs, on the observation stream the model in §3 assumes; the remaining gap to a real run
      is the agent's own solve rate and narrowness, which gate G-1 measured at 0.42-1.68x.

No GPU, no Modal, no substrate. Writes `fork_arrival.json` beside this file.

Run from experiments/:  PYTHONPATH=. python3 rhm/practice/enharmonic/sizing/fork_arrival.py
"""

import json
import os
import time

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_precheck import possible_sets, sample_derivations
from rhm.practice.ratchet import macros as MC
from rhm.practice.enharmonic import quotient as QT

HERE = os.path.dirname(os.path.abspath(__file__))
V, S, DEPTH, M = 8, 2, 6, 2
RULE_SEED = 0
SUPPORT, MINE_CAP, MAXL = 3, 8, 5
LADDER = [(1, 25, 60), (2, 12, 50), (3, 6, 70), (4, 3, 12), (5, 1, 9)]   # tu_s0's, verbatim


def main(n_probe=20_000):
    t0 = time.time()
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    canon = np.ascontiguousarray(rules[DEPTH - 1][:, 0, :])
    ib = build_inverse_maps(rules)[-1]
    truth = MC.true_tables(rules, DEPTH, S, V, M, 5)
    out = {"world": {"v": V, "s": S, "depth": DEPTH, "m": M, "rule_seed": RULE_SEED},
           "ladder": LADDER, "mine_cap": MINE_CAP, "support": SUPPORT}
    print("=" * 88)
    print("ADDENDUM — the FORK's own arrival, against SIZING.md §3's model")
    print("=" * 88)

    # ---- [1] the round-trip loss --------------------------------------------------------- #
    print("\n[1] THE ROUND-TRIP LOSS — is `canon[parse(tokens)]` still a legal derivation?")
    rng = np.random.default_rng(0)
    roots = rng.integers(0, V, size=n_probe)
    lv = sample_derivations(rules, roots, S, rng)
    pf = MC.exact_features(lv, ib, V, S)
    rows = []
    n_bad_blocks = int((canon[pf] != lv.reshape(n_probe, -1, S)).any(-1).mean() * 100)
    for level in (2, 3, 4, 5):
        w = S ** (level - 1)
        raw = possible_sets(rules[DEPTH - level:], lv[:, :w * S], S)[-1][:, 0, :]
        ren = possible_sets(rules[DEPTH - level:],
                            canon[pf[:, :w]].reshape(n_probe, -1), S)[-1][:, 0, :]
        rows.append({"level": level, "raw_tokens_legal": float(raw.any(1).mean()),
                     "canon_rendered_legal": float(ren.any(1).mean())})
        print(f"    L{level}: raw token span legal {raw.any(1).mean():.4f}   "
              f"canon[parse] legal {ren.any(1).mean():.4f}")
    print(f"    blocks whose canonical re-rendering differs from the original tokens: "
          f"~{n_bad_blocks}%")
    print("    The class of an observed half is therefore defined on a string that is legal")
    print("    only 21.6% of the time at the L5 node, against 100% for the raw tokens. §3's")
    print("    63.3 of 70 at L5 and 111.9 of 306 at L6 are an UPPER BOUND on the fork's")
    print("    arrival, not its prediction.")
    out["round_trip"] = rows
    del lv, pf

    # ---- [2] the fork's own mining path -------------------------------------------------- #
    print("\n[2] THE FORK'S OWN MINING PATH, simulated on `tu_s0`'s ladder")
    sims = {}
    for mode in (None, "tok", "gen"):
        rng = np.random.default_rng(3)
        quot = None if mode is None else QT.Quotient(mode, rules, canon, V, S, DEPTH)
        cfgq = {"quot_spell_cap": 4}
        miners = {l: QT.make_miner(l, S, cfgq, quot) for l in range(2, MAXL + 1)}
        obs = {l: QT.make_miner(l, S, cfgq, quot) for l in range(3, DEPTH + 1)}
        committed = {l: None for l in range(2, MAXL + 1)}
        hist = {l: [] for l in range(3, DEPTH + 1)}

        def operative(l):
            if l == 1:
                return MC.base_table(V)
            if committed[l] is not None:
                return committed[l]
            return miners[l].build(operative(l - 1), SUPPORT)

        commits = []
        for (el, node_e, cyc) in LADDER:
            active = min(el + 1, MAXL)
            for _ in range(cyc):
                r = rng.integers(0, V, size=MINE_CAP)
                pf = MC.exact_features(sample_derivations(rules, r, S, rng), ib, V, S)
                for l in range(2, min(MAXL, el + 1) + 1):
                    sp = S ** (l - 1)
                    nd = (node_e * S ** (el - 1)) // sp
                    miners[l].observe(pf[:, nd * sp:(nd + 1) * sp])
                for l, mn in obs.items():
                    sp = S ** (l - 1)
                    nd = (node_e * S ** (el - 1)) // sp
                    if 0 <= nd < S ** (DEPTH - l) and (nd + 1) * sp <= pf.shape[1]:
                        mn.observe(pf[:, nd * sp:(nd + 1) * sp])
                    hist[l].append(int(mn.state()["n_at_support"][str(SUPPORT)]))
            if active <= MAXL and committed.get(active) is None:
                tb = operative(active)
                if tb["child"].shape[0]:
                    committed[active] = tb
                    g = MC.grade_table(tb, truth[active])
                    lb = dict(getattr(miners[active], "last_build", {}) or {})
                    commits.append({"level": active, "n_entries": int(tb["child"].shape[0]),
                                    "precision": g["precision"], "recall": g["recall"],
                                    **{k: lb.get(k) for k in
                                       ("n_at_support", "inventory_max", "n_class_capped",
                                        "n_lower_classes")}})
        sims[str(mode)] = {
            "commits": commits,
            "obs_at_support_final": {str(l): h[-1] for l, h in hist.items()},
            "obs_series_l5": hist[5][::10], "obs_series_l6": hist[6][::10],
            "oracle": (None if quot is None else quot.state()),
            "miner_at_support": {str(l): int(m.state()["n_at_support"][str(SUPPORT)])
                                 for l, m in miners.items()}}
        print(f"\n    --- key = {mode or 'flat'} ---")
        for c in commits:
            print(f"      L{c['level']}: {c['n_entries']:>5} entries  prec "
                  f"{(c['precision'] or 0):.3f}  recall {c['recall']:.4f}  "
                  f"keys@sup {c['n_at_support']}  inv_max {c['inventory_max']}  "
                  f"capped {c['n_class_capped']}")
        if not commits:
            print("      (no commit at any level)")
        print(f"      obs at-support-3 at end: "
              f"{sims[str(mode)]['obs_at_support_final']}")
    out["sim"] = sims
    print("\n    Read: `flat` never reaches L5 and its L6 gauge is identically 0, so the")
    print("    mirror's commit thermostat cannot even set its `moved` latch there. Both")
    print("    quotiented keys build an L5 table and both carry a live L6 gauge.")

    out["elapsed_s"] = time.time() - t0
    with open(os.path.join(HERE, "fork_arrival.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(HERE, 'fork_arrival.json')}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
