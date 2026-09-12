"""[en_s3] Why a proposal found no within-tol pair — offline, exact, no GPU.

`en_s3`'s ledger arm entered the merge block 15 times and found no alias group on 10 of them,
while its own class count was still well above the token alphabet (10-11 classes at L2 against
9 token classes and 7 demand groups; 26-31 at L3 against 11 and 9). Three different things
produce that same "no group" line and the run's record cannot separate them:

  (a) THE ALIASES WERE ALREADY MERGED — the classes still outnumber the token alphabet, but no
      two SURVIVING classes share a token class, so there is nothing left to merge.
  (b) THE PROBE NEVER LOOKED — `merge_use_frac` keeps only the rows carrying the top 90% of
      beam use, one representative per class, so a pair that exists among the classes is never
      scored if either side is outside that set.
  (c) THE PROBE LOOKED AND REFUSED — the pair was scored and its transfer loss sat above tol,
      which at this node is the subset band, or the two rows are structurally indistinguishable
      and land at 0.

Everything needed to separate them is reconstructible with no substrate, because the probe is
`fourwall.entry_profile` — a function of the DGP, the instances and the rows, and of nothing
the model learned:

  * the OPERATIVE ROWS at a proposal: at L2 the committed book is the at-support keys at its
    commit cycle (`ClassMiner`'s L1 classes are the features themselves, so its keys ARE flat
    rows); at L3 and above the book is `ClassMiner.build`'s cross product over the level below,
    which is exact here because `n_class_capped` is 0 on every build in this tag (the spelling
    cap never binds). Both are asserted against the run's own `n_rows`.
  * the CLASS MAP at that cycle: the singleton map plus every merge the arm took before it,
    each logged with its members.
  * the PROBED SET: the same use-share ordering the block applies, replayed from the arm's own
    slot record, asserted against the run's own `n_probed_rows`.
  * the PROBE ITSELF: `context_instances` at the same seed and `transfer_profile` verbatim.

Usage (from experiments/):
    python3 rhm/practice/enharmonic/alias_audit.py --tag en_s3
"""

import argparse
import itertools
import json
import os

import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.practice.ratchet import macros as MC
from rhm.practice.enharmonic import quotient as QT
from rhm.practice.enharmonic import merge as MG
from rhm.practice.tutti.tutti import context_instances

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
V, S, DEPTH, M = 8, 2, 6, 2
LAD = {1: 25, 2: 12, 3: 6, 4: 3, 5: 1}
# the node's own collapses, `sizing/SIZING.md` 5(a): what a forced-transfer probe CANNOT
# resolve at that mining node however many instances it is given.
COLL = {2: [frozenset({0, 3}), frozenset({2, 7})], 3: [frozenset({1, 6})],
        4: [frozenset({2, 7})]}
DEAD = {2: [], 3: [frozenset({1})], 4: [frozenset({7})]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="en_s3")
    # [en_s8] no default arm list: a tag's arms are whatever it ran. The old default
    # named `en_s2b`'s three and silently printed "not fetched locally" for two of them
    # on every later tag, while leaving that tag's own composed arm out of the file.
    ap.add_argument("--arms", default="",
                    help="comma-separated; default is every arm directory in the tag")
    a = ap.parse_args()
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=0)
    canon = np.ascontiguousarray(rules[DEPTH - 1][:, 0, :])

    def tok(rows, level):
        if not rows:
            return []
        P = QT.token_class_sets(rules, np.array(rows, np.int64), level, canon,
                                V, S, DEPTH)[:, 0, :]
        return [frozenset(np.nonzero(x)[0].tolist()) for x in P]

    def demand(c, level):
        if not c or any(c == d for d in DEAD.get(level, [])):
            return "DEAD/JUNK"
        for g in COLL.get(level, []):
            if c <= g and (c & g):
                return tuple(sorted(g))
        return tuple(sorted(c))

    print("=" * 108)
    print(f"ALIAS AUDIT — {a.tag}: what was there to merge, and what the probe saw of it")
    print("=" * 108)
    print("  `exist`  pairs of SURVIVING learner classes whose representatives share a token")
    print("           class — the merges that are structurally available at that proposal.")
    print("  `scored` how many of those the probe actually scored (both sides in the probed")
    print("           set, both repairing something). `<=tol` / `>tol` split them by loss.")
    print("  `ex(tok)` counts pairs sharing a TOKEN class; `ex(dmd)` pairs sharing a DEMAND")
    print("  cell, which is the coarser set the probe can actually resolve as one and the set")
    print("  a loss of 0 corresponds to. `scored`/`<=tol`/`>tol` are over `ex(dmd)`;")
    print("  `all<=tol` is every scored pair within tol, whatever its type.")
    print("  `ex(dmd) > 0` with `scored == 0` is (b): the use filter never showed the pair to")
    print("  the probe. `ex(dmd) == 0` is (a). `scored > 0` with `<=tol == 0` is (c).")
    print("")
    print("  AN OPEN-INVENTORY ARM IS APPROXIMATE HERE, AND NOT A LITTLE. This rebuild")
    print("  assumes the operative book of an adopted level IS the book its commit froze")
    print("  -- which is exactly what open_inventory abolishes: there the commit freezes")
    print("  the KEY and the book is the live build, moving every cycle. So for an open")
    print("  arm the rows/cls/probed/scored columns reconstruct a book the arm did not")
    print("  run, and every row is flagged [MISMATCH]. The committed-book coverage lines")
    print("  above each table are exact in all cases. The arm is printed rather than")
    print("  skipped so its structural columns (token classes, demand cells) are on the")
    print("  record.")

    _arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not _arms:
        _root = os.path.join(FIG, a.tag)
        _arms = sorted(d for d in os.listdir(_root)
                       if os.path.isfile(os.path.join(_root, d, "results.json")))
    for arm in _arms:
        p = os.path.join(FIG, a.tag, arm, "results.json")
        if not os.path.isfile(p):
            print(f"\n{arm}: not fetched locally — skipped")
            continue
        r = json.load(open(p))
        cfg = r["config"]
        mcfg = r.get("merge_cfg") or {}
        tol = float(mcfg.get("merge_tol", 0.20))
        cyc = r["log"]["cycle"]
        took = [e for e in r["merge_events"]
                if e.get("kind") == "merge" and e.get("taken")]
        props = [e for e in r["merge_events"] if e.get("kind") == "merge_proposal"]
        commits = {int(e["level"]): int(e["cycle"]) for e in r["events"]
                   if e["kind"] == "commit"}

        def find_of(level, upto):
            """The arm's own class map at `level` just before cycle `upto`."""
            rep = {}

            def find(x):
                while rep.get(x, x) != x:
                    x = rep[x]
                return x
            for e in took:
                if e["level"] != level or e["cycle"] >= upto:
                    continue
                mem = sorted({find(tuple(int(z) for z in x)) for x in e["members"]})
                keep = mem[0]
                for x in mem[1:]:
                    rep[x] = keep
            return find

        def keys_at(i, lv):
            st = (r["log"]["miner"][i] or {}).get(str(lv)) or {}
            return [tuple(tuple(int(z) for z in h) for h in k)
                    for k in (st.get("keys_at_support") or [])]

        def quot_at(upto):
            """The arm's own map at every level, just before cycle `upto`, replayed through
            the same `merge_group` the op used."""
            q = MG.LearnedQuotient()
            for e in took:
                if e["cycle"] >= upto:
                    continue
                q.merge_group(int(e["level"]), [tuple(int(z) for z in x)
                                                for x in e["members"]])
            return q

        def table_at(level, upto, q):
            """The operative BOOK at `level` just before cycle `upto`, built by the run's own
            `ClassMiner.build` so the ROW ORDER is the run's too — the use vector the probe
            filters on is indexed by it."""
            low = MC.base_table(V) if level == 2 else table_at(level - 1, upto, q)
            c = commits.get(level)
            i = cyc.index(c if (c is not None and c <= upto) else upto)
            mn = QT.ClassMiner(level, S, q, spell_cap=int(cfg.get("quot_spell_cap", 4)))
            mn.counts = {k: int(cfg["mine_support"]) for k in keys_at(i, level)}
            return mn.build(low, int(cfg["mine_support"]))

        # THE RECONSTRUCTION'S PREMISE, stated per arm: the cross-product is exact only where
        # the spelling cap never binds, because the ranking it would apply needs per-class
        # spelling counts that the compact log does not carry. `n_class_capped` says whether it
        # bound; where it did, the rows below are an approximation and every row is flagged.
        capb = max([int((q["build"].get(str(lv)) or {}).get("n_class_capped") or 0)
                    for q in r["log"]["quot"] if q for lv in (2, 3, 4, 5)] or [0])
        print(f"\n{'=' * 104}\n{arm}   tol={tol}  use_frac={mcfg.get('merge_use_frac')}  "
              f"n_probe={mcfg.get('merge_n_probe')}  max_rows={cfg.get('merge_max_rows')}  "
              f"max n_class_capped over all builds = {capb}"
              + ("   (0 -> the cross product is exact and every row below is asserted against "
                 "the run's own n_rows / n_probed_rows / n_pairs)" if capb == 0 else
                 "   (>0 -> THE SPELLING CAP BOUND: the rebuilt book is an approximation and "
                 "the mismatches below are that, not a finding)"))
        # [en_s4] THE COMMITTED BOOKS, in the same coordinates, so an endogenous arm's
        # coverage at its commit is comparable with `given_cat_tok`'s "3 of 13".
        clos_all = QT.class_closure(rules, canon, V, S, DEPTH, 5)
        for lv in sorted(commits):
            q0 = quot_at(commits[lv] + 1)
            tb = table_at(lv, commits[lv] + 1, q0)
            tbr = [tuple(int(z) for z in row) for row in tb["flat"]]
            tc0 = [c for c in tok(tbr, lv) if c]
            dm0 = {demand(c, lv) for c in tok(tbr, lv)}
            print(f"    committed L{lv}@c{commits[lv]}: {len(tbr)} rows, "
                  f"{len({q0.id_of(r0, lv) for r0 in tbr})} classes, "
                  f"{len(set(tc0))} of {clos_all[lv]['n_classes']} token classes, "
                  f"{len(dm0)} demand cells")
        print(f"  {'cycle':>6} {'L':>2} {'rows':>5} {'cls':>4} {'tok':>4} {'dmd':>4} "
              f"{'probed':>7} {'pairs':>6} {'ex(tok)':>8} {'ex(dmd)':>8} {'scored':>7} "
              f"{'<=tol':>6} {'>tol':>5} {'all<=tol':>9} "
              f"{'loss of the scored existing pairs':>36} {'why':>26}")
        for pr in props:
            ml, cy = int(pr["level"]), int(pr["cycle"])
            i = cyc.index(cy)
            q = quot_at(cy)
            tbl = table_at(ml, cy, q)
            rows = [tuple(int(z) for z in row) for row in tbl["flat"]]
            f = find_of(ml, cy)
            cls_of = {}
            for row in rows:
                cls_of.setdefault(f(row), []).append(row)
            reps = sorted(cls_of)
            ok_rows = (len(rows) == pr.get("n_rows")
                       and len(reps) == pr.get("n_classes_before"))
            tcs = tok(reps, ml)
            dm = {demand(c, ml) for c in tcs}
            dcs = [demand(c, ml) for c in tcs]
            exist = [(x, y) for x in range(len(reps)) for y in range(x + 1, len(reps))
                     if tcs[x] and tcs[x] == tcs[y]]
            # the DEMAND version: pairs the probe can actually resolve as one. `SIZING.md` 5(a)
            # measured the node's own collapses, so this is the coarser (and larger) set, and
            # it is the one a loss of 0 corresponds to.
            exist_d = [(x, y) for x in range(len(reps)) for y in range(x + 1, len(reps))
                       if dcs[x] != "DEAD/JUNK" and dcs[x] == dcs[y]]
            # THE PROBED SET, by the block's own rule: use share from the slot record, top
            # `use_frac` of the mass, then one representative per class, capped.
            use = np.zeros(len(rows), float)
            for q in r["log"]["slot"][:i + 1]:
                for nd, vec in ((q.get("beam") or {}).get(str(ml)) or {}).items():
                    if len(vec) == len(rows):
                        use += np.asarray(vec, float)
            if use.sum() > 0:
                use = use / use.sum()
            keep = list(range(len(rows)))
            if use.sum() > 0 and mcfg.get("merge_use_frac"):
                acc, keep = 0.0, []
                for i_ in list(np.argsort(-use)):
                    keep.append(int(i_))
                    acc += use[i_]
                    if acc >= float(mcfg["merge_use_frac"]):
                        break
                keep = sorted(set(keep)) if len(keep) >= 2 else list(range(len(rows)))
            seen, sub = set(), []
            for i_ in sorted(keep, key=lambda z: (-use[z], z)):
                c = f(rows[i_])
                if c in seen:
                    continue
                seen.add(c)
                sub.append(i_)
                if len(sub) >= int(cfg.get("merge_max_rows", 64)):
                    break
            sub_rows = [rows[i_] for i_ in sorted(sub)] if len(sub) >= 2 else \
                [rows[i_] for i_ in keep[:2]]
            ok_probe = len(sub_rows) == pr.get("n_probed_rows")
            # the probe itself, at the proposal's own seed and node
            node = (LAD[int(pr["era"])] * S ** (int(pr["era"]) - 1)) // S ** (ml - 1)
            mr, mx = context_instances(
                rules, {"name": f"L{ml}n{LAD[int(pr['era'])]}", "level": ml, "nodes": [node]},
                int(mcfg.get("merge_n_probe", 256)), S, DEPTH, V, M,
                seed=int(cfg["seed"]) + 810_000 + 1000 * cy)
            P, _g = MG.transfer_profile(rules, mx, mr, sub_rows, node, ml, S, canon)
            pairs, live = MG.pair_losses(P)
            ok_pairs = len(pairs) == pr.get("n_pairs")
            idx = {tuple(row): k for k, row in enumerate(sub_rows)}
            look = {(q["i"], q["j"]): q["loss"] for q in pairs}

            def scored_of(ex):
                out = []
                for x, y in ex:
                    ix, iy = idx.get(reps[x]), idx.get(reps[y])
                    if ix is None or iy is None:
                        continue
                    lo = look.get((min(ix, iy), max(ix, iy)))
                    if lo is not None:
                        out.append(lo)
                return out
            sc_t, sc = scored_of(exist), scored_of(exist_d)
            under = sum(1 for z in sc if z <= tol)
            all_under = sum(1 for q in pairs if q["loss"] <= tol)
            dist = ("—" if not sc else
                    f"min {min(sc):.3f} med {float(np.median(sc)):.3f} max {max(sc):.3f}")
            why = ("(a) nothing left to merge" if not exist_d else
                   "(b) probe never saw them" if not sc else
                   "(c) scored, all above tol" if not under else
                   f"{under} within tol")
            flag = "" if (ok_rows and ok_probe and ok_pairs) else \
                f"  [MISMATCH rows={ok_rows} probed={ok_probe} pairs={ok_pairs}]"
            print(f"  c{cy:>5} {ml:>2} {len(rows):>5} {len(reps):>4} "
                  f"{len({c for c in tcs if c}):>4} {len(dm):>4} {len(sub_rows):>7} "
                  f"{len(pairs):>6} {len(exist):>8} {len(exist_d):>8} {len(sc):>7} "
                  f"{under:>6} {len(sc) - under:>5} {all_under:>9} "
                  f"{dist:>36} {why:>26}{flag}")


if __name__ == "__main__":
    main()
