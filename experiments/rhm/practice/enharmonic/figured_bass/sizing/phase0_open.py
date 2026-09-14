"""[figured_bass] Q0, offline: what the book would have held had the commit frozen only the KEY.

The Q1 runs froze three things at once. `enharmonic.py`'s `operative(ell)` returns
`committed[ell]` the moment a level commits, so from that cycle on every build above it looks
up in the table that was frozen. This lane replays the banked `en_s0` / `en_s1` logs and asks
the counterfactual the node's SPEC poses: had the commit frozen only the PARTITION and the
executor's adoption, leaving the inventory to keep filling from the stream, what would the book
have held at every cycle after the commit, and what could the level above have built over it?

WHAT IS AND IS NOT COUNTERFACTUAL HERE. The open bit changes `operative()` only. It does not
change what the miners OBSERVE, so every arm's per-cycle `keys_at_support` — the object this
replay is driven by — is taken verbatim from the log. It WOULD change what the executor
executes (the DP maxes over a bigger inventory), hence the solve rate, hence the stream. So
every "open" column below is an ALL-ELSE-EQUAL replay on the banked observation stream, not a
prediction of a run. Same contract as `tutti/sizing`'s `buildable()` and `../../sizing/
fork_arrival.py`.

THE THREE REGIMES.
  `frozen`   the run.  `operative(ell) = committed[ell]`.
  `open4`    the L4 commit adopts the level but leaves its inventory live; L2's commit still
             freezes (so this isolates the L4 bit the SPEC is about).
  `openAll`  every commit adopts only; no inventory is ever frozen.
L3 is LIVE IN ALL THREE, because no arm in `en_s0`/`en_s1` ever committed L3 — the `tutti`
precedent the SPEC cites is already the regime these runs were in at L3.

HOW THE REPLAY WORKS. `quotient.py`'s `ClassMiner.build` and `macros.py`'s `Miner.build` are
re-implemented here (not imported) so a build can be driven from a logged key list instead of
from a live `counts` dict. Gate F-1 asserts the re-implementation reproduces every logged
`last_build` field on every cycle where the log carries a genuine build.

THE ONE APPROXIMATION, AND EXACTLY WHAT IT COSTS. `ClassMiner.build`'s `pick()` ranks a class's
lower rows by the miner's own `spell` counts before the `spell_cap` truncation, and those
counts are not logged. Three consequences, all measured below rather than assumed:
  * ROW COUNTS are exact under any pick — the count is `min(|class|, cap)` either way (F-1).
  * Under the 'tok' key CLASS COVERAGE is exact: a token class id is a function of the class
    pair by the closure property `quotient.py` documents, so which spellings are picked cannot
    move it (F-2 checks this against a cap-free run as well as against the reversed pick).
  * ROW IDENTITY is NOT reproduced, and under the 'min' key class coverage is not either (a
    min-label class holds rows of several token classes). F-3 reports the row-identity gap
    against the logged `true_mask`; the 'min' numbers are given as pick=lo / pick=hi /
    cap-free, the last a strict upper bound.

Usage (from experiments/):
    python3 rhm/practice/enharmonic/figured_bass/sizing/phase0_open.py
"""

import argparse
import collections
import gzip
import json
import os

import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.practice.ratchet import macros as MC
from rhm.practice.enharmonic import quotient as QT

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "figures")
V, S, DEPTH, M = 8, 2, 6, 2
SUPPORT = 3
SPELL_CAP = 4
MAXL = 5
REGIMES = (("frozen", set()), ("open4", {4}), ("openAll", {2, 3, 4, 5}))


# --------------------------------------------------------------------------- #
# the world
# --------------------------------------------------------------------------- #

def world():
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=0)
    canon = np.ascontiguousarray(rules[DEPTH - 1][:, 0, :])
    return rules, canon


def class_step(rules, c1, c2, ell):
    layer = rules[DEPTH - ell]
    return frozenset(f for f in range(V) for r in range(M)
                     if int(layer[f, r, 0]) in c1 and int(layer[f, r, 1]) in c2)


def alphabets(rules, canon):
    """The token-class alphabet AND the legal class PAIRS per level, as frozensets.
    `quotient.class_closure` returns only the counts; §2 needs the pairs themselves."""
    bottom = rules[DEPTH - 1]

    def blk(t):
        return frozenset(f for f in range(V) for r in range(bottom.shape[1])
                         if tuple(int(z) for z in bottom[f, r]) == t)
    writable = sorted({tuple(int(z) for z in canon[f]) for f in range(V)})
    cur = sorted({blk(t) for t in writable}, key=lambda c: sorted(c))
    alpha, pairs = {1: list(cur)}, {}
    for ell in range(2, 7):
        pr = {}
        for c1 in cur:
            for c2 in cur:
                got = class_step(rules, c1, c2, ell)
                if got:
                    pr[(c1, c2)] = got
        pairs[ell] = pr
        cur = sorted(set(pr.values()), key=lambda c: sorted(c))
        alpha[ell] = list(cur)
    return alpha, pairs


def mask_of(c):
    return int(sum(1 << f for f in c))


# --------------------------------------------------------------------------- #
# the two builds, replayed from a logged key list
# --------------------------------------------------------------------------- #

def class_build(keys, lower_rows, quot, level, spell_cap=SPELL_CAP, pick="lo"):
    """`quotient.ClassMiner.build`, driven by a logged `keys_at_support` list.

    `keys` are already in the miner's own `sorted()` order; `lower_rows` is the operative lower
    table's `flat` as a list of tuples. `pick` in {lo, hi, free}: index order / reversed /
    no cap. Returns (rows, stats) with `stats` in `last_build`'s exact schema."""
    want = (S ** (level - 1)) // S
    lower_flat = [r for r in lower_rows if len(r) == want]
    lut = {}
    for i, row in enumerate(lower_flat):
        lut[row] = i                                   # last writer wins, as the donor's
    by_class = collections.defaultdict(list)
    for row, i in lut.items():
        c = quot.id_of(row, level - 1)
        if c is not None:
            by_class[c].append(i)
    for c in by_class:
        by_class[c].sort()

    def pick_ix(c):
        ix = by_class.get(c, [])
        if pick == "free" or len(ix) <= spell_cap:
            return ix, False
        return (ix[:spell_cap] if pick == "lo" else ix[-spell_cap:]), True

    rows, n_at_sup, n_kept, inv, capped = [], 0, 0, [], 0
    for key in keys:
        n_at_sup += 1
        picks = []
        for i in range(S):
            ix, cap = pick_ix(key[i])
            if not ix:
                picks = None
                break
            capped += int(cap)
            picks.append(ix)
        if picks is None:
            continue
        n_kept += 1
        combos = [[]]
        for ix in picks:
            combos = [pre + [j] for pre in combos for j in ix]
        inv.append(len(combos))
        rows.extend(tuple(x for j in cmb for x in lower_flat[j]) for cmb in combos)
    stats = {"n_at_support": int(n_at_sup), "n_keys_built": int(n_kept),
             "n_entries": int(len(rows)),
             "inventory_max": int(max(inv) if inv else 0),
             "inventory_mean": (float(np.mean(inv)) if inv else 0.0),
             "n_class_capped": int(capped),
             "n_lower_classes": int(len(by_class)),
             "n_lower_rows": int(len(lower_flat))}
    return rows, stats


def flat_build(keys, lower_rows, level):
    """`macros.Miner.build`, driven by a logged `keys_at_support` list. A flat arm's built row
    IS its key, so the ratchet is the only thing to apply."""
    span = S ** (level - 1)
    half = span // S
    lut = {row: i for i, row in enumerate(lower_rows)}
    rows = [k for k in keys if len(k) == span
            and all(k[i * half:(i + 1) * half] in lut for i in range(S))]
    return rows, {"n_at_support": len(keys), "n_keys_built": len(rows),
                  "n_entries": len(rows), "n_lower_rows": len(lower_rows)}


# --------------------------------------------------------------------------- #
# the log
# --------------------------------------------------------------------------- #

def load(tag, arm):
    p = os.path.join(FIG, tag, arm, "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def load_entry(tag, arm):
    p = os.path.join(FIG, tag, arm, "entry.json.gz")
    return json.load(gzip.open(p)) if os.path.isfile(p) else None


def keys_at(res, i, level):
    st = (res["log"]["miner"][i] or {}).get(str(level)) or {}
    return [tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])
            if isinstance(k, list)]


def commits(res):
    return {int(e["level"]): int(e["cycle"]) for e in res["events"]
            if e.get("kind") == "commit"}


def era_bounds(res):
    cyc, era = res["log"]["cycle"], res["log"]["era"]
    return [(era[i], cyc[i]) for i in range(len(cyc)) if i == 0 or era[i] != era[i - 1]]


# --------------------------------------------------------------------------- #
# the replay
# --------------------------------------------------------------------------- #

def replay(res, quot, open_levels, pick="lo"):
    """Per cycle, the operative book at every level under a stated freeze regime."""
    cyc = res["log"]["cycle"]
    cm = commits(res)
    is_flat = quot is None
    base = [(f,) for f in range(V)]
    frozen_book, out = {}, []
    for i, c in enumerate(cyc):
        books, stats = {1: base}, {}
        for ell in range(2, MAXL + 1):
            cc = cm.get(ell)
            ks = keys_at(res, i, ell)
            if cc is not None and c >= cc and ell not in open_levels:
                if ell not in frozen_book:
                    frozen_book[ell] = (flat_build(ks, books[ell - 1], ell)[0] if is_flat
                                        else class_build(ks, books[ell - 1], quot, ell,
                                                         pick=pick)[0])
                books[ell] = frozen_book[ell]
                stats[ell] = {"frozen": True, "n_entries": len(books[ell])}
            else:
                rows, st = (flat_build(ks, books[ell - 1], ell) if is_flat
                            else class_build(ks, books[ell - 1], quot, ell, pick=pick))
                books[ell], st["frozen"] = rows, False
                stats[ell] = st
        out.append({"cycle": c, "books": books, "stats": stats})
    return out


def book_classes(rows, level, rules, canon, mode):
    """The classes a book covers, in the arm's own key coordinate (what `ClassMiner.build`'s
    `by_class` holds) AND in token coordinates (what an L5 lookup is denominated in)."""
    if not rows:
        return set(), set()
    P = QT.token_class_sets(rules, np.array(rows, np.int64), level, canon,
                            V, S, DEPTH)[:, 0, :]
    tok = {frozenset(np.nonzero(r)[0].tolist()) for r in P}
    tok = {c for c in tok if c}
    own = {min(c) for c in tok} if mode == "gen" else {mask_of(c) for c in tok}
    return own, tok


ARMS = [("en_s0", "given_cat_tok", "tok"),
        ("en_s0", "given_cat_min", "gen"),
        ("en_s0", "flat", None),
        ("en_s1", "flat_yk_tok", None),
        ("en_s1", "flat_yk_min", None)]
FIELDS = ("n_at_support", "n_keys_built", "n_entries", "inventory_max", "inventory_mean",
          "n_class_capped", "n_lower_classes", "n_lower_rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "phase0_open.json"))
    a = ap.parse_args()
    rules, canon = world()
    alpha, pairs = alphabets(rules, canon)
    truth = MC.true_tables(rules, DEPTH, S, V, M, 5)
    tset = {e: {tuple(int(x) for x in r) for r in truth[e]["flat"]} for e in (2, 3, 4, 5)}
    minpairs = {(min(c1), min(c2)) for c1, c2 in pairs[5]}
    OUT = {"world": {"v": V, "s": S, "depth": DEPTH, "m": M, "rule_seed": 0,
                     "mine_support": SUPPORT, "spell_cap": SPELL_CAP},
           "n_classes": {str(k): len(vv) for k, vv in alpha.items()},
           "n_legal_pairs": {str(k): len(vv) for k, vv in pairs.items()},
           "n_legal_pairs_min_coords_L5": len(minpairs)}

    print("=" * 104)
    print("FIGURED BASS — Q0: what the book would have held had the commit frozen only the key")
    print("=" * 104)
    print("offline, CPU-only.  Driven by the banked per-cycle `keys_at_support`; the open bit")
    print("changes `operative()` only, so every open column is an ALL-ELSE-EQUAL replay on the")
    print("banked observation stream, not a prediction of a run.")
    print(f"world: v={V} s={S} depth={DEPTH} m={M} rule_seed=0 support={SUPPORT} "
          f"spell_cap={SPELL_CAP}")
    print(f"token-class alphabet L1..L6: {[len(alpha[k]) for k in range(1, 7)]}   "
          f"legal class PAIRS L2..L6: {[len(pairs[k]) for k in range(2, 7)]}   "
          f"(L5 pairs in min coordinates: {len(minpairs)})")
    print("regimes: frozen = the run | open4 = only L4's inventory stays live | openAll = no")
    print("         inventory is ever frozen.  L3 is live in ALL THREE: no arm committed L3.")

    loaded, quots = {}, {}
    for tag, arm, mode in ARMS:
        r = load(tag, arm)
        if r is None:
            print(f"    (missing {tag}/{arm})")
            continue
        loaded[arm] = (tag, r, mode)
        quots[arm] = None if mode is None else QT.Quotient(mode, rules, canon, V, S, DEPTH)

    # ------------------------------------------------------------------ the clock ------ #
    print("\n[0] THE ARMS' CLOCKS — and where the commit sits in them")
    print("    the committable miners are era-gated to `era_level + 1`, so the L4 miner starts")
    print("    at era 3 and the L5 miner at era 4.  That gate, not the freeze, is why an L5")
    print("    key count is 0 before era 4.")
    print(f"    {'arm':>16} {'key':>5} {'cycles':>7} {'era starts (era:cycle)':>34} "
          f"{'commits':>26}")
    clocks = {}
    for arm, (tag, r, mode) in loaded.items():
        eb = era_bounds(r)
        cm = commits(r)
        clocks[arm] = {"tag": tag, "n_cycles": len(r["log"]["cycle"]), "eras": eb,
                       "commits": cm}
        print(f"    {arm:>16} {str(mode or 'flat'):>5} {len(r['log']['cycle']):>7} "
              f"{' '.join(f'{e}:c{c}' for e, c in eb):>34} "
              f"{' '.join(f'L{l}@c{c}' for l, c in sorted(cm.items())):>26}")
    OUT["sec0_clocks"] = clocks

    # ------------------------------------------------------------------ F-1 ------------ #
    print("\n[F-1] GATE — the replayed build reproduces the logged one, field for field")
    print("      (all eight `last_build` fields, on every cycle where the log carries a genuine")
    print("       live build, i.e. `n_lower_rows` > 0.  A committed level's entry is stale, and")
    print("       the audition's `build(base_table(v))` counterfactual leaves a degenerate one;")
    print("       both are skipped, and counted.)")
    print(f"      {'arm':>16} {'level':>6} {'checked':>8} {'agree':>7} {'skipped':>8} "
          f"{'mismatch cycles':>22} {'fields':>20}")
    gate1 = {}
    for arm, (tag, r, mode) in loaded.items():
        if mode is None:
            continue
        rep = replay(r, quots[arm], set(), pick="lo")
        for ell in range(2, MAXL + 1):
            n_ck = n_ok = n_sk = 0
            bad, badf = [], set()
            for i, row in enumerate(rep):
                lg = ((r["log"]["quot"][i] or {}).get("build") or {}).get(str(ell)) or {}
                st = row["stats"][ell]
                if not lg.get("n_lower_rows") or st.get("frozen"):
                    n_sk += 1
                    continue
                n_ck += 1
                diff = [f for f in FIELDS if abs(float(st[f]) - float(lg[f])) >= 1e-9]
                n_ok += int(not diff)
                if diff:
                    bad.append(row["cycle"])
                    badf |= set(diff)
            gate1[f"{arm}/L{ell}"] = {"checked": n_ck, "agree": n_ok, "skipped": n_sk,
                                      "mismatch_cycles": bad, "fields": sorted(badf)}
            rng = (f"c{bad[0]}..c{bad[-1]} ({len(bad)})" if bad else "—")
            print(f"      {arm:>16} {'L' + str(ell):>6} {n_ck:>8} {n_ok:>7} {n_sk:>8} "
                  f"{rng:>22} {(','.join(sorted(badf)) or '—'):>20}")
    print("\n      flat arms — the replayed committed table against the commit event's")
    print("      `n_entries` (a flat arm logs no `last_build`):")
    for arm, (tag, r, mode) in loaded.items():
        if mode is not None:
            continue
        rep = replay(r, None, set())
        line = []
        for ell, cc in sorted(commits(r).items()):
            i = r["log"]["cycle"].index(cc)
            got = len(rep[i]["books"][ell])
            wnt = int([e for e in r["events"]
                       if e.get("kind") == "commit" and e["level"] == ell][0]["n_entries"])
            line.append(f"L{ell}@c{cc}: {got}/{wnt}{'' if got == wnt else '  <-- MISMATCH'}")
            gate1[f"{arm}/L{ell}@commit"] = {"replay": got, "logged": wnt}
        print(f"      {arm:>16}  " + " | ".join(line))
    OUT["F1_gate"] = gate1

    # ------------------------------------------------------------------ F-2 ------------ #
    print("\n[F-2] GATE — is the answer pick-invariant?  class coverage and L5 lookup-ability")
    print("      under pick=lo (index order), pick=hi (reversed) and cap-free (spell_cap off,")
    print("      a strict upper bound), over EVERY cycle from the L4 commit to the end.")
    print(f"      {'arm':>16} {'quantity':>26} {'lo':>12} {'hi':>12} {'cap-free':>12} "
          f"{'invariant':>10}")
    gate2 = {}
    for arm, (tag, r, mode) in loaded.items():
        if mode is None:
            continue
        c4 = commits(r).get(4)
        if c4 is None:
            continue
        series = {}
        for pk in ("lo", "hi", "free"):
            rep = replay(r, quots[arm], {2, 3, 4, 5}, pick=pk)
            own_s, look_s = [], []
            for i, cy in enumerate(r["log"]["cycle"]):
                if cy < c4:
                    continue
                own, tok = book_classes(rep[i]["books"][4], 4, rules, canon, mode)
                own_s.append(len(own))
                look_s.append(sum(1 for p in minpairs if p[0] in own and p[1] in own)
                              if mode == "gen" else
                              sum(1 for c1, c2 in pairs[5]
                                  if mask_of(c1) in own and mask_of(c2) in own))
            series[pk] = {"own_last": own_s[-1], "own_max": max(own_s),
                          "look_last": look_s[-1], "look_max": max(look_s)}
        gate2[arm] = series
        for q in ("own_last", "own_max", "look_last", "look_max"):
            inv = series["lo"][q] == series["hi"][q] == series["free"][q]
            print(f"      {arm:>16} {q:>26} {series['lo'][q]:>12} {series['hi'][q]:>12} "
                  f"{series['free'][q]:>12} {('yes' if inv else 'NO'):>10}")
    OUT["F2_pick_invariance"] = gate2

    # ------------------------------------------------------------------ F-3 ------------ #
    print("\n[F-3] GATE — row IDENTITY.  The committed book's `true_mask` (per-row, in row")
    print("      order) against the replay's.  This is where the unlogged `spell` ranking bites:")
    print("      the count is right, the identity of the picked spellings is not.")
    print(f"      {'arm':>16} {'level':>6} {'n rows':>7} {'true (replay lo/hi)':>21} "
          f"{'true (logged)':>14} {'elementwise':>12}")
    gate3 = {}
    for arm, (tag, r, mode) in loaded.items():
        ent = load_entry(tag, arm)
        if ent is None:
            print(f"      {arm:>16}  (no entry.json.gz in the compact mirror — skipped)")
            continue
        reps = {pk: replay(r, quots[arm], set(), pick=pk) for pk in ("lo", "hi")}
        for ell, cc in sorted(commits(r).items()):
            i = r["log"]["cycle"].index(cc)
            want = (ent[i].get("true_mask") or {}).get(str(ell))
            if want is None:
                continue
            got = {pk: [int(tuple(x) in tset[ell]) for x in reps[pk][i]["books"][ell]]
                   for pk in ("lo", "hi")}
            same = got["lo"] == list(want)
            gate3[f"{arm}/L{ell}"] = {"n": len(want), "logged_true": int(sum(want)),
                                      "replay_true_lo": int(sum(got["lo"])),
                                      "replay_true_hi": int(sum(got["hi"])),
                                      "elementwise_lo": bool(same)}
            print(f"      {arm:>16} {'L' + str(ell):>6} {len(want):>7} "
                  f"{f'{sum(got["lo"])} / {sum(got["hi"])}':>21} {int(sum(want)):>14} "
                  f"{('MATCH' if same else 'differs'):>12}")
    OUT["F3_row_identity"] = gate3

    # ------------------------------------------------------------------ 1 --------------- #
    print("\n" + "=" * 104)
    print("[1] THE LIVE INVENTORY'S CLASS COVERAGE — per cycle from the L4 commit to the end")
    print("=" * 104)
    print("    `own` = classes in the arm's own key ('tok': of the 13 L4 token classes; 'min':")
    print("    of 8 min-labels).  `tok` = token classes of the same rows.  Rows printed only")
    print("    where something moves.  L3 is live in every regime (never committed).")
    sec1 = {}
    for arm, (tag, r, mode) in loaded.items():
        if mode is None:
            continue
        c4 = commits(r).get(4)
        if c4 is None:
            continue
        reps = {n: replay(r, quots[arm], lv, pick="lo") for n, lv in REGIMES}
        rows = []
        for i, cy in enumerate(r["log"]["cycle"]):
            if cy < c4:
                continue
            rec = {"cycle": cy, "L4_keys_at_support": len(keys_at(r, i, 4)),
                   "L3_keys_at_support": len(keys_at(r, i, 3))}
            for n, _ in REGIMES:
                bk = reps[n][i]["books"][4]
                own, tok = book_classes(bk, 4, rules, canon, mode)
                rec[n] = {"rows": len(bk), "own": len(own), "tok": len(tok)}
            l3 = reps["openAll"][i]["books"][3]
            o3, t3 = book_classes(l3, 3, rules, canon, mode)
            rec["L3_live"] = {"rows": len(l3), "own": len(o3), "tok": len(t3)}
            rows.append(rec)
        sec1[arm] = rows
        print(f"\n    --- {arm}  (key={mode}, L4 commit c{c4}, "
              f"{len(r['log']['cycle'])} cycles, era4 starts c"
              f"{dict(era_bounds(r)).get(4)}) ---")
        print(f"    {'cycle':>6} {'L4k@s':>6} {'L3k@s':>6} || "
              + " || ".join(f"{n + ' rows':>12} {'own':>4} {'tok':>4}"
                            for n, _ in REGIMES)
              + f" || {'L3 rows':>8} {'own':>4} {'tok':>4}")
        prev = None
        for rec in rows:
            sig = tuple(rec[n][k] for n, _ in REGIMES for k in ("rows", "own")) \
                + (rec["L3_live"]["own"], rec["L3_live"]["rows"])
            if prev is not None and sig == prev and rec is not rows[-1]:
                continue
            prev = sig
            print(f"    {rec['cycle']:>6} {rec['L4_keys_at_support']:>6} "
                  f"{rec['L3_keys_at_support']:>6} || "
                  + " || ".join(f"{rec[n]['rows']:>12} {rec[n]['own']:>4} {rec[n]['tok']:>4}"
                                for n, _ in REGIMES)
                  + f" || {rec['L3_live']['rows']:>8} {rec['L3_live']['own']:>4} "
                    f"{rec['L3_live']['tok']:>4}")
    OUT["sec1_L4_coverage"] = sec1

    # ------------------------------------------------------------------ 2 --------------- #
    print("\n" + "=" * 104)
    print("[2] WHAT L5 COULD HAVE BUILT OVER IT")
    print("=" * 104)
    print("    `look` = legal L5 class pairs both of whose halves the L4 book holds a row of")
    print(f"    (the `buildable()` idiom in class coordinates), of {len(pairs[5])} in token")
    print(f"    coordinates / {len(minpairs)} in min coordinates.  `L5k@s` = the COMMITTABLE L5")
    print("    miner's own keys at support 3 (era-gated to era 4).  `obs5` = the ungated")
    print("    observation panel's count (`obs_hist[5]`) — a different miner, same stream.")
    print("    `built`/`rows` = the L5 build `ClassMiner.build` would return that cycle.")
    sec2 = {}
    for arm, (tag, r, mode) in loaded.items():
        if mode is None:
            continue
        c4 = commits(r).get(4)
        if c4 is None:
            continue
        reps = {n: replay(r, quots[arm], lv, pick="lo") for n, lv in REGIMES}
        oh5 = (r.get("obs_hist") or {}).get("5") or []
        rows = []
        for i, cy in enumerate(r["log"]["cycle"]):
            if cy < c4:
                continue
            rec = {"cycle": cy, "L5_keys_at_support": len(keys_at(r, i, 5)),
                   "obs_hist_5": (int(oh5[i]) if i < len(oh5) else None)}
            for n, _ in REGIMES:
                bk = reps[n][i]["books"][4]
                own, _t = book_classes(bk, 4, rules, canon, mode)
                look = (sum(1 for p in minpairs if p[0] in own and p[1] in own)
                        if mode == "gen" else
                        sum(1 for c1, c2 in pairs[5]
                            if mask_of(c1) in own and mask_of(c2) in own))
                _l5, st5 = class_build(keys_at(r, i, 5), bk, quots[arm], 5, pick="lo")
                rec[n] = {"lookupable": look, "L5_keys_built": st5["n_keys_built"],
                          "L5_rows": st5["n_entries"]}
            rows.append(rec)
        sec2[arm] = rows
        print(f"\n    --- {arm}  (key={mode}, L4 commit c{c4}) ---")
        print(f"    {'cycle':>6} {'L5k@s':>6} {'obs5':>5} || "
              + " || ".join(f"{n + ' look':>13} {'built':>6} {'rows':>5}" for n, _ in REGIMES))
        prev = None
        for rec in rows:
            sig = tuple(rec[n][k] for n, _ in REGIMES
                        for k in ("lookupable", "L5_keys_built", "L5_rows")) \
                + (rec["L5_keys_at_support"],)
            if prev is not None and sig == prev and rec is not rows[-1]:
                continue
            prev = sig
            print(f"    {rec['cycle']:>6} {rec['L5_keys_at_support']:>6} "
                  f"{str(rec['obs_hist_5']):>5} || "
                  + " || ".join(f"{rec[n]['lookupable']:>13} {rec[n]['L5_keys_built']:>6} "
                                f"{rec[n]['L5_rows']:>5}" for n, _ in REGIMES))
        firsts = {n: next((rec["cycle"] for rec in rows if rec[n]["L5_rows"] > 0), None)
                  for n, _ in REGIMES}
        sec2[arm + "__first_nonempty_L5_build"] = firsts
        print("    first cycle with a NON-EMPTY L5 build:  "
              + "   ".join(f"{n}={firsts[n]}" for n, _ in REGIMES))
    OUT["sec2_L5"] = sec2

    # ------------------------------------------------------------------ 3 --------------- #
    print("\n" + "=" * 104)
    print("[3] THE FLAT KEY — the same replay where the key is the spelling")
    print("=" * 104)
    print("    A flat arm's book IS its at-support keys filtered by the ratchet, so the replay")
    print("    is exact (F-3).  `tok` = the token classes those rows cover, which is what the")
    print("    contrast with [1] is denominated in.  `L5` = rows an L5 flat build would return.")
    sec3 = {}
    for arm, (tag, r, mode) in loaded.items():
        if mode is not None:
            continue
        cm = commits(r)
        c4 = cm.get(4)
        reps = {n: replay(r, None, lv) for n, lv in REGIMES}
        rows = []
        start = c4 if c4 is not None else cm.get(2, r["log"]["cycle"][0])
        for i, cy in enumerate(r["log"]["cycle"]):
            if cy < start:
                continue
            rec = {"cycle": cy, "L4_keys_at_support": len(keys_at(r, i, 4)),
                   "L5_keys_at_support": len(keys_at(r, i, 5))}
            for n, _ in REGIMES:
                bk = reps[n][i]["books"][4]
                _own, tok = book_classes(bk, 4, rules, canon, "tok")
                l5, _ = flat_build(keys_at(r, i, 5), bk, 5)
                rec[n] = {"rows": len(bk), "tok": len(tok), "L5_rows": len(l5),
                          "L3_rows": len(reps[n][i]["books"][3])}
            rows.append(rec)
        sec3[arm] = rows
        print(f"\n    --- {arm} ({tag})  L2 commit c{cm.get(2)}, L4 commit c{c4}, "
              f"{len(r['log']['cycle'])} cycles ---")
        print(f"    {'cycle':>6} {'L4k@s':>6} {'L5k@s':>6} || "
              + " || ".join(f"{n + ' rows':>12} {'tok':>4} {'L5':>3}" for n, _ in REGIMES)
              + f" | {'L3 rows':>7}")
        prev = None
        for rec in rows:
            sig = tuple(rec[n][k] for n, _ in REGIMES for k in ("rows", "tok", "L5_rows"))
            if prev is not None and sig == prev and rec is not rows[-1]:
                continue
            prev = sig
            print(f"    {rec['cycle']:>6} {rec['L4_keys_at_support']:>6} "
                  f"{rec['L5_keys_at_support']:>6} || "
                  + " || ".join(f"{rec[n]['rows']:>12} {rec[n]['tok']:>4} "
                                f"{rec[n]['L5_rows']:>3}" for n, _ in REGIMES)
                  + f" | {rec['openAll']['L3_rows']:>7}")
    OUT["sec3_flat"] = sec3

    # ------------------------------------------------------------------ 4 --------------- #
    print("\n" + "=" * 104)
    print("[4] THE EXECUTOR'S EXPOSURE — rows the DP maxes over, frozen vs live")
    print("=" * 104)
    print("    `macro_features`' max-sum DP runs over the operative table's ENTRIES; under a")
    print("    class key each surviving key contributes up to spell_cap**s = 16 spellings, so")
    print("    the expansion choice is the inventory.  Reported at the L4 commit, at the median")
    print("    cycle after it, and at the last cycle.  `inv max/mean` and `cap` are the openAll")
    print("    L4 build's own `inventory_max` / `inventory_mean` / `n_class_capped`.")
    print(f"    {'arm':>16} {'key':>5} {'cycle':>6} {'frozen':>7} {'open4':>7} {'openAll':>8} "
          f"{'openAll/frozen':>15} {'inv max':>8} {'inv mean':>9} {'cap binds':>10} "
          f"{'L5 rows (froz/openAll)':>23}")
    sec4 = {}
    for arm, (tag, r, mode) in loaded.items():
        c4 = commits(r).get(4)
        if c4 is None:
            continue
        q = quots[arm]
        reps = {n: replay(r, q, lv, pick="lo") for n, lv in REGIMES}
        idxs = [i for i, cy in enumerate(r["log"]["cycle"]) if cy >= c4]
        recs = []
        for label, i in (("commit", idxs[0]), ("median", idxs[len(idxs) // 2]),
                         ("last", idxs[-1])):
            cy = r["log"]["cycle"][i]
            n = {k: len(reps[k][i]["books"][4]) for k, _ in REGIMES}
            st = reps["openAll"][i]["stats"][4]
            if q is None:
                l5f = len(flat_build(keys_at(r, i, 5), reps["frozen"][i]["books"][4], 5)[0])
                l5o = len(flat_build(keys_at(r, i, 5), reps["openAll"][i]["books"][4], 5)[0])
            else:
                l5f = class_build(keys_at(r, i, 5), reps["frozen"][i]["books"][4], q, 5)[1]
                l5o = class_build(keys_at(r, i, 5), reps["openAll"][i]["books"][4], q, 5)[1]
                l5f, l5o = l5f["n_entries"], l5o["n_entries"]
            rec = {"label": label, "cycle": cy, **n,
                   "inv_max": st.get("inventory_max"), "inv_mean": st.get("inventory_mean"),
                   "cap": st.get("n_class_capped"), "L5_rows_frozen": l5f,
                   "L5_rows_openAll": l5o}
            recs.append(rec)
            ratio = (n["openAll"] / n["frozen"]) if n["frozen"] else float("nan")
            im = rec["inv_mean"]
            print(f"    {arm:>16} {str(mode or 'flat'):>5} {cy:>6} {n['frozen']:>7} "
                  f"{n['open4']:>7} {n['openAll']:>8} {ratio:>15.2f} "
                  f"{str(rec['inv_max'] if rec['inv_max'] is not None else '—'):>8} "
                  f"{(f'{im:.2f}' if im is not None else '—'):>9} "
                  f"{str(rec['cap'] if rec['cap'] is not None else '—'):>10} "
                  f"{f'{l5f} / {l5o}':>23}")
        sec4[arm] = recs
    print("\n    the same last cycle with `spell_cap` OFF — the full cross-product the class")
    print("    key licenses, which is what the cap is holding back:")
    print(f"    {'arm':>16} {'L4 rows (cap 4)':>16} {'L4 rows (cap-free)':>19} "
          f"{'L5 rows (cap 4)':>16} {'L5 rows (cap-free)':>19}")
    capfree = {}
    for arm, (tag, r, mode) in loaded.items():
        if mode is None or commits(r).get(4) is None:
            continue
        q = quots[arm]
        got = {}
        for pk in ("lo", "free"):
            rep = replay(r, q, {2, 3, 4, 5}, pick=pk)
            i = len(rep) - 1
            got[pk] = (len(rep[i]["books"][4]),
                       class_build(keys_at(r, i, 5), rep[i]["books"][4], q, 5,
                                   pick=pk)[1]["n_entries"])
        capfree[arm] = {"L4_cap4": got["lo"][0], "L4_capfree": got["free"][0],
                        "L5_cap4": got["lo"][1], "L5_capfree": got["free"][1]}
        print(f"    {arm:>16} {got['lo'][0]:>16} {got['free'][0]:>19} "
              f"{got['lo'][1]:>16} {got['free'][1]:>19}")
    OUT["sec4_capfree"] = capfree
    OUT["sec4_exposure"] = sec4


    # ------------------------------------------------------------------ 5 --------------- #
    print("\n" + "=" * 104)
    print("[5] THE KEYS THEMSELVES — what the frozen book holds, what the live one does, and")
    print("    which L5 keys name a class the frozen book lacks")
    print("=" * 104)
    print("    `legal` counts the at-support keys that are legal class pairs of the world (the")
    print("    40 at L4 / 73 at L5 the closure enumerates); the rest are junk keys the miner")
    print("    accrued.  Classes are printed as the feature sets they are.")
    sec5 = {}
    legal4 = {(mask_of(c1), mask_of(c2)) for c1, c2 in pairs[4]}
    legal5 = {(mask_of(c1), mask_of(c2)) for c1, c2 in pairs[5]}
    for arm, (tag, r, mode) in loaded.items():
        if mode is None:
            continue
        c4 = commits(r).get(4)
        if c4 is None:
            continue
        reps = {n: replay(r, quots[arm], lv, pick="lo") for n, lv in REGIMES}
        out = {}
        for label, i in (("at the L4 commit", r["log"]["cycle"].index(c4)),
                         ("at the last cycle", len(r["log"]["cycle"]) - 1)):
            k4, k5 = keys_at(r, i, 4), keys_at(r, i, 5)
            fr = book_classes(reps["frozen"][i]["books"][4], 4, rules, canon, mode)
            op = book_classes(reps["openAll"][i]["books"][4], 4, rules, canon, mode)
            blocked = [k for k in k5 if not (k[0] in op[0] and k[1] in op[0])]
            blocked_fr = [k for k in k5 if not (k[0] in fr[0] and k[1] in fr[0])]
            rec = {"cycle": r["log"]["cycle"][i],
                   "L4_keys_at_support": len(k4),
                   "L4_keys_legal": (sum(1 for k in k4 if tuple(k) in legal4)
                                     if mode == "tok" else None),
                   "L5_keys_at_support": len(k5),
                   "L5_keys_legal": (sum(1 for k in k5 if tuple(k) in legal5)
                                     if mode == "tok" else None),
                   "L5_keys_blocked_by_frozen_book": len(blocked_fr),
                   "L5_keys_blocked_by_live_book": len(blocked),
                   "frozen_classes": sorted(sorted(c) for c in fr[1]),
                   "live_classes": sorted(sorted(c) for c in op[1])}
            out[label] = rec
            print(f"\n    {arm} — {label} (c{rec['cycle']}, key={mode})")
            print(f"      L4 keys at support {rec['L4_keys_at_support']}"
                  + (f" ({rec['L4_keys_legal']} legal, of 40 legal L4 class pairs)"
                     if rec["L4_keys_legal"] is not None else "")
                  + f"   |   L5 keys at support {rec['L5_keys_at_support']}"
                  + (f" ({rec['L5_keys_legal']} legal, of 73)"
                     if rec["L5_keys_legal"] is not None else ""))
            print(f"      frozen book holds {len(rec['frozen_classes'])} token classes: "
                  + " ".join("{" + ",".join(map(str, c)) + "}"
                             for c in rec["frozen_classes"]))
            print(f"      live   book holds {len(rec['live_classes'])} token classes: "
                  + " ".join("{" + ",".join(map(str, c)) + "}"
                             for c in rec["live_classes"]))
            print(f"      L5 keys at support blocked by the FROZEN book: "
                  f"{rec['L5_keys_blocked_by_frozen_book']} of "
                  f"{rec['L5_keys_at_support']}   |   by the LIVE book: "
                  f"{rec['L5_keys_blocked_by_live_book']}")
        sec5[arm] = out
    OUT["sec5_keys"] = sec5

    with open(a.json, "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
