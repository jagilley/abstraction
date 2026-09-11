"""Reduction for `enharmonic` Q1 (the supplied quotient) — numbers, not interpretation.

Reads the COMPACT local mirror (`fetch_compact.py`), so `log["entry"]` lives in
`entry.json.gz` beside each arm and only `entry_beam.json` is needed here.

Sections, in the order the node's SPEC asks for them:
  [A] the arms, their lifetimes and what they committed
  [B] |T5| at support, and whether L5 committed
  [C] the era-5 signature against `flat` (the value clock past the certified range)
  [D] pi's per-slot mass
  [E] the deletion battery, with the GROWTH columns reduced
  [F] both precisions per level, in BOTH spaces — feature (`true_mask`, what the arc has
      always reported) and TOKEN (`possible_sets(canon[row])`, what the executor writes and
      what the exact grader sees; `sizing/SIZING.md` section 6 is why they differ)
  [G] the L4 book's class coverage per cycle — the constraint Q0's addendum says binds L5 once
      arrival is solved. Read off `log["quot"][c]["build"]["5"]["n_lower_classes"]` for a
      quotiented arm and computed through `quotient.py` for `flat`.
  [H] the oracle bill and the priced clock

Usage (from experiments/):
    python3 rhm/practice/enharmonic/analyze_enharmonic.py --tag en_s0
"""

import argparse
import json
import os
import sys

import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.practice.ratchet import macros as MC
from rhm.practice.enharmonic import quotient as QT

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
V, S, DEPTH, M = 8, 2, 6, 2


def load(tag):
    root = os.path.join(FIG, tag)
    out = []
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if os.path.isfile(p):
            out.append((arm, json.load(open(p))))
    # `flat` first, then the treated arms
    return sorted(out, key=lambda kv: (kv[0] != "flat", kv[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="en_s0")
    ap.add_argument("--yoke-tag", default="", dest="yoke_tag",
                    help="a tag of clock-yoke arms to check against --tag's sources")
    ap.add_argument("--bank", default="",
                    help="banked arms from other tags to reduce alongside this one, as "
                         "TAG:ARM[,TAG:ARM...] (e.g. en_s0:flat,en_s2:given_cat_tok). "
                         "They enter `arms` as full participants; provenance is printed.")
    a = ap.parse_args()
    arms = load(a.tag)
    prov = {arm: a.tag for arm, _ in arms}
    for spec in [z for z in a.bank.split(",") if z.strip()]:
        btag, _, barm = spec.strip().partition(":")
        bp = os.path.join(FIG, btag, barm, "results.json")
        if not os.path.isfile(bp):
            print(f"    (bank {spec} not found locally — skipped)")
            continue
        _nm = barm if barm not in dict(arms) else f"{barm}@{btag}"
        arms.append((_nm, json.load(open(bp))))
        prov[_nm] = btag
    arms = sorted(arms, key=lambda kv: (kv[0] != "flat", kv[0]))
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=0)
    canon = np.ascontiguousarray(rules[DEPTH - 1][:, 0, :])
    truth = MC.true_tables(rules, DEPTH, S, V, M, 5)
    truth_flat = {e: {tuple(int(x) for x in r) for r in truth[e]["flat"]} for e in (2, 3, 4, 5)}
    # the level's token-class alphabet, for the coverage denominators
    clos = QT.class_closure(rules, canon, V, S, DEPTH, 6)

    def class_step(c1, c2, ell):
        """`possible_sets`' composition step on classes — the same rule `class_closure`
        iterates, applied to one pair."""
        layer = rules[DEPTH - ell]
        return frozenset(f for f in range(V) for rr in range(M)
                         if int(layer[f, rr, 0]) in c1 and int(layer[f, rr, 1]) in c2)

    # the yoke tag's arms, so [G2] and [M] cover every arm, treated and yoked
    extra_arms = []
    if a.yoke_tag:
        _yr = os.path.join(FIG, a.yoke_tag)
        if os.path.isdir(_yr):
            for _ya in sorted(os.listdir(_yr)):
                _yp = os.path.join(_yr, _ya, "results.json")
                if os.path.isfile(_yp) and _ya not in dict(arms):
                    extra_arms.append((_ya, json.load(open(_yp))))

    def endo_half_class(half, lv):
        """A LEARNED (endo) class id is a representative flat row of level `lv`; its token
        class is that row's `possible_sets`, so the endogenous partition is read in the same
        coordinates as `tok`/`min` and its coverage denominators mean the same thing."""
        row = tuple(int(x) for x in half)
        if lv < 2:
            return frozenset({row[0]}) if len(row) == 1 else frozenset()
        tc = tok_classes([row], lv)
        return tc[0] if tc else frozenset()

    def key_classes(st, lv):
        """The level-`lv` classes the miner's at-support KEYS compose to (the [F] object)."""
        ks = (st or {}).get("keys_at_support") or []
        keyed = (st or {}).get("keyed_by") or ""
        out = []
        for k in ks:
            if not (isinstance(k, list) and len(k) == S):
                continue
            if "tok" in keyed:
                c1 = frozenset(i for i in range(V) if int(k[0]) >> i & 1)
                c2 = frozenset(i for i in range(V) if int(k[1]) >> i & 1)
            elif "endo" in keyed:
                c1 = endo_half_class(k[0], lv - 1)
                c2 = endo_half_class(k[1], lv - 1)
            elif keyed:
                c1, c2 = frozenset({int(k[0])}), frozenset({int(k[1])})
            else:
                return []
            out.append(class_step(c1, c2, lv))
        return out

    def tok_classes(rows, level):
        if not rows:
            return []
        P = QT.token_class_sets(rules, np.array(rows, np.int64), level, canon,
                                V, S, DEPTH)[:, 0, :]
        return [frozenset(np.nonzero(r)[0].tolist()) for r in P]

    print("=" * 96)
    print(f"ENHARMONIC — {a.tag}   (reduction; numbers, not interpretation)")
    print("=" * 96)
    print("reduced by: python3 rhm/practice/enharmonic/analyze_enharmonic.py "
          + " ".join(sys.argv[1:]) + "   (from experiments/)")
    # the run's own provenance, in `tutti/figures/tu_s0_reduction.txt`'s idiom
    setup = {}
    _sp = os.path.join(FIG, a.tag, "setup.json")
    if os.path.isfile(_sp):
        setup = json.load(open(_sp))
    if setup:
        cf = setup.get("config") or {}
        print(f"era caps {setup.get('era_caps')} = {setup.get('cap_total')} cycles/arm "
              f"scheduled; setup {setup.get('t_setup_s', 0):,.0f}s; "
              f"max_macro_level {cf.get('max_macro_level')}; mine_support "
              f"{cf.get('mine_support')}; seed {cf.get('seed')}")
        print(f"floors that governed the run: {setup.get('floors')}")
        mk = {k: cf.get(k) for k in ("tol_yield_l5", "tol_yield_l6", "merge_every",
                                     "merge_n_pairs", "merge_margin", "merge_max_rows",
                                     "quot_spell_cap") if cf.get(k) is not None}
        if mk:
            print(f"quotient/merge knobs: {mk}")
    others = sorted({t for t in prov.values() if t != a.tag})
    if others:
        print("banked arms reduced alongside this tag: "
              + ", ".join(f"{arm} (from {t})" for arm, t in sorted(prov.items())
                          if t != a.tag))
    if a.yoke_tag:
        print(f"clock-yoke tag for section [Y]: {a.yoke_tag}")

    # ---- [A] lifetimes and commits ------------------------------------------------------ #
    print("\n[A] THE ARMS — lifetime, loop actions, and what was committed")
    print(f"    {'arm':>16} {'cycles':>7} {'t_cum(g)':>12} {'commits (level@cycle, entries)':>46} "
          f"{'advances':>9}")
    for arm, r in arms:
        ev = [(e["level"], e["cycle"], e["n_entries"]) for e in r["events"]
              if e["kind"] == "commit"]
        adv = [e["cycle"] for e in r["events"] if e["kind"] == "advance"]
        cs = " ".join(f"L{l}@c{c}:{n}" for l, c, n in ev) or "(none)"
        print(f"    {arm:>16} {len(r['log']['cycle']):>7} {r['log']['t_cum'][-1]:>12,.0f} "
              f"{cs:>46} {len(adv):>9}")
        q = r.get("quotient") or {}
        print(f"    {'':>16} quotient={q.get('mode')} spell_cap={q.get('spell_cap')} "
              f"oracle_reads={q.get('n_read')} drops={q.get('n_drop')} ties={q.get('n_tie')}")

    # ---- [B] L5 ------------------------------------------------------------------------- #
    print("\n[B] L5 — the rung the node exists for")
    print(f"    {'arm':>16} {'L5 committed':>13} {'|T5| at support (last)':>23} "
          f"{'L5 obs@sup3':>12} {'L6 obs@sup3':>12} {'read_level (era 4/5)':>21}")
    for arm, r in arms:
        c5 = [e for e in r["events"] if e["kind"] == "commit" and e["level"] == 5]
        mn5 = (r["log"]["miner"][-1] or {}).get("5") or {}
        oh = r.get("obs_hist") or {}
        pan = r["log"]["panel"]
        rl = sorted({q["read_level"] for q in pan if q.get("era", 0) >= 4})
        print(f"    {arm:>16} {('YES c' + str(c5[0]['cycle']) if c5 else 'no'):>13} "
              f"{str(mn5.get('n_at_support', {}).get('3')):>23} "
              f"{str((oh.get('5') or [None])[-1]):>12} {str((oh.get('6') or [None])[-1]):>12} "
              f"{str(rl):>21}")

    # ---- [C] the era-5 signature -------------------------------------------------------- #
    print("\n[C] THE SIGNATURE — the value clock in the consumption eras, against `flat`")
    # [figured_bass] CAVEAT ON ANY `ungate_l5` ROW BELOW. The knob moves no loop action —
    # `given_cat_tok_ung5` and the banked `given_cat_tok` have bit-identical commits, advances
    # and `why`, and identical `n_moves` on all 176 cycles — but it is NOT stream-neutral. An
    # ungated L5 miner makes the audition block's L5 build non-empty far earlier (c91 against
    # c156 in `fb_s0`), and each non-empty L5 audition draws from the shared torch stream. `e`
    # first differs from the anchor at c92, one cycle after that first audition. So an era-4/5
    # `e` difference between an `ung5` arm and its non-ungated twin is downstream of an RNG
    # shift and is not, on this evidence, a performance effect.
    print("    [figured_bass] `ungate_l5` arms: the knob moves no loop action but is NOT")
    print("    stream-neutral — the first non-empty L5 audition moves from c156 to c91 and")
    print("    draws on the shared torch stream; `e` diverges from the anchor at c92. Read an")
    print("    `ung5`-vs-anchor era-4/5 gap as an RNG shift, not a performance effect.")
    base = dict(arms).get("flat")
    if base is None:
        print("    (no `flat` arm in this tag and none banked — the `vs flat` column is blank)")
    print(f"    {'arm':>16} {'era':>4} {'cycles':>7} {'e (mean)':>10} {'e (last)':>10} "
          f"{'succ (mean)':>12} {'vs flat (e)':>12}")
    for arm, r in arms:
        for era in (4, 5):
            idx = [i for i, e in enumerate(r["log"]["era"]) if e == era]
            if not idx:
                print(f"    {arm:>16} {era:>4} {'0':>7}   (era never reached)")
                continue
            e_ = [r["log"]["e"][i] for i in idx]
            s_ = [r["log"]["succ"][i] for i in idx]
            bi = ([i for i, x in enumerate(base["log"]["era"]) if x == era]
                  if base is not None else [])
            be = np.mean([base["log"]["e"][i] for i in bi]) if bi else float("nan")
            print(f"    {arm:>16} {era:>4} {len(idx):>7} {np.mean(e_):>10.4f} {e_[-1]:>10.4f} "
                  f"{np.mean(s_):>12.4f} {np.mean(e_) - be:>+12.4f}")

    # ---- [D] pi's per-slot mass --------------------------------------------------------- #
    print("\n[D] pi's PER-SLOT MASS at the last cycle (the routed policy over macro slots)")
    for arm, r in arms:
        pr = [q for q in r["log"]["prop"] if q]
        last = pr[-1] if pr else None
        print(f"    {arm:>16} {json.dumps(last)[:150] if last else '(no prop log)'}")

    # ---- [E] the battery ---------------------------------------------------------------- #
    print("\n[E] THE DELETION BATTERY — the growth columns")
    print("     [en_s3] CAVEAT ON `built`: it builds each level over `operative(ell-1)` — the")
    print("     ARM'S OWN committed-or-live lower table — in EVERY condition, so under")
    print("     `b_table` its lower table is undeleted and `b_table == a_full` on `built` is an")
    print("     identity of the probe, not a finding (inherited from `tutti`/`native`, whose")
    print("     own comment calls it the structural half). `built_fresh`, new this round,")
    print("     builds each level over the CONDITION'S OWN fresh chain (`base_table` at the")
    print("     bottom, each level feeding the next; a dead level kills the chain above it).")
    print("     The informative half has always been `at_support`, which is table-free.")
    for arm, r in arms:
        ab = r.get("ablation") or {}
        if not ab:
            print(f"    {arm:>16} (no battery)")
            continue
        print(f"    {arm:>16}")
        for cname, cell in ab.items():
            if not isinstance(cell, dict) or "eras" not in cell:
                print(f"        {cname:>12}: {str(cell)[:90]}")
                continue
            for j, era_c in (cell.get("eras") or {}).items():
                mine = era_c.get("mine") or {}
                g = " ".join(
                    f"L{l}[sup {d['at_support']} obs {d['n_obs']} built {d['built']}"
                    + (f" fresh {d['built_fresh']}]" if "built_fresh" in d else "]")
                    for l, d in sorted(mine.items(), key=lambda kv: int(kv[0])))
                print(f"        {cname:>12} era{j}: e={era_c.get('e')} "
                      f"solved={era_c.get('n_solved')}  {g}")

    # ---- [E2] THE INFORMATIVE HALF: the at-support stream per condition, per era -------- #
    print("\n[E2] THE BATTERY'S TABLE-FREE HALF — at-support keys per condition per era, L5")
    print("     first (the rung the node exists for) and L2-L4 beside it. This is the column")
    print("     [E]'s caveat says to read: it is mined from the condition's OWN chosen")
    print("     trajectories and no table enters it.")
    for arm, r in arms:
        ab = r.get("ablation") or {}
        cells = [(c, v) for c, v in ab.items() if isinstance(v, dict) and "eras" in v]
        if not cells:
            continue
        print(f"    {arm:>18}")
        for lv in ("5", "4", "3", "2"):
            for cname, cell in cells:
                ser, bf = [], []
                for j in sorted((cell.get("eras") or {}), key=int):
                    d = ((cell["eras"][j].get("mine") or {}).get(lv) or {})
                    ser.append(d.get("at_support"))
                    bf.append(d.get("built_fresh"))
                if not any(x for x in ser) and lv != "5":
                    continue
                print(f"        L{lv} {cname:>9}  at_support/era {ser}"
                      + (f"   built_fresh/era {bf}" if any(x is not None for x in bf) else ""))

    # ---- [F] both precisions ------------------------------------------------------------ #
    print("\n[F] PRECISION IN BOTH SPACES, per committed level")
    print(f"    {'arm':>16} {'lvl':>4} {'rows':>6} {'true (feature)':>15} {'prec_feature':>13} "
          f"{'true (token)':>13} {'prec_token':>11} {'recall_feature':>15}")
    books = {}
    for arm, r in arms:
        for e in [q for q in r["events"] if q["kind"] == "commit"]:
            lv = int(e["level"])
            keys = e.get("keys") or None
            books.setdefault(arm, {})
        # the committed rows are not in the event; replay them from the last cycle's grade
        # and the miner is class-keyed, so use `log["vocab"]` sizes and `committed_grade`
        cg = r["log"]["committed_grade"][-1] or {}
        for lv in sorted(cg):
            if cg[lv] is None:
                continue
            print(f"    {arm:>16} {lv:>4} {cg[lv]['n_learned']:>6} "
                  f"{cg[lv]['n_correct']:>15} {(cg[lv]['precision'] or 0):>13.4f} "
                  f"{'—':>13} {'—':>11} {cg[lv]['recall']:>15.4f}")
    print("    (token-space precision needs the committed ROWS; see [F2])")

    # ---- [F2] token space --------------------------------------------------------------- #
    print("\n[F2] TOKEN SPACE. `sizing/SIZING.md` section 6: 68-96% of what `true_mask` calls")
    print("     junk are legal PROGRAMS, so feature-space precision understates. What is exact")
    print("     from the compact log differs by arm, and the difference is stated rather than")
    print("     papered over:")
    print("       flat            the committed ROWS are the miner's own keys, so both")
    print("                       precisions are exact.")
    print("       given_cat_*     the committed rows are a CROSS-PRODUCT the log does not")
    print("                       carry; what IS exact is the KEY-level legality — the")
    print("                       fraction of committed class-pair keys whose `class_step` is")
    print("                       non-empty, i.e. whose spellings compose to a legal program at")
    print("                       all. The row-weighted number needs a per-commit row dump,")
    print("                       which is now in the fork for Q2 and was not here.")
    print(f"\n    {'arm':>16} {'lvl':>4} {'n':>6} {'prec_feature':>13} {'prec_token':>11} "
          f"{'unit':>6} {'classes held':>13} {'of alphabet':>12}")
    for arm, r in arms:
        st = r["log"]["miner"][-1] or {}
        vocab = r["log"]["vocab"][-1] or {}
        for lv_s in sorted(st, key=int):
            lv = int(lv_s)
            if vocab.get(lv_s) is None:
                continue
            ks = (st[lv_s] or {}).get("keys_at_support") or []
            keyed = (st[lv_s] or {}).get("keyed_by")
            if not ks:
                continue
            if not keyed:                      # FLAT: the keys are the rows
                rows = [tuple(k) for k in ks if len(k) == S ** (lv - 1)]
                if not rows:
                    continue
                tc = tok_classes(rows, lv)
                nf = sum(1 for k in rows if k in truth_flat.get(lv, set()))
                nt = sum(1 for c in tc if c)
                print(f"    {arm:>16} {lv:>4} {len(rows):>6} {nf / len(rows):>13.4f} "
                      f"{nt / len(rows):>11.4f} {'row':>6} "
                      f"{len({c for c in tc if c}):>13} {clos[lv]['n_classes']:>12}")
            else:                              # CLASS-KEYED: key-level legality, exact
                # a 'tok' class id IS the bitmask over the v features, so it decodes with no
                # lookup; a 'min' class id is a single feature.
                ok = tot = 0
                held = set()
                for k in ks:
                    if not (isinstance(k, list) and len(k) == S):
                        continue
                    if "tok" in keyed:
                        c1 = frozenset(i for i in range(V) if int(k[0]) >> i & 1)
                        c2 = frozenset(i for i in range(V) if int(k[1]) >> i & 1)
                    elif "endo" in keyed:      # learned: the id IS a representative row
                        c1 = endo_half_class(k[0], lv - 1)
                        c2 = endo_half_class(k[1], lv - 1)
                    else:                      # min-label: a single feature id
                        c1, c2 = frozenset({int(k[0])}), frozenset({int(k[1])})
                    if not c1 or not c2:
                        continue
                    got = class_step(c1, c2, lv)
                    tot += 1
                    ok += bool(got)
                    if got:
                        held.add(got)
                if tot:
                    print(f"    {arm:>16} {lv:>4} {tot:>6} {'—':>13} {ok / tot:>11.4f} "
                          f"{'key':>6} {len(held):>13} {clos[lv]['n_classes']:>12}")

    # ---- [G] the L4 book's class coverage ----------------------------------------------- #
    print("\n[G] THE L4 BOOK'S CLASS COVERAGE — what the L5 build could look up")
    print("    (per cycle, off `log[\"quot\"][c][\"build\"]`.  DENOMINATOR: a SUPPLIED quotient")
    print("     ('tok'/'min') is keyed by the fixed token-class alphabet, 13 L4 classes and 73")
    print("     legal L5 class pairs, so `lower classes` is out of 13.  An ENDOGENOUS quotient")
    print("     has no fixed alphabet: its ids are learned union-find representatives over the")
    print("     raw rows it has actually seen, so the only honest denominator is the RAW CLASS")
    print("     COUNT — `n_lower_rows`, the distinct flat lower rows the map partitions, one")
    print("     class each before any merge.  Both are printed per row below.)")
    for arm, r in arms:
        qz = [q for q in r["log"]["quot"] if q]
        if not qz:
            print(f"    {arm:>16} (flat — no quotient; its L4 book is graded in [F2])")
            continue
        qmode = (r.get("quotient") or {}).get("mode")
        endo = qmode == "endo"
        print(f"    {arm:>16}  quotient={qmode}  denominator="
              f"{'raw class count (n_lower_rows)' if endo else 'token-class alphabet'}")
        print(f"        {'build':>7} {'cycles':>7} {'keys@sup':>16} {'keys built':>14} "
              f"{'entries':>14} {'lower classes':>16} {'of (raw)':>12} {'of (alphabet)':>14} "
              f"{'cap binds':>11}")
        for lv in ("2", "3", "4", "5"):
            bl = [(q["build"].get(lv) or {}) for q in qz
                  if (q["build"].get(lv) or {}).get("n_lower_rows")]
            if not bl:
                continue
            def rng(k, rows=bl):
                vals = [x.get(k) for x in rows if x.get(k) is not None]
                return f"{min(vals)}..{max(vals)}" if vals else "-"
            alpha = clos[int(lv) - 1]["n_classes"]
            print(f"        {'L' + lv:>7} {len(bl):>7} {rng('n_at_support'):>16} "
                  f"{rng('n_keys_built'):>14} {rng('n_entries'):>14} "
                  f"{rng('n_lower_classes'):>16} {rng('n_lower_rows'):>12} "
                  f"{('—' if endo else str(alpha)):>14} {rng('n_class_capped'):>11}")

    # ---- [G2] the L4 book AT COMMIT, every arm ------------------------------------------ #
    print("\n[G2] THE L4 BOOK AT ITS COMMIT — cycle and class coverage, side by side")
    print("     A quotiented arm reads `n_lower_classes` off the first L5 build after the")
    print("     commit; a FLAT arm's committed rows are its own at-support keys filtered by the")
    print("     ratchet, replayed here, and their token classes counted the same way.")
    print("     DENOMINATOR, as in [G]: a supplied quotient and a flat arm are counted out of")
    print("     the 13-class L4 token alphabet; an ENDOGENOUS arm is counted out of its RAW")
    print("     CLASS COUNT (the distinct flat L4 rows its learned map partitions).")
    print(f"    {'arm':>18} {'key':>6} {'L4 commit':>10} {'rows':>6} "
          f"{'L4 classes at commit':>21} {'of':>10} {'L4 classes, live miner at end':>31}")

    def flat_rows_at(r, cyc_i, level):
        """`Miner.build`'s ratchet, replayed from the logged key streams — the flat arm's
        committed rows at that cycle."""
        ks = {}
        for lv in range(2, level + 1):
            st = (r["log"]["miner"][cyc_i] or {}).get(str(lv)) or {}
            ks[lv] = [tuple(k) for k in (st.get("keys_at_support") or [])]
        low = {tuple(int(x) for x in q) for q in MC.base_table(V)["flat"]}
        for lv in range(2, level + 1):
            half = (S ** (lv - 1)) // S
            rows = [k for k in ks[lv]
                    if len(k) == S ** (lv - 1)
                    and all(k[i * half:(i + 1) * half] in low for i in range(S))]
            low = set(rows)
        return rows

    for arm, r in list(arms) + extra_arms:
        ev4 = [e for e in r["events"] if e["kind"] == "commit" and e["level"] == 4]
        q_on = bool([q for q in r["log"]["quot"] if q])
        endo = (r.get("quotient") or {}).get("mode") == "endo"
        if not ev4:
            print(f"    {arm:>18} {('class' if q_on else 'flat'):>6} {'(none)':>10}")
            continue
        c4 = int(ev4[0]["cycle"])
        i4 = r["log"]["cycle"].index(c4)
        if q_on:
            after = [(r["log"]["quot"][i]["build"].get("5") or {})
                     for i in range(i4, len(r["log"]["cycle"]))
                     if r["log"]["quot"][i]
                     and (r["log"]["quot"][i]["build"].get("5") or {}).get("n_lower_rows")]
            nc = after[0]["n_lower_classes"] if after else None
            nr = after[0]["n_lower_rows"] if after else ev4[0]["n_entries"]
            st = (r["log"]["miner"][-1] or {}).get("4") or {}
            live = len({z for z in (key_classes(st, 4) or []) if z})
        else:
            rows = flat_rows_at(r, i4, 4)
            nc = len({c for c in tok_classes(rows, 4) if c}) if rows else 0
            nr = len(rows)
            rows_end = flat_rows_at(r, len(r["log"]["cycle"]) - 1, 4)
            live = len({c for c in tok_classes(rows_end, 4) if c}) if rows_end else 0
        den = f"{nr} raw" if endo else f"{clos[4]['n_classes']} tok"
        print(f"    {arm:>18} {('class' if q_on else 'flat'):>6} {('c' + str(c4)):>10} "
              f"{nr:>6} {str(nc):>21} {den:>10} {live:>31}")

    # ---- [M] the merge events ------------------------------------------------------------ #
    print("\n[M] MERGE EVENTS — every proposal, taken or refused")
    _s3 = any(e.get("kind") == "merge_proposal"
              for _a, _r in list(arms) + extra_arms for e in (_r.get("merge_events") or []))
    if _s3:
        print("     [en_s3] THIS TAG'S RECORDS ARE THE WHOLE-PARTITION SHAPE: one")
        print("     `merge_proposal` per entry into the block (what the probe found and what")
        print("     the op did with it) and one `merge` event per alias GROUP the closure")
        print("     found. The ledger audition is at l+1 — the level a level-l merge actually")
        print("     changes — and carries `x_differ`, the fraction of instances on which the")
        print("     kept and merged tables write a DIFFERENT state. `x_differ == 0` with")
        print("     `e_merge == e_keep` means the DP's winning entry survived the merge and")
        print("     the equality says nothing about it; `x_differ > 0` with the errors equal")
        print("     means the merge was neutral on the level above. The note below describes")
        print("     `en_s2b`'s records and is kept for reading the banked tags.")
    print("     WHAT `e_keep`/`e_merge` EXECUTED IN en_s2b (code facts, that round's file):")
    print("       * the ledger audition calls `audition_macro` (l.4009-4015), whose body is")
    print("         `MC.apply_any` (l.4013 -> `ratchet/macros.py` l.232-246): `macro_features`'")
    print("         max-sum DP over the macro's OWN TABLE, then `canon[feats]` scattered in.")
    print("         The perf executor `ex` is threaded only into `beam_moves*` (l.1576, 1784,")
    print("         4563, 4829, 5194) and is never passed here, so no span head fires and no")
    print("         slot is resolved in either audition.")
    print("       * `e_keep` grades `tbl_m = operative(ml)`; `e_merge` grades `tbl_try =")
    print("         miners[ml].build(operative(ml-1), support)` (l.4317-4326), taken after")
    print("         `quot.merge(ml, ca, cb)` + `miners[ml].rekey()` (l.4319-4320).")
    print("         `ClassMiner.rekey` re-keys with `id_of(x, self.level - 1)` = the level-")
    print("         (ml-1) map, which a level-ml merge does not touch; and `operative(ell)`")
    print("         (l.4524-4532) returns that same `miners[ell].build(...)` expression at any")
    print("         level that is not frozen.  So at a LIVE level `tbl_try` is `tbl_m` row for")
    print("         row and the two auditions grade the identical table; at a FROZEN level they")
    print("         differ only by the live drift since the commit.  The level a level-ml merge")
    print("         does change is ml+1, whose miners are re-keyed in the TAKE branch (l.4367-")
    print("         4370) and which no audition here grades.")
    print("       * `n_entries_after` is `miners[ml].build(...)` after the take (l.4371/4381),")
    print("         the live rebuild at the merged level, not a table the merge re-shaped.")
    for arm, r in list(arms) + extra_arms:
        me = r.get("merge_events") or []
        if not me:
            continue
        props = [e for e in me if e.get("kind") == "merge_proposal"]
        if props:
            grps = [e for e in me if e.get("kind") == "merge"]
            gr_g = sum(e.get("gradings") or 0 for e in props)
            print(f"    {arm:>18} licence={r.get('merge_mode')} proposals={len(props)} "
                  f"groups={len(grps)} taken={sum(1 for e in grps if e.get('taken'))} "
                  f"refused={sum(1 for e in grps if not e.get('taken') and not e.get('skipped'))} "
                  f"skipped={sum(1 for e in grps if e.get('skipped'))} "
                  f"probe gradings={gr_g:,} "
                  f"({gr_g / max(r['log']['t_cum'][-1], 1):.4%} of priced time)")
            print(f"        {'cycle':>6} {'L':>2} {'froz':>5} {'pairs':>6} {'grp':>4} "
                  f"{'took':>4} {'ref':>4} {'skip':>4} {'undef':>5} {'classes':>13} "
                  f"{'T(l+1)':>15}")
            for p in props:
                print(f"        c{p['cycle']:>5} {p['level']:>2} "
                      f"{str(p.get('frozen'))[:5]:>5} {str(p.get('n_pairs')):>6} "
                      f"{p.get('n_groups', 0):>4} {p.get('groups_taken', 0):>4} "
                      f"{p.get('groups_refused', 0):>4} {p.get('groups_skipped', 0):>4} "
                      f"{p.get('groups_undefined', 0):>5} "
                      f"{str(p.get('n_classes_before')) + '->' + str(p.get('n_classes_after')):>13} "
                      f"{str(p.get('n_entries_next_before')) + '->' + str(p.get('n_entries_next_after')):>15}")
            print(f"        --- every GROUP, ALL CURRENCIES and their floors ---")
            print(f"        {'cycle':>6} {'L':>2} {'g':>3} {'n':>3} {'loss':>6} "
                  f"{'mass rise':>10} {'mass flr':>9} {'exp rise':>11} {'exp flr':>9} "
                  f"{'n_rem':>6} {'build':>8} {'entries':>11} {'count':>6} {'yld':>5} "
                  f"{'ldg':>5} {'TAKEN':>5}")
            for e in grps:
                if e.get("skipped"):
                    continue
                _ent = (f"{e.get('entries_before')}->{e.get('entries_after')}"
                        if e.get("entries_before") is not None else "—")
                print(f"        c{e['cycle']:>5} {e['level']:>2} {e.get('group', 0):>3} "
                      f"{e.get('group_size', 0):>3} {e.get('loss', 0):>6.3f} "
                      f"{e.get('mass_rise', 0):>+10.5f} {e.get('mass_floor', 0):>9.5f} "
                      f"{(e.get('exp_rise') if e.get('exp_rise') is not None else 0):>+11.7f} "
                      f"{(e.get('exp_floor') if e.get('exp_floor') is not None else 0):>9.5f} "
                      f"{str(e.get('exp_n_rem')):>6} "
                      f"{(e.get('build_rise') if e.get('build_rise') is not None else 0):>+8.5f} "
                      f"{_ent:>11} {e.get('rise', 0):>+6.1f} "
                      f"{str(e.get('licensed_yield'))[:5]:>5} "
                      f"{str(e.get('licensed_ledger'))[:5]:>5} "
                      f"{str(e.get('taken'))[:5]:>5}")
                if not e.get("taken"):
                    continue
                lv1, cy = int(e["next_level"]), int(e["cycle"])
                ser = (r.get("obs_hist") or {}).get(str(lv1)) or []
                i = r["log"]["cycle"].index(cy) if cy in r["log"]["cycle"] else None
                if i is not None and ser:
                    lp = r["log"]["loop"]
                    nxt = lp[i + 1] if i + 1 < len(lp) else {}
                    print(f"            L{lv1} at-support around it: "
                          f"{ser[max(0, i - 4):i + 1]} | {ser[i + 1:i + 6]}   "
                          f"latch next cycle n_since="
                          f"{nxt.get('c_n_since', nxt.get('n_since'))} "
                          f"moved={nxt.get('c_moved', nxt.get('moved'))}")
                fr = [q for q in r["events"] if q["kind"] == "commit"
                      and q["level"] in (e["level"], lv1) and int(q["cycle"]) >= cy]
                print(f"            froze after it at: "
                      f"{[(q['level'], q['cycle'], q['n_entries']) for q in fr] or 'never'}")
            continue
        cand = [e for e in me if "loss" in e]
        took = [e for e in me if e.get("taken")]
        print(f"    {arm:>18} licence={r.get('merge_mode')} proposals={len(me)} "
              f"with_candidate={len(cand)} taken={len(took)} "
              f"gradings={sum(e.get('gradings') or 0 for e in me):,} "
              f"({sum(e.get('gradings') or 0 for e in me) / max(r['log']['t_cum'][-1], 1):.4%}"
              f" of priced time)")
        for e in cand:
            print(f"        c{e['cycle']:>4} L{e['level']} frozen={e.get('frozen')} "
                  f"rows={e.get('n_rows')} probed={e.get('n_probed_rows')} "
                  f"pairs_scored={e.get('n_pairs')} picks={e.get('n_candidates')} "
                  f"loss={e.get('loss'):.4f} rise={e.get('rise'):+.1f} "
                  f"floor={e.get('floor'):.4f} yield={e.get('licensed_yield')} "
                  f"ledger={e.get('licensed_ledger')} "
                  f"e_keep={e.get('e_keep')} e_merge={e.get('e_merge')} "
                  f"TAKEN={e.get('taken')} -> {e.get('n_entries_after')} entries "
                  f"grade_after={e.get('grade_after')}")
            if not e.get("taken"):
                continue
            # the next level's yield series either side of the merge, the latch, and the freeze
            lv1, cy = int(e["next_level"]), int(e["cycle"])
            ser = (r.get("obs_hist") or {}).get(str(lv1)) or []
            i = r["log"]["cycle"].index(cy) if cy in r["log"]["cycle"] else None
            if i is not None and ser:
                print(f"            L{lv1} at-support around the merge: "
                      f"{ser[max(0, i - 4):i + 1]} | {ser[i + 1:i + 6]}")
                lp = r["log"]["loop"]
                nxt = lp[i + 1] if i + 1 < len(lp) else {}
                print(f"            commit owner's latch on the next cycle: "
                      f"n_since={nxt.get('c_n_since', nxt.get('n_since'))} "
                      f"moved={nxt.get('c_moved', nxt.get('moved'))} "
                      f"(reset by the merge -> n_since 1)")
            fr = [q for q in r["events"] if q["kind"] == "commit"
                  and q["level"] == e["level"] and int(q["cycle"]) >= cy]
            print(f"            level {e['level']} froze after the merge at: "
                  f"{[(q['cycle'], q['n_entries']) for q in fr] or 'never'}")

    # ---- [M2] the rate: what the probe scored, what the op could take, what it took ------ #
    print("\n[M2] THE RATE — what the probe scored, what the op took, and the alias ceiling")
    print("     `n_pairs` is every pair the probe SCORED at that proposal; `groups` is every")
    print("     alias group the closure found and `taken` how many the licence took. There is")
    print("     no selection cap, so nothing is left on the table by the OP — only by the")
    print("     licence.  (FIRST PASS, `en_s2b` and earlier: one pair per proposal, and its")
    print("     `n_candidates` column is `len(picks)` capped at 1, not the size of the offer.)")
    print(f"    {'arm':>18} {'proposals':>10} {'groups':>7} {'taken':>6} {'refused':>8} "
          f"{'empty':>6} {'pairs scored':>13} {'cycles':>7} {'takes/100cyc':>13}")
    for arm, r in list(arms) + extra_arms:
        me = r.get("merge_events") or []
        if not me:
            continue
        props = [e for e in me if e.get("kind") == "merge_proposal"]
        if props:
            grps = [e for e in me if e.get("kind") == "merge" and not e.get("skipped")]
            took = sum(1 for e in grps if e.get("taken"))
            ncyc = len(r["log"]["cycle"])
            print(f"    {arm:>18} {len(props):>10} {len(grps):>7} {took:>6} "
                  f"{len(grps) - took:>8} "
                  f"{sum(1 for p in props if not p.get('n_groups')):>6} "
                  f"{sum(int(p.get('n_pairs') or 0) for p in props):>13,} {ncyc:>7} "
                  f"{100 * took / max(ncyc, 1):>13.2f}")
            continue
        picks = sum(int(e.get("n_candidates") or 0) for e in me)
        took = sum(1 for e in me if e.get("taken"))
        nocand = sum(1 for e in me if not (e.get("n_candidates") or 0)
                     and not e.get("forced"))
        scored = sum(int(e.get("n_pairs") or 0) for e in me)
        ncyc = len(r["log"]["cycle"])
        print(f"    {arm:>18} {len(me):>10} {picks:>6} {took:>6} {len(me) - took:>8} "
              f"{nocand:>8} {scored:>13,} {ncyc:>7} {100 * took / max(ncyc, 1):>13.2f}")

    print("\n     THE ALIAS CEILING AT L2 (offline, exact, and the only level whose ROWS the")
    print("     compact log carries — the keys there are flat pairs).  `sizing/SIZING.md` 5(a):")
    print("     a forced-transfer probe resolves the TOKEN CLASS and nothing else, and at the")
    print("     L2 mining node two class pairs collapse further ({0}=={0,3}, {2}=={7}), so the")
    print("     exact-alias relation on rows is `same demand group`.  `merges available` is")
    print("     rows - groups: the number of merge OPS that would collapse the book onto its")
    print("     own demand partition.  L3/L4 rows are cross-products this section does not")
    print("     rebuild; `alias_audit.py` does rebuild them (exactly, wherever the spelling cap")
    print("     never bound) and carries the same ceiling per proposal at every level.")
    collapse = [frozenset({0}) | frozenset({0, 3}), frozenset({2}) | frozenset({7})]

    def demand_group(c):
        if not c:
            return "DEAD/JUNK"
        for g in collapse:
            if c <= g and c & g:
                return tuple(sorted(g))
        return tuple(sorted(c))

    print(f"    {'arm':>18} {'cycle':>7} {'what':>10} {'rows':>5} {'tok classes':>12} "
          f"{'demand groups':>14} {'alias pairs':>12} {'merges avail':>13} {'L2 taken':>9}")
    for arm, r in list(arms) + extra_arms:
        me2 = [e for e in (r.get("merge_events") or []) if e["level"] == 2]
        if not me2:
            continue
        took2 = sum(1 for e in me2 if e.get("taken"))
        c2 = [e["cycle"] for e in r["events"] if e["kind"] == "commit" and e["level"] == 2]
        props2 = [e for e in (r.get("merge_events") or [])
                  if e.get("kind") == "merge_proposal" and e["level"] == 2]
        # ONE ROW PER PROPOSAL: a whole-partition tag emits several `merge` events per cycle
        # and the book they all read is the same one.
        seen_c, cycs = set(), ([("commit", c2[0])] if c2 else [])
        for e in (props2 or me2):
            if e["cycle"] not in seen_c:
                seen_c.add(e["cycle"])
                cycs.append(("proposal", e["cycle"]))
        for what, cyc in cycs:
            if cyc not in r["log"]["cycle"]:
                continue
            st = (r["log"]["miner"][r["log"]["cycle"].index(cyc)] or {}).get("2") or {}
            ks = st.get("keys_at_support") or []
            rows = [tuple(int(x[0]) for x in k) for k in ks
                    if len(k) == S and all(isinstance(z, list) and len(z) == 1 for z in k)]
            if not rows:
                continue
            cl = tok_classes(rows, 2)
            gp = [demand_group(c) for c in cl]
            npair = sum(1 for i in range(len(rows)) for j in range(i + 1, len(rows))
                        if gp[i] == gp[j])
            print(f"    {arm:>18} {('c' + str(cyc)):>7} {what:>10} {len(rows):>5} "
                  f"{len(set(cl)):>12} {len(set(gp)):>14} {npair:>12} "
                  f"{len(rows) - len(set(gp)):>13} {took2:>9}")

    # ---- [M3] the ledger split by pair type, with the blindness diagnostic beside it ----- #
    print("\n[M3] THE LEDGER AUDITION AT l+1, SPLIT BY PAIR TYPE")
    print("     Type is read off the members' TOKEN CLASSES: `equal` (one class), `subset`")
    print("     (some member's class strictly inside another's), `disjoint` (some pair shares")
    print("     no feature), `overlap` otherwise. `x_differ` is the fraction of audition")
    print("     instances on which the kept and merged tables write a DIFFERENT state and")
    print("     `succ_differ` the fraction on which the grade flips: with `x_differ == 0` an")
    print("     `e_merge == e_keep` is the DP's argmax surviving the merge, not a measurement")
    print("     of it.")

    def group_kind(members, level):
        cs = [c for c in tok_classes([tuple(int(z) for z in x) for x in members], level)]
        cs = [c for c in cs if c]
        if not cs:
            return "junk"
        if len(set(cs)) == 1:
            return "equal"
        for i_ in range(len(cs)):
            for j_ in range(i_ + 1, len(cs)):
                if cs[i_] < cs[j_] or cs[j_] < cs[i_]:
                    return "subset"
        for i_ in range(len(cs)):
            for j_ in range(i_ + 1, len(cs)):
                if not (cs[i_] & cs[j_]):
                    return "disjoint"
        return "overlap"

    for arm, r in list(arms) + extra_arms:
        grps = [e for e in (r.get("merge_events") or [])
                if e.get("kind") == "merge" and e.get("e_keep") is not None]
        undef = [e for e in (r.get("merge_events") or [])
                 if e.get("kind") == "merge" and e.get("ledger_undefined")]
        if not grps and not undef:
            continue
        print(f"    {arm:>18} resolved={len(grps)} undefined={len(undef)} "
              f"{sorted({str(e.get('ledger_undefined')) for e in undef})}")
        if not grps:
            continue
        print(f"        {'type':>9} {'n':>3} {'e_keep':>17} {'e_merge':>17} "
              f"{'delta':>19} {'x_differ':>17} {'succ_differ':>14} {'taken':>6}")
        by = {}
        for e in grps:
            by.setdefault(group_kind(e.get("members") or [], int(e["level"])), []).append(e)
        for k in ("equal", "subset", "disjoint", "overlap", "junk"):
            gs = by.get(k)
            if not gs:
                continue
            ek = [float(e["e_keep"]) for e in gs]
            em = [float(e["e_merge"]) for e in gs]
            dd = [b - a for a, b in zip(ek, em)]
            xd = [e.get("x_differ") for e in gs if e.get("x_differ") is not None]
            sd = [e.get("succ_differ") for e in gs if e.get("succ_differ") is not None]
            def rng(z):
                return f"{np.mean(z):.4f}[{min(z):.4f},{max(z):.4f}]" if z else "—"
            print(f"        {k:>9} {len(gs):>3} {rng(ek):>17} {rng(em):>17} "
                  f"{(f'{np.mean(dd):+.4f}[{min(dd):+.4f},{max(dd):+.4f}]' if dd else '—'):>19} "
                  f"{rng(xd):>17} {rng(sd):>14} {sum(1 for e in gs if e.get('taken')):>6}")
        n0 = sum(1 for e in grps if (e.get("x_differ") or 0) == 0)
        eq = sum(1 for e in grps if abs(float(e["e_merge"]) - float(e["e_keep"])) < 1e-12)
        print(f"        x_differ == 0 on {n0} of {len(grps)} resolved auditions; "
              f"e_merge == e_keep on {eq} of {len(grps)}")
        # [en_s4] THE KEEP-CASE SPLIT. `table` is table-against-table with `fourwall`'s margin;
        # `absent` is the branch this round added — no l+1 table under the current key, so
        # `e_keep` is the same instances with NO l+1 move applied (1.0 exactly, since they are
        # sampled broken) and the licence is STRICT: the merged key's macro has to repair at
        # least one instance. `repaired` counts the takes where it did.
        print(f"        {'keep case':>12} {'n':>4} {'taken':>6} {'e_keep':>17} "
              f"{'e_merge':>17} {'repaired (succ_differ>0)':>25}")
        for case in ("table", "absent"):
            gs = [e for e in grps if e.get("ledger_keep_case") == case]
            if not gs:
                continue
            ek = [float(e["e_keep"]) for e in gs]
            em = [float(e["e_merge"]) for e in gs]
            tk = [e for e in gs if e.get("taken")]
            rep = sum(1 for e in tk if (e.get("succ_differ") or 0) > 0)
            print(f"        {case:>12} {len(gs):>4} {len(tk):>6} "
                  f"{f'{np.mean(ek):.4f}[{min(ek):.4f},{max(ek):.4f}]':>17} "
                  f"{f'{np.mean(em):.4f}[{min(em):.4f},{max(em):.4f}]':>17} "
                  f"{(f'{rep} of {len(tk)} takes' if tk else '—'):>25}")
        _uk = [e for e in grps if e.get("ledger_keep_case") is None]
        if _uk:
            print(f"        {'(no case)':>12} {len(_uk):>4}  — records from a tag written "
                  f"before the keep-case field existed")

    # ---- [M4] merge precision, in both spaces ------------------------------------------- #
    print("\n[M4] MERGE PRECISION — was a taken group actually one class?")
    print("     DEMAND partition (what a forced-transfer probe can resolve): the token class")
    print("     with the node's own collapses from `sizing/SIZING.md` 5(a) applied, and every")
    print("     dead-or-junk row folded into one all-fail cell — that is the ceiling, and a")
    print("     merge inside one demand cell is one no probe could refuse.")
    print("     FEATURE partition (`true_mask`'s space, the arc's historical report): the")
    print("     level's own true parent map, `quotient.py`'s `gen` mode. A row with no true")
    print("     parent has no feature class and is counted separately rather than as a miss.")
    COLL = {2: [frozenset({0, 3}), frozenset({2, 7})],
            3: [frozenset({1, 6})],
            4: [frozenset({2, 7})]}
    DEAD = {2: [], 3: [frozenset({1})], 4: [frozenset({7})]}
    gen_q = QT.Quotient("gen", rules, canon, V, S, DEPTH)

    def demand_cell(row, level):
        c = tok_classes([tuple(int(z) for z in row)], level)
        c = c[0] if c else frozenset()
        if not c or any(c == d for d in DEAD.get(level, [])):
            return "DEAD/JUNK"
        for g in COLL.get(level, []):
            if c <= g and (c & g):
                return tuple(sorted(g))
        return tuple(sorted(c))

    print(f"    {'arm':>18} {'taken':>6} {'DEMAND: one cell':>17} {'precision':>10} "
          f"{'FEATURE: one class':>19} {'precision':>10} {'no feature class':>17}")
    for arm, r in list(arms) + extra_arms:
        took = [e for e in (r.get("merge_events") or [])
                if e.get("kind") == "merge" and e.get("taken") and e.get("members")]
        if not took:
            continue
        d_ok = f_ok = f_na = 0
        for e in took:
            lv = int(e["level"])
            mem = [tuple(int(z) for z in x) for x in e["members"]]
            d_ok += len({demand_cell(x, lv) for x in mem}) == 1
            fc = [gen_q.id_of(x, lv) for x in mem]
            if any(z is None for z in fc):
                f_na += 1
            else:
                f_ok += len(set(fc)) == 1
        n = len(took)
        print(f"    {arm:>18} {n:>6} {d_ok:>17} {d_ok / n:>10.3f} {f_ok:>19} "
              f"{(f_ok / max(n - f_na, 1)):>10.3f} {f_na:>17}")

    # ---- [X] the expansion-choice instrument -------------------------------------------- #
    print("\n[X] THE EXPANSION CHOICE — did the row the DP picked admit what the instance asked")
    print("     Per macro call on the AUDITION DP path (every `audition_macro` the fork makes:")
    print("     the commit audition's candidate/true/random/held/live cells, the shadow oracle")
    print("     audition and the merge ledger's), the chosen row's TOKEN CLASS against the")
    print("     feature the instance actually demands at that (level, node) — the clean")
    print("     derivation's latent there. `contains` is the share of calls whose chosen class")
    print("     holds the demanded feature; `succ` is the share that then graded as a repair.")
    print("     Oracle-contained (every read counted in `exp_reads`) and inert to the run")
    print("     (gate E-7; G-F 0.000e+00 with it on).")
    for arm, r in list(arms) + extra_arms:
        ex = r["log"].get("exp") or []
        if not any(ex):
            continue
        eras_l = r["log"]["era"]
        cells = sorted({k for q in ex if q for k in q})
        print(f"    {arm:>22}  exp_reads={r.get('exp_reads'):,}  cells={cells}")
        print(f"        {'cell':>8} {'era':>4} {'calls':>8} {'contains':>10} {'rate':>7} "
              f"{'succ':>8} {'rate':>7}")
        for cell in cells:
            for era in sorted(set(eras_l)):
                c = s_ = k = 0
                for i, q in enumerate(ex):
                    if not q or eras_l[i] != era or cell not in q:
                        continue
                    k += q[cell]["calls"]
                    c += q[cell]["contains"]
                    s_ += q[cell]["succ"]
                if not k:
                    continue
                print(f"        {cell:>8} {era:>4} {k:>8,} {c:>10,} {c / k:>7.3f} "
                      f"{s_:>8,} {s_ / k:>7.3f}")

    # ---- [Y] the clock yokes ------------------------------------------------------------ #
    if a.yoke_tag:
        print("\n[Y] THE CLOCK YOKES — realised against planned (gate X-5's check, post hoc)")
        yroot = os.path.join(FIG, a.yoke_tag)
        src_of = {"flat_yk_tok": "given_cat_tok", "flat_yk_min": "given_cat_min",
                  "flat_yk_endo_yield": "endo_yield",
                  "flat_yk_endo_ledger": "endo_ledger",
                  "flat_yk_endo_force": "endo_yield_force",
                  # [en_s5] `figured_bass`'s open-inventory arm is a yoke of `given_cat_tok`,
                  # so it belongs in this table too — same clock, one bit different.
                  "given_cat_tok_open_yk": "given_cat_tok",
                  # [en_s6] the composed arms
                  "endo_ledger_open_ung5_yk": "endo_ledger",
                  "flat_yk_endo_ledger_open_ung5": "endo_ledger_open_ung5",
                  "given_cat_tok_open_ung5_yk": "given_cat_tok"}
        srcs = dict(arms)
        for yarm in sorted(os.listdir(yroot)):
            yp = os.path.join(yroot, yarm, "results.json")
            if not os.path.isfile(yp):
                continue
            yr = json.load(open(yp))
            src = src_of.get(yarm)
            sr = srcs.get(src)
            if sr is None:
                print(f"    {yarm:>18} source {src!r} not in --tag {a.tag}")
                continue
            plan = [(x["kind"], x["cycle"]) for x in (sr.get("loop_actions") or [])
                    if not x.get("cancelled")]
            got = [(x["kind"], x["cycle"]) for x in (yr.get("loop_actions") or [])
                   if not x.get("cancelled")]
            canc = [(x["kind"], x["cycle"], x.get("cancelled"), x.get("level"))
                    for x in (yr.get("loop_actions") or []) if x.get("cancelled")]
            pc = sorted(c for k, c in plan if k == "commit")
            gc = sorted(c for k, c in got if k == "commit")
            pa = sorted(c for k, c in plan if k == "advance")
            ga = sorted(c for k, c in got if k == "advance")
            print(f"    {yarm:>18} <- {src}")
            print(f"        advances  planned {pa}")
            print(f"                  realised {ga}   MATCH={pa == ga}")
            print(f"        commits   planned {pc}")
            print(f"                  realised {gc}   MATCH={pc == gc}")
            print(f"        cancelled (empty flat build): {canc}")
            print(f"        lifetime  source {len(sr['log']['cycle'])} cycles / "
                  f"{sr['log']['t_cum'][-1]:,.0f}g   yoke {len(yr['log']['cycle'])} cycles / "
                  f"{yr['log']['t_cum'][-1]:,.0f}g")
            ycom = [(e['level'], e['cycle'], e['n_entries']) for e in yr["events"]
                    if e["kind"] == "commit"]
            scom = [(e['level'], e['cycle'], e['n_entries']) for e in sr["events"]
                    if e["kind"] == "commit"]
            print(f"        installed source {scom}")
            print(f"                  yoke   {ycom}")
            for era in (4, 5):
                yi = [i for i, x in enumerate(yr["log"]["era"]) if x == era]
                si = [i for i, x in enumerate(sr["log"]["era"]) if x == era]
                if yi and si:
                    ye = float(np.mean([yr["log"]["e"][i] for i in yi]))
                    se = float(np.mean([sr["log"]["e"][i] for i in si]))
                    print(f"        era{era}: yoke e {ye:.4f} ({len(yi)} cyc) vs source "
                          f"{se:.4f} ({len(si)} cyc)   delta {se - ye:+.4f}")

    # ---- [H] the bill ------------------------------------------------------------------- #
    print("\n[H] THE BILL")
    print(f"    {'arm':>16} {'t_cum (g)':>12} {'oracle reads':>13} {'slot record cycles':>19} "
          f"{'slot mass':>12} {'merge events':>13}")
    for arm, r in arms:
        sl = [q for q in r["log"]["slot"] if q]
        mass = sum(sum(vv) for q in sl for byn in (q.get("beam") or {}).values()
                   for vv in byn.values())
        print(f"    {arm:>16} {r['log']['t_cum'][-1]:>12,.0f} "
              f"{str((r.get('quotient') or {}).get('n_read')):>13} {len(sl):>19} "
              f"{mass:>12,} {len(r.get('merge_events') or []):>13}")

    # ---- [O] [figured_bass] the open inventory, per cycle past the commit ---------------- #
    #      Reads `log["open"]`, which every `fb_*` arm carries and no banked `en_*` arm does,
    #      so a banked anchor prints one line saying so rather than being silently omitted.
    #      `operative` vs `committed` is the executor's exposure ([4] of
    #      `figured_bass/sizing/SIZING.md`) measured in run instead of replayed; `lower cls`
    #      is the level-(l-1) book's class coverage at the L5 lookup, i.e. [1]'s object.
    print("\n[O] THE OPEN INVENTORY — operative vs committed, per cycle past the L4 commit")
    print("    `committed` = the rows the commit froze.  `operative` = the rows the DP")
    print("    actually maxed over that cycle.  With the open bit OFF they are equal by")
    print("    construction and the arm prints one row.  `lower cls` is the number of L4")
    print("    classes the operative L4 book holds a row of, out of the 13-class token")
    print("    alphabet — the constraint the whole node is about.")
    for arm, r in arms + extra_arms:
        op = r["log"].get("open") or []
        if not any(op):
            print(f"    {arm:>18} (no `log[\"open\"]` — banked before the open bit; see [G]/[G2])")
            continue
        ev4 = [e for e in r["events"] if e["kind"] == "commit" and e["level"] == 4]
        c4 = int(ev4[0]["cycle"]) if ev4 else None
        st = (op[-1] or {}).get("stat") or {}
        print(f"\n    --- {arm}  open_inventory={r.get('open_inventory')} "
              f"ungate_l5={r.get('ungate_l5')}  L4 commit "
              f"{('c' + str(c4)) if c4 else '(none)'}  "
              f"rebuilds={st.get('n_rebuild')} moved={st.get('n_cycles_moved')} "
              f"empty_fallback={st.get('n_empty_fallback')} ---")
        print(f"        {'cycle':>6} " + " ".join(f"{'L' + str(l) + ' com/op':>14}"
                                                  for l in (2, 3, 4, 5))
              + f" {'L4 lower cls':>13} {'of':>4}")
        prev = None
        for i, cy in enumerate(r["log"]["cycle"]):
            q = op[i] if i < len(op) else None
            if not q or (c4 is not None and cy < c4):
                continue
            b5 = (q.get("blocked") or {}).get("5") or {}
            sig = tuple((q["committed"].get(str(l)), q["operative"].get(str(l)))
                        for l in (2, 3, 4, 5)) + (b5.get("n_lower_classes"),)
            if prev is not None and sig == prev and i != len(r["log"]["cycle"]) - 1:
                continue
            prev = sig
            cells = " ".join(
                f"{str(q['committed'].get(str(l))) + '/' + str(q['operative'].get(str(l))):>14}"
                for l in (2, 3, 4, 5))
            print(f"        {cy:>6} {cells} {str(b5.get('n_lower_classes')):>13} "
                  f"{clos[4]['n_classes']:>4}")

    # ---- [O2] [figured_bass] the keys the operative book blocks -------------------------- #
    print("\n[O2] THE RATCHET'S BITE — level-l keys at support the operative level-(l-1) book")
    print("     cannot look up.  This is what an open inventory is FOR: `figured_bass`'s")
    print("     offline replay found the frozen L4 book blocking 5 of `given_cat_tok`'s 7 L5")
    print("     keys at support at the end of `en_s0`, and the live one blocking 2.")
    print(f"     {'arm':>18} {'level':>6} {'cycle':>7} {'keys@sup':>9} {'blocked':>8} "
          f"{'lower cls':>10} {'lower rows':>11}")
    for arm, r in arms + extra_arms:
        op = r["log"].get("open") or []
        if not any(op):
            continue
        ev4 = [e for e in r["events"] if e["kind"] == "commit" and e["level"] == 4]
        c4 = int(ev4[0]["cycle"]) if ev4 else r["log"]["cycle"][0]
        idx = [i for i, cy in enumerate(r["log"]["cycle"]) if cy >= c4 and i < len(op) and op[i]]
        if not idx:
            continue
        for lv in ("4", "5"):
            for label, i in (("commit", idx[0]), ("median", idx[len(idx) // 2]),
                             ("last", idx[-1])):
                b = ((op[i] or {}).get("blocked") or {}).get(lv)
                if not b:
                    continue
                print(f"     {arm:>18} {('L' + lv):>6} {r['log']['cycle'][i]:>7} "
                      f"{b['n_at_support']:>9} {b['n_blocked']:>8} "
                      f"{b['n_lower_classes']:>10} {b['n_lower_rows']:>11}"
                      f"   ({label})")

    # ---- [O3] [figured_bass] the corridor across the commit ------------------------------ #
    #      The SPEC's open question in one table: can an executor adopt a level whose content
    #      keeps moving under it?  Parity, misfire rate and slot open/close churn in a window
    #      either side of the L4 commit, for every arm — so an open arm is read against its
    #      own frozen anchor on the same clock rather than against a prior.
    W = 20
    print(f"\n[O3] THE CORRIDOR ACROSS THE COMMIT — {W} cycles either side of the L4 commit")
    print("     `parity` = mean over slots of `log[\"span\"][c][\"parity\"]` (exact-match rate")
    print("     of the head's execution against the DP's intention).  `misfire` =")
    print("     n_misfire / max(n_fired, 1) off the meter's row.  `churn` = cycles on which")
    print("     the set of OPEN slots changed.  `n_open` = mean open slots.")
    print("     WHAT THIS WINDOW CAN AND CANNOT SAY ABOUT THE OPEN BIT. An arm that never")
    print("     committed L4 is anchored on its L2 commit, and in `fb_s0` the open arms are")
    print("     BIT-IDENTICAL to their frozen twins through c31 — the operative table first")
    print("     exceeds the committed one at c31 and `e` first differs at c32 — so a window")
    print("     that ends at c31 (L2@c12 + 20) is entirely inside the identical stretch. Rows")
    print("     that agree to the digit there are reporting that fact and NOTHING about the")
    print("     open bit. Read the per-slot b(s) series instead, and see the note in [C].")
    print("     THE ANCHOR EVENT is the L4 commit where there is one and the arm's FIRST")
    print("     commit otherwise, named in the `at` column — an arm that never committed L4")
    print("     still adopted a level, and 'can the executor adopt a level whose content keeps")
    print("     moving under it' is asked of whatever level it did adopt.")
    print(f"     {'arm':>18} {'at':>8} {'window':>7} {'cyc':>5} {'parity':>8} {'misfire':>8} "
          f"{'churn':>6} {'n_open':>7} {'e':>8}")
    for arm, r in arms + extra_arms:
        cms = sorted((int(e["cycle"]), int(e["level"])) for e in r["events"]
                     if e["kind"] == "commit")
        if not cms:
            continue
        ev4 = [c for c, lv in cms if lv == 4]
        c4 = ev4[0] if ev4 else cms[0][0]
        at = f"L{dict((c, lv) for c, lv in cms)[c4]}@c{c4}"
        perf_by_c = {int(q["c"]): q for q in (r["log"].get("perf") or []) if q and "c" in q}
        for label, lo, hi in (("before", c4 - W, c4 - 1), ("after", c4, c4 + W - 1)):
            ix = [i for i, cy in enumerate(r["log"]["cycle"]) if lo <= cy <= hi]
            if not ix:
                continue
            par, mis, opn, chn, prev_open = [], [], [], 0, None
            for i in ix:
                sp = (r["log"]["span"][i] or {}) if i < len(r["log"]["span"]) else {}
                pv = [x for x in (sp.get("parity") or {}).values() if x is not None]
                if pv:
                    par.append(float(np.mean(pv)))
                om = sp.get("open") or {}
                if om:
                    opn.append(sum(1 for x in om.values() if x))
                    cur = frozenset(k for k, x in om.items() if x)
                    if prev_open is not None and cur != prev_open:
                        chn += 1
                    prev_open = cur
                pq = perf_by_c.get(int(r["log"]["cycle"][i]))
                if pq and int(pq.get("n_fired") or 0) > 0:
                    mis.append(int(pq.get("n_misfire") or 0) / int(pq["n_fired"]))
            ee = float(np.mean([r["log"]["e"][i] for i in ix]))
            print(f"     {arm:>18} {at:>8} {label:>7} {len(ix):>5} "
                  f"{(f'{np.mean(par):.4f}' if par else '—'):>8} "
                  f"{(f'{np.mean(mis):.4f}' if mis else '—'):>8} "
                  f"{chn:>6} {(f'{np.mean(opn):.2f}' if opn else '—'):>7} {ee:>8.4f}")

    # ---- [Y2] [figured_bass] the yoke replay check, from the plan the arm was HANDED ------ #
    #      [Y] above needs `--yoke-tag` and a hard-coded source map, so it cannot check a yoke
    #      whose source is BANKED in another tag (`--yoke-from-tag`, which `fb_s1` uses). This
    #      one needs neither: `config["yoke_plan"]` is the exact plan the arm received, so
    #      "planned == realised" is a comparison against the arm's own record and works for a
    #      same-tag, cross-tag or banked source alike. Cancellations are REPORTED, never
    #      asserted away — a replayed commit whose live build is empty is a legitimate outcome
    #      of the open bit and is the thing most worth seeing.
    print("\n[Y2] THE YOKE REPLAY — realised against the plan the arm was handed")
    print("     `plan` is `config[\"yoke_plan\"]`, i.e. what the source's realised actions were")
    print("     at hand-off; `realised` is this arm's own uncancelled `loop_actions`.")
    any_yoke = False
    for arm, r in arms + extra_arms:
        plan = (r.get("config") or {}).get("yoke_plan")
        if not plan:
            continue
        any_yoke = True
        if isinstance(plan, str):
            plan = json.loads(plan)
        acts = r.get("loop_actions") or []
        pl = sorted((x["kind"], int(x["cycle"])) for x in plan)
        got = sorted((x["kind"], int(x["cycle"])) for x in acts if not x.get("cancelled"))
        canc = [(x["kind"], int(x["cycle"]), x.get("level"), x.get("cancelled"))
                for x in acts if x.get("cancelled")]
        miss = [x for x in pl if x not in got]
        extra = [x for x in got if x not in pl]
        print(f"\n     {arm}  (open_inventory={r.get('open_inventory')} "
              f"ungate_l5={r.get('ungate_l5')}, {len(r['log']['cycle'])} cycles)")
        print(f"        plan     ({len(pl):>2}): {pl}")
        print(f"        realised ({len(got):>2}): {got}")
        print(f"        MATCH={pl == got}"
              + (f"   missing={miss}" if miss else "")
              + (f"   unplanned={extra}" if extra else ""))
        print(f"        cancelled (reported, not asserted): {canc if canc else 'none'}")
        # what a replayed COMMIT actually installed, and what the level held afterwards
        for e in r["events"]:
            if e.get("kind") != "commit":
                continue
            op = [q for q in (r["log"].get("open") or []) if q]
            i = r["log"]["cycle"].index(int(e["cycle"]))
            after = [q["operative"].get(str(e["level"])) for q in op[i:]] if op else []
            print(f"        L{e['level']}@c{e['cycle']}: installed {e['n_entries']} rows"
                  f" (live build={e.get('installed_is_live_build')},"
                  f" by_clock={e.get('by_clock')},"
                  f" open={e.get('open_inventory')})"
                  + (f"; operative after: {after[0]} -> {after[-1]} over {len(after)} cycles"
                     if after else ""))
    if not any_yoke:
        print("     (no arm in this tag carries a yoke plan)")


if __name__ == "__main__":
    main()
