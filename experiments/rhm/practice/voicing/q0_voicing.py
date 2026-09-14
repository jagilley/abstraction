"""[voicing] Q0 — offline, CPU, no substrate: what the banked `en_s9` / `en_s8` logs already
say about the chooser, before a line of the fork is written.

Five questions, in the SPEC's order:

  [V1] VOLUME — macro calls and captured rows per slot per cycle, against the self-imitation
       buffer's own caps, and against every join to the verdict the beam makes available. The
       own-attempt route's row count, before it is built.
  [V2] THE DP'S CLASS ACCURACY TODAY — the expansion-choice instrument per (level, node, era),
       which is the number the head has to beat, plus where the instrument does and does not
       look.
  [V3] RECORD vs RECALL — the inverse map's collisions on this draw, propagated to the level-l
       tuples the books actually hold, in both class spaces (the token class and the arm's own
       learned one). Sizes the Q1 contrast.
  [V4] THE L5 BOOK — classes against spellings: what execution chooses among, what the build
       needs, and where `quot_spell_cap` binds.
  [V5] THE OFFLINE CEILING — what can and cannot be fitted with no GPU, stated with the
       reason.

Everything here is a pure function of the DGP (`rule_seed 0`), the banked per-cycle logs and
the logged `picks`, so it runs with numpy alone. The operative book at any logged build is
rebuilt EXACTLY (gate VQ-1 below asserts the row count against the run's own `n_entries` at
every level of every logged build), which `alias_audit.py`'s approximation could not do: it
re-ranks the spelling cap by index, and this reads the run's own `picks`.

Usage (from experiments/):
    python3 rhm/practice/voicing/q0_voicing.py                       # en_s9, all sections
    python3 rhm/practice/voicing/q0_voicing.py --tag en_s8 --arm endo_ledger
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.practice.ratchet import macros as MC
from rhm.practice.enharmonic import quotient as QT
from rhm.practice.enharmonic import merge as MG

HERE = os.path.dirname(os.path.abspath(__file__))
EN_FIG = os.path.join(os.path.dirname(HERE), "enharmonic", "figures")
V, S, DEPTH, M = 8, 2, 6, 2


# --------------------------------------------------------------------------------------- #
# the tag
# --------------------------------------------------------------------------------------- #
class Arm:
    def __init__(self, tag, arm):
        self.tag, self.arm = tag, arm
        p = os.path.join(EN_FIG, tag, arm, "results.json")
        self.r = json.load(open(p))
        self.log = self.r["log"]
        self.cfg = self.r["config"]
        self.cyc = self.log["cycle"]
        self.n = len(self.cyc)
        self.sup = int(self.cfg["mine_support"])
        self.commits = {int(e["level"]): int(e["cycle"]) for e in self.r["events"]
                        if e["kind"] == "commit"}
        self.took = [e for e in self.r.get("merge_events", [])
                     if e.get("kind") == "merge" and e.get("taken")]
        eb = os.path.join(EN_FIG, tag, arm, "entry_beam.json")
        self.beam_rec = json.load(open(eb)) if os.path.isfile(eb) else None
        # [voicing] the open bit decides whether an ADOPTED level's operative table is the
        # live build or the one its commit froze — `enharmonic.py::operative`. The rebuild
        # has to follow it or the lower rows are the wrong book (gate VQ-1 catches it).
        st = self.r.get("open_stat") or {}
        self.open_inv = bool(st.get("open_inventory", self.r.get("open_inventory")))
        self._book_cache = {}

    # -- the arm's own class map, replayed ------------------------------------------------ #
    def quot_at(self, upto):
        q = MG.LearnedQuotient()
        for e in self.took:
            if e["cycle"] >= upto:
                continue
            q.merge_group(int(e["level"]), [tuple(int(z) for z in x) for x in e["members"]])
        return q

    # -- the operative book, EXACTLY, from the logged picks ------------------------------- #
    def keys_at(self, i, lv):
        st = (self.log["miner"][i] or {}).get(str(lv)) or {}
        return [tuple(tuple(int(z) for z in h) for h in k)
                for k in (st.get("keys_at_support") or [])]

    def build_of(self, i, lv):
        return ((self.log["quot"][i] or {}).get("build", {}) or {}).get(str(lv)) or {}

    def operative_index(self, i, lv):
        """The log index whose level-`lv` build IS the arm's OPERATIVE table at index `i`.

        Open arm: the live build at `i` (that is what `open_inventory` means). Closed arm at
        an adopted level: the build at that level's commit cycle, which is what `committed[lv]`
        froze. Distinct from `book(i, lv)`, which is always the MINER's live build at `i` —
        which is what `log["quot"][i]["build"]` records, because `last_build` is set by the
        gauge path's own call and that one builds live at every level."""
        c = self.commits.get(lv)
        if self.open_inv or c is None or c > self.cyc[i]:
            return i
        j = self.cyc.index(c)
        while j > 0 and not self.build_of(j, lv).get("picks"):
            j -= 1
        return j

    def lower_rows(self, i, lv):
        """`ClassMiner.build`'s `lower_flat`: the width-filtered, dedup'd rows of level lv-1,
        in the run's own order.

        WHICH lower table is not a constant. `log["quot"][i]["build"]` is the miner's
        `last_build`, and more than one call can set it in a cycle: the gauge path builds LIVE
        over LIVE at every level, while `operative(lv-1)` hands a closed arm the table its
        commit FROZE. The build's own `n_lower_rows` says which one it was, so the chain is
        selected by that number rather than assumed — and gate VQ-1 then asserts the row count
        the choice produces. On an open arm the two candidates coincide."""
        want = S ** (lv - 2)

        def prep(rows):
            lf = [t for t in rows if len(t) == want]
            return list({t: None for t in lf})      # last-writer-wins dedup, insertion order

        if lv == 2:
            return prep([tuple(int(x) for x in row) for row in MC.base_table(V)["flat"]])
        want_n = self.build_of(i, lv).get("n_lower_rows")
        cands = [prep(self.book(i, lv - 1))]
        j = self.operative_index(i, lv - 1)
        if j != i:
            cands.append(prep(self.book(j, lv - 1)))
        for lf in cands:
            if want_n is None or len(lf) == want_n:
                return lf
        return cands[0]

    def book(self, i, lv):
        """The exact rows of the level-`lv` build logged at index `i`. Empty where the build
        did not run that cycle."""
        if (i, lv) in self._book_cache:
            return self._book_cache[(i, lv)]
        b = self.build_of(i, lv)
        if not b or not b.get("picks"):
            return []
        lf = self.lower_rows(i, lv)
        picks = {k: [int(x) for x in v] for k, v in b["picks"].items()}
        rows = []
        for key in sorted(self.keys_at(i, lv)):
            sel = []
            for h in key:
                ix = picks.get(str(h))
                if not ix:
                    sel = None
                    break
                sel.append(ix)
            if sel is None:
                continue
            combos = [[]]
            for ix in sel:
                combos = [pre + [j] for pre in combos for j in ix]
            for cb in combos:
                t = ()
                for j in cb:
                    if j >= len(lf):
                        t = None
                        break
                    t = t + lf[j]
                if t is not None:
                    rows.append(t)
        self._book_cache[(i, lv)] = rows
        return rows

    def book_keyed(self, i, lv):
        """Same rows, paired with the at-support key (the class pair) each came from."""
        b = self.build_of(i, lv)
        if not b or not b.get("picks"):
            return []
        lf = self.lower_rows(i, lv)
        picks = {k: [int(x) for x in v] for k, v in b["picks"].items()}
        out = []
        _ = None
        for key in sorted(self.keys_at(i, lv)):
            sel = []
            for h in key:
                ix = picks.get(str(h))
                if not ix:
                    sel = None
                    break
                sel.append(ix)
            if sel is None:
                continue
            combos = [[]]
            for ix in sel:
                combos = [pre + [j] for pre in combos for j in ix]
            for cb in combos:
                t = ()
                for j in cb:
                    if j >= len(lf):
                        t = None
                        break
                    t = t + lf[j]
                if t is not None:
                    out.append((key, t))
        return out


def era_spans(a):
    """log index -> era, and the era boundaries."""
    return a.log["era"]


def fmt(x, n=3):
    return "None" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


# --------------------------------------------------------------------------------------- #
# GATE VQ-1 — the rebuild is the run's own book
# --------------------------------------------------------------------------------------- #
def gate_vq1(a, out):
    bad, n_chk = [], 0
    for i in range(a.n):
        for lv in (2, 3, 4, 5, 6):
            b = a.build_of(i, lv)
            if not b or not b.get("picks"):
                continue
            n_chk += 1
            got = len(a.book(i, lv))
            if got != int(b["n_entries"]):
                bad.append((a.cyc[i], lv, got, int(b["n_entries"])))
    out(f"[VQ-1] exact-rebuild check over {n_chk} logged builds "
        f"({a.tag}:{a.arm}): {'PASS' if not bad else 'FAIL'}")
    if bad:
        for z in bad[:10]:
            out(f"        c{z[0]} L{z[1]}: rebuilt {z[2]} vs logged {z[3]}")
    assert not bad, "VQ-1 FAILED: the rebuilt book is not the run's own"
    return n_chk


# --------------------------------------------------------------------------------------- #
# [V1] VOLUME
# --------------------------------------------------------------------------------------- #
def section_v1(a, out):
    cfg = a.cfg
    out("")
    out("=" * 100)
    out(f"[V1] VOLUME — the own-attempt route's rows, before it is built   ({a.tag}:{a.arm})")
    out("=" * 100)
    out(f"  knobs of record: span_capture(per_call)={cfg['span_capture']}  "
        f"span_buf_cap={cfg['span_buf_cap']}  span_hold_cap={cfg['span_hold_cap']}  "
        f"span_hold_frac={cfg['span_hold_frac']}")
    out(f"                   span_batch={cfg['span_batch']}  gen_steps={cfg['gen_steps']}  "
        f"budget={cfg['budget']}  n_pr={cfg['n_pr']}  n_aud={cfg['n_aud']}")
    out(f"  head consumption per OPEN slot per cycle = span_batch x gen_steps = "
        f"{int(cfg['span_batch']) * int(cfg['gen_steps'])} row-draws (with replacement), "
        f"against a buffer capped at {cfg['span_buf_cap']}.")
    out(f"  `span_train_terms` skips a slot with fewer than 8 buffered rows.")
    out("")

    era = era_spans(a)
    eras = sorted(set(era))
    # (a) macro rows materialised vs rows captured
    out("  (a) per cycle, averaged within era. `mat_*` are the executor's own row counts;")
    out("      `captured` is D(buf+hold) summed over slots, exact only while no cap binds.")
    out("      `cap_slots` counts slots whose buffer is already at `span_buf_cap`.")
    out("")
    out("      era  cycles   mat_base    mat_dp  mat_head   macro_rows  captured  cap_slots"
        "  n_open")
    prev = None
    for e in eras:
        idx = [i for i in range(a.n) if era[i] == e]
        mb = np.mean([a.log["blocks"][i]["mat_base"] for i in idx])
        md = np.mean([a.log["blocks"][i]["mat_dp"] for i in idx])
        mh = np.mean([a.log["blocks"][i]["mat_head"] for i in idx])
        caps, nop, capt = [], [], []
        for i in idx:
            sp = a.log["span"][i]
            tot = sum(sp["buf"].values()) + sum(sp["hold"].values())
            if i > 0:
                sp0 = a.log["span"][i - 1]
                t0 = sum(sp0["buf"].values()) + sum(sp0["hold"].values())
                capt.append(tot - t0)
            caps.append(sum(1 for k, z in sp["buf"].items()
                            if z >= int(cfg["span_buf_cap"])))
            nop.append(sp["n_open"])
        out(f"      {e:>3}  {len(idx):>6}  {mb:>9.0f} {md:>9.0f} {mh:>9.0f}   "
            f"{md + mh:>10.0f}  {np.mean(capt) if capt else 0:>8.0f}  {np.mean(caps):>9.1f}"
            f"  {np.mean(nop):>6.1f}")
        prev = e
    out("")

    # (b) the per-slot capture rate, measured where no cap binds
    out("  (b) THE CAPTURE RATE PER SLOT PER CYCLE, measured on slots below cap. Every macro")
    out("      call stores at most `per_call` rows of the batch*width beam block, open or")
    out("      closed, in EVERY phase (practice beam, metering beam, probes, auditions) —")
    out("      `SpanExecutor.apply` captures before it branches on `open`.")
    out("")
    rates = collections.defaultdict(list)
    for i in range(1, a.n):
        sp, sp0 = a.log["span"][i], a.log["span"][i - 1]
        for k in sp["buf"]:
            if k not in sp0["buf"]:
                continue
            d = (sp["buf"][k] + sp["hold"].get(k, 0)) - (sp0["buf"][k] + sp0["hold"].get(k, 0))
            if sp["buf"][k] < int(cfg["span_buf_cap"]) and d > 0:
                rates[k.split(":")[0]].append(d)
    out("      level   n_obs   median rows/slot/cycle   -> implied macro calls/slot/cycle")
    for lv in sorted(rates):
        v_ = np.array(rates[lv])
        out(f"      L{lv}     {len(v_):>5}   {np.median(v_):>21.0f}   "
            f"{np.median(v_) / int(cfg['span_capture']):>32.1f}")
    out("")
    out("      => the buffer is ALREADY a subsample: the beam block is n_pr x width rows and")
    out("         at most `per_call` of them are kept per call.")
    out("")

    # (c) the verdict's own rates: what a solved-only filter keeps
    out("  (c) WHAT A SOLVED-ONLY FILTER KEEPS. Three joins the beam makes available, with")
    out("      the rate each would retain, per era:")
    out("        tip      `succ > 0.5` per (instance, beam slot) — the verdict the plant's own")
    out("                 `solved` already uses (`finetune_generator`'s diet).")
    out("        answer   `ps > 0.5` at the tip the beam actually answered with, per instance.")
    out("        any-tip  at least one of the instance's W tips solved (an instance-level bag,")
    out("                 `embouchure`'s own coarse label one organ up) — NOT logged per")
    out("                 instance in this tag; bounded below by `answer` and above by `tip`.")
    out("")
    out("      era  n_sol_tip/n_tip   n_sol_inst/n_pr   (= tip rate, answer rate)")
    for e in eras:
        idx = [i for i in range(a.n) if era[i] == e]
        tr = np.mean([a.log["gate"][i]["n_sol_tip"] / max(
            a.log["perf"][i]["cells"]["n_tip"], 1) for i in idx])
        ar = np.mean([a.log["gate"][i]["n_sol_inst"] / max(int(a.cfg["n_pr"]), 1)
                      for i in idx])
        out(f"      {e:>3}  {tr:>15.3f}   {ar:>15.3f}")
    out("")

    # (d) the row count that results
    out("  (d) THE ROW COUNT THE OWN-ATTEMPT ROUTE WOULD HAVE, per slot, per cycle:")
    out("      captured x P(solved). At the median capture rate above and the era-4 tip rate,")
    out("      an L2-L4 slot's SOLVED buffer at steady state is `span_buf_cap` x P(solved).")
    for e in eras:
        idx = [i for i in range(a.n) if era[i] == e]
        tr = np.mean([a.log["gate"][i]["n_sol_tip"] / max(
            a.log["perf"][i]["cells"]["n_tip"], 1) for i in idx])
        med = np.median(rates.get("4", rates.get("3", [0])))
        out(f"      era {e}: tip rate {tr:.3f} -> ~{med * tr:>6.0f} solved rows/slot/cycle, "
            f"steady-state solved buffer ~{int(cfg['span_buf_cap']) * tr:>6.0f} "
            f"(consumption {int(cfg['span_batch']) * int(cfg['gen_steps'])}/cycle)")
    out("")

    # (f) WHO CAPTURES, AND WHAT A TIGHTER JOIN COSTS
    out("  (f) THE TWO PRICED BEAMS, AND THE JOIN LADDER. `ex.capture` is on in exactly two")
    out("      places per cycle — the practice beam (`n_pr` instances, per-TIP `succ`) and the")
    out("      metering beam (`n_rt` instances, grade at the ANSWER only). The battery and")
    out("      every probe set `capture = False`, so a captured row always has SOME verdict.")
    out(f"      Each contributes budget x per_call = {int(cfg['budget']) * int(cfg['span_capture'])}"
        f" rows per slot per cycle, which is the 384 the L5 slot shows on its first cycle.")
    out("")
    out("      The beam expands EVERY move at EVERY step and keeps `width` children of")
    out("      `width x n_moves` candidates, so 1 - 1/n_moves of all writes are discarded by")
    out("      the value head before any verdict exists. Four joins, with the rows each leaves:")
    out("")
    out("      era  n_moves  width   writes/cyc   kept writes   on solved tips   macro share"
        "   macro rows on solved tips")
    for e in eras:
        idx = [i for i in range(a.n) if era[i] == e]
        nm = np.mean([a.log["n_moves"][i] for i in idx])
        w = np.mean([a.log["width"][i] for i in idx])
        tr = np.mean([a.log["gate"][i]["n_sol_tip"] / max(
            a.log["perf"][i]["cells"]["n_tip"], 1) for i in idx])
        mac = np.mean([(a.log["blocks"][i]["mat_dp"] + a.log["blocks"][i]["mat_head"]) /
                       max(a.log["blocks"][i]["mat_dp"] + a.log["blocks"][i]["mat_head"] +
                           a.log["blocks"][i]["mat_base"], 1) for i in idx])
        n_pr = int(cfg["n_pr"])
        writes = n_pr * w * nm * int(cfg["budget"])
        kept = n_pr * w * int(cfg["budget"])
        out(f"      {e:>3}  {nm:>7.0f}  {w:>5.1f}   {writes:>10.0f}   {kept:>11.0f}   "
            f"{kept * tr:>14.0f}   {mac:>11.3f}   {kept * tr * mac:>24.0f}")
    out("")
    out("      The last column is the PRACTICE beam only, summed over all 30 slots. Divided by")
    out("      the slots that are open it is the per-slot row count a kept-and-solved join")
    out("      would deliver; compare it with (b)'s capture rate and with the 1280 draws the")
    out("      head consumes per slot per cycle.")
    out("")
    out("      WHO IS THE CHOOSER, by era — `mat_dp` is the DP's rows, `mat_head` the corridor")
    out("      head's. `enharmonic` finding 5 names the DP; by era 3 the head writes almost")
    out("      all of it, and the head is trained to imitate the DP.")
    out("      era   mat_dp   mat_head   head share of macro rows")
    for e in eras:
        idx = [i for i in range(a.n) if era[i] == e]
        md = np.mean([a.log["blocks"][i]["mat_dp"] for i in idx])
        mh = np.mean([a.log["blocks"][i]["mat_head"] for i in idx])
        out(f"      {e:>3}  {md:>8.0f}  {mh:>9.0f}   {mh / max(md + mh, 1):>24.3f}")
    out("")
    out("      AND HOW FAR THE HEAD IS FROM THE DP IT IMITATES — `n_misfire` counts fired rows")
    out("      whose span did not match `dp_features` exactly, `n_fired` the rows it wrote.")
    out("      era   n_fired   n_misfire   misfire rate")
    for e in eras:
        idx = [i for i in range(a.n) if era[i] == e]
        nf = np.sum([a.log["blocks"][i]["n_fired"] for i in idx])
        nm_ = np.sum([a.log["blocks"][i]["n_misfire"] for i in idx])
        out(f"      {e:>3}  {nf:>9.0f}  {nm_:>10.0f}   {nm_ / max(nf, 1):>12.4f}")
    out("")

    # (e) the L5 slot's own life
    out("  (e) THE L5 SLOT'S WHOLE LIFE — the level the node is about. `buf`/`hold` are that")
    out("      slot's own counters; `open` is the parity gate; `sacc` is the head's")
    out("      self-imitation train accuracy against the DP on its own buffer.")
    out("")
    out("      cyc era  buf5:0  buf5:1 hold5:0 hold5:1 open5:0 open5:1 par5:0 par5:1 "
        "sacc5:0 sacc5:1")
    for i in range(a.n):
        sp = a.log["span"][i]
        if "5:0" not in sp["buf"] and "5:1" not in sp["buf"] and "5:0" not in sp["open"]:
            continue
        g = lambda d, k: d.get(k)
        pv = lambda x: "  -  " if x is None else f"{x:.3f}"
        out(f"      {a.cyc[i]:>3} {era[i]:>3}  {str(g(sp['buf'], '5:0')):>6} "
            f"{str(g(sp['buf'], '5:1')):>7} {str(g(sp['hold'], '5:0')):>7} "
            f"{str(g(sp['hold'], '5:1')):>7} {str(g(sp['open'], '5:0')):>7} "
            f"{str(g(sp['open'], '5:1')):>7} {pv(g(sp['parity'], '5:0')):>6} "
            f"{pv(g(sp['parity'], '5:1')):>6} {pv(g(sp['sacc'], '5:0')):>7} "
            f"{pv(g(sp['sacc'], '5:1')):>7}")
    out("")


# --------------------------------------------------------------------------------------- #
# [V2] THE DP'S CLASS ACCURACY TODAY
# --------------------------------------------------------------------------------------- #
def section_v2(a, out, banked=()):
    out("")
    out("=" * 100)
    out(f"[V2] THE DP'S CLASS ACCURACY TODAY — the expansion-choice instrument  "
        f"({a.tag}:{a.arm})")
    out("=" * 100)
    out("  `contains`     the chosen row's token class holds the CLEAN derivation's latent")
    out("                 feature at that node.")
    out("  `contains_rep` it holds SOME feature that repairs the instance there")
    out("                 (`fourwall.consistent_features`) — the honest reference, and the")
    out("                 one the head would be graded against.")
    out("  `succ`         the repair then graded as a success.")
    out("  SCOPE, from `enharmonic.py`'s own header: this is the AUDITION DP path only. It")
    out("  is not the beam's materialisation, and on an OPEN slot the beam's chooser is the")
    out("  corridor head, which this instrument never sees.")
    out("")

    def table(ar, label):
        era = ar.log["era"]
        acc = collections.defaultdict(lambda: collections.Counter())
        for i in range(ar.n):
            ex = ar.log["exp"][i] or {}
            for cell, z in ex.items():
                k = (era[i], cell)
                acc[k]["calls"] += z["calls"]
                acc[k]["contains"] += z["contains"]
                acc[k]["succ"] += z["succ"]
                acc[k]["rep"] += (z.get("contains_rep") or 0)
        out(f"  --- {label}")
        out("      era  cell     calls   contains   contains_rep   succ")
        for (e, cell) in sorted(acc, key=lambda z: (z[0], z[1])):
            z = acc[(e, cell)]
            n = max(z["calls"], 1)
            out(f"      {e:>3}  {cell:<6} {z['calls']:>7}   {z['contains'] / n:>8.3f}   "
                f"{z['rep'] / n:>12.3f}   {z['succ'] / n:>5.3f}")
        out("")
        return acc

    table(a, f"{a.tag}:{a.arm}")
    for b in banked:
        table(b, f"{b.tag}:{b.arm}  (banked)")


# --------------------------------------------------------------------------------------- #
# [V3] RECORD vs RECALL
# --------------------------------------------------------------------------------------- #
def section_v3(a, out, rules, canon, bottom):
    out("")
    out("=" * 100)
    out(f"[V3] RECORD vs RECALL — the inverse map's collisions, propagated  ({a.tag}:{a.arm})")
    out("=" * 100)
    out("  The learner writes `canon[f]` for a level-1 feature f. Its ear (the generator, the")
    out("  reader, both students of `bottom_map`) reads that back as")
    out("      g(f) = bottom_map[canon[f,0] + v*canon[f,1]] .")
    out("  Where g(f) != f the RECORD and the RECALL of the same write disagree at level 1,")
    out("  and the disagreement propagates up every tuple that contains f.")
    out("")
    powers = V ** np.arange(S)
    # every (feature, rule) that writes a code, so the two collisions the SPEC names can be
    # separated into the one a CANONICAL write can reach and the one it cannot.
    leaf = np.asarray(rules[DEPTH - 1], np.int64)                  # (v, m, s)
    codes_all = (leaf * powers).sum(2)                             # (v, m)
    writers = collections.defaultdict(list)
    for f in range(V):
        for r_ in range(M):
            writers[int(codes_all[f, r_])].append((f, r_))
    coll = {c: w for c, w in writers.items() if len(w) > 1}
    out(f"  `bottom_map` collisions on this draw (code -> the (feature, rule) pairs that "
        f"write it): {dict(sorted(coll.items()))}")
    out("  A macro writes `canon[f]` = rule 0 ONLY, so a collision is reachable by a write")
    out("  exactly when two features collide on their rule-0 codes.")
    for c_, w in sorted(coll.items()):
        r0 = [f for f, r_ in w if r_ == 0]
        out(f"      code {c_}: writers {w} -> rule-0 writers {r0} "
            f"{'(REACHABLE by a canonical write)' if len(r0) > 1 else '(not reachable)'}")
    out("")
    code = (canon * powers).sum(1)
    g = bottom[code]
    bad = [f for f in range(V) if g[f] != f]
    out(f"  canon codes  : {list(map(int, code))}")
    out(f"  g(f)         : {list(map(int, g))}")
    out(f"  level-1 features whose own canonical rendering reads back as another feature: "
        f"{bad}  ({len(bad)} of {V})")
    for f in bad:
        out(f"      f={f} renders as leaves {list(map(int, canon[f]))} (code "
            f"{int(code[f])}) and reads back as f'={int(g[f])}")
    out("")

    # propagate to the books
    out("  PROPAGATED TO THE BOOKS THE ARM ACTUALLY HELD. For each logged build, every row of")
    out("  the operative table is a level-1 tuple T (the RECORD of what the row writes). Its")
    out("  RECALL is g(T) elementwise. Two class spaces:")
    out("      tok    the token class (`quotient.token_class_sets`) — the identifiable one.")
    out("      learn  the arm's own learned class (`merge.LearnedQuotient` at that cycle),")
    out("             which is what the miner and the corridor would key on.")
    out("  A learned class is a union-find over tuples the arm has SEEN, so a recalled tuple")
    out("  outside the book is its own singleton — the recall key cannot name it. That case")
    out("  is counted apart (`unnamed`).")
    out("")
    out("      cyc  L   rows  rowsT!=T   tok-differs  learn-differs  learn-unnamed")
    idxs = [i for i in range(a.n) if any(a.build_of(i, lv).get("picks")
                                         for lv in (2, 3, 4, 5))]
    probe_i = idxs[-1:] + [i for i in idxs if a.cyc[i] in
                           (a.commits.get(4), a.commits.get(5))]
    probe_i = sorted(set(probe_i))
    tot = collections.Counter()
    for i in probe_i:
        q = a.quot_at(a.cyc[i] + 1)
        for lv in (2, 3, 4, 5):
            rows = a.book(i, lv)
            if not rows:
                continue
            R = np.array(rows, np.int64)
            Rg = g[R]
            n_tup = int((Rg != R).any(1).sum())
            Ptok = QT.token_class_sets(rules, R, lv, canon, V, S, DEPTH)[:, 0, :]
            Ptokg = QT.token_class_sets(rules, Rg, lv, canon, V, S, DEPTH)[:, 0, :]
            tok_diff = int((Ptok != Ptokg).any(1).sum())
            seen = {tuple(int(x) for x in r) for r in R}
            lrn = [q.id_of(tuple(int(x) for x in r), lv) for r in R]
            lrn_g = [q.id_of(tuple(int(x) for x in r), lv) for r in Rg]
            unnamed = sum(1 for j, r in enumerate(Rg)
                          if tuple(int(x) for x in r) not in seen)
            l_diff = sum(1 for x, y in zip(lrn, lrn_g) if x != y)
            out(f"      {a.cyc[i]:>3} L{lv} {len(rows):>6}  {n_tup:>8}   {tok_diff:>11}"
                f"  {l_diff:>13}  {unnamed:>13}")
            tot[(lv, "rows")] += len(rows)
            tot[(lv, "tup")] += n_tup
            tot[(lv, "tok")] += tok_diff
            tot[(lv, "lrn")] += l_diff
            tot[(lv, "unn")] += unnamed
    out("")
    out("      pooled over the cycles above:")
    for lv in (2, 3, 4, 5):
        if not tot[(lv, "rows")]:
            continue
        n = tot[(lv, "rows")]
        out(f"        L{lv}: {n:>5} rows — tuple differs {tot[(lv, 'tup')] / n:.3f}, "
            f"token class differs {tot[(lv, 'tok')] / n:.3f}, learned class differs "
            f"{tot[(lv, 'lrn')] / n:.3f}, recall unnamed {tot[(lv, 'unn')] / n:.3f}")
    out("")
    out("  THE NEURAL EAR IS A SECOND, SEPARATE LOSS on top of this structural one. Its rate")
    out("  is in the run's own record and is NOT folded in here:")
    ra = a.r.get("config", {})
    pr = [p for p in a.log.get("probe", []) if p]
    if pr:
        out(f"      plant parse_acc (the generator, the chooser's ear): first "
            f"{pr[0]['plant']['parse_acc']:.4f}  last {pr[-1]['plant']['parse_acc']:.4f}")
        out(f"      reader read_acc (the frozen listener)            : first "
            f"{pr[0]['plant']['read_acc']:.4f}  last {pr[-1]['plant']['read_acc']:.4f}")
    out("      Both are measured by `plant_probe` on `probe_clean` — CLEAN held-out")
    out("      configurations, against `bottom_map`. The reader's accuracy on a span the")
    out("      LEARNER wrote, sitting inside a partly-damaged configuration, is not measured")
    out("      anywhere in this tag. That number is exactly what an `own_recall` arm would")
    out("      expose, and it is the only channel left once the structural one reads 0.000.")
    out("")


# --------------------------------------------------------------------------------------- #
# [V4] THE L5 BOOK — classes against spellings
# --------------------------------------------------------------------------------------- #
def section_v4(a, out, rules, canon):
    out("")
    out("=" * 100)
    out(f"[V4] THE BOOK — classes against spellings, and where the cap binds  "
        f"({a.tag}:{a.arm})")
    out("=" * 100)
    out(f"  quot_spell_cap = {a.cfg['quot_spell_cap']} spellings per class per HALF, so a")
    out(f"  level-l class-pair key materialises at most cap^s = "
        f"{int(a.cfg['quot_spell_cap']) ** S} rows.")
    out("")
    idxs = [i for i in range(a.n) if a.build_of(i, 5).get("picks")]
    show = sorted(set([i for i in idxs if a.cyc[i] in (a.commits.get(5),)] + idxs[-1:]))
    for i in show:
        out(f"  --- cycle {a.cyc[i]}")
        out("      L  at_sup  built  rows  inv_mean  inv_max  n_class_capped  "
            "lower_rows  lower_classes")
        for lv in (2, 3, 4, 5):
            b = a.build_of(i, lv)
            if not b:
                continue
            out(f"      {lv}  {b['n_at_support']:>6}  {b['n_keys_built']:>5}  "
                f"{b['n_entries']:>4}  {b['inventory_mean']:>8.2f}  {b['inventory_max']:>7}  "
                f"{b['n_class_capped']:>14}  {b['n_lower_rows']:>10}  "
                f"{b['n_lower_classes']:>13}")
        out("")
        b5 = a.build_of(i, 5)
        nrc = {k: int(v) for k, v in b5["n_rows_in_class"].items()}
        pk = {k: v for k, v in b5["picks"].items()}
        held = sorted(nrc.values(), reverse=True)
        out(f"      L5's lower (L4) classes: {len(nrc)} classes over "
            f"{b5['n_lower_rows']} rows; spellings per class = {held}")
        out(f"      of those, {sum(1 for k in nrc if nrc[k] > int(a.cfg['quot_spell_cap']))}"
            f" hold more than the cap and are truncated to "
            f"{int(a.cfg['quot_spell_cap'])}; {sum(1 for k in nrc if nrc[k] == 1)} are "
            f"singletons.")
        # class coverage of the book itself
        keyed = a.book_keyed(i, 5)
        rows = [t for _, t in keyed]
        if rows:
            R = np.array(rows, np.int64)
            P = QT.token_class_sets(rules, R, 5, canon, V, S, DEPTH)[:, 0, :]
            tc = [frozenset(np.nonzero(x)[0].tolist()) for x in P]
            keys = [k for k, _ in keyed]
            out(f"      the L5 BOOK: {len(rows)} rows, {len(set(keys))} class-pair keys, "
                f"{len(set(tc))} distinct token classes ({sum(1 for c in tc if not c)} rows "
                f"off-grammar)")
            per_key = collections.defaultdict(set)
            for k, c in zip(keys, tc):
                per_key[k].add(c)
            multi = sum(1 for k in per_key if len(per_key[k]) > 1)
            out(f"      token classes per key: "
                f"{sorted(collections.Counter(len(s_) for s_ in per_key.values()).items())}"
                f"  ({multi} of {len(per_key)} keys are not single-class)")
            out("      => WHAT EXECUTION CHOOSES AMONG: one row of "
                f"{len(rows)}; the class choice is over {len(set(keys))} keys and the "
                f"remaining {len(rows) - len(set(keys))} are spelling choices inside a key.")
    out("")
    # the DP's actual use of the L5 book
    out("  THE DP's OWN USE OF THE L5 BOOK (the entry recorder, beam phase, closed path; the")
    out("  slot recorder, beam phase, fired path). Both are bincounts over the operative")
    out("  table's rows.")
    out("")
    if a.beam_rec:
        out("      cyc  level  rows  distinct rows used  top-1 share  top-4 share  calls"
            "   keys used / keys held   token classes used")
        cyc2i = {c: i for i, c in enumerate(a.cyc)}
        for rec in a.beam_rec:
            bm = rec.get("beam") or {}
            for lv, vec in sorted(bm.items()):
                vv = np.array(vec, np.int64)
                if vv.sum() == 0:
                    continue
                srt = np.sort(vv)[::-1]
                keyinfo = ""
                i = cyc2i.get(rec["cycle"])
                if i is not None:
                    kb = a.book_keyed(a.operative_index(i, int(lv)), int(lv))
                    if len(kb) == len(vv):
                        used = [j for j in range(len(vv)) if vv[j] > 0]
                        ku = {kb[j][0] for j in used}
                        Ru = np.array([kb[j][1] for j in used], np.int64)
                        Pu = QT.token_class_sets(rules, Ru, int(lv), canon, V, S,
                                                 DEPTH)[:, 0, :]
                        tcu = {frozenset(np.nonzero(x)[0].tolist()) for x in Pu}
                        keyinfo = (f"   {len(ku)} / {len({k for k, _ in kb})}"
                                   f"                 {len(tcu)}")
                out(f"      {rec['cycle']:>3}  L{lv}   {len(vv):>4}  {int((vv > 0).sum()):>18}"
                    f"  {srt[0] / vv.sum():>11.3f}  {srt[:4].sum() / vv.sum():>11.3f}"
                    f"  {int(vv.sum()):>6}{keyinfo}")
    out("")
    out("      `keys used / keys held` is the whole point of the column: a use concentrated")
    out("      INSIDE one class-pair key is a spelling choice and cannot change the class; a")
    out("      use spread over keys is a class choice. Blank where the rebuilt book's row")
    out("      count does not match the recorder's vector (the recorder is over the operative")
    out("      table at the cycle it fired, which an open arm moves).")
    out("")
    out("      slot recorder (fired path, the DP's intention reference on OPEN slots):")
    out("      cyc  slot   rows  distinct  top-1 share  calls")
    for i in range(a.n):
        sr = (a.log["slot"][i] or {}).get("beam") or {}
        for lv in sorted(sr):
            for node in sorted(sr[lv]):
                vv = np.array(sr[lv][node], np.int64)
                if vv.sum() == 0 or int(lv) < 5:
                    continue
                srt = np.sort(vv)[::-1]
                out(f"      {a.cyc[i]:>3}  {lv}:{node:<3}  {len(vv):>4}  "
                    f"{int((vv > 0).sum()):>8}  {srt[0] / vv.sum():>11.3f}  "
                    f"{int(vv.sum()):>6}")
    out("")
    out("  HOW MUCH THE WRITE VARIES AT ALL, per (era, level), over the slot recorder's own")
    out("  per-(level, node) vectors on the fired path. `distinct` is rows actually written in")
    out("  a cycle, `top1` the share of the cycle's calls that went to one row. This is the")
    out("  CONTRAST an own-attempt route has to learn from: a slot that writes one row on")
    out("  every call in a cycle has the same record on its solved and its unsolved calls.")
    out("")
    out("      era  L   slot-cycles   mean rows in table   mean distinct written   mean top1")
    agg = collections.defaultdict(list)
    for i in range(a.n):
        sr = (a.log["slot"][i] or {}).get("beam") or {}
        for lv in sorted(sr):
            for node in sorted(sr[lv]):
                vv = np.array(sr[lv][node], np.int64)
                if vv.sum() == 0:
                    continue
                agg[(a.log["era"][i], int(lv))].append(
                    (len(vv), int((vv > 0).sum()), float(np.max(vv) / vv.sum())))
    for k in sorted(agg):
        z = np.array(agg[k], float)
        out(f"      {k[0]:>3}  {k[1]}   {len(z):>11}   {z[:, 0].mean():>18.1f}   "
            f"{z[:, 1].mean():>21.2f}   {z[:, 2].mean():>9.3f}")
    out("")


# --------------------------------------------------------------------------------------- #
# [V5] THE OFFLINE CEILING
# --------------------------------------------------------------------------------------- #
def section_v5(a, out, rules, canon):
    out("")
    out("=" * 100)
    out("[V5] THE OFFLINE CEILING — what is cheap and what is not")
    out("=" * 100)
    out("  THE SPEC's FORM (a head fitted to the oracle's per-node repair set on the audition")
    out("  path, `embouchure`'s `own_verdict` analogue) IS NOT AVAILABLE OFFLINE HERE, and")
    out("  the reason is a property of the fork, not of effort:")
    out("    * the head's input is `trunk(core, obs)` — the pooled hiddens of the arm's OWN")
    out("      trained generator at that cycle. No banked tag carries the trunk's weights")
    out("      (the compact mirror is `results.json` + `entry*.json`), so there is no x to")
    out("      fit on.")
    out("    * the contexts are not banked either: `SpanExecutor._store` keeps them in memory")
    out("      and the log records only their COUNT (`span.buf` / `span.hold`).")
    out("    * `embouchure` could fit `own_verdict` offline because its head's input was a")
    out("      (feature, register) pair, not a learned representation, and its rows were in")
    out("      `spell_rows` on the volume.")
    out("  So the ceiling that is analogous here is an IN-RUN one (a `voicing` arm whose")
    out("  target is the repair set) and it belongs to Q1/Q2, not to Q0.")
    out("")
    out("  WHAT IS CHEAP OFFLINE, and is computed instead — the TABLE-SIDE ceiling. Whatever")
    out("  the chooser is, it picks one row of the operative table. So per (level, node) the")
    out("  best any chooser could do is bounded by whether the table HOLDS a row whose token")
    out("  class contains a feature that repairs the instance. The gap between that bound and")
    out("  the instrument's `contains_rep` is what a better chooser could win; the rest is")
    out("  the book's.")
    out("")
    out("  Computed on the run's own operative books. `features coverable` is the UNION of the")
    out("  token classes the book holds, against the alphabet v = 8. It is a property of the")
    out("  book, not a claim that all 8 are demanded at that node.")
    out("")
    out("      cyc  L   rows  token classes in book  features coverable  of v=8  "
        "rows needed for full cover")
    idxs = [i for i in range(a.n) if a.build_of(i, 5).get("picks")]
    show = sorted(set([i for i in idxs if a.cyc[i] == a.commits.get(5)] + idxs[-1:]))
    for i in show:
        for lv in (2, 3, 4, 5):
            rows = a.book(i, lv)
            if not rows:
                continue
            R = np.array(rows, np.int64)
            P = QT.token_class_sets(rules, R, lv, canon, V, S, DEPTH)[:, 0, :]
            cov = P.any(0)
            tc = {frozenset(np.nonzero(x)[0].tolist()) for x in P}
            # greedy minimum cover of the coverable features
            need, chosen = set(np.nonzero(cov)[0].tolist()), 0
            sets = [set(np.nonzero(x)[0].tolist()) for x in P]
            while need:
                best = max(sets, key=lambda s_: len(s_ & need))
                if not (best & need):
                    break
                need -= best
                chosen += 1
            out(f"      {a.cyc[i]:>3} L{lv} {len(rows):>6}  {len(tc):>21}  "
                f"{int(cov.sum()):>18}  {V:>6}  {chosen:>25}")
    out("")
    out("  `rows needed for full cover` is the size of the smallest sub-book whose classes")
    out("  cover every feature the book can reach at all: what EXECUTION needs, against what")
    out("  the book holds.")
    out("")


# --------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="en_s9")
    ap.add_argument("--arm", default="endo_ledger_open_ung5_ra")
    ap.add_argument("--bank", default="en_s8:endo_ledger_open_ung5,en_s8:endo_ledger")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "vo_q0_reduction.txt"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    rules = generate_rules_distinct(V, S, DEPTH, M, seed=0)
    canon = np.ascontiguousarray(rules[DEPTH - 1][:, 0, :])
    bottom = build_inverse_maps(rules)[-1]

    a = Arm(args.tag, args.arm)
    banked = []
    for spec in [z for z in args.bank.split(",") if z.strip()]:
        t, m_ = spec.split(":")
        p = os.path.join(EN_FIG, t, m_, "results.json")
        if os.path.isfile(p):
            banked.append(Arm(t, m_))

    out("=" * 100)
    out("[voicing] Q0 — offline, CPU, no substrate")
    out("=" * 100)
    out(f"  anchor  : {a.tag}:{a.arm}   ({a.n} cycles, commits "
        f"{ {k: v for k, v in sorted(a.commits.items())} })")
    out(f"  banked  : {', '.join(f'{b.tag}:{b.arm}' for b in banked) or '(none)'}")
    out(f"  world   : rule_seed 0, v={V} s={S} depth={DEPTH} m={M}")
    out("")
    gate_vq1(a, out)
    for b in banked:
        gate_vq1(b, out)

    section_v1(a, out)
    section_v2(a, out, banked)
    section_v3(a, out, rules, canon, bottom)
    section_v4(a, out, rules, canon)
    section_v5(a, out, rules, canon)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
