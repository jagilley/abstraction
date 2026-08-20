"""Reduce Rung B1 sessions to decision tables, timing, hold-vs-bail, and justification excerpts.

    python3 rhm/practice/teacher_slot/verbal/reduce.py --tag b1 > outputs/reduction_b1.txt
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

FIRST_DISRUPTION = 8000       # the first step at which the kept arm's readouts break
PRE = 6000                    # last decision point strictly before any disruption is visible


def wrap(s, w=96, ind="        "):
    if not s:
        return ind + "(none)"
    return "\n".join(textwrap.wrap(s, w, initial_indent=ind, subsequent_indent=ind))


def summarise(sess: dict) -> dict:
    ch = [(int(s), c) for s, c in sess["choices"]]
    firsts = [s for s, c in ch if c == "REPLACE_T"]
    first = firsts[0] if firsts else None
    # sustained removal: first REPLACE_T after which no LEAVE_T ever appears again
    sustained = None
    for i, (s, c) in enumerate(ch):
        if c == "REPLACE_T" and all(c2 == "REPLACE_T" for _, c2 in ch[i:]):
            sustained = s
            break
    # bail-out: a LEAVE_T immediately after a REPLACE_T
    bails = [ch[i][0] for i in range(1, len(ch))
             if ch[i][1] == "LEAVE_T" and ch[i - 1][1] == "REPLACE_T"]
    n_rep = sum(1 for _, c in ch if c == "REPLACE_T")
    rem_arms = [(t["step"], t["removal_arm_m"]) for t in sess["turns"]
                if t.get("removal_arm_m") is not None]
    mismatch = None
    if first is not None:
        ms = [m for st, m in rem_arms if st >= first]
        if ms:
            mismatch = abs(ms[0] - first)
    return {"choices": ch, "first": first, "sustained": sustained, "bails": bails,
            "n_rep": n_rep, "n_pts": len(ch), "mismatch": mismatch,
            "pre_removal": first is not None and first <= PRE,
            "final": ch[-1][1] if ch else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--excerpt-steps", default="1000,2000,4000,6000,8000,19000")
    a = ap.parse_args()
    with open(os.path.join(OUT, f"sessions_{a.tag}.json")) as f:
        D = json.load(f)
    ex_steps = {int(x) for x in a.excerpt_steps.split(",")}

    L = []
    P = L.append
    P("=" * 96)
    P(f"teacher_slot/verbal — Rung B1   tag {a.tag}   reasoner arg '{D['model_arg']}'   "
      f"splice '{D['splice']}'")
    P("=" * 96)
    models = sorted({m for s in D["sessions"] for t in s["turns"] for at_ in t["attempts"]
                     for m in (at_["call"].get("model_reported") or [])})
    P(f"  models reported by the CLI across all calls: {models}")
    P(f"  system prompt: {D['system_prompt']!r}")
    P(f"  sessions: {len(D['sessions'])}   errors: {len(D.get('errors', []))}")
    for e in D.get("errors", []):
        P(f"    ERROR {e}")
    P("")

    P("=" * 96)
    P("1. DECISIONS — one row per session, one column per decision point")
    P("=" * 96)
    P("   L = LEAVE_T, R = REPLACE_T.  Steps are the session clock.")
    P("")
    by_cell: dict[str, list] = {}
    for s in D["sessions"]:
        by_cell.setdefault(s["cell"], []).append(s)
    summaries = {}
    for cell, sess in sorted(by_cell.items()):
        P(f"   -- {cell}  ({sess[0]['info']} information, kept arm {sess[0]['kept_arm']})")
        for s in sorted(sess, key=lambda x: x["sample"]):
            sm = summarise(s)
            summaries[(cell, s["sample"])] = sm
            seq = "  ".join(f"{st}:{'R' if c == 'REPLACE_T' else 'L'}" for st, c in sm["choices"])
            P(f"      s{s['sample']} [{s['opt_order']}]  {seq}")
        P("")

    P("=" * 96)
    P("2. TIMING, HOLD-vs-BAIL, AND SPLICE MISMATCH")
    P("=" * 96)
    P("   first        first decision point at which REPLACE_T was chosen")
    P("   sustained    first REPLACE_T never followed by a LEAVE_T")
    P("   pre?         was `first` at or before step 6000, i.e. before any disruption is")
    P("                visible in the readouts (the money cell's question)")
    P("   bails        decision points at which LEAVE_T immediately followed a REPLACE_T")
    P("   mism         |m - S| for the removal arm spliced in at the first removal; 0 = the")
    P("                post-removal trajectory is an exact measurement of that removal step")
    P(f"   {'cell':<11}{'s':>2} {'first':>7} {'sustained':>10} {'pre?':>5} {'bails':>16} "
      f"{'%R':>6} {'mism':>6} {'final':>10}")
    for (cell, samp), sm in sorted(summaries.items()):
        P(f"   {cell:<11}{samp:>2} {str(sm['first']):>7} {str(sm['sustained']):>10} "
          f"{('yes' if sm['pre_removal'] else 'no'):>5} "
          f"{(','.join(map(str, sm['bails'])) or '-'):>16} "
          f"{100*sm['n_rep']/max(1, sm['n_pts']):>5.0f}% {str(sm['mismatch']):>6} "
          f"{sm['final'] or '-':>10}")
    P("")

    P("=" * 96)
    P("3. CELL SUMMARY")
    P("=" * 96)
    for cell in sorted(by_cell):
        sms = [v for (c, _), v in summaries.items() if c == cell]
        n = len(sms)
        pre = sum(1 for v in sms if v["pre_removal"])
        any_r = sum(1 for v in sms if v["first"] is not None)
        bail = sum(1 for v in sms if v["bails"])
        firsts = [v["first"] for v in sms if v["first"] is not None]
        P(f"   {cell:<11}  n={n}  removed at some point {any_r}/{n}  "
          f"removed pre-disruption (<= {PRE}) {pre}/{n}  bailed at least once {bail}/{n}"
          + (f"  first-removal steps {sorted(firsts)}" if firsts else ""))
    P("")

    P("=" * 96)
    P("4. JUSTIFICATIONS — verbatim, at selected decision points")
    P("=" * 96)
    for cell in sorted(by_cell):
        for s in sorted(by_cell[cell], key=lambda x: x["sample"]):
            for t in s["turns"]:
                if t["step"] not in ex_steps:
                    continue
                p = t.get("parsed") or {}
                P(f"   [{cell} s{s['sample']} step {t['step']}]  -> {p.get('choice')}")
                P("     RELIANCE:"); P(wrap(p.get("reliance")))
                P("     STABILITY:"); P(wrap(p.get("stability")))
                P("     REASONS:"); P(wrap(p.get("reasons")))
                P("")

    P("=" * 96)
    P("5. PROVENANCE — every displayed row, and where its numbers came from")
    P("=" * 96)
    for cell in sorted(by_cell):
        for s in sorted(by_cell[cell], key=lambda x: x["sample"]):
            P(f"   [{cell} s{s['sample']}]")
            for r in s["final_rows"]:
                lag = "" if (r["dec_step"] in (None, r["step"])) else f" dec@{r['dec_step']}"
                P(f"      session {r['step']:>6}  <- record {r['underlying']:>6}  "
                  f"{r['in_stream']:<7} {r['src']}{lag}")
            P("")

    print("\n".join(L))


if __name__ == "__main__":
    main()
