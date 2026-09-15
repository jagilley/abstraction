"""[sotto] The reducer — `voicing/analyze_voicing.py` forked, every section kept working on
the banked `voicing` tags, plus the four this node adds.

Sections:
  [A] lifetimes, commits, the in-tag identity check against the banked anchor
  [B] the corridor: slots open, parity, misfire, and WHEN each arm's corridor closed
  [C] the sampler, realised: deviation and on-table projection per (level, node) per era
  [D] the chooser's own use of the book, live — Q0 section 5's table on the executed population
  [E] contains / contains_rep on the priced beam (the head's class accuracy, held-out)
  [F] the in-situ read-back: record class against read-back class
  [G] the record's own volume and the objective's diet
  [H] L5/L6 arrival, class coverage, merges, and the bill
  [I] era-4/5 error, with the caveat
  [J] the critic's held-out AUC, per slot, FILED and PROBE rows apart
  [K] epsilon's own counters and the frontier (Q2; empty in a Q3 tag)
  [L] governance: which slots the critic earned and when
  [M] the corridor under a moved write, with the shift split by path
  [N] the composed choice: how often the critic moved the write off the DP's row  [Q3]
  [O] babbling off-stream: the probe's counts, what it graded, and its share of the bill [Q3]
  [Y] the clock yokes: realised against planned, gate Y-1's check post hoc               [Q3b]
  [Z] THE L4/L5 PANEL: every readout the round turns on, at the cells it turns on        [Q3b]
  [S] THE SEED TABLE: per seed, what the ladder did and WHY each advance fired          [Q3d]
  [P] THE MIRROR: the outcome model against the world — on held-out experience, and on the
      probes it graded                                                                   [sotto]
  [Q] DISAGREEMENT AS A DETECTOR: does the committee's spread find the mirror's own error [sotto]
  [R] THE BLIND REGION: where the mirror and the world disagree, by level and by whether the
      substituted class is in the slot's frequently-written set                          [sotto]
  [T] THE MIRROR'S GATE TABLE: M-1r / M-2r / M-3r / M-4r, re-asserted post hoc off the arm
      files                                                                              [sotto]

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/analyze_sotto.py --tag so_s1 \\
        --yoke-src vo_s3:voi3_dp \\
        --bank vo_s3:voi3_dp,vo_s3b:voi3b_comp_yk,vo_s3b:voi3b_comp_pr_yk
"""

import argparse
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
# [sotto] the FLOOR and the CEILING are banked in the parent node (`vo_s3b`, `vo_s3e`) and the
# anchor two nodes up, so the loader searches three trees.
VO_FIG = os.path.join(os.path.dirname(HERE), "figures")
EN_FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "enharmonic", "figures")


def load(tag, arm):
    for root in (FIG, VO_FIG, EN_FIG):
        p = os.path.join(root, tag, arm, "results.json")
        if os.path.isfile(p):
            return json.load(open(p))
    return None


def _auc(scores, labels):
    """[sotto] `vo_auc`'s rank identity, ties averaged, so a constant reader scores 0.5 and
    one class returns None. Duplicated here rather than imported so the reducer stays pure
    numpy; gate VO-12 checks the substrate's copy and this one is asserted against it below."""
    y = np.asarray(labels, np.float64)
    x = np.asarray(scores, np.float64)
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _mirror_run_gate(r, arm):
    """[sotto] Section [T] calls the SUBSTRATE'S OWN `vo_gate_mirror_run`, so the reduction
    cannot drift from the gate the preflight ran. Imported lazily: the reducer must still run
    where the substrate's imports are unavailable, and then [T] says so instead of raising."""
    from rhm.practice.voicing.sotto_voce import sotto_voce as SV
    return SV.vo_gate_mirror_run(r, arm=arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vo_s1")
    ap.add_argument("--arms", default="")
    ap.add_argument("--bank", default="en_s9:endo_ledger_open_ung5_ra")
    # [Q3b] the tag holding each yoke's SOURCE arm, as `tag:arm`. `enharmonic`'s [Y] uses a
    # hard-coded source map; here the map is in `ARMS[...]["loop"]["of"]` and only the tag the
    # source lives in has to be supplied, because a yoke may read its plan off a banked tag.
    ap.add_argument("--yoke-src", default="",
                    help="tag:arm holding the yokes' source, e.g. vo_s3:voi3_dp")
    # [Q3d] `tag:arm:label,...` — arms to put side by side in the seed table [S]. The label is
    # free text (normally the seed) because the seed is a RUN-level flag and does not survive
    # into the arm name.
    ap.add_argument("--seeds", default="",
                    help="tag:arm:label,... for section [S], e.g. vo_s3:voi3_dp:0,...")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out_path = a.out or os.path.join(FIG, f"{a.tag}_reduction.txt")
    lines = []

    def o(s=""):
        lines.append(s)
        print(s)

    tag_dir = next((os.path.join(rt, a.tag) for rt in (FIG, VO_FIG, EN_FIG)
                    if os.path.isdir(os.path.join(rt, a.tag))), os.path.join(FIG, a.tag))
    arms = [x for x in a.arms.split(",") if x] or sorted(
        d for d in os.listdir(tag_dir)
        if os.path.isfile(os.path.join(tag_dir, d, "results.json")))
    R = {arm: load(a.tag, arm) for arm in arms}
    B = {}
    for spec in [z for z in a.bank.split(",") if z.strip()]:
        t, m = spec.split(":")
        r = load(t, m)
        if r:
            B[spec] = r

    o("=" * 100)
    o(f"[sotto] {a.tag} — reduction")
    o("=" * 100)
    o(f"  arms   : {', '.join(arms)}")
    o(f"  banked : {', '.join(B) or '(none)'}")
    o("")

    # ---------------------------------------------------------------- [A] lifetimes/commits
    o("=" * 100)
    o("[A] LIFETIMES, COMMITS, AND THE IN-TAG IDENTITY CHECK")
    o("=" * 100)
    o("    arm                       cyc  commits (level@cycle)                 s/cycle")
    for arm in arms:
        r = R[arm]
        com = ", ".join(f"L{e['level']}@c{e['cycle']}" for e in r["events"]
                        if e["kind"] == "commit")
        n = len(r["log"]["cycle"])
        t = r["log"]["t_cum"][-1] if r["log"].get("t_cum") else None
        o(f"    {arm:24} {n:4}  {com:36}")
    for spec, r in B.items():
        com = ", ".join(f"L{e['level']}@c{e['cycle']}" for e in r["events"]
                        if e["kind"] == "commit")
        o(f"    {spec:24} {len(r['log']['cycle']):4}  {com:36}   (banked)")
    o("")
    # the identity check: the anchor against its banked source, series by series
    anch = next((x for x in arms if x.endswith("_dp")), None)
    SER = ("e", "succ", "dres", "n_solved", "n_mined", "n_moves", "width", "e_practice",
           "vloss", "gloss", "t_cum", "g_per_solve", "m_per_solve")
    if anch:
        o(f"    IDENTITY CHECK — {anch} against EVERY banked anchor. The anchor names no")
        o("    consumer knob, so it must be the banked arm to the bit; anything else means the")
        o("    record or an instrument is not inert.")
        for src, rb in B.items():
            la, lb = R[anch]["log"], rb["log"]
            worst = []
            for k in SER:
                if k not in la or k not in lb:
                    continue
                x, y = np.asarray(la[k], float), np.asarray(lb[k], float)
                L = min(len(x), len(y))
                worst.append((k, float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")))
            ca = [(e["level"], e["cycle"]) for e in R[anch]["events"] if e["kind"] == "commit"]
            cb = [(e["level"], e["cycle"]) for e in rb["events"] if e["kind"] == "commit"]
            o(f"      vs {src}:  MAX|delta| OVER ALL SERIES = "
              f"{max(d for _, d in worst):.3e}   commits equal: {ca == cb}")
            o("        " + "  ".join(f"{k}={d:.3e}" for k, d in worst))
            if ca != cb:
                o(f"        commits: {ca}  vs  {cb}")
    o("")

    # ---------------------------------------------------------------- [B] the corridor
    o("=" * 100)
    o("[B] THE CORRIDOR — slots open, parity, misfire, and when it closed")
    o("=" * 100)
    o("    `n_open` is slots past the firing gate; `misfire` is fired rows whose span did not")
    o("    match `dp_features` exactly. An arm whose corridor closes stops executing with the")
    o("    head and falls back to the DP on every macro call.")
    o("")
    o("    arm                    open@max  open@last  first c with open=0 after a peak"
      "   misfire(all)  misfire(era>=3)")
    for arm in arms:
        log = R[arm]["log"]
        op = [c["n_open"] for c in log["span"]]
        mx = max(op) if op else 0
        closed = None
        seen_peak = False
        for i, z in enumerate(op):
            if z >= max(1, mx // 2):
                seen_peak = True
            if seen_peak and z == 0:
                closed = log["cycle"][i]
                break
        nf = sum(b["n_fired"] for b in log["blocks"])
        nm = sum(b["n_misfire"] for b in log["blocks"])
        idx3 = [i for i in range(len(log["cycle"])) if log["era"][i] >= 3]
        nf3 = sum(log["blocks"][i]["n_fired"] for i in idx3)
        nm3 = sum(log["blocks"][i]["n_misfire"] for i in idx3)
        o(f"    {arm:24} {mx:6}  {op[-1]:9}  {str(closed):30}   {nm/max(nf,1):.4f}"
          f"        {nm3/max(nf3,1):.4f}")
    o("")
    o("    n_open per 10 cycles (the corridor's whole life):")
    for arm in arms:
        log = R[arm]["log"]
        op = [c["n_open"] for c in log["span"]]
        o(f"      {arm:22} " + " ".join(f"{op[i]:3}" for i in range(0, len(op), 10)))
    o(f"      {'cycle':22} " + " ".join(
        f"{R[arms[0]]['log']['cycle'][i]:3}"
        for i in range(0, len(R[arms[0]]['log']['cycle']), 10)))
    o("")

    # ---------------------------------------------------------------- [C] the sampler
    o("=" * 100)
    o("[C] THE SAMPLER, REALISED — deviation and on-table projection")
    o("=" * 100)
    o("    `dev`  the sampled entry left the on-table argmax (the quantity the temperature is")
    o("           set on; `DESIGN.md` §17 predicted this would fall below the preflight's")
    o("           0.44-0.67 because a sharper plant deviates less at fixed T).")
    o("    `ont`  the on-table argmax was not the head's FREE-RUN emission — the projection")
    o("           component that rides with the write axis (`DESIGN.md` §11).")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        agg = collections.defaultdict(lambda: [0, 0, 0])
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            for cell, z in (v.get("xp") or {}).items():
                agg[(log["era"][i], cell)][0] += z["n"]
                agg[(log["era"][i], cell)][1] += z["shift"]
                agg[(log["era"][i], cell)][2] += z["ontable_shift"]
        if not agg:
            o(f"    {arm}: argmax arm — never sampled.")
            continue
        o(f"    {arm}:")
        o("      era  cell        n        dev     ont")
        for k in sorted(agg, key=lambda z: (z[0], int(z[1].split('n')[0]),
                                            int(z[1].split('n')[1]))):
            z = agg[k]
            o(f"      {k[0]:>3}  {k[1]:<6} {z[0]:>9}  {z[1]/max(z[0],1):>7.3f} "
              f"{z[2]/max(z[0],1):>7.3f}")
        o("")

    # ---------------------------------------------------------------- [D] the write, live
    o("=" * 100)
    o("[D] WHAT THE CHOOSER WROTE — Q0 §5's table, live, on the EXECUTED population")
    o("=" * 100)
    o("    Filed writes only (kept by the beam and on a graded tip). `rows` is the operative")
    o("    table's size, `dist` distinct tuples written, `keys` distinct class-pair keys,")
    o("    `tok` distinct token classes, `top1` the share of the era's filed writes on one")
    o("    tuple. Q0's banked numbers at the same nodes: L4 2.6-3.6 of 155-187 rows, top1")
    o("    0.91-0.94; L5 1.5-1.6 of 144-152, top1 0.80-0.93 (the FIRED population).")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        agg = collections.defaultdict(lambda: collections.Counter())
        tup = collections.defaultdict(collections.Counter)
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            for cell, z in (v.get("var") or {}).items():
                if cell == "_":
                    continue
                k = (log["era"][i], cell)
                agg[k]["n"] += z["n"]
                agg[k]["rows"] = max(agg[k]["rows"], z["rows"])
                agg[k]["dist"] += z["distinct"]
                agg[k]["keys"] += z["keys"]
                agg[k]["tok"] += z["tok"]
                agg[k]["cyc"] += 1
                agg[k]["top1w"] += z["top1"] * z["n"]
        if not agg:
            o(f"    {arm}: nothing filed.")
            continue
        o(f"    {arm}:")
        o("      era  cell      n_filed  rows   dist/cyc  keys/cyc  tok/cyc   top1(w)")
        for k in sorted(agg, key=lambda z: (z[0], int(z[1].split('n')[0]),
                                            int(z[1].split('n')[1]))):
            z = agg[k]
            c = max(z["cyc"], 1)
            o(f"      {k[0]:>3}  {k[1]:<6} {z['n']:>8}  {z['rows']:>4}   "
              f"{z['dist']/c:>7.2f}  {z['keys']/c:>7.2f}  {z['tok']/c:>6.2f}   "
              f"{z['top1w']/max(z['n'],1):>7.3f}")
        o("")

    # ---------------------------------------------------------------- [E] contains / rep
    o("=" * 100)
    o("[E] THE WRITER'S CLASS ACCURACY ON THE PRICED BEAM (held-out: fresh instances/cycle)")
    o("=" * 100)
    o("    `contains` against the clean latent, `rep` against the set of features that REPAIR")
    o("    the instance there. On an open slot the writer IS the head, so `rep` is the head's")
    o("    class accuracy; where the corridor is closed it is the DP's. Q0 §3's audition-path")
    o("    numbers for the DP at the frontier: 4n3 contains_rep 0.718 (era 4), 5n1 0.582.")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        agg = collections.defaultdict(lambda: collections.Counter())
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            for src in ("contains", "rep"):
                for cell, z in (v.get(src) or {}).items():
                    agg[(log["era"][i], cell)][src + "_n"] += z["n"]
                    agg[(log["era"][i], cell)][src + "_h"] += z["hit"]
        if not agg:
            continue
        o(f"    {arm}:")
        o("      era  cell     contains n    contains   rep n     rep")
        for k in sorted(agg, key=lambda z: (z[0], int(z[1].split('n')[0]),
                                            int(z[1].split('n')[1]))):
            z = agg[k]
            cn, ch = z["contains_n"], z["contains_h"]
            rn, rh = z["rep_n"], z["rep_h"]
            o(f"      {k[0]:>3}  {k[1]:<6} {cn:>9}  {ch/max(cn,1):>9.3f}  {rn:>7}  "
              f"{rh/max(rn,1):>7.3f}")
        o("")

    # ---------------------------------------------------------------- [F] the read-back
    o("=" * 100)
    o("[F] THE IN-SITU READ-BACK — the number DESIGN §15 says decides whether Q2's")
    o("    `own_readback` has anything to read")
    o("=" * 100)
    o("    The frozen reader run on the span the learner just wrote, IN PLACE, quotiented")
    o("    through the arm's own class map and tallied against the record's class. Never")
    o("    summed with the verdict. Q0 §4: the STRUCTURAL disagreement on the token class is")
    o("    0.000 by construction on this draw, so everything here is the neural ear plus the")
    o("    learned quotient.")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        agg = collections.defaultdict(lambda: collections.Counter())
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            for cell, z in (v.get("readback") or {}).items():
                k = (log["era"][i], cell)
                for q in ("n", "class_agree", "tuple_agree", "unnamed"):
                    agg[k][q] += z.get(q, 0)
        if not agg:
            continue
        o(f"    {arm}:")
        o("      era  cell         n   class_agree  tuple_agree   unnamed")
        for k in sorted(agg, key=lambda z: (z[0], int(z[1].split('n')[0]),
                                            int(z[1].split('n')[1]))):
            z = agg[k]
            n = max(z["n"], 1)
            o(f"      {k[0]:>3}  {k[1]:<6} {z['n']:>7}  {z['class_agree']/n:>10.3f}  "
              f"{z['tuple_agree']/n:>10.3f}  {z['unnamed']/n:>8.3f}")
        o("")

    # ---------------------------------------------------------------- [G] the record's diet
    o("=" * 100)
    o("[G] THE RECORD'S VOLUME AND THE OBJECTIVE'S DIET")
    o("=" * 100)
    o("    `filed` writes kept by the beam and on a graded tip; `solved` their verdict;")
    o("    `unnamed` training rows whose recorded class has no member in the current table")
    o("    (the argmax arms' own hazard — DESIGN §14); `verify_bad` gate V-1's counter.")
    o("")
    o("    arm                    written    filed   solved%   unnamed   verify   v_bad"
      "   buf(last, max over slots)")
    for arm in arms:
        log = R[arm]["log"]
        V = [v for v in log["vo"] if v]
        if not V:
            o(f"    {arm:24} (recorder off)")
            continue
        w = sum(v["n_write"] for v in V)
        f = sum(v["n_filed"] for v in V)
        s_ = sum(v["n_solved"] for v in V)
        u = sum(v["n_unnamed"] for v in V)
        ver = sum(v["n_verify"] for v in V)
        vb = sum(v["n_verify_bad"] for v in V)
        buf = V[-1].get("buf") or {}
        o(f"    {arm:24} {w:>8} {f:>8}   {s_/max(f,1):>6.3f}  {u:>8}  {ver:>7}  {vb:>6}"
          f"   {max(buf.values()) if buf else 0}")
    o("")

    # ---------------------------------------------------------------- [H] L5/L6, merges, bill
    o("=" * 100)
    o("[H] ARRIVAL, CLASS COVERAGE, MERGES AND THE BILL")
    o("=" * 100)
    o("    arm                   L5@sup(end) L6@sup(end)  L4 classes(end)  merges taken"
      "  exp_reads   t_cum")
    for arm in list(arms) + list(B):
        r = R.get(arm) or B[arm]
        log = r["log"]
        pan = [p for p in log["panel"] if p]
        at = pan[-1].get("at_support", {}) if pan else {}
        q = [z for z in log["quot"] if z and z.get("build")]
        l4 = None
        for z in reversed(q):
            b = z["build"].get("4") or {}
            if b.get("n_lower_classes"):
                l4 = b.get("n_lower_classes")
                break
        mt = sum(1 for e in r.get("merge_events", [])
                 if e.get("kind") == "merge" and e.get("taken"))
        o(f"    {arm:24} {str(at.get('5')):>10} {str(at.get('6')):>11}  {str(l4):>14}"
          f"  {mt:>12}  {r.get('exp_reads', 0):>9}  {log['t_cum'][-1]:>10.0f}")
    o("")

    # ---------------------------------------------------------------- [I] era error
    o("=" * 100)
    o("[I] ERA-4/5 ERROR — with the caveat")
    o("=" * 100)
    o("    `enharmonic`'s matched-clock gaps between near-identical arms reach 0.08, and these")
    o("    arms are SELF-PACED and lived different numbers of cycles, so error is NOT a claim")
    o("    here. The structural readouts above are.")
    o("")
    o("    arm                   " + "  ".join(f"era{e}" for e in (1, 2, 3, 4, 5)))
    for arm in list(arms) + list(B):
        r = R.get(arm) or B[arm]
        log = r["log"]
        row = []
        for e in (1, 2, 3, 4, 5):
            idx = [i for i in range(len(log["cycle"])) if log["era"][i] == e]
            row.append(f"{np.mean([log['e'][i] for i in idx]):.3f}" if idx else "  -  ")
        o(f"    {arm:24} " + "  ".join(f"{x:>5}" for x in row))
    o("")

    # ---------------------------------------------------------------- [J] the critic
    o("=" * 100)
    o("[J] THE CRITIC — held-out AUC against the verdict, per slot, over the run")
    o("=" * 100)
    o("    The held-out split is by the parity gate's own BIJECTIVE row code, not a coin flip,")
    o("    so a context recurring across cycles cannot sit on both sides. `auc` is None where")
    o("    one class is absent on the held-out rows — an AUC is undefined then, and saying so")
    o("    beats reporting 0.5. DENOMINATORS AND BASE RATES ARE PART OF THE READING: the")
    o("    preflight's AUCs sat on 16-26 rows at base rate 0.045-0.08 and were a path check.")
    o("")
    o("    [sotto] `auc_world` is the probe AUC scored against the WORLD'S verdict on the")
    o("    same rows rather than against the verdict that was FILED. On a world-graded arm the")
    o("    two columns are the same number; on a model-graded arm they are not, and the one")
    o("    the question is about is `auc_world` — can the critic rank what the world would have")
    o("    said, having been taught by a mirror. `filed==world` is the share of the filed")
    o("    probe rows whose verdict the world agreed with, i.e. the mirror's own accuracy on")
    o("    the rows it actually put into the critic.")
    o("")
    o("    [Q3] THE TWO DIETS ARE READ APART and are NEVER pooled. The filed rows are the")
    o("    practice beam's own writes — a narrow, self-selected candidate set (Q0 finding 3:")
    o("    at the frontier almost every write falls in one token class) — and the probe rows")
    o("    are the babbler's substitutions, a DIFFERENT class by construction. A pooled AUC")
    o("    would read that candidate-set difference as discrimination. Both columns carry")
    o("    their own denominator and base rate.")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        per = collections.defaultdict(list)
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            for k, q in (v.get("critic") or {}).items():
                per[k].append((log["cycle"][i], q.get("n"), q.get("base_rate"), q.get("auc"),
                               q.get("probe_n"), q.get("probe_base_rate"),
                               q.get("probe_auc"), q.get("probe_auc_world"),
                               q.get("probe_file_vs_world")))
        if not per:
            o(f"    {arm}: no critic.")
            continue
        o(f"    {arm}:")
        o("      FILED rows                                              PROBE rows [Q3], "
          "the AUC scored against the WORLD [sotto]")
        o("      slot   reads  n(last) base(last)  auc f/med/last  >.5   "
          "n(last) base(last)  auc f/med/last  >.5   auc_world f/med/last  filed==world")
        for k in sorted(per, key=lambda z: (int(z.split(":")[0]), int(z.split(":")[1]))):
            rows = per[k]
            au = [z[3] for z in rows if z[3] is not None]
            pa = [z[6] for z in rows if z[6] is not None]
            pw = [z[7] for z in rows if len(z) > 7 and z[7] is not None]
            fw = [z[8] for z in rows if len(z) > 8 and z[8] is not None]
            pn, pb = rows[-1][4], rows[-1][5]
            fil = (f"{au[0]:.3f}/{float(np.median(au)):.3f}/{au[-1]:.3f}  "
                   f"{float(np.mean([a > 0.5 for a in au])):.2f}" if au
                   else "  (undefined: one class)   ")
            prb = (f"{pa[0]:.3f}/{float(np.median(pa)):.3f}/{pa[-1]:.3f}  "
                   f"{float(np.mean([a > 0.5 for a in pa])):.2f}" if pa
                   else ("  (undefined: one class)   " if pn else "        (no probe)   "))
            wld = (f"{pw[0]:.3f}/{float(np.median(pw)):.3f}/{pw[-1]:.3f}" if pw
                   else "      (n/a)      ")
            fvw = (f"{float(np.mean(fw)):.3f}" if fw else "  -  ")
            o(f"      {k:5} {len(rows):>6}  {rows[-1][1]:>7} {rows[-1][2]:>10.3f}  {fil}   "
              f"{(pn or 0):>7} "
              + (f"{pb:>10.3f}" if pb is not None else f"{'-':>10}")
              + f"  {prb}   {wld}   {fvw}")
        o("")

    # ---------------------------------------------------------------- [K] the explorer
    o("=" * 100)
    o("[K] EPSILON — its OWN counters, and the frontier it is confined to")
    o("=" * 100)
    o("    `n_eps` counts calls where epsilon FIRED; `eps_shift` those where the uniform class")
    o("    draw actually changed the write (it is below `n_eps` because a uniform draw can land")
    o("    on the class the argmax would have written). These are separate from `shift`, which")
    o("    also counts the CRITIC re-deciding — on a governed arm the two are not separable")
    o("    from `shift` alone. Q1 §19 is why the frontier confinement exists at all.")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        agg = collections.defaultdict(lambda: collections.Counter())
        off, act = [], 0
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            lv = set()
            for cell, z in (v.get("xp") or {}).items():
                agg[cell]["n"] += z["n"]
                agg[cell]["shift"] += z["shift"]
                agg[cell]["n_eps"] += z.get("n_eps", 0)
                agg[cell]["eps_shift"] += z.get("eps_shift", 0)
                if z.get("n_eps", 0) > 0:
                    lv.add(int(cell.split("n")[0]))
            if lv:
                act += 1
                if v.get("frontier") is None or lv != {int(v["frontier"])}:
                    off.append((log["cycle"][i], v.get("frontier"), sorted(lv)))
        tot_e = sum(z["n_eps"] for z in agg.values())
        if tot_e == 0:
            o(f"    {arm}: epsilon never fired (argmax arm).")
            o("")
            continue
        o(f"    {arm}:  total n_eps = {tot_e}  over {act} cycles with epsilon activity")
        o(f"      OFF-FRONTIER CYCLES: {len(off)}"
          + (f"   {off[:5]}" if off else "   (epsilon never left the frontier)"))
        o("      cell     n        n_eps   eps/n    eps_shift/n_eps   shift/n (eps+critic)")
        for cell in sorted(agg, key=lambda z: (int(z.split("n")[0]), int(z.split("n")[1]))):
            z = agg[cell]
            if not z["n_eps"]:
                continue
            o(f"      {cell:<6} {z['n']:>8} {z['n_eps']:>8}  {z['n_eps']/max(z['n'],1):>6.3f}"
              f"   {z['eps_shift']/max(z['n_eps'],1):>14.3f}   {z['shift']/max(z['n'],1):>7.3f}")
        o("      the frontier, per cycle (cycle:level):")
        fr = [(log["cycle"][i], v.get("frontier")) for i, v in enumerate(log["vo"])
              if v and v.get("frontier") is not None]
        runs = []
        for c, f in fr:
            if runs and runs[-1][1] == f:
                runs[-1][2] = c
            else:
                runs.append([c, f, c])
        o("        " + "  ".join(f"c{a}-c{b}:L{f}" for a, f, b in runs))
        o("")

    # ---------------------------------------------------------------- [L] governance
    o("=" * 100)
    o("[L] GOVERNANCE — which slots the critic earned, when, and how often it moved the write")
    o("=" * 100)
    o(f"    A slot is governed once it holds `vo_critic_min` filed rows. `shift - eps_shift`")
    o("    is the part of the re-decision the CRITIC is responsible for.")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        gov_first, gov_last = {}, {}
        for i, v in enumerate(log["vo"]):
            if not v:
                continue
            for k in (v.get("governed") or []):
                gov_first.setdefault(k, log["cycle"][i])
                gov_last[k] = log["cycle"][i]
        if not gov_first:
            o(f"    {arm}: the critic governed no slot.")
            o("")
            continue
        agg = collections.defaultdict(lambda: collections.Counter())
        for v in log["vo"]:
            if not v:
                continue
            for cell, z in (v.get("xp") or {}).items():
                agg[cell]["n"] += z["n"]
                agg[cell]["shift"] += z["shift"]
                agg[cell]["eps_shift"] += z.get("eps_shift", 0)
        o(f"    {arm}:  {len(gov_first)} slots governed")
        o("      slot   first  last   calls     critic-moved share")
        for k in sorted(gov_first, key=lambda z: (int(z.split(":")[0]), int(z.split(":")[1]))):
            cell = f"{k.split(':')[0]}n{k.split(':')[1]}"
            z = agg.get(cell, {})
            n_ = z.get("n", 0)
            cm = max(z.get("shift", 0) - z.get("eps_shift", 0), 0)
            o(f"      {k:5} c{gov_first[k]:<5} c{gov_last[k]:<5} {n_:>7}   "
              f"{(cm / n_ if n_ else 0.0):>17.3f}")
        o("")

    # ---------------------------------------------------------------- [M] the corridor
    o("=" * 100)
    o("[M] THE CORRIDOR UNDER A MOVED WRITE — misfire with the treatment's own draws removed")
    o("=" * 100)
    o("    `n_misfire` counts FIRED rows (the head executed) whose span did not match")
    o("    `dp_features` exactly. Where the treatment RE-DECIDED the write, a mismatch with the")
    o("    DP's row IS the treatment and not a defect — so the raw rate is not the corridor's")
    o("    health on a treated arm.")
    o("")
    o("    [Q3] THE SPLIT Q2 COULD NOT MAKE. `n_sampled` / `n_xp_shift` count every")
    o("    re-decided call, OPEN and CLOSED alike, and `n_fired` counts only calls the head")
    o("    executed — so on an arm whose corridor is mostly shut, Q2's `shift - misfire` was")
    o("    a subtraction across two different denominators and `excess` had to be withheld")
    o("    wherever `shift > n_misfire`. The recorder now tallies the shift BY PATH")
    o("    (`n_shift_fired` / `n_shift_closed`), so `excess` below is")
    o("        (n_misfire - shift_FIRED) / n_fired")
    o("    on the same denominator throughout. It is the misfire the moved write does NOT")
    o("    explain. A negative value means the head matched the DP on rows the treatment had")
    o("    moved — possible, and printed rather than clipped. Q2's undecomposable form is kept")
    o("    in the last two columns so a Q2 tag reduces to the same numbers it did.")
    o("")
    o("    arm                    open max/last  n_fired    misfire   shift_fire  shift_clos"
      "   excess    [Q2 form]")
    for arm in arms:
        log = R[arm]["log"]
        op = [c["n_open"] for c in log["span"]]
        nf = sum(b["n_fired"] for b in log["blocks"])
        nm = sum(b["n_misfire"] for b in log["blocks"])
        sm = sum(v["n_sampled"] for v in log["vo"] if v)
        sh = sum(v["n_xp_shift"] for v in log["vo"] if v)
        sf = sum(v.get("n_shift_fired", 0) for v in log["vo"] if v)
        sc = sum(v.get("n_shift_closed", 0) for v in log["vo"] if v)
        # a tag banked BEFORE the split carries neither counter; say so instead of
        # printing a zero that would read as "the treatment never moved a fired write"
        has = any(("n_shift_fired" in v) for v in log["vo"] if v)
        exc = f"{(nm - sf) / max(nf, 1):>8.4f}" if has else "     n/a"
        q2 = (f"{(nm - sh) / max(nf, 1):.4f}" if sh <= nm else "n/a")
        o(f"    {arm:22} {max(op) if op else 0:>6} /{op[-1] if op else 0:>5}  {nf:>9}   "
          f"{nm/max(nf,1):>7.4f}   {sf:>9}   {sc:>9}  {exc}   "
          f"shift/sampled {sh/max(sm,1):.4f} excess {q2}")
    o("")
    o("    n_open per 10 cycles:")
    for arm in arms:
        op = [c["n_open"] for c in R[arm]["log"]["span"]]
        o(f"      {arm:22} " + " ".join(f"{op[i]:3}" for i in range(0, len(op), 10)))
    o("")

    # ------------------------------------------------------- [N] the composed choice [Q3]
    o("=" * 100)
    o("[N] THE COMPOSED CHOICE — how often the critic moved the write off the DP's row")
    o("=" * 100)
    o("    On a governed slot the composed chooser is  argmax_t  z(dp_t / span) + w * z(critic_t)")
    o("    over the on-table candidates, each z-scored over the candidate SET. `moved` is the")
    o("    share of governed calls where that argmax is not the DP's own argmax — MEASURED at")
    o("    the choice, never inferred from the write.")
    o("")
    o("    `dp_vs_head` is printed beside it and is a DIFFERENT quantity: on an OPEN slot the")
    o("    executor's own would-be write is the head's emission, not the DP's row, so the two")
    o("    priors already disagree before the critic is consulted. Reading `moved` without it")
    o("    would attribute that standing disagreement to the critic. It is 0 on a closed slot")
    o("    by construction (there the DP IS the writer).")
    o("")
    o("    A `moved` share near 0 says the critic is decorative at this w; near 1 says the DP")
    o("    is. Neither is a defect and neither is read as one here.")
    o("")
    for arm in arms:
        log = R[arm]["log"]
        agg = collections.defaultdict(collections.Counter)
        for v in log["vo"]:
            if not v:
                continue
            for cell, z in (v.get("moved") or {}).items():
                for k_ in ("n", "moved", "dp_vs_head"):
                    agg[cell][k_] += z.get(k_, 0)
        if not agg:
            o(f"    {arm}: no composed choice (governance off, or a Q2 tag).")
            o("")
            continue
        w_ = R[arm]["config"].get("vo_w")
        md = R[arm]["config"].get("vo_govern_mode")
        o(f"    {arm}:  mode={md}  w={w_}")
        o("      cell    governed calls      moved   share    dp_vs_head share")
        for cell in sorted(agg, key=lambda z: (int(z.split("n")[0]), int(z.split("n")[1]))):
            z = agg[cell]
            n_ = z["n"]
            o(f"      {cell:6} {n_:>15}  {z['moved']:>9}  {z['moved']/max(n_,1):>6.3f}   "
              f"{z['dp_vs_head']/max(n_,1):>16.3f}")
        tn = sum(z["n"] for z in agg.values())
        tm = sum(z["moved"] for z in agg.values())
        o(f"      {'ALL':6} {tn:>15}  {tm:>9}  {tm/max(tn,1):>6.3f}")
        o("")

    # ------------------------------------------------------------ [O] babbling off-stream
    o("=" * 100)
    o("[O] BABBLING OFF-STREAM — the probe's counts, what it graded, and its share of the bill")
    o("=" * 100)
    o("    Per governed slot per cycle, up to `vo_probe_n` filed contexts: the trajectory's")
    o("    FINAL configuration with a DIFFERENT on-table class substituted at that slot,")
    o("    rendered and graded once. The verdict answers 'would this write ALONE have been")
    o("    right, holding the rest of the trajectory fixed' — not 'would the beam have")
    o("    solved it'; the rest of the trajectory is the beam's own, unchanged. Said here")
    o("    because the probe's solve rate is only comparable to the FILED rate under that")
    o("    reading.")
    o("")
    o("    PRICED: every probe grading is a grounding on the meter at `d_fb`, inside the")
    o("    cycle's own ledger. `share` below is of the WHOLE cycle's priced ledger, the")
    o("    probe's own cost included — the number the design's ~5% cap is stated against.")
    o("    Gate V-4B asserts the bill moved by exactly the probe count and gate V-5 that")
    o("    nothing else did.")
    o("")
    o("    arm                    probes   graded    solved   solve rate    bill: mean / max"
      "    cycles billed")
    for arm in arms:
        log = R[arm]["log"]
        vv = [v for v in log["vo"] if v]
        np_ = sum(v.get("n_probe", 0) for v in vv)
        ng = sum(v.get("n_probe_ground", 0) for v in vv)
        ns = sum(v.get("n_probe_solved", 0) for v in vv)
        bill = [b["share"] for b in (log.get("vo_bill") or []) if b["probe_ground"] > 0]
        if not np_:
            o(f"    {arm:22} (no probe)")
            continue
        o(f"    {arm:22} {np_:>6}  {ng:>7}  {ns:>8}   {ns/max(np_,1):>10.4f}    "
          f"{(float(np.mean(bill)) if bill else 0.0):.4f} / "
          f"{(max(bill) if bill else 0.0):.4f}    {len(bill):>7} / {len(log['cycle'])}")
    o("")
    o("    the FILED solve rate on the same slots, for the comparison the reading above")
    o("    licenses (both are per-write verdicts; the filed one is on the write that was")
    o("    actually made, the probe one on a substitution into the same final state):")
    for arm in arms:
        log = R[arm]["log"]
        vv = [v for v in log["vo"] if v]
        nf = sum(v.get("n_filed", 0) for v in vv)
        sv = sum(v.get("n_solved", 0) for v in vv)
        np_ = sum(v.get("n_probe", 0) for v in vv)
        ns = sum(v.get("n_probe_solved", 0) for v in vv)
        if not np_:
            continue
        o(f"      {arm:22} filed {sv:>7}/{nf:<7} = {sv/max(nf,1):.4f}    "
          f"probe {ns:>7}/{np_:<7} = {ns/max(np_,1):.4f}")
    o("")
    o("    the bill per era, so a share that grows with the ladder is visible:")
    for arm in arms:
        log = R[arm]["log"]
        bl = log.get("vo_bill") or []
        if not any(b["probe_ground"] for b in bl):
            continue
        per = collections.defaultdict(lambda: [0.0, 0.0])
        for i, b in enumerate(bl):
            per[log["era"][i]][0] += b["probe_priced"]
            per[log["era"][i]][1] += b["cycle_priced"]
        o(f"      {arm:22} " + "  ".join(
            f"e{e}:{(v[0]/v[1] if v[1] else 0.0):.4f}" for e, v in sorted(per.items())))
    o("")

    # ------------------------------------------------- [P] the mirror against the world [sotto]
    o("=" * 100)
    o("[P] THE MIRROR — the outcome model against the world")
    o("=" * 100)
    o("    The outcome model reads a RENDERED CONFIGURATION and its root — what the world's")
    o("    grader reads — and is trained by BCE against the world's verdict on the")
    o("    configurations the learner actually produced (the filed rows' `fin` and `y`,")
    o("    deduplicated by (configuration, root) within the cycle). It is a separate object:")
    o("    its own parameters, its own optimizer, its own numpy streams, never the plant's.")
    o("")
    o("    TWO NUMBERS, and the gap between them is the whole question. On HELD-OUT")
    o("    EXPERIENCE the mirror is scored on rows drawn from the same distribution it was")
    o("    polished on. On PROBES it is scored on substitutions the learner never wrote — off")
    o("    its own support by construction — against the world's verdict, which every arm")
    o("    computes as an instrument and no arm consumes (gate M-1).")
    o("")
    o("    arm                  mode        K   rows   held-out EXPERIENCE      on PROBES")
    o("                                                 acc     AUC    base     acc    base"
      "   n")
    for arm in arms:
        r = R[arm]
        oms = [v for v in (r["log"].get("vo_om") or []) if v]
        mode = str(r.get("vo_om_mode") or "world")
        mir = r.get("vo_mirror") or {}
        n_m = sum(int(c["n"]) for c in mir.values())
        tp = sum(int(c["tp"]) for c in mir.values()); tn = sum(int(c["tn"]) for c in mir.values())
        if not oms:
            o(f"    {arm:20} {mode:10}  (no outcome model)")
            continue
        last = oms[-1]
        acc_p = (tp + tn) / max(1, n_m)
        base_p = sum(int(c["world_solved"]) for c in mir.values()) / max(1, n_m)
        o(f"    {arm:20} {mode:10} {last['K']:>2} {last['n_rows']:>6}   "
          f"{(last['hold_acc'] or 0):.4f}  {(last['hold_auc'] or 0):.4f}  "
          f"{(last['hold_base'] or 0):.4f}   {acc_p:.4f}  {base_p:.4f} {n_m:>7}")
    o("")
    o("    the mirror's own trajectory (held-out accuracy and AUC against the world, per")
    o("    cycle, thinned): a mirror that is still improving at the last cycle is a mirror")
    o("    whose blind region is a function of how much experience it has had, not of the")
    o("    architecture.")
    for arm in arms:
        oms = [(i, v) for i, v in enumerate(R[arm]["log"].get("vo_om") or []) if v]
        if not oms:
            continue
        step = max(1, len(oms) // 8)
        o(f"      {arm:20} " + "  ".join(
            f"c{R[arm]['log']['cycle'][i]}:{(v['hold_auc'] or 0):.3f}"
            for i, v in oms[::step]))
    o("")
    o("    the committee's spread on held-out experience and the agreement threshold it set")
    o("    (the threshold is the q-th quantile of that spread, q = `vo_om_agree_q`):")
    for arm in arms:
        oms = [v for v in (R[arm]["log"].get("vo_om") or []) if v]
        if not oms or (oms[-1].get("K") or 1) < 2:
            continue
        o(f"      {arm:20} spread(last) {oms[-1]['hold_spread_mean']:.5f}   "
          f"thr(last) {oms[-1]['thr']:.5f}   "
          f"experience rows {oms[-1]['n_rows']} ({oms[-1]['n_hold']} held out)   "
          f"world-graded rows admitted {oms[-1]['n_push_world']}")
    o("")

    # --------------------------------------- [Q] disagreement as a detector of error [sotto]
    o("=" * 100)
    o("[Q] DISAGREEMENT AS A DETECTOR — does the committee's spread find the mirror's own error")
    o("=" * 100)
    o("    `committee_head` read disagreement at 0.716 as a reward-free reader of 'is this")
    o("    learnable', and `ballistic/directed` read it as finding where you LACK DATA rather")
    o("    than where data went stale. A counterfactual is by construction where you lack")
    o("    data, so the question here is whether the spread is sighted where the mean is not:")
    o("    the AUC of the spread as a detector of 'the committee's mean verdict disagrees with")
    o("    the world' on the probes themselves.")
    o("")
    o("    Read off the capped per-probe instrument sample (`vo_probe_rows`), which is the")
    o("    only place the two are stored row by row. `n` is that sample's size, not the run's")
    o("    probe count — the denominator is part of the reading.")
    o("")
    o("    arm                     n   err rate   AUC(spread -> wrong)   AUC(|p-0.5| -> right)")
    for arm in arms:
        rows = R[arm].get("vo_probe_rows") or []
        if not rows:
            o(f"    {arm:20} (no per-probe sample)")
            continue
        sp = np.asarray([z["sp"] for z in rows], float)
        wrong = np.asarray([1.0 if z["ym"] != z["yw"] else 0.0 for z in rows], float)
        conf = np.abs(np.asarray([z["p"] for z in rows], float) - 0.5)
        d1 = _auc(sp, wrong)
        d2 = _auc(conf, 1.0 - wrong)
        o(f"    {arm:20} {len(rows):>6}  {float(wrong.mean()):>8.4f}   "
          f"{('n/a' if d1 is None else f'{d1:.4f}'):>20}   "
          f"{('n/a' if d2 is None else f'{d2:.4f}'):>20}")
    o("")
    o("    THE SAMPLE'S WINDOW MATTERS AND IS PRINTED. Before the reservoir fix the sample")
    o("    filled front-to-back and stopped, so on `so_s1` / `so_s2` every row comes from the")
    o("    first five cycles in which probes fired — a FIVE-CYCLE-OLD mirror, not the run's.")
    o("    Read the between-cycle panel below instead on those tags.")
    for arm in arms:
        rows = R[arm].get("vo_probe_rows") or []
        if rows and "c" in rows[0]:
            cs = sorted({z["c"] for z in rows})
            o(f"      {arm:20} sample cycles {cs[0]}-{cs[-1]} ({len(cs)} distinct of "
              f"{len(R[arm]['log']['cycle'])})")
    o("")
    o("    BETWEEN CYCLES, over the WHOLE run, from the per-cycle mirror counters (every")
    o("    cycle, every probe, no sample): does a cycle in which the committee is more spread")
    o("    have a higher mirror error rate? A different question from the between-probe one —")
    o("    it asks whether the spread tracks the mirror's error as the mirror MATURES rather")
    o("    than which individual counterfactual it is wrong about — and it is the one this")
    o("    tag's logging can answer over its whole life.")
    o("      arm                cycles   rho(spread, err)   AUC(spread -> err>median)   "
      "err first/last")
    for arm in arms:
        ser = []
        for i, v in enumerate((R[arm]["log"].get("vo") or [])):
            mir = (v or {}).get("mirror") or {}
            n = sum(int(c["n"]) for c in mir.values())
            if n < 32:
                continue
            good = sum(int(c["tp"]) + int(c["tn"]) for c in mir.values())
            spd = sum(float(c["sp_sum"]) for c in mir.values()) / n
            ser.append((R[arm]["log"]["cycle"][i], spd, 1.0 - good / n))
        if len(ser) < 8:
            o(f"      {arm:20} (too few cycles with probes)")
            continue
        sp = np.asarray([z[1] for z in ser], float)
        er = np.asarray([z[2] for z in ser], float)
        rho = (float(np.corrcoef(sp, er)[0, 1]) if sp.std() > 0 else float("nan"))
        d = _auc(sp, (er > np.median(er)).astype(float))
        o(f"      {arm:20} {len(ser):>6}   {rho:>16.4f}   "
          f"{('n/a' if d is None else f'{d:.4f}'):>25}   {er[0]:.4f}/{er[-1]:.4f}")
    o("")
    o("    the same, per cycle, for the mirror's own error trajectory (thinned) — so a")
    o("    detector AUC near 0.5 can be read against whether the error moved at all:")
    for arm in arms:
        ser = []
        for i, v in enumerate((R[arm]["log"].get("vo") or [])):
            mir = (v or {}).get("mirror") or {}
            n = sum(int(c["n"]) for c in mir.values())
            if n < 32:
                continue
            good = sum(int(c["tp"]) + int(c["tn"]) for c in mir.values())
            ser.append((R[arm]["log"]["cycle"][i], 1.0 - good / n))
        if not ser:
            continue
        step = max(1, len(ser) // 8)
        o(f"      {arm:20} " + "  ".join(f"c{c}:{e:.3f}" for c, e in ser[::step]))
    o("")
    o("    the same, split by whether the substituted class is in the slot's frequently-")
    o("    written set (`sup`), because that is the axis the blind region is expected on:")
    for arm in arms:
        rows = R[arm].get("vo_probe_rows") or []
        if not rows:
            continue
        for b in (1, 0):
            sub = [z for z in rows if z["sup"] == b]
            if len(sub) < 16:
                continue
            sp = np.asarray([z["sp"] for z in sub], float)
            wrong = np.asarray([1.0 if z["ym"] != z["yw"] else 0.0 for z in sub], float)
            d1 = _auc(sp, wrong)
            o(f"      {arm:20} sup={b}  n={len(sub):>6}  err {float(wrong.mean()):.4f}  "
              f"AUC(spread->wrong) {('n/a' if d1 is None else f'{d1:.4f}')}")
    o("")

    # ------------------------------------------------------------- [R] the blind region [sotto]
    o("=" * 100)
    o("[R] THE BLIND REGION — where the mirror and the world disagree")
    o("=" * 100)
    o("    Per (level, in the slot's frequently-written set): the 2x2 of the mirror's verdict")
    o("    against the world's on the probes, the world's own solve rate there, the mirror's")
    o("    mean probability and the committee's mean spread, and what the agreement rule did")
    o("    (`agree` = the committee concurred; `filed` = the row reached the critic's buffer).")
    o("")
    o("    A mirror trained on the learner's own experience is the same grader in a mirror")
    o("    (`ideas/heterogeneous_graders.md` §8). This table is how far it extends before it")
    o("    goes blind, LOCATED rather than summarised.")
    o("")
    for arm in arms:
        mir = R[arm].get("vo_mirror") or {}
        if not mir:
            continue
        o(f"    {arm}:")
        o("      cell     n     world   mirror   acc     TP    FP    FN    TN    p̄      "
          "spread   agree   filed")
        for k in sorted(mir):
            c = mir[k]
            n = max(1, int(c["n"]))
            o(f"      {k:7} {int(c['n']):>6}  {c['world_solved']/n:.4f}  "
              f"{(c['tp']+c['fp'])/n:.4f}  {(c['tp']+c['tn'])/n:.4f}  "
              f"{int(c['tp']):>5} {int(c['fp']):>5} {int(c['fn']):>5} {int(c['tn']):>5}  "
              f"{c['p_sum']/n:.4f}  {c['sp_sum']/n:.5f}  {c['agree']/n:.4f}  "
              f"{c['filed']/n:.4f}")
        tot = {kk: sum(c[kk] for c in mir.values())
               for kk in ("n", "tp", "fp", "fn", "tn", "agree", "filed", "world_solved")}
        n = max(1, int(tot["n"]))
        o(f"      {'ALL':7} {int(tot['n']):>6}  {tot['world_solved']/n:.4f}  "
          f"{(tot['tp']+tot['fp'])/n:.4f}  {(tot['tp']+tot['tn'])/n:.4f}  "
          f"{int(tot['tp']):>5} {int(tot['fp']):>5} {int(tot['fn']):>5} {int(tot['tn']):>5}"
          f"                    {tot['agree']/n:.4f}  {tot['filed']/n:.4f}")
        o("")

    # ------------------------------------------------- [T] the mirror's gate table [sotto]
    o("=" * 100)
    o("[T] THE MIRROR'S GATE TABLE — M-1r / M-2r / M-3r / M-4r, re-asserted post hoc")
    o("=" * 100)
    o("    The same assertions the preflight ran, re-run here on the paid tag's own arm files.")
    o("    A RAISE below is a failure of the round, not of the reduction.")
    o("")
    for arm in arms:
        r = R[arm]
        if not (r.get("vo_om_mode") or r.get("vo_probe_ground")):
            continue
        try:
            d = _mirror_run_gate(r, arm)
            o(f"    [PASS] {arm:20} {json.dumps(d)}")
        except AssertionError as e:
            o(f"    [FAIL] {arm:20} {str(e)[:400]}")
    o("")

    # ------------------------------------------------------------------- [Y] the yokes [Q3b]
    o("=" * 100)
    o("[Y] THE CLOCK YOKES — realised against planned (gate Y-1's check, post hoc)")
    o("=" * 100)
    o("    A clock yoke replays its source's realised commit and advance CYCLES and decides")
    o("    nothing else. That is the round's whole control: the chooser is read at L4 and L5")
    o("    with the same slots open at the same cycles as the anchor, so a level-resolved")
    o("    readout is not silently comparing two different ladders. Gate Y-1 asserts the match")
    o("    in `preflight`; this is the same check on the paid tag.")
    o("")
    o("    `cancelled` actions are EXCLUDED from the match and printed separately: a replayed")
    o("    commit whose build is empty is cancelled by the substrate, not by the clock. Those")
    o("    are a finding about the arm's BOOK — it had nothing to commit at a cycle the anchor")
    o("    did — and are the one place a yoked arm's ladder can still fall short of its source.")
    o("")
    src = None
    if a.yoke_src and ":" in a.yoke_src:
        st, sa = a.yoke_src.split(":", 1)
        src = load(st, sa)
        o(f"    source: {a.yoke_src}" + ("" if src else "   NOT FOUND"))
    for arm in arms:
        la = [x for x in (R[arm].get("loop_actions") or []) if not x.get("cancelled")]
        canc = [(x["kind"], x["cycle"], x.get("level"))
                for x in (R[arm].get("loop_actions") or []) if x.get("cancelled")]
        gc = sorted(x["cycle"] for x in la if x["kind"] == "commit")
        ga = sorted(x["cycle"] for x in la if x["kind"] == "advance")
        if src is None:
            o(f"    {arm:22} commits {gc}  advances {ga}")
            continue
        sl = [x for x in (src.get("loop_actions") or []) if not x.get("cancelled")]
        pc = sorted(x["cycle"] for x in sl if x["kind"] == "commit")
        pa = sorted(x["cycle"] for x in sl if x["kind"] == "advance")
        o(f"    {arm}")
        o(f"      commits   planned {pc}")
        o(f"                realised {gc}    MATCH = {pc == gc}")
        o(f"      advances  planned {pa}")
        o(f"                realised {ga}    MATCH = {pa == ga}")
        o(f"      cancelled (the book had nothing to commit there): {canc or 'none'}")
        o(f"      commit LEVELS realised: "
          f"{[(e['level'], e['cycle']) for e in R[arm]['events'] if e['kind'] == 'commit']}")
        o(f"      lifetime  source {len(src['log']['cycle'])} cycles / "
          f"{src['log']['t_cum'][-1]:,.0f}g    yoke {len(R[arm]['log']['cycle'])} cycles / "
          f"{R[arm]['log']['t_cum'][-1]:,.0f}g")
        o("")
    o("")

    # ------------------------------------------------------- [Z] the L4/L5 panel [Q3b]
    o("=" * 100)
    o("[Z] THE L4/L5 PANEL — every readout the round turns on, at the cells it turns on")
    o("=" * 100)
    o("    Q3 could not produce this table: no treated arm except Q2's replace-critic had an")
    o("    L4 or L5 slot at all, because each one reached or missed L3 through the commit")
    o("    latch's response to its own L2 writes. With the ladder yoked, every arm has the")
    o("    anchor's slots at the anchor's cycles and the chooser is finally READ where the")
    o("    node is about.")
    o("")
    o("    rep     the written class holds SOME feature that repairs the instance there — the")
    o("            honest measure, and on an open slot it IS the head's class accuracy on")
    o("            held-out data (fresh instances every cycle).")
    o("    cont    it holds the CLEAN derivation's latent — a stricter and less forgiving read;")
    o("            the gap between the two is the chooser spreading over classes the clean")
    o("            derivation did not carry (`DESIGN.md` §23's correction).")
    o("    dist    distinct tuples written, `top1` the share of filed writes on one tuple.")
    o("    moved   the composed argmax was not the DP's row (composed arms only).")
    o("")
    o("    [sotto] THE BANKED FLOOR AND CEILING ARE IN THIS TABLE, marked `(banked)`, so the")
    o("    three mirror arms are read beside the composed chooser on the FILED diet (the floor)")
    o("    and on the WORLD-graded probe (the ceiling) without leaving the page. A pooled")
    o("    L2/L3 panel follows it, which is the cell that replicated across seeds in `voicing`")
    o("    and the one a frontier number should not be read ahead of.")
    o("")
    for arm in list(arms) + list(B):
        log = (R[arm] if arm in R else B[arm])["log"]
        cells = {}
        for v in log["vo"]:
            if not v:
                continue
            for src_k, dst in (("rep", "rep"), ("contains", "cont"), ("var", "var"),
                               ("moved", "moved")):
                for cell, z in (v.get(src_k) or {}).items():
                    if not cell[0].isdigit() or int(cell.split("n")[0]) < 4:
                        continue
                    c = cells.setdefault(cell, collections.Counter())
                    if dst in ("rep", "cont"):
                        c[dst + "_n"] += z["n"]; c[dst + "_hit"] += z["hit"]
                    elif dst == "var":
                        c["var_n"] += z["n"]; c["dist"] += z["distinct"]
                        c["top1w"] += z["top1"] * z["n"]; c["cyc"] += 1
                        c["rows"] = z["rows"]
                    else:
                        c["mv_n"] += z["n"]; c["moved"] += z["moved"]
        if not cells:
            o(f"    {arm}: no L4/L5 cell was ever written.")
            o("")
            continue
        o(f"    {arm}:" + ("   (banked)" if arm in B else ""))
        o("      cell    rows   writes    rep          cont         dist  top1    moved")
        for cell in sorted(cells, key=lambda z: (int(z.split("n")[0]), int(z.split("n")[1]))):
            c = cells[cell]
            rp = (f"{c['rep_hit']/c['rep_n']:.3f} ({c['rep_n']:>6})" if c["rep_n"]
                  else "     -        ")
            ct = (f"{c['cont_hit']/c['cont_n']:.3f} ({c['cont_n']:>6})" if c["cont_n"]
                  else "     -        ")
            mv = (f"{c['moved']/c['mv_n']:.3f}" if c["mv_n"] else "  -  ")
            o(f"      {cell:6} {c['rows']:>6} {c['var_n']:>8}   {rp}  {ct}  "
              f"{c['dist']/max(c['cyc'],1):>5.1f} {c['top1w']/max(c['var_n'],1):>6.3f}   {mv}")
        o("")

    o("    [sotto] THE POOLED L2/L3 PANEL — the cell `voicing` replicated on every seed that")
    o("    produced those rungs (+18.5 / +24.5 / +19.3% relative for the composed chooser on")
    o("    the filed diet). `rep` pooled over every L2 and L3 cell, with the denominator.")
    o("")
    o("      arm                        rep (L2/L3 pooled)        n        vs floor")
    _pool = {}
    for arm in list(arms) + list(B):
        log = (R[arm] if arm in R else B[arm])["log"]
        n = h = 0
        for v in log["vo"]:
            if not v:
                continue
            for cell, z in (v.get("rep") or {}).items():
                if cell[0].isdigit() and int(cell.split("n")[0]) in (2, 3):
                    n += z["n"]; h += z["hit"]
        if n:
            _pool[arm] = (h / n, n)
    _floor = next((k for k in _pool if k.endswith("vo_s3b:voi3b_comp_yk")
                   or k.endswith("voi3b_comp_yk")), None)
    _fv = _pool[_floor][0] if _floor else None
    for arm, (r_, n_) in sorted(_pool.items(), key=lambda z: -z[1][0]):
        rel = (f"{(r_ - _fv) / _fv * 100:+7.1f}%" if _fv else "      -")
        o(f"      {arm:26} {r_:.4f}          {n_:>8}   {rel}"
          + ("   (banked)" if arm in B else ""))
    o("")

    # ------------------------------------------------------------- [S] the seed table [Q3d]
    if a.seeds:
        o("=" * 100)
        o("[S] THE SEED TABLE — what the ladder did, and WHY each advance fired")
        o("=" * 100)
        o("    Q3c turned the node's seed check into a seed check on the ANCHOR's own climb:")
        o("    at seed 0 it committed L2/L3/L4/L5 over 201 cycles, at seed 1 it committed L2")
        o("    and nothing else and ended at c108. The difference is not in the commit rule")
        o("    alone — it is in the ADVANCE. The two policies are split: the commit owner reads")
        o("    `yield` per level, the advance owner reads `dsil`. An era left by `quiet` is the")
        o("    thermostat CHOOSING to move on; an era left by `cap` is the schedule forcing it.")
        o("    An arm that advances by `quiet` before its commit owner has licensed the rung")
        o("    walks up the ladder without earning it, and every rung above is then unreachable.")
        o("    So `why` is printed for every advance, beside how far into the era it fired.")
        o("")
        o("    Nothing here is an average over seeds. Four draws of one character is a spread,")
        o("    not an estimate, and the table is printed so the spread can be seen.")
        o("")
        for spec in [z for z in a.seeds.split(",") if z.strip()]:
            st, sa, lab = (spec.split(":") + ["", ""])[:3]
            r = load(st, sa)
            if r is None:
                o(f"    seed {lab or '?'} ({st}:{sa}): NOT FOUND")
                continue
            log = r["log"]
            la = [x for x in (r.get("loop_actions") or []) if not x.get("cancelled")]
            ev = [(e["level"], e["cycle"], e.get("n_entries"))
                  for e in r["events"] if e["kind"] == "commit"]
            caps = list(r.get("config", {}).get("era_caps") or ())
            o(f"    seed {lab or '?'}  ({st}:{sa})   {len(log['cycle'])} cycles   "
              f"t_cum {log['t_cum'][-1]:,.0f}g")
            o(f"      ladder reached : "
              + (" ".join(f"L{l}@c{c}({n})" for l, c, n in ev) or "(nothing committed)"))
            o(f"      commits        : " + (" | ".join(
                f"c{x['cycle']} L{x['level']} by {x['why']} "
                f"(owner={x.get('owner')} V={x.get('owner_V')!s:.7} "
                f"tol={x.get('owner_v_tol')!s:.7} mult={x.get('owner_v_mult')!s:.5})"
                for x in la if x["kind"] == "commit") or "(none)"))
            o(f"      advances       : " + (" | ".join(
                f"c{x['cycle']} era{x['era']} by {x['why'].upper()} "
                f"at {x['c_in_era']}/{caps[x['era']-1] if x['era']-1 < len(caps) else '?'}"
                for x in la if x["kind"] == "advance") or "(none)"))
            nq = sum(1 for x in la if x["kind"] == "advance" and x["why"] == "quiet")
            nc = sum(1 for x in la if x["kind"] == "advance" and x["why"] == "cap")
            o(f"                       {nq} by quiet / {nc} by cap")
            cert = [(i + 1, c) for i, c in enumerate(log.get("cert") or [])]
            first = {}
            for cyc_i, c in cert:
                if c and c.get("run"):
                    first.setdefault(int(c["run"]), log["cycle"][cyc_i - 1])
            o(f"      certificate    : " + (r.get("shadow_cert") and "" or "")
              + (", ".join(f"run{k}@c{v}" for k, v in sorted(first.items())) or "(never fired)"))
            # THE A3 SERIES THROUGH ERA 2 — the audition accuracy of the L3 candidate table,
            # which is the quantity the commit licence ultimately rides on and the one the run's
            # own cycle line prints as `A3=`. Printed as a series rather than summarised,
            # because the question is whether it was still MOVING when the era was left: the
            # thermostat's rule is "hold while it still moves, act when it quiets inside its
            # measured dead zone", so an era left early by `quiet` on a wandering gauge and an
            # era left by `cap` on a settled one are different failures.
            for lvl, era in ((3, 2), (4, 3)):
                aa = [(log["cycle"][i], (log["aud"][i] or {}).get(str(lvl), {}).get("cand"))
                      for i in range(len(log["cycle"]))
                      if log["era"][i] == era and i < len(log["aud"])]
                vs = [v for _, v in aa if v is not None]
                if not vs:
                    o(f"      era-{era} A{lvl}       : (no reading — the era was never entered)")
                    continue
                o(f"      era-{era} A{lvl}       : n={len(vs)} range {min(vs):.3f}-{max(vs):.3f} "
                  f"first {vs[0]:.3f} last {vs[-1]:.3f}")
                o("                       series: "
                  + " ".join(f"{v:.2f}" for v in vs[::max(1, len(vs) // 14)]))
            o("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
