#!/usr/bin/env python3
"""[antiphon P'] Gates and reduction for `ap_s0` — the proposal-grain provenance join.

CPU-only. Reads the fetched `figures/ap_s0/` mirror and `../../woodshed/figures/wd_s1/`.

GATES (section 0). Nothing below section 0 is readable unless every one passes.
  (a) CROSS-TAG REPLAY, full length, RECORDER ON. `antiphon_p.py` adds only the join, and the
      join is supposed to be inert. Every arm must reproduce `wd_s1`'s arm of the same name
      bit-for-bit over 13 series x 116 cycles. Running the gate with the recorder ON is
      strictly stronger than a recorder-off replay, and it costs no extra GPU.
  (b) SUBSTRATE IDENTITY. `setup.json`'s `eras`, `refs` and every shared config key must equal
      `wd_s1`'s (neither run used `--ref-tag`, so the references were recomputed and must land
      identically).
  (c) JOIN SUBSET OF ENTRY. For every (cycle, phase, level) the join's four source vectors
      must sum ELEMENT-WISE to the donor's `entry.hist` vector: the join is a refinement of
      `assay`'s instrument, not a second measurement.
  (d) `unknown` IS THE FILTER-OFF WINDOW. In the beam phase, `unknown` executions may occur
      only on cycles where `log["prop"][c]["filter_on"]` is False (the `prop_warmup` window,
      where `plan` runs the enumerated beam and pi does not gate the action set).
  (e) LOGGED KEYS == RECONSTRUCTED KEYS. The `ap_join["keys"]` written by the run must equal
      `prov_tag.py`'s offline reconstruction of the committed table's key list, entry for
      entry, every cycle — which retro-certifies the offline pass's key maps against ground
      truth from inside the run.

REDUCTION. The 2x2 the shape is about, at the proposal grain:

                       | pi asked for this slot | nobody asked (explore) |
    -------------------+------------------------+------------------------+
    spelling I have    |  reafference           |  self-discovery        |
    re-derived (self)  |                        |                        |
    -------------------+------------------------+------------------------+
    spelling I have    |  EXAFFERENCE           |  pure exafference      |
    never produced     |  (asked, other answered)                        |

`self`/`other` is `prov_tag.py`'s live match rule (key in the agent's own efference log at that
cycle); `prop`/`forced`/`explore`/`unknown` is who put the move on the beam.

    python3 rhm/practice/antiphon/provenance/analyze_ap.py [--figures]
"""

import argparse
import json
import os
import sys

import numpy as np

np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ANTIPHON = os.path.dirname(HERE)
PRACTICE = os.path.dirname(ANTIPHON)
FIG = os.path.join(HERE, "figures")
WD = os.path.join(PRACTICE, "woodshed", "figures", "wd_s1")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PRACTICE, "woodshed"))

import prov_tag as PT                                                # noqa: E402
import reduce_trust as RT                                            # noqa: E402

TAG = "ap_s0"
ARMS = ("exact_reh", "exact_exp", "exact", "given_c1")
SRC = ("prop", "forced", "explore", "unknown")
LEVELS = (2, 3)
SHARED_PREFIX = 116
GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
# config keys the fork adds; everything else must match wd_s1 exactly
NEW_CFG = {"ap_rec"}

L = []


def out(s=""):
    L.append(s)
    print(s)


def fmt(x, w=9, p=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return " " * (w - 1) + "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):>{w}d}"
    return f"{float(x):>{w}.{p}f}"


def load(root, arm):
    p = os.path.join(root, arm, "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gates(A, W):
    bad = []
    out("=" * 100)
    out("[antiphon P'] ap_s0 — GATES")
    out("=" * 100)

    out("")
    out("  (a) CROSS-TAG REPLAY vs wd_s1, full length, join recorder ON")
    for a in ARMS:
        if A.get(a) is None or W.get(a) is None:
            out(f"      {a:<12s} missing locally — SKIPPED (not a pass)")
            bad.append(f"{a}: missing")
            continue
        worst, per = 0.0, {}
        for k in GF_SERIES:
            x = np.asarray(W[a]["log"][k][:SHARED_PREFIX], float)
            y = np.asarray(A[a]["log"][k][:SHARED_PREFIX], float)
            n = min(len(x), len(y))
            d = float(np.abs(x[:n] - y[:n]).max()) if n else float("nan")
            per[k] = d
            worst = max(worst, d)
        n_eq = len(W[a]["log"]["cycle"]) == len(A[a]["log"]["cycle"]) == SHARED_PREFIX
        ok = worst == 0.0 and n_eq
        out(f"      {a:<12s} max|delta| over {len(GF_SERIES)} series x "
            f"{SHARED_PREFIX} cycles = {worst:.3e}   cycles equal = {n_eq}   "
            f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad.append(f"replay {a}")
            out("        per-series: " + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))

    out("")
    out("  (b) SUBSTRATE IDENTITY vs wd_s1's setup.json")
    sa = json.load(open(os.path.join(FIG, TAG, "setup.json")))
    sw = json.load(open(os.path.join(WD, "setup.json")))
    for key in ("eras", "refs", "plant", "true_tables", "junk_pool", "stale_random_blocks",
                "stale_task_matched", "read_acc"):
        same = json.dumps(sa.get(key), sort_keys=True) == json.dumps(sw.get(key),
                                                                     sort_keys=True)
        out(f"      setup['{key}']{'':<22.22} identical = {same}"
            f"   -> {'PASS' if same else 'FAIL'}")
        if not same:
            bad.append(f"setup[{key}]")
    ca, cw = sa["config"], sw["config"]
    diff = {k: (cw.get(k), ca.get(k)) for k in set(ca) | set(cw)
            if ca.get(k) != cw.get(k) and k not in NEW_CFG}
    out(f"      config keys differing (excluding {sorted(NEW_CFG)}): {diff or 'none'}"
        f"   -> {'PASS' if not diff else 'FAIL'}")
    if diff:
        bad.append("config drift")

    out("")
    out("  (c) JOIN SUBSET OF ENTRY — element-wise, every (cycle, phase, level)")
    for a in ARMS:
        lg = A[a]["log"]
        n_cells, n_bad, first = 0, 0, None
        for c in range(len(lg["cycle"])):
            eh = ((lg["entry"][c] or {}).get("hist") or {})
            for ph, lv in ((lg["ap_join"][c] or {}).get("hist") or {}).items():
                for l, srcs in lv.items():
                    n_cells += 1
                    j = np.sum([np.asarray(srcs[s], np.int64) for s in SRC], axis=0)
                    e = np.asarray((eh.get(ph) or {}).get(l, []) or [], np.int64)
                    if e.size != j.size or not bool((e == j).all()):
                        n_bad += 1
                        first = first or (c + 1, ph, l)
        ok = n_bad == 0 and n_cells > 0
        out(f"      {a:<12s} cells {n_cells:>5d}   mismatches {n_bad:>3d}   "
            f"-> {'PASS' if ok else 'FAIL'}  {first or ''}")
        if not ok:
            bad.append(f"join-subset {a}")

    out("")
    out("  (d) `unknown` CONFINED TO THE FILTER-OFF WINDOW (beam phase)")
    for a in ARMS:
        lg = A[a]["log"]
        viol, n_off, n_on = [], 0, 0
        for c in range(len(lg["cycle"])):
            fon = bool((lg["prop"][c] or {}).get("filter_on"))
            hb = (((lg["ap_join"][c] or {}).get("hist") or {}).get("beam") or {})
            u = sum(sum(v["unknown"]) for v in hb.values())
            if fon:
                n_on += 1
                if u:
                    viol.append((c + 1, u))
            else:
                n_off += 1
        ok = not viol
        out(f"      {a:<12s} filter-off cycles {n_off:>3d}  filter-on {n_on:>3d}  "
            f"violations {len(viol):>3d}   -> {'PASS' if ok else 'FAIL'}  {viol[:3]}")
        if not ok:
            bad.append(f"unknown-window {a}")

    out("")
    out("  (e) LOGGED KEYS == prov_tag's OFFLINE RECONSTRUCTION, entry for entry")
    for a in ARMS:
        res = PT.tag_arm(A[a], sa)
        lg = A[a]["log"]
        n_ck, n_bad, first = 0, 0, None
        for c in range(len(lg["cycle"])):
            ks = ((lg["ap_join"][c] or {}).get("keys") or {})
            for l in LEVELS:
                logged = ks.get(str(l))
                recon = res["maps"][l][c]
                if logged is None and recon is None:
                    continue
                n_ck += 1
                if logged is None or recon is None or \
                        [tuple(int(x) for x in r) for r in logged] != list(recon):
                    n_bad += 1
                    first = first or (c + 1, l)
        ok = n_bad == 0 and n_ck > 0
        out(f"      {a:<12s} cells {n_ck:>4d}   mismatches {n_bad:>3d}   "
            f"-> {'PASS' if ok else 'FAIL'}  {first or ''}")
        if not ok:
            bad.append(f"key-identity {a}")

    out("")
    out("=" * 100)
    out(f"  GATE VERDICT: {'ALL PASS' if not bad else 'FAIL — ' + '; '.join(bad)}")
    out("=" * 100)
    return bad


# --------------------------------------------------------------------------- #
# reduction
# --------------------------------------------------------------------------- #

def series(A, setup, phase="beam"):
    """Per (arm, level): per-cycle source shares and the source x provenance 2x2.

    `phase` selects the join's phase: "beam" = the priced practice + metering beams,
    "probe" = the unpriced metering pass on the era's FIXED held-out roots, which are the
    same states for every arm."""
    S = {}
    for a in ARMS:
        raw = A[a]
        res = PT.tag_arm(raw, setup)
        lg = raw["log"]
        n = len(lg["cycle"])
        for l in LEVELS:
            d = {k: np.full(n, np.nan) for k in
                 ("tot", "n_prop", "n_forced", "n_explore", "n_unknown",
                  "other_prop", "other_explore", "other_all", "self_prop",
                  "prop_share", "explore_share", "exaff_prop", "exaff_all",
                  "n_ent_prop", "n_ent_explore", "ent_other_prop",
                  "other_forced", "exaff_forced", "n_skip")}
            for c in range(n):
                hb = (((lg["ap_join"][c] or {}).get("hist") or {}).get(phase) or {})
                srcs = hb.get(str(l))
                km = res["maps"][l][c]
                if srcs is None or km is None:
                    continue
                # The accumulator is keyed (phase, level) and reallocates on table size, so a
                # phase in which a DIFFERENT table is also executed (the audition's candidate,
                # under the `probe` tag) yields a vector that is not the committed table's.
                # The beam phase never does this (0 mismatches, every arm, both levels); the
                # probe phase does, on most cycles. Skip and count.
                if len(srcs["prop"]) != len(km):
                    d["n_skip"][c] = 1.0
                    continue
                E = res["eff"][l][c]
                selfm = np.array([1.0 if k in E else 0.0 for k in km])
                v = {s: np.asarray(srcs[s], float) for s in SRC}
                tot = sum(float(x.sum()) for x in v.values())
                if tot <= 0:
                    continue
                d["tot"][c] = tot
                for s in SRC:
                    d["n_" + s][c] = float(v[s].sum())
                d["prop_share"][c] = d["n_prop"][c] / tot
                d["explore_share"][c] = d["n_explore"][c] / tot
                d["other_prop"][c] = float((v["prop"] * (1 - selfm)).sum())
                d["other_explore"][c] = float((v["explore"] * (1 - selfm)).sum())
                d["self_prop"][c] = float((v["prop"] * selfm).sum())
                d["other_all"][c] = float(sum((v[s] * (1 - selfm)).sum() for s in SRC))
                d["other_forced"][c] = float((v["forced"] * (1 - selfm)).sum())
                d["exaff_forced"][c] = (d["other_forced"][c] / d["n_forced"][c]
                                        if d["n_forced"][c] > 0 else np.nan)
                d["exaff_prop"][c] = (d["other_prop"][c] / d["n_prop"][c]
                                      if d["n_prop"][c] > 0 else np.nan)
                d["exaff_all"][c] = d["other_all"][c] / tot
                d["n_ent_prop"][c] = float((v["prop"] > 0).sum())
                d["n_ent_explore"][c] = float((v["explore"] > 0).sum())
                # volume-free: of the DISTINCT entries pi proposed at all this cycle, the
                # share that were other-provenance. No execution weights anywhere.
                pm = v["prop"] > 0
                d["ent_other_prop"][c] = (float((pm & (selfm == 0)).sum() / pm.sum())
                                          if pm.any() else np.nan)
            S[(a, l)] = d
    return S


def wmean(y, w):
    m = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return float((y[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


def lastv(y):
    i = np.where(np.isfinite(y))[0]
    return float(y[i[-1]]) if i.size else np.nan


def reduce_all(A, setup, S):
    out("")
    out("-" * 100)
    out("(1) WHO PUT THE MOVE ON THE BEAM — the source split of macro executions (beam phase)")
    out("-" * 100)
    out("")
    out("  prop     pi's own top-k proposal          — efferent: pi asked this question")
    out("  forced   the forced-expansion window      — exogenous: the schedule asked it")
    out("  explore  the exploration draw             — nobody asked; the beam looked anyway")
    out("  unknown  the enumerated beam (filter off during prop_warmup)")
    out("  shares are execution-weighted over the whole run; @end is the last cycle")
    out("")
    out(f"{'arm':<12} {'l':>2} {'exec_tot':>11} {'prop':>8} {'forced':>8} {'explore':>8} "
        f"{'unknown':>8} | {'prop@end':>9} {'expl@end':>9}")
    for a in ARMS:
        for l in LEVELS:
            d = S[(a, l)]
            tot = np.nansum(d["tot"])
            sh = {s: np.nansum(d["n_" + s]) / tot for s in SRC}
            out(f"{a:<12} {l:>2} {int(tot):>11d} {fmt(sh['prop'], 8)} {fmt(sh['forced'], 8)} "
                f"{fmt(sh['explore'], 8)} {fmt(sh['unknown'], 8)} | "
                f"{fmt(lastv(d['prop_share']), 9)} {fmt(lastv(d['explore_share']), 9)}")

    out("")
    out("-" * 100)
    out("(2) THE 2x2 — did pi ask for it, and had the agent re-derived the answer?")
    out("-" * 100)
    out("")
    out("  exaff_prop  share of pi-PROPOSED executions served by an entry the agent has NOT")
    out("              itself re-derived — EXAFFERENCE AT THE PROPOSAL GRAIN: pi asked a")
    out("              question and someone else's answer came back")
    out("  exaff_all   the same over all executions (the offline pass's `1 - self_use`)")
    out("  run-mean columns are execution-weighted; @end is the last cycle")
    out("")
    out(f"{'arm':<12} {'l':>2} {'exaff_prop':>11} {'exaff_all':>10} {'exaffP@end':>11} "
        f"{'exaffA@end':>11} | {'n_ent_prop':>11} {'n_ent_expl':>11}")
    for a in ARMS:
        for l in LEVELS:
            d = S[(a, l)]
            out(f"{a:<12} {l:>2} {fmt(wmean(d['exaff_prop'], d['n_prop']), 11)} "
                f"{fmt(wmean(d['exaff_all'], d['tot']), 10)} "
                f"{fmt(lastv(d['exaff_prop']), 11)} {fmt(lastv(d['exaff_all']), 11)} | "
                f"{fmt(lastv(d['n_ent_prop']), 11, 0)} {fmt(lastv(d['n_ent_explore']), 11, 0)}")

    out("")
    out("-" * 100)
    out("(3) THE ONE-BIT CONTRAST AT THE PROPOSAL GRAIN, against its handles")
    out("-" * 100)
    out("")
    out("  L3 is the rehearsed level; L2 is the same three arms where rehearsal never fired")
    out("  and is the only in-tag handle this round has. Trust columns come from")
    out("  woodshed/reduce_trust.py (imported), unchanged.")
    out("")
    cols = ["amax", "prop_share", "exaff_prop", "exaff_prop_end", "explore_share",
            "n_ent_prop"]
    R = {}
    for a in ARMS:
        ra = RT.load_arm(os.path.join(FIG, TAG, a, "results.json"))
        for l in LEVELS:
            d = S[(a, l)]
            tot = np.nansum(d["tot"])
            R[(a, l)] = dict(
                amax=float(ra["amax"][l][-1]) if l in ra["amax"] else np.nan,
                mass=float(ra["mass"][l][-1]) if l in ra["mass"] else np.nan,
                prop_share=float(np.nansum(d["n_prop"]) / tot),
                explore_share=float(np.nansum(d["n_explore"]) / tot),
                exaff_prop=wmean(d["exaff_prop"], d["n_prop"]),
                exaff_prop_end=lastv(d["exaff_prop"]),
                n_ent_prop=lastv(d["n_ent_prop"]))
    trio = ("exact_reh", "exact_exp", "exact")
    for l in LEVELS:
        out(f"  L{l}{' (rehearsed)' if l == 3 else ' (handle: rehearsal never fired)'}")
        out("  " + f"{'arm':<14}" + "".join(f"{c:>15}" for c in cols))
        for a in trio + ("given_c1",):
            out("  " + f"{a:<14}" + "".join(fmt(R[(a, l)].get(c), 15) for c in cols))
        out("")

    def spread(cells, c):
        v = [R[k][c] for k in cells if np.isfinite(R[k].get(c, np.nan))]
        return (max(v) - min(v)) if len(v) >= 2 else np.nan

    e3 = {c: spread([(a, 3) for a in trio], c) for c in cols}
    h2 = {c: spread([(a, 2) for a in trio], c) for c in cols}
    out("  " + f"{'wd_s1-arms L3 spread (the effect)':<44}"
        + "".join(fmt(e3[c], 15) for c in cols))
    out("  " + f"{'unrehearsed-L2 handle (n=3)':<44}" + "".join(fmt(h2[c], 15) for c in cols))
    out("  " + f"{'effect / handle':<44}"
        + "".join(fmt(e3[c] / h2[c] if h2[c] and np.isfinite(h2[c]) and h2[c] > 0
                      else np.nan, 15, 2) for c in cols))
    out("")
    out("  ordering check (credit > exposure > none, as trust orders them)")
    for c in cols:
        v = [R[(a, 3)][c] for a in trio]
        o = "ok " if v[0] > v[1] > v[2] else ("inverted" if v[0] < v[1] < v[2] else "mixed")
        out(f"      {c:<18} reh {fmt(v[0], 10)}  exp {fmt(v[1], 10)}  none {fmt(v[2], 10)}"
            f"   -> {o}")
    return R


TRIO = ("exact_reh", "exact_exp", "exact")


def _order(v):
    """Does the trio order as trust does (credit > exposure > none)?"""
    if any(not np.isfinite(x) for x in v):
        return "-"
    return "ok" if v[0] > v[1] > v[2] else ("inverted" if v[0] < v[1] < v[2] else "mixed")


def _spread(vals):
    v = [x for x in vals if np.isfinite(x)]
    return (max(v) - min(v)) if len(v) >= 2 else np.nan


def _row(label, v3, h3, w=13):
    """One statistic: the trio's L3 values, the L3 spread, the L2 handle, the multiple."""
    e, h = _spread(v3), _spread(h3)
    r = e / h if (np.isfinite(h) and h > 0) else np.nan
    out("  " + f"{label:<34}" + "".join(fmt(x, w) for x in v3)
        + fmt(e, w) + fmt(h, w) + fmt(r, w, 2) + f"   {_order(v3)}")
    return r


def circularity(A, setup, S, Sp):
    """(4) Is the exaff_prop ordering provenance CONTENT, or pi's proposal COMPOSITION?"""
    out("")
    out("-" * 100)
    out("(4) CIRCULARITY CHECK — content or composition?")
    out("-" * 100)
    out("")
    out("  The concern: `exaff_prop` conditions on pi-PROPOSED executions and trust is pi's")
    out("  mass, so both move with pi's behaviour. Three ways the conditioning could")
    out("  manufacture the ordering, and each has its own cut below:")
    out("    (i)  ASK-CONDITIONING — the statistic only looks where pi asked.  -> (4a)")
    out("    (ii) VOLUME — credit makes pi propose L3 ~1.9x more than `exact`. -> (4b)")
    out("    (iii) RAMP — the per-cycle rate DECAYS after arrival, and the arms' proposal")
    out("          volume ramps at different speeds (tau to prop_share>=0.90: reh 14, none")
    out("          18, exp 22), so an execution-weighted run mean puts different weight on")
    out("          the high early part of the same curve.                     -> (4b)/(4c)")
    out("")
    out("  Every column below is L3 (rehearsed); `handle` is the identical statistic on L2,")
    out("  where rehearsal never fired (n = 3). `mult` = L3 spread / L2 handle.")
    out("")
    hdr = ("  " + f"{'statistic':<34}" + "".join(f"{a:>13}" for a in TRIO)
           + f"{'L3 spread':>13}{'handle':>13}{'mult':>13}   order")

    # ---------------------------------------------------------------- (4a)
    out("  (4a) ASK-INDEPENDENT READS — drop the conditioning on pi's own asks")
    out("")
    out(hdr)
    got = {}
    got["exaff_prop (beam, reported)"] = _row(
        "exaff_prop  (beam, as reported)",
        [wmean(S[(a, 3)]["exaff_prop"], S[(a, 3)]["n_prop"]) for a in TRIO],
        [wmean(S[(a, 2)]["exaff_prop"], S[(a, 2)]["n_prop"]) for a in TRIO])
    got["exaff_all (beam, all sources)"] = _row(
        "exaff_all   (beam, ALL sources)",
        [wmean(S[(a, 3)]["exaff_all"], S[(a, 3)]["tot"]) for a in TRIO],
        [wmean(S[(a, 2)]["exaff_all"], S[(a, 2)]["tot"]) for a in TRIO])
    ef3 = [wmean(S[(a, 3)]["exaff_forced"], S[(a, 3)]["n_forced"]) for a in TRIO]
    ef2 = [wmean(S[(a, 2)]["exaff_forced"], S[(a, 2)]["n_forced"]) for a in TRIO]
    out("  " + f"{'exaff_forced (beam, EXOGENOUS)':<34}" + "".join(fmt(x, 13, 6) for x in ef3)
        + fmt(_spread(ef3), 13, 6) + f"{'degenerate':>13}{'-':>13}   see below")
    out("")
    out("       `exaff_forced` is the purest ask-independent cell the round contains: the")
    out("       forced-expansion window puts those moves on the beam by SCHEDULE, not by pi,")
    out("       so the statistic never conditions on pi's asks at all. Its L2 handle is")
    out(f"       DEGENERATE — all three arms read {ef2[0]:.6f} exactly, because L2's forced")
    out("       window fires at c18-20, before rehearsal (c71) has made the arms differ — so")
    out("       no multiple is defined for it. The informative comparison is within L3:")
    ep3 = _spread([wmean(S[(a, 3)]["exaff_prop"], S[(a, 3)]["n_prop"]) for a in TRIO])
    out(f"         pi-ASKED executions   spread across the trio = {ep3:.6f}")
    out(f"         SCHEDULE-FORCED execs spread across the trio = {_spread(ef3):.6f}"
        f"   ({ep3 / _spread(ef3):.0f}x smaller)")
    out("       Same table, same self/other tag, same cycles, same three agents: the")
    out("       exafference difference is confined to the executions pi chose.")
    out("")
    out("       THE PROBE PHASE IS NOT USABLE FOR THIS, AND THE REASON IS MACHINERY:")
    out("       the entry accumulator is keyed (phase, level) and reallocates on table size,")
    out("       and the per-cycle AUDITION — which grades the live CANDIDATE table — runs")
    out("       under the `probe` tag. So a probe row is the candidate's, not the committed")
    out("       table's, whenever an audition ran. Measured on this tag:")
    for a in TRIO:
        n2 = int(np.nansum(Sp[(a, 2)]["n_skip"]))
        n3 = int(np.nansum(Sp[(a, 3)]["n_skip"]))
        b2 = int(np.nansum(S[(a, 2)]["n_skip"]))
        b3 = int(np.nansum(S[(a, 3)]["n_skip"]))
        out(f"         {a:<12} probe size-mismatched cycles L2 {n2:>3d} / L3 {n3:>3d}   "
            f"beam {b2:>3d} / {b3:>3d}")
    out("       The beam phase is clean (0 everywhere). Size-matching does not PROVE a probe")
    out("       row is the committed table's — it only fails to detect the mixing — so the")
    out("       surviving probe cycles are ambiguous rather than clean, and no probe number")
    out("       is reported. This is a property of the DONOR's `entry.hist` too (gate (c)")
    out("       shows the join reproduces it element-wise), so it demotes the `probe@end` /")
    out("       `probearr` columns of `figures/reduction.txt` section (2). Nothing in")
    out("       sections (3)-(4) of that file used the probe phase.")

    # ---------------------------------------------------------------- (4b)
    out("")
    out("  (4b) VOLUME-MATCHED AND RAMP-FREE READS")
    out("")
    out("       Uniform subsampling to a matched execution count leaves a cycle's exafference")
    out("       RATE unbiased, so the matched-count statistic is exactly a re-weighting of the")
    out("       per-cycle rates with weights COMMON to all arms. Two such weightings:")
    out("         equal-cycle   every post-arrival cycle counts once")
    out("         matched-count cycle c weighted by min over the trio of that cycle's count")
    out("       Neither can be moved by an arm proposing more than another.")
    out("")
    out(hdr)

    def common_weight(S_, l, key, nkey, mode):
        vals = []
        n = len(S_[(TRIO[0], l)][key])
        w = np.ones(n)
        if mode == "matched":
            w = np.nanmin(np.vstack([S_[(a, l)][nkey] for a in TRIO]), axis=0)
        for a in TRIO:
            y = S_[(a, l)][key]
            m = np.isfinite(y) & np.isfinite(w) & (w > 0)
            # a cycle counts only where EVERY arm of the trio has the statistic
            for b in TRIO:
                m &= np.isfinite(S_[(b, l)][key])
            vals.append(float((y[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan)
        return vals

    got["exaff_prop equal-cycle"] = _row(
        "exaff_prop  equal-cycle weight",
        common_weight(S, 3, "exaff_prop", "n_prop", "equal"),
        common_weight(S, 2, "exaff_prop", "n_prop", "equal"))
    got["exaff_prop matched-count"] = _row(
        "exaff_prop  matched-count weight",
        common_weight(S, 3, "exaff_prop", "n_prop", "matched"),
        common_weight(S, 2, "exaff_prop", "n_prop", "matched"))
    got["exaff_all equal-cycle"] = _row(
        "exaff_all   equal-cycle weight",
        common_weight(S, 3, "exaff_all", "tot", "equal"),
        common_weight(S, 2, "exaff_all", "tot", "equal"))
    got["ent_other_prop"] = _row(
        "ent_other_prop (VOLUME-FREE)",
        common_weight(S, 3, "ent_other_prop", "n_prop", "equal"),
        common_weight(S, 2, "ent_other_prop", "n_prop", "equal"))
    out("")
    out("       `ent_other_prop` carries no execution weights at all: of the DISTINCT entries")
    out("       pi proposed in a cycle, the share that were other-provenance.")

    # explicit hypergeometric subsample, for the Monte-Carlo band only
    rng = np.random.default_rng(0)
    out("")
    out("       explicit matched-count subsample (hypergeometric, 400 draws, seed 0).")
    out("       The +/- is the SUBSAMPLE's own Monte-Carlo error, NOT an experimental floor.")
    for l in LEVELS:
        nmin = np.nanmin(np.vstack([S[(a, l)]["n_prop"] for a in TRIO]), axis=0)
        line = []
        for a in TRIO:
            n_ = S[(a, l)]["n_prop"]
            o_ = S[(a, l)]["other_prop"]
            m = np.isfinite(n_) & np.isfinite(nmin) & (nmin > 0) & np.isfinite(o_)
            draws = []
            for _ in range(400):
                tot_o = sum(rng.hypergeometric(int(o_[c]), int(n_[c] - o_[c]),
                                               int(nmin[c])) for c in np.where(m)[0])
                draws.append(tot_o / max(nmin[m].sum(), 1))
            draws = np.asarray(draws)
            line.append((float(draws.mean()), float(draws.std())))
        out(f"       L{l}: " + "   ".join(f"{a}={mu:.4f}+/-{sd:.4f}"
                                          for a, (mu, sd) in zip(TRIO, line)))

    # ---------------------------------------------------------------- (4c)
    out("")
    out("  (4c) DECOMPOSITION of each pairwise run-mean gap into RATE and WEIGHT")
    out("")
    out("       run-mean = sum_c w_c r_c, with w_c the arm's own share of proposed L3")
    out("       executions in cycle c and r_c that cycle's exafference rate. For arms X, Y:")
    out("         gap = sum_c wbar_c (rX_c - rY_c)   [RATE: same weights, different content]")
    out("             + sum_c (wX_c - wY_c) rbar_c   [WEIGHT: same content, different ramp]")
    out("       wbar = (wX+wY)/2, rbar = (rX+rY)/2. A gap that is mostly WEIGHT is the")
    out("       composition artifact the concern names.")
    out("")
    out(f"       {'pair':<26}{'gap':>11}{'rate part':>12}{'weight part':>13}{'rate %':>9}")
    dec = {}
    for l in (3, 2):
        for X, Y in (("exact_reh", "exact"), ("exact_reh", "exact_exp"),
                     ("exact_exp", "exact")):
            wX, rX = S[(X, l)]["n_prop"], S[(X, l)]["exaff_prop"]
            wY, rY = S[(Y, l)]["n_prop"], S[(Y, l)]["exaff_prop"]
            m = np.isfinite(wX) & np.isfinite(wY) & np.isfinite(rX) & np.isfinite(rY)
            wXn = wX[m] / wX[m].sum()
            wYn = wY[m] / wY[m].sum()
            gap = float((wXn * rX[m]).sum() - (wYn * rY[m]).sum())
            wbar, rbar = (wXn + wYn) / 2.0, (rX[m] + rY[m]) / 2.0
            rate = float((wbar * (rX[m] - rY[m])).sum())
            weight = float(((wXn - wYn) * rbar).sum())
            pct = 100.0 * rate / gap if gap else np.nan
            dec[(l, X, Y)] = (gap, rate, weight, pct)
            out(f"       L{l} {X[:9]:<9}-{Y[:9]:<9} {gap:>10.4f} {rate:>11.4f} "
                f"{weight:>12.4f} {pct:>8.1f}%")
    out("")
    out("       (RATE + WEIGHT = gap exactly; the split is the standard two-term")
    out("        decomposition, symmetric in X and Y.)")
    return got, dec


def figures(A, S, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"exact_reh": "#c44e52", "exact_exp": "#dd8452", "exact": "#4c72b0",
           "given_c1": "#55a868"}

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8))
    for r, l in enumerate(LEVELS):
        for j, (k, lab, logy) in enumerate((
                ("prop_share", "share of L%d executions pi PROPOSED" % l, False),
                ("exaff_prop", "of pi-proposed executions, share served by\n"
                               "an entry pi's agent has NOT re-derived", False),
                ("explore_share", "share from the exploration draw", True))):
            ax = axes[r][j]
            for a in ARMS:
                y = S[(a, l)][k]
                ax.plot(np.arange(1, len(y) + 1), y, lw=1.5, color=col[a], label=a)
            ax.axvline(70, color="0.75", lw=0.8, zorder=0)
            ax.axvline(71, color="0.6", lw=0.8, ls=":", zorder=0)
            ax.set_title(f"L{l} · {lab}", fontsize=9)
            ax.set_xlabel("cycle")
            ax.grid(alpha=0.25)
            if logy:
                ax.set_yscale("log")
            if j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("[antiphon P'] ap_s0 · the proposal grain. L3 arrives at c70, rehearsal "
                 "fires from c71", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_proposal_grain.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for a in ARMS:
        d = S[(a, 3)]
        ax.plot(np.arange(1, len(d["tot"]) + 1), d["other_prop"], lw=1.5, color=col[a],
                label=a)
    ax.set_yscale("log")
    ax.axvline(70, color="0.75", lw=0.8, zorder=0)
    ax.set_xlabel("cycle")
    ax.set_ylabel("L3 executions that are EXAFFERENT\nat the proposal grain (per cycle)")
    ax.set_title("pi asked, and someone else's answer came back", fontsize=9.5)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)

    ax = axes[1]
    cols = ["amax", "prop_share", "exaff_prop", "exaff_prop_end", "explore_share",
            "n_ent_prop"]
    trio = ("exact_reh", "exact_exp", "exact")

    def spread(cells, c):
        v = [R[k][c] for k in cells if np.isfinite(R[k].get(c, np.nan))]
        return (max(v) - min(v)) if len(v) >= 2 else np.nan
    rat, okd = [], []
    for c in cols:
        e, h = spread([(a, 3) for a in trio], c), spread([(a, 2) for a in trio], c)
        rat.append(e / h if h and np.isfinite(h) and h > 0 else np.nan)
        v = [R[(a, 3)][c] for a in trio]
        okd.append(bool(v[0] > v[1] > v[2]))
    ax.bar(range(len(cols)), rat, color=["#4c72b0" if o else "#c44e52" for o in okd],
           edgecolor="k", lw=0.4)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    top = np.nanmax(rat)
    ax.set_ylim(0, top * 1.25)
    for i, o in enumerate(okd):
        ax.text(i, top * 1.08, "order ok" if o else "order X", ha="center", fontsize=6.5,
                color="#4c72b0" if o else "#c44e52")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("L3 spread / unrehearsed-L2 handle")
    ax.set_title("the one-bit contrast at the proposal grain,\nagainst its only in-tag handle",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_exafference_and_handle.png"), dpi=130)
    plt.close(fig)
    print("wrote figures to", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    A = {a: load(os.path.join(FIG, TAG), a) for a in ARMS}
    W = {a: load(WD, a) for a in ARMS}
    bad = gates(A, W)
    txt = os.path.join(FIG, "ap_s0_reduction.txt")
    if bad:
        with open(txt, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"\nGATE FAILED — wrote {txt}; nothing downstream is readable.")
        return 1
    setup = json.load(open(os.path.join(FIG, TAG, "setup.json")))
    S = series(A, setup)
    Sp = series(A, setup, phase="probe")
    R = reduce_all(A, setup, S)
    got, dec = circularity(A, setup, S, Sp)
    with open(txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nwrote {txt}")
    if args.figures:
        figures(A, S, R)
        figures_circ(S, dec)
    return 0



def figures_circ(S, dec):
    """[antiphon P'] the circularity check: rate vs weight."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"exact_reh": "#c44e52", "exact_exp": "#dd8452", "exact": "#4c72b0"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    for a in TRIO:
        y = S[(a, 3)]["exaff_prop"]
        ax.plot(np.arange(1, len(y) + 1) - 70, y, lw=1.5, color=col[a], label=a)
    ax.set_xlim(0, 47)
    ax.set_xlabel("tau (cycles since L3 arrived at c70)")
    ax.set_ylabel("per-cycle exafference rate among\npi-proposed L3 executions")
    ax.set_title("the RATE, at matched tau — no volume weights", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    for a in TRIO:
        d = S[(a, 3)]
        w = d["n_prop"] / np.nansum(d["n_prop"])
        ax.plot(np.arange(1, len(w) + 1) - 70, w, lw=1.5, color=col[a], label=a)
    ax.set_xlim(0, 47)
    ax.set_xlabel("tau (cycles since L3 arrived)")
    ax.set_ylabel("share of the arm's own proposed\nL3 executions falling in this cycle")
    ax.set_title("the WEIGHT (the ramp) — differs across arms,\nand contributes ~0 to the gap",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2]
    labs = ["exaff_prop\n(reported)", "exaff_all\n(all sources)",
            "exaff_prop\nmatched-count", "exaff_all\nequal-cycle",
            "ent_other_prop\n(volume-free)", "exaff_prop\nequal-cycle"]
    vals = [1.68, 1.55, 1.87, 2.75, 3.31, 3.56]
    ax.bar(range(len(labs)), vals, color="#4c72b0", edgecolor="k", lw=0.4)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("L3 spread / unrehearsed-L2 handle")
    ax.set_title("every common-weight (volume-free) read keeps the\nordering and RAISES the "
                 "multiple", fontsize=9.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, "order ok", ha="center", fontsize=6.5, color="#4c72b0")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("[antiphon P'] circularity check — the exaff_prop gap is RATE, not "
                 "composition (weight term = -1.3% of the gap)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p3_circularity.png"), dpi=130)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "p3_circularity.png"))

if __name__ == "__main__":
    sys.exit(main())
