"""[sostenuto] THE REDUCTION — the admission set on the live build, read off the arm files.

Local and CPU-only: `fetch_compact.py --tag <tag> --fetch --replace` first, then this against
`figures/<tag>/<arm>/results.json`, with the banked no-walk reference (`sb_sv_yk`) read from
`../soundboard/figures/<bank tag>/sb_sv_yk/results.json`.

WHAT IT REPORTS, in the order the brief asks for it:

  [A] the gates, run-level, off the arm files (A-1r .. A-4r, A-9r, and the in-loop A-1c/A-4/A-5)
  [B] per arm and per level, RESTRICTED TO THAT LEVEL'S OWN ERA: the admitted table's error
      on the disjoint test pool against the UNGATED live build's error on the SAME pool at the
      SAME pass — the paired, no-walk counterfactual measured in-arm. The pool is drawn on the
      era's own cell, so a level-l macro's audition is only comparable inside the era whose
      cell is at level l; pooling over eras reads the CELL MOVING UP as the table getting worse.
  [B2] the same rows at MATCHED CYCLES across arms, so admitted-vs-live is compared on the
      same cycle and not on each arm's own pass schedule
  [C] the junk share admitted — precision and recall of the admitted table against the true one
  [D] the per-candidate confusion against the world gate (pp5's TT/TF/FT/FF), per arm and level
  [E] the deep-era task error, cycle for cycle, against the banked `sb_sv_yk`
  [F] the readout's held-out series
  [G] the fallbacks, the world reads billed to each arm's decisions, and the bill
  [H] the cost split: the walk's own wall clock against the cycle's and the container's
  [I] THE COVERAGE TRACE: every at-support key offered, by era, with its decision; and for
      every L3-L5 key on the ungated arm, whether its halves were keys the world arm REFUSED
      (the demand-blindness test)
  [J] the deep-era window, cycle for cycle, beside the commits and the advances

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/aliquot/sostenuto/reduce_sostenuto.py --tag st_s1
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
BANK = os.path.join(os.path.dirname(HERE), "soundboard", "figures")
GATE_OF = {"st_gw_yk": "world", "st_gr_yk": "read", "st_gn_yk": "none",
           "st_gy_yk": "yield", "st_gd_yk": "demand"}                  # [sostenuto r2]
ORDER = ("st_gw_yk", "st_gr_yk", "st_gn_yk", "st_gy_yk", "st_gd_yk")
R2 = ("st_gy_yk", "st_gd_yk")                                         # [sostenuto r2]
# Okabe-Ito blue / vermillion / bluish green / reddish purple / orange: validated (lightness
# band, chroma floor, CVD separation, normal-vision floor, contrast) on the light chart surface.
# The no-walk reference is NOT a sixth category — it is a recessive dashed rule with a label.
COL = {"st_gw_yk": "#0072B2", "st_gr_yk": "#D55E00", "st_gn_yk": "#009E73",
       "st_gy_yk": "#CC79A7", "st_gd_yk": "#E69F00"}
REF_COL = "#8a8a86"
LAB = {"st_gw_yk": "world gate", "st_gr_yk": "read gate", "st_gn_yk": "no gate",
       "st_gy_yk": "yield gate", "st_gd_yk": "demand gate"}


def load(tag, arm, root=None):
    p = os.path.join(root or os.path.join(FIG, tag), arm, "results.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def passes(a):
    """[(cycle, level_row), ...] flattened, in cycle order."""
    adm = (a or {}).get("adm") or {}
    out = []
    for ps in adm.get("passes") or []:
        for row in ps.get("levels") or []:
            out.append((int(ps["cycle"]), row, float(ps.get("t_s") or 0.0)))
    return out


def era_windows(r):
    """{era index (1-based) -> (first cycle, last cycle, level, node)} off the arm's own log."""
    er = np.asarray(r["log"]["era"], int)
    out = {}
    for i, e in enumerate(r["eras"], start=1):
        w = np.nonzero(er == i)[0]
        if w.size:
            out[i] = (int(w[0]) + 1, int(w[-1]) + 1, int(e["level"]), int(e["node"]))
    return out


def own_era(r, lv):
    """The era whose CELL is at level `lv` — the only window in which a level-`lv` macro's
    audition on that era's cell is comparable with another pass's."""
    for i, e in enumerate(r["eras"], start=1):
        if int(e["level"]) == int(lv):
            return i
    return None


def half_keys(fk):
    """A key's two halves, as flat keys of the level below."""
    fk = tuple(int(z) for z in fk)
    h = len(fk) // 2
    return (fk[:h], fk[h:])


def decisions(r):
    """{(level, flat key) -> row} for every candidate the arm ever offered."""
    out = {}
    for ps in (r.get("adm") or {}).get("passes") or []:
        for lr in ps["levels"]:
            for st in lr["steps"]:
                out[(int(lr["level"]), tuple(int(z) for z in st["key"]))] = {
                    "cycle": int(ps["cycle"]), "era": int(lr["era"]),
                    "count": int(st["count"]), "admit": bool(st["admit_gate"]),
                    "admit_world": bool(st["admit_world"]), "unfit": bool(st["unfit"]),
                    "n_changed": st.get("n_changed"), "why": st.get("why")}
    return out


def fmt(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def section(lines, title):
    lines.append("")
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)


def reduce_tag(tag, bank_tag, bank_arm, out_txt, mk_figs=True, with_tags=()):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "..")))
    # [sostenuto r2] arms may come from more than one tag: round 2's two arms live in their own
    # tag and round 1's three are read from theirs, NOT re-run. The first tag holding an arm
    # wins, and where each arm came from is printed at the top of the reduction.
    arm_tag = {}
    arms = {}
    for t_ in (tag,) + tuple(with_tags):
        for a in ORDER:
            if a not in arms:
                r_ = load(t_, a)
                if r_ is not None:
                    arms[a], arm_tag[a] = r_, t_
    arms = {a: arms[a] for a in ORDER if a in arms}
    assert arms, f"no arm of {tag} found under {FIG}"
    ref = load(bank_tag, bank_arm, root=os.path.join(BANK, bank_tag))
    L = []
    L.append(f"[sostenuto] {tag} — THE ADMISSION SET ON THE LIVE BUILD")
    L.append(f"  arms {list(arms)}   no-walk reference {bank_tag}/{bank_arm}"
             f"{'' if ref is not None else '  (NOT FOUND)'}")
    L.append("  read from: " + "  ".join(f"{a}<-{arm_tag[a]}" for a in arms))

    # ---------------------------------------------------------------- [A] the gates -----
    section(L, "[A] THE GATES, RUN-LEVEL, OFF THE ARM FILES")
    try:
        from rhm.practice.voicing.sotto_voce.aliquot.sostenuto.sostenuto import sos_gate_run
    except Exception as e:                                       # pragma: no cover
        sos_gate_run = None
        L.append(f"    (could not import the gate block: {e!r})")
    for a, r in arms.items():
        if sos_gate_run is None:
            break
        # [sostenuto r2] a red run-level gate is REPORTED, with its message, not allowed to
        # take the reduction down with it: the arm's numbers are still the record.
        try:
            g = sos_gate_run(r, a)
        except AssertionError as e_:
            L.append(f"    {a:<10} FAIL — {e_}")
            continue
        L.append(f"    {a:<10} PASS")
        for k in ("A-1r:n_passes", "A-1r:n_offered", "A-2r:conf", "A-4r:max_pool_overlap",
                  "A-9r:n_world_billed", "A-1c(repro vs readgate.gated_walk)",
                  "A-4(read handed the world's level)",
                  "A-5(a constant level admits everything)", "n_unfit",
                  "n_pending_serve", "n_admitted_empty", "n_empty_fallback",
                  "n_rekey_collide"):
            L.append(f"      {k:<44} {g.get(k)}")
        rep = ((r.get("adm") or {}).get("repro") or {}).get("counterfactual")
        if rep:
            L.append(f"      counterfactual on the repro pass: " + "  ".join(
                f"{k}: adm {v['n_admitted']}/{v['n_admitted'] + v['n_rejected']} "
                f"e {v['e_final']:.4f}" for k, v in rep.items()))

    # ------------------------------------------------- [B] the quality column -----------
    section(L, "[B] THE ADMITTED TABLE'S TEST-POOL ERROR — PAIRED, and INSIDE THE LEVEL'S OWN ERA")
    L.append("    The gate and test pools are drawn on the ERA'S OWN CELL, so a level-l macro's")
    L.append("    audition is only comparable with another pass's inside the era whose cell is")
    L.append("    at level l. Pooling over eras reads the CELL MOVING UP as the table getting")
    L.append("    worse: the late-pass rises to 0.8-0.97 are a level-2 macro being auditioned on")
    L.append("    an era-4 cell, not a level-2 table that has decayed. The comparison of record")
    L.append("    is therefore PAIRED and WITHIN-PASS: `admitted - live` is the same plant, the")
    L.append("    same pool and the same cycle, so it isolates the gate.")
    L.append("")
    era_w = {a: era_windows(r) for a, r in arms.items()}
    for a, r in arms.items():
        L.append(f"    {a}  era windows " + "  ".join(
            f"e{i}[{w[0]}-{w[1]}]L{w[2]}n{w[3]}" for i, w in sorted(era_w[a].items())))
    L.append("")
    L.append(f"    {'arm':<10} {'L':>2} {'era':>4} {'window':>9} {'n':>3} {'adm-live':>9}"
             f" {'median':>9} {'<0':>3} {'=0':>3} {'>0':>3} {'e_test(adm)':>12} {'e(live)':>9}")
    series = {}
    for a, r in arms.items():
        rows = passes(r)
        ow = era_w[a]
        for lv in sorted({q[1]["level"] for q in rows}):
            sel_all = [q for q in rows if q[1]["level"] == lv]
            series[(a, lv)] = ([q[0] for q in sel_all],
                               np.array([q[1]["e_test_admitted"] for q in sel_all], float),
                               np.array([q[1]["e_test_live"] for q in sel_all], float))
            ei = own_era(r, lv)
            for scope, sel in (("own", [q for q in sel_all if ei and q[1]["era"] == ei]),
                               ("all", sel_all)):
                if not sel:
                    L.append(f"    {a:<10} {lv:>2} {(str(ei) if scope == 'own' else 'all'):>4}"
                             f" {'—':>9} {0:>3}" + "  (no pass in that era)")
                    continue
                ea = np.array([q[1]["e_test_admitted"] for q in sel], float)
                el = np.array([q[1]["e_test_live"] for q in sel], float)
                d = ea - el
                w = (f"{ow[ei][0]}-{ow[ei][1]}" if (scope == "own" and ei in ow) else "c1-end")
                L.append(f"    {a:<10} {lv:>2} {(str(ei) if scope == 'own' else 'all'):>4}"
                         f" {w:>9} {len(d):>3} {d.mean():>+9.4f} {np.median(d):>+9.4f}"
                         f" {int((d < 0).sum()):>3} {int((d == 0).sum()):>3}"
                         f" {int((d > 0).sum()):>3} {ea.mean():>12.4f} {el.mean():>9.4f}")
        L.append("")
    L.append("    `none` is identically 0.0000 in every cell by construction — its admitted")
    L.append("    table IS the live build — which is the reduction's own consistency check.")

    # ------------------------------------------------ [B2] at matched cycles ------------
    section(L, "[B2] THE SAME ROWS AT MATCHED CYCLES, across arms")
    L.append("    A pass fires only where a level has an undecided at-support key, so the arms")
    L.append("    run passes on different cycles. These are the (cycle, level) cells where at")
    L.append("    least two arms both ran one: the plants have diverged by then, but the pool")
    L.append("    is drawn on the same seed family and the cycle is the same.")
    L.append("")
    cells = {}
    for a, r in arms.items():
        for cy, row, _ in passes(r):
            cells.setdefault((cy, row["level"]), {})[a] = row
    shared_cells = {k: v for k, v in cells.items() if len(v) >= 2}
    L.append(f"    {'cyc':>4} {'L':>2} " + " ".join(
        f"{LAB[a] + ' adm/live':>22}" for a in ORDER if a in arms))
    for (cy, lv) in sorted(shared_cells):
        v = shared_cells[(cy, lv)]
        L.append(f"    {cy:>4} {lv:>2} " + " ".join(
            (f"{v[a]['e_test_admitted']:.4f}/{v[a]['e_test_live']:.4f}"
             f" ({v[a]['n_kept_rows']}/{v[a]['n_live_rows']})").rjust(22)
            if a in v else "—".rjust(22) for a in ORDER if a in arms))
    L.append(f"    {len(shared_cells)} shared (cycle, level) cells of "
             f"{len(cells)} distinct cells over the three arms")

    # ------------------------------------------------------- [C] the junk share ---------
    section(L, "[C] WHAT THE ADMITTED TABLE HOLDS — precision, recall, junk share")
    L.append(f"    {'arm':<10} {'L':>2} {'rows(adm)':>9} {'rows(live)':>10} {'share':>6}"
             f" {'prec':>6} {'junk':>6} {'recall':>7} {'keys adm':>8} {'keys rej':>8}")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        rows = passes(r)
        for lv in sorted({q[1]["level"] for q in rows}):
            last = [q[1] for q in rows if q[1]["level"] == lv][-1]
            pr = last.get("tab_precision")
            L.append(f"    {a:<10} {lv:>2} {last['n_kept_rows']:>9} {last['n_live_rows']:>10}"
                     f" {(last['n_kept_rows'] / max(1, last['n_live_rows'])):>6.2f}"
                     f" {fmt(pr, 3):>6} {fmt(None if pr is None else 1 - pr, 3):>6}"
                     f" {fmt(last.get('tab_recall'), 3):>7}"
                     f" {len(adm.get('admitted', {}).get(str(lv), [])):>8}"
                     f" {len(adm.get('rejected', {}).get(str(lv), [])):>8}")

    # ------------------------------------------- [D] the per-candidate confusion --------
    section(L, "[D] THE PER-CANDIDATE CONFUSION AGAINST THE WORLD GATE (pp5's table)")
    L.append("    TT admitted and the world agrees · TF admitted where the world would refuse")
    L.append("       (THE JUNK LET IN) · FT refused where the world would admit (THE GOOD")
    L.append("       ENTRIES REFUSED) · FF refused and the world agrees.")
    L.append("    pp5, offline, at the loop's budget: world admits ~95% of what it is offered,")
    L.append("    so raw agreement is not evidence; the composition is. `read` there: TF 0.022")
    L.append("    FT 0.050 against ungated's TF 0.045 FT 0.000.")
    L.append("")
    L.append("")
    L.append("    THE READOUT IS NOT FIT BEFORE ITS FIRST REFIT (`vo_om_min` 512 filed rows).")
    L.append("    Until then `VoProjBank.predict` answers 0.5 for every row, the pooled")
    L.append("    difference is exactly 0 and the strict rule admits — so those trials are the")
    L.append("    NO-GATE arm's decisions and a read-vs-world comparison on them is vacuous.")
    L.append("    They are marked `[unfit]` and the read's confusion is restated without them.")
    L.append("")
    L.append(f"    {'arm':<10} {'L':>3} {'n':>5} {'TT':>5} {'TF':>5} {'FT':>5} {'FF':>5}"
             f" {'agree':>6} {'TF rate':>8} {'FT rate':>8} {'unfit':>6}")
    for a, r in arms.items():
        rows = passes(r)
        for lv in sorted({q[1]["level"] for q in rows}) + ["all", "fit", "deep"]:
            if lv == "deep":
                sel_steps = [st for _, lr, _ in rows for st in lr["steps"]
                             if lr["level"] >= 3]
            elif lv == "fit":
                sel_steps = [st for _, lr, _ in rows for st in lr["steps"]
                             if not st["unfit"]]
            else:
                sel_steps = [st for _, lr, _ in rows for st in lr["steps"]
                             if lv == "all" or lr["level"] == lv]
            if not sel_steps:
                continue
            c = {k: 0 for k in ("TT", "TF", "FT", "FF")}
            for st in sel_steps:
                c["TT" if (st["admit_gate"] and st["admit_world"])
                  else "TF" if st["admit_gate"]
                  else "FT" if st["admit_world"] else "FF"] += 1
            n = sum(c.values())
            uf = sum(1 for st in sel_steps if st["unfit"])
            ltag = (f"{lv}" if isinstance(lv, int) else lv)     # NOT `tag`: that is the
            if uf == n and n:                                     # function's parameter, and
                ltag += "*"                                       # shadowing it emptied [H]
            L.append(f"    {a:<10} {ltag:>3} {n:>5} {c['TT']:>5} {c['TF']:>5} {c['FT']:>5}"
                     f" {c['FF']:>5} {(c['TT'] + c['FF']) / n:>6.3f} {c['TF'] / n:>8.3f}"
                     f" {c['FT'] / n:>8.3f} {uf:>6}"
                     + ("   [unfit: the no-gate arm's decisions]" if uf == n and n else ""))
        L.append("")
    L.append("    row `fit`  = every trial the readout was fit for (includes 4 late L2 ones);")
    L.append("    row `deep` = L3-L5 alone, which is where the read gate ever had a live")
    L.append("                 decision, since L2 was decided before the readout's first fit.")
    L.append("    The `world` arm's agreement is 1.000 by construction — it IS the world gate —")
    L.append("    and its FF column is the only informative one there (what the oracle refused).")

    # ---- the read gate's own signal, on the trials where it had one -------------------- #
    section(L, "[D2] THE READ GATE'S OWN SIGNAL, on the trials where it had one")
    L.append("    `p_trial - p_cur` is a difference of two pooled levels over the SAME fired")
    L.append("    gate pool. A candidate the executor's DP never picks changes no instance, so")
    L.append("    the difference is exactly 0 and the strict rule admits — which is the world")
    L.append("    gate's own behaviour on an invisible candidate (`e_trial == e_cur`, test `<=`).")
    L.append("")
    L.append(f"    {'arm':<10} {'n(fit)':>7} {'d=0':>5} {'d>0':>5} {'d<0':>5} {'min d':>9}"
             f" {'max d':>9} {'median |d|':>11} {'median n_changed':>17}")
    for a, r in arms.items():
        st = [q for _, lr, _ in passes(r) for q in lr["steps"]
              if (not q["unfit"]) and q["p_trial"] is not None]
        if not st:
            continue
        d = np.array([q["p_trial"] - q["p_cur"] for q in st])
        nc = [q["n_changed"] for q in st if q["n_changed"] is not None]
        L.append(f"    {a:<10} {len(d):>7} {int((d == 0).sum()):>5} {int((d > 0).sum()):>5}"
                 f" {int((d < 0).sum()):>5} {d.min():>+9.5f} {d.max():>+9.5f}"
                 f" {np.median(np.abs(d)):>11.5f}"
                 f" {(np.median(nc) if nc else float('nan')):>17.1f}")

    # ------------------------------------------------------ [E] the task error ----------
    section(L, "[E] THE TASK ERROR, CYCLE FOR CYCLE, against the banked no-walk reference")
    L.append(f"    {'arm':<10} " + " ".join(f"{'era' + str(i):>7}" for i in range(1, 6))
             + f" {'cycles':>7}")
    tab = {}
    for a, r in list(arms.items()) + ([(bank_arm + " (bank)", ref)] if ref else []):
        e = np.asarray(r["log"]["e"], float)
        er = np.asarray(r["log"]["era"], int)
        tab[a] = [float(e[er == i].mean()) if (er == i).any() else None for i in range(1, 6)]
        L.append(f"    {a:<10} " + " ".join(f"{fmt(q, 4):>7}" for q in tab[a])
                 + f" {len(e):>7}")
    if ref is not None:
        L.append("")
        L.append("    delta against the reference, per era (negative = the walk arm is better)")
        base = tab[bank_arm + " (bank)"]
        for a in arms:
            L.append(f"    {a:<10} " + " ".join(
                f"{(tab[a][i] - base[i]):>+7.4f}" if (tab[a][i] is not None
                                                      and base[i] is not None) else f"{'—':>7}"
                for i in range(5)))
    if "st_gn_yk" in tab:
        # [sostenuto r2] THE NONE ARM'S OWN FOOTPRINT, netted out. `st_gn_yk` admits everything
        # and still differs from the no-walk reference (a new key is served only from the
        # next pass, up to `recert_every` cycles late), so a gate's effect is read against it.
        L.append("")
        L.append("    delta against the NONE arm (st_gn_yk), per era — the gate net of the op's")
        L.append("    own footprint (negative = the gate arm is better than admitting everything)")
        bn = tab["st_gn_yk"]
        for a in arms:
            if a == "st_gn_yk":
                continue
            L.append(f"    {a:<10} " + " ".join(
                f"{(tab[a][i] - bn[i]):>+7.4f}" if (tab[a][i] is not None
                                                    and bn[i] is not None) else f"{'—':>7}"
                for i in range(5)))

    # -------------------------------------------------- [F] the readout's series --------
    section(L, "[F] THE READOUT'S HELD-OUT SERIES (the organ the `read` arm gates with)")
    L.append(f"    {'arm':<20} {'first refit':>11} {'hold_auc@end':>12} {'mean(last 50)':>14}"
             f" {'ECE@end':>8} {'n_refit':>8}")
    for a, r in list(arms.items()) + ([(bank_arm + " (bank)", ref)] if ref else []):
        om = [q for q in r["log"]["vo_om"] if isinstance(q, dict) and q.get("n_refit")]
        au = [q.get("hold_auc") for q in om if q.get("hold_auc") is not None]
        first = next((i for i, q in enumerate(r["log"]["vo_om"])
                      if isinstance(q, dict) and q.get("n_refit")), None)
        L.append(f"    {a:<20} {str(first):>11} {fmt(au[-1] if au else None):>12}"
                 f" {fmt(float(np.mean(au[-50:])) if au else None):>14}"
                 f" {fmt(om[-1].get('hold_ece') if om else None, 3):>8}"
                 f" {(om[-1].get('n_refit') if om else None)!s:>8}")

    # ------------------------------------------------------ [G] the bill ----------------
    section(L, "[G] THE BILL, THE FALLBACKS, AND THE WORLD READS BILLED TO THE DECISIONS")
    L.append(f"    {'arm':<10} {'gate':>6} {'offered':>8} {'admit':>6} {'reject':>7}"
             f" {'auditions':>10} {'world billed':>13} {'pending':>8} {'adm-empty':>10}"
             f" {'fallback':>9} {'rekey':>6} {'t_cum':>12}")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        tc = r["log"]["t_cum"][-1] if r["log"]["t_cum"] else None
        L.append(f"    {a:<10} {str(adm.get('gate')):>6} {adm.get('n_offered', 0):>8}"
                 f" {adm.get('n_admit', 0):>6} {adm.get('n_reject', 0):>7}"
                 f" {adm.get('n_auditions', 0):>10} {adm.get('n_world_billed', 0):>13}"
                 f" {sum((adm.get('n_pending_serve') or {}).values()):>8}"
                 f" {sum((adm.get('n_admitted_empty') or {}).values()):>10}"
                 f" {sum((adm.get('n_empty_fallback') or {}).values()):>9}"
                 f" {adm.get('n_rekey_collide', 0):>6} {fmt(tc, 0):>12}")
    if ref is not None:
        L.append(f"    {bank_arm + ' (bank)':<10} {'—':>6} {'—':>8} {'—':>6} {'—':>7}"
                 f" {'—':>10} {'—':>13} {'—':>8} {'—':>10} {'—':>9} {'—':>6}"
                 f" {fmt(ref['log']['t_cum'][-1], 0):>12}")

    # ------------------------------------------------------ [H] the cost ----------------
    section(L, "[H] WHERE THE WALL CLOCK WENT")
    sm = {}
    for t_ in reversed([tag] + list(with_tags)):         # [sostenuto r2] the arm's own tag wins
        smp = os.path.join(FIG, t_, "summary.json")
        if os.path.isfile(smp):
            with open(smp) as fh:
                sm_ = json.load(fh)
            for k_ in ("elapsed_s_per_container", "cycle_seconds", "peak_rss_mib"):
                sm.setdefault(k_, {}).update({a_: v_ for a_, v_ in (sm_.get(k_) or {}).items()
                                              if arm_tag.get(a_) == t_})
            if t_ == tag:
                sm["elapsed_s"] = sm_.get("elapsed_s")
    bsm = {}
    bsmp = os.path.join(BANK, bank_tag, "summary.json")
    if os.path.isfile(bsmp):
        with open(bsmp) as fh:
            bsm = json.load(fh)
    L.append(f"    {'arm':<10} {'container s':>12} {'s/cycle':>9} {'walk t_s':>9}"
             f" {'passes':>7} {'walk share':>11} {'peak RSS MiB':>13}")
    for a, r in arms.items():
        adm = r.get("adm") or {}
        ts = [float(p.get("t_s") or 0.0) for p in adm.get("passes") or []]
        el = (sm.get("elapsed_s_per_container") or {}).get(a)
        cs = (sm.get("cycle_seconds") or {}).get(a)
        rss = (sm.get("peak_rss_mib") or {}).get(a)
        # a `--merge-only` re-merge cannot recover the container's own elapsed (it comes from
        # the live call's return), so fall back to `s/cycle x cycles` and MARK it: that is the
        # cycle loop alone and excludes the ~500 s shared setup each container also pays.
        est = ""
        if el is None and cs is not None:
            el = float(cs) * len(r["log"]["e"])
            est = "*"
        L.append(f"    {a:<10} {(fmt(el, 0) + est):>12} {fmt(cs, 2):>9} {sum(ts):>9.1f}"
                 f" {len(ts):>7} {(sum(ts) / float(el) if el else float('nan')):>11.5f}"
                 f" {fmt(rss, 0):>13}")
    for a in ((bsm.get("cycle_seconds") or {})):
        L.append(f"    {bank_tag + ':' + a:<10} "
                 f"{fmt((bsm.get('elapsed_s_per_container') or {}).get(a), 0):>12} "
                 f"{fmt(bsm['cycle_seconds'][a], 2):>9} {'—':>9} {'—':>7} {'—':>11} "
                 f"{fmt((bsm.get('peak_rss_mib') or {}).get(a), 0):>13}")
    L.append("")
    _els = [q for q in (sm.get("elapsed_s_per_container") or {}).values() if q]
    L.append(f"    sweep wall clock {fmt(sm.get('elapsed_s'), 0)} s against the longest arm's "
             f"{fmt(max(_els) if _els else None, 0)} s. A `*` marks a container time")
    L.append("    ESTIMATED as s/cycle x cycles because a `--merge-only` re-merge cannot recover")
    L.append("    the live call's own elapsed; it is the cycle loop alone and excludes the")
    L.append("    ~500 s shared setup. Where both are real, the gap between the sweep's wall")
    L.append("    clock and the longest arm is Modal's container scheduling (the arms are")
    L.append("    `.starmap`ed and do not all start at once), not work. The walk's own is the")
    L.append("    `walk t_s` column and is a ten-thousandth of the arm; what the op actually")
    L.append("    costs is in `s/cycle`, because the served table changes the beam's and the")
    L.append("    DP's work every cycle. The per-era head dumps are 5 x 2.1 MB written five")
    L.append("    times per arm and the final refit is one solve, both far below a second.")

    # ---------------------------------------------- [I] the coverage trace ---------------
    section(L, "[I] THE COVERAGE TRACE — every key offered, and the demand-blindness test")
    dec = {a: decisions(r) for a, r in arms.items()}
    L.append(f"    {'arm':<10} {'L':>2} {'era':>4} {'offered':>8} {'admitted':>9}"
             f" {'refused':>8} {'refused keys (count at decision)':>34}")
    for a, r in arms.items():
        for lv in sorted({k[0] for k in dec[a]}):
            byera = {}
            for (l_, fk), d in dec[a].items():
                if l_ == lv:
                    byera.setdefault(d["era"], []).append((fk, d))
            for e_ in sorted(byera):
                ds = byera[e_]
                # NOT `ref`: that is the banked reference arm, and shadowing it broke the
                # figures (the same class of defect as the `tag` shadowing, DESIGN §7.4).
                refused = [f"{d['count']}" for _, d in ds if not d["admit"]]
                L.append(f"    {a:<10} {lv:>2} {e_:>4} {len(ds):>8}"
                         f" {sum(1 for _, d in ds if d['admit']):>9}"
                         f" {len(refused):>8}"
                         f" {('[' + ','.join(refused) + ']' if refused else ''):>34}")
    L.append("")
    L.append("    THE DEMAND-BLINDNESS TEST. `spiral`/`census` found a gate that prices a")
    L.append("    candidate by PRESENT demand removes the halves the level above will need.")
    L.append("    For every L3-L5 key that the ungated arm ever supported, is at least one of")
    L.append("    its two halves a key the GATE arm REFUSED at the level below?")
    L.append("")
    if "st_gn_yk" in dec and "st_gw_yk" in dec:
        L.append(f"    {'gate arm':<10} {'L':>2} {'none-arm keys':>14} {'half refused':>13}"
                 f" {'half never offered':>19} {'half admitted':>14}")
        for ga in [q for q in ORDER if q in dec and q != "st_gn_yk"]:   # [sostenuto r2]
            ref_by_lv = {lv: {fk for (l_, fk), d in dec[ga].items()
                              if l_ == lv and not d["admit"]}
                         for lv in (2, 3, 4)}
            adm_by_lv = {lv: {fk for (l_, fk), d in dec[ga].items()
                              if l_ == lv and d["admit"]}
                         for lv in (2, 3, 4)}
            for lv in (3, 4, 5):
                keys = [fk for (l_, fk) in dec["st_gn_yk"] if l_ == lv]
                nref = nnev = nadm = 0
                for fk in keys:
                    hs = half_keys(fk)
                    if any(h in ref_by_lv.get(lv - 1, set()) for h in hs):
                        nref += 1
                    elif all(h in adm_by_lv.get(lv - 1, set()) for h in hs):
                        nadm += 1
                    else:
                        nnev += 1
                if keys:
                    L.append(f"    {ga:<10} {lv:>2} {len(keys):>14} {nref:>13}"
                             f" {nnev:>19} {nadm:>14}")
        L.append("")
        L.append("    `half never offered` means the half was not an at-support key on the gate")
        L.append("    arm at all — the arms diverge, so a half can be absent rather than refused,")
        L.append("    and that column is the honest residual of this test rather than evidence.")

    # ---------------------------------------------- [J] the deep-era window --------------
    section(L, "[J] THE DEEP-ERA WINDOW, CYCLE FOR CYCLE")
    ev = {a: [(e["kind"], int(e["cycle"]), e.get("level"))
              for e in r["events"] if e["kind"] in ("commit", "advance")]
          for a, r in list(arms.items()) + ([(bank_arm, ref)] if ref else [])}
    for a, q in ev.items():
        L.append(f"    {a:<10} commits " + str([(c, l) for k, c, l in q if k == "commit"])
                 + "  advances " + str([c for k, c, l in q if k == "advance"]))
    L.append("")
    lo, hi = 175, 201
    names = [q for q in ORDER if q in arms] + ([bank_arm] if ref else [])
    L.append(f"    {'cyc':>4} {'era':>4} " + " ".join(f"{n:>11}" for n in names))
    for cy in range(lo, hi + 1):
        rowtxt = []
        era_here = None
        for n in names:
            rr = arms.get(n) or ref
            e = rr["log"]["e"]
            era_here = era_here or rr["log"]["era"][cy - 1]
            rowtxt.append(f"{e[cy - 1]:>11.4f}" if cy - 1 < len(e) else f"{'—':>11}")
        L.append(f"    {cy:>4} {era_here:>4} " + " ".join(rowtxt))
    L.append("")
    L.append("    L5 keys: the cycle each was OFFERED (within one cadence tick of reaching")
    L.append("    support — the cadence is `recert_every` 5, and the miner's per-key arrival")
    L.append("    cycle is not banked, so the offer cycle is the tightest honest reconstruction)")
    L.append(f"    {'arm':<10} " + "offers (cycle, count, decision)")
    for a in [q for q in ORDER if q in arms]:
        for lv in (4, 5):
            ds = sorted(((d["cycle"], d["count"], "adm" if d["admit"] else "REF")
                         for (l_, fk), d in dec[a].items() if l_ == lv))
            if ds:
                L.append(f"    {a:<10} L{lv}: {ds}")

    # ------------------------------------ [K] round 2: the next level's currency ----------
    if any(a in arms for a in R2):
        section_r2(L, arms, arm_tag, tag, with_tags)

    txt = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as fh:
        fh.write(txt)
    print(txt)
    if mk_figs:
        make_figures(tag, arms, ref, bank_arm, series)
    return txt


def _steps(r):
    """[(level, cycle, step)] for every trial the arm ran, in order."""
    out = []
    for ps in (r.get("adm") or {}).get("passes") or []:
        for lr in ps["levels"]:
            for st in lr["steps"]:
                out.append((int(lr["level"]), int(ps["cycle"]), st))
    return out


def section_r2(L, arms, arm_tag, tag, with_tags):
    """[K] — the round-2 instruments: the twin-prefix fidelity check, the demand record, the
    yield signal, every gate's verdict on each round-2 arm's own trials, and the export gates."""
    section(L, "[K] ROUND 2 — THE GATE IN THE NEXT LEVEL'S CURRENCY: the instruments")

    # ---- K1: bit-identical to the none arm until the first refusal --------------------------
    L.append("  (K1) TWIN-PREFIX FIDELITY. Both round-2 arms share `st_gn_yk`'s stream key, their")
    L.append("       fires draw no RNG and the reader parse is sandboxed, so each must replay the")
    L.append("       none arm bit for bit until its first refusal takes effect.")
    gn = arms.get("st_gn_yk")
    for a in [q for q in R2 if q in arms]:
        r = arms[a]
        first_ref = next((c for lv, c, st in _steps(r) if not st["admit_gate"]), None)
        div = None
        if gn is not None:
            for k_ in ("e", "succ", "t_cum", "g_per_solve", "n_moves"):
                x_, y_ = r["log"][k_], gn["log"][k_]
                for i in range(min(len(x_), len(y_))):
                    if x_[i] != y_[i]:
                        div = i + 1 if div is None else min(div, i + 1)
                        break
        ok = (div is None) or (first_ref is not None and div >= first_ref)
        L.append(f"       {a:<10} first refusal c{first_ref}  first divergence from st_gn_yk "
                 f"c{div}   {'PASS' if ok else 'FAIL — diverged before any refusal'}")
    L.append("")

    # ---- K2: the demand record ----------------------------------------------------------------
    L.append("  (K2) THE DEMAND RECORD on every offered key (both round-2 arms log it). `value` is")
    L.append("       the sum over next-level keys AT SUPPORT having the candidate's class as a half")
    L.append("       (the gate's currency); `raw` the brief's literal sum over all next-level keys.")
    L.append("       `raw>=own` is DESIGN §9.3's Fact 1 (the literal form is the none arm).")
    L.append(f"       {'arm':<10} {'L':>2} {'n':>4} {'silent':>6} {'no nxt':>6} {'val>=1':>6}"
             f" {'val=0':>6} {'med val':>8} {'max val':>8} {'raw>=own':>9} {'med raw':>8}"
             f" {'refused':>8}")
    for a in [q for q in R2 if q in arms]:
        stp = _steps(arms[a])
        for lv in sorted({q[0] for q in stp}):
            ss = [st for l_, c, st in stp if l_ == lv]
            dm = [st.get("demand") for st in ss]
            live = [d for d in dm if d is not None and not d.get("silent")]
            vals = [d["value"] for d in live]
            raws = [d["raw"] for d in dm if d is not None]
            own = sum(1 for st, d in zip(ss, dm) if d is not None and d["raw"] >= st["count"])
            L.append(f"       {a:<10} {lv:>2} {len(ss):>4}"
                     f" {sum(1 for d in dm if d is not None and d.get('silent')):>6}"
                     f" {sum(1 for d in dm if d is None):>6}"
                     f" {sum(1 for v_ in vals if v_ >= 1):>6} {sum(1 for v_ in vals if v_ == 0):>6}"
                     f" {(np.median(vals) if vals else float('nan')):>8.1f}"
                     f" {(max(vals) if vals else 0):>8}"
                     f" {str(own) + '/' + str(len(raws)):>9}"
                     f" {(np.median(raws) if raws else float('nan')):>8.1f}"
                     f" {sum(1 for st in ss if not st['admit_gate']):>8}")
    L.append("")

    # ---- K3: the yield signal -------------------------------------------------------------------
    L.append("  (K3) THE YIELD SIGNAL, `d_yield = mean(share x level)_trial - mean(...)_cur`, on")
    L.append("       every trial of both round-2 arms; `unfit` = before the readout's first fit,")
    L.append("       where the level is the constant 0.5 and the gate compares shares alone.")
    L.append(f"       {'arm':<10} {'L':>2} {'fit':>5} {'n':>4} {'d=0':>5} {'d>0':>5} {'d<0':>5}"
             f" {'min d':>9} {'max d':>9} {'mean share':>11}")
    for a in [q for q in R2 if q in arms]:
        stp = _steps(arms[a])
        for lv in sorted({q[0] for q in stp}):
            for fit in (False, True):
                ss = [st for l_, c, st in stp if l_ == lv and bool(st["unfit"]) != fit
                      and st.get("d_yield") is not None]
                if not ss:
                    continue
                d = np.array([st["d_yield"] for st in ss])
                sh = np.array([st["share_trial"] for st in ss])
                L.append(f"       {a:<10} {lv:>2} {('fit' if fit else 'unfit'):>5} {len(d):>4}"
                         f" {int((d == 0).sum()):>5} {int((d > 0).sum()):>5}"
                         f" {int((d < 0).sum()):>5} {d.min():>+9.5f} {d.max():>+9.5f}"
                         f" {sh.mean():>11.4f}")
    L.append("")

    # ---- K4: every gate's verdict on each round-2 arm's own trials ------------------------------
    L.append("  (K4) EVERY GATE'S VERDICT ON THE SAME TRIALS. On each round-2 arm's own trials")
    L.append("       (same base, same fire), what world / read / yield / demand would each have")
    L.append("       decided. Per-trial, not a counterfactual walk: the base is the arm's own.")
    L.append(f"       {'arm':<10} {'L':>3} {'n':>4} " + " ".join(
        f"{'ref ' + g:>11}" for g in ("world", "read", "yield", "demand"))
        + "   yield=world demand=world yield=demand")
    for a in [q for q in R2 if q in arms]:
        stp = _steps(arms[a])
        thr = float(((arms[a].get("adm") or {}).get("thr")) or 1.0)
        for lvs, nm in (((2,), "2"), ((3, 4, 5), "3-5"), ((2, 3, 4, 5), "all")):
            ss = [st for l_, c, st in stp if l_ in lvs]
            if not ss:
                continue
            dec = {"world": [bool(st["admit_world"]) for st in ss],
                   "read": [(st.get("d_read") is None) or st["d_read"] >= -0.0 for st in ss],
                   "yield": [(st.get("d_yield") is None) or st["d_yield"] >= -0.0 for st in ss],
                   "demand": [(st.get("demand") is None) or bool(st["demand"].get("silent"))
                              or (st["demand"]["value"] - thr >= -0.0) for st in ss]}

            def agree(g1, g2):
                return sum(1 for x_, y_ in zip(dec[g1], dec[g2]) if x_ == y_)
            L.append(f"       {a:<10} {nm:>3} {len(ss):>4} " + " ".join(
                f"{sum(1 for q_ in dec[g] if not q_):>11}" for g in
                ("world", "read", "yield", "demand"))
                + f"   {agree('yield', 'world'):>11} {agree('demand', 'world'):>12}"
                  f" {agree('yield', 'demand'):>12}")
    L.append("")

    # ---- K5: the export gates ------------------------------------------------------------------
    L.append("  (K5) THE EXPORT GATES E-Y / E-D, off the exported passes, in this process")
    try:
        import gzip
        from rhm.practice.voicing.sotto_voce.aliquot.sostenuto.sostenuto import sos_export_gate
        for a in [q for q in R2 if q in arms]:
            for kind in ("unfit", "fit"):
                p = os.path.join(FIG, arm_tag[a], a, f"adm_export_{kind}.json.gz")
                if not os.path.isfile(p):
                    L.append(f"       {a:<10} {kind:<5} (no export)")
                    continue
                with gzip.open(p, "rt") as fh:
                    ex = json.load(fh)
                g = sos_export_gate(ex, log=lambda *_a, **_k: None)
                L.append(f"       {a:<10} {kind:<5} c{g['cycle']} L{g['level']} fires={g['n_fires']}"
                         f" cands={g['n_cands']} next-keys={g['n_next_keys']}"
                         f"  E-Y {g['E-Y:share_is_sb_yield_label']}"
                         f" (share>0 on {g['E-Y:n_instances_share_pos']} instance-fires,"
                         f" weight const {g['E-Y:weight_constant']})"
                         + (f" walk {g['E-Y:walk_rederived']}" if 'E-Y:walk_rederived' in g
                            else "")
                         + f"  E-D {g['E-D:demand_is_hand_count']}"
                         + (f" decisions {g['E-D:decisions_rederived']}"
                            if 'E-D:decisions_rederived' in g else ""))
                for k_, v_ in (g.get("falsify") or {}).items():
                    L.append(f"           {k_:<52} {v_}")
    except Exception as e:                                        # pragma: no cover
        L.append(f"       (export gate could not run: {e!r})")


def make_figures(tag, arms, ref, bank_arm, series):
    """Four panels, one job each. One y-axis per panel, a legend plus direct labels, the
    ungated build as a recessive dashed rule rather than a fourth category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": "#c9c9c4", "axes.labelcolor": "#33332f",
                         "text.color": "#33332f", "xtick.color": "#6b6b66",
                         "ytick.color": "#6b6b66", "font.size": 9,
                         "axes.grid": True, "grid.color": "#e7e7e2", "grid.linewidth": 0.7,
                         "axes.spines.top": False, "axes.spines.right": False})

    lvls = sorted({lv for (_, lv) in series})
    era_w = era_windows(list(arms.values())[0])
    # ---- 1. the quality column: the admitted table's test-pool error, one panel per level
    fig, axes = plt.subplots(1, max(1, len(lvls)), figsize=(3.1 * max(1, len(lvls)), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lv in zip(axes, lvls):
        ends = []
        for a in [q for q in ORDER if q in arms]:
            if (a, lv) not in series:
                continue
            cy, ea, el = series[(a, lv)]
            ax.plot(cy, ea, lw=2.0, color=COL[a], label=LAB[a])
            ends.append([float(ea[-1]), cy[-1], LAB[a]])
        # [sostenuto r2] five arms collide at the right edge: sort the end labels by height and
        # push them apart to a minimum gap (data units), leader-free — the text table [B] is
        # the record, the labels only name the lines
        ends.sort(key=lambda q: q[0])
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.05)
        for yl, cx, lab in ends:
            ax.annotate(lab, (cx, yl), xytext=(3, 0), textcoords="offset points",
                        fontsize=7, color="#33332f", va="center", annotation_clip=False)
        any_a = next((a for a in ORDER if (a, lv) in series), None)
        if any_a:
            cy, _, el = series[(any_a, lv)]
            ax.plot(cy, el, lw=1.4, ls="--", color=REF_COL, label="ungated live build")
        for i, (c0, c1, elv, _n) in sorted(era_w.items()):
            if i > 1:
                ax.axvline(c0, color="#d8d8d3", lw=1.0, zorder=0)
            if elv == lv:
                ax.axvspan(c0, c1, color="#0072B2", alpha=0.06, lw=0, zorder=0)
        ax.set_title(f"L{lv}", fontsize=10)
        ax.set_xlabel("cycle")
    axes[0].set_ylabel("world's error on the test pool")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("The admitted table's error on a disjoint test pool (n 256), per level — "
                 "the shaded band is the level's OWN era", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "q_test_error.png"), dpi=160)
    plt.close(fig)

    # ---- 2. how much of the live build each gate serves
    fig, axes = plt.subplots(1, max(1, len(lvls)), figsize=(3.1 * max(1, len(lvls)), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lv in zip(axes, lvls):
        for a in [q for q in ORDER if q in arms]:
            rows = [r_ for _, r_, _ in passes(arms[a]) if r_["level"] == lv]
            if not rows:
                continue
            cy = [r_["cycle"] for r_ in rows]
            sh = [r_["n_kept_rows"] / max(1, r_["n_live_rows"]) for r_ in rows]
            ax.plot(cy, sh, lw=2.0, color=COL[a], label=LAB[a])
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"L{lv}", fontsize=10)
        ax.set_xlabel("cycle")
    axes[0].set_ylabel("admitted rows / live build rows")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("What share of the miner's live build each gate lets the executor see",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "q_share_served.png"), dpi=160)
    plt.close(fig)

    # ---- 3. the task error, cycle for cycle, against the banked reference
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for a in [q for q in ORDER if q in arms]:
        e = np.asarray(arms[a]["log"]["e"], float)
        ax.plot(np.arange(1, len(e) + 1), e, lw=2.0, color=COL[a], label=LAB[a])
    if ref is not None:
        e = np.asarray(ref["log"]["e"], float)
        ax.plot(np.arange(1, len(e) + 1), e, lw=1.4, ls="--", color=REF_COL,
                label=f"{bank_arm} (no walk)")
    ax.set_xlabel("cycle")
    ax.set_ylabel("task error")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Task error, cycle for cycle", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "q_task_error.png"), dpi=160)
    plt.close(fig)

    # ---- 4. the readout's held-out AUC — the organ the read gate reads
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for a in [q for q in ORDER if q in arms]:
        om = arms[a]["log"]["vo_om"]
        cy = [i + 1 for i, q in enumerate(om)
              if isinstance(q, dict) and q.get("hold_auc") is not None]
        au = [q["hold_auc"] for q in om if isinstance(q, dict) and q.get("hold_auc") is not None]
        ax.plot(cy, au, lw=2.0, color=COL[a], label=LAB[a])
    if ref is not None:
        om = ref["log"]["vo_om"]
        cy = [i + 1 for i, q in enumerate(om)
              if isinstance(q, dict) and q.get("hold_auc") is not None]
        au = [q["hold_auc"] for q in om if isinstance(q, dict) and q.get("hold_auc") is not None]
        ax.plot(cy, au, lw=1.4, ls="--", color=REF_COL, label=f"{bank_arm} (no walk)")
    ax.axhline(0.5, color="#c9c9c4", lw=1.0)
    ax.set_xlabel("cycle")
    ax.set_ylabel("held-out AUC against the world")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("The projection's held-out AUC — the organ the read gate gates with",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "q_readout_auc.png"), dpi=160)
    plt.close(fig)
    # ---- 5. [sostenuto r2] THE COVERAGE TRACE — the round's lead number, as a figure.
    #      Three small multiples, one job each (a count per gate arm), horizontal bars on a
    #      shared order, every bar direct-labelled with its count (the palette's CVD and
    #      contrast WARNs require a secondary encoding; the text reduction is the table view).
    if "st_gn_yk" in arms:
        dec = {a: decisions(r) for a, r in arms.items()}
        none_keys = {lv: [fk for (l_, fk) in dec["st_gn_yk"] if l_ == lv] for lv in (3, 4)}
        gates = [a for a in ORDER if a in arms and a != "st_gn_yk"]
        vals = {"ref2": [], "b3": [], "b4": []}
        never = {"ref2": [], "b3": [], "b4": []}
        for a in gates:
            refd = {lv: {fk for (l_, fk), d in dec[a].items() if l_ == lv and not d["admit"]}
                    for lv in (2, 3)}
            admd = {lv: {fk for (l_, fk), d in dec[a].items() if l_ == lv and d["admit"]}
                    for lv in (2, 3)}
            vals["ref2"].append(len(refd[2]))
            never["ref2"].append(0)
            for k_, lv in (("b3", 3), ("b4", 4)):
                blk = [fk for fk in none_keys[lv]
                       if any(h in refd[lv - 1] for h in half_keys(fk))]
                vals[k_].append(len(blk))
                # the [I] table's residual: no half refused, and not every half admitted —
                # the half was never an at-support key on this arm at all
                never[k_].append(sum(1 for fk in none_keys[lv] if fk not in blk
                                     and not all(h in admd[lv - 1] for h in half_keys(fk))))
        panels = (("ref2", "L2 keys refused"),
                  ("b3", f"no-gate arm's L3 keys with a\nrefused half (of {len(none_keys[3])})"),
                  ("b4", f"no-gate arm's L4 keys with a\nrefused half (of {len(none_keys[4])})"))
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.4), sharey=True)
        ypos = np.arange(len(gates))[::-1]
        for ax, (k_, ttl) in zip(axes, panels):
            v_ = vals[k_]
            ax.barh(ypos, v_, height=0.62, color=[COL[a] for a in gates], edgecolor="#fcfcfb",
                    linewidth=2)
            ax.set_axisbelow(True)
            for y_, q_, nv_ in zip(ypos, v_, never[k_]):
                ax.annotate(str(q_) + (f"  ({nv_} never offered)" if nv_ else ""), (q_, y_),
                            xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8, color="#33332f")
            ax.set_title(ttl, fontsize=9)
            ax.set_xlim(0, max(max(v_) * 1.25, 1) * (1.9 if any(never[k_]) else 1.0))
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(ypos)
        axes[0].set_yticklabels([LAB[a] for a in gates])
        fig.suptitle("The coverage trace: what each gate's refusals cost the level above "
                     "(decide-once; halves matched by flat key)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "q_coverage.png"), dpi=160)
        plt.close(fig)
    print(f"  figures -> {out}/q_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="st_s1")
    ap.add_argument("--bank-tag", default="sb_s1")
    ap.add_argument("--bank-arm", default="sb_sv_yk")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-figs", action="store_true")
    # [sostenuto r2] further tags to read arms from (round 1's, for round 2's reduction)
    ap.add_argument("--with-tag", action="append", default=[])
    a = ap.parse_args()
    out = a.out or os.path.join(FIG, f"{a.tag}_reduction.txt")
    reduce_tag(a.tag, a.bank_tag, a.bank_arm, out, mk_figs=not a.no_figs,
               with_tags=tuple(a.with_tag))


if __name__ == "__main__":
    main()
