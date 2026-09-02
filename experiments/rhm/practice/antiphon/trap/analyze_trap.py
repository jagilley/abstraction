"""THE REDUCTION for `antiphon/trap` — the installed trap.

SKELETON, staged before any GPU. Every section is wired to the fields the run already logs;
the numbers appear when a tag exists. It reuses `analyze_antiphon.py`'s loaders wholesale, so
the trap tag is read exactly as the question tag is, and the two can be merged.

Sections, in the order the round's claims are made:

  §0  FIDELITY   the trap-off arm (`t00_trust`) and the banked `an_s0` arms at the NATIVE dose.
                 Gate G-T lives in `antiphon.py::fidelity_smoke`; this section carries its
                 full-scale companion: every logged series of a trap-off arm against its
                 `an_s0` twin, and the cross-tag substrate-identity assertion at setup.
  §1  CONTROLS   what the trap did NOT move: the per-cycle d* quota filled bin-for-bin, the
                 mined volume (`mine_cap`), cumulative priced spend, lifetime — and, new here,
                 the realised d* histogram against the PRE-SUBSTITUTION native head's, per
                 cycle, which is the measured form of "a distractor cannot redefine difficulty".
  §2  DOSE       the trap's own instrument, in every arm: what the world offered (`menu_frac`),
                 what the selector took (`take`), and what reached the miner (`trap_mined`).
                 The `wd_s0` lesson: a dose that does not land is not a treatment.
                 Also the SOLVE RATE and any `[q!] VOLUME SHORTFALL` — gate T-5, the round's
                 named risk (a junk-rich menu could starve `mine_cap`).
  §3  CLIMB      at-support trajectories split TRUE / JUNK at L2..L4, on the cycle clock and the
                 cumulative priced clock. The trap's first-order cost is here.
  §4  TABLES     what each commit froze — size, recall, PRECISION. The corruption metric, and
                 the one the offline model could not reach.
  §5  PREMIUM    the headline: the selection premium against the world's worthless fraction,
                 with `an_s0` supplying the native-dose point. Per level, per clock.
  §6  VALUE      the three clocks and deep-era recovered fraction against `crescendo`'s floors.
  §7  POLES      the nerdsnipe (does a judge do WORSE than the null?) and the comfort pole, at
                 each dose. The pool-share bound from `phase0_trap.py` [1b] is printed beside
                 the measured take, because it is what the take is being read against.
  §8  GUARD      the trust arm read against the table it stood on. Two reads carried in by the
                 orchestrator's request:
                   (a) the USE-WEIGHTED PRECISION of each arm's committed L3 table, IN-TAG, at
                       its commit cycle and at end of run — `phase0_trap.py` [9]'s table
                       reproduced on this round's own arms, so the guard is read against the
                       table it stood on rather than against `an_s0`'s.
                   (b) the REALISED trust-guard take rate PER CYCLE against the pool statistic
                       (`phase0_trap.py` [1b]) and against the menu fraction — the quota's
                       breadth demand is the named caveat (the guard's admit set is smaller than
                       64, so the selector falls back to the floor for the remainder), and this
                       is where it shows or does not.

Run from experiments/:
    python3 rhm/practice/antiphon/trap/analyze_trap.py --tag tr_s0 --merge-tag an_s0 \
        --fetch --figures
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANTIPHON = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(ANTIPHON))))

from rhm.practice.antiphon import analyze_antiphon as AA        # noqa: E402

FIG = os.path.join(HERE, "figures")
SUPPORT = AA.SUPPORT
AN_S0 = os.path.join(ANTIPHON, "figures", "an_s0")

# the dose each arm was run at, and the banked native-dose arm it is compared with
DOSE = {"t00_trust": 0.0,
        "t15_exo": 0.15, "t15_novel": 0.15, "t15_endo": 0.15,
        "t90_exo": 0.90, "t90_novel": 0.90, "t90_endo": 0.90,
        "t90_bisect": 0.90, "t90_trust": 0.90}
NATIVE_TWIN = {"t15_exo": "q_exo", "t90_exo": "q_exo",
               "t15_novel": "q_novel", "t90_novel": "q_novel",
               "t15_endo": "q_endo", "t90_endo": "q_endo",
               "t90_bisect": "q_bisect", "t00_trust": None, "t90_trust": None}
# `phase0_trap.py` [1b]: the junk share a breadth-first novelty judge takes, whatever f is
POOL_SHARE = {2: 0.409, 3: 0.686}
NATIVE_F = {2: 0.332, 3: 0.546}


def _qrows(lg):
    return [x for x in (lg.get("q") or []) if x]


# each trap arm and the `an_s0` arm it must be BIT-IDENTICAL to until the trap first bites
ERA1_TWIN = {"t90_exo": "q_exo", "t90_bisect": "q_bisect", "t90_novel": "q_novel",
             "t90_endo": "q_endo",
             # at era 1 the clean part of the mined span is a single block — a level-1 feature,
             # and every level-1 feature is a valid row — so `select_trust` falls through to
             # novelty with no ledger, which IS `select_novel`. The twin is therefore q_novel.
             "t90_trust": "q_novel"}


def sec0_fidelity(A, order, out, mtag="an_s0"):
    """THE ROUND'S FREE GATE. The trap atom does not exist at era 1 (the clean part of the mined
    span is one block, i.e. a level-1 feature, and every level-1 feature is a true row), so
    `menu_frac` is 0.000 there by construction — and every trap arm must therefore be
    BIT-IDENTICAL to its `an_s0` counterpart for the whole of era 1, and diverge on the first
    cycle of era 2, which is the first cycle the trap exists at all.

    This is the `an_s1` in-tag twin idiom applied across the dose: it certifies that the trap
    machinery is inert where the trap does not apply, at full scale, and it costs nothing."""
    print("\n" + "=" * 88)
    print("§0  FIDELITY — the trap is inert where the atom does not exist (era 1)")
    print("=" * 88)
    print(f"  {'arm':<12}{'twin':>12}{'era1 cycles':>13}{'max|diff| era1':>16}"
          f"{'first divergence':>18}{'expected':>10}{'':>4}")
    g0 = {}
    for a in order:
        tw = ERA1_TWIN.get(a)
        lab = tw if tw in A else (f"{tw}@{mtag}" if tw and f"{tw}@{mtag}" in A else None)
        if lab is None:
            continue
        la, lb = A[a]["log"], A[lab]["log"]
        n1 = sum(1 for e in la["era"] if e == 1)
        first, mx = None, 0.0
        n = min(len(la["cycle"]), len(lb["cycle"]))
        for i in range(n):
            d = 0.0
            for k in AA.GF_SERIES:
                va, vb = la.get(k), lb.get(k)
                if not va or not vb or i >= len(va) or i >= len(vb):
                    continue
                if va[i] is None or vb[i] is None:
                    continue
                d = max(d, abs(float(va[i]) - float(vb[i])))
            if i < n1:
                mx = max(mx, d)
            if d > 0 and first is None:
                first = int(la["cycle"][i])
        ok = (mx == 0.0) and (first is None or first == n1 + 1)
        g0[a] = {"twin": lab, "era1_cycles": n1, "max_abs_era1": mx,
                 "first_divergence": first, "expected": n1 + 1, "pass": bool(ok)}
        print(f"  {a:<12}{lab:>12}{n1:>13}{mx:>16.3e}"
              f"{(str(first) if first else '-'):>18}{n1 + 1:>10}"
              f"{('  PASS' if ok else '  FAIL'):>6}")
    print("\n  A trap arm that is 0.000e+00 through era 1 and diverges on the first cycle of")
    print("  era 2 says the trap machinery moved nothing except the menu, where the menu moved.")
    out["sec0_era1_twin"] = g0


def sec1_controls(A, order, setup, out):
    """Everything the trap must not move, including the new d*-orthogonality read."""
    print("\n" + "=" * 88)
    print("§1  CONTROLS — what the trap knob did NOT move")
    print("=" * 88)
    print(f"  {'arm':<12}{'dose':>6}{'quota_ok':>10}{'d* sel':>8}{'d* menu':>9}"
          f"{'d* trap':>9}{'d* native':>10}{'vol_ok':>8}{'t_cum':>14}{'cycles':>8}")
    for a in order:
        lg = A[a]["log"]
        q = _qrows(lg)
        tr = [x["trap"] for x in q if x.get("trap")]
        vol = np.mean([(x.get("n_mined") or 0) >= A[a]["config"]["mine_cap"] for x in q]) \
            if q else float("nan")
        print(f"  {a:<12}{DOSE.get(a, 0.0):>6.2f}"
              f"{np.mean([x['quota_ok'] for x in q]):>10.3f}"
              f"{np.mean([x['d_mean'] for x in q]):>8.3f}"
              f"{np.mean([x['d_mean_menu'] for x in q]):>9.3f}"
              f"{(np.mean([t['d_mean_trap'] for t in tr if t['d_mean_trap'] is not None]) if tr else float('nan')):>9.3f}"
              f"{(np.mean([t['d_mean_native'] for t in tr if t['d_mean_native'] is not None]) if tr else float('nan')):>10.3f}"
              f"{vol:>8.3f}{lg['t_cum'][-1]:>14,.0f}{len(lg['cycle']):>8}")
    print("\n  d* trap vs d* native is the measured form of `a distractor cannot redefine")
    print("  difficulty`; `phase0_trap.py` [3] put the offline TV at 0.074 (era 2) / 0.039 (3).")


def sec2_dose(A, order, out):
    """Offered, taken, landed — and the solve rate (gate T-5)."""
    print("\n" + "=" * 88)
    print("§2  DOSE — offered, taken, landed; and whether a junk-rich menu starves the miner")
    print("=" * 88)
    print(f"  {'arm':<12}{'era':>4}{'menu f':>8}{'take':>7}{'mined':>7}"
          f"{'n_solved':>10}{'n_mined':>9}{'shortfall':>10}")
    for a in order:
        q = _qrows(A[a]["log"])
        for lvl in sorted({x["target_level"] for x in q}):
            g = [x for x in q if x["target_level"] == lvl]
            tr = [x["trap"] for x in g if x.get("trap")]
            cap = A[a]["config"]["mine_cap"]
            print(f"  {a:<12}{lvl - 1:>4}"
                  f"{(np.mean([t['menu_frac'] for t in tr]) if tr else 0.0):>8.3f}"
                  f"{(np.mean([t['take'] for t in tr]) if tr else 0.0):>7.3f}"
                  f"{(np.mean([x['trap_mined'] for x in g if x.get('trap_mined') is not None]) if tr else 0.0):>7.3f}"
                  f"{np.mean([x.get('n_solved') or 0 for x in g]):>10.2f}"
                  f"{np.mean([x.get('n_mined') or 0 for x in g]):>9.2f}"
                  f"{np.mean([(x.get('n_mined') or 0) < cap for x in g]):>10.3f}")


def sec5_premium(A, order, setup, out, mtag="an_s0"):
    """THE HEADLINE. True keys at support at each level, at end of era 3 and end of run, for
    each trap arm beside the `an_s0` arm that is the same selector at the NATIVE dose (0.546).
    The premium is each arm's distance from its own dose's null."""
    print("\n" + "=" * 88)
    print("§5  THE SELECTION PREMIUM vs THE WORLD'S WORTHLESS FRACTION")
    print("=" * 88)
    ts = AA.truth_sets(setup)

    def at(lab, level, upto_era=None):
        lg = A[lab]["log"]
        idx = [i for i, e in enumerate(lg["era"])
               if (upto_era is None or e <= upto_era)]
        if not idx:
            return None, None
        ks, _ = AA.keyset(lg, idx[-1], level)
        t = ts.get(level, set())
        return len(ks & t), len(ks - t)

    print("  true / junk keys at support — end of ERA 3 (the level the trap is live on) and")
    print("  end of RUN, per level, trap arm beside its native-dose twin in an_s0")
    for level in (3, 4):
        print(f"\n  --- L{level} ---")
        print(f"  {'selector':<10}{'f=0.90 e3':>12}{'native e3':>12}{'delta':>8}"
              f"{'f=0.90 end':>13}{'native end':>13}{'delta':>8}")
        rows = {}
        for a in order:
            tw = ERA1_TWIN.get(a)
            lab = tw if tw in A else (f"{tw}@{mtag}" if tw and f"{tw}@{mtag}" in A else None)
            if lab is None or a == "t90_trust":
                continue
            t3, _ = at(a, level, upto_era=3)
            n3, _ = at(lab, level, upto_era=3)
            te, _ = at(a, level)
            ne, _ = at(lab, level)
            if t3 is None or n3 is None:
                continue
            rows[a] = {"trap_e3": t3, "native_e3": n3, "trap_end": te, "native_end": ne}
            print(f"  {a:<10}{t3:>12}{n3:>12}{t3 - n3:>+8}"
                  f"{te:>13}{ne:>13}{te - ne:>+8}")
        # the trust arm has no native twin (a new selector), so it is reported against the
        # f = 0.90 NULL, which is the comparison its own dose supports
        if "t90_trust" in A and "t90_exo" in A:
            t3, _ = at("t90_trust", level, upto_era=3)
            e3, _ = at("t90_exo", level, upto_era=3)
            te, _ = at("t90_trust", level)
            ee, _ = at("t90_exo", level)
            print(f"  {'t90_trust':<10}{t3:>12}{'(vs null)':>12}{t3 - e3:>+8}"
                  f"{te:>13}{'(vs null)':>13}{te - ee:>+8}")
            rows["t90_trust"] = {"trap_e3": t3, "null_e3": e3, "trap_end": te, "null_end": ee}
        out.setdefault("sec5_premium", {})[f"L{level}"] = rows
    print("\n  Each trap arm's premium over the f = 0.90 null is the within-dose read; the")
    print("  delta against its native-dose twin is what the dose itself cost that selector.")


def sec7_poles(A, order, setup, out):
    """Does a judge do WORSE than the null? Read against the pool-share bound."""
    print("\n" + "=" * 88)
    print("§7  THE TWO POLES — measured take against the offline pool-share bound")
    print("=" * 88)
    print(f"  {'era':>4}{'pool share':>12}{'menu f':>9}{'null take':>11}"
          f"{'judge take':>12}{'over-take':>11}{'bound':>8}")
    for a in order:
        if a not in ("t15_novel", "t90_novel", "t90_endo", "t15_endo", "t90_trust"):
            continue
        q = _qrows(A[a]["log"])
        for lvl in sorted({x["target_level"] for x in q}):
            era = lvl - 1
            if era not in POOL_SHARE:
                continue
            g = [x["trap"] for x in q if x["target_level"] == lvl and x.get("trap")]
            if not g:
                continue
            f = float(np.mean([t["menu_frac"] for t in g]))
            take = float(np.mean([t["take"] for t in g]))
            print(f"  {era:>4}{POOL_SHARE[era]:>12.3f}{f:>9.3f}{f:>11.3f}"
                  f"{take:>12.3f}{take / max(f, 1e-9):>11.2f}"
                  f"{POOL_SHARE[era] / max(f, 1e-9):>8.2f}")
    print("\n  `over-take` above 1 is a judge doing worse than not selecting at all; `bound` is")
    print("  what `phase0_trap.py` [1b] says a breadth-first novelty judge must land on.")


def _cum_use(lg, lvl, upto=None):
    """Cumulative per-entry beam use at `lvl` through cycle index `upto`, with the truth mask
    that was live there. Returns (use, mask) or (None, None)."""
    tot, mask = None, None
    ent = lg.get("entry") or []
    for i, e in enumerate(ent):
        if upto is not None and i > upto:
            break
        if not e:
            continue
        h = (e.get("hist") or {}).get("beam") or {}
        if lvl not in h:
            continue
        vv = np.asarray(h[lvl], float)
        if tot is None or len(tot) != len(vv):
            tot = np.zeros(len(vv))          # the table grew: restart on the new shape
        tot += vv
        mask = np.asarray(e["true_mask"][lvl])
    return tot, mask


def _prec(tot, mask):
    if tot is None or mask is None or len(tot) != len(mask):
        return None, None
    tp = float(mask.mean()) if len(mask) else None
    up = float((tot * mask).sum() / tot.sum()) if tot.sum() > 0 else None
    return tp, up


def sec8_guard(A, order, out, window=20):
    """(a) the use-weighted precision of each arm's own committed L3 table, in-tag, at its
    commit cycle and at end of run; (b) the trust guard's realised per-cycle take."""
    print("\n" + "=" * 88)
    print("§8  THE GUARD — the beam's own use record, in-tag, and the table it stood on")
    print("=" * 88)
    print("  (a) COMMITTED-TABLE PRECISION vs USE-WEIGHTED PRECISION, this round's own arms.")
    print(f"      At the commit cycle the level's use is empty by construction (the table was")
    print(f"      just frozen), so the earliest informative read is commit + {window} cycles;")
    print("      both that and the end-of-run read are reported.")
    print(f"  {'arm':<12}{'lvl':>4}{'commit':>8}{'rows':>6}{'true':>6}{'tab prec':>10}"
          f"{'use-wt @c+w':>13}{'use-wt @end':>13}{'lift @end':>11}")
    g8 = {}
    for a in order:
        lg = A[a]["log"]
        commits = {int(e["level"]): int(e["cycle"]) for e in A[a].get("events", [])
                   if e["kind"] == "commit"}
        for lvl in ("2", "3", "4"):
            cc = commits.get(int(lvl))
            if cc is None:
                continue
            try:
                ci = lg["cycle"].index(cc)
            except ValueError:
                continue
            wi = min(ci + window, len(lg["cycle"]) - 1)
            tw, mw = _cum_use(lg, lvl, upto=wi)
            te, me = _cum_use(lg, lvl, upto=None)
            tpw, upw = _prec(tw, mw)
            tpe, upe = _prec(te, me)
            if tpe is None:
                continue
            g8.setdefault(a, {})[lvl] = {
                "commit_cycle": cc, "n_rows": int(len(me)), "n_true": int(me.sum()),
                "table_precision": tpe, "use_wt_at_commit_plus_w": upw,
                "use_wt_at_end": upe,
                "lift_at_end": (upe / tpe) if (upe is not None and tpe) else None}
            print(f"  {a:<12}{lvl:>4}{cc:>8}{len(me):>6}{int(me.sum()):>6}{tpe:>10.3f}"
                  f"{(f'{upw:.3f}' if upw is not None else '-'):>13}"
                  f"{(f'{upe:.3f}' if upe is not None else '-'):>13}"
                  f"{(f'{upe / tpe:.2f}x' if upe is not None and tpe else '-'):>11}")
    print("\n  Offline ([9], on the banked an_s0 arms at the NATIVE dose): L3 table 0.25-0.50")
    print("  -> use-weighted 0.73-0.93, lift 1.8-3.2x. The trust arm's guard stood on the")
    print("  `use-wt` column of its OWN row, so that is what its take should be read against.")

    print("\n  (b) THE TRUST GUARD'S REALISED TAKE, PER CYCLE, vs the pool statistic.")
    print("      `phase0_trap.py` [1b]: a breadth-first novelty judge lands on the POOL SHARE")
    print("      (0.686 at era 3) whatever the menu offers. The named caveat is that the")
    print("      guard's admit set is smaller than the 64-per-cycle breadth demand, so the")
    print("      selector falls back to the floor for the remainder — visible here or not.")
    print(f"  {'arm':<12}{'era':>4}{'n_cyc':>6}{'menu f':>8}{'take mean':>11}{'sd':>7}"
          f"{'min':>7}{'max':>7}{'pool share':>12}{'take/pool':>11}")
    series = {}
    for a in order:
        q = _qrows(A[a]["log"])
        for lvl in sorted({x["target_level"] for x in q}):
            era = lvl - 1
            g = [x for x in q if x["target_level"] == lvl and x.get("trap")]
            if not g:
                continue
            tk = np.array([x["trap"]["take"] for x in g], float)
            mf = float(np.mean([x["trap"]["menu_frac"] for x in g]))
            ps = POOL_SHARE.get(era)
            series.setdefault(a, {})[str(era)] = {
                "cycle": [int(x["cycle"]) for x in g], "take": [float(t) for t in tk],
                "menu_frac": mf, "pool_share": ps}
            print(f"  {a:<12}{era:>4}{len(tk):>6}{mf:>8.3f}{tk.mean():>11.3f}{tk.std():>7.3f}"
                  f"{tk.min():>7.3f}{tk.max():>7.3f}"
                  f"{(f'{ps:.3f}' if ps else '-'):>12}"
                  f"{(f'{tk.mean() / ps:.2f}' if ps else '-'):>11}")
    out["sec8_use_precision"] = g8
    out["sec8_take_series"] = series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tr_s0")
    ap.add_argument("--merge-tag", default="an_s0",
                    help="comma-separated tags merged in for the native dose "
                         "(the [antiphon-s2] idiom)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--sections", default="0,1,2,3,4,5,6,7,8")
    args = ap.parse_args()
    if args.fetch:
        AA.fetch(args.tag)
    root = os.path.join(AA.FIG, args.tag)
    if not os.path.isdir(root):
        print(f"no such tag on disk: {root}\n"
              f"this is the staged skeleton — nothing has been run yet.")
        return
    order = [a for a in sorted(os.listdir(root))
             if os.path.isfile(os.path.join(root, a, "results.json"))]
    A = {a: AA._load(root, os.path.join(a, "results.json")) for a in order}
    out = {"tag": args.tag, "arms": order, "dose": {a: DOSE.get(a) for a in order}}
    want = {x.strip() for x in args.sections.split(",") if x.strip()}
    setup = AA._load(root, "setup.json")
    summary = AA._load(root, "summary.json")
    # fold in the native-dose tag, with the same substrate assertions AA makes
    for mtag in [t.strip() for t in args.merge_tag.split(",") if t.strip()]:
        if args.fetch:
            AA.fetch(mtag)
        mroot = os.path.join(AA.FIG, mtag)
        msetup, msummary = AA._load(mroot, "setup.json"), AA._load(mroot, "summary.json")
        if not (msetup and msummary):
            print(f"[merge] {mtag}: no setup/summary; skipped")
            continue
        for k in setup["refs"]:
            if k == "macro_true":
                continue
            assert np.allclose(np.asarray(setup["refs"][k], float),
                               np.asarray(msetup["refs"][k], float)), \
                f"--merge-tag {mtag} has different refs for {k}: not the same substrate"
        assert (setup.get("stale_task_matched") == msetup.get("stale_task_matched")
                and setup["read_acc"] == msetup["read_acc"]), \
            f"--merge-tag {mtag} trained a different substrate"
        for a in msummary["order"]:
            r = AA._load(os.path.join(mroot, a), "results.json")
            if r is None:
                continue
            lab = a if a not in A else f"{a}@{mtag}"
            A[lab] = r
            order.append(lab)
        print(f"[merge] folded {mtag}: {msummary['order']} "
              f"(refs and substrate identical — asserted)")
    trap_arms = [a for a in order if a in DOSE]
    if "0" in want:
        sec0_fidelity(A, trap_arms, out)
    if "1" in want:
        sec1_controls(A, trap_arms, setup, out)
    if "2" in want:
        sec2_dose(A, trap_arms, out)
    if "3" in want:
        AA.sec3_climb(A, order, setup, out)
    if "4" in want:
        AA.sec4_tables(A, order, setup, out)
    if "5" in want:
        AA.sec5_clocks(A, order, out)
        sec5_premium(A, order, setup, out)
    if "6" in want:
        AA.sec6_value(A, order, setup, out)
    if "7" in want:
        sec7_poles(A, trap_arms, setup, out)
    if "8" in want:
        sec8_guard(A, trap_arms, out)
    if args.figures:
        figures(A, order, setup, args.tag, out)
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, f"{args.tag}_reduction.json"), "w") as f:
        json.dump(out, f, indent=1)




def figures(A, order, setup, tag, out):
    """Two panels: the climb the trap is read on, and the take the guard is read on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = AA.truth_sets(setup)
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    col = {"t90_exo": "#888888", "t90_bisect": "#1b7837", "t90_trust": "#762a83",
           "t90_novel": "#d95f02", "t90_endo": "#2166ac"}
    for a in order:
        if a not in col:
            continue
        lg = A[a]["log"]
        xs, ys = [], []
        for i, c in enumerate(lg["cycle"]):
            ks, _ = AA.keyset(lg, i, 4)
            xs.append(c); ys.append(len(ks & ts.get(4, set())))
        ax[0].plot(xs, ys, color=col[a], lw=1.8, label=a)
        nat = ERA1_TWIN.get(a)
        lab = f"{nat}@an_s0" if f"{nat}@an_s0" in A else nat
        if a != "t90_trust" and lab in A:
            lg2 = A[lab]["log"]
            y2 = [len(AA.keyset(lg2, i, 4)[0] & ts.get(4, set()))
                  for i in range(len(lg2["cycle"]))]
            ax[0].plot(lg2["cycle"], y2, color=col[a], lw=1.0, ls=":", alpha=0.65)
    ax[0].axvline(111, color="k", lw=0.8, ls="--", alpha=0.5)
    ax[0].text(113, 2, "era 3 begins\n(trust diverges)", fontsize=7)
    ax[0].set_xlabel("cycle"); ax[0].set_ylabel("true L4 keys at support")
    ax[0].set_title("the climb at f = 0.90 (dotted = same selector at the native 0.546)")
    ax[0].legend(fontsize=7)
    for a in order:
        if a not in col:
            continue
        q = [x for x in (A[a]["log"].get("q") or []) if x and x.get("trap")]
        if not q:
            continue
        ax[1].plot([x["cycle"] for x in q], [x["trap"]["take"] for x in q],
                   color=col[a], lw=1.4, label=a)
    ax[1].axhline(0.900, color="k", lw=0.9, ls="-", alpha=0.6)
    ax[1].text(2, 0.915, "menu f = 0.900 (what the world offered)", fontsize=7)
    for era, ps in POOL_SHARE.items():
        ax[1].axhline(ps, color="k", lw=0.9, ls="--", alpha=0.5)
        ax[1].text(2, ps + 0.012, f"pool share, era {era} = {ps:.3f}", fontsize=7)
    ax[1].set_xlabel("cycle"); ax[1].set_ylabel("distractor share of the 64 posed")
    ax[1].set_title("what each selector took")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(FIG, f"{tag}_trap.png")
    fig.savefig(p, dpi=150)
    print(f"\n[fig] wrote {p}")


if __name__ == "__main__":
    main()
