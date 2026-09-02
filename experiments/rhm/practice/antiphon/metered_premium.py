"""[antiphon-m] THE PREMIUM TABLE — does the question port's premium RISE when questions
are scarce?

WHAT IT COMPUTES. For each selector, its premium over the exogenous ladder — the same arm minus
`q_exo`, in the same tag — on the two quantities the port is claimed to move:

    (a) L4 true keys at support, end of run   (the mining axis, `analyze_antiphon.py` section 3)
    (b) recovered fraction at eras 4 and 5    (the value axis, section 6)

and puts the METERED point (`an_m0`, `n_pr` 24 / `pr_width` 8) beside the ABUNDANCE point
(`an_s0`, 64 / 16). Both tags are schedule-paced at the same caps and lifetime-identical at 201
cycles, and the metered tag trains the same substrate (asserted at setup by `--ref-tag an_s0`
and re-checked here), so the two columns differ in the meter and in nothing else.

THE FLOOR QUESTION, stated rather than borrowed. `an_s0`'s deep-era deltas were read against
`crescendo`'s measured floors (in-node era-4/5 displacement 0.030 / 0.410; earning-family stream
0.087). Those floors were measured AT ABUNDANCE and there is no banked metered twin to
re-measure them on, so they are not transferred. The in-tag floor for the metered column is
computed here from the metered NULL's own displacement structure: the within-era cycle-to-cycle
spread of `q_exo`'s own recovered fraction over the cycles the era statistic is read on. It is a
different object from a displacement floor measured across twinned arms and is labelled as one
everywhere it appears.

CPU only, no GPU, no Modal, no new runs. Run from experiments/:
    PYTHONPATH=. python3 rhm/practice/antiphon/metered_premium.py
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SUPPORT = 3
# the six selectors the metered core runs, in the order the round reads them
SELECTORS = ["q_exo", "q_bisect", "q_endo", "q_novel", "q_comp", "q_trust"]
# crescendo's floors, which apply to the ABUNDANCE column only
AB_FLOORS = {"in_node_e4": 0.030, "in_node_e5": 0.410, "earning_family": 0.087}


def load(tag, arm):
    p = os.path.join(FIG, tag, arm, "results.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def setup(tag):
    p = os.path.join(FIG, tag, "setup.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def truth_sets(su):
    return {int(l): {tuple(int(x) for x in r) for r in d["flat"]}
            for l, d in (su.get("true_tables") or {}).items()}


def era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def rf_series(lg, refs, j):
    """The per-cycle recovered fraction inside era j+1 — section 6's statistic before the
    last-three-cycles mean is taken. The mean of its tail IS section 6's number; its spread is
    what the in-tag floor below is built from."""
    st, fl = refs["stale"][j], refs["floor"][j]
    return [(st - lg["e"][i]) / (st - fl) for i in era_idx(lg, j)]


def recovered(lg, refs, n_era=5):
    out = []
    for j in range(n_era):
        idx = era_idx(lg, j)
        out.append(None if not idx
                   else float(np.mean([lg["e"][i] for i in idx[-3:]])))
    return [None if e is None else (refs["stale"][j] - e) / (refs["stale"][j] - refs["floor"][j])
            for j, e in enumerate(out)]


def true_at_support(r, su, level=4):
    lg, T = r["log"], truth_sets(su)
    i = len(lg["cycle"]) - 1
    if level == 4 and lg.get("gy"):
        st = lg["gy"][i] or {}
    else:
        st = (lg["miner"][i] or {}).get(str(level)) or {}
    ks = {tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])}
    return len(ks & T[level]), len(ks)


def profile(tag, arm, su):
    r = load(tag, arm)
    if r is None:
        return None
    lg = r["log"]
    t4, a4 = true_at_support(r, su, 4)
    t3, a3 = true_at_support(r, su, 3)
    smy = json.load(open(os.path.join(FIG, tag, "summary.json")))
    reg = (smy.get("arms", {}).get(arm) or {}).get("regime")
    if reg is None:
        # `an_s0` predates the in-run regime block, so its column is backfilled from
        # `phase0_metered.json` [M0], which recomputes exactly the same statistics from the
        # same banked logs. Marked `backfilled` so it is never mistaken for an in-run record.
        pm = os.path.join(HERE, "phase0_metered.json")
        if os.path.isfile(pm):
            ar = (json.load(open(pm)).get("abundance_reference") or {}).get(arm)
            if ar:
                reg = {"n_pr": 64, "pr_width": 16, "mine_cap": 8,
                       "solves_per_cycle": ar["solves"], "solves_x_cap": ar["x_cap"],
                       "cap_binds_frac": ar["cap_binds"], "obs_per_cycle": ar["obs"],
                       "prop_rows_per_cycle": ar["prop_rows"],
                       "prop_buf_end_frac": ar["buf_end"],
                       "prop_buf_full_cycle": ar["buf_full_cycle"], "backfilled": True}
    return {"n_cycles": len(lg["cycle"]), "t_cum": float(lg["t_cum"][-1]),
            "rf": recovered(lg, su["refs"]),
            "L4_true": t4, "L4_at_sup": a4, "L3_true": t3, "L3_at_sup": a3,
            "commits": {int(e["level"]): int(e["cycle"])
                        for e in r["events"] if e["kind"] == "commit"},
            "regime": reg,
            "rf_e4_series": rf_series(lg, su["refs"], 3),
            "rf_e5_series": rf_series(lg, su["refs"], 4)}


def null_floor(P, tag):
    """The in-tag floor from the NULL's own displacement structure: the sd of `q_exo`'s own
    per-cycle recovered fraction over the cycles section 6 reads (an era's last three), taken
    at eras 4 and 5. NOT a twinned-arm displacement floor — a different and weaker object."""
    p = P.get(f"{tag}/q_exo")
    if p is None:
        return None
    out = {}
    for key, era in (("e4", "rf_e4_series"), ("e5", "rf_e5_series")):
        s = p[era]
        out[key] = float(np.std(s[-3:], ddof=1)) if len(s) >= 3 else None
        out[key + "_full_era_sd"] = float(np.std(s, ddof=1)) if len(s) >= 2 else None
    return out


def fmt(v, w=9, p=3):
    if v is None:
        return "-".rjust(w)
    return (f"{v:+.{p}f}" if isinstance(v, float) else str(v)).rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abundance-tag", default="an_s0")
    ap.add_argument("--metered-tag", default="an_m0")
    args = ap.parse_args()
    AB, ME = args.abundance_tag, args.metered_tag

    su_ab, su_me = setup(AB), setup(ME)
    assert su_ab, f"no setup.json under figures/{AB}"

    print("=" * 96)
    print(f"THE PREMIUM TABLE — the question port at two meters   "
          f"(abundance {AB} | metered {ME})")
    print("=" * 96)

    same = None
    if su_me:
        same = (all(np.allclose(np.asarray(su_ab["refs"][k], float),
                                np.asarray(su_me["refs"][k], float))
                    for k in su_ab["refs"] if k != "macro_true")
                and su_ab["read_acc"] == su_me["read_acc"]
                and su_ab.get("stale_task_matched") == su_me.get("stale_task_matched"))
        print(f"\n[pre] same substrate as {AB}: {same}   read_acc "
              f"{su_ab['read_acc']} / {su_me['read_acc']}   stale "
              f"{su_ab.get('stale_task_matched')} / {su_me.get('stale_task_matched')}")
        assert same, "the metered tag trained a different substrate — the columns are not comparable"
        print("      refs identical, so section 6's recovered fraction is normalised by the SAME")
        print("      stale/floor pair in both columns and the two premiums are like-for-like.")
    else:
        print(f"\n[pre] figures/{ME} not present — printing the abundance column only.")

    P = {}
    for tag, su in ((AB, su_ab), (ME, su_me)):
        if su is None:
            continue
        for a in SELECTORS:
            p = profile(tag, a, su)
            if p is not None:
                P[f"{tag}/{a}"] = p

    # ---- the regime, side by side --------------------------------------------------------
    print("\n[0] THE REGIME, as measured in each tag (the round's precondition, not its result)")
    print(f"  {'tag/arm':22s}{'n_pr':>6}{'width':>7}{'solv/cyc':>10}{'x cap':>8}"
          f"{'cap binds':>11}{'obs/cyc':>9}{'pi rows':>9}{'buf end':>9}{'buf full':>10}")
    for k, p in P.items():
        g = p["regime"]
        if not g:
            print(f"  {k:22s}{'-':>6}{'-':>7}   (no regime block — tag predates it)")
            continue
        print(f"  {k:22s}{g['n_pr']:>6}{g['pr_width']:>7}{g['solves_per_cycle']:>10.2f}"
              f"{g['solves_x_cap']:>7.2f}x{g['cap_binds_frac']:>10.1%}"
              f"{g['obs_per_cycle']:>9.2f}{g['prop_rows_per_cycle']:>9.0f}"
              f"{g['prop_buf_end_frac']:>8.1%}"
              f"{('c' + str(g['prop_buf_full_cycle'])) if g['prop_buf_full_cycle'] else 'never':>10}"
              + ("   [backfilled from phase0_metered.json]" if g.get("backfilled") else ""))

    # ---- the floors ----------------------------------------------------------------------
    fl_me = null_floor(P, ME) if su_me else None
    print("\n[1] FLOORS. The abundance column is read against crescendo's MEASURED floors")
    print(f"    (in-node era-4 {AB_FLOORS['in_node_e4']}, era-5 {AB_FLOORS['in_node_e5']}, "
          f"earning-family {AB_FLOORS['earning_family']}).")
    if fl_me:
        print("    The metered column has NO banked twin, so those floors are NOT transferred.")
        print("    Its in-tag handle is the metered null's own displacement structure — the sd")
        print("    of `q_exo`'s own per-cycle recovered fraction. This is a weaker object than a")
        print("    twinned-arm displacement floor and is labelled as one:")
        print(f"      era 4: sd over the 3 cycles section 6 reads = {fl_me['e4']:.3f}   "
              f"over the whole era = {fl_me['e4_full_era_sd']:.3f}")
        print(f"      era 5: sd over the 3 cycles section 6 reads = {fl_me['e5']:.3f}   "
              f"over the whole era = {fl_me['e5_full_era_sd']:.3f}")

    # ---- THE PREMIUM TABLE ---------------------------------------------------------------
    print("\n[2] THE PREMIUM — each selector MINUS `q_exo`, within its own tag")
    print(f"  {'selector':12s}" + "".join(
        f"{c:>13}" for c in ("L4true AB", "L4true ME", "rf e4 AB", "rf e4 ME",
                             "rf e5 AB", "rf e5 ME")))
    rows = {}
    for a in SELECTORS:
        if a == "q_exo":
            continue
        cells, ok = [], False
        rec = {}
        for tag in (AB, ME):
            base, arm = P.get(f"{tag}/q_exo"), P.get(f"{tag}/{a}")
            if base is None or arm is None:
                rec[tag] = None
                continue
            ok = True
            rec[tag] = {"d_L4_true": arm["L4_true"] - base["L4_true"],
                        "d_rf_e4": (None if arm["rf"][3] is None or base["rf"][3] is None
                                    else arm["rf"][3] - base["rf"][3]),
                        "d_rf_e5": (None if arm["rf"][4] is None or base["rf"][4] is None
                                    else arm["rf"][4] - base["rf"][4]),
                        "L4_true": arm["L4_true"], "base_L4_true": base["L4_true"]}
        if not ok:
            continue
        rows[a] = rec
        g = lambda t, k: (None if rec.get(t) is None else rec[t][k])
        line = f"  {a:12s}"
        for k in ("d_L4_true", "d_rf_e4", "d_rf_e5"):
            for tag in (AB, ME):
                v = g(tag, k)
                line += (f"{v:>+13d}" if isinstance(v, int) else fmt(v, 13))
        print(line)

    # ---- the premium RATIO, which is the round's question ---------------------------------
    print("\n[3] DOES THE PREMIUM RISE WHEN QUESTIONS ARE SCARCE?  metered premium / abundance")
    print("    premium, per selector. A ratio > 1 means the same selector buys more at the")
    print("    metered knob. Ratios are printed only where the abundance premium is non-zero")
    print("    and both signs agree; where the signs differ the pair is printed instead.")
    print(f"  {'selector':12s}{'L4 true':>22}{'rf e4':>22}{'rf e5':>22}")
    for a, rec in rows.items():
        cells = []
        for k in ("d_L4_true", "d_rf_e4", "d_rf_e5"):
            x = None if rec.get(AB) is None else rec[AB][k]
            y = None if rec.get(ME) is None else rec[ME][k]
            if x is None or y is None:
                cells.append("-")
            elif x == 0:
                cells.append(f"{x:+.3g} -> {y:+.3g}")
            elif (x > 0) != (y > 0):
                cells.append(f"{x:+.3g} -> {y:+.3g} (sign)")
            else:
                cells.append(f"{x:+.3g} -> {y:+.3g}  {y / x:.2f}x")
        print(f"  {a:12s}" + "".join(f"{c:>22}" for c in cells))

    out = {"abundance_tag": AB, "metered_tag": ME, "same_substrate": same,
           "abundance_floors": AB_FLOORS, "metered_null_floor": fl_me,
           "profiles": P, "premium": rows}
    with open(os.path.join(HERE, "metered_premium.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(HERE, 'metered_premium.json')}")


if __name__ == "__main__":
    main()
