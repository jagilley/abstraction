"""pacer_marginal.py — the pacer-alone marginal, on `an_s0`'s own statistics.

WHY THIS EXISTS. `an_s0` carries `q_bisect_loop` (the oracle selector on the donor's
thermostat) but NO pacer-only arm: nothing in the tag is `outer_yield_m4` with the question port
off. So the tag alone cannot separate

    pacer-alone   from   selection-alone   from   their interaction,

and any additivity reading has to borrow the pacer-alone column from the donor tag. This file
computes that column from the banked `cr3_s0` logs on EXACTLY the statistics
`analyze_antiphon.py` reports, so the three marginals sit in one table rather than being
eyeballed across differently-computed numbers.

WHAT MAKES THE BORROW UNUSUALLY TIGHT, AND WHAT IT STILL DOES NOT FIX. The baseline is the same
trajectory on both sides: `an_s0/q_exo` replays `cr3_s0/anchor_long` at 0.000e+00 over all 201
cycles and 13 series (reduction §0), and the two tags trained an identical substrate (`read_acc`,
both stale-buffer numbers exact). So `pacer-alone` and `selection-alone` are differences against
the *same* baseline trajectory, which is more than a generic cross-tag comparison gets. What it
does NOT fix: `outer_yield_m4` was never run under `antiphon.py`, and the G-F gate certified the
fork against `anchor_long` and `given_c1` only. The interaction reading is therefore a CROSS-TAG
reading and is labelled as one everywhere it appears.

CPU only, no GPU, no Modal, no new runs. Run from experiments/:
    PYTHONPATH=. python3 rhm/practice/antiphon/pacer_marginal.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)
AN = os.path.join(HERE, "figures", "an_s0")
AN1 = os.path.join(HERE, "figures", "an_s1")     # [an_s1] the IN-TAG pacer-only arm
CR = os.path.join(PRACTICE, "crescendo", "figures", "cr3_s0")


def load(root, arm):
    with open(os.path.join(root, arm, "results.json")) as f:
        return json.load(f)


def setup(root):
    with open(os.path.join(root, "setup.json")) as f:
        return json.load(f)


def truth_sets(su):
    return {int(l): {tuple(int(x) for x in r) for r in d["flat"]}
            for l, d in (su.get("true_tables") or {}).items()}


def era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def recovered(lg, refs, n_era=5):
    """`analyze_antiphon.py` §6, verbatim: mean `e` over an era's last three cycles,
    normalised (stale - e) / (stale - floor)."""
    st, fl = refs["stale"], refs["floor"]
    out = []
    for j in range(n_era):
        idx = era_idx(lg, j)
        if not idx:
            out.append(None)
            continue
        e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
        out.append((st[j] - e) / (st[j] - fl[j]))
    return out


def l4_share(lg, n_era=5):
    """`analyze_antiphon.py` §8, verbatim: the share of the priced beam's entry selections
    falling at each macro level, averaged over an era's last three cycles."""
    by = {}
    for i, e in enumerate(lg.get("entry") or []):
        beam = ((e or {}).get("hist") or {}).get("beam") or {}
        if not beam:
            continue
        tot = sum(sum(h) for h in beam.values())
        if tot <= 0:
            continue
        by.setdefault(lg["era"][i], []).append(
            {k: float(sum(h)) / tot for k, h in beam.items()})
    out = []
    for j in range(1, n_era + 1):
        v = by.get(j)
        out.append(float(np.mean([d.get("4", 0.0) for d in v[-3:]])) if v else None)
    return out


def keyset(lg, i, level):
    if level == 4 and lg.get("gy"):
        st = lg["gy"][i] or {}
    else:
        st = (lg["miner"][i] or {}).get(str(level)) or {}
    if not st:
        return set()
    return {tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])}


def profile(root, arm, su):
    r = load(root, arm)
    lg, T = r["log"], truth_sets(su)
    cc = {int(e["level"]): int(e["cycle"]) for e in r["events"] if e["kind"] == "commit"}
    tab = {int(e["level"]): {"n": e.get("n_entries"), "recall": e.get("tab_recall"),
                             "prec": e.get("tab_precision")}
           for e in r["events"] if e["kind"] == "commit"}
    last = len(lg["cycle"]) - 1
    end = {}
    for lv in (2, 3, 4):
        if lv not in T:
            continue
        ks = keyset(lg, last, lv)
        end[lv] = {"at_sup": len(ks), "true": len(ks & T[lv]),
                   "recall": len(ks & T[lv]) / len(T[lv])}
    return {"arm": arm, "n_cycles": len(lg["cycle"]), "t_cum": float(lg["t_cum"][-1]),
            "rf": recovered(lg, su["refs"]), "l4_share": l4_share(lg),
            "commits": cc, "tab": tab, "end": end}


def dsub(a, b):
    return [None if (x is None or y is None) else x - y for x, y in zip(a, b)]


def fmt(v, w=8, p=3):
    if v is None:
        return "-".rjust(w)
    return (f"{v:+.{p}f}" if isinstance(v, float) else str(v)).rjust(w)


def main():
    su_an, su_cr = setup(AN), setup(CR)

    print("=" * 92)
    print("THE PACER-ALONE MARGINAL — computed on `an_s0`'s own statistics")
    print("=" * 92)

    # --- the like-for-like precondition, checked rather than assumed -----------------------
    same_refs = all(np.allclose(np.asarray(su_an["refs"][k], float),
                                np.asarray(su_cr["refs"][k], float))
                    for k in su_an["refs"] if k != "macro_true")
    same_sub = (su_an["read_acc"] == su_cr["read_acc"]
                and su_an.get("stale_random_blocks") == su_cr.get("stale_random_blocks")
                and su_an.get("stale_task_matched") == su_cr.get("stale_task_matched"))
    print(f"\n[pre] refs identical across tags: {same_refs}   substrate identical: {same_sub}")
    print(f"      read_acc {su_an['read_acc']} / {su_cr['read_acc']}   "
          f"stale {su_an.get('stale_task_matched')} / {su_cr.get('stale_task_matched')}")
    print("      (and reduction §0: an_s0/q_exo replays cr3_s0/anchor_long at 0.000e+00 over")
    print("       all 201 cycles and 13 series, so BOTH baselines are the same trajectory)")

    # [an_s1] if the in-tag pacer-only arm has landed, it REPLACES the borrowed column and the
    # lane stops being cross-tag. Its own certification (whether it replays
    # `cr3_s0/outer_yield_m4` at 0.000e+00) is computed here too, on the same 13 series the
    # reduction's §0 uses, so this file stands alone.
    in_tag = os.path.isfile(os.path.join(AN1, "outer_yield_m4", "results.json"))
    cert = None
    P = {
        "cr3_s0/anchor_long": profile(CR, "anchor_long", su_cr),
        "cr3_s0/outer_yield_m4": profile(CR, "outer_yield_m4", su_cr),
        "an_s0/q_exo": profile(AN, "q_exo", su_an),
        "an_s0/q_bisect": profile(AN, "q_bisect", su_an),
        "an_s0/q_bisect_loop": profile(AN, "q_bisect_loop", su_an),
    }
    if in_tag:
        su_a1 = setup(AN1)
        P["an_s1/outer_yield_m4"] = profile(AN1, "outer_yield_m4", su_a1)
        a = load(AN1, "outer_yield_m4")["log"]
        b = load(CR, "outer_yield_m4")["log"]
        series = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve",
                  "e_practice", "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
        worst = 0.0
        for k in series:
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            L = min(len(x), len(y))
            worst = max(worst, float(np.abs(x[:L] - y[:L]).max()) if L else float("nan"))
        cert = {"worst": worst, "lifetimes": [len(a["cycle"]), len(b["cycle"])],
                "pass": bool(worst == 0.0 and len(a["cycle"]) == len(b["cycle"]))}
        print(f"\n[cert] an_s1/outer_yield_m4 vs cr3_s0/outer_yield_m4 over "
              f"{cert['lifetimes']} cycles: max|diff| = {worst:.3e} -> "
              f"{'PASS, the pacer lane is IN-TAG' if cert['pass'] else 'FAIL, the lane stays CROSS-TAG'}")

    print("\n[0] the four trajectories, absolute")
    print(f"  {'arm':26s}{'cyc':>5}{'L2c':>5}{'L3c':>5}{'L4c':>5}"
          f"{'L2 rec':>8}{'L3 rec':>8}{'L4 rec':>8}"
          + "".join(f"{'rf e' + str(j + 1):>9}" for j in range(5))
          + "".join(f"{'L4sh e' + str(j + 1):>10}" for j in (2, 3, 4)))
    for k, p in P.items():
        print(f"  {k:26s}{p['n_cycles']:>5}"
              f"{p['commits'].get(2, 0):>5}{p['commits'].get(3, 0):>5}"
              f"{p['commits'].get(4, 0):>5}"
              + "".join(f"{(p['end'].get(l) or {}).get('recall', float('nan')):>8.3f}"
                        for l in (2, 3, 4))
              + "".join(f"{x:>9.3f}" if x is not None else "-".rjust(9) for x in p["rf"])
              + "".join(f"{x:>10.3f}" if x is not None else "-".rjust(10)
                        for x in [p["l4_share"][j] for j in (2, 3, 4)]))

    # --- THE TABLE ------------------------------------------------------------------------
    if in_tag and cert and cert["pass"]:
        pacer = ("an_s1/outer_yield_m4", "an_s0/q_exo")
        pacer_label = "pacer-alone  [IN-TAG]"
    else:
        pacer = ("cr3_s0/outer_yield_m4", "cr3_s0/anchor_long")
        pacer_label = "pacer-alone  [CROSS-TAG]"
    seln = ("an_s0/q_bisect", "an_s0/q_exo")
    comp = ("an_s0/q_bisect_loop", "an_s0/q_exo")
    lanes = [(pacer_label, pacer), ("selection-alone", seln), ("composed", comp)]

    print("\n[1] THE MARGINALS, one statistic per block. Baseline of every lane is the SAME")
    print("    trajectory (cr3_s0/anchor_long == an_s0/q_exo, bit-identical).")
    rows = {}
    print(f"\n  {'lane':28s}" + "".join(f"{'rf e' + str(j + 1):>10}" for j in range(5)))
    for name, (a, b) in lanes:
        d = dsub(P[a]["rf"], P[b]["rf"])
        rows.setdefault(name, {})["rf"] = d
        print(f"  {name:28s}" + "".join(fmt(x, 10) for x in d))
    print(f"  {'sum of the two singles':28s}"
          + "".join(fmt(None if (x is None or y is None) else x + y, 10)
                    for x, y in zip(rows[pacer_label]["rf"],
                                    rows["selection-alone"]["rf"])))
    print(f"  {'composed - that sum':28s}"
          + "".join(fmt(None if (c is None or x is None or y is None) else c - (x + y), 10)
                    for c, x, y in zip(rows["composed"]["rf"],
                                       rows[pacer_label]["rf"],
                                       rows["selection-alone"]["rf"])))

    print(f"\n  {'lane':28s}" + "".join(f"{'L4sh e' + str(j + 1):>10}" for j in (2, 3, 4)))
    for name, (a, b) in lanes:
        d = dsub(P[a]["l4_share"], P[b]["l4_share"])
        rows[name]["l4_share"] = d
        print(f"  {name:28s}" + "".join(fmt(d[j], 10) for j in (2, 3, 4)))

    print(f"\n  {'lane':28s}{'L2 commit':>12}{'L3 commit':>12}{'L4 commit':>12}"
          f"{'L4 book n':>11}{'L4 book prec':>14}")
    for name, (a, b) in lanes:
        ca, cb = P[a]["commits"], P[b]["commits"]
        ta, tb = P[a]["tab"], P[b]["tab"]
        d = {"L2": ca.get(2, 0) - cb.get(2, 0), "L3": ca.get(3, 0) - cb.get(3, 0),
             "L4": ca.get(4, 0) - cb.get(4, 0),
             "book_n": (ta.get(4, {}).get("n"), tb.get(4, {}).get("n")),
             "book_p": (ta.get(4, {}).get("prec"), tb.get(4, {}).get("prec"))}
        rows[name]["commits"] = d
        print(f"  {name:28s}{d['L2']:>+12}{d['L3']:>+12}{d['L4']:>+12}"
              f"{str(d['book_n'][0]) + ' vs ' + str(d['book_n'][1]):>11}"
              f"{str(round(d['book_p'][0], 3) if d['book_p'][0] is not None else None) + ' vs ' + str(round(d['book_p'][1], 3) if d['book_p'][1] is not None else None):>14}")

    print(f"\n  {'lane':28s}{'d L2 recall':>13}{'d L3 recall':>13}{'d L4 recall':>13}"
          f"{'d L4 true@sup':>15}")
    for name, (a, b) in lanes:
        d = {}
        for l in (2, 3, 4):
            ea, eb = P[a]["end"].get(l), P[b]["end"].get(l)
            d[l] = None if not (ea and eb) else ea["recall"] - eb["recall"]
        dt4 = (P[a]["end"][4]["true"] - P[b]["end"][4]["true"]) if 4 in P[a]["end"] else None
        d["true4"] = dt4
        rows[name]["recall"] = d
        print(f"  {name:28s}{fmt(d[2], 13)}{fmt(d[3], 13)}{fmt(d[4], 13)}"
              f"{(('%+d' % dt4) if dt4 is not None else '-'):>15}")

    print("\n[2] CAVEATS, stated where the numbers are")
    if in_tag and cert and cert["pass"]:
        print("    (a) The pacer-alone column is IN-TAG: `an_s1` ran `outer_yield_m4`'s policy")
        print("        under `antiphon.py` with `question_mode` unset and replayed")
        print("        `cr3_s0/outer_yield_m4` at 0.000e+00 over its whole trajectory. The")
        print("        interaction rows are an in-tag reading.")
    else:
        print("    (a) The pacer-alone column is CROSS-TAG: `outer_yield_m4` was run under")
        print("        `crescendo.py`, never under `antiphon.py`. Additivity/interaction")
        print("        readings off the last two rows of the rf block are cross-tag readings,")
        print("        not in-tag claims.")
    print("    (b) INDEPENDENT of (a), and not fixed by any gating run: the two pacer lanes")
    print("        are NOT lifetime-matched. Loop arms leave eras early, so against the")
    print("        schedule baseline the pacer arm runs short. crescendo's construction makes")
    print("        the schedule arm the lifetime ceiling, so a POSITIVE loop delta cannot have")
    print("        been bought with time — but a NEGATIVE one may be a time cost. Read the")
    print("        pacer-alone negatives with that, and the composed positives as conservative.")

    out = {"same_refs": bool(same_refs), "same_substrate": bool(same_sub),
           "in_tag_pacer": bool(in_tag), "certification": cert, "pacer_lane": pacer_label,
           "profiles": P, "marginals": rows}
    with open(os.path.join(HERE, "pacer_marginal.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(HERE, 'pacer_marginal.json')}")


if __name__ == "__main__":
    main()
