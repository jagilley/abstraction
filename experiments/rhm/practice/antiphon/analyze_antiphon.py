"""THE REDUCTION for `antiphon` — the question port.

Sections, in the order the round's claims are made:

  §0  FIDELITY   `q_exo` vs the banked `cr3_s0/anchor_long`, full scale, every logged series.
                 This is the round's load-bearing gate: it says the menu machinery, the reader
                 forward over the whole menu and the value forward over the whole menu are
                 INERT, so any difference between the other arms and `q_exo` is selection and
                 nothing else. Plus in-tag twin windows.
  §1  CONTROLS   what each arm pinned: the per-cycle difficulty quota, the mined volume, the
                 priced spend, the lifetime. `q_comp_free`'s difficulty DEVIATION is reported
                 here as a first-class number, because the deviation is that arm's definition.
  §2  DOSE       delivered vs designed key, per era per arm — `wd_s0`'s lesson as a measurement.
  §3  CLIMB      at-support trajectories (all / true / junk) at L2..L4, against BOTH the cycle
                 clock and the cumulative priced-grounding clock, with Phase 0's predicted band.
  §4  TABLES     what each commit froze: size, recall, precision; buildable true L4 over the
                 frozen L3 (the r^2 wall, by the substrate's own arithmetic); the L4 book.
  §5  CLOCKS     cycles-to-commit per level and cycles-to-recall-theta — "climbing speed".
  §6  VALUE      the three clocks and deep-era recovered fraction against `crescendo`'s floors.
  §7  NERDSNIPE  junk share of what each arm aimed at and of what it banked; `endo` vs `novel`;
                 the delivery ledger's own state.
  §8  TRUST      pi's proposal mass by level (`crescendo` finding 3's instrument).
  §9  USE        `ess_use` — the effective number of table entries in use, from `entry.hist`
                 (the provenance lane's by-product): does selection concentrate USE as well as
                 observation?

Run from experiments/:
    python3 rhm/practice/antiphon/analyze_antiphon.py --tag an_s0 --fetch --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_antiphon"
DONOR = os.path.join(PRACTICE, "crescendo", "figures", "cr3_s0")

GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
SUPPORT = 3

# [antiphon-s2] every loop-paced question arm and the schedule-paced arm it is a twin of.
# For a loop arm the `q_exo` window is uninformative (the port acts on c1 by design), so the
# gate that says the PACER is the only thing that moved is the SIBLING window: same selector,
# same stream, differing in `commit` alone, therefore bit-identical until the loop's own first
# action. Reported as a cycle number, not a boolean.
LOOP_SIBLING = {"q_bisect_loop": "q_bisect", "q_endo_loop": "q_endo",
                "q_comp_loop": "q_comp", "q_novel_loop": "q_novel"}


# --------------------------------------------------------------------------- #
def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def _load(root, name):
    p = os.path.join(root, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def _fmt(v, w=8, p=3):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.{p}f}".rjust(w)
    return str(v).rjust(w)


def _era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def buildable(keys, lower_flats, s=2):
    if not keys:
        return []
    half = len(next(iter(keys))) // s
    return [k for k in keys
            if all(k[i * half:(i + 1) * half] in lower_flats for i in range(s))]


def truth_sets(setup):
    """The true tables the run itself wrote into `setup.json` — no rule regeneration needed."""
    return {int(l): {tuple(int(x) for x in r) for r in d["flat"]}
            for l, d in (setup.get("true_tables") or {}).items()}


def keyset(lg, i, level):
    src = lg["gy"] if level == 4 and lg.get("gy") else lg["miner"]
    st = src[i] if level == 4 and lg.get("gy") else ((lg["miner"][i] or {}).get(str(level)) or {})
    if not st:
        return set(), {}
    return ({tuple(int(x) for x in k) for k in (st.get("keys_at_support") or [])}, st)


# --------------------------------------------------------------------------- #
def sec0_fidelity(A, order, out, setup=None):
    print("\n" + "=" * 88)
    print("§0  FIDELITY — is the menu machinery inert?")
    print("=" * 88)
    # [antiphon-m] the cross-tag replay is a statement about the PORT, at the donor's own
    # operating point. A tag that moves `n_pr` or `pr_width` is a different point on purpose,
    # so `q_exo` cannot and must not replay `cr3_s0/anchor_long` there — the gate is not
    # failed, it is inapplicable, and saying "FAIL" would be a false alarm. The in-process G-F
    # at the metered knob (`an_gf_m`) is what covers the fork there.
    cfgm = (setup or {}).get("config") or {}
    metered = bool(cfgm) and (int(cfgm.get("n_pr", 64)) != 64
                              or int(cfgm.get("pr_width", 16)) != 16)
    ref = _load(os.path.join(DONOR, "anchor_long"), "results.json")
    if metered:
        print(f"  [n/a] this tag runs at n_pr={cfgm.get('n_pr')} pr_width={cfgm.get('pr_width')},"
              f" not the donor's 64/16, so the cross-tag `q_exo` replay does NOT apply.")
        print("        The fork is gated at THIS knob by the in-process G-F (`an_gf_m`);")
        print("        `q_exo` here is the tag's own metered null, not a replay of anything.")
        out["fidelity"] = {"applicable": False, "n_pr": cfgm.get("n_pr"),
                           "pr_width": cfgm.get("pr_width")}
    elif ref is None:
        print("  [skip] cr3_s0/anchor_long not fetched under crescendo/figures/")
    elif "q_exo" not in A:
        print("  [skip] q_exo not in this tag")
    else:
        a, b = A["q_exo"]["log"], ref["log"]
        worst, per = 0.0, {}
        for k in GF_SERIES:
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            L = min(len(x), len(y))
            per[k] = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            worst = max(worst, per[k])
        ev = lambda r: [(e["era"], e["level"], e["cycle"], e["n_entries"])
                        for e in r["events"] if e["kind"] == "commit"]
        same = ev(A["q_exo"]) == ev(ref)
        print(f"  q_exo vs cr3_s0/anchor_long over {min(len(a['cycle']), len(b['cycle']))} "
              f"cycles, {len(GF_SERIES)} series")
        print(f"    max|diff| = {worst:.3e}   commits equal: {same}")
        print(f"    per series: " + "  ".join(f"{k}:{per[k]:.1e}" for k in GF_SERIES))
        print(f"    VERDICT: {'PASS — the menu machinery is inert at full scale' if worst == 0.0 and same else 'FAIL'}")
        out["fidelity"] = {"worst": worst, "commits_equal": bool(same), "per_series": per}

    # [an_s1] §0b — the merged pacer-only arm against ITS donor, full scale. `an_s1` runs
    # `outer_yield_m4`'s policy with `question_mode` unset, so if this is 0.000e+00 the fork
    # reproduces the donor's THERMOSTAT path over its whole 162-cycle trajectory — which is
    # what promotes §6b's pacer-alone column from a cross-tag borrow to an in-tag column.
    if "outer_yield_m4" in A:
        ref2 = _load(os.path.join(DONOR, "outer_yield_m4"), "results.json")
        if ref2 is None:
            print("\n  [skip] cr3_s0/outer_yield_m4 not fetched")
        else:
            a, b = A["outer_yield_m4"]["log"], ref2["log"]
            worst, per = 0.0, {}
            for k in GF_SERIES:
                x, y = np.asarray(a[k], float), np.asarray(b[k], float)
                L = min(len(x), len(y))
                per[k] = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
                worst = max(worst, per[k])
            ev = lambda r: [(e["era"], e["level"], e["cycle"], e["n_entries"])
                            for e in r["events"] if e["kind"] == "commit"]
            same = ev(A["outer_yield_m4"]) == ev(ref2)
            lens = (len(a["cycle"]), len(b["cycle"]))
            print(f"\n  §0b  an_s1/outer_yield_m4 vs cr3_s0/outer_yield_m4 "
                  f"(lifetimes {lens[0]} vs {lens[1]})")
            print(f"    max|diff| = {worst:.3e}   commits equal: {same}")
            print(f"    per series: " + "  ".join(f"{k}:{per[k]:.1e}" for k in GF_SERIES))
            ok = (worst == 0.0 and same and lens[0] == lens[1])
            print(f"    VERDICT: {'PASS — the fork reproduces the donors thermostat path at full scale; §6b promotes to IN-TAG' if ok else 'FAIL — §6b stays cross-tag; diagnose before reading the interaction'}")
            out["cert_pacer"] = {"worst": worst, "commits_equal": bool(same),
                                 "lifetimes": lens, "pass": bool(ok), "per_series": per}

    print("\n  In-tag twin windows: every arm vs q_exo up to its own first divergence")
    if "q_exo" in A:
        base = A["q_exo"]["log"]
        for a in order:
            if a == "q_exo":
                continue
            lg = A[a]["log"]
            L = min(len(lg["cycle"]), len(base["cycle"]))
            first = None
            for i in range(L):
                if any(abs(float(lg[k][i]) - float(base[k][i])) > 0 for k in GF_SERIES):
                    first = i + 1
                    break
            note = ("by design c1 — the port acts on cycle 1"
                    if (A[a].get("question_mode") or None) else
                    "port OFF in this arm — this is the donor's own in-tag twin gate: "
                    "bit-identical to the schedule arm until its own first loop action")
            print(f"    {a:16s} first divergence: c{first}  ({note})" if first
                  else f"    {a:16s} never diverged")

    # [antiphon-s2] THE SIBLING TWIN WINDOW — the gate for a loop-paced question arm.
    sib = {}
    for a in order:
        b = LOOP_SIBLING.get(a)
        if b is None or b not in A:
            continue
        la, lb = A[a]["log"], A[b]["log"]
        L = min(len(la["cycle"]), len(lb["cycle"]))
        first = None
        for i in range(L):
            if any(abs(float(la[k][i]) - float(lb[k][i])) > 0 for k in GF_SERIES):
                first = i + 1
                break
        # `loop_actions` is not carried in the final results.json, so the action clock is
        # read off the EVENTS both arms write: an action is a commit or an era advance, and
        # divergence must begin at the first one either arm takes, never before it.
        act = lambda r: min((int(e["cycle"]) for e in r["events"]
                             if e["kind"] in ("commit", "advance")), default=None)
        act_a, act_b = act(A[a]), act(A[b])
        first_act = min([q for q in (act_a, act_b) if q is not None], default=None)
        ok = (first is None) or (first_act is not None and first >= first_act)
        sib[a] = {"sibling": b, "first_divergence": first, "first_action": first_act,
                  "first_action_loop": act_a, "first_action_sched": act_b,
                  "overlap": L, "pass": bool(ok)}
    if sib:
        print("\n  [antiphon-s2] SIBLING twin windows: each loop-paced question arm vs its")
        print("  schedule-paced sibling (same selector, same stream, `commit` the only diff)")
        for a, q in sib.items():
            print(f"    {a:16s} vs {q['sibling']:12s} first divergence: "
                  f"c{q['first_divergence']}   first action (either arm): c{q['first_action']}"
                  f"  [loop c{q['first_action_loop']} / sched c{q['first_action_sched']}]"
                  f"   -> {'PASS (identical until the first action)' if q['pass'] else 'FAIL'}")
        out["sibling_twin"] = sib


def sec1_controls(A, order, setup, out):
    print("\n" + "=" * 88)
    print("§1  CONTROLS — what each arm pinned")
    print("=" * 88)
    cap = (setup["config"] or {}).get("mine_cap")
    print(f"  {'arm':16s}{'mode':11s}{'cyc':>5}{'quota_ok':>10}{'vol_ok':>8}"
          f"{'d*_sel':>8}{'d*_menu':>9}{'dev':>8}{'t_cum':>12}{'mined':>8}")
    rows = {}
    for a in order:
        lg = A[a]["log"]
        q = [z for z in (lg.get("q") or []) if z]
        if not q:
            print(f"  {a:16s}{'(port off)':11s}{len(lg['cycle']):>5}")
            continue
        qok = float(np.mean([z["quota_ok"] for z in q]))
        vol = [z.get("n_mined") for z in q if z.get("n_mined") is not None]
        volok = float(np.mean([n >= cap for n in vol])) if vol else None
        ds = float(np.mean([z["d_mean"] for z in q]))
        dm = float(np.mean([z["d_mean_menu"] for z in q]))
        rows[a] = {"quota_ok": qok, "vol_ok": volok, "d_sel": ds, "d_menu": dm,
                   "d_dev": ds - dm, "t_cum": float(lg["t_cum"][-1]),
                   "n_cycles": len(lg["cycle"]), "mined": int(sum(vol)),
                   "mode": q[0]["mode"]}
        print(f"  {a:16s}{q[0]['mode']:11s}{len(lg['cycle']):>5}{qok:>10.3f}"
              f"{(volok if volok is not None else float('nan')):>8.3f}"
              f"{ds:>8.3f}{dm:>9.3f}{ds - dm:>+8.3f}{lg['t_cum'][-1]:>12.0f}{sum(vol):>8}")
    print("\n  quota_ok = fraction of cycles whose selected d* histogram equalled the quota")
    print("  (the menu head's, i.e. the donor's own mix). vol_ok = fraction of cycles that")
    print("  mined the full `mine_cap`. `dev` is the difficulty deviation: ~0 by construction")
    print("  in every arm but `q_comp_free`, where it IS the arm's definition.")
    out["controls"] = rows


def sec2_dose(A, order, out):
    print("\n" + "=" * 88)
    print("§2  DELIVERED DOSE vs DESIGN DOSE  (`wd_s0`'s lesson, as an instrument)")
    print("=" * 88)
    print("  P(the key the miner recorded == the key the question designed), per era per arm,")
    print("  measured in EVERY arm including the ones that never see a designed key.")
    eras = sorted({z["cycle"] and e for a in order for e in []} or set())
    rows = {}
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j):>12s}" for j in (1, 2, 3, 4, 5))
          + f"{'all':>12s}{'deliv_true':>12s}{'credit':>9s}")
    for a in order:
        lg = A[a]["log"]
        q = [z for z in (lg.get("q") or []) if z and z.get("dose_hit") is not None]
        if not q:
            continue
        by = {}
        for i, z in enumerate(lg.get("q") or []):
            if not z or z.get("dose_hit") is None:
                continue
            by.setdefault(lg["era"][i], []).append(z)
        line = ""
        cell = {}
        for j in (1, 2, 3, 4, 5):
            zz = by.get(j)
            cell[j] = float(np.mean([z["dose_hit"] for z in zz])) if zz else None
            line += (f"{cell[j]:>12.3f}" if cell[j] is not None else "-".rjust(12))
        alld = float(np.mean([z["dose_hit"] for z in q]))
        dt = float(np.mean([z["delivered_true"] for z in q]))
        cr = float(np.mean([z["credit_rate"] for z in q if z.get("credit_rate") is not None]))
        rows[a] = {"by_era": cell, "all": alld, "delivered_true": dt, "credit": cr}
        print(f"  {a:16s}{line}{alld:>12.3f}{dt:>12.3f}{cr:>9.3f}")
    print("\n  `deliv_true` = fraction of mined observations that were a TRUE table row at all")
    print("  (Phase 0's ceiling on this, from the grammar's own parse ambiguity, is 0.42 at L3")
    print("  and 0.18 at L4 under a uniform draw). `credit` = fraction of observations that")
    print("  advanced a key not already at support — the ledger's currency.")
    out["dose"] = rows


def sec3_climb(A, order, setup, out):
    print("\n" + "=" * 88)
    print("§3  THE CLIMB — at-support coverage against the cycle clock and the PRICED clock")
    print("=" * 88)
    T = truth_sets(setup)
    rows = {}
    for level in (2, 3, 4):
        if level not in T:
            continue
        print(f"\n  LEVEL {level}  (|true| = {len(T[level])})")
        print(f"  {'arm':16s}{'N_obs':>7}{'@sup':>6}{'true':>6}{'junk':>6}"
              f"{'recall':>8}{'prec':>7}   trajectory of TRUE@sup by decile of cycles")
        for a in order:
            lg = A[a]["log"]
            n = len(lg["cycle"])
            traj, last = [], None
            for i in range(n):
                ks, st = keyset(lg, i, level)
                t = len(ks & T[level])
                traj.append(t)
                last = (st, ks, t)
            if last is None or not last[0]:
                continue
            st, ks, t = last
            junk = len(ks) - t
            step = max(1, n // 10)
            spark = "/".join(str(traj[i]) for i in range(step - 1, n, step))
            rows.setdefault(str(level), {})[a] = {
                "n_obs": int(st["n_obs"]), "at_sup": len(ks), "true": t, "junk": junk,
                "recall": t / len(T[level]),
                "precision": (t / len(ks)) if ks else None,
                "traj_true": traj, "t_cum": [float(x) for x in lg["t_cum"]]}
            print(f"  {a:16s}{st['n_obs']:>7}{len(ks):>6}{t:>6}{junk:>6}"
                  f"{t / len(T[level]):>8.3f}"
                  f"{(t / len(ks)) if ks else float('nan'):>7.3f}   {spark}")
        # the priced-clock comparison: true@sup at the SMALLEST common cumulative spend
        gmin = min(rows[str(level)][a]["t_cum"][-1] for a in rows[str(level)]) \
            if rows.get(str(level)) else None
        if gmin:
            print(f"\n    at matched cumulative priced spend ({gmin:.0f} groundings):")
            for a in rows[str(level)]:
                tc = np.asarray(rows[str(level)][a]["t_cum"])
                i = int(np.searchsorted(tc, gmin))
                i = min(i, len(tc) - 1)
                print(f"      {a:16s} true@sup {rows[str(level)][a]['traj_true'][i]:>4} "
                      f"at c{i + 1}")
    out["climb"] = {lv: {a: {k: v for k, v in d.items() if k not in ("traj_true", "t_cum")}
                         for a, d in arms.items()} for lv, arms in rows.items()}
    out["climb_traj"] = rows
    return rows


def sec4_tables(A, order, setup, out):
    print("\n" + "=" * 88)
    print("§4  WHAT THE COMMITS FROZE, and the r^2 wall")
    print("=" * 88)
    T = truth_sets(setup)
    base_l1 = None
    rows = {}
    print("  The `bldbl` row is end-of-run: L4 keys at support whose BOTH halves are in that")
    print("  arm's end-of-run at-support L3 set (the LIVE set, not the committed frozen table),")
    print("  by the substrate's own `buildable()` ratchet. The r^2 figure beside it is Phase 0's")
    print("  law evaluated at that arm's L3 recall, for comparison.")
    print(f"  {'arm':16s}{'lvl':>4}{'cycle':>7}{'entries':>9}{'recall':>8}{'prec':>7}"
          f"{'bldbl L4|live L3':>20}{'true':>6}")
    for a in order:
        lg, ev = A[a]["log"], A[a]["events"]
        commits = [e for e in ev if e["kind"] == "commit"]
        rows[a] = {"commits": [], "buildable_l4": None}
        frozen3 = None
        for e in commits:
            lv, c = int(e["level"]), int(e["cycle"])
            i = c - 1
            ks, _ = keyset(lg, min(i, len(lg["cycle"]) - 1), lv)
            row = {"level": lv, "cycle": c, "n_entries": e.get("n_entries"),
                   "recall": e.get("tab_recall"), "precision": e.get("tab_precision")}
            rows[a]["commits"].append(row)
            if lv == 3:
                frozen3 = {k for k in ks if k in T.get(3, set())}
            b = ""
            if lv == 4 and frozen3 is not None:
                pass
            print(f"  {a:16s}{lv:>4}{c:>7}{_fmt(row['n_entries'], 9)}"
                  f"{_fmt(row['recall'], 8)}{_fmt(row['precision'], 7)}")
        # the r^2 wall, on this arm's own frozen L3 and its own L4 key stream, end of run
        if 3 in T and 4 in T:
            i = len(lg["cycle"]) - 1
            k3, _ = keyset(lg, i, 3)
            k4, _ = keyset(lg, i, 4)
            fz3 = {k for k in k3}
            b = buildable(sorted(k4), fz3)
            bt = [k for k in b if k in T[4]]
            rows[a]["buildable_l4"] = {"n": len(b), "true": len(bt),
                                       "l3_recall": len(k3 & T[3]) / len(T[3])}
            print(f"  {a:16s}{'':4}{'end':>7}{'':9}{'':8}{'':7}"
                  f"{len(b):>20}{len(bt):>6}   (L3 recall "
                  f"{len(k3 & T[3]) / len(T[3]):.3f} -> r^2*816 = "
                  f"{(len(k3 & T[3]) / len(T[3])) ** 2 * len(T[4]):.0f})")
    out["tables"] = rows


def sec5_clocks(A, order, out):
    print("\n" + "=" * 88)
    print("§5  CLIMBING SPEED — cycles to each commit, and cycles to a recall threshold")
    print("=" * 88)
    print(f"  {'arm':16s}{'L2 commit':>11}{'L3 commit':>11}{'L4 commit':>11}"
          f"{'cyc':>6}{'t_cum':>12}")
    rows = {}
    for a in order:
        lg, ev = A[a]["log"], A[a]["events"]
        cc = {int(e["level"]): int(e["cycle"]) for e in ev if e["kind"] == "commit"}
        rows[a] = {"commits": cc, "n_cycles": len(lg["cycle"]),
                   "t_cum": float(lg["t_cum"][-1])}
        print(f"  {a:16s}{_fmt(cc.get(2), 11)}{_fmt(cc.get(3), 11)}{_fmt(cc.get(4), 11)}"
              f"{len(lg['cycle']):>6}{lg['t_cum'][-1]:>12.0f}")
    out["clocks"] = rows


def sec6_value(A, order, setup, out):
    print("\n" + "=" * 88)
    print("§6  THE THREE CLOCKS and deep-era recovered fraction")
    print("=" * 88)
    refs = setup["refs"]
    st, fl = refs["stale"], refs["floor"]
    n_era = len(st)
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>17s}" for j in range(n_era)))
    rows = {}
    for a in order:
        lg = A[a]["log"]
        row, rec = "", []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                rec.append(None); row += "-".rjust(17); continue
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            r = (st[j] - e) / (st[j] - fl[j])
            rec.append(r)
            row += f"{e:.3f} /{r:+.3f}".rjust(17)
        rows[a] = rec
        print(f"  {a:16s}{row}")
    print(f"  {'stale':14s}" + "".join(f"{st[j]:>17.3f}" for j in range(n_era)))
    print(f"  {'floor':14s}" + "".join(f"{fl[j]:>17.3f}" for j in range(n_era)))
    if "q_exo" in rows:
        print(f"\n  DELTA TO q_exo (the exogenous ladder, and the lifetime ceiling)")
        print("  Read against the MEASURED depth-6 stream floors (census finding 7): 0.087 in")
        print("  the earning family, 0.083/0.148/0.344 for eras 3/4/5 in the given family, and")
        print("  crescendo's own in-node era-4/5 displacement floors 0.030/0.410.")
        print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
        for a in order:
            if a == "q_exo":
                continue
            d = [(None if rows[a][j] is None or rows["q_exo"][j] is None
                  else rows[a][j] - rows["q_exo"][j]) for j in range(n_era)]
            print(f"  {a:16s}" + "".join(_fmt(x, 10) for x in d))
    out["value"] = rows


def sec7_nerdsnipe(A, order, setup, out):
    print("\n" + "=" * 88)
    print("§7  THE NERDSNIPE — what each arm AIMED at, and what it BANKED")
    print("=" * 88)
    print("  Phase 0: the grammar's own parse ambiguity makes 104 junk keys reachable at L3")
    print("  and 7573 at L4 — 0.58 and 0.82 of the observation mass under a uniform draw. Junk")
    print("  is maximally novel, forever, so a novelty-greedy judge maximises it by definition.")
    T = truth_sets(setup)
    print(f"\n  {'arm':16s}{'designed_true':>15}{'designed_needy':>15}"
          f"{'banked_true_L3':>16}{'banked_junk_L3':>16}{'banked_true_L4':>16}"
          f"{'banked_junk_L4':>16}")
    rows = {}
    for a in order:
        lg = A[a]["log"]
        q = [z for z in (lg.get("q") or []) if z]
        i = len(lg["cycle"]) - 1
        k3, _ = keyset(lg, i, 3)
        k4, _ = keyset(lg, i, 4)
        t3 = len(k3 & T.get(3, set())); t4 = len(k4 & T.get(4, set()))
        dt = float(np.mean([z["n_designed_true"] / max(z["n_pr"], 1) for z in q])) if q else None
        dn = (float(np.mean([(z["n_designed_true"] - z["n_designed_banked"]) / max(z["n_pr"], 1)
                             for z in q])) if q else None)
        rows[a] = {"designed_true": dt, "designed_needy": dn,
                   "L3_true": t3, "L3_junk": len(k3) - t3,
                   "L4_true": t4, "L4_junk": len(k4) - t4,
                   "ledger": A[a].get("q_ledger")}
        print(f"  {a:16s}{_fmt(dt, 15)}{_fmt(dn, 15)}{t3:>16}{len(k3) - t3:>16}"
              f"{t4:>16}{len(k4) - t4:>16}")
    print("\n  delivery ledger (endo only):")
    for a in order:
        if A[a].get("q_ledger"):
            print(f"    {a:16s}{A[a]['q_ledger']}")
    out["nerdsnipe"] = rows


def sec8_trust(A, order, out):
    print("\n" + "=" * 88)
    print("§8  TRUST — pi's proposal mass by level (crescendo finding 3's instrument)")
    print("=" * 88)
    print("  Share of the PRICED beam's entry selections falling at each macro level, per era")
    print("  (`log['entry']['hist']['beam']`, the entry-identity instrument — the same source")
    print("  crescendo finding 3's monotone L4 mass came off). A level with no committed table")
    print("  is absent by construction.")
    rows = {}
    print(f"  {'arm':16s}{'era':>5}   per-level share of beam entry selections")
    for a in order:
        lg = A[a]["log"]
        ent = lg.get("entry") or []
        by = {}
        for i, e in enumerate(ent):
            beam = ((e or {}).get("hist") or {}).get("beam") or {}
            if not beam:
                continue
            tot = sum(sum(h) for h in beam.values())
            if tot <= 0:
                continue
            by.setdefault(lg["era"][i], []).append(
                {k: float(sum(h)) / tot for k, h in beam.items()})
        rows[a] = {}
        for j, v in sorted(by.items()):
            keys = sorted({k for d in v for k in d})
            m = {k: float(np.mean([d.get(k, 0.0) for d in v[-3:]])) for k in keys}
            rows[a][str(j)] = m
            print(f"  {a:16s}{j:>5}   " + "  ".join(f"L{k}:{x:.3f}" for k, x in m.items()))
    out["trust"] = rows


def sec9_use(A, order, out):
    print("\n" + "=" * 88)
    print("§9  ess_use — the effective number of table entries IN USE (provenance lane's)")
    print("=" * 88)
    print("  exp(entropy) of the per-entry selection histogram from the beam's own executions.")
    print("  Asks whether selection concentrates USE as well as observation.")
    rows = {}
    print(f"  {'arm':16s}" + "".join(f"{'L' + str(l):>12s}" for l in (2, 3, 4))
          + f"{'(n_entries)':>16s}")
    for a in order:
        ent = (A[a]["log"].get("entry") or [])
        last = next((e for e in reversed(ent) if e and e.get("hist")), None)
        if not last:
            continue
        beam = (last.get("hist") or {}).get("beam", {})
        cell, ns = {}, {}
        for lv, h in sorted(beam.items()):
            h = np.asarray(h, float)
            if h.sum() <= 0:
                cell[lv] = None; ns[lv] = int(len(h)); continue
            p = h / h.sum()
            p = p[p > 0]
            cell[lv] = float(np.exp(-(p * np.log(p)).sum()))
            ns[lv] = int(len(h))
        rows[a] = {"ess": cell, "n_entries": ns}
        print(f"  {a:16s}" + "".join(_fmt(cell.get(str(l)), 12) for l in (2, 3, 4))
              + "  " + str(ns))
    out["ess_use"] = rows


# --------------------------------------------------------------------------- #
def figures(tag, out, traj, setup):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = os.path.join(FIG, tag)
    os.makedirs(d, exist_ok=True)
    T = truth_sets(setup)
    order = list(out["controls"].keys()) or list(out["clocks"].keys())
    cols = plt.cm.tab10(np.linspace(0, 1, 10))

    # 1. coverage vs cycle, per level
    lv = [l for l in ("2", "3", "4") if l in traj]
    fig, ax = plt.subplots(1, len(lv), figsize=(5 * len(lv), 4), squeeze=False)
    for j, l in enumerate(lv):
        for i, a in enumerate(traj[l]):
            ax[0][j].plot(np.arange(1, len(traj[l][a]["traj_true"]) + 1),
                          traj[l][a]["traj_true"], color=cols[i % 10], label=a, lw=1.4)
        ax[0][j].set_title(f"L{l}: TRUE keys at support (of {len(T.get(int(l), []))})")
        ax[0][j].set_xlabel("cycle"); ax[0][j].grid(alpha=.3)
    ax[0][0].set_ylabel("true keys at support")
    ax[0][-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(d, "coverage_vs_cycle.png"), dpi=140)
    plt.close(fig)

    # 2. coverage vs cumulative priced spend
    fig, ax = plt.subplots(1, len(lv), figsize=(5 * len(lv), 4), squeeze=False)
    for j, l in enumerate(lv):
        for i, a in enumerate(traj[l]):
            ax[0][j].plot(traj[l][a]["t_cum"][:len(traj[l][a]["traj_true"])],
                          traj[l][a]["traj_true"], color=cols[i % 10], label=a, lw=1.4)
        ax[0][j].set_title(f"L{l}: TRUE@support vs PRICED groundings")
        ax[0][j].set_xlabel("cumulative groundings"); ax[0][j].grid(alpha=.3)
    ax[0][0].set_ylabel("true keys at support")
    ax[0][-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(d, "coverage_vs_priced.png"), dpi=140)
    plt.close(fig)

    # 3. the r^2 payoff plane: realized (L3 recall, buildable true L4) per arm
    fig, ax = plt.subplots(figsize=(6, 4.6))
    r = np.linspace(0, 1, 100)
    ax.plot(r, r ** 2 * len(T.get(4, [816])), "k--", lw=1, label="r$^2$ law")
    for i, a in enumerate(out.get("tables", {})):
        b = out["tables"][a].get("buildable_l4")
        if not b:
            continue
        ax.scatter([b["l3_recall"]], [b["true"]], s=60, color=cols[i % 10], label=a, zorder=3)
    ax.set_xlabel("L3 recall (frozen keys at support)")
    ax.set_ylabel("buildable TRUE L4")
    ax.set_title("the r$^2$ wall: what each arm's L3 bought at L4")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(d, "r2_payoff.png"), dpi=140)
    plt.close(fig)

    # 4. dose + junk share
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = [a for a in out.get("dose", {})]
    ax[0].bar(range(len(names)), [out["dose"][a]["all"] for a in names],
              color=[cols[i % 10] for i in range(len(names))])
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=35, ha="right",
                                                              fontsize=7)
    ax[0].set_ylabel("P(delivered key == designed key)")
    ax[0].set_title("delivered dose vs design dose"); ax[0].grid(alpha=.3, axis="y")
    nn = [a for a in out.get("nerdsnipe", {})]
    j3 = [out["nerdsnipe"][a]["L3_junk"] for a in nn]
    t3 = [out["nerdsnipe"][a]["L3_true"] for a in nn]
    ax[1].bar(range(len(nn)), t3, label="true", color="#2a6f97")
    ax[1].bar(range(len(nn)), j3, bottom=t3, label="junk", color="#c1121f")
    ax[1].set_xticks(range(len(nn))); ax[1].set_xticklabels(nn, rotation=35, ha="right",
                                                            fontsize=7)
    ax[1].set_ylabel("L3 keys at support"); ax[1].legend(fontsize=8)
    ax[1].set_title("what got banked: true vs parse-ambiguous junk")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(d, "dose_and_junk.png"), dpi=140)
    plt.close(fig)
    print(f"\n  figures -> {d}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="an_s0")
    ap.add_argument("--merge-tag", default="",
                    help="one or more comma-separated tags whose arms are folded into this "
                         "reduction (crescendo's --merge-tag pattern; used for an_s1's "
                         "pacer-only arm and an_s2's composed non-oracle arms)")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.tag)
    root = os.path.join(FIG, args.tag)
    setup = _load(root, "setup.json")
    summary = _load(root, "summary.json")
    assert setup and summary, f"no setup/summary under {root}"
    order = summary["order"]
    A = {}
    for a in order:
        r = _load(os.path.join(root, a), "results.json")
        if r:
            A[a] = r
    order = [a for a in order if a in A]
    # [an_s1] fold in a second tag's arms. Their refs must be the same object or §6's
    # recovered fractions would be computed against a different normalisation — asserted,
    # not assumed (`crescendo`'s cross-tag discipline).
    # [antiphon-s2] `--merge-tag` now takes a COMMA-SEPARATED LIST (one tag still works,
    # so `an_s0 --merge-tag an_s1` reproduces exactly). Each tag's refs and substrate are
    # asserted identical before any of its arms enters the reduction.
    for mtag in [t.strip() for t in args.merge_tag.split(",") if t.strip()]:
        if args.fetch:
            fetch(mtag)
        mroot = os.path.join(FIG, mtag)
        msetup, msummary = _load(mroot, "setup.json"), _load(mroot, "summary.json")
        assert msetup and msummary, f"no setup/summary under {mroot}"
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
            r = _load(os.path.join(mroot, a), "results.json")
            if r is None:
                continue
            lab = a if a not in A else f"{a}@{mtag}"
            A[lab] = r
            order.append(lab)
            summary["cycle_seconds"][lab] = msummary["cycle_seconds"].get(a)
        print(f"[merge] folded {mtag}: {msummary['order']} "
              f"(refs and substrate identical — asserted)")
    print("=" * 88)
    print(f"ANTIPHON — the question port.  tag={args.tag}  arms={order}")
    print(f"  menu K={setup.get('question_k')}  modes={setup.get('question_modes')}")
    print(f"  cycle seconds: { {k: round(v, 1) for k, v in summary['cycle_seconds'].items()} }")
    print("=" * 88)

    out = {"tag": args.tag, "order": order}
    sec0_fidelity(A, order, out, setup)
    sec1_controls(A, order, setup, out)
    sec2_dose(A, order, out)
    traj = sec3_climb(A, order, setup, out)
    sec4_tables(A, order, setup, out)
    sec5_clocks(A, order, out)
    sec6_value(A, order, setup, out)
    sec7_nerdsnipe(A, order, setup, out)
    try:
        sec8_trust(A, order, out)
    except Exception as exc:
        print(f"  [skip] trust: {exc}")
    try:
        sec9_use(A, order, out)
    except Exception as exc:
        print(f"  [skip] ess_use: {exc}")
    if args.figures:
        figures(args.tag, out, traj, setup)
    with open(os.path.join(root, "reduction.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(root, 'reduction.json')}")


if __name__ == "__main__":
    main()
