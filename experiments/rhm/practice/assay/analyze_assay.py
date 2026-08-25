"""Reduction for `census` PHASE 1 — numbers, not interpretation.

Sections, in the order the round's questions are asked:

  (0) GATES — the full-scale half of G-F (this tag's in-tag `spiral_route` against `sp_s0`'s,
      over the 32 cycles the two ladders share), the smoke-scale half (from `fidelity_smoke`'s
      `gate.json`), the twin gates, and the G-Y instrument's soundness record.
  (1) DID THE OP BUY COVERAGE — end-of-run committed table entries / recall / precision per arm
      per level, and the frozen-vs-live gap that Phase 0 said was on offer.
  (2) THE MONEY READOUT — eras 4-5 recovered fraction against the
      `spiral_route` -> `given_route` bracket (routing-only, per the finding-9 quarantine),
      with `given_native` carried for cross-tag continuity with `sp_s0`.
  (3) THE TIMING PRICE — `census_gate` vs `yoked_delay` vs `census_extend`: commit cycles,
      width trajectory, cycles-to-native, and the gauge's own history at the gate level.
  (4) THE GRADER BILL — auditions and recerts the extension op paid for, on the ledger.
  (5) G-Y — the next-level (L4) observation stream per arm: the growth direction under a
      fuller vs a sliver L3.
  (6) Plant guard, per-arm cost, shadow certificates.

Usage (from experiments/):
    python3 rhm/practice/census/analyze_census.py --tag cs_s0 --fetch
    python3 rhm/practice/census/analyze_census.py --tag cs_s0 --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
CENSUS_FIG = os.path.join(os.path.dirname(HERE), "census", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_assay"

# the 32 cycles `cs_s0` and `sp_s0` share: era 1 is 48 cycles here and 32 there, and `ec`
# reaches only the loop bound, `at_boundary` (both certified commits precede it) and the
# era-end probe trigger (which fires at c32 either way, and probes consume no RNG).
SHARED_PREFIX = 116
GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]


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


def era_bounds(setup):
    out, c = [], 0
    for i, e in enumerate(setup["eras"]):
        n = int(e["cycles"])
        out.append({"era": i + 1, "first": c + 1, "last": c + n, "name": e["name"]})
        c += n
    return out


def _era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def gates(tag, A, setup, summary, gf_smoke):
    print("\n" + "=" * 78)
    print("(0) GATES")
    print("=" * 78)

    # --- G-F, full scale, cross-tag ---------------------------------------------------- #
    # `assay`'s `anchor` is `census`'s `spiral_route` spec at the SAME ladder and the same
    # stream, so unlike `cs_s0` (which shared only a 32-cycle prefix with `sp_s0`) the whole
    # 116-cycle run is comparable — a full-length bit-for-bit gate, for free, on an arm the
    # round runs anyway.
    sp = os.path.join(CENSUS_FIG, "cs_s0", "spiral_route", "results.json")
    if os.path.isfile(sp) and "anchor" in A:
        old = json.load(open(sp))["log"]
        new = A["anchor"]["log"]
        worst, per = 0.0, {}
        for k in GF_SERIES:
            x = np.asarray(old[k][:SHARED_PREFIX], float)
            y = np.asarray(new[k][:SHARED_PREFIX], float)
            n = min(len(x), len(y))
            d = float(np.abs(x[:n] - y[:n]).max()) if n else float("nan")
            per[k] = d
            worst = max(worst, d)
        print(f"  G-F full scale (cross-tag): as_s0/anchor vs cs_s0/spiral_route over "
              f"c1-c{SHARED_PREFIX}\n    max|delta| = {worst:.3e} over {len(GF_SERIES)} series"
              f"  -> {'PASS' if worst == 0.0 else 'FAIL'}")
        if worst:
            print("    per-series: " + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))
    else:
        print("  G-F full scale: cs_s0/spiral_route not fetched locally — skipped")

    if gf_smoke:
        print(f"  G-F smoke scale (in-process vs census.py): "
              f"max|fork - census| = {gf_smoke['worst_fork']:.3e}, self-replay control = "
              f"{gf_smoke['worst_donor_self_replay']:.3e}, commits equal = "
              f"{gf_smoke['events_equal']}  -> "
              f"{'PASS' if gf_smoke['worst_fork'] <= gf_smoke['worst_donor_self_replay'] else 'FAIL'}")

    # --- twins ------------------------------------------------------------------------- #
    # The pre-treatment window ends at the first cycle at which the treated arm's behaviour
    # COULD differ. For a gate arm that is the cycle the ANCHOR commits and the gate arm does
    # not (holding a commit open is the treatment), not the cycle the gate itself commits.
    print("\n  TWIN GATES (max|de| to each arm's own first surgery):")
    anchor = "anchor"
    for t in [x for x in summary["order"] if x not in (anchor, "given_c1")]:
        if t not in A:
            continue
        n = next((e["cycle"] for e in A[t]["events"] if e["kind"] == "surgery"), None)
        if not n:
            print(f"    {t:12s}: no surgery event"); continue
        w = n - 1
        d = 0.0
        for k in GF_SERIES:
            x = np.asarray(A[t]["log"][k][:w], float)
            y = np.asarray(A[anchor]["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        print(f"    {t:12s} vs anchor c1-c{w:<4d} (first surgery c{n}) max|de| = {d:.3e}"
              f"  -> {'PASS' if d == 0.0 else 'FAIL'}")
    print("    given_c1 vs anchor: NO pre-treatment window by construction (true tables and a")
    print("      different stream from c1) — it is the bracket's top, not a twin.")

    erc = (setup or {}).get("entry_recorder_check")
    if erc:
        print(f"\n  ENTRY-RECORDER CHECK: {erc['verdict']} — the recording copy of "
              f"`macros.macro_features` reproduces\n    the donor bit for bit over "
              f"{erc['n_macros']} macro applications (max|delta| feats "
              f"{erc['max_abs_feats']}, pos {erc['max_abs_pos']}); "
              f"{erc['n_exec_recorded']} selections recorded at levels "
              f"{erc.get('recorded_levels')}")
    jp = (setup or {}).get("junk_pool_provenance")
    if jp:
        print(f"  JUNK POOL (realistic, mined-but-false): {jp['split']}")
        for pr in jp["provenance"]:
            print(f"    {pr['tag']}: found={pr.get('found')} arms={pr.get('arms')} "
                  f"new_keys={pr.get('new_keys')}")
    gy = (setup or {}).get("gy_soundness")
    if gy:
        print(f"\n  G-Y INSTRUMENT SOUNDNESS: {gy['verdict']} — level {gy['level']}, span "
              f"{gy['span_blocks']} blocks, decoupled from max_macro_level: "
              f"{gy['decoupled_from_max_macro_level']}")
        print(f"    hand counts {gy['hand']} == measured {gy['measured']}")
        print(f"    L{gy['level']} node per era: "
              + ", ".join(f"{n['era']}->{n[f'L' + str(gy['level']) + '_node']}"
                          for n in gy["nodes_per_era"]))


def coverage(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(1) DID THE OP BUY COVERAGE? end-of-run COMMITTED tables, and the live gap")
    print("=" * 78)
    print(f"  {'arm':16s}{'lvl':>4s}{'frozen n':>10s}{'recall':>9s}{'prec':>8s}"
          f"{'live n':>8s}{'live rec':>10s}{'live prec':>11s}{'gap':>6s}{'added':>7s}")
    for a in summary["order"]:
        lg = A[a]["log"]
        for lv in ("2", "3"):
            cg = next((c.get(lv) for c in reversed(lg["committed_grade"]) if (c or {}).get(lv)),
                      None)
            fz = next((c.get(lv) for c in reversed(lg["vocab"]) if (c or {}).get(lv)), None)
            au = lg["aud"][-1].get(lv) or {}
            added = sum(e.get("n_admitted", 0) for e in A[a]["events"]
                        if e["kind"] == "extend" and e["level"] == int(lv))
            print(f"  {a:16s}{lv:>4s}{_fmt(fz, 10)}{_fmt((cg or {}).get('recall'), 9)}"
                  f"{_fmt((cg or {}).get('precision'), 8)}{_fmt(au.get('n_entries'), 8)}"
                  f"{_fmt(au.get('tab_recall'), 10)}{_fmt(au.get('tab_precision'), 11)}"
                  f"{_fmt((au.get('n_entries') or 0) - (fz or 0), 6)}{added:7d}")


def money(tag, A, setup, summary):
    refs, n_era = setup["refs"], len(setup["eras"])
    print("\n" + "=" * 78)
    print("(2) THE MONEY READOUT: recovered fraction per era, and the eras-4-5 bracket")
    print("=" * 78)
    st, fl = refs["stale"], refs["floor"]
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>17s}" for j in range(n_era)))
    rows = {}
    for a in summary["order"]:
        lg = A[a]["log"]; row = ""
        rec = []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            r = (st[j] - e) / (st[j] - fl[j])
            rec.append(r)
            row += f"{e:.3f} /{r:+.3f}".rjust(17)
        rows[a] = rec
        print(f"  {a:16s}{row}")
    print(f"  {'stale':16s}" + "".join(f"{st[j]:>17.3f}" for j in range(n_era)))
    print(f"  {'floor':16s}" + "".join(f"{fl[j]:>17.3f}" for j in range(n_era)))
    lo, hi = "anchor", "given_c1"
    if lo in rows and hi in rows:
        print(f"\n  ERAS 4-5 BRACKET (routing-only): {lo} -> {hi}")
        for j in (3, 4):
            if j >= n_era:
                continue
            b0, b1 = rows[lo][j], rows[hi][j]
            span = b1 - b0
            print(f"    era {j + 1}: bracket {b0:+.3f} .. {b1:+.3f} (span {span:+.3f})")
            for a in summary["order"]:
                if a in (lo, hi):
                    continue
                frac = (rows[a][j] - b0) / span if span else float("nan")
                print(f"      {a:16s} {rows[a][j]:+.3f}   "
                      f"fraction of bracket closed: {frac:+.3f}")


def surgery_audit(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(3) THE SURGERY AUDIT — what each arm actually committed, at each level")
    print("=" * 78)
    print("  `mined` is the table the loop earned (the counterfactual the surgery replaced);")
    print("  `after` is what was installed. K is `complete`'s addition count, the matched dose.")
    print(f"  {'arm':12s}{'lvl':>4s}{'cyc':>5s}{'mode':>10s}{'own':>5s}{'own_T':>6s}"
          f"{'added':>6s}{'add_T':>6s}{'rem':>5s}{'K':>4s}{'after':>6s}"
          f"{'mined rec/prec':>16s}{'after rec/prec':>16s}{'unbuild':>8s}")
    rows = []
    for a in summary["order"]:
        for e in A[a]["events"]:
            if e["kind"] != "surgery":
                continue
            rows.append({k: e.get(k) for k in
                         ("arm", "level", "cycle", "mode", "n_own", "n_own_true", "n_added",
                          "n_added_true", "n_removed", "K", "n_after", "tab_recall",
                          "tab_precision", "mined_recall", "mined_precision",
                          "n_unbuildable", "dose_shortfall", "cancelled_empty")})
            mr = (f"{e.get('mined_recall'):.3f}/{e.get('mined_precision'):.3f}"
                  if e.get("mined_recall") is not None else "-")
            ar = (f"{e.get('tab_recall'):.3f}/{e.get('tab_precision'):.3f}"
                  if e.get("tab_recall") is not None else "-")
            print(f"  {a:12s}{e['level']:4d}{e['cycle']:5d}{str(e['mode']):>10s}"
                  f"{e['n_own']:5d}{e['n_own_true']:6d}{e['n_added']:6d}{e['n_added_true']:6d}"
                  f"{e['n_removed']:5d}{str(e.get('K')):>4s}{e.get('n_after', 0):6d}"
                  f"{mr:>16s}{ar:>16s}{e['n_unbuildable']:8d}"
                  + ("   CANCELLED (empty)" if e.get("cancelled_empty") else ""))
    summary.setdefault("_audit", rows)
    print("\n  END-OF-RUN committed tables (the treatment as it stood at c116):")
    print(f"  {'arm':12s}{'L2 n':>6s}{'L2 rec':>8s}{'L2 prec':>9s}{'L3 n':>6s}"
          f"{'L3 rec':>8s}{'L3 prec':>9s}")
    for a in summary["order"]:
        lg = A[a]["log"]
        cg = lg["committed_grade"][-1] or {}
        v = lg["vocab"][-1] or {}
        print(f"  {a:12s}{_fmt(v.get('2'), 6)}{_fmt((cg.get('2') or {}).get('recall'), 8)}"
              f"{_fmt((cg.get('2') or {}).get('precision'), 9)}{_fmt(v.get('3'), 6)}"
              f"{_fmt((cg.get('3') or {}).get('recall'), 8)}"
              f"{_fmt((cg.get('3') or {}).get('precision'), 9)}")


def entry_identity(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(4) PER-EXECUTION ENTRY IDENTITY — trust-damage vs execution-damage")
    print("=" * 78)
    print("  For every macro execution the max-sum DP picks one table entry. Below: the share")
    print("  of BEAM (priced) executions that landed on a TRUE entry, per level per era. If a")
    print("  polluted table's executions still land on true entries, junk hurts by being")
    print("  CARRIED (pi de-funds the level) and not by being EXECUTED.")
    eb = era_bounds(setup)
    for lv in ("2", "3"):
        print(f"\n  level {lv}: true-entry share of beam executions (and executions/cycle)")
        print(f"  {'arm':12s}" + "".join(f"{'era' + str(e['era']):>18s}" for e in eb))
        for a in summary["order"]:
            lg = A[a]["log"]
            if not lg.get("entry"):
                continue
            row = ""
            for e in eb:
                idx = [i for i, q in enumerate(lg["era"]) if q == e["era"]]
                tot = tru = 0
                for i in idx:
                    q = lg["entry"][i] or {}
                    h = ((q.get("hist") or {}).get("beam") or {}).get(lv)
                    mk = (q.get("true_mask") or {}).get(lv)
                    if not h or not mk or len(h) != len(mk):
                        continue
                    tot += sum(h)
                    tru += sum(c for c, t in zip(h, mk) if t)
                row += (f"{tru / tot:.3f} ({tot / max(len(idx), 1):.0f})".rjust(18)
                        if tot else "-".rjust(18))
            print(f"  {a:12s}{row}")
    print("\n  BEAM vs PROBE phase totals (the instrument is unpriced; probe executions are")
    print("  auditions, probes and the battery, and are separated so they cannot contaminate)")
    print(f"  {'arm':12s}{'beam L2':>12s}{'beam L3':>12s}{'probe L2':>12s}{'probe L3':>12s}")
    for a in summary["order"]:
        lg = A[a]["log"]
        if not lg.get("entry"):
            continue
        tot = {}
        for q in lg["entry"]:
            for ph, d in ((q or {}).get("hist") or {}).items():
                for lv, h in d.items():
                    tot[(ph, lv)] = tot.get((ph, lv), 0) + sum(h)
        print(f"  {a:12s}" + "".join(_fmt(tot.get((ph, lv)), 12)
                                     for ph in ("beam", "probe") for lv in ("2", "3")))


def pi_tables(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(5) pi PER-LEVEL MASS, ARGMAX SHARE, AND MACRO EXPANSIONS, per arm per era")
    print("=" * 78)
    eb = era_bounds(setup)
    for metric, key in (("proposal mass p_macro_all", "p"),
                        ("argmax share frac_argmax", "argmax")):
        for lv in (2, 3):
            print(f"\n  L{lv} {metric} (last probe of each era)")
            print(f"  {'arm':12s}" + "".join(f"{'era' + str(e['era']):>10s}" for e in eb))
            for a in summary["order"]:
                row = ""
                for e in eb:
                    pr = [q for q in A[a]["log"]["probe"]
                          if q["era"] == e["era"] and q.get("pi", {}).get("macros")]
                    if not pr:
                        row += "-".rjust(10); continue
                    tot = sum(m.get("p_macro_all" if key == "p" else "frac_argmax", 0.0)
                              for m in pr[-1]["pi"]["macros"] if m["level"] == lv)
                    row += f"{tot:10.3f}"
                print(f"  {a:12s}{row}")
    print("\n  MACRO EXPANSIONS the beam ran (blocks.mat_dp, per-cycle mean by era)")
    print(f"  {'arm':12s}" + "".join(f"{'era' + str(e['era']):>12s}" for e in eb))
    for a in summary["order"]:
        lg = A[a]["log"]; row = ""
        for e in eb:
            idx = [i for i, q in enumerate(lg["era"]) if q == e["era"]]
            row += f"{float(np.mean([lg['blocks'][i].get('mat_dp', 0) for i in idx])):12.0f}"
        print(f"  {a:12s}{row}")


def stream_floor(tag, A, setup, summary):
    """[assay] THE STREAM-POSITION NOISE FLOOR, measured rather than assumed.

    Each displaced twin is its original in every config bit, with the per-arm RNG stream
    displaced by a controlled burn. |original - displaced| is therefore what stream position
    ALONE moves, and it is the floor the arrival residual has to clear."""
    pairs = [(o, o + "_j") for o in ("given_c1", "exact") if o + "_j" in A]
    if not pairs:
        return
    print("\n" + "=" * 78)
    print("(7) THE STREAM-POSITION FLOOR — displaced twins")
    print("=" * 78)
    eb = era_bounds(setup)
    st, fl = setup["refs"]["stale"], setup["refs"]["floor"]

    def _rec(a, j):
        lg = A[a]["log"]; idx = _era_idx(lg, j)
        e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
        return e, (st[j] - e) / (st[j] - fl[j])

    for o, t in pairs:
        b = A[t].get("stream_burn") or {}
        print(f"\n  {t} vs {o}   burn = {b.get('draws')} draws "
              f"(cpu={b.get('cpu')}, cuda={b.get('cuda')}) at: {b.get('point')}")
        print(f"    {'era':>4s}{'orig e':>10s}{'twin e':>10s}{'|de|':>9s}"
              f"{'orig rec':>10s}{'twin rec':>10s}{'|d rec|':>9s}")
        for j, _ in enumerate(eb):
            eo, ro = _rec(o, j)
            et, rt = _rec(t, j)
            print(f"    {j + 1:4d}{eo:10.4f}{et:10.4f}{abs(et - eo):9.4f}"
                  f"{ro:+10.3f}{rt:+10.3f}{abs(rt - ro):9.3f}")
        d35 = [abs(_rec(t, j)[1] - _rec(o, j)[1]) for j in (2, 3, 4)]
        print(f"    eras 3-5 |d recovered fraction|: {[round(x, 3) for x in d35]}  "
              f"max {max(d35):.3f}  mean {float(np.mean(d35)):.3f}")
        lo = np.asarray(A[o]["log"]["e"], float); lt = np.asarray(A[t]["log"]["e"], float)
        n = min(len(lo), len(lt))
        nz = np.nonzero(np.abs(lo[:n] - lt[:n]))[0]
        print(f"    first differing cycle: c{int(nz[0]) + 1 if nz.size else None}; "
              f"per-cycle |de| mean {float(np.abs(lo[:n] - lt[:n]).mean()):.4f} "
              f"max {float(np.abs(lo[:n] - lt[:n]).max()):.4f}")

    if "exact" in A and "given_c1" in A:
        print("\n  THE COMPARISON THIS EXISTS FOR — arrival residual against the floor:")
        print(f"    {'era':>4s}{'given_c1 - exact':>19s}{'floor(given_c1)':>18s}"
              f"{'floor(exact)':>15s}{'residual / max floor':>22s}")
        for j in (2, 3, 4):
            _, rg = _rec("given_c1", j)
            _, re_ = _rec("exact", j)
            resid = rg - re_
            f1 = abs(_rec("given_c1_j", j)[1] - rg) if "given_c1_j" in A else float("nan")
            f2 = abs(_rec("exact_j", j)[1] - re_) if "exact_j" in A else float("nan")
            mx = max(f1, f2)
            print(f"    {j + 1:4d}{resid:+19.3f}{f1:18.3f}{f2:15.3f}"
                  f"{(resid / mx if mx else float('nan')):22.2f}x")


def arrival_axis(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(6) THE ARRIVAL AXIS — `exact` vs `given_c1` at IDENTICAL content")
    print("=" * 78)
    if "exact" not in A or "given_c1" not in A:
        print("  one of the pair is missing"); return
    ce = A["exact"]["log"]["committed_grade"][-1]
    cg = A["given_c1"]["log"]["committed_grade"][-1]
    print(f"  end-of-run content identical: {ce == cg}")
    print(f"    exact    {ce}")
    print(f"    given_c1 {cg}")
    print("  CAVEAT, stated: `exact` carries the anchor's torch stream and `given_c1` carries")
    print("  `given`'s, exactly as `spiral_route` and `given_route` did in `cs_s0` — so the")
    print("  pair differs in arrival AND in stream position, and the stream's own size is")
    print("  bounded by cs_s0's twin data rather than zero.")
    eb = era_bounds(setup)
    st, fl = setup["refs"]["stale"], setup["refs"]["floor"]
    print(f"\n  {'arm':12s}" + "".join(f"{'era' + str(e['era']):>17s}" for e in eb))
    for a in ("anchor", "exact", "given_c1"):
        lg = A[a]["log"]; row = ""
        for j, e in enumerate(eb):
            idx = _era_idx(lg, j)
            v = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            row += f"{v:.3f} /{(st[j] - v) / (st[j] - fl[j]):+.3f}".rjust(17)
        print(f"  {a:12s}{row}")
    print("\n  `probe.all_eras` on the era-4 set from c1 (the arrival trace):")
    cyc = [q["cycle"] for q in A["anchor"]["log"]["probe"]]
    print(f"  {'cycle':>6s}" + "".join(f"{a[:12]:>14s}" for a in summary["order"]))
    for c in cyc:
        if c not in (1, 8, 16, 32, 48, 72, 88, 100, 109, 116):
            continue
        cells = ""
        for a in summary["order"]:
            pr = next((q for q in A[a]["log"]["probe"] if q["cycle"] == c), None)
            cells += _fmt(None if pr is None else pr["all_eras"].get("3"), 14, 4)
        print(f"  {c:6d}{cells}")


def gy_readout(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(5) G-Y: the next-level (L4) observation stream — growth direction")
    print("=" * 78)
    n_era = len(setup["eras"])
    sup = str(A[summary["order"][0]]["config"]["mine_support"])
    print(f"  distinct L4-shaped tuples at support {sup}, end of each era "
          f"(L4 is unearnable: 1,024 entries, ~384 cycles to cover — direction only)")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era))
          + f"{'n_obs':>10s}{'distinct':>10s}")
    for a in summary["order"]:
        lg = A[a]["log"]
        if not lg.get("gy") or lg["gy"][-1] is None:
            continue
        row = ""
        for j in range(n_era):
            i = _era_idx(lg, j)[-1]
            row += _fmt((lg["gy"][i].get("n_at_support") or {}).get(sup), 10)
        f = lg["gy"][-1]
        print(f"  {a:16s}{row}{f['n_obs']:10d}{f['n_distinct']:10d}")


def tail(tag, A, summary):
    print("\n" + "=" * 78)
    print("(6) PLANT GUARD, CERTIFICATES, COST")
    print("=" * 78)
    for a in summary["order"]:
        pr = A[a]["log"]["probe"]
        pa = [q["plant"]["parse_acc"] for q in pr]
        inf = [q["plant"]["infill_acc"] for q in pr]
        sc = summary["arms"][a]["shadow_cert"]
        print(f"  {a:16s} parse {min(pa):.3f}-{max(pa):.3f}  infill {min(inf):.3f}-{max(inf):.3f}"
              f"  cert L2 c{sc['2'].get('fired')} L3 c{sc.get('3', {}).get('fired')}"
              f"  recerts {summary['arms'][a]['n_recerts']:3d}"
              f"  {summary['cycle_seconds'][a]:6.1f} s/cycle")
    print(f"\n  TOTAL {summary.get('elapsed_s', 0):.0f}s = "
          f"{summary.get('elapsed_s', 0) / 3600:.2f} GPU-h")


def figures(tag, A, setup, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = summary["order"]
    refs = setup["refs"]
    bounds, c = [], 0
    for e in setup["eras"]:
        c += int(e.get("cycles")); bounds.append(c)
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    for a in order:
        lg = A[a]["log"]
        ax.plot(lg["cycle"], lg["e"], lw=1.4, label=a)
    lo = 0
    for j, hi in enumerate(bounds):
        ax.hlines(refs["stale"][j], lo + 1, hi, color="0.5", ls=":", lw=1)
        ax.hlines(refs["floor"][j], lo + 1, hi, color="0.5", ls="--", lw=1)
        ax.axvline(hi + 0.5, color="0.85", lw=1)
        ax.text((lo + hi) / 2, 1.02, setup["eras"][j]["name"], ha="center", fontsize=8)
        lo = hi
    for a in order:
        for ev in summary["arms"][a]["commits"]:
            ax.plot(ev["cycle"], A[a]["log"]["e"][ev["cycle"] - 1], "v", ms=6, color="k",
                    zorder=5)
        for ev in summary["arms"][a].get("extends", []):
            if ev.get("n_admitted"):
                ax.plot(ev["cycle"], A[a]["log"]["e"][ev["cycle"] - 1], "^", ms=4,
                        color="tab:green", zorder=4)
    ax.set_xlabel("cycle"); ax.set_ylabel("e (1 - terminal success)")
    ax.set_title(f"{tag}: competence; v = commits, ^ = extensions that admitted entries")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sup = str(A[order[0]]["config"]["mine_support"])
    for a in order:
        lg = A[a]["log"]
        cov = [(c.get("3") or {}).get("recall") or 0 for c in lg["committed_grade"]]
        axes[0].plot(lg["cycle"], cov, lw=1.4, label=a)
        cov2 = [(c.get("2") or {}).get("recall") or 0 for c in lg["committed_grade"]]
        axes[1].plot(lg["cycle"], cov2, lw=1.4, label=a)
        if lg.get("gy") and lg["gy"][-1]:
            axes[2].plot(lg["cycle"],
                         [(q or {}).get("n_at_support", {}).get(sup, 0) for q in lg["gy"]],
                         lw=1.4, label=a)
    for ax_, t in zip(axes, ("committed L3 recall", "committed L2 recall",
                             "G-Y: L4-shaped tuples at support")):
        for hi in bounds:
            ax_.axvline(hi + 0.5, color="0.85", lw=1)
        ax_.set_xlabel("cycle"); ax_.set_title(t); ax_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_coverage.png"), dpi=140)
    plt.close(fig)
    print(f"\n[figures] {out}/fig1_competence.png fig2_coverage.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="as_s0")
    ap.add_argument("--gf-tag", default="gf_smoke")
    ap.add_argument("--merge-tag", default="",
                    help="a second tag whose arms are merged into this reduction (the "
                         "stream-displaced twins). Substrate identity is asserted, not assumed.")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
        try:
            fetch(a.gf_tag)
        except subprocess.CalledProcessError:
            print(f"[warn] no {a.gf_tag} on the volume")
    if a.fetch and a.merge_tag:
        try:
            fetch(a.merge_tag)
        except subprocess.CalledProcessError:
            print(f"[warn] no {a.merge_tag} on the volume")
    root = os.path.join(FIG, a.tag)
    setup, summary = _load(root, "setup.json"), _load(root, "summary.json")
    A = {arm: json.load(open(os.path.join(root, arm, "results.json")))
         for arm in summary["order"]}
    # --- merge the stream-displaced twins ------------------------------------------------ #
    # A separate tag, not a re-run of `as_s0`: `summary.json` is rebuilt on every invocation,
    # so writing the twins into `as_s0` would drop its six originals out of the reduction.
    # The merge asserts the two tags trained the SAME substrate before joining them.
    if a.merge_tag:
        m_root = os.path.join(FIG, a.merge_tag)
        m_setup, m_summary = _load(m_root, "setup.json"), _load(m_root, "summary.json")
        if m_summary is None:
            print(f"[warn] {a.merge_tag} not present locally — twins not merged")
        else:
            bad = [k for k in setup["refs"] if k != "macro_true"
                   and not np.allclose(np.asarray(setup["refs"][k], float),
                                       np.asarray(m_setup["refs"][k], float))]
            same_buf = (setup["stale_random_blocks"] == m_setup["stale_random_blocks"]
                        and setup["stale_task_matched"] == m_setup["stale_task_matched"])
            print(f"\n[merge] {a.merge_tag} -> {a.tag}: refs identical = {not bad}"
                  f"{'' if not bad else ' (differ: ' + str(bad) + ')'}, "
                  f"stale buffer identical = {same_buf}")
            assert not bad and same_buf, (
                f"{a.merge_tag} did not train the same substrate as {a.tag} — the twins are "
                f"not comparable to their originals")
            for arm in m_summary["order"]:
                A[arm] = json.load(open(os.path.join(m_root, arm, "results.json")))
                summary["arms"][arm] = m_summary["arms"][arm]
                summary["cycle_seconds"][arm] = m_summary["cycle_seconds"][arm]
                summary["order"].append(arm)
            print(f"[merge] arms added: {m_summary['order']}")
    gf = _load(os.path.join(FIG, a.gf_tag), "gate.json")
    print(f"=== assay — tag {a.tag} ===")
    print(f"ladder {[(e['name'], e.get('cycles')) for e in setup['eras']]} = "
          f"{setup['total_cycles_per_arm']} cycles/arm; setup {setup['t_setup_s']:.0f}s; "
          f"stale buffer {setup['stale_random_blocks']:.4f} -> {setup['stale_task_matched']:.4f}")
    gates(a.tag, A, setup, summary, gf)
    coverage(a.tag, A, setup, summary)
    money(a.tag, A, setup, summary)
    surgery_audit(a.tag, A, setup, summary)
    entry_identity(a.tag, A, setup, summary)
    pi_tables(a.tag, A, setup, summary)
    arrival_axis(a.tag, A, setup, summary)
    stream_floor(a.tag, A, setup, summary)
    gy_readout(a.tag, A, setup, summary)
    tail(a.tag, A, summary)
    if a.figures:
        figures(a.tag, A, setup, summary)


if __name__ == "__main__":
    main()
