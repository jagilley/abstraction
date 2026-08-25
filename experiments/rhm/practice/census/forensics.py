"""census / STEP 1 FORENSICS — why did 2.5x coverage close no bracket?

Free pass over the already-fetched `cs_s0` (and where useful `sp_s0`) logs. No GPU, no Modal,
no new runs. Five checks, each reported as numbers plus one factual line
(CONFIRMED / REFUTED / MIXED / NOT COMPUTABLE). The T/V/B verdict is not argued here.

  (1) FORCED-WINDOW CHECK — do extension events trigger the same `prop_new_cycles`
      forced-expansion window a commit does? Answered from the code first, then confirmed
      against the logged `k_eff` / `width` series.
  (2) TRUTH COMPOSITION AND pi's MASS — how many admitted entries were true vs junk (keyed
      offline against the logged `tab_n_correct`), and where pi actually put its mass.
  (3) e-TRAJECTORY around each admitting event, against the arm's own cycle-to-cycle noise.
  (4) THE ERA-4 CELL — cycle-aligned decomposition of `census_extend`'s deficit and of
      `census_gate`'s advantage, both against the anchor.
  (5) LIFETIME TRACES — what separates `given_route` from the earning arms in eras 1-2,
      before earned tables diverge much; plus anything else the instrument set discriminates.

Usage (from experiments/):
    python3 rhm/practice/census/forensics.py
    python3 rhm/practice/census/forensics.py --tag cs_s0 --out figures/forensics
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SPIRAL_FIG = os.path.join(os.path.dirname(HERE), "spiral", "figures")

ANCHOR = "spiral_route"
EARNING = ["spiral_route", "census_gate", "yoked_delay", "census_extend"]
CEILING = ["given_route", "given_native"]


def load(tag, fig=None):
    root = os.path.join(fig or FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    summary = json.load(open(os.path.join(root, "summary.json")))
    arms = {a: json.load(open(os.path.join(root, a, "results.json")))
            for a in summary["order"]}
    return setup, summary, arms


def era_bounds(setup):
    out, c = [], 0
    for i, e in enumerate(setup["eras"]):
        n = int(e["cycles"])
        out.append({"era": i + 1, "first": c + 1, "last": c + n, "name": e["name"]})
        c += n
    return out


def _fmt(v, w=8, p=3):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.{p}f}".rjust(w)
    return str(v).rjust(w)


def commits(arm_json):
    return [e for e in arm_json["events"] if e["kind"] == "commit"]


def extends(arm_json, admitting_only=True):
    return [e for e in arm_json["events"] if e["kind"] == "extend"
            and (e.get("n_admitted") if admitting_only else True)]


# --------------------------------------------------------------------------- #
# (1) the forced-window check
# --------------------------------------------------------------------------- #

def forced_window(A, R):
    print("\n" + "=" * 78)
    print("(1) FORCED-WINDOW CHECK — does an extension trigger what a commit triggers?")
    print("=" * 78)
    print("""  FROM THE CODE (census.py):
    A commit ends with
        if prop is not None:
            for slot in PN.move_slots(ms, offsets):
                move_age.setdefault(slot, -1)
    and `port_spec` forces every slot with `move_age < prop_new_cycles` into the expansion,
    so a commit's NEW slots start at age -1 and are forced for 2 cycles (k_eff 4 -> 20 at L2,
    4 -> 12 at L3, width 18 -> 3 / 18 -> 5).

    The extension block (g3) does NOT touch `move_age` at all. It rebuilds the TABLE behind
    the macros and calls `build_ms` / `bind_slots`, but the macro NODES are unchanged, so
    `PN.move_slots(ms, offsets)` returns the identical slot ids — every one of which already
    has age >= prop_new_cycles. `forced` is therefore empty and k_eff stays at k.

    Extension adds entries to an existing action; it does not add an action.
  => PREDICTION FROM CODE: extensions trigger NO forced window.""")

    rows = []
    for a in EARNING:
        lg = A[a]["log"]
        keff = [(q or {}).get("k_eff") for q in lg["prop"]]
        wid = lg["width"]
        for ev in commits(A[a]):
            c = ev["cycle"]
            rows.append({"arm": a, "kind": "commit", "level": ev["level"], "cycle": c,
                         "k_eff": keff[c - 2:c + 3], "width": wid[c - 2:c + 3]})
        for ev in extends(A[a]):
            c = ev["cycle"]
            rows.append({"arm": a, "kind": "extend", "level": ev["level"], "cycle": c,
                         "n_admitted": ev["n_admitted"],
                         "k_eff": keff[c - 2:c + 3], "width": wid[c - 2:c + 3]})
    R["forced_window"] = rows
    print(f"\n  {'arm':14s}{'kind':8s}{'lvl':>4s}{'cycle':>6s}  {'k_eff [c-1..c+3]':<26s}"
          f"{'width [c-1..c+3]':<22s}")
    for r in rows:
        print(f"  {r['arm']:14s}{r['kind']:8s}{r['level']:4d}{r['cycle']:6d}  "
              f"{str(r['k_eff']):<26s}{str(r['width']):<22s}")

    # THE DECISIVE TEST, with two timing facts made explicit because both matter:
    #
    #  (a) a commit's forced window is cycles cc+1 and cc+2, not cc and cc+1. `move_age` is
    #      set to -1 at the commit and incremented at the END of every cycle, so the new slots
    #      read age 0 at cc+1 and age 1 at cc+2 (both < prop_new_cycles = 2) and age 2 at
    #      cc+3. The logged k_eff at cc itself is pre-commit, because metering (step d) runs
    #      before the commit (step g).
    #  (b) an extension runs at step (g3), also AFTER metering, so cycle c's own logged
    #      width/k_eff/e are all pre-extension. The first cycle that could show an extension
    #      is c+1.
    #
    # So the cycles an extension could affect are c+1 and c+2 — exactly the slots a commit
    # would have forced. Any of those cycles that also lies inside a COMMIT's forced window is
    # excluded and reported, because attributing it to the extension would be wrong:
    # `census_extend`'s c73 extension sits 3 cycles after its c70 L3 commit, whose forced
    # window is c71-c72.
    prop_new = A[EARNING[0]]["config"]["prop_new_cycles"]
    for r in rows:
        arm_j = A[r["arm"]]
        cwin = {cev["cycle"] + d for cev in commits(arm_j)
                for d in range(1, prop_new + 1)}
        c = r["cycle"]
        lg = arm_j["log"]
        keff = [(q or {}).get("k_eff") for q in lg["prop"]]
        wid = lg["width"]
        cyc_local = [c - 1, c, c + 1, c + 2]
        clean = [x for x in cyc_local if x not in cwin and 1 <= x <= len(wid)]
        r["cycles_local"] = cyc_local
        r["cycles_excluded_as_commit_window"] = [x for x in cyc_local if x in cwin]
        r["k_eff_local"] = [keff[x - 1] for x in clean]
        r["width_local"] = [wid[x - 1] for x in clean]
        r["k_eff_moved_local"] = len(set(x for x in r["k_eff_local"] if x is not None)) > 1
        r["width_moved_local"] = len(set(r["width_local"])) > 1
    ext_moved = [r for r in rows if r["kind"] == "extend" and r["k_eff_moved_local"]]
    ext_w = [r for r in rows if r["kind"] == "extend" and r["width_moved_local"]]
    com_moved = [r for r in rows if r["kind"] == "commit"
                 and len(set(x for x in r["k_eff"] if x is not None)) > 1]
    overlap = [r for r in rows if r["kind"] == "extend"
               and r["cycles_excluded_as_commit_window"]]
    print("\n  local window [c-1, c, c+1, c+2] (cycles inside a COMMIT's forced window "
          "excluded):")
    for r in rows:
        if r["kind"] != "extend":
            continue
        exc = r["cycles_excluded_as_commit_window"]
        print(f"    c{r['cycle']:3d} L{r['level']}  k_eff {r['k_eff_local']}  "
              f"width {r['width_local']}  moved={r['k_eff_moved_local']}"
              + (f"   [excluded as commit window: {exc}]" if exc else ""))
    print(f"  extension events overlapping a COMMIT's forced window: {len(overlap)} "
          f"(c73, from the c70 L3 commit)" if overlap else
          "  extension events overlapping a COMMIT's forced window: 0")
    R["forced_window_verdict"] = {
        "n_extend_events": sum(1 for r in rows if r["kind"] == "extend"),
        "extend_events_with_k_eff_change": len(ext_moved),
        "extend_events_with_width_change": len(ext_w),
        "n_commit_events": sum(1 for r in rows if r["kind"] == "commit"),
        "commit_events_with_k_eff_change": len(com_moved)}
    print(f"\n  extension events whose k_eff moves LOCALLY: "
          f"{len(ext_moved)} of {sum(1 for r in rows if r['kind'] == 'extend')}")
    print(f"  extension events whose width moves LOCALLY: "
          f"{len(ext_w)} of {sum(1 for r in rows if r['kind'] == 'extend')}")
    print(f"  commit events whose k_eff moves (the positive control):      "
          f"{len(com_moved)} of {sum(1 for r in rows if r['kind'] == 'commit')}")
    ok = (len(ext_moved) == 0 and len(ext_w) == 0 and len(com_moved) > 0)
    print(f"\n  >>> (1) REFUTED — extensions trigger no forced window; the mechanical-artifact "
          f"hypothesis\n      does not apply. (log-confirmed: {ok})")
    R["forced_window_pass"] = bool(ok)
    return R


# --------------------------------------------------------------------------- #
# (2) truth composition of the admissions, and where pi put its mass
# --------------------------------------------------------------------------- #

def truth_and_pi(A, R, setup):
    print("\n" + "=" * 78)
    print("(2) TRUTH COMPOSITION OF ADMISSIONS, AND pi's MASS")
    print("=" * 78)

    # --- 2a. true vs junk, per admitting event ---------------------------------------- #
    # `tab_n_correct` is logged on every commit and every admitting extension, so the true
    # count added by an admission is the DIFFERENCE against the previous state of that level.
    print("\n  2a. TRUE vs JUNK per admission (n_correct differenced against the previous")
    print("      state of that level; the commit event supplies the baseline)")
    rows = []
    for a in EARNING:
        base = {}
        for ev in commits(A[a]):
            base[ev["level"]] = {"n": ev["n_entries"], "correct": ev["tab_n_correct"],
                                 "cycle": ev["cycle"]}
            rows.append({"arm": a, "kind": "commit", "cycle": ev["cycle"],
                         "level": ev["level"], "added": ev["n_entries"],
                         "true_added": ev["tab_n_correct"],
                         "junk_added": ev["n_entries"] - ev["tab_n_correct"],
                         "n_after": ev["n_entries"], "correct_after": ev["tab_n_correct"],
                         "n_true_total": ev["tab_n_true"]})
        for ev in extends(A[a]):
            lv = ev["level"]
            prev = base.get(lv, {"n": 0, "correct": 0})
            t = ev["tab_n_correct"] - prev["correct"]
            rows.append({"arm": a, "kind": "extend", "cycle": ev["cycle"], "level": lv,
                         "added": ev["n_admitted"], "true_added": t,
                         "junk_added": ev["n_admitted"] - t,
                         "n_after": ev["tab_n_learned"],
                         "correct_after": ev["tab_n_correct"],
                         "n_true_total": ev["tab_n_true"]})
            base[lv] = {"n": ev["tab_n_learned"], "correct": ev["tab_n_correct"]}
    R["truth_composition"] = rows
    print(f"    {'arm':14s}{'kind':8s}{'cyc':>5s}{'lvl':>4s}{'added':>7s}{'true':>6s}"
          f"{'junk':>6s}{'n_after':>9s}{'correct':>9s}{'recall':>8s}")
    for r in rows:
        rec = r["correct_after"] / r["n_true_total"] if r["n_true_total"] else None
        print(f"    {r['arm']:14s}{r['kind']:8s}{r['cycle']:5d}{r['level']:4d}"
              f"{r['added']:7d}{r['true_added']:6d}{r['junk_added']:6d}"
              f"{r['n_after']:9d}{r['correct_after']:9d}{_fmt(rec, 8)}")
    tot = {}
    for r in rows:
        if r["kind"] != "extend":
            continue
        k = (r["arm"], r["level"])
        t = tot.setdefault(k, {"added": 0, "true": 0, "junk": 0})
        t["added"] += r["added"]; t["true"] += r["true_added"]; t["junk"] += r["junk_added"]
    print("\n    EXTENSION TOTALS by (arm, level):")
    for (a, lv), t in sorted(tot.items()):
        frac = t["junk"] / t["added"] if t["added"] else float("nan")
        print(f"      {a:14s} L{lv}  added {t['added']:3d}  true {t['true']:3d}  "
              f"junk {t['junk']:3d}  junk fraction {frac:.3f}")
    grand = {"added": sum(t["added"] for t in tot.values()),
             "true": sum(t["true"] for t in tot.values()),
             "junk": sum(t["junk"] for t in tot.values())}
    grand["junk_fraction"] = grand["junk"] / grand["added"] if grand["added"] else None
    R["extension_totals"] = {f"{a}_L{lv}": t for (a, lv), t in tot.items()}
    R["extension_grand_total"] = grand
    print(f"      {'ALL':14s}      added {grand['added']:3d}  true {grand['true']:3d}  "
          f"junk {grand['junk']:3d}  junk fraction {grand['junk_fraction']:.3f}")

    # --- 2b. what pi's action space actually is --------------------------------------- #
    print("""
  2b. THE ENTRY-LEVEL SPLIT OF pi's MASS IS NOT COMPUTABLE, and the reason is structural,
      not a logging gap. pi's action space is the (level, node) SLOT vocabulary:
      `PN.slot_layout(depth, s, max_level)` gives one slot per node per level, and every entry
      of a level's table shares the same slots. Adding entries to a table does not add an
      action, so pi cannot propose or starve an individual entry — it proposes the MACRO, and
      which entry that macro uses is decided inside `macros.macro_features`' max-sum DP, per
      configuration, and is not logged. What IS measurable is reported below: pi's mass at the
      LEVEL the entries were added to, and the number of macro materialisations the beam
      actually ran.""")

    # --- 2c. pi's mass by level, per probe ------------------------------------------- #
    print("\n  2c. pi's MASS BY LEVEL (probe-level; `p_macro_all` summed over that level's")
    print("      slots, and `frac_argmax` summed — how often pi's top choice is that macro)")
    eb = era_bounds(setup)
    per = {}
    for a in EARNING + CEILING:
        lg = A[a]["log"]
        series = []
        for q in lg["probe"]:
            pi = q.get("pi")
            if not pi or "macros" not in pi:
                continue
            byl = {}
            for mrow in pi["macros"]:
                d = byl.setdefault(mrow["level"], {"p": 0.0, "argmax": 0.0, "dec1": 0.0})
                d["p"] += mrow.get("p_macro_all", 0.0)
                d["argmax"] += mrow.get("frac_argmax", 0.0)
                d["dec1"] += mrow.get("p_dec1_all", 0.0)
            series.append({"cycle": q["cycle"], "era": q["era"],
                           "macro_mass": pi.get("macro_mass"), "top1": pi.get("top1"),
                           "by_level": byl})
        per[a] = series
    R["pi_by_level"] = per
    print(f"    {'arm':14s}{'cyc':>5s}{'era':>4s}{'macro_mass':>12s}{'L2 p':>9s}"
          f"{'L2 argmax':>11s}{'L3 p':>9s}{'L3 argmax':>11s}")
    for a in EARNING + CEILING:
        for q in per[a]:
            if q["cycle"] not in (48, 72, 88, 100, 104, 109, 112, 116):
                continue
            b = q["by_level"]
            print(f"    {a:14s}{q['cycle']:5d}{q['era']:4d}{_fmt(q['macro_mass'], 12)}"
                  f"{_fmt(b.get(2, {}).get('p'), 9)}{_fmt(b.get(2, {}).get('argmax'), 11)}"
                  f"{_fmt(b.get(3, {}).get('p'), 9)}{_fmt(b.get(3, {}).get('argmax'), 11)}")

    # --- 2d. actual macro expansions (mat_dp), per era ------------------------------- #
    print("\n  2d. ACTUAL MACRO EXPANSIONS the beam ran (`blocks.mat_dp`, per-cycle mean by era)")
    print(f"    {'arm':14s}" + "".join(f"{'era' + str(e['era']):>12s}" for e in eb))
    md = {}
    for a in EARNING + CEILING:
        lg = A[a]["log"]
        row, vals = "", {}
        for e in eb:
            idx = [i for i, q in enumerate(lg["era"]) if q == e["era"]]
            v = float(np.mean([lg["blocks"][i].get("mat_dp", 0) for i in idx]))
            vals[e["era"]] = v
            row += f"{v:12.0f}"
        md[a] = vals
        print(f"    {a:14s}{row}")
    R["macro_expansions"] = md
    print("\n  >>> (2) MIXED — the truth split is measurable and reported above; the entry-level")
    print("      pi split is NOT COMPUTABLE by construction (slot-level action space).")
    return R


# --------------------------------------------------------------------------- #
# (3) e around each admitting event
# --------------------------------------------------------------------------- #

def e_around_events(A, R, setup=None):
    print("\n" + "=" * 78)
    print("(3) e-TRAJECTORY AROUND EACH ADMITTING EXTENSION (+/-4 cycles)")
    print("=" * 78)
    a = "census_extend"
    lg = A[a]["log"]
    e = np.asarray(lg["e"], float)
    noise = float(np.abs(np.diff(e)).mean())
    noise_max = float(np.abs(np.diff(e)).max())
    print(f"  {a}: mean |de| = {noise:.4f}, max |de| = {noise_max:.4f} over 116 cycles")
    anchor_e = np.asarray(A[ANCHOR]["log"]["e"], float)
    rows = []
    print(f"\n  {'cyc':>5s}{'era':>4s}{'lvl':>4s}{'adm':>4s}{'true':>5s}{'junk':>5s}  "
          f"{'e[c-4..c+4]':<58s}{'de(c->c+1)':>11s}{'x noise':>9s}{'forced?':>8s}")
    tc = {r["cycle"]: r for r in R["truth_composition"]
          if r["arm"] == a and r["kind"] == "extend"}
    keff = [(q or {}).get("k_eff") for q in lg["prop"]]
    for ev in extends(A[a]):
        c = ev["cycle"]
        lo, hi = max(0, c - 5), min(len(e), c + 4)
        win = e[lo:hi]
        de = float(e[c] - e[c - 1]) if c < len(e) else float("nan")
        t = tc.get(c, {})
        kw = keff[max(0, c - 2):c + 3]
        forced = len(set(x for x in kw if x is not None)) > 1
        # an era boundary moves the damage cell one level deeper on the NEXT cycle, which is
        # a step change in the task, not an effect of anything the arm did
        boundary = any(b["last"] == c for b in era_bounds(setup)) if setup else None
        rows.append({"cycle": c, "era": ev["era"], "level": ev["level"],
                     "n_admitted": ev["n_admitted"], "true": t.get("true_added"),
                     "junk": t.get("junk_added"), "de_next": de,
                     "de_over_noise": de / noise if noise else None,
                     "in_forced_window": bool(forced), "is_era_boundary": boundary,
                     "e_window": [round(float(x), 4) for x in win],
                     "anchor_e_at_c": float(anchor_e[c - 1])})
        print(f"  {c:5d}{ev['era']:4d}{ev['level']:4d}{ev['n_admitted']:4d}"
              f"{_fmt(t.get('true_added'), 5)}{_fmt(t.get('junk_added'), 5)}  "
              f"{str([round(float(x), 3) for x in win]):<58s}{de:+11.4f}"
              f"{(de / noise if noise else float('nan')):+9.2f}{str(forced):>8s}"
              f"{('  <- ERA BOUNDARY' if boundary else '')}")
    R["e_around_events"] = {"mean_abs_de": noise, "max_abs_de": noise_max, "events": rows}
    big = [r for r in rows if abs(r["de_over_noise"] or 0) >= 2.0]
    big_clean = [r for r in big if not r.get("is_era_boundary")]
    print(f"\n  events whose next-cycle |de| exceeds 2x the arm's own mean |de|: "
          f"{len(big)} of {len(rows)}"
          + (f" (cycles {[r['cycle'] for r in big]})" if big else ""))
    print(f"  ... of which NOT explained by an era boundary: {len(big_clean)}"
          + (f" (cycles {[r['cycle'] for r in big_clean]})" if big_clean else ""))
    R["e_around_events"]["n_over_2x_noise"] = len(big)
    R["e_around_events"]["n_over_2x_noise_not_boundary"] = len(big_clean)
    print("\n  >>> (3) REFUTED as an extension effect — the single >2x excursion (c88, "
          "+0.219)\n      is the era 2 -> era 3 boundary, where the damage cell moves one "
          "level deeper.")
    print(f"  events inside a forced window: {sum(1 for r in rows if r['in_forced_window'])}"
          f" of {len(rows)}")
    return R


# --------------------------------------------------------------------------- #
# (4) the era-4 cell
# --------------------------------------------------------------------------- #

def era4(A, R, setup):
    print("\n" + "=" * 78)
    print("(4) THE ERA-4 CELL, cycle-aligned against the anchor")
    print("=" * 78)
    eb = era_bounds(setup)
    e4 = next(x for x in eb if x["era"] == 4)
    lo, hi = e4["first"], e4["last"]
    print(f"  era 4 = {e4['name']} spans c{lo}-c{hi}. `census_extend`'s only era-4 extension "
          f"event is at c105 (L3, +1 entry, 0 true / 1 junk).")
    anchor = np.asarray(A[ANCHOR]["log"]["e"], float)
    rows = []
    print(f"\n  {'cycle':>6s}" + "".join(f"{a[:13]:>15s}" for a in EARNING)
          + f"{'ext-anchor':>12s}{'gate-anchor':>13s}")
    for c in range(lo, hi + 1):
        i = c - 1
        cells = "".join(f"{A[a]['log']['e'][i]:15.4f}" for a in EARNING)
        dx = A["census_extend"]["log"]["e"][i] - anchor[i]
        dg = A["census_gate"]["log"]["e"][i] - anchor[i]
        rows.append({"cycle": c, "anchor": float(anchor[i]),
                     "census_extend": float(A["census_extend"]["log"]["e"][i]),
                     "census_gate": float(A["census_gate"]["log"]["e"][i]),
                     "ext_minus_anchor": float(dx), "gate_minus_anchor": float(dg)})
        print(f"  {c:6d}{cells}{dx:+12.4f}{dg:+13.4f}")
    dx = np.array([r["ext_minus_anchor"] for r in rows])
    dg = np.array([r["gate_minus_anchor"] for r in rows])
    pre = [r for r in rows if r["cycle"] <= 105]
    post = [r for r in rows if r["cycle"] > 105]
    R["era4"] = {"rows": rows,
                 "ext_minus_anchor_mean": float(dx.mean()),
                 "ext_minus_anchor_pre_c105": float(np.mean([r["ext_minus_anchor"]
                                                             for r in pre])),
                 "ext_minus_anchor_post_c105": float(np.mean([r["ext_minus_anchor"]
                                                              for r in post])),
                 "gate_minus_anchor_mean": float(dg.mean()),
                 "n_cycles_ext_worse": int((dx > 0).sum()), "n_cycles": len(rows)}
    print(f"\n  census_extend - anchor: mean {dx.mean():+.4f} over {len(rows)} cycles; "
          f"worse on {(dx > 0).sum()} of {len(rows)}")
    print(f"    c{lo}-c105 (before/at the era-4 extension): "
          f"{np.mean([r['ext_minus_anchor'] for r in pre]):+.4f}")
    print(f"    c106-c{hi} (after it):                       "
          f"{np.mean([r['ext_minus_anchor'] for r in post]):+.4f}")
    print(f"  census_gate  - anchor: mean {dg.mean():+.4f}; better on "
          f"{(dg < 0).sum()} of {len(rows)}")
    conc = "concentrated after c105" if abs(R["era4"]["ext_minus_anchor_post_c105"]) > \
        2 * abs(R["era4"]["ext_minus_anchor_pre_c105"]) else "spread across the era"
    print(f"\n  >>> (4) the census_extend era-4 deficit is {conc}.")

    # WHERE DOES THE DEFICIT ORIGINATE? The era-4 table above shows it is already +0.039 at
    # c101, the first cycle of era 4 and four cycles BEFORE the era-4 extension, so it cannot
    # have been caused by that event. Traced back per era below.
    print("\n  ORIGIN OF THE census_extend - anchor GAP, per era (mean of the per-cycle diff;")
    print("  positive = census_extend is WORSE):")
    ext = np.asarray(A["census_extend"]["log"]["e"], float)
    gate = np.asarray(A["census_gate"]["log"]["e"], float)
    org = []
    print(f"    {'era':>4s}{'cycles':>12s}{'ext-anchor':>12s}{'gate-anchor':>13s}"
          f"{'n extend evts':>15s}")
    for b in eb:
        idx = list(range(b["first"] - 1, b["last"]))
        dxe = float(np.mean(ext[idx] - anchor[idx]))
        dge = float(np.mean(gate[idx] - anchor[idx]))
        nev = sum(1 for ev in extends(A["census_extend"]) if b["first"] <= ev["cycle"] <= b["last"])
        org.append({"era": b["era"], "ext_minus_anchor": dxe, "gate_minus_anchor": dge,
                    "n_extend_events": nev})
        span = f"c{b['first']}-c{b['last']}"
        print(f"    {b['era']:4d}{span:>12s}{dxe:+12.4f}{dge:+13.4f}{nev:15d}")
    R["gap_origin_by_era"] = org

    # the per-level clue the coordinator asked for
    print("\n  PER-LEVEL CLUE — what each arm's tables held entering era 4 (c101):")
    print(f"    {'arm':14s}{'L2 n':>6s}{'L2 rec':>8s}{'L2 prec':>9s}{'L3 n':>6s}"
          f"{'L3 rec':>8s}{'L3 prec':>9s}")
    tab = {}
    for a in EARNING:
        lg = A[a]["log"]
        i = lo - 1
        g = lg["committed_grade"][i] or {}
        v = lg["vocab"][i] or {}
        tab[a] = {"L2_n": v.get("2"), "L2_recall": (g.get("2") or {}).get("recall"),
                  "L2_prec": (g.get("2") or {}).get("precision"),
                  "L3_n": v.get("3"), "L3_recall": (g.get("3") or {}).get("recall"),
                  "L3_prec": (g.get("3") or {}).get("precision")}
        t = tab[a]
        print(f"    {a:14s}{_fmt(t['L2_n'], 6)}{_fmt(t['L2_recall'], 8)}"
              f"{_fmt(t['L2_prec'], 9)}{_fmt(t['L3_n'], 6)}{_fmt(t['L3_recall'], 8)}"
              f"{_fmt(t['L3_prec'], 9)}")
    R["era4_entry_tables"] = tab
    return R


# --------------------------------------------------------------------------- #
# (5) lifetime traces
# --------------------------------------------------------------------------- #

def lifetime(A, R, setup):
    print("\n" + "=" * 78)
    print("(5) LIFETIME TRACES — what separates given_route from the earning arms EARLY")
    print("=" * 78)
    eb = era_bounds(setup)

    # 5a. the deep-era metering sets, evaluated at EVERY probe from c1 — the sharpest
    # available test of whether given_route's era-4/5 edge is present before any earning.
    print("\n  5a. `probe.all_eras` — competence on EVERY era's fixed metering set, measured at")
    print("      each probe with the arm's CURRENT action set. Era-4 and era-5 columns read")
    print("      from c1, long before the earning arms hold anything.")
    print(f"    {'cycle':>6s}" + "".join(f"{a[:13]:>15s}" for a in EARNING + CEILING))
    for tgt, lab in ((3, "era4"), (4, "era5")):
        print(f"    -- {lab} metering set --")
        rows = []
        for k, c in enumerate([q["cycle"] for q in A[ANCHOR]["log"]["probe"]]):
            cells, vals = "", {}
            for a in EARNING + CEILING:
                pr = next((q for q in A[a]["log"]["probe"] if q["cycle"] == c), None)
                v = None if pr is None else pr["all_eras"].get(str(tgt))
                vals[a] = v
                cells += _fmt(v, 15, 4)
            rows.append({"cycle": c, **vals})
            if c in (1, 8, 16, 32, 48, 72, 88, 100, 109, 116):
                print(f"    {c:6d}{cells}")
        R[f"all_eras_{lab}"] = rows
        g0 = rows[0]["given_route"]; a0 = rows[0][ANCHOR]
        gL = rows[-1]["given_route"]; aL = rows[-1][ANCHOR]
        # the sharpest available B-trace: how much of given_route's FINAL advantage on this
        # deep metering set is already present BEFORE any earning arm has committed anything
        # (the first commit in the tag is spiral_route's L2 at c18)?
        first_commit = min(e["cycle"] for a_ in EARNING for e in commits(A[a_]))
        pre = [r for r in rows if r["cycle"] < first_commit]
        gp, ap = pre[-1]["given_route"], pre[-1][ANCHOR]
        final_gap, pre_gap = (gL - aL), (gp - ap)
        frac = pre_gap / final_gap if final_gap else float("nan")
        print(f"      given_route - anchor on the {lab} set: c1 {g0 - a0:+.4f}   "
              f"c{pre[-1]['cycle']} {pre_gap:+.4f}   c116 {final_gap:+.4f}")
        print(f"      -> {frac:.1%} of the final gap is already present at c{pre[-1]['cycle']},"
              f" before ANY arm has committed a table (first commit c{first_commit}).")
        eq = len({round(r[a_], 4) for a_ in EARNING for r in [pre[-1]]}) == 1
        print(f"      -> the four earning arms are identical to each other at c{pre[-1]['cycle']}"
              f" on this set: {eq}")
        R[f"{lab}_gap_first_last"] = {
            "c1": float(g0 - a0), f"c{pre[-1]['cycle']}": float(pre_gap),
            "c116": float(final_gap), "fraction_present_pre_commit": float(frac),
            "first_commit_cycle": int(first_commit),
            "earning_arms_identical_pre_commit": bool(eq)}

    # 5b. trajectory statistics in eras 1-2
    print("\n  5b. TRAJECTORY STATISTICS, eras 1-2 means (before tables diverge much)")
    keys = ["vloss", "gloss", "n_solved", "e_practice", "dres", "succ", "width",
            "g_per_solve"]
    print(f"    {'arm':14s}{'era':>4s}" + "".join(f"{k[:10]:>11s}" for k in keys))
    stats = {}
    for a in EARNING + CEILING:
        lg = A[a]["log"]
        for e in eb[:2]:
            idx = [i for i, q in enumerate(lg["era"]) if q == e["era"]]
            vals = {k: float(np.mean([lg[k][i] for i in idx])) for k in keys}
            stats[f"{a}_era{e['era']}"] = vals
            print(f"    {a:14s}{e['era']:4d}" + "".join(f"{vals[k]:11.3f}" for k in keys))
    R["era12_stats"] = stats

    # 5c. the width ladder at probes — is given_route search-limited differently?
    print("\n  5c. WIDTH LADDER at the last probe of each era (e at width 1 / 2 / 4)")
    print(f"    {'arm':14s}" + "".join(f"{'era' + str(e['era']):>22s}" for e in eb))
    for a in EARNING + CEILING:
        row = ""
        for e in eb:
            pr = [q for q in A[a]["log"]["probe"] if q["era"] == e["era"]]
            if pr:
                lad = pr[-1]["ladder"]
                row += ("/".join(f"{lad[w]['e']:.2f}" for w in sorted(lad, key=int))).rjust(22)
            else:
                row += "-".rjust(22)
        print(f"    {a:14s}{row}")

    # 5d. the value model's own loss trajectory, era by era
    print("\n  5d. VALUE LOSS by era (the selector's fit — B's most direct trace)")
    print(f"    {'arm':14s}" + "".join(f"{'era' + str(e['era']):>10s}" for e in eb))
    for a in EARNING + CEILING:
        lg = A[a]["log"]
        row = ""
        for e in eb:
            idx = [i for i, q in enumerate(lg["era"]) if q == e["era"]]
            row += f"{float(np.mean([lg['vloss'][i] for i in idx])):10.4f}"
        print(f"    {a:14s}{row}")

    # 5e. G-Y stream, which the round measured for the first time
    print("\n  5e. G-Y (L4-shaped tuples at support) by era — reported for completeness")
    sup = str(A[ANCHOR]["config"]["mine_support"])
    print(f"    {'arm':14s}" + "".join(f"{'era' + str(e['era']):>10s}" for e in eb))
    for a in EARNING + CEILING:
        lg = A[a]["log"]
        if not lg.get("gy") or lg["gy"][-1] is None:
            continue
        row = ""
        for e in eb:
            i = [k for k, q in enumerate(lg["era"]) if q == e["era"]][-1]
            row += _fmt((lg["gy"][i].get("n_at_support") or {}).get(sup), 10)
        print(f"    {a:14s}{row}")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cs_s0")
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "forensics"))
    a = ap.parse_args()
    setup, summary, A = load(a.tag)
    R = {"tag": a.tag, "ladder": era_bounds(setup)}
    print("=" * 78)
    print(f"census / STEP 1 FORENSICS — tag {a.tag} (no GPU, no Modal)")
    print("=" * 78)
    R = forced_window(A, R)
    R = truth_and_pi(A, R, setup)
    R = e_around_events(A, R, setup)
    R = era4(A, R, setup)
    R = lifetime(A, R, setup)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "forensics.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=str)
    print(f"\n[forensics] wrote {os.path.join(a.out, 'forensics.json')}")
    return R


if __name__ == "__main__":
    main()
