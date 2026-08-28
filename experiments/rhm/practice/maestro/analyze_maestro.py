"""Reduction for `conductor` (A1) — numbers, not interpretation.

Sections, in the order the round's questions are asked:

  (0) GATES — the full-scale cross-tag replay (this tag's `anchor` against `as_s0/anchor`, the
      whole 116 cycles), the smoke-scale half from `fidelity_smoke`'s `gate.json`, the in-tag
      twin gate (every loop arm against the anchor up to its OWN first action — the strongest
      available statement that the reads are non-invasive), the observation panel's agreement
      with the donor's G-Y miner, and the label-free / floor / policy gates the run recorded.
  (1) THE ACTION TRACE — per arm: what the loop did and when. Commits (cycle, level, the
      statistic at the instant it acted), era advances, chosen vs capped, and whether the arm
      rode its caps to the end.
  (2) GAUGE vs YOKE — the mandatory control (census finding 4). Divergence cycles between each
      gauge arm and its clock-replayed yoke, over every logged series.
  (3) THE THREE CLOCKS — shadow-certificate cycles (read-only in every arm), committed recall
      per level, and recovered fraction per era against the era refs.
  (4) pi PER-LEVEL MASS, ARGMAX SHARE, MACRO EXPANSIONS at every probe.
  (5) THE LEDGER — what each arm read, what it cost, and the priced-time difference between a
      gauge arm and its yoke (which pays nothing for the same trajectory).
  (6) THE MEASURED FLOORS, IN-TAG — the null-ABBA contrast re-derived on this run's own
      schedule arm, beside the offline floors that governed it. `--floors` prints this alone,
      which is how the endo dead zone is measured on the smoke tag before the main run.
  (7) THE SHADOW PANEL — every candidate gauge in every arm, replayed offline through the same
      rule, so a counterfactual decision trace exists for each gauge in each arm.
  (8) THE ADDRESS-BOOK BATTERY, plant guard, certificates, cost.

Usage (from experiments/):
    python3 rhm/practice/maestro/analyze_maestro.py --tag ma_s0 --fetch --figures
    python3 rhm/practice/conductor/analyze_conductor.py --tag cd_smoke --fetch --floors
"""

import argparse
import json
import os
import subprocess

import numpy as np

# [maestro] the FORKED policy module (A1's classes byte-for-byte plus the learned one).
from rhm.practice.maestro.policy import QuietPolicy, null_abba, _sd

try:                          # the arm table, for the yoke's source map; optional (needs modal)
    from rhm.practice.maestro import maestro as MA
except Exception:             # pragma: no cover - the reduction must run without modal
    MA = None

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ASSAY_FIG = os.path.join(os.path.dirname(HERE), "assay", "figures")
# [maestro] A1's fetched copies: the DIRECT donor this fork must replay.
CD_FIG = os.path.join(os.path.dirname(HERE), "conductor", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_maestro"

GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
# `t_cum` carries the ledger, so a priced arm and its unpriced yoke differ on it BY DESIGN.
# The trajectory series are what a yoke has to reproduce.
TRAJ_SERIES = [k for k in GF_SERIES if k != "t_cum"]
PANEL_KEYS = ("ledger", "yield", "endo", "endo_cell", "endo_excess", "yield_active")


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


def arm_era_bounds(lg):
    """Per-ARM era boundaries, read off the log. Under the outer loop an era's length is the
    arm's own decision, so the ladder's counts are only the schedule arm's."""
    out = {}
    for i, q in enumerate(lg["era"]):
        out.setdefault(q, [i + 1, i + 1])[1] = i + 1
    return [{"era": k, "first": v[0], "last": v[1], "n": v[1] - v[0] + 1}
            for k, v in sorted(out.items())]


def _era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def _first_action(res):
    acts = [a for a in (res.get("loop_actions") or [])]
    return min([a["cycle"] for a in acts], default=None)


def _series_skip(res):
    """Era boundaries and commit cycles as 0-based indices — regime changes, not noise."""
    lg = res["log"]
    commits = {int(e["cycle"]) for e in res["events"] if e.get("kind") == "commit"}
    skip = set()
    for i in range(len(lg["era"])):
        if i > 0 and lg["era"][i] != lg["era"][i - 1]:
            skip.add(i)
        if (i + 1) in commits:
            skip.add(i)
    return skip


# --------------------------------------------------------------------------- #

def gates(tag, A, setup, summary, gf_smoke):
    print("\n" + "=" * 78)
    print("(0) GATES")
    print("=" * 78)

    # [maestro] TWO cross-tag replays, not one. The anchor's is the substrate gate. The
    # `outer_yield` one is the round's STRONGEST fidelity statement: A1's thermostat, replayed
    # in this fork, with the learned class present in the file and the level-state keys present
    # in the panel — nothing upstream of it changed, so it must reproduce `cd_s0/outer_yield`
    # bit for bit over all 139 cycles. If it does not, the comparator has moved and no
    # learned-vs-thermostat difference in this tag means anything.
    # [maestro] the cross-tag replay is a FULL-CONFIG gate. Pointed at a `--quick` tag it
    # compares an 800-step substrate against a full one and fails meaninglessly, so it is
    # skipped there rather than printed as a failure a later reader has to re-derive.
    if (setup.get("config") or {}).get("controller_steps", 0) < 5000:
        print("  CROSS-TAG REPLAY: skipped — this tag is `--quick`, so a full-config "
              "comparison is not defined (the substrates differ by construction)")
        _refs = ()
    else:
        _refs = (
            ("anchor", "cd_s0", CD_FIG, "the substrate"),
            ("outer_yield", "cd_s0", CD_FIG, "A1's THERMOSTAT, the comparator"),
            ("anchor", "as_s0", ASSAY_FIG, "the chain back to the donor of A1's donor"))
    for arm, ref_tag, ref_root, why in _refs or (
            ):
        ap = os.path.join(ref_root, ref_tag, arm, "results.json")
        if os.path.isfile(ap) and arm in A:
            old, new_ = json.load(open(ap))["log"], A[arm]["log"]
            worst, per = 0.0, {}
            n = min(len(old["cycle"]), len(new_["cycle"]))
            for k in GF_SERIES:
                x, y = np.asarray(old[k][:n], float), np.asarray(new_[k][:n], float)
                per[k] = float(np.abs(x - y).max()) if n else float("nan")
                worst = max(worst, per[k])
            oc = sorted((int(e["cycle"]), int(e["level"]))
                        for e in json.load(open(ap)).get("events", [])
                        if e.get("kind") == "commit")
            nc = sorted((int(e["cycle"]), int(e["level"]))
                        for e in A[arm].get("events", []) if e.get("kind") == "commit")
            ok = (worst == 0.0) and (oc == nc)
            print(f"  CROSS-TAG REPLAY ({why}): {tag}/{arm} vs {ref_tag}/{arm} over c1-c{n}")
            print(f"    max|delta| = {worst:.3e} over {len(GF_SERIES)} series; "
                  f"commits {nc} vs {oc}  -> {'PASS' if ok else 'FAIL'}")
            if worst:
                print("    per-series: "
                      + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))
        else:
            print(f"  CROSS-TAG REPLAY ({why}): {ref_tag}/{arm} not fetched locally — skipped")

    if gf_smoke:
        ok = gf_smoke["worst_fork"] <= gf_smoke["worst_donor_self_replay"]
        print(f"  G-F smoke scale (in-process vs conductor.py, panel ON, learning OFF): "
              f"max|fork - donor| = {gf_smoke['worst_fork']:.3e}, self-replay control = "
              f"{gf_smoke['worst_donor_self_replay']:.3e}, commits equal = "
              f"{gf_smoke['events_equal']}  -> {'PASS' if ok else 'FAIL'}")

    # THE PRE-TREATMENT WINDOW ends at the first cycle at which the two arms' behaviour COULD
    # differ — which is the first action taken by EITHER of them. The anchor is not passive: it
    # commits on its own certificate (c18 here), and after that it is the ANCHOR that has been
    # treated, so a window running past it compares a committed arm against an uncommitted one
    # and "fails" on the schedule's own action. `assay/analyze_assay.py` states this for its
    # gate arms in as many words; the first version of this gate ignored it and reported
    # outer_yield/yoked_yield as FAIL at 2.6e+02 when they are 0.000e+00 through c17 and first
    # differ at c19 — one cycle after the anchor's commit.
    a_commits = [e["cycle"] for e in A["anchor"]["events"] if e.get("kind") == "commit"]
    a_first = min(a_commits) if a_commits else None
    print(f"\n  IN-TAG TWIN GATE (each arm vs the anchor, to the first action by EITHER; "
          f"the anchor's own first commit is c{a_first}):")
    for t in summary["order"]:
        if t == "anchor":
            continue
        fa = _first_action(A[t])
        n_both = min(len(A[t]["log"]["cycle"]), len(A["anchor"]["log"]["cycle"]))
        cands = [x for x in (fa, a_first) if x is not None]
        if not cands:
            w, note = n_both, "neither acted"
        else:
            w = min(cands) - 1
            who = "own" if (fa is not None and fa == min(cands)) else "anchor"
            note = f"first action c{min(cands)} ({who})"
        d = 0.0
        for k in TRAJ_SERIES:
            x = np.asarray(A[t]["log"][k][:w], float)
            y = np.asarray(A["anchor"]["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        print(f"    {t:14s} c1-c{w:<4d} ({note:16s}) max|delta| = {d:.3e}"
              f"  -> {'PASS' if d == 0.0 else 'FAIL'}")

    print("\n  OBSERVATION PANEL vs the donor's G-Y miner (level 4, every cycle):")
    for a in summary["order"]:
        r = A[a]
        print(f"    {a:14s} agree={r.get('obs_gy_agree')} over {r.get('obs_gy_agree_n')} cycles")

    for key, label in (("policy_gate", "POLICY GATE (offline, no GPU)"),
                       ("endo_gate", "ENDO READ: LABEL-FREE BY CONSTRUCTION"),
                       ("floor_gate", "DEAD ZONES: measured, not defaulted")):
        g = (setup or {}).get(key)
        if not g:
            continue
        if key == "policy_gate":
            fails = [k for k, v in g.items() if k != "ALL" and not v.get("pass")]
            print(f"\n  {label}: {'ALL PASS' if g.get('ALL') else 'FAIL ' + str(fails)} "
                  f"({sum(1 for k in g if k != 'ALL')} checks)")
        elif key == "endo_gate":
            print(f"\n  {label}: {g['verdict']} — positions depend only on {g['depends_on']}; "
                  f"none of {g['banned_absent']} appears in the read's code")
            for r in g["rows"]:
                print(f"    era {r['era']} cell {r['cell']} blocks {r['cell_blocks']} -> "
                      f"parent L{r['parent_level']}n{r['parent_node']} blocks "
                      f"{r['parent_blocks']}"
                      + ("  (parent IS the root: the read is the unconditional marginal)"
                         if r["parent_is_root"] else ""))
        else:
            print(f"\n  {label}: {g['verdict']}, matches floors.json = "
                  f"{g.get('matches_floors_json')}")
            print(f"    {g['used']}")
            print(f"    provenance: {g['provenance']}")
    b = (setup or {}).get("endo_bench")
    if b:
        print(f"\n  ENDO PRICE (measured on the run's own GPU): one read over {b['n']} "
              f"sequences = {b['t_endo_s'] * 1e3:.1f} ms = {b['price_g']} "
              f"grounding-equivalents; charged {setup['config']['endo_price']}g")


def action_trace(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(1) THE ACTION TRACE — what the loop did, and when")
    print("=" * 78)
    print(f"  {'arm':14s}{'policy':>10s}{'read':>8s}{'cycles':>7s}{'commits':>9s}"
          f"{'advances':>10s}{'chosen':>8s}{'capped':>8s}{'rode cap':>10s}")
    for a in summary["order"]:
        lp = A[a].get("loop") or {}
        print(f"  {a:14s}{str(lp.get('policy')):>10s}{str(lp.get('read')):>8s}"
              f"{len(A[a]['log']['cycle']):7d}{lp.get('n_commits', 0):9d}"
              f"{lp.get('n_advances', 0):10d}{lp.get('n_chosen', 0):8d}"
              f"{lp.get('n_capped', 0):8d}{str(lp.get('rode_cap_to_end')):>10s}")

    print("\n  PER-ARM ACTIONS (cycle, kind, level, why, and the statistic at that instant)")
    for a in summary["order"]:
        acts = A[a].get("loop_actions") or []
        eb = arm_era_bounds(A[a]["log"])
        print(f"\n    {a}  eras: " + ", ".join(
            f"e{e['era']} c{e['first']}-{e['last']} ({e['n']}c)" for e in eb))
        if not acts:
            print("      (no loop actions — the schedule arm)")
            continue
        for q in acts:
            vm = q.get("v_mult")
            print(f"      c{q['cycle']:<4d} era{q['era']} {q['kind']:8s} "
                  f"L{q.get('level') if q.get('level') else '-':<2} why={q['why']:6s} "
                  f"V={_fmt(q.get('V'), 9, 4)} tol={_fmt(q.get('v_tol'), 8, 4)} "
                  f"V/tol={_fmt(vm, 8, 2)}"
                  + (f"  CANCELLED({q['cancelled']})" if q.get("cancelled") else ""))

    print("\n  COMMIT CYCLES AND WHAT WAS COMMITTED")
    print(f"  {'arm':14s}{'lvl':>4s}{'cycle':>7s}{'era':>5s}{'prov':>6s}{'entries':>9s}"
          f"{'recall':>8s}{'prec':>8s}{'cert@':>7s}{'driver':>10s}")
    for a in summary["order"]:
        for ev in summary["arms"][a]["commits"]:
            print(f"  {a:14s}{ev['level']:>4d}{ev['cycle']:>7d}{ev['era']:>5d}"
                  f"{str(ev['provisional']):>6s}{ev['n_entries']:>9d}"
                  f"{_fmt(ev.get('tab_recall'), 8)}{_fmt(ev.get('tab_precision'), 8)}"
                  f"{str(ev.get('cert_fired')):>7s}{str(ev.get('driver')):>10s}")


# --------------------------------------------------------------------------- #
# [maestro] THE POLICY'S OWN RECORD — what was learned, and where it differs
# --------------------------------------------------------------------------- #
def policy_record(tag, A, setup, summary):
    """The fitted policies that governed the run, their offline diagnostics, and the
    learned-vs-thermostat action diff. Facts only; nothing here is interpreted."""
    print("\n" + "=" * 78)
    print("(1b) THE LEARNED POLICY'S OWN RECORD — the mixtures, the fit, the action diff")
    print("=" * 78)
    fitted = setup.get("fitted") or {}
    if not fitted:
        print("  (no learned arms in this tag)")
        return
    fitjson = None
    fp = os.path.join(HERE, "fit.json")
    if os.path.isfile(fp):
        fitjson = json.load(open(fp))
    gauges = fitted["yield"]["gauges"]
    print("  THE FITTED MIXTURES (unit-norm weights on each gauge's paired-interval slope,")
    print("  IN FLOOR UNITS; the bucket is `will_commit` = would a firing install a table).")
    _e = ", ".join("1" if g == "yield" else "0" for g in gauges)
    print(f"  A1's thermostat is the point ({_e}) with theta = 1 in this same class.")
    print(f"    {'arm/reward':16s}{'bucket':>8s}   " + "".join(f"{g:>14s}" for g in gauges)
          + f"{'theta':>9s}")
    for rw in ("yield", "task"):
        f = fitted[rw]
        for b in f["buckets"]:
            lbl = "installs" if b == "1" else "advances"
            print(f"    learned_{rw:<8s}{lbl:>8s}   "
                  + "".join(f"{x:>+14.4f}" for x in f["mixes"][b])
                  + f"{f['thetas'][b]:>9.4f}")
    if fitjson:
        print("\n  OFFLINE FIT DIAGNOSTICS (from fit.json; corpus "
              f"{fitjson['n_windows']} windows over {len(fitjson['fits']['yield']['corpus'])} "
              "distinct trajectories, leave-one-ARM-out)")
        print(f"    {'reward':8s}{'bucket':>8s}{'n':>6s}{'lambda':>9s}{'LOAO mse':>10s}"
              f"{'thermostat':>11s}{'ratio':>7s}{'R2 vs 0':>9s}{'fold cos':>26s}")
        for rw in ("yield", "task"):
            for b in ("1", "0"):
                d = fitjson["fits"][rw]["diagnostics"][b]
                fc = ", ".join(f"{x:.2f}" for x in d["fold_cos"])
                print(f"    {rw:8s}{b:>8s}{d['n']:>6d}{d['lam']:>9g}{d['loao_mse']:>10.4f}"
                      f"{d['thermostat_mse']:>11.4f}"
                      f"{d['mse_ratio_vs_thermostat']:>7.3f}{d['r2_vs_zero']:>+9.3f}"
                      f"{fc:>26s}")
        print("    'thermostat' is the hand-written rule's own predictor of the same forward")
        print("    rate (that currency's trailing rate, coefficient 1, no free parameters);")
        print("    'ratio' < 1 means the fitted mixture predicts it better OUT OF ARM.")
        rel = fitjson["relation"]
        print("\n  HOW THE TWO MIXTURES RELATE (cosine)")
        for b in ("1", "0"):
            r = rel[b]
            print(f"    will_commit={b}:  yield.task {r['cos_yield_task']:+.3f}   "
                  f"yield.thermostat {r['cos_yield_thermostat']:+.3f}   "
                  f"task.thermostat {r['cos_task_thermostat']:+.3f}")

    print("\n  THE ACTION DIFF — learned vs the hand-written thermostat, IN THIS TAG")
    print("  (both ran live on their own trajectories, so this is a difference between two")
    print("   runs and not a counterfactual; the pre-GPU replay is in fit.json.)")
    ref = "outer_yield"
    if ref in A:
        rl = [(q["cycle"], q["kind"], q.get("level"), q["why"])
              for q in (A[ref].get("loop_actions") or []) if not q.get("cancelled")]
        print(f"    {ref:16s} (thermostat) {rl}")
        for arm in ("learned_yield", "learned_task"):
            if arm not in A:
                continue
            ll = [(q["cycle"], q["kind"], q.get("level"), q["why"])
                  for q in (A[arm].get("loop_actions") or []) if not q.get("cancelled")]
            print(f"    {arm:16s}              {ll}")
            for kind in ("commit", "advance"):
                t = [q[0] for q in rl if q[1] == kind]
                l = [q[0] for q in ll if q[1] == kind]
                d = [l[i] - t[i] for i in range(min(len(t), len(l)))]
                print(f"      {kind:8s}: thermostat {t}  learned {l}  "
                      f"delta(cycles, pairwise) {d}")

    print("\n  WHAT THE LEARNED POLICY READ AT EACH OF ITS OWN ACTIONS")
    for arm in ("learned_yield", "learned_task"):
        if arm not in A:
            continue
        print(f"    {arm}")
        for q in (A[arm].get("loop_actions") or []):
            pg = q.get("per_gauge") or {}
            comp = "  ".join(
                f"{g}={_fmt((pg.get(g) or {}).get('V'), 7, 2)}" for g in gauges)
            print(f"      c{q['cycle']:<4d} era{q['era']} {q['kind']:8s} "
                  f"L{q.get('level') if q.get('level') else '-':<2} why={q['why']:6s} "
                  f"bucket={str(q.get('bucket')):>4s} Vhat={_fmt(q.get('V'), 8, 3)} "
                  f"theta={_fmt(q.get('v_tol'), 7, 3)} Vhat/theta={_fmt(q.get('v_mult'), 7, 2)}"
                  f"   inputs (floor units): {comp}"
                  + (f"  CANCELLED({q['cancelled']})" if q.get("cancelled") else ""))


def gauge_vs_yoke(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(2) GAUGE vs YOKE — the mandatory timing control (census finding 4)")
    print("=" * 78)
    print("  The yoke replays the gauge arm's realised action cycles by clock and reads")
    print("  nothing. A divergence would mean the reads are not inert; identity means the")
    print("  criterion contributed nothing beyond a cycle number, which is a finding, and the")
    print("  priced-time column is then what the read cost for a trajectory a clock reproduces.")
    pairs = []
    for a in summary["order"]:
        lp = A[a].get("loop") or {}
        if lp.get("policy") == "yoke":
            # [maestro] `yoked_learned` yokes onto `learned_yield`, so the name map is no
            # longer a single prefix swap; read the source off the arm table instead.
            src = ((MA.ARMS.get(a, {}).get("loop") or {}).get("of")
                   if MA is not None else None) or a.replace("yoked_", "outer_")
            if src in A:
                pairs.append((src, a))
    if not pairs:
        print("  (no yoked arms in this tag)")
        return
    for src, yk in pairs:
        n = min(len(A[src]["log"]["cycle"]), len(A[yk]["log"]["cycle"]))
        worst, first_div, per = 0.0, None, {}
        for k in TRAJ_SERIES:
            x = np.asarray(A[src]["log"][k][:n], float)
            y = np.asarray(A[yk]["log"][k][:n], float)
            d = np.abs(x - y)
            per[k] = float(d.max()) if n else float("nan")
            worst = max(worst, per[k])
            nz = np.nonzero(d)[0]
            if nz.size:
                first_div = nz[0] + 1 if first_div is None else min(first_div, nz[0] + 1)
        ca = [(e["level"], e["cycle"]) for e in summary["arms"][src]["commits"]]
        cb = [(e["level"], e["cycle"]) for e in summary["arms"][yk]["commits"]]
        aa = [e["cycle"] for e in summary["arms"][src].get("advances", [])]
        ab = [e["cycle"] for e in summary["arms"][yk].get("advances", [])]
        ta = A[src]["log"]["t_cum"][-1]
        tb = A[yk]["log"]["t_cum"][-1]
        print(f"\n  {src} vs {yk}  (n={n} cycles)")
        print(f"    max|delta| over {len(TRAJ_SERIES)} trajectory series = {worst:.3e}; "
              f"first divergence cycle = {first_div}")
        if worst:
            print("    per-series: " + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))
        print(f"    commits {ca} vs {cb}   equal = {ca == cb}")
        print(f"    advances {aa} vs {ab}   equal = {aa == ab}")
        print(f"    priced time t_cum {ta:.0f} vs {tb:.0f}  (difference {ta - tb:+.0f} = what "
              f"the read cost)")


def three_clocks(tag, A, setup, summary):
    refs, n_era = setup["refs"], len(setup["eras"])
    print("\n" + "=" * 78)
    print("(3) THE THREE CLOCKS: certificate, coverage, value")
    print("=" * 78)

    print("  CLOCK 1 — the SHADOW certificate (read-only in every arm, incl. the driven ones)")
    print(f"  {'arm':14s}{'L2 fired':>10s}{'L3 fired':>10s}{'L2 c2cert':>11s}"
          f"{'L3 c2cert':>11s}{'commit L2':>11s}{'commit L3':>11s}")
    for a in summary["order"]:
        sc = summary["arms"][a]["shadow_cert"]
        cm = {e["level"]: e["cycle"] for e in summary["arms"][a]["commits"]}
        print(f"  {a:14s}{str(sc['2'].get('fired')):>10s}"
              f"{str(sc.get('3', {}).get('fired')):>10s}"
              f"{str(sc['2'].get('cycles_to_cert')):>11s}"
              f"{str(sc.get('3', {}).get('cycles_to_cert')):>11s}"
              f"{str(cm.get(2)):>11s}{str(cm.get(3)):>11s}")

    print("\n  CLOCK 2 — COVERAGE: end-of-run committed table per level")
    print(f"  {'arm':14s}{'lvl':>4s}{'frozen n':>10s}{'recall':>9s}{'prec':>8s}"
          f"{'live n':>8s}{'live rec':>10s}{'live prec':>11s}")
    for a in summary["order"]:
        lg = A[a]["log"]
        for lv in ("2", "3"):
            cg = next((c.get(lv) for c in reversed(lg["committed_grade"]) if (c or {}).get(lv)),
                      None)
            fz = next((c.get(lv) for c in reversed(lg["vocab"]) if (c or {}).get(lv)), None)
            au = lg["aud"][-1].get(lv) or {}
            print(f"  {a:14s}{lv:>4s}{_fmt(fz, 10)}{_fmt((cg or {}).get('recall'), 9)}"
                  f"{_fmt((cg or {}).get('precision'), 8)}{_fmt(au.get('n_entries'), 8)}"
                  f"{_fmt(au.get('tab_recall'), 10)}{_fmt(au.get('tab_precision'), 11)}")

    print("\n  CLOCK 3 — VALUE: recovered fraction per era (e, and (stale-e)/(stale-floor))")
    st, fl = refs["stale"], refs["floor"]
    print(f"  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>17s}" for j in range(n_era)))
    rows = {}
    for a in summary["order"]:
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
        print(f"  {a:14s}{row}")
    print(f"  {'stale':14s}" + "".join(f"{st[j]:>17.3f}" for j in range(n_era)))
    print(f"  {'floor':14s}" + "".join(f"{fl[j]:>17.3f}" for j in range(n_era)))
    print("\n  DELTA TO THE ANCHOR (the scheduled crank), per era, in recovered fraction")
    print("  Read against the MEASURED depth-6 stream floor (census finding 7): 0.087 in the")
    print("  earning family and 0.083/0.148/0.344 in the given family for eras 3/4/5. Ranks")
    print("  and signs in the deep eras are the currency; fractions are not.")
    print(f"  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
    for a in summary["order"]:
        if a == "anchor":
            continue
        print(f"  {a:14s}" + "".join(
            _fmt(None if (rows[a][j] is None or rows["anchor"][j] is None)
                 else rows[a][j] - rows["anchor"][j], 10)
            for j in range(n_era)))
    return rows


def pi_tables(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(4) pi PER-LEVEL MASS, ARGMAX SHARE, AND MACRO EXPANSIONS, per arm per era")
    print("=" * 78)
    n_era = len(setup["eras"])
    for metric, key in (("proposal mass p_macro_all", "p"),
                        ("argmax share frac_argmax", "argmax")):
        for lv in (2, 3):
            print(f"\n  L{lv} {metric} (last probe of each era)")
            print(f"  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>10s}"
                                             for j in range(n_era)))
            for a in summary["order"]:
                row = ""
                for j in range(n_era):
                    pr = [q for q in A[a]["log"]["probe"]
                          if q["era"] == j + 1 and q.get("pi", {}).get("macros")]
                    if not pr:
                        row += "-".rjust(10); continue
                    tot = sum(m.get("p_macro_all" if key == "p" else "frac_argmax", 0.0)
                              for m in pr[-1]["pi"]["macros"] if m["level"] == lv)
                    row += f"{tot:10.3f}"
                print(f"  {a:14s}{row}")
    print("\n  MACRO EXPANSIONS the beam ran (blocks.mat_dp, per-cycle mean by era)")
    print(f"  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>12s}" for j in range(n_era)))
    for a in summary["order"]:
        lg = A[a]["log"]
        row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            row += (f"{float(np.mean([lg['blocks'][i].get('mat_dp', 0) for i in idx])):12.0f}"
                    if idx else "-".rjust(12))
        print(f"  {a:14s}{row}")


def ledger_table(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(5) THE LEDGER — what each arm read, and what it cost")
    print("=" * 78)
    print("  ear's convention: the learner's own experience is free, a counterfactual read is")
    print("  priced, and instruments logged in every arm are free. Only the POLICY's input")
    print("  stream is charged, and the charge lands in `counts['ground']` -> `t_cum`, which")
    print("  nothing in the cycle loop reads.")
    price = int((setup.get("config") or {}).get("endo_price") or 0)
    print(f"  {'arm':14s}{'read':>12s}{'n reads':>9s}{'priced':>8s}{'spend(g)':>10s}"
          f"{'t_cum end':>13s}{'t_cum+read':>13s}{'read % of':>11s}")
    recon = []
    for a in summary["order"]:
        L = A[a].get("ledger") or {}
        lp = A[a].get("loop") or {}
        t = A[a]["log"]["t_cum"][-1]
        sp = L.get("spend_g", 0)
        n_reads = L.get("n_reads") or {}
        key = lp.get("read")
        # RECONSTRUCTION. `charge` records `n_reads` before it prices, so a read that was
        # charged 0 through the `cd_s0` price-key mismatch is recovered exactly as
        # n_reads x the pinned per-read price. Flagged, never silently substituted.
        r = sp
        if key and key not in ("ledger", "yield") and sp == 0 and n_reads.get(key):
            r = int(n_reads[key]) * price
            recon.append(a)
        tt = t + (r - sp)
        print(f"  {a:14s}{str(key):>12s}{sum(n_reads.values()):9d}"
              f"{L.get('n_priced_reads', 0):8d}{r:10d}{t:13.0f}{tt:13.0f}"
              f"{(r / tt * 100 if tt else 0):10.3f}%"
              + ("   <- reconstructed" if a in recon else ""))
    if recon:
        print(f"\n  RECONSTRUCTED for {recon}: this tag priced the endo read under the key "
              f"`endo` while the driven key was `endo_excess`, so `charge` recorded the reads "
              f"and priced them at 0. Trajectories are unaffected BY CONSTRUCTION — a charge "
              f"reaches only counts['ground'] -> t_cum, which nothing in the cycle loop reads, "
              f"and the arm's yoke reproduces it bit-for-bit either way. The spend column above "
              f"is n_reads x {price} g/read (the pinned `endo_bench` price); `t_cum end` is what "
              f"the run recorded and `t_cum+read` what it would have been.")


def in_tag_floors(tag, A, setup, summary, quiet=False):
    """(6) THE MEASURED FLOORS, IN-TAG. The donor's null-ABBA contrast, re-derived on the
    schedule arm's own panel series — a fixed-condition series by construction. This is how the
    ENDO dead zone is measured (it has no offline series), and how the two that do are checked
    against the offline derivation that governed the run."""
    if not quiet:
        print("\n" + "=" * 78)
        print("(6) THE MEASURED FLOORS, RE-DERIVED IN-TAG on the schedule arm")
        print("=" * 78)
    ref = "anchor" if "anchor" in A else summary["order"][0]
    res = A[ref]
    panel = res["log"].get("panel") or []
    if not panel:
        print("  (no shadow panel in this tag)")
        return {}
    skip = _series_skip(res)
    cfg = res["config"]
    span, W = cfg.get("loop_span", 1), cfg.get("loop_W", 4)
    gov = {"ledger": cfg.get("tol_ledger"), "yield": None, "endo": cfg.get("tol_endo")}
    out = {}
    print(f"  reference arm: {ref} ({len(panel)} cycles, {len(skip)} windows dropped at era "
          f"boundaries / commits); span={span} W={W}")
    print(f"  {'gauge':12s}{'n_win':>7s}{'sd(N)':>11s}{'mean(N)':>11s}{'v_tol=sd/2':>12s}"
          f"{'mean(D)':>11s}{'D/floor':>9s}{'governed by':>14s}")
    for k in PANEL_KEYS:
        ser = [q.get(k) for q in panel]
        if any(x is None for x in ser):
            continue
        N, D = null_abba(ser, skip=skip, span=span, W=W)
        if len(N) < 8:
            continue
        tol = _sd(N) / (W ** 0.5)
        mD = sum(D) / len(D)
        g = gov.get(k)
        if k == "yield":
            g = f"L3 {cfg.get('tol_yield_l3'):.4f}/L4 {cfg.get('tol_yield_l4'):.4f}"
        out[k] = {"n_windows": len(N), "sd_N": _sd(N), "mean_N": sum(N) / len(N),
                  "v_tol_in_tag": tol, "mean_D": mD}
        print(f"  {k:12s}{len(N):7d}{_sd(N):11.5f}{sum(N) / len(N):+11.5f}{tol:12.5f}"
              f"{mD:+11.5f}{(mD / tol if tol else 0):9.2f}"
              f"{(f'{g:.5f}' if isinstance(g, float) else str(g)):>14s}")
    if "endo" in out:
        print(f"\n  >>> THE ENDO DEAD ZONE, measured on {tag}/{ref}: "
              f"v_tol = {out['endo']['v_tol_in_tag']:.6g}")
        print(f"      pass it to the main run as  --tol-endo {out['endo']['v_tol_in_tag']:.6g}")
    return out


def shadow_panel(tag, A, setup, summary, floors_in_tag):
    print("\n" + "=" * 78)
    print("(7) THE SHADOW PANEL — every gauge replayed through the same rule, in every arm")
    print("=" * 78)
    print("  Counterfactual: what each gauge WOULD have licensed in each arm, per era, if it")
    print("  had been the driven read. Exact only up to the arm's first action (after that the")
    print("  arm's own trajectory is the one the gauge was measured on), so `first_action` is")
    print("  printed beside it and a shadow decision must never be read as a measured one.")
    cfg = A[summary["order"][0]]["config"]
    tolmap = {"ledger": cfg.get("tol_ledger"), "endo": cfg.get("tol_endo"),
              "endo_cell": cfg.get("tol_endo"), "endo_excess": cfg.get("tol_endo"),
              "yield_active": cfg.get("tol_yield_l3")}
    for a in summary["order"]:
        panel = A[a]["log"].get("panel") or []
        if not panel:
            continue
        eb = arm_era_bounds(A[a]["log"])
        fa = _first_action(A[a])
        print(f"\n    {a}  (first action c{fa})")
        for k in PANEL_KEYS:
            if any(q.get(k) is None for q in panel):
                continue
            tol = tolmap.get(k)
            if k == "yield":
                pass
            if tol is None and k != "yield":
                continue
            rows = []
            for e in eb:
                lo, hi = e["first"] - 1, e["last"] - 1
                sub = panel[lo:hi + 1]
                t = (cfg["tol_yield_l3"] if k == "yield"
                     and sub and sub[0].get("yield_level") == 3
                     else cfg["tol_yield_l4"]) if k == "yield" else tol
                p = QuietPolicy(k, v_tol=t, burn=cfg.get("loop_burn", 4),
                                span=cfg.get("loop_span", 1), W=cfg.get("loop_W", 4))
                first, armed = None, None
                for i, q in enumerate(sub):
                    p.step(lo + i + 1, {k: q[k]})
                    if p.moved and armed is None:
                        armed = lo + i + 1
                    if p.quiet and first is None:
                        first = lo + i + 1
                rows.append(f"e{e['era']}[{e['n']}c armed@{armed} quiet@{first}]")
            print(f"      {k:12s} " + "  ".join(rows))


# --------------------------------------------------------------------------- #
# [maestro] THE STREAM-POSITION FLOOR — displaced twins (`../assay/`'s mechanic)
# --------------------------------------------------------------------------- #
def stream_floor(tag, A, setup, summary):
    """Each displaced twin is its original in every config bit, with the per-arm RNG stream
    displaced by a controlled burn. `|original - displaced|` is therefore what stream POSITION
    alone moves — the floor a pairwise ordering has to clear.

    A2 asks one thing of it beyond `assay`'s use: the learned-vs-thermostat era-4/5 ordering is
    the cell census finding 7 demotes, so the readout is whether that RANK SURVIVES
    DISPLACEMENT, on both draws. The twins' own action cycles are printed beside it — whether a
    loop rule acts at different cycles under a displaced stream is itself a reading on how much
    of `when it acts` is the gauge and how much is the draw."""
    pairs = [(o, o + "_j") for o in ("outer_yield", "learned_yield") if o + "_j" in A]
    if not pairs:
        return
    print("\n" + "=" * 78)
    print("(9) THE STREAM-POSITION FLOOR — displaced twins")
    print("=" * 78)
    st, fl = setup["refs"]["stale"], setup["refs"]["floor"]
    n_era = len(setup["eras"])

    def _rec(a, j):
        lg = A[a]["log"]
        idx = _era_idx(lg, j)
        if not idx:
            return None, None
        e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
        return e, (st[j] - e) / (st[j] - fl[j])

    for o, tw in pairs:
        b = A[tw].get("stream_burn") or {}
        print(f"\n  {tw} vs {o}   burn = {b.get('draws')} draws "
              f"(cpu={b.get('cpu')}, cuda={b.get('cuda')}) at: {b.get('point')}")
        print(f"    {'era':>4s}{'orig e':>10s}{'twin e':>10s}{'|de|':>9s}"
              f"{'orig rec':>10s}{'twin rec':>10s}{'|d rec|':>9s}")
        for j in range(n_era):
            eo, ro = _rec(o, j)
            et, rt = _rec(tw, j)
            if eo is None or et is None:
                continue
            print(f"    {j + 1:4d}{eo:10.4f}{et:10.4f}{abs(et - eo):9.4f}"
                  f"{ro:+10.3f}{rt:+10.3f}{abs(rt - ro):9.3f}")
        d = [abs(_rec(tw, j)[1] - _rec(o, j)[1]) for j in range(2, n_era)
             if _rec(tw, j)[1] is not None and _rec(o, j)[1] is not None]
        if d:
            print(f"    eras 3-{n_era} |d recovered fraction|: {[round(x, 3) for x in d]}  "
                  f"max {max(d):.3f}  mean {float(np.mean(d)):.3f}")
        lo = np.asarray(A[o]["log"]["e"], float)
        lt = np.asarray(A[tw]["log"]["e"], float)
        n = min(len(lo), len(lt))
        nz = np.nonzero(np.abs(lo[:n] - lt[:n]))[0]
        print(f"    first differing cycle: c{int(nz[0]) + 1 if nz.size else None}; "
              f"per-cycle |de| mean {float(np.abs(lo[:n] - lt[:n]).mean()):.4f} "
              f"max {float(np.abs(lo[:n] - lt[:n]).max()):.4f}")
        ao = [(q["cycle"], q["kind"], q.get("level"), q["why"])
              for q in (A[o].get("loop_actions") or []) if not q.get("cancelled")]
        at = [(q["cycle"], q["kind"], q.get("level"), q["why"])
              for q in (A[tw].get("loop_actions") or []) if not q.get("cancelled")]
        print(f"    ACTIONS  original : {ao}")
        print(f"    ACTIONS  displaced: {at}")
        co = [q[0] for q in ao if q[1] == "commit"]
        ct = [q[0] for q in at if q[1] == "commit"]
        vo = [q[0] for q in ao if q[1] == "advance"]
        vt = [q[0] for q in at if q[1] == "advance"]
        print(f"      commits  {co} -> {ct}   delta "
              f"{[ct[i] - co[i] for i in range(min(len(co), len(ct)))]}")
        print(f"      advances {vo} -> {vt}   delta "
              f"{[vt[i] - vo[i] for i in range(min(len(vo), len(vt)))]}")
        print(f"      n_cycles {len(A[o]['log']['cycle'])} -> "
              f"{len(A[tw]['log']['cycle'])}")

    # --- the comparison this exists for ------------------------------------------------ #
    if all(k in A for k in ("outer_yield", "learned_yield",
                            "outer_yield_j", "learned_yield_j")):
        print("\n  THE COMPARISON THIS EXISTS FOR — does the learned-vs-thermostat rank")
        print("  survive displacement? Two independent draws of the same ordering; the")
        print("  per-arm displacement magnitudes are the floor it is read against.")
        print(f"    {'era':>4s}{'draw A (undisplaced)':>23s}{'draw B (displaced)':>21s}"
              f"{'floor: |d| oy':>15s}{'|d| ly':>9s}{'rank holds':>12s}")
        for j in range(n_era):
            _, r_oy = _rec("outer_yield", j)
            _, r_ly = _rec("learned_yield", j)
            _, r_oyj = _rec("outer_yield_j", j)
            _, r_lyj = _rec("learned_yield_j", j)
            if None in (r_oy, r_ly, r_oyj, r_lyj):
                continue
            a_d, b_d = r_ly - r_oy, r_lyj - r_oyj
            holds = (a_d > 0) == (b_d > 0)
            print(f"    {j + 1:4d}{a_d:+23.3f}{b_d:+21.3f}"
                  f"{abs(r_oyj - r_oy):15.3f}{abs(r_lyj - r_ly):9.3f}"
                  f"{('YES' if holds else 'NO'):>12s}")
        print("    (draw A/B = learned_yield - outer_yield in recovered fraction; 'rank holds'")
        print("     means the sign of that difference is the same on both draws.)")


def battery_and_tail(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(8) THE ADDRESS-BOOK BATTERY, plant guard, certificates, cost")
    print("=" * 78)
    sup = str(A[summary["order"][0]]["config"]["mine_support"])
    conds = ("a_full", "b_table", "b_span", "c_prims")
    print("  END-OF-RUN BATTERY: e per condition (last era), and the next-level currency split")
    print(f"  {'arm':14s}{'cond':>9s}{'e(last era)':>13s}{'g/solve':>9s}{'solved':>8s}"
          f"{'T3@sup':>8s}{'T3built':>9s}")
    for a in summary["order"]:
        ab = A[a].get("ablation") or {}
        for c in conds:
            cell = ab.get(c)
            if not cell or not isinstance(cell, dict) or "eras" not in cell:
                continue
            last = sorted(cell["eras"])[-1]
            r = cell["eras"][last]
            m3 = (r.get("mine") or {}).get("3", {})
            print(f"  {a:14s}{c:>9s}{_fmt(r.get('e'), 13)}{_fmt(r.get('g_solve'), 9, 1)}"
                  f"{_fmt(r.get('n_solved'), 8)}{_fmt(m3.get('at_support'), 8)}"
                  f"{_fmt(m3.get('built'), 9)}")

    print("\n  G-Y / OBSERVATION PANEL: distinct next-level tuples at support, end of each era")
    n_era = len(setup["eras"])
    for a in summary["order"]:
        lg = A[a]["log"]
        oh = A[a].get("obs_hist") or {}
        row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                row += "-".rjust(9); continue
            i = idx[-1]
            row += f"{(oh.get('3') or [None])[i] if oh.get('3') else None}/" \
                   f"{(oh.get('4') or [None])[i] if oh.get('4') else None}".rjust(9)
        print(f"  {a:14s} L3/L4 at support {sup}: {row}")

    print("\n  PLANT GUARD, CERTIFICATES, COST")
    for a in summary["order"]:
        pr = A[a]["log"]["probe"]
        pa = [q["plant"]["parse_acc"] for q in pr]
        inf = [q["plant"]["infill_acc"] for q in pr]
        sc = summary["arms"][a]["shadow_cert"]
        print(f"  {a:14s} parse {min(pa):.3f}-{max(pa):.3f}  infill {min(inf):.3f}-{max(inf):.3f}"
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
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)

    # fig 1 — competence, with each arm's OWN era boundaries and its own actions marked
    fig, ax = plt.subplots(figsize=(12, 5.5))
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, a in enumerate(order):
        lg = A[a]["log"]
        ax.plot(lg["cycle"], lg["e"], lw=1.4, color=cols[i % 10], label=a)
        for q in (A[a].get("loop_actions") or []):
            if q.get("cancelled"):
                continue
            y = lg["e"][q["cycle"] - 1]
            ax.plot(q["cycle"], y, "v" if q["kind"] == "commit" else "s",
                    ms=7 if q["kind"] == "commit" else 5, color=cols[i % 10],
                    mec="k", mew=0.6, zorder=5)
        for ev in summary["arms"][a]["commits"]:
            ax.plot(ev["cycle"], lg["e"][ev["cycle"] - 1], "v", ms=7, color=cols[i % 10],
                    mec="k", mew=0.6, zorder=5)
    ax.set_xlabel("cycle"); ax.set_ylabel("e (1 - terminal success)")
    ax.set_title(f"{tag}: competence.  v = commit, square = era advance "
                 f"(era lengths differ per arm under the loop)")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=140); plt.close(fig)

    # fig 2 — the gauge each arm read, its dead zone, and where it acted
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for i, a in enumerate(order):
        panel = A[a]["log"].get("panel") or []
        if not panel:
            continue
        cyc = [q["cycle"] for q in panel]
        for ax_, k in zip(axes, ("yield", "endo", "ledger")):
            ys = [q.get(k) for q in panel]
            if any(y is None for y in ys):
                continue
            ax_.plot(cyc, ys, lw=1.2, color=cols[i % 10], label=a)
    for ax_, k, t in zip(axes, ("yield", "endo", "ledger"),
                         ("yield: -at_support one level up", "endo: parent-span NLL",
                          "ledger: own error on the era's cell")):
        ax_.set_xlabel("cycle"); ax_.set_title(t, fontsize=10); ax_.legend(fontsize=6)
    fig.suptitle(f"{tag}: the shadow panel — every gauge in every arm (error convention)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_panel.png"), dpi=140)
    plt.close(fig)

    # fig 3 — the three clocks
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    n_era = len(setup["eras"])
    for i, a in enumerate(order):
        lg = A[a]["log"]
        axes[0].plot(lg["cycle"], [(c.get("2") or {}).get("recall") or 0
                                   for c in lg["committed_grade"]], lw=1.4,
                     color=cols[i % 10], label=a)
        axes[1].plot(lg["cycle"], [(c.get("3") or {}).get("recall") or 0
                                   for c in lg["committed_grade"]], lw=1.4,
                     color=cols[i % 10], label=a)
        rec = []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            rec.append(np.nan if not idx else
                       (refs["stale"][j] - float(np.mean([lg["e"][q] for q in idx[-3:]])))
                       / (refs["stale"][j] - refs["floor"][j]))
        axes[2].plot(range(1, n_era + 1), rec, "o-", lw=1.4, color=cols[i % 10], label=a)
    for ax_, t in zip(axes, ("committed L2 recall", "committed L3 recall",
                             "recovered fraction per era")):
        ax_.set_title(t, fontsize=10); ax_.legend(fontsize=6)
    axes[0].set_xlabel("cycle"); axes[1].set_xlabel("cycle"); axes[2].set_xlabel("era")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_clocks.png"), dpi=140)
    plt.close(fig)
    print(f"\n[figures] {out}/fig1_competence.png fig2_panel.png fig3_clocks.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ma_s0")
    ap.add_argument("--gf-tag", default="cd_gf")
    # [maestro] the stream-displaced twins live in their OWN tag, not in `ma_s0`: `summary.json`
    # is rebuilt on every invocation, so writing them into `ma_s0` would drop its five originals
    # out of the reduction (`../assay/analyze_assay.py`'s reason, unchanged).
    ap.add_argument("--merge-tag", default="",
                    help="a second tag (e.g. ma_s1) whose arms are merged into this "
                         "reduction after asserting both tags trained the SAME substrate")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--floors", action="store_true",
                    help="print ONLY the in-tag null-ABBA floors (how the endo dead zone is "
                         "measured on the smoke tag before the main run)")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
        try:
            fetch(a.gf_tag)
        except subprocess.CalledProcessError:
            print(f"[warn] no {a.gf_tag} on the volume")
        if a.merge_tag:
            try:
                fetch(a.merge_tag)
            except subprocess.CalledProcessError:
                print(f"[warn] no {a.merge_tag} on the volume")
    root = os.path.join(FIG, a.tag)
    setup, summary = _load(root, "setup.json"), _load(root, "summary.json")
    A = {arm: json.load(open(os.path.join(root, arm, "results.json")))
         for arm in summary["order"]
         if os.path.isfile(os.path.join(root, arm, "results.json"))}
    summary["order"] = [x for x in summary["order"] if x in A]
    # --- merge the stream-displaced twins, asserting substrate identity first ----------- #
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
            added = []
            for arm in m_summary["order"]:
                fp = os.path.join(m_root, arm, "results.json")
                if not os.path.isfile(fp):
                    continue
                A[arm] = json.load(open(fp))
                summary["arms"][arm] = m_summary["arms"][arm]
                summary["cycle_seconds"][arm] = m_summary["cycle_seconds"][arm]
                summary["order"].append(arm)
                added.append(arm)
            print(f"[merge] arms added: {added}")
            rc = (m_setup.get("ref_check") or {})
            if rc:
                print(f"[merge] the twin tag's own in-job substrate assertion vs "
                      f"{rc.get('ref_tag')}: refs_identical = {rc.get('refs_identical')}")
    if a.floors:
        print(f"=== maestro — in-tag floors, tag {a.tag} ===")
        in_tag_floors(a.tag, A, setup, summary)
        return
    gf = _load(os.path.join(FIG, a.gf_tag), "gate.json")
    print(f"=== maestro (A2) — tag {a.tag} ===")
    print(f"ladder {[(e['name'], e.get('cycles')) for e in setup['eras']]} = "
          f"{setup['total_cycles_per_arm']} cycles/arm scheduled; caps "
          f"{setup.get('era_caps')} = {setup.get('cap_total')}; setup "
          f"{setup['t_setup_s']:.0f}s; stale buffer {setup['stale_random_blocks']:.4f} -> "
          f"{setup['stale_task_matched']:.4f}")
    print(f"floors that governed the run: {setup.get('floors')}")
    gates(a.tag, A, setup, summary, gf)
    action_trace(a.tag, A, setup, summary)
    policy_record(a.tag, A, setup, summary)
    gauge_vs_yoke(a.tag, A, setup, summary)
    three_clocks(a.tag, A, setup, summary)
    pi_tables(a.tag, A, setup, summary)
    ledger_table(a.tag, A, setup, summary)
    f = in_tag_floors(a.tag, A, setup, summary)
    shadow_panel(a.tag, A, setup, summary, f)
    battery_and_tail(a.tag, A, setup, summary)
    stream_floor(a.tag, A, setup, summary)
    if a.figures:
        figures(a.tag, A, setup, summary)


if __name__ == "__main__":
    main()
