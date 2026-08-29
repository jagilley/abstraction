"""Reduction for `crescendo` (A3) — numbers, not interpretation.

Forked from `../maestro/analyze_maestro.py`; every addition marked `# [crescendo]`. A1's and
A2's sections are unchanged and run first, so the round is readable against its donors before
any A3-specific number is printed. Then:

  (A) THE CROSSING — did an L4 commit occur, and what did it contain? Entries, recall and
      precision against the DGP's own level-4 table, set beside the BUILDABLE CEILING: the
      `T4 subset T3xT3` ratchet means the frozen L3 caps what any L4 table can hold, so the
      honest denominator for an L4 commit is not the level (816 distinct true tuples) but the
      subset of the observation stream whose two halves are both rows of the frozen L3.
  (B) THE SIGNATURE — the value clock at eras 4-5 for the treatment against its ceiling
      control (lifetime-identical by construction) and against the schedule.
  (C) THE GAUGE THAT PACED IT — the L4 at-support series the commit was read on, its
      paired-interval slope against its measured floor, `read_level` per era, and the
      L5/L6 series (logged uncharged in every arm) that were the alternative.
  (D) THE PAIR'S TWIN WINDOW — the treatment and its ceiling control must be bit-identical up
      to the treatment's own L4 commit; after it they differ in one installed table.
  (E) EXTENSION — the r**2-wall bypass: extend events per level, and what they bought.

DONOR DOCSTRING FOLLOWS.

Reduction for `conductor` (A1) — numbers, not interpretation.

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
    python3 rhm/practice/crescendo/analyze_crescendo.py --tag cr3_s0 --fetch --figures
"""

import argparse
import json
import os
import subprocess

import numpy as np

# [maestro] the FORKED policy module (A1's classes byte-for-byte plus the learned one).
from rhm.practice.maestro.policy import QuietPolicy, null_abba, _sd
# [crescendo] Phase 0's offline replay of the substrate's own `Miner.build`, reused here so
# the buildable-L4 ceiling in the reduction is computed by the same code that sized the round
# (and that gate B-1 checked against 18 logged commit events).
from rhm.practice.crescendo.phase0_l4 import buildable, flats_of, key_stream

try:                          # the arm table, for the yoke's source map; optional (needs modal)
    from rhm.practice.crescendo import crescendo as MA
except Exception:             # pragma: no cover - the reduction must run without modal
    MA = None

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ASSAY_FIG = os.path.join(os.path.dirname(HERE), "assay", "figures")
# [maestro] A1's fetched copies: the DIRECT donor this fork must replay.
CD_FIG = os.path.join(os.path.dirname(HERE), "conductor", "figures")
# [crescendo] A2's fetched copies: THIS fork's direct donor.
MA_FIG = os.path.join(os.path.dirname(HERE), "maestro", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_crescendo"

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
        # [crescendo] the arms are RENAMED and the ladder changes from era 3 on, so each
        # reference carries its donor's arm name and the WINDOW over which the two runs are
        # comparable. `outer_yield_m4` vs `ma_s0/outer_yield` is the round's strongest
        # fidelity statement and it is free: A1's era-1 and era-2 CAPS are unchanged here
        # (60, 50) and `max_macro_level=4` touches no shared RNG until `miners[4]` is
        # non-empty, which the mining gate `min(maxl, era.level+1)` defers to era 3. So the
        # treatment must reproduce A2's thermostat BIT FOR BIT through both earning eras.
        _refs = (
            ("outer_yield_m4", "outer_yield", "ma_s0", MA_FIG,
             "A1's THERMOSTAT through the earning eras — the round's strongest gate", 2),
            ("anchor_long", "anchor", "ma_s0", MA_FIG, "the substrate", 1),
            ("anchor_long", "anchor", "cd_s0", CD_FIG, "the chain back to A1", 1))
    for arm, ref_arm, ref_tag, ref_root, why, thru_era in _refs or (
            ):
        ap = os.path.join(ref_root, ref_tag, ref_arm, "results.json")
        if os.path.isfile(ap) and arm in A:
            old, new_ = json.load(open(ap))["log"], A[arm]["log"]
            worst, per = 0.0, {}
            # [crescendo] the comparable window: through the last cycle of `thru_era` in BOTH
            # runs. Beyond it the ladders differ by construction (era-3 cap 15 -> 100), so a
            # full-length comparison would report a designed difference as a fidelity failure.
            _last = lambda lg: max([i + 1 for i, q in enumerate(lg["era"])
                                    if q <= thru_era] or [0])
            n = min(len(old["cycle"]), len(new_["cycle"]), _last(old), _last(new_))
            for k in GF_SERIES:
                x, y = np.asarray(old[k][:n], float), np.asarray(new_[k][:n], float)
                per[k] = float(np.abs(x - y).max()) if n else float("nan")
                worst = max(worst, per[k])
            oc = sorted((int(e["cycle"]), int(e["level"]))
                        for e in json.load(open(ap)).get("events", [])
                        if e.get("kind") == "commit" and int(e["cycle"]) <= n)
            nc = sorted((int(e["cycle"]), int(e["level"]))
                        for e in A[arm].get("events", [])
                        if e.get("kind") == "commit" and int(e["cycle"]) <= n)
            ok = (worst == 0.0) and (oc == nc)
            print(f"  CROSS-TAG REPLAY ({why}): {tag}/{arm} vs {ref_tag}/{ref_arm} "
                  f"over c1-c{n} (through era {thru_era})")
            print(f"    max|delta| = {worst:.3e} over {len(GF_SERIES)} series; "
                  f"commits {nc} vs {oc}  -> {'PASS' if ok else 'FAIL'}")
            if worst:
                print("    per-series: "
                      + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))
        else:
            print(f"  CROSS-TAG REPLAY ({why}): {ref_tag}/{ref_arm} not fetched locally "
                  f"— skipped")

    if gf_smoke:
        ok = gf_smoke["worst_fork"] <= gf_smoke["worst_donor_self_replay"]
        print(f"  G-F smoke scale (in-process vs maestro.py, panel ON, maxl=3): "
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
    # [crescendo] the in-tag schedule arm is `anchor_long` here; the donor's literal "anchor"
    # was a tag constant, not a claim about the gate.
    _anc = "anchor_long" if "anchor_long" in A else ("anchor" if "anchor" in A
                                                    else summary["order"][0])
    a_commits = [e["cycle"] for e in A[_anc]["events"] if e.get("kind") == "commit"]
    a_first = min(a_commits) if a_commits else None
    print(f"\n  IN-TAG TWIN GATE (each arm vs {_anc}, to the first action by EITHER; "
          f"{_anc}'s own first commit is c{a_first}):")
    for t in summary["order"]:
        if t == _anc:
            continue
        fa = _first_action(A[t])
        n_both = min(len(A[t]["log"]["cycle"]), len(A[_anc]["log"]["cycle"]))
        cands = [x for x in (fa, a_first) if x is not None]
        if not cands:
            w, note = n_both, "neither acted"
        else:
            w = min(cands) - 1
            who = "own" if (fa is not None and fa == min(cands)) else "schedule"
            note = f"first action c{min(cands)} ({who})"
        d = 0.0
        for k in TRAJ_SERIES:
            x = np.asarray(A[t]["log"][k][:w], float)
            y = np.asarray(A[_anc]["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        print(f"    {t:14s} c1-c{w:<4d} ({note:16s}) max|delta| = {d:.3e}"
              f"  -> {'PASS' if d == 0.0 else 'FAIL'}")

    print("\n  OBSERVATION PANEL vs the donor's G-Y miner (level 4, every cycle):")
    for a in summary["order"]:
        r = A[a]
        print(f"    {a:16s} agree={r.get('obs_gy_agree')} over {r.get('obs_gy_agree_n')} cycles")

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
    print(f"  {'arm':16s}{'policy':>10s}{'read':>8s}{'cycles':>7s}{'commits':>9s}"
          f"{'advances':>10s}{'chosen':>8s}{'capped':>8s}{'rode cap':>10s}")
    for a in summary["order"]:
        lp = A[a].get("loop") or {}
        print(f"  {a:16s}{str(lp.get('policy')):>10s}{str(lp.get('read')):>8s}"
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
    print(f"  {'arm':16s}{'lvl':>4s}{'cycle':>7s}{'era':>5s}{'prov':>6s}{'entries':>9s}"
          f"{'recall':>8s}{'prec':>8s}{'cert@':>7s}{'driver':>10s}")
    for a in summary["order"]:
        for ev in summary["arms"][a]["commits"]:
            print(f"  {a:16s}{ev['level']:>4d}{ev['cycle']:>7d}{ev['era']:>5d}"
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
    print(f"  {'arm':16s}{'L2 fired':>10s}{'L3 fired':>10s}{'L2 c2cert':>11s}"
          f"{'L3 c2cert':>11s}{'commit L2':>11s}{'commit L3':>11s}")
    for a in summary["order"]:
        sc = summary["arms"][a]["shadow_cert"]
        cm = {e["level"]: e["cycle"] for e in summary["arms"][a]["commits"]}
        print(f"  {a:16s}{str(sc['2'].get('fired')):>10s}"
              f"{str(sc.get('3', {}).get('fired')):>10s}"
              f"{str(sc['2'].get('cycles_to_cert')):>11s}"
              f"{str(sc.get('3', {}).get('cycles_to_cert')):>11s}"
              f"{str(cm.get(2)):>11s}{str(cm.get(3)):>11s}")

    print("\n  CLOCK 2 — COVERAGE: end-of-run committed table per level")
    print(f"  {'arm':16s}{'lvl':>4s}{'frozen n':>10s}{'recall':>9s}{'prec':>8s}"
          f"{'live n':>8s}{'live rec':>10s}{'live prec':>11s}")
    for a in summary["order"]:
        lg = A[a]["log"]
        # [crescendo] level 4 joins the coverage clock — it is the rung the round is about.
        for lv in ("2", "3", "4"):
            cg = next((c.get(lv) for c in reversed(lg["committed_grade"]) if (c or {}).get(lv)),
                      None)
            fz = next((c.get(lv) for c in reversed(lg["vocab"]) if (c or {}).get(lv)), None)
            au = lg["aud"][-1].get(lv) or {}
            print(f"  {a:16s}{lv:>4s}{_fmt(fz, 10)}{_fmt((cg or {}).get('recall'), 9)}"
                  f"{_fmt((cg or {}).get('precision'), 8)}{_fmt(au.get('n_entries'), 8)}"
                  f"{_fmt(au.get('tab_recall'), 10)}{_fmt(au.get('tab_precision'), 11)}")

    print("\n  CLOCK 3 — VALUE: recovered fraction per era (e, and (stale-e)/(stale-floor))")
    st, fl = refs["stale"], refs["floor"]
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>17s}" for j in range(n_era)))
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
        print(f"  {a:16s}{row}")
    print(f"  {'stale':14s}" + "".join(f"{st[j]:>17.3f}" for j in range(n_era)))
    print(f"  {'floor':14s}" + "".join(f"{fl[j]:>17.3f}" for j in range(n_era)))
    # [crescendo] the in-tag schedule arm is `anchor_long` — and, its era lengths being the
    # caps, it is also the LIFETIME CEILING: no loop arm can have run longer, so a positive
    # delta here cannot have been bought with extra practice time (A1's and A2's open caveat).
    anc = "anchor_long" if "anchor_long" in rows else "anchor"
    print(f"\n  DELTA TO {anc.upper()} (the scheduled crank AND the lifetime ceiling), "
          f"per era, in recovered fraction")
    print("  Read against the MEASURED depth-6 stream floor (census finding 7): 0.087 in the")
    print("  earning family and 0.083/0.148/0.344 in the given family for eras 3/4/5. Ranks")
    print("  and signs in the deep eras are the currency; fractions are not.")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
    for a in summary["order"]:
        if a == anc:
            continue
        print(f"  {a:16s}" + "".join(
            _fmt(None if (rows[a][j] is None or rows[anc][j] is None)
                 else rows[a][j] - rows[anc][j], 10)
            for j in range(n_era)))
    return rows


def pi_tables(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(4) pi PER-LEVEL MASS, ARGMAX SHARE, AND MACRO EXPANSIONS, per arm per era")
    print("=" * 78)
    n_era = len(setup["eras"])
    for metric, key in (("proposal mass p_macro_all", "p"),
                        ("argmax share frac_argmax", "argmax")):
        # [crescendo] level 4 joins the table — it is the rung the round is about, and the
        # probe already carries its slots (`decompose_probe` is passed `maxl`).
        for lv in (2, 3, 4):
            print(f"\n  L{lv} {metric} (last probe of each era)")
            print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}"
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
                print(f"  {a:16s}{row}")
    print("\n  MACRO EXPANSIONS the beam ran (blocks.mat_dp, per-cycle mean by era)")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>12s}" for j in range(n_era)))
    for a in summary["order"]:
        lg = A[a]["log"]
        row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            row += (f"{float(np.mean([lg['blocks'][i].get('mat_dp', 0) for i in idx])):12.0f}"
                    if idx else "-".rjust(12))
        print(f"  {a:16s}{row}")


def ledger_table(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(5) THE LEDGER — what each arm read, and what it cost")
    print("=" * 78)
    print("  ear's convention: the learner's own experience is free, a counterfactual read is")
    print("  priced, and instruments logged in every arm are free. Only the POLICY's input")
    print("  stream is charged, and the charge lands in `counts['ground']` -> `t_cum`, which")
    print("  nothing in the cycle loop reads.")
    price = int((setup.get("config") or {}).get("endo_price") or 0)
    print(f"  {'arm':16s}{'read':>12s}{'n reads':>9s}{'priced':>8s}{'spend(g)':>10s}"
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
        print(f"  {a:16s}{str(key):>12s}{sum(n_reads.values()):9d}"
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
    ref = ("anchor_long" if "anchor_long" in A
           else ("anchor" if "anchor" in A else summary["order"][0]))
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
    # [crescendo] A3's signature pair, each against its own displaced twin.
    pairs = [(o, o + "_j") for o in ("outer_yield_m4", "ceiling_m3",
                                     "outer_yield", "learned_yield") if o + "_j" in A]
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
    print(f"  {'arm':16s}{'cond':>9s}{'e(last era)':>13s}{'g/solve':>9s}{'solved':>8s}"
          f"{'T3@sup':>8s}{'T3built':>9s}{'T4@sup':>8s}{'T4built':>9s}")
    for a in summary["order"]:
        ab = A[a].get("ablation") or {}
        for c in conds:
            cell = ab.get(c)
            if not cell or not isinstance(cell, dict) or "eras" not in cell:
                continue
            last = sorted(cell["eras"])[-1]
            r = cell["eras"][last]
            m3 = (r.get("mine") or {}).get("3", {})
            # [crescendo] the T4 currency, beside T3. There is no T5 column because `miners`
            # stop at `max_macro_level`: the level-5 currency this substrate can show is the
            # observation panel's L5 at-support series, printed in section (C).
            m4 = (r.get("mine") or {}).get("4", {})
            print(f"  {a:16s}{c:>9s}{_fmt(r.get('e'), 13)}{_fmt(r.get('g_solve'), 9, 1)}"
                  f"{_fmt(r.get('n_solved'), 8)}{_fmt(m3.get('at_support'), 8)}"
                  f"{_fmt(m3.get('built'), 9)}{_fmt(m4.get('at_support'), 8)}"
                  f"{_fmt(m4.get('built'), 9)}")

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
        print(f"  {a:16s} L3/L4 at support {sup}: {row}")

    print("\n  PLANT GUARD, CERTIFICATES, COST")
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
    ap.add_argument("--tag", default="cr3_s0")
    ap.add_argument("--gf-tag", default="cr3_gf")
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
        print(f"=== crescendo — in-tag floors, tag {a.tag} ===")
        in_tag_floors(a.tag, A, setup, summary)
        return
    gf = _load(os.path.join(FIG, a.gf_tag), "gate.json")
    print(f"=== crescendo (A3) — tag {a.tag} ===")
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
    rows = three_clocks(a.tag, A, setup, summary)
    pi_tables(a.tag, A, setup, summary)
    ledger_table(a.tag, A, setup, summary)
    f = in_tag_floors(a.tag, A, setup, summary)
    shadow_panel(a.tag, A, setup, summary, f)
    battery_and_tail(a.tag, A, setup, summary)
    stream_floor(a.tag, A, setup, summary)
    signature_both_draws(a.tag, A, setup, summary)
    # [crescendo] the A3 sections. They run LAST so the round is readable against its donors
    # first: if the gates above did not pass, nothing below means anything.
    cross = crossing(a.tag, A, setup, summary)
    signature(a.tag, A, setup, summary, rows)
    frontier_gauge(a.tag, A, setup, summary)
    pair_window(a.tag, A, setup, summary)
    if "outer_yield_m4_j" in A:
        pair_window(a.tag, A, setup, summary, "outer_yield_m4_j", "ceiling_m3_j", header=False)
    extension(a.tag, A, setup, summary, cross)
    if a.figures:
        figures(a.tag, A, setup, summary)



# =========================================================================== #
# [crescendo] THE A3 SECTIONS — the signature, and what paced it
# =========================================================================== #

TREAT, CEIL, SCHED, EXT = "outer_yield_m4", "ceiling_m3", "anchor_long", "outer_yield_m4x"


def _commits(res, lv=None):
    return [e for e in res.get("events", [])
            if e.get("kind") == "commit" and (lv is None or int(e["level"]) == lv)]


def _truth_flats():
    """The DGP's own tables, rebuilt offline (pure numpy, no GPU) so the reduction can state
    the level's real size. `setup.json` carries them too, but only up to `max_macro_level`."""
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.ratchet import macros as MC
    rules = generate_rules_distinct(8, 2, 6, 2, seed=0)
    t = MC.true_tables(rules, 6, 2, 8, 2, 4)
    return {l: flats_of(t[l]) for l in (2, 3, 4)}, MC


def crossing(tag, A, setup, summary):
    """(A) THE CROSSING — did the crank reach the new rung, and what did it install?"""
    print("\n" + "=" * 78)
    print("(A) THE CROSSING: the L4 commit, and the r**2 ceiling it was drawn from")
    print("=" * 78)
    tf, MC = _truth_flats()
    sup = int((setup.get("config") or {}).get("mine_support") or 3)
    print(f"  The DGP's levels, as DISTINCT flat tuples (the recall denominator): "
          f"L2 {len(tf[2])}, L3 {len(tf[3])}, L4 {len(tf[4])}.")
    print(f"  `T4 subset T3xT3`: an L4 entry exists only if BOTH of its 4-tuple halves are rows")
    print(f"  of the operative L3 table, so a frozen L3 of recall r caps L4 near r**2.")

    print(f"\n  {'arm':16s}{'L4 commit':>11s}{'c_in_era':>9s}{'entries':>9s}{'recall':>9s}"
          f"{'prec':>8s}{'buildable':>11s}{'of which true':>15s}{'share taken':>13s}")
    out = {}
    for a in summary["order"]:
        res = A[a]
        ev = _commits(res, 4)
        lg = res["log"]
        # the buildable ceiling AT THE COMMIT CYCLE (or at end of run if no commit), computed
        # by replaying `Miner.build` on the logged key streams — Phase 0's gate B-1 machinery.
        k3, k4 = key_stream(res, 3), key_stream(res, 4)
        cyc_at = int(ev[0]["cycle"]) if ev else int(lg["cycle"][-1])
        # the FROZEN L3 the L4 table had to be built over
        c3 = next((int(e["cycle"]) for e in _commits(res, 3)), None)
        base = flats_of(MC.base_table(8))
        k2 = key_stream(res, 2)
        c2 = next((int(e["cycle"]) for e in _commits(res, 2)), None)
        fz2 = set(buildable(k2.get(c2, []), base)) if c2 else set()
        fz3 = set(buildable(k3.get(c3, []), fz2)) if c3 else set()
        bld = buildable(k4.get(cyc_at, []), fz3) if fz3 else []
        n_true = len(set(bld) & tf[4])
        e0 = ev[0] if ev else {}
        share = (None if not bld or not ev else e0["n_entries"] / len(bld))
        print(f"  {a:16s}{str(e0.get('cycle')):>11s}{str(e0.get('c_in_era')):>9s}"
              f"{_fmt(e0.get('n_entries'), 9)}{_fmt(e0.get('tab_recall'), 9, 4)}"
              f"{_fmt(e0.get('tab_precision'), 8)}{len(bld):>11d}{n_true:>15d}"
              f"{_fmt(share, 13)}")
        out[a] = {"commit": e0 or None, "buildable": len(bld), "buildable_true": n_true,
                  "frozen_l3": len(fz3), "at_cycle": cyc_at}
    print(f"\n  `buildable` = distinct L4 tuples at support {sup} in the arm's own observation")
    print("  panel whose two halves are both rows of its FROZEN L3, at the cycle named. It is")
    print("  the honest ceiling for that arm's L4 commit; `share taken` is what it took of it.")
    print("  A cancelled commit (the empty-table guard) appears in the action trace, not here.")
    return out


def signature(tag, A, setup, summary, rows):
    """(B) THE SIGNATURE — the value clock past the range the previous turn certified."""
    n_era = len(setup["eras"])
    print("\n" + "=" * 78)
    print("(B) THE SIGNATURE: does the value clock hold past the certified range?")
    print("=" * 78)
    print("  The pair the signature is read on is TREATMENT vs CEILING CONTROL. They share the")
    print("  substrate, the stream, the head, the miners, the panel, the auditions, every era")
    print("  boundary and every L2/L3 commit cycle. They differ in ONE bit: whether the L4")
    print("  commit installed a table. Lifetime is identical by construction, so no era-4/5")
    print("  difference here can be a difference in practice time.")
    have = [x for x in (TREAT, CEIL, SCHED, EXT) if x in rows]
    print(f"\n  RECOVERED FRACTION per era (higher is better)")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era))
          + f"{'cycles':>9s}")
    for a in have:
        print(f"  {a:16s}" + "".join(_fmt(rows[a][j], 10) for j in range(n_era))
              + f"{len(A[a]['log']['cycle']):>9d}")
    if TREAT in rows and CEIL in rows:
        print(f"\n  THE SIGNATURE CELL: {TREAT} - {CEIL}, per era")
        print(f"  {'':16s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
        d = [None if (rows[TREAT][j] is None or rows[CEIL][j] is None)
             else rows[TREAT][j] - rows[CEIL][j] for j in range(n_era)]
        print(f"  {'delta':16s}" + "".join(_fmt(x, 10) for x in d))
        print("  Against the measured depth-6 stream floor (census finding 7): 0.087 earning-")
        print("  family; 0.083 / 0.148 / 0.344 given-family at eras 3 / 4 / 5.")
        print(f"  {'x floor(e4)':16s}" + _fmt(None if d[3] is None else d[3] / 0.148, 10)
              + f"   {'x floor(e5)':16s}"
              + _fmt(None if d[4] is None else d[4] / 0.344, 10))
        print("  NOTE: eras 4 and 5 are the eras beyond the range A1/A2 could earn. Era 3 is")
        print("  the earning era for L4 itself, so a delta there is the cost or benefit of the")
        print("  commit DURING earning, not the signature.")
    return d if (TREAT in rows and CEIL in rows) else None


def frontier_gauge(tag, A, setup, summary):
    """(C) THE GAUGE THAT PACED THE CROSSING, and the one that was measured degenerate."""
    print("\n" + "=" * 78)
    print("(C) THE GAUGE AT THE FRONTIER: what paced the L4 commit, and the L5 alternative")
    print("=" * 78)
    fl = setup.get("floors") or {}
    print(f"  Driven read: `yield` = -(distinct level-`read_level` tuples at support).")
    print(f"  `read_level` = min(active+1, gy_level=4), so at era 3 (active=4) it is the level")
    print(f"  BEING EARNED, not one above it. Floors that governed the run: "
          f"L3 {fl.get('yield_L3')}, L4 {fl.get('yield_L4')}.")
    print(f"\n  read_level actually used, per era (logged per cycle)")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>9s}"
                                     for j in range(len(setup["eras"]))))
    for a in summary["order"]:
        pn = A[a]["log"].get("panel") or []
        row = ""
        for j in range(len(setup["eras"])):
            lv = sorted({int(p["read_level"]) for p in pn if p["era"] == j + 1})
            row += (",".join(map(str, lv)) or "-").rjust(9)
        print(f"  {a:16s}{row}")

    print("\n  THE OBSERVATION PANEL, end of run: distinct tuples at support, per level.")
    print("  Levels 5 and 6 are logged UNCHARGED in every arm and were never driven — Phase 0")
    print("  measured them degenerate (max 1-2 over an entire run, first non-zero at c89-124),")
    print("  which is the reason the L4 commit is paced on the L4 stream's own quieting.")
    print(f"  {'arm':16s}{'L3':>8s}{'L4':>8s}{'L5':>8s}{'L6':>8s}{'L5 first>0':>12s}")
    for a in summary["order"]:
        oh = A[a].get("obs_hist") or {}
        f5 = next((i + 1 for i, x in enumerate(oh.get("5") or []) if x > 0), None)
        print(f"  {a:16s}" + "".join(_fmt((oh.get(str(l)) or [None])[-1], 8)
                                     for l in (3, 4, 5, 6)) + f"{str(f5):>12s}")

    print("\n  THE L4 AT-SUPPORT SERIES over era 3 (the series the commit was read on):")
    print("  per-era first -> last, and the mean per-cycle increment against the L4 floor.")
    tol4 = float(fl.get("yield_L4") or 0) or None
    print(f"  {'arm':16s}{'era':>5s}{'n':>6s}{'first':>8s}{'last':>8s}{'d/cycle':>10s}"
          f"{'x floor':>9s}")
    for a in summary["order"]:
        lg, oh = A[a]["log"], (A[a].get("obs_hist") or {})
        s4 = oh.get("4") or []
        for j in range(len(setup["eras"])):
            idx = _era_idx(lg, j)
            if not idx or not s4:
                continue
            i0, i1 = idx[0], min(idx[-1], len(s4) - 1)
            if i1 <= i0:
                continue
            rate = (s4[i1] - s4[i0]) / (i1 - i0)
            print(f"  {a:16s}{j + 1:>5d}{len(idx):>6d}{s4[i0]:>8d}{s4[i1]:>8d}"
                  f"{rate:>10.3f}" + _fmt(None if not tol4 else rate / tol4, 9))


def pair_window(tag, A, setup, summary, treat=None, ceil=None, header=True):
    """(D) THE PAIR'S TWIN WINDOW — one bit, and where it starts to matter."""
    treat, ceil = treat or TREAT, ceil or CEIL
    if header:
        print("\n" + "=" * 78)
        print("(D) TREATMENT vs CEILING CONTROL: bit-identity up to the L4 commit")
        print("=" * 78)
    print(f"\n  --- {treat}  vs  {ceil} ---")
    if treat not in A or ceil not in A:
        print("  one of the pair is missing — skipped")
        return
    t, c = A[treat], A[ceil]
    ev = _commits(t, 4)
    cut = int(ev[0]["cycle"]) if ev else None
    lt, lc = t["log"], c["log"]
    n = min(len(lt["cycle"]), len(lc["cycle"]))
    win = (cut - 1) if cut else n
    worst, first_div, per = 0.0, None, {}
    for k in TRAJ_SERIES:
        x, y = np.asarray(lt[k][:n], float), np.asarray(lc[k][:n], float)
        d = np.abs(x - y)
        per[k] = float(d[:win].max()) if win else 0.0
        worst = max(worst, per[k])
        nz = np.nonzero(d)[0]
        if nz.size:
            first_div = nz[0] + 1 if first_div is None else min(first_div, int(nz[0]) + 1)
    ok = worst == 0.0
    print(f"  the treatment's L4 commit: c{cut}. Pre-commit window c1-c{win}.")
    print(f"  max|delta| over {len(TRAJ_SERIES)} trajectory series in the window = "
          f"{worst:.3e}  -> {'PASS' if ok else 'FAIL'}")
    print(f"  first divergence anywhere: c{first_div} "
          f"({'== the commit cycle' if first_div == cut else 'NOT the commit cycle'})")
    ac = lambda r: [(e["era"], e["cycle"]) for e in r["events"] if e["kind"] == "advance"]
    cm = lambda r: [(e["level"], e["cycle"]) for e in _commits(r) if int(e["level"]) <= 3]
    print(f"  era advances replayed exactly: {ac(t) == ac(c)}  {ac(t)}")
    print(f"  L2/L3 commits replayed exactly: {cm(t) == cm(c)}  {cm(t)}")
    print(f"  lifetimes: treatment {len(lt['cycle'])} cycles, ceiling {len(lc['cycle'])} "
          f"— matched = {len(lt['cycle']) == len(lc['cycle'])}")
    if not ok:
        print("  per-series in the window: "
              + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))


def extension(tag, A, setup, summary, cross):
    """(E) THE r**2-WALL BYPASS — what post-commit extension bought at each level."""
    print("\n" + "=" * 78)
    print("(E) EXTENSION: the r**2-wall bypass")
    print("=" * 78)
    print("  Phase 0 measured the wall binds and that extension is the lever: buildable L4 over")
    print("  the FROZEN L3 vs over the LIVE L3 was 15 vs 25 (cd_s0/outer_yield), 13 vs 26")
    print("  (ma_s0/learned_yield), 8 vs 13 (anchor) — i.e. extension roughly doubles the")
    print("  ceiling an L4 commit is drawn from.")
    print(f"\n  {'arm':16s}{'lvl':>5s}{'events':>8s}{'cands':>8s}{'admitted':>10s}"
          f"{'final n':>9s}{'final recall':>14s}{'final prec':>12s}")
    for a in summary["order"]:
        res = A[a]
        exs = [e for e in res.get("events", []) if e.get("kind") == "extend"]
        lg = res["log"]
        for lv in ("2", "3", "4"):
            evs = [e for e in exs if str(e.get("level")) == lv]
            cg = next((c.get(lv) for c in reversed(lg["committed_grade"]) if (c or {}).get(lv)),
                      None)
            fz = next((c.get(lv) for c in reversed(lg["vocab"]) if (c or {}).get(lv)), None)
            if not evs and cg is None:
                continue
            print(f"  {a:16s}{lv:>5s}{len(evs):>8d}"
                  f"{sum(int(e.get('n_candidates') or 0) for e in evs):>8d}"
                  f"{sum(int(e.get('n_admitted') or 0) for e in evs):>10d}"
                  f"{_fmt(fz, 9)}{_fmt((cg or {}).get('recall'), 14, 4)}"
                  f"{_fmt((cg or {}).get('precision'), 12)}")
    print("\n  and what the treatment vs the extension arm actually held at L4:")
    print(f"  {'arm':16s}{'frozen L3':>11s}{'buildable L4':>14s}{'L4 entries':>12s}"
          f"{'L4 true':>9s}")
    for a in (TREAT, EXT, CEIL, SCHED):
        if a not in cross:
            continue
        x = cross[a]
        e0 = x.get("commit") or {}
        print(f"  {a:16s}{x['frozen_l3']:>11d}{x['buildable']:>14d}"
              f"{_fmt(e0.get('n_entries'), 12)}{x['buildable_true']:>9d}")


# --------------------------------------------------------------------------- #
# [crescendo] (F) THE SIGNATURE ON TWO DRAWS — added for `cr3_s1`
# --------------------------------------------------------------------------- #

TREAT_J, CEIL_J = "outer_yield_m4_j", "ceiling_m3_j"


def signature_both_draws(tag, A, setup, summary):
    """(F) The signature cell recomputed on the displaced draw, beside the original.

    The pair is rebuilt on each draw: `ceiling_m3_j` yokes to `outer_yield_m4_j`, not to the
    original treatment, so on BOTH draws the two arms are lifetime-identical, share every era
    boundary and every L2/L3 commit cycle, and differ in one bit. What varies between the two
    columns is the luck, and nothing else.

    Displacement can DEMOTE or SUPPORT an ordering; it can never promote a cell to
    seed-replicated (`as_s1`'s standing caveat — a burn moves stream POSITION, not the seed or
    the rule draw)."""
    if TREAT_J not in A or CEIL_J not in A:
        return
    print("\n" + "=" * 78)
    print("(F) THE SIGNATURE ON TWO DRAWS")
    print("=" * 78)
    st, fl = setup["refs"]["stale"], setup["refs"]["floor"]
    n_era = len(setup["eras"])

    def rec(a, j):
        lg = A[a]["log"]
        idx = _era_idx(lg, j)
        if not idx:
            return None
        e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
        return (st[j] - e) / (st[j] - fl[j])

    draws = [("A (cr3_s0)", TREAT, CEIL), ("B (displaced)", TREAT_J, CEIL_J)]
    print(f"\n  RECOVERED FRACTION per era, per draw")
    print(f"  {'draw':>14s}  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}"
                                                    for j in range(n_era)) + f"{'cycles':>9s}")
    for lbl, tr, ce in draws:
        for a in (tr, ce):
            print(f"  {lbl:>14s}  {a:16s}"
                  + "".join(_fmt(rec(a, j), 10) for j in range(n_era))
                  + f"{len(A[a]['log']['cycle']):>9d}")

    print(f"\n  THE SIGNATURE CELL (treatment - ceiling control), per draw")
    print(f"  {'draw':>14s}  " + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
    cells = {}
    for lbl, tr, ce in draws:
        d = [None if (rec(tr, j) is None or rec(ce, j) is None) else rec(tr, j) - rec(ce, j)
             for j in range(n_era)]
        cells[lbl] = d
        print(f"  {lbl:>14s}  " + "".join(_fmt(x, 10) for x in d))
    a_, b_ = cells["A (cr3_s0)"], cells["B (displaced)"]
    print(f"\n  SIGN AGREEMENT and magnitude, eras 3-{n_era}")
    print(f"  {'era':>4s}{'draw A':>10s}{'draw B':>10s}{'same sign':>11s}"
          f"{'min |d|':>9s}{'x0.087':>8s}{'x given':>9s}")
    given = {3: 0.083, 4: 0.148, 5: 0.344}
    for j in range(2, n_era):
        if a_[j] is None or b_[j] is None:
            continue
        same = (a_[j] > 0) == (b_[j] > 0)
        mn = min(abs(a_[j]), abs(b_[j]))
        g = given.get(j + 1)
        print(f"  {j + 1:>4d}{a_[j]:>10.3f}{b_[j]:>10.3f}{str(same):>11s}"
              f"{mn:>9.3f}{mn / 0.087:>8.2f}" + (f"{mn / g:>9.2f}" if g else "-".rjust(9)))
    print("\n  The conservative reading of a two-draw cell is the SMALLER magnitude, which is")
    print("  what `min |d|` is; the floors are census finding 7's pooled values (earning-family")
    print("  0.087; given-family 0.083/0.148/0.344 at eras 3/4/5), and each arm's OWN measured")
    print("  displacement magnitude is in section (9) beside them — where they disagree the")
    print("  in-node number is the narrower claim.")

    print("\n  DID THE L4 COMMIT RECUR ON THE DISPLACED DRAW?")
    print(f"  {'arm':16s}{'L4 cycle':>10s}{'c_in_era3':>11s}{'entries':>9s}{'recall':>9s}"
          f"{'prec':>8s}{'cycles':>8s}")
    for a in (TREAT, TREAT_J):
        if a not in A:
            continue
        ev = [e for e in A[a]["events"] if e["kind"] == "commit" and e["level"] == 4]
        e0 = ev[0] if ev else {}
        lg = A[a]["log"]
        e3 = [i + 1 for i, q in enumerate(lg["era"]) if q == 3]
        cie = (e0["cycle"] - e3[0] + 1) if (e0 and e3) else None
        print(f"  {a:16s}{str(e0.get('cycle')):>10s}{str(cie):>11s}"
              f"{_fmt(e0.get('n_entries'), 9)}{_fmt(e0.get('tab_recall'), 9, 4)}"
              f"{_fmt(e0.get('tab_precision'), 8)}{len(lg['cycle']):>8d}")


if __name__ == "__main__":
    main()