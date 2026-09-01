"""Reduction for `intonation` (E′ round 2, the delta_perf node) — numbers, not interpretation.

Forked from `../tacet/analyze_tacet.py`; every addition marked `# [intonation]`. A1's, A2's,
A3's and E3′'s sections are unchanged and run first, so the round is readable against its
donors before any delta_perf number is printed. Then:

  (P1) THE METER — per macro slot: the parity trajectory, when the slot first FIRED (at
       `span_tau_fire`) against when it would have opened at tau = 0.95, how many rows it
       executed, its misfire rate, its benchmark b(s), and the run's gate calibration
       (g0/theta from the arm's own active/passive g medians). Plus the ACT/PLAYBACK
       dissection: `b - e` on rows the head actually realized, against `b - e_shadow` on the
       held-out rows the DP realized (the parity read computes the latter for free, on
       identical contexts, with no efference copy) — this substrate's version of Gadagkar's
       playback control, and the check that the centered gate is not decorative.
  (P2) THE 2x2 — executed-as-intended x solved, per arm and per era: counts at the tip and at
       the mined answer, WHAT GOT MINED from each cell, the pi mass that formed, and the era
       table beside them. This is the readout the arc has never had; the `bad_solved` cell is
       the lucky success (`census` finding 5's DP-junk filter from the other side) and
       `int_failed` is the honest failure the grade-only rule discards.
  (P3) THE GAIN COMPARISON — delta_perf vs uniform vs raw-e as a per-sample gain on the head's
       plasticity (S13(b)'s hygiene comparison, on RHM): realised mean weight and credit
       coverage (the matched-budget check), parity and misfire trajectories, the plant guard,
       and the era table.
  (P4) THE GATE ARMS — delta_perf-gated selection against the outcome-delta gate at exactly
       matched volume and against the ungated baseline, in the 2x2's own units: which cells
       each gate kept, the tie fraction, and the tables and pi mass that resulted.
  (P5) THE BENCHMARK TIMESCALE — S13(c) says b(s)'s rate has an interior optimum and nobody has
       measured it here, so the per-cycle per-slot sums are logged and delta is RECOMPUTED
       offline across a ladder of alphas: what fraction of executions delta would have called
       better-than-benchmark, and how the kept set moves. A free sweep off the record.

Usage (from experiments/):
    python3 rhm/practice/intonation/analyze_intonation.py --tag in_s0 --fetch --figures

DONOR DOCSTRING FOLLOWS.

Reduction for `tacet` (E3') — numbers, not interpretation.

Forked from `../crescendo/analyze_crescendo.py`; every addition marked `# [tacet]`. A1's, A2's
and A3's sections are unchanged and run first, so the round is readable against its donors
before any gate-specific number is printed. Then:

  (G) THE GATE — what each arm actually refused. Per arm and per era: how many solved
      trajectories the grade admitted, how many the gate kept, the delta and margin of the kept
      set against the dropped set, the pi replay buffer's realised inflow, and the mining
      channel's volume (which is matched by construction — only its content moves). Plus the
      COUNTERFACTUAL OVERLAP: every arm logs the gate's inputs whether or not it has a gate, so
      what a delta gate WOULD have kept in the ungated baseline, and how far each gate's
      selection sits from a random one, are computed offline for every arm.
  (H) THE PACING COUNTERFACTUAL — the gate arms are clock yokes, so their pacing is the
      baseline's by construction and their CONTENT effect is what section (3) reads. What their
      own pacing would have been is recovered offline: A1's thermostat, replayed unchanged on
      each arm's own logged L4 at-support series (`phase0_l4.py`'s machinery, the same replay
      that sized A3 before any GPU), against what the baseline actually did.

The A3 sections (A)-(F) still run and are still meaningful: (A) reads the L4 table each arm
installed at the baseline's commit cycle, which is readout (i) of this round's question, and
(3)/(4) carry readouts (iii) and (ii). Sections (B), (D), (E) and (9) need arms this tag does
not carry (`ceiling_m3`, `outer_yield_m4x`, the displaced twins) and skip themselves.

Usage (from experiments/):
    python3 rhm/practice/tacet/analyze_tacet.py --tag tc_s0 --fetch --figures

DONOR DOCSTRING FOLLOWS.

Reduction for `crescendo` (A3) — numbers, not interpretation.

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
    from rhm.practice.tacet import tacet as MA          # [tacet]
except Exception:             # pragma: no cover - the reduction must run without modal
    MA = None

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
ASSAY_FIG = os.path.join(os.path.dirname(HERE), "assay", "figures")
# [maestro] A1's fetched copies: the DIRECT donor this fork must replay.
CD_FIG = os.path.join(os.path.dirname(HERE), "conductor", "figures")
# [crescendo] A2's fetched copies: A3's direct donor.
MA_FIG = os.path.join(os.path.dirname(HERE), "maestro", "figures")
# [tacet] A3's fetched copies: THIS fork's direct donor, and the tag this round's ungated arm
# must reproduce over its whole life.
CR_FIG = os.path.join(os.path.dirname(HERE), "crescendo", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_ostinato"                          # [ostinato]

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
        # [tacet] THE ROUND'S STRONGEST GATE IS NOW FULL-LIFE, not windowed. A3 had to bound
        # its cross-tag replay at era 2 because its ladder changed from era 3 on. Nothing about
        # the world moves here — same ladder, same caps, same arm, same seed — so this tag's
        # ungated arm must reproduce `cr3_s0/outer_yield_m4` over EVERY cycle it ran, and
        # `thru_era=None` means exactly that. If it does not, the gate arms have no baseline.
        _refs = (
            ("outer_yield_m4", "outer_yield_m4", "cr3_s0", CR_FIG,
             "A3's TREATMENT, WHOLE LIFE — this round's baseline IS that arm", None),
            ("outer_yield_m4", "outer_yield", "ma_s0", MA_FIG,
             "A1's thermostat through the earning eras — the chain back to A2", 2))
    for arm, ref_arm, ref_tag, ref_root, why, thru_era in _refs or (
            ):
        ap = os.path.join(ref_root, ref_tag, ref_arm, "results.json")
        if os.path.isfile(ap) and arm in A:
            old, new_ = json.load(open(ap))["log"], A[arm]["log"]
            worst, per = 0.0, {}
            # [crescendo] the comparable window: through the last cycle of `thru_era` in BOTH
            # runs. Beyond it the ladders differ by construction (era-3 cap 15 -> 100), so a
            # full-length comparison would report a designed difference as a fidelity failure.
            # [tacet] `thru_era=None` == the whole life of both runs (see `_refs`).
            _last = lambda lg: (len(lg["cycle"]) if thru_era is None else
                                max([i + 1 for i, q in enumerate(lg["era"])
                                     if q <= thru_era] or [0]))
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
                  f"over c1-c{n} ({'whole life' if thru_era is None else f'through era {thru_era}'})")
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
    # [tacet] IN THIS TAG THE TWIN GATE IS NOT A GATE. Every non-baseline arm carries its
    # treatment from cycle 1 (the gate reorders the mining subsample and restricts pi's diet
    # on the very first cycle; `prop_warmup` gates the FILTER, not the training), so a
    # divergence inside the window is the round's variable and not a fidelity failure. The
    # fidelity statements that DO bind in this tag are the full-life cross-tag replay above
    # and the G-F smoke; the number to read below is the first divergence cycle, i.e. when the
    # gate first bound. `preflight`'s T-1 is where a gate-free yoke is asserted bit-identical.
    print(f"\n  IN-TAG TWIN GATE (each arm vs {_anc}, to the first action by EITHER; "
          f"{_anc}'s own first commit is c{a_first}).")
    print("  [tacet] NOTE: every arm here is treated from c1, so a non-zero delta below is the")
    print("  round's variable — read it as WHEN THE GATE FIRST BOUND, not as a failure.")
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
              f"  -> {'identical' if d == 0.0 else 'differs'}")

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
    # [tacet] IN THIS TAG THE YOKES ARE THE TREATMENTS. Every gate arm is a clock yoke of the
    # ungated baseline carrying one extra knob, so a divergence here is not a fidelity failure
    # — it is the round's variable, and `first divergence cycle` reads as WHEN THE GATE FIRST
    # BOUND. `commits`/`advances` equal is the property that still has to hold: it is what
    # makes the arms lifetime- and pacing-identical. (`preflight`'s T-1 is where the yoke's own
    # inertness is asserted, on a gate-free yoke.)
    print("  [tacet] In this tag the yokes ARE the treatments (each carries one gate knob), so")
    print("  a non-zero delta below is the round's variable and `first divergence` reads as")
    print("  WHEN THE GATE FIRST BOUND. What must still hold is commits/advances EQUAL — that")
    print("  is what makes the arms lifetime- and pacing-identical.")
    _yoke_of = {"gate_delta_hi": "outer_yield_m4", "gate_delta_lo": "outer_yield_m4",
                "gate_delib": "outer_yield_m4", "gate_random": "outer_yield_m4",
                "gate_all": "outer_yield_m4", "gate_off_y": "outer_yield_m4",
                "ceiling_m3": "outer_yield_m4", "ceiling_m3_j": "outer_yield_m4_j",
                "yoked_learned": "learned_yield"}
    pairs = []
    for a in summary["order"]:
        lp = A[a].get("loop") or {}
        if lp.get("policy") == "yoke":
            # [maestro] `yoked_learned` yokes onto `learned_yield`, so the name map is no
            # longer a single prefix swap; read the source off the arm table instead.
            # [tacet] with a static fallback map, so the section survives the arm table not
            # being importable (it needs modal, and the reduction must run without it).
            src = (((MA.ARMS.get(a, {}).get("loop") or {}).get("of") if MA is not None else None)
                   or _yoke_of.get(a) or a.replace("yoked_", "outer_"))
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
    # [tacet] this tag carries no schedule arm — the reference every arm is read against is
    # the UNGATED BASELINE, which is also the clock every gate arm replays, so it is
    # lifetime-identical to all of them rather than merely a ceiling on their lifetimes.
    anc = ("anchor_long" if "anchor_long" in rows
           else ("anchor" if "anchor" in rows else summary["order"][0]))
    print(f"\n  DELTA TO {anc.upper()}, per era, in recovered fraction")
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
    ap.add_argument("--tag", default="in_s0")             # [intonation]
    ap.add_argument("--gf-tag", default="in_gf")          # [intonation]
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
        print(f"=== intonation — in-tag floors, tag {a.tag} ===")
        in_tag_floors(a.tag, A, setup, summary)
        return
    gf = _load(os.path.join(FIG, a.gf_tag), "gate.json")
    print(f"=== intonation (E\u2032 round 2, delta_perf) — tag {a.tag} ===")
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
    if CEIL in A:
        signature(a.tag, A, setup, summary, rows)
        pair_window(a.tag, A, setup, summary)
    frontier_gauge(a.tag, A, setup, summary)
    if "outer_yield_m4_j" in A:
        pair_window(a.tag, A, setup, summary, "outer_yield_m4_j", "ceiling_m3_j", header=False)
    extension(a.tag, A, setup, summary, cross)
    # [tacet] the E3' sections, LAST for the same reason A3's were: if the gates above did not
    # pass, nothing below means anything.
    gate_report(a.tag, A, setup, summary)
    table_overlap(a.tag, A, setup, summary)
    pacing_counterfactual(a.tag, A, setup, summary)
    # [intonation] the delta_perf sections, LAST for the same reason: if the gates above did
    # not pass, nothing below means anything.
    perf_meter_report(a.tag, A, setup, summary)
    two_by_two(a.tag, A, setup, summary)
    gain_report(a.tag, A, setup, summary)
    perf_gate_report(a.tag, A, setup, summary)
    benchmark_timescale(a.tag, A, setup, summary)
    perf_pricing(a.tag, A, setup, summary)
    # [ostinato] the sweep's own sections, LAST for the same reason.
    rung_ladder(a.tag, A, setup, summary)
    rung_orderings(a.tag, A, setup, summary)
    if a.figures:
        figures(a.tag, A, setup, summary)
        gate_figures(a.tag, A, setup, summary)
        perf_figures(a.tag, A, setup, summary)



# =========================================================================== #
# [crescendo] THE A3 SECTIONS — the signature, and what paced it
# =========================================================================== #

# [intonation] this tag's baseline is `perf_log` (A3's treatment PLUS the live executor), and
# it carries no ceiling / extension / schedule arm, so A3's sections (B), (D), (E) and (F) skip
# themselves as they were written to. (A) and (C) still read this tag's own L4 commit.
TREAT, CEIL, SCHED, EXT = "os_log", "ceiling_m3", "anchor_long", "outer_yield_m4x"  # [ostinato]


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



# =========================================================================== #
# [tacet] THE E3' SECTIONS — what the gate refused, and what it would have paced
# =========================================================================== #

# [intonation] the ungated baseline is `perf_log`: grade-only selection, live executor,
# delta_perf metered and consumed by nothing.
BASE = "os_log"                                          # [ostinato]


def _gate_rows(res):
    return res["log"].get("gate") or []


def _mode_of(a, A):
    g = _gate_rows(A[a])
    return (g[-1].get("mode") if g else None) or (A[a]["config"] or {}).get("gate_mode")


def _topk(score, k, hi=True):
    """Indices of the k best entries of `score`, ties broken by index — the same rule
    `tacet.gate_order` uses in-run, so an offline counterfactual is the in-run gate."""
    idx = np.arange(score.shape[0])
    key = (-score, idx) if hi else (score, idx)
    return set(idx[np.lexsort(key[::-1])][:k].tolist())


def gate_report(tag, A, setup, summary):
    """(G) THE GATE — what each arm consumed and what it refused."""
    print("\n" + "=" * 78)
    print("(G) THE GATE: what each arm refused, and what the refusal was made of")
    print("=" * 78)
    cfg0 = A[summary["order"][0]]["config"]
    print(f"  gate_frac = {cfg0.get('gate_frac')} of SOLVED TIPS kept on the pi channel; the")
    print(f"  mining cap ({cfg0.get('mine_cap')}) is untouched, so mining volume is matched and")
    print(f"  only its content moves. The value buffer is UNGATED in every arm (delta is the")
    print(f"  residual of the value head's own forecast).")
    n_era = len(setup["eras"])

    print(f"\n  PER-ARM TOTALS over the whole run")
    print(f"  {'arm':16s}{'mode':>10s}{'sol tips':>10s}{'kept':>8s}{'kept %':>8s}"
          f"{'pi rows':>10s}{'mined':>8s}{'pbuf end':>10s}{'fallbacks':>10s}")
    for a in summary["order"]:
        g = _gate_rows(A[a])
        if not g:
            print(f"  {a:16s}{'(no gate log)':>10s}")
            continue
        ns = sum(q["n_sol_tip"] for q in g)
        nk = sum(q["n_keep_tip"] for q in g)
        npair = sum(q.get("n_pairs", 0) for q in g)
        nm = sum(A[a]["log"]["n_mined"])
        print(f"  {a:16s}{str(_mode_of(a, A)):>10s}{ns:>10d}{nk:>8d}"
              f"{(100.0 * nk / ns if ns else 0):>7.1f}%{npair:>10d}{nm:>8d}"
              f"{(g[-1].get('pbuf_n') or 0):>10d}"
              f"{sum(1 for q in g if q.get('fallback')):>10d}")
    print("  `kept` is 0 in an arm with no gate: the gate log records the FEATURES in every arm")
    print("  and the SELECTION only where a mode is set — an ungated arm keeps everything the")
    print("  grade admitted, i.e. `sol tips`.")

    print(f"\n  pi ROWS ENTERING THE REPLAY BUFFER, per era (the diet's volume)")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
    for a in summary["order"]:
        lg, g = A[a]["log"], _gate_rows(A[a])
        row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            row += (f"{sum(g[i].get('n_pairs', 0) for i in idx if i < len(g)):10d}"
                    if idx and g else "-".rjust(10))
        print(f"  {a:16s}{row}")

    print(f"\n  WHAT THE GATE KEPT vs WHAT IT DROPPED (mean over the run, gated arms only)")
    print(f"  {'arm':16s}{'delta kept':>12s}{'delta dropped':>15s}{'gap':>9s}"
          f"{'margin kept':>13s}{'margin all':>12s}")
    for a in summary["order"]:
        g = [q for q in _gate_rows(A[a]) if q.get("d_kept") is not None]
        if not g:
            continue
        dk = float(np.mean([q["d_kept"] for q in g]))
        dd = [q["d_drop"] for q in g if q.get("d_drop") is not None]
        mk = float(np.mean([q["m_kept"] for q in g]))
        ma = float(np.mean([float(np.mean(q["margin"])) for q in g]))
        print(f"  {a:16s}{dk:>12.4f}"
              f"{(float(np.mean(dd)) if dd else float('nan')):>15.4f}"
              f"{(dk - float(np.mean(dd)) if dd else float('nan')):>9.4f}"
              f"{mk:>13.4f}{ma:>12.4f}")

    print(f"\n  THE FEATURES THEMSELVES, in every arm (per-instance, whole run)")
    print("  delta_ans = grade - sigmoid(v) at the answer the beam gave; margin = the")
    print("  deliberation state (top1 - top2 over the instance's tip value scores).")
    print("  `margin` is the GATED form (chosen minus the mean of the unchosen); `m2` is the")
    print("  textbook top1-top2 margin, logged but not gated on — `m2=0 frac` is how often it")
    print("  is exactly zero (a duplicate runner-up), which is why the gate reads the mean form.")
    print(f"  {'arm':16s}{'solve rate':>11s}{'d|solved':>10s}{'d|failed':>10s}"
          f"{'sd d':>8s}{'margin':>9s}{'sd marg':>9s}{'m2=0 frac':>11s}{'corr(d,-m)':>12s}")
    for a in summary["order"]:
        g = _gate_rows(A[a])
        if not g:
            continue
        d = np.concatenate([np.asarray(q["d_ans"], float) for q in g])
        m = np.concatenate([np.asarray(q["margin"], float) for q in g])
        y = np.concatenate([np.asarray(q["ps"], float) for q in g])
        m2 = (np.concatenate([np.asarray(q["m2"], float) for q in g])
              if "m2" in g[0] else None)
        c = (float(np.corrcoef(d, -m)[0, 1]) if d.std() > 0 and m.std() > 0 else float("nan"))
        print(f"  {a:16s}{y.mean():>11.4f}"
              f"{(d[y > 0.5].mean() if (y > 0.5).any() else float('nan')):>10.4f}"
              f"{(d[y < 0.5].mean() if (y < 0.5).any() else float('nan')):>10.4f}"
              f"{d.std():>8.4f}{m.mean():>9.4f}{m.std():>9.4f}"
              f"{(float((m2 == 0).mean()) if m2 is not None else float('nan')):>11.4f}"
              f"{c:>12.4f}")

    print(f"\n  THE COUNTERFACTUAL GATES, per arm, on that arm's OWN logged features")
    print("  Of the solved instances each cycle, take the top half by each rule and report the")
    print("  mean pairwise JACCARD overlap between the rules' selections. This is what a gate")
    print("  arm's mining reorder could at most have moved, computed in every arm including")
    print("  the ungated baseline — a gate whose selection is the same set as another's cannot")
    print("  produce a different table.")
    rules_ = ("delta_hi", "delta_lo", "delib", "random")
    _ab = {"delta_hi": "dhi", "delta_lo": "dlo", "delib": "dlb", "random": "rnd"}
    print(f"  {'arm':16s}" + "".join(f"{_ab[r1] + '/' + _ab[r2]:>12s}"
                                     for i, r1 in enumerate(rules_)
                                     for r2 in rules_[i + 1:]))
    rng = np.random.default_rng(0)
    for a in summary["order"]:
        g = _gate_rows(A[a])
        if not g:
            continue
        acc = {}
        for q in g:
            d = np.asarray(q["d_ans"], float)
            m = np.asarray(q["margin"], float)
            sol = np.flatnonzero(np.asarray(q["ps"], int) > 0)
            if sol.shape[0] < 2:
                continue
            k = max(1, sol.shape[0] // 2)
            sets = {"delta_hi": {sol[i] for i in _topk(d[sol], k, True)},
                    "delta_lo": {sol[i] for i in _topk(d[sol], k, False)},
                    "delib": {sol[i] for i in _topk(m[sol], k, False)},
                    "random": set(rng.permutation(sol)[:k].tolist())}
            for i, r1 in enumerate(rules_):
                for r2 in rules_[i + 1:]:
                    u = len(sets[r1] | sets[r2])
                    acc.setdefault((r1, r2), []).append(
                        len(sets[r1] & sets[r2]) / u if u else 0.0)
        print(f"  {a:16s}" + "".join(
            f"{float(np.mean(acc[(r1, r2)])):12.3f}" if (r1, r2) in acc else "-".rjust(12)
            for i, r1 in enumerate(rules_) for r2 in rules_[i + 1:]))


def _committed_flats(res, level, MC):
    """The flat entries the arm's committed level-`level` table actually holds, reconstructed
    by replaying `Miner.build`'s ratchet test on the LOGGED key stream at the commit cycle over
    the arm's own frozen lower table — Phase 0's gate B-1 machinery, which reproduced 18/18
    logged commit events entry-for-entry. Returns (set, n_expected) so the reconstruction can
    be checked against the event's own `n_entries` in the table below.

    `key_stream(res, 4)` reads the G-Y instrument, which observes in EVERY era; the COMMITTED
    level-4 miner is era-gated and lives in `log["miner"]["4"]`, so level 4 is read from there.
    """
    ev = next((e for e in _commits(res, level)), None)
    if ev is None:
        return None, None
    cyc = int(ev["cycle"])
    lower = flats_of(MC.base_table(8)) if level == 2 else         (_committed_flats(res, level - 1, MC)[0] or set())
    src = res["log"]["miner"]
    i = cyc - 1
    st = (src[i] or {}).get(str(level)) if 0 <= i < len(src) else None
    keys = [tuple(int(x) for x in k) for k in ((st or {}).get("keys_at_support") or [])]
    return set(buildable(keys, lower)), int(ev["n_entries"])


def table_overlap(tag, A, setup, summary):
    """(G2) WHAT THE TABLES CONTAIN, AND HOW MUCH OF IT IS THE SAME TABLE."""
    print("\n" + "=" * 78)
    print("(G2) TABLE CONTENT: reconstruction, truth, and overlap between arms")
    print("=" * 78)
    tf, MC = _truth_flats()
    print("  Each arm's committed table replayed entry-for-entry from its own logged key")
    print("  stream (Phase 0's gate B-1 machinery). `recon == n` is the reconstruction check.")
    got = {}
    for lv in (2, 3, 4):
        print(f"\n  L{lv}  (the DGP has {len(tf[lv])} distinct true tuples)")
        print(f"  {'arm':16s}{'commit c':>10s}{'n':>6s}{'recon':>7s}{'true':>6s}"
              f"{'precision':>11s}{'recall':>9s}")
        for a in summary["order"]:
            fl_, n_ = _committed_flats(A[a], lv, MC)
            if fl_ is None:
                continue
            got[(a, lv)] = fl_
            ev = _commits(A[a], lv)[0]
            nt = len(fl_ & tf[lv])
            print(f"  {a:16s}{ev['cycle']:>10d}{n_:>6d}{len(fl_):>7d}{nt:>6d}"
                  f"{(nt / len(fl_) if fl_ else float('nan')):>11.3f}"
                  f"{(nt / len(tf[lv])):>9.4f}")
        arms_ = [a for a in summary["order"] if (a, lv) in got]
        if len(arms_) < 2:
            continue
        print(f"\n    pairwise JACCARD of the committed L{lv} entry sets")
        print("    " + " " * 16 + "".join(f"{a[:14]:>15s}" for a in arms_))
        for a in arms_:
            row = ""
            for b in arms_:
                u = len(got[(a, lv)] | got[(b, lv)])
                row += f"{(len(got[(a, lv)] & got[(b, lv)]) / u if u else 0.0):15.3f}"
            print(f"    {a:16s}{row}")
    print("\n  A gate that changes what the learner learns from can only change the table")
    print("  through the entries it mined; identical sets here mean the gate moved trust")
    print("  (section 4) and pacing (section H) but not the address book.")


def pacing_counterfactual(tag, A, setup, summary):
    """(H) WHAT THE GATE WOULD HAVE PACED — A1's thermostat replayed on each arm's own series.

    The gate arms are clock yokes, so their realised pacing is the baseline's by construction
    and section (3)'s era table is a CONTENT comparison. This recovers the other half for free:
    `phase0_l4.py::replay_frontier_rule`'s replay, run on each arm's own logged panel from
    era-3 start, says when its own L4 stream would have gone quiet had it been driving. It is
    not a prediction of what a free-paced gate arm would do (an action changes everything
    downstream) — it is the counterfactual first firing, which is what sizes the difference.
    """
    print("\n" + "=" * 78)
    print("(H) THE PACING COUNTERFACTUAL: when each arm's own gauge would have fired")
    print("=" * 78)
    fl = setup.get("floors") or {}
    tol = {3: float(fl.get("yield_L3")), 4: float(fl.get("yield_L4"))}
    print(f"  A1's thermostat, unchanged (span 1, W 4, burn 4, alpha 0.5), floors "
          f"L3 {tol[3]:.6f} / L4 {tol[4]:.6f}, replayed from era-3 start on each arm's own")
    print(f"  logged panel. The BASELINE's realised L4 commit cycle is what every yoked arm")
    print(f"  actually did, so the column to read is `would fire` against it.")
    b_l4 = next((int(e["cycle"]) for e in _commits(A[BASE], 4)), None) if BASE in A else None
    print(f"\n  {'arm':16s}{'era3 c0':>9s}{'n':>5s}{'would fire (c)':>16s}"
          f"{'c_in_era3':>11s}{'vs baseline':>13s}{'all firings (c_in_era3)':>26s}")
    for a in summary["order"]:
        pn = [p for p in (A[a]["log"].get("panel") or []) if p["era"] >= 3]
        if not pn:
            print(f"  {a:16s}{'(no era-3 panel)':>9s}")
            continue
        pol = QuietPolicy("yield", v_tol_by_level=tol, span=1, W=4, burn=4, alpha=0.5)
        pol.acted("era_start", 0, why="era3")
        c0, fires = pn[0]["cycle"], []
        for p in pn:
            info = pol.step(p["cycle"], p)
            if info.get("quiet"):
                fires.append(p["cycle"])
                pol.acted("commit", p["cycle"])
        f0 = fires[0] if fires else None
        print(f"  {a:16s}{c0:>9d}{len(pn):>5d}{str(f0):>16s}"
              f"{str(None if f0 is None else f0 - c0 + 1):>11s}"
              f"{str(None if (f0 is None or b_l4 is None) else f0 - b_l4):>13s}"
              f"{str([c - c0 + 1 for c in fires]):>26s}")
    print(f"\n  the baseline's REALISED L4 commit: c{b_l4}. Every gate arm replayed it by clock,")
    print("  which is what makes section (3)'s era table a content comparison.")


def gate_figures(tag, A, setup, summary):
    """[tacet] three figures: the diet, the features, and the era table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)
    order = summary["order"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a in order:
        g = _gate_rows(A[a])
        if not g:
            continue
        c = [q["c"] for q in g]
        ax[0].plot(c, [q.get("n_pairs", 0) for q in g], lw=1.0, label=a)
        ax[1].plot(c, [float(np.mean(q["d_ans"])) for q in g], lw=1.0, label=a)
        ax[2].plot(c, [float(np.mean(q["margin"])) for q in g], lw=1.0, label=a)
    for k, t in enumerate(("pi rows added per cycle (the diet)",
                           "mean delta_ans per cycle (grade - v at the answer)",
                           "mean margin per cycle (the deliberation state)")):
        ax[k].set_title(t, fontsize=9)
        ax[k].set_xlabel("cycle")
        ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gate_diet.png"), dpi=140)
    plt.close(fig)

    # the era table, per arm
    st, fl_ = setup["refs"]["stale"], setup["refs"]["floor"]
    n_era = len(setup["eras"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for a in order:
        lg = A[a]["log"]
        ys = []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                ys.append(np.nan); continue
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            ys.append((st[j] - e) / (st[j] - fl_[j]))
        ax.plot(range(1, n_era + 1), ys, marker="o", lw=1.4, label=a)
    ax.set_xlabel("era"); ax.set_ylabel("recovered fraction")
    ax.set_title("the value clock per era, per gate arm (all clock-yoked to the baseline)",
                 fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gate_eras.png"), dpi=140)
    plt.close(fig)

    # pi per-level mass
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    for k, lv in enumerate((2, 3, 4)):
        for a in order:
            pr = [q for q in A[a]["log"]["probe"] if q.get("pi", {}).get("macros")]
            if not pr:
                continue
            ax[k].plot([q["cycle"] for q in pr],
                       [sum(m.get("p_macro_all", 0.0) for m in q["pi"]["macros"]
                            if m["level"] == lv) for q in pr], lw=1.2, label=a)
        ax[k].set_title(f"pi proposal mass on L{lv}", fontsize=9)
        ax[k].set_xlabel("cycle"); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gate_pi_mass.png"), dpi=140)
    plt.close(fig)
    print(f"\n[fig] wrote gate_diet.png, gate_eras.png, gate_pi_mass.png to {out}")


# =========================================================================== #
# [intonation] THE delta_perf SECTIONS
# =========================================================================== #

PERF_ARMS = ("perf_log", "perf_gain", "perf_raw", "perf_gate", "outcome_gate")


def _perf_rows(res):
    return res["log"].get("perf") or []


def _is_perf(res):
    return bool((res.get("config") or {}).get("perf_meter")) and bool(_perf_rows(res))


def _slot_totals(res):
    """Per macro slot, summed over the run: rows metered, rows fired, sum e, sum g, sum delta,
    sum (b - e), exact hits. Straight out of the per-cycle sums the runner logged."""
    tot = {}
    for row in _perf_rows(res):
        for k, c in (row.get("slot") or {}).items():
            t = tot.setdefault(k, {q: 0.0 for q in ("n", "n_fire", "sum_e", "sum_g", "sum_d",
                                                    "sum_bme", "n_exact")})
            for q in t:
                t[q] += float(c.get(q, 0) or 0)
    return tot


def _ungated_base(A, summary):
    """The tag's own ungated metered baseline: metered, no plasticity gain, no learning gate.
    Derived from the arms' CONFIGS rather than from a hardcoded name, so a re-run of this node
    in another regime (`ma_s0`'s `mperf_*` arms) reports through the same sections."""
    for a in summary["order"]:
        c = (A.get(a) or {}).get("config") or {}
        if _is_perf(A[a]) and not c.get("perf_gain") and not c.get("gate_mode"):
            return a
    return BASE if BASE in A else (summary["order"][0] if summary["order"] else None)


def _gain_arms(A, summary):
    return [a for a in summary["order"]
            if a in A and _is_perf(A[a]) and (A[a]["config"] or {}).get("perf_gain")]


def _gate_arms(A, summary):
    return [a for a in summary["order"]
            if a in A and _is_perf(A[a]) and (A[a]["config"] or {}).get("gate_mode")]


def _busiest(A, arms):
    """The metered arm that actually executed the most rows — the one whose per-slot and
    per-alpha tables are non-empty. Reading `arms[0]` instead would silently print an empty
    table whenever the first arm in the tag never minted a slot."""
    return max(arms, key=lambda a: sum(int(q.get("n_fired", 0)) for q in _perf_rows(A[a])))


def _era_of_cycle(lg):
    return {int(c): int(e) for c, e in zip(lg["cycle"], lg["era"])}


def perf_meter_report(tag, A, setup, summary):
    """(P1) THE METER. Did the fallible executor actually fire, actually miss, and did the
    benchmark and the agency gate behave? Plus the ACT/PLAYBACK dissection."""
    arms = [a for a in summary["order"] if a in A and _is_perf(A[a])]
    if not arms:
        return
    print("\n" + "=" * 78)
    print("(P1) THE METER — a live, fallible executor, and delta_perf's own components")
    print("=" * 78)
    print("  `tau_fire` is the FIRING threshold; `span_tau` (0.95) still governs the parity")
    print("  RECORD, so `opened_below_tau` counts slot-openings the donor's gate would have")
    print("  refused. `misfire` is a row whose emitted span did not exactly match what")
    print("  `apply_any` would have written on the same observation.\n")
    print(f"  {'arm':14s}{'tau_f':>7s}{'fired':>10s}{'misfire':>9s}{'rate':>8s}"
          f"{'slots':>7s}{'open@f':>8s}{'open@.95':>10s}{'g0':>8s}{'theta':>8s}"
          f"{'calib@c':>9s}{'degen':>7s}{'med g act':>10s}")
    for a in arms:
        r, lg = A[a], A[a]["log"]
        pr = _perf_rows(r)
        fired = sum(int(q.get("n_fired", 0)) for q in pr)
        miss = sum(int(q.get("n_misfire", 0)) for q in pr)
        ge = r.get("gate_events") or []
        below = sum(1 for e in ge if e.get("open") and not e.get("open_tau"))
        last_sp = lg["span"][-1] if lg.get("span") else {}
        par = last_sp.get("parity") or {}
        n_at95 = sum(1 for v in par.values() if v is not None and v >= 0.95)
        ce = (r.get("calib_events") or [{}])[0]
        print(f"  {a:14s}{_fmt((r['config'] or {}).get('span_tau_fire'), 7, 2)}"
              f"{fired:>10d}{miss:>9d}{miss / max(fired, 1):>8.4f}"
              f"{len(par):>7d}{int(last_sp.get('n_open') or 0):>8d}{n_at95:>10d}"
              f"{_fmt(ce.get('g0'), 8, 4)}{_fmt(ce.get('theta'), 8, 4)}"
              f"{str(ce.get('cycle')):>9s}"
              f"{str(ce.get('degenerate')):>7s}{_fmt(ce.get('m_active'), 10, 4)}")

    print("\n  PER MACRO SLOT (the syllable), summed over the run:")
    a0 = _busiest(A, arms)
    print(f"  arm {a0} (the metered arm with the most executions).")
    tot = _slot_totals(A[a0])
    last = _perf_rows(A[a0])[-1]
    print(f"  {'slot':>7s}{'rows':>9s}{'fired':>9s}{'mean e':>9s}{'exact':>8s}"
          f"{'mean g':>9s}{'mean d':>9s}{'b-e':>9s}{'b(s)':>8s}{'parity':>8s}")
    par_last = (A[a0]["log"]["span"][-1].get("parity") or {}) if A[a0]["log"].get("span") else {}
    for k in sorted(tot, key=lambda z: (int(z.split(":")[0]), int(z.split(":")[1]))):
        t = tot[k]
        n = max(t["n"], 1.0)
        print(f"  {k:>7s}{int(t['n']):>9d}{int(t['n_fire']):>9d}"
              f"{t['sum_e'] / n:>9.4f}{t['n_exact'] / n:>8.3f}{t['sum_g'] / n:>9.4f}"
              f"{t['sum_d'] / n:>9.4f}{t['sum_bme'] / n:>9.4f}"
              f"{_fmt((last.get('bench') or {}).get(k), 8, 4)}"
              f"{_fmt(par_last.get(k), 8, 3)}")

    print("\n  ACT vs PLAYBACK (Gadagkar's control, on this substrate).")
    print("  ACT     rows the head REALIZED: e measured against what `apply_any` would have")
    print("          written, agency g > 0 (the slot command explains part of the span).")
    print("  PLAYBK  the SAME head, the SAME contexts, but the DP realized the move: the parity")
    print("          read computes the head's would-be error on every held-out row of every")
    print("          slot each cycle. No efference copy for the head's output, so g == 0 by")
    print("          construction and the CENTERED gate closes. `b - e` is the ungated signal;")
    print("          if it fires in both columns and delta only in ACT, the gate is carrying")
    print("          the discrimination (S13(a)'s dissection).")
    print(f"\n  {'arm':14s}{'e ACT':>9s}{'e PLAYBK':>10s}{'b-e ACT':>10s}"
          f"{'b-e PLAY':>10s}{'delta ACT':>11s}{'delta PLAY':>11s}{'g ACT':>8s}")
    for a in arms:
        r = A[a]
        tot = _slot_totals(r)
        n = max(sum(t["n"] for t in tot.values()), 1.0)
        e_act = sum(t["sum_e"] for t in tot.values()) / n
        bme_act = sum(t["sum_bme"] for t in tot.values()) / n
        d_act = sum(t["sum_d"] for t in tot.values()) / n
        g_act = sum(t["sum_g"] for t in tot.values()) / n
        # the playback column, from the parity series and the benchmark, both already logged
        pe, pbme, npb = 0.0, 0.0, 0
        for row in _perf_rows(r):
            c = int(row["c"])
            sp = next((q for q, cc in zip(r["log"]["span"], r["log"]["cycle"]) if cc == c), None)
            if not sp:
                continue
            for k, blk in (sp.get("parity_block") or {}).items():
                if blk is None:
                    continue
                b = (row.get("bench") or {}).get(k)
                pe += 1.0 - float(blk)
                pbme += ((b - (1.0 - float(blk))) if b is not None else 0.0)
                npb += 1
        ce_ = (r.get("calib_events") or [{}])[0]
        g0, th = ce_.get("g0"), ce_.get("theta")
        gate0 = (1.0 / (1.0 + np.exp(-(0.0 - g0) / max(th, 1e-6)))) \
            if (g0 is not None and th) else None
        print(f"  {a:14s}{e_act:>9.4f}{(pe / max(npb, 1)):>10.4f}{bme_act:>10.4f}"
              f"{(pbme / max(npb, 1)):>10.4f}{d_act:>11.4f}"
              + (f"{(pbme / max(npb, 1)) * gate0:>11.4f}" if gate0 is not None
                 else "-".rjust(11))
              + f"{g_act:>8.4f}")
    print("\n  (PLAYBK's delta column is the ungated `b - e_shadow` multiplied by the gate's")
    print("   value AT g = 0, which is what the mechanism would assign those rows. Playback")
    print("   rows are never scored IN-RUN: the head did not execute them, so no credit is")
    print("   assigned at all — the exclusion is structural, not a threshold.)")
    print("\n  THE AGENCY VARIABLE'S OWN DISTRIBUTION, from the raw per-row sample (ACT rows):")
    print(f"  {'arm':14s}{'n rows':>9s}{'g == 0':>9s}{'g > 0':>9s}{'mean g|g>0':>12s}"
          f"{'e|g==0':>9s}{'e|g>0':>9s}")
    for a in arms:
        rows = [q for row in _perf_rows(A[a]) for q in (row.get("rows") or [])]
        if not rows:
            continue
        g = np.asarray([q[3] for q in rows], float)
        e = np.asarray([q[2] for q in rows], float)
        z = g <= 0
        print(f"  {a:14s}{len(rows):>9d}{z.mean():>9.4f}{(~z).mean():>9.4f}"
              + (f"{g[~z].mean():>12.4f}" if (~z).any() else "-".rjust(12))
              + (f"{e[z].mean():>9.4f}" if z.any() else "-".rjust(9))
              + (f"{e[~z].mean():>9.4f}" if (~z).any() else "-".rjust(9)))


def two_by_two(tag, A, setup, summary):
    """(P2) THE 2x2 — executed-as-intended x solved. The readout this node exists for."""
    arms = [a for a in summary["order"] if a in A and _is_perf(A[a])]
    if not arms:
        return
    print("\n" + "=" * 78)
    print("(P2) THE 2x2: EXECUTED-AS-INTENDED x SOLVED")
    print("=" * 78)
    print("  Per surviving trajectory. `int` = every head execution on it exactly matched its")
    print("  intention; `bad` = at least one did not; `none` = the head executed nothing on it")
    print("  (no performance to grade — kept apart, never folded into `int`).")
    print("  `bad+solved` is THE LUCKY SUCCESS: an ill-executed trajectory the grade rewards.")
    print("  `int+failed` is THE HONEST FAILURE: well executed, task not solved — the cell the")
    print("  grade-only rule discards entirely.\n")
    keys = ("int_solved", "int_failed", "bad_solved", "bad_failed", "non_solved", "non_failed")
    for a in arms:
        lg = A[a]["log"]
        eoc = _era_of_cycle(lg)
        cells = [q["cells"] for q in _perf_rows(A[a]) if q.get("cells")]
        if not cells:
            continue
        n_era = len(setup["eras"])
        print(f"  --- {a} ---")
        print(f"  {'era':>4s}" + "".join(f"{k:>12s}" for k in keys)
              + f"{'exec rows':>11s}{'misfire':>9s}{'d int':>8s}{'d bad':>8s}")
        for j in range(n_era):
            sel = [c for c in cells if eoc.get(int(c["c"])) == j + 1]
            if not sel:
                continue
            row = {k: sum(int(c.get(k, 0)) for c in sel) for k in keys}
            pr = [q for q in _perf_rows(A[a]) if eoc.get(int(q["c"])) == j + 1]
            di = [c["d_int"] for c in sel if c.get("d_int") is not None]
            db = [c["d_bad"] for c in sel if c.get("d_bad") is not None]
            print(f"  {j + 1:>4d}" + "".join(f"{row[k]:>12d}" for k in keys)
                  + f"{sum(int(q.get('n_fired', 0)) for q in pr):>11d}"
                  + f"{sum(int(q.get('n_misfire', 0)) for q in pr):>9d}"
                  + (f"{np.mean(di):>8.4f}" if di else "-".rjust(8))
                  + (f"{np.mean(db):>8.4f}" if db else "-".rjust(8)))
        tot = {k: sum(int(c.get(k, 0)) for c in cells) for k in keys}
        n_int = tot["int_solved"] + tot["int_failed"]
        n_bad = tot["bad_solved"] + tot["bad_failed"]
        print(f"  {'ALL':>4s}" + "".join(f"{tot[k]:>12d}" for k in keys))
        print(f"       P(solve | as intended) = "
              f"{tot['int_solved'] / max(n_int, 1):.4f}   "
              f"P(solve | NOT as intended) = {tot['bad_solved'] / max(n_bad, 1):.4f}   "
              f"lucky-success share of solves = "
              f"{tot['bad_solved'] / max(tot['int_solved'] + tot['bad_solved'], 1):.4f}")

    print("\n  WHAT GOT MINED, BY CELL (at the beam's own answer — the mined unit), per arm:")
    print(f"  {'arm':14s}{'mined':>9s}{'as intended':>13s}{'NOT as int':>12s}"
          f"{'no exec':>9s}{'mean d':>9s}   ans-2x2 (int_s/int_f/bad_s/bad_f/non_s/non_f)")
    for a in arms:
        g = _gate_rows(A[a])
        mi = [q for q in g if q.get("mine_n") is not None]
        if not mi:
            continue
        cells = [q["cells"] for q in _perf_rows(A[a]) if q.get("cells")]
        ans = [sum(int(c.get(k, 0)) for c in cells) for k in
               ("ans_int_solved", "ans_int_failed", "ans_bad_solved", "ans_bad_failed",
                "ans_non_solved", "ans_non_failed")]
        mp = [q["mine_p"] for q in mi if q.get("mine_p") is not None]
        print(f"  {a:14s}{sum(q['mine_n'] for q in mi):>9d}"
              f"{sum(q.get('mine_int', 0) for q in mi):>13d}"
              f"{sum(q.get('mine_bad', 0) for q in mi):>12d}"
              f"{sum(q.get('mine_non', 0) for q in mi):>9d}"
              + (f"{np.mean(mp):>9.4f}" if mp else "-".rjust(9))
              + "   " + "/".join(str(x) for x in ans))


def gain_report(tag, A, setup, summary):
    """(P3) delta_perf vs uniform vs raw-e as a per-sample gain on the head's plasticity."""
    b0 = _ungated_base(A, summary)
    trio = ([b0] if b0 else []) + _gain_arms(A, summary)
    if len(trio) < 2:
        return
    print("\n" + "=" * 78)
    print("(P3) THE PLASTICITY GAIN: delta_perf vs UNIFORM vs RAW-e")
    print("=" * 78)
    print(f"  `{b0}` is the UNGATED control (uniform weights, the donor's own span loss).")
    print("  A `delta` arm weights each stored row by exp(-delta/tau_w); a `raw`/`rawx` arm by")
    print("  the error alone (no benchmark, no agency gate) — S13(b)'s hygiene comparison.")
    print("  All are budget-matched by running-mean normalisation, which `mean w` checks:")
    print("  `raw` (w_raw = e literally) does NOT achieve it on this substrate and `rawx`")
    print("  (w_raw = exp(+e/tau_w), the delta arm's own transform) does.\n")
    print(f"  {'arm':14s}{'mode':>8s}{'mean w':>9s}{'cred frac':>11s}{'fired':>10s}"
          f"{'misfire':>9s}{'rate':>8s}{'end parity L2':>15s}{'L3':>8s}{'L4':>8s}")
    for a in trio:
        r = A[a]
        pr = _perf_rows(r)
        ws = [q["wstat"] for q in pr if q.get("wstat")]
        fired = sum(int(q.get("n_fired", 0)) for q in pr)
        miss = sum(int(q.get("n_misfire", 0)) for q in pr)
        par = (r["log"]["span"][-1].get("parity") or {}) if r["log"].get("span") else {}
        lvl = {}
        for k, v in par.items():
            if v is not None:
                lvl.setdefault(int(k.split(":")[0]), []).append(v)
        print(f"  {a:14s}{str((r['config'] or {}).get('perf_gain')):>8s}"
              + (f"{sum(q['sum_w'] for q in ws) / max(sum(q['n'] for q in ws), 1):>9.4f}"
                 if ws else "-".rjust(9))
              + (f"{sum(q['n_cred'] for q in ws) / max(sum(q['n'] for q in ws), 1):>11.4f}"
                 if ws else "-".rjust(11))
              + f"{fired:>10d}{miss:>9d}{miss / max(fired, 1):>8.4f}"
              + "".join(_fmt(np.mean(lvl[L]) if lvl.get(L) else None,
                             15 if L == 2 else 8, 3) for L in (2, 3, 4)))

    st, fl = setup["refs"]["stale"], setup["refs"]["floor"]
    n_era = len(setup["eras"])
    print(f"\n  ERA TABLE (raw e, last 3 cycles of each era; recovered fraction in brackets)")
    print(f"  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>18s}" for j in range(n_era)))
    for a in trio:
        lg = A[a]["log"]
        out = []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                out.append("-".rjust(18)); continue
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            out.append(f"{e:>10.4f} [{(st[j] - e) / (st[j] - fl[j]):>5.2f}]")
        print(f"  {a:14s}" + "".join(out))
    print(f"  {'stale':14s}" + "".join(f"{st[j]:>18.4f}" for j in range(n_era)))
    print(f"  {'floor':14s}" + "".join(f"{fl[j]:>18.4f}" for j in range(n_era)))

    print("\n  THE PLANT GUARD (the span loss reaches the shared trunk, so it has to be read;")
    print("   `handle/`'s measured interference was -0.094 / -0.044 on parse at 3-9x floor):")
    print(f"  {'arm':14s}{'parse min':>11s}{'parse max':>11s}{'parse last':>12s}"
          f"{'infill min':>12s}{'infill max':>12s}")
    for a in [x for x in summary["order"] if x in A]:
        pb = [q for q in A[a]["log"].get("probe", []) if q and q.get("plant")]
        if not pb:
            continue
        pa = [q["plant"]["parse_acc"] for q in pb]
        inf = [q["plant"]["infill_acc"] for q in pb]
        print(f"  {a:14s}{min(pa):>11.4f}{max(pa):>11.4f}{pa[-1]:>12.4f}"
              f"{min(inf):>12.4f}{max(inf):>12.4f}")


def perf_gate_report(tag, A, setup, summary):
    """(P4) delta_perf-gated selection against the outcome-delta gate and the ungated rule."""
    b0 = _ungated_base(A, summary)
    arms = ([b0] if b0 else []) + _gate_arms(A, summary)
    if len(arms) < 2:
        return
    print("\n" + "=" * 78)
    print("(P4) THE GATE ARMS: delta_perf vs outcome-delta vs ungated, at matched volume")
    print("=" * 78)
    print("  A `perf_*` gate ranks solved trajectories by delta_perf (`perf_hi` = summed;")
    print("  `perf_mean` = executed-first then per-execution mean, the coverage fix); a")
    print("  `delta_hi` gate by delta = grade - v(s) (`tacet`'s rule). Both keep the same")
    print("  COUNT on every cycle, so the only thing that differs is which trajectories.")
    print("  `ties` is on the summed key; `ties(exec)` is on the key `perf_mean` ranks by,")
    print("  i.e. WITHIN the executed set — the number that replaces in_s0's 57%.\n")
    print(f"  {'arm':14s}{'mode':>10s}{'kept tips':>11s}{'of solved':>11s}"
          f"{'kept as-int':>12s}{'ties':>8s}{'exec cand':>11s}{'ties(exec)':>11s}"
          f"{'mined int':>11s}{'mined bad':>11s}")
    for a in arms:
        g = _gate_rows(A[a])
        if not g:
            continue
        print(f"  {a:14s}{str(_mode_of(a, A)):>10s}"
              f"{sum(q.get('n_keep_tip', 0) for q in g):>11d}"
              f"{sum(q.get('n_sol_tip', 0) for q in g):>11d}"
              f"{sum(q.get('int_kept', 0) or 0 for q in g):>12d}"
              f"{sum(q.get('p_tie', 0) or 0 for q in g):>8d}"
              f"{sum(q.get('n_exec_cand', 0) or 0 for q in g):>11d}"
              f"{sum(q.get('p_tie_exec', 0) or 0 for q in g):>11d}"
              f"{sum(q.get('mine_int', 0) or 0 for q in g):>11d}"
              f"{sum(q.get('mine_bad', 0) or 0 for q in g):>11d}")

    st, fl = setup["refs"]["stale"], setup["refs"]["floor"]
    n_era = len(setup["eras"])
    print(f"\n  ERA TABLE (raw e; recovered fraction in brackets), and the delta to the")
    print(f"  ungated baseline `perf_log` in recovered fraction:")
    print(f"  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>18s}" for j in range(n_era)))
    rec = {}
    for a in arms:
        lg = A[a]["log"]
        out, rr = [], []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                out.append("-".rjust(18)); rr.append(None); continue
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            r_ = (st[j] - e) / (st[j] - fl[j])
            rr.append(r_)
            out.append(f"{e:>10.4f} [{r_:>5.2f}]")
        rec[a] = rr
        print(f"  {a:14s}" + "".join(out))
    if b0 in rec:
        print(f"\n  {'arm':14s}" + "".join(f"{'era' + str(j + 1):>10s}" for j in range(n_era)))
        for a in arms:
            if a == b0:
                continue
            print(f"  {a:14s}" + "".join(
                _fmt(None if (rec[a][j] is None or rec[b0][j] is None)
                     else rec[a][j] - rec[b0][j], 10) for j in range(n_era)))
        print("  Read against census finding 7's pooled depth-6 stream floor: 0.087 in the")
        print("  earning family; 0.083 / 0.148 / 0.344 in the given family at eras 3/4/5.")

    print("\n  WHAT EACH ARM'S TABLES AND pi ENDED UP HOLDING (per level, end of run):")
    print(f"  {'arm':14s}{'lvl':>4s}{'entries':>9s}{'recall':>9s}{'prec':>8s}"
          f"{'L4 pi mass':>12s}")
    for a in arms:
        lg = A[a]["log"]
        cg = lg.get("committed_grade") or [{}]
        last = cg[-1] or {}
        pr = [q for q in (lg.get("probe") or []) if q.get("pi", {}).get("macros")]
        pm = (sum(m.get("p_macro_all", 0.0) for m in pr[-1]["pi"]["macros"]
                  if m["level"] == 4) if pr else None)
        for L in ("2", "3", "4"):
            t = last.get(L)
            if not t:
                continue
            print(f"  {a if L == '2' else '':14s}{L:>4s}"
                  f"{_fmt(t.get('n_entries'), 9, 0)}{_fmt(t.get('recall'), 9, 4)}"
                  f"{_fmt(t.get('precision'), 8)}"
                  + (f"{_fmt(pm, 12, 4)}" if L == "4" else "".rjust(12)))


def benchmark_timescale(tag, A, setup, summary):
    """(P5) S13(c)'s interior optimum, read off the record rather than chosen.

    The runner logs, per cycle and per slot, the summed e and the row count. That is exactly
    what a per-slot EWMA benchmark consumes, so b(s) can be re-run offline at any alpha and
    the resulting delta's SIGN STRUCTURE recomputed — which is the quantity the gain and the
    gate both actually read. Magnitudes are not recoverable per row (only per cycle), so this
    is a sign/fraction readout and is labelled as one."""
    _pa = [a for a in summary["order"] if a in A and _is_perf(A[a])]
    a0 = _busiest(A, _pa) if _pa else None
    if a0 is None:
        return
    print("\n" + "=" * 78)
    print("(P5) THE BENCHMARK TIMESCALE — delta recomputed offline across alpha")
    print("=" * 78)
    pr = _perf_rows(A[a0])
    alphas = [0.005, 0.02, 0.05, 0.1, 0.3, 1.0]
    print(f"  arm {a0}; per-cycle per-slot means, {len(pr)} cycles.")
    print(f"  {'alpha':>8s}{'eff. window':>13s}{'frac b>e':>10s}{'mean b-e':>10s}"
          f"{'sd(b-e)':>10s}{'slots':>7s}")
    for al in alphas:
        bench, num, pos, vals = {}, 0, 0, []
        for row in pr:
            for k, c in (row.get("slot") or {}).items():
                n = float(c.get("n", 0) or 0)
                if n <= 0:
                    continue
                e = float(c.get("sum_e", 0.0)) / n
                b = bench.get(k)
                if b is None:
                    bench[k] = e
                    continue
                vals.append(b - e)
                pos += 1 if (b - e) > 0 else 0
                num += 1
                bench[k] = (1 - al) * b + al * e
        if not num:
            continue
        v = np.asarray(vals, float)
        print(f"  {al:>8.3f}{1.0 / al:>13.1f}{pos / num:>10.4f}{v.mean():>10.5f}"
              f"{v.std():>10.5f}{len(bench):>7d}")
    print("  (`eff. window` is 1/alpha in units of the update — here one CYCLE per slot, so")
    print("   the in-run benchmark, which updates per macro CALL, is faster than any row here;")
    print("   the in-run alpha is printed in each arm's config as `perf_alpha`.)")


def perf_pricing(tag, A, setup, summary):
    """(P6) WHAT THE NODE DID NOT PRICE, priced anyway so the choice is auditable.

    `t_misfire` is what a botched span WOULD have cost had misfires been charged one
    materialisation each; it is deliberately not in `t` (the ledger is the arc's cross-tag
    instrument and making it arm-dependent would confound "the gate changed what was learned"
    with "the gate changed what things cost"). `blk_ref` is what the INTENTION REFERENCE
    consumed on the head's firing path — which is why `native/` finding 5's table-ablation
    claim is not made in a metering arm, and why the `blk_head` vs `blk_dp` counterfactual is
    printed here beside it rather than claimed."""
    arms = [a for a in summary["order"] if a in A and _is_perf(A[a])]
    if not arms:
        return
    print("\n" + "=" * 78)
    print("(P6) THE TWO UNPRICED COLUMNS, PRICED — never folded into `t`")
    print("=" * 78)
    print(f"  {'arm':14s}{'n_misfire':>11s}{'t_misfire':>11s}{'t_cum':>12s}{'% of t':>9s}"
          f"{'blk_ref':>12s}{'blk_head':>11s}{'blk_dp':>11s}")
    for a in arms:
        lg = A[a]["log"]
        pr = _perf_rows(A[a])
        nm = sum(int(q.get("n_misfire", 0)) for q in pr)
        tm = sum(float(q.get("t_misfire", 0)) for q in pr)
        br = sum(int(q.get("blk_ref", 0)) for q in pr)
        bh = sum(float(b.get("blk_head", 0)) for b in lg["blocks"])
        bd = sum(float(b.get("blk_dp", 0)) for b in lg["blocks"])
        t = float(lg["t_cum"][-1])
        print(f"  {a:14s}{nm:>11d}{tm:>11.0f}{t:>12.0f}{100 * tm / t:>9.4f}"
              f"{br:>12d}{bh:>11.0f}{bd:>11.0f}")
    print("  Pricing misfires would move the ledger by 0.016-0.032% — below every floor in the")
    print("  tag, so the choice is behaviourally inert and is recorded as a convention, not a")
    print("  result. `blk_ref` exceeds `blk_head` by the span factor: the meter consumes the")
    print("  DP's block evidence that the head's own execution does not.")


def perf_figures(tag, A, setup, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [a for a in summary["order"] if a in A and _is_perf(A[a])]
    if not arms:
        return
    out = os.path.join(FIG, tag)
    keys = ("int_solved", "int_failed", "bad_solved", "bad_failed")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for a in arms:
        pr = _perf_rows(A[a])
        c = [q["c"] for q in pr]
        axes[0].plot(c, [q.get("n_misfire", 0) / max(q.get("n_fired", 0), 1) for q in pr],
                     lw=1, label=a)
        axes[1].plot(c, [np.mean([v for v in (q.get("bench") or {}).values()])
                         if q.get("bench") else np.nan for q in pr], lw=1, label=a)
    axes[0].set_title("misfire rate (executed != intended)"); axes[0].set_xlabel("cycle")
    axes[1].set_title("mean benchmark b(s) over slots"); axes[1].set_xlabel("cycle")
    a0 = _busiest(A, arms)
    cells = [q["cells"] for q in _perf_rows(A[a0]) if q.get("cells")]
    if cells:
        cc = [c["c"] for c in cells]
        bot = np.zeros(len(cells))
        for k in keys:
            v = np.asarray([c.get(k, 0) for c in cells], float)
            axes[2].bar(cc, v, bottom=bot, width=1.0, label=k)
            bot += v
        axes[2].set_title(f"the 2x2 per cycle — {a0}"); axes[2].set_xlabel("cycle")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "perf_meter.png"), dpi=130)
    plt.close(fig)
    print(f"\n[fig] wrote perf_meter.png to {out}")


# =========================================================================== #
# [ostinato] THE SWEEP'S OWN SECTIONS — the rung as an independent variable,
#            measured, and the benchmarked-vs-raw ordering at each rung
# =========================================================================== #

# `census` finding 7's pooled depth-6 stream floors, the arc's own denominators.
EARN_FLOOR = 0.087
GIVEN_FLOOR = {3: 0.083, 4: 0.148, 5: 0.344}


def _rho_of(res):
    return float(((res.get("config") or {}).get("perf_bench_frac", 1.0)))


def _rung_arms(A, summary):
    """Every metered delta-gain arm, ordered by rho descending — the ladder."""
    out = [a for a in summary["order"]
           if a in A and _is_perf(A[a]) and (A[a]["config"] or {}).get("perf_gain") == "delta"]
    return sorted(out, key=lambda a: -_rho_of(A[a]))


def _raw_arm(A, summary):
    for a in summary["order"]:
        if a in A and _is_perf(A[a]) and (A[a]["config"] or {}).get("perf_gain") in ("raw",
                                                                                     "rawx"):
            return a
    return None


def _bench_stats(res, eras_of_cycle=None, era=None):
    """Per-run (or per-era) rollup of the benchmark's OWN evidence, off `n`/`n_bench`/`n_upd`.

    Returns rows, benched rows, updates, and the two derived quantities the sweep is about:
      rows/update            how many performances one b(s) update WOULD have integrated
      benched rows/update    how many it ACTUALLY integrated (the rung)
    plus sd(e) pooled from the per-slot sum_e / sum_e2, so an SE(b) can be formed."""
    n = nb = nu = 0
    se = se2 = 0.0
    for q in _perf_rows(res):
        if era is not None and eras_of_cycle is not None and eras_of_cycle.get(q["c"]) != era:
            continue
        for c in (q.get("slot") or {}).values():
            n += int(c.get("n", 0)); nb += int(c.get("n_bench", 0))
            nu += int(c.get("n_upd", 0))
            se += float(c.get("sum_e", 0.0)); se2 += float(c.get("sum_e2", 0.0))
    mu = se / n if n else float("nan")
    var = (se2 / n - mu * mu) if n else float("nan")
    return {"rows": n, "benched": nb, "updates": nu,
            "rho_realised": (nb / n if n else float("nan")),
            "rows_per_upd": (n / nu if nu else float("nan")),
            "bench_rows_per_upd": (nb / nu if nu else float("nan")),
            "mean_e": mu, "sd_e": (var ** 0.5 if var and var > 0 else float("nan"))}


def rung_ladder(tag, A, setup, summary):
    """(R1) THE INDEPENDENT VARIABLE, MEASURED.

    `perf_bench_frac` (rho) is what the benchmark is ALLOWED to see; what it actually saw is a
    measurement, because a call carries a variable number of rows and at least one row always
    gets through. This section reports the declared rung beside the realised one, the effective
    sample count standing behind one b(s) read, and the resulting noise-to-signal — which is
    the quantity S13(c)'s estimability condition is about."""
    rungs = _rung_arms(A, summary)
    if not rungs:
        return
    alpha = float((A[rungs[0]]["config"] or {}).get("perf_alpha", 0.05))
    # an EWMA at rate alpha averages an effective (2 - alpha)/alpha independent updates
    eff_upd = (2.0 - alpha) / alpha
    print("\n" + "=" * 78)
    print("(R1) THE RUNG, MEASURED — how much repetition the benchmark actually got")
    print("=" * 78)
    print("  rho = `perf_bench_frac`: the fraction of each execution's rows b(s) integrates.")
    print("  Every row still receives a delta and still carries credit; only the BENCHMARK's")
    print("  own evidence is thinned, so total volume, difficulty, the diet and the EWMA's")
    print("  update INSTANTS (hence its staleness) are identical across rungs and only its")
    print(f"  sampling jitter moves. alpha = {alpha} -> an EWMA read averages ~{eff_upd:.0f}")
    print("  updates, so eff_n = (benched rows/update) x that, and SE(b) = sd(e)/sqrt(eff_n).")
    print("  `|b-e|` is the arm's own mean ungated signal (sum_bme / rows) — the thing SE(b)")
    print("  has to be small against for the benchmark to be carrying anything.\n")
    print(f"  {'arm':10s}{'rho':>7s}{'rho real':>9s}{'rows':>11s}{'updates':>9s}"
          f"{'rows/upd':>10s}{'bench/upd':>10s}{'eff n':>9s}{'sd(e)':>8s}"
          f"{'SE(b)':>9s}{'|b-e|':>9s}{'signal/SE':>10s}")
    for a in rungs:
        st = _bench_stats(A[a])
        bme = 0.0
        for q in _perf_rows(A[a]):
            for c in (q.get("slot") or {}).values():
                bme += abs(float(c.get("sum_bme", 0.0)))
        sig = bme / st["rows"] if st["rows"] else float("nan")
        eff = st["bench_rows_per_upd"] * eff_upd
        seb = st["sd_e"] / (eff ** 0.5) if eff and eff == eff else float("nan")
        print(f"  {a:10s}{_rho_of(A[a]):7.3f}{st['rho_realised']:9.4f}{st['rows']:11d}"
              f"{st['updates']:9d}{st['rows_per_upd']:10.1f}{st['bench_rows_per_upd']:10.2f}"
              f"{eff:9.1f}{st['sd_e']:8.4f}{seb:9.5f}{sig:9.5f}"
              f"{(sig / seb if seb else float('nan')):10.2f}")

    eo = _era_of_cycle(A[rungs[0]]["log"])
    n_era = len(setup["eras"])
    print("\n  PER ERA — `bench/upd` (the realised repetition behind one b(s) update). The")
    print("  landmarks this ladder is sized against, from the banked record: `in_s0`")
    print("  (abundance) ran 159.4 rows/call at era 5 and `ma_s0` (scarcity) ran 75.4, i.e.")
    print("  scarcity's OWN thinning was rho_eff = 0.47 — calls/slot/cycle was unchanged")
    print("  (14.85 vs 14.26), so what scarcity removed was performances per update, not")
    print("  recurrences of the context.")
    print(f"  {'arm':10s}" + "".join(f"{'era' + str(j + 1):>12s}" for j in range(n_era)))
    for a in rungs:
        row = ""
        for j in range(n_era):
            st = _bench_stats(A[a], eo, j + 1)
            row += (f"{st['bench_rows_per_upd']:12.2f}" if st["updates"] else "-".rjust(12))
        print(f"  {a:10s}{row}")
    print(f"  {'(unthinned)':10s}" + "".join(
        (lambda st: f"{st['rows_per_upd']:12.1f}" if st["updates"] else "-".rjust(12))(
            _bench_stats(A[rungs[0]], eo, j + 1)) for j in range(n_era)))


def rung_orderings(tag, A, setup, summary):
    """(R2) THE BENCHMARKED-vs-RAW ORDERING AT EACH RUNG.

    The comparison the sweep exists for. `os_rawx` and `os_log` are rung-INVARIANT by
    construction (neither reads b(s)), so one raw line and one uniform line serve the whole
    ladder and every rung is compared to the same two references. Deltas are in recovered
    fraction and are also given as multiples of the measured floors, which is the arc's
    convention for what counts as a difference at all."""
    rungs = _rung_arms(A, summary)
    raw = _raw_arm(A, summary)
    base = _ungated_base(A, summary)
    if not rungs or raw is None:
        return
    st_, fl_ = setup["refs"]["stale"], setup["refs"]["floor"]
    n_era = len(setup["eras"])

    def rec(a):
        lg, out = A[a]["log"], []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            if not idx:
                out.append(None); continue
            e = float(np.mean([lg["e"][i] for i in idx[-3:]]))
            out.append((st_[j] - e) / (st_[j] - fl_[j]))
        return out

    R = {a: rec(a) for a in ([base] if base else []) + [raw] + rungs}
    print("\n" + "=" * 78)
    print("(R2) BENCHMARKED vs RAW, AT EACH RUNG — the ordering, with the floors")
    print("=" * 78)
    print(f"  `{raw}` (raw-e gain, no benchmark, no agency gate) and `{base}` (uniform")
    print("  plasticity) are RUNG-INVARIANT: neither reads b(s), so they are the same arm at")
    print("  every rung. Recovered fraction, last 3 cycles of each era.\n")
    print(f"  {'arm':10s}{'rho':>7s}" + "".join(f"{'era' + str(j + 1):>10s}"
                                                for j in range(n_era)))
    for a in ([base] if base else []) + [raw] + rungs:
        rho = _rho_of(A[a]) if a in rungs else float("nan")
        print(f"  {a:10s}" + (f"{rho:7.3f}" if a in rungs else "-".rjust(7))
              + "".join(_fmt(R[a][j], 10, 4) for j in range(n_era)))

    for ref, name in ((raw, "RAW"), (base, "UNIFORM")):
        if ref is None:
            continue
        print(f"\n  DELTA vs {name} (`{ref}`), in recovered fraction; [x] = multiple of the")
        print(f"  pooled earning-family floor {EARN_FLOOR}. A positive cell is the benchmark")
        print(f"  paying at that rung.")
        print(f"  {'arm':10s}{'rho':>7s}" + "".join(f"{'era' + str(j + 1):>18s}"
                                                    for j in range(n_era)))
        for a in rungs:
            row = ""
            for j in range(n_era):
                if R[a][j] is None or R[ref][j] is None:
                    row += "-".rjust(18); continue
                d = R[a][j] - R[ref][j]
                row += f"{d:>+11.4f} [{abs(d) / EARN_FLOOR:>4.2f}]"
            print(f"  {a:10s}{_rho_of(A[a]):7.3f}{row}")
        print(f"  given-family floors for reference: {GIVEN_FLOOR}")

    # ---- the 2x2, per rung ------------------------------------------------------------ #
    print("\n  THE 2x2 PER RUNG (executed-as-intended x solved, summed over the run). The")
    print("  occupancy the sweep has to keep non-degenerate for any of the above to mean")
    print("  anything: `bad/solved` is the lucky-success cell, `int/failed` the honest one.")
    keys = ("int_solved", "int_failed", "bad_solved", "bad_failed", "non_solved", "non_failed")
    print(f"  {'arm':10s}{'rho':>7s}" + "".join(f"{k:>12s}" for k in keys)
          + f"{'P(sol|int)':>11s}{'P(sol|bad)':>11s}{'ratio':>8s}{'ill/solved':>11s}")
    for a in ([base] if base else []) + [raw] + rungs:
        tot = {k: 0 for k in keys}
        for q in _perf_rows(A[a]):
            c = q.get("cells") or {}
            for k in keys:
                tot[k] += int(c.get(k, 0) or 0)
        pi_ = tot["int_solved"] / max(tot["int_solved"] + tot["int_failed"], 1)
        pb_ = tot["bad_solved"] / max(tot["bad_solved"] + tot["bad_failed"], 1)
        solved = tot["int_solved"] + tot["bad_solved"]
        print(f"  {a:10s}" + (f"{_rho_of(A[a]):7.3f}" if a in rungs else "-".rjust(7))
              + "".join(f"{tot[k]:12d}" for k in keys)
              + f"{pi_:11.4f}{pb_:11.4f}{(pi_ / pb_ if pb_ else float('nan')):8.2f}"
              + f"{(tot['bad_solved'] / max(solved, 1)):11.4f}")

    # ---- trust formation, per rung ---------------------------------------------------- #
    print("\n  TRUST FORMATION per rung: pi's L4 proposal mass and argmax share at the last")
    print("  probe of the run (the arc's trust readout), beside end parity.")
    print(f"  {'arm':10s}{'rho':>7s}{'L4 mass':>10s}{'L4 argmax':>11s}{'L3 mass':>10s}"
          f"{'end par L2':>12s}{'L3':>8s}{'L4':>8s}{'mean w':>9s}")
    for a in ([base] if base else []) + [raw] + rungs:
        pr = [q for q in A[a]["log"]["probe"] if q and q.get("pi", {}).get("macros")]
        mass = {lv: float("nan") for lv in (3, 4)}
        amax = float("nan")
        if pr:
            ms = pr[-1]["pi"]["macros"]
            for lv in (3, 4):
                mass[lv] = sum(m.get("p_macro_all", 0.0) for m in ms if m["level"] == lv)
            amax = sum(m.get("frac_argmax", 0.0) for m in ms if m["level"] == 4)
        par = (A[a]["log"]["span"][-1].get("parity") or {}) if A[a]["log"].get("span") else {}
        lvl = {}
        for k, v in par.items():
            if v is not None:
                lvl.setdefault(int(k.split(":")[0]), []).append(v)
        ws = [q["wstat"] for q in _perf_rows(A[a]) if q.get("wstat")]
        mw = (sum(q["sum_w"] for q in ws) / max(sum(q["n"] for q in ws), 1)) if ws else None
        print(f"  {a:10s}" + (f"{_rho_of(A[a]):7.3f}" if a in rungs else "-".rjust(7))
              + f"{mass[4]:10.4f}{amax:11.4f}{mass[3]:10.4f}"
              + "".join(_fmt(np.mean(lvl[L]) if lvl.get(L) else None,
                             12 if L == 2 else 8, 4) for L in (2, 3, 4))
              + (f"{mw:9.4f}" if mw is not None else "-".rjust(9)))


if __name__ == "__main__":
    main()
