"""Reduction for `spiral`. PHASE A only so far: numbers, not interpretation.

Phase A is a feasibility block, so the reduction is a set of tables rather than a narrative:

  (0) GATES — the pass/fail line for every gate this round carries, read off the artifacts
      rather than re-asserted: `tall`'s T-1..T-4 and the arc's C-M/C-R/G-D (CPU,
      `gates_remote`), `native`'s P-1..P-7 / C-1..C-5 on this substrate (`gates_d6_remote`),
      G-F the donor-fidelity gate at depth 4 (`fidelity_d4`), and the in-tag depth-6 twin.
  (1) THE DESCENT TABLE — per arm: `stale`/`floor`, competence at c1, over the first and last
      five cycles and at its best, each as a RECOVERED FRACTION of the stale->floor range, and
      the pre-commit slope. `tall/dens0`'s reference row is printed beside it: recovered
      fraction 0.262 at last-5 with 0.233 ALREADY PRESENT AT c1, pre-commit slope -0.004/cycle.
      The c1 decomposition is the load-bearing column — it separates what the densified value
      head supplies from what the loop earns.
  (2) THE COMMIT-CYCLE PRICING READOUT — the cycles around each arm's commit, with n_moves,
      k_eff, beam width, groundings per solve and `e`. `dens0` measured +0.180 in `e` at the
      cycle after the L2 commit against a mean cycle-to-cycle |delta| of 0.019 (a 4.6x
      outlier), because the action set grew 32 -> 48 and `fit_width` dropped the beam from
      width 2 to 1 at G=482. Printed here as the jump in units of that arm's own |delta|.
  (3) COST — seconds per cycle per arm, setup and references, so Phase B is sized from measured
      cycle cost rather than from a projection.
  (4) TASK-MATCHED COLLECTION — terminal success of a value buffer collected on the era's own
      damage cell, against the random-block buffer this run actually used.

Usage (from experiments/):
    python3 rhm/practice/spiral/analyze_spiral.py --tag pa0 --fetch
    python3 rhm/practice/spiral/analyze_spiral.py --tag pa0 --fid-tag fid_d4
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_spiral"

# `tall/dens0`'s measured reference, the one Phase A's descent gate is read against.
DENS0 = {"stale": 0.883, "floor": 0.102, "rec_last5": 0.262, "rec_c1": 0.233,
         "slope_pre_commit": -0.004, "commit_jump": 0.180, "delta_cycle_mean": 0.019,
         "delta_cycle_max": 0.039, "buffer_terminal_success": 0.133}
DEPTH4 = {"rec": "0.50-0.80", "buffer_terminal_success": 0.296}


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def _load(tag, name):
    p = os.path.join(FIG, tag, name)
    return json.load(open(p)) if os.path.isfile(p) else None


def descent_table(gate):
    d = gate.get("descent", {})
    if not d:
        return
    print("\n=== (1) THE DESCENT GATE, routing live "
          "(recovered fraction of the stale->floor range) ===")
    hdr = (f"{'arm':30s}{'stale':>8s}{'floor':>8s}{'e_c1':>8s}{'e_l5':>8s}{'e_min':>8s}"
           f"{'rec_c1':>8s}{'rec_l5':>8s}{'rec_bst':>9s}{'slope_pre':>11s}{'n_pre':>6s}"
           f"{'commit':>8s}")
    print(hdr)
    for arm, r in d.items():
        print(f"{arm:30s}{r['stale']:8.3f}{r['floor']:8.3f}{r['e_c1']:8.4f}"
              f"{r['e_last5']:8.4f}{r['e_min']:8.4f}{r['rec_c1']:8.3f}{r['rec_last5']:8.3f}"
              f"{r['rec_best']:9.3f}{r['slope_pre_commit']:+11.5f}{r['n_pre_commit']:6d}"
              f"{str(r.get('commit_cycle')):>8s}")
    print(f"{'dens0 (enum, no ports)':30s}{DENS0['stale']:8.3f}{DENS0['floor']:8.3f}"
          f"{'-':>8s}{'-':>8s}{'-':>8s}{DENS0['rec_c1']:8.3f}{DENS0['rec_last5']:8.3f}"
          f"{'-':>9s}{DENS0['slope_pre_commit']:+11.5f}")
    print(f"{'depth 4 (the target)':30s}{'':8s}{'':8s}{'':8s}{'':8s}{'':8s}"
          f"{'':8s}{DEPTH4['rec']:>8s}")
    print("\n  commits per arm:")
    for arm, r in d.items():
        for c in r.get("commit", []):
            print(f"    {arm:28s} L{c['level']} c{c['cycle']} entries={c['n_entries']} "
                  f"recall={c.get('tab_recall')} prec={c.get('tab_precision')} "
                  f"provisional={c.get('provisional')}")


def pricing_table(gate):
    pr = gate.get("pricing_at_commit", {})
    d = gate.get("descent", {})
    if not pr:
        return
    print("\n=== (2) THE COMMIT-CYCLE PRICING READOUT ===")
    print(f"  dens0: e jumped +{DENS0['commit_jump']:.3f} the cycle after the L2 commit "
          f"(mean |delta| {DENS0['delta_cycle_mean']:.3f}, max {DENS0['delta_cycle_max']:.3f} "
          f"-> a {DENS0['commit_jump'] / DENS0['delta_cycle_mean']:.1f}x outlier), because "
          f"n_moves 32->48 dropped the beam width 2->1 at G=482.")
    for arm, rows in pr.items():
        if not rows:
            continue
        cc = d.get(arm, {}).get("commit_cycle")
        mean_d = d.get(arm, {}).get("delta_cycle_mean", float("nan"))
        print(f"\n  {arm}   (commit at c{cc}; this arm's mean cycle-to-cycle |de| = "
              f"{mean_d:.4f})")
        print(f"    {'cycle':>6s}{'n_moves':>9s}{'k_eff':>7s}{'filter':>8s}{'width':>7s}"
              f"{'g/solve':>9s}{'e':>9s}{'de':>9s}{'de/mean':>9s}")
        prev = None
        for r in rows:
            de = float("nan") if prev is None else r["e"] - prev
            ratio = float("nan") if (prev is None or not mean_d) else de / mean_d
            print(f"    {r['cycle']:6d}{r['n_moves']:9d}{str(r['k_eff']):>7s}"
                  f"{str(r['filter_on']):>8s}{r['width']:7d}{r['g_per_solve']:9.1f}"
                  f"{r['e']:9.4f}{de:+9.4f}{ratio:+9.2f}")
            prev = r["e"]


def gates_table(gate, setup, fid):
    print("\n=== (0) GATES ===")
    f6 = gate.get("fidelity_d6", {})
    if f6.get("series") is not None:
        worst = f6.get("max_abs_delta_behaviour")
        print(f"  in-tag depth-6 twin {f6['arms']} over {f6['cycles']} cycles: "
              f"behaviour max|delta| = {worst:.3e}  -> {'PASS' if worst == 0.0 else 'FAIL'}")
        print(f"    declared root-encode surcharge on g/solve: "
              f"{f6.get('root_encode_surcharge')}")
    if fid:
        print(f"  G-F donor fidelity at depth 4: max|fork - donor| = "
              f"{fid['worst_fork']:.3e}, self-replay control = "
              f"{fid['worst_donor_self_replay']:.3e}, commit events equal = "
              f"{fid['events_equal']}  -> "
              f"{'PASS' if fid['worst_fork'] <= fid['worst_donor_self_replay'] else 'FAIL'}")
    if setup:
        print(f"  read_acc = {setup['read_acc']:.4f}  plant = {setup['plant']}")
        print(f"  stale buffer terminal success = "
              f"{setup['stale_buffer_terminal_success']:.4f}   "
              f"[dens0 n_corrupt=1 {DENS0['buffer_terminal_success']} | "
              f"depth 4 {DEPTH4['buffer_terminal_success']}]")
        pz = setup.get("pricing", {})
        if pz:
            print(f"  pricing: action set {pz['action_set']}")
            print(f"           enum width at G={setup['config']['g_budget']}: "
                  f"{pz['width_enum']}   (to hold w2: {pz['g_to_hold_w2_enum']})")
            print(f"           route width by k: {pz['width_route']}")


def cost_table(gate, setup):
    print("\n=== (3) COST ===")
    if setup:
        print(f"  setup {setup['t_setup_s']:.0f}s   references {setup['t_refs_s']:.0f}s")
    for arm, sec in gate.get("cycle_seconds", {}).items():
        print(f"  {arm:30s} {sec:6.1f} s/cycle")
    t = gate.get("totals", {})
    if t:
        print(f"  TOTAL {t['elapsed_s']:.0f}s = {t['elapsed_s'] / 3600:.2f} GPU-h")
    tm = gate.get("task_matched_collection")
    if tm:
        print("\n=== (4) TASK-MATCHED COLLECTION DAMAGE ===")
        print(f"  terminal success {tm['terminal_success']:.4f} on {tm['n_episodes']} episodes "
              f"at {tm['era']}, against random-block {tm['reference_random_blocks']:.4f} "
              f"[dens0 best {DENS0['buffer_terminal_success']} | "
              f"depth 4 {DEPTH4['buffer_terminal_success']}]  ({tm['seconds']:.0f}s)")


# =========================================================================== #
# PHASE B
# =========================================================================== #

def _era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def _emean(lg, idx, k=3):
    return float(np.mean([lg["e"][i] for i in idx[-k:]]))


def phase_b(tag, arms_json, setup, summary):
    A, refs = arms_json, setup["refs"]
    order = summary["order"]
    st, fl = refs["stale"], refs["floor"]
    n_era = len(setup["eras"])
    print(f"=== spiral / PHASE B — tag {tag} ===")
    print(f"ladder {[(e['name'], e.get('cycles')) for e in setup['eras']]} = "
          f"{setup['total_cycles_per_arm']} cycles/arm; "
          f"{summary.get('elapsed_s', 0) / 3600:.2f} GPU-h; setup {setup['t_setup_s']:.0f}s")
    print(f"read_acc {setup['read_acc']:.4f}; stale buffer random-block "
          f"{setup['stale_random_blocks']:.4f} -> task-matched {setup['stale_task_matched']:.4f}")

    # ---- (0) gates ------------------------------------------------------------------- #
    print("\n=== (0) GATES ===")
    def dmax(a, b, n, key="e"):
        x = np.asarray(A[a]["log"][key], float); y = np.asarray(A[b]["log"][key], float)
        n = min(n, len(x), len(y))
        return float(np.abs(x[:n] - y[:n]).max())
    if "fid" in A and "given" in A:
        d = dmax("fid", "given", 10 ** 6)
        print(f"  in-tag fidelity  fid == given, ALL cycles          max|de| = {d:.3e}  "
              f"-> {'PASS' if d == 0.0 else 'FAIL'}")
    warm = A[order[0]]["config"]["prop_warmup"]
    for t, base, n, why in (("spiral_route", "enum_live", warm, "prop_warmup"),
                            ("spiral", "enum_live", warm, "prop_warmup"),
                            ("enum_live_g722", "enum_live", None, "first commit")):
        if t not in A or base not in A:
            continue
        if n is None:
            n = next((e["cycle"] for e in A[base]["events"] if e["kind"] == "commit"), 1)
        d = dmax(t, base, n)
        print(f"  twin  {t:15s} == {base:14s} c1-c{n:<4d} ({why:12s}) max|de| = {d:.3e}"
              f"  -> {'PASS' if d == 0.0 else 'FAIL'}")
    if "given_native" in A:
        print("  twin  given_native  == given          : NO pre-treatment window by "
              "construction (its span slots exist from c1 because the true table is committed "
              "at construction) — this is precisely why `fid` is carried instead")

    # ---- (1) THE RATE CLAIM ------------------------------------------------------------ #
    print("\n=== (1) THE RATE CLAIM: cycles-to-certification, per level per arm ===")
    print("  `fired` is the absolute cycle; `net` subtracts the cycle the candidate table "
          "first became\n  non-empty, since a certificate cannot fire before the vocabulary "
          "exists.")
    print(f"  {'arm':16s}{'L2 fired':>10s}{'L2 net':>8s}{'L3 fired':>10s}{'L3 net':>8s}"
          f"{'L3-L2 gap':>11s}")
    for a in order:
        sc = summary["arms"][a]["shadow_cert"]
        f2, f3 = sc["2"]["fired"], sc.get("3", {}).get("fired")
        n2, n3 = sc["2"]["cycles_to_cert"], sc.get("3", {}).get("cycles_to_cert")
        gap = (f3 - f2) if (f2 and f3) else None
        print(f"  {a:16s}{str(f2):>10s}{str(n2):>8s}{str(f3):>10s}{str(n3):>8s}"
              f"{str(gap):>11s}")

    # ---- (2) commits ------------------------------------------------------------------- #
    print("\n=== (2) COMMITS: what fired, what it was worth, what it cost ===")
    print(f"  {'arm':16s}{'lvl':>4s}{'cyc':>5s}{'prov':>6s}{'ent':>5s}{'recall':>8s}{'prec':>7s}"
          f"{'n_moves':>12s}{'width':>9s}{'k_eff':>10s}{'G':>6s}")
    for a in order:
        for c in summary["arms"][a]["commits"]:
            print(f"  {a:16s}{c['level']:4d}{c['cycle']:5d}{str(c['provisional']):>6s}"
                  f"{c['n_entries']:5d}{c['tab_recall']:8.3f}{c['tab_precision']:7.3f}"
                  f"{str(c['n_moves_before']) + '->' + str(c['n_moves_after']):>12s}"
                  f"{str(c['width_before']) + '->' + str(c['width_after']):>9s}"
                  f"{str(c['k_eff_before']) + '->' + str(c['k_eff_after']):>10s}"
                  f"{c['g_budget']:6d}")

    # ---- (3) competence ---------------------------------------------------------------- #
    print("\n=== (3) COMPETENCE per era: mean e over the era's last 3 cycles, and the "
          "recovered\n    fraction of that era's own stale->floor range ===")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>17s}" for j in range(n_era)))
    for a in order:
        lg = A[a]["log"]; row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j); e = _emean(lg, idx)
            row += f"{e:.3f} /{(st[j] - e) / (st[j] - fl[j]):+.3f}".rjust(17)
        print(f"  {a:16s}{row}")
    print(f"  {'stale':16s}" + "".join(f"{st[j]:>17.3f}" for j in range(n_era)))
    print(f"  {'floor':16s}" + "".join(f"{fl[j]:>17.3f}" for j in range(n_era)))

    # ---- (4) the wall autopsy ---------------------------------------------------------- #
    print("\n=== (4) WALL AUTOPSY ===")
    print("  DIET — distinct level-shaped tuples at mine_support in the chosen trajectories, "
          "end of era:")
    print(f"  {'arm':16s}" + "".join(f"{'e' + str(j + 1) + '.L2':>8s}{'e' + str(j + 1) + '.L3':>8s}"
                                     for j in range(3)))
    for a in order:
        lg = A[a]["log"]; row = ""
        for j in range(3):
            mn = lg["miner"][_era_idx(lg, j)[-1]]
            for ell in ("2", "3"):
                row += f"{str(mn.get(ell, {}).get('n_at_support', {}).get('3')):>8s}"
        print(f"  {a:16s}{row}")
    print("\n  CORRIDOR — span slots open (min-max per era; 24 slots exist) and best parity:")
    for a in order:
        lg = A[a]["log"]
        if not lg["span"] or not any(c["open"] for c in lg["span"]):
            print(f"  {a:16s} no span head, or no slot ever opened")
            continue
        cells = []
        for j in range(n_era):
            idx = _era_idx(lg, j)
            op = [lg["span"][i]["n_open"] for i in idx]
            par = [x for i in idx for x in lg["span"][i]["parity"].values() if x is not None]
            cells.append(f"e{j + 1}:{min(op)}-{max(op)}"
                         + (f"@{max(par):.3f}" if par else ""))
        print(f"  {a:16s} " + "  ".join(cells))
    print("\n  GRADER — audition calibration at each commit, against the true macro's own "
          "one-action\n  ceiling and against what the frozen unit then scored where it was "
          "CONSUMED (recert):")
    print(f"  {'arm':16s}{'lvl':>4s}{'aud priced':>12s}{'aud oracle':>12s}{'true macro':>12s}"
          f"{'  recert e_frozen, era k+1'}")
    for a in order:
        for c in summary["arms"][a]["commits"]:
            rc = [e for e in A[a]["events"]
                  if e["kind"] == "recert" and e["level"] == c["level"]]
            rcv = (f"{rc[0]['e_frozen']:.4f} -> {rc[-1]['e_frozen']:.4f} (n={len(rc)})"
                   if rc else "-")
            tm = refs["macro_true"][c["era"] - 1][str(c["level"])]
            ap_ = "-" if c["audition"] is None else f"{c['audition']:.4f}"
            print(f"  {a:16s}{c['level']:4d}{ap_:>12s}{c['aud_oracle']:12.4f}{tm:12.4f}  {rcv}")

    # ---- (5) can't-decompose ----------------------------------------------------------- #
    print("\n=== (5) THE CAN'T-DECOMPOSE SIGNATURE (routed arms): pi's top-1 mass and entropy "
          "at\n    the last probe of each era — does the policy sharpen as levels consolidate? ===")
    for a in order:
        pr = [q for q in A[a]["log"]["probe"] if "pi" in q]
        if not pr:
            continue
        cells = []
        for j in range(n_era):
            qs = [q for q in pr if q["era"] == j + 1]
            if qs:
                cells.append(f"e{j + 1}: top1 {qs[-1]['pi']['top1']:.3f} "
                             f"H {qs[-1]['pi']['entropy']:.3f}")
        print(f"  {a:16s} " + "   ".join(cells))

    # ---- (6) cost-to-depth ------------------------------------------------------------- #
    print("\n=== (6) COST-TO-DEPTH on the priced ledger: priced time spent IN each era "
          "(millions)\n    / mean e over that era's last 3 cycles. Eras 3-5 are pure "
          "consumption. ===")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>18s}" for j in range(n_era)))
    for a in order:
        lg = A[a]["log"]; row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            t0 = lg["t_cum"][idx[0] - 1] if idx[0] > 0 else 0.0
            row += f"{(lg['t_cum'][idx[-1]] - t0) / 1e6:7.2f}M /{_emean(lg, idx):.3f}".rjust(18)
        print(f"  {a:16s}{row}")

    # ---- (7) plant guard + cost -------------------------------------------------------- #
    print("\n=== (7) PLANT GUARD (flat is the pass) AND COST ===")
    for a in order:
        pr = A[a]["log"]["probe"]
        pa = [q["plant"]["parse_acc"] for q in pr]
        inf = [q["plant"]["infill_acc"] for q in pr]
        print(f"  {a:16s} parse {min(pa):.3f}-{max(pa):.3f}  infill {min(inf):.3f}-{max(inf):.3f}"
              f"  ({len(pr)} probes)  recerts {summary['arms'][a]['n_recerts']:3d}"
              f"  {summary['cycle_seconds'][a]:6.1f} s/cycle")


def phase_b_extra(tag, A, setup, summary):
    order, n_era = summary["order"], len(setup["eras"])

    print("\n=== (8) WIDTH AND g-PER-SOLVE BY ERA (the priced beam the arm actually ran) ===")
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>16s}" for j in range(n_era)))
    for a in order:
        lg = A[a]["log"]; row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            w = sorted({lg["width"][i] for i in idx})
            g = float(np.mean([lg["g_per_solve"][i] for i in idx]))
            row += f"w{'/'.join(map(str, w))} g{g:.0f}".rjust(16)
        print(f"  {a:16s}{row}")

    print("\n=== (9) RECERT: the frozen unit graded where it is CONSUMED, vs what the LIVE "
          "table\n    would score there. n_diff = answers the swap would actually change. ===")
    print(f"  {'arm':16s}{'lvl':>4s}{'cyc':>5s}{'e_frozen':>10s}{'e_live':>9s}"
          f"{'n_frozen':>10s}{'n_live':>8s}{'n_diff':>8s}{'live rec/prec':>15s}")
    for a in order:
        for e in A[a]["events"]:
            if e["kind"] != "recert":
                continue
            lr = (f"{e['live_recall']:.2f}/{e['live_precision']:.2f}"
                  if "live_recall" in e else "-")
            el = "-" if e.get("e_live") is None else f"{e['e_live']:9.4f}"
            print(f"  {a:16s}{e['level']:4d}{e['cycle']:5d}{e['e_frozen']:10.4f}{el:>9s}"
                  f"{e['n_frozen']:10d}{e['n_live']:8d}{str(e.get('n_diff')):>8s}{lr:>15s}")

    print("\n=== (10) ABLATION BATTERY (final trained state, per era; e / groundings-per-solve;"
          "\n     `at_sup` = distinct level-1 tuples at support in THAT condition's own chosen"
          "\n     trajectories — the informative half of next-level currency) ===")
    for a in order:
        ab = A[a].get("ablation")
        if not ab:
            print(f"  {a:16s} no battery")
            continue
        print(f"  {a}:")
        for cond in ("a_full", "b_table", "b_span", "c_prims"):
            cell = ab.get(cond)
            if cell is None:
                print(f"    {cond:9s} (not applicable)")
                continue
            if "skipped" in cell:
                print(f"    {cond:9s} {cell['skipped']}")
                continue
            bits = []
            for j in range(n_era):
                r = cell["eras"].get(str(j + 1 - 1)) or cell["eras"].get(str(j))
                if r:
                    bits.append(f"e{j + 1}:{r['e']:.3f}/g{r['g_solve']:.0f}"
                                f"/T3@{r['mine'].get('3', {}).get('at_support')}")
            print(f"    {cond:9s} nm={cell['n_moves']:3d}  " + "  ".join(bits))


def figures(tag, A, setup, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order, refs = summary["order"], setup["refs"]
    n_era = len(setup["eras"])
    bounds, c = [], 0
    for e in setup["eras"]:
        c += int(e.get("cycles")); bounds.append(c)
    out = os.path.join(FIG, tag)
    os.makedirs(out, exist_ok=True)

    # fig1 — competence trajectory, all arms, with era boundaries and the two references
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
    ax.set_xlabel("cycle"); ax.set_ylabel("e (1 - terminal success)")
    ax.set_title(f"{tag}: competence; dotted = stale, dashed = exact-DP floor, "
                 f"triangles = commits")
    ax.legend(fontsize=7, ncol=4); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=140); plt.close(fig)

    # fig2 — the rate claim: cycles-to-cert per level per arm
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(order))
    for k, (ell, off) in enumerate((("2", -0.18), ("3", 0.18))):
        vals = [summary["arms"][a]["shadow_cert"].get(ell, {}).get("cycles_to_cert") or 0
                for a in order]
        ax.bar(xs + off, vals, 0.34, label=f"L{ell}")
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("cycles to certification\n(net of first candidate)")
    ax.set_title(f"{tag}: the rate claim"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_rate.png"), dpi=140); plt.close(fig)

    # fig3 — the walls: corridor parity (n_open) and the diet stream
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for a in order:
        lg = A[a]["log"]
        if lg["span"] and any(c["open"] for c in lg["span"]):
            axes[0].plot(lg["cycle"], [c["n_open"] for c in lg["span"]], lw=1.4, label=a)
        stream = [q.get("3", {}).get("n_at_support", {}).get("3", 0) for q in lg["miner"]]
        axes[1].plot(lg["cycle"], stream, lw=1.4, label=a)
    for ax_, t in zip(axes, ("corridor: span slots open (of 24)",
                             "diet: distinct L3-shaped tuples at support")):
        for hi in bounds:
            ax_.axvline(hi + 0.5, color="0.85", lw=1)
        ax_.set_xlabel("cycle"); ax_.set_title(t); ax_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_walls.png"), dpi=140)
    plt.close(fig)
    print(f"\n[figures] {out}/fig1_competence.png fig2_rate.png fig3_walls.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pa0")
    ap.add_argument("--fid-tag", default="fid_d4")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
        try:
            fetch(a.fid_tag)
        except subprocess.CalledProcessError:
            print(f"[warn] no {a.fid_tag} on the volume")
    gate = _load(a.tag, "gate.json") or {}
    setup = _load(a.tag, "setup.json")
    fid = _load(a.fid_tag, "gate.json")
    summary = _load(a.tag, "summary.json")
    if summary is not None:                      # a Phase-B tag
        root = os.path.join(FIG, a.tag)
        arms_json = {arm: json.load(open(os.path.join(root, arm, "results.json")))
                     for arm in summary["order"]}
        if fid:
            print(f"G-F donor fidelity at depth 4 ({a.fid_tag}): "
                  f"max|fork - donor| = {fid['worst_fork']:.3e}, self-replay control = "
                  f"{fid['worst_donor_self_replay']:.3e}  -> "
                  f"{'PASS' if fid['worst_fork'] <= fid['worst_donor_self_replay'] else 'FAIL'}")
        phase_b(a.tag, arms_json, setup, summary)
        phase_b_extra(a.tag, arms_json, setup, summary)
        if a.figures:
            figures(a.tag, arms_json, setup, summary)
        return
    print(f"=== spiral / PHASE A — tag {a.tag} ===")
    gates_table(gate, setup, fid)
    descent_table(gate)
    pricing_table(gate)
    cost_table(gate, setup)


if __name__ == "__main__":
    main()
