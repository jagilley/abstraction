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
SPIRAL_FIG = os.path.join(os.path.dirname(HERE), "spiral", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_census"

# the 32 cycles `cs_s0` and `sp_s0` share: era 1 is 48 cycles here and 32 there, and `ec`
# reaches only the loop bound, `at_boundary` (both certified commits precede it) and the
# era-end probe trigger (which fires at c32 either way, and probes consume no RNG).
SHARED_PREFIX = 32
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


def _era_idx(lg, j):
    return [i for i, q in enumerate(lg["era"]) if q == j + 1]


def gates(tag, A, setup, summary, gf_smoke):
    print("\n" + "=" * 78)
    print("(0) GATES")
    print("=" * 78)

    # --- G-F, full scale, cross-tag ---------------------------------------------------- #
    sp = os.path.join(SPIRAL_FIG, "sp_s0", "spiral_route", "results.json")
    if os.path.isfile(sp) and "spiral_route" in A:
        old = json.load(open(sp))["log"]
        new = A["spiral_route"]["log"]
        worst, per = 0.0, {}
        for k in GF_SERIES:
            x = np.asarray(old[k][:SHARED_PREFIX], float)
            y = np.asarray(new[k][:SHARED_PREFIX], float)
            n = min(len(x), len(y))
            d = float(np.abs(x[:n] - y[:n]).max()) if n else float("nan")
            per[k] = d
            worst = max(worst, d)
        print(f"  G-F full scale (cross-tag): cs_s0/spiral_route vs sp_s0/spiral_route over "
              f"c1-c{SHARED_PREFIX}\n    max|delta| = {worst:.3e} over {len(GF_SERIES)} series"
              f"  -> {'PASS' if worst == 0.0 else 'FAIL'}")
        if worst:
            print("    per-series: " + ", ".join(f"{k}={d:.2e}" for k, d in per.items() if d))
    else:
        print("  G-F full scale: sp_s0/spiral_route not fetched locally — skipped")

    if gf_smoke:
        print(f"  G-F smoke scale (in-process vs spiral.py): "
              f"max|fork - spiral| = {gf_smoke['worst_fork']:.3e}, self-replay control = "
              f"{gf_smoke['worst_donor_self_replay']:.3e}, commits equal = "
              f"{gf_smoke['events_equal']}  -> "
              f"{'PASS' if gf_smoke['worst_fork'] <= gf_smoke['worst_donor_self_replay'] else 'FAIL'}")

    # --- twins ------------------------------------------------------------------------- #
    # The pre-treatment window ends at the first cycle at which the treated arm's behaviour
    # COULD differ. For a gate arm that is the cycle the ANCHOR commits and the gate arm does
    # not (holding a commit open is the treatment), not the cycle the gate itself commits.
    print("\n  TWIN GATES (max|de| over the pre-treatment window):")
    anchor = "spiral_route"
    a_c2 = next((e["cycle"] for e in A[anchor]["events"]
                 if e["kind"] == "commit" and e["level"] == 2), None) if anchor in A else None
    for t, why, n in (("census_gate", "anchor commits L2, gate holds", a_c2),
                      ("yoked_delay", "anchor commits L2, yoke holds", a_c2),
                      ("census_extend", "first admitting extension", None)):
        if t not in A or anchor not in A:
            continue
        if n is None:
            n = next((e["cycle"] for e in A[t]["events"]
                      if e["kind"] == "extend" and e.get("n_admitted")), None)
        if not n:
            print(f"    {t:15s}: no divergence point found")
            continue
        w = n - 1
        d = 0.0
        for k in GF_SERIES:
            x = np.asarray(A[t]["log"][k][:w], float)
            y = np.asarray(A[anchor]["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        print(f"    {t:15s} vs {anchor} c1-c{w:<4d} ({why:30s}) max|de| = {d:.3e}"
              f"  -> {'PASS' if d == 0.0 else 'FAIL'}")
    if "given_route" in A and "given_native" in A:
        print("    given_route vs given_native: NO pre-treatment window by construction — both")
        print("      hold the true tables from c1 and given_native's span head trains from c1.")
        print("      The routing-only ceiling is read against sp_s0's given_native cross-tag.")

    # --- the yoked control's own identity check ------------------------------------------ #
    if "yoked_delay" in A and "census_gate" in A:
        d, ev_eq = 0.0, None
        for k in GF_SERIES:
            x = np.asarray(A["yoked_delay"]["log"][k], float)
            y = np.asarray(A["census_gate"]["log"][k], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        ev = lambda a_: [(e["kind"], e.get("level"), e["cycle"]) for e in A[a_]["events"]]
        ev_eq = ev("yoked_delay") == ev("census_gate")
        print(f"\n  YOKED IDENTITY (the dissection): yoked_delay vs census_gate over ALL "
              f"cycles\n    max|delta| = {d:.3e}, event streams identical = {ev_eq}")
        print("    If this is 0.0 the gate's CRITERION contributed nothing beyond choosing the")
        print("    commit cycle: everything separating census_gate from the anchor is delay.")

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
    lo, hi = "spiral_route", "given_route"
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


def timing(tag, A, setup, summary):
    print("\n" + "=" * 78)
    print("(3) THE TIMING PRICE: gate vs yoked vs extend")
    print("=" * 78)
    yk = summary.get("yoke") or {}
    print(f"  yoke case: {yk.get('case')}  cycle {yk.get('cycle')}  "
          f"provisional {yk.get('provisional')}")
    print(f"\n  {'arm':16s}{'L2 commit':>11s}{'prov':>6s}{'quiet':>7s}{'cert L2':>9s}"
          f"{'L3 commit':>11s}{'cert L3':>9s}{'gauge@commit':>14s}")
    for a in summary["order"]:
        sc = summary["arms"][a]["shadow_cert"]
        cs = {c["level"]: c for c in summary["arms"][a]["commits"]}
        c2, c3 = cs.get(2), cs.get(3)
        print(f"  {a:16s}{_fmt((c2 or {}).get('cycle'), 11)}"
              f"{_fmt((c2 or {}).get('provisional'), 6)}"
              f"{_fmt((c2 or {}).get('gate_quiet'), 7)}"
              f"{_fmt(sc['2'].get('fired'), 9)}{_fmt((c3 or {}).get('cycle'), 11)}"
              f"{_fmt(sc.get('3', {}).get('fired'), 9)}"
              f"{_fmt((c2 or {}).get('gauge_at_sup'), 14)}")
    print("\n  WIDTH / g-per-solve by era (the delay's price, on the ledger):")
    n_era = len(setup["eras"])
    print(f"  {'arm':16s}" + "".join(f"{'era' + str(j + 1):>16s}" for j in range(n_era)))
    for a in summary["order"]:
        lg = A[a]["log"]; row = ""
        for j in range(n_era):
            idx = _era_idx(lg, j)
            w = sorted({lg["width"][i] for i in idx})
            g = float(np.mean([lg["g_per_solve"][i] for i in idx]))
            row += f"w{'/'.join(map(str, w))} g{g:.0f}".rjust(16)
        print(f"  {a:16s}{row}")
    print("\n  GAUGE HISTORY at the gate level (L2 at-support per cycle, first 60):")
    for a in summary["order"]:
        gh = (summary["arms"][a].get("gauge_hist") or {}).get("2")
        if gh:
            print(f"    {a:16s} {gh[:60]}")


def grader_bill(tag, A, summary):
    print("\n" + "=" * 78)
    print("(4) THE GRADER BILL: what the extension op paid, and what it bought")
    print("=" * 78)
    print(f"  {'arm':16s}{'extend evts':>12s}{'candidates':>12s}{'auditions':>11s}"
          f"{'admitted':>10s}{'rejected':>10s}{'priced g':>11s}")
    for a in summary["order"]:
        xs = [e for e in A[a]["events"] if e["kind"] == "extend"]
        if not xs:
            continue
        n_aud = A[a]["config"]["n_aud"]
        auds = sum(e["n_auditions"] for e in xs)
        print(f"  {a:16s}{len(xs):12d}{sum(e['n_candidates'] for e in xs):12d}{auds:11d}"
              f"{sum(e['n_admitted'] for e in xs):10d}"
              f"{sum(e['rejected'] for e in xs):10d}{auds * n_aud:11d}")
    print("\n  EXTENSION EVENTS that admitted something:")
    print(f"  {'arm':16s}{'cyc':>5s}{'era':>5s}{'lvl':>4s}{'n->n':>10s}{'e_before':>10s}"
          f"{'e_after':>9s}{'recall':>9s}{'prec':>8s}")
    for a in summary["order"]:
        for e in A[a]["events"]:
            if e["kind"] != "extend" or not e.get("n_admitted"):
                continue
            print(f"  {a:16s}{e['cycle']:5d}{e['era']:5d}{e['level']:4d}"
                  f"{str(e['n_frozen']) + '->' + str(e.get('n_after')):>10s}"
                  f"{_fmt(e.get('e_before'), 10, 4)}{_fmt(e.get('e_after'), 9, 4)}"
                  f"{_fmt(e.get('tab_recall'), 9)}{_fmt(e.get('tab_precision'), 8)}")


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
    ap.add_argument("--tag", default="cs_s0")
    ap.add_argument("--gf-tag", default="gf_smoke")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
        try:
            fetch(a.gf_tag)
        except subprocess.CalledProcessError:
            print(f"[warn] no {a.gf_tag} on the volume")
    root = os.path.join(FIG, a.tag)
    setup, summary = _load(root, "setup.json"), _load(root, "summary.json")
    A = {arm: json.load(open(os.path.join(root, arm, "results.json")))
         for arm in summary["order"]}
    gf = _load(os.path.join(FIG, a.gf_tag), "gate.json")
    print(f"=== census / PHASE 1 — tag {a.tag} ===")
    print(f"ladder {[(e['name'], e.get('cycles')) for e in setup['eras']]} = "
          f"{setup['total_cycles_per_arm']} cycles/arm; setup {setup['t_setup_s']:.0f}s; "
          f"stale buffer {setup['stale_random_blocks']:.4f} -> {setup['stale_task_matched']:.4f}")
    gates(a.tag, A, setup, summary, gf)
    coverage(a.tag, A, setup, summary)
    money(a.tag, A, setup, summary)
    timing(a.tag, A, setup, summary)
    grader_bill(a.tag, A, summary)
    gy_readout(a.tag, A, setup, summary)
    tail(a.tag, A, summary)
    if a.figures:
        figures(a.tag, A, setup, summary)


if __name__ == "__main__":
    main()
