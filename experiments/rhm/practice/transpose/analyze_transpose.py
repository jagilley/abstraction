"""Reduce a `transpose` run.

    python3 rhm/practice/transpose/analyze_transpose.py --tag tp_s0 --fetch --figures
    python3 rhm/practice/transpose/analyze_transpose.py --tag tf_s0 --fetch --fidelity

Readouts, in the order the round asks for them:
  0. FIDELITY -- with drift off this fork IS `recital` in its `ear` mode; `tf_s0` must
     reproduce `ear/er_s0` cycle for cycle on all eight of `ear`'s arms.
  1. THE WORLD -- how far the piece was actually transposed: per-epoch vocabulary survival,
     corpus survival, and the moving references (`stale`, `floor`, the true macro's ceiling).
     Every later readout is normalised against these, because a moving world moves the
     denominators too.
  2. THE COMMIT-POLICY GRADE -- per era, and on the terminal all-eras exam: raw error,
     recovered fraction against that epoch's own `stale`/`floor`, and the earned-vs-given
     fraction against BOTH forks of `given` (frozen initial tables, live tracking tables).
  3. POST-COMMIT UNIT ERROR -- the round's structurally new series. `held` is the committed
     macro's own audition; in every prior round its trajectory is exactly flat because nothing
     could invalidate it. Reported as slope and total rise from the commit, beside the
     TRACKING macro (`true`) on the same set, which is the same series with the news bought.
  4. STALENESS -- committed-table precision/recall against the CURRENT truth over time, and
     against the truth it was mined in.
  5. RECERT -- firings, swaps, and the frozen-vs-live margin. Baseline: 0 swaps in 24 (`ear`)
     and 0 in 288 (`recital`).
  6. CHURN -- entries added/removed from the live mined table, and how many tuples the
     decaying miner forgot. `tall` retired this question a priori under a static grammar
     (the miner is monotone by construction); here it is a measurement again.
  7. INSTRUMENTS -- plant guard, reader, bank freshness, mined-vs-random, pricing/width.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
EAR_FIG = os.path.join(os.path.dirname(HERE), "ear", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_transpose"
ORDER = ["never_base", "given", "given_live", "practice_gated", "practice_prov",
         "prov_norecert", "practice_climb", "climb_decay", "gated_decay"]


def fetch(tag):
    os.makedirs(FIG, exist_ok=True)
    subprocess.run(["modal", "volume", "get", "--force", VOLUME, f"{REMOTE}/{tag}", FIG],
                   check=True)


def load(tag):
    root = os.path.join(FIG, tag)
    setup = json.load(open(os.path.join(root, "setup.json")))
    arms = {}
    for arm in sorted(os.listdir(root)):
        p = os.path.join(root, arm, "results.json")
        if os.path.isfile(p):
            arms[arm] = json.load(open(p))
    keyed = {a: arms[a] for a in ORDER if a in arms}
    keyed.update({a: r for a, r in arms.items() if a not in keyed})
    return setup, keyed


def _f(x, n=3, w=0):
    return (" " * max(w - 1, 0) + ".") if x is None else f"{x:{w}.{n}f}"


def era_slices(log):
    era = np.asarray(log["era"], int)
    return {int(k): np.nonzero(era == k)[0] for k in np.unique(era)}


def ref_series(setup, log, key, era_idx=None):
    """A moving reference, aligned to the arm's own cycles: `world[epoch]['refs'][key][era]`.
    Older runs without per-epoch refs fall back to the static setup reference."""
    world = setup.get("world") or []
    eps = log.get("epoch") or [0] * len(log["cycle"])
    eras = np.asarray(log["era"], int) - 1
    out = []
    for i, ep in enumerate(eps):
        j = era_idx if era_idx is not None else int(eras[i])
        w = world[int(ep)] if int(ep) < len(world) else None
        r = (w or {}).get("refs")
        out.append(r[key][j] if r else setup["refs"][key][j])
    return np.asarray(out, float)


def recovered(setup, log):
    """(stale - e) / (stale - floor) against the references of the epoch the cycle lived in.
    `tall`'s operating-regime check, made epoch-relative because the world moves."""
    st, fl = ref_series(setup, log, "stale"), ref_series(setup, log, "floor")
    e = np.asarray(log["e"], float)
    return (st - e) / np.maximum(st - fl, 1e-9)


# --------------------------------------------------------------------------- #
# 0. fidelity
# --------------------------------------------------------------------------- #

def fidelity(tag, ref_root=None, ref_tag="er_s0", n=90):
    """Drift off == `recital`'s `ear` mode == `ear`. Anything but IDENTICAL means the drift
    layer leaked into the substrate."""
    print(f"\n-- FIDELITY: {tag} against ear/{ref_tag} (first {n} cycles) --")
    ref_root = ref_root or EAR_FIG
    _, mine = load(tag)
    rows = []
    for arm in mine:
        p = os.path.join(ref_root, ref_tag, arm, "results.json")
        if not os.path.isfile(p):
            print(f"{arm:16s} (no reference arm on disk)")
            continue
        ref = json.load(open(p))["log"]
        cur = mine[arm]["log"]
        k = min(n, len(ref["e"]), len(cur["e"]))
        worst, fields = 0.0, {}
        for f in ("e", "t_cum", "n_moves", "level", "era", "width", "succ", "dres",
                  "e_practice", "g_per_solve"):
            if f not in ref or f not in cur:
                continue
            d = np.abs(np.asarray(ref[f][:k], float) - np.asarray(cur[f][:k], float)).max()
            fields[f] = float(d)
            worst = max(worst, float(d))
        ok = worst == 0.0
        rows.append((arm, k, worst, ok))
        print(f"{arm:16s} first {k:3d} cycles: max|delta| over "
              f"{len(fields)} fields = {worst:.2e}  {'IDENTICAL' if ok else 'DIVERGES'}"
              + ("" if ok else "  " + json.dumps({k_: v for k_, v in fields.items() if v})))
    if not rows:
        print("   (no reference run on disk -- fetch ear/er_s0 first)")
    return rows


# --------------------------------------------------------------------------- #
# 1. the world
# --------------------------------------------------------------------------- #

def world_report(setup):
    world = setup.get("world") or []
    cfg = setup["config"]
    eras = setup["eras"]
    on = cfg.get("drift_every", 0) > 0 and cfg.get("drift_cells", 0) > 0
    print(f"\n-- THE WORLD: drift {'ON' if on else 'OFF'} "
          f"(level {cfg.get('drift_level')}, {cfg.get('drift_cells')} rule cells every "
          f"{cfg.get('drift_every')} cycles from c{cfg.get('drift_start')}), "
          f"{len(world)} epochs --")
    if not world:
        return
    hdr = " ".join(f"{e['name']:>16s}" for e in eras)
    print(f"{'ep':>3} {'cyc':>4} {'L2surv':>7} {'L3surv':>7} {'corpus':>7} "
          f"{'L2/ev':>6} | stale/floor/true2 per era: {hdr}")
    for w in world:
        mi, mp = w["mag"]["vs_init"], w["mag"]["vs_prev"]
        r = w.get("refs") or {}
        cells = " ".join(
            f"{_f(r['stale'][i], 2):>4s}/{_f(r['floor'][i], 2):>4s}/"
            f"{_f((r.get('macro_true') or [{}])[i].get('2'), 2):>4s}     "
            for i in range(len(eras))) if r else ""
        print(f"{w['epoch']:>3} {w['cycle']:>4} {_f(mi['2']['survival'], 3, 7)} "
              f"{_f(mi['3']['survival'], 3, 7)} {w['mag']['corpus_vs_init']:>7.3f} "
              f"{_f(mp['2']['survival'], 3, 6)} | {cells}")


# --------------------------------------------------------------------------- #
# 2. the commit-policy grade
# --------------------------------------------------------------------------- #

def terminal_ladder(r):
    pr = [p for p in r["log"]["probe"] if "all_eras" in p]
    return (pr[-1]["all_eras"], pr[-1]["cycle"]) if pr else (None, None)


# grader cells the arm ACTUALLY PAYS. `log["gcost"]` records the whole grid every cycle
# including the cells that are unpriced instruments for this arm, so summing it blindly
# over-reports (it credited `never_base` with 75.7% of its budget in grading it never bought).
_PAYS = {"never_base": (True, None), "given": (True, None), "given_live": (True, None),
         "practice_gated": (True, "era"), "practice_late": (True, "era"),
         "practice_self": (False, "mfg"), "taught": (False, "real"),
         "practice_prov": (False, None), "prov_norecert": (False, None),
         "practice_climb": (False, "mfg"), "fixed_climb": (False, "mfg"),
         "climb_decay": (False, "mfg"), "gated_decay": (True, "era")}


def paid_grader(r):
    """Priced time this arm actually spent on the evaluation layer: the ratchet's era audition
    (only while the active level is uncommitted, and only if the arm pays it), this arm's own
    grader cell (same condition), and every recert (always paid)."""
    log, cfg = r["log"], r["config"]
    pay_era, cell = _PAYS.get(cfg.get("arm_base", ""), (False, None))
    tot = 0.0
    for i, gc in enumerate(log["gcost"]):
        active = str(log["level"][i] + 1)
        open_ = (log["vocab"][i].get(active) is None) if active in log["vocab"][i] else False
        want = ["recert"]
        if open_ and pay_era:
            want.append("era")
        if open_ and cell:
            want.append(cell)
        for k in want:
            c = gc.get(k)
            if c:
                tot += c.get("ground", 0) * cfg["d_fb"] + c.get("mat", 0) * cfg["c_mat"]
    return tot


def grade_report(setup, arms, tail=5):
    eras = setup["eras"]
    n_eras = len(eras)
    print(f"\n-- THE COMMIT-POLICY GRADE: per-era error (mean of the last {tail} cycles of "
          f"each era), and the recovered fraction against THAT EPOCH's stale/floor --")
    hdr = " ".join(f"{e['name']:>13s}" for e in eras)
    print(f"{'arm':16s} {hdr} {'t_cum':>10s} {'grader%':>8s}")
    per = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        rec = recovered(setup, log)
        cells, cellr = [], []
        for k in range(1, n_eras + 1):
            idx = sl.get(k)
            if idx is None or not len(idx):
                cells.append(None); cellr.append(None); continue
            cells.append(float(np.mean(np.asarray(log["e"])[idx[-tail:]])))
            cellr.append(float(np.mean(rec[idx[-tail:]])))
        gpaid = paid_grader(r)
        per[arm] = {"e": cells, "rec": cellr, "t_cum": log["t_cum"][-1],
                    "grader_paid": gpaid}
        row = " ".join(f"{_f(cells[i], 3, 5)}({_f(cellr[i], 2, 4)})" for i in range(n_eras))
        print(f"{arm:16s} {row} {log['t_cum'][-1]:>10.3g} "
              f"{100 * gpaid / max(log['t_cum'][-1], 1e-9):>7.1f}%")

    print(f"\n-- TERMINAL ALL-ERAS EXAM (unpriced instrument: every arm's error on EVERY "
          f"era's held-out set, in the FINAL world, with its final action set) --")
    print(f"{'arm':16s} {hdr} {'mean':>9s} {'deep':>9s}")
    lad = {}
    for arm, r in arms.items():
        L, cyc = terminal_ladder(r)
        if L is None:
            continue
        vals = [L[str(j)] for j in range(n_eras)]
        lad[arm] = vals
        print(f"{arm:16s} " + " ".join(f"{v:>13.4f}" for v in vals)
              + f" {np.mean(vals):>9.4f} {vals[-1]:>9.4f}")

    # earned-vs-given against BOTH forks of the teacher
    for ref in ("given", "given_live"):
        if ref not in lad or "never_base" not in lad:
            continue
        print(f"\n-- earned-vs-given against `{ref}`: "
              f"(e_never - e_arm)/(e_never - e_{ref}), on the terminal exam --")
        nb, gv = np.asarray(lad["never_base"]), np.asarray(lad[ref])
        den = nb - gv
        print(f"{'arm':16s} " + " ".join(f"{e['name']:>13s}" for e in eras)
              + f"   (denominators {np.round(den, 3).tolist()})")
        for arm, vals in lad.items():
            frac = [((nb[i] - vals[i]) / den[i]) if abs(den[i]) > 0.02 else None
                    for i in range(n_eras)]
            print(f"{arm:16s} " + " ".join(f"{_f(f, 3, 13)}" for f in frac))
    return per, lad


# --------------------------------------------------------------------------- #
# 3. the post-commit unit error trajectory
# --------------------------------------------------------------------------- #

def postcommit_report(setup, arms):
    print("\n-- POST-COMMIT UNIT ERROR: the committed macro's own audition (`held`) from the "
          "cycle it was committed. Baseline in every prior round: EXACTLY FLAT --")
    print(f"{'arm':16s} {'lvl':>3s} {'c*':>4s} {'n':>4s} {'held@0':>7s} {'held@end':>9s} "
          f"{'rise':>7s} {'slope/c':>9s} | {'true@0':>7s} {'true@end':>9s} {'gap_end':>8s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        for ev in [e for e in r["events"] if e["kind"] == "commit"]:
            ell, c0 = str(ev["level"]), ev["cycle"]
            idx = [i for i, c in enumerate(log["cycle"]) if c >= c0]
            h = [(log["cycle"][i], log["aud"][i].get(ell, {}).get("held")) for i in idx]
            h = [(c, x) for c, x in h if x is not None]
            t = [(log["cycle"][i], log["aud"][i].get(ell, {}).get("true")) for i in idx]
            t = [(c, x) for c, x in t if x is not None]
            if len(h) < 3:
                continue
            cs = np.asarray([c for c, _ in h], float)
            ys = np.asarray([x for _, x in h], float)
            slope = float(np.polyfit(cs, ys, 1)[0])
            tv = [x for _, x in t]
            out[(arm, ell)] = {"cycle": c0, "n": len(h), "held0": ys[0], "held1": ys[-1],
                               "rise": float(ys[-1] - ys[0]), "slope": slope,
                               "true0": (tv[0] if tv else None),
                               "true1": (tv[-1] if tv else None)}
            gap = (ys[-1] - tv[-1]) if tv else None
            print(f"{arm:16s} {ell:>3s} {c0:>4d} {len(h):>4d} {ys[0]:>7.3f} {ys[-1]:>9.3f} "
                  f"{ys[-1] - ys[0]:>+7.3f} {slope:>+9.5f} | "
                  f"{_f(tv[0] if tv else None, 3, 7)} {_f(tv[-1] if tv else None, 3, 9)} "
                  f"{_f(gap, 3, 8)}")
    return out


# --------------------------------------------------------------------------- #
# 4. staleness
# --------------------------------------------------------------------------- #

def stale_report(setup, arms):
    print("\n-- STALENESS of committed content: set precision of the committed table against "
          "the CURRENT truth (and against the truth it was mined in) --")
    print(f"{'arm':16s} {'lvl':>3s} {'n':>4s} {'prec@commit':>12s} {'prec@end':>9s} "
          f"{'rec@end':>8s} {'prec_init@end':>14s}")
    for arm, r in arms.items():
        log = r["log"]
        if not log.get("stale"):
            continue
        for ell in sorted({k for row in log["stale"] for k in row}):
            rows = [(log["cycle"][i], row.get(ell) or {})
                    for i, row in enumerate(log["stale"]) if (row.get(ell) or {}).get("n")]
            if not rows:
                continue
            first, last = rows[0][1], rows[-1][1]
            print(f"{arm:16s} {ell:>3s} {last['n']:>4d} "
                  f"{_f(first.get('prec_now'), 3, 12)} {_f(last.get('prec_now'), 3, 9)} "
                  f"{_f(last.get('recall_now'), 3, 8)} {_f(last.get('prec_init'), 3, 14)}")


# --------------------------------------------------------------------------- #
# 5. recert -- the safety net, with something to catch
# --------------------------------------------------------------------------- #

def recert_report(arms):
    print("\n-- RECERT: the live safety net. Baseline 0 swaps in 24 (`ear`) and 0 in 288 "
          "(`recital`), because a static grammar cannot invalidate committed content --")
    print(f"{'arm':16s} {'n':>4s} {'swaps':>6s} {'mean(froz-live)':>16s} "
          f"{'max(froz-live)':>15s} {'froz_prec_end':>14s} {'live_prec_end':>14s}")
    tot = {"n": 0, "swaps": 0}
    for arm, r in arms.items():
        ev = [e for e in r["events"] if e["kind"] == "recert"]
        if not ev:
            continue
        d = [e["e_frozen"] - e["e_live"] for e in ev if "e_live" in e]
        fp = [e.get("froz_precision") for e in ev if e.get("froz_precision") is not None]
        lp = [e.get("live_precision") for e in ev if e.get("live_precision") is not None]
        sw = sum(1 for e in ev if e["swapped"])
        tot["n"] += len(ev); tot["swaps"] += sw
        print(f"{arm:16s} {len(ev):>4d} {sw:>6d} {_f(np.mean(d) if d else None, 4, 16)} "
              f"{_f(max(d) if d else None, 4, 15)} {_f(fp[-1] if fp else None, 3, 14)} "
              f"{_f(lp[-1] if lp else None, 3, 14)}")
        for e in ev:
            if e["swapped"]:
                print(f"    swap  L{e['level']} c{e['cycle']} ep{e.get('epoch')}: "
                      f"{e['n_frozen']}->{e['n_live']} entries, "
                      f"e {e['e_frozen']:.4f} -> {e['e_live']:.4f}, "
                      f"prec {_f(e.get('froz_precision'), 3)} -> "
                      f"{_f(e.get('live_precision'), 3)}")
    print(f"{'TOTAL':16s} {tot['n']:>4d} {tot['swaps']:>6d}")
    return tot


# --------------------------------------------------------------------------- #
# 6. churn
# --------------------------------------------------------------------------- #

def churn_report(arms):
    print("\n-- CHURN of the LIVE mined table. `tall` retired this question a priori under a "
          "static grammar (the miner is monotone by construction) --")
    print(f"{'arm':16s} {'decay':>6s} {'lvl':>3s} {'added':>6s} {'removed':>8s} "
          f"{'n_end':>6s} {'forgotten':>10s}")
    for arm, r in arms.items():
        log = r["log"]
        dec = r["config"].get("mine_decay") if r["config"].get("arm_base", "").endswith(
            "decay") else None
        if not log.get("churn"):
            continue
        for ell in sorted({k for row in log["churn"] for k in row}):
            rows = [row[ell] for row in log["churn"] if ell in row]
            if not rows:
                continue
            forg = log["miner"][-1].get(ell, {}).get("n_forgotten")
            print(f"{arm:16s} {_f(dec, 2, 6)} {ell:>3s} "
                  f"{sum(x['added'] for x in rows):>6d} "
                  f"{sum(x['removed'] for x in rows):>8d} {rows[-1]['n']:>6d} "
                  f"{'.' if forg is None else forg:>10}")


# --------------------------------------------------------------------------- #
# 7. instruments
# --------------------------------------------------------------------------- #

def instruments(setup, arms):
    print("\n-- INSTRUMENTS (measured every cycle, never acted on) --")
    print(f"{'arm':16s} {'parse0':>7s} {'parse1':>7s} {'infill0':>8s} {'infill1':>8s} "
          f"{'read':>6s} {'bank_fresh0':>12s} {'bank_fresh1':>12s} {'n_moves':>8s} "
          f"{'width':>6s}")
    for arm, r in arms.items():
        log = r["log"]
        pr = [p["plant"] for p in log["probe"] if "plant" in p]
        bf = [x for x in (log.get("bank_fresh") or []) if x is not None]
        if not pr:
            continue
        print(f"{arm:16s} {pr[0]['parse_acc']:>7.3f} {pr[-1]['parse_acc']:>7.3f} "
              f"{pr[0]['infill_acc']:>8.3f} {pr[-1]['infill_acc']:>8.3f} "
              f"{pr[-1]['read_acc']:>6.3f} {_f(bf[0] if bf else None, 3, 12)} "
              f"{_f(bf[-1] if bf else None, 3, 12)} "
              f"{log['n_moves'][-1]:>8d} {log['width'][-1]:>6d}")

    print("\n-- mined-vs-random (the earned table against matched-size random subsets of the "
          "CURRENT true table, on the era's shadow set) --")
    print(f"{'arm':16s} {'lvl':>3s} {'cand-rand mean':>15s} {'wins':>10s}")
    for arm, r in arms.items():
        log = r["log"]
        for ell in ("2", "3"):
            d = [(a[ell]["cand"] - a[ell]["rand_k"]) for a in log["aud"]
                 if ell in a and a[ell].get("cand") is not None
                 and a[ell].get("rand_k") is not None]
            if not d:
                continue
            print(f"{arm:16s} {ell:>3s} {np.mean(d):>+15.4f} "
                  f"{sum(1 for x in d if x < 0):>5d}/{len(d):<4d}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, setup, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    eras, cfg = setup["eras"], setup["config"]
    world = setup.get("world") or []
    ev_cycles = [w["cycle"] for w in world if w["epoch"] > 0]
    colors = {a: c for a, c in zip(arms, list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors))}

    def mark(ax):
        for c in ev_cycles:
            ax.axvline(c, color="0.85", lw=0.6, zorder=0)
        for k in range(1, len(eras)):
            ax.axvline(k * cfg["era_cycles"], color="0.35", lw=1.0, ls=":")

    # fig1 -- competence, and the recovered fraction against the MOVING references
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for arm, r in arms.items():
        log = r["log"]
        axes[0].plot(log["cycle"], log["e"], label=arm, color=colors[arm], lw=1.3)
        axes[1].plot(log["cycle"], recovered(setup, log), label=arm, color=colors[arm], lw=1.3)
        for e in r["events"]:
            if e["kind"] == "commit":
                axes[0].plot([e["cycle"]], [e["e_task"]], "o", color=colors[arm], ms=5)
    if arms:
        any_log = list(arms.values())[0]["log"]
        axes[0].plot(any_log["cycle"], ref_series(setup, any_log, "stale"), color="k",
                     ls="--", lw=1.0, label="stale (moving)")
        axes[0].plot(any_log["cycle"], ref_series(setup, any_log, "floor"), color="k",
                     ls=":", lw=1.0, label="floor (moving)")
    for ax, t in ((axes[0], "e = 1 - success @ declared budget (dot = commit)"),
                  (axes[1], "recovered fraction (stale - e)/(stale - floor)")):
        mark(ax); ax.set_xlabel("cycle"); ax.set_title(t); ax.legend(fontsize=6)
    fig.suptitle(f"{tag}: grey = drift event, dotted = era boundary")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=130)

    # fig2 -- the post-commit unit error trajectory, beside the tracking macro
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for arm, r in arms.items():
        log = r["log"]
        for j, ell in enumerate(("2", "3")):
            held = [a.get(ell, {}).get("held") for a in log["aud"]]
            true = [a.get(ell, {}).get("true") for a in log["aud"]]
            axes[j].plot(log["cycle"], [np.nan if x is None else x for x in held],
                         color=colors[arm], lw=1.4, label=f"{arm} held")
            if arm == list(arms)[-1]:
                axes[j].plot(log["cycle"], [np.nan if x is None else x for x in true],
                             color="k", lw=1.0, ls="--", label="the TRACKING true macro")
    for j, ell in enumerate(("2", "3")):
        mark(axes[j]); axes[j].set_xlabel("cycle")
        axes[j].set_ylabel(f"level-{ell} macro audition error")
        axes[j].set_title(f"post-commit unit error, level {ell}")
        axes[j].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_postcommit.png"), dpi=130)

    # fig3 -- staleness and churn
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for arm, r in arms.items():
        log = r["log"]
        if not log.get("stale"):
            continue
        for ell, ls in (("2", "-"), ("3", "--")):
            pr = [(row.get(ell) or {}).get("prec_now") for row in log["stale"]]
            axes[0].plot(log["cycle"], [np.nan if x is None else x for x in pr],
                         color=colors[arm], ls=ls, lw=1.3, label=f"{arm} L{ell}")
            n = [(row.get(ell) or {}).get("n") for row in log["churn"]]
            axes[1].plot(log["cycle"], [np.nan if x is None else x for x in n],
                         color=colors[arm], ls=ls, lw=1.3, label=f"{arm} L{ell} live")
    axes[0].set_ylabel("precision of COMMITTED table vs the CURRENT truth")
    axes[0].set_title("staleness of committed content")
    axes[1].set_ylabel("entries in the LIVE mined table")
    axes[1].set_title("vocabulary size (churn)")
    for ax in axes:
        mark(ax); ax.set_xlabel("cycle"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_staleness.png"), dpi=130)

    # fig4 -- the world, and the recert margin
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    if world:
        cyc = [w["cycle"] for w in world]
        axes[0].plot(cyc, [w["mag"]["vs_init"]["2"]["survival"] for w in world], "o-",
                     label="L2 vocabulary survival vs initial")
        axes[0].plot(cyc, [w["mag"]["vs_init"]["3"]["survival"] for w in world], "s-",
                     label="L3 vocabulary survival vs initial")
        axes[0].plot(cyc, [w["mag"]["corpus_vs_init"] for w in world], "^-",
                     label="corpus survival vs initial")
        axes[0].set_xlabel("cycle"); axes[0].set_ylabel("fraction still legal")
        axes[0].set_title("how far the piece was transposed"); axes[0].legend(fontsize=7)
    for arm, r in arms.items():
        ev = [e for e in r["events"] if e["kind"] == "recert" and "e_live" in e]
        if not ev:
            continue
        axes[1].plot([e["cycle"] for e in ev],
                     [e["e_frozen"] - e["e_live"] for e in ev], "o-",
                     color=colors[arm], lw=1.2, ms=4, label=arm)
    axes[1].axhline(cfg.get("recert_margin", 0.05), color="k", ls="--", lw=1.0,
                    label="swap margin")
    axes[1].axhline(0.0, color="0.6", lw=0.8)
    mark(axes[1]); axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("e(frozen) - e(live)   [positive = the frozen table is worse]")
    axes[1].set_title("the recert channel"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_world_recert.png"), dpi=130)
    print(f"\nfigures -> {out}/fig1..fig4")


# --------------------------------------------------------------------------- #

def report(tag, tail=5):
    setup, arms = load(tag)
    cfg = setup["config"]
    print(f"\n=== {tag} ===")
    print("eras: " + " | ".join(
        f"{e['name']} d0={setup['refs']['d0'][i]:.2f} stale={setup['refs']['stale'][i]:.3f} "
        f"floor={setup['refs']['floor'][i]:.3f}" for i, e in enumerate(setup["eras"])))
    print(f"era clock {cfg['era_cycles']} cycles/era; G={cfg['g_budget']}; "
          f"mine_support={cfg['mine_support']} decay={cfg.get('mine_decay')} "
          f"floor={cfg.get('mine_floor')}; recert every {cfg['recert_every']} "
          f"margin {cfg['recert_margin']} all_levels={cfg.get('recert_all')}")
    world_report(setup)
    per, lad = grade_report(setup, arms, tail=tail)
    postcommit_report(setup, arms)
    stale_report(setup, arms)
    recert_report(arms)
    churn_report(arms)
    instruments(setup, arms)
    return setup, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tp_s0")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--ref-tag", default="er_s0")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    if a.fidelity:
        fidelity(a.tag, ref_tag=a.ref_tag)
        return
    setup, arms = report(a.tag, tail=a.tail)
    if a.figures:
        figures(a.tag, setup, arms)


if __name__ == "__main__":
    main()
