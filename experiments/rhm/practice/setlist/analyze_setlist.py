"""Reduce a `setlist` run.

    python3 rhm/practice/setlist/analyze_setlist.py --tag sl_s0 --fetch --figures
    python3 rhm/practice/setlist/analyze_setlist.py --tag sf_s0 --fetch --fidelity

Readouts, in the order the round asks for them:
  0. FIDELITY -- with demand drift off this fork is `transpose` with drift off, which is
     `recital` in its `ear` mode, which is `ear`; `sf_s0` must reproduce `ear/er_s0`.
  1. THE ADMISSIBILITY GATE -- the complement of `transpose`'s. Demand must move (KL, frozen
     coverage) while the world stays exactly as hard (`d0`, `on_grammar`, `stale`, `floor`) and
     the full true table's coverage of demand stays flat (the demand-invariant denominator).
     If the gate holds, everything below is demand-specific BY CONSTRUCTION.
  2. THE COMMIT-POLICY GRADE and THE CONCENTRATION PREMIUM -- earned-vs-given is exactly the
     quantity demand-drift should erode, bracketed by two oracles: `demand_frozen` (perfect
     concentration on epoch-0 demand) and `demand_live` (perfect concentration, tracking).
  3. POST-COMMIT UNIT ERROR, with the typed decomposition `transpose` needed a paired control
     for: `held - dem` is frozen content against perfectly demand-tracking content (the news
     this round supplies), `held - true` is frozen content against full coverage.
  4. THE TYPED PAIR -- the committed table's precision against the truth (which CANNOT move
     here) beside its coverage of the current demand (which is what drifts).
  5. RECERT, with its native stimulus present. Baseline 0/24 (`ear`), 0/288 (`recital`),
     1/62 under truth-drift (`transpose`).
  6. CHURN, and whether a forgetting miner re-concentrates.
  7. GRADER STALENESS -- the self-manufactured audition is built from the agent's own bank,
     which lags a drifting demand. Its age, and mfg-vs-real disagreement per epoch.
  8. INSTRUMENTS.
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
REMOTE = "rhm_practice_setlist"
ORDER = ["never_base", "given", "demand_frozen", "demand_live", "practice_gated",
         "practice_prov", "prov_norecert", "practice_climb", "climb_decay"]
_PAYS = {"never_base": (True, None), "given": (True, None), "demand_frozen": (True, None),
         "demand_live": (True, None), "practice_gated": (True, "era"),
         "practice_prov": (False, None), "prov_norecert": (False, None),
         "practice_climb": (False, "mfg"), "climb_decay": (False, "mfg"),
         "practice_late": (True, "era"), "practice_self": (False, "mfg"),
         "taught": (False, "real")}


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


def paid_grader(r):
    """Priced time actually spent on the evaluation layer (the whole grid is logged every
    cycle; only some cells are bought). Validated in `transpose` against `ear`'s published
    29.2% for `never_base`."""
    log, cfg = r["log"], r["config"]
    pay_era, cell = _PAYS.get(cfg.get("arm_base", ""), (False, None))
    tot = 0.0
    for i, gc in enumerate(log["gcost"]):
        active = str(log["level"][i] + 1)
        open_ = (log["vocab"][i].get(active) is None) if active in log["vocab"][i] else False
        want = ["recert"] + (["era"] if open_ and pay_era else []) \
            + ([cell] if open_ and cell else [])
        for k in want:
            c = gc.get(k)
            if c:
                tot += c.get("ground", 0) * cfg["d_fb"] + c.get("mat", 0) * cfg["c_mat"]
    return tot


# --------------------------------------------------------------------------- #
# 0. fidelity
# --------------------------------------------------------------------------- #

def fidelity(tag, ref_tag="er_s0", n=90):
    print(f"\n-- FIDELITY: {tag} against ear/{ref_tag} (first {n} cycles) --")
    _, mine = load(tag)
    rows = []
    for arm in mine:
        p = os.path.join(EAR_FIG, ref_tag, arm, "results.json")
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
        print(f"{arm:16s} first {k:3d} cycles: max|delta| over {len(fields)} fields = "
              f"{worst:.2e}  {'IDENTICAL' if ok else 'DIVERGES'}"
              + ("" if ok else "  " + json.dumps({a: b for a, b in fields.items() if b})))
    if not rows:
        print("   (no reference run on disk -- fetch ear/er_s0 first)")
    return rows


# --------------------------------------------------------------------------- #
# 1. the admissibility gate
# --------------------------------------------------------------------------- #

def gate_report(setup):
    world, cfg, eras = setup.get("world") or [], setup["config"], setup["eras"]
    on = cfg.get("demand_every", 0) > 0 and cfg.get("demand_sigma", 0) > 0
    print(f"\n-- THE ADMISSIBILITY GATE: demand {'ON' if on else 'OFF'} "
          f"(sigma {cfg.get('demand_sigma')}, kappa {cfg.get('demand_kappa')}, layers "
          f"{cfg.get('demand_levels')}, every {cfg.get('demand_every')} cycles from "
          f"c{cfg.get('demand_start')}), {len(world)} epochs --")
    print("   DEMAND MUST MOVE | THE WORLD MUST NOT")
    print(f"{'ep':>3} {'cyc':>4} | {'KLprev':>7} {'KLinit':>7} {'H':>6} {'frozCov':>8} "
          f"{'trueCov':>8} | " + " ".join(f"{e['name']:>7s} d0/stale/floor" for e in eras))
    for w in world:
        mg, r = w.get("mag") or {}, w.get("refs") or {}
        cells = "  ".join(f"{r['d0'][i]:.2f}/{r['stale'][i]:.2f}/{r['floor'][i]:.2f}"
                          for i in range(len(eras))) if r else ""
        print(f"{w['epoch']:>3} {w['cycle']:>4} | "
              f"{mg.get('kl_from_prev', {}).get('2', 0):>7.4f} "
              f"{mg.get('kl_from_init', {}).get('2', 0):>7.4f} "
              f"{mg.get('entropy', {}).get('2', 0):>6.3f} "
              f"{mg.get('init_dtable_cover', {}).get('2', 0):>8.3f} "
              f"{mg.get('true_cover', {}).get('2', 0):>8.3f} | {cells}")
    if len(world) > 1 and world[0].get("refs"):
        arr = lambda k, i: [w["refs"][k][i] for w in world if w.get("refs")]
        print("\n   flatness (max - min across epochs; the gate wants these SMALL):")
        for i, e in enumerate(eras):
            print(f"     {e['name']:>6s}  d0 {np.ptp(arr('d0', i)):.3f}   "
                  f"on_gram {np.ptp(arr('on_grammar', i)):.3f}   "
                  f"stale {np.ptp(arr('stale', i)):.3f}   "
                  f"floor {np.ptp(arr('floor', i)):.3f}")
        tc = [w["mag"]["true_cover"]["2"] for w in world if w.get("mag", {}).get("true_cover")]
        fc = [w["mag"]["init_dtable_cover"]["2"] for w in world
              if w.get("mag", {}).get("init_dtable_cover")]
        if tc:
            print(f"\n   DEMAND-INVARIANT DENOMINATOR: full true table's coverage of demand "
                  f"{min(tc):.3f}-{max(tc):.3f} (spread {np.ptp(tc):.3f})")
            print(f"   DEMAND MOVED: a concentration frozen at epoch 0 covers "
                  f"{fc[0]:.3f} -> {fc[-1]:.3f} (min {min(fc):.3f})")


# --------------------------------------------------------------------------- #
# 2. the grade and the concentration premium
# --------------------------------------------------------------------------- #

def terminal_ladder(r):
    pr = [p for p in r["log"]["probe"] if "all_eras" in p]
    return (pr[-1]["all_eras"], pr[-1]["cycle"]) if pr else (None, None)


def grade_report(setup, arms, tail=5):
    eras = setup["eras"]
    n_eras = len(eras)
    print(f"\n-- THE COMMIT-POLICY GRADE: per-era error (mean of the last {tail} cycles) --")
    hdr = " ".join(f"{e['name']:>9s}" for e in eras)
    print(f"{'arm':16s} {hdr} {'t_cum':>10s} {'grader%':>8s}")
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        cells = [float(np.mean(np.asarray(log["e"])[sl[k][-tail:]]))
                 if k in sl and len(sl[k]) else None for k in range(1, n_eras + 1)]
        gp = paid_grader(r)
        print(f"{arm:16s} " + " ".join(f"{_f(c, 4, 9)}" for c in cells)
              + f" {log['t_cum'][-1]:>10.3g} "
              f"{100 * gp / max(log['t_cum'][-1], 1e-9):>7.1f}%")

    print("\n-- TERMINAL ALL-ERAS EXAM (unpriced instrument, in the FINAL demand) --")
    print(f"{'arm':16s} {hdr} {'mean':>9s} {'deep':>9s}")
    lad = {}
    for arm, r in arms.items():
        L, _ = terminal_ladder(r)
        if L is None:
            continue
        vals = [L[str(j)] for j in range(n_eras)]
        lad[arm] = vals
        print(f"{arm:16s} " + " ".join(f"{v:>9.4f}" for v in vals)
              + f" {np.mean(vals):>9.4f} {vals[-1]:>9.4f}")

    if "given" in lad and "never_base" in lad:
        print("\n-- THE CONCENTRATION PREMIUM: earned-vs-given "
              "(e_never - e_arm)/(e_never - e_given). `given` is pure COVERAGE and is "
              "demand-invariant; > 1 means concentration beat coverage --")
        nb, gv = np.asarray(lad["never_base"]), np.asarray(lad["given"])
        den = nb - gv
        print(f"{'arm':16s} " + " ".join(f"{e['name']:>9s}" for e in eras)
              + f"   (denominators {np.round(den, 3).tolist()})")
        for arm, vals in lad.items():
            frac = [((nb[i] - vals[i]) / den[i]) if abs(den[i]) > 0.02 else None
                    for i in range(n_eras)]
            print(f"{arm:16s} " + " ".join(f"{_f(f, 3, 9)}" for f in frac))
    return lad


# --------------------------------------------------------------------------- #
# 3. post-commit unit error, typed
# --------------------------------------------------------------------------- #

def postcommit_report(arms):
    print("\n-- POST-COMMIT UNIT ERROR. `held` is the committed macro's own audition; `dem` "
          "is a PERFECTLY DEMAND-TRACKING concentration on the same set, `true` is full "
          "coverage. (held - dem) is the news this round supplies --")
    print(f"{'arm':16s} {'lvl':>3s} {'c*':>4s} {'n':>4s} {'held@0':>7s} {'held@end':>9s} "
          f"{'rise':>7s} {'slope/c':>9s} | {'h-dem@0':>8s} {'h-dem@end':>10s} "
          f"{'h-dem mean':>11s} | {'h-true mean':>12s}")
    out = {}
    for arm, r in arms.items():
        log = r["log"]
        for ev in [e for e in r["events"] if e["kind"] == "commit"]:
            ell, c0 = str(ev["level"]), ev["cycle"]
            idx = [i for i, c in enumerate(log["cycle"]) if c >= c0]
            trip = [(log["cycle"][i], log["aud"][i].get(ell, {})) for i in idx]
            h = [(c, a.get("held")) for c, a in trip if a.get("held") is not None]
            gd = [(a.get("held") - a.get("dem")) for _, a in trip
                  if a.get("held") is not None and a.get("dem") is not None]
            gt = [(a.get("held") - a.get("true")) for _, a in trip
                  if a.get("held") is not None and a.get("true") is not None]
            if len(h) < 3:
                continue
            cs = np.asarray([c for c, _ in h], float)
            ys = np.asarray([x for _, x in h], float)
            slope = float(np.polyfit(cs, ys, 1)[0])
            out[(arm, ell)] = {"rise": float(ys[-1] - ys[0]), "slope": slope,
                               "gap_dem": (float(np.mean(gd)) if gd else None)}
            print(f"{arm:16s} {ell:>3s} {c0:>4d} {len(h):>4d} {ys[0]:>7.3f} {ys[-1]:>9.3f} "
                  f"{ys[-1] - ys[0]:>+7.3f} {slope:>+9.5f} | "
                  f"{_f(gd[0] if gd else None, 3, 8)} {_f(gd[-1] if gd else None, 3, 10)} "
                  f"{_f(np.mean(gd) if gd else None, 3, 11)} | "
                  f"{_f(np.mean(gt) if gt else None, 3, 12)}")
    return out


# --------------------------------------------------------------------------- #
# 4. the typed pair: precision cannot move, coverage does
# --------------------------------------------------------------------------- #

def typed_pair(arms):
    print("\n-- THE TYPED PAIR: precision against the TRUTH (cannot move here -- nothing "
          "becomes false) beside coverage of the CURRENT DEMAND (what drifts) --")
    print(f"{'arm':16s} {'lvl':>3s} {'n':>4s} {'prec@commit':>12s} {'prec@end':>9s} "
          f"{'cov@commit':>11s} {'cov@end':>8s} {'cov_min':>8s} {'cov@init_dem':>13s}")
    for arm, r in arms.items():
        log = r["log"]
        if not log.get("stale"):
            continue
        for ell in sorted({k for row in log["stale"] for k in row}):
            rows = [row.get(ell) or {} for row in log["stale"] if (row.get(ell) or {}).get("n")]
            if not rows:
                continue
            cov = [x.get("cover_now") for x in rows if x.get("cover_now") is not None]
            print(f"{arm:16s} {ell:>3s} {rows[-1]['n']:>4d} "
                  f"{_f(rows[0].get('prec_now'), 3, 12)} {_f(rows[-1].get('prec_now'), 3, 9)} "
                  f"{_f(cov[0] if cov else None, 3, 11)} "
                  f"{_f(cov[-1] if cov else None, 3, 8)} "
                  f"{_f(min(cov) if cov else None, 3, 8)} "
                  f"{_f(rows[-1].get('cover_init'), 3, 13)}")


# --------------------------------------------------------------------------- #
# 5. recert -- with its native stimulus
# --------------------------------------------------------------------------- #

def recert_report(arms):
    print("\n-- RECERT, with its native stimulus present. Baseline: 0 swaps in 24 (`ear`), "
          "0 in 288 (`recital`), 1 in 62 under TRUTH-drift (`transpose`) --")
    print(f"{'arm':16s} {'n':>4s} {'swaps':>6s} {'mean(f-l)':>10s} {'max(f-l)':>9s} "
          f"{'froz_cov_end':>13s} {'live_cov_end':>13s} {'mean cov gap':>13s}")
    tot = {"n": 0, "swaps": 0}
    for arm, r in arms.items():
        ev = [e for e in r["events"] if e["kind"] == "recert"]
        if not ev:
            continue
        d = [e["e_frozen"] - e["e_live"] for e in ev if "e_live" in e]
        fc = [e.get("froz_cover") for e in ev if e.get("froz_cover") is not None]
        lc = [e.get("live_cover") for e in ev if e.get("live_cover") is not None]
        gap = [e["live_cover"] - e["froz_cover"] for e in ev
               if e.get("live_cover") is not None and e.get("froz_cover") is not None]
        sw = sum(1 for e in ev if e["swapped"])
        tot["n"] += len(ev); tot["swaps"] += sw
        print(f"{arm:16s} {len(ev):>4d} {sw:>6d} {_f(np.mean(d) if d else None, 4, 10)} "
              f"{_f(max(d) if d else None, 4, 9)} {_f(fc[-1] if fc else None, 3, 13)} "
              f"{_f(lc[-1] if lc else None, 3, 13)} "
              f"{_f(np.mean(gap) if gap else None, 4, 13)}")
        for e in ev:
            if e["swapped"]:
                print(f"    swap L{e['level']} c{e['cycle']} ep{e.get('epoch')}: "
                      f"{e['n_frozen']}->{e['n_live']} entries, e {e['e_frozen']:.4f} -> "
                      f"{e['e_live']:.4f}, cover {_f(e.get('froz_cover'), 3)} -> "
                      f"{_f(e.get('live_cover'), 3)}")
    print(f"{'TOTAL':16s} {tot['n']:>4d} {tot['swaps']:>6d}")
    return tot


# --------------------------------------------------------------------------- #
# 6/7/8. churn, grader staleness, instruments
# --------------------------------------------------------------------------- #

def churn_report(arms):
    print("\n-- CHURN of the LIVE mined table, and whether it RE-CONCENTRATES --")
    print(f"{'arm':16s} {'decay':>6s} {'lvl':>3s} {'added':>6s} {'removed':>8s} {'n_end':>6s} "
          f"{'forgot':>7s} {'cov@0':>7s} {'cov@end':>8s} {'cov_mean':>9s}")
    for arm, r in arms.items():
        log, cfg = r["log"], r["config"]
        dec = cfg.get("mine_decay") if cfg.get("arm_base", "").endswith("decay") else None
        if not log.get("churn"):
            continue
        for ell in sorted({k for row in log["churn"] for k in row}):
            rows = [row[ell] for row in log["churn"] if ell in row]
            cov = [x.get("cover") for x in rows if x.get("cover") is not None]
            forg = log["miner"][-1].get(ell, {}).get("n_forgotten")
            print(f"{arm:16s} {_f(dec, 2, 6)} {ell:>3s} "
                  f"{sum(x['added'] for x in rows):>6d} "
                  f"{sum(x['removed'] for x in rows):>8d} {rows[-1]['n']:>6d} "
                  f"{('.' if forg is None else forg):>7} "
                  f"{_f(cov[0] if cov else None, 3, 7)} "
                  f"{_f(cov[-1] if cov else None, 3, 8)} "
                  f"{_f(np.mean(cov) if cov else None, 3, 9)}")


def grader_report(arms):
    print("\n-- GRADER STALENESS: the self-manufactured audition is built from the agent's "
          "own bank, which lags a drifting demand --")
    print(f"{'arm':16s} {'builds':>7s} {'age@end':>8s} {'mfg_pol':>8s} {'real_pol':>9s} "
          f"{'mfg-real':>9s} {'corr':>6s} {'bank_fresh0':>12s} {'bank_fresh1':>12s}")
    for arm, r in arms.items():
        log = r["log"]
        mp = [c.get("mfg", {}).get("pol") for c in log["clim"]]
        rp = [c.get("real", {}).get("pol") for c in log["clim"]]
        pair = [(a, b) for a, b in zip(mp, rp) if a is not None and b is not None]
        ages = [c.get("mfg", {}).get("age") for c in log["clim"]
                if c.get("mfg", {}).get("age") is not None]
        bf = [x for x in (log.get("bank_fresh") or []) if x is not None]
        corr = (float(np.corrcoef([a for a, _ in pair], [b for _, b in pair])[0, 1])
                if len(pair) > 3 else None)
        print(f"{arm:16s} {len(log.get('mfg') or []):>7d} "
              f"{(ages[-1] if ages else 0):>8d} "
              f"{_f(np.mean([a for a, _ in pair]) if pair else None, 3, 8)} "
              f"{_f(np.mean([b for _, b in pair]) if pair else None, 3, 9)} "
              f"{_f(np.mean([a - b for a, b in pair]) if pair else None, 3, 9)} "
              f"{_f(corr, 2, 6)} {_f(bf[0] if bf else None, 3, 12)} "
              f"{_f(bf[-1] if bf else None, 3, 12)}")


def instruments(arms):
    print("\n-- INSTRUMENTS --")
    print(f"{'arm':16s} {'parse0':>7s} {'parse1':>7s} {'infill0':>8s} {'infill1':>8s} "
          f"{'read':>6s} {'n_moves':>8s} {'width':>6s}")
    for arm, r in arms.items():
        log = r["log"]
        pr = [p["plant"] for p in log["probe"] if "plant" in p]
        if not pr:
            continue
        print(f"{arm:16s} {pr[0]['parse_acc']:>7.3f} {pr[-1]['parse_acc']:>7.3f} "
              f"{pr[0]['infill_acc']:>8.3f} {pr[-1]['infill_acc']:>8.3f} "
              f"{pr[-1]['read_acc']:>6.3f} {log['n_moves'][-1]:>8d} {log['width'][-1]:>6d}")
    print("\n-- mined-vs-random (earned table vs matched-size random subsets of the true "
          "table; both are demand-blind, so this is the demand-neutral control) --")
    print(f"{'arm':16s} {'lvl':>3s} {'cand-rand':>10s} {'wins':>10s}")
    for arm, r in arms.items():
        for ell in ("2", "3"):
            d = [(a[ell]["cand"] - a[ell]["rand_k"]) for a in r["log"]["aud"]
                 if ell in a and a[ell].get("cand") is not None
                 and a[ell].get("rand_k") is not None]
            if d:
                print(f"{arm:16s} {ell:>3s} {np.mean(d):>+10.4f} "
                      f"{sum(1 for x in d if x < 0):>5d}/{len(d):<4d}")


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
            ax.axvline(c, color="0.9", lw=0.5, zorder=0)
        for k in range(1, len(eras)):
            ax.axvline(k * cfg["era_cycles"], color="0.35", lw=1.0, ls=":")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for arm, r in arms.items():
        log = r["log"]
        axes[0].plot(log["cycle"], log["e"], label=arm, color=colors[arm], lw=1.3)
        for e in r["events"]:
            if e["kind"] == "commit":
                axes[0].plot([e["cycle"]], [e["e_task"]], "o", color=colors[arm], ms=5)
    if world and world[0].get("refs"):
        cyc = [w["cycle"] for w in world]
        for k, ls in (("stale", "--"), ("floor", ":")):
            axes[1].plot(cyc, [w["refs"][k][0] for w in world], ls, color="k", label=k)
        axes[1].plot(cyc, [w["mag"]["true_cover"]["2"] for w in world if w.get("mag")],
                     "^-", label="true table coverage of demand")
        axes[1].plot(cyc, [w["mag"]["init_dtable_cover"]["2"] for w in world if w.get("mag")],
                     "s-", label="epoch-0 concentration's coverage")
    axes[0].set_xlabel("cycle"); axes[0].set_ylabel("e (dot = commit)")
    axes[0].set_title(f"{tag}: competence"); axes[0].legend(fontsize=6); mark(axes[0])
    axes[1].set_xlabel("cycle")
    axes[1].set_title("THE GATE: demand moves (squares) while the world does not (k lines)")
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_competence_gate.png"), dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for arm, r in arms.items():
        log = r["log"]
        for j, ell in enumerate(("2", "3")):
            gd = [(a.get(ell, {}).get("held") - a[ell]["dem"])
                  if a.get(ell, {}).get("held") is not None
                  and a.get(ell, {}).get("dem") is not None else np.nan for a in log["aud"]]
            axes[j].plot(log["cycle"], gd, color=colors[arm], lw=1.3, label=arm)
    for j, ell in enumerate(("2", "3")):
        mark(axes[j]); axes[j].axhline(0, color="0.6", lw=0.8)
        axes[j].set_xlabel("cycle")
        axes[j].set_ylabel(f"held - dem, level {ell}")
        axes[j].set_title(f"frozen content vs perfectly demand-tracking content, L{ell}")
        axes[j].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_postcommit.png"), dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for arm, r in arms.items():
        log = r["log"]
        if not log.get("stale"):
            continue
        for ell, ls in (("2", "-"), ("3", "--")):
            cv = [(row.get(ell) or {}).get("cover_now") for row in log["stale"]]
            pc = [(row.get(ell) or {}).get("prec_now") for row in log["stale"]]
            axes[0].plot(log["cycle"], [np.nan if x is None else x for x in cv],
                         color=colors[arm], ls=ls, lw=1.3, label=f"{arm} L{ell}")
            axes[1].plot(log["cycle"], [np.nan if x is None else x for x in pc],
                         color=colors[arm], ls=ls, lw=1.3, label=f"{arm} L{ell}")
    axes[0].set_ylabel("coverage of CURRENT demand"); axes[0].set_title("what drifts")
    axes[1].set_ylabel("precision against the TRUTH")
    axes[1].set_title("what cannot move here (the typed pair)")
    for ax in axes:
        mark(ax); ax.set_xlabel("cycle"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_typed_pair.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, r in arms.items():
        ev = [e for e in r["events"] if e["kind"] == "recert" and "e_live" in e]
        if ev:
            ax.plot([e["cycle"] for e in ev], [e["e_frozen"] - e["e_live"] for e in ev],
                    "o-", color=colors[arm], lw=1.2, ms=4, label=arm)
    ax.axhline(cfg.get("recert_margin", 0.05), color="k", ls="--", lw=1.0, label="swap margin")
    ax.axhline(0.0, color="0.6", lw=0.8)
    mark(ax); ax.set_xlabel("cycle")
    ax.set_ylabel("e(frozen) - e(live)   [positive = the frozen table is worse]")
    ax.set_title("the recert channel, with its native stimulus"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_recert.png"), dpi=130)
    print(f"\nfigures -> {out}/fig1..fig4")


def report(tag, tail=5):
    setup, arms = load(tag)
    cfg = setup["config"]
    print(f"\n=== {tag} ===")
    print("eras: " + " | ".join(
        f"{e['name']} stale={setup['refs']['stale'][i]:.3f} floor={setup['refs']['floor'][i]:.3f}"
        for i, e in enumerate(setup["eras"])))
    print(f"era clock {cfg['era_cycles']}; G={cfg['g_budget']}; mine_support="
          f"{cfg['mine_support']} decay={cfg.get('mine_decay')}; recert every "
          f"{cfg['recert_every']} margin {cfg['recert_margin']}")
    gate_report(setup)
    grade_report(setup, arms, tail=tail)
    postcommit_report(arms)
    typed_pair(arms)
    recert_report(arms)
    churn_report(arms)
    grader_report(arms)
    instruments(arms)
    return setup, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sl_s0")
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
