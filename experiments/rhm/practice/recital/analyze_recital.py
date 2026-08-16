"""Reduce a `recital` run.

    python3 rhm/practice/recital/analyze_recital.py --tag rc_s0 --fetch --figures
    python3 rhm/practice/recital/analyze_recital.py --tag rf_s0 --fetch --fidelity
    python3 -c "import sys; sys.path.insert(0,'rhm/practice/recital'); \
        import analyze_recital as A; A.predict('rc_s0')"

Readouts, in the order the round asks for them:
  0. FIDELITY -- with `--t-budget 0` this fork is `ear`; `rf_s0` must reproduce `er_s0` cycle
     for cycle, on all eight of `ear`'s arms (torch's global RNG carries across arms, so the
     gate is only exact if the same arms run in the same order for the same number of cycles)
  1. THE MATCHED-BUDGET GRADE -- every arm's error on EVERY era's held-out metering set at the
     same total priced budget (the terminal all-eras probe: the final exam on the whole
     ladder, an unpriced instrument), plus the same curve at intermediate budget fractions
  2. BOUNDARY PLACEMENTS -- where each policy actually put its era boundaries, in cycles and
     in priced time, with the reason it advanced and what it was holding when it did
  3. REGRET at matched budget against the best FIXED schedule in the sweep
  4. priced competence per era actually visited, the earned-vs-given fraction, cost-to-depth
  5. certificate / advancement firings against the offline replay's prediction
  6. GRADER COST, the grader grid, the oracle check, the seam, mined-vs-random, plant guard
"""

import argparse
import json
import os
import subprocess

import numpy as np

from rhm.practice.ear.analyze_ear import (GRADERS, era_slices, interp, replay_unitlp, series)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
EAR_FIG = os.path.join(os.path.dirname(HERE), "ear", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_recital"
ORDER = ["never_base", "given", "fixed_climb", "pace_cert", "pace_task", "pace_comp"]
# which grader cell each arm's certificate reads (used to mark the grid and price it)
ARM_READS = {"practice_gated": ("era", "act"), "practice_late": ("era", "act"),
         "practice_self": ("mfg", "pol"), "taught": ("real", "pol"),
         "practice_climb": ("mfg", "pol"), "fixed_climb": ("mfg", "pol"),
         "sched_frac": ("mfg", "pol"), "pace_cert": ("mfg", "pol"),
         "pace_task": ("mfg", "pol"), "pace_comp": ("mfg", "pol")}


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


def read_cell(r):
    """Which grader cell this arm's certificate ACTUALLY read, derived from the logs rather
    than from the arm's name -- `nm=` renames make the name unreliable, and `sf1`/`sf2` are
    global config defaults so they identify nothing. `log["cert"]["A"]` is the value the
    certificate consumed; find the (context, evaluator) cell of the grid it came from.

    Returns None for an arm with no certificate at all (`never_base`, `given`), which is the
    honest answer: those arms buy no auditions and owe the grader nothing.
    """
    log = r["log"]
    votes = {}
    for c, cl in zip(log["cert"], log["clim"]):
        if c.get("A") is None:
            continue
        for ctx, ev in GRADERS:
            if cl.get(ctx, {}).get(ev) == c["A"]:
                votes[(ctx, ev)] = votes.get((ctx, ev), 0) + 1
    return max(votes, key=votes.get) if votes else None


def base_of(arm, r):
    """The ARMS key an arm was instantiated from, when it is recoverable. Runs from this file
    onward stamp `config.arm_base`; older runs fall back to the label's prefix. Only used for
    display -- pricing goes through `read_cell`, which cannot be fooled by a rename."""
    b = r.get("config", {}).get("arm_base")
    if b in ARM_READS:
        return b
    for k in sorted(ARM_READS, key=len, reverse=True):
        if arm.startswith(k):
            return k
    return arm


def terminal_ladder(r):
    """The last all-eras probe: this arm's error on EVERY era's fixed metering set, with its
    final action set and final plant. The matched-budget final exam."""
    pr = [p for p in r["log"]["probe"] if "all_eras" in p]
    return (pr[-1]["all_eras"], pr[-1]["cycle"], pr[-1].get("t_cum")) if pr else (None, None, None)


def ladder_at(r, t_at):
    """All-eras probe interpolated to a priced-time checkpoint -- the matched-budget curve."""
    pr = [p for p in r["log"]["probe"] if "all_eras" in p and p.get("t_cum") is not None]
    if not pr:
        return None
    t = [p["t_cum"] for p in pr]
    return {k: interp(t, [p["all_eras"][k] for p in pr], t_at) for k in pr[0]["all_eras"]}


# --------------------------------------------------------------------------- #
# 0. the fork's fidelity gate
# --------------------------------------------------------------------------- #

def fidelity(tag, ref_root=None, ref_tag="er_s0", n=90):
    """With `--t-budget 0` the loop is `ear`'s loop. Anything other than IDENTICAL means the
    advancement layer leaked into the substrate."""
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
        de = np.abs(np.asarray(ref["e"][:k]) - np.asarray(cur["e"][:k]))
        dt = np.abs(np.asarray(ref["t_cum"][:k]) - np.asarray(cur["t_cum"][:k]))
        ok = de.max() == 0 and dt.max() == 0
        rows.append((arm, k, float(de.max()), float(dt.max()), ok))
        print(f"{arm:16s} first {k:3d} cycles: max|de|={de.max():.2e}  "
              f"max|dt_cum|={dt.max():.2e}  {'IDENTICAL' if ok else 'DIVERGES'}")
    if not rows:
        print("   (no reference run on disk -- fetch ear/er_s0 first)")
    return rows


# --------------------------------------------------------------------------- #
# the main report
# --------------------------------------------------------------------------- #

def report(tag, tail=5):
    setup, arms = load(tag)
    eras, refs, cfg = setup["eras"], setup["refs"], setup["config"]
    T = float(cfg.get("t_budget", 0.0))
    n_eras = len(eras)
    print(f"\n=== {tag} ===")
    print("eras: " + " | ".join(
        f"{e['name']} d0={refs['d0'][i]:.2f} stale_base={refs['stale'][i]:.3f} "
        f"stale_true={refs['stale_true'][i]:.3f} floor={refs['floor'][i]:.3f}"
        for i, e in enumerate(eras)))
    print(f"total priced budget T={T:.0f} matched across arms; declared per-solve grounding "
          f"budget G={cfg['g_budget']}; detector (c={cfg['sil_c']}, c_v={cfg['sil_cv']}, "
          f"W={cfg['sil_W']}, hold={cfg['sil_hold']}, lp_min_drop={cfg['lp_min_drop']}, "
          f"min_cycle={cfg['sil_min_cycle']}) shared by BOTH pacers")

    # ---- 1. THE MATCHED-BUDGET GRADE ---------------------------------------------------
    print(f"\n-- MATCHED-BUDGET GRADE: error on every era's held-out set at the end of the "
          f"SAME total budget (terminal all-eras probe; unpriced instrument) --")
    hdr = " ".join(f"{eras[j]['name']:>9s}" for j in range(n_eras))
    print(f"{'arm':14s} {'cyc':>4s} {'t_end':>10s} {'%T':>5s} {hdr} {'mean':>8s} "
          f"{'era_end':>7s}")
    grade = {}
    for arm, r in arms.items():
        lad, cyc, t_at = terminal_ladder(r)
        log = r["log"]
        t_end = log["t_cum"][-1]
        if lad is None:
            continue
        vals = [lad[str(j)] for j in range(n_eras)]
        grade[arm] = {"ladder": vals, "mean": float(np.mean(vals)), "deep": vals[-1],
                      "t_end": t_end, "cycles": len(log["cycle"]),
                      "era_end": int(log["era"][-1]),
                      "complete": r["complete"]}
        print(f"{arm:14s} {len(log['cycle']):4d} {t_end:10.0f} "
              f"{(100 * t_end / T if T > 0 else 0):5.1f} "
              + " ".join(f"{x:9.4f}" for x in vals)
              + f" {np.mean(vals):8.4f} {int(log['era'][-1]):7d}"
              + ("" if r["complete"] else "  [INCOMPLETE]"))

    if T > 0:
        print("\n-- the same exam at intermediate matched budgets (deepest era's cell) --")
        fr = [0.25, 0.50, 0.75, 1.00]
        print(f"{'arm':14s} " + " ".join(f"{'t=' + str(int(100 * f)) + '%':>9s}" for f in fr))
        for arm, r in arms.items():
            row = []
            for f in fr:
                lad = ladder_at(r, f * T)
                row.append(np.nan if lad is None else lad[str(n_eras - 1)])
            print(f"{arm:14s} " + " ".join(f"{x:9.4f}" for x in row))

    # ---- 2. BOUNDARY PLACEMENTS ---------------------------------------------------------
    print("\n-- BOUNDARY PLACEMENTS: where each policy put its era boundaries --")
    print(f"{'arm':14s} {'from':>4s}->{'to':<3s} {'cyc':>4s} {'c_in':>4s} {'reason':>9s} "
          f"{'t_at':>10s} {'%T':>5s} {'t_era':>10s} {'cert_run':>8s} {'cert_drop':>9s} "
          f"{'task_run':>8s} {'task_drop':>9s} {'vocab':>12s}")
    bounds = {}
    for arm, r in arms.items():
        bounds[arm] = []
        for ev in r["events"]:
            if ev["kind"] != "advance":
                continue
            bounds[arm].append(ev)
            g = lambda x: "        ." if x is None else f"{x:9.4f}"
            print(f"{arm:14s} {ev['from_era']:4d}->{ev['to_era']:<3d} {ev['cycle']:4d} "
                  f"{ev['c_in_era']:4d} {ev['reason']:>9s} {ev['t_cum']:10.0f} "
                  f"{(100 * ev['frac_of_budget'] if ev['frac_of_budget'] else 0):5.1f} "
                  f"{ev['t_era']:10.0f} {ev['cert_run']:8d} {g(ev['cert_drop'])} "
                  f"{ev['task_run']:8d} {g(ev['task_drop'])} "
                  f"{json.dumps(ev['committed']):>12s}")
        if not bounds[arm]:
            print(f"{arm:14s}   (never advanced -- ended in era "
                  f"{int(r['log']['era'][-1])})")

    # ---- 3. REGRET against the best fixed schedule --------------------------------------
    fixed = [a for a in grade if a.startswith("fixed_climb") or a.startswith("sched")]
    if fixed:
        print("\n-- REGRET at matched budget against the BEST FIXED SCHEDULE in the sweep --")
        for key, nm in (("deep", f"{eras[-1]['name']} cell"), ("mean", "mean over the ladder")):
            best = min(fixed, key=lambda a: grade[a][key])
            print(f"   best fixed schedule on {nm}: {best} = {grade[best][key]:.4f}")
            for arm in grade:
                if arm in ("never_base", "given"):
                    continue
                print(f"      {arm:14s} {key}={grade[arm][key]:.4f}  "
                      f"regret={grade[arm][key] - grade[best][key]:+.4f}")

    # ---- 4. per-era competence actually visited ------------------------------------------
    print(f"\n-- priced competence per era ACTUALLY VISITED (mean e over the last {tail} "
          f"cycles in that era) --")
    per = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        e = np.asarray(log["e"], float)
        t = np.asarray(log["t_cum"], float)
        per[arm] = {k: {"e": float(e[idx[-tail:]].mean()), "e_last": float(e[idx[-1]]),
                        "n": len(idx), "t_end": float(t[idx[-1]]),
                        "t_era": float(t[idx[-1]] - (t[idx[0] - 1] if idx[0] else 0.0)),
                        "nm": int(log["n_moves"][idx[-1]])}
                    for k, idx in sl.items()}
        per[arm]["_t"] = t
        per[arm]["_e"] = e
    ks = sorted({k for a in per for k in per[a] if isinstance(k, int)})
    print(f"{'arm':14s} " + " ".join(f"{'era' + str(k):>30s}" for k in ks))
    for arm, x in per.items():
        cells = []
        for k in ks:
            if k in x:
                cells.append(f"  e={x[k]['e']:.4f} (n{x[k]['n']:3d}, t={x[k]['t_era']:9.0f}, "
                             f"nm{x[k]['nm']:2d})")
            else:
                cells.append(f"{'  --- never visited ---':>30s}")
        print(f"{arm:14s} " + " ".join(cells))

    if "never_base" in per and "given" in per:
        print("\n-- EARNED-VS-GIVEN fraction on the TERMINAL LADDER (matched budget): "
              "(e_never - e_arm)/(e_never - e_given), per era cell --")
        for arm in grade:
            if arm in ("never_base", "given"):
                continue
            row = []
            for j in range(n_eras):
                den = grade["never_base"]["ladder"][j] - grade["given"]["ladder"][j]
                num = grade["never_base"]["ladder"][j] - grade[arm]["ladder"][j]
                row.append(num / den if abs(den) > 1e-9 else float("nan"))
            print(f"{arm:14s} " + " ".join(f"{x:10.3f}" for x in row))
        print("   (denominators: " + " ".join(
            f"{eras[j]['name']}={grade['never_base']['ladder'][j] - grade['given']['ladder'][j]:+.4f}"
            for j in range(n_eras)) + ")")
        print("\n-- cost-to-depth on the terminal ladder (deepest cell - shallowest) --")
        for arm in grade:
            print(f"{arm:14s} {grade[arm]['ladder'][-1] - grade[arm]['ladder'][0]:+.4f}")

    # ---- 5. commits, recerts, and the offline prediction ---------------------------------
    print("\n-- commit events --")
    print(f"{'arm':14s} {'era':>3s} {'lvl':>3s} {'cyc':>4s} {'c_in':>4s} {'prov':>5s} "
          f"{'entries':>7s} {'recall':>7s} {'prec':>6s} {'A_read':>7s} {'mfg_pol':>7s} "
          f"{'real_pol':>8s} {'sil':>4s}")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "commit":
                continue
            f = lambda x: "    .  " if x is None else f"{x:7.4f}"
            pr = ev['tab_precision']
            print(f"{arm:14s} {ev['era']:3d} {ev['level']:3d} {ev['cycle']:4d} "
                  f"{ev['c_in_era']:4d} {str(ev.get('provisional')):>5s} "
                  f"{ev['n_entries']:7d} {ev['tab_recall']:7.3f} "
                  f"{(pr if pr is not None else float('nan')):6.3f} "
                  f"{f(ev['shadow'])} {f(ev.get('aud_mfg_pol'))} "
                  f"{f(ev.get('aud_real_pol'))[:8]:>8s} {ev['sil_run']:4d}")

    print("\n-- RECERT events (frozen committed table vs the live mined one, in consumption) --")
    n_swap, deltas = {}, []
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "recert":
                continue
            n_swap[arm] = n_swap.get(arm, 0) + int(bool(ev["swapped"]))
            el = ev.get("e_live")
            if el is not None:
                deltas.append(ev["e_frozen"] - el)
    if deltas:
        d = np.asarray(deltas)
        print(f"   {len(d)} recerts over all arms: frozen-minus-live mean={d.mean():+.4f} "
              f"sd={d.std():.4f} max={d.max():+.4f} (margin={cfg['recert_margin']}); "
              "swaps: " + ", ".join(f"{a}={n}" for a, n in n_swap.items()))
    else:
        print("   (none)")

    predict(tag, arms=arms, setup=setup)

    # ---- 6. grader cost, the grid, the oracle check, the seam, mined-vs-random, plant -----
    print("\n-- GRADER COST: what the evaluation layer spent, and what it would have cost --")
    d_fb, c_mat = cfg["d_fb"], cfg["c_mat"]
    for arm, r in arms.items():
        log = r["log"]
        tot = {}
        for g in log["gcost"]:
            for k, c in g.items():
                tot[k] = tot.get(k, 0.0) + c["ground"] * d_fb + c["mat"] * c_mat
        cell = read_cell(r)
        paid = (tot.get(cell[0], 0.0) if cell else 0.0) + tot.get("recert", 0.0)
        t_end = log["t_cum"][-1]
        print(f"{arm:14s} priced total={t_end:11.0f}  reads="
              f"{('/'.join(cell) if cell else 'none'):8s} grader PAID={paid:9.0f} "
              f"({100 * paid / max(t_end, 1):5.2f}%)   would-have-cost: "
              + " ".join(f"{k}={x:.0f}" for k, x in sorted(tot.items())))

    print("\n-- THE GRADER GRID: every audition series, per arm and era. `>` marks the cell "
          "this arm's certificate actually read --")
    for arm, r in arms.items():
        log = r["log"]
        gr = read_cell(r)
        fired = {ev["era"]: ev["cycle"] for ev in r["events"] if ev["kind"] == "commit"}
        for k, idx in era_slices(log).items():
            active = int(log["level"][idx[0]]) + 1
            rows = []
            for ctx, ev in GRADERS:
                S = series(log, idx, ctx, ev, active)
                vals = [x for x in S if x is not None]
                if not vals:
                    continue
                mark = ">" if gr == (ctx, ev) else " "
                noise = (float(np.std(np.diff(vals)) / np.sqrt(2)) if len(vals) > 2 else
                         float("nan"))
                rows.append(f"  {mark}{ctx:>4s}/{ev:<3s} n={len(vals):3d} "
                            f"first={vals[0]:.3f} min={min(vals):.3f} "
                            f"span={vals[0] - min(vals):.3f} last5={np.mean(vals[-5:]):.3f} "
                            f"step-noise={noise:.3f}")
            if rows:
                print(f"{arm:14s} era{k} (earning L{active}, n={len(idx)} cycles)"
                      + (f"  [commit c{fired[k]}]" if k in fired else ""))
                print("\n".join(rows))

    print("\n-- ORACLE CHECK on the self-manufactured audition contexts (instrument) --")
    for arm, r in arms.items():
        for ev in r["log"].get("mfg", []):
            st = ev["stats"]
            print(f"{arm:14s} era{ev['era']} L{ev['level']}n{ev['node']} c{ev['cycle']} "
                  f"n={ev['n']} bank={ev['bank']} lower={ev['lower_entries']}  "
                  f"on_gram={st['on_grammar']:.3f} d*={st['d_mean']:.2f} "
                  f"broken={st['frac_broken']:.3f} node_empty={st['node_empty']:.3f} "
                  f"node_overlap={st.get('node_overlap', float('nan')):.3f}")
    print("\n   agreement -- the SAME candidate on the manufactured set and on the real one:")
    for arm, r in arms.items():
        log = r["log"]
        for k, idx in era_slices(log).items():
            active = int(log["level"][idx[0]]) + 1
            for ev in ("pol", "act"):
                M = series(log, idx, "mfg", ev, active)
                R = series(log, idx, "real", ev, active)
                pair = [(a, b) for a, b in zip(M, R) if a is not None and b is not None]
                if len(pair) < 4:
                    continue
                a = np.array([p[0] for p in pair]); b = np.array([p[1] for p in pair])
                rho = (float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9
                       else float("nan"))
                print(f"{arm:14s} era{k} L{active} {ev:4s} n={len(pair):3d}  "
                      f"bias={np.mean(a - b):+.4f} mad={np.mean(np.abs(a - b)):.4f} "
                      f"corr={rho:+.3f}")

    print("\n-- the depth seam: the arm's grader at commit vs realised competence one era on --")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "commit":
                continue
            nxt = ev["era"] + 1
            if nxt not in per[arm]:
                continue
            real = per[arm][nxt]["e"]
            for name, val in (("read", ev["shadow"]), ("era_act", ev.get("aud_era_act")),
                              ("mfg_pol", ev.get("aud_mfg_pol")),
                              ("real_pol", ev.get("aud_real_pol"))):
                if val is None:
                    continue
                print(f"{arm:14s} L{ev['level']} era{ev['era']} {name:8s}={val:.4f} -> "
                      f"era{nxt} competence={real:.4f}  ratio={real / max(val, 1e-6):.2f}")

    print("\n-- mined vs a matched-size random subset of the TRUE table (negative = mined "
          "wins) --")
    for arm, r in arms.items():
        log = r["log"]
        for k, idx in era_slices(log).items():
            for lvl in ("2", "3"):
                d = [(log["aud"][i][lvl].get("cand"), log["aud"][i][lvl].get("rand_k"))
                     for i in idx if lvl in log["aud"][i]]
                d = [(a, b) for a, b in d if a is not None and b is not None]
                if len(d) < 5:
                    continue
                diff = np.array([a - b for a, b in d])
                print(f"{arm:14s} era{k} L{lvl}: mean {diff.mean():+.4f} "
                      f"({int((diff < 0).sum())}/{len(diff)} cycles mined wins)")

    print("\n-- plant guard (clean held-out): parse / infill feature accuracy --")
    for arm, r in arms.items():
        pr = [(p["cycle"], p["plant"]["parse_acc"], p["plant"]["infill_acc"])
              for p in r["log"]["probe"] if "plant" in p]
        if pr:
            print(f"{arm:14s} " + " ".join(f"c{c}:{a:.3f}/{b:.3f}"
                                           for c, a, b in pr[::max(1, len(pr) // 8)]))
    return setup, arms, per, grade


# --------------------------------------------------------------------------- #
# 5b. the offline prediction: what each pacer WOULD have said, on every series
# --------------------------------------------------------------------------- #

def predict(tag, arms=None, setup=None):
    """Replay BOTH pacers over every arm's own logged series, era by era, and put the
    prediction next to what the arm actually did. `pace_cert` should advance exactly where
    the mfg/pol replay fires; `pace_task` exactly where the task-error replay fires; the
    fixed-clock arms show what the same signals would have said had anyone read them."""
    if arms is None:
        setup, arms = load(tag)
    alpha = setup["config"]["alpha"]
    kw = dict(alpha=alpha, c=setup["config"]["sil_c"], cv=setup["config"]["sil_cv"],
              W=setup["config"]["sil_W"], hold=setup["config"]["sil_hold"],
              min_cycle=setup["config"]["sil_min_cycle"],
              min_drop=setup["config"]["lp_min_drop"])
    print("\n-- OFFLINE REPLAY of both pacers on every arm's own series, per era "
          "(c_in_era at which each WOULD fire), against what the arm actually did --")
    print(f"{'arm':14s} {'era':>3s} {'n':>4s} {'cert@':>6s} {'task@':>6s} {'actual':>7s} "
          f"{'reason':>9s} {'cert_span':>9s} {'task_span':>9s}")
    for arm, r in arms.items():
        log = r["log"]
        adv = {ev["from_era"]: ev for ev in r["events"] if ev["kind"] == "advance"}
        for k, idx in era_slices(log).items():
            active = int(log["level"][idx[0]]) + 1
            S = series(log, idx, "mfg", "pol", active)
            E = [log["e"][i] for i in idx]
            fc, _ = replay_unitlp(S, **kw)
            ft, _ = replay_unitlp(E, **kw)
            sv = [x for x in S if x is not None]
            cs = (max(sv) - min(sv)) if sv else float("nan")
            ts = max(E) - min(E)
            a = adv.get(k)
            print(f"{arm:14s} {k:3d} {len(idx):4d} {str(fc):>6s} {str(ft):>6s} "
                  f"{(str(a['c_in_era']) if a else '-'):>7s} "
                  f"{(a['reason'] if a else '-'):>9s} {cs:9.3f} {ts:9.3f}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, setup, arms, per, grade):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    eras = setup["eras"]
    cfg = setup["config"]
    T = float(cfg.get("t_budget", 0.0))
    n_eras = len(eras)
    colors = {a: c for a, c in zip(arms, list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors))}

    # fig1 -- competence against priced time, with each arm's OWN boundaries marked
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for arm, r in arms.items():
        log = r["log"]
        axes[0].plot(log["t_cum"], log["e"], label=arm, color=colors[arm], lw=1.3)
        axes[1].plot(log["cycle"], log["era"], label=arm, color=colors[arm], lw=1.6,
                     alpha=0.8)
        for ev in r["events"]:
            if ev["kind"] == "advance":
                axes[0].axvline(ev["t_cum"], color=colors[arm], ls="--", alpha=0.45)
            if ev["kind"] == "commit":
                axes[0].plot([ev["t_cum"]], [ev["e_task"]], "o", color=colors[arm], ms=5)
    if T > 0:
        axes[0].axvline(T, color="k", lw=1.2)
        axes[0].text(T, axes[0].get_ylim()[1], " budget", va="top", fontsize=7)
    axes[0].set_xlabel("cumulative priced time")
    axes[0].set_ylabel("e = 1 - success @ declared budget, on the CURRENT era")
    axes[0].set_title(f"{tag}: competence (dashed = that arm's own era boundary, "
                      "dot = commit)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("cycle"); axes[1].set_ylabel("era index")
    axes[1].set_yticks(range(1, n_eras + 1),
                       [e["name"] for e in eras])
    axes[1].set_title("the curriculum each policy chose")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_pacing.png"), dpi=130)

    # fig2 -- the two pacing signals, per arm, with the advancement marked
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for arm, r in arms.items():
        log = r["log"]
        idx = np.arange(len(log["cycle"]))
        # the mfg/policy cell always grades the level ACTIVE in that cycle, so it can be read
        # straight off `clim` without knowing which era the cycle belongs to
        S = [log["clim"][i].get("mfg", {}).get("pol") for i in idx]
        axes[0].plot(log["cycle"], [np.nan if x is None else x for x in S],
                     color=colors[arm], label=arm, lw=1.1)
        axes[1].plot(log["cycle"], log["e"], color=colors[arm], label=arm, lw=1.1)
        for ev in r["events"]:
            if ev["kind"] == "advance":
                for ax in axes:
                    ax.axvline(ev["cycle"], color=colors[arm], ls="--", alpha=0.45)
    axes[0].set_ylabel("unit-LP audition (mfg / policy)")
    axes[0].set_title(f"{tag}: the two signals a self-paced agent can read "
                      "(dashed = that arm's advancement)")
    axes[1].set_ylabel("metered task error e (current era)")
    axes[1].set_xlabel("cycle")
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_signals.png"), dpi=130)

    # fig3 -- the matched-budget final exam and the regret against the best fixed schedule
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = [a for a in grade]
    x = np.arange(len(names))
    w = 0.8 / max(n_eras, 1)
    for j in range(n_eras):
        axes[0].bar(x + j * w, [grade[a]["ladder"][j] for a in names], width=w,
                    label=eras[j]["name"])
    axes[0].set_xticks(x + 0.4 - w / 2, names, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("e on that era's held-out set")
    axes[0].set_title("matched-budget final exam: the whole ladder at the same total "
                      "priced time")
    axes[0].legend(fontsize=7)
    fixed = [a for a in grade if a.startswith("fixed_climb") or a.startswith("sched")]
    if fixed:
        best = min(fixed, key=lambda a: grade[a]["deep"])
        reg = [grade[a]["deep"] - grade[best]["deep"] for a in names]
        axes[1].bar(x, reg, color=["tab:grey" if a in fixed else "tab:red" for a in names])
        axes[1].axhline(0, color="k", lw=1)
        axes[1].set_xticks(x, names, rotation=30, ha="right", fontsize=7)
        axes[1].set_ylabel(f"regret on the {eras[-1]['name']} cell")
        axes[1].set_title(f"regret vs the best FIXED schedule ({best})")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_regret.png"), dpi=130)

    # fig4 -- table recall and the cumulative priced cost of the whole grader grid
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for arm, r in arms.items():
        log = r["log"]
        for lvl, ls in ((2, "-"), (3, "--")):
            R = [c.get(str(lvl), {}).get("tab_recall") for c in log["aud"]]
            axes[0].plot(log["cycle"], [np.nan if x is None else x for x in R],
                         color=colors[arm], ls=ls, lw=1.1,
                         label=f"{arm} L{lvl}" if lvl == 2 else None)
        cum, run = [], 0.0
        for g in log["gcost"]:
            run += sum(c["ground"] * cfg["d_fb"] + c["mat"] * cfg["c_mat"]
                       for c in g.values())
            cum.append(run)
        axes[1].plot(log["t_cum"], cum, color=colors[arm], label=arm, lw=1.2)
    axes[0].set_title("earned-table recall against the DGP's vocabulary (solid L2, dashed L3)")
    axes[0].set_xlabel("cycle"); axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=6)
    axes[1].set_title("cumulative cost of the whole grader grid (priced units)")
    axes[1].set_xlabel("cumulative priced time"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_vocab_cost.png"), dpi=130)

    # fig5 -- the schedule-sweep profile (does later-is-better turn over?) and, beside it, the
    # vocabulary pacer's own series against the certificate's, era by era
    has_v = any(r["log"].get("vcert") for r in arms.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    prof = []
    for arm, r in arms.items():
        if arm in ("never_base", "given"):
            continue
        adv = [e for e in r["events"] if e["kind"] == "advance"]
        if adv and arm in grade:
            prof.append((100 * adv[0]["frac_of_budget"], grade[arm]["ladder"][-1], arm))
    prof.sort()
    sw = [p for p in prof if "sched" in p[2]]
    axes[0].plot([p[0] for p in sw], [p[1] for p in sw], "-o", color="tab:blue",
                 label="fixed schedules")
    for x, y, a in prof:
        axes[0].annotate(a, (x, y), fontsize=7, xytext=(3, 4),
                         textcoords="offset points")
        if "sched" not in a:
            axes[0].plot([x], [y], "*", ms=14, color="tab:red")
    if "given" in grade:
        axes[0].axhline(grade["given"]["ladder"][-1], color="k", ls=":", lw=1.2,
                        label="given (the DGP's own tables)")
    axes[0].set_xlabel("first boundary, % of total priced budget")
    axes[0].set_ylabel(f"e on the {eras[-1]['name']} held-out set")
    axes[0].set_title("does bottom-heaviness turn over?  (star = self-paced)")
    axes[0].legend(fontsize=7)

    if has_v:
        # only the arm that actually READS the series -- every arm logs it as an instrument,
        # but overplotting ten of them (each resetting at its own boundaries) is unreadable
        for arm, r in arms.items():
            log = r["log"]
            if not log.get("vcert") or read_cell(r) is None or "vocab" not in arm:
                continue
            V = [c["A"] for c in log["vcert"]]
            if not any(x is not None for x in V):
                continue
            axes[1].plot(log["cycle"], [np.nan if x is None else x for x in V],
                         color=colors[arm], lw=1.4, label=f"{arm} admission rate")
            A = [c["A"] for c in log["cert"]]
            axes[1].plot(log["cycle"], [np.nan if x is None else x for x in A],
                         color=colors[arm], lw=1.0, ls="--", alpha=0.6,
                         label=f"{arm} unit-LP audition")
            for ev in r["events"]:
                if ev["kind"] == "advance":
                    axes[1].axvline(ev["cycle"], color=colors[arm], ls=":", alpha=0.7)
        axes[1].set_xlabel("cycle")
        axes[1].set_ylabel("series value")
        axes[1].set_title("the vocabulary pacer's series vs the certificate's\n"
                          "(dotted = that arm's advancement)")
        axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_sweep_vocab.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
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
    setup, arms, per, grade = report(a.tag, tail=a.tail)
    if a.figures:
        figures(a.tag, setup, arms, per, grade)


if __name__ == "__main__":
    main()
