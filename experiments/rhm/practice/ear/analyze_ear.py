"""Reduce an `ear` run.

    python3 rhm/practice/ear/analyze_ear.py --tag er_s0 --fetch --figures
    python3 rhm/practice/ear/analyze_ear.py --tag calm_s0 --fetch --cal
    python3 -c "import sys; sys.path.insert(0,'rhm/practice/ear'); import analyze_ear as A; \
        A.grader_grid('cald_s0', level=3, era=2)"

Readouts, in the order the round asks for them:
  0. FIDELITY -- do `never_base` / `given` / `practice_gated` reproduce `rr_s0` cycle for cycle
  1. priced competence per depth era, the earned-vs-given fraction, matched-priced-time margins
  2. commits and RECERTS: when, on what table, provisional or certified, and whether a live
     regrade swapped the committed vocabulary
  3. THE GRADER GRID -- the (ctx x ev) audition series per arm: what each grader was saying,
     including the ones the arm did not read, so "what would grader X have done" is an offline
     replay rather than another run
  4. THE ORACLE CHECK -- does the self-manufactured context distribution match the real one, by
     signature (on-grammar / d* / broken / node-empty) and by the thing that matters, its
     audition reading against the real set's
  5. GRADER COST -- what the evaluation layer spent, as priced time and as a share of it
  6. the seam (commit audition vs realised competence one era on), compounding, mined-vs-random,
     and the plant guard
"""

import argparse
import itertools
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RATCHET_FIG = os.path.join(os.path.dirname(HERE), "ratchet", "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_ear"
ORDER = ["never_base", "given", "practice_gated", "practice_late", "practice_self",
         "taught", "practice_prov", "practice_climb"]
GRADERS = [("era", "act"), ("era", "pol"), ("mfg", "act"), ("mfg", "pol"),
           ("real", "act"), ("real", "pol")]


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


def interp(t_series, y_series, t_at):
    t, y = np.asarray(t_series, float), np.asarray(y_series, float)
    if t_at <= t[0]:
        return float(y[0])
    if t_at >= t[-1]:
        return float(y[-1])
    return float(np.interp(t_at, t, y))


def era_slices(log):
    era = np.asarray(log["era"], int)
    return {int(k): np.nonzero(era == k)[0] for k in np.unique(era)}


def series(log, idx, ctx, ev, active):
    """One cell of the grader grid over an era's cycles. `ev` in {act, pol, pol_base,
    act_true, margin}; `ctx` in {era, mfg, real}. The era-k/action cell lives in the ratchet's
    own `aud` block (a 512-instance set), everything else in `clim` (n_aud)."""
    out = []
    for i in idx:
        if ctx == "era" and ev == "act":
            out.append(log["aud"][i].get(str(active), {}).get("cand"))
        elif ev == "neg_margin":
            c = log["clim"][i].get(ctx, {})
            mg = c.get("margin")
            out.append(None if mg is None else -mg)
        else:
            out.append(log["clim"][i].get(ctx, {}).get(ev))
    return out


# --------------------------------------------------------------------------- #
# 0. the fork's fidelity gate
# --------------------------------------------------------------------------- #

def fidelity(tag, ref_tag="rr_s0", arms=("never_base", "given", "practice_gated"), n=30):
    """Arms whose code path is the ratchet's, run first and with a dedicated RNG stream for
    everything new, should reproduce `rr_s0` cycle for cycle. Anything else means the
    evaluation layer leaked into the substrate."""
    print("\n-- FIDELITY: this fork's ratchet-path arms against ratchet/rr_s0 --")
    _, mine = load(tag)
    rows = []
    for arm in arms:
        p = os.path.join(RATCHET_FIG, ref_tag, arm, "results.json")
        if arm not in mine or not os.path.isfile(p):
            continue
        ref = json.load(open(p))["log"]
        cur = mine[arm]["log"]
        k = min(n, len(ref["e"]), len(cur["e"]))
        de = np.abs(np.asarray(ref["e"][:k]) - np.asarray(cur["e"][:k]))
        dt = np.abs(np.asarray(ref["t_cum"][:k]) - np.asarray(cur["t_cum"][:k]))
        rows.append((arm, k, float(de.max()), float(dt.max())))
        print(f"{arm:16s} first {k:3d} cycles: max|de|={de.max():.2e}  "
              f"max|dt_cum|={dt.max():.2e}  "
              f"{'IDENTICAL' if de.max() == 0 and dt.max() == 0 else 'DIVERGES'}")
    if not rows:
        print("   (no reference run on disk -- run analyze_ratchet.py --tag rr_s0 --fetch)")
    return rows


# --------------------------------------------------------------------------- #
# the main report
# --------------------------------------------------------------------------- #

def report(tag, tail=5):
    setup, arms = load(tag)
    eras, refs = setup["eras"], setup["refs"]
    cfg = setup["config"]
    print(f"\n=== {tag} ===")
    print("eras: " + " | ".join(
        f"{e['name']} d0={refs['d0'][i]:.2f} stale_base={refs['stale'][i]:.3f} "
        f"stale_true={refs['stale_true'][i]:.3f} floor={refs['floor'][i]:.3f} "
        f"macro_true={refs['macro_true'][i]}" for i, e in enumerate(eras)))
    print(f"declared budget: base w={refs['width_base']} ({refs['g_base']}g), "
          f"true w={refs['width_true']} ({refs['g_true']}g); n_aud={cfg['n_aud']} "
          f"mfg_source={cfg.get('mfg_source')} mfg_render={cfg.get('mfg_render')} "
          f"recert_every={cfg['recert_every']} margin={cfg['recert_margin']}")

    # ---- 1. priced competence per era -------------------------------------------------
    print(f"\n-- per-era competence (mean e over the last {tail} cycles) and priced time --")
    per = {}
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        e = np.asarray(log["e"], float)
        t = np.asarray(log["t_cum"], float)
        per[arm] = {k: {"e": float(e[idx[-tail:]].mean()), "e_last": float(e[idx[-1]]),
                        "t_end": float(t[idx[-1]]),
                        "t_era": float(t[idx[-1]] - (t[idx[0] - 1] if idx[0] else 0.0)),
                        "nm": int(log["n_moves"][idx[-1]]), "w": int(log["width"][idx[-1]])}
                    for k, idx in sl.items()}
        per[arm]["_t"] = t
        per[arm]["_e"] = e
        per[arm]["_complete"] = r["complete"]
    ks = sorted(era_slices(list(arms.values())[0]["log"]).keys())
    print(f"{'arm':16s} " + " ".join(f"{'era' + str(k):>26s}" for k in ks) + f"{'t_cum':>12s}")
    for arm, x in per.items():
        cells = " ".join(f"  e={x[k]['e']:.4f} (t={x[k]['t_era']:9.0f}, nm{x[k]['nm']}w{x[k]['w']})"
                         for k in ks)
        print(f"{arm:16s} {cells} {x[ks[-1]]['t_end']:12.0f}"
              + ("" if x["_complete"] else "  [INCOMPLETE]"))

    if "never_base" in per and "given" in per:
        print("\n-- EARNED-VS-GIVEN fraction:  (e_never - e_arm) / (e_never - e_given) --")
        print(f"{'arm':16s} " + " ".join(f"{'era' + str(k):>10s}" for k in ks))
        for arm in per:
            if arm in ("never_base", "given"):
                continue
            row = []
            for k in ks:
                den = per["never_base"][k]["e"] - per["given"][k]["e"]
                row.append((per["never_base"][k]["e"] - per[arm][k]["e"]) / den
                           if abs(den) > 1e-9 else float("nan"))
            print(f"{arm:16s} " + " ".join(f"{x:10.3f}" for x in row))
        print("   (denominators: " + " ".join(
            f"era{k}={per['never_base'][k]['e'] - per['given'][k]['e']:+.4f}" for k in ks) + ")")

        print("\n-- matched priced time (each arm's e vs never_base's own trajectory at t) --")
        for arm, x in per.items():
            if arm == "never_base":
                continue
            cells = []
            for k in ks:
                nb = interp(per["never_base"]["_t"], per["never_base"]["_e"], x[k]["t_end"])
                cells.append(f"era{k} {nb - x[k]['e_last']:+.4f}")
            print(f"{arm:16s} " + "  ".join(cells))

    # ---- 2. commits and recerts --------------------------------------------------------
    print("\n-- commit events --")
    print(f"{'arm':16s} {'era':>3s} {'lvl':>3s} {'cyc':>4s} {'c_in':>4s} {'prov':>5s} "
          f"{'entries':>7s} {'recall':>7s} {'prec':>6s} {'A_read':>7s} {'era_act':>7s} "
          f"{'mfg_pol':>7s} {'real_pol':>8s} {'sil':>4s}")
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "commit":
                continue
            f = lambda x: "    .  " if x is None else f"{x:7.4f}"
            print(f"{arm:16s} {ev['era']:3d} {ev['level']:3d} {ev['cycle']:4d} "
                  f"{ev['c_in_era']:4d} {str(ev.get('provisional')):>5s} "
                  f"{ev['n_entries']:7d} {ev['tab_recall']:7.3f} "
                  f"{(ev['tab_precision'] if ev['tab_precision'] is not None else float('nan')):6.3f} "
                  f"{f(ev['shadow'])} {f(ev.get('aud_era_act'))} {f(ev.get('aud_mfg_pol'))} "
                  f"{f(ev.get('aud_real_pol'))[:8]:>8s} {ev['sil_run']:4d}")

    print("\n-- RECERT events (the frozen committed table vs the live mined one, POLICY "
          "evaluator, in the era where the unit is consumed) --")
    print(f"{'arm':16s} {'era':>3s} {'lvl':>3s} {'cyc':>4s} {'e_frozen':>8s} {'e_live':>8s} "
          f"{'d_e':>7s} {'n_fro':>5s} {'n_liv':>5s} {'ndiff':>5s} {'liv_rec':>7s} {'swap':>5s}")
    n_swap, deltas = {}, []
    for arm, r in arms.items():
        for ev in r["events"]:
            if ev["kind"] != "recert":
                continue
            n_swap[arm] = n_swap.get(arm, 0) + int(bool(ev["swapped"]))
            el = ev.get("e_live")
            if el is not None:
                deltas.append(ev["e_frozen"] - el)
            s_live = "       ." if el is None else "%8.4f" % el
            s_d = "      ." if el is None else "%+7.4f" % (ev["e_frozen"] - el)
            s_rec = ev.get("live_recall")
            s_rec = "      ." if s_rec is None else "%7.3f" % s_rec
            print(f"{arm:16s} {ev['era']:3d} {ev['level']:3d} {ev['cycle']:4d} "
                  f"{ev['e_frozen']:8.4f} {s_live} {s_d} "
                  f"{ev['n_frozen']:5d} {ev['n_live']:5d} "
                  f"{(ev.get('n_diff') if ev.get('n_diff') is not None else -1):5d} "
                  f"{s_rec} {str(ev['swapped']):>5s}")
    if deltas:
        d = np.asarray(deltas)
        print(f"   frozen-minus-live over all recerts: mean={d.mean():+.4f} sd={d.std():.4f} "
              f"max={d.max():+.4f}  (margin={cfg['recert_margin']})")
    if n_swap:
        print("   swaps per arm: " + ", ".join(f"{a}={n}" for a, n in n_swap.items()))

    # ---- 3. the grader grid ------------------------------------------------------------
    print("\n-- THE GRADER GRID: every audition series, per arm and era. `>` marks the cell "
          "this arm's certificate actually read --")
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        gr = {"practice_gated": ("era", "act"), "practice_late": ("era", "act"),
              "practice_self": ("mfg", "pol"), "taught": ("real", "pol"),
              "practice_climb": ("mfg", "pol")}.get(arm)
        fired = {ev["era"]: ev["cycle"] for ev in r["events"] if ev["kind"] == "commit"}
        for k, idx in sl.items():
            active = int(log["level"][idx[0]]) + 1
            rows = []
            for ctx, ev in GRADERS:
                S = series(log, idx, ctx, ev, active)
                if all(x is None for x in S):
                    continue
                mark = ">" if gr == (ctx, ev) else " "
                vals = [x for x in S if x is not None]
                rows.append(f"  {mark}{ctx:>4s}/{ev:<3s} span={max(vals) - min(vals):.3f} "
                            f"min={min(vals):.3f} last5={np.mean(vals[-5:]):.3f} : "
                            + " ".join("  .  " if x is None else f"{x:.2f}" for x in S))
            for ctx in ("era", "mfg", "real"):
                B = series(log, idx, ctx, "pol_base", active)
                vals = [x for x in B if x is not None]
                if vals:
                    rows.append(f"   {ctx:>4s}/base  span={max(vals) - min(vals):.3f} "
                                f"first={vals[0]:.3f} last={vals[-1]:.3f}   "
                                "(no-candidate policy drift -- the confound instrument)")
            if rows:
                print(f"{arm:16s} era{k} (earning L{active})"
                      + (f"  [commit c{fired[k]}]" if k in fired else ""))
                print("\n".join(rows))

    # ---- 4. the oracle check on self-manufactured contexts ------------------------------
    print("\n-- ORACLE CHECK on the self-manufactured audition contexts --")
    print("   signature (real damage is on_grammar 1.000, frac_broken 1.000, node_empty 0.000, "
          "node_overlap 0.000):")
    for arm, r in arms.items():
        for ev in r["log"].get("mfg", []):
            st = ev["stats"]
            print(f"{arm:16s} era{ev['era']} L{ev['level']}n{ev['node']} c{ev['cycle']} "
                  f"n={ev['n']} bank={ev['bank']} lower_entries={ev['lower_entries']}  "
                  f"on_gram={st['on_grammar']:.3f} d*={st['d_mean']:.2f} "
                  f"broken={st['frac_broken']:.3f} node_empty={st['node_empty']:.3f} "
                  f"node_overlap={st.get('node_overlap', float('nan')):.3f}")
    print("\n   agreement, the only way that matters -- the SAME candidate graded on the "
          "manufactured set and on the real one, over the cycles where both exist:")
    for arm, r in arms.items():
        log = r["log"]
        for k, idx in era_slices(log).items():
            active = int(log["level"][idx[0]]) + 1
            for ev in ("pol", "act", "act_true"):
                M = series(log, idx, "mfg", ev, active)
                R = series(log, idx, "real", ev, active)
                pair = [(a, b) for a, b in zip(M, R) if a is not None and b is not None]
                if len(pair) < 4:
                    continue
                a = np.array([p[0] for p in pair]); b = np.array([p[1] for p in pair])
                rho = (float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9
                       else float("nan"))
                print(f"{arm:16s} era{k} L{active} {ev:8s} n={len(pair):3d}  "
                      f"mfg-real bias={np.mean(a - b):+.4f}  mad={np.mean(np.abs(a - b)):.4f}  "
                      f"corr={rho:+.3f}")

    # ---- 5. grader cost -----------------------------------------------------------------
    print("\n-- GRADER COST: what the evaluation layer spent, and what it would have cost --")
    d_fb, c_mat = cfg["d_fb"], cfg["c_mat"]
    for arm, r in arms.items():
        log = r["log"]
        tot = {}
        for g in log["gcost"]:
            for k, c in g.items():
                tot[k] = tot.get(k, 0.0) + c["ground"] * d_fb + c["mat"] * c_mat
        paid_key = {"practice_gated": "era", "practice_late": "era", "never_base": "era",
                    "given": "era", "practice_self": "mfg", "taught": "real",
                    "practice_climb": "mfg"}.get(arm)
        paid = tot.get(paid_key, 0.0) if paid_key else 0.0
        paid += tot.get("recert", 0.0)
        t_end = log["t_cum"][-1]
        print(f"{arm:16s} priced total={t_end:11.0f}  grader PAID={paid:9.0f} "
              f"({100 * paid / max(t_end, 1):.2f}%)   would-have-cost: "
              + " ".join(f"{k}={x:.0f}" for k, x in sorted(tot.items())))

    # ---- 6. the seam, compounding, mined-vs-random, plant guard --------------------------
    print("\n-- the depth seam: what the arm's grader said at commit vs its realised "
          "competence in the era where the unit is consumed --")
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
                print(f"{arm:16s} L{ev['level']} era{ev['era']} {name:8s}={val:.4f} -> "
                      f"era{nxt} competence={real:.4f}  ratio={real / max(val, 1e-6):.2f}")

    print("\n-- compounding: how fast the level-3 audition descends in era 2, per arm --")
    for arm, r in arms.items():
        log = r["log"]
        sl = era_slices(log)
        if 2 not in sl:
            continue
        idx = sl[2]
        for ctx, ev in (("era", "act"), ("mfg", "pol"), ("real", "pol")):
            A = np.array([x if x is not None else np.nan
                          for x in series(log, idx, ctx, ev, 3)], float)
            ok = ~np.isnan(A)
            if ok.sum() < 4:
                continue
            half = A[ok][0] - 0.5 * (A[ok][0] - np.nanmin(A))
            reach = np.nonzero(ok & (A <= half))[0]
            noise = float(np.nanstd(np.diff(A[ok])) / np.sqrt(2))
            print(f"{arm:16s} era2 {ctx}/{ev:3s}: L2 vocab={log['vocab'][idx[0]].get('2')} "
                  f"first={A[ok][0]:.3f} min={np.nanmin(A):.3f} span={A[ok][0] - np.nanmin(A):.3f} "
                  f"step-noise={noise:.3f} half-descent at c_in_era="
                  f"{(reach[0] + 1) if len(reach) else None}")

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
                print(f"{arm:16s} era{k} L{lvl}: mean {diff.mean():+.4f} "
                      f"({int((diff < 0).sum())}/{len(diff)} cycles mined wins)")

    print("\n-- plant guard (clean held-out): parse / infill feature accuracy --")
    for arm, r in arms.items():
        pr = [(p["cycle"], p["plant"]["parse_acc"], p["plant"]["infill_acc"])
              for p in r["log"]["probe"] if "plant" in p]
        if pr:
            print(f"{arm:16s} " + " ".join(f"c{c}:{a:.3f}/{b:.3f}"
                                           for c, a, b in pr[::max(1, len(pr) // 8)]))
    return setup, arms, per


# --------------------------------------------------------------------------- #
# offline detector replay -- the counterfactual "what would grader X have done"
# --------------------------------------------------------------------------- #

def replay_unitlp(A, *, alpha, c, cv, W, hold, min_cycle, min_drop):
    """The unit-LP certificate replayed over a recorded audition series, exactly as `run_arm`
    runs it: silence PLUS the explicit 'it descended first' precondition."""
    b = ref = emin = None
    hist, dh, run = [], [], 0
    for i, a in enumerate(A, start=1):
        if a is None or (isinstance(a, float) and a != a):
            continue
        if ref is None:
            ref = b = emin = a
        emin = min(emin, a)
        d = b - a
        scale = max(ref - emin, b, 1e-6)
        hist.append(a); dh.append(d)
        we, wd = hist[-W:], dh[-W:]
        quiet = (len(we) >= W and abs(float(np.mean(wd))) < c * scale
                 and float(np.std(we)) < cv * scale)
        run = run + 1 if quiet else 0
        b = b + alpha * (a - b)
        if run >= hold and i >= min_cycle and (ref - emin) >= min_drop:
            return i, a
    return None, None


def grader_grid(tags, level=3, era=2, arms=None, grid=True):
    """Replay the certificate over EVERY cell of the grader grid, for every arm -- the
    calibration that costs no GPU and that sets the detector for a climbed grader."""
    for tag in tags.split(","):
        setup, all_arms = load(tag)
        alpha = setup["config"]["alpha"]
        print(f"\n=== grader grid: {tag} (era {era}, earning level {level}) ===")
        for a, r in all_arms.items():
            if arms and a not in arms:
                continue
            log = r["log"]
            idx = np.nonzero(np.asarray(log["era"], int) == era)[0]
            if len(idx) == 0:
                continue
            for ctx, ev in GRADERS:
                S = series(log, idx, ctx, ev, level)
                vals = [x for x in S if x is not None]
                if len(vals) < 6:
                    continue
                noise = float(np.std(np.diff(vals)) / np.sqrt(2))
                print(f"-- {a} [{ctx}/{ev}]: first={vals[0]:.3f} min={min(vals):.3f} "
                      f"span={vals[0] - min(vals):.3f} last5={np.mean(vals[-5:]):.3f} "
                      f"step-noise={noise:.3f}  span/noise={((vals[0] - min(vals)) / max(noise, 1e-9)):.1f}")
                if not grid:
                    continue
                for c, cv, W, hold, md in itertools.product(
                        (0.06, 0.10), (0.10, 0.15, 0.20), (5, 7), (2, 3), (0.05, 0.10, 0.15)):
                    fc, fe = replay_unitlp(S, alpha=alpha, c=c, cv=cv, W=W, hold=hold,
                                           min_cycle=6, min_drop=md)
                    if fc is None:
                        continue
                    print(f"     c={c:.2f} cv={cv:.2f} W={W} hold={hold} drop={md:.2f} -> "
                          f"fire c{fc:2d}  A={fe:.3f}  vs min {fe - min(vals):+.3f}  "
                          f"vs last5 {fe - np.mean(vals[-5:]):+.3f}")


# --------------------------------------------------------------------------- #
# cal_mfg
# --------------------------------------------------------------------------- #

def cal_report(tag):
    root = os.path.join(FIG, tag)
    d = json.load(open(os.path.join(root, "cal_mfg.json")))
    print(f"\n=== {tag} (is a manufactured test a faithful test?) ===")
    print(f"G-R recert: {d['gates']['recert']}")
    for name, cell in d["cells"].items():
        print(f"\n-- era {name} -> auditioning level {cell['level']} at node {cell['node']} "
              f"(consumed in {cell['next_era']}) --")
        ref = cell["sets"].get("real")
        for sname, row in cell["sets"].items():
            st = row["stats"]
            print(f"  {sname:11s} n={row['n']:4d} on_gram={st['on_grammar']:.3f} "
                  f"d*={st['d_mean']:.2f} broken={st['frac_broken']:.3f} "
                  f"node_empty={st['node_empty']:.3f} "
                  f"node_ovl={st.get('node_overlap', float('nan')):.3f}")
            ka = sorted(row["act"], key=int)
            print(f"      act (coverage 1..full): "
                  + " ".join(f"{k}:{row['act'][k]:.3f}" for k in ka))
            print(f"      pol  base={row['pol_base']:.3f} | "
                  + " ".join(f"{k}:{row['pol'][k]:.3f}" for k in ka))
            for ev in ("act", "pol"):
                sp = max(row[ev].values()) - min(row[ev].values())
                print(f"      {ev} span over coverage = {sp:.3f}"
                      + (f"   (real: {max(ref[ev].values()) - min(ref[ev].values()):.3f})"
                         if ref and sname != "real" else ""))
            if ref and sname != "real":
                for ev in ("act", "pol"):
                    a = np.array([row[ev][k] for k in ka])
                    b = np.array([ref[ev][k] for k in ka])
                    rho = (float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9
                           else float("nan"))
                    print(f"      vs real [{ev}]: bias={np.mean(a - b):+.4f} "
                          f"mad={np.mean(np.abs(a - b)):.4f} corr={rho:+.3f}")
    return d


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def figures(tag, setup, arms, per):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    eras = setup["eras"]
    colors = {a: c for a, c in zip(arms, list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors))}
    first = list(arms.values())[0]["log"]
    bounds = np.cumsum([len(era_slices(first)[k]) for k in sorted(era_slices(first))])

    # fig1 -- competence, per cycle and against priced time
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for arm, r in arms.items():
        log = r["log"]
        axes[0].plot(log["cycle"], log["e"], label=arm, color=colors[arm], lw=1.3)
        axes[1].plot(log["t_cum"], log["e"], label=arm, color=colors[arm], lw=1.3)
        for ev in r["events"]:
            if ev["kind"] == "commit":
                axes[0].axvline(ev["cycle"], color=colors[arm], ls=":", alpha=0.5)
            if ev["kind"] == "recert" and ev["swapped"]:
                axes[0].plot([ev["cycle"]], [log["e"][ev["cycle"] - 1]], "*",
                             color=colors[arm], ms=9)
    for b in bounds[:-1]:
        axes[0].axvline(b + 0.5, color="grey", lw=0.8)
    axes[0].set_xlabel("cycle"); axes[0].set_ylabel("e = 1 - success @ declared budget")
    axes[0].set_title(f"{tag}: competence, eras " + " -> ".join(e["name"] for e in eras)
                      + "  (dotted = commit, star = recert swap)")
    axes[1].set_xlabel("cumulative priced time"); axes[1].set_ylabel("e")
    axes[1].set_title("priced cost to competence")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=130)

    # fig2 -- the grader grid: what each grader was saying, per level
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    cells = [("era", "act"), ("mfg", "pol"), ("real", "pol")]
    for row, lvl in enumerate((2, 3)):
        for col, (ctx, ev) in enumerate(cells):
            ax = axes[row][col]
            for arm, r in arms.items():
                log = r["log"]
                idx = np.arange(len(log["cycle"]))
                S = series(log, idx, ctx, ev, lvl)
                ax.plot(log["cycle"], [np.nan if x is None else x for x in S],
                        color=colors[arm], label=arm, lw=1.1)
                for e_ in r["events"]:
                    if e_["kind"] == "commit" and e_["level"] == lvl:
                        ax.axvline(e_["cycle"], color=colors[arm], ls=":", alpha=0.6)
            for b in bounds[:-1]:
                ax.axvline(b + 0.5, color="grey", lw=0.8)
            ax.set_title(f"level-{lvl} candidate, grader = {ctx}/{ev}")
            ax.set_ylabel("audition e")
    for ax in axes[1]:
        ax.set_xlabel("cycle")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_graders.png"), dpi=130)

    # fig3 -- the oracle check: manufactured vs real, same candidate, same cycle
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for arm, r in arms.items():
        log = r["log"]
        idx = np.arange(len(log["cycle"]))
        for j, ev in enumerate(("pol", "act")):
            M = series(log, idx, "mfg", ev, 2) + series(log, idx, "mfg", ev, 3)
            R = series(log, idx, "real", ev, 2) + series(log, idx, "real", ev, 3)
            pts = [(a, b) for a, b in zip(M, R) if a is not None and b is not None]
            if pts:
                axes[j].scatter([p[0] for p in pts], [p[1] for p in pts], s=9,
                                color=colors[arm], label=arm, alpha=0.7)
    for j, ev in enumerate(("pol", "act")):
        axes[j].plot([0, 1], [0, 1], "k--", lw=1)
        axes[j].set_xlabel("audition on SELF-MANUFACTURED contexts")
        axes[j].set_ylabel("audition on REAL level-k+1 contexts")
        axes[j].set_title(f"does the manufactured test agree? ({ev} evaluator)")
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_oracle.png"), dpi=130)

    # fig4 -- table recall, and the grader's priced share
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
            run += sum(c["ground"] * setup["config"]["d_fb"]
                       + c["mat"] * setup["config"]["c_mat"] for c in g.values())
            cum.append(run)
        axes[1].plot(log["cycle"], cum, color=colors[arm], label=arm, lw=1.2)
    axes[0].set_title("earned-table recall against the DGP's own vocabulary "
                      "(solid L2, dashed L3)")
    axes[0].set_xlabel("cycle"); axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=6)
    axes[1].set_title("cumulative cost of the whole grader grid (priced units)")
    axes[1].set_xlabel("cycle"); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_vocab_cost.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--tail", type=int, default=5)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    if a.cal:
        cal_report(a.tag)
        return
    setup, arms, per = report(a.tag, tail=a.tail)
    if a.fidelity:
        fidelity(a.tag)
    if a.figures:
        figures(a.tag, setup, arms, per)


if __name__ == "__main__":
    main()
