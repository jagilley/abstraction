"""Reduce a `fourwall` run.

    python3 rhm/practice/fourwall/analyze_fourwall.py --tag fw_s0 --fetch --figures

Readouts, in the order the round asks for them (the instrument list is the commitment; no
outcome map is pre-registered, and nothing here interprets):

  0. GATE 0 -- is the level-2 identity expensive to infer WITHOUT the wall? The exact
     consistent-set size, a linear probe on the agent's own state, and what a right index buys
     on the DGP's own tables. If the parse is free the wall buys nothing and phase 1 is vacuous.
  1. PHASE-1 PRICED TIME TO COMPETENCE -- the scaffold's value: priced time to each error
     threshold, and phase-1 terminal error, per arm.
  2. THE ROTATION RESPONSE -- error in the 3 cycles before and the 3 after each rotation.
  3. PHASE 2 -- terminal error, deep (L3n1) error, and the priced spend split into practice,
     metering, recert and merge.
  4. LIBRARY KEY CARDINALITY OVER TIME -- the MDL readout. Index classes, stored entries and
     DISTINCT stored programs, per arm, per phase.
  5. RECERT activity, per arm and per phase (`setlist` baseline: 9/62 swaps under its own
     native currency).
  6. MERGE -- candidates, refusals and firings, split phase 1 (where the wall is a perfect key
     and merge should have nothing to do) against phase 2.
  7. THE WALL-PERMUTATION PROBE -- e(wall permuted) - e, over time. On `free_selector` this is
     spontaneous binding; on the keyed arms it is what a rotation would cost right now; on
     `unkeyed` it is a null by construction and reads the instrument's noise floor.
  8. THE FORCED-TRANSFER PROFILE AT THE FIRST ROTATION -- per committed cell, per entry, exact.
  9. INSTRUMENTS -- the plant guard, mining yield, per-cell coverage.
"""

import argparse
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
VOLUME = "rhm-scaling-data"
REMOTE = "rhm_practice_fourwall"
ORDER = ["given_key", "wall_track", "wall_merge", "wall_rekey", "scratch_rekey",
         "rekey_mothball", "rekey_delete", "unkeyed", "free_selector"]


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
    return (" " * max(w - 1, 0) + ".") if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:{w}.{n}f}"


def phases(log):
    ph = np.asarray(log["phase"], int)
    return np.nonzero(ph == 1)[0], np.nonzero(ph == 2)[0]


def rotations(setup):
    q = [r["q"] for r in setup["schedule"]]
    return [setup["schedule"][i]["cycle"] for i in range(1, len(q)) if q[i] != q[i - 1]]


# --------------------------------------------------------------------------- #

def report(tag, setup, arms):
    cfg, refs = setup["config"], setup["refs"]
    rots = rotations(setup)
    print(f"\n########## fourwall / {tag} ##########")
    print(f"arms {list(arms)}  era {cfg['era']}  key L{cfg['key_level']}n{cfg['key_node']}  "
          f"phase1 {cfg['phase1']}  period {cfg['rot_period']}  cycles {cfg['max_cycles']}")
    print(f"rotations at cycles {rots}")

    # -- 0. GATE 0 -------------------------------------------------------------------
    print("\n=== 0. GATE 0: is the key expensive to infer without the wall? ===")
    print(f"  key entropy {refs['key_entropy']:.3f} of {np.log(cfg['v']):.3f} nats;  "
          f"mass {json.dumps({k: round(x, 3) for k, x in refs['key_mass'].items()})}")
    print(f"  mean # features that repair: {refs['mean_consistent']:.3f} of {cfg['v']}   "
          f"(a random feature repairs {refs['p_random_feature']:.3f}; "
          f"uniquely determined {refs['frac_uniquely_determined']:.3f})")
    print(f"  linear probe on the agent's state {refs['probe']['acc']:.3f} "
          f"(majority {refs['probe']['majority']:.3f});  "
          f"unkeyed true-table DP recovers f {refs['id_acc_unkeyed_dp']:.3f}")
    print(f"  ONE ACTION  e(true unkeyed) {refs['e_act_true_unkeyed']:.4f}  ->  "
          f"e(true KEYED) {refs['e_act_true_keyed']:.4f}")
    for sfx, name in (("", "no NOOP"), ("_noop", "with NOOP")):
        if f"e_pol_base{sfx}" not in refs:
            continue
        print(f"  POLICY [{name:9s}] base {refs['e_pol_base' + sfx]:.4f}  "
              f"true-unkeyed {refs['e_pol_true_unkeyed' + sfx]:.4f}  "
              f"true-KEYED {refs['e_pol_true_keyed' + sfx]:.4f}   "
              f"index buys {refs['e_pol_true_unkeyed' + sfx] - refs['e_pol_true_keyed' + sfx]:+.4f}")
    print(f"  floor(base exact DP) {refs['floor_base']:.4f}   d0 {refs['d0']:.3f}   "
          f"on_grammar {refs['on_grammar']:.3f}")

    # -- 1. phase-1 priced time to competence ----------------------------------------
    print("\n=== 1. PHASE-1 PRICED TIME TO COMPETENCE (the scaffold's value) ===")
    thr = [0.60, 0.50, 0.40, 0.30, 0.20]
    print(f"{'arm':16s} " + " ".join(f"t(e<={t:.2f})".rjust(12) for t in thr)
          + f" {'e_p1_last5':>11s} {'t_p1':>11s}")
    for arm, r in arms.items():
        log = r["log"]
        i1, _ = phases(log)
        e = np.asarray(log["e"])[i1]
        t = np.asarray(log["t_cum"])[i1]
        cells = []
        for th in thr:
            hit = np.nonzero(e <= th)[0]
            cells.append(f"{t[hit[0]]:12.3g}" if hit.size else " " * 11 + ".")
        print(f"{arm:16s} " + " ".join(cells)
              + f" {e[-5:].mean():11.4f} {t[-1]:11.3g}")

    # -- 2. the rotation response ------------------------------------------------------
    print("\n=== 2. THE ROTATION RESPONSE (mean e over 3 cycles before / after) ===")
    print(f"{'arm':16s} " + " ".join(f"c{c}".rjust(14) for c in rots))
    for arm, r in arms.items():
        log = r["log"]
        cyc = np.asarray(log["cycle"]); e = np.asarray(log["e"])
        cells = []
        for c in rots:
            pre = e[(cyc >= c - 3) & (cyc < c)]
            post = e[(cyc >= c) & (cyc < c + 3)]
            cells.append(f"{pre.mean():.3f}->{post.mean():.3f}".rjust(14)
                         if pre.size and post.size else " " * 13 + ".")
        print(f"{arm:16s} " + " ".join(cells))

    # -- 3. phase 2 --------------------------------------------------------------------
    print("\n=== 3. PHASE 2: terminal error, deep error, and where the price went ===")
    print(f"{'arm':16s} {'e_p2_mean':>10s} {'e_last10':>9s} {'e_deep_p2':>10s} "
          f"{'t_total':>11s} {'recert%':>8s} {'merge%':>7s} {'rekey%':>7s} "
          f"{'n_solved/cyc':>13s}")
    for arm, r in arms.items():
        log = r["log"]
        _i1, i2 = phases(log)
        e = np.asarray(log["e"])
        deep = [p["e_deep"] for p in log["probe"] if p["cycle"] > setup["config"]["phase1"]
                and "e_deep" in p]
        tot = log["t_cum"][-1]

        def spend(key):
            return sum(g.get(key, {}).get("ground", 0) + 0.05 * g.get(key, {}).get("mat", 0)
                       for g in log["gcost"])

        print(f"{arm:16s} {e[i2].mean():10.4f} {e[-10:].mean():9.4f} "
              f"{(np.mean(deep) if deep else np.nan):10.4f} {tot:11.4g} "
              f"{100 * spend('recert') / tot:8.2f} {100 * spend('merge') / tot:7.2f} "
              f"{100 * spend('rekey') / tot:7.2f} {np.mean(log['n_solved']):13.1f}")

    # -- 3b. the lifetime priced-performance integral -----------------------------------
    print("\n=== 3b. LIFETIME PRICED-PERFORMANCE INTEGRAL  A = SUM (1-e) dt ===")
    print(f"{'arm':16s} {'A_total':>12s} {'A/t (mean succ)':>17s} {'A_phase1':>12s} "
          f"{'A_phase2':>12s} {'A2/t2':>9s}")
    for arm, r in arms.items():
        log = r["log"]
        t = np.asarray(log["t_cum"], float)
        e = np.asarray(log["e"], float)
        dt = np.diff(np.concatenate([[0.0], t]))
        a = (1.0 - e) * dt
        i1, i2 = phases(log)
        print(f"{arm:16s} {a.sum():12.5g} {a.sum() / t[-1]:17.4f} "
              f"{a[i1].sum():12.5g} {a[i2].sum():12.5g} "
              f"{a[i2].sum() / dt[i2].sum():9.4f}")

    # -- 4. library key cardinality ------------------------------------------------------
    print("\n=== 4. LIBRARY KEY CARDINALITY OVER TIME (the MDL readout) ===")
    print(f"{'arm':16s} {'wall p1end':>10s} {'wall end':>8s} {'earned end':>10s} "
          f"{'committed':>9s} {'entries':>7s} {'distinct':>8s} {'stored/served':>13s} "
          f"{'merges':>6s} {'rekeys':>6s}")
    for arm, r in arms.items():
        log = r["log"]
        i1, _i2 = phases(log)
        a, z = log["lib"][i1[-1]], log["lib"][-1]
        ratio = (z["n_entries_total"] / z["n_entries_distinct"]) if z["n_entries_distinct"] else np.nan
        print(f"{arm:16s} {a['n_classes']:10d} {z['n_classes']:8d} "
              f"{z.get('n_earned', 0):10d} {z['n_committed']:9d} "
              f"{z['n_entries_total']:7d} {z['n_entries_distinct']:8d} "
              f"{_f(ratio, 2, 13)} {z['n_merges']:6d} {z.get('n_rekeys', 0):6d}")

    # -- 4b. the re-key op: migration, alignment, and time-to-basis ----------------------
    if any(r["log"].get("rekey") and any(r["log"]["rekey"]) for r in arms.values()):
        print("\n=== 4b. RE-KEY: index migration, earned-vs-true alignment, time-to-basis ===")
        print(f"{'arm':16s} {'basis@cyc':>9s} {'path':>12s} {'basis@t':>11s} "
              f"{'mig>=.5':>8s} {'mig>=.9':>8s} {'mig end':>8s} {'cover end':>9s} "
              f"{'router acc':>11s} {'ARI earned':>11s} {'gate e_inc->e_earn':>19s}")
        for arm, r in arms.items():
            rk = [x for x in (r["log"].get("rekey") or []) if x]
            if not rk:
                continue
            mig = np.asarray(r["log"].get("mig") or [0.0])
            cyc = np.asarray(r["log"]["cycle"])
            on = [e for e in r["events"] if e["kind"] == "rekey_on"]
            hit = lambda thr: (int(cyc[np.nonzero(mig >= thr)[0][0]])
                               if (mig >= thr).any() else None)
            last = rk[-1]
            gt = (f"{last['e_inc']:.3f}->{last['e_earned']:.3f}"
                  if last.get("e_earned") is not None else ".")
            print(f"{arm:16s} {_f(on[0]['cycle'] if on else None, 0, 9)} "
                  f"{(on[0].get('path', '.') if on else '.'):>12s} "
                  f"{_f(on[0]['t_cum'] if on else None, 4, 11)} "
                  f"{_f(hit(0.5), 0, 8)} {_f(hit(0.9), 0, 8)} {mig[-1]:8.3f} "
                  f"{_f(last.get('n_cover'), 0, 9)} {_f(last.get('router_acc'), 3, 11)} "
                  f"{_f(last.get('ari_earned'), 3, 11)} {gt:>19s}")

        # two DIFFERENT quantities, reported separately (fw_s1 conflated them under one header)
        pr0 = list(arms.values())[0]["log"]["probe"][::4]
        hdr = " ".join(f"c{p['cycle']}".rjust(7) for p in pr0)
        print("\n  (i) ARI of the EFFECTIVE INDEX (what actually routes) vs the true "
              "f-partition:")
        print(f"  {'arm':16s} " + hdr)
        for arm, r in arms.items():
            print(f"  {arm:16s} " + " ".join(_f(p.get("ari_eff"), 3, 7)
                                             for p in r["log"]["probe"][::4]))
        print("\n  (ii) ARI of the EARNED BASIS itself (the router's partition), where one "
              "exists:")
        print(f"  {'arm':16s} " + hdr)
        for arm, r in arms.items():
            pr = r["log"]["probe"][::4]
            if not any("ari_router" in p for p in pr):
                continue
            print(f"  {arm:16s} " + " ".join(_f(p.get("ari_router"), 3, 7) for p in pr))

        # -- the revert net --------------------------------------------------------------
        print("\n  THE REVERT NET (ear precedent: the recert net fired 0 times in 24):")
        print(f"  {'arm':16s} {'checks':>7s} {'reverts':>8s} {'active end':>11s} "
              f"{'mean e_earn-e_inc':>18s}")
        for arm, r in arms.items():
            rc = [e for e in r["events"] if e["kind"] == "revert_check"]
            if not rc:
                continue
            d = [e["e_earned"] - e["e_incumbent"] for e in rc]
            act = r["log"]["lib"][-1].get("n_earned_active", 0)
            print(f"  {arm:16s} {len(rc):7d} {sum(e['reverted'] for e in rc):8d} "
                  f"{act:11d} {float(np.mean(d)):18.4f}")

    # -- 4c. the retire op: rent, fallback, and where the freed budget went ----------------
    if any(any(e["kind"] == "retire" for e in r["events"]) for r in arms.values()):
        print("\n=== 4c. RETIRE: the scaffold's rent, and what the fallback was worth ===")
        print(f"{'arm':16s} {'mode':>9s} {'n_retired':>10s} {'first@cyc':>10s} "
              f"{'last@cyc':>9s} {'wall kept':>10s} {'entries kept':>13s}")
        for arm, r in arms.items():
            rt = [e for e in r["events"] if e["kind"] == "retire"]
            if not rt:
                continue
            z = r["log"]["lib"][-1]
            print(f"{arm:16s} {rt[0]['mode']:>9s} {len(rt):10d} {rt[0]['cycle']:10d} "
                  f"{rt[-1]['cycle']:9d} {z['n_classes']:10d} "
                  f"{z['n_entries_total']:13d}")

        print("\n  SPEND COMPOSITION, before vs after the first retirement "
              "(% of the arm's priced time in that span):")
        keys = ["recert", "rekey", "revert", "merge"]
        print(f"  {'arm':16s} {'span':>7s} " + " ".join(k.rjust(8) for k in keys)
              + f" {'other':>8s} {'t/cycle':>10s} {'e_mean':>8s}")
        for arm, r in arms.items():
            rt = [e for e in r["events"] if e["kind"] == "retire"]
            log = r["log"]
            cyc = np.asarray(log["cycle"])
            e = np.asarray(log["e"])
            sp = log.get("spend") or []
            if not sp:
                continue
            c0 = rt[0]["cycle"] if rt else None
            spans = ([("pre", cyc < c0), ("post", cyc >= c0)] if c0
                     else [("all", np.ones(len(cyc), bool))])
            for name, m in spans:
                tot = sum(sp[i]["total"] for i in np.flatnonzero(m))
                if tot <= 0:
                    continue
                cells = [100 * sum(sp[i].get(k, 0) for i in np.flatnonzero(m)) / tot
                         for k in keys]
                other = 100 - sum(cells)
                print(f"  {arm:16s} {name:>7s} " + " ".join(f"{x:8.2f}" for x in cells)
                      + f" {other:8.2f} {tot / max(m.sum(), 1):10.4g} {e[m].mean():8.4f}")

        print("\n  THE FALLBACK POPULATION (instances the router will not take):")
        print(f"  {'arm':16s} {'frac end':>9s} {'e_mig end':>10s} {'e_fallback end':>15s} "
              f"{'e_fb pre-retire':>16s} {'e_fb post-retire':>17s}")
        for arm, r in arms.items():
            fbs = r["log"].get("fallback") or []
            if not fbs or not any("e_fallback" in f for f in fbs):
                continue
            cyc = np.asarray(r["log"]["cycle"])
            rt = [e for e in r["events"] if e["kind"] == "retire"]
            c0 = rt[0]["cycle"] if rt else None
            pre = [f["e_fallback"] for f, c in zip(fbs, cyc)
                   if "e_fallback" in f and c0 and c < c0 and f["frac"] < 0.5]
            post = [f["e_fallback"] for f, c in zip(fbs, cyc)
                    if "e_fallback" in f and c0 and c >= c0]
            last = fbs[-1]
            print(f"  {arm:16s} {last['frac']:9.3f} {_f(last.get('e_mig'), 4, 10)} "
                  f"{_f(last.get('e_fallback'), 4, 15)} "
                  f"{_f(float(np.mean(pre)) if pre else None, 4, 16)} "
                  f"{_f(float(np.mean(post)) if post else None, 4, 17)}")

    # -- 5. recert ------------------------------------------------------------------------
    print("\n=== 5. RECERT (setlist baseline under its own currency: 9/62 swaps) ===")
    print(f"{'arm':16s} {'p1 swaps/n':>12s} {'p2 swaps/n':>12s} {'mean margin':>12s}")
    for arm, r in arms.items():
        rec = [e for e in r["events"] if e["kind"] == "recert"]
        p1 = [e for e in rec if e["cycle"] <= setup["config"]["phase1"]]
        p2 = [e for e in rec if e["cycle"] > setup["config"]["phase1"]]
        mar = [e["e_frozen"] - e["e_live"] for e in rec if "e_live" in e]
        c1 = "%d/%d" % (sum(bool(e["swapped"]) for e in p1), len(p1))
        c2 = "%d/%d" % (sum(bool(e["swapped"]) for e in p2), len(p2))
        print(f"{arm:16s} {c1:>12s} {c2:>12s} "
              f"{_f(float(np.mean(mar)) if mar else None, 4, 12)}")

    # -- 6. merge --------------------------------------------------------------------------
    print("\n=== 6. MERGE (phase 1 is the negative control: the wall is a perfect key there) ===")
    for arm, r in arms.items():
        fired = [e for e in r["events"] if e["kind"] == "merge"]
        ref = [e for e in r["events"] if e["kind"] == "merge_refused"]
        if not fired and not ref:
            continue
        p1f = [e for e in fired if e["cycle"] <= setup["config"]["phase1"]]
        print(f"  {arm}: fired {len(fired)} (phase1 {len(p1f)}), audition-refused {len(ref)}")
        for e in fired:
            print(f"    c{e['cycle']:>4} q{e['q']}  {e['a']}+{e['b']}->{e['into']}  "
                  f"loss {e['loss']:+.3f}  x_ab {e['x_ab']:.2f} x_bb {e['x_bb']:.2f}  "
                  f"e_keep {_f(e.get('e_keep'), 3)} e_merge {_f(e.get('e_merge'), 3)}  "
                  f"members {e['members_after']}")
        for e in ref[:8]:
            print(f"    c{e['cycle']:>4} q{e['q']}  REFUSED {e['a']}+{e['b']}  "
                  f"e_keep {_f(e.get('e_keep'), 3)} e_merge {_f(e.get('e_merge'), 3)}")

    # -- 7. the wall-permutation probe -------------------------------------------------------
    print("\n=== 7. WALL-PERMUTATION PROBE: e(wall permuted) - e ===")
    print(f"{'arm':16s} {'phase1 mean':>12s} {'phase1 last':>12s} {'phase2 mean':>12s}")
    for arm, r in arms.items():
        pr = r["log"]["probe"]
        d1 = [p["e_wall_permuted"] - p["e"] for p in pr
              if p["cycle"] <= setup["config"]["phase1"]]
        d2 = [p["e_wall_permuted"] - p["e"] for p in pr
              if p["cycle"] > setup["config"]["phase1"]]
        print(f"{arm:16s} {_f(float(np.mean(d1)) if d1 else None, 4, 12)} "
              f"{_f(d1[-1] if d1 else None, 4, 12)} "
              f"{_f(float(np.mean(d2)) if d2 else None, 4, 12)}")

    # -- 8. the forced-transfer profile at the first rotation ----------------------------------
    print("\n=== 8. FORCED-TRANSFER PROFILE AT THE FIRST ROTATION (exact, per entry) ===")
    for arm, r in arms.items():
        ev = [e for e in r["events"] if e["kind"] == "rotation"]
        if not ev:
            continue
        e0 = ev[0]
        print(f"  {arm} (c{e0['cycle']}, q {e0['q_from']}->{e0['q_to']}):")
        for row in e0["profile"]:
            print(f"    cls {row['cls']:>2}  n {row.get('n_pre', 0):>4}->{row.get('n_post', 0):<4}"
                  f"  any {_f(row.get('any_pre'), 3)}->{_f(row.get('any_post'), 3)}"
                  f"  per-entry pre {[round(z, 2) for z in row.get('per_entry_pre', [])]}"
                  f"  post {[round(z, 2) for z in row.get('per_entry_post', [])]}")

    # -- 9. instruments --------------------------------------------------------------------
    print("\n=== 9. INSTRUMENTS ===")
    print(f"{'arm':16s} {'parse first/last':>18s} {'infill first/last':>19s} "
          f"{'mined/cyc':>10s} {'cover end':>10s}")
    for arm, r in arms.items():
        log = r["log"]
        pl = [p["plant"] for p in log["probe"]]
        cov = list(log["cover"][-1].values())
        par = "%.3f/%.3f" % (pl[0]["parse_acc"], pl[-1]["parse_acc"])
        inf = "%.3f/%.3f" % (pl[0]["infill_acc"], pl[-1]["infill_acc"])
        mined = float(np.mean(log["n_mined"]))
        print(f"{arm:16s} {par:>18s} {inf:>19s} {mined:10.2f} "
              f"{_f(float(np.mean(cov)) if cov else None, 3, 10)}")


# --------------------------------------------------------------------------- #

def figures(tag, setup, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = os.path.join(FIG, tag)
    cfg = setup["config"]
    rots = rotations(setup)
    colors = {a: c for a, c in zip(arms, plt.cm.tab10.colors)}

    def mark(ax):
        for c in rots:
            ax.axvline(c, color="0.85", lw=0.8, zorder=0)
        ax.axvline(cfg["phase1"] + 0.5, color="0.35", lw=1.2, ls=":")

    # fig1 -- competence, and the deep probe
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for arm, r in arms.items():
        log = r["log"]
        axes[0].plot(log["cycle"], log["e"], label=arm, color=colors[arm], lw=1.3)
        for e in r["events"]:
            if e["kind"] == "commit":
                axes[0].plot([e["cycle"]], [log["e"][e["cycle"] - 1]], ".",
                             color=colors[arm], ms=5)
        pr = log["probe"]
        axes[1].plot([p["cycle"] for p in pr], [p["e_deep"] for p in pr], "-",
                     color=colors[arm], lw=1.3, label=arm)
    for k, ax in enumerate(axes):
        mark(ax); ax.set_xlabel("cycle"); ax.legend(fontsize=7)
    refs = setup["refs"]
    axes[0].axhline(refs["e_pol_true_keyed"], color="k", ls="--", lw=0.9)
    axes[0].axhline(refs["e_pol_true_unkeyed"], color="k", ls=":", lw=0.9)
    axes[0].set_ylabel("e (dot = a cell commits)")
    axes[0].set_title(f"{tag}: competence on a FIXED held-out set "
                      f"(-- true keyed, : true unkeyed)")
    axes[1].set_ylabel("e on L3n1 damage")
    axes[1].set_title("the deep probe (never trained on, never priced)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_competence.png"), dpi=130)

    # fig2 -- the MDL readout
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for arm, r in arms.items():
        log = r["log"]
        for j, k in enumerate(("n_classes", "n_entries_total", "n_entries_distinct")):
            axes[j].plot(log["cycle"], [x[k] for x in log["lib"]], color=colors[arm],
                         lw=1.4, label=arm)
    for j, k in enumerate(("index classes", "stored entries (total)",
                           "distinct programs stored")):
        mark(axes[j]); axes[j].set_xlabel("cycle"); axes[j].set_ylabel(k)
        axes[j].set_title(k); axes[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_index.png"), dpi=130)

    # fig3 -- the forced-transfer profile at the first rotation
    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (arm, r) in zip(axes, arms.items()):
        ev = [e for e in r["events"] if e["kind"] == "rotation"]
        if not ev:
            ax.set_title(f"{arm}: no rotation event"); continue
        rows = ev[0]["profile"]
        cls = [row["cls"] for row in rows]
        pre = [row.get("any_pre", np.nan) for row in rows]
        post = [row.get("any_post", np.nan) for row in rows]
        idx = np.arange(len(cls))
        ax.bar(idx - 0.2, pre, 0.4, label="its old demand", color="0.35")
        ax.bar(idx + 0.2, post, 0.4, label="the demand it is handed", color="tab:red")
        ax.set_xticks(idx); ax.set_xticklabels(cls); ax.set_ylim(0, 1.05)
        ax.set_xlabel("index class"); ax.set_ylabel("fraction its table repairs")
        ax.set_title(f"{arm} @ c{ev[0]['cycle']}"); ax.legend(fontsize=7)
    fig.suptitle("FORCED TRANSFER at the first rotation (exact, model-free)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_transfer.png"), dpi=130)

    # fig4 -- the wall-permutation probe and the priced grader
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for arm, r in arms.items():
        pr = r["log"]["probe"]
        axes[0].plot([p["cycle"] for p in pr],
                     [p["e_wall_permuted"] - p["e"] for p in pr], "-",
                     color=colors[arm], lw=1.4, label=arm)
        log = r["log"]
        spend = np.cumsum([g.get("recert", {}).get("ground", 0)
                           + g.get("merge", {}).get("ground", 0) for g in log["gcost"]])
        axes[1].plot(log["cycle"], 100 * spend / np.asarray(log["t_cum"]),
                     color=colors[arm], lw=1.4, label=arm)
    for ax in axes:
        mark(ax); ax.set_xlabel("cycle"); ax.legend(fontsize=7)
    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[0].set_ylabel("e(wall permuted) - e")
    axes[0].set_title("SPONTANEOUS BINDING: what permuting the wall costs")
    axes[1].set_ylabel("% of priced time")
    axes[1].set_title("the priced maintenance layer (recert + merge)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_probe_spend.png"), dpi=130)

    # fig5 -- the re-key op: migration curve and earned-vs-true alignment
    if any(r["log"].get("rekey") and any(r["log"]["rekey"]) for r in arms.values()):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for arm, r in arms.items():
            log = r["log"]
            if log.get("mig"):
                axes[0].plot(log["cycle"], log["mig"], color=colors[arm], lw=1.5, label=arm)
            pr = log["probe"]
            axes[1].plot([p["cycle"] for p in pr],
                         [p.get("ari_router", p.get("ari_eff", np.nan)) for p in pr],
                         color=colors[arm], lw=1.5, label=arm)
            rk = [x for x in (log.get("rekey") or []) if x
                  and x.get("e_earned") is not None]
            if rk:
                axes[2].plot([x["cycle"] for x in rk], [x["e_earned"] for x in rk],
                             "-", color=colors[arm], lw=1.5, label=f"{arm} earned")
                axes[2].plot([x["cycle"] for x in rk], [x["e_inc"] for x in rk],
                             "--", color=colors[arm], lw=1.2, alpha=0.7,
                             label=f"{arm} incumbent")
        for ax in axes:
            mark(ax); ax.set_xlabel("cycle"); ax.legend(fontsize=7)
        axes[0].set_ylabel("fraction routed by an earned class")
        axes[0].set_title("INDEX MIGRATION: wall-classes -> earned classes")
        axes[1].set_ylabel("ARI vs the true f-partition"); axes[1].axhline(0, color="0.6", lw=0.8)
        axes[1].set_title("alignment of the EARNED basis with the truth (oracle readout)")
        axes[2].set_ylabel("held-out policy error"); axes[2].set_ylim(0, 1.05)
        axes[2].set_title("THE MIGRATION GATE, graded in consumption: "
                          "earned (solid) vs incumbent (dashed)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_rekey.png"), dpi=130)
    print(f"\nfigures -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.tag)
    setup, arms = load(a.tag)
    report(a.tag, setup, arms)
    if a.figures:
        figures(a.tag, setup, arms)


if __name__ == "__main__":
    main()
