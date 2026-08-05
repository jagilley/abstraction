"""Read the composed-loop runs and print the 2x2, its interaction, and the collapse check.

Usage from experiments/:
    python3 rhm/directed_sculpting/full_loop/composed_loop/aggregate.py
    python3 rhm/directed_sculpting/full_loop/composed_loop/aggregate.py --pattern 'composed_loop_lv_*'

`--pattern` is REQUIRED to be explicit about which runs are being pooled, for the reason
`level_moves/aggregate.py` records: mirroring two sweeps into `figures/` and keying results by
seed silently overwrote cells and produced a table that looked complete.

READING ORDER, and it is not optional
--------------------------------------
1. THE COLLAPSE CHECK first. If `endo_rollout`'s `roll_vs_fm_pred` rank correlation sits at the
   `fm_pred_vs_fm_pred_fresh` homogeneous floor, the outer loop is ranking moves like the inner
   loop and any gain that arm shows is an inner-loop gain wearing an outer-loop label. The
   published floor is +0.946 with the external teacher at -0.220 and the endogenous one at
   -0.130; a number near the floor voids the arm.
2. THE CONTENT CONTROL second. `endo_random` at or below `frozen` on ballistic is what makes the
   other arms readable. `endo_expansion` §3 found a zero-information target topping the PR table
   at 145% of the DP teacher while sitting at the no-loop floor on control.
3. Only then the 2x2, on BALLISTIC and FRESH-FM TOP1. PR is printed and is NOT a ranker.
"""

import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_ORDER = ("frozen", "dense", "evaluative", "endo_rollout", "endo_random")


def load(pattern):
    runs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "figures", pattern, "results.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[os.path.basename(os.path.dirname(path))] = r
    return runs


def graded_rows(arm_res):
    return [row for row in arm_res["rounds"] if "grades" in row]


def final_grade(arm_res, depth):
    """Last graded round's readouts at damage depth `depth`."""
    g = graded_rows(arm_res)
    if not g or str(depth) not in g[-1]["grades"]:
        return None
    return g[-1]["grades"][str(depth)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="composed_loop_*")
    args = ap.parse_args()

    runs = load(args.pattern)
    if not runs:
        print(f"no runs matching figures/{args.pattern}/results.json")
        return

    by_level = {}
    for name, r in runs.items():
        by_level.setdefault(r["config"]["max_level"], []).append((name, r))
    levels = sorted(by_level)
    seeds = {r["config"]["seed"] for _, rs in by_level.items() for _, r in rs}
    print(f"runs: {list(runs)}\naction spaces: {levels}   seeds: {sorted(seeds)}")
    if len(seeds) == 1:
        print("*** SINGLE SEED -- every number below is one draw. No SEs, no t-stats, and no\n"
              "*** contrast here is resolvable against run-to-run variance until seeds land.")

    ref = next(iter(runs.values()))
    sched_levels = ref["config"]["sched_levels"]
    L = ref["config"]["tree_depth"]

    # ---- 1. the collapse check ---------------------------------------------------------
    print(f"\n{'=' * 78}\n1. COLLAPSE CHECK -- does the outer signal rank moves like the "
          f"inner one?\n{'=' * 78}")
    print(f"{'action space':>14} {'arm':>14} {'roll_vs_fm':>12} {'dp_vs_fm':>10} "
          f"{'HOMOG FLOOR':>12}  verdict")
    for lv in levels:
        for name, r in by_level[lv]:
            for arm in ARM_ORDER:
                if arm not in r["results"]:
                    continue
                dis = [row["disagree"] for row in r["results"][arm]["rounds"]
                       if row.get("disagree_has_fresh")]
                if not dis:
                    continue
                def m(key):
                    vals = [d[key]["rank_corr"] for d in dis if key in d]
                    return float(np.mean(vals)) if vals else float("nan")
                roll, dp, floor = m("roll_vs_fm_pred"), m("dp_vs_fm_pred"), \
                    m("fm_pred_vs_fm_pred_fresh")
                gap = floor - roll
                verdict = ("COLLAPSED?" if np.isfinite(gap) and gap < 0.15
                           else "distinct" if np.isfinite(gap) else "-")
                print(f"{('L' + str(lv)):>14} {arm:>14} {roll:+12.3f} {dp:+10.3f} "
                      f"{floor:+12.3f}  {verdict}")

    # ---- 2. the content control --------------------------------------------------------
    print(f"\n{'=' * 78}\n2. CONTENT CONTROL -- `endo_random` must not beat `frozen` on "
          f"control\n{'=' * 78}")
    deep = sched_levels[-1]
    print(f"{'action space':>14} {'arm':>14} {'ballistic':>10} {'freshtop1':>10} "
          f"{'PR':>8} {'d' + str(L):>8}   (at damage depth {deep})")
    for lv in levels:
        for name, r in by_level[lv]:
            for arm in ("frozen", "endo_random", "dense"):
                if arm not in r["results"]:
                    continue
                g = final_grade(r["results"][arm], deep)
                last = r["results"][arm]["rounds"][-1]
                if g is None:
                    continue
                print(f"{('L' + str(lv)):>14} {arm:>14} {g['ballistic']:10.3f} "
                      f"{g['fm']['value_top1_agree']:10.3f} {last['depth']['PR']:8.2f} "
                      f"{last['depth'][f'd{L}']:8.3f}")
    print("  PR is printed for continuity and is NOT a ranker (endo_expansion §3: a random\n"
          "  target reached 145% of the DP teacher's PR lift with ballistic at the floor).")

    # ---- 3. the 2x2 and its interaction ------------------------------------------------
    for metric, label in (("ballistic", "BALLISTIC CONTROL (fresh span FM)"),
                          ("freshtop1", "FRESH-FM value_top1_agree")):
        print(f"\n{'=' * 78}\n3. THE 2x2 -- {label}\n{'=' * 78}")
        for dmg in sched_levels:
            print(f"\n  damage depth {dmg}")
            print(f"    {'judge':>14} " + "".join(f"{'L' + str(lv):>12}" for lv in levels)
                  + f"{'level gain':>13}")
            cellv = {}
            for arm in ("evaluative", "endo_rollout", "frozen"):
                row = []
                for lv in levels:
                    vals = []
                    for name, r in by_level[lv]:
                        if arm not in r["results"]:
                            continue
                        g = final_grade(r["results"][arm], dmg)
                        if g is None:
                            continue
                        vals.append(g["ballistic"] if metric == "ballistic"
                                    else g["fm"]["value_top1_agree"])
                    row.append(float(np.mean(vals)) if vals else float("nan"))
                cellv[arm] = row
                gain = (row[-1] - row[0]) if len(row) > 1 else float("nan")
                print(f"    {arm:>14} " + "".join(f"{x:12.3f}" for x in row)
                      + f"{gain:13.3f}")
            if len(levels) > 1 and all(k in cellv for k in ("evaluative", "endo_rollout")):
                ge = cellv["evaluative"][-1] - cellv["evaluative"][0]
                gn = cellv["endo_rollout"][-1] - cellv["endo_rollout"][0]
                print(f"    {'INTERACTION':>14} (endo level gain - external level gain) = "
                      f"{gn - ge:+.3f}")
                # Recovery is LIFT OVER THE NO-LOOP FLOOR, not a raw ratio. `endo_expansion`'s
                # 34%/53% anchors are (endo - frozen) / (external - frozen); a raw ratio
                # credits the endogenous arm with everything the warm start already had, which
                # on this substrate is most of the number -- `frozen` reaches 0.316 ballistic
                # at L4 against `endo_expansion`'s 0.024, because hierarchical damage at
                # n_corrupt=2 with a value trained on the move set is a far stronger start.
                if "frozen" in cellv:
                    for i, lv in enumerate(levels):
                        fl, ex, en = (cellv["frozen"][i], cellv["evaluative"][i],
                                      cellv["endo_rollout"][i])
                        den = ex - fl
                        rec = (en - fl) / den if abs(den) > 1e-9 else float("nan")
                        print(f"    {'recovery @L' + str(lv):>14} "
                              f"(endo-frozen)/(ext-frozen) = ({en:.3f}-{fl:.3f})/"
                              f"({ex:.3f}-{fl:.3f}) = {rec:+.2f}")
                    print(f"    {'':>14} anchors: endo_expansion 0.34 ballistic / 0.53 "
                          f"transferable plannability")

    # ---- 4. teacher fidelity, the diagnosed binding constraint --------------------------
    print(f"\n{'=' * 78}\n4. TEACHER FIDELITY -- 'a grader is a ceiling'\n{'=' * 78}")
    print(f"{'action space':>14} {'arm':>14} {'agree_dp':>10} {'tree_share':>11} "
          f"{'mean_lvl':>9} {'oracle_lvl':>11} {'informative':>12} {'meter_mat':>11}")
    for lv in levels:
        for name, r in by_level[lv]:
            for arm in ("evaluative", "endo_rollout", "endo_random"):
                if arm not in r["results"]:
                    continue
                tt = [row["teacher"] for row in r["results"][arm]["rounds"]
                      if row.get("teacher")]
                if not tt:
                    continue
                last = r["results"][arm]["rounds"][-1]
                print(f"{('L' + str(lv)):>14} {arm:>14} "
                      f"{np.mean([t['agree_dp'] for t in tt]):10.3f} "
                      f"{np.mean([t['tree_share'] for t in tt]):11.3f} "
                      f"{np.mean([t['mean_level'] for t in tt]):9.2f} "
                      f"{np.mean([t['oracle_mean_level'] for t in tt]):11.2f} "
                      f"{np.mean([t['informative_frac'] for t in tt]):12.3f} "
                      f"{last['meter']['materialisations']:11.3g}")
    print("  meter_mat differs across action spaces BY DESIGN -- an evaluative grader over a\n"
          "  richer action set costs more to consult. No arm gets extra gradient steps.")

    # ---- 4b. the PI instruction: does pointing the teacher upward recover anything? ------
    floors = ("endo_floor", "endo_floor_static", "endo_floor_anti", "endo_treefloor")
    have_floor = any(a in r["results"] for r in runs.values() for a in floors)
    if have_floor:
        print(f"\n{'=' * 78}\n4b. THE PI INSTRUCTION -- ~14 bits of 'aim higher' against the "
              f"DP's ~83k\n{'=' * 78}")
        for lv in levels:
            for dmg in sched_levels:
                base = {}
                for arm in ("frozen", "evaluative", "endo_rollout") + floors:
                    vals = [final_grade(r["results"][arm], dmg)
                            for _, r in by_level[lv] if arm in r["results"]]
                    vals = [g["ballistic"] for g in vals if g is not None]
                    if vals:
                        base[arm] = float(np.mean(vals))
                if not any(a in base for a in floors):
                    continue
                fl = base.get("frozen", float("nan"))
                ex = base.get("evaluative", float("nan"))
                den = ex - fl
                print(f"\n  L{lv}, damage depth {dmg}   "
                      f"(floor={fl:.3f}, external ceiling={ex:.3f})")
                for arm in ("endo_rollout",) + floors:
                    if arm not in base:
                        continue
                    rec = (base[arm] - fl) / den if abs(den) > 1e-9 else float("nan")
                    vs_tree = (base[arm] - base["endo_treefloor"]
                               if "endo_treefloor" in base else float("nan"))
                    print(f"    {arm:>20} ballistic={base[arm]:.3f}  "
                          f"recovery={rec:+.2f}  vs_treefloor={vs_tree:+.3f}")
                print("    A floor arm must beat endo_treefloor, not just endo_rollout -- "
                      "otherwise it bought\n    RELEVANCE (only tree moves are deep here), "
                      "not ALTITUDE.")

        print(f"\n  H1/H2 -- can the teacher rank abstract moves once pointed at them?")
        print(f"    {'arm':>20} {'l_min':>14} {'agree|allowed':>14} {'chance':>8} "
              f"{'excl_oracle':>12} {'allowed_tree':>13}")
        for lv in levels:
            for name, r in by_level[lv]:
                for arm in floors:
                    if arm not in r["results"]:
                        continue
                    tt = [row["teacher"] for row in r["results"][arm]["rounds"]
                          if row.get("teacher")]
                    if not tt:
                        continue
                    print(f"    {arm:>20} {str([t['min_level'] for t in tt][::3]):>14} "
                          f"{np.nanmean([t['agree_dp_within_allowed'] for t in tt]):14.3f} "
                          f"{np.mean([t['chance_within_allowed'] for t in tt]):8.3f} "
                          f"{np.mean([t['floor_excludes_oracle_frac'] for t in tt]):12.3f} "
                          f"{np.mean([t['allowed_tree_share'] for t in tt]):13.3f}")
        print("    agree|allowed >> chance => the within-level ranking was always fine and the\n"
              "    deficit was purely DIRECTIONAL. Near chance => it cannot rank abstract moves\n"
              "    at all, and pointing it upward cannot help.")

    # ---- 5. does the teacher climb with the damage schedule? ---------------------------
    print(f"\n{'=' * 78}\n5. DOES THE TEACHER FOLLOW THE SCHEDULE? mean target level by "
          f"damage depth\n{'=' * 78}")
    for lv in levels:
        if lv == 1:
            continue
        for name, r in by_level[lv]:
            for arm in ("evaluative", "endo_rollout", "endo_random"):
                if arm not in r["results"]:
                    continue
                per = {}
                for row in r["results"][arm]["rounds"]:
                    if row.get("teacher"):
                        per.setdefault(row["damage_level"], []).append(
                            (row["teacher"]["mean_level"],
                             row["teacher"]["oracle_mean_level"]))
                if not per:
                    continue
                cells = "  ".join(
                    f"dmg{d}: {np.mean([a for a, _ in v]):.2f}/{np.mean([b for _, b in v]):.2f}"
                    for d, v in sorted(per.items()))
                print(f"  L{lv} {arm:>14}  {cells}   (arm/oracle)")
    print("  The oracle column IS the schedule made visible: level_ladder's G-L found the exact\n"
          "  DP's best-move level climbing 1 -> 2 -> 3 with damage depth at n_corrupt=2. If the\n"
          "  oracle column is flat, the DGP knob did not bite and nothing else here is readable.")


if __name__ == "__main__":
    main()
