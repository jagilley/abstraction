"""Aggregate E4 -- the endogenous-expansion arms, paired per seed against the published anchors.

Five tables, in the order the argument needs them:

  1. EXPANSION      per-arm PR / d4 / ballistic / fresh-FM top1, and the fraction of the DP
                    teacher's lift each arm recovers, computed PAIRED PER SEED (each seed has
                    its own `frozen` floor and its own `evaluative` ceiling, so an unpaired
                    ratio of means would mix seed-level offsets into the recovery number).
  2. TEACHER        how good each endogenous teacher is: agreement with the full DP `k*`, the
                    share of its targets that are tree blocks, and how often it has an opinion
                    at all.
  3. DISAGREEMENT   the pre-registered instrument. The homogeneous floor (co-trained FM vs
                    fresh FM) against the heterogeneous rates. If the endogenous arm sits at
                    the floor, the loops collapsed and nothing else in this file is readable.
  4. WIREHEAD       reported vs realised success, and the beam's belief scalar vs its true
                    success -- `RHM_EDIT_CONTROL`'s Layer-1/Layer-2 pair.
  5. METER          what each teacher cost, in materialisations and in privileged DP calls.

Run from experiments/:
  python3 rhm/directed_sculpting/full_loop/endo_expansion/aggregate.py
  python3 rhm/directed_sculpting/full_loop/endo_expansion/aggregate.py --figures <dir>
"""

import argparse
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ("frozen", "dense", "evaluative", "endo_rollout", "endo_shuffled", "endo_random",
         "endo_value", "alloc_kstar", "alloc_uniform")


def _ms(xs):
    """mean, standard error of the mean, n -- over seeds."""
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem, int(a.size)


def _f(m, e, w=6, p=3):
    return f"{m:{w}.{p}f}±{e:.{p}f}" if np.isfinite(m) else " " * (w + 4) + "n/a"


def load(figdir, pattern="endo_expansion_e4*"):
    """{seed: results-dict}, MERGING arm-groups that share a seed, skipping smokes.

    The eight arms are split across two Modal jobs per seed (`e4a_*` = the core question,
    `e4b_*` = the controls and the allocation cut) because they run in parallel and each fits
    comfortably inside one timeout. That split is exact rather than approximate: `_run_arm`
    re-seeds the global torch RNG identically at every arm and derives every other stream from
    `seed` alone, so an arm's trajectory does not depend on which other arms shared its job.
    The shared setup (controller / generator / value / warm FM) is likewise a deterministic
    function of `seed`, so both jobs fork the same warm start -- which is checked here by
    comparing the round-0 belief.
    """
    out, round0 = {}, {}
    for path in sorted(glob.glob(os.path.join(figdir, pattern, "results.json"))):
        d = json.load(open(path))
        cfg = d["config"]
        if cfg["rounds"] < 5:                       # a --quick smoke
            continue
        seed = cfg["seed"]
        if seed not in out:
            out[seed] = d
            round0[seed] = d["belief_round0"]["PR"]
        else:
            if abs(round0[seed] - d["belief_round0"]["PR"]) > 1e-6:
                raise SystemExit(
                    f"seed {seed}: the two arm-groups did not fork the same warm start "
                    f"(round-0 PR {round0[seed]} vs {d['belief_round0']['PR']}) -- the split "
                    f"is not exact and the arms are not comparable.")
            out[seed]["results"].update(d["results"])
            out[seed]["config"]["arms"] = list(out[seed]["results"])
    return out


def _last(rows, key, sub=None):
    for r in reversed(rows):
        if key in r:
            return r[key][sub] if sub else r[key]
    return None


def _grade(rows, path):
    """Last graded value at `path` = tuple of keys."""
    for r in reversed(rows):
        cur = r
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        if cur is not None:
            return cur
    return None


def expansion_table(runs, depth_key):
    """Per-arm endpoint metrics and the paired recovery fraction of the DP teacher's lift."""
    metrics = {
        "PR": lambda rr: rr[-1]["depth"]["PR"],
        depth_key: lambda rr: rr[-1]["depth"][depth_key],
        "ballistic": lambda rr: _grade(rr, ("xworld", "base", "ballistic_w16")),
        "freshtop1": lambda rr: _grade(rr, ("xworld", "base", "fm", "value_top1_agree")),
    }
    per = {}
    for seed, d in runs.items():
        for arm, res in d["results"].items():
            for name, fn in metrics.items():
                try:
                    per.setdefault(arm, {}).setdefault(name, {})[seed] = fn(res["rounds"])
                except (KeyError, IndexError, TypeError):
                    per.setdefault(arm, {}).setdefault(name, {})[seed] = None

    r0 = _ms([d["belief_round0"]["PR"] for d in runs.values()])
    print(f"\n{'=' * 100}\n1. EXPANSION -- endpoint at the last round, {len(runs)} seed(s). "
          f"Round-0 belief PR {r0[0]:.2f}±{r0[1]:.2f}\n{'=' * 100}")
    print(f"{'arm':15s} {'PR r1':>13s} {'PR rN':>13s} {'d4':>13s} {'ballistic':>13s} "
          f"{'freshFM top1':>13s}")
    for arm in ORDER:
        if arm not in per:
            continue
        pr1 = _ms([runs[s]["results"][arm]["rounds"][0]["depth"]["PR"] for s in runs])
        cells = [_f(*_ms(list(per[arm][n].values()))[:2], 8, 3) for n in metrics]
        print(f"{arm:15s} {_f(pr1[0], pr1[1], 8, 3)} " + " ".join(cells))

    if "frozen" in per and "evaluative" in per:
        print(f"\n  recovery of the DP teacher's lift, PAIRED PER SEED "
              f"(arm-frozen)/(evaluative-frozen):")
        print(f"  {'arm':15s} " + " ".join(f"{n:>16s}" for n in metrics))
        for arm in ORDER:
            if arm in ("frozen", "evaluative") or arm not in per:
                continue
            cells = []
            for name in metrics:
                vals = []
                for s in runs:
                    a, f0, e = per[arm][name].get(s), per["frozen"][name].get(s), \
                        per["evaluative"][name].get(s)
                    if None in (a, f0, e) or abs(e - f0) < 1e-9:
                        continue
                    vals.append((a - f0) / (e - f0))
                m, sem, n = _ms(vals)
                cells.append(f"{m:>+10.1%}±{sem:.1%}" if np.isfinite(m) else f"{'n/a':>16s}")
            print(f"  {arm:15s} " + " ".join(cells))


def teacher_table(runs):
    print(f"\n{'=' * 100}\n2. TEACHER -- how good is each target, against the full DP `k*` it "
          f"never sees\n{'=' * 100}")
    print("   chance agreement with a 14-way argmax is 0.071; the tree is 8 of 14 blocks, so a "
          "block-uniform\n   teacher would score tree_share 0.571 with no relevance at all.")
    print("   NOTE `endo_shuffled` describes its PRE-SHUFFLE teacher, which is by construction "
          "the same\n   object `endo_rollout` uses -- identical rows here are the check that the "
          "two arms saw the\n   same teacher and differ only in whether it was paired with the "
          "right state. The DELIVERED\n   shuffled target's agreement with `k*` is not this "
          "number; `endo_random` is the true\n   zero-information floor.\n")
    print(f"{'arm':15s} {'agree_dp':>14s} {'tree_share':>14s} {'informative':>14s} "
          f"{'plan_acc(end)':>14s}")
    for arm in ORDER:
        rows = [(s, d["results"][arm]["rounds"]) for s, d in runs.items()
                if arm in d["results"]]
        if not rows or not rows[0][1][0].get("teacher"):
            continue
        agg = {}
        for key in ("agree_dp", "tree_share", "informative_frac"):
            agg[key] = _ms([np.mean([r["teacher"][key] for r in rr]) for _, rr in rows])
        pa = _ms([rr[-1].get("plan_acc") for _, rr in rows])
        print(f"{arm:15s} {_f(*agg['agree_dp'][:2], 9)} {_f(*agg['tree_share'][:2], 9)} "
              f"{_f(*agg['informative_frac'][:2], 9)} {_f(*pa[:2], 9)}")
    ni = _ms([np.mean([r["teacher"]["dp_no_improving_frac"]
                       for r in list(d["results"].values())[-1]["rounds"] if r.get("teacher")])
              for d in runs.values()])
    print(f"\n  fraction of states where NO move lowers the exact `d*` at all "
          f"(the published `evaluative` arm teaches its lowest-index tie-break, block 0, on "
          f"exactly these): {_f(*ni[:2], 5)}")
    for arm in ("alloc_kstar", "alloc_uniform"):
        rows = [d["results"][arm]["rounds"] for d in runs.values() if arm in d["results"]]
        if not rows:
            continue
        tf = _ms([np.mean([r["teacher"]["allowed_tree_frac"] for r in rr]) for rr in rows])
        print(f"  {arm}: mean tree fraction of the blocks the DP was allowed to look at = "
              f"{_f(*tf[:2], 5)}")


PAIRS = [("fm_pred_vs_fm_pred_fresh", "HOMOGENEOUS floor (co-trained FM vs fresh FM)"),
         ("dp_vs_fm_pred", "external evaluative vs dense"),
         ("roll_vs_fm_pred", "ENDOGENOUS evaluative vs dense"),
         ("value_vs_fm_pred", "reported evaluative vs dense"),
         ("belief_vs_fm_pred", "belief scalar vs dense"),
         ("roll_vs_dp", "endogenous vs external (teacher quality)"),
         ("value_vs_roll", "reported vs paid")]


def disagreement_table(runs):
    print(f"\n{'=' * 100}\n3. GRADER DISAGREEMENT -- the pre-registered instrument, mean over "
          f"rounds\n{'=' * 100}")
    print("   top1 = fraction of probe states where the two graders' argmax move differs")
    print("   corr = mean per-state correlation of the two move-score vectors (+1 = homogeneous)")
    print("   cost = how much of the SECOND grader, in its own normalised units, the first "
          "asks you to give up\n")
    for key, label in PAIRS:
        print(f"  [{label}]  ({key})")
        print(f"  {'arm':15s} {'top1':>14s} {'corr':>14s} {'cost':>14s}")
        any_row = False
        for arm in ORDER:
            per_seed = {"top1_disagree": [], "rank_corr": [], "cost": []}
            for d in runs.values():
                if arm not in d["results"]:
                    continue
                vals = [r["disagree"][key] for r in d["results"][arm]["rounds"]
                        if key in r.get("disagree", {})]
                if not vals:
                    continue
                for m in per_seed:
                    per_seed[m].append(float(np.mean([x[m] for x in vals])))
            if not per_seed["top1_disagree"]:
                continue
            any_row = True
            print(f"  {arm:15s} " + " ".join(_f(*_ms(per_seed[m])[:2], 9)
                                             for m in ("top1_disagree", "rank_corr", "cost")))
        if not any_row:
            print("  (not computed)")
        print()


def wirehead_table(runs):
    print(f"\n{'=' * 100}\n4. WIREHEAD -- does the REPORTED currency outrun the PAID one?\n"
          f"{'=' * 100}")
    print(f"{'arm':15s} {'bias r1':>13s} {'bias rN':>13s} {'Δbias':>13s} {'corr rN':>13s} "
          f"{'beam P(r*)':>13s} {'beam true':>13s}")
    for arm in ORDER:
        rows = [d["results"][arm]["rounds"] for d in runs.values() if arm in d["results"]]
        if not rows:
            continue
        b1 = _ms([rr[0]["value_calib"]["bias"] for rr in rows])
        bn = _ms([rr[-1]["value_calib"]["bias"] for rr in rows])
        db = _ms([rr[-1]["value_calib"]["bias"] - rr[0]["value_calib"]["bias"] for rr in rows])
        cn = _ms([rr[-1]["value_calib"]["corr"] for rr in rows])
        bp = _ms([_grade(rr, ("beam_grades", "belief_p_rstar")) for rr in rows])
        bt = _ms([_grade(rr, ("beam_grades", "true_success")) for rr in rows])
        print(f"{arm:15s} {_f(*b1[:2], 8)} {_f(*bn[:2], 8)} {_f(*db[:2], 8)} "
              f"{_f(*cn[:2], 8)} {_f(*bp[:2], 8)} {_f(*bt[:2], 8)}")


def meter_table(runs):
    print(f"\n{'=' * 100}\n5. METER -- what the teacher cost, cumulative over the run\n"
          f"{'=' * 100}")
    print(f"{'arm':15s} {'materialisations':>20s} {'privileged DP calls':>22s} "
          f"{'mat per label':>16s}")
    for arm in ORDER:
        rows = [d["results"][arm]["rounds"] for d in runs.values() if arm in d["results"]]
        if not rows:
            continue
        gs = [d["config"]["ground_states"] * d["config"]["rounds"] for d in runs.values()]
        mat = _ms([rr[-1]["meter"]["materialisations"] for rr in rows])
        dpc = _ms([rr[-1]["meter"]["dp_calls"] for rr in rows])
        per = mat[0] / np.mean(gs) if np.isfinite(mat[0]) else float("nan")
        print(f"{arm:15s} {mat[0]:>20.4g} {dpc[0]:>22.4g} {per:>16.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    runs = load(args.figures)
    if not runs:
        raise SystemExit(f"no non-smoke results under {args.figures}")
    depth_key = f"d{list(runs.values())[0]['config']['tree_depth']}"
    print(f"E4 -- endogenous expansion. seeds {sorted(runs)}; "
          f"arms {list(runs.values())[0]['config']['arms']}")
    expansion_table(runs, depth_key)
    teacher_table(runs)
    disagreement_table(runs)
    wirehead_table(runs)
    meter_table(runs)


if __name__ == "__main__":
    main()
