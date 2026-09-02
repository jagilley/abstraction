"""[antiphon-m] PHASE 0 FOR THE STRONGLY-METERED ROUND (`an_m0`) — sized offline, no GPU.

THE QUESTION. Every question-port run so far ran at ABUNDANCE: `n_pr` = 64 posed per cycle,
~15 solved, `mine_cap` = 8 taken by the donor's subsample. The knob's effect on the table is
diluted twice and every arm asks more than it can use. The claim under test is the roadmap's:
knowing WHICH questions to ask is load-bearing when questions are SCARCE.

THE KNOB, and why it is that one. `n_pr` 64 -> 24 and `pr_width` 16 -> 8, exactly
`intonation`'s Round A (`ma_s0`): *the meter can afford 24 graded performances a cycle instead
of 64, and a width-8 search instead of width-16*. The grammar, the ladder, the caps, the plan
budget, `mine_support` and every floor are untouched, so the world's truth does not move.
`mine_cap` is DELIBERATELY NOT SCALED — it is the arc's own constant and the fact that it STOPS
BINDING is the regime. Its consequence is stated rather than engineered around: with solves
below the cap the mining subsample is never taken, so the mined stream IS the solved stream.

THE METHOD is `ma_s0`'s: size it from the incumbent's OWN per-cycle record (a `--quick` smoke
cannot size a regime — at quick scale every regime looks scarce), predict an UPPER bound by
holding the solve FRACTION fixed (a narrower beam solves a smaller fraction, so the run must
come in below), then verify at real scale with a one-arm era-1 probe before the main run.

Reads `an_s0`'s banked logs and `phase0.json`; writes `phase0_metered.json`. Run from
experiments/:
    PYTHONPATH=. python3 rhm/practice/antiphon/phase0_metered.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

N_PR_OLD, N_PR_NEW = 64, 24
PR_WIDTH_OLD, PR_WIDTH_NEW = 16, 8
MINE_CAP = 8
SUPPORT = 3
PROP_BUF_CAP = 60_000
K = "2048"
# the ladder `an_s0` ran and `an_m0` will run: era -> cycles
LADDER = {1: 60, 2: 50, 3: 70, 4: 12, 5: 9}
EARNS = {1: 2, 2: 3, 3: 4}
REF_ARMS = ("q_exo", "q_bisect", "q_endo", "q_novel", "q_comp", "q_comp_free")


def log_of(tag, arm):
    with open(os.path.join(FIG, tag, arm, "results.json")) as f:
        return json.load(f)["log"]


def main():
    P = json.load(open(os.path.join(HERE, "phase0.json")))
    reach = P["reach"]
    n_t3 = P["sizes"]["3"]["distinct_flats"]
    n_t4 = P["sizes"]["4"]["distinct_flats"]
    d_prior = float(P["calibration"]["3"]["true_ratio_median"])

    print("=" * 88)
    print("PHASE 0 (metered) — the question port at the strongly-metered knob, sized offline")
    print(f"  n_pr {N_PR_OLD} -> {N_PR_NEW}   pr_width {PR_WIDTH_OLD} -> {PR_WIDTH_NEW}   "
          f"mine_cap {MINE_CAP} (NOT scaled)   support {SUPPORT}   K {K}")
    print("=" * 88)

    # ---- [M0] the abundance reference, from an_s0's own per-cycle record ------------------
    print("\n[M0] THE ABUNDANCE REFERENCE — what regime `an_s0` actually ran in")
    print(f"  {'arm':13s}{'solv/cyc':>10}{'x cap':>8}{'cap binds':>11}"
          f"{'obs/cyc':>9}{'pi rows/cyc':>13}{'buf end':>9}{'buf full':>10}")
    ref, scaled = {}, []
    for a in REF_ARMS:
        L = log_of("an_s0", a)
        sv = np.array([q["n_solved"] for q in L["q"]
                       if q and q.get("n_solved") is not None], float)
        pr = np.array([(x or {}).get("n_pairs", 0) for x in L["prop"]], float)
        pb = np.array([(x or {}).get("n", 0) for x in L["prop"]], float)
        full = next((i + 1 for i, x in enumerate(pb) if x >= PROP_BUF_CAP), None)
        ref[a] = {"solves": float(sv.mean()), "x_cap": float(sv.mean()) / MINE_CAP,
                  "cap_binds": float((sv >= MINE_CAP).mean()),
                  "obs": float(np.mean(L["n_mined"])),
                  "prop_rows": float(pr.mean()),
                  "buf_end": float(pb[-1]) / PROP_BUF_CAP, "buf_full_cycle": full}
        r = ref[a]
        print(f"  {a:13s}{r['solves']:>10.2f}{r['x_cap']:>7.2f}x{r['cap_binds']:>10.1%}"
              f"{r['obs']:>9.2f}{r['prop_rows']:>13.0f}{r['buf_end']:>8.1%}"
              f"{('c' + str(full)) if full else 'never':>10}")
        scaled.append(sv * N_PR_NEW / N_PR_OLD)
        # per-era solve fraction, for the budget below
        for j in LADDER:
            idx = [i for i, q in enumerate(L["era"]) if q == j]
            v = [L["q"][i]["n_solved"] for i in idx
                 if L["q"][i] and L["q"][i].get("n_solved") is not None]
            ref[a].setdefault("frac_by_era", {})[j] = float(np.mean(v)) / N_PR_OLD if v else None
    print("  `intonation` Round A's own abundance point (`in_s0`): 14.9 solves/cyc, 1.87x cap,")
    print("  cap binds 95.4%, pi buffer full at c26 — the two arcs sit at the same operating")
    print("  point, which is what makes Round A's metered numbers a usable prior here.")

    # ---- [M1] the prediction ------------------------------------------------------------
    sc = np.concatenate(scaled)
    pred = {"solves_per_cycle": float(sc.mean()),
            "solves_median": float(np.median(sc)),
            "x_cap": float(sc.mean()) / MINE_CAP,
            "cap_binds_frac": float((sc >= MINE_CAP).mean()),
            "obs_per_cycle": float(np.minimum(sc, MINE_CAP).mean()),
            "prop_rows_per_cycle": float(np.mean([ref[a]["prop_rows"] for a in REF_ARMS])
                                         * N_PR_NEW / N_PR_OLD)}
    pred["prop_buf_full_cycle"] = int(np.ceil(PROP_BUF_CAP / pred["prop_rows_per_cycle"]))
    print("\n[M1] THE PREDICTION at `n_pr` = 24 — the solve FRACTION held fixed, so an UPPER")
    print("     BOUND by construction: `pr_width` 16 -> 8 is a narrower search and will solve a")
    print("     smaller fraction. `ma_s0` predicted 5.60 this way and realized 4.28-4.65.")
    print(f"     solves/cycle          {pred['solves_per_cycle']:.2f} "
          f"(median {pred['solves_median']:.2f}) = {pred['x_cap']:.2f}x the cap {MINE_CAP}")
    print(f"     cap-binding fraction  {pred['cap_binds_frac']:.1%}   "
          f"(abundance: {np.mean([ref[a]['cap_binds'] for a in REF_ARMS]):.1%})")
    print(f"     obs/cycle             {pred['obs_per_cycle']:.2f}  "
          f"(abundance: {np.mean([ref[a]['obs'] for a in REF_ARMS]):.2f} — the cap)")
    print(f"     pi rows/cycle         {pred['prop_rows_per_cycle']:.0f}  -> buffer of "
          f"{PROP_BUF_CAP} fills at ~c{pred['prop_buf_full_cycle']} "
          f"(abundance: c26-c35)")
    print("     `prop_buf_cap` is NOT scaled either, for the same reason `mine_cap` is not.")

    # ---- [M2] the per-era observation budget --------------------------------------------
    print("\n[M2] THE PER-ERA OBSERVATION BUDGET — abundance vs metered")
    obs_era, budget_ab, budget_me = {}, {}, {}
    for j, cyc in LADDER.items():
        v = []
        for a in REF_ARMS:
            L = log_of("an_s0", a)
            v += [min(L["q"][i]["n_solved"] * N_PR_NEW / N_PR_OLD, MINE_CAP)
                  for i, q in enumerate(L["era"])
                  if q == j and L["q"][i] and L["q"][i].get("n_solved") is not None]
        obs_era[j] = float(np.mean(v))
        budget_ab[j], budget_me[j] = cyc * MINE_CAP, cyc * obs_era[j]
    print(f"  {'era':>4}{'cyc':>5}{'obs/cyc':>9}{'ab obs':>8}{'me obs':>8}{'ratio':>7}"
          f"{'ab keys':>9}{'me keys':>9}   earns")
    for j, cyc in LADDER.items():
        print(f"  {j:>4}{cyc:>5}{obs_era[j]:>9.2f}{budget_ab[j]:>8.0f}{budget_me[j]:>8.0f}"
              f"{budget_me[j] / budget_ab[j]:>7.2f}{int(budget_ab[j] // SUPPORT):>9}"
              f"{int(budget_me[j] // SUPPORT):>9}   "
              + (f"L{EARNS[j]}" if j in EARNS else "-"))
    print(f"  TOTAL {sum(budget_ab.values()):.0f} -> {sum(budget_me.values()):.0f} observations "
          f"({sum(budget_me.values()) / sum(budget_ab.values()):.2f}x)")

    # ---- [M3] reach vs budget: where the budget starts to BIND ---------------------------
    print("\n[M3] REACH vs BUDGET at K = 2048 — the sizing fact the round turns on")
    rb = {}
    for lv, era in ((2, 1), (3, 2), (4, 3)):
        r = reach[str(lv)][K]
        ka, km = int(budget_ab[era] // SUPPORT), int(budget_me[era] // SUPPORT)
        rb[lv] = {"reach": r, "keys_ab": ka, "keys_me": km, "binds_me": bool(r > km),
                  "binds_ab": bool(r > ka)}
        print(f"  L{lv}: menu reaches {r:>4} true keys | budget can drive {ka:>4} (abundance) "
              f"vs {km:>4} (metered)  -> "
              + ("BUDGET BINDS at the metered knob" if r > km and r <= ka else
                 ("binds in both" if r > ka else "reach <= budget in both")))
    print("  Under abundance the budget could in principle drive every reachable key at every")
    print("  level. At the metered knob it cannot at L4 — so WHICH of the reachable keys the")
    print("  questions aim at starts to determine the L4 book. That is the structural")
    print("  condition selection needs, and this round is the first in the arc that has it.")

    # ---- [M4] the predicted window at the metered budget ---------------------------------
    print("\n[M4] PREDICTED WINDOW at K = 2048, metered budget (abundance in parentheses)")
    print(f"  delta = delivery fidelity; the banked prior from `phase0.json` [2] is "
          f"{d_prior:.2f}")
    print(f"  {'delta':>6}{'L3@sup':>8}{'L3 recall':>11}{'L4@sup':>8}{'buildable true L4':>19}")
    rows = []
    for d in (0.3, round(d_prior, 3), 0.7, 1.0):
        cell = {}
        for lab, bud in (("metered", budget_me), ("abundance", budget_ab)):
            l3 = min(reach["3"][K] * d, bud[2] * d / SUPPORT)
            l4 = min(reach["4"][K] * d, bud[3] * d / SUPPORT)
            rec3 = l3 / n_t3
            cell[lab] = {"L3_at_sup": l3, "L3_recall": rec3, "L4_at_sup": l4,
                         "buildable_true_L4": rec3 ** 2 * n_t4}
        m, a = cell["metered"], cell["abundance"]
        rows.append({"delta": d, **cell})
        print(f"  {d:>6.2f}{m['L3_at_sup']:>8.1f}{m['L3_recall']:>11.3f}"
              f"{m['L4_at_sup']:>8.1f}{m['buildable_true_L4']:>19.1f}   "
              f"({a['L3_at_sup']:.1f} / {a['L3_recall']:.3f} / {a['L4_at_sup']:.1f} / "
              f"{a['buildable_true_L4']:.1f})")
    print("  The L3 column is UNCHANGED — at L3 the reach still sits under the metered budget,")
    print("  so the round's L3 prediction is `an_s0`'s. The L4@sup column is where the meter")
    print("  bites, and it is the oracle's predicted ceiling that the round is sized against.")
    print("  Incumbent, measured: `an_s0/q_exo` at its own L3 commit (c75) had L3@sup 8, true 3,")
    print("  recall 0.054 -> buildable true L4 ~6; end-of-run 22 true L4 keys at support. The")
    print("  abundance oracle reached 50.")

    out = {"knob": {"n_pr": [N_PR_OLD, N_PR_NEW], "pr_width": [PR_WIDTH_OLD, PR_WIDTH_NEW],
                    "mine_cap": MINE_CAP, "prop_buf_cap": PROP_BUF_CAP, "K": int(K)},
           "abundance_reference": ref, "prediction": pred,
           "obs_per_cycle_by_era": obs_era,
           "budget_abundance": budget_ab, "budget_metered": budget_me,
           "reach_vs_budget": rb, "predicted_window": rows}
    with open(os.path.join(HERE, "phase0_metered.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {os.path.join(HERE, 'phase0_metered.json')}")
    return out


if __name__ == "__main__":
    main()
