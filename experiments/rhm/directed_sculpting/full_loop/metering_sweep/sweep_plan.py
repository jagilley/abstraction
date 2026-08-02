"""The metering sweep's grid, derived rather than hand-typed.

The whole design rests on one identity, so it is computed in one place and printed with the
launch commands, rather than being written into a README where it can drift from what ran:

    M     = n_channels*mon_n + n_channels*floor_n*floor_draws + forecast_n     (monitor sequences)
    ratio = mon_price * M / collect_budget
    T     = mon_price * M + collect_budget                                     (total spend)

E1-A holds T and M fixed and sweeps the PRICE: B = T/(1+r), price = r*B/M. Information is
constant (the taps see bit-identical monitoring quantities at every point) and only the price
moves -- which is the only way price can bite, because nothing in `ladder.py` reads the meter,
so a price multiplier on its own is a provable no-op (see PREREGISTRATION.md §2).

E1-B relaxes T (price fixed at 1.0, collection raised) -- the "data becomes free" limb the
fixed-T design cannot reach, and the source of the matched-ratio pairs.

E2 holds B and price fixed and sweeps the monitoring QUANTITY -- the amount-sweep, which by
PREREGISTRATION.md §2(ii) must NOT move the uniform->oracle prize, and is run as the control on
that argument.

Usage (from experiments/):
    python3 rhm/directed_sculpting/full_loop/metering_sweep/sweep_plan.py            # the table
    python3 rhm/directed_sculpting/full_loop/metering_sweep/sweep_plan.py --commands # + launcher
"""

import argparse

N_CH = 5                      # tree, structA, structB, noiseA, noiseB
PUB = dict(mon_n=1280, floor_n=512, floor_draws=3, forecast_n=512, collect_budget=8192)
ARMS = "uniform,reducible_only,visits_only,value_red,oracle,oracle_dup"
SEEDS = (1, 2, 3)


def monitor_sequences(mon_n, floor_n, floor_draws, forecast_n, n_ch=N_CH):
    return n_ch * mon_n + n_ch * floor_n * floor_draws + forecast_n


PUB_KL = 0.30                 # ladder.py's `drift_kl`, per node, in every published run
M_PUB = monitor_sequences(**{k: PUB[k] for k in ("mon_n", "floor_n", "floor_draws", "forecast_n")})
T_PUB = M_PUB + PUB["collect_budget"]           # 14592 + 8192 = 22784
R_PUB = M_PUB / PUB["collect_budget"]           # 1.78125, every ladder run in the repo


def e1a_points(ratios=(0.0625, 0.25, 0.5, R_PUB, 4.0, 8.0, 22.0)):
    """Price sweep at fixed total spend. Returns (label, ratio, collect_budget, price)."""
    out = []
    for r in ratios:
        b = int(round(T_PUB / (1.0 + r)))
        price = r * b / M_PUB
        out.append((f"r{r:g}".replace(".", "p"), r, b, price))
    return out


def e1b_points(budgets=(32768, 65536)):
    """Abundance limb: price fixed at 1.0, collection raised, total spend NOT fixed."""
    return [(f"ab{b}", M_PUB / b, b, 1.0) for b in budgets]


def e2_points(scales=(0.25, 4.0)):
    """Information sweep: monitoring QUANTITY scaled, budget and price fixed."""
    out = []
    for sc in scales:
        cfg = dict(mon_n=int(PUB["mon_n"] * sc), floor_n=int(PUB["floor_n"] * sc),
                   floor_draws=PUB["floor_draws"], forecast_n=int(PUB["forecast_n"] * sc))
        m = monitor_sequences(**cfg)
        out.append((f"info{sc:g}".replace(".", "p"), m / PUB["collect_budget"], cfg, m))
    return out


# --------------------------------------------------------------------------------------- #
# Round 2. Round 1 found the prize RISING as the meter turns off, and diagnosed why the
# doc's prediction never got its chance: E(d) steepens, leaving ~16 doublings to the
# aleatoric floor. These four limbs are built to reach the domain where the claim is
# testable on its own terms, rather than to re-measure the same range harder.
# --------------------------------------------------------------------------------------- #

def e1c_points(budgets=(262144,)):
    """Brute force: 4x the top of round 1, to look for the turn in E(d) directly."""
    return [(f"big{b}", M_PUB / b, b, 1.0) for b in budgets]


def e3_points(kls=(0.15, 0.075, 0.03, 0.01)):
    """THE DAMAGE SWEEP -- the cheap route into the saturation domain.

    What sets the steady-state error on this ladder is not data in the abstract, it is
    data PER UNIT OF DAMAGE: the round is `advance_drift -> collect -> refit`, so a budget
    of B against a drift of `drift_kl` sits at the same point of the repair balance as a
    budget of 2B against a drift of 2*`drift_kl`. Round 1 moved the numerator over 66x at
    real GPU cost; the denominator moves 30x for free (the published 0.30 -> 0.01), and it
    is the same axis `full_loop` §6 already calls "samples per event at fixed KL/event".

    Lowering damage is what lets a FIXED budget converge toward the forward model's own
    floor -- which is the condition the doc's prediction requires and round 1 could not
    reach. `verify_distractors` P4 is the anchor: a STATIC geometry goes inert, every
    channel's LP under the noise floor by step 7500. Inert is what saturation looks like,
    and it is exactly where the metering claim says the prize must vanish.

    Stops at 0.01 rather than 0.0 because `calibrate_drift` bisects for a target KL and a
    target of exactly zero is degenerate.
    """
    return [(f"kl{k:g}".replace(".", "p"), k) for k in kls]


def e3b_points(corners=((0.01, 32768),)):
    """The deep-saturation CORNER: low damage AND high budget together.

    `drift_kl`=0.01 with B=32768 is 120x the published data-per-damage -- ~15x past round 1's
    most abundant point, at ~1/8 its GPU cost. If the prize does not vanish here it does not
    vanish for a reachable reason, and P8's second branch (interference is structural) is the
    reading rather than a possibility.
    """
    return [(f"kl{k:g}b{b}".replace(".", "p"), k, b) for k, b in corners]


def e4_points():
    """DATA vs COMPUTE -- the main threat to round 1's E(d) steepening.

    `n_steps = fm_epochs * collect_budget // batch_size`, so round 1's budget sweep moved
    distinct data AND gradient steps together at a fixed 8 epochs. Two orthogonal cuts:

      fixed COMPUTE (`fm_epochs * B` = 65536 = the published product), data varying 16x
      fixed DATA    (B = 8192),                                    compute varying 16x

    If the steepening is a compute artefact the fixed-compute cut is flat; if it is real
    the fixed-data cut is flat. The published (8192, 8) cell sits in both and is already
    measured, so only four new points are needed.
    """
    out = []
    for b, ep in ((2048, 32), (32768, 2)):                    # fixed compute, varying data
        out.append((f"fc{b}", b, ep, "fixed-compute"))
    for ep in (2, 32):                                        # fixed data, varying compute
        out.append((f"fd{ep}", PUB["collect_budget"], ep, "fixed-data"))
    return out


def e6_points(steps=256, cells=((1024, 64), (4096, 16), (16384, 4), (65536, 1))):
    """THE ISO-COMPUTE CURVE -- and after E4, the one that matters.

    E4 found that round 1's E(d) "steepening" was a COMPUTE artefact. At fixed 8 epochs,
    raising B raises distinct data (helps) AND gradient steps (hurts -- 16x the steps on the
    same 8192 samples costs +0.117 in tree error, i.e. the loop OVERFITS within the round).
    Hold `fm_epochs * B` fixed instead and the curve behaves normally: -0.061 -> -0.026 per
    doubling, FLATTENING toward a floor. The prize follows the shape round 1 pre-registered
    and could not find: 0.0354 -> 0.0633 -> 0.0519, HUMPED, turning over at the abundant end.

    Iso-compute is also the right scaling for the doc's own analogy. "Train on all of it" vs
    curation is a choice made at a FIXED TRAINING BUDGET over a corpus of varying size -- more
    corpus does not buy more gradient steps. Round 1's fixed-epoch scaling silently gave the
    abundant end more compute as well as more data, which is what hid the turnover.

    Existing cells at 256 steps/round: (2048, 32), (8192, 8), (32768, 2). These four complete
    a 7-point curve spanning 64x in data at exactly matched compute.
    """
    return [(f"ic{b}", b, ep, steps) for b, ep in cells]


def e7_points(cells=((131072, 2),)):
    """A SECOND iso-compute level (1024 steps/round), to check the turnover is not specific
    to one compute budget. Two of its three cells already exist -- `fd32` (8192, 32) and
    `ab32768` (32768, 8) both sit at 1024 steps/round -- so one new point completes it."""
    return [(f"ic2_{b}", b, ep, ep * b // 256) for b, ep in cells]


def e5_points(totals=(17408, 22784, 45568)):
    """THE HONEST LADDER -- each arm pays for the taps its OWN drive reads.

    The published ladder charges every arm the same monitor bill on purpose, "so the ladder
    measures allocation only". That isolates the allocation rule and hides the economics:
    a smart allocator never pays for its own smartness. Here one shared total is split by
    what each policy actually needs, so `uniform` (which reads nothing) collects the whole
    budget and `value_red` buys 64% less data in exchange for knowing where to spend it.

    This is the metering question in its first-class form -- IS BEING SMART WORTH WHAT IT
    COSTS TO BE SMART -- and it has never been askable, because monitoring has always been
    a subsidy applied equally to every rung. It also puts a price on the idea doc §6 claim
    that the cheap curation is the weak curation: the relevance tap costs `forecast_n` = 512
    sequences, the reducibility tap costs 14080, a 27.5x difference in price that the equal-
    charge ladder makes invisible.
    """
    bills = {a: sum({"lp": N_CH * PUB["mon_n"],
                     "floor": N_CH * PUB["floor_n"] * PUB["floor_draws"],
                     "fc": PUB["forecast_n"]}[t] for t in reads)
             for a, reads in (("uniform", ()), ("oracle", ()), ("oracle_dup", ()),
                              ("visits_only", ("fc",)),
                              ("reducible_only", ("lp", "floor")),
                              ("value_red", ("lp", "floor", "fc")))}
    return [(f"own{t}", t, bills) for t in totals]


# The PUBLISHED arm set (`channel_env.POLICIES`, in order) plus the noise-floor arm appended
# LAST, so the first ten arms consume the RNG stream exactly as `ladder_fix_s*` did and the
# comparison is arm-for-arm rather than merely like-for-like.
FULL_ARMS = ("uniform,error_only,lprog_only,visits_only,value,value_satiety,"
             "reducible_only,value_red,value_red_satiety,oracle,oracle_dup")


def e8_points(epochs=(1, 2, 4)):
    """RE-RUN THE PUBLISHED LADDER AT FEWER EPOCHS.

    E4 measured that the loop overfits the round's collected sample: at B=8192, 16x the
    gradient steps costs +0.117 tree error, and 2 epochs beats the published 8 by 0.037 on
    the six-arm set. Every headline number on this node -- the tap repair (-0.054 +/- 0.009),
    the value_red/visits_only tie, the 84%/18% recoveries, the satiety leak, the 13-18x value
    separation -- was measured at 8 epochs, i.e. demonstrably over-trained.

    This asks the only question that matters about that: does the published ORDERING survive?
    Absolute levels being pessimistic is a footnote; the ordering flipping would put the
    node's conclusions at risk. Full ten-arm set so the arms E4 omitted (`value`, `lprog_only`,
    and both satiety rungs) are covered, at the published B, mon_n, rounds and n_eval, so
    `fm_epochs` is the only thing that differs from `ladder_fix_s*`.
    """
    return [(f"ep{e}", e) for e in epochs]


def _cmd(tag, seed, extra):
    return (f"modal run --detach rhm/directed_sculpting/full_loop/ladder.py::ladder "
            f"--tag {tag} --seed {seed} --rounds 12 --n-eval 1024 "
            f"--policies {ARMS} {extra}")


def commands():
    """Every launch command, in the order they should go out."""
    cmds = []
    for label, r, b, price in e1a_points():
        for s in SEEDS:
            cmds.append(_cmd(f"met_{label}_s{s}", s,
                             f"--collect-budget {b} --mon-price {price:.6f}"))
    for label, r, b, price in e1b_points():
        for s in SEEDS:
            cmds.append(_cmd(f"met_{label}_s{s}", s,
                             f"--collect-budget {b} --mon-price {price:.6f}"))
    for label, r, cfg, m in e2_points():
        for s in SEEDS:
            cmds.append(_cmd(f"met_{label}_s{s}", s,
                             f"--collect-budget {PUB['collect_budget']} --mon-price 1.0 "
                             f"--mon-n {cfg['mon_n']} --floor-n {cfg['floor_n']} "
                             f"--forecast-n {cfg['forecast_n']}"))
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", action="store_true")
    args = ap.parse_args()

    print(f"published monitoring quantity M = {M_PUB} sequences/round "
          f"(5*{PUB['mon_n']} LP + 5*{PUB['floor_n']}*{PUB['floor_draws']} floor "
          f"+ {PUB['forecast_n']} forecast)")
    print(f"published collect_budget B = {PUB['collect_budget']}  ->  ratio = {R_PUB:.5f}  "
          f"(matches every ladder run in the repo)")
    print(f"total spend T = {T_PUB}\n")

    print("E1-A  price at fixed total spend (headline)")
    print(f"  {'label':10s} {'ratio':>9s} {'B':>7s} {'price':>8s} {'T_realised':>11s} {'B/B_pub':>8s}")
    for label, r, b, price in e1a_points():
        print(f"  {label:10s} {r:>9.4f} {b:>7d} {price:>8.4f} {price*M_PUB + b:>11.1f} "
              f"{b / PUB['collect_budget']:>8.2f}x")

    print("\nE1-B  abundance limb (price fixed at 1.0, T not fixed)")
    print(f"  {'label':10s} {'ratio':>9s} {'B':>7s} {'price':>8s} {'T_realised':>11s} {'B/B_pub':>8s}")
    for label, r, b, price in e1b_points():
        print(f"  {label:10s} {r:>9.4f} {b:>7d} {price:>8.4f} {price*M_PUB + b:>11.1f} "
              f"{b / PUB['collect_budget']:>8.2f}x")

    print("\nE2  information at fixed price and fixed budget (the control)")
    print(f"  {'label':10s} {'ratio':>9s} {'mon_n':>7s} {'floor_n':>8s} {'fc_n':>6s} {'M':>7s}")
    for label, r, cfg, m in e2_points():
        print(f"  {label:10s} {r:>9.4f} {cfg['mon_n']:>7d} {cfg['floor_n']:>8d} "
              f"{cfg['forecast_n']:>6d} {m:>7d}")

    print("\n--- ROUND 2: reaching the domain where the doc's claim is testable ---")
    print("E1-C  brute force, 4x the top of round 1")
    for label, r, b, price in e1c_points():
        print(f"  {label:10s} ratio={r:.4f} B={b} price={price:.2f} "
              f"fm_steps/round={8 * b // 256}")
    print("E3    the DAMAGE sweep -- data per unit of damage, moved from the denominator")
    print(f"  {'label':10s} {'drift_kl':>9s} {'vs published':>13s} {'equivalent to':>26s}")
    for label, k in e3_points():
        print(f"  {label:10s} {k:>9.3f} {PUB_KL / k:>12.0f}x "
              f"{'B=' + str(int(PUB['collect_budget'] * PUB_KL / k)):>26s}")
    print("E4    DATA vs COMPUTE at matched product (published product = 8*8192 = 65536)")
    print(f"  {'label':10s} {'B':>7s} {'fm_epochs':>10s} {'steps/round':>12s} {'cut':>14s}")
    for label, b, ep, cut in e4_points():
        print(f"  {label:10s} {b:>7d} {ep:>10d} {ep * b // 256:>12d} {cut:>14s}")
    print("E5    the HONEST ladder -- each arm pays for the taps its own drive reads")
    _, _, bills = e5_points()[0]
    print(f"  per-round monitoring bill: " + ", ".join(f"{a}={c}" for a, c in bills.items()))
    print(f"  {'label':10s} {'T':>7s} | " + " ".join(f"{a[:9]:>9s}" for a in bills))
    for label, t, bl in e5_points():
        print(f"  {label:10s} {t:>7d} | " + " ".join(f"{t - c:>9d}" for c in bl.values()))

    print("\nmatched-ratio pairs (same r, different B -- the discriminator):")
    print(f"  r~0.23-0.25 : E1-A B=18227 (price 0.312)  vs  E1-B B=65536 (price 1.0)")
    print(f"  r~0.445     : E2  B=8192   (M=3648)       vs  E1-B B=32768 (price 1.0)")

    cmds = commands()
    n2 = (len(e1c_points()) + len(e3_points()) + len(e4_points()) + len(e5_points())) * 3
    print(f"\nround 1: {len(cmds)} runs ({len(e1a_points())*3} E1-A + {len(e1b_points())*3} "
          f"E1-B + {len(e2_points())*3} E2)   round 2: {n2} runs "
          f"({len(e1c_points())*3} E1-C + {len(e3_points())*3} E3 + {len(e4_points())*3} E4 "
          f"+ {len(e5_points())*3} E5)")
    if args.commands:
        print()
        for c in cmds:
            print(c)


if __name__ == "__main__":
    main()
