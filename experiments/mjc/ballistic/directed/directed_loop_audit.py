"""S2 audit — re-reads stored `directed_loop.py` results and asks whether the headline null
(`the conjunction never beats lprog-only in a loop`) is something the harness could actually
have measured. It is a LOCAL, read-only script: no Modal, no re-running, no new physics. Every
number it prints comes out of `figures/directed_loop_<tag>/results.json`, which is why the
correction it supports is reproducible rather than an assertion.

Why this exists. The S2 conclusion was read off `excess damage`, a CONTROL metric, and used to
downgrade the idea doc's two-taps claim. Four checks, each of which the stored per-round records
already answer:

  (1) SEED DECOMPOSITION. Is the aggregate a consistent ordering or one seed's blow-up?
  (2) RECOVERY vs PLATEAU. `excess damage` integrates over the whole `drift_every` epoch, but
      recovery finishes in ~2-3 rounds. What FRACTION of the score accrues after recovery is
      already complete? That fraction is the asymptotic metric S2 itself declared uninformative
      ("the asymptotic metric is the wrong metric, and it nulls").
  (3) MECHANISM vs CONTROL. `drift_value_loop` Cut 3 established control is a NEAR-BLIND grader
      of FM quality and that the value-relevant FM prediction error is the sighted one. Do the
      two instruments agree here? The value-relevant error is `per_err[j]` for the on-reach
      REDUCIBLE region, averaged over the same measured epochs.
  (4) MEASUREMENT SUBSIDY. Every round, every policy spends `probe_n x K` + `monitor_n x K`
      teleported transitions, free and everywhere, to decide where to spend `budget`. The
      visitation/relevance term exists to tell you WHERE TO LOOK; if looking is free, it can
      only ever be a tiebreaker on repair effort. Prints the ratio.

Run (from `experiments/`):
    python3 mjc/ballistic/directed/directed_loop_audit.py --tags loopC_s0 loopC_s1 loopC_s2
    python3 mjc/ballistic/directed/directed_loop_audit.py --tags loopB_s0 loopB_s1 loopB_s2
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECOVERY_ROUNDS = 3          # rounds after a drift within which recovery completes (see (2))


def load(tag):
    p = os.path.join(HERE, "figures", f"directed_loop_{tag}", "results.json")
    with open(p) as fh:
        return json.load(fh)


def schedule_of(cfg):
    """Rebuild the drift schedule from config. Older runs (loopA/loopB) predate `drift_cycle`
    and alternated over the reducible regions, which is what `cycle = cfg[...] or reducible`
    did in `directed_loop.py`."""
    REG = cfg["regions"]
    reducible = [j for j in range(len(REG)) if REG[j]["reducible"]]
    cycle = cfg.get("drift_cycle") or reducible
    sch = {}
    for i, t in enumerate(range(0, cfg["rounds"], cfg["drift_every"])):
        sch[t] = cycle[i % len(cycle)]
    return sch


def geometry(cfg):
    REG = cfg["regions"]
    on_reach = [j for j in range(len(REG)) if abs(REG[j]["center"][1]) <= 0.3]
    off_reach = [j for j in range(len(REG)) if j not in on_reach]
    # the value-relevant region: on the reach AND reducible — the one the metric can see
    relevant = [j for j in on_reach if REG[j]["reducible"]]
    return on_reach, off_reach, relevant


def per_run(tag):
    R = load(tag)
    cfg = R["config"]
    de, T = cfg["drift_every"], cfg["rounds"]
    sch = schedule_of(cfg)
    on_reach, off_reach, relevant = geometry(cfg)
    events = sorted(t for t, j in sch.items() if j in on_reach)
    measured = np.concatenate([np.arange(t0, min(t0 + de, T)) for t0 in events])
    floor = min(float(np.min([r["ballistic_cem"] for r in h])) for h in R["results"].values())

    out = {}
    for p, hist in R["results"].items():
        b = np.array([r["ballistic_cem"] for r in hist])
        pe = np.array([r["per_err"] for r in hist])
        al = np.array([r["alloc"] for r in hist], float)
        # a `nan` allocation row is the uniform-box no-op: budget spread over the whole box
        share = np.nan_to_num(al, nan=cfg["budget"] / len(cfg["regions"]))
        share = share / np.maximum(share.sum(1, keepdims=True), 1e-9)
        epochs = [b[t0:min(t0 + de, T)] for t0 in events]
        out[p] = dict(
            excess=float(np.mean([(e - floor).sum() for e in epochs])),
            early=float(np.mean([(e[:RECOVERY_ROUNDS] - floor).sum() for e in epochs])),
            late=float(np.mean([(e[RECOVERY_ROUNDS:] - floor).sum() for e in epochs])),
            relevant_err=float(pe[measured][:, relevant].mean()),
            offreach_share=float(share[measured][:, off_reach].sum(1).mean()),
        )
    return cfg, events, floor, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    runs = [per_run(t) for t in args.tags]
    cfg0 = runs[0][0]
    policies = [p for p in cfg0["policies"] if p in runs[0][3]]
    K = len(cfg0["regions"])

    print(f"\n{'='*92}\nS2 AUDIT — tags: {', '.join(args.tags)}")
    print(f"rounds={cfg0['rounds']} drift_every={cfg0['drift_every']} budget={cfg0['budget']}/round "
          f"buf_cap={cfg0['buf_cap']} K={K}")
    print(f"on-reach drift events per seed: {runs[0][1]}  (epochs the metric integrates over)")

    # ---- (4) measurement subsidy --------------------------------------------------------- #
    probe = cfg0["probe_n"] * K
    monitor = cfg0["monitor_n"] * K
    print(f"\n(4) MEASUREMENT SUBSIDY — free teleported transitions per round vs the budget "
          f"they allocate:\n    probe {cfg0['probe_n']}x{K}={probe}  + monitor "
          f"{cfg0['monitor_n']}x{K}={monitor}  = {probe+monitor} free  vs  budget "
          f"{cfg0['budget']}   -> {(probe+monitor)/cfg0['budget']:.0f}x subsidy")
    print("    (relevance tells you WHERE TO LOOK; here looking is free, so it can only be a "
          "tiebreaker on repair)")

    # ---- (1) seed decomposition ----------------------------------------------------------- #
    print(f"\n(1) SEED DECOMPOSITION — excess damage per seed (lower better)")
    hdr = "".join(f"{t:>12s}" for t in args.tags)
    print(f"{'policy':>13s}{hdr}{'mean':>10s}{'sem':>8s}   ordering")
    for p in policies:
        v = np.array([r[3][p]["excess"] for r in runs])
        row = "".join(f"{x:12.3f}" for x in v)
        print(f"{p:>13s}{row}{v.mean():10.3f}{v.std()/max(np.sqrt(len(v)),1):8.3f}")

    # ---- (2) recovery vs plateau ---------------------------------------------------------- #
    print(f"\n(2) RECOVERY vs PLATEAU — share of `excess damage` accrued AFTER round t0+"
          f"{RECOVERY_ROUNDS}, i.e. after recovery is done")
    print(f"{'policy':>13s}{'early':>10s}{'late':>10s}{'late share':>12s}")
    for p in policies:
        e = float(np.mean([r[3][p]["early"] for r in runs]))
        l = float(np.mean([r[3][p]["late"] for r in runs]))
        print(f"{p:>13s}{e:10.3f}{l:10.3f}{l/max(e+l,1e-9):11.0%}")
    lm = float(np.mean([np.mean([r[3][p]["late"] for p in policies]) for r in runs]))
    em = float(np.mean([np.mean([r[3][p]["early"] for p in policies]) for r in runs]))
    print(f"    -> {lm/(lm+em):.0%} of the 'adaptation speed' metric is post-recovery plateau, "
          f"i.e. the asymptotic metric S2 itself declared uninformative.")

    # ---- (3) mechanism vs control --------------------------------------------------------- #
    _, _, relevant = geometry(cfg0)
    rn = ", ".join(cfg0["regions"][j]["name"] for j in relevant)
    print(f"\n(3) MECHANISM vs CONTROL — value-relevant FM error ({rn}) over the same epochs")
    print(f"{'policy':>13s}{hdr}{'mean':>10s}    | control rank vs mechanism rank")
    mech = {p: np.array([r[3][p]["relevant_err"] for r in runs]) for p in policies}
    ctrl = {p: np.array([r[3][p]["excess"] for r in runs]) for p in policies}
    mrank = sorted(policies, key=lambda p: mech[p].mean())
    crank = sorted(policies, key=lambda p: ctrl[p].mean())
    for p in policies:
        row = "".join(f"{x:12.4f}" for x in mech[p])
        print(f"{p:>13s}{row}{mech[p].mean():10.4f}    | ctrl #{crank.index(p)+1}  mech #{mrank.index(p)+1}")
    print(f"    mechanism ranking: {' < '.join(mrank)}")
    print(f"    control   ranking: {' < '.join(crank)}")

    # the specific pair the null is about
    if "value" in policies and "lprog-only" in policies:
        w = int((mech["value"] < mech["lprog-only"]).sum())
        print(f"\n    value vs lprog-only — mechanism (value-relevant FM error): value lower in "
              f"{w}/{len(runs)} seeds")
        w2 = int((ctrl["value"] < ctrl["lprog-only"]).sum())
        print(f"    value vs lprog-only — control  (excess damage):            value lower in "
              f"{w2}/{len(runs)} seeds")

    # ---- allocation sanity ---------------------------------------------------------------- #
    print(f"\n(bonus) OFF-REACH BUDGET SHARE inside the measured epochs — is the relevance-"
          f"weighted arm actually being relevance-weighted?")
    for p in policies:
        v = np.array([r[3][p]["offreach_share"] for r in runs])
        print(f"{p:>13s}  " + " ".join(f"{x:.3f}" for x in v) + f"   mean {v.mean():.3f}")
    print()


if __name__ == "__main__":
    main()
