"""Cross-seed aggregation for jacobian_teacher Phase C (`committee.py`). Local, read-only.

The one question: does the committee's cross-member SIGN AGREEMENT stand in for the oracle's
direction reading? Three ways of asking it, in order of how much weight they can bear:

  1. THE CONDITIONAL ACCURACY. Of the composed-endpoint-Jacobian entries the committee is
     UNANIMOUS about, what fraction have the sign the plant actually has — against the same
     fraction where the members split. This needs no correlation and no ordering assumption;
     it is the direct statement that agreement carries information about being right.
  2. THE ORDERING. Rank correlation across rungs between `dir_agree` and each teacher's
     outcome, printed beside the same correlation for the oracle's `vjp_cos`. If agreement
     orders the rungs the way the oracle does, it can replace it where the oracle is
     unavailable — which is everywhere outside this experiment.
  3. THE POOLING COMPARISON. `ebl_committee` (mean of the members' teaching vectors) vs
     `ebl_sensory` through the committee MEAN MODEL vs `ebl_committee_gated` (mean, masked by
     agreement) vs `mixed_agree` (β read off agreement). The endpoint map composes the
     one-step model H times, so these are genuinely different objects.

Run (from `experiments/`):
    python3 mjc/jacobian_teacher/committee_figure.py --tags jc_s0
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    with open(os.path.join(HERE, "figures", f"jc_{tag}", "committee.json")) as fh:
        return json.load(fh)


def plateau_mean(curve, key, k=4):
    """Mean of the last `k` readouts, not the single final one: these arms oscillate around
    their optimum rather than settling (Adam at fixed lr on a shrinking gradient), so the last
    readout samples the phase of that oscillation. See the same helper in
    `jacobian_teacher_figure.py`."""
    v = [c[key] for c in curve if key in c][-k:]
    return float(np.mean(v)) if v else float("nan")


def grad_snr_early(curve, frac=0.25):
    """Mean `grad_snr` over the EARLY readouts only.

    The half-batch gradient cosine measures teacher noise, and it does so honestly only while
    there is a signal to be noisy about. At convergence the batch-mean action gradient is near
    zero, so the cosine between two half-batch means is dominated by whatever is left and swings
    freely — Phase A's `ebl_sensory` runs +0.996, +0.999, +0.998 early and then -0.842, +0.983,
    -0.932 at plateau. Averaging the whole run mixes a measurement with an artifact; the early
    window is the measurement.
    """
    rows = [c for c in curve if "grad_snr" in c]
    if not rows:
        return float("nan")
    cut = max(rows[-1]["reaches"] * frac, rows[0]["reaches"])
    early = [c["grad_snr"] for c in rows if c["reaches"] <= cut]
    return float(np.mean(early)) if early else float(rows[0]["grad_snr"])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        # A constant series has no ordering; argsort's index tie-break would invent one.
        # `rbl_cont` is FM-independent by construction and its column IS constant — it read
        # rho = +1.000 against the oracle before this guard.
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    R = [load(t) for t in args.tags]
    cfg = R[0]["config"]
    rungs = [l["name"] for l in R[0]["ladder"]]
    labels = sorted({k for r in R for l in r["ladder"] for k in l["arms"]})
    W = max(len(l) for l in labels) + 2

    print(f"\n{'=' * 118}\njacobian_teacher Phase C — committee direction agreement")
    print(f"tags: {', '.join(args.tags)}   K={cfg['k_ens']} rpf_beta={cfg['rpf_beta']}  "
          f"budget {cfg['reach_budget']} executed reaches/arm  n_eval={cfg['n_eval']}")
    print(f"shared initial motor program: "
          f"{np.mean([r['initial_policy']['train_band'] for r in R]):.4f}")

    def col(fn):
        return [float(np.mean([fn(r['ladder'][i]) for r in R])) for i in range(len(rungs))]

    print(f"\n--- 1. does agreement carry information about being RIGHT? (sign of the composed "
          f"endpoint Jacobian vs the finite-difference plant oracle) ---")
    print(f"{'rung':>16s} {'dir_agree':>10s} {'unanimous':>10s} {'split':>8s} {'lift':>7s} "
          f"{'corr':>8s} {'meanJ cos':>10s} {'oracle vjp_cos':>15s}")
    for i, name in enumerate(rungs):
        c = [r["ladder"][i]["committee"] for r in R]
        u = float(np.mean([x["correct_when_unanimous"] for x in c]))
        sp = float(np.mean([x["correct_when_split"] for x in c]))
        print(f"{name:>16s} {np.mean([x['dir_agree'] for x in c]):10.4f} {u:10.3f} {sp:8.3f} "
              f"{u - sp:+7.3f} {np.mean([x['agree_vs_correct_corr'] for x in c]):+8.4f} "
              f"{np.mean([x['mean_jac_vs_oracle']['cos'] for x in c]):10.4f} "
              f"{np.mean([r['ladder'][i]['readings']['jacE']['vjp_cos'] for r in R]):15.4f}")
    print("   ('lift' = how much more often a unanimous entry has the plant's sign than a "
          "split one. It is the whole claim, and it needs no ordering assumption.)")

    print(f"\n--- 2. does agreement ORDER the rungs the way the oracle does? "
          f"(Spearman across rungs; outcome is an error) ---")
    ag = col(lambda l: l["committee"]["dir_agree"])
    agw = col(lambda l: l["committee"]["dir_agree_w"])
    vj = col(lambda l: l["readings"]["jacE"]["vjp_cos"])
    fe = col(lambda l: l["readings"]["fm_err"])
    print(f"{'teacher':>{W}s} {'vs dir_agree':>13s} {'vs dir_agreeW':>14s} "
          f"{'vs vjp_cos (oracle)':>20s} {'vs fm_err':>10s}")
    PK = cfg.get("plateau_evals", 4)
    for l in labels:
        out = [float(np.mean([plateau_mean(r["ladder"][i]["arms"][l]["curve"],
                                           "train_band", PK)
                              for r in R if l in r["ladder"][i]["arms"]]))
               for i in range(len(rungs))]
        print(f"{l:>{W}s} {spearman(ag, out):13.3f} {spearman(agw, out):14.3f} "
              f"{spearman(vj, out):20.3f} {spearman(fe, out):10.3f}")
    print(f"   agreement vs the oracle's own reading, rung for rung: "
          f"Spearman(dir_agree, vjp_cos) = {spearman(ag, vj):+.3f}, "
          f"Spearman(dir_agreeW, vjp_cos) = {spearman(agw, vj):+.3f}")

    print(f"\n--- 3. how should a committee be pooled? PLATEAU-MEAN train-band endpoint "
          f"error ---")
    print(f"{'rung':>16s} " + "".join(f"{l:>{W}s}" for l in labels) +
          f"{'bal_cem':>{W}s}{'reactive':>{W}s}")
    for i, name in enumerate(rungs):
        row = f"{name:>16s} "
        for l in labels:
            v = [plateau_mean(r["ladder"][i]["arms"][l]["curve"], "train_band",
                              cfg.get("plateau_evals", 4))
                 for r in R if l in r["ladder"][i]["arms"]]
            row += f"{np.mean(v):{W}.4f}" if v else f"{'—':>{W}s}"
        inc = [r["ladder"][i]["incumbents"] for r in R]
        row += f"{np.mean([x['ballistic_cem'] for x in inc]):{W}.4f}"
        row += f"{np.mean([x['reactive'] for x in inc]):{W}.4f}"
        print(row)

    print(f"\n--- 4. speed, variability, and the β agreement actually chose ---")
    for key, cap in (("train_band_reaches_to_90", "reaches to 90% of own gain"),
                     ("train_band_plateau_sd", "plateau sd"),
                     ("grad_snr_early", "grad_snr, EARLY window (half-batch cosine)")):
        print(f"  {cap}")
        print(f"{'rung':>16s} " + "".join(f"{l:>{W}s}" for l in labels))
        for i, name in enumerate(rungs):
            row = f"{name:>16s} "
            for l in labels:
                v = ([grad_snr_early(r["ladder"][i]["arms"][l]["curve"])
                      for r in R if l in r["ladder"][i]["arms"]] if key == "grad_snr_early"
                     else [r["ladder"][i]["arms"][l]["summary"].get(key)
                           for r in R if l in r["ladder"][i]["arms"]])
                v = [x for x in v if x is not None and np.isfinite(x)]
                m = float(np.mean(v)) if v else float("nan")
                row += (f"{'—':>{W}s}" if not np.isfinite(m)
                        else (f"{int(m):{W}d}" if "reaches" in key else f"{m:{W}.4f}"))
            print(row)
    if "mixed_agree" in labels:
        print(f"  β that `mixed_agree` read off the committee (mean over the run)")
        for i, name in enumerate(rungs):
            b = [c["beta_eff"] for r in R
                 for c in r["ladder"][i]["arms"]["mixed_agree"]["curve"] if "beta_eff" in c]
            print(f"{name:>16s} {np.mean(b) if b else float('nan'):10.4f}")


if __name__ == "__main__":
    main()
