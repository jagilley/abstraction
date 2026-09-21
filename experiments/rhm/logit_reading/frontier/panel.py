"""Q1: the endogenous panel on a learner that climbs (`traj_a1_s42`, 13 checkpoints).

Per leaf level and per checkpoint, four candidate readings of "which rung am I absorbing":

  (a) the EXCESS, `CE_l - H(q)_l` (oracle) and `NLL_l - H(q)_l` (what the learner can
      actually compute). altitude Q1 says the identity drives this to zero in every class
      the model can scale separately, so it should be ~0 at every level on the training
      distribution -- absorbed levels, the rung being absorbed, and the levels above it.
      `ideas/calibration_and_violation_are_one_object.md` §10.2 carries a corrected
      version of this claim; the row is cheap and is carried here as a check.
  (b) the LOSS TREND, `-Delta CE_l` and `-Delta NLL_l` between adjacent checkpoints --
      learning progress, a derivative.
  (c) the PERIOD DETECTOR of altitude Q2 A, here also in a nested bottom-up form that
      turns the offset estimates into a LEVEL FIELD.
  (d) Q3's candidate gain, already computed (`altitude/results/tables.md` Q3b2); pulled
      in by the reduction, not recomputed.

Ground truth: the oracle's `KL(p_L || q)` by leaf level and its decrease across
checkpoints (altitude Q2 B, `altitude/frontier.py::residual_sweep`).

THE NEW ARM: everything in (a)-(b) recomputed with the level of each position assigned by
the model's own nested entropy-period detector instead of `leaf_levels`. Does the panel
survive losing the oracle's labels, and at which levels?

All of this is a CPU reduction of the per-checkpoint `stepNNNNNN_calibration.npz` arrays
`logit_reading/calibration.py` already wrote (logits, every observer's predictive, the
windows and their true phase; 4096 windows, eval_seed 4242, shared by all 13 checkpoints).

Run from experiments/:
  modal run --detach -m rhm.logit_reading.frontier.panel::panel_sweep \
      --traj-dir /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42
"""

import json
import os

import modal
import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.frontier.common import (
    _ignore, log_softmax, ent, xent, detrend, leaf_templates, detector_block,
    endo_levels, oracle_levels, level_panel, confusion)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-logit-frontier", image=image)

OUT_SUBDIR = "frontier"


def panel_ckpt(npz_path, L=6, s=2, j_lo=0):
    """The whole per-checkpoint panel from one calibration npz."""
    z_ = np.load(npz_path)
    z = z_["logits"].astype(np.float64)
    preds = z_["preds"].astype(np.float64)
    windows, phase = z_["windows"], z_["phase"]
    n, T, v = z.shape
    tok = windows[:, 1:]

    logq = log_softmax(z)
    q = np.exp(logq)
    p = preds[..., L, :]
    H_q = ent(q)
    H_p = ent(p)
    CE = xent(p, logq)
    KL = CE - H_p
    NLL = -np.take_along_axis(logq, tok[..., None], -1)[..., 0]

    leaf = (phase[:, None] + 1 + np.arange(T)[None, :]) % (s ** L)
    lev = oracle_levels(phase, T, L, s)
    H_k = np.stack([ent(preds[..., k, :]) for k in range(L + 1)], -1)
    tmpls = leaf_templates(H_k, leaf, L, s)
    del preds

    out = {"npz": npz_path, "n_windows": int(n), "T": int(T), "j_lo": j_lo,
           "overall": {"H_q": float(H_q.mean()), "H_p": float(H_p.mean()),
                       "CE": float(CE.mean()), "NLL": float(NLL.mean()),
                       "KL_pL_q": float(KL.mean()),
                       "excess_oracle": float((CE - H_q).mean()),
                       "excess_realised": float((NLL - H_q).mean()),
                       "se_excess_realised": float((NLL - H_q).std() / np.sqrt(NLL.size))}}

    # ---- (a)+(b) with ORACLE labels, and the ground truth ----
    out["by_level_oracle"] = level_panel(lev, L, H_q, NLL, CE, KL, H_p)

    # ---- (c) the detector, and the endogenous level field ----
    Xd = detrend(H_q)
    det = detector_block(Xd, tmpls, phase, L, s, j_lo)
    out["detector"] = {k: det[k] for k in ("acc_unconstrained_template",
                                           "acc_unconstrained_meanprof", "chance")}
    out["by_level_endo"] = {}
    out["confusion"] = {}
    for mode in ("template", "meanprof"):
        elev = endo_levels(det[mode]["r_hat_L"], T, L, s)
        out["by_level_endo"][mode] = level_panel(elev, L, H_q, NLL, CE, KL, H_p)
        out["confusion"][mode] = confusion(lev, elev, L)
        out["detector"][mode] = {"acc_nested": det[mode]["acc_nested"],
                                 "frac_exact_field": float((elev == lev).mean())}
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=8192, cpu=4.0)
def panel_sweep(traj_dir: str, out_name: str = "frontier_panel", j_lo: int = 0,
                limit: int = 0):
    import glob
    import resource
    import time
    volume.reload()
    paths = sorted(glob.glob(f"{traj_dir}/step*_calibration.npz"))
    if limit:
        paths, out_name = paths[:limit], f"{out_name}_limited"   # never clobber a full run
    print(f"{len(paths)} checkpoints in {traj_dir}", flush=True)
    rows = []
    for pth in paths:
        t0 = time.time()
        r = panel_ckpt(pth, j_lo=j_lo)
        r["step"] = int(os.path.basename(pth)[4:10])
        rows.append(r)
        bl = r["by_level_oracle"]
        print(f"step {r['step']:>6} ({time.time() - t0:.0f}s)  "
              f"excess_or " + " ".join(f"{bl[l]['excess_oracle']:+.3f}" for l in sorted(bl))
              + "  KL " + " ".join(f"{bl[l]['KL_pL_q']:.3f}" for l in sorted(bl))
              + "  nested_acc(t) " + " ".join(
                  f"{r['detector']['template']['acc_nested'][k]:.2f}" for k in range(1, 7))
              + f"  field {r['detector']['template']['frac_exact_field']:.2f}", flush=True)
    out_dir = f"{os.path.dirname(traj_dir.rstrip('/'))}/{OUT_SUBDIR}"
    os.makedirs(out_dir, exist_ok=True)
    dst = f"{out_dir}/{out_name}.json"
    with open(dst, "w") as f:
        json.dump(rows, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB",
          flush=True)
    print(f"saved -> {dst}", flush=True)
    return dst
