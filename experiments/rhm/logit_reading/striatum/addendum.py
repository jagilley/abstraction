"""Addendum column: the model's REALISED horizon excess surprise on the stimulus windows.

`task.py` stores the single-position excess at the anchor, `-log q_{t0-1}(x_t0) - H(q_{t0-1})`.
The banked `coeruleus/` head predicts a different object -- the horizon sum

    y_t = sum_{u=t}^{t+H} ( -log q_u(x_{u+1}) - H(q_u) ),   H = 8

so "does the excess head read damage through something other than the surprise it was
trained to predict" needs the horizon sum as the comparator, not the one-position excess.
This writes it for every window and position, non-destructively, beside the main artefacts.

Run:
  modal run -m rhm.logit_reading.striatum.addendum::hexcess_ckpt \
      --ckpt /data/v16_s2_L6_m4_distinct/logit_reading/traj_a1_s42/step064000.pt \
      --stim-tag swap65k
"""

import numpy as np

from rhm.shared import volume, DATA_DIR
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key
from rhm.logit_reading.striatum.task import app


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def hexcess_ckpt(ckpt: str, stim_tag: str = "swap65k", horizon: int = 8):
    H = horizon
    import torch
    volume.reload()
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, "cuda")
    T = model.block_size
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")

    def run(W):
        n = len(W)
        nll = np.zeros((n, T), np.float32)
        Hq = np.zeros((n, T), np.float32)
        with torch.no_grad():
            for c0 in range(0, n, 256):
                sl = slice(c0, min(c0 + 256, n))
                x = torch.as_tensor(W[sl, :T], device="cuda")
                lg, _ = model(x)
                lsm = torch.log_softmax(lg.float(), -1)
                nxt = torch.as_tensor(W[sl, 1:T + 1], device="cuda")
                nll[sl] = (-lsm.gather(-1, nxt[..., None])[..., 0]).cpu().numpy()
                Hq[sl] = (-(lsm.exp() * lsm).sum(-1)).cpu().numpy()
        d = (nll - Hq).astype(np.float64)
        c = np.concatenate([np.zeros((n, 1)), np.cumsum(d, 1)], 1)
        out = np.full((n, T), np.nan, np.float32)
        out[:, :T - H] = (c[:, H + 1:] - c[:, :-(H + 1)]).astype(np.float32)
        return out, nll, Hq

    he, _, _ = run(S["windows_edit"])
    ho, _, _ = run(S["windows_orig"])
    stem = ckpt[:-3]
    np.savez_compressed(f"{stem}_striatum_{stim_tag}_hexcess.npz",
                        hexcess_edit=he, hexcess_orig=ho, H=np.int64(H))
    volume.commit()
    print(f"saved -> {stem}_striatum_{stim_tag}_hexcess.npz  "
          f"finite frac {np.isfinite(he).mean():.3f}", flush=True)
    return {"ckpt": ckpt, "tag": stim_tag, "H": H}


@app.function(volumes={DATA_DIR: volume}, timeout=2 * 3600, memory=2048)
def hexcess_sweep(cells: str = "", horizon: int = 8):
    """CPU coordinator: `ckpt:stim_tag` cells, comma separated, one container each.

    Added 2026-09-17 for `shaped/`, which needs this column on three fine-tuned trunks x
    two venues; the single-cell entry point above is unchanged."""
    import os
    volume.reload()
    args = []
    for c in cells.split(","):
        if not c.strip():
            continue
        ck, tg = (c.split(":") + ["swap65k"])[:2]
        if not os.path.exists(ck):
            print(f"MISSING {ck}", flush=True)
            continue
        args.append((ck, tg, horizon))
    print(f"{len(args)} cells: {[(a[0].split('/')[-2:], a[1]) for a in args]}", flush=True)
    outs = list(hexcess_ckpt.starmap(args, return_exceptions=True))
    for o in outs:
        print(str(o)[:300], flush=True)
    return [str(o)[:300] for o in outs]
