"""`H_pre` for every banked same-prefix twin pair (route ii of `precision/reduce.py`).

`norm/task.py` records `H_pre = H(q_(t0-1))` on every FIXED row but not on the twin pairs:
its `trace(..., want_lsm=True)` computes the log-softmax at `t_v - 1` for exactly those
windows and keeps only the two surprisals `s_v`, `s_c`.  The fixed rows cover the twins'
windows -- the two eligibility conditions are the same -- but are saved for the `split == 2`
TEST windows only, so joining them supplies the axis for about a quarter of the pairs, which
is too thin for the 2-D surface.  This file closes the gap with ONE forward pass per cell
and reproduces nothing else: the pair set, the twin selection and the critic are read off
the banked npz exactly as they were saved.

It writes `<ckpt stem>_norm_<tag>_Hpre.npz` beside the banked file, with
`tw_w`, `tw_t_v`, `tw_H_pre` and `tw_s_v` in the banked pairs' own order, plus the same
`gl_*` columns when the cell carries the glitch world (there `H_pre` is on the CLEAN prefix,
which is why the join cannot supply it at all).

**Gate.**  The recomputed violator surprisal is compared against the banked `tw__s_v` on
every pair and the max absolute difference is returned and asserted below `--tol`; if the
checkpoint, the stimuli or the window indexing were not the ones the cell was run with, that
number is large and the cell fails rather than writing a wrong column.

Run:
  modal run -m rhm.logit_reading.striatum.norm.precision.hpre::hpre_sweep
  modal run -m rhm.logit_reading.striatum.norm.precision.hpre::hpre_cell \
      --ckpt /data/.../traj_a1_s42/step064000.pt --stim-tag swap65k
"""

import json
import os
import resource
import time

import numpy as np

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import load_trajectory_ckpt, tb_key


def _ignore(path):
    sp = str(path)
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    return not sp.endswith(".py")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-norm-precision-hpre", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=1800, memory=6144,
              max_containers=6)
def hpre_cell(ckpt: str, stim_tag: str = "swap65k", tag: str = "", bs: int = 512,
              tol: float = 2e-3):
    """One (checkpoint, venue) cell: the entropy of the trunk's forecast at `t_v - 1` for
    every banked twin pair, on the stream that pair's prefix lives on."""
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    T = model.block_size
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")

    stem = ckpt[:-3]
    sfx = f"_norm_{stim_tag}" + (f"_{tag}" if tag else "")
    npz_p, json_p = f"{stem}{sfx}.npz", f"{stem}{sfx}.json"
    assert os.path.exists(npz_p), f"missing {npz_p}"
    Z = np.load(npz_p)
    r = json.load(open(json_p)) if os.path.exists(json_p) else {}

    # the cell's own window subsample: the sweep runs with max_windows = 0, so the banked
    # `w` index directly into the full stimulus arrays.  Asserted, never assumed.
    n_all = int(S["windows_edit"].shape[0])
    assert int(r.get("n", n_all)) == n_all, (
        f"{npz_p} was run on {r.get('n')} of {n_all} windows; the banked `w` do not index "
        f"the full stimulus array and this file would read the wrong prefixes")

    def entropies(W, w_idx, tpos):
        """(H at t-1, surprisal of the token at t) for each (window, position)."""
        H = np.zeros(len(w_idx), np.float64)
        sv = np.zeros(len(w_idx), np.float64)
        with torch.no_grad():
            for c0 in range(0, len(w_idx), bs):
                sl = slice(c0, min(c0 + bs, len(w_idx)))
                x = torch.as_tensor(W[w_idx[sl], :T], device=dev)
                lg, _, _ = model(x, return_intermediates=True)
                lsm = torch.log_softmax(lg.float(), -1)
                tt = torch.as_tensor(tpos[sl], device=dev)
                ar = torch.arange(x.shape[0], device=dev)
                q = lsm[ar, tt - 1]                                  # (b, v)
                H[sl] = (-(q.exp() * q).sum(-1)).cpu().numpy()
                sv[sl] = (-q.gather(-1, x[ar, tt][:, None])[:, 0]).cpu().numpy()
        return H, sv

    out, meta = {}, {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag,
                     "tag": tag, "n_windows": n_all}
    for pre, key_, stream in (("tw__", "tw", "windows_edit"),
                              ("gl__", "gl", "windows_orig")):
        if pre + "w" not in Z.files:
            continue
        w = np.asarray(Z[pre + "w"]).astype(np.int64)
        tv = np.asarray(Z[pre + "t_v"]).astype(np.int64)
        W = S[stream]
        H, sv = entropies(W, w, tv)
        dev_s = float(np.abs(sv - np.asarray(Z[pre + "s_v"], np.float64)).max())
        meta[f"{key_}_n_pairs"] = int(len(w))
        meta[f"{key_}_max_abs_ds_v_against_banked"] = dev_s
        meta[f"{key_}_H_mean"] = float(H.mean())
        meta[f"{key_}_H_sd"] = float(H.std())
        meta[f"{key_}_stream"] = stream
        out[f"{key_}_w"] = w
        out[f"{key_}_t_v"] = tv
        out[f"{key_}_H_pre"] = H.astype(np.float32)
        out[f"{key_}_s_v"] = sv.astype(np.float32)
        print(f"{pre}: {len(w)} pairs  H {H.mean():.4f} +- {H.std():.4f}  "
              f"max |s_v - banked| {dev_s:.2e}", flush=True)
        # the glitch world's flagged token is on the CLEAN prefix; the banked `s_v` there
        # is `s_g`, the glitch's own surprisal under that prefix, so the same gate applies.
        assert dev_s <= tol, (
            f"{pre}: recomputed s_v differs from the banked column by {dev_s:.3e} > {tol}; "
            f"the checkpoint, the stimuli or the window indexing is not the cell's")
    assert out, f"{npz_p} carries no twin pairs"
    meta["_gate"] = "recomputed s_v matches the banked column on every pair"
    out["meta"] = json.dumps(meta)
    p = f"{stem}{sfx}_Hpre.npz"
    np.savez_compressed(p, **out)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {p}  ({time.time() - t00:.0f}s, peak RSS {rss:.2f} GB)", flush=True)
    print(json.dumps(meta, cls=NumpyEncoder), flush=True)
    return meta


@app.function(volumes={DATA_DIR: volume}, timeout=2 * 3600, memory=2048)
def hpre_sweep(cells: str = "", seeds: str = "42,43,44", steps: str = "0,8000,64000",
               tags: str = "a1,swap65k", tag: str = ""):
    """One container per cell, fanned out from a CPU coordinator."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    args = []
    if cells:
        for c in cells.split(","):
            p = c.split(":")
            args.append((p[0], p[1], tag))
    else:
        for s in seeds.split(","):
            for st in steps.split(","):
                ck = f"{D}/traj_a1_s{s}/step{int(st):06d}.pt"
                if not os.path.exists(ck):
                    print(f"MISSING {ck}", flush=True)
                    continue
                for tg in tags.split(","):
                    if not os.path.exists(f"{ck[:-3]}_norm_{tg}"
                                          + (f"_{tag}" if tag else "") + ".npz"):
                        print(f"MISSING npz for {ck} {tg}", flush=True)
                        continue
                    args.append((ck, tg, tag))
    print(f"{len(args)} cells", flush=True)
    outs = list(hpre_cell.starmap(args, return_exceptions=True))
    for a, o in zip(args, outs):
        print(f"{a[0].split('/')[-2]}/{a[0].split('/')[-1]} {a[1]}: "
              f"{json.dumps(o, cls=NumpyEncoder, default=str)[:400]}", flush=True)
    return [str(o)[:400] for o in outs]
