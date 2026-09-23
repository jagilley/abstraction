"""Mediators of the event, recorded on the banked `norm/` rows: how far the world model
actually MOVED when the flagged token arrived.

`precision/reduce.py` found the value response's dependence on `H_pre` at matched surprisal.
This file records, on exactly the banked rows and pairs, the two candidate carriers of that
dependence:

  `kl_update`  -- `phasic.py`'s object, `KL(q_t || q_(t-1))`, the model's FORECAST movement
                  across the event (`violation._kl(exp(lq_next), lq_prev)`; computed here in
                  log space, which is the same quantity without the 1e-30 clip).
  `H_next`     -- `H(q_t)`, the entropy after the event, beside `H_pre` before it.
  `dstate_<b>` -- `||s_t - s_(t-1)||_2` at block `b`, the REPRESENTATION's movement across
                  the event.  `post_block7` is `norm/task.py`'s `prim`, the block the critic
                  is fitted on; `post_block3` and `post_embed` are the cheap insurance.
  `dstate_pair_<b>` (pairs only) -- `||s_t^flagged - s_t^twin||_2`, the state difference
                  whose linear projection is `dR` (`orbitofrontal/projection/` used it as the
                  exact ceiling).

Nothing is refit and no banked file is rewritten: the rows, the pair sets, the twin
selection and the critics are all READ from the banked npz, and only these columns are
computed.  Output: `<ckpt stem>_norm_<tag>[_<tag2>]_med.npz`.

**Gates**, asserted per cell and returned, so a wrong checkpoint / stimulus / index can never
be written silently:
  fixed rows -- the recomputed `H_pre` and the recomputed surprisal at the anchor must equal
                the banked `<an>__H_pre` and `<an>__nll_e` on every row;
  pairs      -- the recomputed flagged and twin surprisals must equal the banked `s_v`, `s_c`
                on every pair, and the two members' `H_pre` must be identical (the prefix is
                bit-identical by construction, which `task.py` also asserts).

The glitch world's own `H_pre` lives on the CLEAN prefix, so it cannot come from a join
against the fixed rows at all; this pass supplies it, which is why the glitch cells need it.

Run:
  modal run -m rhm.logit_reading.striatum.norm.precision.mediate::mediate_sweep
  modal run -m rhm.logit_reading.striatum.norm.precision.mediate::mediate_sweep \
      --tags swap65k --tag glitch
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
app = modal.App("rhm-norm-precision-mediate", image=image)

BLOCKS = ["post_block7", "post_block3", "post_embed"]


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=8192,
              max_containers=6)
def mediate_cell(ckpt: str, stim_tag: str = "swap65k", tag: str = "", bs: int = 384,
                 tol: float = 2e-3, blocks: str = ",".join(BLOCKS)):
    import torch
    volume.reload()
    t00 = time.time()
    dev = "cuda"
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, dev)
    T = model.block_size
    bl = [b for b in blocks.split(",") if b]
    key = tb_key(cfg["v"], cfg["s"], cfg["L"], cfg["m"])
    ddir = f"{DATA_DIR}/{key}/logit_reading"
    S = np.load(f"{ddir}/stimuli_{stim_tag}.npz")

    stem = ckpt[:-3]
    sfx = f"_norm_{stim_tag}" + (f"_{tag}" if tag else "")
    npz_p, json_p = f"{stem}{sfx}.npz", f"{stem}{sfx}.json"
    assert os.path.exists(npz_p), f"missing {npz_p}"
    Z = np.load(npz_p)
    r = json.load(open(json_p)) if os.path.exists(json_p) else {}
    n_all = int(S["windows_edit"].shape[0])
    assert int(r.get("n", n_all)) == n_all, (
        f"{npz_p} was run on {r.get('n')} of {n_all} windows; the banked indices do not "
        f"index the full stimulus array")

    def batch(Xf, tt, Xt=None):
        """One batch of (window, position).  Returns the per-row event statistics of the
        flagged stream, and, when a twin stream is given, the same for it plus the pair's
        state distance.  `tt` is the event position; `q_(t-1)` is the forecast OF the token
        at `tt`, made one position earlier, so `H_pre` depends only on the shared prefix."""
        ar = torch.arange(Xf.shape[0], device=dev)

        def one(X):
            lg, _, inter = model(X, return_intermediates=True)
            lsm = torch.log_softmax(lg.float(), -1)
            qp, qn = lsm[ar, tt - 1], lsm[ar, tt]
            st = {b: inter[b][ar, tt].float() for b in bl}
            sm = {b: inter[b][ar, tt - 1].float() for b in bl}
            o = {"H_pre": -(qp.exp() * qp).sum(-1),
                 "H_next": -(qn.exp() * qn).sum(-1),
                 # violation._kl(exp(lq_next), lq_prev) == sum q_t (log q_t - log q_(t-1))
                 "kl_update": (qn.exp() * (qn - qp)).sum(-1),
                 "s_flag": -qp.gather(-1, X[ar, tt][:, None])[:, 0]}
            for b in bl:
                o[f"dstate_{b}"] = (st[b] - sm[b]).norm(dim=-1)
            return o, st

        of, stf = one(Xf)
        if Xt is None:
            return of, None
        ot, stt = one(Xt)
        for b in bl:
            of[f"dstate_pair_{b}"] = (stf[b] - stt[b]).norm(dim=-1)
        return of, ot

    def run(Wf, tpos, Wt=None):
        acc_f, acc_t = {}, {}
        with torch.no_grad():
            for c0 in range(0, len(Wf), bs):
                sl = slice(c0, min(c0 + bs, len(Wf)))
                Xf = torch.as_tensor(Wf[sl, :T], device=dev)
                Xt = None if Wt is None else torch.as_tensor(Wt[sl, :T], device=dev)
                tt = torch.as_tensor(tpos[sl], device=dev)
                of, ot = batch(Xf, tt, Xt)
                for k, v in of.items():
                    acc_f.setdefault(k, []).append(v.cpu().numpy())
                if ot is not None:
                    for k, v in ot.items():
                        acc_t.setdefault(k, []).append(v.cpu().numpy())
        f = {k: np.concatenate(v).astype(np.float64) for k, v in acc_f.items()}
        t = {k: np.concatenate(v).astype(np.float64) for k, v in acc_t.items()} or None
        return f, t

    out, meta = {}, {"ckpt": ckpt, "step": cfg.get("step"), "stim_tag": stim_tag,
                     "tag": tag, "blocks": bl, "n_windows": n_all}

    # ---- the fixed rows, both anchors, on the EDITED stream -------------------------------
    We = S["windows_edit"]
    for an in ("tv", "fd"):
        if f"{an}__t0" not in Z.files:
            continue
        w = np.asarray(Z[f"{an}__w"]).astype(np.int64)
        t0 = np.asarray(Z[f"{an}__t0"]).astype(np.int64)
        f, _ = run(We[w], t0)
        gH = float(np.abs(f["H_pre"] - np.asarray(Z[f"{an}__H_pre"], np.float64)).max())
        gS = float(np.abs(f["s_flag"] - np.asarray(Z[f"{an}__nll_e"], np.float64)).max())
        meta[f"{an}_n"] = int(len(w))
        meta[f"{an}_gate_H_pre"] = gH
        meta[f"{an}_gate_nll_e"] = gS
        out[f"{an}_w"], out[f"{an}_t0"] = w, t0
        for k, v in f.items():
            out[f"{an}_{k}"] = v.astype(np.float32)
        print(f"{an}: {len(w)} rows  gate H_pre {gH:.2e}  gate nll_e {gS:.2e}  "
              f"kl {f['kl_update'].mean():.4f}  dstate7 {f['dstate_post_block7'].mean():.3f}",
              flush=True)
        assert gH <= tol and gS <= tol, (
            f"{an}: recomputed H_pre / surprisal differ from the banked columns by "
            f"{gH:.3e} / {gS:.3e} > {tol}")

    # ---- the pairs, both worlds ------------------------------------------------------------
    # `tw__` lives on the EDITED stream (its prefix carries the edit); `gl__` lives on the
    # UNEDITED one (the glitch is flagged under a CLEAN prefix), which is why the glitch
    # world's H_pre cannot come from a join against the fixed rows.
    for pre, kk, stream in (("tw__", "tw", "windows_edit"), ("gl__", "gl", "windows_orig")):
        if pre + "w" not in Z.files:
            continue
        w = np.asarray(Z[pre + "w"]).astype(np.int64)
        tv = np.asarray(Z[pre + "t_v"]).astype(np.int64)
        xv = np.asarray(Z[pre + "x_v"]).astype(np.int64)
        xc = np.asarray(Z[pre + "x_c"]).astype(np.int64)
        W = S[stream]
        ar = np.arange(len(w))
        Wf, Wt = W[w].copy(), W[w].copy()
        if pre == "tw__":
            assert (Wf[ar, tv] == xv).all(), "tw__: the edited stream's own token is not x_v"
        Wf[ar, tv], Wt[ar, tv] = xv, xc
        f, t = run(Wf, tv, Wt)
        gV = float(np.abs(f["s_flag"] - np.asarray(Z[pre + "s_v"], np.float64)).max())
        gC = float(np.abs(t["s_flag"] - np.asarray(Z[pre + "s_c"], np.float64)).max())
        gP = float(np.abs(f["H_pre"] - t["H_pre"]).max())
        meta[f"{kk}_n_pairs"] = int(len(w))
        meta[f"{kk}_gate_s_v"] = gV
        meta[f"{kk}_gate_s_c"] = gC
        meta[f"{kk}_gate_H_pre_within_pair"] = gP
        meta[f"{kk}_stream"] = stream
        meta[f"{kk}_H_mean"] = float(f["H_pre"].mean())
        meta[f"{kk}_H_sd"] = float(f["H_pre"].std())
        out[f"{kk}_w"], out[f"{kk}_t_v"] = w, tv
        out[f"{kk}_H_pre"] = f["H_pre"].astype(np.float32)
        for k in ("H_next", "kl_update", "s_flag"):
            out[f"{kk}_{k}_v"] = f[k].astype(np.float32)
            out[f"{kk}_{k}_t"] = t[k].astype(np.float32)
        for b in bl:
            out[f"{kk}_dstate_{b}_v"] = f[f"dstate_{b}"].astype(np.float32)
            out[f"{kk}_dstate_{b}_t"] = t[f"dstate_{b}"].astype(np.float32)
            out[f"{kk}_dstate_pair_{b}"] = f[f"dstate_pair_{b}"].astype(np.float32)
        print(f"{pre}: {len(w)} pairs  gate s_v {gV:.2e}  s_c {gC:.2e}  "
              f"H_pre within pair {gP:.2e}  H {f['H_pre'].mean():.4f}  "
              f"dpair7 {f['dstate_pair_post_block7'].mean():.3f}", flush=True)
        assert gV <= tol and gC <= tol, (
            f"{pre}: recomputed surprisals differ from the banked columns by "
            f"{gV:.3e} / {gC:.3e} > {tol}")
        assert gP <= tol, f"{pre}: the pair's two H_pre differ by {gP:.3e}"

    assert out, f"{npz_p} carries neither fixed rows nor pairs"
    meta["_gate"] = ("recomputed H_pre / surprisals match the banked columns on every row "
                     "and every pair")
    out["meta"] = json.dumps(meta)
    p = f"{stem}{sfx}_med.npz"
    np.savez_compressed(p, **out)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"saved -> {p}  ({time.time() - t00:.0f}s, peak RSS {rss:.2f} GB)", flush=True)
    print(json.dumps(meta, cls=NumpyEncoder), flush=True)
    return meta


@app.function(volumes={DATA_DIR: volume}, timeout=4 * 3600, memory=2048)
def mediate_sweep(cells: str = "", seeds: str = "42,43,44",
                  steps: str = "0,8000,64000", tags: str = "a1,swap65k", tag: str = "",
                  blocks: str = ",".join(BLOCKS)):
    """One container per cell.  `--tag glitch` reads the `_glitch` cells instead."""
    volume.reload()
    D = f"{DATA_DIR}/v16_s2_L6_m4_distinct/logit_reading"
    args = []
    if cells:
        for c in cells.split(","):
            p = c.split(":")
            args.append((p[0], p[1], tag, 384, 2e-3, blocks))
    else:
        for s in seeds.split(","):
            for st in steps.split(","):
                ck = f"{D}/traj_a1_s{s}/step{int(st):06d}.pt"
                if not os.path.exists(ck):
                    print(f"MISSING {ck}", flush=True)
                    continue
                for tg in tags.split(","):
                    q = f"{ck[:-3]}_norm_{tg}" + (f"_{tag}" if tag else "") + ".npz"
                    if not os.path.exists(q):
                        print(f"MISSING {q}", flush=True)
                        continue
                    args.append((ck, tg, tag, 384, 2e-3, blocks))
    print(f"{len(args)} cells", flush=True)
    outs = list(mediate_cell.starmap(args, return_exceptions=True))
    for a, o in zip(args, outs):
        print(f"{a[0].split('/')[-2]}/{a[0].split('/')[-1]} {a[1]} {a[2] or '-'}: "
              f"{json.dumps(o, cls=NumpyEncoder, default=str)[:500]}", flush=True)
    return [str(o)[:500] for o in outs]
