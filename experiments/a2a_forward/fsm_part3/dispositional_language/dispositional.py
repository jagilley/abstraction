"""Dispositional self-knowledge: does the first-person advantage extend from "what am I
computing right now" to "how is my computation changing"?

Parent battery: a2a_forward/confabulation/confabulation.py  (read its README first)
Paper:          papers/forward_self_models_paper2.md  (SS2-5, SS8)
Design notes:   DESIGN.md  (every choice below, with its reason)

THE QUESTION
  Paper 2's target is OCCURRENT: at one position, at one training snapshot, the report
  head names the direction class of M's own forward-model residual r = a_j - FM(a_i), and
  no capacity-matched outside observer given tokens + M's output distribution matches it.
  For behavioural facts (own correctness, own entropy) the observer matches or beats the
  self-report. That dissociation is the result.

  This node asks whether the same privilege extends to DISPOSITIONAL facts -- facts about
  M's own dynamics rather than its instantaneous state:

    IMPL-prospective    will my computation here reorganize between now and c+k
    BEHAV-prospective   will my answer here get better between now and c+k
    IMPL-retrospective  how much has my computation here churned over the last w snapshots
    BEHAV-retrospective how much has my loss here dropped over the last w snapshots

  These are implementation and behaviour facts respectively, so the framework's prediction
  is that the two IMPL rows carry an advantage and the two BEHAV rows do not, in both time
  directions. Any other pattern is informative.

WHAT IS DIFFERENT FROM THE PARENT BATTERY
  1. M is trained ONCE (the OL arm, 10k steps, the parent's recipe verbatim, down to the
     RNG consumption order) and CHECKPOINTED along the trajectory. Everything downstream
     runs at every checkpoint on the SAME held-out report sequences, so a target can be a
     per-position quantity indexed by (position, checkpoint). The final wake state is also
     written to the harness's standard wake_ckpt/{ck_key}.pt so a sibling can load it.
  2. The report head is trained POOLED across checkpoints and evaluated both on held-out
     sequences at training checkpoints and on held-out sequences at a held-out contiguous
     CHECKPOINT BLOCK -- so the result is a rule about states, not a per-checkpoint
     constant the head memorised.
  3. Observers get a learned CHECKPOINT-INDEX embedding. Without it O_input is degenerate
     (the report tokens are identical at every checkpoint, so a tokens-only observer could
     not even tell which snapshot it is looking at). An outsider watching a training run
     obviously knows which snapshot it sees; giving it the index is the generous choice.
  4. A HISTORY OBSERVER (O_hist) is added: tokens + M's own output summary at checkpoints
     c, c-1, ..., c-L. This is the natural ceiling for the retrospective targets -- it asks
     whether the fact is knowable from M's behavioural history at all. If O_hist beats the
     self-report on a retrospective row, that row is public in exactly the sense paper 2's
     ENT row is public.

THE INSTRUMENT-VARIATION SUBTLETY (and the four readings that answer it)
  r_c and r_{c+k} come from DIFFERENT freshly-trained instrument FMs, so a naive
  "1 - cos(r_c, r_{c+k})" mixes M's own change with instrument variation. Three readings,
  all computed:
    raw      1 - cos(r_c^{(0)}, r_{c+k}^{(0)})               -- the naive quantity
    excess   raw - floor_c(p), where floor_c is the mean over the ens_n fresh FMs at the
             SAME checkpoint of 1 - cos(r_c^{(m)}, r_c^{(m')}) -- the no-change floor,
             subtracted per position
    fixed    1 - cos(r_c^{FM_c}, (a_j at c+k) - FM_c(a_0 at c+k)) -- the instrument is
             literally held fixed: the FM trained at c is applied at c+k
  The fourth control is structural: the categorical targets are WITHIN-CHECKPOINT quantile
  classes, so anything constant across positions at a given (c, k) -- residual-stream basis
  drift, a global instrument-variation level, the global learning rate -- is removed by
  construction, which is also what stops a pooled head from scoring by memorising a
  per-checkpoint constant.

THE JUNK-RESIDUAL TRAP still applies, per checkpoint
  Every IMPL number is computed at two instrument capacities (16:0.5 default, 4:0.25) and
  every checkpoint carries its own `ens_cos` and residual-structure readings. Early
  checkpoints are the live risk here rather than over-capacity: a barely-trained M is
  nearly trivial to predict, the FM saturates, and the residual is noise (see
  rhm/residual_decomposition/trajectory -- at step 0 a fresh FM reads cosine ~0.996). The
  schedule therefore starts at step 300 and the per-checkpoint guard table is printed, so
  any row can be dropped after the fact.

THE AUTOCORRELATION GATE (committee_head Phase B's lesson)
  mjc/committee_head trained a head on FM-derived signals to predict future learning
  progress and sat at AUROC 0.459 because the per-round target's autocorrelation was
  +0.007 -- white noise about a per-region mean. "Probe-able does not imply learnable from
  a given target." So before any head is trained, every target's across-checkpoint lag-1
  autocorrelation and between/within variance split are computed and printed. Smoothed
  variants (mean over k = 1..4) and a horizon sweep are computed unconditionally, so a
  widen-k or smoothing fallback needs no second launch.

SHAPE OF THE JOB
  `dispositional_test` is a CPU coordinator. It runs the wake trajectory on one GPU
  container, fans the per-checkpoint instrument work out over `max_dop` GPU containers
  (independent by construction -- same GPU-hours, a fraction of the wall clock), then runs
  the pooled battery in one GPU container. Every phase is resumable from the volume: the
  wake checkpoints, the per-checkpoint parts and the fresh-FM weights are all cached.

Run (chromatic; the FineWeb-Edu shards are at /data/tokens on `language-reduction-data`):

  # smoke (~10 min, everything tiny, numbers meaningless)
  modal run a2a_forward/fsm_part3/dispositional_language/dispositional.py::dispositional_test \
      --smoke --tag smoke

  # headline
  modal run --detach \
      a2a_forward/fsm_part3/dispositional_language/dispositional.py::dispositional_test \
      --tag main
"""

import json
import os

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder
from a2a_forward.confabulation.confabulation import (
    _kmeans_fit, _kmeans_assign, _cluster_quality, _residual_structure,
    _ensemble_cos, _make_head, _balance, _syntactic_labels,
)


# ======================================================================
# Small utilities
# ======================================================================

def ckpt_schedule(n_steps, n_ckpt, first=300):
    """Log-spaced checkpoints. Learning is front-loaded, so equal RATIOS of steps buy
    roughly equal amounts of learning per interval -- which is what makes "horizon k in
    checkpoints" a comparable unit across the trajectory."""
    first = max(1, min(first, n_steps // 4))
    r = (n_steps / first) ** (1.0 / (n_ckpt - 1))
    steps = sorted({int(round(first * r ** i)) for i in range(n_ckpt)})
    steps[-1] = n_steps
    return steps


def _apply_lnf(x, ci_rows, lnf_w, lnf_b):
    """M's own final LayerNorm, per row's checkpoint. The report site sits before ln_f in
    M, but M's own output channel reads ln_f(a_rep), so this is still M's state -- and it
    removes the activation-scale drift along a training trajectory, which would otherwise
    let a pooled head read "which checkpoint am I at" off the norm alone."""
    xn = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    return xn * lnf_w[ci_rows] + lnf_b[ci_rows]


def _spearman(pred, targ):
    """Spearman rho between two 1-D tensors (ranks then Pearson)."""
    import torch
    n = pred.numel()
    if n < 3:
        return float("nan")
    rp = torch.empty(n); rp[torch.argsort(pred)] = torch.arange(n, dtype=torch.float)
    rt = torch.empty(n); rt[torch.argsort(targ)] = torch.arange(n, dtype=torch.float)
    rp = rp - rp.mean(); rt = rt - rt.mean()
    den = float(rp.norm() * rt.norm())
    return float((rp * rt).sum() / den) if den > 0 else float("nan")


def _cos_rows(u, v, chunk=256):
    """Per-position cosine between two (N, P, d) tensors of UNIT rows, held in fp16 and
    accumulated in fp32 a few sequences at a time (a full-tensor fp32 cast is ~0.7 GB per
    operand at the report-set sizes used here)."""
    import torch
    out = torch.empty(u.shape[0], u.shape[1], dtype=torch.float32)
    for i in range(0, u.shape[0], chunk):
        out[i:i + chunk] = (u[i:i + chunk].float() * v[i:i + chunk].float()).sum(-1)
    return out


def _unit16(r, P, chunk=256):
    """Row-normalise (N, T, d) -> (N, P, d) unit directions in fp16, chunked."""
    import torch
    import torch.nn.functional as F
    out = torch.empty(r.shape[0], P, r.shape[-1], dtype=torch.float16)
    for i in range(0, r.shape[0], chunk):
        out[i:i + chunk] = F.normalize(r[i:i + chunk, :P].float(), dim=-1).half()
    return out


def _derive(cfg):
    """Values every phase needs, derived once from the config dict."""
    d = dict(cfg)
    d["T"] = cfg["block_size"]
    d["cib"] = int(cfg["predict_from"].replace("post_block", "").replace("post_embed", "-1"))
    d["j_idx"] = int(cfg["predict_to"].replace("post_block", ""))
    rb = cfg["report_block"] or f"post_block{cfg['n_layer'] - 1}"
    d["report_block"] = rb
    d["r_idx"] = int(rb.replace("post_block", ""))
    d["mid_blocks"] = list(range(d["j_idx"] + 1, d["r_idx"] + 1))
    d["n_pred_blocks"] = d["j_idx"] - d["cib"]
    d["n_pred_params"] = d["n_pred_blocks"] * 12 * cfg["n_embd"] ** 2
    d["inst_caps"] = [(int(c.split(":")[0]), float(c.split(":")[1]))
                      for c in cfg["inst_caps_str"].split(",")]
    d["cap_tags"] = [f"h{dh}m{mm:g}" for (dh, mm) in d["inst_caps"]]
    d["caps"] = [tuple(int(z) for z in c.split(":")) for c in cfg["observer_caps"].split(",")]
    d["horizons"] = [int(z) for z in cfg["horizons_str"].split(",")]
    d["out_dir"] = f"{DATA_DIR}/a2a_forward/confabulation/dispositional/{cfg['tag'] or 'run'}"
    d["ck_dir"] = f"{d['out_dir']}/ckpt"
    d["fm_dir"] = f"{d['out_dir']}/fm"
    d["part_dir"] = f"{d['out_dir']}/parts"
    return d


def _load_tokens(cfg):
    import glob
    import numpy as np
    import torch
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]
    all_tokens, total = [], 0
    for path in sorted(glob.glob(os.path.join(data_dir, "shard_*.npy"))):
        toks = np.load(path)
        all_tokens.append(toks)
        total += len(toks)
        if total >= cfg["n_tokens"]:
            break
    data = torch.from_numpy(np.concatenate(all_tokens)[:cfg["n_tokens"]].astype(np.int64))
    split = int(0.9 * len(data))
    return data[:split], data[split:], vocab_size


def _report_set(cfg, val_data):
    import numpy as np
    import torch
    T = cfg["block_size"]
    rng = np.random.default_rng(cfg["report_seed"])
    starts = rng.integers(0, len(val_data) - T - 1, size=cfg["n_report_sequences"])
    return torch.stack([val_data[i:i + T] for i in starts])


# ======================================================================
# Phase 1: the wake trajectory (OL arm), checkpointed
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=12288)
def wake_trajectory(cfg: dict):
    """Forks confabulation_test's OL wake path VERBATIM, including the RNG consumption
    order (the throwaway `_init` GPT, the re-seed, the forward-model construction), so the
    final model is bit-for-bit the object the parent battery's OL arm would have produced
    at this seed -- which is what lets it be written to the harness's standard wake_ckpt
    path for a sibling to load. Checkpoints are saved along the way."""
    import time
    import torch
    import torch.nn.functional as F
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    d = _derive(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, seed = d["T"], cfg["seed"]
    train_data, val_data, vocab_size = _load_tokens(cfg)
    os.makedirs(d["ck_dir"], exist_ok=True)
    steps_list = cfg["steps_list"]
    ckpt_paths = {s: f"{d['ck_dir']}/step{s}.pt" for s in steps_list}

    ck_key = (f"ol_L{cfg['n_layer']}H{cfg['n_head']}D{cfg['n_embd']}_P{cfg['n_tokens']}"
              f"_T{T}_s{cfg['n_steps']}_{cfg['predict_from']}to{cfg['predict_to']}"
              f"_inj{cfg['inject_after_block']}_fm{cfg['fwd_n_layer']}x{cfg['fwd_d_head']}"
              f"x{cfg['fwd_mlp_mult']:g}_seed{seed}")
    std_dir = f"{DATA_DIR}/a2a_forward/confabulation/wake_ckpt"
    os.makedirs(std_dir, exist_ok=True)
    std_path = f"{std_dir}/{ck_key}.pt"

    if all(os.path.exists(p) for p in ckpt_paths.values()) and not cfg["refresh_wake"]:
        print(f"  wake trajectory already on volume ({len(steps_list)} ckpts) -- skip")
        return {"std_ck_path": std_path, "steps": steps_list}

    def get_batch(sd, gen):
        ix = torch.randint(len(sd) - T - 1, (cfg["batch_size"],), generator=gen)
        x = torch.stack([sd[i:i + T] for i in ix])
        y = torch.stack([sd[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    torch.manual_seed(seed)
    _init = GPT(vocab_size, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    init_state = {k: v.cpu().clone() for k, v in _init.state_dict().items()}
    del _init
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    model = GPT(vocab_size, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    model.load_state_dict(init_state)
    fm_ct = TransformerForwardModel(
        d_model=cfg["n_embd"], d_head=cfg["fwd_d_head"], n_head=cfg["fwd_n_head"],
        n_layer=cfg["fwd_n_layer"], mlp_mult=cfg["fwd_mlp_mult"], block_size=T).to(device)

    opt_main = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    opt_fwd = torch.optim.AdamW(fm_ct.parameters(), lr=cfg["fwd_lr"], weight_decay=0.01)
    train_gen = torch.Generator().manual_seed(seed + 1)
    want, t0 = set(steps_list), time.time()
    for step in range(cfg["n_steps"]):
        if step in want:
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}},
                       ckpt_paths[step])
        model.train(); fm_ct.train()
        x, y = get_batch(train_data, train_gen)
        _, lm_loss, inter = model(x, y, return_intermediates=True)
        fwd_loss = F.mse_loss(fm_ct(inter[cfg["predict_from"]].detach()),
                              inter[cfg["predict_to"]].detach())
        opt_main.zero_grad()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()
        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fm_ct.parameters(), 1.0)
        opt_fwd.step()
        if step % cfg["eval_interval"] == 0 or step == cfg["n_steps"] - 1:
            model.eval()
            g = torch.Generator().manual_seed(cfg["report_seed"] + 5)
            with torch.no_grad():
                vx, vy = get_batch(val_data, g)
                vl = float(model(vx, vy)[1])
            print(f"    step {step:6d}: lm={float(lm_loss):.4f} val={vl:.4f} "
                  f"fwd_mse={float(fwd_loss):.5f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}},
               ckpt_paths[cfg["n_steps"]])
    torch.save({"model": model.state_dict(), "fm": fm_ct.state_dict()}, std_path)
    volume.commit()
    print(f"  saved {len(steps_list)} trajectory checkpoints -> {d['ck_dir']}")
    print(f"  saved standard wake checkpoint -> {std_path}", flush=True)
    return {"std_ck_path": std_path, "steps": steps_list}


# ======================================================================
# Phase 2: per-checkpoint instrument work (one container per checkpoint)
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=16384)
def checkpoint_probe(ci: int, step: int, cfg: dict):
    """Everything that depends on ONE checkpoint: M's side of the report set, the fresh
    instrument FMs at each capacity, the residual, the junk-residual guards and the
    per-position instrument no-change floor. Writes one part file; cross-checkpoint
    quantities are formed by the aggregator.

    The M forward is shared across the instrument capacities. This is not an
    approximation: the parent battery gives every capacity the same `fm_seed`, hence the
    same data generator, hence the identical batch stream -- so one forward serves both,
    and the ensemble members within a capacity still get fully independent data."""
    import time
    import torch
    import torch.nn.functional as F
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    d = _derive(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, P, seed = d["T"], d["T"] - 1, cfg["seed"]
    n_embd = cfg["n_embd"]
    os.makedirs(d["part_dir"], exist_ok=True)
    os.makedirs(d["fm_dir"], exist_ok=True)
    part_path = f"{d['part_dir']}/ck{ci:03d}.pt"
    if os.path.exists(part_path) and not cfg["refresh_parts"]:
        print(f"  [ck{ci}] part exists -- skip")
        return {"ci": ci, "cached": True}

    t0 = time.time()
    train_data, val_data, vocab_size = _load_tokens(cfg)
    rep_tok = _report_set(cfg, val_data)
    N = rep_tok.shape[0]

    model = GPT(vocab_size, T, cfg["n_layer"], cfg["n_head"], n_embd).to(device)
    model.load_state_dict(torch.load(f"{d['ck_dir']}/step{step}.pt",
                                     map_location=device, weights_only=True)["model"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad = False
    mids = [model.transformer.h[i] for i in d["mid_blocks"]]
    obs_topk = cfg["obs_topk"]

    def get_batch(sd, gen):
        ix = torch.randint(len(sd) - T - 1, (cfg["batch_size"],), generator=gen)
        x = torch.stack([sd[i:i + T] for i in ix])
        y = torch.stack([sd[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    # ---- M's side of the report set ----
    IDS = torch.empty(N, P, obs_topk, dtype=torch.int32)
    PRB = torch.empty(N, P, obs_topk, dtype=torch.float16)
    STA = torch.empty(N, P, 4, dtype=torch.float32)
    LOSS = torch.empty(N, P, dtype=torch.float32)
    ENT = torch.empty(N, P, dtype=torch.float32)
    CORR = torch.empty(N, P, dtype=torch.bool)
    c_a0, c_aj, c_ar, mass = [], [], [], 0.0
    with torch.no_grad():
        for i in range(0, N, 64):
            xb = rep_tok[i:i + 64].to(device)
            b = xb.shape[0]
            lg, _, vi = model(xb, return_intermediates=True)
            lp = F.log_softmax(lg, dim=-1)
            pr = lp.exp()
            tp, ti = pr.topk(obs_topk, dim=-1)
            mass += float(tp.sum(-1).mean()) * b
            top2 = lg.topk(2, dim=-1).values
            ent = -(pr * lp).sum(-1)
            STA[i:i + b] = torch.stack([ent, tp[..., 0],
                                        top2[..., 0] - top2[..., 1],
                                        (1.0 - tp.sum(-1)).clamp_min(0)],
                                       dim=-1)[:, :P].cpu()
            IDS[i:i + b] = ti[:, :P].int().cpu()
            PRB[i:i + b] = tp[:, :P].half().cpu()
            ENT[i:i + b] = ent[:, :P].cpu()
            tgt = rep_tok[i:i + b, 1:].to(device)
            LOSS[i:i + b] = (-lp[:, :P].gather(2, tgt.unsqueeze(-1)).squeeze(-1)).cpu()
            CORR[i:i + b] = (lg[:, :P].argmax(-1) == tgt).cpu()
            c_a0.append(vi[cfg["predict_from"]].cpu())
            c_aj.append(vi[cfg["predict_to"]].cpu())
            c_ar.append(vi[d["report_block"]][:, :P].cpu())
            del lg, lp, pr, tp, ti, top2, vi
    a0 = torch.cat(c_a0); aj = torch.cat(c_aj); a_rep = torch.cat(c_ar)
    del c_a0, c_aj, c_ar
    torch.cuda.empty_cache()

    # report-head input: the real blocks between a_j and the report site, run over full
    # sequences so their attention is intact (the parent's `make_report_input`)
    Xp = []
    with torch.no_grad():
        for i in range(0, N, 64):
            z = aj[i:i + 64].to(device)
            for blk in mids:
                z = blk(z)
            Xp.append(z[:, :P].cpu())
    X = torch.cat(Xp).reshape(N * P, n_embd)
    recon = float((X - a_rep.reshape(N * P, n_embd)).abs().max())
    assert recon < 1e-3, f"report-input reconstruction mismatch ({recon:.2e})"
    del Xp, a_rep

    with torch.no_grad():
        g = torch.Generator().manual_seed(cfg["report_seed"] + 5)
        val = sum(float(model(*get_batch(val_data, g))[1]) for _ in range(10)) / 10

    # ---- fresh instrument FMs: ens_n independent members at each capacity ----
    def make_fm(dh, mm):
        return TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=cfg["fwd_n_head"],
                                       n_layer=cfg["fwd_n_layer"], mlp_mult=mm,
                                       block_size=T).to(device)

    resid_by_cap = {ct: [] for ct in d["cap_tags"]}
    fm0_sd = {}
    for jj in range(max(1, cfg["ens_n"])):
        fm_seed = seed + 911 + 37 * jj
        fms, opts, paths, need = {}, {}, {}, False
        for (dh, mm), ct in zip(d["inst_caps"], d["cap_tags"]):
            torch.manual_seed(fm_seed)
            fms[ct] = make_fm(dh, mm)
            paths[ct] = f"{d['fm_dir']}/{ct}_step{step}_s{jj}.pt"
            if os.path.exists(paths[ct]) and not cfg["refresh_fm"]:
                fms[ct].load_state_dict(torch.load(paths[ct], map_location=device,
                                                   weights_only=True))
            else:
                need = True
                opts[ct] = torch.optim.AdamW(fms[ct].parameters(), lr=cfg["fwd_lr"],
                                             weight_decay=0.01)
        if need:
            g = torch.Generator().manual_seed(fm_seed + 1)
            for _ in range(cfg["fresh_fm_steps"]):
                xb, yb = get_batch(train_data, g)
                with torch.no_grad():
                    _, _, vi = model(xb, yb, return_intermediates=True)
                src, tg = vi[cfg["predict_from"]], vi[cfg["predict_to"]]
                for ct, opt in opts.items():
                    fms[ct].train()
                    F.mse_loss(fms[ct](src), tg).backward()
                    torch.nn.utils.clip_grad_norm_(fms[ct].parameters(), 1.0)
                    opt.step(); opt.zero_grad()
            for ct in opts:
                torch.save({k: v.cpu() for k, v in fms[ct].state_dict().items()}, paths[ct])
        for ct in d["cap_tags"]:
            fms[ct].eval()
            with torch.no_grad():
                r = torch.cat([(aj[i:i + 64].to(device) - fms[ct](a0[i:i + 64].to(device))).cpu()
                               for i in range(0, N, 64)])
            resid_by_cap[ct].append(r)
            if jj == 0:
                fm0_sd[ct] = {k: v.cpu() for k, v in fms[ct].state_dict().items()}
        del fms, opts
        torch.cuda.empty_cache()

    caps_out = {}
    y_world = _syntactic_labels(rep_tok, vocab_size,
                                __import__("tiktoken").get_encoding("gpt2"))
    for (dh, mm), ct in zip(d["inst_caps"], d["cap_tags"]):
        resids = resid_by_cap[ct]
        r0 = resids[0]
        fwd_cos = float(F.cosine_similarity(aj - r0, aj, dim=-1).mean())
        res_norm = float(r0.norm(dim=-1).mean())
        units = [_unit16(rr, P) for rr in resids]
        fl, npair = torch.zeros(N, P), 0
        for i in range(len(units)):
            for j2 in range(i + 1, len(units)):
                fl += 1.0 - _cos_rows(units[i], units[j2])
                npair += 1
        fl = fl / max(npair, 1)
        nd = min(1000, N)
        ens = (_ensemble_cos([rr[:nd].reshape(-1, n_embd) for rr in resids])
               if len(resids) > 1 else float("nan"))
        rs = _residual_structure(r0[:nd].reshape(-1, n_embd), y_world[:nd].reshape(-1))
        fm_params = sum(v.numel() for v in fm0_sd[ct].values())
        caps_out[ct] = {
            "unit": units[0], "floor": fl, "fm0": fm0_sd[ct],
            "guards": {"step": step, "fm_params": fm_params,
                       "pct_of_predicted": 100 * fm_params / d["n_pred_params"],
                       "fwd_cosine": fwd_cos, "res_norm": res_norm, "ens_cos": ens,
                       "eta2_norm": rs["eta2_norm"], "eta2_dir": rs["eta2_dir"],
                       "eta2_vec": rs["eta2_vec"], "cohens_d": rs["cohens_d"],
                       "floor_mean": float(fl.mean())}}
        print(f"  [ck{ci:02d} step{step:6d} {ct}] cos={fwd_cos:.4f} |r|={res_norm:.3f} "
              f"ens_cos={ens:.3f} eta2_norm={rs['eta2_norm']:.4f} "
              f"floor={float(fl.mean()):.4f}", flush=True)
        del units, resids, resid_by_cap[ct]
        torch.cuda.empty_cache()

    lnf = model.transformer.ln_f
    torch.save({"ci": ci, "step": step, "val": val, "topk_mass": mass / N,
                "X": X.half(), "a0": a0[:, :P].half(), "aj": aj[:, :P].half(),
                "ids": IDS, "prb": PRB, "sta": STA,
                "loss": LOSS, "ent": ENT, "corr": CORR,
                "lnf_w": lnf.weight.detach().cpu(), "lnf_b": lnf.bias.detach().cpu(),
                "caps": caps_out}, part_path)
    volume.commit()
    print(f"  [ck{ci}] part written in {time.time() - t0:.0f}s  (val={val:.4f})", flush=True)
    return {"ci": ci, "cached": False, "val": val,
            "guards": {ct: caps_out[ct]["guards"] for ct in d["cap_tags"]}}


# ======================================================================
# Phase 3-5: the pooled battery
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=24576)
def analyze(cfg: dict):
    import resource
    import time
    import numpy as np
    import tiktoken
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from a2a_forward.model import Block
    from a2a_forward.forward_model import TransformerForwardModel

    t_start = time.time()
    d = _derive(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, P = d["T"], d["T"] - 1
    n_embd, seed = cfg["n_embd"], cfg["seed"]
    caps, cap_tags, inst_caps = d["caps"], d["cap_tags"], d["inst_caps"]
    horizons, retro_w = d["horizons"], cfg["retro_w"]
    n_qclass, obs_topk, hist_lags = cfg["n_qclass"], cfg["obs_topk"], cfg["hist_lags"]
    steps_list = cfg["steps_list"]
    n_ckpt = len(steps_list)
    hold_ckpts = cfg["hold_ckpts"]
    train_ckpts = [c for c in range(n_ckpt) if c not in set(hold_ckpts)]

    train_data, val_data, vocab_size = _load_tokens(cfg)
    rep_tok = _report_set(cfg, val_data)
    N = rep_tok.shape[0]
    y_world_full = _syntactic_labels(rep_tok, vocab_size, tiktoken.get_encoding("gpt2"))
    n_str = int(0.8 * N)
    s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)
    del train_data

    print(f"{'=' * 78}\nDISPOSITIONAL BATTERY  N={N}  checkpoints={n_ckpt}")
    print(f"  steps: {steps_list}")
    print(f"  held-out checkpoint block: {hold_ckpts}   train: {train_ckpts}")
    print(f"  horizons k={horizons}  retro w={retro_w}  q-classes={n_qclass}")
    print(f"  instruments={cap_tags}  observers={caps}  hist_lags={hist_lags}\n"
          f"{'=' * 78}", flush=True)

    # ---------------- load parts, folding in cross-checkpoint quantities -------------
    X_all = torch.empty(n_ckpt * N * P, n_embd, dtype=torch.float16)
    A0_all = torch.empty(n_ckpt, N, P, n_embd, dtype=torch.float16)
    IDS_all = torch.empty(n_ckpt, N, P, obs_topk, dtype=torch.int32)
    PRB_all = torch.empty(n_ckpt, N, P, obs_topk, dtype=torch.float16)
    STA_all = torch.empty(n_ckpt, N, P, 4, dtype=torch.float32)
    LOSS_all = torch.empty(n_ckpt, N, P, dtype=torch.float32)
    ENT_all = torch.empty(n_ckpt, N, P, dtype=torch.float32)
    CORR_all = torch.empty(n_ckpt, N, P, dtype=torch.bool)
    LNF_W = torch.empty(n_ckpt, n_embd); LNF_B = torch.empty(n_ckpt, n_embd)
    floor = {ct: torch.zeros(n_ckpt, N, P) for ct in cap_tags}
    guards = {ct: {} for ct in cap_tags}
    D_raw = {ct: {k: torch.full((n_ckpt, N, P), float("nan")) for k in horizons}
             for ct in cap_tags}
    D_fix = {ct: {k: torch.full((n_ckpt, N, P), float("nan")) for k in horizons}
             for ct in cap_tags}
    win_unit = {ct: {} for ct in cap_tags}
    win_fm = {ct: {} for ct in cap_tags}
    keep = max(horizons)
    val_traj, mass_traj = [], []
    aj_last = None

    for ci in range(n_ckpt):
        pt = torch.load(f"{d['part_dir']}/ck{ci:03d}.pt", map_location="cpu",
                        weights_only=False)
        X_all[ci * N * P:(ci + 1) * N * P] = pt["X"]
        A0_all[ci] = pt["a0"]; IDS_all[ci] = pt["ids"]; PRB_all[ci] = pt["prb"]
        STA_all[ci] = pt["sta"]; LOSS_all[ci] = pt["loss"]; ENT_all[ci] = pt["ent"]
        CORR_all[ci] = pt["corr"]
        LNF_W[ci] = pt["lnf_w"]; LNF_B[ci] = pt["lnf_b"]
        val_traj.append(pt["val"]); mass_traj.append(pt["topk_mass"])
        aj = pt["aj"]
        for (dh, mm), ct in zip(inst_caps, cap_tags):
            cp = pt["caps"][ct]
            floor[ct][ci] = cp["floor"]
            guards[ct][ci] = cp["guards"]
            win_unit[ct][ci] = cp["unit"]
            win_fm[ct][ci] = cp["fm0"]
            for k in horizons:
                src = ci - k
                if src < 0:
                    continue
                D_raw[ct][k][src] = 1.0 - _cos_rows(win_unit[ct][src], cp["unit"])
                fmf = TransformerForwardModel(
                    d_model=n_embd, d_head=dh, n_head=cfg["fwd_n_head"],
                    n_layer=cfg["fwd_n_layer"], mlp_mult=mm, block_size=T).to(device)
                fmf.load_state_dict(win_fm[ct][src]); fmf.eval()
                with torch.no_grad():
                    rf = torch.cat([(aj[i:i + 64].float().to(device)
                                     - fmf(A0_all[ci, i:i + 64].float().to(device))).cpu()
                                    for i in range(0, N, 64)])
                D_fix[ct][k][src] = 1.0 - _cos_rows(win_unit[ct][src], _unit16(rf, P))
                del fmf, rf
            for old in [c_ for c_ in list(win_unit[ct]) if c_ < ci - keep]:
                del win_unit[ct][old], win_fm[ct][old]
        if ci == n_ckpt - 1:
            aj_last = aj
        del pt, aj
        torch.cuda.empty_cache()
    print(f"  parts loaded; peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
    for ct in cap_tags:
        win_unit[ct].clear(); win_fm[ct].clear()

    # ================= build the dispositional targets =================
    def qclass(vals, valid_c):
        """Within-checkpoint quantile classes, boundaries fit on TRAIN sequences only."""
        lab = torch.full(vals.shape, -1, dtype=torch.int8)
        qs = torch.tensor([(i + 1) / n_qclass for i in range(n_qclass - 1)])
        for c in valid_c:
            b = torch.quantile(vals[c][s_tr].reshape(-1).float(), qs)
            lab[c] = torch.bucketize(vals[c].float(), b).to(torch.int8)
        return lab

    def ageclass(ev, valid_c, w):
        lab = torch.full((n_ckpt, N, P), -1, dtype=torch.int8)
        for c in valid_c:
            age = torch.full((N, P), w, dtype=torch.int8)
            for i in range(w, 0, -1):
                if c - i >= 0:
                    age = torch.where(ev[c - i], torch.tensor(i - 1, dtype=torch.int8), age)
            lab[c] = age
        return lab

    targets = {}

    def add_cls(name, vals, valid_c, family, direction, cap=None, primary=False,
                ladder=True):
        targets[name] = {"kind": "cls", "y": qclass(vals, valid_c),
                         "valid_c": list(valid_c), "n_out": n_qclass, "family": family,
                         "direction": direction, "cap": cap, "primary": primary,
                         "ladder": ladder, "cont": vals}

    fc = cfg["full_cap_targets"]
    for ct in cap_tags:
        default = ct == cap_tags[0]
        for k in horizons:
            vc = [c for c in range(n_ckpt) if c + k < n_ckpt]
            # k = first horizon carries the three instrument readings; the widest
            # horizon is the widen-k fallback if the gate says k=1 is white noise; the
            # middle horizons are computed (for the gate) but not laddered.
            lad = (default or fc) and k in (horizons[0], horizons[-1])
            add_cls(f"IMPL_PROSP_k{k}_{ct}", D_raw[ct][k], vc, "IMPL", "prosp", ct,
                    default and k == horizons[0], lad or k == horizons[0])
            if k == horizons[0]:
                add_cls(f"IMPL_PROSPEXC_k{k}_{ct}", D_raw[ct][k] - floor[ct], vc,
                        "IMPL", "prosp", ct)
                add_cls(f"IMPL_PROSPFIX_k{k}_{ct}", D_fix[ct][k], vc, "IMPL", "prosp",
                        ct, False, default or fc)
        vcs = [c for c in range(n_ckpt) if c + max(horizons) < n_ckpt]
        sm = torch.zeros(n_ckpt, N, P)
        for k in horizons:
            sm += torch.nan_to_num(D_raw[ct][k], nan=0.0)
        add_cls(f"IMPL_PROSPSM_{ct}", sm / len(horizons), vcs, "IMPL", "prosp", ct,
                False, default or fc)
        d1 = D_raw[ct][horizons[0]]
        vcr = [c for c in range(n_ckpt) if c - retro_w >= 0]
        integ = torch.zeros(n_ckpt, N, P)
        for c in vcr:
            for i in range(1, retro_w + 1):
                integ[c] += d1[c - i]
        add_cls(f"IMPL_RETRO_w{retro_w}_{ct}", integ, vcr, "IMPL", "retro", ct, default)
        ev = torch.zeros(n_ckpt, N, P, dtype=torch.bool)
        for c in range(n_ckpt - horizons[0]):
            ev[c] = d1[c] >= torch.quantile(d1[c][s_tr].reshape(-1).float(), 0.75)
        targets[f"IMPL_AGE_w{retro_w}_{ct}"] = {
            "kind": "cls", "y": ageclass(ev, vcr, retro_w), "valid_c": vcr,
            "n_out": retro_w + 1, "family": "IMPL", "direction": "retro", "cap": ct,
            "primary": False, "ladder": default or fc, "cont": None}

    for k in horizons:
        vc = [c for c in range(n_ckpt) if c + k < n_ckpt]
        dl = torch.full((n_ckpt, N, P), float("nan"))
        for c in vc:
            dl[c] = LOSS_all[c] - LOSS_all[c + k]
        add_cls(f"BEHAV_PROSP_k{k}", dl, vc, "BEHAV", "prosp", None,
                k == horizons[0], k in (horizons[0], horizons[-1]))
    vcs = [c for c in range(n_ckpt) if c + max(horizons) < n_ckpt]
    smb = torch.zeros(n_ckpt, N, P)
    for c in vcs:
        for k in horizons:
            smb[c] += (LOSS_all[c] - LOSS_all[c + k]) / len(horizons)
    add_cls("BEHAV_PROSPSM", smb, vcs, "BEHAV", "prosp")
    vcr = [c for c in range(n_ckpt) if c - retro_w >= 0]
    integb = torch.zeros(n_ckpt, N, P)
    for c in vcr:
        for i in range(1, retro_w + 1):
            integb[c] += (LOSS_all[c - i] - LOSS_all[c - i + 1]).clamp_min(0)
    add_cls(f"BEHAV_RETRO_w{retro_w}", integb, vcr, "BEHAV", "retro", None, True)
    evb = torch.zeros(n_ckpt, N, P, dtype=torch.bool)
    for c in range(n_ckpt - 1):
        dd = LOSS_all[c] - LOSS_all[c + 1]
        evb[c] = dd >= torch.quantile(dd[s_tr].reshape(-1).float(), 0.75)
    targets[f"BEHAV_AGE_w{retro_w}"] = {
        "kind": "cls", "y": ageclass(evb, vcr, retro_w), "valid_c": vcr,
        "n_out": retro_w + 1, "family": "BEHAV", "direction": "retro", "cap": None,
        "primary": False, "ladder": True, "cont": None}

    for nm in [n_ for n_, t in list(targets.items()) if t["primary"]]:
        t = targets[nm]
        targets[nm + "_RANK"] = {"kind": "rank", "y": t["cont"], "valid_c": t["valid_c"],
                                 "n_out": 1, "family": t["family"],
                                 "direction": t["direction"], "cap": t["cap"],
                                 "primary": False, "ladder": True, "cont": t["cont"]}

    # ================= the autocorrelation gate =================
    print(f"\n{'=' * 78}\nAUTOCORRELATION GATE  (train sequences only)")
    print("  probe-able does not imply learnable FROM A GIVEN TARGET (committee_head")
    print("  Phase B). r1 = lag-1 Pearson across checkpoints, rho1 its rank version,")
    print("  btw = fraction of variance that is a per-position constant.")
    print(f"  {'target':36s} {'r1':>7s} {'rho1':>7s} {'btw/tot':>8s} {'nc':>4s}")
    autocorr = {}
    for nm, t in targets.items():
        if t["cont"] is None or nm.endswith("_RANK"):
            continue
        vc = t["valid_c"]
        V = t["cont"][vc][:, s_tr].reshape(len(vc), -1)
        V = V[:, torch.isfinite(V).all(0)]
        r1s, rh1s = [], []
        for i in range(len(vc) - 1):
            if vc[i + 1] != vc[i] + 1:
                continue
            a, b = V[i] - V[i].mean(), V[i + 1] - V[i + 1].mean()
            den = float(a.norm() * b.norm())
            r1s.append(float((a * b).sum() / den) if den > 0 else float("nan"))
            sub = torch.randperm(V.shape[1])[:20000]
            rh1s.append(_spearman(V[i][sub], V[i + 1][sub]))
        tot = float(V.reshape(-1).var())
        btw = float(V.mean(0).var() / tot) if tot > 0 else 0.0
        r1 = float(np.nanmean(r1s)) if r1s else float("nan")
        rh1 = float(np.nanmean(rh1s)) if rh1s else float("nan")
        autocorr[nm] = {"r1": r1, "rho1": rh1, "between_frac": btw, "n_ckpt": len(vc)}
        print(f"  {nm:36s} {r1:+7.3f} {rh1:+7.3f} {btw:8.3f} {len(vc):4d}")
    print(f"{'=' * 78}", flush=True)

    # ================= heads and observers =================
    LNF_W_d, LNF_B_d = LNF_W.to(device), LNF_B.to(device)

    def rows_for(ci_list, seqs):
        ci_t = torch.tensor(ci_list)[:, None, None]
        return (ci_t * (N * P) + seqs[None, :, None] * P
                + torch.arange(P)[None, None, :]).reshape(-1)

    def ci_of(rows):
        return torch.div(rows, N * P, rounding_mode="floor")

    def items_of(ci_list, seqs, shuffle=True):
        ci_t = torch.tensor(ci_list)[:, None].expand(len(ci_list), len(seqs))
        it = torch.stack([ci_t.reshape(-1),
                          seqs[None, :].expand(len(ci_list), len(seqs)).reshape(-1)], -1)
        if shuffle:
            it = it[torch.randperm(len(it),
                                   generator=torch.Generator().manual_seed(seed + 77))]
        return it

    def fit_head(y_flat, kind, n_out, rows_tr, use_lnf, linear=False, rows_by_ck=None):
        """kind='cls' -> cross-entropy on the within-checkpoint quantile class.
        kind='rank' -> RankNet, with the minibatch drawn from a SINGLE checkpoint so that
        every pair compares two positions at the same snapshot."""
        torch.manual_seed(seed)
        net = (nn.Linear(n_embd, n_out).to(device) if linear
               else _make_head(n_embd, n_out, cfg["head_hidden"], device, seed))
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["head_lr"], weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        bs = min(cfg["head_bs"], len(rows_tr))
        ckk = list(rows_by_ck) if rows_by_ck else []
        for st in range(cfg["head_steps"]):
            if kind == "rank":
                pool = rows_by_ck[ckk[st % len(ckk)]]
                si = pool[torch.randint(len(pool), (min(bs, len(pool)),), generator=g)]
            else:
                si = rows_tr[torch.randint(len(rows_tr), (bs,), generator=g)]
            xb = X_all[si].to(device).float()
            if use_lnf:
                xb = _apply_lnf(xb, ci_of(si).to(device), LNF_W_d, LNF_B_d)
            out = net(xb)
            tb = y_flat[si].to(device)
            if kind == "cls":
                loss = F.cross_entropy(out, tb.long())
            else:
                s_ = out.squeeze(-1)
                i1 = torch.randperm(s_.numel(), device=device)
                dd_ = s_ - s_[i1]
                dy = tb.float() - tb.float()[i1]
                m = dy.abs() > 0
                loss = (F.softplus(-dd_[m] * dy[m].sign()).mean() if m.any()
                        else (s_ * 0).sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        net.eval()
        return net

    def score_head(net, y_flat, kind, rows_te, use_lnf, bs=16384):
        acc, n = 0.0, 0
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows_te), bs):
                si = rows_te[i:i + bs]
                xb = X_all[si].to(device).float()
                if use_lnf:
                    xb = _apply_lnf(xb, ci_of(si).to(device), LNF_W_d, LNF_B_d)
                out = net(xb)
                if kind == "cls":
                    acc += float((out.argmax(-1).cpu() == y_flat[si].long()).sum())
                    n += len(si)
                else:
                    preds.append(out.squeeze(-1).cpu())
        if kind == "cls":
            return acc / max(n, 1)
        pr = torch.cat(preds); tg = y_flat[rows_te].float(); cis = ci_of(rows_te)
        rhos = []
        for c in cis.unique():
            m = cis == c
            sub = torch.randperm(int(m.sum()))[:30000]
            rhos.append(_spearman(pr[m][sub], tg[m][sub]))
        return float(np.nanmean(rhos))

    class ObserverD(nn.Module):
        """The parent's Observer plus (a) a checkpoint-index embedding for every mode and
        (b) a `hist` mode that also sees M's own output summary at the previous
        `hist_lags` checkpoints -- the natural ceiling for the retrospective targets."""

        def __init__(self, n_out, o_layer, o_embd, mode, act_dim=0):
            super().__init__()
            self.mode = mode
            self.wte = nn.Embedding(vocab_size, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            self.cke = nn.Embedding(n_ckpt, o_embd)
            if mode in ("io", "hist"):
                self.pshape = nn.Linear(obs_topk, o_embd)
                self.sproj = nn.Linear(4, o_embd)
            if mode == "hist":
                self.lage = nn.Embedding(hist_lags + 1, o_embd)
            if mode == "act":
                self.aproj = nn.Linear(act_dim, o_embd)
            self.h = nn.ModuleList([
                Block(o_embd, min(max(o_layer, 1) * 2, 8), T) for _ in range(o_layer)])
            if not cfg["observer_causal"]:
                for blk in self.h:
                    blk.attn.bias.fill_(1.0)
            self.ln_f = nn.LayerNorm(o_embd)
            self.head = nn.Linear(o_embd, n_out)

        def io_emb(self, ids, pp, st):
            return ((self.wte(ids) * pp.unsqueeze(-1)).sum(-2) + self.pshape(pp)
                    + self.sproj(st))

        def forward(self, tok, ci, ids=None, pp=None, st=None, act=None,
                    hids=None, hpp=None, hst=None, hmask=None):
            B, t = tok.shape
            x = self.wpe(torch.arange(t, device=tok.device))[None].expand(B, t, -1)
            x = x + self.wte(tok) + self.cke(ci)[:, None, :]
            if self.mode == "io":
                x = x + self.io_emb(ids, pp, st)
            if self.mode == "act":
                x = x + self.aproj(act)
            if self.mode == "hist":
                for l in range(hist_lags + 1):
                    e = self.io_emb(hids[:, l], hpp[:, l], hst[:, l]) \
                        + self.lage(torch.full((1,), l, device=tok.device,
                                               dtype=torch.long))[None]
                    x = x + e * hmask[:, l][:, None, None]
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    def obs_batch(items, mode):
        ci, sq = items[:, 0], items[:, 1]
        kw = {"tok": rep_tok[sq, :P].to(device), "ci": ci.to(device)}
        if mode == "io":
            kw.update(ids=IDS_all[ci, sq].long().to(device),
                      pp=PRB_all[ci, sq].float().to(device),
                      st=STA_all[ci, sq].to(device))
        elif mode == "act":
            kw.update(act=A0_all[ci, sq].float().to(device))
        elif mode == "hist":
            lag = torch.arange(hist_lags + 1)[None, :]
            src = (ci[:, None] - lag).clamp_min(0)
            msk = (ci[:, None] - lag >= 0).float()
            sq2 = sq[:, None].expand_as(src)
            kw.update(hids=IDS_all[src, sq2].long().to(device),
                      hpp=PRB_all[src, sq2].float().to(device),
                      hst=STA_all[src, sq2].to(device), hmask=msk.to(device))
        return kw

    def train_observer(tgt, o_layer, o_embd, mode, items_tr, items_te_list,
                       data_frac=1.0, items_by_ck=None):
        torch.manual_seed(seed + 31)
        o = ObserverD(tgt["n_out"], o_layer, o_embd, mode, act_dim=n_embd).to(device)
        opt = torch.optim.AdamW(o.parameters(), lr=cfg["obs_lr"], weight_decay=0.01)
        g = torch.Generator().manual_seed(seed + 32)
        pool = items_tr[:max(1, int(data_frac * len(items_tr)))]
        Y, kind, obs_bs = tgt["y"], tgt["kind"], cfg["obs_bs"]
        ckk = list(items_by_ck) if items_by_ck else []
        for st in range(cfg["obs_steps"]):
            if kind == "rank":
                p_ = items_by_ck[ckk[st % len(ckk)]]
                p_ = p_[:max(1, int(data_frac * len(p_)))]
                it = p_[torch.randint(len(p_), (min(obs_bs, len(p_)),), generator=g)]
            else:
                it = pool[torch.randint(len(pool), (min(obs_bs, len(pool)),), generator=g)]
            out = o(**obs_batch(it, mode))
            tb = Y[it[:, 0], it[:, 1]].to(device)
            if kind == "cls":
                loss = F.cross_entropy(out.reshape(-1, tgt["n_out"]),
                                       tb.reshape(-1).long())
            else:
                s_ = out.squeeze(-1).reshape(-1); tf = tb.reshape(-1).float()
                i1 = torch.randperm(s_.numel(), device=device)
                dd_ = s_ - s_[i1]; dy = tf - tf[i1]
                m = dy.abs() > 0
                loss = (F.softplus(-dd_[m] * dy[m].sign()).mean() if m.any()
                        else (s_ * 0).sum())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        o.eval()
        scores = []
        with torch.no_grad():
            for items_te in items_te_list:
                if len(items_te) == 0:
                    scores.append(float("nan")); continue
                if kind == "cls":
                    acc = n = 0
                    for i in range(0, len(items_te), 32):
                        it = items_te[i:i + 32]
                        out = o(**obs_batch(it, mode)).cpu()
                        tb = Y[it[:, 0], it[:, 1]].long()
                        acc += int((out.argmax(-1) == tb).sum()); n += tb.numel()
                    scores.append(acc / max(n, 1))
                else:
                    per = {}
                    for i in range(0, len(items_te), 32):
                        it = items_te[i:i + 32]
                        out = o(**obs_batch(it, mode)).squeeze(-1).cpu()
                        tb = Y[it[:, 0], it[:, 1]].float()
                        for r in range(len(it)):
                            per.setdefault(int(it[r, 0]), [[], []])
                            per[int(it[r, 0])][0].append(out[r])
                            per[int(it[r, 0])][1].append(tb[r])
                    rh = []
                    for c, (pl, tl) in per.items():
                        pv, tv = torch.cat(pl), torch.cat(tl)
                        sub = torch.randperm(pv.numel())[:30000]
                        rh.append(_spearman(pv[sub], tv[sub]))
                    scores.append(float(np.nanmean(rh)))
        del o
        torch.cuda.empty_cache()
        return scores

    results_targets = {}

    def run_target(nm, tgt, light=False):
        vc = set(tgt["valid_c"])
        tr_c = [c for c in train_ckpts if c in vc]
        te_c = [c for c in hold_ckpts if c in vc]
        if not tr_c:
            return None
        y_flat = tgt["y"].reshape(-1)
        rows_tr = rows_for(tr_c, s_tr)
        rows_te_seq = rows_for(tr_c, s_te)
        rows_te_ck = rows_for(te_c, s_te) if te_c else torch.zeros(0, dtype=torch.long)
        it_tr, it_te_seq = items_of(tr_c, s_tr), items_of(tr_c, s_te)
        it_te_ck = items_of(te_c, s_te) if te_c else torch.zeros(0, 2, dtype=torch.long)
        rk = tgt["kind"] == "rank"
        rows_by_ck = {c: rows_for([c], s_tr) for c in tr_c} if rk else None
        it_by_ck = {c: items_of([c], s_tr) for c in tr_c} if rk else None

        row = {"family": tgt["family"], "direction": tgt["direction"], "cap": tgt["cap"],
               "kind": tgt["kind"], "n_out": tgt["n_out"], "train_ckpts": tr_c,
               "held_ckpts": te_c}
        if tgt["kind"] == "cls":
            row["chance_heldck"] = (_balance(tgt["y"][te_c][:, s_te].reshape(-1).long(),
                                             tgt["n_out"]) if te_c else float("nan"))
            row["chance_seq"] = _balance(tgt["y"][tr_c][:, s_te].reshape(-1).long(),
                                         tgt["n_out"])
        else:
            row["chance_heldck"] = row["chance_seq"] = 0.0

        for variant, use_lnf in ([("lnf", True)] if light
                                 else [("lnf", True), ("raw", False)]):
            h = fit_head(y_flat, tgt["kind"], tgt["n_out"], rows_tr, use_lnf,
                         rows_by_ck=rows_by_ck)
            row[f"self_{variant}_seq"] = score_head(h, y_flat, tgt["kind"],
                                                    rows_te_seq, use_lnf)
            row[f"self_{variant}_heldck"] = (
                score_head(h, y_flat, tgt["kind"], rows_te_ck, use_lnf)
                if len(rows_te_ck) else float("nan"))
            del h
        if not light:
            hl = fit_head(y_flat, tgt["kind"], tgt["n_out"], rows_tr, True, linear=True,
                          rows_by_ck=rows_by_ck)
            row["self_linear_seq"] = score_head(hl, y_flat, tgt["kind"], rows_te_seq, True)
            row["self_linear_heldck"] = (
                score_head(hl, y_flat, tgt["kind"], rows_te_ck, True)
                if len(rows_te_ck) else float("nan"))
            del hl
        torch.cuda.empty_cache()

        obs, ote = {}, [it_te_seq, it_te_ck]
        for (ol_, oe_) in ([caps[-1]] if light else caps):
            capn = f"{ol_}L{oe_}D"
            if not light:
                obs[f"O_input@{capn}"] = train_observer(tgt, ol_, oe_, "input", it_tr,
                                                        ote, items_by_ck=it_by_ck)
            obs[f"O_io@{capn}"] = train_observer(tgt, ol_, oe_, "io", it_tr, ote,
                                                 items_by_ck=it_by_ck)
        ol_, oe_ = caps[-1]
        obs[f"O_act@{ol_}L{oe_}D"] = train_observer(tgt, ol_, oe_, "act", it_tr, ote,
                                                    items_by_ck=it_by_ck)
        obs[f"O_hist@{ol_}L{oe_}D"] = train_observer(tgt, ol_, oe_, "hist", it_tr, ote,
                                                     items_by_ck=it_by_ck)
        if not light:
            obs[f"O_io@{ol_}L{oe_}D_half_data"] = train_observer(
                tgt, ol_, oe_, "io", it_tr, ote, data_frac=0.5, items_by_ck=it_by_ck)
        row["observers"] = obs
        iok = [k for k in obs if k.startswith("O_io") and "half" not in k]
        row["best_O_io_seq"] = max(obs[k][0] for k in iok)
        row["best_O_io_heldck"] = max(obs[k][1] for k in iok)
        row["adv_seq"] = row["self_lnf_seq"] - row["best_O_io_seq"]
        row["adv_heldck"] = row["self_lnf_heldck"] - row["best_O_io_heldck"]
        print(f"  [{nm:36s}] self={row['self_lnf_seq']:.3f}/{row['self_lnf_heldck']:.3f}"
              f" bestOio={row['best_O_io_seq']:.3f}/{row['best_O_io_heldck']:.3f}"
              f" ADV={row['adv_seq']:+.3f}/{row['adv_heldck']:+.3f}"
              f" O_act={obs[f'O_act@{ol_}L{oe_}D'][1]:.3f}"
              f" O_hist={obs[f'O_hist@{ol_}L{oe_}D'][1]:.3f}"
              f" ch={row['chance_heldck']:.3f}", flush=True)
        results_targets[nm] = row
        return row

    print(f"\n{'=' * 78}\nDISPOSITIONAL BATTERY  "
          f"(scores are held-out-seq @ train-ckpt / held-out-seq @ held-out-ckpt)\n"
          f"{'=' * 78}", flush=True)
    order = ([n_ for n_, t in targets.items() if t["primary"]]
             + [n_ for n_ in targets if n_.endswith("_RANK")]
             + [n_ for n_, t in targets.items()
                if not t["primary"] and not n_.endswith("_RANK")])
    for nm in order:
        if not targets[nm].get("ladder", True):
            continue
        run_target(nm, targets[nm], light=not targets[nm]["primary"])

    # ================= paper 2's occurrent battery at the final checkpoint ===========
    occurrent = {}
    if not cfg["skip_occurrent"]:
        print(f"\n{'=' * 78}\nOCCURRENT CONTROL BATTERY @ final checkpoint "
              f"(step {steps_list[-1]})\n{'=' * 78}", flush=True)
        cf = n_ckpt - 1
        rows_tr1, rows_te1 = rows_for([cf], s_tr), rows_for([cf], s_te)
        it_tr1, it_te1 = items_of([cf], s_tr), items_of([cf], s_te)
        ct0, (dh0, mm0) = cap_tags[0], inst_caps[0]
        fmF = TransformerForwardModel(d_model=n_embd, d_head=dh0, n_head=cfg["fwd_n_head"],
                                      n_layer=cfg["fwd_n_layer"], mlp_mult=mm0,
                                      block_size=T).to(device)
        fmF.load_state_dict(torch.load(
            f"{d['fm_dir']}/{ct0}_step{steps_list[-1]}_s0.pt",
            map_location=device, weights_only=True))
        fmF.eval()
        with torch.no_grad():
            Rf = torch.cat([(aj_last[i:i + 64].float().to(device)
                             - fmF(A0_all[cf, i:i + 64].float().to(device))).cpu()
                            for i in range(0, N, 64)]).reshape(N * P, n_embd)
        cents = _kmeans_fit(Rf[:n_str * P].to(device), cfg["impl_k"], seed=seed)
        y_impl = _kmeans_assign(Rf.to(device), cents).cpu()
        cqw, cqb, cqs = _cluster_quality(Rf.to(device), y_impl.to(device), cents)
        y_impl_cos = F.normalize(Rf, dim=-1)

        def as_frame(v, dtype):
            z = torch.zeros(n_ckpt, N, P, dtype=dtype)
            z[cf] = v.reshape(N, P).to(dtype)
            return z

        y_ent = torch.bucketize(ENT_all[cf].reshape(-1), torch.quantile(
            ENT_all[cf][s_tr].reshape(-1), torch.tensor([.25, .5, .75])))
        occ_targets = {
            "IMPL": {"y": as_frame(y_impl, torch.int8), "n_out": cfg["impl_k"]},
            "BEHAV": {"y": as_frame(CORR_all[cf].long(), torch.int8), "n_out": 2},
            "ENT": {"y": as_frame(y_ent, torch.int8), "n_out": 4},
            "WORLD": {"y": as_frame(y_world_full[:, :P], torch.int8), "n_out": 5},
        }
        for tn, spec in occ_targets.items():
            spec.update(kind="cls", valid_c=[cf], family="OCC", direction="occ",
                        cap=ct0, primary=False, ladder=True, cont=None)
            full = tn == "IMPL"
            y_flat = spec["y"].reshape(-1)
            h = fit_head(y_flat, "cls", spec["n_out"], rows_tr1, False)
            self_s = score_head(h, y_flat, "cls", rows_te1, False)
            obs = {}
            for (ol_, oe_) in (caps if full else [caps[-1]]):
                capn = f"{ol_}L{oe_}D"
                if full:
                    obs[f"O_input@{capn}"] = train_observer(spec, ol_, oe_, "input",
                                                            it_tr1, [it_te1])
                obs[f"O_io@{capn}"] = train_observer(spec, ol_, oe_, "io", it_tr1, [it_te1])
            ol_, oe_ = caps[-1]
            obs[f"O_act@{ol_}L{oe_}D"] = train_observer(spec, ol_, oe_, "act",
                                                        it_tr1, [it_te1])
            best_io = max(v[0] for k, v in obs.items() if k.startswith("O_io"))
            ch = _balance(spec["y"][cf][s_te].reshape(-1).long(), spec["n_out"])
            occurrent[tn] = {"self": self_s, "best_O_io": best_io,
                             "advantage": self_s - best_io, "chance": ch,
                             "observers": {k: v[0] for k, v in obs.items()}}
            print(f"  [{tn:9s}] self={self_s:.3f} bestO_io={best_io:.3f} "
                  f"adv={self_s - best_io:+.3f} chance={ch:.3f}", flush=True)
            del h
            torch.cuda.empty_cache()

        torch.manual_seed(seed)
        net = _make_head(n_embd, n_embd, cfg["head_hidden"], device, seed)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["head_lr"], weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        for _ in range(cfg["head_steps"]):
            si = rows_tr1[torch.randint(len(rows_tr1),
                                        (min(cfg["head_bs"], len(rows_tr1)),),
                                        generator=g)]
            out = net(X_all[si].to(device).float())
            (1.0 - F.cosine_similarity(out, y_impl_cos[si - cf * N * P].to(device),
                                       dim=-1)).mean().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        net.eval()
        with torch.no_grad():
            acc = 0.0
            for i in range(0, len(rows_te1), 16384):
                si = rows_te1[i:i + 16384]
                acc += float(F.cosine_similarity(
                    net(X_all[si].to(device).float()),
                    y_impl_cos[si - cf * N * P].to(device), dim=-1).sum())
            self_cos = acc / len(rows_te1)

        def train_obs_cos(o_layer, o_embd, mode):
            torch.manual_seed(seed + 31)
            o = ObserverD(n_embd, o_layer, o_embd, mode, act_dim=n_embd).to(device)
            opt2 = torch.optim.AdamW(o.parameters(), lr=cfg["obs_lr"], weight_decay=0.01)
            g2 = torch.Generator().manual_seed(seed + 32)
            Yc = y_impl_cos.reshape(N, P, n_embd)
            for _ in range(cfg["obs_steps"]):
                it = it_tr1[torch.randint(len(it_tr1), (cfg["obs_bs"],), generator=g2)]
                (1.0 - F.cosine_similarity(o(**obs_batch(it, mode)),
                                           Yc[it[:, 1]].to(device), dim=-1)).mean().backward()
                torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
                opt2.step(); opt2.zero_grad()
            o.eval()
            tot = n_ = 0.0
            with torch.no_grad():
                for i in range(0, len(it_te1), 32):
                    it = it_te1[i:i + 32]
                    out = o(**obs_batch(it, mode)).cpu()
                    tb = Yc[it[:, 1]]
                    tot += float(F.cosine_similarity(out, tb, dim=-1).sum())
                    n_ += tb.shape[0] * tb.shape[1]
            del o
            torch.cuda.empty_cache()
            return tot / n_

        cos_obs = {f"O_io@{caps[-1][0]}L{caps[-1][1]}D": train_obs_cos(*caps[-1], "io"),
                   f"O_act@{caps[-1][0]}L{caps[-1][1]}D": train_obs_cos(*caps[-1], "act")}
        best_io_cos = max(v for k, v in cos_obs.items() if k.startswith("O_io"))
        occurrent["IMPL_COS"] = {"self": self_cos, "best_O_io": best_io_cos,
                                 "advantage": self_cos - best_io_cos, "chance": 0.0,
                                 "observers": cos_obs,
                                 "cluster_quality": {"within": cqw, "between": cqb,
                                                     "separation": cqs}}
        print(f"  [IMPL_COS ] self={self_cos:.3f} bestO_io={best_io_cos:.3f} "
              f"adv={self_cos - best_io_cos:+.3f} (cluster sep={cqs:.3f})", flush=True)
        del net, Rf, fmF
        torch.cuda.empty_cache()

    # ================= dump + summary =================
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out = {"config": {**{k: v for k, v in cfg.items()},
                      "peak_rss_gb": peak_gb, "wall_s": time.time() - t_start},
           "val_loss_traj": val_traj, "topk_mass_traj": mass_traj,
           "guards": guards, "autocorr": autocorr,
           "targets": results_targets, "occurrent": occurrent}
    fn = f"{d['out_dir']}/results.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\n{'=' * 78}\nSUMMARY")
    print("  advantage = self-report - best capacity-matched O_io, on held-out sequences")
    print("  at HELD-OUT CHECKPOINTS (strict) and at train checkpoints.")
    print("  READ EVERY IMPL ROW AGAINST ITS ens_cos/eta2 COLUMN.")
    print(f"\n  guard table (default instrument {cap_tags[0]}):")
    print(f"    {'ck':>3s} {'step':>6s} {'val':>6s} {'cos':>7s} {'|r|':>7s} "
          f"{'ens_cos':>8s} {'eta2_n':>7s} {'floor':>7s}")
    for ci in range(n_ckpt):
        gd = guards[cap_tags[0]][ci]
        print(f"    {ci:3d} {gd['step']:6d} {val_traj[ci]:6.3f} {gd['fwd_cosine']:7.4f} "
              f"{gd['res_norm']:7.3f} {gd['ens_cos']:8.3f} {gd['eta2_norm']:7.4f} "
              f"{gd['floor_mean']:7.4f}")
    ol_, oe_ = caps[-1]
    print(f"\n  {'target':36s} {'fam':>5s} {'dir':>6s} {'self':>7s} {'bestOio':>8s} "
          f"{'ADV':>8s} {'O_act':>7s} {'O_hist':>7s} {'chance':>7s}")
    for nm, r in results_targets.items():
        print(f"  {nm:36s} {r['family']:>5s} {r['direction']:>6s} "
              f"{r['self_lnf_heldck']:7.3f} {r['best_O_io_heldck']:8.3f} "
              f"{r['adv_heldck']:+8.3f} {r['observers'][f'O_act@{ol_}L{oe_}D'][1]:7.3f} "
              f"{r['observers'][f'O_hist@{ol_}L{oe_}D'][1]:7.3f} {r['chance_heldck']:7.3f}")
    if occurrent:
        print("\n  occurrent battery @ final checkpoint:")
        for tn, r in occurrent.items():
            print(f"    {tn:9s} self={r['self']:.3f} bestO_io={r['best_O_io']:.3f} "
                  f"adv={r['advantage']:+.3f} chance={r['chance']:.3f}")
    print(f"\n  peak RSS {peak_gb:.2f} GB   wall {(time.time() - t_start) / 60:.1f} min")
    print(f"  saved -> {fn}\n{'=' * 78}", flush=True)
    return out


# ======================================================================
# CPU coordinator
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=4096)
def dispositional_test(
    n_tokens: int = 10_000_000, block_size: int = 128,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block2",
    report_block: str = "", inject_after_block: int = 1,
    fwd_n_layer: int = 2, fwd_d_head: int = 64, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
    n_steps: int = 10_000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    n_ckpt: int = 16, ckpt_first: int = 300,
    n_report_sequences: int = 2000, report_seed: int = 999,
    fresh_fm_steps: int = 3000,
    inst_caps_str: str = "16:0.5,4:0.25", ens_n: int = 3, impl_k: int = 8,
    head_hidden: int = 256, head_steps: int = 3000, head_lr: float = 1e-3,
    head_bs: int = 4096,
    horizons_str: str = "1,2,4", retro_w: int = 4, n_qclass: int = 4,
    hold_ckpts_str: str = "", full_cap_targets: bool = False,
    observer_caps: str = "1:64,2:128,4:256",
    obs_steps: int = 3000, obs_lr: float = 3e-4, obs_bs: int = 32,
    obs_topk: int = 64, observer_causal: bool = False, hist_lags: int = 4,
    skip_occurrent: bool = False,
    eval_interval: int = 500, seed: int = 42, tag: str = "",
    max_dop: int = 8,
    refresh_wake: bool = False, refresh_fm: bool = False, refresh_parts: bool = False,
    smoke: bool = False,
):
    """CPU coordinator: wake -> per-checkpoint fan-out -> pooled battery."""
    import time
    if smoke:
        n_steps, n_report_sequences, fresh_fm_steps = 300, 128, 100
        head_steps, obs_steps, n_tokens = 150, 80, 2_000_000
        observer_caps, eval_interval = "1:64,2:128", 100
        inst_caps_str, ens_n = "16:0.5,4:0.25", 2
        n_ckpt, ckpt_first = 8, 15
        horizons_str, retro_w, head_bs = "1,2", 2, 2048
        max_dop = 8

    steps_list = ckpt_schedule(n_steps, n_ckpt, ckpt_first)
    n_ckpt = len(steps_list)
    if hold_ckpts_str:
        hold = sorted(int(z) for z in hold_ckpts_str.split(","))
    else:
        mid = n_ckpt // 2
        hold = [mid, mid + 1, mid + 2][:max(1, n_ckpt // 5)]

    cfg = dict(
        n_tokens=n_tokens, block_size=block_size, n_layer=n_layer, n_head=n_head,
        n_embd=n_embd, predict_from=predict_from, predict_to=predict_to,
        report_block=report_block, inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult, n_steps=n_steps, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, fwd_lr=fwd_lr, steps_list=steps_list,
        hold_ckpts=hold, n_report_sequences=n_report_sequences,
        report_seed=report_seed, fresh_fm_steps=fresh_fm_steps,
        inst_caps_str=inst_caps_str, ens_n=ens_n, impl_k=impl_k,
        head_hidden=head_hidden, head_steps=head_steps, head_lr=head_lr,
        head_bs=head_bs, horizons_str=horizons_str, retro_w=retro_w,
        n_qclass=n_qclass, full_cap_targets=full_cap_targets,
        observer_caps=observer_caps, obs_steps=obs_steps, obs_lr=obs_lr,
        obs_bs=obs_bs, obs_topk=obs_topk, observer_causal=observer_causal,
        hist_lags=hist_lags, skip_occurrent=skip_occurrent,
        eval_interval=eval_interval, seed=seed, tag=tag,
        refresh_wake=refresh_wake, refresh_fm=refresh_fm, refresh_parts=refresh_parts,
        smoke=smoke)

    t0 = time.time()
    print(f"{'=' * 78}\nDISPOSITIONAL SELF-KNOWLEDGE  tag={tag or 'run'}"
          f"{'  [SMOKE]' if smoke else ''}")
    print(f"  checkpoints ({n_ckpt}): {steps_list}")
    print(f"  held-out block: {hold}   max_dop={max_dop}\n{'=' * 78}", flush=True)

    w = wake_trajectory.remote(cfg)
    print(f"  wake done ({(time.time() - t0) / 60:.1f} min): {w['std_ck_path']}",
          flush=True)

    pending = list(enumerate(steps_list))
    for i in range(0, len(pending), max_dop):
        wave = pending[i:i + max_dop]
        handles = [checkpoint_probe.spawn(ci, st, cfg) for ci, st in wave]
        for h in handles:
            h.get()
        volume.reload()
        print(f"  wave {i // max_dop}: checkpoints {[c for c, _ in wave]} done "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = analyze.remote(cfg)
    print(f"  ALL DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
    return res
