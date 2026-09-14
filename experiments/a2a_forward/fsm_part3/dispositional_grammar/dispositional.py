"""Dispositional self-report: does the first-person advantage extend from "what am I
computing now" to "how is my computation changing"?

Parent:    ../../../rhm/confabulation/README.md  -- the occurrent confabulation battery, whose wake recipe,
           instrument-capacity guards and observer ladder this file forks. Helpers are
           IMPORTED from `rhm_confabulation`, not copied, so the two stay in lockstep.
Substrate: ../../../rhm/RHM_LATENT_LOOP_README.md  (`ntp_aux_cl`, val 2.3829)

THE QUESTION
  Paper 2's target is OCCURRENT: the direction class of M's residual at one position at
  one training snapshot. The residual is an implementation fact, and reports about it
  beat every capacity-matched observer (+0.09..+0.11 here) while reports about behaviour
  do not. This file asks whether the same privilege extends to DISPOSITIONAL facts --
  how M's own computation is changing, has changed, and will change -- which requires
  training the report head ACROSS a training trajectory rather than at one snapshot.

  2 x 2:  {implementation, behaviour} x {prospective, retrospective}, all read WITHIN
  hierarchy level, because in RHM "age of acquisition" is nearly "which tree level" and
  level is an input fact any observer reads off the token index (committee_head Phase C:
  the realised-drop label is 83% between-level).

WHAT IS MEASURED
  M (`ntp_aux_cl`) is trained once with a log-spaced checkpoint schedule. At every
  checkpoint c: fresh instrument FMs (ens_n seeds, two capacities), the residual
  r_c(p) = a6_c(p) - FM_c(a0_c(p)), the guards (`ens_cos`, hierarchy-eta2), M's
  per-position loss and entropy, the late-stream report state a_c(p) = post_block7.

  Targets at horizon k (in checkpoint indices; the schedule is geometric in steps, so
  a backward window spans the same step RATIO as a forward one):

    IMPL_PRO     8-way class of r_{c+k}(p)          "what will my computation here be"
    IMPL_RETRO   8-way class of r_{c-k}(p)          "what was my computation here"
    IMPL_NOW     8-way class of r_c(p)              the occurrent anchor, pooled over c
    REORG_PRO    within-level quartile of 1-cos(r^(c)_c, r^(c)_{c+k})   [fixed instrument]
    REORG_RETRO  within-level quartile of 1-cos(r^(c)_{c-k}, r^(c)_c)
    BEHAV_PRO    within-level quartile of  l_c(p) - l_{c+k}(p)   "will my answer improve"
    BEHAV_RETRO  within-level quartile of  l_{c-k}(p) - l_c(p)   "has my answer improved"

  PRO and RETRO are the SAME statistic read forward and backward from c, so any
  asymmetry between the two directions is about the direction of time and not about the
  target's functional form.

  Every head (self, confabulator, observer) is trained pooled over checkpoints and
  evaluated on (a) held-out sequences at trained checkpoints and (b) held-out sequences
  at a held-out contiguous checkpoint window -- so a positive number is a rule about
  states, not a memorised per-checkpoint constant.

  Observer ladder: O_input / O_io (tokens + M_c's logits) capacity-swept, half-data
  control, O_act (M_c's own post_block0 -- activation-access ceiling), O_hist (tokens +
  logits at c AND at c-k -- the natural ceiling for the retrospective rows: it can
  compute BEHAV_RETRO exactly), O_acthist (a0 at c-k and c, implementation rows).

Run:
  modal run a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run --smoke
  modal run --detach a2a_forward/fsm_part3/dispositional_grammar/dispositional.py::dispositional_run \
      --tag main
"""

import json
import math
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-dispositional", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def ckpt_schedule(n_steps, n_ckpt, first):
    """Geometric in steps: a backward window of k checkpoints spans the same step ratio
    as a forward one, so PRO and RETRO are matched in training-time units."""
    if n_ckpt <= 1:
        return [n_steps]
    r = (n_steps / first) ** (1.0 / (n_ckpt - 1))
    steps = sorted({int(round(first * r ** i)) for i in range(n_ckpt)})
    steps[-1] = n_steps
    return steps


def level_of_pos(T, s, L):
    """Report position p (0..T-2) predicts token p+1; its hierarchy level is the s-adic
    valuation of p+1, capped at L-1 (PER_LEVEL_LOSS_README)."""
    lv = []
    for p in range(T - 1):
        q, e = p + 1, 0
        while q % s == 0 and e < L - 1:
            q //= s
            e += 1
        lv.append(e)
    return lv


def part_dir(key, tag):
    return f"{DATA_DIR}/rhm_confabulation/{key}/dispositional/{tag}"


# ======================================================================
# Stage A -- the wake, with a checkpoint schedule
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=86400, memory=12288)
def wake_trajectory(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    cond: str = "ntp_aux_cl",
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, fwd_lr: float = 1e-3,
    lam_aux: float = 1.0, lam_local: float = 1.0, ug_hidden: int = 64,
    pool_size: int = 200000, data_seed: int = 7, seed: int = 42,
    n_ckpt: int = 16, first_ckpt: int = 400, eval_interval: int = 1000,
    tag: str = "main",
):
    """`rhm_confabulation.confabulation_test`'s Phase-1 wake, VERBATIM (same DGP, model,
    FM, gate, init, losses, RNG streams), with state_dict snapshots at a geometric
    checkpoint schedule. Only M's weights are saved: the battery runs on standalone
    (non-injected) forward passes with fresh instrument FMs, exactly as the parent does.
    """
    import numpy as np
    import resource
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.confabulation.rhm_confabulation import _generate_with_traces
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    cib = int(predict_from.replace("post_block", ""))
    use_aux, use_loop = "aux" in cond, "cl" in cond
    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    cdir = f"{part_dir(key, tag)}/ckpt"
    os.makedirs(cdir, exist_ok=True)
    cpath = lambda st: f"{cdir}/{cond}_step{st}_seed{seed}.pt"

    print(f"{'=' * 74}\nWAKE TRAJECTORY  {key}  {cond}  {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  schedule ({len(sched)}): {sched}")
    if all(os.path.exists(cpath(st)) for st in sched):
        print("  all checkpoints present -- skipping training")
        return {"schedule": sched, "resumed": True}

    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    pool_seqs, pool_lf, _ = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))
    pool_anc = [torch.from_numpy(pool_lf[e].astype(np.int64)) for e in range(L)]
    anc_idx = [torch.arange(T) // (s ** (L - e)) for e in range(L)]
    spanend = [torch.tensor([p for p in range(T) if (p + 1) % (s ** (L - e)) == 0])
               for e in range(L)]
    sup_blocks = [f"post_block{i}" for i in range(n_layer)]
    corpus = pool_x.reshape(-1)
    n_corpus, arangeT = corpus.shape[0], torch.arange(T)
    del pool_seqs, pool_lf

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    def get_aux_batch(gen):
        idx = torch.randint(0, pool_size, (batch_size,), generator=gen)
        x = pool_x[idx].to(device)
        labels = [pool_anc[e][idx][:, anc_idx[e]].to(device) for e in range(L)]
        return x, labels

    def aux_loss(inter, labels, aux_heads):
        head_out = {b: aux_heads[b](inter[b]).view(batch_size, T, L, v) for b in sup_blocks}
        per_level = []
        for e in range(L):
            se = spanend[e]
            ces = [F.cross_entropy(head_out[b][:, se, e, :].reshape(-1, v),
                                   labels[e][:, se].reshape(-1)) for b in sup_blocks]
            per_level.append(torch.stack(ces).mean())
        return torch.stack(per_level).mean()

    class UnifiedGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(nn.Linear(2 * d_model, d_hidden), nn.GELU(),
                                          nn.Linear(d_hidden, d_model))
            self.projection = nn.Linear(d_model, d_model)
            for layer in (self.gate_net[-1], self.projection):
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

        def forward(self, activations, fwd_pred):
            gw = torch.sigmoid(self.gate_net(torch.cat([activations, fwd_pred], dim=-1)))
            return gw * self.projection(fwd_pred), gw

    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)

    def eval_ntp(mdl):
        mdl.eval()
        gen = torch.Generator().manual_seed(999 + 5)     # report_seed + 5, as the parent
        tot = 0.0
        with torch.no_grad():
            for _ in range(10):
                x, y = get_ntp_batch(gen)
                tot += mdl(x, y)[1].item()
        return tot / 10

    main_params = list(model.parameters())
    aux_heads = None
    if use_aux:
        aux_heads = nn.ModuleDict({b: nn.Linear(n_embd, L * v) for b in sup_blocks}).to(device)
        main_params += list(aux_heads.parameters())
    ugate = fm_ct = opt_fwd = None
    if use_loop:
        ugate = UnifiedGate(n_embd, ug_hidden).to(device)
        main_params += list(ugate.parameters())
        fm_ct = TransformerForwardModel(d_model=n_embd, d_head=fwd_d_head,
                                        n_head=fwd_n_head, n_layer=fwd_n_layer,
                                        mlp_mult=fwd_mlp_mult, block_size=T).to(device)
        opt_fwd = torch.optim.AdamW(fm_ct.parameters(), lr=fwd_lr, weight_decay=0.01)
    opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)

    train_gen = torch.Generator().manual_seed(seed + 1)
    aux_gen = torch.Generator().manual_seed(seed + 2)
    sched_set, vals = set(sched), {}
    for step in range(n_steps):
        if step in sched_set:
            torch.save(model.state_dict(), cpath(step))
            vals[step] = eval_ntp(model)
            volume.commit()
            print(f"    [ckpt] step {step:6d}  val={vals[step]:.4f}")
        model.train()
        if use_loop:
            fm_ct.train(); ugate.train()
        x, y = get_ntp_batch(train_gen)
        cache = {}
        if use_loop:
            def ug_cb(act, _c=cache):
                fp = fm_ct(act.detach())
                _c["pred"] = fp
                inj, gw = ugate(act.detach(), fp.detach())
                _c["gw"] = gw
                return inj
            logits, _, inter = model(
                x, return_intermediates=True, cerebellar_fn=ug_cb,
                cerebellar_input_block=cib, cerebellar_inject_block=inject_after_block)
        else:
            logits, _, inter = model(x, return_intermediates=True)
        loss = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1))
        ntp = loss
        if use_aux:
            xa, labels = get_aux_batch(aux_gen)
            _, _, inter_a = model(xa, return_intermediates=True)
            loss = loss + lam_aux * aux_loss(inter_a, labels, aux_heads)
        if use_loop:
            fwd_pred, tgt_acts = cache["pred"], inter[predict_to]
            r = tgt_acts - fwd_pred.detach()
            loss = loss + lam_local * (cache["gw"].detach().mean() * r ** 2).mean()
        opt_main.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, 1.0)
        opt_main.step()
        if use_loop:
            opt_fwd.zero_grad()
            F.mse_loss(fwd_pred, tgt_acts.detach()).backward()
            torch.nn.utils.clip_grad_norm_(fm_ct.parameters(), 1.0)
            opt_fwd.step()
        if step % eval_interval == 0:
            print(f"    step {step:6d}: ntp={ntp.item():.4f} val={eval_ntp(model):.4f}")

    torch.save(model.state_dict(), cpath(n_steps))
    vals[n_steps] = eval_ntp(model)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  FINAL val={vals[n_steps]:.4f}  (harness-validation reference: 2.3829)")
    print(f"  peak RSS {rss:.2f} GB")
    out = {"schedule": sched, "val_by_step": vals, "resumed": False,
           "final_val": vals[n_steps], "peak_rss_gb": rss}
    with open(f"{part_dir(key, tag)}/wake.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out


# ======================================================================
# Stage B -- per-checkpoint instrument + cached state
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=12288)
def instrument_at_checkpoint(
    step: int,
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    fwd_n_layer: int = 1, fwd_n_head: int = 8, fwd_lr: float = 1e-3,
    cond: str = "ntp_aux_cl", seed: int = 42,
    pool_size: int = 200000, data_seed: int = 7, batch_size: int = 64,
    n_report_sequences: int = 3000, report_seed: int = 999,
    fresh_fm_steps: int = 3000, ens_n: int = 3, diag_seqs: int = 2000,
    inst_caps_str: str = "16:1.0,4:0.25", tag: str = "main",
):
    """Fresh instrument FMs on the frozen M_step, the two junk-residual guards, and the
    cached report-set state every later stage reads. Parent recipe verbatim
    (`train_fresh_fm`): same seeds (seed+911+37j), lr, wd, clip, step count."""
    import numpy as np
    import resource
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _compute_hierarchy_eta2
    from rhm.confabulation.rhm_confabulation import (
        _generate_with_traces, _eta2_by_level, _ensemble_cos)
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    pdir = part_dir(key, tag)
    os.makedirs(f"{pdir}/parts", exist_ok=True)
    part = f"{pdir}/parts/step{step}.npz"
    meta_fn = f"{pdir}/parts/step{step}_meta.json"
    if os.path.exists(part) and os.path.exists(meta_fn):
        print(f"  step {step}: part exists -- skipping")
        with open(meta_fn) as f:
            return json.load(f)

    caps = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in inst_caps_str.split(",")]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    # FM training distribution: flat corpus windows, exactly as the parent's wake
    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus, arangeT = corpus.shape[0], torch.arange(T)
    del pool_seqs

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        f"{pdir}/ckpt/{cond}_step{step}_seed{seed}.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rep_seqs, rep_lf, rep_lr = _generate_with_traces(rules, n_report_sequences, report_seed)
    rep_x = torch.from_numpy(rep_seqs.astype(np.int64)).to(device)
    N = rep_x.shape[0]

    def make_fm(dh, mm):
        return TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=fwd_n_head,
                                       n_layer=fwd_n_layer, mlp_mult=mm,
                                       block_size=T).to(device)

    def train_fresh_fm(fm_seed, dh, mm):
        torch.manual_seed(fm_seed)
        fm = make_fm(dh, mm)
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        gen = torch.Generator().manual_seed(fm_seed + 1)
        for _ in range(fresh_fm_steps):
            fm.train()
            xb, _ = get_ntp_batch(gen)
            with torch.no_grad():
                _, _, vi = model(xb, return_intermediates=True)
            F.mse_loss(fm(vi[predict_from]), vi[predict_to]).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        fm.eval()
        for p in fm.parameters():
            p.requires_grad = False
        return fm

    # ---- cached M state on the fixed report set ----
    rb = f"post_block{n_layer - 1}"
    c_a0, c_a6, c_x7, c_lg = [], [], [], []
    with torch.no_grad():
        for i in range(0, N, 128):
            lg, _, vi = model(rep_x[i:i + 128], return_intermediates=True)
            c_a0.append(vi[predict_from].half().cpu()); c_a6.append(vi[predict_to].half().cpu())
            c_x7.append(vi[rb].half().cpu()); c_lg.append(lg.half().cpu())
    a0 = torch.cat(c_a0); a6 = torch.cat(c_a6); x7 = torch.cat(c_x7); lg = torch.cat(c_lg)
    del c_a0, c_a6, c_x7, c_lg
    lp = F.log_softmax(lg[:, :T - 1].float(), -1)
    tgt = torch.from_numpy(rep_seqs.astype(np.int64))[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)            # (N, T-1)
    ent = -(lp.exp() * lp).sum(-1)                                  # (N, T-1)
    correct = (lg[:, :T - 1].float().argmax(-1) == tgt)

    last_block = model.transformer.h[n_layer - 1]

    def carry(z):                       # a6-space tensor -> report site (post_block{n-1})
        out = []
        with torch.no_grad():
            for i in range(0, z.shape[0], 128):
                out.append(last_block(z[i:i + 128].to(device).float()).half().cpu())
        return torch.cat(out)

    def fm_apply(fm, src, lim=None):
        lim = src.shape[0] if lim is None else min(lim, src.shape[0])
        out = []
        with torch.no_grad():
            for i in range(0, lim, 128):
                out.append(fm(src[i:min(i + 128, lim)].to(device).float()).half().cpu())
        return torch.cat(out)

    meta = {"step": step, "N": int(N), "caps": {}}
    save = {"a0": a0.numpy(), "a6": a6.numpy(), "x7": x7.numpy(),
            "logit": lg.numpy(), "nll": nll.numpy().astype(np.float32),
            "ent": ent.numpy().astype(np.float32),
            "correct": correct.numpy().astype(np.int8),
            "tok": rep_seqs.astype(np.int16)}
    fmdir = f"{pdir}/fm"
    os.makedirs(fmdir, exist_ok=True)
    nd = min(diag_seqs, N)
    for ci, (dh, mm) in enumerate(caps):
        ctag = f"h{dh}m{mm:g}"
        fms = [train_fresh_fm(seed + 911 + 37 * j, dh, mm) for j in range(max(1, ens_n))]
        for j, fm in enumerate(fms):
            torch.save(fm.state_dict(), f"{fmdir}/{ctag}_step{step}_s{j}.pt")
        pred = fm_apply(fms[0], a0)
        resid = (a6.float() - pred.float())
        fwd_cos = float(F.cosine_similarity(pred.float(), a6.float(), dim=-1).mean())
        res_norm = float(resid.norm(dim=-1).mean())
        ens = (_ensemble_cos([(a6[:nd].float() - fm_apply(f_, a0, nd).float()).reshape(-1, n_embd)
                              for f_ in fms]) if len(fms) > 1 else float("nan"))
        eta2p = _eta2_by_level(resid[:nd], [lf[:nd] for lf in rep_lf], s, L, T)
        ref = _compute_hierarchy_eta2(resid[:nd].numpy(), [lf[:nd] for lf in rep_lf],
                                      [lr_[:nd] for lr_ in rep_lr], s=s, L=L)
        meta["caps"][ctag] = {
            "fm_params": int(sum(p.numel() for p in fms[0].parameters())),
            "fwd_cosine": fwd_cos, "res_norm": res_norm, "ens_cos": ens,
            "eta2_pooled": eta2p,
            "eta2_reference_last_token": {f"d{L - e}": ref[f"level_{e}"]["feature_eta2"]
                                          for e in range(L)},
            "rule_eta2_reference": {f"d{L - e}": ref[f"level_{e}"]["rule_eta2"]
                                    for e in range(L)},
        }
        print(f"  step {step:6d} {ctag:8s}: cos={fwd_cos:.4f} |r|={res_norm:.3f} "
              f"ens_cos={ens:.3f}  d6η²(ref)={ref['level_5']['feature_eta2']:.3f} "
              f"d5η²(ref)={ref['level_4']['feature_eta2']:.3f}")
        if ci == 0:                         # confabulator state, default instrument only
            save["x7_confab"] = carry(pred).numpy()
        del fms, pred, resid
        torch.cuda.empty_cache()

    np.savez(part, **save)
    with open(meta_fn, "w") as f:
        json.dump(meta, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    meta["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  step {step}: saved {part}  peak RSS {meta['peak_rss_gb']:.2f} GB")
    return meta



# ======================================================================
# Stage C -- targets, the level/checkpoint confounds, and the autocorrelation gate
# ======================================================================

def _var_share(x, groups):
    """Between-group share of the target's variance (the committee_head Phase C
    diagnostic: how much of this label is just 'which level' / 'which checkpoint')."""
    import numpy as np
    gm = np.zeros(x.shape, dtype=np.float64)
    for g in np.unique(groups):
        sel = groups == g
        gm[sel] = x[sel].mean()
    tot = float(np.var(x.astype(np.float64)))
    return float(np.var(gm) / tot) if tot > 0 else 0.0


def _cat_lift(y, groups, K):
    """How much better than the pooled majority a group-conditional constant does,
    as a fraction of the available headroom. 0 = the group tells you nothing."""
    import numpy as np
    pooled = np.bincount(y, minlength=K).max() / y.size
    acc = 0.0
    for g in np.unique(groups):
        sel = groups == g
        acc += sel.sum() * (np.bincount(y[sel], minlength=K).max() / sel.sum())
    acc /= y.size
    return float((acc - pooled) / max(1e-9, 1.0 - pooled)), float(acc)


def _gate_stats(X, lv, k):
    """X: (n_c, N, P) continuous target at horizon k. Autocorrelation across checkpoints
    of the per-(sequence, position) value -- raw, and after removing the per-(checkpoint,
    level) mean. The second is the honest reading: the level means carry the trajectory's
    shape and level is free to any observer from the token index. If it is ~0 the target
    is white noise at this resolution and no head can learn it (committee_head Phase B,
    where `lprog`'s round-to-round autocorrelation was +0.007).

    TWO lags are reported because the targets are DIFFERENCES over a k-checkpoint window:
      lag 1        adjacent windows. For k=1 they share an endpoint with OPPOSITE signs,
                   which manufactures a negative autocorrelation whatever the signal is.
                   Reported, never used to choose k.
      lag k+1      fully disjoint windows -- no shared term, so no induced correlation.
                   This is the statistic the horizon rule reads."""
    import numpy as np
    n_c = X.shape[0]
    Z = X.astype(np.float64).copy()
    for c in range(n_c):
        for l in np.unique(lv):
            Z[c][:, lv == l] -= Z[c][:, lv == l].mean()

    def ac(A, lag):
        vals = []
        for c in range(n_c - lag):
            u, w = A[c].reshape(-1), A[c + lag].reshape(-1)
            if u.std() > 1e-9 and w.std() > 1e-9:
                vals.append(float(np.corrcoef(u, w)[0, 1]))
        return float(np.mean(vals)) if vals else float("nan")

    Xd = X.astype(np.float64)
    pos_map = np.broadcast_to(Xd.mean(axis=(0, 1))[None, None, :], Xd.shape)
    return {"autocorr_lag1_raw": ac(Xd, 1), "autocorr_lag1_within": ac(Z, 1),
            # windows [c,c+k] and [c+lag,c+lag+k] share an index only when lag == k,
            # so lag 1 is contaminated at k == 1 and clean at every k >= 2.
            "lag1_clean": bool(k != 1),
            "autocorr_disjoint_raw": ac(Xd, k + 1),
            "autocorr_disjoint_within": ac(Z, k + 1),
            "n_disjoint_pairs": max(0, n_c - (k + 1)),
            "mean": float(Xd.mean()), "std": float(Xd.std()),
            "between_position_var_share":
                float(np.var(pos_map) / max(1e-12, np.var(Xd)))}


def _bin_within(x, lv_rows, ci_rows, n_bins, train_mask):
    """Quantile bins computed WITHIN (level, checkpoint) on train-sequence rows only.
    Balancing inside each (level, checkpoint) cell removes both confounds by
    construction: the level confound (age of acquisition ~ tree level, and level is free
    to any observer from the token index) and the checkpoint confound (everything
    improves early). Pooled-quantile variants are kept for the record."""
    import numpy as np
    y = np.zeros(x.shape, dtype=np.int8)
    for c in np.unique(ci_rows):
        cm = ci_rows == c
        for l in np.unique(lv_rows):
            sel = cm & (lv_rows == l)
            tr = sel & train_mask
            if tr.sum() < n_bins * 4:
                continue
            qs = np.quantile(x[tr], np.linspace(0, 1, n_bins + 1)[1:-1])
            y[sel] = np.searchsorted(qs, x[sel], side="right")
    return y


def _bin_pooled(x, n_bins, train_mask):
    import numpy as np
    qs = np.quantile(x[train_mask], np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(qs, x, side="right").astype(np.int8)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=20480)
def build_targets(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
    n_embd: int = 256, fwd_n_layer: int = 1, fwd_n_head: int = 8,
    n_steps: int = 20000, n_ckpt: int = 16, first_ckpt: int = 400, seed: int = 42,
    inst_caps_str: str = "16:1.0,4:0.25", horizon: int = 2, k_gate_max: int = 4,
    impl_k: int = 8, n_bins: int = 4, seq_train_frac: float = 0.8,
    held_window: int = 3, tag: str = "main", auto_k: bool = True,
    gate_threshold: float = 0.10, out_suffix: str = "",
):
    import numpy as np
    import resource
    import torch
    import torch.nn.functional as F
    from rhm.confabulation.rhm_confabulation import _kmeans_fit, _kmeans_assign
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    P = T - 1
    key = tb_key(v, s, L, m)
    pdir = part_dir(key, tag)
    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    n_ck = len(sched)
    lv = np.array(level_of_pos(T, s, L))
    caps = [(int(c.split(":")[0]), float(c.split(":")[1])) for c in inst_caps_str.split(",")]
    kmax = max(1, min(k_gate_max, (n_ck - 1) // 2))

    print(f"{'=' * 74}\nBUILD TARGETS  {key}  tag={tag}\n  schedule {sched}")
    parts = [np.load(f"{pdir}/parts/step{st}.npz") for st in sched]
    N = parts[0]["a0"].shape[0]
    n_str = int(seq_train_frac * N)
    tok = parts[0]["tok"]
    a0 = np.stack([p["a0"] for p in parts])
    a6 = np.stack([p["a6"] for p in parts])
    nll = np.stack([p["nll"] for p in parts]).astype(np.float32)
    ent = np.stack([p["ent"] for p in parts]).astype(np.float32)
    for p in parts:
        p.close()
    print(f"  N={N}  a0/a6 {a0.nbytes / 1e9:.2f}+{a6.nbytes / 1e9:.2f} GB in RAM")

    def load_fm(dh, mm, st, sj):
        fm = TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=fwd_n_head,
                                     n_layer=fwd_n_layer, mlp_mult=mm, block_size=T).to(device)
        fm.load_state_dict(torch.load(f"{pdir}/fm/h{dh}m{mm:g}_step{st}_s{sj}.pt",
                                      map_location=device))
        fm.eval()
        for p_ in fm.parameters():
            p_.requires_grad = False
        return fm

    def resid(fm, ci):
        out = []
        with torch.no_grad():
            for i in range(0, N, 256):
                src = torch.from_numpy(a0[ci, i:i + 256]).to(device).float()
                tgt = torch.from_numpy(a6[ci, i:i + 256]).to(device).float()
                out.append((tgt - fm(src)).half().cpu())
        return torch.cat(out)

    results = {"schedule": sched, "N": int(N), "n_seq_train": n_str,
               "levels": lv.tolist(), "caps": {}}
    # Which levels the model actually learns, and when. Root-adjacent levels the model
    # never learns have no dynamics to report on; they are noted, never averaged in.
    results["level_trajectory"] = {
        str(sched[c]): {
            "mean_nll_by_level": {int(l): float(nll[c][:, lv == l].mean())
                                  for l in range(L)},
            "mean_ent_by_level": {int(l): float(ent[c][:, lv == l].mean())
                                  for l in range(L)}}
        for c in range(n_ck)}
    out_npz = {"levels": lv, "tok": tok, "schedule": np.array(sched)}
    chosen_k = horizon

    for cap_i, (dh, mm) in enumerate(caps):
        ctag = f"h{dh}m{mm:g}"
        print(f"\n  --- instrument {ctag} ---")
        fms0 = [load_fm(dh, mm, st, 0) for st in sched]
        fms1 = [load_fm(dh, mm, st, 1) for st in sched]
        R = [resid(fms0[c], c) for c in range(n_ck)]          # fresh instrument per ckpt
        rmean = [r.float().mean(dim=(0, 1), keepdim=True) for r in R]

        def centered_rows(c, r=None):
            """Per-checkpoint-centered, flattened (N*P, d) residual rows on GPU. Centering
            removes the per-checkpoint offset (a global drift of the representation) so
            the codebook is about residual DIRECTION, not about which checkpoint."""
            rr = R[c] if r is None else r
            return (rr[:, :P].to(device).float()
                    - (rmean[c] if r is None
                       else r.float().mean(dim=(0, 1), keepdim=True)).to(device)
                    ).reshape(-1, n_embd)

        def reorg_fixed(c, d, fmlist):
            """1 - cos(r^(c)_c, r^(c)_{c+d}): the instrument is IDENTICAL on both sides,
            so the no-change floor is exactly 0 and none of the change is the
            instrument's. The fresh-instrument reading is kept alongside, where the
            floor is 1 - ens_cos."""
            fm = fmlist[c]
            cos = np.empty((N, P), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, N, 256):
                    base = (torch.from_numpy(a6[c, i:i + 256]).to(device).float()
                            - fm(torch.from_numpy(a0[c, i:i + 256]).to(device).float()))
                    oth = (torch.from_numpy(a6[c + d, i:i + 256]).to(device).float()
                           - fm(torch.from_numpy(a0[c + d, i:i + 256]).to(device).float()))
                    cos[i:i + 256] = F.cosine_similarity(
                        base[:, :P], oth[:, :P], dim=-1).cpu().numpy()
            return 1.0 - cos

        def reorg_fresh(c, d):
            with torch.no_grad():
                return (1.0 - F.cosine_similarity(
                    R[c][:, :P].float(), R[c + d][:, :P].float(), dim=-1).numpy())

        # ---------- the autocorrelation gate (before any head is trained) ----------
        GATE_NMS = ("BEHAV_PRO", "BEHAV_RETRO", "REORG_PRO", "REORG_RETRO")
        gate, gmin = {}, {}
        for k in range(1, kmax + 1):
            gc = list(range(k, n_ck - k))          # every checkpoint the horizon supports
            tg = {"BEHAV_PRO": np.stack([nll[c] - nll[c + k] for c in gc]),
                  "BEHAV_RETRO": np.stack([nll[c - k] - nll[c] for c in gc]),
                  "REORG_PRO": np.stack([reorg_fixed(c, +k, fms0) for c in gc]),
                  "REORG_RETRO": np.stack([reorg_fixed(c, -k, fms0) for c in gc])}
            gate[f"k{k}"] = {nm: _gate_stats(X, lv, k) for nm, X in tg.items()}
            gate[f"k{k}"]["n_checkpoints"] = len(gc)
            _v = [gate[f"k{k}"][nm]["autocorr_disjoint_within"] for nm in GATE_NMS]
            gmin[k] = min((-1e9 if x != x else x) for x in _v)
            print(f"      k={k} ({len(gc)} ckpts): " + "  ".join(
                f"{nm} disj={gate[f'k{k}'][nm]['autocorr_disjoint_within']:+.3f} "
                f"lag1={gate[f'k{k}'][nm]['autocorr_lag1_within']:+.3f}"
                for nm in GATE_NMS))
        if cap_i == 0 and auto_k:
            ok = [k for k in range(1, kmax + 1) if gmin[k] >= gate_threshold]
            chosen_k = ok[0] if ok else max(gmin, key=lambda z: gmin[z])
            print(f"      GATE -> k={chosen_k} (rule: smallest k whose four disjoint-window "
                  f"within-level autocorrelations are all >= {gate_threshold}; else the k "
                  f"maximising that minimum. min-by-k: "
                  + " ".join(f"k{z}={gmin[z]:+.3f}" for z in sorted(gmin)) + ")")
        k = chosen_k
        tc_all = list(range(k, n_ck - k))
        n_tc = len(tc_all)
        hstart = max(0, (n_tc - held_window) // 2)
        held_t = list(range(hstart, min(n_tc, hstart + held_window)))
        train_t = [i for i in range(n_tc) if i not in held_t]

        # ---------- codebook: pooled over TRAIN checkpoints x TRAIN sequences ----------
        rng = np.random.default_rng(0)
        fit_rows = []
        for ti in train_t:
            c = tc_all[ti]
            sel = torch.from_numpy(rng.choice(n_str, size=min(n_str, 400), replace=False))
            rr = R[c][sel]
            fit_rows.append((rr[:, :P].to(device).float()
                             - rmean[c].to(device)).reshape(-1, n_embd))
        cents = _kmeans_fit(torch.cat(fit_rows), impl_k, seed=seed)
        del fit_rows
        torch.cuda.empty_cache()
        cls = np.stack([_kmeans_assign(centered_rows(c), cents).cpu().numpy()
                        .reshape(N, P).astype(np.int8) for c in range(n_ck)])
        pers = {f"k{kk}": {
            "persistence": float(np.mean([(cls[c] == cls[c + kk]).mean() for c in gc])),
            "chance": float(np.mean([
                float((np.bincount(cls[c].reshape(-1), minlength=impl_k) / (N * P)
                       * np.bincount(cls[c + kk].reshape(-1), minlength=impl_k)
                       / (N * P)).sum()) for c in gc]))}
            for kk in range(1, kmax + 1)}
        print("      IMPL class persistence: " + "  ".join(
            f"k={kk}: {pers[f'k{kk}']['persistence']:.3f} "
            f"(chance {pers[f'k{kk}']['chance']:.3f})" for kk in range(1, kmax + 1)))

        agree = []
        for c in tc_all:
            r1 = resid(fms1[c], c)
            c1 = _kmeans_assign(centered_rows(c, r1), cents).cpu().numpy().reshape(N, P)
            agree.append(float((c1 == cls[c]).mean()))
            del r1
        cls_rel = float(np.mean(agree))
        print(f"      IMPL class instrument agreement (FM seed0 vs seed1, one codebook): "
              f"{cls_rel:.3f}")

        # ---------- final targets at the chosen horizon ----------
        ci_rows = np.repeat(np.arange(n_tc), N * P)
        lv_rows = np.tile(lv, n_tc * N)
        pos_rows = lv_rows * 0 + np.tile(np.arange(P), n_tc * N)
        seq_rows = np.tile(np.repeat(np.arange(N), P), n_tc)
        train_mask = seq_rows < n_str
        cont = {"BEHAV_PRO": np.stack([nll[c] - nll[c + k] for c in tc_all]),
                "BEHAV_RETRO": np.stack([nll[c - k] - nll[c] for c in tc_all]),
                "REORG_PRO": np.stack([reorg_fixed(c, +k, fms0) for c in tc_all]),
                "REORG_RETRO": np.stack([reorg_fixed(c, -k, fms0) for c in tc_all]),
                "REORG_PRO_fresh": np.stack([reorg_fresh(c, +k) for c in tc_all]),
                "REORG_RETRO_fresh": np.stack([reorg_fresh(c, -k) for c in tc_all])}
        rel = {}
        for nm, ds in (("REORG_PRO", +1), ("REORG_RETRO", -1)):
            x1 = np.stack([reorg_fixed(c, ds * k, fms1) for c in tc_all])
            rel[nm] = float(np.corrcoef(cont[nm].reshape(-1), x1.reshape(-1))[0, 1])
        print("      REORG instrument reliability (seed0 vs seed1 target): "
              + "  ".join(f"{nm}={x:+.3f}" for nm, x in rel.items()))

        for nm, X in cont.items():
            out_npz[f"{ctag}__{nm}_cont"] = X.astype(np.float32)
            out_npz[f"{ctag}__{nm}_bin"] = _bin_within(
                X.reshape(-1), lv_rows, ci_rows, n_bins, train_mask).reshape(n_tc, N, P)
            out_npz[f"{ctag}__{nm}_binpooled"] = _bin_pooled(
                X.reshape(-1), n_bins, train_mask).reshape(n_tc, N, P)
        out_npz[f"{ctag}__IMPL_PRO_cls"] = np.stack([cls[c + k] for c in tc_all])
        out_npz[f"{ctag}__IMPL_RETRO_cls"] = np.stack([cls[c - k] for c in tc_all])
        out_npz[f"{ctag}__IMPL_NOW_cls"] = np.stack([cls[c] for c in tc_all])
        out_npz[f"{ctag}__ENT_bin"] = _bin_within(
            np.stack([ent[c] for c in tc_all]).reshape(-1), lv_rows, ci_rows,
            n_bins, train_mask).reshape(n_tc, N, P)

        conf = {}
        for nm in ("IMPL_PRO_cls", "IMPL_RETRO_cls", "IMPL_NOW_cls"):
            y = out_npz[f"{ctag}__{nm}"].reshape(-1).astype(np.int64)
            lift_l, acc_l = _cat_lift(y, lv_rows, impl_k)
            lift_c, acc_c = _cat_lift(y, ci_rows, impl_k)
            lift_p, acc_p = _cat_lift(y, pos_rows, impl_k)
            conf[nm] = {"base_majority": float(np.bincount(y, minlength=impl_k).max() / y.size),
                        "base_level": acc_l, "base_checkpoint": acc_c, "base_position": acc_p,
                        "lift_level": lift_l, "lift_checkpoint": lift_c,
                        "lift_position": lift_p}
        for nm, X in cont.items():
            x = X.reshape(-1)
            yb = out_npz[f"{ctag}__{nm}_bin"].reshape(-1).astype(np.int64)
            yp = out_npz[f"{ctag}__{nm}_binpooled"].reshape(-1).astype(np.int64)
            conf[nm] = {
                "between_level_var_share": _var_share(x, lv_rows),
                "between_position_var_share": _var_share(x, pos_rows),
                "between_checkpoint_var_share": _var_share(x, ci_rows),
                "bin_base_majority": float(np.bincount(yb, minlength=n_bins).max() / yb.size),
                "bin_base_position": _cat_lift(yb, pos_rows, n_bins)[1],
                "binpooled_base_level": _cat_lift(yp, lv_rows, n_bins)[1],
                "binpooled_base_majority":
                    float(np.bincount(yp, minlength=n_bins).max() / yp.size)}
        results["caps"][ctag] = {"gate": gate, "class_persistence": pers,
                                 "cls_instrument_agreement": cls_rel,
                                 "reorg_instrument_reliability": rel,
                                 "target_confounds": conf,
                                 "target_trajectory": {
                                     nm: {str(sched[tc_all[t]]):
                                          {int(l): float(X[t][:, lv == l].mean())
                                           for l in range(L)}
                                          for t in range(n_tc)}
                                     for nm, X in cont.items()}}
        del R, rmean, fms0, fms1, cont
        torch.cuda.empty_cache()

    results["horizon"] = chosen_k
    results["tc_indices"] = tc_all
    results["tc_steps"] = [sched[c] for c in tc_all]
    results["held_target_indices"] = held_t
    results["held_steps"] = [sched[tc_all[t]] for t in held_t]
    results["train_target_indices"] = train_t
    out_npz["tc_indices"] = np.array(tc_all)
    out_npz["held_t"] = np.array(held_t)
    out_npz["train_t"] = np.array(train_t)
    np.savez(f"{pdir}/targets{out_suffix}.npz", **out_npz)
    results["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(f"{pdir}/targets_meta{out_suffix}.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n  horizon k={chosen_k}  target ckpts {results['tc_steps']}")
    print(f"  held-out checkpoint window {results['held_steps']}")
    print(f"  saved -> {pdir}/targets{out_suffix}.npz   peak RSS {results['peak_rss_gb']:.2f} GB")
    return results


# ======================================================================
# Stage D -- the report head, the confabulator, and the observer ladder
# ======================================================================

OBS_PLAN_FULL = [
    ("O_input", "1:64", [], True, 1.0), ("O_input", "8:256", [], True, 1.0),
    ("O_io", "1:64", ["logit_c"], True, 1.0), ("O_io", "2:128", ["logit_c"], True, 1.0),
    ("O_io", "4:192", ["logit_c"], True, 1.0), ("O_io", "8:256", ["logit_c"], True, 1.0),
    ("O_io_half", "8:256", ["logit_c"], True, 0.5),
    ("O_hist", "8:256", ["logit_c", "logit_prev"], True, 1.0),
    ("O_act", "8:256", ["a0_c"], False, 1.0),
]
OBS_PLAN_IMPL_EXTRA = [("O_acthist", "8:256", ["a0_c", "a0_prev"], False, 1.0)]
CONT_OBS = ["O_io@8:256", "O_hist@8:256", "O_act@8:256"]


def _spearman(a, b):
    import numpy as np
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=20480)
def battery_cell(
    cell: str, target: str, cap: str = "h16m1", readout: str = "cat",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
    n_layer: int = 8, n_embd: int = 256, n_ckpt: int = 16, first_ckpt: int = 400,
    n_steps: int = 20000, seed: int = 42, impl_k: int = 8, n_bins: int = 4,
    head_hidden: int = 256, head_steps: int = 5000, head_lr: float = 1e-3,
    obs_steps: int = 5000, obs_lr: float = 3e-4, obs_bs: int = 64,
    seq_train_frac: float = 0.8, tag: str = "main", full_ladder: bool = True,
    with_cont: bool = True, with_confab: bool = True, out_suffix: str = "",
    cont_only: bool = False,
):
    """One (target, instrument) cell of the dispositional battery.

    Every head -- self, confabulator, observer -- trains on the SAME rows (train
    sequences x train checkpoints) and is scored on two disjoint held-out sets:
      held_seq   held-out sequences at trained checkpoints
      held_both  held-out sequences at a held-out contiguous CHECKPOINT window
    so a positive number is a rule about states, not a per-checkpoint constant.

    All continuous inputs (the report state x7, M's logits, M's a0) are centered per
    checkpoint on train rows. Centering is information-preserving within a checkpoint and
    removes the cross-checkpoint offset symmetrically from self and observers, so nobody
    can win by recognising which snapshot they are looking at."""
    import numpy as np
    import resource
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import Block

    device = "cuda"
    L, T = depth, s ** depth
    P = T - 1
    key = tb_key(v, s, L, m)
    pdir = part_dir(key, tag)
    os.makedirs(f"{pdir}/cells", exist_ok=True)
    outfn = f"{pdir}/cells/{cell}{out_suffix}.json"
    if os.path.exists(outfn):
        print(f"  cell {cell}: exists -- skipping")
        with open(outfn) as f:
            return json.load(f)

    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    tz = np.load(f"{pdir}/targets{out_suffix}.npz")
    tc_all = tz["tc_indices"].tolist()
    held_t = tz["held_t"].tolist()
    train_t = tz["train_t"].tolist()
    lv = tz["levels"]
    tok_np = tz["tok"].astype(np.int64)
    n_tc = len(tc_all)
    with open(f"{pdir}/targets_meta{out_suffix}.json") as f:
        tmeta = json.load(f)
    k = tmeta["horizon"]
    N = tmeta["N"]
    n_str = int(seq_train_frac * N)

    if cont_only:
        readout, with_confab = "both", False
    y_key = f"{cap}__{target}"
    y = tz[y_key].astype(np.int64).reshape(-1)
    n_cls = impl_k if target.endswith("_cls") else n_bins
    ycont = None
    if with_cont and readout == "both" and f"{cap}__{target.replace('_bin', '')}_cont" in tz:
        ycont = tz[f"{cap}__{target.replace('_bin', '')}_cont"].astype(np.float32)

    # ---- cached state for the target checkpoints and their k-lagged partners ----
    need = sorted({tc_all[t] for t in range(n_tc)} | {tc_all[t] - k for t in range(n_tc)})
    cache = {}
    for c in need:
        cache[c] = np.load(f"{pdir}/parts/step{sched[c]}.npz")

    def gather(field, offset, dims=None):
        out = np.empty((n_tc, N, P, dims), dtype=np.float16)
        for t in range(n_tc):
            z = cache[tc_all[t] + offset][field][:, :P]
            out[t] = z if dims is None or z.shape[-1] == dims else z[..., :dims]
        return out

    d_lg = cache[need[0]]["logit"].shape[-1]
    streams = {"logit_c": gather("logit", 0, d_lg),
               "logit_prev": gather("logit", -k, d_lg),
               "a0_c": gather("a0", 0, n_embd)}
    x7 = gather("x7", 0, n_embd)
    x7cf = gather("x7_confab", 0, n_embd) if with_confab else None
    impl_family = target.startswith("IMPL") or target.startswith("REORG")
    if impl_family and full_ladder and not cont_only:
        streams["a0_prev"] = gather("a0", -k, n_embd)
    for c in need:
        cache[c].close()
    del cache

    # ---- per-checkpoint centering on train rows ----
    def center(A):
        for t in range(n_tc):
            mu = A[t, :n_str].astype(np.float32).reshape(-1, A.shape[-1]).mean(0)
            A[t] -= mu.astype(np.float16)
    for A in list(streams.values()):
        center(A)
    center(x7)
    if x7cf is not None:
        center(x7cf)

    rows = lambda ts, lo, hi: np.concatenate(
        [np.arange(t * N * P + lo * P, t * N * P + hi * P) for t in ts])
    tr_rows = rows(train_t, 0, n_str)
    evA = rows(train_t, n_str, N)
    evB = rows(held_t, n_str, N) if held_t else evA[:0]
    lv_all = np.tile(lv, n_tc * N)
    pos_all = np.tile(np.arange(P), n_tc * N)
    print(f"{'=' * 74}\nCELL {cell}  target={y_key}  n_cls={n_cls}  k={k}")
    print(f"  rows train={len(tr_rows)}  held_seq={len(evA)}  held_both={len(evB)}"
          f"  ckpts train={[sched[tc_all[t]] for t in train_t]} held={[sched[tc_all[t]] for t in held_t]}")

    def by_level(corr, idx):
        lvi = lv_all[idx]
        return {int(l): float(corr[lvi == l].mean()) for l in np.unique(lvi)}

    def scored(corr_A, corr_B):
        return {"held_seq": float(corr_A.mean()), "held_both": float(corr_B.mean()) if len(corr_B) else float("nan"),
                "held_seq_by_level": by_level(corr_A, evA),
                "held_both_by_level": by_level(corr_B, evB) if len(corr_B) else {}}

    res = {"cell": cell, "target": y_key, "n_cls": n_cls, "horizon": k,
           "cont_only": bool(cont_only)}

    # ---- baselines ----
    maj = np.bincount(y[tr_rows], minlength=n_cls).argmax()
    posmaj = np.zeros(P, dtype=np.int64)
    for p in range(P):
        sel = tr_rows[pos_all[tr_rows] == p]
        posmaj[p] = np.bincount(y[sel], minlength=n_cls).argmax()
    base = {"majority": scored((y[evA] == maj).astype(np.float64),
                               (y[evB] == maj).astype(np.float64)),
            "position_majority": scored((y[evA] == posmaj[pos_all[evA]]).astype(np.float64),
                                        (y[evB] == posmaj[pos_all[evB]]).astype(np.float64))}
    res["baselines"] = base
    print(f"  baselines: majority={base['majority']['held_seq']:.3f} "
          f"position={base['position_majority']['held_seq']:.3f}")

    # ---- the self report head (and the confabulator) ----
    yt = torch.from_numpy(y)

    def fit_mlp(Xnp, n_out, mode="cat", tgt=None, tag_=""):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xnp.shape[-1], head_hidden), nn.GELU(),
                            nn.Linear(head_hidden, n_out)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=head_lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        Xf = torch.from_numpy(Xnp.reshape(-1, Xnp.shape[-1]))
        trt = torch.from_numpy(tr_rows)
        for _ in range(head_steps):
            si = trt[torch.randint(len(trt), (4096,), generator=g)]
            xb = Xf[si].to(device).float()
            if mode == "cat":
                loss = F.cross_entropy(net(xb), yt[si].to(device))
            else:
                loss = F.mse_loss(net(xb).squeeze(-1), tgt[si].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        net.eval()

        def ev(idx):
            outs = []
            with torch.no_grad():
                for i in range(0, len(idx), 32768):
                    si = torch.from_numpy(idx[i:i + 32768])
                    o = net(Xf[si].to(device).float())
                    outs.append((o.argmax(-1).cpu() if mode == "cat"
                                 else o.squeeze(-1).cpu()))
            return torch.cat(outs).numpy()
        return ev

    if not cont_only:
        ev_self = fit_mlp(x7, n_cls)
        res["self"] = scored((ev_self(evA) == y[evA]).astype(np.float64),
                             (ev_self(evB) == y[evB]).astype(np.float64))
        print(f"  self: held_seq={res['self']['held_seq']:.3f} "
              f"held_both={res['self']['held_both']:.3f}")
    if x7cf is not None and not cont_only:
        ev_cf = fit_mlp(x7cf, n_cls)
        res["confabulator"] = scored((ev_cf(evA) == y[evA]).astype(np.float64),
                                     (ev_cf(evB) == y[evB]).astype(np.float64))
        print(f"  confabulator: held_seq={res['confabulator']['held_seq']:.3f}")

    # ---- observers ----
    class Observer(nn.Module):
        def __init__(self, n_out, o_layer, o_embd, extra_dim, tokens):
            super().__init__()
            self.tokens = tokens
            if tokens:
                self.wte = nn.Embedding(v, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            self.proj = nn.Linear(extra_dim, o_embd) if extra_dim else None
            self.h = nn.ModuleList([Block(o_embd, min(o_layer * 2, 8), T, causal=True)
                                    for _ in range(o_layer)])
            self.ln_f = nn.LayerNorm(o_embd)
            self.head = nn.Linear(o_embd, n_out)

        def forward(self, tokb, extra=None):
            B, t = (tokb.shape if self.tokens else extra.shape[:2])
            x = self.wpe(torch.arange(t, device=self.wpe.weight.device))[None].expand(B, t, -1)
            if self.tokens:
                x = x + self.wte(tokb)
            if self.proj is not None:
                x = x + self.proj(extra)
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    tok_t = torch.from_numpy(tok_np[:, :P])
    ycube = torch.from_numpy(y.reshape(n_tc, N, P))

    def fit_observer(o_layer, o_embd, extra_names, tokens, frac, n_out,
                     mode="cat", tgtcube=None):
        torch.manual_seed(seed + 31)
        ed = sum(streams[e].shape[-1] for e in extra_names)
        o = Observer(n_out, o_layer, o_embd, ed, tokens).to(device)
        opt = torch.optim.AdamW(o.parameters(), lr=obs_lr, weight_decay=0.01)
        g = torch.Generator().manual_seed(seed + 32)
        pool_n = max(1, int(frac * n_str))

        def batch(t, si_np):
            xb = tok_t[torch.from_numpy(si_np)].to(device) if tokens else None
            eb = (torch.cat([torch.from_numpy(streams[e][t][si_np]) for e in extra_names], -1)
                  .to(device).float() if extra_names else None)
            return xb, eb
        for st in range(obs_steps):
            t = train_t[int(torch.randint(len(train_t), (1,), generator=g))]
            si = torch.randint(pool_n, (obs_bs,), generator=g)
            xb, eb = batch(t, si.numpy())
            out = o(xb, eb)
            if mode == "cat":
                loss = F.cross_entropy(out.reshape(-1, n_out),
                                       ycube[t][si].reshape(-1).to(device))
            else:
                loss = F.mse_loss(out.squeeze(-1), tgtcube[t][si].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        o.eval()

        def ev(ts):
            outs = []
            with torch.no_grad():
                for t in ts:
                    for i in range(n_str, N, 128):
                        si = np.arange(i, min(i + 128, N))
                        xb, eb = batch(t, si)
                        out = o(xb, eb)
                        outs.append((out.argmax(-1).cpu() if mode == "cat"
                                     else out.squeeze(-1).cpu()).reshape(-1))
            return torch.cat(outs).numpy() if outs else np.zeros(0)
        pa, pb = ev(train_t), ev(held_t)
        del o
        torch.cuda.empty_cache()
        return pa, pb

    if not cont_only:
        plan = list(OBS_PLAN_FULL) if full_ladder else [
            p for p in OBS_PLAN_FULL if p[0] in ("O_io", "O_hist", "O_act") and p[1] == "8:256"]
        if impl_family and full_ladder:
            plan = plan + OBS_PLAN_IMPL_EXTRA
        res["observers"] = {}
        for (nm, capstr, extras, tokens, frac) in plan:
            ol, oe = (int(z) for z in capstr.split(":"))
            pa, pb = fit_observer(ol, oe, extras, tokens, frac, n_cls)
            r = scored((pa == y[evA]).astype(np.float64), (pb == y[evB]).astype(np.float64))
            res["observers"][f"{nm}@{capstr}"] = r
            print(f"  {nm}@{capstr}: held_seq={r['held_seq']:.3f} held_both={r['held_both']:.3f}")

        bio = max(r_["held_seq"] for kk, r_ in res["observers"].items()
                  if kk.startswith("O_io@"))
        bio_b = max(r_["held_both"] for kk, r_ in res["observers"].items()
                    if kk.startswith("O_io@"))
        res["advantage"] = {"held_seq": res["self"]["held_seq"] - bio,
                            "held_both": res["self"]["held_both"] - bio_b,
                            "vs_hist": res["self"]["held_seq"]
                                       - res["observers"].get("O_hist@8:256", {}).get("held_seq", float("nan")),
                            "vs_act": res["self"]["held_seq"]
                                      - res["observers"].get("O_act@8:256", {}).get("held_seq", float("nan"))}
        lvA = lv_all[evA]
        res["advantage_by_level"] = {}
        bio_key = max((kk for kk in res["observers"] if kk.startswith("O_io@")),
                      key=lambda kk: res["observers"][kk]["held_seq"])
        for l in np.unique(lvA):
            res["advantage_by_level"][int(l)] = (
                res["self"]["held_seq_by_level"][int(l)]
                - res["observers"][bio_key]["held_seq_by_level"][int(l)])
        print(f"  ADVANTAGE held_seq={res['advantage']['held_seq']:+.3f}  "
              f"held_both={res['advantage']['held_both']:+.3f}  "
              f"by level " + " ".join(f"L{l}:{x:+.3f}" for l, x in res["advantage_by_level"].items()))

    # ---- continuous readout (secondary) ----
    if ycont is not None:
        # z-score within (checkpoint, level) on train-sequence rows: the same
        # conditioning the categorical bins get, so the two readouts agree on what
        # "within level" means.
        zc = ycont.astype(np.float64)                       # (n_tc, N, P)
        z = np.zeros_like(zc)
        for t in range(n_tc):
            for l in np.unique(lv):
                cols = np.where(lv == l)[0]
                blk = zc[t][:, cols]
                mu, sd = blk[:n_str].mean(), blk[:n_str].std() + 1e-9
                z[t][:, cols] = (blk - mu) / sd
        z = z.reshape(-1)
        zt = torch.from_numpy(z.astype(np.float32))
        lvA, lvB = lv_all[evA], lv_all[evB]

        def cont_score(pa, pb):
            d = {"held_seq": _spearman(pa, z[evA]),
                 "held_both": _spearman(pb, z[evB]) if len(evB) else float("nan"),
                 "held_seq_by_level": {int(l): _spearman(pa[lvA == l], z[evA][lvA == l])
                                       for l in np.unique(lvA)},
                 "held_both_by_level": ({int(l): _spearman(pb[lvB == l], z[evB][lvB == l])
                                         for l in np.unique(lvB)} if len(evB) else {})}
            return d

        ev_c = fit_mlp(x7, 1, mode="cont", tgt=zt)
        pa_s, pb_s = ev_c(evA), (ev_c(evB) if len(evB) else np.zeros(0))
        res["cont"] = {"self": cont_score(pa_s, pb_s)}
        preds = {"z_evA": z[evA].astype(np.float32), "z_evB": z[evB].astype(np.float32),
                 "lv_evA": lvA, "lv_evB": lvB,
                 "self_evA": pa_s.astype(np.float32), "self_evB": pb_s.astype(np.float32)}
        zcube = zt.reshape(n_tc, N, P)
        for spec in CONT_OBS:
            nm, capstr = spec.split("@")
            ol, oe = (int(x_) for x_ in capstr.split(":"))
            extras, tokens = {"O_io": (["logit_c"], True),
                              "O_hist": (["logit_c", "logit_prev"], True),
                              "O_act": (["a0_c"], False)}[nm]
            pa, pb = fit_observer(ol, oe, extras, tokens, 1.0, 1, mode="cont",
                                  tgtcube=zcube)
            res["cont"][spec] = cont_score(pa, pb)
            preds[f"{nm}_evA"] = pa.astype(np.float32)
            preds[f"{nm}_evB"] = pb.astype(np.float32)
        np.savez(f"{pdir}/cells/{cell}{out_suffix}_contpred.npz", **preds)
        print("  [cont] " + "  ".join(f"{kk}={vv['held_seq']:+.3f}"
                                      for kk, vv in res["cont"].items()))
        bio_c = res["cont"].get("O_io@8:256", {}).get("held_seq", float("nan"))
        print("  [cont by level] self " + " ".join(
            f"L{l}:{v:+.3f}" for l, v in res["cont"]["self"]["held_seq_by_level"].items()))
        res["cont_advantage_by_level"] = {
            int(l): (res["cont"]["self"]["held_seq_by_level"][int(l)]
                     - max(res["cont"].get(sp, {}).get("held_seq_by_level", {}).get(int(l), -9)
                           for sp in CONT_OBS))
            for l in res["cont"]["self"]["held_seq_by_level"]}

    res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(outfn, "w") as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"  saved -> {outfn}  peak RSS {res['peak_rss_gb']:.2f} GB")
    return res


# ======================================================================
# Stage E -- the occurrent control battery at the final checkpoint
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=20480)
def control_battery(
    target: str = "IMPL",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    fwd_n_layer: int = 1, fwd_n_head: int = 8, fwd_d_head: int = 16,
    fwd_mlp_mult: float = 1.0, fwd_lr: float = 1e-3,
    cond: str = "ntp_aux_cl", n_steps: int = 20000, n_ckpt: int = 16,
    first_ckpt: int = 400, seed: int = 42,
    pool_size: int = 200000, data_seed: int = 7, batch_size: int = 64,
    n_report_sequences: int = 12000, report_seed: int = 999,
    fresh_fm_steps: int = 3000, impl_k: int = 8, world_level: int = 3,
    head_hidden: int = 256, head_steps: int = 3000, head_lr: float = 1e-3,
    observer_caps: str = "1:64,2:128,4:192,8:256",
    obs_steps: int = 3000, obs_lr: float = 3e-4, obs_bs: int = 64,
    tag: str = "main",
):
    """Paper 2's occurrent row (IMPL / BEHAV / ENT / WORLD) on THIS model's final
    checkpoint, with the parent's conventions verbatim (12k report sequences, one
    sequence-level split, uncentered residual, fresh instrument at h16m1). This is the
    'battery shown alive on this substrate' gate that the dispositional rows are read
    against; published reference is IMPL +0.096 / BEHAV -0.001 / ENT -0.035 / WORLD
    -0.038 at val 2.3829."""
    import numpy as np
    import resource
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT, Block
    from rhm.rhm_data import generate_rules_distinct
    from rhm.confabulation.rhm_confabulation import (
        _generate_with_traces, _kmeans_fit, _kmeans_assign, _balance,
        _make_head, _fit_head, _head_acc, _ensemble_cos)
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    pdir = part_dir(key, tag)
    os.makedirs(f"{pdir}/cells", exist_ok=True)
    outfn = f"{pdir}/cells/CTRL_{target}.json"
    if os.path.exists(outfn):
        with open(outfn) as f:
            return json.load(f)
    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    caps = [tuple(int(z) for z in c.split(":")) for c in observer_caps.split(",")]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus, arangeT = corpus.shape[0], torch.arange(T)
    del pool_seqs

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        f"{pdir}/ckpt/{cond}_step{sched[-1]}_seed{seed}.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    val = 0.0
    gen = torch.Generator().manual_seed(report_seed + 5)
    with torch.no_grad():
        for _ in range(10):
            x, y_ = get_ntp_batch(gen)
            val += model(x, y_)[1].item() / 10
    print(f"  CTRL {target}: final-checkpoint val={val:.4f} (reference 2.3829)")

    rep_seqs, rep_lf, _ = _generate_with_traces(rules, n_report_sequences, report_seed)
    rep_x = torch.from_numpy(rep_seqs.astype(np.int64)).to(device)
    N = rep_x.shape[0]
    rb = f"post_block{n_layer - 1}"
    c_a0, c_a6, c_x7, c_lg = [], [], [], []
    with torch.no_grad():
        for i in range(0, N, 128):
            lg_, _, vi = model(rep_x[i:i + 128], return_intermediates=True)
            c_a0.append(vi[predict_from].cpu()); c_a6.append(vi[predict_to].cpu())
            c_x7.append(vi[rb].cpu()); c_lg.append(lg_.cpu())
    a0 = torch.cat(c_a0); a6 = torch.cat(c_a6)
    x7 = torch.cat(c_x7); logit = torch.cat(c_lg)
    del c_a0, c_a6, c_x7, c_lg
    tok = torch.from_numpy(rep_seqs.astype(np.int64))

    n_str = int(0.8 * N)
    s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)
    pos_of = lambda si: (si[:, None] * (T - 1) + torch.arange(T - 1)[None, :]).reshape(-1)
    tr, te = pos_of(s_tr), pos_of(s_te)
    flat = lambda z: z[:, :T - 1].reshape(N * (T - 1), -1).contiguous()
    LG = flat(logit)
    X_full = flat(x7)

    if target == "IMPL":
        torch.manual_seed(seed + 911)
        fm = TransformerForwardModel(d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                                     n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                                     block_size=T).to(device)
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        g2 = torch.Generator().manual_seed(seed + 912)
        for _ in range(fresh_fm_steps):
            fm.train()
            xb, _ = get_ntp_batch(g2)
            with torch.no_grad():
                _, _, vi = model(xb, return_intermediates=True)
            F.mse_loss(fm(vi[predict_from]), vi[predict_to]).backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        fm.eval()
        pred = []
        with torch.no_grad():
            for i in range(0, N, 128):
                pred.append(fm(a0[i:i + 128].to(device)).cpu())
        pred = torch.cat(pred)
        resid = a6 - pred
        fwd_cos = float(F.cosine_similarity(pred, a6, dim=-1).mean())
        R = flat(resid)
        cents = _kmeans_fit(R[tr].to(device), impl_k, seed=seed)
        yv = _kmeans_assign(R.to(device), cents).cpu()
        ncls = impl_k
        extra_diag = {"fwd_cosine": fwd_cos}
    elif target == "BEHAV":
        yv = (logit[:, :T - 1].argmax(-1).reshape(-1) == tok[:, 1:].reshape(-1)).long()
        ncls, extra_diag = 2, {}
    elif target == "ENT":
        lp = F.log_softmax(LG, -1)
        e_ = -(lp.exp() * lp).sum(-1)
        yv = torch.bucketize(e_, torch.quantile(e_[tr], torch.tensor([.25, .5, .75])))
        ncls, extra_diag = 4, {}
    else:
        wn = rep_lf[world_level][:, (np.arange(T) // (s ** (L - world_level)))]
        yv = torch.from_numpy(wn.astype(np.int64))[:, :T - 1].reshape(-1)
        ncls, extra_diag = v, {}

    h = _fit_head(_make_head(n_embd, ncls, head_hidden, device, seed),
                  X_full, yv, tr, head_steps, head_lr, device)
    self_acc = _head_acc(h, X_full, yv, te, device)

    class Observer(nn.Module):
        def __init__(self, n_cls, o_layer, o_embd, extra_dim=0, tokens=True):
            super().__init__()
            self.tokens = tokens
            if tokens:
                self.wte = nn.Embedding(v, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            self.proj = nn.Linear(extra_dim, o_embd) if extra_dim else None
            self.h = nn.ModuleList([Block(o_embd, min(o_layer * 2, 8), T, causal=True)
                                    for _ in range(o_layer)])
            self.ln_f = nn.LayerNorm(o_embd)
            self.head = nn.Linear(o_embd, n_cls)

        def forward(self, tokb, extra=None):
            B, t = (tokb.shape if self.tokens else extra.shape[:2])
            x = self.wpe(torch.arange(t, device=self.wpe.weight.device))[None].expand(B, t, -1)
            if self.tokens:
                x = x + self.wte(tokb)
            if self.proj is not None:
                x = x + self.proj(extra)
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    yt = yv.view(N, T - 1)

    def train_observer(o_layer, o_embd, extra_src, use_tokens, frac=1.0):
        torch.manual_seed(seed + 31)
        o = Observer(ncls, o_layer, o_embd,
                     extra_dim=(0 if extra_src is None else extra_src.shape[-1]),
                     tokens=use_tokens).to(device)
        opt = torch.optim.AdamW(o.parameters(), lr=obs_lr, weight_decay=0.01)
        g = torch.Generator().manual_seed(seed + 32)
        pool = s_tr[:max(1, int(frac * len(s_tr)))]
        for _ in range(obs_steps):
            si = pool[torch.randint(len(pool), (obs_bs,), generator=g)]
            xb = tok[si, :T - 1].to(device)
            eb = None if extra_src is None else extra_src[si, :T - 1].to(device)
            F.cross_entropy(o(xb, eb).reshape(-1, ncls),
                            yt[si].reshape(-1).to(device)).backward()
            torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        o.eval()
        corr = tot_ = 0
        with torch.no_grad():
            for i in range(0, len(s_te), 64):
                si = s_te[i:i + 64]
                xb = tok[si, :T - 1].to(device)
                eb = None if extra_src is None else extra_src[si, :T - 1].to(device)
                p_ = o(xb, eb).argmax(-1).cpu()
                corr += int((p_ == yt[si]).sum()); tot_ += p_.numel()
        del o
        torch.cuda.empty_cache()
        return corr / tot_

    obs = {}
    for (ol, oe) in caps:
        obs[f"O_io@{ol}:{oe}"] = train_observer(ol, oe, logit, True)
    for (ol, oe) in (caps[0], caps[-1]):
        obs[f"O_input@{ol}:{oe}"] = train_observer(ol, oe, None, True)
    ol, oe = caps[-1]
    obs[f"O_io_half@{ol}:{oe}"] = train_observer(ol, oe, logit, True, frac=0.5)
    obs[f"O_act@{ol}:{oe}"] = train_observer(ol, oe, a0, False)
    bio = max(vv for kk, vv in obs.items() if kk.startswith("O_io@"))
    res = {"target": target, "val": val, "self": self_acc, "observers": obs,
           "best_O_io": bio, "advantage": self_acc - bio,
           "baseline": _balance(yv[te], ncls), **extra_diag,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    print(f"  CTRL {target}: self={self_acc:.3f} best_O_io={bio:.3f} "
          f"advantage={self_acc - bio:+.3f} base={res['baseline']:.3f}")
    print("    " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in obs.items()))
    with open(outfn, "w") as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return res


# ======================================================================
# Coordinator (CPU) -- wake -> instruments -> targets -> cells -> summary
# ======================================================================

def _waves(fn, kw_list, max_dop):
    out = []
    for i in range(0, len(kw_list), max_dop):
        hs = [fn.spawn(**kw) for kw in kw_list[i:i + max_dop]]
        out += [h.get() for h in hs]
    return out


@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=4096, cpu=2.0)
def dispositional_run(
    tag: str = "main", smoke: bool = False, max_dop: int = 6,
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, seed: int = 42,
    cond: str = "ntp_aux_cl", n_steps: int = 20000, n_ckpt: int = 16,
    first_ckpt: int = 400, n_report_sequences: int = 3000,
    fresh_fm_steps: int = 3000, ens_n: int = 3, diag_seqs: int = 2000,
    inst_caps_str: str = "16:1.0,4:0.25", horizon: int = 2, k_gate_max: int = 4,
    held_window: int = 3, head_steps: int = 5000, obs_steps: int = 5000,
    ctrl: bool = True, auto_k: bool = True, out_suffix: str = "",
    horizons: str = "",
):
    """`horizons` (e.g. "1,3"): build targets and run the cells at each horizon in turn,
    writing `<tag>_k<h>_summary.json` per horizon. The wake, the instrument parts and the
    occurrent control battery are shared across horizons -- only `build_targets` and the
    cells are repeated, which is the cheap half."""
    key = tb_key(v, s, depth, m)
    pdir = part_dir(key, tag)
    os.makedirs(pdir, exist_ok=True)
    ctrl_kw = {}
    if smoke:
        n_steps, n_ckpt, first_ckpt = 300, 6, 50
        n_report_sequences, fresh_fm_steps, ens_n, diag_seqs = 300, 150, 2, 150
        horizon, k_gate_max, held_window = 1, 2, 1
        head_steps, obs_steps = 200, 200
        ctrl_kw = dict(n_report_sequences=800, fresh_fm_steps=150, head_steps=200,
                       obs_steps=200, observer_caps="1:64,2:128")
    base = dict(v=v, s=s, depth=depth, m=m, seed=seed, tag=tag)
    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    print(f"### dispositional_run tag={tag} smoke={smoke}\n### schedule {sched}")

    wake = wake_trajectory.remote(cond=cond, n_steps=n_steps, n_ckpt=n_ckpt,
                                  first_ckpt=first_ckpt, **base)
    print(f"### wake done: {wake}")

    inst = _waves(instrument_at_checkpoint, [
        dict(step=st, cond=cond, n_report_sequences=n_report_sequences,
             fresh_fm_steps=fresh_fm_steps, ens_n=ens_n, diag_seqs=diag_seqs,
             inst_caps_str=inst_caps_str, **base) for st in sched], max_dop)
    print(f"### instruments done ({len(inst)})")

    ctrl_res = []
    if ctrl:
        ctrl_res = _waves(control_battery, [
            dict(target=t, cond=cond, n_steps=n_steps, n_ckpt=n_ckpt,
                 first_ckpt=first_ckpt, **base, **ctrl_kw)
            for t in ("IMPL", "BEHAV", "ENT", "WORLD")], max_dop)
        print(f"### control battery done ({len(ctrl_res)})")

    caps = [f"h{c.split(':')[0]}m{float(c.split(':')[1]):g}" for c in inst_caps_str.split(",")]
    c0 = caps[0]
    hlist = ([(int(x), f"_k{int(x)}", False) for x in horizons.split(",")]
             if horizons else [(horizon, out_suffix, auto_k)])
    out = {}
    for (hh, sfx, ak) in hlist:
        tmeta = build_targets.remote(
            v=v, s=s, depth=depth, m=m, n_steps=n_steps, n_ckpt=n_ckpt,
            first_ckpt=first_ckpt, seed=seed, inst_caps_str=inst_caps_str,
            horizon=hh, k_gate_max=k_gate_max, held_window=held_window,
            tag=tag, auto_k=ak, out_suffix=sfx)
        print(f"### targets{sfx} done: k={tmeta['horizon']} steps={tmeta['tc_steps']}")
        cellkw = dict(n_ckpt=n_ckpt, first_ckpt=first_ckpt, n_steps=n_steps,
                      head_steps=head_steps, obs_steps=obs_steps,
                      out_suffix=sfx, **base)
        cells = [
            dict(cell="IMPL_PRO", target="IMPL_PRO_cls", cap=c0),
            dict(cell="IMPL_RETRO", target="IMPL_RETRO_cls", cap=c0),
            dict(cell="IMPL_NOW", target="IMPL_NOW_cls", cap=c0),
            dict(cell="REORG_PRO", target="REORG_PRO_bin", cap=c0, readout="both"),
            dict(cell="REORG_RETRO", target="REORG_RETRO_bin", cap=c0, readout="both"),
            dict(cell="BEHAV_PRO", target="BEHAV_PRO_bin", cap=c0, readout="both"),
            dict(cell="BEHAV_RETRO", target="BEHAV_RETRO_bin", cap=c0, readout="both"),
            dict(cell="ENT_NOW", target="ENT_bin", cap=c0),
        ]
        if len(caps) > 1:
            cells += [dict(cell="IMPL_PRO_" + caps[1], target="IMPL_PRO_cls", cap=caps[1],
                           with_confab=False),
                      dict(cell="IMPL_RETRO_" + caps[1], target="IMPL_RETRO_cls",
                           cap=caps[1], with_confab=False)]
        cres = _waves(battery_cell, [{**cellkw, **c} for c in cells], max_dop)
        print(f"### cells{sfx} done ({len(cres)})")

        summary = {"tag": tag, "arm": cond, "suffix": sfx, "smoke": smoke,
                   "schedule": sched, "wake": wake,
                   "instruments": {str(mm["step"]): mm for mm in inst},
                   "targets": tmeta,
                   "cells": {c["cell"]: c for c in cres},
                   "control": {c["target"]: c for c in ctrl_res}}
        fn = f"{pdir}/{tag}{sfx}_summary.json"
        with open(fn, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        volume.commit()
        print(f"\n### saved -> {fn}")
        for nm, c in summary["cells"].items():
            print(f"  {nm:22s} self={c['self']['held_seq']:.3f} "
                  f"adv={c['advantage']['held_seq']:+.3f} "
                  f"adv(ckpt-held)={c['advantage']['held_both']:+.3f}")
        for nm, c in summary["control"].items():
            print(f"  CTRL {nm:17s} self={c['self']:.3f} adv={c['advantage']:+.3f}")
        out[sfx or "_default"] = summary
    return out


@app.local_entrypoint()
def main(tag: str = "main", smoke: bool = False, max_dop: int = 6):
    dispositional_run.remote(tag=tag, smoke=smoke, max_dop=max_dop)


# ======================================================================
# Re-analysis: the per-position no-change floor, by level
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=12288)
def floor_by_level(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, n_embd: int = 256,
    fwd_n_layer: int = 1, fwd_n_head: int = 8,
    n_steps: int = 20000, n_ckpt: int = 16, first_ckpt: int = 400,
    cap: str = "h16m1", ens_n: int = 3, tags: str = "ol,main",
):
    """`1 - cos(r^{s_i}_c(p), r^{s_j}_c(p))` per position, per level: the disagreement
    between INDEPENDENT fresh FMs at the SAME checkpoint. This is the no-change floor for
    the *fresh-instrument* reorganization reading `1 - cos(r_c, r_{c+k})`, which is how
    much of that reading is instrument variation rather than M's change. The headline
    REORG rows use the FIXED-instrument reading instead, whose floor is exactly 0 by
    construction (the same FM sits on both sides), so this number bounds the fresh
    variant only. Retrains nothing: it reloads the saved instrument FMs and the cached
    report-set state."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    P = T - 1
    key = tb_key(v, s, depth, m)
    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    lv = np.array(level_of_pos(T, s, L))
    dh = int(cap.split("m")[0][1:])
    mm = float(cap.split("m")[1])
    out = {}
    for tag in [t.strip() for t in tags.split(",")]:
        pdir = part_dir(key, tag)
        out[tag] = {}
        for st in sched:
            z = np.load(f"{pdir}/parts/step{st}.npz")
            a0, a6 = z["a0"], z["a6"]
            z.close()
            N = a0.shape[0]
            rs = []
            for j in range(ens_n):
                fm = TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=fwd_n_head,
                                             n_layer=fwd_n_layer, mlp_mult=mm,
                                             block_size=T).to(device)
                fm.load_state_dict(torch.load(f"{pdir}/fm/{cap}_step{st}_s{j}.pt",
                                              map_location=device))
                fm.eval()
                r = []
                with torch.no_grad():
                    for i in range(0, N, 256):
                        src = torch.from_numpy(a0[i:i + 256]).to(device).float()
                        tgt = torch.from_numpy(a6[i:i + 256]).to(device).float()
                        r.append((tgt - fm(src))[:, :P].half().cpu())
                rs.append(torch.cat(r))
                del fm
                torch.cuda.empty_cache()
            pairs_raw, pairs_cen = [], []
            for i in range(ens_n):
                for j in range(i + 1, ens_n):
                    with torch.no_grad():
                        ri, rj = rs[i].float(), rs[j].float()
                        pairs_raw.append((1.0 - F.cosine_similarity(ri, rj, dim=-1)).numpy())
                        ci = ri - ri.mean(dim=(0, 1), keepdim=True)
                        cj = rj - rj.mean(dim=(0, 1), keepdim=True)
                        pairs_cen.append((1.0 - F.cosine_similarity(ci, cj, dim=-1)).numpy())
            raw = np.mean(np.stack(pairs_raw), 0)          # (N, P)
            cen = np.mean(np.stack(pairs_cen), 0)
            out[tag][str(st)] = {
                "floor_raw_by_level": {int(l): float(raw[:, lv == l].mean())
                                       for l in range(L)},
                "floor_centered_by_level": {int(l): float(cen[:, lv == l].mean())
                                            for l in range(L)},
                "floor_raw_pooled": float(raw.mean()),
                "floor_centered_pooled": float(cen.mean())}
            print(f"  {tag} step {st:6d}: floor(raw) pooled={raw.mean():.3f}  "
                  + " ".join(f"L{l}={out[tag][str(st)]['floor_raw_by_level'][l]:.3f}"
                             for l in range(L)))
            del rs
            torch.cuda.empty_cache()
    fn = f"{DATA_DIR}/rhm_confabulation/{key}/dispositional/floor_by_level_{cap}.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"  saved -> {fn}")
    return out


# ======================================================================
# Follow-up: the excess-over-floor reorganization target, and the runner
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=16384)
def add_excess_targets(
    tag: str = "ol", out_suffix: str = "_k1",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, n_embd: int = 256,
    fwd_n_layer: int = 1, fwd_n_head: int = 8,
    n_steps: int = 20000, n_ckpt: int = 16, first_ckpt: int = 400,
    cap: str = "h16m1", ens_n: int = 3, n_bins: int = 4, seq_train_frac: float = 0.8,
):
    """Adds `REORG_{PRO,RETRO}_excess` to an existing targets file: the FRESH-instrument
    reorganization minus the per-position same-checkpoint no-change floor, so the grammar
    carries all three readings the language sibling has —

      raw fresh    1 - cos(r_c, r_{c±k})            floor = the instrument's own disagreement
      excess       that, minus floor_c(p)           floor removed per position
      fixed        1 - cos(r^{(c)}_c, r^{(c)}_{c±k}) floor = 0 by construction

    floor_c(p) = mean over independent fresh-FM pairs of 1 - cos(r^{s_i}_c(p), r^{s_j}_c(p)).
    It is taken at checkpoint `c` and not averaged with `c±k` because the no-change
    counterfactual is exactly "M did not move, so the far instrument is just another fresh
    FM on M_c". Retrains nothing; reloads the saved instruments and cached state."""
    import numpy as np
    import resource
    import torch
    import torch.nn.functional as F
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    P = T - 1
    key = tb_key(v, s, depth, m)
    pdir = part_dir(key, tag)
    sched = ckpt_schedule(n_steps, n_ckpt, first_ckpt)
    lv = np.array(level_of_pos(T, s, L))
    dh, mm = int(cap.split("m")[0][1:]), float(cap.split("m")[1])

    tz = np.load(f"{pdir}/targets{out_suffix}.npz")
    store = {k_: tz[k_] for k_ in tz.files}
    tz.close()
    with open(f"{pdir}/targets_meta{out_suffix}.json") as f:
        tmeta = json.load(f)
    k, N = tmeta["horizon"], tmeta["N"]
    tc_all = store["tc_indices"].tolist()
    n_tc = len(tc_all)
    n_str = int(seq_train_frac * N)
    print(f"{'=' * 74}\nADD EXCESS TARGETS  tag={tag} suffix={out_suffix} k={k} "
          f"n_tc={n_tc}")

    def load_fm(st, j):
        fm = TransformerForwardModel(d_model=n_embd, d_head=dh, n_head=fwd_n_head,
                                     n_layer=fwd_n_layer, mlp_mult=mm,
                                     block_size=T).to(device)
        fm.load_state_dict(torch.load(f"{pdir}/fm/{cap}_step{st}_s{j}.pt",
                                      map_location=device))
        fm.eval()
        for p_ in fm.parameters():
            p_.requires_grad = False
        return fm

    floor = np.empty((n_tc, N, P), dtype=np.float32)
    for t, c in enumerate(tc_all):
        z = np.load(f"{pdir}/parts/step{sched[c]}.npz")
        a0, a6 = z["a0"], z["a6"]
        z.close()
        rs = []
        for j in range(ens_n):
            fm = load_fm(sched[c], j)
            r = []
            with torch.no_grad():
                for i in range(0, N, 256):
                    src = torch.from_numpy(a0[i:i + 256]).to(device).float()
                    tgt = torch.from_numpy(a6[i:i + 256]).to(device).float()
                    r.append((tgt - fm(src))[:, :P].half().cpu())
            rs.append(torch.cat(r))
            del fm
            torch.cuda.empty_cache()
        pairs = []
        with torch.no_grad():
            for i in range(ens_n):
                for j in range(i + 1, ens_n):
                    pairs.append((1.0 - F.cosine_similarity(
                        rs[i].float(), rs[j].float(), dim=-1)).numpy())
        floor[t] = np.mean(np.stack(pairs), 0)
        del rs, a0, a6
        torch.cuda.empty_cache()
        print(f"    step {sched[c]:6d}: floor pooled={floor[t].mean():.3f}  "
              + " ".join(f"L{l}={floor[t][:, lv == l].mean():.3f}" for l in range(L)))

    ci_rows = np.repeat(np.arange(n_tc), N * P)
    lv_rows = np.tile(lv, n_tc * N)
    seq_rows = np.tile(np.repeat(np.arange(N), P), n_tc)
    train_mask = seq_rows < n_str
    added = {}
    for nm in ("REORG_PRO", "REORG_RETRO"):
        fresh = store[f"{cap}__{nm}_fresh_cont"]
        ex = (fresh - floor).astype(np.float32)
        store[f"{cap}__{nm}_excess_cont"] = ex
        store[f"{cap}__{nm}_excess_bin"] = _bin_within(
            ex.reshape(-1), lv_rows, ci_rows, n_bins, train_mask).reshape(n_tc, N, P)
        store[f"{cap}__{nm}_excess_binpooled"] = _bin_pooled(
            ex.reshape(-1), n_bins, train_mask).reshape(n_tc, N, P)
        added[nm] = {
            "floor_by_level": {int(l): float(floor[:, :, lv == l].mean()) for l in range(L)},
            "fresh_by_level": {int(l): float(fresh[:, :, lv == l].mean()) for l in range(L)},
            "excess_by_level": {int(l): float(ex[:, :, lv == l].mean()) for l in range(L)},
            "fixed_by_level": {int(l): float(store[f"{cap}__{nm}_cont"][:, :, lv == l].mean())
                               for l in range(L)},
            "floor_share_by_level": {
                int(l): float(floor[:, :, lv == l].mean() / fresh[:, :, lv == l].mean())
                for l in range(L)},
            "rank_corr_fresh_vs_excess": float(np.corrcoef(
                np.argsort(np.argsort(fresh.reshape(-1))),
                np.argsort(np.argsort(ex.reshape(-1))))[0, 1]),
        }
        print(f"  {nm}: floor share by level " + " ".join(
            f"L{l}={added[nm]['floor_share_by_level'][l]:.2f}" for l in range(L))
            + f"   rank corr(fresh, excess)={added[nm]['rank_corr_fresh_vs_excess']:.3f}")
    np.savez(f"{pdir}/targets{out_suffix}.npz", **store)
    fn = f"{pdir}/excess_meta{out_suffix}.json"
    with open(fn, "w") as f:
        json.dump({"horizon": k, "tc_steps": tmeta["tc_steps"], "cap": cap,
                   "ens_n": ens_n, "targets": added,
                   "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"  saved -> {fn}")
    return added


@app.function(volumes={DATA_DIR: volume}, timeout=86400, memory=4096, cpu=2.0)
def followups(
    tag: str = "ol", max_dop: int = 6, cap: str = "h16m1",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, seed: int = 42,
    n_steps: int = 20000, n_ckpt: int = 16, first_ckpt: int = 400,
    head_steps: int = 5000, obs_steps: int = 5000, suffixes: str = "_k1,_k3",
):
    """(1) the fresh and excess-over-floor reorganization readings through the full cell
    battery, and (2) the continuous rank readout re-scored within level with its
    predictions saved -- both horizons, reusing everything already on the volume."""
    key = tb_key(v, s, depth, m)
    pdir = part_dir(key, tag)
    base = dict(v=v, s=s, depth=depth, m=m, seed=seed, tag=tag)
    cellkw = dict(n_ckpt=n_ckpt, first_ckpt=first_ckpt, n_steps=n_steps,
                  head_steps=head_steps, obs_steps=obs_steps, cap=cap, **base)
    out = {}
    for sfx in [z.strip() for z in suffixes.split(",")]:
        ex = add_excess_targets.remote(tag=tag, out_suffix=sfx, cap=cap, n_steps=n_steps,
                                       n_ckpt=n_ckpt, first_ckpt=first_ckpt,
                                       v=v, s=s, depth=depth, m=m)
        print(f"### excess targets{sfx} added")
        jobs = []
        for nm in ("REORG_PRO", "REORG_RETRO"):
            for variant in ("fresh", "excess"):
                jobs.append({**cellkw, "out_suffix": sfx,
                             "cell": f"{nm}_{variant}", "target": f"{nm}_{variant}_bin",
                             "readout": "both", "with_confab": True})
        for nm in ("REORG_PRO", "REORG_RETRO", "BEHAV_PRO", "BEHAV_RETRO"):
            jobs.append({**cellkw, "out_suffix": sfx, "cell": f"{nm}_cont",
                         "target": f"{nm}_bin", "readout": "both", "cont_only": True})
        cres = _waves(battery_cell, jobs, max_dop)
        print(f"### follow-up cells{sfx} done ({len(cres)})")
        out[sfx] = {"excess_meta": ex, "cells": {c["cell"]: c for c in cres}}
    fn = f"{pdir}/{tag}_followups_summary.json"
    with open(fn, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n### saved -> {fn}")
    for sfx, d in out.items():
        for nm, c in d["cells"].items():
            if c.get("cont_only"):
                cs = c.get("cont", {})
                print(f"  {sfx} {nm:22s} [cont] self={cs.get('self', {}).get('held_seq', float('nan')):+.3f}")
            else:
                print(f"  {sfx} {nm:22s} self={c['self']['held_seq']:.3f} "
                      f"adv={c['advantage']['held_seq']:+.3f} "
                      f"adv(ck)={c['advantage']['held_both']:+.3f}")
    return out
