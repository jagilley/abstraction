"""Conditional revision, Gate 0: does moving the FM's conditioning gap from DEPTH to
TIME give its residual an aleatoric component?

See SPEC.md. Gate 0 only -- measurement, no intervention, no oracle BP, no arms.

  depth FM   (the incumbent):  FM( h0[<=t] )  ->  h6[t]
  temporal FM (this cut)    :  FM( h6[<=t] )  ->  Delta_t = h6[t+1] - h6[t]

The depth FM's predictor and target have identical information sets, so its residual
is "I lacked capacity". The temporal FM's target depends on x_{t+1}, which the corpus
supplies and the FM cannot hold -- so for the first time in this line the residual has
a nonzero aleatoric component. Gate 0 asks whether that shows up as a relationship to
token surprisal `nll`, and whether the level profile flips from anti-localised to
aligned.

Four design choices from SPEC.md "The object", all honoured here:
  1. deepest block, not shallowest -- target is built on h6 (post_block6), so the
     residual is not the token by construction.
  2. predict the UPDATE Delta_t, not the state h6[t+1] -- residual streams carry state
     forward hard (ACTIVE_VISION scored 0.94 by echoing carried state).
  3. full prefix -- the FM is causal over all of h6[<=t], so the conditioning gap is
     exactly one token.
  4. open-loop, stop-grad on the frozen main model -- the main model is in eval() and
     every activation is produced under no_grad. Nothing can collapse.

FM architecture is TransformerForwardModel 1L/8H/16d, identical to
endogenous_teacher's, so parameter count and capacity are matched.

Readout parametrisation (stated because it is the one place this deviates from a
literal reuse of endogenous_teacher's call): TransformerForwardModel is a residual
stream (`out = x + f(x)`), so feeding it h6 and reading `out` biases the prediction
toward h6 itself -- a fine init when the target is another state (depth FM), a bad one
when the target is a small update. The primary temporal FM therefore reads the FM's
OWN residual update, `pred = fm(h6) - h6`, which starts at ~0 and is exactly "predict
the update". Parameters are untouched. A `direct` variant (`pred = fm(h6)`) is trained
in the same loop as a control so the parametrisation cannot be load-bearing.

Controls trained alongside, sharing one frozen forward pass:
  depth_frozen  -- h0[<=t] -> h6[t] trained against the FROZEN base for the same step
                   budget. The depth FM in endogenous_teacher was CO-trained against a
                   moving base; this arm matches the temporal FM's protocol exactly, so
                   any difference is the axis change and not the training protocol.
  depth_cotrain -- h0[<=t] -> h6[t] co-trained during base training, i.e. literally
                   endogenous_teacher's FM. Its Gate-0 numbers should reproduce that
                   README's (-0.336 / 0.113 / 88.7%); if they do not, the substrate
                   differs and nothing here is comparable.

Run:
  # smoke (small, ~2 min)
  modal run -m rhm.conditional_revision.conditional_revision::gate0 \
      --base-steps 400 --fm-steps 400 --pool-size 20000 --n-eval-sequences 1024 \
      --gate0-batches 4 --tag smoke
  # the real thing (~1.5h on an L4)
  modal run --detach -m rhm.conditional_revision.conditional_revision::gate0 \
      --base-steps 12000 --fm-steps 12000 --tag gate0
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-conditional-revision", image=image)


def tb_key(v, s, L, m):
    """Volume dir. NOTE: the `_distinct` suffix is real -- the regime uses
    generate_rules_distinct, and `/data/v16_s2_L6_m4` does not exist on the volume.
    SPEC.md's path was wrong; endogenous_teacher/README.md's was right."""
    return f"{setting_key(v, s, L, m)}_distinct"


def _r2(y, x):
    """R^2 of predicting y from x by simple linear regression (flat tensors)."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum() + 1e-8
    beta = (xc * yc).sum() / denom
    pred = beta * xc
    ss_res = ((yc - pred) ** 2).sum()
    ss_tot = (yc * yc).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)


def _orthogonalise(y, x):
    """Residual of y after OLS regression on x (both flat, same device)."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum() + 1e-8
    beta = (xc * yc).sum() / denom
    return yc - beta * xc


def _corr(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _gate0_stats(R, NLL):
    """The four pooled Gate-0 numbers for a (B,T') residual/nll pair."""
    r2_pooled = _r2(R.reshape(-1), NLL.reshape(-1))
    r2_within = float(
        sum(_r2(R[:, t], NLL[:, t]) for t in range(R.shape[1])) / R.shape[1])
    orth = float(_orthogonalise(R.reshape(-1), NLL.reshape(-1)).var()
                 / (R.reshape(-1).var() + 1e-12))
    return {
        "r2_pooled": r2_pooled,
        "r2_within_position": r2_within,
        "orth_variance_fraction": orth,
        "corr_pooled": _corr(R.reshape(-1).cpu().numpy(), NLL.reshape(-1).cpu().numpy()),
    }


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def gate0(
    # DGP -- endogenous_teacher / RHM_LATENT_LOOP's regime, so reference lines transfer
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    # Model
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # Blocks
    shallow_block: str = "post_block0", deep_block: str = "post_block6",
    # FM (identical to endogenous_teacher's)
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    # Training
    base_steps: int = 12000, fm_steps: int = 12000,
    batch_size: int = 64, lr: float = 3e-4, fwd_lr: float = 1e-3,
    weight_decay: float = 0.01,
    pool_size: int = 200000, data_seed: int = 7,
    # Measurement
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    log_interval: int = 1000, gate0_batches: int = 20,
    save_ckpt: bool = True, base_ckpt: str = "",
    seed: int = 42, tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    chance = 1.0 / v
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    print(f"{'=' * 78}\nCONDITIONAL REVISION -- GATE 0   {key}  {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  occupancy m/v^(s-1)={m / v ** (s - 1):.3f}  chance={chance:.4f}  T={T}")
    print(f"  depth FM    {shallow_block}[<=t] -> {deep_block}[t]")
    print(f"  temporal FM {deep_block}[<=t] -> Delta_t = {deep_block}[t+1] - {deep_block}[t]")
    print(f"  FM {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d   base_steps={base_steps}  "
          f"fm_steps={fm_steps} (frozen base)")
    print(f"{'=' * 78}", flush=True)

    # ---------------- data ----------------
    print(f"Generating pool ({pool_size:,} seqs)...", flush=True)
    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))
    corpus = pool_x.reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    # Flat concatenated windows: what training sees. Position carries no fixed
    # hierarchy level here, which is the point -- it removes the position<->level
    # confound from the pooled statistics.
    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    # Aligned eval set: hierarchy level IS defined per position, so the level profile
    # lives here.
    eval_seqs, eval_lf, _ = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    y_level = {ell: torch.from_numpy(eval_lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
               for ell in range(L)}
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    # position -> HIGHEST hierarchy node completed there (smallest ell = closest to root).
    # Identical convention to endogenous_teacher so the level tables line up.
    pos_top_level = np.array(
        [min(ell for ell in range(L + 1) if (p + 1) % (s ** (L - ell)) == 0)
         for p in range(T)], dtype=np.int64)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    # ---------------- base training (plain NTP + co-trained depth FM) ----------------
    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    fm_depth_cotrain = make_fm()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_dc = torch.optim.AdamW(fm_depth_cotrain.parameters(), lr=fwd_lr,
                               weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    ckpt_path = f"{DATA_DIR}/{key}/conditional_revision/base_{n_layer}L{n_head}H{n_embd}D_" \
                f"steps{base_steps}_seed{seed}.pt"
    loaded = False
    src = base_ckpt or ckpt_path
    if os.path.exists(src):
        print(f"\n--- BASE: loading cached checkpoint {src} ---", flush=True)
        sd = torch.load(src, map_location=device)
        model.load_state_dict(sd["model"])
        fm_depth_cotrain.load_state_dict(sd["fm_depth_cotrain"])
        loaded = True
    else:
        print(f"\n--- BASE: {base_steps} steps, plain NTP + co-trained depth FM ---",
              flush=True)
        for step in range(base_steps):
            model.train(); fm_depth_cotrain.train()
            x, y = get_ntp_batch(gen)
            logits, _, inter = model(x, y, return_intermediates=True)
            nll_pp = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1), reduction="none")
            loss = nll_pp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            # open loop: the FM watches detached activations, nothing flows back
            pred = fm_depth_cotrain(inter[shallow_block].detach())
            fwd_loss = F.mse_loss(pred, inter[deep_block].detach())
            opt_dc.zero_grad(); fwd_loss.backward(); opt_dc.step()
            if step % log_interval == 0 or step == base_steps - 1:
                print(f"  base {step:6d}  ntp {loss.item():.4f}  fm_mse {fwd_loss.item():.4f}",
                      flush=True)

    # ---- the main model is FROZEN from here on. Design choice 4. ----
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def per_level_recovery():
        acts = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, eval_x.shape[0], 256):
                _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
                for b in block_names:
                    acts[b].append(inter[b][:, -1, :].float())
        acts = {b: torch.cat(vs) for b, vs in acts.items()}
        out = {}
        for ell in range(L):
            best = max(
                max(_probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr),
                    _probe_acc(acts[b], y_level[ell], v, device, mlp_steps, probe_lr,
                               hidden=mlp_hidden))
                for b in block_names)
            out[f"d{L - ell}"] = best   # repo convention: d1 shallowest, d6 = root
        return out

    def eval_val():
        g = torch.Generator().manual_seed(eval_seed + 5)
        tot = 0.0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_ntp_batch(g)
                _, loss = model(x, y)
                tot += loss.item()
        return tot / 20

    base_levels = per_level_recovery()
    base_val = eval_val()
    print(f"\nBASE  val {base_val:.4f}   levels {base_levels}", flush=True)
    print("  (reference: d1 0.979  d3 0.836  root 0.088 -- endogenous_teacher / "
          "RHM_LATENT_LOOP)", flush=True)

    if save_ckpt and not loaded:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "fm_depth_cotrain": fm_depth_cotrain.state_dict(),
                    "config": {"v": v, "s": s, "L": L, "m": m, "n_layer": n_layer,
                               "n_head": n_head, "n_embd": n_embd,
                               "base_steps": base_steps, "seed": seed,
                               "rule_seed": rule_seed, "data_seed": data_seed}},
                   ckpt_path)
        volume.commit()
        print(f"  saved base checkpoint -> {ckpt_path}", flush=True)

    # ---------------- train the frozen-base FMs ----------------
    # One frozen forward pass feeds all three. Every activation below is produced under
    # no_grad, so stop-grad on the main model is structural, not a flag.
    fm_temporal = make_fm()          # primary: pred = fm(h6) - h6  -> Delta
    fm_temporal_direct = make_fm()   # control: pred = fm(h6)       -> Delta
    fm_depth_frozen = make_fm()      # protocol-matched depth control
    fms = {"temporal": fm_temporal, "temporal_direct": fm_temporal_direct,
           "depth_frozen": fm_depth_frozen}
    opts = {k: torch.optim.AdamW(f.parameters(), lr=fwd_lr, weight_decay=weight_decay)
            for k, f in fms.items()}

    def fm_predict(name, h_shallow, h_deep):
        """Prediction and target for each FM. Temporal ones are defined on t=0..T-2."""
        if name == "temporal":
            return (fm_temporal(h_deep) - h_deep)[:, :-1, :]
        if name == "temporal_direct":
            return fm_temporal_direct(h_deep)[:, :-1, :]
        if name == "depth_frozen":
            return fm_depth_frozen(h_shallow)
        raise ValueError(name)

    print(f"\n--- FMs: {fm_steps} steps against the FROZEN base ---", flush=True)
    gen_fm = torch.Generator().manual_seed(seed + 1000)
    for step in range(fm_steps):
        for f in fms.values():
            f.train()
        x, y = get_ntp_batch(gen_fm)
        with torch.no_grad():
            _, _, inter = model(x, y, return_intermediates=True)
            h_shallow, h_deep = inter[shallow_block], inter[deep_block]
            delta = h_deep[:, 1:, :] - h_deep[:, :-1, :]     # delta[:,t] = h6[t+1]-h6[t]
        losses = {}
        for name in fms:
            tgt = h_deep if name == "depth_frozen" else delta
            l = F.mse_loss(fm_predict(name, h_shallow, h_deep), tgt)
            opts[name].zero_grad(); l.backward(); opts[name].step()
            losses[name] = l.item()
        if step % log_interval == 0 or step == fm_steps - 1:
            print(f"  fm {step:6d}  " +
                  "  ".join(f"{k} {vv:.4f}" for k, vv in losses.items()), flush=True)

    for f in fms.values():
        f.eval()
    fm_depth_cotrain.eval()

    # ---------------- signals ----------------
    ALL_FMS = ["depth_cotrain", "depth_frozen", "temporal", "temporal_direct"]

    def signals(x, y):
        """Per-position nll and each FM's RELATIVE residual, all on t=0..T-2.

        Relative residual = ||target - pred|| / ||target||, matching
        endogenous_teacher's `rres` convention exactly so the numbers are comparable.
        The absolute residual norm is returned too, because ||Delta|| is not constant
        and the normalisation is a real choice.
        """
        with torch.no_grad():
            logits, _, inter = model(x, y, return_intermediates=True)
            nll = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                                  reduction="none").reshape(x.shape[0], T)
            h_shallow, h_deep = inter[shallow_block], inter[deep_block]
            delta = h_deep[:, 1:, :] - h_deep[:, :-1, :]
            rel, absn, cos = {}, {}, {}
            for name in ALL_FMS:
                if name == "depth_cotrain":
                    pred, tgt = fm_depth_cotrain(h_shallow), h_deep
                elif name == "depth_frozen":
                    pred, tgt = fm_depth_frozen(h_shallow), h_deep
                else:
                    pred, tgt = fm_predict(name, h_shallow, h_deep), delta
                if tgt.shape[1] == T:            # depth: trim to the shared t range
                    pred, tgt = pred[:, :-1, :], tgt[:, :-1, :]
                r = (tgt - pred).norm(dim=-1)
                rel[name] = r / (tgt.norm(dim=-1) + 1e-6)
                absn[name] = r
                cos[name] = F.cosine_similarity(pred, tgt, dim=-1)
        return nll[:, :T - 1], rel, absn, cos

    # ---------------- GATE 0: flat concatenated windows ----------------
    print(f"\n{'=' * 78}\nGATE 0 (a) -- flat concatenated windows (the training "
          f"distribution)\n{'=' * 78}", flush=True)
    g0 = torch.Generator().manual_seed(eval_seed + 21)
    acc_nll, acc_rel, acc_abs, acc_cos = [], {k: [] for k in ALL_FMS}, \
        {k: [] for k in ALL_FMS}, {k: [] for k in ALL_FMS}
    for _ in range(gate0_batches):
        x, y = get_ntp_batch(g0)
        nll, rel, absn, cos = signals(x, y)
        acc_nll.append(nll)
        for k in ALL_FMS:
            acc_rel[k].append(rel[k]); acc_abs[k].append(absn[k]); acc_cos[k].append(cos[k])
    NLL = torch.cat(acc_nll)
    REL = {k: torch.cat(vs) for k, vs in acc_rel.items()}
    ABS = {k: torch.cat(vs) for k, vs in acc_abs.items()}
    COS = {k: torch.cat(vs) for k, vs in acc_cos.items()}

    flat = {}
    for k in ALL_FMS:
        d = _gate0_stats(REL[k], NLL)
        d["abs_residual"] = _gate0_stats(ABS[k], NLL)
        d["mean_rel_residual"] = float(REL[k].mean())
        d["mean_abs_residual"] = float(ABS[k].mean())
        d["mean_cos"] = float(COS[k].mean())
        flat[k] = d
    print(f"  {'fm':<16}{'corr':>9}{'R2 pool':>10}{'R2 within':>11}{'orth%':>8}"
          f"{'rel_res':>9}{'cos':>8}")
    for k in ALL_FMS:
        d = flat[k]
        print(f"  {k:<16}{d['corr_pooled']:>+9.4f}{d['r2_pooled']:>10.4f}"
              f"{d['r2_within_position']:>11.4f}{100 * d['orth_variance_fraction']:>7.1f}%"
              f"{d['mean_rel_residual']:>9.4f}{d['mean_cos']:>8.4f}")
    print("  reference (endogenous_teacher depth FM): corr -0.336  R2 0.113 / 0.112  "
          "orth 88.7%", flush=True)

    # ---------------- GATE 0: aligned sequences, level profile ----------------
    print(f"\n{'=' * 78}\nGATE 0 (b) -- aligned sequences (hierarchy level is defined "
          f"per position)\n{'=' * 78}", flush=True)
    n_aligned = min(2048, eval_x.shape[0] // 256 * 256)
    a_nll = np.zeros(T - 1)
    a_rel = {k: np.zeros(T - 1) for k in ALL_FMS}
    a_abs = {k: np.zeros(T - 1) for k in ALL_FMS}
    al_nll, al_rel = [], {k: [] for k in ALL_FMS}
    n_chunk = 0
    for i in range(0, n_aligned, 256):
        xa = eval_x[i:i + 256]
        # last column's NTP target does not exist; signals() already drops t=T-1
        ya = torch.cat([xa[:, 1:], xa[:, :1]], dim=1)
        nll, rel, absn, _ = signals(xa, ya)
        a_nll += nll.mean(0).cpu().numpy()
        al_nll.append(nll)
        for k in ALL_FMS:
            a_rel[k] += rel[k].mean(0).cpu().numpy()
            a_abs[k] += absn[k].mean(0).cpu().numpy()
            al_rel[k].append(rel[k])
        n_chunk += 1
    a_nll /= n_chunk
    a_rel = {k: vv / n_chunk for k, vv in a_rel.items()}
    a_abs = {k: vv / n_chunk for k, vv in a_abs.items()}
    ANLL = torch.cat(al_nll)
    AREL = {k: torch.cat(vs) for k, vs in al_rel.items()}

    aligned_pooled = {k: _gate0_stats(AREL[k], ANLL) for k in ALL_FMS}
    print(f"  pooled over aligned positions:")
    print(f"  {'fm':<16}{'corr':>9}{'R2 pool':>10}{'R2 within':>11}{'orth%':>8}")
    for k in ALL_FMS:
        d = aligned_pooled[k]
        print(f"  {k:<16}{d['corr_pooled']:>+9.4f}{d['r2_pooled']:>10.4f}"
              f"{d['r2_within_position']:>11.4f}"
              f"{100 * d['orth_variance_fraction']:>7.1f}%")

    by_level = {}
    for ell in range(L + 1):
        mask = (pos_top_level[:T - 1] == ell)
        if not mask.sum():
            continue
        row = {"n_pos": int(mask.sum()), "mean_nll": float(a_nll[mask].mean())}
        for k in ALL_FMS:
            row[f"mean_rel_{k}"] = float(a_rel[k][mask].mean())
            row[f"mean_abs_{k}"] = float(a_abs[k][mask].mean())
        by_level[f"level{ell}"] = row

    lv_keys = [k for k in by_level]
    lv_nll = [by_level[k]["mean_nll"] for k in lv_keys]
    level_profile_corr = {k: _corr(lv_nll, [by_level[j][f"mean_rel_{k}"] for j in lv_keys])
                          for k in ALL_FMS}

    print(f"\n  {'level':<9}{'n':>4}{'mean_nll':>10}" +
          "".join(f"{k:>17}" for k in ALL_FMS))
    for lk in lv_keys:
        d = by_level[lk]
        print(f"  {lk:<9}{d['n_pos']:>4}{d['mean_nll']:>10.4f}" +
              "".join(f"{d[f'mean_rel_{k}']:>17.4f}" for k in ALL_FMS))
    print("\n  level-profile corr(mean_nll, mean_residual) across levels "
          "(negative = anti-localised):")
    for k in ALL_FMS:
        print(f"    {k:<16}{level_profile_corr[k]:>+8.3f}")
    print("  reference (endogenous_teacher depth FM level table): "
          "level1 nll 2.609 res 0.269 ... level6 nll 0.802 res 0.436", flush=True)

    # ---------------- verdict ----------------
    tf, ta = flat["temporal"], aligned_pooled["temporal"]
    if tf["r2_pooled"] > 0.9:
        branch = "KILL_SURPRISAL_RESTATED"
        reading = ("R^2(r_temporal ~ nll) > 0.9 -- the temporal residual is token "
                   "surprisal re-expressed in state space. SPEC's primary kill, cheap.")
    elif abs(tf["corr_pooled"]) < 0.10 and level_profile_corr["temporal"] < 0:
        branch = "KILL_AXIS_DID_NOTHING"
        reading = ("corr ~ 0 and the level profile is still anti-localised -- the "
                   "conditioning gap is not the operative variable. Falsifies the "
                   "reframe, not just the instrument.")
    else:
        branch = "INTERMEDIATE"
        reading = ("neither kill fired: a nontrivial relationship to nll with a "
                   "substantial orthogonal component remains. SPEC's proceed branch. "
                   "Check the SIGN and the level profile before calling it a pass -- "
                   "the registered prediction is corr clearly POSITIVE and the level "
                   "profile ALIGNED with nll.")
    matches_prediction = (tf["corr_pooled"] > 0 and level_profile_corr["temporal"] > 0)
    print(f"\n{'=' * 78}\nGATE 0 VERDICT (PROVISIONAL -- interpretation is Jasper's "
          f"call)\n{'=' * 78}")
    print(f"  branch: {branch}\n  {reading}")
    print(f"  registered prediction (corr>0 AND level profile aligned): "
          f"{'MET' if matches_prediction else 'NOT MET'}", flush=True)

    results = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "n_layer": n_layer, "n_head": n_head,
            "n_embd": n_embd, "shallow_block": shallow_block, "deep_block": deep_block,
            "fwd_n_layer": fwd_n_layer, "fwd_n_head": fwd_n_head,
            "fwd_d_head": fwd_d_head, "fwd_mlp_mult": fwd_mlp_mult,
            "base_steps": base_steps, "fm_steps": fm_steps, "batch_size": batch_size,
            "lr": lr, "fwd_lr": fwd_lr, "pool_size": pool_size, "data_seed": data_seed,
            "n_eval_sequences": n_eval_sequences, "eval_seed": eval_seed,
            "gate0_batches": gate0_batches, "seed": seed, "tag": tag,
            "base_ckpt_loaded": loaded,
        },
        "base": {"val": base_val, "levels": base_levels},
        "reference_depth_fm": {
            "source": "endogenous_teacher/README.md Gate 0",
            "corr": -0.336, "r2_pooled": 0.113, "r2_within_position": 0.112,
            "orth_variance_fraction": 0.887,
            "by_level_nll": [2.609, 2.611, 2.616, 2.523, 2.077, 0.802],
            "by_level_rres": [0.269, 0.259, 0.236, 0.319, 0.444, 0.436],
        },
        "flat_windows": flat,
        "aligned_pooled": aligned_pooled,
        "aligned_by_level": by_level,
        "level_profile_corr": level_profile_corr,
        "verdict": {"branch": branch, "reading": reading,
                    "registered_prediction_met": bool(matches_prediction),
                    "status": "PROVISIONAL"},
    }

    out_dir = f"{DATA_DIR}/{key}/conditional_revision"
    os.makedirs(out_dir, exist_ok=True)
    name = f"results{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
