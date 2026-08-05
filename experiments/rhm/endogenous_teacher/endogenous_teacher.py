"""The endogenous teaching signal: can a model's own surprise steer its learning?

In NTP the text is stimulus, target, and teaching signal at once. In a brain the word
is only the stimulus; the teaching signal is generated internally. This cut asks the
smallest measurable version of that: holding the exogenous signal (token surprisal)
fixed, does an endogenously-generated one (the FM residual on the model's own
activations) change WHAT the model learns?

Design (see PREREGISTRATION.md):
  base = plain-NTP RHM model + open-loop co-trained FM (post_block0 -> post_block6)
  arms = continued training from the shared base, differing ONLY in how the
         per-position NTP loss is weighted:

    uniform      w = 1
    nll          w ordered by per-position NTP loss     (EXOGENOUS teacher)
    res          w ordered by relative FM residual      (ENDOGENOUS teacher)
    res_orth     w ordered by residual-after-regressing-out-nll  (the sharp arm)
    res_shuffled res weights, permuted                  (the control)

  All arms use RANK-NORMALISED weights (w = 2*rank/(N-1), mean 1), so every arm has a
  bit-identical weight multiset -- only the assignment differs. Weight scale, variance
  and effective lr are matched by construction.

Gate 0 (kill criterion, runs before any arm): R^2(rres ~ nll) on the training
distribution. > 0.9 -> the endogenous signal is a restatement of the exogenous one and
the question is empty; < 0.5 -> proceed.

Primary readout is the DEPTH PROFILE (per-level ancestor recovery d1..d6), not val loss:
the registered prediction is that the exogenous teacher concentrates learning at the
shallow levels (deep levels are vm^(l+2)-diluted and invisible to token loss) while the
endogenous one shifts it up, because the residual maps the model's frontier rather than
the corpus's surface.

Regime matches RHM_LATENT_LOOP exactly (v16 s2 L6 m4, 8L/8H/256D, FM 1L/8H/16d) so its
reference lines transfer: BP root 0.80; token-NTP frontier d1 0.98 d3 0.88 d4 0.51
root 0.08; greedy floor d3 0.751 d4 0.708.

Run:
  # gate 0 only (kill check) -- cheap
  modal run -m rhm.endogenous_teacher.endogenous_teacher::endogenous_teacher \
      --arms "" --base-steps 2000 --tag gate0
  # full cut
  modal run --detach -m rhm.endogenous_teacher.endogenous_teacher::endogenous_teacher
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
app = modal.App("rhm-endogenous-teacher", image=image)

ARMS = ["uniform", "nll", "res", "res_orth", "res_shuffled"]


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def _rank_weights(score, gen=None, shuffle=False):
    """Rank-normalise a flat score tensor to weights in [0,2] with mean ~1.

    Identical multiset for every arm -> weight scale/variance/effective-lr matched by
    construction; only the ASSIGNMENT to positions differs. `shuffle` permutes the
    resulting weights (the res_shuffled control).
    """
    import torch
    n = score.numel()
    order = torch.argsort(score.reshape(-1))
    w = torch.empty(n, device=score.device)
    ranks = torch.arange(n, device=score.device, dtype=score.dtype)
    w[order] = 2.0 * ranks / max(n - 1, 1)
    if shuffle:
        w = w[torch.randperm(n, generator=gen, device=score.device)]
    return w


def _orthogonalise(y, x):
    """Residual of y after per-batch OLS regression on x (both flat, same device)."""
    import torch
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum() + 1e-8
    beta = (xc * yc).sum() / denom
    return yc - beta * xc


def _r2(y, x):
    """R^2 of predicting y from x by simple linear regression (flat tensors)."""
    import torch
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum() + 1e-8
    beta = (xc * yc).sum() / denom
    pred = beta * xc
    ss_res = ((yc - pred) ** 2).sum()
    ss_tot = (yc * yc).sum() + 1e-8
    return float(1.0 - ss_res / ss_tot)


def _participation_ratio(X):
    """PR = (sum s_i)^2 / sum s_i^2 over the eigenvalues of the covariance."""
    import torch
    Xc = X - X.mean(0, keepdim=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    return float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12))


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def endogenous_teacher(
    # DGP -- RHM_LATENT_LOOP's regime (real learnable-but-unlearned frontier)
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    # Model
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # FM (open-loop: it watches, it does not inject)
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0,
    # Training
    base_steps: int = 8000, arm_steps: int = 6000,
    batch_size: int = 64, lr: float = 3e-4, fwd_lr: float = 1e-3,
    weight_decay: float = 0.01,
    pool_size: int = 200000, data_seed: int = 7,
    arms: str = "uniform,nll,res,res_orth,res_shuffled",
    # Measurement
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    log_interval: int = 1000, gate0_batches: int = 20,
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
    cib = int(predict_from.replace("post_block", ""))
    cob = predict_to
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    for a in arm_list:
        assert a in ARMS, f"unknown arm {a!r}; known: {ARMS}"
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    print(f"{'=' * 74}\nENDOGENOUS TEACHER  {key}  {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  occupancy m/v^(s-1)={m / v ** (s - 1):.3f}  chance={chance:.4f}  T={T}")
    print(f"  FM {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d  {predict_from}->{predict_to} (open loop)")
    print(f"  base_steps={base_steps}  arm_steps={arm_steps}  arms={arm_list}")
    print(f"{'=' * 74}", flush=True)

    # ---------------- data ----------------
    print(f"Generating pool ({pool_size:,} seqs)...", flush=True)
    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))
    corpus = pool_x.reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    # Flat concatenated windows: EVERY position is NTP-supervised (the phase-diversity
    # fix from RHM_LATENT_LOOP), and position carries no fixed hierarchy level -- which
    # also removes the position<->level confound from the per-position weighting.
    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    # Aligned eval set: hierarchy level IS defined per position here, so this is where
    # per-level probes and the level-attribution half of Gate 0 live.
    eval_seqs, eval_lf, _ = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    y_level = {ell: torch.from_numpy(eval_lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
               for ell in range(L)}
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    # position -> the HIGHEST hierarchy node completed there (smallest ell = closest to
    # root). A level-ell node spans s^(L-ell) leaves, so p completes it when
    # (p+1) % s^(L-ell) == 0; ell=L (span 1) holds everywhere, so the min is defined.
    pos_top_level = np.array(
        [min(ell for ell in range(L + 1) if (p + 1) % (s ** (L - ell)) == 0)
         for p in range(T)], dtype=np.int64)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    # ---------------- per-position signals ----------------
    def signals(model, fm, x, y):
        """(nll_pp, rres_pp, h6) for a batch -- the two teaching signals, detached."""
        logits, _, inter = model(x, y, return_intermediates=True)
        nll_pp = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1), reduction="none")
        h_from, h_to = inter[predict_from], inter[cob]
        pred = fm(h_from.detach())
        resid = h_to.detach() - pred
        rres = (resid.norm(dim=-1) / (h_to.detach().norm(dim=-1) + 1e-6)).detach()
        return nll_pp, rres.reshape(-1), pred, h_to

    def arm_weights(arm, nll_pp, rres, gen):
        if arm == "uniform":
            return torch.ones_like(nll_pp)
        if arm == "nll":
            return _rank_weights(nll_pp.detach())
        if arm == "res":
            return _rank_weights(rres)
        if arm == "res_shuffled":
            return _rank_weights(rres, gen=gen, shuffle=True)
        if arm == "res_orth":
            return _rank_weights(_orthogonalise(rres, nll_pp.detach()))
        raise ValueError(arm)

    # ---------------- measurement ----------------
    def per_level_recovery(model):
        model.eval()
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
            # Repo convention (rhm_latent_loop.py:174): ell=0 is the ROOT, and levels are
            # named d{L-ell}, so d1 is the shallowest (just above the leaves) and d6=root.
            out[f"d{L - ell}"] = best
        return out, acts[cob]

    def eval_val(model):
        model.eval()
        gen = torch.Generator().manual_seed(eval_seed + 5)
        tot = 0.0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_ntp_batch(gen)
                _, loss = model(x, y)
                tot += loss.item()
        return tot / 20

    def fm_health(model, fm):
        model.eval(); fm.eval()
        gen = torch.Generator().manual_seed(eval_seed + 11)
        cos, rr = [], []
        with torch.no_grad():
            for _ in range(10):
                x, y = get_ntp_batch(gen)
                _, _, inter = model(x, y, return_intermediates=True)
                h_from, h_to = inter[predict_from], inter[cob]
                pred = fm(h_from)
                cos.append(F.cosine_similarity(pred.reshape(-1, n_embd),
                                               h_to.reshape(-1, n_embd), dim=-1).mean().item())
                rr.append(((h_to - pred).norm(dim=-1) / (h_to.norm(dim=-1) + 1e-6)).mean().item())
        return float(np.mean(cos)), float(np.mean(rr))

    # ---------------- train the shared base ----------------
    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    fm = make_fm()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    print(f"\n--- BASE: {base_steps} steps, plain NTP + open-loop FM ---", flush=True)
    for step in range(base_steps):
        model.train(); fm.train()
        x, y = get_ntp_batch(gen)
        nll_pp, rres, pred, h_to = signals(model, fm, x, y)
        loss = nll_pp.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        fwd_loss = F.mse_loss(pred, h_to.detach())
        opt_fwd.zero_grad(); fwd_loss.backward(); opt_fwd.step()
        if step % log_interval == 0 or step == base_steps - 1:
            print(f"  base {step:6d}  ntp {loss.item():.4f}  fm_mse {fwd_loss.item():.4f}",
                  flush=True)

    base_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
    base_fm_state = {k: t.cpu().clone() for k, t in fm.state_dict().items()}
    base_levels, base_h6 = per_level_recovery(model)
    base_val = eval_val(model)
    base_cos, base_rres = fm_health(model, fm)
    base_pr = _participation_ratio(base_h6)
    print(f"\nBASE  val {base_val:.4f}  fwd_cos {base_cos:.4f}  rel_res {base_rres:.4f}  "
          f"PR {base_pr:.2f}")
    print(f"BASE  levels {base_levels}", flush=True)

    # ---------------- GATE 0: is the endogenous signal a restatement? ----------------
    print(f"\n{'=' * 74}\nGATE 0 -- R^2(rres ~ nll) on the TRAINING distribution\n{'=' * 74}",
          flush=True)
    model.eval(); fm.eval()
    g0 = torch.Generator().manual_seed(eval_seed + 21)
    all_nll, all_res, all_pos = [], [], []
    with torch.no_grad():
        for _ in range(gate0_batches):
            x, y = get_ntp_batch(g0)
            nll_pp, rres, _, _ = signals(model, fm, x, y)
            all_nll.append(nll_pp.reshape(batch_size, T))
            all_res.append(rres.reshape(batch_size, T))
    NLL = torch.cat(all_nll); RES = torch.cat(all_res)
    r2_pooled = _r2(RES.reshape(-1), NLL.reshape(-1))
    r2_within = float(np.mean([_r2(RES[:, t], NLL[:, t]) for t in range(T)]))
    # what fraction of residual variance survives orthogonalisation (the res_orth signal)
    orth_frac = float((_orthogonalise(RES.reshape(-1), NLL.reshape(-1)).var()
                       / (RES.reshape(-1).var() + 1e-12)))
    gate0 = {
        "r2_rres_from_nll_pooled": r2_pooled,
        "r2_rres_from_nll_within_position": r2_within,
        "orth_variance_fraction": orth_frac,
        "corr_pooled": float(np.corrcoef(RES.reshape(-1).cpu().numpy(),
                                         NLL.reshape(-1).cpu().numpy())[0, 1]),
        "verdict": ("KILL" if r2_pooled > 0.9 else
                    "PROCEED" if r2_pooled < 0.5 else "UNDERPOWERED"),
    }
    print(f"  R^2 pooled                 {r2_pooled:.4f}")
    print(f"  R^2 within-position (mean) {r2_within:.4f}")
    print(f"  orthogonal variance frac   {orth_frac:.4f}")
    print(f"  corr(rres, nll)            {gate0['corr_pooled']:+.4f}")
    print(f"  VERDICT: {gate0['verdict']}  "
          f"(>0.9 kill / <0.5 proceed)", flush=True)

    # Level attribution on ALIGNED sequences (position <-> level is defined there).
    # Chunked: full intermediates for 2048x64x256x9 would be ~1.2GB.
    # The last position's NTP target does not exist, so it is excluded throughout.
    nll_acc, rres_acc, n_chunk = np.zeros(T), np.zeros(T), 0
    n_aligned = min(2048, eval_x.shape[0] // 256 * 256)
    with torch.no_grad():
        for i in range(0, n_aligned, 256):
            xa = eval_x[i:i + 256]
            ya = torch.cat([xa[:, 1:], xa[:, :1]], dim=1)      # last col invalid, masked below
            nll_a, rres_a, _, _ = signals(model, fm, xa, ya)
            nll_acc += nll_a.reshape(-1, T).mean(0).cpu().numpy()
            rres_acc += rres_a.reshape(-1, T).mean(0).cpu().numpy()
            n_chunk += 1
    nll_a, rres_a = nll_acc / n_chunk, rres_acc / n_chunk
    valid = np.ones(T, dtype=bool); valid[T - 1] = False
    by_level = {}
    for ell in range(L + 1):
        mask = (pos_top_level == ell) & valid
        if mask.sum():
            by_level[f"level{ell}"] = {"n_pos": int(mask.sum()),
                                       "mean_nll": float(nll_a[mask].mean()),
                                       "mean_rres": float(rres_a[mask].mean())}
    gate0["aligned_by_level"] = by_level
    print("\n  aligned level attribution (mean over positions completing that level):")
    for k, d in by_level.items():
        print(f"    {k}: n={d['n_pos']:3d}  nll {d['mean_nll']:.4f}  rres {d['mean_rres']:.4f}")

    results = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "n_layer": n_layer, "n_head": n_head,
            "n_embd": n_embd, "predict_from": predict_from, "predict_to": predict_to,
            "base_steps": base_steps, "arm_steps": arm_steps, "batch_size": batch_size,
            "lr": lr, "fwd_lr": fwd_lr, "seed": seed, "arms": arm_list, "tag": tag,
        },
        "base": {"val": base_val, "levels": base_levels, "fwd_cos": base_cos,
                 "rel_res": base_rres, "pr": base_pr},
        "gate0": gate0,
        "arms": {},
    }

    if gate0["verdict"] == "KILL":
        print("\nGate 0 says KILL -- the endogenous signal restates the exogenous one. "
              "Skipping arms.", flush=True)
        arm_list = []

    # ---------------- the arms ----------------
    for arm in arm_list:
        print(f"\n{'=' * 74}\nARM {arm}  ({arm_steps} steps from shared base)\n{'=' * 74}",
              flush=True)
        model.load_state_dict({k: t.to(device) for k, t in base_state.items()})
        fm.load_state_dict({k: t.to(device) for k, t in base_fm_state.items()})
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(seed + 1000)
        wgen = torch.Generator(device=device).manual_seed(seed + 2000)
        traj = []
        for step in range(arm_steps):
            model.train(); fm.train()
            x, y = get_ntp_batch(gen)
            nll_pp, rres, pred, h_to = signals(model, fm, x, y)
            w = arm_weights(arm, nll_pp, rres, wgen).detach()
            loss = (w * nll_pp).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            fwd_loss = F.mse_loss(pred, h_to.detach())
            opt_fwd.zero_grad(); fwd_loss.backward(); opt_fwd.step()
            if step % log_interval == 0 or step == arm_steps - 1:
                traj.append({"step": step, "weighted": loss.item(),
                             "raw_ntp": nll_pp.mean().item(),
                             "mean_rres": rres.mean().item()})
                print(f"  {arm} {step:6d}  weighted {loss.item():.4f}  "
                      f"raw {nll_pp.mean().item():.4f}  rres {rres.mean().item():.4f}",
                      flush=True)
        lv, h6 = per_level_recovery(model)
        val = eval_val(model)
        cos, rr = fm_health(model, fm)
        pr = _participation_ratio(h6)
        results["arms"][arm] = {"val": val, "levels": lv, "fwd_cos": cos, "rel_res": rr,
                                "pr": pr, "traj": traj,
                                "delta_levels": {k: lv[k] - base_levels[k] for k in lv}}
        print(f"\n  {arm}: val {val:.4f} (base {base_val:.4f})  fwd_cos {cos:.4f}  PR {pr:.2f}")
        print(f"  {arm}: levels {lv}")
        print(f"  {arm}: Δlevels " +
              " ".join(f"{k} {lv[k] - base_levels[k]:+.3f}" for k in lv), flush=True)

    # ---------------- summary ----------------
    if results["arms"]:
        print(f"\n{'=' * 74}\nSUMMARY (Δ from shared base)\n{'=' * 74}")
        hdr = f"{'arm':<14}{'val':>9}" + "".join(f"{f'd{i+1}':>8}" for i in range(L)) + f"{'PR':>8}"
        print(hdr)
        print(f"{'base':<14}{base_val:>9.4f}" +
              "".join(f"{base_levels[f'd{i+1}']:>8.3f}" for i in range(L)) + f"{base_pr:>8.2f}")
        for arm, r in results["arms"].items():
            print(f"{arm:<14}{r['val']:>9.4f}" +
                  "".join(f"{r['levels'][f'd{i+1}']:>8.3f}" for i in range(L)) +
                  f"{r['pr']:>8.2f}")
        print("\nΔ vs base:")
        print(hdr)
        for arm, r in results["arms"].items():
            print(f"{arm:<14}{r['val'] - base_val:>+9.4f}" +
                  "".join(f"{r['delta_levels'][f'd{i+1}']:>+8.3f}" for i in range(L)) +
                  f"{r['pr'] - base_pr:>+8.2f}")

    out_dir = f"{DATA_DIR}/{key}/endogenous_teacher"
    os.makedirs(out_dir, exist_ok=True)
    name = f"results{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
