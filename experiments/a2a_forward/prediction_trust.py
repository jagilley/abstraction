"""Prediction trust tests: does the model's use of the injection track the
forward model's per-direction reliability?

Two tests on the controlled-retrain checkpoints (no retraining):

Test A (Wiener gain analysis). The injection is additive: h <- h + G*p, so a
change of the prediction p along direction v moves the residual stream along
G*v. The model's per-direction "usage gain" therefore factorizes as
  S_full(v) = ||G v|| * S_down(G v / ||G v||)
where S_down is the downstream (blocks 2-3) output sensitivity per unit
h-perturbation. Both factors are learned. We compare these measured gains
against closed-form normative quantities derived from the forward model's
error statistics along each direction v:
  wiener gain  a*(v) = Cov(v.tgt, v.pred) / Var(v.pred)
  snr(v)            = Var(v.tgt) / Var(v.residual)
If the model encodes the forward model's reliability spectrum, measured gains
should correlate with the normative gains across directions.

Test B (counterfactual error injection / differential trust). Corrupt the
residual stream along the gate-image of (a) the forward model's habitual
error directions (top residual-covariance eigenvectors), (b) its most
trusted directions (bottom eigenvectors), (c) random directions, at matched
corruption norm in prediction space. Measure Delta-loss / KL / output
movement per direction set.

Hypotheses (we do not pre-commit to a sign):
- Wiener/discounting account: the model down-weights components of the
  prediction along directions where the forward model habitually errs.
  Usage gain correlates POSITIVELY with reliability; corruption along
  trusted directions moves the output MORE than along habitual-error
  directions.
- Error-monitoring account: downstream layers specialize in compensating
  along habitual-error directions, so sensitivity is HIGHER there.
  Opposite sign.
- Generic-landscape null: the per-direction profile is shared with the
  open-loop model (which never trained with the channel), i.e. it reflects
  loss-landscape anisotropy of any 4-layer LM, not trust.

The open-loop (OL) and injection-removed (CL-M) conditions are measured
along the SAME h-space directions (G v from the closed-loop gate), so any
CL-specific differential is attributable to training with the channel.

Reliability statistics are computed on the closed-loop forward pass WITH
injection active (the distribution the forward model was trained against).

Prior experiments: CONTROLLED_RETRAIN_README.md (checkpoints, probe gap),
JACOBIAN_ANALYSIS_README.md (sensitivity machinery),
DIRECTIONAL_STEER_README.md (directional self-knowledge, partial causal use).
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=10800,
    memory=32768,
)
def a2a_prediction_trust(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
    n_stat_batches: int = 40,
    n_jac_batches: int = 6,
    n_testb_batches: int = 6,
    jac_eps_scale: float = 0.05,
    jac_chunk: int = 8,
    n_set_dirs: int = 16,
    testb_scales: str = "0.5,1.0,2.0",
    n_perm: int = 2000,
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy import stats as sps
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))
    b_scales = [float(s) for s in testb_scales.split(",")]

    print(f"PREDICTION TRUST TESTS on {device}")
    print(f"  Predict: {predict_from} -> {predict_to}, "
          f"inject after block {inject_after_block}")
    print(f"  Test B scales: {b_scales}, dirs/set: {n_set_dirs}")

    # ==========================================================
    # Load data
    # ==========================================================
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = torch.from_numpy(
        np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]
    print(f"Loaded {len(data):,} tokens, {len(val_data):,} for eval")

    # ==========================================================
    # Load models (controlled retrain checkpoints)
    # ==========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    open_dir = os.path.join(ckpt_root, "open_loop")
    closed_dir = os.path.join(ckpt_root, "closed_loop")
    print(f"Loading from: {ckpt_root}")

    model_open = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_open.load_state_dict(torch.load(
        os.path.join(open_dir, "model.pt"),
        map_location=device, weights_only=True))

    model_closed = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "model.pt"),
        map_location=device, weights_only=True))

    fwd_closed = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "fwd_model.pt"),
        map_location=device, weights_only=True))

    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(closed_dir, "gate.pt"),
        map_location=device, weights_only=True))

    for m in [model_open, model_closed, fwd_closed, gate]:
        m.eval()
    W_G = gate.projection.weight.data  # (D, D); bias cancels in perturbations
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    # ==========================================================
    # Batch helpers and forward helpers
    # ==========================================================
    eval_gen = torch.Generator().manual_seed(seed + 700)
    eval_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=eval_gen)
        for _ in range(n_stat_batches)
    ]

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    def compute_h_base(model, x, use_injection):
        """Activation entering block inject_after_block+1 (post any injection).

        Returns (h, pred): pred is the forward model's prediction if
        use_injection, else None.
        """
        with torch.no_grad():
            tok_emb = model.transformer.wte(x)
            pos = torch.arange(0, x.size(1), device=x.device)
            pos_emb = model.transformer.wpe(pos)
            h = model.transformer.drop(tok_emb + pos_emb)

            cb_src, pred = None, None
            for i in range(inject_after_block + 1):
                h = model.transformer.h[i](h)
                if use_injection and i == cerebellar_input_block:
                    cb_src = h
            if use_injection and cb_src is not None:
                pred = fwd_closed(cb_src)
                h = h + gate(pred)
        return h, pred

    predict_to_block = int(predict_to.replace("post_block", ""))

    def continue_to_target(model, h):
        """Run blocks inject+1 .. predict_to_block from h, return post_blockN."""
        with torch.no_grad():
            for i in range(inject_after_block + 1, predict_to_block + 1):
                h = model.transformer.h[i](h)
        return h

    def lnf_from_h(model, h):
        """Run remaining blocks + ln_f from h. (Pre-lm_head representation.)"""
        for i in range(inject_after_block + 1, n_layer):
            h = model.transformer.h[i](h)
        return model.transformer.ln_f(h)

    def logits_from_h(model, h):
        return model.lm_head(lnf_from_h(model, h))

    conditions = {
        "CL+M": (model_closed, True),
        "CL-M": (model_closed, False),
        "OL":   (model_open, False),
    }

    # Per-position logit-movement via M = W_head^T W_head (avoids
    # materializing two (B,T,V) logit tensors for a norm).
    M_head = {}
    for name, (model, _) in conditions.items():
        W = model.lm_head.weight.data  # (V, D)
        M_head[name] = W.t() @ W       # (D, D)

    with torch.no_grad():
        x_ref, y_ref = make_batch(eval_indices[0])
        h_ref, _ = compute_h_base(model_open, x_ref, False)
        b1_std = float(h_ref.std())
    print(f"OL post_block1 activation std: {b1_std:.4f}")

    # ==========================================================
    # PHASE 1: Reliability statistics (closed loop, injection active)
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Forward model error statistics (with injection)")
    print(f"{'='*60}")

    preds, tgts = [], []
    with torch.no_grad():
        for idx in eval_indices:
            x, y = make_batch(idx)
            h_inj, pred = compute_h_base(model_closed, x, True)
            tgt = continue_to_target(model_closed, h_inj)
            preds.append(pred.reshape(-1, n_embd).cpu())
            tgts.append(tgt.reshape(-1, n_embd).cpu())
    P = torch.cat(preds).numpy().astype(np.float64)
    T_ = torch.cat(tgts).numpy().astype(np.float64)
    R = T_ - P
    n_samp = P.shape[0]
    pred_std = float(P.std())
    res_rms = float(np.sqrt((R ** 2).sum(axis=1).mean()))
    cos_pt = float(np.mean(
        (P * T_).sum(1) / (np.linalg.norm(P, axis=1)
                           * np.linalg.norm(T_, axis=1) + 1e-9)))
    print(f"  {n_samp:,} positions | pred entry std {pred_std:.4f} | "
          f"residual norm {res_rms:.4f} | pred-tgt cosine {cos_pt:.4f}")

    P_c = P - P.mean(axis=0, keepdims=True)
    T_c = T_ - T_.mean(axis=0, keepdims=True)
    R_c = R - R.mean(axis=0, keepdims=True)

    cov_R = (R_c.T @ R_c) / (n_samp - 1)
    eigval_r, eigvec_r = np.linalg.eigh(cov_R)
    eigval_r, eigvec_r = eigval_r[::-1].copy(), eigvec_r[:, ::-1].copy()

    rng = np.random.default_rng(seed + 41)
    Q_rand, _ = np.linalg.qr(rng.standard_normal((n_embd, n_embd)))

    bases = {"res_eig": eigvec_r, "random": Q_rand}

    def normative_stats(V):
        """Per-direction stats for columns of V (D, K)."""
        pv = P_c @ V   # (N, K)
        tv = T_c @ V
        rv = R_c @ V
        var_p = pv.var(axis=0)
        var_t = tv.var(axis=0)
        var_r = rv.var(axis=0)
        cov_tp = (tv * pv).mean(axis=0)
        wiener = cov_tp / (var_p + 1e-12)
        corr_tp = cov_tp / (np.sqrt(var_t * var_p) + 1e-12)
        snr = var_t / (var_r + 1e-12)
        return {
            "var_pred": var_p, "var_tgt": var_t, "var_res": var_r,
            "cov_tp": cov_tp, "wiener": wiener,
            "corr_tp": corr_tp, "rel": corr_tp ** 2, "snr": snr,
        }

    norm_stats = {bn: normative_stats(V) for bn, V in bases.items()}

    for bn in bases:
        ns = norm_stats[bn]
        print(f"\n  [{bn}] per-direction spread:")
        print(f"    var_res:  {ns['var_res'].min():.5f} .. "
              f"{ns['var_res'].max():.5f} "
              f"(ratio {ns['var_res'].max()/max(ns['var_res'].min(),1e-12):.1f}x)")
        print(f"    snr:      {ns['snr'].min():.2f} .. {ns['snr'].max():.2f}")
        print(f"    wiener:   {ns['wiener'].min():.3f} .. "
              f"{ns['wiener'].max():.3f}")
        print(f"    corr_tp:  {ns['corr_tp'].min():.3f} .. "
              f"{ns['corr_tp'].max():.3f}")

    # Gate gains per direction
    W_G_np = W_G.cpu().numpy().astype(np.float64)
    gate_gain = {bn: np.linalg.norm(W_G_np @ V, axis=0)
                 for bn, V in bases.items()}
    for bn in bases:
        g = gate_gain[bn]
        print(f"  [{bn}] gate gain ||Gv||: {g.min():.4f} .. {g.max():.4f} "
              f"(ratio {g.max()/max(g.min(),1e-12):.1f}x)")

    # ==========================================================
    # PHASE 2: Measured usage gains (directional Jacobian, central diff)
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Measured usage gains at the injection point")
    print(f"{'='*60}")
    eps_jac = jac_eps_scale * b1_std
    print(f"  eps = {jac_eps_scale} x b1_std = {eps_jac:.5f}, "
          f"{n_jac_batches} batches, chunk {jac_chunk}")

    # Unit h-space directions: gate images of each basis direction
    u_hat = {}      # (K, D) unit dirs on device
    for bn, V in bases.items():
        img = (W_G_np @ V).T  # (K, D)
        img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-12)
        u_hat[bn] = torch.from_numpy(img.astype(np.float32)).to(device)

    def movement_for_dirs(model, cond_name, h_base, dirs, eps):
        """RMS logit movement per unit eps for each unit dir. dirs: (K, D)."""
        Mh = M_head[cond_name]
        K = dirs.shape[0]
        B, T, D = h_base.shape
        out = torch.zeros(K, device=device)
        with torch.no_grad():
            for c0 in range(0, K, jac_chunk):
                d_c = dirs[c0:c0 + jac_chunk]          # (k, D)
                k = d_c.shape[0]
                shift = eps * d_c.view(k, 1, 1, D)
                h_p = (h_base.unsqueeze(0) + shift).reshape(k * B, T, D)
                h_m = (h_base.unsqueeze(0) - shift).reshape(k * B, T, D)
                x_p = lnf_from_h(model, h_p)
                x_m = lnf_from_h(model, h_m)
                dx = (x_p - x_m).reshape(k, B, T, D)
                mov2 = torch.einsum("kbtd,de,kbte->kbt", dx, Mh, dx)
                out[c0:c0 + k] = mov2.clamp_min(0).sqrt().mean(dim=(1, 2))
        return out / (2 * eps)

    s_down = {cn: {bn: np.zeros(n_embd) for bn in bases}
              for cn in conditions}
    for cn, (model, use_inj) in conditions.items():
        for bi in range(n_jac_batches):
            x, y = make_batch(eval_indices[bi])
            h_base, _ = compute_h_base(model, x, use_inj)
            for bn in bases:
                mv = movement_for_dirs(model, cn, h_base, u_hat[bn], eps_jac)
                s_down[cn][bn] += mv.cpu().numpy() / n_jac_batches
        print(f"  [{cn}] S_down measured "
              f"(res_eig mean {s_down[cn]['res_eig'].mean():.4f})")

    # Linearity check on a subset
    x, y = make_batch(eval_indices[0])
    h_base, _ = compute_h_base(model_closed, x, True)
    mv_full = movement_for_dirs(
        model_closed, "CL+M", h_base, u_hat["res_eig"][:8], eps_jac)
    mv_half = movement_for_dirs(
        model_closed, "CL+M", h_base, u_hat["res_eig"][:8], eps_jac / 2)
    lin_dev = float((mv_full - mv_half).abs().max()
                    / (mv_half.abs().mean() + 1e-12))
    print(f"  Linearity check (eps vs eps/2): max rel deviation {lin_dev:.4f}")

    s_full = {cn: {bn: s_down[cn][bn] * gate_gain[bn] for bn in bases}
              for cn in conditions}

    # ==========================================================
    # PHASE 3: Gain-vs-reliability correlations
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Do measured gains track normative gains?")
    print(f"{'='*60}")

    def partial_corr(a, b, controls):
        """Pearson corr of a, b after residualizing both on controls."""
        X = np.column_stack([np.ones_like(a)] + controls)
        ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
        rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
        return float(np.corrcoef(ra, rb)[0, 1])

    def perm_p(a, b, n=n_perm):
        obs = abs(np.corrcoef(a, b)[0, 1])
        prng = np.random.default_rng(seed + 17)
        cnt = 0
        for _ in range(n):
            if abs(np.corrcoef(a, prng.permutation(b))[0, 1]) >= obs:
                cnt += 1
        return (cnt + 1) / (n + 1)

    corr_results = {}
    for bn in bases:
        ns = norm_stats[bn]
        log_snr = np.log(ns["snr"] + 1e-12)
        corr_results[bn] = {}

        print(f"\n  === basis: {bn} ===")
        # Gate-level
        g = gate_gain[bn]
        gate_block = {
            "pearson_gate_wiener": float(np.corrcoef(g, ns["wiener"])[0, 1]),
            "spearman_gate_wiener": float(sps.spearmanr(g, ns["wiener"])[0]),
            "pearson_gate_logsnr": float(np.corrcoef(g, log_snr)[0, 1]),
            "pearson_gate_varres": float(np.corrcoef(g, ns["var_res"])[0, 1]),
            "pearson_gate_varpred": float(np.corrcoef(g, ns["var_pred"])[0, 1]),
            "partial_gate_wiener_ctrl_varpred_vartgt": partial_corr(
                g, ns["wiener"], [ns["var_pred"], ns["var_tgt"]]),
            "perm_p_gate_wiener": perm_p(g, ns["wiener"]),
        }
        corr_results[bn]["gate"] = gate_block
        print(f"  gate ||Gv||  vs wiener: r={gate_block['pearson_gate_wiener']:+.3f} "
              f"(rho={gate_block['spearman_gate_wiener']:+.3f}, "
              f"perm p={gate_block['perm_p_gate_wiener']:.4f})")
        print(f"               vs log snr: "
              f"r={gate_block['pearson_gate_logsnr']:+.3f} | "
              f"vs var_res: r={gate_block['pearson_gate_varres']:+.3f} | "
              f"partial(wiener | var_pred, var_tgt): "
              f"{gate_block['partial_gate_wiener_ctrl_varpred_vartgt']:+.3f}")

        # System-level, per condition
        for cn in conditions:
            sd, sf = s_down[cn][bn], s_full[cn][bn]
            blk = {
                "pearson_sfull_wiener": float(
                    np.corrcoef(sf, ns["wiener"])[0, 1]),
                "spearman_sfull_wiener": float(sps.spearmanr(sf, ns["wiener"])[0]),
                "pearson_sfull_logsnr": float(np.corrcoef(sf, log_snr)[0, 1]),
                "pearson_sdown_logsnr": float(np.corrcoef(sd, log_snr)[0, 1]),
                "pearson_sdown_varres": float(
                    np.corrcoef(sd, ns["var_res"])[0, 1]),
                "spearman_sdown_varres": float(
                    sps.spearmanr(sd, ns["var_res"])[0]),
                "partial_sfull_wiener_ctrl_varpred_vartgt": partial_corr(
                    sf, ns["wiener"], [ns["var_pred"], ns["var_tgt"]]),
                "perm_p_sdown_varres": perm_p(sd, ns["var_res"]),
            }
            corr_results[bn][cn] = blk
            print(f"  [{cn:>4s}] S_full vs wiener: "
                  f"r={blk['pearson_sfull_wiener']:+.3f} "
                  f"(rho={blk['spearman_sfull_wiener']:+.3f}) | "
                  f"S_down vs var_res: r={blk['pearson_sdown_varres']:+.3f} "
                  f"(rho={blk['spearman_sdown_varres']:+.3f}, "
                  f"p={blk['perm_p_sdown_varres']:.4f})")

        # CL-specific differential: does CL's S_down profile differ from OL's
        # in a way aligned with reliability?
        d_prof = s_down["CL+M"][bn] / (s_down["OL"][bn] + 1e-12)
        diff_block = {
            "pearson_clOLratio_varres": float(
                np.corrcoef(d_prof, ns["var_res"])[0, 1]),
            "spearman_clOLratio_varres": float(
                sps.spearmanr(d_prof, ns["var_res"])[0]),
            "pearson_clOLratio_logsnr": float(
                np.corrcoef(d_prof, log_snr)[0, 1]),
            "perm_p_clOLratio_varres": perm_p(d_prof, ns["var_res"]),
            "pearson_sdown_CL_OL": float(
                np.corrcoef(s_down["CL+M"][bn], s_down["OL"][bn])[0, 1]),
        }
        corr_results[bn]["cl_vs_ol"] = diff_block
        print(f"  S_down(CL+M)/S_down(OL) vs var_res: "
              f"r={diff_block['pearson_clOLratio_varres']:+.3f} "
              f"(rho={diff_block['spearman_clOLratio_varres']:+.3f}, "
              f"p={diff_block['perm_p_clOLratio_varres']:.4f}); "
              f"corr(S_down CL, OL)={diff_block['pearson_sdown_CL_OL']:+.3f}")

    # ==========================================================
    # PHASE 4: Test B — counterfactual error injection
    # ==========================================================
    print(f"\n{'='*60}")
    print("PHASE 4: Counterfactual error injection (differential trust)")
    print(f"{'='*60}")

    V_res = eigvec_r
    rand16 = rng.standard_normal((n_embd, n_set_dirs))
    rand16 /= np.linalg.norm(rand16, axis=0, keepdims=True)
    dir_sets = {
        "error_top": V_res[:, :n_set_dirs],
        "trusted_bottom": V_res[:, -n_set_dirs:],
        "random": rand16,
    }
    set_info = {}
    for sn, V in dir_sets.items():
        img = W_G_np @ V                       # (D, K)
        ns_set = normative_stats(V)
        set_info[sn] = {
            "mean_gate_gain": float(np.linalg.norm(img, axis=0).mean()),
            "mean_var_res": float(ns_set["var_res"].mean()),
            "mean_snr": float(ns_set["snr"].mean()),
            "mean_wiener": float(ns_set["wiener"].mean()),
        }
        print(f"  [{sn:>14s}] mean ||Gd||={set_info[sn]['mean_gate_gain']:.4f}, "
              f"var_res={set_info[sn]['mean_var_res']:.5f}, "
              f"snr={set_info[sn]['mean_snr']:.1f}, "
              f"wiener={set_info[sn]['mean_wiener']:.3f}")

    dir_imgs = {sn: torch.from_numpy((W_G_np @ V).T.astype(np.float32)).to(device)
                for sn, V in dir_sets.items()}   # (K, D) h-space, unnormalized

    testb = {cn: {sn: {s: {"dloss": [], "kl": []} for s in b_scales}
                  for sn in dir_sets} for cn in conditions}

    with torch.no_grad():
        for cn, (model, use_inj) in conditions.items():
            for bi in range(n_testb_batches):
                x, y = make_batch(eval_indices[bi])
                h_base, _ = compute_h_base(model, x, use_inj)
                logits_b = logits_from_h(model, h_base)
                loss_b = F.cross_entropy(
                    logits_b.view(-1, logits_b.size(-1)), y.view(-1)).item()
                logp_b = F.log_softmax(logits_b, dim=-1)
                p_b = logp_b.exp()

                for sn, imgs in dir_imgs.items():
                    for di in range(imgs.shape[0]):
                        u = imgs[di]
                        for s in b_scales:
                            eps = s * pred_std
                            h_pert = h_base + eps * u.view(1, 1, -1)
                            logits_p = logits_from_h(model, h_pert)
                            loss_p = F.cross_entropy(
                                logits_p.view(-1, logits_p.size(-1)),
                                y.view(-1)).item()
                            logp_p = F.log_softmax(logits_p, dim=-1)
                            kl = (p_b * (logp_b - logp_p)).sum(-1).mean().item()
                            testb[cn][sn][s]["dloss"].append(loss_p - loss_b)
                            testb[cn][sn][s]["kl"].append(kl)
            print(f"  [{cn}] test B complete")

    testb_summary = {}
    print(f"\n  {'cond':>6s} {'scale':>6s} {'set':>15s} "
          f"{'dloss':>10s} {'KL':>10s} {'KL/||Gd||^2':>12s}")
    for cn in conditions:
        testb_summary[cn] = {}
        for s in b_scales:
            testb_summary[cn][str(s)] = {}
            for sn in dir_sets:
                dl = float(np.mean(testb[cn][sn][s]["dloss"]))
                kl = float(np.mean(testb[cn][sn][s]["kl"]))
                g2 = set_info[sn]["mean_gate_gain"] ** 2
                testb_summary[cn][str(s)][sn] = {
                    "dloss": dl, "kl": kl, "kl_per_gate2": kl / g2,
                    "dloss_per_dir": [
                        float(v) for v in np.array(
                            testb[cn][sn][s]["dloss"]).reshape(
                                n_testb_batches, -1).mean(axis=0)],
                    "kl_per_dir": [
                        float(v) for v in np.array(
                            testb[cn][sn][s]["kl"]).reshape(
                                n_testb_batches, -1).mean(axis=0)],
                }
                print(f"  {cn:>6s} {s:>6.2f} {sn:>15s} "
                      f"{dl:>+10.5f} {kl:>10.5f} {kl/g2:>12.5f}")

    print(f"\n  --- Differential trust ratios (KL trusted_bottom / error_top) ---")
    ratio_summary = {}
    for cn in conditions:
        ratio_summary[cn] = {}
        for s in b_scales:
            kt = testb_summary[cn][str(s)]["trusted_bottom"]["kl"]
            ke = testb_summary[cn][str(s)]["error_top"]["kl"]
            ratio_summary[cn][str(s)] = kt / (ke + 1e-12)
            print(f"  {cn:>6s} s={s:<4.2f}: {kt/(ke+1e-12):.3f}")
    print("  (>1: output moves more along trusted dirs — discounting account;")
    print("   <1: moves more along habitual-error dirs — monitoring account;")
    print("   CL ratio ~= OL ratio: generic landscape, not trust)")

    # ==========================================================
    # SUMMARY
    # ==========================================================
    print(f"\n{'='*60}")
    print("PREDICTION TRUST SUMMARY")
    print(f"{'='*60}")
    bn = "res_eig"
    ns = norm_stats[bn]
    print(f"\n1. Gate alignment (residual eigenbasis):")
    print(f"   corr(||Gv||, wiener gain) = "
          f"{corr_results[bn]['gate']['pearson_gate_wiener']:+.3f} "
          f"(perm p={corr_results[bn]['gate']['perm_p_gate_wiener']:.4f})")
    print(f"\n2. System usage gains:")
    for cn in conditions:
        print(f"   [{cn:>4s}] corr(S_down, var_res) = "
              f"{corr_results[bn][cn]['pearson_sdown_varres']:+.3f}")
    print(f"   CL/OL profile ratio vs var_res: "
          f"{corr_results[bn]['cl_vs_ol']['pearson_clOLratio_varres']:+.3f}")
    print(f"\n3. Differential trust (KL trusted/error at s=1.0):")
    for cn in conditions:
        print(f"   [{cn:>4s}] {ratio_summary[cn].get('1.0', float('nan')):.3f}")

    # ==========================================================
    # Save
    # ==========================================================
    save_dir = os.path.join(ckpt_root, "prediction_trust")
    os.makedirs(save_dir, exist_ok=True)
    from a2a_forward.shared import NumpyEncoder
    output = {
        "config": {
            "n_tokens": n_tokens, "seed": seed,
            "n_stat_batches": n_stat_batches,
            "n_jac_batches": n_jac_batches,
            "n_testb_batches": n_testb_batches,
            "jac_eps_scale": jac_eps_scale, "jac_eps": eps_jac,
            "linearity_max_rel_dev": lin_dev,
            "n_set_dirs": n_set_dirs, "testb_scales": b_scales,
            "pred_std": pred_std, "residual_rms": res_rms,
            "pred_tgt_cosine": cos_pt,
            "gate_norm": gate.injection_norm(),
            "b1_std": b1_std,
        },
        "normative_stats": {
            bn: {k: v.tolist() for k, v in ns_.items()}
            for bn, ns_ in norm_stats.items()
        },
        "residual_eigvals": eigval_r.tolist(),
        "gate_gain": {bn: g.tolist() for bn, g in gate_gain.items()},
        "s_down": {cn: {bn: s_down[cn][bn].tolist() for bn in bases}
                   for cn in conditions},
        "s_full": {cn: {bn: s_full[cn][bn].tolist() for bn in bases}
                   for cn in conditions},
        "correlations": corr_results,
        "testb_set_info": set_info,
        "testb_summary": testb_summary,
        "testb_trust_ratios": ratio_summary,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nResults saved to {save_dir}")
    return output


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
):
    result = a2a_prediction_trust.remote(
        n_tokens=n_tokens,
        predict_from=predict_from,
        predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head,
        fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
    )
    print("\nPrediction trust tests complete.")
    bn = "res_eig"
    c = result["correlations"][bn]
    print(f"  corr(||Gv||, wiener) = {c['gate']['pearson_gate_wiener']:+.3f}")
    for cn in ["CL+M", "CL-M", "OL"]:
        print(f"  [{cn}] corr(S_down, var_res) = "
              f"{c[cn]['pearson_sdown_varres']:+.3f}")
    print(f"  CL/OL ratio vs var_res = "
          f"{c['cl_vs_ol']['pearson_clOLratio_varres']:+.3f}")
    for cn, ratios in result["testb_trust_ratios"].items():
        print(f"  [{cn}] KL trusted/error ratios: {ratios}")
