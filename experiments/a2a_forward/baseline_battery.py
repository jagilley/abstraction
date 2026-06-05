"""Baseline battery: is forward self-prediction uniquely useful?

Tests whether the benefits of forward model injection (LM improvement,
self-knowledge, perturbation robustness) are specific to injecting a
prediction of the model's own future computation, or whether any
structured injection would produce the same effects.

5 conditions, all with identical lr/seed/init/data order:
  1. open_loop:   no injection (baseline)
  2. forward:     inject gate(fwd_model(post_block0)) — the real thing
  3. shifted:     inject gate(shift(fwd_model(post_block0), k)) — causal
                  shift by k positions, destroying position-specific content
  4. random_proj: inject gate(frozen_random_proj(post_block0)) — learned
                  linear function of early activations
  5. autoencoder: inject gate(autoenc(post_block0)) — compressed
                  self-reconstruction through the same capacity bottleneck

All conditions co-train a forward model (post_block0 → post_block3) for
probing, so the self-knowledge metric is comparable across conditions.

After training, measures:
  - Validation LM loss (with and without injection)
  - Self-knowledge probes (R² for forward model's residual, across layers)
  - Perturbation robustness (response norm, loss degradation, cos(R,δ))
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "forward", "shifted", "random_proj", "autoencoder"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,  # 6 hours
    memory=32768,
)
def a2a_baseline_battery(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 10_000,
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    shift_k: int = 10,
    seed: int = 42,
    probe_batches: int = 40,
    probe_steps: int = 500,
    n_perturb_dirs: int = 16,
    perturb_s: float = 2.0,
    perturb_eval_batches: int = 20,
):
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1")
    )
    print(f"BASELINE BATTERY on {device}")
    print(f"  lr={lr}, seed={seed}, n_steps={n_steps}")
    print(f"  Forward model: {fwd_n_layer}L, {predict_from} → {predict_to}")
    print(f"  Injection: after block {inject_after_block}")
    print(f"  Shift k={shift_k} for shifted baseline")
    print(f"  Conditions: {CONDITIONS}")

    # =========================================================
    # Load data
    # =========================================================
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
    data = np.concatenate(all_tokens)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    print(f"Loaded {len(data):,} tokens (vocab_size={vocab_size})")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # =========================================================
    # Pre-generate ALL batch indices (shared across conditions)
    # =========================================================
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)

    n_evals = n_steps // eval_interval + 2
    train_indices = [
        torch.randint(len(train_data) - block_size - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(n_steps)
    ]
    eval_indices = [
        [torch.randint(len(val_data) - block_size - 1, (batch_size,),
                        generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]
    print(f"Pre-generated {n_steps} train batches, {n_evals} eval points, "
          f"{probe_batches} probe batches")

    def make_batch(indices, split_data):
        x = torch.stack(
            [split_data[i.item():i.item() + block_size] for i in indices])
        y = torch.stack(
            [split_data[i.item() + 1:i.item() + block_size + 1]
             for i in indices])
        return x.to(device), y.to(device)

    # =========================================================
    # Initialize shared weights (same init for all conditions)
    # =========================================================
    torch.manual_seed(seed)
    init_model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    init_fwd = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)

    init_model_state = {
        k: v.cpu().clone() for k, v in init_model.state_dict().items()
    }
    init_fwd_state = {
        k: v.cpu().clone() for k, v in init_fwd.state_dict().items()
    }

    # Separate init for autoencoder (independent from forward model)
    torch.manual_seed(seed + 100)
    init_autoenc = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    init_autoenc_state = {
        k: v.cpu().clone() for k, v in init_autoenc.state_dict().items()
    }

    # Frozen random projection for random_proj baseline
    rng_proj = np.random.default_rng(seed + 200)
    random_proj_weight = torch.from_numpy(
        (rng_proj.standard_normal((n_embd, n_embd)) / np.sqrt(n_embd)
         ).astype(np.float32)
    ).to(device)

    del init_model, init_fwd, init_autoenc
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =========================================================
    # Output directory
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_root = (f"{DATA_DIR}/a2a_forward/baseline_battery/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    # =========================================================
    # Training function (parameterized by condition)
    # =========================================================
    def train_run(condition: str):
        label = condition.upper().replace("_", "-")
        print(f"\n{'='*60}")
        print(f"  Training {label} (lr={lr}, seed={seed})")
        print(f"{'='*60}")

        has_injection = condition != "open_loop"

        # --- Create models ---
        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_model_state)

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fwd_model.load_state_dict(init_fwd_state)

        gate = CerebellarGate(n_embd).to(device)

        autoenc_model = None
        if condition == "autoencoder":
            autoenc_model = TransformerForwardModel(
                d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                block_size=block_size,
            ).to(device)
            autoenc_model.load_state_dict(init_autoenc_state)

        # --- Optimizers ---
        if has_injection:
            main_params = list(model.parameters()) + list(gate.parameters())
        else:
            main_params = list(model.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

        opt_autoenc = None
        if autoenc_model is not None:
            opt_autoenc = torch.optim.AdamW(
                autoenc_model.parameters(), lr=fwd_lr, weight_decay=0.01)

        # --- History ---
        history = {
            "lm_loss": [], "fwd_mse": [],
            "val_lm": [], "val_lm_no_inj": [],
            "val_fwd_mse": [], "val_cosine": [],
        }
        if has_injection:
            history["gate_norm"] = []
        if condition == "autoencoder":
            history["autoenc_mse"] = []

        # --- Build cerebellar_fn for this condition ---
        def make_cerebellar_fn(fwd_m, gate_m, cond, autoenc_m=None,
                               fwd_pred_cache=None):
            if cond == "forward":
                def fn(act):
                    pred = fwd_m(act.detach())
                    if fwd_pred_cache is not None:
                        fwd_pred_cache["pred"] = pred
                    return gate_m(pred.detach())
                return fn

            elif cond == "shifted":
                def fn(act):
                    pred = fwd_m(act.detach())
                    if fwd_pred_cache is not None:
                        fwd_pred_cache["pred"] = pred
                    # Causal shift: position t gets prediction from t-k
                    shifted = torch.zeros_like(pred)
                    if shift_k < pred.shape[1]:
                        shifted[:, shift_k:, :] = pred[:, :-shift_k, :]
                    return gate_m(shifted.detach())
                return fn

            elif cond == "random_proj":
                def fn(act):
                    proj = act.detach() @ random_proj_weight.T
                    return gate_m(proj)
                return fn

            elif cond == "autoencoder":
                def fn(act):
                    recon = autoenc_m(act.detach())
                    return gate_m(recon.detach())
                return fn

            return None

        # --- Training loop ---
        eval_idx = 0
        for step in range(n_steps):
            model.train()
            fwd_model.train()
            gate.train()
            if autoenc_model is not None:
                autoenc_model.train()

            x, y = make_batch(train_indices[step], train_data)

            if has_injection:
                fwd_pred_cache = {}
                cb_fn = make_cerebellar_fn(
                    fwd_model, gate, condition, autoenc_model,
                    fwd_pred_cache)

                _, lm_loss, intermediates = model(
                    x, y, return_intermediates=True,
                    cerebellar_fn=cb_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                target = intermediates[predict_to].detach()
                source = intermediates[predict_from].detach()

                # Forward model always predicts post_block0 → post_block3
                if condition in ("forward", "shifted"):
                    fwd_pred = fwd_pred_cache.get("pred")
                    if fwd_pred is None:
                        fwd_pred = fwd_model(source)
                else:
                    fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)

                # Autoencoder loss (reconstruction of post_block0)
                autoenc_loss = None
                if autoenc_model is not None:
                    autoenc_pred = autoenc_model(source)
                    autoenc_loss = F.mse_loss(autoenc_pred, source)
            else:
                _, lm_loss, intermediates = model(
                    x, y, return_intermediates=True)
                source = intermediates[predict_from].detach()
                target = intermediates[predict_to].detach()
                fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)

            # --- Backward passes ---
            opt_main.zero_grad()
            lm_loss.backward()
            if has_injection:
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            if opt_autoenc is not None and autoenc_loss is not None:
                opt_autoenc.zero_grad()
                autoenc_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    autoenc_model.parameters(), 1.0)
                opt_autoenc.step()

            # --- Eval ---
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                gate.eval()
                if autoenc_model is not None:
                    autoenc_model.eval()

                with torch.no_grad():
                    val_lm_acc = 0.0
                    val_lm_noinj_acc = 0.0
                    val_fwd_acc = 0.0
                    val_cos_acc = 0.0

                    for eb_idx in eval_indices[eval_idx]:
                        vx, vy = make_batch(eb_idx, val_data)

                        if has_injection:
                            eval_cb = make_cerebellar_fn(
                                fwd_model, gate, condition, autoenc_model)
                            _, vl, _ = model(
                                vx, vy, return_intermediates=True,
                                cerebellar_fn=eval_cb,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                            val_lm_acc += float(vl)

                        _, vl_noinj, vi = model(
                            vx, vy, return_intermediates=True)
                        if not has_injection:
                            val_lm_acc += float(vl_noinj)
                        val_lm_noinj_acc += float(vl_noinj)

                        src = vi[predict_from]
                        tgt = vi[predict_to]
                        pred = fwd_model(src)
                        val_fwd_acc += float(F.mse_loss(pred, tgt))
                        val_cos_acc += float(
                            F.cosine_similarity(pred, tgt, dim=-1).mean())

                    n = n_eval_batches
                    val_lm = val_lm_acc / n
                    val_lm_noinj = val_lm_noinj_acc / n
                    val_fwd = val_fwd_acc / n
                    val_cos = val_cos_acc / n

                    history["lm_loss"].append((step, float(lm_loss)))
                    history["fwd_mse"].append((step, float(fwd_loss)))
                    history["val_lm"].append((step, val_lm))
                    history["val_lm_no_inj"].append((step, val_lm_noinj))
                    history["val_fwd_mse"].append((step, val_fwd))
                    history["val_cosine"].append((step, val_cos))

                    if has_injection:
                        gn = gate.injection_norm()
                        history["gate_norm"].append((step, gn))
                        delta = val_lm - val_lm_noinj
                        print(f"  [{label}] step {step:6d}: "
                              f"lm={val_lm:.4f} no_inj={val_lm_noinj:.4f} "
                              f"Δ={delta:+.4f} | cos={val_cos:.4f} "
                              f"gate={gn:.4f}")
                    else:
                        print(f"  [{label}] step {step:6d}: "
                              f"lm={val_lm:.4f} | cos={val_cos:.4f}")

                    if autoenc_model is not None:
                        autoenc_pred = autoenc_model(vi[predict_from])
                        ae_mse = float(F.mse_loss(
                            autoenc_pred, vi[predict_from]))
                        history["autoenc_mse"].append((step, ae_mse))

                eval_idx += 1

        # --- Save checkpoint ---
        save_dir = os.path.join(save_root, condition)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(fwd_model.state_dict(),
                    os.path.join(save_dir, "fwd_model.pt"))
        if has_injection:
            torch.save(gate.state_dict(), os.path.join(save_dir, "gate.pt"))
        if autoenc_model is not None:
            torch.save(autoenc_model.state_dict(),
                        os.path.join(save_dir, "autoenc_model.pt"))
        print(f"  Saved checkpoint to {save_dir}")

        return model, fwd_model, gate, autoenc_model, history

    # =========================================================
    # Run all conditions
    # =========================================================
    models = {}
    fwd_models = {}
    gates = {}
    autoenc_models = {}
    histories = {}

    for cond in CONDITIONS:
        m, f, g, ae, h = train_run(cond)
        models[cond] = m
        fwd_models[cond] = f
        gates[cond] = g
        autoenc_models[cond] = ae
        histories[cond] = h

    # =========================================================
    # Self-knowledge probes
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  SELF-KNOWLEDGE PROBES ({probe_batches} batches, "
          f"{probe_steps} steps)")
    print(f"{'='*60}")

    def collect_probe_data(model, fwd_model):
        model.eval()
        fwd_model.eval()
        acts = {k: [] for k in layer_keys}
        residual_vecs = []
        residual_norms = []

        with torch.no_grad():
            for pidx in probe_indices:
                vx, vy = make_batch(pidx, val_data)
                _, _, vi = model(vx, vy, return_intermediates=True)

                for k in layer_keys:
                    acts[k].append(vi[k].reshape(-1, n_embd).cpu())

                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fwd_model(src)
                res = tgt - pred
                residual_vecs.append(res.reshape(-1, n_embd).cpu())
                residual_norms.append(
                    res.norm(dim=-1).reshape(-1).cpu())

        acts = {k: torch.cat(v) for k, v in acts.items()}
        return {
            "acts": acts,
            "residual_vec": torch.cat(residual_vecs),
            "residual_norm": torch.cat(residual_norms),
        }

    n_samples = None
    n_train_probe = None
    train_idx = None
    test_idx = None

    def vector_probe(X, target_vec):
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        T_np = target_vec.numpy() if isinstance(
            target_vec, torch.Tensor) else target_vec
        T_np = T_np.astype(np.float32)

        mu_x = X_np[train_idx].mean(axis=0, keepdims=True)
        sd_x = X_np[train_idx].std(axis=0, keepdims=True) + 1e-6
        X_s = (X_np - mu_x) / sd_x

        probe = nn.Linear(X_np.shape[1], T_np.shape[1]).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        Xtr = torch.from_numpy(X_s[train_idx]).float().to(device)
        Ttr = torch.from_numpy(T_np[train_idx]).float().to(device)
        bs = min(4096, n_train_probe)
        for _ in range(probe_steps):
            idx = torch.randint(n_train_probe, (bs,), device=device)
            pred = probe(Xtr[idx])
            loss = F.mse_loss(pred, Ttr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            Xte = torch.from_numpy(X_s[test_idx]).float().to(device)
            pred = probe(Xte).cpu().numpy()
        Tte = T_np[test_idx]
        mse = ((pred - Tte) ** 2).mean()
        var = Tte.var()
        r2 = float(1.0 - mse / var) if var > 0 else 0.0
        return r2

    # Collect probe data for all conditions
    probe_data = {}
    for cond in CONDITIONS:
        print(f"Collecting probe data for {cond}...")
        probe_data[cond] = collect_probe_data(
            models[cond], fwd_models[cond])

    # Set up train/test split (same for all conditions)
    n_samples = probe_data["open_loop"]["residual_norm"].shape[0]
    n_train_probe = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train_probe]
    test_idx = perm[n_train_probe:]

    print(f"Probe dataset: {n_samples} samples "
          f"({n_train_probe} train, {n_samples - n_train_probe} test)")

    # Run vector probes
    probe_results = {}
    print(f"\n{'layer':>12s}", end="")
    for cond in CONDITIONS:
        print(f"  {cond:>12s}", end="")
    print(f"  {'Δ fwd-open':>12s}")
    print("-" * (14 + 14 * len(CONDITIONS) + 14))

    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            r2 = vector_probe(
                probe_data[cond]["acts"][lk],
                probe_data[cond]["residual_vec"])
            if cond not in probe_results:
                probe_results[cond] = {}
            probe_results[cond][lk] = r2
            print(f"  {r2:>12.4f}", end="")
        delta = probe_results["forward"][lk] - probe_results["open_loop"][lk]
        print(f"  {delta:>+12.4f}")

    print(f"\n--- Self-knowledge Δ R² (condition − open_loop) ---")
    print(f"{'layer':>12s}", end="")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        print(f"  {cond:>12s}", end="")
    print()
    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            if cond == "open_loop":
                continue
            delta = probe_results[cond][lk] - probe_results["open_loop"][lk]
            print(f"  {delta:>+12.4f}", end="")
        print()

    # =========================================================
    # Perturbation robustness test
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  PERTURBATION ROBUSTNESS (s={perturb_s}, "
          f"{n_perturb_dirs} dirs, {perturb_eval_batches} batches)")
    print(f"{'='*60}")

    rng_perturb = np.random.default_rng(seed + 300)
    perturb_dirs = []
    for i in range(n_perturb_dirs):
        d = rng_perturb.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        perturb_dirs.append(d)

    # Compute perturbation scale from open-loop model
    with torch.no_grad():
        x_ref, y_ref = make_batch(
            eval_indices[0][0], val_data)
        _, _, vi_ref = models["open_loop"](
            x_ref, y_ref, return_intermediates=True)
        b1_std = float(vi_ref["post_block1"].std())
    print(f"post_block1 activation std: {b1_std:.4f}")

    perturb_gen = torch.Generator().manual_seed(seed + 400)
    perturb_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=perturb_gen)
        for _ in range(perturb_eval_batches)
    ]

    def run_with_perturbation(model, x, y, perturbation=None):
        if perturbation is not None:
            def cb_fn(act, _p=perturbation):
                return _p.unsqueeze(0).unsqueeze(0).expand(
                    act.shape[0], act.shape[1], -1)
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        else:
            return model(x, y, return_intermediates=True)

    robustness_results = {}

    for cond in CONDITIONS:
        model = models[cond]
        model.eval()
        print(f"\n  [{cond.upper()}]")

        response_norms = []
        cosines_with_delta = []
        loss_deltas = []

        with torch.no_grad():
            for di in range(n_perturb_dirs):
                dir_t = torch.from_numpy(perturb_dirs[di]).to(device)
                perturbation = (perturb_s * b1_std) * dir_t

                rn_batch, cos_batch, ld_batch = [], [], []

                for p_idx in perturb_indices:
                    x, y = make_batch(p_idx, val_data)
                    tgt = x[:, 1:]

                    logits_base, _, inter_base = run_with_perturbation(
                        model, x, y)
                    b3_base = inter_base["post_block3"][:, :-1]
                    loss_base = -F.log_softmax(
                        logits_base[:, :-1], dim=-1
                    ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                    logits_pert, _, inter_pert = run_with_perturbation(
                        model, x, y, perturbation)
                    b3_pert = inter_pert["post_block3"][:, :-1]
                    loss_pert = -F.log_softmax(
                        logits_pert[:, :-1], dim=-1
                    ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                    response = (b3_pert - b3_base).reshape(-1, n_embd)
                    r_norm = response.norm(dim=-1)
                    cos = F.cosine_similarity(
                        response,
                        dir_t.unsqueeze(0).expand_as(response),
                        dim=-1)

                    valid = r_norm > 1e-10
                    if valid.sum() > 0:
                        rn_batch.append(r_norm[valid].mean().item())
                        cos_batch.append(cos[valid].mean().item())
                        ld_batch.append(
                            (loss_pert - loss_base).mean().item())

                if rn_batch:
                    response_norms.append(np.mean(rn_batch))
                    cosines_with_delta.append(np.mean(cos_batch))
                    loss_deltas.append(np.mean(ld_batch))

        robustness_results[cond] = {
            "response_norm": float(np.mean(response_norms)),
            "response_norm_std": float(np.std(response_norms)),
            "cosine_with_delta": float(np.mean(cosines_with_delta)),
            "cosine_with_delta_std": float(np.std(cosines_with_delta)),
            "loss_delta": float(np.mean(loss_deltas)),
            "loss_delta_std": float(np.std(loss_deltas)),
        }

        print(f"    ||R||={np.mean(response_norms):.4f} "
              f"cos(R,δ)={np.mean(cosines_with_delta):+.4f} "
              f"Δloss={np.mean(loss_deltas):+.4f}")

    # Robustness summary
    print(f"\n--- Perturbation robustness summary (s={perturb_s}) ---")
    print(f"{'Condition':>12s} {'||R||':>8s} {'cos(R,δ)':>10s} "
          f"{'Δloss':>8s} {'||R||/OL':>8s} {'Δloss/OL':>8s}")
    print("-" * 62)
    ol_norm = robustness_results["open_loop"]["response_norm"]
    ol_loss = robustness_results["open_loop"]["loss_delta"]
    for cond in CONDITIONS:
        r = robustness_results[cond]
        r_ratio = r["response_norm"] / ol_norm if ol_norm > 0 else float("nan")
        l_ratio = r["loss_delta"] / ol_loss if ol_loss > 0 else float("nan")
        print(f"{cond:>12s} {r['response_norm']:>8.4f} "
              f"{r['cosine_with_delta']:>+10.4f} "
              f"{r['loss_delta']:>+8.4f} {r_ratio:>8.3f} {l_ratio:>8.3f}")

    # =========================================================
    # Final LM loss comparison
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  FINAL LM LOSS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':>12s} {'val_lm':>8s} {'no_inj':>8s} "
          f"{'Δ(inj)':>8s} {'Δ(vs OL)':>9s}")
    print("-" * 50)

    ol_final = histories["open_loop"]["val_lm"][-1][1]
    for cond in CONDITIONS:
        h = histories[cond]
        val_lm = h["val_lm"][-1][1]
        val_noinj = h["val_lm_no_inj"][-1][1]
        inj_delta = val_lm - val_noinj if cond != "open_loop" else 0.0
        vs_ol = val_lm - ol_final
        print(f"{cond:>12s} {val_lm:>8.4f} {val_noinj:>8.4f} "
              f"{inj_delta:>+8.4f} {vs_ol:>+9.4f}")

    # =========================================================
    # Save everything
    # =========================================================
    result = {
        "config": {
            "lr": lr, "fwd_lr": fwd_lr, "seed": seed,
            "n_tokens": n_tokens, "n_steps": n_steps,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
            "shift_k": shift_k,
            "perturb_s": perturb_s, "n_perturb_dirs": n_perturb_dirs,
            "conditions": CONDITIONS,
        },
        "lm_loss": {},
        "probe_results": probe_results,
        "robustness": robustness_results,
        "histories": histories,
    }

    for cond in CONDITIONS:
        h = histories[cond]
        result["lm_loss"][cond] = {
            "final_val_lm": h["val_lm"][-1][1],
            "final_val_lm_no_inj": h["val_lm_no_inj"][-1][1],
        }

    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nAll results saved to {save_root}")

    # =========================================================
    # Executive summary
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  EXECUTIVE SUMMARY")
    print(f"{'='*60}")

    print("\n1. LM loss improvement (val_lm with injection − open_loop):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        delta = histories[cond]["val_lm"][-1][1] - ol_final
        print(f"    {cond:>12s}: {delta:+.4f} nats")

    print("\n2. Self-knowledge (mean Δ R² across layers, vs open_loop):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        mean_delta = np.mean([
            probe_results[cond][lk] - probe_results["open_loop"][lk]
            for lk in layer_keys
        ])
        print(f"    {cond:>12s}: {mean_delta:+.4f}")

    print("\n3. Perturbation robustness (Δloss / open_loop Δloss):")
    for cond in CONDITIONS:
        r = robustness_results[cond]
        l_ratio = r["loss_delta"] / ol_loss if ol_loss > 0 else float("nan")
        print(f"    {cond:>12s}: {l_ratio:.3f}x "
              f"({'more' if l_ratio > 1 else 'less'} degradation)")

    fwd_unique_lm = all(
        histories["forward"]["val_lm"][-1][1]
        <= histories[c]["val_lm"][-1][1]
        for c in CONDITIONS if c != "open_loop"
    )
    fwd_unique_robust = all(
        robustness_results["forward"]["loss_delta"]
        <= robustness_results[c]["loss_delta"]
        for c in CONDITIONS if c != "open_loop"
    )
    fwd_unique_sk = all(
        np.mean([probe_results["forward"][lk] for lk in layer_keys])
        >= np.mean([probe_results[c][lk] for lk in layer_keys])
        for c in CONDITIONS if c != "open_loop"
    )

    print(f"\n  Forward model uniquely best at LM loss:    {fwd_unique_lm}")
    print(f"  Forward model uniquely best at robustness: {fwd_unique_robust}")
    print(f"  Forward model uniquely best at SK probes:  {fwd_unique_sk}")

    return result


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_steps: int = 10_000,
    lr: float = 3e-4,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    shift_k: int = 10,
):
    result = a2a_baseline_battery.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
        lr=lr,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
        shift_k=shift_k,
    )
    print("\n=== BASELINE BATTERY COMPLETE ===")
    print("\nLM loss:")
    for cond in CONDITIONS:
        lm = result["lm_loss"][cond]
        print(f"  {cond:>12s}: val={lm['final_val_lm']:.4f} "
              f"no_inj={lm['final_val_lm_no_inj']:.4f}")

    print("\nSelf-knowledge (Δ R² vs open_loop, mean across layers):")
    import numpy as np
    layer_keys = [f"post_block{i}" for i in range(4)]
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        mean_delta = np.mean([
            result["probe_results"][cond][lk]
            - result["probe_results"]["open_loop"][lk]
            for lk in layer_keys
        ])
        print(f"  {cond:>12s}: {mean_delta:+.4f}")

    print("\nRobustness (Δloss ratio vs open_loop):")
    ol_loss = result["robustness"]["open_loop"]["loss_delta"]
    for cond in CONDITIONS:
        ratio = result["robustness"][cond]["loss_delta"] / ol_loss \
            if ol_loss > 0 else 0
        print(f"  {cond:>12s}: {ratio:.3f}x")
