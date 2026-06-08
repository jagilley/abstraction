"""Emotion injection experiment: evaluative vs predictive self-referential signals.

Tests whether injecting an evaluative signal (loss prediction = "how difficult
will this be?") produces qualitatively different effects than injecting a
predictive signal (activation prediction = "what will my computation look like?").

Biological motivation: the thalamus routes multiple signal types into the cortex.
The cerebellum provides predictions about future computation (forward model).
The amygdala/emotional system provides evaluative signals about the current
situation's relevance and difficulty (loss predictor). Both are processed by the
cortex through its own weights. This experiment tests whether these two signal
types produce qualitatively different representational reorganization.

5 conditions, all with identical lr/seed/init/data order:
  1. open_loop:        no injection (baseline)
  2. forward:          inject gate(fwd_model(post_block0)) — cerebellum analog
  3. loss_scalar:      inject egate(loss_pred_1d(post_block0)) — scalar emotion
  4. loss_embed:       inject egate(loss_pred_8d(post_block0)) — multi-dim emotion
  5. forward_plus_loss: inject both forward + scalar loss — combined

All conditions co-train a forward model (post_block0 → post_block3) for probing.

After training, measures:
  - Validation LM loss (with and without injection)
  - Self-knowledge probes (R² for forward model's residual, across layers)
  - Perturbation robustness (response norm, loss degradation, cos(R,δ))
  - Loss predictor accuracy (R² of per-token loss prediction)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "forward", "loss_scalar", "loss_embed", "forward_plus_loss"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def a2a_emotion_injection(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    loss_pred_lr: float = 1e-3,
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
    loss_pred_hidden: int = 128,
    loss_embed_dim: int = 8,
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
        LossPredictor, EmotionGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1")
    )
    print(f"EMOTION INJECTION EXPERIMENT on {device}")
    print(f"  lr={lr}, seed={seed}, n_steps={n_steps}")
    print(f"  Forward model: {fwd_n_layer}L, {predict_from} → {predict_to}")
    print(f"  Injection: after block {inject_after_block}")
    print(f"  Loss predictor: hidden={loss_pred_hidden}, embed_dim={loss_embed_dim}")
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

    # Loss predictor inits (separate seeds for scalar vs embed)
    torch.manual_seed(seed + 100)
    init_lp_scalar = LossPredictor(
        d_model=n_embd, embed_dim=1, hidden=loss_pred_hidden
    ).to(device)
    init_lp_scalar_state = {
        k: v.cpu().clone() for k, v in init_lp_scalar.state_dict().items()
    }

    torch.manual_seed(seed + 101)
    init_lp_embed = LossPredictor(
        d_model=n_embd, embed_dim=loss_embed_dim, hidden=loss_pred_hidden
    ).to(device)
    init_lp_embed_state = {
        k: v.cpu().clone() for k, v in init_lp_embed.state_dict().items()
    }

    del init_model, init_fwd, init_lp_scalar, init_lp_embed
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =========================================================
    # Output directory
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_root = (f"{DATA_DIR}/a2a_forward/emotion_injection/"
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
        has_loss_pred = condition in ("loss_scalar", "loss_embed",
                                      "forward_plus_loss")
        has_forward_inj = condition in ("forward", "forward_plus_loss")
        has_loss_inj = condition in ("loss_scalar", "loss_embed",
                                     "forward_plus_loss")

        # --- Create models ---
        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_model_state)

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fwd_model.load_state_dict(init_fwd_state)

        # Cerebellar gate (for forward injection)
        cerebellar_gate = CerebellarGate(n_embd).to(device) \
            if has_forward_inj else None

        # Loss predictor + emotion gate
        loss_predictor = None
        emotion_gate = None
        if has_loss_pred:
            if condition == "loss_embed":
                edim = loss_embed_dim
                loss_predictor = LossPredictor(
                    d_model=n_embd, embed_dim=edim, hidden=loss_pred_hidden
                ).to(device)
                loss_predictor.load_state_dict(init_lp_embed_state)
            else:
                edim = 1
                loss_predictor = LossPredictor(
                    d_model=n_embd, embed_dim=1, hidden=loss_pred_hidden
                ).to(device)
                loss_predictor.load_state_dict(init_lp_scalar_state)
            emotion_gate = EmotionGate(edim, n_embd).to(device)

        # --- Optimizers ---
        main_params = list(model.parameters())
        if cerebellar_gate is not None:
            main_params += list(cerebellar_gate.parameters())
        if emotion_gate is not None:
            main_params += list(emotion_gate.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

        opt_lp = None
        if loss_predictor is not None:
            opt_lp = torch.optim.AdamW(
                loss_predictor.parameters(), lr=loss_pred_lr, weight_decay=0.01)

        # --- History ---
        history = {
            "lm_loss": [], "fwd_mse": [],
            "val_lm": [], "val_lm_no_inj": [],
            "val_fwd_mse": [], "val_cosine": [],
        }
        if cerebellar_gate is not None:
            history["cerebellar_gate_norm"] = []
        if emotion_gate is not None:
            history["emotion_gate_norm"] = []
        if has_loss_pred:
            history["loss_pred_mse"] = []
            history["val_loss_pred_r2"] = []

        # --- Build cerebellar_fn for this condition ---
        def make_cerebellar_fn(fwd_m, cb_gate, lp, eg, cond,
                               fwd_pred_cache=None):
            if cond == "forward":
                def fn(act):
                    pred = fwd_m(act.detach())
                    if fwd_pred_cache is not None:
                        fwd_pred_cache["pred"] = pred
                    return cb_gate(pred.detach())
                return fn

            elif cond == "loss_scalar" or cond == "loss_embed":
                def fn(act):
                    embed, _ = lp(act.detach())
                    return eg(embed.detach())
                return fn

            elif cond == "forward_plus_loss":
                def fn(act):
                    pred = fwd_m(act.detach())
                    if fwd_pred_cache is not None:
                        fwd_pred_cache["pred"] = pred
                    embed, _ = lp(act.detach())
                    return (cb_gate(pred.detach())
                            + eg(embed.detach()))
                return fn

            return None

        # --- Training loop ---
        eval_idx = 0
        for step in range(n_steps):
            model.train()
            fwd_model.train()
            if cerebellar_gate is not None:
                cerebellar_gate.train()
            if loss_predictor is not None:
                loss_predictor.train()
            if emotion_gate is not None:
                emotion_gate.train()

            x, y = make_batch(train_indices[step], train_data)

            if has_injection:
                fwd_pred_cache = {}
                cb_fn = make_cerebellar_fn(
                    fwd_model, cerebellar_gate, loss_predictor,
                    emotion_gate, condition, fwd_pred_cache)

                logits, lm_loss, intermediates = model(
                    x, y, return_intermediates=True,
                    cerebellar_fn=cb_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                target = intermediates[predict_to].detach()
                source = intermediates[predict_from].detach()

                if has_forward_inj:
                    fwd_pred = fwd_pred_cache.get("pred")
                    if fwd_pred is None:
                        fwd_pred = fwd_model(source)
                else:
                    fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)
            else:
                logits, lm_loss, intermediates = model(
                    x, y, return_intermediates=True)
                source = intermediates[predict_from].detach()
                target = intermediates[predict_to].detach()
                fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)

            # --- Loss predictor training ---
            loss_pred_loss = None
            if has_loss_pred:
                with torch.no_grad():
                    per_token_ce = F.cross_entropy(
                        logits.detach().reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        reduction='none'
                    ).reshape(x.size(0), block_size)

                _, pred_loss = loss_predictor(source)
                loss_pred_loss = F.mse_loss(pred_loss, per_token_ce.detach())

            # --- Backward passes ---
            opt_main.zero_grad()
            lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
            opt_fwd.step()

            if opt_lp is not None and loss_pred_loss is not None:
                opt_lp.zero_grad()
                loss_pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_predictor.parameters(), 1.0)
                opt_lp.step()

            # --- Eval ---
            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                if cerebellar_gate is not None:
                    cerebellar_gate.eval()
                if loss_predictor is not None:
                    loss_predictor.eval()
                if emotion_gate is not None:
                    emotion_gate.eval()

                with torch.no_grad():
                    val_lm_acc = 0.0
                    val_lm_noinj_acc = 0.0
                    val_fwd_acc = 0.0
                    val_cos_acc = 0.0
                    val_lp_preds = []
                    val_lp_targets = []

                    for eb_idx in eval_indices[eval_idx]:
                        vx, vy = make_batch(eb_idx, val_data)

                        if has_injection:
                            eval_cb = make_cerebellar_fn(
                                fwd_model, cerebellar_gate,
                                loss_predictor, emotion_gate, condition)
                            vlogits, vl, _ = model(
                                vx, vy, return_intermediates=True,
                                cerebellar_fn=eval_cb,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )
                            val_lm_acc += float(vl)

                        noinj_logits, vl_noinj, vi = model(
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

                        if has_loss_pred:
                            per_token_ce = F.cross_entropy(
                                noinj_logits.reshape(
                                    -1, noinj_logits.size(-1)),
                                vy.reshape(-1),
                                reduction='none'
                            ).reshape(vx.size(0), block_size)
                            _, vpred_loss = loss_predictor(src)
                            val_lp_preds.append(vpred_loss.cpu())
                            val_lp_targets.append(per_token_ce.cpu())

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

                    log_parts = [f"lm={val_lm:.4f}"]
                    if has_injection:
                        delta = val_lm - val_lm_noinj
                        log_parts.append(f"no_inj={val_lm_noinj:.4f}")
                        log_parts.append(f"Δ={delta:+.4f}")

                    log_parts.append(f"cos={val_cos:.4f}")

                    if cerebellar_gate is not None:
                        gn = cerebellar_gate.injection_norm()
                        history["cerebellar_gate_norm"].append((step, gn))
                        log_parts.append(f"cgate={gn:.4f}")

                    if emotion_gate is not None:
                        en = emotion_gate.injection_norm()
                        history["emotion_gate_norm"].append((step, en))
                        log_parts.append(f"egate={en:.4f}")

                    if has_loss_pred and val_lp_preds:
                        all_preds = torch.cat(val_lp_preds).numpy()
                        all_tgts = torch.cat(val_lp_targets).numpy()
                        lp_mse = float(np.mean((all_preds - all_tgts) ** 2))
                        lp_var = float(np.var(all_tgts))
                        lp_r2 = 1.0 - lp_mse / lp_var if lp_var > 0 else 0.0
                        history["loss_pred_mse"].append((step, lp_mse))
                        history["val_loss_pred_r2"].append((step, lp_r2))
                        log_parts.append(f"lp_r²={lp_r2:.4f}")

                    print(f"  [{label}] step {step:6d}: "
                          + " | ".join(log_parts))

                eval_idx += 1

        # --- Save checkpoint ---
        save_dir = os.path.join(save_root, condition)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(fwd_model.state_dict(),
                    os.path.join(save_dir, "fwd_model.pt"))
        if cerebellar_gate is not None:
            torch.save(cerebellar_gate.state_dict(),
                        os.path.join(save_dir, "cerebellar_gate.pt"))
        if loss_predictor is not None:
            torch.save(loss_predictor.state_dict(),
                        os.path.join(save_dir, "loss_predictor.pt"))
        if emotion_gate is not None:
            torch.save(emotion_gate.state_dict(),
                        os.path.join(save_dir, "emotion_gate.pt"))
        print(f"  Saved checkpoint to {save_dir}")

        return model, fwd_model, cerebellar_gate, loss_predictor, \
            emotion_gate, history

    # =========================================================
    # Run all conditions
    # =========================================================
    models = {}
    fwd_models = {}
    cerebellar_gates = {}
    loss_predictors = {}
    emotion_gates = {}
    histories = {}

    for cond in CONDITIONS:
        m, f, cg, lp, eg, h = train_run(cond)
        models[cond] = m
        fwd_models[cond] = f
        cerebellar_gates[cond] = cg
        loss_predictors[cond] = lp
        emotion_gates[cond] = eg
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

    probe_data = {}
    for cond in CONDITIONS:
        print(f"Collecting probe data for {cond}...")
        probe_data[cond] = collect_probe_data(
            models[cond], fwd_models[cond])

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
        print(f"  {cond:>16s}", end="")
    print(f"  {'Δ fwd-open':>12s}")
    print("-" * (14 + 18 * len(CONDITIONS) + 14))

    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            r2 = vector_probe(
                probe_data[cond]["acts"][lk],
                probe_data[cond]["residual_vec"])
            if cond not in probe_results:
                probe_results[cond] = {}
            probe_results[cond][lk] = r2
            print(f"  {r2:>16.4f}", end="")
        delta = probe_results["forward"][lk] - probe_results["open_loop"][lk]
        print(f"  {delta:>+12.4f}")

    print(f"\n--- Self-knowledge Δ R² (condition − open_loop) ---")
    print(f"{'layer':>12s}", end="")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        print(f"  {cond:>16s}", end="")
    print()
    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            if cond == "open_loop":
                continue
            delta = probe_results[cond][lk] - probe_results["open_loop"][lk]
            print(f"  {delta:>+16.4f}", end="")
        print()

    # Depth ratio (b3/b0)
    print(f"\n--- Self-knowledge depth ratio (post_block3 / post_block0 Δ R²) ---")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        d0 = probe_results[cond]["post_block0"] - \
            probe_results["open_loop"]["post_block0"]
        d3 = probe_results[cond]["post_block3"] - \
            probe_results["open_loop"]["post_block3"]
        ratio = d3 / d0 if abs(d0) > 1e-6 else float("nan")
        print(f"  {cond:>16s}: {ratio:.3f} (b0={d0:+.4f}, b3={d3:+.4f})")

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

    with torch.no_grad():
        x_ref, y_ref = make_batch(eval_indices[0][0], val_data)
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
    print(f"{'Condition':>16s} {'||R||':>8s} {'cos(R,δ)':>10s} "
          f"{'Δloss':>8s} {'||R||/OL':>8s} {'Δloss/OL':>8s}")
    print("-" * 68)
    ol_norm = robustness_results["open_loop"]["response_norm"]
    ol_loss = robustness_results["open_loop"]["loss_delta"]
    for cond in CONDITIONS:
        r = robustness_results[cond]
        r_ratio = r["response_norm"] / ol_norm if ol_norm > 0 else float("nan")
        l_ratio = r["loss_delta"] / ol_loss if ol_loss > 0 else float("nan")
        print(f"{cond:>16s} {r['response_norm']:>8.4f} "
              f"{r['cosine_with_delta']:>+10.4f} "
              f"{r['loss_delta']:>+8.4f} {r_ratio:>8.3f} {l_ratio:>8.3f}")

    # =========================================================
    # Final LM loss comparison
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  FINAL LM LOSS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':>16s} {'val_lm':>8s} {'no_inj':>8s} "
          f"{'Δ(inj)':>8s} {'Δ(vs OL)':>9s}")
    print("-" * 55)

    ol_final = histories["open_loop"]["val_lm"][-1][1]
    for cond in CONDITIONS:
        h = histories[cond]
        val_lm = h["val_lm"][-1][1]
        val_noinj = h["val_lm_no_inj"][-1][1]
        inj_delta = val_lm - val_noinj if cond != "open_loop" else 0.0
        vs_ol = val_lm - ol_final
        print(f"{cond:>16s} {val_lm:>8.4f} {val_noinj:>8.4f} "
              f"{inj_delta:>+8.4f} {vs_ol:>+9.4f}")

    # Loss predictor quality
    if any(histories[c].get("val_loss_pred_r2") for c in CONDITIONS):
        print(f"\n--- Loss predictor quality (final R²) ---")
        for cond in CONDITIONS:
            h = histories[cond]
            if h.get("val_loss_pred_r2"):
                r2 = h["val_loss_pred_r2"][-1][1]
                print(f"  {cond:>16s}: R²={r2:.4f}")

    # =========================================================
    # Save everything
    # =========================================================
    result = {
        "config": {
            "lr": lr, "fwd_lr": fwd_lr, "loss_pred_lr": loss_pred_lr,
            "seed": seed,
            "n_tokens": n_tokens, "n_steps": n_steps,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
            "loss_pred_hidden": loss_pred_hidden,
            "loss_embed_dim": loss_embed_dim,
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

    print("\n1. LM loss (injection benefit, val_lm with inj − val_lm no inj):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        h = histories[cond]
        inj_benefit = h["val_lm"][-1][1] - h["val_lm_no_inj"][-1][1]
        print(f"    {cond:>16s}: {inj_benefit:+.4f} nats")

    print("\n2. Self-knowledge (Δ R² vs open_loop, by layer):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        deltas = [probe_results[cond][lk] - probe_results["open_loop"][lk]
                  for lk in layer_keys]
        mean_d = np.mean(deltas)
        d0 = deltas[0]
        d3 = deltas[-1]
        ratio = d3 / d0 if abs(d0) > 1e-6 else float("nan")
        print(f"    {cond:>16s}: mean={mean_d:+.4f} "
              f"b0={d0:+.4f} b3={d3:+.4f} depth_ratio={ratio:.3f}")

    print("\n3. Perturbation robustness (Δloss / open_loop Δloss):")
    for cond in CONDITIONS:
        r = robustness_results[cond]
        l_ratio = r["loss_delta"] / ol_loss if ol_loss > 0 else float("nan")
        print(f"    {cond:>16s}: {l_ratio:.3f}x")

    print("\n4. Key comparisons:")
    fwd_rob = robustness_results["forward"]["loss_delta"]
    ls_rob = robustness_results["loss_scalar"]["loss_delta"]
    le_rob = robustness_results["loss_embed"]["loss_delta"]
    fl_rob = robustness_results["forward_plus_loss"]["loss_delta"]

    print(f"    Forward vs loss_scalar robustness: "
          f"{fwd_rob/ol_loss:.3f}x vs {ls_rob/ol_loss:.3f}x")
    print(f"    Forward vs loss_embed robustness: "
          f"{fwd_rob/ol_loss:.3f}x vs {le_rob/ol_loss:.3f}x")
    print(f"    Combined (fwd+loss) robustness: "
          f"{fl_rob/ol_loss:.3f}x "
          f"(additive would be ~{(fwd_rob + ls_rob - ol_loss)/ol_loss:.3f}x)")

    # Check if combination is superadditive
    fwd_sk_mean = np.mean([
        probe_results["forward"][lk] - probe_results["open_loop"][lk]
        for lk in layer_keys])
    ls_sk_mean = np.mean([
        probe_results["loss_scalar"][lk] - probe_results["open_loop"][lk]
        for lk in layer_keys])
    fl_sk_mean = np.mean([
        probe_results["forward_plus_loss"][lk]
        - probe_results["open_loop"][lk]
        for lk in layer_keys])
    print(f"    Combined SK: {fl_sk_mean:+.4f} "
          f"(fwd={fwd_sk_mean:+.4f}, loss={ls_sk_mean:+.4f}, "
          f"sum={fwd_sk_mean + ls_sk_mean:+.4f})")

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
    loss_pred_hidden: int = 128,
    loss_embed_dim: int = 8,
):
    result = a2a_emotion_injection.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
        lr=lr,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
        loss_pred_hidden=loss_pred_hidden,
        loss_embed_dim=loss_embed_dim,
    )
    print("\n=== EMOTION INJECTION EXPERIMENT COMPLETE ===")
    print("\nLM loss:")
    for cond in CONDITIONS:
        lm = result["lm_loss"][cond]
        print(f"  {cond:>16s}: val={lm['final_val_lm']:.4f} "
              f"no_inj={lm['final_val_lm_no_inj']:.4f}")

    import numpy as np
    layer_keys = [f"post_block{i}" for i in range(4)]

    print("\nSelf-knowledge (Δ R² vs open_loop):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        deltas = [result["probe_results"][cond][lk]
                  - result["probe_results"]["open_loop"][lk]
                  for lk in layer_keys]
        print(f"  {cond:>16s}: mean={np.mean(deltas):+.4f} "
              f"b0={deltas[0]:+.4f} b3={deltas[-1]:+.4f}")

    print("\nRobustness (Δloss ratio vs open_loop):")
    ol_loss = result["robustness"]["open_loop"]["loss_delta"]
    for cond in CONDITIONS:
        ratio = result["robustness"][cond]["loss_delta"] / ol_loss \
            if ol_loss > 0 else 0
        print(f"  {cond:>16s}: {ratio:.3f}x")
