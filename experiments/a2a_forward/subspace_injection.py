"""Subspace injection experiment: inject a known low-rank projection of
post_block0 into the residual stream and measure how the model adapts.

Injects rank-k (default k=16) projection of post_block0 activations through
the standard CerebellarGate. The projection directions are random orthonormal
vectors, frozen before training. This gives a fully characterized information
partition: we know exactly which k directions the injection carries and which
256-k it misses.

Key analyses:
  1. Standard self-knowledge probes (validates against prior work)
  2. Subspace variance fraction at each layer (reorganization signal)
  3. Subspace/complement probes (accessibility of known directions)
  4. Anisotropic perturbation (within-subspace vs complement sensitivity)

3 conditions, all with identical lr/seed/init/data order:
  1. open_loop:    no injection
  2. lowrank_proj: inject gate(P @ a_0) where P is rank-k projector
  3. forward:      standard forward model injection (reference)

All conditions co-train a forward model (post_block0 -> post_block3) for
probing, so the self-knowledge metric is comparable across conditions.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "lowrank_proj", "forward"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def a2a_subspace_injection(
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
    subspace_k: int = 16,
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
    print(f"SUBSPACE INJECTION EXPERIMENT on {device}")
    print(f"  lr={lr}, seed={seed}, n_steps={n_steps}")
    print(f"  Subspace k={subspace_k} of {n_embd}")
    print(f"  Forward model: {fwd_n_layer}L, {predict_from} -> {predict_to}")
    print(f"  Injection: after block {inject_after_block}")
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
    # Generate random orthonormal subspace
    # =========================================================
    rng_sub = np.random.default_rng(seed + 500)
    A = rng_sub.standard_normal((n_embd, n_embd)).astype(np.float32)
    Q_full, _ = np.linalg.qr(A)
    Q = Q_full[:, :subspace_k]
    Q_perp = Q_full[:, subspace_k:]
    projector = Q @ Q.T

    Q_t = torch.from_numpy(Q).to(device)
    Q_perp_t = torch.from_numpy(Q_perp).to(device)
    projector_t = torch.from_numpy(projector).to(device)
    print(f"Generated {subspace_k}-d subspace "
          f"({n_embd - subspace_k}-d complement)")

    # =========================================================
    # Initialize shared weights
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
    del init_model, init_fwd
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =========================================================
    # Output directory
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_root = (f"{DATA_DIR}/a2a_forward/subspace_injection/"
                 f"k{subspace_k}/{gap_tag}/inject{inject_after_block}/"
                 f"P_{n_tokens}")

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

        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_model_state)

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fwd_model.load_state_dict(init_fwd_state)

        gate = CerebellarGate(n_embd).to(device)

        if has_injection:
            main_params = list(model.parameters()) + list(gate.parameters())
        else:
            main_params = list(model.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

        history = {
            "lm_loss": [], "fwd_mse": [],
            "val_lm": [], "val_lm_no_inj": [],
            "val_fwd_mse": [], "val_cosine": [],
        }
        if has_injection:
            history["gate_norm"] = []

        def make_cerebellar_fn(fwd_m, gate_m, cond, fwd_pred_cache=None):
            if cond == "forward":
                def fn(act):
                    pred = fwd_m(act.detach())
                    if fwd_pred_cache is not None:
                        fwd_pred_cache["pred"] = pred
                    return gate_m(pred.detach())
                return fn
            elif cond == "lowrank_proj":
                def fn(act):
                    proj = act.detach() @ projector_t
                    return gate_m(proj)
                return fn
            return None

        eval_idx = 0
        for step in range(n_steps):
            model.train()
            fwd_model.train()
            gate.train()

            x, y = make_batch(train_indices[step], train_data)

            if has_injection:
                fwd_pred_cache = {}
                cb_fn = make_cerebellar_fn(
                    fwd_model, gate, condition, fwd_pred_cache)

                _, lm_loss, intermediates = model(
                    x, y, return_intermediates=True,
                    cerebellar_fn=cb_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                target = intermediates[predict_to].detach()
                source = intermediates[predict_from].detach()

                if condition == "forward":
                    fwd_pred = fwd_pred_cache.get("pred")
                    if fwd_pred is None:
                        fwd_pred = fwd_model(source)
                else:
                    fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)
            else:
                _, lm_loss, intermediates = model(
                    x, y, return_intermediates=True)
                source = intermediates[predict_from].detach()
                target = intermediates[predict_to].detach()
                fwd_pred = fwd_model(source)
                fwd_loss = F.mse_loss(fwd_pred, target)

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

            if step % eval_interval == 0 or step == n_steps - 1:
                model.eval()
                fwd_model.eval()
                gate.eval()

                with torch.no_grad():
                    val_lm_acc = 0.0
                    val_lm_noinj_acc = 0.0
                    val_fwd_acc = 0.0
                    val_cos_acc = 0.0

                    for eb_idx in eval_indices[eval_idx]:
                        vx, vy = make_batch(eb_idx, val_data)

                        if has_injection:
                            eval_cb = make_cerebellar_fn(
                                fwd_model, gate, condition)
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
                              f"D={delta:+.4f} | cos={val_cos:.4f} "
                              f"gate={gn:.4f}")
                    else:
                        print(f"  [{label}] step {step:6d}: "
                              f"lm={val_lm:.4f} | cos={val_cos:.4f}")

                eval_idx += 1

        save_dir = os.path.join(save_root, condition)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(fwd_model.state_dict(),
                    os.path.join(save_dir, "fwd_model.pt"))
        if has_injection:
            torch.save(gate.state_dict(), os.path.join(save_dir, "gate.pt"))
        print(f"  Saved checkpoint to {save_dir}")

        return model, fwd_model, gate, history

    # =========================================================
    # Run all conditions
    # =========================================================
    models = {}
    fwd_models = {}
    gates = {}
    histories = {}

    for cond in CONDITIONS:
        m, f, g, h = train_run(cond)
        models[cond] = m
        fwd_models[cond] = f
        gates[cond] = g
        histories[cond] = h

    # =========================================================
    # Collect activations for analysis
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  COLLECTING PROBE DATA")
    print(f"{'='*60}")

    def collect_probe_data(model, fwd_model):
        model.eval()
        fwd_model.eval()
        acts = {k: [] for k in layer_keys}
        residual_vecs = []
        post_block0_list = []

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
                post_block0_list.append(
                    vi["post_block0"].reshape(-1, n_embd).cpu())

        acts = {k: torch.cat(v) for k, v in acts.items()}
        return {
            "acts": acts,
            "residual_vec": torch.cat(residual_vecs),
            "post_block0": torch.cat(post_block0_list),
        }

    probe_data = {}
    for cond in CONDITIONS:
        print(f"  Collecting {cond}...")
        probe_data[cond] = collect_probe_data(models[cond], fwd_models[cond])

    n_samples = probe_data["open_loop"]["residual_vec"].shape[0]
    n_train_probe = int(0.8 * n_samples)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = perm[:n_train_probe]
    test_idx = perm[n_train_probe:]
    print(f"  Probe dataset: {n_samples} samples "
          f"({n_train_probe} train, {n_samples - n_train_probe} test)")

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
        return float(1.0 - mse / var) if var > 0 else 0.0

    # =========================================================
    # Analysis 1: Standard self-knowledge probes
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 1: SELF-KNOWLEDGE PROBES")
    print(f"{'='*60}")

    sk_results = {}
    print(f"\n{'layer':>12s}", end="")
    for cond in CONDITIONS:
        print(f"  {cond:>12s}", end="")
    print()
    print("-" * (14 + 14 * len(CONDITIONS)))

    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            r2 = vector_probe(
                probe_data[cond]["acts"][lk],
                probe_data[cond]["residual_vec"])
            if cond not in sk_results:
                sk_results[cond] = {}
            sk_results[cond][lk] = r2
            print(f"  {r2:>12.4f}", end="")
        print()

    print(f"\n--- Delta R^2 vs open_loop ---")
    print(f"{'layer':>12s}", end="")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        print(f"  D {cond:>9s}", end="")
    print()
    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            if cond == "open_loop":
                continue
            delta = sk_results[cond][lk] - sk_results["open_loop"][lk]
            print(f"  {delta:>+12.4f}", end="")
        print()

    # =========================================================
    # Analysis 2: Subspace-specific analysis
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 2: SUBSPACE STRUCTURE (k={subspace_k})")
    print(f"{'='*60}")

    Q_cpu = Q_t.cpu()
    Q_perp_cpu = Q_perp_t.cpu()

    # 2a. Variance fraction in subspace at each layer
    expected_frac = subspace_k / n_embd
    print(f"\n--- Variance fraction in subspace "
          f"(expected random: {expected_frac:.4f}) ---")
    print(f"{'layer':>12s}", end="")
    for cond in CONDITIONS:
        print(f"  {cond:>12s}", end="")
    print()

    var_fracs = {}
    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            acts = probe_data[cond]["acts"][lk].float()
            acts_c = acts - acts.mean(dim=0)
            sub_var = (acts_c @ Q_cpu).var(dim=0).sum().item()
            total_var = acts_c.var(dim=0).sum().item()
            frac = sub_var / total_var if total_var > 0 else 0.0
            if cond not in var_fracs:
                var_fracs[cond] = {}
            var_fracs[cond][lk] = frac
            print(f"  {frac:>12.4f}", end="")
        print()

    # 2b. Subspace probes: predict P@a_0 from each layer
    print(f"\n--- Subspace probes (R^2: layer acts -> P@a_0, "
          f"{subspace_k}-d target) ---")
    print(f"{'layer':>12s}", end="")
    for cond in CONDITIONS:
        print(f"  {cond:>12s}", end="")
    print()

    sub_probe_results = {}
    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            a0 = probe_data[cond]["post_block0"].float()
            target_sub = (a0 @ Q_cpu).numpy()
            r2 = vector_probe(probe_data[cond]["acts"][lk], target_sub)
            if cond not in sub_probe_results:
                sub_probe_results[cond] = {"subspace": {}, "complement": {}}
            sub_probe_results[cond]["subspace"][lk] = r2
            print(f"  {r2:>12.4f}", end="")
        print()

    # 2c. Complement probes: predict P_perp@a_0 from each layer
    print(f"\n--- Complement probes (R^2: layer acts -> P_perp@a_0, "
          f"{n_embd - subspace_k}-d target) ---")
    print(f"{'layer':>12s}", end="")
    for cond in CONDITIONS:
        print(f"  {cond:>12s}", end="")
    print()

    for lk in layer_keys:
        print(f"{lk:>12s}", end="")
        for cond in CONDITIONS:
            a0 = probe_data[cond]["post_block0"].float()
            target_comp = (a0 @ Q_perp_cpu).numpy()
            r2 = vector_probe(probe_data[cond]["acts"][lk], target_comp)
            sub_probe_results[cond]["complement"][lk] = r2
            print(f"  {r2:>12.4f}", end="")
        print()

    # 2d. Subspace/complement probe gap (lowrank - open)
    print(f"\n--- Subspace vs complement probe gap "
          f"(lowrank_proj - open_loop) ---")
    print(f"{'layer':>12s} {'D sub':>10s} {'D comp':>10s} {'sub-comp':>10s}")
    for lk in layer_keys:
        d_sub = (sub_probe_results["lowrank_proj"]["subspace"][lk] -
                 sub_probe_results["open_loop"]["subspace"][lk])
        d_comp = (sub_probe_results["lowrank_proj"]["complement"][lk] -
                  sub_probe_results["open_loop"]["complement"][lk])
        print(f"{lk:>12s} {d_sub:>+10.4f} {d_comp:>+10.4f} "
              f"{d_sub - d_comp:>+10.4f}")

    # =========================================================
    # Analysis 3: Anisotropic perturbation
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  ANALYSIS 3: ANISOTROPIC PERTURBATION")
    print(f"  (s={perturb_s}, {n_perturb_dirs} dirs/type, "
          f"{perturb_eval_batches} batches)")
    print(f"{'='*60}")

    rng_perturb = np.random.default_rng(seed + 600)

    sub_dirs = []
    for _ in range(n_perturb_dirs):
        c = rng_perturb.standard_normal(subspace_k).astype(np.float32)
        d = Q @ c
        d /= np.linalg.norm(d)
        sub_dirs.append(d)

    comp_dirs = []
    for _ in range(n_perturb_dirs):
        c = rng_perturb.standard_normal(
            n_embd - subspace_k).astype(np.float32)
        d = Q_perp @ c
        d /= np.linalg.norm(d)
        comp_dirs.append(d)

    # Verify orthogonality
    sub_in_sub = np.mean([float(np.abs(Q.T @ d).sum()) for d in sub_dirs])
    comp_in_sub = np.mean([float(np.abs(Q.T @ d).sum()) for d in comp_dirs])
    print(f"  Subspace dirs: mean |Q^T d| = {sub_in_sub:.4f} "
          f"(should be ~{np.sqrt(subspace_k):.2f})")
    print(f"  Complement dirs: mean |Q^T d| = {comp_in_sub:.6f} "
          f"(should be ~0)")

    with torch.no_grad():
        x_ref, y_ref = make_batch(eval_indices[0][0], val_data)
        _, _, vi_ref = models["open_loop"](
            x_ref, y_ref, return_intermediates=True)
        b1_std = float(vi_ref["post_block1"].std())
    print(f"  post_block1 activation std: {b1_std:.4f}")

    perturb_gen = torch.Generator().manual_seed(seed + 400)
    perturb_batch_indices = [
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

    def measure_perturbation_set(model, dirs, label):
        model.eval()
        loss_deltas = []
        response_norms = []

        with torch.no_grad():
            for d_np in dirs:
                dir_t = torch.from_numpy(d_np).to(device)
                perturbation = (perturb_s * b1_std) * dir_t

                ld_batch, rn_batch = [], []
                for p_idx in perturb_batch_indices:
                    x, y = make_batch(p_idx, val_data)
                    tgt = x[:, 1:]

                    logits_base, _, inter_base = run_with_perturbation(
                        model, x, y)
                    loss_base = -F.log_softmax(
                        logits_base[:, :-1], dim=-1
                    ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                    logits_pert, _, inter_pert = run_with_perturbation(
                        model, x, y, perturbation)
                    loss_pert = -F.log_softmax(
                        logits_pert[:, :-1], dim=-1
                    ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

                    response = (inter_pert["post_block3"][:, :-1] -
                                inter_base["post_block3"][:, :-1])
                    rn_batch.append(
                        response.reshape(-1, n_embd).norm(dim=-1)
                        .mean().item())
                    ld_batch.append((loss_pert - loss_base).mean().item())

                loss_deltas.append(np.mean(ld_batch))
                response_norms.append(np.mean(rn_batch))

        return {
            "loss_delta_mean": float(np.mean(loss_deltas)),
            "loss_delta_std": float(np.std(loss_deltas)),
            "loss_delta_sem": float(np.std(loss_deltas) /
                                     np.sqrt(len(loss_deltas))),
            "response_norm_mean": float(np.mean(response_norms)),
            "response_norm_std": float(np.std(response_norms)),
            "per_dir": [float(x) for x in loss_deltas],
        }

    aniso_results = {}
    for cond in CONDITIONS:
        print(f"\n  [{cond.upper()}]")
        sub_r = measure_perturbation_set(models[cond], sub_dirs, "subspace")
        comp_r = measure_perturbation_set(models[cond], comp_dirs, "complement")
        aniso_results[cond] = {"subspace": sub_r, "complement": comp_r}

        ratio = (sub_r["loss_delta_mean"] / comp_r["loss_delta_mean"]
                 if abs(comp_r["loss_delta_mean"]) > 1e-10
                 else float("nan"))
        print(f"    Subspace Dloss:    {sub_r['loss_delta_mean']:+.6f} "
              f"(+/-{sub_r['loss_delta_sem']:.6f})")
        print(f"    Complement Dloss:  {comp_r['loss_delta_mean']:+.6f} "
              f"(+/-{comp_r['loss_delta_sem']:.6f})")
        print(f"    Sub/Comp ratio:    {ratio:.4f}")
        print(f"    Subspace ||R||:    {sub_r['response_norm_mean']:.4f}")
        print(f"    Complement ||R||:  {comp_r['response_norm_mean']:.4f}")

    print(f"\n--- Anisotropic perturbation summary ---")
    print(f"{'Condition':>12s} {'Sub Dl':>9s} {'Comp Dl':>9s} "
          f"{'Sub/Comp':>9s} {'Sub ||R||':>9s} {'Comp ||R||':>10s}")
    print("-" * 62)
    for cond in CONDITIONS:
        s = aniso_results[cond]["subspace"]
        c = aniso_results[cond]["complement"]
        ratio = (s["loss_delta_mean"] / c["loss_delta_mean"]
                 if abs(c["loss_delta_mean"]) > 1e-10
                 else float("nan"))
        print(f"{cond:>12s} {s['loss_delta_mean']:>+9.5f} "
              f"{c['loss_delta_mean']:>+9.5f} "
              f"{ratio:>9.4f} "
              f"{s['response_norm_mean']:>9.4f} "
              f"{c['response_norm_mean']:>10.4f}")

    # Anisotropy index: (comp - sub) / (comp + sub) for each condition
    # Positive = more robust in subspace, negative = less robust
    print(f"\n--- Anisotropy index: (comp_Dl - sub_Dl)/(comp_Dl + sub_Dl) ---")
    print(f"  Positive = more robust in subspace directions")
    for cond in CONDITIONS:
        s_dl = aniso_results[cond]["subspace"]["loss_delta_mean"]
        c_dl = aniso_results[cond]["complement"]["loss_delta_mean"]
        denom = abs(s_dl) + abs(c_dl)
        ai = (c_dl - s_dl) / denom if denom > 1e-10 else 0.0
        print(f"    {cond:>12s}: {ai:+.4f}")

    # =========================================================
    # Final LM loss comparison
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  FINAL LM LOSS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':>12s} {'val_lm':>8s} {'no_inj':>8s} "
          f"{'Reliance':>9s} {'Gate norm':>10s}")
    print("-" * 55)

    for cond in CONDITIONS:
        h = histories[cond]
        val_lm = h["val_lm"][-1][1]
        val_noinj = h["val_lm_no_inj"][-1][1]
        reliance = (val_lm - val_noinj
                    if cond != "open_loop" else 0.0)
        gn = (h["gate_norm"][-1][1]
              if cond != "open_loop" else 0.0)
        print(f"{cond:>12s} {val_lm:>8.4f} {val_noinj:>8.4f} "
              f"{reliance:>+9.4f} {gn:>10.4f}")

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
            "subspace_k": subspace_k,
            "n_perturb_dirs": n_perturb_dirs, "perturb_s": perturb_s,
            "conditions": CONDITIONS,
        },
        "lm_loss": {},
        "sk_probes": sk_results,
        "subspace_probes": sub_probe_results,
        "variance_fractions": var_fracs,
        "anisotropic_perturbation": aniso_results,
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

    # Save the subspace directions for reproducibility
    np.save(os.path.join(save_root, "Q.npy"), Q)
    np.save(os.path.join(save_root, "Q_perp.npy"), Q_perp)

    volume.commit()
    print(f"\nAll results saved to {save_root}")

    # =========================================================
    # Executive summary
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  EXECUTIVE SUMMARY")
    print(f"{'='*60}")

    ol_final = histories["open_loop"]["val_lm"][-1][1]
    print("\n1. LM loss:")
    for cond in CONDITIONS:
        h = histories[cond]
        val = h["val_lm"][-1][1]
        print(f"    {cond:>12s}: {val:.4f} "
              f"(vs OL: {val - ol_final:+.4f})")

    print("\n2. Self-knowledge (mean D R^2 vs open_loop):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        mean_d = np.mean([
            sk_results[cond][lk] - sk_results["open_loop"][lk]
            for lk in layer_keys
        ])
        print(f"    {cond:>12s}: {mean_d:+.4f}")

    print(f"\n3. Variance in subspace (expected: {expected_frac:.4f}):")
    for cond in CONDITIONS:
        mean_f = np.mean([var_fracs[cond][lk] for lk in layer_keys])
        print(f"    {cond:>12s}: {mean_f:.4f} "
              f"({mean_f/expected_frac:.2f}x expected)")

    print("\n4. Anisotropic perturbation:")
    for cond in CONDITIONS:
        s = aniso_results[cond]["subspace"]
        c = aniso_results[cond]["complement"]
        ratio = (s["loss_delta_mean"] / c["loss_delta_mean"]
                 if abs(c["loss_delta_mean"]) > 1e-10
                 else float("nan"))
        print(f"    {cond:>12s}: sub/comp = {ratio:.4f} "
              f"(sub={s['loss_delta_mean']:+.5f}, "
              f"comp={c['loss_delta_mean']:+.5f})")

    print(f"\n5. Subspace probe gap (lowrank_proj - open_loop):")
    for lk in layer_keys:
        d_sub = (sub_probe_results["lowrank_proj"]["subspace"][lk] -
                 sub_probe_results["open_loop"]["subspace"][lk])
        d_comp = (sub_probe_results["lowrank_proj"]["complement"][lk] -
                  sub_probe_results["open_loop"]["complement"][lk])
        print(f"    {lk}: D_sub={d_sub:+.4f}, D_comp={d_comp:+.4f}")

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
    subspace_k: int = 16,
):
    result = a2a_subspace_injection.remote(
        n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
        lr=lr,
        predict_from=predict_from, predict_to=predict_to,
        inject_after_block=inject_after_block,
        fwd_n_layer=fwd_n_layer,
        fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult,
        subspace_k=subspace_k,
    )
    print("\n=== SUBSPACE INJECTION COMPLETE ===")

    import numpy as np
    layer_keys = [f"post_block{i}" for i in range(4)]

    print("\nLM loss:")
    for cond in CONDITIONS:
        lm = result["lm_loss"][cond]
        print(f"  {cond:>12s}: val={lm['final_val_lm']:.4f} "
              f"no_inj={lm['final_val_lm_no_inj']:.4f}")

    print(f"\nSelf-knowledge (D R^2 vs open_loop):")
    for cond in CONDITIONS:
        if cond == "open_loop":
            continue
        mean_d = np.mean([
            result["sk_probes"][cond][lk]
            - result["sk_probes"]["open_loop"][lk]
            for lk in layer_keys
        ])
        print(f"  {cond:>12s}: {mean_d:+.4f}")

    print(f"\nAnisotropy (sub/comp Dloss ratio):")
    for cond in CONDITIONS:
        a = result["anisotropic_perturbation"][cond]
        s = a["subspace"]["loss_delta_mean"]
        c = a["complement"]["loss_delta_mean"]
        ratio = s / c if abs(c) > 1e-10 else float("nan")
        print(f"  {cond:>12s}: {ratio:.4f}")
