"""RHM regime trajectory: track FM residual structure over training.

Two experiments to find the L→m transition:
1. m=3 at L=6 with current model (4L/4H/128D, 0.8M) — intermediate m
2. m=4 at L=6 with scaled model (6L/6H/192D, ~2.7M) — bigger model to learn harder DGP

At each of ~11 training checkpoints, train a fresh FM and measure:
cosine, effective rank, top1 PC, and hierarchy-conditioned eta².
FM sizes chosen at ~14% of gap capacity (matching MNIST ratio).
"""

import json
import os
import modal

from language_reduction_synthetic.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
    )
    .add_local_python_source("language_reduction_synthetic")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-regime-trajectory", image=image)


# ---------------------------------------------------------------------------
# Helpers (from rhm_regime_transition.py)
# ---------------------------------------------------------------------------

def _generate_with_traces(rules, n_sequences, seed=999):
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features = []
    level_rules = []
    current = rng.integers(0, v, size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        level_rules.append(rc.copy())
        next_level = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            next_level[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = next_level
    sequences = current
    level_features.append(sequences.copy())
    return sequences, level_features, level_rules


def _eta2_between(data, labels, grand_mean):
    import numpy as np
    ss = 0.0
    for val in np.unique(labels):
        mask = labels == val
        group_mean = data[mask].mean(axis=0)
        ss += float(mask.sum()) * float(np.sum((group_mean - grand_mean) ** 2))
    return ss


def _compute_hierarchy_eta2(residuals_np, level_features, level_rules, s, L):
    import numpy as np
    n_seq, seq_len, d_model = residuals_np.shape
    res_flat = residuals_np.reshape(-1, d_model)
    grand_mean = res_flat.mean(axis=0)
    ss_total = float(np.sum((res_flat - grand_mean) ** 2))
    res_last = residuals_np[:, -1, :]
    last_mean = res_last.mean(axis=0)
    ss_total_last = float(np.sum((res_last - last_mean) ** 2))
    if ss_total < 1e-12:
        return {f"level_{ell}": {
            "rule_eta2": 0.0, "feature_eta2": 0.0,
            "rule_eta2_last": 0.0, "feature_eta2_last": 0.0,
            "tokens_per_subtree": s ** (L - ell),
        } for ell in range(L)}
    results = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        ancestor_idx_all = np.arange(seq_len) // s_power
        rule_labels_all = level_rules[ell][:, ancestor_idx_all].reshape(-1)
        feat_labels_all = level_features[ell][:, ancestor_idx_all].reshape(-1)
        last_ancestor = (seq_len - 1) // s_power
        rule_labels_last = level_rules[ell][:, last_ancestor]
        feat_labels_last = level_features[ell][:, last_ancestor]
        results[f"level_{ell}"] = {
            "rule_eta2": _eta2_between(res_flat, rule_labels_all, grand_mean) / ss_total,
            "feature_eta2": _eta2_between(res_flat, feat_labels_all, grand_mean) / ss_total,
            "rule_eta2_last": (
                _eta2_between(res_last, rule_labels_last, last_mean) / ss_total_last
                if ss_total_last > 1e-12 else 0.0),
            "feature_eta2_last": (
                _eta2_between(res_last, feat_labels_last, last_mean) / ss_total_last
                if ss_total_last > 1e-12 else 0.0),
            "tokens_per_subtree": int(s_power),
        }
    return results


def _compute_effective_rank(res_flat, n_embd, max_samples=50000):
    import numpy as np
    n = min(max_samples, res_flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(res_flat.shape[0], n, replace=False)
    sub = res_flat[idx] - res_flat[idx].mean(axis=0)
    _, S, _ = np.linalg.svd(sub, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))
    top1_var = float((S[0] ** 2) / (S ** 2).sum())
    return eff_rank, top1_var


def _ensure_corpus(v, s, L, m, n_tokens):
    import numpy as np
    from language_reduction_synthetic.rhm import make_corpus
    key = setting_key(v, s, L, m)
    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    if os.path.exists(corpus_path):
        existing = np.load(corpus_path)
        if len(existing) >= n_tokens:
            return
    corpus, rules, meta = make_corpus(v, s, L, m, n_tokens)
    out_dir = f"{DATA_DIR}/{key}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "corpus.npy"), corpus)
    for ell, r in enumerate(rules):
        np.save(os.path.join(out_dir, f"rules_L{ell}.npy"), r)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()


# ---------------------------------------------------------------------------
# Phase 1: Train main model with checkpoints
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=10800,
    memory=16384,
)
def train_with_checkpoints(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 3,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4,
    seed: int = 42,
    eval_interval: int = 200, n_eval_batches: int = 10,
):
    import torch
    import numpy as np
    from language_reduction_synthetic.model import GPT

    torch.manual_seed(seed)
    np.random.seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"

    volume.reload()
    _ensure_corpus(v, s, L, m, n_tokens)

    data = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    checkpoint_fracs = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0]
    checkpoint_steps = sorted(set(
        round(f * n_steps / eval_interval) * eval_interval
        for f in checkpoint_fracs
    ))

    model_tag = f"{n_layer}L{n_head}H{n_embd}D"
    save_dir = f"{DATA_DIR}/{key}/regime_trajectory_{model_tag}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Training: {key}, {model_tag} ===")
    print(f"  seq_len={seq_len}, n_steps={n_steps}, epochs={n_steps/steps_per_epoch:.1f}")
    print(f"  checkpoints at: {checkpoint_steps}")

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    history = {"train_loss": [], "val_loss": []}
    saved_checkpoints = []

    for step in range(n_steps + 1):
        if step > 0:
            model.train()
            x, y = get_batch(train_data)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                train_loss = float(model(*get_batch(train_data))[1])
                val_loss = sum(float(model(*get_batch(val_data))[1])
                               for _ in range(n_eval_batches)) / n_eval_batches
            history["train_loss"].append((step, train_loss))
            history["val_loss"].append((step, val_loss))

            if step in checkpoint_steps:
                ckpt_path = os.path.join(save_dir, f"ckpt_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                saved_checkpoints.append({
                    "step": step, "val_loss": val_loss, "path": ckpt_path,
                })
                print(f"  step {step:6d}: val={val_loss:.4f} [CHECKPOINT]")
            elif step % (eval_interval * 10) == 0:
                print(f"  step {step:6d}: val={val_loss:.4f}")

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "model_tag": model_tag, "n_steps": n_steps, "n_params": n_params,
        "checkpoint_dir": save_dir,
        "checkpoints": saved_checkpoints,
        "history": history,
    }
    with open(os.path.join(save_dir, "training_info.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"  Saved {len(saved_checkpoints)} checkpoints to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Phase 2: Analyze each checkpoint
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def analyze_checkpoint(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 3,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    checkpoint_path: str = "",
    checkpoint_step: int = 0,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fwd_n_layer: int = 2, fwd_d_head: int = 16, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 1.0,
    fm_train_steps: int = 5000,
    fm_lr: float = 1e-3,
    n_eval_sequences: int = 10000,
    batch_size: int = 64,
    fm_seed: int = 137,
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from language_reduction_synthetic.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(fm_seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"

    volume.reload()

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    corpus = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    corpus = torch.from_numpy(corpus[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data = corpus[:split]
    val_data = corpus[split:]

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    with torch.no_grad():
        main_val_loss = sum(
            float(model(*get_batch(val_data))[1]) for _ in range(20)
        ) / 20

    print(f"=== Analyzing: {key}, step {checkpoint_step}, val_loss={main_val_loss:.4f} ===")

    # --- Train FM ---
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())

    main_block_params = sum(p.numel() for p in list(model.transformer.h[0].parameters()))
    gap_blocks = int(predict_to.replace("post_block", "")) - int(predict_from.replace("post_block", ""))
    gap_params = gap_blocks * main_block_params
    capacity_ratio = fwd_n_params / gap_params if gap_params > 0 else 0

    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fm_lr, weight_decay=0.01)

    for fm_step in range(fm_train_steps):
        fwd_model.train()
        x, y = get_batch(train_data)
        with torch.no_grad():
            _, _, intermediates = model(x, y, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
        predicted = fwd_model(source)
        fwd_loss = F.mse_loss(predicted, target)
        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_fwd.step()
        if fm_step % 2000 == 0:
            print(f"  FM step {fm_step}: mse={float(fwd_loss):.6f}")

    # --- Eval with traced sequences ---
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)

    fwd_model.eval()
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    all_residuals = []
    all_cosines = []
    all_res_norms = []

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch_x = eval_data[i:i + batch_size]
            if batch_x.shape[0] < 2:
                continue
            _, _, intermediates = model(batch_x, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
            predicted = fwd_model(source)
            residual = target - predicted
            cosine = F.cosine_similarity(predicted, target, dim=-1)
            all_residuals.append(residual.cpu().numpy())
            all_cosines.append(float(cosine.mean()))
            all_res_norms.append(float(residual.norm(dim=-1).mean()))

    residuals_np = np.concatenate(all_residuals, axis=0)
    n_complete = residuals_np.shape[0]
    mean_cosine = float(np.mean(all_cosines))
    mean_res_norm = float(np.mean(all_res_norms))

    res_flat = residuals_np.reshape(-1, n_embd)
    eff_rank, top1_var = _compute_effective_rank(res_flat, n_embd)

    trimmed_features = [lf[:n_complete] for lf in level_features]
    trimmed_rules = [lr[:n_complete] for lr in level_rules]
    eta2_results = _compute_hierarchy_eta2(residuals_np, trimmed_features, trimmed_rules, s=s, L=L)

    result = {
        "step": checkpoint_step,
        "main_val_loss": float(main_val_loss),
        "fm_cosine": float(mean_cosine),
        "residual_norm": float(mean_res_norm),
        "effective_rank": float(eff_rank),
        "effective_rank_pct": float(eff_rank / n_embd * 100),
        "top1_pc_variance": float(top1_var),
        "fm_params": fwd_n_params,
        "capacity_ratio_pct": float(capacity_ratio * 100),
        "hierarchy_eta2": eta2_results,
    }

    print(f"  Step {checkpoint_step}: val={main_val_loss:.4f} cos={mean_cosine:.4f} "
          f"res={mean_res_norm:.4f} rank={eff_rank:.1f}/{n_embd} ({eff_rank/n_embd*100:.1f}%)")
    for ell in range(L):
        e = eta2_results[f"level_{ell}"]
        print(f"    L{ell}: rule={e['rule_eta2']:.4f} feat={e['feature_eta2']:.4f} "
              f"rule_last={e['rule_eta2_last']:.4f} feat_last={e['feature_eta2_last']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Orchestrator for a single trajectory
# ---------------------------------------------------------------------------

def _run_trajectory(
    v, s, depth, m, n_tokens,
    n_layer, n_head, n_embd,
    predict_from, predict_to,
    fwd_n_layer, fwd_d_head, fwd_n_head, fwd_mlp_mult,
    fm_train_steps, experiment_label,
):
    L = depth
    key = setting_key(v, s, L, m)
    model_tag = f"{n_layer}L{n_head}H{n_embd}D"

    print(f"\n{'='*100}")
    print(f"EXPERIMENT: {experiment_label}")
    print(f"  Setting: {key} (seq_len={s**L})")
    print(f"  Model: {model_tag}")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}")
    print(f"  Gap: {predict_from} -> {predict_to}")
    print(f"  n_tokens: {n_tokens:,}")
    print(f"{'='*100}")

    # Phase 1
    print("\n--- Phase 1: Training main model ---")
    train_result = train_with_checkpoints.remote(
        v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    )
    print(f"  {key}: val_loss={train_result['checkpoints'][-1]['val_loss']:.4f} "
          f"({train_result['n_steps']} steps, {train_result['n_params']/1e3:.0f}K params)")

    # Phase 2
    print("\n--- Phase 2: Analyzing checkpoints ---")
    analysis_handles = []
    for ckpt in train_result["checkpoints"]:
        h = analyze_checkpoint.spawn(
            v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            checkpoint_path=ckpt["path"],
            checkpoint_step=ckpt["step"],
            predict_from=predict_from, predict_to=predict_to,
            fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
            fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
            fm_train_steps=fm_train_steps,
        )
        analysis_handles.append((ckpt["step"], h))
    print(f"  Spawned {len(analysis_handles)} analysis jobs")

    all_results = []
    for step, h in analysis_handles:
        r = h.get()
        all_results.append(r)
    all_results.sort(key=lambda x: x["step"])

    # --- Summary table ---
    print(f"\n{'='*140}")
    print(f"{experiment_label}: {key}, {model_tag}")
    print(f"FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}, "
          f"gap: {predict_from}->{predict_to}")
    print(f"{'='*140}")

    header = (f"{'step':>6}  {'val':>7}  {'cos':>7}  {'1-cos':>7}  {'res':>7}  "
              f"{'rank':>6}  {'rank%':>6}  {'top1%':>6}  "
              + "  ".join(f"{'fL'+str(e):>7}" for e in range(L))
              + "  "
              + "  ".join(f"{'fL'+str(e)+'*':>7}" for e in range(L))
              + "  "
              + "  ".join(f"{'rL'+str(e)+'*':>7}" for e in range(L)))
    print(header)
    print("-" * len(header))

    for r in all_results:
        eta = r["hierarchy_eta2"]
        feat_agg = "  ".join(
            f"{eta[f'level_{e}']['feature_eta2']:>7.4f}" for e in range(L))
        feat_last = "  ".join(
            f"{eta[f'level_{e}']['feature_eta2_last']:>7.4f}" for e in range(L))
        rule_last = "  ".join(
            f"{eta[f'level_{e}']['rule_eta2_last']:>7.4f}" for e in range(L))
        print(f"{r['step']:>6}  {r['main_val_loss']:>7.4f}  "
              f"{r['fm_cosine']:>7.4f}  {1-r['fm_cosine']:>7.4f}  "
              f"{r['residual_norm']:>7.4f}  {r['effective_rank']:>6.1f}  "
              f"{r['effective_rank_pct']:>6.1f}  {r['top1_pc_variance']*100:>6.1f}  "
              f"{feat_agg}  {feat_last}  {rule_last}")

    # Save
    save_dir = f"{DATA_DIR}/rhm_regime_trajectory"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{key}_{model_tag}.json"

    full_results = {
        "experiment": experiment_label,
        "params": {
            "v": v, "s": s, "L": L, "m": m, "n_tokens": n_tokens,
            "model": model_tag,
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}",
            "gap": f"{predict_from}->{predict_to}",
            "fm_train_steps": fm_train_steps,
        },
        "training": train_result["history"],
        "analysis": all_results,
    }
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/{fname}")
    return full_results


# ---------------------------------------------------------------------------
# Entry points (launch independently with --detach)
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=4096)
def run_m3_current():
    """m=3, L=6, current model (4L/4H/128D), matched FM (2L/1H/16d/mlp1)."""
    return _run_trajectory(
        v=8, s=2, depth=6, m=3, n_tokens=5_000_000,
        n_layer=4, n_head=4, n_embd=128,
        predict_from="post_block0", predict_to="post_block3",
        fwd_n_layer=2, fwd_d_head=16, fwd_n_head=1, fwd_mlp_mult=1.0,
        fm_train_steps=5000,
        experiment_label="m=3, L=6, 4L/4H/128D (current model)",
    )


@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=4096)
def run_m4_scaled():
    """m=4, L=6, scaled model (6L/6H/192D ~2.7M), matched FM (2L/1H/24d/mlp1)."""
    return _run_trajectory(
        v=8, s=2, depth=6, m=4, n_tokens=20_000_000,
        n_layer=6, n_head=6, n_embd=192,
        predict_from="post_block0", predict_to="post_block3",
        fwd_n_layer=2, fwd_d_head=24, fwd_n_head=1, fwd_mlp_mult=1.0,
        fm_train_steps=5000,
        experiment_label="m=4, L=6, 6L/6H/192D (scaled model)",
    )
