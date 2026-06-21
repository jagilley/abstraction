"""RHM regime transition: L-regime → m-regime as the model learns.

Tests whether the FM residual transitions from diffuse (L-regime) to
rule-conditioned (m-regime) as the main model learns the DGP. Three
settings at fixed L=4, varying m (2, 4, 8) span easy→hard:

- m=2 (easy/MNIST-like): model learns well → expect full transition,
  eta² rises as residual becomes rule-discriminative
- m=4 (medium): model partially learns → expect partial transition
- m=8 (hard/language-like): model barely learns → expect no transition,
  residual stays diffuse, eta² stays low

At ~11 training checkpoints per setting: freeze model, train fresh FM on
its frozen activations, compute FM residual's eta² w.r.t. the RHM's
hierarchical rule/feature structure (ground truth from the DGP).

The "regime" distinction:
- L-regime: model is still learning compositional structure. FM errors
  are about depth-of-computation, diffuse across dimensions. Gate would
  close (nothing structured to discriminate). Analogous to language.
- m-regime: model has learned the rules, FM misses which-of-m-rules was
  used. Residual is low-rank, rule-discriminative. Gate would open
  selectively. Analogous to MNIST.
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
    )
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-regime-transition", image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_with_traces(rules, n_sequences, seed=999):
    """Generate RHM sequences with full ancestry trace.

    Returns:
        sequences: (n_seq, s^L) int64
        level_features: list of L+1 arrays.
            level_features[ell] shape (n_seq, s^ell) = features at level ell.
            level_features[L] == sequences (leaf tokens).
        level_rules: list of L arrays.
            level_rules[ell] shape (n_seq, s^ell) = rule choices at level ell.

    Ancestry lookup for token at position p:
        ancestor_idx = p // s^(L - ell)
        feature = level_features[ell][seq, ancestor_idx]
        rule    = level_rules[ell][seq, ancestor_idx]
    """
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
    """Between-group sum of squares for eta² computation."""
    import numpy as np
    ss = 0.0
    for val in np.unique(labels):
        mask = labels == val
        group_mean = data[mask].mean(axis=0)
        ss += float(mask.sum()) * float(np.sum((group_mean - grand_mean) ** 2))
    return ss


def _compute_hierarchy_eta2(residuals_np, level_features, level_rules, s, L):
    """Compute eta² of FM residual w.r.t. features/rules at each hierarchy level.

    Computes two variants per level:
    - Aggregate: pooled across all sequence positions
    - Last-position: only the final token (maximum causal context)

    The last-position variant should be strictly stronger because the
    autoregressive model has seen the full sequence at that point.
    """
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
                if ss_total_last > 1e-12 else 0.0
            ),
            "feature_eta2_last": (
                _eta2_between(res_last, feat_labels_last, last_mean) / ss_total_last
                if ss_total_last > 1e-12 else 0.0
            ),
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
    from rhm.rhm import make_corpus

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
    timeout=7200,
    memory=16384,
)
def train_with_checkpoints(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4,
    seed: int = 42,
    eval_interval: int = 200, n_eval_batches: int = 10,
):
    """Train main model, saving checkpoints at ~11 log-spaced intervals."""
    import torch
    import numpy as np
    from rhm.model import GPT

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

    save_dir = f"{DATA_DIR}/{key}/regime_transition"
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Training: {key}, {n_layer}L/{n_head}H/{n_embd}D ===")
    print(f"  n_steps={n_steps}, epochs={n_steps/steps_per_epoch:.1f}, "
          f"checkpoints at: {checkpoint_steps}")

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
        "n_steps": n_steps, "n_params": n_params,
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
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    checkpoint_path: str = "",
    checkpoint_step: int = 0,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    fwd_n_layer: int = 1, fwd_d_head: int = 32, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
    fm_train_steps: int = 5000,
    fm_lr: float = 1e-3,
    n_eval_sequences: int = 10000,
    batch_size: int = 64,
    fm_seed: int = 137,
):
    """At one checkpoint: train FM on frozen activations, compute metrics + hierarchy eta²."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(fm_seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"

    volume.reload()

    # --- Load frozen main model ---
    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Load training corpus for FM training ---
    corpus = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    corpus = torch.from_numpy(corpus[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data = corpus[:split]
    val_data = corpus[split:]

    # --- Evaluate main model val_loss (on corpus val split, same as training) ---
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

    # --- Train FM on frozen model activations ---
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())

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

    # --- Generate eval sequences with traces ---
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(
        rules, n_eval_sequences,
    )

    # --- Compute FM residuals on traced eval sequences ---
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

    # --- Effective rank ---
    res_flat = residuals_np.reshape(-1, n_embd)
    eff_rank, top1_var = _compute_effective_rank(res_flat, n_embd)

    # --- Hierarchy eta² ---
    trimmed_features = [lf[:n_complete] for lf in level_features]
    trimmed_rules = [lr[:n_complete] for lr in level_rules]

    eta2_results = _compute_hierarchy_eta2(
        residuals_np, trimmed_features, trimmed_rules, s=s, L=L,
    )

    result = {
        "step": checkpoint_step,
        "main_val_loss": float(main_val_loss),
        "fm_cosine": float(mean_cosine),
        "residual_norm": float(mean_res_norm),
        "effective_rank": float(eff_rank),
        "effective_rank_pct": float(eff_rank / n_embd * 100),
        "top1_pc_variance": float(top1_var),
        "fm_params": fwd_n_params,
        "hierarchy_eta2": eta2_results,
    }

    print(f"  Step {checkpoint_step}: val={main_val_loss:.4f} cos={mean_cosine:.4f} "
          f"res={mean_res_norm:.4f} rank={eff_rank:.1f}/{n_embd}")
    for ell in range(L):
        e = eta2_results[f"level_{ell}"]
        print(f"    L{ell}: rule={e['rule_eta2']:.4f} feat={e['feature_eta2']:.4f} "
              f"rule_last={e['rule_eta2_last']:.4f} feat_last={e['feature_eta2_last']:.4f} "
              f"(subtree={e['tokens_per_subtree']})")

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    timeout=21600,
    memory=4096,
)
def rhm_regime_transition(
    v: int = 8, s: int = 2, depth: int = 4,
    m_values: str = "2,4,8",
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    fwd_n_layer: int = 1, fwd_d_head: int = 32, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
    fm_train_steps: int = 5000,
    n_eval_sequences: int = 10000,
):
    """Run regime transition experiment across m-values."""
    ms = [int(x) for x in m_values.split(",")]
    L = depth

    print(f"=== RHM Regime Transition: L={L}, m in {ms} ===")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D, v={v}, s={s}")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d, "
          f"gap: {predict_from} -> {predict_to}")

    # Phase 1: train main models in parallel
    print("\n--- Phase 1: Training main models ---")
    train_handles = {}
    for mi in ms:
        h = train_with_checkpoints.spawn(
            v=v, s=s, depth=depth, m=mi, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        )
        train_handles[mi] = h

    train_results = {}
    for mi, h in train_handles.items():
        train_results[mi] = h.get()
        tr = train_results[mi]
        print(f"  m={mi}: {len(tr['checkpoints'])} checkpoints, "
              f"final val={tr['checkpoints'][-1]['val_loss']:.4f}")

    # Phase 2: analyze all checkpoints in parallel
    print("\n--- Phase 2: Analyzing checkpoints ---")
    analysis_handles = []
    for mi, tr in train_results.items():
        for ckpt in tr["checkpoints"]:
            h = analyze_checkpoint.spawn(
                v=v, s=s, depth=depth, m=mi, n_tokens=n_tokens,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                checkpoint_path=ckpt["path"],
                checkpoint_step=ckpt["step"],
                predict_from=predict_from, predict_to=predict_to,
                fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
                fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
                fm_train_steps=fm_train_steps,
                n_eval_sequences=n_eval_sequences,
            )
            analysis_handles.append((mi, ckpt["step"], h))

    print(f"  Spawned {len(analysis_handles)} analysis jobs")

    all_results = {}
    for mi, step, h in analysis_handles:
        r = h.get()
        if mi not in all_results:
            all_results[mi] = []
        all_results[mi].append(r)

    for mi in all_results:
        all_results[mi].sort(key=lambda x: x["step"])

    # --- Summary tables ---
    print(f"\n{'='*140}")
    print(f"RHM REGIME TRANSITION: L={L}, v={v}, s={s}")
    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d")
    print(f"{'='*140}")

    for mi in ms:
        results = all_results[mi]
        label = "(easy)" if mi == min(ms) else "(hard)" if mi == max(ms) else "(medium)"
        print(f"\n--- m={mi} {label} ---")

        header = (f"{'step':>6}  {'val_loss':>8}  {'cos':>6}  {'res':>7}  {'rank':>6}  "
                  + "  ".join(f"{'rL'+str(e):>7}" for e in range(L))
                  + "  "
                  + "  ".join(f"{'fL'+str(e):>7}" for e in range(L))
                  + "  "
                  + "  ".join(f"{'rL'+str(e)+'*':>7}" for e in range(L))
                  + "  "
                  + "  ".join(f"{'fL'+str(e)+'*':>7}" for e in range(L)))
        print(header)
        print("-" * len(header))

        for r in results:
            eta = r["hierarchy_eta2"]
            rule_agg = "  ".join(
                f"{eta[f'level_{e}']['rule_eta2']:>7.4f}" for e in range(L))
            feat_agg = "  ".join(
                f"{eta[f'level_{e}']['feature_eta2']:>7.4f}" for e in range(L))
            rule_last = "  ".join(
                f"{eta[f'level_{e}']['rule_eta2_last']:>7.4f}" for e in range(L))
            feat_last = "  ".join(
                f"{eta[f'level_{e}']['feature_eta2_last']:>7.4f}" for e in range(L))
            print(f"{r['step']:>6}  {r['main_val_loss']:>8.4f}  "
                  f"{r['fm_cosine']:>6.4f}  {r['residual_norm']:>7.4f}  "
                  f"{r['effective_rank']:>6.1f}  "
                  f"{rule_agg}  {feat_agg}  {rule_last}  {feat_last}")

    # Save
    save_dir = f"{DATA_DIR}/rhm_regime_transition"
    os.makedirs(save_dir, exist_ok=True)

    full_results = {
        "experiment": "rhm_regime_transition",
        "params": {
            "v": v, "s": s, "L": L, "m_values": ms,
            "n_tokens": n_tokens,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d",
            "gap": f"{predict_from}->{predict_to}",
            "fm_train_steps": fm_train_steps,
            "n_eval_sequences": n_eval_sequences,
        },
        "training": {str(mi): train_results[mi]["history"] for mi in ms},
        "analysis": {str(mi): all_results[mi] for mi in ms},
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/results.json")
    return full_results
