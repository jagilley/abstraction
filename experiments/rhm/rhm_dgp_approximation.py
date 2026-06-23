"""FM as DGP approximation: does the forward model learn the RHM composition rules?

The forward self-models paper claims FMs dissociate representation from computation.
On RHM data, the main model's computation IS hierarchical composition — applying
rules to compose level-l features from level-(l-1) features. If the FM captures
this computation, its predictions should carry the same rule-conditioned structure
as the actual layer activations.

Three measurements at each hierarchy level:
  1. eta²(actual_activations, rule_l) — ground truth: how much does the model's
     actual computation vary by which rule was used?
  2. eta²(fm_predictions, rule_l) — does the FM predict differently depending on
     which composition rule was used?
  3. eta²(source_activations, rule_l) — baseline: what rule structure is already
     in the FM's input?

The FM "approximates the DGP" to the extent that (2) ≈ (1) and (2) > (3):
the FM adds rule-conditioned structure beyond its input, matching what the model
actually computes.

Also: per-hierarchy-level cosine similarity. If the FM approximates the DGP,
it should predict best at levels the main model has learned.

Uses the trained 2.7M model (6L/6H/192D) at L=6/m=4 from the regime trajectory.
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

app = modal.App("rhm-dgp-approximation", image=image)


# ---------------------------------------------------------------------------
# Helpers
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


def _position_levels(seq_len, s):
    """s-adic valuation for positions 0..seq_len-1."""
    import numpy as np
    levels = np.zeros(seq_len, dtype=np.int64)
    for t in range(seq_len):
        if t == 0:
            levels[t] = 0
            continue
        p = t
        level = 0
        while p % s == 0:
            p //= s
            level += 1
        levels[t] = level
    return levels


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
# Main experiment
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def dgp_approximation(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    checkpoint_step: int = 20000,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fwd_n_layer: int = 2, fwd_d_head: int = 24, fwd_n_head: int = 1,
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
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(fm_seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"
    model_tag = f"{n_layer}L{n_head}H{n_embd}D"

    volume.reload()

    # --- Load main model ---
    ckpt_dir = f"{DATA_DIR}/{key}/regime_trajectory_{model_tag}"
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{checkpoint_step}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Run rhm_regime_trajectory.py::run_m4_scaled first."
        )

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded main model: {key}, {model_tag}, step {checkpoint_step}")

    # --- Load corpus for FM training ---
    _ensure_corpus(v, s, L, m, n_tokens)
    corpus = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    corpus = torch.from_numpy(corpus[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data = corpus[:split]

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        return x.to(device)

    # --- Train FM ---
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())

    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fm_lr, weight_decay=0.01)

    print(f"Training FM: {fwd_n_params/1e3:.1f}K params, {fm_train_steps} steps")
    for fm_step in range(fm_train_steps):
        fwd_model.train()
        x = get_batch(train_data)
        with torch.no_grad():
            _, _, intermediates = model(x, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
        predicted = fwd_model(source)
        fwd_loss = F.mse_loss(predicted, target)
        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_fwd.step()
        if fm_step % 1000 == 0:
            print(f"  FM step {fm_step}: mse={float(fwd_loss):.6f}")

    # --- Eval on traced sequences ---
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(rules, n_eval_sequences)
    pos_levels = _position_levels(seq_len, s)

    fwd_model.eval()
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    all_source = []
    all_target = []
    all_predicted = []

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch_x = eval_data[i:i + batch_size]
            if batch_x.shape[0] < 2:
                continue
            _, _, intermediates = model(batch_x, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
            predicted = fwd_model(source)
            all_source.append(source.cpu().numpy())
            all_target.append(target.cpu().numpy())
            all_predicted.append(predicted.cpu().numpy())

    source_np = np.concatenate(all_source, axis=0)      # (N, seq_len, d)
    target_np = np.concatenate(all_target, axis=0)       # (N, seq_len, d)
    predicted_np = np.concatenate(all_predicted, axis=0)  # (N, seq_len, d)
    residual_np = target_np - predicted_np
    N = source_np.shape[0]

    # Trim traces to match
    level_features = [lf[:N] for lf in level_features]
    level_rules = [lr[:N] for lr in level_rules]

    # --- Global cosine ---
    target_t = torch.from_numpy(target_np.reshape(-1, n_embd))
    pred_t = torch.from_numpy(predicted_np.reshape(-1, n_embd))
    global_cosine = float(F.cosine_similarity(pred_t, target_t, dim=-1).mean())
    global_res_norm = float(np.sqrt(np.mean(residual_np ** 2)))

    print(f"\n{'='*80}")
    print(f"GLOBAL: cosine={global_cosine:.4f}, 1-cos={1-global_cosine:.4f}, "
          f"res_norm={global_res_norm:.4f}")
    print(f"{'='*80}")

    # --- Per-hierarchy-level cosine ---
    print(f"\n--- Per-hierarchy-level cosine ---")
    per_level_cosine = {}
    for lvl in range(L):
        pos_mask = pos_levels == lvl
        if not pos_mask.any():
            continue
        t_sub = torch.from_numpy(target_np[:, pos_mask, :].reshape(-1, n_embd))
        p_sub = torch.from_numpy(predicted_np[:, pos_mask, :].reshape(-1, n_embd))
        cos = float(F.cosine_similarity(t_sub, p_sub, dim=-1).mean())
        r_sub = residual_np[:, pos_mask, :]
        res_norm = float(np.sqrt(np.mean(r_sub ** 2)))
        n_pos = int(pos_mask.sum())
        per_level_cosine[lvl] = {
            "cosine": cos, "1_minus_cos": 1 - cos,
            "res_norm": res_norm, "n_positions": n_pos,
        }
        print(f"  Level {lvl}: cosine={cos:.4f} (1-cos={1-cos:.4f}) "
              f"res_norm={res_norm:.4f}  [{n_pos} positions]")

    # --- eta² for source, predictions, actual, residual at each level ---
    print(f"\n--- Rule/feature eta² by hierarchy level ---")
    print(f"{'':>3}  {'source':^24}  {'fm_pred':^24}  {'actual':^24}  {'residual':^24}")
    print(f"{'L':>3}  {'rule':>11} {'feat':>11}  {'rule':>11} {'feat':>11}  "
          f"{'rule':>11} {'feat':>11}  {'rule':>11} {'feat':>11}")
    print("-" * 123)

    eta2_results = {}

    for ell in range(L):
        s_power = s ** (L - ell)

        # Labels: each position's ancestor at level ell
        ancestor_idx = np.arange(seq_len) // s_power  # shape (seq_len,)
        rule_labels = level_rules[ell][:, ancestor_idx].reshape(-1)  # (N * seq_len,)
        feat_labels = level_features[ell][:, ancestor_idx].reshape(-1)

        src_flat = source_np.reshape(-1, n_embd)
        tgt_flat = target_np.reshape(-1, n_embd)
        pred_flat = predicted_np.reshape(-1, n_embd)
        res_flat = residual_np.reshape(-1, n_embd)

        def compute_eta2(data, labels):
            gm = data.mean(axis=0)
            ss_total = float(np.sum((data - gm) ** 2))
            if ss_total < 1e-12:
                return 0.0
            return _eta2_between(data, labels, gm) / ss_total

        src_rule = compute_eta2(src_flat, rule_labels)
        src_feat = compute_eta2(src_flat, feat_labels)
        pred_rule = compute_eta2(pred_flat, rule_labels)
        pred_feat = compute_eta2(pred_flat, feat_labels)
        tgt_rule = compute_eta2(tgt_flat, rule_labels)
        tgt_feat = compute_eta2(tgt_flat, feat_labels)
        res_rule = compute_eta2(res_flat, rule_labels)
        res_feat = compute_eta2(res_flat, feat_labels)

        eta2_results[f"level_{ell}"] = {
            "source_rule": src_rule, "source_feat": src_feat,
            "fm_pred_rule": pred_rule, "fm_pred_feat": pred_feat,
            "actual_rule": tgt_rule, "actual_feat": tgt_feat,
            "residual_rule": res_rule, "residual_feat": res_feat,
            "tokens_per_subtree": int(s_power),
        }

        print(f"{ell:>3}  {src_rule:>11.5f} {src_feat:>11.5f}  "
              f"{pred_rule:>11.5f} {pred_feat:>11.5f}  "
              f"{tgt_rule:>11.5f} {tgt_feat:>11.5f}  "
              f"{res_rule:>11.5f} {res_feat:>11.5f}")

    # --- FM rule recovery ratio ---
    print(f"\n--- FM rule recovery ratio: eta²(fm_pred) / eta²(actual) ---")
    print(f"{'L':>3}  {'rule_ratio':>11}  {'feat_ratio':>11}  {'interpretation':>30}")
    print("-" * 63)
    for ell in range(L):
        e = eta2_results[f"level_{ell}"]
        rule_ratio = e["fm_pred_rule"] / e["actual_rule"] if e["actual_rule"] > 1e-8 else float('nan')
        feat_ratio = e["fm_pred_feat"] / e["actual_feat"] if e["actual_feat"] > 1e-8 else float('nan')
        if rule_ratio > 0.8:
            interp = "FM captures rule structure"
        elif rule_ratio > 0.4:
            interp = "FM partially captures rules"
        elif rule_ratio > 0.1:
            interp = "FM weakly captures rules"
        else:
            interp = "FM misses rule structure"
        print(f"{ell:>3}  {rule_ratio:>11.3f}  {feat_ratio:>11.3f}  {interp:>30}")

    # --- eta² on LAST position only (maximum context) ---
    print(f"\n--- eta² on last position only (maximum causal context) ---")
    print(f"{'':>3}  {'source':^24}  {'fm_pred':^24}  {'actual':^24}  {'residual':^24}")
    print(f"{'L':>3}  {'rule':>11} {'feat':>11}  {'rule':>11} {'feat':>11}  "
          f"{'rule':>11} {'feat':>11}  {'rule':>11} {'feat':>11}")
    print("-" * 123)

    eta2_last = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        last_ancestor = (seq_len - 1) // s_power
        rule_labels = level_rules[ell][:, last_ancestor]
        feat_labels = level_features[ell][:, last_ancestor]

        src_last = source_np[:, -1, :]
        tgt_last = target_np[:, -1, :]
        pred_last = predicted_np[:, -1, :]
        res_last = residual_np[:, -1, :]

        def compute_eta2_1d(data, labels):
            gm = data.mean(axis=0)
            ss_total = float(np.sum((data - gm) ** 2))
            if ss_total < 1e-12:
                return 0.0
            return _eta2_between(data, labels, gm) / ss_total

        src_rule = compute_eta2_1d(src_last, rule_labels)
        src_feat = compute_eta2_1d(src_last, feat_labels)
        pred_rule = compute_eta2_1d(pred_last, rule_labels)
        pred_feat = compute_eta2_1d(pred_last, feat_labels)
        tgt_rule = compute_eta2_1d(tgt_last, rule_labels)
        tgt_feat = compute_eta2_1d(tgt_last, feat_labels)
        res_rule = compute_eta2_1d(res_last, rule_labels)
        res_feat = compute_eta2_1d(res_last, feat_labels)

        eta2_last[f"level_{ell}"] = {
            "source_rule": src_rule, "source_feat": src_feat,
            "fm_pred_rule": pred_rule, "fm_pred_feat": pred_feat,
            "actual_rule": tgt_rule, "actual_feat": tgt_feat,
            "residual_rule": res_rule, "residual_feat": res_feat,
        }

        print(f"{ell:>3}  {src_rule:>11.5f} {src_feat:>11.5f}  "
              f"{pred_rule:>11.5f} {pred_feat:>11.5f}  "
              f"{tgt_rule:>11.5f} {tgt_feat:>11.5f}  "
              f"{res_rule:>11.5f} {res_feat:>11.5f}")

    # --- Last-position recovery ratios ---
    print(f"\n--- Last-position recovery ratio ---")
    print(f"{'L':>3}  {'rule_ratio':>11}  {'feat_ratio':>11}")
    print("-" * 33)
    for ell in range(L):
        e = eta2_last[f"level_{ell}"]
        rule_ratio = e["fm_pred_rule"] / e["actual_rule"] if e["actual_rule"] > 1e-8 else float('nan')
        feat_ratio = e["fm_pred_feat"] / e["actual_feat"] if e["actual_feat"] > 1e-8 else float('nan')
        print(f"{ell:>3}  {rule_ratio:>11.3f}  {feat_ratio:>11.3f}")

    # --- Added structure: fm_pred vs source ---
    print(f"\n--- Added structure: eta²(fm_pred) - eta²(source) ---")
    print(f"  (Positive = FM added rule/feature structure beyond its input)")
    print(f"{'L':>3}  {'rule_added':>11}  {'feat_added':>11}")
    print("-" * 33)
    for ell in range(L):
        e = eta2_results[f"level_{ell}"]
        rule_added = e["fm_pred_rule"] - e["source_rule"]
        feat_added = e["fm_pred_feat"] - e["source_feat"]
        sign_r = "+" if rule_added > 0 else ""
        sign_f = "+" if feat_added > 0 else ""
        print(f"{ell:>3}  {sign_r}{rule_added:>10.5f}  {sign_f}{feat_added:>10.5f}")

    # --- Save ---
    save_dir = f"{DATA_DIR}/rhm_dgp_approximation"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "experiment": "FM as DGP approximation",
        "params": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": model_tag, "checkpoint_step": checkpoint_step,
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}",
            "fm_params": fwd_n_params,
            "gap": f"{predict_from}->{predict_to}",
            "n_eval_sequences": n_eval_sequences,
        },
        "global": {
            "cosine": global_cosine,
            "1_minus_cos": 1 - global_cosine,
            "res_norm": global_res_norm,
        },
        "per_level_cosine": per_level_cosine,
        "eta2_all_positions": eta2_results,
        "eta2_last_position": eta2_last,
    }

    fname = f"{key}_{model_tag}_step{checkpoint_step}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/{fname}")
    return result


# ---------------------------------------------------------------------------
# Training trajectory variant: track DGP approximation over training
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=10800,
    memory=16384,
)
def dgp_approximation_trajectory(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fwd_n_layer: int = 2, fwd_d_head: int = 24, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 1.0,
    fm_train_steps: int = 5000,
    fm_lr: float = 1e-3,
    n_eval_sequences: int = 5000,
    batch_size: int = 64,
    fm_seed: int = 137,
):
    """Track DGP approximation quality at multiple training checkpoints."""
    import glob

    L = depth
    key = setting_key(v, s, L, m)
    model_tag = f"{n_layer}L{n_head}H{n_embd}D"
    ckpt_dir = f"{DATA_DIR}/{key}/regime_trajectory_{model_tag}"

    volume.reload()

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint dir not found: {ckpt_dir}. "
            f"Run rhm_regime_trajectory.py first."
        )

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt")))
    steps = [int(f.split("step")[1].split(".")[0]) for f in ckpt_files]

    # Use a subset: early, mid, late
    target_fracs = [0.0, 0.05, 0.20, 0.50, 1.0]
    max_step = max(steps)
    selected_steps = []
    for frac in target_fracs:
        target = frac * max_step
        closest = min(steps, key=lambda s: abs(s - target))
        if closest not in selected_steps:
            selected_steps.append(closest)
    selected_steps.sort()

    print(f"Tracking DGP approximation at steps: {selected_steps}")

    all_results = []
    for step in selected_steps:
        print(f"\n--- Step {step} ---")
        r = dgp_approximation.remote(
            v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            checkpoint_step=step,
            predict_from=predict_from, predict_to=predict_to,
            fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
            fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
            fm_train_steps=fm_train_steps, fm_lr=fm_lr,
            n_eval_sequences=n_eval_sequences, batch_size=batch_size,
            fm_seed=fm_seed,
        )
        all_results.append(r)

    # --- Summary table ---
    print(f"\n{'='*120}")
    print(f"DGP APPROXIMATION TRAJECTORY: {key}, {model_tag}")
    print(f"{'='*120}")

    # Feature recovery ratio at last position
    print(f"\nFeature eta² recovery ratio (last position): eta²(fm_pred) / eta²(actual)")
    header = f"{'step':>6}  {'cos':>7}" + "".join(f"  {'fL'+str(e):>7}" for e in range(L))
    print(header)
    print("-" * len(header))
    for r in all_results:
        step = r["params"]["checkpoint_step"]
        cos = r["global"]["cosine"]
        ratios = []
        for ell in range(L):
            e = r["eta2_last_position"][f"level_{ell}"]
            ratio = e["fm_pred_feat"] / e["actual_feat"] if e["actual_feat"] > 1e-8 else float('nan')
            ratios.append(ratio)
        row = f"{step:>6}  {cos:>7.4f}" + "".join(f"  {r:>7.3f}" for r in ratios)
        print(row)

    save_dir = f"{DATA_DIR}/rhm_dgp_approximation"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"trajectory_{key}_{model_tag}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump({
            "experiment": "DGP approximation trajectory",
            "params": {
                "v": v, "s": s, "L": L, "m": m,
                "model": model_tag,
                "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}",
            },
            "results": all_results,
        }, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return all_results
