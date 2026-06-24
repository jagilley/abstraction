"""FM-surprise-weighted NTP: principled gradient reallocation via self-model surprise.

Instead of focal loss (heuristic: (1-p_t)^gamma) or confidence thresholding
(heuristic: 1[p_t < tau]), weight the NTP loss at each position by the
co-trained forward model's angular prediction error at that position.

  surprise_t = 1 - cos(FM_predicted_t, actual_t)
  weight_t = surprise_t / mean(surprise)
  loss = mean(weight_t * -log p(y_t))

Self-scheduling: early in training, FM is random -> cosine ~ 0 everywhere ->
surprise ~ 1 everywhere -> weights ~ 1 -> standard NTP. As FM improves and
captures easy positions first, weights shift to positions the FM still
can't predict. No heuristic gamma or tau; the model's own self-model defines
what's "boring."

Conditions:
  ce: standard cross-entropy (baseline)
  focal: focal loss gamma=2 (heuristic reference from prior experiment)
  fm_weighted: FM-surprise-weighted NTP (the new thing)
  fm_weighted_ll: FM-surprise-weighted NTP + uniform local loss (full architecture)

All conditions: L=6/m=4, 6L/6H/192D (~2.7M params), v=8, s=2, 20M tokens
FM (co-trained): 2L, d_head=24, n_head=1, predicting post_block0 -> post_block3

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_fm_weighted_ntp::fm_weighted_ntp_sweep
  modal run --detach -m rhm.rhm_fm_weighted_ntp::fm_weighted_ll_run
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

app = modal.App("rhm-fm-weighted-ntp", image=image)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _focal_loss(logits, targets, gamma, vocab_size):
    import torch
    import torch.nn.functional as F
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    log_probs = F.log_softmax(logits_flat, dim=-1)
    log_pt = log_probs.gather(1, targets_flat.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal_weight = (1.0 - pt) ** gamma
    return -(focal_weight * log_pt).mean()


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _position_levels(seq_len, s):
    """s-adic valuation for each prediction position 1..seq_len-1."""
    import numpy as np
    levels = np.zeros(seq_len - 1, dtype=np.int64)
    for t in range(seq_len - 1):
        p = t + 1
        level = 0
        while p % s == 0:
            p //= s
            level += 1
        levels[t] = level
    return levels


def _eval_per_level(model, eval_seqs, seq_len, s, L, v, batch_size, device):
    """Per-level cross-entropy, accuracy, and output entropy (standard CE eval)."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    levels = _position_levels(seq_len, s)
    max_level = int(levels.max())
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    per_pos_loss_sum = np.zeros(seq_len - 1)
    per_pos_correct_sum = np.zeros(seq_len - 1)
    per_pos_entropy_sum = np.zeros(seq_len - 1)
    n_samples = 0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]
            bs = batch.shape[0]
            logits, _ = model(batch)
            pred_logits = logits[:, :-1, :]
            targets = batch[:, 1:]
            log_probs = F.log_softmax(pred_logits, dim=-1)
            target_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            pos_loss = -target_log_probs
            preds = pred_logits.argmax(dim=-1)
            correct = (preds == targets).float()
            probs = F.softmax(pred_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            per_pos_loss_sum += pos_loss.sum(dim=0).cpu().numpy()
            per_pos_correct_sum += correct.sum(dim=0).cpu().numpy()
            per_pos_entropy_sum += entropy.sum(dim=0).cpu().numpy()
            n_samples += bs

    per_pos_loss = per_pos_loss_sum / n_samples
    per_pos_acc = per_pos_correct_sum / n_samples
    per_pos_entropy = per_pos_entropy_sum / n_samples

    level_results = {}
    for lvl in range(max_level + 1):
        mask = levels == lvl
        n_pos = int(mask.sum())
        if n_pos == 0:
            continue
        level_results[lvl] = {
            "level": lvl,
            "n_positions": n_pos,
            "avg_loss": float(per_pos_loss[mask].mean()),
            "avg_accuracy": float(per_pos_acc[mask].mean()),
            "avg_entropy": float(per_pos_entropy[mask].mean()),
        }

    return {
        "per_level": level_results,
        "overall_loss": float(per_pos_loss.mean()),
        "overall_accuracy": float(per_pos_acc.mean()),
        "overall_entropy": float(per_pos_entropy.mean()),
    }


def _eval_fm_metrics(model, fwd_model, eval_seqs, seq_len, s, L, batch_size,
                     device, predict_from, predict_to):
    """Per-level FM cosine, residual norm, and effective NTP weight on eval seqs."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    levels = _position_levels(seq_len, s)
    max_level = int(levels.max())
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    # Accumulate per-position stats across all T positions
    per_pos_cos_sum = np.zeros(seq_len)
    per_pos_res_norm_sum = np.zeros(seq_len)
    n_samples = 0

    model.eval()
    fwd_model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]
            if batch.shape[0] < 2:
                continue
            bs = batch.shape[0]
            _, _, intermediates = model(batch, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
            predicted = fwd_model(source)
            cos = F.cosine_similarity(predicted, target, dim=-1)
            res_norm = (target - predicted).norm(dim=-1)
            per_pos_cos_sum += cos.sum(dim=0).cpu().numpy()
            per_pos_res_norm_sum += res_norm.sum(dim=0).cpu().numpy()
            n_samples += bs

    per_pos_cos = per_pos_cos_sum / n_samples
    per_pos_res = per_pos_res_norm_sum / n_samples

    # Effective weights from 1-cos surprise (aligned with prediction positions)
    surprise = 1.0 - per_pos_cos
    per_pos_weight = surprise / (surprise.mean() + 1e-8)

    # Group by hierarchy level (first T-1 positions align with levels array)
    level_fm = {}
    for lvl in range(max_level + 1):
        mask = levels == lvl
        n_pos = int(mask.sum())
        if n_pos == 0:
            continue
        level_fm[lvl] = {
            "avg_cosine": float(per_pos_cos[:-1][mask].mean()),
            "avg_res_norm": float(per_pos_res[:-1][mask].mean()),
            "avg_weight": float(per_pos_weight[:-1][mask].mean()),
        }

    return {
        "overall_cosine": float(per_pos_cos.mean()),
        "overall_res_norm": float(per_pos_res.mean()),
        "per_level": level_fm,
    }


def _ensure_corpus(v, s, L, m, n_tokens):
    import numpy as np
    from rhm.rhm_data import make_corpus
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
    res_last = residuals_np[:, -1, :]
    last_mean = res_last.mean(axis=0)
    ss_total_last = float(np.sum((res_last - last_mean) ** 2))
    if ss_total_last < 1e-12:
        return {f"level_{ell}": {"feature_eta2_last": 0.0, "rule_eta2_last": 0.0}
                for ell in range(L)}
    results = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        last_ancestor = (seq_len - 1) // s_power
        rule_labels = level_rules[ell][:, last_ancestor]
        feat_labels = level_features[ell][:, last_ancestor]
        results[f"level_{ell}"] = {
            "feature_eta2_last": _eta2_between(res_last, feat_labels, last_mean) / ss_total_last,
            "rule_eta2_last": _eta2_between(res_last, rule_labels, last_mean) / ss_total_last,
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


# ---------------------------------------------------------------------------
# Phase 1: Train model (handles ce, focal, and fm_weighted)
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=10800, memory=16384)
def train_and_eval(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    ll_lambda: float = 0.1,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    fwd_n_layer: int = 2, fwd_d_head: int = 24, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 1.0,
    fm_lr: float = 1e-3,
    batch_size: int = 64, lr: float = 3e-4,
    n_eval_sequences: int = 5000,
    seed: int = 42,
):
    """Train model with specified loss, evaluate per-level loss at checkpoints.

    loss_type: "ce" (standard), "focal" (focal loss), "fm_weighted" (FM-surprise NTP),
               "fm_weighted_ll" (FM-surprise NTP + uniform local loss)
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from rhm.rhm_data import generate_sequences_batched

    torch.manual_seed(seed)
    np.random.seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    device = "cuda"

    volume.reload()
    _ensure_corpus(v, s, L, m, n_tokens)

    data = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    tokens_per_step = batch_size * seq_len
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    eval_interval = 200
    checkpoint_fracs = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0]
    checkpoint_steps = sorted(set(
        round(f * n_steps / eval_interval) * eval_interval for f in checkpoint_fracs
    ))

    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    fwd_model = None
    opt_fm = None
    uses_fm = loss_type in ("fm_weighted", "fm_weighted_ll")
    if uses_fm:
        from a2a_forward.forward_model import TransformerForwardModel
        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=seq_len,
        ).to(device)
        opt_fm = torch.optim.AdamW(fwd_model.parameters(), lr=fm_lr, weight_decay=0.01)

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs = generate_sequences_batched(rules, n_eval_sequences, seed=12345)

    run_label = {
        "ce": "ce", "focal": f"focal_g{focal_gamma}",
        "fm_weighted": "fm_weighted", "fm_weighted_ll": "fm_wt_ll",
    }[loss_type]

    print(f"=== Training: {key}, {n_layer}L/{n_head}H/{n_embd}D, {run_label} ===")
    print(f"  seq_len={seq_len}, n_steps={n_steps}")
    print(f"  checkpoints: {checkpoint_steps}")

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - seq_len - 1, (batch_size,))
        x = torch.stack([split_data[i:i + seq_len] for i in ix])
        y = torch.stack([split_data[i + 1:i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    trajectory = []

    for step in range(n_steps + 1):
        if step > 0:
            model.train()
            x, y = get_batch(train_data)
            B, T = x.shape

            if uses_fm:
                logits, _, intermediates = model(x, return_intermediates=True)
                source = intermediates[predict_from].detach()
                target = intermediates[predict_to].detach()

                fwd_model.train()
                predicted = fwd_model(source)

                # Compute surprise weights BEFORE FM update (using current FM)
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(predicted, target, dim=-1)
                    surprise = 1.0 - cos_sim
                    weights = surprise / (surprise.mean() + 1e-8)

                # Update FM
                fm_loss = F.mse_loss(predicted, target)
                opt_fm.zero_grad()
                fm_loss.backward()
                torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
                opt_fm.step()

                # Weighted NTP loss
                per_pos_loss = F.cross_entropy(
                    logits.view(-1, v), y.view(-1), reduction='none'
                ).view(B, T)
                loss = (weights * per_pos_loss).mean()

                # Uniform local loss: push intermediate activations toward FM-predictability
                if loss_type == "fm_weighted_ll":
                    target_live = intermediates[predict_to]
                    with torch.no_grad():
                        fm_pred_sg = fwd_model(source)
                    local_loss = ll_lambda * F.mse_loss(target_live, fm_pred_sg)
                    loss = loss + local_loss

                if step % 2000 == 0:
                    ll_str = ""
                    if loss_type == "fm_weighted_ll":
                        ll_str = f" ll={float(local_loss):.4f}"
                    print(f"    step {step}: fm_mse={float(fm_loss):.6f} "
                          f"fm_cos={float(cos_sim.mean()):.4f} "
                          f"weight_std={float(weights.std()):.3f}{ll_str}")

            elif loss_type == "focal":
                logits, _ = model(x)
                loss = _focal_loss(logits, y, focal_gamma, v)
            else:
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, v), y.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Checkpoint evaluation
        if step in checkpoint_steps:
            model.eval()
            with torch.no_grad():
                vl = sum(float(model(*get_batch(val_data))[1]) for _ in range(10)) / 10

            per_level = _eval_per_level(
                model, eval_seqs, seq_len, s, L, v, batch_size, device)

            entry = {
                "step": step,
                "val_loss": vl,
                "overall_loss": per_level["overall_loss"],
                "overall_accuracy": per_level["overall_accuracy"],
                "overall_entropy": per_level["overall_entropy"],
                "per_level": per_level["per_level"],
            }

            if uses_fm and fwd_model is not None:
                fm_metrics = _eval_fm_metrics(
                    model, fwd_model, eval_seqs, seq_len, s, L, batch_size,
                    device, predict_from, predict_to)
                entry["fm_metrics"] = fm_metrics

            trajectory.append(entry)

            level_str = "  ".join(
                f"L{lvl}={per_level['per_level'][lvl]['avg_loss']:.3f}"
                for lvl in sorted(per_level['per_level'].keys())
            )
            extra = ""
            if "fm_metrics" in entry:
                fm = entry["fm_metrics"]
                extra = f"  fm_cos={fm['overall_cosine']:.4f}"
                wts = [fm["per_level"].get(lvl, {}).get("avg_weight", 0)
                       for lvl in range(L)]
                extra += f"  wt=[{', '.join(f'{w:.2f}' for w in wts)}]"
            print(f"  step {step:6d}: val={vl:.4f}  {level_str}{extra}")

    # Save model (and FM for fm_weighted)
    save_dir = f"{DATA_DIR}/rhm_fm_weighted_ntp"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"model_{run_label}.pt")
    torch.save(model.state_dict(), model_path)

    fm_path = None
    if uses_fm and fwd_model is not None:
        fm_path = os.path.join(save_dir, f"fm_{run_label}.pt")
        torch.save(fwd_model.state_dict(), fm_path)

    result = {
        "loss_type": loss_type,
        "run_label": run_label,
        "setting": key, "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
        "model": f"{n_layer}L/{n_head}H/{n_embd}D",
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "n_steps": n_steps, "n_tokens": n_tokens,
        "model_path": model_path,
        "fm_path": fm_path,
        "trajectory": trajectory,
    }
    if loss_type == "focal":
        result["focal_gamma"] = focal_gamma
    if uses_fm:
        result["predict_from"] = predict_from
        result["predict_to"] = predict_to
        result["fwd_n_layer"] = fwd_n_layer
        result["fwd_d_head"] = fwd_d_head
    if loss_type == "fm_weighted_ll":
        result["ll_lambda"] = ll_lambda

    with open(os.path.join(save_dir, f"trajectory_{run_label}.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return result


# ---------------------------------------------------------------------------
# Phase 2: Post-hoc FM evaluation on converged model
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=3600, memory=16384)
def fm_eval(
    model_path: str = "",
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    run_label: str = "ce",
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fwd_n_layer: int = 2, fwd_d_head: int = 24, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 1.0,
    fm_train_steps: int = 5000,
    fm_lr: float = 1e-3,
    n_eval_sequences: int = 10000,
    batch_size: int = 64,
    seed: int = 137,
):
    """Train a fresh FM on the converged model and measure residual properties."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    device = "cuda"

    volume.reload()

    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    corpus = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    corpus = torch.from_numpy(corpus[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data = corpus[:split]

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - seq_len - 1, (batch_size,))
        x = torch.stack([split_data[i:i + seq_len] for i in ix])
        return x.to(device)

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=seq_len,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())

    main_block_params = sum(p.numel() for p in list(model.transformer.h[0].parameters()))
    gap_blocks = int(predict_to.replace("post_block", "")) - int(predict_from.replace("post_block", ""))
    gap_params = gap_blocks * main_block_params
    capacity_ratio = fwd_n_params / gap_params if gap_params > 0 else 0

    print(f"=== FM eval: {run_label}, FM {fwd_n_params/1e3:.0f}K params "
          f"({capacity_ratio*100:.0f}% of gap) ===")

    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fm_lr, weight_decay=0.01)

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
        if fm_step % 2000 == 0:
            print(f"  FM step {fm_step}: mse={float(fwd_loss):.6f}")

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
    eta2 = _compute_hierarchy_eta2(residuals_np, trimmed_features, trimmed_rules, s=s, L=L)

    result = {
        "run_label": run_label,
        "fm_cosine": float(mean_cosine),
        "residual_norm": float(mean_res_norm),
        "effective_rank": float(eff_rank),
        "effective_rank_pct": float(eff_rank / n_embd * 100),
        "top1_pc_variance": float(top1_var),
        "fm_params": fwd_n_params,
        "capacity_ratio_pct": float(capacity_ratio * 100),
        "hierarchy_eta2": eta2,
    }

    print(f"  {run_label}: cos={mean_cosine:.4f} res_norm={mean_res_norm:.4f} "
          f"rank={eff_rank:.1f}/{n_embd} ({eff_rank/n_embd*100:.1f}%) "
          f"top1={top1_var*100:.1f}%")
    for ell in range(L):
        e = eta2[f"level_{ell}"]
        print(f"    L{ell}: feat={e['feature_eta2_last']:.4f} rule={e['rule_eta2_last']:.4f}")

    save_dir = f"{DATA_DIR}/rhm_fm_weighted_ntp"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"fm_eval_{run_label}.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=4096)
def fm_weighted_ntp_sweep(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
):
    """Compare ce, focal gamma=2, and FM-surprise-weighted NTP."""
    import numpy as np

    L = depth
    key = setting_key(v, s, L, m)

    runs = [
        ("ce",         {}),
        ("focal",      {"focal_gamma": 2.0}),
        ("fm_weighted", {}),
    ]

    print(f"\n{'='*100}")
    print(f"FM-WEIGHTED NTP SWEEP: {key}, {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  conditions: {[r[0] for r in runs]}")
    print(f"  n_tokens: {n_tokens:,}")
    print(f"{'='*100}")

    # --- Phase 1: Train all models in parallel ---
    print("\n--- Phase 1: Training models ---")
    train_handles = []
    for loss_type, extra_kwargs in runs:
        h = train_and_eval.spawn(
            v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            loss_type=loss_type, **extra_kwargs,
        )
        train_handles.append((loss_type, h))

    train_results = {}
    for loss_type, h in train_handles:
        result = h.get()
        label = result["run_label"]
        train_results[label] = result
        final = result["trajectory"][-1]
        print(f"  {label}: val={final['val_loss']:.4f} "
              f"acc={final['overall_accuracy']:.4f}")

    # --- Phase 2: Post-hoc FM evaluation in parallel ---
    print("\n--- Phase 2: Post-hoc FM evaluation ---")
    fm_handles = []
    for label, result in train_results.items():
        h = fm_eval.spawn(
            model_path=result["model_path"],
            v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            run_label=label,
        )
        fm_handles.append((label, h))

    fm_results = {}
    for label, h in fm_handles:
        fm_results[label] = h.get()

    # --- Summary tables ---
    uniform = float(np.log(v))
    run_labels = [r["run_label"] for _, r in sorted(train_results.items(),
                  key=lambda x: ["ce", "focal_g2.0", "fm_weighted"].index(x[0])
                  if x[0] in ["ce", "focal_g2.0", "fm_weighted"] else 99)]
    max_level = max(
        int(lvl) for r in train_results.values()
        for lvl in r["trajectory"][-1]["per_level"].keys()
    )

    print(f"\n{'='*120}")
    print(f"FM-WEIGHTED NTP SWEEP RESULTS: {key}, {n_layer}L/{n_head}H/{n_embd}D")
    print(f"Uniform baseline: {uniform:.4f}")
    print(f"{'='*120}")

    # Table 1: Per-level loss at convergence
    print("\n--- Per-level loss at convergence (standard CE eval) ---")
    header = (f"{'run':>14}  {'val':>7}"
              + "".join(f"  {'L'+str(l):>7}" for l in range(max_level + 1))
              + "  |"
              + "".join(f"  {'L'+str(l)+' acc':>7}" for l in range(max_level + 1)))
    print(header)
    print("-" * len(header))
    for label in run_labels:
        final = train_results[label]["trajectory"][-1]
        pl = final["per_level"]
        row = f"{label:>14}  {final['val_loss']:>7.4f}"
        for lvl in range(max_level + 1):
            if lvl in pl:
                row += f"  {pl[lvl]['avg_loss']:>7.4f}"
            else:
                row += f"  {'--':>7}"
        row += "  |"
        for lvl in range(max_level + 1):
            if lvl in pl:
                row += f"  {pl[lvl]['avg_accuracy']:>7.4f}"
            else:
                row += f"  {'--':>7}"
        print(row)

    # Table 2: Per-level loss change vs baseline (ce)
    baseline_pl = train_results["ce"]["trajectory"][-1]["per_level"]
    baseline_val = train_results["ce"]["trajectory"][-1]["val_loss"]
    print("\n--- Per-level loss change vs baseline (ce) ---")
    header2 = f"{'run':>14}  {'Δval':>7}" + "".join(
        f"  {'ΔL'+str(l):>7}" for l in range(max_level + 1))
    print(header2)
    print("-" * len(header2))
    for label in run_labels:
        final = train_results[label]["trajectory"][-1]
        pl = final["per_level"]
        delta_val = final["val_loss"] - baseline_val
        row = f"{label:>14}  {delta_val:>+7.4f}"
        for lvl in range(max_level + 1):
            if lvl in pl and lvl in baseline_pl:
                delta = pl[lvl]["avg_loss"] - baseline_pl[lvl]["avg_loss"]
                row += f"  {delta:>+7.4f}"
            else:
                row += f"  {'--':>7}"
        print(row)

    # Table 3: Fraction of baseline learning retained per level
    print("\n--- Fraction of baseline learning retained ---")
    header3 = f"{'run':>14}" + "".join(
        f"  {'L'+str(l)+' ret%':>8}" for l in range(min(4, max_level + 1)))
    print(header3)
    print("-" * len(header3))
    for label in run_labels:
        if label == "ce":
            continue
        final = train_results[label]["trajectory"][-1]
        pl = final["per_level"]
        row = f"{label:>14}"
        for lvl in range(min(4, max_level + 1)):
            if lvl in pl and lvl in baseline_pl:
                bl = uniform - baseline_pl[lvl]["avg_loss"]
                ml = uniform - pl[lvl]["avg_loss"]
                retained = (ml / bl * 100) if bl > 0.001 else 0
                row += f"  {retained:>7.1f}%"
            else:
                row += f"  {'--':>8}"
        print(row)

    # Table 4: Post-hoc FM residual properties
    print("\n--- Post-hoc FM residual properties (fresh FM on converged model) ---")
    header4 = (f"{'run':>14}  {'cos':>7}  {'1-cos':>7}  {'res_norm':>9}  "
               f"{'rank%':>6}  {'top1%':>6}")
    for ell in range(L):
        header4 += f"  {'fL'+str(ell)+'*':>7}"
    print(header4)
    print("-" * len(header4))
    for label in run_labels:
        fm = fm_results[label]
        eta = fm["hierarchy_eta2"]
        row = (f"{label:>14}  {fm['fm_cosine']:>7.4f}  {1-fm['fm_cosine']:>7.4f}  "
               f"{fm['residual_norm']:>9.4f}  {fm['effective_rank_pct']:>6.1f}  "
               f"{fm['top1_pc_variance']*100:>6.1f}")
        for ell in range(L):
            row += f"  {eta[f'level_{ell}']['feature_eta2_last']:>7.4f}"
        print(row)

    # Table 5: FM-weighted per-level weight trajectory (fm_weighted only)
    if "fm_weighted" in train_results:
        print("\n--- FM-surprise weight trajectory (fm_weighted condition) ---")
        print(f"  Shows how gradient reallocation evolves as the FM matures")
        header5 = f"{'step':>6}  {'fm_cos':>7}" + "".join(
            f"  {'L'+str(l)+' wt':>7}" for l in range(max_level + 1))
        print(header5)
        print("-" * len(header5))
        for entry in train_results["fm_weighted"]["trajectory"]:
            if "fm_metrics" not in entry:
                continue
            fm = entry["fm_metrics"]
            row = f"{entry['step']:>6}  {fm['overall_cosine']:>7.4f}"
            for lvl in range(max_level + 1):
                if lvl in fm["per_level"]:
                    row += f"  {fm['per_level'][lvl]['avg_weight']:>7.3f}"
                else:
                    row += f"  {'--':>7}"
            print(row)

    # Save combined results
    combined = {
        "experiment": "fm_weighted_ntp_sweep",
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "model": f"{n_layer}L/{n_head}H/{n_embd}D",
        "n_tokens": n_tokens,
        "uniform_baseline": uniform,
        "conditions": [r[0] for r in runs],
        "train_results": train_results,
        "fm_results": fm_results,
    }
    save_dir = f"{DATA_DIR}/rhm_fm_weighted_ntp"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "sweep_results.json"), "w") as f:
        json.dump(combined, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/sweep_results.json")
    return combined


@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=4096)
def fm_weighted_ll_run(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    ll_lambda: float = 0.1,
):
    """Train fm_weighted_ll (FM-surprise NTP + uniform local loss) and evaluate."""
    import numpy as np

    L = depth
    key = setting_key(v, s, L, m)

    print(f"\n{'='*100}")
    print(f"FM-WEIGHTED NTP + LOCAL LOSS: {key}, {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  ll_lambda: {ll_lambda}")
    print(f"  n_tokens: {n_tokens:,}")
    print(f"{'='*100}")

    # --- Train ---
    print("\n--- Training fm_weighted_ll ---")
    result = train_and_eval.remote(
        v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        loss_type="fm_weighted_ll", ll_lambda=ll_lambda,
    )
    final = result["trajectory"][-1]
    print(f"  fm_wt_ll: val={final['val_loss']:.4f} acc={final['overall_accuracy']:.4f}")

    # --- Post-hoc FM eval ---
    print("\n--- Post-hoc FM evaluation ---")
    fm_result = fm_eval.remote(
        model_path=result["model_path"],
        v=v, s=s, depth=depth, m=m, n_tokens=n_tokens,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        run_label="fm_wt_ll",
    )

    # --- Summary (compare with prior sweep results if available) ---
    uniform = float(np.log(v))
    max_level = max(int(lvl) for lvl in final["per_level"].keys())

    print(f"\n{'='*100}")
    print(f"FM-WEIGHTED NTP + LOCAL LOSS RESULTS")
    print(f"Uniform baseline: {uniform:.4f}")
    print(f"{'='*100}")

    # Load prior sweep results for comparison if available
    prior_path = f"{DATA_DIR}/rhm_fm_weighted_ntp/sweep_results.json"
    prior = None
    if os.path.exists(prior_path):
        with open(prior_path) as f:
            prior = json.load(f)

    # Per-level loss
    print("\n--- Per-level loss at convergence ---")
    labels_to_show = []
    all_results = {}

    if prior:
        for label in ["ce", "focal_g2.0", "fm_weighted"]:
            if label in prior["train_results"]:
                labels_to_show.append(label)
                all_results[label] = prior["train_results"][label]

    labels_to_show.append("fm_wt_ll")
    all_results["fm_wt_ll"] = result

    header = (f"{'run':>14}  {'val':>7}"
              + "".join(f"  {'L'+str(l):>7}" for l in range(max_level + 1))
              + "  |"
              + "".join(f"  {'L'+str(l)+' acc':>7}" for l in range(max_level + 1)))
    print(header)
    print("-" * len(header))
    for label in labels_to_show:
        r = all_results[label]
        f_entry = r["trajectory"][-1]
        pl = f_entry["per_level"]
        row = f"{label:>14}  {f_entry['val_loss']:>7.4f}"
        for lvl in range(max_level + 1):
            k = lvl if lvl in pl else str(lvl)
            if k in pl:
                row += f"  {pl[k]['avg_loss']:>7.4f}"
            else:
                row += f"  {'--':>7}"
        row += "  |"
        for lvl in range(max_level + 1):
            k = lvl if lvl in pl else str(lvl)
            if k in pl:
                row += f"  {pl[k]['avg_accuracy']:>7.4f}"
            else:
                row += f"  {'--':>7}"
        print(row)

    # Per-level delta vs ce
    if prior and "ce" in prior["train_results"]:
        baseline_pl = prior["train_results"]["ce"]["trajectory"][-1]["per_level"]
        baseline_val = prior["train_results"]["ce"]["trajectory"][-1]["val_loss"]
        print("\n--- Per-level loss change vs baseline (ce) ---")
        header2 = f"{'run':>14}  {'Δval':>7}" + "".join(
            f"  {'ΔL'+str(l):>7}" for l in range(max_level + 1))
        print(header2)
        print("-" * len(header2))
        for label in labels_to_show:
            r = all_results[label]
            f_entry = r["trajectory"][-1]
            pl = f_entry["per_level"]
            delta_val = f_entry["val_loss"] - baseline_val
            row = f"{label:>14}  {delta_val:>+7.4f}"
            for lvl in range(max_level + 1):
                k = lvl if lvl in pl else str(lvl)
                bk = lvl if lvl in baseline_pl else str(lvl)
                if k in pl and bk in baseline_pl:
                    delta = pl[k]["avg_loss"] - baseline_pl[bk]["avg_loss"]
                    row += f"  {delta:>+7.4f}"
                else:
                    row += f"  {'--':>7}"
            print(row)

    # Post-hoc FM properties
    print("\n--- Post-hoc FM residual properties ---")
    fm_labels = []
    all_fm = {}
    if prior and "fm_results" in prior:
        for label in ["ce", "focal_g2.0", "fm_weighted"]:
            if label in prior["fm_results"]:
                fm_labels.append(label)
                all_fm[label] = prior["fm_results"][label]
    fm_labels.append("fm_wt_ll")
    all_fm["fm_wt_ll"] = fm_result

    header3 = (f"{'run':>14}  {'cos':>7}  {'1-cos':>7}  {'res_norm':>9}  "
               f"{'rank%':>6}  {'top1%':>6}")
    for ell in range(L):
        header3 += f"  {'fL'+str(ell)+'*':>7}"
    print(header3)
    print("-" * len(header3))
    for label in fm_labels:
        fm = all_fm[label]
        eta = fm["hierarchy_eta2"]
        row = (f"{label:>14}  {fm['fm_cosine']:>7.4f}  {1-fm['fm_cosine']:>7.4f}  "
               f"{fm['residual_norm']:>9.4f}  {fm['effective_rank_pct']:>6.1f}  "
               f"{fm['top1_pc_variance']*100:>6.1f}")
        for ell in range(L):
            row += f"  {eta[f'level_{ell}']['feature_eta2_last']:>7.4f}"
        print(row)

    # Weight trajectory
    print("\n--- FM-surprise weight trajectory (fm_wt_ll) ---")
    header4 = f"{'step':>6}  {'fm_cos':>7}" + "".join(
        f"  {'L'+str(l)+' wt':>7}" for l in range(max_level + 1))
    print(header4)
    print("-" * len(header4))
    for entry in result["trajectory"]:
        if "fm_metrics" not in entry:
            continue
        fm = entry["fm_metrics"]
        row = f"{entry['step']:>6}  {fm['overall_cosine']:>7.4f}"
        for lvl in range(max_level + 1):
            if lvl in fm["per_level"]:
                row += f"  {fm['per_level'][lvl]['avg_weight']:>7.3f}"
            else:
                row += f"  {'--':>7}"
        print(row)

    # Save
    save_dir = f"{DATA_DIR}/rhm_fm_weighted_ntp"
    os.makedirs(save_dir, exist_ok=True)
    combined = {
        "experiment": "fm_weighted_ll_run",
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "model": f"{n_layer}L/{n_head}H/{n_embd}D",
        "n_tokens": n_tokens, "ll_lambda": ll_lambda,
        "train_result": result,
        "fm_result": fm_result,
    }
    with open(os.path.join(save_dir, "fm_wt_ll_results.json"), "w") as f:
        json.dump(combined, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/fm_wt_ll_results.json")
    return combined
