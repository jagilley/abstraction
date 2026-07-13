"""Per-level loss decomposition for RHM autoregressive models.

Since we know the DGP, each next-token prediction maps to a specific level
of the hierarchy based on which boundary it crosses:

  Level k = predictions where the target is the first token of a new s^k subtree.
  Formally: position p has level = v_s(p), the s-adic valuation (trailing zeros in base s).

  Level 0:   within an s-tuple (easiest, local rule)
  Level L-1: crosses root boundary (hardest, needs full-sequence context)

For s=2, L=6 (seq_len=64):
  Level 0: 32 positions  (half the sequence)
  Level 1: 16 positions
  Level 2:  8 positions
  Level 3:  4 positions
  Level 4:  2 positions
  Level 5:  1 position   (root boundary)

Two experiments:
  per_level_single   — train model, evaluate per-level loss at convergence
  per_level_trajectory — track per-level loss over training (bottom-up learning?)
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "torch==2.7.0")
    .add_local_python_source("rhm")
)

app = modal.App("rhm-per-level-loss", image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _position_levels(seq_len, s):
    """Hierarchical level for predicting each position 1..seq_len-1.

    Returns array of length seq_len-1 where entry t = s-adic valuation of (t+1).
    """
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
    """Per-level cross-entropy and accuracy on sequence-aligned data.

    Feeds complete sequences (length seq_len), uses logits[:, :-1] to predict
    positions 1..seq_len-1. Groups by hierarchical level.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    levels = _position_levels(seq_len, s)
    max_level = int(levels.max())

    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    per_pos_loss_sum = np.zeros(seq_len - 1)
    per_pos_correct_sum = np.zeros(seq_len - 1)
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

            per_pos_loss_sum += pos_loss.sum(dim=0).cpu().numpy()
            per_pos_correct_sum += correct.sum(dim=0).cpu().numpy()
            n_samples += bs

    per_pos_loss = per_pos_loss_sum / n_samples
    per_pos_acc = per_pos_correct_sum / n_samples

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
        }

    return {
        "per_level": level_results,
        "overall_loss": float(per_pos_loss.mean()),
        "overall_accuracy": float(per_pos_acc.mean()),
        "per_position_loss": per_pos_loss.tolist(),
        "per_position_accuracy": per_pos_acc.tolist(),
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


# ---------------------------------------------------------------------------
# Single setting: train and evaluate
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="T4", timeout=7200, memory=16384)
def per_level_single(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4,
    n_eval_sequences: int = 10000,
    seed: int = 42,
):
    """Train a model and evaluate per-level loss at convergence."""
    import torch
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

    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"Training: {key}, {n_layer}L/{n_head}H/{n_embd}D, {n_steps} steps")

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - seq_len - 1, (batch_size,))
        x = torch.stack([split_data[i:i + seq_len] for i in ix])
        y = torch.stack([split_data[i + 1:i + seq_len + 1] for i in ix])
        return x.to(device), y.to(device)

    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vl = sum(float(model(*get_batch(val_data))[1]) for _ in range(10)) / 10
            print(f"  step {step}: train={loss:.4f} val={vl:.4f}")

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs = generate_sequences_batched(rules, n_eval_sequences, seed=12345)

    result = _eval_per_level(model, eval_seqs, seq_len, s, L, v, batch_size, device)
    result.update({
        "setting": key, "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
        "model": f"{n_layer}L/{n_head}H/{n_embd}D", "n_steps": n_steps,
        "n_eval_sequences": n_eval_sequences,
        "uniform_baseline": float(np.log(v)),
    })

    uniform = result["uniform_baseline"]
    print(f"\n{'='*60}")
    print(f"PER-LEVEL LOSS: {key} ({n_layer}L/{n_head}H/{n_embd}D)")
    print(f"Overall: loss={result['overall_loss']:.4f} acc={result['overall_accuracy']:.4f}")
    print(f"Uniform baseline: {uniform:.4f}")
    print(f"{'Level':>6}  {'Pos':>5}  {'Loss':>8}  {'Acc':>8}  {'vs unif':>8}")
    print("-" * 43)
    for lvl in sorted(result["per_level"].keys()):
        r = result["per_level"][lvl]
        reduction = (uniform - r["avg_loss"]) / uniform * 100
        print(f"{lvl:>6}  {r['n_positions']:>5}  {r['avg_loss']:>8.4f}  "
              f"{r['avg_accuracy']:>8.4f}  {reduction:>7.1f}%")

    save_dir = f"{DATA_DIR}/rhm_per_level_loss"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{key}_{n_layer}L{n_head}H{n_embd}D.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return result


# ---------------------------------------------------------------------------
# Training trajectory: per-level loss at each checkpoint
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=16384)
def per_level_trajectory(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4,
    n_eval_sequences: int = 5000,
    seed: int = 42,
    eval_interval: int = 200,
    n_steps_override: int = 0,
    weight_decay: float = 0.01,
):
    """Track per-level loss over training to reveal bottom-up learning."""
    import torch
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
    if n_steps_override > 0:
        n_steps = n_steps_override

    checkpoint_fracs = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0]
    checkpoint_steps = sorted(set(
        round(f * n_steps / eval_interval) * eval_interval for f in checkpoint_fracs
    ))

    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs = generate_sequences_batched(rules, n_eval_sequences, seed=12345)

    print(f"Per-level trajectory: {key}, {n_layer}L/{n_head}H/{n_embd}D")
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
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step in checkpoint_steps:
            model.eval()
            with torch.no_grad():
                vl = sum(float(model(*get_batch(val_data))[1]) for _ in range(10)) / 10

            per_level = _eval_per_level(model, eval_seqs, seq_len, s, L, v, batch_size, device)

            entry = {
                "step": step,
                "val_loss": vl,
                "overall_loss": per_level["overall_loss"],
                "overall_accuracy": per_level["overall_accuracy"],
                "per_level": per_level["per_level"],
            }
            trajectory.append(entry)

            level_str = "  ".join(
                f"L{lvl}={per_level['per_level'][lvl]['avg_loss']:.3f}"
                for lvl in sorted(per_level["per_level"].keys())
            )
            print(f"  step {step:6d}: val={vl:.4f}  {level_str}")

    uniform = float(np.log(v))
    max_level = max(int(lvl) for t in trajectory for lvl in t["per_level"].keys())

    print(f"\n{'='*90}")
    print(f"PER-LEVEL LOSS TRAJECTORY: {key} ({n_layer}L/{n_head}H/{n_embd}D)")
    print(f"Uniform baseline: {uniform:.4f}")
    print(f"{'='*90}")

    header = f"{'step':>6}  {'val':>7}" + "".join(f"  {'L'+str(l)+' loss':>8}" for l in range(max_level + 1))
    header += "  |" + "".join(f"  {'L'+str(l)+' acc':>7}" for l in range(max_level + 1))
    print(header)
    print("-" * len(header))
    for t in trajectory:
        row = f"{t['step']:>6}  {t['val_loss']:>7.4f}"
        for lvl in range(max_level + 1):
            if lvl in t["per_level"]:
                row += f"  {t['per_level'][lvl]['avg_loss']:>8.4f}"
            else:
                row += f"  {'--':>8}"
        row += "  |"
        for lvl in range(max_level + 1):
            if lvl in t["per_level"]:
                row += f"  {t['per_level'][lvl]['avg_accuracy']:>7.4f}"
            else:
                row += f"  {'--':>7}"
        print(row)

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
        "model": f"{n_layer}L/{n_head}H/{n_embd}D", "n_steps": n_steps,
        "uniform_baseline": uniform,
        "trajectory": trajectory,
    }

    save_dir = f"{DATA_DIR}/rhm_per_level_loss"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"trajectory_{key}_{n_layer}L{n_head}H{n_embd}D.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nSaved to {save_dir}/{fname}")
    return result


# ---------------------------------------------------------------------------
# Sweep: cross-setting comparison
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=4096)
def per_level_sweep(
    v: int = 8, s: int = 2,
    settings: str = "L4_m2,L4_m8,L6_m2,L6_m4,L8_m2",
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
):
    """Compare per-level loss across DGP settings."""
    configs = []
    for setting in settings.split(","):
        parts = setting.strip().split("_")
        L_val = int(parts[0][1:])
        m_val = int(parts[1][1:])
        configs.append((L_val, m_val))

    print(f"Per-level sweep: {len(configs)} settings, {n_layer}L/{n_head}H/{n_embd}D")

    handles = []
    for L_val, m_val in configs:
        seq_len = s ** L_val
        n_tokens = min(20_000_000, max(500_000, 5000 * seq_len))
        h = per_level_single.spawn(
            v=v, s=s, depth=L_val, m=m_val, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        )
        handles.append((L_val, m_val, h))

    results = []
    for L_val, m_val, h in handles:
        results.append(h.get())

    all_levels = sorted(set(
        int(lvl) for r in results for lvl in r["per_level"].keys()
    ))

    print(f"\n{'='*100}")
    print("CROSS-SETTING PER-LEVEL LOSS COMPARISON")
    print(f"{'='*100}")

    header = f"{'Setting':>16}  {'overall':>8}" + "".join(f"  {'L'+str(l):>7}" for l in all_levels)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['setting']:>16}  {r['overall_loss']:>8.4f}"
        for lvl in all_levels:
            if lvl in r["per_level"]:
                row += f"  {r['per_level'][lvl]['avg_loss']:>7.4f}"
            else:
                row += f"  {'--':>7}"
        print(row)

    print(f"\nUniform baseline: {results[0]['uniform_baseline']:.4f}")

    return results
