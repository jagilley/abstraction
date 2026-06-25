"""RHM RL Ratchet: REINFORCE with FM supervision for compositional learning.

Tests whether RL's sparse-but-rich supervision + FM process supervision
produces cleaner hierarchical decomposition than dense NTP + FM.

Key hypothesis: RL reward is sparse but rich — it constrains the entire
generation globally, like a classification signal. The FM's process
supervision provides credit assignment that the sparse reward cannot.
Together, they should produce intermediate representations that cleanly
decompose the DGP's hierarchical structure (measurable via per-layer
feature eta²), unlike vanilla RL which may find entangled solutions.

Design:
  Phase 1: Pre-train with shared NTP (pretrain_steps steps)
  Phase 2: Fine-tune with 4 conditions, all compute-matched on gradient steps:
    RL_FM:  4 cycles × (wake_steps RL wake + distill_steps distill sleep) + repoint
    RL:     total_finetune_steps steps RL only (no FM, no ratchet)
    NTP_FM: 4 cycles × (wake_steps NTP wake + distill_steps distill sleep) + repoint
    OL:     total_finetune_steps steps NTP continued

  Generation: prefix = first seq_len//2 tokens, generate remaining half
  Reward: fraction of correct tokens in suffix (sequence-level REINFORCE)

  Model: 6L/6H/192D GPT (~2.68M params)
  FM: 1L/1H/16D causal (50.4K params, 1.9%), post_block0 -> post_block5
  DGP: L=6, m=2, v=8, s=2 (seq_len=64)

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_rl_ratchet::rhm_rl_ratchet
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

app = modal.App("rhm-rl-ratchet", image=image)


# ======================================================================
# Helpers (shared with other RHM experiments)
# ======================================================================

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


def _suffix_position_levels(prefix_len, seq_len, s):
    """Hierarchy level for each suffix position."""
    levels = []
    for i in range(prefix_len, seq_len):
        p = i
        level = 0
        while p > 0 and p % s == 0:
            p //= s
            level += 1
        levels.append(level)
    return levels


def _eval_per_level(model, eval_seqs, seq_len, s, L, v, batch_size, device):
    """Per-level CE using hierarchy-aligned sequences."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    levels = _position_levels(seq_len, s)
    max_level = int(levels.max())
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    per_pos_loss_sum = np.zeros(seq_len - 1)
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
            target_log_probs = log_probs.gather(
                2, targets.unsqueeze(-1)).squeeze(-1)
            per_pos_loss_sum += (-target_log_probs).sum(dim=0).cpu().numpy()
            n_samples += bs

    per_pos_loss = per_pos_loss_sum / n_samples
    level_results = {}
    for lvl in range(max_level + 1):
        mask = levels == lvl
        if mask.sum() == 0:
            continue
        level_results[lvl] = {"loss": float(per_pos_loss[mask].mean())}
    return {"per_level": level_results, "overall_loss": float(per_pos_loss.mean())}


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


def _compute_hierarchy_eta2(acts_np, level_features, level_rules, s, L):
    """Compute feature/rule eta² from activations at last sequence position."""
    import numpy as np
    n_seq, seq_len, d_model = acts_np.shape
    last = acts_np[:, -1, :]
    last_mean = last.mean(axis=0)
    ss_total = float(np.sum((last - last_mean) ** 2))
    if ss_total < 1e-12:
        return {f"level_{ell}": {"feature_eta2": 0.0, "rule_eta2": 0.0}
                for ell in range(L)}

    def eta2_between(data, labels, grand_mean):
        ss = 0.0
        for val in np.unique(labels):
            mask = labels == val
            group_mean = data[mask].mean(axis=0)
            ss += float(mask.sum()) * float(
                np.sum((group_mean - grand_mean) ** 2))
        return ss

    results = {}
    for ell in range(L):
        s_power = s ** (L - ell)
        last_ancestor = (seq_len - 1) // s_power
        results[f"level_{ell}"] = {
            "feature_eta2": eta2_between(
                last, level_features[ell][:, last_ancestor], last_mean
            ) / ss_total,
            "rule_eta2": eta2_between(
                last, level_rules[ell][:, last_ancestor], last_mean
            ) / ss_total,
        }
    return results


# ======================================================================
# New: generation and per-layer eta²
# ======================================================================

def _generate_suffix(model, prefix, suffix_len, temperature,
                     fm, ugate, cerebellar_input_block, inject_after_block):
    """Autoregressively generate suffix_len tokens given prefix."""
    import torch
    import torch.nn.functional as F

    x = prefix.clone()
    generated = []

    for _ in range(suffix_len):
        if fm is not None and ugate is not None:
            def ug_cb(act, _fm=fm, _ug=ugate):
                fp = _fm(act)
                inj, _ = _ug(act, fp)
                return inj
            logits, _, _ = model(
                x, return_intermediates=True,
                cerebellar_fn=ug_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )
        else:
            logits, _ = model(x)

        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token)
        x = torch.cat([x, next_token], dim=1)

    return torch.cat(generated, dim=1)


def _compute_per_layer_eta2(model, eval_seqs, level_features, level_rules,
                            n_layer, s, L, batch_size, device):
    """Feature eta² at each main model layer for each hierarchy level."""
    import torch
    import numpy as np

    model.eval()
    layer_keys = [f"post_block{i}" for i in range(n_layer)]
    layer_acts = {k: [] for k in layer_keys}

    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i + batch_size]
            if batch.shape[0] < 2:
                continue
            _, _, intermediates = model(batch, return_intermediates=True)
            for k in layer_keys:
                layer_acts[k].append(intermediates[k].cpu().numpy())

    results = {}
    for layer_key in layer_keys:
        acts = np.concatenate(layer_acts[layer_key], axis=0)
        n_complete = acts.shape[0]
        results[layer_key] = _compute_hierarchy_eta2(
            acts,
            [lf[:n_complete] for lf in level_features],
            [lr[:n_complete] for lr in level_rules],
            s=s, L=L,
        )
    return results


# ======================================================================
# Main experiment
# ======================================================================

@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,
    memory=32768,
)
def rhm_rl_ratchet(
    # DGP
    v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
    n_tokens: int = 20_000_000,
    # Model
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    # FM
    predict_from: str = "post_block0",
    predict_to: str = "post_block5",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16,
    fwd_n_head: int = 1, fwd_mlp_mult: float = 0.5,
    # Pre-training
    pretrain_steps: int = 10000,
    # Fine-tuning
    n_cycles: int = 4,
    wake_steps: int = 2000,
    distill_steps: int = 500,
    retrain_steps: int = 2000,
    # Training
    lr: float = 3e-4, fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4, distill_alpha: float = 0.5,
    lambda_local: float = 1.0, ug_hidden: int = 64,
    batch_size: int = 64,
    # RL
    temperature: float = 1.0,
    rl_loss_scale: float = 1.0,
    lambda_ntp: float = 1.0,
    ntp_mask_rate: float = 0.95,
    only_rl_fm: bool = False,
    # Eval
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    probe_batches: int = 30,
    probe_steps: int = 500,
    n_eval_sequences: int = 5000,
    n_gen_eval_seqs: int = 1000,
    seed: int = 42,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from rhm.rhm_data import generate_sequences_batched
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    prefix_len = seq_len // 2
    suffix_len = seq_len - prefix_len
    cerebellar_input_block = int(predict_from.replace("post_block", ""))

    steps_per_cycle = wake_steps + distill_steps
    total_finetune_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    suffix_levels = _suffix_position_levels(prefix_len, seq_len, s)
    max_suffix_level = max(suffix_levels)

    print(f"RHM RL RATCHET on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D, "
          f"{predict_from}->{predict_to}, inject after block {inject_after_block}")
    print(f"  Pre-train: {pretrain_steps} NTP steps")
    print(f"  Fine-tune: {n_cycles} cycles x ({wake_steps} wake + {distill_steps} sleep)"
          f" = {total_finetune_steps} steps")
    print(f"  Generation: prefix={prefix_len}, suffix={suffix_len}")
    print(f"  lambda_ntp={lambda_ntp}, ntp_mask_rate={ntp_mask_rate} "
          f"(sparse NTP regularization during RL)")
    print(f"  Suffix levels: {dict(zip(range(max_suffix_level+1), [suffix_levels.count(l) for l in range(max_suffix_level+1)]))}")

    # ------------------------------------------------------------------
    # UnifiedGate (same as sparse ratchet)
    # ------------------------------------------------------------------
    class UnifiedGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            self.projection = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            n_params = sum(p.numel() for p in self.parameters())
            print(f"UnifiedGate: {n_params/1e3:.1f}K params")

        def forward(self, activations, fwd_pred):
            gate_w = torch.sigmoid(self.gate_net(
                torch.cat([activations, fwd_pred], dim=-1)))
            injection = gate_w * self.projection(fwd_pred)
            return injection, gate_w

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading data...")
    volume.reload()
    _ensure_corpus(v, s, L, m, n_tokens)

    data = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"  {len(data):,} tokens, train={len(train_data):,}, val={len(val_data):,}")

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(
        rules, n_eval_sequences, seed=12345)
    print(f"  {n_eval_sequences} eval sequences with hierarchy traces")

    # Generate sequence-aligned data for RL training
    n_rl_seqs = 100000
    print(f"  Generating {n_rl_seqs} sequence-aligned RL training sequences...")
    rl_seqs = generate_sequences_batched(rules, n_rl_seqs, seed=seed + 5000)
    rl_seqs_tensor = torch.from_numpy(rl_seqs.astype(np.int64))
    print(f"  RL data: {n_rl_seqs} sequences x {seq_len} tokens")

    # ------------------------------------------------------------------
    # Pre-generate batch indices (shared across conditions)
    # ------------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    repoint_gen = torch.Generator().manual_seed(seed + 3)
    rl_gen = torch.Generator().manual_seed(seed + 4)

    max_ntp_steps = pretrain_steps + total_finetune_steps
    train_indices = [
        torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(max_ntp_steps)
    ]
    max_repoint_steps = n_cycles * retrain_steps
    repoint_indices = [
        torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                       generator=repoint_gen)
        for _ in range(max_repoint_steps)
    ]
    max_rl_steps = total_finetune_steps
    rl_indices = [
        torch.randint(n_rl_seqs, (batch_size,), generator=rl_gen)
        for _ in range(max_rl_steps)
    ]

    n_evals = 300
    eval_indices = [
        [torch.randint(len(val_data) - seq_len - 1, (batch_size,),
                        generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]
    probe_indices = [
        torch.randint(len(val_data) - seq_len - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]

    def make_ntp_batch(indices_list, step, split_data):
        indices = indices_list[step]
        x = torch.stack([split_data[i:i + seq_len] for i in indices])
        y = torch.stack([split_data[i + 1:i + seq_len + 1] for i in indices])
        return x.to(device), y.to(device)

    def make_rl_batch(step):
        indices = rl_indices[step]
        seqs = rl_seqs_tensor[indices].to(device)
        prefix = seqs[:, :prefix_len]
        true_suffix = seqs[:, prefix_len:]
        return seqs, prefix, true_suffix

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    def make_gpt():
        return GPT(v, seq_len, n_layer, n_head, n_embd).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=seq_len,
        ).to(device)

    # ------------------------------------------------------------------
    # Save initial weights
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    init_model = make_gpt()
    init_fm = make_fm()
    init_model_state = {k: v_.cpu().clone()
                        for k, v_ in init_model.state_dict().items()}
    init_fm_state = {k: v_.cpu().clone()
                     for k, v_ in init_fm.state_dict().items()}
    del init_model, init_fm
    torch.cuda.empty_cache()

    # ==================================================================
    # Shared evaluation helpers
    # ==================================================================

    eval_counter = [0]

    def eval_model(model):
        model.eval()
        idx = eval_counter[0] % len(eval_indices)
        eval_counter[0] += 1
        total_loss = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                x = torch.stack([val_data[i:i + seq_len]
                                 for i in eval_indices[idx][bi]]).to(device)
                y = torch.stack([val_data[i + 1:i + seq_len + 1]
                                 for i in eval_indices[idx][bi]]).to(device)
                _, loss = model(x, y)
                total_loss += loss.item()
        return total_loss / n_eval_batches

    def measure_robustness(model, label, n_batches=20):
        model.eval()
        eps_list = [0.5, 1.0, 2.0]
        results = {}
        with torch.no_grad():
            for eps in eps_list:
                deltas = []
                for bi in range(min(probe_batches, n_batches)):
                    x = torch.stack([val_data[i:i + seq_len]
                                     for i in probe_indices[bi]]).to(device)
                    y = torch.stack([val_data[i + 1:i + seq_len + 1]
                                     for i in probe_indices[bi]]).to(device)
                    _, base_loss = model(x, y)
                    torch.manual_seed(seed + bi + 5000)
                    noise = torch.randn(
                        batch_size, seq_len, n_embd, device=device) * eps

                    def noise_fn(act, _n=noise):
                        return _n

                    _, pert_loss, _ = model(
                        x, y, return_intermediates=True,
                        cerebellar_fn=noise_fn,
                        cerebellar_input_block=inject_after_block,
                        cerebellar_inject_block=inject_after_block,
                    )
                    deltas.append(pert_loss.item() - base_loss.item())
                results[str(eps)] = float(np.mean(deltas))
        print(f"  [{label:>15s}] rob: "
              + " | ".join(f"eps={e}:{results[str(e)]:+.4f}" for e in eps_list))
        return results

    def self_knowledge_probes(model, fm, label):
        model.eval(); fm.eval()
        probe_layers = ["post_block0", f"post_block{n_layer - 1}"]
        acts_by_layer = {k: [] for k in probe_layers}
        residuals = []
        with torch.no_grad():
            for pidx in probe_indices:
                x = torch.stack([val_data[i:i + seq_len]
                                 for i in pidx]).to(device)
                y = torch.stack([val_data[i + 1:i + seq_len + 1]
                                 for i in pidx]).to(device)
                _, _, vi = model(x, y, return_intermediates=True)
                pred = fm(vi[predict_from])
                residuals.append(
                    (vi[predict_to] - pred).reshape(-1, n_embd).cpu())
                for k in probe_layers:
                    acts_by_layer[k].append(vi[k].reshape(-1, n_embd).cpu())

        all_res = torch.cat(residuals)
        for k in probe_layers:
            acts_by_layer[k] = torch.cat(acts_by_layer[k])

        n_total = all_res.shape[0]
        n_train = int(0.8 * n_total)
        perm = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed))
        tr, te = perm[:n_train], perm[n_train:]

        r2_by_layer = {}
        for lk in probe_layers:
            X = acts_by_layer[lk]
            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            X_tr, Y_tr = X[tr].to(device), all_res[tr].to(device)
            X_te, Y_te = X[te].to(device), all_res[te].to(device)
            bs = min(4096, n_train)
            for _ in range(probe_steps):
                si = torch.randint(n_train, (bs,))
                ploss = F.mse_loss(probe(X_tr[si]), Y_tr[si])
                opt.zero_grad(); ploss.backward(); opt.step()
            with torch.no_grad():
                mse = F.mse_loss(probe(X_te), Y_te).item()
                var_val = Y_te.var().item()
                r2 = 1.0 - mse / var_val if var_val > 0 else 0.0
            r2_by_layer[lk] = r2
            del probe, X_tr, Y_tr, X_te, Y_te
            torch.cuda.empty_cache()

        print(f"  [{label}] SK: "
              + " | ".join(f"{k}={v_:.3f}" for k, v_ in r2_by_layer.items()))
        return r2_by_layer

    def compute_residual_stats(model, fm, label):
        model.eval(); fm.eval()
        eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
        all_res = []
        all_cos = []
        with torch.no_grad():
            for i in range(0, len(eval_data), batch_size):
                batch = eval_data[i:i + batch_size]
                if batch.shape[0] < 2:
                    continue
                _, _, vi = model(batch, return_intermediates=True)
                pred = fm(vi[predict_from])
                tgt = vi[predict_to]
                res = tgt - pred
                all_res.append(res.cpu().numpy())
                all_cos.append(
                    float(F.cosine_similarity(pred, tgt, dim=-1).mean()))

        res_np = np.concatenate(all_res, axis=0)
        mean_cos = float(np.mean(all_cos))
        mean_norm = float(np.sqrt((res_np ** 2).sum(axis=-1)).mean())

        res_flat = res_np.reshape(-1, n_embd)
        n = min(50000, res_flat.shape[0])
        rng = np.random.RandomState(42)
        idx = rng.choice(res_flat.shape[0], n, replace=False)
        sub = res_flat[idx] - res_flat[idx].mean(axis=0)
        _, S, _ = np.linalg.svd(sub, full_matrices=False)
        S_norm = S / S.sum()
        eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))

        stats = {
            "fwd_cosine": mean_cos, "res_norm": mean_norm,
            "eff_rank": eff_rank, "eff_rank_pct": eff_rank / n_embd * 100,
        }
        print(f"  [{label}] cos={mean_cos:.4f} norm={mean_norm:.2f} "
              f"rank={eff_rank:.1f}/{n_embd}")
        return stats

    def train_fresh_fm(model, steps, fm_seed, label):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        for step in range(steps):
            fm.train()
            x = torch.stack([train_data[i:i + seq_len]
                             for i in repoint_indices[step]]).to(device)
            with torch.no_grad():
                _, _, vi = model(x, return_intermediates=True)
                source = vi[predict_from]
                target = vi[predict_to]
            pred = fm(source)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()
        fm.eval()
        cos_total = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                x = torch.stack([val_data[i:i + seq_len]
                                 for i in eval_indices[0][bi]]).to(device)
                _, _, vi = model(x, return_intermediates=True)
                p = fm(vi[predict_from])
                cos_total += F.cosine_similarity(
                    p, vi[predict_to], dim=-1).mean().item()
        cos = cos_total / n_eval_batches
        print(f"  [{label}] Fresh FM cos: {cos:.4f}")
        for p in model.parameters():
            p.requires_grad = True
        return fm, cos

    def get_ug_gate_stats(ugate, model, fwd_model):
        ugate.eval(); model.eval(); fwd_model.eval()
        all_gw = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                x = torch.stack([val_data[i:i + seq_len]
                                 for i in probe_indices[bi]]).to(device)
                _, _, vi = model(x, return_intermediates=True)
                src = vi[predict_from]
                pred = fwd_model(src)
                _, gw = ugate(src, pred)
                all_gw.append(gw.mean(dim=1).cpu())
        gw_cat = torch.cat(all_gw, dim=0)
        dim_mean = gw_cat.mean(dim=0)
        return {
            "mean": float(gw_cat.mean()),
            "std": float(gw_cat.std()),
            "sparse_01": float((dim_mean < 0.1).float().mean()),
        }

    def eval_generation_accuracy(model, label, fm=None, ugate=None):
        """Per-level generation accuracy on eval sequences."""
        model.eval()
        if fm is not None:
            fm.eval(); ugate.eval()
        eval_data = torch.from_numpy(
            eval_seqs[:n_gen_eval_seqs].astype(np.int64)).to(device)

        all_correct = []
        with torch.no_grad():
            for i in range(0, n_gen_eval_seqs, batch_size):
                batch = eval_data[i:i + batch_size]
                prefix = batch[:, :prefix_len]
                true_suffix = batch[:, prefix_len:]
                generated = _generate_suffix(
                    model, prefix, suffix_len, temperature,
                    fm, ugate, cerebellar_input_block, inject_after_block)
                correct = (generated == true_suffix).float()
                all_correct.append(correct.cpu())

        all_correct = torch.cat(all_correct, dim=0)
        results = {"overall": float(all_correct.mean())}
        for lvl in range(max_suffix_level + 1):
            mask = torch.tensor([l == lvl for l in suffix_levels])
            if mask.sum() > 0:
                results[f"L{lvl}"] = float(all_correct[:, mask].mean())

        acc_str = " ".join(f"L{k}={results.get(f'L{k}',0):.3f}"
                           for k in range(max_suffix_level + 1))
        print(f"  [{label}] gen_acc: overall={results['overall']:.3f} {acc_str}")
        return results

    def full_checkpoint_eval(model, cond_label, cycle_idx, fm=None, ugate=None,
                             is_rl=False):
        """Full evaluation at a cycle checkpoint."""
        ckpt = {}

        # NTP val loss
        vl = eval_model(model)
        ckpt["val_loss"] = vl

        # Per-level NTP loss
        pl = _eval_per_level(model, eval_seqs, seq_len, s, L, v, batch_size, device)
        ckpt["per_level"] = pl

        # Per-layer feature eta²
        per_layer = _compute_per_layer_eta2(
            model, eval_seqs, level_features, level_rules,
            n_layer, s, L, batch_size, device)
        ckpt["per_layer_eta2"] = per_layer

        # Generation accuracy (for all conditions, to compare)
        gen_acc = eval_generation_accuracy(
            model, f"{cond_label} c{cycle_idx+1}",
            fm=fm if fm is not None else None,
            ugate=ugate if ugate is not None else None)
        ckpt["generation_accuracy"] = gen_acc

        # Generation accuracy WITHOUT FM (standalone model)
        if fm is not None:
            gen_acc_standalone = eval_generation_accuracy(
                model, f"{cond_label} c{cycle_idx+1} (no FM)")
            ckpt["generation_accuracy_standalone"] = gen_acc_standalone

        # FM-dependent measurements
        is_final = (cycle_idx == n_cycles - 1)
        if fm is not None:
            ckpt["residual_stats"] = compute_residual_stats(
                model, fm, f"{cond_label} c{cycle_idx+1}")
            gs = get_ug_gate_stats(ugate, model, fm)
            ckpt["gate_stats"] = gs
            print(f"    Gate: mean={gs['mean']:.3f} "
                  f"sparse(<0.1)={gs['sparse_01']:.3f}")

        if is_final:
            ckpt["robustness"] = measure_robustness(model, f"{cond_label} final")
            if fm is not None:
                ckpt["self_knowledge"] = self_knowledge_probes(
                    model, fm, f"{cond_label} final")

        level_str = " ".join(
            f"L{lvl}={pl['per_level'][lvl]['loss']:.3f}"
            for lvl in sorted(pl['per_level'].keys()))
        print(f"    val={vl:.4f} {level_str}")

        return ckpt

    # ==================================================================
    # Phase 1: Pre-training (shared across all conditions)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  PRE-TRAINING: {pretrain_steps} NTP steps")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(init_model_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for step in range(pretrain_steps):
        model.train()
        x = torch.stack([train_data[i:i + seq_len]
                         for i in train_indices[step]]).to(device)
        y = torch.stack([train_data[i + 1:i + seq_len + 1]
                         for i in train_indices[step]]).to(device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0 or step == pretrain_steps - 1:
            vl = eval_model(model)
            print(f"  step {step:5d}: val={vl:.4f}")

    # Evaluate pre-trained model
    pretrain_pl = _eval_per_level(
        model, eval_seqs, seq_len, s, L, v, batch_size, device)
    level_str = " ".join(
        f"L{lvl}={pretrain_pl['per_level'][lvl]['loss']:.3f}"
        for lvl in sorted(pretrain_pl['per_level'].keys()))
    print(f"  Pre-trained per-level: {level_str}")

    pretrain_gen = eval_generation_accuracy(model, "pretrained")

    pretrained_state = {k: v_.cpu().clone()
                        for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Saved pre-trained checkpoint")

    # ==================================================================
    # Phase 2: Fine-tuning
    # ==================================================================
    all_results = {}

    # ------------------------------------------------------------------
    # Condition 1: RL_FM (ratchet)
    # ------------------------------------------------------------------
    cond = "RL_FM"
    print(f"\n{'='*60}")
    print(f"  {cond}: RL + FM ratchet ({n_cycles} cycles)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    fm = make_fm()
    fm.load_state_dict(init_fm_state)
    ugate = UnifiedGate(n_embd, ug_hidden).to(device)

    checkpoints = {}
    rl_step_idx = 0
    rl_baseline = None

    for cycle in range(n_cycles):
        print(f"\n  --- {cond} Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: RL with FM ===
        print(f"  [WAKE] {wake_steps} RL steps")
        main_params = list(model.parameters()) + list(ugate.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            # --- Generate suffix with FM injection (no grad) ---
            model.eval(); fm.eval(); ugate.eval()
            full_seqs, prefix, true_suffix = make_rl_batch(rl_step_idx)

            with torch.no_grad():
                generated = _generate_suffix(
                    model, prefix, suffix_len, temperature,
                    fm, ugate, cerebellar_input_block, inject_after_block)

            # --- Compute reward ---
            correct = (generated == true_suffix).float()
            reward = correct.mean(dim=1)  # (B,) per-sequence fraction correct
            if rl_baseline is None:
                rl_baseline = reward.mean().item()
            else:
                rl_baseline = 0.95 * rl_baseline + 0.05 * reward.mean().item()
            advantage = (reward - rl_baseline).detach()

            # --- Scoring pass with FM injection (with grad) ---
            model.train(); fm.train(); ugate.train()
            scored_seq = torch.cat([prefix, generated], dim=1)
            fwd_cache = {}

            def ug_cb(act, _cache=fwd_cache):
                fp = fm(act.detach())
                _cache["pred"] = fp
                inj, gw = ugate(act.detach(), fp.detach())
                _cache["gate_w"] = gw
                return inj

            logits, _, intermediates = model(
                scored_seq, return_intermediates=True,
                cerebellar_fn=ug_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

            # REINFORCE loss
            suffix_logits = logits[:, prefix_len - 1:prefix_len - 1 + suffix_len, :]
            log_probs = F.log_softmax(suffix_logits, dim=-1)
            gen_log_probs = log_probs.gather(
                2, generated.unsqueeze(-1)).squeeze(-1)
            rl_loss = -(advantage.unsqueeze(1) * gen_log_probs).mean()

            # Local loss (uniform, gate-scaled)
            fwd_pred = fwd_cache["pred"]
            target_acts = intermediates[predict_to]
            gate_w = fwd_cache["gate_w"]
            gw_scalar = gate_w.detach().mean()
            r = target_acts - fwd_pred.detach()
            local_loss = (gw_scalar * r ** 2).mean()

            # Sparse NTP regularization on a separate batch
            ntp_step = pretrain_steps + cycle * steps_per_cycle + step
            ntp_x = torch.stack([train_data[i:i + seq_len]
                                 for i in train_indices[ntp_step]]).to(device)
            ntp_y = torch.stack([train_data[i + 1:i + seq_len + 1]
                                 for i in train_indices[ntp_step]]).to(device)
            ntp_logits, _ = model(ntp_x)
            ntp_per_pos = F.cross_entropy(
                ntp_logits.view(-1, v), ntp_y.view(-1), reduction='none')
            if ntp_mask_rate > 0:
                ntp_mask = (torch.rand(ntp_per_pos.shape[0],
                                       device=device) >= ntp_mask_rate).float()
                ntp_loss = (ntp_per_pos * ntp_mask).sum() / ntp_mask.sum().clamp(min=1)
            else:
                ntp_loss = ntp_per_pos.mean()

            total_loss = (rl_loss_scale * rl_loss
                          + lambda_ntp * ntp_loss
                          + lambda_local * local_loss)

            opt_main.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            # FM co-training
            fwd_loss = F.mse_loss(fwd_pred, target_acts.detach())
            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == wake_steps - 1:
                vl = eval_model(model)
                mean_reward = reward.mean().item()
                print(f"    step {rl_step_idx:5d} (w{step}): "
                      f"val={vl:.4f} reward={mean_reward:.3f} "
                      f"rl={rl_loss.item():.4f} ntp={ntp_loss.item():.4f} "
                      f"ll={local_loss.item():.5f} gw={gw_scalar:.3f}")

            rl_step_idx += 1

        # === SLEEP: Distillation ===
        print(f"  [SLEEP] Distillation ({distill_steps} steps)")
        teacher = make_gpt()
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_fm = make_fm()
        teacher_fm.load_state_dict(fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters():
            p.requires_grad = False
        teacher_ugate = UnifiedGate(n_embd, ug_hidden).to(device)
        teacher_ugate.load_state_dict(ugate.state_dict())
        teacher_ugate.eval()
        for p in teacher_ugate.parameters():
            p.requires_grad = False

        student = make_gpt()
        student.load_state_dict(model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        ntp_step_offset = pretrain_steps + cycle * steps_per_cycle + wake_steps

        for step in range(distill_steps):
            student.train()
            x = torch.stack([train_data[i:i + seq_len]
                             for i in train_indices[ntp_step_offset + step]]).to(device)
            y = torch.stack([train_data[i + 1:i + seq_len + 1]
                             for i in train_indices[ntp_step_offset + step]]).to(device)

            with torch.no_grad():
                def t_cb(act):
                    fp = teacher_fm(act)
                    inj, _ = teacher_ugate(act, fp)
                    return inj
                teacher_logits, _, _ = teacher(
                    x, return_intermediates=True,
                    cerebellar_fn=t_cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )

            student_logits, _ = student(x)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            ce_loss = F.cross_entropy(
                student_logits.view(-1, v), y.view(-1))
            loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss

            opt_student.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_student.step()

        model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_ugate, student, opt_student
        del opt_main, opt_fwd
        torch.cuda.empty_cache()

        # === REPOINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
        fm, fm_cos = train_fresh_fm(
            model, retrain_steps, fm_seed, f"{cond} c{cycle+1}")

        # === CHECKPOINT ===
        print(f"\n  {cond} CHECKPOINT cycle {cycle + 1}")
        ckpt = full_checkpoint_eval(
            model, cond, cycle, fm=fm, ugate=ugate, is_rl=True)
        ckpt["fm_cosine"] = fm_cos
        checkpoints[(cycle + 1) * steps_per_cycle] = ckpt

    all_results[cond] = {"checkpoints": checkpoints}
    del model, fm, ugate
    torch.cuda.empty_cache()

    if only_rl_fm:
        print("\n  [SKIP] Conditions RL, NTP_FM, OL (only_rl_fm=True)")
        # Jump to save results
        save_dir = f"{DATA_DIR}/rhm_rl_ratchet/{key}_rl_fm_only"
        os.makedirs(save_dir, exist_ok=True)
        result = {
            "config": {
                "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
                "prefix_len": prefix_len, "suffix_len": suffix_len,
                "model": f"{n_layer}L/{n_head}H/{n_embd}D",
                "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D",
                "predict_from": predict_from, "predict_to": predict_to,
                "pretrain_steps": pretrain_steps,
                "n_cycles": n_cycles, "wake_steps": wake_steps,
                "distill_steps": distill_steps, "retrain_steps": retrain_steps,
                "total_finetune_steps": total_finetune_steps,
                "checkpoint_steps": checkpoint_steps,
                "lr": lr, "fwd_lr": fwd_lr, "lambda_local": lambda_local,
                "temperature": temperature, "rl_loss_scale": rl_loss_scale,
                "lambda_ntp": lambda_ntp, "ntp_mask_rate": ntp_mask_rate,
                "batch_size": batch_size, "seed": seed,
                "only_rl_fm": True,
            },
            "pretrain": {
                "per_level": pretrain_pl,
                "generation_accuracy": pretrain_gen,
            },
            "conditions": all_results,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        volume.commit()

        # Print RL_FM summary
        final_cs = checkpoint_steps[-1]
        print(f"\n{'='*80}")
        print(f"SUMMARY: RHM RL Ratchet (RL_FM only, ntp_mask_rate={ntp_mask_rate})")
        print(f"{'='*80}")
        rl_fm_ckpts = all_results["RL_FM"]["checkpoints"]
        print(f"\n  === Val loss trajectory ===")
        for ci, cs in enumerate(checkpoint_steps):
            ckpt = rl_fm_ckpts.get(cs, {})
            print(f"  Cycle {ci+1}: val={ckpt.get('val_loss', 0):.4f}")
        ckpt = rl_fm_ckpts.get(final_cs, {})
        pl = ckpt.get("per_level", {}).get("per_level", {})
        print(f"\n  === Per-level loss (final) ===")
        print("  " + " ".join(f"L{l}={pl[l]['loss']:.3f}" for l in sorted(pl.keys())))
        ga = ckpt.get("generation_accuracy", {})
        print(f"\n  === Generation accuracy (final) ===")
        print(f"  overall={ga.get('overall',0):.3f} " +
              " ".join(f"L{k}={ga.get(f'L{k}',0):.3f}" for k in range(6)))
        gs = ckpt.get("gate_stats", {})
        if gs:
            print(f"\n  Gate: mean={gs['mean']:.3f} sparse(<0.1)={gs['sparse_01']:.3f}")
        sk = ckpt.get("self_knowledge", {})
        if sk:
            print(f"  SK: " + " | ".join(f"{k}={v_:.3f}" for k, v_ in sk.items()))
        ple = ckpt.get("per_layer_eta2", {})
        if ple:
            print(f"\n  === Per-layer feature eta² (final) ===")
            header = f"  {'Layer':>12s}" + "".join(f"  {'L'+str(l):>7s}" for l in range(L))
            print(header)
            for li in range(n_layer):
                lk = f"post_block{li}"
                ld = ple.get(lk, {})
                row = f"  {lk:>12s}"
                for l in range(L):
                    row += f"  {ld.get(f'level_{l}', {}).get('feature_eta2', 0):7.4f}"
                print(row)
        print(f"\n  Saved to {save_dir}/results.json")
        return result

    # ------------------------------------------------------------------
    # Condition 2: RL only (no FM, no ratchet)
    # ------------------------------------------------------------------
    cond = "RL"
    print(f"\n{'='*60}")
    print(f"  {cond}: RL only ({total_finetune_steps} steps)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    checkpoints = {}
    rl_step_idx = 0
    rl_baseline = None

    for step in range(total_finetune_steps):
        # --- Generate suffix (no FM) ---
        model.eval()
        full_seqs, prefix, true_suffix = make_rl_batch(rl_step_idx)

        with torch.no_grad():
            generated = _generate_suffix(
                model, prefix, suffix_len, temperature,
                None, None, cerebellar_input_block, inject_after_block)

        # --- Compute reward ---
        correct = (generated == true_suffix).float()
        reward = correct.mean(dim=1)
        if rl_baseline is None:
            rl_baseline = reward.mean().item()
        else:
            rl_baseline = 0.95 * rl_baseline + 0.05 * reward.mean().item()
        advantage = (reward - rl_baseline).detach()

        # --- Scoring pass (no FM) ---
        model.train()
        scored_seq = torch.cat([prefix, generated], dim=1)
        logits, _ = model(scored_seq)

        suffix_logits = logits[:, prefix_len - 1:prefix_len - 1 + suffix_len, :]
        log_probs = F.log_softmax(suffix_logits, dim=-1)
        gen_log_probs = log_probs.gather(
            2, generated.unsqueeze(-1)).squeeze(-1)
        rl_loss = -(advantage.unsqueeze(1) * gen_log_probs).mean()

        # Sparse NTP regularization on a separate batch
        ntp_step = pretrain_steps + step
        ntp_x = torch.stack([train_data[i:i + seq_len]
                             for i in train_indices[ntp_step]]).to(device)
        ntp_y = torch.stack([train_data[i + 1:i + seq_len + 1]
                             for i in train_indices[ntp_step]]).to(device)
        ntp_logits, _ = model(ntp_x)
        ntp_per_pos = F.cross_entropy(
            ntp_logits.view(-1, v), ntp_y.view(-1), reduction='none')
        if ntp_mask_rate > 0:
            ntp_mask = (torch.rand(ntp_per_pos.shape[0],
                                   device=device) >= ntp_mask_rate).float()
            ntp_loss = (ntp_per_pos * ntp_mask).sum() / ntp_mask.sum().clamp(min=1)
        else:
            ntp_loss = ntp_per_pos.mean()

        total_loss = rl_loss_scale * rl_loss + lambda_ntp * ntp_loss

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == total_finetune_steps - 1:
            vl = eval_model(model)
            print(f"  step {step:5d}: val={vl:.4f} reward={reward.mean():.3f} "
                  f"rl={rl_loss.item():.4f} ntp={ntp_loss.item():.4f}")

        rl_step_idx += 1

        if (step + 1) in checkpoint_steps:
            cycle_idx = checkpoint_steps.index(step + 1)
            print(f"\n  {cond} CHECKPOINT at step {step + 1}")
            ckpt = full_checkpoint_eval(model, cond, cycle_idx, is_rl=True)
            checkpoints[step + 1] = ckpt

    all_results[cond] = {"checkpoints": checkpoints}
    del model, opt
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Condition 3: NTP_FM (ratchet, dense NTP)
    # ------------------------------------------------------------------
    cond = "NTP_FM"
    print(f"\n{'='*60}")
    print(f"  {cond}: NTP + FM ratchet ({n_cycles} cycles)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    fm = make_fm()
    fm.load_state_dict(init_fm_state)
    ugate = UnifiedGate(n_embd, ug_hidden).to(device)

    checkpoints = {}
    ntp_step_idx = pretrain_steps

    for cycle in range(n_cycles):
        print(f"\n  --- {cond} Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: NTP with FM ===
        print(f"  [WAKE] {wake_steps} NTP steps")
        main_params = list(model.parameters()) + list(ugate.parameters())
        opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            model.train(); fm.train(); ugate.train()
            x = torch.stack([train_data[i:i + seq_len]
                             for i in train_indices[ntp_step_idx]]).to(device)
            y = torch.stack([train_data[i + 1:i + seq_len + 1]
                             for i in train_indices[ntp_step_idx]]).to(device)

            fwd_cache = {}

            def ug_cb(act, _cache=fwd_cache):
                fp = fm(act.detach())
                _cache["pred"] = fp
                inj, gw = ugate(act.detach(), fp.detach())
                _cache["gate_w"] = gw
                return inj

            logits, _, intermediates = model(
                x, return_intermediates=True,
                cerebellar_fn=ug_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

            ntp_loss = F.cross_entropy(logits.view(-1, v), y.view(-1))

            fwd_pred = fwd_cache["pred"]
            target_acts = intermediates[predict_to]
            gate_w = fwd_cache["gate_w"]
            gw_scalar = gate_w.detach().mean()
            r = target_acts - fwd_pred.detach()
            local_loss = (gw_scalar * r ** 2).mean()

            total_loss = ntp_loss + lambda_local * local_loss

            opt_main.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            fwd_loss = F.mse_loss(fwd_pred, target_acts.detach())
            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == wake_steps - 1:
                vl = eval_model(model)
                print(f"    step {ntp_step_idx:5d} (w{step}): "
                      f"val={vl:.4f} ntp={ntp_loss.item():.4f} "
                      f"ll={local_loss.item():.5f} gw={gw_scalar:.3f}")

            ntp_step_idx += 1

        # === SLEEP ===
        print(f"  [SLEEP] Distillation ({distill_steps} steps)")
        teacher = make_gpt()
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_fm = make_fm()
        teacher_fm.load_state_dict(fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters():
            p.requires_grad = False
        teacher_ugate = UnifiedGate(n_embd, ug_hidden).to(device)
        teacher_ugate.load_state_dict(ugate.state_dict())
        teacher_ugate.eval()
        for p in teacher_ugate.parameters():
            p.requires_grad = False

        student = make_gpt()
        student.load_state_dict(model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()
            x = torch.stack([train_data[i:i + seq_len]
                             for i in train_indices[ntp_step_idx]]).to(device)
            y = torch.stack([train_data[i + 1:i + seq_len + 1]
                             for i in train_indices[ntp_step_idx]]).to(device)

            with torch.no_grad():
                def t_cb(act):
                    fp = teacher_fm(act)
                    inj, _ = teacher_ugate(act, fp)
                    return inj
                teacher_logits, _, _ = teacher(
                    x, return_intermediates=True,
                    cerebellar_fn=t_cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )

            student_logits, _ = student(x)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            ce_loss = F.cross_entropy(student_logits.view(-1, v), y.view(-1))
            loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss

            opt_student.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_student.step()

            ntp_step_idx += 1

        model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_ugate, student, opt_student
        del opt_main, opt_fwd
        torch.cuda.empty_cache()

        # === REPOINT ===
        fm_seed = seed + 200 * (cycle + 1)
        print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
        fm, fm_cos = train_fresh_fm(
            model, retrain_steps, fm_seed, f"{cond} c{cycle+1}")

        # === CHECKPOINT ===
        print(f"\n  {cond} CHECKPOINT cycle {cycle + 1}")
        ckpt = full_checkpoint_eval(model, cond, cycle, fm=fm, ugate=ugate)
        ckpt["fm_cosine"] = fm_cos
        checkpoints[(cycle + 1) * steps_per_cycle] = ckpt

    all_results[cond] = {"checkpoints": checkpoints}
    del model, fm, ugate
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Condition 4: OL (continued NTP)
    # ------------------------------------------------------------------
    cond = "OL"
    print(f"\n{'='*60}")
    print(f"  {cond}: continued NTP ({total_finetune_steps} steps)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    checkpoints = {}
    ntp_step_idx = pretrain_steps

    for step in range(total_finetune_steps):
        model.train()
        x = torch.stack([train_data[i:i + seq_len]
                         for i in train_indices[ntp_step_idx]]).to(device)
        y = torch.stack([train_data[i + 1:i + seq_len + 1]
                         for i in train_indices[ntp_step_idx]]).to(device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == total_finetune_steps - 1:
            vl = eval_model(model)
            if step % 1000 == 0:
                print(f"  step {step:5d}: val={vl:.4f}")

        ntp_step_idx += 1

        if (step + 1) in checkpoint_steps:
            cycle_idx = checkpoint_steps.index(step + 1)
            print(f"\n  {cond} CHECKPOINT at step {step + 1}")

            # Train a fresh FM for evaluation measurements
            ol_fm, ol_fm_cos = train_fresh_fm(
                model, retrain_steps, seed + 700 + step, f"{cond} c{cycle_idx+1}")
            ckpt = full_checkpoint_eval(model, cond, cycle_idx)
            ckpt["fm_cosine"] = ol_fm_cos
            ckpt["residual_stats"] = compute_residual_stats(
                model, ol_fm, f"{cond} c{cycle_idx+1}")
            ckpt["self_knowledge"] = self_knowledge_probes(
                model, ol_fm, f"{cond} final") if cycle_idx == n_cycles - 1 else {}
            checkpoints[step + 1] = ckpt
            del ol_fm
            torch.cuda.empty_cache()

    all_results[cond] = {"checkpoints": checkpoints}
    del model, opt
    torch.cuda.empty_cache()

    # ==================================================================
    # Save results
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_rl_ratchet/{key}"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "prefix_len": prefix_len, "suffix_len": suffix_len,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D",
            "predict_from": predict_from, "predict_to": predict_to,
            "pretrain_steps": pretrain_steps,
            "n_cycles": n_cycles, "wake_steps": wake_steps,
            "distill_steps": distill_steps, "retrain_steps": retrain_steps,
            "total_finetune_steps": total_finetune_steps,
            "checkpoint_steps": checkpoint_steps,
            "lr": lr, "fwd_lr": fwd_lr, "lambda_local": lambda_local,
            "temperature": temperature, "rl_loss_scale": rl_loss_scale,
            "lambda_ntp": lambda_ntp, "ntp_mask_rate": ntp_mask_rate,
            "batch_size": batch_size, "seed": seed,
        },
        "pretrain": {
            "per_level": pretrain_pl,
            "generation_accuracy": pretrain_gen,
        },
        "conditions": all_results,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ==================================================================
    # Summary
    # ==================================================================
    uniform = float(np.log(v))
    print(f"\n{'='*80}")
    print(f"SUMMARY: RHM RL Ratchet")
    print(f"  Pre-train: {pretrain_steps} NTP steps")
    print(f"  Fine-tune: {n_cycles} cycles x ({wake_steps} wake + {distill_steps} sleep)"
          f" = {total_finetune_steps} steps")
    print(f"  Uniform baseline: {uniform:.4f}")
    print(f"{'='*80}")

    # Val loss trajectory
    print(f"\n  === Val loss at each checkpoint ===")
    print(f"  {'Cycle':>6s}  {'RL_FM':>8s}  {'RL':>8s}  {'NTP_FM':>8s}  {'OL':>8s}")
    for ci, cs in enumerate(checkpoint_steps):
        parts = [f"{ci+1:6d}"]
        for c in ["RL_FM", "RL", "NTP_FM", "OL"]:
            ckpt = all_results.get(c, {}).get("checkpoints", {}).get(cs, {})
            vl = ckpt.get("val_loss", 0)
            parts.append(f"{vl:8.4f}")
        print("  " + "  ".join(parts))

    # Per-level loss at final checkpoint
    final_cs = checkpoint_steps[-1]
    print(f"\n  === Per-level loss at final checkpoint ===")
    header = f"  {'Cond':>8s}  {'val':>7s}"
    for lvl in range(L):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    for c in ["RL_FM", "RL", "NTP_FM", "OL"]:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(final_cs, {})
        pl = ckpt.get("per_level", {}).get("per_level", {})
        vl = ckpt.get("val_loss", 0)
        row = f"  {c:>8s}  {vl:7.4f}"
        for lvl in range(L):
            if lvl in pl:
                row += f"  {pl[lvl]['loss']:7.4f}"
            else:
                row += f"  {'--':>7s}"
        print(row)

    # Generation accuracy at final checkpoint
    print(f"\n  === Generation accuracy at final checkpoint ===")
    header = f"  {'Cond':>8s}  {'overall':>7s}"
    for lvl in range(max_suffix_level + 1):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    for c in ["RL_FM", "RL", "NTP_FM", "OL"]:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(final_cs, {})
        ga = ckpt.get("generation_accuracy", {})
        row = f"  {c:>8s}  {ga.get('overall', 0):7.3f}"
        for lvl in range(max_suffix_level + 1):
            row += f"  {ga.get(f'L{lvl}', 0):7.3f}"
        print(row)

    # Per-layer feature eta² at final checkpoint (key new measurement)
    print(f"\n  === Per-layer feature eta² at final checkpoint ===")
    for c in ["RL_FM", "RL", "NTP_FM", "OL"]:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(final_cs, {})
        ple = ckpt.get("per_layer_eta2", {})
        if not ple:
            continue
        print(f"\n  {c}:")
        header = f"  {'Layer':>12s}"
        for lvl in range(L):
            header += f"  {'L'+str(lvl):>7s}"
        print(header)
        for li in range(n_layer):
            lk = f"post_block{li}"
            layer_data = ple.get(lk, {})
            row = f"  {lk:>12s}"
            for lvl in range(L):
                eta2 = layer_data.get(f"level_{lvl}", {}).get("feature_eta2", 0)
                row += f"  {eta2:7.4f}"
            print(row)

    # Self-knowledge and robustness
    print(f"\n  === Self-knowledge probes at final checkpoint ===")
    print(f"  {'Cond':>8s}  {'post_block0':>12s}  "
          f"{'post_block'+str(n_layer-1):>12s}")
    for c in ["RL_FM", "NTP_FM", "OL"]:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(final_cs, {})
        sk = ckpt.get("self_knowledge", {})
        if sk:
            b0 = sk.get("post_block0", 0)
            bN = sk.get(f"post_block{n_layer-1}", 0)
            print(f"  {c:>8s}  {b0:12.3f}  {bN:12.3f}")

    print(f"\n  === Robustness at final checkpoint (eps=1.0) ===")
    for c in ["RL_FM", "RL", "NTP_FM", "OL"]:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(final_cs, {})
        rob = ckpt.get("robustness", {}).get("1.0", None)
        if rob is not None:
            print(f"  {c:>8s}: {rob:+.4f}")

    # Gate dynamics
    print(f"\n  === Gate dynamics ===")
    for c in ["RL_FM", "NTP_FM"]:
        cond_data = all_results.get(c, {}).get("checkpoints", {})
        print(f"  {c}:")
        for ci, cs in enumerate(checkpoint_steps):
            gs = cond_data.get(cs, {}).get("gate_stats", {})
            if gs:
                print(f"    c{ci+1}: mean={gs['mean']:.3f} "
                      f"sparse(<0.1)={gs['sparse_01']:.3f}")

    print(f"\n  Saved to {save_dir}/results.json")
    return result


@app.local_entrypoint()
def main(
    pretrain_steps: int = 10000,
    n_cycles: int = 4,
    wake_steps: int = 2000,
    distill_steps: int = 500,
    m: int = 2,
    depth: int = 6,
    seed: int = 42,
):
    result = rhm_rl_ratchet.remote(
        pretrain_steps=pretrain_steps,
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        m=m,
        depth=depth,
        seed=seed,
    )
    print("\nRHM RL ratchet complete.")

    config = result["config"]
    print(f"\n  Pre-train: {config['pretrain_steps']} NTP steps")
    print(f"  Fine-tune: {config['n_cycles']} cycles x "
          f"({config['wake_steps']} wake + {config['distill_steps']} sleep) = "
          f"{config['total_finetune_steps']} steps")

    final_step = config["checkpoint_steps"][-1]
    print(f"\n  Final ({final_step} steps):")

    for cond_key, cond_data in result["conditions"].items():
        ckpts = cond_data.get("checkpoints", {})
        c = ckpts.get(final_step) or ckpts.get(str(final_step))
        if c:
            rob = c.get("robustness", {}).get("1.0", None)
            rob_str = f" rob={rob:+.4f}" if rob is not None else ""
            ga = c.get("generation_accuracy", {}).get("overall", None)
            ga_str = f" gen={ga:.3f}" if ga is not None else ""
            print(f"    {cond_key:>8s}: val={c['val_loss']:.4f}{rob_str}{ga_str}")
