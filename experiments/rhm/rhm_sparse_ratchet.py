"""RHM sparse ratchet: wake-sleep with NTP masking.

Tests whether the WS_UG_uniform ratchet produces per-level compositional
improvement when NTP supervision is sparse. The sparsity sweep showed that
FM local loss helps val loss and per-level learning at mask_rate >= 0.90.
This experiment tests whether ratchet cycles (injection + distillation +
FM reinit) compound that improvement across cycles.

Conditions: {WS_UG_uniform, OL} x {mask_rate = 0.75, 0.90, 0.95}

WS_UG_uniform per cycle:
  WAKE (3000 steps): masked NTP + injection via unified gate + gate-scaled
    local loss + FM co-training
  SLEEP (750 steps): distillation — KL(student || teacher_with_injection)
    + masked CE. KL is unmasked (no ground truth needed); the ratchet's
    advantage comes from teacher knowledge at all positions.
  REPOINT (2000 steps): fresh FM trained on consolidated model

OL: masked NTP for 15K steps (same total main-model budget).

Model: 6L/6H/192D (~2.7M params)
FM: 1L/1H/16D causal (50.4K params, 1.9%), post_block0 -> post_block5,
    inject after block 1
DGP: L=6, m=2, v=8, s=2 (seq_len=64)

fresh_data mode: each wake cycle generates a fresh corpus from the same
  composition rules (same DGP, new sequences). OL also cycles through the
  same fresh data on the same schedule. Tests whether data novelty enables
  compounding compositional improvement across ratchet cycles.

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_sparse_ratchet::rhm_sparse_ratchet
  # Fresh data variant:
  modal run --detach -m rhm.rhm_sparse_ratchet::rhm_sparse_ratchet --fresh-data
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

app = modal.App("rhm-sparse-ratchet", image=image)


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
    """Per-level CE and accuracy using hierarchy-aligned sequences."""
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
            target_log_probs = log_probs.gather(
                2, targets.unsqueeze(-1)).squeeze(-1)
            per_pos_loss_sum += (-target_log_probs).sum(dim=0).cpu().numpy()
            per_pos_correct_sum += (
                pred_logits.argmax(dim=-1) == targets
            ).float().sum(dim=0).cpu().numpy()
            n_samples += bs

    per_pos_loss = per_pos_loss_sum / n_samples
    per_pos_acc = per_pos_correct_sum / n_samples

    level_results = {}
    for lvl in range(max_level + 1):
        mask = levels == lvl
        if mask.sum() == 0:
            continue
        level_results[lvl] = {
            "loss": float(per_pos_loss[mask].mean()),
            "accuracy": float(per_pos_acc[mask].mean()),
        }
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


def _compute_hierarchy_eta2(residuals_np, level_features, level_rules, s, L):
    import numpy as np
    n_seq, seq_len, d_model = residuals_np.shape
    res_last = residuals_np[:, -1, :]
    last_mean = res_last.mean(axis=0)
    ss_total = float(np.sum((res_last - last_mean) ** 2))
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
                res_last, level_features[ell][:, last_ancestor], last_mean
            ) / ss_total,
            "rule_eta2": eta2_between(
                res_last, level_rules[ell][:, last_ancestor], last_mean
            ) / ss_total,
        }
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
def rhm_sparse_ratchet(
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
    # Ratchet
    n_cycles: int = 4,
    wake_steps: int = 3000,
    distill_steps: int = 750,
    retrain_steps: int = 2000,
    # Training
    lr: float = 3e-4, fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4, distill_alpha: float = 0.5,
    lambda_local: float = 1.0, ug_hidden: int = 64,
    batch_size: int = 64,
    fixed_local_loss: bool = False,
    skip_ol: bool = False,
    fresh_data: bool = False,
    # Masking sweep
    mask_rates: str = "0.75,0.90,0.95",
    # Eval
    eval_interval: int = 300,
    n_eval_batches: int = 5,
    probe_batches: int = 30,
    probe_steps: int = 500,
    n_eval_sequences: int = 5000,
    seed: int = 42,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    cerebellar_input_block = int(predict_from.replace("post_block", ""))
    mr_list = [float(x) for x in mask_rates.split(",")]

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    print(f"RHM SPARSE RATCHET on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D, "
          f"{predict_from}->{predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep = "
          f"{total_main_steps} total main steps")
    print(f"  Mask rates: {mr_list}")
    print(f"  lambda_local={lambda_local}, fixed_local_loss={fixed_local_loss}")
    print(f"  skip_ol={skip_ol}, fresh_data={fresh_data}")
    ll_tag = "fixed" if fixed_local_loss else "gated"

    # ------------------------------------------------------------------
    # UnifiedGate
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

        def injection_norm(self):
            return self.projection.weight.norm().item()

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

    # ------------------------------------------------------------------
    # Fresh data: generate per-cycle training corpora from the same rules
    # ------------------------------------------------------------------
    if fresh_data:
        from rhm.rhm_data import generate_sequences_batched, sequences_to_corpus

        tokens_per_cycle = steps_per_cycle * batch_size * (seq_len + 1) * 2
        n_seqs_per_cycle = tokens_per_cycle // seq_len + 1000

        print(f"\n  Generating fresh data for {n_cycles} cycles "
              f"({n_seqs_per_cycle} seqs/cycle, "
              f"{n_seqs_per_cycle * seq_len / 1e6:.1f}M tokens/cycle)...")

        cycle_train_datas = []
        for c in range(n_cycles):
            seqs = generate_sequences_batched(
                rules, n_seqs_per_cycle, seed=seed + 2000 + c)
            flat = torch.from_numpy(
                sequences_to_corpus(seqs).astype(np.int64))
            cycle_train_datas.append(flat)
            print(f"    Cycle {c+1}: {len(flat):,} tokens")

    # ------------------------------------------------------------------
    # Pre-generate batch indices and NTP masks (shared across conditions)
    # ------------------------------------------------------------------
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    mask_rng = torch.Generator().manual_seed(seed + 4)

    if fresh_data:
        cycle_train_indices = {}
        cycle_repoint_indices = {}
        for c in range(n_cycles):
            tgen = torch.Generator().manual_seed(seed + 3000 + c)
            n = len(cycle_train_datas[c]) - seq_len - 1
            cycle_train_indices[c] = [
                torch.randint(n, (batch_size,), generator=tgen)
                for _ in range(steps_per_cycle)
            ]
            rgen = torch.Generator().manual_seed(seed + 4000 + c)
            cycle_repoint_indices[c] = [
                torch.randint(n, (batch_size,), generator=rgen)
                for _ in range(retrain_steps)
            ]
    else:
        train_gen = torch.Generator().manual_seed(seed)
        repoint_gen = torch.Generator().manual_seed(seed + 3)
        train_indices = [
            torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                           generator=train_gen)
            for _ in range(total_main_steps)
        ]
        max_repoint_steps = n_cycles * retrain_steps
        repoint_indices = [
            torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                           generator=repoint_gen)
            for _ in range(max_repoint_steps)
        ]

    n_evals = total_main_steps // eval_interval + 10
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

    # Pre-generate mask randoms so OL and WS see identical masks
    mask_randoms = torch.rand(
        total_main_steps, batch_size * seq_len, generator=mask_rng)

    def make_batch(indices, split_data):
        x = torch.stack([split_data[i:i + seq_len] for i in indices])
        y = torch.stack([split_data[i + 1:i + seq_len + 1] for i in indices])
        return x.to(device), y.to(device)

    def masked_ntp_loss(logits, targets, step_idx, mask_rate):
        """Compute masked NTP loss for a given step."""
        per_pos = F.cross_entropy(
            logits.view(-1, v), targets.view(-1), reduction='none')
        if mask_rate > 0:
            mask = (mask_randoms[step_idx] >= mask_rate).float().to(device)
            n_active = mask.sum().clamp(min=1.0)
            return (per_pos * mask).sum() / n_active
        return per_pos.mean()

    def get_train_batch(global_step):
        """Get training batch, abstracting over fixed vs fresh data."""
        if fresh_data:
            c = global_step // steps_per_cycle
            ws = global_step % steps_per_cycle
            return make_batch(cycle_train_indices[c][ws],
                              cycle_train_datas[c])
        else:
            return make_batch(train_indices[global_step], train_data)

    def get_repoint_batch(cycle_idx, step):
        """Get FM retraining batch from the appropriate cycle's data."""
        if fresh_data:
            idx = cycle_repoint_indices[cycle_idx][step % retrain_steps]
            return make_batch(idx, cycle_train_datas[cycle_idx])
        else:
            idx = repoint_indices[step % len(repoint_indices)]
            return make_batch(idx, train_data)

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
    # Save initial weights (shared across all conditions)
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
    print("Saved initial weight states")

    # ==================================================================
    # Shared evaluation helpers
    # ==================================================================

    def eval_model(model, eval_idx=0):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                x, y = make_batch(eval_indices[eval_idx][bi], val_data)
                _, loss = model(x, y)
                total_loss += loss.item()
        return total_loss / n_eval_batches

    def eval_model_with_ug(model, fm, ugate, eval_idx=0):
        model.eval(); fm.eval(); ugate.eval()
        total_loss = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                x, y = make_batch(eval_indices[eval_idx][bi], val_data)

                def cb(act):
                    fp = fm(act)
                    inj, _ = ugate(act, fp)
                    return inj

                _, loss, _ = model(
                    x, y, return_intermediates=True,
                    cerebellar_fn=cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
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
                    x, y = make_batch(probe_indices[bi], val_data)
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
        print(f"  [{label:>25s}] rob: "
              + " | ".join(f"eps={e}:{results[str(e)]:+.4f}" for e in eps_list))
        return results

    def self_knowledge_probes(model, fm, label):
        model.eval(); fm.eval()
        probe_layers = ["post_block0", f"post_block{n_layer - 1}"]
        acts_by_layer = {k: [] for k in probe_layers}
        residuals = []
        with torch.no_grad():
            for pidx in probe_indices:
                x, y = make_batch(pidx, val_data)
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
        top1_var = float((S[0] ** 2) / (S ** 2).sum())

        n_complete = res_np.shape[0]
        eta2 = _compute_hierarchy_eta2(
            res_np,
            [lf[:n_complete] for lf in level_features],
            [lr[:n_complete] for lr in level_rules],
            s=s, L=L,
        )

        stats = {
            "fwd_cosine": mean_cos, "res_norm": mean_norm,
            "eff_rank": eff_rank, "eff_rank_pct": eff_rank / n_embd * 100,
            "top1_pct": top1_var * 100, "hierarchy_eta2": eta2,
        }
        fL4 = eta2.get("level_4", {}).get("feature_eta2", 0)
        print(f"  [{label}] cos={mean_cos:.4f} norm={mean_norm:.2f} "
              f"rank={eff_rank:.1f}/{n_embd} fL4*={fL4:.4f}")
        return stats

    def train_fresh_fm(model, steps, fm_seed, label, cycle_idx=0):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        for step in range(steps):
            fm.train()
            x, _ = get_repoint_batch(cycle_idx, step)
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
                x, y = make_batch(eval_indices[0][bi], val_data)
                _, _, vi = model(x, y, return_intermediates=True)
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
                x, y = make_batch(probe_indices[bi], val_data)
                _, _, vi = model(x, y, return_intermediates=True)
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
            "sparse_02": float((dim_mean < 0.2).float().mean()),
            "dim_mean": dim_mean.tolist(),
        }

    # ==================================================================
    # Run all conditions
    # ==================================================================
    all_results = {}

    for mask_rate in mr_list:
        mr_label = f"m{mask_rate:.2f}"

        # ==============================================================
        # WS_UG_uniform at this mask rate
        # ==============================================================
        ws_label = f"WS_{mr_label}"
        print(f"\n{'='*60}")
        print(f"  {ws_label} ({n_cycles} cycles, mask_rate={mask_rate})")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_model_state)
        fm = make_fm()
        fm.load_state_dict(init_fm_state)
        ugate = UnifiedGate(n_embd, ug_hidden).to(device)

        ws_checkpoints = {}
        ws_history = {"val_loss": [], "gate_stats": []}
        ws_global_step = 0

        for cycle in range(n_cycles):
            print(f"\n  --- {ws_label} Cycle {cycle + 1}/{n_cycles} ---")

            # === WAKE ===
            print(f"  [WAKE] {wake_steps} steps, mask_rate={mask_rate}")
            ug_main_params = (list(model.parameters())
                              + list(ugate.parameters()))
            opt_main = torch.optim.AdamW(
                ug_main_params, lr=lr, weight_decay=0.01)
            opt_fwd = torch.optim.AdamW(
                fm.parameters(), lr=fwd_lr, weight_decay=0.01)

            for step in range(wake_steps):
                model.train(); fm.train(); ugate.train()
                x, y = get_train_batch(ws_global_step)

                fwd_pred_cache = {}

                def ug_cb(act, _cache=fwd_pred_cache):
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

                ntp_loss = masked_ntp_loss(logits, y, ws_global_step, mask_rate)

                fwd_pred = fwd_pred_cache["pred"]
                target_acts = intermediates[predict_to]
                gate_w = fwd_pred_cache["gate_w"]
                r = target_acts - fwd_pred.detach()
                gw_scalar = gate_w.detach().mean()
                if fixed_local_loss:
                    local_loss = (r ** 2).mean()
                else:
                    local_loss = (gw_scalar * r ** 2).mean()

                L_total = ntp_loss + lambda_local * local_loss

                opt_main.zero_grad()
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(ug_main_params, 1.0)
                opt_main.step()

                fwd_loss = F.mse_loss(fwd_pred, target_acts.detach())
                opt_fwd.zero_grad()
                fwd_loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt_fwd.step()

                if step % eval_interval == 0 or step == wake_steps - 1:
                    vl = eval_model(model)
                    ws_history["val_loss"].append((ws_global_step, vl))
                    print(f"    step {ws_global_step:5d} (w{step}): "
                          f"val={vl:.4f} ntp={ntp_loss.item():.4f} "
                          f"ll={local_loss.item():.5f} "
                          f"gw={gw_scalar:.3f}")

                ws_global_step += 1

            gs = get_ug_gate_stats(ugate, model, fm)
            ws_history["gate_stats"].append(
                (ws_global_step, f"post_wake_c{cycle+1}", gs))
            print(f"    Gate: mean={gs['mean']:.3f} "
                  f"sparse(<0.1)={gs['sparse_01']:.3f}")

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

            for step in range(distill_steps):
                student.train()
                x, y = get_train_batch(ws_global_step)

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
                ce_loss = masked_ntp_loss(
                    student_logits, y, ws_global_step, mask_rate)

                loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss

                opt_student.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_student.step()

                if step % eval_interval == 0 or step == distill_steps - 1:
                    vl = eval_model(student)
                    ws_history["val_loss"].append((ws_global_step, vl))
                    print(f"    step {ws_global_step:5d} (s{step}): "
                          f"val={vl:.4f}")

                ws_global_step += 1

            model.load_state_dict(student.state_dict())
            del teacher, teacher_fm, teacher_ugate, student, opt_student
            del opt_main, opt_fwd
            torch.cuda.empty_cache()

            # === RE-POINT: Fresh FM ===
            fm_seed = seed + 100 * (cycle + 1)
            print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
            fm, fm_cos = train_fresh_fm(
                model, retrain_steps, fm_seed, f"{ws_label} c{cycle+1}",
                cycle_idx=cycle)

            gs_post = get_ug_gate_stats(ugate, model, fm)
            ws_history["gate_stats"].append(
                (ws_global_step, f"post_repoint_c{cycle+1}", gs_post))

            # === CHECKPOINT EVALUATION ===
            cycle_step = (cycle + 1) * steps_per_cycle
            print(f"\n  {ws_label} CHECKPOINT at step {cycle_step}")

            ws_vl = eval_model(model)
            ws_inj = eval_model_with_ug(model, fm, ugate)
            ws_dep = ws_vl - ws_inj
            ws_pl = _eval_per_level(
                model, eval_seqs, seq_len, s, L, v, batch_size, device)
            ws_res = compute_residual_stats(model, fm, f"{ws_label} c{cycle+1}")

            ckpt = {
                "val_loss": ws_vl, "val_loss_inj": ws_inj,
                "dep_gap": ws_dep, "fm_cosine": fm_cos,
                "per_level": ws_pl,
                "gate_post_wake": ws_history["gate_stats"][-2][2],
                "gate_post_repoint": gs_post,
                "residual_stats": ws_res,
            }

            is_final = (cycle == n_cycles - 1)
            if is_final:
                ckpt["robustness"] = measure_robustness(
                    model, f"{ws_label} final")
                ckpt["self_knowledge"] = self_knowledge_probes(
                    model, fm, f"{ws_label} final")

            ws_checkpoints[cycle_step] = ckpt

            level_str = " ".join(
                f"L{lvl}={ws_pl['per_level'][lvl]['loss']:.3f}"
                for lvl in sorted(ws_pl['per_level'].keys()))
            print(f"    val={ws_vl:.4f} dep={ws_dep:+.4f} {level_str}")

        del model, fm, ugate
        torch.cuda.empty_cache()

        all_results[ws_label] = {
            "checkpoints": ws_checkpoints,
            "history": ws_history,
        }

        # ==============================================================
        # OL at this mask rate
        # ==============================================================
        if skip_ol:
            print(f"\n  [SKIP] OL_{mr_label} (skip_ol=True)")
            continue

        ol_label = f"OL_{mr_label}"
        print(f"\n{'='*60}")
        print(f"  {ol_label} ({total_main_steps} steps, mask_rate={mask_rate})")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_model_state)
        ol_opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01)

        ol_checkpoints = {}
        ol_history = {"val_loss": []}

        for step in range(total_main_steps):
            model.train()
            x, y = get_train_batch(step)

            logits, _ = model(x)
            ntp_loss = masked_ntp_loss(logits, y, step, mask_rate)

            ol_opt.zero_grad()
            ntp_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ol_opt.step()

            if step % eval_interval == 0 or step == total_main_steps - 1:
                vl = eval_model(model)
                ol_history["val_loss"].append((step, vl))
                if step % 1000 == 0:
                    print(f"  step {step:5d}: val={vl:.4f}")

            if (step + 1) in checkpoint_steps:
                ckpt_step = step + 1
                print(f"\n  {ol_label} CHECKPOINT at step {ckpt_step}")

                ol_vl = eval_model(model)
                ol_pl = _eval_per_level(
                    model, eval_seqs, seq_len, s, L, v, batch_size, device)

                ckpt = {"val_loss": ol_vl, "per_level": ol_pl}

                is_final = (ckpt_step == checkpoint_steps[-1])
                if is_final:
                    ol_fm, ol_fm_cos = train_fresh_fm(
                        model, retrain_steps,
                        seed + 700 + ckpt_step, f"{ol_label} final",
                        cycle_idx=n_cycles - 1)
                    ckpt["fm_cosine"] = ol_fm_cos
                    ckpt["residual_stats"] = compute_residual_stats(
                        model, ol_fm, f"{ol_label} final")
                    ckpt["robustness"] = measure_robustness(
                        model, f"{ol_label} final")
                    ckpt["self_knowledge"] = self_knowledge_probes(
                        model, ol_fm, f"{ol_label} final")
                    del ol_fm
                    torch.cuda.empty_cache()

                ol_checkpoints[ckpt_step] = ckpt

                level_str = " ".join(
                    f"L{lvl}={ol_pl['per_level'][lvl]['loss']:.3f}"
                    for lvl in sorted(ol_pl['per_level'].keys()))
                print(f"    val={ol_vl:.4f} {level_str}")

        del model, ol_opt
        torch.cuda.empty_cache()

        all_results[ol_label] = {
            "checkpoints": ol_checkpoints,
            "history": ol_history,
        }

    # ==================================================================
    # Save results
    # ==================================================================
    save_tag = f"{ll_tag}_fresh" if fresh_data else ll_tag
    save_dir = f"{DATA_DIR}/rhm_sparse_ratchet/{key}_{save_tag}"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D",
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "n_cycles": n_cycles, "wake_steps": wake_steps,
            "distill_steps": distill_steps, "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "checkpoint_steps": checkpoint_steps,
            "mask_rates": mr_list,
            "lr": lr, "fwd_lr": fwd_lr, "lambda_local": lambda_local,
            "distill_lr": distill_lr, "distill_alpha": distill_alpha,
            "ug_hidden": ug_hidden, "batch_size": batch_size,
            "fixed_local_loss": fixed_local_loss, "ll_tag": ll_tag,
            "fresh_data": fresh_data,
            "n_tokens": n_tokens, "seed": seed,
        },
        "conditions": all_results,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ==================================================================
    # Summary tables
    # ==================================================================
    uniform = float(np.log(v))
    print(f"\n{'='*80}")
    print(f"SUMMARY: RHM Sparse Ratchet ({ll_tag} local loss)")
    print(f"  {n_cycles} cycles x ({wake_steps} wake + {distill_steps} sleep)"
          f" = {total_main_steps} steps")
    print(f"  Uniform baseline: {uniform:.4f}")
    print(f"{'='*80}")

    # Val loss trajectory
    print(f"\n  === Val loss trajectory ===")
    header_parts = [f"{'Cycle':>6s}"]
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        header_parts.extend([f"WS_{ml:>5s}", f"OL_{ml:>5s}", f"{'gap':>6s}"])
    print("  " + " | ".join(header_parts))

    for ci, cs in enumerate(checkpoint_steps):
        parts = [f"{ci+1:6d}"]
        for mr in mr_list:
            ml = f"m{mr:.2f}"
            ws_key = f"WS_{ml}"
            ol_key = f"OL_{ml}"
            ws_c = all_results.get(ws_key, {}).get("checkpoints", {}).get(cs, {})
            ol_c = all_results.get(ol_key, {}).get("checkpoints", {}).get(cs, {})
            ws_v = ws_c.get("val_loss", 0)
            ol_v = ol_c.get("val_loss", 0)
            gap = ol_v - ws_v
            parts.extend([f"{ws_v:7.4f}", f"{ol_v:7.4f}", f"{gap:+6.3f}"])
        print("  " + " | ".join(parts))

    # Gate dynamics
    print(f"\n  === Gate dynamics (WS, post-wake) ===")
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        ws_key = f"WS_{ml}"
        print(f"  {ws_key}:")
        for ci in range(n_cycles):
            gs_list = all_results[ws_key]["history"]["gate_stats"]
            wake_gs = [g for g in gs_list if f"post_wake_c{ci+1}" in g[1]]
            if wake_gs:
                gs = wake_gs[0][2]
                print(f"    c{ci+1}: mean={gs['mean']:.3f} "
                      f"sparse(<0.1)={gs['sparse_01']:.3f}")

    # Per-level loss at each cycle
    print(f"\n  === Per-level loss at each cycle (standard CE eval) ===")
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        ws_key = f"WS_{ml}"
        ol_key = f"OL_{ml}"
        print(f"\n  mask_rate={mr}:")

        max_lvl = 0
        for cs in checkpoint_steps:
            for k in [ws_key, ol_key]:
                ckpt = all_results.get(k, {}).get("checkpoints", {}).get(cs, {})
                for lvl in ckpt.get("per_level", {}).get("per_level", {}).keys():
                    max_lvl = max(max_lvl, int(lvl))

        header = f"  {'Cond':>10s} {'Cyc':>3s}  {'val':>7s}"
        for lvl in range(max_lvl + 1):
            header += f"  {'L'+str(lvl):>7s}"
        print(header)
        print("  " + "-" * len(header))

        for ci, cs in enumerate(checkpoint_steps):
            for prefix, cond_key in [("WS", ws_key), ("OL", ol_key)]:
                if cond_key not in all_results:
                    continue
                ckpt = all_results[cond_key]["checkpoints"].get(cs, {})
                pl = ckpt.get("per_level", {}).get("per_level", {})
                vl = ckpt.get("val_loss", 0)
                row = f"  {prefix:>10s} {ci+1:>3d}  {vl:7.4f}"
                for lvl in range(max_lvl + 1):
                    if lvl in pl:
                        row += f"  {pl[lvl]['loss']:7.4f}"
                    else:
                        row += f"  {'--':>7s}"
                print(row)

    # Per-level delta (WS - OL) at each cycle
    print(f"\n  === Per-level delta (WS - OL; negative = WS better) ===")
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        ws_key = f"WS_{ml}"
        ol_key = f"OL_{ml}"
        print(f"\n  mask_rate={mr}:")

        header = f"  {'Cyc':>3s}  {'val':>7s}"
        for lvl in range(6):
            header += f"  {'L'+str(lvl):>7s}"
        print(header)

        for ci, cs in enumerate(checkpoint_steps):
            ws_ckpt = all_results.get(ws_key, {}).get("checkpoints", {}).get(cs, {})
            ol_ckpt = all_results.get(ol_key, {}).get("checkpoints", {}).get(cs, {})
            ws_pl = ws_ckpt.get("per_level", {}).get("per_level", {})
            ol_pl = ol_ckpt.get("per_level", {}).get("per_level", {})
            ws_vl = ws_ckpt.get("val_loss", 0)
            ol_vl = ol_ckpt.get("val_loss", 0)
            row = f"  {ci+1:>3d}  {ws_vl - ol_vl:+7.4f}"
            for lvl in range(6):
                if lvl in ws_pl and lvl in ol_pl:
                    delta = ws_pl[lvl]["loss"] - ol_pl[lvl]["loss"]
                    row += f"  {delta:+7.4f}"
                else:
                    row += f"  {'--':>7s}"
            print(row)

    # FM residual stats at final checkpoint
    final_cs = checkpoint_steps[-1]
    print(f"\n  === FM residual stats at final checkpoint ===")
    print(f"  {'Condition':>12s}  {'cos':>7s}  {'norm':>7s}  "
          f"{'rank%':>6s}  {'top1%':>6s}  {'fL4*':>7s}")
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        for prefix in [f"WS_{ml}", f"OL_{ml}"]:
            if prefix not in all_results:
                continue
            ckpt = all_results[prefix]["checkpoints"].get(final_cs, {})
            rs = ckpt.get("residual_stats", {})
            if rs:
                fL4 = rs.get("hierarchy_eta2", {}).get(
                    "level_4", {}).get("feature_eta2", 0)
                print(f"  {prefix:>12s}  {rs['fwd_cosine']:7.4f}  "
                      f"{rs['res_norm']:7.2f}  {rs['eff_rank_pct']:6.1f}  "
                      f"{rs['top1_pct']:6.1f}  {fL4:7.4f}")
            else:
                print(f"  {prefix:>12s}  (no FM stats)")

    # Robustness
    print(f"\n  === Robustness at final checkpoint (eps=1.0) ===")
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        for prefix in [f"WS_{ml}", f"OL_{ml}"]:
            if prefix not in all_results:
                continue
            ckpt = all_results[prefix]["checkpoints"].get(final_cs, {})
            rob = ckpt.get("robustness", {}).get("1.0", None)
            if rob is not None:
                print(f"  {prefix:>12s}: {rob:+.4f}")

    # Self-knowledge
    print(f"\n  === Self-knowledge probes at final checkpoint ===")
    print(f"  {'Condition':>12s}  {'post_block0':>12s}  "
          f"{'post_block'+str(n_layer-1):>12s}")
    for mr in mr_list:
        ml = f"m{mr:.2f}"
        for prefix in [f"WS_{ml}", f"OL_{ml}"]:
            if prefix not in all_results:
                continue
            ckpt = all_results[prefix]["checkpoints"].get(final_cs, {})
            sk = ckpt.get("self_knowledge", {})
            b0 = sk.get("post_block0", 0)
            bN = sk.get(f"post_block{n_layer-1}", 0)
            print(f"  {prefix:>12s}  {b0:12.3f}  {bN:12.3f}")

    print(f"\n  Saved to {save_dir}/results.json")
    return result


@app.local_entrypoint()
def main(
    mask_rates: str = "0.75,0.90,0.95",
    n_cycles: int = 4,
    wake_steps: int = 3000,
    distill_steps: int = 750,
    lambda_local: float = 1.0,
    fixed_local_loss: bool = False,
    skip_ol: bool = False,
    fresh_data: bool = False,
    m: int = 2,
    depth: int = 6,
    seed: int = 42,
):
    result = rhm_sparse_ratchet.remote(
        mask_rates=mask_rates,
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        lambda_local=lambda_local,
        fixed_local_loss=fixed_local_loss,
        skip_ol=skip_ol,
        fresh_data=fresh_data,
        m=m,
        depth=depth,
        seed=seed,
    )
    print("\nRHM sparse ratchet complete.")

    config = result["config"]
    print(f"\n  {config['n_cycles']} cycles x "
          f"({config['wake_steps']} wake + {config['distill_steps']} sleep) = "
          f"{config['total_main_steps']} main-model steps")
    print(f"  Mask rates: {config['mask_rates']}")

    final_step = config["checkpoint_steps"][-1]
    print(f"\n  Final ({final_step} steps):")

    def get_ckpt(ckpts, step):
        return ckpts.get(step) or ckpts.get(str(step))

    for cond_key, cond_data in result["conditions"].items():
        c = get_ckpt(cond_data["checkpoints"], final_step)
        if c:
            rob = c.get("robustness", {}).get("1.0", None)
            rob_str = f" rob={rob:+.4f}" if rob is not None else ""
            print(f"    {cond_key:>12s}: val={c['val_loss']:.4f}{rob_str}")
