"""RHM RL Gen-Distill Extended: 40-Cycle Ratchet for Grokking Test.

Extends the gen-distill experiment (run 13) to 40 cycles to test the
self-model-driven grokking hypothesis.

Hypothesis: The gen-distill ratchet showed monotonically increasing L3 feature
eta^2 (+15% over 4 cycles, 0.389->0.449) with flat outcome metrics. The RHM's
compact DGP (96 rules = 576 bits for L=6/m=2) and the model's ~150,000x
overparameterization place it in the grokking regime. The FM's DGP-aligned
structural regularization (91-97% of composition rules at learned levels) could
produce a phase transition: the model "clicks" on L3 composition rules, making
L4 accessible and potentially triggering a cascade.

Signals to watch:
  1. Per-level NTP loss at L3 (sudden drop = grokking)
  2. Per-layer feature eta^2 at L3/L4 final layer (precursor trends)
  3. Per-level generation accuracy at L3+ (behavioral grokking)
  4. Activation norm at predict_to layer (stability -- halt if blows up)

weight_decay_mode controls where strong weight decay is applied:
  - "baseline": standard WD (0.01) everywhere (the original behavior)
  - "sleep": strong WD during distillation, standard during wake
  - "wake": strong WD during wake, standard during distillation

Based on run 13 from rhm_rl_gen_distill.py. Only runs gen_distill condition.
Includes a norm failsafe: if activation norms exceed norm_halt_ratio x the
post-pretrain baseline, saves results and halts early.

Reproduction:
  cd experiments/
  modal run --detach -m rhm.ratchet.rhm_rl_gen_distill_extended::rhm_rl_gen_distill_extended
  modal run --detach -m rhm.ratchet.rhm_rl_gen_distill_extended::rhm_rl_gen_distill_extended --weight-decay-mode sleep
  modal run --detach -m rhm.ratchet.rhm_rl_gen_distill_extended::rhm_rl_gen_distill_extended --weight-decay-mode wake
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key
from rhm.ratchet.rhm_rl_ratchet import (
    _position_levels,
    _suffix_position_levels,
    _eval_per_level,
    _ensure_corpus,
    _generate_with_traces,
    _compute_hierarchy_eta2,
    _generate_suffix,
    _compute_per_layer_eta2,
)
from rhm.ratchet.rhm_rl_gen_distill import _generate_suffix_with_logits

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-rl-gen-distill-extended", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=86400,
    memory=32768,
)
def rhm_rl_gen_distill_extended(
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
    n_cycles: int = 40,
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
    # Eval / save
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    probe_batches: int = 30,
    probe_steps: int = 500,
    n_eval_sequences: int = 5000,
    n_gen_eval_seqs: int = 1000,
    save_interval: int = 5,
    milestone_interval: int = 10,
    # Weight decay
    weight_decay_mode: str = "baseline",  # "baseline", "sleep", or "wake"
    strong_weight_decay: float = 0.1,
    # Norm failsafe
    norm_halt_ratio: float = 5.0,
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

    suffix_levels = _suffix_position_levels(prefix_len, seq_len, s)
    max_suffix_level = max(suffix_levels)

    assert weight_decay_mode in ("baseline", "sleep", "wake"), \
        f"Unknown weight_decay_mode: {weight_decay_mode}"
    base_wd = 0.01
    wake_wd = strong_weight_decay if weight_decay_mode == "wake" else base_wd
    sleep_wd = strong_weight_decay if weight_decay_mode == "sleep" else base_wd

    if weight_decay_mode == "baseline":
        save_dir = f"{DATA_DIR}/rhm_rl_gen_distill_extended/{key}"
    else:
        save_dir = (f"{DATA_DIR}/rhm_rl_gen_distill_extended/"
                    f"{key}_wd_{weight_decay_mode}_{strong_weight_decay}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"RHM RL GEN-DISTILL EXTENDED on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D, "
          f"{predict_from}->{predict_to}")
    print(f"  Weight decay mode: {weight_decay_mode} "
          f"(wake={wake_wd}, sleep={sleep_wd})")
    print(f"  Pre-train: {pretrain_steps} NTP steps")
    print(f"  Fine-tune: {n_cycles} cycles x ({wake_steps} wake + "
          f"{distill_steps} sleep) = {total_finetune_steps} steps")
    print(f"  Norm halt ratio: {norm_halt_ratio}x post-pretrain baseline")
    print(f"  Save every {save_interval} cycles, "
          f"milestones every {milestone_interval} cycles")

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
    print(f"  {len(data):,} tokens, train={len(train_data):,}, "
          f"val={len(val_data):,}")

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(
        rules, n_eval_sequences, seed=12345)
    print(f"  {n_eval_sequences} eval sequences with hierarchy traces")

    n_rl_seqs = 100000
    print(f"  Generating {n_rl_seqs} RL training sequences...")
    rl_seqs = generate_sequences_batched(rules, n_rl_seqs, seed=seed + 5000)
    rl_seqs_tensor = torch.from_numpy(rl_seqs.astype(np.int64))

    # ------------------------------------------------------------------
    # Pre-generate batch indices
    # ------------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    repoint_gen = torch.Generator().manual_seed(seed + 3)
    rl_gen = torch.Generator().manual_seed(seed + 4)
    rl_distill_gen = torch.Generator().manual_seed(seed + 6)

    max_ntp_steps = pretrain_steps + total_finetune_steps
    train_indices = [
        torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(max_ntp_steps)
    ]
    repoint_indices = [
        torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                       generator=repoint_gen)
        for _ in range(retrain_steps)
    ]
    max_rl_steps = n_cycles * wake_steps
    rl_indices = [
        torch.randint(n_rl_seqs, (batch_size,), generator=rl_gen)
        for _ in range(max_rl_steps)
    ]
    max_rl_distill_steps = n_cycles * distill_steps
    rl_distill_indices = [
        torch.randint(n_rl_seqs, (batch_size,), generator=rl_distill_gen)
        for _ in range(max_rl_distill_steps)
    ]

    n_evals = 600
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
    # Eval helpers
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

    def compute_activation_norms(model, n_batches=10):
        model.eval()
        layer_norms = {f"post_block{i}": [] for i in range(n_layer)}
        with torch.no_grad():
            for bi in range(min(n_batches, probe_batches)):
                x = torch.stack([val_data[i:i + seq_len]
                                 for i in probe_indices[bi]]).to(device)
                _, _, vi = model(x, return_intermediates=True)
                for lk in layer_norms:
                    layer_norms[lk].append(
                        vi[lk].norm(dim=-1).mean().item())
        return {k: float(np.mean(v_)) for k, v_ in layer_norms.items()}

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
        print(f"  [{label:>20s}] rob: "
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
        eval_data_t = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
        all_res = []
        all_cos = []
        with torch.no_grad():
            for i in range(0, len(eval_data_t), batch_size):
                batch = eval_data_t[i:i + batch_size]
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
        model.eval()
        if fm is not None:
            fm.eval(); ugate.eval()
        eval_data_t = torch.from_numpy(
            eval_seqs[:n_gen_eval_seqs].astype(np.int64)).to(device)

        all_correct = []
        with torch.no_grad():
            for i in range(0, n_gen_eval_seqs, batch_size):
                batch = eval_data_t[i:i + batch_size]
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
        print(f"  [{label}] gen: overall={results['overall']:.3f} {acc_str}")
        return results

    def full_checkpoint_eval(model, cycle_idx, fm, ugate, is_milestone=False):
        ckpt = {}
        ckpt["val_loss"] = eval_model(model)

        ckpt["per_level"] = _eval_per_level(
            model, eval_seqs, seq_len, s, L, v, batch_size, device)

        ckpt["per_layer_eta2"] = _compute_per_layer_eta2(
            model, eval_seqs, level_features, level_rules,
            n_layer, s, L, batch_size, device)

        ckpt["generation_accuracy"] = eval_generation_accuracy(
            model, f"c{cycle_idx+1}", fm=fm, ugate=ugate)
        ckpt["generation_accuracy_standalone"] = eval_generation_accuracy(
            model, f"c{cycle_idx+1} (no FM)")

        ckpt["residual_stats"] = compute_residual_stats(
            model, fm, f"c{cycle_idx+1}")
        gs = get_ug_gate_stats(ugate, model, fm)
        ckpt["gate_stats"] = gs
        ckpt["activation_norms"] = compute_activation_norms(model)

        if is_milestone:
            ckpt["robustness"] = measure_robustness(
                model, f"c{cycle_idx+1} milestone")
            ckpt["self_knowledge"] = self_knowledge_probes(
                model, fm, f"c{cycle_idx+1} milestone")

        pl = ckpt["per_level"]
        level_str = " ".join(
            f"L{lvl}={pl['per_level'][lvl]['loss']:.3f}"
            for lvl in sorted(pl['per_level'].keys()))
        norms = ckpt["activation_norms"]
        print(f"    val={ckpt['val_loss']:.4f} {level_str}")
        print(f"    gate={gs['mean']:.3f} "
              f"cos={ckpt['residual_stats']['fwd_cosine']:.4f} "
              f"res_norm={ckpt['residual_stats']['res_norm']:.1f} "
              f"act_norm[{predict_to}]={norms[predict_to]:.1f}")

        return ckpt

    def print_trajectory(checkpoints, prev_l3_loss=None):
        print(f"\n  {'Cycle':>5s} | {'Val':>7s} | {'L3 Loss':>7s} | "
              f"{'L3 eta2':>7s} | {'L3 Gen':>6s} | {'ResNorm':>7s} | "
              f"{'Gate':>5s} | {'FMCos':>6s} | {'ActNorm':>7s}")
        print(f"  {'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-"
              f"{'-'*7}-+-{'-'*5}-+-{'-'*6}-+-{'-'*7}")
        for ci in sorted(checkpoints.keys()):
            ck = checkpoints[ci]
            vl = ck.get("val_loss", 0)
            pl = ck.get("per_level", {}).get("per_level", {})
            l3_loss = pl.get(3, {}).get("loss", 0)
            ple = ck.get("per_layer_eta2", {})
            l3_eta2 = ple.get(f"post_block{n_layer-1}", {}).get(
                "level_3", {}).get("feature_eta2", 0)
            gas = ck.get("generation_accuracy_standalone", {})
            l3_gen = gas.get("L3", 0)
            rn = ck.get("residual_stats", {}).get("res_norm", 0)
            gate = ck.get("gate_stats", {}).get("mean", 0)
            fc = ck.get("fm_cosine", 0)
            an = ck.get("activation_norms", {}).get(predict_to, 0)

            flag = ""
            if prev_l3_loss is not None and l3_loss > 0 and prev_l3_loss > 0:
                delta = (l3_loss - prev_l3_loss) / prev_l3_loss
                if delta < -0.10:
                    flag = " <-- L3 DROP!"
            prev_l3_loss = l3_loss

            print(f"  {ci:5d} | {vl:7.4f} | {l3_loss:7.3f} | "
                  f"{l3_eta2:7.4f} | {l3_gen:6.3f} | {rn:7.1f} | "
                  f"{gate:5.3f} | {fc:6.4f} | {an:7.1f}{flag}")

    def save_results(checkpoints, wake_trajectory, pretrain_info,
                     halted_at=None):
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
                "lr": lr, "fwd_lr": fwd_lr, "lambda_local": lambda_local,
                "distill_alpha": distill_alpha, "distill_lr": distill_lr,
                "temperature": temperature, "rl_loss_scale": rl_loss_scale,
                "lambda_ntp": lambda_ntp, "ntp_mask_rate": ntp_mask_rate,
                "batch_size": batch_size, "seed": seed,
                "weight_decay_mode": weight_decay_mode,
                "strong_weight_decay": strong_weight_decay,
                "wake_wd": wake_wd, "sleep_wd": sleep_wd,
                "norm_halt_ratio": norm_halt_ratio,
            },
            "pretrain": pretrain_info,
            "checkpoints": checkpoints,
            "wake_trajectory": wake_trajectory,
        }
        if halted_at is not None:
            result["halted_at_cycle"] = halted_at
            result["halt_reason"] = "activation_norm_exceeded"
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ==================================================================
    # Phase 1: Pre-training
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

    pretrain_pl = _eval_per_level(
        model, eval_seqs, seq_len, s, L, v, batch_size, device)
    level_str = " ".join(
        f"L{lvl}={pretrain_pl['per_level'][lvl]['loss']:.3f}"
        for lvl in sorted(pretrain_pl['per_level'].keys()))
    print(f"  Pre-trained per-level: {level_str}")

    pretrain_gen = eval_generation_accuracy(model, "pretrained")
    baseline_norms = compute_activation_norms(model)
    baseline_target_norm = baseline_norms[predict_to]
    norm_halt_threshold = baseline_target_norm * norm_halt_ratio
    print(f"  Baseline activation norm at {predict_to}: {baseline_target_norm:.1f}")
    print(f"  Norm halt threshold: {norm_halt_threshold:.1f} "
          f"({norm_halt_ratio}x baseline)")

    pretrain_info = {
        "per_level": pretrain_pl,
        "generation_accuracy": pretrain_gen,
        "activation_norms": baseline_norms,
    }

    pretrained_state = {k: v_.cpu().clone()
                        for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()

    # ==================================================================
    # Phase 2: Gen-distill ratchet (40 cycles)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  GEN-DISTILL RATCHET: {n_cycles} cycles")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    fm = make_fm()
    fm.load_state_dict(init_fm_state)
    ugate = UnifiedGate(n_embd, ug_hidden).to(device)

    checkpoints = {}
    rl_baseline = None
    wake_trajectory = []
    halted_at = None

    for cycle in range(n_cycles):
        is_milestone = ((cycle + 1) % milestone_interval == 0
                        or cycle == n_cycles - 1)
        print(f"\n  --- Cycle {cycle + 1}/{n_cycles}"
              f"{' [MILESTONE]' if is_milestone else ''} ---")

        # === WAKE: RL with FM ===
        print(f"  [WAKE] {wake_steps} RL steps")
        main_params = list(model.parameters()) + list(ugate.parameters())
        opt_main = torch.optim.AdamW(
            main_params, lr=lr, weight_decay=wake_wd)
        opt_fwd = torch.optim.AdamW(
            fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            model.eval(); fm.eval(); ugate.eval()
            rl_step = cycle * wake_steps + step
            idx = rl_indices[rl_step]
            seqs = rl_seqs_tensor[idx].to(device)
            prefix = seqs[:, :prefix_len]
            true_suffix = seqs[:, prefix_len:]

            with torch.no_grad():
                generated = _generate_suffix(
                    model, prefix, suffix_len, temperature,
                    fm, ugate, cerebellar_input_block,
                    inject_after_block)

            correct = (generated == true_suffix).float()
            reward = correct.mean(dim=1)
            if rl_baseline is None:
                rl_baseline = reward.mean().item()
            else:
                rl_baseline = (0.95 * rl_baseline
                               + 0.05 * reward.mean().item())
            advantage = (reward - rl_baseline).detach()

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

            suffix_logits = logits[
                :, prefix_len - 1:prefix_len - 1 + suffix_len, :]
            log_probs = F.log_softmax(suffix_logits, dim=-1)
            gen_log_probs = log_probs.gather(
                2, generated.unsqueeze(-1)).squeeze(-1)
            rl_loss = -(advantage.unsqueeze(1) * gen_log_probs).mean()

            fwd_pred = fwd_cache["pred"]
            target_acts = intermediates[predict_to]
            gate_w = fwd_cache["gate_w"]
            gw_scalar = gate_w.detach().mean()
            r = target_acts - fwd_pred.detach()
            local_loss = (gw_scalar * r ** 2).mean()

            ntp_step = pretrain_steps + cycle * steps_per_cycle + step
            ntp_x = torch.stack([train_data[i:i + seq_len]
                                 for i in train_indices[ntp_step]]
                                ).to(device)
            ntp_y = torch.stack([train_data[i + 1:i + seq_len + 1]
                                 for i in train_indices[ntp_step]]
                                ).to(device)
            ntp_logits, _ = model(ntp_x)
            ntp_per_pos = F.cross_entropy(
                ntp_logits.view(-1, v), ntp_y.view(-1), reduction='none')
            if ntp_mask_rate > 0:
                ntp_mask = (torch.rand(ntp_per_pos.shape[0],
                                       device=device)
                            >= ntp_mask_rate).float()
                ntp_loss = ((ntp_per_pos * ntp_mask).sum()
                            / ntp_mask.sum().clamp(min=1))
            else:
                ntp_loss = ntp_per_pos.mean()

            total_loss = (rl_loss_scale * rl_loss
                          + lambda_ntp * ntp_loss
                          + lambda_local * local_loss)

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
                print(f"    w{step}: val={vl:.4f} "
                      f"reward={reward.mean().item():.3f} "
                      f"rl={rl_loss.item():.4f} "
                      f"ntp={ntp_loss.item():.4f} "
                      f"ll={local_loss.item():.5f} "
                      f"gw={gw_scalar:.3f}")
                wake_trajectory.append({
                    "cycle": cycle, "step": step,
                    "val_loss": vl, "reward": reward.mean().item(),
                })

        del opt_main, opt_fwd

        # === SLEEP: Gen-based distillation ===
        print(f"  [SLEEP] gen distillation ({distill_steps} steps)")

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
            student.parameters(), lr=distill_lr, weight_decay=sleep_wd)

        for step in range(distill_steps):
            student.train()
            ntp_step = (pretrain_steps + cycle * steps_per_cycle
                        + wake_steps + step)

            rl_d_step = cycle * distill_steps + step
            d_idx = rl_distill_indices[rl_d_step]
            d_seqs = rl_seqs_tensor[d_idx].to(device)
            d_prefix = d_seqs[:, :prefix_len]

            with torch.no_grad():
                t_gen, t_logits = _generate_suffix_with_logits(
                    teacher, d_prefix, suffix_len, temperature,
                    teacher_fm, teacher_ugate,
                    cerebellar_input_block, inject_after_block)

            full_seq = torch.cat([d_prefix, t_gen], dim=1)
            s_logits, _ = student(full_seq)
            s_suffix_logits = s_logits[
                :, prefix_len - 1:prefix_len - 1 + suffix_len, :]

            kl_loss = F.kl_div(
                F.log_softmax(s_suffix_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            )

            ntp_x = torch.stack(
                [train_data[i:i + seq_len]
                 for i in train_indices[ntp_step]]).to(device)
            ntp_y = torch.stack(
                [train_data[i + 1:i + seq_len + 1]
                 for i in train_indices[ntp_step]]).to(device)
            s_ntp_logits, _ = student(ntp_x)
            ce_loss = F.cross_entropy(
                s_ntp_logits.view(-1, v), ntp_y.view(-1))

            loss = (distill_alpha * kl_loss
                    + (1 - distill_alpha) * ce_loss)

            opt_student.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_student.step()

            if step % 100 == 0 or step == distill_steps - 1:
                svl = eval_model(student)
                print(f"    d{step}: val={svl:.4f} "
                      f"kl={kl_loss.item():.4f} ce={ce_loss.item():.4f}")

        model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_ugate, student, opt_student
        torch.cuda.empty_cache()

        # === REPOINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
        fm, fm_cos = train_fresh_fm(
            model, retrain_steps, fm_seed, f"c{cycle+1}")

        # === CHECKPOINT EVAL ===
        print(f"\n  CHECKPOINT cycle {cycle + 1}")
        ckpt = full_checkpoint_eval(
            model, cycle, fm, ugate, is_milestone=is_milestone)
        ckpt["fm_cosine"] = fm_cos
        checkpoints[cycle + 1] = ckpt

        # === NORM FAILSAFE ===
        current_norm = ckpt["activation_norms"][predict_to]
        norm_ratio = current_norm / baseline_target_norm
        if current_norm > norm_halt_threshold:
            print(f"\n  *** NORM HALT TRIGGERED ***")
            print(f"  Activation norm at {predict_to}: {current_norm:.1f} "
                  f"({norm_ratio:.1f}x baseline)")
            print(f"  Threshold: {norm_halt_threshold:.1f} "
                  f"({norm_halt_ratio}x baseline)")
            print(f"  Saving results and halting at cycle {cycle + 1}.")
            halted_at = cycle + 1
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"model_c{cycle+1}_halted.pt"))
            torch.save(fm.state_dict(),
                       os.path.join(save_dir, f"fm_c{cycle+1}_halted.pt"))
            save_results(checkpoints, wake_trajectory, pretrain_info,
                         halted_at=halted_at)
            print_trajectory(checkpoints)
            return {"halted": True, "cycle": halted_at,
                    "checkpoints": checkpoints}

        # === PHASE TRANSITION DETECTION ===
        if cycle > 0:
            prev_pl = checkpoints[cycle].get(
                "per_level", {}).get("per_level", {})
            curr_pl = ckpt.get("per_level", {}).get("per_level", {})
            for lvl in [3, 4, 5]:
                prev_loss = prev_pl.get(lvl, {}).get("loss", 0)
                curr_loss = curr_pl.get(lvl, {}).get("loss", 0)
                if prev_loss > 0 and curr_loss > 0:
                    delta = (curr_loss - prev_loss) / prev_loss
                    if delta < -0.10:
                        print(f"\n  *** PHASE TRANSITION SIGNAL: "
                              f"L{lvl} loss dropped {abs(delta)*100:.1f}% "
                              f"({prev_loss:.3f} -> {curr_loss:.3f}) ***")

        # === PERIODIC SAVE ===
        if (cycle + 1) % save_interval == 0:
            print(f"\n  Saving intermediate results (cycle {cycle+1})")
            save_results(checkpoints, wake_trajectory, pretrain_info)
            print_trajectory(checkpoints)

        # === MILESTONE MODEL CHECKPOINT ===
        if is_milestone:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"model_c{cycle+1}.pt"))
            torch.save(fm.state_dict(),
                       os.path.join(save_dir, f"fm_c{cycle+1}.pt"))
            volume.commit()

    # ==================================================================
    # Final save
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  COMPLETED: {n_cycles} cycles")
    print(f"{'='*60}")

    save_results(checkpoints, wake_trajectory, pretrain_info)

    torch.save(model.state_dict(),
               os.path.join(save_dir, f"model_final.pt"))
    torch.save(fm.state_dict(),
               os.path.join(save_dir, f"fm_final.pt"))
    volume.commit()

    # ==================================================================
    # Summary
    # ==================================================================
    print_trajectory(checkpoints)

    # Per-layer eta² at final layer for L3/L4 (the grokking precursors)
    print(f"\n  === L3/L4 Feature eta² at final layer (grokking signal) ===")
    print(f"  {'Cycle':>5s} | {'L3 @blk0':>8s} | {'L3 @blk3':>8s} | "
          f"{'L3 @blk5':>8s} | {'L4 @blk5':>8s}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for ci in sorted(checkpoints.keys()):
        ple = checkpoints[ci].get("per_layer_eta2", {})
        l3_b0 = ple.get("post_block0", {}).get(
            "level_3", {}).get("feature_eta2", 0)
        l3_b3 = ple.get("post_block3", {}).get(
            "level_3", {}).get("feature_eta2", 0)
        l3_b5 = ple.get(f"post_block{n_layer-1}", {}).get(
            "level_3", {}).get("feature_eta2", 0)
        l4_b5 = ple.get(f"post_block{n_layer-1}", {}).get(
            "level_4", {}).get("feature_eta2", 0)
        print(f"  {ci:5d} | {l3_b0:8.4f} | {l3_b3:8.4f} | "
              f"{l3_b5:8.4f} | {l4_b5:8.4f}")

    # Activation norm trajectory
    print(f"\n  === Activation norm trajectory ({predict_to}) ===")
    print(f"  Baseline: {baseline_target_norm:.1f}")
    for ci in sorted(checkpoints.keys()):
        an = checkpoints[ci].get("activation_norms", {}).get(predict_to, 0)
        ratio = an / baseline_target_norm if baseline_target_norm > 0 else 0
        bar = "#" * min(50, int(ratio * 10))
        print(f"  c{ci:3d}: {an:7.1f} ({ratio:.2f}x) {bar}")

    if halted_at:
        print(f"\n  *** RUN HALTED at cycle {halted_at} "
              f"(norm {norm_halt_ratio}x exceeded) ***")

    print(f"\n  Saved to {save_dir}/results.json")
    return {"halted": False, "checkpoints": checkpoints}


@app.local_entrypoint()
def main(weight_decay_mode: str = "baseline",
         strong_weight_decay: float = 0.1):
    result = rhm_rl_gen_distill_extended.remote(
        weight_decay_mode=weight_decay_mode,
        strong_weight_decay=strong_weight_decay,
    )
    halted = result.get("halted", False)
    if halted:
        print(f"\nRun halted at cycle {result.get('cycle')} "
              f"due to activation norm blowup.")
    else:
        print(f"\nRHM RL gen-distill extended ({len(result.get('checkpoints', {}))} cycles) complete.")
