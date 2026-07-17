"""RHM RL Generation-Based Distillation Test.

Tests whether generation-based distillation preserves NTP quality while
still transferring generation ability from the FM-injected teacher.

Hypothesis: standard NTP distillation (run 3) destroys NTP (val loss 0.82
during wake -> 2.29 after distillation) because the teacher's logits on NTP
data are ~100% injection-dependent (gate at 0.998). Matching those logits
forces the student to replicate computation it can't do standalone.

Generation-based distillation should produce a smaller teacher-student logit
gap: the teacher generates suffixes autoregressively with FM injection, and
the student matches those logits on the generation data. NTP is preserved
via a separate CE loss on ground-truth data (no teacher KL on NTP).

Design: Two conditions with shared pre-training and identical wake phases:
  gen_distill: sleep = KL on teacher-generated suffixes + CE on NTP
  ntp_distill: sleep = KL+CE on NTP data (reproduces run 3)

Both use: sparse NTP (mask=0.95), lambda_local=1.0, 4 cycles, default FM.

Reproduction:
  cd experiments/
  modal run --detach -m rhm.ratchet.rhm_rl_gen_distill::rhm_rl_gen_distill
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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-rl-gen-distill", image=image)


def _generate_suffix_with_logits(model, prefix, suffix_len, temperature,
                                 fm, ugate, cerebellar_input_block,
                                 inject_after_block):
    """Autoregressively generate suffix AND collect per-position logits."""
    import torch
    import torch.nn.functional as F

    x = prefix.clone()
    generated = []
    all_logits = []

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

        next_logits = logits[:, -1, :]
        all_logits.append(next_logits)

        probs = F.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token)
        x = torch.cat([x, next_token], dim=1)

    return torch.cat(generated, dim=1), torch.stack(all_logits, dim=1)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,
    memory=32768,
)
def rhm_rl_gen_distill(
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

    print(f"RHM RL GEN-DISTILL TEST on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D, "
          f"{predict_from}->{predict_to}, inject after block {inject_after_block}")
    print(f"  Pre-train: {pretrain_steps} NTP steps")
    print(f"  Fine-tune: {n_cycles} cycles x ({wake_steps} wake + "
          f"{distill_steps} sleep) = {total_finetune_steps} steps")
    print(f"  Generation: prefix={prefix_len}, suffix={suffix_len}")
    print(f"  Distillation: alpha={distill_alpha}, lr={distill_lr}")
    print(f"  Conditions: gen_distill, ntp_distill")

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
    rl_distill_gen = torch.Generator().manual_seed(seed + 6)

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
        print(f"  [{label}] gen_acc: overall={results['overall']:.3f} "
              f"{acc_str}")
        return results

    def full_checkpoint_eval(model, cond_label, cycle_idx, fm=None,
                              ugate=None):
        ckpt = {}
        vl = eval_model(model)
        ckpt["val_loss"] = vl

        pl = _eval_per_level(
            model, eval_seqs, seq_len, s, L, v, batch_size, device)
        ckpt["per_level"] = pl

        per_layer = _compute_per_layer_eta2(
            model, eval_seqs, level_features, level_rules,
            n_layer, s, L, batch_size, device)
        ckpt["per_layer_eta2"] = per_layer

        gen_acc = eval_generation_accuracy(
            model, f"{cond_label} c{cycle_idx+1}",
            fm=fm if fm is not None else None,
            ugate=ugate if ugate is not None else None)
        ckpt["generation_accuracy"] = gen_acc

        if fm is not None:
            gen_acc_standalone = eval_generation_accuracy(
                model, f"{cond_label} c{cycle_idx+1} (no FM)")
            ckpt["generation_accuracy_standalone"] = gen_acc_standalone

        is_final = (cycle_idx == n_cycles - 1)
        if fm is not None:
            ckpt["residual_stats"] = compute_residual_stats(
                model, fm, f"{cond_label} c{cycle_idx+1}")
            gs = get_ug_gate_stats(ugate, model, fm)
            ckpt["gate_stats"] = gs
            print(f"    Gate: mean={gs['mean']:.3f} "
                  f"sparse(<0.1)={gs['sparse_01']:.3f}")

        if is_final:
            ckpt["robustness"] = measure_robustness(
                model, f"{cond_label} final")
            if fm is not None:
                ckpt["self_knowledge"] = self_knowledge_probes(
                    model, fm, f"{cond_label} final")

        level_str = " ".join(
            f"L{lvl}={pl['per_level'][lvl]['loss']:.3f}"
            for lvl in sorted(pl['per_level'].keys()))
        print(f"    val={vl:.4f} {level_str}")

        return ckpt

    # ==================================================================
    # Phase 1: Pre-training (shared)
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

    pretrained_state = {k: v_.cpu().clone()
                        for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Saved pre-trained checkpoint")

    # ==================================================================
    # Phase 2: Fine-tuning both conditions
    # ==================================================================
    all_results = {}

    for distill_mode in ["gen", "ntp"]:
        cond = f"RL_FM_{distill_mode}"
        print(f"\n{'='*60}")
        print(f"  {cond}: RL + FM ratchet, {distill_mode} distillation "
              f"({n_cycles} cycles)")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(pretrained_state)
        fm = make_fm()
        fm.load_state_dict(init_fm_state)
        ugate = UnifiedGate(n_embd, ug_hidden).to(device)

        checkpoints = {}
        rl_baseline = None
        wake_trajectory = []

        for cycle in range(n_cycles):
            print(f"\n  --- {cond} Cycle {cycle + 1}/{n_cycles} ---")

            # === WAKE: RL with FM (identical for both conditions) ===
            print(f"  [WAKE] {wake_steps} RL steps")
            main_params = list(model.parameters()) + list(ugate.parameters())
            opt_main = torch.optim.AdamW(
                main_params, lr=lr, weight_decay=0.01)
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
                    mean_reward = reward.mean().item()
                    print(f"    step w{step}: val={vl:.4f} "
                          f"reward={mean_reward:.3f} "
                          f"rl={rl_loss.item():.4f} "
                          f"ntp={ntp_loss.item():.4f} "
                          f"ll={local_loss.item():.5f} "
                          f"gw={gw_scalar:.3f}")
                    wake_trajectory.append({
                        "cycle": cycle, "step": step,
                        "val_loss": vl, "reward": mean_reward,
                    })

            del opt_main, opt_fwd

            # === SLEEP: Distillation ===
            if distill_steps > 0:
                print(f"  [SLEEP] {distill_mode} distillation "
                      f"({distill_steps} steps, alpha={distill_alpha})")

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
                    ntp_step = (pretrain_steps + cycle * steps_per_cycle
                                + wake_steps + step)

                    if distill_mode == "gen":
                        # ---- Generation-based distillation ----
                        # Teacher generates suffixes with FM injection
                        rl_d_step = cycle * distill_steps + step
                        d_idx = rl_distill_indices[rl_d_step]
                        d_seqs = rl_seqs_tensor[d_idx].to(device)
                        d_prefix = d_seqs[:, :prefix_len]

                        with torch.no_grad():
                            t_gen, t_logits = \
                                _generate_suffix_with_logits(
                                    teacher, d_prefix, suffix_len,
                                    temperature, teacher_fm, teacher_ugate,
                                    cerebellar_input_block,
                                    inject_after_block)

                        # Student scores teacher's generation
                        full_seq = torch.cat([d_prefix, t_gen], dim=1)
                        s_logits, _ = student(full_seq)
                        s_suffix_logits = s_logits[
                            :, prefix_len - 1:
                            prefix_len - 1 + suffix_len, :]

                        kl_loss = F.kl_div(
                            F.log_softmax(s_suffix_logits, dim=-1),
                            F.softmax(t_logits, dim=-1),
                            reduction="batchmean",
                        )

                        # Separate CE on ground-truth NTP data
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

                    else:
                        # ---- Standard NTP distillation (run 3) ----
                        ntp_x = torch.stack(
                            [train_data[i:i + seq_len]
                             for i in train_indices[ntp_step]]).to(device)
                        ntp_y = torch.stack(
                            [train_data[i + 1:i + seq_len + 1]
                             for i in train_indices[ntp_step]]).to(device)

                        with torch.no_grad():
                            def t_cb(act):
                                fp = teacher_fm(act)
                                inj, _ = teacher_ugate(act, fp)
                                return inj
                            teacher_logits, _, _ = teacher(
                                ntp_x, return_intermediates=True,
                                cerebellar_fn=t_cb,
                                cerebellar_input_block=cerebellar_input_block,
                                cerebellar_inject_block=inject_after_block,
                            )

                        s_logits, _ = student(ntp_x)
                        kl_loss = F.kl_div(
                            F.log_softmax(s_logits, dim=-1),
                            F.softmax(teacher_logits, dim=-1),
                            reduction="batchmean",
                        )
                        ce_loss = F.cross_entropy(
                            s_logits.view(-1, v), ntp_y.view(-1))
                        loss = (distill_alpha * kl_loss
                                + (1 - distill_alpha) * ce_loss)

                    opt_student.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), 1.0)
                    opt_student.step()

                    if step % 100 == 0 or step == distill_steps - 1:
                        svl = eval_model(student)
                        print(f"    distill step {step}: "
                              f"val={svl:.4f} kl={kl_loss.item():.4f} "
                              f"ce={ce_loss.item():.4f}")

                model.load_state_dict(student.state_dict())
                del (teacher, teacher_fm, teacher_ugate,
                     student, opt_student)
            else:
                print(f"  [SLEEP] Skipped (distill_steps=0)")

            torch.cuda.empty_cache()

            # === REPOINT ===
            fm_seed = seed + 100 * (cycle + 1)
            print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
            fm, fm_cos = train_fresh_fm(
                model, retrain_steps, fm_seed, f"{cond} c{cycle+1}")

            # === CHECKPOINT ===
            print(f"\n  {cond} CHECKPOINT cycle {cycle + 1}")
            ckpt = full_checkpoint_eval(
                model, cond, cycle, fm=fm, ugate=ugate)
            ckpt["fm_cosine"] = fm_cos
            checkpoints[(cycle + 1) * steps_per_cycle] = ckpt

        all_results[cond] = {
            "checkpoints": checkpoints,
            "wake_trajectory": wake_trajectory,
        }
        del model, fm, ugate
        torch.cuda.empty_cache()

    # ==================================================================
    # Save results
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_rl_gen_distill/{key}"
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
            "distill_alpha": distill_alpha, "distill_lr": distill_lr,
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
    print(f"SUMMARY: RHM RL Gen-Distill Test")
    print(f"  Distill alpha={distill_alpha}, steps={distill_steps}")
    print(f"  Uniform baseline: {uniform:.4f}")
    print(f"{'='*80}")

    conds = ["RL_FM_gen", "RL_FM_ntp"]

    print(f"\n  === Val loss at each checkpoint ===")
    print(f"  {'Cycle':>6s}  {'gen_distill':>12s}  {'ntp_distill':>12s}")
    for ci, cs in enumerate(checkpoint_steps):
        parts = [f"{ci+1:6d}"]
        for c in conds:
            ckpt = all_results.get(c, {}).get("checkpoints", {}).get(cs, {})
            vl = ckpt.get("val_loss", 0)
            parts.append(f"{vl:12.4f}")
        print("  " + "  ".join(parts))

    final_cs = checkpoint_steps[-1]

    print(f"\n  === Generation accuracy (final) ===")
    header = f"  {'Cond':>12s}  {'overall':>7s}"
    for lvl in range(max_suffix_level + 1):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    for c in conds:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(
            final_cs, {})
        ga = ckpt.get("generation_accuracy", {})
        gas = ckpt.get("generation_accuracy_standalone", {})
        row = f"  {c:>12s}  {ga.get('overall', 0):7.3f}"
        for lvl in range(max_suffix_level + 1):
            row += f"  {ga.get(f'L{lvl}', 0):7.3f}"
        print(row)
        if gas:
            row2 = f"  {'(standalone)':>12s}  {gas.get('overall', 0):7.3f}"
            for lvl in range(max_suffix_level + 1):
                row2 += f"  {gas.get(f'L{lvl}', 0):7.3f}"
            print(row2)

    print(f"\n  === Per-level NTP loss (final) ===")
    header = f"  {'Cond':>12s}  {'val':>7s}"
    for lvl in range(L):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    for c in conds:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(
            final_cs, {})
        pl = ckpt.get("per_level", {}).get("per_level", {})
        vl = ckpt.get("val_loss", 0)
        row = f"  {c:>12s}  {vl:7.4f}"
        for lvl in range(L):
            if lvl in pl:
                row += f"  {pl[lvl]['loss']:7.4f}"
            else:
                row += f"  {'--':>7s}"
        print(row)

    print(f"\n  === FM cosine & gate (final) ===")
    for c in conds:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(
            final_cs, {})
        fc = ckpt.get("fm_cosine", 0)
        gs = ckpt.get("gate_stats", {})
        rs = ckpt.get("residual_stats", {})
        print(f"  {c}: fm_cos={fc:.4f} "
              f"gate_mean={gs.get('mean',0):.3f} "
              f"res_cos={rs.get('fwd_cosine',0):.4f}")

    print(f"\n  === Self-knowledge (final) ===")
    for c in conds:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(
            final_cs, {})
        sk = ckpt.get("self_knowledge", {})
        if sk:
            print(f"  {c}: " + " | ".join(
                f"{k}={v_:.3f}" for k, v_ in sk.items()))

    print(f"\n  === Robustness eps=1.0 (final) ===")
    for c in conds:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(
            final_cs, {})
        rob = ckpt.get("robustness", {}).get("1.0", None)
        if rob is not None:
            print(f"  {c}: {rob:+.4f}")

    print(f"\n  === Per-layer feature eta² (final) ===")
    for c in conds:
        ckpt = all_results.get(c, {}).get("checkpoints", {}).get(
            final_cs, {})
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
                eta2 = layer_data.get(
                    f"level_{lvl}", {}).get("feature_eta2", 0)
                row += f"  {eta2:7.4f}"
            print(row)

    print(f"\n  === Wake trajectory (val loss during wake) ===")
    for c in conds:
        wt = all_results.get(c, {}).get("wake_trajectory", [])
        if wt:
            print(f"  {c}:")
            for entry in wt:
                print(f"    c{entry['cycle']+1} w{entry['step']:4d}: "
                      f"val={entry['val_loss']:.4f}")

    print(f"\n  Saved to {save_dir}/results.json")
    return result


@app.local_entrypoint()
def main():
    result = rhm_rl_gen_distill.remote()
    print("\nRHM RL gen-distill test complete.")
