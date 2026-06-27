"""RHM FOMAML RL Ratchet: Bilevel meta-learning with sparse RL supervision.

The NTP-only FOMAML ratchet (rhm_fomaml_ratchet.py) showed the gate closing
to ~0 (dense NTP makes local loss redundant). The RL-outer variant showed the
gate stuck at 0.5 (REINFORCE variance drowns the meta-gradient).

This version keeps the sparse RL inner loop but replaces the outer objective
with per-level NTP: NTP loss evaluated only at hierarchy levels >= 2
(~15 positions out of 63, deterministic). This has the MNIST structure:
inner is sparse (~3 NTP positions + 1 RL scalar), outer measures something
the inner mostly misses (15 level-2+ positions), and local loss bridges the
gap (all 64 positions × 192 dims). Deterministic outer = no REINFORCE noise.

Wake phase:
  1. Generate suffix autoregressively with FM injection (no_grad)
  2. REINFORCE loss: advantage * log_prob(generated tokens)
  3. Sparse NTP loss (mask_rate=0.95, ~5% of positions get gradient)
  4. FOMAML-gated local loss: LearningGate(mean_act, mean_fm_err) * FM_error²
  5. Inner gradient with create_graph → virtual params
  6. Outer objective: NTP at levels 2+ under virtual params (deterministic)
  7. Gate update via meta-gradient; model update via detached inner gradient

Sleep phase: Gen-based distillation (same as rhm_fomaml_ratchet.py)

Three conditions, all compute-matched:
  WS_LG: RL wake (REINFORCE + sparse NTP + FOMAML-gated local loss)
         + gen-distill sleep + FM repoint
  WS:    RL wake (REINFORCE + sparse NTP, no local loss)
         + gen-distill sleep + FM repoint
  OL:    Continuous NTP (compute-matched baseline)

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_fomaml_rl_ratchet::rhm_fomaml_rl_ratchet
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key
from rhm.rhm_rl_ratchet import (
    _position_levels,
    _suffix_position_levels,
    _eval_per_level,
    _ensure_corpus,
    _generate_with_traces,
    _compute_hierarchy_eta2,
    _compute_per_layer_eta2,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-fomaml-rl-ratchet", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def rhm_fomaml_rl_ratchet(
    # DGP
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    # Model
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    # FM
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1, fwd_d_head: int = 16,
    fwd_n_head: int = 1, fwd_mlp_mult: float = 0.5,
    # Pre-training
    pretrain_steps: int = 10000,
    # Wake-sleep protocol
    n_cycles: int = 4,
    wake_steps: int = 2000,
    distill_steps: int = 500,
    retrain_steps: int = 2000,
    # Training
    lr: float = 3e-4, fwd_lr: float = 1e-3,
    gate_lr: float = 1e-3,
    distill_lr: float = 1e-4, distill_alpha: float = 0.5,
    lambda_local: float = 1.0, lambda_ntp: float = 1.0,
    rl_loss_scale: float = 1.0, ntp_mask_rate: float = 0.95,
    outer_min_level: int = 2,
    lg_hidden: int = 64,
    batch_size: int = 32,
    # Generation
    temperature: float = 1.0,
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
    from torch.func import functional_call
    from rhm.model import GPT
    from rhm.rhm_data import generate_sequences_batched
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

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

    # Outer objective mask: NTP at hierarchy levels >= outer_min_level
    pos_levels = _position_levels(seq_len, s)
    outer_level_mask = torch.tensor(
        [1.0 if pos_levels[t] >= outer_min_level else 0.0
         for t in range(seq_len - 1)],
        device=device,
    )
    n_outer_positions = int(outer_level_mask.sum().item())

    print(f"RHM FOMAML RL RATCHET on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D, "
          f"{predict_from}->{predict_to}, inject after block {inject_after_block}")
    print(f"  Pre-train: {pretrain_steps} NTP steps")
    print(f"  Fine-tune: {n_cycles} cycles x ({wake_steps} wake + "
          f"{distill_steps} sleep) = {total_finetune_steps} steps")
    print(f"  RL: scale={rl_loss_scale}, NTP mask={ntp_mask_rate}, "
          f"lambda_ntp={lambda_ntp}, lambda_local={lambda_local}")
    print(f"  Outer: NTP at levels >= {outer_min_level} "
          f"({n_outer_positions}/{seq_len-1} positions)")
    print(f"  Conditions: WS_LG (FOMAML+RL), WS (RL only), OL")

    # ------------------------------------------------------------------
    # LearningGate (mean-pooled activations, no CLS)
    # ------------------------------------------------------------------
    class LearningGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            n_params = sum(p.numel() for p in self.parameters())
            print(f"LearningGate: {n_params/1e3:.1f}K params "
                  f"(input={2*d_model}, hidden={d_hidden}, output={d_model})")

        def forward(self, source_mean, fm_error_mean):
            x = torch.cat([source_mean, fm_error_mean], dim=-1)
            return torch.sigmoid(self.net(x))

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def gen_suffix(model, prefix, suffix_len_, temperature_,
                   fm, cgate, cb_input_block, cb_inject_block):
        x = prefix.clone()
        generated = []
        for _ in range(suffix_len_):
            if fm is not None and cgate is not None:
                def cb(act, _fm=fm, _cg=cgate):
                    return _cg(_fm(act))
                logits, _, _ = model(
                    x, return_intermediates=True,
                    cerebellar_fn=cb,
                    cerebellar_input_block=cb_input_block,
                    cerebellar_inject_block=cb_inject_block)
            else:
                logits, _ = model(x)
            next_logits = logits[:, -1, :] / temperature_
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            x = torch.cat([x, next_token], dim=1)
        return torch.cat(generated, dim=1)

    def gen_suffix_with_logits(model, prefix, suffix_len_, temperature_,
                               fm, cgate, cb_input_block, cb_inject_block):
        x = prefix.clone()
        generated = []
        all_logits = []
        for _ in range(suffix_len_):
            if fm is not None and cgate is not None:
                def cb(act, _fm=fm, _cg=cgate):
                    return _cg(_fm(act))
                logits, _, _ = model(
                    x, return_intermediates=True,
                    cerebellar_fn=cb,
                    cerebellar_input_block=cb_input_block,
                    cerebellar_inject_block=cb_inject_block)
            else:
                logits, _ = model(x)
            next_logits = logits[:, -1, :]
            all_logits.append(next_logits)
            probs = F.softmax(next_logits / temperature_, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            x = torch.cat([x, next_token], dim=1)
        return torch.cat(generated, dim=1), torch.stack(all_logits, dim=1)

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

    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy")
             for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(
        rules, n_eval_sequences, seed=12345)
    print(f"  {n_eval_sequences} eval sequences with hierarchy traces")

    n_gen_seqs = 100000
    print(f"  Generating {n_gen_seqs} sequences for RL + gen-distill...")
    gen_seqs = generate_sequences_batched(rules, n_gen_seqs, seed=seed + 5000)
    gen_seqs_tensor = torch.from_numpy(gen_seqs.astype(np.int64))
    print(f"  Gen data: {n_gen_seqs} sequences x {seq_len} tokens")

    # ------------------------------------------------------------------
    # Pre-generate batch indices
    # ------------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    repoint_gen = torch.Generator().manual_seed(seed + 3)
    distill_gen = torch.Generator().manual_seed(seed + 4)
    rl_gen = torch.Generator().manual_seed(seed + 5)
    outer_gen = torch.Generator().manual_seed(seed + 6)

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
        torch.randint(n_gen_seqs, (batch_size,), generator=rl_gen)
        for _ in range(max_rl_steps)
    ]
    max_distill_steps = n_cycles * distill_steps
    distill_indices = [
        torch.randint(n_gen_seqs, (batch_size,), generator=distill_gen)
        for _ in range(max_distill_steps)
    ]
    outer_indices = [
        torch.randint(n_gen_seqs, (batch_size,), generator=outer_gen)
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
              + " | ".join(f"eps={e}:{results[str(e)]:+.4f}"
                           for e in eps_list))
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
                _, _, vi = model(x, return_intermediates=True)
                pred = fm(vi[predict_from])
                residuals.append(
                    (vi[predict_to] - pred).reshape(-1, n_embd).cpu())
                for k in probe_layers:
                    acts_by_layer[k].append(
                        vi[k].reshape(-1, n_embd).cpu())
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
        eval_data_t = torch.from_numpy(
            eval_seqs.astype(np.int64)).to(device)
        all_res = []; all_cos = []
        with torch.no_grad():
            for i in range(0, len(eval_data_t), batch_size):
                batch = eval_data_t[i:i + batch_size]
                if batch.shape[0] < 2:
                    continue
                _, _, vi = model(batch, return_intermediates=True)
                pred = fm(vi[predict_from])
                tgt = vi[predict_to]
                all_res.append((tgt - pred).cpu().numpy())
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
        eta2 = _compute_hierarchy_eta2(
            res_np, level_features, level_rules, s, L)
        stats = {
            "fwd_cosine": mean_cos, "res_norm": mean_norm,
            "eff_rank": eff_rank, "hierarchy_eta2": eta2,
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
        opt = torch.optim.AdamW(
            fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        for step in range(steps):
            fm.train()
            x = torch.stack([train_data[i:i + seq_len]
                             for i in repoint_indices[step]]).to(device)
            with torch.no_grad():
                _, _, vi = model(x, return_intermediates=True)
                source = vi[predict_from]; target = vi[predict_to]
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

    def eval_generation_accuracy(model, label, fm=None, cgate=None):
        model.eval()
        if fm is not None: fm.eval()
        if cgate is not None: cgate.eval()
        eval_data_t = torch.from_numpy(
            eval_seqs[:n_gen_eval_seqs].astype(np.int64)).to(device)
        all_correct = []
        with torch.no_grad():
            for i in range(0, n_gen_eval_seqs, batch_size):
                batch = eval_data_t[i:i + batch_size]
                prefix = batch[:, :prefix_len]
                true_suffix = batch[:, prefix_len:]
                generated = gen_suffix(
                    model, prefix, suffix_len, temperature,
                    fm, cgate, cerebellar_input_block, inject_after_block)
                correct = (generated == true_suffix).float()
                all_correct.append(correct.cpu())
        all_correct = torch.cat(all_correct, dim=0)
        results = {"overall": float(all_correct.mean())}
        for lvl in range(max_suffix_level + 1):
            mask = torch.tensor([l == lvl for l in suffix_levels])
            if mask.sum() > 0:
                results[f"L{lvl}"] = float(all_correct[:, mask].mean())
        acc_str = " ".join(f"L{k}={results.get(f'L{k}', 0):.3f}"
                           for k in range(max_suffix_level + 1))
        print(f"  [{label}] gen_acc: overall={results['overall']:.3f} "
              f"{acc_str}")
        return results

    def get_gate_stats(lgate, model, fm):
        lgate.eval(); model.eval(); fm.eval()
        all_gw = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                x = torch.stack([val_data[i:i + seq_len]
                                 for i in probe_indices[bi]]).to(device)
                _, _, vi = model(x, return_intermediates=True)
                pred = fm(vi[predict_from])
                target = vi[predict_to]
                source_mean = vi[predict_from].mean(dim=1)
                fm_error_mean = (target - pred).mean(dim=1)
                gw = lgate(source_mean, fm_error_mean)
                all_gw.append(gw.cpu())
        gw_cat = torch.cat(all_gw, dim=0)
        dim_mean = gw_cat.mean(dim=0)
        return {
            "mean": float(gw_cat.mean()),
            "std": float(gw_cat.std()),
            "sparse_01": float((dim_mean < 0.1).float().mean()),
        }

    def full_checkpoint_eval(model, cond_label, cycle_idx, fm=None):
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
            model, f"{cond_label} c{cycle_idx+1}")
        ckpt["generation_accuracy"] = gen_acc
        if fm is not None:
            ckpt["residual_stats"] = compute_residual_stats(
                model, fm, f"{cond_label} c{cycle_idx+1}")
            ckpt["self_knowledge"] = self_knowledge_probes(
                model, fm, f"{cond_label} c{cycle_idx+1}")
        ckpt["robustness"] = measure_robustness(
            model, f"{cond_label} c{cycle_idx+1}")
        level_str = " ".join(
            f"L{lvl}={pl['per_level'][lvl]['loss']:.3f}"
            for lvl in sorted(pl['per_level'].keys()))
        print(f"    val={vl:.4f} {level_str}")
        return ckpt

    def run_gen_distill_sleep(model, fm, cgate, cycle):
        print(f"  [SLEEP] Gen-distill ({distill_steps} steps)")
        teacher = make_gpt()
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad = False
        teacher_fm = make_fm()
        teacher_fm.load_state_dict(fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters(): p.requires_grad = False
        teacher_cg = CerebellarGate(n_embd).to(device)
        teacher_cg.load_state_dict(cgate.state_dict())
        teacher_cg.eval()
        for p in teacher_cg.parameters(): p.requires_grad = False

        student = make_gpt()
        student.load_state_dict(model.state_dict())
        opt_s = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()
            d_step = cycle * distill_steps + step
            d_seqs = gen_seqs_tensor[distill_indices[d_step]].to(device)
            d_prefix = d_seqs[:, :prefix_len]

            with torch.no_grad():
                t_gen, t_logits = gen_suffix_with_logits(
                    teacher, d_prefix, suffix_len, temperature,
                    teacher_fm, teacher_cg,
                    cerebellar_input_block, inject_after_block)

            full_seq = torch.cat([d_prefix, t_gen], dim=1)
            s_logits, _ = student(full_seq)
            s_suf = s_logits[:, prefix_len-1:prefix_len-1+suffix_len, :]
            kl_loss = F.kl_div(
                F.log_softmax(s_suf, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean")

            ntp_step = (pretrain_steps + cycle * steps_per_cycle
                        + wake_steps + step)
            ntp_x = torch.stack(
                [train_data[i:i + seq_len]
                 for i in train_indices[ntp_step]]).to(device)
            ntp_y = torch.stack(
                [train_data[i + 1:i + seq_len + 1]
                 for i in train_indices[ntp_step]]).to(device)
            s_ntp, _ = student(ntp_x)
            ce_loss = F.cross_entropy(
                s_ntp.view(-1, v), ntp_y.view(-1))

            loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss
            opt_s.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_s.step()

            if step % 100 == 0 or step == distill_steps - 1:
                svl = eval_model(student)
                print(f"    distill {step}: val={svl:.4f} "
                      f"kl={kl_loss.item():.4f} ce={ce_loss.item():.4f}")

        model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_cg, student, opt_s
        torch.cuda.empty_cache()

    # ==================================================================
    # Pre-training (shared)
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
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == pretrain_steps - 1:
            vl = eval_model(model)
            print(f"  step {step:5d}: val={vl:.4f}")

    pretrain_pl = _eval_per_level(
        model, eval_seqs, seq_len, s, L, v, batch_size, device)
    pretrain_gen = eval_generation_accuracy(model, "pretrained")
    pretrained_state = {k: v_.cpu().clone()
                        for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Saved pre-trained checkpoint")

    # ==================================================================
    # Fine-tuning
    # ==================================================================
    all_results = {}

    # ==================================================================
    # CONDITION 1: WS_LG (FOMAML + RL)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION: WS_LG ({n_cycles} cycles, FOMAML + RL wake)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    fm = make_fm()
    fm.load_state_dict(init_fm_state)
    lgate = LearningGate(n_embd, lg_hidden).to(device)

    wslg_checkpoints = {}
    rl_baseline = [None]

    for cycle in range(n_cycles):
        print(f"\n  --- WS_LG Cycle {cycle + 1}/{n_cycles} ---")
        print(f"  [WAKE] FOMAML + RL ({wake_steps} steps)")

        cgate = CerebellarGate(n_embd).to(device)
        main_params = (list(model.parameters())
                       + list(cgate.parameters()))
        n_model_params = len(list(model.parameters()))
        opt_main = torch.optim.AdamW(
            main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        opt_lg = torch.optim.AdamW(
            lgate.parameters(), lr=gate_lr, weight_decay=0.01)

        for step in range(wake_steps):
            rl_step = cycle * wake_steps + step

            # --- 1. Generate suffix with FM injection (no grad) ---
            model.eval(); fm.eval(); cgate.eval()
            seqs = gen_seqs_tensor[rl_indices[rl_step]].to(device)
            prefix = seqs[:, :prefix_len]
            true_suffix = seqs[:, prefix_len:]

            with torch.no_grad():
                generated = gen_suffix(
                    model, prefix, suffix_len, temperature,
                    fm, cgate, cerebellar_input_block, inject_after_block)

            # --- 2. Rewards and advantages ---
            correct = (generated == true_suffix).float()
            reward = correct.mean(dim=1)
            if rl_baseline[0] is None:
                rl_baseline[0] = reward.mean().item()
            else:
                rl_baseline[0] = (0.95 * rl_baseline[0]
                                  + 0.05 * reward.mean().item())
            advantage = (reward - rl_baseline[0]).detach()

            # --- 3. Score generated tokens (with grad) ---
            model.train(); fm.train()
            cgate.train(); lgate.train()
            scored_seq = torch.cat([prefix, generated], dim=1)
            fwd_cache = {}

            def cb_fn(act, _cache=fwd_cache):
                fp = fm(act.detach())
                _cache["pred"] = fp
                return cgate(fp.detach())

            logits, _, intermediates = model(
                scored_seq, return_intermediates=True,
                cerebellar_fn=cb_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

            # RL loss
            suf_logits = logits[
                :, prefix_len-1:prefix_len-1+suffix_len, :]
            log_probs = F.log_softmax(suf_logits, dim=-1)
            gen_lp = log_probs.gather(
                2, generated.unsqueeze(-1)).squeeze(-1)
            rl_loss = -(advantage.unsqueeze(1) * gen_lp).mean()

            # FM prediction + gated local loss
            fwd_pred = fwd_cache["pred"]
            target = intermediates[predict_to]
            source_mean = intermediates[predict_from].mean(
                dim=1).detach()
            fm_error_mean = (target - fwd_pred).detach().mean(dim=1)
            gate_w = lgate(source_mean, fm_error_mean)
            r = target - fwd_pred.detach()
            gated_ll = (gate_w.unsqueeze(1) * r ** 2).mean()

            # Sparse NTP loss
            ntp_step = pretrain_steps + cycle * steps_per_cycle + step
            ntp_x = torch.stack(
                [train_data[i:i + seq_len]
                 for i in train_indices[ntp_step]]).to(device)
            ntp_y = torch.stack(
                [train_data[i + 1:i + seq_len + 1]
                 for i in train_indices[ntp_step]]).to(device)
            ntp_logits, _ = model(ntp_x)
            ntp_per_pos = F.cross_entropy(
                ntp_logits.view(-1, v), ntp_y.view(-1),
                reduction='none')
            ntp_mask = (torch.rand(ntp_per_pos.shape[0],
                                   device=device)
                        >= ntp_mask_rate).float()
            ntp_loss = ((ntp_per_pos * ntp_mask).sum()
                        / ntp_mask.sum().clamp(min=1))

            # --- 4. Inner loss + FOMAML gradient ---
            L_inner = (rl_loss_scale * rl_loss
                       + lambda_ntp * ntp_loss
                       + lambda_local * gated_ll)

            inner_grads = torch.autograd.grad(
                L_inner, main_params, create_graph=True)
            grads_for_update = [g.detach().clone()
                                for g in inner_grads]

            # --- 5. Virtual parameters ---
            virtual_model_params = {
                pname: p - lr * g
                for (pname, p), g in zip(
                    model.named_parameters(),
                    inner_grads[:n_model_params])
            }
            virtual_cgate_params = {
                pname: p - lr * g
                for (pname, p), g in zip(
                    cgate.named_parameters(),
                    inner_grads[n_model_params:])
            }

            # --- 6. Outer: NTP at levels 2+ under virtual params ---
            outer_seqs = gen_seqs_tensor[
                outer_indices[rl_step]].to(device)
            outer_x = outer_seqs[:, :-1]
            outer_y = outer_seqs[:, 1:]

            def meta_cb(act):
                fp = fm(act.detach())
                return functional_call(
                    cgate, virtual_cgate_params, (fp.detach(),))

            meta_out = functional_call(
                model, virtual_model_params,
                args=(outer_x,),
                kwargs={
                    "cerebellar_fn": meta_cb,
                    "cerebellar_input_block": cerebellar_input_block,
                    "cerebellar_inject_block": inject_after_block,
                },
            )
            meta_logits = meta_out[0]
            meta_ce = F.cross_entropy(
                meta_logits.reshape(-1, v),
                outer_y.reshape(-1),
                reduction='none',
            ).view(batch_size, seq_len - 1)
            outer_loss = (
                (meta_ce * outer_level_mask.unsqueeze(0)).sum()
                / (batch_size * outer_level_mask.sum())
            )

            # --- 7. Gate update (meta-gradient) ---
            opt_lg.zero_grad()
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(lgate.parameters(), 1.0)
            opt_lg.step()

            # --- 8. Model update (detached inner gradient) ---
            opt_main.zero_grad()
            for p, g in zip(main_params, grads_for_update):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            # --- 9. FM update ---
            fwd_loss = F.mse_loss(fwd_pred, target.detach())
            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == wake_steps - 1:
                vl = eval_model(model)
                mr = reward.mean().item()
                print(f"    w{step}: val={vl:.4f} "
                      f"rwd={mr:.3f} rl={rl_loss.item():.4f} "
                      f"ntp={ntp_loss.item():.4f} "
                      f"ll={gated_ll.item():.5f} "
                      f"gate={gate_w.mean().item():.3f} "
                      f"cg={cgate.injection_norm():.3f}")

        gs = get_gate_stats(lgate, model, fm)
        print(f"    Gate post-wake: mean={gs['mean']:.3f} "
              f"sparse(<0.1)={gs['sparse_01']:.3f}")

        del opt_main, opt_fwd, opt_lg

        # === SLEEP ===
        run_gen_distill_sleep(model, fm, cgate, cycle)
        del cgate; torch.cuda.empty_cache()

        # === REPOINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
        fm, fm_cos = train_fresh_fm(
            model, retrain_steps, fm_seed, f"WS_LG c{cycle+1}")

        gs_post = get_gate_stats(lgate, model, fm)
        print(f"    Gate post-repoint: mean={gs_post['mean']:.3f}")

        # === CHECKPOINT ===
        print(f"\n  WS_LG CHECKPOINT cycle {cycle + 1}")
        ckpt = full_checkpoint_eval(model, "WS_LG", cycle, fm=fm)
        ckpt["fm_cosine"] = fm_cos
        ckpt["gate_stats_post_wake"] = gs
        ckpt["gate_stats_post_repoint"] = gs_post
        wslg_checkpoints[(cycle + 1) * steps_per_cycle] = ckpt

    all_results["WS_LG"] = {"checkpoints": wslg_checkpoints}
    del model, fm, lgate; torch.cuda.empty_cache()

    # ==================================================================
    # CONDITION 2: WS (RL + injection, no local loss / no FOMAML)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION: WS ({n_cycles} cycles, RL wake)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    fm = make_fm()
    fm.load_state_dict(init_fm_state)

    ws_checkpoints = {}
    rl_baseline = [None]

    for cycle in range(n_cycles):
        print(f"\n  --- WS Cycle {cycle + 1}/{n_cycles} ---")
        print(f"  [WAKE] RL co-training ({wake_steps} steps)")

        cgate = CerebellarGate(n_embd).to(device)
        main_params = (list(model.parameters())
                       + list(cgate.parameters()))
        opt_main = torch.optim.AdamW(
            main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            rl_step = cycle * wake_steps + step

            # Generate
            model.eval(); fm.eval(); cgate.eval()
            seqs = gen_seqs_tensor[rl_indices[rl_step]].to(device)
            prefix = seqs[:, :prefix_len]
            true_suffix = seqs[:, prefix_len:]

            with torch.no_grad():
                generated = gen_suffix(
                    model, prefix, suffix_len, temperature,
                    fm, cgate, cerebellar_input_block,
                    inject_after_block)

            correct = (generated == true_suffix).float()
            reward = correct.mean(dim=1)
            if rl_baseline[0] is None:
                rl_baseline[0] = reward.mean().item()
            else:
                rl_baseline[0] = (0.95 * rl_baseline[0]
                                  + 0.05 * reward.mean().item())
            advantage = (reward - rl_baseline[0]).detach()

            # Score
            model.train(); fm.train(); cgate.train()
            scored_seq = torch.cat([prefix, generated], dim=1)
            fwd_cache = {}

            def ws_cb(act, _cache=fwd_cache):
                fp = fm(act.detach())
                _cache["pred"] = fp
                return cgate(fp.detach())

            logits, _, intermediates = model(
                scored_seq, return_intermediates=True,
                cerebellar_fn=ws_cb,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

            suf_logits = logits[
                :, prefix_len-1:prefix_len-1+suffix_len, :]
            log_probs = F.log_softmax(suf_logits, dim=-1)
            gen_lp = log_probs.gather(
                2, generated.unsqueeze(-1)).squeeze(-1)
            rl_loss = -(advantage.unsqueeze(1) * gen_lp).mean()

            # Sparse NTP
            ntp_step = pretrain_steps + cycle * steps_per_cycle + step
            ntp_x = torch.stack(
                [train_data[i:i + seq_len]
                 for i in train_indices[ntp_step]]).to(device)
            ntp_y = torch.stack(
                [train_data[i + 1:i + seq_len + 1]
                 for i in train_indices[ntp_step]]).to(device)
            ntp_logits, _ = model(ntp_x)
            ntp_per_pos = F.cross_entropy(
                ntp_logits.view(-1, v), ntp_y.view(-1),
                reduction='none')
            ntp_mask = (torch.rand(ntp_per_pos.shape[0],
                                   device=device)
                        >= ntp_mask_rate).float()
            ntp_loss = ((ntp_per_pos * ntp_mask).sum()
                        / ntp_mask.sum().clamp(min=1))

            total_loss = (rl_loss_scale * rl_loss
                          + lambda_ntp * ntp_loss)

            # FM update target
            fwd_pred = fwd_cache["pred"]
            target = intermediates[predict_to].detach()
            fwd_loss = F.mse_loss(fwd_pred, target)

            opt_main.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == wake_steps - 1:
                vl = eval_model(model)
                mr = reward.mean().item()
                print(f"    w{step}: val={vl:.4f} rwd={mr:.3f} "
                      f"rl={rl_loss.item():.4f} "
                      f"ntp={ntp_loss.item():.4f} "
                      f"cg={cgate.injection_norm():.3f}")

        del opt_main, opt_fwd

        # === SLEEP ===
        run_gen_distill_sleep(model, fm, cgate, cycle)
        del cgate; torch.cuda.empty_cache()

        # === REPOINT ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [REPOINT] Fresh FM (seed={fm_seed})")
        fm, fm_cos = train_fresh_fm(
            model, retrain_steps, fm_seed, f"WS c{cycle+1}")

        # === CHECKPOINT ===
        print(f"\n  WS CHECKPOINT cycle {cycle + 1}")
        ckpt = full_checkpoint_eval(model, "WS", cycle, fm=fm)
        ckpt["fm_cosine"] = fm_cos
        ws_checkpoints[(cycle + 1) * steps_per_cycle] = ckpt

    all_results["WS"] = {"checkpoints": ws_checkpoints}
    del model, fm; torch.cuda.empty_cache()

    # ==================================================================
    # CONDITION 3: OL (continuous NTP)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION: OL ({total_finetune_steps} continuous NTP)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(pretrained_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ol_checkpoints = {}

    for step in range(total_finetune_steps):
        model.train()
        ntp_step = pretrain_steps + step
        x = torch.stack([train_data[i:i + seq_len]
                         for i in train_indices[ntp_step]]).to(device)
        y = torch.stack([train_data[i + 1:i + seq_len + 1]
                         for i in train_indices[ntp_step]]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == total_finetune_steps - 1:
            vl = eval_model(model)
            print(f"    step {step}: val={vl:.4f}")

        if (step + 1) in checkpoint_steps:
            ci = checkpoint_steps.index(step + 1)
            print(f"\n  OL CHECKPOINT cycle {ci + 1}")
            fm_seed = seed + 100 * (ci + 1)
            fm_ol, fm_cos = train_fresh_fm(
                model, retrain_steps, fm_seed, f"OL c{ci+1}")
            ckpt = full_checkpoint_eval(
                model, "OL", ci, fm=fm_ol)
            ckpt["fm_cosine"] = fm_cos
            ol_checkpoints[step + 1] = ckpt
            del fm_ol; torch.cuda.empty_cache()

    all_results["OL"] = {"checkpoints": ol_checkpoints}
    del model, opt; torch.cuda.empty_cache()

    # ==================================================================
    # Save
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_fomaml_rl_ratchet/{key}"
    os.makedirs(save_dir, exist_ok=True)
    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D",
            "predict_from": predict_from, "predict_to": predict_to,
            "pretrain_steps": pretrain_steps,
            "n_cycles": n_cycles, "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "lr": lr, "fwd_lr": fwd_lr, "gate_lr": gate_lr,
            "lambda_local": lambda_local, "lambda_ntp": lambda_ntp,
            "rl_loss_scale": rl_loss_scale, "ntp_mask_rate": ntp_mask_rate,
            "outer_min_level": outer_min_level,
            "n_outer_positions": n_outer_positions,
            "distill_alpha": distill_alpha, "distill_lr": distill_lr,
            "lg_hidden": lg_hidden, "temperature": temperature,
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
    print(f"SUMMARY: RHM FOMAML RL Ratchet")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s}")
    print(f"  Uniform baseline: {uniform:.4f}")
    print(f"{'='*80}")

    conds = ["WS_LG", "WS", "OL"]

    print(f"\n  === Val loss ===")
    header = f"  {'Cycle':>6s}" + "".join(
        f"  {c:>12s}" for c in conds)
    print(header)
    for ci, cs in enumerate(checkpoint_steps):
        parts = [f"{ci+1:6d}"]
        for c in conds:
            ckpt = (all_results.get(c, {})
                    .get("checkpoints", {}).get(cs, {}))
            parts.append(f"{ckpt.get('val_loss', 0):12.4f}")
        print("  " + "  ".join(parts))

    final_cs = checkpoint_steps[-1]

    print(f"\n  === Generation accuracy (standalone) ===")
    header = f"  {'Cond':>12s}  {'overall':>7s}"
    for lvl in range(max_suffix_level + 1):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    for c in conds:
        ckpt = (all_results.get(c, {})
                .get("checkpoints", {}).get(final_cs, {}))
        ga = ckpt.get("generation_accuracy", {})
        row = f"  {c:>12s}  {ga.get('overall', 0):7.3f}"
        for lvl in range(max_suffix_level + 1):
            row += f"  {ga.get(f'L{lvl}', 0):7.3f}"
        print(row)

    print(f"\n  === FM cosine ===")
    for c in conds:
        parts = [f"  {c:>12s}"]
        for cs in checkpoint_steps:
            ckpt = (all_results.get(c, {})
                    .get("checkpoints", {}).get(cs, {}))
            parts.append(f"{ckpt.get('fm_cosine', 0):.4f}")
        print("  ".join(parts))

    if "WS_LG" in all_results:
        print(f"\n  === FOMAML gate (WS_LG) ===")
        for ci, cs in enumerate(checkpoint_steps):
            ckpt = all_results["WS_LG"]["checkpoints"].get(cs, {})
            gs_w = ckpt.get("gate_stats_post_wake", {})
            gs_r = ckpt.get("gate_stats_post_repoint", {})
            print(f"    c{ci+1} wake: mean={gs_w.get('mean',0):.3f} "
                  f"sparse={gs_w.get('sparse_01',0):.3f} | "
                  f"repoint: mean={gs_r.get('mean',0):.3f}")

    print(f"\n  === Self-knowledge (final) ===")
    for c in conds:
        ckpt = (all_results.get(c, {})
                .get("checkpoints", {}).get(final_cs, {}))
        sk = ckpt.get("self_knowledge", {})
        if sk:
            print(f"  {c}: " + " | ".join(
                f"{k}={v_:.3f}" for k, v_ in sk.items()))

    print(f"\n  === Robustness eps=1.0 (final) ===")
    for c in conds:
        ckpt = (all_results.get(c, {})
                .get("checkpoints", {}).get(final_cs, {}))
        rob = ckpt.get("robustness", {}).get("1.0", None)
        if rob is not None:
            print(f"  {c}: {rob:+.4f}")

    print(f"\n  === Per-layer L3 eta² (final) ===")
    for c in conds:
        ckpt = (all_results.get(c, {})
                .get("checkpoints", {}).get(final_cs, {}))
        ple = ckpt.get("per_layer_eta2", {})
        if not ple:
            continue
        parts = [f"  {c:>12s}"]
        for li in range(n_layer):
            lk = f"post_block{li}"
            eta2 = (ple.get(lk, {})
                    .get("level_3", {}).get("feature_eta2", 0))
            parts.append(f"b{li}={eta2:.4f}")
        print("  ".join(parts))

    print(f"\n  Saved to {save_dir}/results.json")
    return result


@app.local_entrypoint()
def main():
    result = rhm_fomaml_rl_ratchet.remote()
    print("\nRHM FOMAML RL ratchet complete.")
