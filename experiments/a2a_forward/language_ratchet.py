"""Language multi-cycle gated ratchet experiment.

Tests whether the WS_UG_uniform architecture (unified gate for injection +
uniform local loss + distillation ratchet) produces compounding val loss
improvement on language modeling, as demonstrated on MNIST.

Three conditions, all compute-matched on main-model gradient steps:

1. WS_UG_uniform: Wake-sleep with unified gate (NTP-trained) + uniform local loss
   Per cycle: wake_steps co-training + distill_steps distillation sleep
   Unified gate persists across cycles. FM reinitialized each cycle.

2. CL: Continuous closed-loop (injection only, no distillation, no local loss)
   Same total main-model steps. Uses CerebellarGate (linear, zero-init).

3. OL: Open-loop baseline.
   Same total main-model steps. Standard NTP training.

Architecture: 4L/4H/256D GPT (~28.9M params), 2L/1H/64D causal FM (~660K params),
post_block0 -> post_block3, injection after block 1.

Key question: does the FM local loss add anything beyond what NTP already provides?
In MNIST, the classification loss only supervises [CLS], so the local loss provides
qualitatively new supervision (dense intermediate). In language, NTP supervises
every position -- the local loss must provide complementary information about
*intermediate computation quality* (not just output quality) to matter.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def a2a_language_ratchet(
    # Data
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    batch_size: int = 64,
    # GPT architecture
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    # FM architecture
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    # Wake-sleep protocol
    n_cycles: int = 4,
    wake_steps: int = 2000,
    distill_steps: int = 500,
    retrain_steps: int = 2000,
    # Training params
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    lambda_local: float = 1.0,
    ug_hidden: int = 128,
    # Eval params
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    probe_batches: int = 40,
    probe_steps: int = 500,
    seed: int = 42,
):
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel, CerebellarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", ""))

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    print(f"LANGUAGE GATED RATCHET on {device}")
    print(f"  GPT: {n_layer}L {n_head}H {n_embd}D")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, causal")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  Checkpoints at: {checkpoint_steps}")
    print(f"  lambda_local={lambda_local}, ug_hidden={ug_hidden}")

    # -----------------------------------------------------------------
    # Unified Gate (same architecture as MNIST, scaled for 256-d)
    # -----------------------------------------------------------------
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
            print(f"UnifiedGate: {n_params/1e3:.1f}K params "
                  f"(input={2*d_model}, hidden={d_hidden}, output={d_model})")

        def forward(self, activations, fwd_pred):
            gate_w = torch.sigmoid(self.gate_net(
                torch.cat([activations, fwd_pred], dim=-1)))
            injection = gate_w * self.projection(fwd_pred)
            return injection, gate_w

        def injection_norm(self):
            return self.projection.weight.norm().item()

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    print("\nLoading data...")
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_tokens)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    print(f"  Loaded {len(data):,} tokens (vocab_size={vocab_size})")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # -----------------------------------------------------------------
    # Pre-generate batch indices (shared across all conditions)
    # -----------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    repoint_gen = torch.Generator().manual_seed(seed + 3)

    max_repoint_steps = n_cycles * retrain_steps

    train_indices = [
        torch.randint(len(train_data) - block_size - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(total_main_steps)
    ]
    n_evals = total_main_steps // eval_interval + 10
    eval_indices = [
        [torch.randint(len(val_data) - block_size - 1, (batch_size,),
                        generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]
    probe_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=probe_gen)
        for _ in range(probe_batches)
    ]
    repoint_indices = [
        torch.randint(len(train_data) - block_size - 1, (batch_size,),
                       generator=repoint_gen)
        for _ in range(max_repoint_steps)
    ]

    def make_batch(indices, split_data):
        x = torch.stack([split_data[i:i + block_size] for i in indices])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------
    def make_gpt():
        return GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)

    # -----------------------------------------------------------------
    # Initialize shared weights
    # -----------------------------------------------------------------
    torch.manual_seed(seed)
    init_model = make_gpt()
    init_fm = make_fm()
    init_model_state = {k: v.cpu().clone()
                        for k, v in init_model.state_dict().items()}
    init_fm_state = {k: v.cpu().clone()
                     for k, v in init_fm.state_dict().items()}
    del init_model, init_fm
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =================================================================
    # SHARED HELPERS
    # =================================================================
    def eval_model(model, eval_idx=0):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                x, y = make_batch(eidx, val_data)
                _, loss = model(x, y)
                total_loss += loss.item()
        return total_loss / n_eval_batches

    def eval_model_with_ug(model, fm, ugate, eval_idx=0):
        model.eval(); fm.eval(); ugate.eval()
        total_loss = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                x, y = make_batch(eidx, val_data)

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

    def eval_model_with_cgate(model, fm, cgate, eval_idx=0):
        model.eval(); fm.eval(); cgate.eval()
        total_loss = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                x, y = make_batch(eidx, val_data)

                def cb(act):
                    return cgate(fm(act))

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
                    pidx = probe_indices[bi]
                    x, y = make_batch(pidx, val_data)
                    _, base_loss = model(x, y)
                    torch.manual_seed(seed + bi + 5000)
                    noise = torch.randn(
                        batch_size, block_size, n_embd, device=device) * eps

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
        print(f"  [{label:>20s}] "
              + " | ".join(f"eps={e}: d={results[str(e)]:+.4f}"
                           for e in eps_list))
        return results

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def self_knowledge_probes(model, fm, label):
        model.eval(); fm.eval()
        acts_by_layer = {k: [] for k in layer_keys}
        residuals = []
        with torch.no_grad():
            for pidx in probe_indices:
                x, y = make_batch(pidx, val_data)
                _, _, vi = model(x, y, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                residuals.append((tgt - pred).reshape(-1, n_embd).cpu())
                for k in layer_keys:
                    acts_by_layer[k].append(vi[k].reshape(-1, n_embd).cpu())

        all_res = torch.cat(residuals)
        for k in layer_keys:
            acts_by_layer[k] = torch.cat(acts_by_layer[k])

        n_total = all_res.shape[0]
        n_train = int(0.8 * n_total)
        perm = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed))
        tr, te = perm[:n_train], perm[n_train:]

        r2_by_layer = {}
        for lk in layer_keys:
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
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0
            r2_by_layer[lk] = r2
            del probe, X_tr, Y_tr, X_te, Y_te
            torch.cuda.empty_cache()

        print(f"  [{label}] SK: "
              + " | ".join(f"{k}={v:.3f}" for k, v in r2_by_layer.items()))
        return r2_by_layer

    def compute_residual_stats(model, fm, label):
        model.eval(); fm.eval()
        all_res = []
        with torch.no_grad():
            for pidx in probe_indices:
                x, y = make_batch(pidx, val_data)
                _, _, vi = model(x, y, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                res = tgt - pred
                all_res.append(res.reshape(-1, n_embd).cpu())

        all_res = torch.cat(all_res)
        R = all_res.numpy().astype(np.float64)
        R_c = R - R.mean(axis=0, keepdims=True)
        cov = (R_c.T @ R_c) / (R_c.shape[0] - 1)
        evals = np.linalg.eigvalsh(cov)[::-1]
        evals = np.maximum(evals, 0)
        total_var = evals.sum()
        fracs = evals / total_var if total_var > 0 else evals
        fracs_pos = fracs[fracs > 1e-12]
        eff_rank = float(np.exp(-np.sum(fracs_pos * np.log(fracs_pos))))

        # FM cosine
        cos_total = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                x, y = make_batch(eidx, val_data)
                _, _, vi = model(x, y, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                cos_total += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
        fwd_cosine = cos_total / n_eval_batches

        stats = {
            "eff_rank": eff_rank,
            "top1_pc_frac": float(fracs[0]),
            "top5_pc_frac": float(fracs[:5].sum()),
            "mean_res_norm": float(all_res.norm(dim=-1).mean()),
            "fwd_cosine": fwd_cosine,
        }

        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"res_norm={stats['mean_res_norm']:.3f}, "
              f"fwd_cos={fwd_cosine:.4f}")
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
            ridx = repoint_indices[step % len(repoint_indices)]
            x, _ = make_batch(ridx, train_data)
            with torch.no_grad():
                _, _, vi = model(x, return_intermediates=True)
                source = vi[predict_from]
                target = vi[predict_to]
            pred = fm(source)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()

        fm.eval()
        cos_total = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                x, y = make_batch(eidx, val_data)
                _, _, vi = model(x, y, return_intermediates=True)
                p = fm(vi[predict_from])
                cos_total += F.cosine_similarity(
                    p, vi[predict_to], dim=-1).mean().item()
        cos = cos_total / n_eval_batches
        print(f"  [{label}] Fresh FM cosine: {cos:.4f}")

        for p in model.parameters():
            p.requires_grad = True
        return fm, cos

    def get_ug_gate_stats(ugate, model, fwd_model):
        ugate.eval(); model.eval(); fwd_model.eval()
        all_gw = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                pidx = probe_indices[bi]
                x, y = make_batch(pidx, val_data)
                _, _, vi = model(x, y, return_intermediates=True)
                src = vi[predict_from]
                pred = fwd_model(src)
                _, gw = ugate(src, pred)
                all_gw.append(gw.mean(dim=1).cpu())
        gw_cat = torch.cat(all_gw, dim=0)
        dim_mean = gw_cat.mean(dim=0)
        return {
            "overall_mean": float(gw_cat.mean()),
            "overall_std": float(gw_cat.std()),
            "dim_mean_min": float(dim_mean.min()),
            "dim_mean_max": float(dim_mean.max()),
            "dim_mean_std": float(dim_mean.std()),
            "sparsity_below_0.1": float((dim_mean < 0.1).float().mean()),
            "sparsity_below_0.2": float((dim_mean < 0.2).float().mean()),
            "dim_mean": dim_mean.tolist(),
        }

    # =================================================================
    # CONDITION 1: WS_UG_uniform (Wake-Sleep with Unified Gate)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 1: WS_UG_uniform ({n_cycles} cycles)")
    print(f"{'='*60}")

    wsug_model = make_gpt()
    wsug_model.load_state_dict(init_model_state)
    wsug_fm = make_fm()
    wsug_fm.load_state_dict(init_fm_state)
    wsug_ugate = UnifiedGate(n_embd, ug_hidden).to(device)

    wsug_checkpoints = {}
    wsug_history = {
        "train_loss": [], "val_loss": [], "gate_stats": [],
    }
    wsug_global_step = 0

    for cycle in range(n_cycles):
        print(f"\n  --- WS_UG Cycle {cycle + 1}/{n_cycles} ---")

        # === WAKE: co-training with unified gate + uniform local loss ===
        print(f"  [WAKE] UG co-training, uniform local loss "
              f"({wake_steps} steps)")
        ug_main_params = (list(wsug_model.parameters())
                          + list(wsug_ugate.parameters()))
        opt_main = torch.optim.AdamW(
            ug_main_params, lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(
            wsug_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(wake_steps):
            wsug_model.train(); wsug_fm.train(); wsug_ugate.train()

            idx = train_indices[wsug_global_step]
            x, y = make_batch(idx, train_data)

            fwd_pred_cache = {}

            def ug_cerebellar_fn(act, _cache=fwd_pred_cache):
                fp = wsug_fm(act.detach())
                _cache["pred"] = fp
                inj, gw = wsug_ugate(act.detach(), fp.detach())
                _cache["gate_w"] = gw
                return inj

            _, ntp_loss, intermediates = wsug_model(
                x, y, return_intermediates=True,
                cerebellar_fn=ug_cerebellar_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block,
            )

            fwd_pred = fwd_pred_cache["pred"]
            target = intermediates[predict_to]
            fwd_loss = F.mse_loss(fwd_pred, target.detach())

            gate_w = fwd_pred_cache["gate_w"]
            r = target - fwd_pred.detach()
            gw_scalar = gate_w.detach().mean()
            local_loss = (gw_scalar * r ** 2).mean()
            L_total = ntp_loss + lambda_local * local_loss

            opt_main.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(ug_main_params, 1.0)
            opt_main.step()

            opt_fwd.zero_grad()
            fwd_loss.backward()
            torch.nn.utils.clip_grad_norm_(wsug_fm.parameters(), 1.0)
            opt_fwd.step()

            if step % eval_interval == 0 or step == wake_steps - 1:
                v_loss = eval_model(wsug_model)
                wsug_history["train_loss"].append(
                    (wsug_global_step, ntp_loss.item()))
                wsug_history["val_loss"].append((wsug_global_step, v_loss))

                dep_loss = eval_model_with_ug(
                    wsug_model, wsug_fm, wsug_ugate)
                dep = v_loss - dep_loss
                print(f"    step {wsug_global_step:5d} (wake {step}): "
                      f"val={v_loss:.4f} dep={dep:+.4f} "
                      f"proj={wsug_ugate.injection_norm():.3f}"
                      f" gw={gate_w.mean().item():.3f}"
                      f" ll={local_loss.item():.5f}")

            wsug_global_step += 1

        gs = get_ug_gate_stats(wsug_ugate, wsug_model, wsug_fm)
        wsug_history["gate_stats"].append(
            (wsug_global_step, f"post_wake_c{cycle+1}", gs))
        print(f"    Gate: mean={gs['overall_mean']:.3f} "
              f"std={gs['overall_std']:.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.3f}")

        wake_dep_loss = eval_model_with_ug(
            wsug_model, wsug_fm, wsug_ugate)
        wake_noinj_loss = eval_model(wsug_model)
        wake_dep_gap = wake_noinj_loss - wake_dep_loss
        print(f"    Post-wake dep gap: {wake_dep_gap:+.4f}")

        # === SLEEP: Distillation ===
        print(f"  [SLEEP] Distillation ({distill_steps} steps)")

        teacher = make_gpt()
        teacher.load_state_dict(wsug_model.state_dict())
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_fm = make_fm()
        teacher_fm.load_state_dict(wsug_fm.state_dict())
        teacher_fm.eval()
        for p in teacher_fm.parameters():
            p.requires_grad = False
        teacher_ugate = UnifiedGate(n_embd, ug_hidden).to(device)
        teacher_ugate.load_state_dict(wsug_ugate.state_dict())
        teacher_ugate.eval()
        for p in teacher_ugate.parameters():
            p.requires_grad = False

        student = make_gpt()
        student.load_state_dict(wsug_model.state_dict())
        opt_student = torch.optim.AdamW(
            student.parameters(), lr=distill_lr, weight_decay=0.01)

        for step in range(distill_steps):
            student.train()
            idx = train_indices[wsug_global_step]
            x, y = make_batch(idx, train_data)

            with torch.no_grad():
                def t_ug_cb(act):
                    fp = teacher_fm(act)
                    inj, _ = teacher_ugate(act, fp)
                    return inj
                teacher_logits, _, _ = teacher(
                    x, y, return_intermediates=True,
                    cerebellar_fn=t_ug_cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )

            student_logits, ce_loss = student(x, y)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss

            opt_student.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_student.step()

            if step % eval_interval == 0 or step == distill_steps - 1:
                v_loss = eval_model(student)
                wsug_history["val_loss"].append((wsug_global_step, v_loss))
                print(f"    step {wsug_global_step:5d} (sleep {step}): "
                      f"val={v_loss:.4f}")

            wsug_global_step += 1

        wsug_model.load_state_dict(student.state_dict())
        del teacher, teacher_fm, teacher_ugate, student, opt_student
        del opt_main, opt_fwd
        torch.cuda.empty_cache()

        # === RE-POINT: Fresh FM ===
        fm_seed = seed + 100 * (cycle + 1)
        print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, seed={fm_seed})")
        wsug_fm, fm_cos = train_fresh_fm(
            wsug_model, retrain_steps, fm_seed,
            f"WS_UG c{cycle+1}")

        # === CHECKPOINT EVALUATION ===
        cycle_step = (cycle + 1) * steps_per_cycle
        print(f"\n  WS_UG CHECKPOINT at step {cycle_step} "
              f"(end cycle {cycle + 1})")

        ug_loss = eval_model(wsug_model)
        ug_rob = measure_robustness(wsug_model, f"WS_UG c{cycle+1}")
        ug_sk = self_knowledge_probes(
            wsug_model, wsug_fm, f"WS_UG c{cycle+1}")
        ug_res = compute_residual_stats(
            wsug_model, wsug_fm, f"WS_UG c{cycle+1}")

        gs_post = get_ug_gate_stats(wsug_ugate, wsug_model, wsug_fm)
        wsug_history["gate_stats"].append(
            (cycle_step, f"post_repoint_c{cycle+1}", gs_post))
        print(f"    Gate (post-repoint): mean={gs_post['overall_mean']:.3f} "
              f"std={gs_post['overall_std']:.3f} "
              f"sparse(<0.1)={gs_post['sparsity_below_0.1']:.3f}")

        wsug_checkpoints[cycle_step] = {
            "val_loss": ug_loss,
            "robustness": ug_rob,
            "self_knowledge": ug_sk,
            "residual_stats": ug_res,
            "gate_stats_post_wake": wsug_history["gate_stats"][-2][2],
            "gate_stats_post_repoint": gs_post,
            "wake_dep_gap": wake_dep_gap,
            "fm_cosine": fm_cos,
        }
        print(f"    loss={ug_loss:.4f}")

    # =================================================================
    # CONDITION 2: CL (Continuous Closed-Loop)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 2: CL CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")

    cl_model = make_gpt()
    cl_model.load_state_dict(init_model_state)
    cl_fm = make_fm()
    cl_fm.load_state_dict(init_fm_state)
    cl_cgate = CerebellarGate(n_embd).to(device)

    cl_main_params = list(cl_model.parameters()) + list(cl_cgate.parameters())
    cl_opt_main = torch.optim.AdamW(cl_main_params, lr=lr, weight_decay=0.01)
    cl_opt_fwd = torch.optim.AdamW(
        cl_fm.parameters(), lr=fwd_lr, weight_decay=0.01)

    cl_checkpoints = {}
    cl_history = {"train_loss": [], "val_loss": []}

    for step in range(total_main_steps):
        cl_model.train(); cl_fm.train(); cl_cgate.train()

        idx = train_indices[step]
        x, y = make_batch(idx, train_data)

        fwd_pred_cache = {}

        def cl_cb(act, _cache=fwd_pred_cache):
            fp = cl_fm(act.detach())
            _cache["pred"] = fp
            return cl_cgate(fp.detach())

        _, ntp_loss, intermediates = cl_model(
            x, y, return_intermediates=True,
            cerebellar_fn=cl_cb,
            cerebellar_input_block=cerebellar_input_block,
            cerebellar_inject_block=inject_after_block,
        )
        target = intermediates[predict_to].detach()
        fwd_loss = F.mse_loss(fwd_pred_cache["pred"], target)

        cl_opt_main.zero_grad()
        ntp_loss.backward()
        torch.nn.utils.clip_grad_norm_(cl_main_params, 1.0)
        cl_opt_main.step()

        cl_opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(cl_fm.parameters(), 1.0)
        cl_opt_fwd.step()

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss = eval_model(cl_model)
            cl_history["train_loss"].append((step, ntp_loss.item()))
            cl_history["val_loss"].append((step, v_loss))

            inj_loss = eval_model_with_cgate(cl_model, cl_fm, cl_cgate)
            dep = v_loss - inj_loss
            print(f"  step {step:5d}: val={v_loss:.4f} "
                  f"dep={dep:+.4f} gate={cl_cgate.injection_norm():.3f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            print(f"\n  CL CHECKPOINT at step {ckpt_step}")

            cl_l = eval_model(cl_model)

            # Train a fresh FM for fair SK comparison
            cl_fresh_fm, cl_fm_cos = train_fresh_fm(
                cl_model, retrain_steps, seed + 800 + ckpt_step,
                f"CL s{ckpt_step}")

            cl_r = measure_robustness(cl_model, f"CL step {ckpt_step}")
            cl_s = self_knowledge_probes(
                cl_model, cl_fresh_fm, f"CL s{ckpt_step}")
            cl_res = compute_residual_stats(
                cl_model, cl_fresh_fm, f"CL s{ckpt_step}")

            inj_loss = eval_model_with_cgate(cl_model, cl_fm, cl_cgate)
            dep = cl_l - inj_loss

            cl_checkpoints[ckpt_step] = {
                "val_loss": cl_l,
                "robustness": cl_r,
                "self_knowledge": cl_s,
                "residual_stats": cl_res,
                "dependency_gap": dep,
                "fm_cosine": cl_fm_cos,
            }
            print(f"    loss={cl_l:.4f} dep={dep:+.4f}")

            del cl_fresh_fm
            torch.cuda.empty_cache()

    del cl_opt_main, cl_opt_fwd
    torch.cuda.empty_cache()

    # =================================================================
    # CONDITION 3: OL (Open-Loop)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 3: OL CONTINUOUS ({total_main_steps} steps)")
    print(f"{'='*60}")

    ol_model = make_gpt()
    ol_model.load_state_dict(init_model_state)
    ol_opt = torch.optim.AdamW(ol_model.parameters(), lr=lr, weight_decay=0.01)

    ol_checkpoints = {}
    ol_history = {"train_loss": [], "val_loss": []}

    for step in range(total_main_steps):
        ol_model.train()
        idx = train_indices[step]
        x, y = make_batch(idx, train_data)

        _, ntp_loss = ol_model(x, y)
        ol_opt.zero_grad()
        ntp_loss.backward()
        torch.nn.utils.clip_grad_norm_(ol_model.parameters(), 1.0)
        ol_opt.step()

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss = eval_model(ol_model)
            ol_history["train_loss"].append((step, ntp_loss.item()))
            ol_history["val_loss"].append((step, v_loss))
            if step % 1000 == 0:
                print(f"  step {step:5d}: val={v_loss:.4f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            print(f"\n  OL CHECKPOINT at step {ckpt_step}")

            ol_l = eval_model(ol_model)

            ol_fresh_fm, ol_fm_cos = train_fresh_fm(
                ol_model, retrain_steps, seed + 700 + ckpt_step,
                f"OL s{ckpt_step}")

            ol_r = measure_robustness(ol_model, f"OL step {ckpt_step}")
            ol_s = self_knowledge_probes(
                ol_model, ol_fresh_fm, f"OL s{ckpt_step}")
            ol_res = compute_residual_stats(
                ol_model, ol_fresh_fm, f"OL s{ckpt_step}")

            ol_checkpoints[ckpt_step] = {
                "val_loss": ol_l,
                "robustness": ol_r,
                "self_knowledge": ol_s,
                "residual_stats": ol_res,
                "fm_cosine": ol_fm_cos,
            }
            print(f"    loss={ol_l:.4f}")

            del ol_fresh_fm
            torch.cuda.empty_cache()

    del ol_opt
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"gpt_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/language_ratchet"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    torch.save(wsug_model.state_dict(),
               os.path.join(save_root, "wsug_model.pt"))
    torch.save(wsug_fm.state_dict(),
               os.path.join(save_root, "wsug_fm.pt"))
    torch.save(wsug_ugate.state_dict(),
               os.path.join(save_root, "wsug_ugate.pt"))
    torch.save(cl_model.state_dict(),
               os.path.join(save_root, "cl_model.pt"))
    torch.save(cl_fm.state_dict(),
               os.path.join(save_root, "cl_fm.pt"))
    torch.save(cl_cgate.state_dict(),
               os.path.join(save_root, "cl_cgate.pt"))
    torch.save(ol_model.state_dict(),
               os.path.join(save_root, "ol_model.pt"))

    result = {
        "config": {
            "n_cycles": n_cycles,
            "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "checkpoint_steps": checkpoint_steps,
            "n_tokens": n_tokens,
            "block_size": block_size,
            "batch_size": batch_size,
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "lr": lr, "fwd_lr": fwd_lr,
            "distill_lr": distill_lr, "distill_alpha": distill_alpha,
            "lambda_local": lambda_local, "ug_hidden": ug_hidden,
            "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "fwd_d_head": fwd_d_head,
            "fwd_n_head": fwd_n_head, "fwd_mlp_mult": fwd_mlp_mult,
        },
        "ws_ug_uniform": {
            "checkpoints": wsug_checkpoints,
            "history": wsug_history,
        },
        "cl": {
            "checkpoints": cl_checkpoints,
            "history": cl_history,
        },
        "ol": {
            "checkpoints": ol_checkpoints,
            "history": ol_history,
        },
    }

    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    print(f"\n  {n_cycles} cycles x ({wake_steps} wake + {distill_steps} sleep)"
          f" = {total_main_steps} main-model steps")

    print(f"\n  === Val loss trajectory ===")
    print(f"  {'Step':>6s} | {'WS_UG_uni':>10s} | {'CL':>10s} | {'OL':>10s}")
    for cs in checkpoint_steps:
        wsug_c = wsug_checkpoints[cs]
        cl_c = cl_checkpoints[cs]
        ol_c = ol_checkpoints[cs]
        print(f"  {cs:6d} | {wsug_c['val_loss']:10.4f} "
              f"| {cl_c['val_loss']:10.4f} "
              f"| {ol_c['val_loss']:10.4f}")

    print(f"\n  === Robustness (delta-loss at eps=1.0) ===")
    print(f"  {'Step':>6s} | {'WS_UG_uni':>10s} {'CL':>10s} {'OL':>10s}")
    for cs in checkpoint_steps:
        vals = [wsug_checkpoints[cs]["robustness"]["1.0"],
                cl_checkpoints[cs]["robustness"]["1.0"],
                ol_checkpoints[cs]["robustness"]["1.0"]]
        print(f"  {cs:6d} | " +
              " ".join(f"{v:+10.4f}" for v in vals))

    print(f"\n  === Self-knowledge (R^2 at post_block0) ===")
    print(f"  {'Step':>6s} | {'WS_UG_uni':>10s} {'CL':>10s} {'OL':>10s}")
    for cs in checkpoint_steps:
        vals = [wsug_checkpoints[cs].get("self_knowledge", {}).get("post_block0", 0),
                cl_checkpoints[cs].get("self_knowledge", {}).get("post_block0", 0),
                ol_checkpoints[cs].get("self_knowledge", {}).get("post_block0", 0)]
        print(f"  {cs:6d} | " +
              " ".join(f"{v:10.3f}" for v in vals))

    print(f"\n  === Gate dynamics (WS_UG_uniform) ===")
    for _, label, gs in wsug_history["gate_stats"]:
        print(f"    {label:25s}: mean={gs['overall_mean']:.3f} "
              f"std={gs['overall_std']:.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.3f}")

    print(f"\n  === Residual structure ===")
    print(f"  {'Step':>6s} | {'WS_UG eff_rank':>15s} | {'CL eff_rank':>15s} "
          f"| {'OL eff_rank':>15s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [wsug_checkpoints, cl_checkpoints, ol_checkpoints]:
            rs = ckpts[cs].get("residual_stats", {})
            er = rs.get("eff_rank", 0)
            cos = rs.get("fwd_cosine", 0)
            row.append(f"{er:5.1f} cos={cos:.3f}")
        print(f"  {cs:6d} | " + " | ".join(row))

    # Val loss improvement
    final_cs = checkpoint_steps[-1]
    wsug_final = wsug_checkpoints[final_cs]["val_loss"]
    cl_final = cl_checkpoints[final_cs]["val_loss"]
    ol_final = ol_checkpoints[final_cs]["val_loss"]
    print(f"\n  Final val loss: WS_UG_uniform={wsug_final:.4f} "
          f"CL={cl_final:.4f} OL={ol_final:.4f}")
    if ol_final > 0:
        wsug_pct = (ol_final - wsug_final) / ol_final * 100
        cl_pct = (ol_final - cl_final) / ol_final * 100
        print(f"  WS_UG vs OL: {wsug_pct:+.1f}%")
        print(f"  CL vs OL:    {cl_pct:+.1f}%")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    n_cycles: int = 4,
    wake_steps: int = 2000,
    distill_steps: int = 500,
    retrain_steps: int = 2000,
    lambda_local: float = 1.0,
    ug_hidden: int = 128,
    seed: int = 42,
):
    result = a2a_language_ratchet.remote(
        n_tokens=n_tokens,
        n_cycles=n_cycles,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        lambda_local=lambda_local,
        ug_hidden=ug_hidden,
        seed=seed,
    )
    print("\nLanguage ratchet experiment complete.")

    config = result["config"]
    print(f"\n  {config['n_cycles']} cycles x "
          f"({config['wake_steps']} wake + {config['distill_steps']} sleep) = "
          f"{config['total_main_steps']} main-model steps")

    def get_ckpt(ckpts, step):
        return ckpts.get(step) or ckpts.get(str(step))

    final_step = config["checkpoint_steps"][-1]
    print(f"\n  Final ({final_step} steps):")
    for name, key in [("WS_UG_uniform", "ws_ug_uniform"),
                      ("CL", "cl"), ("OL", "ol")]:
        c = get_ckpt(result[key]["checkpoints"], final_step)
        if c:
            rob = c.get("robustness", {}).get("1.0", 0)
            print(f"    {name:>14s}: loss={c['val_loss']:.4f} "
                  f"rob(eps=1)={rob:+.4f}")
