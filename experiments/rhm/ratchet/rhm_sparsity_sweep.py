"""RHM sparsity sweep: process supervision as a function of NTP sparsity.

Tests whether FM local loss (process supervision) helps more when NTP
task supervision is sparser. At mask_rate=0, NTP supervises all positions;
at mask_rate=0.95, only ~5% of positions get NTP gradient. The FM local
loss provides intermediate supervision at ALL positions regardless of mask.

Conditions: {OL, LL} x {mask_rate=0.0, 0.5, 0.75, 0.9, 0.95}

OL: masked NTP only
LL: masked NTP + lambda * MSE(post_block5, sg(FM(post_block0)))

No injection, no gate, no distillation, no ratchet cycles. The FM co-trains
in the LL condition (learns to predict activations; gradient does not flow
from FM loss to main model). In the OL condition, a fresh FM is trained at
the end for comparison.

Prediction: at mask=0%, LL ~ OL (dense NTP makes the local loss redundant).
As mask rate increases, the LL-OL gap should widen monotonically.

Reproduction:
  cd experiments/
  modal run --detach -m rhm.ratchet.rhm_sparsity_sweep::rhm_sparsity_sweep
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

app = modal.App("rhm-sparsity-sweep", image=image)


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
        level_results[lvl] = {
            "loss": float(per_pos_loss[mask].mean()),
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
def rhm_sparsity_sweep(
    # DGP
    v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
    n_tokens: int = 20_000_000,
    # Model
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    # FM
    predict_from: str = "post_block0",
    predict_to: str = "post_block5",
    fwd_n_layer: int = 1, fwd_d_head: int = 16,
    fwd_n_head: int = 1, fwd_mlp_mult: float = 0.5,
    # Training
    n_steps: int = 15000,
    lr: float = 3e-4, fwd_lr: float = 1e-3,
    lambda_local: float = 1.0,
    batch_size: int = 64,
    # Sparsity sweep
    mask_rates: str = "0.0,0.5,0.75,0.9,0.95",
    # Eval
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    probe_batches: int = 30,
    probe_steps: int = 500,
    n_eval_sequences: int = 5000,
    fm_retrain_steps: int = 2000,
    perturb_block: int = 1,
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
    mask_rate_list = [float(x) for x in mask_rates.split(",")]

    print(f"RHM SPARSITY SWEEP on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  FM: {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D, "
          f"{predict_from}->{predict_to}")
    print(f"  {n_steps} steps, lambda_local={lambda_local}")
    print(f"  Mask rates: {mask_rate_list}")

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

    # ------------------------------------------------------------------
    # Pre-generate batch indices (shared across all conditions)
    # ------------------------------------------------------------------
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    fm_retrain_gen = torch.Generator().manual_seed(seed + 3)

    train_indices = [
        torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                       generator=train_gen)
        for _ in range(n_steps)
    ]
    n_evals = n_steps // eval_interval + 10
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
    fm_retrain_indices = [
        torch.randint(len(train_data) - seq_len - 1, (batch_size,),
                       generator=fm_retrain_gen)
        for _ in range(fm_retrain_steps)
    ]

    # Pre-generate NTP masks so OL and LL see identical masks per mask_rate.
    # Shape: (n_steps, batch_size * seq_len). Values in [0, 1).
    mask_rng = torch.Generator().manual_seed(seed + 4)
    mask_randoms = torch.rand(
        n_steps, batch_size * seq_len, generator=mask_rng)

    def make_batch(indices, split_data):
        x = torch.stack([split_data[i:i + seq_len] for i in indices])
        y = torch.stack([split_data[i + 1:i + seq_len + 1] for i in indices])
        return x.to(device), y.to(device)

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
    # Evaluation helpers
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
                        cerebellar_input_block=perturb_block,
                        cerebellar_inject_block=perturb_block,
                    )
                    deltas.append(pert_loss.item() - base_loss.item())
                results[str(eps)] = float(np.mean(deltas))
        print(f"  [{label:>25s}] rob: "
              + " | ".join(f"eps={e}:{results[str(e)]:+.4f}"
                           for e in eps_list))
        return results

    def compute_residual_stats(model, fm, label):
        model.eval(); fm.eval()
        eval_data_t = torch.from_numpy(
            eval_seqs.astype(np.int64)).to(device)
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
            ridx = fm_retrain_indices[step % len(fm_retrain_indices)]
            x, _ = make_batch(ridx, train_data)
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

    # ==================================================================
    # Run all conditions
    # ==================================================================
    all_results = {}

    for mask_rate in mask_rate_list:
        mr_label = f"mask{mask_rate:.2f}"

        # ==============================================================
        # OL condition: masked NTP only
        # ==============================================================
        ol_label = f"OL_{mr_label}"
        print(f"\n{'='*60}")
        print(f"  {ol_label} ({n_steps} steps, mask_rate={mask_rate})")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_model_state)
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01)

        ol_history = []
        eval_count = 0

        for step in range(n_steps):
            model.train()
            x, y = make_batch(train_indices[step], train_data)

            logits, _ = model(x)
            per_pos_loss = F.cross_entropy(
                logits.view(-1, v), y.view(-1), reduction='none')

            if mask_rate > 0:
                mask = (mask_randoms[step] >= mask_rate).float().to(device)
                n_active = mask.sum().clamp(min=1.0)
                ntp_loss = (per_pos_loss * mask).sum() / n_active
            else:
                ntp_loss = per_pos_loss.mean()

            opt.zero_grad()
            ntp_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % eval_interval == 0 or step == n_steps - 1:
                vl = eval_model(model, eval_count)
                eval_count += 1
                ol_history.append({"step": step, "val_loss": vl})
                if step % 1000 == 0 or step == n_steps - 1:
                    print(f"    step {step:5d}: val={vl:.4f} "
                          f"ntp={ntp_loss.item():.4f}")

        # Final eval
        print(f"\n  {ol_label} FINAL EVAL")
        ol_vl = eval_model(model)
        ol_pl = _eval_per_level(
            model, eval_seqs, seq_len, s, L, v, batch_size, device)
        ol_rob = measure_robustness(model, ol_label)

        ol_fm, ol_fm_cos = train_fresh_fm(
            model, fm_retrain_steps, seed + 800, ol_label)
        ol_res = compute_residual_stats(model, ol_fm, ol_label)
        ol_sk = self_knowledge_probes(model, ol_fm, ol_label)

        all_results[ol_label] = {
            "history": ol_history,
            "val_loss": ol_vl,
            "per_level": ol_pl,
            "robustness": ol_rob,
            "fm_cosine": ol_fm_cos,
            "residual_stats": ol_res,
            "self_knowledge": ol_sk,
        }

        del model, opt, ol_fm
        torch.cuda.empty_cache()

        # ==============================================================
        # LL condition: masked NTP + local loss (FM co-trained)
        # ==============================================================
        ll_label = f"LL_{mr_label}"
        print(f"\n{'='*60}")
        print(f"  {ll_label} ({n_steps} steps, mask_rate={mask_rate})")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_model_state)
        fm = make_fm()
        fm.load_state_dict(init_fm_state)

        opt_main = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01)
        opt_fm = torch.optim.AdamW(
            fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        ll_history = []
        eval_count = 0

        for step in range(n_steps):
            model.train(); fm.train()
            x, y = make_batch(train_indices[step], train_data)

            logits, _, intermediates = model(
                x, return_intermediates=True)

            # Masked NTP loss
            per_pos_loss = F.cross_entropy(
                logits.view(-1, v), y.view(-1), reduction='none')

            if mask_rate > 0:
                mask = (mask_randoms[step] >= mask_rate).float().to(device)
                n_active = mask.sum().clamp(min=1.0)
                ntp_loss = (per_pos_loss * mask).sum() / n_active
            else:
                ntp_loss = per_pos_loss.mean()

            # FM prediction (input detached from main model)
            source = intermediates[predict_from].detach()
            target = intermediates[predict_to]
            fm_pred = fm(source)

            # Local loss: gradient flows through target into main model;
            # FM prediction is stop-gradiented
            local_loss = F.mse_loss(target, fm_pred.detach())

            # Main model update
            total_loss = ntp_loss + lambda_local * local_loss
            opt_main.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_main.step()

            # FM update (separate graph: fm_pred -> fm params)
            fm_loss = F.mse_loss(fm_pred, target.detach())
            opt_fm.zero_grad()
            fm_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fm.step()

            if step % eval_interval == 0 or step == n_steps - 1:
                vl = eval_model(model, eval_count)
                # FM cosine
                model.eval(); fm.eval()
                with torch.no_grad():
                    cos_total = 0.0
                    for bi in range(n_eval_batches):
                        xv, yv = make_batch(
                            eval_indices[eval_count][bi], val_data)
                        _, _, vi = model(
                            xv, yv, return_intermediates=True)
                        p = fm(vi[predict_from])
                        cos_total += F.cosine_similarity(
                            p, vi[predict_to], dim=-1).mean().item()
                    fm_cos = cos_total / n_eval_batches
                eval_count += 1

                ll_history.append({
                    "step": step, "val_loss": vl,
                    "fm_cosine": fm_cos,
                    "local_loss": local_loss.item(),
                })
                if step % 1000 == 0 or step == n_steps - 1:
                    print(f"    step {step:5d}: val={vl:.4f} "
                          f"cos={fm_cos:.4f} ll={local_loss.item():.5f}")

        # Final eval
        print(f"\n  {ll_label} FINAL EVAL")
        ll_vl = eval_model(model)
        ll_pl = _eval_per_level(
            model, eval_seqs, seq_len, s, L, v, batch_size, device)
        ll_rob = measure_robustness(model, ll_label)
        ll_res = compute_residual_stats(model, fm, ll_label)
        ll_sk = self_knowledge_probes(model, fm, ll_label)

        all_results[ll_label] = {
            "history": ll_history,
            "val_loss": ll_vl,
            "per_level": ll_pl,
            "robustness": ll_rob,
            "fm_cosine": ll_history[-1]["fm_cosine"],
            "residual_stats": ll_res,
            "self_knowledge": ll_sk,
        }

        del model, fm, opt_main, opt_fm
        torch.cuda.empty_cache()

    # ==================================================================
    # Save results
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_sparsity_sweep/{key}"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}D",
            "predict_from": predict_from, "predict_to": predict_to,
            "n_steps": n_steps, "lr": lr, "fwd_lr": fwd_lr,
            "lambda_local": lambda_local, "batch_size": batch_size,
            "mask_rates": mask_rate_list,
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
    print(f"SUMMARY: RHM Sparsity Sweep")
    print(f"  {n_steps} steps, lambda_local={lambda_local}, "
          f"uniform baseline={uniform:.4f}")
    print(f"{'='*80}")

    # Val loss by mask rate
    print(f"\n  === Val loss (always unmasked eval) ===")
    print(f"  {'mask':>6s}  {'OL':>7s}  {'LL':>7s}  "
          f"{'gap':>7s}  {'gap%':>7s}")
    for mr in mask_rate_list:
        ml = f"mask{mr:.2f}"
        ol_vl = all_results[f"OL_{ml}"]["val_loss"]
        ll_vl = all_results[f"LL_{ml}"]["val_loss"]
        gap = ol_vl - ll_vl
        gap_pct = gap / ol_vl * 100 if ol_vl > 0 else 0
        print(f"  {mr:6.2f}  {ol_vl:7.4f}  {ll_vl:7.4f}  "
              f"{gap:+7.4f}  {gap_pct:+6.1f}%")

    # Per-level loss at each mask rate
    print(f"\n  === Per-level loss (OL | LL) at each mask rate ===")
    max_lvl = 0
    for r in all_results.values():
        for lvl in r["per_level"]["per_level"].keys():
            max_lvl = max(max_lvl, int(lvl))

    header = f"  {'Cond':>15s}  {'val':>7s}"
    for lvl in range(max_lvl + 1):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    print("  " + "-" * len(header))

    for mr in mask_rate_list:
        ml = f"mask{mr:.2f}"
        for prefix in ["OL", "LL"]:
            cond_key = f"{prefix}_{ml}"
            r = all_results[cond_key]
            pl = r["per_level"]["per_level"]
            row = f"  {cond_key:>15s}  {r['val_loss']:7.4f}"
            for lvl in range(max_lvl + 1):
                if lvl in pl:
                    row += f"  {pl[lvl]['loss']:7.4f}"
                else:
                    row += f"  {'--':>7s}"
            print(row)

    # Per-level delta (LL - OL)
    print(f"\n  === Per-level delta (LL - OL; negative = LL better) ===")
    header = f"  {'mask':>6s}"
    for lvl in range(max_lvl + 1):
        header += f"  {'L'+str(lvl):>7s}"
    print(header)
    for mr in mask_rate_list:
        ml = f"mask{mr:.2f}"
        ol_pl = all_results[f"OL_{ml}"]["per_level"]["per_level"]
        ll_pl = all_results[f"LL_{ml}"]["per_level"]["per_level"]
        row = f"  {mr:6.2f}"
        for lvl in range(max_lvl + 1):
            if lvl in ol_pl and lvl in ll_pl:
                delta = ll_pl[lvl]["loss"] - ol_pl[lvl]["loss"]
                row += f"  {delta:+7.4f}"
            else:
                row += f"  {'--':>7s}"
        print(row)

    # FM cosine and residual stats
    print(f"\n  === FM cosine and residual stats ===")
    print(f"  {'Condition':>15s}  {'cos':>7s}  {'norm':>7s}  "
          f"{'rank%':>6s}  {'top1%':>6s}")
    for mr in mask_rate_list:
        ml = f"mask{mr:.2f}"
        for prefix in ["OL", "LL"]:
            cond_key = f"{prefix}_{ml}"
            rs = all_results[cond_key].get("residual_stats", {})
            if rs:
                print(f"  {cond_key:>15s}  {rs['fwd_cosine']:7.4f}  "
                      f"{rs['res_norm']:7.2f}  {rs['eff_rank_pct']:6.1f}  "
                      f"{rs['top1_pct']:6.1f}")

    # Robustness
    print(f"\n  === Robustness (Dloss at perturbation) ===")
    print(f"  {'mask':>6s}  {'OL 0.5':>7s}  {'LL 0.5':>7s}  "
          f"{'OL 1.0':>7s}  {'LL 1.0':>7s}  "
          f"{'OL 2.0':>7s}  {'LL 2.0':>7s}")
    for mr in mask_rate_list:
        ml = f"mask{mr:.2f}"
        ol_rob = all_results[f"OL_{ml}"]["robustness"]
        ll_rob = all_results[f"LL_{ml}"]["robustness"]
        print(f"  {mr:6.2f}  "
              f"{ol_rob.get('0.5',0):+7.4f}  {ll_rob.get('0.5',0):+7.4f}  "
              f"{ol_rob.get('1.0',0):+7.4f}  {ll_rob.get('1.0',0):+7.4f}  "
              f"{ol_rob.get('2.0',0):+7.4f}  {ll_rob.get('2.0',0):+7.4f}")

    # Self-knowledge
    print(f"\n  === Self-knowledge probes (R^2) ===")
    print(f"  {'Condition':>15s}  {'post_block0':>12s}  "
          f"{'post_block'+str(n_layer-1):>12s}")
    for mr in mask_rate_list:
        ml = f"mask{mr:.2f}"
        for prefix in ["OL", "LL"]:
            cond_key = f"{prefix}_{ml}"
            sk = all_results[cond_key].get("self_knowledge", {})
            b0 = sk.get("post_block0", 0)
            bN = sk.get(f"post_block{n_layer-1}", 0)
            print(f"  {cond_key:>15s}  {b0:12.3f}  {bN:12.3f}")

    print(f"\n  Saved to {save_dir}/results.json")
    return result


@app.local_entrypoint()
def main(
    mask_rates: str = "0.0,0.5,0.75,0.9,0.95",
    n_steps: int = 15000,
    lambda_local: float = 1.0,
    m: int = 2,
    depth: int = 6,
    seed: int = 42,
):
    result = rhm_sparsity_sweep.remote(
        mask_rates=mask_rates,
        n_steps=n_steps,
        lambda_local=lambda_local,
        m=m,
        depth=depth,
        seed=seed,
    )
    print("\nRHM sparsity sweep complete.")

    config = result["config"]
    print(f"\n  {config['n_steps']} steps, mask_rates={config['mask_rates']}")

    for cond_key, cond_data in result["conditions"].items():
        rob_str = ""
        rob = cond_data.get("robustness", {}).get("1.0")
        if rob is not None:
            rob_str = f" rob={rob:+.4f}"
        print(f"    {cond_key:>15s}: val={cond_data['val_loss']:.4f}"
              f"{rob_str}")
