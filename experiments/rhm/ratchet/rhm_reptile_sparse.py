"""RHM Reptile with sparse L2+ inner loop: the untested cell.

Prior experiments showed:
  - Dense NTP meta-learning (FOMAML, Reptile) fails because the inner loop
    trajectory is L0-dominated, drowning out L2+ compositional signal.
  - Sparse supervision on single rule sets (masked NTP, RL) produces the
    first improvements at L2+ but doesn't compound.
  - L2+ outer loss with dense inner loop (rhm_meta_learning_l2) didn't help
    because the inner trajectory is still L0-driven.

This experiment tests the untested combination: sparse L2+ inner loop +
multiple rule sets. The inner loop trains only on hierarchy levels >= 2
(15/63 positions at L=6/s=2), so the trajectory (theta'-theta) is about
L2+ compositional learning, not L0 memorization. Across rule sets, the
transferable compositional structure should accumulate.

Five conditions, all FLOPS-matched:

  Reptile_L2+:         Inner = L2+-only NTP (15/63 positions)
  Reptile_dense:       Inner = dense NTP (63/63 positions)
  Reptile_rand_sparse: Inner = random 24% mask (same # positions as L2+)
  Multi_L2+:           Interleaved L2+-only NTP, no meta structure
  Single_dense:        Standard single-task dense NTP (reference)

Heavy diagnostics:
  - Per-level loss at eval checkpoints
  - Inner loop per-level learning curves (sampled meta-steps)
  - Meta-parameter per-level loss trajectory (zero-shot on held-out rules)
  - Displacement analysis: per-level contribution to (theta'-theta)

Reproduction:
  cd experiments/
  modal run --detach -m rhm.ratchet.rhm_reptile_sparse::rhm_reptile_sparse
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_data import generate_rules, generate_sequences_batched
from rhm.ratchet.rhm_rl_ratchet import (
    _position_levels,
    _eval_per_level,
    _generate_with_traces,
    _compute_per_layer_eta2,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)

app = modal.App("rhm-reptile-sparse", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=36000,
    memory=32768,
)
def rhm_reptile_sparse(
    # DGP
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    # Rule sets
    n_train_rules: int = 32,
    n_eval_rules: int = 8,
    n_tokens_per_rule: int = 2_000_000,
    n_eval_sequences: int = 5000,
    # Model
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    # Reptile
    n_meta_steps: int = 100,
    k_inner: int = 2000,
    inner_lr: float = 3e-4,
    outer_lr: float = 0.5,
    inner_min_level: int = 2,
    # Evaluation
    k_eval: int = 10000,
    eval_ckpt_interval: int = 1000,
    # Training
    batch_size: int = 32,
    # Logging
    meta_eval_interval: int = 10,
    inner_diag_interval: int = 25,
    inner_diag_points: int = 10,
    n_eval_batches: int = 10,
    # Misc
    seed: int = 42,
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT

    device = "cuda"
    L = depth
    seq_len = s ** L
    uniform = float(np.log(v))

    # Build hierarchy-level masks
    pos_levels = _position_levels(seq_len, s)
    l2_mask = torch.tensor(
        [1.0 if pos_levels[t] >= inner_min_level else 0.0
         for t in range(seq_len - 1)], device=device)
    dense_mask = torch.ones(seq_len - 1, device=device)
    n_l2 = int(l2_mask.sum().item())
    n_dense = int(dense_mask.sum().item())

    # Random sparse mask: same number of positions as L2+, randomly chosen
    rng_mask = np.random.RandomState(seed + 999)
    rand_indices = rng_mask.choice(seq_len - 1, size=n_l2, replace=False)
    rand_mask = torch.zeros(seq_len - 1, device=device)
    rand_mask[rand_indices] = 1.0

    # Position-level breakdown of each mask
    mask_level_counts = {}
    for mask_name, mask in [("L2+", l2_mask), ("dense", dense_mask),
                            ("rand", rand_mask)]:
        counts = {}
        for t in range(seq_len - 1):
            lv = int(pos_levels[t])
            counts.setdefault(lv, 0)
            if mask[t] > 0:
                counts[lv] = counts.get(lv, 0) + 1
        mask_level_counts[mask_name] = counts

    total_flops = n_meta_steps * k_inner
    print(f"RHM REPTILE SPARSE on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  Reptile: {n_meta_steps} meta-steps x K={k_inner}")
    print(f"  outer_lr: {outer_lr}, inner_lr: {inner_lr}")
    print(f"  L2+ mask: {n_l2}/{n_dense} positions (levels >= {inner_min_level})")
    print(f"  Random mask: {n_l2}/{n_dense} positions (random)")
    print(f"  Mask level breakdown:")
    for mask_name in ["L2+", "dense", "rand"]:
        counts = mask_level_counts[mask_name]
        parts = [f"L{lv}:{counts.get(lv, 0)}" for lv in range(L)]
        print(f"    {mask_name:>6s}: {' '.join(parts)}")
    print(f"  Total FLOPS per condition: {total_flops:,}")
    print(f"  Rule sets: {n_train_rules} train + {n_eval_rules} eval")
    print(f"  Uniform baseline: {uniform:.4f}")

    # ==================================================================
    # Phase 0: Generate rule set pool
    # ==================================================================
    print(f"\n{'='*60}")
    print("  PHASE 0: GENERATING RULE SET POOL")
    print(f"{'='*60}")

    n_total = n_train_rules + n_eval_rules
    all_train_data = []
    all_val_data = []
    all_eval_seqs = []

    for rs in range(n_total):
        rules = generate_rules(v, s, L, m, seed=rs)
        n_seqs = (n_tokens_per_rule + seq_len - 1) // seq_len
        seqs = generate_sequences_batched(rules, n_seqs, seed=rs + 10000)
        flat = torch.from_numpy(
            seqs.reshape(-1)[:n_tokens_per_rule].astype(np.int64))
        split_idx = int(0.9 * len(flat))
        all_train_data.append(flat[:split_idx])
        all_val_data.append(flat[split_idx:])
        ev_s, lf, lr_ = _generate_with_traces(
            rules, n_eval_sequences, seed=rs + 20000)
        all_eval_seqs.append((ev_s, lf, lr_))
        if rs < 2 or rs == n_total - 1:
            print(f"  Rule set {rs:2d}: {len(flat):,} tokens")
        elif rs == 2:
            print(f"  ...")

    train_ids = list(range(n_train_rules))
    eval_ids = list(range(n_train_rules, n_total))
    print(f"  {n_total} rule sets ready")

    # ==================================================================
    # Helpers
    # ==================================================================
    rng = np.random.RandomState(seed)

    def make_gpt():
        return GPT(v, seq_len, n_layer, n_head, n_embd).to(device)

    def sample_batch(data, bs):
        n = len(data) - seq_len
        starts = torch.randint(n, (bs,))
        x = torch.stack([data[i:i + seq_len] for i in starts]).to(device)
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts]).to(device)
        return x, y

    def masked_ntp_loss(model, x, y, mask):
        logits, _ = model(x)
        pred = logits[:, :-1, :].contiguous().view(-1, v)
        tgt = y[:, :-1].contiguous().view(-1)
        per_pos = F.cross_entropy(
            pred, tgt, reduction='none').view(-1, seq_len - 1)
        return (per_pos * mask.unsqueeze(0)).sum() / (
            per_pos.shape[0] * mask.sum())

    def eval_val_loss(model, rule_id):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(n_eval_batches):
                x, y = sample_batch(all_val_data[rule_id], batch_size)
                _, loss = model(x, y)
                total += loss.item()
        return total / n_eval_batches

    def eval_per_level(model, rule_id):
        return _eval_per_level(
            model, all_eval_seqs[rule_id][0],
            seq_len, s, L, v, batch_size, device)

    def eval_per_level_avg(model, rule_ids):
        """Average per-level loss across multiple rule sets."""
        level_sums = {}
        level_counts = {}
        overall_sums = []
        for rid in rule_ids:
            pl = eval_per_level(model, rid)
            overall_sums.append(pl["overall_loss"])
            for lv, lv_data in pl["per_level"].items():
                lv_int = int(lv) if isinstance(lv, str) else lv
                level_sums[lv_int] = level_sums.get(lv_int, 0.0) + lv_data["loss"]
                level_counts[lv_int] = level_counts.get(lv_int, 0) + 1
        return {
            "overall": float(np.mean(overall_sums)),
            "per_level": {
                lv: level_sums[lv] / level_counts[lv]
                for lv in sorted(level_sums)
            },
        }

    def eval_inner_curve(model, task_id, mask, k_steps, n_points):
        """Run a mini inner loop and record per-level loss at intervals."""
        import copy
        inner_model = make_gpt()
        inner_model.load_state_dict(model.state_dict())
        inner_opt = torch.optim.AdamW(
            inner_model.parameters(), lr=inner_lr, weight_decay=0.01)
        interval = max(1, k_steps // n_points)
        curve = []
        # Step 0
        pl = eval_per_level(inner_model, task_id)
        curve.append({"step": 0, "per_level": pl})
        for k in range(1, k_steps + 1):
            inner_model.train()
            x, y = sample_batch(all_train_data[task_id], batch_size)
            loss = masked_ntp_loss(inner_model, x, y, mask)
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inner_model.parameters(), 1.0)
            inner_opt.step()
            if k % interval == 0 or k == k_steps:
                pl = eval_per_level(inner_model, task_id)
                curve.append({"step": k, "per_level": pl})
        del inner_model, inner_opt
        torch.cuda.empty_cache()
        return curve

    def print_per_level(label, pl_dict, levels=5):
        parts = [f"{label}"]
        for lv in range(min(levels, L)):
            val = pl_dict.get(lv, uniform)
            parts.append(f"L{lv}={val:.4f}")
        parts.append(f"ovr={pl_dict.get('overall', sum(pl_dict.get(lv, uniform) for lv in range(L))/L):.4f}" if 'overall' not in pl_dict else "")
        print(f"    {' '.join(parts)}")

    # ==================================================================
    # Save initial weights
    # ==================================================================
    torch.manual_seed(seed)
    init_model = make_gpt()
    init_state = {k: v_.cpu().clone()
                  for k, v_ in init_model.state_dict().items()}
    n_params = sum(p.numel() for p in init_model.parameters())
    print(f"\n  Model: {n_params / 1e6:.2f}M params")
    del init_model
    torch.cuda.empty_cache()

    # ==================================================================
    # Reptile training (parameterized by inner mask)
    # ==================================================================
    def run_reptile(inner_mask, label, rng_seed):
        print(f"\n{'='*60}")
        n_masked = int(inner_mask.sum().item())
        print(f"  {label}: Reptile {n_meta_steps} x K={k_inner}, "
              f"inner mask {n_masked}/{n_dense} positions")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_state)
        task_rng = np.random.RandomState(rng_seed)

        log = []
        meta_trajectory = []

        for ms in range(n_meta_steps):
            theta = {k: v_.clone() for k, v_ in model.state_dict().items()}
            task_id = int(task_rng.choice(train_ids))

            # Inner loop: masked NTP
            inner_opt = torch.optim.AdamW(
                model.parameters(), lr=inner_lr, weight_decay=0.01)
            model.train()
            inner_loss_sum = 0.0
            for k in range(k_inner):
                x, y = sample_batch(all_train_data[task_id], batch_size)
                loss = masked_ntp_loss(model, x, y, inner_mask)
                inner_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                inner_opt.step()
                inner_loss_sum += loss.item()

            # Reptile update
            theta_prime = {k: v_.clone()
                           for k, v_ in model.state_dict().items()}
            with torch.no_grad():
                for name in theta:
                    theta[name] = theta[name] + outer_lr * (
                        theta_prime[name] - theta[name])
            model.load_state_dict(theta)

            entry = {
                "step": ms,
                "task_id": task_id,
                "inner_loss_avg": inner_loss_sum / k_inner,
            }

            # Periodic eval: zero-shot per-level on held-out rules
            if ms % meta_eval_interval == 0 or ms == n_meta_steps - 1:
                avg_pl = eval_per_level_avg(model, eval_ids[:3])
                entry["zero_shot"] = avg_pl
                meta_trajectory.append({
                    "step": ms, **avg_pl})
                parts = [f"ms={ms:4d}"]
                for lv in range(min(5, L)):
                    parts.append(
                        f"L{lv}={avg_pl['per_level'].get(lv, uniform):.4f}")
                parts.append(f"ovr={avg_pl['overall']:.4f}")
                parts.append(f"inner_avg={inner_loss_sum/k_inner:.4f}")
                print(f"    {' '.join(parts)}")

            # Diagnostic: inner loop per-level learning curve
            if ms % inner_diag_interval == 0 and ms > 0:
                diag_task = int(task_rng.choice(train_ids))
                curve = eval_inner_curve(
                    model, diag_task, inner_mask,
                    k_inner, inner_diag_points)
                entry["inner_curve"] = curve
                # Print summary: step 0 vs step K per-level
                pl0 = curve[0]["per_level"]["per_level"]
                plK = curve[-1]["per_level"]["per_level"]
                parts = [f"  inner_curve ms={ms} (rule {diag_task}):"]
                for lv in range(min(5, L)):
                    l0 = pl0.get(lv, {}).get("loss", uniform)
                    lK = plK.get(lv, {}).get("loss", uniform)
                    parts.append(f"L{lv}:{l0:.3f}->{lK:.3f}")
                print(f"    {' '.join(parts)}")

            log.append(entry)

            del theta, theta_prime, inner_opt
            torch.cuda.empty_cache()

        state = {k: v_.cpu().clone()
                 for k, v_ in model.state_dict().items()}
        del model
        torch.cuda.empty_cache()
        print(f"  {label} done.")
        return state, log, meta_trajectory

    # ==================================================================
    # Multi-task baseline (L2+ masked, FLOPS-matched)
    # ==================================================================
    def run_multi_l2(rng_seed):
        total_steps = n_meta_steps * k_inner
        print(f"\n{'='*60}")
        print(f"  Multi_L2+: {total_steps} L2+-masked NTP steps, "
              f"{n_train_rules} rule sets")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(
            model.parameters(), lr=inner_lr, weight_decay=0.01)
        task_rng = np.random.RandomState(rng_seed)

        log = []
        log_interval = max(1, total_steps // 50)

        for step in range(total_steps):
            model.train()
            rule_id = int(task_rng.choice(train_ids))
            x, y = sample_batch(all_train_data[rule_id], batch_size)
            loss = masked_ntp_loss(model, x, y, l2_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % log_interval == 0 or step == total_steps - 1:
                avg_pl = eval_per_level_avg(model, eval_ids[:3])
                log.append({"step": step, **avg_pl})
                if step % (log_interval * 5) == 0 or step == total_steps - 1:
                    parts = [f"step={step:6d}"]
                    for lv in range(min(5, L)):
                        parts.append(
                            f"L{lv}={avg_pl['per_level'].get(lv, uniform):.4f}")
                    parts.append(f"ovr={avg_pl['overall']:.4f}")
                    print(f"    {' '.join(parts)}")

        state = {k: v_.cpu().clone()
                 for k, v_ in model.state_dict().items()}
        del model, opt
        torch.cuda.empty_cache()
        print("  Multi_L2+ done.")
        return state, log

    # ==================================================================
    # Single-task dense baseline (FLOPS-matched)
    # ==================================================================
    def run_single_dense():
        total_steps = n_meta_steps * k_inner
        print(f"\n{'='*60}")
        print(f"  Single_dense: {total_steps} dense NTP steps, rule 0")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_state)
        opt = torch.optim.AdamW(
            model.parameters(), lr=inner_lr, weight_decay=0.01)

        log = []
        log_interval = max(1, total_steps // 50)

        for step in range(total_steps):
            model.train()
            x, y = sample_batch(all_train_data[0], batch_size)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % log_interval == 0 or step == total_steps - 1:
                avg_pl = eval_per_level_avg(model, eval_ids[:3])
                log.append({"step": step, **avg_pl})
                if step % (log_interval * 5) == 0 or step == total_steps - 1:
                    parts = [f"step={step:6d}"]
                    for lv in range(min(5, L)):
                        parts.append(
                            f"L{lv}={avg_pl['per_level'].get(lv, uniform):.4f}")
                    parts.append(f"ovr={avg_pl['overall']:.4f}")
                    print(f"    {' '.join(parts)}")

        state = {k: v_.cpu().clone()
                 for k, v_ in model.state_dict().items()}
        del model, opt
        torch.cuda.empty_cache()
        print("  Single_dense done.")
        return state, log

    # ==================================================================
    # Run all conditions
    # ==================================================================
    # Same task RNG seed for all Reptile conditions so they see the same
    # rule set sequence — only the inner mask differs.
    reptile_seed = seed + 100

    rep_l2_state, rep_l2_log, rep_l2_traj = run_reptile(
        l2_mask, "Reptile_L2+", reptile_seed)
    rep_dense_state, rep_dense_log, rep_dense_traj = run_reptile(
        dense_mask, "Reptile_dense", reptile_seed)
    rep_rand_state, rep_rand_log, rep_rand_traj = run_reptile(
        rand_mask, "Reptile_rand_sparse", reptile_seed)
    multi_l2_state, multi_l2_log = run_multi_l2(reptile_seed)
    single_state, single_log = run_single_dense()

    # ==================================================================
    # EVALUATION: Fine-tune all conditions on held-out rules (dense NTP)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  EVALUATION: Fine-tune on {n_eval_rules} held-out rules")
    print(f"  K_eval={k_eval}, dense NTP, ckpt every {eval_ckpt_interval}")
    print(f"{'='*60}")

    conditions = [
        ("Reptile_L2+", rep_l2_state),
        ("Reptile_dense", rep_dense_state),
        ("Reptile_rand_sparse", rep_rand_state),
        ("Multi_L2+", multi_l2_state),
        ("Single_dense", single_state),
    ]
    eval_results = {}

    for cond_name, cond_state in conditions:
        eval_results[cond_name] = {}
        print(f"\n  --- {cond_name} ---")

        for eval_id in eval_ids:
            model = make_gpt()
            model.load_state_dict(cond_state)
            opt = torch.optim.AdamW(
                model.parameters(), lr=inner_lr, weight_decay=0.01)

            rule_key = f"rule_{eval_id}"
            checkpoints = []

            # Step 0 (zero-shot)
            pl = eval_per_level(model, eval_id)
            vl = eval_val_loss(model, eval_id)
            checkpoints.append({"step": 0, "val_loss": vl, "per_level": pl})

            # Fine-tune with dense NTP
            for step in range(1, k_eval + 1):
                model.train()
                x, y = sample_batch(all_train_data[eval_id], batch_size)
                _, loss = model(x, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % eval_ckpt_interval == 0 or step == k_eval:
                    pl = eval_per_level(model, eval_id)
                    vl = eval_val_loss(model, eval_id)
                    checkpoints.append({
                        "step": step, "val_loss": vl, "per_level": pl})

            # Final eta2
            eta2 = _compute_per_layer_eta2(
                model, *all_eval_seqs[eval_id],
                n_layer, s, L, batch_size, device)
            checkpoints[-1]["per_layer_eta2"] = eta2

            eval_results[cond_name][rule_key] = checkpoints

            pl_f = checkpoints[-1]["per_level"]["per_level"]
            parts = [f"rule {eval_id}: "
                     f"vl {checkpoints[0]['val_loss']:.3f}"
                     f"->{checkpoints[-1]['val_loss']:.3f}"]
            for lv in range(min(5, L)):
                lf = pl_f.get(lv, {}).get("loss", uniform)
                parts.append(f"L{lv}:{lf:.3f}")
            print(f"    {' '.join(parts)}")

            del model, opt
            torch.cuda.empty_cache()

    # ==================================================================
    # Summary tables
    # ==================================================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    # Compute averaged trajectories per condition
    cond_names = [c[0] for c in conditions]
    summary = {}
    for cond_name in cond_names:
        cond_data = eval_results[cond_name]
        all_steps = sorted(set(
            ck["step"] for rd in cond_data.values() for ck in rd))
        step_avgs = {}
        for st in all_steps:
            level_sums = {}
            level_counts = {}
            overall_sums = []
            for rd in cond_data.values():
                for ck in rd:
                    if ck["step"] != st:
                        continue
                    overall_sums.append(ck["per_level"]["overall_loss"])
                    for lv, lv_data in ck["per_level"]["per_level"].items():
                        lv_int = int(lv) if isinstance(lv, str) else lv
                        level_sums.setdefault(lv_int, 0.0)
                        level_counts.setdefault(lv_int, 0)
                        level_sums[lv_int] += lv_data["loss"]
                        level_counts[lv_int] += 1
            step_avgs[st] = {
                "overall": float(np.mean(overall_sums)),
                "per_level": {
                    lv: level_sums[lv] / level_counts[lv]
                    for lv in sorted(level_sums)
                },
            }
        summary[cond_name] = step_avgs

    # Table 1: Final per-level loss (after 10K fine-tuning)
    print(f"\n  TABLE 1: Per-level loss after {k_eval} fine-tuning steps "
          f"(avg over {n_eval_rules} eval rules)")
    header = f"  {'Cond':>20s}"
    for lv in range(L):
        header += f"  {'L'+str(lv):>7s}"
    header += f"  {'overall':>8s}"
    print(header)
    final_step = max(summary[cond_names[0]].keys())
    for cn in cond_names:
        final = summary[cn][final_step]
        row = f"  {cn:>20s}"
        for lv in range(L):
            row += f"  {final['per_level'].get(lv, uniform):7.4f}"
        row += f"  {final['overall']:8.4f}"
        print(row)

    # Table 2: Zero-shot per-level loss (step 0 of eval, before fine-tuning)
    print(f"\n  TABLE 2: Zero-shot per-level loss (before fine-tuning)")
    print(header)
    for cn in cond_names:
        zero = summary[cn][0]
        row = f"  {cn:>20s}"
        for lv in range(L):
            row += f"  {zero['per_level'].get(lv, uniform):7.4f}"
        row += f"  {zero['overall']:8.4f}"
        print(row)

    # Table 3: Transfer gap relative to Single_dense
    print(f"\n  TABLE 3: Gap vs Single_dense after {k_eval} steps "
          f"(negative = better than Single)")
    header2 = f"  {'Cond':>20s}"
    for lv in range(min(5, L)):
        header2 += f"  {'L'+str(lv):>7s}"
    header2 += f"  {'overall':>8s}"
    print(header2)
    single_final = summary["Single_dense"][final_step]
    for cn in cond_names:
        if cn == "Single_dense":
            continue
        final = summary[cn][final_step]
        row = f"  {cn:>20s}"
        for lv in range(min(5, L)):
            gap = final["per_level"].get(lv, uniform) - \
                  single_final["per_level"].get(lv, uniform)
            row += f"  {gap:+7.4f}"
        gap_ovr = final["overall"] - single_final["overall"]
        row += f"  {gap_ovr:+8.4f}"
        print(row)

    # Table 4: Learning speed — per-level loss at step 1000
    print(f"\n  TABLE 4: Per-level loss at step 1000 (early fine-tuning)")
    print(header)
    for cn in cond_names:
        early = summary[cn].get(1000, summary[cn][min(summary[cn].keys())])
        row = f"  {cn:>20s}"
        for lv in range(L):
            row += f"  {early['per_level'].get(lv, uniform):7.4f}"
        row += f"  {early['overall']:8.4f}"
        print(row)

    # Table 5: Full learning curves for L0, L2, L3
    for lv in [0, 2, 3]:
        print(f"\n  TABLE 5{chr(97+[0,2,3].index(lv))}: "
              f"Level {lv} learning curve")
        header3 = f"  {'step':>6s}"
        for cn in cond_names:
            header3 += f"  {cn[:12]:>12s}"
        print(header3)
        for st in sorted(summary[cond_names[0]].keys()):
            row = f"  {st:6d}"
            for cn in cond_names:
                val = summary[cn][st]["per_level"].get(lv, uniform)
                row += f"  {val:12.4f}"
            print(row)

    # Table 6: Meta-training zero-shot trajectory (Reptile conditions only)
    print(f"\n  TABLE 6: Meta-training zero-shot trajectory")
    for traj_name, traj in [("Reptile_L2+", rep_l2_traj),
                             ("Reptile_dense", rep_dense_traj),
                             ("Reptile_rand_sparse", rep_rand_traj)]:
        print(f"\n  {traj_name}:")
        print(f"  {'ms':>6s}  {'L0':>7s}  {'L1':>7s}  {'L2':>7s}  "
              f"{'L3':>7s}  {'ovr':>7s}")
        for entry in traj:
            row = f"  {entry['step']:6d}"
            for lv in range(min(4, L)):
                row += f"  {entry['per_level'].get(lv, uniform):7.4f}"
            row += f"  {entry['overall']:7.4f}"
            print(row)

    # ==================================================================
    # Save
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_reptile_sparse"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "n_params": n_params,
            "n_train_rules": n_train_rules,
            "n_eval_rules": n_eval_rules,
            "n_meta_steps": n_meta_steps,
            "k_inner": k_inner,
            "inner_lr": inner_lr,
            "outer_lr": outer_lr,
            "inner_min_level": inner_min_level,
            "n_l2_positions": n_l2,
            "k_eval": k_eval,
            "batch_size": batch_size,
            "seed": seed,
            "total_flops_per_condition": total_flops,
            "mask_level_counts": mask_level_counts,
        },
        "training": {
            "Reptile_L2+": {"log": rep_l2_log, "trajectory": rep_l2_traj},
            "Reptile_dense": {"log": rep_dense_log, "trajectory": rep_dense_traj},
            "Reptile_rand_sparse": {"log": rep_rand_log, "trajectory": rep_rand_traj},
            "Multi_L2+": {"log": multi_l2_log},
            "Single_dense": {"log": single_log},
        },
        "evaluation": eval_results,
        "summary": summary,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    import torch as _torch
    for cn, st in conditions:
        safe_name = cn.replace("+", "plus").replace(" ", "_").lower()
        _torch.save(st, os.path.join(save_dir, f"{safe_name}_model.pt"))
    _torch.save(init_state, os.path.join(save_dir, "init_model.pt"))

    volume.commit()
    print(f"\n  Saved to {save_dir}/")
    return result


@app.local_entrypoint()
def main():
    result = rhm_reptile_sparse.remote()
    print("\nRHM Reptile sparse experiment complete.")
