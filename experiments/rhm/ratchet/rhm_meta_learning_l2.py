"""RHM Meta-Learning with L2+ Outer Loss (Privileged DGP Signal).

Tests whether a hierarchy-aware outer loss rescues FOMAML on RHM.
The "cheat": the outer loss uses the s-adic valuation to evaluate
only at hierarchy levels >= 2, removing L0 gradient dominance from
the meta-gradient.

Prior runs showed FOMAML with standard NTP outer loss fails because
the meta-gradient is dominated by L0 signal (same as multi-task NTP).
By restricting the outer loss to L2+ positions, the meta-gradient
explicitly optimizes for compositional transfer across rule sets.

If MAML_L2 produces deeper composition than MAML_all, the diagnosis is
confirmed: FOMAML's second-order benefit was masked by L0-dominated NTP.
The path to a non-cheating version: find a loss that naturally selects
for L2+ (generation accuracy, FM-surprise NTP, sparse masking).

Four conditions (update-matched on model updates):
  MAML_L2:  Inner=NTP on R, Outer=L2+ NTP on R' (the cheat)
  MAML_all: Inner=NTP on R, Outer=full NTP on R' (reference)
  Multi:    NTP on interleaved rules (same #updates as MAML)
  Single:   NTP on one rule set (same #updates as MAML)

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_meta_learning_l2::rhm_meta_learning_l2
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_data import generate_rules, generate_sequences_batched
from rhm.rhm_rl_ratchet import (
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

app = modal.App("rhm-meta-learning-l2", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def rhm_meta_learning_l2(
    # DGP
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    # Rule sets
    n_train_rules: int = 16,
    n_eval_rules: int = 8,
    n_tokens_per_rule: int = 2_000_000,
    n_eval_sequences: int = 5000,
    # Model
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    # MAML
    n_meta_steps: int = 3000,
    k_inner: int = 200,
    inner_lr: float = 3e-4,
    meta_lr: float = 3e-4,
    outer_min_level: int = 2,
    # Evaluation fine-tuning
    k_eval: int = 10000,
    eval_ckpt_interval: int = 1000,
    # General training
    lr: float = 3e-4,
    batch_size: int = 32,
    # Logging
    train_eval_interval: int = 100,
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
    baseline_steps = n_meta_steps
    maml_flops = n_meta_steps * (k_inner + 1)
    uniform = float(np.log(v))

    # Compute hierarchy level mask
    pos_levels = _position_levels(seq_len, s)
    l2_mask = torch.tensor(
        [1.0 if pos_levels[t] >= outer_min_level else 0.0
         for t in range(seq_len - 1)],
        device=device)
    all_mask = torch.ones(seq_len - 1, device=device)
    n_l2_positions = int(l2_mask.sum().item())

    print(f"RHM META-LEARNING L2+ OUTER on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  MAML: {n_meta_steps} meta-steps x (K={k_inner} inner + 1 outer)")
    print(f"  MAML FLOPS: {maml_flops} fwd/bwd passes")
    print(f"  Baselines: {baseline_steps} steps (update-matched)")
    print(f"  Outer mask: levels >= {outer_min_level} "
          f"({n_l2_positions}/{seq_len-1} positions)")
    print(f"  Rule sets: {n_train_rules} train + {n_eval_rules} eval")
    print(f"  Eval: K_eval={k_eval}, ckpt every {eval_ckpt_interval}")
    print(f"  Uniform baseline: {uniform:.4f}")

    # ==================================================================
    # Phase 0: Generate rule set pool
    # ==================================================================
    print(f"\n{'='*60}")
    print("  PHASE 0: GENERATING RULE SET POOL")
    print(f"{'='*60}")

    n_total = n_train_rules + n_eval_rules
    all_rules = []
    all_train_data = []
    all_val_data = []
    all_eval_seqs = []

    for rs in range(n_total):
        rules = generate_rules(v, s, L, m, seed=rs)
        all_rules.append(rules)
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
        x = torch.stack(
            [data[i:i + seq_len] for i in starts]).to(device)
        y = torch.stack(
            [data[i + 1:i + seq_len + 1] for i in starts]).to(device)
        return x, y

    def masked_ntp_loss(model, x, y, mask):
        """NTP loss restricted to positions selected by mask."""
        logits, _ = model(x)
        pred = logits[:, :-1, :].contiguous().view(-1, v)
        tgt = y[:, :-1].contiguous().view(-1)
        per_pos = F.cross_entropy(
            pred, tgt, reduction='none'
        ).view(-1, seq_len - 1)
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

    def eval_per_layer_eta2(model, rule_id):
        seqs, lf, lr_ = all_eval_seqs[rule_id]
        return _compute_per_layer_eta2(
            model, seqs, lf, lr_, n_layer, s, L, batch_size, device)

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
    # MAML training (shared logic, parameterized by outer mask)
    # ==================================================================
    def run_maml(outer_mask, label):
        print(f"\n{'='*60}")
        n_outer_pos = int(outer_mask.sum().item())
        print(f"  {label}: {n_meta_steps} meta-steps, "
              f"K={k_inner}, outer={n_outer_pos}/{seq_len-1} positions")
        print(f"{'='*60}")

        model = make_gpt()
        model.load_state_dict(init_state)
        meta_opt = torch.optim.AdamW(
            model.parameters(), lr=meta_lr, weight_decay=0.01)

        log = []
        # Use a separate RNG so both MAML conditions see the same task sequence
        maml_rng = np.random.RandomState(seed + hash(label) % 10000)

        for ms in range(n_meta_steps):
            saved = {k: v_.clone()
                     for k, v_ in model.state_dict().items()}

            inner_id = int(maml_rng.choice(train_ids))
            outer_id = int(maml_rng.choice(
                [r for r in train_ids if r != inner_id]))

            # Inner loop: standard NTP on inner rule set
            inner_opt = torch.optim.AdamW(
                model.parameters(), lr=inner_lr, weight_decay=0.01)
            model.train()
            inner_loss_sum = 0.0
            for k in range(k_inner):
                x, y = sample_batch(all_train_data[inner_id], batch_size)
                _, loss = model(x, y)
                inner_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                inner_opt.step()
                inner_loss_sum += loss.item()

            # Outer: masked NTP on different rule set
            model.zero_grad()
            x_out, y_out = sample_batch(
                all_train_data[outer_id], batch_size)
            outer_loss = masked_ntp_loss(model, x_out, y_out, outer_mask)
            outer_loss.backward()
            outer_grads = [p.grad.clone() for p in model.parameters()]

            # Restore theta and apply meta-update
            model.load_state_dict(saved)
            meta_opt.zero_grad()
            for p, g in zip(model.parameters(), outer_grads):
                p.grad = g
            meta_opt.step()

            del saved, inner_opt, outer_grads
            torch.cuda.empty_cache()

            entry = {
                "step": ms,
                "outer_loss": float(outer_loss.item()),
                "inner_loss_avg": inner_loss_sum / k_inner,
            }

            if ms % train_eval_interval == 0 or ms == n_meta_steps - 1:
                ev_id = int(maml_rng.choice(train_ids))
                vl = eval_val_loss(model, ev_id)
                entry["eval_val_loss"] = vl
                print(f"  ms={ms:4d}: outer={outer_loss.item():.4f} "
                      f"inner_avg={inner_loss_sum/k_inner:.4f} "
                      f"eval_vl={vl:.4f}")

            log.append(entry)

        state = {k: v_.cpu().clone()
                 for k, v_ in model.state_dict().items()}
        del model, meta_opt
        torch.cuda.empty_cache()
        print(f"  {label} done.")
        return state, log

    # ==================================================================
    # Run both MAML conditions
    # ==================================================================
    maml_l2_state, maml_l2_log = run_maml(l2_mask, "MAML_L2")
    maml_all_state, maml_all_log = run_maml(all_mask, "MAML_all")

    # ==================================================================
    # CONDITION: Multi-task (update-matched)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  MULTI-TASK: {baseline_steps} NTP steps, "
          f"{n_train_rules} rule sets")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(init_state)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01)
    multi_log = []
    log_interval = max(1, baseline_steps // 50)

    for step in range(baseline_steps):
        model.train()
        rule_id = int(rng.choice(train_ids))
        x, y = sample_batch(all_train_data[rule_id], batch_size)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_interval == 0 or step == baseline_steps - 1:
            ev_id = int(rng.choice(train_ids))
            vl = eval_val_loss(model, ev_id)
            multi_log.append({
                "step": step, "train_loss": float(loss.item()),
                "eval_val_loss": vl,
            })
            print(f"  step={step:5d}: train={loss.item():.4f} "
                  f"eval_vl={vl:.4f}")

    multi_state = {k: v_.cpu().clone()
                   for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Multi-task done.\n")

    # ==================================================================
    # CONDITION: Single-task (update-matched)
    # ==================================================================
    print(f"{'='*60}")
    print(f"  SINGLE-TASK: {baseline_steps} NTP steps, rule 0")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(init_state)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01)
    single_log = []

    for step in range(baseline_steps):
        model.train()
        x, y = sample_batch(all_train_data[0], batch_size)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_interval == 0 or step == baseline_steps - 1:
            vl = eval_val_loss(model, 0)
            single_log.append({
                "step": step, "train_loss": float(loss.item()),
                "val_loss": vl,
            })
            print(f"  step={step:5d}: train={loss.item():.4f} "
                  f"val={vl:.4f}")

    single_state = {k: v_.cpu().clone()
                    for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Single-task done.\n")

    # ==================================================================
    # EVALUATION: Fine-tune on held-out rule sets
    # ==================================================================
    print(f"{'='*60}")
    print(f"  EVALUATION: Fine-tune on {n_eval_rules} held-out rule sets")
    print(f"  K_eval={k_eval}, ckpt every {eval_ckpt_interval}")
    print(f"{'='*60}")

    conditions = [
        ("MAML_L2", maml_l2_state),
        ("MAML_all", maml_all_state),
        ("Multi", multi_state),
        ("Single", single_state),
    ]
    eval_results = {}

    for cond_name, cond_state in conditions:
        eval_results[cond_name] = {}
        print(f"\n  --- {cond_name} ---")

        for eval_id in eval_ids:
            model = make_gpt()
            model.load_state_dict(cond_state)
            opt = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=0.01)

            rule_key = f"rule_{eval_id}"
            checkpoints = []

            # Step 0
            pl = eval_per_level(model, eval_id)
            vl = eval_val_loss(model, eval_id)
            checkpoints.append({
                "step": 0, "val_loss": vl, "per_level": pl})

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

            eta2 = eval_per_layer_eta2(model, eval_id)
            checkpoints[-1]["per_layer_eta2"] = eta2
            eval_results[cond_name][rule_key] = checkpoints

            pl_i = checkpoints[0]["per_level"]["per_level"]
            pl_f = checkpoints[-1]["per_level"]["per_level"]
            parts = [f"rule {eval_id}: "
                     f"vl {checkpoints[0]['val_loss']:.3f}"
                     f"->{checkpoints[-1]['val_loss']:.3f}"]
            for lv in range(min(4, L)):
                li = pl_i.get(lv, {}).get("loss", uniform)
                lf = pl_f.get(lv, {}).get("loss", uniform)
                parts.append(f"L{lv}:{li:.3f}->{lf:.3f}")
            print(f"    {' '.join(parts)}")

            del model, opt
            torch.cuda.empty_cache()

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    cond_names = ["MAML_L2", "MAML_all", "Multi", "Single"]

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

    # Final table
    print(f"\n  Per-level loss after {k_eval} fine-tuning steps "
          f"(avg over {n_eval_rules} eval rules):")
    header = f"  {'Cond':>10s}"
    for lv in range(L):
        header += f"  {'L'+str(lv):>7s}"
    header += f"  {'overall':>8s}"
    print(header)

    final_step = max(summary["MAML_L2"].keys())
    for cn in cond_names:
        final = summary[cn][final_step]
        row = f"  {cn:>10s}"
        for lv in range(L):
            row += f"  {final['per_level'].get(lv, uniform):7.4f}"
        row += f"  {final['overall']:8.4f}"
        print(row)

    # L2+ transfer gap: MAML_L2 vs others
    print(f"\n  Transfer gap at L2+ (other - MAML_L2, "
          f"positive = MAML_L2 better):")
    for ref in ["MAML_all", "Multi", "Single"]:
        print(f"\n  {ref} - MAML_L2:")
        header = f"  {'step':>6s}"
        for lv in [2, 3, 4]:
            header += f"  {'L'+str(lv):>7s}"
        print(header)
        for st in sorted(summary["MAML_L2"].keys()):
            row = f"  {st:6d}"
            for lv in [2, 3, 4]:
                ref_val = summary[ref][st]["per_level"].get(lv, uniform)
                l2_val = summary["MAML_L2"][st]["per_level"].get(lv, uniform)
                gap = ref_val - l2_val
                row += f"  {gap:+7.4f}"
            print(row)

    # Learning curves for key levels
    print(f"\n  Learning curves:")
    for lv in [0, 1, 2, 3]:
        print(f"\n  Level {lv}:")
        header = f"  {'step':>6s}"
        for cn in cond_names:
            header += f"  {cn:>10s}"
        print(header)
        for st in sorted(summary["MAML_L2"].keys()):
            row = f"  {st:6d}"
            for cn in cond_names:
                val = summary[cn][st]["per_level"].get(lv, uniform)
                row += f"  {val:10.4f}"
            print(row)

    # ==================================================================
    # Save
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_meta_learning_l2"
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
            "outer_min_level": outer_min_level,
            "n_l2_positions": n_l2_positions,
            "inner_lr": inner_lr,
            "meta_lr": meta_lr,
            "k_eval": k_eval,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "maml_flops": maml_flops,
            "baseline_steps": baseline_steps,
            "matching": "update-matched",
        },
        "training": {
            "MAML_L2": {"log": maml_l2_log},
            "MAML_all": {"log": maml_all_log},
            "Multi": {"log": multi_log},
            "Single": {"log": single_log},
        },
        "evaluation": eval_results,
        "summary": summary,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    import torch as _torch
    _torch.save(maml_l2_state, os.path.join(save_dir, "maml_l2_model.pt"))
    _torch.save(maml_all_state, os.path.join(save_dir, "maml_all_model.pt"))
    _torch.save(multi_state, os.path.join(save_dir, "multi_model.pt"))
    _torch.save(single_state, os.path.join(save_dir, "single_model.pt"))

    volume.commit()
    print(f"\n  Saved to {save_dir}/")
    return result


@app.local_entrypoint()
def main():
    result = rhm_meta_learning_l2.remote()
    print("\nRHM meta-learning L2+ experiment complete.")
