"""RHM Reptile: meta-learning compositional structure via rule-set transfer.

Reptile (Nichol & Schulman, 2018) meta-update: θ ← θ + ε(θ' - θ), where θ'
is the result of K steps of NTP on a sampled rule set. Over many rule sets,
L0-specific components of (θ'-θ) point in different directions and cancel,
while L1+ compositional components (shared across all rule sets) accumulate.

Key advantage over FOMAML: Reptile captures the full K-step trajectory
(including late-K L2+ learning) without needing second-order gradients.
With K=5000, the model reaches ~0.16 nats of L2 learning — a real signal
that FOMAML at K=50-500 never reached.

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_reptile::rhm_reptile
"""

import json
import os
import copy
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_data import generate_rules, generate_sequences_batched
from rhm.rhm_rl_ratchet import (
    _eval_per_level,
    _generate_with_traces,
    _compute_per_layer_eta2,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)

app = modal.App("rhm-reptile", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def rhm_reptile(
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
    n_meta_steps: int = 500,
    k_inner: int = 5000,
    inner_lr: float = 3e-4,
    outer_lr: float = 0.5,
    # Evaluation fine-tuning
    k_eval: int = 10000,
    eval_ckpt_interval: int = 1000,
    # General training
    batch_size: int = 32,
    # Logging
    train_eval_interval: int = 10,
    n_eval_batches: int = 10,
    # Misc
    seed: int = 42,
):
    import torch
    import numpy as np
    from rhm.model import GPT

    device = "cuda"
    L = depth
    seq_len = s ** L
    total_flops = n_meta_steps * k_inner
    uniform = float(np.log(v))

    print(f"RHM REPTILE on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  Reptile: {n_meta_steps} meta-steps x K={k_inner} inner")
    print(f"  outer_lr (epsilon): {outer_lr}")
    print(f"  Total FLOPS: {total_flops:,} fwd/bwd passes")
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

    def eval_on_held_out(model, tag=""):
        """Quick eval: per-level loss on first eval rule set."""
        pl = eval_per_level(model, eval_ids[0])
        vl = eval_val_loss(model, eval_ids[0])
        lvls = pl["per_level"]
        parts = [f"{tag}vl={vl:.4f}"]
        for lv in range(min(5, L)):
            parts.append(f"L{lv}={lvls.get(lv, {}).get('loss', uniform):.4f}")
        print(f"    {' '.join(parts)}")
        return {"val_loss": vl, "per_level": pl}

    # ==================================================================
    # Initialize model
    # ==================================================================
    torch.manual_seed(seed)
    model = make_gpt()
    init_state = {k: v_.cpu().clone()
                  for k, v_ in model.state_dict().items()}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params / 1e6:.2f}M params")

    # ==================================================================
    # REPTILE TRAINING
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  REPTILE: {n_meta_steps} meta-steps x K={k_inner}")
    print(f"{'='*60}")

    reptile_log = []

    for ms in range(n_meta_steps):
        # Save meta-parameters
        theta = {k: v_.clone() for k, v_ in model.state_dict().items()}

        # Sample a rule set for the inner loop
        task_id = int(rng.choice(train_ids))

        # Inner loop: K NTP steps on this rule set
        inner_opt = torch.optim.AdamW(
            model.parameters(), lr=inner_lr, weight_decay=0.01)
        model.train()
        inner_loss_sum = 0.0
        inner_loss_start = None
        inner_loss_end = None
        for k in range(k_inner):
            x, y = sample_batch(all_train_data[task_id], batch_size)
            _, loss = model(x, y)
            inner_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_opt.step()
            lv = loss.item()
            inner_loss_sum += lv
            if k == 0:
                inner_loss_start = lv
            if k == k_inner - 1:
                inner_loss_end = lv

        # Reptile meta-update: θ ← θ + ε(θ' - θ)
        theta_prime = model.state_dict()
        with torch.no_grad():
            for name in theta:
                delta = theta_prime[name] - theta[name]
                theta[name] = theta[name] + outer_lr * delta
        model.load_state_dict(theta)

        # Compute update magnitude for monitoring
        update_norm = 0.0
        with torch.no_grad():
            for name in theta:
                update_norm += (
                    (theta_prime[name] - theta[name]).float().norm() ** 2
                ).item()
        update_norm = update_norm ** 0.5

        del theta, theta_prime, inner_opt
        torch.cuda.empty_cache()

        entry = {
            "step": ms,
            "task_id": task_id,
            "inner_loss_avg": inner_loss_sum / k_inner,
            "inner_loss_start": inner_loss_start,
            "inner_loss_end": inner_loss_end,
            "update_norm": update_norm,
        }

        if ms % train_eval_interval == 0 or ms == n_meta_steps - 1:
            ev = eval_on_held_out(model, tag=f"ms={ms:4d} ")
            entry["eval_val_loss"] = ev["val_loss"]
            entry["eval_per_level"] = ev["per_level"]
            print(f"    inner: {inner_loss_start:.4f}->{inner_loss_end:.4f} "
                  f"(avg {inner_loss_sum/k_inner:.4f}) "
                  f"|Δθ|={update_norm:.4f} rule={task_id}")

        reptile_log.append(entry)

    reptile_state = {k: v_.cpu().clone()
                     for k, v_ in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    print("  Reptile training done.\n")

    # ==================================================================
    # EVALUATION: Fine-tune on held-out rule sets
    # ==================================================================
    print(f"{'='*60}")
    print(f"  EVALUATION: Fine-tune on {n_eval_rules} held-out rule sets")
    print(f"  K_eval={k_eval}, ckpt every {eval_ckpt_interval}")
    print(f"{'='*60}")

    eval_results = {}
    for eval_id in eval_ids:
        model = make_gpt()
        model.load_state_dict(reptile_state)
        opt = torch.optim.AdamW(
            model.parameters(), lr=inner_lr, weight_decay=0.01)

        rule_key = f"rule_{eval_id}"
        checkpoints = []

        # Step 0
        pl = eval_per_level(model, eval_id)
        vl = eval_val_loss(model, eval_id)
        checkpoints.append({"step": 0, "val_loss": vl, "per_level": pl})

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
        eval_results[rule_key] = checkpoints

        pl_i = checkpoints[0]["per_level"]["per_level"]
        pl_f = checkpoints[-1]["per_level"]["per_level"]
        parts = [f"rule {eval_id}: "
                 f"vl {checkpoints[0]['val_loss']:.3f}"
                 f"->{checkpoints[-1]['val_loss']:.3f}"]
        for lv in range(min(5, L)):
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

    all_steps = sorted(set(
        ck["step"] for rd in eval_results.values() for ck in rd))
    step_avgs = {}
    for st in all_steps:
        level_sums = {}
        level_counts = {}
        overall_sums = []
        for rd in eval_results.values():
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

    print(f"\n  Per-level loss trajectory (avg over {n_eval_rules} eval rules):")
    header = f"  {'step':>6s}"
    for lv in range(min(6, L)):
        header += f"  {'L'+str(lv):>7s}"
    header += f"  {'overall':>8s}"
    print(header)
    for st in all_steps:
        row = f"  {st:6d}"
        for lv in range(min(6, L)):
            val = step_avgs[st]["per_level"].get(lv, uniform)
            row += f"  {val:7.4f}"
        row += f"  {step_avgs[st]['overall']:8.4f}"
        print(row)

    # Meta-training trajectory
    print(f"\n  Meta-training trajectory (eval on held-out rule):")
    print(f"  {'ms':>6s}  {'vl':>7s}  {'L0':>7s}  {'L1':>7s}  "
          f"{'L2':>7s}  {'L3':>7s}")
    for entry in reptile_log:
        if "eval_per_level" not in entry:
            continue
        pl = entry["eval_per_level"]["per_level"]
        row = f"  {entry['step']:6d}  {entry['eval_val_loss']:7.4f}"
        for lv in range(min(4, L)):
            row += f"  {pl.get(lv, {}).get('loss', uniform):7.4f}"
        print(row)

    # ==================================================================
    # Save
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_reptile"
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
            "k_eval": k_eval,
            "eval_ckpt_interval": eval_ckpt_interval,
            "batch_size": batch_size,
            "seed": seed,
            "total_flops": total_flops,
        },
        "training": {"log": reptile_log},
        "evaluation": eval_results,
        "summary": step_avgs,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    import torch as _torch
    _torch.save(reptile_state, os.path.join(save_dir, "reptile_model.pt"))
    _torch.save(init_state, os.path.join(save_dir, "init_model.pt"))

    volume.commit()
    print(f"\n  Saved to {save_dir}/")
    return result


@app.local_entrypoint()
def main():
    result = rhm_reptile.remote()
    print("\nRHM Reptile experiment complete.")
