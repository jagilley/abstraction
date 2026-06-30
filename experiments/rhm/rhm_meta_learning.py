"""RHM Meta-Learning: FOMAML with rule-set transfer for compositional learning.

Tests whether bilevel meta-learning across RHM rule sets produces deeper
hierarchy learning than standard NTP. The inner loop trains on one rule set;
the outer loop evaluates on a different rule set (same L, m, s, v). The
meta-gradient optimizes the initialization for cross-rule-set transfer.

Core hypothesis: standard NTP on fixed RHM data concentrates learning at L0
(the easiest hierarchy level) because L0 rules are specific to each rule set
and dominate the gradient. Meta-learning with rule-set transfer creates
gradient pressure toward the *transferable* compositional structure — the
principle of hierarchical composition that's shared across all rule sets —
rather than specific L0 rules that don't transfer.

Three conditions, update-matched (same number of model updates):
  MAML:   N_meta meta-steps x (K_inner inner + 1 outer). Uses N_meta x (K+1)
          FLOPS but only N_meta model updates (one per meta-step).
  Multi:  N_meta NTP steps on interleaved rule sets. Same model updates,
          1/(K+1) the FLOPS. Tests whether diversity alone matches MAML.
  Single: N_meta NTP steps on one rule set. Standard baseline.

Evaluation: fine-tune each condition's trained checkpoint on 8 held-out rule
sets for K_eval steps, measuring per-level loss at checkpoints. Primary metric:
per-level loss trajectory. MAML should learn L1+ faster than Multi and Single.

Reproduction:
  cd experiments/
  modal run --detach -m rhm.rhm_meta_learning::rhm_meta_learning
"""

import json
import os
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

app = modal.App("rhm-meta-learning", image=image)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=21600,
    memory=32768,
)
def rhm_meta_learning(
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
    n_meta_steps: int = 50,
    k_inner: int = 500,
    inner_lr: float = 3e-4,
    meta_lr: float = 3e-4,
    # Evaluation fine-tuning
    k_eval: int = 2000,
    eval_ckpt_interval: int = 200,
    # General training
    lr: float = 3e-4,
    batch_size: int = 32,
    # Logging
    train_eval_interval: int = 10,
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
    maml_flops = n_meta_steps * (k_inner + 1)
    baseline_steps = n_meta_steps  # update-matched: same model updates as MAML
    uniform = float(np.log(v))

    print(f"RHM META-LEARNING on {device}")
    print(f"  DGP: L={L}, m={m}, v={v}, s={s} (seq_len={seq_len})")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  MAML: {n_meta_steps} meta-steps x (K={k_inner} inner + 1 outer)")
    print(f"  MAML FLOPS: {maml_flops} fwd/bwd passes")
    print(f"  Baselines: {baseline_steps} steps (update-matched)")
    print(f"  Rule sets: {n_train_rules} train + {n_eval_rules} eval")
    print(f"  Eval: K_eval={k_eval}, ckpt every {eval_ckpt_interval}")
    print(f"  Uniform baseline: {uniform:.4f}")

    # ==================================================================
    # Phase 0: Generate rule set pool (all in memory)
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
            print(f"  Rule set {rs:2d}: {len(flat):,} tokens "
                  f"(train={split_idx:,}, val={len(flat)-split_idx:,})")
        elif rs == 2:
            print(f"  ...")

    train_ids = list(range(n_train_rules))
    eval_ids = list(range(n_train_rules, n_total))
    total_mem = sum(len(d) * 8 for d in all_train_data + all_val_data)
    print(f"  {n_total} rule sets, ~{total_mem / 1e6:.0f} MB in memory")

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

    # ==================================================================
    # Save initial weights (shared across all conditions)
    # ==================================================================
    torch.manual_seed(seed)
    init_model = make_gpt()
    init_state = {k: v_.cpu().clone()
                  for k, v_ in init_model.state_dict().items()}
    n_params = sum(p.numel() for p in init_model.parameters())
    print(f"\n  Model: {n_params / 1e6:.2f}M params")
    del init_model
    torch.cuda.empty_cache()

    all_results = {}

    # ==================================================================
    # CONDITION 1: MAML (FOMAML with rule-set transfer)
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"  CONDITION 1: MAML")
    print(f"  {n_meta_steps} meta-steps x ({k_inner} inner + 1 outer)")
    print(f"{'='*60}")

    model = make_gpt()
    model.load_state_dict(init_state)
    meta_opt = torch.optim.AdamW(
        model.parameters(), lr=meta_lr, weight_decay=0.01)

    maml_log = []

    for ms in range(n_meta_steps):
        # --- Save meta-parameters ---
        saved = {k: v_.clone() for k, v_ in model.state_dict().items()}

        # --- Sample inner and outer tasks ---
        inner_id = int(rng.choice(train_ids))
        outer_candidates = [r for r in train_ids if r != inner_id]
        outer_id = int(rng.choice(outer_candidates))

        # --- Inner loop: K NTP steps on inner task ---
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

        # --- Outer: NTP loss on different task, get gradient at theta' ---
        model.zero_grad()
        x_out, y_out = sample_batch(all_train_data[outer_id], batch_size)
        _, outer_loss = model(x_out, y_out)
        outer_loss.backward()
        outer_grads = [p.grad.clone() for p in model.parameters()]

        # --- Restore theta and apply meta-update ---
        model.load_state_dict(saved)
        meta_opt.zero_grad()
        for p, g in zip(model.parameters(), outer_grads):
            p.grad = g
        meta_opt.step()

        del saved, inner_opt, outer_grads
        torch.cuda.empty_cache()

        # --- Logging ---
        entry = {
            "step": ms,
            "outer_loss": float(outer_loss.item()),
            "inner_loss_avg": inner_loss_sum / k_inner,
            "inner_rule": inner_id,
            "outer_rule": outer_id,
        }

        if ms % train_eval_interval == 0 or ms == n_meta_steps - 1:
            ev_id = int(rng.choice(train_ids))
            vl = eval_val_loss(model, ev_id)
            entry["eval_val_loss"] = vl
            entry["eval_rule"] = ev_id
            print(f"  ms={ms:3d}: outer={outer_loss.item():.4f} "
                  f"inner_avg={inner_loss_sum/k_inner:.4f} "
                  f"eval_vl={vl:.4f} (rule {ev_id})")

        maml_log.append(entry)

    maml_state = {k: v_.cpu().clone()
                  for k, v_ in model.state_dict().items()}
    del model, meta_opt
    torch.cuda.empty_cache()
    print("  MAML training done.\n")

    # ==================================================================
    # CONDITION 2: Multi-task (NTP on interleaved rule sets)
    # Update-matched: same number of model updates as MAML meta-steps
    # ==================================================================
    print(f"{'='*60}")
    print(f"  CONDITION 2: MULTI-TASK ({baseline_steps} NTP steps, "
          f"{n_train_rules} rule sets)")
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
                "step": step,
                "train_loss": float(loss.item()),
                "eval_val_loss": vl,
                "eval_rule": ev_id,
            })
            print(f"  step={step:5d}: train={loss.item():.4f} "
                  f"eval_vl={vl:.4f} (rule {ev_id})")

    multi_state = {k: v_.cpu().clone()
                   for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Multi-task training done.\n")

    # ==================================================================
    # CONDITION 3: Single-task (NTP on rule set 0)
    # Update-matched: same number of model updates as MAML meta-steps
    # ==================================================================
    print(f"{'='*60}")
    print(f"  CONDITION 3: SINGLE-TASK ({baseline_steps} NTP steps, rule 0)")
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
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": vl,
            })
            print(f"  step={step:5d}: train={loss.item():.4f} "
                  f"val={vl:.4f}")

    single_state = {k: v_.cpu().clone()
                    for k, v_ in model.state_dict().items()}
    del model, opt
    torch.cuda.empty_cache()
    print("  Single-task training done.\n")

    # ==================================================================
    # EVALUATION: Fine-tune on held-out rule sets
    # ==================================================================
    print(f"{'='*60}")
    print(f"  EVALUATION: Fine-tune on {n_eval_rules} held-out rule sets")
    print(f"  K_eval={k_eval}, checkpoints every {eval_ckpt_interval}")
    print(f"{'='*60}")

    conditions = [
        ("MAML", maml_state),
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

            # Step 0: before fine-tuning
            pl = eval_per_level(model, eval_id)
            vl = eval_val_loss(model, eval_id)
            checkpoints.append({"step": 0, "val_loss": vl, "per_level": pl})

            # Fine-tune
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
                        "step": step, "val_loss": vl, "per_level": pl,
                    })

            # Per-layer eta^2 at final step only
            eta2 = eval_per_layer_eta2(model, eval_id)
            checkpoints[-1]["per_layer_eta2"] = eta2

            eval_results[cond_name][rule_key] = checkpoints

            # Print this rule's trajectory
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
    # Summary: average per-level loss across eval rules
    # ==================================================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    summary = {}
    for cond_name in ["MAML", "Multi", "Single"]:
        cond_data = eval_results[cond_name]
        all_steps = sorted(set(
            ck["step"]
            for rd in cond_data.values()
            for ck in rd
        ))
        step_avgs = {}
        for st in all_steps:
            level_sums = {}
            level_counts = {}
            overall_sums = []
            for rd in cond_data.values():
                for ck in rd:
                    if ck["step"] != st:
                        continue
                    overall_sums.append(
                        ck["per_level"]["overall_loss"])
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

    # Final per-level loss table
    print(f"\n  Per-level loss after {k_eval} fine-tuning steps "
          f"(avg over {n_eval_rules} eval rules):")
    header = f"  {'Cond':>8s}"
    for lv in range(L):
        header += f"  {'L'+str(lv):>7s}"
    header += f"  {'overall':>8s}"
    print(header)

    final_step = max(summary["MAML"].keys())
    for cond_name in ["MAML", "Multi", "Single"]:
        final = summary[cond_name][final_step]
        row = f"  {cond_name:>8s}"
        for lv in range(L):
            val = final["per_level"].get(lv, uniform)
            row += f"  {val:7.4f}"
        row += f"  {final['overall']:8.4f}"
        print(row)

    # Transfer gap: Single - MAML (positive = MAML better)
    print(f"\n  Transfer gap (Single - MAML, positive = MAML better):")
    header = f"  {'step':>6s}"
    for lv in range(min(5, L)):
        header += f"  {'L'+str(lv):>7s}"
    print(header)
    for st in sorted(summary["MAML"].keys()):
        row = f"  {st:6d}"
        for lv in range(min(5, L)):
            s_val = summary["Single"][st]["per_level"].get(lv, uniform)
            m_val = summary["MAML"][st]["per_level"].get(lv, uniform)
            gap = s_val - m_val
            row += f"  {gap:+7.4f}"
        print(row)

    # Learning curves for L0, L1, L2
    print(f"\n  Learning curves (per-level loss vs fine-tuning step):")
    for lv in range(min(4, L)):
        print(f"\n  Level {lv}:")
        header = f"  {'step':>6s}"
        for cn in ["MAML", "Multi", "Single"]:
            header += f"  {cn:>8s}"
        print(header)
        for st in sorted(summary["MAML"].keys()):
            row = f"  {st:6d}"
            for cn in ["MAML", "Multi", "Single"]:
                val = summary[cn][st]["per_level"].get(lv, uniform)
                row += f"  {val:8.4f}"
            print(row)

    # ==================================================================
    # Save results and model checkpoints
    # ==================================================================
    save_dir = f"{DATA_DIR}/rhm_meta_learning"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "n_params": n_params,
            "n_train_rules": n_train_rules,
            "n_eval_rules": n_eval_rules,
            "n_tokens_per_rule": n_tokens_per_rule,
            "n_meta_steps": n_meta_steps,
            "k_inner": k_inner,
            "inner_lr": inner_lr,
            "meta_lr": meta_lr,
            "k_eval": k_eval,
            "eval_ckpt_interval": eval_ckpt_interval,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "maml_flops": maml_flops,
            "baseline_steps": baseline_steps,
            "matching": "update-matched (same model updates)",
            "uniform_baseline": uniform,
        },
        "training": {
            "MAML": {"log": maml_log},
            "Multi": {"log": multi_log},
            "Single": {"log": single_log},
        },
        "evaluation": eval_results,
        "summary": summary,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    import torch as _torch
    _torch.save(maml_state, os.path.join(save_dir, "maml_model.pt"))
    _torch.save(multi_state, os.path.join(save_dir, "multi_model.pt"))
    _torch.save(single_state, os.path.join(save_dir, "single_model.pt"))

    volume.commit()
    print(f"\n  Saved to {save_dir}/")
    return result


@app.local_entrypoint()
def main():
    result = rhm_meta_learning.remote()
    print("\nRHM meta-learning experiment complete.")
