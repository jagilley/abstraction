"""Comprehensive scaling sweep: cross-τ evaluation, parameter-limitation check,
and denser P-value curves.

Cross-τ evaluation direction: we evaluate τ=0 models on denoised val data
(not τ>0 models on natural data). A τ>0 model has randomly-initialized
embeddings for tokens eliminated by denoising, so evaluating it on natural
text produces garbage loss at those positions, contaminating the metric.
The τ=0 model has functional embeddings for all tokens, making it a clean
evaluator on any distribution.

New stages:
  cross-eval           — evaluate a trained model on data from a different τ
  scaling-sweep-v2     — train 7 P values at one τ with finer eval settings
  cross-eval-sweep     — evaluate τ=0 models (all P) on each τ's val data
  param-check          — train a larger model at P=100M to check for parameter limitation
  comprehensive-sweep  — orchestrate the full experiment
"""

from language_reduction.shared import app, volume, DATA_DIR, NumpyEncoder
from language_reduction.experiments.pipeline.stages import train_model


# ---------------------------------------------------------------------------
# Cross-τ evaluation
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=1800,
    memory=32768,
)
def cross_eval_model(
    model_tau: float,
    model_P: int,
    eval_tau: float,
    block_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    n_eval_batches: int = 50,
    batch_size: int = 64,
    kappa_mode: str = "energy",
    tie_weights: bool = False,
    save_suffix: str = "",
):
    """Evaluate a trained model on held-out data from a (possibly different) τ.

    For eval_tau=0, uses shard 10 of the original corpus (10M tokens never
    seen by any model). For eval_tau>0, uses the last shard of that τ's
    denoised corpus.
    """
    import os
    import torch
    import numpy as np
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    tied_suffix = "_tied" if tie_weights else ""
    model_dir = (f"{DATA_DIR}/models/tau_{model_tau:.3f}{suffix}"
                 f"/P_{model_P}/T_{block_size}{tied_suffix}{save_suffix}")

    if not os.path.exists(os.path.join(model_dir, "model.pt")):
        print(f"No model at {model_dir}")
        return None

    meta_path = f"{DATA_DIR}/tokens/meta.npy"
    meta = np.load(meta_path, allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd,
                tie_weights=tie_weights).to(device)
    state = torch.load(os.path.join(model_dir, "model.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    if eval_tau == 0.0:
        eval_path = f"{DATA_DIR}/tokens/shard_00010.npy"
    else:
        eval_dir = f"{DATA_DIR}/denoised/tau_{eval_tau:.3f}{suffix}"
        eval_path = os.path.join(eval_dir, "shard_00009.npy")

    if not os.path.exists(eval_path):
        print(f"Eval data not found: {eval_path}")
        return None

    eval_data = torch.from_numpy(np.load(eval_path).astype(np.int64))
    print(f"Cross-eval: model(τ={model_tau}, P={model_P:,}{save_suffix}) "
          f"on eval(τ={eval_tau}), {len(eval_data):,} tokens")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i + bl] for i in ix])
        y = torch.stack([data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_eval_batches):
            x, y = get_batch(eval_data, batch_size, block_size)
            _, loss = model(x, y)
            total_loss += float(loss)

    avg_loss = total_loss / n_eval_batches
    print(f"  CE loss = {avg_loss:.4f}")

    return {
        "model_tau": model_tau,
        "model_P": model_P,
        "eval_tau": eval_tau,
        "cross_eval_loss": avg_loss,
        "n_eval_batches": n_eval_batches,
        "model_config": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "block_size": block_size, "save_suffix": save_suffix,
        },
    }


# ---------------------------------------------------------------------------
# Scaling sweep v2: 7 P values with finer eval
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    timeout=14400,
    memory=16384,
)
def scaling_sweep_v2(
    tau: float = 0.0,
    P_values: str = "100000,300000,1000000,3000000,10000000,30000000,100000000",
    block_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    kappa_mode: str = "energy",
    save_suffix: str = "",
):
    """Train models at 7 P values with ~40 eval checkpoints and 20 eval batches."""
    P_list = [int(x) for x in P_values.split(",")]
    tokens_per_step = 64 * block_size

    handles = []
    for P in P_list:
        n_steps = min(max(5 * P // tokens_per_step, 2000), 20000)
        eval_interval = max(20, n_steps // 40)
        print(f"Launching: τ={tau}, P={P:,}, steps={n_steps}, "
              f"eval_interval={eval_interval}{save_suffix}")
        h = train_model.spawn(
            tau=tau, n_tokens=P, block_size=block_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_steps=n_steps, eval_interval=eval_interval,
            n_eval_batches=20, kappa_mode=kappa_mode,
            save_suffix=save_suffix,
        )
        handles.append((P, h))

    results = []
    for P, h in handles:
        r = h.get()
        results.append(r)
        print(f"  P={P:>12,}: best_val={r['best_val_loss']:.4f} "
              f"final_val={r['final_val_loss']:.4f}")

    print(f"\nAll {len(results)} runs complete for τ={tau}")
    return results


# ---------------------------------------------------------------------------
# Cross-eval sweep: evaluate models on higher-τ val data (valid triangle)
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    timeout=14400,
    memory=16384,
)
def cross_eval_sweep(
    taus: str = "0.0,0.1,0.3,0.5,0.7",
    P_values: str = "100000,300000,1000000,3000000,10000000,30000000,100000000",
    block_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    kappa_mode: str = "energy",
):
    """Cross-evaluate models on data from equal-or-higher τ values.

    A model trained at τ=X can be cleanly evaluated on τ=Y data for Y >= X.
    The τ=Y distribution only contains patterns that survive at τ=Y, and
    since Y >= X, all those patterns also survive at τ=X, so the model has
    functional embeddings for every token in the eval set.

    The reverse (Y < X) is invalid: the eval set may contain tokens whose
    embeddings are random noise in the model trained at higher τ.

    This produces a triangular matrix of (model_τ, eval_τ, P) results.
    """
    tau_list = sorted(float(t) for t in taus.split(","))
    P_list = [int(x) for x in P_values.split(",")]

    handles = []
    for model_tau in tau_list:
        for eval_tau in tau_list:
            if eval_tau < model_tau - 1e-6:
                continue
            if abs(model_tau - eval_tau) < 1e-6:
                continue  # within-τ eval already done during training
            for P in P_list:
                h = cross_eval_model.spawn(
                    model_tau=model_tau, model_P=P, eval_tau=eval_tau,
                    block_size=block_size, n_layer=n_layer, n_head=n_head,
                    n_embd=n_embd, kappa_mode=kappa_mode,
                )
                handles.append((model_tau, eval_tau, P, h))

    n_jobs = len(handles)
    print(f"Launched {n_jobs} cross-eval jobs (triangular: model_τ ≤ eval_τ)")

    results = []
    for model_tau, eval_tau, P, h in handles:
        r = h.get()
        if r is not None:
            results.append(r)
            print(f"  model(τ={model_tau}) on eval(τ={eval_tau}), "
                  f"P={P:>12,}: loss={r['cross_eval_loss']:.4f}")
        else:
            print(f"  model(τ={model_tau}) on eval(τ={eval_tau}), "
                  f"P={P:>12,}: SKIPPED")

    return results


# ---------------------------------------------------------------------------
# Parameter-limitation check
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    timeout=14400,
    memory=16384,
)
def param_limit_check(
    taus: str = "0.0,0.1,0.3,0.5,0.7",
    P: int = 100_000_000,
    block_size: int = 128,
    kappa_mode: str = "energy",
):
    """Train a larger model at P=100M for each τ, then cross-eval.

    Compares 2L/128D (~400K non-embed params) vs 4L/256D (~3.2M).
    """
    tau_list = [float(t) for t in taus.split(",")]
    tokens_per_step = 64 * block_size
    n_steps = min(max(5 * P // tokens_per_step, 2000), 20000)
    eval_interval = max(20, n_steps // 40)
    suffix = "_L4_D256"

    train_handles = []
    for tau in tau_list:
        print(f"Launching param check: τ={tau}, P={P:,}, 4L/256D")
        h = train_model.spawn(
            tau=tau, n_tokens=P, block_size=block_size,
            n_layer=4, n_head=8, n_embd=256,
            n_steps=n_steps, eval_interval=eval_interval,
            n_eval_batches=20, kappa_mode=kappa_mode,
            save_suffix=suffix,
        )
        train_handles.append((tau, h))

    train_results = []
    for tau, h in train_handles:
        r = h.get()
        train_results.append(r)
        print(f"  τ={tau}: best_val={r['best_val_loss']:.4f}")

    return {"train": train_results}


# ---------------------------------------------------------------------------
# Comprehensive sweep: flat-spawn to respect GPU concurrency limits
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    timeout=43200,
    memory=16384,
)
def comprehensive_scaling_sweep(
    taus: str = "0.0,0.1,0.3,0.5,0.7",
    P_values: str = "100000,300000,1000000,3000000,10000000,30000000,100000000",
    block_size: int = 128,
    kappa_mode: str = "energy",
    include_param_check: bool = True,
):
    """Run Phase 1 (7×5 scaling curves) + Phase 3 (param check) as one job.

    All GPU work is spawned as a flat list — Modal queues jobs beyond the
    concurrency limit (10 GPUs). Longest jobs are spawned first so they
    start immediately and short jobs backfill freed slots.
    """
    import json, os
    from language_reduction.measure import measure_empirical_scaling

    tau_list = [float(t) for t in taus.split(",")]
    P_list = [int(x) for x in P_values.split(",")]
    tokens_per_step = 64 * block_size

    # --- Build flat job list ---
    jobs = []
    for tau in tau_list:
        for P in P_list:
            n_steps = min(max(5 * P // tokens_per_step, 2000), 20000)
            jobs.append({
                "tau": tau, "P": P,
                "n_layer": 2, "n_head": 4, "n_embd": 128,
                "n_steps": n_steps,
                "save_suffix": "",
                "phase": "phase1",
            })
    if include_param_check:
        for tau in tau_list:
            n_steps = min(max(5 * 100_000_000 // tokens_per_step, 2000), 20000)
            jobs.append({
                "tau": tau, "P": 100_000_000,
                "n_layer": 4, "n_head": 8, "n_embd": 256,
                "n_steps": n_steps,
                "save_suffix": "_L4_D256",
                "phase": "phase3",
            })

    # Sort longest first so they grab GPUs immediately; short jobs backfill.
    def est_duration(j):
        layer_factor = j["n_layer"] * (j["n_embd"] / 128) ** 2
        return j["n_steps"] * layer_factor
    jobs.sort(key=est_duration, reverse=True)

    # --- Spawn all (Modal queues beyond concurrency limit) ---
    n_total = len(jobs)
    print(f"Spawning {n_total} training jobs "
          f"({len(tau_list)} τ × {len(P_list)} P + "
          f"{len(tau_list) if include_param_check else 0} param-check)")
    print(f"Modal will run up to 10 concurrently, queuing the rest.\n")

    handles = []
    for j in jobs:
        eval_interval = max(20, j["n_steps"] // 40)
        h = train_model.spawn(
            tau=j["tau"], n_tokens=j["P"], block_size=block_size,
            n_layer=j["n_layer"], n_head=j["n_head"], n_embd=j["n_embd"],
            n_steps=j["n_steps"], eval_interval=eval_interval,
            n_eval_batches=20, kappa_mode=kappa_mode,
            save_suffix=j["save_suffix"],
        )
        handles.append((j, h))
        print(f"  spawned: τ={j['tau']}, P={j['P']:>12,}, "
              f"{j['n_layer']}L/{j['n_embd']}D, {j['n_steps']} steps")

    # --- Collect results ---
    print(f"\nWaiting for {n_total} jobs...")
    phase1 = {str(tau): [] for tau in tau_list}
    phase3 = []
    n_done = 0

    for j, h in handles:
        try:
            r = h.get()
            n_done += 1
            tag = f"[{n_done}/{n_total}]"
            if j["phase"] == "phase1":
                phase1[str(j["tau"])].append(r)
                print(f"  {tag} τ={j['tau']}, P={j['P']:>12,}: "
                      f"best_val={r['best_val_loss']:.4f}")
            else:
                phase3.append(r)
                print(f"  {tag} τ={j['tau']}, P={j['P']:>12,} (4L/256D): "
                      f"best_val={r['best_val_loss']:.4f}")
        except Exception as e:
            n_done += 1
            print(f"  [{n_done}/{n_total}] FAILED: τ={j['tau']}, "
                  f"P={j['P']:,} — {e}")

    # Sort each τ's results by P for consistent ordering
    for tau_key in phase1:
        phase1[tau_key].sort(key=lambda r: r["n_tokens"])

    # --- Compute scaling exponents ---
    print("\n" + "=" * 60)
    print("SCALING EXPONENTS (within-τ, 7-point fits)")
    print("=" * 60)

    scaling = {}
    for tau in tau_list:
        tau_key = str(tau)
        training = phase1.get(tau_key, [])
        if len(training) < 3:
            print(f"  τ={tau}: only {len(training)} points, skipping fit")
            continue

        pairs = [(r["n_tokens"], r["best_val_loss"]) for r in training]
        fit = measure_empirical_scaling(pairs)

        scaling[tau_key] = {
            "tau": tau,
            "alpha_within": fit.get("alpha_empirical"),
            "alpha_within_r2": fit.get("r2"),
            "H_inf": fit.get("H_inf_empirical"),
            "n_points": len(training),
            "losses": {r["n_tokens"]: r["best_val_loss"] for r in training},
        }
        a = fit.get("alpha_empirical")
        r2 = fit.get("r2")
        print(f"  τ={tau}: α={a:.4f} (R²={r2:.4f}, {len(training)} points)")

    # --- Save ---
    all_results = {
        "taus": tau_list,
        "P_values": P_list,
        "phase1_training": phase1,
        "phase3_param_check": phase3,
        "scaling_exponents": scaling,
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    out_path = f"{results_dir}/comprehensive_scaling_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nResults saved to {out_path}")

    # --- Summary tables ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'τ':>5} | {'α_within':>10} | {'R²':>8} | {'H_inf':>8} | {'pts':>4}")
    print("-" * 45)
    for tau in tau_list:
        e = scaling.get(str(tau), {})
        if e:
            print(f"{tau:>5.1f} | {e['alpha_within']:>10.4f} | "
                  f"{e['alpha_within_r2']:>8.4f} | "
                  f"{e['H_inf']:>8.4f} | {e['n_points']:>4}")

    if phase3:
        print(f"\nParameter-limitation check (P=100M, within-τ val loss):")
        print(f"{'τ':>5} | {'2L/128D':>12} | {'4L/256D':>12} | {'Δ':>8} | {'regime':>12}")
        print("-" * 58)
        for r_large in sorted(phase3, key=lambda r: r["tau"]):
            tau = r_large["tau"]
            training = phase1.get(str(tau), [])
            r_small = next(
                (r for r in training if r["n_tokens"] == 100_000_000), None
            )
            if r_small:
                sl = r_small["best_val_loss"]
                ll = r_large["best_val_loss"]
                delta = sl - ll
                regime = "param-limited" if delta > 0.1 else "data-limited" if delta < 0.02 else "mixed"
                print(f"{tau:>5.1f} | {sl:>12.4f} | {ll:>12.4f} | "
                      f"{delta:>8.4f} | {regime:>12}")

    # --- Per-τ loss tables ---
    print(f"\nPer-P validation losses:")
    header = f"{'P':>12}"
    for tau in tau_list:
        header += f" | {'τ='+str(tau):>8}"
    print(header)
    print("-" * len(header))
    for P in P_list:
        row = f"{P:>12,}"
        for tau in tau_list:
            training = phase1.get(str(tau), [])
            r = next((r for r in training if r["n_tokens"] == P), None)
            if r:
                row += f" | {r['best_val_loss']:>8.4f}"
            else:
                row += f" | {'—':>8}"
        print(row)

    return all_results
