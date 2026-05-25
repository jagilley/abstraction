"""Pipeline stages 1-8 for the language reduction experiment.

Extracted from modal_app.py. Each function is a Modal remote function
that reads/writes to the shared volume.
"""

from language_reduction.shared import app, volume, DATA_DIR, image


# ---------------------------------------------------------------------------
# Stage 1: Tokenize FineWeb
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=7200,
    memory=16384,
)
def tokenize(
    n_tokens: int = 1_000_000_000,
    shard_size: int = 10_000_000,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
):
    from language_reduction.tokenize_data import tokenize_fineweb

    tokens_dir = f"{DATA_DIR}/tokens"
    tokenize_fineweb(tokens_dir, dataset_name, dataset_config, n_tokens, shard_size)
    volume.commit()


# ---------------------------------------------------------------------------
# Stage 2: Vocabulary reduction + covariance matrices
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=7200,
    memory=32768,
)
def compute_stats(v_prime: int = 3200, n_lags: int = 50, n_shards: int | None = None):
    from language_reduction.statistics import compute_vocab_reduction, compute_covariance_matrices

    tokens_dir = f"{DATA_DIR}/tokens"
    stats_dir = f"{DATA_DIR}/stats"

    print("=== Computing vocabulary reduction ===")
    compute_vocab_reduction(tokens_dir, stats_dir, v_prime, n_shards)
    volume.commit()

    print("\n=== Computing covariance matrices ===")
    compute_covariance_matrices(tokens_dir, stats_dir, n_lags, n_shards)
    volume.commit()


# ---------------------------------------------------------------------------
# Stage 3: Spectral decomposition
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=1800,
    memory=16384,
)
def spectral(n_lags: int = 50, max_rank: int = 64):
    from language_reduction.statistics import spectral_decomposition

    stats_dir = f"{DATA_DIR}/stats"
    print("=== Spectral decomposition ===")
    spectral_decomposition(stats_dir, n_lags, max_rank)
    volume.commit()


# ---------------------------------------------------------------------------
# Stage 4: Denoise corpus (parallel GPU workers)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=1800,
    memory=16384,
)
def denoise_one_shard(shard_idx: int, tau: float, n_lags: int = 10,
                      kappa_mode: str = "energy"):
    """Denoise a single shard on GPU. Called in parallel by denoise()."""
    import numpy as np
    from language_reduction.denoise import denoise_shard, build_inverse_vocab_map

    stats_dir = f"{DATA_DIR}/stats"
    vocab_map = np.load(f"{stats_dir}/vocab_map.npy")
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    inv_vocab_map = build_inverse_vocab_map(vocab_map, top_ids)

    tokens = np.load(f"{DATA_DIR}/tokens/shard_{shard_idx:05d}.npy")
    print(f"Shard {shard_idx} ({len(tokens):,} tokens, tau={tau}, kappa_mode={kappa_mode})")

    denoised = denoise_shard(tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags,
                             kappa_mode=kappa_mode)

    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    out_dir = f"{DATA_DIR}/denoised/tau_{tau:.3f}{suffix}"
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/shard_{shard_idx:05d}.npy", denoised)
    volume.commit()
    print(f"Shard {shard_idx} done")
    return shard_idx


@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=8192,
)
def denoise(tau: float = 0.5, n_lags: int = 10, max_shards: int | None = None,
            kappa_mode: str = "energy"):
    """Denoise corpus by spawning parallel GPU workers per shard."""
    import os, glob, numpy as np

    tokens_dir = f"{DATA_DIR}/tokens"
    shard_paths = sorted(glob.glob(os.path.join(tokens_dir, "shard_*.npy")))
    n_shards = len(shard_paths)
    if max_shards is not None:
        n_shards = min(n_shards, max_shards)

    print(f"=== Denoising {n_shards} shards at tau={tau} (kappa_mode={kappa_mode}, parallel GPU) ===")

    handles = []
    for i in range(n_shards):
        handles.append(denoise_one_shard.spawn(i, tau, n_lags, kappa_mode))

    for h in handles:
        h.get()

    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    out_dir = f"{DATA_DIR}/denoised/tau_{tau:.3f}{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    meta = np.load(os.path.join(tokens_dir, "meta.npy"), allow_pickle=True).item()
    meta["tau"] = tau
    meta["kappa_mode"] = kappa_mode
    meta["n_shards_denoised"] = n_shards
    np.save(os.path.join(out_dir, "meta.npy"), meta)
    volume.commit()
    print(f"All {n_shards} shards denoised at tau={tau} (kappa_mode={kappa_mode})")


# ---------------------------------------------------------------------------
# Stage 4v: Vocab-only reduction (no spectral denoising)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=7200,
    memory=32768,
)
def vocab_reduce(tau: float = 0.3, max_shards: int | None = None):
    """Apply vocab reduction only — map out-of-vocab tokens to nearest neighbors.

    tau controls the replacement rate: the vocabulary is sized so that
    out-of-vocab tokens account for approximately tau of the corpus.
    Output goes to /data/vocab_only/tau_{tau:.3f}/.
    """
    from language_reduction.denoise import vocab_reduce_corpus

    tokens_dir = f"{DATA_DIR}/tokens"
    stats_dir = f"{DATA_DIR}/stats"
    output_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}"

    print(f"=== Vocab-only reduction (tau={tau}) ===")
    vocab_reduce_corpus(tokens_dir, stats_dir, output_dir, tau=tau,
                        max_shards=max_shards)
    volume.commit()


# ---------------------------------------------------------------------------
# Stage 5: Measure beta
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def measure_beta_stage(tau: float = 0.0, n_lags: int = 10,
                       kappa_mode: str = "energy", mode: str = "spectral"):
    import json, os
    from language_reduction.measure import measure_beta
    from language_reduction.shared import resolve_mode_key

    mode_key = resolve_mode_key(mode, tau, kappa_mode)

    if tau == 0.0:
        stats_dir = f"{DATA_DIR}/stats"
    elif mode == "vocab_only":
        from language_reduction.statistics import compute_covariance_matrices
        data_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}"
        stats_dir = f"{data_dir}/stats_beta"
        last_lag = os.path.join(stats_dir, "covariance", f"C_lag_{n_lags:03d}.npy")
        if not os.path.exists(last_lag):
            os.makedirs(stats_dir, exist_ok=True)
            import shutil
            for f in ["vocab_map.npy", "freq.npy", "top_ids.npy"]:
                src = f"{DATA_DIR}/stats/{f}"
                if os.path.exists(src):
                    shutil.copy2(src, f"{stats_dir}/{f}")
            compute_covariance_matrices(data_dir, stats_dir, n_lags)
    else:
        from language_reduction.statistics import compute_covariance_matrices
        suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
        denoised_dir = f"{DATA_DIR}/denoised/tau_{tau:.3f}{suffix}"
        stats_dir = f"{DATA_DIR}/stats_denoised/tau_{tau:.3f}{suffix}"
        os.makedirs(stats_dir, exist_ok=True)
        import shutil
        for f in ["vocab_map.npy", "freq.npy", "top_ids.npy"]:
            src = f"{DATA_DIR}/stats/{f}"
            if os.path.exists(src):
                shutil.copy2(src, f"{stats_dir}/{f}")
        compute_covariance_matrices(denoised_dir, stats_dir, n_lags)

    result = measure_beta(stats_dir, n_lags)

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/beta_{mode_key.replace('/', '_')}.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return result


# ---------------------------------------------------------------------------
# Stage 6: Train AR model
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def train_model(
    tau: float = 0.0,
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    batch_size: int = 64,
    lr: float = 3e-4,
    n_steps: int = 10_000,
    use_rope: bool = False,
    eval_interval: int = 500,
    n_eval_batches: int = 5,
    kappa_mode: str = "energy",
    tie_weights: bool = False,
    save_suffix: str = "",
    mode: str = "spectral",
):
    import os, json, glob
    import torch
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.shared import resolve_data_dir, resolve_mode_key

    mode_key = resolve_mode_key(mode, tau, kappa_mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}: mode={mode}, tau={tau}, P={n_tokens:,}, T={block_size}")

    data_dir = resolve_data_dir(mode, tau, kappa_mode)

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
    print(f"Loaded {len(data):,} tokens")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd, use_rope, tie_weights=tie_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(split_data, bs, bl):
        ix = torch.randint(len(split_data) - bl - 1, (bs,))
        x = torch.stack([split_data[i : i + bl] for i in ix])
        y = torch.stack([split_data[i + 1 : i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    losses = []
    val_losses = []
    best_val_loss = float("inf")

    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vl_accum = 0.0
                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data, batch_size, block_size)
                    _, vl = model(vx, vy)
                    vl_accum += float(vl)
                val_loss = vl_accum / n_eval_batches
            losses.append((step, float(loss)))
            val_losses.append((step, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print(f"  step {step:6d}: train={loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

    # compute n-gram losses for gamma estimation
    print("Computing n-gram losses...")
    model.eval()
    ngram_losses = {}  # n -> average loss at position n
    n_eval_batches = min(100, len(val_data) // (batch_size * block_size))
    per_pos_accum = torch.zeros(block_size - 1, device=device)
    per_pos_count = 0

    with torch.no_grad():
        for _ in range(n_eval_batches):
            x, _ = get_batch(val_data, batch_size, block_size)
            per_pos = model.get_ngram_losses(x)  # (B, T-1)
            per_pos_accum += per_pos.mean(dim=0)
            per_pos_count += 1

    per_pos_avg = (per_pos_accum / per_pos_count).cpu().numpy()
    # L_n = average loss at position n (0-indexed, so position 0 is the 1-gram loss)
    for n in range(len(per_pos_avg)):
        ngram_losses[str(n + 1)] = float(per_pos_avg[n])

    # save
    tied_suffix = "_tied" if tie_weights else ""
    save_dir = f"{DATA_DIR}/models/{mode_key}/P_{n_tokens}/T_{block_size}{tied_suffix}{save_suffix}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    result = {
        "mode": mode,
        "tau": tau,
        "kappa_mode": kappa_mode,
        "n_tokens": n_tokens,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "use_rope": use_rope,
        "tie_weights": tie_weights,
        "final_train_loss": float(losses[-1][1]),
        "final_val_loss": float(val_losses[-1][1]),
        "best_val_loss": best_val_loss,
        "train_curve": losses,
        "val_curve": val_losses,
        "ngram_losses": ngram_losses,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    volume.commit()
    print(f"Model saved to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage 7: Measure gamma
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=1800,
    memory=16384,
)
def measure_gamma_stage(tau: float = 0.0, kappa_mode: str = "energy",
                        mode: str = "spectral"):
    import os, json, glob
    from language_reduction.measure import measure_gamma_from_ngram_losses
    from language_reduction.shared import resolve_mode_key

    mode_key = resolve_mode_key(mode, tau, kappa_mode)
    models_dir = f"{DATA_DIR}/models/{mode_key}"
    if not os.path.exists(models_dir):
        print(f"No models found at {models_dir}")
        return None

    loss_curves = {}
    for p_dir in sorted(glob.glob(os.path.join(models_dir, "P_*"))):
        for t_dir in sorted(glob.glob(os.path.join(p_dir, "T_*"))):
            results_path = os.path.join(t_dir, "results.json")
            if not os.path.exists(results_path):
                continue
            with open(results_path) as f:
                res = json.load(f)
            P = res["n_tokens"]
            for n_str, loss_val in res["ngram_losses"].items():
                if n_str not in loss_curves:
                    loss_curves[n_str] = []
                loss_curves[n_str].append((P, loss_val))

    result = measure_gamma_from_ngram_losses(loss_curves)
    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/gamma_{mode_key.replace('/', '_')}.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    return result


# ---------------------------------------------------------------------------
# Stage 8: Validate predicted vs empirical scaling
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=1800,
    memory=16384,
)
def validate_stage(tau: float = 0.0, kappa_mode: str = "energy",
                    mode: str = "spectral"):
    import os, json, glob
    from language_reduction.measure import predict_alpha_D, measure_empirical_scaling
    from language_reduction.shared import resolve_mode_key

    mode_key = resolve_mode_key(mode, tau, kappa_mode)
    results_dir = f"{DATA_DIR}/results"
    file_key = mode_key.replace("/", "_")

    beta_path = f"{results_dir}/beta_{file_key}.json"
    with open(beta_path) as f:
        beta_result = json.load(f)
    beta = beta_result["beta"]

    gamma_path = f"{results_dir}/gamma_{file_key}.json"
    with open(gamma_path) as f:
        gamma_result = json.load(f)
    gamma = gamma_result["gamma"]

    alpha_predicted = predict_alpha_D(beta, gamma)

    models_dir = f"{DATA_DIR}/models/{mode_key}"
    loss_vs_P = []
    for p_dir in sorted(glob.glob(os.path.join(models_dir, "P_*"))):
        for t_dir in sorted(glob.glob(os.path.join(p_dir, "T_*"))):
            results_path = os.path.join(t_dir, "results.json")
            if not os.path.exists(results_path):
                continue
            with open(results_path) as f:
                res = json.load(f)
            loss_vs_P.append((res["n_tokens"], res.get("best_val_loss", res["final_val_loss"])))

    empirical = measure_empirical_scaling(loss_vs_P)

    summary = {
        "tau": tau,
        "mode": mode,
        "kappa_mode": kappa_mode,
        "beta": beta,
        "gamma": gamma,
        "alpha_predicted": alpha_predicted,
        "alpha_empirical": empirical["alpha_empirical"],
        "r2_empirical": empirical["r2"],
    }
    print(f"\n=== Results for {mode_key} ===")
    print(f"  beta  = {beta:.4f}")
    print(f"  gamma = {gamma:.4f}")
    print(f"  alpha_predicted  = {alpha_predicted:.4f}")
    print(f"  alpha_empirical  = {empirical['alpha_empirical']:.4f} (R²={empirical['r2']:.4f})")

    with open(f"{results_dir}/summary_{file_key}.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    return summary


# ---------------------------------------------------------------------------
# Inspect: spot-check denoising on a small slice (no full shard needed)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def inspect_output(tau: float = 0.3, start: int = 0, length: int = 500):
    """Denoise a small token slice on-the-fly and show original vs denoised."""
    import tiktoken
    import numpy as np
    from language_reduction.denoise import denoise_shard, build_inverse_vocab_map

    enc = tiktoken.get_encoding("gpt2")
    stats_dir = f"{DATA_DIR}/stats"
    vocab_map = np.load(f"{stats_dir}/vocab_map.npy")
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    inv_vocab_map = build_inverse_vocab_map(vocab_map, top_ids)

    # take a small window with context padding for the lags
    n_lags = 10
    pad = n_lags
    full_tokens = np.load(f"{DATA_DIR}/tokens/shard_00000.npy")
    lo = max(0, start - pad)
    hi = min(len(full_tokens), start + length + pad)
    chunk = full_tokens[lo:hi]

    print(f"Denoising {len(chunk)} tokens (slice [{lo}:{hi}]) at tau={tau}...")
    denoised_chunk = denoise_shard(
        chunk, vocab_map, inv_vocab_map, stats_dir, tau, n_lags, batch_size=10000
    )

    # trim padding
    inner_lo = start - lo
    inner_hi = inner_lo + length
    orig = chunk[inner_lo:inner_hi]
    den = denoised_chunk[inner_lo:inner_hi]
    diff_mask = orig != den
    n_diff = int(diff_mask.sum())

    print(f"\n=== ORIGINAL (tokens {start}..{start+length}) ===")
    print(enc.decode(orig.tolist()))
    print(f"\n=== DENOISED (tau={tau}, {n_diff}/{length} changed) ===")
    print(enc.decode(den.tolist()))
    print(f"\n=== SAMPLE REPLACEMENTS ===")
    for i in np.where(diff_mask)[0][:30]:
        o_tok = enc.decode([int(orig[i])])
        d_tok = enc.decode([int(den[i])])
        print(f"  pos {start+i}: '{o_tok}' -> '{d_tok}'")


# ---------------------------------------------------------------------------
# Replacement census: which tokens get replaced at a given tau?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def replacement_census_stage(tau: float = 0.3, max_shards: int | None = None,
                              mode: str = "spectral"):
    import json, os
    from language_reduction.replacement_census import replacement_census
    from language_reduction.shared import NumpyEncoder, resolve_mode_key, resolve_data_dir

    tokens_dir = f"{DATA_DIR}/tokens"
    mode_key = resolve_mode_key(mode, tau)
    denoised_dir = resolve_data_dir(mode, tau)

    print(f"=== Replacement census for {mode_key} ===")
    result = replacement_census(tokens_dir, denoised_dir, max_shards)

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    file_key = mode_key.replace("/", "_")
    with open(f"{results_dir}/replacement_census_{file_key}.json", "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/replacement_census_{file_key}.json")
    return result


# ---------------------------------------------------------------------------
# Compound stages
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=7200,
    memory=16384,
)
def train_scaling_sweep(
    tau: float = 0.0,
    P_values: str = "100000,1000000,10000000,100000000",
    block_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    kappa_mode: str = "energy",
    mode: str = "spectral",
):
    """Kick off training runs at multiple dataset sizes for a single tau.

    Steps auto-scale with P: ~5 epochs, floor 2000, cap 20000.
    """
    P_list = [int(x) for x in P_values.split(",")]
    tokens_per_step = 64 * block_size

    handles = []
    for P in P_list:
        n_steps = min(max(5 * P // tokens_per_step, 2000), 20000)
        print(f"Launching: mode={mode}, tau={tau}, P={P:,}, steps={n_steps}")
        h = train_model.spawn(
            tau=tau, n_tokens=P, block_size=block_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_steps=n_steps, kappa_mode=kappa_mode, mode=mode,
        )
        handles.append((P, h))

    results = []
    for P, h in handles:
        r = h.get()
        results.append(r)
        print(f"  P={P:>12,}: best_val={r['best_val_loss']:.4f} final_val={r['final_val_loss']:.4f}")

    print(f"\nAll {len(results)} runs complete for tau={tau} (kappa_mode={kappa_mode})")
    return results
