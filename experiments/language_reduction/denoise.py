"""Context-dependent spectral denoising of a token corpus.

For each token-occurrence, compute how much of its co-occurrence with
surrounding context is explained by the top-k subspace of PMI(n) versus
the residual. Flag contextually atypical occurrences and replace them
with the candidate that maximizes explained co-occurrence.

Supports GPU acceleration when torch+CUDA are available.
"""

import os
import glob
import numpy as np

from language_reduction.statistics import get_low_rank_factors, get_low_rank_factors_by_effective_rank


def _materialize_low_rank(stats_dir: str, lag: int, tau: float,
                          kappa_mode: str = "energy"):
    """Compute PMI_k(n) = U @ diag(S) @ Vt as a dense (v', v') matrix.

    kappa_mode="energy": truncate at (1-tau) energy fraction (legacy, tau does double duty).
    kappa_mode="effective_rank": truncate at the corpus-derived effective rank (tau unused here).
    """
    if kappa_mode == "effective_rank":
        U, S, Vt = get_low_rank_factors_by_effective_rank(stats_dir, lag)
    else:
        U, S, Vt = get_low_rank_factors(stats_dir, lag, tau)
    if len(S) == 0:
        v = U.shape[0]
        return np.zeros((v, v), dtype=np.float64)
    return (U * S[None, :]) @ Vt


def _load_matrices(stats_dir: str, n_lags: int, tau: float,
                   kappa_mode: str = "energy"):
    """Load full PMI and low-rank PMI_k matrices for all lags."""
    PMI_full = []
    PMI_low = []
    for lag in range(1, n_lags + 1):
        pmi = np.load(os.path.join(stats_dir, "pmi", f"PMI_lag_{lag:03d}.npy"))
        pmi_k = _materialize_low_rank(stats_dir, lag, tau, kappa_mode)
        PMI_full.append(pmi)
        PMI_low.append(pmi_k)
    return PMI_full, PMI_low


# ---------------------------------------------------------------------------
# GPU fast path
# ---------------------------------------------------------------------------

def _denoise_shard_gpu(shard_tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags,
                       kappa_mode="energy"):
    import torch
    device = torch.device("cuda")

    reduced_np = vocab_map[shard_tokens].astype(np.int32)
    v_prime = int(vocab_map.max()) + 1
    seq_len = len(reduced_np)

    PMI_full_np, PMI_low_np = _load_matrices(stats_dir, n_lags, tau, kappa_mode)

    reduced = torch.tensor(reduced_np, dtype=torch.long, device=device)
    PMI_full = [torch.tensor(m, dtype=torch.float32, device=device) for m in PMI_full_np]
    PMI_low = [torch.tensor(m, dtype=torch.float32, device=device) for m in PMI_low_np]

    # --- Phase 1: typicality scoring ---
    explained = torch.zeros(seq_len, dtype=torch.float32, device=device)
    total = torch.zeros(seq_len, dtype=torch.float32, device=device)

    for lag_idx in range(n_lags):
        lag = lag_idx + 1
        src = reduced[:-lag]
        dst = reduced[lag:]

        t_vals = PMI_full[lag_idx][src, dst]
        e_vals = PMI_low[lag_idx][src, dst]

        total[:-lag] += t_vals
        total[lag:] += t_vals
        explained[:-lag] += e_vals
        explained[lag:] += e_vals

    abs_total = total.abs()
    typicality = torch.where(abs_total > 1e-15, explained / abs_total, torch.ones_like(total))
    threshold = float(torch.quantile(typicality, tau))
    flagged_positions = torch.where(typicality < threshold)[0]
    n_flagged = len(flagged_positions)
    print(f"  Flagged {n_flagged:,}/{seq_len:,} ({100*n_flagged/seq_len:.1f}%)")

    if n_flagged == 0:
        return shard_tokens.copy()

    # free full PMI matrices — only need low-rank for replacement
    del PMI_full
    torch.cuda.empty_cache()

    # --- Phase 2: replacement scoring on GPU ---
    BATCH = 200_000
    best_all = torch.zeros(n_flagged, dtype=torch.long, device=device)

    for batch_start in range(0, n_flagged, BATCH):
        batch_end = min(batch_start + BATCH, n_flagged)
        batch_pos = flagged_positions[batch_start:batch_end]
        B = len(batch_pos)
        scores = torch.zeros((B, v_prime), dtype=torch.float32, device=device)

        for lag_idx in range(n_lags):
            lag = lag_idx + 1
            C_k = PMI_low[lag_idx]

            # forward: context at pos + lag
            fwd_mask = batch_pos + lag < seq_len
            fwd_idx = torch.where(fwd_mask)[0]
            if len(fwd_idx) > 0:
                fwd_ctx = reduced[batch_pos[fwd_idx] + lag]
                scores.index_add_(0, fwd_idx, C_k[:, fwd_ctx].T)

            # backward: context at pos - lag
            bwd_mask = batch_pos - lag >= 0
            bwd_idx = torch.where(bwd_mask)[0]
            if len(bwd_idx) > 0:
                bwd_ctx = reduced[batch_pos[bwd_idx] - lag]
                scores.index_add_(0, bwd_idx, C_k[bwd_ctx])

        best_all[batch_start:batch_end] = scores.argmax(dim=1)

        if (batch_start // BATCH) % 5 == 0:
            print(f"    replaced {batch_end:,}/{n_flagged:,}")

    flagged_np = flagged_positions.cpu().numpy()
    best_np = best_all.cpu().numpy().astype(np.int32)

    output = shard_tokens.copy()
    output[flagged_np] = inv_vocab_map[best_np]
    return output


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def _denoise_shard_cpu(shard_tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags,
                       batch_size=50_000, kappa_mode="energy"):
    reduced = vocab_map[shard_tokens].astype(np.int32)
    v_prime = int(vocab_map.max()) + 1
    seq_len = len(reduced)

    PMI_full, PMI_low = _load_matrices(stats_dir, n_lags, tau, kappa_mode)

    # Phase 1: typicality scoring
    explained = np.zeros(seq_len, dtype=np.float64)
    total = np.zeros(seq_len, dtype=np.float64)

    for lag_idx in range(n_lags):
        lag = lag_idx + 1
        src = reduced[:-lag]
        dst = reduced[lag:]

        t_vals = PMI_full[lag_idx][src, dst]
        e_vals = PMI_low[lag_idx][src, dst]

        total[:-lag] += t_vals
        total[lag:] += t_vals
        explained[:-lag] += e_vals
        explained[lag:] += e_vals

    eps = 1e-15
    abs_total = np.abs(total)
    typicality = np.where(abs_total > eps, explained / abs_total, 1.0)
    threshold = np.quantile(typicality, tau)
    flagged_positions = np.where(typicality < threshold)[0]
    n_flagged = len(flagged_positions)
    print(f"  Flagged {n_flagged:,}/{seq_len:,} ({100*n_flagged/seq_len:.1f}%)")

    if n_flagged == 0:
        return shard_tokens.copy()

    # Phase 2: replacement
    best_replacements = np.zeros(n_flagged, dtype=np.int32)
    for batch_start in range(0, n_flagged, batch_size):
        batch_end = min(batch_start + batch_size, n_flagged)
        batch_pos = flagged_positions[batch_start:batch_end]
        B = len(batch_pos)
        scores = np.zeros((B, v_prime), dtype=np.float64)

        for lag_idx in range(n_lags):
            lag = lag_idx + 1
            C_k = PMI_low[lag_idx]

            fwd_valid = batch_pos + lag < seq_len
            if fwd_valid.any():
                fwd_ctx = reduced[batch_pos[fwd_valid] + lag]
                scores[fwd_valid] += C_k[:, fwd_ctx].T

            bwd_valid = batch_pos - lag >= 0
            if bwd_valid.any():
                bwd_ctx = reduced[batch_pos[bwd_valid] - lag]
                scores[bwd_valid] += C_k[bwd_ctx, :]

        best_replacements[batch_start:batch_end] = np.argmax(scores, axis=1)

        if batch_end % (batch_size * 10) < batch_size:
            print(f"    replaced {batch_end:,}/{n_flagged:,}")

    output = shard_tokens.copy()
    output[flagged_positions] = inv_vocab_map[best_replacements]
    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_inverse_vocab_map(vocab_map: np.ndarray, top_ids: np.ndarray):
    """Build reduced_id -> original token ID mapping for the top-v' tokens."""
    v_prime = int(vocab_map.max()) + 1
    inv_map = np.zeros(v_prime, dtype=np.uint16)
    for orig_id in sorted(top_ids):
        r = vocab_map[orig_id]
        inv_map[r] = orig_id
    return inv_map


def denoise_shard(shard_tokens, vocab_map, inv_vocab_map, stats_dir, tau,
                  n_lags=50, batch_size=50_000, kappa_mode="energy"):
    """Denoise a single shard. Uses GPU if available, CPU otherwise.

    kappa_mode="energy": legacy — tau controls both SVD truncation and flagging threshold.
    kappa_mode="effective_rank": tau only controls flagging threshold; SVD truncation
        uses the corpus-derived effective rank.
    """
    if tau <= 0.0:
        return shard_tokens.copy()

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  Using GPU: {torch.cuda.get_device_name()}")
            return _denoise_shard_gpu(
                shard_tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags,
                kappa_mode,
            )
    except ImportError:
        pass

    print("  Using CPU (no CUDA available)")
    return _denoise_shard_cpu(
        shard_tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags, batch_size,
        kappa_mode,
    )


def vocab_reduce_shard(shard_tokens, vocab_map, inv_vocab_map):
    """Apply vocab reduction only: map each token to its top-v' representative.

    Tokens already in the top-v' pass through unchanged. Out-of-vocab tokens
    are mapped to their most frequent bigram neighbor (determined during
    compute_vocab_reduction). No spectral denoising — this is a pure
    frequency-based coarsening of the token vocabulary.
    """
    reduced = vocab_map[shard_tokens]
    return inv_vocab_map[reduced]


def vocab_reduce_corpus(tokens_dir, stats_dir, output_dir, tau=0.3,
                        max_shards=None, n_shards_for_stats=None):
    """Apply vocab reduction to all shards (no spectral denoising).

    tau controls the replacement rate: the vocabulary is sized so that
    out-of-vocab tokens account for approximately tau of the corpus.
    A tau-specific vocab_map is computed and saved alongside the output.
    """
    from language_reduction.statistics import compute_vocab_reduction, v_prime_for_tau

    os.makedirs(output_dir, exist_ok=True)

    # Compute v_prime from tau using the corpus frequency distribution
    freq_path = os.path.join(stats_dir, "freq.npy")
    if not os.path.exists(freq_path):
        raise FileNotFoundError(
            f"freq.npy not found at {stats_dir}. Run the 'stats' stage first.")
    freq = np.load(freq_path)
    v_prime = v_prime_for_tau(freq, tau)

    # Build a tau-specific vocab_map
    tau_stats_dir = os.path.join(output_dir, "stats")
    print(f"Computing vocab reduction for v_prime={v_prime} (tau={tau})...")
    compute_vocab_reduction(tokens_dir, tau_stats_dir, v_prime, n_shards_for_stats)

    vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
    top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    inv_vocab_map = build_inverse_vocab_map(vocab_map, top_ids)

    shard_paths = sorted(glob.glob(os.path.join(tokens_dir, "shard_*.npy")))
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    total_changed = 0
    total_tokens = 0
    for shard_idx, path in enumerate(shard_paths):
        tokens = np.load(path)
        output = vocab_reduce_shard(tokens, vocab_map, inv_vocab_map)
        n_changed = int((tokens != output).sum())
        total_changed += n_changed
        total_tokens += len(tokens)
        print(f"  Shard {shard_idx}: {n_changed:,}/{len(tokens):,} "
              f"({100*n_changed/len(tokens):.1f}%) tokens mapped to neighbors")
        np.save(os.path.join(output_dir, f"shard_{shard_idx:05d}.npy"), output)

    meta = np.load(os.path.join(tokens_dir, "meta.npy"), allow_pickle=True).item()
    meta["mode"] = "vocab_only"
    meta["tau"] = tau
    meta["v_prime"] = v_prime
    meta["source_dir"] = tokens_dir
    np.save(os.path.join(output_dir, "meta.npy"), meta)
    print(f"Vocab reduction complete: {len(shard_paths)} shards, "
          f"tau={tau}, v'={v_prime}, "
          f"{total_changed:,}/{total_tokens:,} tokens changed "
          f"({100*total_changed/total_tokens:.1f}%)")


def denoise_corpus(tokens_dir, stats_dir, output_dir, tau, n_lags=50,
                   max_shards=None, kappa_mode="energy"):
    """Denoise all shards in the corpus (sequential)."""
    os.makedirs(output_dir, exist_ok=True)

    vocab_map = np.load(os.path.join(stats_dir, "vocab_map.npy"))
    top_ids = np.load(os.path.join(stats_dir, "top_ids.npy"))
    inv_vocab_map = build_inverse_vocab_map(vocab_map, top_ids)

    shard_paths = sorted(glob.glob(os.path.join(tokens_dir, "shard_*.npy")))
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    for shard_idx, path in enumerate(shard_paths):
        tokens = np.load(path)
        print(f"Shard {shard_idx} ({len(tokens):,} tokens, tau={tau})...")
        denoised = denoise_shard(
            tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags,
            kappa_mode=kappa_mode,
        )
        out_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npy")
        np.save(out_path, denoised)

    meta = np.load(os.path.join(tokens_dir, "meta.npy"), allow_pickle=True).item()
    meta["tau"] = tau
    meta["kappa_mode"] = kappa_mode
    meta["source_dir"] = tokens_dir
    np.save(os.path.join(output_dir, "meta.npy"), meta)
    print(f"Denoising complete: {len(shard_paths)} shards at tau={tau} (kappa_mode={kappa_mode})")
