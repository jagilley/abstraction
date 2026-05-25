"""Vocabulary reduction, multi-lag covariance matrices, and spectral analysis."""

import os
import glob
import numpy as np
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Vocabulary reduction
# ---------------------------------------------------------------------------

def _embedding_replacement_map(V, sorted_top_ids, id_to_reduced, most_frequent_reduced,
                                encoding="gpt2"):
    """Map each rare token to the nearest top-v' token by embedding cosine similarity.

    Uses the pretrained embedding matrix as a semantic similarity oracle.
    This captures paradigmatic relations (interchangeable words) rather than
    syntagmatic ones (co-occurring words), which is what we want for replacement.
    """
    import torch
    from transformers import GPT2LMHeadModel

    print("Loading GPT-2 embeddings for semantic replacement mapping...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    emb = model.transformer.wte.weight.detach()
    assert emb.shape[0] == V, f"embedding vocab {emb.shape[0]} != corpus vocab {V}"
    emb_norm = emb / emb.norm(dim=1, keepdim=True)

    top_emb = emb_norm[sorted_top_ids]  # (v_prime, 768)
    mapped = id_to_reduced.copy()

    rare_mask = mapped < 0
    rare_ids = np.where(rare_mask)[0]
    if len(rare_ids) > 0:
        rare_emb = emb_norm[rare_ids]  # (n_rare, 768)
        sims = (rare_emb @ top_emb.T).numpy()  # (n_rare, v_prime)
        mapped[rare_ids] = sims.argmax(axis=1).astype(np.int32)

    unmapped = mapped < 0
    if unmapped.any():
        mapped[unmapped] = most_frequent_reduced
    return mapped


def _bigram_replacement_map(shard_paths, V, v_prime, id_to_reduced, sorted_top_ids,
                            freq, most_frequent_reduced):
    """Map each rare token to the top-v' token with highest bigram PMI."""
    rare_top_counts = np.zeros(V * v_prime, dtype=np.int64)

    for path in shard_paths:
        tokens = np.load(path).astype(np.int64)
        a, b = tokens[:-1], tokens[1:]
        a_reduced = id_to_reduced[a]
        b_reduced = id_to_reduced[b]

        mask = (a_reduced == -1) & (b_reduced >= 0)
        if mask.any():
            compound = a[mask] * v_prime + b_reduced[mask]
            rare_top_counts += np.bincount(compound, minlength=V * v_prime)

        mask = (b_reduced == -1) & (a_reduced >= 0)
        if mask.any():
            compound = b[mask] * v_prime + a_reduced[mask]
            rare_top_counts += np.bincount(compound, minlength=V * v_prime)

    rare_top_matrix = rare_top_counts.reshape(V, v_prime)
    top_freq = freq[sorted_top_ids].astype(np.float64)
    rare_top_pmi = rare_top_matrix.astype(np.float64) / np.maximum(top_freq[np.newaxis, :], 1.0)

    mapped = id_to_reduced.copy()
    for tok in range(V):
        if mapped[tok] >= 0:
            continue
        row = rare_top_pmi[tok]
        if row.max() > 0:
            mapped[tok] = int(np.argmax(row))
        else:
            mapped[tok] = most_frequent_reduced
    return mapped


def _print_sample_replacements(vocab_map, freq, sorted_top_ids, encoding="gpt2"):
    """Print sample replacement mappings for the highest-frequency rare tokens."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        decode = lambda tid: enc.decode([tid])
    except Exception:
        decode = lambda tid: f"[{tid}]"

    top_set = set(sorted_top_ids.tolist())
    rare_tokens = [(tok, freq[tok]) for tok in range(len(vocab_map))
                   if tok not in top_set and freq[tok] > 0]
    rare_tokens.sort(key=lambda x: -x[1])

    n = min(50, len(rare_tokens))
    print(f"\nSample replacements (top {n} rare tokens by frequency):")
    print(f"{'rare token':>25s}  {'freq':>8s}  {'replacement':>20s}")
    print("-" * 60)
    for tok, f in rare_tokens[:n]:
        reduced_id = vocab_map[tok]
        target_id = sorted_top_ids[reduced_id]
        print(f"{decode(tok)!r:>25s}  {f:>8,}  {decode(target_id)!r:>20s}")


def compute_vocab_reduction(tokens_dir: str, stats_dir: str, v_prime: int = 3200,
                            n_shards_for_stats: int | None = None,
                            encoding: str = "gpt2"):
    """Build a mapping from full vocab → reduced vocab of size v_prime.

    Strategy: keep the top v_prime tokens by unigram frequency. Map every
    other token to the nearest top-v_prime token by embedding cosine
    similarity (falls back to bigram PMI if embeddings unavailable).
    """
    os.makedirs(stats_dir, exist_ok=True)
    shard_paths = sorted(glob.glob(os.path.join(tokens_dir, "shard_*.npy")))
    if n_shards_for_stats is not None:
        shard_paths = shard_paths[:n_shards_for_stats]

    meta = np.load(os.path.join(tokens_dir, "meta.npy"), allow_pickle=True).item()
    V = meta["vocab_size"]

    # --- unigram counts ---
    freq = np.zeros(V, dtype=np.int64)
    for path in shard_paths:
        tokens = np.load(path)
        counts = np.bincount(tokens, minlength=V)
        freq += counts
    print(f"Unigram counts computed over {len(shard_paths)} shards, {freq.sum():,} tokens")

    # top v_prime tokens by frequency
    top_ids = np.argsort(freq)[::-1][:v_prime]
    # create stable ordering: sorted by original id within the top set
    sorted_top_ids = np.array(sorted(top_ids))
    id_to_reduced = np.full(V, -1, dtype=np.int32)
    for reduced_idx, orig_id in enumerate(sorted_top_ids):
        id_to_reduced[orig_id] = reduced_idx

    most_frequent_reduced = id_to_reduced[top_ids[0]]

    # --- map rare tokens to nearest top-v' token ---
    try:
        id_to_reduced = _embedding_replacement_map(
            V, sorted_top_ids, id_to_reduced, most_frequent_reduced, encoding)
        method = "embedding"
    except Exception as e:
        print(f"Embedding replacement unavailable ({e}), falling back to bigram PMI")
        id_to_reduced = _bigram_replacement_map(
            shard_paths, V, v_prime, id_to_reduced, sorted_top_ids, freq,
            most_frequent_reduced)
        method = "bigram_pmi"

    assert (id_to_reduced >= 0).all(), "all tokens must be mapped"

    np.save(os.path.join(stats_dir, "vocab_map.npy"), id_to_reduced)
    np.save(os.path.join(stats_dir, "freq.npy"), freq)
    np.save(os.path.join(stats_dir, "top_ids.npy"), top_ids)
    print(f"Vocab reduction: {V} → {v_prime} (mapped {V - v_prime} rare tokens, "
          f"method={method})")
    _print_sample_replacements(id_to_reduced, freq, sorted_top_ids, encoding)
    return id_to_reduced


def v_prime_for_tau(freq: np.ndarray, tau: float) -> int:
    """Find the vocabulary size where out-of-vocab tokens account for ~tau of the corpus.

    Sorts tokens by frequency (descending) and finds the cutoff where the
    cumulative frequency of the top-v_prime tokens is (1 - tau) of the total.
    """
    total = freq.sum()
    sorted_freq = np.sort(freq)[::-1]
    cumsum = np.cumsum(sorted_freq)
    target = (1.0 - tau) * total
    v_prime = int(np.searchsorted(cumsum, target)) + 1
    v_prime = max(100, min(v_prime, len(freq)))
    actual_tau = 1.0 - cumsum[v_prime - 1] / total
    print(f"tau={tau:.3f} → v_prime={v_prime} "
          f"(top-{v_prime} covers {100*(1-actual_tau):.1f}% of corpus, "
          f"actual replacement rate ≈ {100*actual_tau:.1f}%)")
    return v_prime


# ---------------------------------------------------------------------------
# Multi-lag covariance matrices
# ---------------------------------------------------------------------------

def compute_covariance_matrices(tokens_dir: str, stats_dir: str, n_lags: int = 50,
                                 n_shards_for_stats: int | None = None):
    """Compute C(n) for n = 1..n_lags in reduced-vocab space.

    C_{mu,nu}(n) = P(X_i=mu, X_{i+n}=nu) - P(X_i=mu)P(X_i=nu)

    Pre-loads all shards into memory once, then sweeps over lags.
    """
    vocab_map = np.load(os.path.join(stats_dir, "vocab_map.npy"))
    v_prime = int(vocab_map.max()) + 1

    shard_paths = sorted(glob.glob(os.path.join(tokens_dir, "shard_*.npy")))
    if n_shards_for_stats is not None:
        shard_paths = shard_paths[:n_shards_for_stats]

    # pre-load all shards as reduced token arrays
    print(f"Loading {len(shard_paths)} shards into memory...")
    reduced_shards = []
    for path in shard_paths:
        tokens = np.load(path)
        reduced_shards.append(vocab_map[tokens].astype(np.int32))

    # compute marginal
    marginal = np.zeros(v_prime, dtype=np.float64)
    total_tokens = 0
    for reduced in reduced_shards:
        counts = np.bincount(reduced, minlength=v_prime).astype(np.float64)
        marginal += counts
        total_tokens += len(reduced)

    marginal /= total_tokens
    print(f"Marginal computed: {total_tokens:,} tokens, V'={v_prime}")

    os.makedirs(os.path.join(stats_dir, "covariance"), exist_ok=True)

    os.makedirs(os.path.join(stats_dir, "pmi"), exist_ok=True)
    outer = np.outer(marginal, marginal)

    for lag in range(1, n_lags + 1):
        joint_counts = np.zeros(v_prime * v_prime, dtype=np.float64)
        n_pairs = 0

        for reduced in reduced_shards:
            src = reduced[:-lag]
            dst = reduced[lag:]
            flat_idx = src.astype(np.int64) * v_prime + dst.astype(np.int64)
            joint_counts += np.bincount(flat_idx, minlength=v_prime * v_prime).astype(np.float64)
            n_pairs += len(src)

        joint = joint_counts.reshape(v_prime, v_prime) / n_pairs
        C_n = joint - outer

        out_path = os.path.join(stats_dir, "covariance", f"C_lag_{lag:03d}.npy")
        np.save(out_path, C_n)

        # PMI(n) = log(P(x,y) / P(x)P(y)), with floor for zero-count pairs
        floor = 1.0 / n_pairs
        pmi_n = np.log(np.maximum(joint, floor)) - np.log(np.maximum(outer, floor))
        np.save(os.path.join(stats_dir, "pmi", f"PMI_lag_{lag:03d}.npy"), pmi_n)

        norm_op = np.linalg.norm(C_n, ord=2)
        norm_f = np.linalg.norm(C_n, ord="fro")
        print(f"  lag {lag:3d}: ||C||_op = {norm_op:.6f}, ||C||_F = {norm_f:.6f}")

    np.save(os.path.join(stats_dir, "marginal.npy"), marginal)
    print(f"Covariance + PMI matrices saved for lags 1..{n_lags}")


# ---------------------------------------------------------------------------
# Spectral decomposition
# ---------------------------------------------------------------------------

def compute_effective_rank(singular_values):
    """Effective rank = exp(H(p)) where p_i = σ_i² / Σσ_j².

    Measures how many independent directions the matrix actually uses.
    Flat spectrum → rank equals dimension. Concentrated spectrum → small rank.
    """
    s_sq = singular_values ** 2
    total = s_sq.sum()
    if total < 1e-15:
        return 1.0
    p = s_sq / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def spectral_decomposition(stats_dir: str, n_lags: int = 50, max_rank: int = 64,
                            use_pmi: bool = True):
    """Compute truncated SVD and store the factors.

    By default decomposes the PMI matrices (use_pmi=True), which factors out
    token frequency and surfaces genuine associative structure. The raw
    covariance C(n) is still used for beta measurement via ||C(n)||_op.

    For each lag, stores U (v_prime x k), S (k,), Vt (k x v_prime)
    where k = min(max_rank, effective rank at 99% energy).
    Also stores the total Frobenius norm squared for tau-based truncation,
    and the effective rank computed from the full singular value spectrum.
    """
    os.makedirs(os.path.join(stats_dir, "svd"), exist_ok=True)
    summary = []
    matrix_type = "PMI" if use_pmi else "C"
    print(f"Decomposing {matrix_type} matrices...")

    for lag in range(1, n_lags + 1):
        if use_pmi:
            M = np.load(os.path.join(stats_dir, "pmi", f"PMI_lag_{lag:03d}.npy"))
        else:
            M = np.load(os.path.join(stats_dir, "covariance", f"C_lag_{lag:03d}.npy"))
        frob_sq = np.sum(M ** 2)

        k = min(max_rank, min(M.shape) - 1)
        U, S, Vt = svds(M, k=k)
        # svds returns in ascending order; reverse to descending
        idx = np.argsort(S)[::-1]
        U, S, Vt = U[:, idx], S[idx], Vt[idx, :]

        cumulative_energy = np.cumsum(S ** 2) / frob_sq

        # full spectrum for effective rank
        all_S = np.linalg.svd(M, compute_uv=False)
        eff_rank = compute_effective_rank(all_S)

        svd_dir = os.path.join(stats_dir, "svd", f"lag_{lag:03d}")
        os.makedirs(svd_dir, exist_ok=True)
        np.save(os.path.join(svd_dir, "U.npy"), U)
        np.save(os.path.join(svd_dir, "S.npy"), S)
        np.save(os.path.join(svd_dir, "Vt.npy"), Vt)
        np.save(os.path.join(svd_dir, "frob_sq.npy"), np.array(frob_sq))
        np.save(os.path.join(svd_dir, "cumulative_energy.npy"), cumulative_energy)
        np.save(os.path.join(svd_dir, "effective_rank.npy"), np.array(eff_rank))

        summary.append({
            "lag": lag,
            "frob_sq": float(frob_sq),
            "top1_energy": float(cumulative_energy[0]),
            "top10_energy": float(cumulative_energy[min(9, k - 1)]),
            "n_components": k,
            "effective_rank": eff_rank,
        })
        print(f"  lag {lag:3d}: top-1 = {cumulative_energy[0]:.3f}, "
              f"top-10 = {cumulative_energy[min(9, k-1)]:.3f}, "
              f"eff_rank = {eff_rank:.1f}, "
              f"||C||²_F = {frob_sq:.2e}")

    np.save(os.path.join(stats_dir, "svd", "summary.npy"), summary)
    print("Spectral decomposition complete")
    return summary


def get_low_rank_factors(stats_dir: str, lag: int, tau: float):
    """Load low-rank C_k(n) factors for a given tau.

    tau controls what fraction of *residual* spectral energy to suppress.
    At tau=0, k=all (keep everything). At tau=1, k=0 (suppress everything).
    We keep the top-k singular vectors accounting for (1-tau) of ||C(n)||²_F.
    """
    svd_dir = os.path.join(stats_dir, "svd", f"lag_{lag:03d}")
    U = np.load(os.path.join(svd_dir, "U.npy"))
    S = np.load(os.path.join(svd_dir, "S.npy"))
    Vt = np.load(os.path.join(svd_dir, "Vt.npy"))
    cum_energy = np.load(os.path.join(svd_dir, "cumulative_energy.npy"))

    target_energy = 1.0 - tau
    if target_energy <= 0:
        return np.zeros_like(U[:, :1]), np.zeros(1), np.zeros_like(Vt[:1, :])

    k = int(np.searchsorted(cum_energy, target_energy)) + 1
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


def get_low_rank_factors_by_effective_rank(stats_dir: str, lag: int):
    """Load low-rank factors truncated at the effective rank.

    The effective rank is a corpus-derived property (not a free parameter):
    it measures how many independent directions the PMI structure uses at
    this lag. Using it as the truncation point decouples the definition of
    "structured co-occurrence" from the intervention threshold tau.
    """
    svd_dir = os.path.join(stats_dir, "svd", f"lag_{lag:03d}")
    U = np.load(os.path.join(svd_dir, "U.npy"))
    S = np.load(os.path.join(svd_dir, "S.npy"))
    Vt = np.load(os.path.join(svd_dir, "Vt.npy"))
    eff_rank = float(np.load(os.path.join(svd_dir, "effective_rank.npy")))

    k = max(1, round(eff_rank))
    k = min(k, len(S))
    return U[:, :k], S[:k], Vt[:k, :]
