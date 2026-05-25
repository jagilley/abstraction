"""Local end-to-end test on a tiny synthetic corpus.

Generates a small corpus with known structure, runs the full pipeline,
and verifies that the denoising produces sensible output.
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from language_reduction.statistics import (
    compute_vocab_reduction,
    compute_covariance_matrices,
    spectral_decomposition,
)
from language_reduction.denoise import denoise_shard, build_inverse_vocab_map
from language_reduction.measure import measure_beta


def make_synthetic_corpus(tmpdir, n_tokens=100_000, vocab_size=200, n_shards=1):
    """Create a corpus with a few dominant patterns and some noise.

    Pattern 1: tokens 0,1 alternate (high-frequency bigram)
    Pattern 2: tokens 2,3 alternate
    Noise: random tokens from 4..199
    """
    tokens_dir = os.path.join(tmpdir, "tokens")
    os.makedirs(tokens_dir)

    rng = np.random.default_rng(42)
    tokens_per_shard = n_tokens // n_shards

    for shard_idx in range(n_shards):
        data = np.zeros(tokens_per_shard, dtype=np.uint16)
        for i in range(tokens_per_shard):
            r = rng.random()
            if r < 0.3:
                data[i] = i % 2
            elif r < 0.6:
                data[i] = 2 + (i % 2)
            else:
                data[i] = rng.integers(4, vocab_size)
        np.save(os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy"), data)

    meta = {
        "n_shards": n_shards,
        "shard_size": tokens_per_shard,
        "total_tokens": n_tokens,
        "encoding": "synthetic",
        "vocab_size": vocab_size,
    }
    np.save(os.path.join(tokens_dir, "meta.npy"), meta)
    return tokens_dir


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=== Creating synthetic corpus ===")
        tokens_dir = make_synthetic_corpus(tmpdir, n_tokens=50_000, vocab_size=200)

        stats_dir = os.path.join(tmpdir, "stats")
        v_prime = 30
        n_lags = 10

        print("\n=== Computing vocabulary reduction ===")
        vocab_map = compute_vocab_reduction(tokens_dir, stats_dir, v_prime)
        assert vocab_map.shape[0] == 200
        assert vocab_map.max() < v_prime
        assert (vocab_map >= 0).all()

        print("\n=== Computing covariance matrices ===")
        compute_covariance_matrices(tokens_dir, stats_dir, n_lags)

        for lag in range(1, n_lags + 1):
            path = os.path.join(stats_dir, "covariance", f"C_lag_{lag:03d}.npy")
            assert os.path.exists(path), f"missing C(lag={lag})"
            C = np.load(path)
            assert C.shape == (v_prime, v_prime)

        print("\n=== Spectral decomposition ===")
        summary = spectral_decomposition(stats_dir, n_lags, max_rank=min(20, v_prime - 1))
        assert len(summary) == n_lags
        # top-1 component should capture a decent fraction for our structured corpus
        print(f"  lag 1 top-1 energy: {summary[0]['top1_energy']:.3f}")

        print("\n=== Measuring beta ===")
        beta_result = measure_beta(stats_dir, n_lags, fit_range=(1, n_lags))
        beta = beta_result["beta"]
        print(f"  beta = {beta:.4f}")

        print("\n=== Denoising (tau=0.3) ===")
        tau = 0.3
        tokens = np.load(os.path.join(tokens_dir, "shard_00000.npy"))
        top_ids = np.load(os.path.join(stats_dir, "top_ids.npy"))
        inv_vocab_map = build_inverse_vocab_map(vocab_map, top_ids)

        denoised = denoise_shard(
            tokens, vocab_map, inv_vocab_map, stats_dir, tau, n_lags,
            batch_size=5000,
        )
        assert denoised.shape == tokens.shape
        assert denoised.dtype == tokens.dtype

        n_changed = (denoised != tokens).sum()
        frac_changed = n_changed / len(tokens)
        print(f"  Changed {n_changed:,}/{len(tokens):,} ({100*frac_changed:.1f}%)")

        # at tau=0.3, roughly 30% should be changed
        assert 0.1 < frac_changed < 0.6, f"unexpected change fraction: {frac_changed}"

        # check that dominant patterns survived
        reduced_orig = vocab_map[tokens]
        reduced_denoised = vocab_map[denoised]
        # tokens 0,1 (mapped to their reduced IDs) should mostly survive
        for t in [0, 1, 2, 3]:
            r = vocab_map[t]
            orig_count = (reduced_orig == r).sum()
            denoised_count = (reduced_denoised == r).sum()
            if orig_count > 0:
                survival = denoised_count / orig_count
                print(f"  Token {t} (reduced {r}): {orig_count} -> {denoised_count} "
                      f"(survival {survival:.2f})")

        print("\n=== All tests passed ===")


def test_glp():
    """Quick sanity check: build a tiny GPT, extract activations, train GLP."""
    import torch
    from language_reduction.model import GPT
    from language_reduction.glp import (
        GLPDenoiser, extract_activations, train_glp,
        glp_denoise, compute_residuals,
    )

    device = "cpu"
    vocab_size, block_size, n_layer, n_embd = 200, 32, 2, 64
    act_dim = n_layer * n_embd  # 128

    print("=== Building tiny GPT ===")
    model = GPT(vocab_size, block_size, n_layer, n_head=2, n_embd=n_embd)

    tokens = torch.randint(0, vocab_size, (1024,))
    print(f"  tokens: {tokens.shape}")

    print("\n=== Extracting activations ===")
    acts = extract_activations(model, tokens, block_size, batch_size=4,
                               max_tokens=256, device=device)
    print(f"  activations: {acts.shape}")
    assert acts.shape == (256, act_dim), f"expected (256, {act_dim}), got {acts.shape}"

    print("\n=== Training GLP (tiny) ===")
    glp_sd, act_stats, losses = train_glp(
        acts, act_dim=act_dim, d_model=64, d_mlp=128, n_layers=2,
        n_steps=200, batch_size=32, lr=1e-4, device=device,
        log_interval=50,
    )
    assert len(losses) > 0
    assert losses[-1][1] < losses[0][1], "loss should decrease"
    print(f"  loss: {losses[0][1]:.4f} -> {losses[-1][1]:.4f}")

    print("\n=== Computing residuals ===")
    glp_model = GLPDenoiser(act_dim=act_dim, d_model=64, d_mlp=128, n_layers=2)
    glp_model.load_state_dict(glp_sd)
    glp_model.eval()

    residuals, manifold = compute_residuals(
        glp_model, act_stats, acts[:32],
        t_start=0.3, num_steps=5, batch_size=16,
    )
    assert residuals.shape == (32, act_dim)
    assert manifold.shape == (32, act_dim)

    recon = acts[:32] - residuals
    cos_sim = torch.nn.functional.cosine_similarity(acts[:32], recon, dim=-1)
    print(f"  mean cosine similarity: {cos_sim.mean():.4f}")
    print(f"  mean residual norm: {residuals.norm(dim=-1).mean():.4f}")

    print("\n=== GLP test passed ===")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "glp":
        test_glp()
    else:
        test_pipeline()
