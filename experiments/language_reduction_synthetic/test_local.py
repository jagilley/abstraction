"""Local sanity checks for RHM data generation and model training."""

import numpy as np
import sys


def test_rhm_generation():
    from language_reduction_synthetic.rhm import generate_rules, generate_sequences, make_corpus

    v, s, L, m = 4, 2, 3, 2
    seq_len = s ** L  # 8

    rules = generate_rules(v, s, L, m, seed=42)
    assert len(rules) == L
    for r in rules:
        assert r.shape == (v, m, s)
        assert r.min() >= 0 and r.max() < v

    seqs = generate_sequences(rules, 100, seed=0)
    assert seqs.shape == (100, seq_len)
    assert seqs.min() >= 0 and seqs.max() < v

    # check that same seed gives same output
    seqs2 = generate_sequences(rules, 100, seed=0)
    assert np.array_equal(seqs, seqs2)

    # different seed gives different output
    seqs3 = generate_sequences(rules, 100, seed=1)
    assert not np.array_equal(seqs, seqs3)

    # check hierarchical structure: tokens within same s-tuple should
    # show statistical dependency (they share a level-1 parent)
    seqs_large = generate_sequences(rules, 10000, seed=0)
    # co-occurrence of adjacent pairs should be non-uniform
    pair_counts = np.zeros((v, v))
    for i in range(0, seq_len, s):
        for row in seqs_large:
            pair_counts[row[i], row[i + 1]] += 1
    # normalize
    pair_freq = pair_counts / pair_counts.sum()
    marginal = pair_freq.sum(axis=1)
    expected = np.outer(marginal, marginal)
    # mutual information should be positive (structure exists)
    mi = 0.0
    for i in range(v):
        for j in range(v):
            if pair_freq[i, j] > 0 and expected[i, j] > 0:
                mi += pair_freq[i, j] * np.log(pair_freq[i, j] / expected[i, j])
    print(f"  Adjacent-pair MI: {mi:.4f} (should be > 0)")
    assert mi > 0.01, f"Adjacent tokens should show dependency, got MI={mi:.4f}"

    # corpus generation
    corpus, _, meta = make_corpus(v, s, L, m, 1000, rule_seed=42, seq_seed=0)
    assert len(corpus) == 1000
    assert meta["vocab_size"] == v
    assert meta["seq_len"] == seq_len

    print("  RHM generation: PASSED")


def test_batched_generation():
    from language_reduction_synthetic.rhm import (
        generate_rules, generate_sequences_batched
    )

    v, s, L, m = 8, 2, 4, 3
    seq_len = s ** L
    rules = generate_rules(v, s, L, m, seed=7)

    seqs = generate_sequences_batched(rules, 500, seed=42, batch_size=100)
    assert seqs.shape == (500, seq_len)
    assert seqs.min() >= 0 and seqs.max() < v

    # same seed → same output
    seqs2 = generate_sequences_batched(rules, 500, seed=42, batch_size=100)
    assert np.array_equal(seqs, seqs2)
    print("  Batched generation: PASSED")


def test_correlation_structure():
    """Verify that correlations decay with distance as expected."""
    from language_reduction_synthetic.rhm import generate_rules, generate_sequences

    v, s, L, m = 8, 2, 6, 4  # seq_len = 64
    rules = generate_rules(v, s, L, m, seed=0)
    seqs = generate_sequences(rules, 50000, seed=0)
    corpus = seqs.reshape(-1)

    # compute ||C(n)||_op for several lags
    norms = []
    lags = [1, 2, 4, 8, 16, 32]
    for n in lags:
        # build co-occurrence matrix C(n)
        C = np.zeros((v, v))
        freq = np.zeros(v)
        N = len(corpus) - n
        for i in range(N):
            C[corpus[i], corpus[i + n]] += 1
            freq[corpus[i]] += 1
        C /= N
        freq /= N
        C -= np.outer(freq, freq)
        norm = np.linalg.norm(C, ord=2)
        norms.append(norm)

    print(f"  Correlation norms ||C(n)||_op:")
    for lag, norm in zip(lags, norms):
        print(f"    n={lag:3d}: {norm:.6f}")

    # correlations should generally decrease (though not monotonically
    # due to the discrete hierarchy)
    assert norms[0] > norms[-1], "Near correlations should be stronger than far"
    print("  Correlation structure: PASSED")


def test_model_forward():
    """Verify model runs on RHM data."""
    import torch
    from language_reduction_synthetic.model import GPT
    from language_reduction_synthetic.rhm import make_corpus

    v, s, L, m = 8, 2, 4, 3  # seq_len = 16
    seq_len = s ** L
    corpus, _, _ = make_corpus(v, s, L, m, 10000)
    data = torch.from_numpy(corpus)

    model = GPT(vocab_size=v, block_size=seq_len, n_layer=2, n_head=2, n_embd=32)

    # forward pass
    x = data[:64].view(4, seq_len)
    y = data[1:65].view(4, seq_len)
    logits, loss = model(x, y)
    assert logits.shape == (4, seq_len, v)
    assert loss is not None and loss.item() > 0
    print(f"  Model forward: loss={loss.item():.4f} (expected ~{np.log(v):.4f} = ln({v}))")
    print("  Model forward: PASSED")


def test_training_loop():
    """Quick training to verify loss decreases."""
    import torch
    from language_reduction_synthetic.model import GPT
    from language_reduction_synthetic.rhm import make_corpus

    v, s, L, m = 8, 2, 4, 3
    seq_len = s ** L
    corpus, _, _ = make_corpus(v, s, L, m, 50000)
    data = torch.from_numpy(corpus)

    model = GPT(vocab_size=v, block_size=seq_len, n_layer=2, n_head=2, n_embd=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    initial_loss = None
    final_loss = None
    for step in range(200):
        ix = torch.randint(len(data) - seq_len - 1, (16,))
        x = torch.stack([data[i:i + seq_len] for i in ix])
        y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        if step == 199:
            final_loss = loss.item()

    print(f"  Training: initial={initial_loss:.4f} → final={final_loss:.4f}")
    assert final_loss < initial_loss, "Loss should decrease with training"
    print("  Training loop: PASSED")


if __name__ == "__main__":
    print("Testing RHM generation...")
    test_rhm_generation()

    print("\nTesting batched generation...")
    test_batched_generation()

    print("\nTesting correlation structure...")
    test_correlation_structure()

    print("\nTesting model forward pass...")
    test_model_forward()

    print("\nTesting training loop...")
    test_training_loop()

    print("\n✓ All tests passed!")
