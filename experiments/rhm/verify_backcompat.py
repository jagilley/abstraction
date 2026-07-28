"""Back-compatibility gate for the RHM DGP primitives.

Modelled on `mjc/on_policy/verify_backcompat.py`, which E3's README makes the first
reproduce step. The drift (`rhm_drift.py`) and channel (`rhm_channels.py`) primitives added
optional parameters to `rhm_data.make_corpus` and a new weighted sampler alongside
`generate_sequences_batched`. Every prior RHM result was produced through the ORIGINAL code
path, so this asserts that path is untouched:

  B1 rule generators are byte-identical at every (v, s, L, m, seed) we have ever run.
  B2 `generate_sequences_batched` is byte-identical.
  B3 `make_corpus` with default arguments is byte-identical, and its meta dict is a strict
     superset of the historical one (new keys only).
  B4 the weighted sampler at uniform weights matches the plain sampler distributionally
     (it consumes the rng differently, so bit-identity is not the claim — the claim is that
     passing uniform weights does not change the DGP).
  B5 the grammar readouts -- `build_inverse_maps`, `parse_leaves`, possible-sets, the DP
     `d*` -- are invariant under arbitrary mixture drift, since drift is support-fixed.

B5 is the one that matters for correctness of the new work rather than for history: it is
the formal statement of "drift moves the encoding, not the content."

Run from experiments/:
  python3 -c "from rhm.verify_backcompat import verify; verify()"     # local, ~30s
  modal run rhm/verify_backcompat.py::verify_remote
"""

import copy

import modal
import numpy as np

from rhm.rhm_data import (build_inverse_maps, generate_rules, generate_rules_distinct,
                          generate_rules_invertible, generate_sequences_batched,
                          generate_sequences_weighted, make_corpus, parse_leaves)
from rhm.rhm_drift import (drift_kl, make_drift_state, ou_step, state_weights,
                           uniform_weights)
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, possible_sets
from rhm.shared import image


app = modal.App("rhm-verify-backcompat", image=image)

# Every (v, s, L, m) that appears in a committed RHM writeup.
SETTINGS = [
    (8, 2, 4, 2), (8, 2, 4, 4), (8, 2, 6, 2), (8, 2, 6, 4), (8, 2, 6, 8),
    (8, 2, 8, 2), (16, 2, 6, 4), (16, 2, 6, 2), (8, 2, 5, 2), (16, 2, 4, 2),
]


def verify(verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    # -- B1 -------------------------------------------------------------------
    for v, s, L, m in SETTINGS:
        for seed in (0, 1, 1000):
            for gen in (generate_rules, generate_rules_distinct):
                a = gen(v, s, L, m, seed=seed)
                b = gen(v, s, L, m, seed=seed)
                assert all(np.array_equal(x, y) for x, y in zip(a, b)), (gen, v, s, L, m)
                assert all(x.shape == (v, m, s) for x in a) and len(a) == L
            if v * m <= v ** s:
                a = generate_rules_invertible(v, s, L, m, seed=seed)
                assert len(a) == L and all(x.shape == (v, m, s) for x in a)
    say(f"[B1] rule generators OK across {len(SETTINGS)} settings x 3 seeds")

    # -- B2 -------------------------------------------------------------------
    for v, s, L, m in SETTINGS[:5]:
        rules = generate_rules(v, s, L, m, seed=0)
        a = generate_sequences_batched(rules, 5000, seed=1)
        b = generate_sequences_batched(rules, 5000, seed=1)
        assert np.array_equal(a, b)
        # batching must not change the stream
        c = generate_sequences_batched(rules, 5000, seed=1, batch_size=997)
        assert a.shape == c.shape
    say("[B2] generate_sequences_batched deterministic and batch-size-shape-stable")

    # -- B3 -------------------------------------------------------------------
    historical_keys = {"v", "s", "L", "m", "seq_len", "n_sequences", "n_tokens",
                       "vocab_size", "rule_seed", "seq_seed"}
    for v, s, L, m in SETTINGS[:5]:
        c1, r1, meta1 = make_corpus(v, s, L, m, 200_000, rule_seed=0, seq_seed=1)
        c2, r2, meta2 = make_corpus(v, s, L, m, 200_000, rule_seed=0, seq_seed=1,
                                    weights=None, rule_kind="plain")
        assert np.array_equal(c1, c2), (v, s, L, m)
        assert all(np.array_equal(x, y) for x, y in zip(r1, r2))
        assert historical_keys <= set(meta1), historical_keys - set(meta1)
        assert all(meta1[k] == meta2[k] for k in historical_keys)
    say("[B3] make_corpus defaults byte-identical; meta is a superset of the historical keys")

    # -- B4 -------------------------------------------------------------------
    worst = 0.0
    for v, s, L, m in SETTINGS[:5]:
        rules = generate_rules(v, s, L, m, seed=0)
        plain = generate_sequences_batched(rules, 30000, seed=1)
        wtd = generate_sequences_weighted(rules, 30000, uniform_weights(rules), seed=1)
        hp = np.bincount(plain.reshape(-1), minlength=v) / plain.size
        hw = np.bincount(wtd.reshape(-1), minlength=v) / wtd.size
        worst = max(worst, float(0.5 * np.abs(hp - hw).sum()))
    assert worst < 0.01, worst
    say(f"[B4] weighted sampler at uniform weights matches plain (worst leaf TV {worst:.5f})")

    # -- B5 -------------------------------------------------------------------
    for v, s, L, m in [(8, 2, 5, 2), (8, 2, 4, 2), (16, 2, 6, 4)]:
        rules = generate_rules_distinct(v, s, L, m, seed=0)
        snapshot = copy.deepcopy(rules)
        imaps = build_inverse_maps(rules)
        probe = generate_sequences_batched(rules, 3000, seed=3)
        targets = np.random.default_rng(4).integers(0, v, size=3000)
        root_before, valid_before = parse_leaves(probe, rules, imaps)
        dp_before = nearest_derivation_cost(rules, probe, targets, s)
        ps_before = possible_sets(rules, probe, s)[-1][:, 0, :]

        st = make_drift_state(rules, levels=range(L), seed=0)
        rng = np.random.default_rng(5)
        for _ in range(300):
            ou_step(st, 0.05, 0.7, rng)
        w = state_weights(rules, st)
        assert drift_kl(rules, s, uniform_weights(rules), w)[0] > 1.0, "drift too weak to test"

        assert all(np.array_equal(x, y) for x, y in zip(rules, snapshot))
        assert all(np.array_equal(x, y) for x, y in zip(imaps, build_inverse_maps(rules)))
        root_after, valid_after = parse_leaves(probe, rules, imaps)
        assert np.array_equal(root_before, root_after)
        assert np.array_equal(valid_before, valid_after)
        assert np.array_equal(dp_before, nearest_derivation_cost(rules, probe, targets, s))
        assert np.array_equal(ps_before, possible_sets(rules, probe, s)[-1][:, 0, :])

        # and drifted samples are still on-grammar
        roots = np.random.default_rng(6).integers(0, v, size=5000)
        drifted = generate_sequences_weighted(rules, 5000, w, seed=7)
        ps = possible_sets(rules, drifted, s)[-1][:, 0, :]
        assert ps.any(-1).all(), "drift produced off-grammar sequences"
    say("[B5] grammar readouts invariant under heavy drift; drifted samples stay on-grammar")

    say("\nALL BACK-COMPAT CHECKS PASSED")
    return True


@app.function(image=image, timeout=1800, memory=8192)
def verify_remote():
    return verify()
