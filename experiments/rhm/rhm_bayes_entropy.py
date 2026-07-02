"""Exact per-level Bayes conditional entropy for the RHM.

Motivation
----------
RHM next-token prediction is NOT a deterministic function (unlike modular
arithmetic): the generative process samples one of `m` production rules,
independently, at every internal node of the parse tree. So even a predictor
that has PERFECTLY recovered the DGP rule tables incurs irreducible loss equal
to the source's conditional entropy H(x_i | x_{<i}). This module computes that
floor exactly.

The Bayes-optimal causal predictor achieves H(x_i | x_{<i}) at each position
(chain rule: sum_i H(x_i | x_{<i}) = H(sequence) exactly). We compute it via
sum-product belief propagation over the KNOWN parse tree with the KNOWN rules.
Because each factor psi(parent, child_tuple) = (1/m) * [tuple is one of parent's
m rules] is a sparse deterministic factor, BP costs only O(v*m*s) per node --
trivial.

Outputs, grouped by s-adic valuation (= hierarchy level, matching
`_position_levels` used everywhere else in this repo):
  - per-level Bayes conditional entropy (nats & bits) -- the floor to measure
    grokking against, instead of "near uniform".
  - aggregate entropy rate H(seq)/seq_len -- the true averaged val-loss floor.
  - the generative upper bound (ln v + n_internal*ln m)/seq_len for a sanity
    check (BP result must be <= this; it is strictly below iff the rules induce
    parse collisions).

Usage (Modal, CPU, ~1 min):
    modal run --detach -m rhm.rhm_bayes_entropy::bayes_entropy \
        --v 16 --m 2 --s 2 --depth 6 --n-seqs 4000

Local correctness self-test (brute-force vs BP on a tiny tree):
    python3 rhm/rhm_bayes_entropy.py
"""

import json
import os

import numpy as np

# ------------------------------------------------------------------------
# Core: exact belief propagation on the RHM parse tree (pure numpy)
# ------------------------------------------------------------------------


def _upward(up_leaf, rules, B, s, L, v):
    """Upward (leaves->root) sum-product messages.

    up_leaf: (B, s^L, v) leaf messages (one-hot for observed, ones for free).
    Returns list `up` where up[d] has shape (B, s^d, v) = likelihood of the
    observed evidence inside node's subtree as a function of the node's value.
    """
    up = [None] * (L + 1)
    up[L] = up_leaf
    for d in range(L - 1, -1, -1):
        n_par = s ** d
        up_child = up[d + 1].reshape(B, n_par, s, v)  # group each parent's s kids
        up_d = np.zeros((B, n_par, v))
        rd = rules[d]  # (v, m, s)
        for a in range(v):
            acc = np.zeros((B, n_par))
            for r in range(rd.shape[1]):  # m rules
                tup = rd[a, r]  # (s,)
                prod = np.ones((B, n_par))
                for k in range(s):
                    prod = prod * up_child[:, :, k, tup[k]]
                acc += prod
            up_d[:, :, a] = acc / rd.shape[1]
        up[d] = up_d
    return up


def _downward(up, rules, B, s, L, v):
    """Downward (root->leaves) sum-product messages.

    down[d] (B, s^d, v) = likelihood of evidence OUTSIDE node's subtree as a
    function of the node's value. Root prior is uniform (root sampled uniformly).
    Posterior at any node ∝ up[d] * down[d].
    """
    down = [None] * (L + 1)
    down[0] = np.full((B, 1, v), 1.0 / v)
    for d in range(L):
        n_par = s ** d
        up_child = up[d + 1].reshape(B, n_par, s, v)
        m = rules[d].shape[1]
        rd = rules[d]
        down_child = np.zeros((B, n_par, s, v))
        for a in range(v):
            da = down[d][:, :, a]  # (B, n_par) message coming from above into parent value a
            for r in range(m):
                tup = rd[a, r]  # (s,)
                ups = [up_child[:, :, k, tup[k]] for k in range(s)]  # each (B, n_par)
                for i in range(s):
                    prod_except_i = np.ones((B, n_par))
                    for k in range(s):
                        if k != i:
                            prod_except_i = prod_except_i * ups[k]
                    down_child[:, :, i, tup[i]] += da * (prod_except_i / m)
        down[d + 1] = down_child.reshape(B, n_par * s, v)
    return down


def bp_conditional_entropies(rules, seqs):
    """Exact H(x_i | x_{<i}) for every prediction position, averaged over `seqs`.

    rules: list of L arrays (v, m, s) (level 0 = top). seqs: (B, s^L) int.
    Returns (H, aggregate) where H has length seq_len; H[0] = H(x_0)
    (unconditional), H[i] = mean over the batch of the entropy of the exact
    posterior P(x_i | x_{<i}=obs). `aggregate` = mean over ALL positions =
    entropy rate estimate (nats/token).
    """
    B, seq_len = seqs.shape
    v, m, s = rules[0].shape
    L = len(rules)
    assert seq_len == s ** L

    H = np.zeros(seq_len)
    for i in range(seq_len):
        # Evidence: leaves 0..i-1 observed (one-hot), leaves i..end free (ones).
        up_leaf = np.ones((B, seq_len, v))
        if i > 0:
            obs = seqs[:, :i]  # (B, i)
            oh = np.zeros((B, i, v))
            np.put_along_axis(oh, obs[:, :, None], 1.0, axis=2)
            up_leaf[:, :i, :] = oh
        up = _upward(up_leaf, rules, B, s, L, v)
        down = _downward(up, rules, B, s, L, v)
        post = up[L][:, i, :] * down[L][:, i, :]  # (B, v)
        post = post / np.clip(post.sum(1, keepdims=True), 1e-30, None)
        ent = -(post * np.log(np.clip(post, 1e-30, None))).sum(1)  # (B,) nats
        H[i] = ent.mean()
    return H, H.mean()


def _position_levels(seq_len, s):
    """s-adic valuation for each prediction position 1..seq_len-1 (matches repo)."""
    levels = np.zeros(seq_len - 1, dtype=np.int64)
    for t in range(seq_len - 1):
        p = t + 1
        level = 0
        while p % s == 0:
            p //= s
            level += 1
        levels[t] = level
    return levels


def summarize(rules, seqs):
    """Full per-position + per-level summary dict."""
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L
    H, agg = bp_conditional_entropies(rules, seqs)

    levels = _position_levels(seq_len, s)  # for positions 1..seq_len-1
    per_level = {}
    for lvl in sorted(set(levels.tolist())):
        pos = np.where(levels == lvl)[0] + 1  # +1: levels indexes target position t->p=t+1
        hs = H[pos]
        per_level[int(lvl)] = {
            "n_positions": int(len(pos)),
            "positions": pos.tolist(),
            "bayes_nats": float(hs.mean()),
            "bayes_bits": float(hs.mean() / np.log(2)),
        }

    n_internal = (s ** L - 1) // (s - 1)
    gen_ub_nats = (np.log(v) + n_internal * np.log(m)) / seq_len
    return {
        "v": v, "m": m, "s": s, "L": L, "seq_len": seq_len,
        "n_sequences": int(seqs.shape[0]),
        "H_x0_nats": float(H[0]),
        "entropy_rate_nats": float(agg),          # includes x0; == H(seq)/seq_len
        "entropy_rate_bits": float(agg / np.log(2)),
        "entropy_rate_pos1plus_nats": float(H[1:].mean()),  # excludes x0 (matches _eval_per_level domain)
        "generative_ub_nats": float(gen_ub_nats),  # BP must be <= this
        "per_level": per_level,
        "per_position_nats": H.tolist(),
    }


# ------------------------------------------------------------------------
# Modal entrypoint
# ------------------------------------------------------------------------

try:
    from rhm.shared import app, volume, DATA_DIR
    _HAS_MODAL = True
except Exception:  # allow local self-test without modal
    _HAS_MODAL = False


if _HAS_MODAL:
    import modal

    @app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
    def bayes_entropy(v: int = 16, m: int = 2, s: int = 2, depth: int = 6,
                      n_seqs: int = 4000, rule_seed: int = 0, seq_seed: int = 999):
        """Exact per-level Bayes conditional entropy for the (v,s,L,m) DGP.

        Uses the SAME distinct-rule construction (generate_rules_distinct,
        rule_seed=0) as the FM-regularizer / thread_b substrate, so the floor is
        directly comparable to those models' per-level NTP loss.
        """
        from rhm.rhm_data import generate_rules_distinct, generate_sequences_batched

        L = depth
        rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
        seqs = generate_sequences_batched(rules, n_seqs, seed=seq_seed)

        out = summarize(rules, seqs)

        os.makedirs(f"{DATA_DIR}/rhm_bayes_entropy", exist_ok=True)
        path = f"{DATA_DIR}/rhm_bayes_entropy/v{v}_s{s}_L{L}_m{m}_rs{rule_seed}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        volume.commit()

        print(f"\n=== Bayes floor: v{v}/s{s}/L{L}/m{m} (distinct rules, seed {rule_seed}) ===")
        print(f"seq_len={out['seq_len']}  n_seqs={n_seqs}")
        print(f"generative UB (unique-parse) : {out['generative_ub_nats']:.4f} nats/token")
        print(f"entropy rate (BP, incl x0)   : {out['entropy_rate_nats']:.4f} nats/token "
              f"({out['entropy_rate_bits']:.4f} bits)")
        print(f"entropy rate (positions 1+)  : {out['entropy_rate_pos1plus_nats']:.4f} nats/token")
        print(f"H(x0) unconditional          : {out['H_x0_nats']:.4f} nats")
        print(f"\n  level   n_pos   Bayes floor (nats)   (bits)")
        for lvl, d in sorted(out["per_level"].items()):
            print(f"  {lvl:>5}   {d['n_positions']:>5}   {d['bayes_nats']:>16.4f}   {d['bayes_bits']:.4f}")
        print(f"\nsaved -> {path}")
        return out


# ------------------------------------------------------------------------
# Local correctness self-test: brute-force enumeration vs BP on a tiny tree
# ------------------------------------------------------------------------

def _brute_force_conditionals(rules):
    """Exact per-position conditional entropies by full enumeration (tiny only)."""
    import itertools
    v, m, s = rules[0].shape
    L = len(rules)
    seq_len = s ** L

    # Enumerate every latent config: root value + one rule choice per internal node.
    n_internal = (s ** L - 1) // (s - 1)
    seq_prob = {}
    for root in range(v):
        for choices in itertools.product(range(m), repeat=n_internal):
            # expand tree; choices consumed breadth-first (root, then depth1 nodes, ...)
            ci = 0
            current = [root]
            for d in range(L):
                nxt = []
                for node in current:
                    r = choices[ci]; ci += 1
                    nxt.extend(rules[d][node, r].tolist())
                current = nxt
            seq = tuple(current)
            p = (1.0 / v) * (1.0 / m) ** n_internal
            seq_prob[seq] = seq_prob.get(seq, 0.0) + p

    seqs = list(seq_prob.keys())
    probs = np.array([seq_prob[sq] for sq in seqs])
    arr = np.array(seqs)  # (N, seq_len)

    H = np.zeros(seq_len)
    for i in range(seq_len):
        # H(x_i | x_{<i}) = sum over prefixes P(prefix) * H(x_i | prefix)
        pref = {}
        for sq, p in zip(seqs, probs):
            key = tuple(sq[:i])
            pref.setdefault(key, {}).setdefault(sq[i], 0.0)
            pref[key][sq[i]] += p
        h = 0.0
        for key, dist in pref.items():
            tot = sum(dist.values())
            hc = -sum((q / tot) * np.log(q / tot) for q in dist.values())
            h += tot * hc
        H[i] = h
    return H, arr, probs


def _self_test():
    from rhm.rhm_data import generate_rules_distinct
    np.random.seed(0)
    for (v, s, L, m) in [(2, 2, 2, 2), (3, 2, 2, 2), (2, 2, 3, 2)]:
        rules = generate_rules_distinct(v, s, L, m, seed=1)
        H_bf, arr, probs = _brute_force_conditionals(rules)

        # sample many sequences from the true distribution, run BP, compare
        rng = np.random.default_rng(0)
        idx = rng.choice(len(arr), size=40000, p=probs)
        seqs = arr[idx]
        H_bp, agg = bp_conditional_entropies(rules, seqs)

        gen_ub = (np.log(v) + ((s ** L - 1) // (s - 1)) * np.log(m))
        Hseq_bf = H_bf.sum()
        max_err = np.abs(H_bf - H_bp).max()
        print(f"v{v}/s{s}/L{L}/m{m}: max |H_bf - H_bp| = {max_err:.4f} nats "
              f"| H(seq) bf = {Hseq_bf:.4f}  gen_UB = {gen_ub:.4f}  "
              f"(collisions: {'yes' if Hseq_bf < gen_ub - 1e-6 else 'no'})")
        assert max_err < 0.02, f"BP disagrees with brute force ({max_err})"
    print("self-test PASSED")


if __name__ == "__main__":
    _self_test()
