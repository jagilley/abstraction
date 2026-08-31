"""tuning/burst — span-local irreducible noise bursts: schedule, construction, and the
model-free calibration of their magnitude.

Deliberately kept OUT of the Modal app (mirroring `../fourwall/lm/wall.py`, whose role this
file plays for the second event type) so every piece is auditable and gate-testable with no
GPU and no substrate.

THE EVENT. `fourwall/lm`'s rotation is index news: the grammar is untouched, only the
`w`<->`z` address moves. A NOISE BURST is the aleatoric null as an event: for a window of
`burst_len` steps, each leaf of the INDEXED SPAN (leaves `key_lo`..`key_hi`, i.e. exactly the
span the key addresses) is independently replaced, with probability `rho`, by a uniform draw
from the leaf alphabet. Nothing about the grammar or the map changes; the replaced tokens
carry no information about anything, so the excess they cost is IRREDUCIBLE by construction
and `skip` is the correct op.

WHY UNIFORM REPLACEMENT. Three properties the design needs, in order:

  (1) IRREDUCIBLE. A replaced token is drawn independently of the latent, of the prefix and
      of everything else. No predictor, however good, can beat the noise-aware mixture
      predictive  (1-rho) P_BP(x|prefix) + rho/v , and the excess of THAT over the clean
      Bayes level is a floor no learning can lower. `oracle_burst_curve` computes both the
      floor and the naive (noise-unaware) cost exactly, by belief propagation, with no model
      in the loop -- which is what makes the "skip is correct" claim model-free.
  (2) SPAN-LOCAL. Confined to leaves 0..15, so the trap the spec names -- "a naive global
      burst would be typed by span-locality alone" -- is closed. What is NOT closed, and is
      reported rather than engineered away, is that a corrupted indexed span is also the
      CONTEXT for the unindexed span, so a burst leaks outside its span more than a rotation
      does (a rotation moves one prefix token; the donor measures its out-span leak at
      ~+0.056 nats). `nll.out` is logged under both counterfactuals so the leak is visible.
  (3) MAGNITUDE-MATCHABLE. `rho` is a continuous knob on the indexed-span NLL spike, so it
      can be tuned to a rotation's spike -- which is what forces the typing gauges to earn
      the distinction rather than reading it off a surprisal difference.

RNG DISCIPLINE. Every draw here comes from a freshly-seeded `np.random.default_rng`, keyed
on (seed, step) for the consumed stream and on (eval_seed, rate index) for the shadow panel.
Nothing touches the batch sampler's generator or any global stream, so an arm's trajectory
is bit-identical to its donor twin at every step before the first burst, and the shadow
panel's corruption is the SAME draw at every checkpoint (the donor's `rand`-mode idiom:
a fixed draw contributes no drift of its own).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------------- #

def burst_windows(starts, length, max_steps):
    """[(start, end_exclusive), ...] for the consumed stream."""
    out = []
    for st in starts:
        st = int(st)
        if st < max_steps:
            out.append((st, int(min(st + length, max_steps))))
    return out


def burst_index(step, windows):
    """Which burst window `step` falls in, or -1."""
    for i, (a, b) in enumerate(windows):
        if a <= step < b:
            return i
    return -1


def burst_onsets(windows):
    return [a for a, _ in windows]


def overlaps(windows, other_steps, guard=0):
    """True if any burst window comes within `guard` of any step in `other_steps`
    (used by the gate to assert bursts never occupy a rotation instant)."""
    for a, b in windows:
        for r in other_steps:
            if a - guard <= r < b + guard:
                return True
    return False


# --------------------------------------------------------------------------- #
# the corruption
# --------------------------------------------------------------------------- #

def train_burst_rng(burst_seed, step):
    """Per-step, arm-independent: every reader in the run sees the SAME corruption."""
    return np.random.default_rng(int(burst_seed) + 9176 * int(step) + 1)


def shadow_burst_rng(eval_seed, rate_i):
    """Fixed per rate, so the shadow panel's corruption never drifts across checkpoints."""
    return np.random.default_rng(int(eval_seed) + 60013 * (int(rate_i) + 1))


def burst_draw(n, span, rho, v, rng):
    """(mask (n, span) bool, vals (n, span) int64) — independent uniform replacement."""
    mask = rng.random((n, span)) < float(rho)
    vals = rng.integers(0, int(v), size=(n, span))
    return mask, vals.astype(np.int64)


def apply_burst(leaf, key_lo, key_hi, mask, vals):
    """A COPY of `leaf` with the indexed span corrupted where `mask`."""
    out = np.array(leaf, copy=True)
    sub = out[:, key_lo:key_hi]
    sub[mask] = vals[mask]
    out[:, key_lo:key_hi] = sub
    return out


def corrupt(leaf, key_lo, key_hi, rho, v, rng):
    """Convenience: draw and apply in one call. Returns (corrupted, mask)."""
    n = leaf.shape[0]
    mask, vals = burst_draw(n, key_hi - key_lo, rho, v, rng)
    return apply_burst(leaf, key_lo, key_hi, mask, vals), mask


# --------------------------------------------------------------------------- #
# the exact, model-free calibration
# --------------------------------------------------------------------------- #

def predictive_probs(rules, leaf, clamp=None, chunk=512, soften=None):
    """Exact P_BP(x_p | x_{<p} [, clamped node]) for the REALISED token, per instance.

    `wall.exact_predictive` returns the per-position mean surprisal; this returns the
    (n, T) probability array, which is what the noise-aware readers below need.

    `soften=(lo, hi, rho)` replaces the hard one-hot leaf evidence on positions
    `lo:hi` by the TRUE noisy-channel likelihood  (1-rho) * onehot + rho / v , i.e. it
    tells belief propagation that the observed indexed-span leaves were corrupted at
    rate `rho`. That is the inference half of the noise-aware optimum; the output half
    is the mixture applied by the caller.
    """
    from rhm.conditional_revision.oracle import _upward, _downward, _leaf_evidence, _norm
    n, T = leaf.shape
    v = rules[0].shape[0]
    out = np.zeros((n, T))
    for lo_i in range(0, n, chunk):
        sub = leaf[lo_i:lo_i + chunk]
        b = sub.shape[0]
        nev = None
        if clamp is not None:
            lev, node, vals = clamp
            from rhm.practice.fourwall.lm.wall import _clamp_evidence
            nev = _clamp_evidence(rules, lev, node, np.asarray(vals)[lo_i:lo_i + chunk],
                                  b, v)
        for plen in range(T):
            ev = _leaf_evidence(sub, plen, v)
            if soften is not None and plen > 0:
                slo, shi, rho = soften
                hi_p = min(shi, plen)
                if hi_p > slo:
                    ev[:, slo:hi_p, :] = ((1.0 - rho) * ev[:, slo:hi_p, :]
                                          + rho / float(v))
            up = _upward(ev, rules, nev=nev)
            down = _downward(up, rules, nev=nev)
            post = _norm((up[len(rules)] * down[len(rules)])[:, plen, :][:, None, :])[:, 0, :]
            out[lo_i:lo_i + b, plen] = post[np.arange(b), sub[:, plen]]
    return np.clip(out, 1e-30, None)


def oracle_burst_curve(rules, leaf, z, key_level, key_node, key_lo, key_hi, rhos, v,
                       eval_seed=999, keyed=True, verbose=True):
    """The burst's cost curve with NO MODEL in the loop.

    For each rate `rho`, corrupt the indexed span, then run exact belief propagation on the
    CORRUPTED stream and read the indexed-span mean surprisal two ways:

      naive   -log P_BP(x_p | x_{<p}, key)                  a reader that does not know the
                                                            noise exists — the analogue of
                                                            the model at the burst instant
      mixture -log[(1-rho) P_BP(x_p | x_{<p}, key) + rho/v] the noise-AWARE optimum, i.e.
                                                            the IRREDUCIBLE floor

      robust  -log[(1-rho) P_BP^soft(x_p | x_{<p}, key) + rho/v] where P_BP^soft ALSO does
                                                            its inference under the noise
                                                            model — the true irreducible
                                                            floor for this event

    `robust - clean` is the part of the burst's cost that NO reader can remove: `skip` is
    correct exactly to the extent this is the whole story. `naive - robust` is what a reader
    that had somehow learned the noise channel could have saved, and it is not learnable in
    a 250-step unannounced window.

    NOTE, measured and reported rather than assumed: on this world the exact BP predictive
    is extremely peaked, so `naive` is dominated by the floor clip and is NOT a usable
    calibration target for the model's own spike (the model is far softer than exact BP).
    The rate is therefore calibrated against the MODEL, in `tune_lm::calib`; this curve's
    job is the irreducibility statement, not the rate.
    """
    clamp = (key_level, key_node, z) if keyed else None
    p_clean = predictive_probs(rules, leaf, clamp=clamp)
    span = slice(key_lo, key_hi)
    clean = float((-np.log(p_clean[:, span])).mean())
    out = {"clean": clean, "rows": []}
    for i, rho in enumerate(rhos):
        rng = shadow_burst_rng(eval_seed, i)
        cor, mask = corrupt(leaf, key_lo, key_hi, rho, v, rng)
        p = predictive_probs(rules, cor, clamp=clamp)[:, span]
        ps = predictive_probs(rules, cor, clamp=clamp,
                              soften=(key_lo, key_hi, rho))[:, span]
        naive = float((-np.log(p)).mean())
        mixture = float((-np.log((1.0 - rho) * p + rho / v)).mean())
        robust = float((-np.log((1.0 - rho) * ps + rho / v)).mean())
        row = {"rho": float(rho), "realised_rate": float(mask.mean()),
               "naive": naive, "mixture": mixture, "robust": robust,
               "d_naive": naive - clean, "d_mixture": mixture - clean,
               "d_robust": robust - clean, "reducible": naive - robust}
        out["rows"].append(row)
        if verbose:
            print(f"  rho {rho:<6.4g} realised {row['realised_rate']:.4f}  "
                  f"naive {naive:8.4f} (+{row['d_naive']:.4f})  "
                  f"mixture {mixture:.4f} (+{row['d_mixture']:.4f})  "
                  f"robust {robust:.4f} (+{row['d_robust']:.4f})  "
                  f"reducible {row['reducible']:.4f}", flush=True)
    return out


def match_rate(rows, target, key="d_naive"):
    """Log-linear interpolation of the rho whose spike equals `target`.

    Returns (rho, bracketed) — `bracketed` False means the target lies outside the ladder
    and the value is an extrapolation, which the caller must report as such.
    """
    xs = [r["rho"] for r in rows]
    ys = [r[key] for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    for a in range(len(xs) - 1):
        if (ys[a] - target) * (ys[a + 1] - target) <= 0 and ys[a + 1] != ys[a]:
            t = (target - ys[a]) / (ys[a + 1] - ys[a])
            lx = np.log(xs[a]) + t * (np.log(xs[a + 1]) - np.log(xs[a]))
            return float(np.exp(lx)), True
    # outside the ladder: extrapolate on the nearest segment
    if target < ys[0]:
        a, b = 0, 1
    else:
        a, b = len(xs) - 2, len(xs) - 1
    if ys[b] == ys[a]:
        return float(xs[b]), False
    t = (target - ys[a]) / (ys[b] - ys[a])
    lx = np.log(xs[a]) + t * (np.log(xs[b]) - np.log(xs[a]))
    return float(np.exp(np.clip(lx, np.log(1e-4), np.log(1.0)))), False
