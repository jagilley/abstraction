"""Support-fixed rule drift for RHM: an OU walk on the mixture over each cell's m synonyms.

WHY THIS SHAPE
--------------
`rhm_channels.py` gives the sequence a relevance structure; drift is what makes the
allocation question *permanent*. Measured there: every channel's learning progress falls
below the noise floor by step 7500 (tree at 4000), so a static geometry poses a live
allocation problem for barely a third of a training run. E3 kept its regions live with a
continuous OU walk on the per-region curl gains for exactly this reason.

The primitive is a **cell-wise OU walk on the mixture weights over that cell's m
synonymous rules**, where a cell is a (level, feature) pair. Design recovered from
`ideas/mjc_learnings_on_rhm.md` §3 (deleted in b727bfd; the deletion was about that doc's
allocation-over-NTP-cells framing, not about this primitive).

Three properties make this the right knob rather than a convenient one:

1. **Support-fixed.** The set of legal s-tuples never changes -- only how often each is
   used. So a sequence that parsed before still parses, `build_inverse_maps` is untouched,
   and the DP `d*` is *exactly* invariant (it depends on the rule support alone). This is
   what makes drift here immune to the catastrophic forgetting that made the wholesale
   rule-seed swap measure nothing (val on old rules 1.41 -> 4.51,
   `residual_decomposition/README.md:148`; hazard recorded at
   `beliefs/dimensionality_expansion.md:144`).

2. **Task-invariant by construction.** The root stays recoverable under any amount of
   mixture drift, so what the agent is graded on -- reach r* -- never moves. Deep structure
   is invariant and surface realisation churns, which is exactly the condition the
   homeostatic climbing argument needs. We are drifting the *encoding*, not the *content*.

3. **Continuous, not abrupt.** `a2a_forward/reaching/curiosity_drift.py:183-188` found an
   abrupt swap breaks a naive learning-progress drive while a smooth morph does not.

MAGNITUDE IS EXACT, NOT ESTIMATED
---------------------------------
`ideas/adaptive_core_and_hierarchy_climb.md` §6 wants the level sweep read at *matched
drift magnitude*, and flags that as design hygiene. Because the generative process is a
chain of independent categorical draws that differ ONLY in their weights, the KL per
sequence between the pre- and post-drift processes decomposes exactly:

    KL(new || old) = sum_{level l} sum_{feature f}  s^l * p_l(f) * KL(w_new[l,f] || w_old[l,f])

where `p_l(f)` is the feature marginal at depth l (propagated through the rule tables) and
`s^l` is the number of nodes at that depth. So drift magnitude is a closed form in nats
per sequence, in the same currency as the repair cost it is supposed to induce -- and
`calibrate_sigma` inverts it to give the per-level sigma that equalises magnitude across
the level sweep. Without this the sweep is confounded: a cell at depth l is instantiated
`s^l` times per sequence, so an unmatched sigma makes deep-level drift up to s^(L-1) times
weaker than surface drift (a factor of 16 at L=5).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #

def uniform_weights(rules):
    """Mixture weights matching plain RHM: uniform over each cell's m rules."""
    v, m, _s = rules[0].shape
    return [np.full((v, m), 1.0 / m) for _ in rules]


def weights_from_theta(theta):
    """Softmax over the last axis. Guarantees every rule keeps positive mass, which is
    what makes the drift support-fixed rather than support-changing."""
    z = theta - theta.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def feature_marginals(rules, s, weights=None):
    """Exact per-depth feature marginal `p_l` under the given mixture weights.

    Returns a list of length L+1: p[0] is the (uniform) root distribution, p[l] the
    distribution over features at depth l, p[L] over leaf tokens.
    """
    v, m, _s = rules[0].shape
    weights = uniform_weights(rules) if weights is None else weights
    p = np.full(v, 1.0 / v)
    out = [p.copy()]
    for ell, layer in enumerate(rules):
        nxt = np.zeros(v)
        w = weights[ell]
        for f in range(v):
            if p[f] == 0.0:
                continue
            for r in range(m):
                pw = p[f] * w[f, r]
                if pw == 0.0:
                    continue
                for i in range(s):
                    nxt[layer[f, r, i]] += pw / s
        out.append(nxt)
        p = nxt
    return out


def node_counts(rules, s, weights=None):
    """Expected number of (depth l, feature f) nodes per sequence: s^l * p_l(f).

    Shape (L, v). This is the multiplier that makes a cell's KL contribution depend on
    which level it sits at.
    """
    p = feature_marginals(rules, s, weights)
    return np.stack([(s ** ell) * p[ell] for ell in range(len(rules))])


# --------------------------------------------------------------------------- #
# Drift state
# --------------------------------------------------------------------------- #

def make_drift_state(rules, levels, n_cells_per_level=None, seed=0):
    """Pick which cells drift, and initialise their logits at uniform.

    `levels` -- iterable of depths (0 = root-ward) whose cells are allowed to drift.
    `n_cells_per_level` -- how many features per level drift (None = all v).
    """
    v, m, _s = rules[0].shape
    rng = np.random.default_rng(seed)
    theta = np.zeros((len(rules), v, m))
    mask = np.zeros((len(rules), v), dtype=bool)
    for ell in levels:
        if n_cells_per_level is None:
            mask[ell, :] = True
        else:
            mask[ell, rng.choice(v, size=n_cells_per_level, replace=False)] = True
    return {"theta": theta, "mask": mask, "levels": sorted(set(levels)),
            "sigma": {}, "kappa": 0.0}


def ou_step(state, kappa, sigma, rng):
    """One Ornstein-Uhlenbeck step on the drifting cells' logits.

    theta <- (1 - kappa) * theta + sigma * N(0, 1), applied only where `mask` is set.
    Mean-reverting to uniform, so the walk is stationary: the *shift distribution* does
    not itself drift. That matters -- `ideas/adaptive_core_and_hierarchy_climb.md` §9 says
    the inner->outer plasticity pathway only earns its keep when the shift-generating
    process is non-stationary, so a stationary walk is the regime where a hardcoded outer
    loop is the right call, which is what we intend to build.

    `sigma` may be a scalar or a per-level dict {level: sigma} (see `calibrate_sigma`).
    """
    theta, mask = state["theta"], state["mask"]
    for ell in range(theta.shape[0]):
        rows = np.flatnonzero(mask[ell])
        if rows.size == 0:
            continue
        sig = sigma[ell] if isinstance(sigma, dict) else sigma
        noise = rng.normal(0.0, sig, size=(rows.size, theta.shape[2]))
        theta[ell, rows] = (1.0 - kappa) * theta[ell, rows] + noise
    state["kappa"] = kappa
    state["sigma"] = sigma if isinstance(sigma, dict) else {
        ell: sigma for ell in state["levels"]}
    return state


def state_weights(rules, state):
    """Mixture weights implied by the current drift state (uniform where not drifting)."""
    w = weights_from_theta(state["theta"])
    out = []
    for ell in range(len(rules)):
        layer = w[ell].copy()
        layer[~state["mask"][ell]] = 1.0 / rules[0].shape[1]
        out.append(layer)
    return out


# --------------------------------------------------------------------------- #
# Magnitude
# --------------------------------------------------------------------------- #

def drift_kl(rules, s, w_old, w_new):
    """Exact KL(new || old) per sequence, in nats, decomposed per level.

    Returns (total, per_level array of length L). The node-count weighting is taken under
    the NEW process, matching the direction of the KL.
    """
    counts = node_counts(rules, s, w_new)               # (L, v)
    per_level = np.zeros(len(rules))
    for ell in range(len(rules)):
        a, b = w_new[ell], w_old[ell]
        kl = np.where(a > 0, a * (np.log(np.maximum(a, 1e-300))
                                  - np.log(np.maximum(b, 1e-300))), 0.0).sum(axis=-1)
        per_level[ell] = float((counts[ell] * kl).sum())
    return float(per_level.sum()), per_level


def stationary_scale(kappa):
    """SD of the OU logits at stationarity, per unit sigma.

    For `theta <- (1-kappa)*theta + sigma*eps` with eps ~ N(0,1), the stationary marginal
    is N(0, sigma^2 / (1 - (1-kappa)^2)) independently per coordinate.
    """
    return 1.0 / np.sqrt(1.0 - (1.0 - kappa) ** 2)


def stationary_kl(rules, s, level, kappa, sigma, n_cells_per_level=None, seed=0,
                  n_draws=2048):
    """Expected KL/sequence at stationarity for drift confined to `level`.

    Draws logits IID from the stationary Gaussian rather than simulating a correlated
    path. At kappa=0.05 the OU autocorrelation time is ~20 steps, so a 200-step path
    carries only ~10 effective samples and the estimate wobbles +-15%; IID draws from the
    known stationary law remove that entirely. Uses common random numbers across sigma
    (the logits are exactly `sigma * scale * z` for fixed standard normals `z`), so the
    calibration objective is deterministic and monotone.
    """
    v, m, _s = rules[0].shape
    base = uniform_weights(rules)
    st = make_drift_state(rules, [level], n_cells_per_level, seed=seed)
    mask = st["mask"]
    z = np.random.default_rng(seed + 1).normal(size=(n_draws, v, m))
    scale = sigma * stationary_scale(kappa)
    kls = []
    for draw in z:
        theta = np.zeros((len(rules), v, m))
        theta[level] = scale * draw
        w = weights_from_theta(theta)
        layers = []
        for j in range(len(rules)):
            layer = w[j].copy()
            layer[~mask[j]] = 1.0 / m
            layers.append(layer)
        kls.append(drift_kl(rules, s, base, layers)[0])
    return float(np.mean(kls))


def calibrate_sigma(rules, s, levels, kappa, target_kl, n_cells_per_level=None,
                    seed=0, n_draws=2048, tol=1e-4, max_iter=60):
    """Per-level sigma that makes each level's drift cost the same KL per sequence.

    Needed because a cell at depth l is instantiated s^l times per sequence: at L=5 a
    surface cell fires 16x per sequence and a root cell once, so a shared sigma makes the
    level sweep measure the level's fan-out rather than the level's invariance -- measured
    at ~12x spread in KL across levels at shared sigma.

    Returns {level: sigma}. Bisection on the closed-form stationary KL.
    """
    out = {}
    for ell in levels:
        lo, hi = 1e-4, 16.0
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            got = stationary_kl(rules, s, ell, kappa, mid, n_cells_per_level, seed, n_draws)
            if abs(got - target_kl) < tol:
                break
            lo, hi = (mid, hi) if got < target_kl else (lo, mid)
        out[ell] = 0.5 * (lo + hi)
    return out


# --------------------------------------------------------------------------- #
# Weighted generation
# --------------------------------------------------------------------------- #

def sample_derivations_weighted(rules, roots, s, rng, weights=None, return_trace=False):
    """`rhm_sculpt_precheck.sample_derivations`, but the rule choice at each node is drawn
    from that cell's mixture weights instead of uniformly.

    With `weights=None` (or all-uniform weights) this reduces exactly to the original --
    asserted in the verification entrypoint, so every prior result stays reachable.

    `return_trace` additionally yields the (level, feature, rule) choices per node, which
    lets the verifier compute the realised log-likelihood ratio and check it against the
    closed-form KL.
    """
    L = len(rules)
    _v, m, _s = rules[0].shape
    weights = uniform_weights(rules) if weights is None else weights
    current = np.asarray(roots)[:, None]
    trace = []
    for ell in range(L):
        B, width = current.shape
        w = weights[ell][current]                       # (B, width, m)
        cdf = np.cumsum(w, axis=-1)
        u = rng.random((B, width, 1)) * cdf[..., -1:]
        choices = (u > cdf).sum(axis=-1)                # (B, width)
        if return_trace:
            trace.append((current.copy(), choices.copy()))
        nxt = np.empty((B, width * s), dtype=np.int64)
        for j in range(width):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], choices[:, j]]
        current = nxt
    return (current, trace) if return_trace else current


def trace_loglik(weights, trace):
    """Sum of log mixture weights along a recorded derivation. (B,) array."""
    total = None
    for ell, (feats, choices) in enumerate(trace):
        lp = np.log(np.maximum(weights[ell][feats, choices], 1e-300)).sum(axis=1)
        total = lp if total is None else total + lp
    return total
