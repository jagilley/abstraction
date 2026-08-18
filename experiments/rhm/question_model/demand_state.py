"""The world's QUESTION STATE: an OU drift on the RHM's generative mixture, on a grid.

WHY A GRID (and why that is not a compromise)
---------------------------------------------
`practice/setlist/demand.py` drifts the full-dimensional logits -- `theta_root` (v,) and
`theta[level]` (v, m) -- by a continuous OU. That is the right object for the practice arc,
whose readout is a *table's coverage* of the demand and needs no posterior over the demand
state. This experiment's whole point is the posterior: the demand channel is
`KL(P(theta|x_{<=t+1}) || P(theta|x_{<=t}))`, so we need a filter, and a filter over R^80 is
not exact.

So the drift is reparametrised as a small number of SCALAR OU components, each riding a fixed
random direction in logit space, and each scalar lives on a K-point grid with a discretised-OU
transition kernel. The world's theta IS a grid point -- the discretisation is a property of the
world, not an approximation in the inference -- so the forward filter over the K^C states is
*exactly* Bayes-optimal and the three-way identity closes to Monte-Carlo error rather than to
grid error. Two further things the grid buys, both of which were open problems in `setlist`:

  * the stationary law is an eigenvector, computable exactly, so the prewarm is a draw from
    the true stationary distribution rather than `typical_demand`'s median-of-48 candidates.
    That hack existed because a continuous OU's stationary law cannot be enumerated; here it
    can, so the lottery is not avoided-by-heuristic but absent.
  * `H(theta)` is finite and known, which CAPS the demand channel at H(theta)/T nats per token
    and makes "the question channel is intrinsically small per token" a computable statement
    rather than a surprise.

What is inherited from `demand.py` / `rhm_drift.py` and not reimplemented: the OU form itself
(mean-reverting, so the shift-generating process is stationary), `weights_from_theta`'s softmax
parametrisation (support-fixed: every rule keeps positive mass, so nothing becomes false and
the KL stays finite), `sample_derivations_weighted`, and the discipline of calibrating the knob
in the currency the experiment reads.

COMPONENTS
----------
Three scalar components, chosen so the demand state is level-resolved in exactly the way
`setlist`'s demand-levels lesson says matters:

  root  -- the root prior over r*, "which feature is asked for". ONE draw per block, so a
           64-token window carries almost no information about it: this component is
           structurally in-context-unlearnable and can only be held in weights. That is not a
           defect, it is the negative control the laundering question needs.
  hi    -- rule mixtures on levels 0..2 (root-ward), 1+2+4 = 7 nodes per block.
  lo    -- rule mixtures on levels 3..5, 8+16+32 = 56 nodes per block: "which realisation is
           asked for", the component a 64-token window can actually infer.

WORLDS
------
  drift    -- theta walks by one OU step per block. The full-history filter has stationary
              residual uncertainty; a 64-token-window learner has strictly more.
  static   -- theta frozen at one stationary draw for the whole corpus. A full-history filter
              identifies it and its demand channel decays to zero; a window learner's does not.
              This is the pair that separates "the drift keeps the channel alive" (true for an
              ideal learner) from "the finite context window keeps it alive" (true for the
              transformer).
  incoh    -- every node redraws its rule from the theta-MARGINAL mixture, and every block its
              root from the theta-marginal root prior. Unigram statistics match `drift`
              exactly; what is destroyed is the within-block COHERENCE, i.e. the existence of a
              theta to infer. The matched control for "did the learner build a question model".
  uniform  -- sigma_s = 0. Dispatches to `rhm_latent_loop._generate_with_traces` itself, so the
              drift-off corpus is BIT-IDENTICAL to every prior run on this regime rather than
              merely equivalent (the `setlist` gate D-1 lesson: a weighted sampler draws
              `rng.random` where the parent draws `rng.integers`, so distributional equality is
              the most one can assert if the call is routed through the new path).
"""

import numpy as np

from rhm.rhm_drift import sample_derivations_weighted, weights_from_theta


# --------------------------------------------------------------------------- #
# spec, grid, transition
# --------------------------------------------------------------------------- #

DEFAULT_COMPONENTS = (("root", ()), ("hi", (0, 1, 2)), ("lo", (3, 4, 5)))


def components_for(L):
    """(root, hi = the top half of the rule layers, lo = the bottom half). At L = 6 this
    is the module docstring's default; the split exists so the demand state is
    level-resolved -- hi moves which FEATURE is asked, lo which REALISATION."""
    h = L // 2
    return (("root", ()), ("hi", tuple(range(h))), ("lo", tuple(range(h, L))))


def make_spec(K=7, sigma_s=1.0, kappa=0.15, half=2.0, dir_seed=0,
              components=None, L=None):
    """A question-state spec. `sigma_s` is the STATIONARY sd of each scalar component
    (the concentration knob) and `kappa` the mean-reversion rate (the mixing knob) --
    `setlist`'s lesson that these are two knobs and must be swept, not solved.

    sigma_s == 0 collapses the grid to the single state theta = 0 (uniform mixture).
    """
    if components is None:
        components = components_for(L) if L is not None else DEFAULT_COMPONENTS
    return {"K": int(K) if sigma_s > 0 else 1, "sigma_s": float(sigma_s),
            "kappa": float(kappa), "half": float(half), "dir_seed": int(dir_seed),
            "components": [(n, tuple(ls)) for n, ls in components]}


def grid(spec):
    """The K-point grid for one scalar component."""
    if spec["K"] == 1:
        return np.zeros(1)
    h = spec["half"] * spec["sigma_s"]
    return np.linspace(-h, h, spec["K"])


def transition(spec):
    """Discretised OU kernel P[i, j] = P(theta' = g[j] | theta = g[i]), row-normalised.

    Continuous form: theta' = (1 - kappa) theta + sigma_step * eps with
    sigma_step = sigma_s * sqrt(1 - (1 - kappa)^2), which is `rhm_drift.stationary_scale`
    read backwards (that function gives sigma_s per unit sigma_step).
    """
    g = grid(spec)
    K = g.size
    if K == 1:
        return np.ones((1, 1))
    k = spec["kappa"]
    sd = spec["sigma_s"] * np.sqrt(max(1.0 - (1.0 - k) ** 2, 1e-12))
    mu = (1.0 - k) * g
    P = np.exp(-0.5 * ((g[None, :] - mu[:, None]) / sd) ** 2)
    return P / P.sum(1, keepdims=True)


def stationary(P, iters=20000, tol=1e-14):
    """Exact stationary distribution of the grid chain (power iteration to machine tol)."""
    K = P.shape[0]
    p = np.full(K, 1.0 / K)
    for _ in range(iters):
        q = p @ P
        if np.abs(q - p).max() < tol:
            p = q
            break
        p = q
    return p / p.sum()


def n_components(spec):
    return len(spec["components"])


def n_states(spec):
    return spec["K"] ** n_components(spec)


def state_grid(spec):
    """(n_states, C) int array of per-component grid indices, and the joint stationary
    distribution over the product grid (components are independent OU walks)."""
    C = n_components(spec)
    K = spec["K"]
    idx = np.stack(np.meshgrid(*[np.arange(K)] * C, indexing="ij"), -1).reshape(-1, C)
    ps = stationary(transition(spec))
    pj = ps[idx].prod(1)
    return idx, pj / pj.sum()


def joint_transition(spec):
    """(n_states, n_states) transition over the product grid. Kronecker of C copies."""
    P = transition(spec)
    idx, _ = state_grid(spec)
    C = idx.shape[1]
    out = np.ones((idx.shape[0], idx.shape[0]))
    for c in range(C):
        out = out * P[np.ix_(idx[:, c], idx[:, c])]
    return out


# --------------------------------------------------------------------------- #
# directions and weights
# --------------------------------------------------------------------------- #

def make_directions(rules, spec):
    """One fixed unit-scale direction in logit space per component.

    Centred (so a shift along the direction is a genuine re-weighting, not a constant
    added to every logit) and unit-sd over the axis the softmax normalises, so
    `theta = 1` means "logits with sd 1" for every component alike -- which is what makes
    one sigma_s comparable across components.
    """
    v, m, _s = rules[0].shape
    L = len(rules)
    rng = np.random.default_rng(1_000_003 + spec["dir_seed"])
    dirs = {}
    for name, levels in spec["components"]:
        if name == "root":
            u = rng.normal(size=v)
            u = u - u.mean()
            dirs[name] = {"root": u / (u.std() + 1e-12)}
        else:
            per = {}
            for d in levels:
                u = rng.normal(size=(v, m))
                u = u - u.mean(1, keepdims=True)
                per[d] = u / (u.std(1, keepdims=True) + 1e-12)
            dirs[name] = {"rules": per}
    dirs["_L"] = L
    dirs["_v"] = v
    dirs["_m"] = m
    return dirs


def theta_weights(spec, dirs, theta):
    """(root_prior (v,), rule_weights [L x (v, m)]) for one theta vector (C,)."""
    L, v, m = dirs["_L"], dirs["_v"], dirs["_m"]
    root_logits = np.zeros(v)
    rule_logits = np.zeros((L, v, m))
    for c, (name, _levels) in enumerate(spec["components"]):
        if "root" in dirs[name]:
            root_logits = root_logits + theta[c] * dirs[name]["root"]
        else:
            for d, u in dirs[name]["rules"].items():
                rule_logits[d] = rule_logits[d] + theta[c] * u
    z = root_logits - root_logits.max()
    e = np.exp(z)
    return e / e.sum(), list(weights_from_theta(rule_logits))


def all_weights(spec, dirs):
    """Per-state (root_prior, rule_weights). Precomputed once; `n_states` is small."""
    idx, _ = state_grid(spec)
    g = grid(spec)
    out = []
    for row in idx:
        out.append(theta_weights(spec, dirs, g[row]))
    return out


def marginal_weights(spec, dirs):
    """The theta-MARGINAL mixture: sum_theta pi(theta) w(theta). The `incoh` world's
    weights -- identical unigram statistics to `drift`, no theta to infer."""
    _idx, pj = state_grid(spec)
    W = all_weights(spec, dirs)
    v, m, L = dirs["_v"], dirs["_m"], dirs["_L"]
    rp = np.zeros(v)
    rw = [np.zeros((v, m)) for _ in range(L)]
    for p, (r, w) in zip(pj, W):
        rp += p * r
        for d in range(L):
            rw[d] += p * w[d]
    return rp, rw


# --------------------------------------------------------------------------- #
# theta trajectories and corpora
# --------------------------------------------------------------------------- #

def theta_trajectory(spec, n_blocks, seed, world="drift"):
    """(n_blocks,) flat state indices. Prewarmed: block 0 is drawn from the EXACT
    stationary distribution of the grid chain, so there is no warm-up transient and no
    single-draw lottery to diagnose after the fact."""
    idx, pj = state_grid(spec)
    rng = np.random.default_rng(seed + 8191)
    ns = idx.shape[0]
    if ns == 1:
        return np.zeros(n_blocks, dtype=np.int64)
    if world == "static":
        return np.full(n_blocks, int(rng.choice(ns, p=pj)), dtype=np.int64)
    P = joint_transition(spec)
    out = np.zeros(n_blocks, dtype=np.int64)
    out[0] = int(rng.choice(ns, p=pj))
    for i in range(1, n_blocks):
        out[i] = int(rng.choice(ns, p=P[out[i - 1]]))
    return out


def sample_corpus(rules, spec, dirs, n_blocks, seed, world="drift", weights_cache=None):
    """Aligned blocks under the question state.

    Returns (seqs (n, T), level_features [L+1], level_rules [L], theta_idx (n,)).
    `level_features[d]` is (n, s^d) and `[L]` the leaves -- the same contract as
    `rhm_latent_loop._generate_with_traces`, which this dispatches to verbatim when the
    demand is off, so the drift-off corpus is bit-identical to every prior run.
    """
    from rhm.rhm_latent_loop import _generate_with_traces
    L = len(rules)
    v, m, s = rules[0].shape

    if world == "uniform" or n_states(spec) == 1:
        seqs, lf, lr = _generate_with_traces(rules, n_blocks, seed)
        return seqs, lf, lr, np.zeros(n_blocks, dtype=np.int64)

    if world == "incoh":
        rp, rw = marginal_weights(spec, dirs)
        rng = np.random.default_rng(seed)
        roots = rng.choice(v, size=n_blocks, p=rp)
        leaves, trace = sample_derivations_weighted(rules, roots, s, rng, rw,
                                                    return_trace=True)
        lf = [t[0] for t in trace] + [leaves]
        lr = [t[1] for t in trace]
        return leaves, lf, lr, np.full(n_blocks, -1, dtype=np.int64)

    W = weights_cache if weights_cache is not None else all_weights(spec, dirs)
    tidx = theta_trajectory(spec, n_blocks, seed, world=world)
    rng = np.random.default_rng(seed)
    T = s ** L
    seqs = np.zeros((n_blocks, T), dtype=np.int64)
    lf = [np.zeros((n_blocks, s ** d), dtype=np.int64) for d in range(L)]
    lf.append(np.zeros((n_blocks, T), dtype=np.int64))
    lr = [np.zeros((n_blocks, s ** d), dtype=np.int64) for d in range(L)]
    for st in np.unique(tidx):
        sel = np.flatnonzero(tidx == st)
        rp, rw = W[st]
        roots = rng.choice(v, size=sel.size, p=rp)
        leaves, trace = sample_derivations_weighted(rules, roots, s, rng, rw,
                                                    return_trace=True)
        seqs[sel] = leaves
        for d in range(L):
            lf[d][sel] = trace[d][0]
            lr[d][sel] = trace[d][1]
        lf[L][sel] = leaves
    return seqs, lf, lr, tidx


# --------------------------------------------------------------------------- #
# currencies (the calibration knobs are set in these units, not in sigma)
# --------------------------------------------------------------------------- #

def theta_entropy(spec):
    """H(theta) at stationarity, in nats -- the CAP on the demand channel per block."""
    _idx, pj = state_grid(spec)
    return float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())


def per_component_entropy(spec):
    p = stationary(transition(spec))
    h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {name: h for name, _ in spec["components"]}


def _feature_marginals(rules, root_prior, rule_w):
    """Exact per-depth feature marginal under (root_prior, rule_w). `rhm_drift`'s
    `feature_marginals` hard-codes a uniform root, which is exactly what drifts here."""
    L = len(rules)
    v, m, s = rules[0].shape
    p = np.asarray(root_prior, dtype=np.float64)
    out = [p.copy()]
    for d in range(L):
        nxt = np.zeros(v)
        w = rule_w[d]
        for a in range(v):
            for r in range(m):
                for k in range(s):
                    nxt[rules[d][a, r, k]] += p[a] * w[a, r]
        out.append(nxt / max(nxt.sum(), 1e-30))
    return out


def block_kl(rules, spec, dirs, theta_new, theta_old):
    """Exact KL(P_new || P_old) per BLOCK (one s^L sequence), in nats, decomposed as in
    `rhm_drift`'s docstring: a chain of independent categorical draws, so

        KL = KL(root_new || root_old)
             + sum_d sum_f s^d p_new_d(f) KL(w_new[d, f] || w_old[d, f])

    Finite by construction because the support never moves -- the property `transpose`
    could not have and this drift does (`demand.py`'s opening argument)."""
    rp1, rw1 = theta_weights(spec, dirs, theta_new)
    rp0, rw0 = theta_weights(spec, dirs, theta_old)
    L = len(rules)
    _v, _m, s = rules[0].shape
    kl = float((rp1 * (np.log(rp1) - np.log(rp0))).sum())
    pm = _feature_marginals(rules, rp1, rw1)
    for d in range(L):
        cell = (rw1[d] * (np.log(rw1[d]) - np.log(rw0[d]))).sum(1)      # (v,)
        kl += float((s ** d) * (pm[d] * cell).sum())
    return kl


def event_kl(rules, spec, dirs, n_steps=1, n_draws=256, seed=0):
    """Mean/sd nats of demand per drift EVENT of `n_steps` OU steps, at stationarity."""
    idx, pj = state_grid(spec)
    g = grid(spec)
    P = joint_transition(spec)
    rng = np.random.default_rng(seed + 4441)
    ns = idx.shape[0]
    if ns == 1:
        return 0.0, 0.0
    rows = []
    for _ in range(n_draws):
        i = int(rng.choice(ns, p=pj))
        j = i
        for _ in range(n_steps):
            j = int(rng.choice(ns, p=P[j]))
        rows.append(block_kl(rules, spec, dirs, g[idx[j]], g[idx[i]]))
    return float(np.mean(rows)), float(np.std(rows))
