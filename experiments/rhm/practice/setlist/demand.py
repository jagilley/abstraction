"""Demand drift: the grammar is FIXED, what the world ASKS FOR moves.

WHY THIS EXISTS. `transpose` installed a live conditioning gap by resampling rule-table cells
-- TRUTH-NEWS, "what is so has changed" -- and found that committed chunks structurally cannot
consume it: a frozen level-3 table whose precision fell 1.000 -> 0.643 cost at most +0.047 in
audition error and ended at -0.006, a mined table that was 36% illegal still beat the *current*
true table by 0.04-0.06, and the recert channel fired 1 swap in 62. What moved instead was the
dense learner (plant parse 0.62 -> 0.50, infill 0.71 -> 0.57). The reading that produced this
round: a chunk is not a belief. It has no truth conditions. It stores DEMAND-CONCENTRATION -- a
prior over what the world asks, mined from having been asked -- so truth-news is simply not
denominated in its currency.

This round supplies the currency that is. Nothing becomes false; the ask moves.

WHAT "DEMAND" IS HERE. The task instance is (r*, damaged configuration): a target root and a
node that must be re-derived. Which vocabulary entry is *demanded* at that node is fixed by the
CLEAN derivation the damage was applied to -- by r* and by the rule choices on the path down to
the node. So the demand is exactly the generative distribution over clean derivations, and
drifting it re-weights which chunks get asked for while every chunk stays perfectly legal.

  * the ROOT PRIOR over r* (v-way) -- literally the request;
  * the MIXTURE WEIGHTS over which of the m synonymous rules each feature expands with, at the
    levels ABOVE the vocabulary being mined.

The level restriction is load-bearing and is why `demand_levels` defaults to the top two rule
layers. Re-weighting the mixture at level 2 would change which SYNONYM of a level-2 feature the
world produces -- and writing either synonym still derives that feature, so the repair succeeds
either way and the manipulation is a measured null by construction. Demand has to move which
FEATURE is asked for at the node, which is decided further up.

WHY `rhm_drift`'s OU MACHINERY IS THE RIGHT TOOL HERE, HAVING BEEN THE WRONG ONE THERE.
`transpose` rejected `full_loop`'s `advance_drift` / `calibrate_sigma_event` because it drifts
the synonym mixture weights: "it cannot invalidate a committed macro -- every entry stays
grammatical, only its frequency moves." That is precisely the property this round needs. The
support never changes, so the KL is finite and nats are the honest currency again (in
`transpose` the support changed and the only honest currency was a survival fraction). What is
adapted rather than reinvented: the OU form, mean-reverting so the shift process is itself
stationary; `weights_from_theta`; `sample_derivations_weighted`; the calibrate-sigma-to-a-target-
EVENT-magnitude discipline; and the prewarm-to-stationarity warning (an OU started at uniform
needs ~1/kappa steps to reach stationarity, so early events are systematically smaller and any
"decay since the start" reading is a warm-up artefact).

The one thing NOT adapted is the magnitude functional. `rhm_drift.drift_kl` measures KL per
SEQUENCE over the whole tree; the quantity this round is about is the demand over the LEVEL-l
VOCABULARY at the nodes the era damages. `demand_kl` measures that directly, so sigma is
calibrated in the currency the experiment reads.

WHAT IS DELIBERATELY UNTOUCHED, and why it is the round's admissibility gate. The grammar, the
true tables, `corrupt_hier`, and every grading path. So `given`'s full table is exactly correct
at every epoch, no entry ever becomes illegal, and the damage operator's difficulty is
unchanged. If the gate holds -- `d0`, `on_grammar` and the exact-DP `floor` flat while demand-KL
and coverage move -- then any post-commit movement is demand-specific BY CONSTRUCTION, which is
the decomposition `transpose` had to buy with a paired within-arm control.
"""

import numpy as np

from rhm.practice.crystallize.units import corrupt_hier
from rhm.practice.ratchet import macros as MC
from rhm.rhm_drift import sample_derivations_weighted, weights_from_theta
from rhm.rhm_sculpt_precheck import nearest_derivation_cost


# --------------------------------------------------------------------------- #
# the demand state
# --------------------------------------------------------------------------- #

def new_demand(rules, levels, seed=0, kappa=0.05, sigma=0.0, prewarm=True):
    """An OU state over (root logits, rule logits at `levels`).

    `levels` index `rules` root-ward, matching `rhm_drift`: 0 is the root layer. At depth 4 the
    default [0, 1] is root->L3 and L3->L2, i.e. the two layers above the mined vocabulary.

    PREWARMED to stationarity by default. `rhm_drift.advance_drift`'s docstring records why:
    an OU walk started at uniform needs ~1/kappa steps to reach stationarity, so the first
    events are systematically smaller than later ones and anything measured as "distance from
    the start" is a warm-up artefact rather than a property of the process. Prewarming also
    makes epoch 0 an honest starting point -- the audience already has tastes; what drifts is
    the tastes, not their existence.
    """
    v, m, _s = rules[0].shape
    rng = np.random.default_rng(seed + 8191)
    scale = 1.0 / np.sqrt(1.0 - (1.0 - kappa) ** 2)          # `rhm_drift.stationary_scale`
    mask = np.zeros(len(rules), bool)
    for ell in levels:
        mask[ell] = True
    st = {"theta_root": np.zeros(v), "theta": np.zeros((len(rules), v, m)),
          "mask": mask, "kappa": float(kappa), "sigma": float(sigma), "n_steps": 0}
    if prewarm and sigma > 0:
        st["theta_root"] = sigma * scale * rng.normal(size=v)
        for ell in np.flatnonzero(mask):
            st["theta"][ell] = sigma * scale * rng.normal(size=(v, m))
    return st


def typical_demand(rules, levels, s, depth, v, m, inv_bottom, seed=0, kappa=0.15, sigma=1.0,
                   n_cand=48, n=2048, level=2):
    """A REPRESENTATIVE stationary starting point instead of a single lottery draw.

    Prewarming to stationarity (`new_demand`) fixes the cold-start transient -- the walk no
    longer begins at maximum-entropy, tasteless demand -- but it replaces it with a subtler
    version of the same artefact: the start is ONE draw from the stationary law, and one draw
    can be unrepresentative. `cal0` caught exactly that. At seed 0 the prewarmed demand had
    entropy 1.86 against a process median of ~2.27, i.e. atypically concentrated, and the OU
    then relaxed toward typical demand. Because the *same* prewarm draw seeds every cell of a
    sweep, every cell inherited the same relaxation, and quantities that follow demand
    concentration (`stale`, the exact-DP `floor`, the true macro's ceiling) trended
    monotonically in the epoch index in cells with completely different sigma -- a signature no
    property of the drift itself could produce.

    So: draw `n_cand` stationary candidates, measure each one's demand entropy, and take the
    one closest to the median. Epoch 0 is then representative by construction and the walk is a
    wander around it rather than a relaxation away from an outlier. The candidate spread is
    returned so the lottery that was avoided is on the record rather than assumed away.
    """
    cands = []
    for j in range(n_cand):
        st = new_demand(rules, levels, seed=seed + 1000 * j, kappa=kappa, sigma=sigma,
                        prewarm=True)
        h = demand_hist(rules, st, level, s, depth, v, m, inv_bottom, n=n, seed=seed + 5)
        cands.append((demand_entropy(h), st))
    ents = [c[0] for c in cands]
    med = float(np.median(ents))
    best = min(cands, key=lambda c: abs(c[0] - med))
    return best[1], {"median_entropy": med, "chosen_entropy": best[0],
                     "candidate_min": float(min(ents)), "candidate_max": float(max(ents)),
                     "n_candidates": n_cand}


def demand_step(st, rng, n_steps=1):
    """`n_steps` OU steps: theta <- (1 - kappa) * theta + sigma * N(0, 1), on the root prior
    and on the drifting rule layers. Mean-reverting, so the shift-generating process is itself
    stationary -- the world's taste wanders, it does not run away."""
    k, sg = st["kappa"], st["sigma"]
    for _ in range(n_steps):
        st["theta_root"] = (1 - k) * st["theta_root"] + sg * rng.normal(size=st["theta_root"].shape)
        for ell in np.flatnonzero(st["mask"]):
            st["theta"][ell] = (1 - k) * st["theta"][ell] + sg * rng.normal(size=st["theta"][ell].shape)
        st["n_steps"] += 1
    return st


def copy_demand(st):
    return {**st, "theta_root": st["theta_root"].copy(), "theta": st["theta"].copy(),
            "mask": st["mask"].copy()}


def root_prior(st):
    z = st["theta_root"] - st["theta_root"].max()
    e = np.exp(z)
    return e / e.sum()


def rule_weights(rules, st):
    """Mixture weights implied by the state; exactly uniform on every non-drifting layer."""
    w = weights_from_theta(st["theta"])
    m = rules[0].shape[1]
    return [w[ell] if st["mask"][ell] else np.full_like(w[ell], 1.0 / m)
            for ell in range(len(rules))]


# --------------------------------------------------------------------------- #
# sampling under demand
# --------------------------------------------------------------------------- #

def sample_pool_demand(rules, n, s, seed, st):
    """`rhm_sculpt_planner._sample_pool` with the demand's root prior and mixture weights.

    NOT stream-identical to the parent even at uniform demand (`sample_derivations` draws
    `rng.integers`, the weighted sampler draws `rng.random`), which is why the runner keeps the
    parent function on the drift-off path rather than routing through here. Distributional
    equivalence at uniform weights is asserted in gate D-1."""
    rng = np.random.default_rng(seed)
    v = rules[0].shape[0]
    roots = rng.choice(v, size=n, p=root_prior(st))
    leaves = sample_derivations_weighted(rules, roots, s, rng, rule_weights(rules, st))
    return roots.astype(np.int64), leaves.astype(np.int64)


def context_instances_demand(rules, ctx, n, s, depth, v, m, seed, st, require_broken=True,
                             with_clean=False):
    """`ratchet.context_instances`, verbatim, with the clean derivation drawn from the CURRENT
    DEMAND. `corrupt_hier` and the rejection rule are untouched: the same level, the same node,
    the same "replace with a feature the observed subtree provably cannot produce". Only *which
    configurations arrive* moves."""
    length = s ** depth
    roots = np.zeros(n, np.int64)
    x = np.zeros((n, length), np.int64)
    clean = np.zeros((n, length), np.int64)
    rng = np.random.default_rng(seed + 90_001)
    need = np.arange(n)
    for attempt in range(64):
        if len(need) == 0:
            break
        r, lv = sample_pool_demand(rules, len(need), s, seed + 7919 * attempt, st)
        xd = corrupt_hier(lv, rules, depth, v, m, s, ctx["level"], ctx["nodes"], rng)
        ok = (nearest_derivation_cost(rules, xd, r, s) > 0) if require_broken \
            else np.ones(len(need), bool)
        roots[need[ok]] = r[ok]
        x[need[ok]] = xd[ok]
        clean[need[ok]] = lv[ok]
        need = need[~ok]
    if len(need):
        raise RuntimeError(f"context {ctx['name']}: {len(need)} instances never broke")
    return (roots, x, clean) if with_clean else (roots, x)


# --------------------------------------------------------------------------- #
# the demand over a level's vocabulary -- the currency this round is read in
# --------------------------------------------------------------------------- #

def demand_hist(rules, st, level, s, depth, v, m, inv_bottom, n=8192, seed=0):
    """P(entry) over the level-`level` vocabulary: draw `n` clean derivations from the current
    demand, read the level-1 span under EVERY level-`level` node, and count.

    Pooled over nodes because a committed macro is instantiated at every node of its level
    (`ratchet.macro_moves`), so the demand a table is graded against is position-independent --
    the same convention the committed action space already uses.
    """
    _r, lv = sample_pool_demand(rules, n, s, seed, st)
    feats = MC.exact_features(lv, inv_bottom, v, s)                 # (n, n_blocks)
    span = s ** (level - 1)
    n_nodes = feats.shape[1] // span
    spans = feats[:, :n_nodes * span].reshape(-1, span)
    keys, counts = np.unique(spans, axis=0, return_counts=True)
    tot = counts.sum()
    return {tuple(int(x) for x in k): c / tot for k, c in zip(keys, counts)}


def demand_kl(h_new, h_old, eps=1e-9):
    """KL(new || old) over the vocabulary demand, in nats. Finite by construction: the support
    of a mixture-weight drift never changes, which is exactly why `transpose` could not use
    this currency and this round can."""
    keys = set(h_new) | set(h_old)
    return float(sum(h_new.get(k, 0.0) * np.log((h_new.get(k, 0.0) + eps)
                                                / (h_old.get(k, 0.0) + eps))
                     for k in keys if h_new.get(k, 0.0) > 0))


def demand_entropy(h):
    return float(-sum(p * np.log(p) for p in h.values() if p > 0))


def coverage(table, hist):
    """Fraction of the CURRENT demand mass that a table can serve -- the demand-side analogue
    of `transpose`'s precision-against-current-truth, and the quantity a chunk actually stores.
    `given`'s full true table has coverage 1.0 at every epoch by construction, which is what
    makes it the demand-invariant denominator of the concentration premium."""
    if table is None or table["flat"].shape[0] == 0:
        return 0.0
    ent = {tuple(int(x) for x in r) for r in table["flat"]}
    return float(sum(p for k, p in hist.items() if k in ent))


# --------------------------------------------------------------------------- #
# the demand-matched oracle table (the concentration ceiling)
# --------------------------------------------------------------------------- #

def demand_table(hist, level, lower, s, cover=0.8):
    """The smallest table covering `cover` of the current demand: PERFECT CONCENTRATION, with
    no coverage the demand does not ask for. Built through `MC.Miner` so the ratchet nesting
    (`T[l]` defined over `T[l-1]` entries) is applied by the parent's own code."""
    order = sorted(hist.items(), key=lambda kv: -kv[1])
    keep, acc = [], 0.0
    for k, p in order:
        keep.append(k); acc += p
        if acc >= cover:
            break
    mn = MC.Miner(level, s)
    for k in keep:
        mn.counts[k] = 1
    mn.n_obs = len(keep)
    return mn.build(lower, 1)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def event_demand_kl(rules, levels, kappa, sigma, n_steps, level, s, depth, v, m, inv_bottom,
                    seed=0, n=8192):
    """The demand-KL of ONE drift event of `n_steps` OU steps, at stationarity. Common random
    numbers across sigma (the same seed draws both histograms and the same walk), so the
    calibration objective is smooth and monotone in sigma."""
    st = new_demand(rules, levels, seed=seed, kappa=kappa, sigma=sigma, prewarm=True)
    h0 = demand_hist(rules, st, level, s, depth, v, m, inv_bottom, n=n, seed=seed + 5)
    demand_step(st, np.random.default_rng(seed + 31), n_steps)
    h1 = demand_hist(rules, st, level, s, depth, v, m, inv_bottom, n=n, seed=seed + 5)
    return demand_kl(h1, h0)


def calibrate_sigma_demand(rules, levels, kappa, target_kl, n_steps, level, s, depth, v, m,
                           inv_bottom, seed=0, n=4096, lo=1e-3, hi=4.0, max_iter=18,
                           tol=5e-4):
    """Binary-search sigma so ONE drift event costs `target_kl` nats of demand -- the
    `calibrate_sigma_event` discipline, in the currency this round actually reads."""
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        got = event_demand_kl(rules, levels, kappa, mid, n_steps, level, s, depth, v, m,
                              inv_bottom, seed=seed, n=n)
        if abs(got - target_kl) < tol:
            return mid, got
        lo, hi = (mid, hi) if got < target_kl else (lo, mid)
    mid = 0.5 * (lo + hi)
    return mid, event_demand_kl(rules, levels, kappa, mid, n_steps, level, s, depth, v, m,
                                inv_bottom, seed=seed, n=n)


def simulate_demand(rules, levels, kappa, sigma, s, depth, v, m, inv_bottom, cycles, period,
                    drift_start, level=2, cover=0.8, seed=0, n=4096):
    """THE PAYBACK BRACKET, demand-side. A table built to cover `cover` of the demand at the
    first epoch is held frozen; its coverage of the CURRENT demand is read at every epoch.

    Too slow and the frozen table never loses coverage (the static case rebuilt); too fast and
    no table mined from 8 spans/cycle can ever track (nothing is worth compiling). The bracket
    is what these rows locate, and the half-life is the number the rate is set by.
    """
    st = new_demand(rules, levels, seed=seed, kappa=kappa, sigma=sigma, prewarm=True)
    rng = np.random.default_rng(seed + 77)
    lower = MC.base_table(v) if level == 2 else None
    h0 = demand_hist(rules, st, level, s, depth, v, m, inv_bottom, n=n, seed=seed + 5)
    frozen = demand_table(h0, level, lower, s, cover=cover)
    rows, prev = [], h0
    n_ep = 1 + max(0, (cycles - drift_start) // period + 1) if period > 0 else 1
    for ep in range(n_ep):
        if ep:
            demand_step(st, rng, 1)
        h = demand_hist(rules, st, level, s, depth, v, m, inv_bottom, n=n, seed=seed + 5)
        rows.append({"epoch": ep, "cycle": (drift_start + (ep - 1) * period) if ep else 0,
                     "kl_from_prev": demand_kl(h, prev), "kl_from_init": demand_kl(h, h0),
                     "entropy": demand_entropy(h),
                     "frozen_coverage": coverage(frozen, h),
                     "n_frozen": int(frozen["child"].shape[0])})
        prev = h
    return rows
