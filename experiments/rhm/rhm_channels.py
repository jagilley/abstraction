"""Multi-channel RHM: give the sequence a task-relevant tree plus irrelevant distractors.

WHY THIS EXISTS
---------------
The mjc E3 outer loop allocates a metered budget by `value = lprog x visits`, and its
ladder resolves only because the arm's geometry supplies three *kinds* of region:

  A   on-reach reducible      learnable AND relevant   -- the target
  B*  off-reach reducible     learnable, irrelevant    -- the trap for `lprog-only`
  D*  off-reach irreducible   unlearnable              -- the trap for `error-only` (noisy TV)

Porting that ladder to RHM sculpting hits a DGP wall: RHM rule usage is uniform by
construction (rules are `(v, m, s)` tables shared across every position and every root,
and the root is drawn uniformly), so there is no non-uniform "relevance" measure to
allocate over. Measured on the plain DGP: the normalised entropy of the per-level
feature marginal is >= 0.93 at every level in every setting we run, and conditioning on
the root decays back to that floor within two levels of the root. `visits` would be flat
and `lprog x visits` would collapse to `lprog` -- exactly the confounded null that
`ideas/two_timescale_value_loop.md` (2026-07-17 substrate correction) warns about.

The fix here (Jasper's): put information in the sequence that is NOT drawn from the RHM
tree. A distractor channel is *structurally* incapable of moving the parsed root, so it
is irrelevant by construction rather than by a corruption-distribution choice -- and it
can still be made learnable (a second, independent grammar) or unlearnable (iid uniform).
That reproduces E3's actual geometry, which was always "one target among five
distractors", not a graded allocation *within* the target.

It also dissolves a tension the idea doc leaves open: relevance now keys on
tree-vs-distractor rather than on how deep the drift is, so the level-of-drift knob is
free to sit at the surface (where the homeostatic climbing payoff is largest) while the
relevance signal stays live.

CONSTRUCTION
------------
One sequence is the concatenation of several fixed-width channels over a SHARED vocab:

  [ tree: s^L tokens ] [ struct_0: s^L0 ] [ struct_1: s^L1 ] [ noise_0 ] [ noise_1 ] ...

  * `tree`   -- the target grammar. Its root is the sculpting task's r*; the DP `d*`
                and the possible-set success are computed on this slice ALONE.
  * `struct` -- an independent RHM with its own rule table (different rule_seed) and its
                own depth/synonymity. Reducible, same marginal statistics, cannot touch r*.
  * `noise`  -- iid uniform over the same vocab. Exactly irreducible: per-token CE is
                pinned at log(v) for any amount of data, with no `amp < gear` tuning of
                the kind that ate three of E3's smoke runs.

Channels are at fixed positions, which mirrors E3 (the arm knows where region A is). What
the agent cannot read off for free is which channel is worth *investing* in -- that needs
the FM's learning-progress and the value's relevance, which is the question.

Block granularity matches sculpting exactly (`n_blocks = seq_len // s`, a block = one
level-1 node = `s` adjacent leaves), so the edit command space `k` indexes distractor
blocks on the same footing as tree blocks. That is deliberate: if distractor blocks were
not editable there would be no trap for `lprog-only` to fall into.

This module is the shared PRIMITIVE. The experiment that certifies it lives at
`directed_sculpting/verify_distractors.py`; drift over these channels is `rhm_drift.py`.
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_drift import (
    calibrate_sigma, calibrate_sigma_event, drift_kl, make_drift_state, ou_step,
    sample_derivations_weighted, state_weights, stationary_theta, uniform_weights)
from rhm.rhm_data import build_inverse_maps, generate_rules_distinct, parse_leaves
from rhm.rhm_sculpt_precheck import (nearest_derivation_cost, possible_sets,
                                     sample_derivations)
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume


app = modal.App("rhm-distractor-dgp", image=image)


# The L=5 target tree plus two structured distractors of differing tightness and two
# noise channels -- 5 allocation cells, close to E3's 6. Struct depths are 3 so each
# contributes 4 blocks (vs the tree's 16), keeping the tree the majority channel.
DEFAULT_SPEC = [
    {"kind": "tree", "name": "tree", "depth": 5, "m": 2, "rule_seed": 0},
    {"kind": "struct", "name": "structA", "depth": 3, "m": 2, "rule_seed": 101},
    {"kind": "struct", "name": "structB", "depth": 3, "m": 4, "rule_seed": 102},
    {"kind": "noise", "name": "noiseA", "n_blocks": 2},
    {"kind": "noise", "name": "noiseB", "n_blocks": 2},
]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def leaf_marginal(rules, v, s):
    """Exact unigram distribution over leaf tokens induced by a rule table.

    Propagates the (uniform) root distribution down through the rule tables. Used to
    give the noise channels the SAME unigram statistics as the tree -- see `make_layout`.
    """
    p = np.full(v, 1.0 / v)
    m = rules[0].shape[1]
    for layer in rules:
        nxt = np.zeros(v)
        for f in range(v):
            if p[f] == 0.0:
                continue
            for r in range(m):
                for i in range(s):
                    nxt[layer[f, r, i]] += p[f] / (m * s)
        p = nxt
    return p


def make_layout(v, s, spec=None, noise_match_marginal=True):
    """Build the channel layout. Returns a dict describing token/block spans per channel.

    Exactly one channel must have kind == "tree" -- it defines the task's root r*, and
    every grammar operation (parse, DP d*, possible-set success) reads only its slice.

    `noise_match_marginal` (default True) draws noise tokens from the TREE's exact leaf
    unigram distribution rather than from uniform. Without it the noise channels are
    trivially identifiable: a finite rule table makes the tree's leaf marginal lumpy
    (measured TV from uniform 0.18 at v=8,L=5,m=2) while iid-uniform noise sits at TV
    0.003, so a unigram histogram separates them before any conditional modelling. The
    policies under test (`error-only`, `lprog-only`, `value`) do not read unigram
    statistics, so this is hygiene rather than a live confound -- but it costs nothing
    and it stops the ladder resting on a distinction we did not intend to supply.
    Irreducibility is unaffected: the tokens are still iid, so conditional entropy equals
    marginal entropy and no amount of data predicts them.
    """
    spec = DEFAULT_SPEC if spec is None else spec
    trees = [c for c in spec if c["kind"] == "tree"]
    if len(trees) != 1:
        raise ValueError(f"exactly one 'tree' channel required, got {len(trees)}")

    channels, tok, blk = [], 0, 0
    for ch in spec:
        kind = ch["kind"]
        if kind in ("tree", "struct"):
            depth, m = ch["depth"], ch["m"]
            rules = generate_rules_distinct(v, s, depth, m, seed=ch["rule_seed"])
            n_tok = s ** depth
        elif kind == "noise":
            depth = m = rules = None
            n_tok = ch["n_blocks"] * s
        else:
            raise ValueError(f"unknown channel kind {kind!r}")
        if n_tok % s:
            raise ValueError(f"channel {ch['name']} width {n_tok} not a multiple of s={s}")
        channels.append({
            "name": ch["name"], "kind": kind, "depth": depth, "m": m,
            "rules": rules, "rule_seed": ch.get("rule_seed"),
            "tok0": tok, "tok1": tok + n_tok,
            "blk0": blk, "blk1": blk + n_tok // s,
        })
        tok += n_tok
        blk += n_tok // s

    tree = next(c for c in channels if c["kind"] == "tree")
    noise_p = leaf_marginal(tree["rules"], v, s) if noise_match_marginal else None
    for ch in channels:
        if ch["kind"] == "noise":
            ch["token_p"] = noise_p
    return {
        "v": v, "s": s, "channels": channels, "noise_token_p": noise_p,
        "total_len": tok, "n_blocks_total": blk,
        "tree": tree, "tree_slice": slice(tree["tok0"], tree["tok1"]),
        "tree_rules": tree["rules"], "tree_depth": tree["depth"], "tree_m": tree["m"],
        "tree_inverse_maps": build_inverse_maps(tree["rules"]),
    }


def token_channel_ids(layout):
    """(total_len,) int array: which channel each token position belongs to."""
    ids = np.empty(layout["total_len"], dtype=np.int64)
    for i, ch in enumerate(layout["channels"]):
        ids[ch["tok0"]:ch["tok1"]] = i
    return ids


def block_channel_ids(layout):
    """(n_blocks_total,) int array: which channel each block belongs to.

    This is the map the outer loop allocates over -- the RHM analog of E3's region list.
    """
    ids = np.empty(layout["n_blocks_total"], dtype=np.int64)
    for i, ch in enumerate(layout["channels"]):
        ids[ch["blk0"]:ch["blk1"]] = i
    return ids


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def sample_channel(layout, ch, roots, rng):
    """Sample one channel's tokens for a batch. `roots` is used only by grammar channels.

    A grammar channel carrying a `drift` state samples its rule choices from that state's
    mixture weights instead of uniformly (see `rhm_rule_drift`). With no drift state this
    is the plain generator, so every prior result stays reachable.
    """
    v, s = layout["v"], layout["s"]
    if ch["kind"] == "noise":
        size = (len(roots), ch["tok1"] - ch["tok0"])
        p = ch.get("token_p")
        return rng.integers(0, v, size=size) if p is None else rng.choice(v, size=size, p=p)
    if ch.get("drift") is not None:
        return sample_derivations_weighted(ch["rules"], roots, s, rng,
                                           state_weights(ch["rules"], ch["drift"]))
    return sample_derivations(ch["rules"], roots, s, rng)


def attach_drift(layout, channel_name, levels, kappa, target_kl, n_cells_per_level=None,
                 seed=0, calibrate="stationary", event_steps=None, init="uniform",
                 frozen=False, sigma=None):
    """Give a grammar channel a calibrated OU drift state over the named levels.

    `calibrate="stationary"` (default, back-compatible): `target_kl` is nats/sequence from
    UNIFORM at stationarity, per level. `calibrate="event"`: `target_kl` is the KL of ONE
    DRIFT EVENT of `event_steps` steps -- the currency §6's "at matched drift magnitude"
    actually means, and the one that keeps a level sweep honest (the stationary calibration
    left per-event magnitudes 4.5x apart across levels at L=4).

    `init="stationary"` starts the logits at a draw from the walk's own stationary law
    instead of at uniform. Combined with `frozen=True` (sigma and kappa both 0) this gives a
    STATIC but non-uniform world -- the entropy-matched baseline a drifting arm should be
    compared against. Anchoring the static world at uniform instead puts it at the simplex's
    maximum-entropy point, so every drifted world is systematically lower-entropy and
    therefore mechanically easier to render and to plan in; see `rhm_drift.stationary_theta`
    for the measurement (~30% on the block FM's stochasticity floor).

    `sigma` lets a caller pass a previously calibrated dict rather than re-solving it.
    """
    ch = next(c for c in layout["channels"] if c["name"] == channel_name)
    if ch["rules"] is None:
        raise ValueError(f"channel {channel_name!r} has no grammar to drift")
    if sigma is None:
        if calibrate == "event":
            if not event_steps:
                raise ValueError("calibrate='event' needs event_steps")
            sigma = calibrate_sigma_event(ch["rules"], layout["s"], levels, kappa, target_kl,
                                          event_steps, n_cells_per_level, seed=seed)
        else:
            sigma = calibrate_sigma(ch["rules"], layout["s"], levels, kappa, target_kl,
                                    n_cells_per_level, seed=seed)
    theta0 = (stationary_theta(ch["rules"], levels, kappa, sigma, n_cells_per_level,
                               seed=seed + 1) if init == "stationary" else None)
    ch["drift"] = make_drift_state(ch["rules"], levels, n_cells_per_level, seed=seed,
                                   theta0=theta0)
    ch["drift_sigma"] = {ell: 0.0 for ell in levels} if frozen else sigma
    ch["drift_kappa"] = 0.0 if frozen else kappa
    ch["drift_calibrated_sigma"] = sigma
    return ch


def drift_step(layout, rng):
    """Advance every channel's drift state by one OU step. Returns {name: KL/seq}."""
    out = {}
    for ch in layout["channels"]:
        if ch.get("drift") is None:
            continue
        ou_step(ch["drift"], ch["drift_kappa"], ch["drift_sigma"], rng)
        out[ch["name"]] = drift_kl(ch["rules"], layout["s"], uniform_weights(ch["rules"]),
                                   state_weights(ch["rules"], ch["drift"]))[0]
    return out


def sample_pool(layout, n, seed):
    """Generate `n` mixed sequences.

    Returns dict with `leaves` (n, total_len), `roots` (n,) = the TREE root (the task's
    r*), and `channel_roots` (n_channels list) for provenance.
    """
    v = layout["v"]
    rng = np.random.default_rng(seed)
    leaves = np.empty((n, layout["total_len"]), dtype=np.int64)
    channel_roots = []
    for ch in layout["channels"]:
        # every grammar channel draws its own independent root
        roots = rng.integers(0, v, size=n)
        leaves[:, ch["tok0"]:ch["tok1"]] = sample_channel(layout, ch, roots, rng)
        channel_roots.append(roots if ch["kind"] != "noise" else None)
    tree_idx = layout["channels"].index(layout["tree"])
    return {"leaves": leaves, "roots": channel_roots[tree_idx],
            "channel_roots": channel_roots}


def resample_channel(layout, leaves, ch_index, rng):
    """Return a copy of `leaves` with channel `ch_index` redrawn from its own process.

    This is the hook the block FM / generator uses at edit time: regenerating a *noise*
    block yields an unpredictable delta-z (E3's "fresh each substep" noisy TV), while
    regenerating a *struct* block yields a predictable one.
    """
    out = leaves.copy()
    ch = layout["channels"][ch_index]
    roots = rng.integers(0, layout["v"], size=len(leaves))
    out[:, ch["tok0"]:ch["tok1"]] = sample_channel(layout, ch, roots, rng)
    return out


# --------------------------------------------------------------------------- #
# Grammar operations -- deliberately read ONLY the tree slice
# --------------------------------------------------------------------------- #

def tree_leaves(layout, leaves):
    return np.ascontiguousarray(leaves[:, layout["tree_slice"]])


def parse_root(layout, leaves):
    """Bottom-up parse of the tree slice.

    CAVEAT: `build_inverse_maps` is last-writer-wins on ambiguous (`generate_rules_distinct`)
    rule tables, so this is only approximate -- measured on the plain undrifted generator at
    L=5,m=2, it calls just 3.9% of genuinely on-grammar sequences valid and recovers the true
    root 23% of the time. Use it for INVARIANCE comparisons (same input -> same answer), which
    is what `check_structural_irrelevance` needs. For validity or success use
    `possible_set_success`, which is exact and is what the sculpting harness grades on.
    """
    return parse_leaves(tree_leaves(layout, leaves), layout["tree_rules"],
                        layout["tree_inverse_maps"])


def possible_set_success(layout, leaves, target_roots):
    """Exact: is `target_roots` in the tree root's possible-set? The sculpting grader."""
    levels = possible_sets(layout["tree_rules"], tree_leaves(layout, leaves), layout["s"])
    root_p = levels[-1][:, 0, :]
    return root_p[np.arange(len(target_roots)), target_roots]


def dp_cost(layout, leaves, target_roots):
    """Exact DP d* -- min token edits to reach any valid r* derivation of the TREE."""
    return nearest_derivation_cost(layout["tree_rules"], tree_leaves(layout, leaves),
                                   target_roots, layout["s"])


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

def check_structural_irrelevance(layout, n=4096, seed=7):
    """P1: distractor content cannot move the parsed root or the DP cost.

    Randomises every non-tree token (far beyond what any edit could do) and asserts the
    grammar readouts are bit-identical.
    """
    # NOTE: the target-root rng must be independent of the pool's. Seeding both the same
    # makes `targets` land exactly on each sequence's own generating root (the pool draws
    # the tree's roots first), so d* == 0 everywhere and the DP half of the check passes
    # vacuously. Cost us one debugging round; keep the offset.
    rng = np.random.default_rng(seed + 4242)
    pool = sample_pool(layout, n, seed)
    leaves, roots = pool["leaves"], pool["roots"]
    targets = rng.integers(0, layout["v"], size=n)

    root_before, valid_before = parse_root(layout, leaves)
    d_before = dp_cost(layout, leaves, targets)

    scrambled = leaves.copy()
    tok_ids = token_channel_ids(layout)
    tree_idx = layout["channels"].index(layout["tree"])
    off = tok_ids != tree_idx
    scrambled[:, off] = rng.integers(0, layout["v"], size=(n, int(off.sum())))

    root_after, valid_after = parse_root(layout, scrambled)
    d_after = dp_cost(layout, scrambled, targets)

    return {
        "roots_identical": bool(np.array_equal(root_before, root_after)),
        "valid_identical": bool(np.array_equal(valid_before, valid_after)),
        "dp_identical": bool(np.array_equal(d_before, d_after)),
        "n_off_tree_tokens": int(off.sum()),
        "mean_dp_cost": float(d_before.mean()),
    }


def check_marginals(layout, n=20000, seed=11):
    """P5: channels must not be separable by unigram statistics alone.

    Reports each channel's unigram entropy (nats) and total-variation distance from
    uniform. If these are all close, a policy cannot tell the channels apart without
    modelling conditional structure -- which is the whole question.
    """
    pool = sample_pool(layout, n, seed)
    leaves, v = pool["leaves"], layout["v"]
    out = {}
    for i, ch in enumerate(layout["channels"]):
        seg = leaves[:, ch["tok0"]:ch["tok1"]].reshape(-1)
        counts = np.bincount(seg, minlength=v).astype(np.float64)
        p = counts / counts.sum()
        nz = p[p > 0]
        out[ch["name"]] = {
            "unigram_entropy_nats": float(-(nz * np.log(nz)).sum()),
            "tv_from_uniform": float(0.5 * np.abs(p - 1.0 / v).sum()),
        }
    out["_uniform_entropy_nats"] = float(np.log(v))
    return out
