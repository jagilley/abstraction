"""The multi-channel sculpting ENVIRONMENT: E3's inner loop, on the distractor DGP.

WHAT THIS IS
------------
`../rhm_channels.py` supplies the DGP (one task tree + structured distractors + noise);
`../rhm_drift.py` supplies support-fixed drift; `../rhm_repair_cost.py` supplies the meter
and the homeostatic readout. This module is the piece that was missing to run the loop:
the sculpting apparatus (controller / generator / block FM / MC value / beams) taught to
live on a MULTI-CHANNEL sequence, so that a move `k` indexes a block of ANY channel and the
grammar readouts still come from the tree slice alone.

Three things had to change relative to `../rhm_sculpt_latent.py`:

1.  **Per-block rendering tables.** Each channel has its own bottom-level rule table, so
    "regenerate block j" must render through *that block's* grammar. Noise blocks have no
    grammar; regenerating one redraws its tokens iid from the noise marginal. That is what
    makes the noise channels an *irreducible* target for the block FM (E3's noisy TV) while
    tree/struct blocks stay reducible -- exactly the property `verify_distractors.py` P2/P3
    certified at the token-CE level, now expressed in the FM's own currency.

2.  **Mixture rendering instead of canonical rendering** (`render="mixture"`, the default
    here; `render="canon"` reproduces the old deterministic behaviour). The old move
    rendered a block's inferred level-1 feature as `rules[L-1][f, 0]` -- the *canonical*
    synonym. Under a frozen generator that makes the edit dynamics a FIXED deterministic
    map, so support-fixed drift would be pure covariate shift and the FM would have nothing
    to re-adapt to. Drawing the synonym from the cell's CURRENT drifted mixture instead
    makes drift bite on the dynamics themselves -- the analog of E3's OU walk on the curl
    gains, and the literal reading of `rhm_drift`'s "we drift the ENCODING, not the
    CONTENT". The DP `d*` is unaffected: every synonym is a legal realisation of the same
    feature, so the feature assignment (which is what `d*` reads) does not move.

3.  **Corruption is confined to tree blocks.** Corrupting a distractor block cannot change
    `d*`, so it would be damage the task cannot see. E3 damaged the target region; so do we.

WHAT STAYS THE SAME
-------------------
The grader. `possible_set_success` and `dp_cost` read the tree slice only, so every
behavioural number is commensurable with the single-channel sculpting arc.

THE THREE TAPS (`ideas/adaptive_core_and_hierarchy_climb.md` §4)
----------------------------------------------------------------
One FM object, read three ways, which is what makes this one system rather than two:
  * forward, for control            -> `open_loop_beam` / `closed_loop_beam`
  * residual-derivative, to explore -> `counterfactual_lp` (the `e` tap)
  * rolled forward, for relevance   -> `forecast_visits` (the `p` tap)

`counterfactual_lp` is deliberately a COUNTERFACTUAL fit rather than a "did my error go
down" difference. A plain finite difference is degenerate in an allocation loop: a channel
you never collect from never improves, so its LP is zero forever and it can never be
chosen -- rich-get-richer, and the noisy-TV filter would be doing no work. Fitting a COPY
of the FM on that channel's monitor sample and reading held-out improvement asks the right
question ("what would I gain by spending here?"), and it rejects the noisy TV for the right
reason: fitting iid noise does not reduce held-out error.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Layout spec
# --------------------------------------------------------------------------- #

def make_spec(tree_depth=5, tree_m=2, struct_depths=(3, 3), struct_ms=(2, 4),
              noise_blocks=(2, 2), struct_shares=None, rule_seed_offset=0):
    """`DEFAULT_SPEC` generalised over tree depth, so the whole loop can be run at the depth
    where the sculpting arc is calibrated (L=4) as well as at the deeper L=5 the E3 port was
    specced for. Shapes are kept proportional: struct channels sit two levels below the tree
    and the noise channels are sized to keep the tree the majority of the blocks.

    `struct_shares` (default all-zero, i.e. the published geometry) gives each struct channel a
    SHARING DEPTH: how many of its top rule tables are spliced in from the tree. Nonzero requires
    that channel to be depth- and m-matched to the tree, since level i must mean the same scale in
    both. See `rhm_channels.share_rules_top` for what the knob means and why it leaves structural
    irrelevance untouched.

    `rule_seed_offset` shifts every channel's rule seed together, so a DGP-level claim can be
    measured across rule draws rather than only across training draws. 0 reproduces every prior
    layout exactly."""
    shares = [0] * len(struct_depths) if struct_shares is None else list(struct_shares)
    if len(shares) != len(struct_depths):
        raise ValueError(f"struct_shares needs one entry per struct channel "
                         f"(got {len(shares)} for {len(struct_depths)})")
    spec = [{"kind": "tree", "name": "tree", "depth": tree_depth, "m": tree_m,
             "rule_seed": 0 + rule_seed_offset}]
    for i, (d, m) in enumerate(zip(struct_depths, struct_ms)):
        entry = {"kind": "struct", "name": f"struct{chr(65 + i)}", "depth": int(d),
                 "m": int(m), "rule_seed": 101 + i + rule_seed_offset}
        if int(shares[i]):
            entry["share_top"], entry["share_from"] = int(shares[i]), "tree"
        spec.append(entry)
    for i, nb in enumerate(noise_blocks):
        spec.append({"kind": "noise", "name": f"noise{chr(65 + i)}", "n_blocks": int(nb)})
    return spec


# --------------------------------------------------------------------------- #
# Per-block rendering / parsing tables
# --------------------------------------------------------------------------- #

def build_block_tables(layout, device):
    """Torch tables indexed by BLOCK, so one vectorised move can act on any channel.

    Returns a dict of tensors:
      syn_blk     (n_blocks, v, max_m, s)  every synonym of every feature, per block's grammar
      m_blk       (n_blocks,)              how many of those synonyms are real
      w_blk       (n_blocks, v, max_m)     current mixture weights (uniform until drifted)
      bottom_blk  (n_blocks, v**s)         leaf-tuple code -> level-1 feature (-1 for noise)
      is_noise    (n_blocks,)              bool
      chan_blk    (n_blocks,)              channel index of each block
      noise_p     (v,)                     the noise channels' token marginal
      tree_blocks (n_tree_blocks,)         block indices belonging to the tree
    """
    import torch

    v, s = layout["v"], layout["s"]
    n_blocks = layout["n_blocks_total"]
    max_m = max((ch["m"] for ch in layout["channels"] if ch["m"]), default=1)

    syn = np.zeros((n_blocks, v, max_m, s), dtype=np.int64)
    m_of = np.ones(n_blocks, dtype=np.int64)
    bottom = np.full((n_blocks, v ** s), -1, dtype=np.int64)
    is_noise = np.zeros(n_blocks, dtype=bool)
    chan = np.zeros(n_blocks, dtype=np.int64)
    powers = (v ** np.arange(s)).astype(np.int64)

    for ci, ch in enumerate(layout["channels"]):
        chan[ch["blk0"]:ch["blk1"]] = ci
        if ch["kind"] == "noise":
            is_noise[ch["blk0"]:ch["blk1"]] = True
            continue
        bot = ch["rules"][-1]                       # (v, m, s) feature -> leaf tuple
        m = ch["m"]
        for b in range(ch["blk0"], ch["blk1"]):
            m_of[b] = m
            syn[b, :, :m, :] = bot
            for f in range(v):
                for r in range(m):
                    bottom[b, int((bot[f, r] * powers).sum())] = f

    # within-channel offset of each block, and a mask that forbids cross-channel attention.
    # These are what a channel-position-invariant forward model needs: with them, the function
    # it applies to the tree's blocks and to a depth-matched distractor's blocks is literally
    # the same function, so "the same grammar at a different position" is the same problem.
    offset = np.zeros(n_blocks, dtype=np.int64)
    for ch in layout["channels"]:
        offset[ch["blk0"]:ch["blk1"]] = np.arange(ch["blk1"] - ch["blk0"])
    chan_mask = chan[:, None] != chan[None, :]

    tree = layout["tree"]
    return {
        "offset_blk": torch.from_numpy(offset).to(device),
        "chan_mask": torch.from_numpy(chan_mask).to(device),
        "syn_blk": torch.from_numpy(syn).to(device),
        "m_blk": torch.from_numpy(m_of).to(device),
        "w_blk": _uniform_w_blk(m_of, v, max_m, device),
        "bottom_blk": torch.from_numpy(bottom).to(device),
        "is_noise": torch.from_numpy(is_noise).to(device),
        "chan_blk": torch.from_numpy(chan).to(device),
        "noise_p": torch.from_numpy(
            (layout["noise_token_p"] if layout["noise_token_p"] is not None
             else np.full(v, 1.0 / v)).astype(np.float32)).to(device),
        "tree_blocks": torch.arange(tree["blk0"], tree["blk1"], device=device),
        "powers": torch.from_numpy(powers).to(device),
        "n_blocks": n_blocks, "v": v, "s": s, "max_m": max_m,
        "n_channels": len(layout["channels"]),
        "channel_names": [c["name"] for c in layout["channels"]],
        "tree_channel": layout["channels"].index(tree),
    }


def _uniform_w_blk(m_of, v, max_m, device):
    import torch
    w = np.zeros((len(m_of), v, max_m), dtype=np.float32)
    for b, m in enumerate(m_of):
        w[b, :, :m] = 1.0 / m
    return torch.from_numpy(w).to(device)


def refresh_w_blk(tb, layout, device):
    """Rebuild the per-block synonym mixture from each channel's CURRENT drift state.

    This is the coupling that makes drift change the edit DYNAMICS rather than only the data
    distribution: a block's inferred feature is rendered by a synonym drawn from the drifted
    mixture, so the same command `k` on the same state produces a different `Δz` after drift.
    Noise blocks keep a uniform (unused) row so `multinomial` never sees an all-zero row.
    """
    import torch
    from rhm.rhm_drift import state_weights

    w = tb["w_blk"].clone().zero_()
    for ci, ch in enumerate(layout["channels"]):
        if ch["kind"] == "noise":
            w[ch["blk0"]:ch["blk1"], :, :] = 1.0 / tb["max_m"]
            continue
        m = ch["m"]
        if ch.get("drift") is not None:
            bw = torch.from_numpy(
                state_weights(ch["rules"], ch["drift"])[-1].astype(np.float32)).to(w.device)
        else:
            bw = torch.full((tb["v"], m), 1.0 / m, device=w.device)
        w[ch["blk0"]:ch["blk1"], :, :m] = bw[None]
    tb["w_blk"] = w
    return tb


def advance_drift(layout, rng, n_steps):
    """Advance every drifting channel by `n_steps` OU steps; report BOTH magnitudes.

    `kl_from_prev` is the size of THIS drift event -- the distance the world moved since the
    last measurement -- and is the quantity `ideas/adaptive_core_and_hierarchy_climb.md` §6
    means by "at matched drift magnitude". `kl_from_uniform` is the accumulated distance from
    the pristine DGP, which is what `rhm_channels.drift_step` returns and is NOT matched
    across rounds: an OU walk started at uniform needs ~1/kappa steps to reach stationarity,
    so early events are systematically smaller. Prewarm with `prewarm_drift` and read
    `kl_from_prev`, or the first rounds' "repair" numbers are divisions by a warm-up artefact
    (a smoke produced a repaired fraction of -22.8 that way).
    """
    from rhm.rhm_drift import drift_kl, ou_step, state_weights, uniform_weights
    out = {}
    for ch in layout["channels"]:
        if ch.get("drift") is None:
            continue
        w_prev = state_weights(ch["rules"], ch["drift"])
        for _ in range(n_steps):
            ou_step(ch["drift"], ch["drift_kappa"], ch["drift_sigma"], rng)
        w_now = state_weights(ch["rules"], ch["drift"])
        out[ch["name"]] = {
            "kl_from_prev": drift_kl(ch["rules"], layout["s"], w_prev, w_now)[0],
            "kl_from_uniform": drift_kl(ch["rules"], layout["s"],
                                        uniform_weights(ch["rules"]), w_now)[0]}
    return out


def snapshot_world(tb):
    """Clone the current per-block synonym mixture -- i.e. the WORLD the edit dynamics live in."""
    return tb["w_blk"].clone()


def set_world(tb, w):
    """Install a world snapshot. Lets one trained system be graded in several worlds without
    retraining, which is what separates "this agent is robust" from "this world is easier"."""
    tb["w_blk"] = w
    return tb


def stochasticity_floor(controller, generator, tb, x, k, *, render, gen, n_draws=4,
                        acted_only=False):
    """The normalised error NO forward model can beat in the CURRENT world, measured rather
    than assumed: redraw the same (x, k) `n_draws` times and read the residual around the
    conditional mean.

    Worth measuring per world, not once. Drift moves the synonym mixture AWAY from uniform,
    and `KL(w||uniform) = log m - H(w)` exactly -- so a drift event destroys precisely as much
    rendering entropy as the KL it creates. A drifted world therefore has LESS synonym
    ambiguity per edit and is mechanically easier to plan in. Comparing a drift-trained agent
    against a static-trained one in their own respective worlds would credit that free lunch to
    robustness. Same identity, third appearance: it is what makes raw CE blind to this drift
    (`rhm_repair_cost`) and what makes the drifted world easier to control in.
    """
    import torch
    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        deltas = []
        for _ in range(n_draws):
            x2 = regenerate_block(generator, x, k, tb, render=render, gen=gen)
            deltas.append(_block_state_chunked(controller, x2) - z)
        d = torch.stack(deltas)
        if acted_only:
            idx = k[None, :, None, None].expand(n_draws, -1, 1, d.shape[3])
            d = d.gather(2, idx)
        mu = d.mean(dim=0, keepdim=True)
        num = (d - mu).pow(2).sum()
        den = d.pow(2).sum().clamp_min(1e-12)
        return float(num / den) * n_draws / max(n_draws - 1, 1)


def prewarm_drift(layout, rng, n_steps=200):
    """Walk the OU process to stationarity before round 1, so every subsequent event is drawn
    from the same shift distribution. `kappa=0.05` gives an autocorrelation time of ~20 steps,
    so 200 steps is ~10 relaxation times."""
    from rhm.rhm_drift import ou_step
    for ch in layout["channels"]:
        if ch.get("drift") is None:
            continue
        for _ in range(n_steps):
            ou_step(ch["drift"], ch["drift_kappa"], ch["drift_sigma"], rng)


def repaired_fraction(damage, residual, floor):
    """(damage - residual) / damage, reported as nan when the event is below resolution.

    `floor` must come from the observed damage series, not a constant. `rhm_repair_cost`
    records the same lesson from the other side: a null gap of -0.0044 is statistically zero
    but a thousandfold above a 1e-6 guard, and dividing by it produced a confident
    "progress 1.052" on a drift that never happened.
    """
    if not np.isfinite(damage) or abs(damage) < floor:
        return float("nan")
    return float((damage - residual) / damage)


# --------------------------------------------------------------------------- #
# The move: regenerate one block, through that block's own grammar
# --------------------------------------------------------------------------- #

def regenerate_block(generator, x, k, tb, *, render="mixture", gen=None, chunk=16384):
    """Regenerate block `k` of each row. Returns a new (B, T) sequence.

    Grammar block: mask the block, let the generator infer its level-1 feature from the rest
    (the on-manifold move of `rhm_generative_planner`), then render that feature as a synonym
    drawn from the block's current mixture (`render="mixture"`) or its canonical synonym
    (`render="canon"`, the pre-drift behaviour of `_regenerate`).
    Noise block: redraw its `s` tokens iid from the noise marginal -- unpredictable by
    construction, which is what traps an `error-only` policy.
    """
    import torch

    if x.shape[0] > chunk:
        return torch.cat([regenerate_block(generator, x[i:i + chunk], k[i:i + chunk], tb,
                                           render=render, gen=gen, chunk=chunk)
                          for i in range(0, x.shape[0], chunk)], dim=0)

    s, v = tb["s"], tb["v"]
    batch = x.shape[0]
    pos = k[:, None] * s + torch.arange(s, device=x.device)[None, :]        # (B, s)
    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = generator.block_logits(obs)                                    # (B, n_blocks, v)
    feat = logits.gather(1, k[:, None, None].expand(-1, 1, v)).squeeze(1).argmax(-1)  # (B,)

    if render == "canon":
        choice = torch.zeros(batch, dtype=torch.long, device=x.device)
    else:
        w = tb["w_blk"][k, feat]                                            # (B, max_m)
        choice = torch.multinomial(w, 1, generator=gen).squeeze(1)
    tup = tb["syn_blk"][k, feat, choice]                                    # (B, s)

    noise_tok = torch.multinomial(tb["noise_p"], batch * s, replacement=True,
                                  generator=gen).view(batch, s)
    new = torch.where(tb["is_noise"][k][:, None], noise_tok, tup)
    return x.clone().scatter_(1, pos, new)


def corrupt_tree(leaves_np, tree_blocks_np, n_corrupt, v, s, rng):
    """Corrupt `n_corrupt` random TREE blocks per row to random symbols.

    Only tree blocks: a corrupted distractor block leaves `d*` untouched, so it would be
    damage the task is structurally unable to see.
    """
    out = leaves_np.copy()
    for b in range(out.shape[0]):
        for blk in rng.choice(tree_blocks_np, size=n_corrupt, replace=False):
            out[b, blk * s:(blk + 1) * s] = rng.integers(0, v, size=s)
    return out


# --------------------------------------------------------------------------- #
# Training the two frozen instruments on the multi-channel sequence
# --------------------------------------------------------------------------- #

def train_generator_channels(generator, train_leaves, tb, *, batch_size, n_steps, lr, device):
    """MLM over blocks, per-block grammar, noise blocks excluded from the loss.

    Noise blocks carry no feature to predict (that is the point of them), so including them
    would train the generator against label noise and blur the very distinction the ladder
    rests on. They are still visible as context.
    """
    import torch
    import torch.nn.functional as F

    s, n_blocks = tb["s"], tb["n_blocks"]
    blk_ids = torch.arange(n_blocks, device=device)
    generator.train()
    opt = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    report = max(1, n_steps // 5)
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, train_leaves.shape[0], (batch_size,))
        leaves = train_leaves[idx].to(device)
        codes = (leaves.view(batch_size, n_blocks, s) * tb["powers"]).sum(-1)
        feats = tb["bottom_blk"][blk_ids[None, :].expand(batch_size, -1), codes]
        n_mask = int(torch.randint(1, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        chosen = order[:, :n_mask]
        pos = chosen[:, :, None] * s + torch.arange(s, device=device)
        obs = leaves.clone().scatter_(1, pos.reshape(batch_size, -1),
                                      torch.full((batch_size, n_mask * s), -1,
                                                 device=device, dtype=leaves.dtype))
        logits = generator.block_logits(obs)
        sel = torch.zeros(batch_size, n_blocks, dtype=torch.bool, device=device)
        sel.scatter_(1, chosen, True)
        sel = sel & (feats >= 0)
        if not sel.any():
            continue
        loss = F.cross_entropy(logits[sel], feats[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            acc = (logits[sel].argmax(-1) == feats[sel]).float().mean().item()
            print(f"  generator step {step:5d}/{n_steps}: loss={loss.item():.4f} fill_acc={acc:.3f}")


# --------------------------------------------------------------------------- #
# Visited-state distribution and transitions
# --------------------------------------------------------------------------- #

def sample_states(layout, tb, generator, *, n, n_corrupt, presteps, seed, device,
                  render="mixture", gen=None):
    """Start states from the CURRENT world: fresh derivations, tree-corrupted, then a random
    number of generator pre-steps. Returns (x (n,T) torch, roots (n,) torch).

    Fresh every call -- `verify_repair_cost`'s harness bug was charging a meter for re-drawn
    pool sequences, i.e. measuring gradient steps wearing a costume. Each state here is a
    sequence the agent had to go and get.
    """
    import torch
    from rhm.rhm_channels import sample_pool

    rng = np.random.default_rng(seed)
    pool = sample_pool(layout, n, seed)
    tree_blocks = tb["tree_blocks"].cpu().numpy()
    x = torch.from_numpy(corrupt_tree(pool["leaves"], tree_blocks, n_corrupt,
                                      layout["v"], layout["s"], rng)).to(device)
    roots = torch.from_numpy(pool["roots"].astype(np.int64)).to(device)
    for _ in range(presteps):
        k = torch.randint(0, tb["n_blocks"], (n,), device=device, generator=gen)
        x = regenerate_block(generator, x, k, tb, render=render, gen=gen)
    return x, roots


def transition_targets(controller, generator, x, k, tb, *, render="mixture", gen=None):
    """(z, Δz) for command `k` from state `x`. One materialisation per transition."""
    import torch
    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        x2 = regenerate_block(generator, x, k, tb, render=render, gen=gen)
        z2 = _block_state_chunked(controller, x2)
    return z, z2 - z


def train_block_fm(fm, controller, generator, layout, tb, *, n_steps, batch_size, n_corrupt,
                   budget, lr, seed, render, gen, device, pool_size=20_000, pool_refresh=250):
    """Fit the block FM on the visited distribution, FRESH EVERY STEP.

    Mirrors `rhm_sculpt_latent._train_block_fm` exactly (corrupt start -> g random
    regenerations -> one command `k` -> predict Δz), with one addition: the clean-sequence
    pool is redrawn every `pool_refresh` steps from the CURRENT (possibly drifted) layout, so
    the FM tracks the world rather than a snapshot of it.

    Freshness is load-bearing and was learned the hard way. A first cut materialised a
    1024-state buffer and reused it for ~300 steps, i.e. ~40k distinct transitions across a
    12000-step run against the reference's ~3M. The FM flatlined at normalised error ~1.0
    against a floor of EXACTLY 0.000 (deterministic `canon` rendering) with delta_cos 0.15 vs
    the reference's 0.49 -- a readout that looked like "this task is unlearnable" and was
    really "this trainer saw 1% of the data".
    """
    import torch
    import torch.nn.functional as F
    from rhm.rhm_channels import sample_pool

    rng = np.random.default_rng(seed)
    tree_blocks = tb["tree_blocks"].cpu().numpy()
    fm.train()
    opt = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report = max(1, n_steps // 5)
    pool = None
    for step in range(1, n_steps + 1):
        if pool is None or step % pool_refresh == 1:
            pool = sample_pool(layout, pool_size, int(rng.integers(0, 2 ** 31)))["leaves"]
        idx = rng.integers(0, pool.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(corrupt_tree(pool[idx], tree_blocks, c, tb["v"], tb["s"],
                                          rng)).to(device)
        for _ in range(int(rng.integers(0, budget + 1))):
            kk = torch.randint(0, tb["n_blocks"], (batch_size,), device=device, generator=gen)
            x = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
        k = torch.randint(0, tb["n_blocks"], (batch_size,), device=device, generator=gen)
        with torch.no_grad():
            z = controller.block_state(x)
            x2 = regenerate_block(generator, x, k, tb, render=render, gen=gen)
            target = controller.block_state(x2) - z
        pred = fm(z, k)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            cos = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1).mean().item()
            print(f"  block FM step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cos:.3f}")
    fm.eval()
    return fm


def build_channel_local_fm():
    """A block FM with NO absolute-position parameters — the position-invariant readout.

    `rhm_sculpt_latent._build_block_fm` keys two embedding tables on the ABSOLUTE block index:
    `action_embedding[k]` marks which block the command touched, and `block_position[b]` tags
    each block. That makes the tree's blocks and a distractor's blocks different problems even
    when their grammars are bit-identical, which is what `partial_hetero`'s first pass measured:
    a tree-trained FM read on a distractor whose rule tables are the SAME tables scores above
    1.0, i.e. worse than predicting no change at all.

    Three changes, and nothing else:
      * attention MASKED to within-channel (`tb["chan_mask"]`), so a block's forecast is a
        function of its own channel's latents only -- which is also correct on the merits, since
        the channels are generatively independent;
      * WITHIN-CHANNEL offset embeddings (`tb["offset_blk"]`) instead of absolute block tags;
      * ONE shared acted marker instead of a per-block table.

    Plus per-block latent CENTERING (`set_center`). The frozen controller adds an absolute
    `position_embedding` before its encoder, so its block latents carry a positional component
    that a weight-shared FM would have to undo without ever seeing the target channel. The
    target needs no correction -- it is a difference of latents, so any additive positional term
    already cancels -- so centering the INPUT is the whole fix, and it is estimated from the
    frozen encoder's marginal statistics with no channel labels anywhere.

    Same `(z_block, k) -> (B, n_blocks, D)` signature as the published FM, so it drops into
    `per_channel_error`, `counterfactual_lp`, `forecast_visits` and both beams unchanged.
    """
    import torch
    import torch.nn as nn

    class ChannelLocalBlockFM(nn.Module):
        def __init__(self, state_dim, n_blocks, n_head, n_layer):
            super().__init__()
            self.acted = nn.Parameter(torch.randn(state_dim) * 0.02)
            self.offset_embedding = nn.Embedding(n_blocks, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.norm = nn.LayerNorm(state_dim)
            self.head = nn.Linear(state_dim, state_dim)
            self.register_buffer("mu", torch.zeros(n_blocks, state_dim), persistent=True)
            self.register_buffer("offsets", torch.zeros(n_blocks, dtype=torch.long),
                                 persistent=True)
            self.register_buffer("attn_mask", torch.zeros(n_blocks, n_blocks, dtype=torch.bool),
                                 persistent=True)

        def configure(self, tb):
            """Install the layout's offsets and the within-channel attention mask."""
            self.offsets.copy_(tb["offset_blk"].to(self.offsets.device))
            self.attn_mask.copy_(tb["chan_mask"].to(self.attn_mask.device))
            return self

        def set_center(self, mu):
            self.mu.copy_(mu.to(self.mu.device))
            return self

        def forward(self, z_block, k):
            batch, n_blocks, dim = z_block.shape
            marker = torch.zeros_like(z_block)
            marker.scatter_(1, k[:, None, None].expand(-1, 1, dim),
                            self.acted.view(1, 1, dim).expand(batch, 1, dim))
            hidden = ((z_block - self.mu[None])
                      + self.offset_embedding(self.offsets)[None] + marker)
            return self.head(self.norm(self.encoder(hidden, mask=self.attn_mask)))

    return ChannelLocalBlockFM


def block_latent_mean(controller, layout, tb, *, n, seed, device, chunk=4096):
    """Per-block mean of the frozen controller's block latents, on clean draws from the CURRENT
    world. Channel-label-free — it is a marginal statistic of the encoder, nothing more. This is
    what `build_channel_local_fm().set_center` subtracts."""
    import torch
    from rhm.rhm_channels import sample_pool

    leaves = sample_pool(layout, n, seed)["leaves"]
    total = torch.zeros(tb["n_blocks"], controller.token_embedding.weight.shape[1],
                        device=device)
    seen = 0
    with torch.no_grad():
        for i in range(0, n, chunk):
            x = torch.from_numpy(leaves[i:i + chunk]).to(device)
            total += controller.block_state(x).sum(0)
            seen += x.shape[0]
    return total / max(1, seen)


def _block_state_chunked(controller, x, chunk=4096):
    import torch
    if x.shape[0] <= chunk:
        return controller.block_state(x)
    return torch.cat([controller.block_state(x[i:i + chunk])
                      for i in range(0, x.shape[0], chunk)], dim=0)


def _fm_chunked(fm, z, k, chunk=16384):
    import torch
    if z.shape[0] <= chunk:
        return fm(z, k)
    return torch.cat([fm(z[i:i + chunk], k[i:i + chunk])
                      for i in range(0, z.shape[0], chunk)], dim=0)


def blocks_of_channel(tb, c):
    import torch
    return torch.nonzero(tb["chan_blk"] == c, as_tuple=False).squeeze(1)


# --------------------------------------------------------------------------- #
# The `e` tap: per-channel FM error and COUNTERFACTUAL learning progress
# --------------------------------------------------------------------------- #

def fm_error(fm, z, k, target, acted_only=False):
    """Normalised MSE: sum||pred - target||^2 / sum||target||^2 over the batch.

    Scale-free on purpose. Raw MSE is not comparable across channels -- a channel whose
    edits move the latent further has a larger raw error at equal predictive quality -- so a
    raw-error policy would be reading move magnitude, not reducibility. Normalised, 1.0 is
    "no better than predicting no change" and 0.0 is perfect, and the irreducible noise
    channels sit near 1.0 by construction. Raw MSE is reported alongside for the record.

    RATIO-OF-SUMS, NOT MEAN-OF-RATIOS. A generator regeneration frequently returns the block
    unchanged (the argmax infill re-picks the same feature and the mixture re-picks the same
    synonym), so `Δz` is EXACTLY zero on a sizeable minority of transitions. Per-transition
    normalisation then divides by ~0 and the readout explodes -- a first pass reported tree
    errors of 8e11 and, worse, ranked the arms by how often they happened to draw a null
    transition rather than by predictive quality.

    ACTED-BLOCK VARIANT. `acted_only=True` restricts the readout to the block the command
    touched. The full-tensor number is diluted by the many UNTOUCHED blocks, whose contextual
    ripple is small and largely unpredictable, so a healthy FM can look useless on it; the
    acted block is where a command's consequence actually lives and is the honest read of
    whether the forecast is any good.
    """
    import torch
    with torch.no_grad():
        pred = _fm_chunked(fm, z, k)
        if acted_only:
            idx = k[:, None, None].expand(-1, 1, target.shape[2])
            pred = pred.gather(1, idx)
            target = target.gather(1, idx)
        num = (pred - target).pow(2).flatten(1).sum(1)
        den = target.pow(2).flatten(1).sum(1)
        nmse = float(num.sum() / den.sum().clamp_min(1e-12))
        raw = float(num.mean() / (target.shape[1] * target.shape[2]))
    return nmse, raw


def fm_one_step_check(fm, value, controller, generator, tb, x, roots, *, render, gen, device):
    """Stage-3b's FM diagnostics, per channel: does the FM's predicted Δz point the right way,
    and does the FM-rolled value rank moves like the materialised one? `value_top1_agree` is
    the bottleneck the whole sculpting arc turns on."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        batch = x.shape[0]
        true_v = torch.empty(batch, tb["n_blocks"], device=device)
        pred_v = torch.empty(batch, tb["n_blocks"], device=device)
        cos = torch.zeros(tb["n_blocks"], device=device)
        for k in range(tb["n_blocks"]):
            kk = torch.full((batch,), k, device=device, dtype=torch.long)
            x2 = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            tz = _block_state_chunked(controller, x2)
            pz = z + _fm_chunked(fm, z, kk)
            cos[k] = F.cosine_similarity((pz - z).flatten(1), (tz - z).flatten(1), dim=-1).mean()
            true_v[:, k] = value(tz.mean(dim=1), roots)
            pred_v[:, k] = value(pz.mean(dim=1), roots)
        top1 = float((true_v.argmax(1) == pred_v.argmax(1)).float().mean())
        tc = true_v - true_v.mean(1, keepdim=True)
        pc = pred_v - pred_v.mean(1, keepdim=True)
        rank = float(((tc * pc).sum(1) / (tc.norm(dim=1) * pc.norm(dim=1)).clamp_min(1e-8)).mean())
        per_ch = np.zeros(tb["n_channels"])
        cnt = np.zeros(tb["n_channels"])
        chan = tb["chan_blk"].cpu().numpy()
        cosn = cos.cpu().numpy()
        for k in range(tb["n_blocks"]):
            per_ch[chan[k]] += cosn[k]
            cnt[chan[k]] += 1
    return {"delta_cos": float(cosn.mean()), "value_top1_agree": top1,
            "value_rank_corr": rank,
            "delta_cos_per_channel": (per_ch / np.maximum(cnt, 1)).tolist()}


def per_channel_error(fm, controller, generator, tb, x_probe, *, render, gen, seed=0):
    """Frozen-probe FM error per channel. Same states every round; only world and model move."""
    import torch
    out = {}
    g = torch.Generator(device=x_probe.device).manual_seed(seed)
    for c in range(tb["n_channels"]):
        blks = blocks_of_channel(tb, c)
        k = blks[torch.randint(0, len(blks), (x_probe.shape[0],), device=x_probe.device,
                               generator=g)]
        z, target = transition_targets(controller, generator, x_probe, k, tb,
                                       render=render, gen=gen)
        nmse, raw = fm_error(fm, z, k, target)
        out[c] = {"nmse": nmse, "raw_mse": raw,
                  "nmse_acted": fm_error(fm, z, k, target, acted_only=True)[0]}
    return out


def counterfactual_lp(fm, controller, generator, tb, x_mon, *, lp_steps, batch_size, lr,
                      meter, render, gen, device):
    """The `e` tap. For each channel: fit a COPY of the FM on that channel's monitor sample,
    read the held-out error drop. Charged to the monitor meter.

    Counterfactual rather than retrospective for two reasons (see module docstring): it does
    not require having already spent there, and fitting iid noise does not lower held-out
    error, so the noisy TV is rejected by the estimator itself rather than by a threshold.
    """
    import copy
    import torch
    import torch.nn.functional as F

    n = x_mon.shape[0]
    half = n // 2
    out = {}
    for c in range(tb["n_channels"]):
        blks = blocks_of_channel(tb, c)
        k = blks[torch.randint(0, len(blks), (n,), device=device)]
        z, target = transition_targets(controller, generator, x_mon, k, tb,
                                       render=render, gen=gen)
        meter.charge_monitor(n)
        e_before, _ = fm_error(fm, z[half:], k[half:], target[half:])
        probe = copy.deepcopy(fm)
        probe.train()
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(lp_steps):
            idx = torch.randint(0, half, (min(batch_size, half),), device=device)
            loss = F.mse_loss(probe(z[idx], k[idx]), target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
        probe.eval()
        e_after, _ = fm_error(probe, z[half:], k[half:], target[half:])
        out[c] = {"e_before": e_before, "e_after": e_after, "lp": e_before - e_after}
        del probe
    return out


def measure_floors(controller, generator, tb, x, *, meter, n_draws, render, gen, device,
                   seed=0):
    """Per-channel ALEATORIC FLOOR: the error no forward model can beat, measured by executing
    the same command from the same state `n_draws` times and reading the residual around the
    conditional mean. Charged to the monitor meter -- this is work the agent does, not a
    privileged lookup, and nothing about it needs a channel label.

    This is the honest replacement for the idea doc's "filtered by fresh-FM-ensemble
    invariance so it rejects the noisy TV" (§4). An ensemble filter infers irreducibility;
    repeat-execution MEASURES it, which is a thing RHM's exact re-executability hands us and
    a physical substrate does not.
    """
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for c in range(tb["n_channels"]):
        blks = blocks_of_channel(tb, c)
        k = blks[torch.randint(0, len(blks), (x.shape[0],), device=device, generator=g)]
        with torch.no_grad():
            z = _block_state_chunked(controller, x)
            deltas = []
            for _ in range(n_draws):
                x2 = regenerate_block(generator, x, k, tb, render=render, gen=gen)
                deltas.append(_block_state_chunked(controller, x2) - z)
                meter.charge_monitor(x.shape[0])
            d = torch.stack(deltas)
            idx = k[None, :, None, None].expand(n_draws, -1, 1, d.shape[3])
            d = d.gather(2, idx)                                  # acted block only
            mu = d.mean(dim=0, keepdim=True)
            num = float((d - mu).pow(2).sum())
            den = float(d.pow(2).sum()) + 1e-12
        out[c] = (num / den) * n_draws / max(n_draws - 1, 1)       # unbiased for n draws
    return out


def reducible_fraction(errors, floors, n_channels, eps=1e-6):
    """What fraction of a channel's CURRENT error is even removable: `(e - floor) / e`.

    The `e` tap, done as a STOCK rather than a FLOW -- and that is the whole fix. A
    counterfactual-fit LP measures marginal return, which early in training is dominated by
    how STARVED a channel is rather than how REDUCIBLE it is: under a block-proportional warm
    start the 1-block noise channels are furthest back on their learning curve, so probing
    them yields the biggest held-out drop and the noisy-TV *filter* becomes the noisy-TV
    *attractor* (measured mean LP: tree +0.044 < structA +0.053 < structB +0.057 < noise
    +0.064/+0.066 -- exactly inverted).

    Floor-relative reducibility cannot be fooled that way, because it does not ask where the
    channel sits on its curve. Measured on the same run: tree 0.709 ~ structA 0.712 >
    structB 0.500 > noise 0.454/0.444 -- correctly ordered. It is also far lower-variance,
    needing two error measurements instead of an SGD fit (the fit's noise, not its bias, is
    what actually cost `value` the ladder: implied tree share 61% against a realised 44% with
    sd 0.309).
    """
    return {c: max(errors[c] - floors[c], 0.0) / max(errors[c], eps) for c in range(n_channels)}


def ema(prev, now, alpha, n_channels):
    """Exponential smoothing of a per-channel tap. `learning progress` is a derivative
    estimate, and a one-shot derivative off a fresh sample is dominated by sampling noise --
    the arm that used it swung between 1% and 93% of its budget on the target across rounds."""
    if prev is None:
        return dict(now)
    return {c: (1.0 - alpha) * prev[c] + alpha * now[c] for c in range(n_channels)}


# --------------------------------------------------------------------------- #
# The `p` tap: forecast visitation by ROLLING the FM (no materialisation)
# --------------------------------------------------------------------------- #

def forecast_visits(fm, value, controller, tb, x0, roots, *, budget, device):
    """Roll the FM open-loop under the greedy planner and count which blocks it acts on.

    E3's relevance tap verbatim: the allocation signal for "where will the plan actually go"
    comes from running the forward model forward, not from executing anything. Returns the
    per-channel share of forecast moves, plus the mean |ΔV| the value assigns to acting on
    each channel (the direct readout of whether the VALUE devalues distractors).
    """
    import torch
    with torch.no_grad():
        z = _block_state_chunked(controller, x0)
        batch = z.shape[0]
        counts = torch.zeros(tb["n_channels"], device=device)
        dv_sum = torch.zeros(tb["n_channels"], device=device)
        dv_cnt = torch.zeros(tb["n_channels"], device=device)
        for _ in range(budget):
            v_now = value(z.mean(dim=1), roots)
            scores = []
            for k in range(tb["n_blocks"]):
                kk = torch.full((batch,), k, device=device, dtype=torch.long)
                scores.append(value((z + _fm_chunked(fm, z, kk)).mean(dim=1), roots))
            scores = torch.stack(scores, dim=1)                       # (B, n_blocks)
            dv = (scores - v_now[:, None]).abs().mean(dim=0)          # (n_blocks,)
            dv_sum.index_add_(0, tb["chan_blk"], dv)
            dv_cnt.index_add_(0, tb["chan_blk"], torch.ones_like(dv))
            best = scores.argmax(dim=1)
            counts.index_add_(0, tb["chan_blk"][best], torch.ones(batch, device=device))
            z = z + _fm_chunked(fm, z, best)
        visits = (counts / counts.sum().clamp_min(1)).cpu().numpy()
        dvalue = (dv_sum / dv_cnt.clamp_min(1)).cpu().numpy()
    return visits, dvalue


def ground_truth_relevance(layout, tb, generator, x0, roots, *, render, gen, device):
    """Reduction in the exact DP `d*` from acting on each channel -- the privileged answer to
    "which tokens are real". Zero for every non-tree channel by construction (P1), so any
    nonzero learned value-sensitivity off-tree is spurious.

    Two readouts, because they say different things: `mean` is what a random move in that
    channel does (negative on the tree -- a random regeneration usually makes things worse),
    and `best` is what the BEST move available in that channel does, which is the relevance a
    planner can actually cash in.
    """
    import torch
    from rhm.rhm_channels import dp_cost

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x0.cpu().numpy(), roots_np)
    gains = np.zeros((tb["n_blocks"], len(roots_np)))
    with torch.no_grad():
        for k in range(tb["n_blocks"]):
            kk = torch.full((x0.shape[0],), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x0, kk, tb, render=render, gen=gen)
            gains[k] = d_cur - dp_cost(layout, xk.cpu().numpy(), roots_np)
    chan = tb["chan_blk"].cpu().numpy()
    mean = np.zeros(tb["n_channels"])
    best = np.zeros(tb["n_channels"])
    for c in range(tb["n_channels"]):
        sel = chan == c
        mean[c] = gains[sel].mean()
        best[c] = gains[sel].max(axis=0).mean()
    return {"mean_dstar_gain": mean, "best_dstar_gain": best,
            "dstar_mean": float(d_cur.mean())}


def value_relevance_check(value, controller, generator, layout, tb, x0, roots, *, render,
                          gen, device):
    """Does the LEARNED value agree with the exact DP about which moves matter?

    For every one of the `n_blocks` moves, materialise it and record both the value it lands
    on and the DP `d*` it lands on. Reports, per channel: the mean SIGNED ΔV (does the value
    think acting here helps?), and globally: the fraction of the value's top-1 move that is a
    tree block, and the correlation between the value's move ranking and the DP's. This is the
    non-circular form of "the value system devalues distractors" -- the value was trained only
    on terminal possible-set success, never told which channel is which.
    """
    import torch
    from rhm.rhm_channels import dp_cost

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x0.cpu().numpy(), roots_np)
    n = x0.shape[0]
    dv = np.zeros((tb["n_blocks"], n))
    dd = np.zeros((tb["n_blocks"], n))
    with torch.no_grad():
        v_now = value(_block_state_chunked(controller, x0).mean(dim=1), roots).cpu().numpy()
        for k in range(tb["n_blocks"]):
            kk = torch.full((n,), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x0, kk, tb, render=render, gen=gen)
            dv[k] = value(_block_state_chunked(controller, xk).mean(dim=1),
                          roots).cpu().numpy() - v_now
            dd[k] = d_cur - dp_cost(layout, xk.cpu().numpy(), roots_np)
    chan = tb["chan_blk"].cpu().numpy()
    top1 = dv.argmax(axis=0)
    dvc = dv - dv.mean(axis=0, keepdims=True)
    ddc = dd - dd.mean(axis=0, keepdims=True)
    corr = float(np.mean((dvc * ddc).sum(0)
                         / np.maximum(np.linalg.norm(dvc, axis=0) * np.linalg.norm(ddc, axis=0),
                                      1e-9)))
    return {"mean_signed_dvalue": {tb["channel_names"][c]: float(dv[chan == c].mean())
                                   for c in range(tb["n_channels"])},
            "top1_share": {tb["channel_names"][c]: float((chan[top1] == c).mean())
                           for c in range(tb["n_channels"])},
            "value_vs_dp_rank_corr": corr}


# --------------------------------------------------------------------------- #
# Control: the two latent beams (the FM read forward)
# --------------------------------------------------------------------------- #

def open_loop_beam(controller, generator, fm, value, tb, layout, x0, roots, *, budget,
                   beam_width, render, gen, device, chunk_len=3, return_final=False):
    """BALLISTIC control: plan `chunk_len` steps entirely in imagination (roll the FM's own
    predictions, no re-grounding), COMMIT and execute them, re-ground, repeat.

    This is the grader that can see FM quality. The re-grounded (closed-loop) beam executes
    and re-encodes every step, so search substitutes for the forecast and the FM's quality is
    largely erased -- measured on this very substrate in the length-gen arc (closed-loop
    w64 c=1: arity-2 / arity-1 = 0.672 / 0.672, erased; open-loop: 0.439 / 0.178). E3 hit the
    same wall from the other side: its reactive control was near-saturated and the ladder
    resolved only on the sighted FM-error grader.

    `chunk_len` is a real parameter, not a convenience. A one-step FM's imagined rollout stays
    veridical only to ~6 steps on this task and both arities then collapse (length-gen Round
    3's boundary; a dedicated multi-step FM recovers only ~14% of that penalty). Committing
    the whole `edit_budget=10` plan in one shot therefore grades imagination drift rather than
    FM quality -- a first smoke returned 0.000 for every arm. Chunked commitment is also what
    E3's ballistic controller did: commit a plan segment, then re-ground.

    `return_final=True` additionally hands back the terminal sequences, so the SAME run can be
    read by the PAID grader (possible-set success) and by the REPORTED one (the controller's own
    log P(r*)). That pair is `RHM_EDIT_CONTROL`'s Layer-1/Layer-2 table, and their divergence is
    the wireheading signature. Default False leaves every existing caller bit-identical.
    """
    import torch
    from rhm.rhm_channels import possible_set_success

    with torch.no_grad():
        x = x0.to(device)
        roots = roots.to(device)
        left = budget
        while left > 0:
            h = min(chunk_len, left)
            left -= h
            plan = _imagine_plan(controller, fm, value, tb, x, roots, horizon=h,
                                 beam_width=beam_width, device=device)
            for t in range(h):
                x = regenerate_block(generator, x, plan[:, t], tb, render=render, gen=gen)
        succ = possible_set_success(layout, x.cpu().numpy(), roots.cpu().numpy())
    if return_final:
        return float(succ.mean()), x, roots
    return float(succ.mean())


def _imagine_plan(controller, fm, value, tb, x0, roots, *, horizon, beam_width, device):
    """Beam search purely in imagination from the TRUE latent of `x0`. Returns (B, horizon)
    move indices. Nothing is materialised, so the plan is a function of the FM alone."""
    import torch

    with torch.no_grad():
        z = _block_state_chunked(controller, x0)
        batch = z.shape[0]
        n_blocks, dim = z.shape[1], z.shape[2]
        beams_z = z[:, None]                                    # (B, W, n_blocks, D)
        moves = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        for _ in range(horizon):
            flat = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for k in range(tb["n_blocks"]):
                kk = torch.full((batch * width,), k, device=device, dtype=torch.long)
                scores.append(value((flat + _fm_chunked(fm, flat, kk)).mean(dim=1), roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * tb["n_blocks"])
            keep = min(beam_width, width * tb["n_blocks"])
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, tb["n_blocks"], rounding_mode="floor")
            move = top % tb["n_blocks"]
            # recompute the FM for the KEPT (parent, move) pairs only. Caching every
            # candidate's predicted latent instead would need a (B, W*n_blocks, n_blocks, D)
            # tensor -- 2.5 GB at B=512/W=16 here -- for one extra forward pass saved.
            parent_z = beams_z.gather(
                1, parent[:, :, None, None].expand(-1, -1, n_blocks, dim))
            fz = parent_z.reshape(batch * keep, n_blocks, dim)
            beams_z = (fz + _fm_chunked(fm, fz, move.reshape(-1))).reshape(
                batch, keep, n_blocks, dim)
            prev = (moves.gather(1, parent[:, :, None].expand(-1, -1, moves.shape[2]))
                    if moves.shape[2] else moves.new_zeros(batch, keep, 0))
            moves = torch.cat([prev, move[:, :, None]], dim=2)
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, dim),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
    return moves[torch.arange(batch, device=device), best]          # (B, horizon)


def closed_loop_beam(controller, generator, fm, value, tb, layout, x0, roots, *, budget,
                     beam_width, render, gen, device):
    """Re-grounded (Dreamer-style) latent beam: rank by a ONE-step FM prediction from the
    beam's TRUE latent, materialise only the kept tips. The near-blind control."""
    import torch
    from rhm.rhm_channels import possible_set_success

    with torch.no_grad():
        x0 = x0.to(device)
        roots = roots.to(device)
        batch, length = x0.shape
        beams_x = x0[:, None, :]
        beams_z = _block_state_chunked(controller, x0)[:, None]
        width = 1
        for _ in range(budget):
            n_blocks, dim = beams_z.shape[2], beams_z.shape[3]
            flat_z = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for k in range(tb["n_blocks"]):
                kk = torch.full((batch * width,), k, device=device, dtype=torch.long)
                scores.append(value((flat_z + _fm_chunked(fm, flat_z, kk)).mean(dim=1), roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * tb["n_blocks"])
            keep = min(beam_width, width * tb["n_blocks"])
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, tb["n_blocks"], rounding_mode="floor")
            move = top % tb["n_blocks"]
            parent_x = beams_x.gather(1, parent[:, :, None].expand(-1, -1, length))
            flat_x = parent_x.reshape(batch * keep, length)
            new_x = regenerate_block(generator, flat_x, move.reshape(-1), tb,
                                     render=render, gen=gen)
            beams_x = new_x.reshape(batch, keep, length)
            beams_z = _block_state_chunked(controller, new_x).reshape(
                batch, keep, beams_z.shape[2], beams_z.shape[3])
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, beams_z.shape[3]),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
        x_final = beams_x[torch.arange(batch, device=device), best]
        succ = possible_set_success(layout, x_final.cpu().numpy(), roots.cpu().numpy())
    return float(succ.mean())


# --------------------------------------------------------------------------- #
# Value collection on the multi-channel task
# --------------------------------------------------------------------------- #

def collect_value_buffer(controller, generator, layout, tb, *, n_episodes,
                         batch_size, n_corrupt, budget, epsilon, seed, render, gen, device):
    """Monte-Carlo value data: roll a cheap behaviour policy from corrupt starts and label
    EVERY visited state by its rollout's terminal possible-set success on the TREE.

    The behaviour policy is controller-greedy over all 28 blocks (distractors included) with
    eps-random exploration -- deliberately NOT restricted to the tree, so that whether the
    learned value devalues distractors is measured rather than assumed.
    """
    import torch
    from rhm.rhm_channels import possible_set_success, sample_pool

    rng = np.random.default_rng(seed)
    tree_blocks = tb["tree_blocks"].cpu().numpy()
    configs, roots_all, succ_all = [], [], []
    done = 0
    while done < n_episodes:
        b = min(batch_size, n_episodes - done)
        done += b
        pool = sample_pool(layout, b, int(rng.integers(0, 2 ** 31)))
        c = int(rng.integers(1, n_corrupt + 1))
        start = corrupt_tree(pool["leaves"], tree_blocks, c, layout["v"], layout["s"], rng)
        x = torch.from_numpy(start).to(device)
        roots = torch.from_numpy(pool["roots"].astype(np.int64)).to(device)
        traj = [x.clone()]
        for _ in range(budget):
            x = _behavior_step(controller, generator, x, roots, tb, epsilon,
                               render=render, gen=gen, device=device)
            traj.append(x.clone())
        succ = torch.from_numpy(
            possible_set_success(layout, x.cpu().numpy(), pool["roots"]).astype(np.float32))
        for st in traj:
            configs.append(st.cpu())
            roots_all.append(roots.cpu())
            succ_all.append(succ)
    return torch.cat(configs), torch.cat(roots_all), torch.cat(succ_all)


def _behavior_step(controller, generator, x, roots, tb, epsilon, *, render, gen, device):
    import torch
    batch = x.shape[0]
    scores, props = [], []
    with torch.no_grad():
        for k in range(tb["n_blocks"]):
            kk = torch.full((batch,), k, device=device, dtype=torch.long)
            p = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            sc = controller.root_logits(controller.state(p)).log_softmax(-1) \
                .gather(1, roots[:, None]).squeeze(1)
            scores.append(sc)
            props.append(p)
        scores = torch.stack(scores, dim=1)
        props = torch.stack(props, dim=1)
        chosen = scores.argmax(dim=1)
        explore = torch.rand(batch, device=device) < epsilon
        chosen = torch.where(explore, torch.randint(0, tb["n_blocks"], (batch,), device=device),
                             chosen)
        return props.gather(1, chosen[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)


# --------------------------------------------------------------------------- #
# Belief instrumentation on the tree slice
# --------------------------------------------------------------------------- #

def make_tree_probe(layout, tb, seed):
    """A FROZEN probe: fixed tree derivations (with ground-truth ancestors) embedded in fresh
    distractor context. Support-fixed drift never moves the rule tables, so these labels stay
    valid for the whole run (`verify_backcompat` B5) -- the depth half of the two-instrument
    read is available without re-labelling."""
    from rhm.rhm_channels import sample_pool
    from rhm.rhm_latent_loop import _generate_with_traces

    tree = layout["tree"]
    n = 3000
    leaves_tree, level_feats, _ = _generate_with_traces(tree["rules"], n, seed)
    pool = sample_pool(layout, n, seed + 5)
    full = pool["leaves"].copy()
    full[:, tree["tok0"]:tree["tok1"]] = leaves_tree
    return full.astype(np.int64), level_feats, pool


def tree_depth_probe(controller, probe_leaves, level_feats, layout, tb, device,
                     probe_steps=300):
    """Per-level linear recovery of a tree block's ancestor feature from its per-block latent.

    Reads the climb DIRECTLY, which is what stops "fewer rounds to recover" being confused
    with "warmer model" (§6): climbing says repair cost falls AND depth rises; ordinary
    continued training says depth stays put.
    """
    import torch
    from rhm.rhm_sculpt_deepbelief import _block_ancestor_index, _probe_acc

    tree = layout["tree"]
    L, s, v = tree["depth"], layout["s"], layout["v"]
    n_tree_blocks = (tree["tok1"] - tree["tok0"]) // s
    anc = _block_ancestor_index(n_tree_blocks, s, L)
    x = torch.from_numpy(probe_leaves).to(device)
    with torch.no_grad():
        zb = _block_state_chunked(controller, x)[:, tree["blk0"]:tree["blk1"]]
    n_seq, nb, dim = zb.shape
    X = zb.reshape(n_seq * nb, dim)
    out = {}
    for ell in range(L):
        y = torch.from_numpy(level_feats[ell][:, anc[ell]].reshape(-1)).to(device)
        out[f"d{ell + 1}"] = _probe_acc(X, y, v, device, probe_steps, 1e-3)
    from rhm.rhm_sculpt_deepbelief import _participation_ratio
    out["PR"] = _participation_ratio(X)
    return out


# --------------------------------------------------------------------------- #
# Allocation policies (the outer loop)
# --------------------------------------------------------------------------- #

POLICIES = ("uniform", "error_only", "lprog_only", "visits_only", "value",
            "value_satiety", "reducible_only", "value_red", "value_red_satiety", "oracle")


def allocate(policy, *, errors, lprog, visits, tree_channel, n_channels, lp_floor,
             reducible=None, red_floor=0.0, satiety=None, beta_sat=1.0, eps=0.02,
             shared_channels=None):
    """Per-channel budget shares. `eps` is an identical uniform floor for every policy, so
    the arms differ only in their signal (and so that a policy is never handed an all-zero
    signal). Returns (weights, drive) with `drive` the per-channel signal actually used.

    `value_satiety` is the one non-E3 rung: LP is measured against the noise-calibrated floor
    so that a mastered or irreducible channel goes NEGATIVE rather than merely to zero, and a
    depletable `satiety` state accumulates with spending so that a channel which has stopped
    paying is actively pushed out rather than left to be re-picked by estimator noise. This is
    `ideas/adaptive_core_and_hierarchy_climb.md` §10/§12's one-line change, and its prediction
    is specifically that E3's 29% budget leak into the irreducible regions falls.
    """
    e = np.asarray([errors[c] for c in range(n_channels)], dtype=np.float64)
    lp = np.asarray([lprog[c] for c in range(n_channels)], dtype=np.float64)
    vis = np.asarray(visits, dtype=np.float64)

    if policy == "uniform":
        drive = np.ones(n_channels)
    elif policy == "error_only":
        drive = np.maximum(e, 0.0)
    elif policy == "lprog_only":
        drive = np.maximum(lp, 0.0)
    elif policy == "visits_only":
        drive = vis.copy()
    elif policy == "value":
        drive = np.maximum(lp, 0.0) * vis
    elif policy == "value_satiety":
        # The satiety state raises the exclusion threshold MULTIPLICATIVELY on the
        # noise-calibrated floor, so the knob is scale-free (LP here is a difference of
        # normalised MSEs, ~1e-2; a subtractive `beta*sat` with sat ~ O(1) would zero every
        # channel). Read the drive as signed: `lp - floor*(1 + beta*sat)` goes NEGATIVE on a
        # channel that has stopped paying and keeps being fed, where plain `relu(lp)` merely
        # goes to zero and is re-picked by estimator noise. Allocation renormalises over the
        # positive part, so the negative lobe expresses as exact exclusion.
        sat = np.zeros(n_channels) if satiety is None else np.asarray(satiety)
        drive = np.maximum(lp - lp_floor * (1.0 + beta_sat * sat), 0.0) * vis
        if drive.sum() <= 0:
            # Everything satiated. Fall back to the UNSATIATED value drive, not to uniform:
            # uniform would hand 2/5 of the budget to the irreducible channels, i.e. the
            # satiating rung would lose to plain `value` for a reason that has nothing to do
            # with satiety. Falling back to `value` keeps the rung nested inside it.
            drive = np.maximum(lp, 0.0) * vis
    elif policy in ("reducible_only", "value_red", "value_red_satiety"):
        red = np.asarray([reducible[c] for c in range(n_channels)], dtype=np.float64)
        if policy == "reducible_only":
            drive = red.copy()
        elif policy == "value_red":
            drive = red * vis
        else:
            # Satiety on a drive that is not inverted, so §10's prediction is finally
            # testable. The exclusion threshold is calibrated off the IRREDUCIBLE channels'
            # own apparent reducibility -- they cannot be reduced, so whatever they show is
            # what "no real headroom" looks like on this instrument -- and it HARDENS with
            # accumulated spending, so a channel that has stopped paying is pushed out rather
            # than left to be re-picked by estimator noise.
            sat = np.zeros(n_channels) if satiety is None else np.asarray(satiety)
            drive = np.maximum(red - red_floor * (1.0 + beta_sat * sat), 0.0) * vis
            if drive.sum() <= 0:
                drive = red * vis                    # fall back to the unsatiated drive
    elif policy == "oracle":
        drive = np.zeros(n_channels)
        drive[tree_channel] = 1.0
    elif policy == "oracle_dup":
        # A BIT-FOR-BIT DUPLICATE of `oracle`, run as a second arm so that `oracle_dup - oracle`
        # measures this instrument's noise floor AT THE OPERATING POINT rather than importing a
        # floor measured somewhere else. `partial_hetero` §2 got the same number for free (its
        # `oracle_shared` degenerates to `oracle` at zero sharing, giving -0.0033 +/- 0.0039);
        # the metering sweep needs it at every budget, because the floor is not budget-invariant.
        # Deliberately NOT in POLICIES, so no default run's arm set changes.
        drive = np.zeros(n_channels)
        drive[tree_channel] = 1.0
    elif policy == "oracle_shared":
        # The TRANSFER-AWARE oracle. `oracle` is privileged about RELEVANCE -- it is handed the
        # channel labels and puts everything on the only channel that can move `d*`. That is the
        # optimal allocation only when off-tree data is worth nothing to the LEARNER, which is
        # true by construction in the published geometry and false under partial sharing. This
        # rung spreads evenly over the tree and every channel that shares rule tables with it, so
        # `oracle_shared - oracle` measures exactly the gap between stipulated relevance and
        # actual data value. With no shared channels it is `oracle` verbatim.
        drive = np.zeros(n_channels)
        drive[tree_channel] = 1.0
        for c in (shared_channels or ()):
            drive[int(c)] = 1.0
    else:
        raise ValueError(policy)

    total = drive.sum()
    w = np.full(n_channels, 1.0 / n_channels) if total <= 0 else drive / total
    w = (1.0 - eps * n_channels) * w + eps
    return w / w.sum(), drive


# --------------------------------------------------------------------------- #
# The trainable belief: Stage 5's internalization ladder, on the channel task
# --------------------------------------------------------------------------- #

def collect_grounded_moves(layout, tb, generator, x, roots, *, render, gen, device):
    """The grounded (controller-independent) imitation target `k* = argmin_k d*(regen(x,k))`.

    Only the TREE candidates need the DP. A distractor move cannot change the feature
    assignment of the tree slice, so `d*` after it is EXACTLY `d*` before it (P1) -- filled in
    analytically rather than measured, which is both exact and 1.75x cheaper at L=5.

    Note the teacher is NOT trivially "always a tree block": when no tree move lowers `d*`,
    every distractor move ties with the do-nothing cost and the lowest-index tie-break can
    land off-tree. A wasted move is a wasted move either way, and leaving that in keeps the
    evaluative grader from being a relabelled channel oracle.
    """
    import torch
    from rhm.rhm_channels import dp_cost

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x.cpu().numpy(), roots_np)
    cand = np.repeat(d_cur[None, :], tb["n_blocks"], axis=0).astype(np.int64)
    with torch.no_grad():
        for k in tb["tree_blocks"].tolist():
            kk = torch.full((x.shape[0],), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            cand[k] = dp_cost(layout, xk.cpu().numpy(), roots_np)
    kstar = cand.argmin(axis=0)
    return (torch.from_numpy(kstar).to(device), float(d_cur.mean()),
            float(cand.min(axis=0).mean()))


def belief_update(mode, controller, block_fm, value, generator, layout, tb, *, train_leaves,
                  train_roots, gstates, groots, gkstar, n_steps, batch_size, lr, lam_fm,
                  lam_plan, tau, n_corrupt, edit_budget, render, gen, device, p_full=0.5,
                  seed=0, pstates=None, proots=None):
    """One chunk of belief training. Nested exactly as Stage 5's ladder, so the contrasts
    isolate the same things they isolated there:

      frozen      root-CE only.
      dense       + lam_fm * asymmetric FM local loss -- ENDOGENOUS "be predictable" pressure
                  (the FM predicts the controller's OWN latents; no task grounding). This is
                  `fm_cotrain`, the rung that CAPPED on a fixed task (belief PR flat 6.7->6.8,
                  transferable plannability DOWN 0.357->0.304).
      evaluative  + lam_plan * CE(softmax_k value((z + FM(z,k)).mean)/tau, k*) against the
                  exact-DP best move. This is `planner`, the rung that EXPANDED on a fixed
                  task (PR 6.7->17.0, top1 0.357->0.521) and then consolidated.

    `dense subset evaluative`, so `evaluative - dense` isolates GROUNDING and nothing else --
    which is the whole content of the 2x2's missing cell.

    `pstates`/`proots` optionally give the PLAN term its own state pool, defaulting to
    `gstates`/`groots` (bit-identical to every existing caller). This exists because an
    endogenous teacher, unlike the DP, has no opinion on some states -- a Monte-Carlo terminal
    return that is constant across all 14 candidates carries zero information, and training a
    hard CE on a coin flip there is training on noise. Filtering those out of the PLAN pool
    while leaving the DENSE pool untouched keeps the dense term's state distribution matched
    across arms, so `endo - evaluative` still isolates the teacher and nothing else.
    """
    import torch
    import torch.nn.functional as F

    controller.train()
    params = list(controller.parameters())
    if mode in ("dense", "evaluative"):
        block_fm.train()
        params += list(block_fm.parameters())
    if mode == "evaluative":
        value.train()
        params += list(value.parameters())
    elif mode not in ("frozen", "dense"):
        raise ValueError(mode)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n_blocks = tb["n_blocks"]
    n_pool = train_leaves.shape[0]
    arange_r = torch.arange(n_blocks, device=device)
    last = {}
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        n_rev = n_blocks if torch.rand(()).item() < p_full else int(
            torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        obs = _reveal(leaves, order[:, :n_rev], tb["s"])
        root_logits = controller.root_logits(controller.block_state(obs).mean(dim=1))
        loss = F.cross_entropy(root_logits, roots)
        last["root_acc"] = float((root_logits.argmax(-1) == roots).float().mean())

        if mode in ("dense", "evaluative"):
            sub = rng.integers(0, gstates.shape[0], size=batch_size)
            x = gstates[torch.from_numpy(sub).to(device)]
            k = torch.randint(0, n_blocks, (batch_size,), device=device)
            with torch.no_grad():
                x2 = regenerate_block(generator, x, k, tb, render=render, gen=gen)
            z = controller.block_state(x)
            z2 = controller.block_state(x2)
            delta = z2 - z
            # a2a-faithful ASYMMETRIC local loss: the FM learns the REAL detached dynamics,
            # the controller is PULLED toward the FM's detached prediction. Non-collapsing by
            # construction; the root-CE anchor plus the PR/depth probes catch it if it isn't.
            loss = loss + lam_fm * (F.mse_loss(block_fm(z.detach(), k), delta.detach())
                                    + F.mse_loss(delta, block_fm(z, k).detach()))

        if mode == "evaluative":
            px = gstates if pstates is None else pstates
            pr = groots if proots is None else proots
            sub = rng.integers(0, px.shape[0], size=batch_size)
            si = torch.from_numpy(sub).to(device)
            zp = controller.block_state(px[si])
            rp = pr[si]
            logits = torch.stack([
                value((zp + block_fm(zp, arange_r[j].expand(batch_size))).mean(dim=1), rp)
                for j in range(n_blocks)], dim=1) / tau
            loss = loss + lam_plan * F.cross_entropy(logits, gkstar[si])
            last["plan_acc"] = float((logits.argmax(1) == gkstar[si]).float().mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    controller.eval()
    if block_fm is not None:
        block_fm.eval()
    if value is not None:
        value.eval()
    return last


def _reveal(leaves, blocks, s):
    """Masked observation revealing only `blocks` (mask token -1, as the controller expects)."""
    import torch
    batch, length = leaves.shape
    obs = torch.full((batch, length), -1, dtype=leaves.dtype, device=leaves.device)
    if blocks.shape[1] == 0:
        return obs
    pos = (blocks[:, :, None] * s + torch.arange(s, device=leaves.device)).reshape(batch, -1)
    return obs.scatter(1, pos, leaves.gather(1, pos))


def root_ce(controller, leaves, roots, device, chunk=2048):
    """Held-out root cross-entropy under FULL observation -- the belief's own grader."""
    import torch
    import torch.nn.functional as F
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, leaves.shape[0], chunk):
            x = leaves[i:i + chunk].to(device)
            r = roots[i:i + chunk].to(device)
            logit = controller.root_logits(controller.block_state(x).mean(dim=1))
            tot += float(F.cross_entropy(logit, r, reduction="sum"))
            n += x.shape[0]
    return tot / max(n, 1)
