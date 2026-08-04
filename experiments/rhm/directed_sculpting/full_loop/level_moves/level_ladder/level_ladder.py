"""The level-indexed ALLOCATION ladder: does an endogenous drive climb when the world
makes deep moves necessary?

THE QUESTION. `full_loop/README.md` open item 2 -- *"a level-ordered allocation space, so
§12's satiety-gated 'recruit ell+1' has an upward direction to recruit in. Currently it
recruits sideways into a distractor."* `level_moves/` built the ACTION space (a move that
commits to one level-ell feature) and the READOUT (per-(channel, level) value sensitivity),
and deliberately stopped there. This is the allocation half.

WHY THE OBVIOUS DESIGN DOES NOT WORK, AND WHAT REPLACES IT. The obvious cut is "give the
budget a level index and see whether the value climbs it." `level_moves` §10 already
measured the answer: the value's preference SLOPE against move level does not resolve
(+0.060 +- 0.108, t = +0.97). What DOES resolve (§11) is that the value's abstraction
premium tracks *how deep the error is* -- L2 slope -0.033, t = -6.81, against the oracle's
-0.025. So the value supplies **error-depth matching, not a depth ordering**: it can say
"a level-ell commitment is worth it here", never "go higher".

So do not ask the drive to supply "up". Make the WORLD supply it, and ask whether the
allocator follows. That is `climb.py` §6's logic transplanted one axis over:

    climb:  "deep levels are more invariant -> a learner climbs" failed because surface
            re-fit repaired 91% of each event, so climbing was never NECESSARY. Starving
            samples/event made it necessary and the depth advantage migrated (t = -5.44).
    here:   "deep moves explain more -> the value prefers them" does not resolve because
            shallow moves were always SUFFICIENT (`corrupt_tree` leaves 29.4% of tree
            blocks off-grammar, so every error is repairable one block at a time).
            `corrupt_tree_hier` puts the error at a level, and a schedule moves it deeper.

At damage depth d, a move at level < d provably cannot reach the error -- not a soft
preference but an exact fact from `possible_sets`. The allocator is never told d. The
question is whether an endogenous tap climbs on schedule, and whether that beats a
level-blind drive on the identical world.

WHY THE FM HAD TO CHANGE (and why it is a small change). `level_moves` recorded the span
FM as the standing blocker: `forecast_visits` rolls `_fm_chunked(fm, z, kk)` indexed by
block, so it cannot index a level-ell node. But `BlockLatentFM`'s TARGET is already the
whole (n_blocks, D) latent delta -- only the action CONDITIONING is per-block, written as
one scattered marker. `SpanLatentFM` marks every block in the move's span and adds a level
embedding, which at level 1 is exactly zero by construction. Gate C4 asserts the two are
bit-identical on the flat move set, so the published FM is the max_level=1 special case
rather than an analogue.

THE ARMS. Six, differing only in the allocation rule; every arm forks from the same warm
FM on a bit-identical world sequence, as the published ladder does.

  uniform        flat over cells -- the floor
  oracle_level   privileged: drive = the exact DP's best Delta-d* per cell. The ceiling,
                 and the thing to check FIRST: if it does not beat uniform there is no
                 prize on this axis and nothing else here is readable
  channel_only   the published channel drive, spread UNIFORMLY over levels within a
                 channel. THE load-bearing control -- if it ties the cell-indexed arms,
                 the level axis is decoration and we have learned that cheaply
  visits_level   endogenous: cell-indexed forecast visitation (the ladder's best tap)
  value_level    endogenous: cell-indexed |Delta-V| -- the tap `level_moves` §11's finding
                 is actually about, so this is the arm that cashes it
  satiety_level  §12 proper: `visits_level` with a depletable per-cell satiety exclusion,
                 so a cell that has stopped paying is pushed out rather than re-picked

THE CONTROLS THAT MAKE IT FALSIFIABLE.
  * `channel_only` above -- does the level index buy anything at all?
  * `--damage-schedule` held STATIC at one depth. If allocation climbs there too, we are
    watching deep moves become relatively more useful as the FM improves (a training-time
    artifact), not necessity tracking. This is the discriminator and it is not optional.
  * The reducibility tap is deliberately NOT ported. `metering_sweep` priced it at ~1200x
    worse cost-effectiveness than `visits_only` (14080 sequences/round for 4% recovery),
    and `value_red` ties `visits_only` on this geometry by construction. Porting it would
    be GPU spent to reproduce a known tie.

WHAT IS HELD FIXED. Everything else is the published ladder: same drift primitive and
calibration, same monitor:collect ratio, same warm-start protocol, same frozen probe
discipline. The value is trained ONCE, on a MIXTURE over damage depths, and frozen -- so
the arms differ in the allocation rule and not in how sighted their grader is. That is
deliberate: it isolates "does the allocator follow the world" from "does the value
re-learn", which are two questions and only the first one is this cut.

SCOPE. No claim here about whether the value SUPPLIES an upward direction -- `level_moves`
§10 says it does not, and this cut is built around that rather than against it. What is
tested is whether error depth, made necessary by the world, is followed by an endogenous
allocator over a level-ordered budget.

Run from experiments/:
  python3 -c "from rhm.directed_sculpting.full_loop.level_moves.level_ladder.level_ladder \
      import selfcheck; selfcheck()"                                    # C4, no GPU
  modal run rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::gate
  modal run rhm/directed_sculpting/full_loop/level_moves/level_ladder/level_ladder.py::level_ladder --quick
"""

import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import dp_cost, make_layout, sample_pool
from rhm.rhm_repair_cost import Meter
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    _block_state_chunked, advance_drift, build_block_tables, make_spec, prewarm_drift,
    refresh_w_blk, train_generator_channels)
from rhm.directed_sculpting.full_loop.ladder import PricedMeter, calibrate_drift, reset_drift
from rhm.directed_sculpting.full_loop.level_moves.level_moves import (
    apply_moves, build_move_set, certify_damage, collect_value_buffer_moves, flat_move_set,
    make_damage, moves_table, regenerate_node, sample_states_damaged, tb_numpy)


app = modal.App("rhm-ds-level-ladder", image=image)


# --------------------------------------------------------------------------- #
# Cells: the (channel, level) index the budget is spent over
# --------------------------------------------------------------------------- #

def build_cells(ms, tb):
    """The allocation index. One cell per (channel, level) present in the move set.

    The published ladder allocates over CHANNELS and then picks a block within the chosen
    channel uniformly (`ladder.py`: `k_col[sel] = blks[torch.randint(...)]`), so "level" has
    no representation anywhere in the budget. This is the index that gives it one.

    Returns cell ids in a canonical (channel, level) order plus the move->cell map and the
    per-cell move lists. At `max_level=1` there is exactly one cell per channel, so the cell
    index IS the channel index and this ladder reduces to the published one.
    """
    import torch

    keys, moves_of = [], {}
    for mi, mv in enumerate(ms["moves"]):
        assert not mv.get("lazy"), "lazy twins are a probe-only control, never an action"
        key = (mv["channel"], mv["level"])
        if key not in moves_of:
            keys.append(key)
            moves_of[key] = []
        moves_of[key].append(mi)
    keys.sort()
    cell_of = np.zeros(ms["n_moves"], dtype=np.int64)
    for ci, key in enumerate(keys):
        for mi in moves_of[key]:
            cell_of[mi] = ci
    names = tb["channel_names"]
    return {
        "keys": keys,
        "n_cells": len(keys),
        "names": [f"{names[c]}|L{ell}" for c, ell in keys],
        "channel_of_cell": np.array([c for c, _ in keys], dtype=np.int64),
        "level_of_cell": np.array([ell for _, ell in keys], dtype=np.int64),
        "moves_of_cell": [moves_of[k] for k in keys],
        "is_noise": [ms["moves"][moves_of[k][0]]["kind"] == "noise" for k in keys],
        "tree_channel": tb["tree_channel"],
        "cell_of_move": cell_of,
        "cell_of_move_t": torch.from_numpy(cell_of).to(ms["channel_of"].device),
    }


def move_index_tensors(ms, tb, device):
    """`(span_mask, level_of)` for the span FM: which blocks a move rewrites, and at what level.

    `span_mask` is (n_moves, n_blocks) float -- 1.0 on every block inside the move's span. At
    level 1 each row is one-hot, which is what makes C4's reduction exact.
    """
    import torch

    n_blocks = tb["n_blocks"]
    span_mask = torch.zeros(ms["n_moves"], n_blocks, device=device)
    for mi, mv in enumerate(ms["moves"]):
        span_mask[mi, mv["blk0"]:mv["blk0"] + mv["span"]] = 1.0
    level_of = torch.tensor([mv["level"] for mv in ms["moves"]], device=device,
                            dtype=torch.long)
    return span_mask, level_of


# --------------------------------------------------------------------------- #
# The span FM: `BlockLatentFM` with the action conditioning generalised to a span
# --------------------------------------------------------------------------- #

def build_span_fm():
    """Cerebellar FM over per-block latents, conditioned on a level-indexed MOVE.

    `rhm_sculpt_latent._build_block_fm` writes one scattered `action_embedding[k]` marker at
    the acted block and predicts the whole (n_blocks, D) delta. The only thing that is
    per-block is the CONDITIONING, so generalising to a span is:

      marker[b] = span_mask[move, b] * (action_embedding[b] + level_embedding[level(move)])

    Two design points, both load-bearing.

    * `action_embedding` stays indexed by BLOCK, not by move. A move is identified by which
      blocks it touches plus its level, so the FM cannot memorise a move id -- and the
      level-1 rows are the same parameters the published FM uses.
    * `level_embedding` is subtracted at level 1 (`- weight[1]`), so a level-1 move's marker
      is EXACTLY `action_embedding[k]` at the acted block and zero elsewhere, whatever the
      embedding learns. That is what makes C4 an identity rather than an approximation, and
      it means the published FM is this FM's `max_level=1` restriction.

    Weights init to zero on `level_embedding`, so at step 0 every level is conditioned
    identically and the level structure has to be LEARNED from the transitions rather than
    handed over at initialisation.
    """
    import torch
    import torch.nn as nn

    class SpanLatentFM(nn.Module):
        def __init__(self, state_dim, n_blocks, max_level, span_mask, level_of,
                     n_head, n_layer):
            super().__init__()
            self.action_embedding = nn.Embedding(n_blocks, state_dim)
            self.block_position = nn.Embedding(n_blocks, state_dim)
            self.level_embedding = nn.Embedding(max_level + 1, state_dim)
            nn.init.zeros_(self.level_embedding.weight)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.norm = nn.LayerNorm(state_dim)
            self.head = nn.Linear(state_dim, state_dim)
            self.register_buffer("positions", torch.arange(n_blocks), persistent=False)
            self.register_buffer("span_mask", span_mask, persistent=False)
            self.register_buffer("level_of", level_of, persistent=False)

        def forward(self, z_block, mv):
            mask = self.span_mask[mv]                                  # (B, n_blocks)
            act = self.action_embedding(self.positions)[None]          # (1, n_blocks, D)
            lvl = (self.level_embedding(self.level_of[mv])
                   - self.level_embedding.weight[1])[:, None, :]       # (B, 1, D), 0 at L1
            marker = mask[..., None] * (act + lvl)
            hidden = z_block + self.block_position(self.positions)[None] + marker
            return self.head(self.norm(self.encoder(hidden)))

    return SpanLatentFM


def _fm_chunked_mv(fm, z, mv, chunk=16384):
    import torch
    if z.shape[0] <= chunk:
        return fm(z, mv)
    return torch.cat([fm(z[i:i + chunk], mv[i:i + chunk])
                      for i in range(0, z.shape[0], chunk)], dim=0)


def transition_targets_moves(controller, generator, x, mv, ms, tb, *, render="mixture",
                             gen=None):
    """(z, Delta-z) for MOVE `mv` from state `x`. The move-indexed `transition_targets`."""
    import torch
    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        x2 = apply_moves(generator, x, mv, ms, tb, render=render, gen=gen)
        z2 = _block_state_chunked(controller, x2)
    return z, z2 - z


def fm_error_moves(fm, z, mv, target, ms, acted_only=False):
    """Normalised MSE, optionally restricted to the blocks the move actually rewrote.

    `acted_only` is the headline in the published ladder because the all-blocks number is
    diluted by untouched blocks. Here "acted" is the move's SPAN, so a level-4 move's acted
    set is the whole tree slice -- which is correct, and worth remembering when comparing
    acted errors across levels (they are averages over different numbers of blocks).
    """
    import torch
    with torch.no_grad():
        pred = _fm_chunked_mv(fm, z, mv)
        if acted_only:
            w = fm.span_mask[mv][..., None]
            num = float(((pred - target) ** 2 * w).sum())
            den = float((target ** 2 * w).sum())
        else:
            num = float(((pred - target) ** 2).sum())
            den = float((target ** 2).sum())
        raw = num / (target.shape[0] * target.shape[1] * target.shape[2])
    return num / max(den, 1e-12), raw


# --------------------------------------------------------------------------- #
# The `p` tap, per cell: roll the FM under the greedy planner over MOVES
# --------------------------------------------------------------------------- #

def forecast_visits_cells(fm, value, controller, tb, ms, cells, x0, roots, *, budget, device):
    """`channel_env.forecast_visits` over the level-indexed move set.

    Returns `(visits, dvalue)` per CELL: the share of forecast moves that land in each
    (channel, level), and the mean |Delta-V| the value assigns to acting there. Everything
    is a rollout of the FM's own predictions -- nothing is materialised -- so this is the
    same metered tap the published ladder buys, priced identically.
    """
    import torch
    with torch.no_grad():
        z = _block_state_chunked(controller, x0)
        batch = z.shape[0]
        n_cells = cells["n_cells"]
        c_of = cells["cell_of_move_t"]
        counts = torch.zeros(n_cells, device=device)
        dv_sum = torch.zeros(n_cells, device=device)
        dv_cnt = torch.zeros(n_cells, device=device)
        for _ in range(budget):
            v_now = value(z.mean(dim=1), roots)
            scores = []
            for mi in range(ms["n_moves"]):
                mm = torch.full((batch,), mi, device=device, dtype=torch.long)
                scores.append(value((z + _fm_chunked_mv(fm, z, mm)).mean(dim=1), roots))
            scores = torch.stack(scores, dim=1)                        # (B, n_moves)
            dv = (scores - v_now[:, None]).abs().mean(dim=0)           # (n_moves,)
            dv_sum.index_add_(0, c_of, dv)
            dv_cnt.index_add_(0, c_of, torch.ones_like(dv))
            best = scores.argmax(dim=1)
            counts.index_add_(0, c_of[best], torch.ones(batch, device=device))
            z = z + _fm_chunked_mv(fm, z, best)
        raw = counts / counts.sum().clamp_min(1)
        # PER-NODE, and this is not cosmetic. `tree|L1` has 8 nodes competing for argmax and
        # `tree|L4` has one, so a raw count share hands the shallow cells a factor of 8 that
        # has nothing to do with where the plan wants to go. `level_moves` §4 settled the
        # same point for `top1_share` ("the fair comparison since level 1 has 8 nodes
        # competing and level 4 has one") and this is that correction on the allocation tap.
        # The raw share is returned too: `channel_only` must read it, because aggregating to
        # a channel is exactly a sum over that channel's nodes.
        per_node = counts / dv_cnt.clamp_min(1)
        visits = (per_node / per_node.sum().clamp_min(1e-12)).cpu().numpy()
        dvalue = (dv_sum / dv_cnt.clamp_min(1)).cpu().numpy()
    return visits, dvalue, raw.cpu().numpy()


# --------------------------------------------------------------------------- #
# The matched-span premium tap: the one endogenous drive span cannot dominate
# --------------------------------------------------------------------------- #

def random_node(generator, x, move, ms, tb, *, render="mixture", gen=None):
    """Rewrite `move`'s span with UNIFORMLY RANDOM legal level-1 features.

    The span-matched partner for every move at every level. `regenerate_node` writes the
    generator's best legal derivation into the span; this writes an arbitrary legal one into
    the SAME positions. So the pair differs in exactly one respect -- whether the content is
    informed -- and the `s**(ell-1)`-token displacement of `z.mean(dim=1)` is identical
    inside the pair.

    ON-GRAMMAR ON PURPOSE. An off-grammar twin would be trivially worse and the premium would
    collapse into a grammaticality readout; `level_moves` §10 measured that hazard directly
    (only 4.4% of its premium survived where legality bought Delta-`d*` = 0). Every feature in
    [0, v) has `m_blk` legal synonyms, so a uniform draw over features is legal by
    construction, and the synonym is drawn from the block's CURRENT drifted mixture exactly as
    the informed move draws it.

    Relation to `level_moves`' lazy twin: that twin is *informed but uncommitted* (per-block
    argmax, no DP) and exists only at level >= 2, because at level 1 it IS the committed move.
    That makes it the right control for isolating COMMITMENT, and the wrong one for an
    allocation drive, which needs a signal defined identically at every level of the index it
    allocates over. This twin is uninformed at every level, so the premium means the same
    thing in every cell: how much is knowing what belongs here worth, at this scale.
    """
    import torch

    s, batch = tb["s"], x.shape[0]
    blk0, span = move["blk0"], move["span"]
    pos = (torch.arange(blk0, blk0 + span, device=x.device)[:, None] * s
           + torch.arange(s, device=x.device)[None, :]).reshape(-1)[None, :].expand(batch, -1)
    if move["kind"] == "noise":
        tok = torch.multinomial(tb["noise_p"], batch * span * s, replacement=True,
                                generator=gen).view(batch, span * s)
        return x.clone().scatter_(1, pos, tok)

    blocks = torch.arange(blk0, blk0 + span, device=x.device)[None, :].expand(batch, -1)
    feats = torch.randint(0, tb["v"], (batch, span), device=x.device, generator=gen)
    if render == "canon":
        choice = torch.zeros(batch, span, dtype=torch.long, device=x.device)
    else:
        w = tb["w_blk"][blocks.reshape(-1), feats.reshape(-1)]
        choice = torch.multinomial(w, 1, generator=gen).view(batch, span)
    tup = tb["syn_blk"][blocks.reshape(-1), feats.reshape(-1), choice.reshape(-1)]
    return x.clone().scatter_(1, pos, tup.view(batch, span * s))


def premium_cells(value, controller, generator, tb, ms, cells, x0, roots, *, render, gen,
                  device):
    """The matched-span premium per cell: V(informed move) - V(span-matched random rewrite).

    WHY THIS TAP EXISTS. Both taps the published ladder supplies are dominated by span once
    they are given a level index, and this was measured rather than assumed (see the node's
    writeup): raw |Delta-V| pins the allocation at mean level ~2.97 and per-node forecast
    visitation at ~3.15, both flat, while the exact oracle moves 2.15 -> 2.68 across the
    damage schedule. A level-ell move rewrites `s**(ell-1)` blocks and displaces the pooled
    latent more whether or not it buys anything -- `level_moves` §9's span null, appearing as
    an allocation signal rather than as a readout. Differencing against a twin over the
    IDENTICAL positions cancels it inside the pair by construction, which is the same
    resolution §10 reached for the readout: a matched control does what a matched regression
    cannot when the components are not independent.

    Endogenous: no channel labels, no rule tables, no DP, nothing from the ground truth. It
    is MATERIALISED rather than rolled through the FM (the FM has no twin actions to model),
    so it must be metered -- `metering_sweep` is where a materialised tap gets priced, and
    the caller charges `2 * n_moves` sequence encodings per probe state.
    """
    import torch

    n_cells = cells["n_cells"]
    prem = np.zeros(n_cells)
    cnt = np.zeros(n_cells)
    with torch.no_grad():
        v_now = value(_block_state_chunked(controller, x0).mean(dim=1), roots)
        for mi, mv in enumerate(ms["moves"]):
            xk = regenerate_node(generator, x0, mv, ms, tb, render=render, gen=gen)
            xr = random_node(generator, x0, mv, ms, tb, render=render, gen=gen)
            dv = (value(_block_state_chunked(controller, xk).mean(dim=1), roots)
                  - value(_block_state_chunked(controller, xr).mean(dim=1), roots))
            ci = int(cells["cell_of_move"][mi])
            prem[ci] += float(dv.mean())
            cnt[ci] += 1.0
    return prem / np.maximum(cnt, 1.0)


# --------------------------------------------------------------------------- #
# The privileged answer: what each cell is actually worth, exactly
# --------------------------------------------------------------------------- #

def cell_dstar_gain(layout, generator, x0, roots, ms, cells, tb, *, render, gen):
    """Exact DP Delta-`d*` per cell: `mean` for a random move there, `best` for the best one.

    This is `channel_env.ground_truth_relevance` extended to levels, and it is doing two
    jobs. It is `oracle_level`'s drive (privileged, unmetered, like the published `oracle`),
    and it is the GATE: under hierarchical damage at depth d, the best gain in cells with
    level < d must collapse toward 0 while cells at level >= d stay positive. If that
    ordering is absent the world never made abstraction necessary and no allocator can be
    scored for climbing.
    """
    import torch

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x0.cpu().numpy(), roots_np)
    gains = np.zeros((ms["n_moves"], x0.shape[0]))
    with torch.no_grad():
        for mi, mv in enumerate(ms["moves"]):
            xk = regenerate_node(generator, x0, mv, ms, tb, render=render, gen=gen)
            gains[mi] = d_cur - dp_cost(layout, xk.cpu().numpy(), roots_np)
    out = {}
    for ci, mis in enumerate(cells["moves_of_cell"]):
        sub = gains[mis]
        out[ci] = {"mean": float(sub.mean()), "best": float(sub.max(axis=0).mean())}
    return out, float(d_cur.mean())


# --------------------------------------------------------------------------- #
# Ballistic control over moves
# --------------------------------------------------------------------------- #

def _imagine_plan_moves(controller, fm, value, ms, x0, roots, *, horizon, beam_width, device):
    """`channel_env._imagine_plan` with the candidate set being MOVES rather than blocks."""
    import torch

    n_moves = ms["n_moves"]
    with torch.no_grad():
        z = _block_state_chunked(controller, x0)
        batch = z.shape[0]
        n_blocks, dim = z.shape[1], z.shape[2]
        beams_z = z[:, None]
        moves = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        for _ in range(horizon):
            flat = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for mi in range(n_moves):
                mm = torch.full((batch * width,), mi, device=device, dtype=torch.long)
                scores.append(value((flat + _fm_chunked_mv(fm, flat, mm)).mean(dim=1), roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * n_moves)
            keep = min(beam_width, width * n_moves)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, n_moves, rounding_mode="floor")
            move = top % n_moves
            parent_z = beams_z.gather(
                1, parent[:, :, None, None].expand(-1, -1, n_blocks, dim))
            fz = parent_z.reshape(batch * keep, n_blocks, dim)
            beams_z = (fz + _fm_chunked_mv(fm, fz, move.reshape(-1))).reshape(
                batch, keep, n_blocks, dim)
            prev = (moves.gather(1, parent[:, :, None].expand(-1, -1, moves.shape[2]))
                    if moves.shape[2] else moves.new_zeros(batch, keep, 0))
            moves = torch.cat([prev, move[:, :, None]], dim=2)
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, dim),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
    return moves[torch.arange(batch, device=device), best]


def open_loop_beam_moves(controller, generator, fm, value, ms, tb, layout, x0, roots, *,
                         budget, beam_width, render, gen, device, chunk_len=3):
    """BALLISTIC control over the level-indexed action space -- the sighted grader.

    Identical protocol to `channel_env.open_loop_beam` (plan `chunk_len` steps in imagination,
    commit, re-ground) with `regenerate_block` replaced by `apply_moves`. This is what makes
    hierarchical damage repairable at all: a block-only planner cannot execute a level-ell
    commitment, so under `--damage-schedule` deeper than 1 its ceiling is set by the damage
    rather than by the FM.
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
            plan = _imagine_plan_moves(controller, fm, value, ms, x, roots, horizon=h,
                                       beam_width=beam_width, device=device)
            for t in range(h):
                x = apply_moves(generator, x, plan[:, t], ms, tb, render=render, gen=gen)
        succ = possible_set_success(layout, x.cpu().numpy(), roots.cpu().numpy())
    return float(succ.mean())


def per_cell_error(fm, controller, generator, tb, ms, cells, x_probe, *, render, gen, seed=0):
    """Frozen-probe FM error per cell. Same states every round; only world and model move."""
    import torch

    out = {}
    g = torch.Generator(device=x_probe.device).manual_seed(seed)
    for ci, mis in enumerate(cells["moves_of_cell"]):
        pick = torch.tensor(mis, device=x_probe.device)
        mv = pick[torch.randint(0, len(mis), (x_probe.shape[0],), device=x_probe.device,
                                generator=g)]
        z, target = transition_targets_moves(controller, generator, x_probe, mv, ms, tb,
                                             render=render, gen=gen)
        nmse, raw = fm_error_moves(fm, z, mv, target, ms)
        out[ci] = {"nmse": nmse, "raw_mse": raw,
                   "nmse_acted": fm_error_moves(fm, z, mv, target, ms, acted_only=True)[0]}
    return out


# --------------------------------------------------------------------------- #
# Allocation over cells
# --------------------------------------------------------------------------- #

CELL_POLICIES = ("uniform", "oracle_channel", "oracle_level", "channel_only", "visits_level",
                 "value_level", "premium_level", "satiety_level")

# Which taps each drive actually buys. `fc` is the FM rollout (forecast visits + |ΔV|);
# `prem` is the MATERIALISED matched-span premium, which is far more expensive per probe
# state and is charged as such. The ladder still computes both for every arm so the monitor
# charge is identical and only the allocation rule differs -- `metering_sweep` is where that
# subsidy is priced, and this table is what a priced re-run would read.
TAP_READS_CELL = {
    "uniform": (), "oracle_channel": (), "oracle_level": (),
    "channel_only": ("fc",), "visits_level": ("fc",), "value_level": ("fc",),
    "premium_level": ("prem",), "satiety_level": ("prem",),
}


def allocate_cells(policy, *, cells, visits, dvalue, oracle_gain, premium=None,
                   visits_raw=None, satiety=None, beta_sat=1.0, eps=0.01):
    """Per-cell budget shares. `eps` is an identical uniform floor for every policy, so the
    arms differ only in their signal.

    `visits` is the PER-NODE forecast rate and `visits_raw` the raw count share.
    `channel_only` reads the raw one, because aggregating to a channel IS a sum over that
    channel's nodes -- that is what the published ladder computes, and using the per-node
    tap there would make the control a different estimator rather than the same one at a
    coarser index.

    `channel_only` is the control that decides whether this whole axis is worth anything: it
    aggregates the SAME endogenous tap to the channel level, then spreads uniformly over the
    levels within a channel -- exactly what the published ladder does when it picks a block
    uniformly inside the chosen channel. If it ties `visits_level`, the level index is
    decoration.

    `value_level` reads RAW |Delta-V| and is expected to be span-dominated: a level-ell move
    rewrites `s**(ell-1)` blocks and so displaces `z.mean(dim=1)` more whether or not it buys
    anything (`level_moves` §9 measured the null rising 2.79x L1->L4 in channels where level
    means nothing at all). It is kept as an arm precisely because that confound has never
    been seen as an ALLOCATION signal, only as a readout -- if it sits high and flat while
    `oracle_level` climbs, that is the span null cashed out in the budget.

    `satiety_level` is §12's rule on an index that finally has an "up": the exclusion
    threshold hardens with accumulated spending, so a cell that has stopped paying is pushed
    out and the mass renormalises onto whatever is left -- which, once the shallow cells in
    a channel are satiated, is the deeper ones. Note this still does not make the DRIVE
    prefer depth (`level_moves` §10); it makes depth what remains.
    """
    n = cells["n_cells"]
    vis = np.asarray(visits, dtype=np.float64)
    dv = np.asarray(dvalue, dtype=np.float64)

    if policy == "uniform":
        drive = np.ones(n)
    elif policy == "oracle_level":
        drive = np.maximum(np.array([oracle_gain[c]["best"] for c in range(n)]), 0.0)
    elif policy == "oracle_channel":
        # The published `oracle` rung in cell coordinates: all budget to the tree, UNIFORM
        # across its levels. This is the arm that isolates the level index, and without it the
        # `channel_only` contrast is confounded -- `channel_only` puts ~83% on the tree while
        # `oracle_level` puts ~99%, so differencing them bundles better CHANNEL selection
        # together with better LEVEL selection. `oracle_channel` holds the channel allocation
        # at `oracle_level`'s and varies only the within-channel level profile, which is the
        # single controlled comparison the whole node turns on.
        drive = np.array([1.0 if cells["channel_of_cell"][c] == cells["tree_channel"] else 0.0
                          for c in range(n)])
    elif policy == "visits_level":
        drive = vis.copy()
    elif policy == "value_level":
        drive = dv.copy()
    elif policy == "channel_only":
        # aggregate the RAW tap to the CHANNEL, then spread uniformly within it
        raw = vis if visits_raw is None else np.asarray(visits_raw, dtype=np.float64)
        ch = cells["channel_of_cell"]
        drive = np.zeros(n)
        for c in np.unique(ch):
            sel = ch == c
            drive[sel] = raw[sel].sum() / sel.sum()
    elif policy == "premium_level":
        drive = np.maximum(np.asarray(premium, dtype=np.float64), 0.0)
    elif policy == "satiety_level":
        # §12 on the span-corrected tap. Satiety on a drive that is span-dominated would
        # merely harden the confound, which is the same mistake `full_loop` §3 caught when it
        # ran satiety on an INVERTED reducibility tap and got lateral recruitment -- satiety
        # amplifies whatever its drive already believes, so the drive has to be right first.
        prem = np.maximum(np.asarray(premium, dtype=np.float64), 0.0)
        sat = np.zeros(n) if satiety is None else np.asarray(satiety)
        # the floor is calibrated off the NOISE cells: their content is redrawn iid by both
        # the move and its twin, so whatever premium they show is what "no real signal here"
        # looks like on this instrument -- the self-calibrating null `verify_distractors`
        # used for LP and `full_loop` §3 used for reducibility, in the premium's currency.
        noise = [ci for ci in range(n) if cells["is_noise"][ci]]
        floor = float(max(prem[ci] for ci in noise)) if noise else 0.0
        drive = np.maximum(prem - floor * (1.0 + beta_sat * sat), 0.0)
        if drive.sum() <= 0:
            drive = prem.copy()         # everything satiated: fall back to the unsatiated tap
    else:
        raise ValueError(f"unknown cell policy {policy!r}")

    drive = np.maximum(drive, 0.0)
    w = drive / drive.sum() if drive.sum() > 0 else np.ones(n) / n
    w = (1.0 - eps) * w + eps / n
    return w / w.sum(), drive


def mean_level_of_spend(w, cells, tree_channel):
    """The direct §12 readout: where in the hierarchy the budget actually went.

    Restricted to the TREE channel and renormalised, because a distractor's level index is
    not on the same ladder -- spending on `structA|L2` is not "climbing", it is the lateral
    recruitment `full_loop` §3 already measured. Reported alongside the tree share so the
    two failure modes (never climbs / climbs the wrong channel) stay distinguishable.
    """
    sel = cells["channel_of_cell"] == tree_channel
    mass = w[sel].sum()
    if mass <= 0:
        return float("nan"), 0.0
    lv = cells["level_of_cell"][sel]
    return float((w[sel] * lv).sum() / mass), float(mass)


# --------------------------------------------------------------------------- #
# The damage schedule
# --------------------------------------------------------------------------- #

def parse_schedule(spec, rounds):
    """`"1,2,3"` -> one depth per equal-length block of rounds; `"2"` -> static at 2.

    A schedule is the whole experiment: `1,2,3` deepens the error over the run and asks
    whether allocation follows; a static `2` is the control that separates necessity
    tracking from "deep moves get more useful as the FM improves".
    """
    levels = [int(x) for x in str(spec).split(",") if x != ""]
    if len(levels) == 1:
        return [levels[0]] * rounds
    per = int(np.ceil(rounds / len(levels)))
    out = [levels[min(i // per, len(levels) - 1)] for i in range(rounds)]
    return out


def make_damage_mixture(layout, tb_np, levels):
    """Damage drawn uniformly from `levels` per batch -- the value's training distribution.

    The value is trained ONCE on this mixture and frozen, so every round's damage depth is
    in-distribution for it. That is what isolates the question: the grader is equally
    sighted at every depth by construction, so any climbing in the allocation is the
    ALLOCATOR following the world and not the value re-learning it.
    """
    dmgs = [make_damage(layout, tb_np, int(lv)) for lv in levels]
    def damage(leaves, c, rng):
        return dmgs[int(rng.integers(0, len(dmgs)))](leaves, c, rng)
    return damage


# --------------------------------------------------------------------------- #
# C4: the span FM reduces to the published block FM
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, tree_depth=4, seed=1, n=64, state_dim=96):
    """C4 -- on the flat move set the span FM is BIT-IDENTICAL to `BlockLatentFM`.

    This is what lets every published number stay a special case rather than an analogue.
    Copies the shared parameters across, runs both on the same latents, and asserts exact
    equality. Also asserts the two structural properties the reduction rests on: level-1
    span masks are one-hot, and the level-1 marker is exactly zero whatever the level
    embedding holds.
    """
    import torch
    from rhm.rhm_sculpt_latent import _build_block_fm, _fm_chunked

    torch.manual_seed(seed)
    device = torch.device("cpu")
    spec = make_spec(tree_depth=tree_depth, struct_depths=[2, 2], struct_ms=[2, 4],
                     noise_blocks=[1, 1])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    ms_flat = flat_move_set(layout, tb, device)
    n_blocks = tb["n_blocks"]

    assert ms_flat["n_moves"] == n_blocks, (
        f"flat move set has {ms_flat['n_moves']} moves against {n_blocks} blocks")
    for mi, mv in enumerate(ms_flat["moves"]):
        assert mv["blk0"] == mi and mv["span"] == 1, "flat move set is not block order"

    span_mask, level_of = move_index_tensors(ms_flat, tb, device)
    assert bool((span_mask.sum(1) == 1).all()), "level-1 span masks are not one-hot"
    assert bool((level_of == 1).all()), "flat move set contains a level > 1"

    SpanFM, BlockFM = build_span_fm(), _build_block_fm()
    span = SpanFM(state_dim, n_blocks, ms_flat["max_level"], span_mask, level_of,
                  n_head=4, n_layer=2)
    block = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2)
    # a nonzero level embedding must STILL cancel at level 1 -- that is the point of the
    # `- weight[1]` subtraction, so test it perturbed rather than at its zero init.
    with torch.no_grad():
        span.level_embedding.weight.normal_(0.0, 1.0)
    sd = span.state_dict()
    block.load_state_dict({k: sd[k] for k in block.state_dict()}, strict=True)
    span.eval(), block.eval()

    z = torch.randn(n, n_blocks, state_dim)
    k = torch.randint(0, n_blocks, (n,))
    with torch.no_grad():
        a = _fm_chunked_mv(span, z, k)
        b = _fm_chunked(block, z, k)
    err = float((a - b).abs().max())
    assert err == 0.0, f"C4 FAILED: span FM differs from block FM by {err:.3e} at level 1"

    # and the level index must actually do something above level 1, or the FM cannot
    # represent a commitment at all and the whole ladder is measuring the block FM.
    ms_level = build_move_set(layout, tb, device)
    sm2, lo2 = move_index_tensors(ms_level, tb, device)
    span2 = SpanFM(state_dim, n_blocks, ms_level["max_level"], sm2, lo2, n_head=4, n_layer=2)
    with torch.no_grad():
        span2.level_embedding.weight.normal_(0.0, 1.0)
        deep = [mi for mi, mv in enumerate(ms_level["moves"]) if mv["level"] >= 2]
        mv2 = torch.tensor([deep[0]] * n)
        mv1 = torch.tensor([0] * n)
        d = float((span2(z, mv2) - span2(z, mv1)).abs().max())
    assert d > 0, "the span FM is blind to the level index"

    cells = build_cells(ms_level, tb)
    print(f"C4 PASS -- span FM == block FM on all {n_blocks} level-1 moves (max |diff| 0.0), "
          f"with a perturbed level embedding")
    print(f"  level move set: {ms_level['n_moves']} moves over {cells['n_cells']} cells "
          f"{json.dumps(moves_table(ms_level, tb))}")
    print(f"  cells: {cells['names']}")
    return {"c4_max_diff": err, "n_moves": ms_level["n_moves"],
            "n_cells": cells["n_cells"], "cells": cells["names"]}


# --------------------------------------------------------------------------- #
# Warm start: the shared FM every arm forks from
# --------------------------------------------------------------------------- #

def warm_span_fm(fm, controller, generator, layout, tb, ms, cells, *, n_steps, batch_size,
                 n_corrupt, lr, seed, render, gen, device, damage, edit_budget,
                 pool_refresh=250):
    """Fit the span FM on a UNIFORM-over-cells budget, fresh states every step.

    Mirrors `channel_env.train_block_fm`'s protocol (corrupt start -> a few random moves ->
    one command -> predict Delta-z, with the clean pool redrawn every `pool_refresh` steps),
    with the command drawn uniformly over CELLS rather than over blocks. Uniform over cells
    rather than over moves on purpose: over moves, the 8 level-1 tree nodes would swamp the
    single root, and the warm FM would arrive at round 1 already specialised toward the
    shallow end -- which is the very thing the arms are supposed to differ on.

    Freshness is load-bearing here for the same reason it was there: a reused buffer put the
    published FM at normalised error ~1.0 against a floor of 0.000.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    fm.train()
    opt = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report = max(1, n_steps // 5)
    pool = None
    for step in range(1, n_steps + 1):
        if pool is None or step % pool_refresh == 1:
            pool = sample_pool(layout, 20_000, int(rng.integers(0, 2 ** 31)))
        idx = rng.integers(0, pool["leaves"].shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        start = damage(pool["leaves"][idx], c, rng)
        x = torch.from_numpy(start).to(device)
        for _ in range(int(rng.integers(0, edit_budget // 2 + 1))):
            mv = torch.from_numpy(_draw_moves(rng, cells, batch_size,
                                              np.ones(cells["n_cells"]) / cells["n_cells"])
                                  ).to(device)
            x = apply_moves(generator, x, mv, ms, tb, render=render, gen=gen)
        mv = torch.from_numpy(_draw_moves(rng, cells, batch_size,
                                          np.ones(cells["n_cells"]) / cells["n_cells"])
                              ).to(device)
        z, target = transition_targets_moves(controller, generator, x, mv, ms, tb,
                                             render=render, gen=gen)
        loss = F.mse_loss(fm(z, mv), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            print(f"    warm {step:6d}/{n_steps}  mse {loss.item():.5f}")
    fm.eval()
    return fm


def _draw_moves(rng, cells, n, w):
    """Sample `n` move ids: draw a CELL from `w`, then a node inside it uniformly.

    Two-stage on purpose -- it is what makes the budget a budget over (channel, level) and
    not over moves. Drawing moves directly in proportion to `w` would hand cells with more
    nodes a bigger share of the same weight, so `tree|L1` (8 nodes) and `tree|L4` (1 node)
    would not be comparable columns. `ladder.py` makes the same choice one level up
    (`chan_pick` then a uniform block inside the channel).
    """
    pick = rng.choice(cells["n_cells"], size=n, p=w)
    out = np.empty(n, dtype=np.int64)
    for ci in np.unique(pick):
        sel = pick == ci
        mis = cells["moves_of_cell"][ci]
        out[sel] = np.array(mis)[rng.integers(0, len(mis), size=int(sel.sum()))]
    return out


def print_gate(gate_rows, cells, tree_c, title="G-L"):
    """G-L: does the world make deep moves NECESSARY as the error deepens?

    `argmax L` is the readout that matters -- which level the exact DP would spend on. If it
    does not rise with the damage depth, the world never made abstraction necessary and no
    allocator can be scored for climbing, however the magnitudes look.
    """
    tree_cells = [ci for ci in range(cells["n_cells"])
                  if cells["channel_of_cell"][ci] == tree_c]
    hdr = "  ".join(f"{cells['names'][ci]:>12s}" for ci in tree_cells)
    print(f"\n{'=' * 100}\n=== G-L  best Δd* available in each TREE cell, by where the error "
          f"lives  [{title}]\n{'=' * 100}")
    print(f"{'damage':>7s}  {'d* mean':>8s}  {hdr}   {'argmax L':>9s}")
    for lv, row in sorted(gate_rows.items()):
        best = [row["gain"][ci]["best"] for ci in tree_cells]
        cellstr = "  ".join(f"{b:>+12.3f}" for b in best)
        arg = cells["level_of_cell"][tree_cells[int(np.argmax(best))]]
        print(f"{lv:>7d}  {row['dstar_mean']:>8.3f}  {cellstr}   {arg:>9d}")
    print("\n  Prediction: at damage depth d, cells with level < d collapse toward 0 while\n"
          "  cells at level >= d stay positive, so `argmax L` rises with the damage depth.")


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote(tree_depth: int = 4):
    """C4 and the move/cell census, on CPU."""
    return selfcheck(tree_depth=tree_depth)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def gate(v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96, tree_depth: int = 4,
         struct_depths: str = "2,2", struct_ms: str = "2,4", noise_blocks: str = "1,1",
         controller_steps: int = 12_000, generator_steps: int = 12_000,
         batch_size: int = 256, n_corrupt: int = 3, n_eval: int = 512,
         damage_levels: str = "0,1,2,3", n_corrupt_sweep: str = "1,2,3",
         tag: str = "v1", quick: bool = False):
    """G-D + G-L: certify the damage model, then check the world makes depth NECESSARY.

    This is the gate the ladder is not worth running without, and it is cheap: it needs a
    generator (for `regenerate_node`) and the exact DP, and no value, FM or loop. G-L's table
    is reused as `oracle_level`'s drive, so nothing here is throwaway.

    `n_corrupt` is swept because it is not a free parameter here the way it is in the
    published ladder. `corrupt_tree_hier` damages `min(n_corrupt, s**(depth-level))` nodes,
    and at level 3 there are only 2 nodes -- so `n_corrupt=3` damages the ENTIRE tree, every
    move helps a lot, and the "only a level->=k commitment can reach it" structure is swamped
    by there being nothing left to condition on. The informative regime is one wrong subtree
    inside an otherwise correct tree, and this sweep is what identifies it rather than
    assuming it.
    """
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_sculpt_latent import _build_rich_controller

    if quick:
        controller_steps = generator_steps = 600
        n_eval = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    T, n_blocks, tree_c = layout["total_len"], tb["n_blocks"], tb["tree_channel"]
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)
    ms = build_move_set(layout, tb, device)
    cells = build_cells(ms, tb)
    print(f"Gate: T={T} tokens / {n_blocks} blocks, {ms['n_moves']} moves over "
          f"{cells['n_cells']} cells {json.dumps(moves_table(ms, tb))}")

    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
    Controller, Generator = _build_rich_controller(), _build_generator()
    controller = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, torch.from_numpy(pool["leaves"]),
                           torch.from_numpy(pool["roots"].astype(np.int64)),
                           batch_size=batch_size, n_blocks=n_blocks, block_size=s,
                           n_steps=controller_steps, lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, torch.from_numpy(pool["leaves"]), tb,
                             batch_size=batch_size, n_steps=generator_steps, lr=3e-4,
                             device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    del pool

    tb_np = tb_numpy(tb)
    levels = [int(x) for x in damage_levels.split(",")]
    cs = [int(x) for x in n_corrupt_sweep.split(",")] if n_corrupt_sweep else [n_corrupt]
    sweep, certs = {}, {}
    for c in cs:
        rows = {}
        for lv in levels:
            cert = certify_damage(layout, tb_np, level=lv, n_corrupt=c,
                                  n=1024 if quick else 4096)
            certs[f"c{c}_d{lv}"] = cert
            print(f"  G-D c={c} damage={lv}: on-grammar {cert['on_grammar']:.4f}, "
                  f"d* mean {cert['dstar_mean']:.3f}, d*==0 {cert['dstar_zero_frac']:.4f}")
            if lv >= 1:
                assert cert["on_grammar"] > 0.999, f"G-D FAILED at c={c} level {lv}"
            x0, r0 = sample_states_damaged(layout, tb, generator, n=n_eval, n_corrupt=c,
                                           presteps=0, seed=seed + 42, device=device,
                                           render=render, gen=gen,
                                           damage=make_damage(layout, tb_np, lv))
            gain, dmean = cell_dstar_gain(layout, generator, x0, r0, ms, cells, tb,
                                          render=render, gen=gen)
            rows[lv] = {"gain": gain, "dstar_mean": dmean,
                        "dstar_zero_frac": cert["dstar_zero_frac"]}
        print_gate(rows, cells, tree_c, title=f"n_corrupt = {c}")
        sweep[c] = rows
    rows = sweep[n_corrupt if n_corrupt in sweep else cs[-1]]

    metrics = {"config": {"v": v, "s": s, "seed": seed, "tree_depth": tree_depth,
                          "struct_depths": struct_depths, "n_eval": n_eval,
                          "n_corrupt": n_corrupt, "damage_levels": levels,
                          "n_corrupt_sweep": cs, "quick": quick},
               "cells": cells["names"], "census": moves_table(ms, tb),
               "certify": certs,
               "gate": {str(c): {str(lv): {"dstar_mean": r["dstar_mean"],
                                           "dstar_zero_frac": r["dstar_zero_frac"],
                                           "gain": {cells["names"][ci]: r["gain"][ci]
                                                    for ci in range(cells["n_cells"])}}
                                 for lv, r in rws.items()}
                        for c, rws in sweep.items()},
               "elapsed_s": time.time() - started}
    out = f"{DATA_DIR}/directed_sculpting/level_ladder/gate_{tag}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nWrote {out}/results.json  ({metrics['elapsed_s']:.0f}s)")
    return metrics


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=28800, memory=32768)
def level_ladder(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96, tree_depth: int = 4,
    struct_depths: str = "2,2", struct_ms: str = "2,4", noise_blocks: str = "1,1",
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 4_000, fm_epochs: int = 2,
    batch_size: int = 256, n_corrupt: int = 2, edit_budget: int = 6, explore_eps: float = 0.3,
    rounds: int = 12, collect_budget: int = 8192, forecast_n: int = 512,
    premium_n: int = 256,
    n_probe_states: int = 1024, n_eval: int = 512, grade_every: int = 3,
    ballistic_chunk: int = 3, beam_width: int = 16,
    damage_schedule: str = "1,2,3", max_level: int = 0,
    drift_kappa: float = 0.05, drift_kl: float = 0.05, drift_steps_per_round: int = 20,
    beta_sat: float = 1.0, alloc_eps: float = 0.01, mon_price: float = 1.0,
    policies: str = ",".join(CELL_POLICIES), taps: str = "fc,prem",
    tag: str = "v1", quick: bool = False,
):
    """The six-arm cell-indexed ladder under a damage depth that deepens across rounds.

    `--damage-schedule 1,2,3` splits the run into three equal blocks and moves the error
    one level deeper in each; `--damage-schedule 2` holds it static and is the control that
    separates necessity tracking from "deep moves get more useful as the FM improves".
    `--max-level 1` collapses the cell index onto the channel index, which is the published
    ladder's allocation space and the back-compat reading.

    `n_corrupt` DEFAULTS TO 2 HERE, not to the published ladder's 3, and the deviation is
    forced rather than tuned. `corrupt_tree_hier` damages `min(n_corrupt, s**(depth-level))`
    nodes and level 3 has only 2 of them, so at `n_corrupt=3` the entire tree is a wrong
    derivation, there is no clean context left to condition on, and every move helps roughly
    equally -- gate G-L measures the exact DP's argmax level stuck at 3 for EVERY damage
    depth, i.e. the necessity structure is swamped and no allocator could be scored. At
    `n_corrupt=2` the argmax climbs 1/1/3/3 with `d*==0` on under 4% of rows (`n_corrupt=1`
    climbs 1/1/2/3, one rung finer, but leaves 12-14% of rows undamaged at depth >= 2 and
    fails `level_moves`' own G-D threshold). Run `gate` before changing this.
    """
    import torch
    import torch.nn.functional as F

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = 600
        value_episodes, fm_warm_steps = 4_000, 400
        rounds, collect_budget, forecast_n = 4, 1024, 128
        premium_n, n_probe_states, n_eval = 64, 256, 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    T, n_blocks, tree_c = layout["total_len"], tb["n_blocks"], tb["tree_channel"]
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)

    ms = build_move_set(layout, tb, device, max_level=(max_level or None))
    cells = build_cells(ms, tb)
    span_mask, level_of = move_index_tensors(ms, tb, device)
    sched = parse_schedule(damage_schedule, rounds)
    sched_levels = sorted(set(sched))
    policy_list = [p for p in policies.split(",") if p]

    print(f"Level ladder: T={T} tokens / {n_blocks} blocks, {ms['n_moves']} moves over "
          f"{cells['n_cells']} cells {json.dumps(moves_table(ms, tb))}")
    print(f"  cells   {cells['names']}")
    print(f"  damage schedule (round -> depth): {sched}")
    print(f"  arms    {policy_list}")

    # ---- shared frozen instruments ------------------------------------------------------
    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))
    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, SpanFM = _build_value_head(), build_span_fm()

    controller = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, train_leaves, tb, batch_size=batch_size,
                             n_steps=generator_steps, lr=3e-4, device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    del train_leaves, train_roots, pool

    # ---- the damage model, certified before anything trains on it -----------------------
    tb_np = tb_numpy(tb)
    certs = {}
    for lv in sched_levels:
        cert = certify_damage(layout, tb_np, level=lv, n_corrupt=n_corrupt,
                              n=1024 if quick else 4096)
        certs[lv] = cert
        print(f"  G-D damage={lv}: on-grammar {cert['on_grammar']:.4f}, "
              f"d* mean {cert['dstar_mean']:.3f}, d*==0 {cert['dstar_zero_frac']:.4f}")
        assert cert["on_grammar"] > 0.999, f"G-D FAILED at level {lv}"
        assert cert["dstar_zero_frac"] < 0.05, f"G-D FAILED at level {lv}: no damage"
    mixture = make_damage_mixture(layout, tb_np, sched_levels)

    # ---- the value: trained ONCE on the mixture, then frozen -----------------------------
    # Equally sighted at every depth in the schedule by construction, so any climbing in the
    # allocation is the ALLOCATOR following the world rather than the grader re-learning it.
    print(f"\nValue: {value_episodes} rollouts over {ms['n_moves']} moves, damage mixture "
          f"{sched_levels}")
    cfg, rts, suc = collect_value_buffer_moves(
        controller, generator, layout, tb, ms, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render=render, gen=gen, device=device, damage=mixture)
    print(f"  buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    value = ValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    value.eval()
    for p in value.parameters():
        p.requires_grad_(False)
    value_terminal_success = float(suc.mean())
    del cfg, rts, suc

    # ---- frozen evaluation material -----------------------------------------------------
    # x_probe is drawn from the MIXTURE so one FM-error number is comparable across every
    # round; the ballistic eval sets are per-depth so success is interpretable within a
    # schedule block. Both are frozen -- only the world and the model move.
    x_probe, _ = sample_states_damaged(layout, tb, generator, n=n_probe_states,
                                       n_corrupt=n_corrupt, presteps=2, seed=seed + 41,
                                       device=device, render=render, gen=gen, damage=mixture)
    x_fc0, r_fc0 = sample_states_damaged(layout, tb, generator, n=forecast_n,
                                         n_corrupt=n_corrupt, presteps=edit_budget // 2,
                                         seed=seed + 45, device=device, render=render,
                                         gen=gen, damage=mixture)
    # FROZEN premium probe, for the same reason the published ladder freezes its monitor
    # states: the tap is read every round and resampling injects state-draw variance into a
    # signal that is already a difference of two noisy quantities. The transitions still move
    # every round (the world drifts, the render is stochastic); only the start states are held.
    x_prem0, r_prem0 = sample_states_damaged(layout, tb, generator, n=premium_n,
                                             n_corrupt=n_corrupt, presteps=edit_budget // 2,
                                             seed=seed + 46, device=device, render=render,
                                             gen=gen, damage=mixture)

    evals, oracle_by_level = {}, {}
    for lv in sched_levels:
        xe, re = sample_states_damaged(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                       presteps=0, seed=seed + 42, device=device,
                                       render=render, gen=gen,
                                       damage=make_damage(layout, tb_np, lv))
        evals[lv] = (xe, re)
        # `d*` is a function of the rule SUPPORT alone and the drift is support-fixed (gate
        # B5), so the exact per-cell relevance depends only on the damage depth -- compute it
        # once per depth rather than once per round per arm. This table is both G-L and
        # `oracle_level`'s drive.
        gain, dmean = cell_dstar_gain(layout, generator, xe, re, ms, cells, tb,
                                      render=render, gen=gen)
        oracle_by_level[lv] = gain
        print(f"  oracle relevance at damage={lv} (d* {dmean:.3f}) computed")
    print_gate({lv: {"gain": oracle_by_level[lv], "dstar_mean": certs[lv]["dstar_mean"]}
                for lv in sched_levels}, cells, tree_c)

    # ---- warm-start FM, shared by every arm ----------------------------------------------
    warm_fm = SpanFM(state_dim, n_blocks, ms["max_level"], span_mask, level_of,
                     n_head=4, n_layer=2).to(device)
    print(f"\nWarm-starting the span FM ({fm_warm_steps} steps, uniform over cells)")
    warm_span_fm(warm_fm, controller, generator, layout, tb, ms, cells,
                 n_steps=fm_warm_steps, batch_size=batch_size, n_corrupt=n_corrupt,
                 lr=1e-3, seed=seed + 51, render=render, gen=gen, device=device,
                 damage=mixture, edit_budget=edit_budget)

    drift_spec = calibrate_drift(layout, drift_kappa, drift_kl, seed + 601)
    print("Calibrated drift: " + ", ".join(
        f"{n}@L{r['level']} KL/seq={r['kl_per_seq']:.2f}" for n, r in drift_spec.items()))

    # ---- the arms ------------------------------------------------------------------------
    results = {}
    for policy in policy_list:
        print(f"\n{'#' * 72}\n# ARM: {policy}\n{'#' * 72}")
        results[policy] = _run_policy_cells(
            policy, warm_fm, controller, generator, value, layout, tb, ms, cells, drift_spec,
            sched=sched, evals=evals, oracle_by_level=oracle_by_level, x_probe=x_probe,
            x_fc0=x_fc0, r_fc0=r_fc0, x_prem0=x_prem0, r_prem0=r_prem0,
            tb_np=tb_np, tree_c=tree_c, premium_n=premium_n,
            rounds=rounds, collect_budget=collect_budget, forecast_n=forecast_n,
            fm_epochs=fm_epochs, batch_size=batch_size, n_corrupt=n_corrupt,
            edit_budget=edit_budget, drift_steps_per_round=drift_steps_per_round,
            grade_every=grade_every, beta_sat=beta_sat, alloc_eps=alloc_eps,
            taps=tuple(t for t in taps.split(",") if t),
            ballistic_chunk=ballistic_chunk, beam_width=beam_width, mon_price=mon_price,
            render=render, gen=gen, device=device, seed=seed, state_dim=state_dim,
            SpanFM=SpanFM, span_mask=span_mask, level_of=level_of)

    # ---- the two tables this experiment exists to produce ---------------------------------
    print(f"\n{'=' * 96}\n=== THE PRIZE -- tree FM error and ballistic control\n{'=' * 96}")
    print(f"{'arm':16s} {'tree FM err ↓':>14s} {'ballistic ↑':>12s} {'tree share':>11s} "
          f"{'mean level':>11s} {'meter':>7s}")
    for p in policy_list:
        r = results[p]
        print(f"{p:16s} {r['tree_err_mean']:>14.4f} {r['ballistic_mean']:>12.3f} "
              f"{r['tree_share_mean']:>11.3f} {r['mean_level_mean']:>11.3f} "
              f"{r['meter_ratio']:>7.2f}")

    print(f"\n{'=' * 96}\n=== THE CLIMB -- mean level of TREE spend, by round\n{'=' * 96}")
    print(f"{'arm':16s} " + " ".join(f"r{i + 1:<5d}" for i in range(rounds)))
    print(f"{'damage depth':16s} " + " ".join(f"{d:<6d}" for d in sched))
    for p in policy_list:
        tr = [row["mean_level"] for row in results[p]["rounds"]]
        print(f"{p:16s} " + " ".join(f"{x:<6.2f}" for x in tr))
    print("\n  A climbing allocator's row rises with the damage-depth row. `channel_only` is\n"
          "  the control: it CANNOT climb (uniform within a channel), so its row is flat by\n"
          "  construction and its prize is what the level index has to beat.")

    metrics = {
        "config": {"v": v, "s": s, "seed": seed, "tree_depth": tree_depth,
                   "struct_depths": struct_depths, "struct_ms": struct_ms,
                   "noise_blocks": noise_blocks, "state_dim": state_dim, "rounds": rounds,
                   "collect_budget": collect_budget, "forecast_n": forecast_n,
                   "premium_n": premium_n,
                   "n_probe_states": n_probe_states, "n_eval": n_eval,
                   "edit_budget": edit_budget, "n_corrupt": n_corrupt, "fm_epochs": fm_epochs,
                   "fm_warm_steps": fm_warm_steps, "value_episodes": value_episodes,
                   "damage_schedule": damage_schedule, "schedule": sched,
                   "max_level": max_level, "beam_width": beam_width,
                   "ballistic_chunk": ballistic_chunk, "grade_every": grade_every,
                   "drift_kappa": drift_kappa, "drift_kl": drift_kl,
                   "drift_steps_per_round": drift_steps_per_round, "beta_sat": beta_sat,
                   "alloc_eps": alloc_eps, "mon_price": mon_price, "policies": policy_list, "taps": taps,
                   "quick": quick},
        "cells": cells["names"],
        "cell_levels": cells["level_of_cell"].tolist(),
        "cell_channels": cells["channel_of_cell"].tolist(),
        "census": moves_table(ms, tb),
        "certify": certs,
        "value_terminal_success": value_terminal_success,
        "gate": {str(lv): {cells["names"][ci]: oracle_by_level[lv][ci]
                           for ci in range(cells["n_cells"])} for lv in sched_levels},
        "drift": drift_spec,
        "results": results,
        "elapsed_s": time.time() - started,
    }
    out = f"{DATA_DIR}/directed_sculpting/level_ladder/ladder_{tag}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nWrote {out}/results.json  ({metrics['elapsed_s'] / 60:.1f} min)")
    return metrics


def _run_policy_cells(policy, warm_fm, controller, generator, value, layout, tb, ms, cells,
                      drift_spec, *, sched, evals, oracle_by_level, x_probe, x_fc0, r_fc0,
                      tb_np, tree_c, rounds, collect_budget, forecast_n, premium_n,
                      x_prem0, r_prem0, fm_epochs,
                      batch_size, n_corrupt, edit_budget, drift_steps_per_round, grade_every,
                      beta_sat, alloc_eps, taps, ballistic_chunk, beam_width, mon_price,
                      render, gen, device, seed, state_dim, SpanFM, span_mask, level_of):
    """One arm's trajectory. Every arm re-initialises the drift state from the SAME spec and
    steps it with the SAME rng, so all arms see a bit-identical world sequence and differ
    only in the allocation rule.

    Every arm also PAYS for the forecast tap whether or not its drive reads it, which is the
    published ladder's choice: it isolates the allocation rule at the cost of hiding the
    economics (`uniform` and `oracle_level` are getting a tap they never use for free).
    `metering_sweep` is where that subsidy gets priced; it is not this cut's variable.
    """
    import copy
    import torch
    import torch.nn.functional as F

    fm = copy.deepcopy(warm_fm).to(device)
    meter = Meter() if mon_price == 1.0 else PricedMeter(mon_price)
    reset_drift(layout, drift_spec)
    drift_rng = np.random.default_rng(seed + 777)
    prewarm_drift(layout, drift_rng, n_steps=200)
    refresh_w_blk(tb, layout, device)
    alloc_rng = np.random.default_rng(seed + 603)
    draw_rng = np.random.default_rng(seed + 604)
    satiety = np.zeros(cells["n_cells"])
    rows = []

    for rnd in range(1, rounds + 1):
        dmg_lv = sched[rnd - 1]
        advance_drift(layout, drift_rng, drift_steps_per_round)
        refresh_w_blk(tb, layout, device)

        # --- the p tap (metered, identical charge for every arm) -------------------------
        visits, dvalue, visits_raw = forecast_visits_cells(
            fm, value, controller, tb, ms, cells, x_fc0, r_fc0, budget=edit_budget,
            device=device)
        meter.charge_monitor(forecast_n)

        # --- the matched-span premium tap (metered; materialised, so much dearer) ---------
        # Skippable via `--taps fc`: it is the dearest tap by an order of magnitude and no arm
        # in a pricing run (uniform / oracle_level / channel_only) reads it. Dropping it
        # changes the meter, so a run with `--taps fc` is not meter-comparable to one without.
        if "prem" in taps:
            premium = premium_cells(value, controller, generator, tb, ms, cells, x_prem0,
                                    r_prem0, render=render, gen=gen, device=device)
            # each probe state is materialised twice per move (the move and its span-matched
            # twin), so the honest charge is 2 * n_moves sequences per state, not one.
            meter.charge_monitor(premium_n * 2 * ms["n_moves"])
        else:
            premium = np.zeros(cells["n_cells"])

        # --- allocate ---------------------------------------------------------------------
        w, drive = allocate_cells(policy, cells=cells, visits=visits, dvalue=dvalue,
                                  visits_raw=visits_raw, premium=premium,
                                  oracle_gain=oracle_by_level[dmg_lv], satiety=satiety,
                                  beta_sat=beta_sat, eps=alloc_eps)
        sat_prev = satiety.copy()
        satiety = 0.7 * satiety + w
        mean_lv, tree_share = mean_level_of_spend(w, cells, tree_c)

        # --- collect + fit ------------------------------------------------------------------
        x_col, _ = sample_states_damaged(layout, tb, generator, n=collect_budget,
                                         n_corrupt=n_corrupt, presteps=edit_budget // 2,
                                         seed=seed + 10_000 + rnd * 7, device=device,
                                         render=render, gen=gen,
                                         damage=make_damage(layout, tb_np, dmg_lv))
        mv_col = torch.from_numpy(_draw_moves(draw_rng, cells, collect_budget, w)).to(device)
        z_col, t_col = transition_targets_moves(controller, generator, x_col, mv_col, ms, tb,
                                                render=render, gen=gen)
        meter.charge_collect(collect_budget)

        fm.train()
        opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(max(1, fm_epochs * collect_budget // batch_size)):
            idx = torch.randint(0, collect_budget, (batch_size,), device=device)
            loss = F.mse_loss(fm(z_col[idx], mv_col[idx]), t_col[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()
        fm.eval()
        del z_col, t_col, x_col

        # --- grade (privileged, unmetered) ---------------------------------------------------
        errs = per_cell_error(fm, controller, generator, tb, ms, cells, x_probe,
                              render=render, gen=gen, seed=seed + 88)
        tree_cells = [ci for ci in range(cells["n_cells"])
                      if cells["channel_of_cell"][ci] == tree_c]
        # all-blocks nmse: the denominator is the full-sequence latent norm for every move, so
        # this is the one error number comparable ACROSS cells of different span. The acted
        # variant is reported per cell but not aggregated, since its denominator is the span.
        tree_err = float(np.mean([errs[ci]["nmse"] for ci in tree_cells]))
        row = {"round": rnd, "damage_level": dmg_lv,
               "alloc": {cells["names"][ci]: float(w[ci]) for ci in range(cells["n_cells"])},
               "drive": {cells["names"][ci]: float(drive[ci])
                         for ci in range(cells["n_cells"])},
               "visits": {cells["names"][ci]: float(visits[ci])
                          for ci in range(cells["n_cells"])},
               "visits_raw": {cells["names"][ci]: float(visits_raw[ci])
                              for ci in range(cells["n_cells"])},
               "dvalue": {cells["names"][ci]: float(dvalue[ci])
                          for ci in range(cells["n_cells"])},
               "premium": {cells["names"][ci]: float(premium[ci])
                           for ci in range(cells["n_cells"])},
               "satiety": {cells["names"][ci]: float(sat_prev[ci])
                           for ci in range(cells["n_cells"])},
               "error_probe": {cells["names"][ci]: errs[ci]["nmse"]
                               for ci in range(cells["n_cells"])},
               "error_probe_acted": {cells["names"][ci]: errs[ci]["nmse_acted"]
                                     for ci in range(cells["n_cells"])},
               "tree_fm_err": tree_err, "mean_level": mean_lv, "tree_share": tree_share,
               "meter": {"collect": meter.collect, "monitor": meter.monitor,
                         "ratio": meter.ratio}}
        if rnd % grade_every == 0 or rnd == rounds:
            xe, re = evals[dmg_lv]
            row["ballistic"] = open_loop_beam_moves(controller, generator, fm, value, ms, tb,
                                                    layout, xe, re, budget=edit_budget,
                                                    beam_width=beam_width, render=render,
                                                    gen=gen, device=device,
                                                    chunk_len=ballistic_chunk)
        print(f"  r{rnd:2d} dmg={dmg_lv}  tree_err {tree_err:.4f}  meanL {mean_lv:.2f}  "
              f"tree {tree_share:.2f}  "
              + ("ball %.3f  " % row["ballistic"] if "ballistic" in row else "")
              + " ".join(f"{cells['names'][ci].replace('|', '')}:{w[ci]:.2f}"
                         for ci in range(cells["n_cells"])))
        rows.append(row)

    ball = [r["ballistic"] for r in rows if "ballistic" in r]
    return {"rounds": rows,
            "tree_err_mean": float(np.mean([r["tree_fm_err"] for r in rows])),
            "tree_err_last": float(rows[-1]["tree_fm_err"]),
            "ballistic_mean": float(np.mean(ball)) if ball else float("nan"),
            "ballistic_last": float(ball[-1]) if ball else float("nan"),
            "mean_level_mean": float(np.nanmean([r["mean_level"] for r in rows])),
            "tree_share_mean": float(np.mean([r["tree_share"] for r in rows])),
            "meter_ratio": float(rows[-1]["meter"]["ratio"])}


# --------------------------------------------------------------------------- #
# The endogenous gate: can ANY readout on this value track the oracle's ordering?
# --------------------------------------------------------------------------- #

def drive_candidates(value, controller, generator, tb, ms, ms_lazy, cells, x0, roots, *,
                     render, gen):
    """Every endogenous per-cell drive we can build from this value, side by side.

    G-L established that the WORLD orders the levels correctly. This is the matching gate on
    the other side: does anything the learner can compute reproduce that ordering? Six
    candidates, differing in how they try to remove the span confound:

      abs_dv_mean     mean |ΔV| over the cell's nodes -- what `value_level` allocates by
      dv_mean         mean SIGNED ΔV: does the value think acting here HELPS
      dv_best         mean over states of the best ΔV in the cell -- the direct analogue of
                      the oracle's `best`, which is the quantity a planner can actually cash
      dv_best_span    `dv_best` divided by the span, the crudest possible span correction
      prem_random     V(informed) - V(span-matched RANDOM rewrite). Span-matched but not
                      informedness-matched: a random rewrite destroys more at a bigger span,
                      so this inherits a span term of its own
      prem_lazy       V(commit) - V(`level_moves`' lazy twin), i.e. informed-but-uncommitted
                      at the same node. Span- AND informedness-matched, the only pair that
                      isolates commitment -- but undefined at level 1, where the lazy twin IS
                      the committed move

    Scored two ways per damage depth: the rank correlation with the oracle's `best` Δ`d*`
    across TREE cells, and the mean level each drive would allocate to. A drive that tracks
    has a positive rank correlation AND a mean level that rises with the damage depth.
    """
    import torch

    n_cells = cells["n_cells"]
    acc = {k: np.zeros(n_cells) for k in
           ("abs_dv_mean", "dv_mean", "dv_best", "dv_best_span", "prem_random", "prem_lazy")}
    cnt = np.zeros(n_cells)
    span_of = np.ones(n_cells)
    best_stack = {ci: [] for ci in range(n_cells)}

    lazy_of = {}
    for mi, mv in enumerate(ms_lazy["moves"]):
        if mv.get("lazy"):
            lazy_of[(mv["channel"], mv["level"], mv["node"])] = mi

    with torch.no_grad():
        v_now = value(_block_state_chunked(controller, x0).mean(dim=1), roots)
        for mi, mv in enumerate(ms["moves"]):
            ci = int(cells["cell_of_move"][mi])
            xk = regenerate_node(generator, x0, mv, ms, tb, render=render, gen=gen)
            vk = value(_block_state_chunked(controller, xk).mean(dim=1), roots)
            dv = vk - v_now
            xr = random_node(generator, x0, mv, ms, tb, render=render, gen=gen)
            vr = value(_block_state_chunked(controller, xr).mean(dim=1), roots)
            acc["abs_dv_mean"][ci] += float(dv.abs().mean())
            acc["dv_mean"][ci] += float(dv.mean())
            acc["prem_random"][ci] += float((vk - vr).mean())
            key = (mv["channel"], mv["level"], mv["node"])
            if key in lazy_of:
                xl = regenerate_node(generator, x0, ms_lazy["moves"][lazy_of[key]], ms_lazy,
                                     tb, render=render, gen=gen)
                vl = value(_block_state_chunked(controller, xl).mean(dim=1), roots)
                acc["prem_lazy"][ci] += float((vk - vl).mean())
            best_stack[ci].append(dv.cpu().numpy())
            cnt[ci] += 1.0
            span_of[ci] = mv["span"]
    for ci in range(n_cells):
        acc["dv_best"][ci] = float(np.max(np.stack(best_stack[ci]), axis=0).mean())
        acc["dv_best_span"][ci] = acc["dv_best"][ci] / span_of[ci]
    for k in ("abs_dv_mean", "dv_mean", "prem_random", "prem_lazy"):
        acc[k] = acc[k] / np.maximum(cnt, 1.0)
    return acc


def implied_mean_level(drive, cells, tree_c):
    """Where a drive would send the budget, restricted to the tree and renormalised."""
    sel = np.array([cells["channel_of_cell"][ci] == tree_c for ci in range(cells["n_cells"])])
    d = np.maximum(np.asarray(drive, dtype=float), 0.0)[sel]
    lv = cells["level_of_cell"][sel]
    return float((d * lv).sum() / d.sum()) if d.sum() > 0 else float("nan")


def rank_corr(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def drive_check(v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
                tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
                noise_blocks: str = "1,1", controller_steps: int = 12_000,
                generator_steps: int = 12_000, value_steps: int = 12_000,
                value_episodes: int = 40_000, batch_size: int = 256, n_corrupt: int = 2,
                edit_budget: int = 6, explore_eps: float = 0.3, n_eval: int = 512,
                damage_levels: str = "1,2,3", tag: str = "v1", quick: bool = False):
    """G-E: the ENDOGENOUS gate. Does any drive this value supports track G-L's ordering?

    The ladder is not worth running until this passes. Three drives were already found pinned
    at a mean level of ~2.96-3.58 while the exact oracle moves 2.15 -> 2.68 across the damage
    schedule, so the live hypothesis is that `z.mean(dim=1)` pooling makes a large rewrite
    look valuable regardless of content -- `level_moves` open item 6 ("attack the pooling"),
    which this either confirms as the binding constraint or refutes by finding a readout that
    survives.

    No FM, no loop, no allocation: value, generator and the exact DP only.
    """
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = 600
        value_episodes, n_eval = 4_000, 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    T, n_blocks, tree_c = layout["total_len"], tb["n_blocks"], tb["tree_channel"]
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)
    ms = build_move_set(layout, tb, device)
    ms_lazy = build_move_set(layout, tb, device, lazy_twins=True)
    cells = build_cells(ms, tb)
    levels = [int(x) for x in damage_levels.split(",")]

    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
    Controller, Generator, ValueHead = (_build_rich_controller(), _build_generator(),
                                        _build_value_head())
    controller = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, torch.from_numpy(pool["leaves"]),
                           torch.from_numpy(pool["roots"].astype(np.int64)),
                           batch_size=batch_size, n_blocks=n_blocks, block_size=s,
                           n_steps=controller_steps, lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, torch.from_numpy(pool["leaves"]), tb,
                             batch_size=batch_size, n_steps=generator_steps, lr=3e-4,
                             device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    del pool

    tb_np = tb_numpy(tb)
    mixture = make_damage_mixture(layout, tb_np, levels)
    cfg, rts, suc = collect_value_buffer_moves(
        controller, generator, layout, tb, ms, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render=render, gen=gen, device=device, damage=mixture)
    print(f"  value buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    value = ValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    value.eval()
    for p in value.parameters():
        p.requires_grad_(False)
    del cfg, rts, suc

    tree_cells = [ci for ci in range(cells["n_cells"])
                  if cells["channel_of_cell"][ci] == tree_c]
    keys = ("abs_dv_mean", "dv_mean", "dv_best", "dv_best_span", "prem_random", "prem_lazy")
    out = {}
    for lv in levels:
        x0, r0 = sample_states_damaged(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                       presteps=0, seed=seed + 42, device=device,
                                       render=render, gen=gen,
                                       damage=make_damage(layout, tb_np, lv))
        oracle, dmean = cell_dstar_gain(layout, generator, x0, r0, ms, cells, tb,
                                        render=render, gen=gen)
        drives = drive_candidates(value, controller, generator, tb, ms, ms_lazy, cells,
                                  x0, r0, render=render, gen=gen)
        obest = np.array([oracle[ci]["best"] for ci in range(cells["n_cells"])])
        out[lv] = {"dstar_mean": dmean,
                   "oracle_best": {cells["names"][ci]: float(obest[ci])
                                   for ci in range(cells["n_cells"])},
                   "oracle_mean_level": implied_mean_level(obest, cells, tree_c),
                   "drives": {k: {cells["names"][ci]: float(drives[k][ci])
                                  for ci in range(cells["n_cells"])} for k in keys},
                   "rank_corr_tree": {k: rank_corr(drives[k][tree_cells], obest[tree_cells])
                                      for k in keys},
                   "rank_corr_all": {k: rank_corr(drives[k], obest) for k in keys},
                   "mean_level": {k: implied_mean_level(drives[k], cells, tree_c)
                                  for k in keys}}

    print(f"\n{'=' * 104}\n=== G-E  can any endogenous drive track the oracle's level "
          f"ordering?\n{'=' * 104}")
    print(f"{'drive':16s} " + "  ".join(f"{'d=' + str(lv) + ' meanL':>13s}" for lv in levels)
          + "   " + "  ".join(f"{'d=' + str(lv) + ' rank':>12s}" for lv in levels))
    print(f"{'ORACLE':16s} " + "  ".join(f"{out[lv]['oracle_mean_level']:>13.3f}"
                                         for lv in levels)
          + "   " + "  ".join(f"{1.0:>12.3f}" for lv in levels))
    for k in keys:
        print(f"{k:16s} " + "  ".join(f"{out[lv]['mean_level'][k]:>13.3f}" for lv in levels)
              + "   " + "  ".join(f"{out[lv]['rank_corr_tree'][k]:>+12.3f}" for lv in levels))
    print("\n  PASS = a drive whose mean level RISES with the damage depth (like ORACLE's)\n"
          "  and whose rank correlation with the oracle across tree cells is positive.\n"
          "  If every row is flat and high, `z.mean(dim=1)` pooling is the binding constraint\n"
          "  and the level ladder waits on `level_moves` open item 6.")

    metrics = {"config": {"v": v, "s": s, "seed": seed, "tree_depth": tree_depth,
                          "n_corrupt": n_corrupt, "n_eval": n_eval, "damage_levels": levels,
                          "value_episodes": value_episodes, "quick": quick},
               "cells": cells["names"], "tree_cells": [cells["names"][c] for c in tree_cells],
               "per_damage": out, "elapsed_s": time.time() - started}
    o = f"{DATA_DIR}/directed_sculpting/level_ladder/drive_{tag}"
    os.makedirs(o, exist_ok=True)
    with open(f"{o}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nWrote {o}/results.json  ({metrics['elapsed_s'] / 60:.1f} min)")
    return metrics
