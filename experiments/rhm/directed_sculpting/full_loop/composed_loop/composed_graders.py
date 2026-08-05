"""Move-indexed teachers, belief update, and disagreement instrument.

`endo_expansion/endo_graders.py` generalised from BLOCKS to the level-indexed MOVE set. Every
function here is the move-indexed image of one there, and at `flat_move_set` (level 1 only) each
is the same computation over the same candidate set -- the move set IS the block set in block
order, which `level_ladder.selfcheck`'s C4 already asserts for the FM half.

WHY A SEPARATE MODULE RATHER THAN A FLAG ON THE PUBLISHED ONE. `endo_graders` is load-bearing
for a merged node (`endo_expansion/`, PR #29) whose numbers must stay reachable. Every function
below changes the candidate axis from `n_blocks` to `n_moves`, which changes the shape of every
returned array, so a flag would be a rewrite wearing a default. Copying is the repo convention
when the alternative is many backwards-compatible modifications to a published instrument.

THE ONE THING THAT IS GENUINELY NEW, not a port
-------------------------------------------------
`belief_update_moves`. `channel_env.belief_update`'s plan term scores each candidate by rolling
the block FM one step and reading the value:

    logits[k] = value((z + block_fm(z, k)).mean(1), r) / tau ;  CE(logits, k*)

The move-indexed form stacks over MOVES and uses the span FM, so the target a teacher writes
back can be *commit to this level-ell feature* rather than only *regenerate this block*. That is
the whole content of the action-space axis: at `max_level=1` it reduces to the published term
exactly, and above it the belief is being taught an abstraction rather than a location.

THE DENSE-TERM MOVE DRAW, AND WHY IT IS NOT UNIFORM OVER MOVES
----------------------------------------------------------------
The dense (FM-local) term needs a move per step, and the obvious two choices both confound the
action-space contrast:

  uniform over MOVES   the tree holds 15 of 23 moves at max_level=4 but 8 of 14 at max_level=1,
                       so the two arms' FMs see different CHANNEL mixes. The distractor
                       geometry is the published ladder's whole relevance structure; moving it
                       between arms would make "level moves help" partly "the level arm saw
                       more tree".
  uniform over CELLS   worse: 4 tree cells of 10 vs 1 of 5, a 2x swing in tree share.

`draw_moves_block_matched` fixes both by construction. Draw a BLOCK uniformly -- the published
`torch.randint(0, n_blocks)` convention exactly -- then draw uniformly among the moves whose
span covers it. In a grammar channel of depth d every block is covered by exactly one node per
level, so this gives each level equal weight *within* a channel while leaving the channel
marginal bit-identical to uniform-over-blocks. At `max_level=1` it IS uniform-over-blocks.
"""

import numpy as np

from rhm.rhm_channels import dp_cost, possible_set_success
from rhm.directed_sculpting.full_loop.channel_env import _block_state_chunked, _reveal
from rhm.directed_sculpting.full_loop.level_moves.level_moves import (
    _behavior_step_moves, apply_moves, regenerate_node)
from rhm.directed_sculpting.full_loop.level_moves.level_ladder.level_ladder import (
    _fm_chunked_mv)


# --------------------------------------------------------------------------- #
# The matched move draw
# --------------------------------------------------------------------------- #

def moves_covering_blocks(ms, n_blocks):
    """`cover[b]` = the move ids whose span contains block `b`.

    Used by `draw_moves_block_matched`. Asserted non-empty for every block, because a block no
    move can reach is a block the dense term can never train on and the planner can never act
    on -- silently, and it would look like an FM that simply failed there.
    """
    cover = [[] for _ in range(n_blocks)]
    for mi, mv in enumerate(ms["moves"]):
        if mv.get("lazy"):
            continue
        for b in range(mv["blk0"], mv["blk0"] + mv["span"]):
            cover[b].append(mi)
    for b, c in enumerate(cover):
        assert c, f"block {b} is covered by no move"
    return cover


def draw_moves_block_matched(rng, cover, n):
    """Uniform over blocks, then uniform over the moves covering the drawn block.

    Channel marginal identical to the published uniform-over-blocks draw; level marginal flat
    within a channel. See the module docstring for why neither naive alternative works.
    """
    b = rng.integers(0, len(cover), size=n)
    out = np.empty(n, dtype=np.int64)
    for bb in np.unique(b):
        sel = b == bb
        mis = cover[int(bb)]
        out[sel] = np.array(mis)[rng.integers(0, len(mis), size=int(sel.sum()))]
    return out


# --------------------------------------------------------------------------- #
# The external evaluative teacher: the exact DP, over moves
# --------------------------------------------------------------------------- #

def grounded_candidates_moves(layout, tb, generator, ms, x, roots, *, render, gen, device):
    """`cand[mi, i]` = `d*` after move `mi` from state `i`, plus `d_cur`.

    `endo_graders.grounded_candidates` over moves. Distractor moves are filled with `d_cur`
    rather than materialised, because P1 (`verify_distractors`) makes "a non-tree move cannot
    change `d*`" exact rather than approximate -- and `partial_hetero`'s G1 re-certifies it at
    every sharing depth. The tree moves at every level ARE materialised, which is the expensive
    part and the reason one DP pass per round is shared across all arms.
    """
    import torch

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x.cpu().numpy(), roots_np)
    cand = np.repeat(d_cur[None, :], ms["n_moves"], axis=0).astype(np.int64)
    tree_c = tb["tree_channel"]
    with torch.no_grad():
        for mi, mv in enumerate(ms["moves"]):
            if mv["channel"] != tree_c or mv.get("lazy"):
                continue
            xk = regenerate_node(generator, x, mv, ms, tb, render=render, gen=gen)
            cand[mi] = dp_cost(layout, xk.cpu().numpy(), roots_np)
    return cand, d_cur


def kstar_from_candidates(cand, d_cur, rng):
    """Argmin over moves with RANDOM tie-breaking, plus the informative mask.

    Lowest-index tie-breaking is what `metered_target`'s docstring records costing a readout: at
    L=4 move 0 is a tree move, so every uninformative state would be silently taught a
    tree-shaped target and the teacher's `tree_share` would be manufactured out of ties. On the
    level-indexed set the same hazard is worse, because move 0 is a level-1 tree move, so ties
    would manufacture a SHALLOW teacher as well as a tree-shaped one -- which is precisely the
    thing the action-space contrast is trying to measure.
    """
    best = cand.min(axis=0)
    informative = best < d_cur
    ties = cand <= best[None, :] + 1e-9
    k = (rng.random(cand.shape) * ties).argmax(axis=0)
    return k.astype(np.int64), informative


# --------------------------------------------------------------------------- #
# The endogenous evaluative teacher: Monte-Carlo terminal task success, over moves
# --------------------------------------------------------------------------- #

def rollout_terminal_success_moves(controller, generator, layout, tb, ms, x, roots, *,
                                   horizon, epsilon, render, gen, device):
    """Roll the controller-greedy behaviour policy over MOVES and grade the terminal state.

    The grader is `possible_set_success` -- the environment's own terminal reward. No DP, no
    rule tables, no channel labels. Each step materialises all `n_moves` proposals, so an honest
    meter charges `horizon * n_moves` per row; that cost RISES with the action space, which is
    a real property of evaluative grading over a richer action set and is reported, not
    equalised away.
    """
    import torch

    with torch.no_grad():
        xx = x
        for _ in range(horizon):
            xx = _behavior_step_moves(controller, generator, xx, roots, ms, tb, epsilon,
                                      render=render, gen=gen, device=device)
        succ = possible_set_success(layout, xx.cpu().numpy(), roots.cpu().numpy())
    return succ.astype(np.float64)


def collect_rollout_moves_mv(controller, generator, layout, tb, ms, x, roots, *, horizon,
                             epsilon, n_roll, render, gen, device, rng, allowed=None):
    """The ENDOGENOUS evaluative teacher over moves. Returns `(R_hat, mv_hat, informative)`.

    `allowed` is an optional bool mask over moves -- the "PI instruction". Disallowed moves are
    NOT rolled out at all (a student told not to work below an altitude does not run those
    experiments either), so a tighter instruction is CHEAPER, and their `R_hat` rows are left at
    -inf so they cannot win the argmax or count toward the informative mask.

    The instruction constrains only what the teacher may COMMIT to. The behaviour policy inside
    the rollout stays unrestricted, because the rollout must estimate the same quantity it
    estimates in the unfloored arm -- what actually happens after this move, under the agent's
    own policy. Flooring the behaviour policy too would change the estimand rather than the
    teacher, and the arms would no longer differ in one variable.

    Every move is evaluated, distractors and deep commitments alike; the teacher is never told
    which channel is the tree nor which level is "right". A wasted move still costs a step of
    the remaining budget, so the rollout penalises distractors on its own -- whether it does is
    `teacher_tree_share`, a measurement.

    The informative mask and the tie-break are `endo_graders.collect_rollout_moves`' verbatim,
    and both matter more here: terminal success is binary over a SHORT horizon, and the level
    set has more candidates, so the probability that `R_hat` is constant across all of them
    changes with the action space. `informative_frac` is therefore reported per arm and is the
    first thing to read if the two action spaces' teachers behave differently.
    """
    import torch

    n_moves, n = ms["n_moves"], x.shape[0]
    if allowed is None:
        allowed = np.ones(n_moves, dtype=bool)
    assert allowed.any(), "the instruction excluded every move"
    r_hat = np.full((n_moves, n), -np.inf)
    for mi, mv in enumerate(ms["moves"]):
        if mv.get("lazy") or not allowed[mi]:
            continue
        r_hat[mi] = 0.0
        for _ in range(n_roll):
            xk = regenerate_node(generator, x, mv, ms, tb, render=render, gen=gen)
            r_hat[mi] += rollout_terminal_success_moves(
                controller, generator, layout, tb, ms, xk, roots, horizon=horizon,
                epsilon=epsilon, render=render, gen=gen, device=device)
        r_hat[mi] /= n_roll

    best = r_hat.max(axis=0)
    ties = r_hat >= best[None, :] - 1e-9
    # `min` over the ALLOWED rows only -- with -inf fills a plain min would call every state
    # informative, which would silently disable the mask that exists to stop the plan term
    # training a hard CE on a coin flip.
    worst = np.where(allowed[:, None], r_hat, np.inf).min(axis=0)
    informative = worst < best - 1e-9
    mv_hat = (rng.random(r_hat.shape) * ties).argmax(axis=0)
    return r_hat, mv_hat.astype(np.int64), informative


# --------------------------------------------------------------------------- #
# The wireheading readout
# --------------------------------------------------------------------------- #

def value_calibration_moves(controller, generator, value, layout, tb, ms, x, roots, *,
                            horizon, epsilon, n_roll, render, gen, device):
    """Does the value's REPORTED success probability outrun the one it realises over moves?

    `endo_graders.value_calibration`, with the behaviour policy over the arm's own action space
    so that "what the value predicts" and "what the policy achieves" are denominated in the same
    actions. Comparing a value trained on the level set against a block-only realisation would
    make `bias` an action-space artifact rather than a wireheading readout.
    """
    import torch

    with torch.no_grad():
        p = torch.sigmoid(value(_block_state_chunked(controller, x).mean(dim=1),
                                roots)).cpu().numpy().astype(np.float64)
    y = np.zeros(x.shape[0])
    for _ in range(n_roll):
        y += rollout_terminal_success_moves(controller, generator, layout, tb, ms, x, roots,
                                            horizon=horizon, epsilon=epsilon, render=render,
                                            gen=gen, device=device)
    y /= n_roll
    pc, yc = p - p.mean(), y - y.mean()
    den = float(np.linalg.norm(pc) * np.linalg.norm(yc))
    return {"reported_mean": float(p.mean()), "realised_mean": float(y.mean()),
            "bias": float(p.mean() - y.mean()),
            "corr": float((pc * yc).sum() / den) if den > 1e-12 else float("nan")}


# --------------------------------------------------------------------------- #
# The grader-disagreement instrument, over moves
# --------------------------------------------------------------------------- #

def move_scores_mv(controller, generator, fm, value, layout, tb, ms, x, roots, *, horizon,
                   epsilon, n_roll, render, gen, device, fm_fresh=None, want_dp=True,
                   want_roll=True):
    """Score every MOVE at every probe state under graders of different type. (N, n_moves) each.

    `endo_graders.move_scores` over moves. One change of substance: `fm_pred` is the negative
    squared FM error over the move's SPAN rather than at a single acted block, because a
    level-ell move rewrites `s**(ell-1)` blocks and scoring only one of them would read a deep
    move's prediction quality off an arbitrary sixteenth of what it did. Raw squared error, not
    NMSE, for the reason that docstring records (a regeneration often returns the span
    unchanged, so per-transition normalisation divides by ~0); only the within-state ranking is
    used, and the span sum preserves it.

    Note the span sum is over MORE blocks for deeper moves, so `fm_pred` is not a level-neutral
    magnitude. It is used only through `argmax` and `rank_corr` within a state, where a
    monotone per-move rescaling would still matter -- so this readout is reported as a
    disagreement diagnostic and is NOT the arm's FM quality metric. `fm_error_moves(acted_only)`
    on the frozen probe is that.
    """
    import torch

    n, n_moves = x.shape[0], ms["n_moves"]
    roots_np = roots.cpu().numpy()
    out = {k: np.zeros((n, n_moves)) for k in ("fm_pred", "belief", "value")}
    if fm_fresh is not None:
        out["fm_pred_fresh"] = np.zeros((n, n_moves))
    if want_dp:
        out["dp"] = np.zeros((n, n_moves))
        d_cur = dp_cost(layout, x.cpu().numpy(), roots_np)
    if want_roll:
        out["roll"] = np.zeros((n, n_moves))

    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        v_now = value(z.mean(dim=1), roots)
        logp_now = controller.root_logits(z.mean(dim=1)).log_softmax(-1) \
            .gather(1, roots[:, None]).squeeze(1)
        for mi, mv in enumerate(ms["moves"]):
            mm = torch.full((n,), mi, device=device, dtype=torch.long)
            xk = regenerate_node(generator, x, mv, ms, tb, render=render, gen=gen)
            zk = _block_state_chunked(controller, xk)
            delta = zk - z
            w = fm.span_mask[mm][..., None]
            out["fm_pred"][:, mi] = -(((_fm_chunked_mv(fm, z, mm) - delta) ** 2) * w) \
                .sum(dim=(1, 2)).cpu().numpy()
            if fm_fresh is not None:
                out["fm_pred_fresh"][:, mi] = -(((_fm_chunked_mv(fm_fresh, z, mm) - delta) ** 2)
                                                * w).sum(dim=(1, 2)).cpu().numpy()
            out["value"][:, mi] = (value(zk.mean(dim=1), roots) - v_now).cpu().numpy()
            out["belief"][:, mi] = (
                controller.root_logits(zk.mean(dim=1)).log_softmax(-1)
                .gather(1, roots[:, None]).squeeze(1) - logp_now).cpu().numpy()
            if want_dp:
                out["dp"][:, mi] = d_cur - dp_cost(layout, xk.cpu().numpy(), roots_np)
            if want_roll:
                acc = np.zeros(n)
                for _ in range(n_roll):
                    acc += rollout_terminal_success_moves(
                        controller, generator, layout, tb, ms, xk, roots, horizon=horizon,
                        epsilon=epsilon, render=render, gen=gen, device=device)
                out["roll"][:, mi] = acc / n_roll
    return out


# --------------------------------------------------------------------------- #
# The move-indexed belief update -- the one genuinely new object
# --------------------------------------------------------------------------- #

def belief_update_moves(mode, controller, span_fm, value, generator, layout, tb, ms, cover, *,
                        train_leaves, train_roots, gstates, groots, gmvstar, n_steps,
                        batch_size, lr, lam_fm, lam_plan, tau, render, gen, device,
                        p_full=0.5, seed=0, pstates=None, proots=None):
    """`channel_env.belief_update` with the candidate axis being MOVES.

      frozen      root-CE only -- the no-loop floor. Action-space independent by construction,
                  so the two `frozen` arms are a free consistency check on the harness.
      dense       + lam_fm * the asymmetric FM local loss over a block-matched random move.
                  ENDOGENOUS but HOMOGENEOUS -- a functional of the model's own dense error.
      evaluative  + lam_plan * CE over ALL moves of the FM-rolled value against `mv*`.

    `dense subset evaluative` exactly as in the published version, so `evaluative - dense` still
    isolates grounding. At `max_level=1` every term reduces to the published one: the move set is
    the block set in block order, `draw_moves_block_matched` is uniform-over-blocks, and C4 makes
    the span FM bit-identical to the block FM.

    `pstates`/`proots` give the PLAN term its own state pool, as upstream: an endogenous teacher
    has no opinion on states where the rollout is constant across candidates, and a hard CE on a
    coin flip is training on noise. The DENSE pool stays the full `gstates`, so arms differ in
    their target and in nothing else.
    """
    import torch
    import torch.nn.functional as F

    controller.train()
    params = list(controller.parameters())
    if mode in ("dense", "evaluative"):
        span_fm.train()
        params += list(span_fm.parameters())
    if mode == "evaluative":
        value.train()
        params += list(value.parameters())
    elif mode not in ("frozen", "dense"):
        raise ValueError(mode)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n_blocks, n_moves = tb["n_blocks"], ms["n_moves"]
    n_pool = train_leaves.shape[0]
    arange_m = torch.arange(n_moves, device=device)
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
            mv = torch.from_numpy(draw_moves_block_matched(rng, cover, batch_size)).to(device)
            with torch.no_grad():
                x2 = apply_moves(generator, x, mv, ms, tb, render=render, gen=gen)
            z = controller.block_state(x)
            z2 = controller.block_state(x2)
            delta = z2 - z
            loss = loss + lam_fm * (F.mse_loss(span_fm(z.detach(), mv), delta.detach())
                                    + F.mse_loss(delta, span_fm(z, mv).detach()))

        if mode == "evaluative":
            px = gstates if pstates is None else pstates
            pr = groots if proots is None else proots
            sub = rng.integers(0, px.shape[0], size=batch_size)
            si = torch.from_numpy(sub).to(device)
            zp = controller.block_state(px[si])
            rp = pr[si]
            logits = torch.stack([
                value((zp + span_fm(zp, arange_m[j].expand(batch_size))).mean(dim=1), rp)
                for j in range(n_moves)], dim=1) / tau
            loss = loss + lam_plan * F.cross_entropy(logits, gmvstar[si])
            last["plan_acc"] = float((logits.argmax(1) == gmvstar[si]).float().mean())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    controller.eval()
    span_fm.eval()
    value.eval()
    return last


# --------------------------------------------------------------------------- #
# FM diagnostics over moves
# --------------------------------------------------------------------------- #

def fm_one_step_check_moves(fm, value, controller, generator, tb, ms, x, roots, *, render, gen,
                            device):
    """`channel_env.fm_one_step_check` over moves.

    `value_top1_agree` -- does the FM-rolled value rank moves the way the MATERIALISED value
    does -- is the bottleneck the whole sculpting arc turns on, and is the transferable readout
    the expansion 2x2 reports alongside ballistic control. Read on a FRESH FM it is the honest
    one: an arm cannot improve it by making its own co-trained FM agree with its own value.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        batch, n_moves = x.shape[0], ms["n_moves"]
        true_v = torch.empty(batch, n_moves, device=device)
        pred_v = torch.empty(batch, n_moves, device=device)
        cos = torch.zeros(n_moves, device=device)
        for mi, mv in enumerate(ms["moves"]):
            mm = torch.full((batch,), mi, device=device, dtype=torch.long)
            x2 = regenerate_node(generator, x, mv, ms, tb, render=render, gen=gen)
            tz = _block_state_chunked(controller, x2)
            pz = z + _fm_chunked_mv(fm, z, mm)
            cos[mi] = F.cosine_similarity((pz - z).flatten(1), (tz - z).flatten(1),
                                          dim=-1).mean()
            true_v[:, mi] = value(tz.mean(dim=1), roots)
            pred_v[:, mi] = value(pz.mean(dim=1), roots)
        top1 = float((true_v.argmax(1) == pred_v.argmax(1)).float().mean())
        tc = true_v - true_v.mean(1, keepdim=True)
        pc = pred_v - pred_v.mean(1, keepdim=True)
        rank = float(((tc * pc).sum(1)
                      / (tc.norm(dim=1) * pc.norm(dim=1)).clamp_min(1e-8)).mean())
        cosn = cos.cpu().numpy()
        per_level = {}
        for mi, mv in enumerate(ms["moves"]):
            per_level.setdefault(int(mv["level"]), []).append(float(cosn[mi]))
    return {"delta_cos": float(cosn.mean()), "value_top1_agree": top1,
            "value_rank_corr": rank,
            "delta_cos_per_level": {k: float(np.mean(v)) for k, v in sorted(per_level.items())}}
