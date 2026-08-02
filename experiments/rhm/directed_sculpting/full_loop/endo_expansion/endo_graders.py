"""Endogenous evaluative teachers, the metered allocator, and the grader-disagreement instrument.

Everything the published expansion 2x2 does NOT have. `../expansion.py` has one teacher --
`collect_grounded_moves`, the exact-DP best move `k*` -- and it is external, precomputed, and
privileged. `ideas/meta_learning_under_metered_data.md` names the resulting hole as the central
one in the program: *"`visits` is endogenous but was never tested for expansion... Conversely the
DP `k*` teacher expands but is external. No run is both endogenous and expanding."*

THE TWO OUTER-LOOP ACTIONS, KEPT APART
--------------------------------------
  choose WHAT   write back a per-state target the inner loop then fits.
                -> `collect_rollout_moves` (paid), `collect_value_moves` (reported).
  choose WHERE  allocate a metered budget over where the target gets computed.
                -> `forecast_visits_per_block` + `grounded_candidates`/`metered_target`.
An arm does one or the other. Doing both at once makes the result unattributable, which is the
whole reason the metered-data doc keeps *"allocation decides where `k*` is computed"* separate
from the teacher itself.

PAID VS REPORTED -- WHY THERE ARE TWO ENDOGENOUS TEACHERS
----------------------------------------------------------
`ideas/adaptive_core_and_hierarchy_climb.md` §7: *"an endogenous value must be denominated in a
currency the agent PAYS, not one it REPORTS."* The repo has four instances of the reported
version capping or being gamed, sharpest in `RHM_EDIT_CONTROL`: a value collapsed onto the belief
scalar `P(root=r*)` drives that scalar to ~1.0 while ~99% of sequences go off-grammar. So:

  `collect_rollout_moves`  target = argmax_k of a MONTE-CARLO estimate of terminal task success
                           after executing k and then behaving. Sampled, not asserted; the
                           agent cannot inflate it without actually reaching the goal. Costs
                           n_blocks * n_roll * (1 + horizon*n_blocks) materialisations PER STATE,
                           which is the point -- this is what an evaluative grader costs when
                           nobody hands you `d*`.
  `collect_value_moves`    target = argmax_k V(z of the materialised next state). Same shape,
                           reported currency. Built ON PURPOSE as the contrast that turns §7's
                           design rule into a measurement.

Neither touches `dp_cost`, the rule tables, or channel labels. Both touch the environment's own
terminal reward, which is exactly what *evaluative* means (`heterogeneous_graders.md` §4:
"referencing outcomes, and the informative outcomes are the ones you would rather not sample")
and is the only signal the ladder's `visits` MC value has ever been trained on.

THE DISAGREEMENT INSTRUMENT
---------------------------
`heterogeneous_graders.md` §9 names heterogeneous-vs-homogeneous grader disagreement as the
load-bearing test of the entire frame, calls it cheap on existing substrate, and it has never
been built. `grader_disagreement` is the first half of it, nearly for free: five move-scorers on
one frozen probe, all pairwise top-1 disagreement / rank correlation / magnitude. Its job here is
narrower and non-negotiable -- if the endogenous arm's outer signal ranks moves like the dense
inner signal, THE TWO LOOPS HAVE COLLAPSED INTO ONE and any gain it shows is an inner-loop gain
wearing an outer-loop label. That is the diagnosis to reach for before concluding anything.

The homogeneous floor is `fm_pred` vs `fm_pred_fresh` -- same objective, different init and data,
i.e. `mjc/ballistic/directed/` S1's seed-ensemble baseline, which is what a heterogeneous grader
has to beat.
"""

import numpy as np

from rhm.directed_sculpting.full_loop.channel_env import (
    _behavior_step, _block_state_chunked, _fm_chunked, regenerate_block)


# --------------------------------------------------------------------------- #
# The paid currency: Monte-Carlo terminal task success
# --------------------------------------------------------------------------- #

def rollout_terminal_success(controller, generator, layout, tb, x, roots, *, horizon,
                             epsilon, render, gen, device):
    """Roll the controller-greedy behaviour policy `horizon` steps and grade the terminal state.

    The grader is `possible_set_success` -- the environment's own terminal reward, exact and
    binary, and the only label `collect_value_buffer` has ever used. No DP, no rule table, no
    channel labels. Each step materialises all `n_blocks` proposals (that is what the behaviour
    policy costs), so a caller wanting an honest meter should charge `horizon * n_blocks` per row.
    """
    from rhm.rhm_channels import possible_set_success
    import torch

    with torch.no_grad():
        xx = x
        for _ in range(horizon):
            xx = _behavior_step(controller, generator, xx, roots, tb, epsilon,
                                render=render, gen=gen, device=device)
        succ = possible_set_success(layout, xx.cpu().numpy(), roots.cpu().numpy())
    return succ.astype(np.float64)


def collect_rollout_moves(controller, generator, layout, tb, x, roots, *, horizon, epsilon,
                          n_roll, render, gen, device, rng):
    """The ENDOGENOUS evaluative teacher. Returns (R_hat, k_hat, informative).

      R_hat        (n_blocks, N) MC estimate of terminal task success after move k.
      k_hat        (N,) hard argmax with RANDOM tie-breaking.
      informative  (N,) bool -- False where R_hat is constant across all candidates.

    Every candidate is evaluated, including distractor blocks. The teacher is NOT told which
    blocks belong to the tree; `collect_grounded_moves` may skip them because a distractor move
    provably cannot move `d*`, but that is privileged knowledge and using it here would smuggle
    the channel oracle back in. A wasted move still costs a step of the remaining budget, so the
    rollout penalises distractors on its own -- and whether it actually does is a measurement
    (`teacher_tree_share`), not an assumption.

    WHY THE INFORMATIVE MASK. Terminal success is binary and the horizon is short, so `R_hat` is
    frequently constant across all 14 candidates (every move loses, or every move wins). A hard
    argmax there is a coin flip, and training a CE on a coin flip is training on noise -- it is
    the `endo_shuffled` control applied to a random subset of states, silently. Masking those
    states out of the plan term and reporting the surviving fraction is the honest handling; the
    dense term still sees the unfiltered pool, so arms stay matched on everything else.

    TIE-BREAKING. `u * ties` gives every tied maximum an independent U(0,1) key and every
    non-maximum exactly 0, so argmax picks uniformly among the maxima. Lowest-index tie-breaking
    would put the teacher on block 0 whenever the rollout is uninformative, which at L=4 is a
    tree block -- i.e. it would manufacture a tree-shaped teacher out of pure noise.
    """
    import torch

    n_blocks, n = tb["n_blocks"], x.shape[0]
    r_hat = np.zeros((n_blocks, n))
    for k in range(n_blocks):
        kk = torch.full((n,), k, device=device, dtype=torch.long)
        for _ in range(n_roll):
            xk = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            r_hat[k] += rollout_terminal_success(controller, generator, layout, tb, xk, roots,
                                                 horizon=horizon, epsilon=epsilon,
                                                 render=render, gen=gen, device=device)
    r_hat /= n_roll

    best = r_hat.max(axis=0)
    ties = (r_hat >= best[None, :] - 1e-9)
    informative = (r_hat.min(axis=0) < best - 1e-9)
    k_hat = (rng.random(r_hat.shape) * ties).argmax(axis=0)
    return r_hat, k_hat.astype(np.int64), informative


def collect_value_moves(controller, generator, value, tb, x, roots, *, render, gen, device):
    """The REPORTED-currency teacher: argmax_k V(z of the MATERIALISED next state).

    Deliberately built to wirehead. The value is trainable in this loop, so this target trains
    the belief to make the value's imagined ranking agree with the value's own materialised
    ranking -- a closed circuit with no paid anchor, which is `RHM_EDIT_CONTROL`'s failure
    geometry exactly. It is here as the CONTRAST that makes §7's paid/reported design rule a
    measurement instead of an assertion, and it is read on `value_calibration`, not on its
    headline.
    """
    import torch

    n_blocks, n = tb["n_blocks"], x.shape[0]
    scores = torch.empty(n, n_blocks, device=device)
    with torch.no_grad():
        for k in range(n_blocks):
            kk = torch.full((n,), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            scores[:, k] = value(_block_state_chunked(controller, xk).mean(dim=1), roots)
    return scores.argmax(dim=1), scores


# --------------------------------------------------------------------------- #
# choose WHERE: endogenous allocation of the EXTERNAL target
# --------------------------------------------------------------------------- #

def forecast_visits_per_block(fm, value, controller, tb, x0, roots, *, budget, device):
    """`forecast_visits` at BLOCK granularity -- roll the FM open-loop under the greedy planner
    and count which blocks the imagined plan acts on.

    Identical machinery to the ladder's relevance tap (the `p` tap), just not aggregated to
    channels, because the metered decision here is *which blocks does the DP get to look at*.
    Zero environment interaction: nothing is materialised, the whole forecast runs on the FM's
    own predictions, which is what makes it cheap enough to be the allocator.
    """
    import torch

    with torch.no_grad():
        z = _block_state_chunked(controller, x0)
        batch = z.shape[0]
        counts = torch.zeros(tb["n_blocks"], device=device)
        for _ in range(budget):
            scores = []
            for k in range(tb["n_blocks"]):
                kk = torch.full((batch,), k, device=device, dtype=torch.long)
                scores.append(value((z + _fm_chunked(fm, z, kk)).mean(dim=1), roots))
            scores = torch.stack(scores, dim=1)
            best = scores.argmax(dim=1)
            counts.index_add_(0, best, torch.ones(batch, device=device))
            z = z + _fm_chunked(fm, z, best)
        return (counts / counts.sum().clamp_min(1)).cpu().numpy()


def grounded_candidates(layout, tb, generator, x, roots, *, render, gen, device):
    """The full DP candidate table: `cand[k, i]` = `d*` after move k from state i, plus `d_cur`.

    Identical in content to `collect_grounded_moves` -- tree blocks are materialised and scored,
    distractors are filled with `d_cur` because a distractor move provably cannot move `d*` (P1),
    which is exact rather than an approximation. Returning the whole table rather than only its
    argmin is what lets one DP pass per round serve three purposes: the `evaluative` arm's
    target, every metered arm's target, and the `agree_dp` diagnostic every endogenous arm is
    scored against. One computation, so no arm's teacher is compared across a different
    state draw.
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
    return cand, d_cur


def metered_target(cand, d_cur, allowed, rng):
    """`k*` computed on ONLY the blocks in `allowed`; everything else assumed to be a no-op.

    sculpt-continual's own untested next step #3 (*"allocating where `k*` is computed"*), which
    that node flags as possibly strengthening the weak ratchet **or collapsing it**.

    The fill for unevaluated blocks is `d_cur` -- "I did not look, so I assume no change." Right
    for a distractor, optimistically neutral for an unlooked-at tree block, and crucially it does
    NOT require knowing which is which: the metered teacher is privileged about `d*` on the
    blocks it paid to evaluate and about nothing else. (The values for allowed *distractor*
    blocks are filled analytically rather than materialised, because P1 makes that exact and not
    an approximation -- but the meter is still charged for them, since a teacher without P1 would
    have had to look.)

    RANDOM TIE-BREAK AND AN INFORMATIVE MASK, AND WHY THIS IS NOT COSMETIC. The smoke ran with
    lowest-index tie-breaking and `alloc_uniform` reported a tree share of **1.000** while the
    DP was only allowed to look at blocks {9, 4, 13} -- one tree block out of three. Every state
    where no allowed block improved `d*` was silently taught "regenerate block 0", and at L=4
    block 0 is a tree block. That is a free on-target teacher manufactured out of an arbitrary
    tie-break, handed preferentially to the arm that deserves it least. So: ties break uniformly
    among the minima, and states where NO allowed block strictly improves `d*` are marked
    uninformative and dropped from the plan term -- the same convention the rollout teacher uses,
    which is what makes the two comparable.

    `evaluative` deliberately keeps the published lowest-index convention (it is the reproduced
    anchor and must not move); its own no-improving rate is reported so the size of the artifact
    on that arm is visible rather than assumed small.
    """
    allowed = [int(b) for b in allowed]
    masked = np.repeat(d_cur[None, :], cand.shape[0], axis=0).astype(cand.dtype)
    for k in allowed:
        masked[k] = cand[k]
    best = masked.min(axis=0)
    informative = best < d_cur
    ties = masked <= best[None, :] + 1e-9
    k = (rng.random(masked.shape) * ties).argmax(axis=0)
    return k.astype(np.int64), informative


# --------------------------------------------------------------------------- #
# The wireheading readout: paid vs reported, per round
# --------------------------------------------------------------------------- #

def value_calibration(controller, generator, value, layout, tb, x, roots, *, horizon, epsilon,
                      n_roll, render, gen, device):
    """Does the value's REPORTED success probability outrun the one it actually realises?

    `V` is a BCE-trained logit of eventual terminal success, so `sigmoid(V)` is directly
    comparable to the realised success rate of rolling the current behaviour policy from the same
    states. Wireheading is `bias` climbing: the model asserting more success than it collects.
    `corr` separates "inflated but still ordered" from "detached".

    Read alongside `beam_belief_vs_truth`. Together they are `RHM_EDIT_CONTROL`'s Layer-1/Layer-2
    table -- the belief scalar going to 1.0 while true success stays at 0 -- ported to this loop.
    """
    import torch

    with torch.no_grad():
        p = torch.sigmoid(value(_block_state_chunked(controller, x).mean(dim=1),
                                roots)).cpu().numpy().astype(np.float64)
    y = np.zeros(x.shape[0])
    for _ in range(n_roll):
        y += rollout_terminal_success(controller, generator, layout, tb, x, roots,
                                      horizon=horizon, epsilon=epsilon, render=render,
                                      gen=gen, device=device)
    y /= n_roll
    pc, yc = p - p.mean(), y - y.mean()
    den = float(np.linalg.norm(pc) * np.linalg.norm(yc))
    return {"reported_mean": float(p.mean()), "realised_mean": float(y.mean()),
            "bias": float(p.mean() - y.mean()),
            "corr": float((pc * yc).sum() / den) if den > 1e-12 else float("nan")}


def beam_belief_vs_truth(controller, layout, x_final, roots):
    """The ballistic beam's terminal states, read by BOTH graders.

    `true_success` is the paid one (possible-set, exact). `belief_p_rstar` is the reported one
    (the controller's own probability that the sequence realises r*). Edit-control's finding was
    that these come apart completely -- 1.0 against 0.006 -- when the loop is allowed to shape
    what it is graded by. Divergence here, in the same direction, is wireheading.
    """
    import torch
    from rhm.rhm_channels import possible_set_success

    with torch.no_grad():
        logp = controller.root_logits(
            _block_state_chunked(controller, x_final).mean(dim=1)).log_softmax(-1)
        p = logp.gather(1, roots[:, None]).squeeze(1).exp().mean().item()
    succ = possible_set_success(layout, x_final.cpu().numpy(), roots.cpu().numpy())
    return {"belief_p_rstar": float(p), "true_success": float(succ.mean())}


# --------------------------------------------------------------------------- #
# The grader-disagreement instrument (heterogeneous_graders.md §9, first half)
# --------------------------------------------------------------------------- #

def move_scores(controller, generator, fm, value, layout, tb, x, roots, *, horizon, epsilon,
                n_roll, render, gen, device, fm_fresh=None, want_dp=True, want_roll=True):
    """Score every candidate move at every probe state under five graders of different TYPE.

    Returns a dict name -> (N, n_blocks) float array. All five see the SAME materialised next
    states, so nothing here is a comparison between different data.

      fm_pred        -(acted-block squared FM error). Literally what the `dense` arm's loss
                     rewards: "how predictable is this move to me". Raw squared error rather
                     than the normalised NMSE on purpose -- a generator regeneration often
                     returns the block unchanged, so per-transition normalisation divides by ~0
                     (the hazard `fm_error`'s docstring records costing a readout of 8e11). Only
                     the WITHIN-STATE ranking is used, and raw error preserves it.
      fm_pred_fresh  the same under an independently trained FM. Same objective, different init
                     and data -> the HOMOGENEOUS disagreement floor.
      belief         Δ log P(root=r*) under the controller. The edit-control gameable scalar.
      dp             -Δ`d*`, exact. Privileged; DIAGNOSTIC ONLY -- it enters a loss in the
                     `evaluative` and `alloc_*` arms and nowhere else.
      roll           the endogenous MC terminal-success estimate.
      value          Δ V on the materialised next state -- the reported evaluative grader.
    """
    import torch
    from rhm.rhm_channels import dp_cost

    n, n_blocks = x.shape[0], tb["n_blocks"]
    roots_np = roots.cpu().numpy()
    out = {k: np.zeros((n, n_blocks)) for k in ("fm_pred", "belief", "value")}
    if fm_fresh is not None:
        out["fm_pred_fresh"] = np.zeros((n, n_blocks))
    if want_dp:
        out["dp"] = np.zeros((n, n_blocks))
        d_cur = dp_cost(layout, x.cpu().numpy(), roots_np)
    if want_roll:
        out["roll"] = np.zeros((n, n_blocks))

    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        v_now = value(z.mean(dim=1), roots)
        logp_now = controller.root_logits(z.mean(dim=1)).log_softmax(-1) \
            .gather(1, roots[:, None]).squeeze(1)
        for k in range(n_blocks):
            kk = torch.full((n,), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            zk = _block_state_chunked(controller, xk)
            delta = (zk - z)[:, k, :]
            out["fm_pred"][:, k] = -(_fm_chunked(fm, z, kk)[:, k, :]
                                     - delta).pow(2).sum(-1).cpu().numpy()
            if fm_fresh is not None:
                out["fm_pred_fresh"][:, k] = -(_fm_chunked(fm_fresh, z, kk)[:, k, :]
                                               - delta).pow(2).sum(-1).cpu().numpy()
            out["value"][:, k] = (value(zk.mean(dim=1), roots) - v_now).cpu().numpy()
            out["belief"][:, k] = (
                controller.root_logits(zk.mean(dim=1)).log_softmax(-1)
                .gather(1, roots[:, None]).squeeze(1) - logp_now).cpu().numpy()
            if want_dp:
                out["dp"][:, k] = d_cur - dp_cost(layout, xk.cpu().numpy(), roots_np)
            if want_roll:
                acc = np.zeros(n)
                for _ in range(n_roll):
                    acc += rollout_terminal_success(controller, generator, layout, tb, xk,
                                                    roots, horizon=horizon, epsilon=epsilon,
                                                    render=render, gen=gen, device=device)
                out["roll"][:, k] = acc / n_roll
    return out


def _pair_stats(a, b):
    """How much do two graders disagree about moves, and how much does it cost?

      top1_disagree  fraction of states where the two argmaxes differ. The headline.
      rank_corr      mean per-state centred correlation of the two score vectors. +1 is
                     homogeneous-by-construction; ~0 or negative is a genuine second opinion.
      cost           mean over states of (b[argmax b] - b[argmax a]) / (max b - min b) -- how
                     much of grader b, in its own normalised units, grader a asks you to give up
                     at its own preferred move. This is the MAGNITUDE half; a disagreement rate
                     alone cannot tell a tie-break from a real conflict.
      pctile         mean percentile of a's argmax within b's own ranking. 1.0 = they agree,
                     0.5 = a's choice is median under b, 0.0 = a picks b's WORST move.
    """
    ia, ib = a.argmax(axis=1), b.argmax(axis=1)
    rows = np.arange(a.shape[0])
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    corr = np.where(den > 1e-12, (ac * bc).sum(1) / np.maximum(den, 1e-12), np.nan)
    span = np.maximum(b.max(1) - b.min(1), 1e-12)
    cost = (b[rows, ib] - b[rows, ia]) / span
    pctile = (b < b[rows, ia][:, None]).mean(axis=1)
    return {"top1_disagree": float((ia != ib).mean()),
            "rank_corr": float(np.nanmean(corr)),
            "cost": float(cost.mean()),
            "pctile": float(pctile.mean())}


def grader_disagreement(scores, pairs=None):
    """All requested ordered pairs of `move_scores` output, as {"a_vs_b": _pair_stats(a, b)}.

    The pairs that carry the argument:
      dp_vs_fm_pred          the external evaluative teacher's licence to disagree, exercised.
      roll_vs_fm_pred        the endogenous one's. IF THIS SITS AT THE HOMOGENEOUS FLOOR THE
                             TWO LOOPS HAVE COLLAPSED and any gain is an inner-loop gain.
      fm_pred_vs_fm_pred_fresh   the homogeneous floor itself (S1's seed-ensemble baseline).
      roll_vs_dp             how good the endogenous teacher is as an approximation of the
                             external one -- the quantity the headline lift should scale with.
      value_vs_roll          reported against paid. Divergence is the wireheading surface.
    """
    default = [("dp", "fm_pred"), ("roll", "fm_pred"), ("value", "fm_pred"),
               ("belief", "fm_pred"), ("fm_pred", "fm_pred_fresh"),
               ("dp", "fm_pred_fresh"), ("roll", "fm_pred_fresh"),
               ("roll", "dp"), ("value", "dp"), ("value", "roll"), ("belief", "dp")]
    out = {}
    for a, b in (pairs or default):
        if a in scores and b in scores:
            out[f"{a}_vs_{b}"] = _pair_stats(scores[a], scores[b])
    return out
