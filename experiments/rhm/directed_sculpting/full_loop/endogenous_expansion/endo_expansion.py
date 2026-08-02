"""Can an ENDOGENOUS evaluative grader expand? -- the target-source ladder.

`ideas/meta_learning_under_metered_data.md` states this node's central gap in its own
"What this does not establish":

    "`visits` is endogenous but was never tested for expansion -- `expansion.py` has no
     allocation policy, no oracle, and no `visits` tap at all. Conversely the DP `k*` teacher
     expands but is external and precomputed, and sculpt-continual says that is *why* it works
     ('so the loop can't wirehead its own value'). NO RUN IS BOTH ENDOGENOUS AND EXPANDING."

This is that run.


WHY VERSION (b) AND NOT (a)
---------------------------
Two experiments were on the table. (a) keep the DP `k*` teacher and let an endogenous tap
decide *where* `k*` is computed under a metered budget (sculpt-continual next-step #3).
(b) replace `k*` with an endogenous target, so nothing external enters the loop.

Built (b), for three reasons and one that decides it:

  1. (a) introduces no new grader TYPE. The expansion signal stays the DP; the endogenous
     signal only samples it. A positive would say "endogenous relevance is a decent sampler of
     an oracle" -- a venue-change on the ladder's already-published allocation result (§3,
     `visits_only` recovers 84% of the oracle prize), not a test of the type gap that
     `heterogeneous_graders` §4b is about.
  2. The idea doc's own prediction is that an endogenous tap "wired into the *expansion* setup
     reproduces some of the DP teacher's lift." That is only tested when the tap GENERATES the
     target.
  3. (a)'s readable regime has to be found first: at `ground_states=3072` the DP target is
     almost certainly not budget-bound, so metering it risks mjc-S2's measurement-subsidy null
     -- a calibration cost paid before any question gets asked.
  4. THE DECIDER: the wirehead hazard is the stated reason to prefer (a), and this substrate is
     the rare one where that hazard is *diagnosable*. The exact DP `d*` and `possible_set_success`
     exist, and can be held OUT of the loop and used as instruments. Edit-control's belief
     gaming (P(r*) -> 1 while ~99% off-grammar) was caught exactly this way. A hazard you can
     measure is a reason to instrument, not a reason to ask a smaller question.


THE DESIGN -- one variable: WHERE THE TARGET COMES FROM
--------------------------------------------------------
Every arm below is `expansion.py`'s `belief_update` at a matched budget. `frozen` and `dense`
are bit-compatible with the published 2x2. The four evaluative arms are IDENTICAL in loss,
optimizer, budget, and data -- they differ only in the vector `gkstar`. That is the whole
experiment.

  frozen      root-CE only.                                    [published anchor: PR 6.99->6.93]
  dense       + endogenous asymmetric FM local loss.           [published anchor: PR 6.31->6.61]
  evaluative  target = argmin_k d*(regen(x,k)).  EXTERNAL      [published anchor: PR 9.03->11.56]
              exact DP. The published ceiling.
  critic      target = argmax_k V(state(regen(x,k))).          ** THE QUESTION **
              ENDOGENOUS. `V` is the MC critic: trained only on terminal
              `possible_set_success` of paid rollouts, never shown a channel label, never shown
              `d*`. Refreshed every round from fresh rollouts under the CURRENT belief (policy
              iteration), and FROZEN while it grades -- sculpt-continual's `_reinternalize_belief`
              structure with the `k*` teacher deleted and nothing else changed.
              The next state is MATERIALIZED through the generator, not imagined through the FM,
              so FM hallucination is not a channel into the target.
  mirror      target = argmax_k V(z + FM(z,k)), i.e. the argmax of THE VERY LOGITS THE LOSS
              TRAINS, frozen at round start.  ** THE WIREHEAD CONTROL **
              `heterogeneous_graders` §8's "the same grader in a mirror", literally: a
              round-frozen self-teacher (the data2vec/EMA shape that caps at ~20% of the depth
              gap). If `critic` behaves like `mirror` rather than like `evaluative`, the
              endogenous signal is not functioning as a second grader.
  dp_noised_a target = the exact DP `k*`, corrupted to a UNIFORM-RANDOM block w.p. `p` chosen so
  dp_noised_g its top-1 agreement with `k*` matches `critic`'s; and the same corrupted to a
              random TREE block w.p. `p` chosen so its realised Dd* matches `critic`'s.
              ** THE ACCURACY CONTROLS, and the reason a null here is a claim **
              Without them, a `critic` null cannot distinguish "endogenous graders cannot expand"
              from "our endogenous grader is simply less accurate."

              There are TWO because calibration showed the two currencies do not match together:
              at equal top-1 agreement (0.26) the critic delivers 0.42 of the DP's realised gain
              while a uniformly-corrupted DP delivers only 0.27 -- the critic's errors are benign
              (it stays on the tree 95% of the time) and uniform corruption's are not. So no
              single control is "matched accuracy", and the honest move is to BRACKET the critic
              from both sides: `dp_noised_a` is worse than the critic in gain, `dp_noised_g` is
              better than it in agreement. Then:
                  both controls expand, critic does not -> the deficit is STRUCTURAL. Fidelity is
                      bracketed, so what kills it is that the endogenous target's errors are
                      correlated with the learner's own blind spots -- the mirror-grader claim,
                      measured.
                  neither control expands               -> the deficit is FIDELITY. Expansion needs
                      a target above some accuracy floor and endogeneity per se is not the axis.
                  the controls straddle                 -> the answer is quantitative, and the
                      bracket is the result.
                  critic expands                        -> the central gap closes.

Static world only. §5 measured drift ORTHOGONAL to all of this at 3 seeds (evaluative-frozen
DPR +4.63 +- 0.46 static vs +4.89 +- 0.73 drift; Dd4 +0.193 vs +0.187, statistically
identical), so the drift column buys nothing here and its compute buys four more target sources.


WHAT COUNTS AS "ENDOGENOUS" -- the line, stated before the run
---------------------------------------------------------------
  ALLOWED   terminal binary goal achievement (`possible_set_success`): the environment telling
            the agent whether it hit the goal it was given. This is reward. The agent PAYS for
            it in rollout transitions -- `adaptive_core` §7's "denominated in a currency the
            agent pays, not one it reports."
  FORBIDDEN the DP `d*`: an exact, dense, per-state, per-move optimal-action label computed by
            a solver holding the rule tables. This is an oracle.

`critic` and `mirror` touch only the allowed side. `evaluative` and the `dp_noised_*` controls
are on the forbidden side by construction, and are there as the ceiling and as the controls.


PRE-REGISTERED: WHAT A WIREHEADED SUCCESS LOOKS LIKE
-----------------------------------------------------
Four prior instances are on record (`fm_cotrain` caps; data2vec caps at ~20% of the depth gap;
edit-control's belief gaming; mjc #4e gaming loss-scale), so this is the primary hazard, not
noise. Declared in advance, a wireheaded positive is:

  RISES   belief PR (the headline)  ·  `plan_acc` (agreement with its OWN target)
          `belief_success` (the value's own sigmoid at the plan's terminal state)
  FLAT/   `policy_kstar_agree` (does the loop's own top-1 move actually equal the DP's?)
  FALLS   `policy_gain_frac` (the loop's realised Dd*, as a fraction of the DP's)
          d1..d4 on the FROZEN probe  ·  fresh-FM top-1  ·  ballistic (true task success)

  and the edit-control signature specifically: `wirehead_index = belief_success - ballistic`
  climbing while `ballistic` does not. That is `P(r*) -> 1 while ~99% off-grammar` in this
  loop's currency.

A real success moves PR AND d4 AND ballistic AND `policy_gain_frac` together.

Nothing on the "flat/falls" list ever enters a loss. `d*` is computed ONLY on a frozen 1024-state
diagnostic probe drawn once and never trained on, and only for arms that already pay for it
(`evaluative`, `dp_noised_*`) or as an instrument (all arms). The probe's exact-gain matrix is
world- and round-invariant (the generator is frozen and the world is static) so it is computed
once and shared -- a genuinely fixed instrument.


CALIBRATE -> MEASURE -> CALIBRATE
----------------------------------
`calibrate` runs the round-0 system and reports, for every candidate target source, its top-1
agreement with the DP and its realised Dd* against three anchors (DP `k*` = ceiling, uniform
random = floor, do-nothing = 0). It is what sets both `dp_noised_*` corruption rates, and it is what
makes a `critic` null interpretable: if the endogenous target agrees with the DP at chance, a
null says only that this critic is bad. Run it before the ladder.

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/endogenous_expansion/endo_expansion.py::calibrate --quick
  modal run rhm/directed_sculpting/full_loop/endogenous_expansion/endo_expansion.py::endo_expansion --quick
  modal run --detach rhm/directed_sculpting/full_loop/endogenous_expansion/endo_expansion.py::endo_expansion \
      --tag endo_s1 --seed 1
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import attach_drift, make_layout, sample_pool
from rhm.rhm_drift import make_drift_state, stationary_theta
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    _block_state_chunked, _fm_chunked, _imagine_plan, belief_update, build_block_tables,
    collect_grounded_moves, collect_value_buffer, fm_one_step_check, make_spec,
    make_tree_probe, refresh_w_blk, regenerate_block, root_ce, sample_states, set_world,
    snapshot_world, stochasticity_floor, train_block_fm, train_generator_channels,
    tree_depth_probe)


app = modal.App("rhm-ds-endo-expansion", image=image)

# `frozen`/`dense` reproduce the published 2x2's static rows; the other five are one
# `belief_update` with five different `gkstar` vectors.
CONDITIONS = ("frozen", "dense", "evaluative", "critic", "mirror",
              "dp_noised_a", "dp_noised_g")

# Which arms carry a target at all, and whether that target is allowed to see the DP.
TARGET_OF = {"evaluative": "dp", "critic": "critic", "mirror": "mirror",
             "dp_noised_a": "dp_noised_a", "dp_noised_g": "dp_noised_g"}
USES_ORACLE = {"dp", "dp_noised_a", "dp_noised_g"}


# --------------------------------------------------------------------------- #
# The target sources -- the single variable
# --------------------------------------------------------------------------- #

def critic_target(value_g, controller, generator, tb, x, roots, *, render, gen, device):
    """ENDOGENOUS evaluative target: argmax_k V(state(materialize(x, k))).

    `V` is the MC critic -- a sigmoid over terminal `possible_set_success`, trained on paid
    rollouts, never shown a channel label or a `d*`. It is FROZEN here: the belief update that
    consumes this target cannot move the thing producing it, which is the whole difference
    between a second grader and a mirror.

    The candidate next states are MATERIALIZED through the generator rather than imagined
    through the FM. That closes the obvious wirehead channel (an FM free to hallucinate the
    latent the value likes best), and it costs one real environment transition per candidate --
    the currency this arm pays in.
    """
    import torch
    n = x.shape[0]
    with torch.no_grad():
        sc = torch.empty(tb["n_blocks"], n, device=device)
        for k in range(tb["n_blocks"]):
            kk = torch.full((n,), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            sc[k] = value_g(_block_state_chunked(controller, xk).mean(dim=1), roots)
        return sc.argmax(dim=0)


def policy_top1(value, fm, controller, tb, x, roots, device):
    """The loop's OWN current best move: argmax_k V(z + FM(z, k)).

    Two jobs. As a READOUT it is the wirehead discriminator -- scored against the DP on the
    frozen probe, it says whether the loop's preference is tracking the world or only its own
    target. As the `mirror` arm's TARGET (frozen at round start) it is `heterogeneous_graders`
    §8's "same grader in a mirror" made literal: the loss's logits are exactly this quantity,
    so training toward it adds no information the loop did not already hold.
    """
    import torch
    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        sc = torch.stack([
            value((z + _fm_chunked(fm, z, torch.full((x.shape[0],), k, device=device,
                                                     dtype=torch.long))).mean(dim=1), roots)
            for k in range(tb["n_blocks"])], dim=0)
        return sc.argmax(dim=0)


def agree_match_p(agree_dp, agree_target, n_blocks):
    """Uniform-corruption rate that lands a DP target at top-1 agreement `agree_target`.

    Replacing with a uniform block w.p. `p` moves agreement linearly from `agree_dp` (the
    instrument's own ceiling -- not 1.0, because mixture rendering is stochastic, so a re-drawn
    DP disagrees with the frozen probe's `k*` a few percent of the time) toward the chance rate
    `1/n_blocks`. Solve the line.

    Errors are drawn INDEPENDENTLY of the state, which is exactly the point: `critic`'s errors
    are systematic -- it errs where its own value is miscalibrated -- and these are not. Matched
    rate, opposite error structure.
    """
    lo = 1.0 / n_blocks
    return float(np.clip((agree_dp - agree_target) / max(agree_dp - lo, 1e-9), 0.0, 1.0))


def gain_match_p(gain_dp, gain_corrupt, gain_target):
    """Corruption rate that lands a DP target at realised gain fraction `gain_target`.

    `E[gain](p) = (1-p) * gain_dp + p * gain_corrupt` exactly (the replacement is independent),
    so the solve is a line here too. Used with a random-TREE-block corruption, which keeps the
    error on-manifold and so degrades realised gain much more slowly per unit of disagreement
    than uniform corruption does.
    """
    return float(np.clip((gain_dp - gain_target) / max(gain_dp - gain_corrupt, 1e-9), 0.0, 1.0))


def corrupt_target(kstar, p, n_blocks, rng, pool=None):
    """Replace each entry w.p. `p` by a draw from `pool` (default: every block, uniform)."""
    out = np.array(kstar, copy=True)
    hit = rng.random(out.shape[0]) < p
    n = int(hit.sum())
    out[hit] = (rng.integers(0, n_blocks, n) if pool is None
                else np.asarray(pool)[rng.integers(0, len(pool), n)])
    return out


# --------------------------------------------------------------------------- #
# The frozen move probe -- the instrument, held out of every loss
# --------------------------------------------------------------------------- #

def make_move_probe(layout, tb, generator, x, roots, *, render, gen, device):
    """Exact Dd* for EVERY move on a held-out state set, computed once.

    The generator is frozen and the world is static, so this matrix does not depend on the
    round or the arm -- it is a fixed instrument, like `make_tree_probe`'s frozen ancestor
    labels. Distractor rows are filled analytically at gain 0 (P1, certified by
    `verify_backcompat` B5 / `check_structural_irrelevance`), which is exact, not an approximation.

    Returns the DP's own choice `kstar` (the ceiling), the gain matrix, and the two anchors a
    realised gain has to be read against: a uniform-random move, and doing nothing (0).
    """
    import torch
    from rhm.rhm_channels import dp_cost

    roots_np = roots.cpu().numpy()
    d_cur = dp_cost(layout, x.cpu().numpy(), roots_np)
    cand = np.repeat(d_cur[None, :], tb["n_blocks"], axis=0).astype(np.float64)
    with torch.no_grad():
        for k in tb["tree_blocks"].tolist():
            kk = torch.full((x.shape[0],), k, device=device, dtype=torch.long)
            xk = regenerate_block(generator, x, kk, tb, render=render, gen=gen)
            cand[k] = dp_cost(layout, xk.cpu().numpy(), roots_np)
    gain = d_cur[None, :] - cand                                    # (n_blocks, n)
    kstar = cand.argmin(axis=0)
    idx = np.arange(len(roots_np))
    return {"gain": gain, "kstar": kstar, "d_cur": d_cur, "idx": idx,
            "gain_kstar": float(gain[kstar, idx].mean()),
            "gain_random": float(gain.mean()),
            "chan": tb["chan_blk"].cpu().numpy(),
            "tree_channel": int(tb["tree_channel"])}


def score_moves(probe, k_chosen, n_blocks):
    """Score any move vector against the frozen probe. `gain_frac` normalises to the DP.

    `gain_frac = 1.0` is the exact-DP ceiling and `0.0` is doing nothing; a uniform-random move
    is reported separately because on this task it is NEGATIVE (a random regeneration usually
    makes things worse), so 0 is not the naive floor.
    """
    k = np.asarray(k_chosen)
    g = float(probe["gain"][k, probe["idx"]].mean())
    hist = np.bincount(k, minlength=n_blocks).astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    nz = hist[hist > 0]
    return {"kstar_agree": float((k == probe["kstar"]).mean()),
            "dstar_gain": g,
            "gain_frac": g / max(probe["gain_kstar"], 1e-9),
            "tree_share": float((probe["chan"][k] == probe["tree_channel"]).mean()),
            "entropy": float(-(nz * np.log(nz)).sum())}


# --------------------------------------------------------------------------- #
# Control, with the belief's own confidence read alongside the truth
# --------------------------------------------------------------------------- #

def beam_with_confidence(controller, generator, fm, value, tb, layout, x0, roots, *, budget,
                         beam_width, render, gen, device, chunk_len=3):
    """`open_loop_beam`, plus what the loop THINKS it achieved.

    Edit-control's failure was invisible to the loop and obvious to an outside grader: the
    belief scalar went to ~1.0 while ~99% of sequences went off-grammar. `belief_success` is
    that scalar in this loop's currency (the value's own sigmoid at the plan's terminal state)
    and `possible_set_success` is the truth. Their difference is the wirehead index.
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
        true_succ = float(possible_set_success(layout, x.cpu().numpy(),
                                               roots.cpu().numpy()).mean())
        believed = float(torch.sigmoid(
            value(_block_state_chunked(controller, x).mean(dim=1), roots)).mean())
    return {"ballistic_w16": true_succ, "belief_success": believed,
            "wirehead_index": believed - true_succ}


# --------------------------------------------------------------------------- #
# The representation-side readout, on trial
# --------------------------------------------------------------------------- #

def frontier_probe(controller, generator, fm, tb, x, roots, *, render, seed, device):
    """The residual decomposition of the belief, as a CANDIDATE expansion readout.

    Why this exists: PR -- which is `R_act`, the participation ratio of the belief's own
    covariance spectrum -- turned out to ANTI-correlate with target fidelity in the first pass
    of this ladder (the exact DP teacher produced the LOWEST dPR of the four evaluative arms,
    3/3 seeds). `rhm/residual_decomposition/` proposes a different instrument for the same
    question: `R_res_participation`, the participation ratio of the FRONTIER spectrum
    `{(1-rho_i) lam_i}`, which counts how many of the model's OWN working directions still
    carry unexplained computation. It is a genuinely different quantity from `R_act` -- it needs
    a paired forward model to define `rho` -- so it is worth putting on trial.

    Two things that must be carried, both from the source nodes:

      * `R_res_participation` IS NOT MONOTONE IN MODEL QUALITY (`mjc/expansion`'s own
        calibration). A model that explains nothing leaves the frontier equal to the activation
        spectrum, so its count lands at `R_act`; a model that absorbs the loud directions and
        leaves the quiet ones FLATTENS what remains and pushes the count ABOVE `R_act`. So it
        cannot be read as "more is more expansion" without a calibration -- which is exactly
        what this ladder supplies, since the arms come pre-ordered by target fidelity and by
        three functional readouts.
      * beta is expected to be UNUSABLE here. `heterogeneous_graders` §4b records that beta is
        unusable below ~100 directions and this belief is 96-dimensional. It is reported anyway,
        as a check on that threshold rather than as a readout.

    The absorber is the FRESH FM -- independently trained per arm on the current belief -- not
    the co-trained one. Using the co-trained FM would confound `rho` with arm-specific FM
    quality, which is a difference between the arms rather than a property of the belief.

    Runs on its OWN torch Generator so that adding this readout cannot perturb the main RNG
    stream: the committed `endo_s*` results stay bit-reproducible.
    """
    import torch
    from rhm.residual_decomposition.analyze_summaries import shadow_law
    from rhm.residual_decomposition.decomposition import full_decomposition

    g2 = torch.Generator(device=device).manual_seed(seed)
    n = x.shape[0]
    with torch.no_grad():
        z = _block_state_chunked(controller, x)
        k = torch.randint(0, tb["n_blocks"], (n,), device=device, generator=g2)
        x2 = regenerate_block(generator, x, k, tb, render=render, gen=g2)
        tz = _block_state_chunked(controller, x2)              # (n, n_blocks, D) -- the truth
        pz = z + _fm_chunked(fm, z, k)                         # (n, n_blocks, D) -- the forecast
    A = tz.reshape(-1, tz.shape[-1]).cpu().numpy()
    P = pz.reshape(-1, pz.shape[-1]).cpu().numpy()
    # tree-only as well: the distractor channels are irreducible by construction (P1), so they
    # pad the frontier with junk no model could ever absorb. Which of the two tracks function is
    # itself part of what this trial measures.
    tb0, tb1 = int(tb["tree_blocks"][0]), int(tb["tree_blocks"][-1]) + 1
    At = tz[:, tb0:tb1].reshape(-1, tz.shape[-1]).cpu().numpy()
    Pt = pz[:, tb0:tb1].reshape(-1, pz.shape[-1]).cpu().numpy()

    out = {}
    for name, (a, p) in (("all", (A, P)), ("tree", (At, Pt))):
        d = full_decomposition(a, p, seed=seed)
        beta, beta_r2 = shadow_law(d["geometry"])   # beta is FITTED, not returned by geometry
        g = d["geometry"]
        out[name] = {
            "basic": d["basic"], "naive": d["naive"], "repaired": d["repaired"],
            "beta": beta, "beta_r2": beta_r2,
            "alignment_index": g["alignment_index"],
            "absorption_mean": g["absorption_mean"],
            "absorption_step_excess": g["absorption_step_excess"],
        }
    return out


# --------------------------------------------------------------------------- #
# The critic refresh -- the currency this loop pays in
# --------------------------------------------------------------------------- #

def refresh_critic(value_g, controller, generator, layout, tb, *, n_episodes, n_steps,
                   batch_size, n_corrupt, budget, epsilon, seed, render, gen, device):
    """Re-derive the MC critic from fresh paid rollouts under the CURRENT belief.

    Policy iteration, exactly as sculpt-continual's wake phase: the behaviour policy is
    controller-greedy with eps exploration over ALL blocks (distractors included, so whether the
    critic devalues them stays measured rather than assumed), and every visited state is
    labelled by its rollout's terminal `possible_set_success`.

    Start states stay broad -- corruptions of a fresh random pool, not the loop's own visited
    distribution. That is deliberate: sculpt-continual flags a policy-following buffer as the
    thing that "could strengthen the ratchet -- or collapse it", and a collapsing target
    distribution would confound the question this run is asking. `target_entropy` in the
    per-round record is the readout that would catch it anyway.

    Warm-started, so the critic accumulates across rounds instead of being re-fit from cold on a
    shifting latent space. Its labels are ground truth every round, so warm-starting cannot
    drift it off the outcome.
    """
    from rhm.rhm_latent_planner import _train_value_mc

    cfg, rts, suc = collect_value_buffer(
        controller, generator, layout, tb, n_episodes=n_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=budget, epsilon=epsilon, seed=seed, render=render,
        gen=gen, device=device)
    _train_value_mc(value_g, controller, cfg, rts, suc, batch_size=512, n_steps=n_steps,
                    lr=3e-4, device=device)
    base = float(suc.mean())
    del cfg, rts, suc
    # transitions actually spent: every episode walks `budget` steps, and each step evaluates
    # all `n_blocks` candidate regenerations.
    return {"rollout_transitions": int(n_episodes * budget * tb["n_blocks"]),
            "behaviour_success": base}


# --------------------------------------------------------------------------- #
# Shared setup: one round-0 system every arm forks from
# --------------------------------------------------------------------------- #

def _setup(*, v, s, seed, state_dim, tree_depth, struct_depths, struct_ms, noise_blocks,
           edit_budget, n_corrupt, controller_steps, generator_steps, value_steps,
           value_episodes, fm_warm_steps, batch_size, explore_eps, n_eval, n_diag,
           n_novel, drift_kappa, drift_kl, drift_steps_per_round, quick, device):
    """Everything that must be identical across arms, built once."""
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    spec = make_spec(tree_depth=tree_depth, struct_depths=struct_depths,
                     struct_ms=struct_ms, noise_blocks=noise_blocks)
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    T, n_blocks, L = layout["total_len"], tb["n_blocks"], tree_depth
    render = "mixture"
    gen = torch.Generator(device=device).manual_seed(seed)
    surface = L - 1

    # The world never moves in this run (static only, see the header), but W0 is still drawn
    # from the drift walk's own stationary law rather than parked at uniform -- the published
    # `entropy_matched_base` fix. Uniform is the simplex's entropy MAXIMUM, so a uniform-anchored
    # world has systematically more synonyms in play per edit and is ~18% harder to plan in.
    # Keeping the published anchor's base world is what makes `frozen`/`dense`/`evaluative`
    # comparable to the published numbers at all.
    attach_drift(layout, "tree", [surface], drift_kappa, drift_kl, seed=seed + 601,
                 calibrate="event", event_steps=drift_steps_per_round, init="stationary",
                 frozen=True)
    sigma = dict(layout["tree"]["drift_calibrated_sigma"])
    theta0 = stationary_theta(layout["tree"]["rules"], [surface], drift_kappa, sigma,
                              seed=seed + 602)
    layout["tree"]["drift"]["theta"] = theta0
    refresh_w_blk(tb, layout, device)
    W0 = snapshot_world(tb)

    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, BlockFM = _build_value_head(), _build_block_fm()

    pool = sample_pool(layout, 100_000 if not quick else 20_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))

    controller0 = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller0, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, train_leaves, tb, batch_size=batch_size,
                             n_steps=generator_steps, lr=3e-4, device=device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    print(f"Collecting value data ({value_episodes} rollouts)")
    cfg, rts, suc = collect_value_buffer(
        controller0, generator, layout, tb, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render=render, gen=gen, device=device)
    value0 = ValueHead(state_dim, v).to(device)
    _train_value_mc(value0, controller0, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    print(f"  value buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    setup_transitions = int(value_episodes * edit_budget * n_blocks)
    del cfg, rts, suc

    fm0 = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    train_block_fm(fm0, controller0, generator, layout, tb, n_steps=fm_warm_steps,
                   batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                   lr=1e-3, seed=seed + 51, render=render, gen=gen, device=device)

    x_eval, r_eval = sample_states(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 42, device=device, render=render,
                                   gen=gen)
    probe_leaves, level_feats, _ = make_tree_probe(layout, tb, seed + 43)

    # Grading worlds. `own` == `base` by construction in a static run (the world never moves)
    # and is kept only so the column set matches the published 2x2; the informative columns are
    # the two held-out `novel` realisations, which no arm ever saw and which differ from each
    # other -- a genuine cross-world spread rather than a single held-out draw.
    worlds = {"base": W0}
    for j in range(n_novel):
        _th = stationary_theta(layout["tree"]["rules"], [surface], drift_kappa, sigma,
                               seed=seed + 9001 + 17 * j)
        layout["tree"]["drift"] = make_drift_state(layout["tree"]["rules"], [surface], None,
                                                   seed=seed + 9001 + 17 * j, theta0=_th)
        layout["tree"]["drift_sigma"] = {surface: 0.0}
        layout["tree"]["drift_kappa"] = 0.0
        refresh_w_blk(tb, layout, device)
        worlds[f"novel{j + 1}"] = snapshot_world(tb)
    # restore the LAYOUT to W0, not just `tb`: `sample_states` draws derivations through
    # `layout[...]["drift"]`, so a stale novel state would sample from one world and render
    # edits in another.
    layout["tree"]["drift"] = make_drift_state(layout["tree"]["rules"], [surface], None,
                                               seed=seed + 601, theta0=theta0)
    layout["tree"]["drift_sigma"], layout["tree"]["drift_kappa"] = {surface: 0.0}, 0.0
    set_world(tb, W0)

    x_floor, _ = sample_states(layout, tb, generator, n=512, n_corrupt=n_corrupt,
                               presteps=edit_budget // 2, seed=seed + 44, device=device,
                               render=render, gen=gen)
    for wn, W in worlds.items():
        set_world(tb, W)
        blks = tb["tree_blocks"]
        kk = blks[torch.randint(0, len(blks), (x_floor.shape[0],), device=device)]
        print(f"  world '{wn}': tree stochasticity floor = "
              f"{stochasticity_floor(controller0, generator, tb, x_floor, kk, render=render, gen=gen, acted_only=True):.4f}")
    set_world(tb, W0)

    # the frozen move probe: held-out states, never trained on, exact gains for every move
    dx, dr = sample_states(layout, tb, generator, n=n_diag, n_corrupt=n_corrupt,
                           presteps=edit_budget // 2, seed=seed + 45, device=device,
                           render=render, gen=gen)
    probe = make_move_probe(layout, tb, generator, dx, dr, render=render, gen=gen,
                            device=device)
    # The instrument's own ceiling. Re-running the exact DP on the probe states does NOT return
    # the probe's frozen `k*` exactly, because mixture rendering is stochastic: `regen(x, k)`
    # draws a synonym, so two exact solvers can prefer different blocks on the same state. That
    # self-disagreement is the noise floor of `kstar_agree`, it is a few percent, and every
    # `kstar_agree` in this run has to be read against it rather than against 1.0. `gain_frac`
    # is the robust readout for exactly this reason and is the one the controls are matched on.
    kstar_redraw, _, _ = collect_grounded_moves(layout, tb, generator, dx, dr, render=render,
                                                gen=gen, device=device)
    probe["kstar_redraw"] = kstar_redraw.cpu().numpy()
    probe["gain_treerandom"] = float(probe["gain"][tb["tree_blocks"].cpu().numpy()].mean())
    dp_self = score_moves(probe, probe["kstar_redraw"], tb["n_blocks"])
    print(f"  move probe: n={n_diag}  d*_cur={probe['d_cur'].mean():.3f}  "
          f"DP gain={probe['gain_kstar']:.3f}  random-move gain={probe['gain_random']:.3f}  "
          f"tree-random gain={probe['gain_treerandom']:.3f}")
    print(f"  instrument ceiling (exact DP re-drawn vs the frozen probe): "
          f"k*agree={dp_self['kstar_agree']:.3f}  gain_frac={dp_self['gain_frac']:.3f}")

    return dict(layout=layout, tb=tb, T=T, n_blocks=n_blocks, L=L, render=render, gen=gen,
                surface=surface, sigma=sigma, theta0=theta0, W0=W0, worlds=worlds,
                controller0=controller0, generator=generator, value0=value0, fm0=fm0,
                BlockFM=BlockFM, train_leaves=train_leaves, train_roots=train_roots,
                x_eval=x_eval, r_eval=r_eval, x_floor=x_floor, dx=dx, dr=dr, probe=probe,
                dp_self=dp_self, probe_leaves=probe_leaves, level_feats=level_feats,
                setup_transitions=setup_transitions)


# --------------------------------------------------------------------------- #
# Calibration -- run this BEFORE the ladder
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=14400, memory=32768)
def calibrate(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
    noise_blocks: str = "1,1", edit_budget: int = 6, n_corrupt: int = 3,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 8_000, batch_size: int = 256,
    explore_eps: float = 0.3, n_eval: int = 1024, n_diag: int = 1024, n_novel: int = 2,
    drift_kappa: float = 0.05, drift_kl: float = 0.60, drift_steps_per_round: int = 20,
    tag: str = "cal_v1", quick: bool = False,
):
    """How good is each candidate target, at round 0, against the exact DP?

    This is the gate. If the endogenous target agrees with the DP at chance, a null in the
    ladder says only that this critic is bad; if it agrees well, a null says something about
    endogeneity. It also fixes `dp_noised`'s corruption rate, which is the control that makes
    the ladder's result a claim rather than an anecdote.
    """
    import torch

    if quick:
        controller_steps = generator_steps = value_steps = fm_warm_steps = 800
        value_episodes, n_eval, n_diag = 6_000, 128, 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    S = _setup(v=v, s=s, seed=seed, state_dim=state_dim, tree_depth=tree_depth,
               struct_depths=[int(x) for x in struct_depths.split(",")],
               struct_ms=[int(x) for x in struct_ms.split(",")],
               noise_blocks=[int(x) for x in noise_blocks.split(",")],
               edit_budget=edit_budget, n_corrupt=n_corrupt,
               controller_steps=controller_steps, generator_steps=generator_steps,
               value_steps=value_steps, value_episodes=value_episodes,
               fm_warm_steps=fm_warm_steps, batch_size=batch_size, explore_eps=explore_eps,
               n_eval=n_eval, n_diag=n_diag, n_novel=n_novel, drift_kappa=drift_kappa,
               drift_kl=drift_kl, drift_steps_per_round=drift_steps_per_round,
               quick=quick, device=device)

    probe, nb = S["probe"], S["n_blocks"]
    tree_blocks = S["tb"]["tree_blocks"].cpu().numpy()
    rng = np.random.default_rng(seed + 4242)
    sources = {}

    kstar = probe["kstar_redraw"]
    sources["dp"] = dict(S["dp_self"])
    sources["critic"] = score_moves(probe, critic_target(
        S["value0"], S["controller0"], S["generator"], S["tb"], S["dx"], S["dr"],
        render=S["render"], gen=S["gen"], device=device).cpu().numpy(), nb)
    sources["mirror"] = score_moves(probe, policy_top1(
        S["value0"], S["fm0"], S["controller0"], S["tb"], S["dx"], S["dr"],
        device=device).cpu().numpy(), nb)
    sources["random"] = score_moves(probe, rng.integers(0, nb, n_diag), nb)
    sources["tree_random"] = score_moves(
        probe, tree_blocks[rng.integers(0, len(tree_blocks), n_diag)], nb)

    pa = agree_match_p(sources["dp"]["kstar_agree"], sources["critic"]["kstar_agree"], nb)
    pg = gain_match_p(sources["dp"]["gain_frac"],
                      probe["gain_treerandom"] / max(probe["gain_kstar"], 1e-9),
                      sources["critic"]["gain_frac"])
    sources["dp_noised_a"] = score_moves(probe, corrupt_target(kstar, pa, nb, rng), nb)
    sources["dp_noised_g"] = score_moves(
        probe, corrupt_target(kstar, pg, nb, rng, pool=tree_blocks), nb)

    print(f"\n{'=' * 84}\n=== CALIBRATION: how good is each target source, against the exact "
          f"DP? ===\n{'=' * 84}")
    print(f"  probe: n={n_diag}  mean d*={probe['d_cur'].mean():.3f}  "
          f"DP gain={probe['gain_kstar']:.4f}  (do-nothing = 0.0)")
    print(f"  {'source':<14}{'k*agree':>9}{'Dd*':>9}{'gain_frac':>11}"
          f"{'tree%':>8}{'H(k)':>7}")
    for name, m in sources.items():
        print(f"  {name:<14}{m['kstar_agree']:>9.3f}{m['dstar_gain']:>9.4f}"
              f"{m['gain_frac']:>11.3f}{m['tree_share']:>8.3f}{m['entropy']:>7.2f}")
    print(f"\n  the two controls bracket the critic:")
    print(f"    dp_noised_a (uniform corruption, AGREEMENT-matched): p={pa:.4f}")
    print(f"    dp_noised_g (tree corruption,    GAIN-matched)     : p={pg:.4f}")
    print(f"  -> pass --noise-match-acc {sources['critic']['kstar_agree']:.6f} "
          f"--noise-match-gain {sources['critic']['gain_frac']:.6f} to the ladder")

    out = {"config": {"seed": seed, "n_diag": n_diag, "n_blocks": nb, "quick": quick},
           "probe": {"d_cur_mean": float(probe["d_cur"].mean()),
                     "gain_kstar": probe["gain_kstar"],
                     "gain_random": probe["gain_random"],
                     "gain_treerandom": probe["gain_treerandom"]},
           "instrument_ceiling": S["dp_self"],
           "sources": sources, "p_agree": pa, "p_gain": pg,
           "critic_kstar_agree": sources["critic"]["kstar_agree"],
           "critic_gain_frac": sources["critic"]["gain_frac"],
           "elapsed_seconds": time.time() - started}
    out_dir = f"{DATA_DIR}/directed_sculpting/endo_calibrate_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({out['elapsed_seconds']:.0f}s)")
    return out


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=28800, memory=32768)
def endo_expansion(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
    noise_blocks: str = "1,1", edit_budget: int = 6, n_corrupt: int = 3,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 8_000, batch_size: int = 256,
    explore_eps: float = 0.3, rounds: int = 10, belief_steps: int = 1500,
    ground_states: int = 3072, anchor_pool: int = 40_000,
    lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0,
    drift_kappa: float = 0.05, drift_kl: float = 0.60, drift_steps_per_round: int = 20,
    n_eval: int = 1024, n_diag: int = 1024, n_novel: int = 2, n_probe: int = 2000,
    probe_steps: int = 250, fresh_fm_steps: int = 5000, grade_every: int = 2,
    ballistic_chunk: int = 3,
    critic_refresh_episodes: int = 12_000, critic_refresh_steps: int = 4_000,
    noise_match_acc: float = -1.0, noise_match_gain: float = -1.0,
    conditions: str = ",".join(CONDITIONS), tag: str = "v1", quick: bool = False,
):
    """Six arms, one variable: where the belief update's target comes from."""
    import torch

    if quick:
        controller_steps = generator_steps = value_steps = fm_warm_steps = 800
        value_episodes, rounds, belief_steps = 6_000, 3, 250
        ground_states, anchor_pool, n_eval, n_diag = 512, 8_000, 128, 384
        n_probe, probe_steps, fresh_fm_steps, grade_every = 600, 80, 400, 2
        critic_refresh_episodes, critic_refresh_steps = 2_000, 400

    cond_list = [c for c in conditions.split(",") if c]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    S = _setup(v=v, s=s, seed=seed, state_dim=state_dim, tree_depth=tree_depth,
               struct_depths=[int(x) for x in struct_depths.split(",")],
               struct_ms=[int(x) for x in struct_ms.split(",")],
               noise_blocks=[int(x) for x in noise_blocks.split(",")],
               edit_budget=edit_budget, n_corrupt=n_corrupt,
               controller_steps=controller_steps, generator_steps=generator_steps,
               value_steps=value_steps, value_episodes=value_episodes,
               fm_warm_steps=fm_warm_steps, batch_size=batch_size, explore_eps=explore_eps,
               n_eval=n_eval, n_diag=n_diag, n_novel=n_novel, drift_kappa=drift_kappa,
               drift_kl=drift_kl, drift_steps_per_round=drift_steps_per_round,
               quick=quick, device=device)
    nb, L = S["n_blocks"], S["L"]

    # Both corruption rates are fixed ONCE, from the ROUND-0 critic (identical across arms,
    # since every arm forks from the same round-0 system), and held constant for the whole run.
    # The control is therefore a constant of the run rather than something that moves with the
    # arm it is controlling for -- and the critic arm's own per-round `target_diag` is logged so
    # any drift away from the match is visible rather than absorbed.
    c0 = score_moves(S["probe"], critic_target(
        S["value0"], S["controller0"], S["generator"], S["tb"], S["dx"], S["dr"],
        render=S["render"], gen=S["gen"], device=device).cpu().numpy(), nb)
    a0 = c0["kstar_agree"] if noise_match_acc < 0.0 else float(noise_match_acc)
    g0 = c0["gain_frac"] if noise_match_gain < 0.0 else float(noise_match_gain)
    p_agree = agree_match_p(S["dp_self"]["kstar_agree"], a0, nb)
    p_gain = gain_match_p(S["dp_self"]["gain_frac"],
                          S["probe"]["gain_treerandom"] / max(S["probe"]["gain_kstar"], 1e-9),
                          g0)
    print(f"\nround-0 critic vs the exact DP: k*agree={a0:.4f}  gain_frac={g0:.4f}"
          f"   (instrument ceiling k*agree={S['dp_self']['kstar_agree']:.4f})")
    print(f"  dp_noised_a (uniform corruption, agreement-matched): p={p_agree:.4f}")
    print(f"  dp_noised_g (tree corruption,    gain-matched)     : p={p_gain:.4f}")

    set_world(S["tb"], S["W0"])
    depth0 = tree_depth_probe(S["controller0"], S["probe_leaves"], S["level_feats"],
                              S["layout"], S["tb"], device, probe_steps=probe_steps)
    print("Round-0 belief: " + " ".join(f"{k}={val:.3f}" for k, val in depth0.items()))

    results = {}
    for cond in cond_list:
        print(f"\n{'#' * 74}\n# CONDITION: {cond}   (target = "
              f"{TARGET_OF.get(cond, 'none')})\n{'#' * 74}")
        results[cond] = _run_condition(
            cond, S, seed=seed, state_dim=state_dim, rounds=rounds,
            belief_steps=belief_steps, ground_states=ground_states, anchor_pool=anchor_pool,
            batch_size=batch_size, n_corrupt=n_corrupt, edit_budget=edit_budget,
            lam_fm=lam_fm, lam_plan=lam_plan, tau=tau, probe_steps=probe_steps,
            fresh_fm_steps=fresh_fm_steps, grade_every=grade_every,
            ballistic_chunk=ballistic_chunk, explore_eps=explore_eps,
            critic_refresh_episodes=critic_refresh_episodes,
            critic_refresh_steps=critic_refresh_steps, p_agree=p_agree, p_gain=p_gain,
            device=device)

    _summarise(results, cond_list, depth0, L)

    metrics = {
        "config": {"v": v, "s": s, "tree_depth": tree_depth, "total_len": S["T"],
                   "n_blocks": nb, "edit_budget": edit_budget, "rounds": rounds,
                   "belief_steps": belief_steps, "ground_states": ground_states,
                   "conditions": cond_list, "render": S["render"], "seed": seed,
                   "world": "static", "n_diag": n_diag, "n_novel": n_novel,
                   "critic_refresh_episodes": critic_refresh_episodes,
                   "critic_refresh_steps": critic_refresh_steps,
                   "noise_match_acc": a0, "noise_match_gain": g0,
                   "p_agree": p_agree, "p_gain": p_gain,
                   "setup_rollout_transitions": S["setup_transitions"]},
        "belief_round0": depth0, "critic_round0": c0,
        "instrument_ceiling": S["dp_self"],
        "move_probe": {"d_cur_mean": float(S["probe"]["d_cur"].mean()),
                       "gain_kstar": S["probe"]["gain_kstar"],
                       "gain_random": S["probe"]["gain_random"],
                       "gain_treerandom": S["probe"]["gain_treerandom"]},
        "results": results, "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/endo_expansion_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


def _run_condition(cond, S, *, seed, state_dim, rounds, belief_steps, ground_states,
                   anchor_pool, batch_size, n_corrupt, edit_budget, lam_fm, lam_plan, tau,
                   probe_steps, fresh_fm_steps, grade_every, ballistic_chunk, explore_eps,
                   critic_refresh_episodes, critic_refresh_steps, p_agree, p_gain, device):
    """One arm. Forks from the SAME controller/FM/value at a matched belief-update budget;
    only `gkstar` differs between the four evaluative arms."""
    import torch

    layout, tb, generator = S["layout"], S["tb"], S["generator"]
    render = S["render"]
    probe, nb = S["probe"], S["n_blocks"]
    target = TARGET_OF.get(cond)
    mode = "evaluative" if target else cond

    controller = copy.deepcopy(S["controller0"])
    for p in controller.parameters():
        p.requires_grad_(True)
    fm = copy.deepcopy(S["fm0"])
    value = copy.deepcopy(S["value0"])
    for mod in (fm, value):
        for p in mod.parameters():
            p.requires_grad_(True)
    # the GRADER's critic is a separate object from the one inside the loss, so the belief
    # update cannot move the thing that is grading it. This is the difference between `critic`
    # and `mirror`, and it is why they are worth running side by side.
    value_g = copy.deepcopy(S["value0"]) if target == "critic" else None

    set_world(tb, S["W0"])
    gen = torch.Generator(device=device).manual_seed(seed + 777)
    rng = np.random.default_rng(seed + 909)
    rows = []
    paid = {"rollout_transitions": 0, "dp_state_move_evals": 0,
            "materialisations": 0}

    for rnd in range(1, rounds + 1):
        anchor = sample_pool(layout, anchor_pool, seed + 6000 + rnd)
        anchor_leaves = torch.from_numpy(anchor["leaves"])
        anchor_roots = torch.from_numpy(anchor["roots"].astype(np.int64))
        gx, gr = sample_states(layout, tb, generator, n=ground_states, n_corrupt=n_corrupt,
                               presteps=edit_budget // 2, seed=seed + 7000 + rnd,
                               device=device, render=render, gen=gen)

        gk = None
        refresh_info = {}
        if target == "critic":
            refresh_info = refresh_critic(
                value_g, controller, generator, layout, tb,
                n_episodes=critic_refresh_episodes, n_steps=critic_refresh_steps,
                batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                epsilon=explore_eps, seed=seed + 31_000 + rnd, render=render, gen=gen,
                device=device)
            paid["rollout_transitions"] += refresh_info["rollout_transitions"]
            # `value_g` is never in any optimizer the belief update builds, so it cannot be
            # moved by the loss it grades -- eval() is the only freeze needed, and leaving
            # requires_grad alone keeps the next round's refresh trainable.
            value_g.eval()
            gk = critic_target(value_g, controller, generator, tb, gx, gr, render=render,
                               gen=gen, device=device)
            paid["materialisations"] += ground_states * nb
        elif target == "mirror":
            gk = policy_top1(value, fm, controller, tb, gx, gr, device=device)
        elif target is not None:
            gk, _, _ = collect_grounded_moves(layout, tb, generator, gx, gr, render=render,
                                              gen=gen, device=device)
            # only the tree candidates are actually DP'd; distractors are filled analytically
            paid["dp_state_move_evals"] += ground_states * (len(tb["tree_blocks"]) + 1)
            paid["materialisations"] += ground_states * len(tb["tree_blocks"])
            if target == "dp_noised_a":
                gk = torch.from_numpy(corrupt_target(
                    gk.cpu().numpy(), p_agree, nb, rng)).to(device)
            elif target == "dp_noised_g":
                gk = torch.from_numpy(corrupt_target(
                    gk.cpu().numpy(), p_gain, nb, rng,
                    pool=tb["tree_blocks"].cpu().numpy())).to(device)

        info = belief_update(mode, controller, fm, value, generator, layout, tb,
                             train_leaves=anchor_leaves, train_roots=anchor_roots,
                             gstates=gx, groots=gr, gkstar=gk, n_steps=belief_steps,
                             batch_size=batch_size, lr=3e-4, lam_fm=lam_fm,
                             lam_plan=lam_plan, tau=tau, n_corrupt=n_corrupt,
                             edit_budget=edit_budget, render=render, gen=gen, device=device,
                             seed=seed + rnd)

        depth = tree_depth_probe(controller, S["probe_leaves"], S["level_feats"], layout, tb,
                                 device, probe_steps=probe_steps)
        row = {"round": rnd, "depth": depth,
               "root_ce": root_ce(controller, anchor_leaves[:4096], anchor_roots[:4096],
                                  device),
               "root_acc": info.get("root_acc"), "plan_acc": info.get("plan_acc"),
               "paid": dict(paid), **({"critic_refresh": refresh_info} if refresh_info else {})}

        # -- the instrument, on the frozen probe, never in any loss ------------------------
        # `policy` is the wirehead discriminator: `plan_acc` says the loop agrees with its own
        # target, `policy_kstar_agree` says whether that preference tracks the world.
        row["policy"] = score_moves(probe, policy_top1(value, fm, controller, tb, S["dx"],
                                                       S["dr"], device=device).cpu().numpy(),
                                    nb)
        if target == "critic":
            row["target_diag"] = score_moves(probe, critic_target(
                value_g, controller, generator, tb, S["dx"], S["dr"], render=render, gen=gen,
                device=device).cpu().numpy(), nb)
        elif target == "mirror":
            row["target_diag"] = dict(row["policy"])
        elif target is not None:
            # the DP arms' target does not depend on the model, so it is scored from the
            # RE-DRAWN probe `k*` (not the frozen one) -- i.e. against the same instrument
            # ceiling the endogenous arms are scored against, corrupted the same way the arm's
            # in-loop target is.
            drng = np.random.default_rng(seed + 5150 + rnd)
            row["target_diag"] = score_moves(probe, corrupt_target(
                probe["kstar_redraw"],
                {"dp": 0.0, "dp_noised_a": p_agree, "dp_noised_g": p_gain}[target], nb, drng,
                pool=(tb["tree_blocks"].cpu().numpy() if target == "dp_noised_g" else None)),
                nb)

        if rnd % grade_every == 0 or rnd == rounds:
            # FRESH, independent consumer: a value+FM trained from scratch on the (possibly
            # reorganised) belief, so a lifted beam is TRANSFERABLE plannability rather than the
            # co-trained pair getting better at each other.
            fresh = S["BlockFM"](state_dim, nb, n_head=4, n_layer=2).to(device)
            train_block_fm(fresh, controller, generator, layout, tb, n_steps=fresh_fm_steps,
                           batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                           lr=1e-3, seed=seed + 8000 + rnd, render=render, gen=gen,
                           device=device)
            own = snapshot_world(tb)
            row["xworld"] = {}
            for wn, W in list(S["worlds"].items()) + [("own", own)]:
                set_world(tb, W)
                blks = tb["tree_blocks"]
                kk = blks[torch.randint(0, len(blks), (S["x_floor"].shape[0],),
                                        device=device)]
                cell = beam_with_confidence(controller, generator, fresh, value, tb, layout,
                                            S["x_eval"], S["r_eval"], budget=edit_budget,
                                            beam_width=16, render=render, gen=gen,
                                            device=device, chunk_len=ballistic_chunk)
                cell["fm"] = fm_one_step_check(fresh, value, controller, generator, tb,
                                               S["x_eval"], S["r_eval"], render=render,
                                               gen=gen, device=device)
                cell["tree_floor"] = stochasticity_floor(controller, generator, tb,
                                                         S["x_floor"], kk, render=render,
                                                         gen=gen, acted_only=True)
                row["xworld"][wn] = cell
            set_world(tb, S["W0"])
            row["frontier"] = frontier_probe(controller, generator, fresh, tb, S["x_eval"],
                                             S["r_eval"], render=render,
                                             seed=seed + 31_337 + rnd, device=device)
            set_world(tb, own)
            row["fresh_fm"] = row["xworld"]["base"]["fm"]
            row["ballistic_w16"] = row["xworld"]["base"]["ballistic_w16"]
            row["belief_success"] = row["xworld"]["base"]["belief_success"]
            row["wirehead_index"] = row["xworld"]["base"]["wirehead_index"]
            del fresh

        rows.append(row)
        msg = (f"  r{rnd:02d} PR={depth['PR']:.2f} "
               + " ".join(f"{k}={val:.3f}" for k, val in depth.items() if k != "PR")
               + f" rootCE={row['root_ce']:.4f}"
               + f" pol_k*={row['policy']['kstar_agree']:.3f}"
               + f" pol_gain={row['policy']['gain_frac']:.3f}")
        if "target_diag" in row:
            msg += f" tgt_k*={row['target_diag']['kstar_agree']:.3f}"
        if "ballistic_w16" in row:
            msg += (f" ballistic={row['ballistic_w16']:.3f}"
                    f" believed={row['belief_success']:.3f}")
        print(msg)
    return {"rounds": rows, "paid": paid, "target": target,
            "uses_oracle": target in USES_ORACLE}


def _summarise(results, cond_list, depth0, L):
    print(f"\n{'=' * 90}\n=== SUMMARY -- can an endogenous evaluative grader expand? "
          f"===\n{'=' * 90}")
    print(f"round-0 belief: PR={depth0['PR']:.2f}  "
          + " ".join(f"{k}={depth0[k]:.3f}" for k in depth0 if k != "PR"))
    deep = f"d{L}"
    hdr = (f"\n  {'arm':<12}{'PR r1':>8}{'PR rN':>8}{'dPR':>8}{deep + ' r1':>9}"
           f"{deep + ' rN':>9}{'ballist':>9}{'believed':>10}{'wire':>8}"
           f"{'freshTop1':>11}{'pol_k*':>8}{'pol_gain':>10}")
    print(hdr)
    for cond in cond_list:
        rr = results[cond]["rounds"]
        g = [r for r in rr if "ballistic_w16" in r]
        a, b = rr[0], rr[-1]
        print(f"  {cond:<12}{a['depth']['PR']:>8.2f}{b['depth']['PR']:>8.2f}"
              f"{b['depth']['PR'] - a['depth']['PR']:>8.2f}"
              f"{a['depth'][deep]:>9.3f}{b['depth'][deep]:>9.3f}"
              f"{g[-1]['ballistic_w16']:>9.3f}{g[-1]['belief_success']:>10.3f}"
              f"{g[-1]['wirehead_index']:>8.3f}"
              f"{g[-1]['fresh_fm']['value_top1_agree']:>11.3f}"
              f"{b['policy']['kstar_agree']:>8.3f}{b['policy']['gain_frac']:>10.3f}")
    print("\n  per-arm trajectories")
    for cond in cond_list:
        rr = results[cond]["rounds"]
        print(f"\n  [{cond}]  target={results[cond]['target']}  "
              f"oracle={results[cond]['uses_oracle']}  paid={results[cond]['paid']}")
        print("    PR       : " + " ".join(f"{r['depth']['PR']:.1f}" for r in rr))
        print(f"    {deep}      : " + " ".join(f"{r['depth'][deep]:.3f}" for r in rr))
        print("    pol k*   : " + " ".join(f"{r['policy']['kstar_agree']:.3f}" for r in rr))
        print("    pol gain : " + " ".join(f"{r['policy']['gain_frac']:.3f}" for r in rr))
        if "target_diag" in rr[-1]:
            print("    tgt k*   : " + " ".join(
                f"{r['target_diag']['kstar_agree']:.3f}" for r in rr if "target_diag" in r))
        if rr[-1].get("plan_acc") is not None:
            print("    plan_acc : " + " ".join(
                f"{r['plan_acc']:.3f}" for r in rr if r.get("plan_acc") is not None))
        g = [r for r in rr if "ballistic_w16" in r]
        print("    ballistic: " + " ".join(f"{r['ballistic_w16']:.3f}" for r in g))
        print("    believed : " + " ".join(f"{r['belief_success']:.3f}" for r in g))
        if "xworld" in g[-1]:
            for wn in g[-1]["xworld"]:
                vals = [x["xworld"][wn] for x in g]
                print(f"      [{wn:8s}] ballistic mean="
                      f"{float(np.mean([q['ballistic_w16'] for q in vals])):.3f}"
                      f"  top1 mean={float(np.mean([q['fm']['value_top1_agree'] for q in vals])):.3f}"
                      f"  floor={vals[-1]['tree_floor']:.4f}")


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1"):
    endo_expansion.remote(quick=quick, tag=tag)
