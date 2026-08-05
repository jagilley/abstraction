"""The composed loop: does an ENDOGENOUS evaluative judge expand when the action space is
level-indexed and the errors are hierarchical?

WHY THIS RUN EXISTS
-------------------
Three results landed within four days of each other and have never been in the same run:

  `endo_expansion/` (08-01)  an endogenous MC judge -- terminal task success only, no DP, no
                             rule tables, no channel labels -- EXPANDS, recovering 34% of the
                             privileged DP teacher's ballistic lift and 53% of its transferable
                             plannability. Diagnosed shortfall: TARGET FIDELITY, not endogeneity
                             ("a grader is a ceiling"; policy move-quality converges to its
                             target's fidelity from whichever side it starts).
  `level_moves/` (08-03)     the value's abstraction preference is real and GRADED BY WHETHER A
                             COMMITMENT AT THAT LEVEL CAN REACH THE ERROR (L2 slope -0.033,
                             t = -6.81) -- but only once the damage is HIERARCHICAL. The
                             published damage writes random symbols, off-grammar in 29% of tree
                             blocks and therefore repairable one block at a time, so abstraction
                             never had to matter.
  `level_ladder/` (08-04)    levels are worth +0.00016 (t = +0.13) as an ALLOCATION axis and
                             3.51x as an ACTION axis (ballistic 0.121 -> 0.424, t = +10.40),
                             while making the FM WORSE on its dense proxy at the shared cell
                             (-0.106, t = -3.14).

Every expansion number in the repo was measured on the block-only action space under
random-symbol damage. This asks the 2x2 those three imply.

THE 2x2, AND WHY THE INTERACTION IS THE PRIMARY QUANTITY
---------------------------------------------------------
                    | block action space      | level-indexed action space
  ------------------+-------------------------+---------------------------
  external judge    | `evaluative` @ L1       | `evaluative` @ L4
  endogenous judge  | `endo_rollout` @ L1     | `endo_rollout` @ L4

Both main effects are already known in isolation (~34% and 3.51x), so the headline number is
predictable and mostly uninteresting. What is NOT known is the interaction: does a level-indexed
action space pay MORE, LESS, or the same under an endogenous judge as under an exact one?

  * If levels pay through the INNER loop (planner search + FM rollout), the judge's own level
    fidelity should barely matter and the interaction is ~0 -- the two effects multiply and we
    have learned that they compose. That is `level_ladder`'s own proposed mechanism, transplanted.
  * If the interaction is NEGATIVE, the endogenous judge's level fidelity is binding. That is
    the prediction from `level_moves` §11: the value's abstraction premium is systematically
    CONSERVATIVE and the shortfall WIDENS with depth (L2 -0.053 -> -0.078; L3 -0.024 -> -0.097).
  * If it is POSITIVE, hierarchical damage raised the endogenous judge's ceiling -- which
    `level_moves` measured directly (value-oracle rank agreement +0.290 -> +0.583, t = +8.14,
    WHILE terminal success falls 0.475 -> 0.307). Under "a grader is a ceiling" that predicts
    the endogenous arm closes some of its 34% gap here and nowhere else.

ARMS (5, nested; one warm start per action space; matched belief-update budget)
--------------------------------------------------------------------------------
  frozen         root-CE only -- the no-loop floor. Action-space independent BY CONSTRUCTION,
                 so `frozen @ L1` vs `frozen @ L4` is a free harness consistency check: if
                 those two differ, something other than the action space is moving.
  dense          + the asymmetric FM local loss. ENDOGENOUS but HOMOGENEOUS -- a functional of
                 the model's own dense error, with no licence to disagree with the inner loop.
                 The rung that capped BELOW the no-loop floor.
  evaluative     + hard CE against the exact DP's best MOVE. External, precomputed. The ceiling.
  endo_rollout   + hard CE against argmax of a Monte-Carlo estimate of terminal task success
                 over moves. Endogenous AND evaluative.                    ** the cell of interest **
  endo_random    + hard CE against a UNIFORM RANDOM MOVE. The content control, and it is not
                 optional: `endo_expansion` §3 found a zero-information target reaching PR 13.5
                 -- 145% of the DP teacher's PR lift -- with ballistic AT THE NO-LOOP FLOOR.
                 Spearman(PR, ballistic) = +0.27 against +0.87 for fresh-FM top1.

WHAT IS HELD FIXED, AND THE ONE THING THAT IS NOT
---------------------------------------------------
Held: belief-update steps, state pools, warm start, damage schedule, probe states, world
(static -- the published 2x2 measured drift orthogonal at +4.63 +- 0.46 vs +4.89 +- 0.73 PR).
The dense term's move draw is block-matched (`composed_graders.draw_moves_block_matched`) so the
two action spaces see an identical CHANNEL mix.

NOT held, and reported rather than equalised: the endogenous teacher's MATERIALISATION cost
rises with the action space (n_moves * n_roll * (1 + horizon * n_moves) per state, so ~2.7x at
max_level=4). That is a real property of evaluative grading over a richer action set, not a
subsidy -- no arm gets extra GRADIENT steps -- and `meter` carries it.

READ IT ON: ballistic control (fresh span FM, per damage depth), fresh-FM `value_top1_agree`,
and d4. NOT on PR, which `endo_expansion` retired as a ranker.
AND CHECK FIRST: `roll_vs_fm_pred` rank correlation against the `fm_pred_vs_fm_pred_fresh`
homogeneous floor. If the endogenous arm's outer signal ranks moves like the dense inner signal,
THE TWO LOOPS HAVE COLLAPSED and any gain is an inner-loop gain wearing an outer-loop label.
That diagnosis takes precedence over the headline.

Run from experiments/:
  python3 -c "from rhm.directed_sculpting.full_loop.composed_loop.composed_loop import \\
      selfcheck; selfcheck()"                                          # gates, no GPU
  modal run rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop --quick
  modal run --detach rhm/directed_sculpting/full_loop/composed_loop/composed_loop.py::composed_loop \\
      --tag lv_s1 --seed 1 --max-level 4
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import make_layout, sample_pool
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    build_block_tables, make_tree_probe, make_spec, root_ce, snapshot_world, set_world,
    train_generator_channels, tree_depth_probe)
from rhm.directed_sculpting.full_loop.endo_expansion.endo_graders import (
    beam_belief_vs_truth, grader_disagreement)
from rhm.directed_sculpting.full_loop.level_moves.level_moves import (
    apply_moves, build_move_set, certify_damage, collect_value_buffer_moves, flat_move_set,
    make_damage, moves_table, sample_states_damaged, tb_numpy)
from rhm.directed_sculpting.full_loop.level_moves.level_ladder.level_ladder import (
    build_cells, build_span_fm, fm_error_moves, make_damage_mixture, move_index_tensors,
    open_loop_beam_moves, parse_schedule, transition_targets_moves)
from rhm.directed_sculpting.full_loop.composed_loop.composed_graders import (
    belief_update_moves, collect_rollout_moves_mv, draw_moves_block_matched,
    fm_one_step_check_moves, grounded_candidates_moves, kstar_from_candidates,
    move_scores_mv, moves_covering_blocks, value_calibration_moves)


app = modal.App("rhm-ds-composed-loop", image=image)

ARMS = ("frozen", "dense", "evaluative", "endo_rollout", "endo_random",
        "endo_floor", "endo_floor_static", "endo_floor_anti", "endo_treefloor")
_MODE = {"frozen": "frozen", "dense": "dense"}

# --------------------------------------------------------------------------- #
# The "PI instruction": a level floor on what the endogenous teacher may COMMIT to
# --------------------------------------------------------------------------- #
#
# The measured deficit is that the endogenous judge does not CLIMB: its mean target level sits
# at 1.72 across a schedule where the exact DP climbs 2.23 -> 2.80. It can say "an abstraction
# is worth it here"; it cannot say "go up". These arms supply the direction from outside and ask
# how much of the external teacher's advantage that alone recovers.
#
# THE INFORMATION THIS CHANNEL CARRIES IS TINY, WHICH IS THE POINT. One integer per round --
# ~14 bits for a 9-round run -- against the DP teacher's per-state argmax over `n_moves` for
# `ground_states` states every round (~83k bits). If a 14-bit instruction recovers a real share
# of the DP's advantage, then most of what the privileged teacher bought ON THIS AXIS was
# DIRECTION, not content.
#
#   endo_floor          l_min tracks the damage schedule -- the PI who knows the curriculum
#   endo_floor_static   l_min pinned mid -- "just stop working shallow", no curriculum at all
#   endo_floor_anti     l_min reversed -- catches "ANY constraint on the target helps", e.g. by
#                       shrinking the CE's effective output space. Without this arm a positive
#                       result is uninterpretable.
#   endo_treefloor      masked to TREE moves, no level constraint at all.
#
# WHY `endo_treefloor` IS REQUIRED AND IS NOT AN EXTRA. On this geometry the distractors are
# SHALLOWER than the tree (struct_depths 2,2 against tree_depth 4), so a level floor is
# partially a CHANNEL oracle: at l_min=3 only tree moves survive, and "aim higher" becomes
# "act on the tree" -- which the published ladder already prices at 13-18x relevance separation
# and 84% of the allocation prize. A floor arm beating `endo_rollout` could therefore be buying
# relevance rather than altitude. `endo_treefloor` isolates exactly that: it hands over the
# channel information with NO level information. Any floor arm must beat IT, not just
# `endo_rollout`, for the altitude reading to survive. (The alternative fix -- depth-matched
# distractors via `--struct-depths 4,2`, as `partial_hetero` built -- is cleaner but changes
# the geometry, so every comparator would need re-running.)
FLOOR_RULE = {
    "endo_floor": lambda dmg, lv: dmg,
    "endo_floor_static": lambda dmg, lv: int(round(float(np.mean(lv)))),
    "endo_floor_anti": lambda dmg, lv: lv[0] + lv[-1] - dmg,
    "endo_treefloor": lambda dmg, lv: 1,
}


def build_allowed_mask(arm, ms, tb, dmg_level, sched_levels):
    """The instruction, as a bool mask over moves. Returns `(mask, min_level)`.

    `endo_treefloor` masks by CHANNEL; the floor arms mask by LEVEL. Noise channels have no
    grammar and so only ever offer level-1 moves, which means any floor above 1 excludes them
    entirely -- that is a real part of what the instruction does and is reported through
    `allowed_tree_share` rather than corrected away.
    """
    n_moves = ms["n_moves"]
    if arm not in FLOOR_RULE:
        return np.ones(n_moves, dtype=bool), 1
    if arm == "endo_treefloor":
        mask = np.array([mv["channel"] == tb["tree_channel"] for mv in ms["moves"]])
        return mask, 1
    lo = int(FLOOR_RULE[arm](dmg_level, sched_levels))
    mask = np.array([mv["level"] >= lo for mv in ms["moves"]])
    if not mask.any():                      # the move set is shallower than the instruction
        lo = max(mv["level"] for mv in ms["moves"])
        mask = np.array([mv["level"] >= lo for mv in ms["moves"]])
    return mask, lo


def warm_span_fm_matched(fm, controller, generator, layout, tb, ms, cover, *, n_steps,
                         batch_size, n_corrupt, lr, seed, render, gen, device, damage,
                         edit_budget, pool_refresh=250):
    """Fit the span FM on a BLOCK-MATCHED move budget, fresh states every step.

    `level_ladder.warm_span_fm` draws uniformly over CELLS, which is right for that node (its
    budget IS over cells) and wrong here: at `max_level=1` cells are channels, so the block arm's
    warm FM would see the tree in 1/5 of draws against the level arm's 4/10 -- a 2x channel-mix
    swing between the two arms of the contrast this run exists to measure. Block-matched draws
    leave the channel marginal identical to the published `train_block_fm` in both arms while
    still giving each level equal weight inside a channel.

    Freshness is load-bearing for the reason `train_block_fm` records: a reused buffer put the
    published FM at normalised error ~1.0 against a floor of 0.000.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    fm.train()
    opt = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report = max(1, n_steps // 4)
    pool = None
    for step in range(1, n_steps + 1):
        if pool is None or step % pool_refresh == 1:
            pool = sample_pool(layout, 20_000, int(rng.integers(0, 2 ** 31)))
        idx = rng.integers(0, pool["leaves"].shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        start = damage(pool["leaves"][idx], c, rng)
        x = torch.from_numpy(start).to(device)
        for _ in range(int(rng.integers(0, edit_budget // 2 + 1))):
            mv = torch.from_numpy(draw_moves_block_matched(rng, cover, batch_size)).to(device)
            x = apply_moves(generator, x, mv, ms, tb, render=render, gen=gen)
        mv = torch.from_numpy(draw_moves_block_matched(rng, cover, batch_size)).to(device)
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


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, tree_depth=4, seed=1, n_draw=200_000):
    """CG1-CG3: the properties the 2x2's cleanliness rests on. No GPU, no training.

    CG1  the flat move set IS the block set in block order, so `max_level=1` is the published
         action space and not an analogue. (C4, the FM half, lives in `level_ladder.selfcheck`
         and is asserted there; this is the move-set half.)
    CG2  `draw_moves_block_matched` has a channel marginal EQUAL to uniform-over-blocks, at
         both action spaces. This is the confound the whole contrast would otherwise carry.
    CG3  every grammar block is covered by exactly `min(depth, max_level)` moves -- one per
         level -- so the level marginal inside a channel is flat by construction rather than
         by tuning.
    """
    import torch

    device = torch.device("cpu")
    spec = make_spec(tree_depth=tree_depth, struct_depths=[2, 2], struct_ms=[2, 4],
                     noise_blocks=[1, 1])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    n_blocks = tb["n_blocks"]
    chan = tb["chan_blk"].cpu().numpy()
    rng = np.random.default_rng(seed)
    out = {}

    ms_flat = flat_move_set(layout, tb, device)
    assert ms_flat["n_moves"] == n_blocks, "CG1: flat move set is not the block set"
    for mi, mv in enumerate(ms_flat["moves"]):
        assert mv["blk0"] == mi and mv["span"] == 1 and mv["level"] == 1, "CG1: not block order"
    print(f"CG1 PASS -- flat move set == {n_blocks} blocks in block order")

    uniform_chan = np.bincount(chan, minlength=tb["n_channels"]) / n_blocks
    for name, ms in (("L1", ms_flat), ("L4", build_move_set(layout, tb, device))):
        cover = moves_covering_blocks(ms, n_blocks)
        mv = draw_moves_block_matched(rng, cover, n_draw)
        ch = np.array([ms["moves"][int(m)]["channel"] for m in mv])
        got = np.bincount(ch, minlength=tb["n_channels"]) / n_draw
        err = float(np.abs(got - uniform_chan).max())
        assert err < 0.005, f"CG2 FAILED at {name}: channel marginal off by {err:.4f}"
        lv = np.array([ms["moves"][int(m)]["level"] for m in mv])
        out[name] = {"channel_marginal_max_err": err, "n_moves": ms["n_moves"],
                     "level_hist": {int(k): int(c) for k, c in
                                    zip(*np.unique(lv, return_counts=True))}}
        print(f"CG2 PASS -- {name}: {ms['n_moves']} moves, channel marginal within {err:.4f} "
              f"of uniform-over-blocks; level histogram {out[name]['level_hist']}")

        for b in range(n_blocks):
            depth = layout["channels"][chan[b]]["depth"]
            want = min(depth, ms["max_level"]) if depth else 1
            assert len(cover[b]) == want, (
                f"CG3 FAILED at {name}: block {b} covered by {len(cover[b])}, want {want}")
    print("CG3 PASS -- every block covered by exactly one move per available level")

    ms_lv = build_move_set(layout, tb, device)
    cells = build_cells(ms_lv, tb)
    out["census"] = moves_table(ms_lv, tb)
    out["cells"] = cells["names"]
    print(f"  level move set census {json.dumps(out['census'])}")
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote(tree_depth: int = 4):
    return selfcheck(tree_depth=tree_depth)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=28800, memory=32768)
def composed_loop(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
    noise_blocks: str = "1,1", edit_budget: int = 6, n_corrupt: int = 2,
    max_level: int = 0, damage_schedule: str = "1,2,3",
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 6_000, batch_size: int = 256,
    explore_eps: float = 0.3, rounds: int = 12, belief_steps: int = 1500,
    ground_states: int = 2048, anchor_pool: int = 40_000,
    lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0,
    teacher_rolls: int = 2, teacher_eps: float = 0.10, min_informative: int = 128,
    n_dis: int = 256, dis_rolls: int = 1, n_calib: int = 384, calib_rolls: int = 2,
    n_eval: int = 512, n_probe: int = 2000, probe_steps: int = 250,
    fresh_fm_steps: int = 3000, grade_every: int = 3, ballistic_chunk: int = 3,
    beam_width: int = 16, value_action_space: str = "own",
    arms: str = ",".join(ARMS), tag: str = "v1", quick: bool = False,
):
    """One action space, five arms, hierarchical damage that deepens across the run.

    `--max-level 1` is the block-only action space (the published one); `--max-level 4` is the
    level-indexed one. The two are run as SEPARATE JOBS with the same seed, and the 2x2 is read
    across them -- which is also what keeps each job inside a sane wall-clock.

    `n_corrupt` DEFAULTS TO 2, not to the published expansion runs' 3, and the deviation is
    forced rather than tuned: `corrupt_tree_hier` damages `min(n_corrupt, s**(depth-level))`
    nodes, and level 3 has only 2, so at 3 the entire tree is a wrong derivation and the exact
    DP's argmax level sticks at 3 for EVERY damage depth (`level_ladder` gate G-L). The
    necessity structure this run needs would be swamped.
    """
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = 700
        value_episodes, fm_warm_steps = 5_000, 500
        rounds, belief_steps, ground_states, anchor_pool = 3, 250, 384, 8_000
        n_eval, n_probe, probe_steps, fresh_fm_steps = 128, 600, 80, 400
        n_dis, n_calib, min_informative, grade_every = 96, 96, 16, 2

    arm_list = [a for a in arms.split(",") if a]
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
    T, n_blocks, L = layout["total_len"], tb["n_blocks"], tree_depth
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)

    ms = build_move_set(layout, tb, device, max_level=(max_level or None))
    cells = build_cells(ms, tb)
    cover = moves_covering_blocks(ms, n_blocks)
    span_mask, level_of = move_index_tensors(ms, tb, device)
    sched = parse_schedule(damage_schedule, rounds)
    sched_levels = sorted(set(sched))
    n_moves = ms["n_moves"]
    presteps = edit_budget // 2
    teacher_horizon = max(1, edit_budget - presteps - 1)
    calib_horizon = max(1, edit_budget - presteps)
    n_tree_moves = sum(1 for mv in ms["moves"] if mv["channel"] == tb["tree_channel"])

    print(f"Composed loop: max_level={max_level or L}, {n_moves} moves over {cells['n_cells']} "
          f"cells {json.dumps(moves_table(ms, tb))}")
    print(f"  T={T} tokens / {n_blocks} blocks; damage schedule (round -> depth) {sched}")
    print(f"  teacher horizon={teacher_horizon} rolls={teacher_rolls}; arms={arm_list}")

    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, SpanFM = _build_value_head(), build_span_fm()

    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
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
    del pool

    # ---- the damage model, certified before anything trains on it -----------------------
    tb_np = tb_numpy(tb)
    certs = {}
    for lv in sched_levels:
        cert = certify_damage(layout, tb_np, level=lv, n_corrupt=n_corrupt,
                              n=1024 if quick else 4096)
        certs[str(lv)] = cert
        print(f"  G-D damage={lv}: on-grammar {cert['on_grammar']:.4f}, "
              f"d* mean {cert['dstar_mean']:.3f}, d*==0 {cert['dstar_zero_frac']:.4f}")
        assert cert["on_grammar"] > 0.999, f"G-D FAILED at level {lv}"
        assert cert["dstar_zero_frac"] < 0.05, f"G-D FAILED at level {lv}: no damage"
    mixture = make_damage_mixture(layout, tb_np, sched_levels)

    # ---- the value: trained on the arm's OWN action space and the damage mixture --------
    # Own action space, because the published 3.51x action-space contrast was measured that way
    # (`level_ladder` §5, "at matched task and matched allocation") -- a value trained on block
    # moves and read on level moves would be measuring out-of-distribution generalisation.
    # Damage MIXTURE, because a value trained on shallow damage and probed on deep damage
    # conflates "does the judge prefer abstraction" with "does it generalise across depths".
    # `value_action_space` selects which BEHAVIOUR POLICY generates the value's training states.
    #
    # The value is V(state, root) -- `_train_value_mc` never sees an action -- so the action
    # space enters only through WHICH STATES GET VISITED. Level-move rollouts visit states
    # produced by wholesale subtree rewrites; block-move rollouts never do.
    #
    # `value_probe` (vp_l1_clean_s1 vs vp_l4_s1, same damage mixture, same seed) measures what
    # that costs: a value trained on the FLAT space reproduces the published matched-span
    # premium (tree|L3 0.595 / 0.634 / 0.610 against the published 0.618) and tracks the exact
    # DP at rank corr +0.53 at every damage depth, while the LEVEL-trained value halves the
    # premium (0.537 / 0.530 / 0.520) and decays to +0.075 by damage depth 3.
    #
    # But flat is not simply better, and the same probe says why: off its training distribution
    # the flat-trained value is CONFIDENTLY WRONG at the root -- it prefers the committed
    # level-4 move at 0.565-0.596 where the exact DP prefers the lazy twin at 0.22. Handing that
    # value to a planner that can make root moves invites the planner to find and exploit
    # exactly that error, which is why `planner_mean_level` below is instrumented.
    #
    #   own     rollouts over the arm's own (level-indexed) move set -- what every result on
    #           disk used. In-distribution at depth, blunt in the middle.
    #   flat    rollouts over block moves only. Sharp in the middle, out-of-distribution at
    #           depth. The variant the probe recommends and the one carrying the hazard.
    #   mixed   half the episodes from each. Deep-state coverage AND the flat space's
    #           discrimination -- the hedge, and the arm that would have worked if `flat`
    #           hits the hazard.
    if value_action_space not in ("own", "flat", "mixed"):
        raise ValueError(value_action_space)
    ms_flat = flat_move_set(layout, tb, device)
    print(f"Value: {value_episodes} rollouts, damage mixture {sched_levels}, "
          f"action space for value states = '{value_action_space}' "
          f"(own={n_moves} moves, flat={ms_flat['n_moves']})")
    if value_action_space == "mixed":
        half = value_episodes // 2
        c1, r1, s1 = collect_value_buffer_moves(
            controller0, generator, layout, tb, ms_flat, n_episodes=half, batch_size=1024,
            n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
            render=render, gen=gen, device=device, damage=mixture)
        c2, r2, s2 = collect_value_buffer_moves(
            controller0, generator, layout, tb, ms, n_episodes=value_episodes - half,
            batch_size=1024, n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps,
            seed=seed + 32, render=render, gen=gen, device=device, damage=mixture)
        cfg = torch.cat([c1, c2]); rts = torch.cat([r1, r2]); suc = torch.cat([s1, s2])
        del c1, r1, s1, c2, r2, s2
    else:
        cfg, rts, suc = collect_value_buffer_moves(
            controller0, generator, layout, tb,
            ms_flat if value_action_space == "flat" else ms,
            n_episodes=value_episodes, batch_size=1024, n_corrupt=n_corrupt,
            budget=edit_budget, epsilon=explore_eps, seed=seed + 31, render=render, gen=gen,
            device=device, damage=mixture)
    value0 = ValueHead(state_dim, v).to(device)
    _train_value_mc(value0, controller0, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    print(f"  value buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    del cfg, rts, suc

    fm0 = SpanFM(state_dim, n_blocks, ms["max_level"], span_mask, level_of,
                 n_head=4, n_layer=2).to(device)
    warm_span_fm_matched(fm0, controller0, generator, layout, tb, ms, cover,
                         n_steps=fm_warm_steps, batch_size=batch_size, n_corrupt=n_corrupt,
                         lr=1e-3, seed=seed + 51, render=render, gen=gen, device=device,
                         damage=mixture, edit_budget=edit_budget)

    # ---- frozen probes: one eval set PER damage depth, graded at every grading round -----
    # Per depth rather than per round, so a ballistic number at round 12 is comparable to one at
    # round 3 at the SAME depth. Grading every depth every time also shows whether competence at
    # deep damage improves while the schedule is still shallow, which is the discriminator
    # between "the allocator followed the world" and "deep moves just get better with training".
    evals = {}
    for i, lv in enumerate(sched_levels):
        dmg = make_damage(layout, tb_np, lv)
        evals[lv] = sample_states_damaged(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                          presteps=0, seed=seed + 400 + i, device=device,
                                          render=render, gen=gen, damage=dmg)
    probe_leaves, level_feats, _ = make_tree_probe(layout, tb, seed + 43)
    x_dis, r_dis = sample_states_damaged(layout, tb, generator, n=n_dis, n_corrupt=n_corrupt,
                                         presteps=presteps, seed=seed + 45, device=device,
                                         render=render, gen=gen, damage=mixture)
    x_cal, r_cal = sample_states_damaged(layout, tb, generator, n=n_calib, n_corrupt=n_corrupt,
                                         presteps=presteps, seed=seed + 46, device=device,
                                         render=render, gen=gen, damage=mixture)
    x_fm, r_fm = sample_states_damaged(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                       presteps=presteps, seed=seed + 48, device=device,
                                       render=render, gen=gen, damage=mixture)

    W0 = snapshot_world(tb)
    depth0 = tree_depth_probe(controller0, probe_leaves, level_feats, layout, tb, device,
                              probe_steps=probe_steps)
    print("Round-0 belief: " + " ".join(f"{k}={val:.3f}" for k, val in depth0.items()))

    common = dict(
        controller0=controller0, fm0=fm0, value0=value0, generator=generator, layout=layout,
        tb=tb, ms=ms, cover=cover, SpanFM=SpanFM, span_mask=span_mask, level_of=level_of,
        evals=evals, probe_leaves=probe_leaves, level_feats=level_feats, x_dis=x_dis,
        r_dis=r_dis, x_cal=x_cal, r_cal=r_cal, x_fm=x_fm, r_fm=r_fm, sched=sched,
        tb_np=tb_np, layout_v=v, seed=seed, state_dim=state_dim, n_blocks=n_blocks,
        rounds=rounds, belief_steps=belief_steps, ground_states=ground_states,
        anchor_pool=anchor_pool, batch_size=batch_size, n_corrupt=n_corrupt,
        edit_budget=edit_budget, presteps=presteps, teacher_horizon=teacher_horizon,
        teacher_rolls=teacher_rolls, teacher_eps=teacher_eps, min_informative=min_informative,
        calib_horizon=calib_horizon, calib_rolls=calib_rolls, dis_rolls=dis_rolls,
        lam_fm=lam_fm, lam_plan=lam_plan, tau=tau, probe_steps=probe_steps,
        fresh_fm_steps=fresh_fm_steps, grade_every=grade_every,
        ballistic_chunk=ballistic_chunk, beam_width=beam_width, mixture=mixture,
        W0=W0, render=render, device=device)

    results = {}
    for arm in arm_list:
        print(f"\n{'#' * 72}\n# ARM: {arm}  (max_level={max_level or L})\n{'#' * 72}")
        results[arm] = _run_arm(arm, **common)
        volume.commit()

    print(f"\n{'=' * 78}\n=== SUMMARY -- composed loop @ max_level={max_level or L} ===\n"
          f"{'=' * 78}")
    print(f"round-0 belief: PR={depth0['PR']:.2f}  "
          + " ".join(f"{k}={depth0[k]:.3f}" for k in depth0 if k != "PR"))
    for arm in arm_list:
        rr = results[arm]["rounds"]
        graded = [r for r in rr if "grades" in r]
        print(f"\n[{arm}]")
        print(f"  d{L}       : " + "  ".join(f"r{r['round']}={r['depth'][f'd{L}']:.3f}"
                                             for r in rr))
        for lv in sched_levels:
            print(f"  ballistic@dmg{lv}: " + "  ".join(
                f"r{r['round']}={r['grades'][str(lv)]['ballistic']:.3f}" for r in graded))
        print("  freshtop1: " + "  ".join(
            f"r{r['round']}={r['grades'][str(sched_levels[-1])]['fm']['value_top1_agree']:.3f}"
            for r in graded))
        print("  planlvl  : " + "  ".join(
            f"r{r['round']}={r['grades'][str(sched_levels[-1])]['planner_mean_level']:.2f}"
            for r in graded)
            + "   (level of moves the BEAM committed to; rising with falling ballistic ="
              " the value is being exploited at depth)")
        tt = [r["teacher"] for r in rr if r.get("teacher")]
        if tt:
            print(f"  teacher  : agree_dp={np.mean([t['agree_dp'] for t in tt]):.3f} "
                  f"tree_share={np.mean([t['tree_share'] for t in tt]):.3f} "
                  f"mean_level={np.mean([t['mean_level'] for t in tt]):.2f} "
                  f"informative={np.mean([t['informative_frac'] for t in tt]):.3f} "
                  f"| meter mat={rr[-1]['meter']['materialisations']:.3g}")
            if arm in FLOOR_RULE:
                print(f"  instruct : l_min={[t['min_level'] for t in tt]} "
                      f"n_allowed={[t['n_allowed'] for t in tt]} "
                      f"allowed_tree={np.mean([t['allowed_tree_share'] for t in tt]):.3f} "
                      f"excl_oracle={np.mean([t['floor_excludes_oracle_frac'] for t in tt]):.3f} "
                      f"agree|allowed="
                      f"{np.nanmean([t['agree_dp_within_allowed'] for t in tt]):.3f} "
                      f"(chance {np.mean([t['chance_within_allowed'] for t in tt]):.3f})")
        cal = [r["value_calib"] for r in rr]
        print(f"  wirehead : bias r1={cal[0]['bias']:+.3f} -> r{rr[-1]['round']}="
              f"{cal[-1]['bias']:+.3f}")
        dis = [r["disagree"] for r in rr]
        for key in ("roll_vs_fm_pred", "dp_vs_fm_pred", "fm_pred_vs_fm_pred_fresh"):
            vals = [d[key] for d in dis if key in d]
            if vals:
                print(f"  disagree {key:26s}: "
                      f"top1={np.mean([x['top1_disagree'] for x in vals]):.3f} "
                      f"rank_corr={np.mean([x['rank_corr'] for x in vals]):+.3f}")

    metrics = {
        "config": {"v": v, "s": s, "tree_depth": tree_depth, "total_len": T,
                   "n_blocks": n_blocks, "max_level": max_level or L, "n_moves": n_moves,
                   "n_tree_moves": n_tree_moves, "n_cells": cells["n_cells"],
                   "cell_names": cells["names"], "move_census": moves_table(ms, tb),
                   "damage_schedule": sched, "sched_levels": sched_levels,
                   "n_corrupt": n_corrupt, "edit_budget": edit_budget, "presteps": presteps,
                   "rounds": rounds, "belief_steps": belief_steps,
                   "ground_states": ground_states, "teacher_horizon": teacher_horizon,
                   "teacher_rolls": teacher_rolls, "world": "static", "arms": arm_list,
                   "render": render, "seed": seed, "ground_states_": ground_states,
                   "value_action_space": value_action_space},
        "damage_certification": certs, "belief_round0": depth0, "results": results,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/composed_loop_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


def _run_arm(arm, *, controller0, fm0, value0, generator, layout, tb, ms, cover, SpanFM,
             span_mask, level_of, evals, probe_leaves, level_feats, x_dis, r_dis, x_cal, r_cal,
             x_fm, r_fm, sched, tb_np, layout_v, seed, state_dim, n_blocks, rounds,
             belief_steps, ground_states, anchor_pool, batch_size, n_corrupt, edit_budget,
             presteps, teacher_horizon, teacher_rolls, teacher_eps, min_informative,
             calib_horizon, calib_rolls, dis_rolls, lam_fm, lam_plan, tau, probe_steps,
             fresh_fm_steps, grade_every, ballistic_chunk, beam_width, mixture, W0,
             render, device):
    """One arm. Forks from the SAME controller/FM/value with a matched update budget; only the
    teacher differs.

    Every arm pays the SAME instrument charge -- the disagreement probe and the value-calibration
    rollout run every round regardless of whether the arm uses them -- so arms differ in their
    loss and in nothing else. The global torch RNG is re-seeded identically at every arm's start
    (`endo_expansion`'s discipline): without it an arm inherits the previous arm's minibatch
    stream, and running a subset of arms would not reproduce the run that included all of them.
    """
    import torch

    torch.manual_seed(seed + 12_345)
    set_world(tb, W0)
    controller = copy.deepcopy(controller0)
    for p in controller.parameters():
        p.requires_grad_(True)
    fm, value = copy.deepcopy(fm0), copy.deepcopy(value0)
    for mod in (fm, value):
        for p in mod.parameters():
            p.requires_grad_(True)

    mode = _MODE.get(arm, "evaluative")
    gen = torch.Generator(device=device).manual_seed(seed + 777)
    rng = np.random.default_rng(seed + 991)
    tree_c = tb["tree_channel"]
    move_level = np.array([mv["level"] for mv in ms["moves"]])
    move_is_tree = np.array([mv["channel"] == tree_c for mv in ms["moves"]])
    sched_levels = sorted(set(sched))
    meter = {"materialisations": 0.0, "dp_calls": 0.0}
    rows = []
    fresh = None

    for rnd in range(1, rounds + 1):
        dmg_level = sched[rnd - 1]
        damage = make_damage(layout, tb_np, dmg_level)
        anchor = sample_pool(layout, anchor_pool, seed + 6000 + rnd)
        anchor_leaves = torch.from_numpy(anchor["leaves"])
        anchor_roots = torch.from_numpy(anchor["roots"].astype(np.int64))
        gx, gr = sample_states_damaged(layout, tb, generator, n=ground_states,
                                       n_corrupt=n_corrupt, presteps=presteps,
                                       seed=seed + 7000 + rnd, device=device, render=render,
                                       gen=gen, damage=damage)

        # ---- the teacher ------------------------------------------------------------
        gk = px = pr = None
        tinfo = None
        if mode == "evaluative":
            # ONE DP pass per round in EVERY teacher arm: the `evaluative` target, and the
            # `agree_dp` diagnostic every other teacher is scored against. Computing it for
            # all arms means the comparison carries no state-draw confound and the instrument
            # charge is identical -- only `evaluative` ever puts it in a loss.
            cand, d_cur_v = grounded_candidates_moves(layout, tb, generator, ms, gx, gr,
                                                      render=render, gen=gen, device=device)
            mvstar, dp_informative = kstar_from_candidates(cand, d_cur_v, rng)
            meter["dp_calls"] += ground_states * (ms["n_moves"] + 1)
            no_improving = float(1.0 - dp_informative.mean())
            informative = np.ones(gx.shape[0], dtype=bool)
            allowed = np.ones(ms["n_moves"], dtype=bool)
            min_level = 1

            if arm == "evaluative":
                mv_hat, informative = mvstar, dp_informative
                meter["materialisations"] += ground_states * ms["n_moves"]

            elif arm == "endo_rollout" or arm in FLOOR_RULE:
                allowed, min_level = build_allowed_mask(arm, ms, tb, dmg_level, sched_levels)
                _, mv_hat, informative = collect_rollout_moves_mv(
                    controller, generator, layout, tb, ms, gx, gr, horizon=teacher_horizon,
                    epsilon=teacher_eps, n_roll=teacher_rolls, render=render, gen=gen,
                    device=device, rng=rng, allowed=allowed)
                meter["materialisations"] += (ground_states * int(allowed.sum()) * teacher_rolls
                                              * (1 + teacher_horizon * ms["n_moves"]))

            elif arm == "endo_random":
                # The content control. `endo_expansion` §3: a zero-information target reached
                # PR 13.5, 145% of the DP teacher's PR lift, with ballistic AT THE FLOOR. Its
                # agreement with `mv*` and its level histogram are both chance by construction
                # and are reported so the comparison is against a measured floor.
                mv_hat = rng.integers(0, ms["n_moves"], size=gx.shape[0]).astype(np.int64)

            if informative.sum() < min_informative:
                informative = np.ones(gx.shape[0], dtype=bool)
            sel = np.nonzero(informative)[0]
            px = gx[torch.from_numpy(sel).to(device)]
            pr = gr[torch.from_numpy(sel).to(device)]
            gk = torch.from_numpy(mv_hat[sel]).to(device)

            tinfo = {"agree_dp": float((mv_hat == mvstar).mean()),
                     "tree_share": float(move_is_tree[mv_hat].mean()),
                     "mean_level": float(move_level[mv_hat].mean()),
                     "level_hist": {int(k): int(c) for k, c in
                                    zip(*np.unique(move_level[mv_hat], return_counts=True))},
                     "oracle_mean_level": float(move_level[mvstar].mean()),
                     "informative_frac": float(informative.mean()),
                     "dp_no_improving_frac": no_improving,
                     "damage_level": dmg_level,
                     "dstar_cur": float(d_cur_v.mean()),
                     "dstar_best_move": float(cand.min(axis=0).mean()),
                     # ---- the instruction, and whether it helped or just narrowed ----------
                     "min_level": int(min_level),
                     "n_allowed": int(allowed.sum()),
                     "allowed_tree_share": float(move_is_tree[allowed].mean()),
                     # How often the instruction ruled out the RIGHT answer. A floor that is
                     # too aggressive is actively harmful and this is where that shows up.
                     "floor_excludes_oracle_frac": float((~allowed[mvstar]).mean()),
                     # THE H1/H2 DISCRIMINATOR. Among states where the floor left the oracle's
                     # own answer reachable, does the teacher find it? High => the within-level
                     # ranking was always fine and the entire deficit was DIRECTIONAL. Near
                     # chance => the teacher cannot rank abstract moves at all and no amount of
                     # pointing it upward will help.
                     "agree_dp_within_allowed": (
                         float((mv_hat[allowed[mvstar]] == mvstar[allowed[mvstar]]).mean())
                         if allowed[mvstar].any() else float("nan")),
                     "chance_within_allowed": 1.0 / max(int(allowed.sum()), 1)}

        info = belief_update_moves(mode, controller, fm, value, generator, layout, tb, ms,
                                   cover, train_leaves=anchor_leaves,
                                   train_roots=anchor_roots, gstates=gx, groots=gr,
                                   gmvstar=gk, n_steps=belief_steps, batch_size=batch_size,
                                   lr=3e-4, lam_fm=lam_fm, lam_plan=lam_plan, tau=tau,
                                   render=render, gen=gen, device=device, seed=seed + rnd,
                                   pstates=px, proots=pr)

        depth = tree_depth_probe(controller, probe_leaves, level_feats, layout, tb, device,
                                 probe_steps=probe_steps)
        row = {"round": rnd, "damage_level": dmg_level, "depth": depth,
               "root_ce": root_ce(controller, anchor_leaves[:4096], anchor_roots[:4096],
                                  device),
               "root_acc": info.get("root_acc"), "plan_acc": info.get("plan_acc"),
               "teacher": tinfo, "meter": dict(meter)}

        row["value_calib"] = value_calibration_moves(controller, generator, value, layout, tb,
                                                     ms, x_cal, r_cal, horizon=calib_horizon,
                                                     epsilon=teacher_eps, n_roll=calib_rolls,
                                                     render=render, gen=gen, device=device)

        sc = move_scores_mv(controller, generator, fm, value, layout, tb, ms, x_dis, r_dis,
                            horizon=teacher_horizon, epsilon=teacher_eps, n_roll=dis_rolls,
                            render=render, gen=gen, device=device, fm_fresh=fresh)
        row["disagree"] = grader_disagreement(sc)
        row["disagree_has_fresh"] = fresh is not None

        if rnd % grade_every == 0 or rnd == rounds:
            # A FRESH span FM: the transferable grader. An arm cannot improve `value_top1_agree`
            # by making its own co-trained FM agree with its own value, which is what makes this
            # the honest readout and why `endo_expansion` found it the best correlate of
            # ballistic control (Spearman +0.87, against PR's +0.27).
            fresh = SpanFM(state_dim, n_blocks, ms["max_level"], span_mask, level_of,
                           n_head=4, n_layer=2).to(device)
            warm_span_fm_matched(fresh, controller, generator, layout, tb, ms, cover,
                                 n_steps=fresh_fm_steps, batch_size=batch_size,
                                 n_corrupt=n_corrupt, lr=1e-3, seed=seed + 8000 + rnd,
                                 render=render, gen=gen, device=device, damage=mixture,
                                 edit_budget=edit_budget)
            grades = {}
            for lv, (xe, re_) in evals.items():
                succ, x_fin, r_fin, plan_stats = _beam_with_final(
                    controller, generator, fresh, value, ms, tb, layout, xe, re_,
                    budget=edit_budget, beam_width=beam_width, render=render, gen=gen,
                    device=device, chunk_len=ballistic_chunk)
                grades[str(lv)] = {
                    "ballistic": succ, **plan_stats,
                    "beam_grades": beam_belief_vs_truth(controller, layout, x_fin, r_fin),
                    "fm": fm_one_step_check_moves(fresh, value, controller, generator, tb, ms,
                                                  x_fm, r_fm, render=render, gen=gen,
                                                  device=device)}
            row["grades"] = grades
            mvp = torch.from_numpy(draw_moves_block_matched(rng, cover, x_fm.shape[0])).to(device)
            zp, tgtp = transition_targets_moves(controller, generator, x_fm, mvp, ms, tb,
                                                render=render, gen=gen)
            row["fm_probe"] = {
                "own_nmse_acted": fm_error_moves(fm, zp, mvp, tgtp, ms, acted_only=True)[0],
                "fresh_nmse_acted": fm_error_moves(fresh, zp, mvp, tgtp, ms,
                                                   acted_only=True)[0]}

        rows.append(row)
        d = row["disagree"]
        bal = (f" bal@{sched[rnd - 1]}={row['grades'][str(sched[rnd - 1])]['ballistic']:.3f}"
               f" planlvl={row['grades'][str(sched[rnd - 1])]['planner_mean_level']:.2f}"
               if "grades" in row else "")
        print(f"  r{rnd:02d} dmg={dmg_level} PR={depth['PR']:.2f} "
              + " ".join(f"{k}={val:.3f}" for k, val in depth.items() if k != "PR")
              + f" rootCE={row['root_ce']:.4f}" + bal
              + (f" | teach agree_dp={tinfo['agree_dp']:.2f} tree={tinfo['tree_share']:.2f}"
                 f" lvl={tinfo['mean_level']:.2f}/{tinfo['oracle_mean_level']:.2f}"
                 f" info={tinfo['informative_frac']:.2f}" if tinfo else "")
              + f" | vbias={row['value_calib']['bias']:+.3f}"
              + (f" dis(roll,fm)={d['roll_vs_fm_pred']['rank_corr']:+.2f}"
                 if "roll_vs_fm_pred" in d else ""))
    return {"rounds": rows}


def _beam_with_final(controller, generator, fm, value, ms, tb, layout, x0, roots, *, budget,
                     beam_width, render, gen, device, chunk_len):
    """`open_loop_beam_moves` that also returns the terminal states.

    The upstream function returns only the success rate; `beam_belief_vs_truth` needs the final
    sequences to read the BELIEF grader on the same states the paid grader saw. That pair is
    `RHM_EDIT_CONTROL`'s Layer-1/Layer-2 table -- the belief scalar going to 1.0 while true
    success stays at 0 -- and it is the wireheading readout that matters most for an arm whose
    teacher is endogenous.
    """
    import torch
    from rhm.rhm_channels import possible_set_success
    from rhm.directed_sculpting.full_loop.level_moves.level_ladder.level_ladder import (
        _imagine_plan_moves)

    lv_of = np.array([mv["level"] for mv in ms["moves"]])
    tree_of = np.array([mv["channel"] == tb["tree_channel"] for mv in ms["moves"]])
    chosen = []
    with torch.no_grad():
        x = x0.to(device)
        roots = roots.to(device)
        left = budget
        while left > 0:
            h = min(chunk_len, left)
            left -= h
            plan = _imagine_plan_moves(controller, fm, value, ms, x, roots, horizon=h,
                                       beam_width=beam_width, device=device)
            chosen.append(plan.cpu().numpy().ravel())
            for t in range(h):
                x = apply_moves(generator, x, plan[:, t], ms, tb, render=render, gen=gen)
        succ = possible_set_success(layout, x.cpu().numpy(), roots.cpu().numpy())
    # WHAT THE PLANNER ACTUALLY COMMITTED TO. Everything reported so far has been the TEACHER's
    # target level; this is the level of the moves the beam executed, which is the quantity the
    # value's calibration acts on. It is the discriminator for the flat-value hazard: if a
    # flat-trained value is being exploited at the root, this rises toward 4 while ballistic
    # falls. Without it a null in the `flat` arm is unattributable.
    ch = np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)
    plan_stats = {
        "planner_mean_level": float(lv_of[ch].mean()) if ch.size else float("nan"),
        "planner_tree_share": float(tree_of[ch].mean()) if ch.size else float("nan"),
        "planner_level_hist": ({int(k): int(c) for k, c in
                                zip(*np.unique(lv_of[ch], return_counts=True))}
                               if ch.size else {}),
    }
    return float(succ.mean()), x, roots, plan_stats


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def value_probe(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96, tree_depth: int = 4,
    struct_depths: str = "2,2", struct_ms: str = "2,4", noise_blocks: str = "1,1",
    edit_budget: int = 6, n_corrupt: int = 2, max_level: int = 0,
    damage_schedule: str = "1,2,3", controller_steps: int = 12_000,
    generator_steps: int = 12_000, value_steps: int = 12_000, value_episodes: int = 40_000,
    batch_size: int = 256, explore_eps: float = 0.3, n_probe: int = 512,
    probe_max_level: int = 0, tag: str = "v1", quick: bool = False,
):
    """THE PREREQUISITE CHECK: does this run's value prefer an abstract COMMITMENT at MATCHED SPAN?

    Rebuilds `composed_loop`'s round-0 value deterministically -- same seeds, same call order,
    same damage mixture, same action space -- and reads `level_value_probe` on it with lazy
    twins enabled. No loop, no teacher, no FM: this is a property of the warm start every arm
    forks from.

    WHY THE MATCHED-SPAN PAIR AND NOT A LEVEL PROFILE. `level_moves` §9 retracted "the value
    rates higher levels higher": a level-ell move rewrites `s**ell` tokens and displaces the
    pooled latent `z.mean(dim=1)` more regardless of what it buys, and a depth-matched
    distractor with Delta-`d*` certified 0.000 rises 2.79x from L1 to L4 against the tree's
    2.56x. What survived is the PAIRED readout: each committed level-ell move against a lazy
    twin rewriting the identical positions without committing, so the span cancels inside the
    pair. Published: `pref_rate_untied` 0.557 +- 0.018 at L2, 0.618 +- 0.052 at L3, against an
    exact oracle's 0.610 / 0.642.

    TWO THINGS DIFFER HERE from the published probe, which is exactly why it is worth running:
    this value is trained on the DAMAGE MIXTURE (levels 1,2,3 drawn per batch) rather than at
    one depth, and on the LEVEL-INDEXED action space rather than the flat one. Either could
    have destroyed the premium.

    Probed at EACH damage depth separately, because §11's actual finding is that the premium is
    graded by whether a commitment at that level can reach the error (L2 slope -0.033,
    t = -6.81) -- a single pooled number would average that away.
    """
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_rich_controller
    from rhm.directed_sculpting.full_loop.level_moves.level_moves import (
        level_value_probe, print_probe)

    if quick:
        controller_steps = generator_steps = value_steps = 700
        value_episodes, n_probe = 5_000, 128

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
    T, n_blocks = layout["total_len"], tb["n_blocks"]
    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)

    # the ACTION space the value is trained on -- no twins, exactly as the loop builds it
    ms = build_move_set(layout, tb, device, max_level=(max_level or None))
    # The PROBE space is DELIBERATELY DECOUPLED from the training space, and defaults to the
    # full hierarchy. A value trained on the flat action space (`max_level=1`) must still be
    # ASKED about deep commitments, or there is nothing to pair -- twins exist only at level
    # >= 2, so probing a flat value with a flat move set returns an empty table. `level_moves`
    # makes the same choice: both action spaces are read with the level-indexed move set, which
    # is what makes the two trainings comparable at all.
    ms_lazy = build_move_set(layout, tb, device, max_level=(probe_max_level or None),
                             lazy_twins=True)
    sched_levels = sorted(set(parse_schedule(damage_schedule, 9)))

    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead = _build_value_head()

    pool = sample_pool(layout, 20_000 if quick else 100_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))
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
    del pool, train_leaves, train_roots

    tb_np = tb_numpy(tb)
    mixture = make_damage_mixture(layout, tb_np, sched_levels)
    cfg, rts, suc = collect_value_buffer_moves(
        controller, generator, layout, tb, ms, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render=render, gen=gen, device=device, damage=mixture)
    value = ValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    print(f"  value buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    del cfg, rts, suc

    out = {}
    for lv in sched_levels:
        dmg = make_damage(layout, tb_np, lv)
        x0, roots = sample_states_damaged(layout, tb, generator, n=n_probe,
                                          n_corrupt=n_corrupt, presteps=0, seed=seed + 700 + lv,
                                          device=device, render=render, gen=gen, damage=dmg)
        res = level_value_probe(value, controller, generator, layout, tb, ms_lazy, x0, roots,
                                render=render, gen=gen, device=device)
        out[str(lv)] = res
        print_probe(res, f"damage depth {lv}")

    print(f"\n{'=' * 78}\nMATCHED-SPAN PREMIUM by damage depth -- the prerequisite\n{'=' * 78}")
    print(f"{'cell':>14} " + "".join(f"{'dmg' + str(d):>22}" for d in sched_levels))
    print(f"{'':>14} " + "".join(f"{'value/oracle  agree':>22}" for _ in sched_levels))
    keys = sorted({k for r in out.values() for k in r["matched_span"]})
    for k in keys:
        cells = []
        for d in sched_levels:
            m = out[str(d)]["matched_span"].get(k)
            if m is None or m["pref_rate_untied"] is None:
                cells.append(f"{'-':>22}")
            else:
                cells.append(f"{m['pref_rate_untied']:8.3f}/{m['dp_pref_rate_untied']:.3f}"
                             f"{m['agree_untied']:8.3f}")
        print(f"{k:>14} " + "".join(cells))
    print("\n  Published anchors (single damage depth, flat action space, value trained at one\n"
          "  depth): tree|L2 0.557, tree|L3 0.618, against an oracle's 0.610 / 0.642.\n"
          "  0.500 = indifferent between a commitment and the same positions rewritten lazily.")

    metrics = {"config": {"seed": seed, "max_level": max_level or tree_depth,
                          "n_probe": n_probe, "sched_levels": sched_levels,
                          "n_corrupt": n_corrupt, "damage_schedule": damage_schedule},
               "by_damage": out, "elapsed_seconds": time.time() - started}
    out_dir = f"{DATA_DIR}/directed_sculpting/composed_value_probe_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1", seed: int = 1, max_level: int = 0,
         arms: str = ",".join(ARMS)):
    composed_loop.remote(quick=quick, tag=tag, seed=seed, max_level=max_level, arms=arms)
