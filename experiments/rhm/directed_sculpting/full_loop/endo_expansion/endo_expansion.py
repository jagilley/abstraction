"""E4 -- can an ENDOGENOUS evaluative grader EXPAND? The 2x2's missing diagonal.

`ideas/meta_learning_under_metered_data.md` states the hole in its own "What this does not
establish", and it is the most load-bearing one in the program:

    "The two positives do not overlap, and this is the central gap. `visits` is endogenous but
     was never tested for expansion -- `expansion.py` has no allocation policy, no oracle, and
     no `visits` tap at all. Conversely the DP `k*` teacher expands but is external and
     precomputed, and sculpt-continual says that is *why* it works ('so the loop can't wirehead
     its own value'). No run is both endogenous and expanding."

Every expansion result in this repo stands on privileged external ground truth. The one
endogenous evaluative signal that has ever worked -- an MC value trained only on terminal task
success -- has only ever been tested for ALLOCATION. This asks whether it, or something like it,
can drive EXPANSION.

THE TWO OUTER-LOOP ACTIONS, AND WHICH ARM DOES WHICH
------------------------------------------------------
  choose WHAT   write back a target the inner loop fits.
                `dense` `evaluative` `endo_rollout` `endo_shuffled` `endo_value`
  choose WHERE  allocate a metered budget over where the target is computed.
                `alloc_kstar` `alloc_uniform`
No arm does both; doing both at once makes the result unattributable.

THE ARMS (nested, one warm start, matched update budget, static world)
-----------------------------------------------------------------------
  frozen         root-CE only -- the no-loop floor.                        [PR 6.99->6.93]
  dense          + asymmetric FM local loss. ENDOGENOUS but HOMOGENEOUS:
                 a functional of the model's own dense error, so it has no
                 licence to disagree with the inner loop.                  [PR 6.31->6.61]
  evaluative     + hard CE against the exact-DP best move `k*`. EXTERNAL.  [PR 9.03->11.56]
  endo_rollout   + hard CE against `argmax_k` of a MONTE-CARLO estimate of
                 terminal task success. Endogenous AND evaluative.         ** the missing cell **
  endo_shuffled  the same targets, permuted across states. Controls the
                 CE-over-moves loss SHAPE, which by itself feeds gradient
                 to idle capacity and could raise PR with no content.
  endo_value     + hard CE against `argmax_k V(materialised next state)` --
                 the REPORTED-currency teacher. Built to wirehead, as the
                 contrast that makes §7's design rule a measurement.
  alloc_kstar    `k*` computed on only the top-`dp_blocks` blocks by
                 `forecast_visits` -- sculpt-continual's own next step #3.
  alloc_uniform  `k*` on a uniform block draw of the same size -- its control.

Drift is dropped: the published 2x2 measured it orthogonal (evaluative - frozen is
+4.63 +- 0.46 PR static vs +4.89 +- 0.73 drifting). That is a controlled choice, not a shortcut.

THE INSTRUMENT THAT DECIDES WHETHER ANY OF IT MEANS ANYTHING
--------------------------------------------------------------
The program's central negative -- five meta-learners collapsing to plain multitask -- was
diagnosed as *"the outer loop is identical to the inner loop"*, and `fm_cotrain` capping below
the no-loop floor is the same failure in a different costume. So every arm carries a per-round
GRADER-DISAGREEMENT readout on one frozen probe: does the outer signal ever prefer a move the
dense inner signal scores worse, and by how much. Pre-registered predictions in
PREREGISTRATION.md. The one that matters: IF THE ENDOGENOUS ARM'S DISAGREEMENT SITS AT THE
HOMOGENEOUS FLOOR, THE LOOPS HAVE COLLAPSED and any gain it shows is an inner-loop gain wearing
an outer-loop label -- that diagnosis takes precedence over the headline.

The homogeneous floor is the co-trained FM against the independently-trained fresh FM: same
objective, different init and data. That is `mjc/ballistic/directed/` S1's seed-ensemble
baseline, and this is the first half of `heterogeneous_graders.md` §9, named there as the
load-bearing test of the whole frame and unbuilt for 20+ PRs.

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/endo_expansion/endo_expansion.py::endo_expansion --quick
  modal run --detach rhm/directed_sculpting/full_loop/endo_expansion/endo_expansion.py::endo_expansion \\
      --tag e4_s1 --seed 1
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
    belief_update, build_block_tables, collect_value_buffer, fm_one_step_check, make_spec,
    make_tree_probe, open_loop_beam, refresh_w_blk, root_ce, sample_states, set_world,
    snapshot_world, stochasticity_floor, train_block_fm, train_generator_channels,
    tree_depth_probe)
from rhm.directed_sculpting.full_loop.endo_expansion.endo_graders import (
    beam_belief_vs_truth, collect_rollout_moves, collect_value_moves,
    forecast_visits_per_block, grader_disagreement, grounded_candidates, metered_target,
    move_scores, value_calibration)


app = modal.App("rhm-ds-endo-expansion", image=image)

ARMS = ("frozen", "dense", "evaluative", "endo_rollout", "endo_shuffled", "endo_value",
        "alloc_kstar", "alloc_uniform", "endo_random")

# which belief_update mode each arm runs; the teacher is what differs above that
_MODE = {"frozen": "frozen", "dense": "dense"}


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
    entropy_matched_base: bool = True,
    teacher_rolls: int = 3, teacher_eps: float = 0.10, min_informative: int = 128,
    dp_blocks: int = 3, forecast_n: int = 512,
    n_dis: int = 256, dis_rolls: int = 1, n_calib: int = 512, calib_rolls: int = 2,
    n_eval: int = 1024, n_probe: int = 2000, probe_steps: int = 250,
    fresh_fm_steps: int = 3000, grade_every: int = 2, ballistic_chunk: int = 3,
    cross_world: bool = True,
    arms: str = ",".join(ARMS), tag: str = "v1", quick: bool = False,
):
    """Eight arms on one warm start; the expansion trajectory plus the disagreement instrument."""
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = fm_warm_steps = 800
        value_episodes, rounds, belief_steps = 6_000, 3, 250
        ground_states, anchor_pool, n_eval = 384, 8_000, 128
        n_probe, probe_steps, fresh_fm_steps, grade_every = 600, 80, 400, 2
        n_dis, n_calib, forecast_n, min_informative = 96, 128, 128, 16

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
    render = "mixture"
    gen = torch.Generator(device=device).manual_seed(seed)
    surface = L - 1
    # Setup is bit-identical to `../expansion.py` (same seed offsets, same call order) so the
    # three reproduced arms are a back-compat check on the port, not a fresh baseline. The world
    # never moves here -- `attach_drift` runs only to place W0 at the walk's stationary law, the
    # entropy-matched anchor, so this run and the published one start in exchangeable worlds.
    attach_drift(layout, "tree", [surface], drift_kappa, drift_kl, seed=seed + 601,
                 calibrate="event", event_steps=drift_steps_per_round,
                 init="stationary" if entropy_matched_base else "uniform", frozen=True)
    sigma = dict(layout["tree"]["drift_calibrated_sigma"])
    theta0 = (stationary_theta(layout["tree"]["rules"], [surface], drift_kappa, sigma,
                               seed=seed + 602) if entropy_matched_base else None)
    layout["tree"]["drift"]["theta"] = (np.zeros_like(layout["tree"]["drift"]["theta"])
                                        if theta0 is None else theta0)
    refresh_w_blk(tb, layout, device)
    W0 = snapshot_world(tb)

    presteps = edit_budget // 2
    teacher_horizon = max(1, edit_budget - presteps - 1)   # candidate move + horizon == budget
    calib_horizon = max(1, edit_budget - presteps)         # what V(z) is actually predicting
    n_tree_blocks = int(tb["tree_blocks"].numel())
    print(f"E4 endogenous expansion: L={L}, T={T} tokens / {n_blocks} blocks "
          f"({n_tree_blocks} tree), rounds={rounds}, static world, device={device}\n"
          f"  teacher: horizon={teacher_horizon} rolls={teacher_rolls} eps={teacher_eps}; "
          f"metered DP budget {dp_blocks}/{n_blocks} blocks; arms={arm_list}")

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
    del cfg, rts, suc

    fm0 = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    train_block_fm(fm0, controller0, generator, layout, tb, n_steps=fm_warm_steps,
                   batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                   lr=1e-3, seed=seed + 51, render=render, gen=gen, device=device)

    x_eval, r_eval = sample_states(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 42, device=device, render=render,
                                   gen=gen)
    probe_leaves, level_feats, _ = make_tree_probe(layout, tb, seed + 43)

    worlds = {"base": W0}
    if cross_world:
        _th = (stationary_theta(layout["tree"]["rules"], [surface], drift_kappa, sigma,
                                seed=seed + 9001) if entropy_matched_base else None)
        layout["tree"]["drift"] = make_drift_state(layout["tree"]["rules"], [surface], None,
                                                   seed=seed + 9001, theta0=_th)
        layout["tree"]["drift_sigma"] = {surface: 0.0}
        layout["tree"]["drift_kappa"] = 0.0
        refresh_w_blk(tb, layout, device)
        worlds["novel"] = snapshot_world(tb)
    layout["tree"]["drift"] = make_drift_state(layout["tree"]["rules"], [surface], None,
                                               seed=seed + 601, theta0=theta0)
    layout["tree"]["drift_sigma"], layout["tree"]["drift_kappa"] = {surface: 0.0}, 0.0
    set_world(tb, W0)

    x_floor, _ = sample_states(layout, tb, generator, n=512, n_corrupt=n_corrupt,
                               presteps=presteps, seed=seed + 44, device=device,
                               render=render, gen=gen)
    # FROZEN probes for the two new instruments. Same states every round for every arm, so the
    # readouts move only because the models move -- the ladder's own discipline, and the reason
    # its first `e` tap lost to state-draw variance rather than to bias.
    x_dis, r_dis = sample_states(layout, tb, generator, n=n_dis, n_corrupt=n_corrupt,
                                 presteps=presteps, seed=seed + 45, device=device,
                                 render=render, gen=gen)
    x_cal, r_cal = sample_states(layout, tb, generator, n=n_calib, n_corrupt=n_corrupt,
                                 presteps=presteps, seed=seed + 46, device=device,
                                 render=render, gen=gen)
    x_fc, r_fc = sample_states(layout, tb, generator, n=forecast_n, n_corrupt=n_corrupt,
                               presteps=0, seed=seed + 47, device=device, render=render, gen=gen)

    for wn, W in worlds.items():
        set_world(tb, W)
        blks = tb["tree_blocks"]
        kk = blks[torch.randint(0, len(blks), (x_floor.shape[0],), device=device)]
        print(f"  world '{wn}': tree stochasticity floor = "
              f"{stochasticity_floor(controller0, generator, tb, x_floor, kk, render=render, gen=gen, acted_only=True):.4f}")
    set_world(tb, W0)
    depth0 = tree_depth_probe(controller0, probe_leaves, level_feats, layout, tb, device,
                              probe_steps=probe_steps)
    print("Round-0 belief: " + " ".join(f"{k}={val:.3f}" for k, val in depth0.items()))

    common = dict(
        controller0=controller0, fm0=fm0, value0=value0, generator=generator, layout=layout,
        tb=tb, BlockFM=BlockFM, x_eval=x_eval, r_eval=r_eval, probe_leaves=probe_leaves,
        level_feats=level_feats, worlds=worlds, x_floor=x_floor, x_dis=x_dis, r_dis=r_dis,
        x_cal=x_cal, r_cal=r_cal, x_fc=x_fc, r_fc=r_fc, seed=seed, state_dim=state_dim,
        n_blocks=n_blocks, rounds=rounds, belief_steps=belief_steps,
        ground_states=ground_states, anchor_pool=anchor_pool, batch_size=batch_size,
        n_corrupt=n_corrupt, edit_budget=edit_budget, presteps=presteps,
        teacher_horizon=teacher_horizon, teacher_rolls=teacher_rolls, teacher_eps=teacher_eps,
        min_informative=min_informative, calib_horizon=calib_horizon, calib_rolls=calib_rolls,
        dis_rolls=dis_rolls, dp_blocks=dp_blocks, lam_fm=lam_fm, lam_plan=lam_plan, tau=tau,
        probe_steps=probe_steps, fresh_fm_steps=fresh_fm_steps, grade_every=grade_every,
        ballistic_chunk=ballistic_chunk, render=render, device=device)

    results = {}
    for arm in arm_list:
        print(f"\n{'#' * 72}\n# ARM: {arm}\n{'#' * 72}")
        results[arm] = _run_arm(arm, **common)
        volume.commit()

    print(f"\n{'=' * 78}\n=== SUMMARY -- can an endogenous evaluative grader expand? ===\n"
          f"{'=' * 78}")
    print(f"round-0 belief: PR={depth0['PR']:.2f}  "
          + " ".join(f"{k}={depth0[k]:.3f}" for k in depth0 if k != "PR"))
    for arm in arm_list:
        rr = results[arm]["rounds"]
        print(f"\n[{arm}]")
        print("  PR       : " + "  ".join(f"r{r['round']}={r['depth']['PR']:.1f}" for r in rr))
        print(f"  d{L}       : " + "  ".join(f"r{r['round']}={r['depth'][f'd{L}']:.3f}"
                                             for r in rr))
        graded = [r for r in rr if "ballistic_w16" in r]
        if graded:
            print("  ballistic: " + "  ".join(f"r{r['round']}={r['ballistic_w16']:.3f}"
                                              for r in graded))
            print("  freshtop1: " + "  ".join(
                f"r{r['round']}={r['fresh_fm']['value_top1_agree']:.3f}" for r in graded))
        tt = [r["teacher"] for r in rr if r.get("teacher")]
        if tt:
            print(f"  teacher  : agree_dp={np.mean([t['agree_dp'] for t in tt]):.3f} "
                  f"tree_share={np.mean([t['tree_share'] for t in tt]):.3f} "
                  f"informative={np.mean([t['informative_frac'] for t in tt]):.3f} "
                  f"dp_no_improve={np.mean([t['dp_no_improving_frac'] for t in tt]):.3f} "
                  f"| meter mat={rr[-1]['meter']['materialisations']:.3g} "
                  f"dp={rr[-1]['meter']['dp_calls']:.3g}")
        cal = [r["value_calib"] for r in rr]
        print(f"  wirehead : bias r1={cal[0]['bias']:+.3f} -> r{rr[-1]['round']}="
              f"{cal[-1]['bias']:+.3f}  (reported {cal[-1]['reported_mean']:.3f} vs realised "
              f"{cal[-1]['realised_mean']:.3f})")
        dis = [r["disagree"] for r in rr]
        for key in ("roll_vs_fm_pred", "dp_vs_fm_pred", "fm_pred_vs_fm_pred_fresh"):
            vals = [d[key] for d in dis if key in d]
            if vals:
                print(f"  disagree {key:26s}: top1={np.mean([x['top1_disagree'] for x in vals]):.3f} "
                      f"rank_corr={np.mean([x['rank_corr'] for x in vals]):+.3f} "
                      f"cost={np.mean([x['cost'] for x in vals]):.3f}")

    metrics = {
        "config": {"v": v, "s": s, "tree_depth": tree_depth, "total_len": T,
                   "n_blocks": n_blocks, "n_tree_blocks": n_tree_blocks,
                   "edit_budget": edit_budget, "presteps": presteps, "rounds": rounds,
                   "belief_steps": belief_steps, "ground_states": ground_states,
                   "teacher_horizon": teacher_horizon, "teacher_rolls": teacher_rolls,
                   "teacher_eps": teacher_eps, "min_informative": min_informative,
                   "dp_blocks": dp_blocks, "forecast_n": forecast_n, "n_dis": n_dis,
                   "dis_rolls": dis_rolls, "n_calib": n_calib, "calib_rolls": calib_rolls,
                   "world": "static", "arms": arm_list, "render": render, "seed": seed,
                   "entropy_matched_base": entropy_matched_base,
                   "sigma": {str(k): float(x) for k, x in sigma.items()}},
        "belief_round0": depth0, "results": results,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/endo_expansion_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


def _run_arm(arm, *, controller0, fm0, value0, generator, layout, tb, BlockFM, x_eval, r_eval,
             probe_leaves, level_feats, worlds, x_floor, x_dis, r_dis, x_cal, r_cal, x_fc, r_fc,
             seed, state_dim, n_blocks, rounds, belief_steps, ground_states, anchor_pool,
             batch_size, n_corrupt, edit_budget, presteps, teacher_horizon, teacher_rolls,
             teacher_eps, min_informative, calib_horizon, calib_rolls, dis_rolls, dp_blocks,
             lam_fm, lam_plan, tau, probe_steps, fresh_fm_steps, grade_every, ballistic_chunk,
             render, device):
    """One arm. Forks from the SAME controller/FM/value with a matched update budget; only the
    teacher (and, for `alloc_*`, where the teacher is allowed to look) differs.

    Every arm pays the SAME instrument charge -- the disagreement probe and the value-calibration
    rollout run every round regardless of whether the arm uses them -- so the arms differ in
    their loss and in nothing else.

    The global torch RNG is re-seeded here, identically for every arm. Without it an arm inherits
    whatever state the previous arm left behind (`belief_update`'s minibatch draws and
    `_reveal`'s block order both use the global stream), so arms would differ by minibatch
    ordering as well as by teacher, AND running a subset of arms would not reproduce the run that
    included all of them. Re-seeding buys both: matched minibatch order at every arm's start, and
    an arm's trajectory that depends on `seed` alone -- which is what lets the eight arms be
    split across parallel jobs.
    """
    import torch

    torch.manual_seed(seed + 12_345)
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
    alloc_rng = np.random.default_rng(seed + 992)
    n_tree = int(tb["tree_blocks"].numel())
    tree_lo = int(tb["tree_blocks"].min())
    meter = {"materialisations": 0.0, "dp_calls": 0.0}
    rows = []
    fresh = None                     # kept between grading rounds: the homogeneous partner

    for rnd in range(1, rounds + 1):
        anchor = sample_pool(layout, anchor_pool, seed + 6000 + rnd)
        anchor_leaves = torch.from_numpy(anchor["leaves"])
        anchor_roots = torch.from_numpy(anchor["roots"].astype(np.int64))
        gx, gr = sample_states(layout, tb, generator, n=ground_states, n_corrupt=n_corrupt,
                               presteps=presteps, seed=seed + 7000 + rnd, device=device,
                               render=render, gen=gen)

        # ---- the teacher -------------------------------------------------------------
        gk = px = pr = None
        tinfo = None
        if mode == "evaluative":
            # ONE DP pass per round, in EVERY teacher arm, serving three purposes: the
            # `evaluative` target, the metered arms' targets, and the `agree_dp` diagnostic the
            # endogenous arms are scored against. Only `evaluative`/`alloc_*` ever put it in a
            # loss. Computing it for every arm is the ladder's "both estimators every round for
            # every arm" discipline: the comparison is then free of a state-draw confound, and
            # the instrument charge is identical across arms.
            cand, d_cur_v = grounded_candidates(layout, tb, generator, gx, gr, render=render,
                                                gen=gen, device=device)
            kstar_np = cand.argmin(axis=0).astype(np.int64)   # published lowest-index tie-break
            d_cur, d_best = float(d_cur_v.mean()), float(cand.min(axis=0).mean())
            no_improving = float((cand.min(axis=0) >= d_cur_v).mean())
            informative = np.ones(gx.shape[0], dtype=bool)
            allowed = None

            if arm == "evaluative":
                gk = torch.from_numpy(kstar_np).to(device)
                k_hat = kstar_np
                meter["materialisations"] += ground_states * n_tree
                meter["dp_calls"] += ground_states * (n_tree + 1)

            elif arm in ("endo_rollout", "endo_shuffled"):
                r_hat, k_hat, informative = collect_rollout_moves(
                    controller, generator, layout, tb, gx, gr, horizon=teacher_horizon,
                    epsilon=teacher_eps, n_roll=teacher_rolls, render=render, gen=gen,
                    device=device, rng=rng)
                meter["materialisations"] += (ground_states * n_blocks * teacher_rolls
                                              * (1 + teacher_horizon * n_blocks))

            elif arm == "endo_random":
                # The floor of the control ladder, added after the first full run: a UNIFORM
                # random block, carrying no information of any kind. `endo_shuffled` preserves
                # the rollout teacher's marginal over blocks (tree share 0.74), so if PR moves
                # under it, the remaining question is whether PR is responding to that marginal
                # or to the loss SHAPE alone. This arm answers it -- tree share 8/14 = 0.571 and
                # agreement with `k*` at chance, 1/14 = 0.071, both by construction.
                k_hat = rng.integers(0, n_blocks, size=gx.shape[0]).astype(np.int64)

            elif arm == "endo_value":
                k_v, _ = collect_value_moves(controller, generator, value, tb, gx, gr,
                                             render=render, gen=gen, device=device)
                k_hat = k_v.cpu().numpy()
                meter["materialisations"] += ground_states * n_blocks

            elif arm in ("alloc_kstar", "alloc_uniform"):
                if arm == "alloc_kstar":
                    # the ENDOGENOUS allocator: roll the FM open-loop under the greedy planner
                    # and spend the DP budget where the imagined plan actually goes. Zero
                    # environment interaction, zero labels -- the ladder's `p` tap at block
                    # granularity.
                    vis = forecast_visits_per_block(fm, value, controller, tb, x_fc, r_fc,
                                                    budget=edit_budget, device=device)
                    allowed = np.argsort(-vis)[:dp_blocks]
                else:
                    allowed = alloc_rng.choice(n_blocks, size=dp_blocks, replace=False)
                k_hat, informative = metered_target(cand, d_cur_v, allowed, rng)
                meter["materialisations"] += ground_states * dp_blocks
                meter["dp_calls"] += ground_states * (dp_blocks + 1)

            if arm != "evaluative":
                # the plan term trains only where the teacher has an opinion; the dense term
                # still sees the full pool, so arms stay matched on everything but the target.
                if informative.sum() < min_informative:
                    informative = np.ones(gx.shape[0], dtype=bool)
                sel = np.nonzero(informative)[0]
                tgt = k_hat[sel]
                if arm == "endo_shuffled":
                    # permute targets ACROSS STATES, not across blocks: the marginal
                    # distribution over which block gets taught is preserved exactly, so the
                    # only thing destroyed is whether the target is right for THIS state. The
                    # harder of the two possible controls, and the one that isolates content.
                    tgt = tgt[rng.permutation(len(tgt))]
                px = gx[torch.from_numpy(sel).to(device)]
                pr = gr[torch.from_numpy(sel).to(device)]
                gk = torch.from_numpy(tgt).to(device)

            tinfo = {"agree_dp": float((k_hat == kstar_np).mean()),
                     "tree_share": float(((k_hat >= tree_lo)
                                          & (k_hat < tree_lo + n_tree)).mean()),
                     "informative_frac": float(informative.mean()),
                     "dp_no_improving_frac": no_improving,
                     "dstar_cur": d_cur, "dstar_best_move": d_best}
            if allowed is not None:
                tinfo["allowed_blocks"] = [int(b) for b in allowed]
                tinfo["allowed_tree_frac"] = float(np.mean(
                    [(tree_lo <= b < tree_lo + n_tree) for b in allowed]))

        info = belief_update(mode, controller, fm, value, generator, layout, tb,
                             train_leaves=anchor_leaves, train_roots=anchor_roots,
                             gstates=gx, groots=gr, gkstar=gk, n_steps=belief_steps,
                             batch_size=batch_size, lr=3e-4, lam_fm=lam_fm, lam_plan=lam_plan,
                             tau=tau, n_corrupt=n_corrupt, edit_budget=edit_budget,
                             render=render, gen=gen, device=device, seed=seed + rnd,
                             pstates=px, proots=pr)

        depth = tree_depth_probe(controller, probe_leaves, level_feats, layout, tb, device,
                                 probe_steps=probe_steps)
        row = {"round": rnd, "depth": depth,
               "root_ce": root_ce(controller, anchor_leaves[:4096], anchor_roots[:4096], device),
               "root_acc": info.get("root_acc"), "plan_acc": info.get("plan_acc"),
               "teacher": tinfo, "meter": dict(meter)}

        # ---- the wireheading readout (every arm, every round) ------------------------
        row["value_calib"] = value_calibration(controller, generator, value, layout, tb,
                                               x_cal, r_cal, horizon=calib_horizon,
                                               epsilon=teacher_eps, n_roll=calib_rolls,
                                               render=render, gen=gen, device=device)

        # ---- the grader-disagreement instrument (every arm, every round) -------------
        sc = move_scores(controller, generator, fm, value, layout, tb, x_dis, r_dis,
                         horizon=teacher_horizon, epsilon=teacher_eps, n_roll=dis_rolls,
                         render=render, gen=gen, device=device, fm_fresh=fresh)
        row["disagree"] = grader_disagreement(sc)
        row["disagree_has_fresh"] = fresh is not None

        if rnd % grade_every == 0 or rnd == rounds:
            fresh = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
            train_block_fm(fresh, controller, generator, layout, tb, n_steps=fresh_fm_steps,
                           batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                           lr=1e-3, seed=seed + 8000 + rnd, render=render, gen=gen,
                           device=device)
            fresh.eval()
            own = snapshot_world(tb)
            row["xworld"] = {}
            for wn, W in worlds.items():
                set_world(tb, W)
                blks = tb["tree_blocks"]
                kk = blks[torch.randint(0, len(blks), (x_floor.shape[0],), device=device)]
                succ, x_fin, r_fin = open_loop_beam(
                    controller, generator, fresh, value, tb, layout, x_eval, r_eval,
                    budget=edit_budget, beam_width=16, render=render, gen=gen, device=device,
                    chunk_len=ballistic_chunk, return_final=True)
                row["xworld"][wn] = {
                    "ballistic_w16": succ,
                    "beam_grades": beam_belief_vs_truth(controller, layout, x_fin, r_fin),
                    "fm": fm_one_step_check(fresh, value, controller, generator, tb, x_eval,
                                            r_eval, render=render, gen=gen, device=device),
                    "tree_floor": stochasticity_floor(controller, generator, tb, x_floor, kk,
                                                      render=render, gen=gen, acted_only=True)}
            set_world(tb, own)
            row["fresh_fm"] = row["xworld"]["base"]["fm"]
            row["ballistic_w16"] = row["xworld"]["base"]["ballistic_w16"]
            row["beam_grades"] = row["xworld"]["base"]["beam_grades"]

        rows.append(row)
        d = row["disagree"]
        print(f"  r{rnd:02d} PR={depth['PR']:.2f} "
              + " ".join(f"{k}={val:.3f}" for k, val in depth.items() if k != "PR")
              + f" rootCE={row['root_ce']:.4f}"
              + (f" ballistic={row['ballistic_w16']:.3f}" if "ballistic_w16" in row else "")
              + (f" | teach agree_dp={tinfo['agree_dp']:.2f} tree={tinfo['tree_share']:.2f}"
                 f" info={tinfo['informative_frac']:.2f}" if tinfo else "")
              + f" | vbias={row['value_calib']['bias']:+.3f}"
              + (f" dis(roll,fm)={d['roll_vs_fm_pred']['top1_disagree']:.2f}"
                 if "roll_vs_fm_pred" in d else ""))
    return {"rounds": rows}


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1", seed: int = 1, arms: str = ",".join(ARMS)):
    endo_expansion.remote(quick=quick, tag=tag, seed=seed, arms=arms)
