"""E3 -- expansion vs compression: the 2x2's missing cell, run.

`ideas/adaptive_core_and_hierarchy_climb.md` §5 lays out a 2x2 and points at the one cell
nobody has run:

    |                  | fixed task                         | locally drifting task          |
    | dense grader     | terminates -- measured (PR 6.8)    | re-fits forever, no expansion  |
    | evaluative grader| expands once, then consolidates    | ** UNRUN **                    |
    |                  | (PR 6.7 -> 17.0, then 12.8 -> 10.1)|                                |

Three cells have anchors from this repo (sculpting Stage 5, `mjc/expansion` cut #5). The
fourth is the question, and it matters because *"compounding requires a moving frontier"* was
discovered three separate times here -- the ratchet arc, the latent loop, sculpt-continual --
and EVERY ONE was measured with a dense grader in the loop. The exhaustion law may be a law
about COMPRESSION, not about learning.

DESIGN -- one 2x2, two controlled variables, nothing else moving
-----------------------------------------------------------------
  grader in {dense, evaluative}   nested: `evaluative = dense + the grounded DP-best-move CE`,
                                  so `evaluative - dense` isolates GROUNDING and nothing else.
                                  `dense` is Stage 5's `fm_cotrain` (endogenous "be
                                  predictable"); `evaluative` is its `planner` rung.
  world  in {static, drift}       support-fixed OU drift on the tree's surface level. The DP
                                  `d*` is provably invariant under it, so the task the agent
                                  is graded on never moves -- what drifts is the encoding.
                                  BOTH arms start in the same world W0, a draw from the walk's
                                  own stationary law rather than the uniform point, so only
                                  the MOTION differs (see `entropy_matched_base`).

COMPARING A DRIFT ARM TO A STATIC ARM IS HARDER THAN IT LOOKS
--------------------------------------------------------------
Two stacked biases, each of which manufactures a drift advantage out of nothing:

  own-world grading   an arm graded in the world it just spent 20 OU steps fitting has an
                      advantage that is not robustness. Measured: the drift arm's apparent
                      ballistic edge is +0.024 (t=2.15) in its own world and +0.008 (t=0.71)
                      in a common one -- ~65% of it. Hence `cross_world`: every arm is graded,
                      frozen, in `base`, in `novel` (a held-out realisation no arm ever saw),
                      and in its own world, and only the first two are fair.
  the uniform anchor  `KL(w||uniform) = log m - H(w)` exactly, so a drift event destroys
                      precisely as much rendering entropy as the KL it creates: a drifted
                      world has fewer synonyms in play per edit and is MECHANICALLY easier to
                      plan in (tree stochasticity floor 0.160 vs 0.195, ~18%). Hence
                      `entropy_matched_base`.

With both removed, at 3 seeds, the drift-vs-static difference is +0.011 +- 0.046 on ballistic
and slightly NEGATIVE on plannability -- i.e. zero, with the sign flipping across seeds. The
grader-type effect is unaffected and enormous.

Matched belief-update budget, matched collection budget, matched init, identical downstream
probes. A `frozen` floor arm (root-CE only) runs in both worlds, because "did anything expand"
needs the no-loop baseline in the SAME world.

THE READOUT -- expansion, not performance
------------------------------------------
  PR        participation ratio of the per-block belief. The RHM_LATENT_LOOP "grounded target
            feeds gradient to idle capacity" signature, and the thing that moved 6.7 -> 17.0
            (a 2.6x expansion of the belief's effective dimension) in Stage 5 exactly where the
            endogenous control stayed flat at 6.8.
  d1..dL    per-level ancestor recovery. PR without depth would be dimensions recruited for
            nothing.
  TRAJECTORY, not endpoint. Stage 5 measured expand-once-then-consolidate on a fixed task; the
  claim under test is about the SHAPE of the curve over rounds. Exhaustion looks like PR
  plateauing; a value-driven expansion drive looks like it continuing to climb under drift.

Caveat carried in by construction: §"What this does not establish" warns belief-PR may be the
latent-loop's idle-capacity signature rather than a frontier opening, so PR and depth are
reported together and a PR rise with flat depth is NOT read as expansion.

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/expansion.py::expansion --quick
  modal run --detach rhm/directed_sculpting/expansion.py::expansion --tag l4_v1
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
    advance_drift, belief_update, build_block_tables, collect_grounded_moves,
    collect_value_buffer, fm_one_step_check, make_spec, make_tree_probe, open_loop_beam,
    prewarm_drift, refresh_w_blk, root_ce, sample_states, set_world, snapshot_world,
    stochasticity_floor, train_block_fm, train_generator_channels, transition_targets,
    tree_depth_probe)


app = modal.App("rhm-ds-expansion", image=image)

CONDITIONS = ("frozen_static", "dense_static", "evaluative_static",
              "frozen_drift", "dense_drift", "evaluative_drift")


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=28800, memory=32768)
def expansion(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
    noise_blocks: str = "1,1", edit_budget: int = 6, n_corrupt: int = 3,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 8_000, batch_size: int = 256,
    explore_eps: float = 0.3, rounds: int = 10, belief_steps: int = 1500,
    ground_states: int = 3072, anchor_pool: int = 40_000,
    lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0,
    drift_kappa: float = 0.05, drift_kl: float = 0.60, drift_steps_per_round: int = 20,
    drift_prewarm: int = 200, entropy_matched_base: bool = True,
    n_eval: int = 1024, n_probe: int = 2000, probe_steps: int = 250,
    fresh_fm_steps: int = 3000, grade_every: int = 2, ballistic_chunk: int = 3,
    cross_world: bool = True,
    conditions: str = ",".join(CONDITIONS), tag: str = "v1", quick: bool = False,
):
    """The grader-type x drift 2x2, read on the belief's expansion trajectory."""
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = fm_warm_steps = 800
        value_episodes, rounds, belief_steps = 6_000, 3, 250
        ground_states, anchor_pool, n_eval = 512, 8_000, 128
        n_probe, probe_steps, fresh_fm_steps, grade_every = 600, 80, 400, 2

    cond_list = [c for c in conditions.split(",") if c]
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
    # sigma calibrated on the size of ONE EVENT, not on accumulated distance from uniform --
    # "at matched drift magnitude" is a statement about events.
    attach_drift(layout, "tree", [surface], drift_kappa, drift_kl, seed=seed + 601,
                 calibrate="event", event_steps=drift_steps_per_round,
                 init="stationary" if entropy_matched_base else "uniform",
                 frozen=True)
    sigma = dict(layout["tree"]["drift_calibrated_sigma"])
    # W0: the world EVERYTHING is pre-trained in, and the world the `static` arms stay in.
    # Drawn from the walk's own stationary law rather than parked at uniform, so a static arm
    # and a drifting arm hold worlds that are exchangeable in synonym entropy. Anchored at
    # uniform instead, the static arm sits at the simplex's entropy maximum and every drifted
    # world is systematically lower-entropy -- i.e. easier to render and to plan in (~30% on
    # the FM's stochasticity floor), which a naive 2x2 would score as drift-induced robustness.
    theta0 = (stationary_theta(layout["tree"]["rules"], [surface], drift_kappa, sigma,
                               seed=seed + 602) if entropy_matched_base else None)
    layout["tree"]["drift"]["theta"] = (np.zeros_like(layout["tree"]["drift"]["theta"])
                                        if theta0 is None else theta0)
    refresh_w_blk(tb, layout, device)
    W0 = snapshot_world(tb)
    print(f"Expansion 2x2: L={L}, T={T} tokens / {n_blocks} blocks, rounds={rounds}, "
          f"drift on tree surface level {surface} (per-event KL target {drift_kl}, "
          f"sigma {sigma[surface]:.4f}, entropy_matched_base={entropy_matched_base}), "
          f"device={device}")

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

    # ---- the three grading WORLDS -------------------------------------------------------
    # An arm trained under drift is otherwise graded in its OWN drifted world while the static
    # arm is graded in the pristine one -- and those worlds are not equally hard. Drift moves
    # the synonym mixture away from uniform and `KL(w||uniform) = log m - H(w)` exactly, so a
    # drift event DESTROYS rendering entropy: fewer synonyms in play per edit, a lower
    # stochasticity floor, an easier world to plan in. Grading every arm in all three worlds
    # separates "this agent is robust" from "this world is easier", which is the whole
    # question. `novel` is a held-out OU realisation at the same sigma that no arm ever saw --
    # the actual transfer test.
    worlds = {"base": W0}
    if cross_world:
        # `novel`: an INDEPENDENT draw from the same stationary law, seen by no arm. Same
        # expected entropy as W0, so a difference here is transfer and not world difficulty.
        _th = (stationary_theta(layout["tree"]["rules"], [surface], drift_kappa, sigma,
                                seed=seed + 9001) if entropy_matched_base else None)
        layout["tree"]["drift"] = make_drift_state(layout["tree"]["rules"], [surface], None,
                                                   seed=seed + 9001, theta0=_th)
        layout["tree"]["drift_sigma"] = {surface: 0.0 if entropy_matched_base else sigma[surface]}
        layout["tree"]["drift_kappa"] = 0.0 if entropy_matched_base else drift_kappa
        if not entropy_matched_base:
            prewarm_drift(layout, np.random.default_rng(seed + 9002), drift_prewarm)
        refresh_w_blk(tb, layout, device)
        worlds["novel"] = snapshot_world(tb)
    # restore the LAYOUT to W0 too, not just `tb`: `sample_states` draws its derivations
    # through `layout[...]["drift"]`, so leaving the novel state installed would sample
    # sequences from one world and render edits in another.
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
    depth0 = tree_depth_probe(controller0, probe_leaves, level_feats, layout, tb, device,
                              probe_steps=probe_steps)
    print("Round-0 belief: " + " ".join(f"{k}={val:.3f}" for k, val in depth0.items()))

    results = {}
    for cond in cond_list:
        grader, _, world = cond.partition("_")
        print(f"\n{'#' * 72}\n# CONDITION: {cond}  (grader={grader}, world={world})\n{'#' * 72}")
        results[cond] = _run_condition(
            grader, world, controller0, fm0, value0, generator, layout, tb, BlockFM,
            sigma=sigma, theta0=theta0, surface=surface, x_eval=x_eval, r_eval=r_eval,
            probe_leaves=probe_leaves, level_feats=level_feats, worlds=worlds,
            x_floor=x_floor, seed=seed, state_dim=state_dim,
            n_blocks=n_blocks, rounds=rounds, belief_steps=belief_steps,
            ground_states=ground_states, anchor_pool=anchor_pool, batch_size=batch_size,
            n_corrupt=n_corrupt, edit_budget=edit_budget, drift_kappa=drift_kappa,
            drift_steps_per_round=drift_steps_per_round, drift_prewarm=drift_prewarm,
            lam_fm=lam_fm, lam_plan=lam_plan,
            tau=tau, probe_steps=probe_steps, fresh_fm_steps=fresh_fm_steps,
            grade_every=grade_every, ballistic_chunk=ballistic_chunk, render=render,
            device=device)

    print(f"\n{'=' * 78}\n=== SUMMARY -- the 2x2: does an evaluative grader keep expanding "
          f"under drift? ===\n{'=' * 78}")
    print(f"round-0 belief: PR={depth0['PR']:.2f}  "
          + " ".join(f"{k}={depth0[k]:.3f}" for k in depth0 if k != "PR"))
    for cond in cond_list:
        rr = results[cond]["rounds"]
        print(f"\n[{cond}]")
        print("  PR      : " + "  ".join(f"r{r['round']}={r['depth']['PR']:.1f}" for r in rr))
        deep = f"d{L}"
        print(f"  {deep} : " + "  ".join(f"r{r['round']}={r['depth'][deep]:.3f}" for r in rr))
        graded = [r for r in rr if "ballistic_w16" in r]
        if graded:
            print("  ballistic: " + "  ".join(
                f"r{r['round']}={r['ballistic_w16']:.3f}" for r in graded))
            print("  fresh-FM top1: " + "  ".join(
                f"r{r['round']}={r['fresh_fm']['value_top1_agree']:.3f}" for r in graded))
            if "xworld" in graded[-1]:
                for wn in graded[-1]["xworld"]:
                    vals = [g["xworld"][wn] for g in graded]
                    print(f"    [{wn:8s}] ballistic mean="
                          f"{float(np.mean([x['ballistic_w16'] for x in vals])):.3f}"
                          f"  top1 mean={float(np.mean([x['fm']['value_top1_agree'] for x in vals])):.3f}"
                          f"  tree floor={vals[-1]['tree_floor']:.4f}")

    metrics = {
        "config": {"v": v, "s": s, "tree_depth": tree_depth, "total_len": T,
                   "n_blocks": n_blocks, "edit_budget": edit_budget, "rounds": rounds,
                   "belief_steps": belief_steps, "drift_kl_per_seq": drift_kl,
                   "drift_kappa": drift_kappa, "drift_steps_per_round": drift_steps_per_round,
                   "conditions": cond_list, "render": render, "seed": seed,
                   "surface_level": surface,
                   "entropy_matched_base": entropy_matched_base,
                   "calibrate": "event", "sigma": {str(k): float(x) for k, x in sigma.items()}},
        "belief_round0": depth0, "results": results,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/expansion_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


def _run_condition(grader, world, controller0, fm0, value0, generator, layout, tb, BlockFM, *,
                   sigma, theta0, surface, x_eval, r_eval, probe_leaves, level_feats, worlds,
                   x_floor, seed, state_dim,
                   n_blocks, rounds, belief_steps, ground_states, anchor_pool, batch_size,
                   n_corrupt, edit_budget, drift_kappa, drift_steps_per_round, drift_prewarm, lam_fm,
                   lam_plan, tau, probe_steps, fresh_fm_steps, grade_every, ballistic_chunk,
                   render, device):
    """One cell of the 2x2. Forks from the SAME controller/FM/value with a matched update
    budget; only the loss terms and whether the world drifts differ."""
    import torch

    controller = copy.deepcopy(controller0)
    for p in controller.parameters():
        p.requires_grad_(True)
    fm = copy.deepcopy(fm0)
    value = copy.deepcopy(value0)
    for mod in (fm, value):
        for p in mod.parameters():
            p.requires_grad_(True)

    tree_ch = layout["tree"]
    # both arms START in W0; only whether the world MOVES differs (sigma/kappa 0 vs calibrated)
    tree_ch["drift"] = make_drift_state(tree_ch["rules"], [surface], None, seed=seed + 601,
                                        theta0=theta0)
    moving = world == "drift"
    tree_ch["drift_sigma"] = {surface: sigma[surface] if moving else 0.0}
    tree_ch["drift_kappa"] = drift_kappa if moving else 0.0
    for ch in layout["channels"]:
        if ch is not tree_ch:
            ch["drift"] = None
    refresh_w_blk(tb, layout, device)

    gen = torch.Generator(device=device).manual_seed(seed + 777)
    drift_rng = np.random.default_rng(seed + 602)
    if world == "drift" and drift_prewarm:
        prewarm_drift(layout, drift_rng, drift_prewarm)
        refresh_w_blk(tb, layout, device)
    rows = []

    for rnd in range(1, rounds + 1):
        kl = 0.0
        if world == "drift":
            kl = advance_drift(layout, drift_rng, drift_steps_per_round)["tree"]["kl_from_prev"]
            refresh_w_blk(tb, layout, device)

        anchor = sample_pool(layout, anchor_pool, seed + 6000 + rnd)
        anchor_leaves = torch.from_numpy(anchor["leaves"])
        anchor_roots = torch.from_numpy(anchor["roots"].astype(np.int64))
        gx, gr = sample_states(layout, tb, generator, n=ground_states, n_corrupt=n_corrupt,
                               presteps=edit_budget // 2, seed=seed + 7000 + rnd,
                               device=device, render=render, gen=gen)
        gk = d_cur = d_best = None
        if grader == "evaluative":
            gk, d_cur, d_best = collect_grounded_moves(layout, tb, generator, gx, gr,
                                                       render=render, gen=gen, device=device)
        info = belief_update(grader, controller, fm, value, generator, layout, tb,
                             train_leaves=anchor_leaves, train_roots=anchor_roots,
                             gstates=gx, groots=gr, gkstar=gk, n_steps=belief_steps,
                             batch_size=batch_size, lr=3e-4, lam_fm=lam_fm, lam_plan=lam_plan,
                             tau=tau, n_corrupt=n_corrupt, edit_budget=edit_budget,
                             render=render, gen=gen, device=device, seed=seed + rnd)

        depth = tree_depth_probe(controller, probe_leaves, level_feats, layout, tb, device,
                                 probe_steps=probe_steps)
        row = {"round": rnd, "drift_kl": kl, "depth": depth,
               "root_ce": root_ce(controller, anchor_leaves[:4096], anchor_roots[:4096], device),
               "root_acc": info.get("root_acc"), "plan_acc": info.get("plan_acc"),
               "dstar_cur": d_cur, "dstar_best_move": d_best}

        if rnd % grade_every == 0 or rnd == rounds:
            # FRESH, independent consumer: a value + FM trained from scratch on the (possibly
            # reorganised) belief. A lifted beam here is TRANSFERABLE plannability of the
            # belief rather than the co-trained pair getting better at each other -- the
            # REACHING_INTERNAL fresh-external-planner test, ported.
            fresh = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
            train_block_fm(fresh, controller, generator, layout, tb, n_steps=fresh_fm_steps,
                        batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                        lr=1e-3, seed=seed + 8000 + rnd, render=render, gen=gen, device=device)
            # graded in the arm's OWN world first (the fresh FM was trained there), then in
            # every other world with the SAME frozen system -- transfer, not retraining.
            own = snapshot_world(tb)
            row["xworld"] = {}
            for wn, W in list(worlds.items()) + [("own", own)]:
                set_world(tb, W)
                blks = tb["tree_blocks"]
                kk = blks[torch.randint(0, len(blks), (x_floor.shape[0],), device=device)]
                row["xworld"][wn] = {
                    "ballistic_w16": open_loop_beam(controller, generator, fresh, value, tb,
                                                    layout, x_eval, r_eval,
                                                    budget=edit_budget, beam_width=16,
                                                    render=render, gen=gen, device=device,
                                                    chunk_len=ballistic_chunk),
                    "fm": fm_one_step_check(fresh, value, controller, generator, tb, x_eval,
                                            r_eval, render=render, gen=gen, device=device),
                    "tree_floor": stochasticity_floor(controller, generator, tb, x_floor, kk,
                                                      render=render, gen=gen, acted_only=True)}
            set_world(tb, own)
            row["fresh_fm"] = row["xworld"]["own"]["fm"]
            row["ballistic_w16"] = row["xworld"]["own"]["ballistic_w16"]
            del fresh
        rows.append(row)
        print(f"  r{rnd:02d} KL={kl:.3f} PR={depth['PR']:.2f} "
              + " ".join(f"{k}={val:.3f}" for k, val in depth.items() if k != "PR")
              + f" rootCE={row['root_ce']:.4f}"
              + (f" ballistic={row['ballistic_w16']:.3f}" if "ballistic_w16" in row else ""))
    return {"rounds": rows}


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1"):
    expansion.remote(quick=quick, tag=tag)
