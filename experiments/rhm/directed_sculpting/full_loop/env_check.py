"""Certify the multi-channel sculpting ENVIRONMENT before the ladder runs on it.

`verify_distractors.py` certified the DGP at the token-CE level; its own caveat says the
"FM-level LP that the real loop reads is a separate check". This is that check, plus the
two knobs the loop's design turns on. Six properties:

  E1 back-compat        `render="canon"` reproduces the old deterministic edit exactly
                        (same (x,k) twice -> bit-identical result), so every prior sculpting
                        result stays reachable through this code path.
  E2 irreducible floor  the per-channel FLOOR of the block FM's normalised error, measured
                        directly by redrawing the same (x,k) twice. Under mixture rendering
                        an edit is stochastic (the synonym is drawn from the drifted mixture),
                        so no FM can beat that floor -- and the ladder is only meaningful if
                        the floor ORDERS the channels noise > struct > tree. This is P2's
                        "exact irreducibility" restated in the FM's own currency.
  E3 learnability       does a trained FM actually get down to each channel's floor? The
                        quantity the `lprog` tap reads is floor-relative headroom, not raw
                        error, so a channel with a high floor AND no headroom is the noisy TV.
  E4 task solvability   DP `d*`, fraction solvable within `edit_budget`, start success.
                        Open item #2 of the node README predicts budget 9-10 at L=5.
  E5 control horizon    ballistic (open-loop, chunked) beam success vs reactive, swept over
                        the commit length. Picks `chunk_len`: the imagined rollout is only
                        veridical to ~6 steps, so a fully-ballistic 10-step commitment grades
                        rollout drift rather than FM quality (a first smoke returned 0.000).
  E6 relevance          the learned value's |ΔV| per channel against the exact DP Δd* per
                        channel. The direct form of "does the value devalue distractors".

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/env_check.py::env_check --quick
  modal run --detach rhm/directed_sculpting/env_check.py::env_check --tag l5_v1
"""

import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import make_layout, possible_set_success, sample_pool
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    _block_state_chunked, blocks_of_channel, build_block_tables, closed_loop_beam,
    collect_value_buffer, fm_error, fm_one_step_check, forecast_visits,
    ground_truth_relevance, make_spec, open_loop_beam, regenerate_block, sample_states,
    stochasticity_floor, train_block_fm, train_generator_channels,
    transition_targets, value_relevance_check)


app = modal.App("rhm-ds-env-check", image=image)


# `_stochasticity_floor` now lives in channel_env as `stochasticity_floor` (the
# expansion cross-world arm needs it too); aliased here so this entrypoint reads unchanged.
_stochasticity_floor = stochasticity_floor


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=14400, memory=32768)
def env_check(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_steps: int = 12_000, batch_size: int = 256,
    n_corrupt: int = 3, edit_budget: int = 10, explore_eps: float = 0.3,
    n_eval: int = 512, n_floor: int = 1024, renders: str = "canon,mixture",
    chunk_sweep: str = "1,2,3,5,10", tree_depth: int = 5, struct_depths: str = "3,3",
    struct_ms: str = "2,4", noise_blocks: str = "2,2", fm_layer: int = 2, fm_head: int = 4,
    tag: str = "v1", quick: bool = False,
):
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 1200
        value_episodes, n_eval, n_floor = 8_000, 256, 384

    render_list = [r for r in renders.split(",") if r]
    chunks = [int(c) for c in chunk_sweep.split(",")]
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
    names, T, n_blocks = tb["channel_names"], layout["total_len"], tb["n_blocks"]
    kinds = [ch["kind"] for ch in layout["channels"]]
    print(f"Env check: T={T} tokens / {n_blocks} blocks / {len(names)} channels, "
          f"budget={edit_budget}, c={n_corrupt}, device={device}")
    results = {"config": {"v": v, "s": s, "edit_budget": edit_budget, "n_corrupt": n_corrupt,
                          "fm_steps": fm_steps, "renders": render_list, "chunks": chunks,
                          "tree_depth": tree_depth, "total_len": T, "n_blocks": n_blocks,
                          "fm_layer": fm_layer, "fm_head": fm_head},
               "channels": [{k: ch[k] for k in ("name", "kind", "depth", "m", "blk0", "blk1")}
                            for ch in layout["channels"]]}

    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, BlockFM = _build_value_head(), _build_block_fm()
    gen = torch.Generator(device=device).manual_seed(seed)

    pool = sample_pool(layout, 100_000 if not quick else 20_000, seed)
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

    x_eval, r_eval = sample_states(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 42, device=device,
                                   render="canon", gen=gen)
    x_floor, _ = sample_states(layout, tb, generator, n=n_floor, n_corrupt=n_corrupt,
                               presteps=edit_budget // 2, seed=seed + 43, device=device,
                               render="canon", gen=gen)

    # ---- E1 / E2: determinism and the per-channel irreducible floor ---------------------
    print("\n[E1/E2] edit determinism and the per-channel irreducible floor")
    floors, floors_acted = {}, {}
    for render in render_list:
        floors[render], floors_acted[render] = {}, {}
        for c in range(len(names)):
            blks = blocks_of_channel(tb, c)
            k = blks[torch.randint(0, len(blks), (n_floor,), device=device)]
            floors[render][names[c]] = _stochasticity_floor(
                controller, generator, tb, x_floor, k, render=render, gen=gen)
            floors_acted[render][names[c]] = _stochasticity_floor(
                controller, generator, tb, x_floor, k, render=render, gen=gen, acted_only=True)
        print(f"  render={render:8s} all-blocks " + "  ".join(
            f"{n}={floors[render][n]:.4f}" for n in names))
        print(f"  render={render:8s} acted      " + "  ".join(
            f"{n}={floors_acted[render][n]:.4f}" for n in names))
    results["e2_irreducible_floor"] = floors
    results["e2_irreducible_floor_acted"] = floors_acted
    # only the GRAMMAR channels are deterministic under `canon`; noise blocks redraw iid by
    # design, so including them in the determinism check would fail it by construction.
    gram = [n for n, kd in zip(names, kinds) if kd != "noise"]
    det = max(floors["canon"][n] for n in gram) if "canon" in floors else None
    results["e1_canon_deterministic"] = {"max_floor_under_canon_grammar_channels": det,
                                         "pass": None if det is None else bool(det < 1e-6)}

    # ---- E4: task solvability ----------------------------------------------------------
    rel = ground_truth_relevance(layout, tb, generator, x_eval, r_eval, render="canon",
                                 gen=gen, device=device)
    from rhm.rhm_channels import dp_cost
    dstar = dp_cost(layout, x_eval.cpu().numpy(), r_eval.cpu().numpy())
    start_succ = float(possible_set_success(layout, x_eval.cpu().numpy(),
                                            r_eval.cpu().numpy()).mean())
    results["e4_task"] = {"dstar_mean": float(dstar.mean()),
                          "frac_solvable_token_budget": float((dstar <= n_corrupt * s).mean()),
                          "frac_solvable_edit_budget": float((dstar <= edit_budget).mean()),
                          "start_possible_set_success": start_succ}
    print(f"\n[E4] d* mean={dstar.mean():.2f}; solvable within {n_corrupt*s} token edits="
          f"{(dstar <= n_corrupt*s).mean():.3f}, within budget {edit_budget}="
          f"{(dstar <= edit_budget).mean():.3f}; start success={start_succ:.3f}")
    print("  best Δd* per channel: " + "  ".join(
        f"{n}={rel['best_dstar_gain'][i]:+.3f}" for i, n in enumerate(names)))
    results["e6_ground_truth_relevance"] = {
        "mean_dstar_gain": {names[i]: float(rel["mean_dstar_gain"][i]) for i in range(len(names))},
        "best_dstar_gain": {names[i]: float(rel["best_dstar_gain"][i]) for i in range(len(names))}}

    # ---- value (shared across render conditions; collected under `canon`) --------------
    print(f"\nCollecting value data ({value_episodes} rollouts)")
    cfg, rts, suc = collect_value_buffer(
        controller, generator, layout, tb, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render="canon", gen=gen, device=device)
    print(f"  buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    value = ValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    value.eval()
    for p in value.parameters():
        p.requires_grad_(False)
    results["value_buffer_success"] = float(suc.mean())
    del cfg, rts, suc

    # ---- E3 / E5 / E6: per render mode -------------------------------------------------
    results["per_render"] = {}
    for render in render_list:
        print(f"\n{'#'*68}\n# render = {render}\n{'#'*68}")
        fm = BlockFM(state_dim, n_blocks, n_head=fm_head, n_layer=fm_layer).to(device)
        train_block_fm(fm, controller, generator, layout, tb, n_steps=fm_steps,
                       batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                       lr=1e-3, seed=seed + 51, render=render, gen=gen, device=device)

        per_ch = {}
        for c in range(len(names)):
            blks = blocks_of_channel(tb, c)
            k = blks[torch.randint(0, len(blks), (n_floor,), device=device)]
            z, target = transition_targets(controller, generator, x_floor, k, tb,
                                           render=render, gen=gen)
            nmse, raw = fm_error(fm, z, k, target)
            fl = floors[render][names[c]]
            nmse_a, _ = fm_error(fm, z, k, target, acted_only=True)
            fl_a = floors_acted[render][names[c]]
            per_ch[names[c]] = {"nmse": nmse, "raw_mse": raw, "floor": fl,
                                "nmse_acted": nmse_a, "floor_acted": fl_a,
                                "headroom_used": (1.0 - nmse) / max(1.0 - fl, 1e-9),
                                "headroom_used_acted": (1.0 - nmse_a) / max(1.0 - fl_a, 1e-9)}
        print("[E3] FM error/floor  (all blocks): " + "  ".join(
            f"{n}={per_ch[n]['nmse']:.3f}/{per_ch[n]['floor']:.3f}" for n in names))
        print("[E3] FM error/floor (acted block): " + "  ".join(
            f"{n}={per_ch[n]['nmse_acted']:.3f}/{per_ch[n]['floor_acted']:.3f}" for n in names))

        check = fm_one_step_check(fm, value, controller, generator, tb, x_eval, r_eval,
                                  render=render, gen=gen, device=device)
        print(f"[E3] one-step: delta_cos={check['delta_cos']:.3f} "
              f"top1={check['value_top1_agree']:.3f} rank_corr={check['value_rank_corr']:.3f}")

        visits, dvalue = forecast_visits(fm, value, controller, tb, x_eval, r_eval,
                                         budget=edit_budget, device=device)
        print("[E6] forecast visits: " + "  ".join(f"{n}={visits[i]:.3f}"
                                                   for i, n in enumerate(names)))
        print("[E6] value |ΔV|:      " + "  ".join(f"{n}={dvalue[i]:.4f}"
                                                   for i, n in enumerate(names)))
        vrel = value_relevance_check(value, controller, generator, layout, tb, x_eval, r_eval,
                                     render=render, gen=gen, device=device)
        print("[E6] value signed ΔV: " + "  ".join(
            f"{n}={vrel['mean_signed_dvalue'][n]:+.4f}" for n in names))
        print("[E6] value top-1 move share: " + "  ".join(
            f"{n}={vrel['top1_share'][n]:.3f}" for n in names)
            + f"  | value-vs-DP rank corr = {vrel['value_vs_dp_rank_corr']:+.3f}")

        ctrl = {"reactive_w16": closed_loop_beam(controller, generator, fm, value, tb, layout,
                                                 x_eval, r_eval, budget=edit_budget,
                                                 beam_width=16, render=render, gen=gen,
                                                 device=device)}
        for ch_len in chunks:
            for w in (1, 16):
                ctrl[f"ballistic_c{ch_len}_w{w}"] = open_loop_beam(
                    controller, generator, fm, value, tb, layout, x_eval, r_eval,
                    budget=edit_budget, beam_width=w, render=render, gen=gen, device=device,
                    chunk_len=ch_len)
        print("[E5] control: " + "  ".join(f"{k}={val:.3f}" for k, val in ctrl.items()))

        results["per_render"][render] = {
            "fm_per_channel": per_ch, "fm_one_step": check,
            "forecast_visits": {names[i]: float(visits[i]) for i in range(len(names))},
            "value_sensitivity": {names[i]: float(dvalue[i]) for i in range(len(names))},
            "value_relevance": vrel, "control": ctrl}
        del fm

    # ---- verdicts ----------------------------------------------------------------------
    print(f"\n{'='*70}\n--- verdicts ---")
    if results["e1_canon_deterministic"]["pass"] is not None:
        print(f"E1 canon deterministic : "
              f"{'PASS' if results['e1_canon_deterministic']['pass'] else 'FAIL'} "
              f"(max floor {det:.2e})")
    for render in render_list:
        fl = floors[render]
        noise = max(fl[n] for n, kd in zip(names, kinds) if kd == "noise")
        struct = max(fl[n] for n, kd in zip(names, kinds) if kd == "struct")
        tree = fl[names[kinds.index("tree")]]
        ordered = noise > struct and struct >= tree
        print(f"E2 floor ordering [{render}]: noise {noise:.3f} > struct {struct:.3f} "
              f">= tree {tree:.3f} -> {'PASS' if ordered else 'FAIL'}")
        pc = results["per_render"][render]["fm_per_channel"]
        print(f"E3 headroom used  [{render}]: " + "  ".join(
            f"{n}={pc[n]['headroom_used']:+.2f}" for n in names))

    results["elapsed_seconds"] = time.time() - started
    out_dir = f"{DATA_DIR}/directed_sculpting/env_check_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({results['elapsed_seconds']:.0f}s)")
    return results


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1"):
    env_check.remote(quick=quick, tag=tag)


# --------------------------------------------------------------------------- #
# Bisect: where did the block FM's fidelity go?
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=14400, memory=32768)
def fm_bisect(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96, tree_depth: int = 4,
    n_corrupt: int = 3, edit_budget: int = 6, batch_size: int = 256,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, n_eval: int = 512,
    explore_eps: float = 0.3, tag: str = "v1", quick: bool = False,
):
    """The multi-channel block FM reaches delta_cos ~0.15 where the single-channel reference
    reaches 0.49. Four conditions isolate which change cost it, one variable at a time:

      reference   the LITERAL `rhm_sculpt_latent` pipeline on the plain L=4 task. Establishes
                  that 0.49 is reproducible here at all, and is the back-compat anchor.
      treeonly    THIS code path on a tree-only layout (no distractor channels), canon render.
                  Same task as `reference`, different implementation -> any gap is a bug of
                  mine, not a property of the design.
      full_canon  this code path, full 5-channel layout, deterministic render. Isolates what
                  the DISTRACTOR CHANNELS cost (more blocks to model, two of them irreducible).
      full_mix    + mixture rendering. Isolates what making the edit STOCHASTIC costs.

    Reported on the metric the sculpting arc quotes (`delta_cos`, `value_top1_agree`,
    `value_rank_corr`) plus acted-block normalised error, so the comparison is like-for-like.
    """
    import torch

    from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator, _train_generator
    from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
    from rhm.rhm_sculpt_latent import (_build_block_fm, _build_rich_controller, _fm_check,
                                       _train_block_fm)
    from rhm.rhm_sculpt_planner import _collect_value_sculpt, _corrupt, _sample_pool

    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 1000
        value_episodes, n_eval = 8_000, 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()
    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, BlockFM = _build_value_head(), _build_block_fm()
    results = {}

    # ---- condition `reference`: the literal single-channel pipeline ---------------------
    print(f"\n{'#'*68}\n# reference (rhm_sculpt_latent, plain L={tree_depth})\n{'#'*68}")
    T0 = s ** tree_depth
    nb0 = T0 // s
    rules = generate_rules_distinct(v, s, tree_depth, 2, seed=0)
    bottom_map = torch.from_numpy(build_inverse_maps(rules)[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[tree_depth - 1][:, 0, :])).to(device)
    region_index, n_regions = _region_index(nb0, 1, device)
    tr_roots, tr_leaves = _sample_pool(rules, 100_000 if not quick else 20_000, s, seed)
    ctrl = Controller(v, T0, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(ctrl, torch.from_numpy(tr_leaves), torch.from_numpy(tr_roots),
                           batch_size=batch_size, n_blocks=nb0, block_size=s,
                           n_steps=controller_steps, lr=3e-4, device=device, p_full=0.5)
    g0 = Generator(v, T0, s, state_dim, n_head=4, n_layer=2, root_conditioned=False).to(device)
    _train_generator(g0, torch.from_numpy(tr_leaves), torch.from_numpy(tr_roots), bottom_map,
                     batch_size=batch_size, n_blocks=nb0, v=v, block_size=s, mask_min=1,
                     mask_max=nb0, n_steps=generator_steps, lr=3e-4, device=device)
    for mod in (ctrl, g0):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)
    cfg, rts, suc = _collect_value_sculpt(
        ctrl, g0, tr_roots, tr_leaves, canon, region_index, n_regions, rules,
        n_episodes=value_episodes, batch_size=1024, n_blocks=nb0, v=v, s=s,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
    val0 = ValueHead(state_dim, v).to(device)
    _train_value_mc(val0, ctrl, cfg, rts, suc, batch_size=512, n_steps=value_steps, lr=3e-4,
                    device=device)
    val0.eval()
    for p in val0.parameters():
        p.requires_grad_(False)
    fm_ref = BlockFM(state_dim, nb0, n_head=4, n_layer=2).to(device)
    _train_block_fm(fm_ref, ctrl, g0, tr_roots, tr_leaves, canon, region_index, n_regions,
                    n_steps=fm_steps, batch_size=batch_size, n_blocks=nb0, v=v, s=s,
                    n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3, device=device)
    fm_ref.eval()
    er, el = _sample_pool(rules, min(1024, n_eval), s, seed + 99)
    rng = np.random.default_rng(seed + 2)
    x0 = torch.from_numpy(_corrupt(el, nb0, n_corrupt, v, s, rng))
    chk = _fm_check(fm_ref, val0, ctrl, g0, x0, torch.from_numpy(er), canon, region_index,
                    n_regions, s, device)
    results["reference"] = {"value_buffer_success": float(suc.mean()), **chk}
    print(f"  [reference] delta_cos={chk['delta_cos']:.3f} top1={chk['value_top1_agree']:.3f} "
          f"rank_corr={chk['value_rank_corr']:.3f}")
    del cfg, rts, suc, ctrl, g0, val0, fm_ref

    # ---- conditions through THIS code path ---------------------------------------------
    variants = [
        ("treeonly", dict(struct_depths=(), struct_ms=(), noise_blocks=()), "canon"),
        ("full_canon", dict(struct_depths=(2, 2), struct_ms=(2, 4), noise_blocks=(1, 1)), "canon"),
        ("full_mix", dict(struct_depths=(2, 2), struct_ms=(2, 4), noise_blocks=(1, 1)), "mixture"),
    ]
    built = {}
    for name, kw, render in variants:
        key = json.dumps(kw, sort_keys=True)
        print(f"\n{'#'*68}\n# {name} (render={render})\n{'#'*68}")
        if key not in built:
            layout = make_layout(v, s, make_spec(tree_depth=tree_depth, **kw))
            tb = build_block_tables(layout, device)
            T, nb = layout["total_len"], tb["n_blocks"]
            gen = torch.Generator(device=device).manual_seed(seed)
            pool = sample_pool(layout, 100_000 if not quick else 20_000, seed)
            tl = torch.from_numpy(pool["leaves"])
            tr = torch.from_numpy(pool["roots"].astype(np.int64))
            c = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
            _train_edit_controller(c, tl, tr, batch_size=batch_size, n_blocks=nb, block_size=s,
                                   n_steps=controller_steps, lr=3e-4, device=device, p_full=0.5)
            g = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
            train_generator_channels(g, tl, tb, batch_size=batch_size, n_steps=generator_steps,
                                     lr=3e-4, device=device)
            for mod in (c, g):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad_(False)
            cf, rt, sc = collect_value_buffer(c, g, layout, tb, n_episodes=value_episodes,
                                              batch_size=1024, n_corrupt=n_corrupt,
                                              budget=edit_budget, epsilon=explore_eps,
                                              seed=seed + 31, render="canon", gen=gen,
                                              device=device)
            val = ValueHead(state_dim, v).to(device)
            _train_value_mc(val, c, cf, rt, sc, batch_size=512, n_steps=value_steps, lr=3e-4,
                            device=device)
            val.eval()
            for p in val.parameters():
                p.requires_grad_(False)
            xe, re_ = sample_states(layout, tb, g, n=n_eval, n_corrupt=n_corrupt, presteps=0,
                                    seed=seed + 42, device=device, render="canon", gen=gen)
            built[key] = (layout, tb, c, g, val, xe, re_, gen, float(sc.mean()))
            del cf, rt, sc
        layout, tb, c, g, val, xe, re_, gen, vbs = built[key]
        fm = BlockFM(state_dim, tb["n_blocks"], n_head=4, n_layer=2).to(device)
        train_block_fm(fm, c, g, layout, tb, n_steps=fm_steps, batch_size=batch_size,
                       n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3, seed=seed + 51,
                       render=render, gen=gen, device=device)
        chk = fm_one_step_check(fm, val, c, g, tb, xe, re_, render=render, gen=gen,
                                device=device)
        per_ch = {}
        for ci in range(tb["n_channels"]):
            blks = blocks_of_channel(tb, ci)
            kk = blks[torch.randint(0, len(blks), (xe.shape[0],), device=device)]
            z, tgt = transition_targets(c, g, xe, kk, tb, render=render, gen=gen)
            per_ch[tb["channel_names"][ci]] = {
                "nmse": fm_error(fm, z, kk, tgt)[0],
                "nmse_acted": fm_error(fm, z, kk, tgt, acted_only=True)[0]}
        results[name] = {"n_blocks": tb["n_blocks"], "total_len": layout["total_len"],
                         "value_buffer_success": vbs, "per_channel": per_ch, **chk}
        print(f"  [{name}] blocks={tb['n_blocks']} delta_cos={chk['delta_cos']:.3f} "
              f"top1={chk['value_top1_agree']:.3f} rank_corr={chk['value_rank_corr']:.3f}")
        print("    acted-block nmse: " + "  ".join(
            f"{n}={per_ch[n]['nmse_acted']:.3f}" for n in per_ch))
        del fm

    print(f"\n{'='*70}\n--- bisect ---")
    for name in ("reference", "treeonly", "full_canon", "full_mix"):
        if name in results:
            r = results[name]
            print(f"  {name:12s} delta_cos={r['delta_cos']:.3f}  top1={r['value_top1_agree']:.3f}"
                  f"  rank_corr={r['value_rank_corr']:.3f}  "
                  f"value_buffer={r['value_buffer_success']:.3f}")

    results["elapsed_seconds"] = time.time() - started
    out_dir = f"{DATA_DIR}/directed_sculpting/fm_bisect_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({results['elapsed_seconds']:.0f}s)")
    return results
