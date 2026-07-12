"""RHM Sculpting — Stage 3d: latent vs token planning under STOCHASTIC DYNAMICS.

Companion to Stage 3c (rhm_sculpt_latent_po.py, partial observability). Same north
star: find the regime where planning in latents BEATS token-space (materialize-and-
re-encode), not just matches it cheaply. Stage 3b showed equivalence-not-superiority
on the fully-observed, DETERMINISTIC task, argued to be because the token sequence is
a sufficient statistic there.

This file attacks a different lossy channel: NOISE. Make the actuator SLIPPERY -- an
edit lands as intended with prob (1-q), and with prob q "slips" to a uniformly random
feature. Now the consequence of a move is a DISTRIBUTION, not a point. The token beam's
only channel to a move's value is to MATERIALIZE ONE realization and re-encode it -- a
single noisy sample of E[value | move]. The latent beam RANKS by the FM's prediction
from the beam's true latent -- a stable, low-variance estimate that the noisy sample
cannot match. So as q rises, the token beam's SELECTION is corrupted by sampling noise
while the latent beam's stays stable. Prediction: the latent - token gap RISES with q.

This is the exact mirror of the active-query NULL: there a mean-Δ FM was USELESS because
the payoff lived in the VARIANCE the mean discards (revealing hidden content). Here the
payoff lives in the MEAN (a move's expected consequence) and the token beam is FORCED to
sample it once -- so the SAME mean-predicting instrument that was "an efficient surrogate,
not superior" under deterministic dynamics should flip to SUPERIOR under stochastic ones.
Same FM, opposite verdict, because the task's payoff structure flipped.

Design = the PO discipline: EVERY instrument (controller, generator, MC value, block FM)
is trained ONCE under the DETERMINISTIC world and FROZEN -- "an agent that learned the
world, now deployed into a noisier one." The ONLY controlled variable is the slip prob q,
injected at plan/commit time. The latent beam reuses the SAME deterministic FM (Design 1):
its intended-outcome prediction is a stable ranking; note that under a slippery actuator
you cannot control the slip, so ranking moves by their INTENDED consequence is a sound
low-variance heuristic, while the token beam samples the noisy realized consequence. At
q=0 both beams reduce EXACTLY to Stage 3b (asserted). (A follow-up "Design 2" would retrain
the FM on slippery targets so it predicts the true k-dependent expectation E[Δz|z,k] --
that can only help the latent beam further; Design 1 is the cheaper, instrument-reusing
first cut, and it reuses Stage-3b's exact FM so the flip, if real, is zero-retraining.)

NOTE: unlike PO, the achievability ceiling itself DROPS with q (a slippery world is genuinely
harder to control), so BOTH beams' raw success falls -- read the GAP (a paired, per-instance
contrast at fixed q), not the absolute levels, as the test of latent's selection advantage.

Run:
  modal run rhm/rhm_sculpt_latent_stoch.py::sculpt_latent_stoch --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_latent_stoch.py::sculpt_latent_stoch --m 2  # L=4,c=3
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, parse_success_and_heuristic
from rhm.rhm_active_query import _count_parameters
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_sculpt_planner import (
    _beam_plan,
    _collect_value_sculpt,
    _corrupt,
    _encode_chunked,
    _regenerate_chunked,
    _sample_pool,
    _success_ps,
)
from rhm.rhm_sculpt_latent import (
    _build_block_fm,
    _build_rich_controller,
    _block_state_chunked,
    _fm_check,
    _fm_chunked,
    _latent_beam,
    _train_block_fm,
)


app = modal.App("rhm-sculpt-latent-stoch", image=image)


def _stoch_regenerate(generator, x, region_blocks, canon, *, block_size, slip, v, gen):
    """Deterministic generator regen, then a SLIPPERY actuator: each acted block renders
    to the generator's intended feature with prob (1-slip), or a UNIFORMLY RANDOM feature's
    canonical tuple with prob slip. Models stochastic dynamics. slip=0 -> byte-identical to
    the deterministic _regenerate (so beams reduce exactly to Stage 3b at q=0)."""
    import torch
    x_det = _regenerate_chunked(generator, x, region_blocks, canon, block_size=block_size)
    if slip <= 0.0:
        return x_det
    batch = x_det.shape[0]
    positions = region_blocks[:, :, None] * block_size + torch.arange(block_size, device=x_det.device)
    flat_positions = positions.reshape(batch, -1)                                  # (B, R*bs)
    do_slip = torch.rand(region_blocks.shape, device=x_det.device, generator=gen) < slip   # (B, R)
    rand_feat = torch.randint(0, v, region_blocks.shape, device=x_det.device, generator=gen)  # (B, R)
    slip_tuples = canon[rand_feat]                                                  # (B, R, bs)
    det_tuples = x_det.gather(1, flat_positions).reshape(batch, region_blocks.shape[1], block_size)
    chosen = torch.where(do_slip[:, :, None], slip_tuples, det_tuples)              # (B, R, bs)
    x_out = x_det.clone()
    x_out.scatter_(1, flat_positions, chosen.reshape(batch, -1))
    return x_out


def _beam_plan_stoch(controller, generator, value, leaves0, roots, canon, region_index, n_regions,
                     rules, *, s, budget, beam_width, slip, v, seed, device):
    """Token beam under the slippery actuator. Each candidate is materialized as ONE
    realization of the (stochastic) move and re-encoded -- a single noisy sample of the
    move's value. Selection and commit are coupled (the kept, materialized config IS the
    new state), exactly as CoT commits the token it sampled."""
    import torch
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        x = leaves0.to(device); roots = roots.to(device)
        batch, length = x.shape
        beams = x[:, None, :]
        width = 1
        for _ in range(budget):
            flat = beams.reshape(batch * width, length)
            children = [
                _stoch_regenerate(generator, flat,
                                  region_index[torch.full((batch * width,), k, device=device, dtype=torch.long)],
                                  canon, block_size=s, slip=slip, v=v, gen=gen)
                for k in range(n_regions)
            ]
            cand = torch.stack(children, dim=1).reshape(batch, width * n_regions, length)
            tgt = roots.repeat_interleave(width * n_regions)
            scores = value(_encode_chunked(controller, cand.reshape(-1, length)), tgt).reshape(batch, width * n_regions)
            keep = min(beam_width, width * n_regions)
            _, top_idx = scores.topk(keep, dim=1)
            beams = cand.gather(1, top_idx[:, :, None].expand(-1, -1, length))
            width = keep
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams.reshape(-1, length)), tgt).reshape(batch, width)
        best = final.argmax(dim=1)
        x_final = beams[torch.arange(batch, device=device), best]
        success = _success_ps(x_final.cpu().numpy(), roots.cpu().numpy(), rules, s)
    return float(success.mean())


def _latent_beam_stoch(controller, generator, fm, value, leaves0, roots, canon, region_index,
                       n_regions, rules, *, s, budget, beam_width, slip, v, seed, device):
    """Latent beam under the slippery actuator (Design 1: reuse the deterministic FM).
    RANKS candidates by the FM's one-step prediction from the beam's true latent -- a
    stable, low-variance estimate of the move's INTENDED consequence, uncorrupted by the
    slip (which is exogenous and uncontrollable). Only the kept tips are materialized as a
    fresh slippery realization and re-encoded (commit + re-ground). At slip=0 this is the
    Stage-3b _latent_beam exactly."""
    import torch
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        controller.eval(); generator.eval(); fm.eval(); value.eval()
        x0 = leaves0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        beams_x = x0[:, None, :]
        beams_z = controller.block_state(x0)[:, None]
        width = 1
        for _ in range(budget):
            n_blocks, dim = beams_z.shape[2], beams_z.shape[3]
            flat_z = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for k in range(n_regions):
                kk = torch.full((batch * width,), k, device=device, dtype=torch.long)
                pred = (flat_z + _fm_chunked(fm, flat_z, kk)).mean(dim=1)
                scores.append(value(pred, roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * n_regions)
            keep = min(beam_width, width * n_regions)
            _, top_idx = scores.topk(keep, dim=1)
            parent = torch.div(top_idx, n_regions, rounding_mode="floor")
            move = top_idx % n_regions
            parent_x = beams_x.gather(1, parent[:, :, None].expand(-1, -1, length))
            flat_x = parent_x.reshape(batch * keep, length)
            new_x = _stoch_regenerate(generator, flat_x, region_index[move.reshape(-1)], canon,
                                      block_size=s, slip=slip, v=v, gen=gen)
            beams_x = new_x.reshape(batch, keep, length)
            beams_z = _block_state_chunked(controller, new_x).reshape(batch, keep, n_blocks, dim)
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, beams_z.shape[3]),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
        x_final = beams_x[torch.arange(batch, device=device), best]
        success = parse_success_and_heuristic(rules, x_final.cpu().numpy(), roots.cpu().numpy(), s)[0]
    return float(success.mean())


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def sculpt_latent_stoch(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256, n_corrupt: int = 3,
    edit_budget: int = 6, region_size: int = 1, beam_widths: str = "1,16,64",
    slips: str = "0.0,0.1,0.25,0.5,0.75", explore_eps: float = 0.3, quick: bool = False,
):
    """Latent vs token beam across a slippery-actuator sweep q (fixed deterministic
    instruments; only the dynamics noise varies). q=0 must reproduce Stage 3b."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    qs = [float(x) for x in slips.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-latent-STOCH: v={v}, s={s}, L={depth}, m={m}, blocks={n_blocks}, n_corrupt={n_corrupt}, "
          f"budget={edit_budget}, beam_widths={widths}, slips={qs}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)

    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    RichController = _build_rich_controller()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()
    BlockLatentFM = _build_block_fm()

    # --- fixed instruments, trained ONCE under the DETERMINISTIC world ------------
    controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps, lr=3e-4,
                           device=device, p_full=0.5)
    generator = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                              root_conditioned=False).to(device)
    _train_generator(generator, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=generator_steps, lr=3e-4, device=device)
    for module in (controller, generator):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    print(f"Collecting value data ({value_episodes} rollouts)")
    configs, roots_buf, success_buf = _collect_value_sculpt(
        controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
        rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
    print(f"  buffer: {configs.shape[0]} states, terminal success rate {success_buf.mean().item():.3f}")
    value = MCValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                    n_steps=value_steps, lr=3e-4, device=device)

    block_fm = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    print(f"Block FM parameters: {_count_parameters(block_fm):,}")
    _train_block_fm(block_fm, controller, generator, train_roots_np, train_leaves_np, canon,
                    region_index, n_regions, n_steps=fm_steps, batch_size=batch_size, n_blocks=n_blocks,
                    v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3, device=device)
    for module in (value, block_fm):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # --- eval instances (deterministic DP ceiling = the q=0 upper bound) ----------
    planner_batch = min(1024, n_eval_episodes)
    eval_roots_np, eval_leaves_np = _sample_pool(rules, planner_batch, s, train_seed + 99)
    rng = np.random.default_rng(train_seed + 2)
    start_np = _corrupt(eval_leaves_np, n_blocks, n_corrupt, v, s, rng)
    leaves0 = torch.from_numpy(start_np)
    targets = torch.from_numpy(eval_roots_np)
    token_budget = n_corrupt * s
    dstar = nearest_derivation_cost(rules, start_np, eval_roots_np, s)
    frac_solvable = float((dstar <= token_budget).mean())

    fm_check = _fm_check(block_fm, value, controller, generator, leaves0, targets, canon,
                         region_index, n_regions, s, device)
    print(f"\n=== Task DP frac_solvable(q=0)={frac_solvable:.3f} | block FM: delta_cos={fm_check['delta_cos']:.3f}, "
          f"value_top1_agree={fm_check['value_top1_agree']:.3f}, value_rank_corr={fm_check['value_rank_corr']:.3f} ===")

    # --- q=0 sanity anchor: stochastic beams must match Stage-3b beams exactly ----
    anchor_w = widths[-1]
    ref_token = _beam_plan(controller, generator, value, leaves0, targets, canon, region_index,
                           n_regions, rules, s=s, budget=edit_budget, beam_width=anchor_w, device=device)
    ref_latent = _latent_beam(controller, generator, block_fm, value, leaves0, targets, canon,
                              region_index, n_regions, rules, s=s, budget=edit_budget,
                              beam_width=anchor_w, device=device)
    st_token0 = _beam_plan_stoch(controller, generator, value, leaves0, targets, canon, region_index,
                                 n_regions, rules, s=s, budget=edit_budget, beam_width=anchor_w,
                                 slip=0.0, v=v, seed=train_seed + 5, device=device)
    st_latent0 = _latent_beam_stoch(controller, generator, block_fm, value, leaves0, targets, canon,
                                    region_index, n_regions, rules, s=s, budget=edit_budget,
                                    beam_width=anchor_w, slip=0.0, v=v, seed=train_seed + 5, device=device)
    print(f"\n=== q=0 anchor (w{anchor_w}): token ref={ref_token:.3f} st={st_token0:.3f} | "
          f"latent ref={ref_latent:.3f} st={st_latent0:.3f} ===")
    assert abs(ref_token - st_token0) < 1e-6, f"STOCH token beam diverges from Stage-3b at q=0: {ref_token} vs {st_token0}"
    assert abs(ref_latent - st_latent0) < 1e-6, f"STOCH latent beam diverges from Stage-3b at q=0: {ref_latent} vs {st_latent0}"

    # --- the sweep: token vs latent across slip probability q ---------------------
    results = {}
    for q in qs:
        results[q] = {}
        for width in widths:
            arm = train_seed + 5 + int(round(q * 1000)) + width
            tok = _beam_plan_stoch(controller, generator, value, leaves0, targets, canon, region_index,
                                   n_regions, rules, s=s, budget=edit_budget, beam_width=width,
                                   slip=q, v=v, seed=arm, device=device)
            lat = _latent_beam_stoch(controller, generator, block_fm, value, leaves0, targets, canon,
                                     region_index, n_regions, rules, s=s, budget=edit_budget,
                                     beam_width=width, slip=q, v=v, seed=arm, device=device)
            results[q][width] = {"token": tok, "latent": lat, "gap": lat - tok}
            print(f"  q={q:.2f} width={width:3d}:  token={tok:.3f}  latent={lat:.3f}  gap(lat-tok)={lat - tok:+.3f}")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "beam_widths": widths, "slips": qs,
                   "state_dim": state_dim, "value_episodes": value_episodes},
        "dp_frac_solvable_q0": frac_solvable, "value_buffer_success_rate": success_buf.mean().item(),
        "block_fm_check": fm_check,
        "q0_anchor": {"token_ref": ref_token, "token_st": st_token0, "latent_ref": ref_latent,
                      "latent_st": st_latent0},
        "sweep": {str(q): {str(w): results[q][w] for w in widths} for q in qs},
        "elapsed_seconds": time.time() - started,
    }
    print(f"\n=== SUMMARY (stochastic-dynamics sweep, m={m}, c={n_corrupt}) ===")
    print("  gap (latent - token), + => latent wins:")
    for width in widths:
        row = " | ".join(f"q{q:.2f}={results[q][width]['gap']:+.3f}" for q in qs)
        print(f"    width {width:3d}:  {row}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_latent_stoch/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_latent_stoch.remote(m=m, quick=quick)
