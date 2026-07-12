"""RHM Sculpting — Stage 3c: latent vs token planning under PARTIAL OBSERVABILITY.

Stage 3b (rhm_sculpt_latent.py) found that on the FULLY-OBSERVED sculpting task,
planning in latents is an efficient SURROGATE, not a superior one: the latent beam
reaches ~92% of the token beam at ~8x lower cost, but never beats it. The argued
reason (see RHM_SCULPTING_README "What this does NOT establish"): the task is fully
observed and deterministic-given-observed, so the TOKEN sequence is a SUFFICIENT
STATISTIC -- re-encoding materialized tokens is lossless, and a latent FM can at
best approximate that lossless map. There is no "structured divergence tokens can't
carry" when tokens carry everything.

Hypothesis under test: latent > token requires the observation to be an INSUFFICIENT
statistic. Introduce PARTIAL OBSERVABILITY -- a sensor that occludes a fraction p of
blocks -- and the token beam, whose only way to score a candidate is to RE-OBSERVE
it through the lossy sensor, should degrade faster than the latent beam, which CARRIES
a per-block belief and rolls it forward with the FM (using the FM's prediction for
blocks it currently can't see). Prediction: latent - token gap rises with p, crossing
from negative (token wins, the Stage-3b result) to positive (latent wins), then both
collapse as p -> 1 (nobody ever sees anything). A hump.

Controlled variable = ONLY the sensor. Everything else is a FIXED instrument trained
ONCE under full observation (controller, generator, MC value, block FM -- reused from
Stage 3b): "an agent that learned the world when it could see it, now deployed partly
blind." The task difficulty is IDENTICAL across p (the grader always sees the true
final config; corruption / budget / grammar / value target unchanged). At p=0 both
beams reduce EXACTLY to Stage 3b (built-in sanity anchor, asserted numerically).

Scoping (two deliberate choices, see RHM_SCULPTING_README follow-up):
  1. FLICKERING sensor: fresh per-step block occlusion (same mask for all candidates
     within a step), not permanent occlusion. Permanent occlusion hides blocks from
     EVERYONE equally (no belief source, no gap); flickering is the clean POMDP where
     integrating a belief over partial views strictly beats any single view. The
     latent beam filters Kalman-style: observed block -> re-encode, masked block ->
     keep the FM prediction (carry the belief).
  2. Masking degrades the EVALUATION channel (the state estimate used to SCORE
     candidates), not candidate CONSTRUCTION (both arms materialize true tips). This
     isolates exactly the belief-sufficiency claim and avoids a separate confound
     about conditioning the generator on masked context.

Run:
  modal run rhm/rhm_sculpt_latent_po.py::sculpt_latent_po --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_latent_po.py::sculpt_latent_po --m 2  # L=4,c=3
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


app = modal.App("rhm-sculpt-latent-po", image=image)


def _apply_block_mask(x, block_mask, block_size):
    """x: (..., T) int tokens; block_mask: (..., n_blocks) bool (True = occluded).
    Return a COPY with occluded blocks' tokens set to -1 (the controller/generator
    encode negatives as the mask token). The true config is never mutated -- masking
    is purely a sensor applied to what the planner OBSERVES for scoring."""
    leaf_mask = block_mask.repeat_interleave(block_size, dim=-1)  # (..., T)
    return x.masked_fill(leaf_mask, -1)


def _make_masks(batch, budget, n_blocks, p, seed):
    """Fixed, shared per-instance per-step occlusion patterns (B, budget+1, n_blocks),
    True = occluded. Identical masks are handed to BOTH beams so they face the exact
    same sensor -- the only controlled variable."""
    import torch
    rng = np.random.default_rng(seed)
    m = rng.random((batch, budget + 1, n_blocks)) < p
    return torch.from_numpy(m)


def _beam_plan_po(controller, generator, value, leaves0, roots, canon, region_index, n_regions,
                  rules, masks, *, s, budget, beam_width, device):
    """Token beam (materialize-and-re-encode) under the occluding sensor. Candidates
    are CONSTRUCTED from the true config (shared with the latent beam), but SCORED by
    re-encoding a MASKED observation of them -- the token planner's only channel to a
    state estimate is a fresh (lossy) look. The true config is retained for the world
    update + grading; masking only ever touches the copy fed to the encoder.
    masks[:, t] occludes the observation of the candidates produced at step t."""
    import torch
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        x = leaves0.to(device); roots = roots.to(device)
        masks = masks.to(device)
        batch, length = x.shape
        beams = x[:, None, :]
        width = 1
        for t in range(budget):
            flat = beams.reshape(batch * width, length)
            children = [
                _regenerate_chunked(generator, flat,
                                    region_index[torch.full((batch * width,), k, device=device, dtype=torch.long)],
                                    canon, block_size=s)
                for k in range(n_regions)
            ]
            cand = torch.stack(children, dim=1).reshape(batch, width * n_regions, length)  # TRUE configs
            mstep = masks[:, t][:, None, :].expand(batch, width * n_regions, -1)           # (B, W*R, n_blocks)
            cand_obs = _apply_block_mask(cand, mstep, s)                                   # masked OBSERVATION
            tgt = roots.repeat_interleave(width * n_regions)
            scores = value(_encode_chunked(controller, cand_obs.reshape(-1, length)), tgt).reshape(batch, width * n_regions)
            keep = min(beam_width, width * n_regions)
            _, top_idx = scores.topk(keep, dim=1)
            beams = cand.gather(1, top_idx[:, :, None].expand(-1, -1, length))             # keep TRUE configs
            width = keep
        mstep = masks[:, budget][:, None, :].expand(batch, width, -1)
        beams_obs = _apply_block_mask(beams, mstep, s)
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams_obs.reshape(-1, length)), tgt).reshape(batch, width)
        best = final.argmax(dim=1)
        x_final = beams[torch.arange(batch, device=device), best]                          # grade the TRUE config
        success = _success_ps(x_final.cpu().numpy(), roots.cpu().numpy(), rules, s)
    return float(success.mean())


def _latent_beam_po(controller, generator, fm, value, leaves0, roots, canon, region_index,
                    n_regions, rules, masks, *, s, budget, beam_width, device):
    """Latent beam under the occluding sensor, with Kalman-style belief filtering.
    Candidates are RANKED by a one-step FM prediction from the beam's CARRIED belief
    (never re-observing to rank). Kept tips are materialized (true) and observed
    through the masked sensor; the belief is then FILTERED per block: an OBSERVED block
    takes the fresh re-encode, an OCCLUDED block keeps the FM's prediction (belief
    carried forward from when it was last seen). At p=0 this reduces EXACTLY to the
    Stage-3b _latent_beam (all blocks observed -> filter is a full reset)."""
    import torch
    with torch.no_grad():
        controller.eval(); generator.eval(); fm.eval(); value.eval()
        x0 = leaves0.to(device); roots = roots.to(device)
        masks = masks.to(device)
        batch, length = x0.shape
        x0_obs = _apply_block_mask(x0, masks[:, 0], s)
        beams_x = x0[:, None, :]                                   # (B,1,T) TRUE configs
        beams_z = controller.block_state(x0_obs)[:, None]         # (B,1,n_blocks,D) belief
        width = 1
        for t in range(budget):
            n_blocks, dim = beams_z.shape[2], beams_z.shape[3]
            flat_z = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for k in range(n_regions):
                kk = torch.full((batch * width,), k, device=device, dtype=torch.long)
                pred = (flat_z + _fm_chunked(fm, flat_z, kk)).mean(dim=1)   # 1-step FM from carried belief
                scores.append(value(pred, roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * n_regions)
            keep = min(beam_width, width * n_regions)
            _, top_idx = scores.topk(keep, dim=1)
            parent = torch.div(top_idx, n_regions, rounding_mode="floor")
            move = top_idx % n_regions
            parent_x = beams_x.gather(1, parent[:, :, None].expand(-1, -1, length))
            parent_z = beams_z.gather(1, parent[:, :, None, None].expand(-1, -1, n_blocks, dim))
            flat_parent_x = parent_x.reshape(batch * keep, length)
            new_x = _regenerate_chunked(generator, flat_parent_x, region_index[move.reshape(-1)], canon, block_size=s)
            flat_parent_z = parent_z.reshape(batch * keep, n_blocks, dim)
            fm_pred = flat_parent_z + _fm_chunked(fm, flat_parent_z, move.reshape(-1))   # predicted belief after move
            mobs = masks[:, min(t + 1, budget)][:, None, :].expand(batch, keep, n_blocks).reshape(batch * keep, n_blocks)
            new_x_obs = _apply_block_mask(new_x, mobs, s)
            obs_z = _block_state_chunked(controller, new_x_obs)                           # (B*keep, n_blocks, D)
            filt_z = torch.where(mobs[:, :, None], fm_pred, obs_z)                        # occluded -> predict, seen -> observe
            beams_z = filt_z.reshape(batch, keep, n_blocks, dim)
            beams_x = new_x.reshape(batch, keep, length)
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, beams_z.shape[3]),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
        x_final = beams_x[torch.arange(batch, device=device), best]
        success = parse_success_and_heuristic(rules, x_final.cpu().numpy(), roots.cpu().numpy(), s)[0]
    return float(success.mean())


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def sculpt_latent_po(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256, n_corrupt: int = 3,
    edit_budget: int = 6, region_size: int = 1, beam_widths: str = "1,16,64",
    p_masks: str = "0.0,0.25,0.5,0.75", explore_eps: float = 0.3, quick: bool = False,
):
    """Latent vs token beam across an occlusion sweep p (fixed instruments, only the
    sensor varies). p=0 must reproduce Stage 3b."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    ps = [float(x) for x in p_masks.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-latent-PO: v={v}, s={s}, L={depth}, m={m}, blocks={n_blocks}, n_corrupt={n_corrupt}, "
          f"budget={edit_budget}, beam_widths={widths}, p_masks={ps}, device={device}")
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

    # --- fixed instruments, trained ONCE under FULL observation -------------------
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

    # --- eval instances + DP ceiling (identical across p) ------------------------
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
    print(f"\n=== Task DP frac_solvable={frac_solvable:.3f} | block FM: delta_cos={fm_check['delta_cos']:.3f}, "
          f"value_top1_agree={fm_check['value_top1_agree']:.3f}, value_rank_corr={fm_check['value_rank_corr']:.3f} ===")

    # --- p=0 sanity anchor: PO beams must match the Stage-3b beams exactly --------
    zero_masks = _make_masks(planner_batch, edit_budget, n_blocks, 0.0, seed=train_seed + 7)
    anchor_w = widths[-1]
    ref_token = _beam_plan(controller, generator, value, leaves0, targets, canon, region_index,
                           n_regions, rules, s=s, budget=edit_budget, beam_width=anchor_w, device=device)
    ref_latent = _latent_beam(controller, generator, block_fm, value, leaves0, targets, canon,
                              region_index, n_regions, rules, s=s, budget=edit_budget,
                              beam_width=anchor_w, device=device)
    po_token0 = _beam_plan_po(controller, generator, value, leaves0, targets, canon, region_index,
                              n_regions, rules, zero_masks, s=s, budget=edit_budget,
                              beam_width=anchor_w, device=device)
    po_latent0 = _latent_beam_po(controller, generator, block_fm, value, leaves0, targets, canon,
                                 region_index, n_regions, rules, zero_masks, s=s, budget=edit_budget,
                                 beam_width=anchor_w, device=device)
    print(f"\n=== p=0 anchor (w{anchor_w}): token ref={ref_token:.3f} po={po_token0:.3f} | "
          f"latent ref={ref_latent:.3f} po={po_latent0:.3f} ===")
    assert abs(ref_token - po_token0) < 1e-6, f"PO token beam diverges from Stage-3b at p=0: {ref_token} vs {po_token0}"
    assert abs(ref_latent - po_latent0) < 1e-6, f"PO latent beam diverges from Stage-3b at p=0: {ref_latent} vs {po_latent0}"

    # --- the sweep: token vs latent across occlusion p ---------------------------
    results = {}  # p -> {width -> {token, latent, gap}}
    for p in ps:
        masks = _make_masks(planner_batch, edit_budget, n_blocks, p, seed=train_seed + 7)
        results[p] = {}
        for width in widths:
            tok = _beam_plan_po(controller, generator, value, leaves0, targets, canon, region_index,
                                n_regions, rules, masks, s=s, budget=edit_budget, beam_width=width, device=device)
            lat = _latent_beam_po(controller, generator, block_fm, value, leaves0, targets, canon,
                                  region_index, n_regions, rules, masks, s=s, budget=edit_budget,
                                  beam_width=width, device=device)
            results[p][width] = {"token": tok, "latent": lat, "gap": lat - tok}
            print(f"  p={p:.2f} width={width:3d}:  token={tok:.3f}  latent={lat:.3f}  gap(lat-tok)={lat - tok:+.3f}")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "beam_widths": widths, "p_masks": ps,
                   "state_dim": state_dim, "value_episodes": value_episodes},
        "dp_frac_solvable": frac_solvable, "value_buffer_success_rate": success_buf.mean().item(),
        "block_fm_check": fm_check,
        "p0_anchor": {"token_ref": ref_token, "token_po": po_token0, "latent_ref": ref_latent,
                      "latent_po": po_latent0},
        "sweep": {str(p): {str(w): results[p][w] for w in widths} for p in ps},
        "elapsed_seconds": time.time() - started,
    }
    print(f"\n=== SUMMARY (partial-observability sweep, m={m}, c={n_corrupt}) ===")
    print("  gap (latent - token), + => latent wins:")
    for width in widths:
        row = " | ".join(f"p{p:.2f}={results[p][width]['gap']:+.3f}" for p in ps)
        print(f"    width {width:3d}:  {row}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_latent_po/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_latent_po.remote(m=m, quick=quick)
