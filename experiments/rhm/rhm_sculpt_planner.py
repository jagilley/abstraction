"""RHM Sculpting — the LEARNED control experiment (Stage 3a: token-space beam).

The perfect-simulator pre-check (sculpting_control_task.md / rhm_sculpt_precheck.py)
established that editing RHM leaves toward a target root r* has a genuine LOOKAHEAD
PRIZE: the full-horizon DP optimum solves 100%, while a myopic reflex fails on a
large, depth-scaling fraction (L=4,m=2,c=3: weak reflex 0.24, strong r*-aware reflex
0.59; optimum 1.0). Unlike the sensing/query task, choosing an edit is NOT the same
as forecasting its value -- reaching r* needs a COORDINATED multi-edit plan (a
globally-consistent deep derivation) a reflex cannot assemble greedily.

This is the task my earlier corrupt-repair experiments *should* have used. Those
mis-measured the difficulty with a witness-targeted greedy oracle (which hits 1.0
by decomposing into independent per-block fixes) and concluded, wrongly, that the
task was greedy-solvable. Here the ceiling is the DP optimum and success is
ambiguity-aware (r* in the root's possible-set, matching the precheck's DISTINCT
rules), so the coordination is real.

Stage 3a asks the first learned question: does a proper PLANNER capture the prize
that a myopic reflex leaves on the table? We use generator (on-manifold) moves, a
learned Monte-Carlo VALUE (cheap + faithful), and a BEAM planner. Beam width 1 is
the myopic reflex; wider beams keep multiple hypotheses = coordination. If wider
beam -> higher true success, lookahead is learnable and captured. (Stage 3b then
adds a richer non-pooled latent + a cerebellar FM to plan the beam IN LATENTS.)

Run:
  modal run rhm/rhm_sculpt_planner.py::sculpt_planner --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_planner.py::sculpt_planner --m 2  # (L=4,c=3)
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import (
    nearest_derivation_cost,
    parse_success_and_heuristic,
    sample_derivations,
)
from rhm.rhm_active_query import _build_model_classes, _count_parameters
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _regenerate, _train_generator
from rhm.rhm_latent_planner import (
    _behavior_step,
    _build_value_head,
    _region_index,
    _train_value_mc,
)


app = modal.App("rhm-sculpt-planner", image=image)


def _sample_pool(rules, n, s, seed):
    """n valid derivations -> (roots, leaves) numpy."""
    rng = np.random.default_rng(seed)
    v = rules[0].shape[0]
    roots = rng.integers(0, v, size=n)
    leaves = sample_derivations(rules, roots, s, rng)
    return roots.astype(np.int64), leaves.astype(np.int64)


def _corrupt(leaves_np, n_blocks, n_corrupt, v, s, rng):
    """Corrupt n_corrupt random blocks per row to random symbols (precheck-style)."""
    start = leaves_np.copy()
    for b in range(start.shape[0]):
        for blk in rng.choice(n_blocks, size=n_corrupt, replace=False):
            start[b, blk * s:(blk + 1) * s] = rng.integers(0, v, size=s)
    return start


def _success_ps(leaves_np, roots_np, rules, s):
    """Ambiguity-aware success: r* is in the root's possible-set."""
    success, _ = parse_success_and_heuristic(rules, leaves_np, roots_np, s)
    return success


def _encode_chunked(controller, x, chunk=4096):
    import torch
    if x.shape[0] <= chunk:
        return controller.state(x)
    return torch.cat([controller.state(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], dim=0)


def _regenerate_chunked(generator, x, region_blocks, canon, *, block_size, chunk=16384):
    """_regenerate but chunked over the batch (the fused transformer kernel fails past
    ~64k rows, which wide beams exceed)."""
    import torch
    if x.shape[0] <= chunk:
        return _regenerate(generator, x, region_blocks, canon, None, block_size=block_size, sample=False)
    return torch.cat([
        _regenerate(generator, x[i:i + chunk], region_blocks[i:i + chunk], canon, None,
                    block_size=block_size, sample=False)
        for i in range(0, x.shape[0], chunk)], dim=0)


def _collect_value_sculpt(controller, generator, pool_roots, pool_leaves, canon, region_index,
                          n_regions, rules, *, n_episodes, batch_size, n_blocks, v, s, n_corrupt,
                          budget, epsilon, device):
    """Roll the behaviour policy (controller-greedy generator moves + eps) from corrupt
    starts; label EVERY visited state by its rollout's terminal possible-set success."""
    import torch
    rng = np.random.default_rng(4321)
    configs, roots_all, success_all = [], [], []
    collected = 0
    while collected < n_episodes:
        batch = min(batch_size, n_episodes - collected)
        collected += batch
        idx = rng.integers(0, pool_leaves.shape[0], size=batch)
        roots_np = pool_roots[idx]
        # mix corruption levels so easier instances supply positive labels and the
        # value learns a dense distance-to-goal signal across the whole range.
        c = int(rng.integers(1, n_corrupt + 1))
        start_np = _corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)
        x = torch.from_numpy(start_np).to(device)
        roots = torch.from_numpy(roots_np).to(device)
        trajectory = [x.clone()]
        for _ in range(budget):
            x = _behavior_step(controller, generator, x, roots, canon, region_index,
                               n_regions, s, epsilon, device)
            trajectory.append(x.clone())
        success = torch.from_numpy(_success_ps(x.cpu().numpy(), roots_np, rules, s).astype(np.float32))
        for state in trajectory:
            configs.append(state.cpu())
            roots_all.append(roots.cpu())
            success_all.append(success)
    return torch.cat(configs), torch.cat(roots_all), torch.cat(success_all)


def _beam_plan(controller, generator, value, leaves0, roots, canon, region_index, n_regions,
               rules, *, s, budget, beam_width, device):
    """Beam search over budget-length generator-move sequences, scored by the MC value.
    beam_width=1 reproduces the greedy myopic reflex. Graded by possible-set success."""
    import torch
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        x = leaves0.to(device)
        roots = roots.to(device)
        batch, length = x.shape
        beams = x[:, None, :]
        width = 1
        for _ in range(budget):
            flat = beams.reshape(batch * width, length)
            children = [
                _regenerate_chunked(generator, flat,
                                    region_index[torch.full((batch * width,), k, device=device, dtype=torch.long)],
                                    canon, block_size=s)
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


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def sculpt_planner(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, batch_size: int = 256, n_corrupt: int = 3, edit_budget: int = 6,
    region_size: int = 1, beam_widths: str = "1,4,16", explore_eps: float = 0.3, quick: bool = False,
):
    """Learned sculpting: token beam (width 1 = reflex, wider = coordination) vs DP optimum."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-planner (LEARNED): v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
          f"blocks={n_blocks}, n_corrupt={n_corrupt}, move_budget={edit_budget}, "
          f"beam_widths={widths}, device={device}")
    started = time.time()

    # DISTINCT rules (ambiguous parse) to match the precheck.
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)

    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    BeliefController, _StateFM, _ActionFM = _build_model_classes()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()

    controller = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
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

    # Learned Monte-Carlo value (possible-set success labels) ---------------------
    print(f"Collecting value data ({value_episodes} rollouts)")
    configs, roots_buf, success_buf = _collect_value_sculpt(
        controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
        rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
    print(f"  buffer: {configs.shape[0]} states, terminal success rate {success_buf.mean().item():.3f}")
    value = MCValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                    n_steps=value_steps, lr=3e-4, device=device)
    value.eval()
    for parameter in value.parameters():
        parameter.requires_grad_(False)

    # Evaluation instances + DP ceiling ------------------------------------------
    planner_batch = min(1024, n_eval_episodes)
    eval_roots_np, eval_leaves_np = _sample_pool(rules, planner_batch, s, train_seed + 99)
    rng = np.random.default_rng(train_seed + 2)
    start_np = _corrupt(eval_leaves_np, n_blocks, n_corrupt, v, s, rng)
    leaves0 = torch.from_numpy(start_np)
    targets = torch.from_numpy(eval_roots_np)

    token_budget = n_corrupt * s
    dstar = nearest_derivation_cost(rules, start_np, eval_roots_np, s)
    frac_solvable = float((dstar <= token_budget).mean())
    start_success = float(_success_ps(start_np, eval_roots_np, rules, s).mean())
    print(f"\n=== Task check ===  DP d*_mean={dstar.mean():.2f}, frac solvable (<= {token_budget} token edits)={frac_solvable:.3f}; "
          f"start success={start_success:.3f}")

    # Beam planners (width 1 = reflex) -------------------------------------------
    beam_results = {}
    for width in widths:
        succ = _beam_plan(controller, generator, value, leaves0, targets, canon, region_index,
                          n_regions, rules, s=s, budget=edit_budget, beam_width=width, device=device)
        beam_results[width] = succ
        print(f"  beam width {width:3d}: success={succ:.3f}")

    reflex_ref = {"weak_parse_reflex": 0.242, "strong_r*_reflex": 0.585, "dp_optimum": 1.0}  # precheck L4 m2 c3

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "token_budget": token_budget, "region_size": region_size,
                   "beam_widths": widths, "state_dim": state_dim, "value_episodes": value_episodes},
        "dp_frac_solvable": frac_solvable, "dstar_mean": float(dstar.mean()),
        "start_success": start_success, "value_buffer_success_rate": success_buf.mean().item(),
        "beam_success": {str(k): val for k, val in beam_results.items()},
        "precheck_reference_L4m2c3": reflex_ref,
        "elapsed_seconds": time.time() - started,
    }
    print(f"\n=== SUMMARY (learned, m={m}, c={n_corrupt}) ===")
    print(f"  reference: DP optimum=1.00 | strong reflex=0.585 | weak reflex=0.242")
    print(f"  learned:   " + " | ".join(f"beam{k}={v_:.3f}" for k, v_ in beam_results.items()))

    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_planner/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_planner.remote(m=m, quick=quick)
