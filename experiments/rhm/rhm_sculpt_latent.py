"""RHM Sculpting — Stage 3b: planning in LATENTS vs token-space, head-to-head.

Stage 3a (rhm_sculpt_planner.py) showed a learned token-space BEAM captures the
sculpting task's coordination prize (0.29 greedy -> 0.58 beam-64, matching the
strong hand-coded reflex) where greedy collapses -- because the task now genuinely
rewards lookahead. That beam is "CoT-like": it materializes every branch as tokens
and re-encodes it. This file asks the original question: does planning IN LATENTS
-- a cerebellar forward model rolling the belief forward without decoding -- match
that (at far lower cost) or beat it?

Two fixes from the Stage-2 failure (where a pooled-belief FM was too lossy to roll,
latent-greedy 0.06 << token-greedy 0.27):
  1. RICHER LATENT: a per-block (non-pooled) latent z (B, n_blocks, D), so the FM
     retains the per-block structure a pooled mean threw away.
  2. FM = a small transformer over the block latents (captures cross-block cascade).

Clean isolation: the VALUE is identical to the token beam -- it reads the POOLED
belief b = mean_block(z) (and mean-of-block-means == the controller's own pooled
state exactly). Token beam: b from re-encoding the materialized config. Latent
beam: b from pooling the FM-predicted z. Same value, same moves, same beam width --
the ONLY difference is FM-rollout-in-latents vs materialize-and-re-encode. So the
head-to-head measures exactly "planning in latents."

Run:
  modal run rhm/rhm_sculpt_latent.py::sculpt_latent --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_latent.py::sculpt_latent --m 2  # L=4,c=3
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
)
from rhm.rhm_active_query import _count_parameters
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _regenerate, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_sculpt_planner import (
    _beam_plan,
    _collect_value_sculpt,
    _corrupt,
    _regenerate_chunked,
    _sample_pool,
    _success_ps,
)


app = modal.App("rhm-sculpt-latent", image=image)


def _build_rich_controller():
    """BeliefController that also exposes a per-block latent. state() = pooled belief
    (unchanged, drives root head + value); block_state() = (B, n_blocks, D). Note
    mean over blocks of block_state == state exactly (uniform blocks)."""
    import torch
    import torch.nn as nn

    class RichBeliefController(nn.Module):
        def __init__(self, vocab_size, sequence_length, block_size, state_dim, n_head, n_layer):
            super().__init__()
            self.mask_token = vocab_size
            self.block_size = block_size
            self.n_blocks = sequence_length // block_size
            self.token_embedding = nn.Embedding(vocab_size + 1, state_dim)
            self.position_embedding = nn.Embedding(sequence_length, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.final_norm = nn.LayerNorm(state_dim)
            self.root_head = nn.Linear(state_dim, vocab_size)
            self.register_buffer("positions", torch.arange(sequence_length), persistent=False)

        def _encode(self, observation):
            tokens = observation.masked_fill(observation < 0, self.mask_token)
            hidden = self.token_embedding(tokens) + self.position_embedding(self.positions)
            return self.final_norm(self.encoder(hidden))  # (B, T, D)

        def state(self, observation):
            return self._encode(observation).mean(dim=1)  # (B, D)

        def block_state(self, observation):
            hidden = self._encode(observation)
            batch, length, dim = hidden.shape
            return hidden.view(batch, self.n_blocks, self.block_size, dim).mean(dim=2)  # (B, n_blocks, D)

        def root_logits(self, state):
            return self.root_head(state)

        def forward(self, observation):
            state = self.state(observation)
            return self.root_logits(state), state

    return RichBeliefController


def _build_block_fm():
    """Cerebellar FM over per-block latents: (z (B,n_blocks,D), acted block k) -> Δz.
    A small transformer so a move's cross-block cascade is representable."""
    import torch
    import torch.nn as nn

    class BlockLatentFM(nn.Module):
        def __init__(self, state_dim, n_blocks, n_head, n_layer):
            super().__init__()
            self.action_embedding = nn.Embedding(n_blocks, state_dim)
            self.block_position = nn.Embedding(n_blocks, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.norm = nn.LayerNorm(state_dim)
            self.head = nn.Linear(state_dim, state_dim)
            self.register_buffer("positions", torch.arange(n_blocks), persistent=False)

        def forward(self, z_block, k):
            batch, n_blocks, dim = z_block.shape
            marker = torch.zeros_like(z_block)
            marker.scatter_(1, k[:, None, None].expand(-1, 1, dim), self.action_embedding(k)[:, None, :])
            hidden = z_block + self.block_position(self.positions)[None] + marker
            return self.head(self.norm(self.encoder(hidden)))

    return BlockLatentFM


def _fm_chunked(fm, z, k, chunk=16384):
    import torch
    if z.shape[0] <= chunk:
        return fm(z, k)
    return torch.cat([fm(z[i:i + chunk], k[i:i + chunk]) for i in range(0, z.shape[0], chunk)], dim=0)


def _block_state_chunked(controller, x, chunk=4096):
    import torch
    if x.shape[0] <= chunk:
        return controller.block_state(x)
    return torch.cat([controller.block_state(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], dim=0)


def _train_block_fm(fm, controller, generator, pool_roots, pool_leaves, canon, region_index,
                    n_regions, *, n_steps, batch_size, n_blocks, v, s, n_corrupt, budget, lr, device):
    """Predict the per-block latent delta of regenerating a region, on the visited
    distribution (corrupt starts + random generator regenerations)."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(777)
    fm.train()
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, pool_leaves.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))
        for _ in range(g):
            k = torch.randint(0, n_regions, (batch_size,), device=device)
            x = _regenerate(generator, x, region_index[k], canon, None, block_size=s, sample=False)
        k = torch.randint(0, n_regions, (batch_size,), device=device)
        with torch.no_grad():
            z = controller.block_state(x)
            x2 = _regenerate(generator, x, region_index[k], canon, None, block_size=s, sample=False)
            target = controller.block_state(x2) - z
        pred = fm(z, k)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            cos = F.cosine_similarity(pred.reshape(batch_size, -1), target.reshape(batch_size, -1), dim=-1).mean().item()
            print(f"  block FM step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cos:.3f}")


def _fm_check(fm, value, controller, generator, leaves0, roots, canon, region_index, n_regions, s, device):
    """One-step: block-latent delta cosine, and whether the FM-predicted pooled belief
    gives the same value ranking over moves as the true (materialized) one."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        x = leaves0.to(device)
        roots = roots.to(device)
        z = controller.block_state(x)
        batch = x.shape[0]
        true_v = torch.empty(batch, n_regions, device=device)
        pred_v = torch.empty(batch, n_regions, device=device)
        cos_accum = 0.0
        for k in range(n_regions):
            kk = torch.full((batch,), k, device=device, dtype=torch.long)
            x2 = _regenerate(generator, x, region_index[kk], canon, None, block_size=s, sample=False)
            true_z = controller.block_state(x2)
            pred_z = z + fm(z, kk)
            cos_accum += F.cosine_similarity((pred_z - z).reshape(batch, -1), (true_z - z).reshape(batch, -1), dim=-1).mean().item()
            true_v[:, k] = value(true_z.mean(dim=1), roots)
            pred_v[:, k] = value(pred_z.mean(dim=1), roots)
        top1 = (true_v.argmax(1) == pred_v.argmax(1)).float().mean().item()
        tc = true_v - true_v.mean(1, keepdim=True)
        pc = pred_v - pred_v.mean(1, keepdim=True)
        rank = ((tc * pc).sum(1) / (tc.norm(dim=1) * pc.norm(dim=1)).clamp_min(1e-8)).mean().item()
    return {"delta_cos": cos_accum / n_regions, "value_top1_agree": top1, "value_rank_corr": rank}


def _latent_beam(controller, generator, fm, value, leaves0, roots, canon, region_index,
                 n_regions, rules, *, s, budget, beam_width, device):
    """Beam search that RE-GROUNDS each step (Dreamer-style). Candidates are RANKED by
    a ONE-step FM prediction from the beam's TRUE latent (so the FM never rolls open-
    loop -> no compounding drift); only the kept beam tips are materialized + re-
    encoded to true latents for the next step. The FM buys cheap ranking (materialize
    W tips/step, not W*n_regions candidates like the token beam). beam_width 1 = greedy.
    """
    import torch
    with torch.no_grad():
        controller.eval(); generator.eval(); fm.eval(); value.eval()
        x0 = leaves0.to(device)
        roots = roots.to(device)
        batch, length = x0.shape
        beams_x = x0[:, None, :]                              # (B, W, T) true configs
        beams_z = controller.block_state(x0)[:, None]        # (B, W, n_blocks, D) true latents
        width = 1
        for _ in range(budget):
            n_blocks, dim = beams_z.shape[2], beams_z.shape[3]
            flat_z = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for k in range(n_regions):
                kk = torch.full((batch * width,), k, device=device, dtype=torch.long)
                pred = (flat_z + _fm_chunked(fm, flat_z, kk)).mean(dim=1)   # 1-step FM, pooled
                scores.append(value(pred, roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * n_regions)
            keep = min(beam_width, width * n_regions)
            _, top_idx = scores.topk(keep, dim=1)
            parent = torch.div(top_idx, n_regions, rounding_mode="floor")   # (B, keep)
            move = top_idx % n_regions
            parent_x = beams_x.gather(1, parent[:, :, None].expand(-1, -1, length))  # (B, keep, T)
            flat_x = parent_x.reshape(batch * keep, length)
            new_x = _regenerate_chunked(generator, flat_x, region_index[move.reshape(-1)], canon, block_size=s)
            beams_x = new_x.reshape(batch, keep, length)
            beams_z = _block_state_chunked(controller, new_x).reshape(batch, keep, beams_z.shape[2], beams_z.shape[3])
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, beams_z.shape[3]),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
        x_final = beams_x[torch.arange(batch, device=device), best]
        success = parse_success_and_heuristic(rules, x_final.cpu().numpy(), roots.cpu().numpy(), s)[0]
    return float(success.mean())


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def sculpt_latent(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256, n_corrupt: int = 3,
    edit_budget: int = 6, region_size: int = 1, beam_widths: str = "1,16,64", explore_eps: float = 0.3,
    quick: bool = False,
):
    """Latent-space beam vs token-space beam on the sculpting task (matched value/moves)."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-latent (planning IN LATENTS): v={v}, s={s}, L={depth}, m={m}, blocks={n_blocks}, "
          f"n_corrupt={n_corrupt}, budget={edit_budget}, beam_widths={widths}, device={device}")
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

    # value (pooled belief, possible-set labels) + block-latent FM ----------------
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

    # eval instances + DP ceiling -------------------------------------------------
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

    # head-to-head: token beam vs latent beam, matched widths --------------------
    token_beam, latent_beam = {}, {}
    for width in widths:
        token_beam[width] = _beam_plan(controller, generator, value, leaves0, targets, canon,
                                       region_index, n_regions, rules, s=s, budget=edit_budget,
                                       beam_width=width, device=device)
        latent_beam[width] = _latent_beam(controller, generator, block_fm, value, leaves0, targets,
                                          canon, region_index, n_regions, rules, s=s, budget=edit_budget,
                                          beam_width=width, device=device)
        print(f"  width {width:3d}:  token={token_beam[width]:.3f}   latent={latent_beam[width]:.3f}")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "beam_widths": widths, "state_dim": state_dim,
                   "value_episodes": value_episodes},
        "dp_frac_solvable": frac_solvable, "value_buffer_success_rate": success_buf.mean().item(),
        "block_fm_check": fm_check,
        "token_beam_success": {str(k): val for k, val in token_beam.items()},
        "latent_beam_success": {str(k): val for k, val in latent_beam.items()},
        "precheck_reference_L4m2c3": {"weak_reflex": 0.242, "strong_reflex": 0.585, "dp_optimum": 1.0},
        "elapsed_seconds": time.time() - started,
    }
    print(f"\n=== SUMMARY (planning in latents, m={m}, c={n_corrupt}) ===")
    print(f"  reference: DP=1.00 | strong reflex=0.585 | weak reflex=0.242")
    print("  token beam:  " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in token_beam.items()))
    print("  latent beam: " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in latent_beam.items()))

    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_latent/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_latent.remote(m=m, quick=quick)
