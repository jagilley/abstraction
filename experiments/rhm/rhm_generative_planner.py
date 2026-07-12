"""Active-RHM Stage B: on-manifold planning via a learned generator.

Context (see rhm_edit_control.py / ACTIVE_RHM_README.md). Editing an RHM sequence
toward a target root is plannable in the controller's BELIEF space (a mean-delta
arity-2 forward model closes ~0.7-1.0 of the random->oracle gap, vs ~0.1 for the
same FM on the epistemic reveal task), but a ground-truth check exposed the belief
as GAMEABLE off the valid-sequence manifold: raw token edits pass through
off-grammar "word-salad" configs where the controller (a perfect parser ON valid
sequences) is uncalibrated, so a planner maximizing its P(r*) reaches configs it
mislabels as r* while ~99% are off-grammar (true success ~0). Only a ground-truth
oracle achieves true control. The bottleneck is belief FAITHFULNESS off-manifold.

The fix tested here (Jasper's proposal): don't robustify the belief off-manifold;
constrain the ACTIONS to stay (softly) on-manifold by making moves come from a
learned GENERATOR. The generator is the cortex proposing on-manifold moves; the
controller judges only near-manifold sequences it understands; direction toward
r* comes from the planner's SEARCH (select proposals that raise P(r*)), not from
conditioning the generator on r* (which would let one big regeneration solve it
and dissolve the planning problem). We also keep an r*-conditioned generator as a
second condition to measure how much directedness the generator needs before it
trivializes.

Stage 1 (this file, token-space): does a generator-defined move space kill the
gaming -- true success jumps from ~0 toward the gt_oracle -- with the SAME judge
and task, only the action space changed? This is also the CoT-like control for the
later latent-space planner (re-encodes tokens each step; throws the latent away).

Move primitive: pick a REGION (an aligned subtree of blocks), mask it, let G infer
its level-1 features from the rest, render via canonical tuples. Leaf-valid by
construction; upper-grammar validity is only BIASED (G learned the joint) -- so the
proposal is *soft* on-manifold, and how off-manifold it lands is the structured
divergence / cerebellar-residual signal (measured as gt_valid).

Run from experiments/ (one detached run per m):
  modal run --detach rhm/rhm_generative_planner.py::generative_planner --m 2
Quick smoke test:
  modal run rhm/rhm_generative_planner.py::generative_planner --m 2 --quick
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import (
    build_inverse_maps,
    generate_rules_invertible,
    parse_leaves,
)
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_active_query import (
    _batch_from_pool,
    _build_model_classes,
    _count_parameters,
    _generate_episode_batch,
)
from rhm.rhm_edit_control import (
    _apply_block_edit,
    _plan_edits,
    _sample_init_states,
    _train_edit_controller,
)


app = modal.App("rhm-generative-planner", image=image)


# --------------------------------------------------------------------------- #
# Learned generator (block-level masked infiller)                              #
# --------------------------------------------------------------------------- #

def _build_generator():
    """Block infiller G: (sequence with masked blocks [+ optional root]) -> per-
    block level-1 feature logits. Rendered via canonical tuples, so every fill is
    leaf-valid; upper-grammar consistency is learned, not enforced (soft prior)."""
    import torch
    import torch.nn as nn

    class BlockInfiller(nn.Module):
        def __init__(self, vocab_size, sequence_length, block_size, state_dim,
                     n_head, n_layer, root_conditioned):
            super().__init__()
            self.mask_token = vocab_size
            self.block_size = block_size
            self.n_blocks = sequence_length // block_size
            self.root_conditioned = root_conditioned
            self.token_embedding = nn.Embedding(vocab_size + 1, state_dim)
            self.position_embedding = nn.Embedding(sequence_length, state_dim)
            if root_conditioned:
                self.root_embedding = nn.Embedding(vocab_size, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.final_norm = nn.LayerNorm(state_dim)
            self.feature_head = nn.Linear(state_dim, vocab_size)
            self.register_buffer("positions", torch.arange(sequence_length), persistent=False)

        def block_logits(self, observation, root=None):
            tokens = observation.masked_fill(observation < 0, self.mask_token)
            hidden = self.token_embedding(tokens) + self.position_embedding(self.positions)
            if self.root_conditioned and root is not None:
                hidden = hidden + self.root_embedding(root)[:, None, :]
            hidden = self.final_norm(self.encoder(hidden))
            batch, length, dim = hidden.shape
            pooled = hidden.view(batch, self.n_blocks, self.block_size, dim).mean(dim=2)
            return self.feature_head(pooled)  # (B, n_blocks, v)

    return BlockInfiller


def _block_features(leaves, bottom_map, powers, n_blocks, block_size):
    """True level-1 feature of every block (leaf s-tuple -> feature). Blocks are
    always valid tuples here, so features are always defined."""
    codes = (leaves.view(leaves.shape[0], n_blocks, block_size) * powers).sum(-1)
    return bottom_map[codes]


def _train_generator(generator, train_leaves, train_roots, bottom_map, *, batch_size,
                     n_blocks, v, block_size, mask_min, mask_max, n_steps, lr, device):
    """MLM-style: mask a random subset of blocks, predict their true features from
    the rest (+ root if conditioned). Teaches G the on-manifold joint over blocks."""
    import torch
    import torch.nn.functional as F

    powers = v ** torch.arange(block_size, device=device)
    generator.train()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, roots = _batch_from_pool(train_leaves, train_roots, batch_size, device)
        true_feats = _block_features(leaves, bottom_map, powers, n_blocks, block_size)
        n_mask = int(torch.randint(mask_min, mask_max + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        masked_blocks = order[:, :n_mask]
        positions = masked_blocks[:, :, None] * block_size + torch.arange(block_size, device=device)
        observation = leaves.clone()
        observation.scatter_(1, positions.reshape(batch_size, -1),
                             torch.full((batch_size, n_mask * block_size), -1, device=device, dtype=leaves.dtype))
        logits = generator.block_logits(observation, roots if generator.root_conditioned else None)
        mask = torch.zeros(batch_size, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_blocks, True)
        loss = F.cross_entropy(logits[mask], true_feats[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            accuracy = (logits[mask].argmax(dim=-1) == true_feats[mask]).float().mean().item()
            print(f"  generator{'(r*)' if generator.root_conditioned else '   '} "
                  f"step {step:5d}/{n_steps}: loss={loss.item():.4f} fill_acc={accuracy:.3f} (n_mask={n_mask})")


def _regenerate(generator, x, region_blocks, canon, root, *, block_size, sample):
    """Regenerate the given blocks: mask them, let G infer their features from the
    rest, render canonical tuples. Returns a new sequence (leaf-valid)."""
    import torch

    batch = x.shape[0]
    positions = region_blocks[:, :, None] * block_size + torch.arange(block_size, device=x.device)
    flat_positions = positions.reshape(batch, -1)
    observation = x.clone()
    observation.scatter_(1, flat_positions, torch.full_like(flat_positions, -1))
    logits = generator.block_logits(observation, root if generator.root_conditioned else None)
    if sample:
        feats = torch.distributions.Categorical(logits=logits).sample()
    else:
        feats = logits.argmax(dim=-1)
    region_feats = feats.gather(1, region_blocks)  # (B, region_size)
    tuples = canon[region_feats]  # (B, region_size, block_size)
    new = x.clone()
    new.scatter_(1, flat_positions, tuples.reshape(batch, -1))
    return new


def _self_consistency(generator, x, bottom_map, powers, n_blocks, block_size):
    """Leave-one-out self-consistency: mean over blocks of log P_G(actual feature |
    the rest). High = the config is a near-fixed-point of the generator = on-manifold;
    low = the generator is surprised = off-manifold. This is the generator's own
    residual/novelty signal, used as an on-manifold veto during planning."""
    import torch

    batch = x.shape[0]
    actual = (x.view(batch, n_blocks, block_size) * powers).sum(-1)
    actual = bottom_map[actual]  # (B, n_blocks) true feature per block
    total = torch.zeros(batch, device=x.device)
    for j in range(n_blocks):
        observation = x.clone()
        positions = j * block_size + torch.arange(block_size, device=x.device)
        observation[:, positions] = -1
        logp = generator.block_logits(observation)[:, j].log_softmax(dim=-1)
        total = total + logp.gather(1, actual[:, j:j + 1]).squeeze(1)
    return total / n_blocks


def _plan_generative(controller, generator, leaves0, targets, canon, rules, inverse_maps,
                     bottom_map, target_features, *, n_blocks, v, block_size, budget,
                     region_size, sample, consistency_weight, device):
    """Greedy on-manifold planner. Each step: for every candidate region, propose a
    G-regeneration, score the result, commit the best. The score is the controller's
    log P(r*) (direction toward r*) plus consistency_weight * the generator's
    self-consistency (an on-manifold veto). consistency_weight=0 reproduces the
    pure controller-P(r*) selection that still games the soft off-manifold residue.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        controller.eval()
        generator.eval()
        x = leaves0.to(device).clone()
        targets = targets.to(device)
        canon = canon.to(device)
        if target_features is not None:
            target_features = target_features.to(device)
        batch = x.shape[0]
        powers = v ** torch.arange(block_size, device=device)
        n_regions = n_blocks // region_size
        region_index = torch.stack([
            torch.arange(r * region_size, (r + 1) * region_size, device=device)
            for r in range(n_regions)
        ])  # (n_regions, region_size)
        root_arg = targets if generator.root_conditioned else None
        target_prob_trace = []
        valid_trace = []

        for _ in range(budget):
            proposals = []
            scores = []
            for r in range(n_regions):
                region_blocks = region_index[r][None, :].expand(batch, -1)
                proposal = _regenerate(generator, x, region_blocks, canon, root_arg,
                                       block_size=block_size, sample=sample)
                logits = controller.root_logits(controller.state(proposal))
                score = logits.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
                if consistency_weight > 0:
                    score = score + consistency_weight * _self_consistency(
                        generator, proposal, bottom_map, powers, n_blocks, block_size)
                proposals.append(proposal)
                scores.append(score)
            scores = torch.stack(scores, dim=1)  # (B, n_regions)
            proposals = torch.stack(proposals, dim=1)  # (B, n_regions, T)
            best = scores.argmax(dim=1)
            x = proposals.gather(1, best[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)

            current = controller.root_logits(controller.state(x))
            target_prob_trace.append(
                current.log_softmax(dim=-1).gather(1, targets[:, None]).exp().mean().item())
            _, valid = parse_leaves(x.cpu().numpy(), rules, inverse_maps)
            valid_trace.append(float(valid.mean()))

        final_logits = controller.root_logits(controller.state(x))
        target_success = (final_logits.argmax(dim=-1) == targets).float().mean().item()
        target_prob = final_logits.log_softmax(dim=-1).gather(1, targets[:, None]).exp().mean().item()
        gt_roots, gt_valid = parse_leaves(x.cpu().numpy(), rules, inverse_maps)
        gt_roots = torch.from_numpy(gt_roots).to(device)
        gt_valid_t = torch.from_numpy(gt_valid).to(device)
        gt_success = ((gt_roots == targets) & gt_valid_t).float().mean().item()
        gt_feature_match = None
        if target_features is not None:
            final_feats = _block_features(x, bottom_map, powers, n_blocks, block_size)
            gt_feature_match = (final_feats == target_features).float().mean().item()

    return {
        "target_success": target_success,
        "target_prob": target_prob,
        "gt_success": gt_success,
        "gt_on_manifold": float(gt_valid.mean()),
        "gt_feature_match": gt_feature_match,
        "target_prob_by_step": target_prob_trace,
        "on_manifold_by_step": valid_trace,
    }


# --------------------------------------------------------------------------- #
# Experiment                                                                   #
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def generative_planner(
    v: int = 8,
    s: int = 2,
    depth: int = 4,
    m: int = 2,
    rule_seed: int = 0,
    train_seed: int = 1,
    n_train_episodes: int = 100_000,
    n_eval_episodes: int = 4_096,
    state_dim: int = 96,
    controller_steps: int = 12_000,
    generator_steps: int = 12_000,
    batch_size: int = 256,
    edit_budget: int = 6,
    n_corrupt: int = 4,
    region_size: int = 1,
    sample_moves: bool = False,
    consistency_weight: float = 1.0,
    quick: bool = False,
):
    """Does a generator-defined (on-manifold) move space kill the off-manifold
    gaming that raw token edits suffered? Compare G-move planners to the raw-edit
    controller-oracle (games the belief) and the ground-truth oracle (true ceiling).
    """
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    if quick:
        controller_steps = generator_steps = 800
        n_train_episodes, n_eval_episodes = 20_000, 1_024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Generative-planner RHM: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
          f"blocks={n_blocks}, budget={edit_budget}, n_corrupt={n_corrupt}, "
          f"region_size={region_size}, sample_moves={sample_moves}, device={device}")
    started = time.time()

    rules = generate_rules_invertible(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    inverse_maps_torch = [torch.from_numpy(t).to(device) for t in inverse_maps]
    bottom_map = inverse_maps_torch[-1]
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)

    train_roots_np, train_leaves_np = _generate_episode_batch(rules, n_train_episodes, train_seed)
    eval_roots_np, eval_leaves_np = _generate_episode_batch(rules, n_eval_episodes, train_seed + 1)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)
    eval_leaves = torch.from_numpy(eval_leaves_np)
    eval_roots = torch.from_numpy(eval_roots_np)

    BeliefController, _StateFM, _ActionFM = _build_model_classes()
    BlockInfiller = _build_generator()

    # Judge (same lenient controller as rhm_edit_control): reliable ON-manifold.
    controller = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    print(f"Controller parameters: {_count_parameters(controller):,}")
    _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        full_logits = controller.root_logits(controller.state(eval_leaves.to(device)))
        full_acc = (full_logits.argmax(dim=-1) == eval_roots.to(device)).float().mean().item()
    print(f"Controller full-sequence root accuracy: {full_acc:.3f}")

    # Generators: unconditional (direction from search) and r*-conditioned.
    generator = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                              root_conditioned=False).to(device)
    generator_cond = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                                   root_conditioned=True).to(device)
    print(f"Generator parameters: {_count_parameters(generator):,}")
    print("Training unconditional generator")
    _train_generator(generator, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=generator_steps, lr=3e-4, device=device)
    print("Training r*-conditioned generator")
    _train_generator(generator_cond, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=generator_steps, lr=3e-4, device=device)
    for g in (generator, generator_cond):
        g.eval()
        for parameter in g.parameters():
            parameter.requires_grad_(False)

    # Planning episodes (corrupt-and-repair), shared across planners.
    planner_batch = min(2_048, n_eval_episodes)
    leaves0, targets, target_features = _sample_init_states(
        rules, canon_np, inverse_maps, n_episodes=planner_batch, n_blocks=n_blocks, v=v,
        block_size=s, mode="corrupt", n_corrupt=n_corrupt, seed=train_seed + 2)

    # --- baselines (raw block edits): random / controller-oracle / gt_oracle -----
    def _raw_plan(planner, fm=None):
        return _plan_edits(controller, fm, leaves0, targets, canon, rules, inverse_maps,
                           inverse_maps_torch, target_features, n_blocks=n_blocks, v=v,
                           block_size=s, budget=edit_budget, planner=planner, device=device)
    plans = {
        "raw_random": _raw_plan("random"),
        "raw_controller_oracle": _raw_plan("oracle"),      # games the belief off-manifold
        "gt_oracle": _raw_plan("gt_oracle"),               # true achievable ceiling
    }

    # --- on-manifold generator planners ------------------------------------------
    def _gen_plan(gen, cw):
        return _plan_generative(controller, gen, leaves0, targets, canon, rules, inverse_maps,
                                bottom_map, target_features, n_blocks=n_blocks, v=v, block_size=s,
                                budget=edit_budget, region_size=region_size, sample=sample_moves,
                                consistency_weight=cw, device=device)
    plans["gen_uncond"] = _gen_plan(generator, 0.0)
    plans["gen_rootcond"] = _gen_plan(generator_cond, 0.0)
    plans["gen_uncond_consistent"] = _gen_plan(generator, consistency_weight)
    plans["gen_rootcond_consistent"] = _gen_plan(generator_cond, consistency_weight)

    # --- gap-closed vs the true ceiling ------------------------------------------
    rnd = plans["raw_random"]["gt_success"]
    ceil = plans["gt_oracle"]["gt_success"]
    def _closed(value):
        gap = ceil - rnd
        return (value - rnd) / gap if abs(gap) > 1e-8 else 0.0
    gt_gap_closed = {name: _closed(plans[name]["gt_success"])
                     for name in ("raw_controller_oracle", "gen_uncond", "gen_rootcond",
                                  "gen_uncond_consistent", "gen_rootcond_consistent")}

    metrics = {
        "config": {
            "v": v, "s": s, "depth": depth, "m": m, "rule_seed": rule_seed,
            "sequence_length": sequence_length, "n_blocks": n_blocks,
            "edit_budget": edit_budget, "n_corrupt": n_corrupt, "region_size": region_size,
            "sample_moves": sample_moves, "state_dim": state_dim,
            "controller_steps": controller_steps, "generator_steps": generator_steps,
        },
        "controller_full_accuracy": full_acc,
        "planning": plans,
        "gt_gap_closed": gt_gap_closed,
        "elapsed_seconds": time.time() - started,
    }

    print(f"\n=== Planning (m={m}, corrupt={n_corrupt}, budget={edit_budget}, region={region_size}) ===")
    print(f"  {'planner':22s} {'P(r*)':>7s} {'ctrl_succ':>9s} {'gt_succ':>8s} {'gt_valid':>8s} {'feat_match':>10s}")
    for name, r in plans.items():
        fm = r["gt_feature_match"]
        print(f"  {name:22s} {r['target_prob']:7.3f} {r['target_success']:9.3f} "
              f"{r['gt_success']:8.3f} {r['gt_on_manifold']:8.3f} "
              f"{(f'{fm:.3f}' if fm is not None else 'n/a'):>10s}")
    print("  fraction of random->gt_oracle TRUE-success gap closed:")
    for name, val in gt_gap_closed.items():
        print(f"    {name:22s} {val:+.2f}")
    print(f"  (raw_controller_oracle final on-manifold rate = {plans['raw_controller_oracle']['gt_on_manifold']:.3f}; "
          f"gen_uncond = {plans['gen_uncond']['gt_on_manifold']:.3f})")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}_c{n_corrupt}_r{region_size}"
    output_dir = f"{DATA_DIR}/rhm_generative_planner/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    generative_planner.remote(m=m, quick=quick)
