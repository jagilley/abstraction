"""Active-RHM Part 3: planning in LATENTS with a learned MONTE-CARLO value.

Context (RHM_EDIT_CONTROL_README.md). Part 2 reached true-success ~0.65 with a
generator (on-manifold moves) + a cerebellar self-consistency veto, but greedy /
token-space. Part 3 attacks the two residuals -- off-manifold DRIFT and greedy
MYOPIA -- by (1) a value grounded in truth (fixes drift, ungameable) and (2) a
latent cerebellar FM for cheap multi-step lookahead (fixes myopia), tested
head-to-head against token-space.

The value design is the crux (and a mid-experiment correction). A *faithful
verifier* -- "is this a valid r* config?" -- is honest but SPARSE: it is silent on
the ~99% of half-edited configs the planner visits, so greedy barely works and
lookahead COLLAPSES (its internal rollout steers by the silent signal and wanders).
A treasure detector that only beeps on the treasure. The fix is a proper VALUE:
not the sparse reward ("am I at the goal?") but the dense expected-return ("how
likely am I to REACH the goal from here?"), which is dense even when the reward is
sparse because it smears the sparse signal backward over experience. We LEARN it,
Monte-Carlo: roll a cheap behaviour policy from many starts, label every visited
state by whether that rollout eventually reached a true r* config, regress a
compact root-conditioned V(z, r*) to it. Dense, faithful, and exactly what
lookahead needs to climb -- the standard model-based-RL shape (learned world model
+ learned value), ~the cortex/cerebellum division we started from.

Run from experiments/:
  modal run rhm/rhm_latent_planner.py::latent_planner --m 2 --quick   # smoke
  modal run --detach rhm/rhm_latent_planner.py::latent_planner --m 2  # (and 3,4)
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_invertible, parse_leaves
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
from rhm.rhm_generative_planner import (
    _block_features,
    _build_generator,
    _plan_generative,
    _regenerate,
    _train_generator,
)


app = modal.App("rhm-latent-planner", image=image)


def _region_index(n_blocks, region_size, device):
    import torch
    n_regions = n_blocks // region_size
    return torch.stack([
        torch.arange(r * region_size, (r + 1) * region_size, device=device)
        for r in range(n_regions)
    ]), n_regions


def _corrupt_starts(leaves_pool, roots_pool, canon, *, batch_size, n_blocks, v,
                    block_size, n_corrupt, device):
    """Sample corrupt-and-repair starts: a valid config with c~[1,n_corrupt] blocks
    set to random features. Returns (x, true_root)."""
    import torch
    leaves, roots = _batch_from_pool(leaves_pool, roots_pool, batch_size, device)
    x = leaves.clone()
    c = int(torch.randint(1, n_corrupt + 1, ()).item())
    order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
    for i in range(c):
        feats = torch.randint(0, v, (batch_size,), device=device)
        x = _apply_block_edit(x, order[:, i], feats, canon, block_size)
    return x, roots


# --------------------------------------------------------------------------- #
# Compact root-conditioned Monte-Carlo value                                  #
# --------------------------------------------------------------------------- #

def _build_value_head():
    import torch
    import torch.nn as nn

    class MCValueHead(nn.Module):
        """V(z, r*) -> scalar logit of eventually reaching a valid r* config."""
        def __init__(self, state_dim, vocab_size):
            super().__init__()
            self.root_embedding = nn.Embedding(vocab_size, state_dim)
            self.net = nn.Sequential(
                nn.LayerNorm(2 * state_dim),
                nn.Linear(2 * state_dim, 4 * state_dim),
                nn.GELU(),
                nn.Linear(4 * state_dim, 1),
            )

        def forward(self, z, root):
            return self.net(torch.cat([z, self.root_embedding(root)], dim=-1)).squeeze(-1)

    return MCValueHead


def _behavior_step(controller, generator, x, targets, canon, region_index, n_regions,
                   block_size, epsilon, device):
    """One controller-greedy generator move (regenerate the region maximizing the
    controller's P(r*)), with epsilon-random exploration. The cheap behaviour policy
    whose Monte-Carlo outcomes label the value."""
    import torch
    batch = x.shape[0]
    scores, proposals = [], []
    for k in range(n_regions):
        region_blocks = region_index[torch.full((batch,), k, device=device, dtype=torch.long)]
        proposal = _regenerate(generator, x, region_blocks, canon, None, block_size=block_size, sample=False)
        s = controller.root_logits(controller.state(proposal)).log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
        scores.append(s)
        proposals.append(proposal)
    scores = torch.stack(scores, dim=1)
    proposals = torch.stack(proposals, dim=1)
    chosen = scores.argmax(dim=1)
    explore = torch.rand(batch, device=device) < epsilon
    chosen = torch.where(explore, torch.randint(0, n_regions, (batch,), device=device), chosen)
    return proposals.gather(1, chosen[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)


def _collect_value_data(controller, generator, leaves_pool, roots_pool, canon, region_index,
                        n_regions, rules, inverse_maps, *, n_episodes, batch_size, n_blocks,
                        v, block_size, n_corrupt, budget, epsilon, device):
    """Roll the behaviour policy from many corrupt starts; return (configs, roots,
    success) for EVERY visited state, labelled by its rollout's terminal success."""
    import torch
    configs, roots_all, success_all = [], [], []
    collected = 0
    while collected < n_episodes:
        batch = min(batch_size, n_episodes - collected)
        collected += batch
        x, roots = _corrupt_starts(leaves_pool, roots_pool, canon, batch_size=batch,
                                   n_blocks=n_blocks, v=v, block_size=block_size,
                                   n_corrupt=n_corrupt, device=device)
        trajectory = [x.clone()]
        for _ in range(budget):
            x = _behavior_step(controller, generator, x, roots, canon, region_index,
                               n_regions, block_size, epsilon, device)
            trajectory.append(x.clone())
        gt_roots, valid = parse_leaves(x.cpu().numpy(), rules, inverse_maps)
        success = torch.from_numpy(((gt_roots == roots.cpu().numpy()) & valid).astype(np.float32))
        for state in trajectory:
            configs.append(state.cpu())
            roots_all.append(roots.cpu())
            success_all.append(success)
    return torch.cat(configs), torch.cat(roots_all), torch.cat(success_all)


def _train_value_mc(value, controller, configs, roots, success, *, batch_size, n_steps, lr, device):
    import torch
    import torch.nn.functional as F
    value.train()
    optimizer = torch.optim.AdamW(value.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)
    total = configs.shape[0]
    base_rate = success.mean().item()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, total, (batch_size,))
        x = configs[idx].to(device)
        r = roots[idx].to(device)
        y = success[idx].to(device)
        with torch.no_grad():
            z = controller.state(x)
        logit = value(z, r)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            pred = (logit > 0).float()
            acc = (pred == y).float().mean().item()
            print(f"  value(MC) step {step:5d}/{n_steps}: loss={loss.item():.4f} acc={acc:.3f} "
                  f"(base success rate {base_rate:.3f})")


def _value_check(value, controller, leaves_pool, roots_pool, canon, *, n_blocks, v,
                 block_size, n_corrupt, device):
    """Is V a dense distance-to-goal proxy? Mean predicted success should fall as the
    number of corrupted blocks (roughly, distance from the goal) rises."""
    import torch
    means = {}
    with torch.no_grad():
        for c in range(0, n_corrupt + 1):
            leaves, roots = _batch_from_pool(leaves_pool, roots_pool, 1024, device)
            x = leaves.clone()
            if c > 0:
                order = torch.rand(1024, n_blocks, device=device).argsort(dim=1)
                for i in range(c):
                    feats = torch.randint(0, v, (1024,), device=device)
                    x = _apply_block_edit(x, order[:, i], feats, canon, block_size)
            means[c] = torch.sigmoid(value(controller.state(x), roots)).mean().item()
    return means


# --------------------------------------------------------------------------- #
# Latent cerebellar forward model                                             #
# --------------------------------------------------------------------------- #

def _sample_visited(controller, generator, leaves_pool, roots_pool, canon, region_index,
                    n_regions, *, batch_size, n_blocks, v, block_size, n_corrupt, budget, device):
    """A config from the visited distribution (corrupt start + 0..budget random
    generator regenerations). Used to train the latent FM."""
    import torch
    x, roots = _corrupt_starts(leaves_pool, roots_pool, canon, batch_size=batch_size,
                               n_blocks=n_blocks, v=v, block_size=block_size,
                               n_corrupt=n_corrupt, device=device)
    if torch.rand(()).item() < 0.5:
        g = int(torch.randint(0, budget + 1, ()).item())
        for _ in range(g):
            k = torch.randint(0, n_regions, (batch_size,), device=device)
            x = _regenerate(generator, x, region_index[k], canon, None, block_size=block_size, sample=False)
    return x, roots


def _train_latent_fm(fm, controller, generator, leaves_pool, roots_pool, canon,
                     region_index, n_regions, *, batch_size, n_blocks, v, block_size,
                     n_corrupt, budget, n_steps, lr, device):
    import torch
    import torch.nn.functional as F
    fm.train()
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)
    for step in range(1, n_steps + 1):
        x, _ = _sample_visited(controller, generator, leaves_pool, roots_pool, canon,
                               region_index, n_regions, batch_size=batch_size, n_blocks=n_blocks,
                               v=v, block_size=block_size, n_corrupt=n_corrupt, budget=budget, device=device)
        k = torch.randint(0, n_regions, (batch_size,), device=device)
        with torch.no_grad():
            b = controller.state(x)
            x2 = _regenerate(generator, x, region_index[k], canon, None, block_size=block_size, sample=False)
            target = controller.state(x2) - b
        pred = fm(b, k)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            cos = F.cosine_similarity(pred, target, dim=-1).mean().item()
            print(f"  latent FM step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cos:.3f}")


def _latent_fm_check(fm, value, controller, generator, leaves0, targets, canon,
                     region_index, n_regions, *, block_size, device):
    """One-step check: FM delta accuracy, and (what matters for planning) whether the
    FM-predicted next latent gives the same VALUE ranking over moves as the true one."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        x = leaves0.to(device)
        targets = targets.to(device)
        b = controller.state(x)
        batch = x.shape[0]
        true_v = torch.empty(batch, n_regions, device=device)
        pred_v = torch.empty(batch, n_regions, device=device)
        cos_accum = 0.0
        for k in range(n_regions):
            kk = torch.full((batch,), k, device=device, dtype=torch.long)
            x2 = _regenerate(generator, x, region_index[kk], canon, None, block_size=block_size, sample=False)
            true_next = controller.state(x2)
            pred_next = b + fm(b, kk)
            cos_accum += F.cosine_similarity(pred_next - b, true_next - b, dim=-1).mean().item()
            true_v[:, k] = value(true_next, targets)
            pred_v[:, k] = value(pred_next, targets)
        top1 = (true_v.argmax(1) == pred_v.argmax(1)).float().mean().item()
        tc = true_v - true_v.mean(1, keepdim=True)
        pc = pred_v - pred_v.mean(1, keepdim=True)
        rank = ((tc * pc).sum(1) / (tc.norm(dim=1) * pc.norm(dim=1)).clamp_min(1e-8)).mean().item()
    return {"delta_cos": cos_accum / n_regions, "value_top1_agree": top1, "value_rank_corr": rank}


# --------------------------------------------------------------------------- #
# Lookahead planners (latent vs token), MPC                                    #
# --------------------------------------------------------------------------- #

def _encode_chunked(controller, x, chunk=4096):
    import torch
    if x.shape[0] <= chunk:
        return controller.state(x)
    return torch.cat([controller.state(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)], dim=0)


def _latent_lookahead(b, fm, value, targets, n_regions, depth):
    """For each candidate FIRST move, greedily roll the FM `depth` steps in latent
    space (choosing each next move by V), return the final V(r*). depth=1 => greedy."""
    import torch
    batch = b.shape[0]
    lat = torch.stack([b + fm(b, torch.full((batch,), k, device=b.device, dtype=torch.long))
                       for k in range(n_regions)], dim=1)  # (B, nr, D)
    for _ in range(depth - 1):
        flat = lat.reshape(-1, lat.shape[-1])
        tgt = targets.repeat_interleave(n_regions)
        cand = torch.stack([flat + fm(flat, torch.full((flat.shape[0],), k, device=b.device, dtype=torch.long))
                            for k in range(n_regions)], dim=1)  # (fb, nr, D)
        vv = value(cand.reshape(-1, cand.shape[-1]), tgt.repeat_interleave(n_regions)).reshape(flat.shape[0], n_regions)
        best = vv.argmax(1)
        flat = cand[torch.arange(flat.shape[0], device=b.device), best]
        lat = flat.reshape(batch, n_regions, -1)
    vfin = value(lat.reshape(-1, lat.shape[-1]), targets.repeat_interleave(n_regions)).reshape(batch, n_regions)
    return vfin


def _token_lookahead(x, controller, generator, value, targets, canon, region_index,
                     n_regions, depth, block_size):
    """Token-space analog: materialize every branch (G -> tokens -> re-encode)."""
    import torch
    batch = x.shape[0]
    branches = torch.stack([
        _regenerate(generator, x, region_index[torch.full((batch,), k, device=x.device, dtype=torch.long)],
                    canon, None, block_size=block_size, sample=False)
        for k in range(n_regions)], dim=1)  # (B, nr, T)
    for _ in range(depth - 1):
        flat = branches.reshape(-1, branches.shape[-1])
        fb = flat.shape[0]
        tgt = targets.repeat_interleave(n_regions)
        cand = torch.stack([
            _regenerate(generator, flat, region_index[torch.full((fb,), k, device=x.device, dtype=torch.long)],
                        canon, None, block_size=block_size, sample=False)
            for k in range(n_regions)], dim=1)  # (fb, nr, T)
        vv = value(_encode_chunked(controller, cand.reshape(-1, cand.shape[-1])),
                   tgt.repeat_interleave(n_regions)).reshape(fb, n_regions)
        best = vv.argmax(1)
        flat = cand[torch.arange(fb, device=x.device), best]
        branches = flat.reshape(batch, n_regions, -1)
    vfin = value(_encode_chunked(controller, branches.reshape(-1, branches.shape[-1])),
                 targets.repeat_interleave(n_regions)).reshape(batch, n_regions)
    return vfin


def _plan_lookahead(controller, generator, fm, value, leaves0, targets, canon, rules,
                    inverse_maps, bottom_map, target_features, *, n_blocks, v, block_size,
                    budget, region_size, depth, mode, device):
    import torch
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        if fm is not None:
            fm.eval()
        region_index, n_regions = _region_index(n_blocks, region_size, device)
        x = leaves0.to(device).clone()
        targets = targets.to(device)
        canon = canon.to(device)
        powers = v ** torch.arange(block_size, device=device)
        valid_trace = []
        for _ in range(budget):
            b = controller.state(x)
            if mode == "latent":
                scores = _latent_lookahead(b, fm, value, targets, n_regions, depth)
            else:
                scores = _token_lookahead(x, controller, generator, value, targets, canon,
                                          region_index, n_regions, depth, block_size)
            best = scores.argmax(1)
            x = _regenerate(generator, x, region_index[best], canon, None, block_size=block_size, sample=False)
            _, valid = parse_leaves(x.cpu().numpy(), rules, inverse_maps)
            valid_trace.append(float(valid.mean()))
        gt_roots, gt_valid = parse_leaves(x.cpu().numpy(), rules, inverse_maps)
        gt_roots = torch.from_numpy(gt_roots).to(device)
        gt_valid_t = torch.from_numpy(gt_valid).to(device)
        gt_success = ((gt_roots == targets) & gt_valid_t).float().mean().item()
        feat = None
        if target_features is not None:
            final_feats = _block_features(x, bottom_map, powers, n_blocks, block_size)
            feat = (final_feats == target_features.to(device)).float().mean().item()
    return {"gt_success": gt_success, "gt_on_manifold": float(gt_valid.mean()),
            "gt_feature_match": feat, "on_manifold_by_step": valid_trace}


# --------------------------------------------------------------------------- #
# Experiment                                                                   #
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def latent_planner(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 4_096, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256,
    edit_budget: int = 6, n_corrupt: int = 4, region_size: int = 1, plan_depth: int = 3,
    explore_eps: float = 0.3, quick: bool = False,
):
    """Learned MC value + latent cerebellar FM; latent vs token-space lookahead."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Latent-planner RHM: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
          f"blocks={n_blocks}, budget={edit_budget}, n_corrupt={n_corrupt}, "
          f"region_size={region_size}, plan_depth={plan_depth}, device={device}")
    started = time.time()

    rules = generate_rules_invertible(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    inverse_maps_torch = [torch.from_numpy(t).to(device) for t in inverse_maps]
    bottom_map = inverse_maps_torch[-1]
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)

    train_roots_np, train_leaves_np = _generate_episode_batch(rules, n_train_episodes, train_seed)
    eval_roots_np, eval_leaves_np = _generate_episode_batch(rules, n_eval_episodes, train_seed + 1)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    BeliefController, _StateFM, ActionForwardModel = _build_model_classes()
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

    # --- learned Monte-Carlo value ----------------------------------------------
    print(f"Collecting value data ({value_episodes} rollouts, eps={explore_eps})")
    configs, roots_buf, success_buf = _collect_value_data(
        controller, generator, train_leaves, train_roots, canon, region_index, n_regions,
        rules, inverse_maps, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks,
        v=v, block_size=s, n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
    print(f"  buffer: {configs.shape[0]} states, terminal success rate {success_buf.mean().item():.3f}")
    value = MCValueHead(state_dim, v).to(device)
    print(f"Value head parameters: {_count_parameters(value):,}")
    _train_value_mc(value, controller, configs, roots_buf, success_buf,
                    batch_size=512, n_steps=value_steps, lr=3e-4, device=device)

    # --- latent cerebellar FM ----------------------------------------------------
    latent_fm = ActionForwardModel(state_dim, hidden_dim=2 * state_dim, n_actions=n_regions,
                                   action_dim=16).to(device)
    print(f"Latent FM parameters: {_count_parameters(latent_fm):,}")
    _train_latent_fm(latent_fm, controller, generator, train_leaves, train_roots, canon,
                     region_index, n_regions, batch_size=batch_size, n_blocks=n_blocks, v=v,
                     block_size=s, n_corrupt=n_corrupt, budget=edit_budget, n_steps=fm_steps,
                     lr=1e-3, device=device)
    for module in (value, latent_fm):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # --- planning episodes + foundational checks --------------------------------
    planner_batch = min(2_048, n_eval_episodes)
    leaves0, targets, target_features = _sample_init_states(
        rules, canon_np, inverse_maps, n_episodes=planner_batch, n_blocks=n_blocks, v=v,
        block_size=s, mode="corrupt", n_corrupt=n_corrupt, seed=train_seed + 2)

    v_by_c = _value_check(value, controller, train_leaves, train_roots, canon, n_blocks=n_blocks,
                          v=v, block_size=s, n_corrupt=n_corrupt, device=device)
    fm_check = _latent_fm_check(latent_fm, value, controller, generator, leaves0, targets, canon,
                                region_index, n_regions, block_size=s, device=device)
    print("\n=== Foundational checks ===")
    print("  value density (mean predicted success by #corrupted blocks; should fall with distance):")
    print("    " + "  ".join(f"c={c}:{val:.2f}" for c, val in v_by_c.items()))
    print(f"  latent FM: delta_cos={fm_check['delta_cos']:.3f}, "
          f"value_top1_agree={fm_check['value_top1_agree']:.3f}, "
          f"value_rank_corr={fm_check['value_rank_corr']:.3f}")

    def _raw(planner):
        return _plan_edits(controller, None, leaves0, targets, canon, rules, inverse_maps,
                           inverse_maps_torch, target_features, n_blocks=n_blocks, v=v,
                           block_size=s, budget=edit_budget, planner=planner, device=device)

    def _look(mode, d):
        return _plan_lookahead(controller, generator, latent_fm, value, leaves0, targets, canon,
                               rules, inverse_maps, bottom_map, target_features, n_blocks=n_blocks,
                               v=v, block_size=s, budget=edit_budget, region_size=region_size,
                               depth=d, mode=mode, device=device)

    plans = {
        "gt_oracle": _raw("gt_oracle"),
        "stage1_gen_veto": _plan_generative(controller, generator, leaves0, targets, canon, rules,
                                            inverse_maps, bottom_map, target_features, n_blocks=n_blocks,
                                            v=v, block_size=s, budget=edit_budget, region_size=region_size,
                                            sample=False, consistency_weight=1.0, device=device),
        "value_token_greedy": _look("token", 1),
        "value_token_lookahead": _look("token", plan_depth),
        "value_latent_greedy": _look("latent", 1),
        "value_latent_lookahead": _look("latent", plan_depth),
    }

    rnd = _raw("random")["gt_success"]
    ceil = plans["gt_oracle"]["gt_success"]
    def _closed(x):
        gap = ceil - rnd
        return (x - rnd) / gap if abs(gap) > 1e-8 else 0.0
    gap_closed = {name: _closed(plans[name]["gt_success"]) for name in plans}

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "sequence_length": sequence_length,
                   "n_blocks": n_blocks, "edit_budget": edit_budget, "n_corrupt": n_corrupt,
                   "region_size": region_size, "plan_depth": plan_depth, "explore_eps": explore_eps,
                   "value_episodes": value_episodes, "state_dim": state_dim},
        "value_density_by_corruption": v_by_c,
        "value_buffer_success_rate": success_buf.mean().item(),
        "latent_fm_check": fm_check,
        "planning": plans,
        "gap_closed": gap_closed,
        "elapsed_seconds": time.time() - started,
    }

    print(f"\n=== Planning (m={m}, corrupt={n_corrupt}, budget={edit_budget}, depth={plan_depth}) ===")
    print(f"  {'planner':24s} {'gt_succ':>8s} {'gt_valid':>8s} {'feat':>6s} {'gap_closed':>10s}")
    for name, r in plans.items():
        feat = r.get("gt_feature_match")
        print(f"  {name:24s} {r['gt_success']:8.3f} {r['gt_on_manifold']:8.3f} "
              f"{(f'{feat:.2f}' if feat is not None else 'n/a'):>6s} {gap_closed[name]:+10.2f}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}_c{n_corrupt}_d{plan_depth}"
    output_dir = f"{DATA_DIR}/rhm_latent_planner/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    latent_planner.remote(m=m, quick=quick)
