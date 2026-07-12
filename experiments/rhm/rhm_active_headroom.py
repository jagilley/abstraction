"""Active-RHM Step 0.5: is the VoI-saturation a property of the OBJECTIVE?

Step 0 (`rhm_active_internal.py`) found that a root-prediction-trained belief
already saturates the observation-decodable value-of-information at every m: a
high-capacity raw-observation head extracts no more VoI-ranking signal than the
pooled belief. That removed the belief-*pooling* bottleneck as the limiter and
localized any remaining internalization headroom to the controller's OBJECTIVE.

Reaching's internalization had headroom because its baseline was a model-free
POLICY (`mf`), not a sufficient statistic -- a policy operator solves the task
without being organized for forward-model planning, and internalization
reorganizes it. RHM's baseline is a root-predictor (a sufficient statistic), so
it VoI-saturates. This script tests whether that saturation is caused by the
root-prediction objective specifically, or is intrinsic to any competent belief.

Controlled experiment (hold everything from Step 0 fixed, vary ONE thing):
  - C_root : root-predictor controller (Step 0's controller). Defines the VoI
             TARGET  H(root | C_root.state(next_obs))  AND the oracle AND the
             final root prediction. Fixed "truth".
  - C_pol  : model-free query policy. Its belief is shaped ONLY by imitating
             C_root's greedy oracle reveals (no root-prediction pressure). The
             faithful `mf` analog.

Then probe how much of the (fixed) VoI target each belief carries:
  belief-VoI corr on C_root   (reproduces Step 0)
  belief-VoI corr on C_pol    (NEW -- policy-shaped belief)
  raw-obs ceiling             (reproduces Step 0)

Read:
  C_pol ~= C_root ~= ceiling  -> VoI-sufficiency is objective-independent; even a
                                 policy baseline saturates -> no internalization
                                 headroom -> clean boundary result.
  C_pol  <  ceiling           -> policy-shaping loses observation-decodable VoI
                                 -> internalization (int_plan) has a target -> go.

Run from experiments/ (three parallel detached runs, one per m):
  modal run --detach rhm/rhm_active_headroom.py::headroom_probe --m 2
  modal run --detach rhm/rhm_active_headroom.py::headroom_probe --m 3
  modal run --detach rhm/rhm_active_headroom.py::headroom_probe --m 4
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_active_query import (
    _batch_from_pool,
    _build_model_classes,
    _count_parameters,
    _generate_episode_batch,
    _plan_queries,
    _reveal_blocks,
    _sample_transition,
    _train_controller,
)
from rhm.rhm_active_voi import (
    _build_entropy_model,
    _posterior_entropy,
)
from rhm.rhm_active_internal import (
    _build_observation_entropy_model,
    _train_obs_entropy_fm,
    _voi_ranking_diagnostic_obs,
)


app = modal.App("rhm-active-headroom", image=image)


# --------------------------------------------------------------------------- #
# Model-free query policy (the `mf` analog): belief shaped by imitating reveals
# --------------------------------------------------------------------------- #

def _oracle_action(target_ctrl, leaves, roots, revealed, block_size, device):
    """Greedy content-peeking best next reveal: the candidate whose true content
    most reduces target_ctrl's root CE. Returns (action, base_obs, queried)."""
    import torch
    import torch.nn.functional as F

    batch_size, sequence_length = leaves.shape
    n_actions = sequence_length // block_size
    base_obs = _reveal_blocks(leaves, revealed, block_size, mask_token=-1)
    queried = torch.zeros(batch_size, n_actions, dtype=torch.bool, device=device)
    if revealed.shape[1] > 0:
        queried.scatter_(1, revealed, True)

    offsets = torch.arange(block_size, device=device)
    scores = torch.full((batch_size, n_actions), float("inf"), device=device)
    for a in range(n_actions):
        cand = base_obs.clone()
        positions = a * block_size + offsets
        cand[:, positions] = leaves[:, positions]
        logits = target_ctrl.root_logits(target_ctrl.state(cand))
        scores[:, a] = F.cross_entropy(logits, roots, reduction="none")
    scores = scores.masked_fill(queried, float("inf"))
    return scores.argmin(dim=1), base_obs, queried


def _train_policy_controller(policy_ctrl, policy_head, target_ctrl, train_leaves, train_roots, *,
                             batch_size, budget, block_size, n_steps, lr, device):
    """Shape policy_ctrl's belief purely by imitating target_ctrl's oracle reveals."""
    import torch
    import torch.nn.functional as F

    target_ctrl.eval()
    policy_ctrl.train()
    policy_head.train()
    params = list(policy_ctrl.parameters()) + list(policy_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, roots = _batch_from_pool(train_leaves, train_roots, batch_size, device)
        n_actions = leaves.shape[1] // block_size
        n_revealed = int(torch.randint(0, budget, ()).item())  # 0..budget-1 already revealed
        order = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        revealed = order[:, :n_revealed]
        with torch.no_grad():
            target_action, base_obs, queried = _oracle_action(
                target_ctrl, leaves, roots, revealed, block_size, device)
        logits = policy_head(policy_ctrl.state(base_obs)).masked_fill(queried, float("-inf"))
        loss = F.cross_entropy(logits, target_action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            acc = (logits.argmax(dim=1) == target_action).float().mean().item()
            print(f"  policy step {step:5d}/{n_steps}: loss={loss.item():.4f} oracle-match={acc:.3f}")


# --------------------------------------------------------------------------- #
# Split VoI head training / diagnostic: belief from one controller, target from
# another (reduces to Step 0's functions when belief_ctrl is target_ctrl).
# --------------------------------------------------------------------------- #

def _train_entropy_fm_split(forward_model, belief_ctrl, target_ctrl, train_leaves, *,
                            batch_size, budget, block_size, n_steps, lr, device):
    import torch
    import torch.nn.functional as F

    belief_ctrl.eval()
    target_ctrl.eval()
    forward_model.train()
    optimizer = torch.optim.AdamW(forward_model.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, _ = _batch_from_pool(train_leaves, train_leaves, batch_size, device)
        observation, action, next_observation = _sample_transition(leaves, budget, block_size)
        with torch.no_grad():
            state = belief_ctrl.state(observation)
            target_entropy = _posterior_entropy(target_ctrl.root_logits(target_ctrl.state(next_observation)))
        prediction = forward_model(state, action)
        loss = F.mse_loss(prediction, target_entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forward_model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            centered_p = prediction - prediction.mean()
            centered_t = target_entropy - target_entropy.mean()
            denom = (centered_p.norm() * centered_t.norm()).clamp_min(1e-8)
            corr = (centered_p @ centered_t / denom).item()
            print(f"  split entropy FM step {step:5d}/{n_steps}: mse={loss.item():.4f} corr={corr:.3f}")


def _voi_ranking_diagnostic_split(forward_model, belief_ctrl, target_ctrl, leaves, *,
                                  budget, block_size, batch_size, device):
    """belief from belief_ctrl, realized-entropy target from target_ctrl."""
    import torch

    with torch.no_grad():
        forward_model.eval()
        eval_leaves = leaves[:batch_size].to(device)
        n_actions = eval_leaves.shape[1] // block_size
        base_order = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        base_actions = base_order[:, :1]
        base_observation = _reveal_blocks(eval_leaves, base_actions, block_size, mask_token=-1)
        base_state = belief_ctrl.state(base_observation)

        candidates = base_order[:, 1:]
        candidate_count = candidates.shape[1]
        candidate_flat = candidates.reshape(-1)
        states = base_state[:, None, :].expand(-1, candidate_count, -1).reshape(-1, base_state.shape[-1])
        repeated_base = base_actions[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 1)
        all_actions = torch.cat((repeated_base, candidate_flat[:, None]), dim=1)
        expanded_leaves = eval_leaves[:, None, :].expand(-1, candidate_count, -1).reshape(-1, eval_leaves.shape[1])
        next_observations = _reveal_blocks(expanded_leaves, all_actions, block_size, mask_token=-1)
        realized_entropy = _posterior_entropy(target_ctrl.root_logits(target_ctrl.state(next_observations)))
        predicted_entropy = forward_model(states, candidate_flat)

        centered_p = predicted_entropy - predicted_entropy.mean()
        centered_t = realized_entropy - realized_entropy.mean()
        pearson = (centered_p @ centered_t / (centered_p.norm() * centered_t.norm()).clamp_min(1e-8)).item()

        predicted_grid = predicted_entropy.view(batch_size, candidate_count)
        realized_grid = realized_entropy.view(batch_size, candidate_count)
        top1_agreement = (predicted_grid.argmin(dim=1) == realized_grid.argmin(dim=1)).float().mean().item()
        pc = predicted_grid - predicted_grid.mean(dim=1, keepdim=True)
        tc = realized_grid - realized_grid.mean(dim=1, keepdim=True)
        per_instance = (pc * tc).sum(dim=1) / (pc.norm(dim=1) * tc.norm(dim=1)).clamp_min(1e-8)
        per_instance_corr = per_instance.mean().item()

    return {
        "pearson_all_pairs": pearson,
        "per_instance_query_corr": per_instance_corr,
        "top1_argmin_agreement": top1_agreement,
        "random_top1_baseline": 1.0 / candidate_count,
    }


# --------------------------------------------------------------------------- #
# Decoupled planners: reveal-selection uses `belief_ctrl`; final root prediction
# always uses `infer_ctrl` (= C_root), so only the selection signal differs.
# --------------------------------------------------------------------------- #

def _plan_queries_voi_split(infer_ctrl, belief_ctrl, forward_model, leaves, roots, *,
                            budget, block_size, device):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        infer_ctrl.eval()
        belief_ctrl.eval()
        forward_model.eval()
        leaves = leaves.to(device)
        roots = roots.to(device)
        batch_size, sequence_length = leaves.shape
        n_actions = sequence_length // block_size
        observation = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=device)
        queried = torch.zeros(batch_size, n_actions, dtype=torch.bool, device=device)

        for _ in range(budget):
            state = belief_ctrl.state(observation)
            candidates = torch.arange(n_actions, device=device)[None, :].expand(batch_size, -1)
            repeated_state = state[:, None, :].expand(-1, n_actions, -1).reshape(-1, state.shape[-1])
            predicted_entropy = forward_model(repeated_state, candidates.reshape(-1)).view(batch_size, n_actions)
            chosen = predicted_entropy.masked_fill(queried, float("inf")).argmin(dim=1)
            positions = chosen[:, None] * block_size + torch.arange(block_size, device=device)
            observation.scatter_(1, positions, leaves.gather(1, positions))
            queried.scatter_(1, chosen[:, None], True)

        final_logits = infer_ctrl.root_logits(infer_ctrl.state(observation))
        return {
            "root_accuracy": (final_logits.argmax(dim=-1) == roots).float().mean().item(),
            "root_ce": F.cross_entropy(final_logits, roots).item(),
        }


def _plan_queries_policy(infer_ctrl, belief_ctrl, policy_head, leaves, roots, *,
                         budget, block_size, device):
    """Direct model-free policy planner: pick reveals by argmax policy logits."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        infer_ctrl.eval()
        belief_ctrl.eval()
        policy_head.eval()
        leaves = leaves.to(device)
        roots = roots.to(device)
        batch_size, sequence_length = leaves.shape
        n_actions = sequence_length // block_size
        observation = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=device)
        queried = torch.zeros(batch_size, n_actions, dtype=torch.bool, device=device)

        for _ in range(budget):
            logits = policy_head(belief_ctrl.state(observation))
            chosen = logits.masked_fill(queried, float("-inf")).argmax(dim=1)
            positions = chosen[:, None] * block_size + torch.arange(block_size, device=device)
            observation.scatter_(1, positions, leaves.gather(1, positions))
            queried.scatter_(1, chosen[:, None], True)

        final_logits = infer_ctrl.root_logits(infer_ctrl.state(observation))
        return {
            "root_accuracy": (final_logits.argmax(dim=-1) == roots).float().mean().item(),
            "root_ce": F.cross_entropy(final_logits, roots).item(),
        }


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def headroom_probe(
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
    policy_steps: int = 12_000,
    forward_steps: int = 24_000,
    obs_hidden: int = 128,
    obs_layers: int = 4,
    obs_heads: int = 4,
    batch_size: int = 256,
    budget: int = 4,
    quick: bool = False,
):
    """Does a model-free-policy belief carry less VoI than a root-predictor belief?"""
    import time
    import torch
    import torch.nn as nn

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_actions = sequence_length // s
    if budget >= n_actions:
        raise ValueError("budget must leave at least one unrevealed query action.")
    if quick:
        controller_steps, policy_steps, forward_steps = 400, 400, 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Headroom probe: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
          f"actions={n_actions}, budget={budget}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    train_roots_np, train_leaves_np = _generate_episode_batch(rules, n_train_episodes, train_seed)
    eval_roots_np, eval_leaves_np = _generate_episode_batch(rules, n_eval_episodes, train_seed + 1)
    train_roots = torch.from_numpy(train_roots_np)
    train_leaves = torch.from_numpy(train_leaves_np)
    eval_roots = torch.from_numpy(eval_roots_np)
    eval_leaves = torch.from_numpy(eval_leaves_np)

    BeliefController, _StateFM, _ActionFM = _build_model_classes()
    EntropyForwardModel = _build_entropy_model()
    ObservationEntropyModel = _build_observation_entropy_model()

    # --- C_root: root-predictor controller (Step 0's controller = fixed truth) --
    print("\n=== C_root: root-predictor controller ===")
    c_root = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    print(f"C_root parameters: {_count_parameters(c_root):,}")
    _train_controller(c_root, train_leaves, train_roots, batch_size=batch_size, budget=budget,
                      block_size=s, n_steps=controller_steps, lr=3e-4, device=device)
    c_root.eval()
    for p in c_root.parameters():
        p.requires_grad_(False)

    # --- C_pol: model-free query policy (belief shaped by imitating reveals) ----
    print("\n=== C_pol: model-free query policy ===")
    c_pol = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    policy_head = nn.Linear(state_dim, n_actions).to(device)
    print(f"C_pol parameters: {_count_parameters(c_pol) + _count_parameters(policy_head):,}")
    _train_policy_controller(c_pol, policy_head, c_root, train_leaves, train_roots,
                             batch_size=batch_size, budget=budget, block_size=s,
                             n_steps=policy_steps, lr=3e-4, device=device)
    c_pol.eval()
    for p in c_pol.parameters():
        p.requires_grad_(False)
    for p in policy_head.parameters():
        p.requires_grad_(False)

    eval_batch = min(batch_size, n_eval_episodes)
    planner_batch = min(2_048, n_eval_episodes)
    planning_leaves = eval_leaves[:planner_batch]
    planning_roots = eval_roots[:planner_batch]

    def _fresh_belief_head():
        return EntropyForwardModel(state_dim, hidden_dim=4 * state_dim,
                                   n_actions=n_actions, action_dim=16).to(device)

    # --- belief-VoI head on C_root (reproduces Step 0) -------------------------
    print("\n=== belief-VoI head on C_root (target = C_root) ===")
    head_root = _fresh_belief_head()
    _train_entropy_fm_split(head_root, c_root, c_root, train_leaves, batch_size=batch_size,
                            budget=budget, block_size=s, n_steps=forward_steps, lr=1e-3, device=device)
    diag_root = _voi_ranking_diagnostic_split(head_root, c_root, c_root, eval_leaves,
                                              budget=budget, block_size=s, batch_size=eval_batch, device=device)

    # --- belief-VoI head on C_pol (NEW -- policy-shaped belief) -----------------
    print("\n=== belief-VoI head on C_pol (belief = C_pol, target = C_root) ===")
    head_pol = _fresh_belief_head()
    _train_entropy_fm_split(head_pol, c_pol, c_root, train_leaves, batch_size=batch_size,
                            budget=budget, block_size=s, n_steps=forward_steps, lr=1e-3, device=device)
    diag_pol = _voi_ranking_diagnostic_split(head_pol, c_pol, c_root, eval_leaves,
                                             budget=budget, block_size=s, batch_size=eval_batch, device=device)

    # --- raw-obs ceiling (reproduces Step 0, target = C_root) ------------------
    print("\n=== raw-observation VoI head (ceiling, target = C_root) ===")
    obs_head = ObservationEntropyModel(v, sequence_length, hidden_dim=obs_hidden,
                                       n_actions=n_actions, action_dim=16,
                                       n_head=obs_heads, n_layer=obs_layers).to(device)
    _train_obs_entropy_fm(obs_head, c_root, train_leaves, batch_size=batch_size,
                          budget=budget, block_size=s, n_steps=forward_steps, lr=3e-4, device=device)
    diag_obs = _voi_ranking_diagnostic_obs(obs_head, c_root, eval_leaves,
                                           budget=budget, block_size=s, batch_size=eval_batch, device=device)

    # --- planners (all final-predict via C_root) -------------------------------
    plans = {}
    plans["random"] = _plan_queries(c_root, c_root, planning_leaves, planning_roots,
                                    budget=budget, block_size=s, planner="random", device=device)
    plans["voi_croot"] = _plan_queries_voi_split(c_root, c_root, head_root, planning_leaves, planning_roots,
                                                 budget=budget, block_size=s, device=device)
    plans["voi_cpol"] = _plan_queries_voi_split(c_root, c_pol, head_pol, planning_leaves, planning_roots,
                                                budget=budget, block_size=s, device=device)
    plans["policy_direct"] = _plan_queries_policy(c_root, c_pol, policy_head, planning_leaves, planning_roots,
                                                  budget=budget, block_size=s, device=device)
    plans["oracle"] = _plan_queries(c_root, c_root, planning_leaves, planning_roots,
                                    budget=budget, block_size=s, planner="oracle", device=device)

    rnd, orc = plans["random"]["root_accuracy"], plans["oracle"]["root_accuracy"]
    gap = max(orc - rnd, 1e-8)
    corr_root = diag_root["per_instance_query_corr"]
    corr_pol = diag_pol["per_instance_query_corr"]
    corr_obs = diag_obs["per_instance_query_corr"]
    summary = {
        "corr_croot_belief": corr_root,
        "corr_cpol_belief": corr_pol,
        "corr_raw_obs_ceiling": corr_obs,
        "headroom_gap_croot": corr_obs - corr_root,     # Step 0: ~0 (saturated)
        "headroom_gap_cpol": corr_obs - corr_pol,       # NEW: >0 => policy loses VoI
        "policy_vs_root_belief": corr_pol - corr_root,  # <0 => policy-shaping carries less VoI
        "gap_closed_voi_croot": (plans["voi_croot"]["root_accuracy"] - rnd) / gap,
        "gap_closed_voi_cpol": (plans["voi_cpol"]["root_accuracy"] - rnd) / gap,
        "gap_closed_policy_direct": (plans["policy_direct"]["root_accuracy"] - rnd) / gap,
    }

    metrics = {
        "config": {
            "v": v, "s": s, "depth": depth, "m": m, "rule_seed": rule_seed,
            "train_seed": train_seed, "sequence_length": sequence_length,
            "n_actions": n_actions, "budget": budget, "state_dim": state_dim,
            "controller_steps": controller_steps, "policy_steps": policy_steps,
            "forward_steps": forward_steps, "obs_hidden": obs_hidden,
            "obs_layers": obs_layers, "obs_heads": obs_heads,
        },
        "diag_croot_belief": diag_root,
        "diag_cpol_belief": diag_pol,
        "diag_raw_obs_ceiling": diag_obs,
        "planning": plans,
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    }

    print(f"\n=== SUMMARY (m={m}) ===")
    print("per-instance query-rank corr (does the belief carry VoI?):")
    print(f"  C_root belief (root-predictor) : {corr_root:.3f}")
    print(f"  C_pol  belief (model-free pol) : {corr_pol:.3f}")
    print(f"  raw-obs ceiling                : {corr_obs:.3f}")
    print(f"  headroom gap  (ceiling - C_root) = {summary['headroom_gap_croot']:+.3f}  (Step 0: ~0)")
    print(f"  headroom gap  (ceiling - C_pol ) = {summary['headroom_gap_cpol']:+.3f}  <-- >0 => internalization target")
    print(f"  policy - root belief             = {summary['policy_vs_root_belief']:+.3f}  <-- <0 => policy loses VoI")
    print("planner root accuracy (final prediction always via C_root):")
    for name in ("random", "voi_croot", "voi_cpol", "policy_direct", "oracle"):
        print(f"  {name:14s} acc={plans[name]['root_accuracy']:.3f} ce={plans[name]['root_ce']:.3f}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_active_headroom/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2):
    headroom_probe.remote(m=m)
