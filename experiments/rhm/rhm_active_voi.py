"""Active-RHM: a value-of-information forward model for epistemic queries.

The prior experiments (`rhm_active_query.py`, `rhm_active_planning.py`) established
a clean negative: a deterministic one-step FM predicting E[Delta belief | b, u]
CANNOT rank queries by informativeness, because value-of-information lives in the
VARIANCE of the belief update across the query's unknown contents, which a point
prediction averages away. A fidelity sweep confirmed it: more accurate Delta
prediction did not improve planning; at m=4 the planner sat exactly at random
while an oracle reached +0.66.

The Bayesian fix: don't predict the mean update, predict the DECISION VARIABLE.
An arity-2 head g(b, u) regresses the EXPECTED POSTERIOR ENTROPY after revealing
query u:
    target = H(root | b, o_u)     (realized posterior entropy on the actual leaves)
    MSE regression => g(b,u) -> E_{o_u}[ H(root | b, o_u) ]   (the BED quantity)
Since H(root|b) is constant across candidate queries, argmin g(b,u) = argmax EIG,
i.e. greedy Bayesian experimental design. Entropy is a NONLINEAR function of the
belief, so its conditional mean stays query-discriminative where the mean Delta
did not.

Head-to-head on ONE shared frozen controller, per m:
  random  |  belief-Delta greedy-entropy (the null)  |  VoI greedy-EIG (new)  |  oracle

Run from experiments/ (two parallel detached runs):
  modal run --detach rhm/rhm_active_voi.py::voi_planning --m 2
  modal run --detach rhm/rhm_active_voi.py::voi_planning --m 4
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
    _train_forward_model,
    _transition_metrics,
)


app = modal.App("rhm-active-voi", image=image)


def _build_entropy_model():
    """Arity-2 head that predicts the expected posterior entropy of a query."""
    import torch
    import torch.nn as nn

    class EntropyForwardModel(nn.Module):
        def __init__(self, state_dim, hidden_dim, n_actions, action_dim):
            super().__init__()
            self.action_embedding = nn.Embedding(n_actions, action_dim)
            self.net = nn.Sequential(
                nn.LayerNorm(state_dim + action_dim),
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, state, action):
            features = torch.cat((state, self.action_embedding(action)), dim=-1)
            return self.net(features).squeeze(-1)

    return EntropyForwardModel


def _posterior_entropy(logits):
    import torch.nn.functional as F

    probabilities = logits.softmax(dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)


def _train_entropy_fm(forward_model, controller, train_leaves, *, batch_size,
                      budget, block_size, n_steps, lr, device):
    import torch
    import torch.nn.functional as F

    controller.eval()
    forward_model.train()
    optimizer = torch.optim.AdamW(forward_model.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, _ = _batch_from_pool(train_leaves, train_leaves, batch_size, device)
        observation, action, next_observation = _sample_transition(leaves, budget, block_size)
        with torch.no_grad():
            state = controller.state(observation)
            next_logits = controller.root_logits(controller.state(next_observation))
            target_entropy = _posterior_entropy(next_logits)
        prediction = forward_model(state, action)
        loss = F.mse_loss(prediction, target_entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forward_model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            # correlation of predicted vs realized entropy on this batch
            centered_p = prediction - prediction.mean()
            centered_t = target_entropy - target_entropy.mean()
            denom = centered_p.norm() * centered_t.norm()
            corr = (centered_p @ centered_t / denom.clamp_min(1e-8)).item()
            print(f"  entropy FM step {step:5d}/{n_steps}: mse={loss.item():.4f} corr={corr:.3f}")


def _voi_ranking_diagnostic(forward_model, controller, leaves, *, budget, block_size,
                            batch_size, device):
    """Does the head rank candidate queries by true expected informativeness?

    From a base state with one block revealed, score every remaining candidate
    query by predicted vs realized posterior entropy, and report how well the head
    ranks them: Pearson corr over all (instance, query) pairs and top-1 agreement
    (does argmin predicted entropy match argmin realized entropy).
    """
    import torch

    with torch.no_grad():
        forward_model.eval()
        eval_leaves = leaves[:batch_size].to(device)
        n_actions = eval_leaves.shape[1] // block_size
        base_order = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        base_actions = base_order[:, :1]
        base_observation = _reveal_blocks(eval_leaves, base_actions, block_size, mask_token=-1)
        base_state = controller.state(base_observation)

        candidates = base_order[:, 1:]
        candidate_count = candidates.shape[1]
        states = base_state[:, None, :].expand(-1, candidate_count, -1).reshape(-1, base_state.shape[-1])
        candidate_flat = candidates.reshape(-1)
        repeated_base = base_actions[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 1)
        all_actions = torch.cat((repeated_base, candidate_flat[:, None]), dim=1)
        next_observations = _reveal_blocks(
            eval_leaves[:, None, :].expand(-1, candidate_count, -1).reshape(-1, eval_leaves.shape[1]),
            all_actions, block_size, mask_token=-1,
        )
        realized_entropy = _posterior_entropy(controller.root_logits(controller.state(next_observations)))
        predicted_entropy = forward_model(states, candidate_flat)

        centered_p = predicted_entropy - predicted_entropy.mean()
        centered_t = realized_entropy - realized_entropy.mean()
        pearson = (centered_p @ centered_t / (centered_p.norm() * centered_t.norm()).clamp_min(1e-8)).item()

        predicted_grid = predicted_entropy.view(batch_size, candidate_count)
        realized_grid = realized_entropy.view(batch_size, candidate_count)
        top1_agreement = (predicted_grid.argmin(dim=1) == realized_grid.argmin(dim=1)).float().mean().item()
        # per-instance rank corr of predicted vs realized query ranking (Spearman-ish via values)
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


def _plan_queries_voi(controller, forward_model, leaves, roots, *, budget, block_size, device):
    """Greedy expected-information-gain planner: pick min predicted posterior entropy."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        controller.eval()
        forward_model.eval()
        leaves = leaves.to(device)
        roots = roots.to(device)
        batch_size, sequence_length = leaves.shape
        n_actions = sequence_length // block_size
        observation = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=device)
        queried = torch.zeros(batch_size, n_actions, dtype=torch.bool, device=device)
        entropy_trace = []

        for _ in range(budget):
            state = controller.state(observation)
            candidates = torch.arange(n_actions, device=device)[None, :].expand(batch_size, -1)
            repeated_state = state[:, None, :].expand(-1, n_actions, -1).reshape(-1, state.shape[-1])
            predicted_entropy = forward_model(repeated_state, candidates.reshape(-1)).view(batch_size, n_actions)
            chosen = predicted_entropy.masked_fill(queried, float("inf")).argmin(dim=1)
            positions = chosen[:, None] * block_size + torch.arange(block_size, device=device)
            observation.scatter_(1, positions, leaves.gather(1, positions))
            queried.scatter_(1, chosen[:, None], True)
            current_logits = controller.root_logits(controller.state(observation))
            entropy_trace.append(_posterior_entropy(current_logits).mean().item())

        final_logits = controller.root_logits(controller.state(observation))
        return {
            "root_accuracy": (final_logits.argmax(dim=-1) == roots).float().mean().item(),
            "root_ce": F.cross_entropy(final_logits, roots).item(),
            "entropy_by_query": entropy_trace,
        }


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def voi_planning(
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
    forward_steps: int = 24_000,
    batch_size: int = 256,
    budget: int = 4,
    quick: bool = False,
):
    """Head-to-head: belief-Delta greedy-entropy vs VoI greedy-EIG, one controller."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_actions = sequence_length // s
    if budget >= n_actions:
        raise ValueError("budget must leave at least one unrevealed query action.")
    if quick:
        controller_steps, forward_steps = 400, 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"VoI planning: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
          f"actions={n_actions}, budget={budget}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    train_roots_np, train_leaves_np = _generate_episode_batch(rules, n_train_episodes, train_seed)
    eval_roots_np, eval_leaves_np = _generate_episode_batch(rules, n_eval_episodes, train_seed + 1)
    train_roots = torch.from_numpy(train_roots_np)
    train_leaves = torch.from_numpy(train_leaves_np)
    eval_roots = torch.from_numpy(eval_roots_np)
    eval_leaves = torch.from_numpy(eval_leaves_np)

    BeliefController, StateForwardModel, ActionForwardModel = _build_model_classes()
    EntropyForwardModel = _build_entropy_model()

    controller = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    print(f"Controller parameters: {_count_parameters(controller):,}")
    _train_controller(
        controller, train_leaves, train_roots, batch_size=batch_size, budget=budget,
        block_size=s, n_steps=controller_steps, lr=3e-4, device=device,
    )
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    planner_batch = min(2_048, n_eval_episodes)
    planning_leaves = eval_leaves[:planner_batch]
    planning_roots = eval_roots[:planner_batch]
    eval_batch = min(batch_size, n_eval_episodes)

    # --- belief-Delta baseline (the null we are trying to beat) ----------------
    print("\n=== belief-Delta FM (greedy entropy over predicted state) ===")
    belief_fm = ActionForwardModel(state_dim, hidden_dim=4 * state_dim, n_actions=n_actions, action_dim=16).to(device)
    _train_forward_model(
        belief_fm, controller, train_leaves, train_roots, batch_size=batch_size, budget=budget,
        block_size=s, n_steps=forward_steps, lr=1e-3, device=device,
    )
    belief_transition = _transition_metrics(
        belief_fm, controller, eval_leaves, budget=budget, block_size=s,
        batch_size=eval_batch, device=device,
    )

    # --- VoI entropy head ------------------------------------------------------
    print("\n=== VoI entropy FM (predicts expected posterior entropy) ===")
    entropy_fm = EntropyForwardModel(state_dim, hidden_dim=4 * state_dim, n_actions=n_actions, action_dim=16).to(device)
    print(f"VoI FM parameters: {_count_parameters(entropy_fm):,}")
    _train_entropy_fm(
        entropy_fm, controller, train_leaves, batch_size=batch_size, budget=budget,
        block_size=s, n_steps=forward_steps, lr=1e-3, device=device,
    )
    voi_diagnostic = _voi_ranking_diagnostic(
        entropy_fm, controller, eval_leaves, budget=budget, block_size=s,
        batch_size=eval_batch, device=device,
    )

    # --- Planners: random / belief-Delta / VoI / oracle, same controller -------
    plans = {}
    plans["random"] = _plan_queries(
        controller, controller, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="random", device=device,
    )
    plans["belief_delta"] = _plan_queries(
        controller, belief_fm, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="action", device=device,
    )
    plans["voi"] = _plan_queries_voi(
        controller, entropy_fm, planning_leaves, planning_roots,
        budget=budget, block_size=s, device=device,
    )
    plans["oracle"] = _plan_queries(
        controller, controller, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="oracle", device=device,
    )

    metrics = {
        "config": {
            "v": v, "s": s, "depth": depth, "m": m, "rule_seed": rule_seed,
            "train_seed": train_seed, "sequence_length": sequence_length,
            "n_actions": n_actions, "budget": budget, "state_dim": state_dim,
            "controller_steps": controller_steps, "forward_steps": forward_steps,
        },
        "belief_delta_transition": belief_transition,
        "voi_ranking_diagnostic": voi_diagnostic,
        "planning": plans,
        "elapsed_seconds": time.time() - started,
    }

    print(f"\n=== SUMMARY (m={m}) ===")
    print("VoI ranking diagnostic (does the head rank queries by informativeness?):")
    print(f"  pearson(pred,realized entropy) = {voi_diagnostic['pearson_all_pairs']:.3f}")
    print(f"  per-instance query rank corr   = {voi_diagnostic['per_instance_query_corr']:.3f}")
    print(f"  top-1 argmin agreement         = {voi_diagnostic['top1_argmin_agreement']:.3f} "
          f"(random {voi_diagnostic['random_top1_baseline']:.3f})")
    print("Planner root accuracy:")
    for name in ("random", "belief_delta", "voi", "oracle"):
        print(f"  {name:14s} acc={plans[name]['root_accuracy']:.3f} ce={plans[name]['root_ce']:.3f} "
              f"ent_trace={[round(x,3) for x in plans[name]['entropy_by_query']]}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_active_voi/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2):
    voi_planning.remote(m=m)
