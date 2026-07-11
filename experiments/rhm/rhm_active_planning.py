"""Active-RHM follow-up: is the planning null a forward-model-fidelity problem?

The parent experiment (`rhm_active_query.py`) found that on RHM the arity-2 FM
captures query-conditional structure (cmd_cos ~0.36) but the entropy planner that
reads it sits at the random floor, while an oracle that scores candidates by true
post-reveal root CE reaches ~0.92. This script tests the three hypotheses raised
in interpreting that null, on a SHARED frozen controller (trained once per m):

  1. FIDELITY SWEEP. Train arity-2 belief-delta FMs across a (steps x hidden)
     grid, and for each record transition cos AND planner accuracy. If planner
     acc climbs toward the oracle as FM cos climbs, the null is "the FM wasn't
     good enough", not "arity can't be used on RHM".

  2. POSTERIOR TARGET. An arity-2 FM that predicts the change in ROOT LOGITS per
     query (not the belief delta), planned by predicted posterior entropy. This
     routes the query signal straight to the decision variable instead of through
     the controller readout on a noisy predicted belief.

  3. m SWEEP. Run the whole thing at m=2 (parent setting) and m=4 (the
     compositional regime CLAUDE.md flags as interesting) to confirm the null and
     the planning prize are not m=2 artifacts.

Run from experiments/ (two parallel detached runs, one per m):
  modal run --detach rhm/rhm_active_planning.py::active_planning --m 2
  modal run --detach rhm/rhm_active_planning.py::active_planning --m 4
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


app = modal.App("rhm-active-planning", image=image)


def _build_posterior_model():
    """Arity-2 FM that predicts the change in root logits from (belief, query)."""
    import torch
    import torch.nn as nn

    class PosteriorForwardModel(nn.Module):
        def __init__(self, state_dim, hidden_dim, n_actions, action_dim, vocab_size):
            super().__init__()
            self.action_embedding = nn.Embedding(n_actions, action_dim)
            self.net = nn.Sequential(
                nn.LayerNorm(state_dim + action_dim),
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, vocab_size),
            )

        def forward(self, state, action):
            return self.net(torch.cat((state, self.action_embedding(action)), dim=-1))

    return PosteriorForwardModel


def _train_posterior_fm(forward_model, controller, train_leaves, *, batch_size,
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
            base_logits = controller.root_logits(state)
            next_logits = controller.root_logits(controller.state(next_observation))
            target_delta = next_logits - base_logits
        prediction = forward_model(state, action)
        loss = F.mse_loss(prediction, target_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forward_model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            cosine = F.cosine_similarity(prediction, target_delta, dim=-1).mean().item()
            print(f"  posterior FM step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cosine:.3f}")


def _posterior_transition_metrics(forward_model, controller, leaves, *, budget,
                                  block_size, batch_size, device):
    """Cos + command-conditional cos of the predicted root-logit delta."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        forward_model.eval()
        eval_leaves = leaves[:batch_size].to(device)
        n_actions = eval_leaves.shape[1] // block_size
        base_order = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        base_actions = base_order[:, :1]
        base_observation = _reveal_blocks(eval_leaves, base_actions, block_size, mask_token=-1)
        base_state = controller.state(base_observation)
        base_logits = controller.root_logits(base_state)

        candidates = base_order[:, 1:]
        candidate_count = candidates.shape[1]
        states = base_state[:, None, :].expand(-1, candidate_count, -1).reshape(-1, base_state.shape[-1])
        candidate_actions = candidates.reshape(-1)
        repeated_base = base_actions[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 1)
        all_actions = torch.cat((repeated_base, candidate_actions[:, None]), dim=1)
        next_observations = _reveal_blocks(
            eval_leaves[:, None, :].expand(-1, candidate_count, -1).reshape(-1, eval_leaves.shape[1]),
            all_actions, block_size, mask_token=-1,
        )
        repeated_base_logits = base_logits[:, None, :].expand(-1, candidate_count, -1).reshape(
            -1, base_logits.shape[-1])
        target_updates = controller.root_logits(controller.state(next_observations)) - repeated_base_logits
        predicted_updates = forward_model(states, candidate_actions)
        transition_cosine = F.cosine_similarity(predicted_updates, target_updates, dim=-1).mean().item()

        target_updates = target_updates.view(batch_size, candidate_count, -1)
        predicted_updates = predicted_updates.view(batch_size, candidate_count, -1)
        centered_target = target_updates - target_updates.mean(dim=1, keepdim=True)
        centered_prediction = predicted_updates - predicted_updates.mean(dim=1, keepdim=True)
        target_norm = target_updates.mean(dim=1).norm(dim=-1).mean().clamp_min(1e-8)
        command_spread = (centered_target.norm(dim=-1).mean() / target_norm).item()
        if centered_prediction.norm(dim=-1).mean().item() < 1e-8:
            conditional_cosine = 0.0
        else:
            conditional_cosine = F.cosine_similarity(
                centered_prediction.reshape(batch_size, -1),
                centered_target.reshape(batch_size, -1), dim=-1,
            ).mean().item()

    return {
        "transition_cosine": transition_cosine,
        "command_conditional_cosine": conditional_cosine,
        "command_relative_spread": command_spread,
    }


def _plan_queries_posterior(controller, forward_model, leaves, roots, *, budget,
                            block_size, device):
    """External planner scoring candidates by predicted post-query root entropy."""
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
            base_logits = controller.root_logits(state)
            candidates = torch.arange(n_actions, device=device)[None, :].expand(batch_size, -1)
            repeated_state = state[:, None, :].expand(-1, n_actions, -1).reshape(-1, state.shape[-1])
            delta_logits = forward_model(repeated_state, candidates.reshape(-1)).view(batch_size, n_actions, -1)
            predicted_logits = base_logits[:, None, :] + delta_logits
            probabilities = predicted_logits.softmax(dim=-1)
            scores = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
            chosen = scores.masked_fill(queried, float("inf")).argmin(dim=1)
            positions = chosen[:, None] * block_size + torch.arange(block_size, device=device)
            observation.scatter_(1, positions, leaves.gather(1, positions))
            queried.scatter_(1, chosen[:, None], True)
            current_logits = controller.root_logits(controller.state(observation))
            current_probabilities = current_logits.softmax(dim=-1)
            entropy_trace.append(
                (-(current_probabilities * current_probabilities.clamp_min(1e-8).log()).sum(dim=-1)).mean().item()
            )

        final_logits = controller.root_logits(controller.state(observation))
        return {
            "root_accuracy": (final_logits.argmax(dim=-1) == roots).float().mean().item(),
            "root_ce": F.cross_entropy(final_logits, roots).item(),
            "entropy_by_query": entropy_trace,
        }


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def active_planning(
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
    batch_size: int = 256,
    budget: int = 4,
    quick: bool = False,
):
    """Shared-controller fidelity sweep + posterior FM, at one m."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_actions = sequence_length // s
    if budget >= n_actions:
        raise ValueError("budget must leave at least one unrevealed query action.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Active RHM planning: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
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
    PosteriorForwardModel = _build_posterior_model()

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

    metrics = {
        "config": {
            "v": v, "s": s, "depth": depth, "m": m, "rule_seed": rule_seed,
            "train_seed": train_seed, "sequence_length": sequence_length,
            "n_actions": n_actions, "budget": budget, "state_dim": state_dim,
            "controller_steps": controller_steps,
        },
        "references": {},
        "fidelity_sweep": [],
        "posterior": {},
    }

    # --- Reference planners (controller-only; FM-independent) -----------------
    # random/oracle branches never touch the FM; pass controller as a harmless
    # stand-in since _plan_queries calls forward_model.eval() unconditionally.
    metrics["references"]["random"] = _plan_queries(
        controller, controller, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="random", device=device,
    )
    metrics["references"]["oracle"] = _plan_queries(
        controller, controller, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="oracle", device=device,
    )
    print(f"\nReferences: random acc={metrics['references']['random']['root_accuracy']:.3f} "
          f"oracle acc={metrics['references']['oracle']['root_accuracy']:.3f}")

    # --- (1) Fidelity sweep: arity-2 belief-delta FM ---------------------------
    if quick:
        fidelity_grid = [(400, state_dim), (400, 4 * state_dim)]
    else:
        fidelity_grid = [
            (steps, hidden)
            for steps in (8_000, 24_000, 48_000)
            for hidden in (state_dim, 4 * state_dim)
        ]
    print("\n=== Fidelity sweep (arity-2 belief-delta FM) ===")
    for steps, hidden in fidelity_grid:
        fm = ActionForwardModel(state_dim, hidden_dim=hidden, n_actions=n_actions, action_dim=16).to(device)
        print(f"\n-- belief-delta FM steps={steps} hidden={hidden} params={_count_parameters(fm):,}")
        _train_forward_model(
            fm, controller, train_leaves, train_roots, batch_size=batch_size, budget=budget,
            block_size=s, n_steps=steps, lr=1e-3, device=device,
        )
        transition = _transition_metrics(
            fm, controller, eval_leaves, budget=budget, block_size=s,
            batch_size=eval_batch, device=device,
        )
        plan = _plan_queries(
            controller, fm, planning_leaves, planning_roots,
            budget=budget, block_size=s, planner="action", device=device,
        )
        entry = {
            "steps": steps, "hidden": hidden, "params": _count_parameters(fm),
            "transition_cosine": transition["transition_cosine"],
            "command_conditional_cosine": transition["command_conditional_cosine"],
            "command_relative_spread": transition["command_relative_spread"],
            "planner_accuracy": plan["root_accuracy"],
            "planner_ce": plan["root_ce"],
        }
        metrics["fidelity_sweep"].append(entry)
        print(f"   cos={entry['transition_cosine']:.3f} cmd_cos={entry['command_conditional_cosine']:.3f} "
              f"planner_acc={entry['planner_accuracy']:.3f}")

    # --- (2) Posterior-target arity-2 FM --------------------------------------
    print("\n=== Posterior-target FM (predicts root-logit delta) ===")
    posterior_results = {}
    posterior_grid = [(400, 4 * state_dim)] if quick else [(24_000, 4 * state_dim), (48_000, 4 * state_dim)]
    for steps, hidden in posterior_grid:
        fm = PosteriorForwardModel(
            state_dim, hidden_dim=hidden, n_actions=n_actions, action_dim=16, vocab_size=v
        ).to(device)
        print(f"\n-- posterior FM steps={steps} hidden={hidden} params={_count_parameters(fm):,}")
        _train_posterior_fm(
            fm, controller, train_leaves, batch_size=batch_size, budget=budget,
            block_size=s, n_steps=steps, lr=1e-3, device=device,
        )
        transition = _posterior_transition_metrics(
            fm, controller, eval_leaves, budget=budget, block_size=s,
            batch_size=eval_batch, device=device,
        )
        plan = _plan_queries_posterior(
            controller, fm, planning_leaves, planning_roots,
            budget=budget, block_size=s, device=device,
        )
        posterior_results[f"steps{steps}_hidden{hidden}"] = {
            "steps": steps, "hidden": hidden, "params": _count_parameters(fm),
            "transition_cosine": transition["transition_cosine"],
            "command_conditional_cosine": transition["command_conditional_cosine"],
            "command_relative_spread": transition["command_relative_spread"],
            "planner_accuracy": plan["root_accuracy"],
            "planner_ce": plan["root_ce"],
        }
        print(f"   logit-delta cos={transition['transition_cosine']:.3f} "
              f"cmd_cos={transition['command_conditional_cosine']:.3f} "
              f"planner_acc={plan['root_accuracy']:.3f}")
    metrics["posterior"] = posterior_results
    metrics["elapsed_seconds"] = time.time() - started

    print("\n=== SUMMARY (m={}) ===".format(m))
    print(f"random acc      = {metrics['references']['random']['root_accuracy']:.3f}")
    print(f"oracle acc      = {metrics['references']['oracle']['root_accuracy']:.3f}")
    print("belief-delta fidelity sweep (cos -> planner_acc):")
    for e in metrics["fidelity_sweep"]:
        print(f"  steps={e['steps']:6d} hidden={e['hidden']:4d}  cos={e['transition_cosine']:.3f} "
              f"cmd_cos={e['command_conditional_cosine']:.3f} planner_acc={e['planner_accuracy']:.3f}")
    print("posterior-target FM:")
    for k, e in posterior_results.items():
        print(f"  {k}: logit_cos={e['transition_cosine']:.3f} cmd_cos={e['command_conditional_cosine']:.3f} "
              f"planner_acc={e['planner_accuracy']:.3f}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_active_planning/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2):
    active_planning.remote(m=m)
