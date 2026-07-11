"""Active-RHM arity experiment: query-conditioned belief transitions.

The standard RHM setup is autonomous: a model passively receives a sequence.
This experiment turns inference into a controlled process. An agent observes a
masked RHM leaf sequence, chooses which leaf block to reveal, and predicts the
hidden root. A frozen belief controller supplies a state b; matched forward
models predict the controller update either from b alone or from (b, query).

The primary question is the reaching-style arity test: can an external planner
using F(b, u) choose more informative queries than any F(b), even when F(b) has
substantially more capacity?

Run from experiments/:
  modal run --detach rhm/rhm_active_query.py::active_query
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume


app = modal.App("rhm-active-query", image=image)


def _count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _generate_episode_batch(rules, n_episodes, seed):
    """Sample roots and their fully expanded leaf sequences from fixed rules."""
    rng = np.random.default_rng(seed)
    depth = len(rules)
    vocab_size, multiplicity, branching = rules[0].shape
    roots = rng.integers(0, vocab_size, size=n_episodes, dtype=np.int64)
    current = roots[:, None]

    for level in range(depth):
        choices = rng.integers(0, multiplicity, size=current.shape, dtype=np.int64)
        current = rules[level][current, choices].reshape(n_episodes, -1)

    return roots, current


def _reveal_blocks(leaves, actions, block_size, mask_token):
    """Return a masked observation after revealing the requested leaf blocks."""
    import torch

    batch_size, sequence_length = leaves.shape
    observation = torch.full(
        (batch_size, sequence_length), mask_token, dtype=torch.long, device=leaves.device
    )
    if actions.shape[1] == 0:
        return observation

    offsets = torch.arange(block_size, device=leaves.device)
    positions = actions.unsqueeze(-1) * block_size + offsets
    positions = positions.reshape(batch_size, -1)
    values = leaves.gather(1, positions)
    return observation.scatter(1, positions, values)


def _sample_transition(leaves, max_revealed, block_size):
    """Sample an observed state, a legal query, and the resulting observation."""
    import torch

    batch_size, sequence_length = leaves.shape
    n_actions = sequence_length // block_size
    n_revealed = int(torch.randint(0, max_revealed, ()).item())
    order = torch.rand(batch_size, n_actions, device=leaves.device).argsort(dim=1)
    previous_actions = order[:, :n_revealed]
    action = order[:, n_revealed]
    observation = _reveal_blocks(leaves, previous_actions, block_size, mask_token=-1)
    next_actions = torch.cat((previous_actions, action[:, None]), dim=1)
    next_observation = _reveal_blocks(leaves, next_actions, block_size, mask_token=-1)
    return observation, action, next_observation


def _build_model_classes():
    """Create Torch modules only in the remote environment that has Torch installed."""
    import torch
    import torch.nn as nn

    class BeliefController(nn.Module):
        """Frozen inference controller over a partially revealed RHM sequence."""

        def __init__(self, vocab_size, sequence_length, state_dim, n_head, n_layer):
            super().__init__()
            self.mask_token = vocab_size
            self.token_embedding = nn.Embedding(vocab_size + 1, state_dim)
            self.position_embedding = nn.Embedding(sequence_length, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim,
                nhead=n_head,
                dim_feedforward=4 * state_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.final_norm = nn.LayerNorm(state_dim)
            self.root_head = nn.Linear(state_dim, vocab_size)
            self.register_buffer("positions", torch.arange(sequence_length), persistent=False)

        def state(self, observation):
            tokens = observation.masked_fill(observation < 0, self.mask_token)
            hidden = self.token_embedding(tokens) + self.position_embedding(self.positions)
            hidden = self.encoder(hidden)
            return self.final_norm(hidden.mean(dim=1))

        def root_logits(self, state):
            return self.root_head(state)

        def forward(self, observation):
            state = self.state(observation)
            return self.root_logits(state), state

    class StateForwardModel(nn.Module):
        """Arity-1 forward model: predicts a belief update from belief alone."""

        def __init__(self, state_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )

        def forward(self, state, action=None):
            del action
            return self.net(state)

    class ActionForwardModel(nn.Module):
        """Arity-2 forward model: predicts a belief update from belief and query."""

        def __init__(self, state_dim, hidden_dim, n_actions, action_dim):
            super().__init__()
            self.action_embedding = nn.Embedding(n_actions, action_dim)
            self.net = nn.Sequential(
                nn.LayerNorm(state_dim + action_dim),
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )

        def forward(self, state, action):
            return self.net(torch.cat((state, self.action_embedding(action)), dim=-1))

    return BeliefController, StateForwardModel, ActionForwardModel


def _batch_from_pool(leaves, roots, batch_size, device):
    import torch

    indices = torch.randint(0, leaves.shape[0], (batch_size,))
    return leaves[indices].to(device), roots[indices].to(device)


def _train_controller(controller, train_leaves, train_roots, *, batch_size, budget,
                      block_size, n_steps, lr, device):
    import torch
    import torch.nn.functional as F

    controller.train()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, roots = _batch_from_pool(train_leaves, train_roots, batch_size, device)
        n_revealed = int(torch.randint(0, budget + 1, ()).item())
        n_actions = leaves.shape[1] // block_size
        actions = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        observation = _reveal_blocks(
            leaves, actions[:, :n_revealed], block_size, mask_token=-1
        )
        logits, _ = controller(observation)
        loss = F.cross_entropy(logits, roots)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()

        if step % report_every == 0 or step == n_steps:
            accuracy = (logits.argmax(dim=-1) == roots).float().mean().item()
            print(f"  controller step {step:5d}/{n_steps}: loss={loss.item():.4f} acc={accuracy:.3f}")


def _train_forward_model(forward_model, controller, train_leaves, train_roots, *,
                         batch_size, budget, block_size, n_steps, lr, device):
    import torch
    import torch.nn.functional as F

    del train_roots
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    forward_model.train()
    optimizer = torch.optim.AdamW(forward_model.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, _ = _batch_from_pool(train_leaves, train_leaves, batch_size, device)
        observation, action, next_observation = _sample_transition(
            leaves, budget, block_size
        )
        with torch.no_grad():
            state = controller.state(observation)
            next_state = controller.state(next_observation)
            target_delta = next_state - state
        prediction = forward_model(state, action)
        loss = F.mse_loss(prediction, target_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forward_model.parameters(), 1.0)
        optimizer.step()

        if step % report_every == 0 or step == n_steps:
            cosine = F.cosine_similarity(prediction, target_delta, dim=-1).mean().item()
            print(f"  FM step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cosine:.3f}")


def _transition_metrics(forward_model, controller, leaves, *, budget, block_size,
                        batch_size, device):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        forward_model.eval()
        eval_leaves = leaves[:batch_size].to(device)
        observation, action, next_observation = _sample_transition(eval_leaves, budget, block_size)
        state = controller.state(observation)
        target_delta = controller.state(next_observation) - state
        prediction = forward_model(state, action)
        transition_cosine = F.cosine_similarity(prediction, target_delta, dim=-1).mean().item()
        transition_mse = F.mse_loss(prediction, target_delta).item()

        n_actions = eval_leaves.shape[1] // block_size
        base_order = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        base_actions = base_order[:, :1]
        base_observation = _reveal_blocks(eval_leaves, base_actions, block_size, mask_token=-1)
        base_state = controller.state(base_observation)
        candidates = base_order[:, 1:]
        candidate_count = candidates.shape[1]
        states = base_state[:, None, :].expand(-1, candidate_count, -1).reshape(-1, base_state.shape[-1])
        candidate_actions = candidates.reshape(-1)
        repeated_base = base_actions[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 1)
        all_actions = torch.cat((repeated_base, candidate_actions[:, None]), dim=1)
        next_observations = _reveal_blocks(
            eval_leaves[:, None, :].expand(-1, candidate_count, -1).reshape(-1, eval_leaves.shape[1]),
            all_actions,
            block_size,
            mask_token=-1,
        )
        target_updates = controller.state(next_observations) - states
        predicted_updates = forward_model(states, candidate_actions)
        target_updates = target_updates.view(batch_size, candidate_count, -1)
        predicted_updates = predicted_updates.view(batch_size, candidate_count, -1)
        centered_target = target_updates - target_updates.mean(dim=1, keepdim=True)
        centered_prediction = predicted_updates - predicted_updates.mean(dim=1, keepdim=True)
        target_norm = target_updates.mean(dim=1).norm(dim=-1).mean().clamp_min(1e-8)
        command_spread = (centered_target.norm(dim=-1).mean() / target_norm).item()
        conditional_norm = centered_prediction.norm(dim=-1)
        if conditional_norm.mean().item() < 1e-8:
            conditional_cosine = 0.0
        else:
            conditional_cosine = F.cosine_similarity(
                centered_prediction.reshape(batch_size, -1),
                centered_target.reshape(batch_size, -1),
                dim=-1,
            ).mean().item()

    return {
        "transition_cosine": transition_cosine,
        "transition_mse": transition_mse,
        "command_relative_spread": command_spread,
        "command_conditional_cosine": conditional_cosine,
    }


def _plan_queries(controller, forward_model, leaves, roots, *, budget, block_size,
                  planner, device, action_permutation=None):
    """Run a frozen external planner against the active RHM environment."""
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
            legal = ~queried

            if planner == "random" or planner == "state":
                scores = torch.rand(batch_size, n_actions, device=device)
            elif planner == "oracle":
                candidate_states = []
                for action_index in range(n_actions):
                    actions = torch.full((batch_size, 1), action_index, device=device)
                    candidate_observation = observation.clone()
                    positions = actions * block_size + torch.arange(block_size, device=device)
                    candidate_observation.scatter_(1, positions, leaves.gather(1, positions))
                    candidate_states.append(controller.state(candidate_observation))
                candidate_states = torch.stack(candidate_states, dim=1)
                logits = controller.root_logits(candidate_states)
                scores = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    roots[:, None].expand(-1, n_actions).reshape(-1),
                    reduction="none",
                ).view(batch_size, n_actions)
            else:
                scored_actions = candidates
                if action_permutation is not None:
                    scored_actions = action_permutation[scored_actions]
                repeated_state = state[:, None, :].expand(-1, n_actions, -1).reshape(-1, state.shape[-1])
                predicted_delta = forward_model(repeated_state, scored_actions.reshape(-1))
                predicted_state = repeated_state + predicted_delta
                logits = controller.root_logits(predicted_state).view(batch_size, n_actions, -1)
                probabilities = logits.softmax(dim=-1)
                scores = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)

            chosen = scores.masked_fill(~legal, float("inf")).argmin(dim=1)
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
def active_query(
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
    forward_steps: int = 8_000,
    batch_size: int = 256,
    budget: int = 4,
):
    """Train frozen-controller FMs and test arity with external query planning."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    if sequence_length % s != 0:
        raise ValueError("Leaf query blocks must partition the leaf sequence.")
    n_actions = sequence_length // s
    if budget >= n_actions:
        raise ValueError("budget must leave at least one unrevealed query action.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(
        f"Active RHM query experiment: v={v}, s={s}, L={depth}, m={m}, "
        f"T={sequence_length}, actions={n_actions}, budget={budget}, device={device}"
    )
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    train_roots_np, train_leaves_np = _generate_episode_batch(rules, n_train_episodes, train_seed)
    eval_roots_np, eval_leaves_np = _generate_episode_batch(rules, n_eval_episodes, train_seed + 1)
    train_roots = torch.from_numpy(train_roots_np)
    train_leaves = torch.from_numpy(train_leaves_np)
    eval_roots = torch.from_numpy(eval_roots_np)
    eval_leaves = torch.from_numpy(eval_leaves_np)

    BeliefController, StateForwardModel, ActionForwardModel = _build_model_classes()
    controller = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    print(f"Controller parameters: {_count_parameters(controller):,}")
    _train_controller(
        controller,
        train_leaves,
        train_roots,
        batch_size=batch_size,
        budget=budget,
        block_size=s,
        n_steps=controller_steps,
        lr=3e-4,
        device=device,
    )

    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    effective_fm = ActionForwardModel(
        state_dim, hidden_dim=state_dim, n_actions=n_actions, action_dim=16
    ).to(device)
    state_forward_models = {
        str(hidden_dim): StateForwardModel(state_dim, hidden_dim).to(device)
        for hidden_dim in (state_dim, 2 * state_dim, 4 * state_dim)
    }
    print(f"Arity-2 FM parameters: {_count_parameters(effective_fm):,}")
    for name, model in state_forward_models.items():
        print(f"Arity-1 FM hidden={name} parameters: {_count_parameters(model):,}")

    print("Training arity-2 forward model")
    _train_forward_model(
        effective_fm,
        controller,
        train_leaves,
        train_roots,
        batch_size=batch_size,
        budget=budget,
        block_size=s,
        n_steps=forward_steps,
        lr=1e-3,
        device=device,
    )
    for name, model in state_forward_models.items():
        print(f"Training arity-1 forward model hidden={name}")
        _train_forward_model(
            model,
            controller,
            train_leaves,
            train_roots,
            batch_size=batch_size,
            budget=budget,
            block_size=s,
            n_steps=forward_steps,
            lr=1e-3,
            device=device,
        )

    metrics = {
        "config": {
            "v": v,
            "s": s,
            "depth": depth,
            "m": m,
            "rule_seed": rule_seed,
            "train_seed": train_seed,
            "sequence_length": sequence_length,
            "n_actions": n_actions,
            "budget": budget,
            "state_dim": state_dim,
            "controller_steps": controller_steps,
            "forward_steps": forward_steps,
        },
        "parameter_counts": {
            "controller": _count_parameters(controller),
            "arity_2": _count_parameters(effective_fm),
            **{f"arity_1_hidden_{name}": _count_parameters(model) for name, model in state_forward_models.items()},
        },
        "transition": {
            "arity_2": _transition_metrics(
                effective_fm, controller, eval_leaves, budget=budget, block_size=s,
                batch_size=min(batch_size, n_eval_episodes), device=device,
            ),
        },
        "planning": {},
    }
    for name, model in state_forward_models.items():
        metrics["transition"][f"arity_1_hidden_{name}"] = _transition_metrics(
            model, controller, eval_leaves, budget=budget, block_size=s,
            batch_size=min(batch_size, n_eval_episodes), device=device,
        )

    planner_batch = min(2_048, n_eval_episodes)
    planning_leaves = eval_leaves[:planner_batch]
    planning_roots = eval_roots[:planner_batch]
    metrics["planning"]["random"] = _plan_queries(
        controller, effective_fm, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="random", device=device,
    )
    metrics["planning"]["arity_1_largest"] = _plan_queries(
        controller, state_forward_models[str(4 * state_dim)], planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="state", device=device,
    )
    metrics["planning"]["arity_2"] = _plan_queries(
        controller, effective_fm, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="action", device=device,
    )
    permutation = torch.roll(torch.arange(n_actions, device=device), shifts=1)
    metrics["planning"]["arity_2_shuffled_action"] = _plan_queries(
        controller, effective_fm, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="action", device=device,
        action_permutation=permutation,
    )
    metrics["planning"]["oracle"] = _plan_queries(
        controller, effective_fm, planning_leaves, planning_roots,
        budget=budget, block_size=s, planner="oracle", device=device,
    )
    metrics["elapsed_seconds"] = time.time() - started

    print("\n=== Transition metrics ===")
    for name, result in metrics["transition"].items():
        print(
            f"{name:24s} cos={result['transition_cosine']:.3f} "
            f"cmd_cos={result['command_conditional_cosine']:.3f} "
            f"cmd_spread={result['command_relative_spread']:.3f}"
        )
    print("\n=== Query planning ===")
    for name, result in metrics["planning"].items():
        print(
            f"{name:24s} root_acc={result['root_accuracy']:.3f} "
            f"root_ce={result['root_ce']:.3f}"
        )

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_active_query/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    torch.save(
        {
            "controller": controller.state_dict(),
            "arity_2": effective_fm.state_dict(),
            "arity_1": {name: model.state_dict() for name, model in state_forward_models.items()},
            "config": metrics["config"],
        },
        f"{output_dir}/models.pt",
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main():
    active_query.remote()
