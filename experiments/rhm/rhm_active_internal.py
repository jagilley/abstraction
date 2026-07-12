"""Active-RHM Step 0: decompose the VoI planning null with a ceiling probe.

Context. `rhm_active_voi.py` showed a value-of-information head g(b, u) predicting
expected posterior entropy plans well at m=2 (per-instance query-rank corr ~0.30,
closes ~38% of the random->oracle gap) but is a PRINCIPLED NULL at m=4 (corr ~0.07,
planner at random). The open question (ACTIVE_RHM_README next-step #2): is the m=4
null FUNDAMENTAL (value-of-information is genuinely carried by the hidden content,
so no function of the observation can rank queries) or CONTROLLER-LIMITED (the
frozen belief `b` is a weak sufficient statistic that discarded VoI-relevant
information which is still present in the raw observation)?

This decides whether the internalization experiment (co-training the controller to
carry more VoI -- the REACHING_INTERNAL analog) has any headroom to capture.

The probe. Train TWO VoI heads with the SAME target -- realized posterior entropy
H(root | controller.state(next_obs)) -- differing ONLY in their input:

  - belief-b head  : reads the controller's pooled belief `b`  (reproduces voi)
  - raw-obs head   : reads the RAW revealed observation tokens with its own,
                     higher-capacity transformer (bypasses the pooling bottleneck)

The gap in per-instance query-rank corr trichotomizes the null:
  belief_carried              = corr(belief head)
  obs_decodable_but_not_in_b  = corr(raw-obs head) - corr(belief head)   <-- what
                                internalization could attack
  fundamentally_content       = residual up to the content-peeking oracle planner

The behavioral version: plug each head into a greedy-EIG planner (same controller
does the root prediction; only the reveal-selection VoI estimator differs) and
compare fraction of the random->oracle gap closed.

Run from experiments/ (three parallel detached runs, one per m):
  modal run --detach rhm/rhm_active_internal.py::ceiling_probe --m 2
  modal run --detach rhm/rhm_active_internal.py::ceiling_probe --m 3
  modal run --detach rhm/rhm_active_internal.py::ceiling_probe --m 4
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
    _plan_queries_voi,
    _posterior_entropy,
    _train_entropy_fm,
    _voi_ranking_diagnostic,
)


app = modal.App("rhm-active-internal", image=image)


def _build_observation_entropy_model():
    """Ceiling VoI head: predicts expected posterior entropy from the RAW
    observation, bypassing the controller's pooled-belief bottleneck.

    Its own transformer over the observation tokens (higher capacity than the
    controller) so that a low per-instance corr here is evidence the signal is
    genuinely not decodable from the observation, not a capacity artifact.
    """
    import torch
    import torch.nn as nn

    class ObservationEntropyModel(nn.Module):
        def __init__(self, vocab_size, sequence_length, hidden_dim, n_actions,
                     action_dim, n_head, n_layer):
            super().__init__()
            self.mask_token = vocab_size
            self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
            self.position_embedding = nn.Embedding(sequence_length, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_head,
                dim_feedforward=4 * hidden_dim,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.final_norm = nn.LayerNorm(hidden_dim)
            self.action_embedding = nn.Embedding(n_actions, action_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim + action_dim),
                nn.Linear(hidden_dim + action_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.register_buffer("positions", torch.arange(sequence_length), persistent=False)

        def encode(self, observation):
            tokens = observation.masked_fill(observation < 0, self.mask_token)
            hidden = self.token_embedding(tokens) + self.position_embedding(self.positions)
            hidden = self.encoder(hidden)
            return self.final_norm(hidden.mean(dim=1))

        def forward(self, observation, action):
            features = torch.cat((self.encode(observation), self.action_embedding(action)), dim=-1)
            return self.head(features).squeeze(-1)

    return ObservationEntropyModel


def _train_obs_entropy_fm(model, controller, train_leaves, *, batch_size, budget,
                          block_size, n_steps, lr, device):
    """Same target/regression as `_train_entropy_fm`, but the head reads the raw
    observation rather than the controller's pooled belief."""
    import torch
    import torch.nn.functional as F

    controller.eval()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, _ = _batch_from_pool(train_leaves, train_leaves, batch_size, device)
        observation, action, next_observation = _sample_transition(leaves, budget, block_size)
        with torch.no_grad():
            target_entropy = _posterior_entropy(controller.root_logits(controller.state(next_observation)))
        prediction = model(observation, action)
        loss = F.mse_loss(prediction, target_entropy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            centered_p = prediction - prediction.mean()
            centered_t = target_entropy - target_entropy.mean()
            denom = (centered_p.norm() * centered_t.norm()).clamp_min(1e-8)
            corr = (centered_p @ centered_t / denom).item()
            print(f"  obs entropy FM step {step:5d}/{n_steps}: mse={loss.item():.4f} corr={corr:.3f}")


def _voi_ranking_diagnostic_obs(model, controller, leaves, *, budget, block_size,
                                batch_size, device):
    """Raw-obs analog of `_voi_ranking_diagnostic`: the head reads the base
    observation (expanded per candidate) instead of the base belief state."""
    import torch

    with torch.no_grad():
        model.eval()
        eval_leaves = leaves[:batch_size].to(device)
        n_actions = eval_leaves.shape[1] // block_size
        base_order = torch.rand(batch_size, n_actions, device=device).argsort(dim=1)
        base_actions = base_order[:, :1]
        base_observation = _reveal_blocks(eval_leaves, base_actions, block_size, mask_token=-1)

        candidates = base_order[:, 1:]
        candidate_count = candidates.shape[1]
        candidate_flat = candidates.reshape(-1)
        repeated_base = base_actions[:, None, :].expand(-1, candidate_count, -1).reshape(-1, 1)
        all_actions = torch.cat((repeated_base, candidate_flat[:, None]), dim=1)
        expanded_leaves = eval_leaves[:, None, :].expand(-1, candidate_count, -1).reshape(-1, eval_leaves.shape[1])
        next_observations = _reveal_blocks(expanded_leaves, all_actions, block_size, mask_token=-1)
        realized_entropy = _posterior_entropy(controller.root_logits(controller.state(next_observations)))

        base_observation_expanded = (
            base_observation[:, None, :].expand(-1, candidate_count, -1).reshape(-1, base_observation.shape[-1])
        )
        predicted_entropy = model(base_observation_expanded, candidate_flat)

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


def _plan_queries_voi_obs(controller, model, leaves, roots, *, budget, block_size, device):
    """Greedy-EIG planner whose VoI estimator reads the raw observation. The
    CONTROLLER still does the final root prediction -- only reveal selection
    differs from the belief-b planner."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        controller.eval()
        model.eval()
        leaves = leaves.to(device)
        roots = roots.to(device)
        batch_size, sequence_length = leaves.shape
        n_actions = sequence_length // block_size
        observation = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=device)
        queried = torch.zeros(batch_size, n_actions, dtype=torch.bool, device=device)
        entropy_trace = []

        for _ in range(budget):
            candidates = torch.arange(n_actions, device=device)[None, :].expand(batch_size, -1)
            repeated_obs = observation[:, None, :].expand(-1, n_actions, -1).reshape(-1, sequence_length)
            predicted_entropy = model(repeated_obs, candidates.reshape(-1)).view(batch_size, n_actions)
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
def ceiling_probe(
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
    obs_hidden: int = 128,
    obs_layers: int = 4,
    obs_heads: int = 4,
    batch_size: int = 256,
    budget: int = 4,
    quick: bool = False,
):
    """belief-b VoI head vs raw-observation VoI head (ceiling), one controller."""
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
    print(f"Ceiling probe: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
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

    controller = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    print(f"Controller parameters: {_count_parameters(controller):,}")
    _train_controller(
        controller, train_leaves, train_roots, batch_size=batch_size, budget=budget,
        block_size=s, n_steps=controller_steps, lr=3e-4, device=device,
    )
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    eval_batch = min(batch_size, n_eval_episodes)
    planner_batch = min(2_048, n_eval_episodes)
    planning_leaves = eval_leaves[:planner_batch]
    planning_roots = eval_roots[:planner_batch]

    # --- belief-b VoI head (reproduces the voi experiment) ---------------------
    print("\n=== belief-b VoI head (reads pooled belief) ===")
    belief_head = EntropyForwardModel(state_dim, hidden_dim=4 * state_dim,
                                      n_actions=n_actions, action_dim=16).to(device)
    print(f"belief-b head parameters: {_count_parameters(belief_head):,}")
    _train_entropy_fm(belief_head, controller, train_leaves, batch_size=batch_size,
                      budget=budget, block_size=s, n_steps=forward_steps, lr=1e-3, device=device)
    belief_diag = _voi_ranking_diagnostic(belief_head, controller, eval_leaves,
                                          budget=budget, block_size=s, batch_size=eval_batch, device=device)

    # --- raw-observation VoI head (the ceiling) --------------------------------
    print("\n=== raw-observation VoI head (reads raw tokens, own transformer) ===")
    obs_head = ObservationEntropyModel(v, sequence_length, hidden_dim=obs_hidden,
                                       n_actions=n_actions, action_dim=16,
                                       n_head=obs_heads, n_layer=obs_layers).to(device)
    print(f"raw-obs head parameters: {_count_parameters(obs_head):,}")
    _train_obs_entropy_fm(obs_head, controller, train_leaves, batch_size=batch_size,
                          budget=budget, block_size=s, n_steps=forward_steps, lr=3e-4, device=device)
    obs_diag = _voi_ranking_diagnostic_obs(obs_head, controller, eval_leaves,
                                           budget=budget, block_size=s, batch_size=eval_batch, device=device)

    # --- planners: random / belief-VoI / obs-VoI / oracle, same controller -----
    plans = {}
    plans["random"] = _plan_queries(controller, controller, planning_leaves, planning_roots,
                                    budget=budget, block_size=s, planner="random", device=device)
    plans["voi_belief"] = _plan_queries_voi(controller, belief_head, planning_leaves, planning_roots,
                                            budget=budget, block_size=s, device=device)
    plans["voi_obs"] = _plan_queries_voi_obs(controller, obs_head, planning_leaves, planning_roots,
                                             budget=budget, block_size=s, device=device)
    plans["oracle"] = _plan_queries(controller, controller, planning_leaves, planning_roots,
                                    budget=budget, block_size=s, planner="oracle", device=device)

    # --- trichotomy of the null ------------------------------------------------
    rnd, orc = plans["random"]["root_accuracy"], plans["oracle"]["root_accuracy"]
    gap = max(orc - rnd, 1e-8)
    corr_belief = belief_diag["per_instance_query_corr"]
    corr_obs = obs_diag["per_instance_query_corr"]
    trichotomy = {
        # per-instance query-rank corr decomposition
        "corr_belief_carried": corr_belief,
        "corr_obs_decodable_gap": corr_obs - corr_belief,      # internalization's target
        "corr_raw_obs_ceiling": corr_obs,
        # planner gap-closed decomposition
        "gap_closed_belief": (plans["voi_belief"]["root_accuracy"] - rnd) / gap,
        "gap_closed_obs": (plans["voi_obs"]["root_accuracy"] - rnd) / gap,
    }

    metrics = {
        "config": {
            "v": v, "s": s, "depth": depth, "m": m, "rule_seed": rule_seed,
            "train_seed": train_seed, "sequence_length": sequence_length,
            "n_actions": n_actions, "budget": budget, "state_dim": state_dim,
            "controller_steps": controller_steps, "forward_steps": forward_steps,
            "obs_hidden": obs_hidden, "obs_layers": obs_layers, "obs_heads": obs_heads,
        },
        "belief_head_diagnostic": belief_diag,
        "obs_head_diagnostic": obs_diag,
        "planning": plans,
        "trichotomy": trichotomy,
        "elapsed_seconds": time.time() - started,
    }

    print(f"\n=== SUMMARY (m={m}) ===")
    print("per-instance query-rank corr (the plannability diagnostic):")
    print(f"  belief-b head : {corr_belief:.3f}")
    print(f"  raw-obs head  : {corr_obs:.3f}   (ceiling)")
    print(f"  obs-decodable-but-not-in-b gap = {corr_obs - corr_belief:+.3f}  <-- internalization target")
    print("planner root accuracy (same controller does root prediction):")
    for name in ("random", "voi_belief", "voi_obs", "oracle"):
        print(f"  {name:12s} acc={plans[name]['root_accuracy']:.3f} ce={plans[name]['root_ce']:.3f}")
    print(f"fraction of random->oracle gap closed:  belief={trichotomy['gap_closed_belief']:+.3f}  "
          f"obs={trichotomy['gap_closed_obs']:+.3f}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_active_internal/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2):
    ceiling_probe.remote(m=m)
