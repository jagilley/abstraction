"""Active-RHM Stage A: editing/rewriting as a genuine CONTROL task.

Context and motivation
----------------------
The active-QUERY line (rhm_active_query / _voi / _internal / _headroom, see
ACTIVE_RHM_README.md) established that active-inference RHM is "inference in
disguise": a query does not change the world, it only sharpens a belief about a
fixed hidden root. Its payoff lives in the VARIANCE over hidden content, which a
mean-delta forward model averages away (the Phase-2 planning null), and where it
is belief-carried you must evaluate queries to act at all (act ~= plan -> no
internalization headroom, Phase 4).

This experiment adds the missing ingredient: DYNAMICS. The agent now *rewrites*
tokens toward a target root r*. Because the sequence is fully observed, an edit's
consequence is a DETERMINISTIC function of the observed state (you know exactly
what you are writing) -- there is no hidden-content variance to average. That is
precisely the reaching regime, where a mean-delta forward model is exact and
lookahead is what is hard. The hierarchy makes edits interact (editing a block's
level-1 feature cascades up the shared parse tree), so a forward model must
simulate the cascade -- a real dynamics model.

The action primitive is a BLOCK edit: set leaf block j (an s-tuple) to a
canonical valid tuple of level-1 feature g. This keeps the leaf->level-1 layer
on-grammar always (so the belief stays near-manifold) while upper levels can be
off-grammar (that is the planning challenge: reach a fully-valid r* config). The
grammar is COLLISION-FREE (generate_rules_invertible) so the true root of any
on-grammar sequence is exact ground truth -- the planner plans in the frozen
controller's belief space but is *also* graded by the true parse, closing the
"controller defines its own target" loophole.

Predictions (each the mirror image of an active-query result):
  1. Arity USABILITY ports this time: the arity-2 edit planner F(b, edit) beats
     random and its own shuffled-action control, and approaches the oracle -- at
     every m. (In active-query the arity-2 planner was ~a null.)
  2. No m=4 collapse on the control side. The SAME controller/grammar is run on
     the reveal (inference) task as an internal control; the reveal planner
     reproduces the epistemic behaviour while the edit planner does not.

Run from experiments/ (three parallel detached runs, one per m):
  modal run --detach rhm/rhm_edit_control.py::edit_control --m 2
  modal run --detach rhm/rhm_edit_control.py::edit_control --m 3
  modal run --detach rhm/rhm_edit_control.py::edit_control --m 4
Quick smoke test:
  modal run rhm/rhm_edit_control.py::edit_control --m 2 --quick
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
    _plan_queries,
    _reveal_blocks,
    _train_forward_model,
    _transition_metrics,
)


app = modal.App("rhm-edit-control", image=image)


def _parse_leaves_torch(leaves, inverse_maps, v, block_size):
    """On-GPU batched bottom-up parse (torch mirror of rhm_data.parse_leaves).

    inverse_maps: list of long tensors (one per level, length v**block_size) with
    code->feature or -1. Returns (roots, valid) on the same device.
    """
    import torch

    powers = v ** torch.arange(block_size, device=leaves.device)
    current = leaves
    valid = torch.ones(leaves.shape[0], dtype=torch.bool, device=leaves.device)
    for level in range(len(inverse_maps) - 1, -1, -1):
        batch, width = current.shape
        codes = (current.view(batch, width // block_size, block_size) * powers).sum(-1)
        feats = inverse_maps[level][codes]
        valid &= (feats >= 0).all(dim=1)
        current = feats.clamp_min(0)
    return current[:, 0], valid


def _train_edit_controller(controller, train_leaves, train_roots, *, batch_size,
                           n_blocks, block_size, n_steps, lr, device, p_full=0.5):
    """Controller trainer emphasizing FULL observation (the editing regime).

    With probability p_full a batch is fully revealed (target=root); otherwise the
    reveal count is uniform on [0, n_blocks], preserving the partial-observation
    competence the reveal contrast needs. This yields a DENSE root belief P(root)
    that rises smoothly as a config becomes more r*-consistent, giving greedy
    planning a usable gradient. (A strict off-grammar 'verifier' head was tried and
    rejected: root validity is a global property, so its signal is ~0 until the
    final fix — too sparse for a myopic planner. The honest check on whether this
    dense belief is *gamed* is the ground-truth feature-match metric + gt_oracle.)
    """
    import torch
    import torch.nn.functional as F

    controller.train()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        leaves, roots = _batch_from_pool(train_leaves, train_roots, batch_size, device)
        if torch.rand(()).item() < p_full:
            n_revealed = n_blocks
        else:
            n_revealed = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        observation = _reveal_blocks(leaves, order[:, :n_revealed], block_size, mask_token=-1)
        logits, _ = controller(observation)
        loss = F.cross_entropy(logits, roots)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            accuracy = (logits.argmax(dim=-1) == roots).float().mean().item()
            print(f"  controller step {step:5d}/{n_steps}: loss={loss.item():.4f} "
                  f"acc={accuracy:.3f} (n_rev={n_revealed})")


# --------------------------------------------------------------------------- #
# Edit dynamics primitives                                                     #
# --------------------------------------------------------------------------- #

def _apply_block_edit(leaves, blocks, feats, canon, block_size):
    """Set block `blocks` of each row to the canonical tuple of feature `feats`.

    leaves (B, T), blocks (B,), feats (B,), canon (v, block_size). Returns a new
    tensor (does not mutate `leaves`). Vectorized; safe on expanded/non-contiguous
    inputs because it clones first.
    """
    import torch

    new = leaves.clone()
    tuples = canon[feats]  # (B, block_size)
    positions = blocks[:, None] * block_size + torch.arange(block_size, device=leaves.device)
    return new.scatter_(1, positions, tuples)


def _random_edits(x, canon, n_blocks, v, block_size, n_edits, device):
    """Apply n_edits independent uniform-random block edits to every row."""
    import torch

    batch_size = x.shape[0]
    for _ in range(n_edits):
        blocks = torch.randint(0, n_blocks, (batch_size,), device=device)
        feats = torch.randint(0, v, (batch_size,), device=device)
        x = _apply_block_edit(x, blocks, feats, canon, block_size)
    return x


def _sample_edit_transition(leaves_pool, canon, *, batch_size, n_blocks, v,
                            block_size, max_edits, device):
    """Sample (state, edit-command, next-state) along a random-edit trajectory.

    A base valid sequence is perturbed by k in [0, max_edits] random block edits
    (covering the on- and off-grammar configs the planner actually visits), then
    one more random block edit is the transition to predict. The command index is
    a = block * v + feature.
    """
    import torch

    indices = torch.randint(0, leaves_pool.shape[0], (batch_size,))
    x = leaves_pool[indices].to(device).clone()
    k = int(torch.randint(0, max_edits + 1, ()).item())
    x = _random_edits(x, canon, n_blocks, v, block_size, k, device)

    blocks = torch.randint(0, n_blocks, (batch_size,), device=device)
    feats = torch.randint(0, v, (batch_size,), device=device)
    action = blocks * v + feats
    next_x = _apply_block_edit(x, blocks, feats, canon, block_size)
    return x, action, next_x


def _train_edit_fm(forward_model, controller, leaves_pool, canon, *, batch_size,
                   n_blocks, v, block_size, max_edits, n_steps, lr, device):
    """Train a forward model to predict the controller's belief delta under an
    edit. Arity-2 F(b, a) uses the command; arity-1 F(b) ignores it."""
    import torch
    import torch.nn.functional as F

    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    forward_model.train()
    optimizer = torch.optim.AdamW(forward_model.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)

    for step in range(1, n_steps + 1):
        x, action, next_x = _sample_edit_transition(
            leaves_pool, canon, batch_size=batch_size, n_blocks=n_blocks, v=v,
            block_size=block_size, max_edits=max_edits, device=device,
        )
        with torch.no_grad():
            state = controller.state(x)
            target_delta = controller.state(next_x) - state
        prediction = forward_model(state, action)
        loss = F.mse_loss(prediction, target_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forward_model.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            cosine = F.cosine_similarity(prediction, target_delta, dim=-1).mean().item()
            print(f"  edit FM step {step:5d}/{n_steps}: mse={loss.item():.5f} cos={cosine:.3f}")


def _edit_transition_metrics(forward_model, controller, leaves_pool, canon, *,
                             n_blocks, v, block_size, max_edits, batch_size, device):
    """Command-conditional structure of an edit forward model.

    command_conditional_cosine is the query-dependent part of the update: from a
    fixed base state, enumerate every candidate edit, center predictions and true
    updates across candidates, and measure their cosine. Arity-1 -> ~0 (can't see
    the command); arity-2 -> high. command_relative_spread measures how much the
    true update actually varies across candidates (the plannable signal).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        forward_model.eval()
        indices = torch.randint(0, leaves_pool.shape[0], (batch_size,))
        x = leaves_pool[indices].to(device).clone()
        x = _random_edits(x, canon, n_blocks, v, block_size, max_edits // 2, device)
        base_state = controller.state(x)

        blocks = torch.randint(0, n_blocks, (batch_size,), device=device)
        feats = torch.randint(0, v, (batch_size,), device=device)
        action = blocks * v + feats
        next_x = _apply_block_edit(x, blocks, feats, canon, block_size)
        true_delta = controller.state(next_x) - base_state
        pred_delta = forward_model(base_state, action)
        transition_cosine = F.cosine_similarity(pred_delta, true_delta, dim=-1).mean().item()
        transition_mse = F.mse_loss(pred_delta, true_delta).item()

        n_cand = n_blocks * v
        cand = torch.arange(n_cand, device=device)
        cand_blocks = cand // v
        cand_feats = cand % v
        expanded = x[:, None, :].expand(-1, n_cand, -1).reshape(batch_size * n_cand, -1)
        blk = cand_blocks[None, :].expand(batch_size, -1).reshape(-1)
        ftr = cand_feats[None, :].expand(batch_size, -1).reshape(-1)
        cand_x = _apply_block_edit(expanded, blk, ftr, canon, block_size)
        base_rep = base_state.repeat_interleave(n_cand, dim=0)
        true_upd = (controller.state(cand_x) - base_rep).view(batch_size, n_cand, -1)
        pred_upd = forward_model(
            base_state[:, None, :].expand(-1, n_cand, -1).reshape(batch_size * n_cand, -1),
            cand[None, :].expand(batch_size, -1).reshape(-1),
        ).view(batch_size, n_cand, -1)

        centered_true = true_upd - true_upd.mean(dim=1, keepdim=True)
        centered_pred = pred_upd - pred_upd.mean(dim=1, keepdim=True)
        mean_norm = true_upd.mean(dim=1).norm(dim=-1).mean().clamp_min(1e-8)
        command_spread = (centered_true.norm(dim=-1).mean() / mean_norm).item()
        if centered_pred.norm(dim=-1).mean().item() < 1e-8:
            command_cosine = 0.0
        else:
            command_cosine = F.cosine_similarity(
                centered_pred.reshape(batch_size, -1),
                centered_true.reshape(batch_size, -1),
                dim=-1,
            ).mean().item()

    return {
        "transition_cosine": transition_cosine,
        "transition_mse": transition_mse,
        "command_relative_spread": command_spread,
        "command_conditional_cosine": command_cosine,
    }


# --------------------------------------------------------------------------- #
# Planning episodes                                                            #
# --------------------------------------------------------------------------- #

def _sample_init_states(rules, canon_np, inverse_maps, *, n_episodes, n_blocks, v,
                        block_size, mode, n_corrupt, seed):
    """Build planning episodes: (initial leaves, target root r*, target features).

    mode="corrupt": start from a valid r* expansion, corrupt n_corrupt distinct
      blocks to random features -> repair back to r* (guaranteed reachable in
      n_corrupt edits). Target = the original generating root; target_features =
      the pristine level-1 feature vector (a witness r* expansion, used by the
      ground-truth oracle and the feature-match metric). mode="flip" targets a
      different root and has no single witness expansion (target_features=None).
    """
    import torch

    roots, leaves_np = _generate_episode_batch(rules, n_episodes, seed)
    targets = torch.from_numpy(roots).clone()
    canon = torch.from_numpy(canon_np)
    target_features = None

    if mode == "corrupt":
        powers = v ** np.arange(block_size)
        pairs = leaves_np.reshape(n_episodes, n_blocks, block_size)
        codes = (pairs * powers).sum(axis=2)
        target_features = torch.from_numpy(inverse_maps[len(rules) - 1][codes]).long()
        leaves = torch.from_numpy(leaves_np)
        order = torch.rand(n_episodes, n_blocks).argsort(dim=1)
        rng = np.random.default_rng(seed + 7)
        for i in range(n_corrupt):
            blocks = order[:, i]
            feats = torch.from_numpy(rng.integers(0, v, size=n_episodes))
            leaves = _apply_block_edit(leaves, blocks, feats, canon, block_size)
    elif mode == "flip":
        rng = np.random.default_rng(seed + 7)
        offset = torch.from_numpy(rng.integers(1, v, size=n_episodes))
        targets = (targets + offset) % v  # guaranteed != original root
        leaves = torch.from_numpy(leaves_np)
    else:
        raise ValueError(f"unknown init mode {mode!r}")

    return leaves, targets, target_features


def _plan_edits(controller, forward_model, leaves0, targets, canon, rules,
                inverse_maps, inverse_maps_torch, target_features, *, n_blocks, v,
                block_size, budget, planner, device, action_permutation=None):
    """Greedy external planner on the edit-control environment.

    Each step scores every candidate block edit (block, feature) and commits the
    argmax over not-yet-edited blocks. Scoring depends on the planner:
      - random: uniform.
      - action (arity-2) / state (arity-1): predicted controller log P(r*) at the
        forecast state b + F(b, edit); arity-1 is edit-blind (tie-broken randomly).
      - oracle: TRUE controller log P(r*) after actually applying each candidate
        (peeks at the belief, not the grammar).
      - gt_oracle: TRUE grammar progress — set a currently-wrong block to its
        witness target feature (peeks at ground truth). Ceiling on true success.
    Graded by the controller (target_success/target_prob) AND the true grammar
    parse (gt_success/gt_on_manifold/gt_feature_match).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        controller.eval()
        if forward_model is not None:
            forward_model.eval()
        x = leaves0.to(device).clone()
        targets = targets.to(device)
        canon = canon.to(device)
        if target_features is not None:
            target_features = target_features.to(device)
        batch_size = x.shape[0]
        n_cand = n_blocks * v
        cand = torch.arange(n_cand, device=device)
        cand_blocks = cand // v
        cand_feats = cand % v
        block_of_cand = cand_blocks[None, :].expand(batch_size, -1)
        powers = v ** torch.arange(block_size, device=device)
        edited = torch.zeros(batch_size, n_blocks, dtype=torch.bool, device=device)
        target_prob_trace = []
        feature_match_trace = []

        def _block_features(seq):
            # bottom-level inverse map: leaf s-tuple -> its level-1 feature
            codes = (seq.view(batch_size, n_blocks, block_size) * powers).sum(-1)
            return inverse_maps_torch[-1][codes]

        for _ in range(budget):
            state = controller.state(x)

            if planner == "random":
                scores = torch.rand(batch_size, n_cand, device=device)
            elif planner == "oracle":
                scores = torch.empty(batch_size, n_cand, device=device)
                for a in range(n_cand):
                    blk = torch.full((batch_size,), a // v, device=device, dtype=torch.long)
                    ftr = torch.full((batch_size,), a % v, device=device, dtype=torch.long)
                    cand_x = _apply_block_edit(x, blk, ftr, canon, block_size)
                    logits = controller.root_logits(controller.state(cand_x))
                    scores[:, a] = logits.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)
            elif planner == "gt_oracle":
                cur_feats = _block_features(x)
                block_target = target_features.gather(1, block_of_cand)  # (B, n_cand)
                would_match = (cand_feats[None, :].expand(batch_size, -1) == block_target).float()
                currently = (cur_feats.gather(1, block_of_cand) == block_target).float()
                scores = would_match - currently  # +1 only for fixing a wrong block
            else:  # arity-2 ("action") or arity-1 ("state") forward-model planner
                scored = cand[None, :].expand(batch_size, -1).reshape(-1)
                if action_permutation is not None:
                    scored = action_permutation[scored]
                state_rep = state[:, None, :].expand(-1, n_cand, -1).reshape(batch_size * n_cand, -1)
                predicted_state = state_rep + forward_model(state_rep, scored)
                logits = controller.root_logits(predicted_state).view(batch_size, n_cand, -1)
                scores = logits.log_softmax(dim=-1).gather(
                    2, targets[:, None, None].expand(-1, n_cand, 1)
                ).squeeze(-1)
                if planner == "state":
                    # arity-1 F(b) is edit-blind: identical score for every candidate.
                    # Break ties randomly so it reads as the uninformed baseline it is.
                    scores = scores + 1e-3 * torch.rand_like(scores)

            block_edited = edited.gather(1, block_of_cand)
            chosen = scores.masked_fill(block_edited, float("-inf")).argmax(dim=1)
            x = _apply_block_edit(x, chosen // v, chosen % v, canon, block_size)
            edited.scatter_(1, (chosen // v)[:, None], True)

            current = controller.root_logits(controller.state(x))
            target_prob_trace.append(
                current.log_softmax(dim=-1).gather(1, targets[:, None]).exp().mean().item()
            )
            if target_features is not None:
                feature_match_trace.append((_block_features(x) == target_features).float().mean().item())

        final_logits = controller.root_logits(controller.state(x))
        target_success = (final_logits.argmax(dim=-1) == targets).float().mean().item()
        target_prob = final_logits.log_softmax(dim=-1).gather(1, targets[:, None]).exp().mean().item()
        target_ce = F.cross_entropy(final_logits, targets).item()

        gt_roots, gt_valid = parse_leaves(x.cpu().numpy(), rules, inverse_maps)
        gt_roots = torch.from_numpy(gt_roots).to(device)
        gt_valid = torch.from_numpy(gt_valid).to(device)
        gt_success = ((gt_roots == targets) & gt_valid).float().mean().item()
        gt_feature_match = None
        if target_features is not None:
            gt_feature_match = (_block_features(x) == target_features).float().mean().item()

    return {
        "target_success": target_success,
        "target_prob": target_prob,
        "target_ce": target_ce,
        "gt_success": gt_success,
        "gt_on_manifold": gt_valid.float().mean().item(),
        "gt_feature_match": gt_feature_match,
        "target_prob_by_step": target_prob_trace,
        "gt_feature_match_by_step": feature_match_trace,
    }


# --------------------------------------------------------------------------- #
# Experiment                                                                   #
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def edit_control(
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
    edit_budget: int = 4,
    n_corrupt: int = 4,
    init_mode: str = "corrupt",
    reveal_budget: int = 4,
    quick: bool = False,
):
    """Editing-as-control arity battery, with a same-controller reveal contrast."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    n_edit_actions = n_blocks * v
    if edit_budget > n_blocks:
        raise ValueError("edit_budget cannot exceed the number of blocks.")
    if reveal_budget >= n_blocks:
        raise ValueError("reveal_budget must leave at least one unrevealed block.")
    if quick:
        controller_steps, forward_steps = 500, 500
        n_train_episodes, n_eval_episodes = 20_000, 1_024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(
        f"Edit-control RHM: v={v}, s={s}, L={depth}, m={m}, T={sequence_length}, "
        f"blocks={n_blocks}, edit_actions={n_edit_actions}, edit_budget={edit_budget}, "
        f"n_corrupt={n_corrupt}, mode={init_mode}, device={device}"
    )
    started = time.time()

    # Collision-free grammar so the true parse is exact ground truth.
    rules = generate_rules_invertible(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])  # (v, s) canonical level-1 tuples
    canon = torch.from_numpy(canon_np).to(device)

    train_roots_np, train_leaves_np = _generate_episode_batch(rules, n_train_episodes, train_seed)
    eval_roots_np, eval_leaves_np = _generate_episode_batch(rules, n_eval_episodes, train_seed + 1)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)
    eval_leaves = torch.from_numpy(eval_leaves_np)
    eval_roots = torch.from_numpy(eval_roots_np)

    BeliefController, StateForwardModel, ActionForwardModel = _build_model_classes()
    inverse_maps_torch = [torch.from_numpy(table).to(device) for table in inverse_maps]

    # Dense-belief controller (full-observation weighted). Predicts the root on
    # valid + partial sequences; its P(root) is a dense proxy for r*-consistency
    # that gives greedy planning a gradient. Whether that dense belief is *gamed*
    # (high P(r*) without true validity) is checked by the ground-truth metrics.
    controller = BeliefController(v, sequence_length, state_dim, n_head=4, n_layer=2).to(device)
    print(f"Controller parameters: {_count_parameters(controller):,}")
    _train_edit_controller(
        controller, train_leaves, train_roots, batch_size=batch_size, n_blocks=n_blocks,
        block_size=s, n_steps=controller_steps, lr=3e-4, device=device, p_full=0.5,
    )
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    # Sanity: controller accuracy on FULL valid sequences (the editing regime).
    with torch.no_grad():
        full_logits = controller.root_logits(controller.state(eval_leaves.to(device)))
        full_acc = (full_logits.argmax(dim=-1) == eval_roots.to(device)).float().mean().item()
    print(f"Controller full-sequence root accuracy: {full_acc:.3f}")

    eval_batch = min(batch_size, n_eval_episodes)
    planner_batch = min(2_048, n_eval_episodes)

    # --- EDIT (control) forward models ---------------------------------------
    edit_fm2 = ActionForwardModel(state_dim, hidden_dim=state_dim,
                                  n_actions=n_edit_actions, action_dim=16).to(device)
    edit_fm1 = StateForwardModel(state_dim, hidden_dim=4 * state_dim).to(device)
    print(f"Edit arity-2 FM parameters: {_count_parameters(edit_fm2):,}")
    print(f"Edit arity-1 FM parameters: {_count_parameters(edit_fm1):,}")
    print("Training arity-2 edit forward model")
    _train_edit_fm(edit_fm2, controller, train_leaves, canon, batch_size=batch_size,
                   n_blocks=n_blocks, v=v, block_size=s, max_edits=edit_budget,
                   n_steps=forward_steps, lr=1e-3, device=device)
    print("Training arity-1 edit forward model")
    _train_edit_fm(edit_fm1, controller, train_leaves, canon, batch_size=batch_size,
                   n_blocks=n_blocks, v=v, block_size=s, max_edits=edit_budget,
                   n_steps=forward_steps, lr=1e-3, device=device)

    edit_transition = {
        "arity_2": _edit_transition_metrics(edit_fm2, controller, eval_leaves, canon,
                                            n_blocks=n_blocks, v=v, block_size=s,
                                            max_edits=edit_budget, batch_size=eval_batch, device=device),
        "arity_1": _edit_transition_metrics(edit_fm1, controller, eval_leaves, canon,
                                            n_blocks=n_blocks, v=v, block_size=s,
                                            max_edits=edit_budget, batch_size=eval_batch, device=device),
    }

    # --- EDIT planning episodes ----------------------------------------------
    leaves0, targets, target_features = _sample_init_states(
        rules, canon_np, inverse_maps, n_episodes=planner_batch, n_blocks=n_blocks,
        v=v, block_size=s, mode=init_mode, n_corrupt=n_corrupt, seed=train_seed + 2,
    )

    def _edit_plan(fm, planner, permutation=None):
        return _plan_edits(
            controller, fm, leaves0, targets, canon, rules, inverse_maps,
            inverse_maps_torch, target_features, n_blocks=n_blocks, v=v, block_size=s,
            budget=edit_budget, planner=planner, device=device, action_permutation=permutation)

    edit_permutation = torch.roll(torch.arange(n_edit_actions, device=device), shifts=1)
    edit_plans = {
        "random": _edit_plan(None, "random"),
        "arity_1": _edit_plan(edit_fm1, "state"),
        "arity_2": _edit_plan(edit_fm2, "action"),
        "arity_2_shuffled": _edit_plan(edit_fm2, "action", edit_permutation),
        "oracle": _edit_plan(None, "oracle"),
    }
    if target_features is not None:
        edit_plans["gt_oracle"] = _edit_plan(None, "gt_oracle")

    # --- REVEAL (inference) contrast on the SAME controller ------------------
    # Reproduces the active-query mean-delta apparatus for a within-script,
    # controlled comparison: same grammar, same controller, mean-delta arity-2 FM.
    reveal_fm = ActionForwardModel(state_dim, hidden_dim=state_dim,
                                   n_actions=n_blocks, action_dim=16).to(device)
    print("Training arity-2 reveal (mean-delta) forward model")
    _train_forward_model(reveal_fm, controller, train_leaves, train_roots,
                         batch_size=batch_size, budget=reveal_budget, block_size=s,
                         n_steps=forward_steps, lr=1e-3, device=device)
    reveal_transition = _transition_metrics(reveal_fm, controller, eval_leaves,
                                            budget=reveal_budget, block_size=s,
                                            batch_size=eval_batch, device=device)
    reveal_leaves = eval_leaves[:planner_batch]
    reveal_roots = eval_roots[:planner_batch]
    reveal_plans = {
        "random": _plan_queries(controller, reveal_fm, reveal_leaves, reveal_roots,
                                budget=reveal_budget, block_size=s, planner="random", device=device),
        "arity_2": _plan_queries(controller, reveal_fm, reveal_leaves, reveal_roots,
                                 budget=reveal_budget, block_size=s, planner="action", device=device),
        "oracle": _plan_queries(controller, reveal_fm, reveal_leaves, reveal_roots,
                                budget=reveal_budget, block_size=s, planner="oracle", device=device),
    }

    # --- gap-closed summaries -------------------------------------------------
    def _fraction_closed(planner_value, random_value, oracle_value):
        gap = oracle_value - random_value
        if abs(gap) < 1e-8:
            return 0.0
        return (planner_value - random_value) / gap

    edit_gap = {
        name: _fraction_closed(edit_plans[name]["target_prob"],
                               edit_plans["random"]["target_prob"],
                               edit_plans["oracle"]["target_prob"])
        for name in ("arity_1", "arity_2", "arity_2_shuffled")
    }
    edit_gap_success = {
        name: _fraction_closed(edit_plans[name]["target_success"],
                               edit_plans["random"]["target_success"],
                               edit_plans["oracle"]["target_success"])
        for name in ("arity_1", "arity_2", "arity_2_shuffled")
    }
    # Ground-truth gap closed: fraction of the random -> gt_oracle TRUE-success gap
    # the belief-space planner reaches. This is the honest control metric (the
    # gt_oracle is the true achievable ceiling; the controller judge cannot game it).
    edit_gap_gt = {}
    if "gt_oracle" in edit_plans:
        edit_gap_gt = {
            name: _fraction_closed(edit_plans[name]["gt_success"],
                                   edit_plans["random"]["gt_success"],
                                   edit_plans["gt_oracle"]["gt_success"])
            for name in ("arity_1", "arity_2", "arity_2_shuffled")
        }
    reveal_gap = _fraction_closed(reveal_plans["arity_2"]["root_accuracy"],
                                  reveal_plans["random"]["root_accuracy"],
                                  reveal_plans["oracle"]["root_accuracy"])

    metrics = {
        "config": {
            "v": v, "s": s, "depth": depth, "m": m, "rule_seed": rule_seed,
            "train_seed": train_seed, "sequence_length": sequence_length,
            "n_blocks": n_blocks, "n_edit_actions": n_edit_actions,
            "edit_budget": edit_budget, "n_corrupt": n_corrupt, "init_mode": init_mode,
            "reveal_budget": reveal_budget, "state_dim": state_dim,
            "controller_steps": controller_steps, "forward_steps": forward_steps,
        },
        "controller_full_accuracy": full_acc,
        "edit_transition": edit_transition,
        "edit_planning": edit_plans,
        "edit_gap_closed_prob": edit_gap,
        "edit_gap_closed_success": edit_gap_success,
        "edit_gap_closed_gt_success": edit_gap_gt,
        "reveal_transition": reveal_transition,
        "reveal_planning": reveal_plans,
        "reveal_gap_closed_acc": reveal_gap,
        "elapsed_seconds": time.time() - started,
    }

    # --- print summary --------------------------------------------------------
    print(f"\n=== EDIT transition metrics (m={m}) ===")
    for name, result in edit_transition.items():
        print(f"  {name:16s} cos={result['transition_cosine']:.3f} "
              f"cmd_cos={result['command_conditional_cosine']:.3f} "
              f"cmd_spread={result['command_relative_spread']:.3f}")
    print(f"\n=== EDIT planning (control; target={init_mode}, budget={edit_budget}) ===")
    for name, result in edit_plans.items():
        match = result["gt_feature_match"]
        match_str = f"{match:.3f}" if match is not None else "  n/a"
        print(f"  {name:16s} P(r*)={result['target_prob']:.3f} "
              f"ctrl_success={result['target_success']:.3f} "
              f"gt_success={result['gt_success']:.3f} "
              f"gt_valid={result['gt_on_manifold']:.3f} "
              f"feat_match={match_str}")
    print("  fraction of random->oracle P(r*) gap closed: " +
          ", ".join(f"{k}={val:+.2f}" for k, val in edit_gap.items()))
    if edit_gap_gt:
        print("  fraction of random->gt_oracle TRUE-success gap closed: " +
              ", ".join(f"{k}={val:+.2f}" for k, val in edit_gap_gt.items()))
    print(f"\n=== REVEAL planning (inference contrast; budget={reveal_budget}) ===")
    for name, result in reveal_plans.items():
        print(f"  {name:16s} root_acc={result['root_accuracy']:.3f} "
              f"root_ce={result['root_ce']:.3f}")
    print(f"  arity-2 reveal cmd_cos={reveal_transition['command_conditional_cosine']:.3f}; "
          f"fraction of random->oracle acc gap closed={reveal_gap:+.2f}")

    tag = f"v{v}_s{s}_L{depth}_m{m}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_edit_control/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    torch.save(
        {
            "controller": controller.state_dict(),
            "edit_fm2": edit_fm2.state_dict(),
            "edit_fm1": edit_fm1.state_dict(),
            "reveal_fm": reveal_fm.state_dict(),
            "config": metrics["config"],
        },
        f"{output_dir}/models.pt",
    )
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    edit_control.remote(m=m, quick=quick)
