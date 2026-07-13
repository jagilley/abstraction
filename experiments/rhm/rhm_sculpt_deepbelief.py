"""RHM Sculpting — Stage 0: does a DEEPER (DGP-aligned) belief improve LATENT planning?

The cross-experiment hypothesis (from the RHM_LATENT_LOOP <-> RHM_SCULPTING discussion):
the sculpting latent beam lags the token beam on the clean channel because the block FM
is a poor one-step ranker (`value_top1_agree ~= 0.36`, `delta_cos ~= 0.49`). Candidate
cause: the controller is trained as a *parser* (root-CE only), so its per-block latent
leaves the deep parse structure that governs an edit's consequence tangled/unrecruited --
the same failure mode RHM_LATENT_LOOP found for token targets. If a *latent* (deep) belief
exposes the hierarchy linearly, the FM should roll it more faithfully -> latent planning
improves, width-gating shrinks, maybe latent beats token even on the clean channel.

This file is the CEILING GATE for that idea (Stage 0). It does NOT ship a method -- it uses
a PRIVILEGED oracle-ancestor auxiliary loss (ground-truth latent tree, `_generate_with_traces`)
to make the belief maximally DGP-aligned, answering the cheap question "is there ANY planning
headroom from a deeper belief?" before investing in the non-privileged (data2vec/EMA) version.

SINGLE CONTROLLED VARIABLE = the controller's training objective. Everything else -- the
generator, the eval instances, the value-collection procedure, the block-FM training, the
beams, all hyperparameters -- is shared/identical across the two conditions:
  - `parser`: the exact Stage-3b baseline (`_train_edit_controller`, root CE only).
  - `oracle`: same architecture + same masking schedule, PLUS a per-block ancestor-label
    aux term (root CE + lam_aux * mean_ell CE(head_ell(block_latent), ancestor_ell)).

Readouts (parser vs oracle): (1) a belief-DEPTH probe (per-level linear recovery of block
ancestors from the per-block latent -- confirms the aux actually deepened the belief);
(2) block-FM one-step diagnostics (`delta_cos`, `value_top1_agree`, `value_rank_corr`);
(3) token vs latent beam success at each width, and the gap. Promise = oracle lifts the FM
diagnostics AND shrinks/flips the latent-token gap.

Run:
  modal run rhm/rhm_sculpt_deepbelief.py::sculpt_deepbelief --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_deepbelief.py::sculpt_deepbelief --m 2  # L=4,c=3
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, parse_success_and_heuristic
from rhm.rhm_active_query import _count_parameters, _reveal_blocks
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
from rhm.rhm_sculpt_planner import _beam_plan, _collect_value_sculpt, _corrupt, _sample_pool
from rhm.rhm_sculpt_latent import (
    _block_state_chunked,
    _build_block_fm,
    _build_rich_controller,
    _fm_check,
    _latent_beam,
    _train_block_fm,
)


app = modal.App("rhm-sculpt-deepbelief", image=image)


def _block_ancestor_index(n_blocks, s, L):
    """For each block b (a level-(L-1) node), the index of its ancestor node at each
    level ell in 0..L-1 (ell=0 root ... ell=L-1 the block itself)."""
    return [np.array([b // (s ** (L - 1 - ell)) for b in range(n_blocks)], dtype=np.int64)
            for ell in range(L)]


def _train_deep_controller(controller, train_leaves, train_roots, train_anc, *, state_dim,
                           batch_size, n_blocks, block_size, L, v, n_steps, lr, lam_aux,
                           device, p_full=0.5):
    """`_train_edit_controller` + a per-block ORACLE ancestor aux term. Identical masking
    schedule and root-CE to the baseline; the ONLY addition is lam_aux * mean_ell CE of a
    per-level linear head reading the per-block latent against the ground-truth ancestor at
    level ell. This is what deepens the belief (the single controlled variable vs `parser`).

    train_anc[ell]: (N, n_blocks) int64 ancestor labels; note mean_block(block_state) == state
    exactly (uniform blocks), so root logits computed from the pooled block latent match the
    baseline's `controller.state(obs)` path numerically."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    aux_heads = nn.ModuleList([nn.Linear(state_dim, v) for _ in range(L)]).to(device)
    controller.train(); aux_heads.train()
    params = list(controller.parameters()) + list(aux_heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)
    n_pool = train_leaves.shape[0]

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        labels = [train_anc[ell][idx].to(device) for ell in range(L)]  # each (B, n_blocks)
        if torch.rand(()).item() < p_full:
            n_revealed = n_blocks
        else:
            n_revealed = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        obs = _reveal_blocks(leaves, order[:, :n_revealed], block_size, mask_token=-1)

        zb = controller.block_state(obs)                 # (B, n_blocks, D)
        root_logits = controller.root_logits(zb.mean(dim=1))
        loss_root = F.cross_entropy(root_logits, roots)
        aux_ces = [F.cross_entropy(aux_heads[ell](zb).reshape(-1, v), labels[ell].reshape(-1))
                   for ell in range(L)]
        loss_aux = torch.stack(aux_ces).mean()
        loss = loss_root + lam_aux * loss_aux

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            racc = (root_logits.argmax(-1) == roots).float().mean().item()
            with torch.no_grad():
                lvl = [(aux_heads[ell](zb).argmax(-1) == labels[ell]).float().mean().item()
                       for ell in range(L)]
            print(f"  deep-ctrl step {step:5d}/{n_steps}: root_acc={racc:.3f} "
                  f"loss_aux={loss_aux.item():.4f} lvl_acc=[{', '.join(f'{a:.2f}' for a in lvl)}] "
                  f"(n_rev={n_revealed})")


def _participation_ratio(X):
    """Effective-rank proxy (Σλ)² / Σλ² of the feature covariance. ∈ [1, D]; collapse -> ~1."""
    import torch
    Xc = (X - X.mean(dim=0, keepdim=True)).float()
    ev = torch.linalg.svdvals(Xc) ** 2
    return float((ev.sum() ** 2) / (ev.pow(2).sum() + 1e-12))


def _subtree_mask(batch_size, n_blocks, s, L, device):
    """Mask one contiguous level-ell subtree per row (ell uniform in [1, L-1]): the masked
    blocks can only be inferred through their common ancestor, FORCING hierarchical inference
    (scattered masking lets the student predict a block from its local neighbour, so it never
    climbs). Always leaves >=1 kept block."""
    import torch
    ell = torch.randint(1, L, (batch_size,), device=device)
    w = s ** (L - 1 - ell)
    n_nodes = n_blocks // w
    j = (torch.rand(batch_size, device=device) * n_nodes).long()
    start = j * w
    block_ids = torch.arange(n_blocks, device=device)[None, :]
    return (block_ids >= start[:, None]) & (block_ids < (start + w)[:, None])


def _span_mask(batch_size, T, device):
    """Contiguous LEAF-span masking: random length (2..T/2) and random start, at leaf
    granularity -- uses NO knowledge of the branching factor, depth, or subtree alignment
    (SpanBERT-style; the fully non-privileged control for whether subtree-aligned masking
    'smuggled' DGP topology into the mlm belief). Guarantees >=2 masked and >=T/2 kept."""
    import torch
    span_len = torch.randint(2, T // 2 + 1, (batch_size,), device=device)
    start = (torch.rand(batch_size, device=device) * (T - span_len + 1).float()).long()
    leaf_ids = torch.arange(T, device=device)[None, :]
    return (leaf_ids >= start[:, None]) & (leaf_ids < (start + span_len)[:, None])


def _train_data2vec_controller(controller, train_leaves, train_roots, *, state_dim, batch_size,
                               n_blocks, block_size, n_steps, lr, lam_d2v, mask_ratio,
                               ema_tau_start, ema_tau_end, device, p_full=0.5, s=2, L=4,
                               mask_mode="subtree"):
    """root CE (functional head) + data2vec aux (masked view -> predict the EMA teacher's
    full-view layer-normed per-position representation at masked positions). NON-PRIVILEGED but
    teacher-capped: on RHM the EMA teacher is only as deep as root-CE, so it recruits little
    depth (Stage-1 isolation). Kept for comparison. Regression head discarded after training."""
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    teacher = copy.deepcopy(controller).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    pred_head = nn.Sequential(nn.Linear(state_dim, state_dim), nn.GELU(),
                              nn.Linear(state_dim, state_dim)).to(device)
    controller.train()
    params = list(controller.parameters()) + list(pred_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 6)
    n_pool = train_leaves.shape[0]

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        if torch.rand(()).item() < p_full:
            n_revealed = n_blocks
        else:
            n_revealed = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        reveal_obs = _reveal_blocks(leaves, order[:, :n_revealed], block_size, mask_token=-1)
        root_logits = controller.root_logits(controller.block_state(reveal_obs).mean(dim=1))
        loss_root = F.cross_entropy(root_logits, roots)

        if mask_mode == "subtree":
            mask_blocks = _subtree_mask(batch_size, n_blocks, s, L, device)
        else:
            mask_blocks = torch.rand(batch_size, n_blocks, device=device) < mask_ratio
            rows = torch.arange(batch_size, device=device)
            perm = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
            mask_blocks[rows, perm[:, 0]] = True
            mask_blocks[rows, perm[:, 1]] = False
        pos_mask = mask_blocks.repeat_interleave(block_size, dim=1)
        d2v_obs = leaves.masked_fill(pos_mask, -1)
        student_out = controller._encode(d2v_obs)
        with torch.no_grad():
            target = F.layer_norm(teacher._encode(leaves), (state_dim,))
        loss_d2v = F.mse_loss(pred_head(student_out)[pos_mask], target[pos_mask])

        loss = loss_root + lam_d2v * loss_d2v
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        tau = ema_tau_start + (ema_tau_end - ema_tau_start) * (step / n_steps)
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), controller.parameters()):
                tp.mul_(tau).add_(sp.detach(), alpha=1 - tau)
        if step % report_every == 0 or step == n_steps:
            racc = (root_logits.argmax(-1) == roots).float().mean().item()
            print(f"  data2vec step {step:5d}/{n_steps}: root_acc={racc:.3f} "
                  f"loss_d2v={loss_d2v.item():.4f} tau={tau:.4f}")


def _train_mlm_controller(controller, train_leaves, train_roots, *, v, state_dim, batch_size,
                          n_blocks, block_size, n_steps, lr, lam_mlm, device, p_full=0.5, s=2, L=4,
                          mask_mode="subtree"):
    """root CE (functional head) + masked-SUBTREE INFILLING aux (predict the true masked LEAF
    tokens from context). GROUND-TRUTH target (not teacher-capped), so it pulls the belief as
    deep as predicting a masked subtree requires -- the best non-privileged belief in Stage-1
    isolation (~40% of the oracle depth gap). Non-privileged: observed tokens + masking only."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    mlm_head = nn.Linear(state_dim, v).to(device)
    controller.train()
    params = list(controller.parameters()) + list(mlm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 6)
    n_pool = train_leaves.shape[0]

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        if torch.rand(()).item() < p_full:
            n_revealed = n_blocks
        else:
            n_revealed = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        reveal_obs = _reveal_blocks(leaves, order[:, :n_revealed], block_size, mask_token=-1)
        root_logits = controller.root_logits(controller.block_state(reveal_obs).mean(dim=1))
        loss_root = F.cross_entropy(root_logits, roots)

        if mask_mode == "span":  # fully non-privileged: no s/L/alignment knowledge
            pos_mask = _span_mask(batch_size, n_blocks * block_size, device)
        elif mask_mode == "subtree":
            pos_mask = _subtree_mask(batch_size, n_blocks, s, L, device).repeat_interleave(block_size, dim=1)
        else:
            mask_blocks = torch.rand(batch_size, n_blocks, device=device) < 0.5
            rows = torch.arange(batch_size, device=device)
            perm = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
            mask_blocks[rows, perm[:, 0]] = True
            mask_blocks[rows, perm[:, 1]] = False
            pos_mask = mask_blocks.repeat_interleave(block_size, dim=1)
        mlm_obs = leaves.masked_fill(pos_mask, -1)
        logits = mlm_head(controller._encode(mlm_obs))
        loss_mlm = F.cross_entropy(logits[pos_mask], leaves[pos_mask])

        loss = loss_root + lam_mlm * loss_mlm
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            racc = (root_logits.argmax(-1) == roots).float().mean().item()
            macc = (logits[pos_mask].argmax(-1) == leaves[pos_mask]).float().mean().item()
            print(f"  mlm step {step:5d}/{n_steps}: root_acc={racc:.3f} mlm_acc={macc:.3f} "
                  f"loss_mlm={loss_mlm.item():.4f}")


def _belief_depth_probe(controller, eval_leaves, level_features, anc_block_idx, *, L, v,
                        device, probe_steps=300):
    """Per-level linear recovery of a block's ancestor feature from its per-block latent
    (FULL observation). The discriminator between parser and oracle belief: intermediate
    levels (1..L-2) are where the parser has no pressure but the aux forces structure."""
    import torch
    with torch.no_grad():
        zb = _block_state_chunked(controller, eval_leaves.to(device))   # (N, n_blocks, D)
    n_seq, n_blocks, dim = zb.shape
    X = zb.reshape(n_seq * n_blocks, dim)
    accs = {}
    for ell in range(L):
        y = torch.from_numpy(level_features[ell][:, anc_block_idx[ell]].reshape(-1)).to(device)
        accs[ell] = _probe_acc(X, y, v, device, probe_steps, 1e-3)
    return accs


def _run_condition(belief, *, controller, generator, value_MC, block_FM, rules, canon,
                   region_index, n_regions, bottom_map, train_leaves, train_roots,
                   train_leaves_np, train_roots_np, train_anc, leaves0, targets, widths,
                   probe_leaves, probe_lf, anc_block_idx, state_dim, sequence_length, s, L, v,
                   n_blocks, n_corrupt, edit_budget, value_episodes, explore_eps, batch_size,
                   controller_steps, value_steps, fm_steps, lam_aux, lam_d2v, mask_mode,
                   ema_tau_start, ema_tau_end, device):
    """Train (controller [parser|oracle|mlm|data2vec] -> value -> block FM) then run both beams.
    The generator + eval instances are passed in already-built/frozen (shared across conditions),
    so the ONLY controlled variable across belief conditions is the controller's objective."""
    import time
    import torch

    started = time.time()
    print(f"\n{'#'*72}\n# CONDITION: belief={belief}\n{'#'*72}")

    if belief == "parser":
        _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                               n_blocks=n_blocks, block_size=s, n_steps=controller_steps, lr=3e-4,
                               device=device, p_full=0.5)
    elif belief == "oracle":
        _train_deep_controller(controller, train_leaves, train_roots, train_anc,
                               state_dim=state_dim, batch_size=batch_size, n_blocks=n_blocks,
                               block_size=s, L=L, v=v, n_steps=controller_steps, lr=3e-4,
                               lam_aux=lam_aux, device=device, p_full=0.5)
    elif belief == "mlm":
        _train_mlm_controller(controller, train_leaves, train_roots, v=v, state_dim=state_dim,
                              batch_size=batch_size, n_blocks=n_blocks, block_size=s,
                              n_steps=controller_steps, lr=3e-4, lam_mlm=lam_d2v, device=device,
                              p_full=0.5, s=s, L=L, mask_mode=mask_mode)
    elif belief == "data2vec":
        _train_data2vec_controller(controller, train_leaves, train_roots, state_dim=state_dim,
                                   batch_size=batch_size, n_blocks=n_blocks, block_size=s,
                                   n_steps=controller_steps, lr=3e-4, lam_d2v=lam_d2v,
                                   mask_ratio=0.5, ema_tau_start=ema_tau_start,
                                   ema_tau_end=ema_tau_end, device=device, p_full=0.5,
                                   s=s, L=L, mask_mode=mask_mode)
    else:
        raise ValueError(belief)
    controller.eval()
    for p in controller.parameters():
        p.requires_grad_(False)

    depth = _belief_depth_probe(controller, probe_leaves, probe_lf, anc_block_idx, L=L, v=v,
                                device=device)
    print(f"  [{belief}] belief-depth probe (block-ancestor acc per level d1..dL): "
          + " ".join(f"d{ell+1}={depth[ell]:.3f}" for ell in range(L)))

    # value (MC, pooled belief) --------------------------------------------------
    configs, roots_buf, success_buf = _collect_value_sculpt(
        controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
        rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
    print(f"  [{belief}] value buffer: {configs.shape[0]} states, "
          f"terminal success {success_buf.mean().item():.3f}")
    value = value_MC(state_dim, v).to(device)
    _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                    n_steps=value_steps, lr=3e-4, device=device)

    # block-latent FM ------------------------------------------------------------
    block_fm = block_FM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    _train_block_fm(block_fm, controller, generator, train_roots_np, train_leaves_np, canon,
                    region_index, n_regions, n_steps=fm_steps, batch_size=batch_size,
                    n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget,
                    lr=1e-3, device=device)
    for module in (value, block_fm):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    fm_check = _fm_check(block_fm, value, controller, generator, leaves0, targets, canon,
                         region_index, n_regions, s, device)
    print(f"  [{belief}] block FM: delta_cos={fm_check['delta_cos']:.3f} "
          f"value_top1_agree={fm_check['value_top1_agree']:.3f} "
          f"value_rank_corr={fm_check['value_rank_corr']:.3f}")

    token_beam, latent_beam = {}, {}
    for width in widths:
        token_beam[width] = _beam_plan(controller, generator, value, leaves0, targets, canon,
                                       region_index, n_regions, rules, s=s, budget=edit_budget,
                                       beam_width=width, device=device)
        latent_beam[width] = _latent_beam(controller, generator, block_fm, value, leaves0,
                                          targets, canon, region_index, n_regions, rules, s=s,
                                          budget=edit_budget, beam_width=width, device=device)
        print(f"  [{belief}] width {width:3d}: token={token_beam[width]:.3f} "
              f"latent={latent_beam[width]:.3f} gap={latent_beam[width]-token_beam[width]:+.3f}")

    return {
        "belief": belief,
        "depth_probe": {f"d{ell+1}": depth[ell] for ell in range(L)},
        "value_buffer_success": success_buf.mean().item(),
        "block_fm_check": fm_check,
        "token_beam_success": {str(k): v_ for k, v_ in token_beam.items()},
        "latent_beam_success": {str(k): v_ for k, v_ in latent_beam.items()},
        "gap": {str(k): latent_beam[k] - token_beam[k] for k in widths},
        "elapsed_seconds": time.time() - started,
    }


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=16384)
def sculpt_deepbelief(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, n_probe: int = 3_000,
    state_dim: int = 96, controller_steps: int = 12_000, generator_steps: int = 12_000,
    value_steps: int = 12_000, fm_steps: int = 12_000, value_episodes: int = 40_000,
    batch_size: int = 256, n_corrupt: int = 3, edit_budget: int = 6, region_size: int = 1,
    beam_widths: str = "1,16,64,256", explore_eps: float = 0.3, lam_aux: float = 1.0,
    lam_d2v: float = 1.0, mask_mode: str = "subtree", ema_tau_start: float = 0.996,
    ema_tau_end: float = 0.9999, beliefs: str = "parser,oracle", quick: bool = False,
):
    """Stage-0/2 gate: parser floor vs oracle ceiling vs non-privileged (mlm/data2vec) beliefs on
    the sculpting latent beam. `--beliefs parser,oracle,mlm,data2vec` runs the full Stage-2 map."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    L = depth
    widths = [int(x) for x in beam_widths.split(",")]
    cond_list = [b for b in beliefs.split(",") if b]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes, n_probe = 20_000, 1_024, 6_000, 1_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-deepbelief (belief-depth ceiling gate): v={v}, s={s}, L={depth}, m={m}, "
          f"blocks={n_blocks}, c={n_corrupt}, budget={edit_budget}, widths={widths}, "
          f"lam_aux={lam_aux}, conditions={cond_list}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)
    anc_block_idx = _block_ancestor_index(n_blocks, s, L)

    # shared training pool WITH traces (oracle aux labels + roots) ---------------
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    gen_leaves, gen_lf, _ = _generate_with_traces(rules, n_train_episodes, train_seed)
    train_leaves_np = gen_leaves.astype(np.int64)
    train_roots_np = gen_lf[0][:, 0].astype(np.int64)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)
    train_anc = [torch.from_numpy(gen_lf[ell][:, anc_block_idx[ell]].astype(np.int64))
                 for ell in range(L)]

    # shared depth-probe eval set (with traces) ---------------------------------
    probe_leaves_np, probe_lf, _ = _generate_with_traces(rules, n_probe, train_seed + 7)
    probe_leaves = torch.from_numpy(probe_leaves_np.astype(np.int64))

    # shared generator (belief-independent) -------------------------------------
    RichController = _build_rich_controller()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()
    BlockLatentFM = _build_block_fm()

    generator = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                              root_conditioned=False).to(device)
    _train_generator(generator, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=generator_steps, lr=3e-4, device=device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    # shared eval instances + DP ceiling ----------------------------------------
    planner_batch = min(1024, n_eval_episodes)
    eval_roots_np, eval_leaves_np = _sample_pool(rules, planner_batch, s, train_seed + 99)
    rng = np.random.default_rng(train_seed + 2)
    start_np = _corrupt(eval_leaves_np, n_blocks, n_corrupt, v, s, rng)
    leaves0 = torch.from_numpy(start_np)
    targets = torch.from_numpy(eval_roots_np)
    token_budget = n_corrupt * s
    dstar = nearest_derivation_cost(rules, start_np, eval_roots_np, s)
    frac_solvable = float((dstar <= token_budget).mean())
    print(f"Task DP frac_solvable={frac_solvable:.3f}")

    # run each belief condition through the identical pipeline -------------------
    results = {}
    for belief in cond_list:
        torch.manual_seed(train_seed)        # identical controller init across conditions
        controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
        if belief == "oracle":
            print(f"Controller params: {_count_parameters(controller):,}  "
                  f"(+ {L} aux heads of {state_dim}x{v})")
        results[belief] = _run_condition(
            belief, controller=controller, generator=generator, value_MC=MCValueHead,
            block_FM=BlockLatentFM, rules=rules, canon=canon, region_index=region_index,
            n_regions=n_regions, bottom_map=bottom_map, train_leaves=train_leaves,
            train_roots=train_roots, train_leaves_np=train_leaves_np, train_roots_np=train_roots_np,
            train_anc=train_anc, leaves0=leaves0, targets=targets, widths=widths,
            probe_leaves=probe_leaves, probe_lf=probe_lf, anc_block_idx=anc_block_idx,
            state_dim=state_dim, sequence_length=sequence_length, s=s, L=L, v=v, n_blocks=n_blocks,
            n_corrupt=n_corrupt, edit_budget=edit_budget, value_episodes=value_episodes,
            explore_eps=explore_eps, batch_size=batch_size, controller_steps=controller_steps,
            value_steps=value_steps, fm_steps=fm_steps, lam_aux=lam_aux, lam_d2v=lam_d2v,
            mask_mode=mask_mode, ema_tau_start=ema_tau_start, ema_tau_end=ema_tau_end, device=device)

    # side-by-side summary -------------------------------------------------------
    print(f"\n{'='*72}\n=== SUMMARY (m={m}, c={n_corrupt}) — belief conditions on the latent beam ===\n{'='*72}")
    print(f"  DP frac_solvable={frac_solvable:.3f}   reference (parser Stage-3b): "
          f"token w256~0.544, latent w256~0.499, delta_cos~0.49, top1~0.36")
    for belief in cond_list:
        r = results[belief]
        fc = r["block_fm_check"]
        print(f"\n  [{belief}]  depth-probe " + " ".join(f"{k}={v_:.2f}" for k, v_ in r["depth_probe"].items()))
        print(f"           FM delta_cos={fc['delta_cos']:.3f} top1={fc['value_top1_agree']:.3f} "
              f"rank_corr={fc['value_rank_corr']:.3f}")
        print("           token:  " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in r["token_beam_success"].items()))
        print("           latent: " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in r["latent_beam_success"].items()))
        print("           gap:    " + " | ".join(f"w{k}={v_:+.3f}" for k, v_ in r["gap"].items()))
    if "parser" in results:
        pp = results["parser"]
        for belief in cond_list:
            if belief == "parser":
                continue
            po = results[belief]
            print(f"\n  Δ({belief} − parser):")
            print(f"    FM delta_cos {po['block_fm_check']['delta_cos']-pp['block_fm_check']['delta_cos']:+.3f}"
                  f"  top1 {po['block_fm_check']['value_top1_agree']-pp['block_fm_check']['value_top1_agree']:+.3f}")
            for k in widths:
                dl = po["latent_beam_success"][str(k)] - pp["latent_beam_success"][str(k)]
                dt = po["token_beam_success"][str(k)] - pp["token_beam_success"][str(k)]
                print(f"    w{k}: Δlatent {dl:+.3f}  Δtoken {dt:+.3f}  Δgap {po['gap'][str(k)]-pp['gap'][str(k)]:+.3f}")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "beam_widths": widths, "state_dim": state_dim,
                   "lam_aux": lam_aux, "value_episodes": value_episodes, "conditions": cond_list},
        "dp_frac_solvable": frac_solvable,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    # include the belief set in the tag so distinct runs (Stage-0 pairs, Stage-2 4-way,
    # single-belief smokes) do not clobber one another's results.json
    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_lam{lam_aux}_seed{rule_seed}_{'-'.join(cond_list)}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_deepbelief/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_deepbelief.remote(m=m, quick=quick)
