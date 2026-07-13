"""RHM Sculpting — Internalization: does closing the planning loop reshape the BELIEF?

The whole sculpting arc so far (Stage 3a/3b/3c/3d/4) trains every instrument ONCE on the
clean world and FREEZES it, then bolts an external beam on top — the exact analog of the
reaching arc's *decoupled* apparatus (operator frozen, FM on frozen transitions, an external
argmax planner). `a2a_forward/REACHING_INTERNAL` took the next step there: make the forecast
ENDOGENOUS (co-train it, use it from inside the loop) and it found the operator reorganizes to
be dramatically more *plannable* — a fresh, independent external planner jumps +0.55 -> +1.00 on
the co-trained operator. This file is that step for sculpting: let the latent-planning objective
reshape the controller's per-block belief `z` (the thing the block FM rolls one step of, and the
thing the beam searches).

SINGLE CONTROLLED VARIABLE = whether the planning loop is closed while the belief trains.
A nested 3-rung ladder on the SAME `parser` belief, matched init + matched controller-update
budget. Everything downstream (fresh MC value, fresh block FM, both beams, the depth probe) is
IDENTICAL across rungs — so the downstream fresh-FM latent beam is literally REACHING_INTERNAL's
"fresh, decoupled external planner on the (possibly-internalized) operator" test.

  rung 1  frozen        : root-CE only (== Stage-3b parser). The loop never touches the belief.
  rung 2  fm_cotrain    : root-CE + lam_fm * MSE(FM(z,k), z2-z), grad into controller + FM.
                          ENDOGENOUS "be predictable" pressure (the FM predicts the controller's
                          OWN latents; no task grounding in that gradient). Collapse is possible
                          and is MEASURED (participation ratio + depth probe), not assumed away.
  rung 3  planner       : rung 2 + lam_plan * CE(softmax_k value((z+FM(z,k)).mean/tau), k*),
                          grad into controller + FM + value. GROUNDED plannability pressure — the
                          imitation target k* = argmin_k d*(regenerate(x,k)) is the exact-DP best
                          move (controller-independent, precomputed). int_plan port.

Because the rungs are nested (rung3 = rung2 + grounded planner term), the contrasts isolate:
  rung2 - rung1  = effect of ENDOGENOUS predictability pressure on the belief.
  rung3 - rung2  = effect of GROUNDED plannability pressure, on top of predictability.

Two headlines, instrumented equally (as requested):
  (A) "belief becomes plannable" (REACHING_INTERNAL port): fresh-FM one-step diagnostics
      (delta_cos / value_top1_agree / value_rank_corr), token & latent beam success + the
      clean-channel gap. Fresh FM = independent consumer, so a lifted beam == transferable
      plannability of the belief.
  (B) "frontier-moving vs capped" (RHM_LATENT_LOOP unification): the per-level belief-depth
      probe (block-ancestor recovery d1..dL) + participation ratio. The sharp prediction: the
      GROUNDED planner term moves belief depth (like RHM_LATENT_LOOP's mlm/oracle grounded
      targets) while the ENDOGENOUS FM-cotrain term caps (like data2vec's teacher-capped target)
      — even though the sculpting reward "reach a true r* config" is grounded by construction.

Also watch the REACHING_INTERNAL dissociation: does grounded internalization raise the
value-relevant plannability (top1/rank_corr, beams, task-relevant depth) while full-state
veridicality (delta_cos) stays flat/drops? plannability != forward-predictability.

Run:
  modal run rhm/rhm_sculpt_internalize.py::sculpt_internalize --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_internalize.py::sculpt_internalize --m 2  # L=4,c=3
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost
from rhm.rhm_active_query import _count_parameters, _reveal_blocks
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _regenerate, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_latent_loop import _generate_with_traces
from rhm.rhm_sculpt_planner import (
    _beam_plan,
    _collect_value_sculpt,
    _corrupt,
    _sample_pool,
)
from rhm.rhm_sculpt_latent import (
    _block_state_chunked,
    _build_block_fm,
    _build_rich_controller,
    _fm_check,
    _latent_beam,
    _train_block_fm,
)
from rhm.rhm_sculpt_deepbelief import (
    _belief_depth_probe,
    _block_ancestor_index,
    _participation_ratio,
    _span_mask,
    _subtree_mask,
)


app = modal.App("rhm-sculpt-internalize", image=image)


# --------------------------------------------------------------------------- #
# Grounded oracle buffer (controller-INDEPENDENT): visited state -> DP best move
# --------------------------------------------------------------------------- #

def _collect_grounded_moves(generator, pool_leaves, pool_roots, canon, region_index, n_regions,
                            rules, *, n_states, batch_size, n_blocks, v, s, n_corrupt, budget,
                            device):
    """Precompute the grounded imitation target for rung 3. Visited states x are drawn from the
    SAME distribution the block FM is trained on (corrupt starts + random frozen-generator
    regenerations); the grounded best move k* = argmin_k d*(regenerate(x,k)) under the exact DP
    (`nearest_derivation_cost`), ties -> lowest index. This is a fact about x and the grammar, so
    it does NOT depend on the controller — precompute once, reuse across the belief phase.
    Returns (states_np (N,T), roots_np (N,), kstar_np (N,), dstar_gap mean|best-cur|)."""
    import torch
    rng = np.random.default_rng(20260713)
    xs, rs, ks = [], [], []
    d_cur_all, d_best_all = [], []
    collected = 0
    while collected < n_states:
        b = min(batch_size, n_states - collected)
        collected += b
        idx = rng.integers(0, pool_leaves.shape[0], size=b)
        roots_np = pool_roots[idx].astype(np.int64)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))
        for _ in range(g):
            kk = torch.randint(0, n_regions, (b,), device=device)
            x = _regenerate(generator, x, region_index[kk], canon, None, block_size=s, sample=False)
        x_np = x.cpu().numpy().astype(np.int64)
        d_cur = nearest_derivation_cost(rules, x_np, roots_np, s)
        dcand = np.empty((b, n_regions), dtype=np.int64)
        for k in range(n_regions):
            kk = torch.full((b,), k, device=device, dtype=torch.long)
            xk = _regenerate(generator, x, region_index[kk], canon, None, block_size=s, sample=False)
            dcand[:, k] = nearest_derivation_cost(rules, xk.cpu().numpy().astype(np.int64), roots_np, s)
        kstar = dcand.argmin(axis=1)
        d_best = dcand.min(axis=1)
        xs.append(x_np); rs.append(roots_np); ks.append(kstar)
        d_cur_all.append(d_cur); d_best_all.append(d_best)
    d_cur_all = np.concatenate(d_cur_all); d_best_all = np.concatenate(d_best_all)
    print(f"  grounded buffer: {sum(x.shape[0] for x in xs)} states, "
          f"d*_cur={d_cur_all.mean():.2f} -> best-move d*={d_best_all.mean():.2f} "
          f"(mean reduction {(d_cur_all - d_best_all).mean():.2f})")
    return np.concatenate(xs), np.concatenate(rs), np.concatenate(ks)


# --------------------------------------------------------------------------- #
# Rung 2: endogenous FM-cotrain ("be predictable")
# --------------------------------------------------------------------------- #

def _train_fm_cotrain(controller, block_fm, generator, train_leaves, train_roots, train_leaves_np,
                      canon, region_index, n_regions, *, batch_size, n_blocks, v, s, n_corrupt,
                      budget, n_steps, lr, lam_fm, device, p_full=0.5):
    """Co-train belief + block FM. Each step: the SAME root-CE anchor as the parser baseline
    (identical masked-reveal schedule), PLUS lam_fm * MSE(FM(z,k), z2-z) where z=block_state(x),
    z2=block_state(regenerate(x,k)) on the FM-training distribution. The MSE gradient flows into
    BOTH the FM and the controller (through z AND z2) — the "be predictable" pressure that
    reshapes the belief so its one-step latent dynamics are FM-rollable. Purely endogenous: the
    target is the controller's own latents, no grounding. The co-trained FM is DISCARDED after
    (a fresh reconstruction FM is trained downstream, like REACHING_INTERNAL's fresh-planner test).
    """
    import torch
    import torch.nn.functional as F

    controller.train(); block_fm.train()
    params = list(controller.parameters()) + list(block_fm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(4242)
    report = max(1, n_steps // 6)
    n_pool = train_leaves.shape[0]
    for step in range(1, n_steps + 1):
        # --- root-CE anchor (masked reveals; identical to `_train_edit_controller`) ---
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        if torch.rand(()).item() < p_full:
            n_rev = n_blocks
        else:
            n_rev = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        obs = _reveal_blocks(leaves, order[:, :n_rev], s, mask_token=-1)
        root_logits = controller.root_logits(controller.block_state(obs).mean(dim=1))
        loss_root = F.cross_entropy(root_logits, roots)

        # --- endogenous FM predictability on the visited (corrupt + regenerate) distribution ---
        cidx = rng.integers(0, train_leaves_np.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(train_leaves_np[cidx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))
        for _ in range(g):
            kk = torch.randint(0, n_regions, (batch_size,), device=device)
            x = _regenerate(generator, x, region_index[kk], canon, None, block_size=s, sample=False)
        k = torch.randint(0, n_regions, (batch_size,), device=device)
        x2 = _regenerate(generator, x, region_index[k], canon, None, block_size=s, sample=False)
        z = controller.block_state(x)
        z2 = controller.block_state(x2)
        delta = z2 - z
        # a2a-faithful ASYMMETRIC local-loss (non-collapsing): (i) FM learns the REAL detached
        # dynamics so it stays veridical regardless of the belief, (ii) the controller is PULLED
        # to make its actual delta match the FM's detached prediction ("be predictable"). root-CE
        # anchors against the constant-belief collapse; PR/depth downstream detect it if it happens.
        loss_fm_train = F.mse_loss(block_fm(z.detach(), k), delta.detach())
        loss_pred = F.mse_loss(delta, block_fm(z, k).detach())

        loss = loss_root + lam_fm * (loss_fm_train + loss_pred)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            racc = (root_logits.argmax(-1) == roots).float().mean().item()
            cos = F.cosine_similarity(block_fm(z, k).reshape(batch_size, -1),
                                      delta.reshape(batch_size, -1), dim=-1).mean().item()
            print(f"  fm-cotrain step {step:5d}/{n_steps}: root_acc={racc:.3f} "
                  f"loss_pred={loss_pred.item():.5f} fm_cos={cos:.3f}")


# --------------------------------------------------------------------------- #
# Rung 3: grounded planner-internalization (int_plan port)
# --------------------------------------------------------------------------- #

def _train_planner_internal(controller, block_fm, value, generator, train_leaves, train_roots,
                            train_leaves_np, gstates_np, groots_np, gkstar_np, canon, region_index,
                            n_regions, *, batch_size, n_blocks, v, s, n_corrupt, budget, n_steps, lr,
                            lam_fm, lam_plan, tau, device, p_full=0.5):
    """Nested on rung 2 (root-CE + endogenous FM-recon) PLUS a differentiable GROUNDED planner:
    logits_k = value((z + FM(z,k)).mean(1), root) / tau, imitation-trained against the grounded
    best move k* (DP argmin, precomputed). Gradient flows into controller + FM + value, so the
    belief reorganizes so that the FM-rolled value ranks the grounded-best move first — the
    int_plan mechanism. The internal value/FM here are vehicles (discarded downstream); the
    belief is what carries over."""
    import torch
    import torch.nn.functional as F

    controller.train(); block_fm.train(); value.train()
    params = list(controller.parameters()) + list(block_fm.parameters()) + list(value.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(4242)
    report = max(1, n_steps // 6)
    n_pool = train_leaves.shape[0]
    gstates = torch.from_numpy(gstates_np)
    groots = torch.from_numpy(groots_np)
    gk = torch.from_numpy(gkstar_np)
    n_ground = gstates.shape[0]
    arange_r = torch.arange(n_regions, device=device)
    for step in range(1, n_steps + 1):
        # --- root-CE anchor ---
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        if torch.rand(()).item() < p_full:
            n_rev = n_blocks
        else:
            n_rev = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        obs = _reveal_blocks(leaves, order[:, :n_rev], s, mask_token=-1)
        root_logits = controller.root_logits(controller.block_state(obs).mean(dim=1))
        loss_root = F.cross_entropy(root_logits, roots)

        # --- endogenous FM-recon (rung-2 term) ---
        cidx = rng.integers(0, train_leaves_np.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(train_leaves_np[cidx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))
        for _ in range(g):
            kk = torch.randint(0, n_regions, (batch_size,), device=device)
            x = _regenerate(generator, x, region_index[kk], canon, None, block_size=s, sample=False)
        k = torch.randint(0, n_regions, (batch_size,), device=device)
        x2 = _regenerate(generator, x, region_index[k], canon, None, block_size=s, sample=False)
        z = controller.block_state(x)
        z2 = controller.block_state(x2)
        delta = z2 - z
        loss_fm_train = F.mse_loss(block_fm(z.detach(), k), delta.detach())
        loss_pred = F.mse_loss(delta, block_fm(z, k).detach())
        loss_fm = loss_fm_train + loss_pred

        # --- grounded differentiable planner (rung-3 term) ---
        pidx = torch.randint(0, n_ground, (batch_size,))
        xp = gstates[pidx].to(device)
        rp = groots[pidx].to(device)
        kt = gk[pidx].to(device)
        zp = controller.block_state(xp)                        # (B, n_blocks, D), differentiable
        logits = torch.stack([
            value((zp + block_fm(zp, arange_r[j].expand(batch_size))).mean(dim=1), rp)
            for j in range(n_regions)
        ], dim=1) / tau                                        # (B, n_regions)
        loss_plan = F.cross_entropy(logits, kt)

        loss = loss_root + lam_fm * loss_fm + lam_plan * loss_plan
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            racc = (root_logits.argmax(-1) == roots).float().mean().item()
            pacc = (logits.argmax(1) == kt).float().mean().item()
            print(f"  planner step {step:5d}/{n_steps}: root_acc={racc:.3f} "
                  f"loss_fm={loss_fm.item():.5f} plan_acc={pacc:.3f} (chance {1/n_regions:.3f})")


# --------------------------------------------------------------------------- #
# Unified belief trainer: base in {parser, mlm} x mode in {frozen, planner}
# --------------------------------------------------------------------------- #

def _train_belief(controller, generator, train_leaves, train_roots, train_leaves_np, canon,
                  region_index, n_regions, *, base, mode, block_fm, value, gstates_np, groots_np,
                  gkstar_np, batch_size, n_blocks, v, s, L, state_dim, n_corrupt, budget, n_steps,
                  lr, lam_mlm, lam_fm, lam_plan, tau, mask_mode, device, p_full=0.5):
    """One nested code path for the belief-quality x internalization 2x2. Loss assembled from:
      root-CE anchor                          (always; identical masked-reveal schedule)
      + lam_mlm * masked-infilling            (base == 'mlm': Stage-4's best non-privileged belief)
      + lam_fm  * asymmetric FM local-loss    (mode == 'planner': endogenous predictability)
      + lam_plan* grounded planner CE          (mode == 'planner': int_plan, imitate DP best-move)
    So `parser_frozen` == the first-run frozen rung and `parser_planner` == its planner rung
    (a built-in reproduction check), while `mlm_planner` composes the two positive axes."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    controller.train()
    params = list(controller.parameters())
    mlm_head = None
    if base == "mlm":
        mlm_head = nn.Linear(state_dim, v).to(device)
        mlm_head.train()
        params += list(mlm_head.parameters())
    elif base != "parser":
        raise ValueError(base)
    if mode == "planner":
        block_fm.train(); value.train()
        params += list(block_fm.parameters()) + list(value.parameters())
        gstates = torch.from_numpy(gstates_np)
        groots = torch.from_numpy(groots_np)
        gk = torch.from_numpy(gkstar_np)
        n_ground = gstates.shape[0]
        arange_r = torch.arange(n_regions, device=device)
    elif mode != "frozen":
        raise ValueError(mode)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(4242)
    report = max(1, n_steps // 6)
    n_pool = train_leaves.shape[0]
    for step in range(1, n_steps + 1):
        # --- root-CE anchor (masked reveals) ---
        idx = torch.randint(0, n_pool, (batch_size,))
        leaves = train_leaves[idx].to(device)
        roots = train_roots[idx].to(device)
        if torch.rand(()).item() < p_full:
            n_rev = n_blocks
        else:
            n_rev = int(torch.randint(0, n_blocks + 1, ()).item())
        order = torch.rand(batch_size, n_blocks, device=device).argsort(dim=1)
        obs = _reveal_blocks(leaves, order[:, :n_rev], s, mask_token=-1)
        root_logits = controller.root_logits(controller.block_state(obs).mean(dim=1))
        loss = F.cross_entropy(root_logits, roots)
        mlm_acc = plan_acc = -1.0

        # --- base: masked-infilling (mlm belief) ---
        if base == "mlm":
            if mask_mode == "span":
                pos_mask = _span_mask(batch_size, n_blocks * s, device)
            else:
                pos_mask = _subtree_mask(batch_size, n_blocks, s, L, device).repeat_interleave(s, dim=1)
            mlm_logits = mlm_head(controller._encode(leaves.masked_fill(pos_mask, -1)))
            loss = loss + lam_mlm * F.cross_entropy(mlm_logits[pos_mask], leaves[pos_mask])
            mlm_acc = (mlm_logits[pos_mask].argmax(-1) == leaves[pos_mask]).float().mean().item()

        # --- mode: internalization terms (FM-recon + grounded planner) ---
        if mode == "planner":
            cidx = rng.integers(0, train_leaves_np.shape[0], size=batch_size)
            c = int(rng.integers(1, n_corrupt + 1))
            x = torch.from_numpy(_corrupt(train_leaves_np[cidx], n_blocks, c, v, s, rng)).to(device)
            g = int(rng.integers(0, budget + 1))
            for _ in range(g):
                kk = torch.randint(0, n_regions, (batch_size,), device=device)
                x = _regenerate(generator, x, region_index[kk], canon, None, block_size=s, sample=False)
            k = torch.randint(0, n_regions, (batch_size,), device=device)
            x2 = _regenerate(generator, x, region_index[k], canon, None, block_size=s, sample=False)
            z = controller.block_state(x)
            z2 = controller.block_state(x2)
            delta = z2 - z
            loss_fm_train = F.mse_loss(block_fm(z.detach(), k), delta.detach())
            loss_pred = F.mse_loss(delta, block_fm(z, k).detach())

            pidx = torch.randint(0, n_ground, (batch_size,))
            xp = gstates[pidx].to(device)
            rp = groots[pidx].to(device)
            kt = gk[pidx].to(device)
            zp = controller.block_state(xp)
            logits = torch.stack([
                value((zp + block_fm(zp, arange_r[j].expand(batch_size))).mean(dim=1), rp)
                for j in range(n_regions)
            ], dim=1) / tau
            loss_plan = F.cross_entropy(logits, kt)
            loss = loss + lam_fm * (loss_fm_train + loss_pred) + lam_plan * loss_plan
            plan_acc = (logits.argmax(1) == kt).float().mean().item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            racc = (root_logits.argmax(-1) == roots).float().mean().item()
            extra = ""
            if mlm_acc >= 0:
                extra += f" mlm_acc={mlm_acc:.3f}"
            if plan_acc >= 0:
                extra += f" plan_acc={plan_acc:.3f}"
            print(f"  belief[{base}/{mode}] step {step:5d}/{n_steps}: root_acc={racc:.3f}{extra}")


# --------------------------------------------------------------------------- #
# Shared frozen-downstream evaluation (IDENTICAL across rungs)
# --------------------------------------------------------------------------- #

def _downstream_eval(rung, controller, *, generator, value_MC, block_FM, rules, canon,
                     region_index, n_regions, train_leaves_np, train_roots_np, leaves0, targets,
                     widths, probe_leaves, probe_lf, anc_block_idx, state_dim, s, L, v, n_blocks,
                     n_corrupt, edit_budget, value_episodes, explore_eps, batch_size, value_steps,
                     fm_steps, device):
    """Freeze the belief, then run the EXACT Stage-3b/Stage-4 pipeline: belief-depth probe (+PR),
    a FRESH MC value, a FRESH reconstruction block FM, one-step FM diagnostics, and both beams.
    Fresh value+FM are the "independent consumer" — a lifted latent beam == transferable
    plannability of the belief (the REACHING_INTERNAL fresh-external-planner test)."""
    import time
    import torch

    started = time.time()
    controller.eval()
    for p in controller.parameters():
        p.requires_grad_(False)

    # (B) frontier-moving headline: belief depth + collapse check ----------------
    depth = _belief_depth_probe(controller, probe_leaves, probe_lf, anc_block_idx, L=L, v=v,
                                device=device)
    with torch.no_grad():
        zb_probe = _block_state_chunked(controller, probe_leaves.to(device))
    pr = _participation_ratio(zb_probe.reshape(-1, state_dim))
    print(f"  [{rung}] depth probe (block-ancestor acc d1..dL): "
          + " ".join(f"d{ell+1}={depth[ell]:.3f}" for ell in range(L)) + f"  PR={pr:.2f}")

    # fresh MC value (grounded success labels) -----------------------------------
    configs, roots_buf, success_buf = _collect_value_sculpt(
        controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
        rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
    print(f"  [{rung}] value buffer: {configs.shape[0]} states, "
          f"terminal success {success_buf.mean().item():.3f}")
    value = value_MC(state_dim, v).to(device)
    _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                    n_steps=value_steps, lr=3e-4, device=device)

    # fresh reconstruction block FM ----------------------------------------------
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
    print(f"  [{rung}] fresh FM: delta_cos={fm_check['delta_cos']:.3f} "
          f"top1={fm_check['value_top1_agree']:.3f} rank_corr={fm_check['value_rank_corr']:.3f}")

    # (A) plannability headline: token vs latent beams ---------------------------
    token_beam, latent_beam = {}, {}
    for width in widths:
        token_beam[width] = _beam_plan(controller, generator, value, leaves0, targets, canon,
                                       region_index, n_regions, rules, s=s, budget=edit_budget,
                                       beam_width=width, device=device)
        latent_beam[width] = _latent_beam(controller, generator, block_fm, value, leaves0, targets,
                                          canon, region_index, n_regions, rules, s=s,
                                          budget=edit_budget, beam_width=width, device=device)
        print(f"  [{rung}] width {width:3d}: token={token_beam[width]:.3f} "
              f"latent={latent_beam[width]:.3f} gap={latent_beam[width]-token_beam[width]:+.3f}")

    return {
        "rung": rung,
        "depth_probe": {f"d{ell+1}": depth[ell] for ell in range(L)},
        "participation_ratio": pr,
        "value_buffer_success": success_buf.mean().item(),
        "block_fm_check": fm_check,
        "token_beam_success": {str(k): val for k, val in token_beam.items()},
        "latent_beam_success": {str(k): val for k, val in latent_beam.items()},
        "gap": {str(k): latent_beam[k] - token_beam[k] for k in widths},
        "elapsed_seconds": time.time() - started,
    }


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=16384)
def sculpt_internalize(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, n_probe: int = 3_000,
    state_dim: int = 96, controller_steps: int = 12_000, generator_steps: int = 12_000,
    value_steps: int = 12_000, fm_steps: int = 12_000, value_episodes: int = 40_000,
    ground_states: int = 40_000, batch_size: int = 256, n_corrupt: int = 3, edit_budget: int = 6,
    region_size: int = 1, beam_widths: str = "1,16,64,256", explore_eps: float = 0.3,
    lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0,
    rungs: str = "frozen,fm_cotrain,planner", quick: bool = False,
):
    """Internalized sculpting: nested 3-rung ladder on the parser belief. Single controlled
    variable = whether (and how much of) the planning loop is closed while the belief trains."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    L = depth
    widths = [int(x) for x in beam_widths.split(",")]
    rung_list = [r for r in rungs.split(",") if r]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000
        ground_states, n_probe = 6_000, 1_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-internalize (close the planning loop onto the belief): v={v}, s={s}, L={depth}, "
          f"m={m}, blocks={n_blocks}, c={n_corrupt}, budget={edit_budget}, widths={widths}, "
          f"lam_fm={lam_fm}, lam_plan={lam_plan}, rungs={rung_list}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)
    anc_block_idx = _block_ancestor_index(n_blocks, s, L)

    # shared training pool (Stage-3b sampling, so the frozen rung reproduces the parser table) --
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    # shared depth-probe eval set (with traces for ground-truth ancestor labels) --------------
    probe_leaves_np, probe_lf, _ = _generate_with_traces(rules, n_probe, train_seed + 7)
    probe_leaves = torch.from_numpy(probe_leaves_np.astype(np.int64))

    RichController = _build_rich_controller()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()
    BlockLatentFM = _build_block_fm()

    # shared frozen generator (belief-independent) --------------------------------------------
    generator = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                              root_conditioned=False).to(device)
    _train_generator(generator, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=generator_steps, lr=3e-4, device=device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    # shared eval instances + DP ceiling ------------------------------------------------------
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

    # grounded imitation buffer for rung 3 (precomputed once, controller-independent) ----------
    if "planner" in rung_list:
        print("Precomputing grounded best-move buffer (DP over candidate moves)...")
        gstates_np, groots_np, gkstar_np = _collect_grounded_moves(
            generator, train_leaves_np, train_roots_np, canon, region_index, n_regions, rules,
            n_states=ground_states, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
            n_corrupt=n_corrupt, budget=edit_budget, device=device)
    else:
        gstates_np = groots_np = gkstar_np = None

    # run each rung through the identical downstream eval, matched init ------------------------
    results = {}
    for rung in rung_list:
        print(f"\n{'#'*72}\n# RUNG: {rung}\n{'#'*72}")
        torch.manual_seed(train_seed)      # identical controller init across rungs
        controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
        if rung == "frozen":
            print(f"Controller params: {_count_parameters(controller):,}")

        if rung == "frozen":
            _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                                   n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                                   lr=3e-4, device=device, p_full=0.5)
        elif rung == "fm_cotrain":
            block_fm_ct = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
            _train_fm_cotrain(controller, block_fm_ct, generator, train_leaves, train_roots,
                              train_leaves_np, canon, region_index, n_regions, batch_size=batch_size,
                              n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget,
                              n_steps=controller_steps, lr=3e-4, lam_fm=lam_fm, device=device)
        elif rung == "planner":
            block_fm_ct = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
            value_ct = MCValueHead(state_dim, v).to(device)
            _train_planner_internal(controller, block_fm_ct, value_ct, generator, train_leaves,
                                    train_roots, train_leaves_np, gstates_np, groots_np, gkstar_np,
                                    canon, region_index, n_regions, batch_size=batch_size,
                                    n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt,
                                    budget=edit_budget, n_steps=controller_steps, lr=3e-4,
                                    lam_fm=lam_fm, lam_plan=lam_plan, tau=tau, device=device)
        else:
            raise ValueError(rung)

        results[rung] = _downstream_eval(
            rung, controller, generator=generator, value_MC=MCValueHead, block_FM=BlockLatentFM,
            rules=rules, canon=canon, region_index=region_index, n_regions=n_regions,
            train_leaves_np=train_leaves_np, train_roots_np=train_roots_np, leaves0=leaves0,
            targets=targets, widths=widths, probe_leaves=probe_leaves, probe_lf=probe_lf,
            anc_block_idx=anc_block_idx, state_dim=state_dim, s=s, L=L, v=v, n_blocks=n_blocks,
            n_corrupt=n_corrupt, edit_budget=edit_budget, value_episodes=value_episodes,
            explore_eps=explore_eps, batch_size=batch_size, value_steps=value_steps,
            fm_steps=fm_steps, device=device)

    # summary --------------------------------------------------------------------------------
    print(f"\n{'='*72}\n=== SUMMARY (m={m}, c={n_corrupt}) — internalizing the sculpting loop ===\n{'='*72}")
    print(f"  DP frac_solvable={frac_solvable:.3f}   reference (Stage-3b parser): "
          f"token w256~0.544, latent w256~0.499, delta_cos~0.49, top1~0.36")
    for rung in rung_list:
        r = results[rung]
        fc = r["block_fm_check"]
        print(f"\n  [{rung}]  depth " + " ".join(f"{k}={v_:.2f}" for k, v_ in r["depth_probe"].items())
              + f"  PR={r['participation_ratio']:.2f}")
        print(f"           fresh FM delta_cos={fc['delta_cos']:.3f} top1={fc['value_top1_agree']:.3f} "
              f"rank_corr={fc['value_rank_corr']:.3f}")
        print("           token:  " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in r["token_beam_success"].items()))
        print("           latent: " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in r["latent_beam_success"].items()))
        print("           gap:    " + " | ".join(f"w{k}={v_:+.3f}" for k, v_ in r["gap"].items()))
    if "frozen" in results:
        base = results["frozen"]
        for rung in rung_list:
            if rung == "frozen":
                continue
            cur = results[rung]
            print(f"\n  Δ({rung} − frozen)  [headline B: does the belief frontier move?]")
            dd = {k: cur["depth_probe"][k] - base["depth_probe"][k] for k in base["depth_probe"]}
            print("    depth " + " ".join(f"{k}={dv:+.3f}" for k, dv in dd.items())
                  + f"  ΔPR={cur['participation_ratio']-base['participation_ratio']:+.2f}")
            print(f"    [headline A: does the belief become plannable?]  "
                  f"Δdelta_cos={cur['block_fm_check']['delta_cos']-base['block_fm_check']['delta_cos']:+.3f} "
                  f"Δtop1={cur['block_fm_check']['value_top1_agree']-base['block_fm_check']['value_top1_agree']:+.3f} "
                  f"Δrank_corr={cur['block_fm_check']['value_rank_corr']-base['block_fm_check']['value_rank_corr']:+.3f}")
            for k in widths:
                dl = cur["latent_beam_success"][str(k)] - base["latent_beam_success"][str(k)]
                dt = cur["token_beam_success"][str(k)] - base["token_beam_success"][str(k)]
                print(f"    w{k}: Δlatent {dl:+.3f}  Δtoken {dt:+.3f}  "
                      f"gap {base['gap'][str(k)]:+.3f}->{cur['gap'][str(k)]:+.3f}")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "beam_widths": widths, "state_dim": state_dim,
                   "lam_fm": lam_fm, "lam_plan": lam_plan, "tau": tau, "value_episodes": value_episodes,
                   "ground_states": ground_states, "rungs": rung_list, "controller_steps": controller_steps},
        "dp_frac_solvable": frac_solvable,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}_{'-'.join(rung_list)}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_internalize/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=16384)
def sculpt_compose(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, n_probe: int = 3_000,
    state_dim: int = 96, controller_steps: int = 12_000, generator_steps: int = 12_000,
    value_steps: int = 12_000, fm_steps: int = 12_000, value_episodes: int = 40_000,
    ground_states: int = 40_000, batch_size: int = 256, n_corrupt: int = 3, edit_budget: int = 6,
    region_size: int = 1, beam_widths: str = "1,16,64,256", explore_eps: float = 0.3,
    lam_mlm: float = 1.0, lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0,
    mask_mode: str = "subtree",
    conditions: str = "parser_frozen,parser_planner,mlm_frozen,mlm_planner", quick: bool = False,
):
    """Compose the two positive axes: belief quality (parser vs mlm) x internalization
    (frozen vs grounded planner), a 2x2 through the identical frozen-downstream eval. Asks:
    does grounded internalization STACK on a deeper static belief (mlm_planner best) or SUBSUME
    the belief-quality lever (parser_planner ~= mlm_planner — the grounded planner recruits the
    depth mlm supplied statically, making the hand-designed objective redundant)? Each condition
    is `<base>_<mode>` with base in {parser, mlm}, mode in {frozen, planner}. `parser_planner`
    reproduces the first run's `planner` rung as a built-in check."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    L = depth
    widths = [int(x) for x in beam_widths.split(",")]
    cond_list = [c for c in conditions.split(",") if c]
    for c in cond_list:
        b, _, md = c.partition("_")
        if b not in ("parser", "mlm") or md not in ("frozen", "planner"):
            raise ValueError(f"condition must be <parser|mlm>_<frozen|planner>, got {c}")
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 1_024, 6_000
        ground_states, n_probe = 6_000, 1_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-compose (belief-quality x internalization 2x2): v={v}, s={s}, L={depth}, m={m}, "
          f"blocks={n_blocks}, c={n_corrupt}, budget={edit_budget}, widths={widths}, "
          f"mask_mode={mask_mode}, conditions={cond_list}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)
    anc_block_idx = _block_ancestor_index(n_blocks, s, L)

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    probe_leaves_np, probe_lf, _ = _generate_with_traces(rules, n_probe, train_seed + 7)
    probe_leaves = torch.from_numpy(probe_leaves_np.astype(np.int64))

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

    planner_batch = min(1024, n_eval_episodes)
    eval_roots_np, eval_leaves_np = _sample_pool(rules, planner_batch, s, train_seed + 99)
    rng = np.random.default_rng(train_seed + 2)
    start_np = _corrupt(eval_leaves_np, n_blocks, n_corrupt, v, s, rng)
    leaves0 = torch.from_numpy(start_np)
    targets = torch.from_numpy(eval_roots_np)
    dstar = nearest_derivation_cost(rules, start_np, eval_roots_np, s)
    frac_solvable = float((dstar <= n_corrupt * s).mean())
    print(f"Task DP frac_solvable={frac_solvable:.3f}")

    if any(c.endswith("planner") for c in cond_list):
        print("Precomputing grounded best-move buffer (DP over candidate moves)...")
        gstates_np, groots_np, gkstar_np = _collect_grounded_moves(
            generator, train_leaves_np, train_roots_np, canon, region_index, n_regions, rules,
            n_states=ground_states, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
            n_corrupt=n_corrupt, budget=edit_budget, device=device)
    else:
        gstates_np = groots_np = gkstar_np = None

    results = {}
    for cond in cond_list:
        base, _, mode = cond.partition("_")
        print(f"\n{'#'*72}\n# CONDITION: {cond}  (base={base}, mode={mode})\n{'#'*72}")
        torch.manual_seed(train_seed)      # identical controller init across conditions
        controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
        block_fm_ct = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device) if mode == "planner" else None
        value_ct = MCValueHead(state_dim, v).to(device) if mode == "planner" else None
        _train_belief(controller, generator, train_leaves, train_roots, train_leaves_np, canon,
                      region_index, n_regions, base=base, mode=mode, block_fm=block_fm_ct,
                      value=value_ct, gstates_np=gstates_np, groots_np=groots_np,
                      gkstar_np=gkstar_np, batch_size=batch_size, n_blocks=n_blocks, v=v, s=s, L=L,
                      state_dim=state_dim, n_corrupt=n_corrupt, budget=edit_budget,
                      n_steps=controller_steps, lr=3e-4, lam_mlm=lam_mlm, lam_fm=lam_fm,
                      lam_plan=lam_plan, tau=tau, mask_mode=mask_mode, device=device)
        results[cond] = _downstream_eval(
            cond, controller, generator=generator, value_MC=MCValueHead, block_FM=BlockLatentFM,
            rules=rules, canon=canon, region_index=region_index, n_regions=n_regions,
            train_leaves_np=train_leaves_np, train_roots_np=train_roots_np, leaves0=leaves0,
            targets=targets, widths=widths, probe_leaves=probe_leaves, probe_lf=probe_lf,
            anc_block_idx=anc_block_idx, state_dim=state_dim, s=s, L=L, v=v, n_blocks=n_blocks,
            n_corrupt=n_corrupt, edit_budget=edit_budget, value_episodes=value_episodes,
            explore_eps=explore_eps, batch_size=batch_size, value_steps=value_steps,
            fm_steps=fm_steps, device=device)

    # summary: the 2x2 read (stack vs subsume) -----------------------------------------------
    print(f"\n{'='*72}\n=== SUMMARY (m={m}, c={n_corrupt}) — belief-quality x internalization ===\n{'='*72}")
    for cond in cond_list:
        r = results[cond]; fc = r["block_fm_check"]
        print(f"\n  [{cond}]  depth " + " ".join(f"{k}={v_:.2f}" for k, v_ in r["depth_probe"].items())
              + f"  PR={r['participation_ratio']:.2f}")
        print(f"           fresh FM delta_cos={fc['delta_cos']:.3f} top1={fc['value_top1_agree']:.3f} "
              f"rank_corr={fc['value_rank_corr']:.3f}")
        print("           token:  " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in r["token_beam_success"].items()))
        print("           latent: " + " | ".join(f"w{k}={v_:.3f}" for k, v_ in r["latent_beam_success"].items()))
        print("           gap:    " + " | ".join(f"w{k}={v_:+.3f}" for k, v_ in r["gap"].items()))

    def _delta(a, b, tag):
        if a not in results or b not in results:
            return
        ra, rb = results[a], results[b]
        w = str(widths[-1])
        print(f"\n  Δ({a} − {b})  [{tag}]")
        print("    depth " + " ".join(
            f"{k}={ra['depth_probe'][k]-rb['depth_probe'][k]:+.3f}" for k in ra["depth_probe"])
            + f"  ΔPR={ra['participation_ratio']-rb['participation_ratio']:+.2f}")
        print(f"    fresh-FM Δtop1={ra['block_fm_check']['value_top1_agree']-rb['block_fm_check']['value_top1_agree']:+.3f}"
              f"  Δrank_corr={ra['block_fm_check']['value_rank_corr']-rb['block_fm_check']['value_rank_corr']:+.3f}"
              f"  Δdelta_cos={ra['block_fm_check']['delta_cos']-rb['block_fm_check']['delta_cos']:+.3f}")
        print(f"    w{w}: Δlatent {ra['latent_beam_success'][w]-rb['latent_beam_success'][w]:+.3f}"
              f"  Δtoken {ra['token_beam_success'][w]-rb['token_beam_success'][w]:+.3f}"
              f"  gap {rb['gap'][w]:+.3f}->{ra['gap'][w]:+.3f}")

    print(f"\n{'-'*60}\n  internalization effect within each belief base:")
    _delta("parser_planner", "parser_frozen", "grounded internalization | parser base")
    _delta("mlm_planner", "mlm_frozen", "grounded internalization | mlm base")
    print(f"\n{'-'*60}\n  belief-quality effect within each mode (does mlm still help?):")
    _delta("mlm_frozen", "parser_frozen", "mlm vs parser | frozen (static belief-quality lever)")
    _delta("mlm_planner", "parser_planner", "mlm vs parser | planner (does mlm add AFTER internalizing?)")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt,
                   "move_budget": edit_budget, "beam_widths": widths, "state_dim": state_dim,
                   "lam_mlm": lam_mlm, "lam_fm": lam_fm, "lam_plan": lam_plan, "tau": tau,
                   "mask_mode": mask_mode, "value_episodes": value_episodes,
                   "ground_states": ground_states, "conditions": cond_list,
                   "controller_steps": controller_steps},
        "dp_frac_solvable": frac_solvable,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}_compose_{mask_mode}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_internalize/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_internalize.remote(m=m, quick=quick)
