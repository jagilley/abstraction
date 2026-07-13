"""RHM Sculpting — Stage 6: CONTINUOUS internalization. Is planning a ratchet on its own
representational substrate?

Stage 5 internalized the planning loop ONCE (a grounded planner reshaped the belief, then
freeze + evaluate). This asks whether that's a *ratchet*: does a better-planning system produce
a more *plannable* belief, which produces a better planner, which produces a still-more-plannable
belief... compounding toward the DP optimum (=1.0, the task's full-horizon-planning ceiling) —
or did the one-shot reshape already extract all the representational headroom, leaving only
ordinary policy improvement?

DEFINITION. Continuous internalization = iterate the internalization loop so belief, FM, value,
and behaviour-policy co-evolve, each round's improved planner supplying the data + targets that
reshape the next round's belief. A warm-started wake-sleep / policy-iteration loop:
  wake  (collect): run the current latent planner as behaviour policy -> rollouts -> grounded
                   terminal-success labels (value) on the visited (improving) distribution.
  sleep (consolidate): continue-train the MC value on the new labels (the policy-iteration engine
                   that breaks the "behaviour-policy success caps the value" ceiling); then
                   RE-INTERNALIZE the belief (grounded planner against the improved *frozen* value
                   + the DP best-move k*), shaping belief + FM.
The value stays a pure MC critic (improved only by policy iteration); the belief is shaped to be
legible to that improving critic in a way that matches ground truth. The grounded k* teacher is
DP-computed (a stable ground-truth teacher, not self-distillation — the same reason grounding was
the pivot in Stage 5), so the loop can't wirehead its own value.

THE DECISIVE CONTRAST — nested arms sharing round 0, matched per-round budget:
  D  round-0 only (= Stage-5 frozen floor).
  C  value-iteration only:      belief = frozen parser; iterate value + data. (does policy
                                improvement alone compound?)
  B  one-shot internalize + VI: internalize the belief ONCE (round 0), freeze it; iterate value.
  A  continuous internalization: re-internalize the belief EVERY round + iterate value.
  C vs D = value-iteration effect; B vs C = one-shot-internalization effect (Stage 5, under VI);
  **A vs B = the continuous-internalization effect** (does re-shaping the belief each round, against
  the continually-improving value, add compounding beyond one-shot?).

PER-ROUND METRICS (the compounding curve, vs round r, against DP=1.0 and strong-reflex=0.585):
  latent/token beam @ w256 (co-trained loop instruments = the actual system);
  PR, depth d3/d4 (the belief-plannability frontier — climbing or saturated?);
  behaviour-policy terminal success (the value ceiling proxy — is policy iteration working?);
  **fresh-FM top1** (an INDEPENDENT FM on the current belief = transferable/MODULAR plannability).
The fresh-vs-cotrained gap is the cancellation-inspired ENTANGLEMENT diagnostic: if the co-trained
beam climbs while fresh-FM top1 stalls/drops, the belief is privatizing to its own FM ("advisor,
not model" — the sculpting analog of summation-dependency; a2a CANCELLATION Payoff-1, which is
substrate-limited on MNIST and wants exactly RHM's gauge-freedom to resolve).

Collapse guard (self-improvement loops narrow their own data): keep explore_eps + mixed corruption
in collection; log the collection distribution's terminal success and the belief PR each round.

Run:
  modal run rhm/rhm_sculpt_continual.py::sculpt_continual --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_continual.py::sculpt_continual --m 2  # L=4,c=3, R=5
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
    _success_ps,
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
)
from rhm.rhm_sculpt_internalize import _collect_grounded_moves


app = modal.App("rhm-sculpt-continual", image=image)


def _collect_value_planner(controller, generator, value, block_fm, pool_roots, pool_leaves, canon,
                           region_index, n_regions, rules, *, n_episodes, batch_size, n_blocks, v, s,
                           n_corrupt, budget, epsilon, device):
    """Wake/collect with the CURRENT latent planner as behaviour policy (greedy value+FM latent step
    + eps exploration; re-grounded each step), labelling every visited state by its rollout's grounded
    terminal success. As value/belief/FM improve, this visits a better distribution and supplies
    higher-success labels — the policy-iteration engine. Returns (configs, roots, success, mean
    terminal success = the behaviour-policy success rate / value-ceiling proxy)."""
    import torch
    rng = np.random.default_rng(4321)
    controller.eval(); value.eval(); block_fm.eval()
    configs, roots_all, success_all, term = [], [], [], []
    collected = 0
    while collected < n_episodes:
        b = min(batch_size, n_episodes - collected)
        collected += b
        idx = rng.integers(0, pool_leaves.shape[0], size=b)
        roots_np = pool_roots[idx]
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)).to(device)
        roots = torch.from_numpy(roots_np).to(device)
        traj = [x.clone()]
        with torch.no_grad():
            for _ in range(budget):
                z = controller.block_state(x)
                scores = torch.stack([
                    value((z + block_fm(z, torch.full((b,), k, device=device, dtype=torch.long))).mean(dim=1), roots)
                    for k in range(n_regions)
                ], dim=1)
                best = scores.argmax(dim=1)
                explore = torch.rand(b, device=device) < epsilon
                rand_moves = torch.randint(0, n_regions, (b,), device=device)
                move = torch.where(explore, rand_moves, best)
                x = _regenerate(generator, x, region_index[move], canon, None, block_size=s, sample=False)
                traj.append(x.clone())
        succ = torch.from_numpy(_success_ps(x.cpu().numpy(), roots_np, rules, s).astype(np.float32))
        term.append(succ.mean().item())
        for st in traj:
            configs.append(st.cpu()); roots_all.append(roots.cpu()); success_all.append(succ)
    return torch.cat(configs), torch.cat(roots_all), torch.cat(success_all), float(np.mean(term))


def _reinternalize_belief(controller, block_fm, value, generator, train_leaves, train_roots,
                          train_leaves_np, gstates_np, groots_np, gkstar_np, canon, region_index,
                          n_regions, *, batch_size, n_blocks, v, s, n_corrupt, budget, n_steps, lr,
                          lam_fm, lam_plan, tau, device, p_full=0.5):
    """Sleep/consolidate: shape belief + FM against the grounded best-move k* under a FROZEN MC value
    (the value is the improving critic; the belief learns to present states legible to it in a way
    that matches ground truth). = Stage-5 `_train_planner_internal` with the value held frozen
    (excluded from the optimizer, eval mode). Warm-started (modules already exist). Loss =
    root-CE anchor + lam_fm*(asymmetric FM local-loss) + lam_plan*(planner CE vs k*)."""
    import torch
    import torch.nn.functional as F

    controller.train(); block_fm.train(); value.eval()
    for p in value.parameters():
        p.requires_grad_(False)
    params = list(controller.parameters()) + list(block_fm.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(4242)
    report = max(1, n_steps // 4)
    n_pool = train_leaves.shape[0]
    gstates = torch.from_numpy(gstates_np)
    groots = torch.from_numpy(groots_np)
    gk = torch.from_numpy(gkstar_np)
    n_ground = gstates.shape[0]
    arange_r = torch.arange(n_regions, device=device)
    for step in range(1, n_steps + 1):
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

        loss = loss_root + lam_fm * (loss_fm_train + loss_pred) + lam_plan * loss_plan
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % report == 0 or step == n_steps:
            pacc = (logits.argmax(1) == kt).float().mean().item()
            print(f"    reinternalize step {step:4d}/{n_steps}: plan_acc={pacc:.3f} "
                  f"loss_fm={(loss_fm_train+loss_pred).item():.4f}")
    controller.eval(); block_fm.eval()


def _round_eval(rung, r, controller, generator, value, block_fm, leaves0, targets, canon,
                region_index, n_regions, rules, probe_leaves, probe_lf, anc_block_idx, BlockFM,
                train_roots_np, train_leaves_np, *, width, state_dim, s, L, v, n_blocks, n_corrupt,
                edit_budget, fm_steps_fresh, batch_size, behaviour_success, value_buffer_success,
                device):
    """Per-round readout. co-trained beam = the loop's actual value+FM (system performance). fresh-FM
    top1 = an INDEPENDENT FM trained from scratch on the current belief (transferable/modular
    plannability — the entanglement diagnostic). PR + depth = the plannability frontier."""
    import torch
    controller.eval()
    depth = _belief_depth_probe(controller, probe_leaves, probe_lf, anc_block_idx, L=L, v=v, device=device)
    with torch.no_grad():
        zb = _block_state_chunked(controller, probe_leaves.to(device))
    pr = _participation_ratio(zb.reshape(-1, state_dim))

    cotrained = _fm_check(block_fm, value, controller, generator, leaves0, targets, canon,
                          region_index, n_regions, s, device)
    token = _beam_plan(controller, generator, value, leaves0, targets, canon, region_index,
                       n_regions, rules, s=s, budget=edit_budget, beam_width=width, device=device)
    latent = _latent_beam(controller, generator, block_fm, value, leaves0, targets, canon,
                          region_index, n_regions, rules, s=s, budget=edit_budget, beam_width=width,
                          device=device)

    # fresh, independent FM on the current belief (modularity/entanglement diagnostic)
    fresh_fm = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    _train_block_fm(fresh_fm, controller, generator, train_roots_np, train_leaves_np, canon,
                    region_index, n_regions, n_steps=fm_steps_fresh, batch_size=batch_size,
                    n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3,
                    device=device)
    fresh_fm.eval()
    for p in fresh_fm.parameters():
        p.requires_grad_(False)
    fresh = _fm_check(fresh_fm, value, controller, generator, leaves0, targets, canon, region_index,
                      n_regions, s, device)

    rec = {
        "round": r, "depth_d3": depth[2], "depth_d4": depth[3], "PR": pr,
        "behaviour_success": behaviour_success, "value_buffer_success": value_buffer_success,
        "token_w": token, "latent_w": latent,
        "cotrained_fm_top1": cotrained["value_top1_agree"], "cotrained_delta_cos": cotrained["delta_cos"],
        "fresh_fm_top1": fresh["value_top1_agree"], "fresh_fm_rank_corr": fresh["value_rank_corr"],
    }
    print(f"  [{rung} r{r}] latent={latent:.3f} token={token:.3f} | fresh_top1={fresh['value_top1_agree']:.3f} "
          f"cotr_top1={cotrained['value_top1_agree']:.3f} | PR={pr:.1f} d3={depth[2]:.3f} "
          f"beh_succ={behaviour_success:.3f} val_buf={value_buffer_success:.3f}")
    return rec


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=16384)
def sculpt_continual(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 2_048, n_probe: int = 3_000,
    state_dim: int = 96, rounds: int = 5, controller_steps0: int = 12_000, generator_steps: int = 12_000,
    value_steps0: int = 12_000, fm_steps0: int = 12_000, internalize_steps0: int = 10_000,
    value_steps_round: int = 5_000, reint_steps_round: int = 6_000, fm_steps_fresh: int = 8_000,
    value_episodes: int = 40_000, value_episodes_round: int = 30_000, ground_states: int = 40_000,
    batch_size: int = 256, n_corrupt: int = 3, edit_budget: int = 6, region_size: int = 1,
    eval_width: int = 256, final_widths: str = "1,16,64,256", explore_eps: float = 0.3,
    lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0, arms: str = "A,B,C",
    quick: bool = False,
):
    """Continuous internalization: the compounding curve across rounds for arms A (continuous) /
    B (one-shot + value-iter) / C (value-iter only). Round 0 (parser belief + FM + value, and for
    A/B the first internalization) is the shared start; D (frozen floor) = C's round 0."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    L = depth
    arm_list = [a for a in arms.split(",") if a]
    final_w = [int(x) for x in final_widths.split(",")]
    if quick:
        rounds = 2
        controller_steps0 = generator_steps = value_steps0 = fm_steps0 = 800
        internalize_steps0 = value_steps_round = reint_steps_round = fm_steps_fresh = 600
        n_train_episodes, n_eval_episodes = 20_000, 1_024
        value_episodes = value_episodes_round = ground_states = 6_000
        n_probe = 1_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-continual (is planning a ratchet on its own substrate?): v={v}, s={s}, L={depth}, "
          f"m={m}, blocks={n_blocks}, c={n_corrupt}, budget={edit_budget}, rounds={rounds}, "
          f"arms={arm_list}, eval_width={eval_width}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
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
    print(f"Task DP frac_solvable={frac_solvable:.3f} (strong reflex ref 0.585)")

    print("Precomputing grounded best-move buffer (fixed broad distribution; stable DP teacher)...")
    gstates_np, groots_np, gkstar_np = _collect_grounded_moves(
        generator, train_leaves_np, train_roots_np, canon, region_index, n_regions, rules,
        n_states=ground_states, batch_size=1024, n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt,
        budget=edit_budget, device=device)

    def reint(controller, block_fm, value, n_steps):
        _reinternalize_belief(controller, block_fm, value, generator, train_leaves, train_roots,
                              train_leaves_np, gstates_np, groots_np, gkstar_np, canon, region_index,
                              n_regions, batch_size=batch_size, n_blocks=n_blocks, v=v, s=s,
                              n_corrupt=n_corrupt, budget=edit_budget, n_steps=n_steps, lr=3e-4,
                              lam_fm=lam_fm, lam_plan=lam_plan, tau=tau, device=device)

    results = {}
    for arm in arm_list:
        print(f"\n{'#'*72}\n# ARM {arm}  ({'continuous' if arm=='A' else 'one-shot+VI' if arm=='B' else 'value-iter only'})\n{'#'*72}")
        torch.manual_seed(train_seed)   # identical init across arms
        controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
        if arm == arm_list[0]:
            print(f"Controller params: {_count_parameters(controller):,}")

        # --- round 0: parser belief + FM + value (shared start), then A/B internalize once ---
        _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                               n_blocks=n_blocks, block_size=s, n_steps=controller_steps0, lr=3e-4,
                               device=device, p_full=0.5)
        block_fm = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
        _train_block_fm(block_fm, controller, generator, train_roots_np, train_leaves_np, canon,
                        region_index, n_regions, n_steps=fm_steps0, batch_size=batch_size,
                        n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3,
                        device=device)
        configs, roots_buf, success_buf = _collect_value_sculpt(
            controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
            rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
            n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, device=device)
        value = MCValueHead(state_dim, v).to(device)
        _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                        n_steps=value_steps0, lr=3e-4, device=device)
        val_buf0 = success_buf.mean().item()
        if arm in ("A", "B"):
            print("  round-0 internalize (shared A/B)...")
            reint(controller, block_fm, value, internalize_steps0)
        for module in (value, block_fm):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
        rec0 = _round_eval(arm, 0, controller, generator, value, block_fm, leaves0, targets, canon,
                           region_index, n_regions, rules, probe_leaves, probe_lf, anc_block_idx,
                           BlockLatentFM, train_roots_np, train_leaves_np, width=eval_width,
                           state_dim=state_dim, s=s, L=L, v=v, n_blocks=n_blocks, n_corrupt=n_corrupt,
                           edit_budget=edit_budget, fm_steps_fresh=fm_steps_fresh,
                           batch_size=batch_size, behaviour_success=val_buf0,
                           value_buffer_success=val_buf0, device=device)
        arm_recs = [rec0]

        # --- rounds 1..R: collect (planner policy) -> value-iterate -> [A: re-internalize] ---
        for r in range(1, rounds + 1):
            for p in value.parameters():
                p.requires_grad_(True)
            configs, roots_buf, success_buf, beh = _collect_value_planner(
                controller, generator, value, block_fm, train_roots_np, train_leaves_np, canon,
                region_index, n_regions, rules, n_episodes=value_episodes_round, batch_size=1024,
                n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget,
                epsilon=explore_eps, device=device)
            _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                            n_steps=value_steps_round, lr=3e-4, device=device)
            val_buf = success_buf.mean().item()
            if arm == "A":
                for p in block_fm.parameters():
                    p.requires_grad_(True)
                reint(controller, block_fm, value, reint_steps_round)
            for module in (value, block_fm):
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
            rec = _round_eval(arm, r, controller, generator, value, block_fm, leaves0, targets, canon,
                              region_index, n_regions, rules, probe_leaves, probe_lf, anc_block_idx,
                              BlockLatentFM, train_roots_np, train_leaves_np, width=eval_width,
                              state_dim=state_dim, s=s, L=L, v=v, n_blocks=n_blocks, n_corrupt=n_corrupt,
                              edit_budget=edit_budget, fm_steps_fresh=fm_steps_fresh,
                              batch_size=batch_size, behaviour_success=beh, value_buffer_success=val_buf,
                              device=device)
            arm_recs.append(rec)

        # --- final-round full width sweep (co-trained instruments) ---
        final_sweep = {}
        for w in final_w:
            t = _beam_plan(controller, generator, value, leaves0, targets, canon, region_index,
                           n_regions, rules, s=s, budget=edit_budget, beam_width=w, device=device)
            la = _latent_beam(controller, generator, block_fm, value, leaves0, targets, canon,
                              region_index, n_regions, rules, s=s, budget=edit_budget, beam_width=w,
                              device=device)
            final_sweep[str(w)] = {"token": t, "latent": la, "gap": la - t}
        results[arm] = {"rounds": arm_recs, "final_width_sweep": final_sweep}

    # --- summary: the compounding curves ---
    print(f"\n{'='*72}\n=== SUMMARY (m={m}, c={n_corrupt}) — CONTINUOUS INTERNALIZATION ===\n{'='*72}")
    print(f"  DP frac_solvable={frac_solvable:.3f}  (strong reflex 0.585; Stage-5 planner latent w256 ~0.591)")
    for arm in arm_list:
        recs = results[arm]["rounds"]
        print(f"\n  ARM {arm} — latent@w{eval_width} by round: " + " -> ".join(f"r{x['round']}:{x['latent_w']:.3f}" for x in recs))
        print(f"           fresh_top1 by round:  " + " -> ".join(f"{x['fresh_fm_top1']:.3f}" for x in recs))
        print(f"           PR by round:          " + " -> ".join(f"{x['PR']:.1f}" for x in recs))
        print(f"           beh_success by round: " + " -> ".join(f"{x['behaviour_success']:.3f}" for x in recs))
    if "A" in results and "B" in results:
        a_last = results["A"]["rounds"][-1]["latent_w"]
        b_last = results["B"]["rounds"][-1]["latent_w"]
        print(f"\n  DECISIVE  A(continuous) − B(one-shot+VI) latent@final: {a_last - b_last:+.3f}  "
              f"(A {a_last:.3f} vs B {b_last:.3f})  [>0 ⇒ continuous internalization compounds]")
    if "B" in results and "C" in results:
        b_last = results["B"]["rounds"][-1]["latent_w"]
        c_last = results["C"]["rounds"][-1]["latent_w"]
        print(f"  ref       B(one-shot) − C(value-iter only) latent@final: {b_last - c_last:+.3f}")

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_corrupt": n_corrupt, "rounds": rounds,
                   "eval_width": eval_width, "final_widths": final_w, "arms": arm_list,
                   "lam_fm": lam_fm, "lam_plan": lam_plan, "state_dim": state_dim},
        "dp_frac_solvable": frac_solvable, "results": results,
        "elapsed_seconds": time.time() - started,
    }
    tag = f"v{v}_s{s}_L{depth}_m{m}_c{n_corrupt}_seed{rule_seed}_R{rounds}_{'-'.join(arm_list)}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_continual/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    sculpt_continual.remote(m=m, quick=quick)
