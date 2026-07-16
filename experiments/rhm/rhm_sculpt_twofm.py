"""RHM Sculpting — ONE forward model vs TWO (cerebellum + hippocampus).

Motivation (see RHM_SCULPTING_README + the hippocampus discussion). The length-gen
finalizer (rhm_sculpt_lengthgen.py) found the forward model is irreplaceable exactly
in the OPEN-LOOP regime: an imagined rollout that never re-grounds (roll the FM's OWN
predicted latent forward, execute only the finally-selected move sequence). That maps
onto the brain's DIVISION of forward-model labor:
  * CEREBELLUM  — fast, one-step, used WITH feedback (online correction). In our infra
    this is the block FM used in the CLOSED-loop re-grounded beam (rank one step from a
    TRUE latent, materialize + re-encode each step).
  * HIPPOCAMPUS — extended, compressed, multi-step rollout at a choice point, used
    WITHOUT executing (vicarious trial-and-error / forward sweeps). In our infra this is
    the OPEN-loop imagined rollout.
The current apparatus uses ONE FM for both regimes. Its objective is one-step accuracy
(it only ever sees TRUE latents as input during training), so when it must roll its own
predictions forward open-loop it drifts past the training horizon — the README's
"needs explicit multi-step-consistency FM training to extend."

The brain evolved TWO specialized organs because the two objectives conflict: one-step
ranking accuracy (cerebellum, closed loop) vs multi-step self-consistency (hippocampus,
open loop). This file separates them and measures how big the single "one FM vs two FM"
variable is.

THE SINGLE LOAD-BEARING VARIABLE — which FM rolls the OPEN loop:
  * ONE-FM (baseline)      : one-step "cerebellar" FM does everything (== lengthgen).
  * TWO-FM (cere + hippo)  : one-step FM for the CLOSED loop; a dedicated MULTI-STEP-
                             consistency "hippocampal" FM for the OPEN loop.
Everything else is shared/frozen (controller, generator, MC value, eval instances, the
closed-loop beam). The multi-step FM is trained on the SAME data as the one-step FM,
differing ONLY in the loss horizon (unrolled H steps, fed its own predictions) — so the
open-loop contrast open_multistep vs open_1step isolates exactly the training objective.

WHY TWO ORGANS (diagnostic condition):
  * MULTISTEP-ONLY         : the multi-step FM does BOTH loops. If multi-step training
                             costs one-step ranking (closed_multistep < closed_1step),
                             a single shared FM cannot serve both regimes — the
                             computational justification for specialization.

Beam lines computed per (horizon c, width):
  token          — token beam (materialize + re-encode; arity-2 by construction)
  closed_1step   — closed-loop re-grounded beam, one-step FM      [cerebellum]
  open_1step     — open-loop imagined rollout,  one-step FM       [ONE-FM open loop]
  open_multistep — open-loop imagined rollout,  multi-step FM     [TWO-FM open loop]
  closed_multistep — closed-loop re-grounded beam, multi-step FM  [tradeoff diagnostic]
  open_a1        — open-loop, arity-1 one-step FM                 [random floor]

Run:
  modal run rhm/rhm_sculpt_twofm.py::twofm --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_twofm.py::twofm --m 2
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, parse_success_and_heuristic
from rhm.rhm_active_query import _count_parameters
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _regenerate, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_sculpt_planner import (
    _beam_plan,
    _collect_value_sculpt,
    _corrupt,
    _sample_pool,
)
from rhm.rhm_sculpt_latent import (
    _build_block_fm,
    _build_rich_controller,
    _fm_check,
    _train_block_fm,
)
from rhm.rhm_sculpt_lengthgen import _build_block_fm_arity1, _latent_beam


app = modal.App("rhm-sculpt-twofm", image=image)


def _train_block_fm_multistep(fm, controller, generator, pool_roots, pool_leaves, canon,
                              region_index, n_regions, *, n_steps, batch_size, n_blocks, v, s,
                              n_corrupt, budget, horizon, lr, device):
    """Hippocampal FM: trained to roll its OWN predictions forward for H steps and stay on
    the TRUE multi-step latent trajectory (open-loop self-consistency).

    Contrast with the one-step (cerebellar) `_train_block_fm`: that FM only ever sees a
    TRUE latent as input, so it has a train/test mismatch in the open-loop rollout, where
    its input is its own (drifting) prediction. Here the FM is UNROLLED during training —
    z_hat_{t} = z_hat_{t-1} + FM(z_hat_{t-1}, k_t), with gradients flowing through the
    whole composition — so it learns to be self-consistent when fed its own outputs.

    Data is drawn IDENTICALLY to `_train_block_fm` (same corrupt levels, same generator
    pre-steps, same random moves); the ONLY difference is the loss horizon H (>=1). At
    H==1 this reduces exactly to the one-step MSE, so H is the single controlled knob."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(778)
    fm.train()
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, pool_leaves.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))  # generator pre-steps -> match visited distribution
        for _ in range(g):
            k = torch.randint(0, n_regions, (batch_size,), device=device)
            x = _regenerate(generator, x, region_index[k], canon, None, block_size=s, sample=False)
        H = int(rng.integers(1, horizon + 1))
        # materialize the TRUE trajectory (no grad) for a fresh random move sequence
        with torch.no_grad():
            z0 = controller.block_state(x)
            ks, true_z = [], []
            xt = x
            for _ in range(H):
                k = torch.randint(0, n_regions, (batch_size,), device=device)
                xt = _regenerate(generator, xt, region_index[k], canon, None, block_size=s, sample=False)
                ks.append(k)
                true_z.append(controller.block_state(xt))
        # roll the FM OPEN-loop from z0 (grad through the composition); match every step
        zhat = z0
        loss = 0.0
        for t in range(H):
            zhat = zhat + fm(zhat, ks[t])
            loss = loss + F.mse_loss(zhat, true_z[t])
        loss = loss / H
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            with torch.no_grad():
                fin = F.cosine_similarity((zhat - z0).reshape(batch_size, -1),
                                          (true_z[-1] - z0).reshape(batch_size, -1), dim=-1).mean().item()
            print(f"  multistep FM step {step:5d}/{n_steps}: mse={loss.item():.5f} "
                  f"H={H} end_cos={fin:.3f}")


def _rollout_fidelity(fm, controller, generator, leaves0, canon, region_index, n_regions,
                      s, horizon, device, seed=0):
    """Open-loop rollout drift curve: roll a FIXED random move sequence with the FM's OWN
    predictions and compare to the true materialized trajectory at each step. Returns
    per-step cosine(zhat_t - z0, z_true_t - z0). The hippocampal FM should stay high
    (veridical) for more steps than the cerebellar one-step FM before drifting."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        x = leaves0.to(device)
        z0 = controller.block_state(x)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        zhat, xt, cos_by_step = z0, x, []
        for _ in range(horizon):
            k = torch.randint(0, n_regions, (x.shape[0],), device=device, generator=gen)
            zhat = zhat + fm(zhat, k)
            xt = _regenerate(generator, xt, region_index[k], canon, None, block_size=s, sample=False)
            z_true = controller.block_state(xt)
            cos = F.cosine_similarity((zhat - z0).reshape(x.shape[0], -1),
                                      (z_true - z0).reshape(x.shape[0], -1), dim=-1).mean().item()
            cos_by_step.append(cos)
    return cos_by_step


def _eval_at(controller, generator, fm_1step, fm_multistep, fm_a1, value, leaves0, eval_roots,
             canon, region_index, n_regions, rules, *, s, budget, widths, device, tiebreak_base):
    """The 6 beam lines at every width. token + {closed,open}x{1step,multistep} + open_a1."""
    out = {"budget": budget, "token": {}, "closed_1step": {}, "open_1step": {},
           "open_multistep": {}, "closed_multistep": {}, "open_a1": {}}

    def lb(fm, width, arity1, open_loop, tb):
        return _latent_beam(controller, generator, fm, value, leaves0, eval_roots, canon,
                            region_index, n_regions, rules, s=s, budget=budget, beam_width=width,
                            device=device, arity1=arity1, open_loop=open_loop, tiebreak_seed=tb)

    for width in widths:
        w = str(width)
        out["token"][w] = _beam_plan(controller, generator, value, leaves0, eval_roots, canon,
                                     region_index, n_regions, rules, s=s, budget=budget,
                                     beam_width=width, device=device)
        out["closed_1step"][w] = lb(fm_1step, width, False, False, 0)
        out["open_1step"][w] = lb(fm_1step, width, False, True, 0)
        out["open_multistep"][w] = lb(fm_multistep, width, False, True, 0)
        out["closed_multistep"][w] = lb(fm_multistep, width, False, False, 0)
        out["open_a1"][w] = lb(fm_a1, width, True, True, tiebreak_base + width)
    return out


def _print_row(label, res, widths):
    w1, wmax = str(widths[0]), str(widths[-1])
    print(f"  {label} (budget {res['budget']}): "
          f"OPEN 1step/multi w{w1}={res['open_1step'][w1]:.3f}/{res['open_multistep'][w1]:.3f} "
          f"w{wmax}={res['open_1step'][wmax]:.3f}/{res['open_multistep'][wmax]:.3f} || "
          f"CLOSED 1step/multi w{wmax}={res['closed_1step'][wmax]:.3f}/{res['closed_multistep'][wmax]:.3f} || "
          f"tok w{wmax}={res['token'][wmax]:.3f} | a1_open w{wmax}={res['open_a1'][wmax]:.3f}")


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def twofm(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 1_024, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256,
    train_corrupt: int = 3, test_corrupts: str = "1,2,3,4,5,6", beam_widths: str = "1,4,16,64",
    region_size: int = 1, explore_eps: float = 0.3, quick: bool = False,
):
    """Train the sculpting apparatus once at the short range (c<=train_corrupt), plus a
    one-step (cerebellar) FM AND a multi-step (hippocampal) FM on identical data, then a
    length sweep isolating the one-FM-vs-two-FM variable in the open-loop imagined rollout."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    c_tests = [int(x) for x in test_corrupts.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 512, 6_000
        c_tests, widths = [1, 3, 5], [1, 16]
    c_tests = [c for c in c_tests if c < n_blocks]  # _corrupt draws c distinct blocks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    train_budget = train_corrupt * s
    print(f"Sculpt-twofm (one FM vs two): v={v}, s={s}, L={depth}, m={m}, blocks={n_blocks}, "
          f"train_corrupt(<=)={train_corrupt}, train_budget={train_budget}, test_corrupts={c_tests}, "
          f"widths={widths}, device={device}")
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon = torch.from_numpy(canon_np).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)

    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    RichController = _build_rich_controller()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()
    BlockLatentFM = _build_block_fm()
    BlockLatentFMArity1 = _build_block_fm_arity1()

    controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
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

    print(f"Collecting value data ({value_episodes} rollouts, c<={train_corrupt}, budget {train_budget})")
    configs, roots_buf, success_buf = _collect_value_sculpt(
        controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
        rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
        n_corrupt=train_corrupt, budget=train_budget, epsilon=explore_eps, device=device)
    print(f"  buffer: {configs.shape[0]} states, terminal success rate {success_buf.mean().item():.3f}")
    value = MCValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                    n_steps=value_steps, lr=3e-4, device=device)

    # THREE FMs, all arity-2 except the a1 floor. one-step and multi-step share the SAME
    # random-move training data (torch reseeded before each) — only the loss horizon differs.
    fm_1step = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    fm_multistep = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    fm_a1 = BlockLatentFMArity1(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    print(f"FM params: 1step {_count_parameters(fm_1step):,} | multistep "
          f"{_count_parameters(fm_multistep):,} | a1 {_count_parameters(fm_a1):,}")

    torch.manual_seed(train_seed + 500)
    print("Training ONE-STEP (cerebellar) FM")
    _train_block_fm(fm_1step, controller, generator, train_roots_np, train_leaves_np, canon,
                    region_index, n_regions, n_steps=fm_steps, batch_size=batch_size,
                    n_blocks=n_blocks, v=v, s=s, n_corrupt=train_corrupt, budget=train_budget,
                    lr=1e-3, device=device)
    torch.manual_seed(train_seed + 500)  # matched init/data draw vs the one-step FM
    print(f"Training MULTI-STEP (hippocampal) FM (unrolled up to H={train_budget})")
    _train_block_fm_multistep(fm_multistep, controller, generator, train_roots_np, train_leaves_np,
                              canon, region_index, n_regions, n_steps=fm_steps, batch_size=batch_size,
                              n_blocks=n_blocks, v=v, s=s, n_corrupt=train_corrupt, budget=train_budget,
                              horizon=train_budget, lr=1e-3, device=device)
    torch.manual_seed(train_seed + 500)
    print("Training arity-1 one-step FM (open-loop random floor)")
    _train_block_fm(fm_a1, controller, generator, train_roots_np, train_leaves_np, canon,
                    region_index, n_regions, n_steps=fm_steps, batch_size=batch_size,
                    n_blocks=n_blocks, v=v, s=s, n_corrupt=train_corrupt, budget=train_budget,
                    lr=1e-3, device=device)
    for module in (value, fm_1step, fm_multistep, fm_a1):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # eval instances (one clean pool, corrupted at each c) ------------------------
    planner_batch = min(1024, n_eval_episodes)
    eval_roots_np, eval_leaves_np = _sample_pool(rules, planner_batch, s, train_seed + 99)
    eval_roots = torch.from_numpy(eval_roots_np)

    def corrupt_at(c_test):
        rng = np.random.default_rng(train_seed + 1000 + c_test)
        return torch.from_numpy(_corrupt(eval_leaves_np, n_blocks, c_test, v, s, rng))

    # one-step FM ranking diagnostics (the closed-loop currency) at the training boundary
    leaves_diag = corrupt_at(train_corrupt)
    fmc_1 = _fm_check(fm_1step, value, controller, generator, leaves_diag, eval_roots, canon,
                      region_index, n_regions, s, device)
    fmc_m = _fm_check(fm_multistep, value, controller, generator, leaves_diag, eval_roots, canon,
                      region_index, n_regions, s, device)
    print(f"\n=== one-step ranking (c={train_corrupt}): "
          f"1step top1/rankcorr/dcos={fmc_1['value_top1_agree']:.3f}/{fmc_1['value_rank_corr']:.3f}/{fmc_1['delta_cos']:.3f} | "
          f"multistep={fmc_m['value_top1_agree']:.3f}/{fmc_m['value_rank_corr']:.3f}/{fmc_m['delta_cos']:.3f} ===")

    # open-loop rollout DRIFT curves (the mechanism) at a train-range and an OOD horizon
    max_budget = max(c_tests) * s
    fidelity = {}
    for c_fid in sorted({train_corrupt, max(c_tests)}):
        lv = corrupt_at(c_fid)
        fidelity[f"c{c_fid}"] = {
            "1step": _rollout_fidelity(fm_1step, controller, generator, lv, canon, region_index,
                                       n_regions, s, max_budget, device, seed=c_fid),
            "multistep": _rollout_fidelity(fm_multistep, controller, generator, lv, canon, region_index,
                                           n_regions, s, max_budget, device, seed=c_fid),
        }
        f1, fm_ = fidelity[f"c{c_fid}"]["1step"], fidelity[f"c{c_fid}"]["multistep"]
        print(f"  rollout drift c={c_fid} (cos vs step): "
              f"1step={[round(x, 2) for x in f1]} | multistep={[round(x, 2) for x in fm_]}")

    print(f"\n=== length sweep: ONE-FM vs TWO-FM in the OPEN loop (budget = c*s) ===")
    length_sweep = {}
    for c_test in c_tests:
        leaves0 = corrupt_at(c_test)
        res = _eval_at(controller, generator, fm_1step, fm_multistep, fm_a1, value, leaves0,
                       eval_roots, canon, region_index, n_regions, rules, s=s, budget=c_test * s,
                       widths=widths, device=device, tiebreak_base=c_test * 100)
        res["dstar_mean"] = float(nearest_derivation_cost(rules, leaves0.numpy(), eval_roots_np, s).mean())
        length_sweep[str(c_test)] = res
        _print_row(f"c={c_test}", res, widths)

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_blocks": n_blocks,
                   "train_corrupt": train_corrupt, "train_budget": train_budget,
                   "test_corrupts": c_tests, "beam_widths": widths, "region_size": region_size,
                   "state_dim": state_dim, "value_episodes": value_episodes,
                   "rule_seed": rule_seed, "train_seed": train_seed},
        "value_buffer_success_rate": success_buf.mean().item(),
        "fm_check_1step": fmc_1, "fm_check_multistep": fmc_m,
        "rollout_fidelity": fidelity,
        "length_sweep": length_sweep,
        "note": "ONE-FM = {closed_1step, open_1step}; TWO-FM = {closed_1step, open_multistep}; "
                "MULTISTEP-ONLY = {closed_multistep, open_multistep}. The load-bearing single "
                "variable is open_multistep vs open_1step (same everything, only the open-loop "
                "FM's training horizon differs). closed_multistep vs closed_1step = the "
                "one-step-accuracy cost that justifies TWO organs. open_a1 = random floor.",
        "elapsed_seconds": time.time() - started,
    }

    tag = f"v{v}_s{s}_L{depth}_m{m}_trainc{train_corrupt}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_twofm/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {output_dir}/results.json  ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    twofm.remote(m=m, quick=quick)
