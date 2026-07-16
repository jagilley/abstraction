"""RHM Sculpting — LENGTH-GENERALIZATION x forward-model ARITY x CLOSED-vs-OPEN loop.

Motivation (see RHM_SCULPTING_README + the human-vision / FoveAgent discussion).
Madan/Ebrahimi/Memisevic (ECCV 2026) show recurrence + strictly-local perception
length-generalizes on visual STATE-TRACKING while global shortcuts break OOD (an
inductive-bias story). Our stronger, domain-general claim: the load-bearing variable
is the extra DEGREE OF FREEDOM — a forward self-model of a CONTROLLED system must take
the command as a 2nd input (arity-2, f(s,u)); a command-blind arity-1 f(s) can only
predict the command-AVERAGED next state ("received wisdom"), and no capacity buys the
missing slot. Adding the command turns an observational map (rung 1) into an
interventional one (rung 2).

Task (RHM_SCULPTING Stage 3b): edit corrupted RHM leaves toward target root r*, via
generator (on-manifold) moves scored by a learned MC value, in a beam that plans IN
LATENTS with a cerebellar block-FM. Trained ONCE at the short range (c<=train_corrupt),
frozen.

Arc so far:
  - Coordination (beam WIDTH) is the length-gen variable (myopic breaks OOD, wide holds).
  - ARITY is load-bearing only at GREEDY / in-distribution and washes out at wide beam:
    a random on-manifold regeneration still makes progress, so wide-beam + value-selection
    compensates for a move-blind FM.
  - Tightening the budget did NOT recover arity at wide beam (budget-slack sweep). The
    neutralizer is structural: the beam RE-GROUNDS each step (execute the move, re-encode
    the TRUE state) and selects on true states — so the forecast is redundant with cheap
    act-and-observe.

FINALIZER (this file). The forward model is irreplaceable exactly when you must plan
WITHOUT executing — an OPEN-LOOP imagined rollout (roll the FM's own predicted latent
forward, no re-grounding). That is the purest "run counterfactuals off to the side."
A move-blind arity-1 FM produces the SAME imagined trajectory for every move sequence,
so it cannot plan at ANY width. Prediction:
  * CLOSED loop (re-grounded): a2 ~ a1 at wide beam (search substitutes).
  * OPEN loop  (imagined):     a2 >> a1 at EVERY width (search cannot substitute).

One length sweep computes, per c and width: token beam + {arity2,arity1} x {closed,open}.

Run:
  modal run rhm/rhm_sculpt_lengthgen.py::lengthgen --m 2 --quick   # smoke
  modal run --detach rhm/rhm_sculpt_lengthgen.py::lengthgen --m 2
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
from rhm.rhm_generative_planner import _build_generator, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_sculpt_planner import (
    _beam_plan,
    _collect_value_sculpt,
    _corrupt,
    _regenerate_chunked,
    _sample_pool,
)
from rhm.rhm_sculpt_latent import (
    _block_state_chunked,
    _build_block_fm,
    _build_rich_controller,
    _fm_check,
    _fm_chunked,
    _train_block_fm,
)


app = modal.App("rhm-sculpt-lengthgen", image=image)


def _build_block_fm_arity1():
    """Arity-1 (command-blind) cerebellar FM: predicts the per-block latent delta from
    z ALONE, ignoring the acted region k. Identical to the arity-2 BlockLatentFM minus
    the action marker, so under the SAME random-k training targets its MSE optimum is
    the region-AVERAGED update E_k[dz | z] — the RHM analog of reaching's FM_state."""
    import torch
    import torch.nn as nn

    class BlockLatentFMArity1(nn.Module):
        def __init__(self, state_dim, n_blocks, n_head, n_layer):
            super().__init__()
            self.block_position = nn.Embedding(n_blocks, state_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=state_dim, nhead=n_head, dim_feedforward=4 * state_dim,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.norm = nn.LayerNorm(state_dim)
            self.head = nn.Linear(state_dim, state_dim)
            self.register_buffer("positions", torch.arange(n_blocks), persistent=False)

        def forward(self, z_block, k):  # k accepted for API-compat, deliberately ignored
            hidden = z_block + self.block_position(self.positions)[None]
            return self.head(self.norm(self.encoder(hidden)))

    return BlockLatentFMArity1


def _latent_beam(controller, generator, fm, value, leaves0, roots, canon, region_index,
                 n_regions, rules, *, s, budget, beam_width, device, arity1=False,
                 open_loop=False, tiebreak_seed=0):
    """Re-grounded (closed-loop) OR imagined (open-loop) latent beam, with an ARITY flag.

    closed-loop (open_loop=False): Stage-3b behaviour — each step materializes the kept
      tips and re-encodes their TRUE latents; final selection on true states. For
      arity1=False this is byte-for-byte the Stage-3b beam.
    open-loop (open_loop=True): the FM rolls its OWN predicted latent forward, never
      re-grounding (imagined rollout); only the finally-selected move sequence is
      executed on x0 and graded. Search cannot substitute for the forecast (no true-state
      feedback during planning), so a move-blind arity-1 FM — whose imagined trajectory
      is identical across all move sequences — is stuck at the random floor at any width.

    arity1=True: FM ignores k; within-parent scores are equal, a tiny random tiebreak
      makes region selection uniform (the FM_state floor)."""
    import torch
    gen = torch.Generator(device=device)
    gen.manual_seed(tiebreak_seed)
    with torch.no_grad():
        controller.eval(); generator.eval(); fm.eval(); value.eval()
        x0 = leaves0.to(device)
        roots = roots.to(device)
        batch, length = x0.shape
        beams_z = controller.block_state(x0)[:, None]                 # (B,1,n_blocks,D) true init
        beams_x = x0[:, None, :]                                      # closed-loop only
        beams_moves = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)  # open-loop only
        width = 1
        for _ in range(budget):
            n_blocks, dim = beams_z.shape[2], beams_z.shape[3]
            flat_z = beams_z.reshape(batch * width, n_blocks, dim)
            roots_bw = roots.repeat_interleave(width)
            scores = []
            for k in range(n_regions):
                kk = torch.full((batch * width,), k, device=device, dtype=torch.long)
                pred = (flat_z + _fm_chunked(fm, flat_z, kk)).mean(dim=1)
                scores.append(value(pred, roots_bw))
            scores = torch.stack(scores, dim=1).reshape(batch, width * n_regions)
            if arity1:
                scores = scores + 1e-4 * torch.rand(scores.shape, generator=gen, device=device)
            keep = min(beam_width, width * n_regions)
            _, top_idx = scores.topk(keep, dim=1)
            parent = torch.div(top_idx, n_regions, rounding_mode="floor")
            move = top_idx % n_regions
            if open_loop:
                # roll the FM's PREDICTED latent for kept (parent, move) pairs; never re-ground
                parent_z = beams_z.gather(1, parent[:, :, None, None].expand(-1, -1, n_blocks, dim))
                flat_pz = parent_z.reshape(batch * keep, n_blocks, dim)
                move_flat = move.reshape(batch * keep)
                beams_z = (flat_pz + _fm_chunked(fm, flat_pz, move_flat)).reshape(batch, keep, n_blocks, dim)
                parent_moves = beams_moves.gather(1, parent[:, :, None].expand(-1, -1, beams_moves.shape[2]))
                beams_moves = torch.cat([parent_moves, move[:, :, None]], dim=2)
            else:
                parent_x = beams_x.gather(1, parent[:, :, None].expand(-1, -1, length))
                flat_x = parent_x.reshape(batch * keep, length)
                new_x = _regenerate_chunked(generator, flat_x, region_index[move.reshape(-1)], canon, block_size=s)
                beams_x = new_x.reshape(batch, keep, length)
                beams_z = _block_state_chunked(controller, new_x).reshape(batch, keep, n_blocks, dim)
            width = keep
        final = value(beams_z.mean(dim=2).reshape(-1, beams_z.shape[3]),
                      roots.repeat_interleave(width)).reshape(batch, width)
        best = final.argmax(dim=1)
        if open_loop:
            best_moves = beams_moves[torch.arange(batch, device=device), best]  # (B, budget)
            x = x0.clone()
            for t in range(best_moves.shape[1]):
                x = _regenerate_chunked(generator, x, region_index[best_moves[:, t]], canon, block_size=s)
            x_final = x
        else:
            x_final = beams_x[torch.arange(batch, device=device), best]
        success = parse_success_and_heuristic(rules, x_final.cpu().numpy(), roots.cpu().numpy(), s)[0]
    return float(success.mean())


def _eval_at(controller, generator, block_fm_a2, block_fm_a1, value, leaves0, eval_roots,
             canon, region_index, n_regions, rules, *, s, budget, widths, device, tiebreak_base):
    """token beam + {arity2,arity1} x {closed,open} latent beams at every width."""
    fm_a2 = _fm_check(block_fm_a2, value, controller, generator, leaves0, eval_roots,
                      canon, region_index, n_regions, s, device)
    fm_a1 = _fm_check(block_fm_a1, value, controller, generator, leaves0, eval_roots,
                      canon, region_index, n_regions, s, device)
    out = {"budget": budget, "fm_check_arity2": fm_a2, "fm_check_arity1": fm_a1,
           "token_beam": {}, "a2_closed": {}, "a1_closed": {}, "a2_open": {}, "a1_open": {}}

    def lb(fm, width, arity1, open_loop, tb):
        return _latent_beam(controller, generator, fm, value, leaves0, eval_roots, canon,
                            region_index, n_regions, rules, s=s, budget=budget, beam_width=width,
                            device=device, arity1=arity1, open_loop=open_loop, tiebreak_seed=tb)

    for width in widths:
        w = str(width)
        out["token_beam"][w] = _beam_plan(controller, generator, value, leaves0, eval_roots, canon,
                                          region_index, n_regions, rules, s=s, budget=budget,
                                          beam_width=width, device=device)
        out["a2_closed"][w] = lb(block_fm_a2, width, False, False, 0)
        out["a1_closed"][w] = lb(block_fm_a1, width, True, False, tiebreak_base + width)
        out["a2_open"][w] = lb(block_fm_a2, width, False, True, 0)
        out["a1_open"][w] = lb(block_fm_a1, width, True, True, tiebreak_base + 1000 + width)
    return out


def _print_row(label, res, widths):
    w1, wmax = str(widths[0]), str(widths[-1])
    print(f"  {label} (budget {res['budget']}): "
          f"CLOSED a2/a1 w{w1}={res['a2_closed'][w1]:.3f}/{res['a1_closed'][w1]:.3f} "
          f"w{wmax}={res['a2_closed'][wmax]:.3f}/{res['a1_closed'][wmax]:.3f} || "
          f"OPEN a2/a1 w{w1}={res['a2_open'][w1]:.3f}/{res['a1_open'][w1]:.3f} "
          f"w{wmax}={res['a2_open'][wmax]:.3f}/{res['a1_open'][wmax]:.3f}")


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=16384)
def lengthgen(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, n_eval_episodes: int = 1_024, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256,
    train_corrupt: int = 3, test_corrupts: str = "1,2,3,4,5,6", beam_widths: str = "1,4,16,64",
    budgets: str = "3,4,5,6,8", budget_sweep_c: int = 3,
    region_size: int = 1, explore_eps: float = 0.3, quick: bool = False,
):
    """Train the sculpting apparatus at the short range (c<=train_corrupt), then a
    length sweep computing token + {arity2,arity1} x {closed,open-loop} at every width."""
    import time
    import torch

    if s != 2:
        raise ValueError("This narrow experiment currently assumes binary RHM branching (s=2).")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    c_tests = [int(x) for x in test_corrupts.split(",")]
    budgets_list = [int(x) for x in budgets.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, n_eval_episodes, value_episodes = 20_000, 512, 6_000
        c_tests, widths, budgets_list = [1, 3, 5], [1, 16], [3, 6]
    c_tests = [c for c in c_tests if c < n_blocks]  # _corrupt draws c distinct blocks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"Sculpt-lengthgen (closed vs open loop): v={v}, s={s}, L={depth}, m={m}, blocks={n_blocks}, "
          f"train_corrupt(<= )={train_corrupt}, test_corrupts={c_tests}, widths={widths}, device={device}")
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
    train_budget = train_corrupt * s

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

    print(f"Collecting value data ({value_episodes} rollouts, c<= {train_corrupt}, budget {train_budget})")
    configs, roots_buf, success_buf = _collect_value_sculpt(
        controller, generator, train_roots_np, train_leaves_np, canon, region_index, n_regions,
        rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks, v=v, s=s,
        n_corrupt=train_corrupt, budget=train_budget, epsilon=explore_eps, device=device)
    print(f"  buffer: {configs.shape[0]} states, terminal success rate {success_buf.mean().item():.3f}")
    value = MCValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                    n_steps=value_steps, lr=3e-4, device=device)

    # arity-2 vs arity-1 block FMs, trained on IDENTICAL data (torch reseeded before each) --
    block_fm_a2 = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    block_fm_a1 = BlockLatentFMArity1(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    print(f"Block FM params: arity-2 {_count_parameters(block_fm_a2):,} | "
          f"arity-1 {_count_parameters(block_fm_a1):,}")
    for name, fm in (("arity-2", block_fm_a2), ("arity-1", block_fm_a1)):
        torch.manual_seed(train_seed + 500)  # matched training data across the two FMs
        print(f"Training {name} block FM")
        _train_block_fm(fm, controller, generator, train_roots_np, train_leaves_np, canon,
                        region_index, n_regions, n_steps=fm_steps, batch_size=batch_size,
                        n_blocks=n_blocks, v=v, s=s, n_corrupt=train_corrupt, budget=train_budget,
                        lr=1e-3, device=device)
    for module in (value, block_fm_a2, block_fm_a1):
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

    print(f"\n=== length sweep: CLOSED (re-grounded) vs OPEN (imagined) loop x arity-2 vs arity-1 ===")
    length_sweep = {}
    for c_test in c_tests:
        leaves0 = corrupt_at(c_test)
        res = _eval_at(controller, generator, block_fm_a2, block_fm_a1, value, leaves0, eval_roots,
                       canon, region_index, n_regions, rules, s=s, budget=c_test * s, widths=widths,
                       device=device, tiebreak_base=c_test * 100)
        res["dstar_mean"] = float(nearest_derivation_cost(rules, leaves0.numpy(), eval_roots_np, s).mean())
        length_sweep[str(c_test)] = res
        _print_row(f"c={c_test}", res, widths)

    # budget-slack sweep at fixed c: does tightening the MOVE budget recover arity at wide
    # beam? (greedy + widest beam only, to keep it cheap). Neutralizer test for round 2.
    bs_widths = sorted({widths[0], widths[-1]})
    print(f"\n=== budget-slack sweep at c={budget_sweep_c} (widths {bs_widths}) ===")
    leaves_bs = corrupt_at(budget_sweep_c)
    dstar_bs = float(nearest_derivation_cost(rules, leaves_bs.numpy(), eval_roots_np, s).mean())
    budget_sweep = {"dstar_mean": dstar_bs, "by_budget": {}}
    for budget in budgets_list:
        res = _eval_at(controller, generator, block_fm_a2, block_fm_a1, value, leaves_bs, eval_roots,
                       canon, region_index, n_regions, rules, s=s, budget=budget, widths=bs_widths,
                       device=device, tiebreak_base=budget * 100 + 1)
        budget_sweep["by_budget"][str(budget)] = res
        _print_row(f"budget={budget}", res, bs_widths)

    metrics = {
        "config": {"v": v, "s": s, "depth": depth, "m": m, "n_blocks": n_blocks,
                   "train_corrupt": train_corrupt, "test_corrupts": c_tests, "beam_widths": widths,
                   "budgets": budgets_list, "budget_sweep_c": budget_sweep_c,
                   "region_size": region_size, "state_dim": state_dim, "value_episodes": value_episodes,
                   "rule_seed": rule_seed, "train_seed": train_seed},
        "value_buffer_success_rate": success_buf.mean().item(),
        "length_sweep": length_sweep,
        "budget_sweep": budget_sweep,
        "note": "CLOSED = re-grounded beam (execute+re-encode each step; search substitutes for the DOF); "
                "OPEN = imagined FM rollout (no re-grounding; search CANNOT substitute). Finalizer: "
                "a2_open >> a1_open at ALL widths while a2_closed ~ a1_closed at wide beam.",
        "elapsed_seconds": time.time() - started,
    }

    tag = f"v{v}_s{s}_L{depth}_m{m}_trainc{train_corrupt}_seed{rule_seed}"
    output_dir = f"{DATA_DIR}/rhm_sculpt_lengthgen/{tag}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as handle:
        json.dump(metrics, handle, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {output_dir}/results.json  ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


@app.local_entrypoint()
def main(m: int = 2, quick: bool = False):
    lengthgen.remote(m=m, quick=quick)
