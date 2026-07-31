"""E1 -- the allocation ladder: does the value system systematically devalue distractors?

The direct port of `mjc/on_policy/directed_on_policy/` (E3, PR #4). Inner loop = reward-free
block-FM re-adaptation under continuous drift + ballistic (open-loop) control. Outer loop =
a metered budget allocated over channels by `value = lprog x visits`, both taps read off the
SAME forward model the inner loop keeps calibrated. Six E3 rungs plus one:

  uniform            the floor
  error_only         chases raw FM error       -> should be trapped by the NOISE channels
  visits_only        chases relevance only
  oracle             privileged: all budget to the tree

  ...and the `e` tap in two estimators, run side by side so the ladder measures the
  ESTIMATOR as a controlled variable (both are computed every round for every arm, so the
  monitor charge is identical and only the allocation rule differs):

  lprog_only         counterfactual-fit LP alone
  value              lprog x visits            -> E3's rung, as first run
  value_satiety      + a depletable satiety state on the LP drive
  reducible_only     FLOOR-CORRECTED reducibility alone -- `(err - aleatoric floor) / err`,
                     with the floor MEASURED by re-executing the same command from the same
                     state (metered, label-free)
  value_red          reducible x visits        -> the repaired rung
  value_red_satiety  + satiety, thresholded on the irreducible channels' own apparent
                     reducibility -> §10/§12's prediction that the irreducible-channel leak
                     falls, finally testable on a drive that is not inverted

WHY TWO ESTIMATORS
------------------
The first run of this ladder had `value` LOSE to `visits_only`, for a reason that is about
the estimator rather than about relevance. A fixed-budget counterfactual fit measures
MARGINAL RETURN, which early in training is dominated by how STARVED a channel is, not by how
REDUCIBLE it is: block-proportional warm-up leaves the 1-block noise channels furthest back on
their learning curve, so probing them yields the biggest held-out drop. Measured mean LP came
out exactly inverted -- tree +0.044 < structA +0.053 < structB +0.057 < noise +0.064/+0.066 --
so the noisy-TV *filter* had become the noisy-TV *attractor*.

Floor-corrected reducibility cannot be fooled that way, because it never asks where the
channel sits on its learning curve. On the same run it orders them correctly (tree 0.709 ~
structA 0.712 > structB 0.500 > noise 0.454/0.444).

And the loss was VARIANCE more than bias: `lprog x visits` implied a 61% tree share on its
round-averaged taps but realised 44%, with a per-round tree-share sd of 0.309 against
`visits_only`'s 0.059. So both taps are now EMA-smoothed and read off FROZEN probe states.

GRADERS, TWO OF DIFFERENT TYPE (the arc's own discipline)
---------------------------------------------------------
  sighted     tree-channel FM error on a FROZEN probe set of states. E3's `regA err`.
  behavioural open-loop (ballistic) latent beam success, graded by possible-set on the tree.
              Open-loop and not re-grounded ON PURPOSE: a re-grounded beam lets search
              substitute for the forecast and erases FM quality (this substrate's own
              measurement: closed-loop w64 arity-2/arity-1 = 0.672/0.672, erased; open-loop
              0.439/0.178). The closed-loop beam is run anyway, as the near-blind control
              E3's near-saturated reactive control turned out to be.

THE INTUITION UNDER TEST
------------------------
"the value system ought to systematically devalue distractors compared to real tokens." That
is measured directly, not only through the ladder: `forecast_visits` returns the mean |ΔV|
the learned MC value assigns to acting on each channel, and `ground_truth_relevance` returns
the mean reduction in the exact DP `d*` from acting on each channel, which is 0 for every
non-tree channel by construction (P1). Agreement between the two is the claim.

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/ladder.py::ladder --quick
  modal run --detach rhm/directed_sculpting/ladder.py::ladder --tag l5_v1
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import attach_drift, make_layout
from rhm.rhm_repair_cost import Meter
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    POLICIES, advance_drift, allocate, block_latent_mean, blocks_of_channel,
    build_block_tables, build_channel_local_fm,
    closed_loop_beam, ema, make_spec, measure_floors, prewarm_drift, reducible_fraction,
    collect_value_buffer, counterfactual_lp, forecast_visits, ground_truth_relevance,
    fm_one_step_check, make_tree_probe, open_loop_beam, per_channel_error,
    refresh_w_blk, sample_states,
    train_block_fm, train_generator_channels, transition_targets, tree_depth_probe)


app = modal.App("rhm-ds-ladder", image=image)


def calibrate_drift(layout, kappa, target_kl_per_node, seed):
    """Drift every REDUCIBLE channel at its own surface level; return the calibrated sigmas.

    All of them, not just the tree: `verify_distractors` P4 showed a static geometry goes
    inert (every channel's LP below the noise floor by step 7500 of 20000), so allocation
    stops being a live question for two thirds of a run. E3 put ALL its reducible regions on
    a continuous OU walk for exactly this reason. Surface level because that is where the
    homeostatic climbing payoff is largest (§6) -- and the distractor construction is what
    lets relevance stay live there (relevance keys on tree-vs-distractor, not on drift depth).

    MATCHED PER NODE, NOT PER SEQUENCE. `calibrate_sigma` equalises KL per SEQUENCE, which is
    the right currency for the level sweep (a deep cell fires once, a surface cell s^ℓ times).
    Across CHANNELS it is the wrong one: the tree has 16 surface nodes and each struct channel
    has 4, so matching per sequence makes every struct cell drift 4x harder than every tree
    cell -- and the FM's learning progress is a per-BLOCK quantity, so the ladder would have
    been reading per-cell drift magnitude while calling it relevance. Multiplying the target
    by s^ℓ matches the per-node magnitude instead, which is what E3 had for free (its regions
    were the same size and each carried its own OU walk).

    Calibration is a 60-step bisection over 2048 stationary draws per channel, so it is done
    ONCE and the sigmas are replayed per arm by `reset_drift` -- otherwise seven policies
    would each pay for the same deterministic number.
    """
    spec = {}
    for ch in layout["channels"]:
        if ch["kind"] == "noise":
            continue
        ell = ch["depth"] - 1
        n_nodes = layout["s"] ** ell
        attach_drift(layout, ch["name"], [ell], kappa, target_kl_per_node * n_nodes,
                     seed=seed + ch["blk0"])
        spec[ch["name"]] = {"level": ell, "sigma": ch["drift_sigma"], "n_nodes": n_nodes,
                            "kl_per_seq": target_kl_per_node * n_nodes,
                            "kappa": kappa, "seed": seed + ch["blk0"]}
    return spec


def reset_drift(layout, spec):
    """Re-initialise every drifting channel's OU state at uniform, reusing cached sigmas.
    Every arm therefore walks a BIT-IDENTICAL world sequence given the same stepping rng."""
    from rhm.rhm_drift import make_drift_state
    for ch in layout["channels"]:
        if ch["name"] not in spec:
            ch["drift"] = None
            continue
        rec = spec[ch["name"]]
        ch["drift"] = make_drift_state(ch["rules"], [rec["level"]], None, seed=rec["seed"])
        ch["drift_sigma"], ch["drift_kappa"] = rec["sigma"], rec["kappa"]


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=28800, memory=32768)
def ladder(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
    noise_blocks: str = "1,1", struct_shares: str = "0,0",
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 12_000, batch_size: int = 256,
    n_corrupt: int = 3, edit_budget: int = 6, explore_eps: float = 0.3,
    rounds: int = 12, collect_budget: int = 8192, mon_n: int = 1280, forecast_n: int = 512,
    floor_n: int = 512, floor_draws: int = 3, tap_ema: float = 0.5,
    fm_epochs: int = 8, lp_steps: int = 40, drift_kappa: float = 0.05,
    drift_kl: float = 0.30, drift_steps_per_round: int = 20, drift_prewarm: int = 200,
    n_eval: int = 512, n_probe_states: int = 1024, grade_every: int = 3,
    ballistic_chunk: int = 3,
    beta_sat: float = 1.0, alloc_eps: float = 0.01,
    policies: str = ",".join(POLICIES), fm_arch: str = "block",
    tag: str = "v1", quick: bool = False,
):
    """The six-policy E3 ladder plus a satiating rung, on the multi-channel sculpting task."""
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = fm_warm_steps = 600
        value_episodes, rounds = 6_000, 3
        collect_budget, mon_n, forecast_n, floor_n = 1024, 256, 128, 128
        n_eval, n_probe_states, grade_every = 128, 256, 2
        lp_steps = 10

    policy_list = [p for p in policies.split(",") if p]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")],
                     struct_shares=[int(x) for x in struct_shares.split(",")])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    names = tb["channel_names"]
    T, n_blocks = layout["total_len"], tb["n_blocks"]
    tree_c = tb["tree_channel"]
    # channels that share rule tables with the tree -- the transfer-aware oracle's target set
    shared_channels = [i for i, ch in enumerate(layout["channels"]) if ch.get("share_top")]
    print(f"Ladder on the distractor DGP: T={T} tokens / {n_blocks} blocks over "
          f"{len(names)} channels; budget={edit_budget}, c={n_corrupt}, device={device}")
    for ch in layout["channels"]:
        print(f"  {ch['name']:9s} {ch['kind']:6s} blocks[{ch['blk0']:2d}:{ch['blk1']:2d}]"
              + (f" depth={ch['depth']} m={ch['m']}" if ch["rules"] is not None else "")
              + (f" share_top={ch['share_top']}<-{ch['share_from']}" if ch.get("share_top") else ""))

    render, gen = "mixture", torch.Generator(device=device).manual_seed(seed)

    # ---- shared frozen instruments (belief / generator / value) --------------------------
    from rhm.rhm_channels import sample_pool
    pool = sample_pool(layout, 100_000 if not quick else 20_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))

    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, BlockFM = _build_value_head(), _build_block_fm()

    controller = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, train_leaves, tb, batch_size=batch_size,
                             n_steps=generator_steps, lr=3e-4, device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)

    print(f"Collecting value data ({value_episodes} rollouts over all {n_blocks} blocks)")
    cfg, rts, suc = collect_value_buffer(
        controller, generator, layout, tb, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render=render, gen=gen, device=device)
    print(f"  buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    value = ValueHead(state_dim, v).to(device)
    _train_value_mc(value, controller, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    value.eval()
    for p in value.parameters():
        p.requires_grad_(False)
    del cfg, rts, suc

    # ---- frozen evaluation material (identical for every policy and round) ---------------
    x_probe, _ = sample_states(layout, tb, generator, n=n_probe_states, n_corrupt=n_corrupt,
                               presteps=2, seed=seed + 41, device=device, render=render, gen=gen)
    x_eval, r_eval = sample_states(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 42, device=device, render=render,
                                   gen=gen)
    probe_leaves, level_feats, _ = make_tree_probe(layout, tb, seed + 43)
    # FROZEN monitor and floor probes. The taps are read every round, so resampling their
    # states injects state-draw variance into a signal that is already a noisy derivative --
    # and it was that variance, not bias, that cost `value` the first ladder (implied tree
    # share 61%, realised 44%, sd 0.309). The transitions still change every round because
    # the world drifts and the render is stochastic; only the start states are held.
    x_mon0, _ = sample_states(layout, tb, generator, n=mon_n, n_corrupt=n_corrupt,
                              presteps=edit_budget // 2, seed=seed + 45, device=device,
                              render=render, gen=gen)
    x_floor0, _ = sample_states(layout, tb, generator, n=floor_n, n_corrupt=n_corrupt,
                                presteps=edit_budget // 2, seed=seed + 46, device=device,
                                render=render, gen=gen)

    # ---- warm-start FM on a uniform budget (so LP is measurable at round 1) --------------
    # every policy forks from this SAME warm state, so the ladder measures allocation only
    if fm_arch == "channel_local":
        # the position-invariant readout: no absolute-block parameters, attention masked to
        # within-channel, and the encoder's per-block latent offset subtracted (label-free)
        BlockFM = build_channel_local_fm()
    elif fm_arch != "block":
        raise ValueError(f"fm_arch must be 'block' or 'channel_local' (got {fm_arch!r})")
    warm_fm = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    if fm_arch == "channel_local":
        warm_fm.configure(tb).set_center(block_latent_mean(
            layout=layout, tb=tb, controller=controller,
            n=4096 if quick else 20_000, seed=seed + 61, device=device))
    train_block_fm(warm_fm, controller, generator, layout, tb, n_steps=fm_warm_steps,
                   batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3,
                   seed=seed + 51, render=render, gen=gen, device=device)

    drift_spec = calibrate_drift(layout, drift_kappa, drift_kl, seed + 601)
    print(f"Calibrated drift (each channel's surface level, matched KL/NODE = {drift_kl}): "
          + ", ".join(f"{n}@L{r['level']} nodes={r['n_nodes']} "
                      f"KL/seq={r['kl_per_seq']:.2f} σ={r['sigma'][r['level']]:.3f}"
                      for n, r in drift_spec.items()))

    rel = ground_truth_relevance(layout, tb, generator, x_eval, r_eval,
                                 render=render, gen=gen, device=device)
    print(f"\nGround-truth relevance (d* mean {rel['dstar_mean']:.2f}; 0 off-tree by P1):")
    print("  mean Δd*: " + "  ".join(f"{n}={rel['mean_dstar_gain'][i]:+.3f}"
                                     for i, n in enumerate(names)))
    print("  best Δd*: " + "  ".join(f"{n}={rel['best_dstar_gain'][i]:+.3f}"
                                     for i, n in enumerate(names)))

    from rhm.rhm_channels import possible_set_success
    start_succ = float(possible_set_success(layout, x_eval.cpu().numpy(),
                                            r_eval.cpu().numpy()).mean())
    fm0 = fm_one_step_check(warm_fm, value, controller, generator, tb, x_eval, r_eval,
                            render=render, gen=gen, device=device)
    print(f"Task: start possible-set success {start_succ:.3f}; warm FM delta_cos="
          f"{fm0['delta_cos']:.3f} top1={fm0['value_top1_agree']:.3f} "
          f"rank_corr={fm0['value_rank_corr']:.3f}")

    depth0 = tree_depth_probe(controller, probe_leaves, level_feats, layout, tb, device)
    print("Belief depth (frozen throughout this experiment): "
          + " ".join(f"{k}={val:.3f}" for k, val in depth0.items()))

    # ---- the ladder ---------------------------------------------------------------------
    results = {}
    for policy in policy_list:
        print(f"\n{'#' * 72}\n# POLICY: {policy}\n{'#' * 72}")
        results[policy] = _run_policy(
            policy, warm_fm, controller, generator, value, layout, tb, drift_spec,
            x_probe=x_probe, x_eval=x_eval, r_eval=r_eval, seed=seed, rounds=rounds,
            x_mon0=x_mon0, x_floor0=x_floor0, floor_draws=floor_draws, tap_ema=tap_ema,
            collect_budget=collect_budget, mon_n=mon_n, forecast_n=forecast_n,
            fm_epochs=fm_epochs, lp_steps=lp_steps, batch_size=batch_size,
            n_corrupt=n_corrupt, edit_budget=edit_budget,
            drift_steps_per_round=drift_steps_per_round, drift_prewarm=drift_prewarm,
            grade_every=grade_every, beta_sat=beta_sat, alloc_eps=alloc_eps,
            ballistic_chunk=ballistic_chunk, shared_channels=shared_channels,
            render=render, device=device)

    # ---- summary ------------------------------------------------------------------------
    print(f"\n{'=' * 78}\n=== SUMMARY -- E3's ladder on RHM (mean over rounds) ===\n{'=' * 78}")
    print(f"{'policy':18s} {'tree FM err ↓':>14s} {'ball mean ↑':>12s} {'reactive':>9s} "
          f"{'→noise':>8s} {'→struct':>8s} {'→tree':>7s} {'tree sd':>8s} {'mon:col':>8s}")
    for policy in policy_list:
        r = results[policy]
        print(f"{policy:18s} {r['tree_err_mean']:>14.4f} {r['ballistic_mean']:>12.3f} "
              f"{r['reactive_final']:>9.3f} {r['share_noise']:>8.1%} "
              f"{r['share_struct']:>8.1%} {r['share_tree']:>7.1%} "
              f"{r['tree_share_sd']:>8.3f} {r['meter_ratio']:>8.2f}x")
    ref = results[policy_list[0]]["rounds"]
    print("\n  the two `e`-tap estimators, averaged over rounds (identical across arms):")
    for key in ("lprog", "reducible"):
        print(f"    {key:10s} " + "  ".join(
            f"{n}={np.mean([x[key][n] for x in ref]):+.4f}" for n in ref[0][key]))

    metrics = {
        "config": {"v": v, "s": s, "tree_depth": tree_depth, "total_len": T,
                   "n_blocks": n_blocks, "rounds": rounds, "ballistic_chunk": ballistic_chunk, "collect_budget": collect_budget,
                   "mon_n": mon_n, "forecast_n": forecast_n, "edit_budget": edit_budget,
                   "n_corrupt": n_corrupt, "drift_kl_per_node": drift_kl,
                   "drift_kappa": drift_kappa, "drift_steps_per_round": drift_steps_per_round,
                   "render": render, "state_dim": state_dim, "policies": policy_list,
                   "beta_sat": beta_sat, "alloc_eps": alloc_eps, "seed": seed,
                   "struct_shares": struct_shares, "shared_channels": shared_channels,
                   "fm_arch": fm_arch},
        "channels": [{k: ch[k] for k in ("name", "kind", "depth", "m", "blk0", "blk1",
                                         "share_top", "share_from")}
                     for ch in layout["channels"]],
        "ground_truth_relevance": {
            "mean_dstar_gain": {names[i]: float(rel["mean_dstar_gain"][i]) for i in range(len(names))},
            "best_dstar_gain": {names[i]: float(rel["best_dstar_gain"][i]) for i in range(len(names))},
            "dstar_mean": rel["dstar_mean"]},
        "start_success": start_succ,
        "warm_fm_check": fm0,
        "drift_spec": {n: {"level": r["level"], "n_nodes": r["n_nodes"],
                           "kl_per_seq": r["kl_per_seq"]} for n, r in drift_spec.items()},
        "belief_depth_frozen": depth0,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/ladder_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


def _run_policy(policy, warm_fm, controller, generator, value, layout, tb, drift_spec, *,
                x_probe, x_eval, r_eval, x_mon0, x_floor0, floor_draws, tap_ema, seed, rounds,
                collect_budget, mon_n, forecast_n, fm_epochs, lp_steps, batch_size,
                n_corrupt, edit_budget,
                drift_steps_per_round, drift_prewarm, grade_every, beta_sat, alloc_eps,
                ballistic_chunk, shared_channels=None,
                render=None, device=None):
    """One policy's trajectory. Every policy re-initialises the drift state from the SAME
    seed and steps it with the same RNG, so all arms see a bit-identical world sequence --
    E3's matched-drift hygiene, obtained here exactly rather than in expectation."""
    import torch
    import torch.nn.functional as F

    fm = copy.deepcopy(warm_fm)
    n_ch = tb["n_channels"]
    names = tb["channel_names"]
    tree_c = tb["tree_channel"]
    meter = Meter()
    gen = torch.Generator(device=device).manual_seed(seed + 777)

    reset_drift(layout, drift_spec)
    refresh_w_blk(tb, layout, device)
    drift_rng = np.random.default_rng(seed + 602)
    alloc_rng = np.random.default_rng(seed + 603)
    # to stationarity before round 1: an OU walk started at uniform needs ~1/kappa steps to
    # relax, so without this the early rounds run at a fraction of the calibrated magnitude
    if drift_prewarm:
        prewarm_drift(layout, drift_rng, drift_prewarm)
        refresh_w_blk(tb, layout, device)
    satiety = np.zeros(n_ch)
    lp_s = red_s = None
    rows, lp_hist = [], []

    for rnd in range(1, rounds + 1):
        # --- the world moves ---------------------------------------------------------
        kls = advance_drift(layout, drift_rng, drift_steps_per_round)
        refresh_w_blk(tb, layout, device)

        # --- e tap, BOTH estimators (metered identically for every policy, so the ladder
        # --- compares allocation rules and not monitoring budgets). Budget split is sized to
        # --- hold E3's anti-subsidy ratio: 5*mon_n LP + 5*floor_n*floor_draws floor + the
        # --- forecast, against collect_budget -> ~1.78x vs E3's 1.84x. -----------------
        lp_info = counterfactual_lp(fm, controller, generator, tb, x_mon0, lp_steps=lp_steps,
                                    batch_size=batch_size, lr=1e-3, meter=meter,
                                    render=render, gen=gen, device=device)
        errors = {c: lp_info[c]["e_before"] for c in range(n_ch)}
        lp_raw = {c: lp_info[c]["lp"] for c in range(n_ch)}
        floors = measure_floors(controller, generator, tb, x_floor0, meter=meter,
                                n_draws=floor_draws, render=render, gen=gen, device=device,
                                seed=seed + 47)
        red_raw = reducible_fraction(errors, floors, n_ch)
        lp_s = ema(lp_s if rnd > 1 else None, lp_raw, tap_ema, n_ch)
        red_s = ema(red_s if rnd > 1 else None, red_raw, tap_ema, n_ch)
        lprog, reducible = lp_s, red_s
        lp_hist.append([lprog[c] for c in range(n_ch)])

        # LP floor calibrated off the IRREDUCIBLE channels, exactly as `verify_distractors`
        # calibrates its LP floor off them: they cannot learn, so their LP is the built-in
        # null for "no learning progress here". Hardcoding a constant instead is what let a
        # first pass credit the noise channels with progress.
        noise_ch = [c for c, ch in enumerate(layout["channels"]) if ch["kind"] == "noise"]
        lp_floor = float(max(abs(lprog[c]) for c in noise_ch)) if noise_ch else 0.0
        # the irreducible channels cannot be reduced, so whatever reducibility they APPEAR to
        # show is what "no real headroom" reads as on this instrument -- the self-calibrating
        # null `verify_distractors` used for its LP floor, in the reducibility currency.
        red_floor = float(max(reducible[c] for c in noise_ch)) if noise_ch else 0.0

        # --- p tap: forecast visitation by rolling the FM (metered) ------------------
        x_fc, r_fc = sample_states(layout, tb, generator, n=forecast_n, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 9500 + rnd * 7, device=device,
                                   render=render, gen=gen)
        meter.charge_monitor(forecast_n)
        visits, dvalue = forecast_visits(fm, value, controller, tb, x_fc, r_fc,
                                         budget=edit_budget, device=device)

        # --- allocate ----------------------------------------------------------------
        w, drive = allocate(policy, errors=errors, lprog=lprog, visits=visits,
                            tree_channel=tree_c, n_channels=n_ch, lp_floor=lp_floor,
                            reducible=reducible, red_floor=red_floor,
                            satiety=satiety, beta_sat=beta_sat, eps=alloc_eps,
                            shared_channels=shared_channels)
        sat_prev = satiety.copy()
        satiety = 0.7 * satiety + w        # depletable state: fills with spending, decays if not

        # --- collect + fit the FM (metered) ------------------------------------------
        x_col, _ = sample_states(layout, tb, generator, n=collect_budget, n_corrupt=n_corrupt,
                                 presteps=edit_budget // 2, seed=seed + 10_000 + rnd * 7,
                                 device=device, render=render, gen=gen)
        # stochastic apportionment (draw each transition's channel from w) rather than
        # rounding: rounding+truncating silently biases against whichever channel sits last
        # in the layout, which here is a noise channel -- i.e. it would fake part of the
        # result the noisy-TV rung is supposed to measure.
        chan_pick = torch.from_numpy(
            alloc_rng.choice(n_ch, size=collect_budget, p=w)).to(device)
        k_col = torch.empty(collect_budget, dtype=torch.long, device=device)
        for c in range(n_ch):
            sel = chan_pick == c
            blks = blocks_of_channel(tb, c)
            k_col[sel] = blks[torch.randint(0, len(blks), (int(sel.sum()),), device=device)]
        z_col, t_col = transition_targets(controller, generator, x_col, k_col, tb,
                                          render=render, gen=gen)
        meter.charge_collect(collect_budget)

        fm.train()
        opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=1e-4)
        n_steps = max(1, fm_epochs * collect_budget // batch_size)
        for _ in range(n_steps):
            idx = torch.randint(0, collect_budget, (batch_size,), device=device)
            loss = F.mse_loss(fm(z_col[idx], k_col[idx]), t_col[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()
        fm.eval()
        del z_col, t_col

        # --- grade (privileged, unmetered) -------------------------------------------
        errs = per_channel_error(fm, controller, generator, tb, x_probe, render=render,
                                 gen=gen, seed=seed + 88)
        # acted-block error is the headline: the all-blocks number is diluted by the many
        # untouched blocks whose contextual ripple no FM predicts (see `fm_error`).
        tree_err = errs[tree_c]["nmse_acted"]
        row = {"round": rnd, "kl_per_channel": kls, "lp_floor": lp_floor,
               "alloc": {names[c]: float(w[c]) for c in range(n_ch)},
               "drive": {names[c]: float(drive[c]) for c in range(n_ch)},
               "lprog": {names[c]: float(lprog[c]) for c in range(n_ch)},
               "lprog_raw": {names[c]: float(lp_raw[c]) for c in range(n_ch)},
               "reducible": {names[c]: float(reducible[c]) for c in range(n_ch)},
               "reducible_raw": {names[c]: float(red_raw[c]) for c in range(n_ch)},
               "floor": {names[c]: float(floors[c]) for c in range(n_ch)},
               "red_floor": red_floor,
               "error_probe": {names[c]: errs[c]["nmse_acted"] for c in range(n_ch)},
               "error_probe_all_blocks": {names[c]: errs[c]["nmse"] for c in range(n_ch)},
               "error_raw": {names[c]: errs[c]["raw_mse"] for c in range(n_ch)},
               "visits": {names[c]: float(visits[c]) for c in range(n_ch)},
               "satiety": {names[c]: float(sat_prev[c]) for c in range(n_ch)},
               "value_sensitivity": {names[c]: float(dvalue[c]) for c in range(n_ch)},
               "tree_fm_err": tree_err,
               "meter": {"collect": meter.collect, "monitor": meter.monitor,
                         "ratio": meter.ratio}}
        if rnd % grade_every == 0 or rnd == rounds:
            row["ballistic_w1"] = open_loop_beam(controller, generator, fm, value, tb, layout,
                                                 x_eval, r_eval, budget=edit_budget,
                                                 beam_width=1, render=render, gen=gen,
                                                 device=device, chunk_len=ballistic_chunk)
            row["ballistic_w16"] = open_loop_beam(controller, generator, fm, value, tb, layout,
                                                  x_eval, r_eval, budget=edit_budget,
                                                  beam_width=16, render=render, gen=gen,
                                                  device=device, chunk_len=ballistic_chunk)
            row["reactive_w16"] = closed_loop_beam(controller, generator, fm, value, tb, layout,
                                                   x_eval, r_eval, budget=edit_budget,
                                                   beam_width=16, render=render, gen=gen,
                                                   device=device)
        rows.append(row)
        print(f"  r{rnd:02d} alloc " + " ".join(f"{names[c][:6]}={w[c]:.2f}" for c in range(n_ch))
              + f" | tree_err={tree_err:.4f}"
              + (f" ballistic={row.get('ballistic_w16'):.3f}" if "ballistic_w16" in row else "")
              + f" | mon:col={meter.ratio:.2f}x")

    graded = [r for r in rows if "ballistic_w16" in r]
    share = np.mean([[r["alloc"][n] for n in names] for r in rows], axis=0)
    kinds = [ch["kind"] for ch in layout["channels"]]
    return {
        "rounds": rows,
        "tree_err_mean": float(np.mean([r["tree_fm_err"] for r in rows])),
        "tree_err_final": float(rows[-1]["tree_fm_err"]),
        "ballistic_final": float(graded[-1]["ballistic_w16"]) if graded else float("nan"),
        "ballistic_mean": float(np.mean([r["ballistic_w16"] for r in graded])) if graded else float("nan"),
        "ballistic_greedy_final": float(graded[-1]["ballistic_w1"]) if graded else float("nan"),
        "reactive_final": float(graded[-1]["reactive_w16"]) if graded else float("nan"),
        "share_by_channel": {names[c]: float(share[c]) for c in range(len(names))},
        "share_noise": float(sum(share[c] for c in range(len(names)) if kinds[c] == "noise")),
        "share_struct": float(sum(share[c] for c in range(len(names)) if kinds[c] == "struct")),
        "share_tree": float(sum(share[c] for c in range(len(names)) if kinds[c] == "tree")),
        "meter_ratio": float(rows[-1]["meter"]["ratio"]),
        "tree_share_sd": float(np.std([r["alloc"][names[tree_c]] for r in rows])),
        "lp_history": lp_hist,
    }


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1"):
    ladder.remote(quick=quick, tag=tag)
