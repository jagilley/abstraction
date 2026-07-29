"""E2 -- the hierarchy climb: does drift make the belief go DEEPER, or just re-fit?

`ideas/adaptive_core_and_hierarchy_climb.md` §6/§11. Drift on a NON-hierarchical task buys an
adaptive core and nothing more -- which is what `mjc/expansion` measured, calibrated, and is
this experiment's control arm rather than a disappointment. Drift on a HIERARCHICAL task can
in principle buy hierarchy-climbing, because higher levels are more invariant under shift and
therefore cheaper to maintain. RHM is the substrate that has a hierarchy to climb and a
per-level probe to read the climb directly.

THE TWO-INSTRUMENT READ (§6), and why one instrument is not enough
-------------------------------------------------------------------
"Fewer rounds to recover" is confoundable with ordinary continued training -- any model gets
faster at fitting as it trains. The discriminator is two instruments of DIFFERENT TYPE:

  repair    how much of the SAME-magnitude damage the SAME budget repairs, per drift event.
  depth     per-level ancestor recovery `d1..dL` on a FROZEN probe (support-fixed drift never
            moves the rule tables, so the labels stay valid -- `verify_backcompat` B5).

  climbing               -> repair improves AND depth rises
  ordinary training      -> repair improves, depth flat
  adaptive core only     -> repair flat, depth flat

THE REPAIR READOUT, AND WHY IT IS A CROSS-ARM DIFFERENCE
---------------------------------------------------------
`rhm_repair_cost` proved the obvious instrument is a null by construction: raw CE is EXACTLY
blind to this drift (the entropy the drift destroys equals the KL it creates), and the
gap-to-a-matched-reference it prescribes instead is itself confounded by ordinary continued
training -- at zero drift the stale->matched gap was +0.0187 nats, LARGER than the +0.0167 an
actual KL=0.68 drift contributed. Its fix is a paired matched-COMPUTE arm and a
difference-in-differences.

Here that arm is not an extra: the `nodrift` condition IS the paired matched-compute arm, run
with bit-identical budgets on undrifted data. So, per round `t`:

    damage(t)   = grader_stale(t)_A - grader(t)_N        measured right after the drift event
    residual(t) = grader_end(t)_A   - grader_end(t)_N     after spending the round's budget
    repaired(t) = (damage - residual) / damage

`repaired(t)` rising with `t` at MATCHED drift magnitude is the homeostatic claim; flat is the
adaptive-core reading.

Two things had to be fixed before "matched magnitude" and "matched control" were true rather
than intended, and BOTH biased toward finding a climb:

  magnitude   `calibrate_sigma` matches KL from UNIFORM at stationarity -- accumulated
              displacement, not the size of an event. The first run of this sweep therefore
              had per-event magnitudes of 0.78 / 1.00 / 3.49 nats across levels, a 4.5x spread
              inside the one comparison meant to hold magnitude fixed.
              `calibrate_sigma_event` targets the event and lands them within 7%.
  control     the OU walk anchors at theta=0, the UNIFORM mixture, which is the simplex's
              MAXIMUM-ENTROPY point. A `nodrift` control parked there sits at maximum synonym
              ambiguity while every drifting arm sits at a typical, lower-entropy, more
              deterministic, easier-to-render draw. `init="stationary"` seats the control at a
              world of matched entropy instead.

THE DOSE-RESPONSE (§6's prediction)
------------------------------------
Sweep WHICH LEVEL drifts at matched KL/sequence. Climbing only pays if the drift is
surface-local and the deep structure is invariant, so the payoff should scale with how deep
the invariance goes: large at `surface`, none at `root`. A monotone curve here is the claim's
own falsifier if it comes out flat.

The belief TRAINS in this experiment (it is frozen in `ladder.py`) -- under the evaluative
grader, because climbing requires the belief to be able to move and §11 argues only
target-SELECTION, not loss reweighting, has ever moved this frontier.

Run from experiments/:
  modal run rhm/directed_sculpting/full_loop/climb.py::climb --quick
  modal run --detach rhm/directed_sculpting/climb.py::climb --tag l4_v1
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_channels import attach_drift, make_layout, sample_pool
from rhm.rhm_drift import make_drift_state, stationary_theta
from rhm.rhm_repair_cost import Meter
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.directed_sculpting.full_loop.channel_env import (
    advance_drift, allocate, belief_update, blocks_of_channel, build_block_tables,
    collect_grounded_moves, ema, measure_floors, prewarm_drift, reducible_fraction,
    repaired_fraction, collect_value_buffer, counterfactual_lp, fm_error, forecast_visits,
    make_spec,
    make_tree_probe, open_loop_beam, per_channel_error, refresh_w_blk, root_ce,
    sample_states, train_block_fm, train_generator_channels, transition_targets,
    tree_depth_probe)


app = modal.App("rhm-ds-climb", image=image)

# level index -> arm name. `rhm_drift` indexes 0 = the root-adjacent rule layer and L-1 = the
# bottom (feature -> leaf) layer, so `surface` is the LAST layer and is the only one whose
# drift also changes the EDIT dynamics (a block is a level-1 node, rendered through that
# layer). That asymmetry is a prediction, not a nuisance: the FM's repair should respond only
# to surface drift while the BELIEF's repair responds at every level.
ARM_LEVELS = {"nodrift": None, "surface": -1, "mid": -3, "root": 0}


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=28800, memory=32768)
def climb(
    v: int = 8, s: int = 2, seed: int = 1, state_dim: int = 96,
    tree_depth: int = 4, struct_depths: str = "2,2", struct_ms: str = "2,4",
    noise_blocks: str = "1,1", edit_budget: int = 6, n_corrupt: int = 3,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    value_episodes: int = 40_000, fm_warm_steps: int = 8_000, batch_size: int = 256,
    explore_eps: float = 0.3, rounds: int = 10, collect_budget: int = 4096,
    mon_n: int = 1024, forecast_n: int = 512, floor_n: int = 256, floor_draws: int = 3,
    tap_ema: float = 0.5, fm_epochs: int = 4, lp_steps: int = 25,
    belief_steps: int = 1200, ground_states: int = 3072, anchor_pool: int = 40_000,
    samples_sweep: str = "4096,1024,256,64",   # scale anchor: 4096 == the pre-fix operating point
    lam_fm: float = 1.0, lam_plan: float = 1.0, tau: float = 1.0,
    drift_kappa: float = 0.05, drift_kl: float = 0.60, drift_steps_per_round: int = 20,
    drift_prewarm: int = 200, entropy_matched_base: bool = True,
    n_eval: int = 512, n_probe_states: int = 1024, n_probe: int = 2000,
    probe_steps: int = 250, grade_every: int = 3, ballistic_chunk: int = 3,
    arms: str = "nodrift,surface", policy: str = "value_red_satiety",
    tag: str = "v1", quick: bool = False,
):
    """Repeated drift events at matched magnitude, read against the per-level depth probe."""
    import torch

    from rhm.rhm_edit_control import _train_edit_controller
    from rhm.rhm_generative_planner import _build_generator
    from rhm.rhm_latent_planner import _build_value_head, _train_value_mc
    from rhm.rhm_sculpt_latent import _build_block_fm, _build_rich_controller

    if quick:
        controller_steps = generator_steps = value_steps = fm_warm_steps = 800
        value_episodes, rounds, belief_steps = 6_000, 3, 200
        collect_budget, mon_n, forecast_n, ground_states = 512, 192, 128, 512
        samples_sweep = "512,64"
        n_eval, n_probe_states, n_probe, probe_steps = 128, 256, 600, 80
        anchor_pool, lp_steps, grade_every = 8_000, 8, 2

    arm_list = [a for a in arms.split(",") if a]
    sweep = [int(x) for x in samples_sweep.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    started = time.time()

    spec = make_spec(tree_depth=tree_depth,
                     struct_depths=[int(x) for x in struct_depths.split(",")],
                     struct_ms=[int(x) for x in struct_ms.split(",")],
                     noise_blocks=[int(x) for x in noise_blocks.split(",")])
    layout = make_layout(v, s, spec)
    tb = build_block_tables(layout, device)
    names, T, n_blocks = tb["channel_names"], layout["total_len"], tb["n_blocks"]
    L, tree_c = tree_depth, tb["tree_channel"]
    render = "mixture"
    gen = torch.Generator(device=device).manual_seed(seed)
    print(f"Climb: L={L}, T={T} tokens / {n_blocks} blocks, budget={edit_budget}, "
          f"arms={arm_list}, rounds={rounds}, device={device}")

    # ---- per-level sigma at MATCHED KL/sequence (the level sweep's hygiene) --------------
    tree_ch = layout["tree"]
    levels = sorted({(L + ARM_LEVELS[a]) % L for a in arm_list if ARM_LEVELS[a] is not None})
    # sigma matched on the size of ONE EVENT. The stationary-KL calibration matches
    # ACCUMULATED displacement from uniform instead, which left this very sweep running at
    # per-event magnitudes of 0.78 / 1.00 / 3.49 nats across levels -- a 4.5x spread inside
    # the one comparison whose entire point is holding magnitude fixed.
    attach_drift(layout, "tree", levels, drift_kappa, drift_kl, seed=seed + 601,
                 calibrate="event", event_steps=drift_steps_per_round,
                 init="stationary" if entropy_matched_base else "uniform", frozen=True)
    sigmas = dict(tree_ch["drift_calibrated_sigma"])
    # W0, shared by every arm including `nodrift`: non-uniform at EVERY swept level, drawn
    # from the walk's own stationary law. Parking `nodrift` at uniform instead would seat the
    # control at the simplex's entropy maximum while each drifting arm sits at a typical
    # (lower-entropy, more deterministic, easier-to-render) draw -- and since that bias favours
    # finding a climb, the previous run's null survives it, but the sweep is only clean here.
    theta0 = (stationary_theta(tree_ch["rules"], levels, drift_kappa, sigmas, seed=seed + 602)
              if entropy_matched_base else None)
    print("Per-EVENT matched sigmas: "
          + "  ".join(f"L{ell}(nodes {s**ell})={sigmas[ell]:.4f}" for ell in levels)
          + f"  | entropy_matched_base={entropy_matched_base}")

    # ---- shared setup -------------------------------------------------------------------
    Controller, Generator = _build_rich_controller(), _build_generator()
    ValueHead, BlockFM = _build_value_head(), _build_block_fm()

    pool = sample_pool(layout, 100_000 if not quick else 20_000, seed)
    train_leaves = torch.from_numpy(pool["leaves"])
    train_roots = torch.from_numpy(pool["roots"].astype(np.int64))

    controller0 = Controller(v, T, s, state_dim, n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller0, train_leaves, train_roots, batch_size=batch_size,
                           n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                           lr=3e-4, device=device, p_full=0.5)
    generator = Generator(v, T, s, state_dim, n_head=4, n_layer=2,
                          root_conditioned=False).to(device)
    train_generator_channels(generator, train_leaves, tb, batch_size=batch_size,
                             n_steps=generator_steps, lr=3e-4, device=device)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    print(f"Collecting value data ({value_episodes} rollouts)")
    cfg, rts, suc = collect_value_buffer(
        controller0, generator, layout, tb, n_episodes=value_episodes, batch_size=1024,
        n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps, seed=seed + 31,
        render=render, gen=gen, device=device)
    value0 = ValueHead(state_dim, v).to(device)
    _train_value_mc(value0, controller0, cfg, rts, suc, batch_size=512, n_steps=value_steps,
                    lr=3e-4, device=device)
    print(f"  value buffer {cfg.shape[0]} states, terminal success {suc.mean().item():.3f}")
    del cfg, rts, suc

    fm0 = BlockFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    train_block_fm(fm0, controller0, generator, layout, tb, n_steps=fm_warm_steps,
                batch_size=batch_size, n_corrupt=n_corrupt, budget=edit_budget,
                lr=1e-3, seed=seed + 51, render=render, gen=gen, device=device)

    # frozen material: same experiment every round, only world and model differ
    x_probe, _ = sample_states(layout, tb, generator, n=n_probe_states, n_corrupt=n_corrupt,
                               presteps=edit_budget // 2, seed=seed + 41, device=device,
                               render=render, gen=gen)
    x_eval, r_eval = sample_states(layout, tb, generator, n=n_eval, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 42, device=device, render=render,
                                   gen=gen)
    probe_leaves, level_feats, _ = make_tree_probe(layout, tb, seed + 43)

    results = {}
    # SAMPLES PER EVENT is the knob, not drift magnitude. Drift makes the deep level cheaper
    # to MAINTAIN; it does not make it NECESSARY. If post-drift surface re-fit fully restores
    # performance, holding the invariant buys nothing measurable and no gradient ever prices
    # the integrated cost -- the agent that never climbed is not paying for the deep level, it
    # just does not need one. Measured on the first sweep: the surface arm repaired 91% of its
    # damage within the round (residual/damage 0.09 over the last 5 rounds) and converged to
    # the nodrift arm's root CE every round. So climbing had no lever, and the null said only
    # that drift ALONE does not induce it.
    #
    # Below the sample count surface re-fit needs, the cheap repair should instead be
    # deep-prior-plus-few-samples -- which is specialization's own line, "a learner that
    # already holds level l recruits l+1 from a small marginal sample". So: hold KL/event
    # fixed (already exact) and sweep samples/event DOWN, expecting a threshold rather than a
    # gradient. Each budget carries its own matched `nodrift` control, so the DiD stays paired.
    for S in sweep:
        scale = S / 4096.0
        # EVERYTHING per-event scales together, monitoring included. Flooring the monitor
        # instead is how a first pass reached a 13.5x monitor:collect ratio at the starved end
        # -- E3's fatal 22x subsidy rebuilt by accident, which would have left the "starved"
        # arms not actually starved and made allocation a non-question exactly where the
        # experiment needs it to bind. The floors below are small enough that the ratio stays
        # bounded; it is reported per arm and does rise at the bottom of the sweep, which is an
        # honest property of metering a very small budget rather than a subsidy.
        cfg = dict(collect_budget=S,
                   ground_states=max(48, int(round(ground_states * scale))),
                   anchor_pool=max(256, int(round(anchor_pool * scale))),
                   belief_steps=max(12, int(round(belief_steps * scale))),
                   mon_n=max(16, int(round(mon_n * scale))),
                   floor_n=max(8, int(round(floor_n * scale))),
                   forecast_n=max(16, int(round(forecast_n * scale))))
        print(f"\n{'=' * 72}\n= SAMPLES/EVENT = {S}   {cfg}\n{'=' * 72}")
        for arm in arm_list:
            key = f"{arm}@{S}"
            print(f"\n{'#' * 72}\n# ARM: {key}\n{'#' * 72}")
            results[key] = _run_arm(
                arm, controller0, fm0, value0, generator, layout, tb, sigmas,
                x_probe=x_probe, x_eval=x_eval, r_eval=r_eval, probe_leaves=probe_leaves,
                level_feats=level_feats, seed=seed, L=L, v=v, s=s, state_dim=state_dim,
                rounds=rounds, fm_epochs=fm_epochs, lp_steps=lp_steps,
                floor_draws=floor_draws, tap_ema=tap_ema,
                batch_size=batch_size, n_corrupt=n_corrupt, edit_budget=edit_budget,
                drift_kappa=drift_kappa, drift_steps_per_round=drift_steps_per_round,
                drift_prewarm=drift_prewarm, levels=levels, theta0=theta0,
                lam_fm=lam_fm, lam_plan=lam_plan, tau=tau, probe_steps=probe_steps,
                grade_every=grade_every, ballistic_chunk=ballistic_chunk, policy=policy,
                render=render, device=device, **cfg)
            results[key]["samples_per_event"] = S
            results[key]["arm"] = arm

    # ---- the cross-arm difference-in-differences ----------------------------------------
    readout = {}
    for S in sweep:
        if f"nodrift@{S}" not in results:
            continue
        ref = results[f"nodrift@{S}"]["rounds"]
        for arm in arm_list:
            if arm == "nodrift":
                continue
            cur = results[f"{arm}@{S}"]["rounds"]
            n = min(len(cur), len(ref))
            # resolution floor from the arm's OWN damage series (half the median), so an
            # event too small to measure is reported out of range instead of dividing to a
            # confident nonsense number.
            floors = {key: 0.5 * float(np.median(
                [abs(cur[t][f"{key}_stale"] - ref[t][f"{key}_stale"]) for t in range(n)]))
                for key in ("tree_fm_err", "root_ce")}
            per_round = []
            for t in range(n):
                rec = {"round": t + 1, "kl_event": cur[t]["drift_kl"],
                       "kl_cumulative": cur[t]["drift_kl_cumulative"]}
                for key in ("tree_fm_err", "root_ce"):
                    dmg = cur[t][f"{key}_stale"] - ref[t][f"{key}_stale"]
                    res = cur[t][f"{key}_end"] - ref[t][f"{key}_end"]
                    rec[f"{key}_damage"] = dmg
                    rec[f"{key}_residual"] = res
                    rec[f"{key}_repaired"] = repaired_fraction(dmg, res, floors[key])
                per_round.append(rec)
            readout[f"{arm}@{S}"] = {"resolution_floors": floors,
                                     "per_round": per_round}

    print(f"\n{'=' * 78}\n=== SUMMARY -- does starving samples/event make the deep level NECESSARY? ==="
          f"\n{'=' * 78}")
    print("The claim under test is that climbing appears only once post-drift SURFACE repair")
    print("stops sufficing. `unrepaired` is the fraction of same-magnitude damage still")
    print("standing at the end of the round: ~0 means invariance is free and nothing prices it.\n")
    print(f"{'S/event':>8s} {'unrepaired':>11s} {'Δd1':>7s} {'Δd2':>7s} {'Δd3':>7s} {'Δd4':>7s} {'ΔPR':>7s}")
    for S in sweep:
        nd = results.get(f"nodrift@{S}")
        for arm in arm_list:
            if arm == "nodrift" or f"{arm}@{S}" not in results:
                continue
            rr, ndr = results[f"{arm}@{S}"]["rounds"], nd["rounds"]
            pr = readout.get(f"{arm}@{S}", {}).get("per_round", [])
            ur = [1.0 - x["root_ce_repaired"] for x in pr[len(pr) // 2:]
                  if np.isfinite(x["root_ce_repaired"])]
            d = {k: (rr[-1]["depth"][k] - rr[0]["depth"][k])
                    - (ndr[-1]["depth"][k] - ndr[0]["depth"][k])
                 for k in ("d1", "d2", "d3", "d4", "PR")}
            print(f"{S:>8d} {(np.mean(ur) if ur else float('nan')):>11.2f} "
                  + " ".join(f"{d[k]:>+7.3f}" for k in ("d1", "d2", "d3", "d4"))
                  + f" {d['PR']:>+7.2f}")
    print("\n  climbing = unrepaired rises AND Δdepth turns positive at the same budget.")
    for S in sweep:
        for arm in arm_list:
            key = f"{arm}@{S}"
            if key not in results:
                continue
            rr = results[key]["rounds"]
            print(f"  [{key:16s}] depth " + "  ".join(
                f"{k}={rr[-1]['depth'][k]:.3f}" for k in ("d1", "d2", "d3", "d4"))
                + f"  PR={rr[-1]['depth']['PR']:.1f}  rootCE={rr[-1]['root_ce_end']:.4f}"
                + f"  meter={rr[-1]['meter']['ratio']:.2f}x")

    metrics = {
        "config": {"v": v, "s": s, "tree_depth": tree_depth, "total_len": T,
                   "n_blocks": n_blocks, "edit_budget": edit_budget, "rounds": rounds,
                   "collect_budget": collect_budget, "belief_steps": belief_steps,
                   "drift_kl_per_seq": drift_kl, "drift_kappa": drift_kappa,
                   "drift_steps_per_round": drift_steps_per_round, "arms": arm_list,
                   "samples_sweep": sweep, "policy_drive": policy,
                   "entropy_matched_base": entropy_matched_base, "calibrate": "event",
                   "policy": policy, "render": render, "seed": seed,
                   "arm_levels": {a: (None if ARM_LEVELS[a] is None else (L + ARM_LEVELS[a]) % L)
                                  for a in arm_list},
                   "matched_sigmas": {str(k): float(x) for k, x in sigmas.items()}},
        "results": results, "did_readout": readout,
        "elapsed_seconds": time.time() - started,
    }
    out_dir = f"{DATA_DIR}/directed_sculpting/climb_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({metrics['elapsed_seconds']:.0f}s)")
    return metrics


def _tree_err(fm, controller, generator, tb, x_probe, *, render, gen, device, seed=88):
    """The sighted grader: acted-block FM error on the TREE channel, frozen probe."""
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    blks = tb["tree_blocks"]
    k = blks[torch.randint(0, len(blks), (x_probe.shape[0],), device=device, generator=g)]
    z, target = transition_targets(controller, generator, x_probe, k, tb, render=render, gen=gen)
    return fm_error(fm, z, k, target, acted_only=True)[0]


def _run_arm(arm, controller0, fm0, value0, generator, layout, tb, sigmas, *, levels, theta0,
             x_probe, x_eval,
             r_eval, probe_leaves, level_feats, seed, L, v, s, state_dim, rounds,
             collect_budget, mon_n, forecast_n, floor_n, floor_draws, tap_ema, fm_epochs,
             lp_steps, belief_steps,
             ground_states, anchor_pool, batch_size, n_corrupt, edit_budget, drift_kappa,
             drift_steps_per_round, drift_prewarm, lam_fm, lam_plan, tau, probe_steps, grade_every,
             ballistic_chunk, policy, render, device):
    """One arm's trajectory. Every arm forks from the SAME controller/FM/value and receives
    bit-identical budgets; the only difference is which level of the tree grammar drifts (and
    `nodrift` is therefore the matched-compute control the DiD needs)."""
    import torch
    import torch.nn.functional as F

    controller = copy.deepcopy(controller0)
    for p in controller.parameters():
        p.requires_grad_(True)
    fm = copy.deepcopy(fm0)
    value = copy.deepcopy(value0)
    for mod in (fm, value):
        for p in mod.parameters():
            p.requires_grad_(True)

    n_ch, names = tb["n_channels"], tb["channel_names"]
    tree_ch = layout["tree"]
    level = None if ARM_LEVELS[arm] is None else (L + ARM_LEVELS[arm]) % L
    # every arm holds the SAME W0 (non-uniform at every swept level); only the arm's own
    # level is given a nonzero sigma/kappa, so the single controlled variable is which level
    # moves. `nodrift` freezes all of them and is therefore the matched-compute control at a
    # world of matched entropy, not at the entropy maximum.
    tree_ch["drift"] = make_drift_state(tree_ch["rules"], levels, None, seed=seed + 601,
                                        theta0=theta0)
    tree_ch["drift_sigma"] = {ell: (sigmas[ell] if ell == level else 0.0) for ell in levels}
    tree_ch["drift_kappa"] = {ell: (drift_kappa if ell == level else 0.0) for ell in levels}
    for ch in layout["channels"]:
        if ch is not tree_ch:
            ch["drift"] = None
    refresh_w_blk(tb, layout, device)

    gen = torch.Generator(device=device).manual_seed(seed + 777)
    drift_rng = np.random.default_rng(seed + 602)
    alloc_rng = np.random.default_rng(seed + 603)
    # walk to stationarity BEFORE round 1 so every event is drawn from the same shift
    # distribution -- otherwise round 1's event is a fraction of round 5's and "repair
    # improves with t" is partly just "the drift got bigger".
    if level is not None and drift_prewarm:
        prewarm_drift(layout, drift_rng, drift_prewarm)
        refresh_w_blk(tb, layout, device)
    meter = Meter()
    satiety = np.zeros(n_ch)
    lp_s = red_s = None
    rows = []

    for rnd in range(1, rounds + 1):
        # --- the drift event ---------------------------------------------------------
        kl, kl_cum = 0.0, 0.0
        if level is not None:
            ev = advance_drift(layout, drift_rng, drift_steps_per_round)["tree"]
            kl, kl_cum = ev["kl_from_prev"], ev["kl_from_uniform"]
            refresh_w_blk(tb, layout, device)

        # a FRESH post-drift eval set: what the world looks like now
        post = sample_pool(layout, 4096, seed + 5000 + rnd)
        post_leaves = torch.from_numpy(post["leaves"])
        post_roots = torch.from_numpy(post["roots"].astype(np.int64))

        stale = {"tree_fm_err": _tree_err(fm, controller, generator, tb, x_probe,
                                          render=render, gen=gen, device=device),
                 "root_ce": root_ce(controller, post_leaves, post_roots, device)}

        # --- the loop's own allocation (e tap, p tap, collect) ------------------------
        x_mon, _ = sample_states(layout, tb, generator, n=mon_n, n_corrupt=n_corrupt,
                                 presteps=edit_budget // 2, seed=seed + 9000 + rnd * 7,
                                 device=device, render=render, gen=gen)
        lp_info = counterfactual_lp(fm, controller, generator, tb, x_mon, lp_steps=lp_steps,
                                    batch_size=batch_size, lr=1e-3, meter=meter,
                                    render=render, gen=gen, device=device)
        errors = {c: lp_info[c]["e_before"] for c in range(n_ch)}
        lp_raw = {c: lp_info[c]["lp"] for c in range(n_ch)}
        # the REPAIRED reducibility tap, so §12's mechanism (satiate on what has stopped
        # paying, move on) is actually present in this loop. The first climb run allocated by
        # `lprog x visits`, whose LP estimator was inverted by data starvation -- so the one
        # mechanism the idea doc says produces climbing was never in the loop being asked to
        # produce it.
        x_fl, _ = sample_states(layout, tb, generator, n=floor_n, n_corrupt=n_corrupt,
                                presteps=edit_budget // 2, seed=seed + 46, device=device,
                                render=render, gen=gen)
        floors_c = measure_floors(controller, generator, tb, x_fl, meter=meter,
                                  n_draws=floor_draws, render=render, gen=gen, device=device,
                                  seed=seed + 47)
        red_raw = reducible_fraction(errors, floors_c, n_ch)
        lp_s = ema(lp_s if rnd > 1 else None, lp_raw, tap_ema, n_ch)
        red_s = ema(red_s if rnd > 1 else None, red_raw, tap_ema, n_ch)
        lprog, reducible = lp_s, red_s
        noise_ch = [c for c, ch in enumerate(layout["channels"]) if ch["kind"] == "noise"]
        lp_floor = float(max(abs(lprog[c]) for c in noise_ch)) if noise_ch else 0.0
        red_floor = float(max(reducible[c] for c in noise_ch)) if noise_ch else 0.0

        x_fc, r_fc = sample_states(layout, tb, generator, n=forecast_n, n_corrupt=n_corrupt,
                                   presteps=0, seed=seed + 9500 + rnd * 7, device=device,
                                   render=render, gen=gen)
        meter.charge_monitor(forecast_n)
        visits, _dv = forecast_visits(fm, value, controller, tb, x_fc, r_fc,
                                      budget=edit_budget, device=device)
        w, _drive = allocate(policy, errors=errors, lprog=lprog, visits=visits,
                             tree_channel=tb["tree_channel"], n_channels=n_ch,
                             lp_floor=lp_floor, reducible=reducible, red_floor=red_floor,
                             satiety=satiety, beta_sat=1.0, eps=0.01)
        satiety = 0.7 * satiety + w

        x_col, _ = sample_states(layout, tb, generator, n=collect_budget, n_corrupt=n_corrupt,
                                 presteps=edit_budget // 2, seed=seed + 10_000 + rnd * 7,
                                 device=device, render=render, gen=gen)
        pick = torch.from_numpy(alloc_rng.choice(n_ch, size=collect_budget, p=w)).to(device)
        k_col = torch.empty(collect_budget, dtype=torch.long, device=device)
        for c in range(n_ch):
            sel = pick == c
            blks = blocks_of_channel(tb, c)
            k_col[sel] = blks[torch.randint(0, len(blks), (int(sel.sum()),), device=device)]
        z_col, t_col = transition_targets(controller, generator, x_col, k_col, tb,
                                          render=render, gen=gen)
        meter.charge_collect(collect_budget)

        fm.train()
        opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(max(1, fm_epochs * collect_budget // batch_size)):
            idx = torch.randint(0, collect_budget, (batch_size,), device=device)
            loss = F.mse_loss(fm(z_col[idx], k_col[idx]), t_col[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()
        fm.eval()
        del z_col, t_col

        # --- the belief update: the EVALUATIVE grader (grounded DP best move) ---------
        anchor = sample_pool(layout, anchor_pool, seed + 6000 + rnd)
        anchor_leaves = torch.from_numpy(anchor["leaves"])
        anchor_roots = torch.from_numpy(anchor["roots"].astype(np.int64))
        gx, gr = sample_states(layout, tb, generator, n=ground_states, n_corrupt=n_corrupt,
                               presteps=edit_budget // 2, seed=seed + 7000 + rnd,
                               device=device, render=render, gen=gen)
        gk, d_cur, d_best = collect_grounded_moves(layout, tb, generator, gx, gr,
                                                   render=render, gen=gen, device=device)
        info = belief_update("evaluative", controller, fm, value, generator, layout, tb,
                             train_leaves=anchor_leaves, train_roots=anchor_roots,
                             gstates=gx, groots=gr, gkstar=gk, n_steps=belief_steps,
                             batch_size=batch_size, lr=3e-4, lam_fm=lam_fm,
                             lam_plan=lam_plan, tau=tau, n_corrupt=n_corrupt,
                             edit_budget=edit_budget, render=render, gen=gen, device=device,
                             seed=seed + rnd)

        end = {"tree_fm_err": _tree_err(fm, controller, generator, tb, x_probe,
                                        render=render, gen=gen, device=device),
               "root_ce": root_ce(controller, post_leaves, post_roots, device)}
        depth = tree_depth_probe(controller, probe_leaves, level_feats, layout, tb, device,
                                 probe_steps=probe_steps)

        row = {"round": rnd, "drift_kl": kl, "drift_kl_cumulative": kl_cum, "depth": depth,
               "tree_fm_err_stale": stale["tree_fm_err"], "tree_fm_err_end": end["tree_fm_err"],
               "root_ce_stale": stale["root_ce"], "root_ce_end": end["root_ce"],
               "dstar_cur": d_cur, "dstar_best_move": d_best,
               "alloc": {names[c]: float(w[c]) for c in range(n_ch)},
               "reducible": {names[c]: float(reducible[c]) for c in range(n_ch)},
               "root_acc": info.get("root_acc"), "plan_acc": info.get("plan_acc"),
               "meter": {"collect": meter.collect, "monitor": meter.monitor,
                         "ratio": meter.ratio}}
        if rnd % grade_every == 0 or rnd == rounds:
            row["ballistic_w16"] = open_loop_beam(controller, generator, fm, value, tb, layout,
                                                  x_eval, r_eval, budget=edit_budget,
                                                  beam_width=16, render=render, gen=gen,
                                                  device=device, chunk_len=ballistic_chunk)
        rows.append(row)
        print(f"  r{rnd:02d} KL={kl:.3f} tree_err {stale['tree_fm_err']:.4f}->"
              f"{end['tree_fm_err']:.4f}  rootCE {stale['root_ce']:.4f}->{end['root_ce']:.4f}"
              f"  depth " + " ".join(f"{k}={val:.3f}" for k, val in depth.items())
              + (f"  ballistic={row['ballistic_w16']:.3f}" if "ballistic_w16" in row else ""))
    return {"rounds": rows, "meter_ratio": float(rows[-1]["meter"]["ratio"])}


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1"):
    climb.remote(quick=quick, tag=tag)
