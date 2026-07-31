"""E5 -- NECESSITY on the arm: does a METER migrate repair from LOCAL patching to SHARED structure?

Up: [`README.md`](README.md) · Node: [`../README.md`](../README.md) (on_policy) · mjc: [`../../README.md`](../../README.md)
Sibling: [`floor_tap.py`](floor_tap.py) (E4) -- shares this folder's machinery, answers a different question.
Ports: [`rhm/directed_sculpting/full_loop/`](../../../rhm/directed_sculpting/full_loop/README.md) §6
(PR #16), the necessity sweep -- the first positive RHM produced on the climbing axis.

WHAT RHM FOUND, AND WHY IT WANTS A SECOND SUBSTRATE. RHM held drift magnitude EXACTLY fixed (0.600
nats/event via `calibrate_sigma_event`) and starved only SAMPLES PER EVENT. The learner's depth
advantage migrated from the surface to the deep levels, monotonically across four budgets
(slope -0.00585 +- 0.00186 per octave, t = -5.44, 3/3 seeds). The diagnosis: at the abundant end the
surface arm "repaired 91% of its damage inside the round", so invariance was FREE and nothing priced
it -- "INVARIANCE != NECESSITY". The reframe the idea doc drew from it
([`ideas/meta_learning_under_metered_data.md`](../../../../ideas/meta_learning_under_metered_data.md)
§3) is that the operative variable is the PRICE PER SAMPLE, not the learner's stage.

Two reasons that claim should be tested here and not only there:

  1. IT IS A CLAIM ABOUT THE SHAPE OF THE REPAIR PROBLEM, so it should not be about RHM's grammar. If
     it reproduces on continuous physics with a different readout, "metered" is a property of the
     problem; if it does not, the RHM result is about the RHM.
  2. RHM CANNOT INSTRUMENT IT. Its repair readout has failed FOUR times, most recently with NEGATIVE
     damage, because `KL(w||uniform) = log m - H(w)` makes its cross-arm root-CE difference
     structurally broken -- a property of RHM's drift primitive, not of the claim. So every RHM
     climbing conclusion rests on the depth probe ALONE, against its own §6 two-instrument
     requirement, at magnitudes of 0.006-0.030. This substrate's readout is a TRANSFER measurement,
     positive-definite and immune to that identity.

THE CONSTRUCTION -- a hierarchy in the NON-STATIONARITY, not in the plant. `mjc/expansion/` already
established that a fixed-DOF plant has no hierarchy to expand INTO, and that is not relitigated here:
nothing in this cut claims expansion. What it needs is only that damage be repairable at two
different levels of generality, which a drift GENERATOR supplies:

    b_j(t) = c_j * beta(t) + eps_j(t)          per-region curl gain

  * `beta` -- ONE shared latent walking over rounds. Deep/general: it explains damage in EVERY region
    at once, so a learner that tracks it repairs regions it never visited.
  * `eps_j` -- independent per-region walks. Surface/local: only re-fitting region j fixes region j.

`share_frac` mixes them, and the per-event damage is matched across arms ANALYTICALLY rather than by
search: with `s_shared = scale*sqrt(f)` and `s_local = scale*sqrt(1-f)`, every region's per-event
gain increment has variance `scale^2` for EVERY `f`. Damage is a deterministic function of that
increment, so the arms are matched by construction -- and it is still MEASURED and reported per arm,
because assuming it is what bit RHM (`calibrate_sigma` matched accumulated displacement rather than
the event, and one level ran at 3.49 nats against a 0.60 target).

The walk is a TRUE MEAN-REVERTING OU, which matters more than it sounds. E3's "OU walk" is a
reflected random walk, and with a per-event sd comparable to the band width it spends much of its
time at the bounds -- where reflection distorts exactly the increment the design guarantees. The
calibration gate measured that directly on an earlier version: `local` came out at sd(Δb) 0.714/2.117
against a construction that says 1.60, with corr(b_A,b_A') = -0.870 where it should be ~0. Mean
reversion gives the walk a stationary distribution that sits inside the band on its own. Verified
offline over 6000 synthetic steps: sd(Δb) = 1.77 for every arm and region, and
corr(b_A,b_A') = +1.000 / +0.712 / +0.014 for f = 1.0 / 0.7 / 0.0 -- tracking `share_frac` exactly.

THE ARMS (all at matched per-event damage, one shared drift trajectory per arm across all budgets):
  * `pure_shared` (f=1.0) -- the CALIBRATION arm, in `mjc/expansion/`'s calibrate->measure discipline.
    Here the shared component is the ONLY structure, so transfer is maximally available; if the
    readout shows no transfer even HERE the instrument is dead and the null is uninterpretable. This
    is the arm that decides whether a null in `shared` means anything.
  * `shared`      (f=0.7) -- the measurement.
  * `local`       (f=0.0) -- the matched CONTROL. Nothing is shared, so any apparent transfer is the
    instrument's own leakage (an MLP's smoothness across tip space), and `shared - local` is the real
    quantity.

THE TWO INSTRUMENTS (RHM's §6 requirement, met here for the first time in this line):
  * SURFACE -- FM error in region A, the region the whole budget is spent in.
  * DEEP    -- FM error in region A', a reducible region that is OFF-REACH and receives ZERO
               collection, all round, every round. Its in-region collection share is logged to prove
               it. Repair there cannot come from local re-fitting; it can only come from structure.
  Both are read BEFORE (post-event, pre-repair) and AFTER (post-fine-tune) every round, so `repair`
  is a difference within a round rather than across arms -- which is what makes it positive-definite.
  Headline: `transfer_frac = transfer_repair / surface_repair`, and its slope against log2(samples
  per event). PREDICTED NEGATIVE (transfer grows as data is starved).

THE GRADER TRACKS THE CURRENT WORLD. E3's per-region probes are gathered once in the round-0 world
and reused while the world keeps drifting, so its `reg_err` is distance-from-a-stale-world. That is
tolerable for a ladder normalised by a matched ceiling and fatal for a before/after repair
measurement. Here a FIXED (S,U) probe set is RE-EXECUTED in the current world each round to get that
world's targets (`reexec`) -- the experimenter's off-budget teleport ruler, exactly the standing of
E3's probes and matched ceiling, and never available to the agent.

FOUR CONFOUNDS, CONTROLLED EXPLICITLY:
  * ALLOCATION IS NOT A VARIABLE. The whole budget goes to region A every round, by construction --
    no policy, no taps, no survey (so the metered spend IS the collection budget, and monitor:collect
    is 0). E5 asks what the learner REPRESENTS, not where it looks; letting a drive choose would let
    allocation drift with budget and confound the migration. E4's repaired drive is a follow-up
    (`--policy`), not the primary.
  * EPISODE LENGTH IS HELD FIXED and PARALLELISM is scaled instead (`n_par = budget // ep_len`). The
    naive way to starve a budget -- keep 16 parallel episodes and cut them short -- would hand the
    starved arms only early-episode states, changing the DISTRIBUTION of data as well as its amount.
  * GRADIENT STEPS ARE HELD FIXED across budgets. Scaling them would confound metered DATA with
    metered COMPUTE, which is a different claim.
  * THE RING BUFFER SCALES with the budget (`2*budget`), or a fixed cap would silently throw away the
    abundant arms' abundance and manufacture the result.

Run:
    cd experiments/            # MODAL_PROFILE=chromatic
    modal run mjc/on_policy/metered_repair/necessity.py::necessity \
        --quick --calibrate-only --tag ncal      # geometry + per-event damage, ~minutes
    modal run mjc/on_policy/metered_repair/necessity.py::necessity --quick --tag nsmoke
    for s in 0 1 2; do        # launch each seed as its OWN client (siblings evict each other)
      modal run --detach mjc/on_policy/metered_repair/necessity.py::necessity --tag nec_s$s --seed $s
    done
    python3 mjc/on_policy/metered_repair/necessity_agg.py --tags nec_s0 nec_s1 nec_s2
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

_ALL_ARMS = "pure_shared:1.0,shared:0.7,local:0.0"
_ALL_BUDGETS = "432,144,48,16"


@app.function(gpu="L4", memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_necessity(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from mjc.embodied import ReachBehaviour

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; n = cfg["n_links"]
    SD, AD = 2 * n, n
    qc = np.array(cfg["q_center"][:n], dtype=np.float64)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    B = cfg["n_eval"]; T = cfg["rounds"]
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

    # ================================================================= #
    # GEOMETRY -- two reducible curl regions: A (most visited, gets the whole budget) and A' (least
    # visited, gets nothing, ever). Both in the EXTENDED/tame radius band, E3 gotcha (ii): a
    # folded/inward region's fast dynamics give ~10x the FM error and swamp the curl signal.
    # ================================================================= #
    def sample_reach_endpoints(n_samp, rng):
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (n_samp, n))
        t0 = fk(q0, Ls)
        eps = np.empty((n_samp, 2)); qg = np.empty((n_samp, n))
        lo, hi = cfg["reach_lo"], cfg["reach_hi"]
        for i in range(n_samp):
            best, best_pen = None, np.inf
            for _ in range(cfg["reach_tries"]):
                d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
                cand = q0[i] + cfg["reach_amp"] * d
                dist = float(np.linalg.norm(fk(cand, Ls) - t0[i]))
                pen = max(0.0, lo - dist) + max(0.0, dist - hi)
                if pen < best_pen:
                    best, best_pen = cand, pen
                if pen == 0.0:
                    break
            eps[i] = fk(best, Ls); qg[i] = best
        return q0, qg, eps, t0.mean(0)

    geo_rng = np.random.default_rng(cfg["seed"] + 20)
    q0_cloud, qg_cloud, cloud, P0 = sample_reach_endpoints(cfg["geom_samples"], geo_rng)
    radius = np.linalg.norm(cloud, axis=1)
    ext = radius > cfg["rad_min"]
    ang_all = np.arctan2(cloud[:, 1] - P0[1], cloud[:, 0] - P0[0])
    hcnt, edges = np.histogram(ang_all[ext], bins=24, range=(-np.pi, np.pi))
    theta_pref = 0.5 * (edges[np.argmax(hcnt)] + edges[np.argmax(hcnt) + 1])
    dth = np.angle(np.exp(1j * (ang_all - theta_pref)))
    ev_mask = (np.abs(dth) < cfg["pref_width"]) & ext
    ev_q0, ev_qg = q0_cloud[ev_mask], qg_cloud[ev_mask]
    sig2 = 2.0 * cfg["region_sigma"] ** 2

    def path_visitation(cands):
        """FM-free occupancy of the eval reaches over candidate sites -- the tip PATH is curved, so a
        straight-corridor test mislabels swung-through regions (E3 gotcha (i))."""
        occ = np.zeros(len(cands))
        for a in np.linspace(0.0, 1.0, cfg["vis_steps"]):
            tips = fk(ev_q0 * (1 - a) + ev_qg * a, Ls)
            dd = ((cands[:, None, 0] - tips[None, :, 0]) ** 2
                  + (cands[:, None, 1] - tips[None, :, 1]) ** 2)
            occ += np.exp(-dd / sig2).sum(1)
        return occ / max(occ.max(), 1e-9)

    vis = path_visitation(cloud)
    G = cloud[ev_mask].mean(0)
    # REACHABILITY, and it is not optional. Candidates come from `sample_reach_endpoints`, whose
    # tip-displacement band [reach_lo, reach_hi] is enforced only BEST-EFFORT (`reach_tries` samples,
    # and the least-bad candidate is kept even when the penalty is non-zero). So the cloud has a tail
    # of points far outside the band -- and A' is chosen as the LEAST-visited point, which selects
    # that tail preferentially. The first run of this gate placed A' 0.81 m from the start tip against
    # a 0.50 m reach band: `collect_toward` could not get the body there, the in-region probe set came
    # back 0/500, and the entire transfer readout would have been NaN. Filter on the achieved distance.
    d_P0 = np.linalg.norm(cloud - P0[None, :], axis=1)
    # `reach_pad` is calibrated from E4's cert_v2, which measured the cutoff directly: regions 0.22 to
    # 0.43 m from the start tip returned 272-1308 in-region probe transitions, one at 0.55 m returned
    # ZERO. The constraint is DYNAMIC reach in `ep_len` steps, not kinematic reachability -- every
    # cloud point is kinematically reachable by construction. The density floor guards the other
    # failure: a point in a sparse tail of the endpoint cloud has few postures landing in its gate.
    reachable = d_P0 <= cfg["reach_hi"] * cfg["reach_pad"]
    gate_r = cfg["region_k"] * cfg["region_sigma"]
    dens = np.array([int((np.linalg.norm(cloud - cloud[i][None, :], axis=1) < gate_r).sum())
                     for i in range(len(cloud))])
    reachable = reachable & (dens >= max(2, int(cfg["min_density_frac"] * len(cloud))))
    print(f"[geom] reachable candidates: {int((ext & reachable).sum())}/{len(cloud)}", flush=True)
    # (no lower bound: close-in points are trivially reachable, and E3 gotcha (i)'s hazard -- a region
    # sitting where every reach starts -- is excluded by visitation, since such a point is the most
    # visited, never the least.)
    ext_idx = np.flatnonzero(ext & reachable)
    A_c = cloud[ext_idx[np.argmax(vis[ext_idx])]]                 # most-visited reachable point
    far = np.linalg.norm(cloud - A_c[None, :], axis=1) > cfg["min_sep"]
    cand_idx = np.flatnonzero(ext & reachable & far)
    Ap_c = cloud[cand_idx[np.argmin(vis[cand_idx])]]              # least-visited reachable point
    # A THIRD REGION, and it is the control the smoke proved necessary. `Aplacebo` is built exactly
    # like A' -- off-reach, reducible, zero collection -- except its curl gain NEVER DRIFTS. So any
    # repair measured there is generic model improvement that has nothing to do with tracking a gain,
    # and `repair(A') - repair(Aplacebo)` isolates the part of A''s repair that is actually about
    # structure. Without it the readout is swamped: the first smoke measured a round-0 repair of
    # 0.393 against a per-event damage of 0.025 (15.9x), all of it the FM learning the reach manifold,
    # which is global and lands on A' whether or not anything is shared. That drove `transfer_frac`
    # to ~1.0 in EVERY arm including `local`, where corr(b_A,b_A') = -0.03.
    far2 = far & (np.linalg.norm(cloud - Ap_c[None, :], axis=1) > cfg["min_sep"])
    cand2 = np.flatnonzero(ext & reachable & far2)
    Apl_c = cloud[cand2[np.argmin(vis[cand2])]]
    REG = [dict(name="A", center=A_c, sigma=cfg["region_sigma"], on_reach=True),
           dict(name="Aprime", center=Ap_c, sigma=cfg["region_sigma"], on_reach=False),
           dict(name="Aplacebo", center=Apl_c, sigma=cfg["region_sigma"], on_reach=False)]
    K = len(REG)
    DRIFT_IDX = {0, 1}                 # A and Aprime drift; Aplacebo is pinned at b1
    names = [r["name"] for r in REG]
    reg_vis = path_visitation(np.array([np.asarray(r["center"]) for r in REG]))
    # `c_j`: the loading of each region on the shared latent. "equal" is the maximally-learnable
    # version and so the fairest first test -- if a shared component cannot be exploited when every
    # region loads on it identically, it will not be exploited when they differ.
    if cfg["share_load"] == "equal":
        c_load = np.ones(K)
    else:
        c_load = np.random.default_rng(cfg["seed"] + 31).uniform(0.6, 1.4, K)
    print(f"[geom] P0={P0.round(3).tolist()} theta_pref={theta_pref:+.2f}", flush=True)
    for j, r in enumerate(REG):
        c = np.asarray(r["center"])
        print(f"[geom]   {names[j]}: center={c.round(3).tolist()} r={np.linalg.norm(c):.2f} "
              f"visitation={reg_vis[j]:.2f} on_reach={r['on_reach']} c_load={c_load[j]:.2f}",
              flush=True)
    geom_gate = {"A_vis": float(reg_vis[0]), "Ap_vis": float(reg_vis[1]),
                 "sep": float(np.linalg.norm(A_c - Ap_c)),
                 "dist_from_P0": [float(np.linalg.norm(np.asarray(r["center"]) - P0)) for r in REG],
                 "c_load": c_load.tolist()}
    geom_gate["pass_A"] = geom_gate["A_vis"] >= cfg["on_vis_min"]
    # NOTE this is a soft check. E5's transfer readout does not care whether the EVAL reaches pass
    # through A' -- it reads a fixed probe set, and what must be zero is A's COLLECTION spilling into
    # A', which fixed allocation guarantees by construction and `in_share_Ap` verifies empirically
    # every round. Eval occupancy touches only the ballistic cash-out.
    geom_gate["pass_Ap"] = geom_gate["Ap_vis"] <= cfg["off_vis_max"]
    print(f"[geom] GATE  A_vis={geom_gate['A_vis']:.2f}(>={cfg['on_vis_min']})->"
          f"{'PASS' if geom_gate['pass_A'] else 'FAIL'}  "
          f"Ap_vis={geom_gate['Ap_vis']:.2f}(<={cfg['off_vis_max']})->"
          f"{'PASS' if geom_gate['pass_Ap'] else 'FAIL'}  sep={geom_gate['sep']:.2f}m  "
          f"dist_from_P0={[round(d, 2) for d in geom_gate['dist_from_P0']]}"
          f"(<={cfg['reach_hi'] * cfg['reach_pad']:.2f})", flush=True)

    def gate_np(tips, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((tips[..., 0] - c[0]) ** 2 + (tips[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    def in_region(states, j):
        tips = fk(np.asarray(states)[:, :n].astype(np.float64), Ls)
        return np.linalg.norm(tips - REG[j]["center"], axis=1) < cfg["region_k"] * REG[j]["sigma"]

    # ================================================================= #
    # ENV
    # ================================================================= #
    def env_dgp(b_state):
        curls = [{"b": float(b_state[j]), "center": tuple(np.asarray(REG[j]["center"]).tolist()),
                  "sigma": REG[j]["sigma"]} for j in range(K)]
        return dict(n_links=n, link_lengths=cfg["link_lengths"][:n], link_masses=cfg["link_masses"][:n],
                    joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                    noise_seed=cfg["seed"] + 999, curl_fields=curls)

    def make_env(b_state):
        return ArmEnv(env_dgp(b_state))

    # ================================================================= #
    # FM  f(s,u) -> Δs  (architecture and training identical to E2/E3)
    # ================================================================= #
    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(SD + AD, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, SD)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    huber = nn.HuberLoss(delta=1.0)

    def train_steps(net, opt, S, U, S2, steps, brng):
        X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def fm_delta(net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1).astype(np.float32), device=device)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    def fm_err(net, S, U, S2):
        if len(S) == 0:
            return float("nan")
        return float(np.linalg.norm(fm_delta(net, S, U) - (S2 - S).astype(np.float32), axis=1).mean())

    def teleport_pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs, qc, cfg["q_range"],
                            cfg["v_explore"])

    b_ref = np.full(K, cfg["b1"], float)
    env_ref = make_env(b_ref)
    nS, nU, nS2 = teleport_pool(env_ref, cfg["pool_n"], cfg["seed"] + 11)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    # ================================================================= #
    # CEM planner + eval rollout
    # ================================================================= #
    def make_plan_fn(net, k_shoot, cem_iters, rng):
        def plan(states, goals):
            Bn = states.shape[0]
            mu = np.zeros((Bn, H, AD), np.float32)
            sig = np.full((Bn, H, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(np.asarray(goals, np.float32), device=device).repeat_interleave(k_shoot, 0)
            s0 = torch.tensor(np.asarray(states, np.float32), device=device).repeat_interleave(k_shoot, 0)
            for _ in range(cem_iters):
                e = rng.standard_normal((Bn, k_shoot, H, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    seqs_t = torch.tensor(seqs.reshape(Bn * k_shoot, H, AD), device=device)
                    cost = torch.zeros(Bn * k_shoot, device=device)
                    for h in range(H):
                        x = torch.cat([s, seqs_t[:, h, :]], 1)
                        s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                        cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                    cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, k_shoot), cfg["cem_elite"], dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1); sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)
        return plan

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        _, _, eps, _ = sample_reach_endpoints(cfg["n_eval"] * 6, rng)
        a = np.arctan2(eps[:, 1] - P0[1], eps[:, 0] - P0[0])
        d = np.abs(np.angle(np.exp(1j * (a - theta_pref))))
        pick = np.argsort(d)[:B]
        goals = eps[pick].astype(np.float32)
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (B, n))
        starts = np.concatenate([q0, rng.normal(0, cfg["v0_std"], (B, n))], 1).astype(np.float32)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, replan_every, cenv):
        states = ev_starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            for b in range(B):
                cenv.set_state(states[b, :n].astype(np.float64), states[b, n:].astype(np.float64))
                s2, _ = cenv.step(acts[b], fs)
                states[b] = s2
        tips = fk(states[:, :n].astype(np.float64), Ls).astype(np.float32)
        return float(np.median(np.linalg.norm(tips - ev_goals, axis=1)))

    # ================================================================= #
    # ON-POLICY collection toward a region (the metered acquisition primitive).
    # `n_par` is scaled with the budget so EPISODE LENGTH stays fixed -- see the docstring's confound
    # list. Cutting episodes short instead would hand the starved arms only early-episode states.
    # ================================================================= #
    def make_fixed_goal_sampler(center, jit):
        c = np.asarray(center, np.float32)

        def sample(states, rng):
            m = len(np.atleast_2d(np.asarray(states)))
            return (c[None, :] + rng.uniform(-jit, jit, (m, 2))).astype(np.float32)
        return sample

    def collect_toward(cenv, center, need, fm, rng_buf, k, n_par=None):
        n_par = int(n_par or cfg["n_par"])
        plan_fn = make_plan_fn(fm, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                               np.random.default_rng(cfg["seed"] + 8000 + k))
        beh = ReachBehaviour(plan_fn, np.zeros((n_par, 2), np.float32), rng_buf,
                             sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
        gs = make_fixed_goal_sampler(center, cfg["goal_jit"])
        S, U, S2, info = collect_pool(cenv, need, rng_buf, fs, qc, cfg["op_q_range"], None,
                                      collection_mode="on_policy", behaviour=beh, goal_sampler=gs,
                                      ep_len=cfg["ep_len"], n_par=n_par, v0_std=cfg["v0_std"],
                                      wrap_limit=cfg["wrap_limit"], return_info=True)
        return S, U, S2, info

    def budget_n_par(budget):
        return int(np.clip(budget // cfg["ep_len"], 1, cfg["n_par"]))

    # ================================================================= #
    # BASE (stale) FM -- trained off-budget on the reference world. Every cell forks from this.
    # ================================================================= #
    # THE BASE FM IS TRAINED ON COMPETENT ON-POLICY REACHES, not on a broad teleport pool, and this
    # is the single change that makes the whole cut measurable. E0 established that the default
    # teleport sampling (`q_range=0.9, v_explore=8`) is MISTUNED for the task manifold -- 0.304 task
    # error against 0.087 for an oracle teleporter -- so a base FM trained that way begins the sweep
    # far off the manifold its probes live on. The first smoke measured the consequence exactly:
    # round-0 repair of 0.393 against a per-event damage of 0.025. Sixteen times the signal, global,
    # and therefore landing on the held-out region in every arm. Start on-manifold and the only thing
    # left to repair is the drift, which is the thing being measured.
    ref_net = _mlp(cfg["seed"] + 43)
    dS, dU, dS2 = teleport_pool(env_ref, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                dS, dU, dS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 401))
    S0_, U0_, S20_, _ = collect_toward(env_ref, G, cfg["base_pool_n"], ref_net,
                                       np.random.default_rng(cfg["seed"] + 10), 10)
    fm_base = _mlp(cfg["seed"] + 40)
    train_steps(fm_base, torch.optim.Adam(fm_base.parameters(), lr=cfg["fm_lr"]),
                S0_, U0_, S20_, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    base_state = copy.deepcopy(fm_base.state_dict())
    # THE REPLAY BUFFER MUST BE ON-MANIFOLD *AND* DRIFT-INVARIANT, and having one without the other
    # breaks the cut in opposite ways. Both failures were measured, one commit apart:
    #   * filtering the on-policy base pool to out-of-EVERY-region left `replay = 0` -- the regions
    #     sit on the reach path, so almost nothing survives a 0.34 m exclusion radius -- and the
    #     fine-tune lost its anchor entirely;
    #   * replacing it with the BROAD TELEPORT pool restored the anchor and wrecked the readout,
    #     A-err 0.12 -> 0.70, because that pool is E0's mistuned off-manifold sampling and 1200 of it
    #     against ~150 on-manifold transitions simply drags the FM off the manifold the probes
    #     measure. Reverting the damage did NOT fix it, which is what identified the replay as the
    #     cause rather than the drift magnitude.
    # So gather replay ON-POLICY across the eval goal SPREAD -- not just the centroid, so it covers
    # the manifold instead of one corridor -- and exclude only the DRIFTING regions. The placebo's
    # data stays valid forever precisely because its gain never moves, which is a free extra anchor.
    rgoals = ev_goals[np.random.default_rng(cfg["seed"] + 461).permutation(len(ev_goals))]
    accS, accU, accS2 = [], [], []
    for gi in range(min(cfg["replay_goals"], len(rgoals))):
        pS, pU, pS2, _ = collect_toward(env_ref, rgoals[gi], cfg["replay_chunk"], ref_net,
                                        np.random.default_rng(cfg["seed"] + 470 + gi), 470 + gi)
        accS.append(pS); accU.append(pU); accS2.append(pS2)
    aS, aU, aS2 = np.concatenate(accS), np.concatenate(accU), np.concatenate(accS2)
    gmax = np.max(np.stack([gate_np(fk(aS[:, :n].astype(np.float64), Ls), j)
                            for j in sorted(DRIFT_IDX)]), 0)
    oidx = np.flatnonzero(gmax < cfg["replay_gate"])
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(oidx)[:cfg["replay_n"]]
    rS, rU, rS2 = aS[rp], aU[rp], aS2[rp]
    if len(rp) < cfg["replay_min"]:
        raise RuntimeError(f"replay anchor too small ({len(rp)}) -- loosen `replay_gate` or raise "
                           f"`replay_goals`")
    print(f"[pretrain] base FM trained on {len(S0_)} competent on-policy reaches; "
          f"replay={len(rp)}/{len(aS)} on-manifold, outside the drifting regions", flush=True)

    # ================================================================= #
    # THE GRADER -- a FIXED (S,U) probe set per region, RE-EXECUTED in the current world each round.
    # Off-budget (teleport), the experimenter's ruler, never available to the agent. This is what
    # makes a within-round before/after `repair` well-defined; E3's stale-world probe cannot.
    # ================================================================= #
    print("[probe] gathering fixed per-region probe states ...", flush=True)
    # ADAPTIVE, because a fixed pool silently under-fills the off-reach region and the whole transfer
    # readout lives on it. ncal_v3 measured the hit rate at region A' at 3.4% (17/500) against 56% at
    # A -- A' is off the eval path by design, which is exactly what makes it hard to aim at. Rather
    # than tune a pool size against an unknown hit rate, collect in chunks until the region has enough
    # in-region states or the try budget runs out, and report what it took. This is off-budget setup
    # (the experimenter's ruler), so the cost is wall-clock, not the meter.
    PROBE = []
    for j in range(K):
        accS, accU, tried = [], [], 0
        for a in range(cfg["probe_max_tries"]):
            pS, pU, _, _ = collect_toward(env_ref, REG[j]["center"], cfg["probe_pool_n"], ref_net,
                                          np.random.default_rng(cfg["seed"] + 900 + 17 * j + a),
                                          900 + 17 * j + a)
            m = in_region(pS, j)
            accS.append(pS[m]); accU.append(pU[m]); tried += len(pS)
            got = int(sum(len(x) for x in accS))
            if got >= cfg["probe_target"]:
                break
        S_, U_ = np.concatenate(accS), np.concatenate(accU)
        PROBE.append((S_, U_))
        print(f"[probe]   {names[j]}: {len(S_)}/{tried} in-region probe states "
              f"(hit rate {len(S_) / max(tried, 1):.1%}, {a + 1} chunk(s))", flush=True)
    # A HARD GATE, because the failure is silent otherwise: an empty probe set makes `fm_err` return
    # NaN, every repair NaN, and every summary column NaN -- which reads as "no effect" rather than
    # "no instrument". This is exactly how the first run of this calibration failed (A' 0/500).
    geom_gate["probe_n"] = [int(len(PROBE[j][0])) for j in range(K)]
    geom_gate["pass_probe"] = all(nn_ >= cfg["min_probe"] for nn_ in geom_gate["probe_n"])
    print(f"[probe] GATE  in-region probe counts={geom_gate['probe_n']} "
          f"(>={cfg['min_probe']} each) -> {'PASS' if geom_gate['pass_probe'] else 'FAIL'}",
          flush=True)
    if not geom_gate["pass_probe"] and not cfg.get("calibrate_only"):
        raise RuntimeError(f"empty/thin in-region probe set: {geom_gate['probe_n']} -- the transfer "
                           f"readout would be NaN. Fix the geometry before running the sweep.")

    def reexec(j, b_state):
        """That world's targets for region j's fixed probe states. The curl is deterministic, so this
        is exact -- no aleatoric term anywhere in this cut."""
        cenv = make_env(b_state)
        S, U = PROBE[j]
        S2 = np.empty_like(S)
        for i in range(len(S)):
            cenv.set_state(S[i, :n].astype(np.float64), S[i, n:].astype(np.float64))
            S2[i], _ = cenv.step(U[i], fs)
        return S2

    # ================================================================= #
    # THE DRIFT GENERATOR -- b_j(t) = c_j*beta(t) + eps_j(t), matched per-event variance across arms.
    # ================================================================= #
    def make_traj(share_frac, seed, steps=None):
        """A true MEAN-REVERTING OU walk on `b`, with a shared increment plus a local one.

        Two earlier versions failed here and the calibration gate caught both.
        (i) Unbounded `beta`/`eps` accumulators with only the DERIVED `b` reflected: the accumulators
            wandered past the legal band, the derived value hit the post-fold `clip` and PARKED --
            consecutive rounds identical, Δb = 0, the events silently stopped.
        (ii) Reflecting the walk itself: better, but with `scale` comparable to the band width the
            walk hits the bounds constantly, and reflection then distorts exactly the quantity the
            design guarantees. Measured: `local` came out at sd(Δb) 0.714/2.117 against a
            construction that says 1.60, with corr(b_A,b_A') = -0.870 where it should be ~0.
        Mean reversion fixes the cause rather than the symptom: the walk has a STATIONARY
        distribution (sd ~ scale/sqrt(2*theta)) that sits inside the band on its own, so the clip
        essentially never fires and every increment keeps its designed variance. E3 called its
        reflected random walk "OU"; this is the actual thing.

        With `c_load` equal and a shared start, the `pure_shared` arm makes b_A and b_A' identically
        equal at all times -- exactly what a calibration arm should be: learning the gain at A hands
        you A' for free. (Verified: corr = +1.000.)"""
        rng = np.random.default_rng(seed)
        s_sh = cfg["scale"] * np.sqrt(share_frac)
        s_lo = cfg["scale"] * np.sqrt(1.0 - share_frac)
        th = cfg["ou_theta"]
        b_cur = np.full(K, cfg["b1"], float)
        n_steps = T if steps is None else steps
        traj = np.zeros((n_steps, K)); dsh = np.zeros(n_steps); dlo = np.zeros((n_steps, K))
        for t in range(n_steps):
            # the EVENT opens the round (RHM's `climb.py` ordering: advance drift, then repair)
            d_shared = s_sh * rng.standard_normal()          # ONE draw, common to every region
            d_local = s_lo * rng.standard_normal(K)          # independent per region
            dsh[t] = d_shared; dlo[t] = d_local
            for j in range(K):
                if j not in DRIFT_IDX:                               # the placebo never moves
                    continue
                v = (b_cur[j] - th * (b_cur[j] - cfg["b1"])          # mean reversion
                     + c_load[j] * d_shared + d_local[j])
                b_cur[j] = float(np.clip(v, cfg["b_lo"], cfg["b_hi"]))
            traj[t] = b_cur.copy()
        return traj, dsh, dlo

    def generator_diagnostics(share_frac, seed, steps=4000):
        """Check the GENERATOR's properties on a long synthetic run, with no environment involved.

        The T-round trajectory is far too short to estimate an increment sd or a cross-region
        correlation -- `--quick`'s 6 rounds give 5 differences -- and reading those noisy numbers as
        a gate is how the previous version's real bug got half-diagnosed and half-masked. The
        generator is free to run for thousands of steps, so its properties are checked THERE and the
        actual trajectory is used only for the measured per-event damage."""
        tj, _, _ = make_traj(share_frac, seed + 5000, steps=steps)
        d = np.diff(tj, axis=0)
        # DRIFTING regions only. Including the placebo, whose increment sd is 0 by
        # construction, made the spread gate read 151% and FAIL a generator that is exact.
        return {"sd_increment": [float(d[:, j].std()) for j in sorted(DRIFT_IDX)],
                "corr_regions": float(np.corrcoef(tj[:, 0], tj[:, 1])[0, 1]),
                "corr_increments": float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]),
                "b_sd": [float(tj[:, j].std()) for j in sorted(DRIFT_IDX)],
                "b_min": float(tj.min()), "b_max": float(tj.max()),
                "clip_frac": float(np.mean((tj <= cfg["b_lo"] + 1e-9) | (tj >= cfg["b_hi"] - 1e-9)))}

    arms = cfg["arms"]                                     # [(name, share_frac), ...]
    trajs = {nm: make_traj(f, cfg["seed"] + 77) for nm, f in arms}

    # Per-event damage, MEASURED (not assumed -- RHM's `calibrate_sigma` bug). Damage = the rise in a
    # FIXED reference model's region-A error caused by one event, with the same probe states
    # re-executed in the before- and after-world so nothing but the world moves.
    damage = {}
    for nm, _f in arms:
        traj = trajs[nm][0]
        per = []
        for t in range(1, min(T, cfg["damage_events"] + 1)):
            e0 = fm_err(fm_base, PROBE[0][0], PROBE[0][1], reexec(0, traj[t - 1]))
            e1 = fm_err(fm_base, PROBE[0][0], PROBE[0][1], reexec(0, traj[t]))
            per.append(abs(e1 - e0))
        # `corr_AAp` is the direct measure of how SHARED an arm actually is -- the correlation of the
        # two regions' gains over rounds. It should be ~1.0 for `pure_shared` and ~0 for `local`, and
        # it is the quantity the transfer claim rests on, so it is reported rather than assumed.
        gd = generator_diagnostics(_f, cfg["seed"] + 77)
        damage[nm] = {"per_event": per, "mean": float(np.mean(per)), "sd": float(np.std(per)),
                      **{("gen_" + k): v for k, v in gd.items()}}
    print("\n=== per-event damage (region A, fixed reference FM; MUST be matched across arms) ===",
          flush=True)
    for nm, _f in arms:
        d = damage[nm]
        print(f"    {nm:>12s}  |Δerr| per event = {d['mean']:.4f} ± {d['sd']:.4f}", flush=True)
        print(f"    {'':>12s}  [generator, 4000 synthetic steps] sd(Δb)="
              f"{[round(v, 3) for v in d['gen_sd_increment']]} (design {cfg['scale']:.2f})  "
              f"corr(b_A,b_A')={d['gen_corr_regions']:+.3f}  "
              f"corr(Δb)={d['gen_corr_increments']:+.3f}  "
              f"b in [{d['gen_b_min']:.1f},{d['gen_b_max']:.1f}]  "
              f"clipped={d['gen_clip_frac']:.1%}", flush=True)
    dm = [damage[nm]["mean"] for nm, _ in arms]
    damage_spread = (max(dm) - min(dm)) / max(np.mean(dm), 1e-9)
    print(f"    spread across arms = {damage_spread:.1%} of the mean "
          f"({'PASS' if damage_spread < cfg['damage_tol'] else 'FAIL'} at "
          f"{cfg['damage_tol']:.0%} tolerance)", flush=True)
    # The analytic guarantee is on the INCREMENT variance, so check it directly -- and check it on the
    # GENERATOR's long synthetic run, not on the T-round trajectory. `--quick`'s 6 rounds give 5
    # differences, which cannot estimate an sd; reading that noise as a gate is how the reflecting-walk
    # bug got half-diagnosed. The generator is free to run for thousands of steps.
    sds = [v for nm, _ in arms for v in damage[nm]["gen_sd_increment"]]
    sd_spread = (max(sds) - min(sds)) / max(np.mean(sds), 1e-9)
    print(f"    sd(Δb) spread across arms/regions = {sd_spread:.1%} "
          f"({'PASS' if sd_spread < cfg['sd_tol'] else 'FAIL'} at {cfg['sd_tol']:.0%})", flush=True)

    # precompute each arm's per-round probe targets ONCE (shared across that arm's budgets)
    targets = {}
    for nm, _f in arms:
        targets[nm] = [[reexec(j, trajs[nm][0][t]) for j in range(K)] for t in range(T)]
    print(f"[probe] re-executed probe targets for {len(arms)} arms x {T} rounds x {K} regions",
          flush=True)

    if cfg.get("calibrate_only"):
        cal = {"config": cfg, "region_names": names, "region_visitation": reg_vis.tolist(),
               "geom_gate": geom_gate, "damage": damage, "damage_spread": float(damage_spread),
               "sd_spread": float(sd_spread), "b_traj": {nm: trajs[nm][0].tolist() for nm, _ in arms}}
        outdir = os.path.join(DATA_DIR, "metered_repair", "necessity_cal", cfg["tag"])
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(cal, fh, indent=2, cls=NumpyEncoder)
        volume.commit()
        ok = (geom_gate["pass_A"] and geom_gate["pass_Ap"] and geom_gate["pass_probe"]
              and damage_spread < cfg["damage_tol"] and sd_spread < cfg["sd_tol"])
        print(f"\n[calibrate] ALL GATES {'PASS' if ok else 'FAIL'} -- wrote {outdir}", flush=True)
        return {"results": cal}

    # ================================================================= #
    # ONE CELL = one (arm, budget). Fixed allocation: the whole budget into region A, every round.
    # ================================================================= #
    def run_cell(nm, budget):
        traj = trajs[nm][0]
        net = _mlp(cfg["seed"] + 40); net.load_state_dict(base_state)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["finetune_lr"])
        buf = None
        cap = max(cfg["buf_cap_min"], cfg["buf_mult"] * budget)
        npar = budget_n_par(budget)
        hist = []
        for t in range(T):
            b_state = traj[t]
            env = make_env(b_state)
            tg = targets[nm][t]
            # --- (1) BEFORE: post-event, pre-repair ---
            e_before = [fm_err(net, PROBE[j][0], PROBE[j][1], tg[j]) for j in range(K)]
            # --- (2) the METER: `budget` on-policy transitions toward A, and nowhere else ---
            rng_c = np.random.default_rng(cfg["seed"] + 11000 + 97 * t)
            cS, cU, cS2, cinfo = collect_toward(env, REG[0]["center"], budget, net, rng_c, 200,
                                                n_par=npar)
            shares = [float(in_region(cS, j).mean()) if len(cS) else float("nan") for j in range(K)]
            d = (cS, cU, cS2)
            buf = d if buf is None else tuple(np.concatenate([buf[i], d[i]])[-cap:] for i in range(3))
            # --- (3) repair. FIXED gradient steps across budgets (data is metered, compute is not) ---
            train_steps(net, opt, np.concatenate([rS, buf[0]]), np.concatenate([rU, buf[1]]),
                        np.concatenate([rS2, buf[2]]), cfg["finetune_steps"],
                        np.random.default_rng(cfg["seed"] + 12000 + t))
            # --- (4) AFTER ---
            e_after = [fm_err(net, PROBE[j][0], PROBE[j][1], tg[j]) for j in range(K)]
            rec = {"round": t, "b_state": b_state.tolist(),
                   "err_before": e_before, "err_after": e_after,
                   "repair": [e_before[j] - e_after[j] for j in range(K)],
                   "in_share": shares, "steps_used": int(cinfo["steps_used"]),
                   "n_transitions": int(len(cS)), "n_par": npar,
                   "n_episodes": int(cinfo["n_episodes"])}
            if (t % cfg["grade_every"] == 0) or (t == T - 1):
                rec["ballistic"] = rollout(make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                                                        np.random.default_rng(cfg["seed"] + 7001)),
                                           H, env)
            hist.append(rec)
            print(f"[{nm:>12s} S={budget:>4d} r{t:02d}] "
                  f"A {e_before[0]:.3f}->{e_after[0]:.3f} (repair {rec['repair'][0]:+.4f})   "
                  f"A' {e_before[1]:.3f}->{e_after[1]:.3f} (repair {rec['repair'][1]:+.4f})   "
                  + (f"ball={rec['ballistic']:.4f}  " if "ballistic" in rec else "")
                  + f"steps={rec['steps_used']} share_A={shares[0]:.2f} share_A'={shares[1]:.2f}",
                  flush=True)
        return hist

    budgets = cfg["budgets"]
    cells = {}
    for nm, _f in arms:
        for bg in budgets:
            key = f"{nm}|{bg}"
            print(f"\n=== cell {key} ===", flush=True)
            cells[key] = run_cell(nm, bg)

    # ================================================================= #
    # SUMMARY -- the transfer fraction, and its slope against log2(samples per event)
    # ================================================================= #
    def cell_summary(h_all):
        # BURN-IN. The first rounds are dominated by whatever manifold adaptation the base FM still
        # owes, which is global and lands on every region regardless of structure. Even with the base
        # FM now trained on-manifold, the first event's repair is the least comparable one, so it is
        # dropped rather than allowed to dominate a mean over few rounds.
        h = h_all[cfg["burn_in"]:] if len(h_all) > cfg["burn_in"] + 1 else h_all
        rp = np.array([r["repair"] for r in h], float)             # (rounds, K)
        eb = np.array([r["err_before"] for r in h], float)
        ea = np.array([r["err_after"] for r in h], float)
        su = float(np.mean([r["steps_used"] for r in h]))
        surf = float(rp[:, 0].mean()); tran = float(rp[:, 1].mean()); plac = float(rp[:, 2].mean())
        # PLACEBO-ADJUSTED transfer: subtract the repair measured at a region that is identical in
        # every way except that its gain never moved. What is left is repair attributable to tracking
        # a gain rather than to the model simply getting better everywhere.
        tran_adj = tran - plac
        bal = [r["ballistic"] for r in h if "ballistic" in r]
        return {"surface_repair": surf, "transfer_repair": tran, "placebo_repair": plac,
                "transfer_repair_adj": tran_adj,
                "transfer_frac": float(tran / surf) if abs(surf) > 1e-9 else float("nan"),
                "transfer_frac_adj": float(tran_adj / surf) if abs(surf) > 1e-9 else float("nan"),
                "err_A_mean": float(ea[:, 0].mean()), "err_Ap_mean": float(ea[:, 1].mean()),
                "err_placebo_mean": float(ea[:, 2].mean()),
                "err_A_before_mean": float(eb[:, 0].mean()),
                "err_Ap_before_mean": float(eb[:, 1].mean()),
                "n_rounds_used": len(h),
                "steps_per_event": su, "ballistic_auc": float(np.mean(bal)) if bal else float("nan"),
                "in_share_Ap": float(np.nanmean([r["in_share"][1] for r in h]))}

    summary = {f"{nm}|{bg}": cell_summary(cells[f"{nm}|{bg}"]) for nm, _f in arms for bg in budgets}

    print("\n=== NECESSITY: repair at the collected region (A) vs the held-out region (A') ===",
          flush=True)
    hdr = ("arm".rjust(12) + " " + "S".rjust(5) + " " + "steps".rjust(6) + " " + "surface".rjust(9)
           + " " + "transfer".rjust(9) + " " + "tr.frac".rjust(8) + " " + "A-err".rjust(7)
           + " " + "A'-err".rjust(7) + " " + "ball".rjust(7) + " " + "share A'".rjust(8))
    print("    " + hdr, flush=True)
    for nm, _f in arms:
        for bg in budgets:
            s = summary[f"{nm}|{bg}"]
            print(f"    {nm:>12s} {bg:>5d} {s['steps_per_event']:6.0f} {s['surface_repair']:+9.4f} "
                  f"{s['transfer_repair']:+9.4f} {s['transfer_frac']:8.3f} {s['err_A_mean']:7.3f} "
                  f"{s['err_Ap_mean']:7.3f} {s['ballistic_auc']:7.4f} {s['in_share_Ap']:8.3f}",
                  flush=True)
            print(f"    {'':>12s} {'':>5s} {'':>6s}  placebo repair {s['placebo_repair']:+.4f} -> "
                  f"raw tr.frac {s['transfer_frac']:.3f}, adjusted {s['transfer_frac_adj']:+.3f}",
                  flush=True)

    def slope(xs, ys):
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 2:
            return float("nan")
        return float(np.polyfit(xs[ok], ys[ok], 1)[0])

    slopes = {}
    lg = [np.log2(summary[f'{arms[0][0]}|{bg}']["steps_per_event"]) for bg in budgets]
    for nm, _f in arms:
        slopes[nm] = slope(lg, [summary[f"{nm}|{bg}"]["transfer_frac"] for bg in budgets])
    print("\n=== the migration: slope of transfer_frac against log2(samples per event) ===", flush=True)
    print("    PREDICTED NEGATIVE -- transfer should grow as data is starved.", flush=True)
    for nm, _f in arms:
        print(f"    {nm:>12s}  slope = {slopes[nm]:+.4f} / octave", flush=True)
    net_slope = float("nan")
    if "shared" in slopes and "local" in slopes:
        dif = [summary[f"shared|{bg}"]["transfer_frac"]
               - summary[f"local|{bg}"]["transfer_frac"] for bg in budgets]
        net_slope = slope(lg, dif)
        print(f"    shared - local (the controlled quantity): "
              + "  ".join(f"S={bg}:{d:+.3f}" for bg, d in zip(budgets, dif)), flush=True)
        print(f"    slope of (shared - local) = {net_slope:+.4f} / octave", flush=True)
    cal_ok = float("nan")
    if "pure_shared" in slopes:
        ps = [summary[f"pure_shared|{bg}"]["transfer_frac"] for bg in budgets]
        lo = [summary[f"local|{bg}"]["transfer_frac"] for bg in budgets] if "local" in slopes else None
        cal_ok = float(np.nanmean(ps) - (np.nanmean(lo) if lo else 0.0))
        print(f"\n    [CALIBRATION] pure_shared transfer_frac {np.nanmean(ps):+.3f} vs local "
              f"{np.nanmean(lo) if lo else float('nan'):+.3f}  ->  instrument dynamic range "
              f"{cal_ok:+.3f}. If this is ~0 the readout cannot see transfer AT ALL and no null "
              f"below it is interpretable.", flush=True)

    out = {"config": cfg, "region_names": names, "region_visitation": reg_vis.tolist(),
           "geom_gate": geom_gate, "damage": damage, "damage_spread": float(damage_spread),
           "sd_spread": float(sd_spread), "b_traj": {nm: trajs[nm][0].tolist() for nm, _f in arms},
           "budgets": budgets, "arms": arms, "cells": cells, "summary": summary,
           "slopes": slopes, "net_slope": net_slope, "calibration_range": cal_ok}
    outdir = os.path.join(DATA_DIR, "metered_repair", "necessity", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def necessity(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = _ALL_ARMS,
    budgets: str = _ALL_BUDGETS,
    rounds: int = 16,
    calibrate_only: bool = False,
    # --- the drift generator ---
    # 1.6, and RAISING IT WAS TRIED AND REVERTED. The reasoning was sound -- the placebo's generic
    # repair is the same size as the transfer signal, so a bigger event should lift the signal clear
    # of it -- and the measurement disagreed. At scale 2.2 with the band widened to [1,11], the walk
    # takes the world far from where the base FM was trained: A-err went 0.12 -> 0.70, the placebo's
    # own repair went 0.01 -> 0.13, and the instrument's dynamic range COLLAPSED from +0.136 to
    # +0.007. The transfer signal does not scale with the damage; the FM's chronic lag does.
    scale: float = 1.6,                       # per-event sd of EACH region's curl gain
    # theta=0.45 chosen offline: increments stay matched across arms at every value, and the
    # stationary sd (1.85) sits well inside the +-4 band so the safety clip fires on only ~3% of
    # steps, against 8% at theta=0.25. Clipping is matched across arms so it does not bias the
    # contrast, but a clipped event is a truncated event and fewer of them is strictly better.
    ou_theta: float = 0.45,                   # mean reversion; stationary sd ~ scale/sqrt(2*theta)
    share_load: str = "equal",                # "equal" | "varied" -- c_j loading on the shared latent
    b1: float = 6.0,
    b_lo: float = 2.0,
    b_hi: float = 10.0,
    # 14: the measured damage is a per-event average and the quick smoke's 4 events gave a 104%
    # spread across arms while the GENERATOR's own increment spread was 1.2%. The analytic
    # guarantee is holding; the measurement was just short. More events, no other change.
    damage_events: int = 14,
    damage_tol: float = 0.25,                 # max spread in measured per-event damage across arms
    sd_tol: float = 0.20,                     # max spread in sd(Δb) across arms/regions
    # --- geometry ---
    region_sigma: float = 0.12,
    region_k: float = 1.5,
    geom_samples: int = 1500,
    rad_min: float = 0.80,
    pref_width: float = 0.5,
    vis_steps: int = 12,
    off_vis_max: float = 0.12,
    on_vis_min: float = 0.40,
    min_sep: float = 0.30,
    # --- arm geometry (E2 / 4c-arm / E3 design point) ---
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,0.8,0.6",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    # --- on-policy collection ---
    ep_len: int = 14,
    op_q_range: float = 0.25,
    n_par: int = 16,
    sigma_u: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    collect_replan_every: int = 14,
    wrap_limit: float = 3.0,
    goal_jit: float = 0.05,
    buf_cap_min: int = 120,
    buf_mult: int = 2,                        # ring buffer = buf_mult * budget (scales WITH the meter)
    replay_n: int = 2500,
    replay_goals: int = 8,                    # eval goals to spread the replay pool over
    replay_chunk: int = 400,                  # on-policy transitions per replay goal
    replay_gate: float = 0.2,                 # exclude data inside the DRIFTING region gates
    replay_min: int = 300,                    # hard floor on the replay anchor
    finetune_lr: float = 3e-4,
    finetune_steps: int = 700,                # FIXED across budgets on purpose
    grade_every: int = 3,
    burn_in: int = 2,                         # rounds dropped from every cell summary
    base_pool_n: int = 4000,                  # competent on-policy reaches the base FM is trained on
    # --- FM ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
    probe_pool_n: int = 1200,                 # per collection CHUNK (see the adaptive loop)
    probe_target: int = 150,                  # in-region probe states to aim for per region
    probe_max_tries: int = 8,                 # chunks before giving up (then min_probe gates)
    # --- control eval ---
    n_eval: int = 24,
    plan_h: int = 14,
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    q_jit: float = 0.25,
    v0_std: float = 0.0,
    reach_amp: float = 1.2,
    reach_lo: float = 0.25,
    reach_hi: float = 0.50,
    reach_tries: int = 40,
    reach_pad: float = 0.90,                  # reach-distance filter, calibrated in E4's cert_v2
    min_density_frac: float = 0.01,           # min share of reach endpoints inside a region's gate
    min_probe: int = 40,                      # min in-region probe states per region (a HARD gate)
):
    import os

    arm_list = []
    for tok in arms.split(","):
        if not tok.strip():
            continue
        nm, f = tok.split(":")
        arm_list.append((nm.strip(), float(f)))
    bud = [int(x) for x in budgets.split(",") if x.strip()]
    if quick:
        rounds = 6; pool_n = 3000; fm_steps = 1500; finetune_steps = 300; replay_n = 1200
        fm_hidden = 128; fm_layers = 2; n_eval = 12; k_shoot = 256; cem_iters = 4
        collect_k_shoot = 128; collect_cem_iters = 3; probe_pool_n = 500; geom_samples = 600
        probe_target = 80; probe_max_tries = 6; base_pool_n = 1500; burn_in = 1
        replay_goals = 5; replay_chunk = 300; replay_min = 150
        damage_events = 4
        if budgets == _ALL_BUDGETS:
            bud = [144, 16]                                  # the two ends -- the cheapest migration
        if arms == _ALL_ARMS:
            arm_list = [("pure_shared", 1.0), ("local", 0.0)]  # the calibration contrast
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, arms=arm_list, budgets=bud, rounds=rounds,
        calibrate_only=calibrate_only, scale=scale, share_load=share_load,
        b1=b1, b_lo=b_lo, b_hi=b_hi, ou_theta=ou_theta, damage_events=damage_events, damage_tol=damage_tol, sd_tol=sd_tol,
        region_sigma=region_sigma, region_k=region_k, geom_samples=geom_samples, rad_min=rad_min,
        pref_width=pref_width, vis_steps=vis_steps, off_vis_max=off_vis_max, on_vis_min=on_vis_min,
        min_sep=min_sep,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, q_range=q_range,
        v_explore=v_explore, ep_len=ep_len, op_q_range=op_q_range, n_par=n_par, sigma_u=sigma_u,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        collect_replan_every=collect_replan_every, wrap_limit=wrap_limit, goal_jit=goal_jit,
        buf_cap_min=buf_cap_min, buf_mult=buf_mult, replay_n=replay_n,
        replay_goals=replay_goals, replay_chunk=replay_chunk, replay_gate=replay_gate,
        replay_min=replay_min,
        finetune_lr=finetune_lr, finetune_steps=finetune_steps, grade_every=grade_every, burn_in=burn_in,
        base_pool_n=base_pool_n,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        pool_n=pool_n, probe_pool_n=probe_pool_n, probe_target=probe_target,
        probe_max_tries=probe_max_tries,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, q_jit=q_jit, v0_std=v0_std,
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi, reach_tries=reach_tries,
        reach_pad=reach_pad, min_probe=min_probe, min_density_frac=min_density_frac,
    )
    out = run_necessity.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "necessity_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
