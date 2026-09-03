"""E3 -- the retracted directed-collection cut, re-attempted on the ON-POLICY ARM. The prize named
in [`README.md`](README.md) §Next-steps #1 and in [`../ballistic/directed/README.md`](../ballistic/directed/README.md)
§"What this cut needs before it can be re-attempted: on-policy collection."

Program: `ideas/two_timescale_value_loop.md`. Memo: [`../COLLECTION_REALISM.md`](../COLLECTION_REALISM.md).

WHAT THIS IS. The full inner-loop + outer-loop + FM setup:
  * INNER LOOP  -- a forward model re-adapted online from reward-free transitions, and a ballistic
                   (feedforward/open-loop) controller that consumes it. (Cut 4b/4c, `ballistic/arm`.)
  * OUTER LOOP  -- a reward-free VALUE signal `learning_progress x visitation` that chooses WHERE to
                   spend a scarce collection budget, round after round, against a MOVING drift.
                   (Cut 5 / `ballistic/directed` S1/S2.)
  * FM          -- the object the two loops share: the inner loop keeps it calibrated reward-free,
                   and where it is accurate is what the outer loop decides.

WHY IT HAD TO MOVE HERE. S2 (`ballistic/directed/directed_loop.py`) built exactly this loop on the
PUSHER under TELEPORT collection, and its central inner claim -- does the RELEVANCE (visitation)
term pay in a loop -- was RETRACTED by its own audit. The reason was structural, not a bug: collecting
was free teleportation (`set_state`) and MEASURING was free and global -- every round every policy
spent `probe_n*K + monitor_n*K = 2240` free teleported transitions, everywhere in the world, to
decide where to spend a budget of 100: a **22x measurement subsidy**. When looking is free and
global, the relevance term ("where should I even look") has no job -- it is demoted to a tiebreaker
on repair effort, and repair is cheap. So the question was unmeasurable, not answered.

THE FIX, WHICH IS THE WHOLE POINT OF THIS NODE. Data is a byproduct of behaviour (`embodied.py`,
`Body` -- metered, per-episode `MjData`, NO `set_state`). Two consequences make "where should I
practise?" a real, zero-sum, metered question:
  1. COLLECTING is on-policy: to gather data aimed at a region you must REACH there, paying episode
     steps. `collect_toward(region)` reaches toward a region's tip-space centre; the transitions it
     lives are the data.
  2. MONITORING is ALSO on-policy and CHARGED. The per-region survey that estimates learning progress
     is itself a set of on-policy reaches drawn from the same `Body` step budget -- so the monitor:
     collect ratio is O(1), not 22x, and it is reported every round. This is the subsidy, removed.
  Relevance now sits UPSTREAM of measurement (you cannot cheaply survey a region you never visit)
  instead of multiplying it afterwards, exactly as `COLLECTION_REALISM.md` §3 argued it must.

THE 2x2 (the design that makes the conjunction the only winner -- `ballistic/directed` S1). Regions
in tip space, crossing REDUCIBLE/IRREDUCIBLE with ON-REACH/OFF-REACH:
      A  reducible curl, ON  the eval reach  <- the right answer
      B  reducible curl, OFF the eval reach  <- catches lprog-only (learnable but irrelevant)
      C  aleatoric noise, ON  the eval reach <- catches visits-only / error-only (relevant, unlearnable)
      D  aleatoric noise, OFF the eval reach <- neither
`value = lprog x visits` is the only signal that lands on A alone: error-only fixates C/D, lprog-only
splits A/B, visits-only splits A/C. Reducible = a Gaussian-gated `curl_field` (E2's substrate,
compensable + aftereffect-capable); irreducible = a Gaussian-gated `noise_field` (fresh per substep,
so held-out error never falls). Both are LOCAL, so *where* you collect is what determines what you learn.

READOUT (the S2 audit's lesson: control is a near-blind grader -- grade by the value-relevant FM
error, and check whether the two instruments agree). Per round, per policy:
  * in-region-A FM prediction error       -- the SIGHTED grader (value-relevant reducible error)
  * per-region FM error B/C/D              -- the abandonment asymmetry (does value leave B/D wrong?)
  * ballistic + reactive control          -- the behavioural cash-out (ballistic-specific payoff)
  * allocation trace + in-region collection share  -- the mechanism
  * monitor_steps vs collect_steps        -- the anti-subsidy accounting (must be O(1))

Run:
    cd experiments/
    modal run mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --quick   # smoke
    for s in 0 1 2; do
      modal run --detach mjc/on_policy/directed_on_policy/directed_on_policy.py::directed_on_policy --tag ladder_s$s --seed $s --mon-n 40
    done
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# the full ladder, and the `--policies` default. Named so `--quick` can tell "the user asked for
# a specific policy" from "the user took the default" without changing either behaviour.
_ALL_POLICIES = "uniform,oracle,value,lprog-only,visits-only,error-only"


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_directed_on_policy(cfg: dict) -> dict:
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
    # GEOMETRY -- place the 2x2 regions in tip space, relative to the arm's reachable fan, and
    # bias the eval task toward the ON-reach direction so on/off-reach is a real distinction.
    # Everything reachable-by-construction (drawn from actual reach endpoints), verified by the
    # printed diagnostics (in-region collection share + eval visitation per region).
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
    radius = np.linalg.norm(cloud, axis=1)              # each endpoint's distance from the base
    ext = radius > cfg["rad_min"]                        # EXTENDED endpoints only -- see below
    # THE ON-REACH DIRECTION MUST BE TAME. Folded/inward reaches (small radius) are fast and
    # high-inertia-swing -- their baseline Δs and FM error are huge (~2.4), which SWAMPS the curl
    # drift signal (the thing the whole experiment measures) and makes learning-progress dominated by
    # fitting baseline dynamics rather than the drift. E2's `curl_center` sat at radius ~0.95 for
    # exactly this reason. So place every region in the EXTENDED band (moderate velocity), and choose
    # the on-reach direction as the densest angular sector AMONG EXTENDED endpoints.
    ang_all = np.arctan2(cloud[:, 1] - P0[1], cloud[:, 0] - P0[0])
    hcnt, edges = np.histogram(ang_all[ext], bins=24, range=(-np.pi, np.pi))
    theta_pref = 0.5 * (edges[np.argmax(hcnt)] + edges[np.argmax(hcnt) + 1])
    dth = np.angle(np.exp(1j * (ang_all - theta_pref)))
    ev_mask = (np.abs(dth) < cfg["pref_width"]) & ext
    G = cloud[ev_mask].mean(0)                          # eval-goal centroid (reaches END near here)

    # ACTUAL PATH VISITATION (FM-free). The reach tip PATH is curved (FK of a near-linear joint sweep
    # is curved in tip space), so a region's relevance is set by how much the eval reaches PASS
    # THROUGH it, not by its distance to the straight P0->G segment (which mislabelled swung-through
    # regions as off-reach). Approximate each on-reach eval reach by interpolating q0->qg in JOINT
    # space and FK-ing to a tip trajectory, then gate-sum occupancy at each candidate. This is the
    # same quantity the live `fm_visits` measures, computed here without a trained model.
    ev_q0, ev_qg = q0_cloud[ev_mask], qg_cloud[ev_mask]
    sig2 = 2.0 * cfg["region_sigma"] ** 2

    def path_visitation(cands):
        occ = np.zeros(len(cands))
        for a in np.linspace(0.0, 1.0, cfg["vis_steps"]):
            tips = fk(ev_q0 * (1 - a) + ev_qg * a, Ls)                  # (M, 2)
            dd = ((cands[:, None, 0] - tips[None, :, 0]) ** 2
                  + (cands[:, None, 1] - tips[None, :, 1]) ** 2)        # (C, M)
            occ += np.exp(-dd / sig2).sum(1)
        return occ / max(occ.max(), 1e-9)

    vis = path_visitation(cloud)
    G = cloud[ev_mask].mean(0)                          # eval-goal centroid = far end of on-reach

    def _nearest(pt):
        return float(np.linalg.norm(cloud - pt[None, :], axis=1).min())

    def _farthest_points(cands, m, seed):
        """Greedy farthest-point sampling: m well-separated points from a candidate set."""
        if len(cands) == 0 or m <= 0:
            return np.zeros((0, 2))
        rng = np.random.default_rng(seed)
        idx = [int(rng.integers(len(cands)))]
        while len(idx) < m and len(idx) < len(cands):
            d = np.min(np.stack([np.linalg.norm(cands - cands[i], axis=1) for i in idx]), 0)
            idx.append(int(np.argmax(d)))
        return cands[idx]

    # THE REGION SET -- SCARCITY AMONG DISTRACTORS, the on-policy S1/S2 ladder. On this arm the curl
    # is easy to learn (~a round's worth of data repairs a region), so with only one target the budget
    # is never binding and allocation is not a lever. S1 / curiosity Phase 2b: the value signal beats
    # uniform only when the reducible frontier is SMALL among MANY distractors. So:
    #   * one ON-REACH reducible target A (the genuinely MOST-visited extended point, where control
    #     depends on the model) -- the right place to spend budget;
    #   * `n_off_red` OFF-REACH REDUCIBLE distractors (low visitation, learnable-but-irrelevant) --
    #     catch lprog-only, which chases reducible structure without asking whether it matters;
    #   * `n_off_noise` OFF-REACH IRREDUCIBLE-NOISE distractors (low visitation, high but UNLEARNABLE
    #     error) -- the noisy-TV trap; catch error-only, which chases raw prediction error.
    # (An ON-REACH noise decoy -- S1/S2's region C, to separate value from visits-only -- is not
    # placeable on-policy: visitation concentrates on the target itself, so there is no
    # "visited-but-irrelevant" territory. `n_on_noise` is kept but is ~always empty here; that
    # emptiness is itself the on-policy-vs-teleport finding.)
    # All extended (tame), classified by ACTUAL path visitation, well-separated (farthest-point).
    ext_idx = np.flatnonzero(ext)
    A_c = cloud[ext_idx[np.argmax(vis[ext_idx])]]      # the genuinely most-visited extended point
    far_from_A = np.linalg.norm(cloud - A_c[None, :], axis=1) > cfg["min_sep"]
    REG = [dict(name="A", center=A_c, kind="curl", reducible=True, on_reach=True)]
    # off-reach distractors (reducible + noise) sampled together so they stay mutually well-separated
    off_cands = cloud[ext & (vis < cfg["off_vis_max"]) & far_from_A]
    off_pts = _farthest_points(off_cands, cfg["n_off_red"] + cfg["n_off_noise"], cfg["seed"] + 30)
    for i, pt in enumerate(off_pts):
        if i < cfg["n_off_red"]:
            REG.append(dict(name=f"Boff{i + 1}", center=pt, kind="curl", reducible=True, on_reach=False))
        else:
            REG.append(dict(name=f"Doff{i - cfg['n_off_red'] + 1}", center=pt, kind="noise",
                            reducible=False, on_reach=False))
    on_cands = cloud[ext & (vis > cfg["on_vis_min"]) & far_from_A]
    for i, pt in enumerate(_farthest_points(on_cands, cfg["n_on_noise"], cfg["seed"] + 31)):
        REG.append(dict(name=f"Con{i + 1}", center=pt, kind="noise", reducible=False, on_reach=True))
    K = len(REG)
    for r in REG:
        r["sigma"] = cfg["region_sigma"]
    names = [r["name"] for r in REG]
    reducible = [j for j in range(K) if REG[j]["reducible"]]
    reg_vis = path_visitation(np.array([np.asarray(r["center"]) for r in REG]))
    print(f"[geom] P0={P0.round(3).tolist()} (r={np.linalg.norm(P0):.2f}) "
          f"A(most-visited)={A_c.round(3).tolist()} (r={np.linalg.norm(A_c):.2f}) "
          f"K={K} regions ({len(reducible)} reducible)", flush=True)
    for j, r in enumerate(REG):
        c = np.asarray(r["center"])
        print(f"[geom]   {r['name']}: center={c.round(3).tolist()} r={np.linalg.norm(c):.2f} "
              f"visitation={reg_vis[j]:.2f} "
              f"kind={r['kind']} reducible={r['reducible']} on_reach={r['on_reach']} "
              f"nearest-cloud={_nearest(c):.3f}m", flush=True)

    def gate_np(tips, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((tips[..., 0] - c[0]) ** 2 + (tips[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    def in_region(states, j):
        tips = fk(np.asarray(states)[:, :n].astype(np.float64), Ls)
        return np.linalg.norm(tips - REG[j]["center"], axis=1) < cfg["region_k"] * REG[j]["sigma"]

    # ================================================================= #
    # ENV -- one plant, current per-region drift gains folded into curl_fields / noise_fields.
    # ================================================================= #
    def env_dgp(b_state):
        """The fully-resolved DGP knob dict for a given per-region drift state. Split out of
        `make_env` so a render pack can carry the EXACT plant it was recorded on (no
        re-derivation, no drift between the loop and the renderer)."""
        curls, noises = [], []
        for j, r in enumerate(REG):
            if r["kind"] == "curl":
                curls.append({"b": float(b_state[j]), "center": tuple(np.asarray(r["center"]).tolist()),
                              "sigma": r["sigma"]})
            else:
                noises.append({"amp": float(cfg["noise_amp"]), "center": tuple(np.asarray(r["center"]).tolist()),
                               "sigma": r["sigma"]})
        dgp = dict(n_links=n, link_lengths=cfg["link_lengths"][:n], link_masses=cfg["link_masses"][:n],
                   joint_damping=cfg["joint_damping"], gear=cfg["gear"], noise_seed=cfg["seed"] + 999)
        if curls:
            dgp["curl_fields"] = curls
        if noises:
            dgp["noise_fields"] = noises
        return dgp

    def make_env(b_state):
        return ArmEnv(env_dgp(b_state))

    # ================================================================= #
    # FM  f(s,u) -> Δs   (identical architecture/training to E2 readapt_local.py)
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
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs, qc, cfg["q_range"], cfg["v_explore"])

    # norm stats from a broad teleport pool of the DRIFTED world (off-budget: the experimenter's
    # ruler, exactly as E2). b_state at round 0 = reducible regions drifted.
    b0_state = np.array([cfg["b1"] if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    env_drift0 = make_env(b0_state)
    nS, nU, nS2 = teleport_pool(env_drift0, cfg["pool_n"], cfg["seed"] + 11)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=np.concatenate([nS, nU], 1).mean(0), sx=np.concatenate([nS, nU], 1).std(0) + 1e-6,
        my=(nS2 - nS).mean(0), sy=(nS2 - nS).std(0) + 1e-6).items()}

    # ================================================================= #
    # CEM planner + eval rollout (reach toward a Cartesian goal; cost = tip-to-goal)
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
        pick = np.argsort(d)[:B]                          # the B most on-reach endpoints
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

    def fm_visits(net):
        """Region occupancy of the ballistic eval plan rolled through the FM ITSELF (no env, no
        budget) -- the relevance signal is purely internal (S1: the FM tells you where you'll be)."""
        plan = make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                            np.random.default_rng(cfg["seed"] + 7001))(ev_starts, ev_goals)
        v = np.zeros(K)
        with torch.no_grad():
            s = torch.tensor(ev_starts, device=device)
            for h in range(H):
                tips = fk_torch(s[:, :n]).cpu().numpy()
                for j in range(K):
                    v[j] += float(gate_np(tips, j).sum())
                x = torch.cat([s, torch.tensor(plan[:, h, :], device=device)], 1)
                s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
        return v / max(v.sum(), 1e-9)

    # ================================================================= #
    # RENDER PACK (off by default) -- everything `render_ladder.py` needs to replay and draw
    # this round's ballistic reaches, including the FM's OWN forecast of them.
    # ================================================================= #
    render_packs = []
    render_rounds = set(cfg.get("render_rounds", []))
    render_pols = set(cfg.get("render_policies", []))

    def render_pack(pname, t, net, b_state):
        """Record the ballistic reaches this FM plans, what the body actually does, and what the
        FM *thought* would happen -- the predicted-vs-actual divergence IS the thing that shrinks
        as the model is re-calibrated, and it is invisible in any scalar we already log.

        SIDE-EFFECT-FREE BY CONSTRUCTION, which is the whole reason this is safe to bolt onto a
        published loop: it uses a FRESH `ArmEnv` (hence its own `_noise_rng`, so the graded
        rollout's noise draws are untouched) and FRESH plan generators (`make_plan_fn` seeds its
        own), and it never trains. With `--render-rounds ""` (the default) it is not called at
        all, so `ladder_s{0,1,2}` reproduce exactly.

        The replay is ONE CONTINUOUS EPISODE per reach rather than `rollout()`'s multiplexed
        `set_state`-per-step -- exact for the contactless arm (`../README.md` §machinery), modulo
        the float32 state round-trip the multiplexed idiom incurs (~1e-7).
        """
        # Record EVERY eval reach by default (`render_n=0`). The point is that the pack's `miss` is
        # then the same 40-reach median the loop grades as `ballistic_cem`, not a 4-reach subsample
        # of it -- those disagree badly (at one round, 9.1 cm over 4 reaches vs 4.9 cm over 40), and
        # a headline number in a video must be the real metric. The video renders only the first
        # few; recording the rest is nearly free.
        R = B if int(cfg["render_n"]) <= 0 else min(int(cfg["render_n"]), B)
        renv = make_env(b_state)
        plan = make_plan_fn(net, cfg["k_shoot"], cfg["cem_iters"],
                            np.random.default_rng(cfg["seed"] + 7001))(ev_starts, ev_goals)
        starts, goals, acts = ev_starts[:R], ev_goals[:R], plan[:R]

        actual = np.zeros((R, H + 1, SD), np.float64)          # what the BODY does
        for b in range(R):
            renv.set_state(starts[b, :n].astype(np.float64), starts[b, n:].astype(np.float64))
            actual[b, 0] = starts[b]
            for h in range(H):
                actual[b, h + 1], _ = renv.step(acts[b, h], fs)

        with torch.no_grad():                                   # what the MODEL expected
            s = torch.tensor(starts, device=device)
            pred = [s.cpu().numpy().copy()]
            for h in range(H):
                x = torch.cat([s, torch.tensor(acts[:, h, :], device=device)], 1)
                s = s + (net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                pred.append(s.cpu().numpy().copy())
            pred = np.stack(pred, 1).astype(np.float64)

        a_tip = fk(actual[..., :n], Ls)
        p_tip = fk(pred[..., :n], Ls)
        per_miss = np.linalg.norm(a_tip[:, -1] - goals, axis=1)
        per_div = np.linalg.norm(p_tip - a_tip, axis=2).mean(1)
        miss = float(np.median(per_miss))
        div = float(per_div.mean())
        print(f"[render] pack {pname} r{t}: R={R} miss={miss:.4f} model-vs-body tip div={div:.4f}",
              flush=True)
        return {"policy": pname, "round": int(t), "b_state": np.asarray(b_state).tolist(),
                "dgp": env_dgp(b_state), "starts": starts.tolist(), "goals": goals.tolist(),
                "actions": acts.tolist(), "actual_states": actual.tolist(),
                "actual_tips": a_tip.tolist(), "pred_tips": p_tip.tolist(),
                "miss": miss, "tip_divergence": div,
                "per_miss": per_miss.tolist(), "per_div": per_div.tolist()}

    # ================================================================= #
    # ON-POLICY reach-toward-a-region collection (the metered acquisition primitive)
    # ================================================================= #
    def make_fixed_goal_sampler(center, jit):
        c = np.asarray(center, np.float32)

        def sample(states, rng):
            m = len(np.atleast_2d(np.asarray(states)))
            return (c[None, :] + rng.uniform(-jit, jit, (m, 2))).astype(np.float32)
        return sample

    def collect_toward(cenv, center, need, fm, rng_buf, k):
        """Reach toward `center` under the current FM; return the transitions the body lived +
        info (steps_used). One env.step per transition, so `need` == steps charged."""
        plan_fn = make_plan_fn(fm, cfg["collect_k_shoot"], cfg["collect_cem_iters"],
                               np.random.default_rng(cfg["seed"] + 8000 + k))
        beh = ReachBehaviour(plan_fn, np.zeros((cfg["n_par"], 2), np.float32), rng_buf,
                             sigma_u=cfg["sigma_u"], replan_every=cfg["collect_replan_every"])
        gs = make_fixed_goal_sampler(center, cfg["goal_jit"])
        S, U, S2, info = collect_pool(cenv, need, rng_buf, fs, qc, cfg["op_q_range"], None,
                                      collection_mode="on_policy", behaviour=beh, goal_sampler=gs,
                                      ep_len=cfg["ep_len"], n_par=cfg["n_par"], v0_std=cfg["v0_std"],
                                      wrap_limit=cfg["wrap_limit"], return_info=True)
        return S, U, S2, info

    # ================================================================= #
    # BASE (stale) FM -- trained off-budget on the PRE-drift world (b=0 in reducible regions; the
    # noise regions are permanent). This is the innate/pre-episode model every policy starts from.
    # ================================================================= #
    b_pre = np.array([0.0 if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    env_pre = make_env(b_pre)
    S0_, U0_, S20_ = teleport_pool(env_pre, cfg["pool_n"], cfg["seed"] + 10)
    fm_base = _mlp(cfg["seed"] + 40)
    train_steps(fm_base, torch.optim.Adam(fm_base.parameters(), lr=cfg["fm_lr"]),
                S0_, U0_, S20_, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    base_state = copy.deepcopy(fm_base.state_dict())
    # replay buffer: out-of-EVERY-region base transitions (valid post-drift; keeps the counterfactual
    # fit from smearing a local rotation globally -- S0's lesson).
    gmax = np.max(np.stack([gate_np(fk(S0_[:, :n].astype(np.float64), Ls), j) for j in range(K)]), 0)
    oidx = np.flatnonzero(gmax < 0.02)
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(oidx)[:cfg["replay_n"]]
    rS, rU, rS2 = S0_[rp], U0_[rp], S20_[rp]
    print(f"[pretrain] base FM trained on pre-drift pool; replay={len(rp)} out-of-region transitions",
          flush=True)

    # ================================================================= #
    # CONTINUOUS DRIFT (COLLECTION_REALISM.md §4): a reflecting random walk on each reducible
    # region's curl gain, so the field is NEVER permanently repaired and allocation stays zero-sum
    # round after round -- the regime cut 4a named as the only one where a value loop earns anything.
    # The trajectory is precomputed ONCE and SHARED across every policy, so the only thing that
    # varies between policies is where they spend the budget (controlled). A single scheduled drift
    # (the S2 style) is available via drift_mode="schedule" but converges once repaired.
    # ================================================================= #
    drift_rng = np.random.default_rng(cfg["seed"] + 77)
    b_ref = np.array([cfg["b1"] if REG[j]["reducible"] else 0.0 for j in range(K)], float)
    b_traj = np.zeros((T, K))
    b_cur = b_ref.copy()
    schedule = {}
    for t in range(T):
        b_traj[t] = b_cur.copy()
        if cfg["drift_mode"] == "ou":
            for j in reducible:
                nb = b_cur[j] + cfg["ou_sigma"] * drift_rng.standard_normal()
                if nb < cfg["b_lo"]:
                    nb = 2 * cfg["b_lo"] - nb                    # reflecting bounds keep it moving
                if nb > cfg["b_hi"]:
                    nb = 2 * cfg["b_hi"] - nb
                b_cur[j] = float(np.clip(nb, cfg["b_lo"], cfg["b_hi"]))
        else:                                                    # "schedule": re-drift periodically
            if t > 0 and t % cfg["drift_every"] == 0:
                cyc = (t // cfg["drift_every"] - 1)
                cycle = cfg["drift_cycle"] or reducible
                j = cycle[cyc % len(cycle)]
                b_cur[j] = float(drift_rng.choice([-1.0, 1.0]) * cfg["b1"])
                schedule[t] = (j, b_cur[j])
    print(f"[setup] drift_mode={cfg['drift_mode']} b_traj[A] over rounds: "
          + " ".join(f"{b_traj[t, 0]:.1f}" for t in range(min(T, 12))), flush=True)

    def _norm_w(v):
        v = np.maximum(np.asarray(v, float), 0.0)
        return v / v.sum() if v.sum() > 1e-12 else np.full(K, 1.0 / K)

    # ================================================================= #
    # one policy's full T-round loop
    # ================================================================= #
    def run_policy(pname):
        net = _mlp(cfg["seed"] + 40); net.load_state_dict(base_state)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["finetune_lr"])
        bufs = {j: None for j in range(K)}
        hist = []
        # round "-1" = the STALE base FM, before this policy has collected anything, graded against
        # the round-0 drifted world. The strongest "early" frame there is: every round >= 0 has
        # already been fine-tuned once on that round's collection.
        if pname in render_pols and -1 in render_rounds:
            render_packs.append(render_pack(pname, -1, net, b_traj[0]))
        for t in range(T):
            b_state = b_traj[t]                                  # shared drift; only alloc varies
            if t in schedule:
                bufs[schedule[t][0]] = None                     # (schedule mode) voided on re-drift
            env = make_env(b_state)
            mon_steps = 0; coll_steps = 0

            # ---------- METERED on-policy survey: reach each region, gather signals ----------
            per_err = np.zeros(K); lprog = np.zeros(K)
            uses_lp = pname in ("value", "value-floor", "lprog-only", "error-only")
            cur = copy.deepcopy(net.state_dict()) if uses_lp else None
            for j in range(K):
                mS, mU, mS2, minfo = collect_toward(env, REG[j]["center"], cfg["mon_n"], net,
                                                    np.random.default_rng(cfg["seed"] + 9000 + 41 * t + j), 100 + j)
                mon_steps += int(minfo["steps_used"])
                m_in = in_region(mS, j)                         # signal read on the IN-region subset
                if m_in.sum() >= cfg["min_in"]:
                    iS, iU, iS2 = mS[m_in], mU[m_in], mS2[m_in]
                else:                                           # too few in-region samples: fall back
                    iS, iU, iS2 = mS, mU, mS2
                per_err[j] = fm_err(net, iS, iU, iS2)
                if uses_lp:
                    h = len(iS) // 2
                    if h >= 2:
                        eb = fm_err(net, iS[h:], iU[h:], iS2[h:])
                        pr = _mlp(cfg["seed"] + 40); pr.load_state_dict(cur)
                        train_steps(pr, torch.optim.Adam(pr.parameters(), lr=cfg["finetune_lr"]),
                                    np.concatenate([rS, iS[:h]]), np.concatenate([rU, iU[:h]]),
                                    np.concatenate([rS2, iS2[:h]]), cfg["lp_steps"],
                                    np.random.default_rng(cfg["seed"] + 9500 + 41 * t + j))
                        ea = fm_err(pr, iS[h:], iU[h:], iS2[h:])
                        lprog[j] = max(eb - ea, 0.0)
            visits = fm_visits(net)
            # ground-truth "stale AND reducible" for the ORACLE (privileged) only
            stale_mask = np.array([1.0 if (REG[j]["reducible"] and per_err[j] > cfg["err_floor"])
                                   else 0.0 for j in range(K)])

            # ---------- the policy: WHERE to spend this round's collection budget ----------
            if pname == "uniform":
                w = None
            elif pname == "oracle":
                tgt = stale_mask * per_err * visits
                w = _norm_w(tgt) if tgt.sum() > 1e-9 else None
            elif pname == "value":
                tv = lprog * visits
                w = _norm_w(tv) if tv.max() >= cfg["value_floor"] else None    # no-op floor (S2 lesson)
            elif pname == "lprog-only":
                w = _norm_w(lprog) if lprog.sum() > 1e-12 else None
            elif pname == "visits-only":
                w = _norm_w(visits)
            elif pname == "error-only":
                w = _norm_w(per_err)
            else:
                raise ValueError(pname)

            rng_c = np.random.default_rng(cfg["seed"] + 11000 + 97 * t)
            in_share = np.full(K, np.nan)
            if w is None:                                       # uniform: spread over all regions
                counts = np.full(K, cfg["budget"] // K, int); counts[0] += cfg["budget"] - counts.sum()
            else:
                counts = np.floor(w * cfg["budget"]).astype(int)
                counts[int(np.argmax(w))] += cfg["budget"] - counts.sum()
            alloc = counts.astype(float)
            for j in range(K):
                if counts[j] <= 0:
                    continue
                cS, cU, cS2, cinfo = collect_toward(env, REG[j]["center"], int(counts[j]), net, rng_c, 200 + j)
                coll_steps += int(cinfo["steps_used"])
                in_share[j] = float(in_region(cS, j).mean())
                d = (cS, cU, cS2)
                bufs[j] = d if bufs[j] is None else tuple(
                    np.concatenate([bufs[j][i], d[i]])[-cfg["buf_cap"]:] for i in range(3))

            # ---------- fine-tune the FM on everything currently believed valid ----------
            parts = [(rS, rU, rS2)] + [b for b in bufs.values() if b is not None]
            train_steps(net, opt, np.concatenate([p[0] for p in parts]),
                        np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
                        cfg["finetune_steps"], np.random.default_rng(cfg["seed"] + 12000 + t))

            # ---------- grade (off-budget, the experimenter's fixed instrument) ----------
            snap = copy.deepcopy(net)
            # per-region FM error on a fixed matched-reach probe set (built once, below)
            reg_err = [fm_err(snap, PB[j][0], PB[j][1], PB[j][2]) if len(PB[j][0]) else float("nan")
                       for j in range(K)]
            if pname in render_pols and t in render_rounds:
                render_packs.append(render_pack(pname, t, snap, b_state))
            bal = rollout(make_plan_fn(snap, cfg["k_shoot"], cfg["cem_iters"],
                                       np.random.default_rng(cfg["seed"] + 7001)), H, env)
            rec = {"round": t, "alloc": alloc.tolist(), "in_share": in_share.tolist(),
                   "per_err_mon": per_err.tolist(), "reg_err": reg_err, "visits": visits.tolist(),
                   "lprog": lprog.tolist(), "stale_mask": stale_mask.tolist(), "b_state": b_state.tolist(),
                   "ballistic_cem": bal, "mon_steps": mon_steps, "coll_steps": coll_steps}
            if (t % cfg["reactive_every"] == 0) or (t == T - 1):
                rec["reactive"] = rollout(make_plan_fn(snap, cfg["k_shoot"], cfg["cem_iters"],
                                                       np.random.default_rng(cfg["seed"] + 7000)), 1, env)
            hist.append(rec)
            astr = "/".join(str(int(x)) for x in alloc)
            print(f"[{pname:>12s} r{t:02d}] alloc={astr:>16s} ball={bal:.4f}"
                  + (f" reac={rec['reactive']:.4f}" if "reactive" in rec else "")
                  + f"  mon/coll={mon_steps}/{coll_steps}"
                  + "  regERR=[" + " ".join(f"{e:.3f}" for e in reg_err) + "]", flush=True)
        return hist

    # ---- fixed per-region PROBE sets (off-budget grader): matched-reach transitions in each region,
    # gathered with a ref FM trained on the DRIFTED world (so the reaches are competent, not deflected
    # by a stale model) and partitioned by region. Plus a matched-FM CEILING (trained on abundant
    # competent on-reach reaches) to normalise the value-relevant region-A error.
    print("[probe] training ref FM (drifted world) + building per-region grader probes ...", flush=True)
    ref_net = _mlp(cfg["seed"] + 43)
    dS, dU, dS2 = teleport_pool(env_drift0, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                dS, dU, dS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 401))
    PB = []
    probeS, probeU, probeS2, _ = collect_toward(env_drift0, G, cfg["probe_pool_n"], ref_net,
                                                np.random.default_rng(cfg["seed"] + 610), 900)
    # populate the off-reach / distractor probes by reaching toward each non-A region centre too
    for j in range(1, K):
        oS, oU, oS2, _ = collect_toward(env_drift0, REG[j]["center"], cfg["probe_pool_n"] // 2, ref_net,
                                        np.random.default_rng(cfg["seed"] + 901 + j), 901 + j)
        probeS = np.concatenate([probeS, oS]); probeU = np.concatenate([probeU, oU])
        probeS2 = np.concatenate([probeS2, oS2])
    for j in range(K):
        m = in_region(probeS, j)
        PB.append((probeS[m], probeU[m], probeS2[m]))
        print(f"[probe]   region {names[j]}: {int(m.sum())} probe transitions", flush=True)
    # matched ceiling: FM trained on abundant competent ON-REACH reaches in the drifted world. Valid
    # floor for the on-reach regions (A, C); undertrained by construction for the off-reach regions
    # (B, D) -- you never reach there -- so those are reported but caveated (E2's lesson).
    ceilS, ceilU, ceilS2, _ = collect_toward(env_drift0, G, cfg["ceil_pool_n"], ref_net,
                                             np.random.default_rng(cfg["seed"] + 620), 910)
    ceil_net = _mlp(cfg["seed"] + 44)
    train_steps(ceil_net, torch.optim.Adam(ceil_net.parameters(), lr=cfg["fm_lr"]),
                ceilS, ceilU, ceilS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 421))
    ceil_err = [fm_err(ceil_net, PB[j][0], PB[j][1], PB[j][2]) if len(PB[j][0]) else float("nan")
                for j in range(K)]
    print("[probe] matched-ceiling per-region err: "
          + " ".join(f"{names[j]}={ceil_err[j]:.3f}" for j in range(K)), flush=True)

    results = {}
    for pname in cfg["policies"]:
        print(f"\n=== policy: {pname} ===", flush=True)
        results[pname] = run_policy(pname)

    # ================================================================= #
    # summary
    # ================================================================= #
    def auc(pname, key):
        v = [r[key] for r in results[pname] if key in r and r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    def reg_auc(pname, j):
        v = [r["reg_err"][j] for r in results[pname] if r["reg_err"][j] == r["reg_err"][j]]
        return float(np.mean(v)) if v else float("nan")

    a_idx = names.index("A") if "A" in names else 0
    print("\n=== summary: mean over rounds (lower=better) ===", flush=True)
    print(f"    {'policy':>12s}  {'ballistic':>9s} {'reactive':>9s} {'A-err':>7s} "
          + " ".join(f"{nm}-err" for nm in names), flush=True)
    summary = {}
    for pname in cfg["policies"]:
        row = {"ballistic_auc": auc(pname, "ballistic_cem"), "reactive_auc": auc(pname, "reactive"),
               "regA_err_auc": reg_auc(pname, a_idx),
               "reg_err_auc": [reg_auc(pname, j) for j in range(K)],
               "mon_steps_total": int(sum(r["mon_steps"] for r in results[pname])),
               "coll_steps_total": int(sum(r["coll_steps"] for r in results[pname])),
               "final_reg_err": results[pname][-1]["reg_err"]}
        row["monitor_collect_ratio"] = row["mon_steps_total"] / max(row["coll_steps_total"], 1)
        summary[pname] = row
        print(f"    {pname:>12s}  {row['ballistic_auc']:9.4f} {row['reactive_auc']:9.4f} "
              f"{row['regA_err_auc']:7.4f} " + " ".join(f"{e:6.3f}" for e in row["reg_err_auc"]), flush=True)

    # the anti-subsidy headline: monitor:collect ratio is O(1), not the 22x it was under teleport
    mr = np.nanmean([summary[p]["monitor_collect_ratio"] for p in cfg["policies"]])
    print(f"\n[anti-subsidy] mean monitor:collect step ratio = {mr:.2f}  "
          f"(S2 teleport was 22x; on-policy survey is metered)", flush=True)

    # the relevance test: does concentrating on the value-relevant region A beat spreading, and does
    # a value/relevance-aware policy ABANDON the off-reach distractors (leave their error high while
    # keeping A low)? off-err = mean FM error over the off-reach reducible distractors.
    off_idx = [j for j in range(K) if not REG[j]["on_reach"] and REG[j]["reducible"]]
    if off_idx:
        def off_err_auc(pname):
            return float(np.nanmean([reg_auc(pname, j) for j in off_idx]))
        print("\n=== relevance test (the retracted S2 claim, now metered) ===", flush=True)
        print(f"    (off-reach reducible distractors: {[names[j] for j in off_idx]})", flush=True)
        for p in cfg["policies"]:
            summary[p]["off_err_auc"] = off_err_auc(p)
            print(f"    {p:>12s}: A-err={summary[p]['regA_err_auc']:.4f}  "
                  f"off-err={off_err_auc(p):.4f}  ballistic={summary[p]['ballistic_auc']:.4f}", flush=True)

    if "uniform" in results and "oracle" in results:
        for key, lbl in (("ballistic_cem", "ballistic"), ("regA_err_auc", "A-FM-err")):
            u = auc("uniform", "ballistic_cem") if key == "ballistic_cem" else summary["uniform"]["regA_err_auc"]
            o = auc("oracle", "ballistic_cem") if key == "ballistic_cem" else summary["oracle"]["regA_err_auc"]
            den = u - o
            print(f"\n  [{lbl}] uniform={u:.4f} oracle={o:.4f} spread={den:+.4f}", flush=True)
            for pname in cfg["policies"]:
                val = auc(pname, "ballistic_cem") if key == "ballistic_cem" else summary[pname]["regA_err_auc"]
                g = (u - val) / den if abs(den) > 1e-9 else float("nan")
                summary[pname][lbl + "_gap_closed"] = g
                print(f"      {pname:>12s}  gap-closed={g:+.2f}", flush=True)

    # ceiling-normalised region-A recovery (the value-relevant, sighted readout): fraction of the
    # stale->matched gap closed on region A, averaged over rounds. 1.0 = matched-FM competence.
    if "A" in names:
        cA = ceil_err[a_idx]
        for pname in cfg["policies"]:
            st = results[pname][0]["reg_err"][a_idx]
            den = st - cA
            rec = [((st - r["reg_err"][a_idx]) / den if abs(den) > 1e-9 else float("nan"))
                   for r in results[pname]]
            summary[pname]["regA_recovery_auc"] = float(np.nanmean(rec))
        print(f"\n=== region-A ceiling-normalised recovery (ceil={ceil_err[a_idx]:.3f}, "
              f"higher=better) ===", flush=True)
        for pname in cfg["policies"]:
            print(f"    {pname:>12s}  {summary[pname]['regA_recovery_auc']:+.2f}", flush=True)

    out = {"config": cfg, "region_names": names,
           "region_centers": {r["name"]: np.asarray(r["center"]).tolist() for r in REG},
           "schedule": {str(k): v for k, v in schedule.items()},
           "P0": P0.tolist(), "G": G.tolist(), "ceil_err": ceil_err,
           "results": results, "summary": summary}
    outdir = os.path.join(DATA_DIR, "directed_on_policy", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)

    # render packs go to their OWN file so `results.json` stays byte-comparable with the
    # published ladder_s{0,1,2} artifacts (`render_ladder.py` reads this one).
    render = None
    if render_packs:
        render = {"meta": {"tag": cfg["tag"], "seed": cfg["seed"], "frame_skip": fs, "plan_H": H,
                           "n_links": n, "link_lengths": Ls.tolist(), "region_k": cfg["region_k"],
                           "P0": P0.tolist(), "G": G.tolist(), "ceil_err": ceil_err,
                           "region_names": names, "ceil_regA": ceil_err[a_idx],
                           "regions": [{"name": r["name"], "center": np.asarray(r["center"]).tolist(),
                                        "sigma": r["sigma"], "kind": r["kind"],
                                        "reducible": bool(r["reducible"]),
                                        "on_reach": bool(r["on_reach"])} for r in REG]},
                  "packs": render_packs}
        with open(os.path.join(outdir, "render_pack.json"), "w") as fh:
            json.dump(render, fh, cls=NumpyEncoder)
        print(f"[save] wrote {len(render_packs)} render packs to {outdir}/render_pack.json", flush=True)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return {"results": out, "render": render}


@app.local_entrypoint()
def directed_on_policy(
    quick: bool = False,
    spawn: bool = False,                      # queue server-side and return; see below

    tag: str = "",
    seed: int = 0,
    policies: str = _ALL_POLICIES,
    rounds: int = 30,
    drift_mode: str = "ou",                   # "ou" = continuous walk (default); "schedule" = S2-style
    ou_sigma: float = 1.6,                     # per-round std of the curl-gain random walk
    b_lo: float = 2.0,
    b_hi: float = 10.0,
    drift_every: int = 8,                     # (schedule mode only)
    drift_cycle: str = "0,0,1",              # (schedule mode only) region indices among `reducible`
    b1: float = 6.0,                          # curl gain reference (E2's design point)
    noise_amp: float = 3.0,                   # aleatoric torque noise in the C/D regions (< gear=8)
    # --- geometry ---
    region_sigma: float = 0.12,
    region_k: float = 1.5,                    # in-region iff ‖tip-center‖ < region_k*sigma
    geom_samples: int = 1500,
    rad_min: float = 0.80,                     # regions must sit beyond this radius (extended/tame)
    pref_width: float = 0.5,                  # half-width (rad) of the on-reach eval sector
    vis_steps: int = 12,                      # tip-path interpolation steps for path-visitation
    off_vis_max: float = 0.12,                # off-reach distractors: path visitation below this
    on_vis_min: float = 0.40,                 # on-reach noise distractors: path visitation above this
    min_sep: float = 0.30,                    # min tip-space separation of any distractor from A
    n_off_red: int = 3,                       # OFF-reach REDUCIBLE distractors (catch lprog-only)
    n_off_noise: int = 2,                     # OFF-reach NOISE distractors (catch error-only / noisy-TV)
    n_on_noise: int = 0,                      # ON-reach noise (rarely placeable on-policy; see code)
    # --- arm geometry (E2 / 4c-arm design point) ---
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
    # --- per-round loop budget (SCARCE, on purpose: repair must be incomplete so allocation is
    #     zero-sum; monitoring is CHARGED separately) ---
    budget: int = 120,                        # directed collection transitions per round
    mon_n: int = 90,                          # survey transitions per region per round (metered)
    min_in: int = 8,                          # min in-region survey samples to trust the signal
    buf_cap: int = 260,                       # ring buffer per region: recent data tracks the walk
    lp_steps: int = 400,
    replay_n: int = 2500,
    finetune_lr: float = 3e-4,
    finetune_steps: int = 700,
    reactive_every: int = 4,
    err_floor: float = 0.05,
    value_floor: float = 1e-4,
    # --- FM ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
    probe_pool_n: int = 2000,
    ceil_pool_n: int = 4000,
    # --- control eval ---
    n_eval: int = 40,
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
    # --- rendering (OFF by default: with render_rounds="" nothing is called and every prior
    #     result reproduces exactly). Rounds are 0-indexed; -1 = the stale base FM. ---
    render_rounds: str = "",
    render_policies: str = "value",
    render_n: int = 0,                        # 0 = every eval reach (so `miss` == the graded median)
):
    import os

    pol = [p for p in policies.split(",") if p.strip()]
    if quick:
        rounds = 14; pool_n = 3000; fm_steps = 1500; finetune_steps = 300
        lp_steps = 150; replay_n = 1200; fm_hidden = 128; fm_layers = 2; n_eval = 12
        k_shoot = 256; cem_iters = 4; collect_k_shoot = 128; collect_cem_iters = 3
        budget = 120; mon_n = 60; probe_pool_n = 800; ceil_pool_n = 1500; geom_samples = 600; n_par = 8
        n_off_red = 3; n_off_noise = 2
        if policies == _ALL_POLICIES:                            # only if not explicitly overridden
            pol = ["uniform", "value", "lprog-only", "error-only"]   # noisy-TV contrast, lean smoke
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, policies=pol, rounds=rounds, drift_mode=drift_mode,
        ou_sigma=ou_sigma, b_lo=b_lo, b_hi=b_hi, drift_every=drift_every,
        drift_cycle=[int(x) for x in drift_cycle.split(",") if x.strip()], b1=b1, noise_amp=noise_amp,
        region_sigma=region_sigma, region_k=region_k, geom_samples=geom_samples, rad_min=rad_min,
        pref_width=pref_width, vis_steps=vis_steps, off_vis_max=off_vis_max, on_vis_min=on_vis_min,
        min_sep=min_sep, n_off_red=n_off_red, n_off_noise=n_off_noise, n_on_noise=n_on_noise,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, q_range=q_range,
        v_explore=v_explore, ep_len=ep_len, op_q_range=op_q_range, n_par=n_par, sigma_u=sigma_u,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        collect_replan_every=collect_replan_every, wrap_limit=wrap_limit, goal_jit=goal_jit,
        budget=budget, mon_n=mon_n, min_in=min_in, buf_cap=buf_cap, lp_steps=lp_steps,
        replay_n=replay_n, finetune_lr=finetune_lr, finetune_steps=finetune_steps,
        reactive_every=reactive_every, err_floor=err_floor, value_floor=value_floor,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        pool_n=pool_n, probe_pool_n=probe_pool_n, ceil_pool_n=ceil_pool_n,
        n_eval=n_eval, plan_H=plan_h, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        reach_tries=reach_tries,
        render_rounds=[int(x) for x in render_rounds.split(",") if x.strip()],
        render_policies=[p for p in render_policies.split(",") if p.strip()],
        render_n=render_n,
    )
    if spawn:
        # ADDITIVE, default-off. `modal run --detach <file>::<local entrypoint>` does not survive
        # the client process dying (the call is cancelled mid-round), which makes this file
        # unrunnable from a restartable container. `.spawn()` queues the call server-side and
        # returns; the durable artifact is then the volume copy rather than the local mirror:
        #     modal volume get mujoco-control-data /directed_on_policy/<tag>/results.json <dest>
        # With spawn=False every prior code path is byte-identical.
        call = run_directed_on_policy.spawn(cfg)
        print(f"[spawn] call {call.object_id} queued for tag={tag!r}")
        print(f"[spawn] results -> /data/directed_on_policy/{tag}/results.json")
        return
    out = run_directed_on_policy.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "directed_on_policy_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
    if out.get("render"):
        with open(os.path.join(localdir, "render_pack.json"), "w") as fh:
            json.dump(out["render"], fh, cls=NumpyEncoder)
        print(f"[local] wrote render_pack.json ({len(out['render']['packs'])} packs) to {localdir}")
