"""Cut #3 — Operator intervention: reward-free re-adaptation after a dynamics shift.

Program: `ideas/physical_control_substrate.md`, cut #3. The one thing MuJoCo
uniquely enables that language/RHM/MNIST cannot: INTERVENE ON THE TRANSITION
OPERATOR (the dynamics) while holding the task fixed. We use that to test the

    FACTORIZATION CLAIM. A model-based agent factorizes control into
    (world-model: "what the world does") x (planner/value: "what I want").
    A dynamics shift corrupts ONLY the world-model factor. So:
      * The model-based (FM + planner) agent re-adapts to new dynamics from
        REWARD-FREE self-supervised interaction alone -- every transition
        (s,u,s') is a dense labeled example of the new dynamics -- after which
        the fixed planner is immediately re-optimal (its objective never changed).
      * A model-free reactive policy pi(s,g)->u entangles both factors, so
        reward-free transitions give it NO learning signal; it needs
        reward-driven improvement, far less sample-efficient -- even though only
        the world changed, not the goal.

This is NOT the static "degradation slope" (a shifted operator makes the stale FM
itself wrong, so that would be theoretically shaky). The mechanistic result is the
RE-ADAPTATION SAMPLE-EFFICIENCY DISSOCIATION: on an identical budget of REWARD-FREE
post-shift interaction, the MB agent recovers toward the oracle ceiling while the MF
agent cannot move at all -- because the operator shift only corrupted the factor that
self-supervision can fix.

TASK: goal-conditioned pusher reaching, PUCK-FREE and momentum-dominated (low joint
damping -> the pusher glides, reaching requires anticipatory braking, a pure feedback
policy overshoots). State s = [px, py, vx, vy] (4-dim); command u = force (2-dim);
goal g = 2D target; performance = final ||pusher_pos - g|| over a short horizon.

AGENTS (both derive from the SAME d0 knowledge, fair):
  MB = arity-2 FM f(s,u)->Ds (the cut #2 idiom, self-supervised on random d0
       interaction) + CEM MPC (roll the FM K sequences, pick the one minimizing
       predicted running ||pos-g|| + vel penalty). Value dynamics-INDEPENDENT + FIXED.
  MF = a COMMITTING motor-program net pi(s,g)->(re-step action sequence), behavior-
       cloned from the MB planner's committed plans at d0. Both agents commit
       `replan_every` steps open-loop with the SAME protocol -- the ONLY difference is
       MB generates the program by rolling an adaptable FM, MF by an amortized net (so
       only MB can re-adapt from reward-free transitions). This removes the earlier
       commit-vs-reactive confound (MF used to re-ground every step).

DYNAMICS SHIFT d0->d1: pusher_mass 1.0 -> 3.0 (+ optional damping drop). The force
-> motion map changes, so a d0-tuned reactive closed loop is mistuned (over/oscillates)
while the FM can be re-fit from reward-free (s,u,s') transitions.

Run:
    cd experiments/
    modal run mjc/dynamics_shift/dynamics_shift.py::dynamics_shift --quick   # smoke
    modal run --detach mjc/dynamics_shift/dynamics_shift.py::run_dynamics_shift  # (full; see entrypoint)
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_dynamics_shift(cfg: dict) -> dict:
    import os
    import copy
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    H = cfg["H"]                       # episode (real) horizon
    Hp = cfg["H_plan"]                 # MPC planning horizon
    K = cfg["K"]                       # MPC shooting samples
    goal_r, start_r = cfg["goal_range"], cfg["start_range"]
    v0 = cfg["v0_std"]
    eps = cfg["success_eps"]
    vel_pen = cfg["vel_pen"]
    re = cfg["replan_every"]           # open-loop commit length (MB planner AND MF policy)

    dgp0 = dict(cfg["dgp_base"])
    dgp1 = dict(cfg["dgp_base"])
    dgp1["pusher_mass"] = cfg["shift_mass"]
    if cfg.get("shift_damping") is not None:
        dgp1["joint_damping"] = cfg["shift_damping"]
    print(f"[dgp] d0={dgp0}\n[dgp] d1={dgp1}", flush=True)

    env0 = PusherEnv(dgp0, with_puck=False)
    env1 = PusherEnv(dgp1, with_puck=False)

    # --------------------------------------------------------------------- #
    # data collection: OU-correlated random interaction (reward-free), so the
    # FM sees a broad range of velocities (not just the random-walk core).
    # --------------------------------------------------------------------- #
    def collect(env, n_transitions, seed, ou_theta=0.2, ou_sigma=0.5):
        """Reward-free OU-correlated interaction, started centrally at broad random
        velocities (NOT env.reset()'s full-arena uniform, which starts on the walls).
        Keeping exploration central in a large arena -> almost no wall contact -> the
        FM trains on clean free-flight dynamics (cut #2's smooth cut)."""
        rng = np.random.default_rng(seed)
        ep_len = cfg["collect_ep_len"]
        pr = cfg["collect_pos_range"]
        n_ep = int(np.ceil(n_transitions / ep_len))
        S, U, S2, C = [], [], [], []
        for _ in range(n_ep):
            p0 = rng.uniform(-pr, pr, size=2)
            env.set_state(p0, rng.normal(0, cfg["v_explore"], size=2))
            a = np.zeros(2)
            for _ in range(ep_len):
                s = env.get_state()
                a = a - ou_theta * a + ou_sigma * rng.normal(size=2)
                u = np.clip(a, -1.0, 1.0)
                s2, info = env.step(u, fs)
                S.append(s); U.append(u.astype(np.float32)); S2.append(s2)
                C.append(info["any_contact"])
        S = np.asarray(S, np.float32)[:n_transitions]
        U = np.asarray(U, np.float32)[:n_transitions]
        S2 = np.asarray(S2, np.float32)[:n_transitions]
        C = np.asarray(C, bool)[:n_transitions]
        print(f"    [collect] N={len(S)} contact_frac={C.mean():.3f} "
              f"|v|p95={float(np.percentile(np.abs(S[:,2:]),95)):.2f} "
              f"|v|max={float(np.abs(S[:,2:]).max()):.2f}", flush=True)
        return S, U, S2, C

    # --------------------------------------------------------------------- #
    # forward model f(s,u) -> Ds (arity-2, SiLU MLP, Huber) -- the cut #2 idiom.
    # Normalization is computed ONCE from d0 and fixed for every FM, so the MPC's
    # I/O convention never changes across d0/d1/oracle/refit.
    # --------------------------------------------------------------------- #
    def build_fm():
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, 4)]
        return nn.Sequential(*lyr).to(device)

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32)
        Y = (S2 - S).astype(np.float32)
        mx, sx = X.mean(0), X.std(0) + 1e-6
        my, sy = Y.mean(0), Y.std(0) + 1e-6
        return {k: torch.tensor(v, device=device)
                for k, v in dict(mx=mx, sx=sx, my=my, sy=sy).items()}

    def train_fm(fm, S, U, S2, norm, epochs, init_state=None):
        if init_state is not None:
            fm.load_state_dict(init_state)
        X = torch.tensor(np.concatenate([S, U], 1), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]
        Yn = (Y - norm["my"]) / norm["sy"]
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"])
        lossf = nn.HuberLoss(delta=1.0)
        n = Xn.shape[0]
        bs = min(cfg["fm_batch"], n)
        fm.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(fm(Xn[idx]), Yn[idx]).backward()
                opt.step()
        fm.eval()
        return fm

    def fm_free_r2(fm, S, U, S2, norm, C=None):
        """held-out R^2 of the FM on Ds (raw units), on FREE-FLIGHT transitions only
        (contact = stiff/unpredictable wall bounces cap the ceiling; cut #1/#2)."""
        if C is not None:
            ff = ~C
            S, U, S2 = S[ff], U[ff], S2[ff]
        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1), device=device)
            Xn = (X - norm["mx"]) / norm["sx"]
            pred = (fm(Xn) * norm["sy"] + norm["my"]).cpu().numpy()
        yt = (S2 - S).astype(np.float32)
        ss_res = ((yt - pred) ** 2).sum()
        ss_tot = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    # --------------------------------------------------------------------- #
    # CEM MPC (vectorized FM rollout in state space). cost = RUNNING distance
    # (sum_t ||pos_t - g||) + vel_pen * ||final_vel||. The running distance rewards
    # reaching FAST and STAYING (staying under momentum requires braking), so the
    # objective is dynamics-INDEPENDENT and FIXED across d0/d1 -- the reused planner
    # factor. CEM (elite-refit) is a strong-enough planner that the CORRECT model
    # wins; weak random shooting let a stale model's velocity over-prediction
    # accidentally brake better, confounding the oracle-vs-stale comparison.
    # --------------------------------------------------------------------- #
    n_elite = cfg["cem_elite"]

    def mpc_action(fm, norm, states, goals, rng):
        B = states.shape[0]
        mu = np.zeros((B, Hp, 2), np.float32)
        sig = np.full((B, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(K, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(K, 0)   # (B*K,4)
        for _ in range(cfg["cem_iters"]):
            eps = rng.standard_normal((B, K, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * eps, -1, 1)        # (B,K,Hp,2)
            with torch.no_grad():
                s = s0.clone()
                seqs_t = torch.tensor(seqs.reshape(B * K, Hp, 2), device=device)
                cost = torch.zeros(B * K, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (fm((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                    cost = cost + (s[:, :2] - g_t).norm(dim=1)             # running dist
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)               # terminal vel
                idx = torch.topk(-cost.reshape(B, K), n_elite, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)  # (B,E,Hp,2)
            mu = elite.mean(1)
            sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)   # FULL refined mean plan (B, Hp, 2)

    def make_mpc_fn(fm, norm, seed):
        rng = np.random.default_rng(seed)
        return lambda states, goals: mpc_action(fm, norm, states, goals, rng)

    # --------------------------------------------------------------------- #
    # MF = COMMITTING motor-program policy pi(s,g) -> a length-`re` open-loop action
    # sequence (flattened 2*re, tanh), executed with the SAME replan_every commit as
    # the MB planner. This removes the commit-vs-reactive confound: both agents emit
    # committed re-step motor programs from (s,g); the ONLY difference is MB generates
    # it by rolling an adaptable FM, MF by an amortized net (so only MB can re-adapt
    # from reward-free transitions).
    # --------------------------------------------------------------------- #
    def build_policy():
        h, L = cfg["pol_hidden"], cfg["pol_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, 2 * re), nn.Tanh()]
        return nn.Sequential(*lyr).to(device)

    pol_norm = {}

    def make_policy_fn(policy):
        policy.eval()
        def fn(states, goals):
            with torch.no_grad():
                x = torch.tensor(np.concatenate([states, goals], 1), device=device)
                xn = (x - pol_norm["mu"]) / pol_norm["sd"]
                # (B, re, 2): a committed re-step motor program (executed open-loop)
                out = policy(xn).cpu().numpy().astype(np.float32)
                return out.reshape(len(states), re, 2)
        return fn

    # --------------------------------------------------------------------- #
    # batched real-env rollout (one env, set_state each step -> B parallel
    # trajectories; MuJoCo is memoryless so this is exact). Returns final states,
    # per-episode final distance, and the logged (state, action) trajectory.
    # --------------------------------------------------------------------- #
    def rollout(env, starts, goals, plan_fn, replan_every=1):
        """Uniform interface: plan_fn(states,goals) -> (B, L, 2) with L>=replan_every.
        The plan is re-queried every `replan_every` steps and its actions executed
        OPEN-LOOP in between. replan_every=1 = re-ground every step (reactive policy /
        per-step MPC). replan_every>1 = the agent must COMMIT to a plan, so its
        world-model becomes load-bearing (a stale model can't be papered over by
        re-grounding — the RHM open-loop finding). MF/random pass length-1 plans."""
        B = starts.shape[0]
        states = starts.copy()
        S_log, U_log = [], []
        plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, goals)          # (B, L, 2)
            acts = plan[:, step % replan_every, :]
            S_log.append(states.copy()); U_log.append(acts.copy())
            for b in range(B):
                env.set_state(states[b, :2], states[b, 2:])
                s2, _ = env.step(acts[b], fs)
                states[b] = s2
        final_dist = np.linalg.norm(states[:, :2] - goals, axis=1)
        return states, final_dist, S_log, U_log

    def sample_goalstart(rng, n):
        gpos = rng.uniform(-goal_r, goal_r, (n, 2)).astype(np.float32)
        spos = rng.uniform(-start_r, start_r, (n, 2)).astype(np.float32)
        svel = (rng.normal(0, v0, (n, 2)) if v0 > 0 else np.zeros((n, 2))).astype(np.float32)
        return np.concatenate([spos, svel], 1).astype(np.float32), gpos

    def eval_agent(env, action_fn, starts, goals, replan_every=1):
        _, fd, _, _ = rollout(env, starts, goals, action_fn, replan_every)
        return {"mean_dist": float(fd.mean()), "median_dist": float(np.median(fd)),
                "success": float((fd < eps).mean())}

    # fixed evaluation set (identical goals/starts for EVERY agent -> controlled)
    eval_rng = np.random.default_rng(cfg["seed"] + 7)
    ev_starts, ev_goals = sample_goalstart(eval_rng, cfg["n_eval"])

    def random_fn(states, goals):
        return eval_rng.uniform(-1, 1, size=(states.shape[0], 1, 2)).astype(np.float32)

    # ===================================================================== #
    # 1. d0 forward model + MB competence
    # ===================================================================== #
    print(f"\n[1] collecting {cfg['n_fm']} d0 transitions + training FM ...", flush=True)
    S0, U0, S20, C0 = collect(env0, cfg["n_fm"], cfg["seed"] + 1)
    norm = make_norm(S0, U0, S20)
    fm_d0 = train_fm(build_fm(), S0, U0, S20, norm, cfg["fm_epochs"])
    r2_d0 = fm_free_r2(fm_d0, S0[-2000:], U0[-2000:], S20[-2000:], norm, C0[-2000:])
    print(f"[1] FM_d0 trained; free-flight Ds R^2={r2_d0:.4f}", flush=True)

    mpc_d0_fn = make_mpc_fn(fm_d0, norm, cfg["seed"] + 100)
    mb_d0 = eval_agent(env0, mpc_d0_fn, ev_starts, ev_goals, replan_every=re)
    rand_d0 = eval_agent(env0, random_fn, ev_starts, ev_goals)
    print(f"[1] MB(d0) on d0: mean_dist={mb_d0['mean_dist']:.4f} "
          f"succ={mb_d0['success']:.3f}  | random floor dist={rand_d0['mean_dist']:.4f}",
          flush=True)

    # ===================================================================== #
    # 2. behavior-clone the MB planner's COMMITTED PLANS at d0 -> motor-program
    #    policy pi(s,g) -> re-step sequence (same commit as MB; the fair MF baseline)
    # ===================================================================== #
    print(f"\n[2] collecting BC data ({cfg['n_bc_ep']} MPC episodes at d0) ...", flush=True)
    bc_rng = np.random.default_rng(cfg["seed"] + 2)
    bc_mpc = make_mpc_fn(fm_d0, norm, cfg["seed"] + 200)
    BX, BY = [], []
    B = cfg["batch_ep"]
    done = 0
    while done < cfg["n_bc_ep"]:
        b = min(B, cfg["n_bc_ep"] - done)
        st, gl = sample_goalstart(bc_rng, b)
        _, _, S_log, U_log = rollout(env0, st, gl, bc_mpc, replan_every=re)
        # target at each REPLAN point = the committed re-step action sequence
        for t in range(0, H - re + 1, re):
            seg = np.stack(U_log[t:t + re], axis=1)          # (b, re, 2)
            BX.append(np.concatenate([S_log[t], gl], 1))     # (b, 6)
            BY.append(seg.reshape(seg.shape[0], re * 2))     # (b, re*2)
        done += b
    BX = np.concatenate(BX, 0).astype(np.float32)   # (M,6)
    BY = np.concatenate(BY, 0).astype(np.float32)   # (M, re*2)
    pol_norm["mu"] = torch.tensor(BX.mean(0), device=device)
    pol_norm["sd"] = torch.tensor(BX.std(0) + 1e-6, device=device)

    print(f"[2] BC: {len(BX)} tuples; training policy ...", flush=True)
    policy_d0 = build_policy()
    optp = torch.optim.Adam(policy_d0.parameters(), lr=cfg["pol_lr"])
    Xp = (torch.tensor(BX, device=device) - pol_norm["mu"]) / pol_norm["sd"]
    Yp = torch.tensor(BY, device=device)
    lossf = nn.MSELoss()
    n = Xp.shape[0]; bs = min(cfg["pol_batch"], n)
    policy_d0.train()
    for _ in range(cfg["bc_epochs"]):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            optp.zero_grad()
            lossf(policy_d0(Xp[idx]), Yp[idx]).backward()
            optp.step()
    policy_d0.eval()
    mf_d0 = eval_agent(env0, make_policy_fn(policy_d0), ev_starts, ev_goals, replan_every=re)
    print(f"[2] MF(d0) on d0: mean_dist={mf_d0['mean_dist']:.4f} succ={mf_d0['success']:.3f} "
          f"(MB d0 was {mb_d0['mean_dist']:.4f}/{mb_d0['success']:.3f} -- BC fairness check)",
          flush=True)

    # ===================================================================== #
    # 3. zero-shot after the shift (d1): stale MB, MF, and the ORACLE ceiling
    # ===================================================================== #
    print(f"\n[3] zero-shot at d1 + oracle ceiling ...", flush=True)
    mb_d1_stale = eval_agent(env1, make_mpc_fn(fm_d0, norm, cfg["seed"] + 101),
                             ev_starts, ev_goals, replan_every=re)
    mf_d1 = eval_agent(env1, make_policy_fn(policy_d0), ev_starts, ev_goals, replan_every=re)
    rand_d1 = eval_agent(env1, random_fn, ev_starts, ev_goals)

    # d1 interaction pool (reward-free), reused for oracle + the refit sweep
    pool_N = max(max(cfg["reward_free_budgets"]), cfg["n_oracle"])
    print(f"[3] collecting d1 reward-free pool of {pool_N} transitions ...", flush=True)
    Sd1, Ud1, S2d1, Cd1 = collect(env1, pool_N, cfg["seed"] + 3)

    fm_oracle = train_fm(build_fm(), Sd1[:cfg["n_oracle"]], Ud1[:cfg["n_oracle"]],
                         S2d1[:cfg["n_oracle"]], norm, cfg["fm_epochs"])
    r2_oracle = fm_free_r2(fm_oracle, Sd1[-2000:], Ud1[-2000:], S2d1[-2000:], norm, Cd1[-2000:])
    mb_d1_oracle = eval_agent(env1, make_mpc_fn(fm_oracle, norm, cfg["seed"] + 102),
                              ev_starts, ev_goals, replan_every=re)
    print(f"[3] d1 zero-shot: MB-stale={mb_d1_stale['mean_dist']:.4f}/"
          f"{mb_d1_stale['success']:.3f}  MF={mf_d1['mean_dist']:.4f}/{mf_d1['success']:.3f}",
          flush=True)
    print(f"[3] d1 oracle(FM refit on {cfg['n_oracle']}, R^2={r2_oracle:.4f}): "
          f"{mb_d1_oracle['mean_dist']:.4f}/{mb_d1_oracle['success']:.3f}  | "
          f"random floor={rand_d1['mean_dist']:.4f}", flush=True)

    # ===================================================================== #
    # 4. REWARD-FREE adaptation sweep (the headline): MB recovers, MF flat.
    #    Each point: FM_d0 warm-started, fine-tuned on the first N d1 (s,u,s')
    #    transitions -- pure self-supervision, no reward. MPC value unchanged.
    # ===================================================================== #
    print(f"\n[4] reward-free MB adaptation sweep ...", flush=True)
    mb_free = {}
    fm_d0_state = copy.deepcopy(fm_d0.state_dict())
    for N in cfg["reward_free_budgets"]:
        if N == 0:
            res = mb_d1_stale   # stale FM = 0 reward-free transitions
        else:
            fm_N = train_fm(build_fm(), Sd1[:N], Ud1[:N], S2d1[:N], norm,
                            cfg["adapt_epochs"], init_state=fm_d0_state)
            res = eval_agent(env1, make_mpc_fn(fm_N, norm, cfg["seed"] + 300 + N),
                             ev_starts, ev_goals, replan_every=re)
        mb_free[N] = res
        print(f"    MB reward-free N={N:6d}: mean_dist={res['mean_dist']:.4f} "
              f"succ={res['success']:.3f}", flush=True)
    # MF reward-free is FLAT by construction: reward-free (s,u,s') has no target for
    # pi(s,g)->motor-program (it doesn't say which program reaches the goal), so MF
    # stays at its d1 zero-shot value at every N.
    mf_free_flat = mf_d1

    # ===================================================================== #
    # 5. REWARD-DRIVEN adaptation for MF (fair follow-up): a deliberately MINIMAL
    #    REINFORCE (Gaussian policy, EMA baseline) on d1 episodes, reward = -final
    #    distance. We only want the slope; this is NOT RL-maxxed.
    # ===================================================================== #
    print(f"\n[5] reward-driven MF adaptation (minimal REINFORCE) ...", flush=True)
    policy_rl = build_policy()
    policy_rl.load_state_dict(policy_d0.state_dict())
    optr = torch.optim.Adam(policy_rl.parameters(), lr=cfg["rl_lr"])
    rl_rng = np.random.default_rng(cfg["seed"] + 4)
    sigma_rl = cfg["rl_sigma"]
    Brl = cfg["rl_batch"]
    ckpts = sorted(cfg["reward_driven_budgets"])
    mf_reward = {0: mf_d1}   # budget 0 = pi_d0 (no reward-driven updates)
    baseline = None
    episodes = 0
    ci = 0
    while ci < len(ckpts) and ckpts[ci] == 0:
        ci += 1
    max_ep = ckpts[-1] if ckpts else 0
    while episodes < max_ep:
        st, gl = sample_goalstart(rl_rng, Brl)
        states = st.copy()
        logps = []
        seq_a = None
        for step in range(H):
            if step % re == 0:                                   # emit a committed program
                x = torch.tensor(np.concatenate([states, gl], 1), device=device)
                xn = (x - pol_norm["mu"]) / pol_norm["sd"]
                seq_mu = policy_rl(xn).reshape(Brl, re, 2)       # (B,re,2) tanh mean, grad
                noise = rl_rng.normal(0, sigma_rl, size=(Brl, re, 2)).astype(np.float32)
                seq_a = np.clip(seq_mu.detach().cpu().numpy() + noise, -1, 1).astype(np.float32)
                seq_a_t = torch.tensor(seq_a, device=device)
                # logprob of the whole committed sequence (sum over its re*2 dims)
                logps.append((-0.5 * ((seq_a_t - seq_mu) / sigma_rl) ** 2).sum((1, 2)))
            a = seq_a[:, step % re, :]
            for b in range(Brl):
                env1.set_state(states[b, :2], states[b, 2:])
                s2, _ = env1.step(a[b], fs)
                states[b] = s2
        R = -np.linalg.norm(states[:, :2] - gl, axis=1).astype(np.float32)
        Rt = torch.tensor(R, device=device)
        baseline = Rt.mean().item() if baseline is None else \
            0.9 * baseline + 0.1 * Rt.mean().item()
        adv = (Rt - baseline).detach()
        loss = -(adv * torch.stack(logps, 0).sum(0)).mean()
        optr.zero_grad(); loss.backward(); optr.step()
        episodes += Brl
        while ci < len(ckpts) and episodes >= ckpts[ci]:
            res = eval_agent(env1, make_policy_fn(policy_rl), ev_starts, ev_goals)
            mf_reward[ckpts[ci]] = res
            print(f"    MF reward-driven ep={ckpts[ci]:6d}: mean_dist={res['mean_dist']:.4f} "
                  f"succ={res['success']:.3f}", flush=True)
            ci += 1

    # ===================================================================== #
    # assemble + figures
    # ===================================================================== #
    results = {
        "config": cfg, "dgp0": dgp0, "dgp1": dgp1,
        "fm_r2_d0": r2_d0, "fm_r2_oracle": r2_oracle,
        "random_floor_d0": rand_d0, "random_floor_d1": rand_d1,
        "mb_d0": mb_d0, "mf_d0": mf_d0,
        "mb_d1_stale": mb_d1_stale, "mf_d1_zeroshot": mf_d1,
        "mb_d1_oracle": mb_d1_oracle,
        "reward_free_budgets": cfg["reward_free_budgets"],
        "mb_reward_free": {str(k): v for k, v in mb_free.items()},
        "mf_reward_free_flat": mf_free_flat,
        "reward_driven_budgets": ckpts,
        "mf_reward_driven": {str(k): v for k, v in mf_reward.items()},
        "H": H,
    }

    print("\n===== DYNAMICS-SHIFT SUMMARY =====", flush=True)
    print(f"d0 competence:  MB dist={mb_d0['mean_dist']:.3f}/succ={mb_d0['success']:.2f}  "
          f"MF dist={mf_d0['mean_dist']:.3f}/succ={mf_d0['success']:.2f}", flush=True)
    print(f"d1 zero-shot:   MB-stale dist={mb_d1_stale['mean_dist']:.3f}  "
          f"MF dist={mf_d1['mean_dist']:.3f}  (oracle ceiling {mb_d1_oracle['mean_dist']:.3f})",
          flush=True)
    print(f"MB reward-free: " + "  ".join(
        f"N{N}={mb_free[N]['mean_dist']:.3f}" for N in cfg["reward_free_budgets"]), flush=True)
    print(f"MF reward-free: FLAT at {mf_free_flat['mean_dist']:.3f} (by construction)", flush=True)
    print(f"MF reward-driven: " + "  ".join(
        f"ep{k}={mf_reward[k]['mean_dist']:.3f}" for k in sorted(mf_reward)), flush=True)

    figures = _make_figures(results)

    outdir = os.path.join(DATA_DIR, "dynamics_shift", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for name, png in figures.items():
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    C_MB, C_MF, C_ORC, C_RND = "#3d6fd1", "#d1603d", "#2f9e44", "#888"
    figs = {}
    H = R["H"]

    def _save(fig):
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # ---- fig1: competence + shift-bites + ceiling (bars, final distance) ----
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = [
        ("MB\nd0", R["mb_d0"]["mean_dist"], C_MB),
        ("MF\nd0", R["mf_d0"]["mean_dist"], C_MF),
        ("MB-stale\nd1 (0-shot)", R["mb_d1_stale"]["mean_dist"], C_MB),
        ("MF\nd1 (0-shot)", R["mf_d1_zeroshot"]["mean_dist"], C_MF),
        ("MB-oracle\nd1 (ceiling)", R["mb_d1_oracle"]["mean_dist"], C_ORC),
    ]
    x = np.arange(len(bars))
    ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars], alpha=0.85)
    for xi, (_, v, _) in zip(x, bars):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(b[1] for b in bars) * 1.4)   # tight; floor is far above
    ax.annotate(f"random floor {R['random_floor_d1']['mean_dist']:.2f}  ↑ (off scale)",
                xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
                fontsize=8, color=C_RND)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("final ||pusher_pos - goal||  (lower = better)")
    ax.set_title("The operator shift bites both agents; the FM-refit oracle is the ceiling")
    figs["fig1_shift_bites.png"] = _save(fig)

    ceil = R["mb_d1_oracle"]["mean_dist"]
    d0comp = R["mb_d0"]["mean_dist"]
    floor = R["random_floor_d1"]["mean_dist"]
    mf_flat = R["mf_reward_free_flat"]["mean_dist"]
    mb_stale = R["mb_d1_stale"]["mean_dist"]
    # tight y-window on the action (0.005-0.13); the random floor (~0.56) is far
    # above and shown as an annotation so it doesn't crush the interesting range.
    ytop = max(mf_flat, mb_stale) * 1.35

    def _refs(ax, floor_text=True):
        ax.axhline(ceil, color=C_ORC, ls="-", lw=1.3, label=f"MB oracle ceiling {ceil:.3f}")
        ax.axhline(d0comp, color="#555", ls=":", lw=1, label=f"d0 competence {d0comp:.3f}")
        ax.set_ylim(0, ytop)
        if floor_text:
            ax.annotate(f"random floor {floor:.2f}  ↑ (off scale)", xy=(0.5, 0.97),
                        xycoords="axes fraction", ha="center", va="top",
                        fontsize=8, color=C_RND)

    # ---- fig2: recovery curves, split by SIGNAL TYPE (two panels) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    Ns = R["reward_free_budgets"]
    mb = [R["mb_reward_free"][str(N)]["mean_dist"] for N in Ns]
    xN = [max(N, 1) for N in Ns]
    axL = axes[0]
    axL.plot(xN, mb, "o-", color=C_MB, lw=2, label="MB (reward-free FM refit)")
    axL.plot(xN, [mf_flat] * len(xN), "s--", color=C_MF, lw=2,
             label="MF (reward-free → no signal, FLAT)")
    _refs(axL)
    axL.set_xscale("log")
    axL.set_xlabel("post-shift REWARD-FREE interaction (transitions)")
    axL.set_ylabel("final ||pusher_pos - goal||")
    axL.set_title("Reward-free: MB recovers, MF cannot move")
    axL.legend(fontsize=8, loc="center right")

    axR = axes[1]
    eps_ = sorted(R["mf_reward_driven"].keys(), key=lambda s: int(s))
    ek = [int(s) for s in eps_]
    mfd = [R["mf_reward_driven"][s]["mean_dist"] for s in eps_]
    axR.plot([max(e, 1) for e in ek], mfd, "s-", color=C_MF, lw=2,
             label="MF (reward-driven REINFORCE)")
    _refs(axR)
    axR.set_xscale("symlog")
    axR.set_xlabel("post-shift REWARD-LABELED episodes  (each = H transitions, 1 scalar reward)")
    axR.set_title("Reward-driven MF: recovers, but needs reward labels")
    axR.legend(fontsize=8, loc="center right")
    fig.suptitle(
        "Factorization: an operator shift corrupts only the world-model factor.\n"
        "MB fixes it from reward-free self-supervision; MF needs reward-driven episodes.",
        fontsize=11)
    figs["fig2_recovery_curves.png"] = _save(fig)

    # ---- fig3: the direct sample-efficiency overlay (ONE interaction axis) ----
    # Put both agents on the same x = post-shift interaction in TRANSITIONS
    # (MF reward-driven episodes -> episodes*H transitions), so the gap is readable
    # on one axis. MB additionally needs NO reward labels; MF needs one per episode.
    # interaction each agent needs to reach ~ceiling (first budget with dist<=1.5*ceil)
    mb_recover_N = next((N for N, d in zip(Ns, mb) if N > 0 and d <= ceil * 1.5), None)
    mf_recover_ep = next((e for e, d in zip(ek, mfd) if d <= ceil * 1.5), None)
    if mb_recover_N and mf_recover_ep:
        ratio = max(1, round((mf_recover_ep * H) / mb_recover_N))
        subtitle = (f"MB reaches ~ceiling by {mb_recover_N} reward-FREE transitions; "
                    f"MF needs ~{mf_recover_ep * H} reward-LABELED ones "
                    f"(~{ratio}× more interaction, + a reward per episode)")
    elif mb_recover_N:
        subtitle = (f"MB reaches ~ceiling by {mb_recover_N} reward-free transitions; "
                    f"MF still short after {max(ek) * H if ek else 0} reward-labeled")
    else:
        subtitle = "MB recovers from reward-free data; MF needs far more, reward-labeled"

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(xN, mb, "o-", color=C_MB, lw=2.2, label="MB — reward-FREE FM refit")
    ax.plot([max(e * H, 1) for e in ek], mfd, "s-", color=C_MF, lw=2.2,
            label="MF — reward-DRIVEN REINFORCE")
    ax.axhline(mf_flat, color=C_MF, ls="--", lw=1.6,
               label=f"MF reward-free (FLAT {mf_flat:.3f})")
    _refs(ax)
    ax.set_xscale("log")
    ax.set_xlabel("post-shift interaction  (transitions;  MF episodes × H)")
    ax.set_ylabel("final ||pusher_pos - goal||")
    ax.set_title("Sample efficiency on one interaction axis\n" + subtitle, fontsize=10)
    ax.legend(fontsize=9, loc="center right")
    figs["fig3_sample_efficiency.png"] = _save(fig)
    return figs


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_replan_ablation(cfg: dict) -> dict:
    """Ablation: sweep the MB planner's open-loop commit length `replan_every` and
    show the reward-free-adaptation BENEFIT (stale-FM cost that re-fitting removes)
    GROWS with commitment horizon — the value of an adapted forward model scales
    with how ballistic the control is. FMs are trained ONCE (they're replan-
    independent); only the MPC eval depends on replan_every, so this is cheap.
    Self-contained (duplicates run_dynamics_shift's small helpers to keep that
    validated path untouched)."""
    import os
    import copy
    import numpy as np
    import torch
    import torch.nn as nn
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mjc.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs, H, Hp, K = cfg["frame_skip"], cfg["H"], cfg["H_plan"], cfg["K"]
    goal_r, start_r, v0, eps = cfg["goal_range"], cfg["start_range"], cfg["v0_std"], cfg["success_eps"]
    vel_pen, n_elite = cfg["vel_pen"], cfg["cem_elite"]
    dgp0 = dict(cfg["dgp_base"]); dgp1 = dict(cfg["dgp_base"])
    dgp1["pusher_mass"] = cfg["shift_mass"]
    if cfg.get("shift_damping") is not None:
        dgp1["joint_damping"] = cfg["shift_damping"]
    env0, env1 = PusherEnv(dgp0, with_puck=False), PusherEnv(dgp1, with_puck=False)

    def collect(env, n_transitions, seed):
        rng = np.random.default_rng(seed)
        ep_len, pr = cfg["collect_ep_len"], cfg["collect_pos_range"]
        S, U, S2 = [], [], []
        for _ in range(int(np.ceil(n_transitions / ep_len))):
            env.set_state(rng.uniform(-pr, pr, 2), rng.normal(0, cfg["v_explore"], 2))
            a = np.zeros(2)
            for _ in range(ep_len):
                s = env.get_state()
                a = a - 0.2 * a + 0.5 * rng.normal(size=2)
                u = np.clip(a, -1, 1)
                s2, _ = env.step(u, fs)
                S.append(s); U.append(u.astype(np.float32)); S2.append(s2)
        return (np.asarray(S, np.float32)[:n_transitions], np.asarray(U, np.float32)[:n_transitions],
                np.asarray(S2, np.float32)[:n_transitions])

    def build_fm():
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        return nn.Sequential(*(lyr + [nn.Linear(h, 4)])).to(device)

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    def train_fm(fm, S, U, S2, norm, epochs, init_state=None):
        if init_state is not None:
            fm.load_state_dict(init_state)
        X = torch.tensor(np.concatenate([S, U], 1), device=device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn, Yn = (X - norm["mx"]) / norm["sx"], (Y - norm["my"]) / norm["sy"]
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"]); lossf = nn.HuberLoss(delta=1.0)
        n = Xn.shape[0]; bs = min(cfg["fm_batch"], n); fm.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(); lossf(fm(Xn[idx]), Yn[idx]).backward(); opt.step()
        fm.eval(); return fm

    def mpc_action(fm, norm, states, goals, rng, replan_every):
        B = states.shape[0]
        mu = np.zeros((B, Hp, 2), np.float32); sig = np.full((B, Hp, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(K, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(K, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((B, K, Hp, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(B * K, Hp, 2), device=device)
                cost = torch.zeros(B * K, device=device)
                for h in range(Hp):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    s = s + (fm((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"])
                    cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(B, K), n_elite, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_mpc(env, fm, norm, seed, replan_every, starts, goals):
        rng = np.random.default_rng(seed)
        B = starts.shape[0]; states = starts.copy(); plan = None
        for step in range(H):
            if step % replan_every == 0:
                plan = mpc_action(fm, norm, states, goals, rng, replan_every)
            acts = plan[:, step % replan_every, :]
            for b in range(B):
                env.set_state(states[b, :2], states[b, 2:]); s2, _ = env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, :2] - goals, axis=1)
        return {"mean_dist": float(fd.mean()), "success": float((fd < eps).mean())}

    # ---- train the three FMs ONCE (replan-independent) -------------------- #
    print("[abl] training FM_d0, FM_oracle, FM_refit ...", flush=True)
    S0, U0, S20 = collect(env0, cfg["n_fm"], cfg["seed"] + 1)
    norm = make_norm(S0, U0, S20)
    fm_d0 = train_fm(build_fm(), S0, U0, S20, norm, cfg["fm_epochs"])
    fm_d0_state = copy.deepcopy(fm_d0.state_dict())
    Sd1, Ud1, S2d1 = collect(env1, max(cfg["n_oracle"], cfg["refit_N"]), cfg["seed"] + 3)
    fm_oracle = train_fm(build_fm(), Sd1[:cfg["n_oracle"]], Ud1[:cfg["n_oracle"]], S2d1[:cfg["n_oracle"]],
                         norm, cfg["fm_epochs"])
    fm_refit = train_fm(build_fm(), Sd1[:cfg["refit_N"]], Ud1[:cfg["refit_N"]], S2d1[:cfg["refit_N"]],
                        norm, cfg["adapt_epochs"], init_state=fm_d0_state)

    rng = np.random.default_rng(cfg["seed"] + 7)
    starts = np.concatenate([rng.uniform(-start_r, start_r, (cfg["n_eval"], 2)),
                             (rng.normal(0, v0, (cfg["n_eval"], 2)) if v0 > 0 else
                              np.zeros((cfg["n_eval"], 2)))], 1).astype(np.float32)
    goals = rng.uniform(-goal_r, goal_r, (cfg["n_eval"], 2)).astype(np.float32)

    # ---- sweep replan_every in EVAL only --------------------------------- #
    rows = []
    for re in cfg["replan_list"]:
        stale = eval_mpc(env1, fm_d0, norm, cfg["seed"] + 101, re, starts, goals)
        orac = eval_mpc(env1, fm_oracle, norm, cfg["seed"] + 102, re, starts, goals)
        ref = eval_mpc(env1, fm_refit, norm, cfg["seed"] + 103, re, starts, goals)
        rows.append({"replan_every": re, "stale": stale, "oracle": orac, "refit": ref})
        print(f"    replan_every={re:2d}: stale={stale['mean_dist']:.4f}  "
              f"oracle={orac['mean_dist']:.4f}  refit={ref['mean_dist']:.4f}  "
              f"benefit(stale-refit)={stale['mean_dist']-ref['mean_dist']:+.4f}", flush=True)

    results = {"config": cfg, "dgp0": dgp0, "dgp1": dgp1, "refit_N": cfg["refit_N"], "rows": rows}

    # ---- figure ---------------------------------------------------------- #
    res = [r["replan_every"] for r in rows]
    stale = [r["stale"]["mean_dist"] for r in rows]
    oracle = [r["oracle"]["mean_dist"] for r in rows]
    refit = [r["refit"]["mean_dist"] for r in rows]
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.fill_between(res, refit, stale, color="#d1603d", alpha=0.15,
                    label="reward-free adaptation benefit (stale − refit)")
    ax.plot(res, stale, "^-", color="#d1603d", lw=2, label="MB stale FM (d0 model on d1)")
    ax.plot(res, refit, "s-", color="#3d6fd1", lw=2,
            label=f"MB reward-free refit ({cfg['refit_N']} transitions)")
    ax.plot(res, oracle, "o-", color="#2f9e44", lw=2, label="MB oracle (fully-adapted FM)")
    ax.set_xlabel("replan_every  (open-loop commit length  ~  how ballistic the control)")
    ax.set_ylabel("final ||pusher_pos - goal||  (lower = better)")
    ax.set_title("The value of an adapted forward model grows with commitment horizon\n"
                 "(per-step re-grounding tolerates a stale model; committing does not)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, max(stale) * 1.15)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    figures = {"fig_replan_ablation.png": buf.getvalue()}

    outdir = os.path.join(DATA_DIR, "dynamics_shift", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for name, png in figures.items():
        with open(os.path.join(outdir, name), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote replan ablation to {outdir}", flush=True)
    return {"results": results, "figures": figures}


@app.local_entrypoint()
def replan_ablation(
    quick: bool = False,
    tag: str = "replan_abl_v1",
    seed: int = 0,
    refit_n: int = 5000,
):
    import os

    replan_list = [1, 2, 4, 6, 8, 12]
    n_fm = n_oracle = 20000
    n_eval = 64
    if quick:
        replan_list = [1, 4, 12]
        n_fm = n_oracle = 3000
        refit_n = 1500
        n_eval = 24
        tag = "replan_abl_smoke"

    cfg = dict(
        tag=tag, seed=seed, replan_list=replan_list, refit_N=refit_n,
        frame_skip=12, H=24, goal_range=0.5, start_range=0.5, v0_std=0.3,
        success_eps=0.1, v_explore=1.2, collect_ep_len=20, collect_pos_range=1.0,
        dgp_base=dict(arena_half=2.0, pusher_mass=1.0, joint_damping=2.0, gear=10.0),
        shift_mass=1.0, shift_damping=0.05,
        K=256, H_plan=15, vel_pen=0.5, cem_iters=3, cem_elite=32, cem_init_sigma=0.8,
        fm_hidden=256, fm_layers=3, fm_epochs=40, fm_lr=1e-3, fm_batch=512,
        n_fm=n_fm, adapt_epochs=60, n_oracle=n_oracle, n_eval=n_eval,
    )
    out = run_replan_ablation.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "dynshift_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for name, png in out["figures"].items():
        with open(os.path.join(localdir, name), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote replan ablation to {localdir}")


@app.local_entrypoint()
def dynamics_shift(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # env / task  (momentum-dominated free-flight reaching in a large arena so the
    # pusher rarely touches walls -> the FM stays on the clean smooth dynamics)
    # OPERATOR SHIFT = friction/drag collapse ("high-friction world -> ice"). At d0
    # the joint drag helps stop the pusher, so the reactive policy barely learns to
    # brake; at d1 the drag is gone, so that policy slides straight past every goal,
    # while the model-based agent re-learns the slippery dynamics from reward-free
    # sliding and re-plans to brake. Keeps d1 feasible (less drag = easier to move).
    frame_skip: int = 12,              # coarse control step (0.024s) -> momentum bites
    horizon: int = 24,
    goal_range: float = 0.5,
    start_range: float = 0.5,
    v0_std: float = 0.3,
    success_eps: float = 0.1,
    joint_damping: float = 2.0,        # d0: high drag -> drag does the stopping
    gear: float = 10.0,
    arena_half: float = 2.0,           # large arena -> reaching stays in free flight
    shift_mass: float = 1.0,           # no mass change (damping-only operator shift)
    shift_damping: float = 0.05,       # d1: near-frictionless "ice" -> pusher glides
    v_explore: float = 1.2,            # start-velocity coverage for the FM
    collect_ep_len: int = 20,          # short episodes -> bounded velocity on ice
    collect_pos_range: float = 1.0,    # central exploration box (arena is +-2) -> ~no walls
    # MPC (CEM)
    k_shoot: int = 256,
    h_plan: int = 15,
    cem_iters: int = 3,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    replan_every: int = 8,             # MB open-loop commit length (must be <= h_plan).
                                       # >1 makes the world-model load-bearing: a stale
                                       # model can't be papered over by re-grounding, so
                                       # the shift bites and reward-free refit recovers.
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_epochs: int = 40,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    n_fm: int = 20000,
    adapt_epochs: int = 60,
    n_oracle: int = 20000,
    # policy / BC (MF is a committing motor-program net -> needs more capacity/data
    # than a reactive policy: it must imitate an re-step open-loop sequence accurately)
    pol_hidden: int = 256,
    pol_layers: int = 3,
    pol_lr: float = 1e-3,
    pol_batch: int = 512,
    bc_epochs: int = 80,
    n_bc_ep: int = 1024,
    batch_ep: int = 64,
    # reward-driven REINFORCE (deliberately minimal; only the slope matters). Higher
    # variance on the committing sequence policy -> larger batch + calmer lr/sigma.
    rl_lr: float = 5e-4,
    rl_sigma: float = 0.15,
    rl_batch: int = 128,
    # eval
    n_eval: int = 96,
):
    import os

    reward_free_budgets = [0, 50, 150, 500, 1500, 5000, 20000]
    reward_driven_budgets = [500, 1000, 2000, 5000, 10000]

    if quick:
        n_fm, fm_epochs, fm_hidden, fm_layers = 3000, 15, 128, 2
        n_bc_ep, bc_epochs = 512, 80
        reward_free_budgets = [0, 500, 2000]
        n_oracle = 2000
        reward_driven_budgets = [512, 2048]
        adapt_epochs = 30
        k_shoot, h_plan, horizon = 96, 12, 26
        cem_elite = 12
        replan_every = min(replan_every, h_plan)
        n_eval, batch_ep, rl_batch = 24, 24, 32
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed,
        frame_skip=frame_skip, H=horizon, goal_range=goal_range, start_range=start_range,
        v0_std=v0_std, success_eps=success_eps, v_explore=v_explore,
        collect_ep_len=collect_ep_len, collect_pos_range=collect_pos_range,
        dgp_base=dict(arena_half=arena_half, pusher_mass=1.0,
                      joint_damping=joint_damping, gear=gear),
        shift_mass=shift_mass,
        shift_damping=(None if shift_damping < 0 else shift_damping),
        K=k_shoot, H_plan=h_plan, vel_pen=vel_pen, replan_every=replan_every,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_epochs=fm_epochs, fm_lr=fm_lr,
        fm_batch=fm_batch, n_fm=n_fm, adapt_epochs=adapt_epochs, n_oracle=n_oracle,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr,
        pol_batch=pol_batch, bc_epochs=bc_epochs, n_bc_ep=n_bc_ep, batch_ep=batch_ep,
        rl_lr=rl_lr, rl_sigma=rl_sigma, rl_batch=rl_batch,
        reward_free_budgets=reward_free_budgets,
        reward_driven_budgets=reward_driven_budgets,
        n_eval=n_eval,
    )
    out = run_dynamics_shift.remote(cfg)

    localdir = os.path.join(os.path.dirname(__file__), "figures", "dynshift_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for name, png in out["figures"].items():
        with open(os.path.join(localdir, name), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
