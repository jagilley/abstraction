"""Cut #3 sequel — DISAGREEMENT-DIRECTED reward-free re-adaptation after a dynamics shift.

Program: `ideas/two_timescale_value_loop.md` (§"Meta-RL: the two-timescale structure";
§"The interface shape", point 3: ensemble-disagreement as the online reducibility proxy)
and `a2a_forward/reaching/CURIOSITY_CONTROL_README.md` (Step 0 selected the disagreement-
MAGNITUDE drive over the LP derivative — that choice is the premise here).

    THE CLAIM. Cut #3 (`dynamics_shift.py`) showed a model-based agent re-adapts to a
    shifted operator from REWARD-FREE interaction (fine-tune the arity-2 FM on d1
    (s,u,s') -> recover toward the oracle ceiling), but it collected those transitions
    by UNDIRECTED OU-random exploration. The question this file answers:

      Does DISAGREEMENT-DIRECTED reward-free exploration re-adapt the FM to the new
      dynamics in FEWER transitions than undirected exploration?

    This is the "drive" earning its keep: a value over WHERE to learn (go where the new
    dynamics are still unlearned) accelerating inner-loop re-adaptation under non-
    stationarity. The scarce resource being measured is TRANSITIONS; the ensemble's
    extra compute is the acknowledged "cost of directing." Reference: Pathak, Gandhi,
    Gupta 2019, "Self-Supervised Exploration via Disagreement".

SHIFT MODE (the load-bearing knob). `shift_mode="global"` = Cut #3's drag collapse
everywhere -> EVERY transition is informative -> no scarcity -> directed==undirected (the
established NULL, `full_v1`). `shift_mode="patch"` = a LOCALIZED ice patch appears (a
Gaussian region of low drag, global damping unchanged) -> only IN-PATCH transitions are
informative -> SCARCITY, the regime where a disagreement-directed drive that concentrates
collection on the patch should re-adapt the patch in FEWER transitions. Patch mode adds a
PER-REGION readout (in-patch vs out-patch FM R^2 on a balanced eval pool) + patch
visitation — the sharp test of "did the drive re-learn the local change faster".

DESIGN (a knob-sweep-with-a-baseline, NOT RL-to-SOTA). Substrate = Cut #3's puck-free
momentum reaching. Compare TWO ways of collecting the post-shift REWARD-FREE d1
transitions:
  1. UNDIRECTED (baseline = Cut #3's `collect`): OU-correlated central exploration.
  2. DISAGREEMENT-DIRECTED: an ensemble of K FMs, each warm-started from FM_d0 with a
     different init perturbation (+ bootstrap-resampled buffers) so they DISAGREE off-
     data. Collect in rounds: from the current state, sample M candidate actions, score
     each by DISAGREEMENT = variance across the K FMs' predicted Δŝ (mean over state
     dims, NORMALIZED units), execute the argmax-disagreement action (+ mild epsilon-
     random mixing). Every `retrain_every` transitions, re-fit the ensemble on the
     GROWING buffer (warm-started from FM_d0). A short RANDOM warmup seeds the buffer so
     the ensemble disagrees where d1 data is sparse (bootstrapping the signal — an
     ensemble on IDENTICAL FM_d0 weights at 0 data agrees, so the per-member init
     perturbation + warmup are load-bearing). Exploration stays CENTRAL (Cut #3's box)
     so the comparison is on the reducible free-flight dynamics — wall contact is
     irreducible and the drive must not chase it.
  3. (OPTIONAL) ERROR-DIRECTED: a single refit FM; score = squared difference of its
     prediction from the STALE FM_d0 (predicted novelty-vs-d0). On this fully-reducible
     task surprise≈disagreement, so it checks whether the ensemble is special.

FAIR COMPARISON (critical): at each transition budget N, take the FIRST N transitions
from each arm's buffer, fine-tune a SINGLE fresh FM (warm-started from FM_d0, IDENTICAL
training procedure to Cut #3 — same epochs/lr) on them, and evaluate PLANNING (CEM MPC
final ||pusher_pos - goal|| on a FIXED eval set) + free-flight FM R^2 on a common held-
out set. ONLY the collection differs. The ensemble is used ONLY to direct collection.

Headline metric: transitions-to-recover = smallest N at which the re-fit FM's planning
reaches within 1.5x of the ORACLE CEILING (a single FM fully re-fit on a large fresh d1
pool). Reported for both arms (directed should be smaller) + the full recovery curve +
the Cut #3 anchors (d0 competence, d1 zero-shot, oracle ceiling, random floor).

Run:
    cd experiments/
    modal run mujoco_control/directed_readapt.py::directed_readapt --quick   # smoke
    modal run --detach mujoco_control/directed_readapt.py::run_directed_readapt  # (full; via entrypoint)
"""

import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=7200, volumes={DATA_DIR: volume})
def run_directed_readapt(cfg: dict) -> dict:
    """Self-contained (duplicates dynamics_shift.py's small helpers to keep that
    validated path untouched — the house pattern of `run_replan_ablation`)."""
    import os
    import copy
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    fs = cfg["frame_skip"]
    H, Hp, K = cfg["H"], cfg["H_plan"], cfg["K"]
    goal_r, start_r, v0, eps = cfg["goal_range"], cfg["start_range"], cfg["v0_std"], cfg["success_eps"]
    vel_pen, n_elite = cfg["vel_pen"], cfg["cem_elite"]
    re = cfg["replan_every"]

    dgp0 = dict(cfg["dgp_base"])
    dgp1 = dict(cfg["dgp_base"])
    if cfg["shift_mode"] == "patch":
        # LOCALIZED shift: a strong force jet appears in a Gaussian patch (global damping
        # unchanged). Only in-patch transitions are informative -> scarcity for the drive.
        dgp1["patch"] = dict(center=cfg["patch_center"], sigma=cfg["patch_sigma"],
                             force=cfg["patch_force"])
    else:
        # GLOBAL shift (Cut #3): drag collapses everywhere -> no scarcity (baseline/null).
        dgp1["pusher_mass"] = cfg["shift_mass"]
        if cfg.get("shift_damping") is not None:
            dgp1["joint_damping"] = cfg["shift_damping"]
    print(f"[dgp] mode={cfg['shift_mode']}\n[dgp] d0={dgp0}\n[dgp] d1={dgp1}", flush=True)

    env0 = PusherEnv(dgp0, with_puck=False)
    env1 = PusherEnv(dgp1, with_puck=False)

    # --------------------------------------------------------------------- #
    # UNDIRECTED reward-free collection = Cut #3's `collect` verbatim (central OU).
    # Reused for BOTH the undirected arm AND the directed arm's random warmup, so the
    # warmup is literally undirected collection (maximally fair).
    # --------------------------------------------------------------------- #
    def collect(env, n_transitions, seed, ou_theta=0.2, ou_sigma=0.5,
                center=None, pr_override=None):
        rng = np.random.default_rng(seed)
        ep_len = cfg["collect_ep_len"]
        pr = cfg["collect_pos_range"] if pr_override is None else pr_override
        c = np.zeros(2) if center is None else np.asarray(center, np.float64)
        n_ep = int(np.ceil(n_transitions / ep_len))
        S, U, S2, C = [], [], [], []
        for _ in range(n_ep):
            p0 = c + rng.uniform(-pr, pr, size=2)
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
        return S, U, S2, C

    # --------------------------------------------------------------------- #
    # forward model f(s,u) -> Δs (arity-2 SiLU MLP, Huber) + FIXED d0 normalization.
    # --------------------------------------------------------------------- #
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

    def fm_free_r2(fm, S, U, S2, norm, C=None):
        if C is not None:
            ff = ~C; S, U, S2 = S[ff], U[ff], S2[ff]
        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1), device=device)
            Xn = (X - norm["mx"]) / norm["sx"]
            pred = (fm(Xn) * norm["sy"] + norm["my"]).cpu().numpy()
        yt = (S2 - S).astype(np.float32)
        ss_res = ((yt - pred) ** 2).sum()
        ss_tot = ((yt - yt.mean(0, keepdims=True)) ** 2).sum()
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    # --------------------------------------------------------------------- #
    # CEM MPC (value = running distance-to-goal + terminal-vel penalty; dynamics-
    # INDEPENDENT and FIXED across d0/d1/refit — the reused planner factor).
    # --------------------------------------------------------------------- #
    def mpc_action(fm, norm, states, goals, rng):
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

    def eval_mpc(env, fm, norm, seed, starts, goals):
        rng = np.random.default_rng(seed)
        states = starts.copy(); plan = None
        for step in range(H):
            if step % re == 0:
                plan = mpc_action(fm, norm, states, goals, rng)
            acts = plan[:, step % re, :]
            for b in range(starts.shape[0]):
                env.set_state(states[b, :2], states[b, 2:]); s2, _ = env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, :2] - goals, axis=1)
        return {"mean_dist": float(fd.mean()), "median_dist": float(np.median(fd)),
                "success": float((fd < eps).mean())}

    # fixed evaluation set (identical goals/starts for EVERY agent -> controlled)
    ev_rng = np.random.default_rng(cfg["seed"] + 7)
    ev_starts = np.concatenate([ev_rng.uniform(-start_r, start_r, (cfg["n_eval"], 2)),
                                (ev_rng.normal(0, v0, (cfg["n_eval"], 2)) if v0 > 0
                                 else np.zeros((cfg["n_eval"], 2)))], 1).astype(np.float32)
    ev_goals = ev_rng.uniform(-goal_r, goal_r, (cfg["n_eval"], 2)).astype(np.float32)

    def eval_random(env, seed):
        rng = np.random.default_rng(seed)
        states = ev_starts.copy()
        for _ in range(H):
            acts = rng.uniform(-1, 1, size=(ev_starts.shape[0], 2)).astype(np.float32)
            for b in range(ev_starts.shape[0]):
                env.set_state(states[b, :2], states[b, 2:]); s2, _ = env.step(acts[b], fs); states[b] = s2
        fd = np.linalg.norm(states[:, :2] - ev_goals, axis=1)
        return {"mean_dist": float(fd.mean()), "success": float((fd < eps).mean())}

    # ===================================================================== #
    # 1. d0 forward model + MB competence (+ random floor)
    # ===================================================================== #
    print(f"\n[1] collecting {cfg['n_fm']} d0 transitions + training FM_d0 ...", flush=True)
    S0, U0, S20, C0 = collect(env0, cfg["n_fm"], cfg["seed"] + 1)
    norm = make_norm(S0, U0, S20)
    fm_d0 = train_fm(build_fm(), S0, U0, S20, norm, cfg["fm_epochs"])
    fm_d0_state = copy.deepcopy(fm_d0.state_dict())
    r2_d0 = fm_free_r2(fm_d0, S0[-2000:], U0[-2000:], S20[-2000:], norm, C0[-2000:])
    mb_d0 = eval_mpc(env0, fm_d0, norm, cfg["seed"] + 100, ev_starts, ev_goals)
    rand_d0 = eval_random(env0, cfg["seed"] + 900)
    print(f"[1] FM_d0 R^2={r2_d0:.4f}  MB(d0)={mb_d0['mean_dist']:.4f}/succ={mb_d0['success']:.3f}  "
          f"random floor={rand_d0['mean_dist']:.4f}", flush=True)

    # ===================================================================== #
    # 2. d1 zero-shot (stale FM) + ORACLE ceiling + common held-out pool
    # ===================================================================== #
    print(f"\n[2] d1 zero-shot + oracle ceiling ...", flush=True)
    mb_d1_stale = eval_mpc(env1, fm_d0, norm, cfg["seed"] + 101, ev_starts, ev_goals)
    rand_d1 = eval_random(env1, cfg["seed"] + 901)

    nh = 2000
    # +nh so the held-out slice is DISJOINT from the oracle's training set (Sp[:n_oracle])
    # -> honest oracle R^2 reference. The arms train on their OWN buffers, so Sh is held-out
    # for them regardless; this only cleans the oracle line.
    pool_N = max(cfg["n_oracle"], cfg["max_budget"]) + nh
    print(f"[2] collecting fresh d1 pool of {pool_N} transitions (oracle + disjoint held-out) ...", flush=True)
    if cfg["shift_mode"] == "patch":
        # the ORACLE must KNOW the local patch -> guarantee patch coverage by mixing a
        # third patch-focused collection into the pool, then shuffle so both the oracle
        # train slice [:n_oracle] and the held-out tail [-nh:] carry patch samples.
        n_pf = pool_N // 3
        Spc, Upc, S2pc, Cpc = collect(env1, pool_N - n_pf, cfg["seed"] + 3)          # central
        Spp, Upp, S2pp, Cpp = collect(env1, n_pf, cfg["seed"] + 33,
                                      center=cfg["patch_center"], pr_override=cfg["patch_sigma"])
        Sp = np.concatenate([Spc, Spp]); Up = np.concatenate([Upc, Upp])
        S2p = np.concatenate([S2pc, S2pp]); Cp = np.concatenate([Cpc, Cpp])
        perm = np.random.default_rng(cfg["seed"] + 34).permutation(len(Sp))
        Sp, Up, S2p, Cp = Sp[perm], Up[perm], S2p[perm], Cp[perm]
    else:
        Sp, Up, S2p, Cp = collect(env1, pool_N, cfg["seed"] + 3)
    Sh, Uh, S2h, Ch = Sp[-nh:], Up[-nh:], S2p[-nh:], Cp[-nh:]     # common held-out for R^2 (disjoint)

    fm_oracle = train_fm(build_fm(), Sp[:cfg["n_oracle"]], Up[:cfg["n_oracle"]],
                         S2p[:cfg["n_oracle"]], norm, cfg["fm_epochs"])
    r2_oracle = fm_free_r2(fm_oracle, Sh, Uh, S2h, norm, Ch)
    mb_d1_oracle = eval_mpc(env1, fm_oracle, norm, cfg["seed"] + 102, ev_starts, ev_goals)
    ceil = mb_d1_oracle["mean_dist"]
    r2_stale = fm_free_r2(fm_d0, Sh, Uh, S2h, norm, Ch)           # stale FM R^2 on d1 (N=0 point)
    print(f"[2] d1 zero-shot MB-stale={mb_d1_stale['mean_dist']:.4f}  "
          f"oracle ceiling={ceil:.4f} (R^2={r2_oracle:.4f})  random floor={rand_d1['mean_dist']:.4f}",
          flush=True)

    # --------------------------------------------------------------------- #
    # PER-REGION eval (patch mode): the sharp readout is "did you re-learn the LOCAL
    # change?". Build a BALANCED eval pool (half reset ON the patch, half central) so
    # in-patch R^2 is well-estimated regardless of how scarce the patch is under a
    # given collection policy, then split by pusher position.
    # --------------------------------------------------------------------- #
    def in_patch(S):
        c = np.asarray(cfg["patch_center"], np.float32)
        return np.linalg.norm(np.asarray(S)[:, :2] - c, axis=1) < cfg["patch_reg_mult"] * cfg["patch_sigma"]

    region = None
    if cfg["shift_mode"] == "patch":
        npp = cfg["n_eval_pool"] // 2
        Si, Ui, S2i, Ci = collect(env1, npp, cfg["seed"] + 41,
                                   center=cfg["patch_center"], pr_override=cfg["patch_sigma"])
        So, Uo, S2o, Co = collect(env1, npp, cfg["seed"] + 42)
        Se = np.concatenate([Si, So]); Ue = np.concatenate([Ui, Uo])
        S2e = np.concatenate([S2i, S2o]); Ce = np.concatenate([Ci, Co])
        m_in = in_patch(Se) & ~Ce
        m_out = (~in_patch(Se)) & ~Ce
        r2in_oracle = fm_free_r2(fm_oracle, Se[m_in], Ue[m_in], S2e[m_in], norm, None)
        r2in_stale = fm_free_r2(fm_d0, Se[m_in], Ue[m_in], S2e[m_in], norm, None)
        region = dict(n_in=int(m_in.sum()), n_out=int(m_out.sum()),
                      r2in_oracle=r2in_oracle, r2in_stale=r2in_stale)
        print(f"[2b] per-region eval: in-patch n={int(m_in.sum())} out n={int(m_out.sum())}  "
              f"in-patch R^2 stale={r2in_stale:.4f} -> oracle={r2in_oracle:.4f}", flush=True)

    def region_r2(fm, mask):
        if region is None or int(mask.sum()) < 20:
            return float("nan")
        return fm_free_r2(fm, Se[mask], Ue[mask], S2e[mask], norm, None)

    # ===================================================================== #
    # 3. DIRECTED collection (disagreement / error). Vectorized over B parallel
    #    central episodes; ensemble refit every retrain_every transitions.
    # ===================================================================== #
    def reset_member(m, k, perturb):
        """Warm-start member k from FM_d0; add a DETERMINISTIC per-member weight
        perturbation (so members disagree off-data even at 0 d1 data)."""
        m.load_state_dict(fm_d0_state)
        if not perturb:
            return
        mrng = np.random.default_rng(int(cfg["seed"]) * 10007 + 101 * k + 13)
        with torch.no_grad():
            for p in m.parameters():
                scale = cfg["ens_pert"] * (float(p.std()) + 1e-8)
                noise = torch.tensor(scale * mrng.standard_normal(tuple(p.shape)),
                                     dtype=p.dtype, device=device)
                p.add_(noise)

    def directed_collect(env, mode, max_N, seed):
        """mode='disagree': K FMs, score = var across members (NORMALIZED Δŝ units).
           mode='error':    1 refit FM, score = ||fm_single - fm_d0||^2 (predicted
                            novelty-vs-d0, a single-FM error surrogate)."""
        rng = np.random.default_rng(seed)
        B, M = cfg["dir_batch"], cfg["dir_cand"]
        ep_len, pr = cfg["collect_ep_len"], cfg["collect_pos_range"]
        eps_rand, warmup_N, retrain_every = cfg["dir_eps"], cfg["dir_warmup"], cfg["dir_retrain_every"]
        Km = cfg["ens_K"] if mode == "disagree" else 1
        members = [build_fm() for _ in range(Km)]
        n_refits = [0]

        def fit_ensemble():
            Sa = np.asarray(S, np.float32); Ua = np.asarray(U, np.float32); S2a = np.asarray(S2, np.float32)
            n = len(Sa)
            for k, mm in enumerate(members):
                reset_member(mm, k, perturb=(mode == "disagree"))
                if cfg["ens_bootstrap"] and mode == "disagree":
                    idx = rng.integers(0, n, size=n)
                else:
                    idx = np.arange(n)
                train_fm(mm, Sa[idx], Ua[idx], S2a[idx], norm, cfg["ens_epochs"], init_state=None)
            n_refits[0] += 1

        def score_batch(states, cand):
            # states (B,4), cand (B,M,2) -> disagreement/error per candidate (B,M)
            s_rep = np.repeat(states[:, None, :], M, axis=1)                 # (B,M,4)
            X = np.concatenate([s_rep, cand], 2).reshape(B * M, 6).astype(np.float32)
            with torch.no_grad():
                Xn = (torch.tensor(X, device=device) - norm["mx"]) / norm["sx"]
                preds = torch.stack([mm(Xn) for mm in members], 0)          # (Km,B*M,4)
                if mode == "disagree":
                    sc = preds.var(0).mean(1)
                else:
                    sc = (preds[0] - fm_d0(Xn)).pow(2).mean(1)
                return sc.reshape(B, M).cpu().numpy()

        # phase A: short RANDOM warmup (undirected central OU), then fit the ensemble
        wS, wU, wS2, wC = collect(env, warmup_N, seed + 5000)
        S = [r for r in wS]; U = [r for r in wU]; S2 = [r for r in wS2]; C = [bool(c) for c in wC]
        fit_ensemble()
        last_refit = len(S) // retrain_every

        # phase B: directed
        while len(S) < max_N:
            states = np.concatenate([rng.uniform(-pr, pr, (B, 2)),
                                     rng.normal(0, cfg["v_explore"], (B, 2))], 1).astype(np.float32)
            for _ in range(ep_len):
                if len(S) >= max_N:
                    break
                cand = rng.uniform(-1, 1, (B, M, 2)).astype(np.float32)
                score = score_batch(states, cand)
                best = score.argmax(1)
                rmask = rng.random(B) < eps_rand
                best = np.where(rmask, rng.integers(0, M, B), best)
                chosen = cand[np.arange(B), best]                            # (B,2)
                for b in range(B):
                    if len(S) >= max_N:
                        break
                    env.set_state(states[b, :2], states[b, 2:])
                    s2, info = env.step(chosen[b], fs)
                    S.append(states[b].copy()); U.append(chosen[b].astype(np.float32))
                    S2.append(s2); C.append(bool(info["any_contact"]))
                    states[b] = s2
                if len(S) // retrain_every > last_refit:
                    fit_ensemble(); last_refit = len(S) // retrain_every
        Sa = np.asarray(S, np.float32)[:max_N]; Ua = np.asarray(U, np.float32)[:max_N]
        S2a = np.asarray(S2, np.float32)[:max_N]; Ca = np.asarray(C, bool)[:max_N]
        print(f"    [directed:{mode}] N={len(Sa)} contact_frac={Ca.mean():.3f} "
              f"n_refits={n_refits[0]} warmup={warmup_N}", flush=True)
        return Sa, Ua, S2a, Ca

    print(f"\n[3] collecting reward-free d1 buffers (budget {cfg['max_budget']}) ...", flush=True)
    und_S, und_U, und_S2, und_C = collect(env1, cfg["max_budget"], cfg["seed"] + 11)
    print(f"    [undirected] N={len(und_S)} contact_frac={und_C.mean():.3f}", flush=True)
    dir_S, dir_U, dir_S2, dir_C = directed_collect(env1, "disagree", cfg["max_budget"], cfg["seed"] + 21)
    arms_raw = {"undirected": (und_S, und_U, und_S2, und_C),
                "directed": (dir_S, dir_U, dir_S2, dir_C)}
    if cfg.get("do_error_arm"):
        err_S, err_U, err_S2, err_C = directed_collect(env1, "error", cfg["max_budget"], cfg["seed"] + 31)
        arms_raw["error"] = (err_S, err_U, err_S2, err_C)

    # ===================================================================== #
    # 4. FAIR sweep: for each arm, first N transitions -> fresh FM warm-started
    #    from FM_d0 (IDENTICAL to Cut #3) -> planning + held-out R^2. Only data differs.
    # ===================================================================== #
    print(f"\n[4] reward-free re-adaptation sweep (budgets={cfg['budgets']}) ...", flush=True)
    budgets = cfg["budgets"]

    def sweep(name, bufS, bufU, bufS2, bufC):
        md, sc, r2, r2in, r2out = [], [], [], [], []
        for N in budgets:
            if N == 0:
                res, r2N, fmN = mb_d1_stale, r2_stale, fm_d0
            else:
                fmN = train_fm(build_fm(), bufS[:N], bufU[:N], bufS2[:N], norm,
                               cfg["adapt_epochs"], init_state=fm_d0_state)
                res = eval_mpc(env1, fmN, norm, cfg["seed"] + 300 + N, ev_starts, ev_goals)
                r2N = fm_free_r2(fmN, Sh, Uh, S2h, norm, Ch)
            md.append(res["mean_dist"]); sc.append(res["success"]); r2.append(r2N)
            r2in.append(region_r2(fmN, m_in) if region is not None else float("nan"))
            r2out.append(region_r2(fmN, m_out) if region is not None else float("nan"))
            print(f"    [{name}] N={N:6d}: dist={res['mean_dist']:.4f} succ={res['success']:.3f} "
                  f"R^2={r2N:.4f}  in-patch_R^2={r2in[-1]:.4f}", flush=True)
        recover_N = next((N for N, d in zip(budgets, md) if N > 0 and d <= ceil * 1.5), None)
        # patch-mode headline: transitions to recover the IN-PATCH FM fidelity
        recover_N_in = None
        if region is not None:
            thr_in = region["r2in_oracle"] * cfg["region_recover_frac"]
            recover_N_in = next((N for N, v in zip(budgets, r2in)
                                 if N > 0 and (v == v) and v >= thr_in), None)
        vis = float(in_patch(bufS).mean()) if region is not None else None
        return {"budgets": list(budgets), "mean_dist": md, "success": sc, "r2": r2,
                "r2_in": r2in, "r2_out": r2out, "recover_N": recover_N,
                "recover_N_in": recover_N_in, "patch_visitation": vis,
                "contact_frac": float(bufC.mean())}

    arms = {name: sweep(name, *buf) for name, buf in arms_raw.items()}

    # ===================================================================== #
    # assemble + figures
    # ===================================================================== #
    results = {
        "config": cfg, "shift_mode": cfg["shift_mode"], "dgp0": dgp0, "dgp1": dgp1,
        "fm_r2_d0": r2_d0, "fm_r2_oracle": r2_oracle, "fm_r2_stale_d1": r2_stale,
        "random_floor_d0": rand_d0, "random_floor_d1": rand_d1,
        "mb_d0": mb_d0, "mb_d1_stale": mb_d1_stale, "mb_d1_oracle": mb_d1_oracle,
        "oracle_ceiling": ceil, "recover_threshold": ceil * 1.5,
        "region": region, "budgets": list(budgets), "arms": arms, "H": H,
    }

    print("\n===== DIRECTED-READAPT SUMMARY =====", flush=True)
    print(f"shift_mode={cfg['shift_mode']}", flush=True)
    print(f"d0 competence:  MB dist={mb_d0['mean_dist']:.3f}/succ={mb_d0['success']:.2f}  "
          f"(random floor {rand_d0['mean_dist']:.2f})", flush=True)
    print(f"d1 zero-shot:   MB-stale dist={mb_d1_stale['mean_dist']:.3f}  "
          f"oracle ceiling {ceil:.3f}  (recover<= {ceil*1.5:.3f})", flush=True)
    for name, a in arms.items():
        print(f"{name:>11s} (planning): " + "  ".join(f"N{N}={d:.3f}" for N, d in zip(budgets, a["mean_dist"]))
              + f"   -> recover@{a['recover_N']}", flush=True)
    if region is not None:
        print(f"per-region (in-patch R^2 recovery; stale={region['r2in_stale']:.3f} "
              f"-> oracle={region['r2in_oracle']:.3f}, thr={region['r2in_oracle']*cfg['region_recover_frac']:.3f}):",
              flush=True)
        for name, a in arms.items():
            print(f"{name:>11s} (in-patch): " + "  ".join(f"N{N}={v:.3f}" for N, v in zip(budgets, a["r2_in"]))
                  + f"   -> recover@{a['recover_N_in']}  visitation={a['patch_visitation']:.3f}", flush=True)

    figures = _make_readapt_figures(results)

    outdir = os.path.join(DATA_DIR, "directed_readapt", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": results, "figures": figures}


def _make_readapt_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"undirected": "#888888", "directed": "#3d6fd1", "error": "#d1603d"}
    MRK = {"undirected": "s--", "directed": "o-", "error": "^-"}
    C_ORC = "#2f9e44"
    figs = {}
    budgets = R["budgets"]
    xN = [max(N, 1) for N in budgets]
    ceil = R["oracle_ceiling"]
    thr = R["recover_threshold"]
    d0comp = R["mb_d0"]["mean_dist"]
    stale = R["mb_d1_stale"]["mean_dist"]
    floor = R["random_floor_d1"]["mean_dist"]
    arms = R["arms"]

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # ---- fig1: recovery curve (planning final-dist vs #reward-free transitions) ----
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ytop = max(stale, max(a["mean_dist"][0] for a in arms.values())) * 1.35
    for name, a in arms.items():
        ax.plot(xN, a["mean_dist"], MRK[name], color=COL[name], lw=2.1,
                label=f"{name} (recover@{a['recover_N']})")
    ax.axhline(ceil, color=C_ORC, ls="-", lw=1.3, label=f"oracle ceiling {ceil:.3f}")
    ax.axhline(thr, color=C_ORC, ls="--", lw=1.0, label=f"recover threshold (1.5x) {thr:.3f}")
    ax.axhline(d0comp, color="#555", ls=":", lw=1.0, label=f"d0 competence {d0comp:.3f}")
    ax.annotate(f"random floor {floor:.2f}  ↑ (off scale)", xy=(0.5, 0.97),
                xycoords="axes fraction", ha="center", va="top", fontsize=8, color="#888")
    ax.set_ylim(0, ytop)
    ax.set_xscale("log")
    ax.set_xlabel("post-shift REWARD-FREE interaction (transitions)")
    ax.set_ylabel("final ||pusher_pos - goal||  (lower = better)")
    ax.set_title("Disagreement-directed vs undirected reward-free re-adaptation\n"
                 "(does directing collection recover the FM in fewer transitions?)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    figs["fig1_recovery_curve.png"] = _save(fig)

    # ---- fig2: transitions-to-recover per arm (the headline bar) ----
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    maxb = max(budgets)
    names = list(arms.keys())
    heights = [arms[n]["recover_N"] if arms[n]["recover_N"] is not None else maxb for n in names]
    reached = [arms[n]["recover_N"] is not None for n in names]
    xs = np.arange(len(names))
    bars = ax.bar(xs, heights, color=[COL[n] for n in names], alpha=0.85,
                  hatch=["" if r else "//" for r in reached])
    for xi, h, r, n in zip(xs, heights, reached, names):
        ax.text(xi, h, (f"{arms[n]['recover_N']}" if r else f"≥{maxb} (not reached)"),
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("transitions to recover (≤ 1.5x oracle ceiling)")
    ax.set_ylim(0, maxb * 1.25)
    ax.set_title("Transitions-to-recover per collection arm\n(lower = the drive earns its keep)",
                 fontsize=10)
    figs["fig2_transitions_to_recover.png"] = _save(fig)

    # ---- fig3: free-flight FM R^2 recovery (the model-fidelity view) ----
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for name, a in arms.items():
        ax.plot(xN, a["r2"], MRK[name], color=COL[name], lw=2.1, label=name)
    ax.axhline(R["fm_r2_oracle"], color=C_ORC, ls="-", lw=1.3,
               label=f"oracle FM R^2 {R['fm_r2_oracle']:.3f}")
    ax.axhline(R["fm_r2_stale_d1"], color="#555", ls=":", lw=1.0,
               label=f"stale FM R^2 (d1) {R['fm_r2_stale_d1']:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("post-shift REWARD-FREE interaction (transitions)")
    ax.set_ylabel("free-flight Δs R^2 on common held-out (higher = better)")
    ax.set_title("FM fidelity recovery (directed vs undirected)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    figs["fig3_r2_recovery.png"] = _save(fig)

    # ---- fig4 (PATCH mode only): in-patch R^2 recovery + patch visitation ----
    reg = R.get("region")
    if reg is not None:
        rfrac = R["config"]["region_recover_frac"]
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6),
                                       gridspec_kw={"width_ratios": [2, 1]})
        for name, a in arms.items():
            axA.plot(xN, a["r2_in"], MRK[name], color=COL[name], lw=2.1,
                     label=f"{name} (recover@{a['recover_N_in']})")
        axA.axhline(reg["r2in_oracle"], color=C_ORC, ls="-", lw=1.3,
                    label=f"oracle in-patch R^2 {reg['r2in_oracle']:.3f}")
        axA.axhline(reg["r2in_oracle"] * rfrac, color=C_ORC, ls="--", lw=1.0,
                    label=f"recover threshold ({rfrac:g}x)")
        axA.axhline(reg["r2in_stale"], color="#555", ls=":", lw=1.0,
                    label=f"stale in-patch R^2 {reg['r2in_stale']:.3f}")
        axA.set_xscale("log")
        axA.set_xlabel("post-shift REWARD-FREE transitions")
        axA.set_ylabel("in-patch free-flight Δs R^2  (re-learned the LOCAL change?)")
        axA.set_title("Localized shift: did the drive re-learn the patch in fewer transitions?", fontsize=10)
        axA.legend(fontsize=8, loc="lower right")
        names = list(arms.keys())
        vis = [arms[n]["patch_visitation"] for n in names]
        axB.bar(np.arange(len(names)), vis, color=[COL[n] for n in names], alpha=0.85)
        for xi, v in enumerate(vis):
            axB.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        axB.set_xticks(np.arange(len(names))); axB.set_xticklabels(names)
        axB.set_ylabel("fraction of collected transitions IN the patch")
        axB.set_title("Does the drive concentrate\non the scarce patch?", fontsize=9)
        figs["fig4_inpatch_recovery.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def directed_readapt(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # env / task (Cut #3's momentum-dominated free-flight reaching; d0->d1 drag collapse)
    frame_skip: int = 12,
    horizon: int = 24,
    goal_range: float = 0.5,
    start_range: float = 0.5,
    v0_std: float = 0.3,
    success_eps: float = 0.1,
    joint_damping: float = 2.0,        # d0: high drag
    gear: float = 10.0,
    arena_half: float = 2.0,
    shift_mass: float = 1.0,           # no mass change (damping-only operator shift)
    shift_damping: float = 0.05,       # d1: near-frictionless "ice"
    # LOCALIZED shift (the scarcity regime where the drive should earn its keep)
    shift_mode: str = "global",        # "global" (Cut #3 collapse; the null) | "patch"
    patch_center_x: float = 0.65,
    patch_center_y: float = 0.0,
    patch_sigma: float = 0.28,
    patch_force_x: float = 0.0,
    patch_force_y: float = 8.0,        # a strong localized force jet inside the patch
    patch_reg_mult: float = 1.5,       # in-patch eval radius = mult * sigma
    region_recover_frac: float = 0.9,  # in-patch R^2 recover threshold = frac * oracle
    n_eval_pool: int = 8000,           # balanced per-region R^2 eval pool
    v_explore: float = 1.2,
    collect_ep_len: int = 20,
    collect_pos_range: float = 1.0,    # central exploration box (arena +-2) -> ~no walls
    # MPC (CEM) -- value FIXED across d0/d1
    k_shoot: int = 256,
    h_plan: int = 15,
    cem_iters: int = 3,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    replan_every: int = 8,             # Cut #3's load-bearing commit length
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_epochs: int = 40,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    n_fm: int = 20000,
    adapt_epochs: int = 60,            # IDENTICAL to Cut #3's reward-free refit
    n_oracle: int = 20000,
    # directed collection (disagreement drive)
    ens_k: int = 4,                    # ensemble size
    ens_pert: float = 0.1,             # per-member init weight perturbation (frac of param std)
    ens_epochs: int = 30,              # epochs per ensemble refit
    ens_bootstrap: bool = True,        # bootstrap-resample the buffer per member
    dir_cand: int = 64,                # candidate actions scored per step
    dir_batch: int = 32,               # parallel central episodes
    dir_eps: float = 0.2,              # epsilon-random action mixing
    dir_warmup: int = 50,             # random warmup transitions (seed disagreement)
    dir_retrain_every: int = 250,      # refit ensemble every N transitions
    error_arm: bool = False,           # optional 3rd arm: single-FM error-vs-d0
    # eval
    n_eval: int = 96,
):
    import os

    # global: small-N regime where Cut #3 recovered (~50). patch: the scarce patch takes
    # undirected FAR longer to cover, so budgets extend much higher.
    if shift_mode == "patch":
        budgets = [0, 100, 250, 500, 1000, 2000, 4000]
    else:
        budgets = [0, 25, 50, 100, 200, 400, 800, 2000]

    if quick:
        n_fm, fm_epochs, fm_hidden, fm_layers = 3000, 15, 128, 2
        n_oracle = 2000
        budgets = [0, 150, 500, 1500] if shift_mode == "patch" else [0, 25, 50, 150, 400]
        adapt_epochs = 30
        k_shoot, h_plan, horizon = 96, 12, 26
        cem_elite = 12
        replan_every = min(replan_every, h_plan)
        n_eval = 24
        n_eval_pool = 3000
        ens_k, ens_epochs = 3, 15
        dir_cand, dir_batch = 48, 16
        dir_warmup, dir_retrain_every = 50, 120
        tag = tag or "smoke"
    tag = tag or "default"

    max_budget = max(budgets)

    cfg = dict(
        tag=tag, seed=seed,
        frame_skip=frame_skip, H=horizon, goal_range=goal_range, start_range=start_range,
        v0_std=v0_std, success_eps=success_eps, v_explore=v_explore,
        collect_ep_len=collect_ep_len, collect_pos_range=collect_pos_range,
        dgp_base=dict(arena_half=arena_half, pusher_mass=1.0,
                      joint_damping=joint_damping, gear=gear),
        shift_mass=shift_mass,
        shift_damping=(None if shift_damping < 0 else shift_damping),
        shift_mode=shift_mode,
        patch_center=[patch_center_x, patch_center_y], patch_sigma=patch_sigma,
        patch_force=[patch_force_x, patch_force_y], patch_reg_mult=patch_reg_mult,
        region_recover_frac=region_recover_frac, n_eval_pool=n_eval_pool,
        K=k_shoot, H_plan=h_plan, vel_pen=vel_pen, replan_every=replan_every,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_epochs=fm_epochs, fm_lr=fm_lr,
        fm_batch=fm_batch, n_fm=n_fm, adapt_epochs=adapt_epochs, n_oracle=n_oracle,
        ens_K=ens_k, ens_pert=ens_pert, ens_epochs=ens_epochs, ens_bootstrap=ens_bootstrap,
        dir_cand=dir_cand, dir_batch=dir_batch, dir_eps=dir_eps, dir_warmup=dir_warmup,
        dir_retrain_every=dir_retrain_every, do_error_arm=error_arm,
        budgets=budgets, max_budget=max_budget, n_eval=n_eval,
    )
    out = run_directed_readapt.remote(cfg)

    localdir = os.path.join(os.path.dirname(__file__), "figures", "directed_readapt_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
