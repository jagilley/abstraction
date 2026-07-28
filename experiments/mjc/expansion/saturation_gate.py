"""Cut #5, Piece 1 — THE SATURATION GATE. Does the frontier instrument read on a control
substrate at all, and where?

Program: `beliefs/dimensionality_expansion.md` (the belief), `ideas/physical_control_substrate.md`
cut #5 (the flagship, unrun), `experiments/mjc/README.md` §Next steps #3 (the corrected spec).
Instrument of record: `rhm/residual_decomposition/README.md` — beta, `R_res_participation`,
frontier mass. This file imports that module rather than vendoring it.

WHY THIS RUNS BEFORE ANYTHING ELSE. Rank has failed as an instrument in this repo three times,
and the re-measurement found ONE cause for all three: a SATURATED forward model. Where the FM
drives relative residual to ~0, beta collapses to 0.11-0.17 and naive residual rank inflates to
94-96% of d_model in every domain tested -- pure noise floor. An arm FM doing (s,u) -> s' is the
NARROWEST prediction gap available (one step), so cut #5 as originally scoped may have no signal
to find on this substrate at all. That is a one-job question and this is the job. Nothing else in
cut #5 is worth building until it answers.

THE PREDICTION BEING TESTED (mjc/README.md §Next steps #3): the expansion signature, if it exists,
shows up at the BALLISTIC HORIZON and is absent at k=1. Note what gap width corresponds to here --
reactive control re-grounds every step (narrow gap), ballistic commits for a horizon (wide gap) --
so **gap width on this substrate IS the commitment-under-delay axis** the whole `ballistic/` line
is about (`ideas/heterogeneous_graders.md` §2 condition 1). If the signature is absent at both,
cut #5 closes on the state-space prediction target and we say so.

WHAT IS MEASURED. For each rollout horizon k, the FM's k-step DISPLACEMENT prediction:

    A_i = s_{t+k} - s_t                      (true, from the plant)
    P_i = the FM rolled out k steps from s_t under the executed command sequence
    Residual = A - P

then `full_decomposition(A, P)` from the rhm module. Displacement, not absolute state: s_{t+k} is
dominated by s_t, which the FM is GIVEN rather than predicting, so an absolute target would report
rho ~ 1 everywhere for trivial reasons.

THE BASIS, AND THE DESIGN GAP THIS PIECE DELIBERATELY ACCEPTS. `R_res_participation` counts the
frontier in "the model's own basis, weighted by the computation actually done" -- which is exactly
why it reads 7 on RHM where naive rank reads 84. Here the basis is A's principal directions, i.e.
the PLANT's displacement covariance on the probe set: an ordering of the DATA, not of a model's
computation. That is a weakened instrument, not a broken one, and it is the cheap half of the
choice recorded in mjc/README.md §Next steps #3 (whose spec -- "basis = the FM's hidden covariance,
residual = k-step rollout error against the plant" -- is not computable as written, since
`repaired_triple` needs A and P in the SAME space and those are 256-d and 2n-d). The honest
alternative is to predict a learned latent, which is what makes the basis well-defined and what
beta actually needs; it is deliberately deferred until this job says whether the cheap version has
any dynamic range.

SCALING. q (rad) and qd (rad/s) are different units, so a raw covariance basis would be a units
artifact -- the velocity dims would define every principal direction. A and P are therefore both
divided by A's per-dim std on the probe (scale only, correlations preserved, applied identically
to both so the residual relation is exact). The unscaled version is computed alongside as a
sensitivity check, not as the headline.

FOUR AXES, ONE JOB:
  1. **k** (the gap): 1, 2, 4, 8, 14, 20. k=1 is what every cut in this tree has ever fitted;
     k=14 is `plan_H`, the ballistic commitment horizon; k=20 is past the n=5 composition horizon
     (arm_substrate P4: n=5 is 14 steps, n=3 is 20-23), i.e. deliberately where the FM is no
     longer trustworthy.
  2. **n_links** (does capacity bind): n=3 has a capacity requirement of 32 against our h=256 FM
     -- MASSIVE slack, so saturation is expected. n=5 requires 256, i.e. the FM sits at its own
     frontier. arm_substrate P1 makes this a cheap capacity-binding axis and cut #5's own design
     constraint ("chain >= 5, or expansion is free and therefore not a decision").
  3. **FM variant**: `matched` (trained in the operating world) is primary; `stale` (trained
     field-free, graded under the curl) is the POSITIVE CONTROL -- an instrument that cannot tell
     a known-wrong model from a right one certainly cannot detect an expansion; `small` (h=64 vs
     256) is the capacity-invariance check that beta is reading the computation and not the FM,
     which is the gate that would catch this readout if it were circular.
  4. **probe distribution**: `task` (transitions a matched-FM reach actually visits -- the rule
     from arm_substrate finding #1, and the distribution every FM-error number in this tree is
     graded on) vs `broad` (the teleport collection pool). Reported side by side because the basis
     is defined by the probe and E0 already showed the broad teleport distribution is mistuned.

BETA IS THIN HERE AND IS LABELLED AS SUCH. beta is a log-log slope across PARTICIPATING directions;
the arm's state target gives 2n = 6 (n=3) or 10 (n=5) of them. `rhm`'s own `shadow_law` returns NaN
below 8 directions, which n=3 cannot clear at all. The fit below relaxes that guard but ALWAYS
reports `n_dirs` beside the number. Frontier mass and `R_res_participation` survive at this size;
beta does not, and this is the second independent argument for lifting the prediction target to a
learned latent (mjc/README.md §Next steps #3).

THE INSTRUMENT IS DEGENERATE AT BOTH ENDS, AND THE SMOKE SHOWED THE SECOND ONE. The known failure
is the lower one: a SATURATED FM drives frontier mass to ~0 and the leftover is float noise. But
the k-sweep also has an UPPER degeneracy that the rhm domains never reached, because they never
rolled a model out: once the open-loop rollout diverges, the FM explains none of the displacement,
frontier mass runs to ~1, every rho_i -> 0, and there is again no graded structure to read -- "the
model has nothing to say" is exactly as uninformative as "the model said everything". So the
readable regime is a WINDOW, `frontier_floor < frontier_mass < frontier_ceiling`, and the verdict
below reports which k values fall inside it rather than assuming k=plan_H does.

PRE-REGISTERED DECISION RULE (printed as a verdict block at the end; primary = matched FM, task
probe, h=256):
  * READS  -- there is a non-empty usable window of k (floor < frontier mass < ceiling), frontier
              mass RISES with k across it, and somewhere in that window the `stale` FM separates
              from `matched` by at least `stale_ratio`.
              -> cut #5 proceeds to Piece 2 (support-fixed null) on the state-space target.
  * LATENT -- the window exists but stale does not separate anywhere inside it (or the window is
              a single k). The gap axis is alive but the readout cannot tell a known-wrong model
              from a right one at this dimensionality.
              -> lift the prediction target to a learned latent before Piece 2.
  * CLOSE  -- the window is empty: frontier mass is at the floor for small k and through the
              ceiling for large k, with no k in between, in both probes.
              -> the frontier instrument does not read on this substrate; close cut #5 on the
              state-space target and record it.

Substrate config is arm_substrate P5 / `ballistic/arm` BYTE-IDENTICAL at n=3 (joint_damping=0.5,
gear=8.0, frame_skip=10, q_range=0.9, v_explore=8.0, plan_H=14, k_shoot=1024, cem_iters=8), so
this calibrates the FM those cuts actually ran. Collection is TELEPORT on purpose: this is an
instrument calibration OF THE EXISTING CUTS, not a new cut, so the node's on-policy convention
(README §Collection convention) does not apply here and would change the thing being calibrated.
Pieces 2 and 3 are new cuts and go on-policy.

Run:
    cd experiments/
    modal run mjc/expansion/saturation_gate.py::saturation_gate --quick            # smoke
    modal run --detach mjc/expansion/saturation_gate.py::saturation_gate --tag gate_v1
"""

import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


def shadow_law_flex(act_var, res_var, min_dirs: int = 4):
    """res_var(i) ~ act_var(i)^beta over A's principal directions. Returns (beta, r2, n_dirs).

    Identical fit to `rhm.residual_decomposition.analyze_summaries.shadow_law`, with its
    hard `min_dirs=8` guard relaxed and made explicit -- the arm's state target has only
    2n = 6 or 10 directions, so the rhm default returns NaN at n=3 by construction. NEVER
    read beta here without n_dirs beside it: an R^2 over six points is not evidence of a law.
    """
    import numpy as np

    a = np.asarray(act_var, dtype=float)
    v = np.asarray(res_var, dtype=float)
    msk = (a > 0) & (v > 0)
    if int(msk.sum()) < min_dirs:
        return float("nan"), float("nan"), int(msk.sum())
    beta, c = np.polyfit(np.log(a[msk]), np.log(v[msk]), 1)
    pred = beta * np.log(a[msk]) + c
    lv = np.log(v[msk])
    denom = ((lv - lv.mean()) ** 2).sum()
    r2 = 1.0 - ((lv - pred) ** 2).sum() / denom if denom > 0 else float("nan")
    return float(beta), float(r2), int(msk.sum())


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_saturation_gate(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs, H = cfg["frame_skip"], cfg["plan_H"]
    b0, b1 = cfg["b0"], cfg["b1"]
    ks = cfg["k_list"]
    k_max = max(ks)
    T = cfg["n_legs"] * H                      # trajectory length; needs T >= k_max + slack
    assert T > k_max, f"trajectory length {T} must exceed k_max {k_max}"
    print(f"[setup] device={device} n_links={cfg['n_links_list']} curl b0={b0}->b1={b1} "
          f"k_list={ks} T={T} n_traj={cfg['n_traj']} n_windows={cfg['n_windows']}", flush=True)

    rows = []
    per_n_meta = {}

    for n in cfg["n_links_list"]:
        SD, AD = 2 * n, n
        Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
        Ms = list(cfg["link_masses"][:n])
        qc = np.asarray(cfg["q_center"][:n], dtype=np.float64)
        Lt = torch.tensor(Ls, device=device, dtype=torch.float32)

        def make_env(b):
            return ArmEnv(dict(n_links=n, link_lengths=list(Ls), link_masses=Ms,
                               joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                               curl_field={"b": float(b)}))

        env1 = make_env(b1)        # the OPERATING world: probes, grading, matched/small FMs
        env0 = make_env(b0)        # field-free: where the `stale` FM is trained

        def fk_torch(q):
            ang = torch.cumsum(q, dim=1)
            return torch.stack([(Lt * torch.cos(ang)).sum(1),
                                (Lt * torch.sin(ang)).sum(1)], 1)

        # ---------------------------------------------------------------- FM f(s,u)->Δs
        def _mlp(seed, hidden):
            g = torch.Generator(device="cpu").manual_seed(seed)
            lyr = [nn.Linear(SD + AD, hidden), nn.SiLU()]
            for _ in range(cfg["fm_layers"] - 1):
                lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
            net = nn.Sequential(*(lyr + [nn.Linear(hidden, SD)]))
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g)
                    nn.init.zeros_(m.bias)
            return net.to(device)

        def pool(cenv, nn_, seed):
            return collect_pool(cenv, nn_, np.random.default_rng(seed), fs,
                                qc, cfg["q_range"], cfg["v_explore"])

        # Normalization fixed ONCE from the operating dynamics and shared by every FM
        # variant, so all three are scored in the same units (the `ballistic/arm` rule).
        nS, nU, nS2 = pool(env1, cfg["pool_n"], cfg["seed"] + 11)
        X = np.concatenate([nS, nU], 1).astype(np.float32)
        Y = (nS2 - nS).astype(np.float32)
        norm = {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

        huber = nn.HuberLoss(delta=1.0)      # cut #1: Huber, not MSE

        def train_fm(net, S, U, S2, seed):
            opt = torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"])
            Xt = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
            Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
            Xn = (Xt - norm["mx"]) / norm["sx"]; Yn = (Yt - norm["my"]) / norm["sy"]
            bs = min(cfg["fm_batch"], len(S)); brng = np.random.default_rng(seed)
            net.train()
            for _ in range(cfg["fm_steps"]):
                idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
                opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            return net

        def fm_delta_t(net, s_t, u_t):
            """Torch, no grad: (N,SD),(N,AD) -> predicted Δs in RAW units."""
            x = torch.cat([s_t, u_t], 1)
            return net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]

        def fm_rollout(net, S0, Useq):
            """k-step open-loop rollout. S0 (N,SD), Useq (N,k,AD) -> predicted displacement."""
            with torch.no_grad():
                s0 = torch.tensor(S0, device=device, dtype=torch.float32)
                us = torch.tensor(Useq, device=device, dtype=torch.float32)
                s = s0.clone()
                for j in range(us.shape[1]):
                    s = s + fm_delta_t(net, s, us[:, j, :])
                return (s - s0).cpu().numpy().astype(np.float64)

        # ------------------------------------------------------ the three FM variants
        mS, mU, mS2 = pool(env1, cfg["pool_n"], cfg["seed"] + 20)
        sS, sU, sS2 = pool(env0, cfg["pool_n"], cfg["seed"] + 21)
        nets = {}
        nets["matched"] = train_fm(_mlp(cfg["seed"] + 41, cfg["fm_hidden"]),
                                   mS, mU, mS2, cfg["seed"] + 301)
        # `small` sees the SAME data as `matched`; capacity is the only difference, which is
        # what makes it a capacity-invariance check on beta rather than a data ablation.
        nets["small"] = train_fm(_mlp(cfg["seed"] + 43, cfg["fm_hidden_small"]),
                                 mS, mU, mS2, cfg["seed"] + 303)
        nets["stale"] = train_fm(_mlp(cfg["seed"] + 42, cfg["fm_hidden"]),
                                 sS, sU, sS2, cfg["seed"] + 302)
        print(f"[n={n}] FMs trained: matched/small(h={cfg['fm_hidden_small']}) on b1, "
              f"stale on b0", flush=True)

        # ---------------------------------------------------------------- CEM planner
        Kc, ne_el = cfg["k_shoot"], cfg["cem_elite"]

        def mpc_plan(net, states, goals, rng):
            Bn = states.shape[0]
            mu = np.zeros((Bn, H, AD), np.float32)
            sig = np.full((Bn, H, AD), cfg["cem_init_sigma"], np.float32)
            g_t = torch.tensor(goals, device=device, dtype=torch.float32).repeat_interleave(Kc, 0)
            s0 = torch.tensor(states, device=device, dtype=torch.float32).repeat_interleave(Kc, 0)
            for _ in range(cfg["cem_iters"]):
                e = rng.standard_normal((Bn, Kc, H, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    seqs_t = torch.tensor(seqs.reshape(Bn * Kc, H, AD), device=device)
                    cost = torch.zeros(Bn * Kc, device=device)
                    for h in range(H):
                        s = s + fm_delta_t(net, s, seqs_t[:, h, :])
                        cost = cost + (fk_torch(s[:, :n]) - g_t).norm(dim=1)
                    cost = cost + cfg["vel_pen"] * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, Kc), ne_el, dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1); sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)

        def sample_goals(q0, rng):
            """Reach-band-filtered goals, sampled in JOINT space (no IK) then FK'd.

            The band rejection is load-bearing on a redundant arm: a random joint delta
            usually lands in the null space and barely moves the tip, which silently fills
            the probe with trivial reaches (arm_substrate `eval_geometry`). Vectorized over
            candidates; picks the minimum-penalty candidate, so it always returns something.
            """
            t0 = fk(q0, Ls)
            d = rng.normal(0, 1, (cfg["reach_tries"], len(q0), n))
            d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
            cand = q0[None] + cfg["reach_amp"] * d
            tips = fk(cand, Ls)
            dist = np.linalg.norm(tips - t0[None], axis=-1)
            pen = (np.maximum(0.0, cfg["reach_lo"] - dist)
                   + np.maximum(0.0, dist - cfg["reach_hi"]))
            best = np.argmin(pen, axis=0)
            ar = np.arange(len(q0))
            return tips[best, ar].astype(np.float32), float(np.mean(dist[best, ar]))

        # ------------------------------------------------- the two trajectory probes
        def task_trajectories(net):
            """`n_legs` chained BALLISTIC reaches executed in the true world, recording
            (state, command) at every step. Chained rather than one reach so the trajectory
            is longer than k_max while staying continuously ON TASK -- a single H-step reach
            would give exactly one window at k=H, and padding it by hovering at the goal
            would fill the probe with near-zero-velocity states and distort the basis."""
            rng = np.random.default_rng(cfg["seed"] + 700)
            N = cfg["n_traj"]
            q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (N, n))
            s = np.concatenate([q0, rng.normal(0, cfg["v0_std"], (N, n))], 1).astype(np.float32)
            S_hist = [s.copy()]; U_hist = []
            for leg in range(cfg["n_legs"]):
                goals, mean_reach = sample_goals(s[:, :n].astype(np.float64), rng)
                plan = np.empty((N, H, AD), np.float32)
                ch = cfg["plan_chunk"]
                for i in range(0, N, ch):
                    j = min(i + ch, N)
                    plan[i:j] = mpc_plan(net, s[i:j], goals[i:j], rng)
                for h in range(H):
                    u = plan[:, h, :]
                    for b in range(N):
                        env1.set_state(s[b, :n].astype(np.float64), s[b, n:].astype(np.float64))
                        s[b], _ = env1.step(u[b], fs)
                    S_hist.append(s.copy()); U_hist.append(u.copy())
                print(f"[n={n}][task-probe] leg {leg}: mean reach {mean_reach:.3f} m", flush=True)
            return np.stack(S_hist, 1), np.stack(U_hist, 1)

        def broad_trajectories():
            """Teleported starts from the COLLECTION distribution, driven by smoothed
            (OU) random commands. The explicit contrast to the task probe: E0 already showed
            this distribution is mistuned relative to what the task visits, and the basis is
            defined by whichever probe you use, so both are reported."""
            rng = np.random.default_rng(cfg["seed"] + 800)
            N = cfg["n_traj"]
            q0 = qc[None, :] + rng.uniform(-cfg["q_range"], cfg["q_range"], (N, n))
            qd0 = rng.normal(0.0, cfg["v_explore"], (N, n))
            s = np.concatenate([q0, qd0], 1).astype(np.float32)
            u = rng.uniform(-1, 1, (N, AD)).astype(np.float32)
            rho = cfg["ou_rho"]
            S_hist = [s.copy()]; U_hist = []
            for _ in range(T):
                u = np.clip(rho * u + np.sqrt(1 - rho ** 2)
                            * rng.normal(0, 1, (N, AD)), -1, 1).astype(np.float32)
                for b in range(N):
                    env1.set_state(s[b, :n].astype(np.float64), s[b, n:].astype(np.float64))
                    s[b], _ = env1.step(u[b], fs)
                S_hist.append(s.copy()); U_hist.append(u.copy())
            return np.stack(S_hist, 1), np.stack(U_hist, 1)

        # `max_absq` is measured PER PROBE (reset_wrap between them) because it is a lifetime
        # max on the shared env otherwise, and the two probes leave the legible region for
        # completely different reasons. arm_substrate's reference is max|q| = 2.3-2.9 rad under
        # controlled rollouts; `ArmEnv.wrapped()` flags > 3.0 rad. A broad probe teleported to
        # qd ~ N(0, v_explore=8) and driven open-loop for T*frame_skip physics steps WILL wrap,
        # which is a fact about that probe, not a bug — but it must be visible, because a basis
        # measured on a wrapped trajectory is a basis on states the FM was never trained on.
        probes, probe_meta = {}, {}
        for pname, build in (("task", lambda: task_trajectories(nets["matched"])),
                             ("broad", broad_trajectories)):
            env1.reset_wrap()
            nf0 = env1.nonfinite()
            probes[pname] = build()
            tS, tU = probes[pname]
            probe_meta[pname] = {"max_absq": float(env1.max_absq()),
                                 "wrapped": bool(env1.wrapped()),
                                 "nonfinite": int(env1.nonfinite() - nf0)}
            print(f"[n={n}][probe:{pname}] trajectories {tS.shape} commands {tU.shape} "
                  f"max|q|={probe_meta[pname]['max_absq']:.2f} rad "
                  f"wrapped={probe_meta[pname]['wrapped']} "
                  f"nonfinite={probe_meta[pname]['nonfinite']}", flush=True)
        per_n_meta[str(n)] = probe_meta

        # ------------------------------------------------- windows -> decomposition
        def windows(tS, tU, k, seed):
            """One window per drawn episode at a uniformly random valid offset, so the
            SAMPLE COUNT IS IDENTICAL at every k. Effective rank and participation ratio
            are both sample-size sensitive at small N, so letting n_windows grow as k
            shrinks would confound the whole sweep."""
            rng = np.random.default_rng(seed)
            N, Tn = tU.shape[0], tU.shape[1]
            nw = cfg["n_windows"]
            ep = rng.integers(0, N, nw)
            off = rng.integers(0, Tn - k + 1, nw)
            S0 = tS[ep, off]
            A = (tS[ep, off + k] - S0).astype(np.float64)
            U = np.stack([tU[ep, off + j] for j in range(k)], 1)
            return S0, U, A

        for pname, (tS, tU) in probes.items():
            for vname, net in nets.items():
                for k in ks:
                    S0, Useq, A = windows(tS, tU, k, cfg["seed"] + 900 + k)
                    P = fm_rollout(net, S0, Useq)
                    # THE MEAN-BLINDNESS DIAGNOSTIC. `repaired_triple` mean-centers both A and
                    # the residual, so any part of the residual that is a CONSTANT OFFSET is
                    # invisible to frontier mass, rho, and beta alike. That matters specifically
                    # here and did not matter in RHM/language: a stale FM under a curl field is
                    # wrong in a systematic, one-signed way (it omits a force), so its residual
                    # can be dominated by exactly the component the instrument discards — which
                    # would let a KNOWN-WRONG model read as having a SMALLER frontier than a
                    # matched one. The smoke showed that inversion at n=5, so the fraction is
                    # recorded at every cell rather than inferred after the fact.
                    Rres = A - P
                    resid_mean_frac = float((Rres.mean(axis=0) ** 2).sum()
                                            / max((Rres ** 2).sum(axis=1).mean(), 1e-30))
                    sd = A.std(axis=0) + 1e-12
                    dec = full_decomposition(A / sd, P / sd, ks=tuple(range(1, SD)),
                                             seed=cfg["seed"])
                    beta, br2, ndir = shadow_law_flex(
                        dec["geometry"]["act_variance_spectrum"],
                        dec["geometry"]["res_variance_in_act_basis"],
                        min_dirs=cfg["beta_min_dirs"])
                    # sensitivity: same quantities WITHOUT the per-dim scaling, so a reader
                    # can see how much of the basis is a rad-vs-rad/s units artifact.
                    dec_raw = full_decomposition(A, P, ks=tuple(range(1, SD)), seed=cfg["seed"])
                    beta_raw, _, _ = shadow_law_flex(
                        dec_raw["geometry"]["act_variance_spectrum"],
                        dec_raw["geometry"]["res_variance_in_act_basis"],
                        min_dirs=cfg["beta_min_dirs"])
                    row = {
                        "n_links": n, "probe": pname, "variant": vname, "k": k, "d_state": SD,
                        "rel_residual": dec["basic"]["relative_residual"],
                        "cosine": dec["basic"]["mean_cosine"],
                        "frontier_mass": dec["repaired"]["frontier_mass"],
                        "R_res_participation": dec["repaired"]["R_res_participation"],
                        "R_act_pr": dec["repaired"]["R_act_pr"],
                        "naive_R_res": dec["naive"]["R_res"],
                        "naive_R_act": dec["naive"]["R_act"],
                        "alignment_index": dec["geometry"]["alignment_index"],
                        "absorption_mean": dec["geometry"]["absorption_mean"],
                        "beta": beta, "beta_r2": br2, "beta_n_dirs": ndir,
                        "frontier_mass_raw": dec_raw["repaired"]["frontier_mass"],
                        "R_res_participation_raw": dec_raw["repaired"]["R_res_participation"],
                        "beta_raw": beta_raw,
                        # raw (unscaled) magnitudes, so these numbers are directly comparable
                        # to `fm_err` as reported by ballistic/arm and arm_substrate P5.
                        "mean_residual_norm": dec_raw["basic"]["mean_residual_norm"],
                        "mean_target_norm": dec_raw["basic"]["mean_target_norm"],
                        "resid_mean_frac": resid_mean_frac,
                        "act_variance_spectrum": dec["geometry"]["act_variance_spectrum"],
                        "res_variance_in_act_basis": dec["geometry"]["res_variance_in_act_basis"],
                    }
                    rows.append(row)
                    print(f"[n={n}][{pname:5s}][{vname:7s}] k={k:2d}  "
                          f"rel_res={row['rel_residual']:.4f}  "
                          f"frontier={row['frontier_mass']:.4f}  "
                          f"R_res_part={row['R_res_participation']:.2f}/{SD}  "
                          f"naive_R_res={row['naive_R_res']:.2f}  "
                          f"align={row['alignment_index']:+.3f}  "
                          f"resid_mean={resid_mean_frac:.3f}  "
                          f"beta={beta:.3f} (R2={br2:.3f}, {ndir} dirs)", flush=True)

    # ------------------------------------------------------------------ the verdict
    def get(n, probe, variant, k, key):
        for r in rows:
            if (r["n_links"] == n and r["probe"] == probe
                    and r["variant"] == variant and r["k"] == k):
                return r[key]
        return None

    verdict = {}
    floor, ceil_ = cfg["frontier_floor"], cfg["frontier_ceiling"]
    print("\n" + "=" * 104, flush=True)
    print("VERDICT — pre-registered decision rule (primary: matched FM, task probe)", flush=True)
    print(f"  usable window: {floor} < frontier_mass < {ceil_}  "
          f"(below = saturated FM / noise floor; above = diverged rollout, nothing explained)",
          flush=True)
    print("=" * 104, flush=True)
    for n in cfg["n_links_list"]:
        print(f"  --- n_links={n} (d_state={2*n}) — task probe, per k ---", flush=True)
        window, ratios = [], {}
        for k in ks:
            fm_ = get(n, "task", "matched", k, "frontier_mass")
            st_ = get(n, "task", "stale", k, "frontier_mass")
            sm_ = get(n, "task", "small", k, "frontier_mass")
            pr_ = get(n, "task", "matched", k, "R_res_participation")
            rr_ = get(n, "task", "matched", k, "rel_residual")
            inw = (fm_ is not None and floor < fm_ < ceil_)
            ratio = (st_ / max(fm_, 1e-12)) if (st_ is not None and fm_ is not None) else float("nan")
            if inw:
                window.append(k); ratios[k] = ratio
            mf_ = get(n, "task", "matched", k, "resid_mean_frac")
            mfs = get(n, "task", "stale", k, "resid_mean_frac")
            print(f"    k={k:2d}  rel_res={rr_:.4f}  frontier={fm_:.4f}  "
                  f"R_res_part={pr_:.2f}/{2*n}  |  stale={st_:.4f} (x{ratio:.2f})  "
                  f"small={sm_:.4f}  |  resid_mean m/s={mf_:.3f}/{mfs:.3f}"
                  f"  |  {'IN WINDOW' if inw else '--'}", flush=True)
        rises = (len(window) >= 2
                 and get(n, "task", "matched", window[-1], "frontier_mass")
                 > get(n, "task", "matched", window[0], "frontier_mass"))
        best_k = max(ratios, key=lambda kk: ratios[kk]) if ratios else None
        stale_sep = best_k is not None and ratios[best_k] >= cfg["stale_ratio"]
        if window and rises and stale_sep:
            call = "READS"
        elif window:
            call = "LATENT"
        else:
            call = "CLOSE"
        verdict[str(n)] = {
            "usable_window_k": window, "frontier_rises_across_window": bool(rises),
            "stale_ratio_by_k": ratios, "best_separating_k": best_k,
            "best_stale_ratio": ratios.get(best_k) if best_k is not None else None,
            "stale_separates": bool(stale_sep), "call": call,
            "rel_residual_k1": get(n, "task", "matched", 1, "rel_residual"),
            "frontier_mass_k1": get(n, "task", "matched", 1, "frontier_mass"),
            "beta_at_best_k": (get(n, "task", "matched", best_k, "beta")
                               if best_k is not None else None),
            "beta_capacity_delta_at_best_k": (
                abs(get(n, "task", "matched", best_k, "beta")
                    - get(n, "task", "small", best_k, "beta"))
                if best_k is not None else None),
        }
        print(f"    ==> window={window or 'EMPTY'}  rises={rises}  "
              f"best stale separation x{ratios.get(best_k, float('nan')):.2f} at k={best_k}"
              f"  ==> {call}", flush=True)
    print("\n  READS  -> proceed to Piece 2 on the state-space target", flush=True)
    print("  LATENT -> lift the prediction target to a learned latent before Piece 2", flush=True)
    print("  CLOSE  -> the frontier instrument does not read here; record and close", flush=True)
    print(f"\n  env sanity (joint excursion / diverged steps): {per_n_meta}", flush=True)
    print("=" * 104 + "\n", flush=True)

    out = {"config": cfg, "rows": rows, "verdict": verdict, "env_meta": per_n_meta}
    outdir = os.path.join(DATA_DIR, "expansion_gate", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)

    # ---------------------------------------------------------------------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = [("rel_residual", "relative residual"),
                   ("frontier_mass", "frontier mass"),
                   ("R_res_participation", "R_res_participation")]
        ns = cfg["n_links_list"]
        fig, axes = plt.subplots(len(ns), 3, figsize=(14, 4.0 * len(ns)), squeeze=False)
        colors = {"matched": "#1f77b4", "stale": "#d62728", "small": "#7f7f7f"}
        for i, n in enumerate(ns):
            for j, (key, lab) in enumerate(metrics):
                ax = axes[i][j]
                for vname in ("matched", "stale", "small"):
                    for pname, ls in (("task", "-"), ("broad", "--")):
                        ys = [get(n, pname, vname, k, key) for k in ks]
                        ax.plot(ks, ys, ls, marker="o", ms=3, color=colors[vname],
                                alpha=1.0 if pname == "task" else 0.45,
                                label=f"{vname}/{pname}" if (i == 0 and j == 0) else None)
                ax.axvline(H, color="k", lw=0.8, ls=":", alpha=0.6)
                ax.set_xlabel("rollout horizon k (steps)")
                ax.set_ylabel(lab)
                ax.set_title(f"n_links={n} — {lab}")
                if key == "R_res_participation":
                    ax.axhline(2 * n, color="k", lw=0.8, ls="-.", alpha=0.4)
                if key in ("rel_residual", "frontier_mass"):
                    ax.set_yscale("log")
        axes[0][0].legend(fontsize=7, ncol=2)
        fig.suptitle("Cut #5 Piece 1 — saturation gate: does the frontier instrument read?\n"
                     "solid = task probe, faded dashed = broad probe; dotted vline = ballistic "
                     "horizon (plan_H)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(outdir, "fig_saturation_gate.png"), dpi=150)
        plt.close(fig)
        print(f"[figure] wrote fig_saturation_gate.png", flush=True)
    except Exception as e:                                     # never lose results to a plot
        print(f"[figure] skipped: {e}", flush=True)

    volume.commit()
    print(f"[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def saturation_gate(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # --- the gap sweep: 1 = every cut in this tree; 14 = plan_H (ballistic commitment);
    #     20 = past the n=5 composition horizon (arm_substrate P4) ---
    k_list: str = "1,2,3,4,6,8,10,14,20",
    # --- capacity-binding axis: n=3 has huge slack (cap req 32), n=5 binds (256) ---
    n_links_list: str = "3,5",
    link_lengths: str = "0.4,0.4,0.3,0.25,0.2",
    link_masses: str = "1.0,1.0,0.6,0.4,0.3",
    q_center: str = "0.4,0.8,0.6,0.4,0.3",
    # --- the drift, for the `stale` positive control: Shadmehr curl, field-free -> field ---
    b0: float = 0.0,
    b1: float = 6.0,
    # --- substrate: arm_substrate P5 / ballistic/arm defaults, held byte-identical ---
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    # --- FM ---
    fm_hidden: int = 256,
    fm_hidden_small: int = 64,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
    # --- probes ---
    n_traj: int = 256,
    n_legs: int = 2,
    n_windows: int = 384,
    ou_rho: float = 0.7,
    plan_chunk: int = 32,
    # --- planner (sized to the action-sequence dim, arm_substrate P3) ---
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
    # --- readout thresholds (the pre-registered rule; see module docstring) ---
    beta_min_dirs: int = 4,
    frontier_floor: float = 0.02,
    frontier_ceiling: float = 0.90,
    stale_ratio: float = 1.5,
):
    import os

    ks = [int(x) for x in k_list.split(",") if x.strip()]
    ns = [int(x) for x in n_links_list.split(",") if x.strip()]
    if quick:
        ks = [1, 4, 14]
        ns = [3, 5]      # both, so the smoke exercises the D=6 and D=10 paths
        pool_n = 3000; fm_steps = 1200
        n_traj = 48; n_windows = 96; plan_chunk = 16
        k_shoot = 256; cem_iters = 4
        fm_hidden = 128; fm_hidden_small = 32; fm_layers = 2
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, k_list=ks, n_links_list=ns,
        link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        b0=b0, b1=b1,
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
        q_range=q_range, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_hidden_small=fm_hidden_small, fm_layers=fm_layers,
        fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps, pool_n=pool_n,
        n_traj=n_traj, n_legs=n_legs, n_windows=n_windows, ou_rho=ou_rho,
        plan_chunk=plan_chunk,
        plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, q_jit=q_jit, v0_std=v0_std,
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi, reach_tries=reach_tries,
        beta_min_dirs=beta_min_dirs, frontier_floor=frontier_floor,
        frontier_ceiling=frontier_ceiling, stale_ratio=stale_ratio,
    )
    out = run_saturation_gate.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "expansion_gate_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
