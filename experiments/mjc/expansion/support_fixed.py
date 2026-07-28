"""Cut #5, Piece 2 — THE SUPPORT-FIXED NULL. Does a drift that only moves the target function
leave the model's frontier flat?

Program: `beliefs/dimensionality_expansion.md` §Scope (2026-07-27) · `mjc/README.md` §Next steps #3.
Instrument: `rhm/residual_decomposition/` — `R_res_participation` (primary), frontier mass
(secondary), reported alongside relative residual. **beta is NOT used**: Piece 1 measured it
failing its own capacity-invariance control by 6-59x on this substrate (|delta| 0.06-0.27 at n=5
against rhm's +-0.01 over the same 4x FM range, fit R^2 0.26-0.51 against 0.95-0.99 in the rhm
domains, over 10 directions). It is still computed and stored for the record, and never read.

WHAT THE BELIEF PREDICTS, AND WHY A PREDICTED NULL IS WORTH RUNNING. §Scope draws the line this
cut exists to make operational:

    drift moves the TARGET FUNCTION;  expansion grows the SPACE OF REPRESENTABLE FUNCTIONS.

A one-parameter walk does the first and structurally cannot do the second -- it demands RE-FITTING
existing directions, never opening new ones. So a fixed-DOF plant under support-fixed drift should
hold the frontier FLAT, and on the belief's own terms that is HEALTH, not a failed compressor.
The two converge only under support-GROWING drift, which is Piece 3.

A predicted null is normally ambiguous -- "the plant genuinely does not expand" and "the instrument
is blind" look identical. **Piece 1 removed that ambiguity**: on this exact substrate, at this exact
horizon, the instrument separates a known-wrong FM from a matched one by 2.7x at k=14 (3/3 seeds,
monotone in k). So a flat frontier here is attributable to the plant, not to a dead readout -- and
that separation doubles as an EFFECT-SIZE YARDSTICK, which is why the `frozen` variant below is
carried through every round.

THREE CONDITIONS, GIVING BOTH CONTRASTS FROM ONE RUN. All three start from the byte-identical
pretrained FM and the byte-identical probe, and see the same number of new transitions per round.
  * `static`  -- global curl held at the OU mean, never drifting. The matched control for
                 CONTINUED TRAINING ALONE. It is not optional: the RHM 2x2 found `R_act` rises in
                 EVERY arm including open-loop, i.e. ordinary continued training moves these
                 numbers on its own, and that is exactly the artifact that would otherwise be
                 read as expansion.
  * `global`  -- ONE curl gain under an OU walk. The purest support-fixed drift, and the literal
                 form the substrate memo proposed ("friction slowly changing").
  * `regions` -- FOUR spatially-gated curl regions at FIXED centers, each gain under its own
                 independent OU walk. Still support-fixed (the set of places where the operator
                 differs never changes) but the target function moves in four parameters, not one.
  `static` vs `global` isolates THE DRIFT with geometry fixed; `global` vs `regions` isolates HOW
  MANY PARAMETERS DRIFT with the drift on. If `regions` opens the frontier where `global` does
  not, the drift/expansion boundary is blurrier than §Scope claims, and that is the finding.

THREE VARIANTS GRADED EVERY ROUND, two of which are calibration:
  * `live`   -- fine-tuned on each round's fresh transitions. Tracks the drift.
  * `frozen` -- the round-0 snapshot (pretrained at the OU mean), never updated. Its error should
                grow with HOW FAR THE WORLD HAS WANDERED FROM THE MEAN, which is a per-round,
                within-run tracking control rather than a two-number comparison.
  * `rail`   -- an FM pretrained in the FIELD-FREE world (b=0) and never updated: Piece 1's exact
                stale model, carried through as the strong yardstick.

  WHY `rail` EXISTS. A calibration run showed `frozen` and `live` reaching *identical* mean
  relative residual (0.371 vs 0.371) even as the OU walked: inside the [0,6] gain box, a model
  pretrained at the box's MIDDLE is never off by more than 3.0, and at n=5/k=14 the FM's own
  14-step compounding error swamps a mismatch that small. `frozen` still tracked the drift
  beautifully (corr(|b-mu|, rel_residual) = +0.958, frontier mass +0.943 — the instrument is
  awake), but a tracking correlation is a weaker answer than a visible gap to "was the drift big
  enough to matter?". `rail` supplies the gap at a 0-to-6 mismatch, i.e. Piece 1's protocol
  exactly, WITHOUT perturbing `live`'s trajectory the way re-pointing the pretraining world would.
  If neither `rail` separates nor `frozen` tracks, the run is uninterpretable and says so.

THE FIXED PROBE, AND WHY IT IS BUILT THE WAY IT IS. The probe is a frozen set of
(initial state, k-step command sequence) pairs, drawn once from matched-FM ballistic reaches in
the reference world. Every round, in every condition, THE SAME pairs are re-executed in that
round's plant to obtain the true k-step displacement A, and rolled through the FM to obtain P.
So the "experiment" is literally identical across rounds and conditions and only the world and
the model differ -- the RHM 2x2's fixed-held-out-probe fix, which is what makes it legitimate to
attribute a change in `R_res_participation` to the model's computation. `R_act` is reported every
round as the check on this: under a support-fixed drift the plant's displacement covariance should
barely move, and if it does, any frontier movement is confounded and must be read against it.

COLLECTION IS TELEPORT, DELIBERATELY, AGAINST THE NODE'S DEFAULT-FOR-NEW-CUTS CONVENTION. The
README's collection-convention box carves this case out itself: *"On-policy data is compound by
construction (model quality <-> data quality are coupled). Tier-A dissociations that need a clean
FM-quality axis should stay on teleport -- that is the trade the flag exists to make explicit."*
Pieces 2 and 3 are exactly that: the dependent variable is the FM's representational frontier as a
function of the INTERVENTION, so coupling the data distribution to the model would make every
frontier change compound. Where you collect is not the question here; it is the question in E3.

Run:
    cd experiments/
    modal run mjc/expansion/support_fixed.py::support_fixed --quick              # smoke
    bash mjc/expansion/train_piece2.sh                                           # 3 seeds
    python3 mjc/expansion/support_fixed_agg.py --tags sf_s0 sf_s1 sf_s2
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.expansion.saturation_gate import shadow_law_flex


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_support_fixed(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.arm_env import ArmEnv, collect_pool, fk
    from rhm.residual_decomposition.decomposition import full_decomposition

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs, H, n = cfg["frame_skip"], cfg["plan_H"], cfg["n_links"]
    SD, AD = 2 * n, n
    ks = cfg["k_list"]; k_max = max(ks)
    Ls = np.asarray(cfg["link_lengths"][:n], dtype=np.float64)
    Ms = list(cfg["link_masses"][:n])
    qc = np.asarray(cfg["q_center"][:n], dtype=np.float64)
    Lt = torch.tensor(Ls, device=device, dtype=torch.float32)
    R = cfg["n_regions"]
    print(f"[setup] device={device} n_links={n} rounds={cfg['n_rounds']} k_list={ks} "
          f"conditions={cfg['conditions']} OU(theta={cfg['ou_theta']}, sigma={cfg['ou_sigma']}, "
          f"mu={cfg['ou_mu']}, clip=[{cfg['ou_lo']},{cfg['ou_hi']}])", flush=True)

    # ------------------------------------------------------------------ the plant
    def make_env(cond, gains, centers):
        """Same chain in every condition; only the curl geometry differs."""
        d = dict(n_links=n, link_lengths=list(Ls), link_masses=Ms,
                 joint_damping=cfg["joint_damping"], gear=cfg["gear"])
        if cond.startswith("regions"):
            d["curl_fields"] = [{"b": float(gains[j]), "center": list(centers[j]),
                                 "sigma": cfg["region_sigma"]} for j in range(len(gains))]
        else:                                   # static and global share the geometry
            d["curl_field"] = {"b": float(gains[0])}
        return ArmEnv(d)

    def fk_torch(q):
        ang = torch.cumsum(q, dim=1)
        return torch.stack([(Lt * torch.cos(ang)).sum(1), (Lt * torch.sin(ang)).sum(1)], 1)

    # ------------------------------------------------------------------ FM f(s,u)->Δs
    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h = cfg["fm_hidden"]
        lyr = [nn.Linear(SD + AD, h), nn.SiLU()]
        for _ in range(cfg["fm_layers"] - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, SD)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def pool(cenv, nn_, seed):
        return collect_pool(cenv, nn_, np.random.default_rng(seed), fs,
                            qc, cfg["q_range"], cfg["v_explore"])

    # Reference world = every condition's round-0 world: gains at the OU mean.
    g0 = np.full(R, cfg["ou_mu"], dtype=np.float64)
    env_ref = make_env("global", g0, None)

    # Normalization fixed ONCE from the reference world and shared by every FM in every
    # condition and round, so nothing is ever scored in its own moving units.
    nS, nU, nS2 = pool(env_ref, cfg["pool_n"], cfg["seed"] + 11)
    Xn_ = np.concatenate([nS, nU], 1).astype(np.float32)
    Yn_ = (nS2 - nS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=Xn_.mean(0), sx=Xn_.std(0) + 1e-6, my=Yn_.mean(0), sy=Yn_.std(0) + 1e-6).items()}

    huber = nn.HuberLoss(delta=1.0)

    def train_steps(net, opt, S, U, S2, steps, brng):
        Xt = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=device)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xs = (Xt - norm["mx"]) / norm["sx"]; Ys = (Yt - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xs[idx]), Ys[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def fm_delta_t(net, s_t, u_t):
        x = torch.cat([s_t, u_t], 1)
        return net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]

    def fm_rollout(net, S0, Useq):
        with torch.no_grad():
            s0 = torch.tensor(S0, device=device, dtype=torch.float32)
            us = torch.tensor(Useq, device=device, dtype=torch.float32)
            s = s0.clone()
            for j in range(us.shape[1]):
                s = s + fm_delta_t(net, s, us[:, j, :])
            return (s - s0).cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------ CEM (probe only)
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
        t0 = fk(q0, Ls)
        d = rng.normal(0, 1, (cfg["reach_tries"], len(q0), n))
        d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        cand = q0[None] + cfg["reach_amp"] * d
        tips = fk(cand, Ls)
        dist = np.linalg.norm(tips - t0[None], axis=-1)
        pen = (np.maximum(0.0, cfg["reach_lo"] - dist)
               + np.maximum(0.0, dist - cfg["reach_hi"]))
        best = np.argmin(pen, axis=0); ar = np.arange(len(q0))
        return tips[best, ar].astype(np.float32), float(np.mean(dist[best, ar]))

    # ============================================================== 0) pretrain + probe
    pS, pU, pS2 = pool(env_ref, cfg["pool_n"], cfg["seed"] + 20)
    fm0 = _mlp(cfg["seed"] + 41)
    train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                pS, pU, pS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    print(f"[pretrain] FM trained in the reference world (all gains = {cfg['ou_mu']})", flush=True)

    # THE STRONG YARDSTICK: Piece 1's stale model — pretrained field-free (b=0), never updated,
    # graded in every drifted world alongside `live` and `frozen`. Additive instrumentation: it
    # never touches `live`, so the null it calibrates stays clean.
    env_rail = make_env("global", np.zeros(R), None)
    rS, rU, rS2 = pool(env_rail, cfg["pool_n"], cfg["seed"] + 22)
    fm_rail = _mlp(cfg["seed"] + 44)
    train_steps(fm_rail, torch.optim.Adam(fm_rail.parameters(), lr=cfg["fm_lr"]),
                rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 304))
    print("[pretrain] rail FM trained field-free (b=0) — the Piece 1 stale yardstick", flush=True)

    # THE FIXED PROBE: (initial state, k_max-step command sequence) pairs, drawn ONCE from
    # matched-FM ballistic reaches in the reference world and frozen for the whole run.
    #
    # WINDOWS ARE SAMPLED MID-MOTION AT RANDOM OFFSETS, NOT FROM THE REACH START. Two reasons,
    # and the first is load-bearing rather than cosmetic:
    #   1. THE CURL FIELD IS VELOCITY-DEPENDENT — F = b·[[0,-1],[1,0]]·v_tip — so it is exactly
    #      ZERO at rest. A probe whose every window begins at a standstill (v0_std=0) starts
    #      each rollout in the one state where the drifted parameter has no effect at all,
    #      which is precisely the wrong place to measure a drift from. Mid-motion starts carry
    #      the velocity the field acts on.
    #   2. It reproduces the construction Piece 1 validated (chained reaches, one random window
    #      per episode), so this piece measures in the regime Piece 1 certified in-window
    #      (n=5, k=14, matched: rel_residual 0.314, frontier mass 0.138) rather than in an
    #      untested one.
    def build_probe(net):
        rng = np.random.default_rng(cfg["seed"] + 700)
        Nt = cfg["n_traj"]
        q0 = qc[None, :] + rng.uniform(-cfg["q_jit"], cfg["q_jit"], (Nt, n))
        s = np.concatenate([q0, rng.normal(0, cfg["v0_std"], (Nt, n))], 1).astype(np.float32)
        S_hist, U_hist = [s.copy()], []
        for leg in range(cfg["n_legs"]):
            goals, mean_reach = sample_goals(s[:, :n].astype(np.float64), rng)
            plan = np.empty((Nt, H, AD), np.float32)
            for i in range(0, Nt, cfg["plan_chunk"]):
                j = min(i + cfg["plan_chunk"], Nt)
                plan[i:j] = mpc_plan(net, s[i:j], goals[i:j], rng)
            for h in range(H):
                u = plan[:, h, :]
                for b in range(Nt):
                    env_ref.set_state(s[b, :n].astype(np.float64), s[b, n:].astype(np.float64))
                    s[b], _ = env_ref.step(u[b], fs)
                S_hist.append(s.copy()); U_hist.append(u.copy())
            print(f"[probe] leg {leg}: mean planned reach {mean_reach:.3f} m", flush=True)
        tS, tU = np.stack(S_hist, 1), np.stack(U_hist, 1)
        T = tU.shape[1]
        assert T > k_max, f"probe trajectory length {T} must exceed k_max {k_max}"
        ep = rng.integers(0, Nt, cfg["n_probe"])
        off = rng.integers(0, T - k_max + 1, cfg["n_probe"])
        pS0 = tS[ep, off]
        pU = np.stack([tU[ep, off + j] for j in range(k_max)], 1)
        spd = np.linalg.norm(pS0[:, n:], axis=1)
        print(f"[probe] {cfg['n_probe']} frozen (s0, {k_max}-step command) pairs from {Nt} "
              f"chained reaches (T={T}); window-start joint speed mean={spd.mean():.2f} "
              f"min={spd.min():.2f} rad/s (0 would mean the curl field is off at every start)",
              flush=True)
        return pS0, pU

    probe_S0, probe_U = build_probe(fm0)

    def execute_probe(cenv):
        """Re-run the frozen probe in `cenv`. Returns the true state trajectory (N, k_max+1, SD)
        plus the out-of-training-box diagnostics Piece 1's wrapped-task-probe caveat asks for."""
        N = probe_S0.shape[0]
        traj = np.empty((N, k_max + 1, SD), np.float32)
        traj[:, 0] = probe_S0
        s = probe_S0.copy()
        for h in range(k_max):
            for b in range(N):
                cenv.set_state(s[b, :n].astype(np.float64), s[b, n:].astype(np.float64))
                s[b], _ = cenv.step(probe_U[b, h], fs)
            traj[:, h + 1] = s
        q = traj[..., :n]; qd = traj[..., n:]
        # The FM's teleport training box is q_center +- q_range with qd ~ N(0, v_explore).
        # Piece 1's task probe left it (max|q| 2.9-4.8 rad vs ArmEnv's 3.0 legibility limit),
        # which is the leading suspect for its n=3 non-monotonicity, so it is measured here.
        outside_q = float(np.mean(np.any(np.abs(q - qc) > cfg["q_range"], axis=-1)))
        fast = float(np.mean(np.any(np.abs(qd) > 2.0 * cfg["v_explore"], axis=-1)))
        return traj, {"frac_outside_q_box": outside_q, "frac_fast": fast,
                      "max_absq": float(np.abs(q).max()),
                      "nonfinite": int(cenv.nonfinite())}

    # Region centers: FIXED for the whole run, placed by farthest-point sampling on the tip
    # positions the probe actually visits, so every region is genuinely ON the probe's path
    # (E3 gotcha #1: geometry is not a substitute for measured visitation).
    ref_traj, ref_diag = execute_probe(env_ref)
    tips = fk(ref_traj[..., :n].reshape(-1, n).astype(np.float64), Ls)
    sel = [int(np.random.default_rng(cfg["seed"] + 5).integers(len(tips)))]
    dmin = np.linalg.norm(tips - tips[sel[0]], axis=1)
    for _ in range(R - 1):
        j = int(np.argmax(dmin)); sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(tips - tips[j], axis=1))
    centers = tips[sel]
    pair = [float(np.linalg.norm(centers[a] - centers[b]))
            for a in range(R) for b in range(a + 1, R)]
    print(f"[regions] {R} centers, pairwise distance min={min(pair):.3f} max={max(pair):.3f} m "
          f"(gate sigma={cfg['region_sigma']}); probe diag={ref_diag}", flush=True)

    # ============================================================== 1) the rounds
    def measure(net, traj, tag_bits):
        out = []
        for k in ks:
            A = (traj[:, k] - traj[:, 0]).astype(np.float64)
            P = fm_rollout(net, probe_S0, probe_U[:, :k, :])
            Rr = A - P
            resid_mean_frac = float((Rr.mean(axis=0) ** 2).sum()
                                    / max((Rr ** 2).sum(axis=1).mean(), 1e-30))
            sd = A.std(axis=0) + 1e-12
            dec = full_decomposition(A / sd, P / sd, ks=tuple(range(1, SD)), seed=cfg["seed"])
            beta, br2, ndir = shadow_law_flex(dec["geometry"]["act_variance_spectrum"],
                                              dec["geometry"]["res_variance_in_act_basis"],
                                              min_dirs=cfg["beta_min_dirs"])
            out.append({**tag_bits, "k": k,
                        "rel_residual": dec["basic"]["relative_residual"],
                        "frontier_mass": dec["repaired"]["frontier_mass"],
                        "R_res_participation": dec["repaired"]["R_res_participation"],
                        # R_act is THE confound check: the probe is frozen, so under a
                        # support-fixed drift the plant's displacement covariance should
                        # barely move. Any frontier movement must be read against it.
                        "R_act_pr": dec["repaired"]["R_act_pr"],
                        "naive_R_res": dec["naive"]["R_res"],
                        "alignment_index": dec["geometry"]["alignment_index"],
                        "resid_mean_frac": resid_mean_frac,
                        "mean_residual_norm": float(np.linalg.norm(Rr, axis=1).mean()),
                        "mean_target_norm": float(np.linalg.norm(A, axis=1).mean()),
                        # stored for the record, never read: Piece 1 measured beta failing
                        # its own capacity-invariance control on this substrate.
                        "beta": beta, "beta_r2": br2, "beta_n_dirs": ndir})
        return out

    # Condition semantics, from the name: anything ending "static" does not drift, anything
    # starting "regions" uses the multi-region geometry. That gives the full 2x2 —
    #   static / global        : one-parameter geometry, no drift / drift
    #   regions_static / regions: four-region geometry,  no drift / drift
    # which is what separates HOW MANY PARAMETERS DRIFT from HOW HARD THE OPERATOR IS. The
    # first pass lacked `regions_static` and so confounded them: `regions` turned out to be a
    # harder DGP outright (live relative residual 0.38-0.51 vs 0.27-0.34 elsewhere), not merely
    # a differently-drifting one.
    CANON = ["static", "global", "regions", "regions_static"]

    def geom_of(cond):
        return "regions" if cond.startswith("regions") else "global"

    def drifts(cond):
        return not cond.endswith("static")

    rows, gain_log = [], {}
    for cond in cfg["conditions"]:
        gains = g0.copy()
        # Seeded by the condition's CANONICAL index, never by hash(cond) and never by its
        # position in the requested list: Python salts string hashes per process, and a
        # list-position seed would silently change `regions`' drift trajectory as soon as
        # another condition was added ahead of it, breaking comparability across runs.
        rng_ou = np.random.default_rng(
            cfg["seed"] + 3000 + 101 * (CANON.index(cond) if cond in CANON else len(CANON)))
        rng_buf = np.random.default_rng(cfg["seed"] + 500)
        fm_live = copy.deepcopy(fm0)
        opt = torch.optim.Adam(fm_live.parameters(), lr=cfg["fm_lr"])
        gain_log[cond] = []
        bufS = np.zeros((0, SD), np.float32); bufU = np.zeros((0, AD), np.float32)
        bufS2 = np.zeros((0, SD), np.float32)
        for rd in range(cfg["n_rounds"]):
            if drifts(cond) and rd > 0:
                gains = np.clip(gains + cfg["ou_theta"] * (cfg["ou_mu"] - gains)
                                + cfg["ou_sigma"] * rng_ou.normal(0, 1, R),
                                cfg["ou_lo"], cfg["ou_hi"])
            if geom_of(cond) == "global":
                gains[:] = gains[0]        # one parameter, not R of them
            env_r = make_env(cond, gains, centers)
            gain_log[cond].append(
                [float(x) for x in (gains if geom_of(cond) == "regions" else gains[:1])])

            aS, aU, aS2 = pool(env_r, cfg["round_n"], cfg["seed"] + 4000 + 17 * rd)
            if cfg["buffer_mode"] == "accumulate":
                # A SLIDING/GROWING buffer rather than this round's 2000 transitions alone.
                # `fresh` (the default, and what the first pass ran) fine-tunes 1000 Adam steps
                # on 2000 samples = 256 epochs every round, which churns the model: Piece 3,
                # which accumulates, produced less than half the seed spread on the same
                # readout (+-0.42 vs +-0.55 at k=8). Under drift, old data IS from a different
                # world, so this trades tracking sharpness for stability — which is the right
                # trade when the dependent variable is the frontier rather than the tracking.
                bufS = np.concatenate([bufS, aS])[-cfg["buffer_cap"]:]
                bufU = np.concatenate([bufU, aU])[-cfg["buffer_cap"]:]
                bufS2 = np.concatenate([bufS2, aS2])[-cfg["buffer_cap"]:]
            else:
                bufS, bufU, bufS2 = aS, aU, aS2
            train_steps(fm_live, opt, bufS, bufU, bufS2, cfg["finetune_steps"],
                        np.random.default_rng(cfg["seed"] + 600 + rd))

            traj, diag = execute_probe(env_r)
            g_eff = gains if geom_of(cond) == "regions" else gains[:1]
            bits = {"condition": cond, "round": rd,
                    "transitions": int((rd + 1) * cfg["round_n"]),
                    "train_set_size": int(len(bufS)),
                    "gain_mean": float(np.mean(g_eff)),
                    "gain_drift_from_start": float(np.abs(g_eff - cfg["ou_mu"]).mean()),
                    **{f"probe_{k}": v for k, v in diag.items()}}
            rows += measure(fm_live, traj, {**bits, "variant": "live"})
            rows += measure(fm0, traj, {**bits, "variant": "frozen"})
            rows += measure(fm_rail, traj, {**bits, "variant": "rail"})

            pk = [r for r in rows if r["round"] == rd and r["condition"] == cond
                  and r["k"] == cfg["k_primary"]]
            lv = next(r for r in pk if r["variant"] == "live")
            fz = next(r for r in pk if r["variant"] == "frozen")
            rl = next(r for r in pk if r["variant"] == "rail")
            print(f"[{cond:8s} rd={rd}] gain~{bits['gain_mean']:.2f} "
                  f"(|drift|={bits['gain_drift_from_start']:.2f})  k={cfg['k_primary']}  "
                  f"live: R_res_part={lv['R_res_participation']:.2f}/{SD} "
                  f"frontier={lv['frontier_mass']:.4f} rel_res={lv['rel_residual']:.4f} "
                  f"R_act={lv['R_act_pr']:.2f} | frozen: {fz['R_res_participation']:.2f}/"
                  f"{fz['frontier_mass']:.4f} | rail: {rl['R_res_participation']:.2f}/"
                  f"{rl['frontier_mass']:.4f} (x{rl['frontier_mass']/max(lv['frontier_mass'],1e-12):.2f})",
                  flush=True)

    # ------------------------------------------------------------------ the readout
    kp = cfg["k_primary"]

    def series(cond, variant, key):
        return [r[key] for r in sorted(
            (r for r in rows if r["condition"] == cond and r["variant"] == variant
             and r["k"] == kp), key=lambda r: r["round"])]

    summary = {}
    print("\n" + "=" * 104, flush=True)
    print(f"PIECE 2 — support-fixed drift, k={kp}. Prediction: the frontier stays FLAT under "
          f"drift (that is health).", flush=True)
    print("=" * 104, flush=True)
    for cond in cfg["conditions"]:
        s = {}
        for variant in ("live", "frozen", "rail"):
            pr = series(cond, variant, "R_res_participation")
            fmv = series(cond, variant, "frontier_mass")
            rr = series(cond, variant, "rel_residual")
            ra = series(cond, variant, "R_act_pr")
            half = max(1, len(pr) // 2)
            s[variant] = {
                "R_res_participation": pr, "frontier_mass": fmv,
                "rel_residual": rr, "R_act_pr": ra,
                # late-vs-early rather than last-vs-first: one round is noise, a half is a trend.
                "R_res_part_early": float(np.mean(pr[:half])),
                "R_res_part_late": float(np.mean(pr[-half:])),
                "R_res_part_delta": float(np.mean(pr[-half:]) - np.mean(pr[:half])),
                "frontier_early": float(np.mean(fmv[:half])),
                "frontier_late": float(np.mean(fmv[-half:])),
                "R_act_delta": float(np.mean(ra[-half:]) - np.mean(ra[:half])),
            }
        s["frozen_over_live_frontier_late"] = float(
            s["frozen"]["frontier_late"] / max(s["live"]["frontier_late"], 1e-12))
        s["rail_over_live_frontier_late"] = float(
            s["rail"]["frontier_late"] / max(s["live"]["frontier_late"], 1e-12))
        summary[cond] = s
        print(f"  {cond:8s} | live  R_res_part {s['live']['R_res_part_early']:.2f} -> "
              f"{s['live']['R_res_part_late']:.2f} (D={s['live']['R_res_part_delta']:+.2f})  "
              f"frontier {s['live']['frontier_early']:.4f} -> {s['live']['frontier_late']:.4f}  "
              f"R_act D={s['live']['R_act_delta']:+.2f}", flush=True)
        print(f"  {' '*8} | frozen R_res_part {s['frozen']['R_res_part_early']:.2f} -> "
              f"{s['frozen']['R_res_part_late']:.2f} (D={s['frozen']['R_res_part_delta']:+.2f})  "
              f"frontier {s['frozen']['frontier_early']:.4f} -> "
              f"{s['frozen']['frontier_late']:.4f}   [frozen/live frontier late = "
              f"{s['frozen_over_live_frontier_late']:.2f}x]", flush=True)
        print(f"  {' '*8} | rail   R_res_part {s['rail']['R_res_part_early']:.2f} -> "
              f"{s['rail']['R_res_part_late']:.2f}  frontier {s['rail']['frontier_early']:.4f} "
              f"-> {s['rail']['frontier_late']:.4f}   [rail/live frontier late = "
              f"{s['rail_over_live_frontier_late']:.2f}x  <- Piece 1 ref: 2.7x]", flush=True)
    print("\n  Read: `static` is the continued-training-alone control (the RHM 2x2 found these "
          "numbers\n  move on their own); `frozen` going stale under drift while `live` does not "
          "is the\n  embedded positive control — if it does not separate, the run is "
          "uninterpretable.", flush=True)
    print("=" * 104 + "\n", flush=True)

    out = {"config": cfg, "rows": rows, "summary": summary, "gain_log": gain_log,
           "region_centers": centers.tolist(), "ref_probe_diag": ref_diag}
    outdir = os.path.join(DATA_DIR, "expansion_support_fixed", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        met = [("R_res_participation", "R_res_participation (frontier dimensionality)"),
               ("frontier_mass", "frontier mass"),
               ("rel_residual", "relative residual"),
               ("R_act_pr", "R_act (confound check — probe is frozen)")]
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
        col = {"static": "#7f7f7f", "global": "#1f77b4", "regions": "#2ca02c",
               "regions_static": "#9467bd"}
        for j, (key, lab) in enumerate(met):
            ax = axes[j]
            for cond in cfg["conditions"]:
                for variant, ls in (("live", "-"), ("frozen", "--")):
                    y = series(cond, variant, key)
                    ax.plot(range(len(y)), y, ls, marker="o", ms=3, color=col.get(cond),
                            alpha=1.0 if variant == "live" else 0.45,
                            label=f"{cond}/{variant}" if j == 0 else None)
            ax.set_xlabel("round"); ax.set_ylabel(lab); ax.set_title(lab, fontsize=9)
            if key == "R_res_participation":
                ax.axhline(SD, color="k", lw=0.8, ls="-.", alpha=0.4)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Cut #5 Piece 2 — support-fixed drift, n={n}, k={kp}. "
                     f"Prediction: live frontier FLAT under drift; frozen opens.", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(outdir, "fig_support_fixed.png"), dpi=150)
        plt.close(fig)
        print("[figure] wrote fig_support_fixed.png", flush=True)
    except Exception as e:
        print(f"[figure] skipped: {e}", flush=True)

    volume.commit()
    print(f"[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def support_fixed(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    conditions: str = "static,global,regions,regions_static",
    n_rounds: int = 8,
    round_n: int = 2000,
    # `fresh` reproduces the first pass bit-identically (tags sf_s0/1/2) and stays the default
    # so that run remains re-runnable; `accumulate` is the corrected protocol.
    buffer_mode: str = "fresh",
    buffer_cap: int = 16000,
    # --- the readout: n=5, R_res_participation primary, no beta.
    # k=8, NOT k=14. Piece 3's calibration showed the DIMENSIONALITY signal is weakest at
    # k=14 (+0.27 +- 0.74 for an intervention as blunt as adding three degrees of freedom)
    # and resolves at k=8 (+0.72 +- 0.42). The first pass read this null at k=14, i.e. at the
    # horizon where it had the least power. Re-aggregating the SAME stored rows at k=8 already
    # tightened it from +0.68 +- 1.00 to +0.29 +- 0.55 (global - static). ---
    k_list: str = "1,4,6,8,10,14",
    k_primary: int = 8,
    # --- the drift: OU walk on curl gain(s). Support-FIXED by construction. ---
    # Sized empirically, not guessed. The gain is clipped to [0,6] (arm_readapt's established
    # well-behaved drift magnitude), so the box — not sigma — bounds the excursion: mean
    # |b-mu| saturates at ~1.6 / 1.9 / 2.0 / 2.1 for (theta,sigma) = (.15,1.5) / (.10,2.0) /
    # (.05,2.5) / (.02,3.0). Weak mean reversion is chosen to spend more time at the rails
    # (18% at each), because the FROZEN FM is pretrained at mu and only goes stale to the
    # extent the world leaves mu. A first calibration at theta=.15/sigma=1.5 over 3 rounds
    # reached |drift| 0.55 and separated frozen from live by only 1.08x — too small to license
    # reading a null.
    ou_mu: float = 3.0,
    ou_sigma: float = 2.5,
    ou_theta: float = 0.05,
    ou_lo: float = 0.0,
    ou_hi: float = 6.0,
    n_regions: int = 4,
    region_sigma: float = 0.20,
    # --- arm: arm_substrate's recommended n=5 design point ---
    n_links: int = 5,
    link_lengths: str = "0.4,0.4,0.3,0.25,0.2",
    link_masses: str = "1.0,1.0,0.6,0.4,0.3",
    q_center: str = "0.4,0.8,0.6,0.4,0.3",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    q_range: float = 0.9,
    v_explore: float = 8.0,
    # --- FM ---
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    finetune_steps: int = 1000,
    pool_n: int = 14000,
    # --- probe ---
    n_probe: int = 384,
    n_traj: int = 192,
    n_legs: int = 2,
    plan_chunk: int = 32,
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
    beta_min_dirs: int = 4,
):
    import os

    ks = [int(x) for x in k_list.split(",") if x.strip()]
    conds = [c for c in conditions.split(",") if c]
    if quick:
        n_rounds = 3; round_n = 800; pool_n = 3000
        fm_steps = 1200; finetune_steps = 300
        n_probe = 96; n_traj = 48; plan_chunk = 16; k_shoot = 256; cem_iters = 4
        fm_hidden = 128; fm_layers = 2
        ks = [1, k_primary]
        tag = tag or "smoke"
    tag = tag or "default"
    # A k_primary outside k_list makes every summary lookup raise StopIteration from inside a
    # Modal coroutine, which surfaces as an opaque "coroutine raised StopIteration" with no
    # hint at the cause. Fail here instead, before any compute is spent.
    assert k_primary in ks, f"k_primary={k_primary} must be in k_list={ks}"

    cfg = dict(
        tag=tag, seed=seed, conditions=conds, n_rounds=n_rounds, round_n=round_n,
        buffer_mode=buffer_mode, buffer_cap=buffer_cap,
        k_list=ks, k_primary=k_primary,
        ou_mu=ou_mu, ou_sigma=ou_sigma, ou_theta=ou_theta, ou_lo=ou_lo, ou_hi=ou_hi,
        n_regions=n_regions, region_sigma=region_sigma,
        n_links=n_links,
        link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
        q_range=q_range, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, finetune_steps=finetune_steps, pool_n=pool_n,
        n_probe=n_probe, n_traj=n_traj, n_legs=n_legs,
        plan_chunk=plan_chunk, plan_H=plan_h, k_shoot=k_shoot,
        cem_iters=cem_iters, cem_elite=cem_elite, cem_init_sigma=cem_init_sigma,
        vel_pen=vel_pen, q_jit=q_jit, v0_std=v0_std, reach_amp=reach_amp,
        reach_lo=reach_lo, reach_hi=reach_hi, reach_tries=reach_tries,
        beta_min_dirs=beta_min_dirs,
    )
    out = run_support_fixed.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "support_fixed_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
