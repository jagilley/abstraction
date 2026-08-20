"""Round 0 -- the admissibility gates for state-conditioned commitment on the arm.

Methodology: `rhm/practice/typed_gaps/` -- offline calibration -> admissibility gate -> main run,
and **a gate that fails is a finding, not a bug**. Nothing in this file is the experiment; it is
the measurement that says whether the experiment is askable on this world, and it fixes every knob
the main run will use by measurement rather than by guess (the etude's calibration record is the
template). Each WORLD in `--worlds` is one Modal container, exactly the etude's arms-as-containers
idiom, so the difficulty knob is **swept, not solved** (`typed_gaps`: the sigma-per-event objective
was non-monotone, and solving for it would have picked the wrong point).

FOUR GATES.

  G1  BOUNDARY INFORMATION. Does the hand-over into the drilled segment carry information a
      committed unit could use? Three parts:
        (a) the hand-over spread under at-tempo upstream execution is >> the metering noise floor;
        (b) a single fixed committed unit's realised error VARIES MATERIALLY with the hand-over --
            read as `best_fixed / per_state_oracle`, i.e. `min_j mean_i E[j,i]` over
            `mean_i min_j E[j,i]` on the SAME audition matrix. That ratio is the ceiling on
            everything `library` could ever buy, and it costs no extra rollouts;
        (c) with motor noise live, post-commit drift must stop being exactly 0.0000 -- the etude's
            structural fact, re-measured.

  G2  MULTIMODALITY. Do valid renditions form >= 2 solution modes whose position-wise MEAN is much
      worse than any contributor? Joint redundancy is tried first; the `push_fields` soft obstacle
      (`arm_env._apply_push_field`, additive and off by default) is the sanctioned escalation, and
      whether it is needed is a measured result, not a design assumption.

  G3  USABLE RANGE. Stale-FM -> corridor-matched-ceiling-FM separation on the drilled segment at
      performance tempo, well above the metering noise floor. (The etude's was 0.0826.)

  G4  COMMITMENT IS A BET. The segment horizon H sits at or beyond the plant's composition horizon,
      so open-loop execution is non-trivially risky pre-practice and viable post-mastery.

Two calibrations that are not gates but would silently ruin the run:
  CAL-P  planner sizing, measured for BOTH forward models. `arm_substrate` P3 is the single biggest
         confound on this plant: under-optimised CEM makes ballistic error planning-noise-dominated,
         which flattens the FM-quality axis toward zero (1.73x vs 4.63x on the same world). If the
         stale/ceiling gap grows with planner size, G3 was masked rather than absent.
  CAL-D  the descent clock: how much of the stale->ceiling range one practice cycle closes. The
         etude found 88% closed by cycle 2 at its first setting and had to cut `n_grad` 40 -> 5.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/fingering/gates.py::gates --quick
    python3 mjc/practice/fingering/launch_detached.py --fn gates --tag g0 --seed 0
    python3 mjc/practice/fingering/analyze_gates.py --tag g0 --fetch
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# THE PIECE, picked analytically (pure-numpy FK; the search is recorded in FILES.md). From the mean
# start tip (0.197, 0.778): a 0.32 m approach down-and-right to W1, then a 0.44 m drilled sweep
# right to W2, a ~70 deg turn between them. Every point on both paths has radius in [0.57, 0.89] of
# a 1.10 m arm, so the piece stays off the singular full-extension shell and off the base, and the
# drilled segment EXTENDS the arm as it goes (M(q) changes along the passage, so the passage is
# dynamically nonuniform rather than a translation). The patch sits on the drilled midpoint with
# sigma 0.08, which puts its gate weight at both waypoints -- and at the approach's closest
# approach -- at 0.023, against the etude's 0.044.
DEF_W1 = "0.3063,0.4778"
DEF_W2 = "0.7463,0.4778"


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_gates(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.arm_env import fk
    from mjc.embodied import make_arm_goal_sampler, pool_diagnostics, cmd_state_corr
    from mjc.practice.fingering.world import World, Ledger, start_postures, null_direction, kmeans

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    n, SD, AD = W.n, W.SD, W.AD
    led = Ledger()
    world = cfg["world"]
    out = {"config": cfg, "world": world, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_fingering", cfg["tag"], world)
    os.makedirs(outdir, exist_ok=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    def P(*a):
        print(f"[{world}]", *a, flush=True)

    P(f"[setup] device={device} n={n} H_app={W.H_app} H_drill={W.H_drill} "
      f"W1={W.W1.tolist()} W2={W.W2.tolist()} patch c={np.round(W.patch_center, 4).tolist()} "
      f"sigma={W.patch_sigma} curl_b={W.curl_b} push_a={W.push_a}")

    # ================================================================= C0 sanity
    rng = np.random.default_rng(cfg["seed"] + 1)
    qs = W.qc[None, :] + rng.uniform(-0.5, 0.5, (8, n))
    fk_err = 0.0
    for q in qs:
        W.env.set_state(q, np.zeros(n))
        fk_err = max(fk_err, float(np.linalg.norm(W.env.tip_pos() - fk(q, W.Ls))))
    p0 = fk(W.qc, W.Ls)
    gates_at = dict(W1=float(W.gate(W.W1[None, :])[0]), W2=float(W.gate(W.W2[None, :])[0]),
                    p0=float(W.gate(p0[None, :])[0]))
    out["c0"] = dict(fk_vs_mujoco_max=fk_err, p0=p0.tolist(),
                     seg_app=float(np.linalg.norm(W.W1 - p0)),
                     seg_drill=float(np.linalg.norm(W.W2 - W.W1)), gate_at=gates_at,
                     radii=dict(p0=float(np.linalg.norm(p0)), W1=float(np.linalg.norm(W.W1)),
                                W2=float(np.linalg.norm(W.W2)),
                                mid=float(np.linalg.norm(W.patch_center))))
    P(f"[C0] fk vs mujoco max {fk_err:.2e} | seg_app={out['c0']['seg_app']:.3f} "
      f"seg_drill={out['c0']['seg_drill']:.3f} | gate@p0/W1/W2 = "
      f"{gates_at['p0']:.4f}/{gates_at['W1']:.4f}/{gates_at['W2']:.4f}")
    assert fk_err < 1e-9, "analytic FK disagrees with MuJoCo -- the CEM cost would score a "\
                          "different arm than the simulator executes"

    # ================================================================= the diet (on-policy)
    # Stage 1 (B1): OU torque episodes -- no model in the loop, because there is no model yet.
    S1, U1, S21, i1 = W.collect_ou(cfg["pool_ou"], np.random.default_rng(cfg["seed"] + 11))
    W.set_norm(S1, U1, S21)
    boot = W.mlp(cfg["seed"] + 40)
    W.train_steps(boot, torch.optim.Adam(boot.parameters(), lr=cfg["fm_lr"]),
                  S1, U1, S21, cfg["fm_steps_boot"], np.random.default_rng(cfg["seed"] + 300))
    # Stage 2 (B3): reaches under the bootstrap model, goals drawn from the postures the body
    # actually resets to (`make_arm_goal_sampler`, so the reach band is honest).
    gs = make_arm_goal_sampler(W.Ls, cfg["reach_amp"], cfg["reach_lo"], cfg["reach_hi"])
    S2_, U2_, S22, i2 = W.collect_reach(cfg["pool_reach"], np.random.default_rng(cfg["seed"] + 12),
                                        boot, gs)
    Sa = np.concatenate([S1, S2_]); Ua = np.concatenate([U1, U2_]); S2a = np.concatenate([S21, S22])
    Se, Ue, S2e, keep_frac = W.exclude_region(Sa, Ua, S2a, w=cfg["exclude_w"])
    W.set_norm(Se, Ue, S2e)
    fm0 = W.mlp(cfg["seed"] + 41)
    W.train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                  Se, Ue, S2e, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    fm_app = copy.deepcopy(fm0)          # the FROZEN approach controller: identical for every arm
    for p in fm_app.parameters():
        p.requires_grad_(False)
    diag = pool_diagnostics(Se, Ue, n)
    P(f"[diet] ou={len(S1)} reach={len(S2_)} | exclude kept {keep_frac:.3f} | "
      f"max|corr(u,s)| ou={cmd_state_corr(S1, U1)['max_abs_corr']:.3f} "
      f"reach={cmd_state_corr(S2_, U2_)['max_abs_corr']:.3f} | "
      f"speed p95={diag.get('speed_p95', float('nan')):.2f}")

    # ================================================================= geometries + the approach
    # Three fixed posture sets, seeded independently and shared by every arm of the main run:
    #   rt -- metering (agent-side)   sc -- audition score set (agent-side, charged)
    #   ev -- held-out grade (instrument, free)
    def geom(m, seed, mode=None):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(seed),
                              cfg["q_jit"], cfg["null_jit"], mode or cfg["start_mode"])

    q_rt = geom(cfg["n_rt"], cfg["seed"] + 5000)
    q_sc = geom(cfg["n_score"], cfg["seed"] + 5100)
    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)
    approach_pf = W.plan_fn(fm_app, W.H_app)

    def app_plan(q, seed):
        """The mastered approach: one full-horizon plan from the actual start posture, computed by
        the FROZEN planner. Deterministic given (q, seed), hence identical across arms -- so the
        hand-over distribution is a property of the WORLD, not of the arm being graded."""
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        g = np.tile(W.W1[None, :], (len(q), 1))
        return approach_pf(s, g, np.random.default_rng(seed))

    plan_rt = app_plan(q_rt, cfg["seed"] + 6000)
    plan_sc = app_plan(q_sc, cfg["seed"] + 6100)
    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    REACT = World.reactive_unit()

    def run(fm, unit, q, plan, seed, sigma, who="instrument", kind="probe", replan=1,
            collect=False, drill=True):
        return W.traverse(fm, unit, q, np.random.default_rng(seed), sigma, led, who=who, kind=kind,
                          drill_replan=replan, approach_plan=plan, collect=collect, drill=drill)

    # ---- the CEILING forward model: the same diet PLUS the piece's own CORRIDOR.
    # An oracle instrument, free and labelled as such. It is deliberately CORRIDOR-MATCHED rather
    # than globally accurate: the etude's cal_s1 measured that a corridor-specialised FM beats a
    # globally-accurate one on the corridor (bridge_assembly's spatial-matching law -- a ballistic
    # controller is a narrow line-integral), so a "ceiling" collected on a broad uniform volume is
    # not the ceiling for this passage and would understate G3's usable range.
    corr = []
    for i in range(cfg["n_corridor"]):
        qq = geom(cfg["batch"], cfg["seed"] + 5400 + i)
        r = run(fm0, REACT, qq, app_plan(qq, cfg["seed"] + 6400 + i), cfg["seed"] + 6500 + i,
                cfg["sigma_practice"], who="instrument", kind="ceiling_diet",
                replan=cfg["corridor_replan"], collect=True)
        corr.append(r["trans"])
    Sk = np.concatenate([Sa] + [c[0] for c in corr])
    Uk = np.concatenate([Ua] + [c[1] for c in corr])
    S2k = np.concatenate([S2a] + [c[2] for c in corr])
    fm_ceil = W.mlp(cfg["seed"] + 42)
    W.train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                  Sk, Uk, S2k, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))

    def fm_err(net, S, U, S2):
        return float(np.linalg.norm(W.fm_delta(net, S, U) - (S2 - S), axis=1).mean())

    Sc_ = np.concatenate([c[0] for c in corr]); Uc_ = np.concatenate([c[1] for c in corr])
    S2c_ = np.concatenate([c[2] for c in corr])
    reg = W.gate(W.tip(Sc_)) > 0.3
    out["diet"] = dict(n_ou=len(S1), n_reach=len(S2_), n_corridor=len(Sc_),
                       exclude_keep_frac=keep_frac, n_excluded_pool=len(Se),
                       corridor_in_region_frac=float(reg.mean()), diagnostics=diag,
                       corr_ou=cmd_state_corr(S1, U1), corr_reach=cmd_state_corr(S2_, U2_),
                       corr_corridor=cmd_state_corr(Sc_, Uc_))
    out["fm_probe"] = dict(
        stale_corridor_region=fm_err(fm0, Sc_[reg], Uc_[reg], S2c_[reg]) if reg.any() else None,
        stale_corridor_clean=fm_err(fm0, Sc_[~reg], Uc_[~reg], S2c_[~reg]),
        ceil_corridor_region=fm_err(fm_ceil, Sc_[reg], Uc_[reg], S2c_[reg]) if reg.any() else None,
        ceil_corridor_clean=fm_err(fm_ceil, Sc_[~reg], Uc_[~reg], S2c_[~reg]),
        n_corridor_in_region=int(reg.sum()))
    fp = out["fm_probe"]
    P(f"[fm] corridor-slice error  stale reg/clean = {fp['stale_corridor_region']}/"
      f"{fp['stale_corridor_clean']:.4f}   ceiling = {fp['ceil_corridor_region']}/"
      f"{fp['ceil_corridor_clean']:.4f}  (n_in_region={int(reg.sum())}/{len(Sc_)})")
    save()

    # ================================================================= G4 composition horizon
    ho = run(fm0, REACT, q_ev, plan_ev, cfg["seed"] + 7000, cfg["sigma_perf"], drill=False)["hand"]
    ball = W.plan_fn(fm_ceil, W.H_drill)(ho, np.tile(W.W2[None, :], (len(ho), 1)),
                                         np.random.default_rng(cfg["seed"] + 7001))

    def comp_horizon(net):
        true_tips = np.empty((len(ho), W.H_drill, 2))
        for i in range(len(ho)):
            W.env.set_state(ho[i, :n].astype(np.float64), ho[i, n:].astype(np.float64))
            for h in range(W.H_drill):
                W.env.step(ball[i, h], W.fs)
                true_tips[i, h] = W.env.tip_pos()
        s = ho.copy()
        errs = np.empty((len(ho), W.H_drill))
        for h in range(W.H_drill):
            s = s + W.fm_delta(net, s, ball[:, h, :])
            errs[:, h] = np.linalg.norm(fk(s[:, :n].astype(np.float64), W.Ls)
                                        - true_tips[:, h], axis=1)
        return np.median(errs, 0)

    ch_stale, ch_ceil = comp_horizon(fm0), comp_horizon(fm_ceil)

    def cross(curve, thr):
        idx = np.nonzero(curve > thr)[0]
        return int(idx[0] + 1) if len(idx) else int(len(curve) + 1)

    out["g4"] = dict(h_drill=W.H_drill, curve_stale=ch_stale.tolist(),
                     curve_ceiling=ch_ceil.tolist(),
                     horizon_stale_005=cross(ch_stale, 0.05), horizon_ceil_005=cross(ch_ceil, 0.05),
                     horizon_stale_002=cross(ch_stale, 0.02), horizon_ceil_002=cross(ch_ceil, 0.02),
                     err_at_H_stale=float(ch_stale[-1]), err_at_H_ceil=float(ch_ceil[-1]))
    P(f"[G4] composition horizon (tip err > 0.05 m): stale {out['g4']['horizon_stale_005']} "
      f"ceiling {out['g4']['horizon_ceil_005']} (>0.02: {out['g4']['horizon_stale_002']}/"
      f"{out['g4']['horizon_ceil_002']}) | H_drill={W.H_drill} | err@H stale={ch_stale[-1]:.4f} "
      f"ceiling={ch_ceil[-1]:.4f}")
    save()

    # ================================================================= CAL-P planner sizing
    # Measured for BOTH models: if the stale/ceiling gap GROWS with planner size, an undersized
    # planner was masking the FM-quality axis rather than the axis being absent (P3, exactly).
    cal = {}
    g2t = np.tile(W.W2[None, :], (len(ho), 1))
    for spec in cfg["planner_grid"]:
        ks, ci = int(spec[0]), int(spec[1])
        row = {}
        for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
            pl = W.plan_fn(net, W.H_drill, k_shoot=ks, cem_iters=ci)(
                ho, g2t, np.random.default_rng(cfg["seed"] + 7100))
            e = np.array([W.replay(ho[i:i + 1], pl[i], W.W2)[0] for i in range(len(ho))])
            row[nm] = dict(median=float(np.median(e)), mean=float(e.mean()),
                           p90=float(np.percentile(e, 90)))
        row["gap"] = row["stale"]["median"] - row["ceiling"]["median"]
        cal[f"k{ks}_i{ci}"] = row
        P(f"[CAL-P] k_shoot={ks:5d} iters={ci} -> drilled ballistic median "
          f"stale {row['stale']['median']:.4f}  ceiling {row['ceiling']['median']:.4f}  "
          f"gap {row['gap']:+.4f}")
    out["cal_planner"] = cal
    save()

    # ================================================================= G3 usable range + noise
    def meter(fm, unit, seed, sigma=None, replan=1, who="instrument"):
        return run(fm, unit, q_rt, plan_rt, seed, cfg["sigma_perf"] if sigma is None else sigma,
                   replan=replan, who=who, kind="metering")

    med = lambda L: np.array([np.median(x) for x in L])
    m_stale = med([meter(fm0, REACT, cfg["seed"] + 7200 + r, replan=W.H_drill)["e_drill"]
                   for r in range(cfg["n_meter_rep"])])
    m_ceil = med([meter(fm_ceil, REACT, cfg["seed"] + 7300 + r, replan=W.H_drill)["e_drill"]
                  for r in range(cfg["n_meter_rep"])])
    m_react = med([meter(fm0, REACT, cfg["seed"] + 7400 + r, replan=1)["e_drill"]
                   for r in range(cfg["n_meter_rep"])])
    noise_floor = float(np.std(np.concatenate([m_stale - m_stale.mean(),
                                               m_ceil - m_ceil.mean()])))
    out["g3"] = dict(stale_tempo=float(m_stale.mean()), ceiling_tempo=float(m_ceil.mean()),
                     reactive_stale=float(m_react.mean()),
                     usable_range=float(m_stale.mean() - m_ceil.mean()),
                     metering_noise_sd=noise_floor, n_rep=cfg["n_meter_rep"],
                     range_over_noise=float((m_stale.mean() - m_ceil.mean())
                                            / max(noise_floor, 1e-9)),
                     ballistic_over_reactive=float(m_stale.mean() / max(m_react.mean(), 1e-9)))
    P(f"[G3] drilled @tempo: stale {m_stale.mean():.4f}  ceiling {m_ceil.mean():.4f}  "
      f"reactive(stale) {m_react.mean():.4f} | usable range {out['g3']['usable_range']:.4f} "
      f"| metering noise sd {noise_floor:.4f} -> {out['g3']['range_over_noise']:.1f}x")
    save()

    # ================================================================= warmup practice (CAL-D)
    # Exactly the main run's practice loop. Drilled segment practised CLOSED-LOOP (etude finding 6:
    # closed-loop traces are the better commitment candidates -- no overspeed drilling).
    fm = copy.deepcopy(fm0)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    buf, trace_buf, desc = [], [], []
    brng = np.random.default_rng(cfg["seed"] + 800)
    for c in range(1, cfg["n_warm"] + 1):
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + c)
        pr = W.traverse(fm, REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + c),
                        cfg["sigma_practice"], led, who="agent", kind="practice", drill_replan=1,
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + c), collect=True)
        buf.append(pr["trans"])
        trace_buf.append((pr["hand"].copy(), pr["acts"].copy(), pr["e_drill"].copy()))
        if len(trace_buf) > cfg["trace_window"]:
            trace_buf.pop(0)
        if len(buf) > cfg["trace_window"]:
            buf.pop(0)
        PX, PY = W.tensors(np.concatenate([b[0] for b in buf]),
                           np.concatenate([b[1] for b in buf]),
                           np.concatenate([b[2] for b in buf]))
        W.train_online(fm, opt, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])
        e = meter(fm, REACT, cfg["seed"] + 4000, replan=W.H_drill, who="agent")["e_drill"]
        desc.append(float(np.median(e)))
        P(f"[warm c{c:2d}] drilled@tempo(median) {desc[-1]:.4f}  practice_med "
          f"{np.median(pr['e_drill']):.4f}  t_priced={led.t_priced:8.1f}s")
    rng_range = max(m_stale.mean() - m_ceil.mean(), 1e-9)
    out["cal_descent"] = dict(e_by_cycle=desc,
                              frac_of_range_closed=[(m_stale.mean() - d) / rng_range for d in desc],
                              n_grad=cfg["n_grad"], adapt_lr=cfg["adapt_lr"])
    P(f"[CAL-D] fraction of stale->ceiling range closed by cycle: "
      f"{[round(f, 3) for f in out['cal_descent']['frac_of_range_closed']]}")
    save()

    # ================================================================= the audition
    sc = W.traverse(fm, REACT, q_sc, np.random.default_rng(cfg["seed"] + 4400),
                    cfg["sigma_perf"], led, who="agent", kind="score_set",
                    approach_plan=plan_sc, drill=False)
    S0 = sc["hand"]
    pool_hand = np.concatenate([t[0] for t in trace_buf])
    pool_acts = np.concatenate([t[1] for t in trace_buf])
    pool_err = np.concatenate([t[2] for t in trace_buf])
    crng = np.random.default_rng(cfg["seed"] + 953)
    ci = crng.permutation(len(pool_acts))[:cfg["n_cand"]]      # UNIFORM, never top-of-pool
    cands = pool_acts[ci]
    E = W.audition(cands, S0, W.W2, led, who="agent")

    fixed_unit, jf = World.select_fixed(E, cands)
    lib_unit, lib_info = W.select_library(E, cands, S0, cfg["n_lib"],
                                          np.random.default_rng(cfg["seed"] + 954))
    per_state_oracle = float(E.min(0).mean())
    best_fixed = float(E.mean(1).min())
    ev_cols, ho_cols = np.arange(0, E.shape[1], 2), np.arange(1, E.shape[1], 2)
    j_in = int(np.argmin(E[:, ev_cols].mean(1)))
    curse = dict(in_sample=float(E[j_in, ev_cols].mean()), held_out=float(E[j_in, ho_cols].mean()),
                 own_err_pick=float(E[int(np.argmin(pool_err[ci])), ho_cols].mean()))
    # What the LIBRARY scores on HELD-OUT columns when it is built on the other half. This is the
    # per-key optimism check crystallize's calibration forced: at 32 instances/key the per-key
    # in-sample -> held-out inflation was 1.56x against the global pick's 1.10x, which is why
    # `n_score` has to be sized PER CELL rather than overall. If the held-out library still beats
    # the held-out fixed unit, the state-conditioning is real rather than a selection artifact.
    def library_heldout(cols_fit, cols_eval, n_lib):
        Sf = S0[cols_fit]
        mu, sd = Sf.mean(0), Sf.std(0) + 1e-6
        C, lab = kmeans((Sf - mu) / sd, n_lib, np.random.default_rng(cfg["seed"] + 958))
        default = int(np.argmin(E[:, cols_fit].mean(1)))
        pick = {}
        for c_ in range(len(C)):
            m = lab == c_
            pick[c_] = int(np.argmin(E[:, cols_fit][:, m].mean(1))) if m.any() else default
        Ze = (S0[cols_eval] - mu) / sd
        ke = ((Ze[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        vals = [E[pick.get(int(ke[i]), default), cols_eval[i]] for i in range(len(cols_eval))]
        return float(np.mean(vals)), [int(v) for v in pick.values()]

    lib_heldout, lib_ho_picks = library_heldout(ev_cols, ho_cols, cfg["n_lib"])
    out["audition"] = dict(n_pool=int(len(pool_acts)), n_cand=int(len(cands)),
                           n_score=int(len(S0)), score_med=float(np.median(E.mean(1))),
                           best_fixed=best_fixed, per_state_oracle=per_state_oracle,
                           oracle_gain=float(best_fixed / max(per_state_oracle, 1e-9)),
                           library=lib_info, curse=curse,
                           library_heldout=lib_heldout, library_heldout_picks=lib_ho_picks,
                           fixed_heldout=curse["held_out"],
                           lib_over_fixed_heldout=float(curse["held_out"]
                                                        / max(lib_heldout, 1e-9)),
                           t_priced_after_audition=led.t_priced)
    P(f"[audition] pool={len(pool_acts)} cand={len(cands)} score={len(S0)} | best fixed "
      f"{best_fixed:.4f}  per-state oracle {per_state_oracle:.4f}  gain "
      f"{out['audition']['oracle_gain']:.2f}x | lib cells={lib_info['cell_sizes']} "
      f"distinct={lib_info['n_distinct']}")
    P(f"[audition] held-out: fixed {curse['held_out']:.4f}  library {lib_heldout:.4f} -> "
      f"{out['audition']['lib_over_fixed_heldout']:.2f}x | in-sample fixed {curse['in_sample']:.4f} "
      f"| min-own-err pick {curse['own_err_pick']:.4f}")
    save()

    # ================================================================= G1 boundary information
    ho_pos = W.tip(S0)
    speeds = np.linalg.norm(S0[:, n:], axis=1)
    nd = np.stack([null_direction(S0[i, :n].astype(np.float64), W.Ls) for i in range(len(S0))])
    null_coord = np.einsum("ij,ij->i", S0[:, :n] - S0[:, :n].mean(0)[None, :], nd)
    e_fixed = E[jf]
    Z = (S0 - S0.mean(0)) / (S0.std(0) + 1e-9)
    Zd = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    beta, *_ = np.linalg.lstsq(Zd, e_fixed, rcond=None)
    r2 = float(1.0 - (e_fixed - Zd @ beta).var() / max(e_fixed.var(), 1e-12))
    order = np.argsort(Zd @ beta)
    d = max(1, len(order) // 10)
    out["g1"] = dict(
        handover_tip_sd=float(np.linalg.norm(ho_pos.std(0))),
        handover_tip_sd_xy=ho_pos.std(0).tolist(),
        handover_tip_disp=float(np.linalg.norm(ho_pos.mean(0) - W.W1)),
        handover_speed_mean=float(speeds.mean()), handover_speed_sd=float(speeds.std()),
        handover_q_sd=S0[:, :n].std(0).tolist(), handover_qd_sd=S0[:, n:].std(0).tolist(),
        null_coord_sd=float(null_coord.std()), metering_noise_sd=noise_floor,
        spread_over_noise=float(np.linalg.norm(ho_pos.std(0)) / max(noise_floor, 1e-9)),
        fixed_unit_err_mean=float(e_fixed.mean()), fixed_unit_err_sd=float(e_fixed.std()),
        fixed_unit_err_cv=float(e_fixed.std() / max(e_fixed.mean(), 1e-9)),
        r2_err_on_handover=r2, decile_best=float(e_fixed[order[:d]].mean()),
        decile_worst=float(e_fixed[order[-d:]].mean()),
        per_state_oracle=per_state_oracle, best_fixed=best_fixed,
        oracle_gain=float(best_fixed / max(per_state_oracle, 1e-9)))
    P(f"[G1] hand-over tip sd {out['g1']['handover_tip_sd']:.4f} (disp "
      f"{out['g1']['handover_tip_disp']:.4f}, joint speed {out['g1']['handover_speed_mean']:.3f}, "
      f"null-coord sd {out['g1']['null_coord_sd']:.4f}) vs metering noise {noise_floor:.4f} "
      f"-> {out['g1']['spread_over_noise']:.1f}x")
    P(f"[G1] fixed unit err {e_fixed.mean():.4f} +- {e_fixed.std():.4f} "
      f"(cv {out['g1']['fixed_unit_err_cv']:.2f}) | R^2 on hand-over {r2:.3f} | decile "
      f"best/worst {out['g1']['decile_best']:.4f}/{out['g1']['decile_worst']:.4f} | oracle gain "
      f"{out['g1']['oracle_gain']:.2f}x")

    drift = [float(np.median(meter(fm, fixed_unit, cfg["seed"] + 7500 + r, who="agent")["e_drill"]))
             for r in range(cfg["n_drift"])]
    k = max(1, len(drift) // 3)
    out["g1"].update(post_commit_series=drift, post_commit_sd=float(np.std(drift)),
                     post_commit_drift=float(np.mean(drift[-k:]) - np.mean(drift[:k])))
    P(f"[G1] post-commit metering {np.round(drift, 4).tolist()} -> sd "
      f"{out['g1']['post_commit_sd']:.4f}, drift {out['g1']['post_commit_drift']:+.4f}")
    save()

    # ================================================================= G2 multimodality
    thr = float(np.quantile(pool_err, cfg["success_q"]))
    ok = np.nonzero(pool_err <= thr)[0]
    ok = ok[np.random.default_rng(cfg["seed"] + 955).permutation(len(ok))[:cfg["n_mode"]]]
    A_ok = pool_acts[ok]
    E_ok = W.audition(A_ok, S0, W.W2, led, who="agent")
    contrib = E_ok.mean(1)
    e_mean = W.replay(S0, A_ok.mean(0), W.W2)
    g2 = dict(success_threshold=thr, n_successful=int(len(A_ok)),
              contributor_mean=float(contrib.mean()), contributor_med=float(np.median(contrib)),
              contributor_best=float(contrib.min()), mean_of_successful=float(e_mean.mean()),
              ratio_mean_over_med=float(e_mean.mean() / max(np.median(contrib), 1e-9)),
              ratio_mean_over_best=float(e_mean.mean() / max(contrib.min(), 1e-9)))
    fin = np.empty((len(A_ok), SD), np.float32)
    paths = np.empty((len(A_ok), W.H_drill, 2), np.float32)
    s0m = S0[np.random.default_rng(cfg["seed"] + 956).integers(0, len(S0), len(A_ok))]
    for i in range(len(A_ok)):
        W.env.set_state(s0m[i, :n].astype(np.float64), s0m[i, n:].astype(np.float64))
        for h in range(W.H_drill):
            W.env.step(A_ok[i, h], W.fs)
            paths[i, h] = W.env.tip_pos()
        fin[i] = W.env.get_state()

    def mode_report(X, ks=(2, 3)):
        X = np.asarray(X, np.float64).reshape(len(X), -1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        r = {}
        for k_ in ks:
            C, lab = kmeans(X, k_, np.random.default_rng(cfg["seed"] + 957))
            within = np.mean([np.linalg.norm(X[lab == j] - C[j], axis=1).mean()
                              for j in range(len(C)) if (lab == j).any()])
            D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
            between = float(D[np.triu_indices(len(C), 1)].min()) if len(C) > 1 else 0.0
            # the mean WITHIN each cluster, replayed: if a world is genuinely multimodal the
            # within-mode mean survives and the across-mode mean does not
            within_means = []
            for j in range(len(C)):
                if (lab == j).sum() >= 2:
                    within_means.append(
                        float(W.replay(S0, A_ok[lab == j].mean(0), W.W2).mean()))
            r[f"k{k_}"] = dict(sizes=[int((lab == j).sum()) for j in range(len(C))],
                               within=float(within), between=between,
                               separation=float(between / max(within, 1e-9)),
                               per_cluster_score=[float(contrib[lab == j].mean())
                                                  if (lab == j).any() else None
                                                  for j in range(len(C))],
                               within_cluster_mean_replay=within_means)
        return r

    g2["modes_commands"] = mode_report(A_ok)
    g2["modes_terminal_state"] = mode_report(fin)
    g2["modes_tip_path"] = mode_report(paths)
    out["g2"] = g2
    P(f"[G2] mean-of-{len(A_ok)}-successful {g2['mean_of_successful']:.4f} vs contributor med "
      f"{g2['contributor_med']:.4f} / best {g2['contributor_best']:.4f} -> "
      f"{g2['ratio_mean_over_med']:.2f}x / {g2['ratio_mean_over_best']:.2f}x")
    for nm in ("modes_commands", "modes_terminal_state", "modes_tip_path"):
        P(f"[G2] {nm:22s} k2 sep={g2[nm]['k2']['separation']:.3f} sizes={g2[nm]['k2']['sizes']} "
          f"scores={[None if v is None else round(v, 4) for v in g2[nm]['k2']['per_cluster_score']]}"
          f" within-mean-replay={[round(v, 4) for v in g2[nm]['k2']['within_cluster_mean_replay']]}")

    # ================================================================= verdict
    out["verdict"] = {
        "G1_spread_over_noise": out["g1"]["spread_over_noise"],
        "G1_oracle_gain": out["g1"]["oracle_gain"],
        "G1_r2_err_on_handover": out["g1"]["r2_err_on_handover"],
        "G1_post_commit_sd": out["g1"]["post_commit_sd"],
        "G2_mean_over_med": g2["ratio_mean_over_med"],
        "G2_mean_over_best": g2["ratio_mean_over_best"],
        "G3_usable_range": out["g3"]["usable_range"],
        "G3_range_over_noise": out["g3"]["range_over_noise"],
        "G4_horizon_stale": out["g4"]["horizon_stale_005"],
        "G4_horizon_ceiling": out["g4"]["horizon_ceil_005"],
        "G4_H_drill": W.H_drill,
        "lib_over_fixed_heldout": out["audition"]["lib_over_fixed_heldout"],
    }
    out["ledger"] = led.snapshot()
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[verdict] {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out['verdict'].items()})}")
    P(f"[save] {outdir}")
    return out


@app.local_entrypoint()
def gates(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # WORLDS: "<label>:<curl_b>:<push_a>" -- one Modal container each, single seed. The difficulty
    # knob is swept rather than solved, and whether the soft obstacle is needed for G2 is measured.
    worlds: str = "b6:6:0,b14:14:0,b26:26:0,b14obs:14:-9",
    # ---- the piece
    w1: str = DEF_W1,
    w2: str = DEF_W2,
    h_app: int = 14,
    h_drill: int = 20,
    q_jit: float = 0.15,
    null_jit: float = 0.30,
    start_mode: str = "iso",
    # ---- the plant
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,1.1,0.8",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    timestep: float = 0.002,
    wrap_limit: float = 3.0,
    # ---- the hard region
    patch_sigma: float = 0.08,
    patch_center: str = "",
    push_sigma: float = 0.0,
    # ---- the diet
    pool_ou: int = 6000,
    pool_reach: int = 12000,
    exclude_w: float = 0.1,
    n_corridor: int = 10,
    corridor_replan: int = 5,
    ep_len: int = 14,
    n_par: int = 16,
    op_q_range: float = 0.25,
    sigma_u: float = 0.15,
    ou_sigma: float = 0.7,
    ou_theta: float = 0.15,
    collect_k_shoot: int = 512,
    collect_cem_iters: int = 5,
    reach_amp: float = 1.2,
    reach_lo: float = 0.25,
    reach_hi: float = 0.50,
    # ---- FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    fm_steps_boot: int = 3000,
    adapt_lr: float = 3e-4,
    n_grad: int = 6,
    replay_frac: float = 0.5,
    # ---- controllers
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    planner_grid: str = "256:4,512:6,1024:8",
    # ---- practice / metering
    n_warm: int = 16,
    batch: int = 24,
    n_rt: int = 48,
    n_eval: int = 48,
    n_meter_rep: int = 5,
    n_drift: int = 8,
    trace_window: int = 6,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    # ---- the compile op
    n_cand: int = 40,
    n_score: int = 192,      # >=48 score states per library cell: crystallize measured
                             # per-key in-sample->held-out optimism at 1.56x with only 32
    n_lib: int = 4,
    n_mode: int = 24,
    success_q: float = 0.35,
    # ---- policy head (the `regress` control's machinery, exercised in round 1)
    pol_hidden: int = 128,
    pol_layers: int = 2,
    pol_lr: float = 1e-3,
    pol_steps: int = 1500,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500; n_corridor = 3
        fm_steps = 1200; fm_steps_boot = 600
        n_warm = 3; batch = 8; n_rt = 12; n_eval = 12; n_meter_rep = 2; n_drift = 3
        n_cand = 6; n_score = 16; n_mode = 8; n_lib = 2
        k_shoot = 256; cem_iters = 4; planner_grid = "256:4"
        pol_steps = 300
        tag = tag or "smoke"
    tag = tag or "default"
    wl = []
    for spec in worlds.split(","):
        if not spec.strip():
            continue
        parts = spec.split(":")
        wl.append((parts[0], float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
    base = dict(
        tag=tag, seed=seed,
        w1=[float(x) for x in w1.split(",")], w2=[float(x) for x in w2.split(",")],
        h_app=h_app, h_drill=h_drill, q_jit=q_jit, null_jit=null_jit, start_mode=start_mode,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, timestep=timestep,
        wrap_limit=wrap_limit, patch_sigma=patch_sigma,
        patch_center=([float(x) for x in patch_center.split(",")] if patch_center else None),
        push_sigma=(push_sigma or patch_sigma), push_center=None,
        pool_ou=pool_ou, pool_reach=pool_reach, exclude_w=exclude_w,
        n_corridor=n_corridor, corridor_replan=corridor_replan,
        ep_len=ep_len, n_par=n_par, op_q_range=op_q_range, sigma_u=sigma_u,
        ou_sigma=ou_sigma, ou_theta=ou_theta,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, fm_steps_boot=fm_steps_boot, adapt_lr=adapt_lr, n_grad=n_grad,
        replay_frac=replay_frac,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        planner_grid=[[int(v) for v in p.split(":")] for p in planner_grid.split(",") if p],
        n_warm=n_warm, batch=batch, n_rt=n_rt, n_eval=n_eval, n_meter_rep=n_meter_rep,
        n_drift=n_drift, trace_window=trace_window,
        sigma_practice=sigma_practice, sigma_perf=sigma_perf, d_fb=d_fb,
        n_cand=n_cand, n_score=n_score, n_lib=n_lib, n_mode=n_mode, success_q=success_q,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps,
    )
    cfgs = [{**base, "world": lb, "curl_b": b, "push_a": a} for lb, b, a in wl]
    outs = list(run_gates.map(cfgs))
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for o in outs:
        with open(os.path.join(localdir, f"gates_{o['world']}.json"), "w") as fh:
            json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(outs)} world files to {localdir}")
