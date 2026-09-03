"""jacobian_teacher Phase A — the four gates that must pass before any treatment is run.

[`SPEC.md`](SPEC.md) §Phase A. Nothing here is a treatment; every number is either an identity
check or a property of the instrument. Gate 2 is the one that can produce a *finding* rather
than a pass/fail, and the SPEC says so: "If a forecast-accurate FM does not have a
direction-accurate Jacobian, that is the first finding of this node and it changes what Phase B
can mean; record it before anything else."

  1. FORK FIDELITY. `core.fork_defaults()` is re-derived from `../ballistic/arm/arm_readapt.py`'s
     own source with `ast` and compared key by key; then a fixed command sequence is executed
     through the forked substrate and its trajectory hashed, and `fk_torch` is checked against
     the analytic `fk` the fork's cost function and this node's endpoint map both call.
  2. THE JACOBIAN ORACLE IS TRUSTWORTHY. Kinematic half: the analytic ∂fk/∂q against MuJoCo's
     own `mj_jacSite` (`fk` itself is already verified to ~2e-16, arm_substrate P0). Dynamic
     half: a central-difference STEP-SIZE SWEEP, since a finite-difference derivative has no
     error bar of its own — the plateau between truncation error (eps too large) and
     cancellation error (eps too small) is the evidence. Then the headline: the MATCHED-CEILING
     FM's Jacobian against that oracle, one-step and composed-endpoint.
  3. THE REWARD ARM REPRODUCES. `rbl` on this substrate, plus the measured norm ratio
     ‖g_ebl‖/‖g_rbl‖ that says what `mix_scale` would make β an interpolating axis rather than
     a switch between two differently-sized vectors.
  4. `ebl_imagined` AT THE MATCHED FM ≈ `ballistic_cem`. Gradient planning and sampling planning
     on the SAME model should agree when the model is right. If they do not, the composed
     rollout or the CEM sizing is wrong (arm_substrate P3: a pusher-tuned `k_shoot=256,
     cem_iters=4` silently flattens this substrate's transmission slope into a null).

Run:
    cd experiments/
    modal run mjc/jacobian_teacher/phase_a.py::phase_a --quick                  # smoke
    modal run --detach mjc/jacobian_teacher/phase_a.py::phase_a --tag pa_s0 --seed 0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

FORK_SRC = "ballistic/arm/arm_readapt.py"


def _entrypoint_defaults(path):
    """Re-derive `arm_readapt`'s signature defaults from its SOURCE (no import: importing it
    would register a second Modal function). Returns them in `core.fork_defaults()`'s spelling."""
    import ast

    tree = ast.parse(open(path).read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "arm_readapt")
    names = [a.arg for a in fn.args.args][-len(fn.args.defaults):]
    raw = {n: ast.literal_eval(d) for n, d in zip(names, fn.args.defaults)}
    out = {}
    for k, v in raw.items():
        if k in ("milestones",):
            out[k] = [int(x) for x in v.split(",") if x.strip()]
        elif k in ("link_lengths", "link_masses", "q_center"):
            out[k] = [float(x) for x in v.split(",")]
        elif k == "plan_h":
            out["plan_H"] = v
        else:
            out[k] = v
    return out


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_phase_a(cfg: dict) -> dict:
    import hashlib
    import os
    import numpy as np
    import torch

    from mjc import arm_env as arm_env_mod
    from mjc.arm_env import fk
    from mjc.jacobian_teacher import core as C

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    sub = C.ArmSubstrate(cfg)
    tcfg = {k: cfg[k] for k in C.teacher_defaults()}
    n, H, AD = sub.n, sub.H, sub.AD
    res = {"config": cfg, "gates": {}}
    print(f"[setup] device={sub.device} n={n} H={H} AD={AD} b0={cfg['b0']} -> b1={cfg['b1']}",
          flush=True)

    # ================================================================= GATE 1
    print("\n=== GATE 1: fork fidelity ===", flush=True)
    g1 = {"passed": True}
    src = os.path.join(os.path.dirname(os.path.dirname(arm_env_mod.__file__)), "mjc", FORK_SRC)
    if not os.path.exists(src):
        src = os.path.join(os.path.dirname(arm_env_mod.__file__), FORK_SRC)
    forked = C.fork_defaults()
    if os.path.exists(src):
        upstream = _entrypoint_defaults(src)
        mism = {k: (forked[k], upstream[k]) for k in forked
                if k in upstream and forked[k] != upstream[k]}
        missing = sorted(set(forked) - set(upstream))
        g1.update(source=src, n_compared=len(set(forked) & set(upstream)),
                  mismatches=mism, not_in_upstream=missing)
        g1["passed"] = not mism
        print(f"[1a] compared {g1['n_compared']} defaults against {FORK_SRC}: "
              f"{'IDENTICAL' if not mism else 'MISMATCH ' + str(mism)}", flush=True)
        if missing:
            print(f"[1a] (keys absent upstream, expected only for renames: {missing})", flush=True)
    else:
        g1["passed"] = False
        g1["source"] = f"NOT FOUND (looked at {src})"
        print(f"[1a] FAIL: could not locate {FORK_SRC}", flush=True)

    env1 = sub.make_env(cfg["b1"])
    rng = np.random.default_rng(12345)                       # fixed, independent of cfg['seed']
    q0 = sub.qc[None, :] + rng.uniform(-0.25, 0.25, (8, n))
    st = np.concatenate([q0, np.zeros((8, n))], 1).astype(np.float32)
    seq = np.tanh(rng.normal(0, 1, (8, H, AD))).astype(np.float32)
    fin = sub.execute(env1, st, seq)
    g1["traj_hash"] = hashlib.sha256(np.ascontiguousarray(fin).tobytes()).hexdigest()[:16]
    g1["traj_tip_mean"] = float(np.linalg.norm(fk(fin[:, :n].astype(np.float64), sub.Ls),
                                               axis=1).mean())
    g1["nonfinite"] = env1.nonfinite()
    print(f"[1b] fixed-command trajectory hash = {g1['traj_hash']}  "
          f"mean |tip| = {g1['traj_tip_mean']:.6f}  nonfinite = {g1['nonfinite']}", flush=True)

    qt = torch.tensor(fin[:, :n], device=sub.device, dtype=torch.float32)
    g1["fk_torch_vs_fk"] = float(np.abs(sub.fk_torch(qt).detach().cpu().numpy()
                                        - fk(fin[:, :n].astype(np.float64), sub.Ls)).max())
    print(f"[1c] max |fk_torch - fk| = {g1['fk_torch_vs_fk']:.3e} "
          f"(float32 round-off; the CEM cost and the endpoint map share this path)", flush=True)
    g1["passed"] = g1["passed"] and g1["fk_torch_vs_fk"] < 1e-5 and g1["nonfinite"] == 0
    res["gates"]["fork_fidelity"] = g1

    # ================================================================= GATE 2 (a) kinematic
    print("\n=== GATE 2: the Jacobian oracle ===", flush=True)
    g2 = {}
    import mujoco

    qs = sub.qc[None, :] + rng.uniform(-0.9, 0.9, (32, n))
    J_an = sub.fk_jac(qs)                                     # (32, 2, n)
    J_mj = np.zeros_like(J_an)
    jp = np.zeros((3, env1.model.nv)); jr = np.zeros((3, env1.model.nv))
    for b in range(len(qs)):
        env1.set_state(qs[b], np.zeros(n))
        mujoco.mj_jacSite(env1.model, env1.data, jp, jr, env1.tip_sid)
        J_mj[b] = jp[:2, :n]
    g2["fk_jac_vs_mj_jacSite_max"] = float(np.abs(J_an - J_mj).max())
    print(f"[2a] kinematic half: max |analytic dfk/dq - mj_jacSite| = "
          f"{g2['fk_jac_vs_mj_jacSite_max']:.3e}", flush=True)

    # ----------------------------------------------------------- (b) FD step-size sweep
    ss, gg, t0 = sub.eval_geometry(cfg["seed"] + 7, B=cfg["fd_sweep_n"])
    seq_p = np.clip(rng.normal(0, 0.5, (len(ss), H, AD)), -1, 1).astype(np.float32)
    sweep = {}
    ref = None
    for eps in cfg["fd_eps_sweep"]:
        Jp = C.plant_endpoint_jac_fd(sub, env1, ss, seq_p, eps)
        if ref is None:
            ref = Jp
            sweep[str(eps)] = {"cos_vs_ref": 1.0, "rel_change": 0.0,
                               "norm": float(np.linalg.norm(Jp))}
        else:
            m = C.direction_metrics(Jp, ref)
            sweep[str(eps)] = {"cos_vs_ref": m["cos"], "rel_change": abs(m["scale"] - 1.0),
                               "norm": float(np.linalg.norm(Jp))}
        print(f"[2b] eps={eps:<8g} cos_vs_ref={sweep[str(eps)]['cos_vs_ref']:.6f}  "
              f"|J|={sweep[str(eps)]['norm']:.4f}", flush=True)
    g2["fd_step_sweep"] = sweep
    g2["fd_eps_used"] = tcfg["fd_eps"]
    print(f"[2b] a central difference has no error bar of its own: the PLATEAU across eps is "
          f"the evidence. Using eps={tcfg['fd_eps']}.", flush=True)

    # ----------------------------------------------------------- (c) the FM ladder's direction
    # Build the probe set and the two anchor FMs exactly as Cut 4c-arm does.
    nS, nU, nS2 = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 11)
    sub.set_norm(nS, nU, nS2)
    ref_net = sub.mlp(cfg["seed"] + 40)
    rS, rU, rS2 = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 400)
    sub.train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                    rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    ev = sub.eval_geometry(cfg["seed"] + 7)
    rec = []
    sub.rollout(lambda s, g: sub.mpc_plan(ref_net, s, g, np.random.default_rng(cfg["seed"] + 6000)),
                1, env1, ev[0], ev[1], ev[2], record=rec)
    sub.rollout(lambda s, g: sub.mpc_plan(ref_net, s, g, np.random.default_rng(cfg["seed"] + 6001)),
                H, env1, ev[0], ev[1], ev[2], record=rec)
    probe = (np.array([r[0] for r in rec], np.float32),
             np.array([r[1] for r in rec], np.float32),
             (np.array([r[2] for r in rec], np.float32)
              - np.array([r[0] for r in rec], np.float32)).astype(np.float32))
    print(f"[probes] task probe {len(probe[0])} transitions (matched-FM reaches)", flush=True)

    env0 = sub.make_env(cfg["b0"])
    fm_ceil = sub.mlp(cfg["seed"] + 42)
    Sc, Uc, S2c = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 20)
    sub.train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                    Sc, Uc, S2c, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
    fm_stale = sub.mlp(cfg["seed"] + 41)
    S0, U0, S20 = sub.pool(env0, cfg["pool_n"], cfg["seed"] + 10)
    sub.train_steps(fm_stale, torch.optim.Adam(fm_stale.parameters(), lr=cfg["fm_lr"]),
                    S0, U0, S20, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))

    pol0, pn0 = sub.clone_policy(fm_stale, dir_center=cfg["train_dir"],
                                 dir_halfwidth=cfg["train_dir_halfwidth"])
    jac_set = sub.eval_geometry(cfg["seed"] + 77, B=max(tcfg["jac_n"], 8),
                                dir_center=cfg["train_dir"],
                                dir_halfwidth=cfg["train_dir_halfwidth"])
    anchors = {}
    for name, net in (("matched_ceiling", fm_ceil), ("stale_predrift", fm_stale)):
        anchors[name] = C.fm_readings(sub, net, env1, probe, jac_set, tcfg, pol0, pn0)
        a = anchors[name]
        print(f"[2c] {name:16s} fm_err={a['fm_err']:.4f} | one-step: cos={a['jac1']['cos']:.4f} "
              f"sign_w={a['jac1']['sign_agree_w']:.3f} scale={a['jac1']['scale']:.3f} | "
              f"endpoint: cos={a['jacE']['cos']:.4f} sign_w={a['jacE']['sign_agree_w']:.3f} "
              f"vjp_cos={a['jacE']['vjp_cos']:.4f} scale={a['jacE']['scale']:.3f}", flush=True)
    g2["anchors"] = anchors
    print("[2c] THE GATE: a forecast-accurate FM is not automatically a direction-accurate one. "
          "Both readings are recorded; neither is interpreted here.", flush=True)
    res["gates"]["jacobian_oracle"] = g2
    volume.commit()

    # ================================================================= GATES 3 & 4
    print("\n=== GATES 3 & 4: the reward arm reproduces; gradient planning ≈ sampling planning "
          "===", flush=True)
    pool_tr = C.make_reach_pool(sub, cfg["seed"] + 800, cfg["train_pool_n"],
                                cfg["train_dir"], cfg["train_dir_halfwidth"])
    ev_band = sub.eval_geometry(cfg["seed"] + 7, dir_center=cfg["train_dir"],
                                dir_halfwidth=cfg["train_dir_halfwidth"])
    res["band_stats"] = sub.last_band_stats
    print(f"[band] train band {cfg['train_dir']:+.0f}±{cfg['train_dir_halfwidth']:.0f}°: "
          f"in-band {100 * sub.last_band_stats['in_band_frac']:.1f}%, mean|offset| "
          f"{sub.last_band_stats['mean_abs_offset']:.1f}°", flush=True)
    eval_sets = {"train_band": (ev_band[0], ev_band[1], ev_band[2], env1)}

    # The fork's incumbents on the same eval set and the same FM: zeroth-order uses of the
    # identical model (`ballistic_cem` samples, `ballistic_bc` pre-compiles) plus `reactive`
    # as the blind-grader control. `ballistic_bc` is cloned PER FM — `pol0` is the stale FM's
    # program and reusing it for the ceiling would report the wrong controller.
    inc = {}
    for name, net in (("matched_ceiling", fm_ceil), ("stale_predrift", fm_stale)):
        r = np.random.default_rng(cfg["seed"] + 7001)
        d, lat, rad = sub.rollout(lambda s, g: sub.mpc_plan(net, s, g, r), H, env1, *ev_band)
        inc[f"ballistic_cem@{name}"] = {"dist": d, "lat": lat, "rad": rad}
        # the SAME objective `ebl_imagined` descends — terminal endpoint distance alone, no
        # running tip cost and no terminal velocity penalty — so gate 4 compares two PLANNERS
        # rather than two objectives
        r = np.random.default_rng(cfg["seed"] + 7002)
        d, lat, rad = sub.rollout(lambda s, g: sub.mpc_plan(net, s, g, r, terminal_only=True),
                                  H, env1, *ev_band)
        inc[f"ballistic_cem_ep@{name}"] = {"dist": d, "lat": lat, "rad": rad}
        if name == "stale_predrift":
            pb, pnb = pol0, pn0                      # by construction, the shared initial policy
        else:
            pb, pnb = sub.clone_policy(net, seed_off=57, dir_center=cfg["train_dir"],
                                       dir_halfwidth=cfg["train_dir_halfwidth"], verbose=False)
        d, lat, rad = C.evaluate_policy(sub, pb, pnb, env1, *ev_band)
        inc[f"ballistic_bc@{name}"] = {"dist": d, "lat": lat, "rad": rad}
        r = np.random.default_rng(cfg["seed"] + 7000)
        d, lat, rad = sub.rollout(lambda s, g: sub.mpc_plan(net, s, g, r), 1, env1, *ev_band)
        inc[f"reactive@{name}"] = {"dist": d, "lat": lat, "rad": rad}
    for k, v in inc.items():
        print(f"[inc] {k:32s} dist={v['dist']:.4f} lat={v['lat']:+.4f} rad={v['rad']:+.4f}",
              flush=True)
    res["incumbents"] = inc

    # the two action gradients on the SAME batch, so the norm ratio is measured not guessed
    idx = np.random.default_rng(cfg["seed"] + 900).integers(0, cfg["train_pool_n"],
                                                            size=tcfg["train_batch"])
    st_, gl_ = pool_tr[0][idx], pool_tr[1][idx]
    with torch.no_grad():
        mu_np = sub.policy_mu(pol0, pn0, st_, gl_).cpu().numpy()
    act = np.clip(mu_np + tcfg["sigma"] * np.random.default_rng(cfg["seed"] + 901)
                  .normal(0, 1, mu_np.shape), -1, 1).astype(np.float32)
    tips = fk(sub.execute(env1, st_, act)[:, :n].astype(np.float64), sub.Ls).astype(np.float32)
    norms = {}
    for t in ("rbl", "ebl_sensory", "ebl_imagined"):
        g_, d_ = C.action_gradient(sub, fm_ceil, t, st_, gl_, mu_np, act, tips, tcfg, [None])
        norms[t] = d_["g_norm"]
    norms["ratio_ebl_over_rbl"] = norms["ebl_sensory"] / max(norms["rbl"], 1e-12)
    print(f"[3a] action-gradient norms on one shared batch: "
          + "  ".join(f"{k}={v:.4g}" for k, v in norms.items()), flush=True)
    print(f"[3a] -> mix_scale = {norms['ratio_ebl_over_rbl']:.4g} would make β an interpolating "
          f"axis; Eq. 2's raw sum is mix_scale = 1.", flush=True)
    res["gradient_norms"] = norms

    arms = {}
    for teacher, net, lbl in (("rbl", fm_ceil, "rbl"),
                              ("ebl_imagined", fm_ceil, "ebl_imagined@matched"),
                              ("ebl_sensory", fm_ceil, "ebl_sensory@matched")):
        pol = sub.build_policy(); pol.load_state_dict(pol0.state_dict())
        curve = C.train_teacher(sub, net, pol, pn0, env1, teacher, tcfg, pool_tr,
                                eval_sets, cfg["seed"], tag="A:")
        arms[lbl] = {"curve": curve,
                     "summary": C.summarize_curve(curve, list(eval_sets),
                                                  tcfg["plateau_evals"])}
        volume.commit()
    res["arms"] = arms

    cem = inc["ballistic_cem@matched_ceiling"]["dist"]
    cem_ep = inc["ballistic_cem_ep@matched_ceiling"]["dist"]
    bc = inc["ballistic_bc@matched_ceiling"]["dist"]
    sm = arms["ebl_imagined@matched"]["summary"]
    imag, imag_best = sm["train_band_final"], sm["train_band_best"]
    res["gates"]["gradient_vs_sampling_planning"] = {
        "ballistic_cem": cem, "ballistic_cem_endpoint_only": cem_ep, "ballistic_bc": bc,
        "ebl_imagined_final": imag, "ebl_imagined_best": imag_best,
        "ratio_vs_cem": imag / max(cem, 1e-12),
        "ratio_vs_cem_ep": imag / max(cem_ep, 1e-12),
        "ratio_vs_bc": imag / max(bc, 1e-12),
        "best_ratio_vs_cem_ep": imag_best / max(cem_ep, 1e-12)}
    print(f"\n[4] ebl_imagined@matched: final={imag:.4f} best={imag_best:.4f}", flush=True)
    print(f"[4]   vs ballistic_bc      = {bc:.4f}  ratio {imag / max(bc, 1e-12):.2f}x   "
          f"<- THE MATCHED COMPARISON: both AMORTISED into one π(s,goal) through the same FM, "
          f"one by cloning a sampling planner, one by gradient descent on the model's own "
          f"endpoint error. Function-approximation error is common to both.", flush=True)
    print(f"[4]   vs ballistic_cem     = {cem:.4f}  ratio {imag / max(cem, 1e-12):.2f}x   "
          f"(the fork's running tip cost + velocity penalty, re-optimised PER EPISODE)",
          flush=True)
    print(f"[4]   vs ballistic_cem_ep  = {cem_ep:.4f}  ratio {imag / max(cem_ep, 1e-12):.2f}x   "
          f"(the same TERMINAL objective ebl_imagined descends, but per-episode — the gap to "
          f"this one is the price of amortisation, not of the gradient)", flush=True)
    print(f"[4] final vs best ({imag:.4f} vs {imag_best:.4f}): a final much worse than the "
          f"best is the model-exploitation signature, and the curve's `imag`/`fcast` columns "
          f"say whether the model's own predicted error kept falling while the real one rose.",
          flush=True)
    res["gates"]["reward_arm"] = arms["rbl"]["summary"]
    print(f"[3b] rbl start={arms['rbl']['summary']['train_band_start']:.4f} -> "
          f"final={arms['rbl']['summary']['train_band_final']:.4f} over "
          f"{tcfg['reach_budget']} executed reaches", flush=True)

    outdir = os.path.join(DATA_DIR, "jacobian_teacher", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "phase_a.json"), "w") as fh:
        json.dump(res, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote {outdir}/phase_a.json", flush=True)
    print("[done] phase_a complete", flush=True)
    return {"results": res}


@app.local_entrypoint()
def phase_a(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    reach_budget: int = 20000,
    train_batch: int = 128,
    eval_every: int = 2000,
    eval_schedule: str = "128,256,512,1024,2048,4096,6144,8192,12288,16384,20000",
    sigma: float = 0.15,
    lr_rbl: float = 5e-4,
    lr_ebl: float = 5e-4,
    fd_eps: float = 1e-3,
    jac_n: int = 16,
    jac_probe_n: int = 512,
    train_pool_n: int = 8192,
    train_dir: float = 0.0,
    train_dir_halfwidth: float = 30.0,
    fd_sweep_n: int = 8,
    band_oversample: int = 40,
):
    import os

    from mjc.jacobian_teacher.defaults import fork_defaults, teacher_defaults

    cfg = dict(fork_defaults())
    cfg.update(teacher_defaults())
    cfg.update(tag=tag or ("smoke" if quick else "phase_a"), seed=seed,
               reach_budget=reach_budget, train_batch=train_batch, eval_every=eval_every,
               eval_schedule=[int(x) for x in eval_schedule.split(",") if x.strip()],
               sigma=sigma, lr_rbl=lr_rbl, lr_ebl=lr_ebl, fd_eps=fd_eps, jac_n=jac_n,
               jac_probe_n=jac_probe_n, train_pool_n=train_pool_n, train_dir=train_dir,
               train_dir_halfwidth=train_dir_halfwidth, fd_sweep_n=fd_sweep_n,
               band_oversample=band_oversample,
               fd_eps_sweep=[1e-3, 1e-4, 3e-4, 3e-3, 1e-2, 3e-2])
    if quick:
        # arm_readapt --quick's substrate downsizing, plus this node's own.
        cfg.update(pool_n=3000, fm_steps=1500, probe_n=800, n_eval=12,
                   k_shoot=256, cem_iters=4, bc_tuples=1200, pol_steps=600,
                   fm_hidden=128, fm_layers=2,
                   reach_budget=1024, train_batch=64, eval_every=256,
                   eval_schedule=[64, 128, 256, 512, 1024],
                   train_pool_n=1024, jac_n=6, jac_probe_n=128, fd_sweep_n=4,
                   fd_eps_sweep=[1e-3, 1e-4, 1e-2])
        print("[quick] NOTE: k_shoot=256/cem_iters=4 is the pusher-tuned budget that "
              "arm_substrate P3 shows flattens this substrate's slope — smoke only.")
    out = run_phase_a.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "phase_a_" + cfg["tag"])
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "phase_a.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote phase_a.json to {localdir}")
