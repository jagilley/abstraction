"""jacobian_teacher Phase B — three teachers on one policy, across the FM-quality ladder.

[`SPEC.md`](SPEC.md) §Phase B. Parent: [`../README.md`](../README.md). Forks
[`../ballistic/arm/arm_readapt.py`](../ballistic/arm/arm_readapt.py) (Cut 4c-arm) through
[`core.py`](core.py); [`phase_a.py`](phase_a.py) gates the fork and the oracle and must be run
first. Reading: Garibbo, Filipe, Aitchison & Costa 2026, eqs. 1–2.

THE OBJECT. One behavior-cloned motor program π(s, goal) -> H×n_act, born once from the
PRE-DRIFT (field-free) FM — the naive arm of the Shadmehr protocol — and then adapted in the
curl world by each teacher in turn, from an identical copy, on identical reaches, with identical
exploration noise. The teachers differ in one line:

    rbl_cont / rbl_bin   g = −δ (u − μ)/σ²                       Eq. 1, δ = r − v, continuous
                                                                 (−distance) or binary (in the
                                                                 reward zone; Izawa & Shadmehr's
                                                                 phenomena are measured on the
                                                                 binary one)
    ebl_sensory          g = (y − y*)ᵀ ∂ŷ/∂u                      the new object: the REALIZED
                                                                 endpoint error pulled back
                                                                 through the FM's action
                                                                 Jacobian. The FM never sees the
                                                                 error; it supplies the
                                                                 coordinate transform.
    ebl_imagined         g = (ŷ − y*)ᵀ ∂ŷ/∂u                      the same gradient with the
                                                                 MODEL's predicted error. No
                                                                 sensory feedback enters at all,
                                                                 so this is gradient planning
                                                                 amortised into π — and it is
                                                                 what separates the JACOBIAN
                                                                 from the FORECAST: on a stale
                                                                 FM the two errors disagree
                                                                 while the Jacobian is shared.
    mixed@β              g = β·g_ebl + (1−β)·g_rbl                Eq. 2

THE AXIS, READ TWO WAYS. Cut 4c-arm's reward-free re-adaptation ladder supplies FMs of
controlled quality (stale pre-drift FM -> milestones -> matched ceiling). Every rung is reported
with BOTH the record's scalar (forecast error on the task probe) and this node's direction
reading (Jacobian accuracy against a finite-difference plant oracle, one-step and composed, plus
`vjp_cos` — the cosine between the teaching vectors the two Jacobians produce for the errors the
policy actually makes). `arm_substrate` P5/P7's standing rule is respected: both are reported per
rung against the ladder's own transition axis; no outcome is regressed on `fm_err`.

THE DYSMETRIA CONTROL. A fixed (2, n_act) sign mask on the vector-Jacobian product flips the FM's
believed sign of ∂y_k/∂u_j, the FM otherwise untouched — the generalization of Garibbo Fig. 3g–j
(where flipping each of a 2×2 sensitivity-derivative matrix's components in turn produced
hypermetria, hypometria, displacement, and one null) to our 2×3. Read on the SIGNED radial and
lateral deviations, which is the one line this node adds to the fork's readout: only a signed
along-reach term separates overshoot from undershoot, and only the lateral one separates either
from displacement.

Per repo norms no outcome is interpreted here.

Run:
    cd experiments/
    modal run mjc/jacobian_teacher/jacobian_teacher.py::jacobian_teacher --quick     # smoke
    modal run --detach mjc/jacobian_teacher/jacobian_teacher.py::jacobian_teacher \
        --tag jt_s0 --seed 0
    # the capacity ladder (a second run, not a cross-product):
    modal run --detach mjc/jacobian_teacher/jacobian_teacher.py::jacobian_teacher \
        --tag jt_cap_s0 --seed 0 --capacity-ladder "8,16,32,64,256"
    python3 mjc/jacobian_teacher/jacobian_teacher_figure.py --tags jt_s0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_jacobian_teacher(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch

    from mjc.arm_env import collect_pool
    from mjc.jacobian_teacher import core as C

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    sub = C.ArmSubstrate(cfg)
    tcfg = {k: cfg[k] for k in C.teacher_defaults()}
    n, H, AD = sub.n, sub.H, sub.AD
    env0, env1 = sub.make_env(cfg["b0"]), sub.make_env(cfg["b1"])
    outdir = os.path.join(DATA_DIR, "jacobian_teacher", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    print(f"[setup] device={sub.device} n={n} H={H} drift b0={cfg['b0']} -> b1={cfg['b1']}\n"
          f"        teachers={cfg['teachers']} betas={cfg['beta_sweep']} "
          f"ladder={'capacity ' + str(cfg['capacity_ladder']) if cfg['capacity_ladder'] else cfg['milestones']}\n"
          f"        budget={tcfg['reach_budget']} executed reaches/arm, batch "
          f"{tcfg['train_batch']}, sigma={tcfg['sigma']}", flush=True)

    # ---------------------------------------------------------------- normalization + probes
    # Fixed once from the OPERATING dynamics (b1) and shared by every FM on the ladder, so
    # snapshots are not each scored in their own units (Cut 4c-arm).
    nS, nU, nS2 = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 11)
    sub.set_norm(nS, nU, nS2)

    ref_net = sub.mlp(cfg["seed"] + 40)
    rS, rU, rS2 = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 400)
    sub.train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                    rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))
    ev_full = sub.eval_geometry(cfg["seed"] + 7)
    rec = []
    for off, re_ in ((6000, 1), (6001, H)):
        sub.rollout(lambda s, g: sub.mpc_plan(ref_net, s, g,
                                              np.random.default_rng(cfg["seed"] + off)),
                    re_, env1, *ev_full, record=rec)
    pS = np.array([r[0] for r in rec], np.float32)
    pU = np.array([r[1] for r in rec], np.float32)
    probe = (pS, pU, (np.array([r[2] for r in rec], np.float32) - pS).astype(np.float32))
    print(f"[probes] task probe {len(pS)} transitions (matched-FM reaches)", flush=True)

    # ---------------------------------------------------------------- the eval geometry
    D, W = cfg["train_dir"], cfg["train_dir_halfwidth"]
    ev_band = sub.eval_geometry(cfg["seed"] + 7, dir_center=D, dir_halfwidth=W)
    gen_sets = {f"gen@{int(o)}": sub.eval_geometry(cfg["seed"] + 700 + int(o),
                                                   dir_center=D + o,
                                                   dir_halfwidth=cfg["gen_halfwidth"])
                for o in cfg["gen_offsets"]}
    held = f"gen@{int(cfg['held_out_offset'])}"
    eval_sets = {"train_band": (*ev_band, env1),
                 "held_out": (*gen_sets[held], env1)}
    jac_set = sub.eval_geometry(cfg["seed"] + 77, B=max(tcfg["jac_n"], 8),
                                dir_center=D, dir_halfwidth=W)
    pool_tr = C.make_reach_pool(sub, cfg["seed"] + 800, cfg["train_pool_n"], D, W)
    band_stats = {}
    for nm, (dc, hw) in [("train_band", (D, W))] + [
            (f"gen@{int(o)}", (D + o, cfg["gen_halfwidth"])) for o in cfg["gen_offsets"]]:
        sub.eval_geometry(cfg["seed"] + 7 if nm == "train_band"
                          else cfg["seed"] + 700 + int(float(nm.split("@")[1])),
                          dir_center=dc, dir_halfwidth=hw)
        band_stats[nm] = sub.last_band_stats
    results_band = band_stats
    print(f"[geometry] train band {D:+.0f}±{W:.0f}°, held-out {held}, "
          f"generalization offsets {cfg['gen_offsets']}", flush=True)
    for nm, st_ in band_stats.items():
        print(f"           {nm:10s} in-band {100 * st_['in_band_frac']:5.1f}%  "
              f"mean|offset| {st_['mean_abs_offset']:5.1f}°", flush=True)

    # ---------------------------------------------------------------- the FM ladder
    rungs = []
    if cfg["capacity_ladder"]:
        # A CAPACITY ladder: matched-data FMs at increasing width. arm_substrate P1 puts the
        # n=3, v_explore=8 requirement for velocity-dim R² >= 0.99 at hidden = 32, so this axis
        # straddles the point where the FM stops being able to represent the plant at all —
        # a different way to be a bad teacher than being stale.
        Sc, Uc, S2c = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 20)
        # The policy must still be born NAIVE and the drift must still be the thing it adapts
        # to, or the capacity ladder is a different experiment from the transition ladder
        # rather than the same one on a second axis. So the shared initial motor program is
        # cloned from a FULL-capacity PRE-DRIFT FM in both runs, and only the TEACHER's
        # capacity varies along this ladder.
        S0, U0, S20 = sub.pool(env0, cfg["pool_n"], cfg["seed"] + 10)
        policy_net = sub.mlp(cfg["seed"] + 41)
        sub.train_steps(policy_net, torch.optim.Adam(policy_net.parameters(), lr=cfg["fm_lr"]),
                        S0, U0, S20, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
        for hid in cfg["capacity_ladder"]:
            c = dict(cfg); c["fm_hidden"] = hid
            s2 = C.ArmSubstrate(c, sub.device); s2.norm = sub.norm
            net = s2.mlp(cfg["seed"] + 42)
            s2.train_steps(net, torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"]),
                           Sc, Uc, S2c, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
            rungs.append({"name": f"h={hid}", "kind": "capacity", "fm_hidden": hid,
                          "transitions": int(cfg["pool_n"]), "net": net})
            print(f"[capacity] trained matched FM at hidden={hid}", flush=True)
    else:
        # Cut 4c-arm's ladder: pretrain field-free, then re-adapt online at b1 from REWARD-FREE
        # transitions, snapshotting at each milestone. The FM's quality is the axis; the
        # policy's learning data are executed reaches by construction (SPEC housekeeping).
        S0, U0, S20 = sub.pool(env0, cfg["pool_n"], cfg["seed"] + 10)
        fm = sub.mlp(cfg["seed"] + 41)
        optf = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"])
        sub.train_steps(fm, optf, S0, U0, S20, cfg["fm_steps"],
                        np.random.default_rng(cfg["seed"] + 301))
        # A SNAPSHOT, not a reference: `fm` keeps being fine-tuned at every milestone below,
        # so binding the live object here would silently make the "naive" policy a clone of the
        # FULLY re-adapted model.
        policy_net = copy.deepcopy(fm)       # rung 0's model: the naive, pre-drift arm
        rng_buf = np.random.default_rng(cfg["seed"] + 500)
        bufS = np.zeros((0, sub.SD), np.float32); bufU = np.zeros((0, AD), np.float32)
        bufS2 = np.zeros((0, sub.SD), np.float32)
        for k, m in enumerate(cfg["milestones"]):
            need = m - len(bufS)
            if need > 0:
                aS, aU, aS2 = collect_pool(env1, need, rng_buf, sub.fs, sub.qc,
                                           cfg["q_range"], cfg["v_explore"])
                bufS = np.concatenate([bufS, aS]); bufU = np.concatenate([bufU, aU])
                bufS2 = np.concatenate([bufS2, aS2])
            if m > 0:
                sub.train_steps(fm, optf, bufS, bufU, bufS2, cfg["finetune_steps"],
                                np.random.default_rng(cfg["seed"] + 600 + k))
            rungs.append({"name": f"m={m}", "kind": "readapt", "transitions": int(m),
                          "net": copy.deepcopy(fm)})
            print(f"[ladder] snapshot at {m} reward-free transitions", flush=True)
        # the matched ceiling — a fresh b1 FM, i.e. full re-adaptation
        fm_ceil = sub.mlp(cfg["seed"] + 42)
        Sc, Uc, S2c = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 20)
        sub.train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                        Sc, Uc, S2c, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
        rungs.append({"name": "matched_ceiling", "kind": "ceiling", "transitions": -1,
                      "net": fm_ceil})
        print("[ladder] matched ceiling trained", flush=True)

    # ---------------------------------------------------------------- the ONE initial policy
    # Cloned from the FIRST rung's FM — the naive, pre-drift model of the Shadmehr protocol —
    # and shared by every arm at every rung. The FM ladder is the TEACHER's quality; the policy
    # every teacher starts from is held fixed, or the arms would not be comparable.
    pol0, pn0 = sub.clone_policy(policy_net, dir_center=D, dir_halfwidth=W)
    d0, l0, r0 = C.evaluate_policy(sub, pol0, pn0, env1, *ev_band)
    print(f"[policy] the shared initial motor program: train-band dist={d0:.4f} "
          f"lat={l0:+.4f} rad={r0:+.4f} in the curl world", flush=True)

    def final_probes(pol, pn):
        out = {}
        for name, (st, gl, t0) in gen_sets.items():
            d, lat, rad = C.evaluate_policy(sub, pol, pn, env1, st, gl, t0)
            out[name] = d; out[name + "_lat"] = lat; out[name + "_rad"] = rad
        # THE AFTEREFFECT: the same policy, graded back in the FIELD-FREE world it was born in.
        # A program that has learned to pre-compensate a field that is no longer there must
        # mis-reach in the mirror direction; Cut 4c-arm reads this on the FM, this node reads
        # it on the policy the FM taught.
        d, lat, rad = C.evaluate_policy(sub, pol, pn, env0, *ev_band)
        out["ae_dist"] = d; out["ae_lat"] = lat; out["ae_rad"] = rad
        return out

    ae0 = final_probes(pol0, pn0)
    print(f"[policy] initial aftereffect (field-free): dist={ae0['ae_dist']:.4f} "
          f"lat={ae0['ae_lat']:+.4f}", flush=True)

    def run_arm(net, teacher, tcfg_over, label):
        t = dict(tcfg); t.update(tcfg_over)
        pol = sub.build_policy(); pol.load_state_dict(pol0.state_dict())
        curve = C.train_teacher(sub, net, pol, pn0, env1, teacher, t, pool_tr, eval_sets,
                                cfg["seed"], tag=label + " ")
        s = C.summarize_curve(curve, list(eval_sets), t["plateau_evals"])
        s.update(final_probes(pol, pn0))
        return {"curve": curve, "summary": s}

    TEACHER_SPEC = {
        "rbl_cont":     ("rbl", {"rbl_reward": "continuous"}),
        "rbl_bin":      ("rbl", {"rbl_reward": "binary"}),
        "ebl_sensory":  ("ebl_sensory", {}),
        "ebl_imagined": ("ebl_imagined", {}),
    }

    # ---------------------------------------------------------------- the grid
    results = {"config": cfg, "band_stats": results_band,
               "initial_policy": {"train_band": d0, "train_band_lat": l0,
                                  "train_band_rad": r0, **ae0},
               "ladder": [], "dysmetria": []}
    n_rungs = len(rungs)
    beta_rungs = {i % n_rungs for i in cfg["beta_rungs"]}
    dys_rungs = {i % n_rungs for i in cfg["dysmetria_rungs"]}
    rbl_rungs = {i % n_rungs for i in cfg["rbl_rungs"]}
    rbl_cache = {}
    print(f"[grid] {n_rungs} rungs x {len(cfg['teachers'])} teachers; beta sweep at rungs "
          f"{sorted(beta_rungs)}, dysmetria at rungs {sorted(dys_rungs)}", flush=True)
    for ri, rung in enumerate(rungs):
        net = rung["net"]
        row = {"name": rung["name"], "kind": rung["kind"],
               "transitions": rung["transitions"]}
        if "fm_hidden" in rung:
            row["fm_hidden"] = rung["fm_hidden"]
        print(f"\n=== rung {ri + 1}/{n_rungs}: {rung['name']} ===", flush=True)
        row["readings"] = C.fm_readings(sub, net, env1, probe, jac_set, tcfg, pol0, pn0)
        rd = row["readings"]
        print(f"[readings] fm_err={rd['fm_err']:.4f} | one-step cos={rd['jac1']['cos']:.4f} "
              f"sign_w={rd['jac1']['sign_agree_w']:.3f} | endpoint cos={rd['jacE']['cos']:.4f} "
              f"sign_w={rd['jacE']['sign_agree_w']:.3f} vjp_cos={rd['jacE']['vjp_cos']:.4f} "
              f"scale={rd['jacE']['scale']:.3f}", flush=True)

        # the fork's zeroth-order incumbents on the identical FM
        inc = {}
        r_ = np.random.default_rng(cfg["seed"] + 7000)
        inc["reactive"], inc["reactive_lat"], inc["reactive_rad"] = sub.rollout(
            lambda s, g: sub.mpc_plan(net, s, g, r_), 1, env1, *ev_band)
        r_ = np.random.default_rng(cfg["seed"] + 7001)
        inc["ballistic_cem"], inc["ballistic_cem_lat"], inc["ballistic_cem_rad"] = sub.rollout(
            lambda s, g: sub.mpc_plan(net, s, g, r_), H, env1, *ev_band)
        # sampling planning on the TERMINAL endpoint objective — the one `ebl_imagined`
        # descends by gradient, so the two are comparable as planners
        r_ = np.random.default_rng(cfg["seed"] + 7002)
        (inc["ballistic_cem_ep"], inc["ballistic_cem_ep_lat"],
         inc["ballistic_cem_ep_rad"]) = sub.rollout(
            lambda s, g: sub.mpc_plan(net, s, g, r_, terminal_only=True), H, env1, *ev_band)
        if ri in (0, n_rungs - 1):
            polb, pnb = sub.clone_policy(net, seed_off=57, dir_center=D, dir_halfwidth=W,
                                         verbose=False)
            (inc["ballistic_bc"], inc["ballistic_bc_lat"],
             inc["ballistic_bc_rad"]) = C.evaluate_policy(sub, polb, pnb, env1, *ev_band)
        row["incumbents"] = inc
        print("[incumbents] " + "  ".join(f"{k}={v:.4f}" for k, v in inc.items()
                                          if not k.endswith(("_lat", "_rad"))), flush=True)

        arms = {}
        for label in cfg["teachers"]:
            teacher, over = TEACHER_SPEC[label]
            # `rbl` never touches the FM, so with the shared initial policy, the shared reach
            # pool and the shared noise draws its run is BIT-IDENTICAL at every rung. It is
            # executed at `rbl_rungs` (default: the first and the last, so the repeat is an
            # audit of that determinism) and reused in between rather than re-burning the
            # budget on an arm whose answer cannot change with the axis.
            if teacher == "rbl" and ri not in rbl_rungs and label in rbl_cache:
                arms[label] = dict(rbl_cache[label])
                arms[label]["reused_from"] = rbl_cache[label]["_rung"]
                continue
            a = run_arm(net, teacher, over, f"{rung['name']}|")
            if teacher == "rbl":
                a["_rung"] = rung["name"]
                if label not in rbl_cache:
                    rbl_cache[label] = a
                else:
                    prev = rbl_cache[label]["summary"]["train_band_final"]
                    now = a["summary"]["train_band_final"]
                    verdict = ("IDENTICAL" if now == prev else
                               "DIFFERS — the harness is not rung-independent, investigate")
                    print(f"[audit] {label} re-run at {rung['name']}: {now:.6f} vs "
                          f"{prev:.6f} at {rbl_cache[label]['_rung']} ({verdict})", flush=True)
            arms[label] = a
        if ri in beta_rungs:
            for b in cfg["beta_sweep"]:
                for mode in cfg["mix_scale_modes"]:
                    # "auto" = the two action gradients rescaled to commensurate norms before
                    # the β-sum, measured once on the arm's first batch; "1" = Eq. 2's literal
                    # raw sum. On this substrate the norms differ by ~2 orders of magnitude
                    # (phase_a gate 3a), so the raw sum's β is not an interpolating axis and
                    # both readings are needed to say what β did.
                    lbl = f"mixed@{b}" if mode == "auto" else f"mixed_raw@{b}"
                    arms[lbl] = run_arm(net, "mixed",
                                        {"beta": float(b),
                                         "mix_scale": 0.0 if mode == "auto" else float(mode),
                                         "rbl_reward": cfg["mix_reward"]},
                                        f"{rung['name']}|")
        row["arms"] = arms
        results["ladder"].append(row)
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2, cls=NumpyEncoder)
        volume.commit()      # incremental: a swallowed post-cancellation commit loses the run
                             # (plasticity_gain's transferable gotcha)

        # ---------------------------------------------------------- the dysmetria control
        if ri in dys_rungs:
            print(f"\n--- dysmetria control at {rung['name']} ---", flush=True)
            masks = [("healthy", np.ones((2, AD), np.float32))]
            for k in range(2):
                for j in range(AD):
                    M = np.ones((2, AD), np.float32); M[k, j] = -1.0
                    masks.append((f"flip_y{'xy'[k]}_u{j}", M))
            Mall = -np.ones((2, AD), np.float32)
            masks.append(("flip_all", Mall))
            for mname, M in masks:
                a = run_arm(net, "ebl_sensory", {"sign_mask": M.tolist()},
                            f"dys:{mname}|")
                s = a["summary"]
                results["dysmetria"].append(
                    {"rung": rung["name"], "mask": mname, "mask_values": M.tolist(),
                     "summary": s, "curve": a["curve"]})
                print(f"  [dysmetria {mname:14s}] dist={s['train_band_final']:.4f} "
                      f"lat={s.get('train_band_lat_final', float('nan')):+.4f} "
                      f"rad={s.get('train_band_rad_final', float('nan')):+.4f} "
                      f"(initial: dist={d0:.4f} lat={l0:+.4f} rad={r0:+.4f})", flush=True)
            with open(os.path.join(outdir, "results.json"), "w") as fh:
                json.dump(results, fh, indent=2, cls=NumpyEncoder)
            volume.commit()

    # ---------------------------------------------------------------- headline print
    print("\n[headline] final train-band endpoint error per rung x teacher "
          "(lower = better; the shared initial policy sits at "
          f"{d0:.4f}):", flush=True)
    labels = sorted({k for r in results["ladder"] for k in r["arms"]})
    head = f"{'rung':>16s} {'fm_err':>8s} {'jacE_cos':>9s} {'vjp_cos':>8s} " + \
           " ".join(f"{l:>14s}" for l in labels)
    print(head, flush=True)
    for r in results["ladder"]:
        line = (f"{r['name']:>16s} {r['readings']['fm_err']:8.4f} "
                f"{r['readings']['jacE']['cos']:9.4f} {r['readings']['jacE']['vjp_cos']:8.4f} ")
        line += " ".join(f"{r['arms'][l]['summary']['train_band_final']:14.4f}"
                         if l in r["arms"] else f"{'—':>14s}" for l in labels)
        print(line, flush=True)

    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote {outdir}/results.json", flush=True)
    print("[done] jacobian_teacher complete", flush=True)
    return {"results": results}


@app.local_entrypoint()
def jacobian_teacher(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    teachers: str = "rbl_cont,rbl_bin,ebl_sensory,ebl_imagined",
    # The FM-quality axis. Cut 4c-arm's own list is "0,400,1000,2500,6000,14000"; arm_substrate
    # P7 found 96% of the ballistic recovery is complete by the first milestone (400), so this
    # default resolves the early step instead of the flat tail. `phase_a` gates the fork's
    # defaults, which are unchanged in `core.fork_defaults()`.
    milestones: str = "0,100,400,1000,2500,6000,14000",
    capacity_ladder: str = "",         # e.g. "8,16,32,64,256": a SECOND run, not a cross-product
    # Eq. 2's mixing weight, swept where mixing is most informative: the stale rung and the
    # ceiling (`-1` = last rung).
    beta_sweep: str = "0.25,0.5,0.75",
    beta_rungs: str = "0,-1",
    mix_scale_modes: str = "auto,1",   # "auto" rescales g_rbl to g_ebl's norm before the
                                       # β-sum; "1" is Eq. 2's literal raw sum. Both are run.
    mix_reward: str = "continuous",
    dysmetria_rungs: str = "0,-1",
    rbl_rungs: str = "0,-1",           # rbl is FM-independent; run it twice as an audit and
                                       # reuse the result at the rungs in between
    # --- the shared budget and policy noise ---
    reach_budget: int = 20000,
    train_batch: int = 128,
    eval_every: int = 2000,
    eval_schedule: str = "128,256,512,1024,2048,4096,6144,8192,12288,16384,20000",
    sigma: float = 0.15,
    lr_rbl: float = 5e-4,
    lr_ebl: float = 5e-4,
    respect_clip: bool = False,
    reward_radius: float = 0.06,
    train_pool_n: int = 8192,
    # --- geometry / generalization ---
    train_dir: float = 0.0,
    train_dir_halfwidth: float = 30.0,
    gen_offsets: str = "0,30,60,90,150,180",
    gen_halfwidth: float = 15.0,
    held_out_offset: float = 90.0,
    # --- instruments ---
    fd_eps: float = 1e-3,
    jac_n: int = 16,
    jac_probe_n: int = 512,
    plateau_evals: int = 4,
    band_oversample: int = 40,
    # --- substrate overrides (defaults are Cut 4c-arm's, gated by phase_a) ---
    fm_hidden: int = 256,
    n_eval: int = 48,
    pool_n: int = 14000,
    fm_steps: int = 6000,
    k_shoot: int = 1024,
    cem_iters: int = 8,
):
    import os

    from mjc.jacobian_teacher.defaults import fork_defaults, teacher_defaults

    cfg = dict(fork_defaults())
    cfg.update(teacher_defaults())
    cfg.update(
        tag=tag or ("smoke" if quick else "default"), seed=seed,
        teachers=[t for t in teachers.split(",") if t],
        milestones=[int(x) for x in milestones.split(",") if x.strip()],
        capacity_ladder=[int(x) for x in capacity_ladder.split(",") if x.strip()],
        beta_sweep=[float(x) for x in beta_sweep.split(",") if x.strip()],
        beta_rungs=[int(x) for x in beta_rungs.split(",") if x.strip()],
        mix_scale_modes=[m for m in mix_scale_modes.split(",") if m],
        mix_reward=mix_reward,
        dysmetria_rungs=[int(x) for x in dysmetria_rungs.split(",") if x.strip()],
        rbl_rungs=[int(x) for x in rbl_rungs.split(",") if x.strip()],
        reach_budget=reach_budget, train_batch=train_batch, eval_every=eval_every,
        eval_schedule=[int(x) for x in eval_schedule.split(",") if x.strip()],
        sigma=sigma, lr_rbl=lr_rbl, lr_ebl=lr_ebl, respect_clip=respect_clip,
        reward_radius=reward_radius, train_pool_n=train_pool_n,
        train_dir=train_dir, train_dir_halfwidth=train_dir_halfwidth,
        gen_offsets=[float(x) for x in gen_offsets.split(",") if x.strip()],
        gen_halfwidth=gen_halfwidth, held_out_offset=held_out_offset,
        fd_eps=fd_eps, jac_n=jac_n, jac_probe_n=jac_probe_n, plateau_evals=plateau_evals,
        band_oversample=band_oversample,
        fm_hidden=fm_hidden, n_eval=n_eval, pool_n=pool_n, fm_steps=fm_steps,
        k_shoot=k_shoot, cem_iters=cem_iters,
    )
    if held_out_offset not in cfg["gen_offsets"]:
        cfg["gen_offsets"].append(held_out_offset)
    if quick:
        cfg.update(milestones=[0, 400], pool_n=3000, fm_steps=1500, finetune_steps=400,
                   probe_n=800, n_eval=12, k_shoot=256, cem_iters=4, bc_tuples=1500,
                   pol_steps=600, fm_hidden=128, fm_layers=2,
                   reach_budget=768, train_batch=64, eval_every=256, train_pool_n=768,
                   eval_schedule=[64, 128, 256, 512, 768],
                   jac_n=6, jac_probe_n=128, gen_offsets=[0.0, 90.0], held_out_offset=90.0,
                   beta_sweep=[0.5], beta_rungs=[-1], dysmetria_rungs=[-1],
                   rbl_rungs=[0, -1], plateau_evals=2,
                   mix_scale_modes=["auto", "1"])
        print("[quick] NOTE: k_shoot=256/cem_iters=4 is the pusher-tuned budget arm_substrate "
              "P3 shows flattens this substrate's slope — smoke only, never a result.")
    out = run_jacobian_teacher.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "jt_" + cfg["tag"])
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
