"""jacobian_teacher Phase C — a committee of forward models, and whether their AGREEMENT about
the direction of ∂y/∂u predicts where the error-based teacher helps.

[`SPEC.md`](SPEC.md) §Committee (Phase C). Depends on nothing in Phase B's outcome; it asks a
different question of the same machinery. Phase B can only score a Jacobian's direction against
a finite-difference plant oracle, which is an EXPERIMENTER'S instrument — `set_state` is not
available to an agent. A committee's cross-member sign agreement is the same reading obtained
from inside: reward-free, oracle-free, and computable online.

  `ebl_committee`        the mean of the members' vector-Jacobian products. Note this is NOT
                         `ebl_sensory` through the committee MEAN model: the endpoint map
                         composes the one-step model H times, so the Jacobian of the mean is
                         not the mean of the Jacobians. Both are run, and the difference
                         between them is the whole point of the pair.
  `ebl_committee_gated`  the same mean, with each action-space component scaled by the members'
                         sign agreement on it — the committee teaches only where it knows which
                         way to push.
  `mixed_agree`          eq. 2 with β read OFF that agreement, per component, instead of swept.
                         `committee_head/SPEC.md` asks this of ALLOCATION; asked here of CREDIT.

`../curiosity_control/`'s standing warning applies and is why the members are graded, not
trusted: on this substrate, with an aleatoric noise source present, members disagree by fitting
different noise realizations rather than converging — `reducible ≈ surprise` there. Contacts and
joint noise are off here (the arm's default), so that failure mode is out of scope for this run
and stays a scope line, not a result.

Run:
    cd experiments/
    modal run mjc/jacobian_teacher/committee.py::committee --quick                    # smoke
    modal run --detach mjc/jacobian_teacher/committee.py::committee --tag jc_s0 --seed 0
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder


def _safe_corr(a, b):
    import numpy as np

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_committee(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch

    from mjc.arm_env import collect_pool, fk
    from mjc.jacobian_teacher import core as C

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    sub = C.ArmSubstrate(cfg)
    tcfg = {k: cfg[k] for k in C.teacher_defaults()}
    n, H, AD = sub.n, sub.H, sub.AD
    K = cfg["k_ens"]
    env0, env1 = sub.make_env(cfg["b0"]), sub.make_env(cfg["b1"])
    outdir = os.path.join(DATA_DIR, "jacobian_teacher", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    print(f"[setup] device={sub.device} K={K} rpf_beta={cfg['rpf_beta']} "
          f"ladder={cfg['milestones']} teachers={cfg['teachers']} "
          f"budget={tcfg['reach_budget']}", flush=True)

    # ---------------------------------------------------------------- probes (as Phase B)
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

    D, W = cfg["train_dir"], cfg["train_dir_halfwidth"]
    ev_band = sub.eval_geometry(cfg["seed"] + 7, dir_center=D, dir_halfwidth=W)
    gen_sets = {f"gen@{int(o)}": sub.eval_geometry(cfg["seed"] + 700 + int(o),
                                                   dir_center=D + o,
                                                   dir_halfwidth=cfg["gen_halfwidth"])
                for o in cfg["gen_offsets"]}
    held = f"gen@{int(cfg['held_out_offset'])}"
    eval_sets = {"train_band": (*ev_band, env1), "held_out": (*gen_sets[held], env1)}
    jac_set = sub.eval_geometry(cfg["seed"] + 77, B=max(tcfg["jac_n"], 8),
                                dir_center=D, dir_halfwidth=W)
    pool_tr = C.make_reach_pool(sub, cfg["seed"] + 800, cfg["train_pool_n"], D, W)

    # ---------------------------------------------------------------- the committee ladder
    S0, U0, S20 = sub.pool(env0, cfg["pool_n"], cfg["seed"] + 10)
    members = C.build_committee(sub, cfg["seed"] + 41, K, cfg["rpf_beta"])
    C.train_committee(sub, members, S0, U0, S20, cfg["fm_steps"], cfg["seed"] + 301)
    print(f"[pretrain] {K} random-prior members fitted on the FIELD-FREE world", flush=True)

    rungs = []
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
            C.train_committee(sub, members, bufS, bufU, bufS2, cfg["finetune_steps"],
                              cfg["seed"] + 600 + k)
        rungs.append({"name": f"m={m}", "transitions": int(m),
                      "members": [copy.deepcopy(x) for x in members]})
        print(f"[ladder] committee snapshot at {m} reward-free transitions", flush=True)
    ceil = C.build_committee(sub, cfg["seed"] + 42, K, cfg["rpf_beta"])
    Sc, Uc, S2c = sub.pool(env1, cfg["pool_n"], cfg["seed"] + 20)
    C.train_committee(sub, ceil, Sc, Uc, S2c, cfg["fm_steps"], cfg["seed"] + 302)
    rungs.append({"name": "matched_ceiling", "transitions": -1, "members": ceil})
    print("[ladder] matched ceiling committee trained", flush=True)

    pol0, pn0 = sub.clone_policy(C.CommitteeMean(rungs[0]["members"]),
                                 dir_center=D, dir_halfwidth=W)
    d0, l0, r0 = C.evaluate_policy(sub, pol0, pn0, env1, *ev_band)
    print(f"[policy] shared initial motor program (cloned through the PRE-DRIFT committee "
          f"mean): train-band dist={d0:.4f} lat={l0:+.4f} rad={r0:+.4f}", flush=True)

    def final_probes(pol, pn):
        out = {}
        for name, (st, gl, t0) in gen_sets.items():
            d, lat, rad = C.evaluate_policy(sub, pol, pn, env1, st, gl, t0)
            out[name] = d; out[name + "_lat"] = lat; out[name + "_rad"] = rad
        d, lat, rad = C.evaluate_policy(sub, pol, pn, env0, *ev_band)
        out["ae_dist"] = d; out["ae_lat"] = lat; out["ae_rad"] = rad
        return out

    def run_arm(net_or_members, teacher, over, label):
        t = dict(tcfg); t.update(over)
        pol = sub.build_policy(); pol.load_state_dict(pol0.state_dict())
        curve = C.train_teacher(sub, net_or_members, pol, pn0, env1, teacher, t, pool_tr,
                                eval_sets, cfg["seed"], tag=label)
        sm = C.summarize_curve(curve, list(eval_sets), t["plateau_evals"])
        sm.update(final_probes(pol, pn0))
        return {"curve": curve, "summary": sm}

    results = {"config": cfg,
               "initial_policy": {"train_band": d0, "train_band_lat": l0,
                                  "train_band_rad": r0, **final_probes(pol0, pn0)},
               "ladder": []}
    n_rungs = len(rungs)
    rbl_rungs = {i % n_rungs for i in cfg["rbl_rungs"]}
    rbl_cache = {}

    for ri, rung in enumerate(rungs):
        mem = rung["members"]
        mean_net = C.CommitteeMean(mem)
        print(f"\n=== rung {ri + 1}/{n_rungs}: {rung['name']} ===", flush=True)
        row = {"name": rung["name"], "transitions": rung["transitions"]}
        row["readings"] = C.fm_readings(sub, mean_net, env1, probe, jac_set, tcfg, pol0, pn0)

        # THE PHASE-C READING: cross-member sign agreement on the composed endpoint Jacobian,
        # on the sequences the shared initial policy actually emits — computable with no oracle
        # and no reward, unlike everything in `readings`.
        st, gl, t0 = [x[: max(tcfg["jac_n"], 8)] for x in jac_set]
        with torch.no_grad():
            seq = sub.policy_mu(pol0, pn0, st, gl).cpu().numpy().astype(np.float32)
        agree, Jmean, Jall = C.committee_direction_agreement(sub, mem, st, seq)
        Jpl = C.plant_endpoint_jac_fd(sub, env1, st, seq, tcfg["fd_eps"])
        w = np.abs(Jpl)
        row["committee"] = {
            "dir_agree": float(agree.mean()),
            "dir_agree_w": float((agree * w).sum() / (w.sum() + 1e-12)),
            "member_spread": float(Jall.std(0).mean()),
            "mean_jac_vs_oracle": C.direction_metrics(Jmean, Jpl),
            # does agreement track being RIGHT? the one thing an oracle can say about a
            # reading whose whole appeal is that it needs no oracle. Degenerate when the
            # committee is unanimous everywhere (zero variance), hence the guard.
            "agree_vs_correct_corr": _safe_corr(
                agree.ravel(), (np.sign(Jmean) == np.sign(Jpl)).astype(float).ravel()),
            "correct_when_unanimous": float(
                ((np.sign(Jmean) == np.sign(Jpl))[agree >= 1.0 - 1e-9]).mean())
            if (agree >= 1.0 - 1e-9).any() else float("nan"),
            "correct_when_split": float(
                ((np.sign(Jmean) == np.sign(Jpl))[agree < 1.0 - 1e-9]).mean())
            if (agree < 1.0 - 1e-9).any() else float("nan"),
        }
        rc = row["committee"]
        print(f"[readings] fm_err={row['readings']['fm_err']:.4f}  endpoint cos="
              f"{row['readings']['jacE']['cos']:.4f} vjp_cos="
              f"{row['readings']['jacE']['vjp_cos']:.4f}", flush=True)
        print(f"[committee] dir_agree={rc['dir_agree']:.4f} (weighted {rc['dir_agree_w']:.4f}) "
              f"spread={rc['member_spread']:.4g}  mean-Jac vs oracle cos="
              f"{rc['mean_jac_vs_oracle']['cos']:.4f} sign_w="
              f"{rc['mean_jac_vs_oracle']['sign_agree_w']:.3f}  corr(agree, correct)="
              f"{rc['agree_vs_correct_corr']:+.4f}  sign-correct when unanimous/split = "
              f"{rc['correct_when_unanimous']:.3f}/{rc['correct_when_split']:.3f}", flush=True)

        inc = {}
        r_ = np.random.default_rng(cfg["seed"] + 7001)
        inc["ballistic_cem"], inc["ballistic_cem_lat"], inc["ballistic_cem_rad"] = sub.rollout(
            lambda s, g: sub.mpc_plan(mean_net, s, g, r_), H, env1, *ev_band)
        r_ = np.random.default_rng(cfg["seed"] + 7000)
        inc["reactive"], inc["reactive_lat"], inc["reactive_rad"] = sub.rollout(
            lambda s, g: sub.mpc_plan(mean_net, s, g, r_), 1, env1, *ev_band)
        row["incumbents"] = inc
        print(f"[incumbents] reactive={inc['reactive']:.4f} "
              f"ballistic_cem={inc['ballistic_cem']:.4f}", flush=True)

        arms = {}
        for label in cfg["teachers"]:
            if label == "rbl_cont":
                if ri not in rbl_rungs and label in rbl_cache:
                    arms[label] = dict(rbl_cache[label])
                    arms[label]["reused_from"] = rbl_cache[label]["_rung"]
                    continue
                a = run_arm(mean_net, "rbl", {"rbl_reward": "continuous"},
                            f"{rung['name']}|")
                a["_rung"] = rung["name"]
                if label in rbl_cache:
                    prev = rbl_cache[label]["summary"]["train_band_final"]
                    now = a["summary"]["train_band_final"]
                    print(f"[audit] rbl_cont re-run at {rung['name']}: {now:.6f} vs "
                          f"{prev:.6f} "
                          f"({'IDENTICAL' if now == prev else 'DIFFERS'})", flush=True)
                else:
                    rbl_cache[label] = a
                arms[label] = a
            elif label == "ebl_sensory":
                # the incumbent: plan-through-the-mean, i.e. the Jacobian OF the mean model
                arms[label] = run_arm(mean_net, "ebl_sensory", {}, f"{rung['name']}|")
            else:
                arms[label] = run_arm(mem, label, {"mix_scale": cfg["mix_scale"]},
                                      f"{rung['name']}|")
        row["arms"] = arms
        results["ladder"].append(row)
        with open(os.path.join(outdir, "committee.json"), "w") as fh:
            json.dump(results, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ---------------------------------------------------------------- headline
    print(f"\n[headline] Phase C — does committee direction AGREEMENT stand in for the "
          f"oracle's direction reading? (shared initial policy at {d0:.4f})", flush=True)
    labels = sorted({k for r in results["ladder"] for k in r["arms"]})
    print(f"{'rung':>16s} {'fm_err':>8s} {'jacEcos':>8s} {'vjpcos':>7s} {'agree':>7s} "
          f"{'agreeW':>7s} {'corr':>7s} " + " ".join(f"{l:>20s}" for l in labels), flush=True)
    for r in results["ladder"]:
        c_, rd = r["committee"], r["readings"]
        line = (f"{r['name']:>16s} {rd['fm_err']:8.4f} {rd['jacE']['cos']:8.4f} "
                f"{rd['jacE']['vjp_cos']:7.4f} {c_['dir_agree']:7.4f} "
                f"{c_['dir_agree_w']:7.4f} {c_['agree_vs_correct_corr']:+7.4f} ")
        line += " ".join(f"{r['arms'][l]['summary']['train_band_final']:20.4f}"
                         if l in r["arms"] else f"{'—':>20s}" for l in labels)
        print(line, flush=True)

    with open(os.path.join(outdir, "committee.json"), "w") as fh:
        json.dump(results, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote {outdir}/committee.json", flush=True)
    print("[done] committee complete", flush=True)
    return {"results": results}


@app.local_entrypoint()
def committee(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    k_ens: int = 5,
    rpf_beta: float = 0.6,             # curiosity_control's default
    teachers: str = "rbl_cont,ebl_sensory,ebl_committee,ebl_committee_gated,mixed_agree",
    milestones: str = "0,400,2500,14000",
    rbl_rungs: str = "0,-1",
    mix_scale: float = 0.0,            # 0 = auto-measured; `mixed_agree` reads β off agreement
    reach_budget: int = 20000,
    train_batch: int = 128,
    eval_schedule: str = "128,256,512,1024,2048,4096,6144,8192,12288,16384,20000",
    sigma: float = 0.15,
    lr_rbl: float = 5e-4,
    lr_ebl: float = 5e-4,
    train_pool_n: int = 8192,
    train_dir: float = 0.0,
    train_dir_halfwidth: float = 30.0,
    gen_offsets: str = "0,30,60,90,150,180",
    gen_halfwidth: float = 15.0,
    held_out_offset: float = 90.0,
    fd_eps: float = 1e-3,
    jac_n: int = 16,
    jac_probe_n: int = 512,
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
        tag=tag or ("smoke_c" if quick else "committee"), seed=seed,
        k_ens=k_ens, rpf_beta=rpf_beta,
        teachers=[t for t in teachers.split(",") if t],
        milestones=[int(x) for x in milestones.split(",") if x.strip()],
        rbl_rungs=[int(x) for x in rbl_rungs.split(",") if x.strip()],
        mix_scale=mix_scale, reach_budget=reach_budget, train_batch=train_batch,
        eval_schedule=[int(x) for x in eval_schedule.split(",") if x.strip()],
        sigma=sigma, lr_rbl=lr_rbl, lr_ebl=lr_ebl, train_pool_n=train_pool_n,
        train_dir=train_dir, train_dir_halfwidth=train_dir_halfwidth,
        gen_offsets=[float(x) for x in gen_offsets.split(",") if x.strip()],
        gen_halfwidth=gen_halfwidth, held_out_offset=held_out_offset,
        fd_eps=fd_eps, jac_n=jac_n, jac_probe_n=jac_probe_n,
        fm_hidden=fm_hidden, n_eval=n_eval, pool_n=pool_n, fm_steps=fm_steps,
        k_shoot=k_shoot, cem_iters=cem_iters,
    )
    if held_out_offset not in cfg["gen_offsets"]:
        cfg["gen_offsets"].append(held_out_offset)
    if quick:
        cfg.update(k_ens=3, milestones=[0, 400], pool_n=3000, fm_steps=1200,
                   finetune_steps=400, probe_n=800, n_eval=12, k_shoot=256, cem_iters=4,
                   bc_tuples=1500, pol_steps=600, fm_hidden=128, fm_layers=2,
                   reach_budget=768, train_batch=64, train_pool_n=768,
                   eval_schedule=[64, 128, 256, 512, 768], jac_n=6, jac_probe_n=128,
                   gen_offsets=[0.0, 90.0], held_out_offset=90.0, plateau_evals=2)
        print("[quick] smoke only — pusher-tuned CEM budget, never a result.")
    out = run_committee.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "jc_" + cfg["tag"])
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "committee.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote committee.json to {localdir}")
