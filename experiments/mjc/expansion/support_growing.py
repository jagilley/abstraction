"""Cut #5, Piece 3 (REDUCED) — DOF accretion as the DIMENSIONALITY-AXIS CALIBRATION.

**This is a control, not the flagship.** It was originally scoped as the belief's central test
("does a support-growing intervention open the frontier where drift does not?"), and that framing
was wrong on this substrate. Unlocking a joint on an arm gives a HIGHER-DIMENSIONAL function, not
a DEEPER one — there is no level to climb where learning the upper level improves the lower. So
the outcome space was "the frontier grows because we made the problem bigger" (near-tautological)
or "it doesn't" (instrument failure). A weak flagship — but exactly the right calibration, because
for a calibration you WANT a case whose answer you already know.

WHAT GAP IT CLOSES. Piece 1's positive control was a WRONG MODEL (a stale FM with the wrong force
field), which certified that the instrument responds to ERROR. Piece 2 then read a null on the
DIMENSIONALITY axis — `R_res_participation` flat under support-fixed drift. Nothing yet certifies
that the instrument responds to dimensionality at all, so that null has an unclosed gap in it.
This run closes it:

    responds to error          -- Piece 1 (stale FM, 2.7x at k=14, 3/3 seeds)
    responds to added directions -- THIS RUN
    does NOT respond to drift  -- Piece 2 (+0.68 +- 1.00 directions, n.s.)
    => drift != expansion, measured rather than assumed.

THE DESIGN, AND THE TAUTOLOGY IT HAS TO AVOID. If joints are locked, their state dims barely move,
so the PLANT's displacement covariance mechanically gains directions when they unlock — and since
the basis is the plant's, `R_res_participation` would move for reasons having nothing to do with
the model. That is not a test of anything. The fix is the RHM 2x2's fixed-held-out-probe rule,
pushed to its limit:

    the probe is drawn from the FULLY-UNLOCKED arm and executed on the FULLY-UNLOCKED arm,
    once, for every round of every condition.

So the target `A` is literally the same array throughout and `R_act` is constant BY CONSTRUCTION
(asserted below, not hoped for). The learner only ever *trains* on the body it currently has; it
is always *graded* against the full-DOF world. Any movement in the frontier is the model's.

Note the direction this implies, which is the opposite of the naive "novelty refills the frontier"
picture and is the sharper test: graded against the full problem, a model that has experienced
only 2 free joints is IGNORANT of the rest, so its frontier should be HIGH and its dimensionality
LARGE; as joints unlock and it absorbs them, the count of directions still carrying unexplained
computation should FALL. That is a direct test of the metric's stated semantics — *how many of the
model's working directions still carry meaningful unexplained computation* — and it fails
informatively: if absorbing three genuine degrees of freedom does not reduce the count, the metric
is not counting what it claims to count.

THREE CONDITIONS, one protocol, one probe, one shared random init:
  * `locked`    -- 3 distal joints pinned for all rounds. The FLOOR: never learns them.
  * `accretion` -- 3 -> 2 -> 1 -> 0 locked, one released every `rounds_per_stage`. The staircase.
  * `full`      -- 0 locked from the start. The CEILING.
Prediction: `locked` high and flat, `full` low and flat, `accretion` descending from one to the
other with a visible step at each release.

Locking is an `equality/joint` constraint (`arm_env.build_xml`, additive and off by default), NOT
a shorter chain: the state stays 2n-dim and the command n-dim, so a single FM architecture spans
every body and "the input layer changed" is never a confound.

Run:
    cd experiments/
    modal run mjc/on_policy/verify_backcompat.py::verify                      # env gate
    modal run mjc/expansion/support_growing.py::support_growing --quick       # smoke
    bash mjc/expansion/train_piece3.sh                                        # 3 seeds
    python3 mjc/expansion/support_growing_agg.py --tags sg_s0 sg_s1 sg_s2
"""

import copy
import json
import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.expansion.saturation_gate import shadow_law_flex


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_support_growing(cfg: dict) -> dict:
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
    LOCKABLE = list(cfg["lockable"])            # 0-based joint indices, distal-first order
    print(f"[setup] device={device} n_links={n} lockable={LOCKABLE} rounds={cfg['n_rounds']} "
          f"rounds_per_stage={cfg['rounds_per_stage']} k={ks} conds={cfg['conditions']}",
          flush=True)

    def locks_for(cond, rd):
        """How many distal joints are still pinned at round `rd` in `cond`."""
        if cond == "locked":
            return len(LOCKABLE)
        if cond == "full":
            return 0
        return max(0, len(LOCKABLE) - rd // cfg["rounds_per_stage"])

    def make_env(n_locked):
        d = dict(n_links=n, link_lengths=list(Ls), link_masses=Ms,
                 joint_damping=cfg["joint_damping"], gear=cfg["gear"],
                 curl_field={"b": float(cfg["curl_b"])})
        if n_locked > 0:
            # pinned AT THE CENTRE POSTURE, so a locked body is the same arm with fewer
            # degrees of freedom rather than a differently-shaped arm.
            d["locked_joints"] = {int(j): float(qc[j]) for j in LOCKABLE[:n_locked]}
        return ArmEnv(d)

    env_full = make_env(0)

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

    def locked_pool(cenv, n_locked, nn_, seed):
        """Teleport collection that RESPECTS the lock.

        `collect_pool` samples every joint freely, which on a locked body would teleport
        pinned joints off their constraint and let the solver snap them back on the first
        substep — manufacturing transitions the body can never actually produce. So locked
        dims are sampled AT the lock angle with zero velocity instead.
        """
        rng = np.random.default_rng(seed)
        lock_idx = LOCKABLE[:n_locked]
        S = np.empty((nn_, SD), np.float32); U = np.empty((nn_, AD), np.float32)
        S2 = np.empty((nn_, SD), np.float32)
        for i in range(nn_):
            q = qc + rng.uniform(-cfg["q_range"], cfg["q_range"], n)
            qd = rng.normal(0.0, cfg["v_explore"], n)
            for j in lock_idx:
                q[j] = qc[j]; qd[j] = 0.0
            cenv.set_state(q, qd)
            u = rng.uniform(-1, 1, AD).astype(np.float32)
            S[i] = cenv.get_state()
            S2[i], _ = cenv.step(u, fs)
            U[i] = u
        # did the constraint actually hold?
        dev = 0.0
        if lock_idx:
            dev = float(np.abs(S2[:, lock_idx] - qc[lock_idx]).max())
        return S, U, S2, dev

    # Normalization from the FULL arm — the world everything is graded against.
    nS, nU, nS2, _ = locked_pool(env_full, 0, cfg["pool_n"], cfg["seed"] + 11)
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

    # =============================================== the ONE probe, on the FULL arm
    ref_net = _mlp(cfg["seed"] + 40)
    rS, rU, rS2, _ = locked_pool(env_full, 0, cfg["pool_n"], cfg["seed"] + 400)
    train_steps(ref_net, torch.optim.Adam(ref_net.parameters(), lr=cfg["fm_lr"]),
                rS, rU, rS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))

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
            plan[i:j] = mpc_plan(ref_net, s[i:j], goals[i:j], rng)
        for h in range(H):
            u = plan[:, h, :]
            for b in range(Nt):
                env_full.set_state(s[b, :n].astype(np.float64), s[b, n:].astype(np.float64))
                s[b], _ = env_full.step(u[b], fs)
            S_hist.append(s.copy()); U_hist.append(u.copy())
        print(f"[probe] leg {leg}: mean planned reach {mean_reach:.3f} m", flush=True)
    tS, tU = np.stack(S_hist, 1), np.stack(U_hist, 1)
    T = tU.shape[1]
    assert T > k_max, f"probe trajectory {T} must exceed k_max {k_max}"
    ep = rng.integers(0, Nt, cfg["n_probe"])
    off = rng.integers(0, T - k_max + 1, cfg["n_probe"])
    probe_S0 = tS[ep, off]
    probe_U = np.stack([tU[ep, off + j] for j in range(k_max)], 1)
    probe_A = {k: (tS[ep, off + k] - probe_S0).astype(np.float64) for k in ks}
    spd = np.linalg.norm(probe_S0[:, n:], axis=1)
    print(f"[probe] {cfg['n_probe']} windows from {Nt} full-arm reaches (T={T}); "
          f"window-start joint speed mean={spd.mean():.2f} rad/s. "
          f"A is computed ONCE on the full arm, so R_act is constant by construction.",
          flush=True)
    # HOW MUCH OF THE PROBE'S MOTION LIVES IN EACH JOINT. This picks `lockable`, and it has to
    # be picked rather than guessed: a first pass locked the three distal joints, which turned
    # out to carry 86.5% of the probe's angular displacement variance, so the locked model was
    # blind to almost all the motion and sat pinned at the UPPER degeneracy (frontier -> 1,
    # every rho -> 0, nothing to read). A calibration needs the impaired arm IMPAIRED BUT NOT
    # HOPELESS — inside Piece 1's readable window, 0.02 < frontier < 0.90.
    share_by_joint = {}
    for k in ks:
        v = probe_A[k].var(axis=0)[:n]
        frac = v / max(v.sum(), 1e-30)
        share_by_joint[k] = frac.tolist()
        print(f"[probe] k={k}: per-joint share of angular displacement variance = "
              + "  ".join(f"j{i+1}:{100*frac[i]:.1f}%" for i in range(n))
              + f"  | lockable {LOCKABLE} total {100*frac[LOCKABLE].sum():.1f}%", flush=True)

    if cfg.get("probe_only"):
        out = {"config": cfg, "share_by_joint": share_by_joint,
               "probe_start_speed_mean": float(spd.mean())}
        outdir = os.path.join(DATA_DIR, "expansion_support_growing", cfg["tag"])
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()
        print("[probe-only] stopping before the conditions — use the shares above to pick "
              "`--lockable` and `--k-primary`.", flush=True)
        return {"results": out}

    # ------------------------------------------------------------------ measurement
    def measure(net, tag_bits):
        out = []
        for k in ks:
            A = probe_A[k]
            P = fm_rollout(net, probe_S0, probe_U[:, :k, :])
            Rr = A - P
            sd = A.std(axis=0) + 1e-12
            dec = full_decomposition(A / sd, P / sd, ks=tuple(range(1, SD)), seed=cfg["seed"])
            beta, br2, ndir = shadow_law_flex(dec["geometry"]["act_variance_spectrum"],
                                              dec["geometry"]["res_variance_in_act_basis"],
                                              min_dirs=cfg["beta_min_dirs"])
            out.append({**tag_bits, "k": k,
                        "rel_residual": dec["basic"]["relative_residual"],
                        "frontier_mass": dec["repaired"]["frontier_mass"],
                        "R_res_participation": dec["repaired"]["R_res_participation"],
                        "R_act_pr": dec["repaired"]["R_act_pr"],
                        "naive_R_res": dec["naive"]["R_res"],
                        "alignment_index": dec["geometry"]["alignment_index"],
                        "resid_mean_frac": float((Rr.mean(axis=0) ** 2).sum()
                                                 / max((Rr ** 2).sum(axis=1).mean(), 1e-30)),
                        "beta": beta, "beta_r2": br2, "beta_n_dirs": ndir})
        return out

    rows, lock_dev = [], {}
    for cond in cfg["conditions"]:
        # every condition starts from the SAME random init and runs the SAME protocol;
        # only the body it is allowed to move differs.
        fm = _mlp(cfg["seed"] + 41)
        opt = torch.optim.Adam(fm.parameters(), lr=cfg["fm_lr"])
        bufS = np.zeros((0, SD), np.float32); bufU = np.zeros((0, AD), np.float32)
        bufS2 = np.zeros((0, SD), np.float32)
        lock_dev[cond] = []
        for rd in range(cfg["n_rounds"]):
            nl = locks_for(cond, rd)
            env_r = make_env(nl)
            aS, aU, aS2, dev = locked_pool(env_r, nl, cfg["round_n"],
                                           cfg["seed"] + 4000 + 17 * rd)
            lock_dev[cond].append(dev)
            # The buffer ACCUMULATES here, unlike Piece 2's drift. Under accretion old data is
            # still true — it is the same physics restricted to a subspace — so discarding it
            # would manufacture forgetting that the intervention does not imply.
            bufS = np.concatenate([bufS, aS]); bufU = np.concatenate([bufU, aU])
            bufS2 = np.concatenate([bufS2, aS2])
            train_steps(fm, opt, bufS, bufU, bufS2, cfg["steps_per_round"],
                        np.random.default_rng(cfg["seed"] + 600 + rd))

            bits = {"condition": cond, "round": rd, "n_locked": nl,
                    "dof_free": n - nl, "transitions": int(len(bufS)),
                    "lock_deviation": dev}
            rows += measure(fm, bits)
            pk = next(r for r in rows if r["round"] == rd and r["condition"] == cond
                      and r["k"] == cfg["k_primary"])
            print(f"[{cond:9s} rd={rd}] free_dof={n-nl}/{n} buf={len(bufS):6d} "
                  f"lock_dev={dev:.2e}  k={cfg['k_primary']}  "
                  f"R_res_part={pk['R_res_participation']:.2f}/{SD}  "
                  f"frontier={pk['frontier_mass']:.4f}  rel_res={pk['rel_residual']:.4f}  "
                  f"R_act={pk['R_act_pr']:.2f}", flush=True)

    # R_act must be IDENTICAL everywhere — A was computed once. This is an assertion about
    # the design, not a hope about the data: if it fails, the probe was not actually frozen.
    ra = [r["R_act_pr"] for r in rows if r["k"] == cfg["k_primary"]]
    spread = float(max(ra) - min(ra))
    print(f"\n[confound check] R_act spread across ALL rounds/conditions = {spread:.2e} "
          f"(must be ~0 — A is one fixed array)", flush=True)
    assert spread < 1e-6, f"R_act moved ({spread}) — the probe was not frozen"

    kp = cfg["k_primary"]

    def series(cond, key):
        return [r[key] for r in sorted(
            (r for r in rows if r["condition"] == cond and r["k"] == kp),
            key=lambda r: r["round"])]

    summary = {c: {k: series(c, k) for k in
                   ("R_res_participation", "frontier_mass", "rel_residual")}
               for c in cfg["conditions"]}
    print("\n" + "=" * 104, flush=True)
    print(f"PIECE 3 (calibration) — does the frontier read DIMENSIONALITY? k={kp}", flush=True)
    print("  Prediction: `locked` high+flat (never learns the extra DOF), `full` low+flat,", flush=True)
    print("  `accretion` descending from one to the other, stepping at each release.", flush=True)
    print("=" * 104, flush=True)
    for cond in cfg["conditions"]:
        pr = summary[cond]["R_res_participation"]
        fmv = summary[cond]["frontier_mass"]
        print(f"  {cond:9s} R_res_part " + " ".join(f"{x:5.2f}" for x in pr))
        print(f"  {' '*9} frontier   " + " ".join(f"{x:5.3f}" for x in fmv))

    # Every k, not just k_primary: whether a cell is inside Piece 1's readable window
    # (0.02 < frontier < 0.90) is exactly what decides which horizon this is legible at,
    # and the first pass was degenerate at k=14 while possibly fine at k=1.
    print(f"\n  Readable-window check across k (floor 0.02 / ceiling 0.90), last round:")
    for k in ks:
        bits = []
        for cond in cfg["conditions"]:
            r = [x for x in rows if x["condition"] == cond and x["k"] == k]
            r = sorted(r, key=lambda x: x["round"])[-1]
            fmv = r["frontier_mass"]
            flag = "OK " if 0.02 < fmv < 0.90 else "DEG"
            bits.append(f"{cond}={fmv:.3f}[{flag}] Rpart={r['R_res_participation']:.2f}")
        print(f"    k={k:2d}: " + "  ".join(bits), flush=True)
    if {"locked", "full"} <= set(cfg["conditions"]):
        lo = np.mean(summary["full"]["R_res_participation"][-2:])
        hi = np.mean(summary["locked"]["R_res_participation"][-2:])
        print(f"\n  SEPARATION (the calibration): locked {hi:.2f} vs full {lo:.2f} "
              f"=> {hi - lo:+.2f} directions", flush=True)
        print("  If this is ~0, the instrument does not read dimensionality and Piece 2's null\n"
              "  cannot be attributed to the plant.", flush=True)
    print("=" * 104 + "\n", flush=True)

    out = {"config": cfg, "rows": rows, "summary": summary, "lock_deviation": lock_dev,
           "R_act_spread": spread}
    outdir = os.path.join(DATA_DIR, "expansion_support_growing", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        met = [("R_res_participation", "R_res_participation (PRIMARY)"),
               ("frontier_mass", "frontier mass"),
               ("rel_residual", "relative residual")]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        col = {"locked": "#d62728", "accretion": "#1f77b4", "full": "#2ca02c"}
        for j, (key, lab) in enumerate(met):
            ax = axes[j]
            for cond in cfg["conditions"]:
                ax.plot(range(cfg["n_rounds"]), series(cond, key), marker="o", ms=4,
                        color=col.get(cond), label=cond if j == 0 else None)
            for rd in range(cfg["n_rounds"]):
                if "accretion" in cfg["conditions"] and rd > 0 and \
                        locks_for("accretion", rd) != locks_for("accretion", rd - 1):
                    ax.axvline(rd, color="k", lw=0.7, ls=":", alpha=0.5)
            ax.set_xlabel("round (dotted = a joint unlocks)"); ax.set_ylabel(lab)
            ax.set_title(lab, fontsize=9)
            if key == "R_res_participation":
                ax.axhline(SD, color="k", lw=0.8, ls="-.", alpha=0.4)
        axes[0].legend(fontsize=8)
        fig.suptitle("Cut #5 Piece 3 (calibration) — DOF accretion, graded on a FIXED full-DOF "
                     "probe (R_act constant by construction)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(outdir, "fig_support_growing.png"), dpi=150)
        plt.close(fig)
        print("[figure] wrote fig_support_growing.png", flush=True)
    except Exception as e:
        print(f"[figure] skipped: {e}", flush=True)

    volume.commit()
    print(f"[save] wrote results to {outdir}", flush=True)
    return {"results": out}


@app.local_entrypoint()
def support_growing(
    quick: bool = False,
    probe_only: bool = False,
    tag: str = "",
    seed: int = 0,
    conditions: str = "locked,accretion,full",
    n_rounds: int = 8,
    rounds_per_stage: int = 2,
    round_n: int = 2000,
    steps_per_round: int = 2000,
    # CHOSEN FROM THE PROBE, NOT GUESSED. Per-joint share of the probe's angular displacement
    # variance (k=14): j1 1.9%, j2 6.9%, j3 15.2%, j4 25.9%, j5 50.1%. The obvious pick — the
    # three distal joints — carries 91.3%, and a model blind to that much sits pinned at the
    # upper degeneracy (frontier -> 1.0, nothing to read; the first smoke did exactly this).
    # j4,j3,j2 carry 48% together: enough that locking them is a real impairment, while the
    # single biggest mover (j5) stays free so the impaired model is not hopeless. Three
    # lockable joints gives a four-stage staircase. Released distal-first (j4, then j3, then j2).
    lockable: str = "3,2,1",
    # k=8, NOT Piece 1/2's k=14. An endpoint check (locked vs full at full FM quality) found
    # only k=8 has BOTH conditions inside Piece 1's readable window: at k=1 the full-DOF model
    # saturates (frontier 0.014, below the 0.02 floor) and at k=14 the impaired model diverges
    # (0.970, above the 0.90 ceiling). The window is narrower here than in Piece 1/2 because
    # this run deliberately spans a much wider model-quality range — a 2-DOF-experienced model
    # graded on a 5-DOF world sits far worse than anything those pieces contained.
    k_list: str = "1,4,6,8,10,14",
    k_primary: int = 8,
    curl_b: float = 3.0,              # held FIXED — this piece varies DOF, never the operator
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
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    pool_n: int = 14000,
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
    lock = [int(x) for x in lockable.split(",") if x.strip()]
    if quick:
        n_rounds = 4; rounds_per_stage = 1; round_n = 800; steps_per_round = 500
        pool_n = 3000; fm_steps = 1200
        n_probe = 96; n_traj = 48; plan_chunk = 16; k_shoot = 256; cem_iters = 4
        fm_hidden = 128; fm_layers = 2
        ks = [1, 14]
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, probe_only=probe_only, conditions=conds, n_rounds=n_rounds,
        rounds_per_stage=rounds_per_stage, round_n=round_n,
        steps_per_round=steps_per_round, lockable=lock,
        k_list=ks, k_primary=k_primary, curl_b=curl_b,
        n_links=n_links,
        link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip,
        q_range=q_range, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, pool_n=pool_n,
        n_probe=n_probe, n_traj=n_traj, n_legs=n_legs, plan_chunk=plan_chunk,
        plan_H=plan_h, k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, q_jit=q_jit, v0_std=v0_std,
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        reach_tries=reach_tries, beta_min_dirs=beta_min_dirs,
    )
    out = run_support_growing.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "support_growing_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
