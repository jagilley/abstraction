"""Cut 5 stage S0 (ballistic/directed): is the region partition SEPARABLE — i.e. is directed
collection even POSSIBLE on this substrate? The go/no-go before any allocation machinery.

Program: `ideas/two_timescale_value_loop.md`. Parent: `ballistic/README.md` (Cuts 4a/4b/4c —
the EFFERENT bridge: ballistic control transmits FM quality ~3x more than reactive, and
reward-free FM re-adaptation restores ballistic competence ~4.3x more). What that arc leaves
open is the AFFERENT directed-collection link: *value -> where-to-gather -> FM -> behavior*.
Cut 4a tried it with a scalar explore/exploit drive `b` and found the corridor geometry
CONFOUNDED (exploit already covered the control-relevant needle; explore chased aleatoric
noise -- the roles inverted).

    THE STRUCTURAL TENSION this stage resolves. Directed collection requires a LOCAL dynamics
    change (only in-region transitions inform it -- otherwise *where* you collect is irrelevant
    by construction, which is exactly what 4b's smooth/global damping axis guaranteed). But 4b
    also found that a local additive force JET is open-loop INCOMPENSABLE: even a perfect FM
    cannot counteract it feedforward, so ballistic control saturates at "fail" for every FM
    quality and nothing transmits. Local-and-incompensable is a dead end.

    THE UNLOCK: perturb the COMMAND channel, not the force channel. A spatially-localized
    rotation of the command->motion map (`PusherEnv.rot_regions`, added for this cut) is
      * LOCAL       -- phi_j is only observable from transitions taken inside region j;
      * COMPENSABLE -- a correct FM lets the planner pre-rotate its commands and reach fine.
    Biologically it is the canonical cerebellar adaptation paradigm (state-dependent
    visuomotor rotation), and a wrong phi_j is textbook dysmetria.

    THREE PRECONDITIONS, one run. Each can kill the design; all are cheap relative to S1/S2.
      A) COMPENSABILITY.   With a MATCHED FM, is ballistic control in the rotated world as
         accurate as in the clean world? If the rotations break open-loop control outright,
         the axis is 4b's incompensable needle again and the design is dead.
      B) CONTROL-RELEVANCE. Ablate ONE region at a time (an FM correct everywhere except that
         it believes region j is unrotated) and grade. Ballistic control should degrade for
         ON-PATH regions and not for OFF-PATH ones -- i.e. WHICH region is stale matters. If
         ballistic performance is flat over j, there is nothing for an allocator to choose.
      C) LOCALITY / TRANSFER. From an everywhere-stale FM, spend a FIXED small budget of
         reward-free transitions in region j ONLY. Does that fix region j's error and NOT the
         others (a diagonal KxK transfer matrix)? If one region's data fixes another's phi,
         the localization is fake and uniform collection gets the same FM for free.
      C also previews the whole cut: single-region-directed vs UNIFORM collection at an
         IDENTICAL budget. If the best region beats uniform under ballistic control (and not
         under reactive), directed collection has a ballistic-specific payoff -- the claim
         S1 (fixed-arm ladder) and S2 (learned online allocator) then establish properly.

Reuses ballistic_transmission.py's env/FM/CEM/rollout stack (self-contained duplication, the
established pattern). Static regions, no drift, no learned allocator -- that is S1/S2.

Run:
    cd experiments/
    modal run mujoco_control/directed_separability.py::directed_separability --quick   # smoke
    modal run --detach mujoco_control/directed_separability.py::directed_separability \
        --tag sep_s0 --seed 0
"""

import copy
import json
import modal

from mujoco_control.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_directed_separability(cfg: dict) -> dict:
    import os
    import numpy as np
    import torch
    import torch.nn as nn

    from mujoco_control.pusher_env import PusherEnv

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fs = cfg["frame_skip"]; H = cfg["plan_H"]; cr = cfg["corridor_r"]
    REG = cfg["regions"]; K = len(REG)
    print(f"[setup] device={device} K={K} regions="
          + "; ".join(f"{r['name']}@({r['center'][0]:+.2f},{r['center'][1]:+.2f}) phi={r['phi']:+.2f}"
                      for r in REG), flush=True)

    # ===================================================================== #
    # envs: identical damped reach world; they differ ONLY in which region rotations are ON.
    #   env_true  = all K rotations active (the world the controller is graded in)
    #   env_clean = no rotations at all (the pre-drift / "stale FM believes this" world)
    #   env_abl[j]= all rotations EXCEPT j  -> an FM trained here is correct everywhere but
    #               believes region j is unrotated. Because phi_k's gate is ~0 outside region
    #               k, this IS a clean single-region ablation (no data splicing needed).
    # ===================================================================== #
    def make_env(phis):
        regions = [dict(center=tuple(r["center"]), sigma=r["sigma"], phi=float(p))
                   for r, p in zip(REG, phis)]
        return PusherEnv(dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                              joint_damping=cfg["damping"], pusher_r=0.12,
                              rot_regions=regions), with_puck=False)

    phis_true = [r["phi"] for r in REG]
    env_true = make_env(phis_true)
    env_clean = make_env([0.0] * K)
    env_abl = [make_env([0.0 if j == k else phis_true[k] for k in range(K)]) for j in range(K)]

    def gate(pos, j):
        c = REG[j]["center"]; s = REG[j]["sigma"]
        return np.exp(-((pos[..., 0] - c[0]) ** 2 + (pos[..., 1] - c[1]) ** 2) / (2.0 * s ** 2))

    # ===================================================================== #
    # FM f(s,u)->Δs  (transmission-cut stack)
    # ===================================================================== #
    def _mlp(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        h, L = cfg["fm_hidden"], cfg["fm_layers"]
        lyr = [nn.Linear(6, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        net = nn.Sequential(*(lyr + [nn.Linear(h, 4)]))
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    def make_norm(S, U, S2):
        X = np.concatenate([S, U], 1).astype(np.float32); Y = (S2 - S).astype(np.float32)
        return {k: torch.tensor(v, device=device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    huber = nn.HuberLoss(delta=1.0)

    def train_steps(net, opt, S, U, S2, steps, brng):
        X = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Y = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (X - norm["mx"]) / norm["sx"]; Yn = (Y - norm["my"]) / norm["sy"]
        bs = min(cfg["fm_batch"], len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def fresh_fm(S, U, S2, seed_off, steps=None):
        net = _mlp(cfg["seed"] + 40)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"])
        train_steps(net, opt, S, U, S2, steps or cfg["fm_steps"],
                    np.random.default_rng(cfg["seed"] + seed_off))
        return net

    def fm_delta(net, states, cmds):
        with torch.no_grad():
            X = torch.tensor(np.concatenate([states, cmds], 1), device=device, dtype=torch.float32)
            return (net((X - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    # ===================================================================== #
    # collection primitives
    #   collect_box     — uniform over the whole workspace box (the UNIFORM arm)
    #   collect_region  — concentrated inside region j (the DIRECTED arm; the primitive the
    #                     S2 allocator will spend its budget through)
    # ===================================================================== #
    def _roll(cenv, pos, rng):
        n = len(pos)
        S = np.empty((n, 4), np.float32); U = np.empty((n, 2), np.float32); S2 = np.empty((n, 4), np.float32)
        for i in range(n):
            vel = rng.normal(0, cfg["v_explore"], 2).astype(np.float32)
            cenv.set_state(pos[i].astype(np.float64), vel.astype(np.float64))
            u = rng.uniform(-1, 1, 2).astype(np.float32)
            s = cenv.get_state(); s2, _ = cenv.step(u, fs)
            S[i] = s; U[i] = u; S2[i] = s2
        return S, U, S2

    def collect_box(cenv, n, rng):
        pos = np.stack([rng.uniform(-cfg["box_x"], cfg["box_x"], n),
                        rng.uniform(-cfg["box_y"], cfg["box_y"], n)], 1).astype(np.float32)
        return _roll(cenv, pos, rng)

    def collect_region(cenv, j, n, rng):
        c = REG[j]["center"]; s = REG[j]["sigma"] * cfg["collect_sigma_frac"]
        pos = np.stack([rng.normal(c[0], s, n), rng.normal(c[1], s, n)], 1).astype(np.float32)
        return _roll(cenv, pos, rng)

    # shared normalization from a true-world box pool (identical across every FM -> controlled)
    nS, nU, nS2 = collect_box(env_true, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    norm = make_norm(nS, nU, nS2)

    # probes: one held-out set per region (in-region error) + one over the whole box (global)
    probes = []
    for j in range(K):
        pS, pU, pS2 = collect_region(env_true, j, cfg["probe_n"], np.random.default_rng(cfg["seed"] + 100 + j))
        probes.append((pS, pU, (pS2 - pS).astype(np.float32)))
    gS, gU, gS2 = collect_box(env_true, cfg["probe_n"] * 2, np.random.default_rng(cfg["seed"] + 2))
    gT = (gS2 - gS).astype(np.float32)

    def err_region(net, j):
        pS, pU, pT = probes[j]
        return float(np.linalg.norm(fm_delta(net, pS, pU) - pT, axis=1).mean())

    def err_global(net):
        return float(np.linalg.norm(fm_delta(net, gS, gU) - gT, axis=1).mean())

    def err_vec(net):
        return {"global": err_global(net), "per_region": [err_region(net, j) for j in range(K)]}

    # ===================================================================== #
    # CEM planner + rollout (identical to the transmission cut)
    # ===================================================================== #
    Kc, ne_el = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]; B = cfg["n_eval"]

    def mpc_plan(net, states, goals, rng):
        Bn = states.shape[0]
        mu = np.zeros((Bn, H, 2), np.float32); sig = np.full((Bn, H, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, H, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kc, H, 2), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(H):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    d = net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                    s = s + d; cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne_el, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def eval_geometry(seed):
        rng = np.random.default_rng(seed)
        sgn = rng.choice([-1.0, 1.0], B).astype(np.float32)
        spos = np.stack([sgn * cr, np.zeros(B, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
        goals = np.stack([-sgn * cr, np.zeros(B, np.float32)], 1) + rng.uniform(-cfg["goal_jit"], cfg["goal_jit"], (B, 2)).astype(np.float32)
        starts = np.concatenate([spos, rng.normal(0, cfg["v0_std"], (B, 2)).astype(np.float32)], 1)
        return starts, goals

    ev_starts, ev_goals = eval_geometry(cfg["seed"] + 7)

    def rollout(plan_fn, cenv, replan_every, want_visits=False):
        """Execute in `cenv`. Optionally accumulate per-region VISITATION along the executed
        trajectory — the occupancy the S2 allocator weights its collection by, and the
        cut-4a guardrail (publish which regions each behavior actually touches)."""
        states = ev_starts.copy(); plan = None
        visits = np.zeros(K)
        for step in range(H):
            if step % replan_every == 0:
                plan = plan_fn(states, ev_goals)
            acts = plan[:, min(step % replan_every, plan.shape[1] - 1), :]
            if want_visits:
                for j in range(K):
                    visits[j] += float(gate(states[:, :2], j).sum())
            for b in range(B):
                cenv.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                s2, _ = cenv.step(acts[b], fs); states[b] = s2
        dist = float(np.median(np.linalg.norm(states[:, :2] - ev_goals, axis=1)))
        if want_visits:
            tot = visits.sum()
            return dist, (visits / tot if tot > 1e-9 else visits)
        return dist

    def grade(net, cenv=None, want_visits=False):
        cenv = cenv if cenv is not None else env_true
        rec = err_vec(net)
        for cname, re_every, off in (("reactive", 1, 7000), ("ballistic_cem", H, 7001)):
            if cname not in cfg["controllers"]:
                continue
            rng = np.random.default_rng(cfg["seed"] + off)
            out = rollout(lambda s, g: mpc_plan(net, s, g, rng), cenv, re_every, want_visits)
            if want_visits:
                rec[cname], rec[cname + "_visits"] = out
            else:
                rec[cname] = out
        return rec

    def line(prefix, rec):
        msg = f"{prefix} fm_err(glob)={rec['global']:.4f} per-region=[" \
              + " ".join(f"{e:.3f}" for e in rec["per_region"]) + "]"
        for c in cfg["controllers"]:
            if c in rec:
                msg += f"  {c}={rec[c]:.4f}"
        return msg

    out = {"config": cfg, "region_names": [r["name"] for r in REG]}

    # ===================================================================== #
    # A) COMPENSABILITY — is a localized command rotation open-loop compensable at all?
    # ===================================================================== #
    print("\n=== A) compensability ===", flush=True)
    cS, cU, cS2 = collect_box(env_clean, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 10))
    fm_clean = fresh_fm(cS, cU, cS2, 300)                      # matched to the CLEAN world
    tS, tU, tS2 = collect_box(env_true, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 11))
    fm_matched = fresh_fm(tS, tU, tS2, 301)                    # matched to the ROTATED world

    a_clean = grade(fm_clean, cenv=env_clean, want_visits=True)   # ceiling: no rotations at all
    a_matched = grade(fm_matched, cenv=env_true)                 # rotated world, correct FM
    a_stale = grade(fm_clean, cenv=env_true)                     # rotated world, stale-everywhere FM
    print(line("[A clean-world  matched FM]", a_clean), flush=True)
    print(line("[A rotated-world matched FM]", a_matched), flush=True)
    print(line("[A rotated-world STALE   FM]", a_stale), flush=True)
    for c in cfg["controllers"]:
        comp = a_matched[c] - a_clean[c]
        dmg = a_stale[c] - a_matched[c]
        print(f"    {c:14s} compensability-cost={comp:+.4f} (want ~0)   "
              f"staleness-damage={dmg:+.4f} (want >> 0)", flush=True)
    out["A"] = {"clean_world": a_clean, "matched": a_matched, "stale": a_stale}
    print(f"[A] ballistic visitation of regions (clean world): "
          + " ".join(f"{REG[j]['name']}={a_clean.get('ballistic_cem_visits', [0]*K)[j]:.2f}" for j in range(K)),
          flush=True)

    # ===================================================================== #
    # B) CONTROL-RELEVANCE — ablate one region at a time; does WHICH region matter?
    # ===================================================================== #
    print("\n=== B) single-region ablation ===", flush=True)
    B_recs = []
    for j in range(K):
        aS, aU, aS2 = collect_box(env_abl[j], cfg["pool_n"], np.random.default_rng(cfg["seed"] + 200 + j))
        fm_j = fresh_fm(aS, aU, aS2, 310 + j)
        rec = grade(fm_j, cenv=env_true)
        rec["ablated"] = REG[j]["name"]
        B_recs.append(rec)
        print(line(f"[B ablate {REG[j]['name']:>10s}]", rec), flush=True)
    for c in cfg["controllers"]:
        costs = [r[c] - a_matched[c] for r in B_recs]
        print(f"    {c:14s} ablation cost per region: "
              + "  ".join(f"{REG[j]['name']}={costs[j]:+.4f}" for j in range(K))
              + f"   spread={max(costs) - min(costs):+.4f}", flush=True)
    out["B"] = B_recs

    # ===================================================================== #
    # C) LOCALITY / TRANSFER + the allocation preview — spend an IDENTICAL small budget
    #    in region j only, vs spread uniformly. Start from the everywhere-stale FM.
    # ===================================================================== #
    print(f"\n=== C) budgeted collection (budget={cfg['budget']} transitions) ===", flush=True)
    base_state = copy.deepcopy(fm_clean.state_dict())

    # REPLAY. Fine-tuning on the new in-region budget ALONE makes the net smear the newly-seen
    # rotation over the whole space (an early build: global error rose 0.035 -> 0.086 while every
    # region's error fell, i.e. fake "transfer"). The agent does not actually lose what it already
    # knows, and OUTSIDE the regions the clean-world pool IS correct true-world data (the gates
    # are ~0 there), so we replay that alongside the budget. This is readapt's growing-buffer
    # pattern restricted to the still-valid part of the buffer.
    gmax_c = np.max(np.stack([gate(cS[:, :2], j) for j in range(K)]), 0)
    out_idx = np.flatnonzero(gmax_c < 0.02)
    rp = np.random.default_rng(cfg["seed"] + 460).permutation(out_idx)[:cfg["replay_n"]]
    rS, rU, rS2 = cS[rp], cU[rp], cS2[rp]
    print(f"[C] replay pool: {len(rp)} out-of-region transitions (of {len(cS)} clean-pool)", flush=True)

    def finetune_from_stale(S, U, S2, seed_off):
        net = _mlp(cfg["seed"] + 40); net.load_state_dict(base_state)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["finetune_lr"])
        mS = np.concatenate([rS, S]); mU = np.concatenate([rU, U]); mS2 = np.concatenate([rS2, S2])
        train_steps(net, opt, mS, mU, mS2, cfg["finetune_steps"],
                    np.random.default_rng(cfg["seed"] + seed_off))
        return net

    C_recs = []
    for j in range(K):
        bS, bU, bS2 = collect_region(env_true, j, cfg["budget"], np.random.default_rng(cfg["seed"] + 400 + j))
        net = finetune_from_stale(bS, bU, bS2, 500 + j)
        rec = grade(net, cenv=env_true); rec["arm"] = f"region:{REG[j]['name']}"
        C_recs.append(rec); print(line(f"[C collect-in {REG[j]['name']:>10s}]", rec), flush=True)
    uS, uU, uS2 = collect_box(env_true, cfg["budget"], np.random.default_rng(cfg["seed"] + 450))
    net_u = finetune_from_stale(uS, uU, uS2, 550)
    rec_u = grade(net_u, cenv=env_true); rec_u["arm"] = "uniform"
    C_recs.append(rec_u); print(line("[C collect-in    uniform]", rec_u), flush=True)
    out["C"] = C_recs
    out["C_stale_ref"] = a_stale

    # transfer matrix: relative in-region error reduction, arm j -> region k
    stale_pr = np.array(a_stale["per_region"])
    T = np.zeros((K, K))
    for j in range(K):
        after = np.array(C_recs[j]["per_region"])
        T[j] = (stale_pr - after) / np.maximum(stale_pr, 1e-9)
    out["transfer"] = T
    print("\n[C] transfer matrix (rows=collected-in, cols=region; frac of stale error removed)", flush=True)
    print("            " + " ".join(f"{r['name']:>10s}" for r in REG), flush=True)
    for j in range(K):
        print(f"    {REG[j]['name']:>8s} " + " ".join(f"{T[j, k]:>10.2f}" for k in range(K)), flush=True)
    diag = float(np.mean([T[j, j] for j in range(K)]))
    offd = float(np.mean([T[j, k] for j in range(K) for k in range(K) if j != k]))
    out["locality"] = {"diag": diag, "offdiag": offd, "ratio": diag / max(offd, 1e-9)}
    print(f"    locality: diag={diag:.2f}  offdiag={offd:.2f}  ratio={diag / max(offd, 1e-9):.1f}x "
          "(want >> 1 -> region data is genuinely local)", flush=True)

    print("\n[C] directed vs uniform at identical budget (control goal-dist; lower better):", flush=True)
    for c in cfg["controllers"]:
        best_j = int(np.argmin([C_recs[j][c] for j in range(K)]))
        adv = rec_u[c] - C_recs[best_j][c]
        print(f"    {c:14s} stale={a_stale[c]:.4f}  uniform={rec_u[c]:.4f}  "
              f"best-region({REG[best_j]['name']})={C_recs[best_j][c]:.4f}  "
              f"directed-advantage={adv:+.4f}", flush=True)
    print("[C] PREDICTION: directed-advantage LARGE for ballistic, ~0 for reactive.", flush=True)

    figures = _make_figures(out)
    outdir = os.path.join(DATA_DIR, "directed_separability", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    for nm, png in figures.items():
        with open(os.path.join(outdir, nm), "wb") as fh:
            fh.write(png)
    volume.commit()
    print(f"[save] wrote results + {len(figures)} figures to {outdir}", flush=True)
    return {"results": out, "figures": figures}


def _make_figures(R):
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COL = {"reactive": "#7048e8", "ballistic_cem": "#e8590c"}
    ctrls = R["config"]["controllers"]; names = R["region_names"]; K = len(names)
    figs = {}

    def _save(fig):
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); return buf.getvalue()

    # A) compensability
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    conds = [("clean world\nmatched FM", R["A"]["clean_world"]),
             ("rotated world\nmatched FM", R["A"]["matched"]),
             ("rotated world\nSTALE FM", R["A"]["stale"])]
    x = np.arange(len(conds)); w = 0.36
    for i, c in enumerate(ctrls):
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, [cd[1][c] for cd in conds], w,
               color=COL.get(c, "#555"), label=c)
    ax.set_xticks(x); ax.set_xticklabels([cd[0] for cd in conds])
    ax.set_ylabel("control goal-dist (lower=better)")
    ax.set_title("A) is a localized command rotation open-loop COMPENSABLE?\n"
                 "want: bar 2 ~ bar 1 (compensable) and bar 3 >> bar 2 (load-bearing)")
    ax.legend(fontsize=8.5)
    figs["fig1_compensability.png"] = _save(fig)

    # B) per-region ablation cost
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(K); w = 0.36
    for i, c in enumerate(ctrls):
        costs = [r[c] - R["A"]["matched"][c] for r in R["B"]]
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, costs, w, color=COL.get(c, "#555"), label=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("control cost of ablating this region")
    ax.set_title("B) does WHICH region is stale matter?\n"
                 "want: on-path >> off-path for ballistic, flat for reactive")
    ax.legend(fontsize=8.5)
    figs["fig2_ablation.png"] = _save(fig)

    # C) transfer matrix
    T = np.array(R["transfer"])
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(T, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(names)
    for j in range(K):
        for k in range(K):
            ax.text(k, j, f"{T[j, k]:.2f}", ha="center", va="center",
                    color="w" if T[j, k] < 0.6 else "k", fontsize=9)
    ax.set_xlabel("region where error is measured"); ax.set_ylabel("region where budget was spent")
    ax.set_title(f"C) locality of collection\ndiag/offdiag = {R['locality']['ratio']:.1f}x")
    fig.colorbar(im, ax=ax, fraction=0.046, label="frac of stale error removed")
    figs["fig3_transfer.png"] = _save(fig)

    # C) directed vs uniform at equal budget
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    arms = [r["arm"] for r in R["C"]]
    x = np.arange(len(arms)); w = 0.36
    for i, c in enumerate(ctrls):
        ax.bar(x + (i - (len(ctrls) - 1) / 2) * w, [r[c] for r in R["C"]], w,
               color=COL.get(c, "#555"), label=c)
        ax.axhline(R["C_stale_ref"][c], color=COL.get(c, "#555"), ls=":", lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("control goal-dist after budgeted collection")
    ax.set_title("C) allocation preview — identical budget, different WHERE\n"
                 "dotted = stale (no collection); want best-region << uniform for ballistic only")
    ax.legend(fontsize=8.5)
    figs["fig4_allocation_preview.png"] = _save(fig)
    return figs


@app.local_entrypoint()
def directed_separability(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    controllers: str = "reactive,ballistic_cem",
    # regions: "cx,cy,phi;..."  — on-path = |cy| small (the reach runs along y=0)
    # Centers/sigma chosen so the gates are near-DISJOINT: cross-gate weight between any two
    # regions < 0.01 (at sigma=0.28 the original x=+-0.22 pair overlapped at w~0.73 each on the
    # corridor midpoint, which would manufacture a fake off-diagonal in the transfer matrix).
    # phi must DIFFER per region: with a common phi the regions are informationally REDUNDANT
    # (an early build ran all four at +1.0 and the transfer matrix came out flat at 1.1x --
    # off-path data taught the on-path rotation, because "rotate everywhere by +1.0" fits every
    # region at once). Distinct, pairwise-far angles make phi_j observable ONLY inside region j.
    regions: str = "-0.30,0.0,1.2; 0.30,0.0,-1.2; 0.0,0.85,2.6; 0.0,-0.85,-2.6",
    region_sigma: float = 0.18,
    # env
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,                 # the regime where ballistic is competent (cut 4c)
    corridor_r: float = 0.5,
    box_x: float = 0.9,                   # uniform-collection box (must cover ALL regions)
    box_y: float = 1.15,
    collect_sigma_frac: float = 0.8,      # in-region collection spread, in units of region sigma
    # collection / probes
    pool_n: int = 9000,
    probe_n: int = 500,
    v_explore: float = 1.2,
    # FM
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 4000,
    # stage C budgeted collection
    budget: int = 400,
    replay_n: int = 2500,                 # out-of-region base data replayed during fine-tune
    finetune_lr: float = 3e-4,
    finetune_steps: int = 1200,
    # control eval
    n_eval: int = 40,
    goal_jit: float = 0.06,
    v0_std: float = 0.3,
    plan_h: int = 30,
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
):
    import os

    ctrl_list = [c for c in controllers.split(",") if c]
    REG = []
    for i, chunk in enumerate([c for c in regions.split(";") if c.strip()]):
        cx, cy, phi = (float(v) for v in chunk.split(","))
        REG.append(dict(center=(cx, cy), sigma=region_sigma, phi=phi,
                        name=("on-path" if abs(cy) <= 0.3 else "off-path") + f"-{i}"))
    if quick:
        pool_n = 2500; probe_n = 200; fm_steps = 1200; finetune_steps = 400
        fm_hidden = 128; fm_layers = 2; n_eval = 12; k_shoot = 96; budget = 250; replay_n = 1200
        tag = tag or "smoke"
    tag = tag or "default"

    cfg = dict(
        tag=tag, seed=seed, controllers=ctrl_list, regions=REG,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        corridor_r=corridor_r, box_x=box_x, box_y=box_y, collect_sigma_frac=collect_sigma_frac,
        pool_n=pool_n, probe_n=probe_n, v_explore=v_explore,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch, fm_steps=fm_steps,
        budget=budget, replay_n=replay_n, finetune_lr=finetune_lr, finetune_steps=finetune_steps,
        n_eval=n_eval, goal_jit=goal_jit, v0_std=v0_std, plan_H=plan_h,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
    )
    out = run_directed_separability.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "directed_separability_" + tag)
    os.makedirs(localdir, exist_ok=True)
    for nm, png in out["figures"].items():
        with open(os.path.join(localdir, nm), "wb") as fh:
            fh.write(png)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out["results"], fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(out['figures'])} figures + results.json to {localdir}")
