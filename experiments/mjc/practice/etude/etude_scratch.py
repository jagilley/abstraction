"""The étude world: a fixed piece, per-segment benchmarks, and the compilation gate (E-gate).

Program: `ideas/practice_manufactures_its_own_credit.md` §1 (allocation / metering / re-chunking)
and §5 (the two-knob map: tempo × strictness). Parents:
  * `mjc/ballistic/` Cut 4b/4c — reactive vs ballistic_cem vs ballistic_bc, and the finding that
    command ROTATIONS are open-loop-compensable while local force kicks are not (so the hard
    passages here are rotations);
  * `mjc/bridge_assembly/` + `mjc/practice/` — the corrected δ, and the audit that found δ-as-graded-
    plasticity-gain buys ~nothing over tuned uniform lr on this substrate. The role that survives is
    DETECTION (sustained δ-silence = mastered), which is what this node consumes δ for. Plasticity
    inside a drilled segment is plain uniform lr — no per-sample gain anywhere in this file.

WHAT IS NEW HERE: sequence structure. A fixed "piece" — K via-points threaded through the
workspace, one segment passing through a localized command-rotation region (the hard passage), the
rest through clean space. The piece is fixed for the whole run, which is the stable target the
benchmark needs; the segment is the recurring context that makes b_k estimable.

    THE UNITS. Primitive level = per-step reactive control (CEM-MPC re-planning every step).
    Chunk = one segment executed open-loop and evaluated only at its boundary. Phrase = adjacent
    compiled segments fused with the internal re-grounding removed (`fuse_units` exists; E-gate
    does not use it).

    TEMPO = `replan_every` against segment duration. One feedback/replan event costs a fixed
    sensorimotor delay `d_fb`; a control step costs `dt_ctrl = frame_skip * timestep` of world
    time. So the piece's traversal TIME is priced, and fewer corrections per segment = a faster
    piece. At replan_every >= H the segment is forced ballistic (one re-grounding, at the boundary).

    METERING. e_k = the boundary error of segment k on an AT-TEMPO run-through (§5's
    "at-tempo + strict: run-throughs" corner: drill slow, check at tempo). b_k = a per-segment
    tabular EWMA over run-throughs (the estimability run found ewma_ctx >= the net). δ_k = b_k − e_k.
    δ-silence = |δ_k| below a fraction of that segment's realized stale range for W consecutive
    visits with low variance. τ-from-pretrain-MAD is deliberately NOT reused (the audit showed it
    degenerates δ to sign(b−e)).

    THE COMPILE OP. `ballistic_bc` fit on the agent's OWN recent executed traces of the segment
    (self-imitation) — π(s0, g) -> the whole H×2 open-loop command sequence. Routing then switches:
    that segment is executed by the compiled unit in practice, in run-throughs, and at every tempo.
    Frozen after compilation by default (`--recompile-every 0`).

    THE ARMS (the gate is the only thing that differs):
      * `never`       — pure reactive routing; pays the full feedback-time price forever.
      * `delta_gate`  — compile when δ-silence fires on the drilled segment.
      * `sched_early` — compile at a fixed early cycle.
      * `sched_late`  — compile at a fixed late cycle.
      * `sloppy`      — compile at the SAME early cycle as `sched_early`, but the drilled segment is
                        practiced open-loop (replan_every = H) throughout, so the traces it compiles
                        from contain uncorrected error. §5's poisoned region, manufactured.

    GRADING prices TIME-TO-TRAVERSE at tempo, not endpoint accuracy alone: reactive wins raw
    accuracy by construction (Cut 4b), so an accuracy-only metric makes compilation unwinnable by
    design. Reported: (cumulative practice time, waypoint error at performance tempo) as a pair per
    milestone, plus the full tempo ladder and the accuracy-only readout for reference.

Run:
    cd experiments/
    modal run mjc/practice/etude/etude.py::etude --quick                       # smoke
    python3 mjc/practice/etude/launch_detached.py --tag cal_s0 --arms never --panel   # calibration
    python3 mjc/practice/etude/launch_detached.py --tag eg_s0 --seed 0         # the E-gate run
    python3 mjc/practice/etude/analyze_etude.py --tag eg_s0
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# waypoints of the piece: a closed square loop, 4 segments of equal length 0.8 — the reach
# regime bridge_assembly/Cut-4b validated (corridor_r 0.4, plan_h 34, gear 10, damping 2.0).
DEF_WAYPOINTS = "-0.4,-0.4; -0.4,0.4; 0.4,0.4; 0.4,-0.4; -0.4,-0.4"
# hard passages: "<seg>:<phi>:<sigma>" — a command rotation centred on that segment's midpoint
DEF_REGIONS = "1:1.2:0.16"

ARM_TABLE = {
    "never":       dict(trigger="never"),
    "delta_gate":  dict(trigger="delta"),
    "sched_early": dict(trigger="cycle", at="early"),
    "sched_late":  dict(trigger="cycle", at="late"),
    "sloppy":      dict(trigger="cycle", at="early", sloppy=True),
    # --- E-3: the compile op becomes SELECTION + verbatim commitment, not distillation-by-
    # regression. disc2_c15: one individual reactive trace committed verbatim scores 0.0763 across
    # the boundary distribution (better than per-s0 CEM replanning at 0.0807), the MEAN of those
    # same traces scores 0.1596, and a BC fit to them 0.2413. The songbird crystallizes a
    # rendition; it does not average its babble.
    "gate_select":      dict(trigger="delta", op="select"),
    "gate_plan":        dict(trigger="delta", op="plan"),
    "gate_over_select": dict(trigger="delta", op="select", overspeed=True),
    "sloppy_select":    dict(trigger="cycle", at="early", sloppy=True, op="select"),
    "sched_early_select": dict(trigger="cycle", at="early", op="select"),
    # --- E-3b: selection by EXPECTED performance (cross-s0 repeatability), b anchored to the
    # committed unit's OWN post-commit level, and plan-commit from a noise-free hand-over estimate.
    # e3_s0 showed min-of-pool selection is a winner's curse (chosen_err 0.0028 -> realised 0.187,
    # worse than the pool median 0.150 and worse than committing a RANDOM trace at 0.076).
    "gate_select_x":      dict(trigger="delta", op="select_x"),
    "gate_plan_eval":     dict(trigger="delta", op="plan_eval"),
    "gate_over_select_x": dict(trigger="delta", op="select_x", overspeed=True),
    "sloppy_select_x":    dict(trigger="cycle", at="early", sloppy=True, op="select_x"),
    # --- E-4: SEQUENTIAL PHRASE ASSEMBLY. Commit segments in piece order; a segment is eligible
    # only once every segment before it is committed. Candidates are scored on hand-over states
    # produced by the CURRENT PERFORMANCE CONFIGURATION (committed upstream executed at tempo),
    # which is the online, ecologically honest form of the seam-matched selection fix. e3b measured
    # the shift it repairs: the hand-over into the drilled segment sits 0.0078 from the waypoint at
    # speed 0.24 under reactive upstream, and 0.1065 at speed 0.71 under at-tempo upstream — a
    # 4.5-sigma offset, and the whole of the 3x optimism gap.
    "seq_seam":     dict(trigger="sequential", op="select_x", score="seam"),
    "seq_practice": dict(trigger="sequential", op="select_x", score="reactive_upstream"),
    "seam_drill":   dict(trigger="sequential", op="select_x", score="seam", straddle=True),
    # E-5: the straddle traces get their OWN buffer, so they ADD candidates instead of displacing
    # full-piece traces from the fixed-length selection window. In e4 `seam_drill`'s pool at commit
    # was 192 candidates spanning 4 cycles against `seq_seam`'s 256 spanning 8 — more seam
    # experience bought with a shorter memory, which confounded the arm.
    "seam_drill_fix": dict(trigger="sequential", op="select_x", score="seam", straddle=True,
                           straddle_sep=True),
}


def parse_waypoints(s: str):
    return [[float(v) for v in p.split(",")] for p in s.split(";") if p.strip()]


def parse_regions(s: str, wps):
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        seg, phi, sig = p.split(":")
        k = int(seg)
        c = [0.5 * (wps[k][0] + wps[k + 1][0]), 0.5 * (wps[k][1] + wps[k + 1][1])]
        out.append(dict(seg=k, center=c, sigma=float(sig), phi=float(phi)))
    return out


@app.function(gpu="L4", memory=32768, timeout=14400, volumes={DATA_DIR: volume})
def run_etude(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch
    import torch.nn as nn

    from mjc.pusher_env import PusherEnv

    import time as _time
    _WALL0 = _time.perf_counter()
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FAST = bool(cfg.get("fast", False))
    if FAST and device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    _gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 77)

    fs = cfg["frame_skip"]
    H = cfg["seg_H"]
    wps = np.array(cfg["waypoints"], np.float32)
    K = len(wps) - 1
    arm = cfg["arm"]
    spec = ARM_TABLE[arm]
    drill = cfg["drill_seg"]
    dt_ctrl = fs * 0.002
    d_fb = cfg["d_fb"]
    print(f"[setup] arm={arm} device={device} K={K} H={H} regions={cfg['regions']} "
          f"drill={drill} cycles={cfg['n_cycles']} B={cfg['batch']}", flush=True)

    # ===================================================================== #
    # 0) ENV. Clean (pretrain / ceiling reference) vs rotated (the live world).
    # ===================================================================== #
    def make_env(rot_on):
        dgp = dict(arena_half=cfg["arena_half"], gear=cfg["gear"],
                   joint_damping=cfg["damping"], pusher_r=0.12)
        if rot_on:
            dgp["rot_regions"] = [dict(center=r["center"], sigma=r["sigma"], phi=r["phi"])
                                  for r in cfg["regions"]]
        return PusherEnv(dgp, with_puck=False)

    env_clean = make_env(False)      # what the FM was pretrained on -> stale in the regions
    env = make_env(True)             # the world the agent actually acts in

    # ===================================================================== #
    # 1) FORWARD MODEL f(s,u) -> Δs   (single deterministic MLP, as in Cut 4b)
    # ===================================================================== #
    def _mlp(seed, din, dout, h, L, out_act=None):
        g = torch.Generator(device="cpu").manual_seed(seed)
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, dout)]
        if out_act is not None:
            lyr += [out_act]
        net = nn.Sequential(*lyr)
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g); nn.init.zeros_(m.bias)
        return net.to(device)

    huber = nn.HuberLoss(delta=1.0)

    rc = np.array([r["center"] for r in cfg["regions"]], np.float64) if cfg["regions"] else None
    rs = np.array([r["sigma"] for r in cfg["regions"]], np.float64) if cfg["regions"] else None

    def region_w(pos):
        if rc is None:
            return 0.0
        return float(np.exp(-((pos[0] - rc[:, 0]) ** 2 + (pos[1] - rc[:, 1]) ** 2)
                            / (2.0 * rs ** 2)).max())

    def collect_box(cenv, n, rng, reject_w=0.0):
        """Teleport collection over a box covering the whole piece (+ margin).

        `reject_w > 0` rejects samples whose region gate exceeds it, so a pool can be collected
        in the TRUE world while leaving the hard passage unmodelled. That makes staleness "a
        passage you have never modelled" rather than "a passage you have modelled wrongly" — the
        difference matters because the pool is also the replay pool, and a wrongly-modelled
        passage in replay actively fights the adaptation practice is supposed to drive."""
        half = cfg["box_half"]
        S = np.empty((n, 4), np.float32); U = np.empty((n, 2), np.float32); S2 = np.empty((n, 4), np.float32)
        i = 0
        while i < n:
            pos = rng.uniform(-half, half, 2)
            if reject_w > 0.0 and region_w(pos) > reject_w:
                continue
            vel = rng.normal(0, cfg["v_explore"], 2)
            cenv.set_state(pos, vel)
            u = rng.uniform(-1, 1, 2).astype(np.float32)
            S[i] = cenv.get_state(); S2[i], _ = cenv.step(u, fs); U[i] = u
            i += 1
        return S, U, S2

    if cfg["pretrain_mode"] == "exclude":
        normS, normU, normS2 = collect_box(env, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1),
                                           reject_w=cfg["pretrain_exclude_w"])
    else:                       # "clean" == the cal_s0 configuration, kept reproducible
        normS, normU, normS2 = collect_box(env_clean, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 1))
    X = np.concatenate([normS, normU], 1).astype(np.float32); Y = (normS2 - normS).astype(np.float32)
    norm = {k: torch.tensor(v, device=device) for k, v in dict(
        mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    def train_steps(net, opt, S, U, S2, steps, brng, bs_cap=512):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        Xn = (Xt - norm["mx"]) / norm["sx"]; Yn = (Yt - norm["my"]) / norm["sy"]
        bs = min(bs_cap, len(S)); net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=device)
            opt.zero_grad(); huber(net(Xn[idx]), Yn[idx]).backward(); opt.step()
        net.eval()

    def _tensors(S, U, S2):
        Xt = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
        Yt = torch.tensor((S2 - S).astype(np.float32), device=device)
        return (Xt - norm["mx"]) / norm["sx"], (Yt - norm["my"]) / norm["sy"]

    def train_online(net, opt, S, U, S2, RX, RY, steps, brng, bs, rfrac):
        """Online FM plasticity: plain uniform lr, each batch part fresh practice experience and
        part replay of the pretrain pool. Restores bridge_assembly's `n_replay`, whose absence in
        cal_s0 let a narrow on-policy diet destroy the pretrained map (fm_clean 0.006 -> 0.067)."""
        PX, PY = _tensors(S, U, S2)
        nb = max(1, int(round(bs * (1.0 - rfrac)))); nr = max(0, bs - nb)
        net.train()
        for _ in range(steps):
            i = torch.tensor(brng.integers(0, len(PX), size=nb), device=device)
            xs, ys = [PX[i]], [PY[i]]
            if nr:
                j = torch.tensor(brng.integers(0, len(RX), size=nr), device=device)
                xs.append(RX[j]); ys.append(RY[j])
            opt.zero_grad()
            huber(net(torch.cat(xs)), torch.cat(ys)).backward(); opt.step()
        net.eval()

    def fm_delta(net, S, U):
        with torch.no_grad():
            Xq = torch.tensor(np.concatenate([S, U], 1), device=device, dtype=torch.float32)
            return (net((Xq - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]).cpu().numpy()

    fm0 = _mlp(cfg["seed"] + 40, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    opt0 = torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"])
    train_steps(fm0, opt0, normS, normU, normS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 300))

    # ceiling FM: trained on the ROTATED world (what a fully re-adapted model looks like)
    cS, cU, cS2 = collect_box(env, cfg["pool_n"], np.random.default_rng(cfg["seed"] + 2))
    fm_ceil = _mlp(cfg["seed"] + 41, 6, 4, cfg["fm_hidden"], cfg["fm_layers"])
    optc = torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"])
    train_steps(fm_ceil, optc, cS, cU, cS2, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))

    # FM probes: in-region vs clean, both on the TRUE (rotated) dynamics
    def region_probe(n, rng, inside):
        S = np.empty((n, 4), np.float32); U = np.empty((n, 2), np.float32); S2 = np.empty((n, 4), np.float32)
        rc = np.array([r["center"] for r in cfg["regions"]], np.float64)
        rs = np.array([r["sigma"] for r in cfg["regions"]], np.float64)
        i = 0
        while i < n:
            pos = rng.uniform(-cfg["box_half"], cfg["box_half"], 2)
            w = np.exp(-((pos[None, 0] - rc[:, 0]) ** 2 + (pos[None, 1] - rc[:, 1]) ** 2)
                       / (2.0 * rs ** 2)).max() if len(rc) else 0.0
            if (w > 0.5) != inside:
                continue
            vel = rng.normal(0, cfg["v_explore"], 2)
            env.set_state(pos, vel)
            u = rng.uniform(-1, 1, 2).astype(np.float32)
            S[i] = env.get_state(); S2[i], _ = env.step(u, fs); U[i] = u
            i += 1
        return S, U, (S2 - S).astype(np.float32)

    prS_in, prU_in, prT_in = region_probe(cfg["probe_n"], np.random.default_rng(cfg["seed"] + 3), True)
    prS_out, prU_out, prT_out = region_probe(cfg["probe_n"], np.random.default_rng(cfg["seed"] + 4), False)

    def fm_probes(net):
        return dict(
            region=float(np.linalg.norm(fm_delta(net, prS_in, prU_in) - prT_in, axis=1).mean()),
            clean=float(np.linalg.norm(fm_delta(net, prS_out, prU_out) - prT_out, axis=1).mean()))

    print(f"[fm] stale {fm_probes(fm0)}   ceiling {fm_probes(fm_ceil)}", flush=True)

    # ===================================================================== #
    # 2) CONTROLLERS. CEM-MPC over the FM (primitive/reactive + ballistic), and the
    #    compiled open-loop motor program (ballistic_bc) as the chunk.
    # ===================================================================== #
    Kc, ne = cfg["k_shoot"], cfg["cem_elite"]; vel_pen = cfg["vel_pen"]

    def mpc_plan(net, states, goals, rng, hh):
        Bn = states.shape[0]
        mu = np.zeros((Bn, hh, 2), np.float32)
        sig = np.full((Bn, hh, 2), cfg["cem_init_sigma"], np.float32)
        g_t = torch.tensor(goals, device=device).repeat_interleave(Kc, 0)
        s0 = torch.tensor(states, device=device).repeat_interleave(Kc, 0)
        if FAST:
            # identical algorithm, but sampling + elite selection stay on the device: no float64
            # numpy gaussians, no H2D copy, no per-iteration D2H sync.
            mu_t = torch.zeros((Bn, hh, 2), device=device)
            sig_t = torch.full((Bn, hh, 2), cfg["cem_init_sigma"], device=device)
            with torch.no_grad():
                for _ in range(cfg["cem_iters"]):
                    e = torch.randn((Bn, Kc, hh, 2), device=device, generator=_gen)
                    seqs_t = (mu_t[:, None] + sig_t[:, None] * e).clamp_(-1, 1)
                    s = s0.clone()
                    flat = seqs_t.reshape(Bn * Kc, hh, 2)
                    cost = torch.zeros(Bn * Kc, device=device)
                    for h in range(hh):
                        x = torch.cat([s, flat[:, h, :]], 1)
                        d = net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                        s = s + d; cost = cost + (s[:, :2] - g_t).norm(dim=1)
                    cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices
                    elite = torch.gather(seqs_t, 1, idx[:, :, None, None].expand(-1, -1, hh, 2))
                    mu_t = elite.mean(1); sig_t = elite.std(1) + 1e-3
            return mu_t.cpu().numpy().astype(np.float32)
        for _ in range(cfg["cem_iters"]):
            e = rng.standard_normal((Bn, Kc, hh, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
            with torch.no_grad():
                s = s0.clone(); seqs_t = torch.tensor(seqs.reshape(Bn * Kc, hh, 2), device=device)
                cost = torch.zeros(Bn * Kc, device=device)
                for h in range(hh):
                    x = torch.cat([s, seqs_t[:, h, :]], 1)
                    d = net((x - norm["mx"]) / norm["sx"]) * norm["sy"] + norm["my"]
                    s = s + d; cost = cost + (s[:, :2] - g_t).norm(dim=1)
                cost = cost + vel_pen * s[:, 2:].norm(dim=1)
                idx = torch.topk(-cost.reshape(Bn, Kc), ne, dim=1).indices.cpu().numpy()
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1); sig = elite.std(1) + 1e-3
        return mu.astype(np.float32)

    def policy_plan(unit, states, goals):
        pol, pn = unit["pol"], unit["pnorm"]
        hh = unit["hh"]
        with torch.no_grad():
            x = (torch.tensor(np.concatenate([states, goals], 1), device=device) - pn["mu"]) / pn["sd"]
            return pol(x).cpu().numpy().astype(np.float32).reshape(len(states), hh, 2)

    def committed(u):
        return u["pol"] is not None or u.get("fixed") is not None

    def fresh_routing():
        return [dict(span=(k, k), hh=H, pol=None, pnorm=None, fixed=None) for k in range(K)]

    def fuse_units(routing, i, j):
        """Phrase formation: replace units i..j with one committed unit spanning them (internal
        re-grounding removed). Not used by E-gate; present so the wrapper supports the next level."""
        span = (routing[i]["span"][0], routing[j]["span"][1])
        hh = sum(u["hh"] for u in routing[i:j + 1])
        return routing[:i] + [dict(span=span, hh=hh, pol=None, pnorm=None, fixed=None)] + routing[j + 1:]

    # ===================================================================== #
    # 3) PIECE TRAVERSAL. Segments run back-to-back from the TRUE achieved state
    #    (boundaries are re-grounding points, not teleports).
    # ===================================================================== #
    def start_states(rng, n):
        p = wps[0][None, :] + rng.uniform(-cfg["start_jit"], cfg["start_jit"], (n, 2)).astype(np.float32)
        v = rng.normal(0, cfg["v0_std"], (n, 2)).astype(np.float32)
        return np.concatenate([p, v], 1).astype(np.float32)

    def traverse(net, routing, replan_every, starts, rng, collect=False, seg_R=None, explore=0.0,
                 probe_unit=None, ignore_committed=False):
        """Execute the whole piece once (batched over performers). Returns per-unit boundary
        errors, the traversal time under the delay model, and (optionally) the transition stream
        and per-unit (s0, action-sequence) traces for self-imitation."""
        Bn = starts.shape[0]
        states = starts.copy()
        n_reg = 0
        unit_s0 = {}
        wp_err, n_fb, traces = [], [], {}
        trS, trU, trS2 = [], [], []
        pbS, pbU, pbS2 = [], [], []
        for ui, unit in enumerate(routing):
            hh = unit["hh"]
            goal = np.tile(wps[unit["span"][1] + 1][None, :], (Bn, 1)).astype(np.float32)
            s0 = states.copy()
            unit_s0[ui] = s0
            acts = np.empty((Bn, hh, 2), np.float32)
            acts_raw = np.empty((Bn, hh, 2), np.float32)
            if unit.get("fixed") is not None and not ignore_committed:
                plan = np.tile(unit["fixed"][None], (Bn, 1, 1)); re = hh; fb = 1
            elif unit["pol"] is not None and not ignore_committed:
                plan = policy_plan(unit, states, goal); re = hh; fb = 1
            else:
                R = replan_every if seg_R is None else seg_R.get(ui, replan_every)
                re = hh if (R <= 0 or R >= hh) else R
                plan = None; fb = int(np.ceil(hh / re))
            for h in range(hh):
                if ignore_committed or (unit["pol"] is None and unit.get("fixed") is None):
                    if h % re == 0:
                        plan = mpc_plan(net, states, goal, rng, hh)
                    a = plan[:, h % re, :]          # 4b's idiom: re-plan full-horizon, use the head
                else:
                    a = plan[:, h, :]               # the compiled unit: one command sequence, committed
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                acts_raw[:, h, :] = a          # pre-noise command = the controller's own target
                if explore > 0.0:
                    # motor variability: the ingredient that makes a COMMAND-coupled perturbation
                    # identifiable from on-policy data at all. Practice only — run-throughs and
                    # the ladder probes are noise-free (performance, not practice).
                    a = np.clip(a + explore * rng.standard_normal(a.shape).astype(np.float32),
                                -1.0, 1.0).astype(np.float32)
                acts[:, h, :] = a
                nxt = np.empty_like(states)
                for b in range(Bn):
                    env.set_state(states[b, :2].astype(np.float64), states[b, 2:].astype(np.float64))
                    s2, _ = env.step(a[b], fs); nxt[b] = s2
                if collect:
                    trS.append(states.copy()); trU.append(a.copy()); trS2.append(nxt.copy())
                    n_reg += int(sum(region_w(states[b, :2]) > 0.3 for b in range(Bn)))
                if probe_unit is not None and ui == probe_unit:
                    pbS.append(states.copy()); pbU.append(a.copy()); pbS2.append(nxt.copy())
                states = nxt
            wp_err.append(np.linalg.norm(states[:, :2] - goal, axis=1))
            n_fb.append(fb)
            if collect:
                traces[ui] = [np.concatenate([s0, goal], 1).astype(np.float32),
                              acts.copy(), acts_raw.copy(), None]
        for ui in traces:
            traces[ui][3] = wp_err[ui].copy()
        total_steps = sum(u["hh"] for u in routing)
        t_piece = total_steps * dt_ctrl + float(np.sum(n_fb)) * d_fb
        out = dict(wp_err=np.stack(wp_err, 1), n_fb=n_fb, t_piece=t_piece, s0=unit_s0,
                   final=np.linalg.norm(states[:, :2] - wps[-1][None, :], axis=1))
        if collect:
            out["trans"] = (np.concatenate(trS, 0), np.concatenate(trU, 0), np.concatenate(trS2, 0))
            out["traces"] = traces; out["n_in_region"] = n_reg
        if probe_unit is not None:
            out["probe_trans"] = (np.concatenate(pbS, 0), np.concatenate(pbU, 0),
                                  np.concatenate(pbS2, 0))
        return out

    # ===================================================================== #
    # 4) THE COMPILE OP: behaviour-clone the agent's own recent traces of a unit.
    # ===================================================================== #
    def replay_seq(s0s, cmds, goal):
        """Execute one command sequence open-loop from each of `s0s`; return the boundary errors."""
        errs = np.empty(len(s0s), np.float32)
        for i in range(len(s0s)):
            env.set_state(s0s[i, :2].astype(np.float64), s0s[i, 2:].astype(np.float64))
            for h in range(cmds.shape[0]):
                env.step(cmds[h], fs)
            errs[i] = np.linalg.norm(env.get_state()[:2] - goal)
        return errs

    def select_x_unit(unit, buf, goal, rng, S0_ext=None):
        """SELECT BY EXPECTED PERFORMANCE. Draw `n_cand` candidate realisations uniformly from the
        pool (uniform, not top-of-pool: ranking by a candidate's OWN realised error is a winner's
        curse — it favours the sequence most finely tuned to its own start state, i.e. the least
        transferable one). Score each by cross-s0 replay over a held-out sample of recent hand-over
        states — the repeatability criterion — and commit the argmin of the MEAN."""
        A = np.concatenate([b[1] for b in buf], 0)
        E = np.concatenate([b[3] for b in buf], 0)
        X = np.concatenate([b[0] for b in buf], 0)
        n = len(A)
        nc = min(cfg["n_cand"], n); ns = min(cfg["n_score"], n)
        cand = rng.permutation(n)[:nc]
        if S0_ext is not None:
            S0 = S0_ext[rng.permutation(len(S0_ext))[:ns]]; score_ix = None
        else:
            score_ix = rng.permutation(n)[:ns]; S0 = X[score_ix, :4]
        mean_err = np.empty(nc, np.float32)
        for j, ci in enumerate(cand):
            keep = slice(None) if score_ix is None else (score_ix != ci)
            mean_err[j] = float(replay_seq(S0[keep], A[ci], goal).mean())
        w = int(np.argmin(mean_err))
        unit["fixed"] = A[cand[w]].astype(np.float32)
        # priced: every scoring rollout is a real committed traversal of the segment
        t_score = float(nc * ns) * (unit["hh"] * dt_ctrl + d_fb)
        return dict(n_pool=int(n), n_cand=int(nc), n_score=int(ns),
                    chosen_own_err=float(E[cand[w]]), chosen_score=float(mean_err[w]),
                    score_med=float(np.median(mean_err)), score_max=float(mean_err.max()),
                    own_err_of_best_scorer=float(E[cand[w]]),
                    score_of_best_own_err=float(mean_err[int(np.argmin(E[cand]))]),
                    t_score=t_score)

    def plan_eval_unit(unit, s0_eval, goal, net, prng):
        """COMMIT ONE PLAN from a NOISE-FREE hand-over estimate (the run-through's own hand-over
        states) rather than the practice-boundary centroid, which carries upstream motor noise.
        e3_s0's `gate_plan` planned from the practice centroid and paid 2.9x vs disc2's probe (a)."""
        c = s0_eval.mean(0)
        i = int(np.argmin(np.linalg.norm(s0_eval - c, axis=1)))
        g = np.tile(np.asarray(goal, np.float32)[None, :], (1, 1))
        pl = mpc_plan(net, s0_eval[i:i + 1].astype(np.float32), g, prng, unit["hh"])
        unit["fixed"] = pl[0].astype(np.float32)
        return dict(n_pool=int(len(s0_eval)), s0_spread=float(np.linalg.norm(np.std(s0_eval[:, :2], 0))))

    def select_unit(unit, buf, k_best=1):
        """SELECT + COMMIT: take the recent realisation with the lowest MEASURED boundary error and
        commit its command sequence verbatim. No averaging, no regression."""
        A = np.concatenate([b[1] for b in buf], 0)
        E = np.concatenate([b[3] for b in buf], 0)
        idx = int(np.argsort(E)[:k_best][0])
        unit["fixed"] = A[idx].astype(np.float32)
        return dict(n_pool=int(len(E)), chosen_err=float(E[idx]),
                    pool_med=float(np.median(E)), pool_min=float(E.min()))

    def plan_unit(unit, buf, net, prng):
        """COMMIT ONE PLAN: a single full-horizon CEM plan from the centroid of the recent
        hand-over distribution, committed verbatim for every episode (disc2 probe (a))."""
        X = np.concatenate([b[0] for b in buf], 0)
        c = X[:, :4].mean(0); i = int(np.argmin(np.linalg.norm(X[:, :4] - c, axis=1)))
        pl = mpc_plan(net, X[i:i + 1, :4], X[i:i + 1, 4:6], prng, unit["hh"])
        unit["fixed"] = pl[0].astype(np.float32)
        return dict(n_pool=int(len(X)), s0_centroid=c.tolist())

    def compile_unit(unit, buf, brng, target="exec", net=None, prng=None):
        """target: 'exec' = the agent's own EXECUTED commands (exploration noise included — what
        eg_s0 cloned); 'clean' = the same traces' pre-noise controller targets; 'cem' = full-horizon
        ballistic-CEM-on-the-current-FM plans from the same start states (Cut 4b's BC source)."""
        hh = unit["hh"]
        Xc = np.concatenate([b[0] for b in buf], 0)
        if target == "cem":
            plans = np.empty((len(Xc), hh, 2), np.float32)
            for i in range(0, len(Xc), 256):
                j = min(i + 256, len(Xc))
                plans[i:j] = mpc_plan(net, Xc[i:j, :4], Xc[i:j, 4:6], prng, hh)
            Yc = plans.reshape(len(Xc), hh * 2)
        else:
            src = 1 if target == "exec" else 2
            Yc = np.concatenate([b[src].reshape(len(b[src]), hh * 2) for b in buf], 0)
        pol = _mlp(cfg["seed"] + 900 + unit["span"][0], 6, 2 * hh,
                   cfg["pol_hidden"], cfg["pol_layers"], out_act=nn.Tanh())
        pn = {"mu": torch.tensor(Xc.mean(0), device=device),
              "sd": torch.tensor(Xc.std(0) + 1e-6, device=device)}
        Xt = (torch.tensor(Xc, device=device) - pn["mu"]) / pn["sd"]
        Yt = torch.tensor(Yc, device=device)
        optp = torch.optim.Adam(pol.parameters(), lr=cfg["pol_lr"]); lossf = nn.MSELoss()
        bs = min(256, len(Xc)); pol.train()
        for _ in range(cfg["pol_steps"]):
            idx = torch.tensor(brng.integers(0, len(Xc), size=bs), device=device)
            optp.zero_grad(); lossf(pol(Xt[idx]), Yt[idx]).backward(); optp.step()
        pol.eval()
        unit["pol"] = pol; unit["pnorm"] = pn
        return float(len(Xc))

    # ===================================================================== #
    # 5) THE PRACTICE LOOP
    # ===================================================================== #
    fm = copy.deepcopy(fm0)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = _tensors(normS, normU, normS2)          # the replay pool == the pretrain pool
    routing = fresh_routing()
    seg_R = {drill: 0} if spec.get("sloppy") else None      # 0 => forced open-loop for that segment
    op = spec.get("op", "bc")
    cert_b = {k: None for k in range(K)}      # benchmark ANCHORED to the COMMITTED unit's own level
    cert_pending = {k: 0 for k in range(K)}   # cycles left in the post-commit calibration window
    cert_win = {k: [] for k in range(K)}
    neg_run = {k: 0 for k in range(K)}        # consecutive cycles of delta < 0 post-commit
    drill_until = 0                           # >0 => inside the overspeed drill window
    reselected = False
    straddle_from = None                      # last committed unit, for the seam drill
    straddle_starts = None
    straddle_buf = {k: [] for k in range(K)}  # separate pool (straddle_sep) -> adds, never displaces

    buf_S = np.zeros((cfg["n_buf"], 4), np.float32); buf_U = np.zeros((cfg["n_buf"], 2), np.float32)
    buf_S2 = np.zeros((cfg["n_buf"], 4), np.float32); buf_n = 0; buf_p = 0
    trace_buf = {k: [] for k in range(K)}

    rt_rng_seed = cfg["seed"] + 5000        # metering run-through geometry (fixed, shared by arms)
    ev_rng_seed = cfg["seed"] + 6000        # held-out outcome geometry (fixed, shared by arms)
    rt_starts = start_states(np.random.default_rng(rt_rng_seed), cfg["n_rt"])
    ev_starts = start_states(np.random.default_rng(ev_rng_seed), cfg["n_eval"])

    bench = {k: None for k in range(K)}
    e0 = {k: None for k in range(K)}
    emin = {k: None for k in range(K)}
    sil_run = {k: 0 for k in range(K)}
    ehist = {k: [] for k in range(K)}
    dhist = {k: [] for k in range(K)}
    alpha = cfg["bench_alpha"]

    log = {"cycle": [], "t_cum": [], "e_rt": [], "b": [], "delta": [], "scale": [],
           "sil_run": [], "e_practice": [], "fm_region": [], "fm_clean": [],
           "fm_corridor": [], "fm_corridor_ceil": [],
           "compiled": [], "n_fb_practice": [], "n_in_region": [], "neg_run": []}
    ladder = []
    events = []
    t_cum = 0.0

    def ladder_probe(cycle):
        rec = {"cycle": cycle, "t_cum": t_cum,
               "compiled": [committed(u) for u in routing]}
        for R in cfg["ladder"]:
            rng = np.random.default_rng(cfg["seed"] + 7000 + R)
            o = traverse(fm, routing, R, ev_starts, rng)
            rec[f"R{R}"] = dict(wp_err=np.median(o["wp_err"], 0).tolist(),
                                wp_err_mean=float(o["wp_err"].mean()),
                                final=float(np.median(o["final"])),
                                t_piece=o["t_piece"], n_fb=o["n_fb"])
        rec["fm"] = fm_probes(fm)
        return rec

    outdir = os.path.join(DATA_DIR, "practice_etude_scratch", cfg["tag"], arm)
    os.makedirs(outdir, exist_ok=True)
    result = {"config": cfg, "arm": arm, "complete": False,
              "setup": {"fm_stale": fm_probes(fm0), "fm_ceiling": fm_probes(fm_ceil)}}

    def checkpoint():
        result["log"] = log; result["ladder"] = ladder; result["events"] = events
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(result, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # reference ladders: the stale FM and the ceiling FM, both under pure-reactive routing.
    result["ref_stale"] = {f"R{R}": traverse(fm0, fresh_routing(), R, ev_starts,
                                             np.random.default_rng(cfg["seed"] + 7000 + R))["wp_err"].tolist()
                           for R in cfg["ladder"]}
    result["ref_ceiling"] = {f"R{R}": traverse(fm_ceil, fresh_routing(), R, ev_starts,
                                               np.random.default_rng(cfg["seed"] + 7000 + R))["wp_err"].tolist()
                             for R in cfg["ladder"]}
    for nm in ("ref_stale", "ref_ceiling"):
        parts = []
        for R in cfg["ladder"]:
            med = np.median(np.array(result[nm][f"R{R}"]), 0).round(4).tolist()
            parts.append(f"R{R}:{med}")
        print(f"[ref] {nm:12s} " + "  ".join(parts), flush=True)
    checkpoint()

    # the detector's scale: the stale range per segment at performance tempo, measured here
    stale_ref = np.median(np.array(result["ref_stale"][f"R{cfg['ladder'][-1]}"], float), 0)
    ladder.append(ladder_probe(0))
    brng = np.random.default_rng(cfg["seed"] + 800)

    for cycle in range(1, cfg["n_cycles"] + 1):
        # --- (a) practice traversal: the drill, at the practice tempo -------------------
        prng = np.random.default_rng(cfg["seed"] + 10_000 + cycle)
        st = start_states(prng, cfg["batch"])
        pr = traverse(fm, routing, cfg["train_R"], st, prng, collect=True, seg_R=seg_R,
                      explore=cfg["explore_sigma"])
        t_cum += pr["t_piece"] * cfg["batch"]
        S, U, S2 = pr["trans"]
        m = len(S)
        for off in range(0, m, cfg["n_buf"]):
            chunk = slice(off, min(off + cfg["n_buf"], m))
            nc = chunk.stop - chunk.start
            idx = (np.arange(nc) + buf_p) % cfg["n_buf"]
            buf_S[idx] = S[chunk]; buf_U[idx] = U[chunk]; buf_S2[idx] = S2[chunk]
            buf_p = int((buf_p + nc) % cfg["n_buf"]); buf_n = min(cfg["n_buf"], buf_n + nc)
        for k, tr in pr["traces"].items():
            trace_buf[k].append(tr)
            if len(trace_buf[k]) > cfg["trace_window"]:
                trace_buf[k].pop(0)

        # --- (a2) the SEAM DRILL: extra practice windows that STRADDLE a committed boundary.
        # Start one segment before the seam, execute the committed unit at tempo, cross into the
        # next segment and practise it reactively. Half a piece per traversal, so it concentrates
        # seam-conditioned experience per unit of practice time — priced honestly into t_cum.
        if straddle_from is not None and straddle_from + 1 < K \
                and not committed(routing[straddle_from + 1]) and cfg["n_straddle"] > 0:
            k0 = straddle_from
            sub = [routing[k0], routing[k0 + 1]]
            sst = straddle_starts[np.random.default_rng(cfg["seed"] + 20_000 + cycle)
                                  .permutation(len(straddle_starts))[:cfg["n_straddle"]]]
            sd = traverse(fm, sub, cfg["train_R"], sst,
                          np.random.default_rng(cfg["seed"] + 30_000 + cycle),
                          collect=True, explore=cfg["explore_sigma"])
            t_cum += sd["t_piece"] * cfg["n_straddle"]
            S_, U_, S2_ = sd["trans"]; m_ = len(S_)
            idx = (np.arange(m_) + buf_p) % cfg["n_buf"]
            buf_S[idx] = S_; buf_U[idx] = U_; buf_S2[idx] = S2_
            buf_p = int((buf_p + m_) % cfg["n_buf"]); buf_n = min(cfg["n_buf"], buf_n + m_)
            tb = straddle_buf if spec.get("straddle_sep") else trace_buf
            tb[k0 + 1].append(sd["traces"][1])                # slice index 1 == segment k0+1
            if len(tb[k0 + 1]) > cfg["trace_window"]:
                tb[k0 + 1].pop(0)

        # --- (b) FM plasticity: plain uniform lr (no per-sample gain anywhere) ----------
        train_online(fm, opt, buf_S[:buf_n], buf_U[:buf_n], buf_S2[:buf_n], RX, RY,
                     cfg["n_grad"], brng, cfg["fm_batch"], cfg["replay_frac"])

        # --- (c) metering: the at-tempo run-through, one δ per segment per cycle --------
        rt = traverse(fm, routing, 0, rt_starts, np.random.default_rng(cfg["seed"] + 4000),
                      probe_unit=drill)
        e_rt = np.median(rt["wp_err"], 0)
        # CORRIDOR-SLICE FM PROBE: prediction error on exactly the (s,u) the committed corridor
        # traverses, vs `fm_region`'s broad uniform-position x N(0,v) x uniform-u volume. The two
        # can diverge by construction (bridge_assembly's spatial-matching law: a ballistic
        # controller is a narrow line-integral, so a corridor-specialised FM can beat a globally
        # accurate one there) — so the broad probe alone cannot say whether the passage is learned.
        cS, cU, cS2 = rt["probe_trans"]
        fm_corr = float(np.linalg.norm(fm_delta(fm, cS, cU) - (cS2 - cS), axis=1).mean())
        fm_corr_ceil = float(np.linalg.norm(fm_delta(fm_ceil, cS, cU) - (cS2 - cS), axis=1).mean())
        deltas, scales = [], []
        for k in range(K):
            e = float(e_rt[k])
            if bench[k] is None:
                bench[k] = e; e0[k] = e; emin[k] = e
            d = bench[k] - e
            emin[k] = min(emin[k], e)
            # the stale RANGE, measured against the setup stale reference at performance tempo.
            # Taking e0 from cycle 1 understates it ~2x, because one cycle of adaptation has
            # already happened by the first run-through, and that made both thresholds ~2x strict.
            if cert_pending[k]:
                # post-commit calibration: measure the committed unit's OWN level before anchoring.
                # e3_s0 froze b at the pre-commit CEM-ballistic level, so delta went negative on the
                # first post-commit cycle by exactly the level gap and stayed there -- mechanical,
                # not temporal, and the recert could not mean "degradation".
                cert_win[k].append(e); cert_pending[k] -= 1
                if cert_pending[k] == 0:
                    cert_b[k] = float(np.mean(cert_win[k]))
                    events.append(dict(kind="anchor", seg=k, cycle=cycle, level=cert_b[k],
                                       window=[float(x) for x in cert_win[k]]))
                    print(f"[anchor] arm={arm} seg={k} cycle={cycle} b_anchored={cert_b[k]:.4f} "
                          f"(window {np.round(cert_win[k], 4).tolist()})", flush=True)
            if cert_b[k] is not None:
                # certificate-anchored: b stops tracking, so post-commit degradation stays visible
                # as sustained delta<0 instead of being habituated away (eg_s0's delta_gate had
                # b climb 0.089 -> 0.257 in ~15 cycles and re-declare the eroded unit "mastered")
                d = cert_b[k] - e
            scale = max(stale_ref[k] - emin[k], (cert_b[k] or bench[k]), 1e-6)
            ehist[k].append(e); dhist[k].append(d)
            we = ehist[k][-cfg["sil_W"]:]; wd = dhist[k][-cfg["sil_W"]:]
            # §1's wording literally: SUSTAINED b-e ~ 0 with LOW VARIANCE. Testing the window MEAN
            # of delta rather than one instantaneous |delta| averages the run-through's per-visit
            # noise instead of demanding every single visit be quiet.
            quiet = (len(we) >= cfg["sil_W"]) and (abs(float(np.mean(wd))) < cfg["sil_c"] * scale) \
                and (float(np.std(we)) < cfg["sil_cv"] * scale)
            sil_run[k] = sil_run[k] + 1 if quiet else 0
            neg_run[k] = neg_run[k] + 1 if (cert_b[k] is not None and d < -cfg["sil_c"] * scale) else 0
            deltas.append(d); scales.append(scale)
            if cert_b[k] is None:
                bench[k] = bench[k] + alpha * (e - bench[k])

        # --- (d) the gate -------------------------------------------------------------
        nonlocal_t = [0.0]
        nonlocal_st = [None]

        def persist(tag_, k=None):
            import torch as _t
            k = drill if k is None else k
            _t.save(fm.state_dict(), os.path.join(outdir, f"fm_{tag_}.pt"))
            np.savez_compressed(
                os.path.join(outdir, f"traces_{tag_}.npz"),
                X=np.concatenate([b[0] for b in trace_buf[k]], 0),
                acts=np.concatenate([b[1] for b in trace_buf[k]], 0),
                raw=np.concatenate([b[2] for b in trace_buf[k]], 0),
                err=np.concatenate([b[3] for b in trace_buf[k]], 0))

        def commit(reason, k=None):
            k = drill if k is None else k
            u = routing[k]
            if op == "select_x":
                S0x = None
                if spec.get("score") in ("seam", "reactive_upstream"):
                    # the score set: hand-over states into unit k produced by the CURRENT
                    # PERFORMANCE CONFIGURATION ("seam"), or by a reactive-upstream traversal
                    # ("reactive_upstream" — e3b's distribution, the attribution control).
                    ru = spec["score"] == "reactive_upstream"
                    # the control must reproduce e3b's hand-over: upstream executed REACTIVELY
                    # (R=1), not merely uncommitted-at-tempo, which would still be ballistic.
                    sp = traverse(fm, routing, 1 if ru else cfg["ladder"][-1],
                                  start_states(np.random.default_rng(cfg["seed"] + 4400), cfg["n_seam"]),
                                  np.random.default_rng(cfg["seed"] + 4401),
                                  ignore_committed=ru)
                    S0x = sp["s0"][k]
                pool = trace_buf[k] + straddle_buf[k]      # union; straddle_buf is empty unless
                info = select_x_unit(u, pool, wps[u["span"][1] + 1],   # the arm uses a separate one
                                     np.random.default_rng(cfg["seed"] + 953 + cycle), S0_ext=S0x)
                if S0x is not None:
                    info["score_s0_disp"] = float(np.linalg.norm(
                        S0x[:, :2].mean(0) - np.asarray(wps[u["span"][0]])))
                    info["score_s0_spread"] = float(np.linalg.norm(S0x[:, :2].std(0)))
                    info["score_s0_speed"] = float(np.linalg.norm(S0x[:, 2:], axis=1).mean())
                    if spec.get("straddle"):
                        nonlocal_st[0] = (k, sp["s0"][k])      # arm the seam drill for k -> k+1
                nonlocal_t[0] += info["t_score"]          # the repeatability test costs real time
            elif op == "plan_eval":
                info = plan_eval_unit(u, rt["s0"][k], wps[u["span"][1] + 1], fm,
                                      np.random.default_rng(cfg["seed"] + 951))
            elif op == "select":
                info = select_unit(u, trace_buf[drill], cfg["k_best"])
            elif op == "plan":
                info = plan_unit(u, trace_buf[drill], fm, np.random.default_rng(cfg["seed"] + 951))
            else:
                info = dict(n_pool=float(compile_unit(u, trace_buf[drill],
                                                      np.random.default_rng(cfg["seed"] + 950))))
            persist(f"seg{k}_c{cycle}", k)
            ev = dict(kind="compile", seg=k, cycle=cycle, op=op, reason=reason,
                      n_tuples=info.get("n_pool", 0.0), t_cum=t_cum,
                      e_rt=float(e_rt[k]), b=float(bench[k]), **info)
            events.append(ev)
            if cfg["cert_anchor"]:
                cert_b[k] = None; cert_win[k] = []
                cert_pending[k] = cfg["cert_cal"]         # anchor AFTER measuring the new level
            print(f"[commit] arm={arm} op={op} reason={reason} cycle={cycle} "
                  f"e_rt={e_rt[drill]:.4f} info={ {k: v for k, v in info.items() if k != 's0_centroid'} }",
                  flush=True)

        # E-4: the commit target moves through the piece in order; a segment is eligible only
        # once every segment before it is committed.
        if spec["trigger"] == "sequential":
            tgt = next((k for k in range(K) if not committed(routing[k])), None)
        else:
            tgt = drill

        fired = False
        if spec["trigger"] == "sequential":
            fired = (tgt is not None and sil_run[tgt] >= cfg["sil_hold"]
                     and cycle >= cfg["sil_min_cycle"])
        elif not committed(routing[drill]) and drill_until == 0:
            if spec["trigger"] == "delta":
                fired = sil_run[drill] >= cfg["sil_hold"] and cycle >= cfg["sil_min_cycle"]
            elif spec["trigger"] == "cycle":
                fired = cycle >= (cfg["sched_early"] if spec["at"] == "early" else cfg["sched_late"])
        if fired and spec.get("overspeed") and cfg["overspeed_cycles"] > 0:
            # §5's overspeed corner: the certificate opens a forced-commitment drill window in
            # which the segment is practised OPEN-LOOP, and the unit is selected from THOSE
            # realisations rather than from closed-loop corrective traces.
            drill_until = cycle + cfg["overspeed_cycles"]
            seg_R = dict(seg_R or {}); seg_R[drill] = 0
            trace_buf[drill] = []
            events.append(dict(kind="overspeed_start", seg=drill, cycle=cycle, until=drill_until,
                               t_cum=t_cum, e_rt=float(e_rt[drill])))
            print(f"[overspeed] arm={arm} cycle={cycle} -> drill open-loop until c{drill_until}", flush=True)
        elif fired:
            commit({"delta": "certificate", "sequential": "sequential_certificate"}
                   .get(spec["trigger"], "schedule"), k=tgt)
        elif drill_until and cycle >= drill_until and not committed(routing[drill]):
            commit("overspeed_drill")
            seg_R = {drill: 0} if spec.get("sloppy") else None
            drill_until = 0
        elif cfg["recert_on_neg"] and committed(routing[drill]) and not reselected \
                and neg_run[drill] >= cfg["sil_W"] \
                and op in ("select", "plan", "select_x", "plan_eval"):
            reselected = True
            commit("recert_on_negative_delta")

        t_cum += nonlocal_t[0]; nonlocal_t[0] = 0.0
        if nonlocal_st[0] is not None:
            straddle_from, straddle_starts = nonlocal_st[0]; nonlocal_st[0] = None
        fp = fm_probes(fm)
        log["cycle"].append(cycle); log["t_cum"].append(t_cum)
        log["e_rt"].append(e_rt.tolist()); log["b"].append([bench[k] for k in range(K)])
        log["delta"].append(deltas); log["scale"].append(scales)
        log["sil_run"].append([sil_run[k] for k in range(K)])
        log["e_practice"].append(np.median(pr["wp_err"], 0).tolist())
        log["fm_region"].append(fp["region"]); log["fm_clean"].append(fp["clean"])
        log["fm_corridor"].append(fm_corr); log["fm_corridor_ceil"].append(fm_corr_ceil)
        log["compiled"].append([committed(u) for u in routing])
        log["neg_run"].append([neg_run[k] for k in range(K)])
        log["n_fb_practice"].append(pr["n_fb"]); log["n_in_region"].append(pr["n_in_region"])

        if cfg["discriminate_at"] and cycle == cfg["discriminate_at"]:
            import copy as _copy
            Rp = cfg["ladder"][-1]
            dstarts = start_states(np.random.default_rng(ev_rng_seed), cfg["disc_n_eval"])
            disc = {"cycle": cycle, "n_tuples": int(sum(len(b[0]) for b in trace_buf[drill])),
                    "e_rt_at_compile": float(e_rt[drill])}
            for R, nm in ((Rp, "ref_cem_ballistic"), (1, "ref_reactive")):
                o = traverse(fm, fresh_routing(), R, dstarts,
                             np.random.default_rng(cfg["seed"] + 7000 + R))
                disc[nm] = np.median(o["wp_err"], 0).tolist()
            for tgt in ("exec", "clean", "cem"):
                rt_ = fresh_routing()
                compile_unit(rt_[drill], trace_buf[drill], np.random.default_rng(cfg["seed"] + 950),
                             target=tgt, net=fm, prng=np.random.default_rng(cfg["seed"] + 951))
                o = traverse(fm, rt_, Rp, dstarts, np.random.default_rng(cfg["seed"] + 7000 + Rp))
                disc[f"bc_{tgt}"] = np.median(o["wp_err"], 0).tolist()
                print(f"[disc] target={tgt:6s} drilled={disc[f'bc_{tgt}'][drill]:.4f} "
                      f"piece={np.mean(disc[f'bc_{tgt}']):.4f}", flush=True)
            # ---------- pure-evaluation probes on the SAME state (no further training) ----------
            goal_d = wps[routing[drill]["span"][1] + 1]
            ref = traverse(fm, fresh_routing(), Rp, dstarts,
                           np.random.default_rng(cfg["seed"] + 7000 + Rp))
            b0 = ref["s0"][drill]                      # the hand-over states the unit is handed
            disc["boundary"] = dict(
                pos_sd=np.std(b0[:, :2], 0).tolist(), vel_sd=np.std(b0[:, 2:], 0).tolist(),
                pos_mean=np.mean(b0[:, :2], 0).tolist(), vel_mean=np.mean(b0[:, 2:], 0).tolist(),
                pos_spread=float(np.linalg.norm(np.std(b0[:, :2], 0))),
                vel_spread=float(np.linalg.norm(np.std(b0[:, 2:], 0))))

            def eval_fixed(seq):
                rt_ = fresh_routing(); rt_[drill]["fixed"] = seq.astype(np.float32)
                o = traverse(fm, rt_, Rp, dstarts, np.random.default_rng(cfg["seed"] + 7000 + Rp))
                return float(np.median(o["wp_err"], 0)[drill])

            # (a) ONE committed CEM plan for every episode — bounds s0-sensitivity of a planner unit
            cen = b0.mean(0); rank = np.argsort(np.linalg.norm(b0 - cen, axis=1))
            cand = b0[rank[[0, len(rank) // 4, len(rank) // 2]]]
            gseed = np.tile(goal_d[None, :], (len(cand), 1)).astype(np.float32)
            plans1 = mpc_plan(fm, cand.astype(np.float32), gseed,
                              np.random.default_rng(cfg["seed"] + 952), H)
            fx = [eval_fixed(plans1[i]) for i in range(len(cand))]
            disc["fixed_cem_median_s0"] = fx[0]; disc["fixed_cem_best_of_3"] = float(min(fx))
            disc["fixed_cem_all"] = fx

            # (b) the MEAN executed / mean pre-noise command sequence, no net, no fitting
            Ex = np.concatenate([b[1] for b in trace_buf[drill]], 0)
            Cl = np.concatenate([b[2] for b in trace_buf[drill]], 0)
            Xs = np.concatenate([b[0] for b in trace_buf[drill]], 0)
            disc["fixed_mean_exec"] = eval_fixed(Ex.mean(0))
            disc["fixed_mean_clean"] = eval_fixed(Cl.mean(0))

            # (c) per-trace replay: own s0 (determinism control) and cross s0 (s0-sensitivity of a
            #     single individual committed program, with no averaging and no fitting)
            def replay(s0s, cmds):
                errs = []
                for i in range(len(s0s)):
                    env.set_state(s0s[i, :2].astype(np.float64), s0s[i, 2:].astype(np.float64))
                    for h in range(cmds.shape[1]):
                        env.step(cmds[i, h], fs)
                    errs.append(float(np.linalg.norm(env.get_state()[:2] - goal_d)))
                return errs
            m = min(128, len(Xs)); sub = np.random.default_rng(0).permutation(len(Xs))[:m]
            own = replay(Xs[sub, :4], Ex[sub])
            perm = np.random.default_rng(1).permutation(sub)
            cross = replay(Xs[perm, :4], Ex[sub])
            disc["replay_own_s0"] = float(np.median(own))
            disc["replay_cross_s0"] = float(np.median(cross))
            disc["replay_own_s0_p90"] = float(np.percentile(own, 90))
            disc["replay_cross_s0_p90"] = float(np.percentile(cross, 90))
            disc["trace_boundary_pos_spread"] = float(np.linalg.norm(np.std(Xs[:, :2], 0)))

            # (d) capacity check: BC on the CEM targets at 4x width
            wide = dict(cfg); wide["pol_hidden"] = cfg["pol_hidden"] * 4
            rt_ = fresh_routing()
            saved = cfg["pol_hidden"]; cfg["pol_hidden"] = saved * 4
            compile_unit(rt_[drill], trace_buf[drill], np.random.default_rng(cfg["seed"] + 950),
                         target="cem", net=fm, prng=np.random.default_rng(cfg["seed"] + 951))
            cfg["pol_hidden"] = saved
            o = traverse(fm, rt_, Rp, dstarts, np.random.default_rng(cfg["seed"] + 7000 + Rp))
            disc["bc_cem_wide4x"] = float(np.median(o["wp_err"], 0)[drill])

            print(f"[disc] (a) fixed CEM plan: median_s0={disc['fixed_cem_median_s0']:.4f} "
                  f"best_of_3={disc['fixed_cem_best_of_3']:.4f}", flush=True)
            print(f"[disc] (b) mean exec seq={disc['fixed_mean_exec']:.4f} "
                  f"mean clean seq={disc['fixed_mean_clean']:.4f}", flush=True)
            print(f"[disc] (c) replay own-s0={disc['replay_own_s0']:.4f} "
                  f"cross-s0={disc['replay_cross_s0']:.4f}", flush=True)
            print(f"[disc] (d) bc_cem wide4x={disc['bc_cem_wide4x']:.4f}", flush=True)
            print(f"[disc] boundary pos_sd={disc['boundary']['pos_spread']:.4f} "
                  f"vel_sd={disc['boundary']['vel_spread']:.4f}", flush=True)
            disc["fm"] = fp
            result["discriminator"] = disc
            print(f"[disc] refs: cem_ballistic={disc['ref_cem_ballistic'][drill]:.4f} "
                  f"reactive={disc['ref_reactive'][drill]:.4f} n_tuples={disc['n_tuples']}", flush=True)
            result["complete"] = True
            checkpoint()
            with open(os.path.join(outdir, "done.txt"), "w") as fh:
                fh.write("ok\n")
            volume.commit()
            return result

        head = (f"[WALL {_time.perf_counter()-_WALL0:7.1f}s] [c{cycle:3d} {arm}] t_cum={t_cum:7.1f}s fm_reg={fp['region']:.4f} "
                f"fm_cln={fp['clean']:.4f} fm_cor={fm_corr:.4f}/{fm_corr_ceil:.4f} "
                f"e_rt={np.round(e_rt, 4).tolist()} d={np.round(deltas, 4).tolist()} "
                f"sil={[sil_run[k] for k in range(K)]}")
        if cycle % cfg["probe_every"] == 0 or cycle == cfg["n_cycles"]:
            ladder.append(ladder_probe(cycle))
            Rf = cfg["ladder"][-1]
            tail = np.round(ladder[-1][f"R{Rf}"]["wp_err"], 4).tolist()
            print(head + f" | R{Rf} wp={tail}", flush=True)
            checkpoint()
        elif cycle % 5 == 0:
            print(head, flush=True)

    result["complete"] = True
    checkpoint()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    print(f"[save] {outdir}", flush=True)
    return result


@app.local_entrypoint()
def etude(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "never,delta_gate,sched_early,sched_late,sloppy",
    # the piece
    waypoints: str = DEF_WAYPOINTS,
    regions: str = DEF_REGIONS,
    drill_seg: int = 1,
    seg_h: int = 34,
    # env
    frame_skip: int = 12,
    arena_half: float = 1.8,
    gear: float = 10.0,
    damping: float = 2.0,
    box_half: float = 0.95,
    v_explore: float = 1.2,
    # FM
    pool_n: int = 12000,
    probe_n: int = 800,
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 256,
    fm_steps: int = 5000,
    adapt_lr: float = 3e-4,
    n_grad: int = 10,        # the adaptation clock. 40 mastered the passage inside ONE cycle
                             # (cal_s1: stale 0.163 -> 0.083 by the first run-through), leaving no
                             # learning curve for a silence detector to see.
    n_buf: int = 24000,
    replay_frac: float = 0.5,          # share of each online batch drawn from the pretrain pool
    pretrain_mode: str = "exclude",    # "exclude" = true world minus the hard passage; "clean" = cal_s0
    pretrain_exclude_w: float = 0.1,
    # compiled unit (ballistic_bc)
    pol_hidden: int = 128,
    pol_layers: int = 2,
    pol_lr: float = 1e-3,
    pol_steps: int = 1500,
    trace_window: int = 8,
    recompile_every: int = 0,
    # practice loop
    n_cycles: int = 60,
    batch: int = 32,
    train_r: int = 1,
    n_rt: int = 96,          # run-through performers: the metering estimator's own averaging
    n_eval: int = 40,
    explore_sigma: float = 0.25,       # motor variability on practice commands (practice only)
    probe_every: int = 4,
    ladder: str = "1,4,12,34",
    start_jit: float = 0.05,
    v0_std: float = 0.1,
    # metering / gate
    bench_alpha: float = 0.2,
    sil_c: float = 0.06,
    sil_cv: float = 0.10,
    sil_w: int = 5,
    sil_hold: int = 2,
    sil_min_cycle: int = 3,
    sched_early: int = 4,
    sched_late: int = 40,
    # controllers
    k_shoot: int = 256,
    cem_iters: int = 4,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    # time model
    d_fb: float = 0.10,
    # offline discriminator: at this cycle, compile the drilled unit three ways and evaluate
    discriminate_at: int = 0,
    disc_n_eval: int = 96,
    # E-3: selection-based commitment, certificate anchoring, the overspeed drill
    k_best: int = 1,
    n_cand: int = 32,        # candidates drawn uniformly from the pool for repeatability scoring
    n_score: int = 16,       # held-out hand-over states each candidate is replayed from
    n_seam: int = 32,        # performers in the seam probe that generates the score set
    n_straddle: int = 16,    # performers per seam-drill straddle traversal (0 = off)
    cert_cal: int = 3,       # post-commit cycles measured before b is anchored
    cert_anchor: bool = False,
    recert_on_neg: bool = False,
    overspeed_cycles: int = 4,
    fast: bool = False,
):
    arm_list = [a for a in arms.split(",") if a]
    for a in arm_list:
        if a not in ARM_TABLE:
            raise ValueError(f"unknown arm {a!r}; known: {sorted(ARM_TABLE)}")
    wps = parse_waypoints(waypoints)
    regs = parse_regions(regions, wps)
    lad = [int(x) for x in ladder.split(",") if x]
    if quick:
        pool_n = 3000; probe_n = 300; fm_steps = 1200; n_cycles = 6; batch = 8
        probe_every = 3; pol_steps = 400; n_grad = 20
        fm_hidden = 128; fm_layers = 2; k_shoot = 128; lad = [1, 34]
        sched_early = 2; sched_late = 5; sil_w = 2; sil_min_cycle = 1
        n_rt = 8; n_eval = 8
        tag = tag or "smoke"
    tag = tag or "default"

    base = dict(
        tag=tag, seed=seed, waypoints=wps, regions=regs, drill_seg=drill_seg, seg_H=seg_h,
        frame_skip=frame_skip, arena_half=arena_half, gear=gear, damping=damping,
        box_half=box_half, v_explore=v_explore,
        pool_n=pool_n, probe_n=probe_n, fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr,
        fm_batch=fm_batch, fm_steps=fm_steps, adapt_lr=adapt_lr, n_grad=n_grad, n_buf=n_buf,
        replay_frac=replay_frac, pretrain_mode=pretrain_mode, pretrain_exclude_w=pretrain_exclude_w,
        explore_sigma=explore_sigma,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps,
        trace_window=trace_window, recompile_every=recompile_every,
        n_cycles=n_cycles, batch=batch, train_R=train_r, n_rt=n_rt, n_eval=n_eval,
        probe_every=probe_every, ladder=lad, start_jit=start_jit, v0_std=v0_std,
        bench_alpha=bench_alpha, sil_c=sil_c, sil_cv=sil_cv, sil_W=sil_w, sil_hold=sil_hold,
        sil_min_cycle=sil_min_cycle, sched_early=sched_early, sched_late=sched_late,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, d_fb=d_fb,
        discriminate_at=discriminate_at, disc_n_eval=disc_n_eval,
        k_best=k_best, n_cand=n_cand, n_score=n_score, n_seam=n_seam,
        n_straddle=n_straddle, cert_cal=cert_cal,
        cert_anchor=cert_anchor, recert_on_neg=recert_on_neg,
        overspeed_cycles=overspeed_cycles, fast=fast,
    )
    cfgs = [{**base, "arm": a} for a in arm_list]
    outs = list(run_etude.map(cfgs))
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for o in outs:
        with open(os.path.join(localdir, f"{o['arm']}.json"), "w") as fh:
            json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(outs)} arm result files to {localdir}")
