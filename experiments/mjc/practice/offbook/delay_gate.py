"""ROUND 4, PHASE A — does an observation delay make depth BIND?

THE STANDING FACT this round attacks. Across O1/O2/O3 the reactive/live-replan strategies won on
error every time, and the chain level was never selected at performance tempo even once. Depth
exists GEOMETRICALLY on this piece — a chain is 60 control steps against a ~21-step composition
horizon — but it has been cheap to flatten, because re-grounding costs exactly one feedback event
and nothing else. O3 settled that this is not a teacher problem (rehearsal at full credit put 203
chain plays on the body; they executed 2.77x worse than the median traversal and the competence band
correctly refused them). If chunks are ever to pay here, the WORLD has to make feedback expensive.

THE INTERVENTION. An observation delay `obs_delay` on the reflex loop: every agent-side feedback
consumer sees the state from Δ control steps ago, and motor noise makes that stale estimate diverge
from the truth. This is the biologically honest form — reflex delay against movement time is why
ballistic chunks exist in animals — and, unlike a feedback-budget cap, it TAXES feedback-consuming
strategies through physics instead of forbidding them.

  *The alternative considered and set aside*: a hard cap on feedback events per traversal. Rejected
  because adoption under necessity-by-fiat weakens the very claim the arc is chasing — a chunk
  adopted because nothing else is legal says nothing about whether it was worth adopting. Recorded
  as the alternative rather than silently not done.

Δ IS IN REAL UNITS: `dt_ctrl = frame_skip * timestep = 0.02 s`, so Δ = 2/4/8/16 is 40/80/160/320 ms
against a 400 ms drilled segment and a 1.2 s phrase. Human proprioceptive loops run ~100 ms and
visual ~200-250 ms, so the middle of this sweep is where a person actually lives.

THE NEUTRAL CRITERION, PRE-FIXED HERE, BEFORE THE RUN, AND BLIND TO THE TRUST QUESTION.
legato F2's lesson is that a knob chosen to make a hypothesis win destroys the axis it was supposed
to measure, so the rule is written down first:

    Δ* = the SMALLEST Δ in the sweep at which the ordering
             e_chain  <=  e_live_seg  <=  e_reactive
         holds on MEAN piece error over the shared eval geometries,
    subject to a COMPETENCE GUARD: min(e_chain, e_live_seg, e_reactive) at Δ* must be
         <= `ref_stale`, the piece error of ballistic-per-segment control under the STALE
         (pre-practice) forward model at Δ = 0 — this node's own arm-neutral "usable range"
         anchor, the same quantity every offbook run already reports in `setup`.
         (i.e. the piece must still be PLAYABLE at Δ*, so the criterion cannot be satisfied by
          universal collapse, which would be a measurement of nothing).

    GUARD REVISED 2026-08-26, BEFORE THE REAL RUN, ON SMOKE EVIDENCE, AND BLIND TO THE TRUST
    QUESTION. The first draft anchored the guard at `2x the best of the three at Δ = 0`. The smoke
    showed that is DEGENERATE: at Δ = 0 the incumbent is reactive MPC, which on this piece runs
    ~0.009 in a real run, so the ceiling lands at ~0.017 — below anything a chunk has ever scored
    here (O3 measured realised chain plays at 0.289). A guard anchored to the undelayed champion
    cannot fire once you tax feedback at all, so it would have vetoed every Δ by construction and
    discriminated nothing. `ref_stale` is task-anchored and arm-neutral instead, which is what
    legato F2 actually prescribes ("a shared calibration knob is chosen on an arm-neutral
    reference"). Both numbers are reported in the output so the original can be read off the
    record; the revision was made on smoke data with no real-run result in hand.

If no Δ in the sweep satisfies both, THE ROUND STOPS AT THE GATE and that is a substrate finding:
on this plant, delaying the reflex loop does not buy the chunk its niche. Δ is NOT to be extended or
re-tuned until the ordering inverts.

Run:
    modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag dsmoke
    modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d0 --seed 0
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.offbook.piece import DEF_WPS, DEF_PATCH_SEG


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_delay_gate(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler
    from mjc.practice.offbook.world import (World, Ledger, Library, RoutePolicy,
                                            start_postures, elite_for)
    from mjc.practice.offbook import nets as N

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    NS = W.n_seg
    led = Ledger()
    out = {"config": cfg, "complete": False,
           "criterion": "smallest D with e_chain<=e_live_seg<=e_reactive on mean piece error, "
                        "subject to min(of the three) <= 2x the best at D=0"}
    outdir = os.path.join(DATA_DIR, "practice_offbook", cfg["tag"], "delay_gate")
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print("[dgate]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}

    # ---------------------------------------------------------------- setup (offbook's diet)
    S1, U1, S21, _ = W.collect_ou(cfg["pool_ou"], np.random.default_rng(cfg["seed"] + 11))
    W.set_norm(S1, U1, S21)
    boot = W.mlp(cfg["seed"] + 40)
    W.train_steps(boot, torch.optim.Adam(boot.parameters(), lr=cfg["fm_lr"]),
                  S1, U1, S21, cfg["fm_steps_boot"], np.random.default_rng(cfg["seed"] + 300))
    gs = make_arm_goal_sampler(W.Ls, cfg["reach_amp"], cfg["reach_lo"], cfg["reach_hi"])
    S2_, U2_, S22, _ = W.collect_reach(cfg["pool_reach"], np.random.default_rng(cfg["seed"] + 12),
                                       boot, gs)
    Sa = np.concatenate([S1, S2_]); Ua = np.concatenate([U1, U2_]); S2a = np.concatenate([S21, S22])
    Se, Ue, S2e, _ = W.exclude_region(Sa, Ua, S2a, w=cfg["exclude_w"])
    W.set_norm(Se, Ue, S2e)
    fm0 = W.mlp(cfg["seed"] + 41)
    W.train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                  Se, Ue, S2e, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    fm_app = copy.deepcopy(fm0)
    for p in fm_app.parameters():
        p.requires_grad_(False)

    def geom(m, seed):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(seed),
                              cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])

    def app_plan(q, seed):
        st = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(st, np.tile(W.goals[0][None, :], (len(q), 1)),
                  np.random.default_rng(seed))[0]

    R_REACT = W.routing("reactive")
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    brng = np.random.default_rng(cfg["seed"] + 800)
    buf, traces = [], []
    P(f"[warm] {cfg['n_warm']} reactive cycles at delay 0 (the piece is learned undelayed; the "
      f"delay is a DECISION constraint applied afterwards) ...")
    for c in range(1, int(cfg["n_warm"]) + 1):
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + c)
        pr = W.traverse(fm, R_REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + c),
                        cfg["sigma_practice"], led, who="agent", kind="practice",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + c), collect=True)
        buf.append(pr["trans"])
        traces.append(dict(acts=pr["acts"].copy(), e_seg=pr["e_seg"].copy(),
                           launches={k: v.copy() for k, v in pr["launches"].items()}))
        if len(buf) > cfg["trace_window"]:
            buf.pop(0)
        PX, PY = W.tensors(*[np.concatenate([b[i] for b in buf]) for i in range(3)])
        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])

    layout = N.SlotLayout(NS, cfg["n_slot"], poison=False)
    library = Library(W, layout, cfg["n_slot"], cfg["seed"] + 2100)
    lrng = np.random.default_rng(cfg["seed"] + 820)
    for k in range(NS):
        for ns in layout.levels(k):
            rows = []
            for h in traces:
                lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k + ns - 1])
                tp = h["acts"][:, lo:hi, :]
                er = np.nanmean(h["e_seg"][:, k:k + ns], 1)
                for b in range(len(tp)):
                    if np.isfinite(tp[b]).all():
                        rows.append((tp[b], h["launches"][k][b], er[b]))
            pick = lrng.permutation(len(rows))[: int(cfg["lib_k"])]
            library.cell(ns, k).add(np.stack([rows[i][0] for i in pick]),
                                    np.stack([rows[i][1] for i in pick]),
                                    np.array([rows[i][2] for i in pick], np.float32),
                                    0, -np.ones((len(pick), ns), int), library.rng)
    P(f"[lib] {library.sizes()}")

    # ---------------------------------------------------------------- the sweep
    q_ev = geom(cfg["n_eval_gate"], cfg["seed"] + 5200)
    plan_ev = app_plan(q_ev, cfg["seed"] + 5250)
    chain_levels = {ns for k in range(NS) for ns in layout.levels(k) if ns > 1}
    seam_S = None

    def mk_pol(levels, prim):
        pol = RoutePolicy("audit", W, layout, library, k_prop=layout.n_slots, eps=0.0,
                          eps_act=0.0, p_rehearse=0.0, aud_horizon=cfg["aud_horizon"],
                          delib_budget=0.0, cem_ladder=[CALP[1]["k_shoot"]],
                          cem_iters=CALP[1]["cem_iters"], k_shoot=CALP[1]["k_shoot"],
                          prim=prim, seed=cfg["seed"] + 2200, device=device)
        pol.levels = levels
        pol.set_norm(np.concatenate([seam_S[k] for k in range(NS)]))
        return pol

    # the arm-neutral playability anchor: ballistic-per-segment under the STALE forward model,
    # undelayed. Same quantity `offbook.py` reports as `ref_stale` in every run's `setup`.
    R_BALL_SEG = W.routing("plan_launch", groups=[1] * NS, **CALP[1])
    ref_stale = float(W.traverse(fm0, R_BALL_SEG, q_ev,
                                 np.random.default_rng(cfg["seed"] + 7200), cfg["sigma_perf"],
                                 led, who="instrument", kind="ref_stale",
                                 approach_plan=plan_ev)["e_piece"].mean())
    out["ref_stale"] = ref_stale
    P(f"[ref] stale ballistic-per-segment (the playability anchor): {ref_stale:.4f}")

    os_ = W.traverse(fm, R_REACT, q_ev, np.random.default_rng(cfg["seed"] + 4400),
                     cfg["sigma_perf"], led, who="instrument", kind="score_set",
                     approach_plan=plan_ev)
    seam_S = {k: os_["launches"][k] for k in range(NS)}

    sweep = {}
    for D in [int(x) for x in cfg["delays"]]:
        W.obs_delay = D
        cell = {}
        # (1) reactive MPC, the incumbent -- replans every step from a state D steps stale
        o = W.traverse(fm, R_REACT, q_ev, np.random.default_rng(cfg["seed"] + 7700),
                       cfg["sigma_perf"], led, who="instrument", kind=f"react_d{D}",
                       approach_plan=plan_ev)
        cell["reactive"] = dict(e=float(o["e_piece"].mean()), e_med=float(np.median(o["e_piece"])),
                                by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)],
                                fb=float(o["n_fb"]), planner_free=False)
        # (2) live segment plan from the stale seam state, flown open-loop
        # (3) one measured chain, keyed/auditioned on the stale seam state -- PLANNER-FREE
        for nm, lv, pm in (("live_seg", set(), True), ("chain", chain_levels, False),
                           ("seg_tape", {1}, False)):
            try:
                oo = W.traverse_route(fm, mk_pol(lv, pm), q_ev,
                                      np.random.default_rng(cfg["seed"] + 7710),
                                      cfg["sigma_perf"], led, who="instrument",
                                      kind=f"{nm}_d{D}", approach_plan=plan_ev)
                cell[nm] = dict(e=float(oo["e_piece"].mean()),
                                e_med=float(np.median(oo["e_piece"])),
                                by_seg=[float(np.median(oo["e_seg"][:, k])) for k in range(NS)],
                                fb=float(oo["n_fb"]),
                                planner_free=(nm != "live_seg"))
            except RuntimeError as ex:
                cell[nm] = {"error": str(ex)}
        sweep[str(D)] = cell
        P(f"[D={D:2d} = {D * W.dt_ctrl * 1000:.0f} ms] " + "  ".join(
            f"{nm}={cell[nm]['e']:.4f}(fb{cell[nm]['fb']:.1f})"
            for nm in ("reactive", "live_seg", "seg_tape", "chain") if "e" in cell[nm]))
        out["sweep"] = sweep
        save()
    W.obs_delay = 0

    # ---------------------------------------------------------------- apply the pre-fixed rule
    d0 = sweep[str(int(cfg["delays"][0]))]
    best0 = min(d0[n]["e"] for n in ("reactive", "live_seg", "chain") if "e" in d0[n])
    verdict = {"best_at_D0": best0, "guard_ceiling": ref_stale,
               "guard_ceiling_original_draft": 2.0 * best0, "delta_star": None, "rows": []}
    for D in [int(x) for x in cfg["delays"]]:
        c = sweep[str(D)]
        if not all("e" in c[n] for n in ("reactive", "live_seg", "chain")):
            continue
        er, el, ec = c["reactive"]["e"], c["live_seg"]["e"], c["chain"]["e"]
        ordered = bool(ec <= el <= er)
        playable = bool(min(er, el, ec) <= ref_stale)
        verdict["rows"].append(dict(delay=D, ms=D * W.dt_ctrl * 1000, e_reactive=er,
                                    e_live_seg=el, e_chain=ec, ordered=ordered,
                                    playable=playable, passes=bool(ordered and playable),
                                    playable_original_draft=bool(min(er, el, ec) <= 2.0 * best0)))
        if ordered and playable and verdict["delta_star"] is None:
            verdict["delta_star"] = D
    out["verdict"] = verdict
    P(f"[verdict] guard ceiling {verdict['guard_ceiling']:.4f} (ref_stale) "
      f"| original draft ceiling was {2.0 * best0:.4f} (2x best at D=0), reported for the record")
    for r in verdict["rows"]:
        P(f"  D={r['delay']:2d} ({r['ms']:5.0f} ms)  chain {r['e_chain']:.4f} <= live "
          f"{r['e_live_seg']:.4f} <= react {r['e_reactive']:.4f} ? ordered={r['ordered']} "
          f"playable={r['playable']} -> {'PASS' if r['passes'] else 'no'}")
    P(f"[verdict] Delta* = {verdict['delta_star']}"
      + ("  -> Phase B is licensed at this delay" if verdict["delta_star"] is not None
         else "  -> NO delay inverts the ordering under the pre-fixed rule; the round STOPS at the "
              "gate, and that is the finding. Do not extend or re-tune the sweep."))
    out["ledger"] = led.snapshot()
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    return out


@app.local_entrypoint()
def delay_gate(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    delays: str = "0,2,4,8,16",
    n_warm: int = 20,
    lib_k: int = 72,
    n_slot: int = 8,
    aud_horizon: int = 0,
    n_eval_gate: int = 24,
    curl_b: float = 14.0,
    push_a: float = 0.0,
    waypoints: str = DEF_WPS,
    h_app: int = 14,
    h_seg: str = "20,20,20",
    patch_seg: int = DEF_PATCH_SEG,
    q_jit: float = 0.15,
    null_jit: float = 0.30,
    start_mode: str = "iso",
    patch_sigma: float = 0.08,
    patch_center: str = "",
    push_sigma: float = 0.0,
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,1.1,0.8",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    timestep: float = 0.002,
    wrap_limit: float = 3.0,
    pool_ou: int = 6000,
    pool_reach: int = 12000,
    exclude_w: float = 0.1,
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
    fm_hidden: int = 256,
    fm_layers: int = 3,
    fm_lr: float = 1e-3,
    fm_batch: int = 512,
    fm_steps: int = 6000,
    fm_steps_boot: int = 3000,
    adapt_lr: float = 3e-4,
    n_grad: int = 6,
    replay_frac: float = 0.5,
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.0,
    vel_pen_mid: float = 0.0,
    w_waypoint: float = 16.0,
    lookahead_gamma: float = 0.0,
    react_look: int = 20,
    plan_max_elems: int = 40_000_000,
    calp: str = "1:1024:8,2:4096:12,3:4096:12",
    batch: int = 16,
    trace_window: int = 10,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500
        fm_steps = 1000; fm_steps_boot = 500
        n_warm = 3; lib_k = 24; n_slot = 3; batch = 8; n_eval_gate = 8
        delays = "0,4,16"
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,2:256:4,3:256:4"
        tag = tag or "dsmoke"
    tag = tag or "d0"
    cfg = dict(
        tag=tag, seed=seed, delays=[int(x) for x in delays.split(",") if x],
        n_warm=n_warm, lib_k=lib_k, n_slot=n_slot, aud_horizon=aud_horizon,
        n_eval_gate=n_eval_gate, curl_b=curl_b, push_a=push_a,
        waypoints=[[float(v) for v in p.split(",")] for p in waypoints.split(";") if p],
        h_app=h_app, h_seg=[int(v) for v in h_seg.split(",")], patch_seg=patch_seg,
        q_jit=q_jit, null_jit=null_jit, start_mode=start_mode, patch_sigma=patch_sigma,
        patch_center=([float(x) for x in patch_center.split(",")] if patch_center else None),
        push_sigma=(push_sigma or patch_sigma), push_center=None,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, timestep=timestep,
        wrap_limit=wrap_limit, pool_ou=pool_ou, pool_reach=pool_reach, exclude_w=exclude_w,
        ep_len=ep_len, n_par=n_par, op_q_range=op_q_range, sigma_u=sigma_u,
        ou_sigma=ou_sigma, ou_theta=ou_theta, collect_k_shoot=collect_k_shoot,
        collect_cem_iters=collect_cem_iters, reach_amp=reach_amp, reach_lo=reach_lo,
        reach_hi=reach_hi, fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr,
        fm_batch=fm_batch, fm_steps=fm_steps, fm_steps_boot=fm_steps_boot,
        adapt_lr=adapt_lr, n_grad=n_grad, replay_frac=replay_frac,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, vel_pen_mid=vel_pen_mid,
        w_waypoint=w_waypoint, lookahead_gamma=lookahead_gamma, react_look=react_look,
        plan_max_elems=plan_max_elems,
        calp={p.split(":")[0]: [int(p.split(":")[1]), int(p.split(":")[2])]
              for p in calp.split(",") if p},
        batch=batch, trace_window=trace_window, sigma_practice=sigma_practice,
        sigma_perf=sigma_perf, d_fb=d_fb, obs_delay=0,
    )
    if spawn:
        c = run_delay_gate.spawn(cfg)
        print(f"[spawn] delay_gate: {c.object_id}")
        print(f"[spawn] detached; results -> /data/practice_offbook/{tag}/delay_gate/results.json")
        return
    o = run_delay_gate.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "delay_gate.json"), "w") as fh:
        json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {localdir}/delay_gate.json")
