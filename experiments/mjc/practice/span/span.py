"""Round 2c -- does the composition horizon GROW with practice while the task metric is flat?

WHAT THIS MEASURES. `legato`'s G5a measured the plant's composition horizon exactly twice: a stale
forward model composes ~21-23 steps before its open-loop tip prediction leaves 0.05 m of the truth,
a corridor-matched ceiling model ~44-58. Two points on a curve nobody has traced. This node traces
it: run `legato`'s practice loop with NOTHING committed (the `never` configuration -- pure reactive
MPC, which is the arm every other arm is identical to before its first commit), snapshot the
forward model at a cadence dense early and sparse late, and at each snapshot measure how far a
single committed unit could reach.

TWO HORIZONS, deliberately separated, because they can move independently:

  (1) THE IMAGINATION HORIZON -- G5a's own construction, planner-free. Roll the forward model
      open-loop along a FIXED set of commands the body actually executed, and score its predicted
      tip against the plant's. The horizon is the first step where the median error crosses a
      threshold. Nothing but the model changes between snapshots, so this is a pure model
      quantity. Cheap (one MLP rollout), so it runs at EVERY cycle.

      The measurement ceiling is the length of the command sequence, and `legato`'s was 60 steps --
      one phrase -- against a ceiling model that already reached 58. So the reference here is TWO
      LAPS of the closed loop, 134 steps (phrase + approach + phrase), stitched from consecutive
      `traverse` calls via the additive `acts_app` readout. A horizon that saturates at the top of
      the probe is a measurement ceiling and is reported as one.

  (2) THE EXECUTED SPAN -- the behavioural analogue. Plan the whole phrase from W1 with the
      snapshot model at CAL-P's phrase sizing, fly it OPEN-LOOP on the real body with performance
      motor noise, and record the per-step tip error. This is `legato`'s `phrase_plan_launch`
      content, probed as a curve rather than as three seam errors. Costs one CEM call, so it runs
      on a sparser schedule.

      FIVE references, all recorded, because "tip error" has no canonical zero on a piece whose
      cost function rewards arriving early at each waypoint and dwelling there:
        `goal`        -- distance to the per-step goal schedule (each step scored against ITS
                         segment's waypoint). What the planner actually minimises, and a sawtooth
                         by construction, so its LEVEL says nothing on its own.
        `goal_excess` -- the same, minus the reference reactive traversal's own `goal` curve FROM
                         THE SAME LAUNCH STATE (paired). Natural zero; the headline candidate.
        `ref`         -- distance to that reference trajectory itself. "How far has this open-loop
                         rendition drifted from a competent one." Non-monotone: both trajectories
                         get pulled back toward each waypoint.
        `path`        -- perpendicular distance to the current leg. Cross-track only, natural zero.
        `ideal`       -- distance to constant-tempo linear interpolation along the leg. Recorded,
                         but the smoke run already showed the competent reference itself leaves it
                         within 2 steps, so its floor is far too high to read a horizon off.
      and three floors, measured once: the reference chain re-executed open-loop with FRESH motor
      noise (the irreducible cost of noise alone under a perfect -- because measured -- plan; this
      is literally `legato`'s frozen content), the same noiselessly, and a `replay` chaos floor
      (the issued commands replayed through `set_state`, which re-enters from a float32 state:
      6e-7 m of drift by step 134, so the plant's own sensitivity is not in the way).

WHY THE `never` ARM. `legato`'s L1 ladder has `e_react` flat at ~0.006-0.019 from cycle 0: the
reactive task metric is at plateau for the whole run, so anything the model's composability does is
invisible to it. That is the situation Iwane et al. 2026 report -- chunk size keeps growing through
the speed plateau -- and it is the reason this node reads the horizon rather than the task error.

NOT AN ECONOMICS NODE. Metering is dropped (it never fed the forward model; it only priced the
agent's self-reads), so `t_priced` here is NOT comparable to `legato`'s ledger. The ladder keeps
`e_react` (the task metric) and `e_ball_seg` (the per-segment ballistic clock); `e_ball_phrase`
comes off the executed-span probe's seam errors, on matched launch states.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/span/span.py::span --quick
    python3 mjc/practice/span/launch_detached.py --tag S1 --seed 0
    python3 mjc/practice/span/analyze_span.py --tag S1 --fetch
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.legato.gates import DEF_WPS, DEF_PATCH_SEG

# The thresholds a horizon is read at. `legato` G5a used 0.05 m (headline) and 0.02 m; the extra
# two bracket them so the sensitivity of every reported number is on the record without a re-run.
THRS = (0.01, 0.02, 0.05, 0.10, 0.20)


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_span(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler, pool_diagnostics, cmd_state_corr
    from mjc.practice.legato.world import World, Ledger, start_postures, elite_for

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    n, NS, AD = W.n, W.n_seg, W.AD
    led = Ledger()
    out = {"config": cfg, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_span", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print("[span]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}
    P(f"[setup] device={device} H_seg={W.H_seg} H_phrase={W.H_phrase} H_app={W.H_app} "
      f"cycles={cfg['n_cycles']} calp={cfg['calp']} thresholds={THRS}")

    # ================================================================= diet + models
    # Byte-identical to `legato.run_legato`'s setup block, from the same seed offsets, so the model
    # this node snapshots is the model that node practised.
    S1, U1, S21, _ = W.collect_ou(cfg["pool_ou"], np.random.default_rng(cfg["seed"] + 11))
    W.set_norm(S1, U1, S21)
    boot = W.mlp(cfg["seed"] + 40)
    W.train_steps(boot, torch.optim.Adam(boot.parameters(), lr=cfg["fm_lr"]),
                  S1, U1, S21, cfg["fm_steps_boot"], np.random.default_rng(cfg["seed"] + 300))
    gs = make_arm_goal_sampler(W.Ls, cfg["reach_amp"], cfg["reach_lo"], cfg["reach_hi"])
    S2_, U2_, S22, _ = W.collect_reach(cfg["pool_reach"], np.random.default_rng(cfg["seed"] + 12),
                                       boot, gs)
    Sa = np.concatenate([S1, S2_]); Ua = np.concatenate([U1, U2_]); S2a = np.concatenate([S21, S22])
    Se, Ue, S2e, keep_frac = W.exclude_region(Sa, Ua, S2a, w=cfg["exclude_w"])
    W.set_norm(Se, Ue, S2e)                       # set ONCE; the snapshots share this normaliser
    fm0 = W.mlp(cfg["seed"] + 41)
    W.train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                  Se, Ue, S2e, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    fm_app = copy.deepcopy(fm0)
    for p in fm_app.parameters():
        p.requires_grad_(False)

    def geom(m, seed):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(seed),
                              cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])

    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)

    def app_plan(q, seed):
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(s, np.tile(W.goals[0][None, :], (len(q), 1)),
                  np.random.default_rng(seed))[0]

    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    R_REACT = W.routing("reactive")
    R_BALL_SEG = W.routing("plan_launch", groups=[1] * NS, **CALP[1])

    # corridor-matched ceiling model -- the upper anchor for the horizon curve, and `legato` G5a's
    # 44 / 58. Same construction, same seed offsets.
    corr = []
    for i in range(cfg["n_corridor"]):
        qq = geom(cfg["batch"], cfg["seed"] + 5400 + i)
        corr.append(W.traverse(fm0, R_REACT, qq, np.random.default_rng(cfg["seed"] + 6500 + i),
                               cfg["sigma_practice"], led, who="instrument", kind="ceiling_diet",
                               replan_every=cfg["corridor_replan"],
                               approach_plan=app_plan(qq, cfg["seed"] + 6400 + i),
                               collect=True)["trans"])
    fm_ceil = W.mlp(cfg["seed"] + 42)
    W.train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                  np.concatenate([Sa] + [c[0] for c in corr]),
                  np.concatenate([Ua] + [c[1] for c in corr]),
                  np.concatenate([S2a] + [c[2] for c in corr]),
                  cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))
    out["setup"] = dict(exclude_keep_frac=keep_frac, corr_reach=cmd_state_corr(S2_, U2_),
                        pool_diag=pool_diagnostics(Se, Ue, n))
    save()

    # ================================================================= the FIXED reference set
    # G5a's `executed` probe, extended to two laps. Lap 1 is byte-identical to the gate's
    # construction (reactive under `fm0`, held-out geometry `q_ev`, the mastered approach plan,
    # seed offset 7001, performance motor noise). Lap 2 restarts the SAME bodies from the state
    # lap 1 ended in -- which is W0, the loop's closing waypoint -- so its approach leg (W0 -> W1)
    # and its three drilled segments continue the trajectory without a gap. Concatenating
    # [lap1.acts | lap2.acts_app | lap2.acts] therefore gives ONE contiguous executed command
    # sequence of H_phrase + H_app + H_phrase steps from a single launch state.
    ex1 = W.traverse(fm0, R_REACT, q_ev, np.random.default_rng(cfg["seed"] + 7001),
                     cfg["sigma_perf"], led, who="instrument", kind="ref_lap1",
                     approach_plan=plan_ev)
    S0_ref = ex1["launch"]
    fin = ex1["final"]
    ex2 = W.traverse(fm0, R_REACT, fin[:, :n].astype(np.float64),
                     np.random.default_rng(cfg["seed"] + 7011), cfg["sigma_perf"], led,
                     who="instrument", kind="ref_lap2", qd0=fin[:, n:].astype(np.float64))
    cmd_ref = np.concatenate([ex1["acts"], ex2["acts_app"], ex2["acts"]], axis=1)
    raw_ref = np.concatenate([ex1["acts_raw"], ex2["acts_app_raw"], ex2["acts_raw"]], axis=1)
    H_REF = cmd_ref.shape[1]
    # THE TRUTH IS THE FLOWN TRAJECTORY, not a replay of it. `traverse` now records the tip after
    # every control step (additive `tips` / `tips_app`), so the FM rollout is scored against what
    # the body actually did. The alternative -- `true_tips(S0_ref, cmd_ref)`, G5a's construction --
    # re-enters the plant through `set_state` from a float32 state, and the arm amplifies that
    # ~1e-7 seed over tens of steps. That amplification is itself worth knowing (it bounds how
    # much of ANY measured divergence is the plant's own sensitivity rather than model error), so
    # it is recorded below as the `replay` floor rather than assumed away.
    true_ref = np.concatenate([ex1["tips"], ex2["tips_app"], ex2["tips"]], axis=1)
    tips_replay = W.true_tips(S0_ref, cmd_ref)
    repro = float(np.median(np.linalg.norm(tips_replay - true_ref, axis=2)[:, -1]))
    P(f"[ref] H_REF={H_REF} steps ({W.H_phrase}+{W.H_app}+{W.H_phrase}) | issued-command replay "
      f"drifts {repro:.2e} m from the flown trajectory by step {H_REF} | lap1 e_piece="
      f"{float(np.median(ex1['e_piece'])):.4f} lap2 e_piece="
      f"{float(np.median(ex2['e_piece'])):.4f}")

    # ---- the piece, as per-step references -------------------------------------------------
    SCHED = W.goal_schedule(0, NS)                                # (H_phrase, 2)

    def leg_ideal(a, b, h):
        t = (np.arange(1, h + 1) / float(h))[:, None]
        return a[None].astype(np.float64) + t * (b - a)[None].astype(np.float64)

    IDEAL = np.concatenate([leg_ideal(W.goals[k], W.goals[k + 1], W.H_seg[k])
                            for k in range(NS)], 0)               # (H_phrase, 2)
    # the two-lap extension: phrase, then the approach leg W0 -> W1, then the phrase again
    IDEAL_REF = np.concatenate([IDEAL, leg_ideal(W.goals[NS], W.goals[0], W.H_app), IDEAL], 0)
    SCHED_REF = np.concatenate([SCHED, np.tile(W.goals[0][None, :], (W.H_app, 1)), SCHED], 0)
    seg_of = np.concatenate([np.full(W.H_seg[k], k) for k in range(NS)])
    LEG_A = np.stack([W.goals[k] for k in seg_of]).astype(np.float64)
    LEG_B = np.stack([W.goals[k + 1] for k in seg_of]).astype(np.float64)

    def path_err(tips, a=None, b=None):
        """Perpendicular distance from each tip to the leg it should be travelling along."""
        a = LEG_A[None] if a is None else a[None]
        b = LEG_B[None] if b is None else b[None]
        ab = b - a
        t = np.clip(((tips - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-12), 0.0, 1.0)
        return np.linalg.norm(tips - (a + t[..., None] * ab), axis=-1)

    def cross(curve, thr):
        """G5a's rule, unchanged: 1-based index of the first step above `thr`; len+1 if never."""
        idx = np.nonzero(np.asarray(curve) > thr)[0]
        return int(idx[0] + 1) if len(idx) else int(len(curve) + 1)

    def curve_row(d):
        med = np.median(d, 0)
        return dict(med=[float(v) for v in med],
                    h={f"{t:g}": cross(med, t) for t in THRS})

    # ---- the floors, measured once ----------------------------------------------------------
    # A MEASURED chain re-executed open-loop is `legato`'s frozen content: the body produced it, so
    # it carries no composition error at all and its only cost is motor noise. That is the floor
    # any live plan is competing against, and it is what says whether an executed-span curve is
    # model error or noise.
    frng = np.random.default_rng(cfg["seed"] + 7101)
    noisy_raw = np.clip(raw_ref + cfg["sigma_perf"] * frng.standard_normal(raw_ref.shape), -1, 1)
    tips_floor = W.true_tips(S0_ref, noisy_raw.astype(np.float32))
    tips_floor0 = W.true_tips(S0_ref, raw_ref)
    floors = {}
    for nm, tp in (("frozen_noisy", tips_floor), ("frozen_clean", tips_floor0),
                   ("replay", tips_replay), ("reference", true_ref)):
        floors[nm] = dict(
            ref=curve_row(np.linalg.norm(tp - true_ref, axis=2)),
            goal=curve_row(np.linalg.norm(tp - SCHED_REF[None], axis=2)),
            goal_excess=curve_row(np.linalg.norm(tp - SCHED_REF[None], axis=2)
                                  - np.linalg.norm(true_ref - SCHED_REF[None], axis=2)),
            ideal=curve_row(np.linalg.norm(tp - IDEAL_REF[None], axis=2)),
            path=curve_row(path_err(tp[:, :W.H_phrase])))
    out["floors"] = floors
    out["reference"] = dict(
        H_ref=H_REF, h_phrase=W.H_phrase, h_app=W.H_app, seg_hi=[int(x) for x in W.seg_hi],
        replay_repro=repro, n_perf=int(len(S0_ref)),
        e_piece_lap1=float(np.median(ex1["e_piece"])),
        e_piece_lap2=float(np.median(ex2["e_piece"])),
        launch_tip=W.tip(S0_ref).tolist(),
        launch_tip_spread=float(np.linalg.norm(W.tip(S0_ref).std(0))),
        sched=SCHED_REF.tolist(), ideal=IDEAL_REF.tolist())
    P(f"[floor] frozen-chain open-loop with fresh noise: vs-ref h@0.05="
      f"{floors['frozen_noisy']['ref']['h']['0.05']} vs-ideal h@0.05="
      f"{floors['frozen_noisy']['ideal']['h']['0.05']} | chaos (replay of the SAME issued "
      f"commands): vs-ref h@0.05={floors['replay']['ref']['h']['0.05']} | the REFERENCE traversal "
      f"itself: vs-ideal h@0.05={floors['reference']['ideal']['h']['0.05']} vs-goal med@seam="
      f"{[round(floors['reference']['goal']['med'][int(x) - 1], 4) for x in W.seg_hi]}")
    save()

    # ================================================================= the two probes
    def imagination(net, S0=None, cmds=None, truth=None):
        """G5a's divergence: open-loop FM rollout vs the plant, along executed commands."""
        S0 = S0_ref if S0 is None else S0
        cmds = cmd_ref if cmds is None else cmds
        truth = true_ref if truth is None else truth
        pred = W.fm_rollout_tips(net, S0, cmds)
        return curve_row(np.linalg.norm(pred - truth, axis=2))

    def executed_span(net, seed):
        """One live phrase plan from W1 under `net`, flown open-loop on the body with performance
        motor noise. `legato`'s `phrase_plan_launch` content, read as a per-step curve."""
        rng = np.random.default_rng(seed)
        pf = W.plan_fn(net, W.H_phrase, vel_pen=W.vel_pen_for(0, NS),
                       wp_mask=W.waypoint_mask(0, W.H_phrase), **CALP[NS])
        cmds, delib = pf(S0_ref, np.tile(SCHED[None], (len(S0_ref), 1, 1)), rng)
        cmds = np.clip(cmds, -1, 1)
        noisy = np.clip(cmds + cfg["sigma_perf"] * rng.standard_normal(cmds.shape), -1, 1)
        tips = W.true_tips(S0_ref, noisy.astype(np.float32))
        e_seam = [float(np.median(np.linalg.norm(
            tips[:, int(x) - 1] - W.goals[k + 1][None, :], axis=1)))
            for k, x in enumerate(W.seg_hi)]
        # PAIRED excess over the reference: how much further from the step's own waypoint is this
        # open-loop rendition than the competent reactive traversal FROM THE SAME LAUNCH STATE?
        # The raw `goal` curve is a sawtooth by construction (each step is scored against its
        # segment's endpoint, and the cost rewards arriving early and dwelling), so its level says
        # nothing on its own; the excess has a natural zero and the reference supplies it.
        gx = (np.linalg.norm(tips - SCHED[None], axis=2)
              - np.linalg.norm(true_ref[:, :W.H_phrase] - SCHED[None], axis=2))
        return dict(
            ref=curve_row(np.linalg.norm(tips - true_ref[:, :W.H_phrase], axis=2)),
            goal=curve_row(np.linalg.norm(tips - SCHED[None], axis=2)),
            goal_excess=curve_row(gx),
            ideal=curve_row(np.linalg.norm(tips - IDEAL[None], axis=2)),
            path=curve_row(path_err(tips)),
            e_seam=e_seam, e_piece=float(np.mean(e_seam)), delib=int(delib))

    # the anchors: the two models `legato` G5a already measured, on THIS reference set
    anchors = {}
    for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
        anchors[nm] = dict(imagination=imagination(net),
                           span=executed_span(net, cfg["seed"] + 7200))
        a = anchors[nm]
        P(f"[anchor/{nm:7s}] imagination h@0.05={a['imagination']['h']['0.05']:3d} "
          f"h@0.02={a['imagination']['h']['0.02']:3d} | executed span excess h@0.05="
          f"{a['span']['goal_excess']['h']['0.05']:3d} path h@0.05="
          f"{a['span']['path']['h']['0.05']:3d} vs-ref h@0.05="
          f"{a['span']['ref']['h']['0.05']:3d} | e_seam "
          f"{[round(v, 4) for v in a['span']['e_seam']]}")
    out["anchors"] = anchors
    save()

    # ================================================================= schedules
    def span_cycles(nc):
        """Dense where the horizon should be EMERGING, sparse once it is only drifting."""
        d, m, me, le = (cfg["span_dense"], cfg["span_mid"], cfg["span_mid_every"],
                        cfg["span_late_every"])
        s = set(range(1, min(d, nc) + 1))
        s |= {c for c in range(d + 1, min(m, nc) + 1) if (c - d) % me == 0}
        s |= {c for c in range(m + 1, nc + 1) if (c - m) % le == 0}
        s.add(nc)
        return sorted(c for c in s if 1 <= c <= nc)

    SPAN_AT = set(span_cycles(cfg["n_cycles"]))
    P(f"[sched] executed-span probes at {sorted(SPAN_AT)}")

    # ================================================================= the practice loop
    # `legato`'s loop with the commitment machinery and the metering removed. Nothing here is
    # committed, so the practice traversal is fully reactive at practice tempo -- exactly the
    # `never` arm, which every other `legato` arm is bit-identical to before its first commit.
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    buf = []
    brng = np.random.default_rng(cfg["seed"] + 800)
    snaps, ladder, weights = [], [], {}
    log = {k: [] for k in ("cycle", "t_cum", "e_practice", "steps_agent")}

    def ladder_probe(cycle):
        rec = {"cycle": cycle, "t_cum": led.t_priced}
        for nm, rt in (("react", R_REACT), ("ball_seg", R_BALL_SEG)):
            o = W.traverse(fm, rt, q_ev, np.random.default_rng(cfg["seed"] + 7050 + cycle),
                           cfg["sigma_perf"], led, who="instrument", kind="ladder",
                           approach_plan=plan_ev)
            rec[f"e_{nm}"] = float(np.median(o["e_piece"]))
            rec[f"e_{nm}_by_seg"] = [float(np.median(o["e_seg"][:, k])) for k in range(NS)]
            if nm == "react":
                # has the launch distribution drifted away from the frozen reference set?
                rec["launch_drift"] = float(np.median(np.linalg.norm(
                    W.tip(o["launch"]) - W.tip(S0_ref), axis=1)))
                rec["launch_tip_spread"] = float(np.linalg.norm(W.tip(o["launch"]).std(0)))
        return rec

    def snapshot(cycle, pr=None):
        rec = {"cycle": cycle, "t_cum": led.t_priced, "imagination": imagination(fm)}
        if pr is not None:
            # the SAME probe on the commands the body executed THIS cycle, at practice tempo.
            # Separates "the model got better on a fixed corridor" from "the corridor narrowed".
            rec["imagination_onpolicy"] = imagination(
                fm, pr["launch"], pr["acts"], pr["tips"])
        if cycle in SPAN_AT or cycle == 0:
            rec["span"] = executed_span(fm, cfg["seed"] + 7300 + cycle)
        return rec

    def report(rec):
        im = rec["imagination"]["h"]
        sp = rec.get("span")
        msg = (f"[c{rec['cycle']:3d}] imagination h@0.05={im['0.05']:3d} h@0.02={im['0.02']:3d}")
        if "imagination_onpolicy" in rec:
            msg += f" (on-policy {rec['imagination_onpolicy']['h']['0.05']:3d})"
        if sp:
            msg += (f" | span excess h@0.05={sp['goal_excess']['h']['0.05']:3d} "
                    f"path h@0.05={sp['path']['h']['0.05']:3d} "
                    f"vs-ref h@0.05={sp['ref']['h']['0.05']:3d} e_piece={sp['e_piece']:.4f}")
        if rec.get("wall") is not None:
            msg += f" | {rec['wall']:.1f}s/cycle"
        P(msg)

    snaps.append(snapshot(0))
    ladder.append(ladder_probe(0))
    report(snaps[-1])
    weights[0] = {k: v.detach().cpu().clone() for k, v in fm.state_dict().items()}

    def dump_weights():
        torch.save({"norm": {k: (v.cpu() if hasattr(v, "cpu") else v)
                             for k, v in W.norm.items()},
                    "cfg": {k: cfg[k] for k in ("fm_hidden", "fm_layers", "seed", "tag")},
                    "weights": weights}, os.path.join(outdir, "fm_snapshots.pt"))
        volume.commit()

    t_cycle = time.time()
    for cycle in range(1, cfg["n_cycles"] + 1):
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + cycle)
        pr = W.traverse(fm, R_REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + cycle),
                        cfg["sigma_practice"], led, who="agent", kind="practice",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + cycle), collect=True)
        buf.append(pr["trans"])
        if len(buf) > cfg["trace_window"]:
            buf.pop(0)
        PX, PY = W.tensors(np.concatenate([b[0] for b in buf]),
                           np.concatenate([b[1] for b in buf]),
                           np.concatenate([b[2] for b in buf]))
        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])
        log["cycle"].append(cycle); log["t_cum"].append(led.t_priced)
        log["e_practice"].append(float(np.median(pr["e_piece"])))
        log["steps_agent"].append(led.steps["agent"])

        snaps.append(snapshot(cycle, pr))
        snaps[-1]["wall"] = time.time() - t_cycle
        t_cycle = time.time()
        report(snaps[-1])
        if cycle % cfg["probe_every"] == 0 or cycle == cfg["n_cycles"]:
            ladder.append(ladder_probe(cycle))
            L = ladder[-1]
            P(f"       ladder e_react={L['e_react']:.4f} e_ball_seg={L['e_ball_seg']:.4f} "
              f"launch_drift={L['launch_drift']:.4f} t={led.t_priced:9.1f}s")
        if cycle % cfg["weight_every"] == 0 or cycle == cfg["n_cycles"]:
            weights[cycle] = {k: v.detach().cpu().clone() for k, v in fm.state_dict().items()}
        if cycle % cfg["save_every"] == 0 or cycle == cfg["n_cycles"]:
            out.update(log=log, snapshots=snaps, ladder=ladder, ledger=led.snapshot())
            save()
            dump_weights()

    out.update(log=log, snapshots=snaps, ladder=ladder, ledger=led.snapshot(), complete=True)
    save()
    dump_weights()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[done] n_snapshots={len(snaps)} n_ladder={len(ladder)} t_priced={led.t_priced:.1f}s")
    return out


@app.local_entrypoint()
def span(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # ---- the world: `legato`'s L1 design point, unchanged
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
    # ---- controllers: the L1 chosen cost cell (gate `l2`), so the horizon is measured under the
    # same objective G5a measured 23 / 58 under.
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.0,
    vel_pen_mid: float = 0.0,
    w_waypoint: float = 16.0,
    lookahead_gamma: float = 0.0,
    commit_lookahead: int = 0,
    react_look: int = 20,
    plan_max_elems: int = 40_000_000,
    calp: str = "1:1024:8,2:4096:12,3:4096:12",
    # ---- the loop
    n_cycles: int = 90,
    batch: int = 24,
    n_eval: int = 32,
    probe_every: int = 3,
    trace_window: int = 6,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    # ---- probe schedules. Imagination runs EVERY cycle (one MLP rollout). The executed span costs
    # a phrase-sized CEM call, so: every cycle to `span_dense`, every `span_mid_every` to
    # `span_mid`, every `span_late_every` after.
    span_dense: int = 12,
    span_mid: int = 30,
    span_mid_every: int = 2,
    span_late_every: int = 3,
    weight_every: int = 5,
    save_every: int = 3,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500; n_corridor = 3
        fm_steps = 1200; fm_steps_boot = 600
        n_cycles = 6; batch = 8; n_eval = 12; probe_every = 2; trace_window = 3
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,2:256:4,3:256:4"
        span_dense = 3; span_mid = 4; span_mid_every = 1; span_late_every = 2
        weight_every = 3; save_every = 2
        tag = tag or "smoke"
    tag = tag or "default"
    cfg = dict(
        tag=tag, seed=seed, curl_b=curl_b, push_a=push_a,
        waypoints=[[float(v) for v in p.split(",")] for p in waypoints.split(";") if p],
        h_app=h_app, h_seg=[int(v) for v in h_seg.split(",")], patch_seg=patch_seg,
        q_jit=q_jit, null_jit=null_jit, start_mode=start_mode, patch_sigma=patch_sigma,
        patch_center=([float(x) for x in patch_center.split(",")] if patch_center else None),
        push_sigma=(push_sigma or patch_sigma), push_center=None,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, timestep=timestep,
        wrap_limit=wrap_limit,
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
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, vel_pen_mid=vel_pen_mid,
        w_waypoint=w_waypoint, lookahead_gamma=lookahead_gamma,
        commit_lookahead=commit_lookahead, react_look=react_look,
        plan_max_elems=plan_max_elems,
        calp={p.split(":")[0]: [int(p.split(":")[1]), int(p.split(":")[2])]
              for p in calp.split(",") if p},
        n_cycles=n_cycles, batch=batch, n_eval=n_eval, probe_every=probe_every,
        trace_window=trace_window, sigma_practice=sigma_practice, sigma_perf=sigma_perf,
        d_fb=d_fb, span_dense=span_dense, span_mid=span_mid, span_mid_every=span_mid_every,
        span_late_every=span_late_every, weight_every=weight_every, save_every=save_every,
    )
    o = run_span.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {localdir}/results.json")
