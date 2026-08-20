"""Round 2b -- committed execution across seams: does frozen, measured content re-enter at phrase
span?

THE QUESTION, IN ONE LINE. Round 1 found that within the plant's composition horizon a committed
unit should carry LIVE content -- one CEM plan from the current forward model at the observed launch
state, flown open-loop, beat every frozen op 1.6x on error at 2.3x less priced time, because a
frozen unit's only channel to behaviour is reselection and it rots as the practice diet narrows. A
PHRASE breaks the premise: three segments is 60 steps, ~3x past where a one-step forward model's
rolled-out tip trajectory is trustworthy, so there is no live plan of the whole phrase to be had. A
chain of measured, EXECUTED renditions has no such bound -- the body produced it, not the model --
so it is limited by execution variance rather than by model composition. The hypothesis is that
frozen content re-enters exactly there.

THE 2x2 (plus the reference). GRANULARITY x CONTENT, with routing and content varied independently,
which is what makes each margin attributable:

                       | live content                 | frozen content
    -------------------+------------------------------+-------------------------------
    per segment (3 fb) | `seg_plan_launch`            | `seg_frozen`
                       | round 1's champion           | keyed library per seam
    -------------------+------------------------------+-------------------------------
    per phrase  (1 fb) | `phrase_plan_launch`         | `phrase_frozen`
                       | the STRUCTURAL-FAILURE arm   | THE HYPOTHESIS ARM
    reference          | `never` -- reactive MPC throughout, 60 delays and 60 deliberations
                       | per traversal, forever. Round 1's outright winner on one easy segment.

`seg_frozen` is not decoration: it is the ONLY control that makes the fusion claim attributable.
`phrase_frozen` differs from it by exactly two things -- two fewer feedback events per traversal,
and no mid-flight re-keying -- so their difference IS the fusion bet, priced.

WHY FUSION IS NOT VACUOUS HERE, WHERE THE ETUDE PROVED IT WAS. The etude's committed units were
STATE-INDEPENDENT command sequences, so two adjacent units concatenated to a bit-identical
trajectory and removing the internal re-grounding was a no-op on the physics: accuracy cost exactly
zero, time win exactly one `d_fb`. Round 1 established that on this plant boundaries carry
information (hand-over R^2 = 0.82 on a committed unit's realised error), so the units here are
state-CONDITIONED -- keyed at each seam by the observed arrival. A fused unit cannot re-key mid
flight; it must choose its whole chain at the phrase launch. That is a real bet with a real price,
and G6 measures the information it gives up before the arms are ever run.

WHERE THE PHRASE CANDIDATES COME FROM -- the etude E-4 pattern, one level up. Segments are committed
IN PIECE ORDER at the clocks G7 measured (bottom-heavy: segments mastered before phrases), each
scored on launch states produced by the CURRENT performance configuration. Once all three are
committed, the practice traversals ARE renditions of the whole phrase, and because the segment units
are state-conditioned, different traversals fire different key combinations -- so the phrase pool is
a pool of genuinely distinct MEASURED chains, not the pool of one that made the etude's escape hatch
collapse. The phrase unit is then a library over those chains, keyed once, at the phrase launch.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/legato/legato.py::legato --quick
    python3 mjc/practice/legato/launch_detached.py --fn legato --tag L1 --seed 0
    python3 mjc/practice/legato/analyze_legato.py --tag L1 --fetch
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.legato.gates import DEF_WPS, DEF_PATCH_SEG

# `gran`  -- what one committed group spans: "seg" (one group per segment) or "phrase" (one group).
# `content` -- what the group issues: None (never commits), "plan_launch" (live), "library" /
#              "fixed" (frozen, selected by consumption-matched audition).
ARM_TABLE = {
    "never":               dict(gran=None, content=None),
    "seg_plan_launch":     dict(gran="seg", content="plan_launch"),
    "seg_frozen":          dict(gran="seg", content="library"),
    "phrase_plan_launch":  dict(gran="phrase", content="plan_launch"),
    "phrase_frozen":       dict(gran="phrase", content="library"),
    # the state-independence ablation: the same fused chain for everybody. If `phrase_frozen` does
    # not beat this, the one feedback event a phrase commitment pays for is again buying nothing --
    # which is the etude's diagnosis reappearing one level up, and worth knowing.
    "phrase_chain_fixed":  dict(gran="phrase", content="fixed"),
    # The key-extractor variants, held in reserve for the gate report to license. Round 1's k-means
    # library recovered only 1.07x of a 4.0x per-state oracle gain with 2 of 4 cells holding
    # DUPLICATE picks, while a linear function of the launch state explained 82% of the realised
    # error variance -- the information was there and isotropic k-means had no reason to align with
    # it. `select_library_proj` fits the key from the audition matrix the arm has already paid for,
    # so it costs nothing extra. Run these only if G6 says the phrase-launch R^2 supports it.
    "seg_frozen_proj":     dict(gran="seg", content="library_proj"),
    "phrase_frozen_proj":  dict(gran="phrase", content="library_proj"),
}
FROZEN = ("library", "library_proj", "fixed")


@app.function(gpu="L4", memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_legato(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler, pool_diagnostics, cmd_state_corr
    from mjc.practice.legato.world import World, Ledger, start_postures, elite_for

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    n, NS = W.n, W.n_seg
    arm = cfg["arm"]
    spec = ARM_TABLE[arm]
    gran, content = spec["gran"], spec["content"]
    led = Ledger()
    out = {"config": cfg, "arm": arm, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_legato", cfg["tag"], arm)
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print(f"[{arm}]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # Planner sizes come from the gate's CAL-P, one per span, so a phrase plan is not handicapped by
    # being handed a segment-sized search (`arm_substrate` P3: an undersized CEM makes ballistic
    # error planning-noise-dominated and can sign-invert the FM-quality axis). They are passed in as
    # config so the calibration that set them is on the record with the run.
    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}
    commit_seg = [int(c) for c in cfg["commit_seg"]]
    commit_phrase = int(cfg["commit_phrase"])

    P(f"[setup] device={device} gran={gran} content={content} H_seg={W.H_seg} "
      f"H_phrase={W.H_phrase} cycles={cfg['n_cycles']} commit_seg={commit_seg} "
      f"commit_phrase={commit_phrase} calp={cfg['calp']} interleave={cfg['interleave_period']}")

    # ================================================================= diet + models
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

    q_rt = geom(cfg["n_rt"], cfg["seed"] + 5000)
    q_sc = geom(cfg["n_score"], cfg["seed"] + 5100)
    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)
    def app_plan(q, seed):
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(s, np.tile(W.goals[0][None, :], (len(q), 1)),
                  np.random.default_rng(seed))[0]

    plan_rt = app_plan(q_rt, cfg["seed"] + 6000)
    plan_sc = app_plan(q_sc, cfg["seed"] + 6100)
    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    R_REACT = W.routing("reactive")
    R_BALL_SEG = W.routing("plan_launch", groups=[1] * NS, **CALP[1])
    R_BALL_PHRASE = W.routing("plan_launch", groups=[NS], **CALP[NS])

    # corridor-matched ceiling reference (oracle instrument, free, identical across arms)
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

    # ================================================================= references + the scale
    def meter(fm, routing, seed, who="agent", replan_every=1):
        return W.traverse(fm, routing, q_rt, np.random.default_rng(seed), cfg["sigma_perf"], led,
                          who=who, kind="metering", approach_plan=plan_rt,
                          replan_every=replan_every)

    ref_stale = float(np.median(meter(fm0, R_BALL_SEG, cfg["seed"] + 7200,
                                      who="instrument")["e_piece"]))
    ref_ceil = float(np.median(meter(fm_ceil, R_BALL_SEG, cfg["seed"] + 7300,
                                     who="instrument")["e_piece"]))
    ref_react = float(np.median(meter(fm0, R_REACT, cfg["seed"] + 7400,
                                      who="instrument")["e_piece"]))
    out["setup"] = dict(ref_stale=ref_stale, ref_ceiling=ref_ceil, ref_reactive=ref_react,
                        usable_range=ref_stale - ref_ceil, exclude_keep_frac=keep_frac,
                        corr_reach=cmd_state_corr(S2_, U2_), pool_diag=pool_diagnostics(Se, Ue, n))
    P(f"[ref] piece @tempo: stale(ball_seg) {ref_stale:.4f}  ceiling {ref_ceil:.4f}  "
      f"reactive {ref_react:.4f}  range {ref_stale - ref_ceil:.4f}")
    save()

    # ================================================================= the practice loop
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    buf, trace_buf = [], []
    brng = np.random.default_rng(cfg["seed"] + 800)
    seg_units = [None] * NS            # committed unit per segment (frozen or live)
    phrase_unit = None
    n_interleaved = n_reselect = 0
    events, ladder = [], []
    log = {k: [] for k in ("cycle", "t_cum", "e_rt", "e_practice", "committed", "steps_agent",
                           "plans_agent", "delib_agent")}

    def perf_routing():
        """What this arm actually plays at performance tempo -- the routing its commitments imply.

        Until a segment is committed the arm plays it reactively, so an arm is a *pure* function of
        which commitments have fired: every arm runs a bit-identical stream up to its first commit,
        which is the etude's `sched_late` property and what makes a single seed readable.
        """
        if phrase_unit is not None:
            return W.routing(None, groups=[NS], units=[phrase_unit])
        units, groups = [], []
        for k in range(NS):
            units.append(seg_units[k] if seg_units[k] is not None else W.reactive_unit())
            groups.append(1)
        return W.routing(None, groups=groups, units=units)

    def n_committed():
        return int(phrase_unit is not None) * NS + sum(u is not None for u in seg_units)

    def ladder_probe(cycle):
        """Held-out grade -- an experimenter instrument, free, and labelled as such. Three readouts:
        this arm's OWN performance configuration (the headline), plus reactive and
        ballistic-per-segment under a common execution, so the arms are also comparable at matched
        control mode."""
        rec = {"cycle": cycle, "t_cum": led.t_priced, "n_committed": n_committed(),
               "steps_agent": led.steps["agent"], "plans_agent": led.plans["agent"],
               "delib_agent": led.delib["agent"]}
        o = W.traverse(fm, perf_routing(), q_ev, np.random.default_rng(cfg["seed"] + 7000 + cycle),
                       cfg["sigma_perf"], led, who="instrument", kind="ladder",
                       approach_plan=plan_ev)
        rec["e_perf"] = float(np.median(o["e_piece"]))
        rec["e_perf_mean"] = float(o["e_piece"].mean())
        rec["e_perf_by_seg"] = [float(np.median(o["e_seg"][:, k])) for k in range(NS)]
        rec["e_app"] = float(np.median(o["e_app"]))
        rec["n_fb_perf"] = int(o["n_fb"])
        rec["t_piece_perf"] = o["t_piece"]
        for nm, rt in (("react", R_REACT), ("ball_seg", R_BALL_SEG),
                       ("ball_phrase", R_BALL_PHRASE)):
            oo = W.traverse(fm, rt, q_ev, np.random.default_rng(cfg["seed"] + 7050 + cycle),
                            cfg["sigma_perf"], led, who="instrument", kind="ladder",
                            approach_plan=plan_ev)
            rec[f"e_{nm}"] = float(np.median(oo["e_piece"]))
        return rec

    def score_set(cycle, seg_lo, n_score):
        """Launch states for a span, produced by the CURRENT performance configuration at
        performance tempo, truncated at the span's start (the etude E-4 seam-matched pattern).
        Charged: the body has to play up to there to find out where it arrives."""
        o = W.traverse(fm, perf_routing(), q_sc[:n_score],
                       np.random.default_rng(cfg["seed"] + 4400 + cycle), cfg["sigma_perf"], led,
                       who="agent", kind="score_set", approach_plan=plan_sc[:n_score],
                       stop_seg=seg_lo)
        return (o["launch"] if seg_lo == 0 else o["launches"][seg_lo])

    def candidates(seg_lo, n_segs, n_cand, cycle, reactive_only=False):
        """Uniform draw from the recent trace pool, sliced to the span. NEVER top-of-pool: ranking a
        candidate by its own realised error is a winner's curse that favours the rendition most
        finely tuned to its own start state, i.e. the least transferable one (etude E-3b;
        crystallize measures the same curse at 3.3x)."""
        src = [t for t in trace_buf if t[3]] if reactive_only else list(trace_buf)
        src = src or list(trace_buf)
        acts = np.concatenate([t[2] for t in src])
        lo, hi = int(W.seg_lo[seg_lo]), int(W.seg_hi[seg_lo + n_segs - 1])
        ci = np.random.default_rng(cfg["seed"] + 953 + cycle).permutation(len(acts))[:n_cand]
        return acts[ci][:, lo:hi], int(len(acts)), int(sum(1 for t in src if t[3]))

    def compile_unit(cycle, seg_lo, n_segs, n_cand, n_score, reason, reactive_only=False):
        """The compile op: audition uniform candidates against a seam-matched score set, then commit
        either the state-conditioned library (`content == "library"`) or the single argmin-of-mean
        unit (`content == "fixed"`). Both read the SAME audition matrix at the same price, so the
        library's advantage is information use, not budget."""
        S0 = score_set(cycle, seg_lo, n_score)
        cands, n_pool, n_react = candidates(seg_lo, n_segs, n_cand, cycle, reactive_only)
        cps = W.checkpoints(seg_lo, n_segs)
        E, Efull = W.audition(cands, S0, cps, led, who="agent")
        info = dict(seg_lo=seg_lo, n_segs=n_segs, span=int(W.span(seg_lo, n_segs)),
                    n_pool=n_pool, n_reactive_traces=n_react, n_cand=int(len(cands)),
                    n_score=int(len(S0)), score_med=float(np.median(E.mean(1))),
                    best_fixed=float(E.mean(1).min()), per_state_oracle=float(E.min(0).mean()),
                    oracle_gain=float(E.mean(1).min() / max(E.min(0).mean(), 1e-9)),
                    launch_tip_spread=float(np.linalg.norm(W.tip(S0).std(0))),
                    launch_speed=float(np.linalg.norm(S0[:, n:], axis=1).mean()),
                    pool_geometry=World.pool_geometry(cands),
                    per_checkpoint_best=[float(Efull[:, :, c].mean(1).min())
                                         for c in range(Efull.shape[2])])
        if content in ("library", "library_proj"):
            if content == "library":
                u, li = W.select_library(E, cands, S0, cfg["n_lib"],
                                         np.random.default_rng(cfg["seed"] + 954 + cycle))
            else:
                u, li = W.select_library_proj(E, cands, S0, cfg["n_lib"])
            info.update(library={k: v for k, v in li.items() if k != "labels"},
                        chosen_score=float(np.mean(
                            [E[li["picks"][int(l_)], i] for i, l_ in enumerate(li["labels"])])))
            if li["n_distinct"] == 1:
                # A library whose cells all pick the SAME candidate is a `fixed` unit wearing a
                # library's name, and the fusion comparison would be vacuous without anyone
                # noticing. Round 1 hit exactly this (2 of 4 duplicate cells at round 0). Loud.
                P(f"[WARN] library at cycle {cycle} span {info['span']} collapsed to ONE distinct "
                  f"pick (cells {li['cell_sizes']}) -- state-conditioning is inert here")
        elif content == "fixed":
            u, j = World.select_fixed(E, cands)
            info.update(chosen_score=float(E[j].mean()))
        else:
            raise ValueError(content)
        events.append(dict(kind="commit", cycle=cycle, arm=arm, reason=reason, content=content,
                           t_cum=led.t_priced, **info))
        P(f"[commit] {reason} cycle={cycle} span={info['span']} pool={n_pool} "
          f"score={info['chosen_score']:.4f} best_fixed={info['best_fixed']:.4f} "
          f"oracle={info['per_state_oracle']:.4f} ({info['oracle_gain']:.2f}x) "
          f"launch_spread={info['launch_tip_spread']:.4f}")
        return u

    ladder.append(ladder_probe(0))

    for cycle in range(1, cfg["n_cycles"] + 1):
        # ---- (a) practice traversal, at practice tempo, with motor noise ----
        # THE INTERLEAVE (round 1b's diet-rent fix): once anything is committed, one practice cycle
        # in `interleave_period` is an honest fully-REACTIVE traversal -- fully priced, and the
        # forward model trains on it exactly as usual -- while metering and run-throughs keep
        # routing the committed configuration. Round 1 measured rot onset at 7-10 consecutive
        # pure-committed cycles, so a period of 3 keeps a ~3x margin. It also keeps improvised
        # renditions flowing into the candidate pool, which is what a reselect needs.
        committed_any = n_committed() > 0
        interleaved = bool(committed_any and cfg["interleave_period"] > 0
                           and cycle % cfg["interleave_period"] == 0)
        pr_routing = R_REACT if interleaved else perf_routing()
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + cycle)
        pr = W.traverse(fm, pr_routing, qp, np.random.default_rng(cfg["seed"] + 10_000 + cycle),
                        cfg["sigma_practice"], led, who="agent",
                        kind="practice_interleave" if interleaved else "practice",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + cycle), collect=True)
        n_interleaved += int(interleaved)
        buf.append(pr["trans"])
        # the 4th field tags a trace as fully REACTIVE, so a reselect can draw only from renditions
        # the agent actually improvised
        trace_buf.append((pr["launch"].copy(), {k: v.copy() for k, v in pr["launches"].items()},
                          pr["acts"].copy(), bool(interleaved or not committed_any),
                          pr["e_seg"].copy()))
        for b_ in (buf, trace_buf):
            if len(b_) > cfg["trace_window"]:
                b_.pop(0)

        # ---- (b) FM plasticity: plain uniform lr, no per-sample delta gain anywhere ----
        PX, PY = W.tensors(np.concatenate([b[0] for b in buf]),
                           np.concatenate([b[1] for b in buf]),
                           np.concatenate([b[2] for b in buf]))
        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])

        # ---- (c) metering: the at-tempo run-through the agent reads (charged) ----
        rt = meter(fm, perf_routing(), cfg["seed"] + 4000 + cycle)
        e = float(np.median(rt["e_piece"]))

        # ---- (d) commitment: SEQUENTIAL and BOTTOM-HEAVY, at cycles read off the gate's clocks ----
        # Segments commit in piece order; the phrase may only fuse once every segment under it is
        # committed. `plan_launch` content has nothing to select, so it commits by routing alone --
        # and charging it a phantom audition to "match budgets" would be dishonest in the other
        # direction. Its cheapness is a real property of the op, and the deliberation axis is where
        # it pays for it.
        if gran == "seg":
            for k in range(NS):
                if seg_units[k] is None and cycle >= commit_seg[k] and (
                        k == 0 or seg_units[k - 1] is not None):
                    if content == "plan_launch":
                        seg_units[k] = W.plan_launch_unit(**CALP[1])
                        events.append(dict(kind="commit", cycle=cycle, arm=arm, seg_lo=k, n_segs=1,
                                           span=W.H_seg[k], content=content, t_cum=led.t_priced,
                                           reason="schedule", chosen_score=None,
                                           note="live content: routing commits, content is planned "
                                                "at launch under the current FM; no audition"))
                        P(f"[commit] schedule cycle={cycle} seg={k} (routing committed; content "
                          f"stays live)")
                    else:
                        seg_units[k] = compile_unit(cycle, k, 1, cfg["n_cand"], cfg["n_score"],
                                                    f"schedule_seg{k}")
                    break                      # at most one commit per cycle: each perturbs the
                                               # downstream boundary distribution (etude E-5)
        elif gran == "phrase":
            if content == "plan_launch":
                if phrase_unit is None and cycle >= commit_phrase:
                    phrase_unit = W.plan_launch_unit(**CALP[NS])
                    events.append(dict(kind="commit", cycle=cycle, arm=arm, seg_lo=0, n_segs=NS,
                                       span=W.H_phrase, content=content, t_cum=led.t_priced,
                                       reason="schedule", chosen_score=None,
                                       note="one live plan for the whole phrase"))
                    P(f"[commit] schedule cycle={cycle} PHRASE (one live plan, span "
                      f"{W.H_phrase})")
            else:
                # bottom-heavy assembly: segments first (state-conditioned, so the phrase pool holds
                # genuinely distinct measured chains), then the fusion.
                done_seg = all(u is not None for u in seg_units)
                if not done_seg:
                    for k in range(NS):
                        if seg_units[k] is None and cycle >= commit_seg[k] and (
                                k == 0 or seg_units[k - 1] is not None):
                            seg_units[k] = compile_unit(cycle, k, 1, cfg["n_cand"], cfg["n_score"],
                                                        f"schedule_seg{k}")
                            break
                elif phrase_unit is None and cycle >= commit_phrase:
                    phrase_unit = compile_unit(cycle, 0, NS, cfg["n_cand_phrase"],
                                               cfg["n_score_phrase"], "fuse_phrase")
                elif (cfg["reselect_every"] > 0 and cycle > commit_phrase
                      and (cycle - commit_phrase) % cfg["reselect_every"] == 0):
                    # `setlist`'s maintenance op, at phrase level: audit and re-select from the
                    # REFRESHED pool. Deliberately a SMALLER audition than the initial fusion -- a
                    # maintenance audit, priced as one.
                    n_reselect += 1
                    phrase_unit = compile_unit(cycle, 0, NS, cfg["reselect_n_cand"],
                                               cfg["reselect_n_score"], "reselect_phrase")

        log["cycle"].append(cycle); log["t_cum"].append(led.t_priced)
        log["e_rt"].append(e); log["e_practice"].append(float(np.median(pr["e_piece"])))
        log["committed"].append(n_committed()); log["steps_agent"].append(led.steps["agent"])
        log["plans_agent"].append(led.plans["agent"])
        log["delib_agent"].append(led.delib["agent"])

        if cycle % cfg["probe_every"] == 0 or cycle == cfg["n_cycles"]:
            ladder.append(ladder_probe(cycle))
            L = ladder[-1]
            P(f"[c{cycle:3d}] e_rt={e:.4f} | perf={L['e_perf']:.4f} "
              f"by-seg {[round(v, 4) for v in L['e_perf_by_seg']]} | react={L['e_react']:.4f} "
              f"ball_seg={L['e_ball_seg']:.4f} ball_phr={L['e_ball_phrase']:.4f} | "
              f"t={led.t_priced:9.1f}s")
            out.update(log=log, ladder=ladder, events=events, ledger=led.snapshot())
            save()
        elif cycle % 5 == 0:
            P(f"[c{cycle:3d}] e_rt={e:.4f} committed={n_committed()} t={led.t_priced:9.1f}s")

    post = [x for x, c in zip(log["e_rt"], log["committed"]) if c >= NS]
    k = max(1, len(post) // 3)
    commits = [ev for ev in events if ev["kind"] == "commit"]
    out.update(log=log, ladder=ladder, events=events, ledger=led.snapshot(),
               summary=dict(
                   final_e_perf=ladder[-1]["e_perf"], final_e_react=ladder[-1]["e_react"],
                   final_e_ball_seg=ladder[-1]["e_ball_seg"],
                   final_e_ball_phrase=ladder[-1]["e_ball_phrase"],
                   final_e_perf_by_seg=ladder[-1]["e_perf_by_seg"],
                   t_priced=led.t_priced, n_commits=len(commits),
                   commit_cycles=[ev["cycle"] for ev in commits],
                   n_interleaved=n_interleaved, n_reselect=n_reselect,
                   fb_per_piece_perf=ladder[-1]["n_fb_perf"],
                   post_commit_sd=float(np.std(post)) if post else None,
                   post_commit_drift=(float(np.mean(post[-k:]) - np.mean(post[:k]))
                                      if post else None)),
               complete=True)
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[done] e_perf={out['summary']['final_e_perf']:.4f} t_priced={led.t_priced:.1f}s "
      f"fb/piece={out['summary']['fb_per_piece_perf']} drift="
      f"{out['summary']['post_commit_drift']}")
    return out


@app.local_entrypoint()
def legato(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = ("never,seg_plan_launch,seg_frozen,phrase_plan_launch,phrase_frozen,"
                 "phrase_chain_fixed"),
    # ---- the world (the gate run's design point)
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
    # ---- controllers. `calp` is "<span>:<k_shoot>:<cem_iters>,..." straight off the gate's CAL-P,
    # one planner size per committed span, so the phrase arm is not starved (P3).
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,          # terminal, only where a window reaches the piece's last waypoint
    vel_pen_mid: float = 0.0,      # every other span end -- READ OFF THE GATE'S CAL-C
    w_waypoint: float = 0.0,       # seam-step cost weight (the l1 repair) -- READ OFF CAL-C
    lookahead_gamma: float = 0.0,  # weight a live unit gives its hand-over -- READ OFF CAL-C
    commit_lookahead: int = 0,     # live units plan past their span, commit only to it -- CAL-C
    react_look: int = 20,
    plan_max_elems: int = 40_000_000,
    calp: str = "1:1024:8,2:2048:10,3:4096:12",
    # ---- practice loop. Commit cycles are read off the gate's mastery clocks (round 1b's rule:
    # first sustained entry within 1 sd of the held-out BALLISTIC plateau -- the reactive curve is
    # the wrong clock and said "mastered" 39 cycles early in round 1).
    n_cycles: int = 90,
    commit_seg: str = "20,30,40",
    commit_phrase: int = 55,
    interleave_period: int = 3,
    reselect_every: int = 15,
    reselect_n_cand: int = 32,
    reselect_n_score: int = 48,
    batch: int = 24,
    n_rt: int = 48,
    n_eval: int = 48,
    probe_every: int = 3,
    trace_window: int = 6,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    # ---- the compile op
    n_cand: int = 64,
    n_score: int = 96,
    n_cand_phrase: int = 48,
    n_score_phrase: int = 96,
    n_lib: int = 4,
    pol_hidden: int = 128,
    pol_layers: int = 2,
    pol_lr: float = 1e-3,
    pol_steps: int = 1500,
):
    arm_list = [a for a in arms.split(",") if a]
    for a in arm_list:
        if a not in ARM_TABLE:
            raise ValueError(f"unknown arm {a!r}; known: {sorted(ARM_TABLE)}")
    if quick:
        pool_ou = 1500; pool_reach = 2500; n_corridor = 3
        fm_steps = 1200; fm_steps_boot = 600
        n_cycles = 8; commit_seg = "2,3,4"; commit_phrase = 5
        batch = 8; n_rt = 12; n_eval = 12; probe_every = 2; trace_window = 3
        n_cand = 6; n_score = 12; n_cand_phrase = 6; n_score_phrase = 12; n_lib = 2
        reselect_every = 3; reselect_n_cand = 4; reselect_n_score = 8
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,2:256:4,3:256:4"; commit_lookahead = 20
        pol_steps = 300
        tag = tag or "smoke"
    tag = tag or "default"
    base = dict(
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
        n_cycles=n_cycles, commit_seg=[int(v) for v in commit_seg.split(",")],
        commit_phrase=commit_phrase, interleave_period=interleave_period,
        reselect_every=reselect_every, reselect_n_cand=reselect_n_cand,
        reselect_n_score=reselect_n_score,
        batch=batch, n_rt=n_rt, n_eval=n_eval, probe_every=probe_every,
        trace_window=trace_window, sigma_practice=sigma_practice, sigma_perf=sigma_perf,
        d_fb=d_fb, n_cand=n_cand, n_score=n_score, n_cand_phrase=n_cand_phrase,
        n_score_phrase=n_score_phrase, n_lib=n_lib,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps,
    )
    outs = list(run_legato.map([{**base, "arm": a} for a in arm_list]))
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for o in outs:
        with open(os.path.join(localdir, f"{o['arm']}.json"), "w") as fh:
            json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(outs)} arm files to {localdir}")
