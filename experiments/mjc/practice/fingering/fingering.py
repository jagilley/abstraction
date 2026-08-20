"""Round 1 -- state-conditioned commitment on the arm: `never` vs `fixed` vs `library` vs `regress`.

The question, in one line: **the étude's finding 7 said hierarchy is meaningful only over boundaries
that carry information; does giving the boundary information make committed skill better, and by
how much, on a motor plant?** `rhm/practice/crystallize/` answered it on sculpting (a library keyed
by the observed launch state beats a single state-independent unit 1.8-3.0x in 6/6 commit states,
while averaging valid realisations destroys them 3.6-5.0x). This is the motor version, with the
boundary information the étude's substrate structurally lacked -- the arm is kinematically
redundant, so the hand arrives at the same *place* in a different *configuration*.

THE FOUR ARMS. They differ ONLY in what happens at the commit cycle; up to it every arm runs a
bit-identical stream (same seeds, same practice, same FM, same audition matrix), which is the
étude's `sched_late` property and is what makes a single seed readable.

  never     reactive CEM-MPC throughout -- in practice AND in performance. Never commits, and pays
            H_drill sensorimotor delays every traversal, forever. The accuracy reference.
  fixed     the étude's E-3b op: n_cand candidates drawn UNIFORMLY from the closed-loop trace pool
            (top-of-pool is a winner's curse), each replayed open-loop from held-out
            consumption-matched hand-over states, commit the argmin of the MEAN. One unit.
  library   THE HEADLINE. The SAME audition matrix, read per cell of a k-means partition of the
            hand-over distribution; at launch the observed state picks the nearest key. Same
            candidates, same score states, same rollouts, same priced budget, same commit cycle --
            the only difference is that the feedback event already being charged at the boundary is
            actually SPENT. (The étude's own diagnosis: "the time model charges a committed unit one
            feedback event per segment, but the unit never consumes the observation that event pays
            for.")
  regress   the gradient-averaging control: a BC head on the same pool, state-conditioned like the
            library and averaging like the mean. Charged the identical audition budget so no arm
            wins on bookkeeping.
  mean      the state-independent averaging corner, completing the 2x2 (selection/averaging x
            state-independent/state-conditioned). Also charged the identical budget.

COMMITMENT TIMING is provisional at a fixed cycle, and the delta-silence certificate rides along as
an INSTRUMENT. Three consecutive frontier refusals in the RHM arc (`ratchet` L3, `ear`, `recital`)
demoted pre-commit certification; `ear` found that committing provisionally and letting consumption
grade is what worked, with the recert safety net firing 0 times in 24. So: commit at
`--commit-cycle`, log what the certificate would have done, and keep one capped recert on sustained
delta<0.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/fingering/fingering.py::fingering --quick
    python3 mjc/practice/fingering/launch_detached.py --fn fingering --tag f0 --seed 0
    python3 mjc/practice/fingering/analyze_fingering.py --tag f0 --fetch
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.fingering.gates import DEF_W1, DEF_W2

# op: what is committed at the commit cycle. `None` == never commit.
ARM_TABLE = {
    "never":           dict(op=None),
    "fixed":           dict(op="fixed"),
    "library_kmeans":  dict(op="library_kmeans"),
    "library_proj":    dict(op="library_proj"),
    "regress":         dict(op="regress"),
    "mean":            dict(op="mean"),
}
SELECT_OPS = ("fixed", "library_kmeans", "library_proj", "mean")

# --- ROUND 1b: TIMING x MAINTENANCE, with the op held fixed at `library_kmeans`.
# Round 1 settled the op question (state-conditioned selection beat state-independent selection
# 1.34x, and unconditional averaging cost 3.0x) and then inverted on the headline: `never` won
# outright, because 60 cycles of practice ran the forward model 2.4x PAST the ceiling reference
# while the committed arms froze at cycle-20 competence AND lost model quality to a narrowed diet.
# 1b converts that inversion into a regime map. The two variables are WHEN you commit and WHAT YOU
# DO AFTER; the op is held fixed so neither confounds the other.
ARM_TABLE.update({
    "commit_early":               dict(op="library_kmeans", at="early"),
    "commit_mastery":             dict(op="library_kmeans", at="mastery"),
    "commit_early_il":            dict(op="library_kmeans", at="early", interleave=True),
    "commit_mastery_il":          dict(op="library_kmeans", at="mastery", interleave=True),
    "commit_mastery_il_reselect": dict(op="library_kmeans", at="mastery", interleave=True,
                                       reselect=True),
    # --- ROUND 1c: LIVE CONTENT under COMMITTED ROUTING -- the third corner of the op taxonomy.
    # 1b showed that once a FROZEN unit is committed, performance is decoupled from the forward
    # model entirely: interleaving restored the model from 2.90x rot to 1.10x and bought exactly
    # zero performance, because a frozen unit never consults it. `plan_launch` reconnects them --
    # committed routing (one re-grounding, one delay, open-loop for H steps) over live content.
    # The load-bearing arm is the one WITHOUT interleave: if the content is genuinely live, diet
    # rot should now hit PERFORMANCE directly, which the frozen arms could not show.
    "plan_launch_early":       dict(op="plan_launch", at="early"),
    "plan_launch_early_il":    dict(op="plan_launch", at="early", interleave=True),
    "plan_launch_mastery_il":  dict(op="plan_launch", at="mastery", interleave=True),
})

# WHY MASTERY = CYCLE 51 (read off round 1's `never` descent; a TEACHER-SCHEDULED boundary informed
# by a prior measurement, not an endogenous pacer -- `recital` found no within-level signal can
# place these boundaries, and a fixed schedule was rank 1 of 8 in every world).
# Rule, fixed in advance and applied once: take `never`'s held-out BALLISTIC error (the quantity a
# committed open-loop unit actually competes against), smooth it with a trailing-3-probe mean, take
# the final-plateau mean and sd over the last 5 probes (0.0384 +- 0.0026), and pick the first probe
# cycle from which the smoothed curve stays within 1 sd (<= 0.0410) for the rest of the run.
# Measured trailing-3 series: c33 0.0451, c36 0.0465, c39 0.0472, c42 0.0503, c45 0.0431,
# c48 0.0425, c51 0.0402, c54 0.0376, c57 0.0353, c60 0.0362 -> FIRST SUSTAINED ENTRY = c51.
# Note the readout matters: `never`'s REACTIVE error is already flat from c12 (~0.009), so a rule
# read off the reactive curve would have said "mastered" 39 cycles earlier and reproduced round 1's
# mistake. Open-loop competence is the later of the two clocks.
MASTERY_RATIONALE = ("round-1 never ballistic, trailing-3 mean, plateau 0.0384+-0.0026; "
                     "first sustained entry within 1 sd (<=0.0410) is c51 "
                     "(reactive curve would have said c12 -- wrong clock)")

# WHY THE INTERLEAVE PERIOD IS 3. Round 1 measured the diet rent directly, as the ratio of each
# arm's own ballistic error to `never`'s at matched cycle: `library_kmeans` sat at 0.85-0.92x for
# the first 7 post-commit cycles, jumped to 2.04x by +10, and saturated at 2.0-3.0x; `fixed` broke
# earlier (+4 to +7) and worse (up to 3.3x). So ROT ONSET IS ~7-10 CONSECUTIVE PURE-COMMITTED
# CYCLES. A period of 3 never allows more than 2 in a row -- roughly a 3x margin inside the
# measured onset window -- at a priced cost of ~46 s per interleaved cycle.
INTERLEAVE_RATIONALE = ("round-1 rot onset at +7 to +10 pure-committed cycles (0.85x -> 2.04x vs "
                        "never), saturating 2-3x; period 3 caps consecutive committed cycles at 2")

# WHY COMMIT AT CYCLE 20 (round-0 CAL-D, world b14, recorded in the run config).
# The descent is oscillatory rather than monotone, and it is NOT metering noise: the warmup
# metering seed is fixed, so the run-through is deterministic given the forward model, and the
# swings are a ballistic grader (4.9x transmission) amplifying a model that is bootstrapping on
# its own on-policy data. Measured envelope over 16 warmup cycles at b14:
#     cycles  1-9   e in [0.044, 0.152], span 0.108, mean 0.110
#     cycles 10-16  e in [0.051, 0.092], span 0.041, mean 0.068     <- 2.6x contraction
# So the envelope settles around c10 and the residual period is ~2-3 cycles. Committing at c20
# puts the commit >=4 cycles past the observed settling point (margin, because the warmup only ran
# to c16), and `trace_window=6` cycles x 24 performers = 144 candidates spans ~2-3 oscillation
# periods, so the pool averages over the oscillation instead of sampling one phase of it. It also
# leaves 40 of 60 cycles post-commit for the 3-cycle anchor window plus a long drift readout.
COMMIT_RATIONALE = ("b14 CAL-D: oscillation envelope contracts 2.6x by c10 (span 0.108 -> 0.041); "
                    "commit at c20 with a 6-cycle (144-trace) pool spanning ~2-3 periods")


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_fingering(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.arm_env import fk
    from mjc.embodied import make_arm_goal_sampler, pool_diagnostics, cmd_state_corr
    from mjc.practice.fingering.world import World, Ledger, start_postures

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    n = W.n
    arm = cfg["arm"]
    spec = ARM_TABLE[arm]
    op = spec["op"]
    at = spec.get("at")
    commit_at = (cfg["commit_cycle"] if at is None
                 else cfg["commit_early_cycle"] if at == "early"
                 else cfg["commit_mastery_cycle"])
    led = Ledger()
    out = {"config": cfg, "arm": arm, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_fingering", cfg["tag"], arm)
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print(f"[{arm}]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    P(f"[setup] device={device} op={op} curl_b={W.curl_b} push_a={W.push_a} "
      f"H_app={W.H_app} H_drill={W.H_drill} cycles={cfg['n_cycles']} commit@{commit_at} "
      f"interleave={bool(spec.get('interleave'))} reselect={bool(spec.get('reselect'))}")

    # ================================================================= diet + models
    # Byte-identical to `gates.py`'s setup block and seeded from the same offsets, so a gate run and
    # a main run at the same tag/seed/world share their forward models exactly.
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
    approach_pf = W.plan_fn(fm_app, W.H_app)

    def app_plan(q, seed):
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        return approach_pf(s, np.tile(W.W1[None, :], (len(q), 1)), np.random.default_rng(seed))

    plan_rt = app_plan(q_rt, cfg["seed"] + 6000)
    plan_sc = app_plan(q_sc, cfg["seed"] + 6100)
    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    REACT = World.reactive_unit()

    # corridor-matched ceiling reference (oracle instrument, free, identical across arms)
    corr = []
    for i in range(cfg["n_corridor"]):
        qq = geom(cfg["batch"], cfg["seed"] + 5400 + i)
        corr.append(W.traverse(fm0, REACT, qq, np.random.default_rng(cfg["seed"] + 6500 + i),
                               cfg["sigma_practice"], led, who="instrument", kind="ceiling_diet",
                               drill_replan=cfg["corridor_replan"],
                               approach_plan=app_plan(qq, cfg["seed"] + 6400 + i),
                               collect=True)["trans"])
    fm_ceil = W.mlp(cfg["seed"] + 42)
    W.train_steps(fm_ceil, torch.optim.Adam(fm_ceil.parameters(), lr=cfg["fm_lr"]),
                  np.concatenate([Sa] + [c[0] for c in corr]),
                  np.concatenate([Ua] + [c[1] for c in corr]),
                  np.concatenate([S2a] + [c[2] for c in corr]),
                  cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 302))

    # ================================================================= references + the scale
    def meter(fm, unit, seed, replan=1, who="agent"):
        return W.traverse(fm, unit, q_rt, np.random.default_rng(seed), cfg["sigma_perf"], led,
                          who=who, kind="metering", drill_replan=replan, approach_plan=plan_rt)

    # THE DETECTOR'S SCALE, measured ON THE METERING SET ITSELF. crystallize's scale bug: its stale
    # reference came from a different draw than metering, the first term of
    # `scale = max(ref_stale - emin, b)` went negative, and the fallback read a slow smooth descent
    # as silence. Same set, same statistic, same noise.
    ref_stale = float(np.median(meter(fm0, REACT, cfg["seed"] + 7200, replan=W.H_drill,
                                      who="instrument")["e_drill"]))
    ref_ceil = float(np.median(meter(fm_ceil, REACT, cfg["seed"] + 7300, replan=W.H_drill,
                                     who="instrument")["e_drill"]))
    ref_react = float(np.median(meter(fm0, REACT, cfg["seed"] + 7400, replan=1,
                                      who="instrument")["e_drill"]))
    out["setup"] = dict(ref_stale=ref_stale, ref_ceiling=ref_ceil, ref_reactive=ref_react,
                        usable_range=ref_stale - ref_ceil, exclude_keep_frac=keep_frac,
                        corr_reach=cmd_state_corr(S2_, U2_),
                        pool_diag=pool_diagnostics(Se, Ue, n))
    P(f"[ref] stale {ref_stale:.4f}  ceiling {ref_ceil:.4f}  reactive {ref_react:.4f}  "
      f"range {ref_stale - ref_ceil:.4f}")
    save()

    # ================================================================= the practice loop
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    buf, trace_buf = [], []
    brng = np.random.default_rng(cfg["seed"] + 800)
    unit = REACT
    committed = False
    bench = None
    cert_b = None
    cert_win, cert_pending = [], 0
    emin = None
    ehist, dhist = [], []
    sil_run = neg_run = 0
    sil_would_fire = None
    recerts = n_interleaved = n_reselect = 0
    log = {k: [] for k in ("cycle", "t_cum", "e_rt", "b", "delta", "scale", "sil_run", "neg_run",
                           "e_practice", "committed", "steps_agent", "plans_agent")}
    events, ladder = [], []

    def perf_unit():
        """What this arm actually plays at performance tempo. `never` re-plans every step and pays
        H_drill delays; every other arm plays its committed unit once it has one."""
        return unit

    def perf_replan():
        return 1 if unit["kind"] == "reactive" else W.H_drill

    def ladder_probe(cycle):
        """Held-out grade -- an experimenter instrument, free, and labelled as such. Three
        readouts: this arm's OWN performance configuration (the headline), plus reactive and
        ballistic-CEM under a common execution so the arms are also comparable at matched control
        mode."""
        rec = {"cycle": cycle, "t_cum": led.t_priced, "committed": committed,
               "steps_agent": led.steps["agent"], "plans_agent": led.plans["agent"]}
        o = W.traverse(fm, perf_unit(), q_ev, np.random.default_rng(cfg["seed"] + 7000 + cycle),
                       cfg["sigma_perf"], led, who="instrument", kind="ladder",
                       drill_replan=perf_replan(), approach_plan=plan_ev)
        rec["e_perf"] = float(np.median(o["e_drill"]))
        rec["e_perf_mean"] = float(o["e_drill"].mean())
        rec["t_piece_perf"] = o["t_piece"]
        rec["e_app"] = float(np.median(o["e_app"]))
        for nm, rp in (("react", 1), ("ball", W.H_drill)):
            oo = W.traverse(fm, REACT, q_ev, np.random.default_rng(cfg["seed"] + 7050 + cycle),
                            cfg["sigma_perf"], led, who="instrument", kind="ladder",
                            drill_replan=rp, approach_plan=plan_ev)
            rec[f"e_{nm}"] = float(np.median(oo["e_drill"]))
        return rec

    ladder.append(ladder_probe(0))

    for cycle in range(1, cfg["n_cycles"] + 1):
        # ---- (a) practice traversal, at the practice tempo, with motor noise ----
        # THE INTERLEAVE (the diet-rent cell): post-commit, one practice cycle in
        # `interleave_period` is an honest REACTIVE traversal of the drilled segment -- fully
        # priced, and the forward model trains on it exactly as usual -- while metering and
        # run-throughs keep routing the committed unit. The unit itself stays frozen, so this
        # isolates DIET MAINTENANCE from unit refresh. See INTERLEAVE_RATIONALE for the period.
        interleaved = bool(committed and spec.get("interleave")
                           and (cycle - commit_at) % cfg["interleave_period"] == 0)
        pr_unit = REACT if interleaved else unit
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + cycle)
        pr = W.traverse(fm, pr_unit, qp, np.random.default_rng(cfg["seed"] + 10_000 + cycle),
                        cfg["sigma_practice"], led, who="agent",
                        kind="practice_interleave" if interleaved else "practice",
                        drill_replan=1, approach_plan=app_plan(qp, cfg["seed"] + 9500 + cycle),
                        collect=True)
        n_interleaved += int(interleaved)
        buf.append(pr["trans"])
        # the 4th field tags a trace as REACTIVE, so a later reselect can draw candidates only
        # from renditions the agent actually improvised (a committed unit's own traces are just
        # that unit plus motor noise -- a pool of one)
        trace_buf.append((pr["hand"].copy(), pr["acts"].copy(), pr["e_drill"].copy(),
                          pr_unit["kind"] == "reactive"))
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
        rt = meter(fm, perf_unit(), cfg["seed"] + 4000 + cycle, replan=perf_replan())
        e = float(np.median(rt["e_drill"]))
        if bench is None:
            bench, emin = e, e
        emin = min(emin, e)
        d = bench - e
        if cert_pending:
            cert_win.append(e); cert_pending -= 1
            if cert_pending == 0:
                cert_b = float(np.mean(cert_win))
                events.append(dict(kind="anchor", cycle=cycle, level=cert_b,
                                   window=[float(x) for x in cert_win]))
                P(f"[anchor] cycle={cycle} b_anchored={cert_b:.4f} "
                  f"(window {np.round(cert_win, 4).tolist()})")
        if cert_b is not None:
            d = cert_b - e
        scale = max(ref_stale - emin, (cert_b if cert_b is not None else bench), 1e-6)
        ehist.append(e); dhist.append(d)
        we, wd = ehist[-cfg["sil_W"]:], dhist[-cfg["sil_W"]:]
        quiet = (len(we) >= cfg["sil_W"] and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                 and float(np.std(we)) < cfg["sil_cv"] * scale)
        sil_run = sil_run + 1 if quiet else 0
        neg_run = neg_run + 1 if (cert_b is not None and d < -cfg["sil_c"] * scale) else 0
        if sil_would_fire is None and sil_run >= cfg["sil_hold"] and cycle >= cfg["sil_min_cycle"]:
            sil_would_fire = cycle          # the certificate as INSTRUMENT, never as the trigger
            events.append(dict(kind="certificate_would_fire", cycle=cycle, e=e, b=bench))
            P(f"[certificate] would have fired at cycle {cycle} (e={e:.4f}) -- logged, not acted on")
        if cert_b is None:
            bench = bench + cfg["bench_alpha"] * (e - bench)

        # ---- (d) commitment: provisional, at a fixed cycle ----
        def commit(reason, n_cand=None, n_score=None, reactive_only=False):
            nonlocal unit, committed, cert_b, cert_win, cert_pending
            if op == "plan_launch":
                # Nothing to select, so nothing to audition -- and charging a phantom audition to
                # "match budgets" would be dishonest in the other direction. The cheapness is a
                # real property of the op, and the d_plan axis is where it pays for it.
                unit = {"kind": "plan_launch"}
                committed = True
                cert_b = None; cert_win = []; cert_pending = cfg["cert_cal"]
                events.append(dict(kind="commit", cycle=cycle, op=op, reason=reason,
                                   t_cum=led.t_priced, e_rt=e, b=bench, n_pool=0, n_cand=0,
                                   n_score=0, best_fixed=None, per_state_oracle=None,
                                   chosen_score=None,
                                   note="live content: routing commits, content is planned at "
                                        "launch under the current FM; no audition"))
                P(f"[commit] op={op} reason={reason} cycle={cycle} e_rt={e:.4f} "
                  f"(routing committed; content stays live)")
                return
            nc = int(n_cand or cfg["n_cand"]); ns = int(n_score or cfg["n_score"])
            # the score set: hand-over states from AT-TEMPO upstream execution (the seam law),
            # generated by the body and charged.
            S0 = W.traverse(fm, REACT, q_sc[:ns], np.random.default_rng(cfg["seed"] + 4400 + cycle),
                            cfg["sigma_perf"], led, who="agent", kind="score_set",
                            approach_plan=plan_sc[:ns], drill=False)["hand"]
            src = [t for t in trace_buf if t[3]] if reactive_only else list(trace_buf)
            src = src or list(trace_buf)
            pool_hand = np.concatenate([t[0] for t in src])
            pool_acts = np.concatenate([t[1] for t in src])
            pool_err = np.concatenate([t[2] for t in src])
            crng = np.random.default_rng(cfg["seed"] + 953 + cycle)
            ci = crng.permutation(len(pool_acts))[:nc]
            cands = pool_acts[ci]
            # EVERY arm pays for the same audition on the same candidates and states, so the
            # priced comparison cannot be won by skipping the grading step.
            E = W.audition(cands, S0, W.W2, led, who="agent")
            info = dict(n_pool=int(len(pool_acts)), n_cand=int(len(cands)), n_score=int(len(S0)),
                        score_med=float(np.median(E.mean(1))),
                        best_fixed=float(E.mean(1).min()),
                        per_state_oracle=float(E.min(0).mean()),
                        score_s0_disp=float(np.linalg.norm(W.tip(S0).mean(0) - W.W1)),
                        score_s0_spread=float(np.linalg.norm(W.tip(S0).std(0))),
                        score_s0_speed=float(np.linalg.norm(S0[:, n:], axis=1).mean()),
                        pool_geometry=World.pool_geometry(cands),
                        pool_geometry_successful=World.pool_geometry(
                            pool_acts[np.argsort(pool_err)[:max(4, len(pool_err) // 3)]]),
                        n_reactive_traces=int(sum(1 for t in src if t[3])),
                        reactive_only=bool(reactive_only))
            labels = None
            if op == "fixed":
                u, j = World.select_fixed(E, cands)
                info.update(chosen_score=float(E[j].mean()), chosen_own_err=float(pool_err[ci][j]),
                            score_of_best_own_err=float(E[int(np.argmin(pool_err[ci]))].mean()))
            elif op == "library_kmeans":
                u, li = W.select_library(E, cands, S0, cfg["n_lib"],
                                         np.random.default_rng(cfg["seed"] + 954 + cycle))
                Z = (S0 - u["key_mu"]) / u["key_sd"]
                kk = ((Z[:, None, :] - u["keys"][None, :, :]) ** 2).sum(-1).argmin(1)
                labels = kk
                info.update(library=li, chosen_score=float(
                    np.mean([E[li["picks"][int(kk[i])], i] for i in range(len(S0))])))
            elif op == "library_proj":
                u, li = W.select_library_proj(E, cands, S0, cfg["n_lib"])
                Zp = np.concatenate([(S0 - u["key_mu"]) / u["key_sd"],
                                     np.ones((len(S0), 1))], 1)
                kk = np.clip(np.searchsorted(u["edges"], Zp @ u["beta"]), 0, cfg["n_lib"] - 1)
                labels = kk
                info.update(library=li, chosen_score=float(
                    np.mean([E[li["picks"][int(kk[i])], i] for i in range(len(S0))])))
            elif op == "regress":
                u = W.fit_bc(pool_hand, pool_acts, cfg["seed"] + 960 + cycle)
                info.update(chosen_score=None, n_fit=int(len(pool_hand)))
            elif op == "mean":
                u = World.mean_unit(pool_acts)
                info.update(chosen_score=float(W.replay(S0, u["fixed"], W.W2).mean()))
            else:
                raise ValueError(op)
            # PER-CELL POOL GEOMETRY -- does the conditioning variable separate the modes that
            # averaging destroys? Oracle-only, charged to the instrument, never read by the agent.
            if labels is not None and cfg["cell_geometry"]:
                info["cell_geometry"] = W.cell_geometry(E, cands, S0, labels, W.W2, led)
            if cfg["persist_commit"]:
                np.savez_compressed(
                    os.path.join(outdir, f"commit_c{cycle}.npz"), E=E, cands=cands, S0=S0,
                    labels=(labels if labels is not None else np.zeros(len(S0), np.int64)),
                    pool_err=pool_err[ci])
            unit = u
            committed = True
            cert_b = None; cert_win = []; cert_pending = cfg["cert_cal"]
            events.append(dict(kind="commit", cycle=cycle, op=op, reason=reason,
                               t_cum=led.t_priced, e_rt=e, b=bench, **info))
            P(f"[commit] op={op} reason={reason} cycle={cycle} e_rt={e:.4f} "
              f"score={info.get('chosen_score')} best_fixed={info['best_fixed']:.4f} "
              f"oracle={info['per_state_oracle']:.4f}")

        if op is not None and not committed and cycle >= commit_at:
            commit("provisional_schedule")
        elif (committed and spec.get("reselect") and cfg["reselect_every"] > 0
              and cycle > commit_at and (cycle - commit_at) % cfg["reselect_every"] == 0):
            # `setlist`'s maintenance op: audit and re-select from the REFRESHED pool. Only
            # meaningful on top of the interleave, which is what puts improvised renditions back
            # into the pool. Deliberately a SMALLER audition than the initial commit -- a
            # maintenance audit, priced as one.
            n_reselect += 1
            commit("reselect", n_cand=cfg["reselect_n_cand"], n_score=cfg["reselect_n_score"],
                   reactive_only=True)
        elif (committed and recerts < cfg["max_recert"] and neg_run >= cfg["sil_W"]
              and op in SELECT_OPS):
            recerts += 1
            commit("recert_on_negative_delta")

        log["cycle"].append(cycle); log["t_cum"].append(led.t_priced)
        log["e_rt"].append(e); log["b"].append(bench); log["delta"].append(d)
        log["scale"].append(scale); log["sil_run"].append(sil_run); log["neg_run"].append(neg_run)
        log["e_practice"].append(float(np.median(pr["e_drill"])))
        log["committed"].append(committed); log["steps_agent"].append(led.steps["agent"])
        log["plans_agent"].append(led.plans["agent"])

        if cycle % cfg["probe_every"] == 0 or cycle == cfg["n_cycles"]:
            ladder.append(ladder_probe(cycle))
            P(f"[c{cycle:3d}] e_rt={e:.4f} d={d:+.4f} sil={sil_run} | perf="
              f"{ladder[-1]['e_perf']:.4f} react={ladder[-1]['e_react']:.4f} "
              f"ball={ladder[-1]['e_ball']:.4f} | t={led.t_priced:8.1f}s")
            out.update(log=log, ladder=ladder, events=events, ledger=led.snapshot())
            save()
        elif cycle % 5 == 0:
            P(f"[c{cycle:3d}] e_rt={e:.4f} d={d:+.4f} sil={sil_run} t={led.t_priced:8.1f}s")

    # ---- the seam-law check: audition score vs the level actually realised after commitment ----
    anchors = [ev for ev in events if ev["kind"] == "anchor"]
    commits = [ev for ev in events if ev["kind"] == "commit"]
    gap = None
    if anchors and commits and commits[0].get("chosen_score"):
        gap = anchors[0]["level"] / max(commits[0]["chosen_score"], 1e-9)
    post = [x for x, c in zip(log["e_rt"], log["committed"]) if c]
    k = max(1, len(post) // 3)
    out.update(log=log, ladder=ladder, events=events, ledger=led.snapshot(),
               summary=dict(
                   final_e_perf=ladder[-1]["e_perf"], final_e_react=ladder[-1]["e_react"],
                   final_e_ball=ladder[-1]["e_ball"], t_priced=led.t_priced,
                   optimism_gap=gap, certificate_would_fire=sil_would_fire, recerts=recerts,
                   commit_at=commit_at, n_interleaved=n_interleaved, n_reselect=n_reselect,
                   post_commit_sd=float(np.std(post)) if post else None,
                   post_commit_drift=(float(np.mean(post[-k:]) - np.mean(post[:k]))
                                      if post else None)),
               complete=True)
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[done] e_perf={out['summary']['final_e_perf']:.4f} t_priced={led.t_priced:.1f}s "
      f"gap={gap} drift={out['summary']['post_commit_drift']}")
    return out


@app.local_entrypoint()
def fingering(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    arms: str = "never,fixed,library_kmeans,library_proj,regress,mean",
    # ---- the world (defaults inherited from the round-0 gate sweep; override per its report)
    curl_b: float = 14.0,
    push_a: float = 0.0,
    w1: str = DEF_W1,
    w2: str = DEF_W2,
    h_app: int = 14,
    h_drill: int = 20,
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
    # ---- controllers
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.5,
    # ---- practice loop
    n_cycles: int = 60,
    commit_cycle: int = 20,      # from round-0 CAL-D -- see COMMIT_RATIONALE
    commit_early_cycle: int = 20,     # round 1b: continuity with round 1
    commit_mastery_cycle: int = 51,   # round 1b: see MASTERY_RATIONALE
    interleave_period: int = 3,       # round 1b: see INTERLEAVE_RATIONALE
    reselect_every: int = 15,
    reselect_n_cand: int = 32,
    reselect_n_score: int = 48,
    cell_geometry: bool = True,
    persist_commit: bool = True,
    batch: int = 24,
    n_rt: int = 48,
    n_eval: int = 48,
    probe_every: int = 3,
    trace_window: int = 6,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    # ---- metering / the certificate (instrument only)
    bench_alpha: float = 0.2,
    sil_c: float = 0.06,
    sil_cv: float = 0.10,
    sil_w: int = 5,
    sil_hold: int = 2,
    sil_min_cycle: int = 4,
    cert_cal: int = 3,
    max_recert: int = 1,
    # ---- the compile op
    n_cand: int = 96,
    n_score: int = 96,
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
        n_cycles = 6; commit_cycle = 3; batch = 8; n_rt = 12; n_eval = 12
        probe_every = 2; trace_window = 3
        n_cand = 6; n_score = 16; n_lib = 2; pol_steps = 300
        k_shoot = 256; cem_iters = 4; sil_w = 2; sil_min_cycle = 1
        tag = tag or "smoke"
    tag = tag or "default"
    base = dict(
        tag=tag, seed=seed, curl_b=curl_b, push_a=push_a,
        w1=[float(x) for x in w1.split(",")], w2=[float(x) for x in w2.split(",")],
        h_app=h_app, h_drill=h_drill, q_jit=q_jit, null_jit=null_jit, start_mode=start_mode,
        patch_sigma=patch_sigma,
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
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen,
        n_cycles=n_cycles, commit_cycle=commit_cycle, batch=batch, n_rt=n_rt, n_eval=n_eval,
        probe_every=probe_every, trace_window=trace_window,
        sigma_practice=sigma_practice, sigma_perf=sigma_perf, d_fb=d_fb,
        bench_alpha=bench_alpha, sil_c=sil_c, sil_cv=sil_cv, sil_W=sil_w, sil_hold=sil_hold,
        sil_min_cycle=sil_min_cycle, cert_cal=cert_cal, max_recert=max_recert,
        n_cand=n_cand, n_score=n_score, n_lib=n_lib,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps,
        commit_early_cycle=commit_early_cycle, commit_mastery_cycle=commit_mastery_cycle,
        interleave_period=interleave_period, reselect_every=reselect_every,
        reselect_n_cand=reselect_n_cand, reselect_n_score=reselect_n_score,
        cell_geometry=cell_geometry, persist_commit=persist_commit,
        commit_rationale=COMMIT_RATIONALE, mastery_rationale=MASTERY_RATIONALE,
        interleave_rationale=INTERLEAVE_RATIONALE,
    )
    outs = list(run_fingering.map([{**base, "arm": a} for a in arm_list]))
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for o in outs:
        with open(os.path.join(localdir, f"{o['arm']}.json"), "w") as fh:
            json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(outs)} arm files to {localdir}")
