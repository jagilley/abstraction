"""PRESTO, PHASE B -- the delay sweep on a piece designed against the reflex delay.

THE STANDING FACT THIS RE-ASKS. `../../offbook/delay_gate.py` (run `d0`) put an observation delay on
the reflex loop and measured degradation ordered exactly by feedback consumption -- reactive
(61 fb/piece) 63.1x, live-per-segment (4 fb) 2.69x, chain (2 fb) 1.69x, segment tape (4 fb) 1.28x --
and still found no delay at which a stored unit was the best PLAYABLE option. The playable region
ended at Delta = 2 (40 ms) and reactive won all of it; the ordering inverted only at Delta = 16 by
universal collapse, which the pre-fixed guard refuses. Delta* = None. The piece was legato's, whose
segments are 400 ms on a 20 ms control loop -- a slow movement. This node changes the PIECE and the
PLANT (segments of 120 ms against a ~100 ms proprioceptive loop; momentum-dominated; no localised
patch) and asks the same question with the same criterion.

WHAT IS SWEPT, AND WHY THESE FOUR STRATEGIES. They differ in exactly one thing -- how much fresh
feedback they consume -- and nothing else:

    reactive    replans every control step from the observation          31 fb / traversal
    live_seg    one CEM plan per segment from the observed seam state     6 fb
    seg_tape    one MEASURED tape per segment, keyed on the same state    6 fb
    chain       one MEASURED tape for the whole 30-step phrase            2 fb

TWO INCUMBENTS AT EVERY DELAY, and the difference between them is the point. offbook's `obs()`
hands a delayed consumer the raw stale state -- act on where you WERE. That is a strawman: a nervous
system with a reflex delay acts on where its own forward model says it now IS, given the commands it
has already issued. `obs_predict` (see `world.py`) supplies that efference copy, and the delay then
costs exactly the part that could not be predicted -- the motor noise the copy does not contain,
plus the model's drift over Delta steps. The PREDICTING triple is the PRIMARY read and is declared
so here, before the run; the NAIVE triple is reported beside it as the continuity read against
offbook `d0`. A result that only beats a strawman is not a result.

THE INCUMBENT IS NOT HANDICAPPED. Reactive is given its BEST lookahead at every delay (the grid is
swept and the minimum taken), because Phase A measured that a delayed controller wants a longer one
(Delta = 4: look 6 -> 0.520, 12 -> 0.427, 30 -> 0.354). Its whole grid is on the record.

THE NEUTRAL CRITERION, CARRIED VERBATIM FROM offbook ROUND 4, FIXED BEFORE THE RUN:

    Delta* = the SMALLEST Delta at which  e_chain <= e_live_seg <= e_reactive  on mean piece error,
             subject to  min(e_chain, e_live_seg, e_reactive) <= ref_play.

    ref_play = max( ref_stale , 0.5 * mean drilled leg )

`ref_stale` is offbook's own anchor -- ballistic-per-segment piece error under the STALE
(pre-practice) forward model at Delta = 0 -- and is carried unchanged. The second component is new
and is declared here with its reason: offbook's `ref_stale` was large (0.146) partly BECAUSE the
curl patch was excluded from the pre-practice model's training, and presto has no patch, so the same
formula can return a much smaller number and veto every Delta by construction. That is the exact
failure offbook's own guard revision was written to avoid ("a guard anchored to the undelayed
champion cannot fire once you tax feedback at all"). Half a drilled leg is task-anchored and
arm-neutral: it says the waypoints are still resolved, i.e. the figure is recognisable. BOTH
components are reported separately so either can be read off the record, and neither depends on any
compared arm.

If no Delta satisfies both, THE ROUND STOPS AT THE GATE and that is a substrate finding. Delta is
NOT to be extended or re-tuned until the ordering inverts (legato F2).

Run:
    modal run mjc/practice/accompanist/presto/delay_gate.py::delay_gate --quick --tag psmoke
    modal run --detach mjc/practice/accompanist/presto/delay_gate.py::delay_gate --spawn --tag p0 --seed 0
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.accompanist.presto.piece import (DEF_R, DEF_DAMPING, DEF_H_SEG, DEF_H_APP,
                                       DEF_CENTER, DEF_ANGLES, waypoints, legs, mean_leg)


@app.function(gpu="L4", memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_delay_gate(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler
    from mjc.practice.accompanist.presto.world import World, Ledger, start_postures, elite_for

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    NS = W.n_seg
    led = Ledger()
    t0 = time.time()
    out = {"config": cfg, "complete": False,
           "criterion": "smallest Delta with e_chain <= e_live_seg <= e_reactive on mean piece "
                        "error, subject to min(of the three) <= ref_play = "
                        "max(ref_stale, 0.5 * mean drilled leg). PRIMARY read = the PREDICTING "
                        "triple (efference copy through the delay); the NAIVE triple is the "
                        "continuity read against offbook d0."}
    outdir = os.path.join(DATA_DIR, "practice_presto", cfg["tag"], "delay_gate")
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print("[pgate]", *a, flush=True)

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
    Se = np.concatenate([S1, S2_]); Ue = np.concatenate([U1, U2_])
    S2e = np.concatenate([S21, S22])
    W.set_norm(Se, Ue, S2e)
    fm0 = W.mlp(cfg["seed"] + 41)
    W.train_steps(fm0, torch.optim.Adam(fm0.parameters(), lr=cfg["fm_lr"]),
                  Se, Ue, S2e, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
    fm_app = copy.deepcopy(fm0)
    for p_ in fm_app.parameters():
        p_.requires_grad_(False)
    P(f"[setup] pools {len(Se)} transitions, stale FM trained ({time.time() - t0:.0f}s)")

    def geom(m, sd):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(sd),
                              cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])

    def app_plan(q, sd):
        st = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=0.0, wp_mask=W.approach_mask())
        return pf(st, np.tile(W.goals[0][None, :], (len(q), 1)), np.random.default_rng(sd))[0]

    # ---- the piece is LEARNED UNDELAYED; the delay is a decision constraint applied afterwards
    # (offbook Round 4's convention, kept: one variable).
    R_REACT = W.routing("reactive")
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    brng = np.random.default_rng(cfg["seed"] + 800)
    buf = []
    for c in range(1, int(cfg["n_warm"]) + 1):
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + c)
        pr = W.traverse(fm, R_REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + c),
                        cfg["sigma_practice"], led, who="agent", kind="practice",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + c), collect=True,
                        look=cfg["react_look"])
        buf.append(pr["trans"])
        if len(buf) > cfg["trace_window"]:
            buf.pop(0)
        PX, PY = W.tensors(*[np.concatenate([b[i] for b in buf]) for i in range(3)])
        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])
    P(f"[warm] {cfg['n_warm']} reactive cycles at delay 0 ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- the three geometry sets
    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200); plan_ev = app_plan(q_ev, cfg["seed"] + 5250)
    q_hv = geom(cfg["n_eval"], cfg["seed"] + 5400); plan_hv = app_plan(q_hv, cfg["seed"] + 5450)
    q_sc = geom(cfg["n_score"], cfg["seed"] + 5500); plan_sc = app_plan(q_sc, cfg["seed"] + 5550)

    def run(net, routing, sd, kind, delay=0, look=None, on="ev", predict=False, stop_seg=None):
        q, pl = {"ev": (q_ev, plan_ev), "hv": (q_hv, plan_hv), "sc": (q_sc, plan_sc)}[on]
        W.obs_delay = int(delay)
        W.obs_predict = bool(predict)
        o = W.traverse(net, routing, q, np.random.default_rng(cfg["seed"] + sd),
                       cfg["sigma_perf"], led, who="instrument", kind=kind, approach_plan=pl,
                       stop_seg=stop_seg, look=(look or cfg["react_look"]))
        W.obs_delay = 0
        W.obs_predict = False
        return o

    def score(o):
        sp = np.linalg.norm(np.diff(o["tips"], axis=1), axis=2) / W.dt_ctrl
        return dict(e=float(np.nanmean(o["e_piece"])), e_med=float(np.nanmedian(o["e_piece"])),
                    by_seg=[float(np.nanmedian(o["e_seg"][:, k])) for k in range(NS)],
                    fb=float(o["n_fb"]), v_mean=float(np.nanmean(sp)),
                    v_max=float(np.nanmax(sp)))

    # ---------------------------------------------------------------- the anchors, before anything
    R_BALL = W.routing("plan_launch", groups=[1] * NS, **CALP[1])
    R_CHAIN_LIVE = W.routing("plan_launch", groups=[NS], **CALP[min(NS, max(CALP))])
    ref_stale = score(run(fm0, R_BALL, 7200, "ref_stale"))["e"]
    half_leg = 0.5 * mean_leg(cfg["R"])
    ref_play = max(ref_stale, half_leg)
    hold = score(run(fm, W.routing("fixed", groups=[NS],
                                   fixed=np.zeros((W.H_phrase, W.AD), np.float32)), 7100, "hold"))
    out.update(ref_stale=ref_stale, half_leg=half_leg, ref_play=ref_play, hold=hold,
               legs=legs(cfg["R"]), mean_leg=mean_leg(cfg["R"]),
               nominal_speed=mean_leg(cfg["R"]) / (W.H_seg[0] * W.dt_ctrl))
    P(f"[anchor] ref_stale {ref_stale:.4f} | half_leg {half_leg:.4f} -> ref_play {ref_play:.4f} "
      f"| do-nothing floor {hold['e']:.4f}")
    save()

    # ---------------------------------------------------------------- the content: a NESTED build
    # THE RECIPE, AND WHY EVERY PART OF IT IS LOAD-BEARING. legato's numbers come from a protocol
    # that is easy to state and easy to drop pieces of, and each piece dropped destroys the content
    # in a different way. Three independent measurements, two of them from sibling rounds on
    # offbook's piece, say which pieces:
    #
    #   * the SCORE SET must come from the deploying configuration. presto's own first smoke drew it
    #     from a reactive traversal: the deployed tape arm ran 0.095 -> 0.430 across the phrase while
    #     its own audition claimed 0.042-0.091, because every tape a library commits moves the next
    #     seam off the reactive manifold.
    #   * the CANDIDATES must too. The sibling `d1` round kept a reactive-harvested pool and
    #     reproduced legato's seam-0 content exactly (chosen 0.041 vs 0.043) while seams 1-2 came out
    #     4x worse AT THE PER-STATE-ORACLE LEVEL -- the pool contained nothing that works from the
    #     committed configuration's launch states, which no selection can repair. legato's own commit
    #     events: at its seam-1/2 commits, 142 of 144 candidates came from practice in the
    #     already-committed configuration.
    #   * the MODEL MUST KEEP TRAINING THROUGH THE COMMITS. `d2` added the nested candidates and
    #     recovered legato's content to within 0.87-1.16x at every seam and the chain cell, with its
    #     keyed nested chain reaching 0.1455 against that node's 0.1460 anchor -- the first stored
    #     strategy to reach playability there. Its remaining residual against legato is most
    #     plausibly that `d2` FROZE the forward model through the nested practice, to keep a
    #     round-4 control bit-identical, while legato's model kept training through its commits.
    #     presto has no cross-round control to preserve, so it follows legato: the model trains
    #     through the between-commit practice cycles and is FROZEN ONLY FOR THE SWEEP.
    #
    # Freezing at the end is what keeps the delay axis clean -- every strategy in the sweep is
    # compared under ONE model, so "which arm the model was last fitted to" cannot be confounded
    # with Delta -- while training through the build is what makes the content legato-quality.
    # `ref_stale` stays defined under the PRE-PRACTICE model `fm0` and was already measured above,
    # so the playability anchor cannot move with any of this.
    #
    # THE MODEL'S OWN COMPETENCE IS ON THE RECORD AT EVERY COMMIT (`e_react`, reactive piece error
    # on the held-out evaluation geometries at Delta = 0). legato CAL-D reads exactly this curve,
    # and it is the quantity that says whether content committed at seam k was frozen against a
    # model that was still improving -- legato's own measured failure mode (`L1` committed at
    # c25-c39 against a ballistic clock that had not plateaued by c72).
    #
    # TWO LIBRARIES, INTERLEAVED IN ONE PRACTICE TRAJECTORY. `select_library` (keyed on the launch
    # state) and `select_fixed` (one argmin-of-mean tape, reading no state at all) are the same op on
    # the same candidates at the same price -- legato's point that a library's advantage is
    # information use, not budget -- but the score set is PART of the op, so each gets its own
    # prefix, its own candidates and its own score set. They are built at the SAME seam in the SAME
    # practice trajectory, so neither sees a better-trained model than the other. The fixed arm is
    # not merely a control here: a unit that reads no state is the only strategy a delay cannot
    # touch at all.
    #
    # PRACTICE IS NOISY, HARVEST IS NOT. Practice runs at `sigma_practice` and trains the model;
    # candidates are harvested separately at `sigma_perf`. legato drew its candidates from the
    # practice trace pool itself; separating them is deliberate (offbook's `d0` harvested at
    # practice noise and its stored strategies never cleared the anchor at any delay) and is stated
    # as a departure.
    q_pr_seed = [cfg["seed"] + 20_000]

    def practice(n, routing, label):
        """`n` practice cycles in `routing`, at practice tempo, training the forward model."""
        for _ in range(int(n)):
            q_pr_seed[0] += 1
            qp = geom(cfg["batch"], q_pr_seed[0])
            pr = W.traverse(fm, routing, qp, np.random.default_rng(q_pr_seed[0] + 500_000),
                            cfg["sigma_practice"], led, who="agent", kind=f"practice_{label}",
                            approach_plan=app_plan(qp, q_pr_seed[0] + 900_000), collect=True,
                            look=cfg["react_look"])
            buf.append(pr["trans"])
            if len(buf) > cfg["trace_window"]:
                buf.pop(0)
            PX, PY = W.tensors(*[np.concatenate([b[i] for b in buf]) for i in range(3)])
            W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                           cfg["replay_frac"])

    def commit_at(k, pick, units, sd0):
        """Build seam `k`'s unit for one library, from ITS OWN prefix: candidates from
        `n_harvest` performance-tempo traversals of the already-committed configuration, score set
        from a held-out traversal of the same configuration stopped at `k` (which runs no planner)."""
        pre = W.routing(None, groups=[1] * NS,
                        units=list(units) + [{"kind": "reactive"}] * (NS - k))
        hv = [run(fm, pre, sd0 + 31 * k + i, f"hv_{pick}{k}_{i}", on="hv")
              for i in range(cfg["n_harvest"])]
        lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k])
        cd_ = np.nan_to_num(np.concatenate([h["acts"] for h in hv]), nan=0.0)[:, lo:hi, :]
        sck = run(fm, pre, sd0 + 31 * k + 17, f"sc_{pick}{k}", on="sc", stop_seg=k)
        S0k = np.asarray(sck["launches"][k], np.float32)
        E, _ = W.audition(cd_, S0k, W.checkpoints(k, 1), led, who="instrument")
        if pick == "key":
            u, li = W.select_library(E, cd_, S0k, cfg["n_lib"],
                                     np.random.default_rng(cfg["seed"] + 954 + k))
            chosen = float(np.mean([E[li["picks"][int(l_)], i]
                                    for i, l_ in enumerate(li["labels"])]))
            cells, ndist = li["cell_sizes"], int(li["n_distinct"])
        else:
            u, j = W.select_fixed(E, cd_)
            chosen, cells, ndist = float(E[j].mean()), [len(S0k)], 1
        info = dict(n_cand=int(len(cd_)), n_score=int(len(S0k)), chosen=chosen,
                    best_fixed=float(E.mean(1).min()),
                    per_state_oracle=float(E.min(0).mean()),
                    oracle_gain=float(E.mean(1).min() / max(E.min(0).mean(), 1e-9)),
                    n_distinct=ndist, cell_sizes=cells,
                    seam_speed=float(np.linalg.norm(S0k[:, W.n:], axis=1).mean()))
        return u, info

    units_key, units_fix = [], []
    content = {"key": {}, "fix": {}}
    clock = []                      # the model's own competence curve, one entry per commit
    for k in range(NS):
        pre_key = W.routing(None, groups=[1] * NS,
                            units=list(units_key) + [{"kind": "reactive"}] * (NS - k))
        practice(cfg["n_between"], pre_key, f"seam{k}")
        e_react = score(run(fm, R_REACT, 6000 + k, f"clock{k}"))["e"]
        e_ball = score(run(fm, R_BALL, 6100 + k, f"clock_ball{k}"))["e"]
        clock.append(dict(seam=k, cycles=cfg["n_warm"] + (k + 1) * cfg["n_between"],
                          e_react=e_react, e_ball_seg=e_ball))
        uk, ik = commit_at(k, "key", units_key, 7800)
        uf, if_ = commit_at(k, "fix", units_fix, 8800)
        units_key.append(uk); units_fix.append(uf)
        content["key"][f"seg{k}"] = ik
        content["fix"][f"seg{k}"] = if_
        P(f"[commit seam {k}] after {clock[-1]['cycles']} practice cycles | e_react {e_react:.4f} "
          f"e_ball_seg {e_ball:.4f} | key chosen {ik['chosen']:.4f} (oracle "
          f"{ik['per_state_oracle']:.4f}, {ik['n_distinct']}/{len(ik['cell_sizes'])} distinct) | "
          f"fix chosen {if_['chosen']:.4f} (oracle {if_['per_state_oracle']:.4f})")
        save()

    # THE CHAIN POOL is grown from the FULLY SEGMENT-COMMITTED configuration, after its own block of
    # practice in that configuration -- legato F5's mechanism, that the segment library manufactures
    # the phrase pool's addressable variation. A chain's launch state is the approach's output,
    # identical under every configuration, so only the POOL is at issue here and the score set is
    # configuration-independent. `chain_react` keeps a reactive-sourced pool beside it as the
    # continuity control ON THE RECIPE ITSELF; `chain` is the PRIMARY.
    R_SEG_TAPE = W.routing(None, groups=[1] * NS, units=units_key)
    R_SEG_FIX = W.routing(None, groups=[1] * NS, units=units_fix)
    practice(cfg["n_between"], R_SEG_TAPE, "chain")
    e_react = score(run(fm, R_REACT, 6000 + NS, f"clock{NS}"))["e"]
    clock.append(dict(seam="chain", cycles=cfg["n_warm"] + (NS + 1) * cfg["n_between"],
                      e_react=e_react,
                      e_ball_seg=score(run(fm, R_BALL, 6100 + NS, f"clock_ball{NS}"))["e"]))
    out["clock"] = clock
    P(f"[clock] e_react by commit: "
      + "  ".join(f"{c['seam']}@c{c['cycles']}:{c['e_react']:.4f}" for c in clock)
      + "  -- flat means content was not frozen against a model that was still improving "
        "(legato CAL-D's measured failure mode)")

    # THE MODEL IS FROZEN FROM HERE. Everything below -- the chain build and the whole sweep -- runs
    # under one model, so no comparison in the sweep can be confounded with training.
    S0_0 = np.asarray(run(fm, R_SEG_TAPE, 9500, "sc_chain", on="sc",
                          stop_seg=0)["launches"][0], np.float32)
    chain_units = {}
    for nm, src in (("chain", R_SEG_TAPE), ("chain_react", R_REACT)):
        hv = [run(fm, src, 9600 + 41 * (nm == "chain_react") + i, f"hv_{nm}_{i}", on="hv")
              for i in range(cfg["n_harvest"])]
        cd_ = np.nan_to_num(np.concatenate([h["acts"] for h in hv]), nan=0.0)
        E, _ = W.audition(cd_, S0_0, W.checkpoints(0, NS), led, who="instrument")
        u, li = W.select_library(E, cd_, S0_0, cfg["n_lib"],
                                 np.random.default_rng(cfg["seed"] + 970))
        chain_units[nm] = u
        content[nm] = dict(n_cand=int(len(cd_)), n_score=int(len(S0_0)),
                           chosen=float(np.mean([E[li["picks"][int(l_)], i]
                                                 for i, l_ in enumerate(li["labels"])])),
                           best_fixed=float(E.mean(1).min()),
                           per_state_oracle=float(E.min(0).mean()),
                           oracle_gain=float(E.mean(1).min() / max(E.min(0).mean(), 1e-9)),
                           n_distinct=int(li["n_distinct"]), cell_sizes=li["cell_sizes"])
        if nm == "chain":
            uf, jf = W.select_fixed(E, cd_)
            chain_units["chain_fixed"] = uf
            content["chain_fixed"] = dict(content[nm], chosen=float(E[jf].mean()),
                                          n_distinct=1, cell_sizes=[len(S0_0)])
    out["content"] = content
    P("[content/chain] " + "  ".join(
        f"{n}: chosen {content[n]['chosen']:.4f} best_fixed {content[n]['best_fixed']:.4f} "
        f"oracle {content[n]['per_state_oracle']:.4f}"
        for n in ("chain", "chain_react", "chain_fixed")))
    save()

    R_CHAIN = W.routing(None, groups=[NS], units=[chain_units["chain"]])
    R_CHAIN_FIX = W.routing(None, groups=[NS], units=[chain_units["chain_fixed"]])
    R_CHAIN_REACT = W.routing(None, groups=[NS], units=[chain_units["chain_react"]])

    # ---------------------------------------------------------------- the sweep
    sweep = {}
    out["sweep"] = sweep
    for D in [int(x) for x in cfg["delays"]]:
        cell = {}
        for pred, sfx in ((False, ""), (True, "p")):
            # the incumbent gets its BEST lookahead at this delay -- Phase A measured that a
            # delayed controller wants a longer one, and an incumbent handicapped by an inherited
            # knob is not an incumbent.
            grid = {str(lk): score(run(fm, R_REACT, 7700, f"react{sfx}_d{D}_l{lk}", delay=D,
                                       look=int(lk), predict=pred))
                    for lk in cfg["look_grid"]}
            best_l = min(grid, key=lambda l: grid[l]["e"])
            cell[f"reactive{sfx}"] = dict(grid[best_l], look=int(best_l),
                                          look_grid={l: v["e"] for l, v in grid.items()})
            cell[f"live_seg{sfx}"] = score(run(fm, R_BALL, 7300, f"live_seg{sfx}_d{D}",
                                               delay=D, predict=pred))
            cell[f"live_chain{sfx}"] = score(run(fm, R_CHAIN_LIVE, 7400, f"live_chain{sfx}_d{D}",
                                                 delay=D, predict=pred))
            cell[f"seg_tape{sfx}"] = score(run(fm, R_SEG_TAPE, 8100, f"seg_tape{sfx}_d{D}",
                                               delay=D, predict=pred))
            cell[f"seg_fixed{sfx}"] = score(run(fm, R_SEG_FIX, 8200, f"seg_fixed{sfx}_d{D}",
                                                delay=D, predict=pred))
            cell[f"chain{sfx}"] = score(run(fm, R_CHAIN, 8300, f"chain{sfx}_d{D}",
                                            delay=D, predict=pred))
            cell[f"chain_fixed{sfx}"] = score(run(fm, R_CHAIN_FIX, 8400, f"chain_fixed{sfx}_d{D}",
                                                  delay=D, predict=pred))
            cell[f"chain_react{sfx}"] = score(run(fm, R_CHAIN_REACT, 8500,
                                                  f"chain_react{sfx}_d{D}", delay=D, predict=pred))
        sweep[str(D)] = cell
        P(f"[D={D:2d} = {D * W.dt_ctrl * 1000:3.0f} ms] naive  " + "  ".join(
            f"{n}={cell[n]['e']:.4f}(fb{cell[n]['fb']:.0f})"
            for n in ("reactive", "live_seg", "seg_tape", "chain", "seg_fixed", "chain_fixed",
                      "chain_react"))
          + f"\n[D={D:2d}          ] predict " + "  ".join(
              f"{n}p={cell[n + 'p']['e']:.4f}"
              for n in ("reactive", "live_seg", "seg_tape", "chain", "seg_fixed", "chain_fixed",
                        "chain_react")))
        save()

    # ---------------------------------------------------------------- the pre-fixed rule
    verdict = {"ref_stale": ref_stale, "half_leg": half_leg, "guard_ceiling": ref_play,
               "primary": "predict", "delta_star": None, "delta_star_naive": None, "rows": []}
    for D in [int(x) for x in cfg["delays"]]:
        c = sweep[str(D)]
        row = {"delay": D, "ms": D * W.dt_ctrl * 1000}
        for sfx, nm in (("p", "predict"), ("", "naive")):
            er, el, ec = (c[f"reactive{sfx}"]["e"], c[f"live_seg{sfx}"]["e"], c[f"chain{sfx}"]["e"])
            ordered = bool(ec <= el <= er)
            best = min(er, el, ec)
            playable = bool(best <= ref_play)
            # MARGINS, not only verdicts. The sibling `d2` round missed this guard by 3.6% on
            # offbook's piece, and a pass/fail column alone would have reported that as an
            # indistinguishable "no". Positive = the condition holds with room to spare; the units
            # are metres for the absolute margins and a fraction of the guard for `guard_margin_f`.
            row[nm] = dict(e_reactive=er, e_live_seg=el, e_chain=ec, ordered=ordered,
                           playable=playable, passes=bool(ordered and playable),
                           playable_ref_stale_only=bool(best <= ref_stale),
                           best=best,
                           guard_margin=float(ref_play - best),
                           guard_margin_f=float((ref_play - best) / max(ref_play, 1e-12)),
                           order_margin_chain_live=float(el - ec),
                           order_margin_live_react=float(er - el),
                           chain_vs_guard=float(ref_play - ec),
                           chain_vs_guard_f=float((ref_play - ec) / max(ref_play, 1e-12)))
            if ordered and playable and verdict[
                    "delta_star" if sfx == "p" else "delta_star_naive"] is None:
                verdict["delta_star" if sfx == "p" else "delta_star_naive"] = D
        verdict["rows"].append(row)
    out["verdict"] = verdict
    P(f"[verdict] guard ceiling {ref_play:.4f} = max(ref_stale {ref_stale:.4f}, "
      f"half_leg {half_leg:.4f})")
    for r in verdict["rows"]:
        for nm in ("predict", "naive"):
            v = r[nm]
            P(f"  D={r['delay']:2d} ({r['ms']:5.0f} ms) {nm:7s}  chain {v['e_chain']:.4f} <= live "
              f"{v['e_live_seg']:.4f} <= react {v['e_reactive']:.4f} ? ordered={v['ordered']} "
              f"(margins {v['order_margin_chain_live']:+.4f} / "
              f"{v['order_margin_live_react']:+.4f})  playable={v['playable']} "
              f"(guard margin {v['guard_margin']:+.4f} = {100 * v['guard_margin_f']:+.1f}%; "
              f"chain alone {v['chain_vs_guard']:+.4f} = {100 * v['chain_vs_guard_f']:+.1f}%) "
              f"-> {'PASS' if v['passes'] else 'no'}")
    P(f"[verdict] Delta* (PRIMARY, predicting incumbent) = {verdict['delta_star']}"
      + ("  -> a stored unit is the best playable option at this delay"
         if verdict["delta_star"] is not None
         else "  -> NO delay inverts the ordering under the pre-fixed rule; the round STOPS at the "
              "gate, and that is the finding. Do not extend or re-tune the sweep.")
      + f"   | Delta* (naive incumbent, offbook's operator) = {verdict['delta_star_naive']}")

    out["ledger"] = led.snapshot()
    out["wall_s"] = time.time() - t0
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
    delays: str = "0,1,2,3,4,5,6,8",
    # ---- THE DESIGN POINT. Every value below is read off the Phase A (`tempo.py`) report; the
    # defaults here are placeholders and the launch command states them explicitly.
    r: float = DEF_R,
    joint_damping: float = DEF_DAMPING,
    lookahead_gamma: float = 0.0,
    vel_pen_mid: float = 0.0,
    react_look: int = 12,
    look_grid: str = "6,12,30",
    calp: str = "1:1024:8,5:4096:12",
    # ----
    h_seg: str = DEF_H_SEG,
    h_app: int = DEF_H_APP,
    n_warm: int = 20,
    n_between: int = 5,
    n_eval: int = 24,
    n_score: int = 48,
    n_harvest: int = 4,
    n_lib: int = 8,
    curl_b: float = 0.0,
    push_a: float = 0.0,
    patch_seg: int = 0,
    patch_sigma: float = 0.08,
    push_sigma: float = 0.0,
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,1.1,0.8",
    gear: float = 8.0,
    frame_skip: int = 10,
    timestep: float = 0.002,
    wrap_limit: float = 3.0,
    q_jit: float = 0.15,
    null_jit: float = 0.30,
    start_mode: str = "iso",
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
    trace_window: int = 10,
    k_shoot: int = 1024,
    cem_iters: int = 8,
    cem_elite: int = 32,
    cem_init_sigma: float = 0.8,
    vel_pen: float = 0.0,
    w_waypoint: float = 16.0,
    plan_max_elems: int = 40_000_000,
    batch: int = 16,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    aud_horizon: int = 0,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500
        fm_steps = 800; fm_steps_boot = 400
        n_warm = 2; n_between = 1; n_eval = 8; n_score = 12; batch = 8
        n_harvest = 1; n_lib = 3
        delays = "0,4"; look_grid = "12"
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,5:256:4"
        tag = tag or "psmoke"
    tag = tag or "p0"
    cfg = dict(
        tag=tag, seed=seed, R=r,
        delays=[int(x) for x in delays.split(",") if x],
        look_grid=[int(x) for x in look_grid.split(",") if x],
        waypoints=[[float(v) for v in p.split(",")] for p in waypoints(r).split(";") if p],
        h_seg=[int(v) for v in h_seg.split(",")], h_app=h_app,
        center=list(DEF_CENTER), angles=list(DEF_ANGLES),
        n_warm=n_warm, n_between=n_between, n_eval=n_eval, n_score=n_score,
        n_harvest=n_harvest, n_lib=n_lib,
        react_look=react_look, lookahead_gamma=lookahead_gamma, vel_pen_mid=vel_pen_mid,
        curl_b=curl_b, push_a=push_a, patch_seg=patch_seg, patch_sigma=patch_sigma,
        push_sigma=(push_sigma or patch_sigma), patch_center=None, push_center=None,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, timestep=timestep,
        wrap_limit=wrap_limit, q_jit=q_jit, null_jit=null_jit, start_mode=start_mode,
        pool_ou=pool_ou, pool_reach=pool_reach, exclude_w=exclude_w, ep_len=ep_len, n_par=n_par,
        op_q_range=op_q_range, sigma_u=sigma_u, ou_sigma=ou_sigma, ou_theta=ou_theta,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, fm_steps_boot=fm_steps_boot, adapt_lr=adapt_lr, n_grad=n_grad,
        replay_frac=replay_frac, trace_window=trace_window,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, w_waypoint=w_waypoint,
        plan_max_elems=plan_max_elems,
        calp={p.split(":")[0]: [int(p.split(":")[1]), int(p.split(":")[2])]
              for p in calp.split(",") if p},
        batch=batch, sigma_practice=sigma_practice, sigma_perf=sigma_perf, d_fb=d_fb,
        aud_horizon=aud_horizon, obs_delay=0, obs_predict=False,
    )
    if spawn:
        h = run_delay_gate.spawn(cfg)
        print(f"[spawn] delay_gate: {h.object_id}")
        print(f"[spawn] detached; results -> /data/practice_presto/{tag}/delay_gate/results.json")
        return
    o = run_delay_gate.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "delay_gate.json"), "w") as fh:
        json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {localdir}/delay_gate.json")
