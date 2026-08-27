"""PRESTO, PHASE A -- can this plant play a piece fast enough that a reflex delay actually binds?

WHAT THIS NODE IS FOR. `../../offbook/`'s Round 4 (`delay_gate.py`, run `d0`) taxed the reflex loop
with an observation delay and confirmed the mechanism exactly: degradation from Delta = 0 to 16 is
ordered by feedback consumption -- reactive (61 fb/piece) 63.1x, live-per-segment (4 fb) 2.69x,
chain (2 fb) 1.69x, segment tape (4 fb) 1.28x. And the gate still failed: the playable region ended
at Delta = 2 (40 ms) and reactive won all of it, so no delay opened a niche for a stored unit. The
standing reading (Jasper's, adopted) is that depth on THAT piece is geometric, not economic -- an
environment property -- because the piece was inherited verbatim from `legato/`, whose question was
a comparison among committed arms, and a 400 ms segment on a 20 ms control loop is a SLOW movement.

Presto changes the piece and the plant and nothing else (`world.py` is a byte-identical fork; gate
G-F asserts it). Two design constraints, both taken from our own record:

  * DIFFICULTY SMOOTH AND GLOBAL, NEVER LOCALISED. `../../../ballistic/README.md` (cut 4b): a localised
    needle is open-loop-INCOMPENSABLE -- even a perfect forward model cannot counteract a strong
    local kick feedforward, which is precisely why biology puts feedback there -- so it saturates
    every ballistic arm at "fail" and measures nothing about the model. Momentum/braking is the
    smooth axis and is open-loop-compensable. So `curl_b = 0` (legato's b14 patch is OFF) and the
    difficulty is carried by TURN RATE at low `joint_damping`.
  * TEMPO SET AGAINST THE REFLEX DELAY. `dt_ctrl = 0.02 s`, so a 6-step segment is 120 ms against a
    human proprioceptive loop of ~100 ms (Delta = 5). Closed-loop correction inside a segment is
    then physically impossible rather than forbidden -- offbook Round 4 rejected a feedback CAP as
    necessity-by-fiat and this node keeps that choice.

WHAT THIS GATE MEASURES, AND WHY EACH PIECE OF IT IS HERE.

  G-F  FORK FIDELITY. presto's `World` reproduces offbook's `World` bit-for-bit on the DONOR piece
       and DONOR plant. The arc's fork discipline: assert the copy, do not trust it.
  G-F2 THE ONE ADDITIVE FLAG'S DEFAULT. `obs_predict` (efference copy through the reflex delay --
       see `world.py`) is a strict no-op at Delta = 0 and when off, asserted bit-for-bit, so the
       naive operator offbook Round 4 ran stays available as the continuity read.
  G-T  THE TEMPO LADDER -- the design decision, taken on evidence. Circumradius R (the speed knob:
       mean leg = 1.176 R, so mean tip speed = 1.176 R / 0.12 s) x `joint_damping` (the momentum
       knob). Per cell, after a short online warm-up on the piece:
         `hold`       zero commands after the approach -- the do-nothing floor, so "played" and
                      "did not play" are distinguishable numbers rather than a vibe;
         `ref_stale`  ballistic-per-segment under the PRE-PRACTICE forward model, undelayed -- the
                      arm-neutral playability anchor offbook used, carried unchanged;
         `react`      reactive MPC at Delta in {0, 2, 4, 6} on offbook's NAIVE delayed
                      observation (act on where you were) -- the continuity read;
         `react_pred` the same controller with EFFERENCE COPY through the delay: it acts on where
                      its own forward model says it now is, given the commands it has already
                      issued. This is the honest STRONG incumbent, and the reason it is here is
                      that a delay a nervous system can bridge with its own forward model is not
                      a delay that forces anything to be stored. Its residual error is exactly the
                      unpredictable part -- the motor noise the efference copy does not contain --
                      which is the physically correct statement of what a reflex delay costs;
         `live_seg`   ballistic per segment under the PRACTISED model at Delta in {0, 4} -- the
                      FEASIBILITY PRECONDITION (`ballistic/` 4b #2: the matched open-loop controller
                      must actually reach, or the axis measures planner starvation);
         `live_chain` one plan over the whole 30-step phrase -- the same precondition one level up,
                      and the direct read on whether the phrase is past the composition horizon.
  G-H  THE COMPOSITION HORIZON, re-measured on THIS plant by `legato` G5a's method (roll the FM
       open-loop along EXECUTED commands, first step where median tip divergence crosses a
       threshold). `arm_substrate` P4 measured 20-23 steps at n=3 on the donor plant; a faster,
       lower-damped plant need not have the same one, and the whole design ("segment under, chain
       over") is a claim about this number.
  G-S  SEAM INFORMATION: per-seam posture spread divided by matched repeat-noise. offbook's G-S
       found the donor's closing seam sat at its own noise floor (1.12x) and `key_frozen` collapsed
       exactly there. A keyed library needs the seams to carry posture information; this says
       whether they do.
  G-P  THE PLANNER LADDER (`arm_substrate` P3, the substrate's single biggest confound): k_shoot at
       the segment span and at the phrase span, plus the reactive controller's lookahead. If error
       is still falling at the top of the ladder the arm is planner-starved and every span
       comparison below is noise.
  G-L  CONTENT CALIBRATION -- the legato-quality recipe, so "measured content still passes" is a
       measured claim before any delay is applied. Tapes harvested at `sigma_perf` (not at practice
       noise), auditioned CROSS-STATE IN THE PLANT against a held-out seam-matched score set
       (`legato/legato.py::compile_unit`), committed as a keyed library (`World.select_library`) and
       as the single argmin-of-mean tape (`World.select_fixed`), at the segment span and the phrase
       span, read at Delta in {0, 4}.

NOTHING HERE PRE-REGISTERS AN OUTCOME. The design point is read off G-T after the fact; the only
thing fixed in advance is what gets measured.

Run:
    modal run mjc/practice/accompanist/presto/tempo.py::tempo --quick --tag tsmoke
    modal run --detach mjc/practice/accompanist/presto/tempo.py::tempo --spawn --tag t0 --seed 0
"""

import json
import os
import time

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.accompanist.presto.piece import (R_LADDER, DAMPING_LADDER, DEF_H_SEG, DEF_H_APP,
                                       DEF_CENTER, DEF_ANGLES, waypoints, legs, mean_leg)

# The DONOR configuration, for gate G-F only. Verbatim from `../../offbook/piece.py`; imported by
# VALUE rather than by import so this module never touches a sibling node's Modal entrypoints.
DONOR_WPS = "0.0359,0.5248;0.3820,0.2739;0.6029,0.6696;0.1968,0.7785"
DONOR_H_SEG = "20,20,20"
DONOR_H_APP = 14
DONOR_PATCH_SEG = 1
DONOR_CURL_B = 14.0
DONOR_DAMPING = 0.5


@app.function(gpu="L4", memory=32768, timeout=28800, volumes={DATA_DIR: volume})
def run_tempo(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler
    from mjc.practice.accompanist.presto.world import World, Ledger, start_postures, elite_for

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    led = Ledger()
    out = {"config": cfg, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_presto", cfg["tag"], "tempo")
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print("[tempo]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    def base_cfg(**kw):
        c = dict(cfg)
        c.update(kw)
        return c

    # ================================================================= G-F: fork fidelity
    # The DONOR path -- offbook's piece, offbook's plant -- run in both worlds at the same seed.
    # Nothing presto changes lives in `world.py`, but "cannot" is what a gate is for.
    t0 = time.time()
    from mjc.practice.offbook.world import World as OffbookWorld

    cf = base_cfg(waypoints=[[float(v) for v in p.split(",")] for p in DONOR_WPS.split(";") if p],
                  h_seg=[int(v) for v in DONOR_H_SEG.split(",")], h_app=DONOR_H_APP,
                  patch_seg=DONOR_PATCH_SEG, curl_b=DONOR_CURL_B, patch_center=None,
                  push_center=None, joint_damping=DONOR_DAMPING, react_look=20)
    WP, WO = World(cf, device), OffbookWorld(cf, device)
    ledP, ledO = Ledger(), Ledger()
    q_f = start_postures(WP.qc, WP.Ls, 8, np.random.default_rng(cfg["seed"] + 31),
                         cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])
    fmP = WP.mlp(cfg["seed"] + 40)
    fmO = copy.deepcopy(fmP)
    S1, U1, S21, _ = WP.collect_ou(1200, np.random.default_rng(cfg["seed"] + 11))
    WP.set_norm(S1, U1, S21); WO.set_norm(S1, U1, S21)
    oA = WP.traverse(fmP, WP.routing("reactive"), q_f, np.random.default_rng(cfg["seed"] + 32),
                     0.06, ledP, who="instrument", kind="gF")
    oB = WO.traverse(fmO, WO.routing("reactive"), q_f, np.random.default_rng(cfg["seed"] + 32),
                     0.06, ledO, who="instrument", kind="gF")
    gF = {k: float(np.nanmax(np.abs(np.asarray(oA[k], np.float64)
                                    - np.asarray(oB[k], np.float64))))
          for k in ("e_piece", "e_seg", "acts", "acts_raw", "tips", "launch", "final")}
    gF["fb_equal"] = bool(oA["n_fb"] == oB["n_fb"])
    gF["t_equal"] = bool(abs(oA["t_total"] - oB["t_total"]) < 1e-12)
    gF["pass"] = bool(max(v for k, v in gF.items() if isinstance(v, float)) == 0.0
                      and gF["fb_equal"] and gF["t_equal"])
    out["gF"] = gF
    P(f"[G-F] fork fidelity vs offbook on the DONOR piece+plant: max|delta| over 7 arrays = "
      f"{max(v for k, v in gF.items() if isinstance(v, float)):.3e}  pass={gF['pass']}")
    # G-F2: the one additive flag's default. `obs_predict` must be a strict no-op both when it is
    # off and (with it on) at Delta = 0, or the continuity read against offbook Round 4 is gone.
    cf2 = dict(cf); cf2["obs_predict"] = True
    WQ = World(cf2, device)
    WQ.set_norm(S1, U1, S21)
    oC = WQ.traverse(copy.deepcopy(fmP), WQ.routing("reactive"), q_f,
                     np.random.default_rng(cfg["seed"] + 32), 0.06, Ledger(),
                     who="instrument", kind="gF2")
    gF2 = {k: float(np.nanmax(np.abs(np.asarray(oA[k], np.float64)
                                     - np.asarray(oC[k], np.float64))))
           for k in ("e_piece", "e_seg", "acts", "acts_raw", "tips", "launch", "final")}
    gF2["pass"] = bool(max(gF2.values()) == 0.0)
    out["gF2"] = gF2
    P(f"[G-F2] obs_predict is a no-op at delay 0: max|delta| = {max(v for v in gF2.values() if isinstance(v, float)):.3e} "
      f"pass={gF2['pass']}")
    del WP, WO, WQ, oA, oB, oC
    save()

    # ================================================================= the ladder
    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}
    cells = {}
    out["cells"] = cells
    out["ladder"] = dict(R=list(cfg["r_ladder"]), damping=list(cfg["damp_ladder"]),
                         delays=list(cfg["delays"]), h_seg=cfg["h_seg"], h_app=cfg["h_app"])

    for damp in cfg["damp_ladder"]:
        # ---- the pools and the pre-practice forward model depend on the PLANT only (the piece
        # never enters `collect_ou` / `collect_reach`, and with `curl_b = 0` `exclude_region` keeps
        # everything), so they are built once per damping and shared across the R ladder. That is
        # not an economy: it is what makes R a one-variable step within a damping row.
        td = time.time()
        cd = base_cfg(joint_damping=damp,
                      waypoints=[[float(v) for v in p.split(",")]
                                 for p in waypoints(cfg["r_ladder"][0]).split(";") if p],
                      h_seg=[int(v) for v in cfg["h_seg"].split(",")], h_app=cfg["h_app"])
        W0w = World(cd, device)
        S1, U1, S21, _ = W0w.collect_ou(cfg["pool_ou"], np.random.default_rng(cfg["seed"] + 11))
        W0w.set_norm(S1, U1, S21)
        boot = W0w.mlp(cfg["seed"] + 40)
        W0w.train_steps(boot, torch.optim.Adam(boot.parameters(), lr=cfg["fm_lr"]),
                        S1, U1, S21, cfg["fm_steps_boot"], np.random.default_rng(cfg["seed"] + 300))
        gs = make_arm_goal_sampler(W0w.Ls, cfg["reach_amp"], cfg["reach_lo"], cfg["reach_hi"])
        S2_, U2_, S22, _ = W0w.collect_reach(cfg["pool_reach"],
                                             np.random.default_rng(cfg["seed"] + 12), boot, gs)
        Se = np.concatenate([S1, S2_]); Ue = np.concatenate([U1, U2_])
        S2e = np.concatenate([S21, S22])
        W0w.set_norm(Se, Ue, S2e)
        fm_stale = W0w.mlp(cfg["seed"] + 41)
        W0w.train_steps(fm_stale, torch.optim.Adam(fm_stale.parameters(), lr=cfg["fm_lr"]),
                        Se, Ue, S2e, cfg["fm_steps"], np.random.default_rng(cfg["seed"] + 301))
        P(f"[plant d={damp}] pools {len(Se)} transitions, stale FM trained in "
          f"{time.time() - td:.0f}s")

        for R in cfg["r_ladder"]:
            key = f"d{damp}_R{R}"
            tc = time.time()
            c = base_cfg(joint_damping=damp,
                         waypoints=[[float(v) for v in p.split(",")]
                                    for p in waypoints(R).split(";") if p],
                         h_seg=[int(v) for v in cfg["h_seg"].split(",")], h_app=cfg["h_app"])
            W = World(c, device)
            NS = W.n_seg
            W.set_norm(Se, Ue, S2e)
            cell = dict(R=R, damping=damp, n_seg=NS, h_phrase=int(W.H_phrase),
                        legs=legs(R), mean_leg=mean_leg(R),
                        nominal_speed=mean_leg(R) / (W.H_seg[0] * W.dt_ctrl),
                        tip_of_qc=[float(x) for x in W.tip(np.concatenate(
                            [W.qc, np.zeros_like(W.qc)])[None, :])[0]])
            cells[key] = cell

            def geom(m, sd):
                return start_postures(W.qc, W.Ls, m, np.random.default_rng(sd),
                                      cfg["q_jit"], cfg["null_jit"], cfg["start_mode"])

            fm_app = copy.deepcopy(fm_stale)
            for p_ in fm_app.parameters():
                p_.requires_grad_(False)

            def app_plan(q, sd):
                st = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
                pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                               wp_mask=W.approach_mask())
                return pf(st, np.tile(W.goals[0][None, :], (len(q), 1)),
                          np.random.default_rng(sd))[0]

            # ---- warm-up: the piece is LEARNED undelayed; the delay is a decision constraint
            # applied afterwards (offbook Round 4's convention, kept).
            R_REACT = W.routing("reactive")
            fm = copy.deepcopy(fm_stale)
            optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
            RX, RY = W.tensors(Se, Ue, S2e)
            brng = np.random.default_rng(cfg["seed"] + 800)
            buf = []
            for cyc in range(1, int(cfg["n_warm"]) + 1):
                qp = geom(cfg["batch"], cfg["seed"] + 9000 + cyc)
                pr = W.traverse(fm, R_REACT, qp, np.random.default_rng(cfg["seed"] + 10_000 + cyc),
                                cfg["sigma_practice"], led, who="agent", kind="practice",
                                approach_plan=app_plan(qp, cfg["seed"] + 9500 + cyc), collect=True,
                                look=cfg["react_look"])
                buf.append(pr["trans"])
                if len(buf) > cfg["trace_window"]:
                    buf.pop(0)
                PX, PY = W.tensors(*[np.concatenate([b[i] for b in buf]) for i in range(3)])
                W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                               cfg["replay_frac"])
            cell["warm_s"] = time.time() - tc

            # THREE DISJOINT geometry sets. `q_hv` produces the candidate tapes, `q_sc` the
            # seam states they are auditioned against, `q_ev` the states everything is GRADED on.
            # Auditioning a library on the states it will be scored on is `etude` E-3b's winner's
            # curse with extra steps, and legato's `score_set` / `candidates` split exists for
            # exactly this reason.
            q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)
            plan_ev = app_plan(q_ev, cfg["seed"] + 5250)
            q_hv = geom(cfg["n_eval"], cfg["seed"] + 5400)
            plan_hv = app_plan(q_hv, cfg["seed"] + 5450)
            q_sc = geom(cfg["n_score"], cfg["seed"] + 5500)
            plan_sc = app_plan(q_sc, cfg["seed"] + 5550)

            look_now = [int(cfg["react_look"])]     # per-cell; CAL rewrites it, cfg is untouched

            def run(net, routing, sd, kind, delay=0, look=None, collect=False, on="ev",
                    predict=False, stop_seg=None):
                q, pl = {"ev": (q_ev, plan_ev), "hv": (q_hv, plan_hv),
                         "sc": (q_sc, plan_sc)}[on]
                W.obs_delay = int(delay)
                W.obs_predict = bool(predict)
                o = W.traverse(net, routing, q, np.random.default_rng(cfg["seed"] + sd),
                               cfg["sigma_perf"], led, who="instrument", kind=kind,
                               approach_plan=pl, collect=collect, stop_seg=stop_seg,
                               look=(look or look_now[0]))
                W.obs_delay = 0
                W.obs_predict = False
                return o

            def score(o):
                sp = np.linalg.norm(np.diff(o["tips"], axis=1), axis=2) / W.dt_ctrl
                return dict(e=float(np.nanmean(o["e_piece"])),
                            e_med=float(np.nanmedian(o["e_piece"])),
                            by_seg=[float(np.nanmedian(o["e_seg"][:, k])) for k in range(NS)],
                            fb=float(o["n_fb"]),
                            v_mean=float(np.nanmean(sp)), v_max=float(np.nanmax(sp)))

            # ---- the do-nothing floor (no knob touches it: zero commands after the approach)
            R_BALL = W.routing("plan_launch", groups=[1] * NS, **CALP[1])
            cell["hold"] = score(run(fm, W.routing(
                "fixed", groups=[NS],
                fixed=np.zeros((W.H_phrase, W.AD), np.float32)), 7100, "hold"))
            cell["half_leg"] = 0.5 * mean_leg(R)

            # ================================================================= CAL
            # THE CONTROLLER'S COST SHAPING, CALIBRATED PER CELL BEFORE ANYTHING ELSE IS READ.
            # The first presto smoke found the donor's inherited shaping is wrong on this piece and
            # said so twice: a `plan_launch` unit committed to a 6-step span with `w_waypoint = 16`
            # on its last step and NO terminal velocity penalty hit its waypoint at any speed it
            # liked (peak tip speed 17.9 m/s against reactive's 7.5) and handed the next segment an
            # unplayable state -- per-segment error running 0.09 -> 0.27 across the phrase. That is
            # `legato` l1's measured disease ("hit waypoints while ignoring HANDOFF VELOCITY") on a
            # piece with three times the seam rate. Two repairs, both swept, neither assumed:
            #
            #   (a) `lookahead_gamma` -- the live unit PLANS one segment past its commitment with
            #       the look-ahead cost discounted by gamma, issuing only the committed part
            #       (legato l2's op). l2 measured gamma = 0 BEST on the donor piece, whose seams
            #       arrive every 400 ms, and recorded it as F3, "there is no controller-level seam
            #       law". Presto's seams arrive every 120 ms, which is exactly why it is re-asked.
            #   (b) `vel_pen_mid` -- a terminal joint-velocity penalty on every non-final window.
            #
            # THE SELECTION RULES, WRITTEN DOWN BEFORE THE GRID IS READ (legato F2: a shared
            # calibration knob may not be chosen by one arm's minimum error, because a knob that
            # minimises error can silently minimise MEASURABILITY -- l0 chose `vel_pen_mid` by
            # reactive's minimum and left only 3% of the outcome attributable to model quality):
            #   gamma*        = argmin of the PRACTISED live-per-segment error. gamma is a knob of
            #                   the LIVE unit alone, so tuning it strengthens the incumbent side of
            #                   this node's comparison, which is the conservative direction.
            #   vel_pen_mid*  = argmax of the USABLE RANGE (stale minus practised live-per-segment
            #                   error, an arm-neutral reference), subject to a competence guard:
            #                   the practised error must be <= half the do-nothing floor. Reactive's
            #                   number is reported at every value because the knob is shared.
            #   look*         = argmin of reactive error at Delta = 0 -- the vanilla baseline, so
            #                   the incumbent is not handicapped by an inherited lookahead sized for
            #                   a piece with 20-step segments. The full look x Delta grid is
            #                   reported in G-P, because a delayed controller prefers a longer one
            #                   and Phase B gives reactive its per-Delta best.
            cal = {"gamma": {}, "vel_pen_mid": {}, "look": {}}
            for gm in cfg["gamma_grid"]:
                W.cfg["lookahead_gamma"] = float(gm)
                cal["gamma"][str(gm)] = score(run(fm, R_BALL, 8000, f"cal_g{gm}"))
            g_star = min(cal["gamma"], key=lambda g: cal["gamma"][g]["e"])
            W.cfg["lookahead_gamma"] = float(g_star)

            guard = 0.5 * cell["hold"]["e"]
            for vp in cfg["vpm_grid"]:
                W.cfg["vel_pen_mid"] = float(vp)
                cal["vel_pen_mid"][str(vp)] = dict(
                    live_seg=score(run(fm, R_BALL, 8010, f"cal_v{vp}"))["e"],
                    react=score(run(fm, R_REACT, 8020, f"cal_vr{vp}"))["e"],
                    live_seg_stale=score(run(fm_stale, R_BALL, 8030, f"cal_vs{vp}"))["e"])
            ok = [v for v, r in cal["vel_pen_mid"].items() if r["live_seg"] <= guard]
            pool_v = ok or list(cal["vel_pen_mid"])          # guard empty -> fall back to all
            v_star = max(pool_v, key=lambda v: (cal["vel_pen_mid"][v]["live_seg_stale"]
                                                - cal["vel_pen_mid"][v]["live_seg"]))
            W.cfg["vel_pen_mid"] = float(v_star)

            for lk in cfg["look_grid"]:
                cal["look"][str(lk)] = score(run(fm, R_REACT, 8040, f"cal_l{lk}",
                                                 look=int(lk)))["e"]
            l_star = int(min(cal["look"], key=lambda l: cal["look"][l]))
            look_now[0] = l_star
            cal["chosen"] = dict(lookahead_gamma=float(g_star), vel_pen_mid=float(v_star),
                                 react_look=l_star, guard=guard,
                                 guard_satisfied=bool(ok),
                                 w_waypoint=float(W.cfg.get("w_waypoint", 0.0)))
            cell["cal"] = cal
            P(f"[CAL {key}] gamma " + " ".join(
                "{}:{:.4f}(vmax {:.1f})".format(g, v["e"], v["v_max"])
                for g, v in cal["gamma"].items())
              + " | vel_pen_mid " + " ".join(
                  "{}: seg {:.4f} react {:.4f} stale {:.4f}".format(
                      vp, r["live_seg"], r["react"], r["live_seg_stale"])
                  for vp, r in cal["vel_pen_mid"].items())
              + " | look " + " ".join("{}:{:.4f}".format(l, e) for l, e in cal["look"].items())
              + f" -> CHOSE gamma={g_star} vel_pen_mid={v_star} look={l_star} "
                f"(guard {guard:.4f}, satisfied={bool(ok)})")
            save()

            # ---- the arm-neutral playability anchor, AT THE CALIBRATED KNOBS
            cell["ref_stale"] = score(run(fm_stale, R_BALL, 7200, "ref_stale"))

            # ---- the incumbent, across delay
            for D in cfg["delays"]:
                cell[f"react_d{D}"] = score(run(fm, R_REACT, 7700, f"react_d{D}", delay=D))
                cell[f"reactp_d{D}"] = score(run(fm, R_REACT, 7700, f"reactp_d{D}", delay=D,
                                                 predict=True))
            # ---- the feasibility precondition, at both spans
            R_CHAIN = W.routing("plan_launch", groups=[NS], **CALP[min(NS, max(CALP))])
            for D in cfg["delays_short"]:
                cell[f"live_seg_d{D}"] = score(run(fm, R_BALL, 7300, f"live_seg_d{D}", delay=D))
                cell[f"live_segp_d{D}"] = score(run(fm, R_BALL, 7300, f"live_segp_d{D}", delay=D,
                                                    predict=True))
                cell[f"live_chain_d{D}"] = score(run(fm, R_CHAIN, 7400, f"live_chain_d{D}",
                                                     delay=D))
            P(f"[G-T {key}] speed {cell['nominal_speed']:.2f} m/s | hold {cell['hold']['e']:.4f} "
              f"| ref_stale {cell['ref_stale']['e']:.4f} | half_leg {cell['half_leg']:.4f} | "
              + " ".join(f"react_d{D} {cell[f'react_d{D}']['e']:.4f}"
                          f"/p{cell[f'reactp_d{D}']['e']:.4f}" for D in cfg["delays"])
              + " | " + " ".join(f"live_seg_d{D} {cell[f'live_seg_d{D}']['e']:.4f}"
                                 for D in cfg["delays_short"])
              + " | " + " ".join(f"live_chain_d{D} {cell[f'live_chain_d{D}']['e']:.4f}"
                                 for D in cfg["delays_short"]))
            save()

            # ---- G-H: the composition horizon, on THIS plant, by legato G5a's method
            ex = run(fm, R_REACT, 7500, "gH_exec", collect=True)
            cmds = np.nan_to_num(ex["acts"], nan=0.0)
            true = W.true_tips(ex["launch"], cmds)
            gh = {}
            for nm, net in (("stale", fm_stale), ("practised", fm)):
                dv = np.median(np.linalg.norm(W.fm_rollout_tips(net, ex["launch"], cmds) - true,
                                              axis=2), 0)
                # thresholds: legato G5a's 0.05 / 0.02 m absolute, PLUS a task-scaled one. A
                # horizon is a level crossing and the level was chosen on a piece whose tip moved
                # 0.021 m per step; presto's moves faster, so the absolute threshold is a different
                # number of steps of travel. `half_leg` is the tolerance the task itself implies.
                gh[nm] = dict(curve=[float(x) for x in dv],
                              h_005=int(next((i + 1 for i, v in enumerate(dv) if v > 0.05),
                                             len(dv) + 1)),
                              h_002=int(next((i + 1 for i, v in enumerate(dv) if v > 0.02),
                                             len(dv) + 1)),
                              h_task=int(next((i + 1 for i, v in enumerate(dv)
                                               if v > cell["half_leg"]), len(dv) + 1)),
                              err_at_seams=[float(dv[int(h) - 1]) for h in W.seg_hi])
            # one-step FM error ON THE TASK DISTRIBUTION (`arm_substrate` methodological finding 1)
            S_, U_, S2_ = ex["trans"]
            pred = W.fm_delta(fm, S_, U_)
            dtrue = np.linalg.norm(S2_ - S_, axis=1)
            gh["one_step_task"] = float(np.mean(np.linalg.norm((S2_ - S_) - pred, axis=1)))
            gh["one_step_task_rel"] = float(np.mean(np.linalg.norm((S2_ - S_) - pred, axis=1)
                                                    / np.maximum(dtrue, 1e-6)))
            dp = np.linalg.norm(S2e[:4000] - Se[:4000], axis=1)
            ep = np.linalg.norm((S2e[:4000] - Se[:4000])
                                - W.fm_delta(fm, Se[:4000], Ue[:4000]), axis=1)
            gh["one_step_pool"] = float(np.mean(ep))
            gh["one_step_pool_rel"] = float(np.mean(ep / np.maximum(dp, 1e-6)))
            cell["gH"] = gh
            P(f"[G-H {key}] composition horizon @0.05 m: stale {gh['stale']['h_005']} "
              f"practised {gh['practised']['h_005']} (@0.02 m: {gh['stale']['h_002']}/"
              f"{gh['practised']['h_002']}; @half_leg {gh['stale']['h_task']}/"
              f"{gh['practised']['h_task']}) | one-step FM err task {gh['one_step_task']:.4f} "
              f"pool {gh['one_step_pool']:.4f} | relative task "
              f"{gh['one_step_task_rel']:.3f} pool {gh['one_step_pool_rel']:.3f}")

            # ---- G-S: does each seam carry posture information above its own repeat noise?
            m_rep = int(cfg["g_s_batch"] // cfg["g_s_repeats"])
            q_rep = np.repeat(geom(m_rep, cfg["seed"] + 5300), cfg["g_s_repeats"], axis=0)
            W.obs_delay = 0
            o_rep = W.traverse(fm, R_REACT, q_rep, np.random.default_rng(cfg["seed"] + 7600),
                               cfg["sigma_perf"], led, who="instrument", kind="gS",
                               approach_plan=app_plan(q_rep, cfg["seed"] + 5350),
                               look=cfg["react_look"])
            gs_ = {}
            for k in range(NS):
                L = np.asarray(o_rep["launches"][k], np.float64)
                grp = L.reshape(m_rep, cfg["g_s_repeats"], -1)
                spread = float(np.linalg.norm(grp.mean(1).std(0)))
                noise = float(np.linalg.norm(grp.std(1).mean(0)))
                tips = W.tip(L).reshape(m_rep, cfg["g_s_repeats"], 2)
                gs_[str(k)] = dict(posture_spread=spread, repeat_noise=noise,
                                   ratio=float(spread / max(noise, 1e-9)),
                                   tip_spread=float(np.linalg.norm(tips.mean(1).std(0))),
                                   tip_noise=float(np.linalg.norm(tips.std(1).mean(0))),
                                   speed=float(np.linalg.norm(L[:, W.n:], axis=1).mean()))
            cell["gS"] = gs_
            P(f"[G-S {key}] posture spread / repeat noise by seam: "
              + " ".join(f"{k}:{v['ratio']:.2f}x" for k, v in gs_.items()))
            save()

            # ---- G-L: the legato-quality content recipe, and what it scores undelayed
            # harvest at PERFORMANCE noise (offbook harvested at sigma_practice = 0.15 and its
            # stored strategies never cleared the anchor); score set held out from the harvest.
            harv = [run(fm, R_REACT, 7800 + 7 * i, f"harvest{i}", on="hv")
                    for i in range(cfg["n_harvest"])]
            cands_all = np.concatenate([h["acts"] for h in harv])

            # THE SCORE SET IS BUILT GREEDILY, UNDER THE CONFIGURATION THAT WILL DEPLOY IT.
            # legato's `score_set` is explicit about this -- "launch states produced by the CURRENT
            # performance configuration at performance tempo, truncated at the span's start" -- and
            # the first presto smoke shows exactly what happens when it is not: auditioned against
            # seam states a REACTIVE traversal produced, the deployed tape arm's per-segment error
            # ran 0.095 -> 0.430 across the phrase while its own audition said 0.042-0.091. The
            # library was being selected for a distribution it would never see, because each tape
            # it commits moves the next seam off the reactive manifold. So segment k's score set is
            # harvested from a traversal that plays the ALREADY-COMMITTED tapes for 0..k-1 and
            # stops there (`stop_seg = k`), which costs almost nothing (no planner runs) and is the
            # only version of the op that is measuring the right thing.
            # EACH library is built greedily against ITS OWN prefix. `select_library` and
            # `select_fixed` are the same op on the same candidates at the same price (legato's
            # point: the library's advantage is information use, not budget) -- but the SCORE SET is
            # part of the op, and a keyed prefix and a fixed prefix put the body in different
            # places. Sharing one matrix graded the fixed arm on a distribution it was never
            # selected for and the smoke returned 0.61, worse than doing nothing. So two greedy
            # passes; the extra cost is five `stop_seg` traversals that run no planner at all.
            units_key, units_fix, gl = [], [], {}
            S0 = {}
            for k in range(NS):
                for tag_, units_, pick in (("key", units_key, "lib"), ("fix", units_fix, "fixed")):
                    pre = W.routing(None, groups=[1] * NS,
                                    units=list(units_) + [{"kind": "reactive"}] * (NS - k))
                    sck = run(fm, pre, 7900 + 13 * k + (0 if tag_ == "key" else 5),
                              f"score_{tag_}{k}", on="sc", stop_seg=k)
                    S0k = np.asarray(sck["launches"][k], np.float32)
                    lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k])
                    cd_ = np.nan_to_num(cands_all[:, lo:hi, :], nan=0.0)
                    E, _ = W.audition(cd_, S0k, W.checkpoints(k, 1), led, who="instrument")
                    if pick == "lib":
                        u, li = W.select_library(E, cd_, S0k, cfg["n_lib"],
                                                 np.random.default_rng(cfg["seed"] + 954 + k))
                        gl[f"seg{k}"] = dict(
                            n_cand=int(len(cd_)), n_score=int(len(S0k)),
                            best_fixed=float(E.mean(1).min()),
                            per_state_oracle=float(E.min(0).mean()),
                            oracle_gain=float(E.mean(1).min() / max(E.min(0).mean(), 1e-9)),
                            n_distinct=int(li["n_distinct"]), cell_sizes=li["cell_sizes"],
                            seam_speed=float(np.linalg.norm(S0k[:, W.n:], axis=1).mean()))
                        S0[k] = S0k
                    else:
                        u, _ = W.select_fixed(E, cd_)
                    units_.append(u)
            ch = np.nan_to_num(cands_all, nan=0.0)
            Ec, _ = W.audition(ch, S0[0], W.checkpoints(0, NS), led, who="instrument")
            u_ck, lic = W.select_library(Ec, ch, S0[0], cfg["n_lib"],
                                         np.random.default_rng(cfg["seed"] + 970))
            u_cf, _ = W.select_fixed(Ec, ch)
            gl["chain"] = dict(n_cand=int(len(ch)), best_fixed=float(Ec.mean(1).min()),
                               per_state_oracle=float(Ec.min(0).mean()),
                               oracle_gain=float(Ec.mean(1).min() / max(Ec.min(0).mean(), 1e-9)),
                               n_distinct=int(lic["n_distinct"]), cell_sizes=lic["cell_sizes"])
            for D in cfg["delays_short"]:
                gl[f"seg_tape_d{D}"] = score(run(fm, W.routing(None, groups=[1] * NS,
                                                              units=units_key),
                                                 8100, f"seg_tape_d{D}", delay=D))
                gl[f"seg_tapep_d{D}"] = score(run(fm, W.routing(None, groups=[1] * NS,
                                                                units=units_key),
                                                  8100, f"seg_tapep_d{D}", delay=D, predict=True))
                gl[f"seg_fixed_d{D}"] = score(run(fm, W.routing(None, groups=[1] * NS,
                                                                units=units_fix),
                                                  8200, f"seg_fixed_d{D}", delay=D))
                gl[f"chain_d{D}"] = score(run(fm, W.routing(None, groups=[NS], units=[u_ck]),
                                              8300, f"chain_d{D}", delay=D))
                gl[f"chain_fixed_d{D}"] = score(run(fm, W.routing(None, groups=[NS],
                                                                  units=[u_cf]),
                                                    8400, f"chain_fixed_d{D}", delay=D))
            cell["gL"] = gl
            P(f"[G-L {key}] " + " | ".join(
                f"{nm}_d{D} {gl[f'{nm}_d{D}']['e']:.4f}"
                for D in cfg["delays_short"]
                for nm in ("seg_tape", "seg_fixed", "chain", "chain_fixed")))

            # ---- G-P: the planner ladder, at both spans, plus the incumbent's lookahead
            if cfg["planner_grid"]:
                gp = {"seg": {}, "chain": {}, "react_look": {}}
                for ks in cfg["planner_grid"]:
                    it = CALP[1]["cem_iters"]
                    gp["seg"][str(ks)] = score(run(fm, W.routing(
                        "plan_launch", groups=[1] * NS, k_shoot=int(ks), cem_iters=it,
                        cem_elite=elite_for(int(ks))), 8500, f"gp_seg{ks}"))["e"]
                    gp["chain"][str(ks)] = score(run(fm, W.routing(
                        "plan_launch", groups=[NS], k_shoot=int(ks),
                        cem_iters=CALP[min(NS, max(CALP))]["cem_iters"],
                        cem_elite=elite_for(int(ks))), 8600, f"gp_chain{ks}"))["e"]
                # the lookahead x delay grid. CAL already chose `look` at Delta = 0 (the vanilla
                # baseline); this is the record of how much a DELAYED controller would rather have
                # a longer one, which is what Phase B uses to give reactive its per-Delta best
                # rather than a single inherited number.
                for lk in cfg["look_grid"]:
                    gp["react_look"][str(lk)] = {
                        f"d{D}": score(run(fm, W.routing("reactive"), 8700,
                                           f"gp_look{lk}_d{D}", delay=D, look=int(lk)))["e"]
                        for D in cfg["delays_short"] if D > 0}
                cell["gP"] = gp
                P(f"[G-P {key}] k_shoot seg " + " ".join(f"{k}:{v:.4f}"
                                                         for k, v in gp["seg"].items())
                  + " | chain " + " ".join(f"{k}:{v:.4f}" for k, v in gp["chain"].items())
                  + " | look " + " ".join(
                      f"{k}:" + "/".join(f"{vv:.4f}" for vv in v.values())
                      for k, v in gp["react_look"].items()))

            cell["cell_s"] = time.time() - tc
            P(f"[cell {key}] done in {cell['cell_s']:.0f}s")
            save()

    out["ledger"] = led.snapshot()
    out["complete"] = True
    out["wall_s"] = time.time() - t0
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    return out


@app.local_entrypoint()
def tempo(
    quick: bool = False,
    spawn: bool = False,
    tag: str = "",
    seed: int = 0,
    r_ladder: str = ",".join(str(x) for x in R_LADDER),
    damp_ladder: str = ",".join(str(x) for x in DAMPING_LADDER),
    delays: str = "0,2,4,6",
    delays_short: str = "0,4",
    h_seg: str = DEF_H_SEG,
    h_app: int = DEF_H_APP,
    n_warm: int = 15,
    n_eval: int = 24,
    n_score: int = 48,
    n_harvest: int = 4,
    n_lib: int = 8,
    g_s_repeats: int = 6,
    g_s_batch: int = 48,
    react_look: int = 12,
    planner_grid: str = "256,1024,4096",
    gamma_grid: str = "0.0,0.5,1.0",
    vpm_grid: str = "0.0,0.1,0.3",
    look_grid: str = "6,12,30",
    # ---- the plant. Every knob below is the DONOR's except `joint_damping` (swept) and
    # `curl_b` / `push_a`, which are ZERO here: the patch is off by design (see the header).
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
    vel_pen_mid: float = 0.0,
    w_waypoint: float = 16.0,
    lookahead_gamma: float = 0.0,
    plan_max_elems: int = 40_000_000,
    calp: str = "1:1024:8,5:4096:12",
    batch: int = 16,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    aud_horizon: int = 0,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500
        fm_steps = 800; fm_steps_boot = 400
        n_warm = 2; n_eval = 8; n_score = 12; batch = 8; n_harvest = 1; n_lib = 3
        g_s_repeats = 4; g_s_batch = 12
        r_ladder = "0.13"; damp_ladder = "0.05"
        delays = "0,4"; delays_short = "0,4"
        planner_grid = "256"; look_grid = "12"
        gamma_grid = "0.0,0.5"; vpm_grid = "0.0,0.1"
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,5:256:4"
        tag = tag or "tsmoke"
    tag = tag or "t0"
    cfg = dict(
        tag=tag, seed=seed,
        r_ladder=[float(x) for x in r_ladder.split(",") if x],
        damp_ladder=[float(x) for x in damp_ladder.split(",") if x],
        delays=[int(x) for x in delays.split(",") if x],
        delays_short=[int(x) for x in delays_short.split(",") if x],
        h_seg=h_seg, h_app=h_app, n_warm=n_warm, n_eval=n_eval, n_score=n_score,
        n_harvest=n_harvest,
        n_lib=n_lib, g_s_repeats=g_s_repeats, g_s_batch=g_s_batch,
        react_look=react_look,
        planner_grid=[int(x) for x in planner_grid.split(",") if x],
        gamma_grid=[float(x) for x in gamma_grid.split(",") if x],
        vpm_grid=[float(x) for x in vpm_grid.split(",") if x],
        look_grid=[int(x) for x in look_grid.split(",") if x],
        center=list(DEF_CENTER), angles=list(DEF_ANGLES),
        curl_b=curl_b, push_a=push_a, patch_seg=patch_seg, patch_sigma=patch_sigma,
        push_sigma=(push_sigma or patch_sigma), patch_center=None, push_center=None,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=damp_ladder and float(damp_ladder.split(",")[0]),
        gear=gear, frame_skip=frame_skip, timestep=timestep, wrap_limit=wrap_limit,
        q_jit=q_jit, null_jit=null_jit, start_mode=start_mode,
        pool_ou=pool_ou, pool_reach=pool_reach, exclude_w=exclude_w, ep_len=ep_len, n_par=n_par,
        op_q_range=op_q_range, sigma_u=sigma_u, ou_sigma=ou_sigma, ou_theta=ou_theta,
        collect_k_shoot=collect_k_shoot, collect_cem_iters=collect_cem_iters,
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi,
        fm_hidden=fm_hidden, fm_layers=fm_layers, fm_lr=fm_lr, fm_batch=fm_batch,
        fm_steps=fm_steps, fm_steps_boot=fm_steps_boot, adapt_lr=adapt_lr, n_grad=n_grad,
        replay_frac=replay_frac, trace_window=trace_window,
        k_shoot=k_shoot, cem_iters=cem_iters, cem_elite=cem_elite,
        cem_init_sigma=cem_init_sigma, vel_pen=vel_pen, vel_pen_mid=vel_pen_mid,
        w_waypoint=w_waypoint, lookahead_gamma=lookahead_gamma, react_look_cfg=react_look,
        plan_max_elems=plan_max_elems,
        calp={p.split(":")[0]: [int(p.split(":")[1]), int(p.split(":")[2])]
              for p in calp.split(",") if p},
        batch=batch, sigma_practice=sigma_practice, sigma_perf=sigma_perf, d_fb=d_fb,
        aud_horizon=aud_horizon, obs_delay=0, obs_predict=False,
        waypoints=[[float(v) for v in p.split(",")] for p in waypoints(
            float(r_ladder.split(",")[0])).split(";") if p],
    )
    if spawn:
        h = run_tempo.spawn(cfg)
        print(f"[spawn] tempo: {h.object_id}")
        print(f"[spawn] detached; results -> /data/practice_presto/{tag}/tempo/results.json")
        return
    o = run_tempo.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "tempo.json"), "w") as fh:
        json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {localdir}/tempo.json")
