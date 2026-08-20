"""Round 2a -- the admissibility gates for COMMITTED EXECUTION ACROSS SEAMS.

Methodology: `rhm/practice/typed_gaps/` -- offline calibration -> admissibility gate -> main run,
and **a gate that fails is a finding, not a bug**. Nothing here is the experiment; it is the
measurement that says whether the experiment is askable on this world, and it fixes by measurement
every knob the main run will use.

Round 1 (`../fingering/`) settled the op taxonomy on ONE segment, inside the plant's composition
horizon, and live content won everything: one CEM plan from the current forward model at the
observed launch state beat every frozen op 1.6x on error at 2.3x less priced time. Round 2's named
unknown is where that stops being true. The claim under test is that **frozen, measured content
re-enters exactly where the committed span exceeds the model's composition horizon** -- a phrase
cannot be planned live in one launch because FM rollout error compounds past ~20-23 steps, while a
chain of measured, executed renditions has no such bound because the body, not the model, produced
it.

THREE GATES, each the one-level-up analogue of a round-1 gate.

  G5  THE COMPOSITION-HORIZON GATE -- the round's load-bearing gate. Does live plan-at-launch error
      degrade STRUCTURALLY with committed span? Two measurements, deliberately separated, because
      `arm_substrate` P3 says an undersized CEM can fake exactly this shape (and can even
      sign-invert the FM-quality axis):
        (a) FM COMPOSITION DIVERGENCE, a pure model quantity with no planner in it: roll the forward
            model open-loop along commands the body actually executed, and along a planned ballistic
            sequence, and measure predicted-vs-true tip error out to the full phrase span. This is
            the mechanism, and it cannot be planner noise because there is no planner in it.
        (b) LIVE PLAN-AT-LAUNCH ERROR VS RE-GROUNDING COUNT, with the planner given its CAL-P best
            shot AT EACH ACTION DIMENSION (60, 120, 180). The same three segments are executed with
            3, 2 and 1 committed groups; because the endpoint is the same waypoint in every case,
            the comparison is matched and the only difference is how many times the arm re-grounded.
      If live phrase-planning does NOT degrade, that is a major finding and gets reported, not
      tuned away.

  G6  SEAM INFORMATION AT PHRASE LEVEL (G1's analogue, one level up). Does the phrase-launch state
      carry information a committed chain could use, and how much information does FUSING give up?
      The second half is the new question: regress a chain's realised per-segment error on the
      phrase-launch state (what a fused unit can condition on) against the same regression on the
      SEAM state (what a segment-wise unit can condition on). The difference is the price of flying
      the seams blind, in the currency of predictable variance.

  G7  USABLE RANGE + NOISE FLOOR AT PHRASE GRANULARITY (G3's analogue), and the MASTERY CLOCKS
      re-measured on this piece. Round 1b's rule, applied per segment and to the phrase: read the
      held-out BALLISTIC descent (the quantity a committed open-loop unit actually competes
      against), smooth it, and take the first sustained entry within 1 sd of the final plateau.
      Reading the REACTIVE curve instead said "mastered" 39 cycles too early in round 1; open-loop
      competence is the later of the two clocks. The main run's commit schedule is read off this.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/legato/gates.py::gates --quick
    python3 mjc/practice/legato/launch_detached.py --fn gates --tag l0 --seed 0
    python3 mjc/practice/legato/analyze_gates.py --tag l0 --fetch
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder

# THE PIECE -- a closed 4-leg loop, picked by the pure-numpy FK search recorded in FILES.md.
# W0 = fk(q_center) = (0.1968, 0.7785) is the start tip and also the last waypoint, so leg D closes
# the loop. Legs: approach 0.300 m (H=14), then three drilled-class segments of 0.428 / 0.453 /
# 0.420 m (H=20 each), i.e. 0.021-0.023 m/step throughout -- the same tempo as round 1's 0.44 m /
# 20-step drilled segment, so a legato segment IS a fingering segment in difficulty class.
# Turns 86.4 / 96.8 / 104.2 / 72.6 deg: a closed quadrilateral's exterior angles sum to 360, so its
# turns average exactly 90 and round 1's "cos in [0.10, 0.75]" window is structurally unavailable.
# Every point on every leg has radius in [0.446, 0.901] of a 1.10 m arm and x >= 0.036, so the piece
# stays off the singular full-extension shell and in front of the base, while sweeping the arm from
# r=0.45 to r=0.90 -- M(q) changes a lot along the loop, which is what makes it a piece and not a
# translation.
DEF_WPS = "0.0359,0.5248;0.3820,0.2739;0.6029,0.6696;0.1968,0.7785"
# The curl patch sits on the midpoint of drilled segment 1 (leg C, W2 -> W3), i.e. the MIDDLE of the
# phrase: gate weight 0.018 at every waypoint and <= 0.023 at the closest approach of every other
# leg (round 1: 0.023; the etude: 0.044). Putting it there means the boundary a phrase commitment
# gives up -- W2, immediately before the hard passage -- is exactly the one that carries the most
# information, which is the sharpest available form of the bet.
DEF_PATCH_SEG = 1


@app.function(gpu="L4", memory=32768, timeout=21600, volumes={DATA_DIR: volume})
def run_gates(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.arm_env import fk
    from mjc.embodied import make_arm_goal_sampler, pool_diagnostics, cmd_state_corr
    from mjc.practice.legato.world import (World, Ledger, start_postures, null_direction, kmeans,
                                           elite_for)

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    n, AD = W.n, W.AD
    NS = W.n_seg
    led = Ledger()
    world = cfg["world"]
    out = {"config": cfg, "world": world, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_legato", cfg["tag"], world)
    os.makedirs(outdir, exist_ok=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    def P(*a):
        print(f"[{world}]", *a, flush=True)

    P(f"[setup] device={device} n={n} n_seg={NS} H_app={W.H_app} H_seg={W.H_seg} "
      f"H_phrase={W.H_phrase} patch_seg={W.patch_seg} curl_b={W.curl_b} push_a={W.push_a}")

    # ================================================================= C0 sanity + piece geometry
    rng = np.random.default_rng(cfg["seed"] + 1)
    qs = W.qc[None, :] + rng.uniform(-0.5, 0.5, (8, n))
    fk_err = 0.0
    for q in qs:
        W.env.set_state(q, np.zeros(n))
        fk_err = max(fk_err, float(np.linalg.norm(W.env.tip_pos() - fk(q, W.Ls))))
    p0 = fk(W.qc, W.Ls)
    wps = [p0] + [np.asarray(g, np.float64) for g in W.goals]
    legs = [(wps[i], wps[i + 1]) for i in range(len(wps) - 1)]
    tvec = np.linspace(0, 1, 41)[:, None]
    leg_pts = [a[None] + tvec * (b - a)[None] for a, b in legs]
    coss = []
    for i in range(len(legs) - 1):
        u = (legs[i][1] - legs[i][0]); u = u / np.linalg.norm(u)
        v = (legs[i + 1][1] - legs[i + 1][0]); v = v / np.linalg.norm(v)
        coss.append(float(u @ v))
    out["c0"] = dict(
        fk_vs_mujoco_max=fk_err, p0=p0.tolist(),
        waypoints=[w.tolist() for w in wps],
        leg_lengths=[float(np.linalg.norm(b - a)) for a, b in legs],
        m_per_step=[float(np.linalg.norm(b - a)) / h
                    for (a, b), h in zip(legs, [W.H_app] + list(W.H_seg))],
        turn_cos=coss,
        radius_range=[float(min(np.linalg.norm(p, axis=1).min() for p in leg_pts)),
                      float(max(np.linalg.norm(p, axis=1).max() for p in leg_pts))],
        gate_at_waypoints=[float(W.gate(w[None, :])[0]) for w in wps],
        gate_on_legs=[float(W.gate(p).max()) for p in leg_pts],
        patch_center=W.patch_center.tolist())
    c0 = out["c0"]
    P(f"[C0] fk vs mujoco {fk_err:.2e} | legs {[round(x,4) for x in c0['leg_lengths']]} "
      f"| m/step {[round(x,4) for x in c0['m_per_step']]} | turn cos "
      f"{[round(x,3) for x in c0['turn_cos']]} | radius {c0['radius_range'][0]:.3f}.."
      f"{c0['radius_range'][1]:.3f}")
    P(f"[C0] gate @ waypoints {[round(g,4) for g in c0['gate_at_waypoints']]} | max on each leg "
      f"{[round(g,4) for g in c0['gate_on_legs']]}")
    assert fk_err < 1e-9, "analytic FK disagrees with MuJoCo -- the CEM cost would score a "\
                          "different arm than the simulator executes"
    save()

    # ================================================================= the diet (on-policy)
    # Structurally identical to round 1's setup block and seeded from the same offsets.
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
    fm_app = copy.deepcopy(fm0)                    # the FROZEN approach controller
    for p in fm_app.parameters():
        p.requires_grad_(False)
    diag = pool_diagnostics(Se, Ue, n)
    P(f"[diet] ou={len(S1)} reach={len(S2_)} | exclude kept {keep_frac:.3f} | speed p95 "
      f"{diag.get('speed_p95', float('nan')):.2f}")

    # ================================================================= geometries + the approach
    def geom(m, seed, mode=None):
        return start_postures(W.qc, W.Ls, m, np.random.default_rng(seed),
                              cfg["q_jit"], cfg["null_jit"], mode or cfg["start_mode"])

    q_rt = geom(cfg["n_rt"], cfg["seed"] + 5000)          # metering (agent-side)
    q_sc = geom(cfg["n_score"], cfg["seed"] + 5100)       # audition score set (agent-side, charged)
    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)        # held-out grade (instrument, free)
    q_cal = geom(cfg["n_cal"], cfg["seed"] + 5300)        # calibration set

    def app_plan(q, seed):
        """The mastered approach: one full-horizon plan from the actual start posture over the
        FROZEN model. Deterministic given (q, seed), hence identical across arms -- so the
        phrase-launch distribution is a property of the WORLD, not of the arm being graded. The
        approach never ends the piece, so it takes `vel_pen_mid` (built per call, because CAL-C
        sets that knob part-way through this run)."""
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        g = np.tile(W.goals[0][None, :], (len(q), 1))
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(s, g, np.random.default_rng(seed))[0]

    def R_react():
        return W.routing("reactive")

    def R_plan(groups, **kw):
        return W.routing("plan_launch", groups=groups, **kw)

    def groupings(ns):
        """Every contiguous partition of `ns` segments, coarsest first: [3], [2,1], [1,2], [1,1,1].
        The extremes are the two arms of the round; the middles localise WHERE the degradation
        comes from, which a two-point comparison could not."""
        if ns == 0:
            return [[]]
        return [[first] + rest for first in range(ns, 0, -1) for rest in groupings(ns - first)]

    def run(fm, routing, q, plan, seed, sigma, who="instrument", kind="probe", collect=False,
            stop_seg=None, replan_every=1):
        return W.traverse(fm, routing, q, np.random.default_rng(seed), sigma, led, who=who,
                          kind=kind, approach_plan=plan, collect=collect, stop_seg=stop_seg,
                          replan_every=replan_every)

    # ---- the CEILING forward model: the same diet PLUS the piece's own CORRIDOR. Corridor-matched
    # rather than globally accurate, per the etude's `cal_s1` and `bridge_assembly`'s spatial
    # matching law: a "ceiling" collected on a broad uniform volume is not the ceiling for THIS loop
    # and would understate G7's usable range.
    def build_ceiling(seed_off):
        """Collect the piece's own CORRIDOR under the current cost shaping and train the ceiling FM
        on diet + corridor. Corridor-matched rather than globally accurate, per the etude's `cal_s1`
        and `bridge_assembly`'s spatial-matching law: a "ceiling" collected on a broad uniform
        volume is not the ceiling for THIS loop and would understate G7's usable range."""
        cc = []
        for i in range(cfg["n_corridor"]):
            qq = geom(cfg["batch"], cfg["seed"] + 5400 + i)
            cc.append(run(fm0, R_react(), qq, app_plan(qq, cfg["seed"] + 6400 + i),
                          cfg["seed"] + 6500 + i, cfg["sigma_practice"], kind="ceiling_diet",
                          collect=True, replan_every=cfg["corridor_replan"])["trans"])
        net = W.mlp(cfg["seed"] + 42)
        W.train_steps(net, torch.optim.Adam(net.parameters(), lr=cfg["fm_lr"]),
                      np.concatenate([Sa] + [c[0] for c in cc]),
                      np.concatenate([Ua] + [c[1] for c in cc]),
                      np.concatenate([S2a] + [c[2] for c in cc]),
                      cfg["fm_steps"], np.random.default_rng(cfg["seed"] + seed_off))
        return net, cc

    # ---- CAL-C: cost shaping. THREE knobs that would each silently decide the round if guessed.
    #  `w_waypoint` -- THE l1 REPAIR. See `World.waypoint_mask`. l0's grader measured arrival at each
    #      waypoint at its seam step while the controller minimised mean distance over a window that
    #      SPANS the seam, so a window crossing a seam was rewarded for leaving waypoint k early.
    #      Measured cost: reactive per-segment error [0.194, 0.165, 0.010] -- fine only on the last
    #      segment, the one with no next waypoint to be pulled toward -- against 0.032 for a
    #      committed unit whose window ends at the waypoint. That crippled `never`, the REFERENCE
    #      arm, and flattened the FM-quality axis to 3% of the error.
    #  `vel_pen_mid` -- round 1 penalised terminal joint velocity unconditionally, right there
    #      because its drilled segment ENDED the piece. On a loop, braking at every seam plays the
    #      piece staccato.
    #  `commit_lookahead` -- a live segment-wise unit COMMITS to its span but may PLAN further,
    #      issuing only the committed part. l0 measured lookahead=0 better than 20, but that was
    #      measured UNDER THE MISMATCH and is contaminated: looking past a seam could only hurt when
    #      the cost rewarded leaving early. Re-calibrated here on the repaired cost.
    #
    # THE CRITERION, changed from l0 and recorded as a methodology finding: calibrate to PRESERVE
    # THE MEASUREMENT AXIS, not to minimise an arm's raw error. l0 chose `vel_pen_mid` by minimum
    # reactive piece error and got a setting under which only 3% of the error was attributable to
    # forward-model quality -- optimising the number while destroying the axis the whole round rides
    # on. Two stages, lexicographic, each measured where it belongs:
    #
    #   COMPETENCE, on the REACTIVE arm. `never` is the reference opponent and round 1 found it wins
    #     outright; a weak `never` would manufacture a win for commitment. Cells more than
    #     `cal_c_competence`x worse than the best reactive error are struck out.
    #   THE AXIS, on BALLISTIC-per-segment, NOT on reactive. Round 1 / `arm_substrate` P5 measured
    #     that ballistic transmits forward-model quality ~4.9x more than reactive does -- reactive
    #     replans every step, so a better model buys it little almost by construction. l0's 3%
    #     FM-attributable on the reactive arm was therefore partly the transmission result, not only
    #     the artifact (ball_seg was 47% in the same run). Scoring the axis on the mode that
    #     transmits it is the honest test, and it is still arm-neutral: every committed arm in this
    #     node flies open-loop, so none of them owns `ball_seg`.
    #
    # If the winning cell still has no axis (range below `axis_min_over_noise` x the noise floor),
    # that is reported as a G7 failure rather than tuned away.
    # The provisional ceiling is corridor-matched to whatever controller collects it, so seeding it
    # at the grid's LOW end would bias the criterion back toward the very setting being repaired
    # (the ceiling would fit the broken corridor and score the repaired cells as further from its
    # training data). Seed it at the grid MIDPOINTS instead -- it only has to rank the cells, and
    # the final ceiling is re-collected under the chosen setting before G7 reports any range.
    cfg["w_waypoint"] = float(cfg["w_waypoint_grid"][len(cfg["w_waypoint_grid"]) // 2])
    cfg["vel_pen_mid"] = float(cfg["vel_pen_mid_grid"][len(cfg["vel_pen_mid_grid"]) // 2])
    fm_ceil_prov, _ = build_ceiling(302)      # provisional: needed to score the fraction at all
    calc = {"cost": {}, "lookahead": {}}
    vp0, w0 = cfg.get("vel_pen_mid", 0.0), cfg.get("w_waypoint", 0.0)
    for wv in cfg["w_waypoint_grid"]:
        for v in cfg["vel_pen_mid_grid"]:
            cfg["w_waypoint"], cfg["vel_pen_mid"] = float(wv), float(v)
            pc = app_plan(q_cal, cfg["seed"] + 6300)
            cell = {}
            for mn, rt in (("react", R_react()),
                           ("seg", R_plan([1] * NS, k_shoot=cfg["k_shoot"],
                                          cem_iters=cfg["cem_iters"])),
                           ("phrase", R_plan([NS], k_shoot=cfg["k_shoot"],
                                             cem_iters=cfg["cem_iters"]))):
                o = run(fm0, rt, q_cal, pc, cfg["seed"] + 6800, cfg["sigma_perf"], kind="cal_c")
                cell[mn] = float(np.median(o["e_piece"]))
                if mn == "react":
                    cell["react_by_seg"] = [float(np.median(o["e_seg"][:, k])) for k in range(NS)]
            for mn, rt in (("react_ceiling", R_react()),
                           ("seg_ceiling", R_plan([1] * NS, k_shoot=cfg["k_shoot"],
                                                  cem_iters=cfg["cem_iters"]))):
                oc = run(fm_ceil_prov, rt, q_cal, pc, cfg["seed"] + 6800, cfg["sigma_perf"],
                         kind="cal_c")
                cell[mn] = float(np.median(oc["e_piece"]))
            cell["react_attributable"] = ((cell["react"] - cell["react_ceiling"])
                                          / max(cell["react"], 1e-9))
            cell["usable_range"] = cell["seg"] - cell["seg_ceiling"]
            cell["fm_attributable"] = cell["usable_range"] / max(cell["seg"], 1e-9)
            calc["cost"][f"w{wv}_v{v}"] = cell
            P(f"[CAL-C] w_waypoint={wv:<5} vel_pen_mid={v:<5} -> react {cell['react']:.4f} "
              f"(by-seg {[round(x, 3) for x in cell['react_by_seg']]}) | ball_seg "
              f"{cell['seg']:.4f} vs ceil {cell['seg_ceiling']:.4f} -> FM-attributable "
              f"{100 * cell['fm_attributable']:.0f}% (reactive {100 * cell['react_attributable']:.0f}%)"
              f" | phrase {cell['phrase']:.4f}")
    best_react = min(c["react"] for c in calc["cost"].values())
    ok = {k: c for k, c in calc["cost"].items()
          if c["react"] <= cfg["cal_c_competence"] * best_react}
    best_k = max(ok, key=lambda k: ok[k]["fm_attributable"])
    best_w = float(best_k.split("_")[0][1:]); best_v = float(best_k.split("_v")[1])
    cfg["w_waypoint"], cfg["vel_pen_mid"] = best_w, best_v
    P(f"[CAL-C] competence guard keeps {len(ok)}/{len(calc['cost'])} cells "
      f"(react <= {cfg['cal_c_competence']}x best {best_react:.4f}); "
      f"max FM-attributable among them -> {best_k}")
    plan_cal = app_plan(q_cal, cfg["seed"] + 6300)

    # ---- CAL-C stage 2 (l2): THE SEAM CELL -- terminal velocity convention x look-ahead weight.
    #
    # l0 and l1 are two sides of ONE failure mode. l0's cost let a planner ignore waypoints, which
    # crippled `never`. l1's cost made it hit waypoints while ignoring HANDOFF VELOCITY, so a
    # segment plan could nail its own waypoint and arrive with a velocity that doomed the next
    # segment -- which crippled `seg_plan_launch` (its segment 2 went to 0.301 while the fused
    # phrase plan got 0.180 there). Each time a shared cost knob is set it can silently handicap one
    # of the contrasted arms, so this stage sets both remaining knobs against an explicitly NEUTRAL
    # criterion.
    #
    #  `vel_pen` (terminal) -- whether a terminal state exists at all is a property of HOW METERING
    #      TRAVERSES THE PIECE, not of the geometry. This node's `traverse` runs ONE LAP (approach +
    #      3 segments, ending at W0), so a single-lap run-through does have a genuine end; the grade
    #      `e_piece` is the mean of the three waypoint arrival errors and never references velocity.
    #      So `vel_pen` encodes a CHOICE: treat the lap as a complete performance that stops at the
    #      end, or as one lap of a continuing loop that carries momentum through the closure. Round
    #      1 inherited "stops at the end" because its piece genuinely ended. Both are measured here
    #      and the chosen convention is recorded with the grade.
    #  `gamma` -- how heavily a live unit weights the segment it hands over to, relative to its own.
    #
    # CRITERION, fixed in advance and deliberately NOT the contrast under study: the MEAN piece
    # error across (reactive, segment arm, phrase arm), subject to the FM axis surviving. Scoring
    # the cost choice on the [1+1+1]-vs-[3] gap would be circular -- it would tune the cost until
    # the hypothesis won. Every cell's full G5b table is recorded so the cost-dependence of the
    # ordering is visible rather than hidden behind the winner.
    calc["seam"] = {}
    for vt in cfg["vel_pen_grid"]:
        for gm in cfg["gamma_grid"]:
            cfg["vel_pen"], cfg["lookahead_gamma"] = float(vt), float(gm)
            pc = app_plan(q_cal, cfg["seed"] + 6300)
            cell = {}
            for mn, rt in (("react", R_react()),
                           ("seg", R_plan([1] * NS, k_shoot=cfg["k_shoot"],
                                          cem_iters=cfg["cem_iters"])),
                           ("phrase", R_plan([NS], k_shoot=cfg["k_shoot"],
                                             cem_iters=cfg["cem_iters"]))):
                o = run(fm0, rt, q_cal, pc, cfg["seed"] + 6820, cfg["sigma_perf"], kind="cal_c")
                cell[mn] = float(np.median(o["e_piece"]))
                cell[f"{mn}_by_seg"] = [float(np.median(o["e_seg"][:, k])) for k in range(NS)]
            oc = run(fm_ceil_prov, R_plan([1] * NS, k_shoot=cfg["k_shoot"],
                                          cem_iters=cfg["cem_iters"]),
                     q_cal, pc, cfg["seed"] + 6820, cfg["sigma_perf"], kind="cal_c")
            cell["seg_ceiling"] = float(np.median(oc["e_piece"]))
            cell["fm_attributable"] = ((cell["seg"] - cell["seg_ceiling"])
                                       / max(cell["seg"], 1e-9))
            cell["neutral"] = float(np.mean([cell["react"], cell["seg"], cell["phrase"]]))
            # the full G5b table AT THIS CELL, on a fixed mid-size planner (CAL-P has not run yet,
            # so this is a relative comparison across groupings, not the final calibrated one)
            cell["g5b"] = {}
            for grp in groupings(NS):
                key = "+".join(str(x) for x in grp)
                cell["g5b"][key] = {}
                for nm, net in (("stale", fm0), ("ceiling", fm_ceil_prov)):
                    og = run(net, W.routing("plan_launch", groups=grp, k_shoot=cfg["k_shoot"],
                                            cem_iters=cfg["cem_iters"]),
                             q_cal, pc, cfg["seed"] + 6830, cfg["sigma_perf"], kind="cal_c")
                    cell["g5b"][key][nm] = dict(
                        piece=float(np.median(og["e_piece"])),
                        by_seg=[float(np.median(og["e_seg"][:, k])) for k in range(NS)])
            calc["seam"][f"vt{vt}_g{gm}"] = cell
            sp = {k: (v["stale"]["piece"], v["ceiling"]["piece"]) for k, v in cell["g5b"].items()}
            P(f"[CAL-C] vel_pen={vt:<4} gamma={gm:<4} -> react {cell['react']:.4f} seg "
              f"{cell['seg']:.4f} (by-seg {[round(x, 3) for x in cell['seg_by_seg']]}) phrase "
              f"{cell['phrase']:.4f} | NEUTRAL {cell['neutral']:.4f} | FM-attrib "
              f"{100 * cell['fm_attributable']:.0f}%")
            P(f"          G5b stale/ceiling: " + "  ".join(
                f"{k}={a:.3f}/{b:.3f}" for k, (a, b) in sp.items()))
    # eligibility = the FM-quality axis still exists at that cell (positive attributable fraction on
    # ballistic-per-segment). If NO cell keeps the axis, that is itself a G7-level finding, so fall
    # back to the whole grid rather than silently picking from an empty set -- and say so.
    elig = {k: c for k, c in calc["seam"].items() if c["fm_attributable"] > 0.0}
    if not elig:
        P("[WARN] no seam cell keeps a positive FM-attributable fraction -- the axis is absent "
          "across the whole cost grid; falling back to the full grid for the neutral choice")
        elig = dict(calc["seam"])
    best_s = min(elig, key=lambda k: elig[k]["neutral"])
    best_vt = float(best_s.split("_g")[0][2:]); best_gm = float(best_s.split("_g")[1])
    cfg["vel_pen"], cfg["lookahead_gamma"] = best_vt, best_gm
    P(f"[CAL-C] chosen by NEUTRAL mean piece error: vel_pen={best_vt} gamma={best_gm} "
      f"(neutral {calc['seam'][best_s]['neutral']:.4f}, FM-attributable "
      f"{100 * calc['seam'][best_s]['fm_attributable']:.0f}%)")
    calc["chosen"] = dict(w_waypoint=best_w, vel_pen_mid=best_v, vel_pen_terminal=best_vt,
                          lookahead_gamma=best_gm,
                          metering_convention="ONE LAP per run-through (approach + 3 segments, "
                                              "ending at W0); grade = mean of the three waypoint "
                                              "arrival errors, which never references velocity",
                          prior_defaults=[w0, vp0],
                          competence_guard=cfg["cal_c_competence"],
                          axis_eligible_cells=sorted(elig),
                          criterion="STAGE 1 (w_waypoint, vel_pen_mid), lexicographic: competence "
                                    "guard on the REACTIVE arm, then max FM-attributable fraction "
                                    "on BALLISTIC-per-segment (the mode that transmits model "
                                    "quality, P5 ~4.9x reactive). STAGE 2 (vel_pen terminal, "
                                    "lookahead gamma): min MEAN piece error across reactive/seg/"
                                    "phrase -- deliberately NEUTRAL across granularities, among "
                                    "cells whose FM axis survives. Scoring stage 2 on the "
                                    "[1+1+1]-vs-[3] gap would be circular, so it is not used; every "
                                    "cell's full G5b table is recorded instead.")
    out["cal_cost"] = calc
    P(f"[CAL-C] chosen w_waypoint={best_w} vel_pen_mid={best_v} vel_pen_terminal={best_vt} lookahead_gamma={best_gm}")
    plan_rt = app_plan(q_rt, cfg["seed"] + 6000)
    plan_sc = app_plan(q_sc, cfg["seed"] + 6100)
    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    plan_cal = app_plan(q_cal, cfg["seed"] + 6300)
    save()

    # the FINAL ceiling, re-collected under the chosen cost shaping (the provisional one was
    # corridor-matched to a different controller, which is fine for ranking cells and not fine for
    # the usable range G7 reports)
    fm_ceil, corr = build_ceiling(303)

    def fm_err(net, S, U, S2):
        return float(np.linalg.norm(W.fm_delta(net, S, U) - (S2 - S), axis=1).mean())

    Sc_ = np.concatenate([c[0] for c in corr]); Uc_ = np.concatenate([c[1] for c in corr])
    S2c_ = np.concatenate([c[2] for c in corr])
    reg = W.gate(W.tip(Sc_)) > 0.3
    out["diet"] = dict(n_ou=len(S1), n_reach=len(S2_), n_corridor=len(Sc_),
                       exclude_keep_frac=keep_frac, n_excluded_pool=len(Se),
                       corridor_in_region_frac=float(reg.mean()), diagnostics=diag,
                       corr_reach=cmd_state_corr(S2_, U2_))
    out["fm_probe"] = dict(
        stale_corridor_region=fm_err(fm0, Sc_[reg], Uc_[reg], S2c_[reg]) if reg.any() else None,
        stale_corridor_clean=fm_err(fm0, Sc_[~reg], Uc_[~reg], S2c_[~reg]),
        ceil_corridor_region=fm_err(fm_ceil, Sc_[reg], Uc_[reg], S2c_[reg]) if reg.any() else None,
        ceil_corridor_clean=fm_err(fm_ceil, Sc_[~reg], Uc_[~reg], S2c_[~reg]),
        n_corridor_in_region=int(reg.sum()))
    fp = out["fm_probe"]
    P(f"[fm] corridor slice  stale reg/clean {fp['stale_corridor_region']}/"
      f"{fp['stale_corridor_clean']:.4f}  ceiling {fp['ceil_corridor_region']}/"
      f"{fp['ceil_corridor_clean']:.4f}  (n_in_region {int(reg.sum())}/{len(Sc_)})")
    save()

    # ================================================================= G5a composition divergence
    # THE MECHANISM, with no planner anywhere in it. Roll the forward model open-loop along command
    # sequences of the full phrase span and score its predicted tip against the plant's actual tip,
    # step by step. Two command sources, because the answer must not depend on which:
    #   `executed` -- what the body actually did under reactive control (on-policy, the honest one);
    #   `planned`  -- a single ballistic phrase plan from the ceiling model (round 1's G4 protocol,
    #                 extended from 20 steps to 60).
    # The `executed` probe replays the ISSUED commands (motor noise included) from the trajectory's
    # OWN launch state, so `true_tips` reproduces the traversal exactly and the FM is scored against
    # what the plant actually did -- not against a different trajectory that happens to share a
    # start distribution.
    ex = run(fm0, R_react(), q_ev, plan_ev, cfg["seed"] + 7001, cfg["sigma_perf"], kind="g5_exec")
    ho, cmd_exec = ex["launch"], ex["acts"]
    kbig = cfg["planner_grid"][-1]
    cmd_plan, _ = W.plan_fn(fm_ceil, W.H_phrase, k_shoot=kbig[0], cem_iters=kbig[1],
                            cem_elite=elite_for(kbig[0]))(
        ho, np.tile(W.goal_schedule(0, NS)[None], (len(ho), 1, 1)),
        np.random.default_rng(cfg["seed"] + 7002))

    def divergence(net, S0, cmds):
        true = W.true_tips(S0, cmds)
        pred = W.fm_rollout_tips(net, S0, cmds)
        return np.median(np.linalg.norm(pred - true, axis=2), 0)

    def cross(curve, thr):
        idx = np.nonzero(np.asarray(curve) > thr)[0]
        return int(idx[0] + 1) if len(idx) else int(len(curve) + 1)

    g5a = {}
    for src, cm, S0 in (("executed", cmd_exec, ho), ("planned", cmd_plan, ho)):
        row = {}
        for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
            c = divergence(net, S0, cm)
            row[nm] = dict(curve=c.tolist(), h_005=cross(c, 0.05), h_002=cross(c, 0.02),
                           err_at=[float(c[int(h) - 1]) for h in W.seg_hi])
        g5a[src] = row
        P(f"[G5a/{src:8s}] composition horizon (>0.05 m): stale {row['stale']['h_005']} "
          f"ceiling {row['ceiling']['h_005']} | err at seg ends stale "
          f"{[round(v,4) for v in row['stale']['err_at']]} ceiling "
          f"{[round(v,4) for v in row['ceiling']['err_at']]}")
    out["g5a"] = g5a
    save()

    # ================================================================= CAL-P planner sizing per span
    # `arm_substrate` P3 is the single biggest confound on this plant, and it bites HARDER here than
    # in round 1: a segment plan optimises 60 action dimensions, a phrase plan 180. Each span is
    # therefore given its own grid, and the verdict that matters is whether error is STILL FALLING
    # at the top of the grid -- if it is, any span degradation G5b reports could be planner
    # starvation rather than model composition, and that has to be said out loud.
    cal = {}
    for ns_ in range(1, NS + 1):
        row = {}
        for spec in cfg["planner_grid"]:
            ks, ci = int(spec[0]), int(spec[1])
            cell = {}
            for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
                o = run(net, R_plan([ns_] + [1] * (NS - ns_), k_shoot=ks, cem_iters=ci,
                                    cem_elite=elite_for(ks)),
                        q_cal, plan_cal, cfg["seed"] + 7100, cfg["sigma_perf"], kind="cal_p",
                        stop_seg=ns_)
                cell[nm] = dict(
                    e_end=float(np.median(o["e_seg"][:, ns_ - 1])),
                    e_span=float(np.nanmedian(np.nanmean(o["e_seg"][:, :ns_], axis=1))),
                    e_by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(ns_)],
                    delib=int(o["delib"]))
            cell["gap"] = cell["stale"]["e_end"] - cell["ceiling"]["e_end"]
            row[f"k{ks}_i{ci}"] = cell
            P(f"[CAL-P] span={ns_} seg ({ns_ * W.H_seg[0]} steps, {ns_ * W.H_seg[0] * AD} dims) "
              f"k={ks:5d} i={ci:2d} -> end-of-span median  stale {cell['stale']['e_end']:.4f}  "
              f"ceiling {cell['ceiling']['e_end']:.4f}  gap {cell['gap']:+.4f}")
        # starvation check: is the top of the grid still buying accuracy?
        keys = [f"k{int(s[0])}_i{int(s[1])}" for s in cfg["planner_grid"]]
        for nm in ("stale", "ceiling"):
            v = [row[k][nm]["e_end"] for k in keys]
            row[f"still_improving_{nm}"] = bool(len(v) > 1 and v[-1] < v[-2] - cfg["cal_p_tol"])
            row[f"last_step_gain_{nm}"] = float(v[-2] - v[-1]) if len(v) > 1 else None
        cal[f"span{ns_}"] = row
    out["cal_planner"] = cal
    for ns_ in range(1, NS + 1):
        r = cal[f"span{ns_}"]
        P(f"[CAL-P] span={ns_}: still improving at top of grid?  stale "
          f"{r['still_improving_stale']} (last gain {r['last_step_gain_stale']})  ceiling "
          f"{r['still_improving_ceiling']} (last gain {r['last_step_gain_ceiling']})")
    save()

    # pick each span's calibrated planner: the smallest grid point within `cal_p_tol` of the best.
    def best_planner(ns_, nm="stale"):
        r = cal[f"span{ns_}"]
        keys = [f"k{int(s[0])}_i{int(s[1])}" for s in cfg["planner_grid"]]
        v = [r[k][nm]["e_end"] for k in keys]
        bi = int(np.argmin(v))
        for i, val in enumerate(v):
            if val <= v[bi] + cfg["cal_p_tol"]:
                bi = i
                break
        s = cfg["planner_grid"][bi]
        return dict(k_shoot=int(s[0]), cem_iters=int(s[1]), cem_elite=elite_for(int(s[0])))

    CALP = {ns_: best_planner(ns_) for ns_ in range(1, NS + 1)}
    out["cal_planner_chosen"] = {f"span{k}": v for k, v in CALP.items()}
    P(f"[CAL-P] chosen: {out['cal_planner_chosen']}")

    # ================================================================= G5b span vs re-groundings
    # The matched comparison: the SAME three segments, from the SAME phrase-launch states, ending at
    # the SAME waypoint -- only the number of committed groups differs. Each group is planned live
    # at its own launch under the model, with that span's CAL-P planner, so a phrase group is not
    # handicapped by being given a segment-sized search.
    def groupings(ns):
        """Every contiguous partition of `ns` segments, coarsest first: [3], [2,1], [1,2], [1,1,1].
        The extremes are the two arms of the round; the middles localise WHERE the degradation
        comes from, which a two-point comparison could not."""
        if ns == 0:
            return [[]]
        return [[first] + rest for first in range(ns, 0, -1) for rest in groupings(ns - first)]

    g5b = {}
    for grp in groupings(NS):
        units = [W.plan_launch_unit(**CALP[g]) for g in grp]
        row = {}
        for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
            o = run(net, W.routing(None, groups=grp, units=units), q_ev, plan_ev,
                    cfg["seed"] + 7200, cfg["sigma_perf"], kind="g5b")
            row[nm] = dict(e_by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)],
                           e_piece=float(np.median(o["e_piece"])),
                           e_final=float(np.median(o["e_seg"][:, -1])),
                           n_fb=int(o["n_fb"]), n_plan=int(o["n_plan"]), delib=int(o["delib"]))
        g5b["+".join(str(x) for x in grp)] = row
        P(f"[G5b] groups {str(grp):10s} (fb {row['stale']['n_fb']}) -> stale piece "
          f"{row['stale']['e_piece']:.4f} final {row['stale']['e_final']:.4f} by-seg "
          f"{[round(v,4) for v in row['stale']['e_by_seg']]} | ceiling piece "
          f"{row['ceiling']['e_piece']:.4f} by-seg "
          f"{[round(v,4) for v in row['ceiling']['e_by_seg']]}")
    # the reactive reference under the same protocol
    for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
        o = run(net, R_react(), q_ev, plan_ev, cfg["seed"] + 7250, cfg["sigma_perf"], kind="g5b")
        g5b.setdefault("reactive", {})[nm] = dict(
            e_by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)],
            e_piece=float(np.median(o["e_piece"])), e_final=float(np.median(o["e_seg"][:, -1])),
            n_fb=int(o["n_fb"]), n_plan=int(o["n_plan"]), delib=int(o["delib"]))
    out["g5b"] = g5b
    P(f"[G5b] reactive -> stale piece {g5b['reactive']['stale']['e_piece']:.4f} | ceiling "
      f"{g5b['reactive']['ceiling']['e_piece']:.4f}")
    save()

    # ================================================================= G7 usable range + noise floor
    def meter(fm, routing, seed, sigma=None, who="instrument", replan_every=1):
        return run(fm, routing, q_rt, plan_rt, seed, cfg["sigma_perf"] if sigma is None else sigma,
                   who=who, kind="metering", replan_every=replan_every)

    modes = {"react": R_react(),
             "ball_seg": R_plan([1] * NS, **CALP[1]),
             "ball_phrase": R_plan([NS], **CALP[NS])}
    g7 = {"modes": {}}
    for mn, rt in modes.items():
        cell = {}
        for nm, net in (("stale", fm0), ("ceiling", fm_ceil)):
            reps = [meter(net, rt, cfg["seed"] + 7300 + 17 * r) for r in range(cfg["n_meter_rep"])]
            piece = np.array([np.median(o["e_piece"]) for o in reps])
            byseg = np.array([[np.median(o["e_seg"][:, k]) for k in range(NS)] for o in reps])
            cell[nm] = dict(piece=float(piece.mean()), piece_sd=float(piece.std()),
                            by_seg=byseg.mean(0).tolist(), by_seg_sd=byseg.std(0).tolist(),
                            n_fb=int(reps[0]["n_fb"]))
        cell["usable_range"] = cell["stale"]["piece"] - cell["ceiling"]["piece"]
        cell["noise_sd"] = float(np.mean([cell["stale"]["piece_sd"], cell["ceiling"]["piece_sd"]]))
        cell["range_over_noise"] = float(cell["usable_range"] / max(cell["noise_sd"], 1e-9))
        cell["usable_range_by_seg"] = [a - b for a, b in zip(cell["stale"]["by_seg"],
                                                             cell["ceiling"]["by_seg"])]
        g7["modes"][mn] = cell
        P(f"[G7] {mn:12s} piece: stale {cell['stale']['piece']:.4f} ceiling "
          f"{cell['ceiling']['piece']:.4f} | range {cell['usable_range']:.4f} noise sd "
          f"{cell['noise_sd']:.4f} -> {cell['range_over_noise']:.1f}x | by-seg range "
          f"{[round(v,4) for v in cell['usable_range_by_seg']]}")
    # THE AXIS CHECK. If the usable range at ballistic-per-segment is not comfortably above the
    # metering noise floor, the FM-quality axis is absent and G7 FAILS -- which is a finding about
    # the world, not a knob to retune (typed_gaps: a gate that fails is a finding).
    g7["axis_present"] = bool(g7["modes"]["ball_seg"]["range_over_noise"]
                              >= cfg["axis_min_over_noise"])
    g7["axis_fm_attributable"] = float(g7["modes"]["ball_seg"]["usable_range"]
                                       / max(g7["modes"]["ball_seg"]["stale"]["piece"], 1e-9))
    if not g7["axis_present"]:
        P(f"[WARN] G7 FAILS: ball_seg usable range {g7['modes']['ball_seg']['usable_range']:.4f} is "
          f"only {g7['modes']['ball_seg']['range_over_noise']:.1f}x the noise floor "
          f"(< {cfg['axis_min_over_noise']}) -- the FM-quality axis is ABSENT on this world")
    g7["ballistic_over_reactive"] = float(g7["modes"]["ball_seg"]["stale"]["piece"]
                                          / max(g7["modes"]["react"]["stale"]["piece"], 1e-9))
    g7["phrase_over_seg"] = float(g7["modes"]["ball_phrase"]["stale"]["piece"]
                                  / max(g7["modes"]["ball_seg"]["stale"]["piece"], 1e-9))
    out["g7"] = g7
    save()

    # ================================================================= CAL-D warmup -> mastery clocks
    # Exactly the main run's practice loop, reactive throughout. Per cycle we meter the BALLISTIC
    # configurations (cheap: 3 plans for `ball_seg`, 1 for `ball_phrase`) because round 1b's rule
    # reads the mastery clock off the open-loop curve; the reactive curve is metered every
    # `react_every` cycles for reference, since reading IT would have said "mastered" 39 cycles too
    # early in round 1.
    fm = copy.deepcopy(fm0)
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    buf, trace_buf = [], []
    brng = np.random.default_rng(cfg["seed"] + 800)
    desc = {k: [] for k in ("cycle", "ball_seg_piece", "ball_phrase_piece", "react_piece",
                            "practice_piece", "t_cum", "fm_corridor")}
    desc_seg = {"ball_seg": [], "ball_phrase": []}
    for c in range(1, cfg["n_warm"] + 1):
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + c)
        pr = W.traverse(fm, R_react(), qp, np.random.default_rng(cfg["seed"] + 10_000 + c),
                        cfg["sigma_practice"], led, who="agent", kind="practice",
                        approach_plan=app_plan(qp, cfg["seed"] + 9500 + c), collect=True)
        buf.append(pr["trans"])
        trace_buf.append((pr["launch"].copy(), {k: v.copy() for k, v in pr["launches"].items()},
                          pr["acts"].copy(), pr["e_seg"].copy()))
        for b_ in (buf, trace_buf):
            if len(b_) > cfg["trace_window"]:
                b_.pop(0)
        PX, PY = W.tensors(np.concatenate([b[0] for b in buf]),
                           np.concatenate([b[1] for b in buf]),
                           np.concatenate([b[2] for b in buf]))
        W.train_online(fm, opt, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])
        bs_ = meter(fm, modes["ball_seg"], cfg["seed"] + 4000, who="agent")
        bp_ = meter(fm, modes["ball_phrase"], cfg["seed"] + 4100, who="agent")
        desc["cycle"].append(c); desc["t_cum"].append(led.t_priced)
        desc["ball_seg_piece"].append(float(np.median(bs_["e_piece"])))
        desc["ball_phrase_piece"].append(float(np.median(bp_["e_piece"])))
        desc["practice_piece"].append(float(np.median(pr["e_piece"])))
        desc_seg["ball_seg"].append([float(np.median(bs_["e_seg"][:, k])) for k in range(NS)])
        desc_seg["ball_phrase"].append([float(np.median(bp_["e_seg"][:, k])) for k in range(NS)])
        if c % cfg["react_every"] == 0 or c == cfg["n_warm"]:
            rr = meter(fm, modes["react"], cfg["seed"] + 4200, who="agent")
            desc["react_piece"].append([c, float(np.median(rr["e_piece"]))])
            # THE MODEL ITSELF, on the corridor slice. Oracle instrument, free. l0's ballistic
            # readout DEGRADED 2.2x over 36 cycles of practice while reactive stayed flat, and a
            # task-error curve alone cannot say whether the model got worse or the controller
            # merely failed to exploit it. This separates them: if `fm_corridor` improves while
            # `ball_seg` does not, the problem is the controller/grader, not the diet; if the model
            # itself degrades, the on-policy diet is narrowing and that is a design question about
            # the pretrain configuration, not a knob.
            try:
                desc["fm_corridor"].append(
                    [c, fm_err(fm, Sc_[~reg], Uc_[~reg], S2c_[~reg]),
                     (fm_err(fm, Sc_[reg], Uc_[reg], S2c_[reg]) if reg.any() else None)])
            except Exception as e:                      # a DIAGNOSTIC must never kill the run
                P(f"[WARN] fm_corridor probe failed at c{c}: {e!r}")
        P(f"[warm c{c:2d}] ball_seg {desc['ball_seg_piece'][-1]:.4f} ball_phrase "
          f"{desc['ball_phrase_piece'][-1]:.4f} practice {desc['practice_piece'][-1]:.4f} "
          f"by-seg(seg) {[round(v,4) for v in desc_seg['ball_seg'][-1]]} t={led.t_priced:9.1f}s")

    def mastery(series, w=3, tail=5):
        """Round 1b's rule, verbatim: trailing-`w` mean of the held-out BALLISTIC curve; plateau =
        mean +- sd over the last `tail` probes; the clock is the first cycle from which the smoothed
        curve stays within 1 sd of the plateau for the rest of the run."""
        v = np.asarray(series, float)
        if len(v) < w + tail:
            return None, None, None, None, None
        sm = np.array([v[max(0, i - w + 1):i + 1].mean() for i in range(len(v))])
        mu, sd = sm[-tail:].mean(), sm[-tail:].std()
        thr = mu + sd
        clock = None
        for i in range(len(sm)):
            if (sm[i:] <= thr).all():
                clock = i + 1
                break
        # HONESTY FLAG. The rule takes the last `tail` probes as "the plateau", which is only
        # meaningful if the curve has actually stopped descending. If the last window still sits
        # materially below the one before it, the run has not plateaued and the clock is a LOWER
        # BOUND -- reporting it as a mastery cycle would repeat round 1's wrong-clock error in a new
        # way. Round 1 read its clock off a 60-cycle main run; this gate has fewer cycles to work
        # with, so the check is explicit rather than assumed.
        prev = sm[-2 * tail:-tail].mean() if len(sm) >= 2 * tail else float("nan")
        plateaued = bool(np.isfinite(prev) and (prev - mu) <= sd)
        return clock, float(mu), float(sd), plateaued, float(prev)

    clocks = {}
    for nm, series in (("ball_seg_piece", desc["ball_seg_piece"]),
                       ("ball_phrase_piece", desc["ball_phrase_piece"])):
        cl, mu, sd, pl, pv = mastery(series)
        clocks[nm] = dict(clock=cl, plateau=mu, plateau_sd=sd, plateaued=pl, prev_window=pv)
    for k in range(NS):
        cl, mu, sd, pl, pv = mastery([r[k] for r in desc_seg["ball_seg"]])
        clocks[f"seg{k}"] = dict(clock=cl, plateau=mu, plateau_sd=sd, plateaued=pl, prev_window=pv)
    rr_series = [v for _, v in desc["react_piece"]]
    cl, mu, sd, pl, pv = mastery(rr_series, w=2, tail=3)
    clocks["react_piece"] = dict(clock=(None if cl is None else desc["react_piece"][cl - 1][0]),
                                 plateau=mu, plateau_sd=sd, plateaued=pl, prev_window=pv)
    for k, v in clocks.items():
        if v["clock"] is not None and not v["plateaued"]:
            P(f"[WARN] {k}: curve had NOT plateaued by the end of the warmup (prev window "
              f"{v['prev_window']:.4f} vs final {v['plateau']:.4f}, sd {v['plateau_sd']:.4f}) -- "
              f"clock c{v['clock']} is a LOWER BOUND, not a mastery cycle")
    rng_range = max(g7["modes"]["ball_seg"]["usable_range"], 1e-9)
    out["cal_descent"] = dict(
        series=desc, by_seg=desc_seg, clocks=clocks, n_grad=cfg["n_grad"],
        adapt_lr=cfg["adapt_lr"],
        frac_of_range_closed=[(g7["modes"]["ball_seg"]["stale"]["piece"] - d) / rng_range
                              for d in desc["ball_seg_piece"]])
    P("[CAL-D] mastery clocks: " + "  ".join(
        f"{k}=c{v['clock']} (plateau {v['plateau']:.4f}+-{v['plateau_sd']:.4f})"
        for k, v in clocks.items() if v["clock"] is not None))
    save()

    # ================================================================= the auditions (seg + phrase)
    # Score sets are seam-matched by construction: the launch states for segment k come from a
    # traversal at PERFORMANCE tempo, truncated at segment k (the etude E-4 pattern). At this point
    # nothing is committed, so upstream is reactive -- which is exactly the configuration the main
    # run's first commit will face.
    sc_full = run(fm, R_react(), q_sc, plan_sc, cfg["seed"] + 4400, cfg["sigma_perf"],
                  who="agent", kind="score_set")
    S0 = {k: sc_full["launches"][k] for k in range(NS)}
    S0_phrase = sc_full["launch"]

    pool_acts = np.concatenate([t[2] for t in trace_buf])
    pool_eseg = np.concatenate([t[3] for t in trace_buf])
    crng = np.random.default_rng(cfg["seed"] + 953)
    ci = crng.permutation(len(pool_acts))[:cfg["n_cand"]]           # UNIFORM, never top-of-pool
    P(f"[pool] {len(pool_acts)} whole-phrase renditions over {len(trace_buf)} cycles; "
      f"piece error median {float(np.median(np.nanmean(pool_eseg, axis=1))):.4f}")

    def audit(seg_lo, n_segs, S0_, cands, tag):
        cps = W.checkpoints(seg_lo, n_segs)
        E, Efull = W.audition(cands, S0_, cps, led, who="agent")
        best_fixed = float(E.mean(1).min())
        oracle = float(E.min(0).mean())
        jf = int(np.argmin(E.mean(1)))
        # crystallize's per-key optimism control: build on even columns, grade on odd ones.
        ev, hoc = np.arange(0, E.shape[1], 2), np.arange(1, E.shape[1], 2)
        j_in = int(np.argmin(E[:, ev].mean(1)))
        Sf = S0_[ev]
        mu, sd = Sf.mean(0), Sf.std(0) + 1e-6
        C, lab = kmeans((Sf - mu) / sd, cfg["n_lib"], np.random.default_rng(cfg["seed"] + 958))
        dflt = int(np.argmin(E[:, ev].mean(1)))
        pick = {c_: (int(np.argmin(E[:, ev][:, lab == c_].mean(1))) if (lab == c_).any() else dflt)
                for c_ in range(len(C))}
        Ze = (S0_[hoc] - mu) / sd
        ke = ((Ze[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        lib_ho = float(np.mean([E[pick.get(int(ke[i]), dflt), hoc[i]] for i in range(len(hoc))]))
        fix_ho = float(E[j_in, hoc].mean())
        u_lib, li = W.select_library(E, cands, S0_, cfg["n_lib"],
                                     np.random.default_rng(cfg["seed"] + 954))
        info = dict(seg_lo=seg_lo, n_segs=n_segs, n_cand=int(len(cands)), n_score=int(len(S0_)),
                    span=int(W.span(seg_lo, n_segs)),
                    score_med=float(np.median(E.mean(1))), best_fixed=best_fixed,
                    per_state_oracle=oracle, oracle_gain=float(best_fixed / max(oracle, 1e-9)),
                    library=li, library_score=float(np.mean(
                        [E[li["picks"][int(l_)], i] for i, l_ in enumerate(li["labels"])])),
                    lib_heldout=lib_ho, fixed_heldout=fix_ho,
                    lib_over_fixed_heldout=float(fix_ho / max(lib_ho, 1e-9)),
                    in_sample_fixed=float(E[j_in, ev].mean()),
                    pool_geometry=World.pool_geometry(cands))
        P(f"[audition/{tag}] span={info['span']:3d} best fixed {best_fixed:.4f} per-state oracle "
          f"{oracle:.4f} gain {info['oracle_gain']:.2f}x | held-out fixed {fix_ho:.4f} library "
          f"{lib_ho:.4f} -> {info['lib_over_fixed_heldout']:.2f}x | cells {li['cell_sizes']} "
          f"distinct {li['n_distinct']}")
        return E, Efull, info, jf, u_lib

    aud = {}
    for k in range(NS):
        lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k])
        _, _, info, _, _ = audit(k, 1, S0[k], pool_acts[ci][:, lo:hi], f"seg{k}")
        aud[f"seg{k}"] = info
    E_ph, Ef_ph, info_ph, jf_ph, lib_ph = audit(0, NS, S0_phrase, pool_acts[ci], "phrase")
    aud["phrase"] = info_ph
    out["audition"] = aud
    save()

    # ================================================================= G6 seam information
    # (a) the phrase-launch distribution, against the metering noise floor (G1's part (a));
    # (b) what a chain's realised error depends on -- and, the new question, HOW MUCH the phrase
    #     launch state can predict about a segment that happens 20 or 40 steps later, compared with
    #     what that segment's OWN seam state predicts. The difference is the price of flying blind,
    #     denominated in predictable variance.
    def r2_of(X, y):
        Z = np.concatenate([(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones((len(X), 1))], 1)
        b, *_ = np.linalg.lstsq(Z, y, rcond=None)
        return float(1.0 - (y - Z @ b).var() / max(y.var(), 1e-12))

    noise = g7["modes"]["ball_seg"]["noise_sd"]
    tip_ph = W.tip(S0_phrase)
    nd = np.stack([null_direction(S0_phrase[i, :n].astype(np.float64), W.Ls)
                   for i in range(len(S0_phrase))])
    nullc = np.einsum("ij,ij->i", S0_phrase[:, :n] - S0_phrase[:, :n].mean(0)[None, :], nd)
    # Replay the best fixed chain from the held-out phrase-launch states, recording the SEAM STATES.
    # WITH MOTOR NOISE: a noiseless replay makes every downstream state a deterministic function of
    # the launch state, so the information measure would read ~1.0 by construction and the fusion
    # question would be unanswerable. Execution variance is exactly what bounds a chain of measured
    # renditions, so it belongs in the measurement.
    chain = pool_acts[ci][jf_ph]
    cps_ph = W.checkpoints(0, NS)
    e_chain, st_chain = W.replay(S0_phrase, chain, cps_ph, return_states=True,
                                 sigma=cfg["sigma_perf"],
                                 rng=np.random.default_rng(cfg["seed"] + 7600))
    e_chain_quiet = W.replay(S0_phrase, chain, cps_ph)
    led.charge("instrument", 2 * len(S0_phrase) * W.H_phrase, 2 * len(S0_phrase), W.dt_ctrl,
               W.d_fb, kind="g6")
    g6 = dict(
        launch_tip_sd=float(np.linalg.norm(tip_ph.std(0))),
        launch_tip_disp=float(np.linalg.norm(tip_ph.mean(0) - W.goals[0])),
        launch_speed_mean=float(np.linalg.norm(S0_phrase[:, n:], axis=1).mean()),
        launch_null_coord_sd=float(nullc.std()),
        metering_noise_sd=noise,
        spread_over_noise=float(np.linalg.norm(tip_ph.std(0)) / max(noise, 1e-9)),
        chain_err_mean=[float(e_chain[:, k].mean()) for k in range(NS)],
        chain_err_cv=[float(e_chain[:, k].std() / max(e_chain[:, k].mean(), 1e-9))
                      for k in range(NS)],
        oracle_gain_phrase=info_ph["oracle_gain"],
        oracle_gain_by_seg=[aud[f"seg{k}"]["oracle_gain"] for k in range(NS)],
        # AUDITION OPTIMISM: the audition grades noiselessly, the arm performs with motor noise, and
        # over a 60-step chain that gap has 3x the distance to compound over that it did in round 1.
        # If it is large, a phrase library is selecting against the wrong objective -- worth knowing
        # BEFORE the main run rather than inferring from it afterwards.
        audition_optimism=[float(e_chain[:, k].mean() / max(e_chain_quiet[:, k].mean(), 1e-9))
                           for k in range(NS)],
        chain_err_quiet=[float(e_chain_quiet[:, k].mean()) for k in range(NS)],
        seam_spread=[dict(seg=k, tip_sd=float(np.linalg.norm(W.tip(st_chain[:, k]).std(0))),
                          speed=float(np.linalg.norm(st_chain[:, k, n:], axis=1).mean()))
                     for k in range(NS)])
    # the fusion price, per segment: predictable variance from the PHRASE launch vs from the SEAM
    r2_rows = []
    for k in range(NS):
        y = e_chain[:, k]
        row = dict(seg=k, r2_from_phrase_launch=r2_of(S0_phrase, y))
        row["r2_from_seam"] = (r2_of(st_chain[:, k - 1], y) if k > 0 else row["r2_from_phrase_launch"])
        row["fusion_information_loss"] = row["r2_from_seam"] - row["r2_from_phrase_launch"]
        r2_rows.append(row)
    g6["r2"] = r2_rows
    out["g6"] = g6
    P(f"[G6] phrase-launch tip sd {g6['launch_tip_sd']:.4f} (null-coord sd "
      f"{g6['launch_null_coord_sd']:.4f}, speed {g6['launch_speed_mean']:.3f}) vs metering noise "
      f"{noise:.4f} -> {g6['spread_over_noise']:.1f}x")
    P(f"[G6] oracle gain: phrase {g6['oracle_gain_phrase']:.2f}x  by-seg "
      f"{[round(v,2) for v in g6['oracle_gain_by_seg']]}")
    P(f"[G6] audition optimism (noisy/quiet chain error by seg): "
      f"{[round(v,2) for v in g6['audition_optimism']]}")
    for r in r2_rows:
        P(f"[G6] seg{r['seg']}: R^2 from phrase launch {r['r2_from_phrase_launch']:.3f} vs from "
          f"its own seam {r['r2_from_seam']:.3f} -> fusion loses "
          f"{r['fusion_information_loss']:+.3f}")
    save()

    # ================================================================= verdict
    ck = out["cal_descent"]["clocks"]
    out["verdict"] = {
        "G5_fm_h005_stale_executed": g5a["executed"]["stale"]["h_005"],
        "G5_fm_h005_ceiling_executed": g5a["executed"]["ceiling"]["h_005"],
        "G5_fm_err_at_seg_ends_stale": g5a["executed"]["stale"]["err_at"],
        "G5_live_piece_1group": g5b.get(str(NS), {}).get("stale", {}).get("e_piece"),
        "G5_live_piece_allsplit": g5b.get("+".join(["1"] * NS), {}).get("stale", {}).get("e_piece"),
        "G5_planner_starved": {f"span{k}": cal[f"span{k}"]["still_improving_stale"]
                               for k in range(1, NS + 1)},
        "G6_spread_over_noise": g6["spread_over_noise"],
        "G6_oracle_gain_phrase": g6["oracle_gain_phrase"],
        "G6_fusion_information_loss": [r["fusion_information_loss"] for r in r2_rows],
        "G7_usable_range_ball_seg": g7["modes"]["ball_seg"]["usable_range"],
        "G7_range_over_noise_ball_seg": g7["modes"]["ball_seg"]["range_over_noise"],
        "G7_ballistic_over_reactive": g7["ballistic_over_reactive"],
        "G7_phrase_over_seg": g7["phrase_over_seg"],
        "CLOCK_seg": [ck.get(f"seg{k}", {}).get("clock") for k in range(NS)],
        "CLOCK_ball_phrase": ck.get("ball_phrase_piece", {}).get("clock"),
    }
    out["ledger"] = led.snapshot()
    out["complete"] = True
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[verdict] {json.dumps(out['verdict'], default=str)}")
    P(f"[save] {outdir}")
    return out


@app.local_entrypoint()
def gates(
    quick: bool = False,
    tag: str = "",
    seed: int = 0,
    # WORLDS: "<label>:<curl_b>:<push_a>" -- one Modal container each. Round 1 already swept the
    # difficulty knob (b6/b14/b26/b14obs) and b14 was the design point on every gate; the knob this
    # round sweeps is SPAN, inside G5, so a single world is the default here.
    worlds: str = "b14:14:0",
    # ---- the piece
    waypoints: str = DEF_WPS,
    h_app: int = 14,
    h_seg: str = "20,20,20",
    patch_seg: int = DEF_PATCH_SEG,
    q_jit: float = 0.15,
    null_jit: float = 0.30,
    start_mode: str = "iso",
    # ---- the plant (byte-identical to round 1's design point)
    n_links: int = 3,
    link_lengths: str = "0.4,0.4,0.3",
    link_masses: str = "1.0,1.0,0.6",
    q_center: str = "0.4,1.1,0.8",
    joint_damping: float = 0.5,
    gear: float = 8.0,
    frame_skip: int = 10,
    timestep: float = 0.002,
    wrap_limit: float = 3.0,
    patch_sigma: float = 0.08,
    patch_center: str = "",
    push_sigma: float = 0.0,
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
    vel_pen: float = 0.5,             # terminal, applied only where a window reaches the piece end
    vel_pen_mid: float = 0.0,         # at every other span end -- SET BY CAL-C, this is the seed
    vel_pen_mid_grid: str = "0.0,0.25,0.5",
    w_waypoint: float = 0.0,          # extra cost weight at seam steps -- SET BY CAL-C (the l1
    w_waypoint_grid: str = "0.0,4.0,16.0",   # repair; 0.0 reproduces l0 exactly)
    cal_c_competence: float = 2.0,    # a cell must be within this factor of the best reactive
                                      # error to be eligible on the FM-attributable criterion
    axis_min_over_noise: float = 3.0, # below this, G7 reports the FM-quality axis as ABSENT
    commit_lookahead: int = 0,        # superseded by `lookahead_gamma` (kept for l0/l1 configs)
    lookahead_grid: str = "0,20",
    vel_pen_grid: str = "0.0,0.5",    # l2 stage 2: is there a terminal state on a closed loop?
    lookahead_gamma: float = 0.0,     # weight a live unit gives the segment it hands over to
    gamma_grid: str = "0.0,0.5,1.0",  # SET BY CAL-C stage 2
    react_look: int = 20,
    plan_max_elems: int = 40_000_000,
    # CAL-P must reach UPWARD from round 1's 1024:8, because a phrase plan optimises 180 action
    # dimensions against a segment's 60 and P3's failure mode is an undersized search.
    planner_grid: str = "1024:8,2048:10,4096:12",
    cal_p_tol: float = 0.002,
    # ---- practice / metering
    n_warm: int = 30,
    batch: int = 24,
    n_rt: int = 48,
    n_eval: int = 48,
    n_cal: int = 24,
    n_meter_rep: int = 3,
    react_every: int = 3,
    trace_window: int = 6,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
    # ---- the compile op
    n_cand: int = 48,
    n_score: int = 192,      # >=48 score states per library cell (crystallize: per-key in-sample ->
    n_lib: int = 4,          # held-out optimism was 1.56x at only 32)
    pol_hidden: int = 128,
    pol_layers: int = 2,
    pol_lr: float = 1e-3,
    pol_steps: int = 1500,
):
    if quick:
        pool_ou = 1500; pool_reach = 2500; n_corridor = 3
        fm_steps = 1200; fm_steps_boot = 600
        n_warm = 6; batch = 8; n_rt = 12; n_eval = 12; n_cal = 8; n_meter_rep = 2
        n_cand = 6; n_score = 16; n_lib = 2; react_every = 2
        k_shoot = 256; cem_iters = 4; planner_grid = "256:4,512:5"
        vel_pen_mid_grid = "0.0,0.5"; lookahead_grid = "0,20"; w_waypoint_grid = "0.0,8.0"
        vel_pen_grid = "0.0,0.5"; gamma_grid = "0.0,1.0"
        pol_steps = 300
        tag = tag or "smoke"
    tag = tag or "default"
    wl = []
    for spec in worlds.split(","):
        if not spec.strip():
            continue
        parts = spec.split(":")
        wl.append((parts[0], float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
    base = dict(
        tag=tag, seed=seed,
        waypoints=[[float(v) for v in p.split(",")] for p in waypoints.split(";") if p],
        h_app=h_app, h_seg=[int(v) for v in h_seg.split(",")], patch_seg=patch_seg,
        q_jit=q_jit, null_jit=null_jit, start_mode=start_mode,
        n_links=n_links, link_lengths=[float(x) for x in link_lengths.split(",")],
        link_masses=[float(x) for x in link_masses.split(",")],
        q_center=[float(x) for x in q_center.split(",")],
        joint_damping=joint_damping, gear=gear, frame_skip=frame_skip, timestep=timestep,
        wrap_limit=wrap_limit, patch_sigma=patch_sigma,
        patch_center=([float(x) for x in patch_center.split(",")] if patch_center else None),
        push_sigma=(push_sigma or patch_sigma), push_center=None,
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
        vel_pen_mid_grid=[float(v) for v in vel_pen_mid_grid.split(",") if v],
        w_waypoint=w_waypoint,
        w_waypoint_grid=[float(v) for v in w_waypoint_grid.split(",") if v],
        cal_c_competence=cal_c_competence, axis_min_over_noise=axis_min_over_noise,
        commit_lookahead=commit_lookahead,
        lookahead_grid=[int(v) for v in lookahead_grid.split(",") if v],
        vel_pen_grid=[float(v) for v in vel_pen_grid.split(",") if v],
        lookahead_gamma=lookahead_gamma,
        gamma_grid=[float(v) for v in gamma_grid.split(",") if v],
        react_look=react_look, plan_max_elems=plan_max_elems,
        planner_grid=[[int(v) for v in p.split(":")] for p in planner_grid.split(",") if p],
        cal_p_tol=cal_p_tol,
        n_warm=n_warm, batch=batch, n_rt=n_rt, n_eval=n_eval, n_cal=n_cal,
        n_meter_rep=n_meter_rep, react_every=react_every, trace_window=trace_window,
        sigma_practice=sigma_practice, sigma_perf=sigma_perf, d_fb=d_fb,
        n_cand=n_cand, n_score=n_score, n_lib=n_lib,
        pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr, pol_steps=pol_steps,
    )
    cfgs = [{**base, "world": lb, "curl_b": b, "push_a": a} for lb, b, a in wl]
    outs = list(run_gates.map(cfgs))
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for o in outs:
        with open(os.path.join(localdir, f"gates_{o['world']}.json"), "w") as fh:
            json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(outs)} world files to {localdir}")
