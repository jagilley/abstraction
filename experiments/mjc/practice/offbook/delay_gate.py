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

ROUND 5 (`d1`), ADDED 2026-08-26 -- THE CONTENT LADDER, BOLTED ON TO THE SAME GATE.
--------------------------------------------------------------------------------------------------
`d0` returned Delta* = None with every stored strategy sitting ABOVE the playability anchor already
at Delta = 0 (`seg_tape` 0.268, `chain` 0.270 vs `ref_stale` 0.146). On the SAME piece and the SAME
plant, `../legato/`'s plant-auditioned, keyed frozen units run 0.1026 (phrase) / 0.1065 (segment) --
UNDER d0's own playability bar before any delay is applied. So d0's stored content is ~2.5x worse
than content this substrate is known to be able to produce, and the question "is there a delay at
which memory pays?" was asked of a library that was never built the way legato builds one.

FOUR SETUP CHOICES SEPARATE d0's LIBRARY FROM LEGATO'S, and this round turns each into a rung:

  1. HARVEST. d0's tapes are the post-noise issued commands (`acts`) of 20 REACTIVE warm-up
     traversals at `sigma_practice = 0.15`, i.e. 2.5x the performance noise, recorded while the
     forward model was still being trained. Replayed open-loop such a tape carries one noise
     realisation PLUS the corrections the closed-loop controller made for it. Legato's chain pool
     additionally comes from SEGMENT-COMMITTED traversals (F5: the lower level's library funds the
     upper level's addressable variation) -- `world.Library`'s own docstring says a chain pool
     harvested from reactive warm-up is the configuration F1 measured at 1.03x, and d0 harvested
     exactly that way. Rungs: `fresh` (same sigma, post-training pool -- the vintage control),
     `perf` (sigma_perf, matched noise seed), `raw` (the SAME traversals' pre-noise commands), `f5`
     (chain cells from segment-committed play).
  2. SELECTION. d0 draws `lib_k = 72` tapes per cell UNIFORMLY and never cross-validates them
     (justified by etude E-3b's winner's curse -- ranking a tape by its OWN realised error). Legato
     instead pays for a PLANT AUDITION: `n_cand` candidates replayed open-loop on `n_score` recorded
     launch states, and `select_library` commits one argmin per cell of the launch distribution.
     That is not the winner's curse -- it scores on states the tape never saw. Rungs: `sel`
     (audition), plus `small` (uniform, size-matched) so the audition is not credited with the
     effect of simply holding fewer entries.
  3./4. THE READ. d0 selects at the seam by FM audition (Phase A G-C: rho = 0.924 at segment span,
     0.337 at chain span -- blind exactly where the chain lives) and launches from whatever posture
     the body reached. Legato KEYS: nearest launch-state centroid picks the rendition. `RoutePolicy`
     already has that mode. Rungs: `key_d0`, `f5_key`, `all4`.

The piece, the plant, the FM, the eval geometries, the Delta sweep, the four strategies, the
pre-fixed criterion and the `ref_stale` anchor are all UNTOUCHED. `reactive` and `live_seg` are
library-INDEPENDENT by construction (reactive never reads the library; `live_seg` restricts the
action set to the live plan), so every arm is scored against the same two, and the only thing that
moves across arms is how the library is built and read.

DISCIPLINE. Every flag added here is additive and its default is the donor's behaviour: with
`--arms ""` (the default) not one line of the round-4 path changes, and the ladder's own `d0` arm
re-runs the stored strategies through the ladder's code and asserts bit-identity against the
round-4 sweep in-run. The plant audition is CHARGED on the ledger (`who="agent"`), as legato
charges it, and each library's build cost is reported.

Run:
    modal run mjc/practice/offbook/delay_gate.py::delay_gate --quick --tag dsmoke
    modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d0 --seed 0
    modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d1 --seed 0 \
        --arms "d0,fresh,perf,raw,f5,small,sel,all4,f5_key,key_d0"
    modal run --detach mjc/practice/offbook/delay_gate.py::delay_gate --spawn --tag d2 --seed 0 \
        --arms "d0,sel,all4,nest_u,nest_u_key,nest,nest_key"

ROUND 6 (`d2`), ADDED 2026-08-26 -- LEGATO'S NESTING, THE INGREDIENT ROUND 5 WAS MISSING.
--------------------------------------------------------------------------------------------------
Round 5's `sel` matched legato at seam 0 and was ~4x worse at seams 1-2 ON THE PER-STATE ORACLE --
the best any candidate achieves per launch state, i.e. a property of the POOL, before selection:

    seam 0   legato chosen 0.0426 / oracle 0.0137     d1 `sel` 0.0414 / 0.0131   (match)
    seam 1   legato chosen 0.1271 / oracle 0.0481     d1 `sel` 0.2171 / 0.1995   (4.1x on oracle)
    seam 2   legato chosen 0.1993 / oracle 0.0554     d1 `sel` 0.3023 / 0.2280   (4.1x on oracle)

legato's own commit events say why: at its seam-1/2 commits only 2 of 6 pool traces were fully
reactive -- the other 4 were practice recorded IN THE ALREADY-COMMITTED CONFIGURATION, so its
candidates for seam k start from the frozen predecessor's arrival distribution, which is the same
distribution the score set is drawn from. `d1` harvested every segment pool from purely reactive
play, so candidate and score-set launch distributions were mismatched at every seam past the first.
`build_nested()` ports legato's sequential assembly whole (practice in the current configuration ->
score -> commit -> repeat; chain cells from the fully segment-committed configuration). Its knobs
are legato `L1`'s own: `n_nest` 7, `nest_window` 6, `nest_batch` 24, `nest_interleave` 3,
`n_cand` 64 / `n_cand_phrase` 48 / `n_score` 96 / `n_pick` 4.

ROUND 7 (`d3`), ADDED 2026-08-27 -- `--nest-adapt`: THE MODEL ADAPTS THROUGH THE NESTING, TOO.
--------------------------------------------------------------------------------------------------
Round 6 reproduced legato's launch-distribution nesting with the FM FROZEN at its post-warm state;
legato's adapted through its practice (39 cycles of online training by its seam-2 commit, 55 by the
phrase commit). That was the one remaining protocol difference, and it is not cosmetic: a tape
harvested off an UNCOMMITTED seam is a reactive rendition whose quality is bounded by the model that
produced it, and `live_seg` is a model bet outright. `--nest-adapt` turns legato's own
`train_online` call on between commits (true-state training data, per round 4's construction), then
freezes the model for the sweep.

Two consequences, handled rather than papered over:
  * the in-run identity of `reactive`/`live_seg` against ROUND 4 dissolves BY CONSTRUCTION. They are
    recomputed once under the adapted model and every arm is scored against those, so the CROSS-ARM
    identity still holds; the round-4 delta is reported instead of asserted. The `warm` library read
    by `audit` (`d0`) and by `key` (`key_d0`) is carried as the in-run control under the one adapted
    model -- and since a `key` read never touches the FM, `key_d0` and `all4` stay bit-identical to
    their round-5/6 twins, which is the check that the model did not leak where it should not.
  * `ref_stale` is DEFINED under the stale pre-practice FM at Delta = 0. It does not move and the
    criterion keeps using it untouched; the same anchor recomputed under the warm and the adapted
    models is reported beside it as two flagged numbers.
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
        # `acts_raw` is an ADDITIVE readout (round 5): the PRE-noise command the controller issued,
        # recorded on the same convention as `acts`. Nothing above reads it and no RNG draw moves,
        # so the round-4 path is byte-unchanged; the ladder's `raw` rung is the only consumer.
        traces.append(dict(acts=pr["acts"].copy(), acts_raw=pr["acts_raw"].copy(),
                           e_seg=pr["e_seg"].copy(),
                           launches={k: v.copy() for k, v in pr["launches"].items()}))
        if len(buf) > cfg["trace_window"]:
            buf.pop(0)
        PX, PY = W.tensors(*[np.concatenate([b[i] for b in buf]) for i in range(3)])
        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                       cfg["replay_frac"])

    layout = N.SlotLayout(NS, cfg["n_slot"], poison=False)

    def pool_rows(pool, ns, k, field="acts"):
        """`[(tape, launch state at seam k, realised error over the span)]` for cell (ns, k),
        sliced out of a harvest pool. `field` picks the ISSUED (post-noise) commands -- the donor's
        convention -- or the RAW pre-noise ones, which is round 5's `raw` rung."""
        rows = []
        lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k + ns - 1])
        for h in pool:
            if k not in h["launches"]:
                continue
            tp = h[field][:, lo:hi, :]
            er = np.nanmean(h["e_seg"][:, k:k + ns], 1)
            for b in range(len(tp)):
                if np.isfinite(tp[b]).all():
                    rows.append((tp[b], h["launches"][k][b], er[b]))
        return rows

    def degenerate_slots(tapes, n_slot):
        """`Cell._assign` hands slot assignment to k-means over CONTENT, and k-means++ cannot run
        when the drawn tapes hold fewer DISTINCT sequences than there are slots: the seeding step
        samples from an all-zero distance distribution and numpy rejects it. Round 4's pool is
        reactive play, where every rendition is unique, so the degeneracy was unreachable. Round
        5's `f5`/`small` pools are cut from KEYED play, where many performers replay the SAME
        stored rendition bit for bit, so it is reachable there -- and the smoke hit it.

        Returns an explicit assignment (identical tapes share a slot, ids by first appearance) in
        exactly that case and `None` -- the donor's k-means path, untouched -- otherwise. The
        distinct count is itself the quantity legato F5 is about (how much addressable variation
        the lower level manufactures), so it is logged either way."""
        X = np.asarray(tapes, np.float64).reshape(len(tapes), -1)
        _, first, inv = np.unique(X, axis=0, return_index=True, return_inverse=True)
        nd = int(len(first))
        if nd >= int(n_slot):
            return None, nd
        ids = {int(v): i for i, v in enumerate(np.argsort(first))}
        return np.array([ids[int(v)] for v in np.asarray(inv).ravel()], int), nd

    def build_uniform(pool_seg, pool_chain=None, lib_k=None, field="acts", pool_by_cell=None):
        """The donor's build, lifted verbatim into a function: `lib_k` UNIFORMLY drawn measured
        tapes per cell, slots fit by k-means over CONTENT, one shared `lrng` consumed cell by cell
        in layout order. `build_uniform(traces)` therefore reproduces round 4's library bit for
        bit; `pool_chain` (round 5) lets the ns>1 cells be drawn from a DIFFERENT pool than the
        ns=1 cells, which is legato F5's construction."""
        lib = Library(W, layout, cfg["n_slot"], cfg["seed"] + 2100)
        lrng = np.random.default_rng(cfg["seed"] + 820)
        K = int(cfg["lib_k"] if lib_k is None else lib_k)
        dist = {}
        for k in range(NS):
            for ns in layout.levels(k):
                # `pool_by_cell` (round 6) is the general form: under legato's SEQUENTIAL assembly
                # every cell has its own candidate pool, because the pool for seam k is practice
                # recorded in the configuration where seams 0..k-1 are already committed.
                src = (pool_by_cell[(ns, k)] if pool_by_cell is not None
                       else (pool_seg if (ns == 1 or pool_chain is None) else pool_chain))
                rows = pool_rows(src, ns, k, field)
                pick = lrng.permutation(len(rows))[:K]
                T = np.stack([rows[i][0] for i in pick])
                L = np.stack([rows[i][1] for i in pick])
                E = np.array([rows[i][2] for i in pick], np.float32)
                lab, nd = degenerate_slots(T, cfg["n_slot"])
                dist[f"{ns}:{k}"] = dict(n_rows=int(len(rows)), n_drawn=int(len(T)),
                                         n_distinct=nd, kmeans_slots=bool(lab is None))
                if lab is None:
                    lib.cell(ns, k).add(T, L, E, 0, -np.ones((len(T), ns), int), lib.rng)
                else:
                    for j in sorted(set(int(x) for x in lab)):
                        m = lab == j
                        lib.cell(ns, k).add(T[m], L[m], E[m], 0,
                                            -np.ones((int(m.sum()), ns), int), lib.rng, slot=j)
        lib.d1_pool = dist
        return lib

    library = build_uniform(traces)
    P(f"[lib] {library.sizes()}")

    # ---------------------------------------------------------------- the sweep
    q_ev = geom(cfg["n_eval_gate"], cfg["seed"] + 5200)
    plan_ev = app_plan(q_ev, cfg["seed"] + 5250)
    chain_levels = {ns for k in range(NS) for ns in layout.levels(k) if ns > 1}
    seam_S = None

    def mk_pol(levels, prim, lib=None, mode="audit"):
        """`lib`/`mode` are round-5 additions whose defaults are the donor's exact call: the round-4
        library and seam-time FM audition. `mode="key"` is legato's frozen key -- nearest launch-
        state centroid picks the slot, its representative tape flies, no audition and no live
        option -- which is `RoutePolicy`'s own `key` branch, unmodified."""
        pol = RoutePolicy(mode, W, layout, library if lib is None else lib,
                          k_prop=layout.n_slots, eps=0.0,
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
    save()

    # ============================================================================== ROUND 5 (`d1`)
    # THE CONTENT LADDER. Gated entirely on `--arms`; with the default (empty) everything below is
    # skipped and the run above IS round 4. See the module docstring for the four rungs.
    # ==============================================================================================
    arm_names = [a.strip() for a in str(cfg.get("arms", "") or "").split(",") if a.strip()]
    if arm_names:
        NAMED = {                       # arm -> (library key, seam-selector mode)
            "d0":     ("warm",  "audit"),   # the in-run control: must equal the round-4 sweep
            "fresh":  ("fresh", "audit"),   # + post-training pool at the SAME sigma (vintage)
            "perf":   ("perf",  "audit"),   # + harvested at sigma_perf, matched noise seed
            "raw":    ("raw",   "audit"),   # + the SAME traversals' PRE-noise commands
            "f5":     ("f5",    "audit"),   # + chain cells from segment-committed play (legato F5)
            "small":  ("small", "audit"),   # size control for `sel`: uniform, n_pick entries
            "sel":    ("sel",   "audit"),   # + legato's cross-state PLANT audition
            "all4":   ("sel",   "key"),     # + legato's launch key   <- all four defects fixed
            "f5_key": ("f5",    "key"),     # the key on uncurated F5 content
            "key_d0": ("warm",  "key"),     # the key alone, on d0's own content
            # ---- round 6 (`d2`): legato's SEQUENTIAL nesting, the missing ingredient ----
            "nest":       ("nest",   "audit"),  # nested pools + plant audition, FM-audit read
            "nest_key":   ("nest",   "key"),    # <- THE LEGATO CONFIGURATION, end to end
            "nest_u":     ("nest_u", "audit"),  # nested pools, uniform pick: isolates the POOL
            "nest_u_key": ("nest_u", "key"),
        }
        bad = [a for a in arm_names if a not in NAMED]
        if bad:
            raise ValueError(f"unknown arms {bad}; known: {sorted(NAMED)}")
        need = {NAMED[a][0] for a in arm_names}
        lad = out.setdefault("ladder", {"arms": {}, "libraries": {}})
        P(f"[ladder] arms={arm_names}  libraries needed={sorted(need)}")

        q_sc = geom(cfg["n_score"], cfg["seed"] + 5100)
        plan_sc = app_plan(q_sc, cfg["seed"] + 5150)

        def snap():
            return dict(t_priced=float(led.t_priced), steps=int(led.steps["agent"]),
                        fb=int(led.fb["agent"]), plans=int(led.plans["agent"]),
                        delib=int(led.delib["agent"]))

        def cost_since(a):
            b = snap()
            return {k: (b[k] - a[k]) for k in b}

        # ------------------------------------------------------------------ the harvest pools
        # All harvest traversals run with the FM FROZEN at its post-warm state and on the SAME
        # practice geometries and the SAME noise seed, so `fresh` and `perf` differ only in the
        # scale of the motor noise and `perf`/`raw` are the identical traversals read two ways.
        def harvest_reactive(sigma, tag):
            pool = []
            for c in range(1, int(cfg["n_harvest"]) + 1):
                qh = geom(cfg["batch"], cfg["seed"] + 9600 + c)
                o = W.traverse(fm, R_REACT, qh,
                               np.random.default_rng(cfg["seed"] + 9800 + c), sigma, led,
                               who="agent", kind=f"harvest_{tag}",
                               approach_plan=app_plan(qh, cfg["seed"] + 9700 + c))
                pool.append(dict(acts=o["acts"].copy(), acts_raw=o["acts_raw"].copy(),
                                 e_seg=o["e_seg"].copy(),
                                 launches={k: v.copy() for k, v in o["launches"].items()}))
            return pool

        def harvest_routed(lib, sigma, tag):
            """legato F5's construction: the pool the CHAIN cells are cut from is produced by
            playing the piece through the arm's own committed SEGMENT library, keyed at each seam,
            so different performers spell different whole-piece chains. `world.Library`'s docstring
            names the alternative -- a chain pool harvested from reactive warm-up -- as the
            configuration F1 measured at 1.03x, which is what round 4 ran."""
            pool = []
            for c in range(1, int(cfg["n_harvest"]) + 1):
                qh = geom(cfg["batch"], cfg["seed"] + 9600 + c)
                o = W.traverse_route(fm, mk_pol({1}, False, lib, "key"), qh,
                                     np.random.default_rng(cfg["seed"] + 9900 + c), sigma, led,
                                     who="agent", kind=f"harvest_{tag}",
                                     approach_plan=app_plan(qh, cfg["seed"] + 9700 + c))
                pool.append(dict(acts=o["acts"].copy(), acts_raw=o["acts_raw"].copy(),
                                 e_seg=o["e_seg"].copy(),
                                 launches={k: v.copy() for k, v in o["launches"].items()}))
            return pool

        # ------------------------------------------------------------------ legato's compile op
        def build_auditioned(pool_seg, pool_chain, field="acts_raw"):
            """`legato/legato.py::compile_unit`, ported onto this node's `Cell`/`RoutePolicy`.

            Cells are committed LEFT TO RIGHT (the etude E-4 seam-matched pattern legato follows):
            cell (1, k)'s score set is the launch distribution at seam k produced by playing
            segments 0..k-1 through the cells already committed, on HELD-OUT geometries. Candidates
            are drawn uniformly (never top-of-pool -- E-3b's winner's curse), replayed OPEN-LOOP on
            the plant from every score state, and `World.select_library` commits one argmin per
            k-means cell of the launch distribution. Each committed rendition is stored in its own
            slot with its cell's launch centroid as the key, so `RoutePolicy`'s `key` branch
            reproduces legato's launch-keyed frozen unit exactly and its `audit` branch sees a
            cross-state-validated action set instead of a uniform pile.

            PRICED: the audition is charged to the agent (`who="agent"`), as legato charges it, and
            so are the score-set traversals -- the body has to play up to a seam to find out where
            it arrives."""
            lib = Library(W, layout, cfg["n_slot"], cfg["seed"] + 2100)
            rb = np.random.default_rng(cfg["seed"] + 2400)
            diag, S = {}, {}
            n_cand, n_pick = int(cfg["n_cand"]), int(cfg["n_pick"])

            def commit(ns, k):
                rows = pool_rows(pool_seg if ns == 1 else pool_chain, ns, k, field)
                ci = rb.permutation(len(rows))[:n_cand]
                cands = np.stack([rows[i][0] for i in ci])
                E, _ = W.audition(cands, S[k], W.checkpoints(k, ns), led, who="agent")
                u, li = W.select_library(E, cands, S[k], n_pick, rb)
                keys_raw = (np.asarray(u["keys"]) * u["key_sd"][None, :]
                            + u["key_mu"][None, :]).astype(np.float32)
                c = lib.cell(ns, k)
                for j in range(len(u["seqs"])):
                    c.add(np.asarray(u["seqs"][j], np.float32)[None], keys_raw[j][None],
                          np.array([float(E[li["picks"][j]].mean())], np.float32),
                          0, -np.ones((1, ns), int), lib.rng, slot=j)
                c.cent = np.asarray(u["seqs"], np.float64).reshape(len(u["seqs"]), -1)
                diag[f"{ns}:{k}"] = dict(
                    n_cand=int(len(cands)), n_score=int(len(S[k])), n_pick=int(len(u["seqs"])),
                    n_distinct=int(li["n_distinct"]), cell_sizes=li["cell_sizes"],
                    score_med=float(np.median(E.mean(1))), best_fixed=float(E.mean(1).min()),
                    per_state_oracle=float(E.min(0).mean()),
                    oracle_gain=float(E.mean(1).min() / max(E.min(0).mean(), 1e-9)),
                    chosen_score=float(np.mean([E[li["picks"][int(l_)], i]
                                                for i, l_ in enumerate(li["labels"])])),
                    launch_tip_spread=float(np.linalg.norm(W.tip(S[k]).std(0))))
                P(f"[compile] cell {ns}:{k} n_cand={len(cands)} n_score={len(S[k])} "
                  f"picks={li['n_distinct']}/{len(u['seqs'])} chosen={diag[f'{ns}:{k}']['chosen_score']:.4f} "
                  f"best_fixed={diag[f'{ns}:{k}']['best_fixed']:.4f} "
                  f"oracle={diag[f'{ns}:{k}']['per_state_oracle']:.4f} "
                  f"({diag[f'{ns}:{k}']['oracle_gain']:.2f}x)")

            S[0] = W.traverse(fm, R_REACT, q_sc, np.random.default_rng(cfg["seed"] + 5300),
                              cfg["sigma_perf"], led, who="agent", kind="score_set",
                              approach_plan=plan_sc, stop_seg=0)["launch"]
            commit(1, 0)
            for k in range(1, NS):
                S[k] = W.traverse_route(fm, mk_pol({1}, False, lib, "key"), q_sc,
                                        np.random.default_rng(cfg["seed"] + 5300 + k),
                                        cfg["sigma_perf"], led, who="agent", kind="score_set",
                                        approach_plan=plan_sc, stop_seg=k)["launches"][k]
                commit(1, k)
            for k in range(NS):
                for ns in layout.levels(k):
                    if ns > 1:
                        commit(ns, k)
            return lib, diag

        # ------------------------------------------------------------------ legato's NESTING
        def build_nested():
            """ROUND 6. `legato/legato.py`'s SEQUENTIAL assembly, ported whole -- the ingredient
            round 5 was missing.

            THE DIAGNOSIS THIS IMPLEMENTS. Round 5's `sel` matched legato at seam 0
            (chosen 0.0414 / per-state oracle 0.0131 vs legato's 0.0426 / 0.0137) and was ~4x worse
            at seams 1-2 *on the per-state oracle* (0.1995 / 0.2280 vs 0.0481 / 0.0554). The oracle
            is the best ANY candidate achieves per launch state, so the gap is in the POOL, before
            selection: round 5 harvested every segment pool from purely REACTIVE traversals, while
            legato's events show 4 of its 6 pool traces at the seam-1/2 commits were practice
            recorded in the ALREADY-COMMITTED configuration. A candidate for seam k has to be a
            rendition that starts from the frozen predecessor's arrival distribution -- the same
            distribution the score set is drawn from. Reactive practice never visits it.

            THE PROCEDURE, legato's own:
              stage k = 0..NS-1 :  practice `n_nest` cycles in the CURRENT configuration at
                                   sigma_practice (committed seams played by their keyed unit,
                                   uncommitted seams reactive, one cycle in `nest_interleave`
                                   fully reactive -- legato's diet-rent interleave), keeping a
                                   `nest_window`-deep trace buffer; then score at seam k from that
                                   same configuration and commit cell (1, k) by plant audition.
              stage NS         :  practice `n_nest` more cycles in the FULLY segment-committed
                                   configuration and commit the chain cells from THOSE traces --
                                   legato F5 done properly, rather than round 5's `f5`
                                   approximation (uniform keyed play).

            Practice and score-set traversals run through `World.traverse` with legato's OWN `lib`
            unit (`World.select_library`'s return value is exactly what `unit_commands`' `"lib"`
            branch consumes), so the configuration the pool is recorded in is legato's, bit for bit.
            The same renditions are also stored into this node's `Cell`s so the gate can read them
            through `RoutePolicy` in either mode.

            ONE DELIBERATE DEVIATION, and it is the main caveat: the forward model stays FROZEN at
            its post-warm state throughout. legato adapted its FM over 90 cycles. Freezing is what
            keeps `reactive`, `live_seg` and `ref_stale` bit-identical to round 4 -- without it the
            in-run control and the whole cross-round comparison dissolve. It means d2 reproduces
            legato's LAUNCH-DISTRIBUTION nesting but not legato's model quality."""
            lib = Library(W, layout, cfg["n_slot"], cfg["seed"] + 2100)
            rb = np.random.default_rng(cfg["seed"] + 2600)
            seg_units = [None] * NS
            tb, by_cell, diag = [], {}, {}
            state = {"cyc": 0}
            # ROUND 7 (`d3`): legato also ADAPTED its forward model through these practice cycles
            # (39 cycles of online training by its seam-2 commit, 55 by the phrase commit), and a
            # tape harvested off an uncommitted seam is a reactive rendition whose quality is
            # bounded by the model that produced it. `nest_adapt` turns that on -- plain
            # `train_online` on the practice transitions between commits, legato's own call, with
            # its own batch stream. Training data stays TRUE-state (round 4's construction: the
            # delay is a decision constraint, never a learning-data treatment), and the model is
            # frozen again for the sweep. At `nest_adapt=False` no transition is collected, no
            # optimizer step is taken, and the build is bit-identical to round 6's.
            adapt = bool(cfg.get("nest_adapt", False))
            tbuf = []
            nrng = np.random.default_rng(cfg["seed"] + 850)
            ladder_n = []

            def react_probe(stage):
                """The FM's own competence, held out and free (`who="instrument"`), recorded at
                every stage boundary so the model behind each commit is on the record next to
                legato's (its `e_react` ran ~0.010 at c25-39)."""
                o = W.traverse(fm, R_REACT, q_ev,
                               np.random.default_rng(cfg["seed"] + 7300 + state["cyc"]),
                               cfg["sigma_perf"], led, who="instrument", kind="nest_probe",
                               approach_plan=plan_ev)
                r = dict(stage=stage, nest_cycle=int(state["cyc"]),
                         e_react_med=float(np.median(o["e_piece"])),
                         e_react_mean=float(o["e_piece"].mean()),
                         by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)])
                ladder_n.append(r)
                P(f"[nest-probe] {stage:<10} nest-cycle {r['nest_cycle']:2d}  e_react "
                  f"med {r['e_react_med']:.4f} mean {r['e_react_mean']:.4f}  "
                  f"by-seg {[round(x, 4) for x in r['by_seg']]}")

            def cur_routing():
                """legato's `perf_routing()`: committed seams play their keyed unit, uncommitted
                seams are reactive. An arm is a pure function of which commitments have fired."""
                return W.routing(None, groups=[1] * NS,
                                 units=[seg_units[k] if seg_units[k] is not None
                                        else W.reactive_unit() for k in range(NS)])

            def practice(n):
                for _ in range(int(n)):
                    state["cyc"] += 1
                    c = state["cyc"]
                    any_c = any(u is not None for u in seg_units)
                    inter = bool(any_c and int(cfg["nest_interleave"]) > 0
                                 and c % int(cfg["nest_interleave"]) == 0)
                    qp = geom(int(cfg["nest_batch"]), cfg["seed"] + 12_000 + c)
                    o = W.traverse(fm, R_REACT if inter else cur_routing(), qp,
                                   np.random.default_rng(cfg["seed"] + 13_000 + c),
                                   cfg["sigma_practice"], led, who="agent",
                                   kind="nest_practice_interleave" if inter else "nest_practice",
                                   approach_plan=app_plan(qp, cfg["seed"] + 12_500 + c),
                                   collect=adapt)
                    tb.append(dict(acts=o["acts"].copy(), acts_raw=o["acts_raw"].copy(),
                                   e_seg=o["e_seg"].copy(),
                                   launches={k: v.copy() for k, v in o["launches"].items()},
                                   reactive=bool(inter or not any_c)))
                    if len(tb) > int(cfg["nest_window"]):
                        tb.pop(0)
                    if adapt:
                        # legato's own plasticity call, verbatim: uniform lr on a window of
                        # practice transitions plus the replay fraction. `collect=True` records
                        # states/actions only and draws no randomness, so with `adapt=False`
                        # nothing here executes and round 6 is reproduced exactly.
                        tbuf.append(o["trans"])
                        if len(tbuf) > int(cfg["nest_window"]):
                            tbuf.pop(0)
                        PX, PY = W.tensors(*[np.concatenate([b[i] for b in tbuf])
                                             for i in range(3)])
                        W.train_online(fm, optf, PX, PY, RX, RY, cfg["n_grad"], nrng,
                                       cfg["fm_batch"], cfg["replay_frac"])

            def commit(ns, k, n_cand):
                """legato's `compile_unit`: score set from the CURRENT configuration truncated at
                the span's start, uniform candidates from the current trace window, plant audition,
                `select_library`. Everything here is charged to the agent."""
                S0 = W.traverse(fm, cur_routing(), q_sc,
                                np.random.default_rng(cfg["seed"] + 5400 + 10 * ns + k),
                                cfg["sigma_perf"], led, who="agent", kind="score_set",
                                approach_plan=plan_sc, stop_seg=k)
                S0 = S0["launch"] if k == 0 else S0["launches"][k]
                snap_ = [dict(h) for h in tb]
                by_cell[(ns, k)] = snap_
                rows = pool_rows(snap_, ns, k, "acts")     # legato's field: post-noise `acts`
                ci = rb.permutation(len(rows))[:int(n_cand)]
                cands = np.stack([rows[i][0] for i in ci])
                E, _ = W.audition(cands, S0, W.checkpoints(k, ns), led, who="agent")
                u, li = W.select_library(E, cands, S0, int(cfg["n_pick"]), rb)
                keys_raw = (np.asarray(u["keys"]) * u["key_sd"][None, :]
                            + u["key_mu"][None, :]).astype(np.float32)
                c = lib.cell(ns, k)
                for j in range(len(u["seqs"])):
                    c.add(np.asarray(u["seqs"][j], np.float32)[None], keys_raw[j][None],
                          np.array([float(E[li["picks"][j]].mean())], np.float32),
                          0, -np.ones((1, ns), int), lib.rng, slot=j)
                c.cent = np.asarray(u["seqs"], np.float64).reshape(len(u["seqs"]), -1)
                if ns == 1:
                    seg_units[k] = u          # legato's own unit object, for the practice routing
                d_ = dict(n_cand=int(len(cands)), n_score=int(len(S0)),
                          n_pick=int(len(u["seqs"])), n_distinct=int(li["n_distinct"]),
                          cell_sizes=li["cell_sizes"], n_pool=int(len(rows)),
                          n_traces=int(len(snap_)),
                          n_reactive_traces=int(sum(1 for h in snap_ if h["reactive"])),
                          cycle=int(state["cyc"]),
                          score_med=float(np.median(E.mean(1))),
                          best_fixed=float(E.mean(1).min()),
                          per_state_oracle=float(E.min(0).mean()),
                          oracle_gain=float(E.mean(1).min() / max(E.min(0).mean(), 1e-9)),
                          chosen_score=float(np.mean([E[li["picks"][int(l_)], i]
                                                      for i, l_ in enumerate(li["labels"])])),
                          launch_tip_spread=float(np.linalg.norm(W.tip(S0).std(0))))
                diag[f"{ns}:{k}"] = d_
                P(f"[nest] commit cell {ns}:{k} at nest-cycle {d_['cycle']}  "
                  f"pool={d_['n_pool']} ({d_['n_traces']} traces, {d_['n_reactive_traces']} "
                  f"reactive)  n_cand={d_['n_cand']}x{d_['n_score']}  "
                  f"chosen={d_['chosen_score']:.4f} best_fixed={d_['best_fixed']:.4f} "
                  f"oracle={d_['per_state_oracle']:.4f} ({d_['oracle_gain']:.2f}x)  "
                  f"spread={d_['launch_tip_spread']:.4f}  picks={d_['n_distinct']}/{d_['n_pick']}")

            react_probe("start")
            for k in range(NS):
                practice(cfg["n_nest"])
                react_probe(f"pre_seg{k}")
                commit(1, k, cfg["n_cand"])
            practice(cfg["n_nest"])            # the fully segment-committed configuration
            react_probe("pre_chain")
            for k in range(NS):
                for ns in layout.levels(k):
                    if ns > 1:
                        commit(ns, k, cfg["n_cand_phrase"])
            return lib, diag, by_cell, ladder_n

        # ------------------------------------------------------------------ build the libraries
        libs, pools = {"warm": library}, {}
        lad["libraries"]["warm"] = dict(
            sizes=library.sizes(), harvest="warm reactive traversals during FM training, "
                                           "post-noise acts at sigma_practice",
            selection=f"uniform {int(cfg['lib_k'])}/cell", build_cost=None,
            pool=getattr(library, "d1_pool", None))
        if need - {"warm"}:
            lad["harvest_cost"] = {}
            if "fresh" in need:
                a0 = snap()
                pools["fresh"] = harvest_reactive(cfg["sigma_practice"], "fresh")
                lad["harvest_cost"]["fresh"] = cost_since(a0)
            if need & {"perf", "raw", "f5", "small", "sel"}:
                a0 = snap()
                pools["perf"] = harvest_reactive(cfg["sigma_perf"], "perf")
                lad["harvest_cost"]["perf"] = cost_since(a0)
            P(f"[harvest] pools {sorted(pools)}: {cfg['n_harvest']} x {cfg['batch']} traversals each")

            def add(name, mk, meta):
                a = snap()
                obj = mk()
                lib_, dg = obj if isinstance(obj, tuple) else (obj, None)
                libs[name] = lib_
                lad["libraries"][name] = dict(sizes=lib_.sizes(), build_cost=cost_since(a),
                                              audition=dg,
                                              pool=getattr(lib_, "d1_pool", None), **meta)
                P(f"[lib:{name}] {lib_.sizes()}  build_cost={lad['libraries'][name]['build_cost']}")

            if "fresh" in need:
                add("fresh", lambda: build_uniform(pools["fresh"]),
                    dict(harvest="post-training reactive pool, post-noise acts @ sigma_practice",
                         selection=f"uniform {int(cfg['lib_k'])}/cell"))
            if need & {"perf", "raw", "f5", "small", "sel"}:
                add("perf", lambda: build_uniform(pools["perf"]),
                    dict(harvest="post-training reactive pool, post-noise acts @ sigma_perf",
                         selection=f"uniform {int(cfg['lib_k'])}/cell"))
            if need & {"raw", "f5", "small", "sel"}:
                add("raw", lambda: build_uniform(pools["perf"], field="acts_raw"),
                    dict(harvest="the SAME sigma_perf traversals, PRE-noise commands",
                         selection=f"uniform {int(cfg['lib_k'])}/cell"))
            if need & {"f5", "small", "sel"}:
                a0 = snap()
                pools["route"] = harvest_routed(libs["raw"], cfg["sigma_perf"], "route")
                lad["harvest_cost"]["route"] = cost_since(a0)
                P(f"[harvest] route pool via `raw`'s keyed segment library: "
                  f"{cfg['n_harvest']} x {cfg['batch']} traversals")
            if "f5" in need:
                add("f5", lambda: build_uniform(pools["perf"], pools["route"], field="acts_raw"),
                    dict(harvest="ns=1 from the raw sigma_perf pool; ns>1 from SEGMENT-COMMITTED "
                                 "play (legato F5)", selection=f"uniform {int(cfg['lib_k'])}/cell"))
            if "small" in need:
                add("small", lambda: build_uniform(pools["perf"], pools["route"],
                                                   lib_k=int(cfg["n_pick"]), field="acts_raw"),
                    dict(harvest="as `f5`",
                         selection=f"uniform {int(cfg['n_pick'])}/cell (size control for `sel`)"))
            if "sel" in need:
                add("sel", lambda: build_auditioned(pools["perf"], pools["route"],
                                                    field="acts_raw"),
                    dict(harvest="as `f5`",
                         selection=f"legato PLANT AUDITION: {int(cfg['n_cand'])} candidates x "
                                   f"{int(cfg['n_score'])} recorded launch states, "
                                   f"{int(cfg['n_pick'])} committed per cell"))
        if cfg.get("nest_adapt", False):
            # the same ballistic-per-segment anchor under the model the nesting STARTS from, so the
            # pre/post pair brackets exactly what the 28 adapted cycles bought. Instrument, free.
            lad["ref_stale_warm"] = float(W.traverse(
                fm, R_BALL_SEG, q_ev, np.random.default_rng(cfg["seed"] + 7240),
                cfg["sigma_perf"], led, who="instrument", kind="ref_stale_warm",
                approach_plan=plan_ev)["e_piece"].mean())
            P(f"[ref] anchor under the WARM (pre-nesting) FM: {lad['ref_stale_warm']:.4f}")
        if need & {"nest", "nest_u"}:
            a0 = snap()
            nest_lib, nest_diag, nest_pools, nest_ladder = build_nested()
            c_nest = cost_since(a0)
            if "nest" in need:
                libs["nest"] = nest_lib
                lad["libraries"]["nest"] = dict(
                    sizes=nest_lib.sizes(), build_cost=c_nest, audition=nest_diag, pool=None,
                    harvest=f"legato NESTING: {int(cfg['n_nest'])} practice cycles per stage at "
                            f"sigma_practice in the CURRENT configuration (batch "
                            f"{int(cfg['nest_batch'])}, window {int(cfg['nest_window'])}, "
                            f"interleave {int(cfg['nest_interleave'])}); chain cells from the "
                            f"fully segment-committed configuration",
                    selection=f"legato PLANT AUDITION: {int(cfg['n_cand'])}"
                              f"(chain {int(cfg['n_cand_phrase'])}) candidates x "
                              f"{int(cfg['n_score'])} recorded launch states, "
                              f"{int(cfg['n_pick'])} committed per cell")
                lad["libraries"]["nest"]["fm_ladder"] = nest_ladder
                lad["libraries"]["nest"]["fm_adapted"] = bool(cfg.get("nest_adapt", False))
                P(f"[lib:nest] {nest_lib.sizes()}  build_cost={c_nest}")
            if "nest_u" in need:
                a1 = snap()
                lu = build_uniform(None, lib_k=int(cfg["n_pick"]), field="acts",
                                   pool_by_cell=nest_pools)
                libs["nest_u"] = lu
                lad["libraries"]["nest_u"] = dict(
                    sizes=lu.sizes(), build_cost=cost_since(a1),
                    audition=None, pool=getattr(lu, "d1_pool", None),
                    harvest="the SAME per-cell nested pools as `nest` (snapshotted at each of its "
                            "commits), so this arm isolates SELECTION given legato's pool",
                    selection=f"uniform {int(cfg['n_pick'])}/cell")
                P(f"[lib:nest_u] {lu.sizes()}")

        # ------------------------------------------------------------------ the per-arm sweep
        # `reactive` and `live_seg` are library-INDEPENDENT (reactive never reads the library;
        # `live_seg` restricts the action set to the live plan), so every arm is scored against the
        # round-4 values for them and only the two STORED strategies are re-run.
        # ---------------- the base the arms are scored against --------------------------------
        # `reactive` and `live_seg` are library-independent but MODEL-dependent. With the FM frozen
        # (rounds 5-6) they are round 4's own numbers and the `d0` arm is bit-identical to it. With
        # `nest_adapt` the nesting has moved the model, so round 4's numbers no longer describe
        # this world: they are recomputed ONCE here under the adapted FM (same rng seeds, same
        # geometries, same code path), every arm is scored against those, and the cross-arm
        # identity assertion still holds. What dissolves is only the identity against ROUND 4 --
        # by construction, and the delta is reported instead of asserted away.
        def base_pair(tag_, predict=False):
            """`reactive` and `live_seg` at every Delta -- the two library-INDEPENDENT strategies,
            recomputed whenever the model or the OBSERVATION OPERATOR changes. Same rng seeds, same
            geometries, same code path as round 4's."""
            out_ = {}
            for D in [int(x) for x in cfg["delays"]]:
                W.obs_delay = D
                W.obs_predict = bool(predict)
                o = W.traverse(fm, R_REACT, q_ev, np.random.default_rng(cfg["seed"] + 7700),
                               cfg["sigma_perf"], led, who="instrument",
                               kind=f"react_{tag_}_d{D}", approach_plan=plan_ev)
                cell = dict(reactive=dict(
                    e=float(o["e_piece"].mean()), e_med=float(np.median(o["e_piece"])),
                    by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)],
                    fb=float(o["n_fb"]), planner_free=False))
                oo = W.traverse_route(fm, mk_pol(set(), True), q_ev,
                                      np.random.default_rng(cfg["seed"] + 7710),
                                      cfg["sigma_perf"], led, who="instrument",
                                      kind=f"live_seg_{tag_}_d{D}", approach_plan=plan_ev)
                cell["live_seg"] = dict(
                    e=float(oo["e_piece"].mean()), e_med=float(np.median(oo["e_piece"])),
                    by_seg=[float(np.median(oo["e_seg"][:, k])) for k in range(NS)],
                    fb=float(oo["n_fb"]), planner_free=False)
                out_[str(D)] = cell
            W.obs_delay = 0
            W.obs_predict = False
            return out_

        adapted = bool(cfg.get("nest_adapt", False)) and bool(need & {"nest", "nest_u"})
        base_sweep = sweep
        if adapted:
            base_sweep = base_pair("adapt")
            for D in [int(x) for x in cfg["delays"]]:
                cell = base_sweep[str(D)]
                P(f"[adapted D={D:2d}] reactive={cell['reactive']['e']:.4f} "
                  f"live_seg={cell['live_seg']['e']:.4f}  (round 4: "
                  f"{sweep[str(D)]['reactive']['e']:.4f} / {sweep[str(D)]['live_seg']['e']:.4f})")
            W.obs_delay = 0
            lad["base_sweep_adapted"] = base_sweep
            # the playability anchor: `ref_stale` is DEFINED under the stale pre-practice FM at
            # Delta = 0 and does not move, so the criterion keeps using it untouched. The same
            # quantity under the adapted FM is reported beside it as a second, FLAGGED number.
            lad["ref_stale_adapted"] = float(W.traverse(
                fm, R_BALL_SEG, q_ev, np.random.default_rng(cfg["seed"] + 7250),
                cfg["sigma_perf"], led, who="instrument", kind="ref_stale_adapted",
                approach_plan=plan_ev)["e_piece"].mean())
            P(f"[ref] ref_stale (STALE FM, the untouched guard) {ref_stale:.4f}  |  "
              f"same anchor under the ADAPTED FM {lad['ref_stale_adapted']:.4f}  (flagged, "
              f"NOT used by the criterion)")

        def arm_sweep(name, lib, mode, base=None, predict=False):
            sw = {}
            base = base_sweep if base is None else base
            for D in [int(x) for x in cfg["delays"]]:
                W.obs_delay = D
                W.obs_predict = bool(predict)
                bcell = base[str(D)]
                cell = {"reactive": bcell["reactive"], "live_seg": bcell["live_seg"]}
                for nm, lv in (("chain", chain_levels), ("seg_tape", {1})):
                    try:
                        oo = W.traverse_route(fm, mk_pol(lv, False, lib, mode), q_ev,
                                              np.random.default_rng(cfg["seed"] + 7710),
                                              cfg["sigma_perf"], led, who="instrument",
                                              kind=f"{name}_{nm}_d{D}", approach_plan=plan_ev)
                        cell[nm] = dict(e=float(oo["e_piece"].mean()),
                                        e_med=float(np.median(oo["e_piece"])),
                                        by_seg=[float(np.median(oo["e_seg"][:, k]))
                                                for k in range(NS)],
                                        fb=float(oo["n_fb"]), planner_free=True)
                    except RuntimeError as ex:
                        cell[nm] = {"error": str(ex)}
                sw[str(D)] = cell
                P(f"[{name}{'+pred' if predict else ''} D={D:2d}] " + "  ".join(
                    f"{q}={cell[q]['e']:.4f}" for q in ("reactive", "live_seg", "seg_tape", "chain")
                    if "e" in cell[q]))
            W.obs_delay = 0
            W.obs_predict = False
            return sw

        def verdict_for(sw):
            """The round-4 rule, verbatim, applied per arm. Nothing here is re-fitted: the anchor
            is the same `ref_stale` and the ordering is the same pre-fixed one."""
            c0 = sw[str(int(cfg["delays"][0]))]
            b0 = min(c0[n]["e"] for n in ("reactive", "live_seg", "chain") if "e" in c0[n])
            v = {"best_at_D0": b0, "guard_ceiling": ref_stale,
                 "guard_ceiling_original_draft": 2.0 * b0, "delta_star": None, "rows": []}
            for D in [int(x) for x in cfg["delays"]]:
                c = sw[str(D)]
                if not all("e" in c[n] for n in ("reactive", "live_seg", "chain")):
                    continue
                er, el, ec = c["reactive"]["e"], c["live_seg"]["e"], c["chain"]["e"]
                ordered = bool(ec <= el <= er)
                playable = bool(min(er, el, ec) <= ref_stale)
                v["rows"].append(dict(delay=D, ms=D * W.dt_ctrl * 1000, e_reactive=er,
                                      e_live_seg=el, e_chain=ec,
                                      e_seg_tape=c["seg_tape"].get("e"),
                                      ordered=ordered, playable=playable,
                                      passes=bool(ordered and playable),
                                      playable_original_draft=bool(min(er, el, ec) <= 2.0 * b0)))
                if ordered and playable and v["delta_star"] is None:
                    v["delta_star"] = D
            return v

        def flat(cell):
            o = []
            for nm in ("seg_tape", "chain"):
                c = cell.get(nm, {})
                if "e" in c:
                    o += [c["e"], c["e_med"], c["fb"]] + list(c["by_seg"])
            return o

        for a in arm_names:
            lk, mode = NAMED[a]
            t0 = snap()
            sw = arm_sweep(a, libs[lk], mode)
            v = verdict_for(sw)
            rec = dict(library=lk, seam_mode=mode, sweep=sw, verdict=v,
                       sweep_cost=cost_since(t0))
            if a == "d0":
                # THE IN-RUN IDENTITY GATE. The ladder's control arm re-runs the stored strategies
                # through the ladder's own code path against round 4's library; with the FM frozen
                # every number must match the round-4 sweep to 0.000e+00 or an additive flag moved
                # something. Under `nest_adapt` the FM has moved on purpose and `d0` is an
                # `audit`-read arm, so the delta IS the model effect: it is reported, not asserted.
                d = max((abs(x - y) for D in [int(z) for z in cfg["delays"]]
                         for x, y in zip(flat(sw[str(D)]), flat(sweep[str(D)]))), default=None)
                rec["identity_vs_round4"] = d
                if adapted:
                    P(f"[identity] `d0` arm vs round 4 under the ADAPTED FM: max|delta| = {d:.3e} "
                      f"(expected nonzero -- the audit read consumes the model)")
                else:
                    P(f"[identity] ladder `d0` arm vs the round-4 sweep: max|delta| = {d:.3e}")
                    if d is None or d != 0.0:
                        raise RuntimeError(
                            f"ladder `d0` arm is NOT bit-identical to the round-4 sweep "
                            f"(max|delta| = {d}); an additive flag changed the donor path")
            if a == "key_d0" and not adapted:
                # `key` reads never touch the forward model, so this arm must be bit-identical to
                # its `d1` twin whatever the FM did. Asserted in-run only where round 4's library
                # and the frozen model make it exactly checkable.
                rec["note"] = "key read: model-independent by construction"
            lad["arms"][a] = rec
            P(f"[arm {a}] lib={lk} seam={mode} Delta* = {v['delta_star']}")
            for r in v["rows"]:
                st = "  n/a" if r["e_seg_tape"] is None else f"{r['e_seg_tape']:.4f}"
                P(f"    D={r['delay']:2d} ({r['ms']:5.0f} ms)  chain {r['e_chain']:.4f} "
                  f"seg_tape {st} | live {r['e_live_seg']:.4f} "
                  f"react {r['e_reactive']:.4f} -> {'PASS' if r['passes'] else 'no'}")
            out["ladder"] = lad
            save()

        # ================================================== ROUND 7b (`d3b`): THE STRONG INCUMBENT
        # The naive delayed controller plans from the state Delta steps ago and does not correct
        # for it, which is a strawman: a nervous system with a reflex delay acts on where its
        # forward model says it now IS, given the commands it has already issued. `../accompanist/presto/`
        # added exactly that (`World.obs_predict`, ported into `world.py` verbatim in round 7b) and
        # measured it removing essentially all of the naive delay penalty on their piece -- which
        # it should, since bridging a Delta-step delay IS a Delta-step FM rollout and Delta = 4-8
        # sits well inside this plant's ~21-step composition horizon.
        #
        # Every strategy is re-swept under the efference-copy operator, not just the incumbent:
        # scoring a predictor-equipped `reactive` against predictor-less stored content would be
        # the mirror strawman. The naive operator is kept as the Round-4 continuity read, both
        # verdicts are reported, and `ref_stale` is untouched in both.
        if cfg.get("dual_obs", False):
            P("[dual] re-sweeping every strategy under the EFFERENCE-COPY operator "
              "(obs_predict): s_hat(t) = FM-rollout(s(t-D), commands issued since)")
            base_pred = base_pair("pred", predict=True)
            lad["base_sweep_predict"] = base_pred
            for D in [int(x) for x in cfg["delays"]]:
                c_, b_ = base_pred[str(D)], base_sweep[str(D)]
                P(f"[dual D={D:2d}] reactive {b_['reactive']['e']:.4f} -> "
                  f"{c_['reactive']['e']:.4f}   live_seg {b_['live_seg']['e']:.4f} -> "
                  f"{c_['live_seg']['e']:.4f}")
            for a in arm_names:
                lk, mode = NAMED[a]
                swp = arm_sweep(a, libs[lk], mode, base=base_pred, predict=True)
                vp = verdict_for(swp)
                lad["arms"][a]["sweep_predict"] = swp
                lad["arms"][a]["verdict_predict"] = vp
                P(f"[arm {a} +pred] Delta* = {vp['delta_star']}")
                for r in vp["rows"]:
                    st = "  n/a" if r["e_seg_tape"] is None else f"{r['e_seg_tape']:.4f}"
                    P(f"    D={r['delay']:2d} ({r['ms']:5.0f} ms)  chain {r['e_chain']:.4f} "
                      f"seg_tape {st} | live {r['e_live_seg']:.4f} react {r['e_reactive']:.4f} "
                      f"-> {'PASS' if r['passes'] else 'no'}")
                out["ladder"] = lad
                save()
            # THE NO-OP GATE. At Delta = 0 the efference copy has nothing to bridge, so every
            # number under both operators must agree to 0.000e+00 or the port moved something.
            if int(cfg["delays"][0]) == 0:
                z = str(int(cfg["delays"][0]))
                worst, n_ = 0.0, 0
                for nm in ("reactive", "live_seg"):
                    for f_ in ("e", "e_med", "fb"):
                        worst = max(worst, abs(base_sweep[z][nm][f_] - base_pred[z][nm][f_])); n_ += 1
                for a in arm_names:
                    for x, y in zip(flat(lad["arms"][a]["sweep"][z]),
                                    flat(lad["arms"][a]["sweep_predict"][z])):
                        worst = max(worst, abs(x - y)); n_ += 1
                lad["obs_predict_noop_at_D0"] = worst
                P(f"[identity] obs_predict is a no-op at Delta = 0: max|delta| = {worst:.3e} "
                  f"over {n_} values")
                if worst != 0.0:
                    raise RuntimeError(f"obs_predict is NOT a no-op at Delta = 0 "
                                       f"(max|delta| = {worst}); the port moved something")

        lad["summary"] = {a: dict(delta_star=lad["arms"][a]["verdict"]["delta_star"],
                                  e0_chain=lad["arms"][a]["sweep"]["0"]["chain"].get("e"),
                                  e0_seg_tape=lad["arms"][a]["sweep"]["0"]["seg_tape"].get("e"),
                                  best_stored_any_delay=min(
                                      [lad["arms"][a]["sweep"][str(D)][nm]["e"]
                                       for D in [int(z) for z in cfg["delays"]]
                                       for nm in ("seg_tape", "chain")
                                       if "e" in lad["arms"][a]["sweep"][str(D)][nm]] or [None]))
                          for a in arm_names}
        P("[ladder] " + json.dumps(lad["summary"]))

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
    # ---- round 5 (`d1`) additions. Every default is the round-4 behaviour: `arms=""` skips the
    # whole ladder, and the rest are only read from inside it. ----
    arms: str = "",
    n_harvest: int = 20,        # post-warm harvest traversals per pool (20 x 16 = 320 rows = d0's)
    n_cand: int = 64,           # legato's `n_cand`
    n_score: int = 96,          # legato's `n_score` (held-out launch states, geometry seed +5100)
    n_pick: int = 4,            # legato's `n_lib` -- committed renditions per cell
    # ---- round 6 (`d2`) additions: legato's sequential nesting. Read only when an arm asks for a
    # `nest*` library; every other path is untouched. Values are legato `L1`'s own. ----
    n_cand_phrase: int = 48,    # legato's `n_cand_phrase` for the chain-span audition
    n_nest: int = 7,            # legato's inter-commit gap (its commits fired c25/32/39)
    nest_window: int = 6,       # legato's `trace_window`
    nest_batch: int = 24,       # legato's `batch` (6 x 24 = its recorded `n_pool` of 144)
    nest_interleave: int = 3,   # legato's `interleave_period` (the diet-rent interleave)
    # ---- round 7 (`d3`): let the forward model ADAPT through the nested practice cycles, as
    # legato's did. Default False = round 6's frozen-model build, bit for bit. ----
    nest_adapt: bool = False,
    # ---- round 7b (`d3b`): also re-sweep every strategy under `../accompanist/presto/`'s efference-copy
    # observation operator, keeping the naive one as the Round-4 continuity read. Default False. ----
    dual_obs: bool = False,

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
        n_harvest = 2; n_cand = 8; n_score = 12; n_pick = 2
        n_cand_phrase = 6; n_nest = 2; nest_window = 2; nest_batch = 8; nest_interleave = 2
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
        arms=arms, n_harvest=n_harvest, n_cand=n_cand, n_score=n_score, n_pick=n_pick,
        n_cand_phrase=n_cand_phrase, n_nest=n_nest, nest_window=nest_window,
        nest_batch=nest_batch, nest_interleave=nest_interleave, nest_adapt=nest_adapt,
        dual_obs=dual_obs, obs_predict=False,
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
