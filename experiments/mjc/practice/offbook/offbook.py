"""offbook -- the port back on the motor substrate: does a committed motor vocabulary consolidate
into the learner, or does it stay an external tape the agent has to look up?

THE QUESTION IN ONE LINE. On mjc every committed unit is an EXTERNAL object -- a keyed tape
(`fingering/`) or a live CEM plan at launch (`legato/`). The RHM side ran the port back and found
that routing (identity's *use*) consolidates into the planner, the corridor (derivable content)
consolidates into the executor, and the table survives as the address book: deleting it costs
current-level performance ~nothing and collapses next-level minability. This node asks the same
question where the units are motor and the model has a composition horizon.

THE MAPPING. The seam is the decision point and the library is the action set. At each re-grounding
point the agent chooses among {library units at each level} union {the live plan}. The enumeration
analogue is an op this node introduces -- SEAM-TIME AUDITION: from the state the body actually
reached, score every candidate's rollout under the current forward model and launch the best. It is
independently licensed (legato G6: a seam is handled AT the seam, or by content selected on realised
seams) and has never been run: `fingering/`'s audition only ever fired at commit time, in the plant,
on recorded score states, and legato's consumption-time selection was a frozen k-means key.

Seam-time audition costs O(K) per seam, so LIBRARY SIZE PAYS RENT -- `tall/`'s widening-action-set
rent, which routing is claimed to dissolve. The two ports:

  * PORT 1, routing. pi(unit-slot | seam state), trained online by self-imitation on the agent's own
    successful traversals; only the top-k proposals are auditioned. O(K) -> O(k).
  * PORT 2, corridor. A span head (slot, seam posture, FM trunk read) -> the unit's measured
    COMMANDS, behind a per-slot parity gate re-checked every cycle. O(members) -> O(1) within a slot.
    Deliberately on the FM trunk: the interference question is the treatment.
  * The address book. The library table stays outside as the mining substrate -- legato F5 measured
    the seed (the segment library manufactures the phrase pool's addressable variation).

TWO CONSTRAINTS THE MJC RUNS IMPOSE ON THE PORT, both respected by construction:
  1. What consolidates must be MEASURED content -- frozen executed traces. `span/` F2/F4: a live
     plan is a model's promise off-distribution, and practice makes that promise worse.
  2. Units must extend BEYOND the composition horizon (~21 steps) or chunks do not pay at all
     (`legato/` F4's crossover). The chain level spans 40-60 steps.

Run:
    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/offbook/offbook.py::offbook --quick
    python3 mjc/practice/offbook/launch_detached.py --fn offbook --tag O1 --seed 0
    python3 mjc/practice/offbook/analyze_offbook.py --tag O1 --fetch
"""

import json
import os

import modal

from mjc.shared import app, volume, DATA_DIR, NumpyEncoder
from mjc.practice.offbook.piece import DEF_WPS, DEF_PATCH_SEG

# `mode`  -- the decision rule at a seam (see `world.RoutePolicy`).
# `ports` -- which heads are BUILT. `wire` builds them and never lets them act (the fidelity arm).
ARM_TABLE = {
    # the reference: reactive MPC throughout, legato's `never`, the donor code path verbatim
    "never":          dict(mode=None, pi=False, span=False),
    # legato's frozen-key op in this node's harness -- continuity with the existing taxonomy
    "key_frozen":     dict(mode="key", pi=False, span=False),
    # seam-time audition over the whole action set: the enumeration reference, priced O(K)
    "audit_all":      dict(mode="audit", pi=False, span=False),
    # PORT 1: pi-gated top-k audition
    "audit_prop_k":   dict(mode="prop", pi=True, span=False),
    # PORT 1 + PORT 2
    "route_native":   dict(mode="native", pi=True, span=True),
    # both ports wired and SHUT: must be bit-identical to `audit_all` for the whole run
    "fid":            dict(mode="audit", pi=True, span=True, wire=True),
    # the in-run form of gate G-P: pi live, k = every legal slot. Bit-identical to `audit_all`.
    "audit_prop_kN":  dict(mode="prop", pi=True, span=False, k_all=True),
    # ROUND 2, the currency arm. Identical to `audit_prop_k` in every respect except WHAT pi's
    # self-imitation counts as a success: O1 filtered targets on raw piece error, and the chain
    # level never earned any trust (mass <= 0.010, executed 0.000). Chains do not pay in meters --
    # they pay in FEEDBACK EVENTS (2/piece against 4), which is the currency legato's steady-state
    # table says the fused chain dominates in. This arm grades pi in that currency. At
    # `pi_fb_rep = 1` it is bit-identical to `audit_prop_k`, which is the cross-tag control gate.
    "prop_priced":    dict(mode="prop", pi=True, span=False, fbq=True),
    # ROUND 3, the rehearsal arm. O2's `prop_priced` (credit still maximal, pi_fb_rep=8) with the
    # launch-level exploration draw MOVED upstream of pi's own gate: a rehearsal draw uniform over
    # every legal slot, launched regardless of what pi proposes. O2 measured that chain trust
    # tracked exposure and only exposure, and that the exposure path was a PRODUCT of two epsilons
    # (`explore_eps` widens the audition set; `eps_act` draws from that set, which is itself
    # pi-gated) -- so the correction sat downstream of the lock-in it was meant to break. This arm
    # makes exposure a first-class treatment. `census/`'s queued "targeted rehearsal of a received
    # vocabulary", instantiated on a plant.
    "prop_rehearse":  dict(mode="prop", pi=True, span=False, fbq=True, reh=True),
}


@app.function(gpu="L4", memory=32768, timeout=43200, volumes={DATA_DIR: volume})
def run_offbook(cfg: dict) -> dict:
    import copy
    import numpy as np
    import torch

    from mjc.embodied import make_arm_goal_sampler, pool_diagnostics, cmd_state_corr
    from mjc.practice.offbook.world import (World, Ledger, Library, RoutePolicy,
                                            start_postures, elite_for, SRC_TAPE, SRC_HEAD, SRC_PRIM)
    from mjc.practice.offbook import nets as N

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = World(cfg, device)
    n, NS = W.n, W.n_seg
    arm = cfg["arm"]
    spec = ARM_TABLE[arm]
    mode = spec["mode"]
    led = Ledger()
    out = {"config": cfg, "arm": arm, "complete": False}
    outdir = os.path.join(DATA_DIR, "practice_offbook", cfg["tag"], arm)
    os.makedirs(outdir, exist_ok=True)

    def P(*a):
        print(f"[{arm}]", *a, flush=True)

    def save():
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(out, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    CALP = {int(k): dict(k_shoot=int(v[0]), cem_iters=int(v[1]), cem_elite=elite_for(int(v[0])))
            for k, v in cfg["calp"].items()}
    commit_seg = [int(c) for c in cfg["commit_seg"]]
    commit_chain = [int(c) for c in cfg["commit_chain"]]
    n_cycles = int(cfg["n_cycles_react"]) if (mode is None and cfg["n_cycles_react"] > 0) \
        else int(cfg["n_cycles"])

    layout = N.SlotLayout(NS, cfg["n_slot"], poison=bool(cfg["n_poison"] > 0))
    P(f"[setup] device={device} mode={mode} slots={layout.n_slots} n_slot={cfg['n_slot']} "
      f"cycles={n_cycles} commit_seg={commit_seg} commit_chain={commit_chain} "
      f"lib_add={cfg['lib_add']} k_prop={cfg['k_prop']} budget={cfg['delib_budget']} "
      f"aud_horizon={cfg['aud_horizon']}")

    # ================================================================= diet + models (legato's)
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
    q_ev = geom(cfg["n_eval"], cfg["seed"] + 5200)

    def app_plan(q, seed):
        s = np.concatenate([q, np.zeros_like(q)], 1).astype(np.float32)
        pf = W.plan_fn(fm_app, W.H_app, vel_pen=cfg.get("vel_pen_mid", 0.0),
                       wp_mask=W.approach_mask())
        return pf(s, np.tile(W.goals[0][None, :], (len(q), 1)),
                  np.random.default_rng(seed))[0]

    plan_rt = app_plan(q_rt, cfg["seed"] + 6000)
    plan_ev = app_plan(q_ev, cfg["seed"] + 6200)
    R_REACT = W.routing("reactive")
    R_BALL_SEG = W.routing("plan_launch", groups=[1] * NS, **CALP[1])
    R_BALL_PHRASE = W.routing("plan_launch", groups=[NS], **CALP[NS])

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

    # THE PLANT GUARD's corridor probe: a FIXED (s, u, s') set off the piece's own corridor, so
    # one-step forward-model error is comparable across arms and across cycles. This is the mjc
    # analogue of RHM's parse/infill staying inert while a head trains into the shared trunk.
    gS = np.concatenate([c[0] for c in corr])[: cfg["guard_n"]]
    gU = np.concatenate([c[1] for c in corr])[: cfg["guard_n"]]
    gS2 = np.concatenate([c[2] for c in corr])[: cfg["guard_n"]]

    def plant_guard(net):
        pred = W.fm_delta(net, gS, gU)
        return float(np.abs(pred - (gS2 - gS)).mean())

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
                        corr_reach=cmd_state_corr(S2_, U2_), pool_diag=pool_diagnostics(Se, Ue, n),
                        guard_fm0=plant_guard(fm0), n_slots=layout.n_slots)
    P(f"[ref] piece @tempo: stale {ref_stale:.4f}  ceiling {ref_ceil:.4f}  react {ref_react:.4f}  "
      f"range {ref_stale - ref_ceil:.4f}  guard0 {out['setup']['guard_fm0']:.5f}")
    save()

    # ================================================================= the ports
    library = Library(W, layout, cfg["n_slot"], cfg["seed"] + 2100)
    pol = RoutePolicy(mode or "audit", W, layout, library,
                      k_prop=(layout.n_slots if spec.get("k_all") else cfg["k_prop"]),
                      eps=cfg["explore_eps"], eps_act=cfg["eps_act"],
                      p_rehearse=(cfg["p_rehearse"] if spec.get("reh") else 0.0),
                      aud_horizon=cfg["aud_horizon"],
                      delib_budget=cfg["delib_budget"],
                      cem_ladder=[int(x) for x in cfg["cem_ladder"]],
                      cem_iters=CALP[1]["cem_iters"], k_shoot=CALP[1]["k_shoot"],
                      prim=bool(cfg["prim"]), span_tau=cfg["span_tau"],
                      span_tol=cfg["span_tol"], span_min_hold=cfg["span_min_hold"],
                      seed=cfg["seed"] + 2200, device=device)
    ptrain = None
    span_opt = None
    if spec.get("pi"):
        pol.pi = N.build_prop(W.SD, NS, layout.n_slots, cfg["seed"] + 2300, device,
                              hidden=cfg["pol_hidden"], layers=cfg["pol_layers"])
        ptrain = N.PropTrainer(pol.pi, layout.n_slots, cfg["pol_lr"], cfg["prop_cap"],
                               cfg["seed"] + 2400, device)
    if spec.get("span"):
        pol.span_head = N.build_span(layout.n_slots, W.SD, W.trunk_dim(), W.H_phrase, W.AD,
                                     cfg["seed"] + 2500, device, hidden=cfg["span_hidden"],
                                     layers=cfg["span_layers"])
        pol.sbuf = N.SpanBuffer(cfg["span_cap"], cfg["span_hold_cap"], cfg["span_hold_frac"],
                                cfg["seed"] + 2600)
        span_opt = torch.optim.Adam(pol.span_head.parameters(), lr=cfg["span_lr"])
    wired_shut = bool(spec.get("wire"))
    if wired_shut:
        # `fid`: both heads exist, neither acts and neither trains. The assertion is that minting
        # them costs the shared stream nothing, so the arm is bit-identical to `audit_all`.
        ptrain = None
        span_opt = None
        pol.pi = None

    # ================================================================= the practice loop
    fm = copy.deepcopy(fm0)
    optf = torch.optim.Adam(fm.parameters(), lr=cfg["adapt_lr"])
    RX, RY = W.tensors(Se, Ue, S2e)
    buf, trace_buf, early_stash = [], [], []
    brng = np.random.default_rng(cfg["seed"] + 800)
    srng = np.random.default_rng(cfg["seed"] + 810)
    lrng = np.random.default_rng(cfg["seed"] + 820)
    events, ladder, parity_log, prop_log, poisoned = [], [], [], [], []
    log = {k: [] for k in ("cycle", "t_cum", "e_rt", "e_practice", "steps_agent", "plans_agent",
                           "delib_agent", "aud_agent", "lib_size", "n_open", "span_loss",
                           "guard_fm", "tape_calls", "head_calls",
                           "pi_kept", "pi_comp", "pi_fb_kept",
                           "pi_cheap_all", "pi_cheap_e", "pi_band_e", "reh_n")}

    def routed():
        return mode is not None and library.any_live()

    def run_piece(fmx, q0, plan0, seed, who, kind, sigma, collect=False, explore=False,
                  capture=False, cycle=0):
        """One traversal. Before anything is committed -- and forever, for `never` -- this is the
        DONOR path with a reactive routing, so every arm runs a bit-identical stream up to its first
        commit (legato's `sched_late` property, and what makes a single seed readable)."""
        if not routed():
            return W.traverse(fmx, R_REACT, q0, np.random.default_rng(seed), sigma, led,
                              who=who, kind=kind, approach_plan=plan0, collect=collect)
        return W.traverse_route(fmx, pol, q0, np.random.default_rng(seed), sigma, led,
                                who=who, kind=kind, approach_plan=plan0, collect=collect,
                                explore=explore, cycle=cycle, capture=capture)

    def ladder_probe(cycle):
        rec = {"cycle": cycle, "t_cum": led.t_priced, "lib": library.sizes(),
               "steps_agent": led.steps["agent"], "plans_agent": led.plans["agent"],
               "delib_agent": led.delib["agent"], "aud_agent": led.aud["agent"]}
        o = run_piece(fm, q_ev, plan_ev, cfg["seed"] + 7000 + cycle, "instrument", "ladder",
                      cfg["sigma_perf"], cycle=cycle)
        rec["e_perf"] = float(np.median(o["e_piece"]))
        rec["e_perf_mean"] = float(o["e_piece"].mean())
        rec["e_perf_by_seg"] = [float(np.median(o["e_seg"][:, k])) for k in range(NS)]
        rec["e_app"] = float(np.median(o["e_app"]))
        rec["n_fb_perf"] = float(o["n_fb"])
        rec["t_piece_perf"] = o["t_piece"]
        rec["mix"] = decision_mix(o)
        for nm, rt in (("react", R_REACT), ("ball_seg", R_BALL_SEG), ("ball_phrase", R_BALL_PHRASE)):
            oo = W.traverse(fm, rt, q_ev, np.random.default_rng(cfg["seed"] + 7050 + cycle),
                            cfg["sigma_perf"], led, who="instrument", kind="ladder",
                            approach_plan=plan_ev)
            rec[f"e_{nm}"] = float(np.median(oo["e_piece"]))
        rec["guard_fm"] = plant_guard(fm)
        return rec

    def decision_mix(o):
        """What the routing actually chose: level mix, source mix, mean auditioned candidates, and
        the poison's share of materialisations -- the readout that separates 'pi starves a bad
        address' from 'enumeration is forced to materialise it at every seam'."""
        if "decisions" not in o or not o["decisions"]:
            return None
        lv, sr, nc, pm, tot, sl = [], [], [], 0, 0, []
        pred, real = [], []
        for d in o["decisions"]:
            lv += d["n_segs"]; sr += d["src"]; nc += d["n_cand"]; sl += d["slot"]
            tot += d["aud"]
            # AUDITION OPTIMISM, per decision: what the model PROMISED the launched candidate would
            # do, against what the body then did over exactly that span. O1 persisted only the mix
            # summaries, so the winner's-curse mechanism had to be inferred from the consequence
            # (the audit_all-minus-audit_prop_k gap grows with the argmax width, corr +0.50, worse
            # in 31/34 probes). Logging it directly makes that test free from here on.
            k0 = int(d["seam"])
            for q, r in enumerate(d["rows"]):
                ns = int(d["n_segs"][q])
                v = float(np.nanmean(o["e_seg"][int(r), k0:k0 + ns]))
                if np.isfinite(v) and np.isfinite(d["score"][q]):
                    pred.append(float(d["score"][q])); real.append(v)
        poison_ids = {layout.slot(ns, k, layout.n_slot)
                      for k in range(NS) for ns in layout.levels(k)} if layout.poison else set()
        return dict(n=len(lv), frac_chain=float(np.mean([x > 1 for x in lv])),
                    frac_tape=float(np.mean([x == SRC_TAPE for x in sr])),
                    frac_head=float(np.mean([x == SRC_HEAD for x in sr])),
                    frac_prim=float(np.mean([x == SRC_PRIM for x in sr])),
                    cand_mean=float(np.mean(nc)), aud_total=int(tot),
                    aud_pred=(float(np.mean(pred)) if pred else None),
                    aud_real=(float(np.mean(real)) if real else None),
                    aud_optimism=(float(np.mean(real) / max(np.mean(pred), 1e-9))
                                  if pred else None),
                    poison_launch=float(np.mean([s in poison_ids for s in sl])),
                    slot_hist={int(s): int(c) for s, c in
                               zip(*np.unique(np.asarray(sl, int), return_counts=True))})

    def harvest(o, cycle):
        """Turn one traversal into library candidates: MEASURED, EXECUTED command tapes with the
        seam state they were launched from and the error they actually achieved. Nothing modelled,
        nothing planned -- `span/` F2/F4's constraint on what may be consolidated."""
        acts, e_seg, lau = o["acts"], o["e_seg"], o["launches"]
        # which ns=1 slot each executed segment belongs to (from the routing that produced it when
        # there was one, else by nearest content centroid) -- the chain pool's SPELLING, which is
        # legato F5's mechanism for why an upper level is addressable at all.
        B = len(acts)
        spell = -np.ones((B, NS), int)
        for d in o.get("decisions", []) or []:
            for r, sid, ns in zip(d["rows"], d["slot"], d["n_segs"]):
                cell = layout.cell_of(int(sid))
                if cell[0] == 1:
                    spell[int(r), int(d["seam"])] = int(cell[2])
        for k in range(NS):
            c1 = library.cell(1, k)
            if c1.cent is not None:
                miss = spell[:, k] < 0
                if miss.any():
                    X = acts[miss, int(W.seg_lo[k]):int(W.seg_hi[k]), :].reshape(int(miss.sum()), -1)
                    spell[miss, k] = ((X[:, None, :] - c1.cent[None, :, :]) ** 2).sum(-1).argmin(1)
        return dict(acts=acts.copy(), e_seg=e_seg.copy(),
                    launches={k: v.copy() for k, v in lau.items()}, spell=spell, cycle=cycle)

    def grow(levels, cycle, reason):
        """One commit: add `lib_add` uniformly-drawn measured tapes per legal cell.

        UNIFORM, never top-of-pool: ranking a candidate by its own realised error is the winner's
        curse (etude E-3b) and favours the rendition most finely tuned to its own start state. The
        library is a STORE, not a shortlist -- what picks among its entries is seam-time audition,
        which scores on states the tape never saw.
        """
        added = {}
        for k in range(NS):
            for ns in layout.levels(k):
                if ns == 1 and 1 not in levels:
                    continue
                if ns > 1 and "chain" not in levels:
                    continue
                rows = []
                for h in trace_buf:
                    if k not in h["launches"]:
                        continue
                    lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k + ns - 1])
                    tp = h["acts"][:, lo:hi, :]
                    ok = np.isfinite(tp).all((1, 2))
                    er = np.nanmean(h["e_seg"][:, k:k + ns], 1)
                    for b in np.nonzero(ok)[0]:
                        rows.append((tp[b], h["launches"][k][b], er[b], h["spell"][b, k:k + ns]))
                if not rows:
                    continue
                pick = lrng.permutation(len(rows))[: int(cfg["lib_add"])]
                library.cell(ns, k).add(
                    np.stack([rows[i][0] for i in pick]),
                    np.stack([rows[i][1] for i in pick]),
                    np.array([rows[i][2] for i in pick], np.float32),
                    cycle, np.stack([rows[i][3] for i in pick]), library.rng)
                added[f"{ns}:{k}"] = int(len(pick))
        # the poison twin: one plausible-but-bad address, pinned to a reserved slot at the first
        # commit. Its members are genuine measured renditions of the right segment harvested from
        # PRE-COMPETENCE cycles -- plausible enough that enumeration must materialise it at every
        # seam, bad enough that a routed policy has every reason to starve it. `native/` finding 6.
        if cfg["n_poison"] > 0 and 1 in levels and not poisoned:
            k = int(cfg["poison_seg"])
            rows = []
            for h in early_stash:
                lo, hi = int(W.seg_lo[k]), int(W.seg_hi[k])
                tp = h["acts"][:, lo:hi, :]
                ok = np.isfinite(tp).all((1, 2))
                er = h["e_seg"][:, k]
                for b in np.nonzero(ok)[0]:
                    rows.append((tp[b], h["launches"][k][b], er[b]))
            if rows:
                order = np.argsort([-r[2] for r in rows])[: int(cfg["n_poison"])]
                library.cell(1, k).add(
                    np.stack([rows[i][0] for i in order]),
                    np.stack([rows[i][1] for i in order]),
                    np.array([rows[i][2] for i in order], np.float32),
                    cycle, -np.ones((len(order), 1), int), library.rng,
                    slot=layout.n_slot)
                added["poison"] = int(len(order))
                poisoned.append(True)
        # `native/prop_net`'s forced-exposure idiom, at commit: for `force_window` cycles every
        # slot of a freshly grown cell is auditioned regardless of pi's (untrained) logit for it.
        for key in added:
            if key == "poison":
                continue
            ns_, k_ = key.split(":")
            pol.force_until[(int(ns_), int(k_))] = cycle + int(cfg["force_window"])
        events.append(dict(kind="commit", cycle=cycle, arm=arm, reason=reason, levels=list(levels),
                           added=added, sizes=library.sizes(), t_cum=led.t_priced))
        P(f"[commit] {reason} c{cycle} added={added} sizes={library.sizes()}")

    def legal_by_seam():
        return {k: pol.legal(k)[0] for k in range(NS)}

    ladder.append(ladder_probe(0))

    for cycle in range(1, n_cycles + 1):
        # ---- (a) practice traversal ----
        qp = geom(cfg["batch"], cfg["seed"] + 9000 + cycle)
        pr = run_piece(fm, qp, app_plan(qp, cfg["seed"] + 9500 + cycle),
                       cfg["seed"] + 10_000 + cycle, "agent", "practice", cfg["sigma_practice"],
                       collect=True, explore=True, cycle=cycle)
        buf.append(pr["trans"])
        h = harvest(pr, cycle)
        trace_buf.append(h)
        if cycle <= int(cfg["early_stash"]):
            early_stash.append(h)
        for b_ in (buf, trace_buf):
            if len(b_) > cfg["trace_window"]:
                b_.pop(0)

        # ---- (b) forward-model plasticity (+ PORT 2's term, into the shared trunk) ----
        PX, PY = W.tensors(np.concatenate([b[0] for b in buf]),
                           np.concatenate([b[1] for b in buf]),
                           np.concatenate([b[2] for b in buf]))
        sfn = ((lambda r: pol.span_terms(W, fm, cfg["span_batch"], r))
               if (span_opt is not None and pol.sbuf is not None) else None)
        sl = W.train_online_span(fm, optf, PX, PY, RX, RY, cfg["n_grad"], brng, cfg["fm_batch"],
                                 cfg["replay_frac"], span_fn=sfn, lam=cfg["span_lam"],
                                 srng=srng, span_opt=span_opt)

        # ---- (c) metering: the at-tempo run-through the agent reads (charged). Port 2's targets
        #          are captured here, on the deployment distribution, and the extra tape audition
        #          that refreshes them is charged like any other audition. ----
        rt = run_piece(fm, q_rt, plan_rt, cfg["seed"] + 4000 + cycle, "agent", "metering",
                       cfg["sigma_perf"], capture=(pol.sbuf is not None), cycle=cycle)
        e = float(np.median(rt["e_piece"]))

        # ---- (d) commits: segments first, chains later, from ROUTED traversals (legato F5) ----
        if cycle in commit_seg:
            grow({1}, cycle, "grow_seg")
            if not pol.norm_set and trace_buf:
                pol.set_norm(np.concatenate([h["launches"][k] for h in trace_buf
                                             for k in range(NS) if k in h["launches"]]))
        if cycle in commit_chain:
            grow({"chain"}, cycle, "grow_chain")

        # ---- (e) PORT 1: self-imitation on the traversals that scored well ----
        # SELECTION BEFORE REGRESSION, and the selection is EXECUTION-GRADED: `e_piece` is the mean
        # of the three waypoint arrivals the BODY actually reached, never the audition's own score.
        # That distinction is load-bearing after gate G-C -- the audition is structurally blind at
        # the chain level, so if credit were assigned by audition score a chain could never earn
        # trust no matter how well it played. Trust here can only be formed by realised use.
        if ptrain is not None and pr.get("decisions"):
            med = float(np.nanmedian(pr["e_piece"]))
            # ROUND 2, THE CURRENCY TERM. O1 filtered pi's targets on raw piece error and the
            # chain level never earned any trust (mass <= 0.010, executed 0.000). Chains do not pay
            # in meters -- they pay in FEEDBACK EVENTS (2/piece against 4), the currency legato's
            # steady-state table says the fused chain dominates in. So the question this arm asks is
            # whether trust forms when pi is graded in the currency chunks actually pay in.
            #
            # FORM: not a second filter but an IMPORTANCE WEIGHT. A quantile filter over `fb` was
            # the first design and was rejected on inspection: cheap traversals are rare (a chain is
            # taken in a few percent of decisions), so the quantile sits on the mass at fb = 4 and
            # nothing is selected -- the dose would silently be zero. A strict `fb == min` filter has
            # the opposite failure: it would drop ~11 of 12 competent traversals per cycle and
            # starve pi of the SEGMENT-level signal it did learn in O1. Replication does neither: the
            # competence stage is O1's verbatim (same bar, same targets, so this is a strict
            # addition), and a traversal that saved feedback events simply counts `pi_fb_rep` times
            # in the buffer. At `pi_fb_rep = 1` the arm is bit-identical to `audit_prop_k`, which is
            # the cross-tag control gate; the dose cannot starve any signal and cannot be silently
            # zero. `d_fb` never enters: every performer plays all 74 control steps, so priced time
            # differs across performers ONLY through `fb`, and ranking by priced time is ranking by
            # `fb` for any positive price -- no meters-per-second exchange rate has to be invented.
            # CHEAP is structural, not a threshold: `fb < 1 + n_seg` means the performer committed
            # at least one unit spanning more than a segment.
            nfb = {}
            for d in pr["decisions"]:
                for r in d["rows"]:
                    nfb[int(r)] = nfb.get(int(r), 1) + 1          # 1 = the approach launch
            rep = int(cfg["pi_fb_rep"]) if spec.get("fbq") else 1
            keep = {int(r) for r in range(len(pr["e_piece"])) if pr["e_piece"][r] <= med}
            cheap = {r for r in keep if nfb.get(r, 1 + NS) < 1 + NS}
            # ROUND-3 INSTRUMENT (added 2026-08-26, after O2's stream came back empty). O2 logged
            # only chain plays that had ALREADY passed the competence band, so when the stream went
            # to zero after the forced-exposure window closed there was no way to tell WHICH of two
            # things happened: chains were played and the band rejected them (the band is
            # mis-specified and the currency treatment never reached stage 2), or chains stopped
            # being played at all (exposure collapsed and the band is innocent). These three fields
            # separate them: how many chain plays occurred BEFORE the band, what they scored, and
            # where the band sat. Pure logging -- no RNG draw moves and no arithmetic changes, so
            # every bit-identity gate still holds.
            cheap_all = [int(r) for r in range(len(pr["e_piece"]))
                         if nfb.get(int(r), 1 + NS) < 1 + NS]
            log["pi_cheap_all"].append(len(cheap_all))
            log["pi_cheap_e"].append(float(np.nanmean([pr["e_piece"][r] for r in cheap_all]))
                                     if cheap_all else None)
            log["pi_band_e"].append(med)
            Z, K, Y = [], [], []
            for d in pr["decisions"]:
                for q, r in enumerate(d["rows"]):
                    if int(r) in keep:
                        w = rep if int(r) in cheap else 1
                        for _ in range(w):
                            Z.append(d["state"][q]); K.append(d["seam"]); Y.append(d["slot"][q])
            log["pi_kept"].append(len(keep)); log["pi_comp"].append(len(cheap))
            log["pi_fb_kept"].append(float(np.mean([nfb.get(r, 1) for r in keep])) if keep else None)
            ptrain.add(Z, K, Y)
            ptrain.train(cfg["prop_steps"], cfg["prop_batch"], legal_by_seam(), pol.norm, NS)

        # ---- (f) PORT 2: parity, re-checked every cycle ----
        if pol.span_head is not None and not wired_shut:
            pv = pol.check_parity(W, fm, led)
            if pv:
                parity_log.append({"cycle": cycle,
                                   "open": int(sum(1 for v in pv.values() if v["open"])),
                                   "n": len(pv),
                                   "frac": {int(k): v["frac"] for k, v in pv.items()}})

        log["cycle"].append(cycle); log["t_cum"].append(led.t_priced)
        log["e_rt"].append(e); log["e_practice"].append(float(np.nanmedian(pr["e_piece"])))
        log["steps_agent"].append(led.steps["agent"]); log["plans_agent"].append(led.plans["agent"])
        log["delib_agent"].append(led.delib["agent"]); log["aud_agent"].append(led.aud["agent"])
        log["lib_size"].append(int(sum(library.sizes().values())))
        log["n_open"].append(int(sum(1 for v in pol.parity.values() if v)))
        log["span_loss"].append(sl)
        log["guard_fm"].append(plant_guard(fm))
        log["tape_calls"].append(pol.tape_calls); log["head_calls"].append(pol.head_calls)
        log["reh_n"].append(int(pr.get("n_reh", 0)))

        if cycle % cfg["probe_every"] == 0 or cycle == n_cycles:
            ladder.append(ladder_probe(cycle))
            L = ladder[-1]
            if pol.pi is not None and trace_buf:
                Zp, Kp = [], []
                for hh in trace_buf[-2:]:
                    for k in range(NS):
                        if k in hh["launches"]:
                            Zp.append(hh["launches"][k]); Kp += [k] * len(hh["launches"][k])
                if Zp:
                    pr_ = N.prop_probe(pol.pi, np.concatenate(Zp), np.array(Kp), legal_by_seam(),
                                       pol.norm, NS, layout.n_slots, layout, device,
                                       spellings=library.spellings())
                    pr_["cycle"] = cycle
                    prop_log.append(pr_)
            P(f"[c{cycle:3d}] e_rt={e:.4f} | perf={L['e_perf']:.4f} "
              f"by-seg {[round(v, 4) for v in L['e_perf_by_seg']]} | react={L['e_react']:.4f} "
              f"| lib={log['lib_size'][-1]} open={log['n_open'][-1]} "
              f"guard={L['guard_fm']:.5f} | t={led.t_priced:9.1f}s")
            out.update(log=log, ladder=ladder, events=events, ledger=led.snapshot(),
                       parity=parity_log, prop=prop_log)
            save()
        elif cycle % 5 == 0:
            P(f"[c{cycle:3d}] e_rt={e:.4f} lib={log['lib_size'][-1]} t={led.t_priced:9.1f}s")

    # ================================================================= the end-of-run battery
    batt = {}
    if mode is not None and library.any_live():
        def battery_probe(tag, seed):
            o = W.traverse_route(fm, pol, q_ev, np.random.default_rng(seed), cfg["sigma_perf"],
                                 led, who="instrument", kind=f"battery_{tag}",
                                 approach_plan=plan_ev, cycle=n_cycles + 1)
            spell = harvest(o, n_cycles + 1)["spell"]
            return dict(e_perf=float(np.median(o["e_piece"])),
                        e_perf_mean=float(o["e_piece"].mean()),
                        by_seg=[float(np.median(o["e_seg"][:, k])) for k in range(NS)],
                        mix=decision_mix(o),
                        minability=library.minability(spell),
                        n_spell=int(len(spell)))
        batt["base"] = battery_probe("base", cfg["seed"] + 8100)
        # (1) primitives ablated: no live plan anywhere -- can the agent play on chunks alone?
        pol.prim = False
        try:
            batt["no_prim"] = battery_probe("no_prim", cfg["seed"] + 8200)
        except RuntimeError as ex:
            batt["no_prim"] = {"error": str(ex)}
        pol.prim = True
        # (2) the table deleted: pi + the span head must serve. Routing-not-pruning predicts the
        #     current level survives and the NEXT level's built entries do not -- while the
        #     observation stream (distinct spellings in the agent's own chosen traversals) does.
        forced = {int(s): True for s in pol.parity}
        if pol.span_head is not None:
            for k in range(NS):
                for ns in layout.levels(k):
                    for j in range(layout.width):
                        forced[layout.slot(ns, k, j)] = True
        keep_parity = dict(pol.parity)
        library.delete()
        if pol.span_head is not None:
            pol.parity = forced           # the battery forces every head open, parity printed beside
        try:
            batt["no_table"] = battery_probe("no_table", cfg["seed"] + 8300)
        except RuntimeError as ex:
            batt["no_table"] = {"error": str(ex)}
        # THE UNCONFOUNDED ADDRESS-BOOK READ. O1's `no_table` cost the routed arms only +0.005 --
        # but with the span head shut, deleting the table simply falls back to the live plan
        # (`frac_prim` went to 1.00), so that number measured the PLANNER's competence, not the
        # table's dispensability. Removing both at once says what is actually left.
        pol.prim = False
        try:
            batt["no_table_no_plan"] = battery_probe("no_table_no_plan", cfg["seed"] + 8350)
        except RuntimeError as ex:
            batt["no_table_no_plan"] = {"error": str(ex)}
        pol.prim = True
        batt["parity_at_deletion"] = {int(k): float(v) for k, v in pol.parity_val.items()}
        pol.parity = keep_parity
        for c in library.cells.values():
            c.deleted = False
        library.deleted = False
        batt["restored"] = battery_probe("restored", cfg["seed"] + 8400)

    post = [x for x, c in zip(log["e_rt"], log["lib_size"]) if c > 0]
    kk = max(1, len(post) // 3)
    out.update(log=log, ladder=ladder, events=events, ledger=led.snapshot(),
               parity=parity_log, prop=prop_log, battery=batt,
               library=dict(sizes=library.sizes(), spellings=library.spellings(),
                            slot_counts={f"{ns}:{k}": [int(x) for x in c.key_n]
                                         for (ns, k), c in library.cells.items()}),
               summary=dict(
                   final_e_perf=ladder[-1]["e_perf"], final_e_react=ladder[-1]["e_react"],
                   final_e_ball_seg=ladder[-1]["e_ball_seg"],
                   final_e_ball_phrase=ladder[-1]["e_ball_phrase"],
                   final_e_perf_by_seg=ladder[-1]["e_perf_by_seg"],
                   final_mix=ladder[-1].get("mix"), final_guard=ladder[-1]["guard_fm"],
                   t_priced=led.t_priced, n_cycles=n_cycles,
                   fb_per_piece_perf=ladder[-1]["n_fb_perf"],
                   tape_calls=pol.tape_calls, head_calls=pol.head_calls,
                   n_open=int(sum(1 for v in pol.parity.values() if v)),
                   post_commit_sd=float(np.std(post)) if post else None,
                   post_commit_drift=(float(np.mean(post[-kk:]) - np.mean(post[:kk]))
                                      if post else None)),
               complete=True)
    save()
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write("ok\n")
    volume.commit()
    P(f"[done] e_perf={out['summary']['final_e_perf']:.4f} t_priced={led.t_priced:.1f}s "
      f"open={out['summary']['n_open']} drift={out['summary']['post_commit_drift']}")
    return out


@app.local_entrypoint()
def offbook(
    quick: bool = False,
    spawn: bool = False,        # detach properly: spawn and exit, do not block the client
    tag: str = "",
    seed: int = 0,
    arms: str = "never,key_frozen,audit_all,audit_prop_k,route_native,fid",
    # ---- the world (legato's calibrated l2 cell, verbatim)
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
    guard_n: int = 4096,
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
    # ---- controllers (legato's l2 cell)
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
    # ---- the library and the two ports
    n_slot: int = 8,
    lib_add: int = 24,
    commit_seg: str = "20,26,32",
    commit_chain: str = "35,38,41",
    n_poison: int = 8,
    poison_seg: int = 0,
    early_stash: int = 3,
    k_prop: int = 2,
    explore_eps: float = 0.15,
    eps_act: float = 0.10,
    p_rehearse: float = 0.0,       # 0 = O2's code path (and the bit-identity gate)
    obs_delay: int = 0,            # ROUND 4: agent-side observation delay, control steps. 0 = O3
    force_window: int = 3,
    pi_fb_rep: int = 1,            # 1 = O1's error-only grading (and the bit-identity gate)
    aud_horizon: int = 0,
    delib_budget: float = 0.0,
    cem_ladder: str = "32,64,128,256,512,1024,2048",
    prim: int = 1,
    pol_hidden: int = 128,
    pol_layers: int = 2,
    pol_lr: float = 1e-3,
    prop_steps: int = 60,
    prop_batch: int = 256,
    prop_cap: int = 40000,
    span_hidden: int = 256,
    span_layers: int = 2,
    span_lr: float = 1e-3,
    span_lam: float = 1.0,
    span_batch: int = 64,
    span_cap: int = 2048,
    span_hold_cap: int = 512,
    span_hold_frac: float = 0.15,
    span_tau: float = 0.9,
    span_tol: float = 0.0,
    span_min_hold: int = 64,
    # ---- practice loop
    n_cycles: int = 120,
    n_cycles_react: int = 60,
    batch: int = 24,
    n_rt: int = 48,
    n_eval: int = 48,
    probe_every: int = 3,
    trace_window: int = 10,
    sigma_practice: float = 0.15,
    sigma_perf: float = 0.06,
    d_fb: float = 0.10,
):
    arm_list = [a for a in arms.split(",") if a]
    for a in arm_list:
        if a not in ARM_TABLE:
            raise ValueError(f"unknown arm {a!r}; known: {sorted(ARM_TABLE)}")
    if quick:
        pool_ou = 1500; pool_reach = 2500; n_corridor = 3
        fm_steps = 1200; fm_steps_boot = 600
        n_cycles = 12; n_cycles_react = 6
        commit_seg = "3,5"; commit_chain = "7,8"
        lib_add = 12; n_slot = 3; n_poison = 4; early_stash = 2; force_window = 2
        batch = 8; n_rt = 12; n_eval = 12; probe_every = 3; trace_window = 4
        k_shoot = 256; cem_iters = 4; calp = "1:256:4,2:256:4,3:256:4"
        cem_ladder = "64,128,256"
        prop_steps = 40; span_batch = 16; span_cap = 256; span_hold_cap = 128
        span_min_hold = 8; guard_n = 512
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
        reach_amp=reach_amp, reach_lo=reach_lo, reach_hi=reach_hi, guard_n=guard_n,
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
        n_slot=n_slot, lib_add=lib_add,
        commit_seg=[int(v) for v in commit_seg.split(",") if v],
        commit_chain=[int(v) for v in commit_chain.split(",") if v],
        n_poison=n_poison, poison_seg=poison_seg, early_stash=early_stash,
        k_prop=k_prop, explore_eps=explore_eps, eps_act=eps_act, p_rehearse=p_rehearse,
        obs_delay=obs_delay, force_window=force_window,
        pi_fb_rep=pi_fb_rep,
        aud_horizon=aud_horizon,
        delib_budget=delib_budget, cem_ladder=[int(x) for x in cem_ladder.split(",") if x],
        prim=prim, pol_hidden=pol_hidden, pol_layers=pol_layers, pol_lr=pol_lr,
        prop_steps=prop_steps, prop_batch=prop_batch, prop_cap=prop_cap,
        span_hidden=span_hidden, span_layers=span_layers, span_lr=span_lr, span_lam=span_lam,
        span_batch=span_batch, span_cap=span_cap, span_hold_cap=span_hold_cap,
        span_hold_frac=span_hold_frac, span_tau=span_tau, span_tol=span_tol,
        span_min_hold=span_min_hold,
        n_cycles=n_cycles, n_cycles_react=n_cycles_react,
        batch=batch, n_rt=n_rt, n_eval=n_eval, probe_every=probe_every,
        trace_window=trace_window, sigma_practice=sigma_practice, sigma_perf=sigma_perf,
        d_fb=d_fb,
    )
    if spawn:
        # THE DETACHED-LAUNCH FIX (2026-08-26). `modal run --detach <file>::<local_entrypoint>` does
        # NOT protect the run: Modal says so in the launch log ("running a local entrypoint in
        # detached mode only keeps the last triggered Modal function alive after the parent process
        # has been killed or disconnected") and `/run-experiment-on-modal` says so in the repo
        # ("running ...::main as a local entrypoint will also result in premature cancellations").
        # A local entrypoint runs on the CLIENT; blocking it in `.map()` for the whole run means the
        # run only lives as long as the launching process does, and when that process is reaped the
        # map's OUTSTANDING inputs are cancelled while completed ones survive. That is exactly the
        # signature both runs died with (O1: 4 arms complete, the 2 still in flight cancelled
        # together; O2: its single arm cancelled 10 minutes in).
        #
        # `.spawn()` makes the entrypoint return in seconds, so there is no long-lived client to
        # reap and the detached app carries the work by itself. CAVEAT, from Modal's own wording:
        # detach keeps "the last triggered Modal function" alive, so a MULTI-arm spawn from one
        # entrypoint is not known to be safe -- a single-arm canary is. For multi-arm rounds, launch
        # one process per arm (or move the fan-out server-side) rather than trusting one spawn loop.
        calls = [(a, run_offbook.spawn({**base, "arm": a})) for a in arm_list]
        for a, c in calls:
            print(f"[spawn] {a}: {c.object_id}")
        print(f"[spawn] {len(calls)} call(s) detached; the client exits now and the app carries on. "
              f"Results land on the volume at /data/practice_offbook/{tag}/<arm>/results.json")
        return
    outs = list(run_offbook.map([{**base, "arm": a} for a in arm_list]))
    localdir = os.path.join(os.path.dirname(__file__), "results", tag)
    os.makedirs(localdir, exist_ok=True)
    for o in outs:
        with open(os.path.join(localdir, f"{o['arm']}.json"), "w") as fh:
            json.dump(o, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote {len(outs)} arm files to {localdir}")
