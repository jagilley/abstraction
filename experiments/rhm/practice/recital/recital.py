"""recital -- taking away the era clock: can practice schedule its OWN recitals?

Round 4 of the practice arc on RHM. Round 2 (`../ratchet/`) earned a level-indexed macro
vocabulary; round 3 (`../ear/`) climbed the grader that certifies it and found that what
rescued the deepest era was not a better certificate but COMMITTING ANYWAY AT THE ERA
BOUNDARY -- the two arms that beat `given` are the two that carry a provisional commit. In
both rounds the boundary itself was handed over: a fixed 30-cycle era clock, externally
imposed, telling the agent when to stop practising level k and start practising level k+1.

This round removes that clock. Each arm holds ONE total priced-time budget for the whole run
and decides for itself when to move from era k's damage distribution to era k+1's. Everything
else is `ear` as built, imported rather than copied.

    Q: can any signal INTERNAL to the agent place the curriculum boundaries well -- or does
       the boundary have to come from outside (the teacher's function, the recital date)?

This wires in component (1) of the idea doc's definition -- ALLOCATION, "make a context recur
at a rate fast relative to competence drift" -- as a decision the agent makes rather than a
schedule it is handed. Metering (2) and re-chunking (3) are already earned by rounds 2-3.

THE ARMS. Three fixed-schedule anchors, three self-paced arms, and a small sweep over fixed
boundary placements that buys the "best fixed schedule" reference regret is measured against.

    never_base    base level-1 moves, fixed clock                       (anchor)
    given         the DGP's own tables from cycle 1, fixed clock        (ceiling)
    fixed_climb   `ear`'s `practice_climb`, fixed clock                 (the arm to beat;
                                                                         also the fidelity cell)
    pace_cert     advance when the unit-LP certificate fires
    pace_task     advance when TASK-level progress on the current era flattens
    pace_comp     advance on the certificate, or on a self-set deadline carved from the
                  agent's own total budget -- whichever comes first
    pace_vocab    advance when the agent's own VOCABULARY stops growing: the same detector on
                  its novel-tuple admission rate. The certificate reads commit-readiness of the
                  current candidate and goes quiet once that candidate saturates in entries;
                  the curriculum question is whether there is still vocabulary here to earn,
                  which is a different quantity and is what this arm reads.
    sched_frac    fixed_climb with boundaries at declared FRACTIONS of the total budget
                  (sf1, sf2) -- the reference sweep

THE COMMIT POLICY IS FIXED ACROSS EVERY PRACTICE ARM (`ear`'s `delta_prov`: certify where the
unit-LP certificate can, commit provisionally at the moment of advancement otherwise). The
ADVANCEMENT POLICY is the only thing that differs between the self-paced arms. Detector knobs
(c, c_v, W, hold, lp_min_drop, sil_min_cycle) are bit-identical to `ratchet`/`ear`'s and are
shared by ALL THREE detectors, so the quantity read -- not the detector -- is the variable.

AGENT-LEGAL READOUTS (the advancement decision may read only these):
  * the arm's own metered task error `e` on the CURRENT era's held-out set -- priced feedback
    it is already buying;
  * the unit-LP audition series on the grader cell this arm reads (mfg/policy: contexts the
    agent manufactures from its own solved derivations, graded by its own policy) -- priced;
  * its own MINING STREAM for the level being earned -- n_obs (level-1 spans read off its own
    solved configurations), n_distinct and n_at_support. These are the agent's own counters;
    recall/precision against the DGP's table stay instruments;
  * its own total priced-time budget and the priced time it has spent -- a clock it owns;
  * the number of levels on the ladder (`max_macro_level`), which is task specification and
    was already known to `ear` through `node_at`.
INSTRUMENTS (measured every cycle, logged, NEVER acted on): the rest of the 3x2 grader grid
(including `real`, which needs the oracle), `A_true`, `rand_k`, the `held`/`live` counterfactual,
`context_stats`, the width ladder, the ALL-ERAS probe (it would leak the future ladder), the
plant guard. The all-eras probe is the terminal exam and is unpriced by construction.

ADVANCEMENT IS MONOTONE -- no dropping back down the ladder. Declared for the minimal round.

FIDELITY. With `--t-budget 0` and `--pace cycles` this file reproduces `ear` exactly: the era
loop is the same loop with a different stopping rule, every RNG stream is untouched, and the
only new per-cycle work (the three pacing detectors) consumes no randomness. `rf_s0` runs `ear`'s
own eight arms in `ear`'s own order and must match `er_s0` cycle for cycle. NOTE that torch's
global RNG carries across arms in `ratchet`/`ear`, so arm k's stream depends on how many cycles
arms 1..k-1 ran; reproducing arm 8 therefore requires running all eight, which is what `rf_s0`
does.

Run from experiments/:
  modal run rhm/practice/recital/recital.py::selfcheck_remote
  modal run rhm/practice/recital/recital.py::recital --quick --tag smoke0
  python3 rhm/practice/recital/launch_detached.py --fn recital --tag rc_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.practice.crystallize.units import grade
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    audition_macro, beam_moves, build_ms, context_instances, era_ctx, finetune_generator,
    fit_width, measure_refs, parse_arms, parse_eras, plant_probe, priced, push, value_steps)
from rhm.practice.ear import grader as GR
from rhm.practice.ear.ear import (
    _cfg as ear_cfg, _shared_plus, node_at, selfcheck as ear_selfcheck, write_results)


app = modal.App("rhm-practice-recital", image=image)

REMOTE = "rhm_practice_recital"


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
#   vocab / commit / grader / pay_era / recert -- `ear`'s, unchanged
#   pace -- WHEN THIS ARM ADVANCES THE DEPTH LADDER, the only thing that differs between the
#           self-paced arms:
#             "cycles" the external clock: after `era_cycles` cycles in the era (`ear`'s)
#             "frac"   fixed boundaries at declared fractions (sf1, sf2) of the total budget
#             "cert"   when the unit-LP certificate fires on the level being earned
#             "task"   when the SAME detector goes silent on the metered task error
#             "comp"   "cert", or a self-set deadline at era j's share of the total budget
#             "vocab"  when the SAME detector goes silent on the agent's own novel-tuple
#                      ADMISSION RATE -- i.e. when its vocabulary stops growing, which is the
#                      quantity the audition is blind to once the candidate table saturates
_CLIMB = {"vocab": "earned", "commit": "delta_prov", "grader": "mfg",
          "pay_era": False, "recert": True}
ARMS = {
    # --- `ear`'s eight, verbatim, with the external clock (this is the fidelity set) -------
    "never_base":     {"vocab": "base",   "commit": None,         "grader": None,
                       "pay_era": True,  "recert": False, "pace": "cycles"},
    "given":          {"vocab": "true",   "commit": None,         "grader": None,
                       "pay_era": True,  "recert": False, "pace": "cycles"},
    "practice_gated": {"vocab": "earned", "commit": "delta",      "grader": "era",
                       "pay_era": True,  "recert": False, "pace": "cycles"},
    "practice_late":  {"vocab": "earned", "commit": "late",       "grader": "era",
                       "pay_era": True,  "recert": False, "pace": "cycles"},
    "practice_self":  {"vocab": "earned", "commit": "delta",      "grader": "mfg",
                       "pay_era": False, "recert": False, "pace": "cycles"},
    "taught":         {"vocab": "earned", "commit": "delta",      "grader": "real",
                       "pay_era": False, "recert": False, "pace": "cycles"},
    "practice_prov":  {"vocab": "earned", "commit": "prov",       "grader": None,
                       "pay_era": False, "recert": True,  "pace": "cycles"},
    "practice_climb": {**_CLIMB, "pace": "cycles"},
    # --- this round ------------------------------------------------------------------------
    "fixed_climb":    {**_CLIMB, "pace": "cycles"},   # alias: `practice_climb`, renamed for
                                                      # readability against the paced arms
    "sched_frac":     {**_CLIMB, "pace": "frac"},
    "pace_cert":      {**_CLIMB, "pace": "cert"},
    "pace_task":      {**_CLIMB, "pace": "task"},
    "pace_comp":      {**_CLIMB, "pace": "comp"},
    "pace_vocab":     {**_CLIMB, "pace": "vocab"},
}


def parse_arms_nm(spec):
    """`ratchet.parse_arms`, plus an explicit `nm=` override that names the arm. Lets the
    reference sweep run several `sched_frac` cells in one job without unreadable labels."""
    out = []
    for label, base, ov in parse_arms(spec):
        if "nm" in ov:
            label = str(ov.pop("nm"))
        out.append((label, base, ov))
    return out


# --------------------------------------------------------------------------- #
# the two pacing detectors -- ONE detector, two quantities
# --------------------------------------------------------------------------- #

def new_detector():
    return {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [], "run": 0}


def step_detector(d, A, cfg, c_in_era):
    """`ratchet`'s unit-LP detector, verbatim, factored out so the certificate and the
    task-progress pacer are literally the same code on two different series: EWMA benchmark
    `b`, delta `d = b - A`, silence = window-mean |d| < c*scale AND sd(A) < c_v*scale held
    `sil_hold` cycles, scale = max(ref - min_so_far, b), PLUS the explicit descent
    precondition (ref - min_so_far) >= lp_min_drop and a floor on `c_in_era`.

    Returns (fired, state-dict-for-the-log). Consumes no randomness -- which is what keeps the
    fixed-clock arms bit-identical to `ear`."""
    if A is None:
        return False, {"A": None, "b": d["b"], "run": int(d["run"]),
                       "ref": d["ref"], "emin": d["emin"]}
    if d["ref"] is None:
        d["ref"] = d["b"] = d["emin"] = A
    d["emin"] = min(d["emin"], A)
    delta = d["b"] - A
    scale = max(d["ref"] - d["emin"], d["b"], 1e-6)
    d["hist"].append(A); d["dhist"].append(delta)
    we, wd = d["hist"][-cfg["sil_W"]:], d["dhist"][-cfg["sil_W"]:]
    quiet = (len(we) >= cfg["sil_W"]
             and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
             and float(np.std(we)) < cfg["sil_cv"] * scale)
    d["run"] = d["run"] + 1 if quiet else 0
    d["b"] = d["b"] + cfg["alpha"] * (A - d["b"])
    dropped = (d["ref"] - d["emin"]) >= cfg["lp_min_drop"]
    fired = (d["run"] >= cfg["sil_hold"] and dropped and c_in_era >= cfg["sil_min_cycle"])
    return bool(fired), {"A": A, "b": d["b"], "run": int(d["run"]),
                         "ref": d["ref"], "emin": d["emin"], "drop": d["ref"] - d["emin"]}


def vocab_rate(hist, W):
    """The NOVEL-TUPLE ADMISSION RATE: of the level-1 spans the agent read off its own solved
    configurations in the last `W` cycles, what fraction were tuples it had never seen before.

    `hist` is [(n_distinct, n_obs), ...] for the level being EARNED, one entry per cycle of the
    current era. Both counters are the agent's own mining stream -- no oracle, no true table;
    recall against the DGP's vocabulary stays an instrument.

    Dimensionless and in [0, 1], so it goes through the same detector at the same knobs: it
    starts near 0.5 (early spans are nearly all new), and decays to 0 exactly when the
    vocabulary stops growing -- descend-then-flat, which is what `step_detector` reads.

    The level-l miner observes ONLY while level l is being earned, so its counters are zero at
    the era's first cycle; that virtual zero is the window's origin. Clamping to `hist[0]`
    instead would make the series start at 0 and so never satisfy the detector's `min_drop`
    precondition -- an artifact of the window rather than a property of the mining stream.
    """
    if not hist:
        return None
    D, N = hist[-1]
    j = len(hist) - 1 - W
    D0, N0 = hist[j] if j >= 0 else (0, 0)
    dN = N - N0
    return ((D - D0) / dN) if dN > 0 else 0.0


# --------------------------------------------------------------------------- #
# the arm loop -- `ear`'s, with the era clock replaced by an advancement POLICY
# --------------------------------------------------------------------------- #

def run_arm(label, base, overrides, shared, cfg, eras, refs, outdir, device):
    import torch
    arm = label
    spec = ARMS[base]
    # stamp the ARMS key the arm was instantiated from, so the reducer can price the right
    # grader cell even when `nm=` has renamed the arm (`write_results` is inherited from
    # `ratchet` and takes no extra field, but it does serialise `cfg` verbatim)
    cfg = {**cfg, **overrides, "arm_base": base}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]
    canon_np, bottom_np = shared["canon_np"], shared["bottom_np"]
    inv_bottom = shared["inverse_maps"][-1]
    pace = spec["pace"]
    T = float(cfg["t_budget"])                      # the whole run's priced-time budget
    n_eras = len(eras)

    # ---- per-arm plant + selector (`ear`'s, verbatim) ------------------------------------
    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(),
                             lr=cfg["value_lr_online"] or cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)
    arng = np.random.default_rng(cfg["seed"] + 4242)

    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
    ms = build_ms(base_ms, committed, s, depth, device)
    p_width = cfg["pr_width"]

    def operative(ell):
        if ell == 1:
            return MC.base_table(v)
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    def pol(x, r_np, mset):
        return GR.policy_e(controller, generator, value, x, r_np, mset, rules, rules_t, canon,
                           depth, v, m, s, budget=cfg["budget"], g_budget=cfg["g_budget"],
                           device=device, beam_moves=beam_moves, fit_width=fit_width)

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    bank_x = torch.zeros(0, shared["length"], dtype=torch.long)
    bank_r = torch.zeros(0, dtype=torch.long)
    t_cum = 0.0
    events = []
    log = {"cycle": [], "era": [], "level": [], "t_cum": [], "e": [], "succ": [], "dres": [],
           "n_moves": [], "width": [], "g_per_solve": [], "e_practice": [], "vloss": [],
           "gloss": [], "n_solved": [], "n_mined": [], "miner": [], "aud": [], "cert": [],
           "probe": [], "vocab": [], "clim": [], "gcost": [], "bank": [], "mfg": [],
           "tcert": [], "vcert": [], "c_in_era": []}

    # ---- fixed held-out sets, per era (built for EVERY era up front, exactly as `ear` does,
    #      so the seeds are identical whether or not the arm ever visits the era) -----------
    meter, shadow, realset = {}, {}, {}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        meter[i] = (r_np, torch.from_numpy(x_np))
        r2, x2 = context_instances(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                                   seed=cfg["seed"] + 6100 + 23 * i)
        shadow[i] = (r2, torch.from_numpy(x2).to(device))
        act_i = era["level"] + 1
        if act_i <= cfg["max_macro_level"]:
            nd = node_at(era, act_i, s)
            rr, xr = context_instances(
                rules, {"name": f"L{act_i}n{nd}", "level": act_i, "nodes": [nd]},
                cfg["n_aud"], s, depth, v, m, seed=cfg["seed"] + 800_000 + 31 * i)
            realset[i] = (rr, torch.from_numpy(xr).to(device))

    # ---- the loop. ONE while-loop over cycles; the era index advances when the arm's own
    #      advancement policy says so, and the run stops when the budget is spent. -----------
    cyc, era_i, c_in_era = 0, 0, 0
    era = eras[era_i]
    active = era["level"] + 1
    cert = new_detector()
    tdet = new_detector()
    vdet = new_detector()
    vhist = []                # [(n_distinct, n_obs)] for the level being earned, this era
    mfg = None
    node_next = node_at(era, active, s) if active <= maxl else None
    t_era_start = 0.0
    next_probe_t = (T / cfg["probe_t_n"]) if T > 0 else float("inf")
    print(f"\n----- arm={arm} pace={pace} budget={T:.0f} ERA 1: damage {era['name']} "
          f"(earning level {active}, consumption node {node_next}) -----", flush=True)

    while True:
        cyc += 1
        c_in_era += 1
        counts = {"mat": 0, "ground": 0}
        gcost = {}

        # --- (a) practice ------------------------------------------------------------
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                                       seed=cfg["seed"] + 100_000 + 1000 * cyc)
        out = beam_moves(controller, generator, value, torch.from_numpy(x_np),
                         torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                         budget=cfg["budget"], beam_width=p_width, device=device,
                         collect=True)
        for key in counts:
            counts[key] += out["counts"][key]
        tips = out["tips_x"]
        B, W, Tlen = tips.shape
        tips_flat = tips.reshape(B * W, Tlen)
        succ, _ = grade(tips_flat.cpu().numpy(), np.repeat(r_np, W), rules, s)
        xs = torch.cat([t.reshape(B * W, Tlen).cpu() for t in out["traj"]])
        rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
        ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(out["traj"])))
        push(buf, xs, rs, ys, cfg["buf_cap"])
        solved = tips_flat[torch.from_numpy(succ > 0.5).to(device)]
        ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
        e_practice = 1.0 - float(ps.mean())

        # --- (b) mine ------------------------------------------------------------------
        mine_src = (out["x"][torch.from_numpy(ps > 0.5).to(device)]
                    if cfg["mine_from"] == "chosen" else solved)
        if cfg["mine_cap"] and mine_src.shape[0] > cfg["mine_cap"]:
            sel = torch.from_numpy(rng.permutation(mine_src.shape[0])[:cfg["mine_cap"]])
            mine_src = mine_src[sel.to(device)]
        if mine_src.shape[0]:
            pf = MC.parse_features(shared["reader"], mine_src, s=s).cpu().numpy()
            for ell in range(2, min(maxl, era["level"] + 1) + 1):
                span = s ** (ell - 1)
                node = (era["node"] * s ** (era["level"] - 1)) // span
                miners[ell].observe(pf[:, node * span:(node + 1) * span])

        # --- (b2) bank the agent's own clean derivations -------------------------------
        keep = torch.from_numpy(ps > 0.5)
        if bool(keep.any()):
            bank_x = torch.cat([bank_x, out["x"][keep.to(device)].cpu()])[-cfg["bank_cap"]:]
            bank_r = torch.cat([bank_r, torch.from_numpy(r_np)[keep]])[-cfg["bank_cap"]:]

        # --- (c) the plant learns, then the selector -----------------------------------
        gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                   shared["bottom_map"], v=v, s=s,
                                   n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                   batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                   device=device, rng=grng)
        vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                            n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                            replay_frac=cfg["replay_frac"], device=device, rng=rng)

        # --- (d) metering under PERFORMANCE conditions ----------------------------------
        mr_np, mx = meter[era_i]
        width = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
        b = beam_moves(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                       rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                       beam_width=width, device=device)
        for key in counts:
            counts[key] += b["counts"][key]
        msucc, mdres = grade(b["x"].cpu().numpy(), mr_np, rules, s)
        e = 1.0 - float(msucc.mean())
        gps = b["counts"]["ground"] / mx.shape[0]

        # --- (e) the RATCHET'S shadow audition (era-k contexts, one action) --------------
        sr_np, sx = shadow[era_i]
        aud = {}
        for ell in range(2, maxl + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            cellA = {}
            tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
            cellA["n_entries"] = int(tbl["child"].shape[0])
            if cellA["n_entries"]:
                mv = MC.to_device(MC.make_macro(ell, node, s, tbl), device)
                cellA["cand"] = audition_macro(generator, sx, sr_np, mv, rules_t, canon,
                                               depth, v, m, s, rules)["e"]
                cellA.update({f"tab_{k}": val for k, val in
                              MC.grade_table(tbl, shared["truth"][ell]).items()})
                if ell == active and committed.get(ell) is None:
                    gcost["era"] = {"ground": sx.shape[0], "mat": sx.shape[0]}
                    if spec["pay_era"]:
                        counts["ground"] += sx.shape[0]
                        counts["mat"] += sx.shape[0]
            else:
                cellA["cand"] = None
            mvt = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
            cellA["true"] = audition_macro(generator, sx, sr_np, mvt, rules_t, canon,
                                           depth, v, m, s, rules)["e"]
            if cellA["n_entries"]:
                full = shared["truth"][ell]
                n_all = full["child"].shape[0]
                k = min(cellA["n_entries"], n_all)
                es = []
                for _ in range(3):
                    kp = np.sort(rng.permutation(n_all)[:k])
                    sub = MC.make_table(ell, full["child"][kp], full["lower"], s)
                    mvr = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                    es.append(audition_macro(generator, sx, sr_np, mvr, rules_t, canon,
                                             depth, v, m, s, rules)["e"])
                cellA["rand_k"] = float(np.mean(es))
                cellA["rand_k_sd"] = float(np.std(es))
            if committed[ell] is not None and spec["vocab"] == "earned":
                mvc = MC.to_device(MC.make_macro(ell, node, s, committed[ell]), device)
                cellA["held"] = audition_macro(generator, sx, sr_np, mvc, rules_t, canon,
                                               depth, v, m, s, rules)["e"]
                live = miners[ell].build(MC.base_table(v) if ell == 2
                                         else miners[ell - 1].build(MC.base_table(v),
                                                                    cfg["mine_support"]),
                                         cfg["mine_support"])
                if live["child"].shape[0]:
                    mvl = MC.to_device(MC.make_macro(ell, node, s, live), device)
                    cellA["live"] = audition_macro(generator, sx, sr_np, mvl, rules_t, canon,
                                                   depth, v, m, s, rules)["e"]
                    cellA["live_entries"] = int(live["child"].shape[0])
            aud[str(ell)] = cellA

        # --- (f) the climbed grader: the (ctx x ev) grid at the active level -------------
        clim = {}
        mfg_ev = None
        if active <= maxl:
            cand = miners[active].build(operative(active - 1), cfg["mine_support"])
            has_cand = cand["child"].shape[0] > 0
            ms_cand = (build_ms(base_ms, {**committed, active: cand}, s, depth, device)
                       if has_cand else None)
            mv_cand = (MC.to_device(MC.make_macro(active, node_next, s, cand), device)
                       if has_cand else None)
            mv_true = MC.to_device(
                MC.make_macro(active, node_next, s, shared["truth"][active]), device)

            if mfg is None and bank_x.shape[0] >= cfg["mfg_min_bank"]:
                lower = (operative(active) if cfg["mfg_source"] == "node"
                         else operative(active - 1))
                if int(lower["child"].shape[0]) >= 2:
                    n_use = min(cfg["n_aud"], int(bank_x.shape[0]))
                    pick = arng.permutation(bank_x.shape[0])[:n_use]
                    seeds = bank_x[pick].numpy()
                    sroots = bank_r[pick].numpy()
                    rfeat = MC.parse_features(
                        shared["reader"], torch.from_numpy(seeds).to(device), s=s
                    ).cpu().numpy()
                    dmg, chg = GR.manufacture_damage(
                        seeds, rfeat, lower, active, node_next, s, canon_np, arng,
                        bottom_np=bottom_np, render=cfg["mfg_render"])
                    if dmg is not None:
                        mfg = {"x": torch.from_numpy(dmg).to(device), "r": sroots,
                               "built_cycle": cyc, "n": int(dmg.shape[0]),
                               "lower_entries": int(lower["child"].shape[0]),
                               "child_changed": chg, "build_mat": int(dmg.shape[0])}
                        mfg_ev = {"arm": arm, "era": era_i + 1, "cycle": cyc,
                                  "level": active, "node": node_next, "n": mfg["n"],
                                  "bank": int(bank_x.shape[0]),
                                  "lower_entries": mfg["lower_entries"],
                                  "child_changed": chg,
                                  "stats": GR.context_stats(dmg, sroots, seeds, rules,
                                                            inv_bottom, v, s, active,
                                                            node_next)}
                        print(f"  [mfg] arm={arm} era{era_i+1} c{cyc}: built {mfg['n']} "
                              f"level-{active} contexts from its own derivations "
                              f"(lower table {mfg['lower_entries']} entries) "
                              f"{json.dumps(mfg_ev['stats'], cls=NumpyEncoder)}", flush=True)

            sets = {"era": (sx[:cfg["n_aud"]], sr_np[:cfg["n_aud"]]),
                    "real": ((realset[era_i][1], realset[era_i][0])
                             if era_i in realset else None),
                    "mfg": ((mfg["x"], mfg["r"]) if mfg is not None else None)}
            for key in ("era", "real", "mfg"):
                ctx = sets.get(key)
                if ctx is None:
                    continue
                xk, rk = ctx
                cc = {"n": int(xk.shape[0])}
                cc["act_true"] = audition_macro(generator, xk, rk, mv_true, rules_t, canon,
                                                depth, v, m, s, rules)["e"]
                pb = pol(xk, rk, ms)
                cc["pol_base"] = pb["e"]
                cc["pol_w"] = pb["w"]
                cost = {"ground": pb["counts"]["ground"], "mat": pb["counts"]["mat"]}
                if has_cand:
                    cc["act"] = audition_macro(generator, xk, rk, mv_cand, rules_t, canon,
                                               depth, v, m, s, rules)["e"]
                    pc = pol(xk, rk, ms_cand)
                    cc["pol"] = pc["e"]
                    cc["pol_g"] = pc["g"]
                    cc["margin"] = cc["pol_base"] - cc["pol"]
                    cost["ground"] += pc["counts"]["ground"]
                    cost["mat"] += pc["counts"]["mat"]
                if key == "mfg" and mfg is not None and mfg["built_cycle"] == cyc:
                    cost["mat"] += mfg["build_mat"]
                gcost[key] = cost
                clim[key] = cc
            gr_ = spec["grader"]
            if gr_ in ("mfg", "real") and gr_ in gcost and committed.get(active) is None:
                counts["ground"] += gcost[gr_]["ground"]
                counts["mat"] += gcost[gr_]["mat"]

        # --- (g) the unit-LP certificate, on this arm's grader cell ---------------------
        gr_ = spec["grader"]
        if gr_ == "era":
            A = aud.get(str(active), {}).get("cand")
        elif gr_ in ("mfg", "real"):
            A = clim.get(gr_, {}).get("pol")
        else:
            A = None
        fire_cert = False
        cert_state = {"A": A, "b": cert["b"], "run": int(cert["run"]),
                      "ref": cert["ref"], "emin": cert["emin"]}
        if A is not None and active <= maxl and committed.get(active) is None:
            fire_cert, cert_state = step_detector(cert, A, cfg, c_in_era)

        # --- (g2) THE TASK-PROGRESS PACER: the SAME detector, on the metered task error.
        #          Always stepped (so its series is a logged instrument in every arm); acted
        #          on only where `pace == "task"`.
        fire_task, tdet_state = step_detector(tdet, e, cfg, c_in_era)

        # --- (g2b) THE VOCABULARY PACER: the SAME detector again, on the agent's own
        #           novel-tuple admission rate for the level being earned. Ungated by
        #           `committed` -- unlike the audition, mining continues after a commit, and
        #           the curriculum question ("is there still vocabulary here to earn?") does
        #           not stop being asked. Always stepped, so its series is a logged instrument
        #           in every arm; acted on only where `pace == "vocab"`.
        if active <= maxl:
            mst = miners[active].state()
            vhist.append((mst["n_distinct"], mst["n_obs"]))
            v_rate = vocab_rate(vhist, cfg["sil_W"])
        else:
            v_rate = None
        fire_vocab, vdet_state = step_detector(vdet, v_rate, cfg, c_in_era)

        # --- (g3) ADVANCEMENT: the only thing that differs between the paced arms ---------
        t_now = t_cum + priced(counts, cfg)
        last_era = era_i >= n_eras - 1
        adv_reason = None
        if not last_era:
            if pace == "cycles":
                if c_in_era >= cfg["era_cycles"]:
                    adv_reason = "clock"
            elif pace == "frac":
                fr = cfg["sf1"] if era_i == 0 else cfg["sf2"]
                if T > 0 and t_now >= fr * T:
                    adv_reason = "frac"
            elif pace == "cert":
                if fire_cert:
                    adv_reason = "cert"
            elif pace == "task":
                if fire_task:
                    adv_reason = "task"
            elif pace == "vocab":
                if fire_vocab:
                    adv_reason = "vocab"
            elif pace == "comp":
                if fire_cert:
                    adv_reason = "cert"
                elif T > 0 and t_now >= (era_i + 1) * T / n_eras:
                    adv_reason = "deadline"
        advance = adv_reason is not None

        # --- (h) the commit rule (IDENTICAL across every practice arm) -------------------
        do_commit, prov = False, False
        if spec["commit"] and active <= maxl and committed.get(active) is None:
            at_boundary = advance if pace != "cycles" else (
                c_in_era >= cfg["era_cycles"] - cfg["prov_offset"])
            if spec["commit"] == "delta":
                do_commit = fire_cert
            elif spec["commit"] == "late":
                do_commit = c_in_era >= cfg["era_cycles"] - cfg["late_offset"]
            elif spec["commit"] == "prov":
                do_commit, prov = at_boundary, True
            elif spec["commit"] == "delta_prov":
                do_commit = bool(fire_cert or at_boundary)
                prov = not fire_cert
        if do_commit:
            tbl = miners[active].build(operative(active - 1), cfg["mine_support"])
            if tbl["child"].shape[0] == 0:
                do_commit = False
        if do_commit:
            held = None
            if not prov:
                cr_np, cx_np = context_instances(rules, era_ctx(era), cfg["n_score"], s,
                                                 depth, v, m,
                                                 seed=cfg["seed"] + 700_000 + 1000 * cyc)
                cx = torch.from_numpy(cx_np).to(device)
                mv = MC.to_device(MC.make_macro(active, node_next, s, tbl), device)
                held = audition_macro(generator, cx, cr_np, mv, rules_t, canon, depth,
                                      v, m, s, rules)["e"]
                counts["ground"] += cx.shape[0]; counts["mat"] += cx.shape[0]
            committed[active] = tbl
            ms = build_ms(base_ms, committed, s, depth, device)
            ev = dict(kind="commit", arm=arm, era=era_i + 1, level=active, cycle=cyc,
                      c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg),
                      provisional=bool(prov), audition=held, shadow=A, e_task=e,
                      aud_real_pol=clim.get("real", {}).get("pol"),
                      aud_mfg_pol=clim.get("mfg", {}).get("pol"),
                      aud_era_act=aud.get(str(active), {}).get("cand"),
                      n_entries=int(tbl["child"].shape[0]),
                      n_moves_after=len(ms), sil_run=int(cert["run"]),
                      **{f"tab_{k}": val for k, val in
                         MC.grade_table(tbl, shared["truth"][active]).items()})
            events.append(ev)
            print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} "
                  f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                  f"recall={ev['tab_recall']:.3f} prec={ev['tab_precision']} "
                  f"audition={held} shadow={A} moves={len(ms)}", flush=True)

        # --- (i) the live recert --------------------------------------------------------
        rc_level = era["level"]
        if (spec["recert"] and rc_level >= 2 and committed.get(rc_level) is not None
                and c_in_era % cfg["recert_every"] == 0):
            live = miners[rc_level].build(operative(rc_level - 1), cfg["mine_support"])
            xr = shadow[era_i][1][:cfg["n_aud"]]
            rr_ = shadow[era_i][0][:cfg["n_aud"]]
            pf_ = pol(xr, rr_, ms)
            counts["ground"] += pf_["counts"]["ground"]
            counts["mat"] += pf_["counts"]["mat"]
            rc_cost = {"ground": pf_["counts"]["ground"], "mat": pf_["counts"]["mat"]}
            rec = {"kind": "recert", "arm": arm, "era": era_i + 1, "level": rc_level,
                   "cycle": cyc, "c_in_era": c_in_era, "e_frozen": pf_["e"],
                   "n_frozen": int(committed[rc_level]["child"].shape[0]),
                   "n_live": int(live["child"].shape[0]), "swapped": False}
            if live["child"].shape[0]:
                ms_live = build_ms(base_ms, {**committed, rc_level: live}, s, depth, device)
                pl_ = pol(xr, rr_, ms_live)
                counts["ground"] += pl_["counts"]["ground"]
                counts["mat"] += pl_["counts"]["mat"]
                rc_cost["ground"] += pl_["counts"]["ground"]
                rc_cost["mat"] += pl_["counts"]["mat"]
                rec["e_live"] = pl_["e"]
                rec["n_diff"] = int((pf_["x"] != pl_["x"]).any(1).sum().item())
                rec.update({f"live_{k}": val for k, val in
                            MC.grade_table(live, shared["truth"][rc_level]).items()})
                if pl_["e"] <= pf_["e"] - cfg["recert_margin"]:
                    committed[rc_level] = live
                    ms = build_ms(base_ms, committed, s, depth, device)
                    rec["swapped"] = True
                    print(f"[recert] arm={arm} era{era_i+1} L{rc_level} c{cyc} SWAP "
                          f"{rec['n_frozen']}->{rec['n_live']} entries  "
                          f"e {pf_['e']:.4f} -> {pl_['e']:.4f}", flush=True)
            gcost["recert"] = rc_cost
            events.append(rec)

        # --- book the cycle -------------------------------------------------------------
        t_cum += priced(counts, cfg)
        budget_done = (T > 0 and t_cum >= T)
        clock_done = (T <= 0 and last_era and c_in_era >= cfg["era_cycles"])
        cap_done = cyc >= cfg["max_cycles"]
        stop = budget_done or clock_done or cap_done

        # --- (j) probes: the width ladder, ALL eras' metering sets, the plant guard.
        #         Unpriced instruments; they consume no randomness, so adding the
        #         matched-budget and terminal triggers cannot perturb the fixed-clock arms.
        take_probe = (c_in_era == 1 or cyc % cfg["probe_every"] == 0
                      or (pace == "cycles" and c_in_era == cfg["era_cycles"])
                      or advance or stop or t_cum >= next_probe_t)
        if t_cum >= next_probe_t and T > 0:
            step = T / cfg["probe_t_n"]
            next_probe_t = (np.floor(t_cum / step) + 1) * step
        if take_probe:
            probe = {"cycle": cyc, "era": era_i + 1, "t_cum": t_cum,
                     "ladder": {}, "all_eras": {}}
            for w in cfg["probe_widths"]:
                bb = beam_moves(controller, generator, value, mx, torch.from_numpy(mr_np),
                                ms, rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                                beam_width=w, device=device)
                sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                probe["ladder"][str(w)] = {"e": 1.0 - float(sc.mean()),
                                           "g": bb["counts"]["ground"] / mx.shape[0]}
            for j in range(len(eras)):
                jr, jx = meter[j]
                wj = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                bb = beam_moves(controller, generator, value, jx, torch.from_numpy(jr), ms,
                                rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                                beam_width=wj, device=device)
                sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                probe["all_eras"][str(j)] = 1.0 - float(sc.mean())
            probe["plant"] = plant_probe(generator, shared, cfg, device)
            log["probe"].append(probe)

        log["cycle"].append(cyc); log["era"].append(era_i + 1)
        log["c_in_era"].append(c_in_era)
        log["level"].append(era["level"])
        log["t_cum"].append(t_cum); log["e"].append(e)
        log["succ"].append(float(msucc.mean()))
        log["dres"].append(float(mdres.mean())); log["n_moves"].append(len(ms))
        log["width"].append(width); log["g_per_solve"].append(gps)
        log["e_practice"].append(e_practice); log["vloss"].append(vloss)
        log["gloss"].append(gloss); log["n_solved"].append(int(solved.shape[0]))
        log["n_mined"].append(int(mine_src.shape[0]))
        log["miner"].append({str(ell): miners[ell].state() for ell in range(2, maxl + 1)})
        log["aud"].append(aud)
        log["clim"].append(clim)
        log["gcost"].append(gcost)
        log["bank"].append(int(bank_x.shape[0]))
        if mfg_ev is not None:
            log["mfg"].append(mfg_ev)
        log["cert"].append(cert_state)
        log["tcert"].append(tdet_state)
        log["vcert"].append(vdet_state)
        log["vocab"].append({str(ell): (None if committed[ell] is None
                                        else int(committed[ell]["child"].shape[0]))
                             for ell in range(2, maxl + 1)})
        a_act = aud.get(str(active), {})
        fmt = lambda x: "  .  " if x is None else f"{x:.3f}"
        print(f"[c{cyc:3d}] arm={arm:12s} era{era_i+1} i{c_in_era:3d} t={t_cum:10.0f} "
              f"e={e:.4f} w={width} nm={len(ms)} A={fmt(A)} "
              f"mfg_pol={fmt(clim.get('mfg', {}).get('pol'))} "
              f"real_pol={fmt(clim.get('real', {}).get('pol'))} "
              f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
              f"vr={fmt(v_rate)} sil={cert['run']}/{tdet['run']}/{vdet['run']}"
              + (f"  ADVANCE[{adv_reason}]" if advance else ""), flush=True)
        if cyc % cfg["checkpoint_every"] == 0:
            write_results(outdir, arm, cfg, eras, refs, log, events, complete=False)

        # --- advance the ladder (monotone; never backwards) ------------------------------
        if advance and not stop:
            events.append({"kind": "advance", "arm": arm, "from_era": era_i + 1,
                           "to_era": era_i + 2, "cycle": cyc, "c_in_era": c_in_era,
                           "reason": adv_reason, "t_cum": t_cum,
                           "t_era": t_cum - t_era_start,
                           "frac_of_budget": (t_cum / T) if T > 0 else None,
                           "committed": {str(k): (None if x is None
                                                  else int(x["child"].shape[0]))
                                         for k, x in committed.items()},
                           "cert_run": int(cert["run"]), "cert_drop": (
                               None if cert["ref"] is None else cert["ref"] - cert["emin"]),
                           "task_run": int(tdet["run"]), "task_drop": (
                               None if tdet["ref"] is None else tdet["ref"] - tdet["emin"]),
                           "vocab_run": int(vdet["run"]), "vocab_drop": (
                               None if vdet["ref"] is None else vdet["ref"] - vdet["emin"]),
                           "vocab_rate": v_rate,
                           "e_task": e})
            era_i += 1
            era = eras[era_i]
            active = era["level"] + 1
            node_next = node_at(era, active, s) if active <= maxl else None
            c_in_era = 0
            cert = new_detector()
            tdet = new_detector()
            vdet = new_detector()
            vhist = []
            mfg = None
            t_era_start = t_cum
            print(f"\n----- arm={arm} ADVANCE[{adv_reason}] -> ERA {era_i + 1}: "
                  f"damage {era['name']} (earning level {active}, node {node_next}) "
                  f"at c{cyc}, t={t_cum:.0f} ({100 * t_cum / T if T > 0 else 0:.1f}% of "
                  f"budget) -----", flush=True)
        if stop:
            print(f"----- arm={arm} STOP at c{cyc} t={t_cum:.0f} era={era_i + 1} "
                  f"({'budget' if budget_done else 'clock' if clock_done else 'cap'}) -----",
                  flush=True)
            break

    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True)
    return {"log": log, "events": events}


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = ear_cfg()
    cfg.update(
        # --- the era clock, removed ------------------------------------------------------
        t_budget=7.0e6,      # the WHOLE run's priced-time budget, matched across arms.
                             # <= 0 restores `ear`'s behaviour exactly (era_cycles per era,
                             # stop after the last one) -- the fidelity mode.
        max_cycles=300,      # safety cap; never expected to bind
        sf1=0.33, sf2=0.67,  # `sched_frac`'s declared boundary fractions of the budget
        probe_t_n=12,        # matched-budget probe checkpoints per run
    )
    cfg.update(kw)
    return cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def recital(
    tag: str = "smoke",
    arms: str = "never_base,given,fixed_climb,pace_cert,pace_task,pace_comp",
    eras: str = "1:6,2:3,3:1", seed: int = 0, era_cycles: int = 30, budget: int = 4,
    # the DGP draw and the substrate draw. Defaults are `ratchet`'s, so omitting them
    # reproduces every prior run; a replicate varies all three together (see FILES.md).
    rule_seed: int = 0, train_seed: int = 1,
    t_budget: float = 7.0e6, max_cycles: int = 300, sf1: float = 0.33, sf2: float = 0.67,
    probe_t_n: int = 12,
    pr_width: int = 16, g_budget: int = 58, max_macro_level: int = 3,
    n_pr: int = 64, n_rt: int = 384, n_score: int = 512, n_aud: int = 192,
    mfg_min_bank: int = 32, n_grad: int = 4,
    gen_lr: float = 1e-4, gen_steps: int = 20, mine_support: int = 3,
    mine_from: str = "chosen", mine_cap: int = 8, bank_cap: int = 1024,
    mfg_render: str = "canon", mfg_source: str = "child", recert_every: int = 5,
    recert_margin: float = 0.05,
    prov_offset: int = 0, value_lr_online: float = 3e-5, sil_c: float = 0.06,
    sil_cv: float = 0.15, sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6,
    lp_min_drop: float = 0.10, late_offset: int = 3, probe_every: int = 4,
    plant_holdout: int = 0, holdout_seed: int = 11,
    d_fb: float = 1.0, c_mat: float = 0.05, quick: bool = False,
):
    import torch
    cfg = _cfg(seed=seed, rule_seed=rule_seed, train_seed=train_seed,
               era_cycles=era_cycles, budget=budget, t_budget=t_budget,
               max_cycles=max_cycles, sf1=sf1, sf2=sf2, probe_t_n=probe_t_n,
               pr_width=pr_width, g_budget=g_budget, max_macro_level=max_macro_level,
               n_pr=n_pr, n_rt=n_rt, n_score=n_score, n_aud=n_aud,
               mfg_min_bank=mfg_min_bank, n_grad=n_grad, gen_lr=gen_lr,
               gen_steps=gen_steps, mine_support=mine_support, mine_from=mine_from,
               mine_cap=mine_cap, bank_cap=bank_cap, mfg_render=mfg_render,
               mfg_source=mfg_source, recert_every=recert_every,
               recert_margin=recert_margin, prov_offset=prov_offset,
               value_lr_online=value_lr_online, sil_c=sil_c, sil_cv=sil_cv, sil_W=sil_win,
               sil_hold=sil_hold, sil_min_cycle=sil_min_cycle, lp_min_drop=lp_min_drop,
               late_offset=late_offset, probe_every=probe_every,
               plant_holdout=plant_holdout, holdout_seed=holdout_seed, d_fb=d_fb, c_mat=c_mat)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   era_cycles=6, n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                   checkpoint_every=2, sil_min_cycle=2, late_offset=1, gen_steps=5,
                   bank_cap=256, recert_every=2, mfg_min_bank=16, max_cycles=30,
                   t_budget=(0.0 if t_budget <= 0 else 5.0e5), probe_t_n=6)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"recital tag={tag} arms={arms} eras={ers} t_budget={cfg['t_budget']:.0f} "
          f"device={device}")

    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {shared['truth'][ell]['child'].shape[0]} entries")
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}")

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in parse_arms_nm(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, ers, refs, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "elapsed": time.time() - started}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_pacers(cfg=None):
    """G-P -- the two pacers are ONE detector on two series, and the advancement rules are
    what they say they are. Pure python, no substrate: replays `step_detector` on synthetic
    series whose answers are known by construction.

      P-1  a monotonically descending series never fires (silence must be silence)
      P-2  a series that descends by >= lp_min_drop and then goes flat DOES fire, and not
           before `sil_min_cycle`
      P-3  a series that is flat from the start does NOT fire -- the descent precondition is
           load-bearing, which is the clause that makes LP starvation consequential
      P-4  the certificate and the task pacer are bit-identical on the same input

      P-6  `vocab_rate` is the admission rate it claims to be: it reads the agent's own
           counters with the era's virtual zero as the window origin, is in [0, 1], starts
           high on a fresh vocabulary, and decays to exactly 0 once no new tuple is admitted
      P-7  the vocabulary pacer, on a REAL-shaped mining stream, fires once growth saturates
           (descend-then-flat), never while the table is still filling, and not at all on a
           stream that never grew -- the same three clauses as P-1/P-2/P-3, on the new series
    """
    cfg = cfg or _cfg()
    out = {}

    def run(series):
        d = new_detector()
        for i, a in enumerate(series, start=1):
            fired, _ = step_detector(d, a, cfg, i)
            if fired:
                return i
        return None

    desc = list(np.linspace(0.60, 0.10, 40))
    out["P1_monotone_descent"] = run(desc)
    assert out["P1_monotone_descent"] is None, "P-1 failed: fired mid-descent"

    dropflat = list(np.linspace(0.60, 0.20, 12)) + [0.20] * 20
    out["P2_drop_then_flat"] = run(dropflat)
    assert out["P2_drop_then_flat"] is not None, "P-2 failed: never fired on a flat tail"
    assert out["P2_drop_then_flat"] >= cfg["sil_min_cycle"], "P-2 failed: fired too early"

    flat = [0.30] * 40
    out["P3_flat_from_start"] = run(flat)
    assert out["P3_flat_from_start"] is None, "P-3 failed: fired without any descent"

    rise = [0.20 + 0.004 * i for i in range(40)]
    out["P3b_rising"] = run(rise)
    assert out["P3b_rising"] is None, "P-3b failed: fired on a rising series"

    d1, d2 = new_detector(), new_detector()
    same = True
    for i, a in enumerate(dropflat, start=1):
        f1, s1 = step_detector(d1, a, cfg, i)
        f2, s2 = step_detector(d2, a, cfg, i)
        same &= (f1 == f2) and (s1["run"] == s2["run"]) and (s1["b"] == s2["b"])
    out["P4_one_detector"] = bool(same)
    assert same, "P-4 failed: the two pacers are not the same detector"

    # G-B: the budget carve `pace_comp` uses is a partition of the total
    T, n = 7.0e6, 3
    marks = [(j + 1) * T / n for j in range(n)]
    out["P5_budget_carve"] = marks
    assert abs(marks[-1] - T) < 1e-6, "P-5 failed: the deadline carve does not sum to T"

    # --- P-6: `vocab_rate` is the admission rate it claims to be ------------------------
    W = cfg["sil_W"]
    # a vocabulary that fills to 12 entries and then stops; 8 spans read per cycle
    def stream(n_new_per_cycle, obs=8):
        """[(n_distinct, n_obs)] built from a per-cycle count of NEWLY seen tuples."""
        h, D, N = [], 0, 0
        for k in n_new_per_cycle:
            D += k; N += obs
            h.append((D, N))
        return h

    fresh = stream([4] + [0] * 9)
    assert abs(vocab_rate(fresh[:1], W) - 0.5) < 1e-12, "P-6 failed: wrong origin"
    assert vocab_rate(fresh, W) == 0.0, "P-6 failed: saturated stream did not reach 0"
    allnew = stream([8] * 6)
    assert abs(vocab_rate(allnew, W) - 1.0) < 1e-12, "P-6 failed: not normalised to 1"
    rates = [vocab_rate(fresh[:i + 1], W) for i in range(len(fresh))]
    out["P6_rate_range"] = [float(min(rates)), float(max(rates))]
    assert 0.0 <= min(rates) and max(rates) <= 1.0, "P-6 failed: rate left [0, 1]"
    assert vocab_rate([], W) is None, "P-6 failed: empty history is not None"

    # --- P-7: the vocabulary pacer, on a real-shaped mining stream ----------------------
    def run_vocab(n_new):
        h, d = [], new_detector()
        for i, k in enumerate(n_new, start=1):
            h.append(stream(n_new[:i])[-1])
            fired, _ = step_detector(d, vocab_rate(h, W), cfg, i)
            if fired:
                return i
        return None

    # still filling the table the whole time -- must NOT fire
    out["P7a_still_growing"] = run_vocab([3] * 30)
    assert out["P7a_still_growing"] is None, "P-7a failed: fired while vocabulary grew"
    # fills for 10 cycles, then saturates -- MUST fire, and not before sil_min_cycle
    out["P7b_saturates"] = run_vocab([4, 3, 2, 2, 1, 1, 1, 1, 0, 0] + [0] * 25)
    assert out["P7b_saturates"] is not None, "P-7b failed: never fired after saturation"
    assert out["P7b_saturates"] >= cfg["sil_min_cycle"], "P-7b failed: fired too early"
    # a stream that never admitted anything -- flat at 0 from the start, must NOT fire
    out["P7c_never_grew"] = run_vocab([0] * 30)
    assert out["P7c_never_grew"] is None, "P-7c failed: fired on a vocabulary that never grew"
    return out


def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3, eras="1:6,2:3,3:1"):
    """`ear`'s gates (C-M / C-R / G-D / G-N / G-S / G-R), plus this round's G-P."""
    out = ear_selfcheck(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, n=n,
                        max_level=max_level, eras=eras)
    out["pacers"] = gate_pacers()
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    out = selfcheck()          # `ear`'s gates print themselves
    print("G-P (the pacers): " + json.dumps(out["pacers"], indent=2, cls=NumpyEncoder))
    return out


@app.local_entrypoint()
def main(quick: bool = True):
    recital.remote(tag="smoke0", quick=quick)
