"""setlist -- same instrument, same theory, different songs requested.

Round 7 of the practice arc on RHM (etude -> crystallize -> ratchet -> ear -> recital -> tall ->
transpose). The demand-drift complement to `transpose`, and the cell the typed-conditioning-gap
reading says is load-bearing.

WHERE THIS COMES FROM. `transpose` installed a live gap by resampling rule cells -- TRUTH-NEWS,
"what is so has changed". The measured answer was that committed chunks cannot consume that
currency: a frozen level-3 table whose precision fell 1.000 -> 0.643 cost at most +0.047 in
audition error and ended at -0.006; a mined table that was 36% illegal still beat the *current*
true table by 0.04-0.06; the recert channel fired 1 swap in 62; and what did move was the dense
learner (plant parse 0.62 -> 0.50, infill 0.71 -> 0.57). A chunk is not a belief -- it has no
truth conditions. It stores DEMAND-CONCENTRATION: a prior over what the world asks, mined from
having been asked.

So the organs are typed by the news they can consume:

    truth-news      ("what is so changed")        -> the dense learner / plant
    interface-news  ("the coupling changed")      -> delta / repair (the mjc bridge arc;
                                                     `transpose`'s level-2 inadmissibility was
                                                     interface-news delivered wholesale)
    demand-news     ("what is asked changed")     -> SELECTION: the vocabulary, the recert,
                                                     the grader.  <-- this round

Recert was never a truth-checker. Its comparison is frozen-table-vs-fresh-table *graded in
consumption*, so under truth-drift both sides are graded against an unchanged demand and tie --
which is exactly the 1/62 that was measured. Under demand-drift the two sides are graded against
a demand that has moved, and the comparison finally has a reason to separate. Whether it does is
the round's question, and it is a question, not a prediction: a null here is the most
informative negative the arc has produced, because it would say the recert has no native
stimulus at all.

THE LEVER (`demand.py`): the grammar is FIXED and the consumption distribution drifts. The
damage process is the demand -- which configurations arrive needing repair -- so drifting the
generative distribution over clean derivations (the root prior over r*, and the mixture weights
on the rule layers ABOVE the mined vocabulary) re-weights which chunks get asked for while every
chunk stays perfectly legal. `corrupt_hier`, the era ladder's (level, node) cells, the true
tables and every grading path are untouched.

ONE LEVER. No grammar drift in this round; `transpose`'s machinery stays available for a
combined cell later.

THE ADMISSIBILITY GATE IS THE COMPLEMENT OF `transpose`'s. There, drift at level 2 was
inadmissible because it degraded the executor: `stale` ran to 0.81-0.95, the exact-DP `floor`
followed from 0.195 to 0.61, and the arms stopped separating within one or two events. Here,
pure demand drift should produce NO WORLD-HARDENING -- the grammar and the difficulty mix are
unchanged, so `d0`, `on_grammar`, `floor` and `stale` should be flat while demand-KL and
coverage move. If that gate holds, any post-commit movement is demand-specific BY CONSTRUCTION:
the decomposition `transpose` had to buy with a paired within-arm control, delivered by the
world itself. It is measured at every epoch and printed before any arm runs.

THE CONCENTRATION-PREMIUM READOUT. `given` hands over the DGP's full table: pure coverage,
demand-invariant by construction (measured at 0.848-0.860 of demand mass at every calibration
cell). The mined arms hold concentration, which is demand-exposed. `ratchet`'s
concentration-beats-coverage margin -- the earned-vs-given fraction -- is therefore exactly the
quantity demand-drift should erode if the typed-gap picture is right. Two oracle arms bracket
it: `demand_frozen` (perfect concentration on epoch-0 demand, never updated) and `demand_live`
(perfect concentration on current demand, re-read every epoch).

Run from experiments/:
  modal run rhm/practice/setlist/setlist.py::selfcheck_remote
  modal run rhm/practice/setlist/setlist.py::cal_demand_remote
  python3 rhm/practice/setlist/launch_detached.py --fn cal_gate --tag cal0 ...
  python3 rhm/practice/setlist/launch_detached.py --fn setlist --tag sl_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.rhm_sculpt_precheck import nearest_derivation_cost
from rhm.practice.crystallize.units import grade, on_grammar_rate
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    audition_macro, beam_moves, build_ms, context_instances, era_ctx, finetune_generator,
    fit_width, measure_refs, parse_eras, plant_probe, priced, push, value_steps)
from rhm.practice.ear import grader as GR
from rhm.practice.ear.ear import _shared_plus, node_at, write_results
from rhm.practice.recital.recital import (
    new_detector, parse_arms_nm, step_detector, vocab_rate)
from rhm.practice.transpose.transpose import (
    ARMS as TRANSPOSE_ARMS, _cfg as transpose_cfg, epoch_refs,
    selfcheck as transpose_selfcheck)
from rhm.practice.transpose import drift as TD
from rhm.practice.setlist import demand as DM


app = modal.App("rhm-practice-setlist", image=image)

REMOTE = "rhm_practice_setlist"


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
#   vocab -- base | true (the DGP's full table: pure COVERAGE, demand-invariant)
#          | dem_frozen (perfect concentration on epoch-0 demand, never updated)
#          | dem_live   (perfect concentration on CURRENT demand, re-read every epoch)
#          | earned
# `given_live` / `true_live` are dropped: nothing becomes false here, so the truth cannot be
# re-read. Its place is taken by the two demand oracles, which is the whole point of the round.
_CLIMB = {"vocab": "earned", "commit": "delta_prov", "grader": "mfg",
          "pay_era": False, "recert": True, "pace": "cycles", "decay": False}
ARMS = {k: dict(v) for k, v in TRANSPOSE_ARMS.items()}
ARMS.update({
    "demand_frozen": {"vocab": "dem_frozen", "commit": None, "grader": None,
                      "pay_era": True, "recert": False, "pace": "cycles", "decay": False},
    "demand_live":   {"vocab": "dem_live", "commit": None, "grader": None,
                      "pay_era": True, "recert": False, "pace": "cycles", "decay": False},
    "climb_decay":   {**_CLIMB, "decay": True},
})


# --------------------------------------------------------------------------- #
# the world: one grammar, a moving audience
# --------------------------------------------------------------------------- #

def build_world(shared, cfg, eras, device, max_cycle, with_refs=True):
    """Every demand epoch the run will pass through, precomputed once at setup.

    Same control as `transpose`: every arm lives in literally the same world at literally the
    same cycle, so a difference between arms cannot be a difference in the draws. What changes
    per epoch is ONLY the distribution the held-out sets are drawn from -- `rules`, `rules_t`
    and `truth` are the same objects at every epoch, which is what makes "nothing becomes
    false" true by construction rather than by care (gate D-2).
    """
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules = shared["rules"]
    rules_t, truth = shared["rules_t"], shared["truth"]
    inv_bottom = shared["inverse_maps"][-1]
    on = cfg["demand_every"] > 0 and cfg["demand_sigma"] > 0
    levels = [int(x) for x in str(cfg["demand_levels"]).split("/")] if on else []
    st, start_stats = None, None
    if on:
        st, start_stats = DM.typical_demand(
            rules, levels, s, depth, v, m, inv_bottom, seed=cfg["demand_seed"],
            kappa=cfg["demand_kappa"], sigma=cfg["demand_sigma"],
            n_cand=cfg["demand_n_cand"], n=2048)
        print(f"[demand] typical start: {json.dumps(start_stats, cls=NumpyEncoder)}",
              flush=True)
    rng = np.random.default_rng(cfg["demand_seed"] + 606)
    n_ep = 1 if not on else 1 + max(0, (max_cycle - cfg["demand_start"])
                                    // cfg["demand_every"] + 1)
    world = []
    for e in range(n_ep):
        if e:
            DM.demand_step(st, rng, 1)
        dem = DM.copy_demand(st) if on else None
        ep = {"epoch": e, "rules": rules, "rules_t": rules_t, "truth": truth, "demand": dem,
              "cycle": (cfg["demand_start"] + (e - 1) * cfg["demand_every"]) if e else 0}
        meter, shadow, realset = {}, {}, {}
        for i, era in enumerate(eras):
            r_np, x_np = ctx(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                             cfg["seed"] + 5000 + 17 * i, dem)
            meter[i] = (r_np, torch.from_numpy(x_np))
            r2, x2 = ctx(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                         cfg["seed"] + 6100 + 23 * i, dem)
            shadow[i] = (r2, torch.from_numpy(x2).to(device))
            act_i = era["level"] + 1
            if act_i <= cfg["max_macro_level"]:
                nd = node_at(era, act_i, s)
                rr, xr = ctx(rules, {"name": f"L{act_i}n{nd}", "level": act_i, "nodes": [nd]},
                             cfg["n_aud"], s, depth, v, m, cfg["seed"] + 800_000 + 31 * i, dem)
                realset[i] = (rr, torch.from_numpy(xr).to(device))
        ep["meter"], ep["shadow"], ep["realset"] = meter, shadow, realset
        ep["probe_clean"] = torch.from_numpy(
            (DM.sample_pool_demand(rules, cfg["n_probe_clean"], s, cfg["seed"] + 31337, dem)
             if on else _sample_pool(rules, cfg["n_probe_clean"], s, cfg["seed"] + 31337))[1])
        # --- the demand itself, and the tables that track it -----------------------------
        hist, dtable = {}, {}
        if on:
            lower = MC.base_table(v)
            for ell in range(2, cfg["max_macro_level"] + 1):
                hist[ell] = DM.demand_hist(rules, dem, ell, s, depth, v, m, inv_bottom,
                                           n=cfg["demand_n"], seed=cfg["demand_seed"] + 5)
                dtable[ell] = DM.demand_table(hist[ell], ell, lower, s, cfg["demand_cover"])
                lower = dtable[ell]
        ep["hist"], ep["dtable"] = hist, dtable
        ep["mag"] = ({} if not on else {
            "kl_from_prev": {str(l): DM.demand_kl(hist[l], world[-1]["hist"][l]) if e else 0.0
                             for l in hist},
            "kl_from_init": {str(l): DM.demand_kl(hist[l], world[0]["hist"][l]) if e else 0.0
                             for l in hist},
            "entropy": {str(l): DM.demand_entropy(hist[l]) for l in hist},
            "true_cover": {str(l): DM.coverage(truth[l], hist[l]) for l in hist},
            "dtable_n": {str(l): int(dtable[l]["child"].shape[0]) for l in dtable},
            "dtable_cover": {str(l): DM.coverage(dtable[l], hist[l]) for l in dtable},
            "init_dtable_cover": {str(l): DM.coverage(world[0]["dtable"][l], hist[l])
                                  for l in dtable} if e else
                                 {str(l): DM.coverage(dtable[l], hist[l]) for l in dtable}})
        if with_refs:
            ep["refs"] = epoch_refs(shared, cfg, eras, device, rules, rules_t, truth,
                                    meter, n_floor=cfg["n_floor"])
            ep["refs"]["d0_check"] = [
                float(nearest_derivation_cost(rules, meter[i][1].numpy(), meter[i][0], s).mean())
                for i in range(len(eras))]
        world.append(ep)
    return world


def ctx(rules, c, n, s, depth, v, m, seed, dem):
    """Draw a damage cell's instances. With no demand state this is the PARENT FUNCTION ITSELF
    (`ratchet.context_instances`), not a reimplementation of it -- which is what makes the
    drift-off replay bit-identical rather than merely equivalent."""
    if dem is None:
        return context_instances(rules, c, n, s, depth, v, m, seed=seed)
    return DM.context_instances_demand(rules, c, n, s, depth, v, m, seed, dem)


def world_summary(world):
    return [{"epoch": e["epoch"], "cycle": e["cycle"], "mag": e["mag"],
             "refs": e.get("refs")} for e in world]


# --------------------------------------------------------------------------- #
# the arm loop -- `transpose`'s, with a moving audience instead of a moving grammar
# --------------------------------------------------------------------------- #

def run_arm(label, base, overrides, shared, cfg, eras, refs, outdir, device, world):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **overrides, "arm_base": base}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    canon = shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]
    canon_np, bottom_np = shared["canon_np"], shared["bottom_np"]
    inv_bottom = shared["inverse_maps"][-1]
    pace = spec["pace"]
    T = float(cfg["t_budget"])
    n_eras = len(eras)
    dem_on = cfg["demand_every"] > 0 and cfg["demand_sigma"] > 0

    cur_ep = 0
    ep = world[0]
    rules, rules_t, truth = ep["rules"], ep["rules_t"], ep["truth"]
    meter, shadow, realset = ep["meter"], ep["shadow"], ep["realset"]
    dem, hist = ep["demand"], ep["hist"]
    hist0 = world[0]["hist"]

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
    dec = cfg["mine_decay"] if spec.get("decay") else 1.0
    miners = {ell: TD.DecayMiner(ell, s, decay=dec, floor=cfg["mine_floor"])
              for ell in range(2, maxl + 1)}
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = truth[ell]
    elif spec["vocab"] in ("dem_frozen", "dem_live"):
        for ell in range(2, maxl + 1):
            committed[ell] = world[0]["dtable"][ell]
    ms = build_ms(base_ms, committed, s, depth, device)
    p_width = cfg["pr_width"]
    prev_entries = {ell: set() for ell in range(2, maxl + 1)}

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
           "tcert": [], "vcert": [], "c_in_era": [],
           "epoch": [], "stale": [], "churn": [], "bank_fresh": []}

    cyc, era_i, c_in_era = 0, 0, 0
    era = eras[era_i]
    active = era["level"] + 1
    cert, tdet, vdet = new_detector(), new_detector(), new_detector()
    vhist = []
    mfg = None
    node_next = node_at(era, active, s) if active <= maxl else None
    t_era_start = 0.0
    next_probe_t = (T / cfg["probe_t_n"]) if T > 0 else float("inf")
    print(f"\n----- arm={arm} vocab={spec['vocab']} decay={dec} ERA 1: damage {era['name']} "
          f"(earning level {active}, consumption node {node_next}) -----", flush=True)

    while True:
        cyc += 1
        c_in_era += 1
        counts = {"mat": 0, "ground": 0}
        gcost = {}

        # --- (a0) THE AUDIENCE CHANGES ITS MIND. Unannounced: nothing below reads `moved`.
        moved = False
        new_ep = 0 if not dem_on else (0 if cyc < cfg["demand_start"] else
                                       1 + (cyc - cfg["demand_start"]) // cfg["demand_every"])
        new_ep = min(new_ep, len(world) - 1)
        if new_ep != cur_ep:
            cur_ep, moved = new_ep, True
            ep = world[cur_ep]
            meter, shadow, realset = ep["meter"], ep["shadow"], ep["realset"]
            dem, hist = ep["demand"], ep["hist"]
            if spec["vocab"] == "dem_live":
                for ell in range(2, maxl + 1):
                    committed[ell] = ep["dtable"][ell]
                ms = build_ms(base_ms, committed, s, depth, device)
            events.append({"kind": "world", "arm": arm, "cycle": cyc, "epoch": cur_ep,
                           "era": era_i + 1, "mag": ep["mag"]})

        # --- (a) practice ------------------------------------------------------------
        r_np, x_np = ctx(rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                         cfg["seed"] + 100_000 + 1000 * cyc, dem)
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
        for ell in range(2, maxl + 1):
            miners[ell].age()
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
                              MC.grade_table(tbl, truth[ell]).items()})
                if dem_on:
                    cellA["cand_cover"] = DM.coverage(tbl, hist[ell])
                if ell == active and committed.get(ell) is None:
                    gcost["era"] = {"ground": sx.shape[0], "mat": sx.shape[0]}
                    if spec["pay_era"]:
                        counts["ground"] += sx.shape[0]
                        counts["mat"] += sx.shape[0]
            else:
                cellA["cand"] = None
            mvt = MC.to_device(MC.make_macro(ell, node, s, truth[ell]), device)
            cellA["true"] = audition_macro(generator, sx, sr_np, mvt, rules_t, canon,
                                           depth, v, m, s, rules)["e"]
            if dem_on and ep["dtable"]:
                mvd = MC.to_device(MC.make_macro(ell, node, s, ep["dtable"][ell]), device)
                cellA["dem"] = audition_macro(generator, sx, sr_np, mvd, rules_t, canon,
                                              depth, v, m, s, rules)["e"]
            if cellA["n_entries"]:
                full = truth[ell]
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
            held_arms = (spec["vocab"] == "earned"
                         or (dem_on and committed[ell] is not None))
            if committed[ell] is not None and held_arms:
                mvc = MC.to_device(MC.make_macro(ell, node, s, committed[ell]), device)
                cellA["held"] = audition_macro(generator, sx, sr_np, mvc, rules_t, canon,
                                               depth, v, m, s, rules)["e"]
                if spec["vocab"] == "earned":
                    live = miners[ell].build(MC.base_table(v) if ell == 2
                                             else miners[ell - 1].build(MC.base_table(v),
                                                                        cfg["mine_support"]),
                                             cfg["mine_support"])
                    if live["child"].shape[0]:
                        mvl = MC.to_device(MC.make_macro(ell, node, s, live), device)
                        cellA["live"] = audition_macro(generator, sx, sr_np, mvl, rules_t,
                                                       canon, depth, v, m, s, rules)["e"]
                        cellA["live_entries"] = int(live["child"].shape[0])
                        if dem_on:
                            cellA["live_cover"] = DM.coverage(live, hist[ell])
            aud[str(ell)] = cellA

        # --- (e2) STALENESS + CHURN, in the currency a chunk is actually denominated in.
        #          `prec_now` is precision against the truth and CANNOT move (nothing becomes
        #          false); `cover_now` is the fraction of the CURRENT demand the committed
        #          table can serve, and is the quantity this round drifts.
        stale_row, churn_row = {}, {}
        if dem_on:
            for ell in range(2, maxl + 1):
                cell = {}
                if committed[ell] is not None:
                    cur = MC.grade_table(committed[ell], truth[ell])
                    cell = {"n": cur["n_learned"], "prec_now": cur["precision"],
                            "recall_now": cur["recall"],
                            "cover_now": DM.coverage(committed[ell], hist[ell]),
                            "cover_init": DM.coverage(committed[ell], hist0[ell])}
                live_t = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                now = TD.entry_set(live_t)
                churn_row[str(ell)] = TD.churn(prev_entries[ell], now)
                churn_row[str(ell)]["cover"] = DM.coverage(live_t, hist[ell])
                prev_entries[ell] = now
                stale_row[str(ell)] = cell

        # --- (f) the climbed grader ------------------------------------------------------
        clim = {}
        if active <= maxl:
            cand = miners[active].build(operative(active - 1), cfg["mine_support"])
            has_cand = cand["child"].shape[0] > 0
            ms_cand = (build_ms(base_ms, {**committed, active: cand}, s, depth, device)
                       if has_cand else None)
            mv_cand = (MC.to_device(MC.make_macro(active, node_next, s, cand), device)
                       if has_cand else None)
            mv_true = MC.to_device(
                MC.make_macro(active, node_next, s, truth[active]), device)

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
                               "built_cycle": cyc, "built_epoch": cur_ep,
                               "n": int(dmg.shape[0]),
                               "lower_entries": int(lower["child"].shape[0]),
                               "child_changed": chg, "build_mat": int(dmg.shape[0])}
                        log["mfg"].append({
                            "arm": arm, "era": era_i + 1, "cycle": cyc, "epoch": cur_ep,
                            "level": active, "node": node_next, "n": mfg["n"],
                            "bank": int(bank_x.shape[0]),
                            "lower_entries": mfg["lower_entries"], "child_changed": chg,
                            "stats": GR.context_stats(dmg, sroots, seeds, rules,
                                                      inv_bottom, v, s, active, node_next)})

            sets = {"era": (sx[:cfg["n_aud"]], sr_np[:cfg["n_aud"]]),
                    "real": ((realset[era_i][1], realset[era_i][0])
                             if era_i in realset else None),
                    "mfg": ((mfg["x"], mfg["r"]) if mfg is not None else None)}
            for key in ("era", "real", "mfg"):
                ctxk = sets.get(key)
                if ctxk is None:
                    continue
                xk, rk = ctxk
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
                if key == "mfg" and mfg is not None:
                    cc["age"] = cyc - mfg["built_cycle"]
                    cc["built_epoch"] = mfg["built_epoch"]
                    if mfg["built_cycle"] == cyc:
                        cost["mat"] += mfg["build_mat"]
                gcost[key] = cost
                clim[key] = cc
            gr_ = spec["grader"]
            if gr_ in ("mfg", "real") and gr_ in gcost and committed.get(active) is None:
                counts["ground"] += gcost[gr_]["ground"]
                counts["mat"] += gcost[gr_]["mat"]

        # --- (g) the unit-LP certificate -------------------------------------------------
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

        fire_task, tdet_state = step_detector(tdet, e, cfg, c_in_era)
        if active <= maxl:
            mst = miners[active].state()
            vhist.append((mst["n_distinct"], mst["n_obs"]))
            v_rate = vocab_rate(vhist, cfg["sil_W"])
        else:
            v_rate = None
        fire_vocab, vdet_state = step_detector(vdet, v_rate, cfg, c_in_era)

        # --- (g3) ADVANCEMENT -------------------------------------------------------------
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

        # --- (h) the commit rule ----------------------------------------------------------
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
                cr_np, cx_np = ctx(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                                   cfg["seed"] + 700_000 + 1000 * cyc, dem)
                cx = torch.from_numpy(cx_np).to(device)
                mv = MC.to_device(MC.make_macro(active, node_next, s, tbl), device)
                held = audition_macro(generator, cx, cr_np, mv, rules_t, canon, depth,
                                      v, m, s, rules)["e"]
                counts["ground"] += cx.shape[0]; counts["mat"] += cx.shape[0]
            committed[active] = tbl
            ms = build_ms(base_ms, committed, s, depth, device)
            ev = dict(kind="commit", arm=arm, era=era_i + 1, level=active, cycle=cyc,
                      c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg), epoch=cur_ep,
                      provisional=bool(prov), audition=held, shadow=A, e_task=e,
                      aud_real_pol=clim.get("real", {}).get("pol"),
                      aud_mfg_pol=clim.get("mfg", {}).get("pol"),
                      aud_era_act=aud.get(str(active), {}).get("cand"),
                      n_entries=int(tbl["child"].shape[0]),
                      cover=(DM.coverage(tbl, hist[active]) if dem_on else None),
                      n_moves_after=len(ms), sil_run=int(cert["run"]),
                      **{f"tab_{k}": val for k, val in
                         MC.grade_table(tbl, truth[active]).items()})
            events.append(ev)
            print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} ep{cur_ep} "
                  f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                  f"recall={ev['tab_recall']:.3f} cover={ev['cover']} "
                  f"audition={held} shadow={A} moves={len(ms)}", flush=True)

        # --- (i) the live recert. Under DEMAND drift this is its native stimulus: both sides
        #         are graded in consumption, and consumption has moved.
        rc_levels = [era["level"]]
        if dem_on and cfg["recert_all"]:
            rc_levels = [l for l in range(2, maxl + 1) if committed.get(l) is not None]
        for rc_level in rc_levels:
            if not (spec["recert"] and rc_level >= 2 and committed.get(rc_level) is not None
                    and c_in_era % cfg["recert_every"] == 0):
                continue
            live = miners[rc_level].build(operative(rc_level - 1), cfg["mine_support"])
            xr = shadow[era_i][1][:cfg["n_aud"]]
            rr_ = shadow[era_i][0][:cfg["n_aud"]]
            pf_ = pol(xr, rr_, ms)
            counts["ground"] += pf_["counts"]["ground"]
            counts["mat"] += pf_["counts"]["mat"]
            rc_cost = {"ground": pf_["counts"]["ground"], "mat": pf_["counts"]["mat"]}
            rec = {"kind": "recert", "arm": arm, "era": era_i + 1, "level": rc_level,
                   "cycle": cyc, "c_in_era": c_in_era, "epoch": cur_ep,
                   "e_frozen": pf_["e"],
                   "n_frozen": int(committed[rc_level]["child"].shape[0]),
                   "n_live": int(live["child"].shape[0]), "swapped": False,
                   "froz_cover": (DM.coverage(committed[rc_level], hist[rc_level])
                                  if dem_on else None)}
            if live["child"].shape[0]:
                ms_live = build_ms(base_ms, {**committed, rc_level: live}, s, depth, device)
                pl_ = pol(xr, rr_, ms_live)
                counts["ground"] += pl_["counts"]["ground"]
                counts["mat"] += pl_["counts"]["mat"]
                rc_cost["ground"] += pl_["counts"]["ground"]
                rc_cost["mat"] += pl_["counts"]["mat"]
                rec["e_live"] = pl_["e"]
                rec["n_diff"] = int((pf_["x"] != pl_["x"]).any(1).sum().item())
                rec["live_cover"] = DM.coverage(live, hist[rc_level]) if dem_on else None
                rec.update({f"live_{k}": val for k, val in
                            MC.grade_table(live, truth[rc_level]).items()})
                if pl_["e"] <= pf_["e"] - cfg["recert_margin"]:
                    committed[rc_level] = live
                    ms = build_ms(base_ms, committed, s, depth, device)
                    rec["swapped"] = True
                    print(f"[recert] arm={arm} era{era_i+1} L{rc_level} c{cyc} SWAP "
                          f"{rec['n_frozen']}->{rec['n_live']} entries  "
                          f"e {pf_['e']:.4f} -> {pl_['e']:.4f}  cover "
                          f"{rec['froz_cover']} -> {rec['live_cover']}", flush=True)
            gcost.setdefault("recert", {"ground": 0, "mat": 0})
            gcost["recert"]["ground"] += rc_cost["ground"]
            gcost["recert"]["mat"] += rc_cost["mat"]
            events.append(rec)

        # --- book the cycle ---------------------------------------------------------------
        t_cum += priced(counts, cfg)
        budget_done = (T > 0 and t_cum >= T)
        clock_done = (T <= 0 and last_era and c_in_era >= cfg["era_cycles"])
        cap_done = cyc >= cfg["max_cycles"]
        stop = budget_done or clock_done or cap_done

        take_probe = (c_in_era == 1 or cyc % cfg["probe_every"] == 0
                      or (pace == "cycles" and c_in_era == cfg["era_cycles"])
                      or advance or stop or t_cum >= next_probe_t)
        if t_cum >= next_probe_t and T > 0:
            step = T / cfg["probe_t_n"]
            next_probe_t = (np.floor(t_cum / step) + 1) * step
        if take_probe:
            probe = {"cycle": cyc, "era": era_i + 1, "t_cum": t_cum, "epoch": cur_ep,
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
            probe["plant"] = plant_probe(generator, {**shared, "probe_clean":
                                                     ep["probe_clean"]}, cfg, device)
            log["probe"].append(probe)

        bank_fresh = None
        if dem_on and bank_x.shape[0]:
            n_bf = min(256, int(bank_x.shape[0]))
            bf = MC.exact_features(bank_x[-n_bf:].numpy(), inv_bottom, v, s)
            span = s ** (max(2, era["level"]) - 1)
            n_nodes = bf.shape[1] // span
            sp = bf[:, :n_nodes * span].reshape(-1, span)
            h = hist[max(2, min(era["level"], maxl))]
            bank_fresh = float(np.mean([h.get(tuple(int(x) for x in row), 0.0) > 0
                                        for row in sp]))

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
        log["cert"].append(cert_state)
        log["tcert"].append(tdet_state)
        log["vcert"].append(vdet_state)
        log["epoch"].append(cur_ep)
        log["stale"].append(stale_row)
        log["churn"].append(churn_row)
        log["bank_fresh"].append(bank_fresh)
        log["vocab"].append({str(ell): (None if committed[ell] is None
                                        else int(committed[ell]["child"].shape[0]))
                             for ell in range(2, maxl + 1)})
        a_act = aud.get(str(active), {})
        fmt = lambda x: "  .  " if x is None else f"{x:.3f}"
        st2 = stale_row.get(str(max(2, min(era["level"], maxl))), {}) or {}
        print(f"[c{cyc:3d}] arm={arm:14s} era{era_i+1} i{c_in_era:3d} ep{cur_ep:2d} "
              f"t={t_cum:10.0f} e={e:.4f} w={width} nm={len(ms)} A={fmt(A)} "
              f"held={fmt(a_act.get('held'))} true={fmt(a_act.get('true'))} "
              f"dem={fmt(a_act.get('dem'))} ent={a_act.get('n_entries')} "
              f"cov_now={fmt(st2.get('cover_now'))} "
              f"sil={cert['run']}/{tdet['run']}/{vdet['run']}"
              + ("  DEMAND-MOVED" if moved else "")
              + (f"  ADVANCE[{adv_reason}]" if advance else ""), flush=True)
        if cyc % cfg["checkpoint_every"] == 0:
            write_results(outdir, arm, cfg, eras, refs, log, events, complete=False)

        if advance and not stop:
            events.append({"kind": "advance", "arm": arm, "from_era": era_i + 1,
                           "to_era": era_i + 2, "cycle": cyc, "c_in_era": c_in_era,
                           "reason": adv_reason, "t_cum": t_cum, "epoch": cur_ep,
                           "t_era": t_cum - t_era_start, "e_task": e})
            era_i += 1
            era = eras[era_i]
            active = era["level"] + 1
            node_next = node_at(era, active, s) if active <= maxl else None
            c_in_era = 0
            cert, tdet, vdet = new_detector(), new_detector(), new_detector()
            vhist = []
            mfg = None
            t_era_start = t_cum
            print(f"\n----- arm={arm} ADVANCE[{adv_reason}] -> ERA {era_i + 1}: "
                  f"damage {era['name']} (earning level {active}, node {node_next}) "
                  f"at c{cyc} -----", flush=True)
        if stop:
            print(f"----- arm={arm} STOP at c{cyc} t={t_cum:.0f} era={era_i + 1} -----",
                  flush=True)
            break

    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True)
    return {"log": log, "events": events}


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = transpose_cfg()
    cfg.update(
        drift_every=0,           # `transpose`'s grammar drift stays OFF in this round
        demand_every=0,          # cycles between demand events; 0 == OFF == the parent exactly
        demand_sigma=2.5,        # CONCENTRATION of demand (OU logit scale)
        demand_kappa=0.15,       # MIXING RATE of demand (OU mean reversion)
        demand_levels="0/1",     # rule layers that drift, root-ward; 0/1 = the two ABOVE the
                                 # mined vocabulary. See `demand.py` for why level 2 is a null.
        demand_start=6,
        demand_seed=0,
        demand_n=8192,           # derivations per demand-histogram estimate
        demand_cover=0.8,        # the demand mass the oracle concentration tables cover
        demand_n_cand=48,        # stationary candidates the representative start is chosen from
        n_floor=256,             # instances the per-epoch exact-DP floor is measured on
        t_budget=0.0,
    )
    cfg.update(kw)
    return cfg


def _world_gate(world, eras, cfg):
    """Print the round's admissibility gate BEFORE any arm runs: demand must move while the
    world stays exactly as hard. The complement of `transpose`'s gate, which failed."""
    print("\n===== D-6 ADMISSIBILITY GATE: does demand move while difficulty does not? =====")
    print(f"{'ep':>3} {'cyc':>4} | {'KL_prev':>8} {'KL_init':>8} {'H':>6} {'trueCov':>8} "
          f"{'demCov0':>8} | per era:  d0 / on_gram / stale / floor")
    for w in world:
        mg = w["mag"] or {}
        r = w.get("refs") or {}
        cells = " ".join(
            f"{r['d0'][i]:.2f}/{r['on_grammar'][i]:.2f}/{r['stale'][i]:.2f}/{r['floor'][i]:.2f}"
            for i in range(len(eras))) if r else ""
        print(f"{w['epoch']:>3} {w['cycle']:>4} | "
              f"{mg.get('kl_from_prev', {}).get('2', 0):>8.4f} "
              f"{mg.get('kl_from_init', {}).get('2', 0):>8.4f} "
              f"{mg.get('entropy', {}).get('2', 0):>6.3f} "
              f"{mg.get('true_cover', {}).get('2', 0):>8.3f} "
              f"{mg.get('init_dtable_cover', {}).get('2', 0):>8.3f} | {cells}")
    if len(world) > 1:
        r0, r1 = world[0].get("refs"), world[-1].get("refs")
        if r0 and r1:
            print("  gate deltas (last - first):  " + "  ".join(
                f"{eras[i]['name']} d0 {r1['d0'][i] - r0['d0'][i]:+.3f} "
                f"stale {r1['stale'][i] - r0['stale'][i]:+.3f} "
                f"floor {r1['floor'][i] - r0['floor'][i]:+.3f}" for i in range(len(eras))))
    print("=" * 78, flush=True)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def setlist(
    tag: str = "smoke",
    arms: str = ("never_base,given,demand_frozen,demand_live,practice_gated,practice_prov,"
                 "prov_norecert,practice_climb,climb_decay"),
    eras: str = "1:6,2:3,3:1", seed: int = 0, era_cycles: int = 30, budget: int = 4,
    rule_seed: int = 0, train_seed: int = 1,
    demand_every: int = 4, demand_sigma: float = 2.5, demand_kappa: float = 0.15,
    demand_levels: str = "0/1", demand_start: int = 6, demand_seed: int = 0,
    demand_n: int = 8192, demand_cover: float = 0.8, demand_n_cand: int = 48,
    n_floor: int = 256, mine_decay: float = 0.95, mine_floor: float = 0.5, recert_all: bool = True,
    t_budget: float = 0.0, max_cycles: int = 300, probe_t_n: int = 12,
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
               max_cycles=max_cycles, probe_t_n=probe_t_n,
               demand_every=demand_every, demand_sigma=demand_sigma,
               demand_kappa=demand_kappa, demand_levels=demand_levels,
               demand_start=demand_start, demand_seed=demand_seed, demand_n=demand_n,
               demand_cover=demand_cover, demand_n_cand=demand_n_cand, n_floor=n_floor,
               mine_decay=mine_decay, mine_floor=mine_floor, recert_all=recert_all,
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
                   demand_start=3, demand_n=2048,
                   demand_every=(cfg["demand_every"] if cfg["demand_every"] <= 0 else 3))
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    max_cycle = (cfg["era_cycles"] * len(ers) if cfg["t_budget"] <= 0 else cfg["max_cycles"])
    print(f"setlist tag={tag} arms={arms} eras={ers} demand=(sigma {cfg['demand_sigma']}, "
          f"kappa {cfg['demand_kappa']}, layers {cfg['demand_levels']}, every "
          f"{cfg['demand_every']} cycles from c{cfg['demand_start']}) device={device}")
    assert cfg["drift_every"] == 0, "one lever: no grammar drift in this round"

    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}")

    t0 = time.time()
    world = build_world(shared, cfg, ers, device, max_cycle)
    print(f"[world] {len(world)} demand epochs built in {time.time() - t0:.0f}s")
    _world_gate(world, ers, cfg)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs,
                   "world": world_summary(world)}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in parse_arms_nm(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, ers, refs, outdir, device,
                                 world)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "elapsed": time.time() - started}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def cal_gate(
    tag: str = "cal0", specs: str = "0/0.0/0.0,4/2.5/0.15,4/1.0/0.05",
    cycles: int = 40, eras: str = "1:6", seed: int = 0,
    arms: str = "never_base,given,practice_climb,climb_decay",
    mine_decay: float = 0.95, quick: bool = False,
):
    """THE ADMISSIBILITY GATE + DESCENT CHECK, at a ladder of demand specs
    (`period/sigma/kappa`), with an in-run no-drift control so no cross-run inference is needed.

    `transpose`'s lesson, applied in the opposite direction. There the risk was that the world
    got harder and the arms stopped separating; here the world should NOT get harder, and the
    risk is the mirror image -- that demand moves and nothing downstream notices. So the gate
    reads both halves: `d0`/`on_grammar`/`stale`/`floor` must stay flat (no world-hardening)
    while demand-KL and the coverage of a frozen concentration table must move.
    """
    import torch
    cfg = _cfg(seed=seed, era_cycles=cycles, t_budget=0.0, max_cycles=cycles + 5,
               demand_start=6, mine_decay=mine_decay, probe_every=8)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   era_cycles=8, n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                   checkpoint_every=2, sil_min_cycle=2, gen_steps=5, bank_cap=256,
                   recert_every=2, mfg_min_bank=16, max_cycles=12, demand_start=3,
                   demand_n=2048)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}", flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    summary = {}
    for spec in specs.split(","):
        per, sig, kap = spec.strip().split("/")
        name = f"p{per}s{sig}k{kap}"
        rcfg = {**cfg, "demand_every": int(per), "demand_sigma": float(sig),
                "demand_kappa": float(kap)}
        world = build_world(shared, rcfg, ers, device, rcfg["era_cycles"])
        print(f"\n##### demand {name}: {len(world)} epochs #####", flush=True)
        _world_gate(world, ers, rcfg)
        sub = os.path.join(outdir, name)
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "setup.json"), "w") as fh:
            json.dump({"config": rcfg, "eras": ers, "refs": refs,
                       "world": world_summary(world)}, fh, indent=2, cls=NumpyEncoder)
        for label, base, ov in parse_arms_nm(arms):
            print(f"\n===== demand {name} arm {label} =====", flush=True)
            r = run_arm(label, base, ov, shared, rcfg, ers, refs, sub, device, world)
            log = r["log"]
            eps = np.asarray(log["epoch"])
            st = np.asarray([world[int(k)]["refs"]["stale"][0] for k in eps])
            fl = np.asarray([world[int(k)]["refs"]["floor"][0] for k in eps])
            ev = np.asarray(log["e"])
            rec = (st - ev) / np.maximum(st - fl, 1e-9)
            summary[f"{name}/{label}"] = {
                "e_first5": float(ev[:5].mean()), "e_last5": float(ev[-5:].mean()),
                "recovered_first5": float(rec[:5].mean()),
                "recovered_last5": float(rec[-5:].mean()),
                "stale_first": float(st[0]), "stale_last": float(st[-1]),
                "floor_first": float(fl[0]), "floor_last": float(fl[-1]),
                "recerts": sum(1 for e in r["events"] if e["kind"] == "recert"),
                "recert_swaps": sum(1 for e in r["events"]
                                    if e["kind"] == "recert" and e["swapped"])}
            print(f"  -> {json.dumps(summary[f'{name}/{label}'], cls=NumpyEncoder)}",
                  flush=True)
        volume.commit()
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print("\n=== GATE + DESCENT ===")
    print(f"{'cell':34s} {'e_first5':>9s} {'e_last5':>8s} {'rec_f5':>8s} {'rec_l5':>8s} "
          f"{'stale0':>7s} {'stale1':>7s} {'floor0':>7s} {'floor1':>7s} {'swaps':>6s}")
    for k, x in summary.items():
        print(f"{k:34s} {x['e_first5']:>9.3f} {x['e_last5']:>8.3f} "
              f"{x['recovered_first5']:>8.3f} {x['recovered_last5']:>8.3f} "
              f"{x['stale_first']:>7.3f} {x['stale_last']:>7.3f} "
              f"{x['floor_first']:>7.3f} {x['floor_last']:>7.3f} "
              f"{x['recert_swaps']:>3d}/{x['recerts']:<2d}")
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return summary


# --------------------------------------------------------------------------- #
# offline calibration + gates
# --------------------------------------------------------------------------- #

def cal_demand(v=8, s=2, depth=4, m=2, rule_seed=0, levels=(0, 1), cycles=90, period=4,
               start=6, sigmas=(0.3, 0.6, 1.0, 1.5, 2.5), kappas=(0.05, 0.15, 0.30, 0.50),
               cover=0.8, n=8192, seed=0):
    """WHERE IS THE INTERESTING DEMAND REGIME? Two knobs, and they are not interchangeable:

      * SIGMA sets the CONCENTRATION of demand (the OU logit scale, hence the entropy of what
        the world asks). This is what a chunk has to exploit and hence what it has to lose.
      * KAPPA sets the MIXING RATE (mean reversion), hence how fast a frozen concentration
        goes out of fashion. The *relative* size of one event is `sqrt(1 - (1-kappa)^2)` of the
        stationary spread and does NOT depend on sigma at all.

    A calibration lesson worth recording: binary-searching sigma against a target per-event KL
    (the `calibrate_sigma_event` recipe that worked for `transpose`'s magnitude) FAILS here and
    runs to the bound. The objective is non-monotone -- as sigma grows the softmax saturates
    toward one-hot and a fixed logit perturbation stops changing the argmax, so per-event KL
    rises and then falls. The two knobs have to be swept, not solved.

    Reported per cell: demand entropy and support, the per-event demand-KL at both mined
    levels, the coverage a table frozen at epoch 0 retains, its half-life in cycles, and the
    full true table's coverage (which must be flat -- it is the demand-invariant denominator of
    the concentration premium).
    """
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)[-1]
    truth = MC.true_tables(rules, depth, s, v, m, 3)
    rows = []
    for sig in sigmas:
        for kap in kappas:
            st = DM.new_demand(rules, list(levels), seed=seed, kappa=kap, sigma=sig)
            h = DM.demand_hist(rules, st, 2, s, depth, v, m, inv, n=n, seed=seed + 5)
            sim = DM.simulate_demand(rules, list(levels), kap, sig, s, depth, v, m, inv,
                                     cycles, period, start, level=2, cover=cover, seed=seed,
                                     n=n)
            cov = [r["frozen_coverage"] for r in sim]
            hl = next((r["cycle"] for r, c in zip(sim, cov) if c <= 0.5 * cov[0]), None)
            rows.append({
                "sigma": sig, "kappa": kap, "entropy": DM.demand_entropy(h),
                "support": len(h), "n_frozen": sim[0]["n_frozen"],
                "kl_event_L2": DM.event_demand_kl(rules, list(levels), kap, sig, 1, 2, s,
                                                  depth, v, m, inv, seed=seed, n=n),
                "kl_event_L3": DM.event_demand_kl(rules, list(levels), kap, sig, 1, 3, s,
                                                  depth, v, m, inv, seed=seed, n=n),
                "true_cover": DM.coverage(truth[2], h),
                "cov_start": cov[0], "cov_end": cov[-1], "half_life": hl})
    return {"sweep": rows}


@app.function(image=image, timeout=3600, memory=16384)
def cal_demand_remote():
    out = cal_demand()
    print(f"{'sigma':>6} {'kappa':>6} {'H':>7} {'supp':>5} {'nfroz':>6} {'KLev2':>8} "
          f"{'KLev3':>8} {'trueCov':>8} {'cov0':>7} {'cov_end':>8} {'half-life':>10}")
    for r in out["sweep"]:
        print(f"{r['sigma']:>6.2f} {r['kappa']:>6.2f} {r['entropy']:>7.3f} "
              f"{r['support']:>5d} {r['n_frozen']:>6d} {r['kl_event_L2']:>8.4f} "
              f"{r['kl_event_L3']:>8.4f} {r['true_cover']:>8.3f} {r['cov_start']:>7.3f} "
              f"{r['cov_end']:>8.3f} {str(r['half_life']):>10}")
    return out


def gate_demand(v=8, s=2, depth=4, m=2, rule_seed=0, max_level=3, n=4096):
    """G-D -- the demand lever is what it says it is, and nothing more.

      D-1  at uniform demand the weighted sampler is DISTRIBUTIONALLY the parent's: the mean
           repair distance and on-grammar rate of drawn instances match `context_instances`
           within sampling error. (It is not stream-identical -- the parent draws
           `rng.integers`, the weighted sampler `rng.random` -- which is exactly why the runner
           keeps the parent function itself on the drift-off path.)
      D-2  NOTHING BECOMES FALSE: the true tables are the same object at every epoch, drawn
           instances stay 100% on-grammar, and the precision of any table against the truth is
           invariant to any amount of demand drift. This is the property `transpose` could not
           have and is what makes the two rounds a typed pair.
      D-3  DEMAND ACTUALLY MOVES: per-event KL > 0 at both mined levels, and the coverage of a
           table frozen at epoch 0 falls.
      D-4  DIFFICULTY IS HELD: `d0` and the on-grammar rate are stationary across epochs (no
           trend), so demand-drift is not difficulty-drift in disguise.
      D-5  the OU is PREWARMED to stationarity -- the first event is not systematically smaller
           than later ones, the warm-up artefact `rhm_drift.advance_drift` warns about.
      D-6  the DEMAND-INVARIANT DENOMINATOR: the full true table's coverage of demand does not
           move with the drift, which is what licenses `given` as the concentration-premium
           reference.
    """
    from rhm.practice.ratchet.ratchet import context_instances as parent_ctx, era_ctx as ec
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)[-1]
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    era = {"name": "L1n6", "level": 1, "node": 6}
    out = {}

    # --- D-1 -------------------------------------------------------------------------------
    st0 = DM.new_demand(rules, [0, 1], seed=0, kappa=0.15, sigma=0.0, prewarm=False)
    pr, px = parent_ctx(rules, ec(era), 1024, s, depth, v, m, seed=101)
    wr, wx = DM.context_instances_demand(rules, ec(era), 1024, s, depth, v, m, 101, st0)
    d_par = float(nearest_derivation_cost(rules, px, pr, s).mean())
    d_wt = float(nearest_derivation_cost(rules, wx, wr, s).mean())
    out["D1_uniform_equivalence"] = {"d0_parent": d_par, "d0_weighted": d_wt,
                                     "og_parent": on_grammar_rate(px, inv, v, s),
                                     "og_weighted": on_grammar_rate(wx, inv, v, s)}
    assert abs(d_par - d_wt) < 0.15, f"D-1 failed: d0 {d_par:.3f} vs {d_wt:.3f}"

    # --- D-2 / D-3 / D-4 / D-6 -------------------------------------------------------------
    st = DM.new_demand(rules, [0, 1], seed=0, kappa=0.15, sigma=2.5, prewarm=True)
    rng = np.random.default_rng(7)
    h0 = DM.demand_hist(rules, st, 2, s, depth, v, m, inv, n=n, seed=5)
    froz = DM.demand_table(h0, 2, MC.base_table(v), s, 0.8)
    prec0 = MC.grade_table(froz, truth[2])["precision"]
    d0s, ogs, covs, kls, tcov = [], [], [], [], []
    for k in range(8):
        if k:
            DM.demand_step(st, rng, 1)
        h = DM.demand_hist(rules, st, 2, s, depth, v, m, inv, n=n, seed=5)
        r_, x_ = DM.context_instances_demand(rules, ec(era), 512, s, depth, v, m, 202, st)
        d0s.append(float(nearest_derivation_cost(rules, x_, r_, s).mean()))
        ogs.append(on_grammar_rate(x_, inv, v, s))
        covs.append(DM.coverage(froz, h))
        tcov.append(DM.coverage(truth[2], h))
        if k:
            kls.append(DM.demand_kl(h, h0))
        assert MC.grade_table(froz, truth[2])["precision"] == prec0, (
            "D-2 failed: a table's precision against the truth moved under demand drift")
    truth2 = MC.true_tables(rules, depth, s, v, m, max_level)
    assert all(np.array_equal(truth[l]["flat"], truth2[l]["flat"]) for l in truth), (
        "D-2 failed: the true tables are not invariant")
    assert min(ogs) == 1.0, "D-2 failed: drawn instances went off-grammar"
    out["D2_nothing_false"] = {"precision_invariant": prec0, "on_grammar": ogs[:3]}
    assert kls[-1] > 0.02 and covs[-1] < covs[0], (
        f"D-3 failed: demand did not move (KL {kls[-1]:.4f}, coverage {covs[0]:.3f} -> "
        f"{covs[-1]:.3f})")
    out["D3_demand_moves"] = {"kl_from_init": kls, "frozen_coverage": covs}
    spread = float(max(d0s) - min(d0s))
    trend = float(np.polyfit(np.arange(len(d0s)), d0s, 1)[0])
    out["D4_difficulty_held"] = {"d0": d0s, "spread": spread, "slope_per_epoch": trend}
    assert spread < 0.25 and abs(trend) < 0.02, (
        f"D-4 failed: difficulty drifted (spread {spread:.3f}, slope {trend:+.4f})")
    out["D6_denominator_flat"] = {"true_cover": tcov,
                                  "spread": float(max(tcov) - min(tcov))}
    assert max(tcov) - min(tcov) < 0.05, "D-6 failed: the true table's coverage moved"

    # --- D-5 -------------------------------------------------------------------------------
    # The artefact is a transient in the CONCENTRATION of demand, not in the size of an event.
    # A cold-started OU begins at theta = 0, i.e. at MAXIMUM-entropy demand -- a world with no
    # taste at all -- and then acquires taste as the walk spreads. Every "how much has the
    # frozen table's coverage decayed since the start" reading would then be measuring the
    # warm-up rather than the drift. Prewarming starts the walk at stationarity, so demand is
    # already as concentrated as it will ever typically be and only its *direction* moves.
    def entropy_path(state, k=8):
        r = np.random.default_rng(3)
        es = [DM.demand_entropy(DM.demand_hist(rules, state, 2, s, depth, v, m, inv,
                                               n=n, seed=5))]
        for _ in range(k):
            DM.demand_step(state, r, 1)
            es.append(DM.demand_entropy(DM.demand_hist(rules, state, 2, s, depth, v, m, inv,
                                                       n=n, seed=5)))
        return es
    warm = entropy_path(DM.new_demand(rules, [0, 1], seed=0, kappa=0.15, sigma=2.5,
                                      prewarm=True))
    cold = entropy_path(DM.new_demand(rules, [0, 1], seed=0, kappa=0.15, sigma=2.5,
                                      prewarm=False))
    w_slope = float(np.polyfit(np.arange(len(warm)), warm, 1)[0])
    c_slope = float(np.polyfit(np.arange(len(cold)), cold, 1)[0])
    out["D5_prewarm"] = {"warm_entropy": warm, "cold_entropy": cold,
                         "warm_slope": w_slope, "cold_slope": c_slope,
                         "warm_start_minus_mean": float(warm[0] - np.mean(warm[1:])),
                         "cold_start_minus_mean": float(cold[0] - np.mean(cold[1:]))}
    assert cold[0] > max(warm) and cold[0] - np.mean(cold[1:]) > 0.2, (
        "D-5 failed: a cold start was expected to begin at maximum-entropy (tasteless) demand")
    assert abs(warm[0] - np.mean(warm[1:])) < abs(cold[0] - np.mean(cold[1:])), (
        "D-5 failed: prewarming did not remove the concentration transient")
    return out


def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3, eras="1:6,2:3,3:1"):
    """`transpose`'s gates (which carry the arc's C-M/C-R/G-D, `ear`'s G-N/G-S/G-R,
    `recital`'s G-P and `transpose`'s own T-1..T-5), plus this round's G-D."""
    out = transpose_selfcheck(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, n=n,
                              max_level=max_level, eras=eras)
    out["demand"] = gate_demand(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed,
                                max_level=max_level)
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    out = selfcheck()
    print("G-D (the demand lever): "
          + json.dumps(out["demand"], indent=2, cls=NumpyEncoder))
    return out


@app.local_entrypoint()
def main(quick: bool = True):
    setlist.remote(tag="smoke0", quick=quick)
