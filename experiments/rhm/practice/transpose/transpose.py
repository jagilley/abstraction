"""transpose -- the piece changes key under your hands: commitment policy against a LIVE
conditioning gap.

Round 6 of the practice arc on RHM (etude -> crystallize -> ratchet -> ear -> recital -> tall).

WHY THIS ROUND EXISTS. A forward model / certificate is epistemically load-bearing only across a
CONDITIONING GAP -- when the outcome it predicts depends on information it cannot condition on,
so its error stream carries news. This substrate has a BOUNDARY gap (hidden ancestors make
launch-time observation informative, which is what state-conditioned commitment buys) and an
EPISTEMIC WITHIN-UNIT gap (a committed macro forgoes priced groundings that would have revealed
facts about the grammar). It has never had a NEWS gap: the grammar is static, damage recurs
i.i.d. within an era, the plant is inert -- so no channel exists by which already-committed
content can become wrong AFTER commitment. The arc's own instruments say exactly this, in four
independent places: recert fired 0/24 (`ear`) and 0/288 (`recital`); `tall` proved the mined
table cannot churn (miner counts only increase); the etude's post-commit drift was exactly
0.0000; `crystallize` found "nothing to certify".

So the arc's standing conclusion -- provisional commitment at the boundary beats certification,
and the recert safety net never fires because mining is selection-filtered upstream -- has only
ever been measured where post-commit invalidation is STRUCTURALLY IMPOSSIBLE. No round has
combined the compile op with a live gap: the early `mjc` nodes (bridge drift, aleatoric_flip)
had the gap and no compile op; `etude` and the whole RHM arc have the op and a static gap. This
round fills the empty cell, and changes NOTHING ELSE.

THE LEVER: within-run grammar drift (`drift.py`). Rule-table cells at level `drift_level` are
resampled on a fixed cycle schedule. A committed level-l macro entry is a level-1 tuple that WAS
derivable at level l; after a resample it may not be, and applying the macro then writes an
off-grammar span. That is post-commit invalidation, which no other lever considered (surface
masking, stochastic execution inside the unit) actually delivers -- those make launch a bet,
they do not make committed content become wrong. Drift is UNANNOUNCED: no arm reads a drift
signal, so the news has to arrive through the error stream.

WHAT THE ROUND VARIES. The commit-policy axis the static regime already ordered -- never /
given / certificate-gated / provisional-at-boundary -- now with the recert channel live, plus
the two mechanisms a live gap makes necessary and the static regime made meaningless:

    never_base      base level-1 moves forever                        (the world-difficulty anchor)
    given           the DGP's tables from cycle 1, FROZEN             (the static handover)
    given_live      the DGP's tables, RE-READ every epoch             (the tracking teacher)
    practice_gated  earned, unit-LP certificate gates the commit      (`ratchet`'s control)
    practice_prov   earned, provisional at the boundary, recert ON    (`ear`'s rescue)
    prov_norecert   earned, provisional at the boundary, recert OFF   (isolates the safety net)
    practice_climb  earned, certify-else-provisional, recert ON       (the settled policy)
    climb_decay     ... plus a vocabulary that can FORGET             (isolates churn)

`given` forks under drift because the true table does: an initial-table arm bounds what a
perfect but static handover is worth, a current-table arm bounds what perfect tracking is worth,
and the gap between them is the price of the news the agent has to buy for itself.

STRUCTURAL BUILD ITEM. `tall` showed the miner cannot churn -- entry counts only increase, so a
support threshold on them is monotone and removal is impossible. Under drift, staleness is real,
so the vocabulary needs a way to lose entries. The mechanism chosen (`drift.DecayMiner`) is
geometric count decay: agent-internal, on its own clock, no oracle and no drift signal. It is
carried by ONE arm so that "can the vocabulary forget?" is a measured contrast rather than a
substrate change.

FIDELITY IS THE ARC'S LAW. With `--drift-every 0 --t-budget 0` this file IS `recital` in its
`ear` mode: the drift epoch is 0 forever, every added instrument is gated behind `drift_on`, and
no new code consumes randomness. `tf_s0` runs `ear`'s own eight arms in `ear`'s own order and
must reproduce `er_s0` cycle for cycle (as `recital`'s `rf_s0` does).

Run from experiments/:
  modal run rhm/practice/transpose/transpose.py::selfcheck_remote
  modal run rhm/practice/transpose/transpose.py::cal_drift_remote
  modal run rhm/practice/transpose/transpose.py::transpose --quick --tag smoke0
  python3 rhm/practice/transpose/launch_detached.py --fn transpose --tag tp_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_data import generate_rules_distinct
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
    ARMS as RECITAL_ARMS, _cfg as recital_cfg, new_detector, parse_arms_nm, selfcheck as
    recital_selfcheck, step_detector, vocab_rate)
from rhm.practice.transpose import drift as D


app = modal.App("rhm-practice-transpose", image=image)

REMOTE = "rhm_practice_transpose"


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
#   vocab   -- base | true (frozen initial tables) | true_live (tracking) | earned
#   decay   -- whether this arm's miner can FORGET (`drift.DecayMiner`, cfg["mine_decay"])
#   everything else is `recital`'s / `ear`'s, unchanged, so the inherited arms replay exactly.
_CLIMB = {"vocab": "earned", "commit": "delta_prov", "grader": "mfg",
          "pay_era": False, "recert": True, "pace": "cycles"}
ARMS = {k: {**v, "decay": False} for k, v in RECITAL_ARMS.items()}
ARMS.update({
    "given_live":     {"vocab": "true_live", "commit": None, "grader": None,
                       "pay_era": True, "recert": False, "pace": "cycles", "decay": False},
    "prov_norecert":  {"vocab": "earned", "commit": "prov", "grader": None,
                       "pay_era": False, "recert": False, "pace": "cycles", "decay": False},
    "climb_decay":    {**_CLIMB, "decay": True},
    "gated_decay":    {"vocab": "earned", "commit": "delta", "grader": "era",
                       "pay_era": True, "recert": False, "pace": "cycles", "decay": True},
})


# --------------------------------------------------------------------------- #
# the world: the whole drift trajectory, precomputed ONCE and shared by every arm
# --------------------------------------------------------------------------- #

def epoch_refs(shared, cfg, eras, device, rules, rules_t, truth, meter, n_floor=128):
    """The arm-independent references for ONE grammar, measured on that epoch's own metering
    sets with the STALE setup nets -- so "how hard did the world just get" is separated from
    "how well is this arm doing". `tall`'s lesson is why the exact-DP floor is here at every
    epoch rather than only at setup: *descent toward the floor* belongs in the gate set, and a
    moving world moves the floor, so a recovered fraction measured against a stale reference
    would be meaningless. It is taken on `n_floor` instances rather than the whole metering set
    because the oracle is the expensive term and a slightly noisier floor is cheap insurance."""
    import torch
    from rhm.practice.crystallize.units import oracle_rollout
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    true_ms = build_ms(base_ms, {ell: truth[ell] for ell in range(2, cfg["max_macro_level"] + 1)},
                       s, depth, device)
    out = {"d0": [], "on_grammar": [], "stale": [], "stale_true": [], "floor": [],
           "macro_true": []}
    for i, era in enumerate(eras):
        r_np, mx = meter[i]
        x_np = mx.numpy()
        out["d0"].append(float(nearest_derivation_cost(rules, x_np, r_np, s).mean()))
        out["on_grammar"].append(on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s))
        k = min(n_floor, x_np.shape[0])
        xo, _ = oracle_rollout(shared["generator0"], mx[:k].to(device), r_np[:k], base_ms,
                               rules, rules_t, shared["canon"], depth, v, m, s, cfg["budget"])
        sc, _ = grade(xo.cpu().numpy(), r_np[:k], rules, s)
        out["floor"].append(1.0 - float(sc.mean()))
        for tag, mset in (("base", base_ms), ("true", true_ms)):
            w = fit_width(len(mset), cfg["budget"], cfg["g_budget"])
            b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], mx,
                           torch.from_numpy(r_np), mset, rules_t, shared["canon"], depth, v, m,
                           s, budget=cfg["budget"], beam_width=w, device=device)
            sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
            out["stale" if tag == "base" else "stale_true"].append(1.0 - float(sc.mean()))
        cell = {}
        for ell in range(2, cfg["max_macro_level"] + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            mv = MC.to_device(MC.make_macro(ell, node, s, truth[ell]), device)
            cell[str(ell)] = audition_macro(shared["generator0"], mx.to(device), r_np, mv,
                                            rules_t, shared["canon"], depth, v, m, s, rules)["e"]
        out["macro_true"].append(cell)
    return out


def build_world(shared, cfg, eras, device, max_cycle, with_refs=True):
    """Precompute every grammar the run will pass through, and everything derived from it.

    Doing this ONCE at setup rather than per arm is not an optimisation, it is the control:
    every arm then lives in literally the same world at literally the same cycle, so a
    difference between arms cannot be a difference in the draws. The held-out sets are rebuilt
    at each epoch FROM THE SAME SEEDS as `ear`/`recital` use, so the only thing that changes
    between epochs is the grammar -- the minimal-change redraw.
    """
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    traj = D.rules_trajectory(shared["rules"], cfg, max_cycle)
    world = []
    for node in traj:
        rules = node["rules"]
        ep = {"epoch": node["epoch"], "cycle": node["cycle"], "changed": node["changed"],
              "rules": rules,
              "rules_t": [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules],
              "truth": MC.true_tables(rules, depth, s, v, m, cfg["max_macro_level"])}
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
        ep["meter"], ep["shadow"], ep["realset"] = meter, shadow, realset
        ep["probe_clean"] = torch.from_numpy(
            _sample_pool(rules, cfg["n_probe_clean"], s, cfg["seed"] + 31337)[1])
        ep["mag"] = {
            "vs_init": D.table_survival(traj[0]["rules"], rules, depth, s, v, m,
                                        cfg["max_macro_level"], truth_b=ep["truth"]),
            "vs_prev": D.table_survival(traj[max(0, node["epoch"] - 1)]["rules"], rules, depth,
                                        s, v, m, cfg["max_macro_level"], truth_b=ep["truth"]),
            "corpus_vs_init": D.corpus_survival(traj[0]["rules"], rules, s, 512,
                                                cfg["seed"] + 4711)}
        if with_refs:
            ep["refs"] = epoch_refs(shared, cfg, eras, device, rules, ep["rules_t"],
                                    ep["truth"], meter)
        world.append(ep)
    return world


def world_summary(world, cfg):
    """The JSON-able record of the world's motion -- what goes into `setup.json` and what the
    reducer reads to say how far the piece was transposed."""
    return [{"epoch": e["epoch"], "cycle": e["cycle"], "n_changed": len(e["changed"]),
             "changed": e["changed"], "mag": e["mag"],
             "refs": e.get("refs")} for e in world]


# --------------------------------------------------------------------------- #
# the arm loop -- `recital`'s, with a moving grammar under it
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
    drift_on = cfg["drift_every"] > 0 and cfg["drift_cells"] > 0

    # ---- the world at epoch 0 ------------------------------------------------------------
    cur_ep = 0
    ep = world[0]
    rules, rules_t, truth = ep["rules"], ep["rules_t"], ep["truth"]
    meter, shadow, realset = ep["meter"], ep["shadow"], ep["realset"]
    truth_frozen = world[0]["truth"]

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
    dec = cfg["mine_decay"] if spec.get("decay") else 1.0
    miners = {ell: D.DecayMiner(ell, s, decay=dec, floor=cfg["mine_floor"])
              for ell in range(2, maxl + 1)}
    if spec["vocab"] in ("true", "true_live"):
        for ell in range(2, maxl + 1):
            committed[ell] = truth_frozen[ell] if spec["vocab"] == "true" else truth[ell]
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
           # --- this round -----------------------------------------------------------------
           "epoch": [], "stale": [], "churn": [], "bank_fresh": []}

    cyc, era_i, c_in_era = 0, 0, 0
    era = eras[era_i]
    active = era["level"] + 1
    cert = new_detector()
    tdet = new_detector()
    vdet = new_detector()
    vhist = []
    mfg = None
    node_next = node_at(era, active, s) if active <= maxl else None
    t_era_start = 0.0
    next_probe_t = (T / cfg["probe_t_n"]) if T > 0 else float("inf")
    print(f"\n----- arm={arm} pace={pace} vocab={spec['vocab']} decay={dec} "
          f"budget={T:.0f} ERA 1: damage {era['name']} (earning level {active}, "
          f"consumption node {node_next}) -----", flush=True)

    while True:
        cyc += 1
        c_in_era += 1
        counts = {"mat": 0, "ground": 0}
        gcost = {}

        # --- (a0) THE WORLD MOVES. Unannounced: nothing below reads `moved`; it only rebinds
        #          the grammar the world is graded by and the held-out sets drawn from it.
        moved = False
        new_ep = D.epoch_of(cyc, cfg)
        if new_ep != cur_ep:
            cur_ep, moved = new_ep, True
            ep = world[cur_ep]
            rules, rules_t, truth = ep["rules"], ep["rules_t"], ep["truth"]
            meter, shadow, realset = ep["meter"], ep["shadow"], ep["realset"]
            if spec["vocab"] == "true_live":
                for ell in range(2, maxl + 1):
                    committed[ell] = truth[ell]
                ms = build_ms(base_ms, committed, s, depth, device)
            events.append({"kind": "world", "arm": arm, "cycle": cyc, "epoch": cur_ep,
                           "era": era_i + 1, "n_changed": len(ep["changed"]),
                           "mag": ep["mag"]})

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

        # --- (b) mine. `age()` first: a tuple's evidence is recency-weighted, so a vocabulary
        #         can FORGET what the world stopped producing. No-op at decay 1.0.
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
            # THE POST-COMMIT UNIT ERROR TRAJECTORY. `ear`/`recital` measured this only for
            # earned-vocabulary arms with a recert; under drift it is the round's central
            # readout for EVERY arm holding committed content, `given` included -- the series
            # whose baseline in every prior round is exactly flat.
            held_arms = (spec["vocab"] == "earned"
                         or (drift_on and committed[ell] is not None))
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
            aud[str(ell)] = cellA

        # --- (e2) STALENESS + CHURN: pure bookkeeping on the tables, no substrate work.
        #          `stale` is what the committed content is worth against the CURRENT grammar
        #          (and, for contrast, against the one it was mined in). `churn` is `tall`'s
        #          entry-identity instrument, which had nothing to measure until now.
        stale_row, churn_row = {}, {}
        if drift_on:
            for ell in range(2, maxl + 1):
                cell = {}
                if committed[ell] is not None:
                    cur = MC.grade_table(committed[ell], truth[ell])
                    old = MC.grade_table(committed[ell], truth_frozen[ell])
                    cell = {"n": cur["n_learned"],
                            "prec_now": cur["precision"], "recall_now": cur["recall"],
                            "prec_init": old["precision"], "recall_init": old["recall"]}
                live_t = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                now = D.entry_set(live_t)
                churn_row[str(ell)] = D.churn(prev_entries[ell], now)
                prev_entries[ell] = now
                stale_row[str(ell)] = cell

        # --- (f) the climbed grader: the (ctx x ev) grid at the active level -------------
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
                        log["mfg"].append(mfg_ev)
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

        # --- (g2/g2b) the task and vocabulary pacers: logged instruments in every arm -----
        fire_task, tdet_state = step_detector(tdet, e, cfg, c_in_era)
        if active <= maxl:
            mst = miners[active].state()
            vhist.append((mst["n_distinct"], mst["n_obs"]))
            v_rate = vocab_rate(vhist, cfg["sil_W"])
        else:
            v_rate = None
        fire_vocab, vdet_state = step_detector(vdet, v_rate, cfg, c_in_era)

        # --- (g3) ADVANCEMENT ------------------------------------------------------------
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
                      c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg), epoch=cur_ep,
                      provisional=bool(prov), audition=held, shadow=A, e_task=e,
                      aud_real_pol=clim.get("real", {}).get("pol"),
                      aud_mfg_pol=clim.get("mfg", {}).get("pol"),
                      aud_era_act=aud.get(str(active), {}).get("cand"),
                      n_entries=int(tbl["child"].shape[0]),
                      n_moves_after=len(ms), sil_run=int(cert["run"]),
                      **{f"tab_{k}": val for k, val in
                         MC.grade_table(tbl, truth[active]).items()})
            events.append(ev)
            print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} ep{cur_ep} "
                  f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                  f"recall={ev['tab_recall']:.3f} prec={ev['tab_precision']} "
                  f"audition={held} shadow={A} moves={len(ms)}", flush=True)

        # --- (i) the live recert. Under drift it reaches EVERY committed level, not only the
        #         era's own: level-2 content keeps going stale through era 3, and a channel
        #         that cannot reach it would decide the round by construction. Ascending
        #         order, so a swapped level-2 table lets level 3 rebuild over it (gate G-R).
        rc_levels = [era["level"]]
        if drift_on and cfg["recert_all"]:
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
                   **{f"froz_{k}": val for k, val in
                      MC.grade_table(committed[rc_level], truth[rc_level]).items()}}
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
                            MC.grade_table(live, truth[rc_level]).items()})
                if pl_["e"] <= pf_["e"] - cfg["recert_margin"]:
                    committed[rc_level] = live
                    ms = build_ms(base_ms, committed, s, depth, device)
                    rec["swapped"] = True
                    print(f"[recert] arm={arm} era{era_i+1} L{rc_level} c{cyc} SWAP "
                          f"{rec['n_frozen']}->{rec['n_live']} entries  "
                          f"e {pf_['e']:.4f} -> {pl_['e']:.4f}", flush=True)
            gcost.setdefault("recert", {"ground": 0, "mat": 0})
            gcost["recert"]["ground"] += rc_cost["ground"]
            gcost["recert"]["mat"] += rc_cost["mat"]
            events.append(rec)

        # --- book the cycle -------------------------------------------------------------
        t_cum += priced(counts, cfg)
        budget_done = (T > 0 and t_cum >= T)
        clock_done = (T <= 0 and last_era and c_in_era >= cfg["era_cycles"])
        cap_done = cyc >= cfg["max_cycles"]
        stop = budget_done or clock_done or cap_done

        # --- (j) probes: the width ladder, ALL eras' metering sets, the plant guard -------
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

        # --- the bank's own freshness: how much of what the agent manufactures its
        #     auditions from is still a legal derivation of the world it is graded in.
        bank_fresh = None
        if drift_on and bank_x.shape[0]:
            n_bf = min(256, int(bank_x.shape[0]))
            bs, _ = grade(bank_x[-n_bf:].numpy(), bank_r[-n_bf:].numpy(), rules, s)
            bank_fresh = float(bs.mean())

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
        st2 = stale_row.get(str(era["level"]), {}) or {}
        print(f"[c{cyc:3d}] arm={arm:14s} era{era_i+1} i{c_in_era:3d} ep{cur_ep:2d} "
              f"t={t_cum:10.0f} e={e:.4f} w={width} nm={len(ms)} A={fmt(A)} "
              f"held={fmt(a_act.get('held'))} true={fmt(a_act.get('true'))} "
              f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
              f"prec_now={fmt(st2.get('prec_now'))} "
              f"vr={fmt(v_rate)} sil={cert['run']}/{tdet['run']}/{vdet['run']}"
              + ("  WORLD-MOVED" if moved else "")
              + (f"  ADVANCE[{adv_reason}]" if advance else ""), flush=True)
        if cyc % cfg["checkpoint_every"] == 0:
            write_results(outdir, arm, cfg, eras, refs, log, events, complete=False)

        # --- advance the ladder (monotone; never backwards) ------------------------------
        if advance and not stop:
            events.append({"kind": "advance", "arm": arm, "from_era": era_i + 1,
                           "to_era": era_i + 2, "cycle": cyc, "c_in_era": c_in_era,
                           "reason": adv_reason, "t_cum": t_cum, "epoch": cur_ep,
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
                  f"at c{cyc}, t={t_cum:.0f} -----", flush=True)
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
    cfg = recital_cfg()
    cfg.update(
        # --- the live conditioning gap ---------------------------------------------------
        drift_every=0,       # cycles between drift events; 0 == OFF == `recital`/`ear` exactly
        drift_cells=1,       # (feature, rule) cells of the drifting layer resampled per event
        drift_level=2,       # which rule layer drifts. >= 2 keeps the leaf rendering exact
        drift_start=6,       # first cycle at which the world moves
        drift_seed=0,        # the world's own stream, independent of every arm's randomness
        # --- a vocabulary that can forget -------------------------------------------------
        mine_decay=1.0,      # geometric ageing of mining counts; 1.0 == `MC.Miner` exactly
        mine_floor=0.5,      # a tuple is forgotten once its aged count falls below this
        recert_all=True,     # under drift the recert reaches every committed level
        t_budget=0.0,        # this round runs on `ear`'s ERA CLOCK, so all arms share a world
    )
    cfg.update(kw)
    return cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def transpose(
    tag: str = "smoke",
    arms: str = ("never_base,given,given_live,practice_gated,practice_prov,prov_norecert,"
                 "practice_climb,climb_decay"),
    eras: str = "1:6,2:3,3:1", seed: int = 0, era_cycles: int = 30, budget: int = 4,
    rule_seed: int = 0, train_seed: int = 1,
    drift_every: int = 5, drift_cells: int = 1, drift_level: int = 2, drift_start: int = 6,
    drift_seed: int = 0, mine_decay: float = 1.0, mine_floor: float = 0.5,
    recert_all: bool = True,
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
               drift_every=drift_every, drift_cells=drift_cells, drift_level=drift_level,
               drift_start=drift_start, drift_seed=drift_seed, mine_decay=mine_decay,
               mine_floor=mine_floor, recert_all=recert_all,
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
                   drift_start=(cfg["drift_start"] if cfg["drift_every"] <= 0 else 3),
                   drift_every=(cfg["drift_every"] if cfg["drift_every"] <= 0 else 3),
                   t_budget=(0.0 if t_budget <= 0 else 5.0e5), probe_t_n=6)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    max_cycle = (cfg["era_cycles"] * len(ers) if cfg["t_budget"] <= 0 else cfg["max_cycles"])
    print(f"transpose tag={tag} arms={arms} eras={ers} drift=(every {cfg['drift_every']} "
          f"cycles from c{cfg['drift_start']}, {cfg['drift_cells']} cells of level "
          f"{cfg['drift_level']}) max_cycle={max_cycle} device={device}")
    assert cfg["drift_level"] >= 2 or cfg["drift_every"] <= 0, (
        "drift at level 1 would move the leaf rendering, the inverse maps and the reader "
        "with it -- the vocabulary effect would be confounded with a broken parse")

    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {shared['truth'][ell]['child'].shape[0]} entries")
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}")

    t0 = time.time()
    world = build_world(shared, cfg, ers, device, max_cycle)
    print(f"[world] {len(world)} epochs built in {time.time() - t0:.0f}s")
    for epn in world:
        if epn["epoch"] == 0:
            continue
        sv = {k: (None if c["survival"] is None else round(c["survival"], 3))
              for k, c in epn["mag"]["vs_init"].items()}
        print(f"  epoch {epn['epoch']:2d} @c{epn['cycle']:3d}: "
              f"survival-vs-init {sv} corpus {epn['mag']['corpus_vs_init']:.3f} "
              f"stale={[round(x, 3) for x in epn['refs']['stale']]} "
              f"floor={[round(x, 3) for x in epn['refs']['floor']]} "
              f"macro_true={[{k: round(x, 3) for k, x in c.items()} for c in epn['refs']['macro_true']]}")

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs,
                   "world": world_summary(world, cfg)}, fh, indent=2, cls=NumpyEncoder)
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


# --------------------------------------------------------------------------- #
# calibration: locate the interesting drift regime, offline, before spending a GPU
# --------------------------------------------------------------------------- #

def cal_drift(v=8, s=2, depth=4, m=2, rule_seed=0, max_macro_level=3, cycles=90,
              era_cycles=30, obs_per_cycle=8, mine_support=3, commit_cycle=25,
              periods=(2, 3, 5, 10, 20, 30), cells=(1, 2, 4), levels=(2, 3),
              decays=(1.0, 0.98, 0.95, 0.90, 0.80), seed=0):
    """WHERE IS THE INTERESTING REGIME? Every knob in this arc is set by a measurement.

    Three measurements, all CPU, all on the DGP's own tables so nothing here depends on a
    trained substrate:

      (1) MAGNITUDE. Per (level, cells): the per-event survival of the level's vocabulary and
          of a corpus drawn from the pristine grammar, and the resulting HALF-LIFE in cycles at
          each candidate period. This is the world's motion in the currency that matters --
          fractions, not nats: hard support drift has no finite KL.
      (2) THE PAYBACK BRACKET. Per (period, cells, decay): the steady-state recall/precision of
          a mined table against the CURRENT truth, and the validity of a table FROZEN at
          `commit_cycle`. Churn faster than the miner can re-support a tuple and recall
          collapses (nothing is ever worth compiling); churn slower than the run and the static
          case is rebuilt. The bracket is what these rows locate.
      (3) THE FORGETTING HORIZON. Per decay: how long a tuple survives with no further
          evidence, in closed form and by simulation -- the quantity that has to sit between
          the re-support time and the drift period for a vocabulary to track.
    """
    rules0 = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    base = dict(v=v, s=s, depth=depth, m=m, max_macro_level=max_macro_level,
                drift_start=6, drift_seed=seed, commit_cycle=commit_cycle)
    out = {"magnitude": [], "bracket": [], "forgetting": []}

    # (1) magnitude ---------------------------------------------------------------------
    for lvl in levels:
        for k in cells:
            rng = np.random.default_rng(seed + 4242)
            r1, _ = D.resample_cells(rules0, depth, lvl, k, rng)
            sv = D.table_survival(rules0, r1, depth, s, v, m, max_macro_level)
            row = {"level": lvl, "cells": k,
                   "corpus_survival": D.corpus_survival(rules0, r1, s, 2048, seed + 77)}
            for ell, cellv in sv.items():
                row[f"L{ell}_survival"] = cellv["survival"]
                row[f"L{ell}_novel"] = cellv["novel"]
            key_lvl = str(max(2, lvl))
            q = sv.get(key_lvl, {}).get("survival")
            row["half_life_cycles"] = {str(P): (None if q is None else
                                                round(D.half_life_cycles(q, P), 1))
                                       for P in periods}
            out["magnitude"].append(row)

    # (2) the payback bracket -------------------------------------------------------------
    for P in periods:
        for k in cells:
            for dec in decays:
                cfg = {**base, "drift_every": P, "drift_cells": k, "drift_level": 2}
                rows = D.simulate_miner(rules0, cfg, cycles, obs_per_cycle, dec, seed + 11,
                                        support=mine_support, level=2)
                tail = rows[-era_cycles:]
                fz = [r for r in rows if r.get("frozen_precision") is not None]
                out["bracket"].append({
                    "period": P, "cells": k, "decay": dec,
                    "recall_tail": float(np.mean([r["recall"] for r in tail])),
                    "prec_tail": float(np.mean([x for x in (r["precision"] for r in tail)
                                                if x is not None] or [float("nan")])),
                    "entries_tail": float(np.mean([r["n_entries"] for r in tail])),
                    "n_forgotten": rows[-1]["n_forgotten"],
                    "frozen_prec_end": (fz[-1]["frozen_precision"] if fz else None),
                    "frozen_recall_end": (fz[-1]["frozen_recall"] if fz else None)})

    # (3) the forgetting horizon ----------------------------------------------------------
    for dec in decays:
        if dec >= 1.0:
            out["forgetting"].append({"decay": dec, "cycles_to_forget": None,
                                      "steady_count_at_lambda_0.5": None})
            continue
        mn = D.DecayMiner(2, s, decay=dec)
        mn.observe(np.zeros((10, 2), np.int64))          # count 10 on one tuple
        n = 0
        while mn.counts and n < 500:
            mn.age(); n += 1
        out["forgetting"].append({
            "decay": dec, "cycles_to_forget_from_count_10": n,
            "cycles_below_support": (None if dec <= 0 else
                                     round(float(np.log(mine_support / 10.0)
                                                 / np.log(dec)), 1)),
            "steady_count_at_lambda_0.5": round(0.5 / (1.0 - dec), 2)})
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def cal_descent(
    tag: str = "cal0", specs: str = "0/1/2,8/1/2,4/1/2,8/1/3",
    cycles: int = 40, mine_decay: float = 0.95,
    arms: str = "never_base,practice_climb,climb_decay",
    eras: str = "1:6", seed: int = 0, budget: int = 4, g_budget: int = 58,
    max_macro_level: int = 3, quick: bool = False,
):
    """THE DESCENT GATE, at a ladder of drift rates -- `tall`'s lesson applied before the fact.

    `tall` was voided because its feasibility gates covered trainability, exactness, damage,
    the ladder gradient and pricing, and never asked *does the policy descend toward the floor
    in a realistic cycle budget*. Drift is exactly the kind of change that could break that: the
    controller, generator and value were all trained on the pristine grammar, so a world that
    moves too fast degrades the substrate rather than testing the vocabulary.

    So: one setup, one era, `cycles` cycles, the same three arms at each drift rate, and the
    readout that matters is the RECOVERED FRACTION against that epoch's own moving references
    `(stale - e) / (stale - floor)`. Rate 0 is the in-run control, so no cross-run inference is
    needed. This is what sets `drift_every` for the main run.
    """
    import torch
    cfg = _cfg(seed=seed, budget=budget, g_budget=g_budget, era_cycles=cycles,
               max_macro_level=max_macro_level, t_budget=0.0, max_cycles=cycles + 5,
               drift_start=6, mine_decay=mine_decay, probe_every=8)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   era_cycles=8, n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                   checkpoint_every=2, sil_min_cycle=2, late_offset=1, gen_steps=5,
                   bank_cap=256, recert_every=2, mfg_min_bank=16, max_cycles=12,
                   drift_start=3)
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
        every, cells, level = [int(x) for x in spec.strip().split("/")]
        rate = f"{every}c{cells}L{level}"
        rcfg = {**cfg, "drift_every": every, "drift_cells": cells, "drift_level": level}
        world = build_world(shared, rcfg, ers, device, rcfg["era_cycles"])
        print(f"\n##### drift every={every} cells={cells} level={level}: "
              f"{len(world)} epochs #####", flush=True)
        for epn in world:
            print(f"  ep{epn['epoch']:2d} @c{epn['cycle']:3d} "
                  f"corpus={epn['mag']['corpus_vs_init']:.3f} "
                  f"stale={[round(x, 3) for x in epn['refs']['stale']]} "
                  f"floor={[round(x, 3) for x in epn['refs']['floor']]} "
                  f"stale_true={[round(x, 3) for x in epn['refs']['stale_true']]}", flush=True)
        sub = os.path.join(outdir, f"d{rate}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "setup.json"), "w") as fh:
            json.dump({"config": rcfg, "eras": ers, "refs": refs,
                       "world": world_summary(world, rcfg)}, fh, indent=2, cls=NumpyEncoder)
        for label, base, ov in parse_arms_nm(arms):
            print(f"\n===== drift {rate} arm {label} =====", flush=True)
            r = run_arm(label, base, ov, shared, rcfg, ers, refs, sub, device, world)
            log = r["log"]
            eps = np.asarray(log["epoch"])
            st = np.asarray([world[int(k)]["refs"]["stale"][0] for k in eps])
            fl = np.asarray([world[int(k)]["refs"]["floor"][0] for k in eps])
            ev = np.asarray(log["e"])
            rec = (st - ev) / np.maximum(st - fl, 1e-9)
            summary[f"{rate}/{label}"] = {
                "e_first5": float(ev[:5].mean()), "e_last5": float(ev[-5:].mean()),
                "recovered_first5": float(rec[:5].mean()),
                "recovered_last5": float(rec[-5:].mean()),
                "stale_first": float(st[0]), "stale_last": float(st[-1]),
                "floor_first": float(fl[0]), "floor_last": float(fl[-1]),
                "corpus_end": world[int(eps[-1])]["mag"]["corpus_vs_init"],
                "commits": [(e["level"], e["cycle"], e["n_entries"], e["tab_recall"])
                            for e in r["events"] if e["kind"] == "commit"],
                "recert_swaps": sum(1 for e in r["events"]
                                    if e["kind"] == "recert" and e["swapped"]),
                "recerts": sum(1 for e in r["events"] if e["kind"] == "recert")}
            print(f"  -> {json.dumps(summary[f'{rate}/{label}'], cls=NumpyEncoder)}",
                  flush=True)
        volume.commit()
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print("\n=== DESCENT GATE ===")
    print(f"{'cell':30s} {'e_first5':>9s} {'e_last5':>8s} {'rec_first5':>11s} "
          f"{'rec_last5':>10s} {'stale':>6s} {'floor':>6s} {'corpus':>7s}")
    for k, x in summary.items():
        print(f"{k:30s} {x['e_first5']:>9.3f} {x['e_last5']:>8.3f} "
              f"{x['recovered_first5']:>11.3f} {x['recovered_last5']:>10.3f} "
              f"{x['stale_last']:>6.3f} {x['floor_last']:>6.3f} {x['corpus_end']:>7.3f}")
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return summary


@app.function(image=image, timeout=3600, memory=16384)
def cal_drift_remote(cycles: int = 90, obs_per_cycle: int = 8):
    out = cal_drift(cycles=cycles, obs_per_cycle=obs_per_cycle)
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_drift(v=8, s=2, depth=4, m=2, rule_seed=0, max_level=3):
    """G-T -- the drift is what it says it is, and nothing more.

      T-1  a level-l resample touches ONLY the level-l layer, changes exactly `n_cells` cells,
           and preserves `generate_rules_distinct`'s within-feature distinctness. In particular
           the BOTTOM layer is bit-identical at every drift level >= 2, so `canon`, the inverse
           maps, the reader and the generator's block head stay exact.
      T-2  drift is LEVEL-INDEXED in the vocabulary too: a level-2 resample changes T2 and T3;
           a level-3 resample changes T3 and leaves T2 bit-identical. That alignment is the
           round's instrument.
      T-3  POST-COMMIT INVALIDATION IS REAL: a table that was perfect before the event is no
           longer perfect after it -- the precondition the whole arc has lacked. Measured as
           set precision against the new truth, and as the on-grammar rate of spans the frozen
           table renders.
      T-4  the decaying miner is `MC.Miner` EXACTLY at decay 1.0 (which is what makes the
           fidelity replay structural, not a coincidence), and DOES forget at decay < 1.
      T-5  with `drift_every = 0` the trajectory has exactly one epoch and it is the pristine
           grammar -- drift off is a no-op by construction, not by care.
    """
    out = {}
    rules0 = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    rng = np.random.default_rng(rule_seed + 4242)

    # --- T-1 ---------------------------------------------------------------------------
    for lvl in (2, 3):
        for k in (1, 3):
            r1, changed = D.resample_cells(rules0, depth, lvl, k, rng)
            assert len(changed) == k, f"T-1: asked for {k} cells, changed {len(changed)}"
            for j in range(depth):
                same = np.array_equal(rules0[j], r1[j])
                if j == depth - lvl:
                    assert not same, f"T-1: the level-{lvl} layer did not move"
                    assert int((rules0[j] != r1[j]).any(-1).sum()) == k, (
                        f"T-1: {k} cells asked, "
                        f"{int((rules0[j] != r1[j]).any(-1).sum())} moved")
                else:
                    assert same, f"T-1: layer {j} moved on a level-{lvl} drift"
            layer = r1[depth - lvl]
            for f in range(v):
                tups = {tuple(int(x) for x in layer[f, j]) for j in range(m)}
                assert len(tups) == m, f"T-1: feature {f} lost within-feature distinctness"
            out[f"T1_level{lvl}_cells{k}"] = {"changed": len(changed),
                                              "bottom_identical": True}

    # --- T-2 ---------------------------------------------------------------------------
    t0 = MC.true_tables(rules0, depth, s, v, m, max_level)
    r2, _ = D.resample_cells(rules0, depth, 2, 1, np.random.default_rng(7))
    r3, _ = D.resample_cells(rules0, depth, 3, 1, np.random.default_rng(7))
    t2 = MC.true_tables(r2, depth, s, v, m, max_level)
    t3 = MC.true_tables(r3, depth, s, v, m, max_level)
    s2 = D.table_survival(rules0, r2, depth, s, v, m, max_level, truth_a=t0, truth_b=t2)
    s3 = D.table_survival(rules0, r3, depth, s, v, m, max_level, truth_a=t0, truth_b=t3)
    assert s2["2"]["survival"] < 1.0, "T-2: a level-2 resample left T2 intact"
    assert s2["3"]["survival"] < 1.0, "T-2: a level-2 resample left T3 intact"
    assert s3["2"]["survival"] == 1.0, "T-2: a level-3 resample moved T2"
    assert s3["3"]["survival"] < 1.0, "T-2: a level-3 resample left T3 intact"
    out["T2_level_indexing"] = {"drift_L2": s2, "drift_L3": s3}

    # --- T-3 ---------------------------------------------------------------------------
    before = MC.grade_table(t0[2], t0[2])
    after = MC.grade_table(t0[2], t2[2])                 # the FROZEN table, new world
    assert before["precision"] == 1.0, "T-3: the true table is not perfect in its own world"
    assert after["precision"] < 1.0, (
        "T-3: a frozen perfect table stayed perfect after a drift event -- there is no live "
        "gap and the round has nothing to measure")
    out["T3_post_commit_invalidation"] = {
        "precision_before": before["precision"], "precision_after": after["precision"],
        "recall_after": after["recall"],
        "entries_invalidated": before["n_correct"] - after["n_correct"]}

    # --- T-4 ---------------------------------------------------------------------------
    stream = np.array([[a, b] for a in range(v) for b in range(v)][:24], np.int64)
    plain, dec1 = MC.Miner(2, s), D.DecayMiner(2, s, decay=1.0)
    for _ in range(6):
        plain.observe(stream); dec1.age(); dec1.observe(stream)
    assert plain.counts == dec1.counts, "T-4: decay 1.0 is not `MC.Miner`"
    assert (plain.build(MC.base_table(v), 3)["child"].tolist()
            == dec1.build(MC.base_table(v), 3)["child"].tolist()), "T-4: tables differ"
    dec2 = D.DecayMiner(2, s, decay=0.8)
    for _ in range(6):
        dec2.age(); dec2.observe(stream)
    n_before = len(dec2.counts)
    for _ in range(40):
        dec2.age()
    assert len(dec2.counts) == 0 and dec2.n_forgotten == n_before, (
        "T-4: a decaying miner did not forget a tuple the world stopped producing")
    out["T4_forgetting"] = {"identical_at_decay_1": True, "n_before": n_before,
                            "n_forgotten": int(dec2.n_forgotten)}

    # --- T-5 ---------------------------------------------------------------------------
    off = {"v": v, "s": s, "depth": depth, "m": m, "drift_every": 0, "drift_cells": 1,
           "drift_level": 2, "drift_start": 6, "drift_seed": 0}
    traj = D.rules_trajectory(rules0, off, 90)
    assert len(traj) == 1 and all(np.array_equal(a, b)
                                  for a, b in zip(traj[0]["rules"], rules0)), (
        "T-5: drift off is not a no-op")
    on = {**off, "drift_every": 5}
    traj_on = D.rules_trajectory(rules0, on, 90)
    assert len(traj_on) == 1 + (90 - 6) // 5 + 1, "T-5: wrong epoch count"
    assert D.epoch_of(5, on) == 0 and D.epoch_of(6, on) == 1 and D.epoch_of(10, on) == 1
    out["T5_off_is_noop"] = {"epochs_off": len(traj), "epochs_on": len(traj_on)}
    return out


def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3, eras="1:6,2:3,3:1"):
    """`recital`'s gates (which carry `ear`'s C-M / C-R / G-D / G-N / G-S / G-R and its own
    G-P), plus this round's G-T."""
    out = recital_selfcheck(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, n=n,
                            max_level=max_level, eras=eras)
    out["drift"] = gate_drift(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed,
                              max_level=max_level)
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    out = selfcheck()
    print("G-T (the drift): " + json.dumps(out["drift"], indent=2, cls=NumpyEncoder))
    return out


@app.local_entrypoint()
def main(quick: bool = True):
    transpose.remote(tag="smoke0", quick=quick, drift_every=3)
