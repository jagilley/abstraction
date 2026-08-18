"""merge -- invariance by ENUMERATION vs invariance by MERGE, and whether the unmetered
monolith buys the quotient.

Round 8 of the practice arc on RHM. `setlist` drifted WHAT IS ASKED and found the maintenance
organ for demand-news is evaluative re-selection. This round drifts the same thing for a
different purpose: to manufacture a confound and then pay it down.

THE WORLD (`library.py` carries the primitives; `../setlist/demand.py` is imported unmodified).
Every task instance is announced with two FREE observables:

  * a VENUE `w`, each carrying its own OU demand state -- private root prior, private mixture
    weights on the rule layers above the mined vocabulary. Under narrow demand (sigma large) a
    venue concentrates, so `w` is a nearly-free name for the level-2 feature the damage
    destroyed. That correlation is not in the DGP; it is an artefact of the sampling policy,
    which is exactly the debt side of allocation. PACED ROTATION (an OU event every `period`
    cycles) moves every venue's taste, so a table frozen per venue goes out of fashion and the
    accumulated per-venue tables converge -- manufactured decorrelation, at a paced rate.
  * a NODE `j` -- which level-2 node the damage hit. The frame.

The key is `(w, j)`. The index is a partition of the key space and MERGE is the op that
coarsens it.

ARMS differ only in (index policy) x (meter, demand):

    never_base   metered, venue demand, no library                the arc's floor
    given        metered, venue demand, the DGP's own level-2 table
    track        metered, venue demand, finest index forever      ENUMERATION
    merge        metered, venue demand, forced-transfer merge     the QUOTIENT, earned
    merge_pool   as `merge`, evidence pooled at merge time
    global       metered, venue demand, one cell from birth       the quotient handed over
    dense        UNMETERED, uniform demand, no library            the monolith / AlphaZero analog
    dense_wd     as `dense` with a capacity meter (weight decay)

THE READOUTS, in the spec's priority order:
  1. in-distribution parity -- `dense` should match or beat everything on the varied set. If it
     does not, the setup is miscalibrated, not the theory vindicated.
  2. transforms OUTSIDE the varied set -- the fifth wall: a level-2 node never damaged and a
     venue never visited, so a keyed library takes a genuine CACHE MISS and the only move is
     executing an existing entry under a transformed frame (Sec 5's evidence route b).
  3. next-level minability -- a ratchet-style mining pass on top of each arm's representation:
     `T[3]` is defined over `T[2]` ENTRIES, so a thin level-2 vocabulary makes level 3
     unrepresentable rather than merely worse.
  4. the parent doc's own Sec 8 instruments -- the scaffold's value (early learning speed with
     the spurious key vs without), which frozen features bind as a function of loudness, key
     granularity over time merge-vs-track across the rate sweep, the never-merge arm's carried
     costs.

Run from experiments/:
  modal run rhm/practice/merge/merge.py::selfcheck_remote
  modal run rhm/practice/merge/merge.py::cal_loud_remote
  python3 rhm/practice/merge/launch_detached.py --fn merge --tag mg_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.practice.crystallize.units import build_move_set, grade, on_grammar_rate, oracle_rollout
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    beam_moves, build_shared, context_instances, finetune_generator, fit_width, push,
    value_steps)
from rhm.practice.setlist import demand as DM
from rhm.practice.merge import library as LIB


app = modal.App("rhm-practice-merge", image=image)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = dict(
        # substrate (ratchet's, unchanged)
        v=8, s=2, depth=4, m=2, rule_seed=0, train_seed=1, seed=0, state_dim=96,
        n_train_episodes=100_000, controller_steps=12_000, generator_steps=12_000,
        reader_steps=6_000, plant_holdout=0, holdout_seed=11,
        value_steps=12_000, value_episodes=40_000, value_batch_collect=1024, batch_size=256,
        value_lr=3e-4, value_lr_online=3e-5, n_corrupt=3, explore_eps=0.3,
        budget=4, pr_width=16, g_budget=58, max_macro_level=2,
        n_grad=4, value_batch=256, replay_frac=0.5, buf_cap=100_000,
        gen_lr=1e-4, gen_steps=20, mine_support=3, probe_support=3, mine_cap=0,
        weight_decay=1e-4,
        d_fb=1.0, c_mat=0.05,
        # the world
        level=2, n_venue=6, train_nodes=(1, 2, 3), hold_node=0,
        demand_levels=(0, 1), sigma=3.0, kappa=0.15, rotate_every=4, rotate_start=4,
        demand_seed=0, demand_n_cand=0, demand_n=2048, l3_node=1,
        # the loop
        cycles=60, n_pr=72, n_rt=96, dense_mult=4,
        merge_every=8, merge_start=16, merge_tau=0.95, merge_probe=32, merge_min_obs=8,
        # the exams
        n_ex=192, n_probe=48, n_e4=1536, n_mine_post=1536, probe_every=6, checkpoint_every=5,
    )
    cfg.update(kw)
    return cfg


ARMS = {
    "never_base": {"index": None,     "vocab": "base",   "demand": "venue"},
    "given":      {"index": None,     "vocab": "true",   "demand": "venue"},
    "track":      {"index": "track",  "vocab": "earned", "demand": "venue"},
    "merge":      {"index": "merge",  "vocab": "earned", "demand": "venue", "rep": "rep"},
    "merge_pool": {"index": "merge",  "vocab": "earned", "demand": "venue", "rep": "pool"},
    "global":     {"index": "global", "vocab": "earned", "demand": "venue"},
    "dense":      {"index": None,     "vocab": "base",   "demand": "uniform", "mult": True},
    "dense_wd":   {"index": None,     "vocab": "base",   "demand": "uniform", "mult": True,
                   "wd": 1e-2},
    "dense_glob": {"index": "global", "vocab": "earned", "demand": "uniform", "mult": True},
}


def parse_arms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        base, *rest = part.split(":")
        ov = {}
        for kv in rest:
            k, val = kv.split("=")
            try:
                ov[k] = int(val)
            except ValueError:
                try:
                    ov[k] = float(val)
                except ValueError:
                    ov[k] = val
        label = base + ("" if not ov else "_" + "_".join(f"{k}{x}" for k, x in ov.items()))
        out.append((label, base, ov))
    return out


# --------------------------------------------------------------------------- #
# the world: venues, keys, and the shared exam battery
# --------------------------------------------------------------------------- #

def build_world(shared, cfg):
    """Venue demand states + the exam instances every arm is graded on.

    EVERY exam set is drawn ONCE and shared by every arm, so no cross-arm comparison rests on
    a re-draw. The three exams are the spec's readouts 1-3:

      E1 HOME     the arm's own turf: instances from each training venue's CURRENT demand at
                  each training node, redrawn at the end of the run so the metered arms are
                  graded on the demand they are actually facing. Where concentration --
                  and hence enumeration -- should look best.
      E2 ENVELOPE uniform demand at the training nodes, announced under each training venue.
                  This is "the varied set": the union of what the rotation exercises, which is
                  what maximally-varied demand converges to, and it is exactly the
                  distribution the UNMETERED arm trains on -- so readout 1's parity gate is
                  read here, biased in the monolith's favour by construction.
      E3 WALL5    uniform demand at the HELD-OUT node, announced under the HELD-OUT venue.
                  Outside every arm's varied set. A keyed library takes a cache miss.
      E3b WALL5n  same held-out node, announced under a TRAINING venue -- separates "new frame"
                  from "new key".
    """
    rules, s, depth, v, m = shared["rules"], cfg["s"], cfg["depth"], cfg["v"], cfg["m"]
    inv_bottom = shared["inverse_maps"][-1]
    venues, vinfo = LIB.build_venues(
        rules, list(cfg["demand_levels"]), cfg["n_venue"] + 1, s, depth, v, m, inv_bottom,
        seed=cfg["demand_seed"], kappa=cfg["kappa"], sigma=cfg["sigma"],
        n_cand=cfg["demand_n_cand"], n_hist=cfg["demand_n"], level=cfg["level"])
    uniform = DM.new_demand(rules, list(cfg["demand_levels"]), seed=cfg["demand_seed"],
                            kappa=cfg["kappa"], sigma=0.0, prewarm=False)
    keys = [(w, j) for w in range(cfg["n_venue"]) for j in cfg["train_nodes"]]

    def ctx(node):
        return {"name": f"L{cfg['level']}n{node}", "level": cfg["level"], "nodes": [node]}

    exams = {}
    for label, node_set, ven, st_of in (
            ("E2", cfg["train_nodes"], list(range(cfg["n_venue"])), lambda w: uniform),
            ("E3", (cfg["hold_node"],), [cfg["n_venue"]], lambda w: uniform),
            ("E3b", (cfg["hold_node"],), list(range(cfg["n_venue"])), lambda w: uniform)):
        cells = {}
        for w in ven:
            for j in node_set:
                r_np, x_np = DM.context_instances_demand(
                    rules, ctx(j), cfg["n_ex"], s, depth, v, m,
                    seed=cfg["seed"] + 400_000 + 977 * w + 31 * j + 7 * len(label), st=st_of(w))
                cells[(w, j)] = (r_np, x_np)
        exams[label] = cells

    # The next-level probe pool: level-3 damage, uniform demand, shared by every arm.
    # `l3_node` is chosen so the level-3 span covers only TRAINING level-2 nodes -- otherwise
    # readout 3 (minability) would be silently contaminated by readout 2 (the fifth wall).
    l3_node = cfg["l3_node"]
    r3, x3 = DM.context_instances_demand(
        rules, {"name": f"L3n{l3_node}", "level": 3, "nodes": [l3_node]}, cfg["n_e4"], s, depth,
        v, m, seed=cfg["seed"] + 510_000, st=uniform)
    # the post-hoc mining pool: level-2 damage, uniform demand, training nodes, shared.
    mine_pool = {}
    for j in cfg["train_nodes"]:
        rm, xm = DM.context_instances_demand(
            rules, ctx(j), cfg["n_mine_post"] // len(cfg["train_nodes"]), s, depth, v, m,
            seed=cfg["seed"] + 610_000 + 53 * j, st=uniform)
        mine_pool[j] = (rm, xm)

    # The SHARED level-3 observation set (an oracle instrument, never consumed by an arm).
    # `n_t3_own` conflates two things -- what an arm's T[2] makes representable, and how many
    # level-3 spans that arm managed to solve and therefore observe. Holding the observations
    # fixed and varying only the lower table isolates the first: `n_t3_shared` is a pure
    # statement about the nesting constraint.
    _r3c, lv3 = DM.sample_pool_demand(rules, cfg["n_e4"], s, cfg["seed"] + 520_000, uniform)
    f3 = MC.exact_features(lv3, inv_bottom, v, s)
    span3 = s ** 2
    shared_m3 = MC.Miner(3, s)
    shared_m3.observe(f3[:, l3_node * span3:(l3_node + 1) * span3])

    joint = LIB.venue_latent_joint(rules, venues[:cfg["n_venue"]], list(cfg["train_nodes"]),
                                   cfg["level"], s, depth, v, m, inv_bottom,
                                   n=cfg["demand_n"], seed=cfg["seed"] + 11)
    return {"venues": venues, "uniform": uniform, "keys": keys, "exams": exams,
            "l3": (r3, x3, l3_node), "mine_pool": mine_pool, "ctx": ctx,
            "shared_m3": shared_m3, "venue_info": vinfo,
            "i_wf_init": LIB.mutual_info(joint), "epochs": []}


def rotate_world(world, cfg, rng):
    LIB.rotate_venues(world["venues"], rng, 1)
    return world


# --------------------------------------------------------------------------- #
# action sets and grading
# --------------------------------------------------------------------------- #

def group_ms(base_ms, table, level, s, depth, device):
    """base level-1 moves + the level-`level` macro instantiated at EVERY node of that level.

    Position-independence is the arc's convention (`ratchet.macro_moves`) and it is what makes
    the merge measurement about the INDEX only: the table itself carries no address, so a
    committed unit cannot be contaminated by the key even in principle."""
    ms = list(base_ms)
    if table is not None and table["child"].shape[0] > 0:
        ms += [MC.to_device(MC.make_macro(level, j, s, table), device)
               for j in range(s ** (depth - level))]
    return ms


def solve(shared, cfg, generator, value, table, r_np, x_np, device, *, width=None,
          level=None, collect=False):
    """One priced solve of a batch with a given vocabulary. Returns the graded outcome and the
    beam's own answer, so mining reads the agent's PERFORMANCE and not its search tree."""
    import torch
    level = level or cfg["level"]
    ms = group_ms(shared["base_ms"], table, level, cfg["s"], cfg["depth"], device)
    w = width or fit_width(len(ms), cfg["budget"], cfg["g_budget"])
    out = beam_moves(shared["controller"], generator, value, torch.from_numpy(x_np),
                     torch.from_numpy(r_np), ms, shared["rules_t"], shared["canon"],
                     cfg["depth"], cfg["v"], cfg["m"], cfg["s"], budget=cfg["budget"],
                     beam_width=w, device=device, collect=collect)
    succ, dres = grade(out["x"].cpu().numpy(), r_np, shared["rules"], cfg["s"])
    out["e"] = 1.0 - float(succ.mean())
    out["dres"] = float(dres.mean())
    out["succ"] = succ
    out["n_moves"] = len(ms)
    out["width"] = w
    return out


def exam(shared, cfg, generator, value, lib, true_tbl, cells, device, miss_range=False):
    """Grade an exam cell-by-cell. `lib` None means the arm has no index (base or `given`).

    On a key the index has no cell for, `KeyedLibrary.fallback_cell` reaches for the cell with
    the most evidence -- there is no principled choice, and `miss_range` additionally reports
    what the BEST and WORST available cells would have scored, so the cache miss is a range
    rather than a point estimate the fallback rule happens to pick out."""
    per, misses = {}, 0
    rng_lo, rng_hi = [], []
    for key, (r_np, x_np) in cells.items():
        if lib is None:
            tbl = true_tbl
        else:
            tbl, _cell = lib.table_for(key)
            if lib.cell_of(key) is None:
                misses += 1
        o = solve(shared, cfg, generator, value, tbl, r_np, x_np, device)
        per[str(key)] = {"e": o["e"], "dres": o["dres"], "n_moves": o["n_moves"],
                         "width": o["width"]}
        if miss_range and lib is not None and lib.cell_of(key) is None:
            es = []
            for cid, cell in lib.cells.items():
                if cell["table"] is None or cell["table"]["child"].shape[0] == 0:
                    continue
                es.append(solve(shared, cfg, generator, value, cell["table"], r_np, x_np,
                                device)["e"])
            if es:
                rng_lo.append(float(min(es))); rng_hi.append(float(max(es)))
    out = {"e": float(np.mean([c["e"] for c in per.values()])),
           "dres": float(np.mean([c["dres"] for c in per.values()])),
           "per_key": per, "n_miss": misses, "n_keys": len(cells)}
    if rng_lo:
        out["e_best_cell"] = float(np.mean(rng_lo))
        out["e_worst_cell"] = float(np.mean(rng_hi))
    return out


# --------------------------------------------------------------------------- #
# readout 3: next-level minability
# --------------------------------------------------------------------------- #

def minability(shared, cfg, generator, value, tables, device):
    """Run a ratchet-style mining pass on top of a level-2 vocabulary and read what it makes
    representable next.

    `tables` is a dict {name: level-2 table}. For each: solve the SHARED level-3 probe pool
    with (base moves + that table's macro), read the agent's own parse of what it solved,
    observe the level-3 span, and build `T[3]` OVER THAT TABLE. `MC.Miner.build` drops any
    level-3 span whose halves are not both entries of the lower table -- the ratchet's nesting,
    which is why this is a claim about representability and not about accuracy."""
    import torch
    r3, x3, node3 = shared["_l3"]
    s, v, m, depth = cfg["s"], cfg["v"], cfg["m"], cfg["depth"]
    span3 = s ** 2
    out = {}
    for name, t2 in tables.items():
        if t2 is None:
            t2 = MC.base_table(v)
        o = solve(shared, cfg, generator, value, t2, r3, x3, device, level=2)
        solved = o["x"][torch.from_numpy(o["succ"] > 0.5).to(device)]
        mn3 = MC.Miner(3, s)
        if solved.shape[0]:
            pf = MC.parse_features(shared["reader"], solved, s=s).cpu().numpy()
            mn3.observe(pf[:, node3 * span3:(node3 + 1) * span3])
        t3 = LIB.next_level(t2, mn3, cfg["probe_support"])
        t3s = LIB.next_level(t2, shared["_shared_m3"], cfg["probe_support"])
        rec = {"n_t2": int(t2["child"].shape[0]), "e_l3_solve": o["e"],
               "n_solved": int(solved.shape[0]), "n_obs3": int(mn3.n_obs),
               "n_distinct3": len(mn3.counts), "n_t3": int(t3["child"].shape[0]),
               "n_t3_shared": int(t3s["child"].shape[0])}
        rec.update({f"t3_{k}": val for k, val in
                    MC.grade_table(t3, shared["truth3"]).items()})
        rec.update({f"t3s_{k}": val for k, val in
                    MC.grade_table(t3s, shared["truth3"]).items()})
        for nm, tt in (("e_t3_macro", t3), ("e_t3s_macro", t3s)):
            if tt["child"].shape[0]:
                mv = MC.to_device(MC.make_macro(3, node3, s, tt), device)
                xf = MC.apply_any(generator, torch.from_numpy(x3).to(device), mv,
                                  shared["rules_t"], shared["canon"], depth, v, m, s)
                su, _ = grade(xf.cpu().numpy(), r3, shared["rules"], s)
                rec[nm] = 1.0 - float(su.mean())
            else:
                rec[nm] = None
        mvt = MC.to_device(MC.make_macro(3, node3, s, shared["truth3"]), device)
        xf = MC.apply_any(generator, torch.from_numpy(x3).to(device), mvt, shared["rules_t"],
                          shared["canon"], depth, v, m, s)
        su, _ = grade(xf.cpu().numpy(), r3, shared["rules"], s)
        rec["e_t3_true"] = 1.0 - float(su.mean())
        out[name] = rec
    return out


def post_mine_t2(shared, cfg, generator, value, table, device):
    """The level-2 vocabulary a representation yields when you mine its own solved repairs.

    For the UNMETERED arm this is the only way to read a vocabulary off it at all -- it holds no
    library -- and it is deliberately generous: the pool is the same for every arm and is drawn
    from uniform demand, i.e. the monolith's home distribution."""
    import torch
    s = cfg["s"]
    span = s ** (cfg["level"] - 1)
    mn = MC.Miner(cfg["level"], s)
    e = []
    for j, (r_np, x_np) in shared["_mine_pool"].items():
        o = solve(shared, cfg, generator, value, table, r_np, x_np, device)
        e.append(o["e"])
        solved = o["x"][torch.from_numpy(o["succ"] > 0.5).to(device)]
        if solved.shape[0]:
            pf = MC.parse_features(shared["reader"], solved, s=s).cpu().numpy()
            mn.observe(pf[:, j * span:(j + 1) * span])
    t2 = mn.build(MC.base_table(cfg["v"]), cfg["probe_support"])
    return t2, {"e_mine_pool": float(np.mean(e)), "n_obs": int(mn.n_obs),
                "n_distinct": len(mn.counts), "n_t2": int(t2["child"].shape[0]),
                **{f"t2_{k}": val for k, val in MC.grade_table(t2, shared["truth"][2]).items()}}


def union_table(lib, v, s, support):
    """The generous upper bound on a keyed arm: what its index WOULD hold if every cell were
    merged for free at the end. Separates "the index cost you the next level" from "the arm
    never saw enough data"."""
    mn = MC.Miner(lib.level, s)
    for cell in lib.cells.values():
        for k, n in cell["miner"].counts.items():
            mn.counts[k] = mn.counts.get(k, 0) + n
        mn.n_obs += cell["miner"].n_obs
    return mn.build(MC.base_table(v), support)


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

def run_arm(label, base, overrides, shared, cfg, world, outdir, device):
    import torch
    spec = ARMS[base]
    cfg = {**cfg, **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    level = cfg["level"]
    mult = cfg["dense_mult"] if spec.get("mult") else 1
    wd = spec.get("wd", cfg["weight_decay"])

    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=wd)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr_online"], weight_decay=wd)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)
    drng = np.random.default_rng(cfg["demand_seed"] + 4242)

    lib = None
    if spec["index"]:
        lib = LIB.KeyedLibrary(world["keys"], level, s, policy=spec["index"],
                               merge_rep=spec.get("rep", "rep"))
    true_tbl = shared["truth"][level] if spec["vocab"] == "true" else None

    # a PRIVATE copy of the venue states: every arm sees the same rotation schedule from the
    # same start, so the demand path is a property of the world and not of arm order.
    venues = [DM.copy_demand(st) for st in world["venues"]]
    uni = world["uniform"]

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    log = {"cycle": [], "epoch": [], "venue": [], "e": [], "dres": [], "e_practice": [],
           "n_cells": [], "storage": [], "n_solved": [], "vloss": [], "gloss": [],
           "n_moves": [], "width": [], "n_entries": [], "probe": [], "merge": []}
    events = []
    epoch = 0
    t0 = time.time()

    for cyc in range(1, cfg["cycles"] + 1):
        # ---- (a) paced rotation: EVERY venue's taste moves, unannounced --------------------
        if (cfg["rotate_every"] and cyc >= cfg["rotate_start"]
                and (cyc - cfg["rotate_start"]) % cfg["rotate_every"] == 0):
            LIB.rotate_venues(venues, drng, 1)
            epoch += 1

        # ---- (b) practice. Metered arms drill ONE venue per cycle (allocation manufactures
        #          recurrence: the venue is held fixed while the drill runs). The unmetered arm
        #          draws from uniform demand -- free i.i.d. variation, no manufactured
        #          recurrence -- at `dense_mult` times the instance count and gradient steps.
        w_cyc = (cyc - 1) % cfg["n_venue"]
        per_node = max(1, (cfg["n_pr"] * mult) // len(cfg["train_nodes"]))
        solved_all, e_pr, mined = [], [], 0
        for j in cfg["train_nodes"]:
            st = uni if spec["demand"] == "uniform" else venues[w_cyc]
            key = (w_cyc, j)
            r_np, x_np = DM.context_instances_demand(
                shared["rules"], world["ctx"](j), per_node, s, depth, v, m,
                seed=cfg["seed"] + 100_000 + 1000 * cyc + 13 * j, st=st)
            tbl = true_tbl if lib is None else lib.table_for(key)[0]
            o = solve(shared, cfg, generator, value, tbl, r_np, x_np, device,
                      width=cfg["pr_width"], collect=True)
            e_pr.append(o["e"])
            ok = torch.from_numpy(o["succ"] > 0.5).to(device)
            chosen = o["x"][ok]
            solved_all.append(chosen)
            tips = o["tips_x"]
            B, W, T = tips.shape
            tflat = tips.reshape(B * W, T)
            tsucc, _ = grade(tflat.cpu().numpy(), np.repeat(r_np, W), shared["rules"], s)
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in o["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(o["traj"])))
            ys = torch.from_numpy(np.tile(tsucc.astype(np.float32), len(o["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            # ---- (c) mining, into the cell this key belongs to ---------------------------
            if lib is not None and chosen.shape[0]:
                src = chosen
                if cfg["mine_cap"] and src.shape[0] > cfg["mine_cap"]:
                    sel = torch.from_numpy(rng.permutation(src.shape[0])[:cfg["mine_cap"]])
                    src = src[sel.to(device)]
                pf = MC.parse_features(shared["reader"], src, s=s).cpu().numpy()
                span = s ** (level - 1)
                lib.observe(key, pf[:, j * span:(j + 1) * span])
                mined += src.shape[0]

        solved = torch.cat(solved_all) if solved_all else torch.zeros(
            0, shared["length"], dtype=torch.long, device=device)
        if lib is not None:
            lib.rebuild(MC.base_table(v), cfg["mine_support"])

        # ---- (d) the plant learns, then the selector --------------------------------------
        gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                   shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                                   n_steps=cfg["gen_steps"] * mult, batch=cfg["batch_size"],
                                   replay_frac=cfg["replay_frac"], device=device, rng=grng)
        vloss = value_steps(value, vopt, shared["controller"], buf, shared["replay"],
                            n_steps=cfg["n_grad"] * mult, batch=cfg["value_batch"],
                            replay_frac=cfg["replay_frac"], device=device, rng=rng)

        # ---- (e) THE MERGE PASS ----------------------------------------------------------
        mrec = None
        if (lib is not None and spec["index"] == "merge" and cyc >= cfg["merge_start"]
                and cyc % cfg["merge_every"] == 0):
            probe_cache = {}

            def score_fn(table, key):
                # the probe instances are drawn ONCE per key per pass: the transfer matrix has
                # to compare tables against the SAME demand sample, or an off-diagonal entry
                # would differ from its diagonal for a sampling reason.
                if key not in probe_cache:
                    w, j = key
                    st = uni if spec["demand"] == "uniform" else venues[w]
                    probe_cache[key] = DM.context_instances_demand(
                        shared["rules"], world["ctx"](j), cfg["merge_probe"], s, depth, v, m,
                        seed=cfg["seed"] + 800_000 + 1000 * cyc + 97 * w + 7 * j, st=st)
                r_np, x_np = probe_cache[key]
                return 1.0 - solve(shared, cfg, generator, value, table, r_np, x_np,
                                   device)["e"]
            before = lib.n_cells()
            mrec = LIB.merge_pass(lib, score_fn, tau=cfg["merge_tau"],
                                  min_obs=cfg["merge_min_obs"])
            mrec["n_cells_before"] = before
            if mrec["n_merges"]:
                events.append({"kind": "merge", "cycle": cyc, "n_merges": mrec["n_merges"],
                               "n_cells_before": before, "n_cells_after": lib.n_cells()})
                print(f"  [merge] arm={label} c{cyc}: {before} -> {lib.n_cells()} cells "
                      f"({mrec['n_merges']} merges)", flush=True)
            lib.rebuild(MC.base_table(v), cfg["mine_support"])

        # ---- (f) metering: the arm's own turf at the current demand ------------------------
        em, dm, nmv, wid = [], [], 0, 0
        for j in cfg["train_nodes"]:
            st = uni if spec["demand"] == "uniform" else venues[w_cyc]
            key = (w_cyc, j)
            r_np, x_np = DM.context_instances_demand(
                shared["rules"], world["ctx"](j), max(16, cfg["n_rt"] // len(cfg["train_nodes"])),
                s, depth, v, m, seed=cfg["seed"] + 200_000 + 1000 * cyc + 13 * j, st=st)
            tbl = true_tbl if lib is None else lib.table_for(key)[0]
            o = solve(shared, cfg, generator, value, tbl, r_np, x_np, device)
            em.append(o["e"]); dm.append(o["dres"]); nmv, wid = o["n_moves"], o["width"]

        # ---- (g) the standing probes (instruments, never priced) ---------------------------
        probe = None
        if cyc % cfg["probe_every"] == 0 or cyc == cfg["cycles"]:
            def sub(cells):
                k = cfg["n_probe"]
                return {key: (r[:k], x[:k]) for key, (r, x) in cells.items()}
            probe = {
                "E2": exam(shared, cfg, generator, value, lib, true_tbl,
                           sub(world["exams"]["E2"]), device)["e"],
                "E3": exam(shared, cfg, generator, value, lib, true_tbl,
                           sub(world["exams"]["E3"]), device)["e"],
            }

        n_ent = 0 if lib is None else lib.storage()
        log["cycle"].append(cyc); log["epoch"].append(epoch); log["venue"].append(int(w_cyc))
        log["e"].append(float(np.mean(em))); log["dres"].append(float(np.mean(dm)))
        log["e_practice"].append(float(np.mean(e_pr)))
        log["n_cells"].append(0 if lib is None else lib.n_cells())
        log["storage"].append(n_ent); log["n_solved"].append(int(solved.shape[0]))
        log["vloss"].append(vloss); log["gloss"].append(gloss)
        log["n_moves"].append(int(nmv)); log["width"].append(int(wid))
        log["n_entries"].append(n_ent); log["probe"].append(probe); log["merge"].append(mrec)
        if cyc % 5 == 0 or cyc == 1:
            print(f"  arm={label} c{cyc:3d} ep{epoch} v{w_cyc} e={np.mean(em):.4f} "
                  f"cells={log['n_cells'][-1]} store={n_ent} solved={int(solved.shape[0])}",
                  flush=True)

    # ---- the final exams ------------------------------------------------------------------
    print(f"  arm={label}: exams", flush=True)
    home = {}
    for w in range(cfg["n_venue"]):
        for j in cfg["train_nodes"]:
            r_np, x_np = DM.context_instances_demand(
                shared["rules"], world["ctx"](j), cfg["n_ex"], s, depth, v, m,
                seed=cfg["seed"] + 900_000 + 977 * w + 31 * j,
                st=(uni if spec["demand"] == "uniform" else venues[w]))
            home[(w, j)] = (r_np, x_np)
    res = {"E1": exam(shared, cfg, generator, value, lib, true_tbl, home, device),
           "E2": exam(shared, cfg, generator, value, lib, true_tbl,
                      world["exams"]["E2"], device),
           "E3": exam(shared, cfg, generator, value, lib, true_tbl,
                      world["exams"]["E3"], device, miss_range=True),
           "E3b": exam(shared, cfg, generator, value, lib, true_tbl,
                       world["exams"]["E3b"], device)}

    # readout 3, on every reading of "the arm's representation" we can defend
    tables = {}
    if lib is None:
        t2_post, post_rec = post_mine_t2(shared, cfg, generator, value, true_tbl, device)
        tables["post"] = t2_post
        if true_tbl is not None:
            tables["held"] = true_tbl
    else:
        fb = lib.fallback_cell()
        tables["fallback"] = None if fb is None else lib.cells[fb]["table"]
        tables["union"] = union_table(lib, v, s, cfg["mine_support"])
        t2_post, post_rec = post_mine_t2(shared, cfg, generator, value,
                                         tables["fallback"], device)
        tables["post"] = t2_post
    res["minability"] = minability(shared, cfg, generator, value, tables, device)
    res["post_mine"] = post_rec

    # the index, and its carried costs
    if lib is not None:
        cellrec = []
        for cid, cell in lib.cells.items():
            cellrec.append({"cell": cid, "keys": sorted(str(k) for k in cell["keys"]),
                            "n_obs": int(cell["miner"].n_obs),
                            "n_distinct": len(cell["miner"].counts),
                            "n_entries": (0 if cell["table"] is None
                                          else int(cell["table"]["child"].shape[0])),
                            **({} if cell["table"] is None else
                               {f"tab_{k}": val for k, val in
                                MC.grade_table(cell["table"], shared["truth"][2]).items()})})
        res["cells"] = cellrec
        res["n_cells"] = lib.n_cells()
        res["storage"] = lib.storage()
        res["merge_history"] = lib.history
        # route (a), the offline audit, for comparison with the forced-transfer route
        ids = sorted(lib.cells)
        res["jaccard"] = {f"{a}|{b}": LIB.alias_jaccard(lib.cells[a]["table"],
                                                        lib.cells[b]["table"])
                          for i, a in enumerate(ids) for b in ids[i + 1:]}
    out = {"arm": label, "base": base, "overrides": overrides, "log": log, "events": events,
           "exam": res, "elapsed": time.time() - t0, "mult": mult, "wd": wd}
    d = os.path.join(outdir, label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder)
    volume.commit()
    print(f"  arm={label} done in {out['elapsed']:.0f}s  "
          f"E1={res['E1']['e']:.4f} E2={res['E2']['e']:.4f} E3={res['E3']['e']:.4f}",
          flush=True)
    return out


# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #

def measure_refs(shared, cfg, world, device):
    """d0 (do-nothing error), the exact-DP floor, and the on-grammar rate, on the E2 envelope.
    The arc's stationarity references: if these move with the rotation, the world got harder
    and nothing below is about demand."""
    import torch
    refs = {}
    cells = world["exams"]["E2"]
    key = sorted(cells)[0]
    r_np, x_np = cells[key]
    su, dr = grade(x_np, r_np, shared["rules"], cfg["s"])
    refs["d0"] = float(dr.mean())
    refs["e0"] = 1.0 - float(su.mean())
    refs["on_grammar"] = on_grammar_rate(x_np, shared["inverse_maps"][-1], cfg["v"], cfg["s"])
    n = min(128, x_np.shape[0])
    xf, _ = oracle_rollout(shared["generator0"], torch.from_numpy(x_np[:n]).to(device),
                           r_np[:n], shared["base_ms"], shared["rules"], shared["rules_t"],
                           shared["canon"], cfg["depth"], cfg["v"], cfg["m"], cfg["s"],
                           cfg["budget"])
    su, _ = grade(xf.cpu().numpy(), r_np[:n], shared["rules"], cfg["s"])
    refs["floor"] = 1.0 - float(su.mean())
    return refs


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_world(v=8, s=2, depth=4, m=2, rule_seed=0, n=4096, sigma=1.0, kappa=0.15,
               n_venue=6, train_nodes=(1, 2, 3), hold_node=0, level=2, demand_levels=(0, 1),
               seed=0):
    """The admissibility gates, offline and substrate-free (setlist's discipline).

      M-1  FIDELITY. At sigma = 0 the venue sampler is DISTRIBUTIONALLY the parent's: same d0,
           same on-grammar rate, and the level-2 demand histogram at a node matches
           `ratchet.context_instances`' to within sampling noise. (Not stream-identical: the
           parent draws `rng.integers`, the weighted sampler `rng.random` -- setlist's D-1.)
      M-2  NOTHING BECOMES FALSE. The true tables are the same object at every epoch, drawn
           instances stay 100% on-grammar, and the DP-relevant difficulty `d0` is stationary
           across rotation events.
      M-3  THE CONFOUND EXISTS. I(venue; level-2 latent) > 0 at sigma > 0 and == 0 at sigma = 0.
      M-4  ROTATION DECORRELATES. The CUMULATIVE joint (pooled over epochs) loses mutual
           information as the rotation proceeds, while the INSTANTANEOUS joint does not -- which
           is the precise sense in which paced rotation pays the confound down.
      M-5  THE FIFTH WALL IS ADMISSIBLE. The level-2 entry mass demanded at the held-out node
           must overlap the mass demanded at training nodes, or nothing could transfer and the
           transform is a measured null by construction (setlist's level-2 lesson).
      M-6  MERGE IS INDEX-ONLY. After a merge the surviving cell's entries are a subset of the
           representative's (rep mode) / of the union (pool mode). No entry is ever a blend.
    """
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)
    inv_bottom = inv[-1]
    span = s ** (level - 1)
    out = {}

    # ---- M-1 -------------------------------------------------------------------------- #
    uni = DM.new_demand(rules, list(demand_levels), seed=seed, kappa=kappa, sigma=0.0,
                        prewarm=False)
    ctx = {"name": f"L{level}n{train_nodes[0]}", "level": level, "nodes": [train_nodes[0]]}
    rp, xp = context_instances(rules, ctx, n // 4, s, depth, v, m, seed=seed + 3)
    rd, xd = DM.context_instances_demand(rules, ctx, n // 4, s, depth, v, m, seed=seed + 3,
                                         st=uni)
    _, d_par = grade(xp, rp, rules, s)
    _, d_dem = grade(xd, rd, rules, s)
    hp = LIB.node_hist(rules, DM.new_demand(rules, list(demand_levels), seed=seed, sigma=0.0),
                       train_nodes[0], level, s, depth, v, m, inv_bottom, n=n, seed=seed + 5)
    hd = LIB.node_hist(rules, uni, train_nodes[0], level, s, depth, v, m, inv_bottom, n=n,
                       seed=seed + 6)
    out["M1"] = {"d0_parent": float(d_par.mean()), "d0_demand": float(d_dem.mean()),
                 "og_parent": on_grammar_rate(xp, inv_bottom, v, s),
                 "og_demand": on_grammar_rate(xd, inv_bottom, v, s),
                 "hist_kl": DM.demand_kl(hd, hp)}
    out["M1"]["pass"] = bool(abs(out["M1"]["d0_parent"] - out["M1"]["d0_demand"]) < 0.15
                             and out["M1"]["og_parent"] == 1.0 == out["M1"]["og_demand"]
                             and out["M1"]["hist_kl"] < 0.02)

    # ---- M-2 / M-3 / M-4 ---------------------------------------------------------------- #
    venues, _ = LIB.build_venues(rules, list(demand_levels), n_venue, s, depth, v, m,
                                 inv_bottom, seed=seed, kappa=kappa, sigma=sigma, level=level)
    zero, _ = LIB.build_venues(rules, list(demand_levels), n_venue, s, depth, v, m, inv_bottom,
                               seed=seed, kappa=kappa, sigma=0.0, level=level)
    truth_a = MC.true_tables(rules, depth, s, v, m, level)[level]
    rng = np.random.default_rng(seed + 17)
    d0s, ogs, i_inst, i_cum = [], [], [], []
    cum = {}
    for ep in range(8):
        if ep:
            LIB.rotate_venues(venues, rng, 1)
        j = train_nodes[0]
        r_np, x_np = DM.context_instances_demand(
            rules, {"name": "g", "level": level, "nodes": [j]}, 512, s, depth, v, m,
            seed=seed + 77 + ep, st=venues[0])
        _, dd = grade(x_np, r_np, rules, s)
        d0s.append(float(dd.mean()))
        ogs.append(on_grammar_rate(x_np, inv_bottom, v, s))
        jt = LIB.venue_latent_joint(rules, venues, list(train_nodes), level, s, depth, v, m,
                                    inv_bottom, n=1024, seed=seed + 400 + ep)
        i_inst.append(LIB.mutual_info(jt))
        for k, c in jt.items():
            cum[k] = cum.get(k, 0) + c
        i_cum.append(LIB.mutual_info(cum))
    truth_b = MC.true_tables(rules, depth, s, v, m, level)[level]
    jt0 = LIB.venue_latent_joint(rules, zero, list(train_nodes), level, s, depth, v, m,
                                 inv_bottom, n=1024, seed=seed + 400)
    out["M2"] = {"truth_identical": bool(np.array_equal(truth_a["flat"], truth_b["flat"])),
                 "d0_by_epoch": d0s, "d0_spread": float(max(d0s) - min(d0s)),
                 "on_grammar_min": float(min(ogs))}
    out["M2"]["pass"] = bool(out["M2"]["truth_identical"] and out["M2"]["on_grammar_min"] == 1.0
                             and out["M2"]["d0_spread"] < 0.35)
    out["M3"] = {"i_sigma": i_inst[0], "i_zero": LIB.mutual_info(jt0)}
    out["M3"]["pass"] = bool(i_inst[0] > 0.10 and out["M3"]["i_zero"] < 0.02)
    out["M4"] = {"i_inst": i_inst, "i_cum": i_cum,
                 "inst_slope": float(np.polyfit(np.arange(len(i_inst)), i_inst, 1)[0]),
                 "cum_slope": float(np.polyfit(np.arange(len(i_cum)), i_cum, 1)[0])}
    out["M4"]["pass"] = bool(out["M4"]["cum_slope"] < out["M4"]["inst_slope"]
                             and i_cum[-1] < i_inst[0])

    # ---- M-5 ---------------------------------------------------------------------------- #
    h_hold = LIB.node_hist(rules, uni, hold_node, level, s, depth, v, m, inv_bottom, n=n,
                           seed=seed + 9)
    train_support = set()
    for j in train_nodes:
        train_support |= set(LIB.node_hist(rules, uni, j, level, s, depth, v, m, inv_bottom,
                                           n=n, seed=seed + 9).keys())
    ov = float(sum(p for k, p in h_hold.items() if k in train_support))
    out["M5"] = {"holdout_mass_covered_by_train_support": ov,
                 "n_hold_entries": len(h_hold), "n_train_entries": len(train_support)}
    out["M5"]["pass"] = bool(ov > 0.5)

    # ---- M-6 ---------------------------------------------------------------------------- #
    keys = [(w, j) for w in range(2) for j in train_nodes[:2]]
    ok = True
    for rep in ("rep", "pool"):
        lib = LIB.KeyedLibrary(keys, level, s, policy="track", merge_rep=rep)
        rr = np.random.default_rng(1)
        for i, k in enumerate(keys):
            lib.observe(k, rr.integers(0, v, size=(24, span)))
        lib.rebuild(MC.base_table(v), 1)
        a, b = sorted(lib.cells)[:2]
        ta = {tuple(int(x) for x in r) for r in lib.cells[a]["table"]["flat"]}
        tb = {tuple(int(x) for x in r) for r in lib.cells[b]["table"]["flat"]}
        lib.merge(a, b, a)
        lib.rebuild(MC.base_table(v), 1)
        tm = {tuple(int(x) for x in r) for r in lib.cells[a]["table"]["flat"]}
        ok = ok and (tm <= ta if rep == "rep" else tm <= (ta | tb))
    out["M6"] = {"pass": bool(ok)}

    out["all_pass"] = bool(all(out[k]["pass"] for k in ("M1", "M2", "M3", "M4", "M5", "M6")))
    return out


@app.function(image=image, timeout=3600, memory=16384)
def gate_world_remote(sigma: float = 1.0, kappa: float = 0.15):
    r = gate_world(sigma=sigma, kappa=kappa)
    print(json.dumps(r, indent=2, cls=NumpyEncoder))
    assert r["all_pass"], "world gate FAILED"
    return r


# --------------------------------------------------------------------------- #
# calibration 1 (offline, zero GPU): loudness x rate
# --------------------------------------------------------------------------- #

def cal_loud(v=8, s=2, depth=4, m=2, rule_seed=0, level=2, n_venue=6, train_nodes=(1, 2, 3),
             demand_levels=(0, 1), sigmas=(0.0, 0.5, 1.0, 1.5, 2.5),
             kappas=(0.15, 0.5), periods=(0, 8, 4, 1), cycles=60, n=1536, seed=0):
    """CALIBRATION 1 -- the loudness knob and the rate axis, SWEPT not solved.

    setlist's lesson, inherited verbatim: binary-searching a knob against a target magnitude
    runs to the bound because the softmax saturates. sigma sets the CONCENTRATION of a venue's
    demand -- hence how loud the free observable is as a name for the latent -- and kappa sets
    the MIXING RATE, hence how fast a per-venue table goes out of fashion. Both are measured.

    Reported per (sigma, kappa, period):
      i_inst    I(venue; level-2 latent) at one instant  -- the AVAILABLE confound
      i_cum     the same on the pooled sample over the whole run -- what paced rotation pays
                the confound down to
      half      cycles for a per-venue table frozen at cycle 0 to lose half its coverage of its
                own venue's current demand (setlist's payback bracket, per venue)
      cov_end   that table's coverage at the end
    """
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv_bottom = build_inverse_maps(rules)[-1]
    rows = []
    for sigma in sigmas:
        for kappa in kappas:
            for period in periods:
                venues, _ = LIB.build_venues(rules, list(demand_levels), n_venue, s, depth, v,
                                             m, inv_bottom, seed=seed, kappa=kappa,
                                             sigma=sigma, level=level)
                rng = np.random.default_rng(seed + 77)
                frozen = {}
                for w, st in enumerate(venues):
                    h = LIB.node_hist(rules, st, train_nodes[0], level, s, depth, v, m,
                                      inv_bottom, n=n, seed=seed + 5)
                    frozen[w] = DM.demand_table(h, level, MC.base_table(v), s, cover=0.8)
                cum, i_inst0, covs = {}, None, []
                n_ep = 0 if period == 0 else cycles // period
                for ep in range(n_ep + 1):
                    if ep:
                        LIB.rotate_venues(venues, rng, 1)
                    jt = LIB.venue_latent_joint(rules, venues, list(train_nodes), level, s,
                                                depth, v, m, inv_bottom, n=n // 2,
                                                seed=seed + 400 + ep)
                    if ep == 0:
                        i_inst0 = LIB.mutual_info(jt)
                    for k, c in jt.items():
                        cum[k] = cum.get(k, 0) + c
                    cv = [DM.coverage(frozen[w],
                                      LIB.node_hist(rules, venues[w], train_nodes[0], level, s,
                                                    depth, v, m, inv_bottom, n=n,
                                                    seed=seed + 5))
                          for w in range(n_venue)]
                    covs.append(float(np.mean(cv)))
                half = next((i * max(period, 1) for i, c in enumerate(covs)
                             if c < 0.5 * covs[0]), None)
                rows.append({"sigma": sigma, "kappa": kappa, "period": period,
                             "i_inst": i_inst0, "i_cum": LIB.mutual_info(cum),
                             "i_cum_frac": (LIB.mutual_info(cum) / i_inst0) if i_inst0 else None,
                             "cov0": covs[0], "cov_end": covs[-1], "half_life": half,
                             "n_epochs": n_ep})
                print(f"  sigma={sigma:.2f} kappa={kappa:.2f} period={period}: "
                      f"i_inst={rows[-1]['i_inst']:.3f} i_cum={rows[-1]['i_cum']:.3f} "
                      f"cov {covs[0]:.3f}->{covs[-1]:.3f} half={half}", flush=True)
    return rows


@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=16384)
def cal_loud_remote(tag: str = "cal_loud", sigmas: str = "0.0,0.5,1.0,1.5,2.5",
                    kappas: str = "0.15,0.5", periods: str = "0,8,4,1", cycles: int = 60):
    rows = cal_loud(sigmas=tuple(float(x) for x in sigmas.split(",")),
                    kappas=tuple(float(x) for x in kappas.split(",")),
                    periods=tuple(int(x) for x in periods.split(",")), cycles=cycles)
    d = f"{DATA_DIR}/rhm_practice_merge/{tag}"
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cal_loud.json"), "w") as fh:
        json.dump(rows, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    return rows


# --------------------------------------------------------------------------- #
# the main run
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def merge(
    tag: str = "smoke",
    arms: str = ("never_base,given,global,track,merge,merge_pool,"
                 "dense,dense_wd,dense_glob"),
    seed: int = 0, cycles: int = 60, n_venue: int = 6, train_nodes: str = "1,2,3",
    hold_node: int = 0, sigma: float = 3.0, kappa: float = 0.15, rotate_every: int = 4,
    rotate_start: int = 4, demand_levels: str = "0/1", demand_seed: int = 0,
    demand_n_cand: int = 0, l3_node: int = 1, n_pr: int = 72, n_rt: int = 96,
    dense_mult: int = 4,
    merge_every: int = 8, merge_start: int = 16, merge_tau: float = 0.95,
    merge_probe: int = 32, merge_min_obs: int = 8, mine_support: int = 3, mine_cap: int = 0,
    budget: int = 4, pr_width: int = 16, g_budget: int = 58, n_grad: int = 4,
    gen_steps: int = 20, value_lr_online: float = 3e-5, n_ex: int = 192, n_probe: int = 48,
    n_e4: int = 384, n_mine_post: int = 768, probe_every: int = 6, quick: bool = False,
):
    import torch
    cfg = _cfg(seed=seed, cycles=cycles, n_venue=n_venue,
               train_nodes=tuple(int(x) for x in train_nodes.split(",")),
               hold_node=hold_node, sigma=sigma, kappa=kappa, rotate_every=rotate_every,
               rotate_start=rotate_start,
               demand_levels=tuple(int(x) for x in demand_levels.split("/")),
               demand_seed=demand_seed, demand_n_cand=demand_n_cand, l3_node=l3_node,
               n_pr=n_pr, n_rt=n_rt,
               dense_mult=dense_mult, merge_every=merge_every, merge_start=merge_start,
               merge_tau=merge_tau, merge_probe=merge_probe, merge_min_obs=merge_min_obs,
               mine_support=mine_support, mine_cap=mine_cap, budget=budget,
               pr_width=pr_width, g_budget=g_budget, n_grad=n_grad, gen_steps=gen_steps,
               value_lr_online=value_lr_online, n_ex=n_ex, n_probe=n_probe, n_e4=n_e4,
               n_mine_post=n_mine_post, probe_every=probe_every)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800, reader_steps=600,
                   value_episodes=6_000, n_train_episodes=20_000, cycles=10, n_pr=24, n_rt=48,
                   n_ex=64, n_probe=24, n_e4=96, n_mine_post=192, dense_mult=2, merge_start=2,
                   merge_every=3, merge_probe=24, merge_min_obs=2, n_venue=3, probe_every=5,
                   gen_steps=5, rotate_start=2, rotate_every=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"merge tag={tag} arms={arms} device={device}")
    print(f"  world: {cfg['n_venue']} venues (+1 held out), nodes {cfg['train_nodes']} "
          f"(+{cfg['hold_node']} held out), sigma={cfg['sigma']} kappa={cfg['kappa']} "
          f"rotate every {cfg['rotate_every']} from c{cfg['rotate_start']}")

    shared = build_shared(cfg, device)
    shared["truth3"] = MC.true_tables(shared["rules"], cfg["depth"], cfg["s"], cfg["v"],
                                      cfg["m"], 3)[3]
    world = build_world(shared, cfg)
    shared["_l3"] = world["l3"]
    shared["_mine_pool"] = world["mine_pool"]
    shared["_shared_m3"] = world["shared_m3"]
    refs = measure_refs(shared, cfg, world, device)
    gate = gate_world(v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
                      rule_seed=cfg["rule_seed"], sigma=cfg["sigma"], kappa=cfg["kappa"],
                      n_venue=cfg["n_venue"], train_nodes=cfg["train_nodes"],
                      hold_node=cfg["hold_node"], level=cfg["level"],
                      demand_levels=cfg["demand_levels"], seed=cfg["seed"], n=2048)
    print(f"[gate] {json.dumps({k: (val.get('pass') if isinstance(val, dict) else val) for k, val in gate.items()})}")
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}")
    print(f"[world] I(venue;latent) at start = {world['i_wf_init']:.4f} nats; "
          f"{len(world['keys'])} training keys; true L2 table "
          f"{shared['truth'][2]['child'].shape[0]} entries, L3 "
          f"{shared['truth3']['child'].shape[0]}")

    outdir = f"{DATA_DIR}/rhm_practice_merge/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "refs": refs, "gate": gate,
                   "i_wf_init": world["i_wf_init"], "keys": [str(k) for k in world["keys"]],
                   "venue_info": world["venue_info"]}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in parse_arms(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, world, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "gate": gate["all_pass"], "elapsed": time.time() - started}


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0):
    """Structural gates with no substrate: the index algebra, the merge invariants, and the
    key machinery. Everything the GPU run relies on that can be checked without one."""
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv_bottom = build_inverse_maps(rules)[-1]
    keys = [(w, j) for w in range(3) for j in (1, 2, 3)]

    # I-1: the three policies give the partitions they claim
    assert LIB.KeyedLibrary(keys, 2, s, policy="track").n_cells() == len(keys)
    assert LIB.KeyedLibrary(keys, 2, s, policy="global").n_cells() == 1
    g = LIB.KeyedLibrary(keys, 2, s, policy="global")
    assert all(g.cell_of(k) == 0 for k in keys)
    print("I-1 partitions ok")

    # I-2: a key outside the index is a CACHE MISS and falls back to the biggest cell
    lib = LIB.KeyedLibrary(keys, 2, s, policy="track")
    rr = np.random.default_rng(0)
    for i, k in enumerate(keys):
        lib.observe(k, rr.integers(0, v, size=(4 + 4 * i, 2)))
    lib.rebuild(MC.base_table(v), 1)
    assert lib.cell_of((9, 9)) is None
    assert lib.fallback_cell() == lib.cell_of(keys[-1])
    t, c = lib.table_for((9, 9))
    assert c == lib.fallback_cell() and t is not None
    print("I-2 cache miss / fallback ok")

    # I-3: merge coarsens the index and never blends content
    a, b = sorted(lib.cells)[:2]
    ta = {tuple(int(x) for x in r) for r in lib.cells[a]["table"]["flat"]}
    before = lib.n_cells()
    lib.merge(a, b, a)
    lib.rebuild(MC.base_table(v), 1)
    assert lib.n_cells() == before - 1
    assert lib.cell_of(keys[0]) == a and lib.cell_of(keys[1]) == a
    tm = {tuple(int(x) for x in r) for r in lib.cells[a]["table"]["flat"]}
    assert tm <= ta, "rep-mode merge introduced content that was not the representative's"
    print("I-3 merge is index-only ok")

    # I-4: a perfect alias merges; a disjoint pair does not
    lib2 = LIB.KeyedLibrary(keys[:3], 2, s, policy="track")
    lib2.observe(keys[0], np.array([[0, 1]] * 8))
    lib2.observe(keys[1], np.array([[0, 1]] * 8))
    lib2.observe(keys[2], np.array([[4, 5]] * 8))
    lib2.rebuild(MC.base_table(v), 1)
    sets = {c: {tuple(int(x) for x in r) for r in cell["table"]["flat"]}
            for c, cell in lib2.cells.items()}

    def score_fn(table, key):
        want = {(0, 1)} if key in keys[:2] else {(4, 5)}
        got = {tuple(int(x) for x in r) for r in table["flat"]}
        return 1.0 if want <= got else 0.0
    rec = LIB.merge_pass(lib2, score_fn, tau=0.95, min_obs=1)
    assert rec["n_merges"] == 1 and lib2.n_cells() == 2, rec
    print(f"I-4 alias merges, non-alias does not ok ({rec['n_merges']} merge)")

    # I-5: mutual information is 0 for an independent joint and > 0 for a deterministic one
    ind = {(w, f): 1 for w in range(3) for f in range(3)}
    det = {(w, w): 3 for w in range(3)}
    assert abs(LIB.mutual_info(ind)) < 1e-9 and LIB.mutual_info(det) > 1.0
    print("I-5 mutual information ok")

    # I-6: the ratchet's nesting still bites -- T3 over a thin T2 is (nearly) empty
    thin = MC.make_table(2, np.array([[0, 1]]), MC.base_table(v), s)
    full = MC.true_tables(rules, depth, s, v, m, 2)[2]
    mn = MC.Miner(3, s)
    _r, lv = _sample_pool(rules, 512, s, 0)
    feats = MC.exact_features(lv, inv_bottom, v, s)
    mn.observe(feats[:, 0:4])
    n_thin = LIB.next_level(thin, mn, 1)["child"].shape[0]
    n_full = LIB.next_level(full, mn, 1)["child"].shape[0]
    assert n_thin < n_full, (n_thin, n_full)
    print(f"I-6 nesting bites: |T3| over 1-entry T2 = {n_thin}, over the true T2 = {n_full}")

    # I-7: the world gates
    g = gate_world(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, n=2048)
    print("I-7 world gate:", json.dumps({k: (val.get("pass") if isinstance(val, dict) else val)
                                         for k, val in g.items()}))
    assert g["all_pass"], json.dumps(g, indent=2, cls=NumpyEncoder)
    print("\nALL SELFCHECKS PASS")
    return {"gate": g}


@app.function(image=image, timeout=3600, memory=16384)
def selfcheck_remote():
    return selfcheck()


@app.local_entrypoint()
def main(quick: bool = True):
    selfcheck_remote.remote()
