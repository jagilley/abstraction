"""tall -- the pacing question at the repo's deep RHM setting (64-token sequences).

`recital` asked whether an agent can pace its own curriculum, on a 4-level grammar (v=8, s=2,
depth=4, m=2). Its ladder topped out at level 3, which is BOTH the deepest level the agent can
earn and one below the grammar's own ceiling -- so every endogenous-signal failure the arc found
at level 3 is confounded: boundary artifact, or a real frontier effect? This node moves to
depth 6, where levels 2-3 are interior to a 6-level grammar and the damage ladder runs to
level 5, and asks the same questions where the two explanations come apart.

THIS FILE IS THE FEASIBILITY PROBE, NOT THE MAIN RUN. It measures, on the real substrate, the
things that decide whether the port is affordable and whether it sculpts cleanly at all:
setup cost, per-cycle cost, reader exactness, the damage ladder's references, and the pricing
of a 32-62 move action set. The arm loop is `recital`'s and is imported, not copied.

WHAT THE OFFLINE CALIBRATION ALREADY SETTLED (CPU only, no GPU spent):

  * m=4 IS INADMISSIBLE FOR THIS SUBSTRATE. The whole arc measures cost-to-depth. The oracle's
    repair distance d* over the nested ladder, on broken-only instances:

        depth 6, m=2:  1.71  2.37  3.23  4.52  5.54   (gradient 3.25x)
        depth 6, m=3:  1.33  1.42  1.33  1.52  1.73   (gradient 1.30x)
        depth 6, m=4:  1.09  1.12  1.12  1.17  1.15   (gradient 1.06x)

    At m=4 every level of damage is one move from repair: synonymy is high enough that some
    single commitment re-derives the observed subtree, so "deeper" damage is not harder and
    the ladder the arc exists to climb is gone. The 2x2 (depth 4/6 x m 2/4) localises this to
    m, not to depth: depth 4 m=4 also collapses (1.04x) and depth 6 m=2 is the STEEPEST ladder
    the arc has ever had (3.25x against depth 4 m=2's 1.69x).

  * m=4 ALSO WALLS OFF THE MACRO LEVELS. entries(l) = v*e(l), e(l) = m*e(l-1)^s -- doubly
    exponential, verified against `MC.true_tables` at levels 2-4:

        m=2:   L2 16    L3 64    L4 1,024    L5 262,144
        m=4:   L2 32    L3 512   L4 131,072  L5 8,589,934,592

    so at m=4 nothing above level 3 is representable, i.e. the very interior levels this round
    exists to test cannot be earned. (m=4's L5 is what OOM-killed the calibration host.)

  * m=2 IS THE ADMISSIBLE SETTING, and it is the one the rest of the arc already uses, so the
    depth-4 results stay directly comparable -- only depth moves.

Run from experiments/:
  modal run rhm/practice/tall/tall.py::gates_remote
  modal run rhm/practice/tall/tall.py::feasibility --cycles 6
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.practice.crystallize.units import (
    build_move_set, corrupt_hier, grade, nearest_derivation_cost, on_grammar_rate)
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    audition_macro, beam_ground, beam_moves, build_ms, context_instances, era_ctx,
    finetune_generator, fit_width, measure_refs, parse_eras, plant_probe, priced, push,
    value_steps)
from rhm.practice.ratchet.ratchet import collect_value_buffer
from rhm.rhm_latent_planner import _build_value_head
from rhm.practice.ear import grader as GR
from rhm.practice.ear.ear import (
    _cfg as ear_cfg, _shared_plus, node_at, selfcheck as ear_selfcheck, write_results)
from rhm.practice.recital.recital import (
    ARMS as RECITAL_ARMS, new_detector, parse_arms_nm, step_detector, vocab_rate)

ARMS = dict(RECITAL_ARMS)
# `recital`'s `sched_frac` reads exactly two boundaries (sf1, sf2) because its ladder had three
# eras. A five-era ladder needs four, so this node adds `sched_list`, whose boundaries are a
# comma-separated list of CUMULATIVE fractions of the total priced budget -- one per era
# transition. Everything else about the arm is `sched_frac`.
ARMS["sched_list"] = {**RECITAL_ARMS["sched_frac"], "pace": "list"}
# `given` is the vocabulary CEILING, so it has to walk the same ladder as the arms it bounds.
# `recital`'s `given` is clock-paced, which on a five-era ladder under a priced budget would
# strand it in era 3; this variant takes the same schedule language as the sched arms so it can
# be given the uniform shape and differ from `sched_uniform` in vocabulary alone.
ARMS["given_list"] = {**RECITAL_ARMS["given"], "pace": "list"}

app = modal.App("rhm-practice-tall")
REMOTE = "rhm_practice_tall"

# the nested ladder at depth 6: node_at(L1n25, l) = 25 // 2**(l-1) = 12, 6, 3, 1
TALL_ERAS = "1:25,2:12,3:6,4:3,5:1"
TALL = dict(depth=6, m=2, v=8, s=2)


def _tall_shared(cfg, device):
    """THE shared-dict constructor for this node. `_shared_plus` gives everything
    `build_shared` returns plus `canon_np`/`bottom_np`, but `probe_clean` is populated by each
    round's ENTRYPOINT rather than by the builder -- so importing the loop without the
    entrypoint silently omits it, and `plant_probe` is the first thing to notice, after setup
    has already been paid. Constructed here exactly as `ear`/`recital` do, once, so every
    entrypoint in this file gets an identical and complete dict.

    Audited against every `shared[...]` read in ratchet.py / ear.py / recital.py:
      base_ms bottom_map bottom_np canon canon_np controller generator0 hold inverse_maps
      length n_blocks probe_clean reader replay rules rules_t truth value0
    `build_shared` supplies all but the last three of those groups, `_shared_plus` adds
    canon_np/bottom_np, and `probe_clean` is the only remaining gap -- this function closes it.
    """
    import torch
    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    missing = [k for k in ("base_ms", "bottom_map", "bottom_np", "canon", "canon_np",
                           "controller", "generator0", "hold", "inverse_maps", "length",
                           "n_blocks", "probe_clean", "reader", "replay", "rules", "rules_t",
                           "truth", "value0") if k not in shared]
    assert not missing, f"shared dict is missing {missing} -- interface drift"
    return shared


def _tall_cfg(**kw):
    """`ear`'s config with the tall grammar substituted. Everything else is inherited so the
    depth-4 runs stay reproducible and the only deliberate difference is the setting."""
    cfg = ear_cfg(**TALL)
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# gates -- all CPU, all inherited, run at the tall ladder
# --------------------------------------------------------------------------- #

def gates(max_level=3, eras=TALL_ERAS, n=256, rule_seed=0):
    """The arc's gates (C-M / C-R / G-D / G-N / G-S / G-R) at depth 6, plus this port's own:

      T-1  the ladder is nested at every one of its five levels and each node index is in
           range for its level (there are s**(depth-l) nodes at level l)
      T-2  damage is 100% on-grammar at EVERY ladder level, and every instance is genuinely
           broken after rejection -- with the acceptance rate, which is what the port pays
      T-3  the oracle's repair distance is monotone in ladder depth (the cost-to-depth the
           whole arc measures); reported with the budget needed to cover it
      T-4  the declared grounding budget that prices a 32-62 move action set
    """
    v, s, depth, m = TALL["v"], TALL["s"], TALL["depth"], TALL["m"]
    from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
    # `ear`'s G-S manufactures an audition for the level each era EARNS (era.level + 1), so it
    # is only defined on eras whose earned level is <= max_level. At depth 4 that was every
    # era, because the ladder was exactly as long as the earnable range. Here the damage ladder
    # is deliberately LONGER than the earnable range -- eras past level max_level-1 are pure
    # consumption, which is the separation this round exists to create -- so the inherited gate
    # is run on the earning prefix and the consumption eras are covered by T-2/T-3 below.
    earning = ",".join(f"{e['level']}:{e['node']}" for e in parse_eras(eras)
                       if e["level"] + 1 <= max_level)
    out = {"inherited": ear_selfcheck(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, n=n,
                                      max_level=max_level, eras=earning),
           "earning_prefix": earning, "full_ladder": eras}
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)[-1]
    ers = parse_eras(eras)

    # T-1
    t1 = []
    for i, era in enumerate(ers):
        n_nodes = s ** (depth - era["level"])
        ok_range = 0 <= era["node"] < n_nodes
        nested = True
        if i + 1 < len(ers):
            nxt = ers[i + 1]
            nested = (era["node"] * s ** (era["level"] - 1)) // s ** (nxt["level"] - 1) \
                == nxt["node"]
        t1.append({"era": era["name"], "n_nodes_at_level": n_nodes,
                   "node_in_range": bool(ok_range), "nests_into_next": bool(nested)})
        assert ok_range, f"T-1 failed: {era['name']} node out of range (max {n_nodes - 1})"
        assert nested, f"T-1 failed: {era['name']} does not nest into the next era"
    out["T1_ladder"] = t1

    # T-2 / T-3
    roots, leaves = _sample_pool(rules, n, s, rule_seed + 31337)
    t23 = {}
    for era in ers:
        rng = np.random.default_rng(7)
        dmg = corrupt_hier(leaves.copy(), rules, depth, v, m, s, era["level"],
                           [era["node"]], rng)
        og = on_grammar_rate(dmg, inv, v, s)
        d = nearest_derivation_cost(rules, dmg, roots, s)
        brk = d > 0
        t23[era["name"]] = {"on_grammar": og, "accept_rate": float(brk.mean()),
                            "dstar_mean": float(d[brk].mean()),
                            "dstar_max": int(d[brk].max()),
                            "p_dstar_gt_budget": {str(b): float((d[brk] > b).mean())
                                                  for b in (4, 6, 8, 10)}}
        assert og == 1.0, f"T-2 failed: damage off-grammar at {era['name']}"
    out["T23_damage"] = t23
    ds = [t23[e["name"]]["dstar_mean"] for e in ers]
    out["T3_depth_gradient"] = float(max(ds) / min(ds))
    assert all(b >= a for a, b in zip(ds, ds[1:])), \
        f"T-3 failed: repair distance is not monotone in depth: {ds}"

    # T-4
    sizes = {f"max_level={lv}": len(build_move_set(depth, s, max_level=lv))
             for lv in range(1, depth)}
    out["T4_action_set"] = sizes
    out["T4_pricing"] = {f"n{n_}_b{b}_w{w}": beam_ground(n_, b, w)
                         for n_ in sorted(set(sizes.values()))
                         for b in (4, 6, 8) for w in (1, 2)}
    out["true_table_entries"] = {f"L{lv}": int(8 * _e(m, lv)) for lv in range(2, 6)}
    return out


def _e(m, lv):
    e = 1
    for _ in range(2, lv + 1):
        e = m * e ** 2
    return e


@app.function(image=image, timeout=3600, memory=32768)
def gates_remote(max_level: int = 3, eras: str = TALL_ERAS):
    out = gates(max_level=max_level, eras=eras)
    print("\n=== TALL GATES ===")
    print(json.dumps({k: v for k, v in out.items() if k != "inherited"},
                     indent=2, cls=NumpyEncoder))
    return out


# --------------------------------------------------------------------------- #
# instrument: per-cycle ENTRY IDENTITY of the mined vocabulary
# --------------------------------------------------------------------------- #

def _install_identity_miner(support):
    """`recital`'s mechanism check could see the mined table's SIZE and its recall but never
    its MEMBERSHIP, so "did the table change or only grow?" was unanswerable. `Miner.state()`
    is already called once per cycle per level and its return is logged verbatim, so extending
    it reaches exactly the missing data without touching the arm loop at all.

    Keys are the miner's own flattened level-1 tuples -- the agent's own representation, stable
    across cycles and comparable across arms. At m=2 the level-2 and level-3 tables top out at
    16 and 64 entries, so this is small.

    Idempotent: re-installing does not re-wrap.
    """
    if getattr(MC.Miner, "_tall_identity", False):
        return
    orig = MC.Miner.state

    def state(self):
        out = orig(self)
        out["keys_at_support"] = sorted(
            [int(x) for x in k] for k, c in self.counts.items() if c >= support)
        return out

    MC.Miner.state = state
    MC.Miner._tall_identity = True


# --------------------------------------------------------------------------- #
# the arm loop -- `recital.run_arm` forked ONLY to add this round's two instruments
# --------------------------------------------------------------------------- #
#
# Copied verbatim from recital.py (programmatically, to rule out transcription error) with
# exactly two additions, both marked `# [tall]`:
#
#   1. FIXED-REFERENCE POLICY PROBE. The inherited probe already evaluates every era's fixed
#      metering set, but always with the arm's CURRENT action set, so policy quality and
#      vocabulary are confounded -- which is exactly what `recital`'s mechanism check could not
#      separate. This adds the same fixed sets evaluated with the BASE action set only, so the
#      vocabulary is held constant and what moves is the policy and value model alone.
#   2. `n_moves` recorded alongside it, so the two are comparable cycle by cycle.
#
# Entry-identity logging is NOT here -- it is a `MC.Miner.state` extension (see
# `_install_identity_miner`), which reaches the same data without touching this loop at all.

def run_arm_tall(label, base, overrides, shared, cfg, eras, refs, outdir, device):
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
            elif pace == "list":
                # [tall] one cumulative budget fraction per era transition, so a ladder of any
                # length can be given an arbitrary SHAPE. cfg["sf_list"] is "f1,f2,f3,f4".
                fl = [float(x) for x in str(cfg["sf_list"]).split(",")]
                if T > 0 and era_i < len(fl) and t_now >= fl[era_i] * T:
                    adv_reason = "list"
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
            # [tall] FIXED-REFERENCE POLICY PROBE: every era's fixed metering set again,
            # but with the BASE action set. Vocabulary is held constant by construction, so a
            # change here is policy/value quality and nothing else. Unpriced instrument;
            # consumes no randomness, so it cannot perturb any arm.
            probe["base_ms"] = {}
            wb = fit_width(len(base_ms), cfg["budget"], cfg["g_budget"])
            for j in range(len(eras)):
                jr, jx = meter[j]
                bb = beam_moves(controller, generator, value, jx, torch.from_numpy(jr),
                                base_ms, rules_t, canon, depth, v, m, s,
                                budget=cfg["budget"], beam_width=wb, device=device)
                sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                probe["base_ms"][str(j)] = 1.0 - float(sc.mean())
            probe["n_moves"] = len(ms)
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


# --------------------------------------------------------------------------- #
# the feasibility probe: what does the tall substrate COST, and does it sculpt?
# --------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def feasibility(cycles: int = 6, era_cycles: int = 0, max_macro_level: int = 3,
                eras: str = TALL_ERAS,
                budget: int = 8, g_budget: int = 260, pr_width: int = 16,
                n_pr: int = 64, n_rt: int = 384, n_score: int = 512, n_aud: int = 192,
                arm: str = "practice_climb", quick_setup: bool = False,
                tag: str = "feas0"):
    """Times the real thing. Reports setup seconds, per-cycle seconds, the references the
    ladder actually has at depth 6, reader exactness, and the priced cost per cycle -- the
    numbers needed to size the main run before committing GPU-hours."""
    import torch
    cfg = _tall_cfg(max_macro_level=max_macro_level, budget=budget, g_budget=g_budget,
                    pr_width=pr_width, n_pr=n_pr, n_rt=n_rt, n_score=n_score, n_aud=n_aud,
                    era_cycles=(era_cycles or cycles), t_budget=0.0, max_cycles=cycles,
                    probe_every=10 ** 9,
                    seed=0, rule_seed=0, train_seed=1,
                    # only read when t_budget > 0 / pace == "frac", but the probe is a
                    # 45-minute GPU run and a KeyError at cycle 1 is not worth the risk
                    probe_t_n=12, sf1=0.33, sf2=0.67)
    if quick_setup:
        cfg.update(controller_steps=1500, generator_steps=1500, value_steps=1500,
                   reader_steps=1000, value_episodes=8000, n_train_episodes=30000)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    print(f"[tall/feasibility] depth={cfg['depth']} m={cfg['m']} seqlen={cfg['s']**cfg['depth']} "
          f"max_macro_level={max_macro_level} budget={budget} g_budget={g_budget} "
          f"eras={eras} arm={arm} cycles={cycles} quick_setup={quick_setup}", flush=True)

    t0 = time.time()
    shared = _tall_shared(cfg, device)
    t_setup = time.time() - t0
    print(f"[tall/feasibility] SETUP {t_setup:.0f}s", flush=True)

    # what the action set and its pricing actually are at this depth
    sizes = {}
    for lv in range(1, cfg["depth"]):
        sizes[lv] = len(build_move_set(cfg["depth"], cfg["s"], max_level=lv))
    w = fit_width(sizes[max_macro_level], budget, g_budget)
    print(f"[tall/feasibility] action set by max_level: {sizes}; "
          f"fit_width({sizes[max_macro_level]}, budget={budget}, g={g_budget}) = {w}", flush=True)

    t0 = time.time()
    refs = measure_refs(shared, cfg, ers, device)
    t_refs = time.time() - t0
    print(f"[tall/feasibility] REFS {t_refs:.0f}s: " +
          json.dumps(refs, cls=NumpyEncoder)[:1200], flush=True)

    # `generator0` is the shared pre-arm plant; each arm deepcopies it, so probing it here
    # measures the plant every arm starts from. `read_acc` is the reader's own training
    # accuracy, already computed in `build_shared` -- it is the quantity that compounds as
    # acc**span when mining at level l, so it is what caps `max_macro_level`.
    plant = plant_probe(shared["generator0"], shared, cfg, device)
    read_acc = float(shared["read_acc"])
    span = cfg["s"] ** (max_macro_level - 1)
    print(f"[tall/feasibility] reader read_acc={read_acc:.4f}  "
          f"compounded over the level-{max_macro_level} span of {span} blocks: "
          f"{read_acc ** span:.4f}", flush=True)
    print(f"[tall/feasibility] plant exactness: {json.dumps(plant, cls=NumpyEncoder)}",
          flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    import os
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/setup.json", "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs, "plant": plant,
                   "read_acc": read_acc, "action_set": sizes, "width": int(w),
                   "t_setup_s": t_setup, "t_refs_s": t_refs}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    t0 = time.time()
    _install_identity_miner(cfg["mine_support"])
    run_arm_tall(arm, arm, {}, shared, cfg, ers, refs, outdir, device)
    t_arm = time.time() - t0
    n_done = len(json.load(open(f"{outdir}/{arm}/results.json"))["log"]["cycle"])
    print(f"\n[tall/feasibility] ARM {arm}: {t_arm:.0f}s for {n_done} cycles "
          f"-> {t_arm / max(n_done, 1):.1f}s/cycle", flush=True)
    print(f"[tall/feasibility] TOTALS setup={t_setup:.0f}s refs={t_refs:.0f}s "
          f"per_cycle={t_arm / max(n_done, 1):.1f}s over {n_done} cycles", flush=True)
    volume.commit()
    return {"t_setup_s": t_setup, "t_refs_s": t_refs,
            "t_per_cycle_s": t_arm / max(n_done, 1), "n_cycles": n_done,
            "action_set": sizes, "width": int(w),
            "refs": refs, "plant": plant, "read_acc": read_acc}


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600,
              memory=32768)
def preflight(cycles: int = 2, max_macro_level: int = 3, eras: str = TALL_ERAS,
              budget: int = 2, g_budget: int = 482):
    """CPU-only interface check: build a TINY shared dict at the tall grammar and call every
    imported function once, in the order `feasibility` calls them.

    This exists because the two `feas0` crashes were both interface drift -- a key the imported
    loop reads but this node's setup never populated -- and both surfaced only AFTER ~510s of
    GPU setup had been paid. The substrate here is trained for a few dozen steps and is
    scientifically worthless; the point is solely that every call SIGNATURE and every
    `shared[...]` access resolves. Run this before any GPU launch.

    It takes a GPU despite being a correctness check: beam search over a 32-56 move action set
    at 64 tokens is minutes-per-cycle on CPU, which defeats the purpose. `budget=2` keeps the
    beam shallow -- the move set, not the rollout depth, is what exercises the interface.
    """
    import torch
    cfg = _tall_cfg(max_macro_level=max_macro_level, budget=budget, g_budget=g_budget,
                    era_cycles=1, t_budget=1.0e6, max_cycles=cycles, probe_every=1,
                    seed=0, rule_seed=0, train_seed=1, probe_t_n=2, sf1=0.33, sf2=0.67,
                    sf_list="0.2,0.4,0.6,0.8", probe_widths=[1],
                    # deliberately tiny -- this is an interface test, not an experiment
                    controller_steps=40, generator_steps=40, value_steps=40, reader_steps=40,
                    value_episodes=256, n_train_episodes=2000, value_batch_collect=128,
                    n_pr=8, n_rt=32, n_score=32, n_aud=16, n_probe_clean=32, mfg_min_bank=4,
                    bank_cap=64, recert_every=2, sil_min_cycle=1, checkpoint_every=10 ** 9)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    ok = {}
    shared = _tall_shared(cfg, device)
    ok["_tall_shared"] = sorted(shared.keys())
    ok["measure_refs"] = bool(measure_refs(shared, cfg, ers, device))
    ok["plant_probe"] = plant_probe(shared["generator0"], shared, cfg, device)
    ok["read_acc"] = float(shared["read_acc"])
    # the densification path: same three calls, same argument order, tiny sizes
    vcfg = {**cfg, "n_corrupt": 1, "budget": 3, "value_episodes": 64}
    vb = collect_value_buffer(shared["controller"], shared["generator0"], shared["rules"],
                              shared["rules_t"], shared["canon"], shared["base_ms"],
                              shared["roots_pool"], shared["leaves_pool"], vcfg, device)
    ok["collect_value_buffer"] = f"{int(vb['x'].shape[0])} states, y={float(vb['y'].mean()):.3f}"
    _v = _build_value_head()(cfg["state_dim"], cfg["v"]).to(device)
    _o = torch.optim.AdamW(_v.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
    value_steps(_v, _o, shared["controller"], vb,
                {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]},
                n_steps=2, batch=8, replay_frac=0.0, device=device,
                rng=np.random.default_rng(0))
    ok["value_steps_on_densified_buffer"] = True
    refs = measure_refs(shared, cfg, ers, device)
    outdir = f"{DATA_DIR}/{REMOTE}/_preflight"
    import os
    os.makedirs(outdir, exist_ok=True)
    _install_identity_miner(cfg["mine_support"])
    for arm, ov in (("given", {}), ("practice_climb", {}), ("pace_cert", {}),
                    ("pace_vocab", {}), ("sched_list", {"sf_list": "0.2,0.4,0.6,0.8"})):
        run_arm_tall(arm, arm, ov, shared, cfg, ers, refs, outdir, device)
        ok[f"run_arm_tall:{arm}"] = True
    # the two instruments must actually appear in the log, not merely fail to crash
    res = json.load(open(f"{outdir}/sched_list/results.json"))
    pr = [q for q in res["log"]["probe"] if "base_ms" in q]
    ident = [c for c in res["log"]["miner"] if "keys_at_support" in c.get("2", {})]
    ok["instrument_fixed_reference_probe"] = f"{len(pr)} probes carry base_ms"
    ok["instrument_entry_identity"] = f"{len(ident)}/{len(res['log']['miner'])} cycles carry keys_at_support"
    assert pr and ident, "an instrument did not reach the log"
    print("\n=== PREFLIGHT OK — every imported call resolved at the tall grammar ===")
    print(json.dumps({k: v for k, v in ok.items() if k != "_tall_shared"},
                     indent=2, cls=NumpyEncoder))
    print(f"shared keys: {ok['_tall_shared']}")
    return ok


# --------------------------------------------------------------------------- #
# the main run
# --------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def tall(tag: str = "tl_s0",
         arms: str = ("given_list:sf_list=0.20/0.40/0.60/0.80:nm=given,"
                      "sched_list:sf_list=0.20/0.40/0.60/0.80:nm=sched_uniform,"
                      "sched_list:sf_list=0.50/0.70/0.80/0.90:nm=sched_bottom,"
                      "sched_list:sf_list=0.32/0.73/0.82/0.91:nm=sched_earnable,"
                      "pace_cert,pace_vocab"),
         eras: str = TALL_ERAS, t_budget: float = 5.6e7, max_cycles: int = 200,
         max_macro_level: int = 3, budget: int = 8, g_budget: int = 482,
         era_cycles: int = 35, probe_every: int = 8, probe_t_n: int = 12,
         seed: int = 0, rule_seed: int = 0, train_seed: int = 1):
    """Six arms at one matched total priced budget on the five-era tall ladder.

    `sf_list` uses `/` as its internal separator because `parse_arms_nm` splits arm specs on
    `:` and overrides on `=`; commas separate ARMS. It is normalised to commas below.
    """
    import torch
    cfg = _tall_cfg(max_macro_level=max_macro_level, budget=budget, g_budget=g_budget,
                    era_cycles=era_cycles, t_budget=t_budget, max_cycles=max_cycles,
                    probe_every=probe_every, probe_t_n=probe_t_n,
                    seed=seed, rule_seed=rule_seed, train_seed=train_seed,
                    sf1=0.33, sf2=0.67)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"tall tag={tag} depth={cfg['depth']} m={cfg['m']} seqlen={cfg['s']**cfg['depth']} "
          f"eras={ers} T={t_budget:.0f} budget={budget} g_budget={g_budget} arms={arms}",
          flush=True)

    _install_identity_miner(cfg["mine_support"])
    shared = _tall_shared(cfg, device)
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}", flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/setup.json", "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs,
                   "read_acc": float(shared["read_acc"])}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    for label, base, ov in parse_arms_nm(arms):
        if "sf_list" in ov:
            ov["sf_list"] = str(ov["sf_list"]).replace("/", ",")
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        run_arm_tall(label, base, ov, shared, cfg, ers, refs, outdir, device)
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}", flush=True)
    volume.commit()


# --------------------------------------------------------------------------- #
# the value-densification test + the DESCENT GATE
# --------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=32768)
def densify(tag: str = "dens0", sweep_episodes: int = 8192, cycles: int = 22,
            era: str = "1:25", max_macro_level: int = 3, budget: int = 8,
            g_budget: int = 482, arm: str = "practice_climb",
            variants: str = "3/8,1/8,1/12,2/12,1/16"):
    """`tl_s0` failed because the stale value buffer's terminal success was 0.076 at depth 6
    against 0.296 at depth 4, so the value head trained on a ~4x sparser positive signal and
    could not rank states.

    DIAGNOSIS. `collect_value_buffer` corrupts `n_corrupt` blocks with random symbols and rolls
    out `budget` moves over the BASE move set. The difficulty that changed with depth is not
    damage severity -- 3 blocks of 32 is a milder corruption than 3 of 8 -- it is SEARCH: the
    agent must land moves on 1-3 specific level-1 nodes out of 32 (depth 6) within 8 moves,
    where at depth 4 it was 1-3 out of 8 within 4. The action space grew 4x while the budget
    grew 2x and the number of targets stayed put.

    So the levers tested are the two that act on search difficulty -- fewer targets
    (`n_corrupt`) and more attempts (collection `budget`) -- rather than more data, which would
    raise the absolute positive count while leaving the 7.6% rate untouched. The task budget is
    NOT changed; only the generic buffer the value head is pre-trained on, which is a prior and
    not the graded task.

    Two readouts, in order:
      1. buffer terminal-success rate per variant (target: depth 4's 0.296)
      2. THE DESCENT GATE -- with the densified value head, does era-1 competence actually move
         from `stale` toward `floor`? Reported as the recovered fraction of that range, against
         depth 4's 50-80% and `tl_s0`'s 3.5-16.5%. This gate joins the permanent feasibility set
         whatever the answer, because its absence is what let `tl_s0` be launched.
    """
    import torch
    cfg = _tall_cfg(max_macro_level=max_macro_level, budget=budget, g_budget=g_budget,
                    era_cycles=cycles, t_budget=0.0, max_cycles=cycles,
                    probe_every=10 ** 9, probe_widths=[1],
                    seed=0, rule_seed=0, train_seed=1, probe_t_n=12, sf1=0.33, sf2=0.67)
    ers = parse_eras(era)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[densify] tag={tag} depth={cfg['depth']} m={cfg['m']} era={era} "
          f"variants={variants} sweep_episodes={sweep_episodes} cycles={cycles}", flush=True)

    _install_identity_miner(cfg["mine_support"])
    t0 = time.time()
    shared = _tall_shared(cfg, device)
    print(f"[densify] SETUP {time.time() - t0:.0f}s", flush=True)

    # ---- 1. the sweep: terminal-success rate of the generic buffer, per variant ----------
    rows = []
    for spec in variants.split(","):
        nc, bg = (int(x) for x in spec.split("/"))
        vcfg = {**cfg, "n_corrupt": nc, "budget": bg, "value_episodes": sweep_episodes}
        t1 = time.time()
        vb = collect_value_buffer(shared["controller"], shared["generator0"], shared["rules"],
                                  shared["rules_t"], shared["canon"], shared["base_ms"],
                                  shared["roots_pool"], shared["leaves_pool"], vcfg, device)
        succ = float(vb["y"].mean())
        rows.append({"n_corrupt": nc, "collect_budget": bg, "terminal_success": succ,
                     "n_states": int(vb["x"].shape[0]), "seconds": time.time() - t1})
        print(f"[densify]   n_corrupt={nc} collect_budget={bg}: terminal_success={succ:.4f} "
              f"({rows[-1]['seconds']:.0f}s)  [depth4=0.296  tl_s0=0.076]", flush=True)
    best = max(rows, key=lambda r: r["terminal_success"])
    print(f"[densify] BEST variant: n_corrupt={best['n_corrupt']} "
          f"collect_budget={best['collect_budget']} success={best['terminal_success']:.4f}",
          flush=True)

    # ---- 2. retrain the value head on a full densified buffer ----------------------------
    vcfg = {**cfg, "n_corrupt": best["n_corrupt"], "collect_budget": best["collect_budget"],
            "budget": best["collect_budget"]}
    t1 = time.time()
    vb = collect_value_buffer(shared["controller"], shared["generator0"], shared["rules"],
                              shared["rules_t"], shared["canon"], shared["base_ms"],
                              shared["roots_pool"], shared["leaves_pool"], vcfg, device)
    print(f"[densify] full densified buffer {vb['x'].shape[0]} states, "
          f"terminal success {float(vb['y'].mean()):.4f} ({time.time() - t1:.0f}s)", flush=True)
    value = _build_value_head()(cfg["state_dim"], cfg["v"]).to(device)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["train_seed"] + 31)
    empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
    value_steps(value, opt, shared["controller"], vb, empty, n_steps=cfg["value_steps"],
                batch=512, replay_frac=0.0, device=device, rng=rng)
    shared["value0"] = value
    shared["replay"] = vb

    # ---- 3. THE DESCENT GATE -------------------------------------------------------------
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[densify] refs {json.dumps(refs, cls=NumpyEncoder)}", flush=True)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/setup.json", "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs, "sweep": rows, "best": best,
                   "read_acc": float(shared["read_acc"])}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    t1 = time.time()
    run_arm_tall(arm, arm, {}, shared, cfg, ers, refs, outdir, device)
    log = json.load(open(f"{outdir}/{arm}/results.json"))["log"]
    n = len(log["cycle"])
    st, fl = refs["stale"][0], refs["floor"][0]
    f5 = float(np.mean(log["e"][:5])); l5 = float(np.mean(log["e"][-5:]))
    rec = (st - l5) / (st - fl)
    print(f"\n[densify] === DESCENT GATE ===", flush=True)
    print(f"[densify] era1 e: first5={f5:.4f} last5={l5:.4f} delta={l5 - f5:+.4f} "
          f"(stale={st:.3f} floor={fl:.3f})", flush=True)
    print(f"[densify] RECOVERED FRACTION = {rec:.3f}   "
          f"[depth4 0.50-0.80 | tl_s0 0.035-0.165]", flush=True)
    print(f"[densify] {n} cycles in {time.time() - t1:.0f}s "
          f"({(time.time() - t1) / max(n, 1):.1f}s/cycle); TOTAL {time.time() - started:.0f}s",
          flush=True)
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump({"sweep": rows, "best": best, "e_first5": f5, "e_last5": l5,
                   "recovered_fraction": rec, "stale": st, "floor": fl,
                   "n_cycles": n, "seconds": time.time() - started}, fh, indent=2,
                  cls=NumpyEncoder)
    volume.commit()
    return {"sweep": rows, "best": best, "recovered_fraction": rec}


@app.local_entrypoint()
def main(cycles: int = 6):
    feasibility.remote(cycles=cycles)
