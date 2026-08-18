"""selfplay -- is free data free COVERAGE? Unlimited data under a self-generated demand
distribution.

WHY THIS ROUND EXISTS. `mg_s0` (see `merge.py` / `FILES.md`) put an unmetered dense learner
against invariance-by-merge and found the affordance-matched monolith (`dense_glob`) effectively
tied the best metered arm in-distribution and led the next-level minability table. But that arm
was handed **free i.i.d. variation over the whole space**, and gate M-5 certified the fifth wall
*inside* the training support. Both are exactly the conditions self-play does not have. In the
[`question_model`](../../question_model/SPEC.md) vocabulary: self-play is training where the
question model and the answer model are the same object, so the demand distribution is the
policy's OWN FOOTPRINT, and channel (ii) collapses. The AlphaZero-shaped hole was unprobed.

WHERE THE SUPPORT AXIS HAD TO GO, AND WHY (calibration, recorded because it killed a design).
The first build restricted support by VENUE -- each venue an OU demand state, the self-play arm
choosing which venues to practise. Measured offline and then confirmed on GPU (`spsmoke0`), that
axis **cannot make a hole in this substrate**: the level-2 vocabulary has only 19 reachable
entries pooled over nodes, every venue demands 9-15 of them, and even at sigma = 25 the union of
two venues' supports covers **87%** of a third venue's demanded mass. `dense_sp_hard` collapsed
to a visitation perplexity of 1.46 venues out of 6 and still covered **96.4%** of every venue's
demand. Venue-restriction buys narrow VOLUME, not a hole.

So the support axis is the **latent itself**: the level-2 entry the damage destroyed, which the
clean derivation names exactly. The world is left at **uniform demand** -- free i.i.d. variation,
literally the distribution `mg_s0`'s monolith trained on -- and the ONLY thing that differs
between arms is *which of the instances the world freely offers an arm chooses to practise on*.
Nothing becomes false, the grammar and the world distribution are untouched objects; this is a
statement about the sampling policy and nothing else, which is the tightest control the substrate
allows.

THE ARMS, as four ways of getting a support:

  * `dense_full`      unmetered, instances taken as the world offers them -- exogenous full
                      support. This is `mg_s0`'s `dense_glob`, re-run here.
  * `dense_visit`     unmetered, `p(e) ~ (n_seen_e + eps)^beta` -- rich-get-richer on its own
                      VISITATION, with no competence coupling. Isolates "the footprint
                      amplifies the world's own skew".
  * `dense_selfplay`  unmetered, `p(e) ~ (n_solved_e + eps)^beta` -- the footprint filtered
                      through COMPETENCE. An entry it fails at accrues solved-count slowly even
                      when it is sampled, so the under-sampling is self-reinforcing. This is the
                      self-play arm. `dense_sp_hard` is the same at beta = 2.
  * `dense_narrow`    unmetered, support restricted to a FIXED EXOGENOUS entry subset whose size
                      is matched in-run to `dense_selfplay`'s realized entry perplexity, drawn
                      by a dedicated RNG so its identity is uncorrelated with world frequency
                      and with competence. Isolates "narrow support" from "SELF-SELECTED
                      support".

The distinctive self-play signature is therefore not narrowness -- three arms are narrow -- but
**holes that sit at the policy's own initial weaknesses**. That is measured against `s0_e`, the
error of the SHARED setup policy on entry `e`, computed once before any arm adapts and identical
for every arm by construction.

THE DIFFICULTY-MATCH FIX carried over from `mg_s0`'s caveat 1: the exam is a per-ENTRY battery
pooled over every node, so no held-out-frame difficulty offset can arise, and every per-entry
readout is reported as excess over `given` -- which holds the DGP's own table, sees every entry,
and therefore measures the intrinsic difficulty profile directly.

Run from experiments/:
  modal run rhm/practice/merge/selfplay.py::selfcheck_sp_remote
  python3 rhm/practice/merge/launch_detached.py --file selfplay --fn selfplay --tag sp_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.practice.crystallize.units import grade
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import build_shared, finetune_generator, push, value_steps
from rhm.practice.setlist import demand as DM
from rhm.practice.merge import library as LIB
from rhm.practice.merge.merge import (
    _cfg, exam, gate_world, minability, parse_arms, post_mine_t2, solve)


app = modal.App("rhm-practice-merge-selfplay", image=image)


ARMS_SP = {
    "never_base":     {"vocab": "base",   "index": None,     "keyp": "full"},
    "given":          {"vocab": "true",   "index": None,     "keyp": "full"},
    "metered_glob":   {"vocab": "earned", "index": "global", "keyp": "full"},
    "dense_full":     {"vocab": "earned", "index": "global", "keyp": "full", "mult": True},
    "dense_visit":    {"vocab": "earned", "index": "global", "keyp": "visit", "mult": True},
    "dense_selfplay": {"vocab": "earned", "index": "global", "keyp": "solved", "mult": True},
    "dense_sp_hard":  {"vocab": "earned", "index": "global", "keyp": "solved", "mult": True,
                       "beta": 2.0},
    "dense_narrow":   {"vocab": "earned", "index": "global", "keyp": "narrow", "mult": True},
    "dense_sp_nolib": {"vocab": "base",   "index": None,     "keyp": "solved", "mult": True},
}


def _sp_cfg(**kw):
    cfg = _cfg()
    cfg.update(
        nodes=(0, 1, 2, 3), l3_node=1, sigma=0.0, rotate_every=0,
        cycles=60, n_pr=72, n_rt=96, dense_mult=4, n_ex=112, n_probe=32,
        n_e4=1024, n_mine_post=1024, probe_every=10,
        sp_beta=1.0, sp_eps=1.0, sp_oversample=6, narrow_seed=7, narrow_k=0,
        narrow_fixed="", pool_per_node=14_000,
    )
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# the world: one uniform demand, a per-ENTRY exam battery
# --------------------------------------------------------------------------- #

def entry_of(clean, node, span, inv_bottom, v, s):
    """The level-2 entry the damage destroyed, read EXACTLY off the clean derivation the damage
    was applied to. This is the latent an agent would otherwise have to infer, and it is the
    coordinate the support is defined on."""
    f = MC.exact_features(clean, inv_bottom, v, s)[:, node * span:(node + 1) * span]
    return [tuple(int(z) for z in row) for row in f]


def build_world_sp(shared, cfg):
    rules, s, depth, v, m = shared["rules"], cfg["s"], cfg["depth"], cfg["v"], cfg["m"]
    inv_bottom = shared["inverse_maps"][-1]
    level, span = cfg["level"], cfg["s"] ** (cfg["level"] - 1)
    uniform = DM.new_demand(rules, list(cfg["demand_levels"]), seed=cfg["demand_seed"],
                            kappa=cfg["kappa"], sigma=0.0, prewarm=False)

    def ctx(node):
        return {"name": f"L{level}n{node}", "level": level, "nodes": [node]}

    # one big pre-draw per node, bucketed by demanded entry: the exam battery. Rare entries
    # carry ~0.4% of the world's mass, so the pool has to be large for the tail to be gradeable
    # at all -- and the tail is exactly where a self-generated support is expected to fail.
    buckets, freq = {}, {}
    for j in cfg["nodes"]:
        r_np, x_np, cl = DM.context_instances_demand(
            rules, ctx(j), cfg["pool_per_node"], s, depth, v, m,
            seed=cfg["seed"] + 300_000 + 31 * j, st=uniform, with_clean=True)
        ents = entry_of(cl, j, span, inv_bottom, v, s)
        for i, e in enumerate(ents):
            freq[e] = freq.get(e, 0) + 1
            buckets.setdefault(e, {"r": [], "x": []})
            buckets[e]["r"].append(r_np[i]); buckets[e]["x"].append(x_np[i])
    entries = sorted(buckets, key=lambda e: -freq[e])
    tot = sum(freq.values())
    cells, n_avail = {}, {}
    for e in entries:
        n = min(cfg["n_ex"], len(buckets[e]["r"]))
        n_avail[e] = len(buckets[e]["r"])
        cells[e] = (np.asarray(buckets[e]["r"][:n]), np.asarray(buckets[e]["x"][:n]))

    l3_node = cfg["l3_node"]
    r3, x3 = DM.context_instances_demand(
        rules, {"name": f"L3n{l3_node}", "level": 3, "nodes": [l3_node]}, cfg["n_e4"], s, depth,
        v, m, seed=cfg["seed"] + 510_000, st=uniform)
    _r3c, lv3 = DM.sample_pool_demand(rules, cfg["n_e4"], s, cfg["seed"] + 520_000, uniform)
    f3 = MC.exact_features(lv3, inv_bottom, v, s)
    span3 = s ** 2
    shared_m3 = MC.Miner(3, s)
    shared_m3.observe(f3[:, l3_node * span3:(l3_node + 1) * span3])
    mine_pool = {}
    for j in cfg["nodes"]:
        rm, xm = DM.context_instances_demand(
            rules, ctx(j), cfg["n_mine_post"] // len(cfg["nodes"]), s, depth, v, m,
            seed=cfg["seed"] + 610_000 + 53 * j, st=uniform)
        mine_pool[j] = (rm, xm)
    return {"uniform": uniform, "ctx": ctx, "entries": entries,
            "freq": {e: freq[e] / tot for e in entries}, "n_avail": n_avail,
            "cells": cells, "l3": (r3, x3, l3_node), "shared_m3": shared_m3,
            "mine_pool": mine_pool, "span": span,
            "keys": [(0, j) for j in cfg["nodes"]]}


def baseline_profile(shared, cfg, world, device):
    """`s0_e`: the SHARED setup policy's error per entry, before any arm has adapted. Identical
    for every arm by construction, which is what makes "holes at the policy's own weaknesses"
    separable from "holes at random"."""
    out = {}
    for e, (r_np, x_np) in world["cells"].items():
        n = min(96, x_np.shape[0])
        out[e] = solve(shared, cfg, shared["generator0"], shared["value0"], None,
                       r_np[:n], x_np[:n], device)["e"]
    return out


# --------------------------------------------------------------------------- #
# support policies: which of the instances the world freely offers get practised
# --------------------------------------------------------------------------- #

def support_weights(policy, cfg, state):
    """p over the level-2 entries, used to RESAMPLE a freshly drawn candidate batch. Note the
    realized entry distribution is `world_freq(e) * p(e)` normalised -- the policy re-weights
    what the world offers, it cannot conjure instances the world never produces."""
    n = len(state["entries"])
    eps, beta = state["eps"], state["beta"]
    if policy == "full":
        return np.ones(n)
    if policy == "narrow":
        w = np.full(n, 1e-9)
        w[state["narrow_set"]] = 1.0
        return w
    if policy == "visit":
        return (state["n_seen"] + eps) ** beta
    if policy == "solved":
        return (state["n_solved"] + eps) ** beta
    raise ValueError(policy)


def resample(ents_idx, w, n, rng):
    p = w[ents_idx]
    if p.sum() <= 0:
        p = np.ones_like(p)
    return rng.choice(len(ents_idx), size=n, replace=True, p=p / p.sum())


def perplexity(q):
    q = np.asarray(q, float)
    q = q / max(q.sum(), 1e-12)
    return float(np.exp(-sum(x * np.log(x) for x in q if x > 0)))


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

def run_arm_sp(label, base, overrides, shared, cfg, world, outdir, device, carry):
    import torch
    spec = ARMS_SP[base]
    cfg = {**cfg, **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    level, span = cfg["level"], world["span"]
    mult = cfg["dense_mult"] if spec.get("mult") else 1
    entries = world["entries"]
    eidx = {e: i for i, e in enumerate(entries)}
    E = len(entries)

    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p_ in generator.parameters():
        p_.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"],
                             weight_decay=cfg["weight_decay"])
    value = copy.deepcopy(shared["value0"]).to(device)
    for p_ in value.parameters():
        p_.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr_online"],
                             weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)
    krng = np.random.default_rng(cfg["seed"] + 5150)

    lib = LIB.KeyedLibrary(world["keys"], level, s, policy="global") if spec["index"] else None
    true_tbl = shared["truth"][level] if spec["vocab"] == "true" else None

    nk = int(round(carry.get("sp_entry_perplexity", max(2.0, E / 3.0))))
    nk = int(min(E, max(1, cfg["narrow_k"] or nk)))
    if str(cfg.get("narrow_fixed", "")).strip():
        # ADDITIVE, default "" -> `sp_s0`'s RNG draw is bit-identical. A named exogenous subset,
        # given as "-"-separated RANKS into `entries` (sorted by DESCENDING world mass), so a
        # narrow arm can be PLACEMENT-matched to a self-play arm's realized support instead of
        # landing wherever the RNG puts it. Still exogenous: fixed before the run, never adapted,
        # no competence coupling. ("-" not "," because `parse_arms` splits arms on ",".)
        narrow_set = np.array(sorted({int(z) for z in
                                      str(cfg["narrow_fixed"]).split("-") if z.strip()}))
        assert narrow_set.min() >= 0 and narrow_set.max() < E, (narrow_set, E)
    else:
        narrow_set = np.sort(np.random.default_rng(cfg["narrow_seed"]).permutation(E)[:nk])
    state = {"entries": entries, "n_seen": np.zeros(E), "n_solved": np.zeros(E),
             "eps": cfg["sp_eps"], "beta": spec.get("beta", cfg["sp_beta"]),
             "narrow_set": narrow_set}
    if spec["keyp"] == "narrow":
        print(f"  [narrow] {label}: {len(narrow_set)} fixed exogenous entries "
              f"{[entries[i] for i in narrow_set]} at mass ranks "
              f"{[int(z) for z in narrow_set]}", flush=True)

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    log = {"cycle": [], "e": [], "e_practice": [], "storage": [], "n_solved": [],
           "vloss": [], "gloss": [], "support_perp": [], "probe": []}
    t0 = time.time()

    for cyc in range(1, cfg["cycles"] + 1):
        w = support_weights(spec["keyp"], cfg, state)
        per_node = max(6, (cfg["n_pr"] * mult) // len(cfg["nodes"]))
        solved_all, e_pr = [], []
        for j in cfg["nodes"]:
            # a FRESH candidate batch from the world (uniform demand, untouched), then
            # resampled by this arm's support policy. Unmetered arms oversample and keep the
            # same number of instances, so the meter is identical across dense arms and only
            # the SHAPE of the realized support differs.
            n_cand = per_node * (1 if spec["keyp"] == "full" else cfg["sp_oversample"])
            r_all, x_all, cl = DM.context_instances_demand(
                shared["rules"], world["ctx"](j), n_cand, s, depth, v, m,
                seed=cfg["seed"] + 100_000 + 1000 * cyc + 13 * j, st=world["uniform"],
                with_clean=True)
            ce = entry_of(cl, j, span, shared["inverse_maps"][-1], v, s)
            ci = np.array([eidx[e] for e in ce])
            take = (np.arange(per_node) if spec["keyp"] == "full"
                    else resample(ci, w, per_node, krng))
            r_np, x_np, sel_e = r_all[take], x_all[take], ci[take]
            tbl = true_tbl if lib is None else lib.cells[0]["table"]
            o = solve(shared, cfg, generator, value, tbl, r_np, x_np, device,
                      width=cfg["pr_width"], collect=True)
            e_pr.append(o["e"])
            np.add.at(state["n_seen"], sel_e, 1)
            np.add.at(state["n_solved"], sel_e[o["succ"] > 0.5], 1)
            ok = torch.from_numpy(o["succ"] > 0.5).to(device)
            chosen = o["x"][ok]
            solved_all.append(chosen)
            tips = o["tips_x"]
            B, Wd, T = tips.shape
            tflat = tips.reshape(B * Wd, T)
            tsucc, _ = grade(tflat.cpu().numpy(), np.repeat(r_np, Wd), shared["rules"], s)
            xs = torch.cat([t.reshape(B * Wd, T).cpu() for t in o["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, Wd), len(o["traj"])))
            ys = torch.from_numpy(np.tile(tsucc.astype(np.float32), len(o["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            if lib is not None and chosen.shape[0]:
                pf = MC.parse_features(shared["reader"], chosen, s=s).cpu().numpy()
                lib.observe((0, j), pf[:, j * span:(j + 1) * span])

        solved = torch.cat(solved_all) if solved_all else torch.zeros(
            0, shared["length"], dtype=torch.long, device=device)
        if lib is not None:
            lib.rebuild(MC.base_table(v), cfg["mine_support"])
        gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                   shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                                   n_steps=cfg["gen_steps"] * mult, batch=cfg["batch_size"],
                                   replay_frac=cfg["replay_frac"], device=device, rng=grng)
        vloss = value_steps(value, vopt, shared["controller"], buf, shared["replay"],
                            n_steps=cfg["n_grad"] * mult, batch=cfg["value_batch"],
                            replay_frac=cfg["replay_frac"], device=device, rng=rng)

        probe = None
        if cyc % cfg["probe_every"] == 0 or cyc == cfg["cycles"]:
            sub = {str(e): (r[:cfg["n_probe"]], x[:cfg["n_probe"]])
                   for e, (r, x) in world["cells"].items()}
            probe = exam(shared, cfg, generator, value, lib, true_tbl, sub, device)["e"]
        log["cycle"].append(cyc); log["e"].append(float(np.mean(e_pr)))
        log["e_practice"].append(float(np.mean(e_pr)))
        log["storage"].append(0 if lib is None else lib.storage())
        log["n_solved"].append(int(solved.shape[0]))
        log["vloss"].append(vloss); log["gloss"].append(gloss); log["probe"].append(probe)
        log["support_perp"].append(perplexity(state["n_seen"]))
        if cyc % 10 == 0 or cyc == 1:
            print(f"  arm={label} c{cyc:3d} e_pr={np.mean(e_pr):.4f} "
                  f"supportPerp={log['support_perp'][-1]:.2f}/{E} "
                  f"store={log['storage'][-1]}", flush=True)

    # ---- the per-entry exam battery (identical instances for every arm) --------------------
    cells = {str(e): world["cells"][e] for e in entries}
    full = exam(shared, cfg, generator, value, lib, true_tbl, cells, device)
    per_entry = {str(e): full["per_key"][str(e)]["e"] for e in entries}

    q_seen = state["n_seen"] / max(1.0, state["n_seen"].sum())
    tables = {}
    if lib is None:
        t2_post, post_rec = post_mine_t2(shared, cfg, generator, value, true_tbl, device)
        tables["post"] = t2_post
        if true_tbl is not None:
            tables["held"] = true_tbl
    else:
        tables["held"] = lib.cells[0]["table"]
        t2_post, post_rec = post_mine_t2(shared, cfg, generator, value, tables["held"], device)
        tables["post"] = t2_post
    mine = minability(shared, cfg, generator, value, tables, device)

    held_set = set()
    if lib is not None and lib.cells[0]["table"] is not None:
        held_set = {tuple(int(z) for z in r) for r in lib.cells[0]["table"]["flat"]}
    out = {"arm": label, "base": base, "overrides": overrides, "log": log, "mult": mult,
           "beta": state["beta"], "entries": [str(e) for e in entries],
           "exam_all": {"e": full["e"], "per_key": full["per_key"]},
           "per_entry": per_entry, "n_seen": state["n_seen"].tolist(),
           "n_solved_by_entry": state["n_solved"].tolist(),
           "visitation": q_seen.tolist(), "perplexity": perplexity(state["n_seen"]),
           "n_zero_seen": int((state["n_seen"] == 0).sum()),
           "held_entries": [str(e) for e in entries if e in held_set],
           "narrow_set": [int(z) for z in state["narrow_set"]],
           "minability": mine, "post_mine": post_rec,
           "storage": 0 if lib is None else lib.storage(),
           "elapsed": time.time() - t0}
    if lib is not None and lib.cells[0]["table"] is not None:
        out["held_table"] = {"n_entries": int(lib.cells[0]["table"]["child"].shape[0]),
                             **{f"tab_{k}": val for k, val in
                                MC.grade_table(lib.cells[0]["table"],
                                               shared["truth"][2]).items()}}
    d = os.path.join(outdir, label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder)
    volume.commit()
    print(f"  arm={label} done in {out['elapsed']:.0f}s  exam={full['e']:.4f}  "
          f"perp={out['perplexity']:.2f}/{len(entries)}  "
          f"unseen entries={out['n_zero_seen']}  held |T2|={len(out['held_entries'])}",
          flush=True)
    return out


# --------------------------------------------------------------------------- #
# the main run
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def selfplay(
    tag: str = "smoke",
    arms: str = ("never_base,given,metered_glob,dense_full,dense_visit,dense_selfplay,"
                 "dense_sp_hard,dense_narrow"),
    seed: int = 0, cycles: int = 60, dense_mult: int = 4, n_pr: int = 72,
    sp_beta: float = 1.0, sp_eps: float = 1.0, sp_oversample: int = 6,
    narrow_seed: int = 7, narrow_k: int = 0, narrow_fixed: str = "",
    pool_per_node: int = 14_000,
    n_ex: int = 112, n_probe: int = 32, n_e4: int = 1024, n_mine_post: int = 1024,
    mine_support: int = 3, probe_every: int = 10, demand_seed: int = 0, quick: bool = False,
):
    import torch
    cfg = _sp_cfg(seed=seed, cycles=cycles, dense_mult=dense_mult, n_pr=n_pr, sp_beta=sp_beta,
                  sp_eps=sp_eps, sp_oversample=sp_oversample, narrow_seed=narrow_seed,
                  narrow_k=narrow_k, narrow_fixed=narrow_fixed,
                  pool_per_node=pool_per_node, n_ex=n_ex, n_probe=n_probe,
                  n_e4=n_e4, n_mine_post=n_mine_post, mine_support=mine_support,
                  probe_every=probe_every, demand_seed=demand_seed)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800, reader_steps=600,
                   value_episodes=6_000, n_train_episodes=20_000, cycles=12, n_pr=24,
                   n_ex=32, n_probe=16, n_e4=96, n_mine_post=192, dense_mult=2,
                   pool_per_node=2_500, gen_steps=5, probe_every=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"selfplay tag={tag} arms={arms} device={device}")
    print(f"  world: UNIFORM demand (free i.i.d. variation), nodes {cfg['nodes']}; support "
          f"axis = the level-2 entry; beta {cfg['sp_beta']} eps {cfg['sp_eps']} "
          f"oversample {cfg['sp_oversample']}")

    shared = build_shared(cfg, device)
    shared["truth3"] = MC.true_tables(shared["rules"], cfg["depth"], cfg["s"], cfg["v"],
                                      cfg["m"], 3)[3]
    world = build_world_sp(shared, cfg)
    shared["_l3"] = world["l3"]
    shared["_mine_pool"] = world["mine_pool"]
    shared["_shared_m3"] = world["shared_m3"]
    gate = gate_world(v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
                      rule_seed=cfg["rule_seed"], sigma=1.0, kappa=cfg["kappa"],
                      n_venue=4, train_nodes=(1, 2, 3), hold_node=0, level=cfg["level"],
                      demand_levels=cfg["demand_levels"], seed=cfg["seed"], n=2048)
    s0 = baseline_profile(shared, cfg, world, device)
    fr = np.array([world["freq"][e] for e in world["entries"]])
    s0v = np.array([s0[e] for e in world["entries"]])
    print(f"[gate] world {json.dumps({k: (val.get('pass') if isinstance(val, dict) else val) for k, val in gate.items()})}")
    print(f"[entries] {len(world['entries'])} level-2 entries; world mass "
          f"{fr.min():.4f}-{fr.max():.4f} (perplexity {perplexity(fr):.2f}); "
          f"exam cells {min(world['n_avail'].values())}-{max(world['n_avail'].values())} "
          f"instances available")
    print(f"[entries] s0 (shared setup-policy error) {s0v.min():.3f}-{s0v.max():.3f}; "
          f"corr(world mass, s0) = {np.corrcoef(fr, s0v)[0, 1]:+.3f}")

    outdir = f"{DATA_DIR}/rhm_practice_selfplay/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "gate": gate, "entries": [str(e) for e in world["entries"]],
                   "freq": {str(e): world["freq"][e] for e in world["entries"]},
                   "n_avail": {str(e): world["n_avail"][e] for e in world["entries"]},
                   "s0": {str(e): s0[e] for e in world["entries"]}},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    carry, results = {}, {}
    for label, base, ov in parse_arms(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        r = run_arm_sp(label, base, ov, shared, cfg, world, outdir, device, carry)
        results[label] = r
        if base == "dense_selfplay" and "sp_entry_perplexity" not in carry:
            carry["sp_entry_perplexity"] = r["perplexity"]
            print(f"  [carry] dense_narrow matched to entry perplexity {r['perplexity']:.2f} "
                  f"-> {int(round(r['perplexity']))} entries", flush=True)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"elapsed": time.time() - started}


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def selfcheck_sp():
    """Structural gates with no substrate: the support policies and the resampler."""
    E = 8
    ents = [(i, i) for i in range(E)]
    st = {"entries": ents, "n_seen": np.zeros(E), "n_solved": np.zeros(E), "eps": 1.0,
          "beta": 1.0, "narrow_set": np.array([2, 5])}
    cfg = _sp_cfg()

    w = support_weights("full", cfg, st)
    assert np.allclose(w, 1.0)
    w = support_weights("narrow", cfg, st)
    assert set(np.flatnonzero(w > 1e-6)) == {2, 5}
    print("S-1 support policies ok")

    # at zero history every policy is the world's own distribution -- the footprint has to be
    # EARNED, there is no exogenous seed
    assert np.allclose(support_weights("solved", cfg, st), 1.0)
    st["n_solved"] = np.array([400.0, 300.0, 20.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    st["n_seen"] = np.array([420.0, 330.0, 60.0, 40.0, 12.0, 9.0, 8.0, 6.0])
    ps = support_weights("solved", cfg, st); ps = ps / ps.sum()
    pv = support_weights("visit", cfg, st); pv = pv / pv.sum()
    assert ps[0] > ps[2] > ps[4]
    # the COMPETENCE coupling: entry 3 was seen 40x and solved once, so `solved` starves it far
    # harder than `visit` does. That difference is the whole self-play-vs-rich-get-richer axis.
    assert ps[3] / ps[0] < pv[3] / pv[0], (ps[3] / ps[0], pv[3] / pv[0])
    print(f"S-2 competence coupling ok: entry seen 40x/solved 1x gets "
          f"{ps[3]:.4f} under `solved` vs {pv[3]:.4f} under `visit`")

    st2 = dict(st, beta=2.0)
    p2 = support_weights("solved", cfg, st2); p2 = p2 / p2.sum()
    assert perplexity(p2) < perplexity(ps)
    print(f"S-3 beta is the amplifier: perplexity {perplexity(ps):.2f} (beta 1) -> "
          f"{perplexity(p2):.2f} (beta 2) of {E}")

    # the resampler reproduces the requested weights on a candidate batch
    rng = np.random.default_rng(0)
    ci = rng.integers(0, E, size=6000)
    take = resample(ci, ps * 1.0, 4000, rng)
    got = np.bincount(ci[take], minlength=E) / 4000
    want = ps * np.bincount(ci, minlength=E) / len(ci)
    want = want / want.sum()
    assert np.abs(got - want).max() < 0.03, (got, want)
    print("S-4 resampler realizes world_freq * p(e) ok")

    g = gate_world(n=2048)
    assert g["all_pass"]
    print("S-5 world gate (inherited from merge.py) passes")
    print("\nALL SELFPLAY SELFCHECKS PASS")
    return {}


@app.function(image=image, timeout=3600, memory=16384)
def selfcheck_sp_remote():
    return selfcheck_sp()


@app.local_entrypoint()
def main():
    selfcheck_sp_remote.remote()
