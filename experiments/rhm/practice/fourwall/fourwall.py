"""fourwall -- a free spurious key, and paying it down by coarsening the index.

Round 8 of the practice arc on RHM (etude -> crystallize -> ratchet -> ear -> recital -> tall ->
transpose / setlist -> here). The sibling of `typed_gaps`' crossed pair, aimed one level out.

WHERE THIS COMES FROM. `transpose` moved WHAT IS TRUE and found committed chunks structurally
deaf to it; `setlist` moved WHAT IS ASKED and woke the verification channel (9/62 recert swaps,
every one coverage-improving). Both moved the CONTENT the library is graded on. This round holds
content and demand fixed and moves the INDEX: every instance carries a surface token `w` that,
in phase 1, names for free the level-2 feature the damage destroyed -- the perfect frozen
correlate that manufactured recurrence hands you whether you asked for it or not -- and a
rotation event then permutes the `w` <-> feature map without deleting `w`. Nothing becomes
false and nothing about the demand over features moves; the held-out instances are literally the
same objects at every cycle. Only the meaning of a free observable rotates.

THE FOUR ARMS (plus one confound control), differing ONLY in what the library is keyed on:

    given_key      key = the true level-2 latent, oracle-supplied.  Rotation-invariant by
                   construction: the upper anchor for an index that is right.
    wall_track     key = `w`.  Under rotation it may re-mine and re-select per w-cell -- the
                   `setlist` maintenance organ, pointed at a regime it was not built for.
    wall_merge     key = `w`, plus MERGE: the first op in the arc that acts on the index rather
                   than on an entry. Coarsens the partition when the exact forced-transfer
                   evidence says two cells are aliases.
    free_selector  no assigned key (one global vocabulary); instead the SELECTOR reads `w` as an
                   extra conditioning input. Measures spontaneous binding, and is the null
                   competitor for "does plain learning reach the quotient with no index op?".
    unkeyed        the same single global vocabulary with a selector that CANNOT see `w`. Not in
                   the original four: it is the control that separates free_selector's two
                   differences (no index vs a selector that reads the wall).

WHY CONTAMINATION IS IMPOSSIBLE HERE, DELIBERATELY. A committed unit is a table of level-1
feature tuples. There is no slot in it for `w`. So `w` can only enter through the index (the
keyed arms) or through the selector's input (free_selector), never through content -- which
makes any wall_track-vs-wall_merge difference PURE INDEX COMPRESSION. Arms whose programs read
`w` are a deliberate non-goal of this round, as are drift-rate sweeps and paced rotation.

Run from experiments/:            # MODAL_PROFILE=chromatic
  modal run rhm/practice/fourwall/fourwall.py::selfcheck_remote
  modal run rhm/practice/fourwall/fourwall.py::gate0 --tag g0
  python3 rhm/practice/fourwall/launch_detached.py --fn fourwall --tag fw_s0 ...
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.practice.crystallize.units import (
    build_move_set, grade, on_grammar_rate, oracle_rollout)
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    audition_macro, beam_moves, build_ms, era_ctx, finetune_generator, fit_width,
    parse_eras, plant_probe, priced, push, value_steps)
from rhm.practice.ear.ear import _shared_plus, write_results
from rhm.practice.setlist import demand as DM
from rhm.practice.fourwall import rekey as RK
from rhm.practice.fourwall import wall as WL
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.rhm_sculpt_precheck import nearest_derivation_cost


app = modal.App("rhm-practice-fourwall", image=image)

REMOTE = "rhm_practice_fourwall"


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

_A = {"key": "wall", "single": False, "merge": False, "wall_value": False, "rekey": False,
      "retire": None}
ARMS = {
    "given_key":     {**_A, "key": "latent"},
    "wall_track":    {**_A},
    "wall_merge":    {**_A, "merge": True},
    "unkeyed":       {**_A, "key": "none", "single": True},
    "free_selector": {**_A, "key": "none", "single": True, "wall_value": True},
    # -- fw_s1: the index's basis EARNED rather than only pruned ----------------------
    "wall_rekey":    {**_A, "rekey": True},
    "scratch_rekey": {**_A, "key": "none", "single": True, "rekey": True},
    # -- fw_s3: what to do with the scaffold once it has been migrated off ------------
    "rekey_mothball": {**_A, "rekey": True, "retire": "mothball"},
    "rekey_delete":   {**_A, "rekey": True, "retire": "delete"},
}


def arm_keys(spec, feat, q, v):
    """The key the arm's library is indexed on. `wall` is the free surface correlate; `latent`
    is the oracle; `none` routes everything to one cell (the key array is still returned so the
    routing code is identical across arms)."""
    if spec["key"] == "wall":
        return (np.asarray(feat, np.int64) + int(q)) % v
    return np.asarray(feat, np.int64)


# --------------------------------------------------------------------------- #
# the selector that can read the wall
# --------------------------------------------------------------------------- #

def build_wall_head(base_value, v, state_dim):
    """`MCValueHead` plus a wall embedding, ZERO-INITIALISED and added to the root embedding.

    At cycle 1 this is bit-identically the plain head as a function, so `free_selector` and
    `unkeyed` start from exactly the same selector and every difference between them is
    acquired online. That is the whole point: the arm measures SPONTANEOUS binding, so the wall
    must buy nothing at initialisation."""
    import torch
    import torch.nn as nn

    class WallHead(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.root_embedding = base.root_embedding
            self.net = base.net
            self.wall_embedding = nn.Embedding(v + 1, state_dim)
            nn.init.zeros_(self.wall_embedding.weight)

        def forward(self, z, root, wall):
            e = self.root_embedding(root) + self.wall_embedding(wall)
            return self.net(torch.cat([z, e], dim=-1)).squeeze(-1)

    return WallHead(copy.deepcopy(base_value))


def build_wall_value(base_value, v, state_dim):
    """A `value(z, root)`-shaped wrapper so `ratchet.beam_moves` is used UNMODIFIED.

    `beam_moves` always calls the value with `roots.repeat_interleave(k)` for an integer k, so
    the wrapper reconstructs the matching wall vector by the same repeat. The sentinel index
    `v` means "no wall" and is what the replay buffer carries."""
    import torch
    import torch.nn as nn

    class WallValue(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head
            self.v = v
            self.w = None

        def set_w(self, w):
            self.w = w

        def forward(self, z, root):
            if self.w is None:
                w = torch.full_like(root, self.v)
            else:
                k = root.shape[0] // self.w.shape[0]
                w = self.w.repeat_interleave(k) if k > 1 else self.w
            return self.head(z, root, w)

    return WallValue(build_wall_head(base_value, v, state_dim))


def value_steps_w(value, opt, controller, buf, replay, *, n_steps, batch, replay_frac,
                  device, rng, v):
    """`ratchet.value_steps` with the wall threaded through. Replay rows carry the `v` sentinel
    (the generic pretraining pool has no wall), fresh rows carry the real one."""
    import torch
    import torch.nn.functional as F
    if buf["x"].shape[0] == 0:
        return 0.0
    value.train()
    n_rep = int(round(batch * replay_frac)) if replay["x"].shape[0] else 0
    n_new = batch - n_rep
    last = 0.0
    for _ in range(n_steps):
        px, pr, py, pw = [], [], [], []
        if n_new:
            i = torch.from_numpy(rng.integers(0, buf["x"].shape[0], size=n_new))
            px.append(buf["x"][i]); pr.append(buf["r"][i]); py.append(buf["y"][i])
            pw.append(buf["w"][i])
        if n_rep:
            i = torch.from_numpy(rng.integers(0, replay["x"].shape[0], size=n_rep))
            px.append(replay["x"][i]); pr.append(replay["r"][i]); py.append(replay["y"][i])
            pw.append(torch.full((n_rep,), v, dtype=torch.long))
        x = torch.cat(px).to(device); r = torch.cat(pr).to(device)
        y = torch.cat(py).to(device); w = torch.cat(pw).to(device)
        with torch.no_grad():
            z = controller.state(x)
        value.set_w(w)
        loss = F.binary_cross_entropy_with_logits(value(z, r), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    value.set_w(None)
    value.eval()
    return last


# --------------------------------------------------------------------------- #
# the world: one grammar, one demand, one set of instances, a rotating index
# --------------------------------------------------------------------------- #

def build_world(shared, cfg, device):
    """Everything the run will ever be graded on, drawn ONCE.

    This is the round's admissibility gate discharged by construction rather than by
    measurement. `setlist` had to prove that demand moved while difficulty did not; here the
    metering instances, their roots, their damage and their latent keys are the SAME ARRAYS at
    every cycle of every arm. Nothing about the world can move, because there is nothing to
    move: the rotation is a relabelling of an observable that no grading path reads."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, inv_bottom = shared["rules"], shared["inverse_maps"][-1]
    on = cfg["demand_sigma"] > 0
    st, start_stats = None, None
    if on:
        levels = [int(x) for x in str(cfg["demand_levels"]).split("/")]
        st, start_stats = DM.typical_demand(
            rules, levels, s, depth, v, m, inv_bottom, seed=cfg["demand_seed"],
            kappa=cfg["demand_kappa"], sigma=cfg["demand_sigma"],
            n_cand=cfg["demand_n_cand"], n=2048)
        print(f"[demand] typical start: {json.dumps(start_stats, cls=NumpyEncoder)}", flush=True)

    era = parse_eras(cfg["era"])[0]
    deep = parse_eras(cfg["deep_era"])[0]
    kl, kn = cfg["key_level"], cfg["key_node"]

    def draw(ctx, n, seed):
        r, x, cl, f = WL.context_instances_wall(rules, ctx, n, s, depth, v, m, seed, st, kl, kn)
        return {"r": r, "x": x, "clean": cl, "f": f}

    world = {"demand": st, "demand_stats": start_stats, "era": era, "deep_era": deep,
             "meter": draw(era_ctx(era), cfg["n_rt"], cfg["seed"] + 5000),
             "deep": draw(era_ctx(deep), cfg["n_deep"], cfg["seed"] + 5500)}
    world["probe_clean"] = torch.from_numpy(
        (WL.sample_pool_latent(rules, cfg["n_probe_clean"], s, cfg["seed"] + 31337, st)[1]
         if on else _sample_pool(rules, cfg["n_probe_clean"], s, cfg["seed"] + 31337)[1]))
    # the demand over the key, and over the level-2 vocabulary -- logged, never consumed
    fm = world["meter"]["f"]
    world["key_mass"] = {str(k): float((fm == k).mean()) for k in range(v)}
    world["key_entropy"] = float(-sum(p * np.log(p) for p in
                                      (np.bincount(fm, minlength=v) / fm.size) if p > 0))
    return world


def true_key_table(truth2, f, s):
    """The oracle keyed vocabulary for feature `f`: exactly its `m` synonymous expansions.
    This is what a perfect index buys, and the reference `given_key` is chasing."""
    keep = np.flatnonzero(truth2["feature"] == int(f))
    return MC.make_table(2, truth2["child"][keep], truth2["lower"], s,
                         truth2["feature"][keep])


# --------------------------------------------------------------------------- #
# GATE 0: is the level-2 identity expensive to infer WITHOUT the wall?
# --------------------------------------------------------------------------- #

def wall_refs(shared, cfg, world, device):
    """The references, and Gate 0.

    If the parse is nearly free -- if the key is determined by the observation, or if a keyed
    true table buys nothing over the unkeyed one -- then the wall is a free correlate of
    something nobody needed, phase 1 is vacuous by construction, and this design is the wrong
    instrument on this substrate. That verdict has to be reachable BEFORE the main run, so it
    is measured here and printed before any arm starts."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    canon_np = shared["canon_np"]
    truth2 = shared["truth"][2]
    kl, kn = cfg["key_level"], cfg["key_node"]
    M = world["meter"]
    x0 = torch.from_numpy(M["x"])
    out = {"n": int(M["x"].shape[0]),
           "d0": float(nearest_derivation_cost(rules, M["x"], M["r"], s).mean()),
           "on_grammar": on_grammar_rate(M["x"], shared["inverse_maps"][-1], v, s),
           "key_entropy": world["key_entropy"], "key_mass": world["key_mass"]}

    # -- (a) the exact information content of the key ---------------------------------
    cons, _g = WL.consistent_features(rules, M["x"], M["r"], kn, kl, s, canon_np, v)
    out["mean_consistent"] = float(cons.sum(1).mean())
    out["p_random_feature"] = float(cons.mean())
    out["frac_uniquely_determined"] = float((cons.sum(1) == 1).mean())
    out["true_f_consistent"] = float(cons[np.arange(cons.shape[0]), M["f"]].mean())
    out["consistent_hist"] = np.bincount(cons.sum(1), minlength=v + 1).tolist()

    # -- (b) what a right index is worth, on the DGP's own tables ----------------------
    mv = MC.to_device(MC.make_macro(kl, kn, s, truth2), device)
    out["e_act_true_unkeyed"] = audition_macro(shared["generator0"], x0.to(device), M["r"], mv,
                                               rules_t, canon, depth, v, m, s, rules)["e"]
    # the keyed oracle, as ONE ACTION: each instance gets the true table of its own feature
    xf = torch.zeros_like(x0)
    for f in range(v):
        idx = np.flatnonzero(M["f"] == f)
        if idx.size == 0:
            continue
        mvf = MC.to_device(MC.make_macro(kl, kn, s, true_key_table(truth2, f, s)), device)
        xf[torch.from_numpy(idx)] = MC.apply_any(
            shared["generator0"], torch.from_numpy(M["x"][idx]).to(device), mvf, rules_t,
            canon, depth, v, m, s).cpu()
    sc, _ = grade(xf.numpy(), M["r"], rules, s)
    out["e_act_true_keyed"] = 1.0 - float(sc.mean())

    # -- the same references under the FULL POLICY, with and without a NOOP -------------
    # A keyed table can one-shot this cell, and without a NOOP every realisation is exactly
    # `budget` moves long (`units.build_move_set`'s own note), so the moves after the repair
    # are pure downside. Whether the selector can hold a finished configuration is an
    # empirical question about THIS substrate, so both action spaces are measured here and the
    # main run takes whichever the measurement says.
    for noop in (False, True):
        bms = build_move_set(depth, s, max_level=1, with_noop=noop)
        sfx = "_noop" if noop else ""
        variants = {"base": list(bms), "true_unkeyed": list(bms) + [mv],
                    "true_allnodes": build_ms(bms, {2: truth2}, s, depth, device)}
        for tag, mset in variants.items():
            w = fit_width(len(mset), cfg["budget"], cfg["g_budget"])
            b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                           torch.from_numpy(M["r"]), mset, rules_t, canon, depth, v, m, s,
                           budget=cfg["budget"], beam_width=w, device=device)
            sc, _ = grade(b["x"].cpu().numpy(), M["r"], rules, s)
            out[f"e_pol_{tag}{sfx}"] = 1.0 - float(sc.mean())
            out[f"width_{tag}{sfx}"] = int(w)
            out[f"g_{tag}{sfx}"] = b["counts"]["ground"] / x0.shape[0]
        pol_ok = np.zeros(x0.shape[0])
        for f in range(v):
            idx = np.flatnonzero(M["f"] == f)
            if idx.size == 0:
                continue
            t = true_key_table(truth2, f, s)
            msk = list(bms) + [MC.to_device(MC.make_macro(kl, kn, s, t), device)]
            wj = fit_width(len(msk), cfg["budget"], cfg["g_budget"])
            b = beam_moves(shared["controller"], shared["generator0"], shared["value0"],
                           torch.from_numpy(M["x"][idx]), torch.from_numpy(M["r"][idx]), msk,
                           rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                           beam_width=wj, device=device)
            sc, _ = grade(b["x"].cpu().numpy(), M["r"][idx], rules, s)
            pol_ok[idx] = sc
        out[f"e_pol_true_keyed{sfx}"] = 1.0 - float(pol_ok.mean())
    base_ms = shared["base_ms"]

    # -- (c) does the unkeyed DP actually recover the identity? -------------------------
    wrote = MC.exact_features(MC.apply_any(shared["generator0"], x0.to(device), mv, rules_t,
                                           canon, depth, v, m, s).cpu().numpy(),
                              shared["inverse_maps"][-1], v, s)
    span = s ** (kl - 1)
    got = wrote[:, kn * span:(kn + 1) * span]
    codes = (got * (v ** np.arange(span))).sum(-1)
    out["id_acc_unkeyed_dp"] = float((shared["inverse_maps"][depth - kl][codes] == M["f"]).mean())

    # -- (d) the exact-DP floor over base moves ----------------------------------------
    xo, _ = oracle_rollout(shared["generator0"], x0.to(device), M["r"], base_ms, rules,
                           rules_t, canon, depth, v, m, s, cfg["budget"])
    sc, _ = grade(xo.cpu().numpy(), M["r"], rules, s)
    out["floor_base"] = 1.0 - float(sc.mean())

    # -- (e) how expensive is the parse for a LEARNER? ----------------------------------
    out["probe"] = key_probe(shared, cfg, world, device)
    return out


def key_probe(shared, cfg, world, device, n=16384, n_hold=4096, steps=1500):
    """A linear read of the key off the agent's OWN state, trained on fresh damaged instances.

    `mean_consistent` says how much information the observation carries about the key; this
    says how much of it a linear probe on the controller's state can actually get at. Between
    them they price the parse the wall makes unnecessary."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules = shared["rules"]
    era = world["era"]
    r, x, _cl, f = WL.context_instances_wall(rules, era_ctx(era), n + n_hold, s, depth, v, m,
                                             cfg["seed"] + 424242, world["demand"],
                                             cfg["key_level"], cfg["key_node"])
    with torch.no_grad():
        zs = []
        for i in range(0, x.shape[0], 4096):
            zs.append(shared["controller"].state(torch.from_numpy(x[i:i + 4096]).to(device)))
        z = torch.cat(zs)
    rr = torch.from_numpy(r).to(device)
    ff = torch.from_numpy(f).to(device)
    emb = nn.Embedding(v, cfg["state_dim"]).to(device)
    head = nn.Linear(cfg["state_dim"], v).to(device)
    opt = torch.optim.AdamW(list(head.parameters()) + list(emb.parameters()), lr=3e-3)
    g = torch.Generator(device="cpu").manual_seed(7)
    for _ in range(steps):
        i = torch.randint(0, n, (512,), generator=g).to(device)
        loss = F.cross_entropy(head(z[i] + emb(rr[i])), ff[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(z[n:] + emb(rr[n:])).argmax(-1)
        acc = float((pred == ff[n:]).float().mean())
        top = float(torch.bincount(ff[n:], minlength=v).max() / ff[n:].shape[0])
    return {"acc": acc, "majority": top, "n_train": n, "n_hold": int(ff[n:].shape[0])}


# --------------------------------------------------------------------------- #
# the keyed beam
# --------------------------------------------------------------------------- #

def beam_keyed(shared, generator, value, x_np, r_np, cls_ids, action, walls, cfg, device,
               collect=False):
    """One beam per distinct ACTION SET, over the classes that hold it, reassembled in order.

    Beam width is `fit_width` of that action set's move count, so pricing stays the arc's: a
    level move costs the same groundings whoever holds it, and a class that has not committed
    yet simply runs the base action space. Classes that share an action set (every uncommitted
    class does) share a beam, so the index costs GPU time only where it actually differs."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    ms_by_gid, gid_of_cls = action
    gids = np.asarray([gid_of_cls[int(c)] for c in np.asarray(cls_ids)], np.int64)
    x_out = torch.zeros(x_np.shape[0], x_np.shape[1], dtype=torch.long)
    counts = {"mat": 0, "ground": 0}
    groups = []
    for g in sorted(set(int(k) for k in gids)):
        idx = np.flatnonzero(gids == g)
        if idx.size == 0:
            continue
        ms = ms_by_gid[g]
        width = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
        if walls is not None:
            value.set_w(torch.from_numpy(walls[idx]).to(device))
        out = beam_moves(shared["controller"], generator, value,
                         torch.from_numpy(x_np[idx]), torch.from_numpy(r_np[idx]), ms,
                         shared["rules_t"], shared["canon"], depth, v, m, s,
                         budget=cfg["budget"], beam_width=width, device=device,
                         collect=collect)
        x_out[torch.from_numpy(idx)] = out["x"].cpu()
        counts["mat"] += out["counts"]["mat"]
        counts["ground"] += out["counts"]["ground"]
        groups.append((idx, out, width))
    if walls is not None:
        value.set_w(None)
    return {"x": x_out, "counts": counts, "groups": groups}


def encode(shared, x_np, device):
    """The agent's own state for a batch — what the router reads, and the same encoding the
    beam already computes internally."""
    import torch
    with torch.no_grad():
        zs = [shared["controller"].state(torch.from_numpy(x_np[i:i + 4096]).to(device))
              for i in range(0, x_np.shape[0], 4096)]
    return torch.cat(zs)


def route(lib, spec, feat, q, v, router, enabled, conf_thr, z, r_t, ignore_active=False):
    """Where an instance's unit is looked up. The wall (or merged) index is the incumbent; an
    earned class takes over where the router is confident, the migration has been adopted, and
    the revert net has not pulled that class. `ignore_active` is for the GATE, which asks what
    the earned basis *would* do before any of it has been adopted."""
    cls = lib.classes_of(arm_keys(spec, feat, q, v)).copy()
    mig = np.zeros(cls.shape[0], bool)
    eids = lib.earned_ids()
    if enabled and router is not None and router.head is not None and eids:
        live = set(eids if ignore_active else lib.active_earned_ids())
        pc, conf = router.predict(z, r_t)
        pcn = pc.cpu().numpy()
        arr = np.asarray(eids, np.int64)
        ok = (conf.cpu().numpy() >= conf_thr) & (pcn < len(eids))
        ok &= np.array([bool(arr[i] in live) if k else False
                        for i, k in zip(np.clip(pcn, 0, len(eids) - 1), ok)])
        cls[ok] = arr[pcn[ok]]
        mig = ok
    return cls, mig


def build_action(lib, base_ms, s, depth, device):
    """The action set per index class: base moves, plus the class's committed macro AT THE
    DAMAGED NODE ONLY. `ratchet.macro_moves` instantiates a macro at every node of its level
    because its vocabulary is position-independent; a KEYED library is not -- its entry is what
    to write at the node the key is about -- so the macro is placed there and nowhere else.

    Returns `(ms_by_gid, gid_of_cls)`; every uncommitted class shares gid -1 and the base set."""
    ms_by_gid = {-1: list(base_ms)}
    gid_of_cls = {-1: -1}          # a DELETED wall cell leaves its keys with no unit at all
    for c in lib.class_ids():
        t = lib.table(c)
        if t is None:
            gid_of_cls[c] = -1
        else:
            gid_of_cls[c] = c
            ms_by_gid[c] = list(base_ms) + [
                MC.to_device(MC.make_macro(lib.level, lib.node, s, t), device)]
    return ms_by_gid, gid_of_cls


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

def run_arm(label, base, overrides, shared, cfg, refs, outdir, device, world):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **overrides, "arm_base": base}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules = shared["rules"]
    canon_np, base_ms = shared["canon_np"], shared["base_ms"]
    controller, truth2 = shared["controller"], shared["truth"][2]
    inv_key = shared["inverse_maps"][depth - cfg["key_level"]]
    kl, kn = cfg["key_level"], cfg["key_node"]
    era = world["era"]
    M, D = world["meter"], world["deep"]
    dem = world["demand"]

    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    if spec["wall_value"]:
        value = build_wall_value(shared["value0"], v, cfg["state_dim"]).to(device)
    else:
        value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    # The wall embedding starts at ZERO and has ~`n_grad` * `max_cycles` steps to become
    # comparable in scale to a root embedding that was trained for 12k. At the shared online
    # lr it structurally cannot, so a null on this arm would measure the budget rather than the
    # binding. It therefore gets its own param group. The asymmetry runs in the conservative
    # direction: `free_selector` is the null competitor, so giving it the better shot is the
    # right way to be wrong.
    wall_ps = [p for n, p in value.named_parameters() if "wall_embedding" in n]
    rest = [p for n, p in value.named_parameters() if "wall_embedding" not in n]
    vopt = torch.optim.AdamW(
        [{"params": rest, "lr": cfg["value_lr_online"]},
         {"params": wall_ps, "lr": cfg["value_lr_online"] * cfg["wall_lr_mult"]}],
        lr=cfg["value_lr_online"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)

    lib = WL.Library(v, s, kl, kn, decay=cfg["mine_decay"], floor=cfg["mine_floor"],
                     support=cfg["mine_support"], ring_cap=cfg["ring_cap"],
                     single=spec["single"])

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0),
           "w": torch.zeros(0, dtype=torch.long)}
    # the re-key bank: recent practice instances, pooled across cells, so `scratch_rekey`
    # (one cell) and `wall_rekey` (eight) draw their basis from the same amount of experience
    bank = {"x": np.zeros((0, shared["length"]), np.int64), "r": np.zeros(0, np.int64),
            "f": np.zeros(0, np.int64)}
    router = RK.Router(cfg["state_dim"], v, device) if spec["rekey"] else None
    rekey_on, rekey_since, revert_streak, share_hist = False, None, {}, {}
    t_cum = 0.0
    events, log = [], {
        "cycle": [], "q": [], "phase": [], "t_cum": [], "e": [], "e_practice": [],
        "succ": [], "dres": [], "vloss": [], "gloss": [], "n_solved": [], "n_mined": [],
        "lib": [], "gcost": [], "probe": [], "cover": [], "cell_e": [],
        "mig": [], "rekey": [], "fallback": [], "spend": [],
    }
    print(f"\n----- arm={arm} key={spec['key']} single={spec['single']} "
          f"merge={spec['merge']} wall_value={spec['wall_value']} -----", flush=True)

    q_prev, persist = 0, {}
    for cyc in range(1, cfg["max_cycles"] + 1):
        counts = {"mat": 0, "ground": 0}
        gcost = {}
        q = WL.rotation_index(cyc, cfg["phase1"], cfg["rot_period"], cfg["rot_step"])
        rotated = (q != q_prev)
        phase = 1 if cyc <= cfg["phase1"] else 2

        # --- (a0) THE WALL TURNS. Unannounced: nothing below reads `rotated`. -----------
        if rotated:
            prof = rotation_profile(lib, rules, canon_np, M, spec, q_prev, q, v, cfg)
            events.append({"kind": "rotation", "arm": arm, "cycle": cyc, "q_from": int(q_prev),
                           "q_to": int(q), "profile": prof})
            print(f"[rot] arm={arm} c{cyc} q {q_prev}->{q}", flush=True)
            q_prev = q

        action = build_action(lib, base_ms, s, depth, device)

        # --- (a) practice -------------------------------------------------------------
        pr, px, _pc, pf = WL.context_instances_wall(
            rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
            cfg["seed"] + 100_000 + 1000 * cyc, dem, kl, kn)
        pz = encode(shared, px, device) if spec["rekey"] else None
        pcls, pmig = route(lib, spec, pf, q, v, router, rekey_on, cfg["rekey_conf"], pz,
                           torch.from_numpy(pr).to(device) if spec["rekey"] else None)
        if spec["rekey"]:
            counts["ground"] += px.shape[0]
        pw = (pf + q) % v
        out = beam_keyed(shared, generator, value, px, pr, pcls, action,
                         pw if spec["wall_value"] else None, cfg, device, collect=True)
        counts["mat"] += out["counts"]["mat"]; counts["ground"] += out["counts"]["ground"]
        tip_solved = []
        for idx, o, _w in out["groups"]:
            tips = o["tips_x"]
            B, W, Tlen = tips.shape
            flat = tips.reshape(B * W, Tlen)
            succ, _ = grade(flat.cpu().numpy(), np.repeat(pr[idx], W), rules, s)
            xs = torch.cat([t.reshape(B * W, Tlen).cpu() for t in o["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(pr[idx], W), len(o["traj"])))
            ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(o["traj"])))
            ws = torch.from_numpy(np.tile(np.repeat(pw[idx], W), len(o["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            buf["w"] = torch.cat([buf["w"], ws])[-cfg["buf_cap"]:]
            tip_solved.append(flat[torch.from_numpy(succ > 0.5).to(flat.device)].cpu())
        solved = torch.cat(tip_solved) if tip_solved else torch.zeros(0, shared["length"],
                                                                     dtype=torch.long)
        ps, _ = grade(out["x"].numpy(), pr, rules, s)
        e_practice = 1.0 - float(ps.mean())

        # --- (b) mine, per index class ------------------------------------------------
        lib.age()
        ok = np.flatnonzero(ps > 0.5)
        if cfg["mine_cap"] and ok.size > cfg["mine_cap"]:
            ok = ok[rng.permutation(ok.size)[:cfg["mine_cap"]]]
        n_mined = int(ok.size)
        if ok.size:
            span = s ** (kl - 1)
            rows = MC.parse_features(shared["reader"],
                                     out["x"][torch.from_numpy(ok)].to(device),
                                     s=s).cpu().numpy()[:, kn * span:(kn + 1) * span]
            for c in sorted(set(int(z) for z in pcls[ok])):
                # a retired cell takes no further mining -- that is what "maintenance stops"
                # means, and it is half of what the retire op is being priced for
                if c in lib.cells and lib.maintained(c):
                    lib.observe(c, rows[pcls[ok] == c])
        for c in sorted(set(int(z) for z in pcls)):
            if c not in lib.cells:
                continue
            sel = np.flatnonzero(pcls == c)
            lib.push_ring(c, px[sel], pr[sel], pf[sel])
        if spec["rekey"]:
            for k, val in (("x", px), ("r", pr), ("f", pf)):
                bank[k] = np.concatenate([bank[k], val])[-cfg["rekey_bank"]:]

        # --- (c) the plant learns, then the selector ----------------------------------
        gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                   shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                                   n_steps=cfg["gen_steps"], batch=cfg["batch_size"],
                                   replay_frac=cfg["replay_frac"], device=device, rng=grng)
        if spec["wall_value"]:
            vloss = value_steps_w(value, vopt, controller, buf, shared["replay"],
                                  n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                  replay_frac=cfg["replay_frac"], device=device, rng=rng, v=v)
        else:
            vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                replay_frac=cfg["replay_frac"], device=device, rng=rng)

        # --- (d) metering, on the SAME held-out instances at every cycle ---------------
        mz = encode(shared, M["x"], device) if spec["rekey"] else None
        mr_t = torch.from_numpy(M["r"]).to(device) if spec["rekey"] else None
        mcls, mmig = route(lib, spec, M["f"], q, v, router, rekey_on, cfg["rekey_conf"],
                           mz, mr_t)
        if spec["rekey"]:
            counts["ground"] += M["x"].shape[0]
        mw = (M["f"] + q) % v
        b = beam_keyed(shared, generator, value, M["x"], M["r"], mcls, action,
                       mw if spec["wall_value"] else None, cfg, device)
        counts["mat"] += b["counts"]["mat"]; counts["ground"] += b["counts"]["ground"]
        msucc, mdres = grade(b["x"].numpy(), M["r"], rules, s)
        e = 1.0 - float(msucc.mean())
        cell_e = {str(c): float(1.0 - msucc[mcls == c].mean()) for c in sorted(set(mcls.tolist()))
                  if (mcls == c).sum()}
        # THE FALLBACK POPULATION: the instances the router is not confident enough to take,
        # which are exactly the ones a retired scaffold cell would otherwise have served.
        fb = {"frac": float(1.0 - mmig.mean()) if mmig.size else 1.0}
        if mmig.any():
            fb["e_mig"] = float(1.0 - msucc[mmig].mean())
        if (~mmig).any():
            fb["e_fallback"] = float(1.0 - msucc[~mmig].mean())

        # --- (e) commitment, per class ------------------------------------------------
        for c in lib.class_ids():
            cell = lib.cells[c]
            if (cell["table"] is not None or cell["n_obs"] < cfg["cell_min_obs"]
                    or not lib.maintained(c)):
                continue
            live = lib.live(c)
            if live["child"].shape[0] == 0:
                continue
            lib.commit(c, live, cyc)
            events.append({"kind": "commit", "arm": arm, "cycle": cyc, "cls": int(c),
                           "q": int(q), "n_obs": int(cell["n_obs"]),
                           "n_entries": int(live["child"].shape[0]),
                           "members": lib.members(c),
                           "feats": table_features(live, inv_key, v),
                           **{f"tab_{k}": val for k, val in
                              MC.grade_table(live, truth2).items()}})
            print(f"[commit] arm={arm} c{cyc} cls={c} obs={cell['n_obs']} "
                  f"entries={live['child'].shape[0]} "
                  f"feats={table_features(live, inv_key, v)}", flush=True)

        # --- (f) recert, per class: the demand-news organ, per cell -------------------
        if cfg["recert_every"] and cyc % cfg["recert_every"] == 0:
            rc = {"ground": 0, "mat": 0}
            for c in lib.class_ids():
                if lib.table(c) is None or not lib.maintained(c):
                    continue
                xr, rr_, fr_ = lib.ring(c, cfg["n_recert"])
                if xr is None or xr.shape[0] < 8:
                    continue
                live = lib.live(c)
                if live["child"].shape[0] == 0:
                    continue
                wr = (fr_ + q) % v if spec["wall_value"] else None
                ef, cf = policy_err(shared, generator, value, xr, rr_, lib.table(c), base_ms,
                                    cfg, device, wr)
                el, cl_ = policy_err(shared, generator, value, xr, rr_, live, base_ms,
                                     cfg, device, wr)
                for k in rc:
                    rc[k] += cf[k] + cl_[k]
                rec = {"kind": "recert", "arm": arm, "cycle": cyc, "cls": int(c), "q": int(q),
                       "e_frozen": ef, "e_live": el, "swapped": False,
                       "n_frozen": int(lib.table(c)["child"].shape[0]),
                       "n_live": int(live["child"].shape[0])}
                if el <= ef - cfg["recert_margin"]:
                    lib.swap(c, live)
                    rec["swapped"] = True
                    print(f"[recert] arm={arm} c{cyc} cls={c} SWAP {rec['n_frozen']}->"
                          f"{rec['n_live']} e {ef:.3f}->{el:.3f}", flush=True)
                events.append(rec)
            counts["ground"] += rc["ground"]; counts["mat"] += rc["mat"]
            gcost["recert"] = rc
            action = build_action(lib, base_ms, s, depth, device)

        # --- (g) MERGE: the op that acts on the index ---------------------------------
        if spec["merge"] and cfg["merge_every"] and cyc % cfg["merge_every"] == 0:
            mg = {"ground": 0, "mat": 0}
            X, ids, graded = WL.transfer_matrix(
                lib, rules, canon_np, n_screen=cfg["n_screen"],
                min_n=cfg["screen_min_n"], min_mass=cfg["screen_min_mass"])
            mg["ground"] += graded
            cands = WL.merge_candidates(X, ids, cfg["merge_tol"])
            # HYSTERESIS. Merge is irreversible and coarsening a right index cannot be undone,
            # so a pair has to survive `merge_persist` consecutive checks before it fires.
            # This is what keeps a bootstrap accident -- two cells that briefly hold the same
            # junk before either has earned its content -- from collapsing the index for good.
            live_pairs = {(cd["a"], cd["b"]) for cd in cands}
            for pair in list(persist):
                if pair not in live_pairs:
                    del persist[pair]
            done = 0
            for cd in cands:
                if done >= cfg["merge_max_per_check"]:
                    break
                a, bb = cd["a"], cd["b"]
                if a not in lib.cells or bb not in lib.cells:
                    continue
                persist[(a, bb)] = persist.get((a, bb), 0) + 1
                if persist[(a, bb)] < cfg["merge_persist"]:
                    continue
                ev = {"kind": "merge_candidate", "arm": arm, "cycle": cyc, "q": int(q),
                      "persist": int(persist[(a, bb)]), **cd}
                if cfg["merge_audit"]:
                    ok_m, aud, cost = merge_audition(shared, generator, value, lib, a, bb,
                                                     base_ms, cfg, device, spec, q)
                    for k in mg:
                        mg[k] += cost[k]
                    ev.update(aud)
                    if not ok_m:
                        ev["kind"] = "merge_refused"
                        events.append(ev)
                        continue
                ev["kind"] = "merge"
                ev["members_before"] = [lib.members(a), lib.members(bb)]
                into = lib.merge(a, bb)
                ev["into"] = int(into)
                ev["members_after"] = lib.members(into)
                ev["n_entries_after"] = int(lib.table(into)["child"].shape[0]) \
                    if lib.table(into) is not None else 0
                events.append(ev)
                persist = {p: k for p, k in persist.items() if a not in p and bb not in p}
                done += 1
                print(f"[merge] arm={arm} c{cyc} {a}+{bb}->{into} loss={cd['loss']:+.3f} "
                      f"members={ev['members_after']} classes={len(lib.class_ids())}",
                      flush=True)
            counts["ground"] += mg["ground"]; counts["mat"] += mg["mat"]
            gcost["merge"] = mg
            if done:
                action = build_action(lib, base_ms, s, depth, device)

        # --- (g2) RE-KEY: the op that EARNS the index's basis --------------------------
        #  The trigger is AGENT-INTERNAL. Reading the rotation boundary directly would be
        #  cleaner code and an oracle: `fw_s0`/`fw_s1` keep the rotation unannounced (the arc's
        #  invariant since `setlist`), so nothing may read `rotated`. What the arm may read is
        #  its own competence, and a collapse in that is what a rotation feels like from
        #  inside.
        rk_row = None
        e_base = float(np.median(log["e"][-cfg["collapse_win"] - 1:-1])) \
            if len(log["e"]) > cfg["collapse_win"] else None
        collapsed = bool(e_base is not None and (e - e_base) > cfg["collapse_delta"])
        if spec["rekey"] and cfg["rekey_every"] and cyc % cfg["rekey_every"] == 0:
            rk_row = rekey_basis(shared, lib, router, bank, cfg, device, rng, active=rekey_on)
            counts["ground"] += rk_row["graded"]
            gcost["rekey"] = {"ground": rk_row["graded"], "mat": 0}
            action = build_action(lib, base_ms, s, depth, device)
            if rk_row["ho"] is not None and lib.earned_ids():
                g, gc = rekey_gate(shared, generator, value, lib, router, spec, bank, cfg,
                                   device, q, action, rk_row["ho"])
                rk_row.update(g)
                counts["ground"] += gc["ground"]; counts["mat"] += gc["mat"]
                gcost["rekey"]["ground"] += gc["ground"]
                gcost["rekey"]["mat"] = gc["mat"]
                # TWO ways in, both agent-internal and both graded in consumption.
                #  (A) PROVISIONAL, on a collapse of the arm's own competence: adopt at once,
                #      unproven, and let the revert net carry the risk -- `ear`'s lesson, where
                #      provisional commitment graded in consumption beat every certificate arm
                #      and the net fired 0 times in 24.
                #  (B) CERTIFIED: the corrected gate says the earned basis is simply better.
                path = None
                if not rekey_on and collapsed:
                    path = "provisional"
                elif not rekey_on and g["e_earned"] <= g["e_inc"] - cfg["rekey_margin"]:
                    path = "certified"
                if path:
                    rekey_on, rekey_since = True, cyc
                    for c in lib.earned_ids():
                        lib.set_active(c, True)
                    revert_streak = {}
                    action = build_action(lib, base_ms, s, depth, device)
                    events.append({"kind": "rekey_on", "arm": arm, "cycle": cyc, "q": int(q),
                                   "path": path, "t_cum": t_cum + priced(counts, cfg),
                                   **{k: x for k, x in rk_row.items() if k != "ho"}})
                    print(f"[rekey] arm={arm} c{cyc} BASIS ADOPTED [{path}]  "
                          f"cover={rk_row['n_cover']} acc={rk_row['router_acc']:.3f} "
                          f"e_inc {g['e_inc']:.3f} vs e_earned {g['e_earned']:.3f} "
                          f"ARI={rk_row['ari_earned']:.3f}", flush=True)
            rk_row.pop("ho", None)
            rk_row.update({"cycle": cyc, "q": int(q), "collapsed": collapsed,
                           "enabled": bool(rekey_on)})
            events.append({"kind": "rekey", "arm": arm, **rk_row})

        # --- (g3) THE REVERT NET: provisional adoption made safe ----------------------
        if (spec["rekey"] and rekey_on and cfg["revert_every"]
                and cyc % cfg["revert_every"] == 0 and lib.active_earned_ids()):
            rv = revert_net(shared, generator, value, lib, spec, cfg, device, q, action,
                            arm, cyc, events, revert_streak)
            counts["ground"] += rv["ground"]; counts["mat"] += rv["mat"]
            gcost["revert"] = rv
            action = build_action(lib, base_ms, s, depth, device)

        # --- (g4) RETIRE: what to do with the scaffold once it has been migrated off ----
        #  Evidence-driven, like every other gate here: a wall cell is retired only after the
        #  index has migrated and that cell's own ROUTING SHARE has stayed negligible across a
        #  window. Nothing is scheduled, and nothing reads the rotation.
        for c in lib.wall_ids():
            share_hist.setdefault(c, []).append(float((mcls == c).mean()))
            share_hist[c] = share_hist[c][-cfg["retire_win"]:]
        if (spec["retire"] and rekey_on and cfg["retire_every"]
                and cyc % cfg["retire_every"] == 0):
            for c in list(lib.wall_ids()):
                h = share_hist.get(c, [])
                if (not lib.maintained(c) or len(h) < cfg["retire_win"]
                        or max(h) >= cfg["retire_share"]):
                    continue
                ev = {"kind": "retire", "arm": arm, "cycle": cyc, "q": int(q), "cls": int(c),
                      "mode": spec["retire"], "max_share": float(max(h)),
                      "mean_share": float(np.mean(h)),
                      "n_entries": (0 if lib.table(c) is None
                                    else int(lib.table(c)["child"].shape[0])),
                      "members": lib.members(c), "t_cum": t_cum + priced(counts, cfg)}
                lib.retire(c, spec["retire"])
                events.append(ev)
                print(f"[retire] arm={arm} c{cyc} cls={c} {spec['retire'].upper()} "
                      f"share<= {ev['max_share']:.3f} entries={ev['n_entries']}", flush=True)
            action = build_action(lib, base_ms, s, depth, device)

        # --- book the cycle -----------------------------------------------------------
        t_cum += priced(counts, cfg)
        take_probe = (cyc == 1 or cyc % cfg["probe_every"] == 0 or cyc == cfg["max_cycles"]
                      or rotated or WL.is_rotation_cycle(cyc + 1, cfg["phase1"],
                                                         cfg["rot_period"]))
        if take_probe:
            probe = {"cycle": cyc, "q": int(q), "t_cum": t_cum}
            # the wall-permutation probe: permute the latent BEFORE key and wall are read, so
            # every arm's own index (and free_selector's own wall input) is misdirected
            fp = (M["f"] + cfg["probe_shift"]) % v
            pcls_p, _mg = route(lib, spec, fp, q, v, router, rekey_on, cfg["rekey_conf"],
                                mz, mr_t)
            wp = (fp + q) % v
            bp = beam_keyed(shared, generator, value, M["x"], M["r"], pcls_p,
                            action, wp if spec["wall_value"] else None, cfg, device)
            sc, _ = grade(bp["x"].numpy(), M["r"], rules, s)
            probe["e_wall_permuted"] = 1.0 - float(sc.mean())
            probe["e"] = e
            # the index's alignment with the true partition -- ORACLE READOUT, never consumed
            probe["ari_eff"] = RK.adjusted_rand(mcls, M["f"])
            probe["ari_wall"] = RK.adjusted_rand(
                lib.classes_of(arm_keys(spec, M["f"], q, v)), M["f"])
            if spec["rekey"] and router is not None and router.head is not None:
                probe["ari_router"] = RK.adjusted_rand(
                    router.predict(mz, mr_t)[0].cpu().numpy(), M["f"])
            dz = encode(shared, D["x"], device) if spec["rekey"] else None
            dcls, _dmg = route(lib, spec, D["f"], q, v, router, rekey_on, cfg["rekey_conf"],
                               dz, torch.from_numpy(D["r"]).to(device) if spec["rekey"] else None)
            dw = (D["f"] + q) % v
            bd = beam_keyed(shared, generator, value, D["x"], D["r"], dcls,
                            action, dw if spec["wall_value"] else None, cfg, device)
            sc, _ = grade(bd["x"].numpy(), D["r"], rules, s)
            probe["e_deep"] = 1.0 - float(sc.mean())
            probe["plant"] = plant_probe(generator, {**shared,
                                                     "probe_clean": world["probe_clean"]},
                                         cfg, device)
            X, ids, _g = WL.transfer_matrix(lib, rules, canon_np, n_screen=cfg["n_screen"])
            probe["xfer"] = {"ids": [int(i) for i in ids],
                             "X": (X.tolist() if X.size else [])}
            log["probe"].append(probe)

        st = lib.state()
        cover = {}
        for c in lib.class_ids():
            t = lib.table(c)
            if t is None:
                continue
            xr, rr_, _fr = lib.ring(c, cfg["n_screen"])
            if xr is None:
                continue
            prof, _g = WL.entry_profile(rules, xr, rr_,
                                        [tuple(int(z) for z in r) for r in t["flat"]],
                                        kn, kl, s, canon_np)
            cover[str(c)] = float(prof.any(axis=0).mean()) if prof.shape[0] else 0.0
        log["cycle"].append(cyc); log["q"].append(int(q)); log["phase"].append(phase)
        log["t_cum"].append(t_cum); log["e"].append(e); log["e_practice"].append(e_practice)
        log["succ"].append(float(msucc.mean())); log["dres"].append(float(mdres.mean()))
        log["vloss"].append(vloss); log["gloss"].append(gloss)
        log["n_solved"].append(int(solved.shape[0])); log["n_mined"].append(n_mined)
        log["lib"].append(st); log["gcost"].append(gcost)
        log["cover"].append(cover); log["cell_e"].append(cell_e)
        log["mig"].append(float(mmig.mean()))
        log["rekey"].append(rk_row)
        log["fallback"].append(fb)
        # spend composition: the named maintenance organs, and everything else (practice +
        # metering). Where the freed budget goes is a readout of this round, not bookkeeping.
        named = {k: g for k, g in gcost.items()}
        tot_c = priced(counts, cfg)
        log["spend"].append({
            **{k: priced({"ground": x.get("ground", 0), "mat": x.get("mat", 0)}, cfg)
               for k, x in named.items()},
            "total": tot_c})
        print(f"[c{cyc:3d}] arm={arm:14s} p{phase} q{q} t={t_cum:10.0f} e={e:.4f} "
              f"ep={e_practice:.3f} cls={st['n_classes']}+{st['n_earned']}e "
              f"cm={st['n_committed']} "
              f"ent={st['n_entries_total']}/{st['n_entries_distinct']} "
              f"mig={mmig.mean():.2f} mined={n_mined}"
              + ("  ROTATED" if rotated else ""), flush=True)
        if cyc % cfg["checkpoint_every"] == 0:
            _checkpoint(outdir, arm, cfg, era, refs, log, events)

    write_results(outdir, arm, cfg, [era], refs, log, events, complete=True)
    return {"log": log, "events": events}


def _checkpoint(outdir, arm, cfg, era, refs, log, events):
    """Mid-run state goes to `checkpoint.json`, NOT `results.json`.

    `fw_s0` wrote both under the same name, so an external monitor watching for
    `results.json` saw an arm as finished at its first checkpoint. `results.json` now appears
    exactly once per arm, at completion."""
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "checkpoint.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": [era], "refs": refs,
                   "log": log, "events": events, "complete": False}, fh, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# helpers used by the loop
# --------------------------------------------------------------------------- #

def table_features(table, inv_key, v):
    """Which level-`key_level` features a table's entries actually derive -- the exact index
    correctness readout. (`build_inverse_maps` is last-writer-wins across features, so a
    handful of colliding tuples may be misattributed; that is a substrate read error, the same
    one `ratchet` records for its precision numbers.)"""
    if table is None or table["flat"].shape[0] == 0:
        return {}
    span = table["flat"].shape[1]
    codes = (table["flat"] * (v ** np.arange(span))).sum(-1)
    f = inv_key[codes]
    return {str(int(k)): int(n) for k, n in zip(*np.unique(f, return_counts=True))}


def policy_err(shared, generator, value, x_np, r_np, table, base_ms, cfg, device, walls=None):
    """The priced grader: the real policy on real instances with one candidate table in place."""
    import torch
    v, s, depth = cfg["v"], cfg["s"], cfg["depth"]
    ms = list(base_ms) + [MC.to_device(MC.make_macro(cfg["key_level"], cfg["key_node"], s,
                                                     table), device)]
    width = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
    if walls is not None:
        value.set_w(torch.from_numpy(np.asarray(walls, np.int64)).to(device))
    out = beam_moves(shared["controller"], generator, value, torch.from_numpy(x_np),
                     torch.from_numpy(r_np), ms, shared["rules_t"], shared["canon"], depth,
                     v, cfg["m"], s, budget=cfg["budget"], beam_width=width, device=device)
    if walls is not None:
        value.set_w(None)
    sc, _ = grade(out["x"].cpu().numpy(), r_np, shared["rules"], s)
    return 1.0 - float(sc.mean()), out["counts"]


def merge_audition(shared, generator, value, lib, a, b, base_ms, cfg, device, spec, q):
    """The priced second evidence route: does keeping the index distinction beat dropping it?

    `e_keep` runs each cell's own table on its own recent demand; `e_merge` runs the
    deduplicated union on both. The distinction is worth its slot only if keeping it wins by
    more than the margin."""
    xa, ra, fa = lib.ring(a, cfg["n_recert"])
    xb, rb, fb = lib.ring(b, cfg["n_recert"])
    cost = {"ground": 0, "mat": 0}
    if xa is None or xb is None or xa.shape[0] < 8 or xb.shape[0] < 8:
        return False, {"audit": "insufficient_ring"}, cost
    ta, tb = lib.table(a), lib.table(b)
    tu = WL.dedup_union([ta, tb], cfg["v"], lib.level, lib.s)
    wa = (fa + q) % cfg["v"] if spec["wall_value"] else None
    wb = (fb + q) % cfg["v"] if spec["wall_value"] else None
    res = {}
    for tag, (t, x, r, w) in {"aa": (ta, xa, ra, wa), "bb": (tb, xb, rb, wb),
                              "ua": (tu, xa, ra, wa), "ub": (tu, xb, rb, wb)}.items():
        err, cc = policy_err(shared, generator, value, x, r, t, base_ms, cfg, device, w)
        res[f"e_{tag}"] = err
        for k in cost:
            cost[k] += cc[k]
    na, nb = xa.shape[0], xb.shape[0]
    e_keep = (na * res["e_aa"] + nb * res["e_bb"]) / (na + nb)
    e_merge = (na * res["e_ua"] + nb * res["e_ub"]) / (na + nb)
    res.update({"e_keep": e_keep, "e_merge": e_merge,
                "n_union": int(tu["child"].shape[0])})
    return bool(e_merge <= e_keep + cfg["merge_audit_margin"]), res, cost


def rekey_basis(shared, lib, router, bank, cfg, device, rng, active):
    """Build the earned basis: what-serves-what, its closure, and a router onto it.

    Everything here is agent-internal. The service matrix is the grammar's DP answering the
    agent's own priced question ("does this program repair this instance"); the cover never
    mentions the latent; the router is trained on labels the cover produced. The `f` labels
    carried alongside are used ONLY for the ARI oracle readout, never in a decision."""
    import torch
    s = cfg["s"]
    kl, kn = cfg["key_level"], cfg["key_node"]
    rules, canon_np = shared["rules"], shared["canon_np"]
    out = {"n_programs": 0, "n_cover": 0, "router_acc": None, "ari_earned": None,
           "graded": 0, "ho": None}
    progs = RK.distinct_programs(lib)
    out["n_programs"] = len(progs)
    if len(progs) < 2 or bank["x"].shape[0] < cfg["rekey_min_bank"]:
        return out
    bx, br, bf = bank["x"], bank["r"], bank["f"]
    S, graded = RK.service_matrix(rules, bx, br, progs, kn, kl, s, canon_np)
    out["graded"] = graded

    n = bx.shape[0]
    perm = rng.permutation(n)
    tr, ho = perm[: n // 2], perm[n // 2:]
    chosen, labels = RK.greedy_cover(S[:, tr], min_gain=cfg["rekey_min_gain"])
    cover = [progs[c] for c in chosen]
    out["n_cover"] = len(cover)
    if not cover:
        return out
    lib.sync_earned(cover, active=active)
    order = {p: i for i, p in enumerate(lib.earned_programs())}
    keep = labels >= 0
    z_tr = encode(shared, bx[tr][keep], device)
    r_tr = torch.from_numpy(br[tr][keep]).to(device)
    y_tr = torch.from_numpy(
        np.array([order[cover[int(l)]] for l in labels[keep]], np.int64)).to(device)
    out["router_acc"] = router.fit(z_tr, r_tr, y_tr, len(lib.earned_ids()),
                                   steps=cfg["rekey_steps"])
    z_ho = encode(shared, bx[ho], device)
    pc, _c = router.predict(z_ho, torch.from_numpy(br[ho]).to(device))
    out["ari_earned"] = RK.adjusted_rand(pc.cpu().numpy(), bf[ho])
    out["ho"] = ho
    return out


def rekey_gate(shared, generator, value, lib, router, spec, bank, cfg, device, q, action, ho):
    """THE MIGRATION GATE, graded in the currency the arms are actually graded in.

    `fw_s1`'s gate asked "does *any* entry of the routed cell's table repair this instance" --
    a COVERAGE question. Stale wall cells answer it well by accumulating entries (6.00 stored
    per program served by c130) while producing a policy error of 0.338, so the gate preferred
    a bloated incumbent the policy did not. This is the etude's seam law in its third
    coordinate, applied to the index: an audition must match the consumption distribution, and
    the consumption here is held-out policy error under the current demand. So both routes are
    run through the real beam on the same held-out instances and compared on the error they
    actually produce."""
    import torch
    v = cfg["v"]
    n_gate = min(cfg["n_gate"], len(ho))
    sel = ho[:n_gate]
    x, r, f = bank["x"][sel], bank["r"][sel], bank["f"][sel]
    z = encode(shared, x, device)
    r_t = torch.from_numpy(r).to(device)
    counts = {"ground": 0, "mat": 0}
    out = {}
    for tag, use_earned in (("inc", False), ("earned", True)):
        cls, _m = route(lib, spec, f, q, v, router, use_earned, cfg["rekey_conf"], z, r_t,
                        ignore_active=True)
        b = beam_keyed(shared, generator, value, x, r, cls, action,
                       (f + q) % v if spec["wall_value"] else None, cfg, device)
        sc, _ = grade(b["x"].numpy(), r, shared["rules"], cfg["s"])
        out[f"e_{tag}"] = 1.0 - float(sc.mean())
        for k in counts:
            counts[k] += b["counts"][k]
    out["n_gate"] = int(n_gate)
    return out, counts


def revert_net(shared, generator, value, lib, spec, cfg, device, q, action, arm, cyc,
               events, streak):
    """`ear`'s recert net, pointed at the index instead of at an entry.

    Adoption is provisional and unproven by design, so the net is what makes that safe: each
    active earned class is graded IN CONSUMPTION on its own recent instances against what the
    incumbent index would have done with them, and a class that reads worse across
    `revert_persist` consecutive checks stops being routed to. A reverted cell keeps its
    content and keeps mining -- only the routing decision is withdrawn."""
    counts = {"ground": 0, "mat": 0}
    v = cfg["v"]
    for c in lib.active_earned_ids():
        xr, rr_, fr_ = lib.ring(c, cfg["n_revert"])
        if xr is None or xr.shape[0] < 8:
            continue
        wr = (fr_ + q) % v if spec["wall_value"] else None
        e_ear, c1 = policy_err(shared, generator, value, xr, rr_, lib.table(c),
                               shared["base_ms"], cfg, device, wr)
        inc = lib.classes_of(arm_keys(spec, fr_, q, v))
        b = beam_keyed(shared, generator, value, xr, rr_, inc, action, wr, cfg, device)
        sc, _ = grade(b["x"].numpy(), rr_, shared["rules"], cfg["s"])
        e_inc = 1.0 - float(sc.mean())
        for k in counts:
            counts[k] += c1[k] + b["counts"][k]
        worse = e_ear > e_inc + cfg["revert_margin"]
        streak[c] = streak.get(c, 0) + 1 if worse else 0
        ev = {"kind": "revert_check", "arm": arm, "cycle": cyc, "cls": int(c), "q": int(q),
              "e_earned": e_ear, "e_incumbent": e_inc, "streak": int(streak[c]),
              "reverted": False, "n": int(xr.shape[0])}
        if streak[c] >= cfg["revert_persist"]:
            lib.set_active(c, False)
            streak[c] = 0
            ev["reverted"] = True
            print(f"[revert] arm={arm} c{cyc} cls={c} REVERTED "
                  f"e {e_ear:.3f} vs incumbent {e_inc:.3f}", flush=True)
        events.append(ev)
    return counts


def rotation_profile(lib, rules, canon_np, M, spec, q_from, q_to, v, cfg):
    """The forced-transfer profile at a rotation, computed EXACTLY, per entry.

    For each committed class: the instances it was serving before the turn, the instances it is
    handed after it, and every committed program's success rate on both. No model, no grader --
    the grammar's own DP decides, which is why this instrument is available at all."""
    kl, kn, s = cfg["key_level"], cfg["key_node"], cfg["s"]
    out = []
    for c in lib.class_ids():
        t = lib.table(c)
        if t is None:
            continue
        entries = [tuple(int(z) for z in r) for r in t["flat"]]
        row = {"cls": int(c), "n_entries": len(entries), "members": lib.members(c)}
        for tag, qq in (("pre", q_from), ("post", q_to)):
            key = arm_keys(spec, M["f"], qq, v)
            sel = np.flatnonzero(lib.classes_of(key) == c)
            if sel.size == 0:
                row[f"n_{tag}"] = 0
                continue
            prof, _g = WL.entry_profile(rules, M["x"][sel], M["r"][sel], entries, kn, kl, s,
                                        canon_np)
            row[f"n_{tag}"] = int(sel.size)
            row[f"any_{tag}"] = float(prof.any(axis=0).mean()) if prof.shape[0] else 0.0
            row[f"per_entry_{tag}"] = [float(z) for z in prof.mean(axis=1)] \
                if prof.shape[0] else []
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = dict(
        v=8, s=2, depth=4, m=2, rule_seed=0, train_seed=1, seed=0, state_dim=96,
        n_train_episodes=100_000, controller_steps=12_000, generator_steps=12_000,
        reader_steps=6_000, plant_holdout=0, holdout_seed=11,
        value_steps=12_000, value_episodes=40_000, value_batch_collect=1024, batch_size=256,
        value_lr=3e-4, value_lr_online=3e-5, n_corrupt=3, explore_eps=0.3,
        # --- the substrate (the arc's depth-4 sculpting regime, single damage cell) -------
        era="2:3", deep_era="3:1", key_level=2, key_node=3, max_macro_level=2,
        budget=4, pr_width=16, g_budget=58, with_noop=False,
        n_pr=64, n_rt=384, n_deep=192, n_probe_clean=512,
        # --- the wall ---------------------------------------------------------------------
        phase1=50, rot_period=10, rot_step=1, probe_shift=3, max_cycles=130,
        # --- demand ------------------------------------------------------------------------
        # STATIC, and by default the DGP's own. Measured before the run (see FILES.md): the
        # marginal over the level-2 feature at the damaged node is ALREADY skewed by the
        # grammar -- H = 1.56 of a possible 2.08 nats at sigma = 0, one feature at 37% and one
        # at 0% -- so manufactured recurrence arrives with the substrate and needs no extra
        # lever. Adding `setlist`'s OU concentration on top starves cells rather than
        # sharpening the question (at sigma = 1.0 one feature takes 53% and three take 0%,
        # which would leave the index almost nothing to be wrong about). One lever: the wall.
        demand_sigma=0.0, demand_kappa=0.15, demand_levels="0/1", demand_seed=0,
        demand_n_cand=48,
        # --- the library ------------------------------------------------------------------
        mine_support=3, mine_cap=24, mine_decay=0.95, mine_floor=0.5, ring_cap=48,
        cell_min_obs=8, recert_every=5, recert_margin=0.05, n_recert=32,
        merge_every=5, merge_tol=0.05, merge_max_per_check=2, merge_audit=True,
        merge_audit_margin=0.05, n_screen=32, merge_persist=2,
        # the carried fix from `fw_s0`'s c15 vacuous merge: a class enters the alias screen
        # only once its ring is deep enough to estimate a rate and it carries enough demand to
        # be worth an index slot. `fw_s0` ran with both at 0.
        screen_min_n=32, screen_min_mass=0.02,
        # --- the re-key op ------------------------------------------------------------------
        rekey_every=5, rekey_bank=512, rekey_min_bank=128, rekey_min_gain=2,
        rekey_steps=400, rekey_conf=0.5, rekey_margin=0.02, n_gate=64,
        # provisional adoption on an agent-internal collapse of the arm's OWN competence,
        # plus `ear`'s net. `fw_s1` ran certify-then-adopt with persist=2 and vetoed its one
        # correct firing (c55, +0.137); there is no persistence on the adoption path here.
        collapse_win=5, collapse_delta=0.10,
        revert_every=5, revert_margin=0.05, revert_persist=2, n_revert=24,
        # the retire op: evidence-driven, post-migration, never scheduled
        retire_every=5, retire_win=10, retire_share=0.02,
        # --- learning / pricing -------------------------------------------------------------
        n_grad=4, value_batch=256, replay_frac=0.5, buf_cap=100_000, wall_lr_mult=20.0,
        gen_lr=1e-4, gen_steps=20, mine_from="chosen",
        d_fb=1.0, c_mat=0.05, checkpoint_every=5, probe_every=5, probe_widths=(1, 4, 16),
    )
    cfg.update(kw)
    return cfg


def _parse_arms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append((part, part, {}))
    return out


def _print_gate(refs, cfg):
    print("\n===== GATE 0: is the level-2 identity expensive to infer WITHOUT the wall? =====")
    print(f"  key entropy over the demand      : {refs['key_entropy']:.3f} "
          f"(max {np.log(cfg['v']):.3f} nats)")
    print(f"  mean # features that REPAIR       : {refs['mean_consistent']:.3f} of {cfg['v']}"
          f"   (a random feature repairs {refs['p_random_feature']:.3f})")
    print(f"  uniquely determined by the obs    : {refs['frac_uniquely_determined']:.3f}")
    print(f"  linear probe on the agent's state : {refs['probe']['acc']:.3f} "
          f"(majority class {refs['probe']['majority']:.3f})")
    print(f"  unkeyed true-table DP picks true f: {refs['id_acc_unkeyed_dp']:.3f}")
    print(f"  ONE ACTION   e(true unkeyed) {refs['e_act_true_unkeyed']:.4f}   "
          f"e(true KEYED) {refs['e_act_true_keyed']:.4f}")
    for sfx, name in (("", "no NOOP"), ("_noop", "with NOOP")):
        print(f"  FULL POLICY [{name:9s}]  e(base) {refs['e_pol_base' + sfx]:.4f}  "
              f"e(true unkeyed) {refs['e_pol_true_unkeyed' + sfx]:.4f}  "
              f"e(true KEYED) {refs['e_pol_true_keyed' + sfx]:.4f}  "
              f"e(true allnodes) {refs['e_pol_true_allnodes' + sfx]:.4f}"
              f"   [w{refs['width_true_unkeyed' + sfx]}, "
              f"{refs['g_true_unkeyed' + sfx]:.0f} g/solve]")
        gap = refs["e_pol_true_unkeyed" + sfx] - refs["e_pol_true_keyed" + sfx]
        print(f"      ==> what a RIGHT INDEX buys on the DGP's own tables: {gap:+.4f}")
    print(f"  floor(base exact DP) {refs['floor_base']:.4f}")
    print(f"  d0 {refs['d0']:.3f}  on_grammar {refs['on_grammar']:.3f}  "
          f"n {refs['n']}")
    print("=" * 78, flush=True)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def fourwall(
    tag: str = "smoke",
    arms: str = "given_key,wall_track,wall_merge,unkeyed,free_selector",
    seed: int = 0, rule_seed: int = 0, train_seed: int = 1,
    era: str = "2:3", deep_era: str = "3:1", key_level: int = 2, key_node: int = 3,
    phase1: int = 50, rot_period: int = 10, rot_step: int = 1, max_cycles: int = 130,
    probe_shift: int = 3,
    demand_sigma: float = 0.0, demand_kappa: float = 0.15, demand_seed: int = 0,
    demand_n_cand: int = 48,
    budget: int = 4, g_budget: int = 58, n_pr: int = 64, n_rt: int = 384, n_deep: int = 192,
    mine_support: int = 3, mine_cap: int = 24, mine_decay: float = 0.95,
    mine_floor: float = 0.5, ring_cap: int = 48, cell_min_obs: int = 8,
    recert_every: int = 5, recert_margin: float = 0.05, n_recert: int = 32,
    merge_every: int = 5, merge_tol: float = 0.05, merge_max_per_check: int = 2,
    merge_audit: bool = True, merge_audit_margin: float = 0.05, n_screen: int = 32,
    merge_persist: int = 2, with_noop: bool = False,
    screen_min_n: int = 32, screen_min_mass: float = 0.02,
    rekey_every: int = 5, rekey_bank: int = 512, rekey_min_bank: int = 128,
    rekey_min_gain: int = 2, rekey_steps: int = 400, rekey_conf: float = 0.5,
    rekey_margin: float = 0.02, n_gate: int = 64,
    collapse_win: int = 5, collapse_delta: float = 0.10,
    revert_every: int = 5, revert_margin: float = 0.05, revert_persist: int = 2,
    n_revert: int = 24, retire_every: int = 5, retire_win: int = 10,
    retire_share: float = 0.02,
    n_grad: int = 4, gen_lr: float = 1e-4, gen_steps: int = 20,
    value_lr_online: float = 3e-5, wall_lr_mult: float = 20.0, probe_every: int = 5,
    d_fb: float = 1.0, c_mat: float = 0.05, quick: bool = False,
):
    import torch
    cfg = _cfg(seed=seed, rule_seed=rule_seed, train_seed=train_seed, era=era,
               deep_era=deep_era, key_level=key_level, key_node=key_node, phase1=phase1,
               rot_period=rot_period, rot_step=rot_step, max_cycles=max_cycles,
               probe_shift=probe_shift, demand_sigma=demand_sigma, demand_kappa=demand_kappa,
               demand_seed=demand_seed, demand_n_cand=demand_n_cand, budget=budget,
               g_budget=g_budget, n_pr=n_pr, n_rt=n_rt, n_deep=n_deep,
               mine_support=mine_support, mine_cap=mine_cap, mine_decay=mine_decay,
               mine_floor=mine_floor, ring_cap=ring_cap, cell_min_obs=cell_min_obs,
               recert_every=recert_every, recert_margin=recert_margin, n_recert=n_recert,
               merge_every=merge_every, merge_tol=merge_tol,
               merge_max_per_check=merge_max_per_check, merge_audit=merge_audit,
               merge_audit_margin=merge_audit_margin, n_screen=n_screen,
               merge_persist=merge_persist, with_noop=with_noop,
               screen_min_n=screen_min_n, screen_min_mass=screen_min_mass,
               rekey_every=rekey_every, rekey_bank=rekey_bank,
               rekey_min_bank=rekey_min_bank, rekey_min_gain=rekey_min_gain,
               rekey_steps=rekey_steps, rekey_conf=rekey_conf,
               rekey_margin=rekey_margin, n_gate=n_gate, collapse_win=collapse_win,
               collapse_delta=collapse_delta, revert_every=revert_every,
               revert_margin=revert_margin, revert_persist=revert_persist,
               n_revert=n_revert, retire_every=retire_every, retire_win=retire_win,
               retire_share=retire_share, n_grad=n_grad,
               gen_lr=gen_lr, gen_steps=gen_steps, value_lr_online=value_lr_online,
               wall_lr_mult=wall_lr_mult, probe_every=probe_every, d_fb=d_fb, c_mat=c_mat)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   n_pr=24, n_rt=96, n_deep=48, n_probe_clean=128, checkpoint_every=2,
                   gen_steps=5, phase1=6, rot_period=3, max_cycles=14, cell_min_obs=3,
                   recert_every=3, merge_every=3, n_recert=16, n_screen=16, probe_every=3,
                   demand_n_cand=8, screen_min_n=16,
                   rekey_every=3, rekey_bank=192, rekey_min_bank=48, rekey_steps=150,
                   n_gate=32, n_revert=12, revert_every=3, collapse_win=3,
                   retire_every=3, retire_win=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"fourwall tag={tag} arms={arms} era={cfg['era']} key=L{cfg['key_level']}n"
          f"{cfg['key_node']} phase1={cfg['phase1']} period={cfg['rot_period']} "
          f"cycles={cfg['max_cycles']} device={device}", flush=True)

    shared = _shared_plus(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    world = build_world(shared, cfg, device)
    refs = wall_refs(shared, cfg, world, device)
    _print_gate(refs, cfg)
    if cfg["with_noop"]:
        shared["base_ms"] = build_move_set(cfg["depth"], cfg["s"], max_level=1, with_noop=True)
        print(f"[noop] base action space carries a NOOP: {len(shared['base_ms'])} moves",
              flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "refs": refs,
                   "world": {k: world[k] for k in ("key_mass", "key_entropy", "demand_stats")},
                   "schedule": [{"cycle": c,
                                 "q": WL.rotation_index(c, cfg["phase1"], cfg["rot_period"],
                                                        cfg["rot_step"])}
                                for c in range(1, cfg["max_cycles"] + 1)]},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in _parse_arms(arms):
        print(f"\n===== arm {label} =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, refs, outdir, device, world)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "elapsed": time.time() - started}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def gate0(tag: str = "g0", era: str = "2:3", key_level: int = 2, key_node: int = 3,
          demand_sigma: float = 0.0, n_rt: int = 1024, quick: bool = False):
    """GATE 0 ALONE, before anything else runs. If the parse is nearly free the wall buys
    nothing by construction and this design is the wrong instrument here."""
    import torch
    cfg = _cfg(era=era, key_level=key_level, key_node=key_node, demand_sigma=demand_sigma,
               n_rt=n_rt)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   n_rt=256, n_deep=64, n_probe_clean=128, demand_n_cand=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    started = time.time()
    shared = _shared_plus(cfg, device)
    world = build_world(shared, cfg, device)
    refs = wall_refs(shared, cfg, world, device)
    _print_gate(refs, cfg)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "gate0.json"), "w") as fh:
        json.dump({"config": cfg, "refs": refs, "key_mass": world["key_mass"]}, fh, indent=2,
                  cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return refs


# --------------------------------------------------------------------------- #
# structural gates (no GPU)
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=512, era="2:3", key_level=2, key_node=3):
    """W-1..W-6: the wall, the key, the library and the merge op, with no substrate."""
    from rhm.rhm_data import generate_rules_distinct
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    ctx = era_ctx(parse_eras(era)[0])
    out = {}

    # W-1: the key is the true latent -- an entry rendered from it must repair the instance.
    r, x, cl, f = WL.context_instances_wall(rules, ctx, n, s, depth, v, m, 0, None,
                                            key_level, key_node)
    ok = WL.entry_success(rules, x, r, WL.feature_leaves(rules, depth, key_level, 0, canon_np),
                          key_node, key_level, s)
    hits = np.zeros(n, bool)
    for fj in range(v):
        sel = np.flatnonzero(f == fj)
        if sel.size:
            hits[sel] = WL.entry_success(
                rules, x[sel], r[sel], WL.feature_leaves(rules, depth, key_level, fj, canon_np),
                key_node, key_level, s)
    out["W1_true_key_repairs"] = float(hits.mean())
    assert hits.mean() > 0.999, "W-1: the true latent must always repair its own instance"

    # W-2: the wall is a bijection of the latent, and rotation is a derangement.
    for q in (0, 1, 3, 7):
        mp = WL.wall_map(q, v)
        assert len(set(mp.tolist())) == v
        if q % v:
            assert not (mp == np.arange(v)).any(), "W-2: a rotation must move every wall"
    out["W2_bijection"] = True

    # W-3: the schedule -- phase 1 stationary, one rotation per period thereafter.
    qs = [WL.rotation_index(c, 6, 3, 1) for c in range(1, 16)]
    assert qs[:6] == [0] * 6 and qs[6] == 1 and qs[8] == 1 and qs[9] == 2, qs
    out["W3_schedule"] = qs

    # W-4: merge coarsens the index and DEDUPLICATES content, never averages it.
    lib = WL.Library(v, s, key_level, key_node, support=1)
    span = s ** (key_level - 1)
    for c in range(v):
        lib.observe(c, np.array([[c % v, (c + 1) % v]] * 3)[:, :span])
        lib.commit(c, lib.live(c), 1)
    n0 = lib.state()["n_entries_distinct"]
    lib.merge(0, 1)
    st = lib.state()
    assert st["n_classes"] == v - 1
    assert st["partition"]["1"] == st["partition"]["0"]
    assert st["n_entries_distinct"] == n0, "W-4: merge must not invent or destroy programs"
    ta = lib.table(st["partition"]["0"])
    assert ta["child"].shape[0] == 2, "W-4: the merged cell holds the union of two programs"
    out["W4_merge"] = st["n_classes"]

    # W-5: an unkeyed library routes everything to one class, whatever the key says.
    single = WL.Library(v, s, key_level, key_node, single=True)
    assert set(single.classes_of(np.arange(v)).tolist()) == {0}
    out["W5_single"] = True

    # W-7: a merged-away class leaves the wall id set, so every id the screen enumerates is
    # still addressable. (`smoke1` found this the hard way: `wall_ids` was the constructor's
    # set and merge never pruned it, so the second screen after a merge raised KeyError.)
    assert 1 not in lib.wall_ids() and 0 in lib.wall_ids()
    for c in lib.wall_ids():
        lib.ring(c)
    out["W7_wall_ids_after_merge"] = lib.wall_ids()

    # W-8: re-key earns cells by PROGRAM IDENTITY -- a surviving class keeps its cell, a
    # dropped one is retired, and neither touches the wall partition.
    lib.sync_earned([(0, 1), (2, 3)])
    first = lib.earned_ids()
    lib.sync_earned([(2, 3), (4, 5)])
    assert lib.earned_ids()[0] == first[1], "a surviving cover program must keep its cell"
    assert len(lib.earned_ids()) == 2 and lib.wall_ids() == out["W7_wall_ids_after_merge"]
    out["W8_earned"] = {"programs": [list(p) for p in lib.earned_programs()],
                        "ids": lib.earned_ids()}

    # W-9: the cover is the closure of "same unit serves them", and is EMPTY with no units.
    S = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], bool)
    ch, lab = RK.greedy_cover(S, min_gain=1)
    assert lab.tolist() == [0, 0, 1, 2] and len(ch) == 3
    assert RK.greedy_cover(np.zeros((0, 5), bool))[0] == []
    out["W9_cover"] = {"chosen": [int(c) for c in ch], "labels": lab.tolist()}
    assert abs(RK.adjusted_rand([0, 0, 1, 1], [5, 5, 9, 9]) - 1.0) < 1e-9
    out["W9_ari_identical"] = 1.0

    # W-6: consistent_features is exact -- the true feature is ALWAYS consistent.
    cons, _g = WL.consistent_features(rules, x[:n], r[:n], key_node, key_level, s, canon_np, v)
    assert cons[np.arange(n), f].all(), "W-6: the true feature must be consistent"
    out["W6_mean_consistent"] = float(cons.sum(1).mean())
    out["W6_true_always"] = float(cons[np.arange(n), f].mean())
    out["W1_feature0_repairs"] = float(ok.mean())
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    return selfcheck()


@app.local_entrypoint()
def main(quick: bool = True):
    fourwall.remote(tag="smoke0", quick=quick)
