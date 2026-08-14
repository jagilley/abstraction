"""Crystallize — certificate-gated compilation on the RHM sculpting substrate.

The practice arc's compile op (`mjc/practice/etude/`) ported off MuJoCo and onto a
production/control task with an exact DP oracle. The etude's substrate had a fatal
degeneracy for the NEXT question: a committed unit there is a state-independent open-loop
command sequence in a deterministic world, so post-commit drift was exactly 0.0000 and
fusion was provably vacuous — "hierarchy is meaningful only over boundaries that carry
information". Sculpting does not have that degeneracy: a committed unit here is a MOVE
program executed by a generator that reads the observed configuration, so the same unit
produces different tokens on different instances and commitment is state-conditioned by
construction.

WHAT IS BEING TESTED (round 1, the E-gate analog).
  Instantiate certificate-gated compilation of a small set of RECURRING repair contexts,
  under PRICED FEEDBACK, and grade it against never-compiling on success x priced time.

THE PIECE. A context is a fixed hierarchical-damage cell: a level-`k` subtree replaced by a
legal derivation of a feature the observed subtree provably cannot produce (100% on-grammar,
so a block-local reader cannot localise the error — `units.corrupt_hier`). The cell recurs;
everything else about the instance (the clean derivation, r*, the wrong feature, the
synonyms) varies. So a committed unit cannot be a blindly-replayed verbatim string.

THE TWO REGIMES.
  * PRACTICE   — closed-loop re-grounded beam at WIDE width. Every candidate is materialised,
                 re-encoded and value-scored (this is the Stage-3b idiom). Feedback is
                 available; it is priced but not budgeted.
  * PERFORMANCE — the same task under a fixed per-solve GROUNDING budget, so the beam that
                 fits is NARROW. A committed context instead pays ONE grounding at launch
                 (the observation that keys the library) plus one final verification, and
                 executes its move program open-loop.
  Compilation therefore transfers the coordination bought by wide-beam practice into a
  regime where width is unaffordable. This is the substrate's own measured law read
  backwards: "in a closed loop, search width substitutes for the forward model" — width is
  exactly what feedback pricing takes away.

WHAT LEARNS. The MC value, online, on the agent's own practice rollouts (value-iteration —
the substrate's measured +0.25 compounding engine). The value is PRETRAINED ON THE GENERIC
corrupt-repair distribution (random-symbol damage, random blocks) and is therefore STALE on
the hierarchical contexts, exactly as the etude's FM was pretrained with the hard passage
excluded. That staleness is what gives the silence detector a descent to detect.

delta IS CONSUMED AS A DETECTOR ONLY. No per-sample plasticity gain anywhere; the value's
online updates are plain uniform-lr AdamW.

Run from experiments/:
  modal run rhm/practice/crystallize/crystallize.py::selfcheck_remote
  modal run rhm/practice/crystallize/crystallize.py::crystallize --quick
  python3 rhm/practice/crystallize/launch_detached.py --tag cal_t0 --mode cal_task
"""

import copy
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _regenerate, _train_generator
from rhm.rhm_latent_planner import _build_value_head
from rhm.rhm_sculpt_planner import _corrupt, _encode_chunked, _sample_pool
from rhm.rhm_sculpt_latent import _build_rich_controller
from rhm.practice.crystallize.units import (
    apply_move, apply_sequence, build_move_set, corrupt_hier, grade, node_features,
    on_grammar_rate, oracle_rollout)


app = modal.App("rhm-practice-crystallize", image=image)


# --------------------------------------------------------------------------- #
# contexts
# --------------------------------------------------------------------------- #

def parse_contexts(spec):
    """"deep:2:2;shallow:1:0,3" -> [{name, level, nodes}, ...]"""
    out = []
    for part in spec.split(";"):
        if not part.strip():
            continue
        name, level, nodes = part.split(":")
        out.append({"name": name, "level": int(level),
                    "nodes": [int(x) for x in nodes.split(",")]})
    return out


def context_instances(rules, ctx, n, s, depth, v, m, seed, require_broken=True):
    """n fresh instances of a context: a clean derivation of a random r*, with the context's
    fixed cells hierarchically damaged.

    `require_broken` rejects instances whose d* is already 0. Ambiguity makes this necessary:
    swapping a node's feature for one it provably cannot derive still leaves r* in the ROOT's
    possible-set ~20% of the time at (L=4, m=2), and an instance that needs no repair is not a
    repair context."""
    length = s ** depth
    roots = np.zeros(n, np.int64)
    x = np.zeros((n, length), np.int64)
    rng = np.random.default_rng(seed + 90_001)
    need = np.arange(n)
    for attempt in range(24):
        if len(need) == 0:
            break
        r, lv = _sample_pool(rules, len(need), s, seed + 7919 * attempt)
        xd = corrupt_hier(lv, rules, depth, v, m, s, ctx["level"], ctx["nodes"], rng)
        ok = (nearest_derivation_cost(rules, xd, r, s) > 0) if require_broken \
            else np.ones(len(need), bool)
        roots[need[ok]] = r[ok]
        x[need[ok]] = xd[ok]
        need = need[~ok]
    if len(need):
        raise RuntimeError(f"context {ctx['name']}: {len(need)} instances never broke")
    return roots, x


# --------------------------------------------------------------------------- #
# the closed-loop beam over level moves (the PRACTICE idiom, and the narrow
# PERFORMANCE policy for an uncommitted context)
# --------------------------------------------------------------------------- #

def beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
               *, budget, beam_width, device, collect=False):
    """Stage-3b's re-grounded token beam, over the level-indexed action space, tracking the
    move sequence of every tip. `collect` also returns each surviving tip's full state
    trajectory (for MC value labels), reconstructed from parent pointers.

    Counts every materialisation and every GROUNDING (materialise + re-encode + value) so
    the feedback price can be applied post hoc."""
    import torch
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        x0 = x0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        n_moves = len(ms)
        beams = x0[:, None, :]
        seqs = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        counts = {"mat": 0, "ground": 0}
        hist_x = [beams.clone()] if collect else None
        hist_par = [] if collect else None
        for _ in range(budget):
            flat = beams.reshape(batch * width, length)
            children = [apply_move(generator, flat, ms[k], rules_t, canon, depth, v, m, s)
                        for k in range(n_moves)]
            cand = torch.stack(children, dim=1).reshape(batch, width * n_moves, length)
            counts["mat"] += batch * width * n_moves
            tgt = roots.repeat_interleave(width * n_moves)
            scores = value(_encode_chunked(controller, cand.reshape(-1, length)), tgt)
            scores = scores.reshape(batch, width * n_moves)
            counts["ground"] += batch * width * n_moves
            keep = min(beam_width, width * n_moves)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, n_moves, rounding_mode="floor")
            mv = top % n_moves
            beams = cand.gather(1, top[:, :, None].expand(-1, -1, length))
            prev = seqs.gather(1, parent[:, :, None].expand(-1, -1, seqs.shape[2])) \
                if seqs.shape[2] else seqs.new_zeros(batch, keep, 0)
            seqs = torch.cat([prev, mv[:, :, None]], dim=2)
            width = keep
            if collect:
                hist_par.append(parent)
                hist_x.append(beams.clone())
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams.reshape(-1, length)), tgt).reshape(batch, width)
        counts["ground"] += batch * width
        best = final.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        out = {"x": beams[rows, best], "seq": seqs[rows, best], "counts": counts,
               "tips_x": beams, "tips_seq": seqs}
        if collect:
            idx = torch.arange(width, device=device)[None, :].expand(batch, -1).contiguous()
            traj = [None] * (budget + 1)
            for t in range(budget, -1, -1):
                traj[t] = hist_x[t].gather(1, idx[:, :, None].expand(-1, -1, length))
                if t > 0:
                    idx = hist_par[t - 1].gather(1, idx)
            out["traj"] = traj                                   # (budget+1) x (B, width, T)
    return out


def run_unit(generator, x0, seq, ms, rules_t, canon, depth, v, m, s, device):
    """PERFORMANCE routing for a committed context: one grounding at launch (the observation
    that keys the library), the stored move program executed open-loop, one final grounding."""
    x0 = x0.to(device)
    x, n_mat = apply_sequence(generator, x0, seq, ms, rules_t, canon, depth, v, m, s)
    return {"x": x, "counts": {"mat": n_mat, "ground": 2 * x0.shape[0]}}


def run_library(generator, x0, unit, roots_np, ms, rules_t, canon, depth, v, m, s, device):
    """PERFORMANCE routing for a committed LIBRARY: the launch observation selects the entry
    (here, the target root — an observation the agent has for free), then that entry's program
    runs open-loop. Still ONE grounding at launch and one at the end, whatever the entry."""
    import torch
    x = x0.to(device).clone()
    n_mat = 0
    for root in range(v):
        sel = np.nonzero(roots_np == root)[0]
        if len(sel) == 0:
            continue
        idx = torch.from_numpy(sel).to(device)
        seq = unit["by_root"].get(str(root), unit["default"])
        xs, nm = apply_sequence(generator, x[idx], seq, ms, rules_t, canon, depth, v, m, s)
        x[idx] = xs
        n_mat += nm
    return {"x": x, "counts": {"mat": n_mat, "ground": 2 * x.shape[0]}}


def run_fixed_span(x0, span_tokens, pos, device):
    """PERFORMANCE routing for a state-INDEPENDENT committed unit: write a fixed token span
    into the damaged region, whatever the instance. The etude's `fixed` unit, verbatim."""
    import torch
    x = x0.to(device).clone()
    idx = pos[None, :].expand(x.shape[0], -1).to(device)
    x.scatter_(1, idx, span_tokens[None, :].expand(x.shape[0], -1).to(device))
    return {"x": x, "counts": {"mat": 0, "ground": 2 * x.shape[0]}}


# --------------------------------------------------------------------------- #
# online value training (plain uniform-lr; delta is never a gain)
# --------------------------------------------------------------------------- #

def value_steps(value, opt, controller, buf, replay, *, n_steps, batch, replay_frac, device, rng):
    import torch
    import torch.nn.functional as F
    if buf["x"].shape[0] == 0:
        return 0.0
    value.train()
    n_rep = int(round(batch * replay_frac)) if replay["x"].shape[0] else 0
    n_new = batch - n_rep
    last = 0.0
    for _ in range(n_steps):
        parts_x, parts_r, parts_y = [], [], []
        if n_new:
            i = torch.from_numpy(rng.integers(0, buf["x"].shape[0], size=n_new))
            parts_x.append(buf["x"][i]); parts_r.append(buf["r"][i]); parts_y.append(buf["y"][i])
        if n_rep:
            i = torch.from_numpy(rng.integers(0, replay["x"].shape[0], size=n_rep))
            parts_x.append(replay["x"][i]); parts_r.append(replay["r"][i]); parts_y.append(replay["y"][i])
        x = torch.cat(parts_x).to(device)
        r = torch.cat(parts_r).to(device)
        y = torch.cat(parts_y).to(device)
        with torch.no_grad():
            z = controller.state(x)
        loss = F.binary_cross_entropy_with_logits(value(z, r), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    value.eval()
    return last


def push(buf, x, r, y, cap):
    """FIFO append (CPU tensors)."""
    import torch
    buf["x"] = torch.cat([buf["x"], x])[-cap:]
    buf["r"] = torch.cat([buf["r"], r])[-cap:]
    buf["y"] = torch.cat([buf["y"], y])[-cap:]


# --------------------------------------------------------------------------- #
# setup: instruments trained ONCE, shared by every arm
# --------------------------------------------------------------------------- #

def build_shared(cfg, device):
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    length = s ** depth
    n_blocks = length // s

    rules = generate_rules_distinct(v, s, depth, m, seed=cfg["rule_seed"])
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    ms = build_move_set(depth, s, max_level=cfg["max_level"], with_noop=cfg["with_noop"])

    roots_np, leaves_np = _sample_pool(rules, cfg["n_train_episodes"], s, cfg["train_seed"])
    leaves = torch.from_numpy(leaves_np)
    roots = torch.from_numpy(roots_np)

    controller = _build_rich_controller()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, leaves, roots, batch_size=cfg["batch_size"],
                           n_blocks=n_blocks, block_size=s, n_steps=cfg["controller_steps"],
                           lr=3e-4, device=device, p_full=0.5)
    generator = _build_generator()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2,
                                   root_conditioned=False).to(device)
    _train_generator(generator, leaves, roots, bottom_map, batch_size=cfg["batch_size"],
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=cfg["generator_steps"], lr=3e-4, device=device)
    for mod in (controller, generator):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)

    # ---- the STALE value: trained on the GENERIC corrupt-repair distribution ----------
    # random-symbol damage at random blocks -- deliberately NOT the hierarchical contexts,
    # so the value arrives at cycle 0 stale on them and the detector has a descent to see.
    print(f"Collecting the generic (stale) value buffer: {cfg['value_episodes']} rollouts")
    vb = collect_value_buffer(controller, generator, value=None, rules=rules, rules_t=rules_t,
                              canon=canon, ms=ms, roots_pool=roots_np, leaves_pool=leaves_np,
                              cfg=cfg, device=device)
    value = _build_value_head()(cfg["state_dim"], v).to(device)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["train_seed"] + 31)
    empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
    print(f"  buffer {vb['x'].shape[0]} states, terminal success {float(vb['y'].mean()):.3f}")
    value_steps(value, opt, controller, vb, empty, n_steps=cfg["value_steps"],
                batch=512, replay_frac=0.0, device=device, rng=rng)

    return {"rules": rules, "rules_t": rules_t, "canon": canon, "inverse_maps": inverse_maps,
            "ms": ms, "controller": controller, "generator": generator, "value0": value,
            "replay": vb, "n_blocks": n_blocks, "length": length}


def collect_value_buffer(controller, generator, value, rules, rules_t, canon, ms,
                         roots_pool, leaves_pool, cfg, device):
    """Behaviour-policy rollouts on the GENERIC distribution, every visited state labelled by
    its rollout's terminal possible-set success. Scoring is by the controller's own root
    log-prob (no value needed), so this runs before the value exists."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    n_blocks = leaves_pool.shape[1] // s
    rng = np.random.default_rng(cfg["train_seed"] + 4321)
    xs, rs, ys = [], [], []
    done = 0
    while done < cfg["value_episodes"]:
        b = min(cfg["value_batch_collect"], cfg["value_episodes"] - done)
        done += b
        idx = rng.integers(0, leaves_pool.shape[0], size=b)
        roots_np = roots_pool[idx]
        c = int(rng.integers(1, cfg["n_corrupt"] + 1))
        start = _corrupt(leaves_pool[idx], n_blocks, c, v, s, rng)
        x = torch.from_numpy(start).to(device)
        roots = torch.from_numpy(roots_np).to(device)
        traj = [x.clone()]
        for _ in range(cfg["budget"]):
            x = behavior_step(controller, generator, x, roots, ms, rules_t, canon,
                              depth, v, m, s, cfg["explore_eps"], device, rng)
            traj.append(x.clone())
        succ, _ = grade(x.cpu().numpy(), roots_np, rules, s)
        y = torch.from_numpy(succ.astype(np.float32))
        for st in traj:
            xs.append(st.cpu()); rs.append(roots.cpu()); ys.append(y)
    return {"x": torch.cat(xs), "r": torch.cat(rs), "y": torch.cat(ys)}


def behavior_step(controller, generator, x, roots, ms, rules_t, canon, depth, v, m, s,
                  eps, device, rng):
    """One controller-greedy level move with epsilon exploration."""
    import torch
    with torch.no_grad():
        batch = x.shape[0]
        props, scores = [], []
        for mv in ms:
            p = apply_move(generator, x, mv, rules_t, canon, depth, v, m, s)
            lp = controller.root_logits(controller.state(p)).log_softmax(-1)
            scores.append(lp.gather(1, roots[:, None]).squeeze(1))
            props.append(p)
        scores = torch.stack(scores, dim=1)
        props = torch.stack(props, dim=1)
        chosen = scores.argmax(dim=1)
        explore = torch.from_numpy(rng.random(batch) < eps).to(device)
        rnd = torch.from_numpy(rng.integers(0, len(ms), size=batch)).to(device)
        chosen = torch.where(explore, rnd, chosen)
        return props.gather(1, chosen[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)


# --------------------------------------------------------------------------- #
# the compile op
# --------------------------------------------------------------------------- #

def audition(pool, cand, score_roots, score_x, shared, cfg):
    """Execute each drawn candidate OPEN-LOOP on the whole score set. Returns (nc, n_score)
    0/1 outcomes and the materialisation count. Candidates are drawn UNIFORMLY from the trace
    pool by the caller — uniform, not top-of-pool: ranking a candidate by its OWN realised
    outcome is a winner's curse, favouring the realisation most finely tuned to its own
    instance, i.e. the least transferable."""
    ms, rules, rules_t, canon = shared["ms"], shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    out, mats = [], 0
    for ci in cand:
        xf, nm = apply_sequence(shared["generator"], score_x, pool[ci]["seq"], ms, rules_t,
                                canon, depth, v, m, s)
        mats += nm
        succ, _ = grade(xf.cpu().numpy(), score_roots, rules, s)
        out.append(succ)
    return np.stack(out), mats


def select_units(pool, score_roots, score_x, shared, cfg, device, rng, key="root",
                 precomputed=None):
    """THE COMPILE OP: selection + commitment of realisations, scored by expected performance
    under the CONSUMPTION distribution.

    `key="global"` commits ONE program for the whole context — the etude's unit shape, and
    here the state-INDEPENDENT control. `key="root"` commits a LIBRARY: one program per target
    root, selected at launch by an observation the agent already has. That is E-5's explicit
    prescription ("a small library selected at launch by the observed hand-over") and the only
    form in which commitment is genuinely state-conditioned.

    Two audition numbers are reported and their difference is the winner's curse, measured:
    `chosen_score` is selected and scored on the SAME instances (in-sample), while
    `audition` selects on half the score set and reports on the other half. Metering later
    uses the same statistic (a mean over instances of 1 - success), so the optimism gap that
    the post-commit anchor reveals is like-for-like."""
    v = cfg["v"]
    n = len(pool)
    nc = min(cfg["n_cand"], n)
    if precomputed is not None:
        cand, succ, mats = precomputed
        nc = len(cand)
    else:
        cand = rng.permutation(n)[:nc]
        succ, mats = audition(pool, cand, score_roots, score_x, shared, cfg)   # (nc, n_score)
    n_score = succ.shape[1]
    half = np.zeros(n_score, bool); half[::2] = True

    def pick(mask):
        return int(succ[:, mask].mean(1).argmax()) if mask.any() else 0

    if key == "global":
        w = pick(np.ones(n_score, bool))
        unit = {"kind": "seq", "seq": [int(x) for x in pool[cand[w]]["seq"]]}
        in_sample = 1.0 - float(succ[w].mean())
        wa = pick(half)
        held = 1.0 - float(succ[wa][~half].mean())
        entries = {"global": unit["seq"]}
    else:
        by_root, hit_in, hit_out = {}, np.zeros(n_score), np.zeros(n_score)
        for root in range(v):
            sel = score_roots == root
            if not sel.any():
                continue
            w = pick(sel)
            by_root[str(root)] = [int(x) for x in pool[cand[w]]["seq"]]
            hit_in[sel] = succ[w][sel]
            wa = pick(sel & half)
            hit_out[sel] = succ[wa][sel]
        default = pick(np.ones(n_score, bool))
        unit = {"kind": "lib", "by_root": by_root,
                "default": [int(x) for x in pool[cand[default]]["seq"]]}
        in_sample = 1.0 - float(hit_in.mean())
        held = 1.0 - float(hit_out[~half].mean())
        entries = by_root

    per_cand = 1.0 - succ.mean(1)
    own_ok = np.array([pool[ci]["own"] > 0.5 for ci in cand])
    info = {"key": key, "n_pool": int(n), "n_cand": int(nc),
            "n_distinct": int(len({tuple(p["seq"]) for p in pool})),
            "entries": entries, "n_entries": len(entries),
            "chosen_score": float(in_sample), "audition": float(held),
            "score_med": float(np.median(per_cand)), "score_best_single": float(per_cand.min()),
            "score_own_ok": float(per_cand[own_ok].mean()) if own_ok.any() else None,
            "frac_own_ok": float(own_ok.mean())}
    return unit, info, {"mat": mats, "ground": nc * n_score}


def average_unit(pool):
    """The AVERAGING control: the position-wise modal move over the pool's VALID realisations.
    The etude measured that averaging valid command sequences destroys them (2.1x) — the mean
    of two renditions is not a rendition. Used as a discriminator probe."""
    valid = [p for p in pool if p["own"] > 0.5] or pool
    arr = np.array([p["seq"] for p in valid])
    seq = [int(np.bincount(arr[:, h]).argmax()) for h in range(arr.shape[1])]
    return seq, float(np.mean([tuple(a) == tuple(seq) for a in arr]))


# --------------------------------------------------------------------------- #
# metering
# --------------------------------------------------------------------------- #

def perform(unit, roots_np, x0, shared, cfg, device):
    """Run one context's held-out instances under PERFORMANCE conditions."""
    ms, rules, rules_t, canon = shared["ms"], shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    import torch
    roots = torch.from_numpy(roots_np)
    if unit is None:
        out = beam_moves(shared["controller"], shared["generator"], shared["value"], x0, roots,
                         ms, rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                         beam_width=cfg["perf_width"], device=device)
    elif unit["kind"] == "seq":
        out = run_unit(shared["generator"], x0, unit["seq"], ms, rules_t, canon,
                       depth, v, m, s, device)
    elif unit["kind"] == "lib":
        out = run_library(shared["generator"], x0, unit, roots_np, ms, rules_t, canon,
                          depth, v, m, s, device)
    elif unit["kind"] == "span":
        out = run_fixed_span(x0, unit["tokens"], unit["pos"], device)
    else:
        raise ValueError(unit["kind"])
    xf = out["x"].cpu().numpy()
    succ, dres = grade(xf, roots_np, rules, s)
    return {"succ": float(succ.mean()), "e": 1.0 - float(succ.mean()),
            "dres": float(dres.mean()), "counts": out["counts"], "x": xf}


def priced(counts, cfg):
    return counts["ground"] * cfg["d_fb"] + counts["mat"] * cfg["c_mat"]


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

ARMS = {
    # the compile TRIGGER and the unit's KEY are the only things that differ.
    "never":       {"trigger": None,     "key": "root",   "at": None},
    "delta_gate":  {"trigger": "delta",  "key": "root",   "at": None},   # library, certificate-gated
    "sched_early": {"trigger": "cycle",  "key": "root",   "at": "early"},
    "sched_late":  {"trigger": "cycle",  "key": "root",   "at": "late"},
    # trigger-MATCHED to sched_late, so single-vs-library is isolated from commit timing.
    "gate_single": {"trigger": "cycle",  "key": "global", "at": "late"},
}


def parse_arms(spec):
    """"never,delta_gate,never:n_grad=4" -> [(label, base, {overrides})]. The override form
    lets a calibration sweep a knob across arms that SHARE one setup."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        base, *rest = part.split(":")
        ov = {}
        for kv in rest:
            k, val = kv.split("=")
            ov[k] = int(val) if val.isdigit() else float(val)
        label = base + ("" if not ov else "_" + "_".join(f"{k}{v}" for k, v in ov.items()))
        out.append((label, base, ov))
    return out


def run_arm(label, base, overrides, shared0, cfg, contexts, refs, outdir, device):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    ms, rules, rules_t, canon = shared0["ms"], shared0["rules"], shared0["rules_t"], shared0["canon"]
    K = len(contexts)

    shared = dict(shared0)
    shared["value"] = copy.deepcopy(shared0["value0"]).to(device)
    shared["value"].eval()
    for p in shared["value"].parameters():
        p.requires_grad_(True)
    # the ONLINE learning rate is a separate knob from the setup one: cald_s0 measured that at
    # the setup lr the whole stale->asymptote descent completes before the first metering, which
    # leaves the silence detector nothing to detect (the etude's cal_s2 trap, in its extreme form).
    opt = torch.optim.AdamW(shared["value"].parameters(),
                            lr=cfg["value_lr_online"] or cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)

    buf = {"x": torch.zeros(0, shared0["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    trace = {k: [] for k in range(K)}                 # per-context FIFO of per-cycle pools
    units = {k: None for k in range(K)}
    bench = {k: None for k in range(K)}
    emin = {k: None for k in range(K)}
    ehist = {k: [] for k in range(K)}
    dhist = {k: [] for k in range(K)}
    sil_run = {k: 0 for k in range(K)}
    cert_b = {k: None for k in range(K)}
    cert_pending = {k: 0 for k in range(K)}
    cert_win = {k: [] for k in range(K)}
    t_cum = 0.0
    events, log = [], {"cycle": [], "t_cum": [], "e": [], "b": [], "delta": [], "scale": [],
                       "sil_run": [], "succ": [], "dres": [], "committed": [],
                       "ground_per_solve": [], "e_practice": [], "vloss": [], "probe": [], "shadow": []}

    # fixed held-out metering set per context (variance across cycles is policy change, not
    # resampling) + a disjoint scoring set the compile op auditions on.
    meter = {}
    for k, ctx in enumerate(contexts):
        r_np, x_np = context_instances(rules, ctx, cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * k)
        meter[k] = (r_np, torch.from_numpy(x_np))
    shadow = {}
    if cfg["shadow_compile"]:
        for k, ctx in enumerate(contexts):
            r_np, x_np = context_instances(rules, ctx, cfg["n_score"], s, depth, v, m,
                                           seed=cfg["seed"] + 6100 + 23 * k)
            shadow[k] = (r_np, torch.from_numpy(x_np).to(device))

    for cycle in range(1, cfg["n_cycles"] + 1):
        cyc_counts = {"mat": 0, "ground": 0}

        # --- (a) practice: wide closed-loop beam on fresh instances of every UNCOMMITTED
        #         context. Committing frees this work; the priced-time axis grades it.
        e_pr = [None] * K
        for k, ctx in enumerate(contexts):
            if units[k] is not None:
                continue
            r_np, x_np = context_instances(rules, ctx, cfg["n_pr"], s, depth, v, m,
                                           seed=cfg["seed"] + 100_000 + 1000 * cycle + k)
            out = beam_moves(shared["controller"], shared["generator"], shared["value"],
                             torch.from_numpy(x_np), torch.from_numpy(r_np), ms, rules_t, canon,
                             depth, v, m, s, budget=cfg["budget"], beam_width=cfg["pr_width"],
                             device=device, collect=True)
            for key in cyc_counts:
                cyc_counts[key] += out["counts"][key]
            tips = out["tips_x"].cpu().numpy()                       # (B, W, T)
            seqs = out["tips_seq"].cpu().numpy()                     # (B, W, H)
            B, W, T = tips.shape
            succ, _ = grade(tips.reshape(B * W, T), np.repeat(r_np, W), rules, s)
            succ = succ.reshape(B, W)
            # MC labels: every state on every surviving tip's own path, labelled by that
            # tip's terminal success.
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in out["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
            ys = torch.from_numpy(np.tile(succ.reshape(-1).astype(np.float32), len(out["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            trace[k].append([{"seq": seqs[i, j], "own": float(succ[i, j])}
                             for i in range(B) for j in range(W)])
            if len(trace[k]) > cfg["trace_window"]:
                trace[k].pop(0)
            ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
            e_pr[k] = 1.0 - float(ps.mean())                          # wide-beam reference

        # --- (b) plasticity: plain uniform-lr value iteration -------------------------
        vloss = value_steps(shared["value"], opt, shared["controller"], buf, shared["replay"],
                            n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                            replay_frac=cfg["replay_frac"], device=device, rng=rng)

        # --- (c) metering under PERFORMANCE conditions --------------------------------
        es, deltas, scales, succs, dress, gps = [], [], [], [], [], []
        for k in range(K):
            r_np, x0 = meter[k]
            res = perform(units[k], r_np, x0, shared, cfg, device)
            for key in cyc_counts:
                cyc_counts[key] += res["counts"][key]
            e = res["e"]
            if bench[k] is None:
                bench[k] = e; emin[k] = e
            emin[k] = min(emin[k], e)
            d = bench[k] - e
            if cert_pending[k]:
                cert_win[k].append(e); cert_pending[k] -= 1
                if cert_pending[k] == 0:
                    cert_b[k] = float(np.mean(cert_win[k]))
                    events.append(dict(kind="anchor", ctx=k, cycle=cycle, level=cert_b[k],
                                       window=[float(x) for x in cert_win[k]]))
                    print(f"[anchor] arm={arm} ctx={k} c{cycle} b={cert_b[k]:.4f}", flush=True)
            if cert_b[k] is not None:
                d = cert_b[k] - e
            scale = max(refs["stale"][k] - emin[k], (cert_b[k] or bench[k]), 1e-6)
            ehist[k].append(e); dhist[k].append(d)
            we = ehist[k][-cfg["sil_W"]:]; wd = dhist[k][-cfg["sil_W"]:]
            quiet = (len(we) >= cfg["sil_W"]
                     and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                     and float(np.std(we)) < cfg["sil_cv"] * scale)
            sil_run[k] = sil_run[k] + 1 if quiet else 0
            if cert_b[k] is None:
                bench[k] = bench[k] + cfg["alpha"] * (e - bench[k])
            es.append(e); deltas.append(d); scales.append(scale)
            succs.append(res["succ"]); dress.append(res["dres"])
            gps.append(res["counts"]["ground"] / max(1, x0.shape[0]))

        # --- (c2) the width ladder: an INSTRUMENT, not the routing, and NOT priced. It puts
        #          the whole feedback Pareto on the record at every probe cycle, so the
        #          headline's dependence on the declared grounding budget is visible rather
        #          than assumed.
        probe = None
        if cycle == 1 or cycle % cfg["probe_every"] == 0 or cycle == cfg["n_cycles"]:
            probe = {}
            for w in cfg["probe_widths"]:
                col = []
                for k in range(K):
                    r_np, x0 = meter[k]
                    b = beam_moves(shared["controller"], shared["generator"], shared["value"],
                                   x0, torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                                   budget=cfg["budget"], beam_width=w, device=device)
                    sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                    col.append(1.0 - float(sc.mean()))
                probe[str(w)] = col
            log["probe"].append({"cycle": cycle, "e": probe})

        # --- (c3) THE SHADOW COMPILE: what the compile op WOULD commit this cycle, and how
        #          well it would score, computed every cycle and never acted on. This is the
        #          quantity the certificate is supposed to gate -- if it is flat from cycle 1
        #          the certificate has nothing to certify, and that is a fact about the
        #          substrate we want before the main run, not after it. Unpriced: an instrument.
        if cfg["shadow_compile"]:
            sh = {}
            for k in range(K):
                pool = [p for cyc in trace[k] for p in cyc]
                if units[k] is not None or not pool:
                    continue
                sr_np, sx = shadow[k]
                srng = np.random.default_rng(cfg["seed"] + 8191 + 31 * cycle + k)
                cell = {"n_pool": len(pool),
                        "n_distinct": len({tuple(p["seq"]) for p in pool})}
                cand = srng.permutation(len(pool))[:min(cfg["n_cand"], len(pool))]
                pre = (*audition(pool, cand, sr_np, sx, shared, cfg),)
                pre = (cand, pre[0], pre[1])
                for key in ("root", "global"):
                    _, inf, _ = select_units(pool, sr_np, sx, shared, cfg, device, srng,
                                             key=key, precomputed=pre)
                    cell[key] = {"audition": inf["audition"], "in_sample": inf["chosen_score"],
                                 "score_med": inf["score_med"],
                                 "score_own_ok": inf["score_own_ok"]}
                sh[str(k)] = cell
            log["shadow"].append({"cycle": cycle, "cells": sh})
            print("   [shadow] " + " ".join(
                f"ctx{k}: lib={c['root']['audition']:.3f} single={c['global']['audition']:.3f} "
                f"med={c['root']['score_med']:.3f} pool={c['n_pool']}/{c['n_distinct']}"
                for k, c in sh.items()), flush=True)

        # --- (d) the gate --------------------------------------------------------------
        for k in range(K):
            if units[k] is not None:
                continue
            fired = False
            if spec["trigger"] == "delta":
                fired = sil_run[k] >= cfg["sil_hold"] and cycle >= cfg["sil_min_cycle"]
            elif spec["trigger"] == "cycle":
                fired = cycle >= (cfg["sched_early"] if spec["at"] == "early"
                                  else cfg["sched_late"])
            if not fired:
                continue
            pool = [p for cyc in trace[k] for p in cyc]
            if not pool:
                continue
            sr_np, sx_np = context_instances(rules, contexts[k], cfg["n_score"], s, depth, v, m,
                                             seed=cfg["seed"] + 700_000 + 1000 * cycle + k)
            sx = torch.from_numpy(sx_np).to(device)
            crng = np.random.default_rng(cfg["seed"] + 953 + 31 * cycle + k)
            unit, info, sc = select_units(pool, sr_np, sx, shared, cfg, device, crng,
                                          key=spec["key"])
            for key in cyc_counts:
                cyc_counts[key] += sc[key]
            units[k] = unit
            cert_b[k] = None; cert_win[k] = []
            cert_pending[k] = cfg["cert_cal"]
            ev = dict(kind="compile", ctx=k, ctx_name=contexts[k]["name"], cycle=cycle,
                      arm=arm, t_cum=t_cum + priced(cyc_counts, cfg),
                      e=es[k], b=float(bench[k]), sil_run=int(sil_run[k]),
                      unit=unit, **info)
            events.append(ev)
            print(f"[commit] arm={arm} ctx={contexts[k]['name']} c{cycle} e={es[k]:.4f} "
                  f"key={spec['key']} score={info['chosen_score']:.4f} "
                  f"audition={info['audition']:.4f} med={info['score_med']:.4f} "
                  f"own_ok={info['score_own_ok']} entries={info['entries']}", flush=True)
            if cfg["discriminate"] and spec["key"] == "root" and spec["trigger"]:
                events.append(discriminate(pool, sr_np, sx, contexts[k], k, cycle, shared,
                                           cfg, device, crng))

        t_cum += priced(cyc_counts, cfg)
        log["cycle"].append(cycle); log["t_cum"].append(t_cum)
        log["e"].append(es); log["b"].append([bench[k] for k in range(K)])
        log["delta"].append(deltas); log["scale"].append(scales)
        log["sil_run"].append([sil_run[k] for k in range(K)])
        log["succ"].append(succs); log["dres"].append(dress)
        log["committed"].append([units[k] is not None for k in range(K)])
        log["ground_per_solve"].append(gps); log["e_practice"].append(e_pr)
        log["vloss"].append(vloss)
        print(f"[c{cycle:3d}] arm={arm} t={t_cum:9.0f} e={np.round(es, 4).tolist()} "
              f"succ={np.round(succs, 3).tolist()} sil={[sil_run[k] for k in range(K)]} "
              f"cmp={[units[k] is not None for k in range(K)]}", flush=True)
        if cycle % cfg["checkpoint_every"] == 0 or cycle == cfg["n_cycles"]:
            write_results(outdir, arm, cfg, contexts, refs, log, events, complete=False)

    write_results(outdir, arm, cfg, contexts, refs, log, events, complete=True)
    return {"log": log, "events": events}


def discriminate(pool, roots_np, x0, ctx, k, cycle, shared, cfg, device, rng):
    """The compile-hit discriminator, at the certificate state (the etude's `disc_c15`
    idiom). Every unit family is evaluated on the SAME held-out performance instances.

    (a) one selected realisation (the compile op's own answer)
    (b) the position-wise modal move sequence  -- averaging, in move space
    (c) ONE realisation's produced token span, written verbatim -- state-INDEPENDENT
    (d) the position-wise modal token span    -- averaging, in token space; the RHM-exact
        form of "the mean of two synonym expansions of the same latent is off-grammar"
    (e) a uniformly random move sequence       -- the floor
    (f) the exact-DP oracle over the same moves -- the ceiling
    (g) the wide practice beam                  -- what the unit is replacing
    """
    import torch
    ms, rules, rules_t, canon = shared["ms"], shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen = shared["generator"]
    out = {"kind": "discriminate", "ctx": k, "ctx_name": ctx["name"], "cycle": cycle}

    def score(xf):
        succ, dres = grade(xf.cpu().numpy(), roots_np, rules, s)
        return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean())}

    valid = [p for p in pool if p["own"] > 0.5] or pool

    for name, key in (("a_selected_single", "global"), ("a_selected_library", "root")):
        sel, info, _ = select_units(pool, roots_np, x0, shared, cfg, device,
                                    np.random.default_rng(int(rng.integers(1 << 30))), key=key)
        r = perform(sel, roots_np, x0.cpu(), shared, cfg, device)
        out[name] = {"e": r["e"], "dres": r["dres"], "audition": info["audition"],
                     "in_sample": info["chosen_score"], "entries": info["entries"]}

    modal, share = average_unit(pool)
    xf, _ = apply_sequence(gen, x0, modal, ms, rules_t, canon, depth, v, m, s)
    out["b_modal_move"] = {**score(xf), "seq": modal, "modal_share": share}

    # the damaged region's token positions, and what the pool's realisations wrote there
    span = s ** (ctx["level"] - 1)
    blocks = np.concatenate([np.arange(j * span, (j + 1) * span) for j in ctx["nodes"]])
    pos = torch.from_numpy((blocks[:, None] * s + np.arange(s)[None, :]).reshape(-1)).to(device)
    # each realisation replayed on ITS OWN held-out instance, so the spans differ the way
    # the agent's own renditions differ across instances -- the thing averaging destroys.
    spans = []
    for i, p in enumerate(valid[:min(cfg["n_span"], x0.shape[0])]):
        xr, _ = apply_sequence(gen, x0[i:i + 1], p["seq"], ms, rules_t, canon, depth, v, m, s)
        spans.append(xr[0, pos].cpu().numpy())
    spans = np.array(spans)

    verb = torch.from_numpy(spans[0]).to(device)
    r = run_fixed_span(x0, verb, pos, device)
    out["c_verbatim_span"] = {**score(r["x"]), "tokens": spans[0].tolist()}

    avg = np.array([np.bincount(spans[:, i], minlength=v).argmax() for i in range(spans.shape[1])])
    r = run_fixed_span(x0, torch.from_numpy(avg).to(device), pos, device)
    inv = shared["inverse_maps"][-1]
    powers = v ** np.arange(s)
    def og(tok):
        codes = (tok.reshape(-1, s) * powers).sum(-1)
        return float((inv[codes] >= 0).mean())
    out["d_modal_span"] = {**score(r["x"]), "tokens": avg.tolist(),
                           "on_grammar": og(avg), "on_grammar_realisations": float(
                               np.mean([og(sp) for sp in spans]))}

    rnd = [int(x) for x in rng.integers(0, len(ms), size=cfg["budget"])]
    xf, _ = apply_sequence(gen, x0, rnd, ms, rules_t, canon, depth, v, m, s)
    out["e_random"] = {**score(xf), "seq": rnd}

    xo, _ = oracle_rollout(gen, x0, roots_np, ms, rules, rules_t, canon, depth, v, m, s,
                           cfg["budget"])
    out["f_dp_oracle"] = score(xo)

    wide = beam_moves(shared["controller"], gen, shared["value"], x0,
                      torch.from_numpy(roots_np), ms, rules_t, canon, depth, v, m, s,
                      budget=cfg["budget"], beam_width=cfg["pr_width"], device=device)
    out["g_practice_beam"] = score(wide["x"])
    narrow = beam_moves(shared["controller"], gen, shared["value"], x0,
                        torch.from_numpy(roots_np), ms, rules_t, canon, depth, v, m, s,
                        budget=cfg["budget"], beam_width=cfg["perf_width"], device=device)
    out["h_perf_beam"] = score(narrow["x"])
    for w in cfg["probe_widths"]:
        b = beam_moves(shared["controller"], gen, shared["value"], x0,
                       torch.from_numpy(roots_np), ms, rules_t, canon, depth, v, m, s,
                       budget=cfg["budget"], beam_width=w, device=device)
        out[f"i_beam_w{w}"] = {**score(b["x"]),
                               "ground_per_solve": b["counts"]["ground"] / x0.shape[0]}
    print(f"[disc] ctx={ctx['name']} c{cycle} " +
          " ".join(f"{a}={b['e']:.3f}" for a, b in out.items() if isinstance(b, dict)), flush=True)
    return out


def write_results(outdir, arm, cfg, contexts, refs, log, events, complete):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "contexts": contexts, "refs": refs,
                   "log": log, "events": events, "complete": complete},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# references measured at setup (agent-invisible where noted)
# --------------------------------------------------------------------------- #

def measure_refs(shared, cfg, contexts, device):
    """ref_stale  -- e per context under PERFORMANCE conditions with the setup (stale) value.
                     This is the detector's scale anchor, taken BEFORE any adaptation (the
                     etude's cal_s1 lesson: taking it from cycle 1 understates it ~2x).
       e_floor    -- e per context under the exact-DP greedy oracle over the same move set.
                     ORACLE READOUT: does the certificate fire at the floor?
       e_wide     -- e per context under the wide practice beam with the stale value.
       d0/on_gram -- the damage gate: mean d* and the on-grammar rate of the damaged config."""
    import torch
    rules, ms, rules_t, canon = shared["rules"], shared["ms"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    shared = dict(shared); shared["value"] = shared["value0"]
    out = {"stale": [], "floor": [], "wide": [], "d0": [], "d0_max": [], "on_grammar": [],
           "ground_beam": [], "ground_unit": []}
    for k, ctx in enumerate(contexts):
        r_np, x_np = context_instances(rules, ctx, cfg["n_ref"], s, depth, v, m,
                                       seed=cfg["seed"] + 4242 + 13 * k)
        x0 = torch.from_numpy(x_np)
        d0 = nearest_derivation_cost(rules, x_np, r_np, s)
        out["d0"].append(float(d0.mean()))
        out["d0_max"].append(float(d0.max()))
        out["on_grammar"].append(on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s))
        res = perform(None, r_np, x0, shared, cfg, device)
        out["stale"].append(res["e"])
        out["ground_beam"].append(res["counts"]["ground"] / cfg["n_ref"])
        out["ground_unit"].append(2.0)
        xo, _ = oracle_rollout(shared["generator"], x0.to(device), r_np, ms, rules, rules_t,
                               canon, depth, v, m, s, cfg["budget"])
        succ, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        out["floor"].append(1.0 - float(succ.mean()))
        wide = beam_moves(shared["controller"], shared["generator"], shared["value0"], x0,
                          torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                          budget=cfg["budget"], beam_width=cfg["pr_width"], device=device)
        succ, _ = grade(wide["x"].cpu().numpy(), r_np, rules, s)
        out["wide"].append(1.0 - float(succ.mean()))
    return out


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = dict(
        v=8, s=2, depth=4, m=2, rule_seed=0, train_seed=1, seed=0, state_dim=96,
        n_train_episodes=100_000, controller_steps=12_000, generator_steps=12_000,
        value_steps=12_000, value_episodes=40_000, value_batch_collect=1024, batch_size=256,
        value_lr=3e-4, value_lr_online=None, shadow_compile=False, n_corrupt=3, explore_eps=0.3, max_level=None, with_noop=False,
        budget=3, pr_width=16, perf_width=1,
        n_pr=64, n_rt=384, n_score=256, n_ref=512, n_cand=32, n_span=32,
        n_cycles=40, n_grad=8, value_batch=256, replay_frac=0.5, buf_cap=200_000,
        trace_window=6, alpha=0.2,
        sil_c=0.06, sil_cv=0.10, sil_W=5, sil_hold=2, sil_min_cycle=4, cert_cal=3,
        sched_early=1, sched_late=30, d_fb=1.0, c_mat=0.05, checkpoint_every=5, discriminate=True,
        probe_widths=(4, 16), probe_every=5,
    )
    cfg.update(kw)
    return cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def crystallize(
    tag: str = "smoke", arms: str = "never,delta_gate,gate_single,sched_early",
    contexts: str = "deep:2:1;shallow:1:6,7", mode: str = "main",
    seed: int = 0, n_cycles: int = 40, budget: int = 3, pr_width: int = 16,
    perf_width: int = 1, n_pr: int = 64, n_grad: int = 8,
    sil_c: float = 0.06, sil_cv: float = 0.10, sil_win: int = 5, sil_hold: int = 2,
    sil_min_cycle: int = 4, sched_early: int = 1, sched_late: int = 30, max_level: int = 0, with_noop: bool = False,
    d_fb: float = 1.0, c_mat: float = 0.05, n_rt: int = 384,
    value_lr_online: float = 0.0, shadow_compile: bool = False,
    probe_every: int = 5, n_score: int = 256, n_cand: int = 32, quick: bool = False,
):
    import torch
    cfg = _cfg(seed=seed, n_cycles=n_cycles, budget=budget, pr_width=pr_width,
               perf_width=perf_width, n_pr=n_pr, n_rt=n_rt, n_grad=n_grad, sil_c=sil_c,
               value_lr_online=(value_lr_online or None), shadow_compile=shadow_compile,
               probe_every=probe_every, n_score=n_score, n_cand=n_cand,
               sil_cv=sil_cv, sil_W=sil_win, sil_hold=sil_hold, sil_min_cycle=sil_min_cycle,
               sched_early=sched_early, sched_late=sched_late, d_fb=d_fb, c_mat=c_mat,
               max_level=(max_level or None), with_noop=with_noop)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_cycles=6, n_pr=24,
                   n_rt=64, n_score=64, n_ref=128, n_cand=8,
                   checkpoint_every=2, sil_min_cycle=2, sched_early=1, sched_late=4)
    ctxs = parse_contexts(contexts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"crystallize tag={tag} mode={mode} arms={arms} contexts={ctxs} device={device}")

    shared = build_shared(cfg, device)
    print(f"action space: {len(shared['ms'])} moves "
          f"({[mv['name'] for mv in shared['ms']]})")
    refs = measure_refs(shared, cfg, ctxs, device)
    print(f"[refs] {json.dumps({k: np.round(v, 4).tolist() for k, v in refs.items()})}")

    outdir = f"{DATA_DIR}/rhm_practice_crystallize/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "contexts": ctxs, "refs": refs,
                   "moves": [mv["name"] for mv in shared["ms"]]}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in parse_arms(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, ctxs, refs, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "elapsed": time.time() - started}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_unit(tag: str = "cal_u0", contexts: str = "deep:2:1;shallow:1:6,7",
             seed: int = 0, budget: int = 3, n_ref: int = 512, with_noop: bool = False,
             widths: str = "1,4,16,64", quick: bool = False):
    """CALIBRATION 1 — the compilation headroom, measured before any loop is built.

    The whole design rests on one quantity: can a COMMITTED (open-loop, feedback-free) unit
    reach the accuracy of a feedback-constrained closed-loop beam? This enumerates the space
    of fixed move programs EXHAUSTIVELY (all 1-move and 2-move programs, plus 3-move programs
    over the best pairs) and grades them on the same held-out instances as the beams at
    several widths and the exact-DP oracle. It also prices every policy in groundings.

    Two units per family are reported, and the difference is the point:
      * `best_in`   -- the best program chosen and scored on the SAME instances (a ceiling,
                       and an audition-side optimism reference)
      * `best_out`  -- chosen on half, scored on the OTHER half (what a compile op can get)
    and a `lib_r*` variant that picks the best program PER TARGET ROOT — the state-conditioned
    library the etude's E-5 prescribed, keyed by an observation available at launch.
    """
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_ref=n_ref, with_noop=with_noop)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_ref=128)
    ctxs = parse_contexts(contexts)
    ws = [int(x) for x in widths.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    shared["value"] = shared["value0"]
    ms, rules, rules_t, canon = shared["ms"], shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen = shared["generator"]
    n_moves = len(ms)
    print(f"action space: {n_moves} moves {[mv['name'] for mv in ms]}")

    out = {"config": cfg, "contexts": ctxs, "moves": [mv["name"] for mv in ms], "cells": {}}
    for k, ctx in enumerate(ctxs):
        r_np, x_np = context_instances(rules, ctx, cfg["n_ref"], s, depth, v, m,
                                       seed=cfg["seed"] + 4242 + 13 * k)
        x0 = torch.from_numpy(x_np).to(device)
        n = cfg["n_ref"]
        half = np.zeros(n, bool); half[::2] = True          # audition / realisation split
        cell = {"d0": float(nearest_derivation_cost(rules, x_np, r_np, s).mean())}

        def run(seq):
            xf, _ = apply_sequence(gen, x0, seq, ms, rules_t, canon, depth, v, m, s)
            succ, _ = grade(xf.cpu().numpy(), r_np, rules, s)
            return succ

        def summarise(progs, succ):
            """succ: (P, n) 0/1. Report in-sample best, held-out best, and the per-r* library
            (also chosen in-sample vs held-out)."""
            e = 1.0 - succ.mean(1)
            bi = int(e.argmin())
            ea = 1.0 - succ[:, half].mean(1)
            bo = int(ea.argmin())
            lib_in, lib_out = [], []
            for root in range(v):
                sel = r_np == root
                if not sel.any():
                    continue
                lib_in.append(succ[:, sel].mean(1).max() * sel.sum())
                j = int(succ[:, sel & half].mean(1).argmax()) if (sel & half).any() else 0
                lib_out.append(succ[j][sel & ~half].sum())
            return {"best_in": float(e[bi]), "best_in_prog": [ms[i]["name"] for i in progs[bi]],
                    "best_out": float(1.0 - succ[bo][~half].mean()),
                    "best_out_prog": [ms[i]["name"] for i in progs[bo]],
                    "median": float(np.median(e)), "worst": float(e.max()),
                    "lib_in": float(1.0 - sum(lib_in) / n),
                    "lib_out": float(1.0 - sum(lib_out) / max(1, (~half).sum()))}

        singles = [[a] for a in range(n_moves)]
        s1 = np.stack([run(p) for p in singles])
        cell["len1"] = summarise(singles, s1)
        cell["per_move"] = {ms[a]["name"]: float(1.0 - s1[a].mean()) for a in range(n_moves)}

        pairs = [[a, b] for a in range(n_moves) for b in range(n_moves)]
        s2 = np.stack([run(p) for p in pairs])
        cell["len2"] = summarise(pairs, s2)

        if budget >= 3:
            top = np.argsort(1.0 - s2.mean(1))[:12]
            triples = [pairs[int(t)] + [c] for t in top for c in range(n_moves)]
            s3 = np.stack([run(p) for p in triples])
            cell["len3"] = summarise(triples, s3)

        beams = {}
        for w in ws:
            b = beam_moves(shared["controller"], gen, shared["value0"], x0,
                           torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                           budget=budget, beam_width=w, device=device)
            succ, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
            beams[str(w)] = {"e": float(1.0 - succ.mean()),
                             "ground_per_solve": b["counts"]["ground"] / n}
        cell["beam"] = beams
        xo, _ = oracle_rollout(gen, x0, r_np, ms, rules, rules_t, canon, depth, v, m, s, budget)
        succ, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        cell["dp_oracle"] = float(1.0 - succ.mean())
        out["cells"][ctx["name"]] = cell
        print(f"\n=== ctx {ctx['name']} (level {ctx['level']}, nodes {ctx['nodes']}, "
              f"d0={cell['d0']:.2f}) ===")
        print(f"  beams: " + " | ".join(f"w{w}: e={beams[str(w)]['e']:.3f} "
                                        f"({beams[str(w)]['ground_per_solve']:.0f}g)" for w in ws))
        print(f"  dp oracle e={cell['dp_oracle']:.3f}   (unit costs 2g)")
        for ln in ("len1", "len2", "len3"):
            if ln in cell:
                c = cell[ln]
                print(f"  {ln}: best_in={c['best_in']:.3f} {c['best_in_prog']} | "
                      f"best_out={c['best_out']:.3f} {c['best_out_prog']} | "
                      f"median={c['median']:.3f} | lib_r* in={c['lib_in']:.3f} out={c['lib_out']:.3f}")

    outdir = f"{DATA_DIR}/rhm_practice_crystallize/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_unit.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_unit.json")
    return out


def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256):
    """C1 -- the level-1 move IS the published `_regenerate` (so level-1 readouts are
    comparable to Stage 3a/3b). C2 -- every deeper move is ONE legal abstract commitment.
    G-D -- hierarchical damage is 100% on-grammar and still moves d*, where the published
    random-symbol damage is not."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    n_blocks = length // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2, root_conditioned=False).to(device)
    gen.eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(_corrupt(leaves_np, n_blocks, 3, v, s, rng)).to(device)
    ms = build_move_set(depth, s)
    out = {"n_moves": len(ms), "moves": [mv["name"] for mv in ms]}

    # C1
    ok = True
    for j in range(n_blocks):
        mv = ms[j]
        a = apply_move(gen, x, mv, rules_t, canon, depth, v, m, s)
        blocks = torch.full((n, 1), j, device=device, dtype=torch.long)
        b = _regenerate(gen, x, blocks, canon, None, block_size=s, sample=False)
        ok &= bool(torch.equal(a, b))
    out["C1_level1_is_published_move"] = ok
    assert ok, "C1 failed: the level-1 move is not `_regenerate`"

    # C2: each parent's chosen rule really produces its children (checked on the DP's own
    # derivation, NOT by parsing tokens back -- `build_inverse_maps` is last-writer-wins).
    c2 = True
    for mv in ms:
        if mv["level"] < 2:
            continue
        feats, pos = node_features(gen, x, mv, rules_t, depth, v, m, s)
        tup = canon[feats].reshape(n, -1).cpu().numpy()
        powers = v ** np.arange(s)
        codes = (tup.reshape(n, -1, s) * powers).sum(-1)
        c2 &= bool((inv[-1][codes] >= 0).all())
    out["C2_deep_moves_on_grammar"] = c2
    assert c2, "C2 failed: a deep move rendered an off-grammar tuple"

    # G-D
    gd = {}
    for level in (1, 2, 3):
        nodes = [0] if level > 1 else [0, 3]
        dmg = corrupt_hier(leaves_np, rules, depth, v, m, s, level, nodes, np.random.default_rng(7))
        d = nearest_derivation_cost(rules, dmg, roots_np, s)
        gd[f"hier_L{level}"] = {"on_grammar": on_grammar_rate(dmg, inv[-1], v, s),
                                "dstar": float(d.mean()), "dstar_zero": float((d == 0).mean())}
    pub = _corrupt(leaves_np, n_blocks, 2, v, s, np.random.default_rng(7))
    d = nearest_derivation_cost(rules, pub, roots_np, s)
    gd["published_random_symbol"] = {"on_grammar": on_grammar_rate(pub, inv[-1], v, s),
                                     "dstar": float(d.mean()), "dstar_zero": float((d == 0).mean())}
    out["GD_damage"] = gd
    assert gd["hier_L2"]["on_grammar"] == 1.0, "G-D failed: hierarchical damage is off-grammar"
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    return selfcheck()


@app.local_entrypoint()
def main(quick: bool = True):
    crystallize.remote(quick=quick)


# --------------------------------------------------------------------------- #
# (A) precheck: does the committed-unit CEILING move when the PLANT learns?
# --------------------------------------------------------------------------- #

def headroom_cell(shared, cfg, r_np, x0, device):
    """Exhaustive enumeration of every 1- and 2-move fixed program on one instance set.
    Reports the global best (in-sample and held-out) and the per-root library. This is the
    committed-unit CEILING: it depends only on the GENERATOR and the damage, never on the
    value, which is exactly why it is the right thing to watch when the plant is allowed to
    learn."""
    ms, rules, rules_t, canon = shared["ms"], shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen = shared["generator"]
    n_moves, n = len(ms), x0.shape[0]
    half = np.zeros(n, bool); half[::2] = True

    def run(seq):
        xf, _ = apply_sequence(gen, x0, seq, ms, rules_t, canon, depth, v, m, s)
        return grade(xf.cpu().numpy(), r_np, rules, s)[0]

    out = {}
    for name, progs in (("len1", [[a] for a in range(n_moves)]),
                        ("len2", [[a, b] for a in range(n_moves) for b in range(n_moves)])):
        succ = np.stack([run(p) for p in progs])
        e = 1.0 - succ.mean(1)
        bi = int(e.argmin()); bo = int((1.0 - succ[:, half].mean(1)).argmin())
        lin, lout = [], []
        for root in range(v):
            sel = r_np == root
            if not sel.any():
                continue
            lin.append(succ[:, sel].mean(1).max() * sel.sum())
            j = int(succ[:, sel & half].mean(1).argmax()) if (sel & half).any() else 0
            lout.append(succ[j][sel & ~half].sum())
        out[name] = {"best_in": float(e[bi]), "best_in_prog": [ms[i]["name"] for i in progs[bi]],
                     "best_out": float(1.0 - succ[bo][~half].mean()),
                     "best_out_prog": [ms[i]["name"] for i in progs[bo]],
                     "median": float(np.median(e)),
                     "lib_in": float(1.0 - sum(lin) / n),
                     "lib_out": float(1.0 - sum(lout) / max(1, (~half).sum()))}
    return out


def finetune_generator(generator, opt, new_leaves, replay_leaves, bottom_map, *, v, s, n_blocks,
                       n_steps, batch, replay_frac, device, rng):
    """The plant learns, GROUNDED ON TERMINAL SUCCESS: only configurations the agent actually
    solved enter the fresh half of the batch (a solved config IS a valid r* derivation, so the
    generator's own masked-infilling target is well defined on it). Replay against the clean
    setup pool keeps it from drifting off the grammar -- the `replay_frac` idiom the etude's
    cal_s0 forced."""
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    last = 0.0
    n_rep = int(round(batch * replay_frac))
    n_new = batch - n_rep
    for _ in range(n_steps):
        parts = []
        if n_new and new_leaves.shape[0]:
            parts.append(new_leaves[torch.from_numpy(
                rng.integers(0, new_leaves.shape[0], size=n_new))])
        if n_rep or not parts:
            k = n_rep if parts else batch
            parts.append(replay_leaves[torch.from_numpy(
                rng.integers(0, replay_leaves.shape[0], size=k))])
        leaves = torch.cat(parts).to(device)
        b = leaves.shape[0]
        feats = bottom_map[(leaves.view(b, n_blocks, s) * powers).sum(-1)]
        keep = (feats >= 0).all(1)
        if keep.sum() < 8:
            continue
        leaves, feats = leaves[keep], feats[keep]
        b = leaves.shape[0]
        n_mask = int(torch.randint(1, n_blocks + 1, ()).item())
        order = torch.rand(b, n_blocks, device=device).argsort(dim=1)
        mb = order[:, :n_mask]
        pos = (mb[:, :, None] * s + torch.arange(s, device=device)).reshape(b, -1)
        obs = leaves.clone().scatter_(1, pos, torch.full_like(pos, -1))
        logits = generator.block_logits(obs)
        mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, mb, True)
        loss = F.cross_entropy(logits[mask], feats[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    return last


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_plant(tag: str = "calp_s0", contexts: str = "deep:2:1;shallow:1:6,7", seed: int = 0,
              budget: int = 3, n_ref: int = 512, n_cycles: int = 20, gen_lr: float = 1e-4,
              gen_steps: int = 20, n_pr: int = 64, pr_width: int = 16,
              replay_frac: float = 0.5, quick: bool = False):
    """(A) PRECHECK, minimal: cald_s1 showed the committable-unit ceiling is FLAT because the
    plant (the generator) is frozen and only the selector (the value) learns. This asks the one
    question that decides round 2: if the PLANT is allowed to learn from the agent's own
    successful repairs, does the ceiling move by more than the 0.003 metering noise floor?

    Ceiling is measured by exhaustive enumeration on a FIXED reference set, before and after.
    Nothing else changes: same instances, same action space, same damage."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_ref=n_ref, n_pr=n_pr, pr_width=pr_width,
               replay_frac=replay_frac)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_ref=128, n_pr=24)
        n_cycles, gen_steps = 3, 5
    ctxs = parse_contexts(contexts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    shared["value"] = shared["value0"]
    ms, rules, rules_t, canon = shared["ms"], shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen, n_blocks = shared["generator"], shared["n_blocks"]
    bottom_map = torch.from_numpy(shared["inverse_maps"][-1]).to(device)
    for p in gen.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(gen.parameters(), lr=gen_lr, weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 4242)

    ref = {}
    for k, ctx in enumerate(ctxs):
        r_np, x_np = context_instances(rules, ctx, cfg["n_ref"], s, depth, v, m,
                                       seed=cfg["seed"] + 4242 + 13 * k)
        ref[k] = (r_np, torch.from_numpy(x_np).to(device))

    def snapshot(tag_):
        out = {}
        for k, ctx in enumerate(ctxs):
            r_np, x0 = ref[k]
            cell = headroom_cell(shared, cfg, r_np, x0, device)
            b = beam_moves(shared["controller"], gen, shared["value0"], x0,
                           torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                           budget=cfg["budget"], beam_width=cfg["pr_width"], device=device)
            cell["beam_w16"] = float(1.0 - grade(b["x"].cpu().numpy(), r_np, rules, s)[0].mean())
            out[ctx["name"]] = cell
            print(f"  [{tag_}] {ctx['name']:8s} len1_out={cell['len1']['best_out']:.3f} "
                  f"len2_out={cell['len2']['best_out']:.3f} lib_out={cell['len2']['lib_out']:.3f} "
                  f"(in {cell['len2']['best_in']:.3f}/{cell['len2']['lib_in']:.3f}) "
                  f"beam_w16={cell['beam_w16']:.3f} best1={cell['len1']['best_out_prog']}",
                  flush=True)
        return out

    print("\n=== ceiling BEFORE (frozen plant) ===")
    before = snapshot("c0")

    replay = shared["replay"]["x"]
    for cycle in range(1, n_cycles + 1):
        solved = []
        for k, ctx in enumerate(ctxs):
            r_np, x_np = context_instances(rules, ctx, cfg["n_pr"], s, depth, v, m,
                                           seed=cfg["seed"] + 300_000 + 977 * cycle + k)
            out = beam_moves(shared["controller"], gen, shared["value0"],
                             torch.from_numpy(x_np), torch.from_numpy(r_np), ms, rules_t, canon,
                             depth, v, m, s, budget=cfg["budget"], beam_width=cfg["pr_width"],
                             device=device)
            tips = out["tips_x"].cpu().numpy()
            B, W, T = tips.shape
            sc = grade(tips.reshape(B * W, T), np.repeat(r_np, W), rules, s)[0]
            solved.append(torch.from_numpy(tips.reshape(B * W, T)[sc > 0.5]))
        new = torch.cat(solved) if solved else torch.zeros(0, shared["length"], dtype=torch.long)
        loss = finetune_generator(gen, opt, new, replay, bottom_map, v=v, s=s, n_blocks=n_blocks,
                                  n_steps=gen_steps, batch=cfg["batch_size"],
                                  replay_frac=cfg["replay_frac"], device=device, rng=rng)
        print(f"  [plant] c{cycle:2d} solved={new.shape[0]:5d} gen_loss={loss:.4f}", flush=True)

    print("\n=== ceiling AFTER (plant trained on its own successful repairs) ===")
    after = snapshot(f"c{n_cycles}")

    out = {"config": cfg, "contexts": ctxs, "n_cycles": n_cycles, "gen_lr": gen_lr,
           "gen_steps": gen_steps, "before": before, "after": after,
           "noise_floor": 0.003, "elapsed": time.time() - started}
    print("\n=== CEILING MOVEMENT (after - before; noise floor 0.003) ===")
    for name in before:
        for fam in ("len1", "len2"):
            for fld in ("best_out", "lib_out", "best_in", "lib_in"):
                d = after[name][fam][fld] - before[name][fam][fld]
                print(f"  {name:8s} {fam} {fld:9s} {before[name][fam][fld]:.3f} -> "
                      f"{after[name][fam][fld]:.3f}   delta={d:+.3f}  "
                      f"({abs(d) / 0.003:.1f}x noise)")
        d = after[name]["beam_w16"] - before[name]["beam_w16"]
        print(f"  {name:8s} beam_w16       {before[name]['beam_w16']:.3f} -> "
              f"{after[name]['beam_w16']:.3f}   delta={d:+.3f}")
    outdir = f"{DATA_DIR}/rhm_practice_crystallize/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cal_plant.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_plant.json")
    return out
