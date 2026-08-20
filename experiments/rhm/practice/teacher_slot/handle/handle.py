"""handle — the empty cell: earned vocabulary living IN WEIGHTS, as a backprop handle.

FORK NOTICE. This file is a VERBATIM COPY of `../../ratchet/ratchet.py` (2026-08-14) with the
handle intervention added. `ratchet/` is not modified at all — its results stay byte-reproducible
— and the copy is justified by the repo convention that semantically meaningful folders beat
de-duplication. The fork's fidelity is not assumed: the no-handle arms here are run as a matched
pair against their handle twins IN THIS TAG, and their numbers are also compared against `rr_s0`'s
stored ones (they differ only in the per-arm torch RNG stream, see PER-ARM STREAMS below, so the
comparison doubles as a replicate noise floor for a node that has only one seed).

WHAT IS NEW (everything else is ratchet's, unchanged):

 * `handle_net.HandleGenerator` — a committed macro entry gets a SLOT in the generator's own
   vocabulary. The generator PREDICTS the earned symbol of every span it is infilling and
   CONDITIONS on the earned symbols of the spans it can see. Read that module's header for the
   design decisions (annotate-not-substitute, zero init, per-entry vs per-level granularity).
 * `finetune_generator_handle` — the plant's own objective gains the macro-symbol cross-entropy,
   at weight `handle_lam`, on exactly the spans the level-1 loss is already supervising. This is
   the gradient path the arc's earned vocabulary never had.
 * Three new arms (`given_handle`, `practice_late_handle`, `practice_late_handle_level`), and
   `plant_probe` extended with the slot-bypassed twin of every plant readout plus the earned
   symbol's own accuracy.

PER-ARM STREAMS. ratchet consumed the GLOBAL torch RNG inside `finetune_generator`, so an arm's
trajectory depended on how many arms ran before it. That is fine for a fixed arm list but makes a
twin comparison impossible (a handle arm appended after its baseline would differ in stream as
well as in treatment). Here every arm seeds its own torch stream from its BASE name, so a handle
arm and its no-handle twin share a stream exactly and are bit-identical until the commit cycle at
which the slot is minted. That is the only reason the contrast is licensed.

--- ratchet's own header follows ---

Round 2 of the practice arc on RHM. Round 1 (`../crystallize/`) found the compile certificate
VACUOUS on a frozen plant: practice trained the judge (the value) while the generator that
executes a committed unit never moved, so the committable content was flat from cycle 1 and
committing at cycle 1 was optimal at 26x less priced time. Its precheck (`calp_s0`) validated
the fix — let the plant learn on the agent's own successful repairs and the committed-unit
ceiling moves at 31-66x the metering noise floor.

WHAT THIS ROUND COMPOSES.

 1. A LEARNING PLANT, carried by every arm. The generator fine-tunes online on configurations
    the agent actually solved (replay 0.5 against the clean setup pool), and the value adapts
    online against it — round 1's precheck confound (its wide beam degraded because the value
    was trained against the old generator) fixed by construction.

 2. A DEPTH-LADDERED DAMAGE SCHEDULE, the `level_ladder` idiom: the damage cell moves one level
    deeper per era, through NESTED cells (L1 n6 subset L2 n3 subset L3 n1), so era k+1's error
    is literally era k's error one level up. At a declared per-solve grounding budget, a level-3
    error is four blocks wide and a base-move agent must spend its whole move budget on it with
    no coordination left over — climbing is necessary, not merely available.

 3. AN EARNED VOCABULARY (`macros.py`). The committed unit is not a whole-solve program (round
    1's unit) but a MACRO ACTION: the same max-sum operator the true level move uses, over a
    table MINED from the agent's own solved configurations and parsed by the agent's own
    generator. Committing it puts a new primitive in the practice action space — the beam
    proposes it, the value scores it, the plant's fine-tuning sees trajectories that use it.
    The level-l table is defined over level-(l-1) ENTRIES, so an incomplete committed level-2
    vocabulary is a hard cap on level 3: the poisoned-region test with a mechanism.

 4. A UNIT-LP CERTIFICATE. Round 1's central lesson, in Jasper's constructive form: certify on
    the LEARNING PROGRESS OF THE COMMITTABLE CONTENT — the shadow-audition trajectory of the
    candidate macro, positive and then flat — never on task-performance level or task-LP, which
    conflate selector and plant improvement. "Positive and then flat" is enforced: silence alone
    cannot certify unless the audition has first descended by `lp_min_drop`.

ARMS differ ONLY in vocabulary-acquisition policy:
    never_base     base level-1 moves forever; must buy depth with search at the declared budget
    given          the DGP's own level-2/3 tables from cycle 1 (the `level_moves` ceiling)
    practice_gated earns them, gated by the unit-LP certificate
    practice_early commits one cycle into each era (minimal coverage) -- the poison test
    practice_late  commits at each era's end (the rent-paying control)

Run from experiments/:
  modal run rhm/practice/teacher_slot/handle/handle.py::handle_run --quick --tag smoke0
  python3 rhm/practice/teacher_slot/handle/launch_detached.py --fn handle_run --tag hr_s0 ...
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
from rhm.rhm_generative_planner import _build_generator, _train_generator
from rhm.rhm_latent_planner import _build_value_head
from rhm.rhm_sculpt_planner import _corrupt, _encode_chunked, _sample_pool
from rhm.rhm_sculpt_latent import _build_rich_controller
from rhm.practice.crystallize.units import (
    apply_move, build_move_set, corrupt_hier, grade, on_grammar_rate, oracle_rollout)
from rhm.practice.ratchet import macros as MC
from rhm.practice.teacher_slot.handle import handle_net as HN


app = modal.App("rhm-practice-ts-handle", image=image)
REMOTE = "rhm_practice_teacher_slot_handle"


# --------------------------------------------------------------------------- #
# the depth ladder
# --------------------------------------------------------------------------- #

def parse_eras(spec):
    """"1:6,2:3,3:1" -> [{level, node, name}] — the nested damage cells, one per era."""
    out = []
    for part in spec.split(","):
        lv, node = part.split(":")
        out.append({"level": int(lv), "node": int(node), "name": f"L{lv}n{node}"})
    return out


def era_ctx(era):
    return {"name": era["name"], "level": era["level"], "nodes": [era["node"]]}


def era_l2_nodes(era, s):
    """The level-2 nodes an era's damage cell spans — where a withheld level-2 tuple has to
    be produced for that era's instance to be repairable."""
    span = s ** (era["level"] - 1)
    b0 = era["node"] * span
    return sorted({b // s for b in range(b0, b0 + max(span, 1))} if span >= s
                  else {b0 // s})


def needs_excluded(clean_np, era, hold, inverse_bottom, v, s):
    """Which instances REQUIRE a withheld level-2 tuple: the clean derivation the damage was
    applied to uses one at the damage cell. The coordinator's constraint-1 gate is read on
    this split — a plant frontier is only a frontier if bootstrap success on it is
    low-but-nonzero; at ~0 the design collapses into exploration-gating."""
    if not hold:
        return np.zeros(clean_np.shape[0], bool)
    tup = level2_tuples(clean_np, inverse_bottom, v, s)
    nodes = era_l2_nodes(era, s)
    bad = np.zeros(clean_np.shape[0], bool)
    for t in hold:
        for nd in nodes:
            bad |= (tup[:, nd, :] == np.array(t)).all(-1)
    return bad


def context_instances(rules, ctx, n, s, depth, v, m, seed, require_broken=True,
                      with_clean=False):
    """n fresh instances of a damage cell (crystallize's, verbatim): a clean derivation of a
    random r*, the cell hierarchically damaged, rejection-sampled on d* > 0."""
    length = s ** depth
    roots = np.zeros(n, np.int64)
    x = np.zeros((n, length), np.int64)
    clean = np.zeros((n, length), np.int64)
    rng = np.random.default_rng(seed + 90_001)
    need = np.arange(n)
    for attempt in range(64):
        if len(need) == 0:
            break
        r, lv = _sample_pool(rules, len(need), s, seed + 7919 * attempt)
        xd = corrupt_hier(lv, rules, depth, v, m, s, ctx["level"], ctx["nodes"], rng)
        ok = (nearest_derivation_cost(rules, xd, r, s) > 0) if require_broken \
            else np.ones(len(need), bool)
        roots[need[ok]] = r[ok]
        x[need[ok]] = xd[ok]
        clean[need[ok]] = lv[ok]
        need = need[~ok]
    if len(need):
        raise RuntimeError(f"context {ctx['name']}: {len(need)} instances never broke")
    return (roots, x, clean) if with_clean else (roots, x)


# --------------------------------------------------------------------------- #
# pricing: the declared per-solve grounding budget picks the beam width
# --------------------------------------------------------------------------- #

def beam_ground(n_moves, budget, w):
    total, width = 0, 1
    for _ in range(budget):
        total += width * n_moves
        width = min(w, width * n_moves)
    return total + width


def fit_width(n_moves, budget, g_budget):
    """The widest beam that fits the declared budget — round 1's performance idiom, now
    arm-dependent because arms carry different numbers of moves and each materialisation is
    a grounding. This is what MATCHED PRICING means here: a level move costs the same
    groundings whoever holds it."""
    best = 1
    for w in range(1, 129):
        if beam_ground(n_moves, budget, w) <= g_budget:
            best = w
        else:
            break
    return best


# --------------------------------------------------------------------------- #
# the beam, over a MIXED action set (base level moves + earned macros)
# --------------------------------------------------------------------------- #

def beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
               *, budget, beam_width, device, collect=False):
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
            children = [MC.apply_any(generator, flat, ms[k], rules_t, canon, depth, v, m, s)
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
            out["traj"] = traj
    return out


# --------------------------------------------------------------------------- #
# online learning: the value (selector) and the generator (plant)
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
        px, pr, py = [], [], []
        if n_new:
            i = torch.from_numpy(rng.integers(0, buf["x"].shape[0], size=n_new))
            px.append(buf["x"][i]); pr.append(buf["r"][i]); py.append(buf["y"][i])
        if n_rep:
            i = torch.from_numpy(rng.integers(0, replay["x"].shape[0], size=n_rep))
            px.append(replay["x"][i]); pr.append(replay["r"][i]); py.append(replay["y"][i])
        x = torch.cat(px).to(device); r = torch.cat(pr).to(device); y = torch.cat(py).to(device)
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


def finetune_generator(generator, opt, new_leaves, replay_leaves, bottom_map, *, v, s, n_blocks,
                       n_steps, batch, replay_frac, device, rng):
    """THE PLANT LEARNS (calp_s0's recipe, verbatim): masked infilling on configurations the
    agent SOLVED — a solved config is a valid r* derivation, so the target is well defined —
    with replay `replay_frac` against the clean setup pool as the anti-drift guard."""
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


def finetune_generator_handle(generator, opt, new_leaves, replay_leaves, bottom_map, *,
                              v, s, n_blocks, n_steps, batch, replay_frac, device, rng,
                              lam=1.0):
    """THE PLANT LEARNS, WITH A HANDLE. `finetune_generator` verbatim — same batches, same
    number of optimizer steps, same masking draws, same level-1 loss — plus, for every level
    that has a committed slot, the cross-entropy of the EARNED SYMBOL of each span the level-1
    loss is already supervising (a span counts as supervised if any of its blocks is masked).

    Two properties make the twin comparison clean:
      * with no slots minted the graph is op-for-op the original, so a handle arm is
        bit-identical to its no-handle twin until its commit cycle;
      * `n_steps` is untouched, so the handle buys no extra optimizer steps — only an extra
        term in the loss. (Plant training is unpriced in this substrate for both arms alike.)"""
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    last, mlast, macc = 0.0, {}, {}
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
        hid = generator.hidden(obs)
        bb, length, dim = hid.shape
        logits = generator.core.feature_head(
            hid.view(bb, n_blocks, s, dim).mean(dim=2))
        mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, mb, True)
        loss = F.cross_entropy(logits[mask], feats[mask])
        for key in generator.levels():
            span = generator.luts[key]["span"]
            tgt = generator.targets_from_feats(feats, key)             # (b, n_spans)
            smask = mask.view(b, n_blocks // span, span).any(-1)       # supervised spans
            if not bool(smask.any()):
                continue
            mlog = generator.macro_logits(key, hidden=hid)[smask]
            mloss = F.cross_entropy(mlog, tgt[smask])
            loss = loss + lam * mloss
            mlast[key] = float(mloss.item())
            macc[key] = float((mlog.argmax(-1) == tgt[smask]).float().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    return last, mlast, macc


def push(buf, x, r, y, cap):
    import torch
    buf["x"] = torch.cat([buf["x"], x])[-cap:]
    buf["r"] = torch.cat([buf["r"], r])[-cap:]
    buf["y"] = torch.cat([buf["y"], y])[-cap:]


# --------------------------------------------------------------------------- #
# setup: instruments trained ONCE, forked per arm
# --------------------------------------------------------------------------- #

def behavior_step(controller, generator, x, roots, ms, rules_t, canon, depth, v, m, s,
                  eps, device, rng):
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


def collect_value_buffer(controller, generator, rules, rules_t, canon, ms,
                         roots_pool, leaves_pool, cfg, device):
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


def train_reader(reader, leaves, bottom_map, *, v, s, n_blocks, n_steps, batch, lr, device,
                 seed=0):
    """THE READ. `_train_generator` computes its loss only on MASKED blocks, so the block head
    at a VISIBLE position is never supervised — `calp2_s0` measured the consequence: 0.63
    accuracy reading a fully visible block, *worse* than the 0.70 it gets infilling a hidden
    one, and masking elsewhere to fix the distribution shift did not help (0.599). That caps an
    earned level-3 vocabulary's precision at ~0.3, which is what `calr_s0` saw.

    So the agent gets a reader: the same architecture, the same corpus, and the SAME
    supervision channel the generator already trains against (`bottom_map` on clean
    configurations), but trained to read rather than to fill. This is bottom-level perceptual
    competence — the inverse of the `canon` rendering the agent already performs — and it is
    deliberately the only thing handed over: every composition above level 1 still has to be
    earned."""
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(reader.parameters(), lr=lr, weight_decay=1e-4)
    powers = v ** torch.arange(s, device=device)
    g = torch.Generator().manual_seed(seed)
    reader.train()
    last = (0.0, 0.0)
    for step in range(n_steps):
        idx = torch.randint(0, leaves.shape[0], (batch,), generator=g)
        x = leaves[idx].to(device)
        feats = bottom_map[(x.view(batch, n_blocks, s) * powers).sum(-1)]
        ok = feats >= 0
        logits = reader.block_logits(x)
        loss = F.cross_entropy(logits[ok], feats[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, n_steps // 4) == 0:
            acc = float((logits[ok].argmax(-1) == feats[ok]).float().mean())
            last = (float(loss.item()), acc)
            print(f"  reader       step {step + 1:5d}/{n_steps}: loss={last[0]:.4f} "
                  f"read_acc={acc:.3f}", flush=True)
    reader.eval()
    for p in reader.parameters():
        p.requires_grad_(False)
    return last


def holdout_set(rules, depth, s, v, m, n_hold, seed):
    """The plant's frontier: `n_hold` of the grammar's level-2 tuples, withheld from the
    setup corpus. Level 2 is chosen because every era's damage requires producing a correct
    level-2 tuple (era 1 at one block of it, era 2 at the whole node, era 3 at both nodes of a
    level-3 span), so the plant's hole and the vocabulary's hole are the SAME hole — which is
    what keeps the compound descent from decoupling."""
    truth2 = MC.true_tables(rules, depth, s, v, m, 2)[2]
    flats = sorted({tuple(int(x) for x in r) for r in truth2["flat"]})
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(flats))[:n_hold]
    return {flats[i] for i in pick}


def level2_tuples(leaves_np, inverse_bottom, v, s):
    """Exact level-1 features per block -> the (N, n_l2, s) level-2 tuples."""
    feats = MC.exact_features(leaves_np, inverse_bottom, v, s)
    return feats.reshape(feats.shape[0], -1, s)


def filter_pool(leaves_np, roots_np, inverse_bottom, v, s, hold):
    """Reject configurations that use a held-out level-2 tuple anywhere. The setup corpus is
    filtered; the EVALUATION instances are drawn from the unfiltered DGP, exactly the
    train-on-generic / evaluate-on-the-real-thing structure round 1 used to make its value
    arrive stale."""
    if not hold:
        return roots_np, leaves_np, 1.0
    tup = level2_tuples(leaves_np, inverse_bottom, v, s)
    bad = np.zeros(leaves_np.shape[0], bool)
    for t in hold:
        bad |= (tup == np.array(t)).all(-1).any(-1)
    keep = ~bad
    return roots_np[keep], leaves_np[keep], float(keep.mean())


def build_shared(cfg, device):
    """Controller / generator / reader / stale value, trained once. The BASE action space
    (level-1 moves only) is what the setup value's behaviour policy explores, so every arm
    starts from the same stale selector whatever vocabulary it is later given."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    length = s ** depth
    n_blocks = length // s

    rules = generate_rules_distinct(v, s, depth, m, seed=cfg["rule_seed"])
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    base_ms = build_move_set(depth, s, max_level=1)

    hold = holdout_set(rules, depth, s, v, m, cfg["plant_holdout"], cfg["holdout_seed"])
    want = cfg["n_train_episodes"]
    over = 1.0
    roots_np, leaves_np = _sample_pool(rules, want, s, cfg["train_seed"])
    if hold:
        # oversample so the FILTERED corpus is still `n_train_episodes` long -- otherwise the
        # holdout arm would also be a less-data arm and the two would be confounded
        _, _, frac = filter_pool(leaves_np, roots_np, inverse_maps[-1], v, s, hold)
        over = max(1.05, 1.0 / max(frac, 0.05)) * 1.15
        roots_np, leaves_np = _sample_pool(rules, int(want * over), s, cfg["train_seed"])
        roots_np, leaves_np, frac = filter_pool(leaves_np, roots_np, inverse_maps[-1],
                                                v, s, hold)
        roots_np, leaves_np = roots_np[:want], leaves_np[:want]
        print(f"  plant holdout: {len(hold)} level-2 tuples withheld; corpus keep-rate "
              f"{frac:.3f}, corpus size {leaves_np.shape[0]}")
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
    reader = _build_generator()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2,
                                root_conditioned=False).to(device)
    read_loss, read_acc = train_reader(reader, leaves, bottom_map, v=v, s=s, n_blocks=n_blocks,
                                       n_steps=cfg["reader_steps"], batch=cfg["batch_size"],
                                       lr=3e-4, device=device, seed=cfg["train_seed"] + 5)
    controller.eval(); generator.eval()
    for p in controller.parameters():
        p.requires_grad_(False)

    print(f"Collecting the generic (stale) value buffer: {cfg['value_episodes']} rollouts")
    vb = collect_value_buffer(controller, generator, rules, rules_t, canon, base_ms,
                              roots_np, leaves_np, cfg, device)
    value = _build_value_head()(cfg["state_dim"], v).to(device)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["train_seed"] + 31)
    empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
    print(f"  buffer {vb['x'].shape[0]} states, terminal success {float(vb['y'].mean()):.3f}")
    value_steps(value, opt, controller, vb, empty, n_steps=cfg["value_steps"],
                batch=512, replay_frac=0.0, device=device, rng=rng)

    truth = MC.true_tables(rules, depth, s, v, m, cfg["max_macro_level"])
    return {"rules": rules, "rules_t": rules_t, "canon": canon, "inverse_maps": inverse_maps,
            "bottom_map": bottom_map, "base_ms": base_ms, "controller": controller,
            "generator0": generator, "reader": reader, "read_acc": read_acc,
            "hold": hold, "value0": value, "replay": vb, "leaves_pool": leaves_np,
            "roots_pool": roots_np, "truth": truth, "n_blocks": n_blocks, "length": length}


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

ARMS = {
    "never_base":     {"vocab": "base",   "commit": None,    "handle": None},
    "given":          {"vocab": "true",   "commit": None,    "handle": None},
    "practice_gated": {"vocab": "earned", "commit": "delta", "handle": None},
    "practice_early": {"vocab": "earned", "commit": "early", "handle": None},
    "practice_late":  {"vocab": "earned", "commit": "late",  "handle": None},
    # --- the empty cell: the same vocabulary, additionally given slots in the plant's own
    #     vocabulary so it predicts and conditions on it. Everything else is the twin's.
    "given_handle":               {"vocab": "true",   "commit": None,   "handle": "entry"},
    "practice_late_handle":       {"vocab": "earned", "commit": "late", "handle": "entry"},
    "practice_late_handle_level": {"vocab": "earned", "commit": "late", "handle": "level"},
}

# A handle arm and its no-handle twin must share a torch RNG stream, or the contrast is
# confounded by stream position. Streams are keyed by the TWIN, not by the arm.
STREAM = {
    "never_base": 0, "given": 1, "practice_gated": 2, "practice_early": 3, "practice_late": 4,
    "given_handle": 1, "practice_late_handle": 4, "practice_late_handle_level": 4,
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


def macro_moves(level, tables, s, depth, device):
    """Instantiate the level-`level` macro at EVERY node of that level, so a committed
    vocabulary is a position-independent rule set (which is what the DGP's own tables are)
    and the committed action space matches `given`'s move-for-move."""
    n_nodes = s ** (depth - level)
    return [MC.to_device(MC.make_macro(level, j, s, tables[level]), device)
            for j in range(n_nodes)]


def build_ms(base_ms, committed, s, depth, device):
    ms = list(base_ms)
    for level in sorted(committed):
        if committed[level] is None:
            continue
        ms += macro_moves(level, {level: committed[level]}, s, depth, device)
    return ms


def audition_macro(generator, x0, roots_np, move, rules_t, canon, depth, v, m, s, rules):
    """The macro's own audition: ONE action on held-out instances of the current era, graded
    by terminal possible-set success. This is the committable content — the quantity the
    unit-LP certificate watches."""
    xf = MC.apply_any(generator, x0, move, rules_t, canon, depth, v, m, s)
    succ, dres = grade(xf.cpu().numpy(), roots_np, rules, s)
    return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean())}


def run_arm(label, base, overrides, shared, cfg, eras, refs, outdir, device):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]
    hmode = spec.get("handle")

    # ---- PER-ARM torch STREAM. ratchet let the global stream run across arms, so an arm's
    #      trajectory depended on the arms before it. Keyed by the TWIN (see `STREAM`), so a
    #      handle arm and its no-handle baseline draw the identical sequence and can only
    #      diverge through the handle itself. ------------------------------------------------
    torch.manual_seed(cfg["train_seed"] * 1000 + 7 + STREAM[base])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["train_seed"] * 1000 + 7 + STREAM[base])

    # ---- per-arm plant + selector (the plant is a substrate property this round: every arm
    #      carries it, and the value co-adapts against it) ----------------------------------
    generator = copy.deepcopy(shared["generator0"]).to(device)
    if hmode:
        generator = HN.wrap(generator, v=v, s=s, n_blocks=shared["n_blocks"],
                            bottom_map=shared["bottom_map"], mode=hmode).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    slot_events = []

    def mint(level, table):
        """Give a freshly committed table a slot in the generator's own vocabulary. Zero-init,
        so this is an exact no-op at the commit instant and the symbol's whole influence is
        earned by gradient descent; new parameters join the plant's optimizer at its own lr."""
        info = generator.add_slot(level, table, mode=hmode)
        gopt.add_param_group({"params": generator.slot_params(level),
                              "lr": cfg["gen_lr"], "weight_decay": 1e-4})
        slot_events.append(info)
        print(f"[slot]   arm={arm} level={level} mode={hmode} n_sym={info['n_sym']} "
              f"span={info['span']} entries={info['n_entries']} "
              f"unique_flat={info['n_flat_unique']} params={info['n_params']}", flush=True)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(),
                             lr=cfg["value_lr_online"] or cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)

    # ---- vocabulary state ---------------------------------------------------------------
    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
            if hmode:
                mint(ell, committed[ell])
    ms = build_ms(base_ms, committed, s, depth, device)
    p_width = cfg["pr_width"]

    def operative(ell):
        """The table macros at level `ell` would be built over: the committed one if this arm
        has committed, else the live mined candidate. This is where the ratchet bites — once
        level l-1 is frozen, level l can only ever be built over what was frozen."""
        if ell == 1:
            return MC.base_table(v)
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    t_cum = 0.0
    events = []
    log = {"cycle": [], "era": [], "level": [], "t_cum": [], "e": [], "succ": [], "dres": [],
           "n_moves": [], "width": [], "g_per_solve": [], "e_practice": [], "vloss": [],
           "gloss": [], "n_solved": [], "n_mined": [], "miner": [], "aud": [], "cert": [],
           "probe": [], "vocab": [], "handle": []}

    # ---- fixed held-out sets, per era ----------------------------------------------------
    meter, shadow = {}, {}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        meter[i] = (r_np, torch.from_numpy(x_np))
        r2, x2 = context_instances(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                                   seed=cfg["seed"] + 6100 + 23 * i)
        shadow[i] = (r2, torch.from_numpy(x2).to(device))

    cyc = 0
    for era_i, era in enumerate(eras):
        active = era["level"] + 1                     # the macro level being EARNED this era
        cert = {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [], "run": 0,
                "fired": None}
        era_start = cyc + 1
        era_end = cyc + cfg["era_cycles"]
        print(f"\n----- arm={arm} ERA {era_i + 1}: damage {era['name']} "
              f"(earning level {active}) c{era_start}..{era_end} -----", flush=True)

        for c_in_era in range(1, cfg["era_cycles"] + 1):
            cyc += 1
            counts = {"mat": 0, "ground": 0}

            # --- (a) practice: wide closed-loop beam on fresh instances of this era's cell
            r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                                           seed=cfg["seed"] + 100_000 + 1000 * cyc)
            out = beam_moves(controller, generator, value, torch.from_numpy(x_np),
                             torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                             budget=cfg["budget"], beam_width=p_width, device=device,
                             collect=True)
            for key in counts:
                counts[key] += out["counts"][key]
            tips = out["tips_x"]
            B, W, T = tips.shape
            tips_flat = tips.reshape(B * W, T)
            succ, _ = grade(tips_flat.cpu().numpy(), np.repeat(r_np, W), rules, s)
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in out["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
            ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(out["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            solved = tips_flat[torch.from_numpy(succ > 0.5).to(device)]
            ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
            e_practice = 1.0 - float(ps.mean())

            # --- (b) THE VOCABULARY IS MINED from what the agent actually solved, parsed by
            #         the agent's OWN generator. Every level's span containing this era's
            #         damage cell is observed; whether an observation becomes an ENTRY is
            #         decided at build time by the operative lower table (the ratchet).
            #         `mine_from="chosen"` records only the beam's OWN final answer per
            #         instance — the agent's performances, not the leaves of its search tree.
            #         With 16 tips per instance the tree saturates a 14-entry table in one
            #         cycle, which would make the certificate vacuous for a bookkeeping reason
            #         rather than a substrate one (round 1's `cald_s0` trap, one level up).
            mine_src = (out["x"][torch.from_numpy(ps > 0.5).to(device)]
                        if cfg["mine_from"] == "chosen" else solved)
            if cfg["mine_cap"] and mine_src.shape[0] > cfg["mine_cap"]:
                sel = torch.from_numpy(rng.permutation(mine_src.shape[0])[:cfg["mine_cap"]])
                mine_src = mine_src[sel.to(device)]
            #         Levels are mined only up to `era_level + 1`: during era k the agent forms
            #         level-(k+1) chunks out of what it is currently producing. Mining level 3
            #         during era 1 would hand era 2 a finished vocabulary before era 2 begins,
            #         which would erase both the poison test and the unit-LP curve it needs.
            if mine_src.shape[0]:
                # read with the READER, not the generator: `calp2_s0` measured that the
                # generator's block head is unsupervised at visible positions (0.63), which
                # capped the earned level-3 vocabulary's precision at ~0.3.
                pf = MC.parse_features(shared["reader"], mine_src, s=s).cpu().numpy()
                for ell in range(2, min(maxl, era["level"] + 1) + 1):
                    span = s ** (ell - 1)
                    node = (era["node"] * s ** (era["level"] - 1)) // span
                    miners[ell].observe(pf[:, node * span:(node + 1) * span])

            # --- (c) the plant learns, then the selector ---------------------------------
            mloss, macc = {}, {}
            if hmode:
                gloss, mloss, macc = finetune_generator_handle(
                    generator, gopt, solved.cpu(), shared["replay"]["x"],
                    shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                    n_steps=cfg["gen_steps"], batch=cfg["batch_size"],
                    replay_frac=cfg["replay_frac"], device=device, rng=grng,
                    lam=cfg["handle_lam"])
            else:
                gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                           shared["bottom_map"], v=v, s=s,
                                           n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                           batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                           device=device, rng=grng)
            vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                replay_frac=cfg["replay_frac"], device=device, rng=rng)

            # --- (d) metering under PERFORMANCE conditions (the declared grounding budget) -
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

            # --- (e) THE SHADOW AUDITION: what the candidate macro would score, every cycle.
            #         `cand` is the earned table; `true` is the DGP's own (an oracle readout,
            #         unpriced); `held` is the frozen committed one (the counterfactual recert
            #         readout — what freezing costs, measured but never acted on).
            sr_np, sx = shadow[era_i]
            aud = {}
            for ell in range(2, maxl + 1):
                span = s ** (ell - 1)
                node = (era["node"] * s ** (era["level"] - 1)) // span
                cell = {}
                # the LIVE candidate, built over whatever this arm's level-(l-1) vocabulary
                # actually is: the frozen one if it committed (the ratchet), the live mined one
                # otherwise, the DGP's own for `given`. Computed for EVERY arm so the
                # compounding question ("does level 3 descend faster given level 2?") is a
                # four-way comparison rather than a within-arm anecdote.
                tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                cell["n_entries"] = int(tbl["child"].shape[0])
                if cell["n_entries"]:
                    mv = MC.to_device(MC.make_macro(ell, node, s, tbl), device)
                    cell["cand"] = audition_macro(generator, sx, sr_np, mv, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    cell.update({f"tab_{k}": val for k, val in
                                 MC.grade_table(tbl, shared["truth"][ell]).items()})
                    if ell == active and committed.get(ell) is None:
                        counts["ground"] += sx.shape[0]      # priced: the agent's own check
                        counts["mat"] += sx.shape[0]
                else:
                    cell["cand"] = None
                mvt = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
                cell["true"] = audition_macro(generator, sx, sr_np, mvt, rules_t, canon,
                                              depth, v, m, s, rules)["e"]
                # MATCHED-SIZE RANDOM CONTROL (oracle instrument, unpriced). `cal_ladder`
                # measured that a k-entry table's audition has huge entry-IDENTITY variance at
                # small k (L3 k=4 spans 0.10-0.97 over uniform draws). If mining recovered a
                # random subset the earned vocabulary would be a lottery; if it beats the
                # matched-size random control, practice mines the USEFUL part of the
                # vocabulary first, which is a claim about practice rather than about size.
                if cell["n_entries"]:
                    full = shared["truth"][ell]
                    n_all = full["child"].shape[0]
                    k = min(cell["n_entries"], n_all)
                    es = []
                    for _ in range(3):
                        keep = np.sort(rng.permutation(n_all)[:k])
                        sub = MC.make_table(ell, full["child"][keep], full["lower"], s)
                        mvr = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                        es.append(audition_macro(generator, sx, sr_np, mvr, rules_t, canon,
                                                 depth, v, m, s, rules)["e"])
                    cell["rand_k"] = float(np.mean(es))
                    cell["rand_k_sd"] = float(np.std(es))
                if committed[ell] is not None and spec["vocab"] == "earned":
                    # the counterfactual recert: what the frozen unit scores now against what
                    # the live vocabulary would score. Measured, never acted on -- committed
                    # macros stay frozen, which is what makes the poison test a real test.
                    mvc = MC.to_device(MC.make_macro(ell, node, s, committed[ell]), device)
                    cell["held"] = audition_macro(generator, sx, sr_np, mvc, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    live = miners[ell].build(MC.base_table(v) if ell == 2
                                             else miners[ell - 1].build(MC.base_table(v),
                                                                        cfg["mine_support"]),
                                             cfg["mine_support"])
                    if live["child"].shape[0]:
                        mvl = MC.to_device(MC.make_macro(ell, node, s, live), device)
                        cell["live"] = audition_macro(generator, sx, sr_np, mvl, rules_t, canon,
                                                      depth, v, m, s, rules)["e"]
                        cell["live_entries"] = int(live["child"].shape[0])
                aud[str(ell)] = cell

            # --- (f) THE UNIT-LP CERTIFICATE: silence on the audition trajectory of the
            #         committable content, with an explicit "it descended first" precondition.
            A = aud.get(str(active), {}).get("cand")   # None until the vocabulary is non-empty
            fire_cert = False
            if A is not None and committed.get(active) is None:
                if cert["ref"] is None:
                    cert["ref"] = A; cert["b"] = A; cert["emin"] = A
                cert["emin"] = min(cert["emin"], A)
                d = cert["b"] - A
                scale = max(cert["ref"] - cert["emin"], cert["b"], 1e-6)
                cert["hist"].append(A); cert["dhist"].append(d)
                we, wd = cert["hist"][-cfg["sil_W"]:], cert["dhist"][-cfg["sil_W"]:]
                quiet = (len(we) >= cfg["sil_W"]
                         and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                         and float(np.std(we)) < cfg["sil_cv"] * scale)
                cert["run"] = cert["run"] + 1 if quiet else 0
                cert["b"] = cert["b"] + cfg["alpha"] * (A - cert["b"])
                dropped = (cert["ref"] - cert["emin"]) >= cfg["lp_min_drop"]
                fire_cert = (cert["run"] >= cfg["sil_hold"] and dropped
                             and c_in_era >= cfg["sil_min_cycle"])

            # --- (g) the commit rule ------------------------------------------------------
            do_commit = False
            if spec["commit"] and active <= maxl and committed.get(active) is None:
                if spec["commit"] == "delta":
                    do_commit = fire_cert
                elif spec["commit"] == "early":
                    do_commit = c_in_era >= cfg["early_offset"]
                elif spec["commit"] == "late":
                    do_commit = c_in_era >= cfg["era_cycles"] - cfg["late_offset"]
            if do_commit:
                tbl = miners[active].build(operative(active - 1), cfg["mine_support"])
                if tbl["child"].shape[0] == 0:
                    do_commit = False
            if do_commit:
                # a fresh held-out audition, PRICED — the agent pays to verify what it commits
                cr_np, cx_np = context_instances(rules, era_ctx(era), cfg["n_score"], s, depth,
                                                 v, m, seed=cfg["seed"] + 700_000 + 1000 * cyc)
                cx = torch.from_numpy(cx_np).to(device)
                span = s ** (active - 1)
                node = (era["node"] * s ** (era["level"] - 1)) // span
                mv = MC.to_device(MC.make_macro(active, node, s, tbl), device)
                held = audition_macro(generator, cx, cr_np, mv, rules_t, canon, depth,
                                      v, m, s, rules)
                counts["ground"] += cx.shape[0]; counts["mat"] += cx.shape[0]
                committed[active] = tbl
                if hmode:
                    mint(active, tbl)          # the earned vocabulary enters the WEIGHTS here
                ms = build_ms(base_ms, committed, s, depth, device)
                cert["fired"] = cyc
                ev = dict(kind="commit", arm=arm, era=era_i + 1, level=active, cycle=cyc,
                          c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg),
                          audition=held["e"], shadow=A, e_task=e,
                          n_entries=int(tbl["child"].shape[0]),
                          n_moves_after=len(ms), sil_run=int(cert["run"]),
                          **{f"tab_{k}": val for k, val in
                             MC.grade_table(tbl, shared["truth"][active]).items()})
                events.append(ev)
                print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} "
                      f"entries={ev['n_entries']} recall={ev['tab_recall']:.3f} "
                      f"prec={ev['tab_precision']} audition={held['e']:.4f} shadow={A} "
                      f"moves={len(ms)}", flush=True)

            # --- (h) probes: the width ladder + all eras' metering sets + the plant guard.
            #         Instruments, never priced.
            probe = None
            if c_in_era == 1 or cyc % cfg["probe_every"] == 0 or c_in_era == cfg["era_cycles"]:
                probe = {"cycle": cyc, "era": era_i + 1, "ladder": {}, "all_eras": {}}
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
                if hmode:
                    probe["plant"].update(handle_probe(generator, shared, cfg, device))
                log["probe"].append(probe)

            t_cum += priced(counts, cfg)
            log["cycle"].append(cyc); log["era"].append(era_i + 1); log["level"].append(era["level"])
            log["t_cum"].append(t_cum); log["e"].append(e); log["succ"].append(float(msucc.mean()))
            log["dres"].append(float(mdres.mean())); log["n_moves"].append(len(ms))
            log["width"].append(width); log["g_per_solve"].append(gps)
            log["e_practice"].append(e_practice); log["vloss"].append(vloss)
            log["gloss"].append(gloss); log["n_solved"].append(int(solved.shape[0]))
            log["n_mined"].append(int(mine_src.shape[0]))
            log["miner"].append({str(ell): miners[ell].state() for ell in range(2, maxl + 1)})
            log["aud"].append(aud)
            log["cert"].append({"A": A, "b": cert["b"], "run": int(cert["run"]),
                                "ref": cert["ref"], "emin": cert["emin"]})
            log["handle"].append({"mloss": mloss, "macc": macc,
                                  "slots": [int(k) for k in generator.levels()]
                                  if hmode else []})
            log["vocab"].append({str(ell): (None if committed[ell] is None
                                            else int(committed[ell]["child"].shape[0]))
                                 for ell in range(2, maxl + 1)})
            a_act = aud.get(str(active), {})
            print(f"[c{cyc:3d}] arm={arm:15s} era{era_i+1} t={t_cum:10.0f} e={e:.4f} "
                  f"w={width} nm={len(ms)} solved={solved.shape[0]:4d} "
                  f"A{active}={a_act.get('cand')} true={a_act.get('true')} "
                  f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
                  f"sil={cert['run']}", flush=True)
            if cyc % cfg["checkpoint_every"] == 0:
                write_results(outdir, arm, cfg, eras, refs, log, events, complete=False,
                              extra={"handle_mode": hmode, "slot_events": slot_events})

    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True,
                  extra={"handle_mode": hmode, "slot_events": slot_events})
    return {"log": log, "events": events}


def handle_probe(generator, shared, cfg, device):
    """The handle's own readouts, all on the SAME clean held-out configurations `plant_probe`
    uses, so the two are directly comparable.

      *_nc      the slot-bypassed twin of every plant readout — the generator's weights alone,
                with the earned symbols not fed in. This is the honest "did the weights move"
                number: `parse_acc` on a fully visible configuration is measured WITH the
                symbols supplied, and a symbol determines its span's level-1 features exactly,
                so the conditioned number can rise for a trivial reason. (At repair time no
                such leak exists: the macro DP masks the span it is repairing, so that span's
                symbol is 0.)
      macro_acc did the earned symbol get learned at all? Accuracy of the macro head on the
                span containing the masked block, against `base` = the rate of the majority
                symbol on the same spans (mostly symbol 0 = "no committed entry covers this").
    """
    import torch
    v, s = cfg["v"], cfg["s"]
    n_blocks = shared["n_blocks"]
    x = shared["probe_clean"].to(device)
    b = x.shape[0]
    powers = v ** torch.arange(s, device=device)
    feats = shared["bottom_map"][(x.view(b, n_blocks, s) * powers).sum(-1)]
    pick = torch.from_numpy(
        np.random.default_rng(0).integers(0, n_blocks, size=b)).to(device)
    pos = pick[:, None] * s + torch.arange(s, device=device)[None, :]
    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    out = {}
    with torch.no_grad():
        parse_nc = generator.block_logits(x, use_slots=False).argmax(-1)
        infill_nc = generator.block_logits(obs, use_slots=False).argmax(-1)
        hid = generator.hidden(obs)
    tgt = feats.gather(1, pick[:, None]).squeeze(1)
    ok, ok1 = feats >= 0, tgt >= 0
    out["parse_acc_nc"] = float((parse_nc[ok] == feats[ok]).float().mean())
    out["infill_acc_nc"] = float(
        (infill_nc.gather(1, pick[:, None]).squeeze(1)[ok1] == tgt[ok1]).float().mean())
    for key in generator.levels():
        span = generator.luts[key]["span"]
        with torch.no_grad():
            ml = generator.macro_logits(key, hidden=hid)
        mt = generator.targets_from_feats(feats, key)
        row = torch.arange(b, device=device)
        hit = (pick // span)                                  # the span holding the masked block
        pred = ml[row, hit].argmax(-1)
        true = mt[row, hit]
        out[f"macro_acc_{key}"] = float((pred == true).float().mean())
        out[f"macro_base_{key}"] = float(
            torch.bincount(true, minlength=generator.luts[key]["n_sym"]).max() / b)
        out[f"macro_infreq0_{key}"] = float((true > 0).float().mean())
        out[f"slot_emb_norm_{key}"] = float(generator.slots[key].emb.weight.norm())
    return out


def plant_probe(generator, shared, cfg, device):
    """The self-imitation-collapse guard: the plant's fidelity to the grammar, measured on
    CLEAN held-out configurations it never trains on directly.
      infill_acc -- masked-block level-1 feature accuracy (the generator's actual job)
      parse_acc  -- unmasked-block feature accuracy, i.e. how right the agent's own parse is,
                    which is what mining reads. Both are oracle readouts, never consumed."""
    import torch
    v, s = cfg["v"], cfg["s"]
    n_blocks = shared["n_blocks"]
    x = shared["probe_clean"].to(device)
    b = x.shape[0]
    powers = v ** torch.arange(s, device=device)
    feats = shared["bottom_map"][(x.view(b, n_blocks, s) * powers).sum(-1)]
    pick = torch.from_numpy(
        np.random.default_rng(0).integers(0, n_blocks, size=b)).to(device)
    pos = pick[:, None] * s + torch.arange(s, device=device)[None, :]
    with torch.no_grad():
        parse = generator.block_logits(x).argmax(-1)
        obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
        infill = generator.block_logits(obs).argmax(-1)
    tgt = feats.gather(1, pick[:, None]).squeeze(1)
    pred = infill.gather(1, pick[:, None]).squeeze(1)
    ok = feats >= 0
    ok1 = tgt >= 0
    with torch.no_grad():
        rd = shared["reader"].block_logits(x).argmax(-1)
    return {"parse_acc": float((parse[ok] == feats[ok]).float().mean()),
            "infill_acc": float((pred[ok1] == tgt[ok1]).float().mean()),
            "read_acc": float((rd[ok] == feats[ok]).float().mean())}


def priced(counts, cfg):
    return counts["ground"] * cfg["d_fb"] + counts["mat"] * cfg["c_mat"]


def write_results(outdir, arm, cfg, eras, refs, log, events, complete, extra=None):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": eras, "refs": refs,
                   "log": log, "events": events, "complete": complete,
                   **(extra or {})},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# references, measured at setup ON THE METERING SETS THEMSELVES
# --------------------------------------------------------------------------- #

def measure_refs(shared, cfg, eras, device):
    """`stale` is the detector's scale anchor and round 1's bug: it measured the reference on
    a 512-instance reference draw and the metering on a different 384-instance draw, so the
    two disagreed by more than the descent and the scale collapsed to a much looser fallback.
    Fixed here — every reference below is measured on the ERA'S OWN METERING SET.

    Also measured per era: the exact-DP floor over the base move set, the base-move and
    true-vocabulary performance beams at the declared budget, and the one-action ceilings of
    the true level-l macro (the number `given` is buying and `practice_*` is chasing)."""
    import torch
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                 for ell in range(2, cfg["max_macro_level"] + 1)},
                       s, depth, device)
    out = {"stale": [], "stale_true": [], "floor": [], "d0": [], "on_grammar": [],
           "width_base": [], "width_true": [], "g_base": [], "g_true": [], "macro_true": []}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np)
        out["d0"].append(float(nearest_derivation_cost(rules, x_np, r_np, s).mean()))
        out["on_grammar"].append(on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s))
        for tag, mset in (("base", base_ms), ("true", true_ms)):
            w = fit_width(len(mset), cfg["budget"], cfg["g_budget"])
            b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                           torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                           budget=cfg["budget"], beam_width=w, device=device)
            sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
            out["stale" if tag == "base" else "stale_true"].append(1.0 - float(sc.mean()))
            out[f"width_{tag}"].append(w)
            out[f"g_{tag}"].append(b["counts"]["ground"] / x0.shape[0])
        xo, _ = oracle_rollout(shared["generator0"], x0.to(device), r_np, base_ms, rules,
                               rules_t, canon, depth, v, m, s, cfg["budget"])
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        out["floor"].append(1.0 - float(sc.mean()))
        cell = {}
        for ell in range(2, cfg["max_macro_level"] + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            mv = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
            cell[str(ell)] = audition_macro(shared["generator0"], x0.to(device), r_np, mv,
                                            rules_t, canon, depth, v, m, s, rules)["e"]
        out["macro_true"].append(cell)
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
        budget=4, pr_width=16, g_budget=58, max_macro_level=3,
        n_pr=64, n_rt=384, n_score=512, n_probe_clean=512,
        era_cycles=24, n_grad=4, value_batch=256, replay_frac=0.5, buf_cap=100_000,
        gen_lr=1e-4, gen_steps=20, mine_support=3, mine_from="chosen", mine_cap=0,
        alpha=0.2, sil_c=0.06, sil_cv=0.10, sil_W=5, sil_hold=2, sil_min_cycle=6,
        lp_min_drop=0.05, early_offset=1, late_offset=2,
        d_fb=1.0, c_mat=0.05, checkpoint_every=5, probe_widths=(1, 4, 16), probe_every=4,
        handle_lam=1.0,
    )
    cfg.update(kw)
    return cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=32768)
def handle_run(
    tag: str = "smoke",
    arms: str = ("never_base,given,given_handle,"
                 "practice_late,practice_late_handle,practice_late_handle_level"),
    eras: str = "1:6,2:3,3:1", seed: int = 0, era_cycles: int = 24, budget: int = 4,
    pr_width: int = 16, g_budget: int = 58, max_macro_level: int = 3,
    n_pr: int = 64, n_rt: int = 384, n_score: int = 512, n_grad: int = 4,
    gen_lr: float = 1e-4, gen_steps: int = 20, mine_support: int = 3,
    mine_from: str = "chosen", mine_cap: int = 0,
    value_lr_online: float = 3e-5, sil_c: float = 0.06, sil_cv: float = 0.10,
    sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6, lp_min_drop: float = 0.05,
    early_offset: int = 1, late_offset: int = 2, probe_every: int = 4,
    plant_holdout: int = 0, holdout_seed: int = 11,
    d_fb: float = 1.0, c_mat: float = 0.05, handle_lam: float = 1.0, quick: bool = False,
):
    import torch
    cfg = _cfg(handle_lam=handle_lam,
               seed=seed, era_cycles=era_cycles, budget=budget, pr_width=pr_width,
               g_budget=g_budget, max_macro_level=max_macro_level, n_pr=n_pr, n_rt=n_rt,
               n_score=n_score, n_grad=n_grad, gen_lr=gen_lr, gen_steps=gen_steps,
               mine_support=mine_support, mine_from=mine_from, mine_cap=mine_cap,
               value_lr_online=value_lr_online, sil_c=sil_c,
               sil_cv=sil_cv, sil_W=sil_win, sil_hold=sil_hold, sil_min_cycle=sil_min_cycle,
               lp_min_drop=lp_min_drop, early_offset=early_offset, late_offset=late_offset,
               probe_every=probe_every, plant_holdout=plant_holdout,
               holdout_seed=holdout_seed, d_fb=d_fb, c_mat=c_mat)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600,
                   value_episodes=6_000, n_train_episodes=20_000, era_cycles=5, n_pr=24,
                   n_rt=96, n_score=96, n_probe_clean=128, checkpoint_every=2,
                   sil_min_cycle=2, early_offset=1, late_offset=1, gen_steps=5)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"handle tag={tag} arms={arms} eras={ers} device={device}")

    shared = build_shared(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {shared['truth'][ell]['child'].shape[0]} entries")
    print(f"base action space: {len(shared['base_ms'])} moves; "
          f"with earned macros: {8 + 4 + 2}")
    refs = measure_refs(shared, cfg, ers, device)
    print(f"[refs] {json.dumps(refs, cls=NumpyEncoder)}")

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    results = {}
    for label, base, ov in parse_arms(arms):
        print(f"\n===== arm {label} (base {base}, overrides {ov}) =====", flush=True)
        results[label] = run_arm(label, base, ov, shared, cfg, ers, refs, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"refs": refs, "elapsed": time.time() - started}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_ladder(tag: str = "call_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
               budget: int = 4, n_ref: int = 512, g_budgets: str = "33,58,86,100,108",
               widths: str = "1,2,3,4,8,16", max_macro_level: int = 3, quick: bool = False):
    """CALIBRATION 1 — the depth ladder, before any loop is built.

    Answers, per era and on identical held-out instances: how hard is level-k damage for the
    BASE move set at each beam width, what does handing over the TRUE level vocabulary buy,
    what does one true macro action buy, and where the exact-DP floor is. This is what picks
    the declared per-solve grounding budget `g_budget`: it must be a regime where depth is
    unaffordable to base moves and affordable with the vocabulary, otherwise the round's
    headline is a property of a budget chosen by hand."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_rt=128)
    ers = parse_eras(eras)
    ws = [int(x) for x in widths.split(",")]
    gs = [int(x) for x in g_budgets.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    l2_ms = build_ms(base_ms, {2: shared["truth"][2]}, s, depth, device)
    true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                 for ell in range(2, max_macro_level + 1)}, s, depth, device)
    sets = (("base", base_ms), ("l2", l2_ms), ("true", true_ms))
    out = {"config": cfg, "eras": ers, "cells": {},
           "n_moves": {t: len(x) for t, x in sets},
           "ground_table": {f"{t}_w{w}": beam_ground(len(mset), budget, w)
                            for t, mset in sets for w in ws}}
    for i, era in enumerate(ers):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np)
        cell = {"d0": float(nearest_derivation_cost(rules, x_np, r_np, s).mean()),
                "on_grammar": on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s)}
        for tag_, mset in sets:
            cell[tag_] = {}
            for w in ws:
                b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                               torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                               budget=budget, beam_width=w, device=device)
                sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                cell[tag_][str(w)] = {"e": 1.0 - float(sc.mean()),
                                      "g": b["counts"]["ground"] / x0.shape[0]}
        cell["at_budget"] = {}
        for g in gs:
            row = {}
            for tag_, mset in sets:
                w = fit_width(len(mset), budget, g)
                row[tag_] = {"w": w, "g": beam_ground(len(mset), budget, w),
                             "e": cell[tag_].get(str(w), {}).get("e")}
            cell["at_budget"][str(g)] = row
        # one macro ACTION, at full and at partial vocabulary coverage. The coverage curve is
        # what decides whether the unit-LP certificate has a resolvable descent to certify:
        # if audition is flat in coverage, the round has no signal and that is worth knowing
        # before the main run rather than after it.
        cell["macro_true"], cell["coverage"] = {}, {}
        crng = np.random.default_rng(cfg["seed"] + 4242)
        for ell in range(2, max_macro_level + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            full = shared["truth"][ell]
            mv = MC.to_device(MC.make_macro(ell, node, s, full), device)
            cell["macro_true"][str(ell)] = audition_macro(
                shared["generator0"], x0.to(device), r_np, mv, rules_t, canon, depth,
                v, m, s, rules)["e"]
            row = {}
            n_all = full["child"].shape[0]
            ks = sorted({1, 2, 4, 8, max(1, n_all // 4), max(1, n_all // 2),
                         max(1, 3 * n_all // 4), n_all})
            for k in ks:
                es = []
                for _ in range(1 if k == n_all else 5):
                    keep = np.sort(crng.permutation(n_all)[:k])
                    sub = MC.make_table(ell, full["child"][keep], full["lower"], s)
                    mvs = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                    es.append(audition_macro(shared["generator0"], x0.to(device), r_np, mvs,
                                             rules_t, canon, depth, v, m, s, rules)["e"])
                row[str(k)] = {"mean": float(np.mean(es)), "sd": float(np.std(es)),
                               "min": float(np.min(es)), "max": float(np.max(es)),
                               "n_entries": k, "frac": k / n_all}
            cell["coverage"][str(ell)] = row
        xo, _ = oracle_rollout(shared["generator0"], x0.to(device), r_np, base_ms, rules,
                               rules_t, canon, depth, v, m, s, budget)
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        cell["dp_floor_base"] = 1.0 - float(sc.mean())
        out["cells"][era["name"]] = cell
        print(f"\n=== era {era['name']} (d0={cell['d0']:.2f}, on_gram={cell['on_grammar']:.3f}) ===")
        for tag_, _ in sets:
            print(f"  {tag_:5s}: " + " | ".join(
                f"w{w}: e={cell[tag_][str(w)]['e']:.3f} ({cell[tag_][str(w)]['g']:.0f}g)"
                for w in ws))
        print(f"  one true macro action: " + " ".join(
            f"L{ell}={cell['macro_true'][str(ell)]:.3f}" for ell in cell["macro_true"]))
        for ell, row in cell["coverage"].items():
            print(f"  coverage L{ell} (entries: e[min,max]): " + " ".join(
                f"{k}:{c['mean']:.3f}[{c['min']:.2f},{c['max']:.2f}]"
                for k, c in sorted(row.items(), key=lambda kv: int(kv[0]))))
        print(f"  DP floor (base moves) = {cell['dp_floor_base']:.3f}")
        for g in gs:
            r = cell["at_budget"][str(g)]
            print(f"  @G={g:4d}: " + " | ".join(
                f"{t} w{r[t]['w']} ({r[t]['g']}g) e={r[t]['e']}" for t, _ in sets))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_ladder.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_ladder.json")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def cal_parse(tag: str = "calp2_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
              budget: int = 4, n_ref: int = 512, max_macro_level: int = 3,
              quick: bool = False):
    """CALIBRATION 3 — the perception bound on an earned vocabulary.

    `calr_s0` measured that the mined tables have LOW PRECISION (level-3: 10 true entries out
    of 34 mined, and `never_base` mined 100 distinct 4-tuples where the grammar has 56). The
    suspected cause is not the mining rule but the READ: the generator is trained by masked
    infilling with `mask_min=1` and had never seen a fully unmasked configuration, so parsing
    one is out of distribution.

    This isolates it. On the SAME oracle-solved configurations, the level-1 parse is taken
    three ways -- unmasked (what `calr_s0` did), with one block masked ELSEWHERE (in
    distribution, every span block still visible), and by the exact bottom inverse map (the
    privileged ceiling) -- and the resulting vocabulary is graded against the DGP's own each
    way. If the masked read recovers precision, the defect is the instrument; if it does not,
    the vocabulary is bounded by the plant and that is a substrate fact about the round."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_rt=128)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen, n_blocks = shared["generator0"], shared["n_blocks"]
    out = {"config": cfg, "eras": ers, "parse": {}, "cells": {}}

    # ---- block-level parse accuracy on CLEAN held-out configs, three ways ---------------
    clean = torch.from_numpy(_sample_pool(rules, cfg["n_rt"], s, cfg["seed"] + 31337)[1]).to(device)
    powers = v ** torch.arange(s, device=device)
    truth_f = shared["bottom_map"][(clean.view(clean.shape[0], n_blocks, s) * powers).sum(-1)]
    ok = truth_f >= 0
    for name, mb in (("unmasked", None), ("mask_elsewhere", 0)):
        pf = MC.parse_features(gen, clean, s=s, mask_block=mb)
        sel = ok.clone()
        if mb is not None:
            sel[:, mb] = False                       # the masked block is inferred, not read
        out["parse"][name] = float((pf[sel] == truth_f[sel]).float().mean())
    out["parse"]["ambiguous_frac"] = float((~ok).float().mean())
    print(f"[parse] clean-config block accuracy: {json.dumps(out['parse'])}", flush=True)

    # ---- the vocabulary each read yields, on oracle-solved repairs ----------------------
    for i, era in enumerate(ers):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np).to(device)
        xo, _ = oracle_rollout(gen, x0, r_np, shared["base_ms"], rules, rules_t, canon,
                               depth, v, m, s, budget)
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        solved = xo[torch.from_numpy(sc > 0.5).to(device)]
        cell = {"n_solved": int(solved.shape[0]), "solve_rate": float(sc.mean()), "reads": {}}
        top_span = s ** (max_macro_level - 1)
        top0 = ((era["node"] * s ** (era["level"] - 1)) // top_span) * top_span
        mb = 0 if top0 > 0 else top0 + top_span
        reads = {"unmasked": MC.parse_features(gen, solved, s=s).cpu().numpy(),
                 "mask_elsewhere": MC.parse_features(gen, solved, s=s,
                                                     mask_block=mb).cpu().numpy(),
                 "exact": MC.exact_features(solved.cpu().numpy(),
                                            shared["inverse_maps"][-1], v, s)}
        for name, pf in reads.items():
            row = {}
            chain, obs = {1: MC.base_table(v)}, {}
            for ell in range(2, max_macro_level + 1):
                sp = s ** (ell - 1)
                nd = (era["node"] * s ** (era["level"] - 1)) // sp
                mn = MC.Miner(ell, s)
                seg = pf[:, nd * sp:(nd + 1) * sp]
                mn.observe(seg[(seg >= 0).all(1)])
                chain[ell] = mn.build(chain[ell - 1], cfg["mine_support"])
                obs[ell] = len(mn.counts)
            for ell in range(2, max_macro_level + 1):
                tbl = chain[ell]
                row[str(ell)] = {"n_entries": int(tbl["child"].shape[0]),
                                 "n_distinct_obs": obs[ell],
                                 **MC.grade_table(tbl, shared["truth"][ell])}
            cell["reads"][name] = row
        out["cells"][era["name"]] = cell
        print(f"\n=== era {era['name']} (oracle solve rate {cell['solve_rate']:.3f}, "
              f"{cell['n_solved']} solved) ===")
        for name, row in cell["reads"].items():
            print(f"  {name:15s}: " + " | ".join(
                f"L{ell}: {c['n_entries']:3d} entries, prec="
                f"{(c['precision'] if c['precision'] is not None else float('nan')):.3f}, "
                f"recall={c['recall']:.3f} (true {c['n_true']})" for ell, c in row.items()))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_parse.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_parse.json")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_stale(tag: str = "cals_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
              budget: int = 4, n_ref: int = 512, g_budget: int = 58,
              holdouts: str = "0,3,5", max_macro_level: int = 3, quick: bool = False):
    """CALIBRATION 4 — the plant's frontier, and the gate that keeps it one.

    `calr_s0` measured a plant that does not move: parse/infill accuracy flat to three
    decimals over 48 cycles, the true-table audition trendless, and at gen_lr 3e-4 actively
    degrading. The cause is that the fine-tuning diet is IN DISTRIBUTION -- solved
    configurations are ordinary clean derivations, and replay 0.5 is against the same corpus,
    so there is nothing to learn. Round 1 solved the same problem for the *value* by training
    it on a distribution that excluded the contexts it would be metered on; this does it for
    the *plant*.

    `plant_holdout` withholds level-2 tuples from the setup corpus while the evaluation
    instances keep coming from the unfiltered DGP. Every era's damage requires producing a
    correct level-2 tuple, so the plant's hole and the vocabulary's hole are the same hole.

    THE GATE (and it is a gate, not a knob): on instances that REQUIRE a withheld tuple,
    bootstrap solvability must be low but NONZERO. A frontier the agent can never cross is not
    a learner regime, it is exploration-gating by another name, and a ~0 reading is a
    halt-and-report rather than something to tune around."""
    import torch
    ers = parse_eras(eras)
    hs = [int(x) for x in holdouts.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    started = time.time()
    out = {"eras": ers, "g_budget": g_budget, "holdouts": {}}

    for h in hs:
        cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, g_budget=g_budget,
                   max_macro_level=max_macro_level, plant_holdout=h)
        if quick:
            cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                       reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                       n_rt=128)
        torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
        print(f"\n########## plant_holdout = {h} ##########", flush=True)
        shared = build_shared(cfg, device)
        rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
        v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
        gen, reader, hold = shared["generator0"], shared["reader"], shared["hold"]
        inv = shared["inverse_maps"][-1]
        base_ms = shared["base_ms"]
        true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                     for ell in range(2, max_macro_level + 1)}, s, depth, device)
        sets = (("base", base_ms), ("true", true_ms))

        clean = torch.from_numpy(
            _sample_pool(rules, cfg["n_rt"], s, cfg["seed"] + 31337)[1]).to(device)
        powers = v ** torch.arange(s, device=device)
        tf = shared["bottom_map"][(clean.view(clean.shape[0], shared["n_blocks"], s)
                                   * powers).sum(-1)]
        ok = tf >= 0
        with torch.no_grad():
            rd = reader.block_logits(clean).argmax(-1)
            gp = gen.block_logits(clean).argmax(-1)
        cell0 = {"read_acc": float((rd[ok] == tf[ok]).float().mean()),
                 "generator_parse_acc": float((gp[ok] == tf[ok]).float().mean()),
                 "n_hold": len(hold), "eras": {}}
        print(f"  READ: reader={cell0['read_acc']:.4f}  "
              f"generator(unsupervised at visible positions)={cell0['generator_parse_acc']:.4f}",
              flush=True)

        for i, era in enumerate(ers):
            r_np, x_np, c_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth,
                                                 v, m, seed=cfg["seed"] + 5000 + 17 * i,
                                                 with_clean=True)
            exc = needs_excluded(c_np, era, hold, inv, v, s)
            x0 = torch.from_numpy(x_np)
            cell = {"frac_excluded": float(exc.mean()), "policies": {}}
            for tag_, mset in sets:
                w = fit_width(len(mset), budget, g_budget)
                b = beam_moves(shared["controller"], gen, shared["value0"], x0,
                               torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                               budget=budget, beam_width=w, device=device)
                sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                cell["policies"][tag_] = {
                    "w": w, "g": b["counts"]["ground"] / x0.shape[0],
                    "e_all": 1.0 - float(sc.mean()),
                    "solve_excluded": float(sc[exc].mean()) if exc.any() else None,
                    "solve_rest": float(sc[~exc].mean()) if (~exc).any() else None}
            # the vocabulary the READER now yields, from exact-DP-solved repairs
            xo, _ = oracle_rollout(gen, x0.to(device), r_np, base_ms, rules, rules_t, canon,
                                   depth, v, m, s, budget)
            so, _ = grade(xo.cpu().numpy(), r_np, rules, s)
            solved = xo[torch.from_numpy(so > 0.5).to(device)]
            cell["oracle_solve_rate"] = float(so.mean())
            cell["vocab"] = {}
            if solved.shape[0]:
                pf = MC.parse_features(reader, solved, s=s).cpu().numpy()
                chain = {1: MC.base_table(v)}
                for ell in range(2, max_macro_level + 1):
                    sp = s ** (ell - 1)
                    nd = (era["node"] * s ** (era["level"] - 1)) // sp
                    mn = MC.Miner(ell, s)
                    seg = pf[:, nd * sp:(nd + 1) * sp]
                    mn.observe(seg[(seg >= 0).all(1)])
                    chain[ell] = mn.build(chain[ell - 1], cfg["mine_support"])
                    cell["vocab"][str(ell)] = {
                        "n_entries": int(chain[ell]["child"].shape[0]),
                        **MC.grade_table(chain[ell], shared["truth"][ell])}
            cell0["eras"][era["name"]] = cell
            pol = cell["policies"]
            print(f"  era {era['name']}: frac_excluded={cell['frac_excluded']:.3f}  "
                  f"oracle_solve={cell['oracle_solve_rate']:.3f}")
            for tag_ in ("base", "true"):
                p = pol[tag_]
                se = p["solve_excluded"]
                print(f"    {tag_:5s} w{p['w']} ({p['g']:.0f}g): e_all={p['e_all']:.3f}  "
                      f"SOLVE|excluded={'n/a' if se is None else f'{se:.3f}'}  "
                      f"solve|rest={p['solve_rest']:.3f}")
            for ell, c in cell["vocab"].items():
                print(f"    vocab L{ell}: {c['n_entries']} entries prec="
                      f"{(c['precision'] if c['precision'] is not None else float('nan')):.3f} "
                      f"recall={c['recall']:.3f}")
        out["holdouts"][str(h)] = cell0

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_stale.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_stale.json")
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3):
    """C-M — the macro operator with the DGP's OWN table is BIT-IDENTICAL to the true level
    move (`units.apply_move`) at every level. This is what makes `given` and `practice_*` the
    same machinery differing only in which tuples are in the table, and therefore what makes
    the earned-vs-given fraction a statement about vocabulary rather than about operators.
    C-R — the ratchet constraint bites: a level-3 table built over a truncated level-2 table
    cannot contain entries whose halves were dropped.
    G-D — the nested damage cells are on-grammar at every level and leave d* > 0 often enough
    that rejection sampling is cheap."""
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
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    out = {"table_sizes": {str(k): int(t["child"].shape[0]) for k, t in truth.items()}}

    # C-M
    cm = {}
    for ell in range(2, max_level + 1):
        ok = True
        for j in range(s ** (depth - ell)):
            true_mv = {"level": ell, "node": j, "blk0": j * s ** (ell - 1),
                       "span": s ** (ell - 1), "name": f"L{ell}n{j}"}
            a = apply_move(gen, x, true_mv, rules_t, canon, depth, v, m, s)
            mac = MC.to_device(MC.make_macro(ell, j, s, truth[ell]), device)
            b = MC.apply_any(gen, x, mac, rules_t, canon, depth, v, m, s)
            ok &= bool(torch.equal(a, b))
        cm[f"L{ell}"] = ok
    out["CM_macro_equals_true_move"] = cm
    assert all(cm.values()), f"C-M failed: {cm}"

    # C-R: truncate T2, rebuild T3 over it, and check the drop
    mn = MC.Miner(3, s)
    flat3 = truth[3]["flat"]
    mn.observe(np.concatenate([flat3] * 3))
    full = mn.build(truth[2], support=1)
    trunc = MC.make_table(2, truth[2]["child"][:6], MC.base_table(v), s)
    part = mn.build(trunc, support=1)
    out["CR_ratchet"] = {"t3_over_full_t2": int(full["child"].shape[0]),
                         "t3_over_truncated_t2": int(part["child"].shape[0]),
                         "t2_full": int(truth[2]["child"].shape[0]), "t2_trunc": 6}
    assert part["child"].shape[0] < full["child"].shape[0], "C-R failed: truncation did not bite"

    # G-D
    gd = {}
    for era in parse_eras("1:6,2:3,3:1"):
        dmg = corrupt_hier(leaves_np, rules, depth, v, m, s, era["level"], [era["node"]],
                           np.random.default_rng(7))
        d = nearest_derivation_cost(rules, dmg, roots_np, s)
        gd[era["name"]] = {"on_grammar": on_grammar_rate(dmg, inv[-1], v, s),
                           "dstar": float(d.mean()), "dstar_zero": float((d == 0).mean())}
        assert gd[era["name"]]["on_grammar"] == 1.0, f"G-D failed at {era['name']}"
    out["GD_damage"] = gd

    # grounding table — what the declared budget buys each action set
    out["ground"] = {f"n{n_}_w{w}": beam_ground(n_, 4, w)
                     for n_ in (8, 12, 14) for w in (1, 2, 3, 4)}
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    return selfcheck()


# --------------------------------------------------------------------------- #
# gates for the handle itself (CPU, no training)
# --------------------------------------------------------------------------- #

def handle_selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3):
    """H-1 .. H-5. The properties the twin comparison rests on.

    H-1  an empty wrapper is BIT-IDENTICAL to the generator it wraps (max|d| == 0.0)
    H-2  minting a zero-init slot is still bit-identical -- the handle is an exact no-op at
         the commit instant, so a handle arm and its twin cannot diverge before commitment
    H-3  symbol lookup is correct: with the DGP's OWN table every span of a clean on-grammar
         configuration gets a non-zero symbol, and the symbol's table row reproduces the
         span's level-1 features exactly
    H-4  a span with any masked block gets symbol 0 (no leak of the span under repair)
    H-5  the macro loss reaches the trunk: one step gives non-zero gradient on the CORE's
         parameters, which is the whole point of the intervention"""
    import torch
    import torch.nn.functional as F
    from rhm.rhm_generative_planner import _build_generator

    out, dev = {}, torch.device("cpu")
    length, n_blocks = s ** depth, s ** depth // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inv[-1]).to(dev)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    torch.manual_seed(0)
    core = _build_generator()(v, length, s, 96, n_head=4, n_layer=2, root_conditioned=False)
    core.eval()
    _, leaves = _sample_pool(rules, n, s, 7)
    x = torch.from_numpy(leaves)

    # The ceiling on symbol coverage is a SUBSTRATE property, not the handle's: `bottom_map` is
    # last-writer-wins over bottom-level synonyms (ratchet's README calls this out as the
    # irreducible read error), so a leaf tuple two level-1 features share is read as one of
    # them. A span containing such a block gets symbol 0 even when a committed entry does cover
    # it. Every arm reads level-1 features through the same map, so this is shared, not a
    # treatment effect -- but it caps H3 coverage below 1.0 and is recorded here.
    bot = rules[depth - 1]
    lc = (bot * (v ** np.arange(s))).sum(-1)
    out["H0_bottom_ambiguous_slots"] = int(sum(
        1 for f in range(v) for r in range(m) if int(bottom_map[int(lc[f, r])]) != f))
    out["H0_bottom_slots"] = int(v * m)

    g = HN.wrap(copy.deepcopy(core), v=v, s=s, n_blocks=n_blocks, bottom_map=bottom_map)
    g.eval()
    with torch.no_grad():
        a, b = core.block_logits(x), g.block_logits(x)
    out["H1_empty_wrapper_maxabs"] = float((a - b).abs().max())

    for ell in range(2, max_level + 1):
        out[f"H_slot_{ell}"] = g.add_slot(ell, truth[ell])
    with torch.no_grad():
        c = g.block_logits(x)
    out["H2_zero_init_maxabs"] = float((c - a).abs().max())

    powers = v ** torch.arange(s)
    feats = bottom_map[(x.view(n, n_blocks, s) * powers).sum(-1)]
    for ell in range(2, max_level + 1):
        key = str(ell)
        span = g.luts[key]["span"]
        ids = g.ids_from_obs(x, key)
        out[f"H3_L{ell}_frac_covered"] = float((ids > 0).float().mean())
        flat = torch.from_numpy(np.asarray(truth[ell]["flat"], np.int64))
        rows = flat[(ids - 1).clamp(min=0)]                     # (n, n_spans, span)
        want = feats.view(n, n_blocks // span, span)
        out[f"H3_L{ell}_roundtrip"] = float(
            ((rows == want).all(-1) | (ids == 0)).float().mean())
        out[f"H3_L{ell}_target_agrees"] = float(
            (g.targets_from_feats(feats, key) == ids).float().mean())
        # H-4: mask block 0 -> the span containing it must be symbol 0
        obs = x.clone()
        obs[:, 0:s] = -1
        out[f"H4_L{ell}_masked_span_zero"] = float((g.ids_from_obs(obs, key)[:, 0] == 0).all())

    # H-5: does the macro loss reach the trunk?
    for p in g.parameters():
        p.requires_grad_(True)
    g.train()
    obs = x.clone()
    obs[:, 0:s] = -1
    hid = g.hidden(obs)
    loss = 0.0
    for key in g.levels():
        span = g.luts[key]["span"]
        tgt = g.targets_from_feats(feats, key)
        loss = loss + F.cross_entropy(g.macro_logits(key, hidden=hid)[:, 0], tgt[:, 0])
    loss.backward()
    out["H5_macro_head_grad"] = float(
        sum(float(p.grad.abs().sum()) for p in g.slots["2"].parameters() if p.grad is not None))
    out["H5_core_grad_step0"] = float(
        sum(float(p.grad.abs().sum()) for p in g.core.parameters() if p.grad is not None))
    # after one optimizer step the head is non-zero, so the trunk starts receiving gradient
    opt = torch.optim.AdamW(g.parameters(), lr=1e-4)
    opt.step()
    g.zero_grad(set_to_none=True)
    hid = g.hidden(obs)
    loss = 0.0
    for key in g.levels():
        tgt = g.targets_from_feats(feats, key)
        loss = loss + F.cross_entropy(g.macro_logits(key, hidden=hid)[:, 0], tgt[:, 0])
    loss.backward()
    out["H5_core_grad_step1"] = float(
        sum(float(p.grad.abs().sum()) for p in g.core.parameters() if p.grad is not None))
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def handle_selfcheck_remote():
    return handle_selfcheck()


@app.local_entrypoint()
def main(quick: bool = True):
    handle_run.remote(quick=quick)
