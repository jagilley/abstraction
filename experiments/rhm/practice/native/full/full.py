"""full — PHASE 2 of `native`: the two ports COMPOSED, plus the ablation battery.

FORK NOTICE, and the choice behind it. This file forks `../prop/prop.py` (Port 1), which is
itself a verbatim fork of `../../ratchet/ratchet.py`, and grafts Port 2 onto it by IMPORTING
`../span/span_net.py` — nothing is copied from `span/`, and `ratchet/`, `teacher_slot/`,
`prop/` and `span/` are all left untouched. The alternative was to re-fork `ratchet.py` and
import both nets; forking `prop.py` was chosen because Port 1's insertions are the larger and
more delicate of the two (a rewritten beam with a cached-encoder-state contract and a fidelity
gate that has already been verified bit-for-bit in-tag), while Port 2's are a clean seam — one
executor object threaded through the beam's materialisation call. Re-deriving Port 1 by hand
would have put the already-gated half at risk to save nothing.

WHAT IS NEW relative to `prop.py`:

 * The executor seam. `beam_moves` and `beam_moves_prop` both materialise through an executor
   object (`span_net.PlainExecutor` / `SpanExecutor`) instead of calling `macros.apply_any`
   directly. A `PlainExecutor` call IS `apply_any` (span/'s gate S-3), so an untreated arm is
   op-for-op ratchet and Port 1's beam is op-for-op `prop.py`'s.
 * `finetune_generator_span` — the plant loss plus the span head's self-imitation term in the
   same optimizer steps, with a `lam == 0` short-circuit that restores the original graph
   exactly (this is what makes the composed fidelity arm a real gate).
 * The NATIVE arms: proposal head (k = 4, Port 1's measured sweet spot) AND span head on the
   same vocabulary — `given_native`, `practice_late_native`.
 * The POISON PAIR: `practice_early` (a frozen 1-entry bogus level-2 table, exogenously
   enumerated) against `practice_early_prop_k4` (the same bad table, natively ROUTED).
 * `ablation_battery` — an end-of-run probe battery on the final trained state, per arm:
   (a) full system, (b) table ablated with macro calls forced through the span head,
   (b') table and span reach both removed, (c) primitives ablated. Plus, under each, the
   next-level mining yield split into its STRUCTURAL component (entries `Miner.build` can
   assemble over the surviving lower table) and its INFORMATIVE one (distinct level-1 tuples
   observed at support in the chosen trajectories, which no lower table gates).

The fork's fidelity is not assumed: the untreated arms are re-run IN THIS TAG as the twins,
and `given_fid` (k = n_moves, span loss off, parity gate nailed shut) is carried as an in-tag
assertion that the composed loop reproduces enumeration bit-for-bit for 90 cycles.

WHAT IS NEW (everything else is ratchet's, unchanged):

 * `prop_net.ProposalHead` — pi(move | z, r*) beside the value, on the SAME encoder read
   (`controller.state(x)` plus the root target), over a fixed (level, node) slot vocabulary.
 * `beam_moves_prop` — at plan time only the top-k proposed moves per tip are materialised and
   scored, instead of all `n_moves`. Everything else about the beam is `beam_moves` verbatim.
   The kept children's encoder states are CACHED and reused as the next step's proposal read,
   because the value already paid for them when it scored those children.
 * `prop_train` — self-imitation on the beam's own chosen trajectories: (state, r*) -> the move
   the surviving-and-SOLVED tip actually took, masked cross-entropy over the live action set.
   Selection happens before the regression, which is the whole reason this is not "gradient
   descent averaging over everything the agent did".
 * Arms `*_prop_k{1,2,4,N}` over three vocabularies (base / true / mined), and the readouts the
   spec commits Port 1 to: the can't-decompose signature, the effective-branching ledger, and
   the next-level currency under native vs exogenous routing.

WHAT IS PRICED, SAID OUT LOUD. A grounding in this substrate is "score one state" — an encoder
pass plus a value read (`beam_moves` charges one per materialised child). The proposal head
reads the SAME z. For every tip below the root that z was already computed and charged when the
value scored that state as a child, so the port caches it and the ledger charges nothing; the
head-only forward on a cached vector is counted separately as `n_prop` and priced at `c_prop`
(default 0.0, with a sensitivity column at c_mat = 0.05 in the reduction). The ROOT state is
the one honest exception — it has never been anybody's child — so the port pays one full
grounding per instance per solve for it, and `fit_width_k` sees that cost when it picks the
width. The freed groundings are REINVESTED, not banked: each arm runs the widest beam that fits
the same declared G = 58 under ITS effective branching factor, which is exactly ratchet's
matched-pricing idiom with n_moves -> k. The banked reading (same width, fewer groundings) is
recoverable from the width ladder logged at every probe.

Run from experiments/:
  modal run rhm/practice/native/prop/prop.py::prop_run --quick --tag smoke0
  python3 rhm/practice/native/prop/launch_detached.py --fn prop_run --tag np_s0 ...

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
  modal run rhm/practice/native/full/full.py::full_selfcheck_remote
  modal run rhm/practice/native/full/full.py::full_run --quick --tag smoke0
  python3 rhm/practice/native/full/launch_detached.py --fn full_run --tag nf_s0 ...

(ratchet's own run lines: `rhm/practice/ratchet/ratchet.py::ratchet`.)
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
from rhm.practice.native.prop import prop_net as PN
from rhm.practice.native.span import span_net as SN


app = modal.App("rhm-practice-native-full", image=image)
REMOTE = "rhm_practice_native_full"


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
               *, budget, beam_width, device, collect=False, ex=None):
    """ratchet's beam, with ONE change (span/'s): every materialisation goes through an executor
    object. `PlainExecutor.apply` IS `macros.apply_any` (span/'s gate S-3), so this is op-for-op
    ratchet for every untreated arm."""
    import torch
    if ex is None:
        ex = SN.PlainExecutor()
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
            children = [ex.apply(generator, flat, ms[k], rules_t, canon, depth, v, m, s)
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
# THE PORT: the same beam, but only the top-k PROPOSED moves per tip are expanded
# --------------------------------------------------------------------------- #

def beam_moves_prop(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
                    *, budget, beam_width, device, collect=False,
                    prop, slots, k, forced=(), n_slots, explore=0, erng=None, ex=None):
    """`beam_moves` with the enumeration replaced by a proposal.

    Structurally line-for-line the original: same children ordering (tip-major, move-minor),
    same `topk` over value scores, same parent/seq bookkeeping, same final re-score. The only
    changes are (i) which children exist at all, and (ii) that the kept children's encoder
    states are carried forward instead of being recomputed for the proposal read.

    `slots` maps position-in-`ms` -> slot id, so the head's output vocabulary is stable across
    an action set that grows at a commit. `forced` is a list of positions in `ms` that are
    expanded unconditionally (a move too new for the head to have a training example on).

    FIDELITY. At k >= len(ms) with `forced` empty, `select_moves` returns `arange(n_moves)` for
    every tip, `expand_selected`'s row gather is the identity, and the assembled child tensor,
    the scores, the topk and the sequences are the enumerated ones bit-for-bit. The only
    difference is the extra root encode, which is charged and touches nothing else."""
    import torch
    if ex is None:
        ex = SN.PlainExecutor()
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval(); prop.eval()
        x0 = x0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        n_moves = len(ms)
        slot_t = torch.as_tensor(slots, dtype=torch.long, device=device)      # (n_moves,)
        beams = x0[:, None, :]
        seqs = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        counts = {"mat": 0, "ground": 0, "prop": 0}
        hist_x = [beams.clone()] if collect else None
        hist_par = [] if collect else None

        # the ROOT read: the one encoder pass the port genuinely adds, charged in full.
        z_tip = _encode_chunked(controller, x0)                               # (batch, D)
        counts["ground"] += batch

        for _ in range(budget):
            flat = beams.reshape(batch * width, length)
            zf = z_tip.reshape(batch * width, -1)
            rf = roots.repeat_interleave(width)
            plog = prop(zf, rf)[:, slot_t]                                    # -> `ms` order
            counts["prop"] += batch * width
            sel = PN.select_moves(plog, k, forced=list(forced))               # (N, k)
            if explore and sel.shape[1] < n_moves:
                # NB: not named `ex` — that is the executor parameter now.
                exp_idx = PN.explore_moves(sel, n_moves, explore, erng, device)
                sel = torch.sort(torch.cat([sel, exp_idx], dim=1), dim=1).values
            kk = sel.shape[1]
            child, n_mat = PN.expand_selected(generator, flat, sel, ms, ex.apply,
                                              rules_t, canon, depth, v, m, s)
            cand = child.reshape(batch, width * kk, length)
            counts["mat"] += n_mat
            tgt = roots.repeat_interleave(width * kk)
            z_all = _encode_chunked(controller, cand.reshape(-1, length))
            scores = value(z_all, tgt).reshape(batch, width * kk)
            counts["ground"] += batch * width * kk
            keep = min(beam_width, width * kk)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, kk, rounding_mode="floor")
            col = top % kk
            mv = sel.reshape(batch, width, kk).gather(
                1, parent[:, :, None].expand(-1, -1, kk)).gather(2, col[:, :, None]).squeeze(2)
            beams = cand.gather(1, top[:, :, None].expand(-1, -1, length))
            z_tip = z_all.reshape(batch, width * kk, -1).gather(
                1, top[:, :, None].expand(-1, -1, z_all.shape[-1]))
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


def plan(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
         *, budget, beam_width, device, collect=False, port=None, ex=None):
    """One entry point for every place the agent plans, so Port 1 is either everywhere the beam
    runs or nowhere; `ex` is Port 2's executor, threaded the same way. `port=None` with a plain
    executor is ratchet's `beam_moves`, byte-for-byte."""
    if port is None:
        return beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth,
                          v, m, s, budget=budget, beam_width=beam_width, device=device,
                          collect=collect, ex=ex)
    return beam_moves_prop(controller, generator, value, x0, roots, ms, rules_t, canon, depth,
                           v, m, s, budget=budget, beam_width=beam_width, device=device,
                           collect=collect, ex=ex, **port)


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


def finetune_generator_span(generator, head, ex, slots, opt, new_leaves, replay_leaves,
                            bottom_map, *, v, s, n_blocks, n_steps, batch, replay_frac, device,
                            rng, span_rng, span_batch, lam=1.0):
    """THE PLANT LEARNS, WITH A CORRIDOR HEAD (`../span/span.py`'s, verbatim apart from the
    `lam == 0` short-circuit). `finetune_generator` op-for-op — same batches, same number of
    optimizer steps, same masking draws, same level-1 loss — plus the span head's self-imitation
    cross-entropy at weight `lam`, averaged over minted slots, in the SAME steps.

    `lam == 0` skips the term entirely rather than adding `0 * term`. Adding a zeroed term would
    be numerically a no-op but not GRAPH-identical, and the composed fidelity arm (`given_fid`)
    is an assertion about graph identity, not about tolerances."""
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    if head is not None:
        head.train()
    last, slast, sacc = 0.0, None, {}
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
        if head is not None and slots and lam:
            sterm, sacc = SN.span_train_terms(generator, head, ex, slots, s, span_batch,
                                              span_rng, device)
            if sterm is not None:
                loss = loss + lam * sterm
                slast = float(sterm.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        if head is not None and lam:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    if head is not None:
        head.eval()
    return last, slast, sacc


def push(buf, x, r, y, cap):
    import torch
    buf["x"] = torch.cat([buf["x"], x])[-cap:]
    buf["r"] = torch.cat([buf["r"], r])[-cap:]
    buf["y"] = torch.cat([buf["y"], y])[-cap:]


# --------------------------------------------------------------------------- #
# THE PORT'S LEARNING: self-imitation of the beam's own chosen trajectories
# --------------------------------------------------------------------------- #

def prop_pairs(out, roots, succ, slots, budget, train_on="solved"):
    """The beam's own chosen trajectories, as (state, root-index, slot) supervision.

    `beam_moves(collect=True)` already hands these back: `traj[t]` is the per-step state of
    every SURVIVING trajectory (ancestry resolved back from the tips) and `tips_seq[:, :, t]`
    is the move that trajectory took at step t. `succ` is those tips' terminal possible-set
    success, so `train_on="solved"` regresses only onto trajectories that WORKED — the
    selection step that makes this self-imitation rather than averaging over everything the
    agent did.

    Move indices are converted to SLOTS here, at collection time, because a position in `ms`
    stops meaning the same thing the moment the action set grows."""
    import torch
    traj, seq = out["traj"], out["tips_seq"]
    B, W = seq.shape[0], seq.shape[1]
    length = traj[0].shape[2]
    slot_t = torch.as_tensor(slots, dtype=torch.long, device=seq.device)
    keep = torch.as_tensor(succ > 0.5, device=seq.device).reshape(B, W) \
        if train_on == "solved" else torch.ones(B, W, dtype=torch.bool, device=seq.device)
    if not bool(keep.any()):
        return None
    rr = roots.to(seq.device)[:, None].expand(B, W)
    xs, as_, rs = [], [], []
    for t in range(budget):
        xs.append(traj[t][keep])                       # (n_keep, T)
        as_.append(slot_t[seq[:, :, t][keep]])         # (n_keep,)
        rs.append(rr[keep])
    return {"x": torch.cat(xs).cpu(), "a": torch.cat(as_).cpu(),
            "r": torch.cat(rs).cpu(), "n_traj": int(keep.sum())}


def prop_train(prop, opt, controller, pbuf, avail_mask, *, n_steps, batch, device, rng):
    """Masked cross-entropy of pi(. | z, r*) against the move the selected trajectory took.

    The mask is the CURRENT action set: slots that do not exist are not competitors and are
    never targets. Buffered pairs from before a commit are replayed under the current mask —
    the mask only ever gains options and the stored target stays legal."""
    import torch
    import torch.nn.functional as F
    n = pbuf["x"].shape[0]
    if n == 0:
        return {"loss": None, "acc": None, "n": 0}
    prop.train()
    last, acc = 0.0, 0.0
    neg = torch.finfo(torch.float32).min / 4
    for _ in range(n_steps):
        i = torch.from_numpy(rng.integers(0, n, size=min(batch, n)))
        x = pbuf["x"][i].to(device)
        r = pbuf["r"][i].to(device)
        a = pbuf["a"][i].to(device)
        with torch.no_grad():
            z = controller.state(x)
        logits = prop(z, r).masked_fill(~avail_mask[None, :], neg)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        acc = float((logits.argmax(-1) == a).float().mean())
    prop.eval()
    return {"loss": last, "acc": acc, "n": int(n)}


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
    # --- ratchet's five, verbatim: the untreated references / twins ------------------------
    "never_base":     {"vocab": "base",   "commit": None,    "prop_k": None, "span": False},
    "given":          {"vocab": "true",   "commit": None,    "prop_k": None, "span": False},
    "practice_gated": {"vocab": "earned", "commit": "delta", "prop_k": None, "span": False},
    "practice_early": {"vocab": "earned", "commit": "early", "prop_k": None, "span": False},
    "practice_late":  {"vocab": "earned", "commit": "late",  "prop_k": None, "span": False},
    # --- Port 1 alone (phase 1's arms, kept so a composed arm can be read against both
    #     single-port results in the same tag if wanted) ---------------------------------
    "given_prop_k4":         {"vocab": "true",   "commit": None,    "prop_k": 4, "span": False},
    "practice_late_prop_k4": {"vocab": "earned", "commit": "late",  "prop_k": 4, "span": False},
    # --- THE POISON TWIN. `practice_early` freezes a 1-entry bogus level-2 table at c1;
    #     ratchet measured that this forecloses level 3's REPRESENTATION. The routed version
    #     asks the SPEC's question: does a natively-routed bad L2 foreclose worse, because the
    #     policy now calls the address instead of merely having it available? Routing only —
    #     the corridor port is not the variable here.
    "practice_early_prop_k4": {"vocab": "earned", "commit": "early", "prop_k": 4, "span": False},
    # --- Port 2 alone -----------------------------------------------------------------------
    "span_true":  {"vocab": "true",   "commit": None,   "prop_k": None, "span": True},
    "span_mined": {"vocab": "earned", "commit": "late", "prop_k": None, "span": True},
    # --- THE FULL NATIVE PRIMITIVE: proposed as one call (Port 1) and run as one pass
    #     (Port 2). k = 4 is Port 1's measured sweet spot — exactly budget-matched to `given`'s
    #     enumeration at G = 58 (57 groundings per solve either way).
    "given_native":         {"vocab": "true",   "commit": None,   "prop_k": 4, "span": True},
    "practice_late_native": {"vocab": "earned", "commit": "late", "prop_k": 4, "span": True},
    # --- THE COMPOSED FIDELITY ARM. Both ports present and wired, both nailed shut: the
    #     proposal filter at k = n_moves (an exact no-op) and the span loss off with the parity
    #     gate unreachable. Must reproduce `given` bit-for-bit for 90 cycles, which is what
    #     licenses reading the composed file's other arms at all.
    "given_fid": {"vocab": "true", "commit": None, "prop_k": -1, "span": True,
                  "cfg": {"span_lam": 0.0, "span_tau": 2.0}},
}

# A treated arm and its untreated twin must share a torch RNG stream, or the contrast is
# confounded by stream position (`handle/`'s discipline). Streams are keyed by the TWIN.
TWIN = {
    "given_prop_k4": "given", "span_true": "given", "given_native": "given",
    "given_fid": "given",
    "practice_late_prop_k4": "practice_late", "span_mined": "practice_late",
    "practice_late_native": "practice_late",
    "practice_early_prop_k4": "practice_early",
}
STREAM = {"never_base": 0, "given": 1, "practice_gated": 2, "practice_early": 3,
          "practice_late": 4}


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
    cfg = {**cfg, **spec.get("cfg", {}), **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]

    # ---- PER-ARM torch STREAM. ratchet let the global stream run across arms, so an arm's
    #      trajectory depended on the arms before it. Keyed by the TWIN (see `TWIN`/`STREAM`),
    #      so a treated arm and its untreated baseline draw the identical sequence and can only
    #      diverge through the port itself. -------------------------------------------------
    stream_key = TWIN.get(base, base)
    torch.manual_seed(cfg["train_seed"] * 1000 + 7 + STREAM[stream_key])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["train_seed"] * 1000 + 7 + STREAM[stream_key])

    # ---- per-arm plant + selector (the plant is a substrate property this round: every arm
    #      carries it, and the value co-adapts against it) ----------------------------------
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

    # ---- PORT 2: the span head, its slots, and the executor (`../span/span.py`'s block,
    #      verbatim). The head is minted off its OWN generator, so its existence costs the
    #      shared stream nothing; its slots start CLOSED, so it cannot fire before parity.
    use_span = bool(spec.get("span"))
    head, slots = None, {}
    slot_events, gate_events = [], []
    span_rng = np.random.default_rng(cfg["seed"] + 20_250_820)     # its OWN stream
    if use_span:
        head = SN.build_head(SN.slot_count(s, depth, maxl), v, cfg["state_dim"],
                             s ** (maxl - 1), cfg["seed"] * 100 + 13, device,
                             hidden_mult=cfg["span_hidden_mult"])
        for p in head.parameters():
            p.requires_grad_(True)
        gopt.add_param_group({"params": list(head.parameters()),
                              "lr": cfg["span_lr"] or cfg["gen_lr"], "weight_decay": 1e-4})
        ex = SN.SpanExecutor(generator, head, slots, cap=cfg["span_buf_cap"],
                             hold_cap=cfg["span_hold_cap"], hold_frac=cfg["span_hold_frac"],
                             per_call=cfg["span_capture"], seed=cfg["seed"] + 5_150_101,
                             v=v, length=shared["length"])
    else:
        ex = SN.PlainExecutor()

    def mint(level, table):
        if not use_span:
            return
        n_nodes = s ** (depth - level)
        for j in range(n_nodes):
            key = SN.slot_key(level, j)
            slots[key] = {"id": SN.slot_index(level, j, s, depth, maxl), "open": False,
                          "move": None, "level": int(level), "node": int(j)}
        slot_events.append({"level": int(level), "n_nodes": int(n_nodes),
                            "n_entries": int(table["child"].shape[0]),
                            "span": int(s ** (level - 1))})
        print(f"[slot]   arm={arm} level={level} nodes={n_nodes} "
              f"entries={table['child'].shape[0]} span={s ** (level - 1)}", flush=True)

    def bind_slots(ms_now):
        """Point every slot at its CURRENT macro move object (the table changes on commit)."""
        for mv in ms_now:
            if mv.get("kind") != "macro":
                continue
            key = SN.slot_key(mv["level"], mv["node"])
            if key in slots:
                slots[key]["move"] = mv

    # ---- vocabulary state ---------------------------------------------------------------
    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
            mint(ell, committed[ell])
    ms = build_ms(base_ms, committed, s, depth, device)
    bind_slots(ms)
    p_width = cfg["pr_width"]

    # ---- THE PORT ------------------------------------------------------------------------
    #      The head is built inside `isolated_rng`, so it consumes none of the shared stream:
    #      a treated arm is bit-identical to its twin until the filter first switches on.
    prop_k_spec = spec.get("prop_k")
    if overrides.get("prop_k") is not None:
        prop_k_spec = int(overrides["prop_k"])
    offsets, n_slots = PN.slot_layout(depth, s, maxl)
    prop = popt = None
    prng = np.random.default_rng(cfg["seed"] + 4207)
    erng = np.random.default_rng(cfg["seed"] + 8821)
    pbuf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
            "r": torch.zeros(0, dtype=torch.long), "a": torch.zeros(0, dtype=torch.long)}
    move_age = {}                       # slot -> cycles this move has been in the action set
    state = {"filter_on": False}
    if prop_k_spec is not None:
        prop = PN.build_head(cfg["state_dim"], v, n_slots,
                             cfg["seed"] * 7919 + 13, device)
        popt = torch.optim.AdamW(prop.parameters(), lr=cfg["prop_lr"], weight_decay=1e-4)
        for slot in PN.move_slots(ms, offsets):
            move_age[slot] = 0

    def port_spec():
        """The `plan` port for the CURRENT action set, and its effective branching factor.
        Returns `(None, n_moves)` while the filter is off — during the warmup the head trains
        on the beam's trajectories but never gates, so the arm is still its twin exactly."""
        if prop is None or not state["filter_on"]:
            return None, len(ms)
        slots = PN.move_slots(ms, offsets)
        k = len(ms) if prop_k_spec < 0 else int(prop_k_spec)
        forced = tuple(i for i, sl in enumerate(slots)
                       if move_age.get(sl, 0) < cfg["prop_new_cycles"])
        k_eff = min(min(k, len(ms)), len(ms) - len(forced)) + len(forced)
        return ({"prop": prop, "slots": slots, "k": k, "forced": forced,
                 "n_slots": n_slots}, k_eff)

    def avail_mask():
        import torch as _t
        mask = _t.zeros(n_slots, dtype=_t.bool, device=device)
        for sl in PN.move_slots(ms, offsets):
            mask[sl] = True
        return mask

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
           "probe": [], "vocab": [], "prop": [], "span": [], "blocks": [],
           "m_per_solve": []}

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
            counts = {"mat": 0, "ground": 0, "prop": 0}
            ex.reset()
            # THE FILTER SWITCHES ON after `prop_warmup` cycles of the run. Until then the head
            # trains on the beam's trajectories but never gates, so (i) the arm is bit-identical
            # to its twin through the warmup and (ii) the filter never gates from a random init.
            if prop is not None and cyc > cfg["prop_warmup"]:
                state["filter_on"] = True
            port, k_eff = port_spec()
            # PRACTICE EXPLORES, PERFORMANCE DOES NOT — the substrate's own convention
            # (`collect_value_buffer` runs its behaviour policy at explore_eps = 0.3). Without
            # it the filtered beam is a closed loop: a move the head does not propose never
            # appears in a solved trajectory and so never becomes a training target.
            port_pr = port
            if port is not None and cfg["prop_explore"]:
                port_pr = dict(port, explore=int(cfg["prop_explore"]), erng=erng)

            # --- (a) practice: wide closed-loop beam on fresh instances of this era's cell
            r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                                           seed=cfg["seed"] + 100_000 + 1000 * cyc)
            out = plan(controller, generator, value, torch.from_numpy(x_np),
                       torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                       budget=cfg["budget"], beam_width=p_width, device=device,
                       collect=True, port=port_pr, ex=ex)
            for key in counts:
                counts[key] += out["counts"].get(key, 0)
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
            sloss, sacc = None, {}
            if use_span:
                gloss, sloss, sacc = finetune_generator_span(
                    generator, head, ex, slots, gopt, solved.cpu(), shared["replay"]["x"],
                    shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                    n_steps=cfg["gen_steps"], batch=cfg["batch_size"],
                    replay_frac=cfg["replay_frac"], device=device, rng=grng,
                    span_rng=span_rng, span_batch=cfg["span_batch"], lam=cfg["span_lam"])
            else:
                gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                           shared["bottom_map"], v=v, s=s,
                                           n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                           batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                           device=device, rng=grng)

            # --- (c2) THE PARITY GATE (span/'s, verbatim). Held-out exact-match against what
            #         the DP would write with the CURRENT trunk, per macro, re-checked every
            #         cycle rather than latched, so a slot the moving plant drifts away from
            #         closes again. Instrument cost only; never priced.
            par = {}
            if use_span and slots:
                par = SN.parity(generator, head, ex, slots, s, v,
                                min_hold=cfg["span_min_hold"])
                for key, cell in par.items():
                    was = slots[key]["open"]
                    now = cell["exact"] is not None and cell["exact"] >= cfg["span_tau"]
                    slots[key]["open"] = now
                    if now != was:
                        gate_events.append({"cycle": cyc, "slot": key, "open": bool(now),
                                            "exact": cell["exact"], "n": cell["n"]})
                        print(f"[gate]   arm={arm} c{cyc} slot {key} "
                              f"{'OPEN' if now else 'CLOSE'} exact={cell['exact']} "
                              f"n={cell['n']}", flush=True)
            vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                replay_frac=cfg["replay_frac"], device=device, rng=rng)

            # --- (c') THE PORT LEARNS: self-imitation of the beam's own chosen trajectories,
            #          restricted to the survivors that actually SOLVED. Its data comes from
            #          the same beam it gates, so the imitation is on-policy from the cycle the
            #          filter switches on; its RNG is its own, so the shared stream is
            #          undisturbed and the twin gate stays licensed.
            pinfo, fallback = {"k": None}, False
            if prop is not None:
                pairs = prop_pairs(out, torch.from_numpy(r_np), succ,
                                   PN.move_slots(ms, offsets), cfg["budget"],
                                   train_on=cfg["prop_train_on"])
                if pairs is None and cfg["prop_train_on"] == "solved":
                    # nothing solved this cycle: a policy that cannot generate a success cannot
                    # be improved by imitating an empty set, so fall back to the beam's own
                    # survivors (which the value did select) rather than skipping the update
                    # and letting a bad filter freeze itself in.
                    pairs = prop_pairs(out, torch.from_numpy(r_np), succ,
                                       PN.move_slots(ms, offsets), cfg["budget"],
                                       train_on="tips")
                    fallback = True
                if pairs is not None:
                    n_cap = cfg["prop_buf_cap"]
                    pbuf["x"] = torch.cat([pbuf["x"], pairs["x"]])[-n_cap:]
                    pbuf["r"] = torch.cat([pbuf["r"], pairs["r"]])[-n_cap:]
                    pbuf["a"] = torch.cat([pbuf["a"], pairs["a"]])[-n_cap:]
                pinfo = prop_train(prop, popt, controller, pbuf, avail_mask(),
                                   n_steps=cfg["prop_steps"], batch=cfg["prop_batch"],
                                   device=device, rng=prng)
                pinfo.update({"k": (len(ms) if prop_k_spec < 0 else int(prop_k_spec)),
                              "k_eff": int(k_eff), "filter_on": bool(state["filter_on"]),
                              "n_pairs": (0 if pairs is None else int(pairs["x"].shape[0])),
                              "fallback": bool(fallback)})

            # --- (d) metering under PERFORMANCE conditions (the declared grounding budget) -
            #         The width is refit to the SAME declared budget under this arm's own
            #         effective branching factor: ratchet's matched-pricing idiom with
            #         n_moves -> k. The groundings the port frees are reinvested in width, not
            #         banked; the banked reading is the width ladder logged at every probe.
            mr_np, mx = meter[era_i]
            port, k_eff = port_spec()
            width = (fit_width(len(ms), cfg["budget"], cfg["g_budget"]) if port is None
                     else PN.fit_width_k(k_eff, cfg["budget"], cfg["g_budget"]))
            b = plan(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                     rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                     beam_width=width, device=device, port=port, ex=ex)
            for key in counts:
                counts[key] += b["counts"].get(key, 0)
            # snapshot the block-level tally NOW: the probes below are unpriced instruments
            # and must not enter the counterfactual ledger any more than the real one.
            blocks_cycle = dict(ex.counts)
            msucc, mdres = grade(b["x"].cpu().numpy(), mr_np, rules, s)
            e = 1.0 - float(msucc.mean())
            gps = b["counts"]["ground"] / mx.shape[0]
            meter_cost = {"g": gps, "mat": b["counts"]["mat"] / mx.shape[0],
                          "prop": b["counts"].get("prop", 0) / mx.shape[0],
                          "w": int(width), "k_eff": int(k_eff)}

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
                ms = build_ms(base_ms, committed, s, depth, device)
                mint(active, tbl)              # the corridor becomes the head's responsibility
                bind_slots(ms)
                if prop is not None:
                    # the head has never had a training example on the new slots. Their output
                    # rows are still at their zero init, so they enter the ranking neutrally
                    # (above whatever the head learned to reject, below what it learned to
                    # accept) — and for `prop_new_cycles` cycles they are expanded regardless,
                    # priced at their true cost, so an untrained logit cannot lock them out.
                    for slot in PN.move_slots(ms, offsets):
                        move_age.setdefault(slot, -1)
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
                # the action set may have grown in (g) this very cycle, so the port is rebuilt
                # here rather than reused: a stale `slots` list would silently make the probe
                # blind to the moves just committed.
                pport, pk = port_spec()
                probe = {"cycle": cyc, "era": era_i + 1, "ladder": {}, "all_eras": {},
                         "k_eff": int(pk), "n_moves": len(ms)}
                for w in cfg["probe_widths"]:
                    bb = plan(controller, generator, value, mx, torch.from_numpy(mr_np),
                              ms, rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=w, device=device, port=pport, ex=ex)
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["ladder"][str(w)] = {
                        "e": 1.0 - float(sc.mean()),
                        "g": bb["counts"]["ground"] / mx.shape[0],
                        "mat": bb["counts"]["mat"] / mx.shape[0],
                        "prop": bb["counts"].get("prop", 0) / mx.shape[0]}
                for j in range(len(eras)):
                    jr, jx = meter[j]
                    wj = (fit_width(len(ms), cfg["budget"], cfg["g_budget"]) if pport is None
                          else PN.fit_width_k(pk, cfg["budget"], cfg["g_budget"]))
                    bb = plan(controller, generator, value, jx, torch.from_numpy(jr), ms,
                              rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=wj, device=device, port=pport, ex=ex)
                    sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                    probe["all_eras"][str(j)] = 1.0 - float(sc.mean())
                probe["plant"] = plant_probe(generator, shared, cfg, device)
                # THE SPAN-HEAD-BYPASSED TWIN of the competence readout (span/'s): the same
                # metering beam with the head switched off, so what the port's FIRING costs or
                # buys is separable from what its gradient did to the trunk.
                if use_span and any(sl["open"] for sl in slots.values()):
                    wn = (fit_width(len(ms), cfg["budget"], cfg["g_budget"]) if pport is None
                          else PN.fit_width_k(pk, cfg["budget"], cfg["g_budget"]))
                    ex.fire = False; ex.capture = False
                    bb = plan(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                              rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=wn, device=device, port=pport, ex=ex)
                    ex.fire = True; ex.capture = True
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["plant"]["e_nospan"] = 1.0 - float(sc.mean())
                # THE CAN'T-DECOMPOSE SIGNATURE, on the era's own metering roots: where pi puts
                # its mass, and — on the states where it proposes a MACRO — how much mass it
                # keeps on that macro's own primitive decomposition. An expert cannot decompose
                # its chunks. Unpriced instrument.
                if prop is not None:
                    with torch.no_grad():
                        zp = _encode_chunked(controller, mx.to(device))
                    rp = torch.from_numpy(mr_np).to(device)
                    am = avail_mask()
                    probe["pi"] = PN.decompose_probe(prop, zp, rp, ms, offsets, s, maxl, am)
                    bm = PN.base_mass_probe(prop, zp, rp, offsets, am)
                    probe["pi"].update({"top1": bm["top1"], "entropy": bm["entropy"]})
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
            log["vocab"].append({str(ell): (None if committed[ell] is None
                                            else int(committed[ell]["child"].shape[0]))
                                 for ell in range(2, maxl + 1)})
            pinfo["n_prop"] = int(counts.get("prop", 0))
            pinfo["n_ground"] = int(counts["ground"]); pinfo["n_mat"] = int(counts["mat"])
            pinfo["meter"] = meter_cost          # the PRICED beam's own cost per instance
            log["prop"].append(pinfo)
            log["m_per_solve"].append(b["counts"]["mat"] / mx.shape[0])
            log["blocks"].append(blocks_cycle)
            tb, hb = ex.sizes() if use_span else ({}, {})
            log["span"].append({
                "parity": {k: cell["exact"] for k, cell in par.items()},
                "parity_block": {k: cell["block"] for k, cell in par.items()},
                "n_hold": {k: cell["n"] for k, cell in par.items()},
                "open": {k: bool(sl["open"]) for k, sl in slots.items()},
                "n_open": int(sum(1 for sl in slots.values() if sl["open"])),
                "buf": tb, "hold": hb, "sloss": sloss, "sacc": sacc})
            for slot in list(move_age):
                move_age[slot] += 1
            a_act = aud.get(str(active), {})
            print(f"[c{cyc:3d}] arm={arm:20s} era{era_i+1} t={t_cum:10.0f} e={e:.4f} "
                  f"w={width} nm={len(ms)} k={pinfo.get('k')}/{pinfo.get('k_eff')} "
                  f"solved={solved.shape[0]:4d} "
                  f"A{active}={a_act.get('cand')} true={a_act.get('true')} "
                  f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
                  f"pacc={pinfo.get('acc')} sil={cert['run']}", flush=True)
            if cyc % cfg["checkpoint_every"] == 0:
                write_results(outdir, arm, cfg, eras, refs, log, events, complete=False,
                              extra={"prop_k": prop_k_spec, "span_mode": use_span,
                                     "twin": TWIN.get(base), "slot_events": slot_events,
                                     "gate_events": gate_events})

    # ------------------------------------------------------------------ #
    # THE ABLATION BATTERY — on the FINAL trained state, after the 90 cycles.
    # ------------------------------------------------------------------ #
    # Run as an end-of-run probe rather than a mid-run switch, so the trajectory every other
    # readout is measured on is never corrupted by the ablation. Unpriced (it is an instrument),
    # but each condition's own per-solve cost is recorded so the conditions are comparable.
    #
    #   a_full   the system as it ran
    #   b_table  THE TABLE IS DELETED. The macro stays in the action set — the policy can still
    #            call the address — but nothing may consult the table at execution time, so
    #            every macro call is served by the span head. POLICY, ON THE RECORD: under this
    #            condition the head serves ALL macro calls regardless of the parity gate, so a
    #            parity shortfall shows up as error rather than being hidden behind a fallback
    #            to the very DP path the condition is supposed to have removed. Arms with no
    #            span head have no execution path at all and are reported as `b_span`.
    #   b_span   table AND span reach removed: pure enumeration over base moves.
    #   c_prims  the primitives are deleted instead; macros only.
    #
    # Next-level currency is split into the two components the prediction confounds:
    #   built        entries `Miner.build` can assemble over the surviving lower table. Under
    #                `b_table` this is 0 BY CONSTRUCTION (T3 subset T2 x T2), so it is the
    #                STRUCTURAL half and carries no information on its own.
    #   at_support   distinct level-1 tuples observed at least `mine_support` times in the
    #                chosen trajectories, keyed by the flat tuple and gated by no table at all.
    #                This is the INFORMATIVE half: does the observation stream of chunk-shaped
    #                spans survive when the routing is native?
    def battery():
        macro_ms = [mv for mv in ms if mv.get("kind") == "macro"]
        base_only = [mv for mv in ms if mv.get("kind") != "macro"]
        conds = {"a_full": (ms, "normal"), "b_table": (ms, "force_head"),
                 "b_span": (base_only, "plain"), "c_prims": (macro_ms, "normal")}
        saved = {k: sl["open"] for k, sl in slots.items()}
        if use_span:
            ex.capture = False
        out = {}
        for cname, (ms_c, mode) in conds.items():
            if not ms_c:
                out[cname] = None
                continue
            if mode == "force_head" and not use_span:
                out[cname] = {"skipped": "no span head — no execution path; see b_span"}
                continue
            if use_span:
                for k_ in slots:
                    slots[k_]["open"] = (mode == "force_head") or (
                        mode == "normal" and saved[k_])
                ex.fire = mode != "plain"
            cell = {"n_moves": len(ms_c), "eras": {}}
            for j, era_j in enumerate(eras):
                jr, jx = meter[j]
                slots_c = PN.move_slots(ms_c, offsets) if prop is not None else None
                if prop is not None:
                    k_c = len(ms_c) if prop_k_spec < 0 else min(int(prop_k_spec), len(ms_c))
                    port_c = {"prop": prop, "slots": slots_c, "k": k_c, "forced": (),
                              "n_slots": n_slots}
                    w_c = PN.fit_width_k(k_c, cfg["budget"], cfg["g_budget"])
                else:
                    port_c, w_c = None, fit_width(len(ms_c), cfg["budget"], cfg["g_budget"])
                ex.reset()
                bb = plan(controller, generator, value, jx, torch.from_numpy(jr), ms_c,
                          rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                          beam_width=w_c, device=device, port=port_c, ex=ex)
                sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                n = jx.shape[0]
                row = {"e": 1.0 - float(sc.mean()), "w": int(w_c),
                       "g_solve": bb["counts"]["ground"] / n,
                       "mat_solve": bb["counts"]["mat"] / n,
                       "blocks": {kk: vv / n for kk, vv in ex.counts.items()}}
                # next-level currency, mined from THIS condition's own chosen trajectories
                keep = bb["x"][torch.from_numpy(sc > 0.5).to(device)]
                mn = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
                if keep.shape[0]:
                    pf = MC.parse_features(shared["reader"], keep, s=s).cpu().numpy()
                    for ell in range(2, maxl + 1):
                        span_l = s ** (ell - 1)
                        node_l = (era_j["node"] * s ** (era_j["level"] - 1)) // span_l
                        mn[ell].observe(pf[:, node_l * span_l:(node_l + 1) * span_l])
                row["n_solved"] = int(keep.shape[0])
                row["mine"] = {}
                for ell in range(2, maxl + 1):
                    st = mn[ell].state()
                    lower = MC.base_table(v) if ell == 2 else operative(ell - 1)
                    built = mn[ell].build(lower, cfg["mine_support"])["child"].shape[0]
                    row["mine"][str(ell)] = {
                        "at_support": int(st["n_at_support"][str(cfg["mine_support"])]),
                        "n_distinct": int(st["n_distinct"]), "n_obs": int(st["n_obs"]),
                        "built": int(built)}
                cell["eras"][str(j)] = row
                print(f"[ablate] arm={arm} {cname:8s} era{j + 1} nm={len(ms_c)} w={w_c} "
                      f"e={row['e']:.4f} g={row['g_solve']:.1f} solved={row['n_solved']} "
                      f"T2@sup={row['mine']['2']['at_support']} "
                      f"T3@sup={row['mine'].get('3', {}).get('at_support')} "
                      f"T3built={row['mine'].get('3', {}).get('built')}", flush=True)
            out[cname] = cell
        if use_span:
            for k_ in slots:
                slots[k_]["open"] = saved[k_]
            ex.fire = True
            ex.capture = True
        out["_parity_at_end"] = (log["span"][-1]["parity"] if log["span"] else {})
        out["_open_at_end"] = saved
        return out

    print(f"\n----- arm={arm} ABLATION BATTERY (final trained state) -----", flush=True)
    ablation = battery()

    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True,
                  extra={"prop_k": prop_k_spec, "span_mode": use_span,
                         "twin": TWIN.get(base), "slot_events": slot_events,
                         "gate_events": gate_events, "ablation": ablation})
    return {"log": log, "events": events, "ablation": ablation}


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
    """ratchet's ledger, plus the proposal reads at `c_prop`. `c_prop` defaults to 0.0: every
    proposal read below the root is a head-only forward on an encoder state the value already
    paid a grounding for, and the one read that is NOT (the root) is charged a full grounding
    inside the beam. The raw count is logged either way so the reduction can re-price it."""
    return (counts["ground"] * cfg["d_fb"] + counts["mat"] * cfg["c_mat"]
            + counts.get("prop", 0) * cfg.get("c_prop", 0.0))


def write_results(outdir, arm, cfg, eras, refs, log, events, complete, extra=None):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": eras, "refs": refs,
                   "log": log, "events": events, "complete": complete, **(extra or {})},
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
        d_fb=1.0, c_mat=0.05, checkpoint_every=5, probe_widths=(1, 2, 4, 8, 16), probe_every=4,
        # --- the port ---------------------------------------------------------------------
        prop_lr=3e-4, prop_steps=24, prop_batch=256, prop_buf_cap=60_000,
        prop_warmup=4, prop_new_cycles=2, prop_explore=1, prop_train_on="solved", c_prop=0.0,
        # --- the port 2 knobs (`../span/`'s, unchanged) ------------------------------------
        span_lam=1.0, span_lr=1e-3, span_batch=64, span_buf_cap=8192, span_hold_cap=2048,
        span_hold_frac=0.1, span_capture=24, span_min_hold=256, span_tau=0.95,
        span_hidden_mult=4,
    )
    cfg.update(kw)
    return cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=32768)
def full_run(
    tag: str = "smoke",
    arms: str = ("given,given_native,given_fid,practice_late,practice_late_native,"
                 "practice_early,practice_early_prop_k4"),
    eras: str = "1:6,2:3,3:1", seed: int = 0, era_cycles: int = 24, budget: int = 4,
    pr_width: int = 16, g_budget: int = 58, max_macro_level: int = 3,
    n_pr: int = 64, n_rt: int = 384, n_score: int = 512, n_grad: int = 4,
    gen_lr: float = 1e-4, gen_steps: int = 20, mine_support: int = 3,
    mine_from: str = "chosen", mine_cap: int = 0,
    value_lr_online: float = 3e-5, sil_c: float = 0.06, sil_cv: float = 0.10,
    sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6, lp_min_drop: float = 0.05,
    early_offset: int = 1, late_offset: int = 2, probe_every: int = 4,
    plant_holdout: int = 0, holdout_seed: int = 11,
    d_fb: float = 1.0, c_mat: float = 0.05, c_prop: float = 0.0,
    prop_lr: float = 3e-4, prop_steps: int = 24, prop_batch: int = 256,
    prop_warmup: int = 4, prop_new_cycles: int = 2, prop_explore: int = 1,
    prop_train_on: str = "solved",
    span_lam: float = 1.0, span_lr: float = 1e-3, span_batch: int = 64,
    span_tau: float = 0.95, span_min_hold: int = 256, span_capture: int = 24,
    quick: bool = False,
):
    import torch
    cfg = _cfg(span_lam=span_lam, span_lr=span_lr, span_batch=span_batch, span_tau=span_tau,
               span_min_hold=span_min_hold, span_capture=span_capture,
               c_prop=c_prop, prop_lr=prop_lr, prop_steps=prop_steps, prop_batch=prop_batch,
               prop_warmup=prop_warmup, prop_new_cycles=prop_new_cycles,
               prop_explore=prop_explore, prop_train_on=prop_train_on,
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
                   sil_min_cycle=2, early_offset=1, late_offset=1, gen_steps=5,
                   prop_warmup=1, prop_steps=12, prop_buf_cap=20_000,
                   # exercise the FIRING path in the smoke: no real run reaches parity in five
                   # cycles, so the smoke opens the gate on presence of held-out data.
                   span_min_hold=32, span_tau=0.0)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"native/full tag={tag} arms={arms} eras={ers} device={device}")

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
# gates for the PORT itself (no training; the fidelity gate's other half)
# --------------------------------------------------------------------------- #

def full_selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=64, max_level=3, budget=3):
    """P-1 .. P-7. The properties the fidelity and twin gates rest on.

    P-1  `select_moves` at k = n_moves returns `arange(n_moves)` for EVERY row, including rows
         of exact ties (a zero-init head gives exactly that) — the tie-ordering hazard the spec
         names, closed by a stable sort rather than by `topk`.
    P-2  `expand_selected` at k = n_moves is BIT-IDENTICAL to the enumerated
         `stack([apply_any(flat, ms[j]) for j], 1)`.
    P-3  `beam_moves_prop` at k = n_moves reproduces `beam_moves` bit-for-bit — same final
         states, same tips, same move sequences, same materialisation and grounding counts
         apart from the one root encode the port pays for and declares.
    P-4  building the head consumes NONE of the global torch stream (the twin gate's
         precondition): the random draw after `build_head` equals the draw without it.
    P-5  the head is exactly uniform at init (zero-init output layer), so a freshly committed
         macro enters the ranking neutrally rather than at an arbitrary trained logit.
    P-6  `select_moves` at k < n_moves returns exactly the k highest-logit moves, ascending,
         and with `forced` returns k + |forced| including every forced index.
    P-7  the (level, node) slot layout is injective over the full 14-move action set, and the
         pricing table `fit_width_k` the k ladder rests on (printed, not asserted).

    And the COMPOSITION's own gates, which are what this phase adds:

    C-1  `PlainExecutor` is `macros.apply_any` on every move, and a non-firing `SpanExecutor`
         is too — so threading an executor through the beam changes nothing by itself.
    C-2  THE COMPOSED FIDELITY GATE: the proposal beam at k = n_moves with a span executor
         present but closed reproduces the enumerating beam bit-for-bit (x, seq, tips, traj,
         materialisation count), the only delta being the declared root encode.
    C-3  an OPEN span slot rewrites exactly the macro's own token positions and nothing else,
         through the composed beam's own executor.
    C-4  minting BOTH heads leaves the shared torch stream exactly where it was — the twin
         gate's precondition, now for two new RNG consumers rather than one."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    n_blocks = length // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device)
    ctrl = _build_rich_controller()(v, length, s, 96, n_head=4, n_layer=2).to(device)
    val = _build_value_head()(96, v).to(device)
    for mod in (gen, ctrl, val):
        mod.eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(_corrupt(leaves_np, n_blocks, 3, v, s, rng)).to(device)
    roots = torch.from_numpy(roots_np).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    offsets, n_slots = PN.slot_layout(depth, s, max_level)
    slots = PN.move_slots(ms, offsets)
    out = {"n_moves": len(ms), "n_slots": n_slots, "slots": slots}

    # P-7 (layout)
    assert len(set(slots)) == len(slots) == len(ms), "slot layout is not injective"
    out["P7_slots_injective"] = True
    out["P7_price"] = {f"k{k}": {"w": PN.fit_width_k(k, 4, 58),
                                 "g": PN.beam_ground_k(k, 4, PN.fit_width_k(k, 4, 58)) + 1}
                       for k in (1, 2, 3, 4, 8, 12, 14)}
    out["P7_price_enum"] = {f"n{n_}": {"w": fit_width(n_, 4, 58),
                                       "g": beam_ground(n_, 4, fit_width(n_, 4, 58))}
                            for n_ in (8, 12, 14)}

    # P-1 / P-6 (selection)
    nm = len(ms)
    ties = torch.zeros(n, nm, device=device)
    rnd = torch.randn(n, nm, device=device)
    ar = torch.arange(nm, device=device)[None, :].expand(n, -1)
    out["P1_ties_k_eq_n"] = bool(torch.equal(PN.select_moves(ties, nm), ar))
    out["P1_rand_k_eq_n"] = bool(torch.equal(PN.select_moves(rnd, nm), ar))
    assert out["P1_ties_k_eq_n"] and out["P1_rand_k_eq_n"], "P-1 failed"
    sel4 = PN.select_moves(rnd, 4)
    want = torch.sort(torch.topk(rnd, 4, dim=1).indices, dim=1).values
    out["P6_topk_matches"] = bool(torch.equal(sel4, want))
    selF = PN.select_moves(rnd, 2, forced=[8, 9])
    out["P6_forced_shape"] = list(selF.shape)
    out["P6_forced_contains"] = bool(((selF == 8).any(1) & (selF == 9).any(1)).all())
    assert out["P6_topk_matches"] and out["P6_forced_contains"], "P-6 failed"
    assert selF.shape[1] == 4, "P-6: forced expansion is not k + |forced| wide"

    # P-2 (expansion)
    flat = x
    enum_children = torch.stack(
        [MC.apply_any(gen, flat, ms[j], rules_t, canon, depth, v, m, s) for j in range(nm)],
        dim=1)
    sel_all = PN.select_moves(torch.zeros(flat.shape[0], nm, device=device), nm)
    got, n_mat = PN.expand_selected(gen, flat, sel_all, ms, MC.apply_any, rules_t, canon,
                                    depth, v, m, s)
    out["P2_expand_maxabs"] = float((got.long() - enum_children.long()).abs().max())
    out["P2_n_mat"] = int(n_mat)
    assert out["P2_expand_maxabs"] == 0.0, "P-2 failed"

    # P-4 / P-5 (the head costs no shared randomness, and is uniform at init)
    torch.manual_seed(1234)
    ref = torch.randn(8)
    torch.manual_seed(1234)
    head = PN.build_head(96, v, n_slots, seed=99, device=device)
    got_draw = torch.randn(8)
    out["P4_stream_untouched"] = bool(torch.equal(ref, got_draw))
    assert out["P4_stream_untouched"], "P-4 failed: the head consumed the shared stream"
    with torch.no_grad():
        z = _encode_chunked(ctrl, x)
        lg = head(z, roots)
    out["P5_init_logit_maxabs"] = float(lg.abs().max())
    assert out["P5_init_logit_maxabs"] == 0.0, "P-5 failed"

    # P-3 (the beam)
    a = beam_moves(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                   budget=budget, beam_width=4, device=device, collect=True)
    b = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                        budget=budget, beam_width=4, device=device, collect=True,
                        prop=head, slots=slots, k=nm, forced=(), n_slots=n_slots)
    out["P3_x_equal"] = bool(torch.equal(a["x"], b["x"]))
    out["P3_seq_equal"] = bool(torch.equal(a["seq"], b["seq"]))
    out["P3_tips_equal"] = bool(torch.equal(a["tips_x"], b["tips_x"]))
    out["P3_traj_equal"] = bool(all(torch.equal(p, q) for p, q in zip(a["traj"], b["traj"])))
    out["P3_counts_enum"] = a["counts"]
    out["P3_counts_prop"] = b["counts"]
    out["P3_extra_ground"] = int(b["counts"]["ground"] - a["counts"]["ground"])
    out["P3_mat_equal"] = bool(a["counts"]["mat"] == b["counts"]["mat"])
    assert all(out[k_] for k_ in ("P3_x_equal", "P3_seq_equal", "P3_tips_equal",
                                  "P3_traj_equal", "P3_mat_equal")), "P-3 failed"
    assert out["P3_extra_ground"] == x.shape[0], "P-3: the root encode is mispriced"
    # ---- C-4: BOTH heads cost the shared stream nothing --------------------------------
    torch.manual_seed(4321)
    ref2 = torch.randn(6)
    torch.manual_seed(4321)
    head1 = PN.build_head(96, v, n_slots, seed=7, device=device)
    shead = SN.build_head(SN.slot_count(s, depth, max_level), v, 96, s ** (max_level - 1),
                          seed=11, device=device)
    out["C4_stream_untouched"] = bool(torch.equal(ref2, torch.randn(6)))
    assert out["C4_stream_untouched"], "C-4 failed: a head consumed the shared stream"

    # ---- C-1: the executor seam is a no-op by itself ------------------------------------
    macros_l = [mv for mv in ms if mv.get("kind") == "macro"]
    slots_c = {SN.slot_key(mv["level"], mv["node"]):
               {"id": SN.slot_index(mv["level"], mv["node"], s, depth, max_level),
                "open": False, "move": mv, "level": mv["level"], "node": mv["node"]}
               for mv in macros_l}
    plain = SN.PlainExecutor()
    sx = SN.SpanExecutor(gen, shead, slots_c, cap=4096, hold_cap=1024, hold_frac=0.1,
                         per_call=32, seed=3, v=v, length=length)
    worst = 0.0
    for mv in ms:
        with torch.no_grad():
            a_ = MC.apply_any(gen, x, mv, rules_t, canon, depth, v, m, s)
            b_ = plain.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
            c_ = sx.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
        worst = max(worst, float((a_ - b_).abs().max()), float((a_ - c_).abs().max()))
    out["C1_executor_seam_maxabs"] = worst
    assert worst == 0.0, "C-1 failed: the executor seam is not a no-op"

    # ---- C-2: THE COMPOSED FIDELITY GATE ------------------------------------------------
    sx.capture = False
    e_ = beam_moves(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                    budget=budget, beam_width=4, device=device, collect=True,
                    ex=SN.PlainExecutor())
    f_ = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                         budget=budget, beam_width=4, device=device, collect=True,
                         prop=head1, slots=slots, k=nm, forced=(), n_slots=n_slots, ex=sx)
    out["C2_x_equal"] = bool(torch.equal(e_["x"], f_["x"]))
    out["C2_seq_equal"] = bool(torch.equal(e_["seq"], f_["seq"]))
    out["C2_tips_equal"] = bool(torch.equal(e_["tips_x"], f_["tips_x"]))
    out["C2_traj_equal"] = bool(all(torch.equal(p_, q_) for p_, q_ in zip(e_["traj"], f_["traj"])))
    out["C2_mat_equal"] = bool(e_["counts"]["mat"] == f_["counts"]["mat"])
    out["C2_extra_ground"] = int(f_["counts"]["ground"] - e_["counts"]["ground"])
    assert all(out[k_] for k_ in ("C2_x_equal", "C2_seq_equal", "C2_tips_equal",
                                  "C2_traj_equal", "C2_mat_equal")), "C-2 failed"
    assert out["C2_extra_ground"] == x.shape[0], "C-2: the root encode is mispriced"

    # ---- C-3: an OPEN slot writes only its own span, through the composed beam ----------
    for k_ in slots_c:
        slots_c[k_]["open"] = True
    sx.fire = True
    c3 = {}
    for mv in macros_l:
        pos = SN.span_positions(mv, x.shape[0], s, device)
        with torch.no_grad():
            got = sx.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
        keepm = torch.ones_like(x, dtype=torch.bool).scatter_(1, pos, False)
        c3[mv["name"]] = {"outside_maxabs": float((got - x)[keepm].abs().max()),
                          "inside_changed": float((got != x).gather(1, pos).float().mean())}
    out["C3_open_slot"] = c3
    out["C3_mat_head"] = int(sx.counts["mat_head"])
    assert all(cc["outside_maxabs"] == 0.0 for cc in c3.values()), "C-3 failed"
    assert out["C3_mat_head"] > 0, "C-3: the head never fired"

    # ---- C-5: THE PRACTICE-PATH CALL SHAPE ----------------------------------------------
    # C-2 exercises the composed beam only at k = n_moves with `forced` and `explore` empty,
    # which is the METERING path. The practice beam runs a different branch — k < n_moves,
    # exploration on — and that branch is where a local name collided with the executor
    # parameter and threw at the first materialisation of the second step. A gate that only
    # covers the metering shape is not a gate on the composition, so it is widened here.
    sx.fire = False
    sx.capture = True
    c5 = {}
    for kk_, exp_, forced_ in ((2, 1, ()), (2, 1, (8, 9)), (4, 0, ()), (1, 1, ())):
        sx.reset()
        g_ = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                             budget=budget, beam_width=4, device=device, collect=True,
                             prop=head1, slots=slots, k=kk_, forced=forced_, n_slots=n_slots,
                             explore=exp_, erng=np.random.default_rng(7), ex=sx)
        tot = sum(sx.counts[q] for q in ("mat_base", "mat_dp", "mat_head"))
        c5[f"k{kk_}_e{exp_}_f{len(forced_)}"] = {
            "x_shape": list(g_["x"].shape), "mat": int(g_["counts"]["mat"]),
            "executor_mat": int(tot), "agree": bool(tot == g_["counts"]["mat"]),
            "seq_width": int(g_["seq"].shape[0])}
        assert list(g_["x"].shape) == list(x.shape), "C-5: wrong output shape"
        assert tot == g_["counts"]["mat"], "C-5: beam and executor disagree on materialisations"
        assert len(g_["traj"]) == budget + 1, "C-5: trajectory collection broke"
    out["C5_practice_shapes"] = c5
    sx.fire = True

    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, gpu="L4", timeout=1800, memory=16384)
def full_selfcheck_remote():
    return full_selfcheck()


@app.local_entrypoint()
def main(quick: bool = True):
    full_run.remote(quick=quick)
