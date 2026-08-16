"""THE EVALUATION LAYER, RE-INSTANTIATED ONE LEVEL UP.

The ratchet round earned a level-indexed vocabulary but graded it with an evaluation layer
that never climbed with it. Its audition was:

    apply the candidate macro ONCE, to held-out instances of the CURRENT era's damage,
    and score terminal success.

Two things are wrong with that, and only the first was named at the time:

 1. WRONG CONTEXTS. The unit is consumed on era-(k+1) damage; it was auditioned on era-k
    damage. (`ratchet`'s finding 6.)

 2. WRONG EVALUATOR -- and this is the larger of the two on this substrate. A macro's
    one-action audition MASKS ITS OWN SPAN and predicts from the surrounding context, which
    is clean in every era. So the one-action score barely depends on the damage level at all:
    the true level-3 macro scores 0.495 / 0.497 / 0.513 on era-1 / era-2 / era-3 damage
    (`rr_s0`'s own `refs.macro_true`). Meanwhile the unit is actually consumed INSIDE a beam
    of `budget` moves at the declared grounding budget, alongside the base moves -- where the
    same vocabulary takes era-3 error from 0.642 to 0.347. Audition and consumption are not
    the same measurement, and the audition's whole dynamic range (0.648 -> 0.537 for
    `practice_late`'s level-3 series) is smaller than the gap between them.

So a climbed grader has TWO coordinates, and this module supplies both:

    ctx  in {era_k, mfg, real}      -- which damage distribution the test is drawn from
    ev   in {action, policy}        -- one macro application, or the agent's own performance
                                       policy WITH the candidate in its action set

`(era_k, action)` is the ratchet's grader. `(real, policy)` is the fully climbed one -- the
teacher's, since manufacturing real level-(k+1) damage needs the oracle. `(mfg, policy)` is
the one practice can build for itself: the agent damages ITS OWN clean derivations at level
k+1, using its own level-k vocabulary as the damage source and its own reader to see what it
is overwriting, then runs its own policy on them. Nothing in `manufacture_damage` touches
`rules`; the oracle enters only through `context_stats`, which is an instrument.

WHY THE MANUFACTURE IS ONE LEVEL DOWN. To damage a level-l node the agent replaces each of
its s level-(l-1) children with a DIFFERENT entry of its level-(l-1) table. Using the
level-(l-1) vocabulary rather than the level-l one matters twice: it is available from cycle 1
(at l=2 the table is just the v level-1 features, which are native), and it keeps the test
from being drawn out of the very table being auditioned.
"""

import numpy as np

from rhm.rhm_sculpt_precheck import nearest_derivation_cost, possible_sets
from rhm.practice.crystallize.units import grade, on_grammar_rate


# --------------------------------------------------------------------------- #
# (1) self-manufactured audition contexts
# --------------------------------------------------------------------------- #

def manufacture_damage(x_clean, feats, lower, level, node, s, canon_np, rng,
                       bottom_np=None, render="canon"):
    """Damage clean configurations at the level-`level` node, natively.

    x_clean : (N, T) int64 -- the agent's OWN clean derivations (solved beam answers)
    feats   : (N, n_blocks) -- level-1 features as the agent READS them (its own reader)
    lower   : the table the damage is drawn from. TWO sources are admissible, and which one is
              faithful is a measurement (`cal_mfg`), not an assumption:
                * a level-(level-1) table -- each of the s children of `node` is independently
                  overwritten. Available from cycle 1 (at level 2 it is just the v level-1
                  features, which are native) and drawn from a DIFFERENT table than the one
                  being auditioned, but the resulting level-`level` chunk need not be a legal
                  derivation of anything, which real `corrupt_hier` damage always is.
                * a level-`level` table -- the whole node is overwritten with one entry, so the
                  damage is a legal chunk whenever the entry is, at the cost of needing the
                  agent to have catalogued level-`level` chunks already.
              Dispatched on `lower["flat"]`'s span; anything else is an error.
    node    : the level-`level` node to damage

    Each part is overwritten with a uniformly chosen entry that DIFFERS from what the agent
    reads there. Returns (x_damaged, per-part fraction that had an alternative available), or
    (None, None) if the table is too small to damage with.

    `render="canon"` writes the canonical leaf tuple of each level-1 feature -- the only
    rendering the agent already performs anywhere (`units.apply_move` uses `canon`).
    `render="rand"` picks a uniform synonym, which requires handing over the bottom rule
    table's other rows; it exists so the calibration can price that handover rather than
    assume it.
    """
    x_clean = np.asarray(x_clean, np.int64)
    feats = np.asarray(feats, np.int64)
    n = x_clean.shape[0]
    span = s ** (level - 1)                         # blocks under the level-`level` node
    flat = np.asarray(lower["flat"], np.int64)      # (E, span_part)
    if flat.shape[1] == span:
        parts = [(node * span, span)]                             # whole-node replacement
    elif span >= s and flat.shape[1] == span // s:
        parts = [((node * s + i) * (span // s), span // s) for i in range(s)]
    else:
        raise ValueError(f"damage table spans {flat.shape[1]} blocks; a level-{level} node "
                         f"needs {span} (whole node) or {span // s if span >= s else '-'} "
                         f"(per child)")
    n_entries = flat.shape[0]
    if n_entries < 2:
        return None, None
    out = x_clean.copy()
    changed = []
    for c0, span_lo in parts:
        cur = feats[:, c0:c0 + span_lo]                              # (N, span_lo)
        match = (flat[None, :, :] == cur[:, None, :]).all(-1)        # (N, E)
        w = (~match).astype(np.float64)
        tot = w.sum(1)
        pick = np.zeros(n, np.int64)
        ok = tot > 0
        if ok.any():
            p = np.cumsum(w[ok] / tot[ok][:, None], axis=1)
            u = rng.random((int(ok.sum()), 1))
            pick[ok] = np.clip((p < u).sum(1), 0, n_entries - 1)
        chunk = flat[pick]                                           # (N, span_lo)
        if render == "rand" and bottom_np is not None:
            r = rng.integers(0, bottom_np.shape[1], size=chunk.shape)
            tup = bottom_np[chunk, r]                                # (N, span_lo, s)
        else:
            tup = canon_np[chunk]
        pos = (np.arange(c0, c0 + span_lo)[None, :, None] * s
               + np.arange(s)[None, None, :])
        pos = np.broadcast_to(pos, (n, span_lo, s)).reshape(n, -1)
        np.put_along_axis(out, pos, tup.reshape(n, -1), axis=1)
        changed.append(float(ok.mean()))
    return out, changed


def context_stats(x_dmg, roots, x_clean, rules, inverse_bottom, v, s, level, node):
    """THE ORACLE CHECK on a manufactured context distribution. Instrument only.

    on_grammar   -- fraction of blocks that are legal leaf tuples (real damage: 1.000)
    frac_broken  -- fraction with d* > 0 (real damage: 1.000, by rejection sampling)
    node_empty   -- fraction where the damaged level-`level` node derives NOTHING, i.e. the
                    manufactured chunk is not a legal level-`level` derivation. Real
                    `corrupt_hier` damage IS a legal derivation of a wrong feature, so this is
                    0.000 there; a positive reading is the manufactured distribution being
                    mis-levelled DOWNWARD, i.e. locally suspicious and repairable block-wise.
    node_overlap -- fraction where the damaged node can still derive something the clean node
                    could: the damage did not change the level-`level` content. Real: 0.000.
    """
    x_dmg = np.asarray(x_dmg, np.int64)
    d = nearest_derivation_cost(rules, x_dmg, np.asarray(roots), s)
    ps_d = possible_sets(rules, x_dmg, s)[level - 1][:, node, :]
    out = {"on_grammar": on_grammar_rate(x_dmg, inverse_bottom, v, s),
           "d_mean": float(d.mean()), "d_sd": float(d.std()),
           "frac_broken": float((d > 0).mean()),
           "node_empty": float((~ps_d.any(-1)).mean())}
    if x_clean is not None:
        ps_c = possible_sets(rules, np.asarray(x_clean, np.int64), s)[level - 1][:, node, :]
        out["node_overlap"] = float((ps_c & ps_d).any(-1).mean())
    return out


# --------------------------------------------------------------------------- #
# (2) the policy evaluator: grade a candidate the way it will be consumed
# --------------------------------------------------------------------------- #

def policy_e(controller, generator, value, x, roots_np, ms, rules, rules_t, canon,
             depth, v, m, s, *, budget, g_budget, device, beam_moves, fit_width):
    """The agent's OWN performance policy, on `x`, with `ms` as the action set: the widest
    beam that fits the declared per-solve grounding budget, scored by terminal success.

    This is the same measurement the metering step makes -- which is the point. A climbed
    grader asks "what would my metered competence be if I committed this?", not "what does one
    application of it achieve in isolation"."""
    import torch
    roots_np = np.asarray(roots_np)
    w = fit_width(len(ms), budget, g_budget)
    b = beam_moves(controller, generator, value, x, torch.from_numpy(roots_np),
                   ms, rules_t, canon, depth, v, m, s,
                   budget=budget, beam_width=w, device=device)
    succ, dres = grade(b["x"].cpu().numpy(), roots_np, rules, s)
    return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean()), "w": w,
            "g": b["counts"]["ground"] / x.shape[0], "counts": b["counts"], "x": b["x"]}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def gate_manufacture(rules, inverse_maps, truth, canon_np, bottom_np, depth, v, m, s,
                     eras, n=256, seed=0):
    """G-N / G-S.

    G-N -- the ladder really is NESTED: the level-(k+1) node enclosing era k's damage cell IS
    era k+1's damage cell. That is what lets the agent audition at the consumption node during
    era k without being told which cell era k+1 will damage.

    G-S -- a self-manufactured audition context is on-grammar 1.000, confined to that node's
    own token span, and correctly levelled. Manufactured here from the TRUE level-k table so
    that table quality is factored out of the gate, and reported beside the real damage's own
    signature so the comparison is visible rather than asserted."""
    from rhm.practice.ratchet.ratchet import context_instances, era_ctx
    from rhm.practice.ratchet import macros as MC
    from rhm.rhm_sculpt_planner import _sample_pool

    out = {}
    rng = np.random.default_rng(seed + 17)
    for i, era in enumerate(eras[:-1]):
        level = era["level"] + 1
        span = s ** (level - 1)
        node = (era["node"] * s ** (era["level"] - 1)) // span
        nxt = eras[i + 1]
        assert nxt["level"] == level and nxt["node"] == node, (
            f"G-N failed: era {i + 1}'s enclosing level-{level} node is {node}, but era "
            f"{i + 2}'s damage cell is L{nxt['level']}n{nxt['node']} -- ladder not nested")
        roots_np, clean_np = _sample_pool(rules, n, s, seed + 991 + i)
        feats = MC.exact_features(clean_np, inverse_maps[-1], v, s)
        lower = truth[level - 1] if level - 1 >= 2 else MC.base_table(v)
        cell = {"level": level, "node": node, "next_era": nxt["name"], "nested": True}
        for render in ("canon", "rand"):
            dmg, chg = manufacture_damage(clean_np, feats, lower, level, node, s, canon_np,
                                          rng, bottom_np=bottom_np, render=render)
            st = context_stats(dmg, roots_np, clean_np, rules, inverse_maps[-1], v, s,
                               level, node)
            st["frac_child_changed"] = chg
            lo, hi = node * span * s, (node * span + span) * s
            outside = np.concatenate([np.arange(0, lo), np.arange(hi, clean_np.shape[1])])
            st["outside_untouched"] = bool((dmg[:, outside] == clean_np[:, outside]).all())
            st["inside_changed"] = float((dmg[:, lo:hi] != clean_np[:, lo:hi]).any(1).mean())
            assert st["on_grammar"] == 1.0, f"G-S failed (off-grammar) {era['name']}/{render}"
            assert st["outside_untouched"], f"G-S failed (leaked outside node) {era['name']}"
            cell[render] = st
        r_np, x_np, c_np = context_instances(rules, era_ctx(nxt), n, s, depth, v, m,
                                             seed=seed + 5000 + 17 * (i + 1), with_clean=True)
        cell["real"] = context_stats(x_np, r_np, c_np, rules, inverse_maps[-1], v, s,
                                     level, node)
        out[era["name"]] = cell
    return out


def gate_recert(truth, v, s, support=1, trunc=6):
    """G-R -- a live recert UNDOES the ratchet's cap. `ratchet`'s C-R showed that committing a
    truncated level-2 table caps the rebuildable level-3 table forever (16 -> 6 entries drops
    T3 56 -> 14). Provisional commitment's claim is that replacing the committed table in
    flight restores it. This asserts exactly that, on the tables themselves."""
    from rhm.practice.ratchet import macros as MC
    mn = MC.Miner(3, s)
    mn.observe(np.concatenate([truth[3]["flat"]] * 3))
    full2 = truth[2]
    trunc2 = MC.make_table(2, full2["child"][:trunc], MC.base_table(v), s)
    n_full = int(mn.build(full2, support)["child"].shape[0])
    n_trunc = int(mn.build(trunc2, support)["child"].shape[0])
    n_restored = int(mn.build(full2, support)["child"].shape[0])
    assert n_trunc < n_full, "G-R failed: truncation did not bite"
    assert n_restored == n_full, "G-R failed: the swap did not restore the level-3 table"
    return {"t3_over_truncated_t2": n_trunc, "t3_over_full_t2": n_full,
            "t3_after_recert_swap": n_restored, "t2_trunc": trunc,
            "t2_full": int(full2["child"].shape[0])}
