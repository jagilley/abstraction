"""[enharmonic Q2] THE ENDOGENOUS QUOTIENT: the merge op, its two licences, and their gates.

Imported by `enharmonic.py`, not forked into it — `quotient.py`'s idiom and for the same
reason: everything here except the priced audition is a pure function of the DGP, the arm's own
committed table and its own counts, so it is gate-able with no GPU and no substrate.

WHAT THE OP IS. A third outer-loop action beside commit and advance. It COARSENS the learner's
own partition of a committed level-l table — two rows found interchangeable become one class —
and the level above is then keyed by the pair of coarsened classes. Because `ClassMiner`
already keys by class pair, an endogenous merge is nothing but a `Quotient` whose class map is
LEARNED rather than supplied: `LearnedQuotient` starts SINGLETON (the class of a half IS the
half), which gate E-0/G-1 proves is `MC.Miner` entry for entry, so a learner that never merges
is `flat` exactly. That is the whole inertness argument.

THE EVIDENCE IS FORCED TRANSFER, NOT THE USE RECORD. `sizing/SIZING.md` section 4 measured the
banked use record as ANTI-informative as a criterion: the pairs it separates most are the ones
that are literally the same program (median use-share ratio 0.0064, 68% of such pairs below 1%
of the winner's mass), because `macro_features`' max-sum DP is an argmax that picks one
representative and starves its aliases. Section 5 measured the alternative exactly:
`fourwall.entry_profile` on DGP-drawn damaged instances at the level's own cell resolves the
token class and nothing else (397/397 groups bit-identical over 4,096 instances), the whole
partition of a committed book costs 20-450 gradings unweighted and 4-130 use-weighted, and the
loss margin separates a true merge (exactly 0.0000) from swallowing a narrow class into an
ambiguous one (median 0.4328). So the use record is kept as a SELECTOR — which is what the
fork's slot-resolved recorder exists for — and forced transfer is the criterion.

THE TWO LICENCES, and why the yield one needs a floor.

  yield   The merge is licensed when the NEXT LEVEL'S at-support count rises by more than that
          level's own measured dead zone, `tol_yield_{l+1}` — the same `null_abba` number the
          mirror's commit owner reads. The floor is not decoration: pooling counts is MONOTONE
          under coarsening, so a bare "the count rises (or does not fall)" test passes every
          proposal including a forced over-merge, and the arm would collapse into "do whatever
          the probe says". Requiring the rise to clear the yield series' own noise scale is the
          same teeth the thermostat has, on the same gauge, one level up. The rise is computed
          by RE-KEYING the next level's own counts under the candidate map — table-free, no
          grading, no GPU — so the licence costs nothing beyond the probe.

  ledger  `fourwall`'s `merge_audition` form, ported to this substrate: `e_keep` is the
          committed table's held-out policy error on the level's own demand, `e_merge` the same
          for the table rebuilt under the candidate map, both through the arc's own
          `audition_macro` and both PRICED into `counts["ground"]`. Licensed iff
          `e_merge <= e_keep + margin`. Deliberately NOT `panel["ledger"]`: on this substrate a
          CORRECT merge has bit-identical forced-transfer profiles, so a within-level reader may
          find it neutral rather than negative, and only a type-matched audition can express
          "this merge cost me nothing within level". Whether it does is left open.

PRICING. Every grading the probe and the audition spend goes through `counts["ground"]` like
any other, so the priced budget stays comparable across arms and only the key and the licence
move.
"""

import collections

import numpy as np

from rhm.practice.ratchet import macros as MC
from rhm.practice.fourwall import wall as W
from rhm.practice.antiphon import questions as QS


# --------------------------------------------------------------------------- #
# the learner's own class map
# --------------------------------------------------------------------------- #

class LearnedQuotient:
    """A union-find over half-tuples whose class id is the class's LEXICOGRAPHICALLY SMALLEST
    member. That representative choice is what keeps the inertness exact: with no merges the
    class of a half IS the half, so `ClassMiner`'s `sorted(self.counts)` is the donor's
    lexicographic sort on the concatenated key and `MC.Miner` is reproduced entry for entry
    (gate M-1, which is E-0/G-1 re-run through this map).

    It is the arm's own state and reads no oracle: `n_read` is counted for symmetry with
    `Quotient` and is always answered from the learner's own table."""

    mode = "endo"

    def __init__(self):
        self.rep = {}                      # (level, tuple) -> representative tuple
        self.members = collections.defaultdict(set)   # (level, rep) -> members
        self.n_read = 0
        self.n_drop = 0
        self.n_tie = 0
        self.n_merges = 0
        self.events = []

    # -- the map ----------------------------------------------------------------------- #
    def find(self, level, tup):
        return self.rep.get((int(level), tup), tup)

    def ids(self, feats, level):
        feats = np.asarray(feats, np.int64).reshape(len(feats), -1)
        self.n_read += feats.shape[0]
        return [self.find(level, tuple(int(x) for x in r)) for r in feats]

    def id_of(self, tup, level):
        return self.find(level, tuple(int(x) for x in tup))

    # -- the op ------------------------------------------------------------------------ #
    def merge(self, level, a, b):
        """Coarsen: classes `a` and `b` become one, represented by the smaller. Returns the
        surviving representative. Idempotent and order-free — `merge(l, a, b)` and
        `merge(l, b, a)` produce the same map, which is what makes a merge a fact about the
        partition rather than about the call."""
        level = int(level)
        a, b = self.find(level, tuple(a)), self.find(level, tuple(b))
        if a == b:
            return a
        keep, drop = (a, b) if a < b else (b, a)
        moving = self.members.get((level, drop), set()) | {drop}
        for t in moving:
            self.rep[(level, t)] = keep
        self.members[(level, keep)] |= moving | {keep}
        self.members.pop((level, drop), None)
        self.n_merges += 1
        return keep

    # -- undo, for a trial merge ---------------------------------------------------------- #
    def snapshot(self):
        """[en_s3] The whole map, cheaply copied, so a REFUSED group can be undone exactly.

        `en_s2b` undid a refused pair by filtering `rep` for entries pointing at the two
        classes, which is correct for one pair of singletons and wrong for anything else: a
        group of three or a merge onto an already-merged representative leaves members behind.
        Whole-partition merges evaluate several groups per proposal, each on the state the
        taken ones left, so the undo has to be exact rather than nearly right."""
        return ({k: v for k, v in self.rep.items()},
                {k: set(v) for k, v in self.members.items()},
                int(self.n_merges))

    def restore(self, snap):
        rep, members, n = snap
        self.rep = {k: v for k, v in rep.items()}
        self.members = collections.defaultdict(set, {k: set(v) for k, v in members.items()})
        self.n_merges = int(n)

    def merge_group(self, level, members):
        """M-2's op applied to a whole alias GROUP: every member into one class, represented by
        the smallest. Order-free by construction — `merge` is — so the group's own order and
        the order the probe happened to emit its pairs in cannot change the result."""
        mem = sorted(tuple(x) for x in members)
        keep = mem[0]
        for other in mem[1:]:
            keep = self.merge(level, keep, other)
        return keep

    def class_of_level(self, level):
        out = collections.defaultdict(set)
        for (lv, t), r in self.rep.items():
            if lv == int(level):
                out[r].add(t)
        return out

    def state(self):
        return {"mode": self.mode, "n_read": int(self.n_read), "n_drop": 0, "n_tie": 0,
                "n_merges": int(self.n_merges),
                "n_classes_coarsened": {str(lv): len({r for (l2, _), r in self.rep.items()
                                                      if l2 == lv})
                                        for lv in sorted({l for l, _ in self.rep})},
                "n_members_bound": len(self.rep)}


# --------------------------------------------------------------------------- #
# the evidence: forced transfer at the level's own cell
# --------------------------------------------------------------------------- #

def transfer_profile(rules, x_np, roots_np, rows, node, level, s, canon_np):
    """`fourwall.entry_profile` verbatim: an (n_rows x n_instances) EXACT success matrix, one
    grading per pair. Returns (profile, n_graded) and nothing is learned anywhere."""
    return W.entry_profile(rules, x_np, roots_np, rows, node, level, s, canon_np)


def pair_losses(P):
    """`fourwall.merge_candidates`' two-directional transfer loss, on a SHARED instance pool:
    with one pool per level there is no per-class ring, so class b's "recent demand" is exactly
    the set b repairs and the criterion reduces to `1 - |A n B| / |B|` in each direction.

    Rows that repair nothing are excluded — they carry no evidence either way, and
    `SIZING.md` section 5(d) measures them at 0-8% of a book."""
    live = [i for i in range(P.shape[0]) if P[i].any()]
    hits = P.sum(1)
    out = []
    for xi in range(len(live)):
        for yi in range(xi + 1, len(live)):
            i, j = live[xi], live[yi]
            inter = int((P[i] & P[j]).sum())
            out.append({"i": i, "j": j,
                        "loss": float(max(1.0 - inter / hits[j], 1.0 - inter / hits[i])),
                        "n_i": int(hits[i]), "n_j": int(hits[j]), "n_both": inter})
    return out, live


def alias_groups(pairs, tol):
    """[en_s3] EVERY within-tol pair, closed into alias GROUPS by M-2's partition op.

    `en_s2b` merged one pair per proposal (`merge_n_pairs = 1`) and its reduction measured what
    that leaves behind: the L2 book it probed carried 13 rows in 8 demand groups — five merge
    ops away from its own demand partition — and the arm took one of them per eight cycles.
    `sizing/SIZING.md` section 5(c) priced the WHOLE book at 20-450 gradings, i.e. the probe
    that is already being paid for, so the one-pair rule was a limit of the op and not of the
    evidence.

    The closure is transitive because M-2's op is: `merge(a,b)` then `merge(b,c)` puts all three
    in one class. That means a group can chain — a~b and b~c within tol while a~c is not — so
    the group reports BOTH its edge maximum (the pairs the probe actually licensed) and its
    closure maximum (every pair inside the group, whether or not the probe scored it within
    tol), and `chained` says when they differ. Nothing is filtered on that here: the licence
    decides, and the record says what it decided about.

    Returns groups sorted BEST LOSS FIRST — the caller evaluates them in that order, each on
    the state the taken ones left, so a refused group does not block the rest."""
    within = [p for p in pairs if p["loss"] <= tol]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in sorted(within, key=lambda q: (q["loss"], q["i"], q["j"])):
        a, b = find(p["i"]), find(p["j"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = collections.defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    look = {(p["i"], p["j"]): p["loss"] for p in pairs}
    out = []
    for root, mem in groups.items():
        mem = sorted(mem)
        edges = [p["loss"] for p in within if p["i"] in mem and p["j"] in mem]
        closed = [look[(a, b)] for xi, a in enumerate(mem) for b in mem[xi + 1:]
                  if (a, b) in look]
        out.append({"members": mem, "n": len(mem), "root": root,
                    "loss_edge_max": float(max(edges)) if edges else 0.0,
                    "loss_edge_mean": float(np.mean(edges)) if edges else 0.0,
                    "loss_closure_max": float(max(closed)) if closed else None,
                    "n_edges": len(edges), "n_pairs_closed": len(closed),
                    "chained": bool(closed and max(closed) > tol)})
    return sorted(out, key=lambda g: (g["loss_edge_max"], -g["n"], g["members"]))


def select_pairs(pairs, use, n_pick, tol, quota_bands=4):
    """Use-share-weighted selection under `questions.py`'s own round-robin kernel, imported
    unchanged — so the difficulty quota and the breadth-first tie-breaking are the arc's and
    not a new rule.

      prio   the pair's combined use share, from the slot-resolved record. `SIZING.md`
             section 4 is why the use record appears HERE and not in the criterion: a merge
             among rows the beam never calls is worth nothing, but the record cannot say which
             rows are interchangeable.
      keys   the pair's own row indices, so the round robin spreads across rows instead of
             merging one popular row with everything.
      d      a loss BAND, `int(loss / tol * quota_bands)` clipped — the difficulty axis.
    """
    cand = [p for p in pairs if p["loss"] <= tol]
    if not cand or n_pick <= 0:
        return []
    prio = np.array([float(use[p["i"]] + use[p["j"]]) for p in cand], float)
    keys = [(p["i"], p["j"]) for p in cand]
    d = np.array([min(quota_bands - 1, int(p["loss"] / max(tol, 1e-9) * quota_bands))
                  for p in cand], np.int64)
    sel = QS._round_robin(prio, keys, d, None, min(int(n_pick), len(cand)))
    return [cand[i] for i in sel]


# --------------------------------------------------------------------------- #
# the yield licence: the rise, and the floor it must clear
# --------------------------------------------------------------------------- #

def yield_rise(miner_next, a, b, support):
    """The next level's at-support count under the candidate map, minus its count now.

    Computed by re-keying `miner_next`'s OWN counts — the same remap `ClassMiner.rekey` would
    apply if the merge were taken — without mutating anything. Table-free, unpriced, and it is
    the one-level-up gauge asked of a partition instead of of a table.

    NOT MONOTONE, and this is why the licence does not use it. Two keys that are BOTH at
    support and share a half collapse into ONE key when that half's classes merge, so the
    count falls by one: `{(C,A): 5, (C,B): 4}` at support 3 goes from 2 keys to 1 and this
    returns -1.0. A coarsening op is then refused for making the next level SMALLER, which is
    the thing the merge exists to do (`T[l+1]` representable at 16 keys instead of 205,824).
    Kept and logged beside the mass rise because it is the gauge the commit thermostat reads,
    so the two currencies have to be comparable in the record; `yield_mass_rise` is what
    licenses."""
    a, b = (a, b) if a < b else (b, a)
    old = sum(1 for c in miner_next.counts.values() if c >= support)
    new = {}
    for key, c in miner_next.counts.items():
        k2 = tuple(a if x == b else x for x in key)
        new[k2] = new.get(k2, 0) + c
    return float(sum(1 for c in new.values() if c >= support) - old), old


def yield_mass_rise(miner_next, a, b, support):
    """THE LICENCE'S GAUGE: the share of the next level's OWN STREAM that lands on keys the
    learner holds at support, after the candidate re-key minus before.

    The one-level-up value of a coarsening is ARRIVAL, not key count — how much of what comes
    next the learner can already represent — and mass is the currency that says so. It is
    monotone under coarsening in the way the count is not: pooling can only lift a key to
    support, never drop one below it, so the at-support mass never falls. It is exactly 0 for a
    merge that lifts nothing (the `{(C,A): 5, (C,B): 4}` collapse above), which is the case the
    count currency scored -1.

    Returns (mass_rise, mass_before, min_mass). `min_mass` is one just-at-support key's worth of
    mass, `support / total`, and is the threshold the caller licenses above — STATED, not
    derived: the mass series has no banked ancestor because `ClassMiner.state()` logged
    at-support KEYS and not their counts. `state()` now logs the mass too, so the next round
    can derive a floor for it by `null_abba` the way every other floor in the arc is derived."""
    a, b = (a, b) if a < b else (b, a)
    tot = sum(miner_next.counts.values())
    if not tot:
        return 0.0, 0.0, 1.0
    before = sum(c for c in miner_next.counts.values() if c >= support) / tot
    new = {}
    for key, c in miner_next.counts.items():
        k2 = tuple(a if x == b else x for x in key)
        new[k2] = new.get(k2, 0) + c
    after = sum(c for c in new.values() if c >= support) / max(sum(new.values()), 1)
    return float(after - before), float(before), float(support) / float(tot)


def _remap(counts, members):
    """One re-key of a counts dict under `members -> min(members)`, the map `merge_group`
    installs. Shared by both group gauges so the licence and the op cannot read different
    objects (the thing M-4 exists to assert)."""
    mem = sorted(tuple(x) for x in members)
    keep, drop = mem[0], set(mem[1:])
    new = {}
    for key, c in counts.items():
        k2 = tuple(keep if x in drop else x for x in key)
        new[k2] = new.get(k2, 0) + c
    return new


def group_rise(miner_next, members, support):
    """[en_s3] `yield_rise` for a whole group. Still the COUNT currency, still not monotone,
    still logged rather than licensed on."""
    old = sum(1 for c in miner_next.counts.values() if c >= support)
    new = _remap(miner_next.counts, members)
    return float(sum(1 for c in new.values() if c >= support) - old), old


def group_mass_rise(miner_next, members, support):
    """[en_s3] THE LICENCE'S GAUGE for a whole group: `yield_mass_rise` with the pair replaced
    by the group's own map. Returns (mass_rise, mass_before, min_mass) exactly as the pair
    version does, so the two are directly comparable in the record and a group of two is the
    pair case."""
    tot = sum(miner_next.counts.values())
    if not tot:
        return 0.0, 0.0, 1.0
    before = sum(c for c in miner_next.counts.values() if c >= support) / tot
    new = _remap(miner_next.counts, members)
    after = sum(c for c in new.values() if c >= support) / max(sum(new.values()), 1)
    return float(after - before), float(before), float(support) / float(tot)


def buildable_mass_rise(miner_next, lower_rows, quot, level, members, support):
    """[en_s5] THE GATE A MERGE ACTS ON: the next level's BUILDABLE at-support mass.

    `en_s4` c188 is why this exists. Six groups were proposed at L4; the arrival gauge
    (`group_mass_rise`) read exactly 0.00000 on all six and the yield licence refused them all,
    while the ledger took them and the L5 build went 1 -> 16 entries. Both readings were
    correct about different things: the keys were ALREADY at support, so no arrival happened —
    what the merge did was make them BUILDABLE. A key `(c1, c2)` is only realisable as rows if
    the operative level-l book actually holds a row of each half-class, which is
    `ClassMiner.build`'s own ratchet; a key whose half names a class the frozen book does not
    carry is at support and worth nothing. Merging that class into one the book does carry
    unblocks it, and no arrival-shaped gauge can see that.

    So the currency here is the share of the next level's stream that lands on keys which are
    at support AND buildable, after the candidate re-key minus before — with the lower book's
    own classes re-mapped by the same merge, because both sides of the ratchet move.

    Returns (rise, before, min_mass) in `group_mass_rise`'s shape, so the two are directly
    comparable and both are logged on every proposal."""
    tot = sum(miner_next.counts.values())
    if not tot:
        return 0.0, 0.0, 1.0
    mem = sorted(tuple(x) for x in members)
    keep, drop = mem[0], set(mem[1:])
    have_b = {quot.id_of(r, level) for r in lower_rows}
    have_a = {keep if c in drop else c for c in have_b}

    def share(counts, have):
        t = max(sum(counts.values()), 1)
        return sum(c for k, c in counts.items()
                   if c >= support and all(h in have for h in k)) / t
    before = share(miner_next.counts, have_b)
    after = share(_remap(miner_next.counts, members), have_a)
    return float(after - before), float(before), float(support) / float(tot)


def expected_mass_rise(miner_next, members, support, n_rem):
    """[en_s5] THE LICENCE'S CURRENCY: DEFERRED ARRIVAL.

    The instantaneous mass gauge asks what is at support NOW. What a merge buys is that the
    next level's keys reach support SOONER as the stream continues — the spec's "representable
    at the existing budget" — and the instantaneous gauge is that quantity at horizon zero,
    which is why it read exactly 0.00000 on all six of `en_s4` c188's groups.

    Model. Over the level's own keys with counts `c_k` and `N_obs = sum c_k`, the level's
    mining node will still receive `N_rem` observations in this run (remaining cycles on the
    gated schedule, times `mine_cap`). Each key's future count is Poisson with rate
    `lam_k = (c_k / N_obs) * N_rem` — its observed share of the stream, projected forward — so
    the EXPECTED MASS AT SUPPORT is

        sum_k (c_k / N_obs) * P(c_k + Poisson(lam_k) >= support)

    and the rise is that quantity under the candidate map minus under the current one. Pooling
    is monotone in it: two at-support keys keep their mass (their probability is already 1),
    and two below-support keys pooled gain both count and rate, so their probability rises.

    At `N_rem = 0` the tail is 1 exactly when `c_k >= support` and 0 otherwise, so this IS
    `group_mass_rise` — gate M-4f asserts that equality rather than trusting it.

    `pge` is `tutti/sizing/phase0_l5.py`'s Poisson tail, imported and not re-derived."""
    from rhm.practice.tutti.sizing.phase0_l5 import pge

    def emass(counts):
        tot = sum(counts.values())
        if not tot:
            return 0.0
        out = 0.0
        for k, c in counts.items():
            share = c / tot
            need = int(support) - int(c)
            p = 1.0 if need <= 0 else pge(share * float(n_rem), need)
            out += share * p
        return float(out)
    before = emass(miner_next.counts)
    after = emass(_remap(miner_next.counts, members))
    return float(after - before), float(before), float(after)


def buildable_entry_rise(miner_next, lower_rows, quot, level, members, support, spell_cap=4):
    """[en_s5] THE NEXT LEVEL'S REALISABLE ROW COUNT, table-free.

    `buildable_mass_rise` above asks which KEYS the lower book can look up. `en_s4` c188 says
    that is not what a merge moves: at that proposal the L5 build had `n_at_support = 1` and
    `n_keys_built = 1` BEFORE and AFTER, and `n_entries` went 1 -> 16. The key was already
    buildable; what the merge changed is how many ROWS it expands to, because pooling two
    classes gives each half of that key four member spellings instead of one and the cross
    product is 4 x 4.

    So this replays `ClassMiner.build`'s own arithmetic — for every at-support key whose halves
    all have members, the product of `min(|class|, spell_cap)` — over the miner's counts and
    the lower book's class inventories, under the current map and under the candidate one. No
    table is materialised and no grading is spent.

    Returns (rise, before, after) as RAW ENTRY COUNTS: the quantity is a count of rows, not a
    share of a stream, and normalising it against a total that also moves would hide the thing
    it is measuring."""
    def entries(counts, cls_rows):
        inv = collections.Counter(cls_rows)
        tot = 0
        for k, c in counts.items():
            if c < support:
                continue
            n = 1
            for h in k:
                mm = min(int(inv.get(h, 0)), int(spell_cap))
                if mm == 0:
                    n = 0
                    break
                n *= mm
            tot += n
        return int(tot)
    mem = sorted(tuple(x) for x in members)
    keep, drop = mem[0], set(mem[1:])
    cls_b = [quot.id_of(r, level) for r in lower_rows]
    cls_a = [keep if c in drop else c for c in cls_b]
    before = entries(miner_next.counts, cls_b)
    after = entries(_remap(miner_next.counts, members), cls_a)
    return float(after - before), int(before), int(after)


# --------------------------------------------------------------------------- #
# the ablation: a merge the licence should refuse
# --------------------------------------------------------------------------- #

def forced_pair(pairs, mode, tol):
    """The merge-precision ablation (the parent spec's thread 6, one rung up). `subset` takes
    the pair whose loss sits just ABOVE `tol` — by `SIZING.md` section 5(e) that band is where
    one class is a strict subset of the other, i.e. a narrow class swallowed by an ambiguous
    one. `junk` takes the highest-loss pair involving a row that repairs nothing, which is the
    token-junk case and never merges on its own (0 of 101 such rows repair anything).

    The point of forcing is that the LICENCE'S VERDICT is still computed and logged, so the
    ablation measures whether the yield licence is too permissive rather than assuming it."""
    if mode == "subset":
        over = sorted([p for p in pairs if p["loss"] > tol], key=lambda p: p["loss"])
        return over[0] if over else None
    if mode == "junk":
        over = sorted(pairs, key=lambda p: -p["loss"])
        return over[0] if over else None
    return None


# --------------------------------------------------------------------------- #
# the ledger audition's fixture reader (gate M-7 only; never used in a run)
# --------------------------------------------------------------------------- #

class FixtureReader:
    """[en_s3] A stand-in `generator` for gate M-7, and for nothing else.

    `macro_features` reads exactly one thing off the generator, `block_logits(obs) ->
    (B, n_blocks, v)`, so a table audition can be exercised with no trained substrate. This
    reader is deliberately FALLIBLE:

      * where the leaves are VISIBLE it is exact — a level-1 feature scores -1e9 unless one of
        its m renderings matches the observed pair.
      * where they are MASKED (`obs == -1`: the damage cell the macro is being asked to repair,
        which `macro_features` blanks itself) every feature is possible. There it scores the
        TRUE feature `skill` higher than the rest and breaks the remaining ties with fixed
        per-instance pseudo-random noise.

    `skill` is the whole point of the fixture. At `skill = 0` the reader knows nothing about the
    cell and the max-sum DP is choosing among the table's entries on noise alone — which is
    where a wrongly merged row can win and cost the audition. A reader that is CERTAIN at the
    cell (a constant score, or a large `skill`) can never be hurt by a bad table: an impossible
    entry scores -1e9, a worse one loses to the true one, and `e_merge == e_keep` for every
    merge, correct or not. A real generator is uncertain exactly there, so the fixture has to
    be, and the gate reads both ends of the range rather than one point of it."""

    def __init__(self, rules, v, s, depth, seed=0, skill=0.0):
        self.leaf = np.asarray(rules[depth - 1], np.int64)      # (v, m, s)
        self.v, self.s, self.depth = int(v), int(s), int(depth)
        self.seed, self.skill = int(seed), float(skill)
        self.truth = None                                       # (B, n_blocks) true features

    def possible(self, x):
        """(B, nb, v): is feature f a legal reading of block b's observed leaves?"""
        import torch
        B, L = x.shape
        nb = L // self.s
        xx = x.view(B, nb, 1, 1, self.s)
        leaf = torch.as_tensor(self.leaf, dtype=x.dtype, device=x.device)[None, None]
        return ((xx == leaf) | (xx < 0)).all(-1).any(-1)

    def set_truth(self, clean_np):
        """The clean derivation's own level-1 features, so `skill` has something to point at.
        A fixture liberty: the batch order is the audition's, which is fixed."""
        import torch
        p = self.possible(torch.as_tensor(np.asarray(clean_np, np.int64)))
        self.truth = p.float().argmax(-1)                        # exactly one f per block
        return self.truth

    def block_logits(self, obs):
        import torch
        B, L = obs.shape
        nb = L // self.s
        ok = self.possible(obs)
        g = np.random.default_rng(self.seed * 1_000_003 + B * 31 + nb)
        noise = torch.as_tensor(g.random((B, nb, self.v)), dtype=torch.float32,
                                device=obs.device)
        if self.skill and self.truth is not None:
            t = self.truth[:B, :nb].to(obs.device)
            noise = noise + self.skill * torch.nn.functional.one_hot(t, self.v).float()
        return torch.where(ok, noise, torch.full_like(noise, -1e9))


def ledger_fixture(v=8, s=2, depth=6, m=2, rule_seed=0, level=2, n_inst=192,
                   support=3, tol=0.20, reader_seeds=(0, 1, 2, 3), skills=(0.0, 1.0),
                   spell_cap=4):
    """[en_s3] GATE M-7's fixture: the ledger audition, on the level a merge actually changes.

    `en_s2b` auditioned `e_keep` and `e_merge` at the MERGED level — and its reduction showed
    why that read equal on all 12 auditions and all 15 forced bad merges: `ClassMiner.rekey`
    re-keys with the level-(l-1) map, which a level-l merge does not touch, so the rebuilt table
    was the kept table row for row. A level-l merge changes the level-(l+1) KEY, and that is
    where the audition belongs.

    This fixture is the check that was missing. It builds a real level-(l+1) table over a real
    level-l book, merges (a) an EXACT ALIAS pair — same token class, probe loss 0 — and (b) a
    forced SUBSET pair — probe loss above tol, one class strictly inside the other — and
    auditions both at l+1 on that level's own demand:

        alias   e_merge == e_keep   the merge costs the level above nothing
        subset  e_merge >  e_keep   the merge costs it something

    Both run through `MC.apply_any` and `units.grade`, the same two calls the priced audition
    makes, with `FixtureReader` in the generator's place. Several reader seeds are used because
    a single one fixes which entry the DP's argmax lands on; the alias claim is asserted on
    EVERY seed (it is exact) and the subset claim on the seeds' worst case (it is a fact about
    a max-sum DP over a table that now contains a wrong row, and whether that row wins is what
    the reader decides)."""
    import torch
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.enharmonic import quotient as QT
    from rhm.practice.tutti.tutti import context_instances
    from rhm.practice.crystallize.units import grade

    LAD = {1: 25, 2: 12, 3: 6, 4: 3, 5: 1}
    nxt = level + 1
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon_np = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    canon_t = torch.as_tensor(canon_np, dtype=torch.long)
    truth = MC.true_tables(rules, depth, s, v, m, nxt)

    # the level-l book, and the level-(l+1) miner keyed by its classes
    lower = truth[level] if level >= 2 else MC.base_table(v)
    rows = [tuple(int(z) for z in r) for r in lower["flat"]]
    quot = LearnedQuotient()
    mn = QT.ClassMiner(nxt, s, quot, spell_cap=spell_cap)
    mn.observe(np.repeat(truth[nxt]["flat"], support, axis=0))

    # the probe, at the level's own cell — `_try_merge`'s own call
    node_l = (LAD[nxt] * s ** (nxt - 1)) // s ** (level - 1)
    r_l, x_l = context_instances(rules, {"name": f"m7L{level}", "level": level,
                                         "nodes": [LAD[level]]},
                                 n_inst, s, depth, v, m, seed=91_001)
    P, graded = transfer_profile(rules, x_l, r_l, rows, node_l, level, s, canon_np)
    pairs, live = pair_losses(P)
    tc = QT.token_class_sets(rules, np.array(rows, np.int64), level, canon_np, v, s, depth)[:, 0, :]
    cls = [frozenset(np.nonzero(r)[0].tolist()) for r in tc]

    def kind_of(p):
        a, b = cls[p["i"]], cls[p["j"]]
        if a == b:
            return "equal"
        if a < b or b < a:
            return "subset"
        return "disjoint" if not (a & b) else "overlap"

    def trial(p):
        """Build the level-(l+1) table under the candidate map, and say whether it MOVED.

        A merge that leaves the level above bit-identical is a vacuous audition — which is
        exactly what `en_s2b` was measuring without knowing it — so the fixture picks pairs
        whose merge actually changes the table and the gate asserts that it did."""
        snap_q, snap_m = quot.snapshot(), mn.snapshot()
        keep = mn.build(lower, support)
        quot.merge_group(level, [rows[p["i"]], rows[p["j"]]])
        mn.rekey()
        try_ = mn.build(lower, support)
        moved = not (try_["flat"].shape == keep["flat"].shape
                     and bool((try_["flat"] == keep["flat"]).all()))
        quot.restore(snap_q)
        mn.restore(snap_m)
        return keep, try_, moved

    def pick(ok):
        """The first candidate, in loss order, that satisfies `ok`, joins two DISTINCT rows and
        moves the table above. `true_tables` carries repeated flat rows, and merging a row with
        itself is a no-op the union-find correctly ignores."""
        best = None
        for p in sorted(pairs, key=lambda q: q["loss"]):
            if rows[p["i"]] == rows[p["j"]] or not ok(p):
                continue
            keep, try_, moved = trial(p)
            if not moved:
                continue
            if try_["child"].shape[0] != keep["child"].shape[0]:
                return p
            best = best or p
        return best

    alias = pick(lambda p: p["loss"] <= 1e-12 and kind_of(p) == "equal")
    subset = pick(lambda p: p["loss"] > tol and kind_of(p) == "subset")
    if subset is None:
        subset = forced_pair(pairs, "subset", tol)
    # THE INSTRUMENT'S ZERO: a merge of a row with itself. The whole path runs — probe, group,
    # `merge_group`, `rekey`, rebuild, two auditions — and the table does not move, so the
    # audition must read exactly 0. Without it a gate that finds `e_merge == e_keep` cannot
    # tell "the merge cost nothing" from "the audition is blind", which is the confusion
    # `en_s2b` was in.
    null = next((p for p in pairs if rows[p["i"]] == rows[p["j"]]), None)
    if null is None and pairs:
        null = dict(pairs[0], j=pairs[0]["i"], loss=0.0)

    # the audition's own instances: level-(l+1) demand at that level's node
    node_n = LAD[nxt]
    r_n, x_n, clean_n = context_instances(rules, {"name": f"m7L{nxt}", "level": nxt,
                                                  "nodes": [node_n]},
                                          n_inst, s, depth, v, m, seed=91_002, with_clean=True)
    x_t = torch.as_tensor(x_n, dtype=torch.long)

    def audition(tbl, reader):
        """[en_s4] An EMPTY table is not 1.0 by fiat any more: it is the executor as it
        stands, i.e. the same instances graded with no level-(l+1) move applied. That is the
        `e_keep` the run uses when the l+1 table is empty under the current key, and it is what
        makes a merge that brings a table into existence auditable at all."""
        if not tbl["child"].shape[0]:
            succ, _ = grade(x_t.cpu().numpy(), r_n, rules, s)
            return float(1.0 - succ.mean())
        mv = MC.to_device(MC.make_macro(nxt, node_n, s, tbl), torch.device("cpu"))
        xf = MC.apply_any(reader, x_t, mv, None, canon_t, depth, v, m, s)
        succ, _ = grade(xf.cpu().numpy(), r_n, rules, s)
        return float(1.0 - succ.mean())

    out = {"level": level, "next_level": nxt, "n_rows": len(rows), "n_pairs": len(pairs),
           "skills": [float(z) for z in skills], "reader_seeds": [int(z) for z in reader_seeds],
           "n_graded_probe": int(graded), "n_instances": n_inst,
           "node_probe": node_l, "node_audition": node_n, "cases": {}}
    for name, p in (("null", null), ("alias", alias), ("subset", subset)):
        if p is None:
            out["cases"][name] = {"found": False}
            continue
        mem = [rows[p["i"]], rows[p["j"]]]
        snap_q, snap_m = quot.snapshot(), mn.snapshot()
        tbl_keep = mn.build(lower, support)
        quot.merge_group(level, mem)
        mn.rekey()
        tbl_try = mn.build(lower, support)
        moved = not (tbl_try["flat"].shape == tbl_keep["flat"].shape
                     and bool((tbl_try["flat"] == tbl_keep["flat"]).all()))
        reads = []
        for sd in reader_seeds:
            for sk in skills:
                rd = FixtureReader(rules, v, s, depth, seed=sd, skill=sk)
                rd.set_truth(clean_n)
                reads.append({"seed": int(sd), "skill": float(sk),
                              "e_keep": audition(tbl_keep, rd),
                              "e_merge": audition(tbl_try, rd)})
        quot.restore(snap_q)
        mn.restore(snap_m)
        d = [r["e_merge"] - r["e_keep"] for r in reads]
        by_skill = {}
        for sk in skills:
            dd = [r["e_merge"] - r["e_keep"] for r in reads if r["skill"] == sk]
            by_skill[str(sk)] = {"min": float(min(dd)), "max": float(max(dd)),
                                 "mean": float(np.mean(dd))}
        out["cases"][name] = {
            "found": True, "kind": kind_of(p), "loss": float(p["loss"]),
            "members": [list(x) for x in mem],
            "class_i": sorted(cls[p["i"]]), "class_j": sorted(cls[p["j"]]),
            "entries_keep": int(tbl_keep["child"].shape[0]),
            "entries_merge": int(tbl_try["child"].shape[0]),
            "table_changed": bool(moved),
            "reads": reads, "delta_by_skill": by_skill,
            "delta_min": float(min(d)), "delta_max": float(max(d)),
            "delta_mean": float(np.mean(d)),
            "equal_on_every_seed": bool(all(abs(z) < 1e-12 for z in d))}
    # ---- the ARRIVAL cases: the l+1 table does not exist until the merge makes it -------- #
    # `en_s3`'s ledger refused every group at the top live level because the l+1 table was
    # empty under the current key and the audition was declared undefined. A miner whose keys
    # all sit one short of support is exactly that state, and pooling two classes is what lifts
    # one over: `tbl_keep` empty, `tbl_try` not. Two sub-cases, because the licence there is
    # STRICT (`e_merge < e_keep`) and the strictness is the whole content of it:
    #   arrival_live  the merged key builds a table that REPAIRS audition instances -> taken.
    #   arrival_null  it builds a table that repairs nothing -> e_merge == e_keep == 1.0 (the
    #                 no-move error on instances sampled broken) -> REFUSED.
    # Without the second, "defined" and "licensed" are the same test and the branch licenses
    # any non-empty table.
    half_w = s ** (nxt - 2) if nxt >= 3 else 1
    true_next = [tuple(int(z) for z in r) for r in truth[nxt]["flat"]]
    lowset = {tuple(int(z) for z in r) for r in lower["flat"]}
    eqpairs = [p for p in sorted(pairs, key=lambda q: q["loss"])
               if p["loss"] <= 1e-12 and rows[p["i"]] != rows[p["j"]]
               and kind_of(p) == "equal"]

    def arrival(a_row, b_row, second):
        """Two keys one short of support sharing a half; merging a with b lifts them over."""
        snap_q, snap_m = quot.snapshot(), mn.snapshot()
        mn_a = QT.ClassMiner(nxt, s, quot, spell_cap=spell_cap)
        mn_a.counts = {(a_row, second): support - 1, (b_row, second): support - 1}
        t_keep = mn_a.build(lower, support)
        quot.merge_group(level, [a_row, b_row])
        mn_a.rekey()
        t_try = mn_a.build(lower, support)
        rd = FixtureReader(rules, v, s, depth, seed=0, skill=max(skills))
        rd.set_truth(clean_n)
        e_k, e_m = audition(t_keep, rd), audition(t_try, rd)
        quot.restore(snap_q)
        mn.restore(snap_m)
        return {"found": True, "kind": "arrival",
                "members": [list(a_row), list(b_row)], "second_half": list(second),
                "entries_keep": int(t_keep["child"].shape[0]),
                "entries_merge": int(t_try["child"].shape[0]),
                "table_changed": bool(t_try["child"].shape[0] != t_keep["child"].shape[0]),
                "keep_case": "absent" if not t_keep["child"].shape[0] else "table",
                "defined": bool(t_try["child"].shape[0] or t_keep["child"].shape[0]),
                "e_keep": e_k, "e_merge": e_m, "delta": e_m - e_k,
                # the run's own rule in the absent case, applied here
                "licensed_strict": bool(e_m < e_k),
                "licensed_margin": bool(e_m <= e_k),
                "note": ("e_keep is the no-move error on instances sampled broken, hence 1.0 "
                         "exactly; `licensed_margin` is what `<= margin` would have said and "
                         "`licensed_strict` is what the run uses")}

    live_case, null_case = None, None
    for p in eqpairs:
        a_row, b_row = rows[p["i"]], rows[p["j"]]
        for r0 in true_next:
            h1, h2 = tuple(r0[:half_w]), tuple(r0[half_w:])
            if h2 not in lowset:
                continue
            if h1 == a_row or h1 == b_row:
                c = arrival(a_row, b_row, h2)
                if live_case is None and c["entries_keep"] == 0 and c["e_merge"] < c["e_keep"]:
                    live_case = c
            if live_case is not None:
                break
        if live_case is not None:
            break
    for p in eqpairs:
        a_row, b_row = rows[p["i"]], rows[p["j"]]
        for r0 in sorted(lowset):
            if r0 in (a_row, b_row):
                continue
            if any(tuple(t[:half_w]) in (a_row, b_row) and tuple(t[half_w:]) == r0
                   for t in true_next):
                continue                      # that is the LIVE construction, not the null one
            c = arrival(a_row, b_row, r0)
            if c["entries_keep"] == 0 and c["e_merge"] >= c["e_keep"]:
                null_case = c
                break
        if null_case is not None:
            break
    out["cases"]["arrival_live"] = live_case or {"found": False}
    out["cases"]["arrival_null"] = null_case or {"found": False}
    return out


# --------------------------------------------------------------------------- #
# THE GATES — offline, no GPU, no substrate
# --------------------------------------------------------------------------- #

def merge_gate(v=8, s=2, depth=6, m=2, rule_seed=0, n=512, verbose=False,
               with_audition=True):
    """M-1  INERTNESS. `LearnedQuotient` with no merges drives `ClassMiner` to `MC.Miner` bit
            for bit — same `child`, same `flat`, same row order — which is why a Q2 arm with
            the op off IS `flat`.
       M-2  THE MERGE IS A PARTITION OP. Idempotent, order-free, and transitive: merging (a,b)
            then (b,c) puts a, b and c in one class with one representative.
       M-3  RE-KEYING IS CONSERVATIVE. `ClassMiner.rekey` after a merge preserves total count
            mass exactly and never increases the number of keys.
       M-4  THE RISE IS MONOTONE AND MATCHES THE REKEY. `yield_rise` is >= 0 for every pair,
            and equals the at-support delta a real merge-plus-rekey produces. This is the gate
            that makes the FLOOR load-bearing rather than decorative.
       M-6  THE CRITERION IS `fourwall`'S. At a fixed tol, `pair_losses` selects exactly the
            pairs `fourwall.merge_candidates` selects from the same success matrix — so the
            merge criterion is the arc's, re-expressed for a shared instance pool and not a
            new rule.
       M-5  THE LOSS MARGIN IS WHERE THE SIZING LANE MEASURED IT. On true rows at a level's own
            cell, same-token-class pairs sit at loss exactly 0 and the RESOLVABLE part of the
            subset band is strictly above it, so a non-empty tol window exists. The band's
            median on this gate's own population (true rows at one level) is ~0.35, against
            `SIZING.md` section 5(e)'s 0.4328 over the 59 arms' committed books pooled across
            L2-L4 — a different population, not a disagreement, and the gate reports its own.
            The band's MINIMUM is 0, and deliberately so: section 5(a)
            measured a STRUCTURAL collapse at every level (2 of 9 / 2 of 11 / 2 of 13 classes
            are indistinguishable at the node at any N), so some subset pairs are invisible to
            any probe. Those are exactly the merges no tol can refuse, and the yield floor is
            the only thing that can — which is the second reason the floor is load-bearing.
            The count is reported as `n_subset_structurally_collapsed`.
    """
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.enharmonic import quotient as QT
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    truth = MC.true_tables(rules, depth, s, v, m, 5)
    rng = np.random.default_rng(23)
    out = {}

    # ---- M-1 inertness ---------------------------------------------------------------- #
    g1 = []
    for level in (2, 3, 4):
        span = s ** (level - 1)
        pool = np.concatenate([truth[level]["flat"],
                               rng.integers(0, v, size=(n // 4, span))], axis=0)
        obs = pool[rng.integers(0, pool.shape[0], size=n)]
        lq = LearnedQuotient()
        a, b = MC.Miner(level, s), QT.ClassMiner(level, s, lq, spell_cap=4)
        a.observe(obs); b.observe(obs)
        lower = MC.base_table(v) if level == 2 else truth[level - 1]
        ta, tb = a.build(lower, 3), b.build(lower, 3)
        same = (ta["child"].shape == tb["child"].shape
                and bool((ta["child"] == tb["child"]).all())
                and bool((ta["flat"] == tb["flat"]).all()))
        g1.append({"level": level, "n_entries": int(ta["child"].shape[0]), "identical": same,
                   "oracle_reads": 0})
        assert same, f"M-1 FAILED at L{level}: an unmerged LearnedQuotient is not MC.Miner"
    out["M-1 inertness"] = g1

    # ---- M-2 the merge is a partition op ---------------------------------------------- #
    lq = LearnedQuotient()
    A, B, C = (0, 1), (2, 3), (4, 5)
    r1 = lq.merge(2, A, B)
    r2 = lq.merge(2, B, A)
    r3 = lq.merge(2, B, C)
    reps = {lq.id_of(x, 2) for x in (A, B, C)}
    out["M-2 partition op"] = {"rep_after_ab": list(r1), "idempotent": bool(r1 == r2),
                               "transitive_one_class": len(reps) == 1,
                               "representative_is_min": bool(list(reps)[0] == min(A, B, C)),
                               "n_merges": lq.n_merges}
    assert r1 == r2 and len(reps) == 1 and list(reps)[0] == min(A, B, C), \
        f"M-2 FAILED: {out['M-2 partition op']}"

    # ---- M-3 / M-4 rekey and rise ------------------------------------------------------ #
    level = 3
    lq = LearnedQuotient()
    mn = QT.ClassMiner(level + 1, s, lq, spell_cap=4)
    # BELOW support on purpose: every key seen twice against a support of 3, so pooling two
    # classes CAN push a key over and the gate is not vacuous. (Seeded at or above support the
    # rise is identically 0 for every pair and M-4 asserts nothing.)
    mn.observe(np.repeat(truth[level + 1]["flat"][:400], 2, axis=0))
    halves = sorted({k[i] for k in mn.counts for i in range(s)})
    mass0, keys0 = sum(mn.counts.values()), len(mn.counts)
    sup = 3
    g4 = []
    # the pairs whose rise is largest, so at least one is strictly positive
    cands = [(halves[i], halves[j]) for i in range(len(halves))
             for j in range(i + 1, len(halves))]
    cands = sorted(cands, key=lambda ab: -yield_rise(mn, ab[0], ab[1], sup)[0])[:2]
    for (a, b) in cands:
        pred, before = yield_rise(mn, a, b, sup)
        mrise, mbefore, mmin = yield_mass_rise(mn, a, b, sup)
        assert mrise >= -1e-12, (
            "M-4 FAILED: the MASS rise is negative — pooling cannot drop a key below support")
        lq.merge(level, a, b)
        mn.rekey()
        after = sum(1 for c in mn.counts.values() if c >= sup)
        g4.append({"predicted_rise": pred, "realised_rise": float(after - before),
                   "at_support_before": before, "at_support_after": after,
                   "mass_rise": mrise, "mass_before": mbefore, "min_mass": mmin})
        assert abs(pred - (after - before)) < 1e-9, (
            f"M-4 FAILED: the predicted rise {pred} is not what rekey produced "
            f"{after - before} — the licence is reading a different object from the op")
    assert sum(mn.counts.values()) == mass0 and len(mn.counts) <= keys0, \
        "M-3 FAILED: rekey did not conserve count mass or did not coarsen"
    out["M-3 rekey conservative"] = {"mass_before": mass0, "mass_after": sum(mn.counts.values()),
                                     "keys_before": keys0, "keys_after": len(mn.counts)}
    # M-4b THE SIGN. A hand-built case where two at-support keys sharing a half collapse: the
    #      COUNT currency scores it -1 (a coarsening penalised for coarsening) while the MASS
    #      currency scores it >= 0. This is the case that broke `en_s2`'s `endo_yield` at c78
    #      and it is why the licence reads mass.
    class _M:
        def __init__(self, c):
            self.counts = c
    _A, _B, _C = (0,), (1,), (2,)
    _m = _M({(_C, _A): 5, (_C, _B): 4, (_C, (9,)): 1})
    _cr, _cb = yield_rise(_m, _A, _B, 3)
    _mr, _mb, _mm = yield_mass_rise(_m, _A, _B, 3)
    out["M-4b the sign"] = {"count_rise": _cr, "count_at_support_before": _cb,
                            "mass_rise": _mr, "mass_before": _mb, "min_mass": _mm}
    assert _cr < 0 and _mr >= -1e-12, (
        f"M-4b FAILED: the collapsing case no longer separates the two currencies "
        f"({out['M-4b the sign']}) — either the count gauge stopped being non-monotone or the "
        f"mass gauge started being")
    out["M-4 rise matches rekey"] = g4
    assert any(r["predicted_rise"] > 0 for r in g4), (
        "M-4 is VACUOUS: no candidate merge raised the next level's at-support count at all, "
        "so the agreement it asserts is between two zeros and the licence is untested")

    # ---- M-5 the loss margin ----------------------------------------------------------- #
    from rhm.practice.tutti.tutti import context_instances
    LAD = {1: 25, 2: 12, 3: 6, 4: 3, 5: 1}
    level = 3
    node = (LAD[level] * s ** (level - 1)) // (s ** (level - 1))
    keys = sorted({tuple(int(x) for x in r) for r in truth[level]["flat"]})[:60]
    tc = QT.token_class_sets(rules, np.array(keys, np.int64), level, canon, v, s, depth)[:, 0, :]
    cls = [frozenset(np.nonzero(r)[0].tolist()) for r in tc]
    roots_, x_ = context_instances(rules, {"name": "m5", "level": level, "nodes": [LAD[level]]},
                                   256, s, depth, v, m, seed=4242)
    P, graded = transfer_profile(rules, x_, roots_, keys, node, level, s, canon)
    pairs, live = pair_losses(P)
    eq = [p["loss"] for p in pairs if cls[p["i"]] == cls[p["j"]]]
    sub_ = [p["loss"] for p in pairs
            if cls[p["i"]] < cls[p["j"]] or cls[p["j"]] < cls[p["i"]]]
    n_collapsed = sum(1 for x in sub_ if x == 0.0)
    live_sub = [x for x in sub_ if x > 0.0]
    out["M-5 loss margin"] = {
        "population": f"the {len(keys)} lowest true L{level} rows at node {node}, N={P.shape[1]}",
        "n_graded": int(graded), "n_pairs": len(pairs),
        "equal_max": (max(eq) if eq else None), "n_equal": len(eq),
        "subset_min": (min(sub_) if sub_ else None),
        "subset_min_resolvable": (min(live_sub) if live_sub else None),
        "subset_median": (float(np.median(sub_)) if sub_ else None),
        "n_subset": len(sub_), "n_subset_structurally_collapsed": n_collapsed,
        "tol_window": [0.0, (min(live_sub) if live_sub else None)]}
    assert not eq or max(eq) == 0.0, "M-5 FAILED: a same-token-class pair has non-zero loss"
    if live_sub:
        assert min(live_sub) > 0.0 and float(np.median(sub_)) > 0.2, (
            f"M-5 FAILED: the resolvable subset band is not separated from the equal band "
            f"({out['M-5 loss margin']}) — the tol window is void")

    # ---- M-6 the criterion IS `fourwall.merge_candidates` ------------------------------- #
    tol_probe = 0.30
    mine = {(p["i"], p["j"]) for p in pairs if p["loss"] <= tol_probe}
    X = np.full((len(live), len(live)), np.nan)
    for xi, i in enumerate(live):
        for yj, j in enumerate(live):
            X[xi, yj] = float((P[i] & P[j]).sum()) / float(P[j].sum())
    theirs = {(live[c["a"]], live[c["b"]])
              for c in W.merge_candidates(X, list(range(len(live))), tol_probe)}
    out["M-6 fourwall criterion"] = {"tol": tol_probe, "n_mine": len(mine),
                                     "n_fourwall": len(theirs), "identical": mine == theirs}
    assert mine == theirs, (
        f"M-6 FAILED: `pair_losses` is not `fourwall.merge_candidates`' criterion "
        f"({len(mine)} vs {len(theirs)} pairs at tol {tol_probe})")

    # ---- M-2b THE GROUP CLOSURE, and that it is ORDER-FREE ------------------------------ #
    # `alias_groups` closes every within-tol pair with the same union-find M-2 gates for a
    # single pair. Two things are asserted: the closure is TRANSITIVE (a~b, b~c => one group),
    # and it does not depend on the order the probe emitted its pairs in — the same property
    # `merge` has, one level up. A chained group (a~b, b~c within tol, a~c not) is REPORTED,
    # not filtered: it is the licence's job to refuse it and the record's job to say so.
    _pairs_fix = [{"i": 0, "j": 1, "loss": 0.0}, {"i": 1, "j": 2, "loss": 0.10},
                  {"i": 0, "j": 2, "loss": 0.55}, {"i": 3, "j": 4, "loss": 0.05},
                  {"i": 4, "j": 5, "loss": 0.90}]
    _g_fwd = alias_groups(_pairs_fix, 0.20)
    _rng2 = np.random.default_rng(7)
    _g_rev = alias_groups(list(reversed(_pairs_fix)), 0.20)
    _shuf = list(_pairs_fix)
    _rng2.shuffle(_shuf)
    _g_shf = alias_groups(_shuf, 0.20)
    out["M-2b group closure"] = {
        "groups": [{k: g[k] for k in ("members", "loss_edge_max", "loss_closure_max",
                                      "chained")} for g in _g_fwd],
        "order_free": bool([g["members"] for g in _g_fwd] == [g["members"] for g in _g_rev]
                           == [g["members"] for g in _g_shf]),
        "best_loss_first": [g["loss_edge_max"] for g in _g_fwd]}
    assert out["M-2b group closure"]["order_free"], (
        f"M-2b FAILED: the closure depends on pair order — {out['M-2b group closure']}")
    # best loss first, so the un-chained {3,4} at 0.05 precedes the chained {0,1,2} at 0.10
    assert [g["members"] for g in _g_fwd] == [[3, 4], [0, 1, 2]], (
        f"M-2b FAILED: the closure is not transitive — {out['M-2b group closure']}")
    assert _g_fwd[1]["chained"] and not _g_fwd[0]["chained"], (
        "M-2b FAILED: a chained group is not being reported as chained")
    assert (out["M-2b group closure"]["best_loss_first"]
            == sorted(out["M-2b group closure"]["best_loss_first"])), (
        "M-2b FAILED: groups are not returned best-loss-first, so the caller's "
        "already-taken-state order is not the order the record claims")

    # ---- M-4c THE GROUP GAUGES MATCH THE GROUP OP --------------------------------------- #
    # M-4 for a whole group: the mass rise is >= 0 and the COUNT rise is what `merge_group`
    # plus `rekey` actually produces. Same assertion, same two currencies, one op up — this is
    # what stops the whole-partition licence reading a different object from the one it applies.
    level = 3
    lq = LearnedQuotient()
    mn = QT.ClassMiner(level + 1, s, lq, spell_cap=4)
    mn.observe(np.repeat(truth[level + 1]["flat"][:400], 2, axis=0))
    halves = sorted({k[i] for k in mn.counts for i in range(s)})
    sup = 3
    grp = sorted(halves, key=lambda h: -group_mass_rise(mn, [h, halves[0]], sup)[0])[:3]
    grp = sorted(set(grp) | {halves[0]})
    pred_c, before_c = group_rise(mn, grp, sup)
    pred_m, before_m, min_m = group_mass_rise(mn, grp, sup)
    mass0, keys0 = sum(mn.counts.values()), len(mn.counts)
    lq.merge_group(level, grp)
    mn.rekey()
    after_c = sum(1 for c in mn.counts.values() if c >= sup)
    after_m = (sum(c for c in mn.counts.values() if c >= sup)
               / max(sum(mn.counts.values()), 1))
    out["M-4c group rise matches group op"] = {
        "n_members": len(grp), "predicted_count_rise": pred_c,
        "realised_count_rise": float(after_c - before_c),
        "predicted_mass_rise": pred_m, "realised_mass_rise": float(after_m - before_m),
        "mass_before": before_m, "min_mass": min_m,
        "one_class_after": len({lq.id_of(g, level) for g in grp}) == 1,
        "mass_conserved": sum(mn.counts.values()) == mass0, "keys_coarsened": len(mn.counts) <= keys0}
    assert pred_m >= -1e-12, "M-4c FAILED: a group's MASS rise is negative"
    assert abs(pred_c - (after_c - before_c)) < 1e-9, (
        f"M-4c FAILED: the predicted group count rise {pred_c} is not what merge_group+rekey "
        f"produced {after_c - before_c}")
    assert abs(pred_m - (after_m - before_m)) < 1e-12, (
        f"M-4c FAILED: the predicted group MASS rise {pred_m} is not what merge_group+rekey "
        f"produced {after_m - before_m} — the licence reads a different object from the op")
    assert out["M-4c group rise matches group op"]["one_class_after"] and \
        out["M-4c group rise matches group op"]["mass_conserved"], \
        f"M-4c FAILED: {out['M-4c group rise matches group op']}"

    # ---- M-7 THE LEDGER AUDITION IS ON THE LEVEL A MERGE CHANGES ------------------------ #
    # The check `en_s2b` lacked. Its ledger auditioned the MERGED level, which a merge at that
    # level cannot move (`ClassMiner.rekey` re-keys the level below the miner's own), so
    # `e_keep == e_merge` on all 12 auditions and all 15 forced bad merges — an equality by
    # construction, read as evidence. Here the audition is at l+1, on a fixture built for the
    # purpose, and three things are asserted:
    #   null    a merge of a row with itself moves nothing and the audition reads EXACTLY 0.
    #           This is the instrument's zero: without it "cost nothing" and "blind" look alike.
    #   alias   an exact-alias merge (same token class, probe loss 0) moves the table above and
    #           never costs it — with an INFORMED reader it is uniformly better, because
    #           pooling an alias class adds legal renderings the DP can pick.
    #   subset  a forced subset merge (a narrow class swallowed by an ambiguous one) moves the
    #           table above and never helps it, and costs it on at least one reader.
    # The informed end is where the claim is asserted: at `skill = 0` the reader knows nothing
    # about the damaged cell, the DP is choosing among entries on noise alone, and both cases
    # move by a couple of instances in either direction. That noise floor is REPORTED, not
    # asserted around.
    if with_audition:
        fx = ledger_fixture(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, tol=0.20)
        out["M-7 ledger audition at l+1"] = fx
        _null, _al, _sb = (fx["cases"].get("null"), fx["cases"].get("alias"),
                           fx["cases"].get("subset"))
        assert _null and _null.get("found") and not _null["table_changed"] \
            and abs(_null["delta_min"]) < 1e-12 and abs(_null["delta_max"]) < 1e-12, (
            f"M-7 FAILED (null): a no-op merge does not read 0 at l+1 — the audition is not "
            f"measuring the table it is handed ({_null})")
        assert _al and _al.get("found") and _al["table_changed"], (
            f"M-7 FAILED (alias): no exact-alias pair moved the level above, so the case is "
            f"vacuous ({_al})")
        assert _sb and _sb.get("found") and _sb["table_changed"], (
            f"M-7 FAILED (subset): no forced subset pair moved the level above ({_sb})")
        _inf = str(max(fx.get("skills", (0.0, 1.0))))
        _da, _ds = _al["delta_by_skill"][_inf], _sb["delta_by_skill"][_inf]
        assert _da["max"] <= 1e-12, (
            f"M-7 FAILED (alias): an exact-alias merge COSTS the level above under an informed "
            f"reader ({_da}) — the audition is not type-matched to the op")
        assert _ds["min"] >= -1e-12 and _sb["delta_max"] > 0, (
            f"M-7 FAILED (subset): a forced subset merge does not cost the level above "
            f"({_ds}, max over readers {_sb['delta_max']}) — the audition cannot refuse a bad "
            f"merge and the ledger licence is decoration")
        _al, _an = fx["cases"].get("arrival_live"), fx["cases"].get("arrival_null")
        for _nm, _ar in (("arrival_live", _al), ("arrival_null", _an)):
            assert _ar and _ar.get("found"), (
                f"M-7 FAILED ({_nm}): the fixture could not build the case where the l+1 table "
                f"does not exist under the current key, so the branch `en_s4` added is "
                f"untested on that side")
            assert _ar["entries_keep"] == 0 and _ar["entries_merge"] > 0, (
                f"M-7 FAILED ({_nm}): the case is not the one it claims to be ({_ar})")
            assert _ar["keep_case"] == "absent" and _ar["defined"] and \
                _ar["e_keep"] is not None and _ar["e_merge"] is not None, (
                f"M-7 FAILED ({_nm}): a merge that brings a non-empty l+1 table into existence "
                f"did not produce a DEFINED audition ({_ar})")
        # and the STRICT licence has to separate them: a table that repairs an instance is
        # taken, one that repairs nothing is refused. Under `<= margin` both would be taken,
        # which is reported beside it.
        assert _al["licensed_strict"] and _al["e_merge"] < _al["e_keep"], (
            f"M-7 FAILED (arrival_live): a merged l+1 table that repairs instances is not "
            f"licensed ({_al})")
        assert not _an["licensed_strict"] and _an["e_merge"] >= _an["e_keep"], (
            f"M-7 FAILED (arrival_null): a merged l+1 table that repairs NOTHING is licensed "
            f"({_an}) — the strict test is not doing anything")
        assert _an["licensed_margin"], (
            f"M-7 note FAILED: `<= margin` was expected to license the null arrival (that is "
            f"why the branch is strict) but did not ({_an})")
        assert _da["max"] < _ds["min"] or (_da["max"] <= 1e-12 < _ds["max"]), (
            f"M-7 FAILED: the audition does not SEPARATE an alias merge from a subset merge "
            f"(alias {_da}, subset {_ds})")

    # ---- M-4d THE BUILDABLE GAUGE SEES WHAT ARRIVAL CANNOT ------------------------------ #
    # `en_s4` c188, in a fixture: two level-(l+1) keys both already AT SUPPORT, both blocked
    # because their halves name a class the operative level-l book does not carry. Arrival is
    # flat — nothing new lands at support — while the merge unblocks both. The gate asserts the
    # arrival rise is exactly 0 and the buildable rise is strictly positive, which is the whole
    # reason the licence's currency changed.
    _A, _B = (0, 1), (2, 3)
    _mn = _M({(_A, _B): 5, (_B, _A): 4})
    _q = LearnedQuotient()
    _lower = [_A]                      # the book holds a row of class A and none of class B
    _ar, _ab, _am = yield_mass_rise(_mn, _A, _B, 3)
    _br, _bb, _bm = buildable_mass_rise(_mn, _lower, _q, 2, [_A, _B], 3)
    _cr, _cb = yield_rise(_mn, _A, _B, 3)
    out["M-4d buildable vs arrival"] = {
        "keys": {"(A,B)": 5, "(B,A)": 4}, "lower_book_classes": [list(_A)],
        "arrival_rise": _ar, "arrival_before": _ab,
        "buildable_rise": _br, "buildable_before": _bb, "count_rise": _cr,
        "min_mass": _bm}
    assert abs(_ar) < 1e-12, (
        f"M-4d FAILED: the arrival gauge moved on a merge that added no arrival "
        f"({out['M-4d buildable vs arrival']}) — the fixture is not the c188 shape")
    assert _bb == 0.0 and _br > 0.0, (
        f"M-4d FAILED: the buildable gauge did not see the unblocking "
        f"({out['M-4d buildable vs arrival']}) — the licence's new currency is blind to the "
        f"case it was introduced for")

    # ---- M-4e THE c188 SHAPE AS IT ACTUALLY WAS ----------------------------------------- #
    # `en_s4` c188, from its own build log: `n_at_support` 1, `n_keys_built` 1, `n_entries`
    # 1 -> 16, `n_lower_classes` 100 -> 42. The key was ALREADY buildable and stayed one key;
    # the merge multiplied its cross-product inventory. M-4d's unblocking shape is a real
    # phenomenon but not this one, and a licence reading key-buildability cannot see c188 at
    # all: a merge maps a class onto another, so the book's class set only shrinks, and the
    # classes a proposal contains come from the book's own rows and are therefore already in
    # it. This gate pins the difference.
    _A = [(0, 1), (0, 2), (0, 3), (0, 4)]          # four rows, four classes, one key
    _mn2 = _M({(_A[0], _A[0]): 5})
    _q2 = LearnedQuotient()
    _ar2, _, _ = yield_mass_rise(_mn2, _A[0], _A[1], 3)
    _br2, _, _ = buildable_mass_rise(_mn2, _A, _q2, 2, _A, 3)
    _er2, _eb2, _ea2 = buildable_entry_rise(_mn2, _A, _q2, 2, _A, 3, spell_cap=4)
    out["M-4e c188 shape"] = {
        "keys": {"(A0,A0)": 5}, "lower_book_classes": len(_A),
        "arrival_rise": _ar2, "buildable_key_rise": _br2,
        "entry_rise": _er2, "entries_before": _eb2, "entries_after": _ea2}
    assert abs(_ar2) < 1e-12 and abs(_br2) < 1e-12, (
        f"M-4e FAILED: a gauge that should be blind to c188 moved ({out['M-4e c188 shape']})")
    assert _eb2 == 1 and _ea2 == 16 and _er2 > 0, (
        f"M-4e FAILED: the entry gauge does not reproduce c188's 1 -> 16 "
        f"({out['M-4e c188 shape']})")

    # ---- M-4f DEFERRED ARRIVAL: the currency that sees c188 ------------------------------ #
    # Two keys one short of support, pooled by the merge, against one key already at support.
    # The INSTANTANEOUS mass is unmoved — nothing crosses support this instant — while the
    # EXPECTED mass at support rises, because the pooled key carries both counts and both
    # rates into the stream that is still to come. `N_rem = 13 * 8` is `en_s4` c188's own
    # remaining budget: 13 cycles left of the run at `mine_cap` 8.
    _K = {((0, 1), (0, 1)): 5, ((2, 3), (2, 3)): 1, ((4, 5), (4, 5)): 1}
    _mn3 = _M(dict(_K))
    _B, _C = (2, 3), (4, 5)
    _im, _, _ = yield_mass_rise(_mn3, _B, _C, 3)
    _xr, _xb, _xa = expected_mass_rise(_mn3, [_B, _C], 3, 13 * 8)
    out["M-4f deferred arrival"] = {
        "keys": {str(k): v for k, v in _K.items()}, "n_rem": 13 * 8,
        "instantaneous_mass_rise": _im, "expected_before": _xb, "expected_after": _xa,
        "expected_rise": _xr}
    assert abs(_im) < 1e-12, (
        f"M-4f FAILED: the instantaneous gauge moved on the deferred-arrival shape "
        f"({out['M-4f deferred arrival']}) — the fixture is not the case it claims")
    assert _xr > 0.0, (
        f"M-4f FAILED: the expected gauge did not see the deferred arrival "
        f"({out['M-4f deferred arrival']}) — pooling two below-support keys must raise the "
        f"probability that the level reaches support before the run ends")
    # AT HORIZON ZERO IT IS THE MASS GAUGE, on every fixture in this file.
    _eq = []
    _P, _Q = (0, 1), (2, 3)
    for _mm, _pair in ((_mn3, (_B, _C)), (_M(dict(_K)), ((0, 1), _B)),
                       (_M({(_P, _Q): 5, (_Q, _P): 4}), (_P, _Q))):
        _m0, _, _ = yield_mass_rise(_mm, _pair[0], _pair[1], 3)
        _x0, _, _ = expected_mass_rise(_mm, list(_pair), 3, 0)
        _eq.append({"mass_rise": _m0, "expected_rise_at_n_rem_0": _x0})
        assert abs(_m0 - _x0) < 1e-12, (
            f"M-4f FAILED: at N_rem = 0 the expected gauge is not the mass gauge ({_eq[-1]})")
    out["M-4f deferred arrival"]["horizon_zero_equals_mass"] = _eq

    out["verdict"] = "PASS"
    if verbose:
        import json as _j
        print(_j.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    merge_gate(verbose=True)
