"""The wall: a free surface correlate of the demanded latent, the keyed library, and merge.

WHY THIS EXISTS. `setlist` drifted WHAT IS ASKED while the grammar stood still, and measured
that the maintenance organ for demand-news is evaluative re-selection. This round drifts
something one level further out: not what is asked, but **what the index means**.

THE MANIPULATION. Every task instance carries a surface token `w` that names -- for free, with
no parse -- the level-`key_level` feature the damage destroyed at node `key_node`. In phase 1
`w = f`, a perfect free correlate of a latent the agent would otherwise have to infer from the
root and the intact siblings. A rotation event then PERMUTES the map (`w = (f + q) mod v`); it
does not delete `w`. Nothing becomes false: the grammar, the damage operator, the demand over
features and the held-out instances are all literally unchanged objects. The only thing that
moves is the correspondence between a free observable and the latent it named. That is
demand-news aimed squarely at the library's INDEX.

  * `wall_map` / `rotation_index`  -- the schedule (cyclic, sudden; paced rotation is a
    deliberate non-goal of this round).
  * `sample_pool_latent` / `context_instances_wall` -- `ratchet.context_instances` with the
    generative latents carried out alongside, so the key is EXACT rather than parsed back out
    of the leaves through the last-writer-wins inverse map.
  * `Library` -- key -> index class -> committed table. Every op the arc already has (mine,
    commit, recert) acts on an entry under a fixed index; `merge` is the first that acts on the
    index itself, coarsening the partition and taking the DEDUPLICATED UNION of the merged
    cells' vocabularies. It never averages content: entries are move programs and the merged
    cell holds the set of distinct programs, which is why any track-vs-merge difference is
    index compression and nothing else.
  * `entry_success` / `transfer_matrix` / `consistent_features` -- the exact, MODEL-FREE
    readouts this substrate makes possible. A committed entry is a level-1 tuple rendered
    canonically into a span, and whether writing it repairs an instance is decided by the
    grammar's own possible-set DP. So "does the unit mined under wall a serve the demand now
    arriving at wall a" is answerable exactly, per entry, with no learned grader in the loop --
    the forced-transfer instrument of the idea doc's Sec 5, delivered by the substrate.

WHAT PROGRAMS CANNOT DO. A table entry is a tuple of level-1 features. There is nowhere in it
for `w` to appear, so a committed unit cannot be contaminated by the wall even in principle.
`w` reaches the agent through exactly two doors: as the library's KEY (the keyed arms) and as an
extra conditioning input to the selector (the `free_selector` arm). Keeping those two doors
separate is the whole design.
"""

from collections import OrderedDict

import numpy as np

from rhm.practice.crystallize.units import corrupt_hier
from rhm.practice.ratchet import macros as MC
from rhm.practice.transpose.drift import DecayMiner
from rhm.practice.setlist import demand as DM
from rhm.rhm_drift import sample_derivations_weighted
from rhm.rhm_sculpt_precheck import nearest_derivation_cost, parse_success_and_heuristic


# --------------------------------------------------------------------------- #
# the wall and its rotation schedule
# --------------------------------------------------------------------------- #

def wall_map(q, v):
    """The wall<->feature map at rotation index `q`: `w = (f + q) mod v`, as an array indexed
    by feature. `q = 0` is the identity, i.e. phase 1, where the wall is a perfect free name
    for the latent."""
    return (np.arange(v, dtype=np.int64) + int(q)) % v


def rotation_index(cyc, phase1, period, step=1):
    """`q` at cycle `cyc`. Phase 1 (cycles 1..`phase1`) is stationary at q = 0; the first
    rotation lands on cycle `phase1 + 1` and phase 2 rotates every `period` cycles thereafter.

    The schedule is a CYCLIC shift, which is the line-dance's own structure: after `v / gcd`
    rotations the wall means what it meant at the start again. That return is not decoration --
    it is a free instrument for whether a stale cell can come back into fashion, the
    demand-side analogue of `setlist`'s mean-reversion caveat."""
    if period <= 0 or cyc <= phase1:
        return 0
    return int(step) * (1 + (cyc - phase1 - 1) // period)


def is_rotation_cycle(cyc, phase1, period):
    if period <= 0 or cyc <= phase1:
        return False
    return (cyc - phase1 - 1) % period == 0


# --------------------------------------------------------------------------- #
# instances, with the generative latent carried out alongside
# --------------------------------------------------------------------------- #

def sample_pool_latent(rules, n, s, seed, st=None):
    """`demand.sample_pool_demand` with `return_trace`, so the level-l features that generated
    each derivation come back with it. `trace[ell][0]` holds the features at level
    `depth - ell`, which is what makes the key EXACT: it is the latent the world used, not a
    parse of the leaves it produced."""
    rng = np.random.default_rng(seed)
    v = rules[0].shape[0]
    if st is None:
        roots = rng.integers(0, v, size=n)
        weights = None
    else:
        roots = rng.choice(v, size=n, p=DM.root_prior(st))
        weights = DM.rule_weights(rules, st)
    leaves, trace = sample_derivations_weighted(rules, roots, s, rng, weights,
                                                return_trace=True)
    return roots.astype(np.int64), leaves.astype(np.int64), trace


def latents_at(trace, depth, level):
    """(B, s**(depth-level)) features at `level`, from a `sample_derivations_weighted` trace."""
    return np.asarray(trace[depth - level][0], np.int64)


def context_instances_wall(rules, ctx, n, s, depth, v, m, seed, st, key_level, key_node,
                           require_broken=True):
    """`ratchet.context_instances` / `demand.context_instances_demand`, verbatim in its damage
    and rejection rule, returning additionally the TRUE level-`key_level` feature at
    `key_node` of the clean derivation the damage was applied to. That feature is the latent
    the wall names."""
    length = s ** depth
    roots = np.zeros(n, np.int64)
    x = np.zeros((n, length), np.int64)
    clean = np.zeros((n, length), np.int64)
    feat = np.zeros(n, np.int64)
    rng = np.random.default_rng(seed + 90_001)
    need = np.arange(n)
    for attempt in range(64):
        if len(need) == 0:
            break
        r, lv, tr = sample_pool_latent(rules, len(need), s, seed + 7919 * attempt, st)
        f = latents_at(tr, depth, key_level)[:, key_node]
        xd = corrupt_hier(lv, rules, depth, v, m, s, ctx["level"], ctx["nodes"], rng)
        ok = (nearest_derivation_cost(rules, xd, r, s) > 0) if require_broken \
            else np.ones(len(need), bool)
        roots[need[ok]] = r[ok]
        x[need[ok]] = xd[ok]
        clean[need[ok]] = lv[ok]
        feat[need[ok]] = f[ok]
        need = need[~ok]
    if len(need):
        raise RuntimeError(f"context {ctx['name']}: {len(need)} instances never broke")
    return roots, x, clean, feat


# --------------------------------------------------------------------------- #
# exact, model-free readouts -- the substrate's gift
# --------------------------------------------------------------------------- #

def feature_leaves(rules, depth, level, f, canon_np):
    """The canonical leaf rendering of level-`level` feature `f`: expand by rule 0 all the way
    down. Any legal derivation of `f` gives the same possible-set verdict, so the canonical one
    is a sufficient probe."""
    feats = np.array([int(f)], np.int64)
    for lv in range(level, 1, -1):
        feats = rules[depth - lv][feats, 0].reshape(-1)
    return canon_np[feats].reshape(-1)


def entry_leaves(entry_flat, canon_np):
    """A table entry (a flat tuple of level-1 features) rendered canonically into leaves --
    exactly what `macros.apply_any` writes once the max-sum DP has chosen the entry."""
    return canon_np[np.asarray(entry_flat, np.int64)].reshape(-1)


def write_span(x_np, node, level, s, leaf_row):
    """Overwrite the level-`level` node's token span on every row."""
    span = s ** (level - 1)
    p0 = node * span * s
    out = x_np.copy()
    out[:, p0:p0 + span * s] = np.asarray(leaf_row, np.int64)[None, :]
    return out


def entry_success(rules, x_np, roots_np, leaf_row, node, level, s):
    """Exact: does writing this span repair the instance? (terminal possible-set success)."""
    xf = write_span(x_np, node, level, s, leaf_row)
    succ, _ = parse_success_and_heuristic(rules, xf, roots_np, s)
    return succ.astype(bool)


def entry_profile(rules, x_np, roots_np, entries_flat, node, level, s, canon_np):
    """(n_entries, n_instances) exact success matrix. The forced-transfer instrument: one row
    per committed program, one column per demanded instance, no learned grader anywhere."""
    if len(entries_flat) == 0 or x_np.shape[0] == 0:
        return np.zeros((0, x_np.shape[0]), bool), 0
    rows = [entry_success(rules, x_np, roots_np, entry_leaves(e, canon_np), node, level, s)
            for e in entries_flat]
    return np.stack(rows), len(entries_flat) * x_np.shape[0]


def consistent_features(rules, x_np, roots_np, node, level, s, canon_np, v):
    """(n_instances, v) -- which level-`level` features, written at `node`, repair the instance.

    This is the information content of the key, measured exactly. If every feature repairs
    every instance the wall names nothing worth naming; if exactly one does, the key is fully
    determined by the observation and the wall is redundant with a parse. GATE 0 reads this."""
    cols = []
    for f in range(v):
        cols.append(entry_success(rules, x_np, roots_np,
                                  feature_leaves(rules, len(rules), level, f, canon_np),
                                  node, level, s))
    return np.stack(cols, axis=1), v * x_np.shape[0]


# --------------------------------------------------------------------------- #
# the library: key -> index class -> committed table
# --------------------------------------------------------------------------- #

def dedup_union(tables, v, level, s, support=1):
    """The merged cell's content: the SET of distinct programs held by the merged cells.

    Not an average. `crystallize`'s law (averaging valid renditions destroys them, 3.6-5.0x)
    is a statement about blending one unit with another; a vocabulary is a set of units, and
    two cells that hold the same program contribute it once. That deduplication is where the
    index compression shows up in storage."""
    mn = MC.Miner(level, s)
    for t in tables:
        if t is None:
            continue
        for row in t["flat"]:
            mn.counts[tuple(int(x) for x in row)] = 1
    mn.n_obs = len(mn.counts)
    return mn.build(MC.base_table(v), support)


EARNED_BASE = 1000


class Library:
    """A lookup table from an observed key to a committed unit, plus the partition that key
    induces.

    Two families of cell live here. **Wall cells** are addressed by the surface key and their
    partition is given; `merge` is the only op that touches `self.cls`, and it can only delete
    distinctions. **Earned cells** (`EARNED_BASE +`) are addressed by a learned router onto
    classes the agent discovered by use — they are created and retired by `sync_earned`, keyed
    by the identity of the program that covers them, so a class that survives a re-key keeps
    the content it has been accumulating. Both families are ordinary cells otherwise: they
    mine, they commit, they recert."""

    def __init__(self, v, s, level, node, *, decay=1.0, floor=0.5, support=3,
                 ring_cap=48, single=False):
        self.v, self.s, self.level, self.node = int(v), int(s), int(level), int(node)
        self.support, self.ring_cap = int(support), int(ring_cap)
        self.decay, self.floor = float(decay), float(floor)
        self.cls = np.zeros(v, np.int64) if single else np.arange(v, dtype=np.int64)
        self.cells = OrderedDict()
        for c in sorted(set(int(x) for x in self.cls)):
            self.cells[c] = self._new_cell()
        self._wall = set(self.cells)
        self._earned = OrderedDict()          # program tuple -> earned cell id
        self._next_earned = EARNED_BASE
        self.n_merges = 0
        self.n_rekeys = 0
        self.n_retired = 0

    def _new_cell(self, active=True):
        return {"miner": DecayMiner(self.level, self.s, decay=self.decay, floor=self.floor),
                "table": None, "n_obs": 0, "committed_cycle": None,
                "ring_x": None, "ring_r": None, "ring_f": None, "n_swaps": 0,
                "active": bool(active), "n_reverts": 0,
                "maintained": True, "retired": None}

    # -- routing ---------------------------------------------------------------
    def classes_of(self, keys):
        return self.cls[np.asarray(keys, np.int64)]

    def class_ids(self):
        return sorted(self.cells)

    def wall_ids(self):
        return sorted(self._wall)

    def earned_ids(self):
        """Earned cells in a STABLE order (creation order), so router label indices mean the
        same thing from one re-key to the next."""
        return list(self._earned.values())

    def active_earned_ids(self):
        """Earned cells the revert net has not pulled. An inactive cell keeps mining and keeps
        its content — it simply stops being routed to, so a revert is a routing decision and
        never a loss of what was learned."""
        return [c for c in self._earned.values() if self.cells[c]["active"]]

    def set_active(self, c, flag):
        self.cells[c]["active"] = bool(flag)
        if not flag:
            self.cells[c]["n_reverts"] += 1

    def maintained(self, c):
        return bool(self.cells[c].get("maintained", True))

    def retire(self, c, mode):
        """The retire op, on an abandoned scaffold cell. Two settings of one knob:

        `mothball` stops all maintenance (no recert, no mining) but keeps the content and
        leaves the cell servable — the idea doc's routing-not-pruning prediction, that the
        wall-index persists as a fallback re-grounding beacon once it is no longer the primary
        address. `delete` removes it outright, so keys that pointed at it route to NO cell
        (class -1, the base action space) and the beacon is gone.

        Both stop maintenance, so mothball-vs-delete isolates the value of the retained
        CONTENT, and either-vs-`fw_s2`'s unretired `wall_rekey` isolates the rent of the
        MAINTENANCE."""
        self.cells[c]["maintained"] = False
        self.cells[c]["retired"] = mode
        if mode == "delete":
            self.cls[self.cls == c] = -1
            del self.cells[c]
            self._wall.discard(c)
        self.n_retired += 1
        return c

    def earned_programs(self):
        return list(self._earned)

    def members(self, c):
        return [int(k) for k in np.flatnonzero(self.cls == c)]

    # -- the re-key op: the index's basis, earned ------------------------------
    def sync_earned(self, programs, seed_tables=True, active=True):
        """Install `programs` (the cover) as the earned classes.

        Cells are keyed by PROGRAM IDENTITY, not by position, so a class that survives a
        re-key keeps the content it has been accumulating and only genuinely new classes start
        empty. Classes whose covering program has dropped out of the cover are retired. This
        is what makes the migration a migration rather than a rebuild."""
        programs = [tuple(int(z) for z in p) for p in programs]
        keep = set(programs)
        retired = [p for p in list(self._earned) if p not in keep]
        for p in retired:
            del self.cells[self._earned.pop(p)]
        created = []
        for p in programs:
            if p in self._earned:
                continue
            cid = self._next_earned
            self._next_earned += 1
            self._earned[p] = cid
            cell = self._new_cell(active=active)
            if seed_tables:
                mn = MC.Miner(self.level, self.s)
                mn.counts[p] = 1
                mn.n_obs = 1
                cell["table"] = mn.build(MC.base_table(self.v), 1)
            self.cells[cid] = cell
            created.append(cid)
        self.n_rekeys += 1
        return created, [1 for _ in retired]

    # -- entries ---------------------------------------------------------------
    def age(self):
        for cell in self.cells.values():
            cell["miner"].age()

    def observe(self, c, feats):
        feats = np.asarray(feats, np.int64)
        if feats.size == 0:
            return
        self.cells[c]["miner"].observe(feats)
        self.cells[c]["n_obs"] += int(feats.shape[0])

    def push_ring(self, c, x_np, r_np, f_np):
        """The cell's own recent consumption distribution, kept for free from practice draws.
        The latent goes in too, so a wall-reading selector can be auditioned under the wall it
        would actually see rather than under a sentinel."""
        cell = self.cells[c]
        if x_np.shape[0] == 0:
            return
        for key, val in (("ring_x", x_np), ("ring_r", r_np), ("ring_f", f_np)):
            cell[key] = val if cell[key] is None else \
                np.concatenate([cell[key], val])[-self.ring_cap:]

    def ring(self, c, n=None):
        cell = self.cells[c]
        if cell["ring_x"] is None:
            return None, None, None
        keys = ("ring_x", "ring_r", "ring_f")
        if n is None or cell["ring_x"].shape[0] <= n:
            return tuple(cell[k] for k in keys)
        return tuple(cell[k][-n:] for k in keys)

    def live(self, c):
        return self.cells[c]["miner"].build(MC.base_table(self.v), self.support)

    def table(self, c):
        return self.cells[c]["table"]

    def commit(self, c, table, cyc):
        self.cells[c]["table"] = table
        self.cells[c]["committed_cycle"] = int(cyc)

    def swap(self, c, table):
        self.cells[c]["table"] = table
        self.cells[c]["n_swaps"] += 1

    # -- the merge op ----------------------------------------------------------
    def merge(self, a, b):
        """Coarsen the index: classes a and b become one. Content is the deduplicated union;
        the miners' counts add, because the merged cell now genuinely sees both streams."""
        a, b = (a, b) if a < b else (b, a)
        ca, cb = self.cells[a], self.cells[b]
        for k, n in cb["miner"].counts.items():
            ca["miner"].counts[k] = ca["miner"].counts.get(k, 0.0) + n
        ca["miner"].n_obs += cb["miner"].n_obs
        ca["n_obs"] += cb["n_obs"]
        ca["n_swaps"] += cb["n_swaps"]
        if ca["table"] is not None or cb["table"] is not None:
            ca["table"] = dedup_union([ca["table"], cb["table"]], self.v, self.level, self.s)
            ca["committed_cycle"] = min(x for x in (ca["committed_cycle"],
                                                    cb["committed_cycle"]) if x is not None)
        for key in ("ring_x", "ring_r", "ring_f"):
            if cb[key] is not None:
                ca[key] = cb[key] if ca[key] is None else \
                    np.concatenate([ca[key], cb[key]])[-self.ring_cap:]
        self.cls[self.cls == b] = a
        del self.cells[b]
        self._wall.discard(b)
        self.n_merges += 1
        return a

    # -- readout ---------------------------------------------------------------
    def state(self):
        ids = self.class_ids()
        sizes = {str(c): (None if self.cells[c]["table"] is None
                          else int(self.cells[c]["table"]["child"].shape[0])) for c in ids}
        distinct = set()
        total = 0
        for c in ids:
            t = self.cells[c]["table"]
            if t is None:
                continue
            for row in t["flat"]:
                distinct.add(tuple(int(x) for x in row))
                total += 1
        wall = self.wall_ids()
        return {"n_classes": len(wall),
                "n_wall_maintained": sum(1 for c in wall if self.maintained(c)),
                "n_retired": int(self.n_retired),
                "n_earned": len(self._earned),
                "n_earned_active": len(self.active_earned_ids()),
                "n_reverts": int(sum(self.cells[c]["n_reverts"]
                                     for c in self._earned.values())),
                "n_all_cells": len(ids),
                "n_committed": sum(1 for c in ids if self.cells[c]["table"] is not None),
                "n_entries_total": int(total),
                "n_entries_distinct": int(len(distinct)),
                "sizes": sizes,
                "partition": {str(k): int(self.cls[k]) for k in range(self.v)},
                "n_merges": int(self.n_merges),
                "n_rekeys": int(self.n_rekeys),
                "n_obs": {str(c): int(self.cells[c]["n_obs"]) for c in ids},
                "n_swaps": {str(c): int(self.cells[c]["n_swaps"]) for c in ids}}


# --------------------------------------------------------------------------- #
# alias evidence: the exact transfer matrix over index classes
# --------------------------------------------------------------------------- #

def transfer_matrix(lib, rules, canon_np, *, n_screen=32, min_n=0, min_mass=0.0, ids=None):
    """`X[a][b]` = the fraction of class b's RECENT DEMAND that some entry of class a's
    committed table repairs -- exact, model-free, from the grammar's own DP.

    The diagonal is each class serving itself. A pair (a, b) whose off-diagonals match their
    diagonals is a pair of ALIASES: the distinction the index draws between them buys nothing,
    which is precisely the condition merge exists to detect. Returns (X, ids, n_graded), and
    `n_graded` is the price.

    `min_n` / `min_mass` are the **demand floor**, carried from `fw_s0`'s c15 diagnosis: a
    3.1%-mass cell merged on a ring of ~30 where both rates read 0.41 and the standard error
    on the comparison was ~0.09, i.e. the criterion fired well inside its own noise. A class
    now enters the screen only once its ring is deep enough to estimate the rate and it carries
    enough of the demand to be worth an index slot. The floor is on EVIDENCE, not on
    performance: a stale cell serving 0% of its own demand must still be able to merge, which
    is the whole point after a rotation."""
    if ids is None:
        ids = lib.wall_ids()
    ring_n = {c: (0 if lib.ring(c, n_screen)[0] is None else lib.ring(c, n_screen)[0].shape[0])
              for c in ids}
    tot = max(sum(ring_n.values()), 1)
    ids = [c for c in ids if lib.table(c) is not None and ring_n[c] >= min_n
           and ring_n[c] / tot >= min_mass]
    n = len(ids)
    X = np.full((n, n), np.nan)
    graded = 0
    entries = {c: [tuple(int(x) for x in r) for r in lib.table(c)["flat"]] for c in ids}
    for j, b in enumerate(ids):
        xb, rb, _fb = lib.ring(b, n_screen)
        for i, a in enumerate(ids):
            prof, g = entry_profile(rules, xb, rb, entries[a], lib.node, lib.level, lib.s,
                                    canon_np)
            graded += g
            X[i, j] = float(prof.any(axis=0).mean()) if prof.shape[0] else 0.0
    return X, ids, graded


def merge_candidates(X, ids, tol):
    """Every pair whose two-directional transfer loss is within `tol`, best first."""
    out = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if not np.isfinite(X[i, j]) or not np.isfinite(X[j, i]):
                continue
            loss = max(X[j, j] - X[i, j], X[i, i] - X[j, i])
            if loss <= tol:
                out.append({"a": ids[i], "b": ids[j], "loss": float(loss),
                            "x_ab": float(X[i, j]), "x_bb": float(X[j, j]),
                            "x_ba": float(X[j, i]), "x_aa": float(X[i, i])})
    return sorted(out, key=lambda d: d["loss"])
