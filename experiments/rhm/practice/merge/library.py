"""The keyed library and the MERGE op: the first op in the arc that acts on the INDEX.

WHY THIS EXISTS. Every op the practice arc already has -- mine, select, commit, recert,
reselect -- acts on an ENTRY under a fixed index. `merge`
(`ideas/recurrence_manufactures_confounds.md` Sec 5) acts on the index itself: the library is a
lookup from observed context to committed unit, any choice of key induces a partition of
situations into cells, and merge COARSENS that partition by discovering which context
distinctions do not matter.

THE MANUFACTURED CONFOUND, on the setlist machinery. A task instance is announced with two free
observables:

  * a VENUE `w` -- the room, the wall, the recording. Each venue carries its own OU demand state
    (`setlist/demand.py`, imported unmodified): a private root prior and private mixture weights
    on the rule layers above the mined vocabulary. Under a NARROW demand (large sigma) a venue
    concentrates on a few derivations, so `w` is a nearly-perfect free name for the level-2
    feature the damage destroyed -- a latent the agent would otherwise have to infer. The venue
    is spuriously correlated with the true latent BECAUSE the sampling policy holds it fixed,
    which is exactly Sec 2's debt side of allocation. Under sigma = 0 every venue is the same
    uniform demand, `w` carries no information, and decorrelation is free -- the flow-of-
    experience regime.
  * a NODE `j` -- which level-2 node the damage hit. The frame.

The key is `(w, j)`; the index is a partition of the key space. `track` never coarsens (coverage
by ENUMERATION -- one entry per cell); `merge` coarsens by forced transfer; `global` is handed
the fully-merged index at birth (the quotient, no scaffold).

WHAT MERGE IS NOT. Compression on the INDEX, never on the content (Sec 5, and the etude's law
that averaging valid renditions destroys them, 2.1x etude / 3.6-5.0x crystallize). A merged cell
keeps ONE representative table -- the one that survives the alias audit -- never a blend and
never an average. `merge_rep="rep"` is that literal reading; `merge_rep="pool"` is the variant
that pools the two cells' mining evidence and rebuilds (still no averaging: a table is a SET of
exact level-1 tuples, so pooling is set union on entries, not a smear of one entry). Both are
run, so "does the quotient need evidence pooling or only index coarsening" is measured rather
than assumed.

THE EVIDENCE ROUTE. Sec 5 gives two: (a) audit of aliases -- compare populated entries under
different keys, expensive, requires having paid to populate them; (b) forced transfer under
cache miss -- execute cell a's entry under cell b's frame and read the success profile, from one
probe, without ever populating b. Route (b) is what the dance runs and what `merge_pass`
implements: the K x K TRANSFER MATRIX `S[a][k] = success of cell a's table on probes drawn from
key k`, graded by the substrate's own possible-set DP (exact, model-free, no learned grader in
the loop). Route (a) is `alias_jaccard`, kept as the offline comparison.

Nothing in `../setlist/`, `../crystallize/` or `../ratchet/` is modified; everything here either
imports them or is new.
"""

import numpy as np

from rhm.practice.ratchet import macros as MC
from rhm.practice.setlist import demand as DM


# --------------------------------------------------------------------------- #
# venues: one OU demand state each
# --------------------------------------------------------------------------- #

def build_venues(rules, levels, n_venue, s, depth, v, m, inv_bottom, *, seed=0, kappa=0.15,
                 sigma=1.0, n_cand=0, n_hist=2048, level=2):
    """`n_venue` independent demand states, one per venue.

    Each is `demand.new_demand` prewarmed to stationarity (D-5), with `typical_demand`'s
    representative-start fix applied per venue when `n_cand > 0` -- the same lottery `cal0`
    caught for a single state applies venue-by-venue here, and a venue that starts atypically
    concentrated would look like an unusually loud wall for a reason that has nothing to do
    with sigma.

    At sigma = 0 every state is theta = 0, i.e. exactly uniform demand, and every venue is the
    same world: the free-i.i.d.-variation regime with the venue tag carrying zero information.
    """
    out, info = [], []
    for w in range(n_venue):
        if sigma > 0 and n_cand:
            st, rec = DM.typical_demand(rules, levels, s, depth, v, m, inv_bottom,
                                        seed=seed + 1_000 * (w + 1), kappa=kappa, sigma=sigma,
                                        n_cand=n_cand, n=n_hist, level=level)
            info.append(rec)
        else:
            st = DM.new_demand(rules, levels, seed=seed + 1_000 * (w + 1), kappa=kappa,
                               sigma=sigma, prewarm=True)
            info.append({})
        out.append(st)
    return out, info


def rotate_venues(venues, rng, n_steps=1):
    """One paced rotation event: every venue's OU takes `n_steps` steps. Mean-reverting, so the
    ensemble of tastes wanders but does not run away, and a taste can come back into fashion
    (the caveat setlist recorded: per-epoch trajectories, not endpoints)."""
    for st in venues:
        DM.demand_step(st, rng, n_steps)
    return venues


# --------------------------------------------------------------------------- #
# the demand a (venue, node) cell actually faces
# --------------------------------------------------------------------------- #

def node_hist(rules, st, node, level, s, depth, v, m, inv_bottom, n=4096, seed=0):
    """P(entry) over the level-`level` vocabulary AT ONE NODE under one venue's demand.

    `demand.demand_hist` pools over every node of the level, matching `ratchet.macro_moves`'
    position-independence. Here the node is part of the KEY, so the histogram has to be read
    per node: that is the whole point of asking whether the node distinction matters."""
    _r, lv = DM.sample_pool_demand(rules, n, s, seed, st)
    feats = MC.exact_features(lv, inv_bottom, v, s)                  # (n, n_blocks)
    span = s ** (level - 1)
    spans = feats[:, node * span:(node + 1) * span]
    keys, counts = np.unique(spans, axis=0, return_counts=True)
    tot = counts.sum()
    return {tuple(int(x) for x in k): c / tot for k, c in zip(keys, counts)}


def mutual_info(joint):
    """I(w; f) in nats from a dict {(w, f): count}. The AVAILABLE side of Sec 2's
    available-vs-taken gap: how much the free observable could tell you about the latent."""
    tot = sum(joint.values())
    if tot == 0:
        return 0.0
    pw, pf = {}, {}
    for (w, f), c in joint.items():
        pw[w] = pw.get(w, 0) + c
        pf[f] = pf.get(f, 0) + c
    out = 0.0
    for (w, f), c in joint.items():
        p = c / tot
        out += p * np.log(p / ((pw[w] / tot) * (pf[f] / tot)))
    return float(out)


def venue_latent_joint(rules, venues, nodes, level, s, depth, v, m, inv_bottom, n=2048, seed=0):
    """{(venue, level-`level` entry): count} pooled over `nodes`. Feeds `mutual_info`."""
    joint = {}
    span = s ** (level - 1)
    for w, st in enumerate(venues):
        _r, lv = DM.sample_pool_demand(rules, n, s, seed + 137 * w, st)
        feats = MC.exact_features(lv, inv_bottom, v, s)
        for node in nodes:
            for row in feats[:, node * span:(node + 1) * span]:
                k = (w, tuple(int(x) for x in row))
                joint[k] = joint.get(k, 0) + 1
    return joint


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #

class KeyedLibrary:
    """key -> cell -> committed table. `policy` fixes what the index is allowed to do.

      track   the finest partition, forever. One entry per observed key: coverage by
              ENUMERATION, carrying Sec 8's storage / per-key curse / coverage costs.
      merge   starts at the finest partition and coarsens by forced transfer.
      global  one cell from birth -- the quotient handed over, no scaffold, no index event.
    """

    def __init__(self, keys, level, s, policy="track", merge_rep="rep"):
        self.level, self.s = level, s
        self.policy = policy
        self.merge_rep = merge_rep
        self.keys = list(keys)
        if policy == "global":
            self.owner = {k: 0 for k in self.keys}
            self.cells = {0: {"keys": set(self.keys), "miner": MC.Miner(level, s),
                              "table": None}}
        else:
            self.owner = {k: i for i, k in enumerate(self.keys)}
            self.cells = {i: {"keys": {k}, "miner": MC.Miner(level, s), "table": None}
                          for i, k in enumerate(self.keys)}
        self.history = []          # every merge event, for the index-granularity trace

    # -- reads ------------------------------------------------------------- #

    def n_cells(self):
        return len(self.cells)

    def cell_of(self, key):
        """The cell serving `key`, or None -- a CACHE MISS, which is what Sec 5's route (b)
        needs to exist for the forced transfer to be forced."""
        return self.owner.get(key)

    def fallback_cell(self):
        """What a keyed library does at a key it has no entry for: reach for the cell with the
        most evidence behind it. There is no principled choice here and that is the finding --
        `analyze` also reports the oracle-best and worst cells, so the miss is a RANGE."""
        if not self.cells:
            return None
        return max(self.cells, key=lambda c: self.cells[c]["miner"].n_obs)

    def table_for(self, key):
        c = self.cell_of(key)
        if c is None:
            c = self.fallback_cell()
        return (None if c is None else self.cells[c]["table"]), c

    def storage(self):
        """Sec 8's carried cost of never merging: total entries held across all cells."""
        return int(sum(0 if c["table"] is None else c["table"]["child"].shape[0]
                       for c in self.cells.values()))

    # -- writes ------------------------------------------------------------ #

    def observe(self, key, feats):
        c = self.cell_of(key)
        if c is None:
            return
        self.cells[c]["miner"].observe(feats)

    def rebuild(self, lower, support, only=None):
        """Rebuild every cell's table from its miner."""
        for cid, cell in self.cells.items():
            if only is not None and cid not in only:
                continue
            cell["table"] = cell["miner"].build(lower, support)

    def merge(self, a, b, keep):
        """Coarsen: cells `a` and `b` become one. `keep` in {a, b} names the REPRESENTATIVE.

        The surviving cell inherits both key sets, and from here on BOTH key streams route into
        its single miner -- routing, not pruning (Sec 3.5 of the parent doc). The two modes
        differ only in what happens to the evidence the loser had already paid for:

          rep   the loser's counts are DISCARDED. The merged cell's content, at the instant of
                the merge, is exactly the representative's -- never a blend, never an average,
                which is Sec 5's constraint read literally. De-walling a dance keeps the frame
                that works and throws the redundant copies away.
          pool  the counts are summed and the table rebuilt. Still no averaging -- a table is a
                SET of exact level-1 tuples, so pooling is set union on entries, not a smear of
                one entry -- but the merge now inherits what the other cell had learned.

        Running both is how "does the quotient need evidence pooling, or only index coarsening"
        gets measured instead of assumed.
        """
        assert a in self.cells and b in self.cells and keep in (a, b)
        drop = b if keep == a else a
        kc, dc = self.cells[keep], self.cells[drop]
        if self.merge_rep == "pool":
            for k, n in dc["miner"].counts.items():
                kc["miner"].counts[k] = kc["miner"].counts.get(k, 0) + n
            kc["miner"].n_obs += dc["miner"].n_obs
        kc["keys"] |= dc["keys"]
        for k in dc["keys"]:
            self.owner[k] = keep
        del self.cells[drop]
        self.history.append({"keep": keep, "drop": drop, "n_cells": len(self.cells)})
        return keep


def alias_jaccard(t_a, t_b):
    """Sec 5's evidence route (a), the AUDIT: how much two populated cells' entry sets agree.
    Offline, and it requires having paid to populate both -- which is why route (b) is the one
    the loop actually runs."""
    if t_a is None or t_b is None:
        return None
    A = {tuple(int(x) for x in r) for r in t_a["flat"]}
    B = {tuple(int(x) for x in r) for r in t_b["flat"]}
    if not A and not B:
        return 1.0
    return float(len(A & B) / max(1, len(A | B)))


# --------------------------------------------------------------------------- #
# the merge op itself
# --------------------------------------------------------------------------- #

def merge_pass(lib, score_fn, *, tau=0.95, min_obs=1, max_merges=None, verbose=False):
    """One merge pass by FORCED TRANSFER (Sec 5, route b).

    `score_fn(table, key) -> success in [0, 1]`: execute `table` as the level-`lib.level` macro
    on fresh probe instances drawn under `key`'s venue and node, and grade by the grammar's own
    possible-set DP. Exact and model-free.

    Build S[a][b] = score(table of cell a, demand of cell b) over live cells. Two cells are
    ALIASES iff each serves the other's demand at least `tau` of what the other's own entry
    serves it -- symmetric, so a merge never happens because one cell is simply better.
    Merge greedily, most-aliased pair first; the survivor is the cell whose table scores higher
    on the OBSERVATION-WEIGHTED pool of the two demands.

    The matrix is computed once per pass and reused as cells collapse (the survivor keeps its
    own row and column), so a pass costs O(K^2) probes, not O(K^3).
    """
    live = [c for c, cell in lib.cells.items()
            if cell["table"] is not None and cell["table"]["child"].shape[0] > 0
            and cell["miner"].n_obs >= min_obs]
    if len(live) < 2:
        return {"n_pairs": 0, "n_merges": 0, "n_cells": lib.n_cells(), "S": {}, "events": []}

    rep_key = {c: sorted(lib.cells[c]["keys"])[0] for c in live}
    S = {a: {b: float(score_fn(lib.cells[a]["table"], rep_key[b])) for b in live} for a in live}

    events, n_merges = [], 0
    while True:
        best, best_gap = None, None
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if a not in lib.cells or b not in lib.cells:
                    continue
                ok_ab = S[a][b] >= tau * S[b][b]
                ok_ba = S[b][a] >= tau * S[a][a]
                if not (ok_ab and ok_ba):
                    continue
                gap = min(S[a][b] - tau * S[b][b], S[b][a] - tau * S[a][a])
                if best_gap is None or gap > best_gap:
                    best, best_gap = (a, b), gap
        if best is None or (max_merges is not None and n_merges >= max_merges):
            break
        a, b = best
        na = max(1, lib.cells[a]["miner"].n_obs)
        nb = max(1, lib.cells[b]["miner"].n_obs)
        pool_a = (S[a][a] * na + S[a][b] * nb) / (na + nb)
        pool_b = (S[b][a] * na + S[b][b] * nb) / (na + nb)
        keep = a if pool_a >= pool_b else b
        drop = b if keep == a else a
        lib.merge(a, b, keep)
        events.append({"a": a, "b": b, "keep": keep, "gap": float(best_gap),
                       "s_ab": S[a][b], "s_ba": S[b][a], "s_aa": S[a][a], "s_bb": S[b][b],
                       "pool_keep": float(max(pool_a, pool_b))})
        n_merges += 1
        live = [c for c in live if c != drop]
        if verbose:
            print(f"    [merge] {a}+{b} -> {keep} gap={best_gap:.3f} "
                  f"cells={lib.n_cells()}", flush=True)
    return {"n_pairs": len(S) ** 2, "n_merges": n_merges, "n_cells": lib.n_cells(),
            "S": {str(a): {str(b): float(x) for b, x in row.items()} for a, row in S.items()},
            "events": events}


# --------------------------------------------------------------------------- #
# next-level minability: Sec 16's currency, read straight off a representation
# --------------------------------------------------------------------------- #

def next_level(table_l2, miner_l3, support):
    """Build the level-3 table over a level-2 vocabulary. `ratchet`'s nesting is the mechanism:
    `T[3]` is defined over `T[2]` ENTRIES, so a level-3 span whose halves are not both in the
    level-2 vocabulary is not merely worse, it is UNREPRESENTABLE and is dropped at build time.
    A thin per-cell level-2 table therefore caps what the next level can represent at all --
    which is what "what does each arm's representation make representable next" measures."""
    return miner_l3.build(table_l2, support)
