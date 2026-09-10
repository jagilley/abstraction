"""[enharmonic] THE QUOTIENT: the class map, the class-keyed miner, and their offline gates.

Imported by `enharmonic.py`, not forked into it — the idiom `maestro/policy.py` and
`antiphon/questions.py` already have, and for the same reason: every rule here is a pure
function of the DGP and the observations, so it is gate-able with no GPU and no substrate.

WHAT A CLASS IS HERE. Two supplied quotients, both single-valued functions of an observed
half, both oracle reads (they consult `rules`) and therefore only reachable in a `given_cat`
arm:

  'tok'  THE TOKEN CLASS. The set of level-l features that the half's CANONICAL LEAF
         RENDERING — `canon[feats]`, exactly what `macros.apply_any` writes — is a legal
         derivation of, read off the grammar's own possible-set DP. `sizing/SIZING.md` §5(a):
         a forced-transfer probe measures this and nothing else (397/397 groups bit-identical
         over 4,096 instances), so it is the operational synonymy relation on this world and
         the EARNABLE ceiling. It is closed under composition (the alphabet is a fixed point
         of the DP's own step), which is what lets `C[l+1] subset C[l] x C[l]` be the ratchet.
         Alphabet 7 / 9 / 11 / 13 / 22 / 42 at L1..L6; legal class pairs 13 / 28 / 40 / 73 /
         306 at L2..L6.

  'gen'  THE MIN-LABEL (OVER-QUOTIENT) KEY — `fourwall`'s re-key basis, v = 8 classes and
         14-15 legal pairs per rung. It is the TOKEN CLASS COARSENED TO ONE FEATURE: the same
         possible-set, tie-broken to its minimum element. On a half the world is unambiguous
         about, that IS the generative parent; on a half it is not, the tie-break MERGES an
         ambiguous tuple with an unambiguous one — a distinction the exact grader can see,
         deliberately collapsed. So the arm it drives (`given_cat_min`) is an OVER-QUOTIENT
         control, not a giftability control: it supplies nothing 'tok' does not also supply,
         and it is read as the parent spec's thread-6 merge-precision ablation one rung up.
         The true relation is a COVER, not a partition —
         2 / 8 / 128 / 37,120 distinct flat tuples at L2..L5 carry more than one parent and
         they carry 23-34% of the true observation mass at every mining node (SIZING.md §1) —
         so the tie-break is stated here and counted at run time (`n_tie`).

         DELIBERATELY built as a coarsening of 'tok' rather than as `feat_parent_sets` on the
         raw feature string. Both are single-valued oracle maps, but a feature-space parent map
         is EMPTY wherever the string is outside the true table, which at the L5 mining node is
         96.6% of observations — so the two arms would then differ in WHICH HALVES THEY CAN
         NAME AT ALL as well as in granularity, and the contrast would no longer be one thing.
         As built, 'tok' and 'gen' share their drop set exactly and differ only in how finely
         they cut what they keep, which is the comparison the arm exists for.

  'singleton'  the class of a half IS the half. Gate-only: with it the class-keyed miner is
         `MC.Miner` exactly, entry for entry and in the same row order, which is the
         0.000e+00 identity the fork's quotient path is certified on (`quotient_gate` G-1).

HOW THE CLASS KEY IS COUNTED AND BUILT. `MC.Miner` keys its counts by the flattened level-1
tuple and applies the ratchet at build time by looking each half up in the lower table's flat
rows. `ClassMiner` keeps that split exactly and moves only the key:

  observe   TABLE-FREE, like the donor's. The key is the PAIR OF HALF-CLASSES. This is the
            line that buys the arrival win — SIZING.md §3 models support accruing on the
            class-pair key (E[keys at support 3] = 63.3 of 70 at L5 by the run's own ~1,600
            observations, against 0.000 true flat keys) — and it only materialises if the
            counts are literally kept this way.
  build     the ratchet, unchanged in kind: a class-pair survives iff the lower table holds at
            least one row of each half-class. What it emits is the CROSS-PRODUCT of the
            lower table's rows in c1 with its rows in c2, so the committed table is still an
            ordinary flat table and `macros.macro_features` is untouched — the DP's max over
            that inventory IS the expansion choice a class token needs. The inventory is
            capped at `spell_cap` spellings per class, ranked by the miner's own observed
            count for that spelling and then by lower-table row order; the true expansion sets
            are 2 / 8 / 128 / 31,744 at L2..L5 (SIZING.md §2), so any cap is a truncation and
            the binding is logged. With singleton classes every class holds exactly one
            spelling, so the cap can never bind and the identity is exact.

Row order is `sorted(counts)` over class-pair keys, then the cross-product in (lower-row,
lower-row) order — and in singleton mode the class key of a half IS its tuple, so the sort is
lexicographic on (h1, h2), which for equal-length halves is the donor's lexicographic sort on
h1 + h2. That is why G-1 is bit-identical and not merely set-equal.
"""

import collections

import numpy as np

from rhm.practice.ratchet import macros as MC
from rhm.rhm_sculpt_precheck import possible_sets


# --------------------------------------------------------------------------- #
# the two class maps
# --------------------------------------------------------------------------- #

def feat_parent_sets(rules, feats, level, v, s, depth):
    """`possible_sets`' own recursion with the bottom layer replaced by the identity: given
    level-1 FEATURE strings (N, s**(level-1)), which level-`level` features they derive.

    This is `MC.true_tables`' `feature` column computed on the fly and table-free, which is
    what makes the 'gen' map usable at L5/L6 where the true table has 2.6e5 rows and the
    level above cannot be enumerated at all. `quotient_gate` G-2 asserts the agreement."""
    feats = np.asarray(feats, np.int64)
    n = feats.shape[0]
    P = np.zeros((n, feats.shape[1], v), bool)
    np.put_along_axis(P, feats[:, :, None], True, axis=2)
    for ell in range(2, level + 1):
        layer = rules[depth - ell]                       # (v, m, s)
        n_par = P.shape[1] // s
        kids = P.reshape(n, n_par, s, v)
        ok = np.ones((n, n_par, v, layer.shape[1]), bool)
        for i in range(s):
            ok &= kids[:, :, i, :][:, :, layer[:, :, i]]
        P = ok.any(-1)
    return P                                             # (N, n_nodes, v)


def token_class_sets(rules, feats, level, canon, v, s, depth):
    """The set of level-`level` features the CANONICAL LEAF RENDERING of each level-1 feature
    string is a legal derivation of. The grammar's own `possible_sets`, on the tokens
    `macros.apply_any` would write."""
    feats = np.asarray(feats, np.int64)
    leaves = canon[feats].reshape(feats.shape[0], -1)
    return possible_sets(rules[depth - level:], leaves, s)[-1]     # (N, n_nodes, v)


class Quotient:
    """The supplied class map, and the only place an `enharmonic` arm reads the oracle.

    Every call is counted (`n_read` rows, `n_drop` classless halves, `n_tie` multiply-parented
    halves under 'gen'), and the counters are logged per arm, so "how much oracle did this arm
    consume" is a number in the record rather than an argument — `pose_questions`' containment
    idiom, applied to the object this node supplies."""

    MODES = ("tok", "gen", "singleton")

    def __init__(self, mode, rules, canon, v, s, depth):
        if mode not in self.MODES:
            raise ValueError(f"unknown quotient mode {mode!r}; expected one of {self.MODES}")
        self.mode = mode
        self.rules, self.v, self.s, self.depth = rules, int(v), int(s), int(depth)
        self.canon = np.asarray(canon, np.int64)
        self.n_read = 0
        self.n_drop = 0
        self.n_tie = 0
        self._cache = {}

    # -- the map ------------------------------------------------------------------------ #
    def ids(self, feats, level):
        """feats (N, s**(level-1)) level-1 features -> a list of N class keys, `None` where the
        half has no class. `level` is the level the STRING is (a half of a level-(level+1)
        key), not the level being mined."""
        feats = np.asarray(feats, np.int64).reshape(len(feats), -1)
        self.n_read += feats.shape[0]
        if self.mode == "singleton":
            return [tuple(int(x) for x in r) for r in feats]
        if self.mode == "tok":
            P = token_class_sets(self.rules, feats, level, self.canon,
                                 self.v, self.s, self.depth)[:, 0, :]
            out = []
            for row in P:
                nz = np.nonzero(row)[0]
                if not nz.size:
                    self.n_drop += 1
                    out.append(None)
                else:
                    out.append(int((1 << nz).sum()))     # a bitmask over the v features
            return out
        P = token_class_sets(self.rules, feats, level, self.canon,
                             self.v, self.s, self.depth)[:, 0, :]
        out = []
        for row in P:
            nz = np.nonzero(row)[0]
            if not nz.size:
                self.n_drop += 1
                out.append(None)
            else:
                if nz.size > 1:
                    self.n_tie += 1
                out.append(int(nz[0]))                   # MIN PARENT, stated and counted
        return out

    def id_of(self, tup, level):
        """One tuple, memoised — used at build time over the lower table's rows, which are the
        same few dozen tuples every cycle."""
        key = (level, tup)
        if key not in self._cache:
            self._cache[key] = self.ids(np.asarray([tup], np.int64), level)[0]
        return self._cache[key]

    def state(self):
        return {"mode": self.mode, "n_read": int(self.n_read), "n_drop": int(self.n_drop),
                "n_tie": int(self.n_tie), "n_cached": len(self._cache)}


# --------------------------------------------------------------------------- #
# the class-keyed miner
# --------------------------------------------------------------------------- #

class ClassMiner(MC.Miner):
    """`MC.Miner` with the key quotiented. Subclassed rather than rewritten so every instrument
    that reaches into a miner — `_install_identity_miner`'s `state` patch, `gauge_hist`,
    `obs_hist`, the question port's `counts` reads — keeps working unchanged."""

    def __init__(self, level, s, quot, spell_cap=4):
        super().__init__(level, s)
        self.quot = quot
        self.spell_cap = int(spell_cap)
        self.spell = collections.defaultdict(collections.Counter)   # class -> spelling counts
        self.n_rows = 0
        self.n_dropped = 0
        self.last_build = {}

    def observe(self, feats):
        feats = np.asarray(feats, np.int64)
        if feats.size == 0:
            return
        half = self.span // self.s
        halves = [feats[:, i * half:(i + 1) * half] for i in range(self.s)]
        ids = [self.quot.ids(h, self.level - 1) for h in halves]
        self.n_obs += feats.shape[0]
        for r in range(feats.shape[0]):
            key = tuple(ids[i][r] for i in range(self.s))
            self.n_rows += 1
            if any(k is None for k in key):
                self.n_dropped += 1
                continue
            self.counts[key] = self.counts.get(key, 0) + 1
            for i in range(self.s):
                self.spell[key[i]][tuple(int(x) for x in halves[i][r])] += 1

    def build(self, lower, support):
        """The ratchet, on classes: a class-pair survives iff the lower table holds a row of
        each half-class. What is emitted is the capped cross-product of those rows."""
        # THE WIDTH CONTRACT, and why it is a filter rather than an assertion. `MC.Miner.build`
        # looks a key half of width `span // s` up in `lower["flat"]` through a dict, so a lower
        # table of the WRONG width simply misses on every key and yields an empty table. The
        # donor relies on that: the audition block's "live vocabulary" counterfactual calls
        # `miners[ell - 1].build(MC.base_table(v), ...)` — a level-(ell-1) miner over a LEVEL-1
        # table — and silently gets nothing back. A class miner asked for the class of a
        # width-1 row at level 3 instead crashes inside `possible_sets`. So rows whose width is
        # not the half width are dropped here, which reproduces the donor's `lut` semantics
        # exactly and keeps the fix inside this file (the fork's own code path is untouched, so
        # G-F stays closed). Gate G-6.
        want = self.span // self.s
        lower_flat = [tuple(int(x) for x in row) for row in lower["flat"]
                      if len(row) == want]
        # `MC.Miner.build`'s own index: one row per DISTINCT flat tuple, last writer wins. Two
        # lower rows with the same flat tuple are the same program (the true tables carry such
        # duplicates), and the donor's `lut` already collapses them — so the class inventory is
        # over distinct SPELLINGS, which is what makes the singleton identity exact.
        lut = {row: i for i, row in enumerate(lower_flat)}
        by_class = collections.defaultdict(list)
        for row, i in lut.items():
            c = self.quot.id_of(row, self.level - 1)
            if c is not None:
                by_class[c].append(i)
        for c in by_class:
            by_class[c].sort()

        def pick(c):
            ix = by_class.get(c, [])
            if len(ix) <= self.spell_cap:
                return ix, False
            cnt = self.spell.get(c, {})
            ranked = sorted(ix, key=lambda i: (-cnt.get(lower_flat[i], 0), i))
            return sorted(ranked[:self.spell_cap]), True

        rows, n_at_sup, n_kept, inv, capped = [], 0, 0, [], 0
        # [en_s5] the MASS on at-support keys, and on the subset the ratchet can actually
        # realise as rows. `en_s4` c188 measured a merge whose whole effect was to move keys
        # from the first set into the second, which no arrival gauge can see.
        mass_tot = max(sum(self.counts.values()), 1)
        mass_sup = mass_built = 0
        for key, c in sorted(self.counts.items()):
            if c < support:
                continue
            n_at_sup += 1
            mass_sup += c
            picks = []
            for i in range(self.s):
                ix, cap = pick(key[i])
                if not ix:
                    picks = None
                    break
                capped += int(cap)
                picks.append(ix)
            if picks is None:
                continue
            n_kept += 1
            mass_built += c
            combos = [[]]
            for ix in picks:
                combos = [pre + [j] for pre in combos for j in ix]
            inv.append(len(combos))
            rows.extend(combos)
        self.last_build = {"n_at_support": int(n_at_sup), "n_keys_built": int(n_kept),
                           "n_entries": int(len(rows)),
                           "inventory_max": int(max(inv) if inv else 0),
                           "inventory_mean": (float(np.mean(inv)) if inv else 0.0),
                           "n_class_capped": int(capped),
                           "n_lower_classes": int(len(by_class)),
                           "n_lower_rows": int(len(lower_flat)),
                           # [en_s5] the at-support mass, and the share of it the ratchet can
                           # realise as rows. `en_s4` c188: a merge whose whole effect was to
                           # move keys from the first into the second. Logged per build so the
                           # next round can derive a floor for it instead of stating one.
                           "mass_at_support": float(mass_sup / mass_tot),
                           "mass_buildable": float(mass_built / mass_tot),
                           # [figured_bass] THE PICK, RECORDED. `pick()` ranks a class's lower
                           # rows by `self.spell` before the `spell_cap` truncation, and those
                           # counts were in no banked tag -- so `figured_bass/sizing`'s offline
                           # replay could reproduce a build's row COUNT and (under 'tok') its
                           # class coverage exactly, but never its row IDENTITY (gate F-3).
                           # These two fields close that for good and cost ~200 bytes a cycle:
                           # `picks` is the surviving lower-row indices per class, which IS the
                           # build up to the cross-product, and `n_rows_in_class` says where
                           # the cap bit. Indices are into `lower_flat` (the width-filtered,
                           # dedup'd lower rows), which a replay reconstructs itself. Keys are
                           # `str(class id)` because an endogenous id is a tuple.
                           "picks": {str(c_): [int(i) for i in pick(c_)[0]]
                                     for c_ in sorted(by_class, key=str)},
                           "n_rows_in_class": {str(c_): int(len(ix_))
                                               for c_, ix_ in sorted(by_class.items(),
                                                                     key=lambda kv: str(kv[0]))}}
        child = np.array(rows, np.int64).reshape(len(rows), self.s)
        return MC.make_table(self.level, child, lower, self.s)

    def snapshot(self):
        """[en_s3] The counts and the spellings, copied, so a TRIAL rekey can be undone.

        `rekey` is destructive — pooled counts cannot be un-pooled — and the whole-partition
        merge rekeys the next level's miner to price a group it may then refuse. The undo has
        to restore the miner exactly, not approximately, or a refused group would leave the
        licence reading a coarser stream than the op installed."""
        return ({k: int(v) for k, v in self.counts.items()},
                {k: collections.Counter(v) for k, v in self.spell.items()})

    def restore(self, snap):
        counts, spell = snap
        self.counts = {k: int(v) for k, v in counts.items()}
        self.spell = collections.defaultdict(collections.Counter,
                                             {k: collections.Counter(v)
                                              for k, v in spell.items()})

    def rekey(self, quot=None):
        """[Q2] Re-key every accrued count under the CURRENT class map, summing collisions.

        A merge coarsens the map, so keys that were distinct become one and their counts add —
        which is `Library.merge`'s "the miners' counts add", on this substrate's key. The
        observation stream is table-free and never replayed: the counts ARE the record, so
        re-keying them is what makes a merge retroactive rather than only forward-looking.
        Returns (n_keys_before, n_keys_after)."""
        if quot is not None:
            self.quot = quot
        before = len(self.counts)
        new = {}
        for key, c in self.counts.items():
            k2 = tuple(self.quot.id_of(x, self.level - 1) if isinstance(x, tuple) else x
                       for x in key)
            new[k2] = new.get(k2, 0) + c
        self.counts = new
        sp = collections.defaultdict(collections.Counter)
        for cid, cnt in self.spell.items():
            c2 = self.quot.id_of(cid, self.level - 1) if isinstance(cid, tuple) else cid
            sp[c2].update(cnt)
        self.spell = sp
        return before, len(self.counts)

    def spell_top(self, k=8):
        """[figured_bass] The `spell` ranking `build`'s `pick()` reads, in the form that makes
        an offline replay of a build exact rather than bracketed.

        `self.spell[c]` is a Counter over every half of class `c` this miner has observed; the
        full map is far too big to log per cycle (thousands of distinct width-8 halves at L4),
        and only its HEAD is ever consulted -- `pick()` takes the top `spell_cap` by
        (-count, lower-row index) and index-orders the count-0 tail. `k` defaults to 8, twice
        the production cap, so the ranking that decided a pick is on the record beside the pick
        itself (`last_build["picks"]`). Keys are `str(class id)`; a spelling is a list of ints.
        """
        out = {}
        for c, cnt in self.spell.items():
            top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:int(k)]
            out[str(c)] = {"n_distinct": int(len(cnt)),
                           "top": [[list(sp), int(n)] for sp, n in top]}
        return out

    def state(self):
        """The donor's fields, plus a JSON-safe `keys_at_support` over CLASS PAIRS (the donor's
        is over flat tuples) and the drop accounting. `_install_identity_miner` patches
        `MC.Miner.state` to add `keys_at_support`; this override supersedes it for a class
        miner, whose keys are not level-1 feature tuples."""
        sup = None
        out = {"level": self.level, "n_obs": self.n_obs, "n_distinct": len(self.counts),
               "n_at_support": {str(t): int(sum(1 for c in self.counts.values() if c >= t))
                                for t in (1, 2, 3, 5, 10)},
               "keyed_by": f"class_pair:{self.quot.mode}",
               # [Q2] the AT-SUPPORT MASS: the share of this level's own stream that lands
               # on keys the learner holds at support. It is the currency the merge's
               # one-level-up licence reads (a coarsening raises mass and can LOWER the
               # key count), and logging it per cycle is what will let a floor be derived
               # for it by `null_abba`, as every other floor in the arc is.
               "mass_at_support": {
                   str(t): (sum(c for c in self.counts.values() if c >= t)
                            / max(sum(self.counts.values()), 1))
                   for t in (1, 2, 3, 5, 10)},
               "n_rows_seen": int(self.n_rows), "n_rows_dropped": int(self.n_dropped),
               "last_build": dict(self.last_build)}
        for t in (3,):
            sup = t
        out["keys_at_support"] = sorted(
            [_jsonable(k) for k, c in self.counts.items() if c >= sup])
        return out


def _jsonable(key):
    return [list(x) if isinstance(x, tuple) else int(x) for x in key]


def make_miner(level, s, cfg, quot):
    """The one place `enharmonic.py` chooses a miner. With `quot` None this is the donor's
    constructor character for character, which is what keeps the fork inert."""
    if quot is None:
        return MC.Miner(level, s)
    return ClassMiner(level, s, quot, spell_cap=int((cfg or {}).get("quot_spell_cap", 4)))


# --------------------------------------------------------------------------- #
# the class alphabet, by closure — the sizing lane's object, re-derived in the fork
# --------------------------------------------------------------------------- #

def class_closure(rules, canon, v, s, depth, max_level=6):
    """The token-class alphabet and the number of legal class PAIRS at every level, by closure
    of `possible_sets`' composition step over the writable blocks. `sizing/phase0_cat.py`
    computes the identical object; G-3 asserts the two agree."""
    bottom = rules[depth - 1]
    canon = np.asarray(canon, np.int64)

    def blk(t):
        return frozenset(f for f in range(v) for r in range(bottom.shape[1])
                         if tuple(int(z) for z in bottom[f, r]) == t)
    writable = sorted({tuple(int(z) for z in canon[f]) for f in range(v)})
    cur = sorted({blk(t) for t in writable}, key=lambda c: sorted(c))
    out = {1: {"n_classes": len(cur), "n_legal_class_pairs": None}}
    for ell in range(2, max_level + 1):
        layer = rules[depth - ell]
        pairs = {}
        for c1 in cur:
            for c2 in cur:
                got = frozenset(f for f in range(v) for r in range(layer.shape[1])
                                if int(layer[f, r, 0]) in c1 and int(layer[f, r, 1]) in c2)
                if got:
                    pairs[(c1, c2)] = got
        nxt = sorted(set(pairs.values()), key=lambda c: sorted(c))
        out[ell] = {"n_classes": len(nxt), "n_legal_class_pairs": len(pairs)}
        cur = nxt
    return out


# --------------------------------------------------------------------------- #
# THE GATES — offline, no GPU, no substrate
# --------------------------------------------------------------------------- #

def quotient_gate(v=8, s=2, depth=6, m=2, rule_seed=0, n=512, verbose=False):
    """E-0. Four assertions, all pure:

      G-1  IDENTITY. With singleton classes `ClassMiner` reproduces `MC.Miner` bit for bit —
           same `child`, same `flat`, same row ORDER, same at-support counts — on the same
           observations, at every level a run commits. This is the 0.000e+00 the quotient path
           is certified on, and it is why `flat` and a singleton `given_cat` are the same arm.
      G-2  THE DGP ARITHMETIC THE COVER NUMBERS REST ON. `feat_parent_sets` reproduces
           `MC.true_tables`' `feature` column, as a SET, for every true row at L2..L4, and the
           multiply-parented counts match the sizing lane's (2 / 8 / 128). This is the object
           SIZING.md §1 is computed from; it is a cross-check of the world, not of a run-time
           map ('gen' coarsens the token class instead — see G-5 and the module docstring).
      G-3  THE 'tok' MAP IS THE SIZED ONE. The closure's alphabet and legal-pair counts are
           `sizing/SIZING.md` §2's, and every true row's parent set is contained in its token
           class.
      G-4  THE RATCHET STILL BITES. Over a lower table missing a class entirely, no class-pair
           naming that class survives `build`.
    """
    from rhm.rhm_data import generate_rules_distinct
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    truth = MC.true_tables(rules, depth, s, v, m, 5)
    rng = np.random.default_rng(17)
    out = {}

    # ---- G-1 identity ---------------------------------------------------------------- #
    q = Quotient("singleton", rules, canon, v, s, depth)
    g1 = []
    for level in (2, 3, 4):
        span = s ** (level - 1)
        # a realistic stream: true rows repeated, plus junk, in a shuffled order
        pool = np.concatenate([truth[level]["flat"],
                               rng.integers(0, v, size=(n // 4, span))], axis=0)
        obs = pool[rng.integers(0, pool.shape[0], size=n)]
        a, b = MC.Miner(level, s), ClassMiner(level, s, q, spell_cap=4)
        a.observe(obs); b.observe(obs)
        lower = (MC.base_table(v) if level == 2 else truth[level - 1])
        ta, tb = a.build(lower, 3), b.build(lower, 3)
        same = (ta["child"].shape == tb["child"].shape
                and bool((ta["child"] == tb["child"]).all())
                and bool((ta["flat"] == tb["flat"]).all())
                and (a.state()["n_at_support"] == {k: vv for k, vv in
                                                   b.state()["n_at_support"].items()}))
        g1.append({"level": level, "n_entries": int(ta["child"].shape[0]), "identical": same})
        assert same, f"G-1 FAILED at L{level}: singleton ClassMiner != MC.Miner"
    out["G-1 identity"] = g1

    # ---- G-2 the 'gen' map ------------------------------------------------------------ #
    g2 = []
    for level in (2, 3, 4):
        rows = {}
        for row, f in zip(truth[level]["flat"], truth[level]["feature"]):
            rows.setdefault(tuple(int(x) for x in row), set()).add(int(f))
        keys = sorted(rows)
        P = feat_parent_sets(rules, np.array(keys, np.int64), level, v, s, depth)[:, 0, :]
        got = [set(np.nonzero(r)[0].tolist()) for r in P]
        agree = all(rows[k] == g for k, g in zip(keys, got))
        multi = sum(1 for g in got if len(g) > 1)
        g2.append({"level": level, "n_distinct": len(keys), "agrees_with_true_tables": agree,
                   "n_multi_parented": multi})
        assert agree, f"G-2 FAILED at L{level}: feat_parent_sets != true_tables' feature column"
    assert [r["n_multi_parented"] for r in g2] == [2, 8, 128], \
        f"G-2 FAILED: cover counts moved from the sizing lane's (2, 8, 128): {g2}"
    out["G-2 gen map"] = g2

    # ---- G-3 the 'tok' map ------------------------------------------------------------ #
    clos = class_closure(rules, canon, v, s, depth, 6)
    sized_c = {1: 7, 2: 9, 3: 11, 4: 13, 5: 22, 6: 42}
    sized_p = {2: 13, 3: 28, 4: 40, 5: 73, 6: 306}
    okc = all(clos[k]["n_classes"] == sized_c[k] for k in sized_c)
    okp = all(clos[k]["n_legal_class_pairs"] == sized_p[k] for k in sized_p)
    sup_ok = True
    for level in (2, 3, 4):
        rows = {}
        for row, f in zip(truth[level]["flat"], truth[level]["feature"]):
            rows.setdefault(tuple(int(x) for x in row), set()).add(int(f))
        keys = sorted(rows)
        T = token_class_sets(rules, np.array(keys, np.int64), level, canon, v, s, depth)[:, 0, :]
        for k, r in zip(keys, T):
            sup_ok &= rows[k] <= set(np.nonzero(r)[0].tolist())
    out["G-3 tok map"] = {"closure": {str(k): vv for k, vv in clos.items()},
                          "matches_sized_alphabet": okc, "matches_sized_pairs": okp,
                          "parent_set_subset_of_token_class": bool(sup_ok)}
    assert okc and okp, f"G-3 FAILED: the class alphabet moved from SIZING.md §2: {clos}"
    assert sup_ok, "G-3 FAILED: a true row's parent set is not inside its token class"

    # ---- G-4 the ratchet -------------------------------------------------------------- #
    qt = Quotient("tok", rules, canon, v, s, depth)
    level = 3
    mnr = ClassMiner(level, s, qt, spell_cap=4)
    mnr.observe(np.repeat(truth[level]["flat"], 4, axis=0))
    full = mnr.build(truth[level - 1], 3)
    lo = truth[level - 1]
    keep_cls = sorted({qt.id_of(tuple(int(x) for x in r), level - 1) for r in lo["flat"]})[:-1]
    idx = [i for i, r in enumerate(lo["flat"])
           if qt.id_of(tuple(int(x) for x in r), level - 1) in keep_cls]
    part = MC.make_table(level - 1, lo["child"][idx], lo["lower"], s)
    short = mnr.build(part, 3)
    out["G-4 ratchet"] = {"level": level,
                          "n_entries_full": int(full["child"].shape[0]),
                          "n_entries_class_removed": int(short["child"].shape[0]),
                          "n_keys_full": mnr.last_build and None,
                          "shrinks": bool(short["child"].shape[0] < full["child"].shape[0])}
    assert short["child"].shape[0] < full["child"].shape[0], \
        "G-4 FAILED: removing a whole class from the lower table did not shrink the build"

    # ---- G-5 'gen' is a coarsening of 'tok', and the parent where the parent is unique -- #
    qt2 = Quotient("tok", rules, canon, v, s, depth)
    qg2 = Quotient("gen", rules, canon, v, s, depth)
    g5 = []
    for level in (2, 3, 4):
        rows = {}
        for row, f in zip(truth[level]["flat"], truth[level]["feature"]):
            rows.setdefault(tuple(int(x) for x in row), set()).add(int(f))
        keys = sorted(rows)
        a = qt2.ids(np.array(keys, np.int64), level)
        b = qg2.ids(np.array(keys, np.int64), level)
        # same drop set
        drops = all((x is None) == (y is None) for x, y in zip(a, b))
        # gen is a function of tok (a coarsening), and never finer
        f_of = {}
        coarsens = True
        for x, y in zip(a, b):
            if x is None:
                continue
            if f_of.setdefault(x, y) != y:
                coarsens = False
        # where the token class is a singleton, gen IS that parent
        exact = all(y == int(np.log2(x)) for x, y in zip(a, b)
                    if x is not None and x & (x - 1) == 0)
        g5.append({"level": level, "n_keys": len(keys), "same_drop_set": drops,
                   "gen_is_a_coarsening_of_tok": coarsens,
                   "gen_is_the_parent_where_unique": exact,
                   "n_classes_tok": len({x for x in a if x is not None}),
                   "n_classes_gen": len({y for y in b if y is not None})})
        assert drops and coarsens and exact, f"G-5 FAILED at L{level}: {g5[-1]}"
        assert g5[-1]["n_classes_gen"] <= v, f"G-5 FAILED: 'gen' has more than v classes"
    out["G-5 gen coarsens tok"] = g5

    # ---- G-6 a wrong-width lower table yields an empty build, as the donor's does -------- #
    q6 = Quotient("tok", rules, canon, v, s, depth)
    g6 = []
    for level in (3, 4):
        a, b = MC.Miner(level, s), ClassMiner(level, s, q6, spell_cap=4)
        obs = np.repeat(truth[level]["flat"], 4, axis=0)
        a.observe(obs); b.observe(obs)
        ta = a.build(MC.base_table(v), 3)          # a level-1 table under a level-`level` miner
        tb = b.build(MC.base_table(v), 3)
        g6.append({"level": level, "donor_entries": int(ta["child"].shape[0]),
                   "class_entries": int(tb["child"].shape[0])})
        assert ta["child"].shape[0] == 0 and tb["child"].shape[0] == 0, (
            f"G-6 FAILED at L{level}: a wrong-width lower table did not yield an empty build "
            f"({g6[-1]}) — the donor's audition counterfactual passes exactly this")
    out["G-6 width contract"] = g6

    out["verdict"] = "PASS"
    if verbose:
        import json as _j
        print(_j.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    quotient_gate(verbose=True)
