"""Live grammar drift for the practice substrate: the world moves WHILE the agent practises.

WHY THIS EXISTS. Every round of the practice arc so far has measured commitment policy in a
regime where post-commit invalidation is STRUCTURALLY IMPOSSIBLE. The RHM sculpting substrate
has a *boundary* conditioning gap (hidden ancestors, so launch-time observation is informative)
and an *epistemic within-unit* gap (a committed macro forgoes priced groundings that would
reveal facts about the grammar) -- but no NEWS gap: the grammar is static, damage recurs i.i.d.
within an era, and the plant is inert, so nothing already committed can become wrong later. The
measured signatures of that absence are everywhere: recert fired 0/24 in `ear` and 0/288 in
`recital`, `tall` proved the mined table cannot churn, the etude's post-commit drift was exactly
0.0000, and `crystallize` found "nothing to certify".

This module installs the missing gap by RESAMPLING RULE-TABLE ENTRIES DURING THE RUN. Among the
levers considered (surface masking, stochastic execution inside a committed unit) this is the
only one under which already-committed content BECOMES WRONG rather than merely being a bet at
launch: a committed level-l macro entry is a level-1 tuple that WAS derivable at level l, and a
level-l rule resample can make it un-derivable, at which point applying the macro writes an
off-grammar span and the repair fails.

WHY NOT `full_loop`'s OU DRIFT. `directed_sculpting/full_loop`'s `advance_drift` /
`rhm_drift.calibrate_sigma_event` drift the SYNONYM MIXTURE WEIGHTS -- which rule of the m is
chosen -- so the drift is soft and measured in nats of KL. That machinery cannot invalidate a
committed macro: every entry stays grammatical, only its frequency moves. Hard support drift is
what this substrate needs, and its honest currency is a FRACTION (of the level's vocabulary that
survives the event), not nats, because the support changes and the KL is infinite. The idea kept
from that machinery is the one that matters: a drift EVENT of measured magnitude, calibrated to
a target rather than guessed, with the per-event and cumulative magnitudes reported separately.

LEVEL INDEXING IS THE INSTRUMENT. `rules[depth - l]` is the (v, m, s) layer mapping level-l
features to s-tuples of level-(l-1) features, and the mined vocabulary is level-indexed too, so
a level-l resample invalidates level-l structure and everything above it, leaving levels below
bit-identical (gate T-2). Drift is therefore confined to levels >= 2 by default, which leaves
the BOTTOM layer (`rules[depth-1]`, level-1 feature -> leaf tuple) untouched -- so `canon`, the
inverse maps, the reader and the generator's block head all stay exact, and the effect is
isolated to the vocabulary rather than confounded with a broken parse.

DRIFT IS UNANNOUNCED. Nothing in the arm loop is told an event happened. Re-mining, count decay
and recert all run on their own clocks. That is what makes the gap a real conditioning gap: the
agent must find out from its error stream, which is the whole point.
"""

import numpy as np

from rhm.practice.ratchet import macros as MC
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.rhm_sculpt_precheck import parse_success_and_heuristic


# --------------------------------------------------------------------------- #
# one drift event
# --------------------------------------------------------------------------- #

def _code_to_tuple(code, v, s):
    return tuple(int((code // v ** i) % v) for i in range(s))


def resample_cells(rules, depth, level, n_cells, rng):
    """Resample `n_cells` (feature, rule) cells of the level-`level` layer.

    Preserves `generate_rules_distinct`'s invariant -- the m rules of a feature are m DISTINCT
    s-tuples -- by rejecting codes the feature already holds. Every other layer is returned
    bit-identical (gate T-1), so drift at level >= 2 never touches the leaf rendering.

    Returns (new_rules, changed) where `changed` lists the (feature, rule, old, new) cells.
    """
    new = [np.array(r, copy=True) for r in rules]
    layer = new[depth - level]
    v, m, s = layer.shape
    cells = rng.permutation(v * m)[:n_cells]
    changed = []
    for c in np.sort(cells):
        f, r = int(c) // m, int(c) % m
        old = tuple(int(x) for x in layer[f, r])
        used = {tuple(int(x) for x in layer[f, j]) for j in range(m)}
        tup = old
        for _ in range(1024):
            cand = _code_to_tuple(int(rng.integers(0, v ** s)), v, s)
            if cand not in used:
                tup = cand
                break
        else:                                       # pragma: no cover -- v^s >> m always
            raise RuntimeError("no distinct replacement tuple available")
        layer[f, r] = np.array(tup, dtype=layer.dtype)
        changed.append({"feature": f, "rule": r, "old": list(old), "new": list(tup)})
    return new, changed


# --------------------------------------------------------------------------- #
# magnitude: the currency a drift event is measured in
# --------------------------------------------------------------------------- #

def table_survival(rules_a, rules_b, depth, s, v, m, max_level, truth_a=None, truth_b=None):
    """Per level 2..max_level: what fraction of level-l vocabulary defined by `rules_a`
    is still legal under `rules_b`.

    This is the quantity a frozen committed table decays along, expressed on the DGP's own
    tables so it is AGENT-INDEPENDENT -- the world's motion, not any arm's luck. Reported
    alongside `novel` (the fraction of the new vocabulary that is new), which is what a miner
    would have to discover to catch up.
    """
    ta = truth_a if truth_a is not None else MC.true_tables(rules_a, depth, s, v, m, max_level)
    tb = truth_b if truth_b is not None else MC.true_tables(rules_b, depth, s, v, m, max_level)
    out = {}
    for ell in range(2, max_level + 1):
        A = {tuple(int(x) for x in r) for r in ta[ell]["flat"]}
        B = {tuple(int(x) for x in r) for r in tb[ell]["flat"]}
        out[str(ell)] = {
            "n_a": len(A), "n_b": len(B), "n_shared": len(A & B),
            "survival": (len(A & B) / len(A)) if A else None,
            "novel": (len(B - A) / len(B)) if B else None}
    return out


def corpus_survival(rules_a, rules_b, s, n, seed):
    """Fraction of derivations sampled from `rules_a` that still parse to their own root under
    `rules_b`. The whole-string reading of the same motion: how much of the old world is still
    a legal instance of the new one."""
    roots, leaves = _sample_pool(rules_a, n, s, seed)
    ok, _ = parse_success_and_heuristic(rules_b, leaves, roots, s)
    return float(np.asarray(ok).mean())


def half_life_cycles(per_event_survival, period):
    """Cycles for a frozen table to lose half its validity, given a per-event survival `q`
    and a drift period. Closed form, so the calibration reports a horizon rather than a rate."""
    q = float(per_event_survival)
    if not (0.0 < q < 1.0):
        return float("inf")
    return period * np.log(0.5) / np.log(q)


# --------------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------------- #

def epoch_of(cyc, cfg):
    """Which drift epoch cycle `cyc` lives in. Epoch 0 is the pristine grammar and runs to
    `drift_start - 1`; every `drift_every` cycles thereafter opens a new one. Indexed by CYCLE,
    not by priced time, so on the fixed era clock EVERY ARM SEES THE SAME WORLD AT THE SAME
    CYCLE and the comparison is a commit-policy comparison rather than a world-luck one."""
    if cfg["drift_every"] <= 0 or cfg["drift_cells"] <= 0:
        return 0
    if cyc < cfg["drift_start"]:
        return 0
    return 1 + (cyc - cfg["drift_start"]) // cfg["drift_every"]


def n_epochs(max_cycle, cfg):
    return epoch_of(max_cycle, cfg) + 1


def rules_trajectory(rules0, cfg, max_cycle):
    """The whole sequence of grammars the run will pass through, drawn ONCE at setup from one
    dedicated stream (`drift_seed`) so it is independent of every arm's own randomness."""
    rng = np.random.default_rng(cfg["drift_seed"] + 606)
    traj = [{"epoch": 0, "rules": rules0, "cycle": 0, "changed": []}]
    for e in range(1, n_epochs(max_cycle, cfg)):
        prev = traj[-1]["rules"]
        nxt, changed = resample_cells(prev, cfg["depth"], cfg["drift_level"],
                                      cfg["drift_cells"], rng)
        traj.append({"epoch": e, "rules": nxt, "changed": changed,
                     "cycle": cfg["drift_start"] + (e - 1) * cfg["drift_every"]})
    return traj


# --------------------------------------------------------------------------- #
# a vocabulary that can LOSE entries
# --------------------------------------------------------------------------- #

class DecayMiner(MC.Miner):
    """`MC.Miner` with geometric count decay, so the mined vocabulary can SHED.

    `tall` established that the arc's miner cannot churn: counts only increase, a support
    threshold on a monotone count is monotone, and the ratchet filter can only drop an entry if
    the lower table shrinks, which it cannot. Under a static grammar that is harmless -- a stale
    entry is a contradiction in terms. Under drift it is fatal: an entry that was legal when it
    was mined stays in the table forever.

    The mechanism chosen here is the cheapest one that is also AGENT-INTERNAL: age every count
    by `decay` once per cycle and forget a tuple once its count falls under `floor`. A tuple
    observed at rate lambda per cycle settles at count lambda/(1-decay), so it survives while
    lambda > (1 - decay) * support and is forgotten about log(support/count)/log(decay) cycles
    after the world stops producing it. No oracle, no drift signal, no re-grading -- just
    recency-weighted mining, on its own clock.

    `decay = 1.0` is `MC.Miner` EXACTLY (counts never move, nothing ever falls below a floor
    < 1), which is what keeps the drift-off runner bit-identical to its parent (gate T-4).
    """

    def __init__(self, level, s, decay=1.0, floor=0.5):
        super().__init__(level, s)
        self.decay = float(decay)
        self.floor = float(floor)
        self.n_forgotten = 0

    def age(self):
        """One cycle of forgetting. Returns how many tuples fell out of the vocabulary."""
        if self.decay >= 1.0:
            return 0
        drop = []
        for k in self.counts:
            self.counts[k] *= self.decay
            if self.counts[k] < self.floor:
                drop.append(k)
        for k in drop:
            del self.counts[k]
        self.n_forgotten += len(drop)
        return len(drop)

    def state(self):
        st = super().state()
        st["n_forgotten"] = int(self.n_forgotten)
        st["decay"] = self.decay
        return st


def entry_set(table):
    """A table's entries as a set of flattened level-1 tuples -- the identity instrument
    `recital` asked for and `tall` built, here with something to measure."""
    return {tuple(int(x) for x in r) for r in table["flat"]}


def churn(prev, now):
    """Added / removed / retained between two entry sets."""
    return {"added": len(now - prev), "removed": len(prev - now),
            "retained": len(prev & now), "n": len(now)}


# --------------------------------------------------------------------------- #
# the offline calibration: where is the interesting regime?
# --------------------------------------------------------------------------- #

def simulate_miner(rules0, cfg, cycles, obs_per_cycle, decay, seed, support=3, level=2):
    """Replay the mining stream against a drifting grammar, on CPU, with no substrate at all.

    The agent's miner sees `obs_per_cycle` level-l spans per cycle, drawn from the CURRENT
    grammar's own level-l vocabulary (the optimistic limit: a perfect parser reading solved
    configurations). Drift runs on the schedule in `cfg`. What comes out is the steady-state
    quality of the built table AGAINST THE CURRENT TRUTH -- which is the quantity that decides
    whether anything is worth compiling at a given drift rate:

      * churn much faster than the miner can re-support a tuple  -> recall collapses, nothing
        is ever worth committing;
      * churn much slower than the run                           -> the static case rebuilt.

    Locating the bracket is a MEASUREMENT, which is what this function is for.
    """
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rng = np.random.default_rng(seed)
    traj = rules_trajectory(rules0, cfg, cycles)
    mn = DecayMiner(level, s, decay=decay)
    frozen = None                   # a table committed at `cfg["commit_cycle"]`
    rows = []
    for cyc in range(1, cycles + 1):
        ep = epoch_of(cyc, cfg)
        rules = traj[ep]["rules"]
        truth = MC.true_tables(rules, depth, s, v, m, level)[level]
        flats = truth["flat"]
        mn.age()
        pick = rng.integers(0, flats.shape[0], size=obs_per_cycle)
        mn.observe(flats[pick])
        built = mn.build(MC.base_table(v), support)
        gt = MC.grade_table(built, truth) if built["child"].shape[0] else {
            "n_learned": 0, "precision": None, "recall": 0.0}
        if frozen is None and cyc == cfg.get("commit_cycle", 25):
            frozen = built
        row = {"cycle": cyc, "epoch": ep, "n_entries": int(gt["n_learned"]),
               "recall": gt["recall"], "precision": gt["precision"],
               "n_forgotten": int(mn.n_forgotten)}
        if frozen is not None and frozen["child"].shape[0]:
            fg = MC.grade_table(frozen, truth)
            row["frozen_precision"] = fg["precision"]
            row["frozen_recall"] = fg["recall"]
        rows.append(row)
    return rows
