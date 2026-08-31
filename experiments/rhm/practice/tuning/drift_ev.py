"""tuning/drift_ev — level-3 rule-cell drift as the third event type, and its calibration.

Kept OUT of the Modal app (as `burst.py` and `../fourwall/lm/wall.py` are) so every piece is
auditable and gate-testable with no GPU.

THE EVENT. `fourwall/lm`'s rotation is index news (the key's meaning moves, the grammar is
untouched). `burst.py`'s noise burst is aleatoric news (nothing learnable moves; the indexed
span is corrupted). A DRIFT is **world news**: one or more (feature, rule) cells of the
level-3 layer are resampled, permanently, so a piece of the grammar itself becomes different.
The key stays valid, the parse must be relearned — the correct op is **track**.

MACHINERY. `resample_cells` is imported verbatim from
[`../transpose/drift.py`](../transpose/drift.py) and never modified: it resamples n cells of one
layer while preserving `generate_rules_distinct`'s within-feature distinctness and leaving every
other layer bit-identical. Two properties of that function carry this design:

  - drift confined to levels >= 2 never touches `rules[depth-1]`, the level-1 feature -> leaf
    tuple layer, so the leaf rendering and every exact-BP oracle stay well-defined;
  - `transpose` measured that level-2 drift degrades its executor while level-3 drift is
    admissible, which is why level 3 is the level used here.

WHY DRIFT IS NOT SPAN-LOCAL, AND WHY THAT IS THE POINT. A rotation moves one prefix token; a
burst corrupts leaves 0-15 only. A resampled level-3 cell changes how a mid-level feature
expands, so it moves the distribution of the WHOLE sequence. Gate 0 measured that a rotation
leaks +0.069 nats off the indexed span while a matched burst leaks +/-0.0002 — so `d_out` is
already a live coordinate, and drift is expected to move it too. The typing triple
T* = (d_key, d_self, d_out) therefore has a shot at all three events from one model:

    d_self ~ 0                  -> rotation   (a key-free view cannot see a key permutation)
    d_self > 0, d_out ~ 0       -> burst      (surface corruption, confined to the span)
    d_self > 0, d_out > 0       -> drift      (the world moved, everywhere)

The drift row is a plausibility to test, not a prediction: the key-free view of a keyed reader
is measurably hollow (Gate 0's mirror control: ~60% of `D_pair`'s clean level; d4 0.27), and
whether a hollow basis still REGISTERS structural news one level up is the open question.

THE MATCHED-SURPRISAL CONTROL A DRIFT NEEDS AND A BURST DID NOT. A burst counterfactual reuses
the same sequences with corrupted tokens, so the comparison is paired. A drift counterfactual
must draw NEW sequences (from the drifted grammar), so a naive comparison against the original
eval set confounds the drift with ordinary draw-to-draw variance. Every drift read here is
therefore taken against a NULL RUNG: a fresh draw from the UNDRIFTED grammar at the same
sequence seed. `d_key(rung k) = nll(rung k) - nll(null rung)`, and the null rung's own offset
from the standing eval set is logged so the size of the control is visible.
"""

import numpy as np

from rhm.practice.transpose.drift import resample_cells       # imported, never modified


# --------------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------------- #

def drift_steps(spec, max_steps):
    """The steps at which the grammar changes, permanently."""
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if x and int(x) < max_steps:
            out.append(int(x))
    return sorted(out)


def epoch_of(step, steps):
    """0 before the first drift; k after the k-th."""
    return int(sum(1 for x in steps if step >= x))


def clear_of(steps, others, guard=0):
    """True if every drift step is at least `guard` away from every step in `others`
    (used by the gate to assert drift never occupies a rotation or a burst window)."""
    for d in steps:
        for o in others:
            if abs(d - o) < guard:
                return False
    return True


# --------------------------------------------------------------------------- #
# the grammars
# --------------------------------------------------------------------------- #

def drift_rng(seed, tag, k):
    return np.random.default_rng(int(seed) + 7919 * (int(k) + 1) + 104729 * int(tag))


def rules_trajectory(base_rules, depth, level, n_cells, steps, seed):
    """[rules_epoch0, rules_epoch1, ...] — each epoch's grammar, cumulative.

    Drift is CUMULATIVE: epoch k+1 is epoch k drifted again, so the world keeps moving
    rather than oscillating around a fixed point. Every worker derives this from the same
    `seed`, so every arm lives in literally the same world at literally the same step.
    """
    traj, changes = [base_rules], []
    for k in range(len(steps)):
        nxt, ch = resample_cells(traj[-1], depth, level, n_cells,
                                 drift_rng(seed, 0, k))
        traj.append(nxt)
        changes.append(ch)
    return traj, changes


def ladder_rules(rules, depth, level, ladder, seed, epoch):
    """Counterfactual grammars for the matched-surprisal ladder AT THIS EPOCH.

    Each rung is the epoch's own grammar drifted by `n_cells` — never a cumulative chain —
    so the ladder measures the marginal cost of a drift of that size from where the reader
    actually stands. Rung 0 of the caller's read is the NULL (this grammar, fresh draw).
    """
    out = []
    for i, n_cells in enumerate(ladder):
        nxt, _ = resample_cells(rules, depth, level, int(n_cells),
                                drift_rng(seed, 1 + epoch, i))
        out.append(nxt)
    return out


# --------------------------------------------------------------------------- #
# magnitude: the agent-independent currency of a drift event
# --------------------------------------------------------------------------- #

def cell_survival(rules_a, rules_b, depth, level):
    """Fraction of the level-`level` layer's (feature, rule) cells that are unchanged.

    The world's motion, not any arm's luck — `transpose`'s point that a hard support change
    has no finite KL, so its honest currency is a fraction.
    """
    a, b = np.asarray(rules_a[depth - level]), np.asarray(rules_b[depth - level])
    v, m, s = a.shape
    same = sum(1 for f in range(v) for r in range(m)
               if tuple(a[f, r]) == tuple(b[f, r]))
    return float(same) / float(v * m)


def layers_identical(rules_a, rules_b, skip):
    """Every layer other than `skip` must be bit-identical (the gate's admissibility check)."""
    for i, (x, y) in enumerate(zip(rules_a, rules_b)):
        if i == skip:
            continue
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return True


def match_cells(rows, target, key="d_key"):
    """Interpolate the n_cells whose spike equals `target`. Returns (n_cells, bracketed)."""
    xs = [float(r["n_cells"]) for r in rows]
    ys = [float(r[key]) for r in rows]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    for a in range(len(xs) - 1):
        if (ys[a] - target) * (ys[a + 1] - target) <= 0 and ys[a + 1] != ys[a]:
            t = (target - ys[a]) / (ys[a + 1] - ys[a])
            return float(xs[a] + t * (xs[a + 1] - xs[a])), True
    a, b = (0, 1) if target < ys[0] else (len(xs) - 2, len(xs) - 1)
    if ys[b] == ys[a]:
        return float(xs[b]), False
    t = (target - ys[a]) / (ys[b] - ys[a])
    return float(max(1.0, xs[a] + t * (xs[b] - xs[a]))), False
