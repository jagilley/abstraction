"""[preplay] `soundboard.py::context_instances` (1137-1160), copied VERBATIM.

The helper is forked in every practice node (twenty copies in `rhm/practice/`); it is copied
again here so this node's container does not import a 1.2 MB module with a `modal.App` at its
top just to draw a pool. Gate F-6 asserts the copy is TEXT-IDENTICAL to soundboard's, so the
copy cannot silently drift from the pool the arms auditioned on.
"""

import numpy as np

from rhm.rhm_sculpt_precheck import nearest_derivation_cost
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.practice.crystallize.units import corrupt_hier


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
