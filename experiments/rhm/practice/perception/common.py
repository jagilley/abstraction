"""perception — shared machinery: the Modal image, the grid arithmetic, and the miner
recomputed on an estimated grid.

THE QUESTION THIS NODE IS BUILT ON. `conductor`'s yield thermostat reads `at_support`:
the count of distinct level-(l+1) tuples observed at least `mine_support` times over the
learner's own solved trajectories. The count is free because RHM hands the miner a GRID
--- the solved configuration is a whole aligned sequence, so a level-l constituent is the
span [node * s^l, (node+1) * s^l) of its leaves, and `(era.level, era.node)` names which
node. On a stream a learner would have to find that grid itself.

`logit_reading/frontier` built the instrument that finds it: a next-token reader's own
per-position entropy profile repeats with the period of the deepest level it has absorbed,
and `common.nested_phase` recovers the depth-k boundary offset bottom-up. Here that
instrument is pointed at the practice learner's own solved pieces, presented at random
offsets inside a stream of its own productions, and `at_support` is recomputed on the grid
the detector recovers.

THE KEYING, stated rather than hidden. `conductor`'s miner keys by tuples of LEVEL-1
FEATURE ids, read off each solved configuration by the practice learner's own generator
block head (`macros.parse_features`). That parse is itself tied to the block grid, and the
generator cannot be evaluated at a shifted grid without either re-running it or rolling its
input, so the endogenous miner here keys by LEAF TOKEN tuples of width s^l instead. Three
series are therefore reported and the middle one is the bridge:

    A  feature-keyed, TRUE grid     what the thermostat actually read (`obs_hist`)
    B  token-keyed,   TRUE grid     the same count in this node's keying
    C  token-keyed,   RECOVERED grid the endogenous gauge

C-vs-B isolates the grid; B-vs-A prices the keying.
"""

import numpy as np

import modal

DATA_DIR = "/data"
REMOTE = "rhm_practice_perception"

# the practice substrate: conductor.DEPTH6 + generate_rules_distinct(seed=0)
V, S, DEPTH, M = 8, 2, 6, 2
T_SEQ = S ** DEPTH                      # 64 -- one piece
RULE_SEED = 0


def _ignore(path):
    """Mount the Python sources this node imports and nothing else.

    `frontier/common.py`'s reason, re-derived: another agent writing a launch log into a
    sibling node's `results/` tree makes the image build race and fail, and the same race
    fires on a sibling's SOURCE while another agent edits it. So: no `results/` trees, no
    `.log` files, nothing that is not a `.py`, and under the two big package directories
    only the siblings this node actually imports."""
    sp = str(path).replace("\\", "/")
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    if not sp.endswith(".py"):
        return True
    for parent, keep in (("/logit_reading/", ("frontier/",)),
                         ("/practice/", ("perception/", "conductor/", "crystallize/",
                                         "ratchet/", "native/"))):
        if parent in sp:
            tail = sp.split(parent, 1)[1]
            if "/" in tail and not tail.startswith(keep):
                return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm", ignore=_ignore)
)
app = modal.App("rhm-practice-perception", image=image)


# --------------------------------------------------------------------------- #
# the grid the miner is handed
# --------------------------------------------------------------------------- #

def span_leaves(era_level, era_node, ell, s=S, depth=DEPTH):
    """`conductor`'s own arithmetic, in LEAF coordinates: the level-`ell` span that
    contains the era's damage cell.

      span_f = s**(ell-1)                       (in level-1 feature units)
      node_f = (era.node * s**(era.level-1)) // span_f
      leaves = [node_f * s**ell, (node_f+1) * s**ell)
    """
    span_f = s ** (ell - 1)
    node_f = (int(era_node) * s ** (int(era_level) - 1)) // span_f
    lo = node_f * span_f * s
    return int(lo), int(lo + span_f * s)


def piece_start_hat(r_hat_L, s=S, depth=DEPTH):
    """The window TOKEN index at which the detector thinks the piece begins.

    `frontier.common.true_boundary_column(phase, P) = (-phase-1) % P` is the COLUMN whose
    TARGET carries leaf residue 0; the target of column j is window token j+1."""
    return (np.asarray(r_hat_L) + 1) % (s ** depth)


# --------------------------------------------------------------------------- #
# the miner, recomputed on a given grid
# --------------------------------------------------------------------------- #

class TokenMiner:
    """`ratchet.macros.Miner` in leaf-token coordinates: count distinct tuples, report how
    many are at support. Cumulative over the run, exactly as the conductor's miners are."""

    def __init__(self, support):
        self.support = int(support)
        self.counts = {}
        self.n_obs = 0

    def observe(self, rows):
        rows = np.asarray(rows, np.int64)
        if rows.size == 0:
            return
        self.n_obs += rows.shape[0]
        for row in rows:
            k = tuple(int(x) for x in row)
            self.counts[k] = self.counts.get(k, 0) + 1

    def at_support(self):
        return int(sum(1 for c in self.counts.values() if c >= self.support))

    def n_distinct(self):
        return len(self.counts)


def extract(windows, starts, lo, hi):
    """(n, span) the tokens a miner handed `starts` as each window's piece origin reads."""
    idx = np.asarray(starts)[:, None] + np.arange(lo, hi)[None, :]
    return np.take_along_axis(np.asarray(windows), idx, 1)


# --------------------------------------------------------------------------- #
# the detector, template-free
# --------------------------------------------------------------------------- #

def meanprof_phase(Xd, L=DEPTH, s=S, j_lo=0):
    """`frontier.common.nested_phase(..., mode="meanprof")`, called with no templates.

    The template variant needs E[H(p_k) | leaf residue] for every observer k on THIS
    grammar, which is a new oracle computation; `meanprof` is fully model-free and
    frontier measured it 0.00-0.09 BELOW the template detector, so what it reports here is
    a lower bound on the reader's reach."""
    from rhm.logit_reading.frontier.common import nested_phase
    return nested_phase(Xd, {}, L=L, s=s, j_lo=j_lo, mode="meanprof")


def unconstrained_acc(Xd, phase, L=DEPTH, s=S, j_lo=0):
    """The per-k accuracies with each period estimated independently (altitude Q2 A's
    statistic), beside the nested ones."""
    from rhm.logit_reading.frontier.common import offset_means, true_boundary_column
    out = {}
    for k in range(1, L + 1):
        P = s ** k
        out[k] = float((offset_means(Xd, P, j_lo).argmax(1)
                        == true_boundary_column(phase, P)).mean())
    return out
