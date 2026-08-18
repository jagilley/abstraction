"""The re-key op: earning the index's basis from what serves what, with no oracle.

WHERE THIS COMES FROM. `fw_s0` gave the library one op that acts on the index — **merge**, which
can only *delete* distinctions. It collapsed a misleading 8-cell wall index to 2 and extinguished
the rotation response, but it could never do better than having no index at all, because
deletion has no way to *build* a class. `fw_s0`'s Gate 0 named the missing regime exactly: the
true key is linearly decodable from the agent's own state at 0.927, and the credit machinery is
not wired to route on it (the consuming DP recovers it at 0.518).

RE-KEY IS THAT WIRING, EARNED. Two contexts belong to the same class iff **the same committed
unit serves them**. That is functional addressing: it never mentions the latent, only the
agent's own record of which of its programs repaired which of its instances. Merge is the
pairwise version of this test; re-key is its closure — cluster contexts by what-serves-them and
migrate the index onto the classes that fall out.

FOUR PIECES, in the order the loop uses them:

  1. `service_matrix` — the exact, model-free record. `S[p, i]` is "does committed program p
     repair instance i", decided by the grammar's own possible-set DP. Priced: it is the agent
     asking its feedback channel a question, once per (program, instance).

  2. `greedy_cover` — the closure. A minimal set of programs that between them serve the pool,
     each instance labelled by the program that covers it. This *is* the partition "same unit
     serves them", and its size is the earned index's cardinality. Nothing here can see `f`.

  3. `Router` — the part that makes an earned class **addressable**. A linear read of the
     agent's own controller state (plus the root) onto the cover's labels. The labels are
     self-generated, so this is the agent learning to recognise a distinction it discovered by
     use rather than one it was handed.

  4. The migration gate, in `run_arm` — adopt the earned basis only where it **serves better
     than the incumbent index**, measured on held-out instances in the same exact currency.
     In phase 1 a wall-keyed incumbent is near-perfect, so the gate stays shut and the scaffold
     is left alone; after a rotation the incumbent collapses and the gate opens. The tear-down
     is therefore *paid for by evidence*, not scheduled.

WHY THIS IS UNDEFINED AT t = 0, AND WHY THAT IS THE POINT. With no committed units there are no
classes: `greedy_cover` of an empty program set is empty. The basis cannot precede the
vocabulary. That is the ratchet's shape one level up — `fw_s0`'s index was handed its cells by
the wall, and the question this round asks is whether cells can be *earned*, and whether having
been handed a scaffold first makes earning them faster.
"""

import numpy as np

from rhm.practice.fourwall import wall as WL


# --------------------------------------------------------------------------- #
# 1. the exact record of what serves what
# --------------------------------------------------------------------------- #

def service_matrix(rules, x_np, roots_np, programs, node, level, s, canon_np):
    """`S[p, i]` = does committed program `p` repair instance `i`. Exact, model-free.

    Returns `(S, n_graded)`; `n_graded` is the price, one grading per (program, instance)."""
    return WL.entry_profile(rules, x_np, roots_np, programs, node, level, s, canon_np)


def distinct_programs(lib):
    """Every distinct committed program the library currently holds, in a stable order."""
    seen = {}
    for c in lib.class_ids():
        t = lib.table(c)
        if t is None:
            continue
        for row in t["flat"]:
            seen.setdefault(tuple(int(z) for z in row), None)
    return list(seen)


# --------------------------------------------------------------------------- #
# 2. the closure: cluster contexts by what serves them
# --------------------------------------------------------------------------- #

def greedy_cover(S, min_gain=1):
    """A minimal set of programs covering the pool, and the induced partition.

    Repeatedly take the program that serves the most still-uncovered instances. The chosen
    programs ARE the earned classes; `labels[i]` is the index (into `chosen`) of the program
    that covers instance `i`, or -1 where nothing does. This is exactly "two contexts are in
    the same class iff the same committed unit serves them", computed without ever naming the
    latent.

    `min_gain` stops the cover from spawning a class for a single stray instance, which would
    make the index cardinality track pool noise rather than structure."""
    if S.shape[0] == 0 or S.shape[1] == 0:
        return [], np.full(S.shape[1] if S.ndim == 2 else 0, -1, np.int64)
    n_prog, n_inst = S.shape
    uncovered = np.ones(n_inst, bool)
    labels = np.full(n_inst, -1, np.int64)
    chosen = []
    while uncovered.any():
        gain = (S & uncovered[None, :]).sum(1)
        best = int(gain.argmax())
        if gain[best] < min_gain:
            break
        labels[S[best] & uncovered] = len(chosen)
        chosen.append(best)
        uncovered &= ~S[best]
    return chosen, labels


def adjusted_rand(a, b):
    """ARI between two labellings. Used only as an ORACLE READOUT: the earned partition against
    the true `f`-partition. Never consumed by any arm."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0:
        return float("nan")
    _ua, ia = np.unique(a, return_inverse=True)
    _ub, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((ia.max() + 1, ib.max() + 1), np.int64)
    np.add.at(cont, (ia, ib), 1)

    def c2(x):
        return (x * (x - 1) / 2.0).sum()

    sij, si, sj = c2(cont), c2(cont.sum(1)), c2(cont.sum(0))
    tot = a.size * (a.size - 1) / 2.0
    exp = si * sj / tot if tot else 0.0
    mx = 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


# --------------------------------------------------------------------------- #
# 3. making an earned class addressable
# --------------------------------------------------------------------------- #

class Router:
    """A linear read of the agent's own state onto self-generated class labels.

    Deliberately the same shape as `fourwall.key_probe`, which is what makes the comparison
    legible: Gate 0 measured that a linear probe recovers the TRUE key at 0.927 from labels the
    agent never receives. This one is trained on labels the agent generates itself — which
    program repaired which instance — so the gap between the two is the price of having to earn
    the distinction instead of being told it."""

    def __init__(self, state_dim, v, device):
        self.state_dim, self.v, self.device = state_dim, v, device
        self.head = None
        self.emb = None
        self.n_cls = 0
        self.fit_acc = float("nan")

    def fit(self, z, r, y, n_cls, steps=400, lr=3e-3, batch=256, seed=0):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        self.n_cls = int(n_cls)
        if self.n_cls < 2 or z.shape[0] < 8:
            self.head = None
            return float("nan")
        self.emb = nn.Embedding(self.v, self.state_dim).to(self.device)
        self.head = nn.Linear(self.state_dim, self.n_cls).to(self.device)
        opt = torch.optim.AdamW(list(self.head.parameters()) + list(self.emb.parameters()),
                                lr=lr)
        g = torch.Generator().manual_seed(seed)
        n = z.shape[0]
        for _ in range(steps):
            i = torch.randint(0, n, (min(batch, n),), generator=g).to(self.device)
            loss = F.cross_entropy(self.head(z[i] + self.emb(r[i])), y[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.fit_acc = float((self.predict(z, r)[0] == y).float().mean())
        return self.fit_acc

    def predict(self, z, r):
        import torch
        if self.head is None:
            return (torch.zeros(z.shape[0], dtype=torch.long, device=self.device),
                    torch.zeros(z.shape[0], device=self.device))
        with torch.no_grad():
            logits = self.head(z + self.emb(r))
            p = logits.softmax(-1)
            conf, cls = p.max(-1)
        return cls, conf
