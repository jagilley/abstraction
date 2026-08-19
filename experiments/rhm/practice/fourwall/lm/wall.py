"""fourwall/lm — wall primitives, exact index oracles, and the endogenous index instruments.

Deliberately kept OUT of the Modal app (mirroring `../wall.py`) so every piece is auditable
and gate-testable with no GPU and no substrate.

THE PORT. `../` (fourwall, fw_s0-s3) manufactured a spurious-but-free library key and ran the
index ops — merge / re-key / retire — on the learner's behalf, on an EXOGENOUS lookup-table
library. This node asks whether those are joints of an ENDOGENOUS reader: an autoregressive NTP
transformer on RHM whose only representation is the one it learned. So:

  the library  = the model's learned grammar (its content)
  the index    = what its early-position predictions are KEYED on -- i.e. how it addresses
                 which sub-grammar applies to the span it is about to emit
  the wall     = a free surface token `w` prepended at position 0, a bijection of the true
                 level-KEY_LEVEL feature at node KEY_NODE (the latent that governs the first
                 s^(L-KEY_LEVEL) leaves)
  a rotation   = an unannounced cyclic permutation of the w<->z map. Nothing becomes false
                 (the grammar is untouched); nothing changes about what is asked (the
                 derivation distribution is the DGP's own, fixed). Only the ADDRESS moves.

WHY THE KEY IS A MID-LEVEL NODE AND NOT THE ROOT. Keying node (KEY_LEVEL=2, KEY_NODE=0) leaves
the other three level-2 nodes unindexed, so every sequence carries its own within-sequence
admissibility control: the wall cannot touch leaves outside its span except through the (weak)
root coupling, and the exact oracle below measures exactly how weak that is. `reread/lm` also
established that d4 (= level 2 here) is the deepest level with real dynamic range at this model
size and budget, so the token-derived pathway for the keyed latent can actually form -- which
is what makes the shortcut-vs-inference race observable rather than foreclosed.

EXACTNESS. Every reference here is computed by the donor's belief propagation
(`conditional_revision.oracle`), never estimated: the per-position Bayes surprisal with and
without the key revealed (the oracle index bracket), the per-level probe ceilings under both
conditions, and the exact wall-conditional next-token distributions the model's own index
cardinality is charted against.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# token layout  (leaves | walls | neutral), shared by every arm so the model is
# bit-identical in shape whatever the arm does with position 0
# --------------------------------------------------------------------------- #

def vocab_size(v):
    """leaves 0..v-1 | wall tokens v..2v-1 | neutral 2v."""
    return 2 * v + 1


def wall_tok(w, v):
    return v + w


def neutral_tok(v):
    return 2 * v


# --------------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------------- #

def era_at(step, phase1, rot_period):
    """0 during phase 1; k>=1 after the k-th rotation event."""
    if rot_period <= 0 or step < phase1:
        return 0
    return (step - phase1) // rot_period + 1


def q_at(step, phase1, rot_period, rot_step, v):
    """The current w<->z offset: w = (z + q) mod v."""
    return (era_at(step, phase1, rot_period) * rot_step) % v


def q_prev_at(step, phase1, rot_period, rot_step, v):
    """The offset in force BEFORE the most recent rotation (== q during phase 1)."""
    e = era_at(step, phase1, rot_period)
    return ((e - 1) * rot_step) % v if e > 0 else 0


def rotation_steps(phase1, rot_period, max_steps):
    if rot_period <= 0:
        return []
    return list(range(phase1, max_steps, rot_period))


def is_derangement(q, v):
    """A rotation must move EVERY wall: no wall keeps its meaning."""
    return q % v != 0


# --------------------------------------------------------------------------- #
# the wall itself
# --------------------------------------------------------------------------- #

def wall_values(kind, z, q, v, rng=None, mode="true", q_prev=0, delta=4):
    """The wall id each instance carries, per arm kind and per EVAL MODE.

    kind   'wall' the informative wall; 'dead' a free but uninformative wall;
           'none' no wall at all (a neutral filler token).
    mode   'true' the map currently in force -- what the arm is trained under
           'rand' the key destroyed (uniform, independent of z): the index ABSENT
           'perm' a fixed derangement of the current map: the index actively MISLEADING
           'old'  the map in force before the most recent rotation: the STALE address

    Returns None for kind 'none' (the caller emits the neutral token).
    """
    n = z.shape[0]
    if kind == "none":
        return None
    if kind == "dead":
        # every mode is an independent uniform draw -- so `true` vs `rand` on this arm
        # is a pure null and whatever the binding instrument reads there is its floor
        return rng.integers(0, v, size=n)
    if mode == "true":
        return (z + q) % v
    if mode == "rand":
        return rng.integers(0, v, size=n)
    if mode == "perm":
        return (z + q + delta) % v
    if mode == "old":
        return (z + q_prev) % v
    raise ValueError(mode)


def build_tokens(leaf, z, kind, v, q=0, rng=None, mode="true", q_prev=0, delta=4):
    """[wall, leaf_0 .. leaf_{T-1}] -> (n, T+1). Model input is [:-1], targets [1:]."""
    n, T = leaf.shape
    out = np.empty((n, T + 1), dtype=np.int64)
    w = wall_values(kind, z, q, v, rng=rng, mode=mode, q_prev=q_prev, delta=delta)
    out[:, 0] = neutral_tok(v) if w is None else wall_tok(w, v)
    out[:, 1:] = leaf
    return out


# --------------------------------------------------------------------------- #
# structural helpers
# --------------------------------------------------------------------------- #

def key_span(L, s, key_level, key_node):
    """The half-open leaf range the keyed latent governs."""
    span = s ** (L - key_level)
    return key_node * span, key_node * span + span


def anc_index(plen, L, s):
    """For a prefix of `plen` leaves, the ancestor of the LAST observed leaf at each
    level -- `reread/lm`'s `last_anc`, generalised to a prefix. Level 0 = root."""
    return {ell: (plen - 1) // (s ** (L - ell)) for ell in range(L)}


def pos_top_level(T, L, s):
    """position p -> the HIGHEST hierarchy node it closes (conditional_revision's
    convention; 0 = the token closes the root)."""
    return np.array([min(ell for ell in range(L + 1) if (p + 1) % (s ** (L - ell)) == 0)
                     for p in range(T)], dtype=np.int64)


# --------------------------------------------------------------------------- #
# the exact oracles (BP; `conditional_revision.oracle`, imported, never modified)
# --------------------------------------------------------------------------- #

def _clamp_evidence(rules, level, node, vals, n, v):
    """A node-evidence list clamping ONE node of ONE level to its true value."""
    L, s = len(rules), rules[0].shape[2]
    nev = [None] * (L + 1)
    ev = np.ones((n, s ** level, v))
    ev[:, node, :] = 0.0
    ev[np.arange(n), node, vals] = 1.0
    nev[level] = ev
    return nev


def exact_predictive(rules, leaf, clamp=None, chunk=1024):
    """Exact mean per-position Bayes surprisal -log P(x_p | x_{<p} [, clamped node]).

    Returns (T,) nats. Position 0 is the MARGINAL P(x_0) -- which is exactly the
    position the wall is supposed to index, and the one `revision_and_entropy` (whose
    signal axis starts at x_1) cannot give us.

    clamp: None, or (level, node, values (n,)) -- the keyed latent revealed.
    """
    from rhm.conditional_revision.oracle import _upward, _downward, _leaf_evidence, _norm
    n, T = leaf.shape
    v = rules[0].shape[0]
    tot = np.zeros(T)
    for lo in range(0, n, chunk):
        sub = leaf[lo:lo + chunk]
        b = sub.shape[0]
        nev = None
        if clamp is not None:
            lev, node, vals = clamp
            nev = _clamp_evidence(rules, lev, node, np.asarray(vals)[lo:lo + chunk], b, v)
        for plen in range(T):
            up = _upward(_leaf_evidence(sub, plen, v), rules, nev=nev)
            down = _downward(up, rules, nev=nev)
            post = _norm((up[len(rules)] * down[len(rules)])[:, plen, :][:, None, :])[:, 0, :]
            p = post[np.arange(b), sub[:, plen]]
            tot[plen] += -np.log(np.clip(p, 1e-30, None)).sum()
    return tot / n


def exact_ceilings(rules, leaf, plen, clamp=None, chunk=1024):
    """Exact P(z_ell | x_{<plen} [, clamped node]) for the ancestor of the last observed
    leaf, per level -- the denominator every probe number is charted against.
    Keys are the repo convention d1 (shallowest) .. dL (root)."""
    from rhm.conditional_revision.oracle import _upward, _downward, _leaf_evidence, _norm
    L, s = len(rules), rules[0].shape[2]
    n = leaf.shape[0]
    v = rules[0].shape[0]
    anc = anc_index(plen, L, s)
    acc = {ell: 0.0 for ell in range(L)}
    for lo in range(0, n, chunk):
        sub = leaf[lo:lo + chunk]
        b = sub.shape[0]
        nev = None
        if clamp is not None:
            lev, node, vals = clamp
            nev = _clamp_evidence(rules, lev, node, np.asarray(vals)[lo:lo + chunk], b, v)
        up = _upward(_leaf_evidence(sub, plen, v), rules, nev=nev)
        down = _downward(up, rules, nev=nev)
        for ell in range(L):
            post = _norm(up[ell] * down[ell])
            acc[ell] += post[:, anc[ell], :].max(-1).sum()
    return {f"d{L - ell}": float(acc[ell] / n) for ell in range(L)}


def exact_leaf0_by_key(rules, key_level, key_node):
    """P(x_0 | z_{key} = a) for every a -- (v, v). The exact reference the model's own
    wall-conditional predictions (and therefore its index CARDINALITY) are read against.

    Only meaningful when the keyed node governs leaf 0, which `key_node = 0` guarantees.
    """
    from rhm.conditional_revision.oracle import _upward, _downward, _leaf_evidence, _norm
    v, m, s = rules[0].shape
    L = len(rules)
    dummy = np.zeros((v, s ** L), dtype=np.int64)
    nev = [None] * (L + 1)
    ev = np.ones((v, s ** key_level, v))
    ev[:, key_node, :] = 0.0
    ev[np.arange(v), key_node, np.arange(v)] = 1.0
    nev[key_level] = ev
    up = _upward(_leaf_evidence(dummy, 0, v), rules, nev=nev)
    down = _downward(up, rules, nev=nev)
    return _norm((up[L] * down[L])[:, 0, :][:, None, :])[:, 0, :]


# --------------------------------------------------------------------------- #
# the index instruments (model-side, but pure array maths so they gate offline)
# --------------------------------------------------------------------------- #

def jsd_matrix(P):
    """Pairwise Jensen-Shannon divergence (nats) of the rows of P (K, v)."""
    K = P.shape[0]
    P = np.clip(P, 1e-30, None)
    P = P / P.sum(-1, keepdims=True)
    lp = np.log(P)
    D = np.zeros((K, K))
    for i in range(K):
        M = 0.5 * (P[i][None, :] + P)
        lm = np.log(np.clip(M, 1e-30, None))
        D[i] = 0.5 * ((P[i][None, :] * (lp[i][None, :] - lm)).sum(-1)
                      + (P * (lp - lm)).sum(-1))
    return np.maximum(D, 0.0)


def cluster_count(D, tau):
    """Single-linkage cluster count at threshold `tau` -- the index's CARDINALITY.
    v classes = a fully keyed index; 1 class = the index has been merged away."""
    K = D.shape[0]
    parent = list(range(K))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(K):
        for j in range(i + 1, K):
            if D[i, j] <= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(K)})


def implied_map(P_model, P_exact):
    """What the model THINKS each wall means.

    For each wall id i, the latent a whose exact conditional P(x_0 | z=a) its own
    P(x_0 | w=i) is closest to (symmetric KL). Returns (assignment (K,), fit (K,)).
    This is the endogenous read of the index's routing table -- and, tracked per wall
    across a rotation, it is what decides whether re-addressing is a STEP or a SMEAR.
    """
    K, A = P_model.shape[0], P_exact.shape[0]
    Pm = np.clip(P_model, 1e-30, None); Pm /= Pm.sum(-1, keepdims=True)
    Pe = np.clip(P_exact, 1e-30, None); Pe /= Pe.sum(-1, keepdims=True)
    C = np.zeros((K, A))
    for i in range(K):
        C[i] = ((Pm[i][None, :] - Pe) * (np.log(Pm[i])[None, :] - np.log(Pe))).sum(-1)
    return C.argmin(1), C.min(1), C


def map_agreement(assign, q, v):
    """Fraction of walls whose implied meaning matches the map at offset q
    (w = (z + q) mod v  <=>  z = (w - q) mod v)."""
    idx = np.arange(len(assign))
    return float((assign == ((idx - q) % v)).mean())
