"""EARNED vocabulary: macro moves whose composition table is mined from the agent's own
successful repairs instead of read off the true grammar.

The crystallize round's committed unit was a whole-solve move PROGRAM. The ratchet round's
committed unit is a **macro action** — a new primitive in the practice action space — because
that is what "the chunk becomes the new syllable" means operationally: after commitment the
agent can *do* a thing it could not do before, and the next era's practice searches over it.

WHAT A MACRO IS. `units.build_move_set` gives a level-ell move that commits a span to one
level-ell feature by a max-sum DP over `rules_t` — the TRUE composition tables. That move is
privileged: above level 1 it reads the DGP. A macro is the same operator with a **learned**
table in place of the true one:

    T[1]  = the v level-1 features                      (native: the generator predicts these)
    T[l]  = a set of ENTRIES, each an s-tuple of T[l-1] entry indices

Scoring is the identical max-sum DP over the learned table, so with T[l] equal to the full
legal set the macro is bit-identical to the true level-l move (gate C-M in `ratchet.selfcheck`).
The whole earned-vs-given contrast is therefore *only* about which tuples are in the table.

WHY IT RATCHETS. T[l] is defined over T[l-1] ENTRIES, not over raw tokens: a level-3 chunk
whose two halves are not both in the level-2 vocabulary is not representable and is dropped at
build time. So the level-2 vocabulary is a hard bound on the level-3 vocabulary — committing an
incomplete T[2] caps T[3] forever, which is the poisoned-region test with a mechanism.

WHERE THE TUPLES COME FROM. Only from configurations the agent actually SOLVED (terminal
possible-set success), parsed by the agent's OWN generator (`block_logits.argmax` on the
unmasked config) rather than by the exact bottom inverse map — so nothing above the generator's
own competence enters the vocabulary. Agreement with the exact map is logged as an oracle
readout, never consumed.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

def _flatten(child, lower_flat):
    """(n, s) child indices + the lower table's (n_lo, span_lo) level-1 expansion
    -> (n, s*span_lo) level-1 features."""
    if child.shape[0] == 0:
        return np.zeros((0, lower_flat.shape[1] * child.shape[1]), np.int64)
    return lower_flat[child].reshape(child.shape[0], -1)


def base_table(v):
    """T[1]: the v level-1 features, each its own entry."""
    idx = np.arange(v, dtype=np.int64)[:, None]
    return {"level": 1, "child": idx, "flat": idx, "feature": idx[:, 0], "lower": None}


def make_table(level, child, lower, s, feature=None):
    child = np.asarray(child, np.int64).reshape(-1, s)
    return {"level": level, "child": child, "flat": _flatten(child, lower["flat"]),
            "feature": (np.full(child.shape[0], -1, np.int64) if feature is None
                        else np.asarray(feature, np.int64)),
            "lower": lower}


def true_tables(rules, depth, s, v, m, max_level):
    """The DGP's own vocabulary, in the macro representation. `given`'s table.

    Level l has v*m*(entries per child)^s entries, because the learned representation cannot
    know that two tuples are SYNONYMS of one parent feature — it keeps them as distinct
    entries. That is deliberate: it is exactly the knowledge an agent mining its own repairs
    does not have, and the max-sum DP is invariant to the distinction."""
    tables = {1: base_table(v)}
    for ell in range(2, max_level + 1):
        lower = tables[ell - 1]
        by_feat = {}
        for i, f in enumerate(lower["feature"]):
            by_feat.setdefault(int(f), []).append(i)
        layer = rules[depth - ell]                      # (v, m, s): level-ell -> level-(ell-1)
        rows, feats = [], []
        for f in range(v):
            for r in range(m):
                kids = layer[f, r]                      # (s,) level-(ell-1) feature ids
                combos = [[]]
                for c in kids:
                    combos = [pre + [j] for pre in combos for j in by_feat[int(c)]]
                for cb in combos:
                    rows.append(cb); feats.append(f)
        tables[ell] = make_table(ell, np.array(rows, np.int64), lower, s, feats)
    return tables


# --------------------------------------------------------------------------- #
# mining
# --------------------------------------------------------------------------- #

class Miner:
    """Counts level-1 feature tuples observed over one span of one level, over the whole run.

    Keying by the *flattened level-1 tuple* (not by lower-table entry ids) keeps the counts
    stable while the lower table is still growing; the ratchet constraint is applied at
    `build` time instead, where a span whose halves are not in the lower vocabulary is
    dropped."""

    def __init__(self, level, s):
        self.level = level
        self.s = s
        self.span = s ** (level - 1)
        self.counts = {}
        self.n_obs = 0

    def observe(self, feats):
        """feats: (N, span) level-1 feature ids read off SOLVED configurations."""
        feats = np.asarray(feats, np.int64)
        if feats.size == 0:
            return
        self.n_obs += feats.shape[0]
        for row in feats:
            k = tuple(int(x) for x in row)
            self.counts[k] = self.counts.get(k, 0) + 1

    def build(self, lower, support):
        """Entries with count >= support whose s halves are all present in `lower`."""
        half = self.span // self.s
        lut = {tuple(int(x) for x in row): i for i, row in enumerate(lower["flat"])}
        rows = []
        for k, c in sorted(self.counts.items()):
            if c < support:
                continue
            kids = [lut.get(k[i * half:(i + 1) * half]) for i in range(self.s)]
            if any(j is None for j in kids):
                continue
            rows.append(kids)
        child = np.array(rows, np.int64).reshape(len(rows), self.s)
        return make_table(self.level, child, lower, self.s)

    def state(self):
        return {"level": self.level, "n_obs": self.n_obs, "n_distinct": len(self.counts),
                "n_at_support": {str(t): int(sum(1 for c in self.counts.values() if c >= t))
                                 for t in (1, 2, 3, 5, 10)}}


def parse_features(generator, x, s=None, mask_block=None, chunk=16384):
    """The agent's OWN parse of a configuration into level-1 features: the generator's block
    head. Native — no inverse map, no rules.

    `mask_block` masks ONE block elsewhere in the configuration before reading. This is not
    cosmetic: the generator is trained by masked infilling with `mask_min=1`, so it has NEVER
    seen a fully unmasked input, and reading one is out of distribution. `calr_s0` measured
    the cost — 0.63 block accuracy on a *visible* block, which compounds to 0.63**4 over a
    level-3 span and made the mined level-3 vocabulary mostly false entries. Masking one block
    outside the span being read puts the input back in distribution while leaving every block
    of the span visible."""
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            xb = x[i:i + chunk]
            if mask_block is not None:
                pos = (mask_block * s + torch.arange(s, device=xb.device))[None, :]
                xb = xb.clone().scatter_(1, pos.expand(xb.shape[0], -1),
                                         torch.full((xb.shape[0], s), -1, dtype=xb.dtype,
                                                    device=xb.device))
            outs.append(generator.block_logits(xb).argmax(-1))
    return torch.cat(outs)


def exact_features(x_np, inverse_bottom, v, s):
    """The exact parse (oracle readout only): -1 where a block is off-grammar."""
    b = x_np.shape[0]
    n_blocks = x_np.shape[1] // s
    powers = v ** np.arange(s)
    codes = (x_np.reshape(b, n_blocks, s) * powers).sum(-1)
    return inverse_bottom[codes]


# --------------------------------------------------------------------------- #
# the macro operator
# --------------------------------------------------------------------------- #

def make_macro(level, node, s, table, name=None):
    span = s ** (level - 1)
    return {"level": level, "node": node, "blk0": node * span, "span": span,
            "table": table, "kind": "macro",
            "name": name or f"M{level}n{node}"}


def macro_features(generator, x, move, s, v):
    """Max-sum DP over the LEARNED table. Mirrors `units.node_features` exactly, with
    `move['table']` in place of `rules_t`. Returns (feats (B, span), pos (B, span*s))."""
    import torch
    blk0, span = move["blk0"], move["span"]
    table = move["table"]
    batch = x.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=x.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=x.device)[None, :]).reshape(-1)
    pos = pos[None, :].expand(batch, -1).contiguous()

    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = generator.block_logits(obs)                      # (B, n_blocks, v)
    cur = logits[:, blk0:blk0 + span, :]                      # (B, span, v) = scores over T[1]
    if move["level"] == 1:
        return cur.argmax(-1), pos

    chain = move["chain"]                                     # list of (n, s) child arrays, l=2..L
    for child in chain:
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, cur.shape[-1])
        total = None
        for i in range(s):
            picked = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
            total = picked if total is None else total + picked
        cur = total                                           # (B, n_par, n_entries)
    best = cur.argmax(-1)                                     # (B, 1) entry at the top
    flat = move["flat"]                                       # (n, span) level-1 features
    return flat[best.reshape(-1)].view(batch, span), pos


def to_device(move, device):
    """Materialise the table chain as device tensors once, so the DP does no host work."""
    import torch
    if move.get("kind") != "macro":
        return move
    out = dict(move)
    tbl = move["table"]
    stack, t = [], tbl
    while t is not None and t["level"] >= 2:                  # top table down to level 2
        stack.append(t)
        t = t.get("lower")
    out["chain"] = [torch.from_numpy(x["child"]).to(device) for x in reversed(stack)]
    out["flat"] = torch.from_numpy(tbl["flat"]).to(device)
    out["n_entries"] = int(tbl["child"].shape[0])
    return out


def apply_any(generator, x, move, rules_t, canon, depth, v, m, s, chunk=16384):
    """One move on every row, dispatching on move kind (base level move vs earned macro)."""
    import torch
    from rhm.practice.crystallize.units import apply_move
    if move.get("kind") != "macro":
        return apply_move(generator, x, move, rules_t, canon, depth, v, m, s, chunk=chunk)
    if x.shape[0] > chunk:
        return torch.cat([apply_any(generator, x[i:i + chunk], move, rules_t, canon,
                                    depth, v, m, s, chunk=chunk)
                          for i in range(0, x.shape[0], chunk)], dim=0)
    feats, pos = macro_features(generator, x, move, s, v)
    tup = canon[feats]
    new = x.clone()
    new.scatter_(1, pos, tup.reshape(x.shape[0], -1))
    return new


def apply_seq(generator, x, seq, ms, rules_t, canon, depth, v, m, s):
    n_mat = 0
    for k in seq:
        x = apply_any(generator, x, ms[int(k)], rules_t, canon, depth, v, m, s)
        n_mat += x.shape[0]
    return x, n_mat


# --------------------------------------------------------------------------- #
# the oracle check: does an earned table correspond to the true rules?
# --------------------------------------------------------------------------- #

def grade_table(learned, truth):
    """Precision / recall of an earned table against the DGP's own, as SETS of flattened
    level-1 tuples. Precision < 1 means the agent recorded a chunk that is not a legal
    derivation of any feature at that level; recall is coverage of the true vocabulary."""
    L = {tuple(int(x) for x in r) for r in learned["flat"]}
    T = {tuple(int(x) for x in r) for r in truth["flat"]}
    inter = len(L & T)
    return {"n_learned": len(L), "n_true": len(T), "n_correct": inter,
            "precision": (inter / len(L)) if L else None,
            "recall": (inter / len(T)) if T else 0.0}
