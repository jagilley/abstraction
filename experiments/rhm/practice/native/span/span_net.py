"""PORT 2 — the CORRIDOR as a native primitive: a span-level head on the generator.

WHAT THE DP PATH IS, AND WHAT THIS REPLACES. `ratchet/macros.py`'s `apply_any` executes a
committed macro by (i) masking the macro's whole span, (ii) one `block_logits` read giving a
per-block level-1 feature score for every block (the "per-block infill" evidence), (iii) a
max-sum DP over the LEARNED table `T[l] -> T[l-1] -> ... -> T[1]` that picks the single table
entry maximising the summed evidence, and (iv) rendering that entry's level-1 features through
`canon`. Steps (ii)-(iv) are the executor's whole knowledge of the chunk: the table is an
exogenous lookup consumed by an exogenous operator, and the generator neither predicts nor
conditions on it.

The span head replaces (ii)+(iii) with ONE emission: from the same masked observation, read the
trunk's own pooled per-block hidden states, condition on WHICH committed macro is being called,
and emit the span's `span` level-1 features jointly (autoregressively across the span's blocks,
so the joint constraint the DP enforces through the table can be represented). `canon` still
renders — that is the agent's own rendering, not the port's business.

WHY THIS IS THE PORT WORTH TRYING, given `teacher_slot/handle/`'s measured negative. `handle/`
piped chunk IDENTITY into the generator: a per-span head over "which committed entry covers this
(fully visible) span", supervised densely. That is an arbitrary binding — synonym splits, a
coverage cap, an index space with no content — and the plant moved negatively, dose-ordered by
engagement, even with the true table. `practice_manufactures_its_own_credit` §3½'s reading is
that identity is non-derivable (pure lookup, no compression available) while the CORRIDOR — the
span's leaf content given the entry — is derivable from the grammar, i.e. truth-typed, and the
executor should be able to hold it. This module is that second port.

DESIGN DECISIONS (choices, not derivations — each is on the record):

 * SHARES THE TRUNK, ON PURPOSE. The head reads the generator's own pooled block hiddens and its
   loss backpropagates into them, exactly as `handle/`'s macro loss did. That is the treatment:
   if corridor content interferes the way identity did, the plant guard has to say so. The head
   is a separate OUTPUT path, so `block_logits` is untouched op-for-op — which means the plant
   guard (`parse_acc` / `infill_acc`) is already the honest slots-bypassed number by
   construction, directly comparable to `handle/`'s `*_nc` columns with no leakage to argue about.
 * CONDITIONING = THE MACRO SLOT, i.e. (level, node): which committed macro is being called.
   The SPEC's phrase is "given the context and a committed entry id"; the entry a call resolves
   to is what the DP computes, so conditioning on the resolved entry would presuppose the thing
   being replaced (and, in a run, nothing supplies it — entry proposal is Port 1's). The head
   therefore takes the call's identity and emits the corridor. Its output alphabet is level-1
   FEATURES, never an entry index: no arbitrary label is ever a target, which is the whole
   identity/corridor distinction made operational.
 * JOINT EMISSION. Per span block j the head sees a projection of EVERY block of the span (so it
   has the evidence the DP sums over), the slot embedding, a global context read, and the
   features already emitted for blocks < j. Independent per-block argmax would be the level-1
   base move, not the macro.
 * OWN RNG. Head construction restores the global torch RNG state and re-initialises every
   parameter from a dedicated `torch.Generator`, so minting a head draws NOTHING from the shared
   stream and a treated arm stays bit-identical to its twin until the head's first optimizer
   step. (Gate S-1.)
 * WHEN CLOSED, THE ORIGINAL CODE PATH RUNS. A macro whose head has not passed the parity gate
   is executed by `macros.apply_any` itself — not by a re-implementation — so "below parity ->
   keep DP+infill" cannot introduce float drift.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# slots: one per (level, node) macro instantiation
# --------------------------------------------------------------------------- #

def slot_count(s, depth, max_level):
    return sum(s ** (depth - ell) for ell in range(2, max_level + 1))


def slot_index(level, node, s, depth, max_level):
    off = 0
    for ell in range(2, max_level + 1):
        if ell == level:
            return off + int(node)
        off += s ** (depth - ell)
    raise KeyError((level, node))


def slot_key(level, node):
    return f"{int(level)}:{int(node)}"


# --------------------------------------------------------------------------- #
# the trunk read (a bit-identical replica of BlockInfiller.block_logits)
# --------------------------------------------------------------------------- #

def trunk(core, observation):
    """`BlockInfiller.block_logits` split into its two halves: the pooled per-block hidden the
    span head reads, and the level-1 feature logits the DP reads. Same ops, same order, so
    `logits` here is bit-identical to `core.block_logits(observation)` (gate S-2).

    `root_conditioned` is False throughout the ratchet substrate; asserted rather than handled."""
    assert not core.root_conditioned
    tokens = observation.masked_fill(observation < 0, core.mask_token)
    h = core.token_embedding(tokens) + core.position_embedding(core.positions)
    h = core.final_norm(core.encoder(h))
    b, length, dim = h.shape
    pooled = h.view(b, core.n_blocks, core.block_size, dim).mean(dim=2)
    return pooled, core.feature_head(pooled)


def span_positions(move, batch, s, device):
    """`macro_features`' `pos`: the token indices the macro rewrites."""
    import torch
    blk0, span = move["blk0"], move["span"]
    blocks = torch.arange(blk0, blk0 + span, device=device)
    pos = (blocks[:, None] * s + torch.arange(s, device=device)[None, :]).reshape(-1)
    return pos[None, :].expand(batch, -1).contiguous()


def dp_features(logits, move, s):
    """`macros.macro_features`' max-sum DP, on already-computed logits. Verbatim, so the head's
    parity target is the executor's own materialisation and nothing else."""
    blk0, span = move["blk0"], move["span"]
    batch = logits.shape[0]
    cur = logits[:, blk0:blk0 + span, :]
    if move["level"] == 1:
        return cur.argmax(-1)
    for child in move["chain"]:
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, cur.shape[-1])
        total = None
        for i in range(s):
            picked = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
            total = picked if total is None else total + picked
        cur = total
    best = cur.argmax(-1)
    return move["flat"][best.reshape(-1)].view(batch, span)


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #

def _build_span_head():
    import torch
    import torch.nn as nn

    class SpanHead(nn.Module):
        """(pooled block hiddens, macro slot) -> the span's `span` level-1 features.

        Emission is autoregressive across the span's blocks: block j's logits see a projection
        of every block of the span (`in_proj`, one Linear per span offset), the slot embedding,
        a global context read, and the features already emitted. `max_span` = s**(max_level-1)."""

        def __init__(self, n_slots, v, dim, max_span, hidden_mult=4):
            super().__init__()
            self.v, self.dim, self.max_span = int(v), int(dim), int(max_span)
            self.slot = nn.Embedding(int(n_slots), int(dim))
            self.step = nn.Embedding(int(max_span), int(dim))
            self.feat = nn.Embedding(int(v) * int(max_span), int(dim))
            self.ctx = nn.Linear(int(dim), int(dim))
            self.in_proj = nn.ModuleList(
                [nn.Linear(int(dim), int(dim)) for _ in range(int(max_span))])
            h = int(dim) * int(hidden_mult)
            self.mlp = nn.Sequential(nn.Linear(int(dim), h), nn.GELU(), nn.Linear(h, int(v)))

        def span_state(self, pooled, blk0, span, slot_id):
            u = self.ctx(pooled.mean(dim=1)) + self.slot(slot_id)
            for j in range(span):
                u = u + self.in_proj[j](pooled[:, blk0 + j, :])
            return u

        def forward(self, pooled, blk0, span, slot_id, teacher=None):
            """teacher: (B, span) ground-truth features for teacher forcing; None = free run.
            Returns (logits (B, span, v), emitted (B, span))."""
            u = self.span_state(pooled, blk0, span, slot_id)
            outs, emitted, state = [], [], u
            for j in range(span):
                lg = self.mlp(state + self.step.weight[j])
                outs.append(lg)
                f = teacher[:, j] if teacher is not None else lg.argmax(-1)
                emitted.append(f)
                state = state + self.feat(f + self.v * j)
            return torch.stack(outs, dim=1), torch.stack(emitted, dim=1)

        def emit(self, pooled, blk0, span, slot_id):
            return self.forward(pooled, blk0, span, slot_id)[1]

    return SpanHead


def build_head(n_slots, v, dim, max_span, seed, device, hidden_mult=4):
    """Mint the head WITHOUT touching the shared torch stream: save the global RNG state,
    construct (module constructors draw from it), restore it, then re-initialise every
    parameter from a dedicated `torch.Generator`. Gate S-1 asserts the restoration."""
    import torch
    st = torch.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    head = _build_span_head()(n_slots, v, dim, max_span, hidden_mult=hidden_mult)
    torch.set_rng_state(st)
    if cst is not None:
        torch.cuda.set_rng_state_all(cst)
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for p in head.parameters():
            if p.dim() >= 2:
                p.copy_(torch.empty(p.shape, dtype=p.dtype).normal_(0.0, 0.02, generator=g))
            else:
                p.zero_()
    return head.to(device)


# --------------------------------------------------------------------------- #
# executors: what the beam calls to materialise one move
# --------------------------------------------------------------------------- #

class PlainExecutor:
    """`ratchet`'s execution, plus the block-level tally the counterfactual ledger needs.
    Used by every untreated arm AND by every treated arm's closed slots, so no arm's numbers
    can drift from `ratchet`'s through a re-implementation."""

    kind = "plain"

    def __init__(self):
        self.counts = new_counts()

    def reset(self):
        self.counts = new_counts()

    def apply(self, generator, x, move, rules_t, canon, depth, v, m, s):
        from rhm.practice.ratchet import macros as MC
        n = x.shape[0]
        if move.get("kind") != "macro":
            self.counts["mat_base"] += n
            self.counts["blk_base"] += n * int(move["span"])
        else:
            self.counts["mat_dp"] += n
            self.counts["blk_dp"] += n * int(move["span"])
        return MC.apply_any(generator, x, move, rules_t, canon, depth, v, m, s)


class SpanExecutor(PlainExecutor):
    """The treated executor. A macro whose slot is OPEN (parity gate passed) is materialised by
    the head in one pass — no DP, no table consulted at execution time. A macro whose slot is
    CLOSED falls back to `macros.apply_any` verbatim.

    Every macro call, open or closed, contributes its masked observation to that slot's
    self-imitation buffer (train / held-out split by a DEDICATED rng, so the shared stream is
    undisturbed). Targets are NOT stored: they are recomputed from the current executor at
    training time, so the head chases the executor it is replacing rather than a stale copy."""

    kind = "span"

    def __init__(self, core, head, slots, *, cap, hold_cap, hold_frac, per_call, seed,
                 v=8, length=16):
        import torch
        super().__init__()
        self.core, self.head = core, head
        self.slots = slots                      # slot_key -> {"move":…, "id":int, "open":bool}
        self.cap, self.hold_cap = int(cap), int(hold_cap)
        self.hold_frac, self.per_call = float(hold_frac), int(per_call)
        self.buf, self.hold = {}, {}
        self.rng = np.random.default_rng(int(seed))
        self.fire = True
        self.capture = True
        # The train/held-out split is keyed by a BIJECTIVE code of the observation row, not by a
        # coin flip. The metering set is fixed per era, so the same masked context recurs across
        # cycles; a coin flip would put copies of one context on both sides and the parity gate
        # would be reading its own training data.
        self.powers = (int(v) + 1) ** torch.arange(int(length), dtype=torch.long)

    # -- self-imitation buffer ---------------------------------------------------------- #
    def _store(self, obs, key):
        import torch
        n = obs.shape[0]
        take = min(self.per_call, n)
        idx = torch.from_numpy(self.rng.permutation(n)[:take]).to(obs.device)
        rows = obs.index_select(0, idx).detach().cpu()
        code = ((rows + 1) * self.powers).sum(1) % 1000003
        is_hold = (code * 48271) % 100 < int(round(self.hold_frac * 100))
        for tgt, cap, sel in ((self.hold, self.hold_cap, is_hold),
                              (self.buf, self.cap, ~is_hold)):
            part = rows[sel]
            if part.shape[0] == 0:
                continue
            tgt[key] = part if key not in tgt else torch.cat([tgt[key], part])[-cap:]

    def sizes(self):
        return {k: int(t.shape[0]) for k, t in self.buf.items()}, \
               {k: int(t.shape[0]) for k, t in self.hold.items()}

    # -- execution ---------------------------------------------------------------------- #
    def apply(self, generator, x, move, rules_t, canon, depth, v, m, s, chunk=16384):
        import torch
        if move.get("kind") != "macro":
            return super().apply(generator, x, move, rules_t, canon, depth, v, m, s)
        key = slot_key(move["level"], move["node"])
        info = self.slots.get(key)
        pos = span_positions(move, x.shape[0], s, x.device)
        if self.capture and info is not None:
            self._store(x.clone().scatter_(1, pos, torch.full_like(pos, -1)), key)
        if not (self.fire and info is not None and info["open"]):
            return super().apply(generator, x, move, rules_t, canon, depth, v, m, s)
        if x.shape[0] > chunk:
            return torch.cat([self.apply(generator, x[i:i + chunk], move, rules_t, canon,
                                         depth, v, m, s, chunk=chunk)
                              for i in range(0, x.shape[0], chunk)], dim=0)
        n = x.shape[0]
        self.counts["mat_head"] += n
        self.counts["blk_head"] += n                       # ONE pass, whatever the span
        obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
        with torch.no_grad():
            pooled, _ = trunk(self.core, obs)
            feats = self.head.emit(pooled, move["blk0"], move["span"],
                                   torch.full((n,), info["id"], dtype=torch.long,
                                              device=x.device))
        new = x.clone()
        new.scatter_(1, pos, canon[feats].reshape(n, -1))
        return new


def new_counts():
    return {"mat_base": 0, "mat_dp": 0, "mat_head": 0,
            "blk_base": 0, "blk_dp": 0, "blk_head": 0}


def add_counts(dst, src):
    for k in dst:
        dst[k] += src.get(k, 0)


# --------------------------------------------------------------------------- #
# the parity gate
# --------------------------------------------------------------------------- #

def parity(core, head, ex, slots, s, v, min_hold=256, chunk=4096):
    """Held-out DP+infill parity, per slot: on observations the head has NEVER been trained on
    (the executor's ~10% held-out split of its own call distribution, keyed by a bijective code
    of the observation so a recurring context can never straddle the split), the exact-match rate of
    the head's emitted span features against what the DP would write on the same instances,
    with the CURRENT trunk. `block` is the same comparison per block, reported so a different
    threshold can be read off the record.

    Measured on the deployment distribution — the beam's own macro calls — rather than on fresh
    damaged instances, because that is the distribution the head is about to be let loose on."""
    import torch
    out = {}
    for key, info in slots.items():
        rows = ex.hold.get(key)
        if info.get("move") is None or rows is None or rows.shape[0] < min_hold:
            out[key] = {"n": 0 if rows is None else int(rows.shape[0]),
                        "exact": None, "block": None}
            continue
        move = info["move"]
        dev = move["flat"].device
        ex_hits, blk_hits, tot = 0, 0, 0
        with torch.no_grad():
            for i in range(0, rows.shape[0], chunk):
                obs = rows[i:i + chunk].to(dev)
                pooled, logits = trunk(core, obs)
                tgt = dp_features(logits, move, s)
                sid = torch.full((obs.shape[0],), info["id"], dtype=torch.long, device=dev)
                got = head.emit(pooled, move["blk0"], move["span"], sid)
                ex_hits += int((got == tgt).all(-1).sum())
                blk_hits += int((got == tgt).sum())
                tot += obs.shape[0]
        out[key] = {"n": tot, "exact": ex_hits / tot,
                    "block": blk_hits / (tot * int(move["span"]))}
    return out


def span_train_terms(core, head, ex, slots, s, batch, rng, device):
    """The self-imitation loss, one term per minted slot, added to the plant's own loss in the
    SAME optimizer step (no extra steps for the treatment — `handle/`'s convention). Targets are
    the current executor's own DP materialisations on stored contexts, so this is distillation
    of the path being replaced, not regression onto stale labels.

    `pooled` carries gradient, so the span loss reaches the shared trunk. That is the treatment."""
    import torch
    import torch.nn.functional as F
    terms, accs = [], {}
    for key, info in slots.items():
        rows = ex.buf.get(key)
        if info.get("move") is None or rows is None or rows.shape[0] < 8:
            continue
        idx = torch.from_numpy(rng.integers(0, rows.shape[0], size=min(batch, rows.shape[0])))
        obs = rows[idx].to(device)
        move = info["move"]
        pooled, logits = trunk(core, obs)
        with torch.no_grad():
            tgt = dp_features(logits.detach(), move, s)
        sid = torch.full((obs.shape[0],), info["id"], dtype=torch.long, device=device)
        lg, _ = head(pooled, move["blk0"], move["span"], sid, teacher=tgt)
        terms.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)))
        accs[key] = float((lg.argmax(-1) == tgt).all(-1).float().mean())
    if not terms:
        return None, accs
    return sum(terms) / len(terms), accs
