"""The two ports, on the motor substrate.

`native/` piped an earned RHM vocabulary back into the learner along two ports and kept the table
outside as the address book. On mjc nothing was ever consolidated: a committed unit is always an
external object -- a keyed tape (`fingering/`) or a live CEM plan at launch (`legato/`). This module
is the arm's version of `native/prop_net.py` + `native/span_net.py`.

  PORT 1 -- ROUTING.  pi(unit-slot | seam state).  The seam is the decision point and the library is
  the action set, so the thing that becomes native is WHICH unit to call from the posture the body
  actually reached. Trained ONLINE by self-imitation on the agent's own SUCCESSFUL traversals
  (selection before regression -- the licensing condition; the targets are chosen by the audition and
  filtered by the traversal's realised score before any gradient sees them). At a seam only the top-k
  proposed slots are auditioned, so the O(K) enumeration cost becomes O(k).
    * `select_slots` is an EXACT no-op at k = n_legal: a stable descending sort truncated at k and
      re-sorted ascending returns the legal set itself, whatever the logits are, ties included. That
      is gate G-P's precondition, and it holds by construction rather than by luck.
    * ZERO-INIT output layer: a slot minted at a commit enters the ranking NEUTRALLY (above what the
      head has learned to reject, below what it has learned to accept) and costs no RNG at the
      instant it is built.
    * PRACTICE EXPLORES, PERFORMANCE DOES NOT. `native/prop`'s smoke measured closed-loop lock-in --
      a move never proposed never enters a chosen trajectory, so never becomes a target. On this
      substrate the same trap is sharper, because a slot that is never auditioned also never gets a
      seam-time score, so nothing else can rescue it either. epsilon-exploration draws from a
      DEDICATED numpy stream, so the shared per-arm stream is undisturbed.

  PORT 2 -- CORRIDOR.  A span head (slot, seam posture, FM trunk read) -> the unit's measured
  COMMANDS. Never the library key: identity is the INPUT, content is the target -- the whole point of
  `teacher_slot/handle/`'s measured negative (identity-as-target is poison) and of the sec. 3-1/2
  split `native/` confirmed. It reads the FORWARD MODEL's own trunk, deliberately: the interference
  question is the treatment, and the plant guard (`e_react` + one-step FM corridor error) is the mjc
  analogue of RHM's parse/infill staying inert. Behind a PER-SLOT parity gate re-checked every cycle;
  a slot below parity is executed by tape playback verbatim, so "below parity -> keep the tape" cannot
  introduce drift.

  The arm-specific readout Port 2 makes available, which RHM could not ask: does a state-conditioned
  head beat VERBATIM tape playback on OFF-KEY seams? `fingering/`'s `regress` failed at exactly this
  unconditionally and conditioning rescued it; here the head is conditioned by construction, so the
  question is whether a chunk's content generalises WITHIN the chunk.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# the slot vocabulary: one block per (level, seam), plus one primitive slot per seam
# --------------------------------------------------------------------------- #

class SlotLayout:
    """Stable global slot ids. A cell is (ns, k): a unit spanning `ns` drilled segments starting at
    seam `k`. Two levels are populated -- ns = 1 (a segment tape) and ns = n_seg - k (a chain running
    to the end of the piece), the second past the plant's composition horizon, which is legato F4's
    precondition for a chunk paying at all. Every cell reserves the SAME number of ids whether or not
    it is populated, so the layout is a property of the piece and not of the run's history, and pi's
    logits mean the same thing at cycle 1 and at cycle 90.
    """

    def __init__(self, n_seg, n_slot, poison=True):
        self.n_seg = int(n_seg)
        self.n_slot = int(n_slot)
        self.width = int(n_slot) + (1 if poison else 0)   # +1 reserved for the poison twin
        self.poison = bool(poison)
        self.off = {}
        cur = 0
        for k in range(self.n_seg):
            for ns in self.levels(k):
                self.off[(ns, k)] = cur
                cur += self.width
        self.prim_off = cur
        cur += self.n_seg
        self.n_slots = cur

    def levels(self, k):
        """The unit spans available at seam k: one segment, and (if it differs) everything left."""
        rest = self.n_seg - int(k)
        return [1] if rest <= 1 else [1, rest]

    def slot(self, ns, k, j):
        return self.off[(int(ns), int(k))] + int(j)

    def prim(self, k):
        return self.prim_off + int(k)

    def cell_of(self, sid):
        """(ns, k, j) or ('prim', k, 0) for a global slot id."""
        if sid >= self.prim_off:
            return ("prim", int(sid - self.prim_off), 0)
        for (ns, k), o in self.off.items():
            if o <= sid < o + self.width:
                return (ns, k, int(sid - o))
        raise KeyError(sid)


# --------------------------------------------------------------------------- #
# RNG isolation (native/prop_net's discipline, verbatim in spirit)
# --------------------------------------------------------------------------- #

class isolated_rng:
    """Run a block with the global torch RNG saved and restored, seeded from a dedicated stream.

    Any NEW consumer of randomness (minting a head) therefore leaves the shared per-arm stream
    exactly where it found it, which is what licenses the `fid` twin gate: an arm with both ports
    wired and shut must be bit-identical to `audit_all` for the whole run.
    """

    def __init__(self, seed, device=None):
        self.seed = int(seed)
        self.cuda = device is not None and str(device).startswith("cuda")

    def __enter__(self):
        import torch
        self._cpu = torch.get_rng_state()
        self._gpu = torch.cuda.get_rng_state_all() if self.cuda else None
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, *exc):
        import torch
        torch.set_rng_state(self._cpu)
        if self._gpu is not None:
            torch.cuda.set_rng_state_all(self._gpu)
        return False


# --------------------------------------------------------------------------- #
# PORT 1 -- the proposal head
# --------------------------------------------------------------------------- #

def _build_prop_head():
    import torch
    import torch.nn as nn

    class PropHead(nn.Module):
        """pi(slot | seam state, seam index) -> logits over the fixed slot vocabulary.

        The read is the SEAM POSTURE, which is `fingering/` G1's measured object: the hand arrives
        within ~3 cm of the waypoint while the arrival POSTURE spreads 0.29 rad along the
        self-motion manifold, and a per-state oracle over committed units is worth 4.0x. pi is
        exactly a learned, cheap approximation to that oracle's argmin.
        """

        def __init__(self, state_dim, n_seg, n_slots, hidden=128, layers=2):
            super().__init__()
            self.n_slots = int(n_slots)
            din = int(state_dim) + int(n_seg)
            lyr = [torch.nn.Linear(din, hidden), torch.nn.SiLU()]
            for _ in range(int(layers) - 1):
                lyr += [torch.nn.Linear(hidden, hidden), torch.nn.SiLU()]
            self.trunk = nn.Sequential(*lyr)
            self.out = nn.Linear(hidden, int(n_slots))
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, z, seam_onehot):
            return self.out(self.trunk(torch.cat([z, seam_onehot], dim=-1)))

    return PropHead


def build_prop(state_dim, n_seg, n_slots, seed, device, hidden=128, layers=2):
    import torch  # noqa: F401
    with isolated_rng(seed, device):
        head = _build_prop_head()(state_dim, n_seg, n_slots, hidden=hidden, layers=layers)
    return head.to(device)


def select_slots(logits, legal, k, forced=(), explore_idx=None):
    """logits: (B, n_slots) numpy. legal: (L,) global slot ids legal at this seam.

    Returns a list of B index arrays (into `legal`) -- the slots to audition, ASCENDING.

    THE FIDELITY PRECONDITION. `np.argsort(..., kind='stable')` on the negated logits orders ties by
    ascending index, so at k >= len(legal) the truncation keeps every legal slot and the ascending
    re-sort returns `arange(len(legal))` bit-for-bit whatever the logits are. `audit_prop_k` at
    k = K is then the SAME candidate set, in the SAME order, as `audit_all`, and gate G-P is an
    assertion about arithmetic rather than about training.

    `forced` (positions into `legal`) is always auditioned in addition -- the primitive action is
    never ranked out by an untrained logit, and its cost is charged at its true price.
    """
    L = len(legal)
    kk = int(min(int(k), L))
    sub = logits[:, np.asarray(legal, int)]
    order = np.argsort(-sub, axis=1, kind="stable")
    out = []
    forced = np.asarray(sorted(set(int(f) for f in forced)), int)
    for b in range(sub.shape[0]):
        sel = order[b, :kk]
        if forced.size:
            sel = np.concatenate([sel, forced])
        if explore_idx is not None and len(explore_idx[b]):
            sel = np.concatenate([sel, np.asarray(explore_idx[b], int)])
        out.append(np.unique(sel))
    return out


def explore_slots(sel, n_legal, eps, rng):
    """One uniformly-drawn NOT-proposed legal slot per performer with probability `eps`.

    Practice only. Drawn from a dedicated stream and returning an empty draw when the selection
    already covers the legal set, so at k = n_legal this consumes nothing that could break G-P.
    """
    out = []
    for s in sel:
        rest = np.setdiff1d(np.arange(int(n_legal)), np.asarray(s, int), assume_unique=False)
        if rest.size == 0 or eps <= 0.0 or rng.random() >= eps:
            out.append(np.zeros(0, int))
        else:
            out.append(np.array([int(rng.choice(rest))], int))
    return out


class PropTrainer:
    """Self-imitation on the agent's own successful traversals.

    A pair enters the buffer only when (a) the audition CHOSE that slot at that seam state and
    (b) the traversal that decision belonged to came in at or under the batch's median piece error.
    Selection before regression, twice over: the argmin picks the target and the realised score
    filters it. Masked cross-entropy over the LEGAL slot set at that seam, so a slot that does not
    exist yet is never a target and never a competitor; a pair stored before a commit is replayed
    under the CURRENT mask, which only ever adds options.
    """

    def __init__(self, head, n_slots, lr, cap, seed, device):
        import torch
        self.head, self.n_slots, self.device = head, int(n_slots), device
        self.opt = torch.optim.Adam(head.parameters(), lr=float(lr))
        self.cap = int(cap)
        self.rng = np.random.default_rng(int(seed))
        self.Z, self.K, self.Y = [], [], []          # seam state, seam index, target slot
        self.n_seen = 0

    def add(self, Z, K, Y):
        if len(Z) == 0:
            return
        self.Z.append(np.asarray(Z, np.float32))
        self.K.append(np.asarray(K, np.int64))
        self.Y.append(np.asarray(Y, np.int64))
        self.n_seen += len(Z)
        tot = sum(len(z) for z in self.Z)
        while tot > self.cap and len(self.Z) > 1:
            tot -= len(self.Z.pop(0)); self.K.pop(0); self.Y.pop(0)

    def size(self):
        return int(sum(len(z) for z in self.Z))

    def train(self, steps, batch, legal_by_seam, norm, n_seg):
        """`legal_by_seam[k]` = global slot ids legal at seam k, under the CURRENT library."""
        import torch
        import torch.nn.functional as F
        if not self.Z:
            return None
        Z = np.concatenate(self.Z); K = np.concatenate(self.K); Y = np.concatenate(self.Y)
        n = len(Z)
        mask = np.zeros((n_seg, self.n_slots), bool)
        for k, ids in legal_by_seam.items():
            mask[int(k), np.asarray(ids, int)] = True
        Zt = torch.tensor((Z - norm[0]) / norm[1], device=self.device)
        Kt = torch.tensor(K, device=self.device)
        Yt = torch.tensor(Y, device=self.device)
        Mt = torch.tensor(mask, device=self.device)
        oh = F.one_hot(Kt, num_classes=int(n_seg)).float()
        self.head.train()
        last = None
        for _ in range(int(steps)):
            idx = torch.tensor(self.rng.integers(0, n, size=min(int(batch), n)),
                               device=self.device)
            lg = self.head(Zt[idx], oh[idx])
            lg = lg.masked_fill(~Mt[Kt[idx]], -float("inf"))
            loss = F.cross_entropy(lg, Yt[idx])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            last = float(loss)
        self.head.eval()
        return last


def prop_probe(head, Z, K, legal_by_seam, norm, n_seg, n_slots, layout, device,
               spellings=None):
    """Where does pi put its mass? Two readouts, both queued by `census/` and never yet run here.

    TRUST FORMATION -- per-level proposal mass over cycles. `native/`+`census/` found that the
    deep-era value of a vocabulary rides mostly on ARRIVAL: trust, pi's per-level proposal mass,
    forms only by time-in-use. This is the series that measures it on a plant.

    THE CAN'T-DECOMPOSE SIGNATURE -- on seam states where pi's argmax is a CHAIN slot, how much mass
    does it keep on that chain's own SEGMENT spelling (the ns=1 slots its members were built out of)?
    An expert cannot decompose their chunks. `spellings[sid]` is the list of ns=1 global slot ids the
    chain slot's members were spelled with.
    """
    import torch
    import torch.nn.functional as F
    head.eval()
    Zt = torch.tensor((np.asarray(Z, np.float32) - norm[0]) / norm[1], device=device)
    Kt = torch.tensor(np.asarray(K, np.int64), device=device)
    oh = F.one_hot(Kt, num_classes=int(n_seg)).float()
    mask = np.zeros((n_seg, n_slots), bool)
    for k, ids in legal_by_seam.items():
        mask[int(k), np.asarray(ids, int)] = True
    Mt = torch.tensor(mask, device=device)
    with torch.no_grad():
        lg = head(Zt, oh).masked_fill(~Mt[Kt], -float("inf"))
        p = torch.softmax(lg, dim=-1).cpu().numpy()
    arg = p.argmax(1)
    out = {"n": int(len(p)), "entropy": float(np.mean(
        -(np.where(p > 0, p, 1.0) * np.log(np.where(p > 0, p, 1.0))).sum(1))),
        "top1": float(p.max(1).mean())}
    # per-level mass (the trust series)
    lvl = {"seg": [], "chain": [], "prim": [], "poison": []}
    for sid in range(n_slots):
        ns, k, j = layout.cell_of(sid)
        if ns == "prim":
            lvl["prim"].append(sid)
        elif ns == 1:
            (lvl["poison"] if (layout.poison and j == layout.n_slot) else lvl["seg"]).append(sid)
        else:
            (lvl["poison"] if (layout.poison and j == layout.n_slot) else lvl["chain"]).append(sid)
    for nm, ids in lvl.items():
        out[f"mass_{nm}"] = float(p[:, ids].sum(1).mean()) if ids else 0.0
        out[f"argmax_{nm}"] = float(np.isin(arg, ids).mean()) if ids else 0.0
    out["mass_by_slot"] = [float(x) for x in p.mean(0)]
    out["argmax_hist"] = [int(x) for x in np.bincount(arg, minlength=n_slots)]
    # the can't-decompose signature
    rows = []
    if spellings:
        for sid, spell in spellings.items():
            ns, k, j = layout.cell_of(sid)
            if ns == "prim" or ns == 1 or not spell:
                continue
            hit = arg == sid
            cell = {"slot": int(sid), "ns": int(ns), "seam": int(k),
                    "n_argmax": int(hit.sum()), "frac_argmax": float(hit.mean()),
                    "p_chain_all": float(p[:, sid].mean()),
                    "p_spell_all": float(p[:, list(spell)].sum(1).mean())}
            if int(hit.sum()):
                cell["p_chain"] = float(p[hit, sid].mean())
                cell["p_spell"] = float(p[hit][:, list(spell)].sum(1).mean())
                cell["ratio"] = cell["p_spell"] / max(cell["p_chain"], 1e-9)
            rows.append(cell)
    out["chains"] = rows
    return out


# --------------------------------------------------------------------------- #
# PORT 2 -- the span head
# --------------------------------------------------------------------------- #

def _build_span_head():
    import torch
    import torch.nn as nn

    class SpanHead(nn.Module):
        """(slot, seam posture, forward-model trunk read) -> the unit's measured commands.

        Output is `max_span * AD` tanh-squashed command values, sliced to the called unit's own
        span. The alphabet is COMMANDS: no arbitrary label is ever a target, which is the
        identity/corridor distinction made operational.

        The FM trunk read is what makes this the treatment rather than a private regressor: the span
        loss backpropagates into the forward model's own hidden layers, exactly as `handle/`'s macro
        loss reached the generator's. If corridor content interferes the way identity did, the plant
        guard has to say so.
        """

        def __init__(self, n_slots, state_dim, trunk_dim, max_span, act_dim,
                     hidden=256, layers=2, emb=32):
            super().__init__()
            self.max_span, self.act_dim = int(max_span), int(act_dim)
            self.slot = nn.Embedding(int(n_slots), int(emb))
            nn.init.normal_(self.slot.weight, 0.0, 0.02)
            din = int(emb) + int(state_dim) + int(trunk_dim)
            lyr = [nn.Linear(din, hidden), nn.SiLU()]
            for _ in range(int(layers) - 1):
                lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
            self.trunk = nn.Sequential(*lyr)
            self.out = nn.Linear(hidden, int(max_span) * int(act_dim))
            self.act = nn.Tanh()

        def forward(self, z, trunk_read, slot_id):
            h = self.trunk(torch.cat([self.slot(slot_id), z, trunk_read], dim=-1))
            y = self.act(self.out(h))
            return y.view(-1, self.max_span, self.act_dim)

    return SpanHead


def build_span(n_slots, state_dim, trunk_dim, max_span, act_dim, seed, device,
               hidden=256, layers=2, emb=32):
    """Mint WITHOUT touching the shared torch stream (gate G-R): save the global RNG state,
    construct, restore, then re-initialise every parameter from a dedicated generator."""
    import torch
    st = torch.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    head = _build_span_head()(n_slots, state_dim, trunk_dim, max_span, act_dim,
                              hidden=hidden, layers=layers, emb=emb)
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


def _split_code(rows):
    """Deterministic train/held-out split keyed by a BIJECTIVE-ish code of the seam state, not by a
    coin flip: the metering geometries are fixed per run, so near-identical seam states recur across
    cycles and a coin flip would put copies of one context on both sides -- the parity gate would then
    be reading its own training data (`native/span`'s measured trap)."""
    q = np.round(np.asarray(rows, np.float64) * 1e4).astype(np.int64)
    code = np.zeros(len(q), np.int64)
    for j in range(q.shape[1]):
        code = (code * 1000003 + (q[:, j] % 1000003)) % (2 ** 61 - 1)
    return (code * 48271) % 100


class SpanBuffer:
    """Per-slot self-imitation store: (seam state, the commands the audition selected for that slot
    at that state). The target is recomputed by the AUDITION at capture time, so the head chases the
    executor it is replacing rather than a stale copy of it.
    """

    def __init__(self, cap, hold_cap, hold_frac, seed):
        self.cap, self.hold_cap = int(cap), int(hold_cap)
        self.hold_frac = float(hold_frac)
        self.buf, self.hold = {}, {}
        self.rng = np.random.default_rng(int(seed))

    def store(self, sid, S, C):
        if len(S) == 0:
            return
        S = np.asarray(S, np.float32); C = np.asarray(C, np.float32)
        is_hold = _split_code(S) < int(round(self.hold_frac * 100))
        for tgt, cap, sel in ((self.hold, self.hold_cap, is_hold),
                              (self.buf, self.cap, ~is_hold)):
            if not sel.any():
                continue
            s2, c2 = S[sel], C[sel]
            if sid in tgt:
                s2 = np.concatenate([tgt[sid][0], s2])[-cap:]
                c2 = np.concatenate([tgt[sid][1], c2])[-cap:]
            tgt[sid] = (s2, c2)

    def sizes(self):
        return ({int(k): int(len(v[0])) for k, v in self.buf.items()},
                {int(k): int(len(v[0])) for k, v in self.hold.items()})
