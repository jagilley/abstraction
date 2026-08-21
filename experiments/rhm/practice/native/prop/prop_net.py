"""PORT 1 — ROUTING. The earned vocabulary becomes the PLANNER's own unit: a proposal head
pi(move | z, r*) beside the value, trained by self-imitation on the beam's own chosen
trajectories, that decides which moves get expanded at all.

WHAT THE ARC HAD. `ratchet/macros.py`'s header states it: a committed macro is a
*controller-side action* — a new primitive in an EXOGENOUS beam that expands EVERY move at
EVERY tip (`beam_moves`: `width x n_moves` children, all materialised, all value-scored).
Nothing proposes. A vocabulary that is never proposed is a vocabulary the agent does not own:
the chunk exists in the action set but the primitive level is still enumerated on every step,
which is the opposite of what "the chunk becomes the policy's unit" means.

WHAT THIS MODULE ADDS.

  pi(. | z, r*)   the SAME encoder read the value consumes (`controller.state(x)` + the root
                  target r*), a separate head, logits over a FIXED slot vocabulary covering
                  every (level, node) of the tree. Slots, not positions in `ms`, so the head
                  survives the action set growing at a commit.
  select_moves    at plan time, expand only the top-k proposed moves per tip. At k = n_moves
                  this is an EXACT no-op (the fidelity gate): the selection is a stable
                  descending sort truncated at k and then re-sorted ascending, so k = n_moves
                  returns `arange(n_moves)` bit-for-bit whatever the logits are, ties included.
  expand_selected the ragged expansion: move j is materialised only on the tips that selected
                  it. At k = n_moves every tip selects every move, the row gather is the
                  identity, and the assembled child tensor is bit-identical to
                  `torch.stack([apply_any(flat, ms[j]) for j], dim=1)` — which is what makes
                  the fidelity gate testable rather than asserted.

DESIGN DECISIONS (choices, not derivations — stated so they can be disagreed with):

 * ZERO-INIT OUTPUT LAYER. At init every slot's logit is exactly 0, so a freshly committed
   macro enters the ranking NEUTRALLY: above whatever the head has learned to reject, below
   whatever it has learned to accept. Untried actions are therefore optimistic-by-construction
   rather than shut out, and the head costs no RNG at the instant it is built.
 * ITS OWN RNG. The head is built inside `isolated_rng`, which saves/restores the global torch
   (and CUDA) generator state, and its training batches come from a dedicated numpy stream. A
   treated arm therefore draws the IDENTICAL shared sequence as its untreated twin and can only
   diverge through the port itself (`handle/`'s per-arm-stream discipline, one level up).
 * SLOTS ARE (level, node). `build_ms` appends macros after the base moves, so positional
   indices happen to be stable for the arms run here — but only by accident of commit order.
   Keying by (level, node) makes it true by construction.
 * MASKED CROSS-ENTROPY OVER THE LIVE ACTION SET. Slots not in the current `ms` are masked out
   of the softmax, so a slot that does not exist yet is never a target and never a competitor.
   Buffered pairs collected BEFORE a commit are replayed under the CURRENT mask: the mask only
   ever adds options, the stored target stays legal, and this keeps one buffer rather than one
   per action set.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# the slot vocabulary: one slot per (level, node) of the tree
# --------------------------------------------------------------------------- #

def slot_layout(depth, s, max_level):
    """{level: offset}, n_slots. L=4,s=2,max_level=3 -> {1:0, 2:8, 3:12}, 14 slots."""
    off, cur = {}, 0
    for ell in range(1, max_level + 1):
        off[ell] = cur
        cur += s ** (depth - ell)
    return off, cur


def move_slots(ms, offsets):
    """The slot id of every move in the live action set, in `ms` order."""
    return [offsets[int(mv["level"])] + int(mv["node"]) for mv in ms]


def decomposition_slots(level, node, s, offsets, max_level):
    """The slots of the moves a (level, node) macro would decompose into, one level down and
    all the way down to primitives. An expert cannot decompose its chunks; a head that still
    puts mass here has not chunked."""
    out = {}
    span = 1
    for lo in range(level - 1, 0, -1):
        span = s ** (level - lo)
        out[lo] = [offsets[lo] + node * span + j for j in range(span)]
    return out


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #

def _build_prop_head():
    import torch
    import torch.nn as nn

    class ProposalHead(nn.Module):
        """pi(move | z, r*) -> logits over the fixed slot vocabulary. Same shape of read as
        `MCValueHead` (`rhm_latent_planner._build_value_head`): the controller state and an
        embedding of the root target, concatenated. Output layer zero-init."""

        def __init__(self, state_dim, vocab_size, n_slots):
            super().__init__()
            self.n_slots = int(n_slots)
            self.root_embedding = nn.Embedding(vocab_size, state_dim)
            self.trunk = nn.Sequential(
                nn.LayerNorm(2 * state_dim),
                nn.Linear(2 * state_dim, 4 * state_dim),
                nn.GELU(),
            )
            self.out = nn.Linear(4 * state_dim, n_slots)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, z, root):
            return self.out(self.trunk(torch.cat([z, self.root_embedding(root)], dim=-1)))

    return ProposalHead


class isolated_rng:
    """Run a block with the global torch RNG saved and restored, seeded from a dedicated
    stream. Any NEW consumer of randomness (the proposal head's init) therefore leaves the
    shared per-arm stream exactly where it found it, which is what licenses the twin gate."""

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


def build_head(state_dim, v, n_slots, seed, device):
    import torch
    with isolated_rng(seed, device):
        head = _build_prop_head()(state_dim, v, n_slots).to(device)
    return head


# --------------------------------------------------------------------------- #
# selection: top-k with a deterministic, no-op-at-k=n tie order
# --------------------------------------------------------------------------- #

def select_moves(plogits, k, forced=None, explore=None):
    """plogits: (N, n_moves) — logits RESTRICTED to the live action set, in `ms` order.
    Returns (N, k_eff) move indices, ascending within each row.

    `torch.topk`'s tie order is not specified; a stable descending sort's is (index ascending),
    so at k = n_moves the truncation keeps every index and the ascending re-sort returns
    `arange(n_moves)` exactly, whatever the logits are. That is the fidelity gate's precondition
    and it holds by construction rather than by luck.

    `forced` (a list of move indices) is always expanded IN ADDITION to the top-k, so
    k_eff = k + len(forced): a move the head has never had a training example for gets its
    first exposure instead of being ranked out by an untrained logit. Priced at its true cost —
    the effective branching factor those cycles is k + len(forced).

    `explore` (an (N, e) index tensor from `explore_moves`) is likewise expanded in addition,
    and is used ONLY in the practice beam."""
    import torch
    n = plogits.shape[1]
    kk = min(int(k), n)
    work = plogits
    if forced:
        work = plogits.clone()
        work[:, list(forced)] = -float("inf")
        kk = min(kk, n - len(forced))
    order = torch.sort(work, dim=1, descending=True, stable=True).indices
    sel = order[:, :kk]
    if forced:
        f = torch.as_tensor(sorted(forced), dtype=sel.dtype, device=sel.device)
        sel = torch.cat([sel, f[None, :].expand(sel.shape[0], -1)], dim=1)
    if explore is not None and explore.shape[1]:
        sel = torch.cat([sel, explore.to(sel.dtype)], dim=1)
    return torch.sort(sel, dim=1).values


def explore_moves(sel_base, n_moves, n_explore, rng, device):
    """`n_explore` moves per tip drawn uniformly from the ones the head did NOT propose.

    PRACTICE EXPLORES, PERFORMANCE DOES NOT — the substrate's own convention
    (`collect_value_buffer` already runs its behaviour policy at `explore_eps` = 0.3). Without
    this a filtered beam is a closed loop: a move the head does not propose never appears in a
    solved trajectory, so it never becomes a training target, so the head never learns it
    (measured in `smoke0`, where a k = 2 arm locked the level-3 macro out of its top five while
    its k = n_moves twin ranked it first). Drawn from a dedicated numpy stream, so the shared
    per-arm torch stream is undisturbed; excludes the already-selected moves so the extra
    expansion is never a wasted duplicate."""
    import torch
    n = sel_base.shape[0]
    r = torch.from_numpy(rng.random((n, int(n_moves)))).to(device).float()
    r.scatter_(1, sel_base, -1.0)
    return r.topk(min(int(n_explore), int(n_moves) - sel_base.shape[1]), dim=1).indices


def expand_selected(generator, flat, sel, ms, apply_any, rules_t, canon, depth, v, m, s):
    """(N, T) tips x (N, k) selected move indices -> (N, k, T) children.

    Move j is materialised only on the rows that selected it, so the materialisation count is
    N*k rather than N*n_moves. At k = n_moves every row selects every move in ascending order,
    so `rows` is `arange(N)` for every j, `flat[rows]` is `flat`, and the assembled tensor is
    the enumerated one bit-for-bit."""
    import torch
    n, kk = sel.shape
    out = torch.zeros(n, kk, flat.shape[1], dtype=flat.dtype, device=flat.device)
    n_mat = 0
    for j in range(len(ms)):
        hit = sel == j
        if not bool(hit.any()):
            continue
        rows, cols = hit.nonzero(as_tuple=True)
        child = apply_any(generator, flat[rows], ms[j], rules_t, canon, depth, v, m, s)
        out[rows, cols] = child
        n_mat += int(rows.numel())
    return out, n_mat


# --------------------------------------------------------------------------- #
# the can't-decompose signature
# --------------------------------------------------------------------------- #

def decompose_probe(prop, z, roots, ms, offsets, s, max_level, avail_mask):
    """On a fixed set of states: where does pi put its mass, and when it proposes a MACRO, how
    much mass does it keep on that macro's own primitive decomposition?

    An expert cannot decompose its chunks. A head that proposes L2n3 while still holding
    comparable mass on L1n6 and L1n7 has an action set with a macro in it, not a chunk."""
    import torch
    with torch.no_grad():
        logits = prop(z, roots)
        logits = logits.masked_fill(~avail_mask[None, :], -float("inf"))
        p = torch.softmax(logits, dim=-1)
    n_slots = p.shape[1]
    arg = p.argmax(dim=1)
    out = {"n": int(p.shape[0]),
           "mean_mass": [float(x) for x in p.mean(0).cpu()],
           "argmax_hist": [int(x) for x in torch.bincount(arg, minlength=n_slots).cpu()]}
    macro_slots = {}
    for mv in ms:
        ell, node = int(mv["level"]), int(mv["node"])
        if ell >= 2:
            macro_slots[offsets[ell] + node] = (ell, node)
    out["macro_mass"] = float(p[:, sorted(macro_slots)].sum(1).mean()) if macro_slots else 0.0
    rows = []
    for slot, (ell, node) in sorted(macro_slots.items()):
        hit = arg == slot
        nh = int(hit.sum())
        dec = decomposition_slots(ell, node, s, offsets, max_level)
        cell = {"slot": slot, "level": ell, "node": node, "n_argmax": nh,
                "frac_argmax": nh / p.shape[0],
                "p_macro_all": float(p[:, slot].mean())}
        for lo, slots in dec.items():
            keep = [j for j in slots if bool(avail_mask[j])]
            cell[f"p_dec{lo}_all"] = float(p[:, keep].sum(1).mean()) if keep else 0.0
        if nh:
            sub = p[hit]
            cell["p_macro"] = float(sub[:, slot].mean())
            for lo, slots in dec.items():
                keep = [j for j in slots if bool(avail_mask[j])]
                cell[f"p_dec{lo}"] = float(sub[:, keep].sum(1).mean()) if keep else 0.0
                cell[f"ratio{lo}"] = (cell[f"p_dec{lo}"] / max(cell["p_macro"], 1e-9))
        rows.append(cell)
    out["macros"] = rows
    return out


def base_mass_probe(prop, z, roots, offsets, avail_mask):
    """The control number the `base_prop_k` arm supplies: with no macros available at all, how
    much mass does the head put on the very same primitive moves? `p_dec` in the macro arms is
    only interpretable against this."""
    import torch
    with torch.no_grad():
        logits = prop(z, roots).masked_fill(~avail_mask[None, :], -float("inf"))
        p = torch.softmax(logits, dim=-1)
    return {"n": int(p.shape[0]), "mean_mass": [float(x) for x in p.mean(0).cpu()],
            "top1": float(p.max(1).values.mean()),
            "entropy": float((-(p.clamp_min(1e-12).log() * p).sum(1)).mean())}


# --------------------------------------------------------------------------- #
# grounding accounting under a proposal filter
# --------------------------------------------------------------------------- #

def beam_ground_k(k, budget, w):
    """`ratchet.beam_ground` with n_moves -> the EFFECTIVE branching factor k."""
    total, width = 0, 1
    for _ in range(budget):
        total += width * k
        width = min(w, width * k)
    return total + width


def fit_width_k(k, budget, g_budget, root_cost=1):
    """The widest beam that fits the declared budget under branching k. `root_cost` is the ONE
    extra encoder pass per instance the port genuinely needs: the root state has never been
    scored as anybody's child, so its z is not already paid for. Every deeper tip's z was
    computed and charged when the value scored it as a child, so the proposal read there is a
    head-only forward on a cached vector and costs no grounding."""
    best = 1
    for w in range(1, 129):
        if beam_ground_k(k, budget, w) + root_cost <= g_budget:
            best = w
        else:
            break
    # a beam cannot be wider than branching lets it become, and reporting a width the beam can
    # never reach would make the logged ladder unreadable.
    return max(1, min(best, int(k) ** int(budget)))
