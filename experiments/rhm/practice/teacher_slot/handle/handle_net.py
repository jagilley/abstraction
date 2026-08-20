"""THE EMPTY CELL: earned vocabulary living IN WEIGHTS, as a backprop handle.

Every earned-vocabulary node in the practice arc kept the vocabulary OUTSIDE the network.
`ratchet/macros.py`'s header states the design plainly: a committed macro is a *controller-side
action* — a new primitive in the beam's action space — and the generator (the plant) neither
predicts it nor conditions on it. Every one of those nodes then found the plant inert, which is
exactly what you would expect if the earned tokens had no gradient path into the weights.

This module opens that cell. A committed macro entry gets a SLOT in the generator's own
vocabulary:

  PREDICTS    — a per-span head over {0 = no committed entry covers this span, 1..K = entry j}.
                It is supervised on exactly the spans the generator is being asked to infill, so
                the earned symbol is a training target of the plant's own objective.
  CONDITIONS  — a per-span embedding added to the token+position embedding of every position in
                a FULLY VISIBLE span. Masked spans get symbol 0, so nothing is given away about
                the span being repaired; the earned symbols summarise the surrounding context.

Both share the trunk, so the new rows are a focal point through which gradients reach the
surrounding representation — `ideas/language_reduction_continual_learning.md`'s "token as
backprop handle", instantiated on sculpting.

DESIGN DECISIONS (stated because they are choices, not derivations):

 * ANNOTATE, DON'T SUBSTITUTE. The macro symbol is *added* to the span's embeddings; the leaf
   tokens stay in the stream. Substituting them would change `block_logits`' shape contract and
   break every downstream reader (the macro DP, the base moves, the mining parse), which is
   deeper surgery than this cheap probe licenses.
 * ZERO INIT, so the handle costs no RNG and is an exact no-op at the instant of commitment.
   The symbol's influence is entirely earned by gradient descent — which is the claim under
   test. A handle arm is therefore BIT-IDENTICAL to its no-handle twin until its commit cycle.
 * TWO GRANULARITIES. `mode="entry"`: one symbol per committed entry (K+1 symbols).
   `mode="level"`: one symbol per level (2 symbols — in-vocabulary / out-of-vocabulary), the
   spec's "per-level macro symbol".
 * COMMITTED ONLY. Candidate (mined-but-uncommitted) tables never get slots; the handle is a
   property of the committed vocabulary, which is what "earned vocabulary in weights" means.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# the slot: one embedding row + one head row per earned symbol
# --------------------------------------------------------------------------- #

def _build_slot():
    import torch.nn as nn

    class Slot(nn.Module):
        def __init__(self, n_sym, dim):
            super().__init__()
            self.emb = nn.Embedding(n_sym, dim)
            self.head = nn.Linear(dim, n_sym)
            nn.init.zeros_(self.emb.weight)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    return Slot


def _build_handle_generator():
    import torch
    import torch.nn as nn

    Slot = _build_slot()

    class HandleGenerator(nn.Module):
        """Wraps a `BlockInfiller` (`rhm_generative_planner._build_generator`) without touching
        it. `block_logits(obs)` keeps the exact same signature and shape contract, so every
        caller in the ratchet substrate — `macros.macro_features`, `crystallize.units.apply_move`,
        `plant_probe`, `macros.parse_features` — works unchanged."""

        def __init__(self, core, *, v, s, n_blocks, bottom_map, mode="entry"):
            super().__init__()
            self.core = core                      # registered FIRST: parameter order is the
            self.slots = nn.ModuleDict()          # core's, so an empty wrapper gives an
            self.mode = mode                      # optimizer identical to the unwrapped one
            self.v, self.s, self.n_blocks = int(v), int(s), int(n_blocks)
            self.register_buffer("bmap", bottom_map.detach().clone().long(), persistent=False)
            self.register_buffer("powers", (int(v) ** torch.arange(int(s))).long(),
                                 persistent=False)
            self.register_buffer("vpow", (int(v) ** torch.arange(8)).long(), persistent=False)
            self.luts = {}                        # str(level) -> dict(lut, span, n_sym)

        # -- delegation so the wrapper is a drop-in ------------------------------------- #
        @property
        def mask_token(self):
            return self.core.mask_token

        @property
        def block_size(self):
            return self.core.block_size

        @property
        def root_conditioned(self):
            return self.core.root_conditioned

        # -- the earned symbols ---------------------------------------------------------- #
        def add_slot(self, level, table, mode=None):
            """Mint symbols for a freshly committed table. Called once per level, at the
            commit cycle. Uses no RNG (zero init) and touches no existing parameter."""
            dev = self.bmap.device
            flat = np.asarray(table["flat"], np.int64)
            K, span = flat.shape
            mode = mode or self.mode
            n_sym = 2 if mode == "level" else K + 1
            codes = (flat * (self.v ** np.arange(span, dtype=np.int64))).sum(1)
            lut = np.zeros(int(self.v) ** span, np.int64)
            for j, c in enumerate(codes):
                lut[int(c)] = 1 if mode == "level" else j + 1
            key = str(int(level))
            self.luts[key] = {"lut": torch.from_numpy(lut).to(dev), "span": int(span),
                              "n_sym": int(n_sym)}
            self.slots[key] = Slot(n_sym, self.core.token_embedding.embedding_dim).to(dev)
            return {"level": int(level), "n_entries": int(K), "n_sym": int(n_sym),
                    "span": int(span), "n_flat_unique": int(len(set(codes.tolist()))),
                    "n_params": int(sum(p.numel() for p in self.slots[key].parameters()))}

        def slot_params(self, level):
            return list(self.slots[str(int(level))].parameters())

        def levels(self):
            return sorted(self.slots.keys(), key=int)

        # -- symbol lookup ---------------------------------------------------------------- #
        def _ids_from_feats(self, feats, key):
            """feats: (B, n_blocks) level-1 feature ids, -1 = unknown/off-grammar."""
            info = self.luts[key]
            span = info["span"]
            b = feats.shape[0]
            f = feats.view(b, self.n_blocks // span, span)
            ok = (f >= 0).all(-1)
            code = (f.clamp(min=0) * self.vpow[:span]).sum(-1)
            return torch.where(ok, info["lut"][code], torch.zeros_like(code))

        def ids_from_obs(self, obs, key):
            """The CONDITIONING read: symbols of fully visible spans, 0 elsewhere. A span with
            any masked block is symbol 0, so the span under repair never leaks its own id."""
            b = obs.shape[0]
            blk = obs.view(b, self.n_blocks, self.core.block_size)
            valid = (blk >= 0).all(-1)
            code = (blk.clamp(min=0) * self.powers).sum(-1)
            feats = torch.where(valid, self.bmap[code], torch.full_like(code, -1))
            return self._ids_from_feats(feats, key)

        def targets_from_feats(self, feats, key):
            """The PREDICTION target: the true symbol of every span, from clean features."""
            return self._ids_from_feats(feats, key)

        # -- the forward ------------------------------------------------------------------ #
        def hidden(self, observation, root=None, use_slots=True):
            core = self.core
            tokens = observation.masked_fill(observation < 0, core.mask_token)
            h = core.token_embedding(tokens) + core.position_embedding(core.positions)
            if core.root_conditioned and root is not None:
                h = h + core.root_embedding(root)[:, None, :]
            if use_slots:
                for key in self.levels():
                    ids = self.ids_from_obs(observation, key)          # (B, n_spans)
                    e = self.slots[key].emb(ids)                       # (B, n_spans, D)
                    h = h + e.repeat_interleave(observation.shape[1] // e.shape[1], dim=1)
            return core.final_norm(core.encoder(h))

        def block_logits(self, observation, root=None, use_slots=True):
            """Identical contract to `BlockInfiller.block_logits`: (B, n_blocks, v)."""
            h = self.hidden(observation, root, use_slots=use_slots)
            b, length, dim = h.shape
            pooled = h.view(b, self.n_blocks, self.core.block_size, dim).mean(dim=2)
            return self.core.feature_head(pooled)

        def macro_logits(self, key, hidden=None, observation=None, root=None):
            """(B, n_spans, n_sym) — the earned symbol the generator predicts for each span."""
            h = self.hidden(observation, root) if hidden is None else hidden
            span_tok = self.luts[key]["span"] * self.core.block_size
            b, length, dim = h.shape
            pooled = h.view(b, length // span_tok, span_tok, dim).mean(dim=2)
            return self.slots[key].head(pooled)

    return HandleGenerator


def wrap(core, *, v, s, n_blocks, bottom_map, mode="entry"):
    return _build_handle_generator()(core, v=v, s=s, n_blocks=n_blocks,
                                     bottom_map=bottom_map, mode=mode)
