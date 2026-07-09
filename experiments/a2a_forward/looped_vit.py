"""Looped (weight-shared, recurrent-depth) Vision Transformer for MNIST.

Companion to vit.py. Where ViT applies n_layer *distinct* blocks once, LoopedViT
applies a single shared operator G (a stack of n_loop_layers blocks) T times with
weight sharing. This is the architecture the self-model-needs-a-loop hypothesis
requires: one reusable operator that is forced to be both the *producer* of the
state a forward model reads and the *consumer* of the forward model's forecast of
that same state (ideas/self_model_needs_a_loop.md).

Recurrence (Universal-Transformer / DEQ-style input injection):

    e        = patch_embed(x) + pos_embed + cls        # fixed input embedding
    s_0      = 0
    s_{t+1}  = G(s_t + e + inject_t)                    # G = shared block stack
    logits   = head(ln_f(s_T)[:, 0])

Re-injecting e each step makes the fixed point input-dependent (h* = g(h*) has a
well-defined, input-conditioned solution) and keeps G a single autonomous operator
applied identically at every step (no per-step timestep embedding, which would make
g_t != g and break the "one operator to model" premise).

The step_inject_fn hook adds an arbitrary additive term to the state each step. In
phase 1 it is None (plain loop). In phase 2 it carries the gated forward-model
forecast: step_inject_fn = lambda t, s: gate(fm(s.detach())).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2a_forward.vit import ViTBlock


class LoopedViT(nn.Module):
    """Weight-shared recurrent-depth ViT with the same I/O contract as ViT.

    Optional non-shared prelude/coda blocks bookend the shared recurrent core
    (the standard recurrent-depth architecture: prelude -> core xT -> coda). With
    them, the FM setup becomes a near-exact analog of the feedforward a2a work: the
    FM reads a mid-stack recurrent state s_t (like post_block0), predicts the next
    state (like the post_block target), the injection lands in the recurrent stream
    (like inject-after-block1), and the coda+head separate readout from the
    recurrent state (like block3 -> head). prelude_layers=coda_layers=0 recovers the
    pure loop (phase 1).

    Args:
        n_loop_layers: number of blocks in the shared operator G (default 1).
        prelude_layers / coda_layers: non-shared feedforward blocks before/after the
            loop (the "looparound" bookends). 0 = pure loop.
        n_steps: default recurrence depth T (can be overridden per forward call).
        inject_input_each_step: DEQ-style re-injection of the (post-prelude) input at
            every step. If False, it is only the initial state and the loop is
            autonomous.
    """

    def __init__(self, img_size=28, patch_size=4, in_channels=1, n_classes=10,
                 n_loop_layers=1, n_head=4, n_embd=128, n_steps=8,
                 prelude_layers=0, coda_layers=0,
                 inject_input_each_step=True):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.n_positions = self.n_patches + 1
        self.n_steps = n_steps
        self.n_loop_layers = n_loop_layers
        self.inject_input_each_step = inject_input_each_step
        self.n_embd = n_embd

        self.patch_proj = nn.Linear(patch_size * patch_size * in_channels, n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Embedding(self.n_positions, n_embd)

        # Non-shared feedforward bookends (the "looparound").
        self.prelude = nn.ModuleList([
            ViTBlock(n_embd, n_head) for _ in range(prelude_layers)
        ])
        self.coda = nn.ModuleList([
            ViTBlock(n_embd, n_head) for _ in range(coda_layers)
        ])

        # The shared operator G (applied T times).
        self.blocks = nn.ModuleList([
            ViTBlock(n_embd, n_head) for _ in range(n_loop_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_classes)

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"LoopedViT: {n_params / 1e6:.2f}M parameters "
              f"(prelude={prelude_layers} + {n_loop_layers} block(s) x T={n_steps} "
              f"loop + coda={coda_layers}, {n_head}H {n_embd}D, "
              f"patch={patch_size}, {self.n_patches} patches, "
              f"input_inject={inject_input_each_step})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_patches, -1)
        return self.patch_proj(x)

    def _embed(self, x):
        B = x.shape[0]
        patches = self._patchify(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, patches], dim=1)
        pos = torch.arange(self.n_positions, device=x.device)
        return h + self.pos_embed(pos)

    def _apply_operator(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def _prelude(self, x):
        for block in self.prelude:
            x = block(x)
        return x

    def _readout(self, s):
        """Shared step-agnostic readout: coda -> ln_f -> [CLS] -> head.

        Applying the SAME readout to any iterate is what lets deep supervision
        turn the loop into an attractor (every held state decodes to the answer)."""
        h = s
        for block in self.coda:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h[:, 0])

    def forward(self, x, targets=None, n_steps=None, return_intermediates=False,
                return_trajectory=False, step_inject_fn=None, perturbation=None,
                readout_all_steps=False):
        """Run the loop.

        Args:
            n_steps: override recurrence depth for this call (else self.n_steps).
            return_intermediates: return dict with post_embed, post_prelude,
                post_step{t}.
            return_trajectory: also return per-step state-change norms
                (delta[t] = ||s_{t+1} - s_t||, rel[t] = delta / ||s_t||),
                for measuring convergence toward a fixed point.
            step_inject_fn(t, operand, state) -> additive tensor injected each step
                (the gated FM forecast in phase 2). operand = s_t + input (what G
                transforms); state = s_t (the raw carried state, so the caller can
                inject a forecast of the *update* FM(operand) - state, which vanishes
                at the fixed point).
            perturbation: (step_index, tensor) added to the state before that step.
            readout_all_steps: also return per-step logits (T, B, n_classes), the
                shared readout applied to every iterate (for deep supervision).
        """
        T = n_steps if n_steps is not None else self.n_steps
        e = self._embed(x)
        p = self._prelude(e)  # post-prelude input, re-injected each step

        intermediates = {}
        if return_intermediates:
            intermediates["post_embed"] = e
            intermediates["post_prelude"] = p

        s = torch.zeros_like(p)
        deltas, rels = [], []
        step_logits = []

        for t in range(T):
            s_in = s + p if self.inject_input_each_step else (p if t == 0 else s)
            if step_inject_fn is not None:
                inj = step_inject_fn(t, s_in, s)
                if inj is not None:
                    s_in = s_in + inj
            if perturbation is not None and t == perturbation[0]:
                s_in = s_in + perturbation[1]

            s_new = self._apply_operator(s_in)

            if return_trajectory:
                d = (s_new - s).norm(dim=-1).mean().item()
                deltas.append(d)
                denom = s.norm(dim=-1).mean().item()
                rels.append(d / denom if denom > 0 else float("nan"))

            s = s_new
            if return_intermediates:
                intermediates[f"post_step{t}"] = s
            if readout_all_steps:
                step_logits.append(self._readout(s))

        logits = step_logits[-1] if readout_all_steps else self._readout(s)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        out = (logits, loss)
        if return_intermediates:
            out = out + (intermediates,)
        if return_trajectory:
            out = out + ({"delta": deltas, "rel_delta": rels},)
        if readout_all_steps:
            out = out + (torch.stack(step_logits, dim=0),)  # (T, B, n_classes)
        return out
