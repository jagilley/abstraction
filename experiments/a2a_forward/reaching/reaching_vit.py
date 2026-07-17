"""Looped ViT as a CONTROLLER: foveal reaching -- the control-regime port of the
"missing u" test (ideas/self_model_needs_a_loop.md; ACTIVE_VISION_README.md).

The active-vision arm gave the loop a genuine command `u` (where to look) and showed
observationally that only an arity-2 forward model f(s, u) can predict the loop's
command-conditional dynamics -- but the CAUSAL-injection arm came back null, because
the command is CONTENT-FREE (it says *where* you look, never *what is there*), so on
a *perception* task (classify the digit) the command steers the trajectory but not
the answer. The forecast is answer-irrelevant.

This module ports the same apparatus to a CONTROL task where the command's
consequences ARE the objective. The agent steers its fovea across the patch grid to
a GOAL patch. Now:
  - the command u_t is a DISPLACEMENT ACTION (stay/up/down/left/right),
  - the fovea position p_t is an ENDOGENOUS controlled state (the action moves it),
  - the objective (reach the goal) is defined over that self-controlled state.
So a forward model conditioned on the action predicts an answer-RELEVANT
consequence (where the fovea ends up), and -- crucially -- an arity-1 f(s) that
cannot condition on the action is STRUCTURALLY unable to support planning at any
capacity (no `u` slot to query counterfactually). This turns "arity beats
resolution" from a fit margin into a behavioral impossibility.

Architecture (Markov reaching; the loop integrates a proprioceptive fovea marker and
a goal marker each step, plus optional glimpse content as a distractor):

    s_0     = 0
    s_{t+1} = G( s_t + obs(p_t, g) [+ inject_t] )         # G = shared ViT operator
    a_t     = policy_head(s_{t+1})   OR   plan(s_{t+1}, FM)   # the two controllers
    p_{t+1} = clip( p_t + delta(a_t) )                    # env dynamics (outside model)

`obs(p_t, g)` = cls + pos_embed + fovea_marker@p_t + goal_marker@g (+ glimpse@p_t).
The model exposes single transitions (`step`), a policy head (the model-free
controller / ceiling), and everything a forward-model planner needs; the dynamics
(delta, clip) and the forward model live in the experiment script (mnist_reaching.py),
mirroring how the glimpse experiment kept the efference marker out of the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2a_forward.vit import ViTBlock

# Displacement action set, in (row, col) grid deltas. Index order is load-bearing:
# the forward model's action embedding and the experiment's oracle use these indices.
ACTION_DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay,up,down,left,right
ACTION_NAMES = ["stay", "up", "down", "left", "right"]


class ReachingLoopedViT(nn.Module):
    """Looped ViT controller for foveal reaching. Same shared-operator recurrence as
    LoopedViT/GlimpseLoopedViT, but driven by a control objective instead of
    classification. `with_content=False` is pure navigation (blank canvas, the
    cleanest arity test); `with_content=True` reveals a GxG glimpse of the image at
    the fovea each step as an answer-irrelevant distractor (real activation dynamics
    the forward model must also forecast -- the true minimal diff from the null
    active-vision experiment)."""

    def __init__(self, img_size=28, patch_size=4, in_channels=1, n_embd=128,
                 n_head=4, n_loop_layers=1, n_steps=14, glimpse_grid=3,
                 with_content=False, n_actions=5):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid ** 2
        self.n_positions = self.n_patches + 1
        self.n_embd = n_embd
        self.n_steps = n_steps
        self.with_content = with_content
        self.glimpse_grid = glimpse_grid
        self.n_actions = n_actions

        self.patch_proj = nn.Linear(patch_size * patch_size * in_channels, n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Embedding(self.n_positions, n_embd)
        # Learned proprioceptive / goal markers placed spatially at p_t and g.
        self.fovea_marker = nn.Parameter(torch.zeros(n_embd))
        self.goal_marker = nn.Parameter(torch.zeros(n_embd))

        self.blocks = nn.ModuleList([
            ViTBlock(n_embd, n_head) for _ in range(n_loop_layers)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.policy_head = nn.Linear(n_embd, n_actions)

        # GxG glimpse window reveal mask (only used when with_content).
        h = glimpse_grid // 2
        M = torch.zeros(self.n_patches, self.n_patches)
        for c in range(self.n_patches):
            cr, cc = c // self.grid, c % self.grid
            for dr in range(-h, h + 1):
                for dc in range(-h, h + 1):
                    r, k = cr + dr, cc + dc
                    if 0 <= r < self.grid and 0 <= k < self.grid:
                        M[c, r * self.grid + k] = 1.0
        self.register_buffer("reveal_mask", M)

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.fovea_marker, std=0.02)
        nn.init.normal_(self.goal_marker, std=0.02)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"ReachingLoopedViT: {n_params / 1e6:.2f}M params "
              f"({n_loop_layers} block(s) x T={n_steps}, {n_head}H {n_embd}D, "
              f"grid={self.grid}x{self.grid}, {n_actions} actions, "
              f"with_content={with_content})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _patch_embeds(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_patches, -1)
        return self.patch_proj(x)  # (B, n_patches, n_embd)

    def _marker_at(self, idx, vec):
        """(B, n_positions, n_embd): `vec` placed at position idx+1 (cls is 0),
        zeros elsewhere. Differentiable one-hot scatter (no in-place)."""
        onehot = F.one_hot(idx + 1, self.n_positions).to(vec.dtype)  # (B, n_pos)
        return onehot.unsqueeze(-1) * vec  # (B, n_pos, n_embd)

    def obs(self, p_idx, g_idx, patch_emb=None):
        """Per-step observation tokens: cls + pos_embed + fovea_marker@p + goal@g
        (+ glimpse content @p if with_content)."""
        B = p_idx.shape[0]
        device = p_idx.device
        pos = torch.arange(self.n_positions, device=device)
        tok = self.cls_token.expand(B, -1, -1) + self.pos_embed(pos)
        tok = tok + self._marker_at(p_idx, self.fovea_marker)
        tok = tok + self._marker_at(g_idx, self.goal_marker)
        if self.with_content and patch_emb is not None:
            mask = self.reveal_mask[p_idx]  # (B, n_patches)
            revealed = patch_emb * mask.unsqueeze(-1)
            cls0 = torch.zeros(B, 1, self.n_embd, device=device)
            tok = tok + torch.cat([cls0, revealed], dim=1)
        return tok

    def _apply_operator(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def init_state(self, B, device):
        return torch.zeros(B, self.n_positions, self.n_embd, device=device)

    def step(self, s, p_idx, g_idx, patch_emb=None, inject=None):
        """One recurrent transition: s' = G(s + obs(p, g) [+ inject]).
        Returns the post-observation state s_{t+1}."""
        s_in = s + self.obs(p_idx, g_idx, patch_emb)
        if inject is not None:
            s_in = s_in + inject
        return self._apply_operator(s_in)

    def policy_logits(self, s):
        """Model-free action logits from the [CLS] readout."""
        return self.policy_head(self.ln_f(s)[:, 0])
