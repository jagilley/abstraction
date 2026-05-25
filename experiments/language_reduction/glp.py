"""GLP (Generative Latent Prior) for the language reduction pipeline.

Trains a flow-matching denoiser on all-layer activations [h0; h1] from the
2-layer GPT-2 model. Supports:
  - Directional residual analysis (feature - manifold_projection)
  - Manifold expansion detection pre/post scaffolding
  - Concept token warm initialization

Architecture: SwiGLU MLP blocks with multiplicative timestep conditioning,
following the GLP paper's denoiser design. Adapted for 256-D feature space
(128-D per transformer layer × 2 layers).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SwiGLUBlock(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_mlp)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.time_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)

    def forward(self, x, t_emb):
        residual = x
        h = self.ln(x)
        up = self.up_proj(h)
        gate = self.gate_proj(h)
        t = self.time_proj(t_emb)
        h = F.silu(gate * t) * up
        return self.down_proj(h) + residual


class GLPDenoiser(nn.Module):
    def __init__(self, act_dim=256, d_model=256, d_mlp=512, n_layers=3):
        super().__init__()
        self.act_dim = act_dim
        self.d_model = d_model
        self.in_proj = nn.Linear(act_dim, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [SwiGLUBlock(d_model, d_mlp) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, act_dim)

    def forward(self, z, t):
        t_emb = self.time_embed(sinusoidal_embedding(t, self.d_model))
        h = self.in_proj(z)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(self.ln(h))


# ---- Activation extraction ----

def extract_lnf_h1_activations(model, tokens, block_size, batch_size=64,
                               max_tokens=None, device="cuda"):
    """Extract ln_f(h1) — the final block output after the final LayerNorm.

    Returns (N, n_embd) where n_embd matches the model's embedding dimension.
    This is the representation that gets dot-producted with lm_head rows to
    produce logits, so with tied weights it lives natively in embedding space.
    """
    model.eval()
    all_acts = []
    n_collected = 0
    n_sequences = len(tokens) // block_size

    for batch_start in range(0, n_sequences, batch_size):
        batch_end = min(batch_start + batch_size, n_sequences)
        x = torch.stack([
            tokens[i * block_size : (i + 1) * block_size]
            for i in range(batch_start, batch_end)
        ]).to(device)

        with torch.no_grad():
            tok_emb = model.transformer.wte(x)
            if hasattr(model.transformer, 'wpe'):
                pos = torch.arange(x.shape[1], device=device)
                tok_emb = tok_emb + model.transformer.wpe(pos)
            h = model.transformer.drop(tok_emb)
            for block in model.transformer.h:
                h = block(h)
            h = model.transformer.ln_f(h)

        all_acts.append(h.reshape(-1, h.shape[-1]).cpu())
        n_collected += all_acts[-1].shape[0]
        if max_tokens and n_collected >= max_tokens:
            break

    acts = torch.cat(all_acts, dim=0)
    return acts[:max_tokens] if max_tokens else acts


def extract_activations(model, tokens, block_size, batch_size=64,
                        max_tokens=None, device="cuda"):
    """Extract [h0; h1; ...] from all transformer Block outputs.

    For a 2-layer GPT with n_embd=128, returns (N, 256) — each token
    position yields a single row that concatenates both layers.
    """
    model.eval()
    all_acts = []
    n_collected = 0
    n_sequences = len(tokens) // block_size

    for batch_start in range(0, n_sequences, batch_size):
        batch_end = min(batch_start + batch_size, n_sequences)
        x = torch.stack([
            tokens[i * block_size : (i + 1) * block_size]
            for i in range(batch_start, batch_end)
        ]).to(device)

        layer_outputs = []
        hooks = []
        for block in model.transformer.h:
            hooks.append(block.register_forward_hook(
                lambda _mod, _inp, out, s=layer_outputs: s.append(out.detach())
            ))

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        h_cat = torch.cat(layer_outputs, dim=-1)       # (B, T, n_layer*n_embd)
        all_acts.append(h_cat.reshape(-1, h_cat.shape[-1]).cpu())
        layer_outputs.clear()

        n_collected += all_acts[-1].shape[0]
        if max_tokens and n_collected >= max_tokens:
            break

    acts = torch.cat(all_acts, dim=0)
    return acts[:max_tokens] if max_tokens else acts


# ---- Training ----

def train_glp(activations, act_dim=256, d_model=256, d_mlp=512, n_layers=3,
              n_steps=20_000, batch_size=512, lr=5e-5, device="cuda",
              seed=42, log_interval=2000):
    """Train a GLP denoiser via flow matching.

    Returns (state_dict, act_stats, loss_curve).
    act_stats = {"mean": (act_dim,), "std": (act_dim,)}.
    """
    act_mean = activations.mean(dim=0)
    act_std = activations.std(dim=0).clamp(min=1e-8)
    acts_norm = ((activations - act_mean) / act_std).to(device)

    torch.manual_seed(seed)
    model = GLPDenoiser(act_dim, d_model, d_mlp, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_data = acts_norm.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GLP denoiser: {n_params/1e6:.2f}M params, training on {n_data:,} activations")
    losses = []

    model.train()
    for step in range(n_steps):
        idx = torch.randint(n_data, (batch_size,), device=device)
        batch = acts_norm[idx]

        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(batch)
        z_t = (1.0 - t[:, None]) * batch + t[:, None] * noise
        target = noise - batch

        loss = F.mse_loss(model(z_t, t), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == n_steps - 1:
            losses.append((step, float(loss)))
            print(f"  step {step:6d}/{n_steps}: loss={loss:.6f}")

    return model.cpu().state_dict(), {"mean": act_mean, "std": act_std}, losses


# ---- Inference ----

@torch.no_grad()
def glp_denoise(model, acts_std, t_start=0.3, num_steps=10):
    """Euler integration from t_start → 0 on standardized activations."""
    device = next(model.parameters()).device
    noise = torch.randn_like(acts_std)
    z = (1.0 - t_start) * acts_std + t_start * noise
    dt = -t_start / num_steps
    for i in range(num_steps):
        t_val = t_start + i * dt
        t_batch = torch.full((z.shape[0],), t_val, device=device)
        z = z + dt * model(z, t_batch)
    return z


@torch.no_grad()
def compute_residuals(glp_model, act_stats, activations,
                      t_start=0.3, num_steps=10, batch_size=4096):
    """r = feature − manifold_projection (in original unnormalized space).

    Returns (residuals, manifold_projections), both shaped (N, act_dim).
    """
    device = next(glp_model.parameters()).device
    mean = act_stats["mean"].to(device)
    std = act_stats["std"].to(device)

    all_manifold = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i : i + batch_size].to(device)
        batch_std = (batch - mean) / std
        manifold_std = glp_denoise(glp_model, batch_std, t_start, num_steps)
        all_manifold.append((manifold_std * std + mean).cpu())

    manifold = torch.cat(all_manifold, dim=0)
    return activations - manifold, manifold


@torch.no_grad()
def velocity_curvature(glp_model, z_std, direction, t=0.5, eps=0.1, n_random=50):
    """Measure how the velocity field curves in a given direction vs. random.

    Returns (curvature_in_direction, mean_random_curvature, ratio).
    High ratio = the manifold encodes structure along this direction.
    """
    device = next(glp_model.parameters()).device
    d_unit = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t_batch = torch.full((z_std.shape[0],), t, device=device)

    v_base = glp_model(z_std, t_batch)
    v_pert = glp_model(z_std + eps * d_unit, t_batch)
    curv_dir = (v_pert - v_base).norm(dim=-1) / eps

    random_curvs = []
    for _ in range(n_random):
        r = torch.randn_like(direction)
        r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_r = glp_model(z_std + eps * r, t_batch)
        random_curvs.append(((v_r - v_base).norm(dim=-1) / eps))
    curv_rand = torch.stack(random_curvs).mean(dim=0)

    ratio = curv_dir / curv_rand.clamp(min=1e-8)
    return curv_dir, curv_rand, ratio


def load_glp(glp_dir, device="cpu"):
    """Load a trained GLP from disk. Returns (model, act_stats, config)."""
    import json, os
    with open(os.path.join(glp_dir, "glp_results.json")) as f:
        results = json.load(f)
    cfg = results["config"]
    model = GLPDenoiser(
        act_dim=cfg["act_dim"], d_model=cfg["d_model"],
        d_mlp=cfg["d_mlp"], n_layers=cfg["n_layers"],
    )
    model.load_state_dict(torch.load(
        os.path.join(glp_dir, "glp_model.pt"), map_location=device
    ))
    act_stats = torch.load(os.path.join(glp_dir, "glp_stats.pt"), map_location=device)
    return model, act_stats, cfg
