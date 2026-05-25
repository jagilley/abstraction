"""VPD (adVersarial Parameter Decomposition) for language reduction models.

Decomposes a trained language reduction GPT into interpretable parameter
subcomponents using the nano_param_decomp implementation from Goodfire's
param-decomp repo (vendored at vendor/param-decomp/).

The key adaptation: our GPT uses a combined c_attn (QKV in one Linear).
We split it into separate q_proj/k_proj/v_proj before decomposing so that
VPD can find independent Q and K subcomponents — this unlocks the paper's
QK-circuit and OV-circuit analysis.

Usage (Modal):
    modal run language_reduction/vpd.py --tau 0.0
    modal run language_reduction/vpd.py --tau 0.3

Usage (local, single GPU):
    python language_reduction/vpd.py --tau 0.0 --local --no-wandb
"""

import math
import os
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from language_reduction.model import GPT

DATA_DIR = "/data"

vpd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.7.0",
        "matplotlib",
        "pillow",
        "wandb",
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_python_source("language_reduction")
    .add_local_dir(
        "vendor/param-decomp/nano_param_decomp",
        remote_path="/root/nano_param_decomp",
    )
)

vpd_volume = modal.Volume.from_name("language-reduction-data", create_if_missing=True)
vpd_app = modal.App("language-reduction-vpd", image=vpd_image)


# ---------------------------------------------------------------------------
# Split combined c_attn into separate Q/K/V projections
# ---------------------------------------------------------------------------


class SplitAttention(nn.Module):
    """CausalSelfAttention with separate q/k/v projections instead of combined c_attn.

    Initialized from a trained GPT's CausalSelfAttention by slicing the c_attn weight.
    Functionally identical — just exposes Q, K, V as separate nn.Linear modules
    so VPD can decompose them independently.
    """

    def __init__(self, attn):
        super().__init__()
        n_embd = attn.n_embd
        n_head = attn.n_head
        head_dim = attn.head_dim
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = head_dim
        self.use_rope = attn.use_rope

        self.q_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=True)

        # Slice the combined c_attn weight [3*n_embd, n_embd] into Q, K, V
        W = attn.c_attn.weight.data
        b = attn.c_attn.bias.data
        self.q_proj.weight.data.copy_(W[:n_embd])
        self.k_proj.weight.data.copy_(W[n_embd : 2 * n_embd])
        self.v_proj.weight.data.copy_(W[2 * n_embd :])
        self.q_proj.bias.data.copy_(b[:n_embd])
        self.k_proj.bias.data.copy_(b[n_embd : 2 * n_embd])
        self.v_proj.bias.data.copy_(b[2 * n_embd :])

        self.o_proj.weight.data.copy_(attn.c_proj.weight.data)
        self.o_proj.bias.data.copy_(attn.c_proj.bias.data)

        if not self.use_rope:
            self.register_buffer("bias", attn.bias.clone())

    def _apply_rope(self, x, seq_len):
        d = self.head_dim
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        dim = torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
        freqs = pos[:, None] / (10000.0 ** (dim[None, :] / d))
        cos = freqs.cos()
        sin = freqs.sin()
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self._apply_rope(q, T)
            k = self._apply_rope(k, T)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if self.use_rope:
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal, float("-inf"))
        else:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


def split_attention_layers(model: GPT) -> GPT:
    """Replace all CausalSelfAttention modules with SplitAttention equivalents."""
    for block in model.transformer.h:
        block.attn = SplitAttention(block.attn)
    return model


# ---------------------------------------------------------------------------
# Component counts — scaled for our 128-dim, 2-layer model
# ---------------------------------------------------------------------------


def make_c_per_module(n_layer: int = 2, n_embd: int = 128) -> dict[str, int]:
    """Determine component counts per module.

    Heuristic from the VPD paper's 4L model: ~2x smaller dim for attention,
    ~1-2x for MLP. These are starting points — tune based on alive counts.
    The paper's 4L (768-dim) used 512 for Q/K, 1024 for V/O, 3072/3584 for MLP.
    Scaling down proportionally for our 128-dim model.
    """
    c_per_module = {}
    for i in range(n_layer):
        c_per_module[f"transformer.h.{i}.attn.q_proj"] = 2 * n_embd
        c_per_module[f"transformer.h.{i}.attn.k_proj"] = 2 * n_embd
        c_per_module[f"transformer.h.{i}.attn.v_proj"] = 2 * n_embd
        c_per_module[f"transformer.h.{i}.attn.o_proj"] = 2 * n_embd
        c_per_module[f"transformer.h.{i}.mlp.0"] = 4 * n_embd
        c_per_module[f"transformer.h.{i}.mlp.2"] = 4 * n_embd
    return c_per_module


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def make_loader(
    data: Tensor, batch_size: int, seq_len: int, seed: int
) -> Iterator[Tensor]:
    """Infinite random-window loader over a flat token tensor."""
    rng = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(len(data) - seq_len, (batch_size,), generator=rng)
        yield torch.stack([data[i : i + seq_len] for i in ix])


def load_tokens(tau: float, n_tokens: int = 10_000_000, kappa_mode: str = "energy") -> Tensor:
    """Load token shards from the Modal volume (or local /data)."""
    import glob

    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    if tau == 0.0:
        data_dir = f"{DATA_DIR}/tokens"
    else:
        data_dir = f"{DATA_DIR}/denoised/tau_{tau:.3f}{suffix}"

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    assert len(shard_paths) > 0, f"No shards found in {data_dir}"

    all_tokens = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_tokens)[:n_tokens]
    return torch.from_numpy(data.astype(np.int64))


def load_model(
    tau: float,
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    n_embd: int = 128,
    kappa_mode: str = "energy",
) -> GPT:
    """Load a trained language reduction model checkpoint."""
    import json

    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    model_dir = f"{DATA_DIR}/models{suffix}/tau_{tau:.3f}/P_{n_tokens}/T_{block_size}"
    results_path = os.path.join(model_dir, "results.json")
    with open(results_path) as f:
        results = json.load(f)

    vocab_size = 50257
    meta_path = f"{DATA_DIR}/tokens/meta.npy"
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True).item()
        vocab_size = meta["vocab_size"]

    model = GPT(
        vocab_size, block_size, n_layer, n_head, n_embd,
        tie_weights=results.get("tie_weights", False),
    )
    state_dict = torch.load(
        os.path.join(model_dir, "model.pt"), weights_only=True, map_location="cpu"
    )
    model.load_state_dict(state_dict)
    print(f"Loaded model from {model_dir} (val_loss={results['best_val_loss']:.4f})")
    return model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VPDConfig:
    tau: float = 0.0
    n_tokens: int = 10_000_000
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    kappa_mode: str = "energy"

    # VPD training
    n_steps: int = 100_000
    batch_size: int = 64
    seq_len: int = 128
    seed: int = 0

    # CI transformer — scaled down from paper defaults (2048/8/16/8192)
    # for our 13M-param target model. 15.9M CI params ≈ target model size.
    ci_d_model: int = 512
    ci_n_blocks: int = 4
    ci_n_heads: int = 8
    ci_mlp_hidden: int = 2048

    use_wandb: bool = True
    wandb_project: str = "language-reduction-vpd"


# ---------------------------------------------------------------------------
# Main decomposition function
# ---------------------------------------------------------------------------


def run_vpd(cfg: VPDConfig):
    """Load model + data, split attention, run VPD decomposition."""
    sys.path.insert(0, "/root")  # for Modal (nano_param_decomp lives at /root/)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "param-decomp"))
    from nano_param_decomp.run import Config as NanoConfig
    from nano_param_decomp.run import decompose
    import nano_param_decomp.run as nano_run

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")

    print(f"=== VPD decomposition for tau={cfg.tau} ===")

    model = load_model(
        cfg.tau, cfg.n_tokens, cfg.block_size,
        cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.kappa_mode,
    )
    model = split_attention_layers(model)

    print("Module structure after splitting c_attn → q_proj/k_proj/v_proj:")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            print(f"  {name:50s} {tuple(mod.weight.shape)}")

    # nano_param_decomp expects forward(input_ids) → logits
    original_forward = model.forward

    def forward_logits_only(self, idx):
        logits, _ = original_forward(idx)
        return logits

    model.forward = types.MethodType(forward_logits_only, model)

    data = load_tokens(cfg.tau, cfg.n_tokens, cfg.kappa_mode)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Loaded {len(data):,} tokens (train={len(train_data):,}, val={len(val_data):,})")

    checkpoint_dir = f"{DATA_DIR}/vpd/tau_{cfg.tau:.3f}"

    # Wrap save_checkpoint to also commit the Modal volume
    _original_save = nano_run.save_checkpoint
    def _save_and_commit(*args, **kwargs):
        _original_save(*args, **kwargs)
        try:
            vpd_volume.commit()
            print("[checkpoint] volume committed", flush=True)
        except Exception:
            pass  # not on Modal (local run)
    nano_run.save_checkpoint = _save_and_commit

    c_per_module = make_c_per_module(cfg.n_layer, cfg.n_embd)
    nano_cfg = NanoConfig(
        C_per_module=c_per_module,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        seed=cfg.seed,
        ci_d_model=cfg.ci_d_model,
        ci_n_blocks=cfg.ci_n_blocks,
        ci_n_heads=cfg.ci_n_heads,
        ci_mlp_hidden=cfg.ci_mlp_hidden,
        eval_batch_size=16,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=1000,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_run_name=f"vpd_tau{cfg.tau:.3f}",
    )

    train_loader = make_loader(train_data, cfg.batch_size, cfg.seq_len, cfg.seed)
    eval_loader = make_loader(val_data, nano_cfg.eval_batch_size, cfg.seq_len, cfg.seed + 1)

    decompose(model, nano_cfg, train_loader, eval_loader)


# ---------------------------------------------------------------------------
# Modal entry point
# ---------------------------------------------------------------------------


def profile_memory(cfg: VPDConfig):
    """Run a few training steps at various batch sizes and report GPU memory."""
    sys.path.insert(0, "/root")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "param-decomp"))
    from nano_param_decomp.run import (
        Config as NanoConfig, install_components, CITransformer, SPDModule,
        faithfulness_loss, importance_minimality_loss, stochastic_recon_loss,
        PersistentPGD, kl_logits, clear_wrapper_masks,
    )
    import time

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / 1e9
    print(f"GPU: {props.name}, {total_mem_gb:.1f} GB")

    model = load_model(cfg.tau, cfg.n_tokens, cfg.block_size,
                       cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.kappa_mode)
    model = split_attention_layers(model)
    original_forward = model.forward
    def forward_logits_only(self, idx):
        logits, _ = original_forward(idx)
        return logits
    model.forward = types.MethodType(forward_logits_only, model)

    data = load_tokens(cfg.tau, cfg.n_tokens, cfg.kappa_mode)
    train_data = data[: int(0.9 * len(data))]

    for batch_size in [64, 128, 256, 512]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            c_per_module = make_c_per_module(cfg.n_layer, cfg.n_embd)
            # Reload model fresh each time
            model2 = load_model(cfg.tau, cfg.n_tokens, cfg.block_size,
                                cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.kappa_mode)
            model2 = split_attention_layers(model2)
            orig_fwd = model2.forward
            def fwd(self, idx):
                logits, _ = orig_fwd(idx)
                return logits
            model2.forward = types.MethodType(fwd, model2)

            nano_cfg = NanoConfig(
                C_per_module=c_per_module, n_steps=10, batch_size=batch_size,
                seq_len=cfg.seq_len, seed=0,
                ci_d_model=cfg.ci_d_model, ci_n_blocks=cfg.ci_n_blocks,
                ci_n_heads=cfg.ci_n_heads, ci_mlp_hidden=cfg.ci_mlp_hidden,
                eval_batch_size=16, eval_freq=999999, slow_eval_freq=999999,
                use_wandb=False, log_every=5,
            )

            loader = make_loader(train_data, batch_size, cfg.seq_len, 0)
            eval_loader = make_loader(train_data, 16, cfg.seq_len, 1)

            from nano_param_decomp.run import decompose
            t0 = time.time()
            decompose(model2, nano_cfg, loader, eval_loader)
            elapsed = time.time() - t0

            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            per_step = elapsed / 10
            projected_50k = per_step * 50_000 / 3600

            print(f"\n>>> batch_size={batch_size}: peak={peak_gb:.1f}/{total_mem_gb:.1f} GB, "
                  f"{per_step:.3f} s/step, 50K steps ≈ {projected_50k:.1f} hours\n")

        except torch.cuda.OutOfMemoryError:
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n>>> batch_size={batch_size}: OOM (peak was {peak_gb:.1f}/{total_mem_gb:.1f} GB)\n")
            torch.cuda.empty_cache()


@vpd_app.function(
    volumes={DATA_DIR: vpd_volume},
    gpu="L4",
    timeout=86400,
    memory=32768,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_vpd_modal(tau: float = 0.0, n_steps: int = 100_000, use_wandb: bool = False,
                  profile: bool = False):
    cfg = VPDConfig(tau=tau, n_steps=n_steps, use_wandb=use_wandb)
    if profile:
        profile_memory(cfg)
    else:
        run_vpd(cfg)


@vpd_app.local_entrypoint()
def main(
    tau: float = 0.0,
    n_steps: int = 100_000,
    local: bool = False,
    no_wandb: bool = False,
    profile: bool = False,
):
    if local:
        cfg = VPDConfig(tau=tau, n_steps=n_steps, use_wandb=not no_wandb)
        if profile:
            profile_memory(cfg)
        else:
            run_vpd(cfg)
    else:
        run_vpd_modal.remote(tau=tau, n_steps=n_steps, use_wandb=not no_wandb,
                             profile=profile)
