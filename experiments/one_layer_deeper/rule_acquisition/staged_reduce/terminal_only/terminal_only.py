"""Terminal-only staged reduce: can a staged *forward pass* discover long division by itself?

**Fork notice.** This file is a fork of
[`../staged_reduce.py`](../staged_reduce.py) (donor, unedited). Prompt layout, architecture,
parameter count, modulus family, hash split, eval-pool construction and `persist()` are copied
verbatim so every readout here is bit-identical to the donor's for a matching modulus set.
What changed is the *supervision* and the *forward pass*; the diff is listed in §"What is
forked and what changed" below.

---

## The question

The donor established that schoolbook long division makes every stage a bounded-quotient
reduce, and that a model trained **only on the single-stage distribution** (uniform
`y' ∈ [0, R·N)`, label `y' mod N`) and chained *at test time* reads 0.9968 on held-out `y` at
3 digits against a monolithic control's 0.4841, and — the only movement on the rule axis
anywhere in this program — 0.9826 on **held-out `N`** at 142 moduli.

The catch for the upstream competition is that the single-stage training distribution is
*self-made*. Constructing `(y', y' mod N)` pairs is legal under the rules (it needs no
trapdoor), but it is data augmentation the participant invents; the only supervision the
benchmark hands you is the **terminal** label, plus label-free self-consistency of the
model's own states. Every composed arm in this program trained terminal-only sits at floor —
but every one of those had a *monolithic* reduce.

So: **can a staged forward pass self-organise the decomposition from terminal labels alone,
given breadth of moduli?**

The task is exactly `sr3_mono`'s — `(N, y) → y mod N` with `y ~ U[0, N^2)`, loss on the final
remainder only. The forward pass is `S` applications of **one tied stage map**:

    r_0 = 0;   for i = S-1 … 0:   r_{i+1} = StageMap(N, r_i, c_i)

where `c_i` is the `i`-th radix-`R` digit of `y` and `StageMap` is the donor's
encoder → tied operator → digit decoder run on the donor's prompt
`BOS DIV N <w digits> X <2w digits of (r·R + c)> ANS`. The remainder is **re-grounded between
stages through the model's own decode**, so the only channel from one stage to the next is a
`w`-digit residue — which is what forces the carried quantity to be residue-like at all.

Two re-grounding modes, both keeping gradient flow to every stage unbroken:

  - `st` — **hard with a straight-through path.** The re-encoded digits are the argmax
    one-hots (so the forward pass is bit-identical to the donor's test-time chain), with
    `hard + p - p.detach()` carrying the gradient. The seam channel is exactly `w` decimal
    digits.
  - `soft` — **soft in-place re-encode**, `ballistic_depth.forward_soft_digits` /
    `dress_rehearsal/submission.py` v3's mechanic one level down: the decoded digit
    distribution is written into the `x` field's remainder slots as a convex combination of
    the *real* digit embeddings. Unbiased gradients, but the seam channel is `w × 10` reals
    rather than `w` digits.

Because `R` is a power of ten, `r·R + c` is the digit string of `r` followed by the digit
string of `c`, right-aligned in the `2w` field with leading zeros. The soft re-encode is
therefore **exact at the digit level** — it writes `r`'s decoded distributions into `r`'s own
slots and leaves `c`'s slots hard — rather than needing any positional surgery.

**Legality.** Radix digits of `y` come from `(N, y)` alone; the loss is the terminal label
only; the optional seam term uses no labels. Nothing here needs a factorisation or `φ(N)`.

## The readouts

Three families, and the third is the point of the node.

1. **chain** — the arm's native staged forward on the donor's chain pools
   (`y ~ U[0, N^2)`, modulus-uniform, split by the same hash of `(N, y)`). Bit-identical
   pools to `sr3_mono` / `sr3_r10` at 8 moduli and `sr3_mono_many` / `sr3_r10_many` at 142,
   so `chain_heldout_y` and `chain_heldout_n` drop straight into the donor's tables. For a
   `soft` arm, `chainhard_*` additionally scores the *hard* argmax chain, which is what
   separates "the decomposition is discrete" from "the seam is a continuous side-channel".
2. **stage** — the tied stage map applied **once** to a bounded-quotient query
   `(N, y')`, `y' ∈ [0, R·N)`, scored against `y' mod N`. These pools are bit-identical to
   `sr3_r10`'s *training* distribution, which this arm never sees. It is the sharpest single
   probe in the node: `stage_all` at or near `sr3_r10`'s 0.9978 means the staged forward
   discovered the same decomposition from terminal labels.
3. **per-stage diagnostics** (`chain_stage_probe`) — for every stage index, on the chain
   pool: whether the decoded intermediate equals the *true* partial remainder
   `(y // R^i) mod N`; whether it merely copies its predecessor (identity collapse); whether
   it lies in `[0, N)`; the modal-value share; and two **relabel purities** —
   `P(decoded is the group mode | N, true partial remainder)` and its inverse. Purity high
   with true-match low is a model carrying a *permuted* residue, which is a decomposition
   discovered but not in the schoolbook coordinate; both low is a monolithic collapse with
   an arbitrary prefix code.

`chain_teacher` (donor's) injects the true partial remainder at every stage, separating the
per-stage rate from propagation.

## What is forked and what changed

| | donor `staged_reduce.py` | here |
|---|---|---|
| training distribution | `y' ~ U[0, R·N)` (single stage) | `y ~ U[0, N^2)` (**the whole task**) |
| supervision | `y' mod N` per stage query | `y mod N`, **terminal only** |
| forward pass | one encode+op+decode | `S` tied applications, re-grounded at each seam |
| chaining | test time only | **training time**, gradient through every stage |
| stage pools `[0, R·N)` | the training set | a **probe** of a distribution never trained on |
| `radix=0` | the monolithic control | unchanged, and identical to `sr3_mono` |

Everything else — `ModulusFamily`, `is_train`, `_div_pool`, `_chain_pool`, `_make_model`,
`certifiable_T`, `persist`, the log grid — is the donor's.

Usage:
  MODAL_PROFILE=chromatic modal run --detach \\
    one_layer_deeper/rule_acquisition/staged_reduce/terminal_only/terminal_only.py::terminal_only \\
    --tag to3a --arms "to3_st10" --steps 600000 --seed 0
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

GPU = os.environ.get("OLD_GPU", "L4")
TIMEOUT = int(os.environ.get("OLD_TIMEOUT", 79200))

DIGIT_OFFSET = 7
SQ_TOK = DIGIT_OFFSET + 10       # 17
DIV_TOK = DIGIT_OFFSET + 10 + 1  # 18
VOCAB = DIV_TOK + 1

HARD_EXAMPLES = 768


def certifiable_T(eps: float, m: int = HARD_EXAMPLES) -> float:
    """Deepest ladder rung a one-step error rate `eps` could certify at ~50% odds."""
    return float("inf") if eps <= 0 else math.log(2.0) / (m * eps)


def stages_for(radix: int, w: int) -> tuple[int, int]:
    """`(radix_eff, n_stages)` — how many radix-`R` chunks cover a `2w`-digit dividend.

    `radix=0` is the monolithic control: one stage over the whole `[0, N^2)` range, which
    makes it the `S = 1` endpoint of the same sweep rather than a separate code path.
    Donor's, verbatim.
    """
    if radix == 0:
        return 10 ** (2 * w), 1
    cap, s = 1, 0
    while cap < 10 ** (2 * w):
        cap *= radix
        s += 1
    return radix, s


BASE_ARM = dict(
    n_digits=3,
    # Structure of the *forward pass*, not of the training distribution: the model applies
    # the tied stage map `S` times, consuming one radix-`R` digit of `y` per application.
    # 0 -> the monolithic S=1 control (identical to the donor's `sr3_mono`).
    # Must be a power of ten so the soft re-encode is digit-aligned.
    radix=10,
    # `st`   -> hard argmax re-encode with a straight-through gradient path
    # `soft` -> convex combination of the real digit embeddings
    stage_mode="st",
    # Label-free seam term. `none` | `ent` (mean entropy of the intermediate digit
    # distributions) | `cos` (the state you hold must match a clean hard re-encode of the
    # remainder you decoded; identically zero under `st`, so only meaningful for `soft`).
    seam_mode="none",
    seam_w=0.0,
    train_frac=0.5,
    dense_n=False,
    min_margin=20,
    max_moduli=8,
    # --- architecture (identical to the donor's BASE_ARM, hence to `exact_atom`'s) ------
    d_model=256,
    n_enc_layers=2,
    d_ff=1024,
    d_op_ff=1024,
    inner_steps=1,
)

ARMS: dict[str, dict] = {
    # ---- the 2x2: {straight-through, soft} x {8, 142 moduli} -------------------------
    # 8 moduli: `min_margin=20, max_moduli=8` — the donor's `sr3_mono` / `sr3_r10` cells,
    # so the chain pools are bit-identical to theirs.
    "to3_st10": dict(radix=10, stage_mode="st"),
    "to3_soft10": dict(radix=10, stage_mode="soft"),
    # 142 moduli: `min_margin=2, max_moduli=0` — the donor's `sr3_mono_many` /
    # `sr3_r10_many` cells. This is where the donor's rule axis moved.
    "to3_st10_many": dict(radix=10, stage_mode="st", min_margin=2, max_moduli=0),
    "to3_soft10_many": dict(radix=10, stage_mode="soft", min_margin=2, max_moduli=0),

    # ---- defined, held in reserve ----------------------------------------------------
    # The monolithic control at terminal supervision *is* the donor's `sr3_mono` /
    # `sr3_mono_many` (same distribution, same loss, S=1); defined here so the S=1 endpoint
    # of this sweep is runnable in-file if a check is ever wanted.
    "to3_mono": dict(radix=0, stage_mode="st"),
    "to3_mono_many": dict(radix=0, stage_mode="st", min_margin=2, max_moduli=0),
    # Family-matched 8-modulus control, against the donor's `sr3_r10_m8`.
    "to3_st10_m8": dict(radix=10, stage_mode="st", min_margin=2, max_moduli=8),
    # Fewer, harder stages.
    "to3_st100_many": dict(radix=100, stage_mode="st", min_margin=2, max_moduli=0),
    # The label-free seam term (legal: no labels).
    "to3_soft10_ent_many": dict(
        radix=10, stage_mode="soft", seam_mode="ent", seam_w=0.03,
        min_margin=2, max_moduli=0),
    "to3_soft10_cos_many": dict(
        radix=10, stage_mode="soft", seam_mode="cos", seam_w=1.0,
        min_margin=2, max_moduli=0),
    # 4 digits, only if 3 moves.
    "to4_st10_many": dict(n_digits=4, radix=10, stage_mode="st",
                          min_margin=2, max_moduli=0),
}

CFG = dict(
    min_factor=3,
    heldout_mod_fraction=0.2,
    family_seed=45,
    n_heads=4,
    steps=300000,
    batch_size=256,
    lr=3e-4,
    warmup=500,
    weight_decay=0.1,
    grad_clip=1.0,
    const_lr=True,
    exhaustive_cap=4_000_000,
    n_fine=262_144,
    n_chain=65_536,
    eval_chunk=8192,
    n_log_points=24,
    seed=0,
    tag="smoke",
)

HASH_M = 1_000_000


def _make_model(cfg, arm, max_len, n_ans, device):
    """The donor's model, plus `Encoder.forward_emb` so a stage can be fed soft embeddings.

    `forward_emb` adds no parameters and `forward` routes through it, so the parameter count
    and the hard-token forward pass are bit-identical to the donor's.
    """
    import torch
    from torch import nn

    d = arm["d_model"]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, d)
            self.pos = nn.Parameter(torch.zeros(max_len, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg["n_heads"],
                dim_feedforward=arm["d_ff"],
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.body = nn.TransformerEncoder(layer, num_layers=arm["n_enc_layers"])
            self.norm = nn.LayerNorm(d)

        def forward_emb(self, emb, read_pos):
            h = emb + self.pos.unsqueeze(0)
            return self.norm(self.body(h))[:, read_pos, :]

        def forward(self, input_ids, read_pos):
            return self.forward_emb(self.tok(input_ids), read_pos)

    class Block(nn.Module):
        """The tied operator, identical in form to every cut in this experiment."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.net = nn.Sequential(
                nn.Linear(d, arm["d_op_ff"]), nn.GELU(), nn.Linear(arm["d_op_ff"], d)
            )

        def forward(self, h):
            return h + self.net(self.norm(h))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder()
            self.op = Block()
            self.dnorm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_ans * 10)
            self.inner = arm["inner_steps"]

        def dec(self, h):
            return self.head(self.dnorm(h)).reshape(h.shape[0], n_ans, 10)

        def roll(self, h, n_task_steps=1):
            for _ in range(n_task_steps * self.inner):
                h = self.op(h)
            return h

    return Model().to(device)


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=TIMEOUT, memory=32768)
def terminal_only(
    tag: str = "smoke",
    arms: str = "to3_st10",
    steps: int = 300000,
    seed: int = 0,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    n_fine: int = 262144,
    n_chain: int = 65536,
    n_log_points: int = 24,
    const_lr: bool = True,
    save_ckpt: bool = True,
    tf32: bool = False,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    base_cfg = {
        **CFG,
        "tag": tag,
        "arms": arms,
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "n_fine": n_fine,
        "n_chain": n_chain,
        "n_log_points": n_log_points,
        "const_lr": const_lr,
    }
    results = {"config": base_cfg, "arm_specs": {}, "arms": {}}

    for arm_name in [a.strip() for a in arms.split(",") if a.strip()]:
        if arm_name not in ARMS:
            raise ValueError(f"unknown arm {arm_name!r}; known: {sorted(ARMS)}")
        arm = {**BASE_ARM, **ARMS[arm_name]}
        cfg = {**base_cfg, **arm}
        results["arm_specs"][arm_name] = arm
        torch.manual_seed(seed)
        np.random.seed(seed)
        w = arm["n_digits"]
        radix_eff, n_stages = stages_for(arm["radix"], w)
        if arm["radix"]:
            if arm["radix"] > 10**w:
                raise ValueError(f"radix {arm['radix']} > 10^{w}: dividend would not fit 2w digits")
            m_c = int(round(math.log10(arm["radix"])))
            if 10**m_c != arm["radix"]:
                raise ValueError("radix must be a power of ten (the soft re-encode is digit-aligned)")
        else:
            m_c = 2 * w  # monolithic: the whole dividend is 'the digit', and there is no seam
        print(
            f"\n===== arm={arm_name} seed={seed} spec={ARMS[arm_name]} "
            f"radix={arm['radix']} -> {n_stages} forward stage(s), mode={arm['stage_mode']}, "
            f"seam={arm['seam_mode']}@{arm['seam_w']} =====",
            flush=True,
        )

        # ------------------------------ the modulus set ------------------------------
        # Donor's, verbatim: same family, same subsample RNG, same held-out pool.
        rng = np.random.default_rng(base_cfg["family_seed"])
        family = ModulusFamily(
            n_digits=w,
            min_margin=arm["min_margin"],
            min_factor=base_cfg["min_factor"],
            heldout_fraction=base_cfg["heldout_mod_fraction"],
            seed=base_cfg["family_seed"],
        )
        fam_train, fam_held = family.split()
        fam_info = family.describe()
        if arm["max_moduli"] and len(fam_train) > arm["max_moduli"]:
            pick = np.random.default_rng(base_cfg["family_seed"] + 1).choice(
                len(fam_train), arm["max_moduli"], replace=False
            )
            fam_train = sorted(fam_train[i] for i in pick)
        if arm["max_moduli"] and len(fam_held) > 4 * arm["max_moduli"]:
            pick = np.random.default_rng(base_cfg["family_seed"] + 2).choice(
                len(fam_held), 4 * arm["max_moduli"], replace=False
            )
            fam_held = sorted(fam_held[i] for i in pick)
        train_mods = [m[0] for m in fam_train]
        held_mods = [m[0] for m in fam_held]

        # The dense-`N` selector is *degenerate* in the donor (`(N*2246822519) % 10 == 0` is
        # exactly `N ≡ 0 mod 10`) and is carried here only so the fork stays a fork. No arm
        # in this node sets `dense_n`; fix the hash before ever doing so.
        dense_all = np.arange(max(2, 10 ** (w - 1)), 10**w, dtype=np.int64)
        dense_held_mask = (
            ((dense_all * 2246822519) % 10 == 0) | np.isin(dense_all, held_mods)
        ) & ~np.isin(dense_all, train_mods)
        dense_train = dense_all[~dense_held_mask]
        dense_held = dense_all[dense_held_mask]

        mods_tr_np = dense_train if arm["dense_n"] else np.array(train_mods, dtype=np.int64)
        mods_he_np = dense_held if arm["dense_n"] else np.array(held_mods, dtype=np.int64)
        pool_tr = torch.tensor(mods_tr_np, dtype=torch.long, device=device)
        print(
            f"[family] {len(train_mods)} train / {len(held_mods)} held semiprime moduli"
            f" | training on {mods_tr_np.size} moduli",
            flush=True,
        )

        # ------------------------------ prompt layout --------------------------------
        # Donor's, verbatim: BOS DIV N <w digits> X <2w digits> ANS.
        w_x = 2 * w
        n_ans = w
        max_len = 1 + 1 + (1 + w) + (1 + w_x) + 1
        read_pos = max_len - 1
        x_start = 4 + w                       # first slot of the 2w-digit dividend field
        r_slot0 = x_start + (w_x - w - m_c)   # where the carried remainder's digits sit
        r_slot1 = r_slot0 + w

        def digits_of(v, width):
            return torch.stack([(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1)

        pow10 = torch.tensor([10 ** (w - 1 - k) for k in range(w)], device=device)

        def value_of(dig):
            """Inverse of `digits_of` for the w-digit answer field."""
            return (dig * pow10).sum(-1)

        def prompts_for(n_vals, x_vals):
            rows = x_vals.shape[0]
            col = lambda t: torch.full((rows, 1), t, dtype=torch.long, device=device)
            return torch.cat(
                [
                    col(TOKEN_IDS["BOS"]),
                    col(DIV_TOK),
                    col(TOKEN_IDS["N"]),
                    digits_of(n_vals, w) + DIGIT_OFFSET,
                    col(TOKEN_IDS["X"]),
                    digits_of(x_vals, w_x) + DIGIT_OFFSET,
                    col(TOKEN_IDS["ANS"]),
                ],
                dim=1,
            )

        # ------------------------------ the hash split -------------------------------
        # Donor's, verbatim.
        thr = int(round(arm["train_frac"] * HASH_M))

        def is_train(n_vals, x_vals):
            return ((x_vals * 2654435761 + n_vals * 40503) % HASH_M) < thr

        def hi_task(n_vals):
            """The *task's* dividend range: always `[0, N^2)` — this arm trains on the whole
            problem, not on a stage. This is the donor's `hi_for` with `radix=0`."""
            return n_vals * n_vals

        def hi_stage(n_vals):
            """The stage *probe's* range, `[0, R·N)` — the donor's `sr3_r10` training space,
            which this arm never trains on."""
            if arm["radix"] == 0:
                return n_vals * n_vals
            return arm["radix"] * n_vals

        def hi_stage_int(n: int) -> int:
            return n * n if arm["radix"] == 0 else arm["radix"] * n

        # ------------------------------ samplers -------------------------------------
        def sample_n(pool, n, gen=None):
            return pool[torch.randint(0, pool.numel(), (n,), device=device, generator=gen)]

        def sample_task(n_pool, n, want_train, gen=None):
            """`(N, y)` with `y ~ U[0, N^2)` on the requested half of the hash split."""
            got_n, got_y, have = [], [], 0
            for _ in range(64):
                nv = sample_n(n_pool, 4 * n, gen)
                hi = hi_task(nv)
                u = torch.rand(4 * n, device=device, generator=gen, dtype=torch.float64)
                y = torch.minimum((u * hi).long(), hi - 1)
                keep = is_train(nv, y) == want_train
                got_n.append(nv[keep])
                got_y.append(y[keep])
                have += int(keep.sum())
                if have >= n:
                    break
            return torch.cat(got_n)[:n], torch.cat(got_y)[:n]

        def sample_stage(n_pool, n, want_train, gen=None):
            got_n, got_y, have = [], [], 0
            for _ in range(64):
                nv = sample_n(n_pool, 4 * n, gen)
                hi = hi_stage(nv)
                u = torch.rand(4 * n, device=device, generator=gen, dtype=torch.float64)
                y = torch.minimum((u * hi).long(), hi - 1)
                keep = is_train(nv, y) == want_train
                got_n.append(nv[keep])
                got_y.append(y[keep])
                have += int(keep.sum())
                if have >= n:
                    break
            return torch.cat(got_n)[:n], torch.cat(got_y)[:n]

        # ------------------------------ eval pools -----------------------------------
        # Donor's `_div_pool`, verbatim except that it reads `hi_stage`, so for a
        # `radix=10` arm these are bit-identical to `sr3_r10`'s stage pools and for a
        # `radix=0` arm bit-identical to `sr3_mono`'s.
        ev_gen = torch.Generator(device=device).manual_seed(12345)
        EVAL: dict[str, tuple] = {}

        def _div_pool(n_pool_np, want_train, cap):
            space = int(sum(hi_stage_int(int(n)) for n in n_pool_np))
            if space <= base_cfg["exhaustive_cap"]:
                ns, ys = [], []
                for n_val in n_pool_np:
                    hi = hi_stage_int(int(n_val))
                    y = torch.arange(hi, device=device, dtype=torch.long)
                    ns.append(torch.full_like(y, int(n_val)))
                    ys.append(y)
                nv, y = torch.cat(ns), torch.cat(ys)
                if want_train is not None:
                    keep = is_train(nv, y) == want_train
                    nv, y = nv[keep], y[keep]
                if nv.numel() <= cap:
                    return nv, y, True
                pick = torch.randperm(nv.numel(), generator=ev_gen, device=device)[:cap]
                return nv[pick], y[pick], False
            pool_t = torch.tensor(n_pool_np, dtype=torch.long, device=device)
            if want_train is None:
                nv = sample_n(pool_t, cap, ev_gen)
                hi = hi_stage(nv)
                u = torch.rand(cap, device=device, generator=ev_gen, dtype=torch.float64)
                y = torch.minimum((u * hi).long(), hi - 1)
                return nv, y, False
            nv, y = sample_stage(pool_t, cap, want_train, ev_gen)
            return nv, y, False

        nv, y, ex = _div_pool(mods_tr_np, False, n_fine)
        EVAL["heldout_y"] = (nv, y, y % nv, ex)
        nv2, y2, ex2 = _div_pool(mods_tr_np, True, min(n_fine, 65536))
        EVAL["seen_y"] = (nv2, y2, y2 % nv2, ex2)
        if mods_he_np.size:
            nvh, yh, exh = _div_pool(mods_he_np, True, n_fine)
            EVAL["heldout_n"] = (nvh, yh, yh % nvh, exh)
        # Appended *after* the donor's three so their `ev_gen` draws are unchanged. This arm
        # trains on neither half of the stage split, so the unsplit pool is the honest read.
        nva, ya, exa = _div_pool(mods_tr_np, None, n_fine)
        EVAL["stage_all"] = (nva, ya, ya % nva, exa)

        # ------------------------------ chain pools ----------------------------------
        # Donor's `_chain_pool`, verbatim. `ch_gen` is independent of `ev_gen`, so these are
        # bit-identical to the donor's for any arm sharing a modulus set.
        ch_gen = torch.Generator(device=device).manual_seed(777)

        def _chain_pool(n_pool_np, want_train, total):
            n_mods = int(n_pool_np.size)
            per = max(64, total // max(1, n_mods))
            mods = torch.tensor(n_pool_np, dtype=torch.long, device=device)
            out_n, out_y = [], []
            for _ in range(64):
                nv = mods.repeat_interleave(per)
                u = torch.rand(nv.numel(), device=device, generator=ch_gen, dtype=torch.float64)
                yv = torch.minimum((u * (nv * nv)).long(), nv * nv - 1)
                if want_train is not None:
                    keep = is_train(nv, yv) == want_train
                    nv, yv = nv[keep], yv[keep]
                out_n.append(nv)
                out_y.append(yv)
                if sum(t.numel() for t in out_n) >= total:
                    break
            nv, yv = torch.cat(out_n), torch.cat(out_y)
            pick = torch.randperm(nv.numel(), generator=ch_gen, device=device)[:total]
            return nv[pick], yv[pick]

        CHAIN: dict[str, tuple] = {}
        cn, cy = _chain_pool(mods_tr_np, False, n_chain)
        CHAIN["chain_heldout_y"] = (cn, cy, cy % cn)
        cn2, cy2 = _chain_pool(mods_tr_np, True, min(n_chain, 32768))
        CHAIN["chain_seen_y"] = (cn2, cy2, cy2 % cn2)
        if mods_he_np.size:
            cn3, cy3 = _chain_pool(mods_he_np, None, n_chain)
            CHAIN["chain_heldout_n"] = (cn3, cy3, cy3 % cn3)

        floors = {}
        for name, (nvp, xin, _, _) in EVAL.items():
            floors[name] = (xin < nvp).float().mean().item()
        for name, (nvp, xin, _) in CHAIN.items():
            floors[name] = (xin < nvp).float().mean().item()
        for name, (nvp, xin, _, exf) in EVAL.items():
            print(
                f"[pool] {name}: n={nvp.numel()} {'exhaustive' if exf else 'sampled'} "
                f"resolution={1.0/max(1,nvp.numel()):.2e} floor={floors[name]:.4f}",
                flush=True,
            )
        for name, (nvp, xin, _) in CHAIN.items():
            print(
                f"[pool] {name}: n={nvp.numel()} modulus-uniform "
                f"resolution={1.0/max(1,nvp.numel()):.2e} floor={floors[name]:.4f}",
                flush=True,
            )

        # ------------------------------ model ----------------------------------------
        model = _make_model(cfg, arm, max_len, n_ans, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{arm_name}] params={n_params:,} max_len={max_len} n_ans={n_ans} "
              f"stages={n_stages} inner={arm['inner_steps']} "
              f"r_slots=[{r_slot0},{r_slot1})", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / base_cfg["warmup"])
            * (1.0 if const_lr else 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))),
        )

        # ------------------------------ the staged forward ---------------------------
        def _st(p):
            """Straight-through hard: argmax one-hot forward, soft gradient backward."""
            hard = torch.zeros_like(p).scatter_(-1, p.argmax(-1, keepdim=True), 1.0)
            return hard + p - p.detach()

        def stage_logits(n_vals, r_probs, c_vals, mode):
            """One application of the tied stage map on `(N, r·R + c)`.

            `r_probs` is `None` at the first stage (`r = 0`, a constant). Otherwise the
            remainder's digit slots of the `2w` dividend field are overwritten with the
            re-encode; `c`'s slots and the leading zeros stay hard. Because `R = 10^m`,
            `digits_of(r·R + c, 2w)` places `r`'s digits at exactly `[r_slot0, r_slot1)`,
            so this is an in-place, positionally exact re-encode.
            """
            if r_probs is None:
                ids = prompts_for(n_vals, c_vals)
                return model.dec(model.roll(model.enc(ids, read_pos)))
            r_hard = value_of(r_probs.argmax(-1))
            ids = prompts_for(n_vals, r_hard * radix_eff + c_vals)
            emb = model.enc.tok(ids)
            if mode == "hard":
                h = model.enc.forward_emb(emb, read_pos)
                return model.dec(model.roll(h))
            p = r_probs if mode == "soft" else _st(r_probs)
            wd = model.enc.tok.weight[DIGIT_OFFSET:DIGIT_OFFSET + 10]
            emb = torch.cat(
                [emb[:, :r_slot0, :], p @ wd, emb[:, r_slot1:, :]], dim=1
            )
            return model.dec(model.roll(model.enc.forward_emb(emb, read_pos)))

        def staged_forward(n_vals, y_vals, mode, keep_seams=False):
            """`S` tied stage applications over the radix digits of `y`, high to low."""
            r_probs = None
            seams = []
            for i in range(n_stages - 1, -1, -1):
                c = (y_vals // radix_eff**i) % radix_eff
                logits = stage_logits(n_vals, r_probs, c, mode)
                r_probs = logits.softmax(-1)
                if keep_seams and i > 0:
                    seams.append(r_probs)
            return logits, seams

        def seam_penalty(seams, n_vals, y_vals, mode):
            """Label-free consistency at the seams. Uses no labels of any kind."""
            if arm["seam_mode"] == "none" or arm["seam_w"] <= 0 or not seams:
                return torch.zeros((), device=device)
            if arm["seam_mode"] == "ent":
                t = 0.0
                for p in seams:
                    t = t - (p.clamp_min(1e-9).log() * p).sum(-1).mean()
                return t / len(seams)
            if arm["seam_mode"] == "cos":
                # "the remainder you decoded must re-encode to the state you hold": the state
                # reached through the soft re-encode must match the state a clean *hard*
                # re-encode of the same decoded remainder would reach.
                t = 0.0
                wd = model.enc.tok.weight[DIGIT_OFFSET:DIGIT_OFFSET + 10]
                for j, p in enumerate(seams):
                    i = n_stages - 2 - j          # the stage this remainder is consumed by
                    c = (y_vals // radix_eff**i) % radix_eff
                    r_hard = value_of(p.argmax(-1))
                    ids = prompts_for(n_vals, r_hard * radix_eff + c)
                    emb = model.enc.tok(ids)
                    h_hard = model.enc.forward_emb(emb, read_pos)
                    soft_emb = torch.cat(
                        [emb[:, :r_slot0, :], p @ wd, emb[:, r_slot1:, :]], dim=1
                    )
                    h_soft = model.enc.forward_emb(soft_emb, read_pos)
                    t = t + (1.0 - F.cosine_similarity(h_soft, h_hard.detach(), dim=-1)).mean()
                return t / len(seams)
            raise ValueError(arm["seam_mode"])

        # ------------------------------ readouts -------------------------------------
        @torch.no_grad()
        def eval_stage():
            """The tied stage map applied **once** to a bounded-quotient query.

            Directly comparable to the donor's STAGE rows: `sr3_r10` reads 0.9978 here
            *having trained on this distribution*; this arm has never seen it.
            """
            model.eval()
            out = {}
            ch = base_cfg["eval_chunk"]
            for name, (nvp, xin, tgt, _) in EVAL.items():
                err = 0
                for i in range(0, nvp.numel(), ch):
                    s = slice(i, min(i + ch, nvp.numel()))
                    h = model.roll(model.enc(prompts_for(nvp[s], xin[s]), read_pos))
                    ok = (model.dec(h).argmax(-1) == digits_of(tgt[s], n_ans)).all(-1)
                    err += int((~ok).sum())
                eps = err / max(1, nvp.numel())
                out[name] = dict(n=int(nvp.numel()), n_err=err, eps=eps, acc=1.0 - eps,
                                 resolution=1.0 / max(1, nvp.numel()),
                                 certifiable_T=certifiable_T(eps))
            model.train()
            return out

        @torch.no_grad()
        def eval_chain(mode=None, prefix="chain", teacher=False, per_modulus=False):
            """The arm's staged forward on the chain pools.

            `mode=None` uses the arm's native re-grounding; `mode="hard"` scores the pure
            argmax chain (identical to native under `st`, a separate readout under `soft`).
            `teacher=True` injects the true partial remainder at every stage — the donor's
            oracle re-projection, separating the per-stage rate from propagation.
            """
            mode = mode or arm["stage_mode"]
            model.eval()
            out = {}
            ch = base_cfg["eval_chunk"]
            for name, (nvp, yin, tgt) in CHAIN.items():
                err = 0
                st_ok, st_tot, st_seen = 0, 0, 0
                okv = torch.zeros(nvp.numel(), dtype=torch.bool, device=device)
                for i0 in range(0, nvp.numel(), ch):
                    s = slice(i0, min(i0 + ch, nvp.numel()))
                    n_, y_ = nvp[s], yin[s]
                    if teacher:
                        r_last = None
                        for i in range(n_stages - 1, -1, -1):
                            c = (y_ // radix_eff**i) % radix_eff
                            src = (y_ // radix_eff ** (i + 1)) % n_
                            inp = src * radix_eff + c
                            h = model.roll(model.enc(prompts_for(n_, inp), read_pos))
                            pv = value_of(model.dec(h).argmax(-1))
                            st_ok += int((pv == ((y_ // radix_eff**i) % n_)).sum())
                            st_tot += pv.numel()
                            st_seen += int(is_train(n_, inp).sum())
                            r_last = pv
                        good = r_last == (y_ % n_)
                    else:
                        logits, _ = staged_forward(n_, y_, mode)
                        good = (logits.argmax(-1) == digits_of(y_ % n_, n_ans)).all(-1)
                    okv[s] = good
                    err += int((~good).sum())
                eps = err / max(1, nvp.numel())
                rec = dict(n=int(nvp.numel()), n_err=err, eps=eps, acc=1.0 - eps,
                           resolution=1.0 / max(1, nvp.numel()),
                           certifiable_T=certifiable_T(eps), n_stages=n_stages, mode=mode)
                if teacher:
                    rec["stage_acc"] = st_ok / max(1, st_tot)
                    rec["stage_n"] = st_tot
                    rec["stage_seen_frac"] = st_seen / max(1, st_tot)
                    rec["chain_model_acc"] = rec["stage_acc"] ** n_stages
                if per_modulus:
                    uq = torch.unique(nvp)
                    accs = {int(m): float(okv[nvp == m].float().mean()) for m in uq}
                    vals = sorted(accs.values())
                    rec["per_modulus_min"] = vals[0]
                    rec["per_modulus_max"] = vals[-1]
                    rec["per_modulus_median"] = vals[len(vals) // 2]
                    if len(accs) <= 16:
                        rec["per_modulus"] = accs
                out[f"{prefix}_{name.split('chain_')[-1]}" if prefix != "chain" else name] = rec
            model.train()
            return out

        def _purity(key: "np.ndarray", val: "np.ndarray") -> float:
            """`E_key[ max_v count(key,v) ] / n` — how nearly `val` is a function of `key`."""
            if key.size == 0:
                return float("nan")
            order = np.lexsort((val, key))
            k, v = key[order], val[order]
            newpair = np.empty(k.size, dtype=bool)
            newpair[0] = True
            np.logical_or((k[1:] != k[:-1]), (v[1:] != v[:-1]), out=newpair[1:])
            start = np.flatnonzero(newpair)
            cnt = np.diff(np.append(start, k.size))
            pk = k[start]
            newkey = np.empty(pk.size, dtype=bool)
            newkey[0] = True
            np.not_equal(pk[1:], pk[:-1], out=newkey[1:])
            grp = np.cumsum(newkey) - 1
            mx = np.zeros(int(grp[-1]) + 1, dtype=np.int64)
            np.maximum.at(mx, grp, cnt)
            return float(mx.sum() / k.size)

        def _perm_within(nkey: "np.ndarray", d: "np.ndarray", rng) -> "np.ndarray":
            """Permute `d` within each modulus — the null that keeps the per-`N` marginal
            and destroys only the dependence on the true partial remainder."""
            out = d.copy()
            order = np.argsort(nkey, kind="stable")
            k = nkey[order]
            bnd = np.empty(k.size, dtype=bool)
            bnd[0] = True
            np.not_equal(k[1:], k[:-1], out=bnd[1:])
            start = np.flatnonzero(bnd)
            for s, e in zip(start, np.append(start[1:], k.size)):
                idx = order[s:e]
                out[idx] = d[idx][rng.permutation(int(e - s))]
            return out

        @torch.no_grad()
        def stage_probe(pool_names=("chain_heldout_y", "chain_heldout_n"), mode=None):
            """Per-stage diagnostics on the chain pool — the node's structural readout.

            For each stage index `i` (high radix digit first), on the arm's own forward:
              `true_match`   decoded intermediate == `(y // R^i) mod N`
              `in_range`     decoded intermediate < N
              `identity`     decoded intermediate == its predecessor
              `modal_share`  share of the single most common decoded value
              `purity`       decoded is the group mode given `(N, true partial remainder)`
                             — a *relabelled* residue reads high here and low on `true_match`
              `inv_purity`   the true partial remainder is the group mode given `(N, decoded)`
              `n_distinct`   distinct decoded values, against `mean(N)`
            """
            mode = mode or arm["stage_mode"]
            model.eval()
            ch = base_cfg["eval_chunk"]
            out = {}
            for name in pool_names:
                if name not in CHAIN:
                    continue
                nvp, yin, _ = CHAIN[name]
                cap = min(nvp.numel(), 32768)
                nvp, yin = nvp[:cap], yin[:cap]
                dec = [[] for _ in range(n_stages)]
                for i0 in range(0, nvp.numel(), ch):
                    s = slice(i0, min(i0 + ch, nvp.numel()))
                    n_, y_ = nvp[s], yin[s]
                    r_probs = None
                    for j, i in enumerate(range(n_stages - 1, -1, -1)):
                        c = (y_ // radix_eff**i) % radix_eff
                        logits = stage_logits(n_, r_probs, c, mode)
                        r_probs = logits.softmax(-1)
                        dec[j].append(value_of(logits.argmax(-1)))
                nn_ = nvp.cpu().numpy()
                yy_ = yin.cpu().numpy()
                prng = np.random.default_rng(20260822)
                rows = []
                prev = None
                for j, i in enumerate(range(n_stages - 1, -1, -1)):
                    d = torch.cat(dec[j]).cpu().numpy()
                    true_r = (yy_ // radix_eff**i) % nn_
                    vals, cnts = np.unique(d, return_counts=True)
                    # Purity is trivially 1.0 at the first stages (the stage input is a
                    # function of `(N, true partial remainder)` there) and is cheap whenever
                    # the decode is near-constant, so every purity carries its own
                    # within-modulus permutation null.
                    d_null = _perm_within(nn_, d, prng)
                    rows.append(dict(
                        stage=j, digit_index=i,
                        true_match=float((d == true_r).mean()),
                        in_range=float((d < nn_).mean()),
                        identity=float((d == prev).mean()) if prev is not None else None,
                        modal_share=float(cnts.max() / d.size),
                        purity=_purity(nn_ * (10**w) + true_r, d),
                        purity_null=_purity(nn_ * (10**w) + true_r, d_null),
                        inv_purity=_purity(nn_ * (10**w) + d, true_r),
                        inv_purity_null=_purity(nn_ * (10**w) + d_null, true_r),
                        n_distinct=int(vals.size),
                        mean_modulus=float(nn_.mean()),
                    ))
                    prev = d
                out[name] = dict(n=int(nvp.numel()), mode=mode, stages=rows)
            model.train()
            return out

        # ------------------------------ persistence ----------------------------------
        out_dir = Path(DATA_DIR) / "staged_reduce" / "terminal_only" / str(tag)
        ck_dir = out_dir / "ckpt"

        def persist(log, final, final_chain, final_teacher, final_probe, done):
            results["arms"][arm_name] = {
                "n_params": n_params,
                "n_stages": n_stages,
                "radix": arm["radix"],
                "stage_mode": arm["stage_mode"],
                "family": fam_info,
                "n_train_moduli": int(mods_tr_np.size),
                "n_heldout_moduli": int(mods_he_np.size),
                "train_moduli": train_mods,
                "heldout_moduli": held_mods,
                "floor_no_reduction": floors,
                "train_log": log,
                "final": final,
                "final_chain": final_chain,
                "final_chain_teacher": final_teacher,
                "final_stage_probe": final_probe,
                "complete": done,
                "gpu": gpu_name,
                "tf32": tf32,
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"results_seed{seed}.json").write_text(
                json.dumps(results, indent=2, cls=NumpyEncoder)
            )
            if save_ckpt:
                ck_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"state_dict": model.state_dict(), "cfg": cfg, "arm": arm_name,
                     "step": log[-1]["step"] if log else 0},
                    ck_dir / f"{arm_name}_seed{seed}.pt",
                )
            volume.commit()

        # ------------------------------ train ----------------------------------------
        model.train()
        log = []
        last_chain = None
        every = max(1, steps // base_cfg["n_log_points"])
        t_start = time.time()
        t_overhead = 0.0
        for step in range(steps):
            nvb, yb = sample_task(pool_tr, batch_size, True)
            logits, seams = staged_forward(
                nvb, yb, arm["stage_mode"],
                keep_seams=(arm["seam_mode"] != "none" and arm["seam_w"] > 0),
            )
            ce = F.cross_entropy(
                logits.reshape(-1, 10), digits_of(yb % nvb, n_ans).reshape(-1)
            )
            sp = seam_penalty(seams, nvb, yb, arm["stage_mode"])
            loss = ce + arm["seam_w"] * sp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_cfg["grad_clip"])
            opt.step()
            sched.step()

            if (step + 1) % every == 0 or step + 1 == steps:
                t_ov0 = time.time()
                train_sec = t_ov0 - t_start - t_overhead
                sps = (step + 1) / max(1e-9, train_sec)
                ev = eval_stage()
                n_logged = len(log)
                do_chain = (n_logged % 3 == 0) or (step + 1 == steps)
                cv = eval_chain() if do_chain else {}
                if do_chain and arm["stage_mode"] == "soft":
                    cv = {**cv, **eval_chain(mode="hard", prefix="chainhard")}
                row = dict(step=step + 1, ce=float(ce), seam=float(sp),
                           train_sec=train_sec, steps_per_sec=sps)
                for k, v in {**ev, **cv}.items():
                    row[f"{k}.eps"] = v["eps"]
                    row[f"{k}.n_err"] = v["n_err"]
                log.append(row)
                summary = " ".join(f"{k}={v['eps']:.2e}({v['n_err']})" for k, v in ev.items())
                csum = " ".join(f"{k}={v['acc']:.4f}" for k, v in cv.items()) or "-"
                print(f"  [{arm_name}] step {step+1:>8} ce={float(ce):.4f} "
                      f"seam={float(sp):.4f} | stage {summary}"
                      f" | chain {csum} | {sps:.1f} step/s ({train_sec/3600:.2f}h train)",
                      flush=True)
                if cv:
                    last_chain = cv
                persist(log, ev, last_chain, None, None, done=False)
                t_overhead += time.time() - t_ov0

        final = eval_stage()
        final_chain = eval_chain(per_modulus=True)
        if arm["stage_mode"] == "soft":
            final_chain = {**final_chain,
                           **eval_chain(mode="hard", prefix="chainhard", per_modulus=True)}
        final_teacher = eval_chain(mode="hard", teacher=True)
        final_probe = stage_probe()
        train_sec = time.time() - t_start - t_overhead
        print(f"  [{arm_name}] trained {steps} steps in {train_sec/3600:.2f}h on {gpu_name} "
              f"({steps/max(1e-9, train_sec):.1f} step/s, "
              f"{t_overhead/3600:.2f}h instrument excluded)", flush=True)
        for name, v in final.items():
            print(f"  [{arm_name}] STAGE {name}: acc={v['acc']:.6f} eps={v['eps']:.3e} "
                  f"({v['n_err']}/{v['n']}, res {v['resolution']:.1e}) "
                  f"certifiable_T={v['certifiable_T']:.3g}   [floor {floors[name]:.4f}]", flush=True)
        for name, v in final_chain.items():
            key = name.replace("chainhard_", "chain_")
            t = final_teacher.get(key, {})
            print(f"  [{arm_name}] CHAIN {name}: acc={v['acc']:.6f} eps={v['eps']:.3e} "
                  f"({v['n_err']}/{v['n']}, res {v['resolution']:.1e}) S={v['n_stages']} "
                  f"certifiable_T={v['certifiable_T']:.3g} | teacher "
                  f"stage_acc={t.get('stage_acc', float('nan')):.6f} "
                  f"-> chain-model {t.get('chain_model_acc', float('nan')):.4f} "
                  f"(seen-frac {t.get('stage_seen_frac', float('nan')):.3f})"
                  f" | per-mod {v.get('per_modulus_min', float('nan')):.3f}"
                  f"-{v.get('per_modulus_max', float('nan')):.3f}"
                  f"   [floor {floors.get(key, float('nan')):.4f}]", flush=True)
        for name, rec in final_probe.items():
            for r in rec["stages"]:
                print(f"  [{arm_name}] PROBE {name} stage {r['stage']} (digit {r['digit_index']}): "
                      f"true={r['true_match']:.4f} in_range={r['in_range']:.4f} "
                      f"identity={r['identity'] if r['identity'] is None else round(r['identity'],4)} "
                      f"modal={r['modal_share']:.4f} "
                      f"purity={r['purity']:.4f}(null {r['purity_null']:.4f}) "
                      f"inv_purity={r['inv_purity']:.4f}(null {r['inv_purity_null']:.4f}) "
                      f"distinct={r['n_distinct']}", flush=True)
        persist(log, final, final_chain, final_teacher, final_probe, done=True)

    print(f"\n[saved] /staged_reduce/terminal_only/{tag}/results_seed{seed}.json", flush=True)
    return results
