"""Staged long division: decompose the reduce until every stage's input space is coverable.

`exact_atom` §5 localised the reduce's failure to the **quotient range**. Bounded quotient
generalises (`divq64` eps 2.4e-3 at 3 digits, `div4_q64` 0.986 at 4 digits with no
memorisation gap); full quotient range collapses (`divqfull` ~0.48 modulus-uniform,
`div4_qfull` 0.0004). And `x^2` for `x < N` spans the full quotient range by construction,
so the composed atom cannot dodge the hard case.

Schoolbook long division is the decomposition that fact implies. Writing `y` in radix `R`
as `c_{S-1} ... c_0` and carrying a remainder `r < N`:

    r <- 0;   for i = S-1 .. 0:   r <- (r * R + c_i) mod N

Every stage is a reduce with dividend `< R * N`, i.e. **quotient <= R - 1** — precisely the
bounded-quotient problem we already know generalises. The stage's input space is `R * N`
per modulus instead of `N^2`, a factor `N/R` smaller: at `R = 10`, `w = 3` that is 100x, and
the whole stage space (8 moduli x 10N ~ 4e4 pairs) is small enough to enumerate.

This is [`ballistic_depth`](../../ballistic_depth/README.md) §9's move one level down. There,
a depth-`T` rollout became `T` independent depth-1 problems at test time by re-grounding the
state through the model's own decode, giving `accuracy(T) ~ p^ceil(T/k)`. Here a full-range
reduce becomes `S` independent bounded-quotient problems by re-grounding the *remainder*
through the model's own decode — and because the remainder is a digit string, the snap is
exact rather than a projection onto a learned manifold.

Two hypotheses, both pre-registered.

  **H1 (coverage).** Chaining `S` bounded-quotient stages beats a monolithic full-range
  reduce at matched width, budget, architecture and parameters. Quantitatively the chain
  model predicts `acc_chain ~ p_stage^S`, so radix trades stage count against per-stage
  difficulty and there should be an interior optimum, exactly as §9's re-projection period
  `k` did.

  **H2 (rule).** Monolithic divide-by-`N` is a different function for every modulus, which
  is presumably why 8 moduli give 8 specialised routines (`divq64` held-out `N` 0.012,
  *below* its 0.015 floor). A single-radix-digit stage is closer to "compare against a small
  multiple of `N`, subtract", much more nearly the *same* function across moduli. So staged
  should transfer to held-out `N` substantially better than monolithic. Nothing across three
  cuts has moved that axis.

**Legality.** The radix digits of `y` and every intermediate remainder are computable from
`(N, y)` alone — no factorisation, no `phi(N)`, no trapdoor. Both the training distribution
and the test-time chain are legal for a real submission. (As in every cut in this
experiment: no submission is built, `T` never enters the prompt, answers are fixed-width, so
these are not upstream-comparable scores.)

**Controls.** The monolithic arm is `radix=0`, which runs through the *same* code path with
`S = 1` — same prompt layout, same architecture, same parameter count, same step budget,
same eval pools. `*_inner{S}` applies the tied operator `S` times per query without
re-grounding, so "more serial compute" is separated from "re-grounding at the seam".

**The instrument.** Two families of readout.

  - *stage* readouts (`heldout_y` / `seen_y` / `heldout_n`) are `exact_atom`'s: exact-match
    error on the arm's own bounded-quotient space, exhaustive where it fits.
  - *chain* readouts are the headline and are **identical across arms by construction**:
    `y ~ U[0, N^2)` drawn **modulus-uniform** (matching training's weighting, not the
    exhaustive enumeration's `N^2` weighting — `exact_atom`'s gotcha), split by the same
    hash of `(N, y)`. `chain_heldout_y` is the composite generalisation number,
    `chain_seen_y` its in-distribution control, `chain_heldout_n` the rule axis.
    `chain_teacher` feeds the *true* intermediate remainder at every stage: it is §9's
    oracle re-projection, and it separates the per-stage rate from error propagation.

Usage:
  MODAL_PROFILE=chromatic modal run --detach \\
    one_layer_deeper/rule_acquisition/staged_reduce/staged_reduce.py::staged_reduce \\
    --tag sr3a --arms "sr3_r10,sr3_r100" --steps 300000 --seed 0
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

GPU = os.environ.get("OLD_GPU", "L4")
TIMEOUT = int(os.environ.get("OLD_TIMEOUT", 43200))

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
    # 0 -> monolithic full-range reduce (S=1). >0 -> long division in base `radix`, so each
    # stage is a reduce with quotient <= radix-1 and the chain has S stages. Must be <= 10^w
    # so that `r * radix + c` still fits the 2w-digit dividend field.
    radix=10,
    train_frac=0.5,
    dense_n=False,
    min_margin=20,
    max_moduli=8,
    # --- architecture (identical to `exact_atom`'s BASE_ARM) -------------------------
    d_model=256,
    n_enc_layers=2,
    d_ff=1024,
    d_op_ff=1024,
    # Serial applications of the tied operator per *query*. The compute-matched control for
    # a chain of S stages: same operator applications, no re-grounding between them.
    inner_steps=1,
)

ARMS: dict[str, dict] = {
    # ---- Phase A: does staging beat monolithic at matched everything? (H1) ----------
    # 3 digits: the width where `exact_atom` §4's oracle reduce reads 0.953 and the
    # full-range reduce reads ~0.48 modulus-uniform.
    "sr3_r2": dict(radix=2),        # S=20, quotient <= 1: compare and conditional-subtract
    "sr3_r10": dict(radix=10),      # S=6,  quotient <= 9: schoolbook long division
    "sr3_r100": dict(radix=100),    # S=3,  quotient <= 99
    "sr3_r1000": dict(radix=1000),  # S=2,  quotient <= 999 (defined, not run by default)
    "sr3_mono": dict(radix=0),      # S=1 — the control. Reproduces `divqfull` at this budget.
    "sr3_mono_inner6": dict(radix=0, inner_steps=6),  # compute-matched to sr3_r10

    # 4 digits: `exact_atom` §5's scissors. `div4_qfull` reads 0.0004 and `div4_q64` 0.986,
    # so this is the width where the monolithic reduce fails hardest.
    "sr4_r10": dict(n_digits=4, radix=10),    # S=8
    "sr4_r100": dict(n_digits=4, radix=100),  # S=4
    "sr4_mono": dict(n_digits=4, radix=0),    # S=1 — the control, ~= `div4_qfull`
    "sr4_mono_inner8": dict(n_digits=4, radix=0, inner_steps=8),

    # ---- Phase B: the rule axis (H2) ------------------------------------------------
    # `min_margin=2, max_moduli=0` is the parent cut's `manymod` setting: 142 train moduli.
    # `dense_n` draws `N` from all 900 three-digit integers — legal because the reduce needs
    # neither a factorisation nor a periodicity margin. `exact_atom` §3's `divqfull_densen`
    # never trained (ce 1.64); the conjecture is that widening the *rule* set only failed
    # because it was paired with the full quotient range, which staging removes.
    "sr3_r10_many": dict(radix=10, min_margin=2, max_moduli=0),
    "sr3_mono_many": dict(radix=0, min_margin=2, max_moduli=0),
    # The family-matched 8-modulus control. `sr3_r10` is at floor on held-out `N`, but it
    # draws from the `min_margin=20` family (59 moduli, range 177-995) while `_many` draws
    # from `min_margin=2` (178 moduli, range 111-995) — so the 8-vs-142 contrast confounds
    # modulus *count* with the family. This arm holds the family fixed and varies only the
    # count, which is what the "breadth is the second factor" claim needs.
    "sr3_r10_m8": dict(radix=10, min_margin=2, max_moduli=8),
    "sr3_r10_dense": dict(radix=10, dense_n=True, min_margin=2, max_moduli=0),
    "sr3_r2_dense": dict(radix=2, dense_n=True, min_margin=2, max_moduli=0),
    "sr3_mono_dense": dict(radix=0, dense_n=True, min_margin=2, max_moduli=0),
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
    # Size of each *chain* pool. Resolution 1/n is reported with every number.
    n_chain=65_536,
    eval_chunk=8192,
    n_log_points=24,
    seed=0,
    tag="smoke",
)

HASH_M = 1_000_000


def _make_model(cfg, arm, max_len, n_ans, device):
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

        def forward(self, input_ids, read_pos):
            h = self.tok(input_ids) + self.pos.unsqueeze(0)
            return self.norm(self.body(h))[:, read_pos, :]

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
def staged_reduce(
    tag: str = "smoke",
    arms: str = "sr3_r10",
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
        if arm["radix"] and arm["radix"] > 10**w:
            raise ValueError(f"radix {arm['radix']} > 10^{w}: dividend would not fit 2w digits")
        print(
            f"\n===== arm={arm_name} seed={seed} spec={ARMS[arm_name]} "
            f"radix={arm['radix']} -> {n_stages} stage(s), quotient<="
            f"{'N-1' if arm['radix'] == 0 else arm['radix'] - 1} =====",
            flush=True,
        )

        # ------------------------------ the modulus set ------------------------------
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

        # The dense-`N` pool: every w-digit integer >= 2 is a legal divisor. Its held-out
        # slice is a superset of the semiprime family's, so a `densen` arm never trains on
        # the moduli its `heldout_n` readout scores.
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
            f" | dense-N {dense_train.size} train / {dense_held.size} held"
            f" | training on {mods_tr_np.size} moduli",
            flush=True,
        )

        # ------------------------------ prompt layout --------------------------------
        # BOS DIV N <w digits> X <2w digits> ANS  — identical for every radix, so the
        # monolithic control differs from a staged arm only in what it is fed.
        w_x = 2 * w
        n_ans = w
        max_len = 1 + 1 + (1 + w) + (1 + w_x) + 1
        read_pos = max_len - 1

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
        thr = int(round(arm["train_frac"] * HASH_M))

        def is_train(n_vals, x_vals):
            return ((x_vals * 2654435761 + n_vals * 40503) % HASH_M) < thr

        def hi_for(n_vals):
            """Upper bound of *this arm's* dividend range: `radix * N`, or `N^2` if monolithic.

            No `min(.., N^2)`: at radix 1000 a legitimate stage input `r*1000 + c` exceeds
            `N^2` for the smaller moduli, and capping it would train the stage on a strictly
            narrower range than the chain feeds it.
            """
            if arm["radix"] == 0:
                return n_vals * n_vals
            return arm["radix"] * n_vals

        def hi_int(n: int) -> int:
            return n * n if arm["radix"] == 0 else arm["radix"] * n

        # ------------------------------ samplers -------------------------------------
        def sample_n(pool, n, gen=None):
            return pool[torch.randint(0, pool.numel(), (n,), device=device, generator=gen)]

        def sample_div(n_pool, n, want_train, gen=None):
            got_n, got_y, have = [], [], 0
            for _ in range(64):
                nv = sample_n(n_pool, 4 * n, gen)
                hi = hi_for(nv)
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
        ev_gen = torch.Generator(device=device).manual_seed(12345)
        EVAL: dict[str, tuple] = {}

        def _div_pool(n_pool_np, want_train, cap):
            space = int(sum(hi_int(int(n)) for n in n_pool_np))
            if space <= base_cfg["exhaustive_cap"]:
                ns, ys = [], []
                for n_val in n_pool_np:
                    hi = hi_int(int(n_val))
                    y = torch.arange(hi, device=device, dtype=torch.long)
                    ns.append(torch.full_like(y, int(n_val)))
                    ys.append(y)
                nv, y = torch.cat(ns), torch.cat(ys)
                keep = is_train(nv, y) == want_train
                nv, y = nv[keep], y[keep]
                if nv.numel() <= cap:
                    return nv, y, True
                pick = torch.randperm(nv.numel(), generator=ev_gen, device=device)[:cap]
                return nv[pick], y[pick], False
            nv, y = sample_div(
                torch.tensor(n_pool_np, dtype=torch.long, device=device), cap, want_train, ev_gen
            )
            return nv, y, False

        # stage-level readouts, on the arm's own dividend range
        nv, y, ex = _div_pool(mods_tr_np, False, n_fine)
        EVAL["heldout_y"] = (nv, y, y % nv, ex)
        nv2, y2, ex2 = _div_pool(mods_tr_np, True, min(n_fine, 65536))
        EVAL["seen_y"] = (nv2, y2, y2 % nv2, ex2)
        if mods_he_np.size:
            nvh, yh, exh = _div_pool(mods_he_np, True, n_fine)
            EVAL["heldout_n"] = (nvh, yh, yh % nvh, exh)

        # ------------------------------ chain pools ----------------------------------
        # `y ~ U[0, N^2)`, **modulus-uniform**, split by the same hash of `(N, y)`. Built
        # from a fixed generator against a fixed modulus list, so every arm sharing a
        # modulus set sees a bit-identical pool and the comparison across radices is exact.
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
            # Permute before truncating: the concatenation is ordered by modulus within each
            # round, so a prefix would over-weight the first moduli of the last round.
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
              f"stages={n_stages} inner={arm['inner_steps']}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / base_cfg["warmup"])
            * (1.0 if const_lr else 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))),
        )

        @torch.no_grad()
        def eval_stage():
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
        def eval_chain(teacher=False, per_modulus=False):
            """`S` bounded-quotient stages, re-grounding the remainder through the decode.

            `teacher=True` injects the *true* intermediate remainder at every stage — §9's
            oracle re-projection — so the per-stage rate is separated from propagation.
            """
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
                    r = torch.zeros_like(y_)
                    for i in range(n_stages - 1, -1, -1):
                        c = (y_ // radix_eff**i) % radix_eff
                        src = ((y_ // radix_eff ** (i + 1)) % n_) if teacher else r
                        inp = src * radix_eff + c
                        h = model.roll(model.enc(prompts_for(n_, inp), read_pos))
                        pv = value_of(model.dec(h).argmax(-1))
                        if teacher:
                            st_ok += int((pv == ((y_ // radix_eff**i) % n_)).sum())
                            st_tot += pv.numel()
                            st_seen += int(is_train(n_, inp).sum())
                        r = pv
                    good = r == (y_ % n_)
                    okv[s] = good
                    err += int((~good).sum())
                eps = err / max(1, nvp.numel())
                rec = dict(n=int(nvp.numel()), n_err=err, eps=eps, acc=1.0 - eps,
                           resolution=1.0 / max(1, nvp.numel()),
                           certifiable_T=certifiable_T(eps), n_stages=n_stages)
                if teacher:
                    rec["stage_acc"] = st_ok / max(1, st_tot)
                    rec["stage_n"] = st_tot
                    rec["stage_seen_frac"] = st_seen / max(1, st_tot)
                    # The chain model: independent stages predict acc = stage_acc ** S.
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
                out[name] = rec
            model.train()
            return out

        # ------------------------------ persistence ----------------------------------
        out_dir = Path(DATA_DIR) / "staged_reduce" / str(tag)
        ck_dir = out_dir / "ckpt"

        def persist(log, final, final_chain, final_teacher, done):
            results["arms"][arm_name] = {
                "n_params": n_params,
                "n_stages": n_stages,
                "radix": arm["radix"],
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
            nvb, yb = sample_div(pool_tr, batch_size, True)
            h = model.roll(model.enc(prompts_for(nvb, yb), read_pos))
            loss = F.cross_entropy(
                model.dec(h).reshape(-1, 10), digits_of(yb % nvb, n_ans).reshape(-1)
            )
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
                # The chain costs `S` forward passes per example, so it is logged on a
                # coarser grid than the stage readout — 8 points is enough to see whether it
                # is still closing, which is the only thing the curve is for.
                n_logged = len(log)
                cv = eval_chain() if (n_logged % 3 == 0 or step + 1 == steps) else {}
                row = dict(step=step + 1, ce=float(loss), train_sec=train_sec, steps_per_sec=sps)
                for k, v in {**ev, **cv}.items():
                    row[f"{k}.eps"] = v["eps"]
                    row[f"{k}.n_err"] = v["n_err"]
                log.append(row)
                summary = " ".join(f"{k}={v['eps']:.2e}({v['n_err']})" for k, v in ev.items())
                csum = " ".join(f"{k}={v['acc']:.4f}" for k, v in cv.items()) or "-"
                print(f"  [{arm_name}] step {step+1:>8} ce={float(loss):.4f} | stage {summary}"
                      f" | chain {csum} | {sps:.1f} step/s ({train_sec/3600:.2f}h train)",
                      flush=True)
                if cv:
                    last_chain = cv
                persist(log, ev, last_chain, None, done=False)
                t_overhead += time.time() - t_ov0

        final = eval_stage()
        final_chain = eval_chain(per_modulus=True)
        final_teacher = eval_chain(teacher=True)
        train_sec = time.time() - t_start - t_overhead
        print(f"  [{arm_name}] trained {steps} steps in {train_sec/3600:.2f}h on {gpu_name} "
              f"({steps/max(1e-9, train_sec):.1f} step/s, "
              f"{t_overhead/3600:.2f}h instrument excluded)", flush=True)
        for name, v in final.items():
            print(f"  [{arm_name}] STAGE {name}: acc={v['acc']:.6f} eps={v['eps']:.3e} "
                  f"({v['n_err']}/{v['n']}, res {v['resolution']:.1e}) "
                  f"certifiable_T={v['certifiable_T']:.3g}   [floor {floors[name]:.4f}]", flush=True)
        for name, v in final_chain.items():
            t = final_teacher[name]
            print(f"  [{arm_name}] CHAIN {name}: acc={v['acc']:.6f} eps={v['eps']:.3e} "
                  f"({v['n_err']}/{v['n']}, res {v['resolution']:.1e}) S={v['n_stages']} "
                  f"certifiable_T={v['certifiable_T']:.3g} | teacher stage_acc={t['stage_acc']:.6f} "
                  f"-> chain-model {t['chain_model_acc']:.4f} (seen-frac {t['stage_seen_frac']:.3f})"
                  f" | per-mod {v.get('per_modulus_min', float('nan')):.3f}"
                  f"-{v.get('per_modulus_max', float('nan')):.3f}"
                  f"   [floor {floors[name]:.4f}]", flush=True)
        persist(log, final, final_chain, final_teacher, done=True)

    print(f"\n[saved] /staged_reduce/{tag}/results_seed{seed}.json", flush=True)
    return results
