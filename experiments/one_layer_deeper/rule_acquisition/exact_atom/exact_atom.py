"""Exact atom: can the one-step map be made *exactly* right, not merely accurate?

The parent cut ([`../README.md`](../README.md)) measured the one-step map against an
*accuracy* target and found the reduce goes 0.022 -> 0.996 once it is shown its own input
space. This cut re-reads the same object against an **exactness** target, because that is
what the benchmark's Hard tier actually gates on.

**Why the target changes.** Hard's ladder advances only when *every* example at the current
`T` is correct (768 of them, from the 1/768 leaderboard granularity). Combined with
`ballistic_depth` §9's test-time re-projection — which turns a depth-`T` rollout into `T`
independent depth-1 problems at `p^ceil(T/k)` — certifying rung `T` needs

    (1 - eps)^(768 * T) >= ~0.5   <=>   eps <~ 9e-4 / T

so each rung of the ladder is worth exactly one factor of two in the one-step error rate,
and the whole ladder T=1..64 is 64x. Depth is the cheap axis; rung zero is the expensive
one. `certifiable_T(eps)` below is that formula, reported next to every number.

Against that target the parent's numbers read differently:

    sqnomod   0.920  -> eps 8e-2  ...  90x too coarse to certify even T=1
    redmod_q8 0.996  -> eps 4e-3  ...  4.4x too coarse
    redmod_qfull 0.773 -> eps 2.3e-1
    sq (held-out x)  0.019, below floor

Every 1.000 anywhere in this program is on *seen* states. The only exactness we have ever
produced is memorisation-exactness. Three cuts, one shared instrument:

  A. `div`  — does eps of the dense reduce *close* with budget, or asymptote?
     `redmod_q8`'s train half is seen ~1000x over at 150k steps, so its residual 0.4% is an
     optimisation/architecture question, not a data one. If eps asymptotes, the coverage
     account is bad news at scale (samples needed grow as N^2) and no amount of the depth
     machinery matters. Arms also put the *rule* axis under a knob the parent could not
     turn: division is defined for **every** N, so `densen` samples N uniformly over all
     w-digit integers rather than the ~178 three-digit semiprimes the depth task's
     periodicity-margin and semiprime constraints admit.

  B. `sqdiv` — the composed atom, with the dense reduce wired *into* it. The parent's next
     step 2. A dense division auxiliary alone should do nothing, because nothing forces the
     `sq` path to route through the division circuit; the intervention that forces it is
     `ballistic_depth`'s closure constraint applied at the multiply/reduce **seam**:
     require `Enc(SQ, N, x) ~ Enc(DIV, N, x^2)`. If that holds, `sq` *is* a division problem
     and the dense auxiliary's coverage transfers into the stage that was starved. So the
     cut is a 2x2, {division auxiliary} x {seam closure}, and the prediction is that only
     the corner with both moves. Both terms need only `(N, x)` — no trapdoor — so the whole
     construction is legal for a real submission.

  C. `mul`  — why is the "easy half" stuck at 0.920 and flat from step 5k to 150k? A
     converged plateau at 8% error is not a solved multiply; under an exactness gate it
     reads as "did not learn multiplication". The parent drew `x` from the union of eight
     three-digit unit groups: at most 999 distinct inputs, ~500 of them training points.
     This cut samples `x` uniformly over `[0, 10^w)` and sweeps `w` and the training-set
     size independently, so `mul5_sparse` matches `mul3`'s ~500 training inputs across a
     100x larger space. Lookup predicts eps tracks the held-out fraction; an algorithm
     predicts eps tracks the absolute count and is flat in `w`.

**The instrument.** Every readout is exact-match error `eps` on a held-out pool that is
*exhaustively enumerated* wherever the problem space allows it (so there is no sampling
noise and no memorisation gap to hide in), and it is logged ~24 times during training so
the shape of `eps` vs budget is visible rather than inferred from endpoints. Resolution is
1/|pool| and is reported alongside, because a claim about eps ~ 1e-5 cannot be made on a
4096-example pool — which is what the parent cut used.

Usage:
  MODAL_PROFILE=chromatic modal run --detach \\
    one_layer_deeper/rule_acquisition/exact_atom/exact_atom.py::exact_atom \\
    --tag div_ladder --arms "divq8,divq64,divqfull" --steps 600000 --seed 0
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

# Long runs (the `memb_d512L8` convergence extension) need a bigger GPU and a longer
# leash than the 300k-step cuts did. Both are env-driven so the default stays exactly
# what every result in the README was produced under: an L4 with a 12h timeout.
GPU = os.environ.get("OLD_GPU", "L4")
TIMEOUT = int(os.environ.get("OLD_TIMEOUT", 43200))

DIGIT_OFFSET = 7
SQ_TOK = DIGIT_OFFSET + 10       # 17
DIV_TOK = DIGIT_OFFSET + 10 + 1  # 18
VOCAB = DIV_TOK + 1

# Hard's gate: every one of `HARD_EXAMPLES` examples correct at the current rung.
HARD_EXAMPLES = 768


def certifiable_T(eps: float, m: int = HARD_EXAMPLES) -> float:
    """Deepest ladder rung a one-step error rate `eps` could certify at ~50% odds.

    `(1-eps)^(m*T) = 0.5` under re-projection at k=1, i.e. `T = ln2 / (m*eps)`. Reported
    everywhere so an accuracy is never quoted without its consequence for the ladder.
    """
    return float("inf") if eps <= 0 else math.log(2.0) / (m * eps)


BASE_ARM = dict(
    task="div",  # div | mul | sqdiv
    # --- DGP -----------------------------------------------------------------------
    n_digits=3,
    # `div` only: `y ~ U[0, min((q_cap+1)*N, N^2))`, so `q_cap` bounds the quotient. The
    # parent's ladder knob, kept identical so `divq*` are direct continuations of
    # `redmod_q*` with a finer instrument and a longer budget.
    q_cap=0,
    # Fraction of the problem space used for training; the rest is the held-out pool.
    # For `mul` this is the coverage knob: `train_frac * 10^n_digits` distinct inputs.
    train_frac=0.5,
    # 0 -> N is drawn from the semiprime family (the depth task's constraint). >0 -> N is
    # drawn uniformly from *all* w-digit integers >= 2. Division needs neither a
    # factorisation nor a periodicity margin, so its rule axis is ~6x denser than the
    # depth task's; this is the only knob in the program that can manufacture rule
    # coverage rather than value coverage.
    dense_n=False,
    min_margin=20,
    max_moduli=8,
    # --- architecture --------------------------------------------------------------
    d_model=256,
    n_enc_layers=2,
    d_ff=1024,
    d_op_ff=1024,
    inner_steps=1,
    # --- `sqdiv` only: the 2x2 -----------------------------------------------------
    # Weight on a uniformly-sampled `(N, y) -> y mod N` batch trained through the *same*
    # encoder/operator/decoder as the squaring task, tagged with a DIV task token.
    div_aux=0.0,
    # Weight on the seam constraint `Enc(SQ, N, x) ~ Enc(DIV, N, x^2)`. This is
    # `ballistic_depth`'s cycle term moved from the rollout seam to the multiply/reduce
    # seam. Target is detached: the DIV branch is anchored by its own CE, so the SQ branch
    # must come to it and collapse is not available (cf. `ballistic_depth` §10's blind spot).
    seam=0.0,
    seam_warmup=0.2,
    # "mse" is scale-*dependent* and was a mistake: the encoder's final LayerNorm gain is
    # shared by both branches, so shrinking it shrinks `h0` and `h_div` together and drives
    # the MSE to ~0 without ever aligning them. Measured: mse 0.0010 at cos **0.081**. The
    # downstream decode is scale-free (the operator and decoder both re-normalise), so
    # nothing pushes back. "cos" is the scale-free version and is what new arms use.
    seam_metric="mse",
    # Auxiliary head decoding `x^2` off the encoder state. The parent's `auxprod`, null on
    # its own; kept as an available knob and off by default.
    aux_product=0.0,
    # `sqdiv` only. False reproduces the parent cut's split: a *per-modulus* permutation of
    # each unit group. That is contaminated for anything `N`-independent — an `x` held out
    # under modulus A is a trained input under modulus B with probability ~1-0.5^k, and the
    # multiply `x -> x^2` does not depend on `N` at all. It is what made `sqnomod` read
    # 0.920 where a clean split reads 0.068. True splits on `x` alone, so a held-out `x` is
    # unseen under *every* modulus. The 3-digit arms above keep False for comparability with
    # the parent; every new arm sets True.
    global_x_split=False,
)

ARMS: dict[str, dict] = {
    # ---- Cut A: does the reduce's error rate close with budget? --------------------
    "divq8": dict(task="div", q_cap=8),
    "divq64": dict(task="div", q_cap=64),
    "divqfull": dict(task="div", q_cap=10**9),
    # capacity control: if eps asymptotes with steps, is it width/depth-bound?
    "divq64_wide": dict(task="div", q_cap=64, d_model=512, n_enc_layers=4, d_ff=2048, d_op_ff=2048),
    # the rule axis, with the constraint that starved it removed
    "divqfull_densen": dict(task="div", q_cap=10**9, dense_n=True),
    "divq64_densen": dict(task="div", q_cap=64, dense_n=True),

    # ---- Cut B: the composed atom, {div aux} x {seam closure} ----------------------
    # `q_cap` must be full range here: the reduce inside `sq` is handed `x^2`, which spans
    # `[0, N^2)`. Capping the quotient would train the auxiliary on a strictly easier
    # distribution than the one the squaring path actually supplies — the exact mistake
    # that made the parent's `redmod` read as a null.
    "sqpad": dict(task="sqdiv"),
    "sqpad_div": dict(task="sqdiv", q_cap=10**9, div_aux=1.0),
    "sqpad_seam": dict(task="sqdiv", seam=1.0),
    "sqpad_div_seam": dict(task="sqdiv", q_cap=10**9, div_aux=1.0, seam=1.0),
    # the arm the synthesis actually predicts: coverage on *both* starved axes at once
    "sqpad_div_seam_densen": dict(
        task="sqdiv", q_cap=10**9, div_aux=1.0, seam=1.0, dense_n=True
    ),
    # The seam re-run with the scale-free loss. `_mse` variants above are kept verbatim so
    # the failed-installation result stays reproducible rather than being edited away.
    "sqpad_div_seamcos": dict(
        task="sqdiv", q_cap=10**9, div_aux=1.0, seam=1.0, seam_metric="cos"
    ),
    "sqpad_div_seamcos_w10": dict(
        task="sqdiv", q_cap=10**9, div_aux=1.0, seam=10.0, seam_metric="cos"
    ),
    "sqpad_seamcos": dict(task="sqdiv", seam=1.0, seam_metric="cos"),

    # ---- Cut C: is the multiply an algorithm or a small table? ---------------------
    "mul3": dict(task="mul", n_digits=3, train_frac=0.5),
    "mul4": dict(task="mul", n_digits=4, train_frac=0.5),
    "mul5": dict(task="mul", n_digits=5, train_frac=0.5),
    # ~500 training inputs, matched to `mul3`, across a 100x larger space. Lookup and
    # algorithm make opposite predictions here.
    "mul5_sparse": dict(task="mul", n_digits=5, train_frac=0.005),
    "mul4_sparse": dict(task="mul", n_digits=4, train_frac=0.05),

    # ---- Cut D: the memorisation boundary -----------------------------------------
    # Cut C's one mechanism: the only arm that generalised (`mul5`) was the only arm that
    # failed to memorise. That predicts something the whole `rule_acquisition` remedy list
    # gets backwards — *more* capacity should make generalisation **worse**, by putting the
    # training set back inside tabling range. All at w=5 (space 10^5) and matched steps, so
    # capacity and training-set size move one at a time against `mul5` as the shared centre.
    "memb_d128":   dict(task="mul", n_digits=5, d_model=128, d_ff=512,  d_op_ff=512),
    "memb_d512":   dict(task="mul", n_digits=5, d_model=512, d_ff=2048, d_op_ff=2048),
    "memb_d512L8": dict(task="mul", n_digits=5, d_model=512, d_ff=2048, d_op_ff=2048,
                        n_enc_layers=8),
    "memb_n5k":    dict(task="mul", n_digits=5, train_frac=0.05),
    "memb_n20k":   dict(task="mul", n_digits=5, train_frac=0.2),

    # ---- Cut E: the composed atom where the multiply is not memorisable ------------
    # At 3 digits `x` has <1000 values, deep inside tabling range, so that setting cannot
    # produce an algorithmic multiply whatever else is installed. 4-digit moduli give ~5000
    # training `x`, at the boundary Cut C located (between 5k and 50k for 2.1M params).
    # The two `div4_*` arms are not optional: the 4-digit reduce is a harder object than the
    # 3-digit one that reads 0.953, so without them a composed null cannot be assigned to a
    # half. Depth 1 only, so no periodicity margin is at stake.
    "div4_q64":   dict(task="div", n_digits=4, q_cap=64),
    "div4_qfull": dict(task="div", n_digits=4, q_cap=10**9),
    "sq4_plain":  dict(task="sqdiv", n_digits=4, global_x_split=True),
    "sq4_div_seamcos": dict(task="sqdiv", n_digits=4, q_cap=10**9, div_aux=1.0,
                           seam=1.0, seam_metric="cos", global_x_split=True),
    # How much was the per-modulus split flattering the 3-digit composed numbers? Same arm
    # as `sqpad_div_seamcos`, clean split, everything else identical.
    "sqpad_div_seamcos_gx": dict(task="sqdiv", q_cap=10**9, div_aux=1.0, seam=1.0,
                                 seam_metric="cos", global_x_split=True),
}

CFG = dict(
    min_factor=3,
    heldout_mod_fraction=0.2,
    family_seed=45,
    n_heads=4,
    steps=600000,
    batch_size=256,
    lr=3e-4,
    warmup=500,
    weight_decay=0.1,
    grad_clip=1.0,
    const_lr=True,
    # Exhaustive enumeration up to this many (N, y) pairs; past it the held-out pool is
    # sampled at `n_fine`. Both report their own resolution.
    exhaustive_cap=4_000_000,
    n_fine=524_288,
    eval_chunk=4096,
    n_log_points=24,
    seed=0,
    tag="smoke",
)

HASH_M = 1_000_000  # hash-split granularity; train iff h < train_frac * HASH_M


def _make_model(cfg, arm, max_len, n_ans, device, n_aux=0):
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
            self.aux = nn.Linear(d, n_aux * 10) if n_aux else None

        def dec(self, h):
            return self.head(self.dnorm(h)).reshape(h.shape[0], n_ans, 10)

        def aux_logits(self, h0):
            return self.aux(h0).reshape(h0.shape[0], n_aux, 10)

        def roll(self, h, n_task_steps=1):
            for _ in range(n_task_steps * self.inner):
                h = self.op(h)
            return h

    return Model().to(device)


@app.function(volumes={DATA_DIR: volume}, gpu=GPU, timeout=TIMEOUT, memory=32768)
def exact_atom(
    tag: str = "smoke",
    arms: str = "divq64",
    steps: int = 600000,
    seed: int = 0,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    n_fine: int = 524288,
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
    # Off by default: every number in the README was produced in strict fp32, and TF32
    # changes matmul mantissas, so a run that flips it is no longer bit-comparable to
    # them. Opt in only where the throughput is what makes the run possible at all.
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Read from the device, not from `GPU`: the module-level constant is set by an env var
    # that exists on the *launching* machine, and the container re-imports without it.
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
        task, w = arm["task"], arm["n_digits"]
        print(f"\n===== arm={arm_name} seed={seed} spec={ARMS[arm_name]} =====", flush=True)

        # ------------------------------ the modulus set ------------------------------
        # `sqdiv` and family-mode `div` need the semiprime family (the squaring task is
        # only defined on it). `mul`'s target does not depend on `N` at all, so its N field
        # is a nuisance drawn from arbitrary w-digit integers — which also avoids the
        # O(pi(10^w)^2) cost of enumerating a 5-digit semiprime family.
        rng = np.random.default_rng(base_cfg["family_seed"])
        fam_info: dict = {}
        if task == "mul":
            train_mods = sorted(
                rng.choice(np.arange(10 ** (w - 1), 10**w), 8, replace=False).tolist()
            )
            held_mods, unit_pools = [], None
        else:
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
            train_mods = [m[0] for m in fam_train]
            held_mods = [m[0] for m in fam_held]
            # Per-modulus unit pools, split train-x / held-out-x exactly as the parent cut
            # does, so `sqpad` is a direct replicate of `sq` (held-out x 0.019).
            urng = np.random.default_rng(base_cfg["family_seed"])
            unit_pools = {}
            thr_x = int(round(arm["train_frac"] * HASH_M))
            for n_val, p, q, _ in fam_train + fam_held:
                u = ModulusFamily.units_for(p, q)
                if arm["global_x_split"]:
                    h = (u.astype(np.int64) * 2654435761) % HASH_M
                    unit_pools[n_val] = (np.sort(u[h < thr_x]), np.sort(u[h >= thr_x]))
                else:
                    perm = urng.permutation(u.size)
                    n_te = int(round((1.0 - arm["train_frac"]) * u.size))
                    unit_pools[n_val] = (
                        np.sort(u[perm[n_te:]]),  # train x
                        np.sort(u[perm[:n_te]]),  # held-out x
                    )

        # The dense-N pool: every w-digit integer >= 2 is a legal divisor. Only reachable
        # because the reduce needs no factorisation and no depth-periodicity margin, which
        # is what caps the depth task's rule axis at ~178 three-digit semiprimes.
        #
        # It carries its own held-out slice, and that slice is a *superset* of the
        # semiprime family's held-out moduli. Without that exclusion a `densen` arm would
        # train the reduce on precisely the moduli its `heldout_n` readout scores, and the
        # rule-axis claim these arms exist to make would be vacuous.
        dense_all = np.arange(max(2, 10 ** (w - 1)), 10**w, dtype=np.int64)
        dense_held_mask = (
            ((dense_all * 2246822519) % 10 == 0) | np.isin(dense_all, held_mods)
        ) & ~np.isin(dense_all, train_mods)  # the seam needs SQ's own moduli trained
        dense_train = dense_all[~dense_held_mask]
        dense_held = dense_all[dense_held_mask]

        train_mods_t = torch.tensor(train_mods, dtype=torch.long, device=device)
        dense_pool_t = torch.tensor(dense_train, device=device)
        dense_held_t = torch.tensor(dense_held, device=device)
        print(
            f"[family] {len(train_mods)} train / {len(held_mods)} held moduli"
            + (f", min_repeat={fam_info.get('depth_first_repeat_min')}" if fam_info else "")
            + f" | dense-N {dense_train.size} train / {dense_held.size} held",
            flush=True,
        )

        # ------------------------------ prompt layout --------------------------------
        # BOS <TASK> N <w digits> X <w_x digits> ANS
        w_x = 2 * w if task in ("div", "sqdiv") else w
        n_ans = 2 * w if task == "mul" else w
        n_aux = 2 * w if arm["aux_product"] > 0 else 0
        max_len = 1 + 1 + (1 + w) + (1 + w_x) + 1
        read_pos = max_len - 1

        def digits_of(v, width):
            return torch.stack([(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1)

        def prompts_for(task_tok, n_vals, x_vals):
            rows = x_vals.shape[0]
            col = lambda t: torch.full((rows, 1), t, dtype=torch.long, device=device)
            return torch.cat(
                [
                    col(TOKEN_IDS["BOS"]),
                    col(task_tok) if isinstance(task_tok, int) else task_tok.view(rows, 1),
                    col(TOKEN_IDS["N"]),
                    digits_of(n_vals, w) + DIGIT_OFFSET,
                    col(TOKEN_IDS["X"]),
                    digits_of(x_vals, w_x) + DIGIT_OFFSET,
                    col(TOKEN_IDS["ANS"]),
                ],
                dim=1,
            )

        # ------------------------------ the hash split -------------------------------
        # An arbitrary hash of the *problem*, not of any learnable predicate, so dense
        # sampling can never make the held-out half leak. `mul`'s target is independent of
        # `N`, so its split keys on `x` alone.
        thr = int(round(arm["train_frac"] * HASH_M))

        def is_train(n_vals, x_vals):
            h = (x_vals * 2654435761 + (0 if task == "mul" else n_vals * 40503)) % HASH_M
            return h < thr

        def hi_for(n_vals):
            """Upper bound of the reduce's input range for each modulus."""
            return torch.minimum((arm["q_cap"] + 1) * n_vals, n_vals * n_vals)

        # ------------------------------ samplers -------------------------------------
        def sample_n(pool, n, gen=None):
            return pool[torch.randint(0, pool.numel(), (n,), device=device, generator=gen)]

        def sample_div(n_pool, n, want_train, gen=None):
            """Uniform `(N, y)` on the requested half of the hash split."""
            got_n, got_y, have = [], [], 0
            for _ in range(64):
                nv = sample_n(n_pool, 4 * n, gen)
                hi = hi_for(nv)
                # float64: at `q_cap=inf` the range is N^2 ~ 1e6 and float32's 24-bit
                # mantissa would quantise the draw, biasing which residues are reachable.
                u = torch.rand(4 * n, device=device, generator=gen, dtype=torch.float64)
                y = (u * hi).long()
                y = torch.minimum(y, hi - 1)
                keep = is_train(nv, y) == want_train
                got_n.append(nv[keep])
                got_y.append(y[keep])
                have += int(keep.sum())
                if have >= n:
                    break
            return torch.cat(got_n)[:n], torch.cat(got_y)[:n]

        def sample_mul(n, want_train, gen=None):
            got_x, have = [], 0
            for _ in range(64):
                x = torch.randint(0, 10**w, (4 * n,), device=device, generator=gen)
                keep = is_train(None, x) == want_train
                got_x.append(x[keep])
                have += int(keep.sum())
                if have >= n:
                    break
            x = torch.cat(got_x)[:n]
            return sample_n(train_mods_t, x.numel(), gen), x

        # For `sqdiv`, `x` is drawn from the modulus's unit group and split by the parent
        # cut's permutation, not by the hash — so the number is comparable to `sq`'s.
        if task == "sqdiv":
            def _pad(which):
                pools = [unit_pools[n][which] for n in train_mods + held_mods]
                width = max(a.size for a in pools)
                out = np.zeros((len(pools), width), dtype=np.int64)
                cnt = np.zeros(len(pools), dtype=np.int64)
                for i, a in enumerate(pools):
                    out[i, : a.size] = a
                    cnt[i] = a.size
                return torch.tensor(out, device=device), torch.tensor(cnt, device=device)

            units_tr, cnt_tr = _pad(0)
            units_te, cnt_te = _pad(1)
            all_mods_t = torch.tensor(train_mods + held_mods, dtype=torch.long, device=device)
            n_train_mod = len(train_mods)

            def sample_sq(mod_lo, mod_hi, units, counts, n, gen=None):
                mi = torch.randint(mod_lo, mod_hi, (n,), device=device, generator=gen)
                c = counts[mi]
                ui = (torch.rand(n, device=device, generator=gen) * c).long().clamp_max_(c - 1)
                return all_mods_t[mi], units[mi, ui]

        # ------------------------------ eval pools -----------------------------------
        # Exhaustive wherever the space allows, so eps carries no sampling noise and the
        # resolution is exactly 1/|pool|. Resolution is reported with every number.
        ev_gen = torch.Generator(device=device).manual_seed(12345)
        EVAL: dict[str, tuple] = {}

        def _div_pool(n_pool_np, want_train, cap):
            space = int(sum(min((arm["q_cap"] + 1) * int(n), int(n) * int(n)) for n in n_pool_np))
            if space <= base_cfg["exhaustive_cap"]:
                ns, ys = [], []
                for n_val in n_pool_np:
                    hi = min((arm["q_cap"] + 1) * int(n_val), int(n_val) ** 2)
                    y = torch.arange(hi, device=device, dtype=torch.long)
                    ns.append(torch.full_like(y, int(n_val)))
                    ys.append(y)
                nv, y = torch.cat(ns), torch.cat(ys)
                keep = is_train(nv, y) == want_train
                nv, y = nv[keep], y[keep]
                if nv.numel() <= cap:
                    return nv, y, True
                # Sub-sample *randomly*, never by truncation: the enumeration is ordered
                # by modulus and then by `y`, so a prefix would be a pool of small
                # dividends on the first few moduli.
                pick = torch.randperm(nv.numel(), generator=ev_gen, device=device)[:cap]
                return nv[pick], y[pick], False
            nv, y = sample_div(
                torch.tensor(n_pool_np, dtype=torch.long, device=device), cap, want_train, ev_gen
            )
            return nv, y, False

        if task == "div":
            # Two independent OOD axes, exactly the pair the parent cut separates:
            # `heldout_y` is a new problem on a trained modulus, `heldout_n` is a new
            # modulus. `seen_y` is the ID control, because §10's blind-spot lesson is that
            # no OOD readout is interpretable without it.
            n_tr = dense_train if arm["dense_n"] else np.array(train_mods, dtype=np.int64)
            n_he = dense_held if arm["dense_n"] else np.array(held_mods, dtype=np.int64)
            nv, y, ex = _div_pool(n_tr, False, n_fine)
            EVAL["heldout_y"] = (DIV_TOK, nv, y, y % nv, ex)
            nv2, y2, ex2 = _div_pool(n_tr, True, min(n_fine, 65536))
            EVAL["seen_y"] = (DIV_TOK, nv2, y2, y2 % nv2, ex2)
            if n_he.size:
                nvh, yh, exh = _div_pool(n_he, True, n_fine)
                EVAL["heldout_n"] = (DIV_TOK, nvh, yh, yh % nvh, exh)

        elif task == "mul":
            x_all = torch.arange(10**w, device=device, dtype=torch.long)
            for name, want in (("heldout_x", False), ("seen_x", True)):
                x = x_all[is_train(None, x_all) == want]
                if x.numel() > n_fine:
                    x = x[torch.randperm(x.numel(), generator=ev_gen, device=device)[:n_fine]]
                nv = sample_n(train_mods_t, x.numel(), ev_gen)
                EVAL[name] = (SQ_TOK, nv, x, x * x, x.numel() <= n_fine)

        else:  # sqdiv
            def _all_units(which, mod_lo, mod_hi):
                ns, xs = [], []
                for i in range(mod_lo, mod_hi):
                    n_val = (train_mods + held_mods)[i]
                    u = torch.tensor(unit_pools[n_val][which], device=device)
                    ns.append(torch.full_like(u, n_val))
                    xs.append(u)
                return torch.cat(ns), torch.cat(xs)

            for name, which, lo_, hi_ in (
                ("seen_x", 0, 0, n_train_mod),
                ("heldout_x", 1, 0, n_train_mod),
                ("heldout_n", 0, n_train_mod, len(train_mods + held_mods)),
            ):
                if hi_ <= lo_:
                    continue
                nv, x = _all_units(which, lo_, hi_)
                EVAL[name] = (SQ_TOK, nv, x, (x * x) % nv, True)
            # The reduce's own readout, so a composed number can be decomposed post hoc
            # into (multiply quality) x (reduce quality) rather than only reported.
            if arm["div_aux"] > 0:
                n_tr = dense_train if arm["dense_n"] else np.array(train_mods, dtype=np.int64)
                nvd, yd, exd = _div_pool(n_tr, False, min(n_fine, 131072))
                EVAL["div_heldout_y"] = (DIV_TOK, nvd, yd, yd % nvd, exd)

        # The floor every held-out number must beat: examples answerable with no knowledge
        # of `N` at all. `mul` has no reduction to skip, so its floor is 0 by construction.
        floors = {}
        for name, (_, nv, xin, _, _) in EVAL.items():
            if task == "mul":
                free = torch.zeros_like(xin, dtype=torch.bool)
            elif task == "div" or name.startswith("div_"):
                free = xin < nv
            else:
                free = (xin * xin) < nv
            floors[name] = free.float().mean().item()
        for name, (_, nv, xin, _, ex) in EVAL.items():
            print(
                f"[pool] {name}: n={nv.numel()} {'exhaustive' if ex else 'sampled'} "
                f"resolution={1.0/max(1,nv.numel()):.2e} floor={floors[name]:.4f}",
                flush=True,
            )

        # ------------------------------ model ----------------------------------------
        model = _make_model(cfg, arm, max_len, n_ans, device, n_aux=n_aux)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{arm_name}] params={n_params:,} max_len={max_len} n_ans={n_ans}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / base_cfg["warmup"])
            * (1.0 if const_lr else 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))),
        )

        @torch.no_grad()
        def eval_fine(cap=None):
            """Exact-match error on every eval pool, in chunks. The whole instrument."""
            model.eval()
            out = {}
            ch = base_cfg["eval_chunk"]
            for name, (tok, nv, xin, tgt, _) in EVAL.items():
                n_tot = nv.numel() if cap is None else min(cap, nv.numel())
                err = 0
                for i in range(0, n_tot, ch):
                    s = slice(i, min(i + ch, n_tot))
                    h = model.roll(model.enc(prompts_for(tok, nv[s], xin[s]), read_pos))
                    ok = (model.dec(h).argmax(-1) == digits_of(tgt[s], n_ans)).all(-1)
                    err += int((~ok).sum())
                eps = err / max(1, n_tot)
                out[name] = dict(
                    n=n_tot, n_err=err, eps=eps, acc=1.0 - eps,
                    resolution=1.0 / max(1, n_tot), certifiable_T=certifiable_T(eps),
                )
            model.train()
            return out

        # ------------------------------ persistence ----------------------------------
        # Written at *every* log point, not only at the end. A multi-million-step run is
        # hours long, and the 300k-step cuts' all-or-nothing write would throw the whole
        # trajectory away on a preemption or a timeout.
        out_dir = Path(DATA_DIR) / "exact_atom" / str(tag)
        ck_dir = out_dir / "ckpt"

        def persist(log, final, done):
            results["arms"][arm_name] = {
                "n_params": n_params,
                "family": fam_info,
                "n_train_moduli": len(train_mods),
                "train_moduli": train_mods,
                "heldout_moduli": held_mods,
                "floor_no_reduction": floors,
                "train_log": log,
                "final": final,
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
        every = max(1, steps // base_cfg["n_log_points"])
        # Wall-clock, with the instrument's own cost held out of it: the benchmark meters
        # *training* seconds, and `eval_fine` on exhaustive pools is ours, not its.
        t_start = time.time()
        t_overhead = 0.0
        for step in range(steps):
            parts = {}
            if task == "div":
                pool = dense_pool_t if arm["dense_n"] else train_mods_t
                nv, y = sample_div(pool, batch_size, True)
                h = model.roll(model.enc(prompts_for(DIV_TOK, nv, y), read_pos))
                loss = F.cross_entropy(
                    model.dec(h).reshape(-1, 10), digits_of(y % nv, n_ans).reshape(-1)
                )
            elif task == "mul":
                nv, x = sample_mul(batch_size, True)
                h = model.roll(model.enc(prompts_for(SQ_TOK, nv, x), read_pos))
                loss = F.cross_entropy(
                    model.dec(h).reshape(-1, 10), digits_of(x * x, n_ans).reshape(-1)
                )
            else:  # sqdiv
                nv, x = sample_sq(0, n_train_mod, units_tr, cnt_tr, batch_size)
                h0 = model.enc(prompts_for(SQ_TOK, nv, x), read_pos)
                h1 = model.roll(h0)
                loss = F.cross_entropy(
                    model.dec(h1).reshape(-1, 10), digits_of((x * x) % nv, n_ans).reshape(-1)
                )
                parts["ce_sq"] = float(loss)
                if arm["div_aux"] > 0:
                    pool = dense_pool_t if arm["dense_n"] else train_mods_t
                    dnv, dy = sample_div(pool, batch_size, True)
                    dh = model.roll(model.enc(prompts_for(DIV_TOK, dnv, dy), read_pos))
                    ce_div = F.cross_entropy(
                        model.dec(dh).reshape(-1, 10), digits_of(dy % dnv, n_ans).reshape(-1)
                    )
                    loss = loss + arm["div_aux"] * ce_div
                    parts["ce_div"] = float(ce_div)
                if arm["seam"] > 0 and step >= arm["seam_warmup"] * steps:
                    # The seam: the state the squaring path hands to the operator must be
                    # the state the *division* path would hand it for the same product.
                    # Detached target — the DIV branch is anchored by its own CE, so this
                    # cannot be satisfied by collapsing both sides.
                    with torch.no_grad():
                        h_div = model.enc(prompts_for(DIV_TOK, nv, x * x), read_pos)
                    seam = (
                        (1.0 - F.cosine_similarity(h0, h_div, dim=-1)).mean()
                        if arm["seam_metric"] == "cos"
                        else F.mse_loss(h0, h_div)
                    )
                    loss = loss + arm["seam"] * seam
                    parts["seam"] = float(seam)
                if n_aux:
                    loss = loss + arm["aux_product"] * F.cross_entropy(
                        model.aux_logits(h0).reshape(-1, 10), digits_of(x * x, n_aux).reshape(-1)
                    )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_cfg["grad_clip"])
            opt.step()
            sched.step()

            if (step + 1) % every == 0 or step + 1 == steps:
                # eps vs budget is the object of Cut A, so it is tracked *during* training
                # on the full-resolution pools, not inferred from two endpoints.
                # Everything from here to `persist` is instrument, not training: the
                # exhaustive eval and the 110MB checkpoint write are ours, and the
                # benchmark's budget is training seconds. Held out of `train_sec` so the
                # rate stays convertible into tier units.
                t_ov0 = time.time()
                train_sec = t_ov0 - t_start - t_overhead
                sps = (step + 1) / max(1e-9, train_sec)
                ev = eval_fine()
                row = dict(step=step + 1, ce=float(loss),
                           train_sec=train_sec, steps_per_sec=sps, **parts)
                for k, v in ev.items():
                    row[f"{k}.eps"] = v["eps"]
                    row[f"{k}.n_err"] = v["n_err"]
                log.append(row)
                summary = " ".join(
                    f"{k}={v['eps']:.2e}({v['n_err']})" for k, v in ev.items()
                )
                print(f"  [{arm_name}] step {step+1:>8} ce={float(loss):.4f} | eps {summary}"
                      f" | {sps:.1f} step/s ({train_sec/3600:.2f}h train)", flush=True)
                persist(log, ev, done=False)
                t_overhead += time.time() - t_ov0

        final = eval_fine()
        train_sec = time.time() - t_start - t_overhead
        print(f"  [{arm_name}] trained {steps} steps in {train_sec/3600:.2f}h on {gpu_name} "
              f"({steps/max(1e-9, train_sec):.1f} step/s, "
              f"{t_overhead/3600:.2f}h instrument excluded)", flush=True)
        for name, v in final.items():
            print(
                f"  [{arm_name}] {name}: acc={v['acc']:.6f} eps={v['eps']:.3e} "
                f"({v['n_err']}/{v['n']}, res {v['resolution']:.1e}) "
                f"certifiable_T={v['certifiable_T']:.3g}   [floor {floors[name]:.4f}]",
                flush=True,
            )
        persist(log, final, done=True)

    print(f"\n[saved] /exact_atom/{tag}/results_seed{seed}.json", flush=True)
    return results
