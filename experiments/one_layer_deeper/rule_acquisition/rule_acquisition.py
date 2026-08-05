"""Rule acquisition: can the *one-step* map `x -> x^2 mod N` be learned so it generalises?

Everything in `ballistic_depth/` and `variable_modulus/` is about composing an operator. This
cut is about the operator itself, because `variable_modulus/` §3 found there is nothing to
compose: held-out `x` and held-out `N` sit at or below the accuracy obtainable with no rule
knowledge at all, and 500k steps under grokking conditions produce no phase transition. Read
at depth 1 that is not a statement about depth — it says the model cannot compute a *single*
modular squaring on an unseen input, so every depth result in this program sits on top of a
memorised atom. The public leaderboard agrees: stuck at `T = 1`, best 6/768.

`ballistic_depth/rule_structure/` then asked which kind of failure that is and answered
**unreachable** rather than **unattractive** — six trained encoders show no group organisation
beyond a permutation null, and an 11.6x larger state set moved the structure *down*. Its
remedy list (representation, compute, pressure) is untested. This cut tests it.

The protocol is fixed across every arm and is the anti-memorisation one: **train at depth 1
only**, with **half of every modulus's bases held out**, so lookup cannot cover the test set.
The readout is held-out-`x` exact match against the *analytic no-reduction floor* — the
fraction of eval examples with `x^2 < N`, which are answerable with no knowledge of `N` at
all — computed on the actual eval pool rather than quoted.

Four families of arm, each a single controlled change from `sq`:

  T (what part of the atom is hard) — the Hilbert move: find the special case.
    sq        x -> x^2 mod N .................. the real atom, and a replicate of the
                                               `variable_modulus` §3 null
    mul2      x -> 2x mod N .................. modular *reduction* alone (one conditional
                                               subtract; no digit convolution)
    sqnomod   x -> x^2 ....................... *multiplication* alone (no reduction). `N` is
                                               still in the prompt so the encoding is
                                               identical and only the target moves.
  R (representation) — `rule_structure`'s first remedy.
    binary    N, x and the answer in base 2 ... in base 2 every partial product is an AND, so
                                               the digit convolution is additive rather than
                                               multiplicative. This is the only order-free
                                               representation change with real teeth here:
                                               the encoder is *bidirectional* over a 9-token
                                               prompt, so digit-order tricks (reversal,
                                               interleaving) are relabelings of free
                                               positional parameters and provably near-vacuous.
    abacus    shared place-value embedding .... ties `N`'s digit-k and `x`'s digit-k to one
                                               significance code instead of two independent
                                               absolute positions. Included precisely because
                                               it is the weak-prior version, so `binary` is
                                               read against something and not against nothing.
  C (compute) — the confound in calling the null "unreachable": 2 encoder layers + 1 MLP is
    very little serial depth for a 3-digit multiply *and* a division.
    enc8      2 -> 8 encoder layers
    inner8    the tied operator applied 8x per task step (serial compute per step, decoupled
              from steps of the recurrence — which is exactly what the benchmark's
              unconstrained depth permits)
    wide4k    d_ff 1024 -> 4096, encoder and operator
  D (pressure) — the capacity-economics remedy, on the axis `rule_structure` §5 could not move.
    manymod   8 -> 142 train moduli. With 8 moduli, eight tables is cheap; the table cost
              grows with the family and the rule cost does not.

  stack       binary + enc8 + inner8 + wide4k + manymod. Run first, with `sq`: if the maximal
              arm also floors, the singles are a bisection of a known null rather than a
              hopeful sweep.

`T` never enters the prompt and no arm trains on an intermediate residue, as in both parent
cuts. Eval runs to depth 2 (one step past training) for free; every family member has a
depth-first-repeat of at least 5, so periodicity is never a shortcut in that range.

Usage:
  MODAL_PROFILE=chromatic modal run --detach \\
    one_layer_deeper/rule_acquisition/rule_acquisition.py::rule_acquisition \\
    --tag cut1 --arms "sq,stack" --steps 150000 --seed 0
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

DIGIT_OFFSET = 7

# Every knob an arm may move. An arm is this dict plus its overrides, so an arm's definition
# below *is* the complete statement of what varies.
BASE_ARM = dict(
    task="sq",  # sq | mul2 | sqnomod | redmod | redmod_u
    # `redmod_u` only: `y` is drawn uniformly from [0, (q_cap+1)*N), so `q_cap` bounds the
    # quotient. The reduce's input space is 2w digits wide, so the `x^2`-matched pool of
    # `redmod` covers ~0.04% of it against multiplication's ~43% — same sample count, three
    # orders of magnitude apart in coverage. This knob separates "division is unreachable"
    # from "division was undersampled", and grades it by how much division is being asked for.
    q_cap=0,
    number_base=10,
    # Auxiliary cross-entropy on the *un-reduced product* `x^2`, decoded from the encoder
    # state before the operator runs. Legal for a benchmark participant — `x^2` needs no
    # trapdoor, only `(N, x)` — and it is not a rollout residue, so it leaks no depth.
    aux_product=0.0,
    abacus=False,
    n_enc_layers=2,
    inner_steps=1,
    d_ff=1024,
    d_op_ff=1024,
    min_margin=20,  # with min_factor=3, n_digits=3 -> 47 train moduli, capped to 8 below
    max_moduli=8,
)

ARMS: dict[str, dict] = {
    # --- T: which half of the atom is unlearnable
    "sq": {},
    "mul2": dict(task="mul2"),
    "sqnomod": dict(task="sqnomod"),
    # The missing cell: `x^2 -> x^2 mod N`, i.e. the *general* reduction (dividing a 2w-digit
    # number by a w-digit one), on exactly the inputs the reduce sees inside `sq`. `mul2`
    # only isolates the easy reduction — `2x < 2N`, so at most one conditional subtract —
    # and therefore cannot distinguish "division is hard" from "the composition is hard".
    "redmod": dict(task="redmod"),
    # The quotient ladder, at fixed 2w input width so only the quotient distribution moves.
    # Dense sampling also removes memorisation as an option: ~8M distinct (N, y) pairs
    # against a 2.1M-parameter model.
    "redmod_q8": dict(task="redmod_u", q_cap=8),
    "redmod_q64": dict(task="redmod_u", q_cap=64),
    "redmod_qfull": dict(task="redmod_u", q_cap=10**9),  # clipped to N-1 per modulus
    # Dense `y` coverage lets the reduce generalise over *problems* but not over *moduli*
    # (`redmod_q64` held-out N = 0.009 on 8 moduli). These add the rule axis back: 142 moduli,
    # so "divide by this particular N" is no longer 8 memorisable routines.
    "redmod_q64_many": dict(task="redmod_u", q_cap=64, min_margin=2, max_moduli=0),
    "redmod_qfull_many": dict(task="redmod_u", q_cap=10**9, min_margin=2, max_moduli=0),
    # Does handing the model the two-stage structure fix it? Same target as `sq`, plus a
    # head that must decode `x^2` from the encoder state, so `enc` multiplies and `op` reduces.
    "auxprod": dict(aux_product=1.0),
    # --- R: representation
    "binary": dict(number_base=2),
    "abacus": dict(abacus=True),
    # --- C: compute
    "enc8": dict(n_enc_layers=8),
    "inner8": dict(inner_steps=8),
    "wide4k": dict(d_ff=4096, d_op_ff=4096),
    # --- D: pressure. min_margin=2 is safe because we only evaluate to depth 2 and the
    #     family's minimum depth-first-repeat at that setting is 5.
    "manymod": dict(min_margin=2, max_moduli=0),
    # --- everything at once
    "stack": dict(
        number_base=2,
        n_enc_layers=8,
        inner_steps=8,
        d_ff=4096,
        d_op_ff=4096,
        min_margin=2,
        max_moduli=0,
    ),
}

CFG = dict(
    n_digits=3,
    min_factor=3,
    heldout_mod_fraction=0.2,
    # Half the bases held out: the anti-memorisation condition. Lookup cannot cover the test
    # set by construction, so held-out `x` measures the rule and nothing else.
    test_x_fraction=0.5,
    family_seed=45,
    eval_max_depth=2,
    d_model=256,
    n_heads=4,
    steps=150000,
    batch_size=256,
    lr=3e-4,
    warmup=500,
    # The prior 500k probe found held-out `x` flat at wd in {0.1, 1.0}; 0.1 is the setting
    # less likely to prevent the map being learned at all in the wider/deeper arms.
    weight_decay=0.1,
    grad_clip=1.0,
    const_lr=True,
    eval_cap=4096,
    seed=0,
    tag="smoke",
)


def _make_model(cfg, arm, max_len, n_ans, sig_idx, device, n_aux=0):
    import torch
    from torch import nn

    d = cfg["d_model"]
    b = arm["number_base"]
    vocab = DIGIT_OFFSET + b

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d)
            self.pos = nn.Parameter(torch.zeros(max_len, d))
            nn.init.normal_(self.pos, std=0.02)
            # Place-value code, shared between the N field and the x field. Index `n_sig`
            # is the "not a digit" slot (BOS / N / X / ANS markers).
            self.abacus = arm["abacus"]
            if self.abacus:
                self.sig = nn.Embedding(int(sig_idx.max()) + 1, d)
                nn.init.normal_(self.sig.weight, std=0.02)
                self.register_buffer("sig_idx", sig_idx)
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
            if self.abacus:
                h = h + self.sig(self.sig_idx).unsqueeze(0)
            return self.norm(self.body(h))[:, read_pos, :]

    class Block(nn.Module):
        """The tied operator, identical in form to both parent cuts."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.net = nn.Sequential(
                nn.Linear(d, arm["d_op_ff"]), nn.GELU(), nn.Linear(arm["d_op_ff"], d)
            )

        def forward(self, h):
            return h + self.net(self.norm(h))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_ans * b)

        def forward(self, h):
            return self.head(self.norm(h)).reshape(h.shape[0], n_ans, b)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder()
            self.op = Block()
            self.dec = Decoder()
            self.inner = arm["inner_steps"]
            # Reads the un-reduced product off the *encoder* state, so the two stages of the
            # atom land on the two stages of the architecture: enc multiplies, op reduces.
            self.aux = nn.Linear(d, n_aux * b) if n_aux else None

        def aux_logits(self, h0):
            return self.aux(h0).reshape(h0.shape[0], n_aux, b)

        def roll(self, h, n_task_steps):
            """`inner_steps` applications of the tied block per step of the task recurrence."""
            for _ in range(n_task_steps * self.inner):
                h = self.op(h)
            return h

    return Model().to(device)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=16384)
def rule_acquisition(
    tag: str = "smoke",
    arms: str = "sq,stack",
    steps: int = 150000,
    seed: int = 0,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    test_x_fraction: float = 0.5,
    eval_cap: int = 4096,
    n_digits: int = 3,
    min_factor: int = 3,
    const_lr: bool = True,
    group_probe: bool = True,
    save_ckpt: bool = True,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from one_layer_deeper.ballistic_depth.rule_structure.dlog_probe import (
        _fourier_conc,
        _translation_r2,
        group_coords,
    )
    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = {
        **CFG,
        "tag": tag,
        "arms": arms,
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "test_x_fraction": test_x_fraction,
        "eval_cap": eval_cap,
        "n_digits": n_digits,
        "min_factor": min_factor,
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
        print(f"\n===== arm={arm_name} seed={seed} spec={ARMS[arm_name]} =====", flush=True)

        # ----------------------------- the DGP -------------------------------------
        family = ModulusFamily(
            n_digits=n_digits,
            min_margin=arm["min_margin"],
            min_factor=min_factor,
            heldout_fraction=base_cfg["heldout_mod_fraction"],
            seed=base_cfg["family_seed"],
        )
        fam_train, fam_held = family.split()
        fam_info = family.describe()
        if fam_info["depth_first_repeat_min"] <= base_cfg["eval_max_depth"]:
            raise ValueError(
                f"depth periodicity leaks into the eval range "
                f"({fam_info['depth_first_repeat_min']} <= {base_cfg['eval_max_depth']})"
            )
        if arm["max_moduli"] and len(fam_train) > arm["max_moduli"]:
            pick = np.random.default_rng(base_cfg["family_seed"] + 1).choice(
                len(fam_train), arm["max_moduli"], replace=False
            )
            fam_train = sorted(fam_train[i] for i in pick)

        members = fam_train + fam_held
        mod_vals = np.array([m[0] for m in members], dtype=np.int64)
        n_train_mod = len(fam_train)
        max_mod = int(mod_vals.max())

        # Only `sq` and `mul2` are recurrences; the decomposition arms are single maps and
        # are scored at depth 1 only. The recurrences get one free step past training.
        recurrent = arm["task"] in ("sq", "mul2")
        eval_depths = list(range(1, base_cfg["eval_max_depth"] + 1)) if recurrent else [1]

        # --- widths, in the arm's number base --------------------------------------
        b = arm["number_base"]
        w = max(1, int(math.ceil(math.log(max_mod + 1, b) - 1e-12)))
        # The reduce arms are *given* a 2w-digit number. Width is held at 2w across the whole
        # quotient ladder so the representation is constant and only the quotient range moves.
        w_x = 2 * w if arm["task"] in ("redmod", "redmod_u") else w
        n_ans = 2 * w if arm["task"] == "sqnomod" else w
        n_aux = 2 * w if arm["aux_product"] > 0 else 0
        fam_info.update(
            n_train_moduli=n_train_mod,
            train_moduli=[m[0] for m in fam_train],
            reachable_states=sum((p - 1) * (q - 1) // 4 for _, p, q, _ in fam_train),
            number_base=b,
            digit_width=w,
            input_digit_width=w_x,
            n_answer_digits=n_ans,
            n_aux_digits=n_aux,
        )
        print(
            f"[family] {n_train_mod} train / {len(fam_held)} held moduli, "
            f"N in [{mod_vals.min()}, {max_mod}], min_repeat={fam_info['depth_first_repeat_min']}, "
            f"reachable={fam_info['reachable_states']} | base={b} width={w} n_ans={n_ans}",
            flush=True,
        )

        # --- per-modulus base pools, split train-x / held-out-x ---------------------
        rng = np.random.default_rng(base_cfg["family_seed"])
        tr_pools, te_pools = [], []
        for _, p, q, _ in members:
            u = ModulusFamily.units_for(p, q)
            perm = rng.permutation(u.size)
            n_te = int(round(test_x_fraction * u.size))
            te_pools.append(np.sort(u[perm[:n_te]]))
            tr_pools.append(np.sort(u[perm[n_te:]]))

        def _pad(pools):
            width = max(a.size for a in pools)
            out = np.zeros((len(pools), width), dtype=np.int64)
            cnt = np.zeros(len(pools), dtype=np.int64)
            for i, a in enumerate(pools):
                out[i, : a.size] = a
                cnt[i] = a.size
            return torch.tensor(out, device=device), torch.tensor(cnt, device=device)

        units_tr, cnt_tr = _pad(tr_pools)
        units_te, cnt_te = _pad(te_pools)
        mods_t = torch.tensor(mod_vals, device=device)
        train_mod_idx = torch.arange(n_train_mod, device=device)
        held_mod_idx = torch.arange(n_train_mod, len(members), device=device)
        print(
            f"[data] {int(cnt_tr[:n_train_mod].sum())} train pairs, "
            f"{int(cnt_te[:n_train_mod].sum())} held-out-x pairs on train moduli",
            flush=True,
        )

        # --- prompt layout: BOS N <w digits> X <w digits> ANS -----------------------
        max_len = 1 + (1 + w) + (1 + w_x) + 1
        read_pos = max_len - 1
        # Significance index per position: place value (0 = least significant) for digit
        # slots, and a dedicated "not a digit" slot for the markers.
        sig = np.full(max_len, max(w, w_x), dtype=np.int64)
        for k in range(w):
            sig[2 + k] = w - 1 - k  # N's digits, most significant first
        for k in range(w_x):
            sig[3 + w + k] = w_x - 1 - k  # the input field's digits
        sig_idx = torch.tensor(sig, device=device)

        def digits_of(v, width):
            return torch.stack(
                [(v // (b**k)) % b for k in range(width - 1, -1, -1)], dim=1
            )

        def prompts_for(n_vals, x_vals):
            n_rows = x_vals.shape[0]
            col = lambda t: torch.full((n_rows, 1), TOKEN_IDS[t], dtype=torch.long, device=device)
            return torch.cat(
                [
                    col("BOS"),
                    col("N"),
                    digits_of(n_vals, w) + DIGIT_OFFSET,
                    col("X"),
                    digits_of(x_vals, w_x) + DIGIT_OFFSET,
                    col("ANS"),
                ],
                dim=1,
            )

        def io_pairs(n_vals, x0, max_d):
            """`(value in the prompt's x field, {depth: target value})`. Exact in int64.

            The decomposition arms differ from `sq` only here: `sqnomod` keeps the input and
            drops the reduction, `redmod` keeps the reduction and is handed the product. So
            `sqnomod` then `redmod` composes to exactly `sq`, on the same base split.
            """
            if arm["task"] == "sqnomod":
                return x0, {1: x0 * x0}
            if arm["task"] == "redmod":
                y = x0 * x0
                return y, {1: y % n_vals}
            out, x = {}, x0
            for d in range(1, max_d + 1):
                x = (x * x) % n_vals if arm["task"] == "sq" else (2 * x) % n_vals
                out[d] = x
            return x0, out

        def sample_pairs(pool, units, counts, n, generator=None):
            mi = pool[torch.randint(0, pool.numel(), (n,), device=device, generator=generator)]
            c = counts[mi]
            ui = (torch.rand(n, device=device, generator=generator) * c).long().clamp_max_(c - 1)
            return mods_t[mi], units[mi, ui]

        def _train_half(n_vals, y):
            """An arbitrary hash split of the division problem space.

            Not a predicate the model can learn and exploit, and not derived from `x`, so the
            held-out half is genuinely unseen no matter how densely the space is sampled.
            """
            return ((y * 2654435761 + n_vals * 40503) % 1000) < 500

        def sample_uniform_y(pool, n, want_train, generator=None):
            got_n, got_y, have = [], [], 0
            for _ in range(32):
                mi = pool[
                    torch.randint(0, pool.numel(), (4 * n,), device=device, generator=generator)
                ]
                nv = mods_t[mi]
                hi = torch.minimum(
                    (arm["q_cap"] + 1) * nv, nv * nv
                )  # never past N^2, the real reduce's range
                y = (torch.rand(4 * n, device=device, generator=generator) * hi).long()
                y = torch.minimum(y, hi - 1)
                keep = _train_half(nv, y) == want_train
                got_n.append(nv[keep])
                got_y.append(y[keep])
                have += int(keep.sum())
                if have >= n:
                    break
            return torch.cat(got_n)[:n], torch.cat(got_y)[:n]

        def draw(pool, units, counts, n, want_train, generator=None):
            """`(N, base, prompt input, {depth: target})` for this arm's task."""
            if arm["task"] == "redmod_u":
                nv, y = sample_uniform_y(pool, n, want_train, generator=generator)
                return nv, y, y, {1: y % nv}
            nv, x0 = sample_pairs(pool, units, counts, n, generator=generator)
            xin, tgts = io_pairs(nv, x0, base_cfg["eval_max_depth"])
            return nv, x0, xin, tgts

        # --- fixed eval pools, built before training so every readout is comparable --
        ev_gen = torch.Generator(device=device).manual_seed(12345)
        EVAL = {}
        for name, pool, units, counts, want_train in (
            ("seen_n_seen_x", train_mod_idx, units_tr, cnt_tr, True),
            ("seen_n_heldout_x", train_mod_idx, units_te, cnt_te, False),
            ("heldout_n", held_mod_idx, units_tr, cnt_tr, True),
        ):
            EVAL[name] = draw(pool, units, counts, eval_cap, want_train, generator=ev_gen)

        # --- the floor, computed on the eval pool rather than quoted ----------------
        # Examples where the modulus never bites: the answer is obtainable with no knowledge
        # of `N` at all. This is the number every held-out reading must beat to mean anything.
        floors = {}
        for name, (nv, x0, xin, _) in EVAL.items():
            if arm["task"] == "redmod_u":
                free = xin < nv
            elif arm["task"] in ("sq", "redmod"):
                # Identical by construction: `redmod` is handed `x^2` for the same `x`, so
                # its free cases are exactly `sq`'s and the two floors are comparable.
                free = (x0 * x0) < nv
            elif arm["task"] == "mul2":
                free = (2 * x0) < nv
            else:
                free = torch.zeros_like(x0, dtype=torch.bool)  # no reduction to skip
            floors[name] = free.float().mean().item()
        print(
            "[floor] no-reduction fraction: "
            + " ".join(f"{k}={v:.4f}" for k, v in floors.items()),
            flush=True,
        )

        # ----------------------------- train ---------------------------------------
        model = _make_model(cfg, arm, max_len, n_ans, sig_idx, device, n_aux=n_aux)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{arm_name}] params={n_params:,} max_len={max_len}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / base_cfg["warmup"])
            * (1.0 if const_lr else 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))),
        )

        @torch.no_grad()
        def eval_exact(cap=None):
            model.eval()
            out = {}
            for name, (nv, _, xin, tgts) in EVAL.items():
                if cap:
                    nv, xin = nv[:cap], xin[:cap]
                h = model.enc(prompts_for(nv, xin), read_pos)
                row = {}
                for dstep in eval_depths:
                    h = model.roll(h, 1)
                    tgt = tgts[dstep][:cap] if cap else tgts[dstep]
                    row[dstep] = (
                        (model.dec(h).argmax(-1) == digits_of(tgt, n_ans))
                        .all(-1).float().mean().item()
                    )
                out[name] = row
            model.train()
            return out

        model.train()
        log = []
        for step in range(steps):
            n_vals, x0, xin, tgts = draw(
                train_mod_idx, units_tr, cnt_tr, batch_size, True
            )
            target = digits_of(tgts[1], n_ans)
            h0 = model.enc(prompts_for(n_vals, xin), read_pos)
            h = model.roll(h0, 1)
            logits = model.dec(h)
            loss = F.cross_entropy(logits.reshape(-1, b), target.reshape(-1))
            if n_aux:
                loss = loss + arm["aux_product"] * F.cross_entropy(
                    model.aux_logits(h0).reshape(-1, b),
                    digits_of(x0 * x0, n_aux).reshape(-1),
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_cfg["grad_clip"])
            opt.step()
            sched.step()

            if (step + 1) % max(1, steps // 30) == 0:
                # Grokking is a transition, so the generalisation axes are tracked *during*
                # training: an endpoint-only readout cannot tell a model that never
                # generalised from one that generalised late.
                acc = (logits.argmax(-1) == target).all(-1).float().mean().item()
                ev = eval_exact(cap=512)
                print(
                    f"  [{arm_name}] step {step+1:>7} ce={loss.item():.4f} train={acc:.3f} | "
                    f"@1 seen={ev['seen_n_seen_x'][1]:.3f} heldx={ev['seen_n_heldout_x'][1]:.3f} "
                    f"heldN={ev['heldout_n'][1]:.3f}",
                    flush=True,
                )
                log.append(
                    dict(
                        step=step + 1,
                        ce=loss.item(),
                        train_exact=acc,
                        **{f"{k}@1": v[1] for k, v in ev.items()},
                    )
                )

        # ----------------------------- evaluate ------------------------------------
        arm_res = {
            "n_params": n_params,
            "family": fam_info,
            "floor_no_reduction": floors,
            "train_log": log,
            "exact": eval_exact(),
        }
        for name, row in arm_res["exact"].items():
            print(
                f"  [{arm_name}] exact@T ({name}): "
                + " ".join(f"{k}:{v:.4f}" for k, v in row.items())
                + f"   [floor {floors[name]:.4f}]",
                flush=True,
            )

        # --- is the representation organised by the group? --------------------------
        # `ballistic_depth/rule_structure/`'s two discriminating statistics, computed per
        # modulus on this cut's encoders so a representation change can be read even if
        # accuracy does not move. Same functions, so the numbers are directly comparable.
        # `redmod_u`'s inputs are drawn from the division problem space, not from a modulus's
        # unit group, so the group coordinate does not index them and the probe is skipped.
        if group_probe and arm["task"] != "redmod_u":
            gp_gen = torch.Generator(device="cpu").manual_seed(0)
            rows = []
            model.eval()
            for n_val, p, q, _ in fam_train[:8]:
                P, Q = p - 1, q - 1
                if P < 4 or Q < 4:
                    continue
                units = ModulusFamily.units_for(p, q)
                a_np, b_np, _, _ = group_coords(p, q, units)
                if len(set(zip(a_np.tolist(), b_np.tolist()))) != units.size:
                    continue
                a = torch.tensor(a_np, device=device)
                bb = torch.tensor(b_np, device=device)
                x0g = torch.tensor(units, device=device)
                nv = torch.full_like(x0g, int(n_val))
                xv, _ = io_pairs(nv, x0g, 1)  # the arm's own prompt input
                with torch.no_grad():
                    emb = torch.cat(
                        [
                            model.enc(prompts_for(nv[i : i + 2048], xv[i : i + 2048]), read_pos).float()
                            for i in range(0, xv.numel(), 2048)
                        ]
                    )
                real_f = _fourier_conc(emb, a, bb, P, Q)
                real_t = _translation_r2(emb, a, bb, P, Q)
                perm = torch.randperm(emb.shape[0], generator=gp_gen).to(device)
                rows.append(
                    dict(
                        modulus=int(n_val),
                        fourier=real_f,
                        fourier_null=_fourier_conc(emb[perm], a, bb, P, Q),
                        translation_r2=real_t,
                        translation_r2_null=_translation_r2(emb[perm], a, bb, P, Q),
                    )
                )
            model.train()
            arm_res["group_structure"] = rows
            if rows:
                mean = lambda k: float(np.mean([r[k] for r in rows]))
                arm_res["group_structure_mean"] = {
                    k: mean(k) for k in ("fourier", "fourier_null", "translation_r2", "translation_r2_null")
                }
                m = arm_res["group_structure_mean"]
                print(
                    f"  [{arm_name}] group ({len(rows)} moduli): fourier "
                    f"{m['fourier']:.4f} (null {m['fourier_null']:.4f}) | translation R2 "
                    f"{m['translation_r2']:.4f} (null {m['translation_r2_null']:.4f})",
                    flush=True,
                )

        results["arms"][arm_name] = arm_res

        if save_ckpt:
            ck_dir = Path(DATA_DIR) / "rule_acquisition" / str(tag) / "ckpt"
            ck_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg, "arm": arm_name},
                ck_dir / f"{arm_name}_seed{seed}.pt",
            )
            print(f"  [{arm_name}] checkpoint saved", flush=True)

    out_dir = Path(DATA_DIR) / "rule_acquisition" / str(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] {out_path}", flush=True)
    return results
