"""Variable modulus: does a ballistic operator have to be *conditioned on the rule*?

The arity-2 cut. [`ballistic_depth/`](README.md) held `N = 893` fixed for every run, so the
tied operator `h <- h + MLP(LN(h))` never had to be a function of anything but the state:
`x -> x^2 mod 893` is one map, and a tied block can simply *be* that map. Sampling `N` per
example makes the true operator arity-2, `F(h, N)`, which is the modular-squaring analogue
of `mjc/arity_torque`'s command-blind forward model and of `RHM_SCULPTING`'s a1/a2 split.

The wrinkle that makes this substrate its own question. `N` is *in the prompt*, so the model
is not forced into rule-blindness — it can fold `N` into `h_0` and let the tied block read it
back out at every step. That is arity-2 in function but arity-1 in *shape*, and it buys a
specific liability: the rolled state must now simultaneously (a) advance `x_t -> x_{t+1}` and
(b) preserve `N` exactly, forever, with no ground truth pulling `N` back. Under
`ballistic_depth` §2's closure reading the encoder manifold is now a two-index family
`Enc(N, x)`, and the rollout has to stay on the *fiber* over the right `N`.

So the arms separate *where the rule lives*, not whether the model has it:

  blind     the encoder never sees `N`. Genuinely rule-blind: the best available prediction
            is the `N`-averaged next state. The floor, and the check that the family is
            diverse enough for the rule to be load-bearing at all.
  fold      `N` in the prompt, operator `F(h)`. The rule is *carried* by the state.
  cond      `N` in the prompt AND re-injected into the operator every step as `F(h, r_N)`,
            with `r_N` computed from `N`'s digits alone. The rule cannot drift.
  *_cyc     each of the above plus `ballistic_depth`'s label-free cycle constraint.

Pre-registered predictions:

  P1  `blind` floors at every depth including T=1.
  P2  `cond` has a longer composition horizon than `fold`, and the gap *opens with depth*
      (rule drift accumulates where re-injection does not).
  P3  cycle helps `fold` more than `cond` — sub-additive, because the cycle term's re-encode
      carries the correct `N` by construction and so pins the rule for free, which is the
      one thing `cond` already has.
  P4  rule retention decays in `t` for `fold` and is flat for `cond`, and its decay tracks
      the `fold` horizon.
  P5  held-out-`N` generalisation: no prior. This is the arity axis proper.

Instruments beyond `ballistic_depth`'s (`exact@T`, `verid@t`, `onmanifold@t`, coldstart):

  inrange@t   fraction of decoded `x_t` that are `< N`. The behavioural, coordinate-free
              rule readout: a rollout that has lost its modulus emits residues that do not
              belong to its own world.
  rulesep@t   cos(h_t(N), h_t(N')) for the *same* `x_0` under a different modulus, against
              its own t=0 value. Measures whether the rollout keeps the two worlds apart —
              coordinate-free, no ground truth, defined for every arm.
  ruleswap@T  causal, `cond` arms only: roll with `r` from the wrong modulus. If `cond`'s
              advantage were extra per-step capacity rather than the rule, this would not
              move. Replaces a parameter-matched control with a direct intervention.

As in `ballistic_depth`, no arm ever trains on an intermediate residue; the trajectory is
instrumentation only, and `T` is given to the operator as a loop count and never appears in
the prompt.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

CFG = dict(
    # --- task / DGP
    n_digits=4,
    min_margin=60,
    min_factor=11,
    heldout_mod_fraction=0.2,
    test_x_fraction=0.1,
    family_seed=45,
    train_depths="1,2,3,4,5,6",
    eval_max_depth=40,
    # --- model
    d_model=256,
    n_enc_layers=2,
    n_heads=4,
    d_ff=1024,
    d_op_ff=1024,
    d_rule=128,
    # --- training
    steps=20000,
    batch_size=256,
    lr=3e-4,
    warmup=500,
    weight_decay=0.01,
    grad_clip=1.0,
    # w=3.0, not w=1.0: ballistic_depth §7 swept the cycle weight at N=893 and found the
    # horizon peaks at w~10 and collapses by w=30. w=3 is inside the broad low side of that
    # peak (46.3 vs 51.0 at the optimum) and further from the cliff. Transferred, not re-swept.
    consist_cycle=3.0,
    consist_reentry=0.0,  # ballistic_depth §4: dose-dependently harmful above w~0.1.
    consist_warmup=0.3,
    # --- run
    arms="blind,fold,cond,fold_cyc,cond_cyc",
    eval_cap=2048,
    seed=0,
    tag="smoke",
)

DIGIT_OFFSET = 7

# see_n: does the encoder get the N field. cond: is the rule re-injected into the operator.
ARM_SPEC = {
    "blind": dict(see_n=False, cond=False, cyc=False),
    "fold": dict(see_n=True, cond=False, cyc=False),
    "cond": dict(see_n=True, cond=True, cyc=False),
    "fold_cyc": dict(see_n=True, cond=False, cyc=True),
    "cond_cyc": dict(see_n=True, cond=True, cyc=True),
}


def _make_model(cfg, spec, max_len, device):
    import torch
    from torch import nn

    from one_layer_deeper.squaring_mod import VOCAB_SIZE

    d = cfg["d_model"]
    n_dig = cfg["n_digits"]

    class Encoder(nn.Module):
        """Prompt -> h0. Also usable on a soft digit distribution (for re-encoding)."""

        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB_SIZE, d)
            self.pos = nn.Parameter(torch.zeros(max_len, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg["n_heads"],
                dim_feedforward=cfg["d_ff"],
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.body = nn.TransformerEncoder(layer, num_layers=cfg["n_enc_layers"])
            self.norm = nn.LayerNorm(d)

        def _run(self, embeds, read_pos):
            h = embeds + self.pos.unsqueeze(0)
            return self.norm(self.body(h))[:, read_pos, :]

        def forward(self, input_ids, read_pos):
            return self._run(self.tok(input_ids), read_pos)

        def forward_soft_digits(self, template_ids, digit_probs, slot, read_pos):
            """Re-encode: same prompt (same N field), x-field replaced by soft digits."""
            embeds = self.tok(template_ids).clone()
            digit_table = self.tok.weight[DIGIT_OFFSET : DIGIT_OFFSET + 10]
            soft = digit_probs @ digit_table
            embeds = torch.cat(
                [embeds[:, :slot], soft, embeds[:, slot + digit_probs.shape[1] :]], dim=1
            )
            return self._run(embeds, read_pos)

    class RuleEncoder(nn.Module):
        """N's digits -> r. Deliberately *not* a readout from the prompt encoder.

        A bidirectional encoder mixes every field, so a rule vector read off the prompt
        would also carry `x_0`, and re-injecting it each step would leak the initial state
        into the rollout — a different intervention than the one under test. Reading the
        digits of `N` directly guarantees `r` is a function of the rule alone. Digit-based
        (rather than a per-modulus embedding) is also what makes held-out `N` conceivable.
        """

        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, cfg["d_rule"])
            self.pos = nn.Parameter(torch.zeros(n_dig, cfg["d_rule"]))
            nn.init.normal_(self.pos, std=0.02)
            self.net = nn.Sequential(
                nn.Linear(n_dig * cfg["d_rule"], d), nn.GELU(), nn.Linear(d, d)
            )
            self.out_norm = nn.LayerNorm(d)

        def forward(self, n_digits):  # [B, n_dig] -> [B, d]
            e = self.emb(n_digits) + self.pos.unsqueeze(0)
            return self.out_norm(self.net(e.reshape(e.shape[0], -1)))

    class Block(nn.Module):
        """The tied operator. `r` is None for arity-1-shaped arms."""

        def __init__(self, conditioned):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.net = nn.Sequential(
                nn.Linear(d, cfg["d_op_ff"]), nn.GELU(), nn.Linear(cfg["d_op_ff"], d)
            )
            self.rule_proj = nn.Linear(d, d, bias=False) if conditioned else None

        def forward(self, h, r=None):
            z = self.norm(h)
            if self.rule_proj is not None:
                z = z + self.rule_proj(r)
            return h + self.net(z)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, n_dig * 10)

        def forward(self, h):
            return self.head(self.norm(h)).reshape(h.shape[0], n_dig, 10)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conditioned = spec["cond"]
            self.enc = Encoder()
            self.dec = Decoder()
            self.op = Block(self.conditioned)
            if self.conditioned:
                self.rule = RuleEncoder()

        def rule_vec(self, n_digits):
            return self.rule(n_digits) if self.conditioned else None

        def roll(self, h, n_steps, r=None, collect=False):
            states = [h] if collect else None
            for _ in range(n_steps):
                h = self.op(h, r)
                if collect:
                    states.append(h)
            return (h, states) if collect else h

    return Model().to(device)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=16384,
)
def variable_modulus(
    tag: str = "smoke",
    arms: str = "blind,fold,cond,fold_cyc,cond_cyc",
    steps: int = 20000,
    seed: int = 0,
    eval_max_depth: int = 40,
    eval_cap: int = 2048,
    train_depths: str = "1,2,3,4,5,6",
    lr: float = 3e-4,
    batch_size: int = 256,
    d_model: int = 256,
    consist_cycle: float = 3.0,
    consist_reentry: float = 0.0,
    consist_warmup: float = 0.3,
    min_margin: int = 60,
    min_factor: int = 11,
    n_digits: int = 4,
    mod_lo: int = 0,  # 0 -> the full digit-width band
    mod_hi: int = 0,
    max_moduli: int = 0,  # 0 -> every train modulus in the family
    d_ff: int = 1024,
    # The grokking recipe: memorisation is defeated by holding out *half* the bases (so
    # lookup cannot cover the test set), high weight decay, and training far past the point
    # where train accuracy saturates. Constant LR because a cosine decaying to ~0 can stall
    # the transition it is supposed to reveal.
    weight_decay: float = 0.01,
    test_x_fraction: float = 0.1,
    const_lr: bool = False,
    save_ckpt: bool = False,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

    cfg = {
        **CFG,
        "tag": tag,
        "arms": arms,
        "steps": steps,
        "seed": seed,
        "eval_max_depth": eval_max_depth,
        "eval_cap": eval_cap,
        "train_depths": train_depths,
        "lr": lr,
        "batch_size": batch_size,
        "d_model": d_model,
        "consist_cycle": consist_cycle,
        "consist_reentry": consist_reentry,
        "consist_warmup": consist_warmup,
        "min_margin": min_margin,
        "min_factor": min_factor,
        "n_digits": n_digits,
        "max_moduli": max_moduli,
        # The *operator's* width, held separate from the encoder's. The capacity that has to
        # cover `reachable_states` is the tied block's, not the encoder's, so this is the
        # knob the capacity diagnostic moves.
        "d_op_ff": d_ff,
        "weight_decay": weight_decay,
        "test_x_fraction": test_x_fraction,
        "const_lr": const_lr,
        "mod_lo": mod_lo or None,
        "mod_hi": mod_hi or None,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_depths = tuple(int(v) for v in str(cfg["train_depths"]).split(","))
    n_dig = cfg["n_digits"]

    # ----------------------------- the DGP -----------------------------------------
    family = ModulusFamily(
        n_digits=n_dig,
        min_margin=cfg["min_margin"],
        min_factor=cfg["min_factor"],
        heldout_fraction=cfg["heldout_mod_fraction"],
        seed=cfg["family_seed"],
        mod_lo=cfg["mod_lo"],
        mod_hi=cfg["mod_hi"],
    )
    fam_train, fam_held = family.split()
    fam_info = family.describe()
    print(f"[family] {json.dumps({k: v for k, v in fam_info.items() if 'moduli' not in k or k.startswith('n_')})}")
    if fam_info["depth_first_repeat_min"] <= eval_max_depth:
        raise ValueError(
            f"depth periodicity leaks into the eval range for at least one modulus "
            f"(min first-repeat {fam_info['depth_first_repeat_min']} <= max depth {eval_max_depth})"
        )

    # Subsampling the *train* moduli is the capacity knob. At depth >= 1 the state collapses
    # into each modulus's quadratic-residue subgroup (size phi(N)/4, since squaring is 4-to-1
    # on the units of a product of two odd primes), and those subgroups are disjoint across
    # moduli — so the tied operator has to realise `n_train_moduli` separate permutations
    # totalling `reachable_states` points. That total, not the number of rules, is what the
    # operator's width has to pay for; the fixed-N cut's was 207.
    if max_moduli and len(fam_train) > max_moduli:
        pick = np.random.default_rng(cfg["family_seed"] + 1).choice(
            len(fam_train), max_moduli, replace=False
        )
        fam_train = sorted(fam_train[i] for i in pick)

    reachable = sum((p - 1) * (q - 1) // 4 for _, p, q, _ in fam_train)
    fam_info["n_train_moduli"] = len(fam_train)
    fam_info["train_moduli"] = [m[0] for m in fam_train]
    fam_info["reachable_states"] = reachable
    fam_info["reachable_states_per_modulus"] = reachable // max(1, len(fam_train))
    print(
        f"[capacity] {len(fam_train)} train moduli -> {reachable} reachable states at depth>=1 "
        f"({reachable // max(1, len(fam_train))}/modulus); fixed-N cut was 207",
        flush=True,
    )

    members = fam_train + fam_held  # (N, p, q, margin)
    mod_vals = np.array([m[0] for m in members], dtype=np.int64)
    n_mods = len(members)
    is_train_mod = np.zeros(n_mods, dtype=bool)
    is_train_mod[: len(fam_train)] = True  # fam_train occupies the first block

    # Per-modulus unit pools, split into a train-x and a held-out-x half. Padded to a
    # rectangle so sampling is a single gather. Held-out x is a *separate* axis from
    # held-out N and both are reported.
    rng = np.random.default_rng(cfg["family_seed"])
    tr_pools, te_pools = [], []
    for n_val, p, q, _ in members:
        u = ModulusFamily.units_for(p, q)
        perm = rng.permutation(u.size)
        n_te = int(round(cfg["test_x_fraction"] * u.size))
        te_pools.append(np.sort(u[perm[:n_te]]))
        tr_pools.append(np.sort(u[perm[n_te:]]))

    def _pad(pools):
        width = max(a.size for a in pools)
        out = np.zeros((len(pools), width), dtype=np.int64)
        cnt = np.zeros(len(pools), dtype=np.int64)
        for i, a in enumerate(pools):
            out[i, : a.size] = a
            cnt[i] = a.size
        return out, cnt

    units_tr, cnt_tr = _pad(tr_pools)
    units_te, cnt_te = _pad(te_pools)
    units_tr = torch.tensor(units_tr, device=device)
    units_te = torch.tensor(units_te, device=device)
    cnt_tr = torch.tensor(cnt_tr, device=device)
    cnt_te = torch.tensor(cnt_te, device=device)
    mods_t = torch.tensor(mod_vals, device=device)
    train_mod_idx = torch.tensor(np.flatnonzero(is_train_mod), device=device)
    held_mod_idx = torch.tensor(np.flatnonzero(~is_train_mod), device=device)
    # Sorted train moduli, for picking a distractor N' > x0 in the rulesep probe.
    train_mods_sorted = torch.sort(mods_t[train_mod_idx]).values

    print(
        f"[data] {n_mods} moduli ({train_mod_idx.numel()} train / {held_mod_idx.numel()} held), "
        f"{int(cnt_tr.sum())} train-x + {int(cnt_te.sum())} heldout-x pairs, "
        f"margin>={fam_info['depth_first_repeat_min']}",
        flush=True,
    )

    def digits_of(v, width=n_dig):
        """[B] -> [B, width] decimal digits, most significant first."""
        return torch.stack(
            [(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1
        )

    def sample_pairs(mod_idx_pool, units, counts, n, generator=None):
        mi = mod_idx_pool[
            torch.randint(0, mod_idx_pool.numel(), (n,), device=device, generator=generator)
        ]
        c = counts[mi]
        ui = (torch.rand(n, device=device, generator=generator) * c).long().clamp_max_(
            c - 1
        )
        return mods_t[mi], units[mi, ui]

    def trajectory(n_vals, x0, depth):
        """[B, depth+1] of x_t = x_{t-1}^2 mod N. Exact in int64 (x^2 < 1e8)."""
        out = [x0]
        x = x0
        for _ in range(depth):
            x = (x * x) % n_vals
            out.append(x)
        return torch.stack(out, dim=1)

    # ----------------------------- fixed eval sets ----------------------------------
    # Built once, before any arm, so every arm is scored on identical problems.
    ev_gen = torch.Generator(device=device).manual_seed(12345)
    EVAL_POOLS = {}
    for name, pool, units, counts in (
        ("seen_n_seen_x", train_mod_idx, units_tr, cnt_tr),
        ("seen_n_heldout_x", train_mod_idx, units_te, cnt_te),
        ("heldout_n", held_mod_idx, units_tr, cnt_tr),
    ):
        nv, xv = sample_pairs(pool, units, counts, eval_cap, generator=ev_gen)
        EVAL_POOLS[name] = (nv, xv, trajectory(nv, xv, eval_max_depth))

    arms_list = [a.strip() for a in str(cfg["arms"]).split(",") if a.strip()]
    results = {"config": cfg, "family": fam_info, "arms": {}}

    for arm in arms_list:
        spec = ARM_SPEC[arm]
        see_n, use_cond, use_cyc = spec["see_n"], spec["cond"], spec["cyc"]
        print(f"\n===== arm={arm} seed={cfg['seed']} spec={spec} =====", flush=True)

        # BOS [N <n_dig digits>] X <n_dig digits> ANS
        max_len = 1 + (1 + n_dig if see_n else 0) + 1 + n_dig + 1
        X_SLOT = 1 + (1 + n_dig if see_n else 0) + 1
        read_pos = max_len - 1

        def prompts_for(n_vals, x_vals, _see_n=see_n):
            b = x_vals.shape[0]
            col = lambda tok: torch.full((b, 1), TOKEN_IDS[tok], dtype=torch.long, device=device)
            parts = [col("BOS")]
            if _see_n:
                parts += [col("N"), digits_of(n_vals) + DIGIT_OFFSET]
            parts += [col("X"), digits_of(x_vals) + DIGIT_OFFSET, col("ANS")]
            return torch.cat(parts, dim=1)

        model = _make_model(cfg, spec, max_len, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{arm}] params={n_params:,} max_len={max_len} x_slot={X_SLOT}", flush=True)

        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / cfg["warmup"])
            * (
                1.0
                if cfg["const_lr"]
                else 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg["steps"])))
            ),
        )

        model.train()
        log = []
        for step in range(cfg["steps"]):
            depth = int(np.random.choice(train_depths))
            n_vals, x0 = sample_pairs(train_mod_idx, units_tr, cnt_tr, cfg["batch_size"])
            traj = trajectory(n_vals, x0, depth)
            digits = digits_of(traj[:, depth])
            ids = prompts_for(n_vals, x0)
            r = model.rule_vec(digits_of(n_vals)) if use_cond else None

            h0 = model.enc(ids, read_pos)
            hT = model.roll(h0, depth, r)
            logits = model.dec(hT)
            ce = F.cross_entropy(logits.reshape(-1, 10), digits.reshape(-1))
            loss = ce
            cyc = torch.zeros((), device=device)
            reentry = torch.zeros((), device=device)

            # Self-generated targets must not teach before their own fidelity exceeds where
            # the system sits — ballistic_depth gotcha 1 (`endo_expansion`'s ceiling).
            consist_on = use_cyc and step >= cfg["consist_warmup"] * cfg["steps"]
            if consist_on and depth >= 2:
                t = int(np.random.randint(1, depth))
                h_t = model.roll(h0, t, r)
                probs = model.dec(h_t).softmax(-1)
                # Template carries the CORRECT N — the rule is an input, not a label. This
                # is exactly why the cycle term is expected to pin the rule for free (P3).
                template = prompts_for(n_vals, torch.zeros_like(x0))
                h_re = model.enc.forward_soft_digits(template, probs, X_SLOT, read_pos)
                cyc = F.mse_loss(h_re, h_t.detach()) + F.mse_loss(h_re.detach(), h_t)
                loss = loss + cfg["consist_cycle"] * cyc
                if cfg["consist_reentry"] > 0:
                    h_end = model.roll(h_re, depth - t, r)
                    reentry = F.cross_entropy(
                        model.dec(h_end).reshape(-1, 10), digits.reshape(-1)
                    )
                    loss = loss + cfg["consist_reentry"] * reentry

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()

            if (step + 1) % max(1, cfg["steps"] // 20) == 0:
                acc = (logits.argmax(-1) == digits).all(-1).float().mean().item()
                # Track the generalisation axes *during* training, not just at the end.
                # Grokking is a transition, and an endpoint-only readout cannot tell a
                # model that never generalised from one that generalised at step 300k.
                model.eval()
                gen = {}
                with torch.no_grad():
                    for pname in ("seen_n_seen_x", "seen_n_heldout_x", "heldout_n"):
                        nv, xv, tj = (z[:256] for z in EVAL_POOLS[pname])
                        rr = model.rule_vec(digits_of(nv)) if use_cond else None
                        hq = model.roll(model.enc(prompts_for(nv, xv), read_pos), 1, rr)
                        gen[pname] = (
                            (model.dec(hq).argmax(-1) == digits_of(tj[:, 1]))
                            .all(-1).float().mean().item()
                        )
                model.train()
                print(
                    f"  [{arm}] step {step+1:>7} d={depth} ce={ce.item():.4f} "
                    f"exact={acc:.3f} cyc={cyc.item():.4f} | gen@1 "
                    f"seenx={gen['seen_n_seen_x']:.3f} heldx={gen['seen_n_heldout_x']:.3f} "
                    f"heldN={gen['heldout_n']:.3f}",
                    flush=True,
                )
                log.append(
                    dict(step=step + 1, depth=depth, ce=ce.item(), exact=acc, cyc=cyc.item(),
                         **{f"gen_{k}": v for k, v in gen.items()})
                )

        # ------------------------------ evaluation ------------------------------
        model.eval()
        arm_res = {"n_params": n_params, "train_log": log}
        depths = list(range(1, eval_max_depth + 1))

        @torch.no_grad()
        def eval_exact(pool_name, swap_rule=False):
            n_vals, x0, traj = EVAL_POOLS[pool_name]
            ids = prompts_for(n_vals, x0)
            h0 = model.enc(ids, read_pos)
            if use_cond:
                nd = digits_of(n_vals)
                if swap_rule:
                    # Causal control: the operator's rule input is a *different* modulus,
                    # while h0 still encodes the true one. If cond's advantage were extra
                    # per-step capacity rather than the rule, this would not move.
                    m = train_mods_sorted.numel()
                    j = torch.randint(0, m, (n_vals.shape[0],), device=device)
                    wrong = train_mods_sorted[j]
                    wrong = torch.where(wrong == n_vals, train_mods_sorted[(j + 1) % m], wrong)
                    nd = digits_of(wrong)
                r = model.rule_vec(nd)
            else:
                r = None
            out, h = {}, h0
            for d in depths:
                h = model.op(h, r)
                pred = model.dec(h).argmax(-1)
                out[d] = (pred == digits_of(traj[:, d])).all(-1).float().mean().item()
            return out

        for pool_name in EVAL_POOLS:
            arm_res[f"exact_{pool_name}"] = eval_exact(pool_name)
        if use_cond:
            arm_res["exact_rule_swapped"] = eval_exact("seen_n_seen_x", swap_rule=True)

        def _show(d):
            keys = [k for k in (1, 2, 4, 6, 8, 10, 15, 20, 30, 40) if k in d]
            return " ".join(f"{k}:{d[k]:.3f}" for k in keys)

        for pool_name in EVAL_POOLS:
            print(f"  [{arm}] exact@T ({pool_name}): {_show(arm_res[f'exact_{pool_name}'])}", flush=True)
        if use_cond:
            print(f"  [{arm}] exact@T (rule SWAPPED): {_show(arm_res['exact_rule_swapped'])}", flush=True)

        # ---- rollout instruments, on the seen-N/seen-x pool ----
        @torch.no_grad()
        def rollout_probe():
            n_vals, x0, traj = EVAL_POOLS["seen_n_seen_x"]
            r = model.rule_vec(digits_of(n_vals)) if use_cond else None
            h0 = model.enc(prompts_for(n_vals, x0), read_pos)

            # Distractor modulus N' > x0, so `x0` is a legal residue in that world too.
            lo = torch.searchsorted(train_mods_sorted, x0 + 1)
            avail = (train_mods_sorted.numel() - lo).clamp_min(1)
            pick = lo + (torch.rand(x0.shape[0], device=device) * avail).long()
            n_prime = train_mods_sorted[pick.clamp_max_(train_mods_sorted.numel() - 1)]
            r_p = model.rule_vec(digits_of(n_prime)) if use_cond else None
            h0p = model.enc(prompts_for(n_prime, x0), read_pos)

            verid, inrange, onman, rulesep = {}, {}, {}, {}
            sep0 = F.cosine_similarity(h0, h0p, dim=-1).mean().item()
            h, hp = h0, h0p
            for t in range(1, eval_max_depth + 1):
                h = model.op(h, r)
                hp = model.op(hp, r_p)
                pred = model.dec(h).argmax(-1)
                val = sum(pred[:, k] * 10 ** (n_dig - 1 - k) for k in range(n_dig))
                verid[t] = (val == traj[:, t]).float().mean().item()
                inrange[t] = (val < n_vals).float().mean().item()
                h_true = model.enc(prompts_for(n_vals, traj[:, t]), read_pos)
                onman[t] = F.cosine_similarity(h, h_true, dim=-1).mean().item()
                rulesep[t] = F.cosine_similarity(h, hp, dim=-1).mean().item()
            return verid, inrange, onman, rulesep, sep0

        v, ir, om, rs, rs0 = rollout_probe()
        arm_res.update(
            veridicality=v,
            inrange=ir,
            on_manifold_cos=om,
            rule_separation=rs,
            rule_separation_t0=rs0,
        )
        print(f"  [{arm}] verid@t:   {_show(v)}", flush=True)
        print(f"  [{arm}] inrange@t: {_show(ir)}", flush=True)
        print(f"  [{arm}] onman@t:   {_show(om)}", flush=True)
        print(f"  [{arm}] rulesep@t: t0:{rs0:.3f} {_show(rs)}", flush=True)

        # ---- cold start: is the operator sound at depth, or is it the state that fails? ----
        @torch.no_grad()
        def coldstart(pool_name, targets, grid):
            n_vals, _, traj = EVAL_POOLS[pool_name]
            r = model.rule_vec(digits_of(n_vals)) if use_cond else None
            out = {}
            for T in targets:
                row = {}
                for t in grid:
                    if t > T:
                        continue
                    h = model.enc(prompts_for(n_vals, traj[:, t]), read_pos)
                    if T - t:
                        h = model.roll(h, T - t, r)
                    pred = model.dec(h).argmax(-1)
                    row[t] = (pred == digits_of(traj[:, T])).all(-1).float().mean().item()
                out[T] = row
            return out

        cs_targets = [T for T in (10, 20, 40) if T <= eval_max_depth]
        cs_grid = [t for t in (0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35) if t <= eval_max_depth]
        arm_res["coldstart_seen_n_seen_x"] = coldstart("seen_n_seen_x", cs_targets, cs_grid)
        for T in cs_targets:
            row = arm_res["coldstart_seen_n_seen_x"][T]
            print(
                f"  [{arm}] coldstart T={T}: " + " ".join(f"t{t}:{x:.3f}" for t, x in row.items()),
                flush=True,
            )

        results["arms"][arm] = arm_res

        if save_ckpt:
            ck_dir = Path(DATA_DIR) / "variable_modulus" / str(cfg["tag"]) / "ckpt"
            ck_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg, "arm": arm},
                ck_dir / f"{arm}_seed{cfg['seed']}.pt",
            )
            print(f"  [{arm}] checkpoint saved", flush=True)

    out_dir = Path(DATA_DIR) / "variable_modulus" / str(cfg["tag"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{cfg['seed']}.json"
    out_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] {out_path}", flush=True)
    return results
