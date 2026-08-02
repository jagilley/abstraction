"""Ballistic depth: does a learned latent operator survive its own iteration?

The cut. Repeated modular squaring is a purely *ballistic* rollout — the model commits at
step 0 and never observes an intermediate state, so unlike every controller in `mjc/`
there is no reactive fallback to re-ground past a stale operator. That makes exact-match
at held-out depth a fully *sighted* grader of forward-model quality (cf. `mjc/ballistic/`
4b, where reactive transmits FM error at slope +0.35 and ballistic at +1.07).

The claim under test, in three parts:

  (1) A continuous latent rollout of this operator must drift, because the map is
      expanding in *every* coordinate. In the discrete-log coordinate `x = g^a`, squaring
      is `a -> 2a`: the doubling map, Lyapunov exponent ln 2, exactly one bit of state
      precision destroyed per step. In the raw residue coordinate it is far worse (local
      derivative 2x, i.e. O(N)). There is no contracting coordinate.

  (2) Therefore depth extrapolation needs the rollout to be error-*correcting*, not merely
      accurate: the state must be re-attracted onto a codebook of residues every step.
      Here that is available in principle — squaring is a bijection on the 2407-element
      quadratic-residue subgroup for all t >= 1.

  (3) Terminal CE alone is one grader (evaluative, sparse in depth). Its failure mode is
      the depth-indexed lookup table. A label-free consistency constraint is the second,
      dense grader; its failure mode alone is collapse to identity. Per
      `ideas/heterogeneous_graders.md` §4 neither alone is a learner, and the constraints
      here are non-mirror (algebraic identities are facts about the operator, not about
      the model's beliefs).

Design. One encoder, one tied operator, one decoder, shared across arms; a 2x2 over
{quantize} x {consistency} plus a non-recurrent control. Depth is the only held-out axis
that carries the conclusion: train on T in 1..6, evaluate T in 1..20. Base values x are
almost all seen in training on purpose — x-generalization is a *separate* question and
leaving it in would confound the depth readout. A 10% held-out-x slice is reported
alongside so the two axes stay visible.

Instruments (the mechanism readouts, which are the trustworthy part):
  - `exact@T`      : exact-match on the decoded residue vs depth.
  - `verid@t`      : the *terminal* decoder applied zero-shot to `h_t` vs the true `x_t`.
                     Never trained at intermediate t, so it reads whether the intermediate
                     state is the same kind of object as a terminal state.
  - `onmanifold@t` : cos(h_t rolled, Enc(x_t)) — the rollout's distance from the encoder's
                     own representation of the true intermediate residue. The direct
                     analogue of `_rollout_fidelity` in `rhm/rhm_sculpt_twofm.py`.
  - `amplify@t`    : ||dh_t|| / ||dh_0|| under an input perturbation sweep. This is the
                     direct test of claim (1): continuous arms should amplify (>1 per
                     step), quantized arms should absorb small perturbations outright.

No arm ever trains on an intermediate residue. The trajectory is instrumentation only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

CFG = dict(
    # --- task
    # N = 893 = 19*47. Chosen so the *success* mode is reachable in budget: 828 units, a
    # 207-element attractor set (squaring is a bijection on it for t>=1), 3-digit answers,
    # and the first depth-repeat at T=67 — far above the eval range, so periodicity is
    # never a shortcut. Scaling N up is the natural second cut, not this one.
    p=19,
    q=47,
    train_depths="1,2,3,4,5,6",
    eval_max_depth=20,
    test_fraction=0.1,
    # --- model
    d_model=256,
    n_enc_layers=2,
    n_heads=4,
    d_ff=1024,
    codebook_size=4096,
    ff_stack_depth=6,  # non-recurrent control: untied blocks, matched to max train depth
    # --- training
    steps=15000,
    batch_size=256,
    lr=3e-4,
    warmup=500,
    weight_decay=0.01,
    grad_clip=1.0,
    vq_commit=0.25,
    consist_cycle=1.0,
    consist_reentry=1.0,
    consist_warmup=0.3,
    dead_code_every=500,
    # --- run
    arms="base,quant,consist,quant_consist,feedforward",
    eval_cap=2000,
    seed=0,
    tag="smoke",
)

DIGIT_OFFSET = 7


def _make_model(cfg, n_answer_digits, arm, max_len, device):
    import torch
    from torch import nn

    from one_layer_deeper.squaring_mod import VOCAB_SIZE

    d = cfg["d_model"]
    n_dig = n_answer_digits

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
            """Re-encode: same prompt, x-field replaced by a soft digit distribution.

            `digit_probs` is [B, n_dig, 10]. Mixing the embedding table keeps the round
            trip differentiable, which is what lets the cycle term reach the operator.
            """
            embeds = self.tok(template_ids).clone()
            digit_table = self.tok.weight[DIGIT_OFFSET : DIGIT_OFFSET + 10]
            soft = digit_probs @ digit_table
            embeds = torch.cat(
                [embeds[:, :slot], soft, embeds[:, slot + digit_probs.shape[1] :]], dim=1
            )
            return self._run(embeds, read_pos)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.net = nn.Sequential(
                nn.Linear(d, cfg["d_ff"]), nn.GELU(), nn.Linear(cfg["d_ff"], d)
            )

        def forward(self, h):
            return h + self.net(self.norm(h))

    class Quantizer(nn.Module):
        """Straight-through VQ — the *re-attraction* the claim says is mandatory."""

        def __init__(self, k):
            super().__init__()
            self.emb = nn.Embedding(k, d)
            nn.init.normal_(self.emb.weight, std=1.0)
            self.register_buffer("usage", torch.zeros(k))
            self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

        def forward(self, z):
            flat = z.reshape(-1, d)
            if self.training and not bool(self.ready):
                # Seed the codebook from real operator outputs. A randn codebook sits at
                # an arbitrary scale relative to the state, which starves the assignment.
                with torch.no_grad():
                    pick = torch.randint(
                        0, flat.shape[0], (self.emb.weight.shape[0],), device=flat.device
                    )
                    self.emb.weight.data.copy_(
                        flat[pick].detach() + 0.01 * torch.randn_like(flat[pick])
                    )
                    self.ready.fill_(True)
            dist = (
                flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.emb.weight.t()
                + self.emb.weight.pow(2).sum(1).unsqueeze(0)
            )
            idx = dist.argmin(1)
            codes = self.emb(idx).reshape(z.shape)
            if self.training:
                with torch.no_grad():
                    self.usage.index_add_(
                        0, idx, torch.ones(idx.shape[0], device=idx.device)
                    )
            loss = (codes.detach() - z).pow(2).mean() * cfg["vq_commit"] + (
                codes - z.detach()
            ).pow(2).mean()
            return z + (codes - z).detach(), loss, idx

        @torch.no_grad()
        def revive_dead(self, z):
            dead = (self.usage == 0).nonzero().flatten()
            self.usage.zero_()
            if dead.numel() == 0:
                return 0
            flat = z.reshape(-1, d)
            pick = torch.randint(0, flat.shape[0], (dead.numel(),), device=flat.device)
            self.emb.weight.data[dead] = flat[pick] + 0.01 * torch.randn_like(flat[pick])
            return int(dead.numel())

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
            self.arm = arm
            self.recurrent = arm != "feedforward"
            self.enc = Encoder()
            self.dec = Decoder()
            if self.recurrent:
                self.op = Block()
            else:
                self.stack = nn.ModuleList(
                    [Block() for _ in range(cfg["ff_stack_depth"])]
                )
            self.quantize = arm in ("quant", "quant_consist")
            if self.quantize:
                self.vq = Quantizer(cfg["codebook_size"])

        def encode(self, input_ids, read_pos):
            return self.enc(input_ids, read_pos)

        def step(self, h):
            z = self.op(h)
            if self.quantize:
                out, loss, _ = self.vq(z)
                return out, loss
            return z, h.new_zeros(())

        def roll(self, h, n_steps, collect=False):
            vq_loss = h.new_zeros(())
            states = [h] if collect else None
            for _ in range(n_steps):
                h, l = self.step(h)
                vq_loss = vq_loss + l
                if collect:
                    states.append(h)
            return (h, vq_loss, states) if collect else (h, vq_loss)

        def forward_ff(self, input_ids, read_pos):
            h = self.encode(input_ids, read_pos)
            for blk in self.stack:
                h = blk(h)
            return h

    return Model().to(device)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=8192,
)
def ballistic_depth(
    tag: str = "smoke",
    arms: str = "base,quant,consist,quant_consist,feedforward",
    steps: int = 15000,
    seed: int = 0,
    eval_max_depth: int = 20,
    eval_cap: int = 2000,
    train_depths: str = "1,2,3,4,5,6",
    lr: float = 3e-4,
    batch_size: int = 256,
    d_model: int = 256,
    codebook_size: int = 4096,
    consist_cycle: float = 1.0,
    consist_reentry: float = 1.0,
    consist_warmup: float = 0.3,
    save_ckpt: bool = False,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from one_layer_deeper.squaring_mod import TOKEN_IDS, TaskSpec, build_trajectories

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
        "codebook_size": codebook_size,
        "consist_cycle": consist_cycle,
        "consist_reentry": consist_reentry,
        "consist_warmup": consist_warmup,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_depths = tuple(int(v) for v in str(cfg["train_depths"]).split(","))
    spec = TaskSpec(
        p=cfg["p"],
        q=cfg["q"],
        train_depths=train_depths,
        eval_depths=tuple(range(1, cfg["eval_max_depth"] + 1)),
        test_fraction=cfg["test_fraction"],
        seed=cfg["seed"],
    )
    info = spec.describe()
    print(f"[task] {json.dumps(info)}")
    if info["depth_first_repeat"] <= spec.max_depth:
        raise ValueError(
            f"depth periodicity leaks into the eval range (first repeat "
            f"{info['depth_first_repeat']} <= max depth {spec.max_depth})"
        )

    data = build_trajectories(spec)
    traj = torch.tensor(data["traj"], device=device)  # [n_units, max_depth+1]
    n_dig = spec.n_answer_digits
    N = spec.modulus

    # --- precomputed token tables, indexed by residue value (0..N-1) -----------------
    def _digits(value, width):
        return [int(c) for c in str(int(value)).rjust(width, "0")]

    n_digits_tbl = torch.tensor(
        [_digits(v, n_dig) for v in range(N)], dtype=torch.long, device=device
    )  # [N, n_dig]
    head_tbl = torch.cat(
        [
            torch.full((N, 1), TOKEN_IDS["BOS"], dtype=torch.long, device=device),
            torch.full((N, 1), TOKEN_IDS["N"], dtype=torch.long, device=device),
            torch.tensor([_digits(N, n_dig)], dtype=torch.long, device=device).expand(N, -1)
            + DIGIT_OFFSET,
            torch.full((N, 1), TOKEN_IDS["X"], dtype=torch.long, device=device),
            n_digits_tbl + DIGIT_OFFSET,
        ],
        dim=1,
    )  # [N, 2 + n_dig + 1 + n_dig]
    X_SLOT = 2 + n_dig + 1
    HEAD_LEN = head_tbl.shape[1]

    def tail_for(depth, include_t):
        if not include_t:
            return torch.tensor([TOKEN_IDS["ANS"]], dtype=torch.long, device=device)
        return torch.tensor(
            [TOKEN_IDS["T"]] + [d + DIGIT_OFFSET for d in _digits(depth, 2)]
            + [TOKEN_IDS["ANS"]],
            dtype=torch.long,
            device=device,
        )

    def prompts_for(values, depth, include_t):
        """values: LongTensor [B] of residues -> [B, max_len] token ids."""
        tail = tail_for(depth, include_t).unsqueeze(0).expand(values.shape[0], -1)
        return torch.cat([head_tbl[values], tail], dim=1)

    arms = [a.strip() for a in str(cfg["arms"]).split(",") if a.strip()]
    results = {"config": cfg, "task": info, "arms": {}}

    train_idx = torch.tensor(data["train_idx"], device=device)
    test_idx = torch.tensor(data["test_idx"], device=device)

    # Held-out-x leaks into the cold-start probe with depth: the residue x_t of a held-out
    # base is often *itself* a base the model trained on, because squaring maps into the
    # 207-element QR subgroup. Record the leak so `coldstart_heldout_x` is never read as a
    # clean generalization number at large t.
    seen_bases = set(traj[train_idx, 0].tolist())
    results_leak = {
        t: float(
            np.mean([v in seen_bases for v in traj[test_idx, t].tolist()])
        )
        for t in range(0, min(spec.max_depth, 20) + 1)
    }
    print(f"[leak] held-out base seen-in-train fraction by t: "
          + " ".join(f"{t}:{v:.2f}" for t, v in results_leak.items()), flush=True)
    results["heldout_base_leak"] = results_leak

    for arm in arms:
        print(f"\n===== arm={arm} seed={cfg['seed']} =====", flush=True)
        include_t = arm == "feedforward"
        recurrent = arm != "feedforward"
        use_consist = arm in ("consist", "quant_consist")
        max_len = HEAD_LEN + (4 if include_t else 1)
        read_pos = max_len - 1  # the ANS position

        model = _make_model(cfg, n_dig, arm, max_len, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{arm}] params={n_params:,} max_len={max_len}", flush=True)

        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: min(1.0, (s + 1) / cfg["warmup"])
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, s / cfg["steps"]))),
        )

        model.train()
        log = []
        for step in range(cfg["steps"]):
            depth = int(np.random.choice(train_depths))
            sel = train_idx[
                torch.randint(0, train_idx.numel(), (cfg["batch_size"],), device=device)
            ]
            x0 = traj[sel, 0]
            digits = n_digits_tbl[traj[sel, depth]]
            ids = prompts_for(x0, depth, include_t)

            if recurrent:
                h0 = model.encode(ids, read_pos)
                hT, vq_loss = model.roll(h0, depth)
            else:
                hT = model.forward_ff(ids, read_pos)
                vq_loss = torch.zeros((), device=device)

            logits = model.dec(hT)
            ce = F.cross_entropy(logits.reshape(-1, 10), digits.reshape(-1))
            loss = ce + vq_loss
            cyc = torch.zeros((), device=device)
            reentry = torch.zeros((), device=device)

            # The consistency terms are self-generated targets, so they must not teach
            # while their own fidelity is below where the system already sits — that is
            # `endo_expansion`'s "a grader is a ceiling, approached from whichever side
            # you start". Hold them off until terminal CE has made real progress.
            consist_on = use_consist and step >= cfg["consist_warmup"] * cfg["steps"]
            if consist_on and depth >= 2:
                t = int(np.random.randint(1, depth))
                h_t, _ = model.roll(h0, t)
                probs = model.dec(h_t).softmax(-1)
                template = prompts_for(
                    torch.zeros_like(x0), depth, include_t
                )  # x-field overwritten below
                h_re = model.enc.forward_soft_digits(template, probs, X_SLOT, read_pos)
                # (a) cycle: the intermediate state must lie in the encoder's range.
                #     Label-free. This is the term that forces re-attraction.
                cyc = F.mse_loss(h_re, h_t.detach()) + F.mse_loss(h_re.detach(), h_t)
                # (b) re-entry: the operator must still reach the answer from a *cleanly
                #     encoded* state, not only from one it drifted into. Reuses the same
                #     terminal label — no new label information enters.
                h_end, vq2 = model.roll(h_re, depth - t)
                reentry = F.cross_entropy(
                    model.dec(h_end).reshape(-1, 10), digits.reshape(-1)
                )
                loss = (
                    loss + cfg["consist_cycle"] * cyc
                    + cfg["consist_reentry"] * reentry + vq2
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()

            if model.quantize and (step + 1) % cfg["dead_code_every"] == 0:
                with torch.no_grad():
                    model.vq.revive_dead(model.op(model.encode(ids, read_pos)))

            if (step + 1) % max(1, cfg["steps"] // 6) == 0:
                acc = (logits.argmax(-1) == digits).all(-1).float().mean().item()
                print(
                    f"  [{arm}] step {step+1:>6} d={depth} ce={ce.item():.4f} "
                    f"exact={acc:.3f} cyc={cyc.item():.4f} re={reentry.item():.4f}",
                    flush=True,
                )
                log.append(
                    dict(step=step + 1, depth=depth, ce=ce.item(), exact=acc,
                         cyc=cyc.item(), reentry=reentry.item())
                )

        # ------------------------------ evaluation ------------------------------
        model.eval()
        arm_res = {"n_params": n_params, "train_log": log}
        cap = cfg["eval_cap"]

        @torch.no_grad()
        def eval_exact(pool):
            pool = pool[:cap]
            out = {}
            for depth in spec.eval_depths:
                ids = prompts_for(traj[pool, 0], depth, include_t)
                digits = n_digits_tbl[traj[pool, depth]]
                if recurrent:
                    h, _ = model.roll(model.encode(ids, read_pos), depth)
                else:
                    h = model.forward_ff(ids, read_pos)
                pred = model.dec(h).argmax(-1)
                out[depth] = (pred == digits).all(-1).float().mean().item()
            return out

        arm_res["exact_seen_x"] = eval_exact(train_idx)
        arm_res["exact_heldout_x"] = eval_exact(test_idx)
        print(
            f"  [{arm}] exact@T (seen x): "
            + " ".join(f"{d}:{v:.3f}" for d, v in arm_res["exact_seen_x"].items()),
            flush=True,
        )

        if recurrent:

            @torch.no_grad()
            def rollout_probe():
                pool = train_idx[:cap]
                h0 = model.encode(prompts_for(traj[pool, 0], 1, include_t), read_pos)
                _, _, states = model.roll(h0, spec.max_depth, collect=True)
                verid, onman, code_use = {}, {}, {}
                for t in range(1, spec.max_depth + 1):
                    truth_val = traj[pool, t]
                    pred = model.dec(states[t]).argmax(-1)
                    verid[t] = (
                        (pred == n_digits_tbl[truth_val]).all(-1).float().mean().item()
                    )
                    h_true = model.encode(
                        prompts_for(truth_val, 1, include_t), read_pos
                    )
                    onman[t] = (
                        F.cosine_similarity(states[t], h_true, dim=-1).mean().item()
                    )
                    if model.quantize:
                        code_use[t] = int(model.vq(states[t])[2].unique().numel())
                return verid, onman, code_use

            v, o, c = rollout_probe()
            arm_res["veridicality"] = v
            arm_res["on_manifold_cos"] = o
            if c:
                arm_res["codes_used"] = c
            print(f"  [{arm}] verid@t: "
                  + " ".join(f"{t}:{x:.3f}" for t, x in v.items()), flush=True)
            print(f"  [{arm}] onman@t: "
                  + " ".join(f"{t}:{x:.3f}" for t, x in o.items()), flush=True)

            @torch.no_grad()
            def amplification(direction, eps_list=(0.01, 0.1, 0.3, 1.0)):
                """Error amplification along `direction` in ('random', 'on_manifold').

                A random direction in d=256 is almost entirely off-manifold, and the model
                is free to squash those for free — so the random readout can understate
                the amplification that actually matters. The on-manifold direction points
                at the encoding of a *different residue*, i.e. the direction in which a
                confusable state actually lies. Calibrating the instrument both ways is
                the same discipline `mjc/expansion/` needed for its rank readout.
                """
                pool = train_idx[: min(cap, 512)]
                h0 = model.encode(prompts_for(traj[pool, 0], 1, include_t), read_pos)
                scale = h0.norm(dim=-1, keepdim=True)
                _, _, base_states = model.roll(h0, spec.max_depth, collect=True)
                if direction == "on_manifold":
                    other = train_idx[
                        torch.randint(0, train_idx.numel(), (pool.numel(),), device=device)
                    ]
                    u = (
                        model.encode(prompts_for(traj[other, 0], 1, include_t), read_pos)
                        - h0
                    )
                else:
                    u = torch.randn_like(h0)
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                out = {}
                for eps in eps_list:
                    _, _, pert = model.roll(
                        h0 + eps * scale * u, spec.max_depth, collect=True
                    )
                    d0 = (pert[0] - base_states[0]).norm(dim=-1).clamp_min(1e-9)
                    out[eps] = {
                        t: ((pert[t] - base_states[t]).norm(dim=-1) / d0).mean().item()
                        for t in range(1, spec.max_depth + 1)
                    }
                return out

            @torch.no_grad()
            def coldstart_probe(pool, targets, grid):
                """Functional re-enterability: can the operator be *restarted*?

                `on_manifold_cos` says the rolled state points the same direction as the
                encoder's representation of the true residue. That is geometry, and
                geometry is a proxy. This is the behavioural version: encode the TRUE
                intermediate residue `x_t` cold — as if it were a fresh problem — roll the
                remaining `T - t` steps, and score exact-match against `x_T`.

                The two outcomes for the base arm mean opposite things. If a cold start at
                t recovers `x_T` where the full rollout to T does not, the operator is a
                sound function of the encoder's states and the failure is *drift* — it
                cannot reach its own inputs. If the cold start fails too, the operator was
                never a function on the encoder's state space at all; it only ever worked
                inside the private trajectory it generates.

                t=0 is the ordinary full rollout, so it double-checks against
                `exact_seen_x[T]` for free.
                """
                pool = pool[:cap]
                out = {}
                for T in targets:
                    row = {}
                    for t in grid:
                        if t > T:
                            continue
                        h = model.encode(prompts_for(traj[pool, t], 1, include_t), read_pos)
                        if T - t > 0:
                            h, _ = model.roll(h, T - t)
                        pred = model.dec(h).argmax(-1)
                        row[t] = (
                            (pred == n_digits_tbl[traj[pool, T]]).all(-1).float().mean().item()
                        )
                    out[T] = row
                return out

            cs_targets = [T for T in (10, 20, 30, 40, 60) if T <= spec.max_depth]
            cs_grid = [
                t for t in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55)
                if t <= spec.max_depth
            ]
            arm_res["coldstart_seen_x"] = coldstart_probe(train_idx, cs_targets, cs_grid)
            arm_res["coldstart_heldout_x"] = coldstart_probe(test_idx, cs_targets, cs_grid)
            for T in cs_targets:
                r = arm_res["coldstart_seen_x"][T]
                print(
                    f"  [{arm}] coldstart T={T}: "
                    + " ".join(f"t{t}:{v:.3f}" for t, v in r.items()),
                    flush=True,
                )

            arm_res["amplification"] = amplification("random")
            arm_res["amplification_on_manifold"] = amplification("on_manifold")
            small = arm_res["amplification"][0.01]
            shown = [t for t in (1, 2, 4, 8, 16, 20) if t in small]
            print(
                f"  [{arm}] amplify@t (eps=0.01): "
                + " ".join(f"{t}:{small[t]:.3g}" for t in shown),
                flush=True,
            )

        results["arms"][arm] = arm_res

        if save_ckpt:
            ck_dir = Path(DATA_DIR) / "ballistic_depth" / str(cfg["tag"]) / "ckpt"
            ck_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "cfg": cfg, "arm": arm},
                ck_dir / f"{arm}_seed{cfg['seed']}.pt",
            )
            print(f"  [{arm}] checkpoint saved", flush=True)

    out_dir = Path(DATA_DIR) / "ballistic_depth" / str(cfg["tag"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{cfg['seed']}.json"
    out_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] {out_path}", flush=True)
    return results


@app.local_entrypoint()
def main(
    tag: str = "smoke",
    arms: str = "base,quant,consist,quant_consist,feedforward",
    steps: int = 15000,
    seeds: str = "0",
    eval_max_depth: int = 20,
    eval_cap: int = 2000,
):
    seed_list = [int(s) for s in seeds.split(",")]
    handles = [
        ballistic_depth.spawn(
            tag=tag,
            arms=arms,
            steps=steps,
            seed=s,
            eval_max_depth=eval_max_depth,
            eval_cap=eval_cap,
        )
        for s in seed_list
    ]
    for s, h in zip(seed_list, handles):
        res = h.get()
        print(f"\n=== seed {s} ===")
        for arm, r in res["arms"].items():
            ex = {int(k): v for k, v in r["exact_seen_x"].items()}
            ood = [v for d, v in ex.items() if d > 6]
            print(
                f"  {arm:>14}  exact@6={ex.get(6, float('nan')):.3f}  "
                f"mean exact@T>6={sum(ood)/max(1, len(ood)):.3f}"
            )
