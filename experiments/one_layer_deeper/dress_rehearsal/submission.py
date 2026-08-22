"""One Layer Deeper submission — a tied operator applied `T` times, with an optional
label-free closure loss, ported from `experiments/one_layer_deeper/ballistic_depth/`.

Two arms, selected by `ARM` below. They share **every parameter of the model**; only the
loss differs, which is how the research node runs its arms and what makes the contrast
readable:

  ARM = "control"  terminal cross-entropy only.
  ARM = "closure"  the same, plus the two label-free consistency terms from
                   `ballistic_depth/` §2 — *cycle* ("the state you rolled into must be one
                   your own encoder could have produced") and *re-entry* ("the operator
                   must still reach the answer from a cleanly re-encoded state"). At fixed
                   `N` those took the composition horizon from T≈13 to T≈51 and held-out-x
                   from 0.001 to 0.32.

## v3: the cycle term re-encodes through the *real* encoder

v2 re-encoded through a separate learned `ReEncoder` tied to the encoder only at the input
residue. That is not the research mechanism and it cannot do the research mechanism's job:
`Enc` itself never saw an intermediate residue, so nothing taught `Enc` to map the residues
the rollout passes through onto states the operator knows — which is exactly why a *fresh*
`x` at test time lands nowhere useful. On the hosted `m6` run that predicted the observed
result precisely (closure's fresh-`x` `T=1` rung sat at the no-reduction floor, 4/140).

v3 does what `ballistic_depth.forward_soft_digits` does: it rebuilds the prompt with the
`x` field replaced by the **soft digit distribution decoded from `h_t`** and runs it through
**the same encoder used at test time**. So the gradient of the cycle term reaches `Enc`, and
`Enc` is trained on intermediate residues.

Two mechanics worth stating:

- **Marker tokens are located by id.** Finding the `[X]` and `[T]` fields needs their token
  ids. This is the same class of knowledge as the `T` parse below — tokenizer layout, used
  for *control flow*, never for output computation. No modular arithmetic is written down;
  the substituted digits are the model's own decode, and every logit still comes from
  learned parameters.
- **Width is matched, not padded.** `x` fields are variable width and the decoded residue
  has its own width, so a canonical zero-padded slot would show `Enc` inputs (leading
  zeros) that never occur at test time. Instead the soft digits are written **in place,
  right-aligned into the `x` field's existing span**, and the cycle and re-entry terms are
  **masked to the rows where the decoded residue's width equals that span's width**. The
  sequence length, padding mask and every position embedding are therefore identical to a
  real prompt, so `Enc`'s input distribution is exactly the test distribution. Typically
  ~70-85% of rows qualify (residues are near-uniform in `[1, N)`, so most share the modal
  width); the rest simply do not contribute the term this step.

## Depth: the operator is applied exactly `T` times

`T` is read from the prompt and the tied operator is iterated that many times, in training
and in evaluation alike. This is the recurrent-depth reading the problem statement asks
for — *"recurrent depth, adaptive computation, extrapolation beyond training depths"* — and
rules 3 and 4 name recurrence and depth curricula as allowed, with rule 4 explicitly not
restricting "computation within a submitted model" and naming "recurrent/iterative
mechanisms" as valid.

The legality reading, stated plainly so it can be checked:

- **What is hard-coded is the tokenizer, not the arithmetic.** Rule 7 forbids a hard-coded
  *algorithm in the forward pass* — "outputs must be produced by the learned model". The
  parse below extracts one integer, the **iteration count**, and nothing else. It performs
  no modular arithmetic, no squaring, no digit manipulation of `N` or `x`, and it never
  touches the output path: every logit is produced by learned parameters from a learned
  state. Removing the parse and looping a constant number of times would change *how many*
  times a learned map is applied, not *what* the map computes.
- **It is control flow, not output computation.** The loop count is an integer index; the
  gradient path from the loss to every parameter that produced the logits is unbroken,
  which is what rule 8 asks. Non-differentiable step counts are what "adaptive computation"
  means everywhere it is used (ACT, halting, TRMs) and are the mechanism the rules
  advertise.
- **The parse uses one public constant.** The dataset config ships `token_ids` and
  `DIGIT_OFFSET`; only `DIGIT_OFFSET` is used here. `T` is the last field of the prompt, so
  the parse walks left from each row's last real token taking digit tokens until it meets a
  non-digit (the field marker), and reads them base-10. No field position, no marker id,
  and nothing about `N` or `x` is assumed.

Everything else about the prompt is read by **learned attention pools**. The only two
places tokenizer knowledge enters are the `T` parse (iteration count) and, in the closure
arm, locating the `x` field so the cycle term can re-encode through the real encoder (v3
above); v1 used a learned rung selector over a fixed 64-step unroll instead of the parse,
and v2 a separate `ReEncoder` instead of the in-place re-encode — both superseded.

## The rest of what the port had to solve

**Answers are right-aligned and their length varies.** The evaluator reads logits at
`target_positions`, the last `len(y)` positions of each row's unpadded input. So the model
decodes a fixed grid of digit distributions, slot `j` meaning "the `j`-th digit from the
right", and writes slot `j` to the position `j` places from that row's last real token.

**`N` varies per example.** A learned query pools a rule vector `r` and re-injects it into
the operator at every step — `variable_modulus/`'s `cond` arm.

**The budget is wall-clock, not steps**, so the learning-rate schedule reads the clock from
`OptimizerSpec.training_time_seconds` rather than assuming a step count.

## Rule notes

- No `torch.load`, no hard-coded weights, no `backward`/`autograd.grad` in this file, no
  nested model or loss calls: the consistency machinery lives inside `forward` and reaches
  the loss through the evaluator's `auxiliary` channel, which is what that channel is for.
  The cycle term's second encoder pass is an ordinary differentiable call to `self.blocks`
  from inside `forward`, not a nested invocation of the model or the loss.
- `self.training` gates *only* the extra consistency computation, the documented sanctioned
  use. The prediction path is identical in train and eval.
- Model state is ~11M elements against the 500M ceiling. Wall clock is what binds.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import (
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
    assert_model_state,
)

# --- arm ------------------------------------------------------------------------------
# `emit_arm.py` rewrites this single line to produce the control file.
ARM = "closure"

# --- architecture ---------------------------------------------------------------------
D_MODEL = 384
N_ENC_LAYERS = 4
N_HEADS = 6
D_FF = 1536
D_OP = 1536

# Tokenizer layout. `DIGIT_OFFSET` is used to count iterations; the two markers are used to
# locate the `x` field for the cycle term's re-encode. Control flow only (see above).
DIGIT_OFFSET = 7
X_MARKER = 3
T_MARKER = 4
MAX_T_DIGITS = 3  # the evaluator's ladder tops out at T=64
MAX_LOOPS = 512  # a safety cap on the iteration count, never reached by the public ladder

# --- optimization ---------------------------------------------------------------------
PEAK_LR = 1e-3
MIN_LR_FRACTION = 0.05
WARMUP_FRACTION = 0.02
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)

# --- closure loss ---------------------------------------------------------------------
# The consistency targets are self-generated, so they must not teach while their own
# fidelity is below where the system already sits (`ballistic_depth` §2). v2 gated that on
# the wall clock alone, which worked on `m6` (the ramp landed after memorisation) and froze
# learning on `m5`/`h1` (it landed mid-plateau). v3 gates on **fitting progress**: the loss
# keeps an EMA of train exact-match, and the ramp starts once that clears
# `EXACT_TRIGGER`, with a clock fallback at `CLOCK_FALLBACK` so a run that never fits still
# gets the term eventually. Both terms are normalised by the state scale, so no weight has
# to track the encoder's norm and switching on cannot shock the loss.
W_CYCLE = 1.0
W_REENTRY = 1.0
EXACT_TRIGGER = 0.5
CLOCK_FALLBACK = 0.6
RAMP_SPAN = 0.15
EXACT_EMA_DECAY = 0.98

# Plain Python state shared between the scheduler, the model and the loss. It is not model
# state: `capture_state_versions` inspects parameters and persistent buffers only, so the
# evaluator's no-mutation check during evaluation is unaffected.
_STATE = {"progress": 0.0, "exact_ema": 0.0, "trigger": None}


def _aux_scale() -> float:
    """Ramp weight for the consistency terms, gated on fitting progress then the clock."""

    if _STATE["trigger"] is None:
        if (
            _STATE["exact_ema"] >= EXACT_TRIGGER
            or _STATE["progress"] >= CLOCK_FALLBACK
        ):
            _STATE["trigger"] = _STATE["progress"]
        else:
            return 0.0
    span = (_STATE["progress"] - _STATE["trigger"]) / RAMP_SPAN
    return max(0.0, min(1.0, span))


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    """Pre-norm bidirectional transformer block over the padded prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, D_FF)
        self.down = nn.Linear(D_FF, D_MODEL)

    def forward(self, x: Tensor, key_padding: Tensor) -> Tensor:
        batch, length, _ = x.shape
        residual = x
        h = self.attention_norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(batch, length, N_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, N_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, N_HEADS, -1).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_padding[:, None, None, :]
        )
        x = residual + self.out(
            attended.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        )
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


class Pool(nn.Module):
    """A learned single-query attention pool over the prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.query = nn.Parameter(torch.zeros(N_HEADS, D_MODEL // N_HEADS))
        nn.init.normal_(self.query, std=0.02)
        self.kv = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, x: Tensor, key_padding: Tensor) -> Tensor:
        batch, length, _ = x.shape
        h = self.norm(x)
        k, v = self.kv(h).chunk(2, dim=-1)
        k = k.view(batch, length, N_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, N_HEADS, -1).transpose(1, 2)
        q = self.query[None, :, None, :].expand(batch, -1, -1, -1)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=key_padding[:, None, None, :]
        )
        return self.out(attended.transpose(1, 2).reshape(batch, D_MODEL))


class Operator(nn.Module):
    """The tied step. `variable_modulus/`'s `cond` arm: the rule is re-injected every step
    rather than only carried in the state."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, D_OP)
        self.rule = nn.Linear(D_MODEL, D_OP, bias=False)
        self.down = nn.Linear(D_OP, D_MODEL)

    def project(self, rule: Tensor) -> Tensor:
        """The rule projection is constant along the rollout, so it is computed once."""

        return self.rule(rule)

    def forward(self, h: Tensor, projected_rule: Tensor) -> Tensor:
        return h + self.down(F.gelu(self.up(self.norm(h)) + projected_rule))


class Model(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.answer_slots = spec.max_seq_len
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        # Reverse position (distance from the row's last real token) is the coordinate the
        # answer lives in, because targets are right-aligned.
        self.reverse_position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_ENC_LAYERS)])
        self.encoder_norm = RMSNorm(D_MODEL)
        self.state_pool = Pool()
        self.rule_pool = Pool()
        self.operator = Operator()
        self.decoder_norm = RMSNorm(D_MODEL)
        self.decoder = nn.Linear(D_MODEL, self.answer_slots * spec.vocab_size)

    # -- the iteration count -------------------------------------------------------
    def _loop_count(self, input_ids: Tensor, key_padding: Tensor) -> Tensor:
        """Read `T` — the number of times to apply the operator — off the prompt tail.

        `T` is the prompt's last field, so walk left from each row's last real token
        taking digit tokens (`id >= DIGIT_OFFSET`) until a non-digit field marker stops
        the run, and read the run base-10. Vectorised over the batch; nothing but the
        loop count comes out of this, and nothing here touches the output path.
        """

        lengths = key_padding.sum(dim=1)
        total = torch.zeros_like(lengths)
        alive = torch.ones_like(lengths, dtype=torch.bool)
        for offset in range(MAX_T_DIGITS):
            index = (lengths - 1 - offset).clamp_min(0)
            token = input_ids.gather(1, index[:, None]).squeeze(1)
            in_range = (lengths - 1 - offset) >= 0
            is_digit = (token >= DIGIT_OFFSET) & in_range & alive
            digit = (token - DIGIT_OFFSET).clamp(0, 9)
            total = total + torch.where(
                is_digit, digit * (10**offset), torch.zeros_like(digit)
            )
            alive = alive & is_digit
        return total.clamp(1, MAX_LOOPS)

    # -- pieces --------------------------------------------------------------------
    def _encode_embeds(self, embeds: Tensor, key_padding: Tensor) -> Tensor:
        """The encoder body, over a token-embedding sequence.

        Split out from `_encode` so the cycle term can feed it a prompt whose `x` field
        carries soft digits — the same parameters, the same positions, the same mask.
        """

        length = embeds.shape[1]
        positions = torch.arange(length, device=embeds.device)
        lengths = key_padding.sum(dim=1, keepdim=True)
        reverse = (lengths - 1 - positions[None, :]).clamp(0, self.config.max_seq_len - 1)
        x = (
            embeds
            + self.position_embedding(positions)[None]
            + self.reverse_position_embedding(reverse)
        )
        for block in self.blocks:
            x = block(x, key_padding)
        return self.encoder_norm(x)

    def _encode(self, input_ids: Tensor, key_padding: Tensor) -> Tensor:
        return self._encode_embeds(self.token_embedding(input_ids), key_padding)

    def _x_field(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Locate the `x` field: its end position (exclusive), its width, and `N`'s width.

        The prompt is `[N] <N digits> [X] <x digits> [T] <T digits>`, so `N`'s field runs
        from position 1 up to the `[X]` marker. `N`'s width bounds any residue's width,
        which is what makes the decoded-width test below well posed.
        """

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        big = input_ids.shape[1] + 1
        x_at = torch.where(input_ids == X_MARKER, positions[None, :], big).min(dim=1).values
        t_at = torch.where(input_ids == T_MARKER, positions[None, :], big).min(dim=1).values
        return t_at, (t_at - x_at - 1).clamp_min(0), (x_at - 1).clamp_min(1)

    def _reencode(
        self,
        input_ids: Tensor,
        key_padding: Tensor,
        digit_probs: Tensor,
        t_at: Tensor,
        width: Tensor,
        n_width: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """`ballistic_depth.forward_soft_digits`, on the evaluator's prompt format.

        Writes the soft digits of a decoded residue into the `x` field **in place and
        right-aligned**, then runs the real encoder. Returns the pooled state and a mask
        of the rows whose decoded width matched the field, which are the only rows the
        cycle and re-entry terms may use.
        """

        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        # Slot k is the k-th digit from the right, so it belongs at position t_at-1-k.
        slot = (t_at[:, None] - 1 - positions[None, :])
        in_field = (slot >= 0) & (slot < width[:, None]) & (slot < self.answer_slots)
        soft = digit_probs @ self.token_embedding.weight
        gathered = soft.gather(
            1, slot.clamp(0, self.answer_slots - 1)[:, :, None].expand(-1, -1, D_MODEL)
        )
        embeds = torch.where(
            in_field[:, :, None], gathered, self.token_embedding(input_ids)
        )
        encoded = self._encode_embeds(embeds, key_padding)

        # Width of the decoded residue: the highest non-zero digit slot, +1. The scan is
        # capped at `N`'s width, because slots above that are never supervised by any row
        # and so carry no meaning. Argmax here is a control decision — which rows qualify —
        # and is not part of any value the gradient flows through.
        tokens = digit_probs.argmax(dim=-1)
        digits = tokens - DIGIT_OFFSET
        slot_index = torch.arange(self.answer_slots, device=input_ids.device)
        in_scan = slot_index[None, :] < n_width[:, None].clamp_max(self.answer_slots)
        nonzero = (digits > 0) & (digits <= 9) & in_scan
        decoded_width = torch.where(
            nonzero, slot_index[None, :], torch.zeros_like(digits)
        ).max(dim=1).values + 1
        valid = (decoded_width == width) & (width > 0)
        return self.state_pool(encoded, key_padding), valid.to(soft.dtype)

    def _roll(self, h: Tensor, rule: Tensor, steps: Tensor) -> Tensor:
        """Apply the tied operator, collecting `h_0 … h_max`, where `max = steps.max()`.

        Only as many steps as the batch actually asks for are ever run, so a batch of
        `T ∈ {2,4,8}` costs eight applications and not a fixed worst case.
        """

        projected = self.operator.project(rule)
        states = [h]
        for _ in range(int(steps.max().item())):
            h = self.operator(h, projected)
            states.append(h)
        return torch.stack(states, dim=1)

    @staticmethod
    def _read(states: Tensor, steps: Tensor) -> Tensor:
        index = steps[:, None, None].expand(-1, 1, states.shape[-1])
        return states.gather(1, index).squeeze(1)

    def _digit_logits(self, h: Tensor) -> Tensor:
        return self.decoder(self.decoder_norm(h)).view(
            h.shape[0], self.answer_slots, self.config.vocab_size
        )

    def _scatter(self, digit_logits: Tensor, key_padding: Tensor) -> Tensor:
        """Slot `j` (the `j`-th digit from the right) -> the position `j` before the row's
        last real token, which is exactly where `target_positions` reads."""

        length = key_padding.shape[1]
        positions = torch.arange(length, device=key_padding.device)
        lengths = key_padding.sum(dim=1, keepdim=True)
        reverse = (lengths - 1 - positions[None, :]).clamp(0, self.answer_slots - 1)
        index = reverse[:, :, None].expand(-1, -1, digit_logits.shape[-1])
        return digit_logits.gather(1, index)

    # -- forward -------------------------------------------------------------------
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, object]:
        if attention_mask is None:
            key_padding = input_ids != 0
        elif attention_mask.dim() == 3:
            key_padding = attention_mask.any(dim=1)
        else:
            key_padding = attention_mask
        key_padding = key_padding.bool()

        steps = self._loop_count(input_ids, key_padding)
        encoded = self._encode(input_ids, key_padding)
        h0 = self.state_pool(encoded, key_padding)
        rule = self.rule_pool(encoded, key_padding)

        states = self._roll(h0, rule, steps)
        read = self._read(states, steps)
        logits = self._scatter(self._digit_logits(read), key_padding)

        if ARM != "closure" or not self.training:
            return logits, None

        ramp = _aux_scale()
        if ramp <= 0.0:
            return logits, {"ramp": 0.0}

        # --- the label-free consistency machinery (training only) -------------------
        # Ordinary differentiable tensor operations inside this forward. Nothing calls the
        # model or the loss recursively and no derivative entry point is touched.
        # Sample an intermediate step `t ∈ [0, T)` and re-enter there.
        span = torch.rand(steps.shape, device=steps.device)
        cut = (span * steps.to(span.dtype)).long().clamp(0, MAX_LOOPS)
        cut = torch.minimum(cut, (steps - 1).clamp_min(0))
        h_t = self._read(states, cut)

        t_at, width, n_width = self._x_field(input_ids)
        reencoded, valid = self._reencode(
            input_ids,
            key_padding,
            self._digit_logits(h_t).softmax(dim=-1),
            t_at,
            width,
            n_width,
        )
        # Re-entry: roll on from the re-encoded state for the steps that remain. Same
        # terminal label, no new label information. The rule is unchanged by the
        # substitution (the `N` field is untouched), so it is reused rather than repooled.
        remaining = steps - cut
        restates = self._roll(reencoded, rule, remaining)
        reentry = self._read(restates, remaining)

        return logits, {
            "ramp": ramp,
            "scale": h0.detach().pow(2).mean().clamp_min(1e-6),
            "h_t": h_t,
            "reencoded": reencoded,
            "valid": valid,
            "reentry_digit_logits": self._digit_logits(reentry),
        }


class ClockSchedule:
    """Wall-clock learning-rate schedule.

    The tier budget is seconds, not steps, so a step-indexed cosine would either anneal too
    early or never anneal at all. `step()` is called by the evaluator after every update;
    it reads the clock.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, model: Model, seconds: float):
        self.optimizer = optimizer
        self.model = model
        self.seconds = max(seconds, 1.0)
        self.started = time.monotonic()
        _STATE.update(progress=0.0, exact_ema=0.0, trigger=None)

    def step(self) -> None:
        progress = min(1.0, (time.monotonic() - self.started) / self.seconds)
        _STATE["progress"] = progress
        if progress < WARMUP_FRACTION:
            scale = progress / WARMUP_FRACTION
        else:
            tail = (progress - WARMUP_FRACTION) / (1.0 - WARMUP_FRACTION)
            scale = MIN_LR_FRACTION + (1.0 - MIN_LR_FRACTION) * 0.5 * (
                1.0 + math.cos(math.pi * tail)
            )
        for group in self.optimizer.param_groups:
            group["lr"] = PEAK_LR * scale


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * WARMUP_FRACTION,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        capturable=spec.device_type == "cuda",
    )
    return OptimizerBundle(
        optimizer,
        scheduler=ClockSchedule(optimizer, model, spec.training_time_seconds),
    )


def _slot_index(batch: TokenLossBatch) -> Tensor:
    """Map each target column to its digit slot (distance from the right end)."""

    counts = batch.valid_mask.sum(dim=1, keepdim=True)
    columns = torch.arange(batch.labels.shape[1], device=batch.labels.device)
    return (counts - 1 - columns[None, :]).clamp_min(0)


def _sequence_cross_entropy(logits: Tensor, batch: TokenLossBatch) -> Tensor:
    valid = batch.valid_mask
    counts = valid.sum(dim=1)
    token_losses = F.cross_entropy(
        logits.transpose(1, 2), batch.labels.clamp_min(0), reduction="none"
    )
    sequence_losses = (token_losses * valid).sum(dim=1) / counts.clamp_min(1)
    return sequence_losses[counts > 0].mean()


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    valid_tokens = batch.valid_mask
    loss = _sequence_cross_entropy(batch.logits, batch)

    # Track how well the model is *fitting*, which is what gates the consistency terms.
    with torch.no_grad():
        rows = valid_tokens.any(dim=1)
        correct = (
            (batch.logits.argmax(dim=-1) == batch.labels) | ~valid_tokens
        ).all(dim=1)[rows]
        if correct.numel():
            exact = float(correct.float().mean().item())
            _STATE["exact_ema"] = (
                EXACT_EMA_DECAY * _STATE["exact_ema"] + (1.0 - EXACT_EMA_DECAY) * exact
            )

    auxiliary = batch.auxiliary
    if not isinstance(auxiliary, dict):
        return loss
    ramp = auxiliary.get("ramp", 0.0)
    if ramp <= 0.0 or "h_t" not in auxiliary:
        return loss

    scale = auxiliary["scale"].float()
    rowmask = auxiliary["valid"].float()
    denominator = rowmask.sum().clamp_min(1.0)

    # (a) cycle: the state you rolled into must be one your own encoder could have
    # produced. Symmetric detach, so gradient reaches both the encoder and the operator.
    h_t = auxiliary["h_t"].float()
    reencoded = auxiliary["reencoded"].float()
    forward_term = (reencoded - h_t.detach()).pow(2).mean(dim=-1)
    backward_term = (reencoded.detach() - h_t).pow(2).mean(dim=-1)
    cycle = ((forward_term + backward_term) * rowmask).sum() / denominator / scale

    # (b) re-entry: the operator must still reach the answer from a cleanly re-encoded
    # state. Reuses the same terminal label — no new label information enters.
    slots = _slot_index(batch)
    reentry_logits = auxiliary["reentry_digit_logits"]
    index = slots[:, :, None].expand(-1, -1, reentry_logits.shape[-1])
    gathered = reentry_logits.gather(1, index)
    token_losses = F.cross_entropy(
        gathered.transpose(1, 2), batch.labels.clamp_min(0), reduction="none"
    )
    counts = valid_tokens.sum(dim=1)
    per_row = (token_losses * valid_tokens).sum(dim=1) / counts.clamp_min(1)
    reentry = (per_row * rowmask).sum() / denominator

    return loss + ramp * (W_CYCLE * cycle + W_REENTRY * reentry)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss if ARM == "closure" else None,
)
