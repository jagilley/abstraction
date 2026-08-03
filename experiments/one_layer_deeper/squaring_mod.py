"""The repeated modular-squaring task, vendored from tilde-research/one-layer-deeper.

Upstream: `data/squaring_mod.py` and `data/counting.py` in that repo. We vendor the
*task semantics and tokenization* verbatim (token ids, decimal digit encoding, field
markers, the trapdoor label computation) so our results stay addressable against the
real benchmark, and drop the ~900 lines of split machinery we do not use.

Two deliberate departures, both stated here because they are the kind of thing that
silently confounds a depth experiment:

1. **We emit the full trajectory** `x_0, x_1, ..., x_T`, not just the terminal answer.
   The evaluator never gives you this. It is *research-only instrumentation* — no arm
   ever trains on an intermediate residue — and it is what makes latent veridicality
   measurable at every rollout step (the analogue of `_rollout_fidelity` in
   `rhm/rhm_sculpt_twofm.py`).

2. **Fixed-width, zero-padded answers.** Upstream emits `number_tokens(result)`, whose
   length varies with the residue (1..4 digits for our modulus), so exact-match tangles
   answer-*length* prediction with answer-*value* prediction. We pad to `len(str(N))`
   digits and score exact match on the decoded integer, which is equivalent on value and
   removes a nuisance variable from the depth question. `tokenize_prompt` still emits the
   upstream variable-length prompt encoding.

The task: given modulus `N`, base `x`, and step count `T`, return `x^(2^T) mod N`.
Without the factorization of `N` the best known general method is `T` serial squarings.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd, lcm

import numpy as np

# --- vendored verbatim from upstream data/squaring_mod.py -------------------------

TOKEN_IDS: dict[str, int] = {
    "PAD": 0,
    "BOS": 1,
    "N": 2,
    "X": 3,
    "T": 4,
    "ANS": 5,
    "EOS": 6,
}
DIGIT_OFFSET = 7
NUM_DIGITS = 10
VOCAB_SIZE = DIGIT_OFFSET + NUM_DIGITS  # 17


def digit_token(digit: int) -> int:
    return DIGIT_OFFSET + digit


def number_tokens(value: int) -> list[int]:
    return [digit_token(int(ch)) for ch in str(value)]


def trapdoor_squaring_mod(x: int, time_steps: int, p: int, q: int) -> int:
    """Upstream's exact-label shortcut: reduce the exponent mod phi(N)."""
    if time_steps < 0:
        raise ValueError("time_steps must be non-negative")
    modulus = p * q
    phi = (p - 1) * (q - 1)
    exponent = pow(2, time_steps, phi)
    return pow(x, exponent, modulus)


# --- our additions ----------------------------------------------------------------


def depth_first_repeat(p: int, q: int, limit: int = 20000) -> tuple[int, int]:
    """(tail, period) of `2^T mod lambda(N)`.

    `x^(2^T)` repeats in T exactly when `2^T mod lambda(N)` does, so `tail + period` is
    the first depth at which the task admits a *periodicity* shortcut instead of a serial
    rollout. Any depth range we evaluate on must sit strictly below it.
    """
    lam = lcm(p - 1, q - 1)
    seen: dict[int, int] = {}
    value = 1
    for step in range(limit):
        if value in seen:
            return seen[value], step - seen[value]
        seen[value] = step
        value = (value * 2) % lam
    raise ValueError("no cycle found within limit")


@dataclass(frozen=True)
class TaskSpec:
    """A fixed-modulus repeated-squaring task.

    Defaults: N = 9853 = 59 * 167. Chosen so that (a) the first depth-repeat is at
    T = 1149, far above any depth we evaluate, so periodicity is never a shortcut, and
    (b) squaring is a *bijection* on the 2407-element quadratic-residue subgroup, so for
    every t >= 1 the state lives in one fixed finite set and perfect attractor structure
    is available to a model that can find it.
    """

    p: int = 59
    q: int = 167
    train_depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    eval_depths: tuple[int, ...] = tuple(range(1, 21))
    test_fraction: float = 0.2
    seed: int = 45

    @property
    def modulus(self) -> int:
        return self.p * self.q

    @property
    def n_answer_digits(self) -> int:
        return len(str(self.modulus - 1))

    @property
    def max_depth(self) -> int:
        return max(self.eval_depths)

    def units(self) -> np.ndarray:
        n = self.modulus
        return np.array([x for x in range(1, n) if gcd(x, n) == 1], dtype=np.int64)

    def describe(self) -> dict:
        tail, period = depth_first_repeat(self.p, self.q)
        units = self.units()
        reachable = {pow(int(v), 2, self.modulus) for v in units}
        return {
            "modulus": self.modulus,
            "p": self.p,
            "q": self.q,
            "n_units": int(units.size),
            "reachable_states_depth_ge_1": len(reachable),
            "depth_first_repeat": tail + period,
            "depth_cycle_tail": tail,
            "depth_cycle_period": period,
            "n_answer_digits": self.n_answer_digits,
            "train_depths": list(self.train_depths),
            "eval_depths": list(self.eval_depths),
        }


def _primes_upto(limit: int) -> list[int]:
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i, ok in enumerate(sieve) if ok]


@dataclass(frozen=True)
class ModulusFamily:
    """A *family* of moduli — the arity-2 DGP.

    Fixed-`N` never forces the operator to be conditioned on the rule: `x -> x^2 mod N` is
    one map, and a tied block can simply *be* that map. Sampling `N` makes the operator
    arity-2, `F(h, N)`, so a rule-blind operator can only predict the `N`-averaged next
    state. This is the modular-squaring analogue of `mjc/arity_torque`'s command-blind
    forward model.

    Three constraints define the family, and each is load-bearing:

    1. **Uniform digit width** (`n_digits=4`, so `1000 <= N <= 9999`). The answer width is
       then constant across the family, so exact-match never tangles answer *length* with
       answer *value* — the same reason `answer_digits` zero-pads in the fixed-`N` cut.
    2. **`depth_first_repeat(p, q) > min_margin` for every member.** `x^(2^T)` is eventually
       periodic in `T`; one leaky modulus in the family would let a model pass depth
       extrapolation on those examples by discovering a cycle instead of iterating.
    3. **`min(p, q) >= min_factor`.** Semiprimes with a tiny factor (e.g. `3 x 347`) have a
       degenerate unit group and a far smaller effective state space, so excluding them
       keeps difficulty roughly homogeneous across the family.

    Defaults give **325 moduli** spanning `N = 1081..9983`, minimum margin 61, median 110.
    """

    n_digits: int = 4
    min_margin: int = 60
    min_factor: int = 11
    heldout_fraction: float = 0.2
    seed: int = 45
    # Optional narrower band inside the digit width. Upstream's own *variable* tiers use
    # 10-11 bit moduli (Easy `e5`) and 12/14/16 bit (Medium `m5`); `(1024, 2047)` is Easy
    # `e5`'s band. Narrowing trades rule diversity — the arity axis — for an easier
    # one-step map. `None` means the full digit-width range.
    mod_lo: int | None = None
    mod_hi: int | None = None

    @property
    def lo(self) -> int:
        return self.mod_lo if self.mod_lo is not None else 10 ** (self.n_digits - 1)

    @property
    def hi(self) -> int:
        return self.mod_hi if self.mod_hi is not None else 10**self.n_digits - 1

    def members(self) -> list[tuple[int, int, int, int]]:
        """Sorted `(N, p, q, first_depth_repeat)` for every member of the family."""
        primes = [p for p in _primes_upto(self.hi // self.min_factor) if p >= self.min_factor]
        out = []
        for i, p in enumerate(primes):
            for q in primes[i + 1 :]:
                n = p * q
                if n > self.hi:
                    break
                if n < self.lo:
                    continue
                tail, period = depth_first_repeat(p, q)
                if tail + period > self.min_margin:
                    out.append((n, p, q, tail + period))
        return sorted(out)

    def split(self) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        """Train / held-out moduli. The held-out set is the arity generalisation axis."""
        members = self.members()
        perm = np.random.default_rng(self.seed).permutation(len(members))
        n_held = int(round(self.heldout_fraction * len(members)))
        held = sorted(members[i] for i in perm[:n_held])
        train = sorted(members[i] for i in perm[n_held:])
        return train, held

    @staticmethod
    def units_for(p: int, q: int) -> np.ndarray:
        """`x` coprime to `N = pq`, i.e. not divisible by `p` or `q`.

        Non-units are excluded because their orbits can repeat in depth *earlier* than
        `depth_first_repeat`, which is derived from `lambda(N)` on the unit group — a
        non-unit would be a periodicity leak the family assertion does not cover.
        """
        n = p * q
        x = np.arange(1, n, dtype=np.int64)
        return x[(x % p != 0) & (x % q != 0)]

    def describe(self) -> dict:
        train, held = self.split()
        allm = train + held
        margins = [m[3] for m in allm]
        return {
            "n_digits": self.n_digits,
            "min_margin": self.min_margin,
            "min_factor": self.min_factor,
            "n_moduli": len(allm),
            "n_train_moduli": len(train),
            "n_heldout_moduli": len(held),
            "modulus_min": min(m[0] for m in allm),
            "modulus_max": max(m[0] for m in allm),
            "depth_first_repeat_min": min(margins),
            "depth_first_repeat_median": int(np.median(margins)),
            "train_moduli": [m[0] for m in train],
            "heldout_moduli": [m[0] for m in held],
        }


def tokenize_prompt(modulus: int, x: int, time_steps: int | None) -> list[int]:
    """Upstream's prompt encoding. `time_steps=None` omits the T field entirely.

    Recurrent arms take T as the loop count and must *not* also see it in the prompt, or
    the encoder could fold depth into the initial state and the rollout would stop being
    the only route to depth. The non-recurrent control gets the full prompt including T.
    """
    ids = [TOKEN_IDS["BOS"], TOKEN_IDS["N"]]
    ids.extend(number_tokens(modulus))
    ids.append(TOKEN_IDS["X"])
    ids.extend(number_tokens(x))
    if time_steps is not None:
        ids.append(TOKEN_IDS["T"])
        ids.extend(number_tokens(time_steps))
    ids.append(TOKEN_IDS["ANS"])
    return ids


def answer_digits(value: int, n_digits: int) -> list[int]:
    """Zero-padded fixed-width decimal digits, most significant first."""
    text = str(value).rjust(n_digits, "0")
    return [int(ch) for ch in text]


def build_trajectories(spec: TaskSpec) -> dict[str, np.ndarray]:
    """Full squaring trajectories for every unit, out to `spec.max_depth`.

    Returns `x_traj[i, t] = x_i^(2^t) mod N` for t = 0..max_depth, plus the train/test
    split over *base values* (so held-out x and held-out depth are separable axes).
    """
    n = spec.modulus
    units = spec.units()
    depths = spec.max_depth
    traj = np.zeros((units.size, depths + 1), dtype=np.int64)
    traj[:, 0] = units
    for t in range(1, depths + 1):
        prev = traj[:, t - 1]
        traj[:, t] = (prev * prev) % n

    # Cross-check every column against the independent trapdoor label path.
    rng = np.random.default_rng(spec.seed)
    for t in rng.choice(depths + 1, size=min(6, depths + 1), replace=False):
        i = int(rng.integers(0, units.size))
        expected = trapdoor_squaring_mod(int(units[i]), int(t), spec.p, spec.q)
        if int(traj[i, t]) != expected:
            raise AssertionError(f"trajectory mismatch at t={t}: {traj[i, t]} != {expected}")

    perm = np.random.default_rng(spec.seed).permutation(units.size)
    n_test = int(round(spec.test_fraction * units.size))
    return {
        "units": units,
        "traj": traj,
        "test_idx": np.sort(perm[:n_test]),
        "train_idx": np.sort(perm[n_test:]),
    }


def encode_split(
    spec: TaskSpec,
    traj: np.ndarray,
    idx: np.ndarray,
    depths: tuple[int, ...],
    *,
    include_t_in_prompt: bool,
) -> dict[str, np.ndarray]:
    """Materialize (prompt, depth, answer, trajectory) tensors for `idx` x `depths`."""
    n_dig = spec.n_answer_digits
    prompts, depth_col, answers, traj_col = [], [], [], []
    for t in depths:
        for i in idx:
            x0 = int(traj[i, 0])
            prompts.append(
                tokenize_prompt(spec.modulus, x0, t if include_t_in_prompt else None)
            )
            depth_col.append(t)
            answers.append(answer_digits(int(traj[i, t]), n_dig))
            traj_col.append(traj[i])
    max_len = max(len(p) for p in prompts)
    input_ids = np.full((len(prompts), max_len), TOKEN_IDS["PAD"], dtype=np.int64)
    attn = np.zeros((len(prompts), max_len), dtype=bool)
    for r, p in enumerate(prompts):
        input_ids[r, : len(p)] = p
        attn[r, : len(p)] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "depth": np.array(depth_col, dtype=np.int64),
        "answer": np.array(answers, dtype=np.int64),
        "traj": np.stack(traj_col),
    }
