"""Embodied collection — data as a byproduct of behaviour, and the meter that charges for it.

The standing memo this implements is [`COLLECTION_REALISM.md`](COLLECTION_REALISM.md). Every FM in
this node is trained on transitions gathered by TELEPORTATION (`set_state` to an arbitrary
configuration and velocity, apply an arbitrary command, record one isolated triple). That is a
deliberate variable-control choice and it stays the default -- but it is discontinuous,
omnisciently-covering, free, and cheap to measure, and those four properties always travel
together. A body has none of them.

This module supplies the other mode, WITHOUT a second substrate: one plant, one set of knobs, a
`collection_mode` flag on the existing collection helpers. Finished cuts keep `teleport` and stay
byte-identical; new cuts opt in.

WHAT IS ACTUALLY ENFORCED HERE (the part a config flag cannot do)

  * `Body` is the ONLY acquisition interface an on-policy agent gets, and it does not expose
    `set_state`. Teleporting is not "discouraged", it is unrepresentable. This matters because the
    22x measurement subsidy in `ballistic/directed/directed_loop.py` did not happen through
    carelessness -- it happened because `collect_region()` was callable for free.
  * Every `env.step` is charged against a step budget, whatever it was for -- collection, probing,
    monitoring. A survey is an excursion you pay for like any other. `COLLECTION_REALISM.md` §5
    says "print the ratio; above ~1 the experiment is subsidised"; a meter you cannot bypass is
    strictly better than a ratio you must remember to print.
  * Episodes are CONTINUOUS: the state you see next is the one you caused. Transitions come out
    with their episode id and within-episode timestep attached, so nothing downstream can silently
    treat them as i.i.d.

THE TWO FLAGS, NOT ONE

`COLLECTION_REALISM.md` §3 folds the behaviour policy into the `on_policy` enum value and then
says, correctly, that the choice "is the explore/exploit question in disguise". If it is that, it
cannot be a hidden default. So acquisition and behaviour are separate axes, and the behaviour rungs
are enumerated so they can be SWEPT:

    B0  teleport                                    -- unchanged; the coverage ceiling, the honest baseline
    B1  OU random-torque episodes (`OUBehaviour`)   -- continuity WITHOUT the model in the loop
    B2  reaches under a FROZEN reference planner    -- on-task, decoupled from model quality
    B3  reaches under the LIVE FM (`ReachBehaviour`)-- the real phenomenon, incl. bootstrapping

B1/B2/B3 differ only in what drives the trajectory, so the ladder DECOMPOSES on-policy's cost into
(a) the coverage lost to continuity, (b) the alignment gained by being on-task, (c) the model<->data
coupling. Running only B0 vs B3 confounds all three.

WHAT ON-POLICY COSTS THAT IS NOT COVERAGE (the instrument to keep an eye on)

Cut #2's arity result rests on i.i.d. commands killing the received-wisdom confound (max
|corr(u,s)| = 0.003). A goal-directed policy's `u` is a FUNCTION of `s` by construction, so B2/B3
reintroduce exactly that correlation. This is the real biological situation rather than a bug, but
it means part of on-policy's sample-efficiency loss is COMMAND-CHANNEL IDENTIFIABILITY, not state
coverage -- separable by sweeping `sigma_u`. `cmd_state_corr()` below is the diagnostic, and it
should be reported beside coverage in every on-policy run.

WHAT DOES NOT GO THROUGH THE METER

The experimenter's instruments. Normalisation statistics, the task/broad probe sets, evaluation
geometry, the matched-FM ceiling, and Cut #2's interventional counterfactual-`u` sweep all stay
teleport-based and off-budget -- otherwise the two modes are graded in different units and cannot
be compared at all. The agent's ACQUISITION changes; the grader does not.

`import mujoco` lives inside the functions so this module is importable without MuJoCo installed
(Modal submits the app from the laptop) -- the `pusher_env.py` / `arm_env.py` contract.
"""

import numpy as np


class BudgetExhausted(RuntimeError):
    """Raised when a `Body` is asked for more environment steps than it was granted.

    Deliberately an exception rather than a clamp: a loop that silently gets fewer transitions
    than it asked for produces a quietly-wrong sample-efficiency number, which is the exact class
    of error this module exists to prevent.
    """


class Body:
    """B parallel episodes through one plant, metered.

    The acquisition interface for `collection_mode="on_policy"`. Note what is absent: there is no
    `set_state`. An agent holding a `Body` can reset an episode and step it, and that is all.

    PER-EPISODE `MjData`. The established `rollout()` idiom in this directory multiplexes B eval
    episodes through ONE `MjData` by `set_state`-ing each episode's carried state back in every
    step. That is exact for the arm (contacts off, MuJoCo memoryless) but not strictly exact for
    the pusher, where `set_state` drops the constraint solver's warm-start. Since this class is the
    thing that is supposed to make trajectories real, it allocates one `MjData` per parallel
    episode from the shared `MjModel` and swaps `env.data` before stepping. Costs a few KB per
    episode and removes a whole class of subtle contact bug.

    Plant-agnostic: everything is read off `env.model` (`nq`/`nv`/`nu`), so `PusherEnv` and
    `ArmEnv` both work unmodified.
    """

    def __init__(self, env, n_par: int, frame_skip: int, budget: int | None = None,
                 reset_cost: int = 0, wrap_limit: float | None = None):
        import mujoco

        self.env = env
        self.n_par = int(n_par)
        self.fs = int(frame_skip)
        self.budget = budget
        self.reset_cost = int(reset_cost)
        self.wrap_limit = wrap_limit
        self.nq = int(env.model.nq)
        self.nv = int(env.model.nv)
        self.state_dim = self.nq + self.nv
        self.act_dim = int(env.model.nu)
        self._orig_data = env.data
        self._datas = [mujoco.MjData(env.model) for _ in range(self.n_par)]
        self._states = np.zeros((self.n_par, self.state_dim), np.float32)
        self._t = np.zeros(self.n_par, np.int64)          # within-episode step index
        self._ep = np.zeros(self.n_par, np.int64)         # global episode id per slot
        self._next_ep = 0
        self.steps_used = 0
        self.n_resets = 0
        self.n_forced_resets = 0

    # ---------------------------------------------------------------- metering

    @property
    def steps_left(self) -> float:
        return float("inf") if self.budget is None else max(0, self.budget - self.steps_used)

    def _charge(self, k: int):
        if self.budget is not None and self.steps_used + k > self.budget:
            raise BudgetExhausted(
                f"asked for {k} more env steps with {self.steps_left:.0f} left "
                f"(budget={self.budget}, used={self.steps_used})")
        self.steps_used += k

    # ---------------------------------------------------------------- episodes

    def _select(self, b: int):
        self.env.data = self._datas[b]

    def release(self):
        """Give the env its original `MjData` back. Call when done collecting, so any later
        experimenter-side instrument (probe surveys, eval rollouts) runs on the env's own data
        exactly as it did before this module existed."""
        self.env.data = self._orig_data

    def reset(self, rng: np.random.Generator, q_center, q_range: float,
              v0_std: float = 0.0, which=None) -> np.ndarray:
        """Start (or restart) episodes at a fresh posture.

        A reset is the one remaining teleport, and it is honest to say so: a body does not get put
        back at the start. It is charged `reset_cost` steps (default 0) and always COUNTED, so the
        residual subsidy is visible rather than hidden. Long episodes buy fewer resets per budget,
        which is the knob that trades the subsidy away.

        `q_center`/`q_range` govern the EPISODE-START distribution here, not every sample -- which
        is the substantive difference from teleport collection, where they governed both.
        """
        idx = range(self.n_par) if which is None else list(which)
        idx = [int(i) for i in idx]
        if self.reset_cost:
            self._charge(self.reset_cost * len(idx))
        qc = np.asarray(q_center, dtype=np.float64)[:self.nq]
        for b in idx:
            q = qc + rng.uniform(-q_range, q_range, self.nq)
            qd = (rng.normal(0.0, v0_std, self.nv) if v0_std > 0
                  else np.zeros(self.nv, dtype=np.float64))
            self._select(b)
            self.env.set_state(q, qd)        # harness-side, not agent-side: see class docstring
            self._states[b] = self.env.get_state()
            self._t[b] = 0
            self._ep[b] = self._next_ep
            self._next_ep += 1
            self.n_resets += 1
        return self._states.copy()

    def state(self) -> np.ndarray:
        return self._states.copy()

    def step(self, u: np.ndarray):
        """Advance every parallel episode one control interval under command `u` (n_par, act_dim).

        Returns `(S, U, S2, info)` -- the transitions the body just LIVED, one per slot. Charges
        `n_par` steps.
        """
        u = np.atleast_2d(np.asarray(u, dtype=np.float32))
        assert u.shape == (self.n_par, self.act_dim), \
            f"command shape {u.shape} != ({self.n_par}, {self.act_dim})"
        self._charge(self.n_par)
        S = self._states.copy()
        S2 = np.empty_like(S)
        for b in range(self.n_par):
            self._select(b)
            s2, _ = self.env.step(u[b], self.fs)
            S2[b] = s2
        self._states = S2.copy()
        info = {"ep": self._ep.copy(), "t": self._t.copy()}
        self._t += 1
        return S, u.astype(np.float32), S2, info

    def wrapped_slots(self) -> np.ndarray:
        """Which episodes have left the legible (non-wrapping) joint range.

        Teleport collection bounded the operating region through the SAMPLING DISTRIBUTION
        (`arm_env.py:57`), and joints are unlimited by design because a joint limit is a stiff
        constraint = another discontinuity. On-policy that guarantee is gone: an OU torque walk
        will happily spin the arm. Episode length plus this check is what replaces it.
        """
        if self.wrap_limit is None:
            return np.zeros(self.n_par, bool)
        return np.abs(self._states[:, :self.nq]).max(1) > self.wrap_limit

    def nonfinite_slots(self) -> np.ndarray:
        return ~np.isfinite(self._states).all(1)


# ===================================================================== #
# Behaviours -- rungs B1..B3 of the ladder. Each is a callable (s, t) -> u.
# ===================================================================== #


class OUBehaviour:
    """B1 -- correlated random torque. No forward model anywhere in the loop.

    The rung that isolates the cost of CONTINUITY alone: trajectories instead of isolated triples,
    but no on-task alignment and no model<->data coupling. Ornstein-Uhlenbeck rather than i.i.d.
    per-step commands because an i.i.d. torque sequence is a white-noise drive that barely moves
    the body -- the same reason `pusher_env.collect_transitions` (the cut-#1/#2 ancestor of this
    whole idea) used an OU walk.

    Keeps command-state correlation near zero like teleport does, so B1 vs B0 is a clean read on
    coverage with the Cut-#2 identifiability confound held OUT.
    """

    def __init__(self, act_dim: int, n_par: int, rng: np.random.Generator,
                 sigma: float = 0.7, theta: float = 0.15):
        self.act_dim, self.n_par, self.rng = int(act_dim), int(n_par), rng
        self.sigma, self.theta = float(sigma), float(theta)
        self.a = np.zeros((self.n_par, self.act_dim), np.float32)

    def reset(self, which=None):
        if which is None:
            self.a[:] = 0.0
        else:
            self.a[list(which)] = 0.0

    def __call__(self, s, t):
        self.a = (self.a - self.theta * self.a
                  + self.sigma * self.rng.normal(size=self.a.shape)).astype(np.float32)
        return np.clip(self.a, -1.0, 1.0)


class ReachBehaviour:
    """B2 / B3 -- goal-directed reaches plus motor noise. The dumbest defensible embodied default.

    B2 and B3 are THE SAME CLASS with a different `plan_fn`:
      * B2 passes a planner rolling a FROZEN reference FM -- the behaviour is on-task but its
        quality is exogenous, so the data distribution does not depend on the model being trained.
      * B3 passes a planner rolling the LIVE FM -- the real situation, and the one that carries the
        genuine bootstrapping problem (`COLLECTION_REALISM.md` §"The honest cost"): a stale model
        moves badly, which gets you worse data, which keeps it stale.
    Anything measured at B3 is compound; B2 is the control that says whether you are looking at the
    phenomenon or at the spiral.

    `replan_every` defaults to the full episode (ballistic / open-loop), which is this arc's
    regime: you cannot replan faster than sensorimotor delay, and Cut 4b showed a re-grounding
    controller is a near-blind grader of the FM.

    `sigma_u` is NOT decoration. A noiseless goal-directed policy visits a measure-zero tube and its
    commands are a deterministic function of state, so the command channel becomes unidentifiable
    (Cut #2). Motor noise is both the biological reality and the knob that trades on-task-ness
    against command-channel coverage -- sweep it, do not tune it once and forget.
    """

    def __init__(self, plan_fn, goals, rng: np.random.Generator,
                 sigma_u: float = 0.1, replan_every: int | None = None):
        self.plan_fn, self.goals, self.rng = plan_fn, np.asarray(goals, np.float32), rng
        self.sigma_u = float(sigma_u)
        self.replan_every = replan_every
        self.plan = None

    def set_goals(self, goals):
        self.goals = np.asarray(goals, np.float32)
        self.plan = None

    def reset(self, which=None):
        self.plan = None

    def __call__(self, s, t):
        re = self.replan_every
        if self.plan is None or (re is not None and t % re == 0):
            self.plan = self.plan_fn(s, self.goals)
        h = t if self.replan_every is None else (t % self.replan_every)
        h = min(h, self.plan.shape[1] - 1)
        u = self.plan[:, h, :]
        if self.sigma_u > 0:
            u = u + self.sigma_u * self.rng.normal(size=u.shape)
        return np.clip(u, -1.0, 1.0).astype(np.float32)


# ===================================================================== #
# The collector
# ===================================================================== #


def collect_on_policy(env, n: int, rng: np.random.Generator, frame_skip: int,
                      q_center, q_range: float, behaviour, ep_len: int,
                      n_par: int = 8, v0_std: float = 0.0,
                      goal_sampler=None, wrap_limit: float | None = None,
                      reset_cost: int = 0, budget: int | None = None,
                      return_info: bool = False):
    """Collect `n` transitions as the byproduct of `n/(n_par*ep_len)` episodes of behaviour.

    Matched to `collect_pool`'s teleport contract on the only currency that makes the two
    comparable: ONE `env.step` per transition. So "the same 400 steps" means the same thing in both
    modes, excursion cost included, and a Tier-B sample-count correction reads directly as "the
    same 400 steps buy less".

    Returns `(S, U, S2)`, or `(S, U, S2, info)` with `return_info=True`. `info` carries the episode
    ids, within-episode timesteps, step accounting, reset counts and wrap/non-finite diagnostics --
    the things that let a downstream consumer notice these are trajectories rather than a pool.
    """
    body = Body(env, n_par=n_par, frame_skip=frame_skip, budget=budget,
                reset_cost=reset_cost, wrap_limit=wrap_limit)
    S, U, S2, EP, TT = [], [], [], [], []
    got = 0
    try:
        while got < n:
            s = body.reset(rng, q_center, q_range, v0_std=v0_std)
            # Goals are drawn FROM the posture the body actually reset to, not from an
            # independent draw. An earlier build sampled goals from their own `q_jit` prior while
            # the body reset over `q_range`, so the goal a reach was aiming at did not correspond
            # to the posture it was starting from -- reaches that are unreachable or trivial, and
            # a "goal-directed" behaviour that is neither.
            if goal_sampler is not None and hasattr(behaviour, "set_goals"):
                behaviour.set_goals(goal_sampler(s, rng))
            if hasattr(behaviour, "reset"):
                behaviour.reset()
            for t in range(ep_len):
                if got >= n:
                    break
                u = behaviour(s, t)
                sa, ua, sb, info = body.step(u)
                take = min(n_par, n - got)
                S.append(sa[:take]); U.append(ua[:take]); S2.append(sb[:take])
                EP.append(info["ep"][:take]); TT.append(info["t"][:take])
                got += take
                s = sb
                # A wrapped or diverged episode is cut short rather than carried: past the legible
                # range the arm is spinning, and past a non-finite state the transitions are not
                # data at all (`arm_env.step`'s NaN guard exists because a diverged rollout once
                # produced finite-LOOKING control numbers).
                bad = body.wrapped_slots() | body.nonfinite_slots()
                if bad.any():
                    body.n_forced_resets += int(bad.sum())
                    break
    finally:
        body.release()

    S = np.concatenate(S).astype(np.float32) if S else np.zeros((0, body.state_dim), np.float32)
    U = np.concatenate(U).astype(np.float32) if U else np.zeros((0, body.act_dim), np.float32)
    S2 = np.concatenate(S2).astype(np.float32) if S2 else np.zeros((0, body.state_dim), np.float32)
    if not return_info:
        return S, U, S2
    info = {"ep": np.concatenate(EP) if EP else np.zeros(0, np.int64),
            "t": np.concatenate(TT) if TT else np.zeros(0, np.int64),
            "steps_used": body.steps_used, "n_resets": body.n_resets,
            "n_forced_resets": body.n_forced_resets,
            "n_episodes": int(body._next_ep), "ep_len": int(ep_len), "n_par": int(n_par)}
    return S, U, S2, info


# ===================================================================== #
# Goal samplers
# ===================================================================== #


def make_arm_goal_sampler(link_lengths, reach_amp: float, reach_lo: float, reach_hi: float,
                          tries: int = 40, upto: int | None = None):
    """Cartesian reach goals for `ArmEnv`, drawn FROM the postures the body actually reset to.

    Protocol: `sampler(states, rng) -> (m, 2)`, where `states` is the post-reset state batch.
    Taking the ACTUAL start postures rather than an independent prior is what keeps the reach band
    honest. An earlier build of this drew its own `q0` from a `q_jit` prior while the body reset
    over `q_range`, so the goal a reach aimed at did not correspond to the posture it started from
    -- half the reaches were then unreachable or trivial, and a "goal-directed" behaviour that is
    neither is not a rung on the ladder, it is noise with a planner attached.

    Same joint-space-with-rejection construction as `ballistic/arm/arm_readapt.py::eval_geometry`,
    and for the same reason: for a redundant arm a random joint delta often lands in the null space
    and barely moves the tip, silently filling the goal set with trivial reaches. Reused so the
    behaviour's goal distribution matches the evaluation's -- if they differed, "on-task" would be
    measuring the wrong task.
    """
    from mjc.arm_env import fk

    Ls = np.asarray(link_lengths, np.float64)
    n = len(Ls)

    def sample(states, rng: np.random.Generator) -> np.ndarray:
        q0 = np.atleast_2d(np.asarray(states, np.float64))[:, :n]
        t0 = fk(q0, Ls, upto=upto)
        out = np.empty((len(q0), 2), np.float64)
        for i in range(len(q0)):
            best, best_pen = None, np.inf
            for _ in range(tries):
                d = rng.normal(0, 1, n); d /= np.linalg.norm(d)
                cand = q0[i] + reach_amp * d
                dist = float(np.linalg.norm(fk(cand, Ls, upto=upto) - t0[i]))
                pen = max(0.0, reach_lo - dist) + max(0.0, dist - reach_hi)
                if pen < best_pen:
                    best, best_pen = cand, pen
                if pen == 0.0:
                    break
            out[i] = fk(best, Ls, upto=upto)
        return out.astype(np.float32)

    return sample


def make_box_goal_sampler(half_x: float, half_y: float):
    """Uniform Cartesian goals in a box -- the pusher-family default. Same
    `sampler(states, rng)` protocol; ignores the start postures by construction."""

    def sample(states, rng: np.random.Generator) -> np.ndarray:
        m = len(np.atleast_2d(np.asarray(states)))
        return np.stack([rng.uniform(-half_x, half_x, m),
                         rng.uniform(-half_y, half_y, m)], 1).astype(np.float32)

    return sample


# ===================================================================== #
# Diagnostics -- what the flag costs, measured
# ===================================================================== #


def cmd_state_corr(S: np.ndarray, U: np.ndarray) -> dict:
    """Cut #2's confound instrument, pointed at the collection mode itself.

    Cut #2 established arity-beats-resolution UNDER I.I.D. COMMANDS, explicitly killing the
    received-wisdom confound with max |corr(u, s)| = 0.003. Teleport collection preserves that by
    construction. A goal-directed behaviour policy does not: its `u` is a function of `s`. So part
    of on-policy's sample-efficiency loss is command-channel IDENTIFIABILITY rather than state
    coverage, and this is the number that separates them. Report it beside coverage, always.
    """
    if len(S) < 3:
        return {"max_abs_corr": float("nan"), "mean_abs_corr": float("nan")}
    X = np.concatenate([S, U], 1).astype(np.float64)
    sd = X.std(0)
    keep = sd > 1e-9
    C = np.corrcoef(X[:, keep], rowvar=False)
    ns = int(keep[:S.shape[1]].sum())
    block = np.abs(C[:ns, ns:])                    # state-rows x command-columns
    if block.size == 0:
        return {"max_abs_corr": float("nan"), "mean_abs_corr": float("nan")}
    return {"max_abs_corr": float(block.max()), "mean_abs_corr": float(block.mean())}


def coverage_stats(S: np.ndarray, U: np.ndarray, refS: np.ndarray, refU: np.ndarray,
                   n_ref: int = 2000, seed: int = 0) -> dict:
    """How much of the world a pool actually covers, measured against a fixed teleport reference.

    Two directions, and they answer different questions:
      * `ref_to_pool` -- for each reference point, distance to the nearest collected point. This is
        COVERAGE of the workspace: large means there is a lot of world this pool never went near.
        It is the quantity teleport collection hands over for free and the one on-policy must pay
        for with excursions.
      * `pool_to_ref` -- the reverse. Large means the pool went places the reference distribution
        does not reach, i.e. the two modes are not sampling the same region at all (expected: an
        on-policy reach runs to ~15 rad/s while `v_explore=8` -- `arm_substrate` P5).

    Distances are in units of the REFERENCE pool's per-dimension std, so the two modes are scored
    on one ruler. Sub-sampled to `n_ref` points; this is a diagnostic, not a headline.
    """
    if len(S) == 0 or len(refS) == 0:
        return {}
    rng = np.random.default_rng(seed)
    A = np.concatenate([S, U], 1).astype(np.float64)
    R = np.concatenate([refS, refU], 1).astype(np.float64)
    sd = R.std(0) + 1e-9
    A, R = A / sd, R / sd
    ia = rng.permutation(len(A))[:n_ref]
    ir = rng.permutation(len(R))[:n_ref]
    A, R = A[ia], R[ir]

    def nn(P, Q):                                   # min distance from each P to any Q
        d = np.sqrt(np.maximum(
            (P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None, :] - 2.0 * P @ Q.T, 0.0))
        return d.min(1)

    r2p, p2r = nn(R, A), nn(A, R)
    return {"ref_to_pool_mean": float(r2p.mean()), "ref_to_pool_p90": float(np.percentile(r2p, 90)),
            "pool_to_ref_mean": float(p2r.mean()), "pool_to_ref_p90": float(np.percentile(p2r, 90)),
            "n_compared": int(len(A))}


def pool_diagnostics(S: np.ndarray, U: np.ndarray, nq: int,
                     refS: np.ndarray | None = None, refU: np.ndarray | None = None,
                     info: dict | None = None, seed: int = 0) -> dict:
    """The full per-mode diagnostic block: identifiability, coverage, velocity, excursion."""
    d = {"n": int(len(S))}
    d.update(cmd_state_corr(S, U))
    if refS is not None:
        d.update(coverage_stats(S, U, refS, refU, seed=seed))
    if len(S):
        qd = S[:, nq:]
        speed = np.linalg.norm(qd, axis=1)
        d["speed_mean"] = float(speed.mean())
        d["speed_p95"] = float(np.percentile(speed, 95))
        d["max_absq"] = float(np.abs(S[:, :nq]).max())
    if info:
        d.update({k: info[k] for k in
                  ("steps_used", "n_resets", "n_forced_resets", "n_episodes") if k in info})
        if info.get("n_resets"):
            d["transitions_per_reset"] = float(len(S) / info["n_resets"])
    return d
