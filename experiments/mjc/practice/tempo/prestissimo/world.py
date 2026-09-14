"""The prestissimo world: `solo/world.py`, FORKED VERBATIM, made PIECE-PARAMETERISED, plus the
dyadic level ladder.

THE FORK IS EXACT WHERE IT MATTERS, and this node can prove it: `World` now takes a
`piece.Piece` instead of reading a module of constants, so the DONOR square and the FAST figure
run through the *same* code. Gate **P-F1** runs this file on `piece.donor_piece()` and asserts it
reproduces `acappella/b1`'s library build and its Delta = 0 and Delta = 8 rows at max|delta| = 0
— i.e. the fork is byte-exact and **the piece is the only variable**. Everything above the line
"WHAT PRESTISSIMO ADDS" is `solo/world.py` with the constants replaced by `self.piece.<x>` and
two additive, inert-at-`k0 = 0` recordings (the approach ledger split and the drilled-only piece
error).

WHAT PRESTISSIMO ADDS (all at the bottom of the file)
    * `LadderLayout` — stable global slot ids for a DYADIC ladder: a level-l cell is (l, k) with
      k a drilled seam and k = 0 mod 2^(l-1), spanning 2^(l-1) segments. `offbook/SlotLayout`'s
      contract (fixed width per cell, ids that mean the same thing at cycle 1 and cycle N, prim
      ids at the end) with `levels(k)` generalised from {segment, tail-chain} to the ladder.
    * `LevelLibrary` — cells of SLOTS of MEMBERS, at every level. A level-1 slot is a k-means
      partition over content (solo's `MemberLibrary`, unchanged in construction); a level-l slot
      for l > 1 **is a pair of committed level-(l-1) slots**, `T[l] subset T[l-1] x T[l-1]`, and
      its members are the auditioned WELDS of its parents' members. That makes the level's
      identity literally its spelling, which is what gives `native/`'s can't-decompose readout a
      form on this substrate for the first time (`solo` finding 9: at segment span it had none).
    * `build_level1_cell` — solo's `build_member_cell`, verbatim in op.
    * `build_level_cell` — the NESTING OP: weld, audition on the configuration that will deploy
      the unit, keep the best pairs. One feedback event at launch and the internal seam's
      feedback dropped: that is what the level buys.
    * `LadderDecider` — the seam decision for every Phase-A arm: the approach plays the
      primitive, a drilled seam chooses among the levels the arm is allowed, by frozen key or by
      plant audition.

The original donor docstrings follow.

The solo world: `acappella/world.py`, FORKED VERBATIM, plus the three things this node adds.

THE FORK IS EXACT WHERE IT MATTERS. Everything below the line "what solo adds" is untouched
donor code: the rollout pool, the ledger, `World.traverse` (with two ADDITIVE recording flags
and one additive per-performer `prim` mask that are inert when unused), `reflex_cmd`,
`plant_cem`, the deciders, the two-level `Library` and `select_tapes`. Gate S-F1 asserts that
the fork reproduces `acappella/b1`'s library build and its Delta = 0 and Delta = 8 rows at
max|delta| = 0.

WHAT SOLO ADDS (all at the bottom of the file)
    * `TrunkNet` / `clone_trunk` — the behaviour-cloned reflex. The learner's own primitive
      executor, cloned from the reflex law's OWN closed-loop traversals (delayed observation,
      seam index, phase within segment -> command), on the `ballistic_bc` / `clone_policy`
      pattern. This is the trunk Port 2 reads and the primitive every seam may choose. There is
      no forward model in it: it predicts no state, only what the reflex law would command.
    * `RolloutPool.run_cl` / `World.rollout_trunk` — a CLOSED-LOOP grounding: the trunk run on
      resettable copies of the plant, with the same observation delay applied INSIDE the trial.
      A tape is auditioned by executing it; so is the primitive.
    * `MemberLibrary` / `build_member_cell` — slots with MEMBERS (offbook decision 3: slots fit
      once over content and frozen), which is what gives Port 2 a within-slot choice to replace.
    * `SoloDecider` — the seam decision for every arm and every battery row.

The original donor docstring follows.

THE A CAPPELLA WORLD: `etude/`'s piece and metering, with **no forward model anywhere**.

WHAT IS FORKED, VERBATIM
    The puck-free corridor world (`../../pusher_env.py`, gear 10, damping 2.0, frame_skip 12), the
    closed square loop of four 34-step segments with the localized command rotation on segment 1,
    boundaries re-grounded from the true achieved state, metering by boundary error at performance
    tempo, `d_fb = 0.10 s`, and motor noise on practice. Constants live in `piece.py` and gate A-F
    asserts them against the donor's own source.

WHAT IS REMOVED
    The forward model. There is no `f(s, u)`, no CEM-in-imagination, no BC policy, no span head.
    Everything the agent knows about the plant it knows by having executed on it.

WHAT REPLACES IT — THE THREE THINGS THAT CAN PRODUCE A COMMAND
    1. THE REFLEX LAW (`reflex`). A fixed PD law tracking the waypoint schedule, gains calibrated
       ONCE on the CLEAN world (the world without the rotation region — the model-free analogue of
       étude's `pretrain_mode=exclude`: the hard passage is *unmodelled*, not wrongly modelled).
       It plans nothing and pays one feedback event per control step. This is "closed-loop by feel"
       in its purest form, and it is also the source of the renditions the library is selected from
       (§3 of `practice_manufactures_its_own_credit`: compilation is self-imitation of one's own
       traces). Zero groundings.
    2. THE PRICED-ROLLOUT SEARCH (`search`). CEM whose candidate command sequences are EXECUTED ON
       A RESETTABLE COPY OF THE PLANT from the realised state — the motor analogue of RHM's
       materialise-and-re-encode. Every rollout is a GROUNDING and is priced. A declared
       per-decision budget G is the economy (`ratchet/`'s G = 58).
       It returns the best sequence it ACTUALLY ROLLED OUT, not the elite mean: on this substrate
       the mean of valid command sequences is not a valid sequence (étude finding 2, 2.1x), and a
       real-rollout search has no reason to average when it can select. The elite-mean's realised
       error is logged as an instrument in the budget gate.
    3. A LIBRARY TAPE (`tape`). Stored executed commands, selected on consumption-distribution
       states, played open-loop. One grounding to audition, zero to play.

THE LEDGER. Three costs in one currency (seconds), each also logged raw so the price surface can
be swept post hoc: execution (`steps x dt_ctrl`), feedback (`n_fb x d_fb`), and grounding
(`ground_steps x dt_ctrl + n_ground x d_fb` — étude's own `t_score` pricing of its audition
rollouts, generalised). Per-performer, per-traversal.

WHY THE ROLLOUT IS EXACT, AND WHY THAT IS NOT A CHEAT. The plant is deterministic and the rollout
is the plant, so a seam-time audition here predicts its own consumption exactly — as RHM's
materialise-and-re-encode does. What limits the search is the BUDGET (a 34x2 continuous sequence is
a 68-dim search space), not model error. The non-trivial calibration is therefore the
LIBRARY-CONSTRUCTION audition, which scores candidates on held-out hand-over states and consumes
them elsewhere — étude's winner's curse and seam-state shift, which do not go away.
"""

import numpy as np

from mjc.practice.tempo.prestissimo import piece as P

# --------------------------------------------------------------------------- #
# the rollout pool: a grounding is a real trial in the world
# --------------------------------------------------------------------------- #

_ENV = None
_FS = P.FRAME_SKIP


def _pool_init(dgp, frame_skip):
    global _ENV, _FS
    from mjc.pusher_env import PusherEnv
    _ENV = PusherEnv(dgp, with_puck=False)
    _FS = int(frame_skip)


def _pool_rollout(payload):
    """(s0 (n,4), cmds (n,L,2), cps [(step, gx, gy), ...]) -> (errs (n,ncp), final (n,4)).

    No RNG inside: a rollout is a deterministic function of (state, commands), so chunking across
    processes is bit-identical to running them in one. That is what licenses the pool at all.
    """
    s0, cmds, cps = payload
    n, L = cmds.shape[0], cmds.shape[1]
    errs = np.empty((n, len(cps)), np.float64)
    fin = np.empty((n, 4), np.float64)
    for i in range(n):
        _ENV.set_state(s0[i, :2].astype(np.float64), s0[i, 2:].astype(np.float64))
        c = 0
        for h in range(L):
            _ENV.step(cmds[i, h], _FS)
            if c < len(cps) and h == cps[c][0]:
                st = _ENV.get_state()
                errs[i, c] = np.hypot(st[0] - cps[c][1], st[1] - cps[c][2])
                c += 1
        fin[i] = _ENV.get_state()
    return errs, fin


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _trunk_forward(w, s, k, f, n_seg):
    """The cloned policy's forward pass in numpy, so a pool worker needs no torch.
    Bit-identical in structure to `TrunkNet.forward`; asserted against it by gate S-T0."""
    n = len(s)
    x = np.zeros((n, 4 + n_seg + 1), np.float64)
    x[:, :4] = s
    x[:, 4 + int(k)] = 1.0
    x[:, -1] = f
    x = (x - w["mu"]) / w["sd"]
    for W, b in w["hid"]:
        x = _silu(x @ W.T + b)
    return np.clip(x @ w["out"][0].T + w["out"][1], -1.0, 1.0)


def _pool_rollout_cl(payload):
    """A CLOSED-LOOP grounding: run the cloned trunk on the plant copy from `s0`, with the
    observation delay applied inside the trial. No RNG inside, so chunking is bit-identical."""
    s0, k, L, cps, w, D, n_seg, H = payload
    n = len(s0)
    errs = np.empty((n, len(cps)), np.float64)
    fin = np.empty((n, 4), np.float64)
    for i in range(n):
        st = np.asarray(s0[i], np.float64)
        _ENV.set_state(st[:2], st[2:])
        hist = [st.copy()]
        c = 0
        for h in range(L):
            kk = int(k) + h // H
            o = hist[max(0, len(hist) - 1 - int(D))]
            u = _trunk_forward(w, o[None, :], kk, (h % H + 1) / float(H), n_seg)[0]
            s2, _ = _ENV.step(u.astype(np.float32), _FS)
            hist.append(np.asarray(s2, np.float64))
            if c < len(cps) and h == cps[c][0]:
                errs[i, c] = np.hypot(s2[0] - cps[c][1], s2[1] - cps[c][2])
                c += 1
        fin[i] = hist[-1]
    return errs, fin


class RolloutPool:
    """`n_proc` worker processes, each with its own `PusherEnv`. MuJoCo is single-threaded per env
    and `_apply_rot_regions` runs per physics substep in Python, so threads do not scale (measured:
    8 threads are 3x SLOWER than 1 — `profile_cost.py`); processes scale ~10.5x on 16."""

    def __init__(self, dgp, frame_skip, n_proc):
        self.n_proc = int(n_proc)
        self.dgp, self.fs = dgp, int(frame_skip)
        self._ex = None
        if self.n_proc > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            self._ex = cf.ProcessPoolExecutor(
                max_workers=self.n_proc, mp_context=mp.get_context("fork"),
                initializer=_pool_init, initargs=(dgp, frame_skip))
        else:
            _pool_init(dgp, frame_skip)

    def run(self, s0, cmds, cps):
        s0 = np.ascontiguousarray(s0, np.float32)
        cmds = np.ascontiguousarray(cmds, np.float32)
        cps = [(int(a), float(b), float(c)) for a, b, c in cps]
        n = len(s0)
        if n == 0:
            return np.zeros((0, len(cps))), np.zeros((0, 4))
        if self._ex is None:
            return _pool_rollout((s0, cmds, cps))
        nb = min(self.n_proc, n)
        bnds = np.linspace(0, n, nb + 1).astype(int)
        parts = [(s0[a:b], cmds[a:b], cps) for a, b in zip(bnds[:-1], bnds[1:]) if b > a]
        res = list(self._ex.map(_pool_rollout, parts))
        return (np.concatenate([r[0] for r in res], 0),
                np.concatenate([r[1] for r in res], 0))

    def run_cl(self, s0, k, L, cps, w, D, n_seg, H):
        """The closed-loop counterpart of `run`, for the trunk (the primitive)."""
        s0 = np.ascontiguousarray(s0, np.float32)
        cps = [(int(a), float(b), float(c)) for a, b, c in cps]
        n = len(s0)
        if n == 0:
            return np.zeros((0, len(cps))), np.zeros((0, 4))
        if self._ex is None:
            return _pool_rollout_cl((s0, k, L, cps, w, D, n_seg, H))
        nb = min(self.n_proc, n)
        bnds = np.linspace(0, n, nb + 1).astype(int)
        parts = [(s0[a:b], k, L, cps, w, D, n_seg, H)
                 for a, b in zip(bnds[:-1], bnds[1:]) if b > a]
        res = list(self._ex.map(_pool_rollout_cl, parts))
        return (np.concatenate([r[0] for r in res], 0),
                np.concatenate([r[1] for r in res], 0))

    def close(self):
        if self._ex is not None:
            self._ex.shutdown(wait=True)
            self._ex = None


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

class Ledger:
    """Raw totals; `snap()` divides by `n_perf` so the reportable currency is PER PERFORMER, PER
    TRAVERSAL — `offbook/`'s "materialisations/traversal" and `legato/`'s "steady state per
    traversal". Every field is also kept raw so the price surface can be swept post hoc
    (`fingering/f1c`'s discipline) — except that here groundings ARE charged, because the price is
    the economy.

    A grounding is priced as étude priced its own audition rollouts (`select_x`'s `t_score` =
    `n_cand * n_score * (hh*dt_ctrl + d_fb)`): the trial's execution time plus one re-grounding.
    """

    FIELDS = ("steps", "fb", "ground", "ground_steps")

    def __init__(self, dt_ctrl=P.DT_CTRL, d_fb=P.D_FB, n_perf=1):
        self.dt_ctrl, self.d_fb = float(dt_ctrl), float(d_fb)
        self.n_perf = int(n_perf)
        self.reset()

    def reset(self):
        for f in self.FIELDS:
            setattr(self, f, 0.0)
        return self

    def add(self, steps=0.0, fb=0.0, ground=0.0, ground_steps=0.0):
        self.steps += float(steps); self.fb += float(fb)
        self.ground += float(ground); self.ground_steps += float(ground_steps)

    def _t(self, steps, fb, ground, gsteps):
        return (steps + gsteps) * self.dt_ctrl + (fb + ground) * self.d_fb

    @property
    def t(self):
        return self._t(self.steps, self.fb, self.ground, self.ground_steps)

    def snap(self):
        n = float(max(1, self.n_perf))
        s, f, g, gs = self.steps / n, self.fb / n, self.ground / n, self.ground_steps / n
        return dict(steps=s, n_fb=f, n_ground=g, ground_steps=gs,
                    t_exec=s * self.dt_ctrl + f * self.d_fb,
                    t_ground=gs * self.dt_ctrl + g * self.d_fb,
                    t_piece=self._t(s, f, g, gs),
                    raw=dict(steps=self.steps, n_fb=self.fb, n_ground=self.ground,
                             ground_steps=self.ground_steps, n_perf=self.n_perf))


# --------------------------------------------------------------------------- #
# the world
# --------------------------------------------------------------------------- #

class World:
    """PRESTISSIMO: the piece is now an argument, not a module of constants. Everything else in
    this class is `solo/world.py` unchanged. `k0` is the first DRILLED seam — segments
    `[0, k0)` are the approach, played closed-loop by the primitive and excluded from the
    comparison (presto decision 3). At `k0 = 0` every added line is inert, which is what lets
    gate P-F1 run the donor square through this class and reproduce `acappella/b1` exactly.
    """

    def __init__(self, cfg, piece=None, n_proc=1, rot=True):
        from mjc.pusher_env import PusherEnv
        self.cfg = cfg
        self.piece = piece if piece is not None else P.donor_piece()
        self.wps = np.array(self.piece.wps, np.float32)
        self.K = self.piece.K_seg
        self.k0 = int(self.piece.k0)
        self.K_drill = self.K - self.k0
        self.H = int(self.piece.H)
        self.fs = int(self.piece.frame_skip)
        self.dt_ctrl = self.fs * self.piece.timestep
        self.d_fb = float(self.piece.d_fb)
        self.rot = bool(rot)
        self.dgp = self.piece.dgp(rot)
        self.env = PusherEnv(self.dgp, with_puck=False)          # the BODY (in-parent, sequential)
        self.pool = RolloutPool(self.dgp, self.fs, n_proc)       # the resettable copies
        self.kp = float(cfg.get("kp", 0.0)); self.kd = float(cfg.get("kd", 0.0))
        # PRESTISSIMO decision 20, additive and inert when None (and structurally inert at
        # k0 = 0, which is what keeps P-F1 exact): separate gains for the APPROACH segments, so
        # the shared lead-in can be a genuinely PRE-COMPUTED, Delta-independent plan (presto
        # decision 3) rather than whatever the per-Delta re-fit happens to be.
        self.kp_app = None
        self.kd_app = None
        # SOLO: when a trunk is installed the PRIMITIVE is the cloned policy, not the PD law.
        # `trunk_np` is the numpy weight bundle the pool workers use (no torch in a worker).
        self.trunk_fn = None            # (states, k, h) -> commands, in-parent (torch)
        self.trunk_np = None            # dict of numpy weights, for `rollout_trunk`

    def close(self):
        self.pool.close()

    # ------------------------------------------------------------------ geometry
    def start_states(self, rng, n):
        p = (self.wps[0][None, :]
             + rng.uniform(-self.piece.start_jit, self.piece.start_jit, (n, 2)).astype(np.float32))
        v = rng.normal(0, self.piece.v0_std, (n, 2)).astype(np.float32)
        return np.concatenate([p, v], 1).astype(np.float32)

    def checkpoints(self, k, ns):
        """The (step, goal) pairs a span of `ns` segments starting at seam `k` crosses."""
        return [((j + 1) * self.H - 1, float(self.wps[k + j + 1][0]), float(self.wps[k + j + 1][1]))
                for j in range(int(ns))]

    def span_len(self, ns):
        return int(ns) * self.H

    # ------------------------------------------------------------------ a grounding
    def rollout(self, s0, cmds, k, ns, led=None, charge=True):
        """Execute `cmds` open-loop on resettable copies of the plant from `s0`. Returns
        (errs (n, ns) at each waypoint the span crosses, final states (n, 4)).

        THIS IS THE GROUNDING. `charge=False` is for instruments only, and every such call is
        labelled at its site."""
        cps = self.checkpoints(k, ns)
        errs, fin = self.pool.run(s0, cmds, cps)
        if led is not None and charge and len(s0):
            led.add(ground=len(s0), ground_steps=len(s0) * cmds.shape[1])
        return errs, fin

    # ------------------------------------------------------------------ the reflex law
    def reflex_cmd(self, states, k, h):
        """PD tracking of the waypoint schedule for segment `k` at step `h`.

        target(h) = wp_k + (h+1)/H * (wp_{k+1} - wp_k), with the schedule's own feedforward
        velocity. One feedback event per step; no search, no model, no memory."""
        a = self.wps[k]; b = self.wps[k + 1]
        f = (h + 1) / float(self.H)
        tgt = a + f * (b - a)
        vtg = (b - a) / (self.H * self.dt_ctrl)
        kp, kd = self.kp, self.kd
        if int(k) < int(self.k0) and self.kp_app is not None:
            kp, kd = float(self.kp_app), float(self.kd_app)
        u = kp * (tgt[None, :] - states[:, :2]) + kd * (vtg[None, :] - states[:, 2:])
        return np.clip(u, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ SOLO: the primitive
    def prim_cmd(self, states, k, h):
        """What the PRIMITIVE commands here. The reflex law itself when no trunk is installed
        (the donor path, which gate S-F1 exercises), the cloned trunk when one is."""
        if self.trunk_fn is None:
            return self.reflex_cmd(states, k, h)
        return self.trunk_fn(states, k, h)

    def rollout_trunk(self, s0, k, ns, led=None, charge=True, obs_delay=0):
        """A CLOSED-LOOP grounding: the trunk executed on resettable copies from `s0`.

        The observation delay is applied INSIDE the trial (the policy reads the copy's state D
        steps ago, clamped at the trial's start). Without that the audition would run the
        primitive under a capability the body does not have and would systematically over-rate
        it at Delta > 0 — which would bias `frac_prim`, this node's adoption readout, in the
        primitive's favour. Charged exactly like an open-loop grounding."""
        assert self.trunk_np is not None, "no trunk installed"
        cps = self.checkpoints(k, ns)
        errs, fin = self.pool.run_cl(s0, k, int(ns) * self.H, cps, self.trunk_np,
                                     int(obs_delay), self.K, self.H)
        if led is not None and charge and len(s0):
            led.add(ground=len(s0), ground_steps=len(s0) * int(ns) * self.H)
        return errs, fin

    # ------------------------------------------------------------------ the priced search
    def plant_cem(self, states, k, rng, led, budget, iters=None, sigma0=None, elite_frac=None,
                  vel_pen=None, instrument=False):
        """CEM on the PLANT. `budget` rollouts per decision, split over `iters` rounds.

        Returns (best actually-rolled-out sequence (n, H, 2), info). `budget` is spent exactly.
        """
        cfg = self.cfg
        iters = int(cfg.get("cem_iters", 4) if iters is None else iters)
        sigma0 = float(cfg.get("cem_init_sigma", 0.8) if sigma0 is None else sigma0)
        ef = float(cfg.get("cem_elite_frac", 0.125) if elite_frac is None else elite_frac)
        vp = float(cfg.get("vel_pen", 0.5) if vel_pen is None else vel_pen)
        n, hh = len(states), self.H
        G = int(budget)
        goal = self.wps[k + 1]
        mu = np.zeros((n, hh, 2), np.float32)
        sig = np.full((n, hh, 2), sigma0, np.float32)
        best_c = np.full(n, np.inf, np.float64)
        best_e = np.full(n, np.inf, np.float64)      # the winner's POSITION error, for free
        best_s = np.zeros((n, hh, 2), np.float32)
        spent = 0                      # PER PERFORMER: the budget G is per DECISION, not per batch
        for it in range(iters):
            m = (G - spent) if it == iters - 1 else max(1, G // iters)
            m = int(max(0, min(m, G - spent)))
            if m <= 0:
                break
            e = rng.standard_normal((n, m, hh, 2)).astype(np.float32)
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1.0, 1.0).astype(np.float32)
            s0 = np.repeat(states, m, axis=0)
            _, fin = self.rollout(s0, seqs.reshape(n * m, hh, 2), k, 1, led)
            spent += m
            perr = np.linalg.norm(fin[:, :2] - goal[None, :], axis=1).reshape(n, m)
            cost = (perr + vp * np.linalg.norm(fin[:, 2:], axis=1).reshape(n, m))
            ne = max(1, int(round(ef * m)))
            order = np.argsort(cost, axis=1, kind="stable")
            idx = order[:, :ne]
            elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
            mu = elite.mean(1).astype(np.float32)
            sig = (elite.std(1) + 1e-3).astype(np.float32)
            b = order[:, 0]
            cb = cost[np.arange(n), b]
            upd = cb < best_c
            best_e = np.where(upd, perr[np.arange(n), b], best_e)
            best_c = np.where(upd, cb, best_c)
            if upd.any():
                best_s[upd] = seqs[np.arange(n), b][upd]
        info = dict(budget=G, spent_per_perf=float(spent), iters=iters,
                    best_err_med=float(np.median(best_e)))
        if instrument:
            # NOT charged: the elite mean's realised error, for étude finding 2 on a real rollout
            em, _ = self.rollout(states, np.clip(mu, -1, 1).astype(np.float32), k, 1, led,
                                 charge=False)
            bs, _ = self.rollout(states, best_s, k, 1, led, charge=False)
            info["elite_mean_err"] = float(np.median(em[:, 0]))
            info["best_sampled_err"] = float(np.median(bs[:, 0]))
        return best_s, info, best_e

    # ------------------------------------------------------------------ the traversal
    def traverse(self, decide, starts, rng, led, explore=0.0, collect=False, obs_delay=0,
                 collect_obs=False, collect_traj=False, obs_delay_app=None):
        """Execute the whole piece once, batched over performers.

        `decide(k, states, need, rng, led)` is called at seam `k` for the performers whose previous
        decision has run out (`need` is their index array into the batch). It returns
            dict(kind="open", cmds=(len(need), span*H, 2), span=(len(need),) int array, ...)
         or dict(kind="reflex", span=ones)
        so a per-performer span (a chain for one performer, a segment for another) is legal — the
        block structure below is per SEGMENT, and a plan simply survives across blocks.

        Boundaries are re-grounding points, not teleports: every segment starts from the TRUE
        achieved state (étude, verbatim).

        OBSERVATION DELAY (`offbook/` round 4's one-variable construction, model-free). Every
        AGENT-SIDE feedback consumer — the reflex law's per-step read, the library key at a launch,
        the seam-time audition's rollout start, the search's initial state — sees the state from
        `obs_delay` control steps ago. **There is nothing to bridge it with**: this node has no
        forward model, so the naive delayed operator is the honest one here rather than the strawman
        `offbook/` d3b showed it to be when a predictor exists.
        EXPERIMENTER-SIDE INSTRUMENTS STAY TRUE-STATE: `wp_err`, `e_seg`, `e_piece`, the recorded
        seam states and the whole ledger are computed from `states`, never from `obs`.
        At `obs_delay = 0` `obs is states` by construction, so the delayed path IS the undelayed
        path (asserted by gate B-F1 against `a0`'s recorded numbers).
        """
        Bn = len(starts)
        led.n_perf = Bn
        states = np.array(starts, np.float32, copy=True)
        D = int(obs_delay)
        # PRESTISSIMO, additive and inert when `obs_delay_app is None`: the APPROACH may be played
        # under its own read delay. The lead-in is already excluded from the comparison (presto
        # decision 3 had it as a shared, pre-computed plan precisely so it was not a variable);
        # with `obs_delay_app = 0` it becomes an INSTRUMENT — every arm launches the drilled figure
        # from a hand-over the incumbent has NOT already lost, so the ladder can be read with and
        # without the approach's own delay collapse. `None` reproduces the previous behaviour
        # exactly, which is what keeps P-F1 (k0 = 0, no approach) untouched.
        Dapp = D if obs_delay_app is None else int(obs_delay_app)
        maxD = max(D, Dapp)
        hist = [states.copy()] if maxD > 0 else None
        cur = [D]                     # the delay in force at the segment being played

        def obs():
            """The agent's read. `hist[-1-D]`, clamped at the traversal start."""
            if cur[0] <= 0:
                return states
            return hist[max(0, len(hist) - 1 - cur[0])]

        Kn, H = self.K, self.H
        plan = np.zeros((Bn, Kn * H, 2), np.float32)
        off = np.zeros(Bn, int)
        left = np.zeros(Bn, int)
        is_reflex = np.zeros(Bn, bool)
        wp_err = np.zeros((Bn, Kn), np.float32)
        seam_states = {}
        decisions = []
        traces = {k: None for k in range(Kn)}
        acts_all = np.zeros((Bn, Kn * H, 2), np.float32)
        raw_all = np.zeros((Bn, Kn * H, 2), np.float32)
        # SOLO, additive and inert unless asked for: the agent's own READ at every step (the
        # behaviour-cloning input) and the realised trajectory (the intention reference).
        obs_all = np.zeros((Bn, Kn * H, 4), np.float32) if collect_obs else None
        traj_all = np.zeros((Bn, Kn * H, 2), np.float32) if collect_traj else None
        # PRESTISSIMO, additive and inert at k0 = 0: the ledger is split at the first DRILLED
        # seam, so the shared approach's cost (the primitive pays one feedback event per STEP for
        # k0*H steps) can be subtracted from the rent table. presto decision 3: the mastered
        # lead-in is not part of the comparison and must not be a second variable.
        k0 = int(self.k0)
        pre = dict(fb=0.0, ground=0.0, ground_steps=0.0)
        for k in range(Kn):
            cur[0] = Dapp if k < k0 else D
            if k == k0:
                pre = dict(fb=led.fb, ground=led.ground, ground_steps=led.ground_steps)
            seam_states[k] = states.copy()          # TRUE state: instrument, never the agent's read
            need = np.where(left == 0)[0]
            intra = None
            if len(need):
                d = decide(k, obs()[need], need, rng, led)
                if d.get("replan") is not None and len(need) == Bn:
                    # ÉTUDE'S TEMPO KNOB, on the plant: `replan_every < H` re-plans the full segment
                    # horizon from the realised state and executes only the head (etude.py's 4b
                    # idiom), paying a fresh feedback event AND a fresh budget every time. Only
                    # supported for a whole-batch decider, which is what the pure-search arms are.
                    intra = (d["replan"], int(d["re"]))
                sp = np.asarray(d.get("span", np.ones(len(need), int)), int)
                if d["kind"] == "reflex":
                    is_reflex[need] = True          # reflex pays one fb per STEP, below
                    left[need] = sp
                    off[need] = 0
                else:
                    # SOLO: a decision may be MIXED — some performers launch a committed unit,
                    # others play the primitive closed-loop. `prim` is the per-performer mask;
                    # absent, this is exactly the donor path.
                    pm = np.asarray(d.get("prim", np.zeros(len(need), bool)), bool)
                    is_reflex[need] = pm
                    c = np.asarray(d["cmds"], np.float32)
                    for j, b in enumerate(need):
                        if pm[j]:
                            continue
                        L = int(sp[j]) * H
                        plan[b, :L] = c[j, :L]
                        off[b] = 0
                    off[need[pm]] = 0
                    left[need] = np.where(pm, 1, sp)
                    led.add(fb=len(need))                 # one re-grounding per decision
                d2 = {kk: vv for kk, vv in d.items() if kk not in ("cmds",)}
                d2["seam"] = k; d2["n"] = len(need)
                decisions.append(d2)
            g0 = k * H
            for h in range(H):
                if intra is not None and h > 0 and h % intra[1] == 0:
                    plan[:, h:h + H] = intra[0](k, h, obs(), rng, led)
                    off[:] = h
                    led.add(fb=Bn)
                a = np.empty((Bn, 2), np.float32)
                if is_reflex.any():
                    idx = np.where(is_reflex)[0]
                    a[idx] = self.prim_cmd(obs()[idx], k, h)
                    led.add(fb=len(idx))          # the reflex pays one feedback event per STEP
                idx = np.where(~is_reflex)[0]
                if len(idx):
                    a[idx] = plan[idx, off[idx]]
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                raw_all[:, g0 + h, :] = a
                if obs_all is not None:
                    obs_all[:, g0 + h, :] = obs()
                if explore > 0.0:
                    a = np.clip(a + explore * rng.standard_normal(a.shape).astype(np.float32),
                                -1.0, 1.0).astype(np.float32)
                acts_all[:, g0 + h, :] = a
                nxt = np.empty_like(states)
                for b in range(Bn):
                    self.env.set_state(states[b, :2].astype(np.float64),
                                       states[b, 2:].astype(np.float64))
                    s2, _ = self.env.step(a[b], self.fs)
                    nxt[b] = s2
                states = nxt
                if traj_all is not None:
                    traj_all[:, g0 + h, :] = states[:, :2]
                if maxD > 0:
                    hist.append(states.copy())
                off[~is_reflex] += 1
            wp_err[:, k] = np.linalg.norm(states[:, :2] - self.wps[k + 1][None, :], axis=1)
            left -= 1
        led.add(steps=Bn * Kn * H)
        med = np.median(wp_err, 0)
        npf = float(max(1, led.n_perf))
        sd = float((Kn - k0) * H)
        fd = (led.fb - pre["fb"]) / npf
        gd = (led.ground - pre["ground"]) / npf
        gsd = (led.ground_steps - pre["ground_steps"]) / npf
        out = dict(wp_err=wp_err, e_seg=med,
                   # PRESTISSIMO: the piece error is over the DRILLED segments. At k0 = 0 this is
                   # `float(np.mean(np.median(wp_err, 0)))` exactly, which is what P-F1 checks.
                   e_piece=float(np.mean(med[k0:])),
                   e_app=(float(np.mean(med[:k0])) if k0 else 0.0),
                   ep=np.asarray(wp_err[:, k0:].mean(1), np.float64),
                   ledger_drilled=dict(steps=sd, n_fb=fd, n_ground=gd, ground_steps=gsd,
                                       t_exec=sd * self.dt_ctrl + fd * self.d_fb,
                                       t_ground=gsd * self.dt_ctrl + gd * self.d_fb,
                                       t_piece=(sd + gsd) * self.dt_ctrl + (fd + gd) * self.d_fb),
                   seam=seam_states, final=np.linalg.norm(states[:, :2] - self.wps[-1][None, :], axis=1),
                   decisions=decisions, ledger=led.snap())
        if obs_all is not None:
            out["obs"] = obs_all
        if traj_all is not None:
            out["traj"] = traj_all
        if collect:
            out["acts"] = acts_all
            out["raw"] = raw_all
            for k in range(Kn):
                traces[k] = dict(s0=seam_states[k].copy(),
                                 cmds=acts_all[:, k * H:(k + 1) * H, :].copy(),
                                 raw=raw_all[:, k * H:(k + 1) * H, :].copy(),
                                 err=wp_err[:, k].copy())
                if traj_all is not None:
                    traces[k]["traj"] = traj_all[:, k * H:(k + 1) * H, :].copy()
            out["traces"] = traces
        return out


# --------------------------------------------------------------------------- #
# the deciders (one per way of producing a command)
# --------------------------------------------------------------------------- #

class ReflexDecider:
    """`never_reflex`. Zero groundings, one feedback event per control step."""

    def __call__(self, k, states, need, rng, led):
        return dict(kind="reflex", span=np.ones(len(states), int), src="reflex")


class SearchDecider:
    """`never_search`. The priced-rollout incumbent at the declared per-decision budget G.

    `replan_every` is étude's tempo knob R: with R >= H the segment is planned once at its seam
    (performance tempo, one feedback event) and with R < H the full segment horizon is re-planned
    from the realised state every R steps, paying a fresh feedback event AND a fresh budget G each
    time. R is the axis on which "the widest search its own action set affords" has a second
    dimension, and it is measured as an instrument rather than assumed away.
    """

    def __init__(self, W, budget, instrument=False, replan_every=None):
        self.W, self.G, self.instrument = W, int(budget), bool(instrument)
        self.re = int(replan_every) if replan_every else int(W.H)
        self.info = []

    def _plan(self, k, states, rng, led, instrument=False):
        cmds, info, _ = self.W.plant_cem(states, k, rng, led, self.G, instrument=instrument)
        info["seam"] = k
        self.info.append(info)
        return cmds

    def __call__(self, k, states, need, rng, led):
        cmds = self._plan(k, states, rng, led, instrument=self.instrument)
        d = dict(kind="open", cmds=cmds, span=np.ones(len(states), int), src="search",
                 budget=self.G, re=self.re)
        if self.re < self.W.H:
            d["replan"] = lambda kk, hh, st, rg, ld: self._plan(kk, st, rg, ld)
        return d


class TapeDecider:
    """A fixed tape per seam, played blind. Zero groundings, one feedback event per launch.
    `tapes[k]` is a (span*H, 2) command array and `spans[k]` its span in segments."""

    def __init__(self, W, tapes, spans=None):
        self.W, self.tapes = W, tapes
        self.spans = spans or {k: 1 for k in tapes}

    def __call__(self, k, states, need, rng, led):
        ns = int(self.spans[k])
        c = np.tile(np.asarray(self.tapes[k], np.float32)[None], (len(states), 1, 1))
        return dict(kind="open", cmds=c, span=np.full(len(states), ns, int), src="tape")


class LibraryDecider:
    """The seam decision this node is about: at the realised seam state, choose among the library
    slots legal here and (optionally) the live search, then launch the winner.

    `mode`:
      "audit_all"   — audition EVERY legal slot on the plant (K groundings) + the primitive (G)
      "audit_k"     — audition only pi's top-k slots (k groundings) + the primitive (G)
      "key"         — no audition at all: pick the slot whose stored launch key is nearest the
                      observed hand-over (`legato/`'s `select_library`), 0 groundings
    `k_prop` selects top-k by `logits_fn(states, k)`; with `logits_fn=None` the top-k is the first
    k slots of the cell, which is what makes the `fid` twin exact at k = K.
    """

    def __init__(self, W, lib, mode="audit_all", k_prop=None, budget=0, logits_fn=None,
                 with_prim=True, levels=("seg",)):
        self.W, self.lib, self.mode = W, lib, mode
        self.k_prop = k_prop
        self.G = int(budget)
        self.logits_fn = logits_fn
        self.with_prim = bool(with_prim)
        self.levels = tuple(levels)
        self.picks = []

    def _legal(self, k):
        out = []
        for ns in self.lib.levels(k):
            lab = "seg" if ns == 1 else "chain"
            if lab not in self.levels:
                continue
            for j, s in enumerate(self.lib.cell(ns, k)):
                out.append((ns, j, s))
        return out

    def __call__(self, k, states, need, rng, led):
        W, H = self.W, self.W.H
        n = len(states)
        legal = self._legal(k)
        if not legal:
            if not (self.with_prim and self.G > 0):
                raise ValueError(f"no legal action at seam {k}")
            c, _, _ = W.plant_cem(states, k, rng, led, self.G)
            self.picks.append(dict(seam=k, n_legal=0, n_aud=0.0, frac_prim=1.0, frac_chain=0.0))
            return dict(kind="open", cmds=c, span=np.ones(n, int), src="prim_only",
                        n_cand=0.0, frac_prim=1.0, frac_chain=0.0)
        if self.mode == "key":
            best_ns = np.ones(n, int)
            keys = np.array([s["key"] for _, _, s in legal], np.float32)
            sd = keys.std(0) + 1e-6
            d = np.linalg.norm((states[:, None, :] - keys[None]) / sd[None, None], axis=2)
            pick = np.argmin(d, axis=1)
            maxspan = max(int(legal[p][0]) for p in np.unique(pick))
            best_c = np.zeros((n, maxspan * H, 2), np.float32)
            for i in range(n):
                ns, j, s = legal[pick[i]]
                best_ns[i] = ns
                best_c[i, :ns * H] = s["cmds"]
            self.picks.append(dict(seam=k, n_legal=len(legal), n_aud=0.0, frac_prim=0.0,
                                   frac_chain=float(np.mean(best_ns > 1)),
                                   slot=[int(legal[p][2]["slot"]) for p in pick],
                                   ns=[int(legal[p][0]) for p in pick]))
            return dict(kind="open", cmds=best_c, span=best_ns, src="key",
                        n_cand=float(len(legal)))
        # --- audition on the plant -------------------------------------------------------
        if self.mode == "audit_k" and self.logits_fn is not None:
            keep = self.logits_fn(states, k, legal, self.k_prop)
        elif self.mode == "audit_k":
            keep = [np.arange(min(int(self.k_prop), len(legal)))] * n
        else:
            keep = [np.arange(len(legal))] * n
        scores = np.full((n, len(legal)), np.inf, np.float64)
        for ci, (ns, j, s) in enumerate(legal):
            rows = np.array([i for i in range(n) if ci in keep[i]], int)
            if not len(rows):
                continue
            cmds = np.tile(np.asarray(s["cmds"], np.float32)[None], (len(rows), 1, 1))
            errs, _ = W.rollout(states[rows], cmds, k, ns, led)
            scores[rows, ci] = errs.mean(1)          # the mean over the span's waypoints
        prim_c = None
        prim_score = np.full(n, np.inf, np.float64)
        if self.with_prim and self.G > 0:
            # the search's own budget already contains the winner's realised boundary error, so the
            # primitive costs G groundings and not G+1 -- it verifies as it searches.
            prim_c, _, prim_score = W.plant_cem(states, k, rng, led, self.G)
        best_ci = np.argmin(scores, axis=1)
        best_sc = scores[np.arange(n), best_ci]
        use_prim = prim_score < best_sc
        maxspan = max([int(legal[c][0]) for c in np.unique(best_ci)] + [1])
        out_c = np.zeros((n, maxspan * H, 2), np.float32)
        out_ns = np.ones(n, int)
        for i in range(n):
            if use_prim[i]:
                out_c[i, :H] = prim_c[i]
                out_ns[i] = 1
            else:
                ns, j, s = legal[best_ci[i]]
                out_c[i, :ns * H] = s["cmds"]
                out_ns[i] = ns
        self.picks.append(dict(seam=k, n_legal=len(legal),
                               # who decided here and over what span — the audition-optimism
                               # instrument needs both, because a chain arm decides for everyone at
                               # seam 0 and for nobody after, so `picks` is ragged by construction.
                               need=[int(b) for b in np.asarray(need, int)],
                               ns=[int(x) for x in out_ns],
                               score_mat=scores.tolist(),
                               slot_ids=[int(s["slot"]) for _, _, s in legal],
                               slot_ns=[int(ns) for ns, _, _ in legal],
                               n_aud=float(np.mean([len(x) for x in keep])),
                               frac_prim=float(use_prim.mean()),
                               frac_chain=float(np.mean(out_ns > 1)),
                               slot=[-1 if use_prim[i] else int(legal[best_ci[i]][2]["slot"])
                                     for i in range(n)],
                               score=[float(min(best_sc[i], prim_score[i])) for i in range(n)]))
        return dict(kind="open", cmds=out_c, span=out_ns, src=self.mode,
                    n_cand=float(np.mean([len(x) for x in keep])),
                    frac_prim=float(use_prim.mean()), frac_chain=float(np.mean(out_ns > 1)))


# --------------------------------------------------------------------------- #
# the library
# --------------------------------------------------------------------------- #

class Library:
    """Cells keyed `(ns, k)`: a unit spanning `ns` segments launched at seam `k`. Two levels, as in
    `offbook/` — `ns = 1` (a segment tape) and `ns = K - k` (a chain running to the end of the
    piece). Slot ids come from `offbook/nets.py::SlotLayout` so pi's logits mean the same thing at
    cycle 1 and at cycle N."""

    def __init__(self, K, n_slot, poison=True):
        from mjc.practice.offbook.nets import SlotLayout
        self.K = int(K)
        self.layout = SlotLayout(K, int(n_slot), poison=poison)
        self.n_slot = int(n_slot)
        self._cells = {(ns, k): [] for k in range(K) for ns in self.layout.levels(k)}

    def levels(self, k):
        return self.layout.levels(k)

    def cell(self, ns, k):
        return self._cells[(int(ns), int(k))]

    def add(self, ns, k, cmds, key, score, meta=None):
        c = self._cells[(int(ns), int(k))]
        j = len(c)
        if j >= self.layout.width:
            raise ValueError(f"cell ({ns},{k}) full at {j}")
        c.append(dict(cmds=np.asarray(cmds, np.float32), key=np.asarray(key, np.float32),
                      score=float(score), slot=int(self.layout.slot(ns, k, j)), j=j,
                      meta=meta or {}))
        return c[-1]

    def legal_by_seam(self):
        out = {}
        for k in range(self.K):
            ids = []
            for ns in self.levels(k):
                ids += [s["slot"] for s in self.cell(ns, k)]
            ids.append(self.layout.prim(k))
            out[k] = ids
        return out

    def sizes(self):
        return {f"{ns}:{k}": len(v) for (ns, k), v in sorted(self._cells.items())}


def select_tapes(W, pool, S0_score, k, ns, rng, led, n_cand, n_keep):
    """étude's `select_x`, on the plant and generalised to a span.

    Candidates are drawn UNIFORMLY from the rendition pool — never ranked by their own realised
    outcome (E-3b's winner's curse: that favours the sequence most finely tuned to its own start
    state, i.e. the least transferable). Each is replayed from `S0_score` held-out hand-over states
    produced by the CURRENT PERFORMANCE CONFIGURATION (the seam-matched score set: E-3b's
    seam-state shift was a 4.5-sigma offset and the whole of its 3x optimism gap), and ranked by
    the MEAN over the span's waypoints (`offbook/`'s cross-level convention).

    Every scoring rollout is a real committed traversal of the span, and is priced.
    Returns the `n_keep` best, plus the full record.
    """
    A = np.asarray(pool["cmds"], np.float32)
    n = len(A)
    nc = int(min(n_cand, n))
    cand = rng.permutation(n)[:nc]
    S0 = np.asarray(S0_score, np.float32)
    m = len(S0)
    sc = np.empty(nc, np.float64)
    for i, ci in enumerate(cand):
        cm = np.tile(A[ci][None], (m, 1, 1))
        errs, _ = W.rollout(S0, cm, k, ns, led)
        sc[i] = float(errs.mean())
    order = np.argsort(sc, kind="stable")
    keep = order[:int(n_keep)]
    own = np.asarray(pool["err"], np.float64)[cand] if "err" in pool else np.full(nc, np.nan)
    rec = dict(n_pool=int(n), n_cand=int(nc), n_score=int(m),
               chosen_score=float(sc[order[0]]), score_med=float(np.median(sc)),
               score_max=float(sc.max()),
               score_of_best_own=float(sc[int(np.nanargmin(own))]) if np.isfinite(own).any() else float("nan"),
               kept_scores=[float(sc[i]) for i in keep])
    return [dict(cmds=A[cand[i]], key=np.asarray(pool["s0"], np.float32)[cand[i]],
                 score=float(sc[i])) for i in keep], rec


# =========================================================================== #
#                         WHAT SOLO ADDS                                      #
# =========================================================================== #
# Everything above this line is `acappella/world.py`, forked. Everything below
# is new, and none of it is a forward model: nothing here predicts a state.
# =========================================================================== #


# --------------------------------------------------------------------------- #
# the trunk: a behaviour-cloned reflex
# --------------------------------------------------------------------------- #

def _build_trunk_cls():
    import torch
    import torch.nn as nn

    class TrunkNet(nn.Module):
        """(delayed observation, seam index, phase within segment) -> command.

        The learner's own primitive executor. It is cloned from the REFLEX LAW's own closed-loop
        traversals — the model-free analogue of `ballistic_bc` / `jacobian_teacher.clone_policy`,
        which cloned a CEM planner's committed programs. It predicts no next state and consults
        no `f(s,u)`; it only reproduces what the law it clones would command from the read it has.

        The output layer is LINEAR, not tanh: the teacher's own command is already clipped to
        [-1,1] and `traverse` clips again, so a squashing output would only add an approximation
        error exactly where the law saturates (which is most of the first half of a segment).
        """

        def __init__(self, state_dim, n_seg, act_dim, hidden=128, layers=2):
            super().__init__()
            din = int(state_dim) + int(n_seg) + 1
            lyr = [nn.Linear(din, hidden), nn.SiLU()]
            for _ in range(int(layers) - 1):
                lyr += [nn.Linear(hidden, hidden), nn.SiLU()]
            self.hid = nn.Sequential(*lyr)
            self.out = nn.Linear(hidden, int(act_dim))
            self.hidden = int(hidden)

        def features(self, x):
            return self.hid(x)

        def forward(self, x):
            return self.out(self.hid(x))

    return TrunkNet


def trunk_inputs(states, k, f, n_seg):
    """[state(4), seam one-hot(n_seg), phase]. The phase is the schedule fraction (h+1)/H, which
    is what makes the law's moving target learnable without ever being told the target."""
    s = np.asarray(states, np.float32)
    x = np.zeros((len(s), 4 + int(n_seg) + 1), np.float32)
    x[:, :4] = s
    x[:, 4 + int(k)] = 1.0
    x[:, -1] = float(f)
    return x


def trunk_bundle(net, norm):
    """The numpy weight bundle a pool worker uses (no torch in a worker process)."""
    hid = []
    for m in net.hid:
        if hasattr(m, "weight"):
            hid.append((m.weight.detach().cpu().numpy().astype(np.float64),
                        m.bias.detach().cpu().numpy().astype(np.float64)))
    return dict(mu=np.asarray(norm[0], np.float64), sd=np.asarray(norm[1], np.float64), hid=hid,
                out=(net.out.weight.detach().cpu().numpy().astype(np.float64),
                     net.out.bias.detach().cpu().numpy().astype(np.float64)))


def clone_trunk(worlds, cfg, seed, device, n_seg, deltas=(0, 8), log=None):
    """Behaviour-clone the reflex law from its OWN traversals.

    `worlds` is a list of (World, gains) pairs — the rotated and the clean world, each with the
    gains the arm's reflex reference uses. Data is collected at every Delta in `deltas` so the
    clone is fit on the read distribution it will actually be asked to serve (the law is a pure
    function of its read, so cloning on the read is the honest target under delay).
    """
    import torch
    import torch.nn as nn
    X, Y = [], []
    for wi, (Wd, gains) in enumerate(worlds):
        kp0, kd0, tf0 = Wd.kp, Wd.kd, Wd.trunk_fn
        Wd.kp, Wd.kd, Wd.trunk_fn = float(gains[0]), float(gains[1]), None
        for D in deltas:
            for c in range(int(cfg["bc_cycles"])):
                rng = np.random.default_rng(int(seed) + 77000 + 991 * wi + 37 * int(D) + c)
                led = Ledger(Wd.dt_ctrl, Wd.d_fb)
                o = Wd.traverse(ReflexDecider(), Wd.start_states(rng, int(cfg["bc_batch"])), rng,
                                led, explore=float(cfg["bc_explore"]), collect=True,
                                obs_delay=int(D), collect_obs=True)
                Bn, T = o["obs"].shape[0], o["obs"].shape[1]
                H = Wd.H
                for t in range(T):
                    k, h = t // H, t % H
                    X.append(trunk_inputs(o["obs"][:, t, :], k, (h + 1) / float(H), n_seg))
                    Y.append(o["raw"][:, t, :].astype(np.float32))
        Wd.kp, Wd.kd, Wd.trunk_fn = kp0, kd0, tf0
    X = np.concatenate(X, 0).astype(np.float32)
    Y = np.concatenate(Y, 0).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = _build_trunk_cls()(4, n_seg, 2, hidden=int(cfg["bc_hidden"]),
                             layers=int(cfg["bc_layers"]))
    g = torch.Generator().manual_seed(int(seed) + 4242)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                p.copy_(torch.empty(p.shape, dtype=p.dtype).normal_(0.0, 0.05, generator=g))
            else:
                p.zero_()
    net = net.to(device)
    Xt = torch.tensor((X - mu) / sd, device=device)
    Yt = torch.tensor(Y, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["bc_lr"]))
    lossf = nn.MSELoss()
    brng = np.random.default_rng(int(seed) + 4243)
    net.train()
    bs = min(int(cfg["bc_batch_sz"]), len(X))
    last = None
    n_steps = int(cfg["bc_steps"])
    for it in range(n_steps):
        # step LR decay: an MSE clone of a piecewise-linear law is limited by the tail of the fit,
        # and a flat LR leaves it there. Not a gate knob — the gate's band is fixed in `cfg`.
        if it in (n_steps // 2, (4 * n_steps) // 5):
            for gp in opt.param_groups:
                gp["lr"] *= 0.25
        idx = torch.tensor(brng.integers(0, len(X), size=bs), device=device)
        opt.zero_grad()
        loss = lossf(net(Xt[idx]), Yt[idx])
        loss.backward()
        opt.step()
        last = float(loss)
        if log is not None and (it + 1) % max(1, int(cfg["bc_steps"]) // 5) == 0:
            log(f"[trunk] step {it + 1}/{cfg['bc_steps']}  mse={last:.3e}")
    net.eval()
    with torch.no_grad():
        fit = float(((net(Xt) - Yt) ** 2).mean())
    return net, (mu, sd), dict(n=int(len(X)), mse_final=last, mse_full=fit)


def make_trunk_fn(net, norm, device, n_seg, H):
    """(states, k, h) -> commands. Torch, in-parent, no grad."""
    import torch

    def fn(states, k, h):
        if len(states) == 0:
            return np.zeros((0, 2), np.float32)
        x = trunk_inputs(states, k, (h % H + 1) / float(H), n_seg)
        with torch.no_grad():
            u = net(torch.tensor((x - norm[0]) / norm[1], device=device)).cpu().numpy()
        return np.clip(u, -1.0, 1.0).astype(np.float32)

    return fn


def make_trunk_read(net, norm, device, n_seg, H):
    """The trunk read the span head is conditioned on: the cloned policy's last hidden layer at
    the seam posture (phase 1/H). FROZEN and DETACHED — see FILES.md decision on the trunk."""
    import torch

    def read(states, k):
        x = trunk_inputs(states, k, 1.0 / float(H), n_seg)
        with torch.no_grad():
            return net.features(torch.tensor((x - norm[0]) / norm[1], device=device))

    return read


# --------------------------------------------------------------------------- #
# the library, with MEMBERS
# --------------------------------------------------------------------------- #

def _kmeans(Xf, n_clu, rng, iters=25):
    """Lloyd's with k-means++ init, over CONTENT (flattened command sequences). Deterministic
    given `rng`. `offbook/` decision 3: slots are fit ONCE over content and frozen, so pi's
    logits, the span head's conditioning and the trust series all address a fixed partition."""
    n = len(Xf)
    n_clu = int(min(n_clu, n))
    cen = np.empty((n_clu, Xf.shape[1]), np.float64)
    cen[0] = Xf[rng.integers(n)]
    d2 = ((Xf - cen[0]) ** 2).sum(1)
    for j in range(1, n_clu):
        p = d2 / max(d2.sum(), 1e-12)
        cen[j] = Xf[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((Xf - cen[j]) ** 2).sum(1))
    lab = np.zeros(n, int)
    for _ in range(int(iters)):
        dd = ((Xf[:, None, :] - cen[None]) ** 2).sum(2)
        nl = dd.argmin(1)
        if (nl == lab).all():
            break
        lab = nl
        for j in range(n_clu):
            m = lab == j
            if m.any():
                cen[j] = Xf[m].mean(0)
    return lab, cen


class MemberLibrary:
    """Cells keyed `(ns, k)`; each cell holds `n_slot` SLOTS and each slot holds MEMBERS.

    This is the one structural departure from `acappella/Library`, and it is what gives Port 2 a
    job: with one tape per slot the span head could only ever reproduce a constant, and the
    within-slot question the SPEC names ("does the conditioned head beat verbatim playback on
    off-key seams") would have no content. With members, the audition's within-slot argmin is a
    real state-conditioned choice, and the head replaces exactly that (m groundings -> 1).

    Slot ids come from `offbook/nets.py::SlotLayout`, so pi's logits mean the same thing here as
    in `offbook/` and in the donor's own `Library`. The poison twin is pinned to the reserved id
    `j = n_slot`.
    """

    def __init__(self, K, n_slot, poison=True):
        from mjc.practice.offbook.nets import SlotLayout
        self.K = int(K)
        self.n_slot = int(n_slot)
        self.layout = SlotLayout(K, int(n_slot), poison=poison)
        self._cells = {(1, k): [] for k in range(K)}

    def add_slot(self, k, members, key, j=None, poison=False, meta=None):
        c = self._cells[(1, int(k))]
        jj = (self.n_slot if poison else (len(c) if j is None else int(j)))
        sl = dict(slot=int(self.layout.slot(1, int(k), jj)), j=int(jj), seam=int(k),
                  poison=bool(poison), key=np.asarray(key, np.float32),
                  members=[dict(cmds=np.asarray(m["cmds"], np.float32),
                                key=np.asarray(m["key"], np.float32),
                                traj=(None if m.get("traj") is None
                                      else np.asarray(m["traj"], np.float32)),
                                score=float(m["score"])) for m in members],
                  meta=meta or {})
        c.append(sl)
        return sl

    def slots(self, k):
        return self._cells[(1, int(k))]

    def slot_ids(self, k):
        return [s["slot"] for s in self.slots(k)]

    def prim(self, k):
        return int(self.layout.prim(int(k)))

    def legal_by_seam(self, with_prim=True):
        out = {}
        for k in range(self.K):
            ids = list(self.slot_ids(k))
            if with_prim:
                ids.append(self.prim(k))
            out[k] = ids
        return out

    def sizes(self):
        return {f"1:{k}": [len(s["members"]) for s in self.slots(k)] for k in range(self.K)}


def build_member_cell(W, pool, S0_score, k, rng, led, n_cand, n_slot, m_member, n_poison):
    """The treatment library's cell at seam `k`.

    THE OP, in order. (1) Draw `n_cand` candidates UNIFORMLY from the rendition pool — never ranked
    by their own realised outcome (etude E-3b's winner's curse: that favours the sequence most
    finely tuned to its own start state, i.e. the least transferable). (2) Score every one on
    held-out hand-over states produced by the current performance configuration. (3) **Keep the
    best `n_slot * m_member`** — this is `select_tapes`' own op, verbatim in intent, and it is what
    makes this library a RE-PARTITION of the donor's rather than a different, worse object.
    (4) Partition those by CONTENT into `n_slot` slots (k-means, fit once, frozen — offbook
    decision 3), taking up to `m_member` per cluster and filling any short slot from the unused
    kept candidates, best score first. (5) The poison twin is the `n_poison` WORST-scoring
    candidates of the FULL draw, pinned to the reserved slot.

    Step (3) is not free-standing taste: clustering the whole draw admits, into every cell, content
    the construction audition had already rejected, and the `ssize2` probe measured what that costs
    — `audit_all` over the all-candidate partition reached 0.1196 at Delta = 8 against the donor
    library's own `lib_seg` 0.0838 on the same pool. The criterion is arm-neutral (it names only
    the library's agreement with the donor construction, is applied identically to every arm, and
    was fixed before any arm-vs-arm number existed) — `legato/` F2's discipline: calibrate to
    preserve the axis, never to make an arm win. Both constructions are on the record.

    (`offbook/` harvested its poison from pre-competence cycles; there is no pre-competence phase
    here — the reflex law is fixed and competent from the first cycle — so the bad-rendition tail
    is the honest analogue, and it is exactly as plausible an address.)
    """
    A = np.asarray(pool["cmds"], np.float32)
    S = np.asarray(pool["s0"], np.float32)
    TR = np.asarray(pool["traj"], np.float32) if pool.get("traj") is not None else None
    n = len(A)
    nc = int(min(n_cand, n))
    cand = rng.permutation(n)[:nc]
    S0 = np.asarray(S0_score, np.float32)
    m = len(S0)
    sc = np.empty(nc, np.float64)
    for i, ci in enumerate(cand):
        errs, _ = W.rollout(S0, np.tile(A[ci][None], (m, 1, 1)), k, 1, led)
        sc[i] = float(errs.mean())
    order = np.argsort(sc, kind="stable")
    n_pick = int(min(int(n_slot) * int(m_member), nc))
    pick = np.asarray(order[:n_pick], int)               # step (3): the construction audition's own keep
    Xf = A[cand[pick]].reshape(n_pick, -1).astype(np.float64)
    lab, _ = _kmeans(Xf, n_slot, rng)
    rank = {int(i): r for r, i in enumerate(pick)}       # position within the kept, by score
    used, slots, n_rep = set(), [], 0
    for j in range(n_slot):
        idx = [int(i) for i in pick if lab[rank[int(i)]] == j and int(i) not in used][:int(m_member)]
        used.update(idx)
        slots.append(idx)
    for j in range(n_slot):                              # repair short slots, best score first
        while len(slots[j]) < int(m_member):
            rest = [int(i) for i in pick if int(i) not in used]
            if not rest:
                break
            slots[j].append(rest[0]); used.add(rest[0]); n_rep += 1
        if not slots[j]:
            break
    poison = [int(i) for i in order[::-1] if int(i) not in used][:int(n_poison)]
    rec = dict(seam=int(k), n_pool=int(n), n_cand=int(nc), n_score=int(m), n_pick=n_pick,
               chosen_score=float(sc[order[0]]), score_med=float(np.median(sc)),
               score_max=float(sc.max()),
               kept_score_max=float(sc[pick].max()), n_repaired=int(n_rep),
               slot_scores=[[float(sc[i]) for i in g] for g in slots],
               slot_sizes=[len(g) for g in slots],
               poison_scores=[float(sc[i]) for i in poison],
               cluster_sizes=[int((lab == j).sum()) for j in range(n_slot)])

    def mk(g):
        return [dict(cmds=A[cand[i]], key=S[cand[i]],
                     traj=(None if TR is None else TR[cand[i]]), score=float(sc[i])) for i in g]

    return [mk(g) for g in slots], mk(poison), rec


# --------------------------------------------------------------------------- #
# the seam decision
# --------------------------------------------------------------------------- #

class SoloDecider:
    """Every arm and every battery row is a setting of this one decider.

    `mode`:
      "key"        — no audition: nearest slot centroid, then nearest member key. 0 groundings.
                     (No primitive: a frozen key has no way to key a closed-loop controller —
                     `offbook/` decision 8's `key_frozen`, said out loud rather than papered over.)
      "audit_all"  — audition every member of every legal slot, plus the primitive. O(K).
      "prop_k"     — PORT 1: pi ranks the slots, only its top-k are auditioned. O(k).
      "prop_kN"    — pi live at k = every legal slot; an exact twin of `audit_all` by construction.
      "native"     — PORT 1 + PORT 2: a slot that has cleared parity contributes ONE head emission
                     instead of all its members (m groundings -> 1); a slot below parity is
                     auditioned verbatim, so "below parity -> keep the tape" cannot introduce drift.
      "no_table"   — the address book is GONE: pi's argmax names a slot and the head emits its
                     content, or names the primitive and the trunk plays. 0 groundings, 1 fb.

    The primitive is auditioned CLOSED-LOOP on the plant (`World.rollout_trunk`) and is always
    forced into the audition set when it is legal — `offbook/`'s rule that an untrained logit may
    never rank the primitive out.
    """

    def __init__(self, W, mlib, mode="audit_all", k_prop=2, with_prim=True, obs_delay=0,
                 prop_fn=None, span_fn=None, open_slots=(), force_all=False,
                 eps_act=0.0, explore_eps=0.0, act_rng=None, exp_rng=None,
                 capture=False, head_forced_open=False):
        self.W, self.lib, self.mode = W, mlib, mode
        self.k_prop = int(k_prop)
        self.with_prim = bool(with_prim)
        self.D = int(obs_delay)
        self.prop_fn, self.span_fn = prop_fn, span_fn
        self.open_slots = set(int(s) for s in (open_slots or ()))
        self.force_all = bool(force_all)
        self.eps_act, self.explore_eps = float(eps_act), float(explore_eps)
        self.act_rng, self.exp_rng = act_rng, exp_rng
        self.capture = bool(capture)
        self.head_forced_open = bool(head_forced_open)
        self.picks = []
        self.cap_rows = []          # (state, slot id, target commands) for the span buffer
        self.prop_rows = []         # (state, seam, chosen slot id, performer) for pi's targets

    # ---------------------------------------------------------------- helpers
    def _propose(self, states, S, k):
        """Which slots (positions into `S`) each performer auditions."""
        from mjc.practice.offbook.nets import select_slots, explore_slots
        n, L = len(states), len(S)
        if self.mode in ("audit_all",) or self.force_all:
            return [np.arange(L) for _ in range(n)]
        kk = L if self.mode == "prop_kN" else self.k_prop
        ids = np.asarray([s["slot"] for s in S], int)
        lg = (self.prop_fn(states, k) if self.prop_fn is not None
              else np.zeros((n, int(ids.max()) + 1)))
        sel = select_slots(lg, ids, kk)
        if self.explore_eps > 0.0 and self.exp_rng is not None:
            ex = explore_slots(sel, L, self.explore_eps, self.exp_rng)
            sel = [np.unique(np.concatenate([a, b])) if len(b) else a for a, b in zip(sel, ex)]
        return sel

    # ---------------------------------------------------------------- the decision
    def __call__(self, k, states, need, rng, led):
        W, H = self.W, self.W.H
        n = len(states)
        S = self.lib.slots(k)
        L = len(S)
        prim_id = self.lib.prim(k)

        if self.mode == "key":
            keys = np.array([s["key"] for s in S], np.float32)
            sd = keys.std(0) + 1e-6
            pick = np.argmin(np.linalg.norm((states[:, None, :] - keys[None]) / sd[None, None],
                                            axis=2), axis=1)
            out_c = np.zeros((n, H, 2), np.float32)
            chosen, mem = [], []
            for i in range(n):
                sl = S[pick[i]]
                mk = np.array([mm["key"] for mm in sl["members"]], np.float32)
                mj = int(np.argmin(np.linalg.norm((states[i][None] - mk) / sd[None], axis=1)))
                out_c[i] = sl["members"][mj]["cmds"]
                chosen.append(int(sl["slot"])); mem.append(mj)
            self._record(k, need, states, chosen, mem, np.zeros(n, bool), np.full(n, np.nan),
                         n_aud=0.0, n_legal=L)
            return dict(kind="open", cmds=out_c, span=np.ones(n, int), src="key",
                        prim=np.zeros(n, bool), n_cand=0.0, frac_prim=0.0)

        if self.mode == "no_table":
            ids = np.asarray([s["slot"] for s in S] + ([prim_id] if self.with_prim else []), int)
            lg = self.prop_fn(states, k)[:, ids]
            pick = ids[np.argmax(lg, axis=1)]
            use_prim = pick == prim_id
            out_c = np.zeros((n, H, 2), np.float32)
            by_slot = {}
            for i in range(n):
                if not use_prim[i]:
                    by_slot.setdefault(int(pick[i]), []).append(i)
            for sid, rows in by_slot.items():
                out_c[np.asarray(rows, int)] = self.span_fn(states[np.asarray(rows, int)], sid, k)
            self._record(k, need, states, [int(x) for x in pick], [-2] * n, use_prim,
                         np.full(n, np.nan), n_aud=0.0, n_legal=L)
            return dict(kind="open", cmds=out_c, span=np.ones(n, int), src="no_table",
                        prim=use_prim, n_cand=0.0, frac_prim=float(use_prim.mean()))

        # ------------------------------------------------------- the audition modes
        prop = self._propose(states, S, k)
        scores = np.full((n, L), np.inf, np.float64)
        memb = np.full((n, L), -1, int)
        emit = {}
        n_aud = np.zeros(n, np.float64)
        for si, sl in enumerate(S):
            rows = np.array([i for i in range(n) if si in prop[i]], int)
            if not len(rows):
                continue
            use_head = (self.mode == "native"
                        and (self.head_forced_open or sl["slot"] in self.open_slots))
            if use_head:
                cm = np.asarray(self.span_fn(states[rows], int(sl["slot"]), k), np.float32)
                errs, _ = W.rollout(states[rows], cm, k, 1, led)
                scores[rows, si] = errs.mean(1)
                memb[rows, si] = -2
                emit[si] = (rows, cm)
                n_aud[rows] += 1.0
            else:
                mm = len(sl["members"])
                s0 = np.repeat(states[rows], mm, axis=0)
                cm = np.tile(np.stack([mb["cmds"] for mb in sl["members"]]), (len(rows), 1, 1))
                errs, _ = W.rollout(s0, cm, k, 1, led)
                e = errs.mean(1).reshape(len(rows), mm)
                mi = e.argmin(1)
                scores[rows, si] = e[np.arange(len(rows)), mi]
                memb[rows, si] = mi
                n_aud[rows] += float(mm)
        prim_score = np.full(n, np.inf, np.float64)
        if self.with_prim:
            errs, _ = W.rollout_trunk(states, k, 1, led, obs_delay=self.D)
            prim_score = errs.mean(1)
            n_aud += 1.0
        best_si = np.argmin(scores, axis=1)
        best_sc = scores[np.arange(n), best_si]
        use_prim = prim_score < best_sc
        # epsilon-greedy AT THE ACTION, practice only, from a dedicated stream (offbook O3):
        # the only channel by which an option the audition ranks badly can ever enter pi's targets.
        if self.eps_act > 0.0 and self.act_rng is not None:
            draw = self.act_rng.random(n)
            for i in range(n):
                if draw[i] >= self.eps_act:
                    continue
                opts = [int(x) for x in prop[i] if np.isfinite(scores[i, x])]
                if self.with_prim:
                    opts.append(-1)
                if not opts:
                    continue
                c = opts[int(self.act_rng.integers(len(opts)))]
                use_prim[i] = (c == -1)
                if c >= 0:
                    best_si[i] = c
        out_c = np.zeros((n, H, 2), np.float32)
        chosen, mem = [], []
        for i in range(n):
            if use_prim[i]:
                chosen.append(int(prim_id)); mem.append(-1)
                continue
            si = int(best_si[i])
            chosen.append(int(S[si]["slot"]))
            mem.append(int(memb[i, si]))
            if memb[i, si] == -2:
                rows, cm = emit[si]
                out_c[i] = cm[int(np.where(rows == i)[0][0])]
            else:
                out_c[i] = S[si]["members"][int(memb[i, si])]["cmds"]
        # PORT 2's targets: EVERY selected slot feeds its buffer, not only the launched one
        # (offbook decision 6) — a head that only learned from launched calls would starve exactly
        # where the corridor most needs consolidating.
        if self.capture:
            for si, sl in enumerate(S):
                rows = np.array([i for i in range(n) if si in prop[i] and memb[i, si] >= 0], int)
                if not len(rows):
                    continue
                tgt = np.stack([sl["members"][int(memb[i, si])]["cmds"] for i in rows])
                self.cap_rows.append((states[rows].copy(), int(sl["slot"]), tgt))
        self._record(k, need, states, chosen, mem, use_prim,
                     np.where(use_prim, prim_score, best_sc),
                     n_aud=float(n_aud.mean()), n_legal=L,
                     prop=[[int(x) for x in p] for p in prop])
        return dict(kind="open", cmds=out_c, span=np.ones(n, int), src=self.mode,
                    prim=use_prim, n_cand=float(np.mean([len(p) for p in prop])),
                    frac_prim=float(use_prim.mean()))

    def _record(self, k, need, states, chosen, mem, use_prim, score, n_aud, n_legal, prop=None):
        self.picks.append(dict(seam=int(k), n_legal=int(n_legal), n_aud=float(n_aud),
                               need=[int(b) for b in np.asarray(need, int)],
                               slot=[int(x) for x in chosen],
                               member=[int(x) for x in mem],
                               frac_prim=float(np.mean(use_prim)),
                               score=[float(x) for x in score],
                               n_prop=(0.0 if prop is None
                                       else float(np.mean([len(p) for p in prop])))))
        self.prop_rows.append((states.copy(), int(k), np.asarray(chosen, int),
                               np.asarray(need, int)))


# =========================================================================== #
#                      WHAT PRESTISSIMO ADDS                                  #
# =========================================================================== #
# Everything above this line is `solo/world.py` with the piece made an
# argument. Everything below is new, and none of it is a forward model:
# nothing here predicts a state. A level is built by WELDING content the body
# already produced and AUDITIONING the weld by executing it.
# =========================================================================== #


class LadderLayout:
    """Stable global slot ids for a DYADIC ladder.

    A level-l cell is `(l, k)` where `k` is a DRILLED seam with `k % 2**(l-1) == 0` and the unit
    spans `2**(l-1)` segments. With `K = 8` drilled segments the ladder is
    L1 at seams 0..7, L2 at 0/2/4/6, L3 at 0/4, L4 at 0 — four rungs whose top is the whole
    drilled figure, which is why `K` is 8 and not presto's 5 (5 does not tile).

    ALIGNED, not position-free. RHM's `T[l]` entries are position-free because a level-l macro
    can be applied wherever its span fits; a piece is not translation-invariant, so an entry here
    carries its seam. Aligning them is what makes `T[l] subset T[l-1] x T[l-1]` a tiling —
    every level-l unit is exactly two committed level-(l-1) units and the top rung is the piece.

    `offbook/nets.py::SlotLayout`'s contract is kept verbatim: every cell reserves the SAME
    number of ids whether or not it is populated (so the layout is a property of the piece and
    not of the run's history, and pi's logits mean the same thing at cycle 1 and at cycle N), one
    id per cell is reserved for the poison twin, and the primitive's per-seam ids sit at the end.
    """

    def __init__(self, K, n_slot, n_levels, poison=True):
        self.K = int(K)
        self.n_slot = int(n_slot)
        self.n_levels = int(n_levels)
        self.poison = bool(poison)
        self.width = int(n_slot) + (1 if poison else 0)
        self.off = {}
        cur = 0
        for ell in range(1, self.n_levels + 1):
            sp = 1 << (ell - 1)
            for k in range(0, self.K, sp):
                if k + sp <= self.K:
                    self.off[(ell, k)] = cur
                    cur += self.width
        self.prim_off = cur
        cur += self.K
        self.n_slots = cur

    @staticmethod
    def span(ell):
        return 1 << (int(ell) - 1)

    def levels(self, k):
        return [ell for ell in range(1, self.n_levels + 1) if (ell, int(k)) in self.off]

    def slot(self, ell, k, j):
        return self.off[(int(ell), int(k))] + int(j)

    def prim(self, k):
        return self.prim_off + int(k)

    def cell_of(self, sid):
        sid = int(sid)
        if sid >= self.prim_off:
            return ("prim", sid - self.prim_off, 0)
        for (ell, k), o in self.off.items():
            if o <= sid < o + self.width:
                return (ell, k, sid - o)
        raise KeyError(sid)


class LevelLibrary:
    """Cells `(level, drilled seam)`; each cell holds slots and each slot holds members.

    A LEVEL-1 slot is `solo/MemberLibrary`'s: a k-means partition over content, members drawn
    from the rendition pool, key = the mean launch state. It has no `parents`.

    A LEVEL-l SLOT FOR l > 1 **IS A PAIR** of committed level-(l-1) slots — `parents = (a, b)`,
    the global ids of the two halves — and its members are auditioned WELDS of their members.
    The unit is launched with ONE feedback event and the internal seam's feedback is dropped:
    that is what the level buys. Because a slot's identity IS its spelling, `native/`'s
    can't-decompose readout has a form here for the first time on this substrate (pi's mass on a
    unit's own two parent slots), which `solo` finding 9 recorded as impossible at segment span.
    """

    def __init__(self, K_drill, n_slot, n_levels, poison=True):
        self.K = int(K_drill)
        self.n_slot = int(n_slot)
        self.n_levels = int(n_levels)
        self.layout = LadderLayout(self.K, self.n_slot, self.n_levels, poison=poison)
        self._cells = {key: [] for key in self.layout.off}

    def add_slot(self, ell, k, members, key, parents=None, j=None, poison=False, meta=None):
        c = self._cells[(int(ell), int(k))]
        jj = (self.n_slot if poison else (len(c) if j is None else int(j)))
        sl = dict(slot=int(self.layout.slot(ell, k, jj)), j=int(jj), level=int(ell), seam=int(k),
                  span=int(self.layout.span(ell)), poison=bool(poison),
                  parents=(None if parents is None else tuple(int(x) for x in parents)),
                  key=np.asarray(key, np.float32),
                  members=[dict(cmds=np.asarray(m["cmds"], np.float32),
                                key=np.asarray(m["key"], np.float32),
                                traj=(None if m.get("traj") is None
                                      else np.asarray(m["traj"], np.float32)),
                                score=float(m["score"]),
                                spell=(None if m.get("spell") is None else tuple(m["spell"])))
                           for m in members],
                  meta=meta or {})
        c.append(sl)
        return sl

    def slots(self, ell, k, with_poison=True):
        c = self._cells.get((int(ell), int(k)), [])
        return c if with_poison else [s for s in c if not s["poison"]]

    def levels_at(self, k, with_poison=False):
        return [ell for ell in range(1, self.n_levels + 1) if self.slots(ell, k, with_poison)]

    def prim(self, k):
        return int(self.layout.prim(int(k)))

    def slot_by_id(self, sid):
        ell, k, _ = self.layout.cell_of(int(sid))
        if ell == "prim":
            return None
        for s in self.slots(ell, k):
            if s["slot"] == int(sid):
                return s
        return None

    def legal_by_seam(self, levels=None, with_prim=True, with_poison=True):
        out = {}
        for k in range(self.K):
            ids = []
            for ell in range(1, self.n_levels + 1):
                if levels is not None and ell not in levels:
                    continue
                ids += [s["slot"] for s in self.slots(ell, k, with_poison)]
            if with_prim:
                ids.append(self.prim(k))
            out[k] = ids
        return out

    def sizes(self):
        return {f"L{ell}:{k}": [len(s["members"]) for s in v]
                for (ell, k), v in sorted(self._cells.items()) if v}

    def counts(self):
        return {f"L{ell}": sum(len(v) for (e, _), v in self._cells.items() if e == ell)
                for ell in range(1, self.n_levels + 1)}


# `build_level1_cell` is `solo/world.py::build_member_cell` under its ladder name: the op is
# identical (uniform draw, score on held-out hand-over states from the deploying configuration,
# keep the construction audition's own best `n_slot * m_member`, k-means over content, poison =
# the worst-scoring tail). Aliased rather than re-written so the two nodes cannot drift.
build_level1_cell = build_member_cell


def _weld(a_cmds, b_cmds):
    """THE NESTING OP, on content. A level-l unit's commands are the concatenation of its two
    committed level-(l-1) halves', played as ONE open-loop span from one feedback event.

    On a deterministic plant this is EXACT, and gate P-N asserts it: executing the weld from a
    state equals executing the first half and then the second half from wherever the first
    actually ended, at max|delta| = 0. `legato/` F4 measured the same thing on the arm from the
    other side — welding measured segment tapes cost -0.004 (free) while welding live plans cost
    2.2x — which is exactly why nothing modelled may enter a library (`span` F2/F4). So the level
    costs nothing in execution; what it changes is that the SECOND half was chosen at the FIRST
    seam, with the hand-over known at construction time instead of read Delta-old at run time.
    """
    return np.concatenate([np.asarray(a_cmds, np.float32), np.asarray(b_cmds, np.float32)], 0)


def build_level_cell(W, lib, ell, d, S0_score, rng, led, m_weld, n_slot, m_member, n_poison):
    """Build the level-`ell` cell at drilled seam `d` by welding committed level-(ell-1) pairs.

    THE OP, in order, mirroring `build_level1_cell` step for step so the two constructions are
    the same object one rung apart:
      (1) enumerate the candidate PAIRS `(a, b)` — every non-poison level-(ell-1) slot at drilled
          seam `d` crossed with every one at `d + span/2`. This is `T[l] subset T[l-1] x T[l-1]`.
      (2) for each pair draw up to `m_weld` member-welds UNIFORMLY (never ranked by their own
          realised outcome — etude E-3b's winner's curse).
      (3) score every weld by EXECUTING it on the plant from `S0_score` held-out hand-over states
          produced by the configuration that will deploy it (legato's seam-matched score set;
          presto decisions 6 and 9 — both the score set and the candidate pool come from the
          already-committed configuration, walked in order).
      (4) a PAIR's score is its best weld's. Keep the best `n_slot` pairs; each becomes a slot,
          with its best `m_member` welds as members.
      (5) the poison twin is the `n_poison` worst-scoring welds of the full draw, pinned to the
          reserved slot id.

    Returns `(slots, poison, rec)` or `(None, None, rec)` if a parent cell is empty.
    """
    sp = LadderLayout.span(ell)
    half = sp // 2
    A = lib.slots(ell - 1, d, with_poison=False)
    B = lib.slots(ell - 1, d + half, with_poison=False)
    rec = dict(level=int(ell), seam=int(d), span=int(sp), n_a=len(A), n_b=len(B))
    if not A or not B:
        rec["skipped"] = "empty parent cell"
        return None, None, rec
    cand = []                                   # (ai, bi, mi, mj, cmds, spell)
    for ai, a in enumerate(A):
        for bi, b in enumerate(B):
            grid = [(i, j) for i in range(len(a["members"])) for j in range(len(b["members"]))]
            take = grid if len(grid) <= int(m_weld) else [
                grid[t] for t in rng.permutation(len(grid))[:int(m_weld)]]
            for (i, j) in take:
                cand.append((ai, bi, int(i), int(j),
                             _weld(a["members"][i]["cmds"], b["members"][j]["cmds"]),
                             (int(a["slot"]), int(b["slot"]))))
    S0 = np.asarray(S0_score, np.float32)
    m = len(S0)
    nc = len(cand)
    kw = int(W.k0) + int(d)
    # one batched grounding call: chunking across the pool is bit-identical by construction
    # (`_pool_rollout` has no RNG), and 2 dispatches beat `nc` of them at 200+ candidates.
    cm = np.stack([c[4] for c in cand]).astype(np.float32)
    s0 = np.tile(S0[None], (nc, 1, 1)).reshape(nc * m, 4)
    cmb = np.repeat(cm, m, axis=0)
    errs, _ = W.rollout(s0, cmb, kw, sp, led)
    sc = errs.reshape(nc, m, sp).mean((1, 2))
    order = np.argsort(sc, kind="stable")
    pair_best = {}
    for t in order:                              # `order` is ascending, so first hit is the best
        key = (cand[t][0], cand[t][1])
        pair_best.setdefault(key, []).append(int(t))
    pairs = sorted(pair_best, key=lambda kk: sc[pair_best[kk][0]])[:int(n_slot)]
    used = set()
    slots = []
    for (ai, bi) in pairs:
        idx = pair_best[(ai, bi)][:int(m_member)]
        used.update(idx)
        a, b = A[ai], B[bi]
        slots.append(dict(
            parents=(int(a["slot"]), int(b["slot"])),
            key=np.asarray(a["key"], np.float32),
            members=[dict(cmds=cand[t][4], key=a["members"][cand[t][2]]["key"],
                          traj=(None if (a["members"][cand[t][2]].get("traj") is None
                                         or b["members"][cand[t][3]].get("traj") is None)
                                else np.concatenate([a["members"][cand[t][2]]["traj"],
                                                     b["members"][cand[t][3]]["traj"]], 0)),
                          score=float(sc[t]), spell=cand[t][5]) for t in idx],
            score=float(sc[idx[0]])))
    poison = [int(t) for t in order[::-1] if int(t) not in used][:int(n_poison)]
    pois = [dict(cmds=cand[t][4], key=A[cand[t][0]]["members"][cand[t][2]]["key"], traj=None,
                 score=float(sc[t]), spell=cand[t][5]) for t in poison]
    rec.update(n_pair=len(pair_best), n_cand=int(nc), n_score=int(m),
               chosen_score=float(sc[order[0]]), score_med=float(np.median(sc)),
               score_max=float(sc.max()),
               slot_scores=[[float(sc[t]) for t in pair_best[p][:int(m_member)]] for p in pairs],
               slot_parents=[[int(A[p[0]]["slot"]), int(B[p[1]]["slot"])] for p in pairs],
               n_distinct_a=len({p[0] for p in pairs}), n_distinct_b=len({p[1] for p in pairs}),
               poison_scores=[float(sc[t]) for t in poison])
    return slots, pois, rec


class FrozenReflexDecider:
    """INSTRUMENT, not an arm: the incumbent's OWN law, evaluated ONCE at the drilled launch and
    played open-loop to the end of the piece.

    `reflex_cmd` is `kp*(tgt(k,h) - x) + kd*(vtg(k) - v)` where `vtg` is the schedule's own
    feedforward velocity for the whole figure — so at a small `kp` the incumbent is playing the
    TOP-RUNG unit's content with a velocity servo on top. Freezing its read at the drilled launch
    and replaying gives exactly the content without the reads. The gap between this and the
    closed-loop reflex is what the 40 charged feedback events actually buy; when the gap is small
    the meter is counting reads the incumbent does not use.

    Pays one feedback event at the drilled launch (the same event every committed arm pays there)
    and zero groundings. The approach is played closed-loop, as for every arm.
    """

    def __init__(self, W):
        self.W = W
        self.picks = []

    def __call__(self, k, states, need, rng, led):
        W, H = self.W, self.W.H
        n = len(states)
        if int(k) < int(W.k0):
            return dict(kind="reflex", span=np.ones(n, int), src="approach")
        ns = int(W.K) - int(k)
        c = np.zeros((n, ns * H, 2), np.float32)
        for j in range(ns):
            for h in range(H):
                c[:, j * H + h, :] = W.reflex_cmd(states, int(k) + j, h)
        self.picks.append(dict(seam=int(k), drilled=int(k) - int(W.k0), n_legal=0, n_aud=0.0,
                               need=[int(b) for b in np.asarray(need, int)],
                               slot=[-3] * n, member=[-3] * n, level=[0] * n, mean_level=0.0,
                               frac_by_level={}, ns=[ns] * n))
        return dict(kind="open", cmds=c, span=np.full(n, ns, int), src="reflex_ol",
                    n_cand=0.0, frac_prim=0.0, mean_level=0.0)


class LadderDecider:
    """The seam decision for every Phase-A arm.

    `k < W.k0` — the APPROACH: the primitive plays, closed loop, one feedback event per step.
    Shared by every arm at every Delta and excluded from the comparison (presto decision 3).

    At a drilled seam, `mode`:
      "key"    — no audition at all: the slot whose stored launch key is nearest the OBSERVED
                 hand-over (normalised), then the nearest member key within it. 0 groundings,
                 1 feedback event. `legato/`'s `select_library`.
      "audit"  — audition every member of every legal slot ON THE PLANT from the observed
                 hand-over, launch the argmin. `m` groundings per legal slot.

    `levels` names which rungs the arm may use. `level_map` (drilled seam -> allowed levels)
    overrides it per seam, which is what builds the greedy nested prefix: seams already committed
    at level l play level l, the rest play l-1 (presto decision 6).

    Selection across levels compares the MEAN over the span's waypoints — `offbook/`'s cross-level
    convention, carried so the numbers stay comparable with `acappella`'s `lib_all` (whose
    delay-fragility, finding 6, is a known instrument and not a defect of this decider).
    """

    def __init__(self, W, lib, mode="audit", levels=(1,), level_map=None, obs_delay=0,
                 with_poison=False, fallback=True, capture=False):
        self.W, self.lib, self.mode = W, lib, str(mode)
        self.levels = tuple(int(x) for x in levels)
        self.level_map = level_map
        self.D = int(obs_delay)
        self.with_poison = bool(with_poison)
        self.fallback = bool(fallback)
        self.capture = bool(capture)
        self.picks = []
        self.rows = []

    def _legal(self, d):
        lv = self.levels if self.level_map is None else tuple(self.level_map.get(int(d), ()))
        out = []
        for ell in sorted(set(lv), reverse=True):
            for s in self.lib.slots(ell, d, with_poison=self.with_poison):
                out.append((int(ell), s))
        if not out and self.fallback:
            for ell in range(self.lib.n_levels, 0, -1):
                cell = self.lib.slots(ell, d, with_poison=self.with_poison)
                if cell:
                    out = [(int(ell), s) for s in cell]
                    break
        return out

    def __call__(self, k, states, need, rng, led):
        W, H = self.W, self.W.H
        n = len(states)
        d = int(k) - int(W.k0)
        if d < 0:
            return dict(kind="reflex", span=np.ones(n, int), src="approach")
        legal = self._legal(d)
        if not legal:
            return dict(kind="reflex", span=np.ones(n, int), src="no_content")
        maxspan = max(int(LadderLayout.span(e)) for e, _ in legal)
        out_c = np.zeros((n, maxspan * H, 2), np.float32)
        out_ns = np.ones(n, int)
        chosen, mem, lvl = [], [], []
        n_aud = 0.0

        if self.mode == "key":
            keys = np.array([s["key"] for _, s in legal], np.float32)
            sd = keys.std(0) + 1e-6
            pick = np.argmin(np.linalg.norm((states[:, None, :] - keys[None]) / sd[None, None],
                                            axis=2), axis=1)
            for i in range(n):
                ell, sl = legal[int(pick[i])]
                mk = np.array([mm["key"] for mm in sl["members"]], np.float32)
                mj = int(np.argmin(np.linalg.norm((states[i][None] - mk) / sd[None], axis=1)))
                sp = int(LadderLayout.span(ell))
                out_c[i, :sp * H] = sl["members"][mj]["cmds"]
                out_ns[i] = sp
                chosen.append(int(sl["slot"])); mem.append(mj); lvl.append(ell)
        else:
            scores = np.full((n, len(legal)), np.inf, np.float64)
            memb = np.full((n, len(legal)), -1, int)
            for ci, (ell, sl) in enumerate(legal):
                sp = int(LadderLayout.span(ell))
                mm = len(sl["members"])
                s0 = np.repeat(states, mm, axis=0)
                cm = np.tile(np.stack([b["cmds"] for b in sl["members"]]), (n, 1, 1))
                errs, _ = W.rollout(s0, cm, k, sp, led)
                e = errs.mean(1).reshape(n, mm)
                mi = e.argmin(1)
                scores[:, ci] = e[np.arange(n), mi]
                memb[:, ci] = mi
                n_aud += float(mm)
            best = np.argmin(scores, axis=1)
            for i in range(n):
                ell, sl = legal[int(best[i])]
                sp = int(LadderLayout.span(ell))
                mj = int(memb[i, int(best[i])])
                out_c[i, :sp * H] = sl["members"][mj]["cmds"]
                out_ns[i] = sp
                chosen.append(int(sl["slot"])); mem.append(mj); lvl.append(ell)

        lv = np.asarray(lvl, int)
        self.picks.append(dict(seam=int(k), drilled=int(d), n_legal=len(legal),
                               n_aud=float(n_aud), need=[int(b) for b in np.asarray(need, int)],
                               slot=[int(x) for x in chosen], member=[int(x) for x in mem],
                               level=[int(x) for x in lvl],
                               mean_level=float(lv.mean()),
                               frac_by_level={str(e): float((lv == e).mean())
                                              for e in sorted(set(lvl))},
                               ns=[int(x) for x in out_ns]))
        return dict(kind="open", cmds=out_c, span=out_ns, src=f"{self.mode}",
                    n_cand=float(len(legal)), frac_prim=0.0,
                    mean_level=float(lv.mean()))
