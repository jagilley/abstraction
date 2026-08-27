"""The a cappella world: `etude/`'s piece and metering, with **no forward model anywhere**.

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

from mjc.practice.acappella import piece as P

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
    def __init__(self, cfg, n_proc=1, rot=True):
        from mjc.pusher_env import PusherEnv
        self.cfg = cfg
        self.wps = np.array(P.WPS, np.float32)
        self.K = P.K_SEG
        self.H = int(cfg.get("seg_H", P.SEG_H))
        self.fs = int(cfg.get("frame_skip", P.FRAME_SKIP))
        self.dt_ctrl = self.fs * P.TIMESTEP
        self.d_fb = float(cfg.get("d_fb", P.D_FB))
        self.rot = bool(rot)
        self.dgp = P.dgp(rot)
        self.env = PusherEnv(self.dgp, with_puck=False)          # the BODY (in-parent, sequential)
        self.pool = RolloutPool(self.dgp, self.fs, n_proc)       # the resettable copies
        self.kp = float(cfg.get("kp", 0.0)); self.kd = float(cfg.get("kd", 0.0))

    def close(self):
        self.pool.close()

    # ------------------------------------------------------------------ geometry
    def start_states(self, rng, n):
        p = self.wps[0][None, :] + rng.uniform(-P.START_JIT, P.START_JIT, (n, 2)).astype(np.float32)
        v = rng.normal(0, P.V0_STD, (n, 2)).astype(np.float32)
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
        u = self.kp * (tgt[None, :] - states[:, :2]) + self.kd * (vtg[None, :] - states[:, 2:])
        return np.clip(u, -1.0, 1.0).astype(np.float32)

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
    def traverse(self, decide, starts, rng, led, explore=0.0, collect=False, obs_delay=0):
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
        hist = [states.copy()] if D > 0 else None

        def obs():
            """The agent's read. `hist[-1-D]`, clamped at the traversal start."""
            if D <= 0:
                return states
            return hist[max(0, len(hist) - 1 - D)]

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
        for k in range(Kn):
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
                    is_reflex[need] = False
                    c = np.asarray(d["cmds"], np.float32)
                    for j, b in enumerate(need):
                        L = int(sp[j]) * H
                        plan[b, :L] = c[j, :L]
                        off[b] = 0
                    left[need] = sp
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
                    a[idx] = self.reflex_cmd(obs()[idx], k, h)
                    led.add(fb=len(idx))          # the reflex pays one feedback event per STEP
                idx = np.where(~is_reflex)[0]
                if len(idx):
                    a[idx] = plan[idx, off[idx]]
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                raw_all[:, g0 + h, :] = a
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
                if D > 0:
                    hist.append(states.copy())
                off[~is_reflex] += 1
            wp_err[:, k] = np.linalg.norm(states[:, :2] - self.wps[k + 1][None, :], axis=1)
            left -= 1
        led.add(steps=Bn * Kn * H)
        out = dict(wp_err=wp_err, e_seg=np.median(wp_err, 0), e_piece=float(np.mean(np.median(wp_err, 0))),
                   seam=seam_states, final=np.linalg.norm(states[:, :2] - self.wps[-1][None, :], axis=1),
                   decisions=decisions, ledger=led.snap())
        if collect:
            out["acts"] = acts_all
            out["raw"] = raw_all
            for k in range(Kn):
                traces[k] = dict(s0=seam_states[k].copy(),
                                 cmds=acts_all[:, k * H:(k + 1) * H, :].copy(),
                                 raw=raw_all[:, k * H:(k + 1) * H, :].copy(),
                                 err=wp_err[:, k].copy())
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
