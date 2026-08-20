"""The fingering world: an approach, a hard passage, and a hand-over that carries information.

Program: `ideas/practice_manufactures_its_own_credit.md`. Direct parents:
  * `mjc/practice/etude/` -- where the practice arc started. Its finding 7 is this node's mandate:
    on the pusher a committed unit was a **state-independent** command sequence, so post-commit
    drift was exactly 0.0000 and fusion was provably vacuous. *Hierarchy is meaningful only over
    boundaries that carry information.*
  * `rhm/practice/crystallize/` -- the named fix, measured on sculpting: **state-conditioned
    commitment (a library keyed by the observed launch state) beats state-independent commitment
    1.8-3.0x**, while averaging valid realisations destroys them 3.6-5.0x.
  * `mjc/arm_substrate/` -- the plant. Composition horizon 14-23 steps at n=3-5; ballistic
    transmits FM quality 6x; capacity binds only at n>=5 (this node wants REDUNDANCY, not capacity
    pressure, so it runs at n=3).
  * `mjc/on_policy/COLLECTION_REALISM.md` -- practice is exactly its tier C (the allocation of
    experience is the dependent variable), so collection here is `Body`-metered and on-policy.

WHY THE ARM CAN ASK THE QUESTION THE PUSHER COULD NOT. The pusher's state is (position, velocity):
arrive at the same place with the same velocity and you are in the same state, full stop. A
3-link arm reaching a 2-D Cartesian waypoint is **kinematically redundant** -- the hand can arrive
at the same *place*, at the same *speed*, in a different *configuration*, and the torque sequence
that then carries it through the passage is a different one. That residual is not noise the
controller failed to remove; it is the null space, and a Cartesian upstream controller has no
reason to remove it. So the hand-over state is a genuine random variable *even when the upstream
movement succeeds*, which is precisely the boundary information the etude lacked.

THE PIECE. Two segments, executed back-to-back from the true achieved state:

    q0  --(approach, H_app, ballistic at tempo, frozen planner, motor noise)-->  W1
        --(drilled,  H_drill, the unit under study)                         -->  W2

  * The **approach** is a mastered skill: one full-horizon plan from the actual start posture,
    rolled out open-loop, computed by a planner over a FROZEN setup forward model. It is
    bit-identical in law across every arm, it corrects the start-posture jitter *in tip space*,
    and it does not correct it *in the null space*. Motor noise then makes the arrival a random
    variable in both. Nothing about it learns, so the hand-over distribution is stationary --
    which is what lets a library key it.
  * The **drilled** segment carries a localized hard region on its midpoint: a Gaussian-gated
    curl field (Shadmehr & Mussa-Ivaldi, `arm_env.curl_fields`), smooth and open-loop compensable,
    with an optional soft obstacle (`arm_env.push_fields`) held in reserve for the multimodality
    gate. The forward model is pretrained with in-region transitions REJECTED
    (`pretrain_mode="exclude"`, the etude's pattern), so the passage is *unmodelled* rather than
    *wrongly modelled* -- the difference matters because the same pool is the replay pool.
  * `H_drill` is set at or beyond the plant's composition horizon, so committing to an open-loop
    sequence is a genuine bet (gate G4).

PRICING (the etude's model, counts logged separately so the price vector stays post-hoc editable).
A control step costs `dt_ctrl = frame_skip * timestep`; every feedback/replan event costs a
declared sensorimotor delay `d_fb`. A reactive drilled segment pays H_drill delays; a committed one
pays exactly one, at the boundary -- and unlike the etude, a *library* unit actually consumes the
observation that event pays for. Auditions, scoring rollouts and the score-set traversals are all
charged. Oracle-only readouts (the ceiling FM, ground-truth region membership) are free and are
labelled as such.

`import mujoco` / `import torch` live inside functions so this module is importable from a laptop
that has neither -- the `arm_env.py` / `pusher_env.py` contract.
"""

import numpy as np

# ------------------------------------------------------------------ piece geometry helpers


def planar_jacobian(q, Ls):
    """Analytic 2xN tip Jacobian of a planar serial chain. J[:, k] = sum_{i>=k} L_i * d/dq of the
    link-i contribution, which for relative joint angles is `L_i * [-sin a_i, cos a_i]`."""
    q = np.asarray(q, dtype=np.float64)
    Ls = np.asarray(Ls, dtype=np.float64)
    a = np.cumsum(q)
    n = len(q)
    J = np.zeros((2, n))
    for k in range(n):
        J[0, k] = -(Ls[k:] * np.sin(a[k:])).sum()
        J[1, k] = (Ls[k:] * np.cos(a[k:])).sum()
    return J


def null_direction(q, Ls):
    """Unit vector spanning the self-motion manifold of a 3-link planar arm at `q` (the 1-D null
    space of the 2x3 tip Jacobian). Returned unnormalised-then-normalised; zero if degenerate."""
    J = planar_jacobian(q, Ls)
    if J.shape[1] != 3:
        # general case: smallest right singular vector
        _, _, Vt = np.linalg.svd(J)
        v = Vt[-1]
    else:
        v = np.cross(J[0], J[1])
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else np.zeros_like(v)


def start_postures(qc, Ls, m, rng, iso_jit: float, null_jit: float, mode: str = "iso",
                   project_iters: int = 6):
    """Draw `m` start postures for the piece.

    `mode="iso"`   -- q0 ~ q_center + U(-iso_jit, iso_jit)^n. The established idiom (`q_jit` in
                      `on_policy/`), and the honest default: a body does not start twice in the
                      same place.
    `mode="null"`  -- postural spread with the START TIP HELD FIXED. Sample an isotropic jitter,
                      then Gauss-Newton project back onto {q : fk(q) = fk(q_center)} and add a
                      draw along the self-motion manifold. This isolates the redundancy channel:
                      every performer's hand starts at the same *place* in a different
                      *configuration*, so any hand-over information the drilled segment gets is
                      provably postural rather than positional. Kept as a knob because whether the
                      cheap `iso` mode already delivers boundary information is gate G1's job to
                      measure, not ours to assume.
    """
    from mjc.arm_env import fk

    qc = np.asarray(qc, dtype=np.float64)
    n = len(qc)
    Q = qc[None, :] + rng.uniform(-iso_jit, iso_jit, (m, n))
    if mode != "null":
        return Q
    p_star = fk(qc, Ls)
    for i in range(m):
        for _ in range(project_iters):
            e = p_star - fk(Q[i], Ls)
            if np.linalg.norm(e) < 1e-6:
                break
            J = planar_jacobian(Q[i], Ls)
            Q[i] = Q[i] + np.linalg.pinv(J) @ e
        Q[i] = Q[i] + null_jit * rng.normal() * null_direction(Q[i], Ls)
    return Q


def kmeans(X, k, rng, iters: int = 40):
    """Plain k-means++ on standardized rows. Small k, small n -- no need for a dependency."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    k = int(min(k, n))
    C = np.empty((k, X.shape[1]))
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        C[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
    lab = np.zeros(n, int)
    for _ in range(iters):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = D.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return C, lab


# ------------------------------------------------------------------ the world


class Ledger:
    """Every environment step, feedback event and priced second, split by who it was for.

    `agent` = anything that informs a decision the agent makes (practice, metering the agent reads,
    auditions, score-set traversals). `instrument` = experimenter-side readouts the agent never
    sees (the held-out ladder, the ceiling-FM reference, ground-truth region membership). The split
    is the thing `COLLECTION_REALISM.md` §5 asks for, made structural instead of remembered.
    """

    def __init__(self):
        self.steps = {"agent": 0, "instrument": 0}
        self.fb = {"agent": 0, "instrument": 0}
        self.t_priced = 0.0            # agent-side priced time -- the headline currency
        self.t_by_kind = {}
        # RAW COUNTS per kind. `t_by_kind` is already-priced seconds and therefore bakes in one
        # choice of `d_fb`; these do not. Pricing in this design is pure bookkeeping (fixed cycle
        # counts, no in-run decision reads the ledger), so keeping the counts makes the whole
        # commitment-pays question a POST-HOC sweep over the price vector rather than a family of
        # runs. Round 1's inversion is exactly the kind of result that needs that curve.
        self.n_by_kind = {}
        # DELIBERATION EVENTS, counted and never charged. No node in this repo has ever priced a
        # plan -- the etude logged plan counts "so a deliberation price can be applied post-hoc"
        # and no round ever applied one. Counting makes "what does frozen content actually buy"
        # quantitative for free: a reactive traversal deliberates H times, a plan-at-launch unit
        # once, a frozen unit not at all, so the three ops separate on an axis that is invisible
        # under a feedback-only price. `d_plan` defaults to 0, so `t_priced` is byte-unchanged and
        # every earlier number stays comparable; the sweep is post-hoc.
        self.plans = {"agent": 0, "instrument": 0}

    def charge(self, who, steps, fb, dt_ctrl, d_fb, kind="other", plans=0):
        self.steps[who] += int(steps)
        self.fb[who] += int(fb)
        self.plans[who] += int(plans)
        t = steps * dt_ctrl + fb * d_fb
        if who == "agent":
            self.t_priced += t
            self.t_by_kind[kind] = self.t_by_kind.get(kind, 0.0) + t
            n = self.n_by_kind.setdefault(kind, {"steps": 0, "fb": 0, "plans": 0})
            n["steps"] += int(steps)
            n["fb"] += int(fb)
            n["plans"] += int(plans)
        return t

    def snapshot(self):
        return {"steps_agent": self.steps["agent"], "steps_instrument": self.steps["instrument"],
                "fb_agent": self.fb["agent"], "fb_instrument": self.fb["instrument"],
                "plans_agent": self.plans["agent"], "plans_instrument": self.plans["instrument"],
                "t_priced": self.t_priced, "t_by_kind": dict(self.t_by_kind),
                "n_by_kind": {k: dict(v) for k, v in self.n_by_kind.items()}}


class World:
    """The plant, the piece, the forward models and the controllers, in one object.

    Deliberately not a fork of `etude.py`'s single-function runner: two rounds share this substrate
    (the gate probe and the arm comparison), and the repo rule is that prior results stay
    reproducible -- so nothing here edits the etude, and the etude's structure is copied rather
    than mutated.
    """

    def __init__(self, cfg: dict, device: str):
        import torch
        from mjc.arm_env import fk

        self.cfg = cfg
        self.device = device
        self.n = int(cfg["n_links"])
        self.SD, self.AD = 2 * self.n, self.n
        self.Ls = np.asarray(cfg["link_lengths"][: self.n], dtype=np.float64)
        self.torch = torch
        self.qc = np.asarray(cfg["q_center"][: self.n], dtype=np.float64)
        self.fs = int(cfg["frame_skip"])
        self.dt_ctrl = self.fs * float(cfg["timestep"])
        self.d_fb = float(cfg["d_fb"])
        self.H_app = int(cfg["h_app"])
        self.H_drill = int(cfg["h_drill"])
        self.W1 = np.asarray(cfg["w1"], dtype=np.float32)
        self.W2 = np.asarray(cfg["w2"], dtype=np.float32)
        self._fk = fk

        # ---- the hard region: a Gaussian-gated curl patch on the drilled segment's midpoint,
        # plus an OPTIONAL soft obstacle held in reserve for the multimodality gate (G2).
        self.patch_center = np.asarray(
            cfg.get("patch_center") or (0.5 * (self.W1 + self.W2)), dtype=np.float64)
        self.patch_sigma = float(cfg["patch_sigma"])
        self.curl_b = float(cfg["curl_b"])
        self.push_a = float(cfg.get("push_a", 0.0))
        self.push_sigma = float(cfg.get("push_sigma", self.patch_sigma))
        self.push_center = np.asarray(
            cfg.get("push_center") or self.patch_center, dtype=np.float64)

        self.env = self._make_env(patch=True)
        self.env_clean = self._make_env(patch=False)
        self._bodies = {}
        self.Lt = torch.tensor(self.Ls, device=device, dtype=torch.float32)
        self.norm = None

    # ------------------------------------------------------------ plant

    def _make_env(self, patch: bool):
        from mjc.arm_env import ArmEnv

        dgp = dict(n_links=self.n,
                   link_lengths=list(self.cfg["link_lengths"][: self.n]),
                   link_masses=list(self.cfg["link_masses"][: self.n]),
                   joint_damping=self.cfg["joint_damping"], gear=self.cfg["gear"],
                   timestep=self.cfg["timestep"])
        if patch:
            if self.curl_b != 0.0:
                dgp["curl_fields"] = [dict(b=self.curl_b,
                                           center=tuple(self.patch_center.tolist()),
                                           sigma=self.patch_sigma)]
            if self.push_a != 0.0:
                dgp["push_fields"] = [dict(a=self.push_a,
                                           center=tuple(self.push_center.tolist()),
                                           sigma=self.push_sigma)]
        return ArmEnv(dgp)

    def body(self, n_par: int):
        """A cached metered `Body` per parallel width. `Body` swaps `env.data`, so every caller
        must `release()` when done -- `traverse` does."""
        from mjc.embodied import Body

        if n_par not in self._bodies:
            self._bodies[n_par] = Body(self.env, n_par=n_par, frame_skip=self.fs,
                                       wrap_limit=self.cfg.get("wrap_limit", 3.0))
        return self._bodies[n_par]

    def tip(self, states):
        """Cartesian tip of a batch of states (numpy). Analytic FK, verified to 2e-16 against
        MuJoCo's own `site_xpos` in `arm_substrate` P0."""
        s = np.atleast_2d(np.asarray(states, dtype=np.float64))
        return self._fk(s[:, : self.n], self.Ls)

    def gate(self, tips):
        """Region membership weight in [0, 1] -- the max over the active local fields. Oracle-only
        (the agent never reads it); used to build the `exclude` pretraining diet and to report
        exposure."""
        t = np.atleast_2d(np.asarray(tips, dtype=np.float64))
        g = np.zeros(len(t))
        if self.curl_b != 0.0:
            d = t - self.patch_center[None, :]
            g = np.maximum(g, np.exp(-(d ** 2).sum(1) / (2 * self.patch_sigma ** 2)))
        if self.push_a != 0.0:
            d = t - self.push_center[None, :]
            g = np.maximum(g, np.exp(-(d ** 2).sum(1) / (2 * self.push_sigma ** 2)))
        return g

    # ------------------------------------------------------------ forward model

    def mlp(self, seed, din=None, dout=None, hidden=None, layers=None, out_act=None):
        import torch
        import torch.nn as nn

        din = self.SD + self.AD if din is None else din
        dout = self.SD if dout is None else dout
        h = self.cfg["fm_hidden"] if hidden is None else hidden
        L = self.cfg["fm_layers"] if layers is None else layers
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, dout)]
        if out_act is not None:
            lyr += [out_act]
        net = nn.Sequential(*lyr)
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5, generator=g)
                nn.init.zeros_(m.bias)
        return net.to(self.device)

    def set_norm(self, S, U, S2):
        import torch

        X = np.concatenate([S, U], 1).astype(np.float32)
        Y = (S2 - S).astype(np.float32)
        self.norm = {k: torch.tensor(v, device=self.device) for k, v in dict(
            mx=X.mean(0), sx=X.std(0) + 1e-6, my=Y.mean(0), sy=Y.std(0) + 1e-6).items()}

    def tensors(self, S, U, S2):
        import torch

        X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=self.device)
        Y = torch.tensor((S2 - S).astype(np.float32), device=self.device)
        return (X - self.norm["mx"]) / self.norm["sx"], (Y - self.norm["my"]) / self.norm["sy"]

    def train_steps(self, net, opt, S, U, S2, steps, brng, bs=None):
        import torch
        import torch.nn as nn

        Xn, Yn = self.tensors(S, U, S2)
        huber = nn.HuberLoss(delta=1.0)
        bs = min(bs or self.cfg["fm_batch"], len(S))
        net.train()
        for _ in range(steps):
            idx = torch.tensor(brng.integers(0, len(S), size=bs), device=self.device)
            opt.zero_grad()
            huber(net(Xn[idx]), Yn[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def train_online(self, net, opt, PX, PY, RX, RY, steps, brng, bs, rfrac):
        """Plain uniform-lr Adam on part fresh practice experience, part replay of the pretrain
        pool. No per-sample delta gain anywhere in this node -- the practice audit reduced delta's
        surviving role to DETECTION, and that is all it is used for."""
        import torch
        import torch.nn as nn

        huber = nn.HuberLoss(delta=1.0)
        nb = max(1, int(round(bs * (1.0 - rfrac))))
        nr = max(0, bs - nb)
        net.train()
        for _ in range(steps):
            i = torch.tensor(brng.integers(0, len(PX), size=nb), device=self.device)
            xs, ys = [PX[i]], [PY[i]]
            if nr and len(RX):
                j = torch.tensor(brng.integers(0, len(RX), size=nr), device=self.device)
                xs.append(RX[j]); ys.append(RY[j])
            opt.zero_grad()
            huber(net(torch.cat(xs)), torch.cat(ys)).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()

    def fm_delta(self, net, S, U):
        import torch

        with torch.no_grad():
            X = torch.tensor(np.concatenate([S, U], 1).astype(np.float32), device=self.device)
            return (net((X - self.norm["mx"]) / self.norm["sx"]) * self.norm["sy"]
                    + self.norm["my"]).cpu().numpy()

    # ------------------------------------------------------------ the diet (on-policy, metered)

    def collect_ou(self, n, rng, ep_len=None, n_par=None, q_range=None):
        """Rung B1 -- correlated random torque, no forward model in the loop. The bootstrap: you
        cannot plan a reach with a model you have not trained yet, and pretending otherwise is how
        an on-policy setup quietly becomes a teleport one."""
        from mjc.embodied import collect_on_policy, OUBehaviour

        cfg = self.cfg
        n_par = int(n_par or cfg["n_par"])
        beh = OUBehaviour(self.AD, n_par, rng, sigma=cfg["ou_sigma"], theta=cfg["ou_theta"])
        return collect_on_policy(self.env, int(n), rng, self.fs, self.qc,
                                 float(q_range if q_range is not None else cfg["op_q_range"]),
                                 behaviour=beh, ep_len=int(ep_len or cfg["ep_len"]),
                                 n_par=n_par, wrap_limit=cfg.get("wrap_limit", 3.0),
                                 return_info=True)

    def collect_reach(self, n, rng, fm, goal_sampler, ep_len=None, n_par=None, q_range=None):
        """Rung B3 -- reaches under the LIVE model, which is the real situation and carries the
        genuine bootstrapping coupling (a stale model moves badly, which gets you worse data)."""
        from mjc.embodied import collect_on_policy, ReachBehaviour

        cfg = self.cfg
        n_par = int(n_par or cfg["n_par"])
        ep_len = int(ep_len or cfg["ep_len"])
        pf = self.plan_fn(fm, ep_len, k_shoot=cfg["collect_k_shoot"],
                          cem_iters=cfg["collect_cem_iters"])
        beh = ReachBehaviour(lambda s, g: pf(s, g, rng), np.zeros((n_par, 2), np.float32), rng,
                             sigma_u=cfg["sigma_u"], replan_every=ep_len)
        return collect_on_policy(self.env, int(n), rng, self.fs, self.qc,
                                 float(q_range if q_range is not None else cfg["op_q_range"]),
                                 behaviour=beh, ep_len=ep_len, n_par=n_par,
                                 goal_sampler=goal_sampler,
                                 wrap_limit=cfg.get("wrap_limit", 3.0), return_info=True)

    def exclude_region(self, S, U, S2, w=0.1):
        """The etude's `pretrain_mode="exclude"`: keep only transitions whose tip is OUTSIDE the
        hard region, so the passage is *unmodelled* rather than *wrongly modelled*. The difference
        is load-bearing because this pool is also the replay pool, and a wrongly-modelled passage
        in replay actively fights the adaptation practice is supposed to drive."""
        keep = self.gate(self.tip(S)) <= w
        return S[keep], U[keep], S2[keep], float(keep.mean())

    # ------------------------------------------------------------ controller

    def fk_torch(self, q):
        import torch

        ang = torch.cumsum(q, dim=1)
        return torch.stack([(self.Lt * torch.cos(ang)).sum(1),
                            (self.Lt * torch.sin(ang)).sum(1)], 1)

    def plan_fn(self, net, hh, k_shoot=None, cem_iters=None):
        """CEM-MPC over the FM, Cartesian tip cost. Sized per `arm_substrate` P3, whose warning is
        the single biggest confound on this plant: a pusher-tuned budget silently converts the
        ballistic transmission effect into a null on a higher-DOF arm."""
        import torch

        cfg = self.cfg
        K = int(k_shoot or cfg["k_shoot"])
        I = int(cem_iters or cfg["cem_iters"])
        ne, vp, s0sig = cfg["cem_elite"], cfg["vel_pen"], cfg["cem_init_sigma"]
        n, AD = self.n, self.AD

        def plan(states, goals, rng):
            states = np.asarray(states, np.float32)
            goals = np.asarray(goals, np.float32)
            Bn = states.shape[0]
            mu = np.zeros((Bn, hh, AD), np.float32)
            sig = np.full((Bn, hh, AD), s0sig, np.float32)
            g_t = torch.tensor(goals, device=self.device).repeat_interleave(K, 0)
            s0 = torch.tensor(states, device=self.device).repeat_interleave(K, 0)
            for _ in range(I):
                e = rng.standard_normal((Bn, K, hh, AD)).astype(np.float32)
                seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
                with torch.no_grad():
                    s = s0.clone()
                    seqs_t = torch.tensor(seqs.reshape(Bn * K, hh, AD), device=self.device)
                    cost = torch.zeros(Bn * K, device=self.device)
                    for h in range(hh):
                        x = torch.cat([s, seqs_t[:, h, :]], 1)
                        s = s + (net((x - self.norm["mx"]) / self.norm["sx"]) * self.norm["sy"]
                                 + self.norm["my"])
                        cost = cost + (self.fk_torch(s[:, :n]) - g_t).norm(dim=1)
                    cost = cost + vp * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, K), min(ne, K), dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1)
                sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)

        return plan

    # ------------------------------------------------------------ the units

    @staticmethod
    def reactive_unit():
        return {"kind": "reactive"}

    def unit_commands(self, unit, hand, fm, rng, h_index=None, cache=None):
        """The command a unit issues. Returns the full (B, H_drill, AD) plan for committed units
        (one feedback event, consumed at launch) -- reactive units are handled inside `traverse`
        because they re-plan mid-flight."""
        kind = unit["kind"]
        B = len(hand)
        if kind == "fixed":
            return np.tile(unit["fixed"][None], (B, 1, 1)).astype(np.float32)
        if kind == "lib":
            # THE POINT OF THE NODE: the feedback event at the boundary is charged either way, and
            # here it is actually SPENT -- the observed hand-over state selects which committed
            # rendition launches. Nearest key in the standardized state the library was built in.
            Z = (hand - unit["key_mu"]) / unit["key_sd"]
            D = ((Z[:, None, :] - unit["keys"][None, :, :]) ** 2).sum(-1)
            j = D.argmin(1)
            return unit["seqs"][j].astype(np.float32)
        if kind == "plan_launch":
            # LIVE CONTENT UNDER COMMITTED ROUTING. The boundary's feedback event -- charged in
            # every committed arm and, in the etude, never actually spent -- buys a fresh
            # full-horizon plan from the observed hand-over under the CURRENT forward model. So
            # the routing is committed (one re-grounding, one delay, open-loop for H steps) while
            # the CONTENT is live. This is the third corner of the op taxonomy: `fixed`/`library`
            # commit content and routing, `never` commits neither.
            #
            # The etude's discriminator ran this head-to-head once (live CEM 0.0807 vs a frozen
            # selected trace 0.0763) and the frozen trace won -- but on a boundary that carried no
            # information, where a state-conditioned plan has nothing to condition on. This is the
            # first time it is asked on a boundary that does.
            #
            # `rng` is the traversal's own stream, so the planner's CEM draws are independent per
            # traversal. Note the ladder's `e_ball` probe deliberately does NOT route through here
            # -- it keeps using REACT + drill_replan=H_drill, so the reference curve stays the
            # exact debugged path it has been all along.
            return self.plan_fn(fm, self.H_drill)(
                hand, np.tile(self.W2[None, :], (B, 1)), rng)
        if kind == "lib_proj":
            Z = np.concatenate([(hand - unit["key_mu"]) / unit["key_sd"],
                                np.ones((B, 1), np.float32)], 1)
            j = np.searchsorted(unit["edges"], Z @ unit["beta"])
            return unit["seqs"][np.clip(j, 0, len(unit["seqs"]) - 1)].astype(np.float32)
        if kind == "bc":
            import torch

            with torch.no_grad():
                x = (torch.tensor(hand.astype(np.float32), device=self.device)
                     - unit["pnorm"]["mu"]) / unit["pnorm"]["sd"]
                y = unit["pol"](x).cpu().numpy()
            return y.reshape(B, self.H_drill, self.AD).astype(np.float32)
        raise ValueError(f"unit kind {kind!r} has no launch-time plan")

    @staticmethod
    def unit_fb(unit):
        """Feedback/replan events a unit costs per traversal of the drilled segment."""
        return 1 if unit["kind"] != "reactive" else None      # None => H_drill / replan_every

    # ------------------------------------------------------------ traversal

    def traverse(self, fm, unit, q0, rng, sigma, ledger, who="agent", kind="other",
                 drill_replan=1, approach_plan=None, collect=False, qd0=None, drill=True):
        """Execute the whole piece once for `len(q0)` performers, through the metered `Body`.

        Boundaries are re-grounding points, not teleports: the drilled segment starts from the
        state the approach actually achieved. Motor noise is applied to the issued command in BOTH
        practice and performance -- the etude's run-throughs were noise-free, which is exactly why
        its post-commit drift was identically 0.0000 (a deterministic chain in a deterministic
        world against a fixed geometry has a constant metering value). A hand that arrives in the
        same place twice is not a plant, it is an artifact.
        """
        q0 = np.asarray(q0, dtype=np.float64)
        B = len(q0)
        body = self.body(B)
        try:
            s = body.reset(rng, self.qc, 0.0, q0=q0, qd0=qd0)
            g1 = np.tile(self.W1[None, :], (B, 1))
            g2 = np.tile(self.W2[None, :], (B, 1))
            trS, trU, trS2 = [], [], []

            # ---- segment 0: the approach. One plan, rolled open-loop at tempo. ----
            plan = (approach_plan if approach_plan is not None
                    else self.plan_fn(fm, self.H_app)(s, g1, rng))
            for h in range(self.H_app):
                a = np.clip(plan[:, h, :], -1, 1)
                if sigma > 0:
                    a = np.clip(a + sigma * rng.standard_normal(a.shape), -1, 1)
                sa, ua, sb, _ = body.step(a.astype(np.float32))
                if collect:
                    trS.append(sa); trU.append(ua); trS2.append(sb)
                s = sb
            hand = s.copy()
            e_app = np.linalg.norm(self.tip(hand) - self.W1[None, :], axis=1)
            fb = 1

            # ---- segment 1: the drilled passage. ----
            acts = np.empty((B, self.H_drill, self.AD), np.float32)
            acts_raw = np.empty_like(acts)
            gate_hits = 0
            n_plan = 0
            if not drill:
                # hand-over harvesting only: run the approach, record the arrival. Used to build
                # the audition score set and the composition-horizon probe, and charged for the
                # approach it actually executed.
                e_drill = np.full(B, np.nan)
            elif unit["kind"] == "reactive":
                re = self.H_drill if drill_replan <= 0 else min(drill_replan, self.H_drill)
                dplan = None
                pf = self.plan_fn(fm, self.H_drill)
                for h in range(self.H_drill):
                    if h % re == 0:
                        dplan = pf(s, g2, rng)
                        fb += 1
                        n_plan += 1        # a reactive traversal deliberates H_drill/re times
                    a0 = np.clip(dplan[:, h % re, :], -1, 1)
                    acts_raw[:, h, :] = a0
                    a = a0 if sigma <= 0 else np.clip(
                        a0 + sigma * rng.standard_normal(a0.shape), -1, 1)
                    acts[:, h, :] = a
                    gate_hits += int((self.gate(self.tip(s)) > 0.3).sum())
                    sa, ua, sb, _ = body.step(a.astype(np.float32))
                    if collect:
                        trS.append(sa); trU.append(ua); trS2.append(sb)
                    s = sb
            else:
                dplan = self.unit_commands(unit, hand, fm, rng)
                fb += 1
                # only a LIVE-content unit deliberates at the boundary; a frozen one issues
                # stored content and the charged feedback event buys nothing
                n_plan += int(unit["kind"] == "plan_launch")
                for h in range(self.H_drill):
                    a0 = np.clip(dplan[:, h, :], -1, 1)
                    acts_raw[:, h, :] = a0
                    a = a0 if sigma <= 0 else np.clip(
                        a0 + sigma * rng.standard_normal(a0.shape), -1, 1)
                    acts[:, h, :] = a
                    gate_hits += int((self.gate(self.tip(s)) > 0.3).sum())
                    sa, ua, sb, _ = body.step(a.astype(np.float32))
                    if collect:
                        trS.append(sa); trU.append(ua); trS2.append(sb)
                    s = sb
            if drill:
                e_drill = np.linalg.norm(self.tip(s) - self.W2[None, :], axis=1)
        finally:
            body.release()

        steps = B * (self.H_app + (self.H_drill if drill else 0))
        t_piece = ledger.charge(who, steps, B * fb, self.dt_ctrl, self.d_fb, kind=kind,
                                plans=B * n_plan)
        out = dict(hand=hand, final=s.copy(), e_app=e_app, e_drill=e_drill,
                   acts=acts, acts_raw=acts_raw, n_fb=fb, t_piece=t_piece / max(B, 1),
                   t_total=t_piece, gate_frac=gate_hits / float(B * self.H_drill))
        if collect:
            out["trans"] = (np.concatenate(trS, 0), np.concatenate(trU, 0),
                            np.concatenate(trS2, 0))
        return out

    # ------------------------------------------------------------ the compile op

    def replay(self, S0, cmds, goal):
        """Execute one command sequence open-loop from each of `S0`; return boundary errors.

        This is the one place the agent re-enters a state it has already been in. It is a *mental
        replay of a remembered arrival*, not an omniscient workspace sample: every state in `S0`
        was produced by genuine at-tempo upstream execution and recorded. It is fully priced
        (`audition_cost`), and it is what makes a PAIRED comparison of candidates possible at all.
        """
        errs = np.empty(len(S0), np.float32)
        n = self.n
        for i in range(len(S0)):
            self.env.set_state(S0[i, :n].astype(np.float64), S0[i, n:].astype(np.float64))
            for h in range(cmds.shape[0]):
                self.env.step(cmds[h], self.fs)
            errs[i] = np.linalg.norm(self.tip(self.env.get_state()[None, :])[0] - goal)
        return errs

    def audition(self, cands, S0, goal, ledger, who="agent"):
        """The (n_cand, n_score) score matrix: every candidate replayed from every held-out
        hand-over state. Priced as `n_cand * n_score` committed traversals of the drilled segment.

        Candidates are drawn UNIFORMLY from the trace pool by the caller, never top-of-pool:
        ranking by a candidate's own realised error is a winner's curse -- it favours the sequence
        most finely tuned to its own start state, i.e. the least transferable one (etude E-3b;
        crystallize measures the same curse at 3.3x)."""
        E = np.empty((len(cands), len(S0)), np.float32)
        for j, c in enumerate(cands):
            E[j] = self.replay(S0, c, goal)
        ledger.charge(who, len(cands) * len(S0) * self.H_drill, len(cands) * len(S0),
                      self.dt_ctrl, self.d_fb, kind="audition")
        return E

    @staticmethod
    def select_fixed(E, cands):
        """E-3b's op: commit the argmin of the MEAN over the consumption distribution."""
        j = int(np.argmin(E.mean(1)))
        return {"kind": "fixed", "fixed": np.asarray(cands[j], np.float32)}, j

    def select_library(self, E, cands, S0, n_lib, rng):
        """State-conditioned commitment: one committed rendition per cell of the hand-over
        distribution, keyed at launch by the observed state.

        Note what this does NOT cost: the score matrix is the SAME matrix `select_fixed` reads.
        Both arms audition the same candidates on the same states at the same price; `fixed` takes
        the argmin of the row mean over ALL columns, `library` takes it per column-block. The
        library's whole advantage is therefore information use, not budget -- exactly the isolation
        `crystallize` ran (same pool, same commit cycle, same two groundings).
        """
        mu, sd = S0.mean(0), S0.std(0) + 1e-6
        Z = (S0 - mu) / sd
        C, lab = kmeans(Z, n_lib, rng)
        k = len(C)
        seqs = np.empty((k, self.H_drill, self.AD), np.float32)
        picks, sizes = [], []
        for c in range(k):
            m = lab == c
            if not m.any():
                j = int(np.argmin(E.mean(1)))
            else:
                j = int(np.argmin(E[:, m].mean(1)))
            seqs[c] = cands[j]
            picks.append(j)
            sizes.append(int(m.sum()))
        jd = int(np.argmin(E.mean(1)))
        return ({"kind": "lib", "keys": C, "key_mu": mu, "key_sd": sd, "seqs": seqs,
                 "default": np.asarray(cands[jd], np.float32)},
                dict(picks=picks, cell_sizes=sizes, n_distinct=len(set(picks)), default=jd))

    def select_library_proj(self, E, cands, S0, n_lib):
        """The same library, keyed on a **supervised 1-D projection** of the hand-over state
        instead of an isotropic partition of it.

        Round 0 measured the gap this exists to close: the per-state oracle over the same
        candidates beats the best single fixed unit **4.0x**, but a 4-cell k-means library
        recovered only **1.07x** of it, with 2 of 4 cells holding duplicate picks. The information
        is there and the extractor was not finding it -- and round 0 also says why: a *linear*
        function of the hand-over state explains **82%** of the realised-error variance, so the
        structure lives along one direction that isotropic k-means has no reason to align with.

        The key is fit from the audition matrix the arm has ALREADY PAID FOR -- regress the chosen
        fixed unit's per-state error on the standardized hand-over state, then cut the resulting
        scalar at `n_lib` quantiles. No extra rollouts, no extra price, and the cells come out
        balanced by construction (which k-means's did not: 58/39/41/54 at round 0's b14).
        """
        mu, sd = S0.mean(0), S0.std(0) + 1e-6
        Z = np.concatenate([(S0 - mu) / sd, np.ones((len(S0), 1))], 1)
        jd = int(np.argmin(E.mean(1)))
        beta, *_ = np.linalg.lstsq(Z, E[jd], rcond=None)
        t = Z @ beta
        r2 = float(1.0 - (E[jd] - t).var() / max(E[jd].var(), 1e-12))
        edges = np.quantile(t, np.linspace(0.0, 1.0, n_lib + 1)[1:-1])
        lab = np.searchsorted(edges, t)
        seqs = np.empty((n_lib, self.H_drill, self.AD), np.float32)
        picks, sizes = [], []
        for c in range(n_lib):
            m = lab == c
            j = int(np.argmin(E[:, m].mean(1))) if m.any() else jd
            seqs[c] = cands[j]
            picks.append(j)
            sizes.append(int(m.sum()))
        return ({"kind": "lib_proj", "beta": beta, "edges": edges, "key_mu": mu, "key_sd": sd,
                 "seqs": seqs, "default": np.asarray(cands[jd], np.float32)},
                dict(picks=picks, cell_sizes=sizes, n_distinct=len(set(picks)),
                     r2_of_key=r2, beta=beta.tolist(), edges=edges.tolist()))

    @staticmethod
    def pool_geometry(A):
        """Geometry of the candidate command pool -- **logged, never acted on**.

        Round 0's G2 inverted: the position-wise mean of successful renditions came out BETTER than
        the median contributor (0.82-0.90x in all four worlds), where the etude measured a 2.1x
        averaging PENALTY on the pusher under a comparably smooth field and the same op. A local
        convexity story cannot be the whole explanation, or the pusher would have shown it too. The
        candidate for what actually differs is the pool's *phase structure*: averaging survives a
        phase-LOCKED pool (every rendition issues roughly the same command at roughly the same
        step, so the mean is a denoised rendition) and destroys a phase-SCATTERED one (the mean of
        two time-shifted solutions is neither).

        `coherence` = ||mean sequence|| / mean ||sequence||: ~1 when phase-locked, ~0 when
        scattered. Logged per commit for every arm, at zero cost, so the geometry question can be
        adjudicated later from the run JSONs without building anything for it now.
        """
        A = np.asarray(A, np.float64)
        m = len(A)
        mean_seq = A.mean(0)
        nrm = np.linalg.norm(A.reshape(m, -1), axis=1)
        coh_h = (np.linalg.norm(A.mean(0), axis=1)
                 / np.maximum(np.linalg.norm(A, axis=2).mean(0), 1e-12))
        F = A.reshape(m, -1)
        Fz = F - F.mean(1, keepdims=True)
        Fz = Fz / (np.linalg.norm(Fz, axis=1, keepdims=True) + 1e-12)
        C = Fz @ Fz.T
        iu = np.triu_indices(m, 1)
        return dict(n=int(m), disp_mean=float(A.std(0).mean()),
                    disp_by_step=A.std(0).mean(1).tolist(),
                    coherence=float(np.linalg.norm(mean_seq) / max(nrm.mean(), 1e-12)),
                    coherence_by_step=coh_h.tolist(),
                    pairwise_corr_mean=float(C[iu].mean()),
                    pairwise_corr_p10=float(np.percentile(C[iu], 10)),
                    pairwise_corr_p90=float(np.percentile(C[iu], 90)),
                    seq_norm_mean=float(nrm.mean()))

    def cell_geometry(self, E, cands, S0, labels, goal, ledger, frac=0.34, who="instrument"):
        """Does the conditioning variable SEPARATE the modes that averaging destroys?

        Round 1 left this open. Its pool coherence was 0.582 over the uniform candidate draw and
        0.701 over the successful subset, and averaging tracked that exactly: the `mean` arm
        (uniform draw) lost 2.4x in audition while round-0's G2 (successful subset) had averaging
        come out ahead. The live hypothesis is that averaging destroys only when the pool mixes
        modes the conditioning variable does not separate -- which predicts that WITHIN a library
        cell the good candidates are more coherent than they are globally, and that the
        within-cell mean survives where the global mean does not.

        Both are measured here, on the same audition matrix, for free: coherence of each cell's
        top tercile against the global top tercile, and a direct replay of each cell's mean
        sequence on that cell's own states against the cell's best single candidate. Oracle-only
        (the agent never reads it), so it is charged to the instrument and labelled as such.
        """
        out = {"frac": frac, "cells": []}
        g_top = np.argsort(E.mean(1))[:max(3, int(len(E) * frac))]
        out["global"] = self.pool_geometry(cands[g_top])
        out["global_mean_replay"] = float(self.replay(S0, cands[g_top].mean(0), goal).mean())
        out["global_best"] = float(E[g_top].mean(1).min())
        n_rep = len(S0)
        for c in sorted(set(int(v) for v in labels)):
            m = labels == c
            if m.sum() < 2:
                continue
            sc = E[:, m].mean(1)
            top = np.argsort(sc)[:max(3, int(len(sc) * frac))]
            mean_seq = cands[top].mean(0)
            e_mean = self.replay(S0[m], mean_seq, goal)
            n_rep += int(m.sum())
            out["cells"].append(dict(
                cell=int(c), n_states=int(m.sum()),
                geometry=self.pool_geometry(cands[top]),
                mean_replay=float(e_mean.mean()),
                best_in_cell=float(sc[top].min()),
                mean_over_best=float(e_mean.mean() / max(sc[top].min(), 1e-9))))
        ledger.charge(who, n_rep * self.H_drill, n_rep, self.dt_ctrl, self.d_fb,
                      kind="cell_geometry")
        return out

    def fit_bc(self, X, Y, seed, steps=None, hidden=None):
        """The gradient-averaging control: regress the whole command sequence on the hand-over
        state. State-CONDITIONED like the library, and AVERAGING like the mean -- which is what
        makes it the discriminating control rather than a strawman."""
        import torch
        import torch.nn as nn

        cfg = self.cfg
        pol = self.mlp(seed, din=self.SD, dout=self.H_drill * self.AD,
                       hidden=hidden or cfg["pol_hidden"], layers=cfg["pol_layers"],
                       out_act=nn.Tanh())
        pn = {"mu": torch.tensor(X.mean(0).astype(np.float32), device=self.device),
              "sd": torch.tensor((X.std(0) + 1e-6).astype(np.float32), device=self.device)}
        Xt = (torch.tensor(X.astype(np.float32), device=self.device) - pn["mu"]) / pn["sd"]
        Yt = torch.tensor(Y.reshape(len(Y), -1).astype(np.float32), device=self.device)
        opt = torch.optim.Adam(pol.parameters(), lr=cfg["pol_lr"])
        lossf = nn.MSELoss()
        brng = np.random.default_rng(seed + 1)
        bs = min(256, len(X))
        pol.train()
        for _ in range(steps or cfg["pol_steps"]):
            idx = torch.tensor(brng.integers(0, len(X), size=bs), device=self.device)
            opt.zero_grad(); lossf(pol(Xt[idx]), Yt[idx]).backward(); opt.step()
        pol.eval()
        return {"kind": "bc", "pol": pol, "pnorm": pn}

    @staticmethod
    def mean_unit(A):
        """Position-wise mean of a set of command sequences. The etude measured this at 2.1x worse
        than any single contributor on the pusher; crystallize at 3.6-5.0x with the mechanism."""
        return {"kind": "fixed", "fixed": np.asarray(A, np.float32).mean(0)}
