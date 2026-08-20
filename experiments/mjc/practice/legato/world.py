"""The legato world: a phrase of drilled-class segments, executed across seams under one commitment.

Program: `ideas/practice_manufactures_its_own_credit.md`. Direct parents:
  * `mjc/practice/fingering/` -- round 1, this node's substrate. It settled the op taxonomy on ONE
    segment (H=20, inside the plant's composition horizon): **live content under committed routing
    (`plan_launch`) dominates every frozen op** 1.6x on error at 2.3x less priced time, frozen
    content rots by diet-narrowing, and `never` (reactive MPC) wins outright on a single easy
    segment at every price. Boundaries carry information (hand-over R^2 = 0.82 on the realised error
    of a committed unit), so state-conditioned commitment is meaningful here in a way it was not on
    the pusher.
  * `mjc/practice/etude/` -- E-4/E-5's sequential phrase assembly (commit in piece order; score sets
    from the current performance configuration) and its **fusion-vacuity** finding, which this node
    overturns. The etude's committed units were STATE-INDEPENDENT command sequences, so two adjacent
    units concatenated to a bit-identical trajectory and removing the internal re-grounding was a
    no-op on the physics. Once the units are state-CONDITIONED -- keyed at each seam by the observed
    arrival -- fusion stops being a bookkeeping identity: the fused unit must choose its whole chain
    at the phrase launch and then fly the seams blind.
  * `mjc/arm_substrate/` P3 (planner sizing: an undersized CEM makes ballistic error
    planning-noise-dominated and can SIGN-INVERT the FM-quality axis) and P4 (composition horizon
    20-23 steps at n=3).
  * `mjc/on_policy/COLLECTION_REALISM.md` -- tier C, so collection is `Body`-metered and on-policy
    and the monitors are charged.

THE NAMED UNKNOWN. Round 1's champion was live content: one CEM plan from the current forward model
at the observed launch state, flown open-loop. That works because H=20 sits at the plant's
composition horizon. A PHRASE does not: three segments is 60 steps, 3x past where a one-step forward
model's rolled-out tip trajectory is trustworthy. But a chain of measured, *executed* renditions has
no composition bound at all -- it was produced by the body, not by the model, so it is limited by
execution variance rather than by model composition. The hypothesis is that frozen, measured content
re-enters exactly where the committed span exceeds the model's composition horizon.

THE PIECE -- a closed 4-leg loop, so the piece has no privileged end and every seam is a real turn.

    q0 --A(approach, H_app, mastered, frozen planner)--> W1
       --B(drill 0, H_seg)--> W2  --C(drill 1, H_seg, THE PATCH)--> W3  --D(drill 2, H_seg)--> W0

  * `W0 = fk(q_center)` is the start tip, so leg D closes the loop back onto the start.
  * A closed quadrilateral's exterior angles sum to 360 deg, so its turns average 90 deg exactly --
    round 1's "cos in [0.10, 0.75]" window is structurally unavailable to a loop and the admissible
    band here is [-0.25, 0.45]. Every seam is therefore a genuine direction change, which is what
    makes crossing one blind a real bet rather than a formality.
  * The PATCH sits on the midpoint of leg C, the MIDDLE of the phrase: the boundary a phrase
    commitment gives up (W2, immediately before the hard passage) is exactly the one that carries
    the most information. That is the sharpest form of the bet.
  * The approach is unchanged in law from round 1: one full-horizon plan from the actual start
    posture over a FROZEN forward model, rolled out open-loop at tempo, identical across arms -- so
    the phrase-launch distribution is a property of the WORLD, not of the arm being graded.

WHAT IS NEW HERE, MECHANICALLY, AGAINST `fingering/world.py`
  1. `plan_fn` takes a PER-STEP goal schedule, so one CEM plan can span several segments and still
     play the piece's shape instead of cutting the corner to the last waypoint. With a constant goal
     the arithmetic is identical to round 1's.
  2. `traverse` executes a ROUTING -- a partition of the drilled segments into contiguous committed
     groups. `never` is 3 reactive groups, `seg_*` is 3 one-segment groups, `phrase_*` is one
     3-segment group. Feedback events are charged per group launch, so the routing IS the price.
  3. Reactive control is receding-horizon over the piece (fixed lookahead, per-step goals), which
     reduces to round 1's reactive unit on a single segment.
  4. The `Ledger` additionally counts DELIBERATION WORK (`k_shoot * cem_iters * horizon` CEM
     rollout-steps), not just plan events. A phrase plan over 180 action dimensions is not one
     segment plan; pricing them the same would hand the phrase arm a subsidy it did not earn.

`import mujoco` / `import torch` live inside functions so this module is importable from a laptop
that has neither -- the `arm_env.py` / `pusher_env.py` contract.
"""

import numpy as np

# Pure-numpy geometry helpers, byte-identical in behaviour to round 1 and therefore IMPORTED rather
# than copied: the redundancy machinery (`null_direction`), the start-posture draw (`start_postures`,
# whose `iso`/`null` modes and RNG consumption must match so a legato world and a fingering world at
# the same seed draw the same geometries) and `kmeans` (the library partition).
from mjc.practice.fingering.world import (  # noqa: F401
    planar_jacobian, null_direction, start_postures, kmeans)


class Ledger:
    """Every environment step, feedback event, plan and priced second, split by who it was for.

    A fork of round 1's `Ledger` with one added counter -- `delib`, the CEM rollout-steps a plan
    actually costs (`k_shoot * cem_iters * horizon`). Round 1 counted PLAN EVENTS and never charged
    them, which was enough when every plan had the same shape. Here it is not: a phrase plan
    optimises a 180-dimensional action sequence and, per `arm_substrate` P3, has to be given a
    bigger CEM budget than a 60-dimensional segment plan or its degradation would be planner
    starvation rather than model composition. Counting both means the deliberation axis can be
    priced either per EVENT (what a nervous system pays to interrupt) or per WORK (what it pays to
    think), post hoc, without re-running anything.

    `agent` = anything that informs a decision the agent makes (practice, metering the agent reads,
    auditions, score-set traversals). `instrument` = experimenter-side readouts the agent never sees
    (the held-out ladder, the ceiling-FM reference, ground-truth region membership).
    """

    def __init__(self):
        self.steps = {"agent": 0, "instrument": 0}
        self.fb = {"agent": 0, "instrument": 0}
        self.plans = {"agent": 0, "instrument": 0}
        self.delib = {"agent": 0, "instrument": 0}
        self.t_priced = 0.0            # agent-side priced time -- the headline currency
        self.t_by_kind = {}
        self.n_by_kind = {}            # RAW counts per kind, so the price vector stays post-hoc

    def charge(self, who, steps, fb, dt_ctrl, d_fb, kind="other", plans=0, delib=0):
        self.steps[who] += int(steps)
        self.fb[who] += int(fb)
        self.plans[who] += int(plans)
        self.delib[who] += int(delib)
        t = steps * dt_ctrl + fb * d_fb
        if who == "agent":
            self.t_priced += t
            self.t_by_kind[kind] = self.t_by_kind.get(kind, 0.0) + t
            n = self.n_by_kind.setdefault(
                kind, {"steps": 0, "fb": 0, "plans": 0, "delib": 0})
            n["steps"] += int(steps)
            n["fb"] += int(fb)
            n["plans"] += int(plans)
            n["delib"] += int(delib)
        return t

    def snapshot(self):
        return {"steps_agent": self.steps["agent"], "steps_instrument": self.steps["instrument"],
                "fb_agent": self.fb["agent"], "fb_instrument": self.fb["instrument"],
                "plans_agent": self.plans["agent"], "plans_instrument": self.plans["instrument"],
                "delib_agent": self.delib["agent"], "delib_instrument": self.delib["instrument"],
                "t_priced": self.t_priced, "t_by_kind": dict(self.t_by_kind),
                "n_by_kind": {k: dict(v) for k, v in self.n_by_kind.items()}}


def elite_for(k_shoot, floor=16, frac=32):
    """CEM elite count for a shooting width. Round 1 ran `k_shoot=1024, cem_elite=32`; keeping the
    elite FRACTION fixed at 1/32 reproduces that exactly and stops a 4x wider search from becoming a
    4x greedier one, which is the failure mode that would make a bigger planner look worse."""
    return int(max(floor, k_shoot // frac))


class World:
    """The plant, the loop, the forward models and the controllers, in one object.

    A copy-fork of `fingering/world.py`'s `World`, not a subclass: the piece, the traversal, the
    controller's goal handling and the compile ops all change shape here, and round 1 must stay
    byte-reproducible. The parts that do NOT change (`planar_jacobian`, `null_direction`,
    `start_postures`, `kmeans`) are imported from it above.
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
        self._fk = fk

        # ---- the piece: goals[0] is the approach's goal (W1); goals[1:] are the drilled segments'
        # goals in piece order. `W0 = fk(q_center)` closes the loop and is the last drilled goal.
        self.goals = np.asarray(cfg["waypoints"], dtype=np.float32)     # (1 + n_seg, 2)
        self.n_seg = len(self.goals) - 1
        self.H_app = int(cfg["h_app"])
        self.H_seg = [int(h) for h in cfg["h_seg"]]
        assert len(self.H_seg) == self.n_seg, "h_seg must give one horizon per drilled segment"
        self.H_phrase = int(sum(self.H_seg))
        self.seg_lo = np.cumsum([0] + self.H_seg[:-1]).astype(int)      # phrase-local start index
        self.seg_hi = np.cumsum(self.H_seg).astype(int)                 # phrase-local end index
        # round-1 compatibility aliases (a single-segment legato world IS the fingering world)
        self.W1 = self.goals[0]
        self.H_drill = self.H_seg[0]

        # ---- the hard region: a Gaussian-gated curl patch on the midpoint of ONE drilled segment.
        self.patch_seg = int(cfg["patch_seg"])
        a = self.goals[self.patch_seg]              # start waypoint of drilled segment patch_seg
        b = self.goals[self.patch_seg + 1]          # its goal
        self.patch_center = np.asarray(
            cfg.get("patch_center") or (0.5 * (a + b)), dtype=np.float64)
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

    # ------------------------------------------------------------ the piece, as schedules

    def span(self, seg_lo, n_segs):
        """Number of control steps in the contiguous drilled span [seg_lo, seg_lo + n_segs)."""
        return int(sum(self.H_seg[seg_lo:seg_lo + n_segs]))

    def goal_schedule(self, seg_lo, n_segs):
        """(H, 2) per-step goal for a committed span: every step is scored against ITS OWN segment's
        waypoint. Without this a single CEM plan over three segments would optimise distance to the
        last waypoint at every step, cut both corners, and never play the piece."""
        rows = []
        for k in range(seg_lo, seg_lo + n_segs):
            rows.append(np.tile(self.goals[k + 1][None, :], (self.H_seg[k], 1)))
        return np.concatenate(rows, 0).astype(np.float32)

    def waypoint_mask(self, g0, hh):
        """Boolean mask over a planning window [g0, g0+hh) of phrase-local steps: True where a
        WAYPOINT IS DUE (the last step of a drilled segment).

        THE REPAIR (l1). The grader measures arrival at each waypoint at its seam step, but the
        running cost only measures mean distance over the window -- so a window that spans a seam is
        rewarded for leaving waypoint k early to get a head start on k+1. Measured in l0: on the
        same launch states with the same model, segment 0 was reached at 0.032 by a unit whose
        window ENDS at the waypoint and at 0.194 by the reactive controller whose 20-step lookahead
        crosses the seam. Six times worse, from lookahead alone -- and it crippled `never`, the
        reference arm, while flattening the FM-quality axis to 3% of the error.

        Weighting the seam steps makes the controller optimise what the task measures. It is applied
        through `plan_fn` to EVERY planner in the node -- reactive, segment-committed, phrase-
        committed and the approach alike -- so no arm gets a private objective. With
        `w_waypoint = 0` every cost reduces exactly to l0's.
        """
        idx = np.arange(int(g0), int(g0) + int(hh))
        return np.isin(idx, np.asarray(self.seg_hi) - 1)

    def approach_mask(self):
        """The approach is graded at W1, so its final step is a waypoint-due step too -- same rule,
        so the mastered lead-in is not quietly running a different objective from the phrase."""
        m = np.zeros(self.H_app, bool)
        m[-1] = True
        return m

    def checkpoints(self, seg_lo, n_segs):
        """[(step_index_within_span, goal)] -- the waypoints a span is graded at."""
        out, h = [], 0
        for k in range(seg_lo, seg_lo + n_segs):
            h += self.H_seg[k]
            out.append((h - 1, self.goals[k + 1].astype(np.float64)))
        return out

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
        from mjc.embodied import Body

        if n_par not in self._bodies:
            self._bodies[n_par] = Body(self.env, n_par=n_par, frame_skip=self.fs,
                                       wrap_limit=self.cfg.get("wrap_limit", 3.0))
        return self._bodies[n_par]

    def tip(self, states):
        s = np.atleast_2d(np.asarray(states, dtype=np.float64))
        return self._fk(s[:, : self.n], self.Ls)

    def gate(self, tips):
        """Region membership weight in [0, 1]. Oracle-only (the agent never reads it)."""
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
        pool. No per-sample delta gain anywhere in this node."""
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

    def fm_rollout_tips(self, net, S0, cmds):
        """Open-loop FM rollout: predicted tip after each of `cmds`' steps, starting at `S0`.

        The pure MODEL quantity behind the composition-horizon question -- no CEM anywhere, so its
        divergence cannot be confused with planner noise. `cmds` is (B, H, AD); returns (B, H, 2).
        """
        from mjc.arm_env import fk

        s = np.asarray(S0, np.float32).copy()
        H = cmds.shape[1]
        out = np.empty((len(s), H, 2))
        for h in range(H):
            s = s + self.fm_delta(net, s, cmds[:, h, :])
            out[:, h] = fk(s[:, : self.n].astype(np.float64), self.Ls)
        return out

    # ------------------------------------------------------------ the diet (on-policy, metered)

    def collect_ou(self, n, rng, ep_len=None, n_par=None, q_range=None):
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
        from mjc.embodied import collect_on_policy, ReachBehaviour

        cfg = self.cfg
        n_par = int(n_par or cfg["n_par"])
        ep_len = int(ep_len or cfg["ep_len"])
        pf = self.plan_fn(fm, ep_len, k_shoot=cfg["collect_k_shoot"],
                          cem_iters=cfg["collect_cem_iters"])
        beh = ReachBehaviour(lambda s, g: pf(s, g, rng)[0], np.zeros((n_par, 2), np.float32), rng,
                             sigma_u=cfg["sigma_u"], replan_every=ep_len)
        return collect_on_policy(self.env, int(n), rng, self.fs, self.qc,
                                 float(q_range if q_range is not None else cfg["op_q_range"]),
                                 behaviour=beh, ep_len=ep_len, n_par=n_par,
                                 goal_sampler=goal_sampler,
                                 wrap_limit=cfg.get("wrap_limit", 3.0), return_info=True)

    def exclude_region(self, S, U, S2, w=0.1):
        """The etude's `pretrain_mode="exclude"`: keep only transitions whose tip is OUTSIDE the
        hard region, so the passage is *unmodelled* rather than *wrongly modelled*."""
        keep = self.gate(self.tip(S)) <= w
        return S[keep], U[keep], S2[keep], float(keep.mean())

    # ------------------------------------------------------------ controller

    def fk_torch(self, q):
        import torch

        ang = torch.cumsum(q, dim=1)
        return torch.stack([(self.Lt * torch.cos(ang)).sum(1),
                            (self.Lt * torch.sin(ang)).sum(1)], 1)

    def plan_fn(self, net, hh, k_shoot=None, cem_iters=None, cem_elite=None, max_elems=None,
                vel_pen=None, wp_mask=None, step_w=None):
        """CEM-MPC over the FM with a PER-STEP goal schedule. Sized per `arm_substrate` P3.

        `plan(states, goals, rng)` accepts `goals` as (B, 2) -- a constant goal, arithmetically
        identical to round 1 -- or (B, hh, 2), a schedule. It returns `(mu, delib)`: the elite-mean
        action sequence and the CEM rollout-steps it cost (`k_shoot * cem_iters * hh`), because a
        180-dimensional phrase plan is not the same act of deliberation as a 60-dimensional segment
        plan and the ledger has to be able to tell them apart.

        `vel_pen` penalises the FINAL state's joint velocity. Round 1 applied it unconditionally,
        which was right there: its drilled segment ENDED the piece, so braking at the last waypoint
        is what the task wanted. On a phrase it is actively wrong -- a plan that brakes to a stop at
        every seam plays the piece staccato, and it would hand a segment-wise arm an artificial
        handicap that has nothing to do with commitment. So the penalty is a per-call argument, and
        callers pass the terminal `vel_pen` only where the planning window actually reaches the
        piece's last waypoint (`vel_pen_mid`, calibrated by CAL-C, everywhere else).

        The batch is CHUNKED so a wide search over a long horizon cannot OOM: the candidate tensor
        is `B * K * hh * AD` floats, which at K=4096, hh=60, B=48 is 141 MB per copy.
        """
        import torch

        cfg = self.cfg
        K = int(k_shoot or cfg["k_shoot"])
        I = int(cem_iters or cfg["cem_iters"])
        ne = int(cem_elite if cem_elite is not None else cfg["cem_elite"])
        vp = float(cfg["vel_pen"] if vel_pen is None else vel_pen)
        s0sig = cfg["cem_init_sigma"]
        me = int(max_elems if max_elems is not None else cfg.get("plan_max_elems", 40_000_000))
        n, AD = self.n, self.AD
        hh = int(hh)
        # per-step cost weight: 1 everywhere, plus `w_waypoint` where a waypoint is due. THE SAME
        # rule for every planner in the node -- see `waypoint_mask`. At w_waypoint=0 this is a
        # vector of ones and the arithmetic is bit-identical to l0. `step_w` overrides it outright,
        # which is how a look-ahead portion gets discounted by gamma (see `unit_commands`).
        if step_w is not None:
            w_np = np.asarray(step_w, np.float32)[:hh]
        else:
            w_np = np.ones(hh, np.float32)
            if wp_mask is not None and float(cfg.get("w_waypoint", 0.0)) != 0.0:
                w_np = w_np + float(cfg["w_waypoint"]) * np.asarray(wp_mask, np.float32)[:hh]
        w_t = torch.tensor(w_np, device=self.device)

        def plan_chunk(states, goals, rng):
            Bn = states.shape[0]
            mu = np.zeros((Bn, hh, AD), np.float32)
            sig = np.full((Bn, hh, AD), s0sig, np.float32)
            g_t = torch.tensor(goals, device=self.device).repeat_interleave(K, 0)  # (Bn*K, hh, 2)
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
                        cost = cost + w_t[h] * (self.fk_torch(s[:, :n]) - g_t[:, h, :]).norm(dim=1)
                    cost = cost + vp * s[:, n:].norm(dim=1)
                    idx = torch.topk(-cost.reshape(Bn, K), min(ne, K), dim=1).indices.cpu().numpy()
                elite = np.take_along_axis(seqs, idx[:, :, None, None], axis=1)
                mu = elite.mean(1)
                sig = elite.std(1) + 1e-3
            return mu.astype(np.float32)

        def plan(states, goals, rng):
            states = np.asarray(states, np.float32)
            goals = np.asarray(goals, np.float32)
            if goals.ndim == 2:
                goals = np.repeat(goals[:, None, :], hh, axis=1)
            Bn = states.shape[0]
            cb = max(1, min(Bn, me // max(1, K * hh * AD)))
            outs = [plan_chunk(states[i:i + cb], goals[i:i + cb], rng)
                    for i in range(0, Bn, cb)]
            return (np.concatenate(outs, 0) if len(outs) > 1 else outs[0]), Bn * K * I * hh

        return plan

    # ------------------------------------------------------------ the units and the routing

    @staticmethod
    def reactive_unit(**kw):
        return {"kind": "reactive", **kw}

    @staticmethod
    def plan_launch_unit(**kw):
        """Live content under committed routing: one plan over the group's whole span, from the
        observed launch state, under the CURRENT forward model. `k_shoot`/`cem_iters`/`cem_elite`
        may be overridden per group, because CAL-P sizes the planner PER ACTION DIMENSION and a
        phrase span has three times a segment's."""
        return {"kind": "plan_launch", **kw}

    def routing(self, kind, groups=None, units=None, **kw):
        """Build a routing: a partition of the drilled segments into contiguous committed groups.

        `groups` is a list of group sizes summing to `n_seg` (default: one group per segment).
        `kind` names the unit for every group when `units` is not given.
        """
        groups = list(groups or [1] * self.n_seg)
        assert sum(groups) == self.n_seg, f"groups {groups} must cover {self.n_seg} segments"
        out, lo = [], 0
        for i, g in enumerate(groups):
            u = (units[i] if units is not None else {"kind": kind, **kw})
            out.append({"seg_lo": lo, "n_segs": int(g), "unit": u})
            lo += g
        return out

    def vel_pen_for(self, seg_lo, n_segs):
        """The terminal velocity penalty for a planning window covering [seg_lo, seg_lo+n_segs):
        the real one only where the window reaches the piece's last waypoint, `vel_pen_mid`
        otherwise. See `plan_fn`."""
        return (self.cfg["vel_pen"] if seg_lo + n_segs >= self.n_seg
                else self.cfg.get("vel_pen_mid", 0.0))

    def unit_commands(self, unit, launch, fm, rng, seg_lo, n_segs):
        """The command sequence a committed unit issues for its group, given the observed launch
        state. Returns `(cmds (B, span, AD), n_plans, delib)`.

        LOOKAHEAD (live content only). A `plan_launch` unit COMMITS to its group's span but may
        PLAN past it, issuing only the first `span` actions. Without this, segment-wise live
        planning is myopic at exactly the seams the round is about -- it would aim each segment at
        its own waypoint with no regard for the state it hands downstream, and lose to a phrase plan
        for a reason that has nothing to do with composition horizons. The extra deliberation is
        charged honestly on the `delib` axis, which is where the strong version of the segment arm
        should pay for being strong.
        """
        kind = unit["kind"]
        B = len(launch)
        H = self.span(seg_lo, n_segs)
        if kind == "fixed":
            return np.tile(unit["fixed"][None], (B, 1, 1)).astype(np.float32), 0, 0
        if kind == "lib":
            # THE FEEDBACK EVENT IS ACTUALLY SPENT: the charged observation at the group's launch
            # selects which committed rendition flies. For a PHRASE group this is the whole bet --
            # one key lookup buys the entire chain, and the seams inside it are crossed blind.
            Z = (launch - unit["key_mu"]) / unit["key_sd"]
            D = ((Z[:, None, :] - unit["keys"][None, :, :]) ** 2).sum(-1)
            return unit["seqs"][D.argmin(1)].astype(np.float32), 0, 0
        if kind == "lib_proj":
            Z = np.concatenate([(launch - unit["key_mu"]) / unit["key_sd"],
                                np.ones((B, 1), np.float32)], 1)
            j = np.searchsorted(unit["edges"], Z @ unit["beta"])
            return unit["seqs"][np.clip(j, 0, len(unit["seqs"]) - 1)].astype(np.float32), 0, 0
        if kind == "plan_launch":
            # GAMMA-SHAPED LOOKAHEAD (l2). A live unit COMMITS to its span but may PLAN one segment
            # further, issuing only the committed part, with the look-ahead portion's cost
            # discounted by `gamma`. gamma=0 is no lookahead at all; gamma=1 weights the next
            # segment as heavily as this one, which l1 measured as WORSE because it drags accuracy
            # off the committed waypoint. The interesting regime is in between: enough weight on
            # what you hand over that the seam is shaped, not so much that you miss your own note.
            gm = float(unit.get("gamma", self.cfg.get("lookahead_gamma", 0.0)))
            ns_p = n_segs + 1 if (gm > 0.0 and seg_lo + n_segs < self.n_seg) else n_segs
            Hp = self.span(seg_lo, ns_p)
            g0 = int(self.seg_lo[seg_lo])
            w = 1.0 + float(self.cfg.get("w_waypoint", 0.0)) * self.waypoint_mask(g0, Hp)
            if ns_p > n_segs:
                w[H:] = w[H:] * gm                   # discount everything past the committed span
            pf = self.plan_fn(fm, Hp, k_shoot=unit.get("k_shoot"), cem_iters=unit.get("cem_iters"),
                              cem_elite=unit.get("cem_elite"),
                              vel_pen=self.vel_pen_for(seg_lo, ns_p), step_w=w)
            gs = np.tile(self.goal_schedule(seg_lo, ns_p)[None], (B, 1, 1))
            cmds, delib = pf(launch, gs, rng)
            return cmds[:, :H, :], 1, delib          # plan past the seam, commit only to the span
        if kind == "bc":
            import torch

            with torch.no_grad():
                x = (torch.tensor(launch.astype(np.float32), device=self.device)
                     - unit["pnorm"]["mu"]) / unit["pnorm"]["sd"]
                y = unit["pol"](x).cpu().numpy()
            return y.reshape(B, H, self.AD).astype(np.float32), 0, 0
        raise ValueError(f"unit kind {kind!r} has no launch-time plan")

    # ------------------------------------------------------------ traversal

    def traverse(self, fm, routing, q0, rng, sigma, ledger, who="agent", kind="other",
                 approach_plan=None, collect=False, qd0=None, stop_seg=None, replan_every=1,
                 look=None):
        """Execute the piece once for `len(q0)` performers, through the metered `Body`.

        Boundaries are re-grounding points, not teleports: every segment starts from the state the
        previous one actually achieved. Motor noise is applied to the issued command in BOTH
        practice and performance.

        `routing` partitions the drilled segments into contiguous committed groups (see `routing()`);
        `stop_seg` truncates execution after that many drilled segments, which is how launch-state
        distributions are harvested (`stop_seg=0` runs the approach only and returns the
        phrase-launch states, charged for the approach it actually executed).

        Feedback events: **1 for the approach launch, then 1 per committed group launch, or one per
        replan inside a reactive group.** That is the whole price of routing, and it is why fusing
        three segments into a phrase is worth exactly two feedback events per traversal.
        """
        q0 = np.asarray(q0, dtype=np.float64)
        B = len(q0)
        stop = self.n_seg if stop_seg is None else int(stop_seg)
        look = int(look or self.cfg.get("react_look", max(self.H_seg)))
        body = self.body(B)
        try:
            s = body.reset(rng, self.qc, 0.0, q0=q0, qd0=qd0)
            trS, trU, trS2 = [], [], []

            # ---- leg A: the approach. One plan, rolled open-loop at tempo. ----
            g1 = np.tile(self.goals[0][None, :], (B, 1))
            if approach_plan is not None:
                plan = approach_plan
            else:
                plan, _ = self.plan_fn(fm, self.H_app, vel_pen=self.cfg.get("vel_pen_mid", 0.0),
                                       wp_mask=self.approach_mask())(s, g1, rng)
            for h in range(self.H_app):
                a = np.clip(plan[:, h, :], -1, 1)
                if sigma > 0:
                    a = np.clip(a + sigma * rng.standard_normal(a.shape), -1, 1)
                sa, ua, sb, _ = body.step(a.astype(np.float32))
                if collect:
                    trS.append(sa); trU.append(ua); trS2.append(sb)
                s = sb
            launch0 = s.copy()                        # the PHRASE-LAUNCH state, at W1
            e_app = np.linalg.norm(self.tip(launch0) - self.goals[0][None, :], axis=1)
            fb = 1
            n_plan = 0
            delib = 0

            # ---- legs B..D: the drilled segments, under the routing ----
            acts = np.full((B, self.H_phrase, self.AD), np.nan, np.float32)
            acts_raw = np.full_like(acts, np.nan)
            # `launches[k]` is the state segment k STARTS from and `e_seg[:, k]` its arrival error.
            # Both are recorded for every segment, including segments interior to a fused group --
            # they are experimenter readouts of the trajectory, not feedback events, and the arm is
            # charged for exactly the groups it launched. That distinction is the whole node: a
            # fused arm's seam states exist, it just never gets to look at them.
            launches = {0: launch0}
            e_seg = np.full((B, self.n_seg), np.nan)
            end_of = {int(self.seg_hi[k]) - 1: k for k in range(self.n_seg)}
            gate_hits = 0
            n_exec = 0
            for grp in routing:
                lo, unit = int(grp["seg_lo"]), grp["unit"]
                if lo >= stop:
                    break
                ns = min(int(grp["n_segs"]), stop - lo)   # a truncated run may cut a group short
                H = self.span(lo, ns)
                base = int(self.seg_lo[lo])
                launches[lo] = s.copy()
                if unit["kind"] == "reactive":
                    re = self.H_phrase if replan_every <= 0 else int(replan_every)
                    # a reactive group looks ahead past its own segments too, out to the piece's
                    # end -- receding-horizon MPC over the PIECE, not over the segment.
                    sched = self.goal_schedule(lo, self.n_seg - lo)
                    dplan = None
                    for h in range(H):
                        if h % re == 0:
                            hh = min(look, self.H_phrase - (base + h))   # clipped at the piece end
                            vp = (self.cfg["vel_pen"] if base + h + hh >= self.H_phrase
                                  else self.cfg.get("vel_pen_mid", 0.0))
                            dplan, dl = self.plan_fn(
                                fm, hh, vel_pen=vp,
                                wp_mask=self.waypoint_mask(base + h, hh))(
                                s, np.tile(sched[None, h:h + hh], (B, 1, 1)), rng)
                            fb += 1; n_plan += 1; delib += dl
                        a0 = np.clip(dplan[:, h % re, :], -1, 1)
                        s, gh = self._apply(body, a0, s, sigma, rng, collect, trS, trU, trS2,
                                            acts, acts_raw, base + h)
                        gate_hits += gh
                        k = end_of.get(base + h)
                        if k is not None:
                            e_seg[:, k] = np.linalg.norm(
                                self.tip(s) - self.goals[k + 1][None, :], axis=1)
                            if k + 1 < self.n_seg:
                                launches[k + 1] = s.copy()
                else:
                    cmds, npl, dl = self.unit_commands(unit, s.copy(), fm, rng, lo, ns)
                    fb += 1; n_plan += npl; delib += dl
                    for h in range(H):
                        a0 = np.clip(cmds[:, h, :], -1, 1)
                        s, gh = self._apply(body, a0, s, sigma, rng, collect, trS, trU, trS2,
                                            acts, acts_raw, base + h)
                        gate_hits += gh
                        k = end_of.get(base + h)
                        if k is not None:
                            e_seg[:, k] = np.linalg.norm(
                                self.tip(s) - self.goals[k + 1][None, :], axis=1)
                            if k + 1 < self.n_seg:
                                launches[k + 1] = s.copy()
                n_exec += H
        finally:
            body.release()

        steps = B * (self.H_app + n_exec)
        t_piece = ledger.charge(who, steps, B * fb, self.dt_ctrl, self.d_fb, kind=kind,
                                plans=B * n_plan, delib=delib)
        e_piece = (np.nanmean(e_seg[:, :stop], axis=1) if stop > 0
                   else np.full(B, np.nan))
        out = dict(launch=launch0, launches=launches, final=s.copy(), e_app=e_app, e_seg=e_seg,
                   e_piece=e_piece, acts=acts, acts_raw=acts_raw, n_fb=fb, n_plan=n_plan,
                   delib=delib, t_piece=t_piece / max(B, 1), t_total=t_piece,
                   gate_frac=gate_hits / float(B * max(n_exec, 1)))
        if collect:
            out["trans"] = (np.concatenate(trS, 0), np.concatenate(trU, 0),
                            np.concatenate(trS2, 0))
        return out

    def _apply(self, body, a0, s, sigma, rng, collect, trS, trU, trS2, acts, acts_raw, idx):
        """One control step: record the raw command, add motor noise, step the metered body."""
        acts_raw[:, idx, :] = a0
        a = a0 if sigma <= 0 else np.clip(a0 + sigma * rng.standard_normal(a0.shape), -1, 1)
        acts[:, idx, :] = a
        gh = int((self.gate(self.tip(s)) > 0.3).sum())
        sa, ua, sb, _ = body.step(a.astype(np.float32))
        if collect:
            trS.append(sa); trU.append(ua); trS2.append(sb)
        return sb, gh

    # ------------------------------------------------------------ the compile op

    def replay(self, S0, cmds, checkpoints, return_states=False, sigma=0.0, rng=None):
        """Execute one command sequence open-loop from each of `S0`; return per-checkpoint errors.

        The one place the agent re-enters a state it has already been in. It is a *mental replay of
        a remembered arrival*, not an omniscient workspace sample: every state in `S0` was produced
        by genuine at-tempo upstream execution and recorded. Fully priced by `audition`.

        `checkpoints` is [(step_index, goal)] -- for a phrase candidate that is one entry per seam
        plus the end, so a chain is graded on whether it plays the PIECE and not merely on where it
        happens to stop. `return_states` additionally hands back the state AT each checkpoint, which
        is how G6 measures what a fused unit gives up: the seam states exist under a phrase
        commitment, the arm just never gets to read them.

        `sigma > 0` applies MOTOR NOISE to the replayed commands. The audition runs noiseless (round
        1's convention, and the right grading instrument -- a mental replay is not a performance),
        but a noiseless replay makes every downstream state a deterministic function of the launch
        state, which would let G6's information measure read ~1.0 by construction. Execution
        variance is precisely what limits a chain of measured renditions, so the gate measures both
        and reports the gap.
        """
        errs = np.empty((len(S0), len(checkpoints)), np.float32)
        n = self.n
        H = cmds.shape[0]
        cps = {int(h): g for h, g in checkpoints}
        st = np.empty((len(S0), len(checkpoints), self.SD), np.float32) if return_states else None
        for i in range(len(S0)):
            self.env.set_state(S0[i, :n].astype(np.float64), S0[i, n:].astype(np.float64))
            c = 0
            for h in range(H):
                u = cmds[h]
                if sigma > 0:
                    u = np.clip(u + sigma * rng.standard_normal(u.shape), -1, 1)
                self.env.step(u, self.fs)
                if h in cps:
                    errs[i, c] = np.linalg.norm(self.env.tip_pos() - cps[h])
                    if return_states:
                        st[i, c] = self.env.get_state()
                    c += 1
        return (errs, st) if return_states else errs

    def true_tips(self, S0, cmds):
        """Ground-truth tip after each of `cmds`' steps, executed in the real plant from each of
        `S0`. The reference the FM rollout is scored against in the composition-horizon gate. Oracle
        instrument: free, and labelled as such."""
        n, H = self.n, cmds.shape[1]
        out = np.empty((len(S0), H, 2))
        for i in range(len(S0)):
            self.env.set_state(S0[i, :n].astype(np.float64), S0[i, n:].astype(np.float64))
            for h in range(H):
                self.env.step(cmds[i, h], self.fs)
                out[i, h] = self.env.tip_pos()
        return out

    def audition(self, cands, S0, checkpoints, ledger, who="agent"):
        """The (n_cand, n_score) score matrix: every candidate replayed from every held-out launch
        state, scored as the MEAN over the span's waypoints. Priced as `n_cand * n_score` committed
        traversals of the span.

        Candidates are drawn UNIFORMLY from the trace pool by the caller, never top-of-pool:
        ranking by a candidate's own realised error is a winner's curse (etude E-3b; crystallize
        measures the same curse at 3.3x). Note the deliberation asymmetry that makes this op
        interesting: an audition rollout replays STORED content, so it costs environment steps and
        feedback events but **zero** deliberation.
        """
        E = np.empty((len(cands), len(S0)), np.float32)
        Efull = np.empty((len(cands), len(S0), len(checkpoints)), np.float32)
        H = cands.shape[1]
        for j, c in enumerate(cands):
            Efull[j] = self.replay(S0, c, checkpoints)
            E[j] = Efull[j].mean(1)
        ledger.charge(who, len(cands) * len(S0) * H, len(cands) * len(S0),
                      self.dt_ctrl, self.d_fb, kind="audition")
        return E, Efull

    @staticmethod
    def select_fixed(E, cands):
        """E-3b's op: commit the argmin of the MEAN over the consumption distribution."""
        j = int(np.argmin(E.mean(1)))
        return {"kind": "fixed", "fixed": np.asarray(cands[j], np.float32)}, j

    def select_library(self, E, cands, S0, n_lib, rng):
        """State-conditioned commitment: one committed rendition per cell of the launch
        distribution, keyed at launch by the observed state.

        Note what this does NOT cost: the score matrix is the SAME matrix `select_fixed` reads.
        Both ops audition the same candidates on the same states at the same price; the library's
        whole advantage is information use, not budget.
        """
        mu, sd = S0.mean(0), S0.std(0) + 1e-6
        Z = (S0 - mu) / sd
        C, lab = kmeans(Z, n_lib, rng)
        k = len(C)
        seqs = np.empty((k,) + cands.shape[1:], np.float32)
        picks, sizes = [], []
        for c in range(k):
            m = lab == c
            j = int(np.argmin(E.mean(1))) if not m.any() else int(np.argmin(E[:, m].mean(1)))
            seqs[c] = cands[j]
            picks.append(j)
            sizes.append(int(m.sum()))
        jd = int(np.argmin(E.mean(1)))
        return ({"kind": "lib", "keys": C, "key_mu": mu, "key_sd": sd, "seqs": seqs,
                 "default": np.asarray(cands[jd], np.float32)},
                dict(picks=picks, cell_sizes=sizes, n_distinct=len(set(picks)), default=jd,
                     labels=lab.tolist()))

    def select_library_proj(self, E, cands, S0, n_lib):
        """The same library, keyed on a SUPERVISED 1-D projection of the launch state rather than an
        isotropic partition of it -- round 0 measured a linear function of the hand-over explaining
        82% of the realised-error variance, which isotropic k-means has no reason to align with. The
        key is fit from the audition matrix the arm has ALREADY PAID FOR."""
        mu, sd = S0.mean(0), S0.std(0) + 1e-6
        Z = np.concatenate([(S0 - mu) / sd, np.ones((len(S0), 1))], 1)
        jd = int(np.argmin(E.mean(1)))
        beta, *_ = np.linalg.lstsq(Z, E[jd], rcond=None)
        t = Z @ beta
        r2 = float(1.0 - (E[jd] - t).var() / max(E[jd].var(), 1e-12))
        edges = np.quantile(t, np.linspace(0.0, 1.0, n_lib + 1)[1:-1])
        lab = np.searchsorted(edges, t)
        seqs = np.empty((n_lib,) + cands.shape[1:], np.float32)
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
                     r2_of_key=r2, beta=beta.tolist(), edges=edges.tolist(),
                     labels=lab.tolist()))

    @staticmethod
    def mean_unit(A):
        """Position-wise mean of a set of command sequences. The etude measured 2.1x worse than any
        contributor on the pusher; crystallize 3.6-5.0x, with the mechanism."""
        return {"kind": "fixed", "fixed": np.asarray(A, np.float32).mean(0)}

    @staticmethod
    def pool_geometry(A):
        """Geometry of the candidate command pool -- logged, never acted on. `coherence` =
        ||mean sequence|| / mean ||sequence||: ~1 when the pool is phase-locked (averaging survives)
        and ~0 when it is phase-scattered (averaging destroys)."""
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
                    coherence=float(np.linalg.norm(mean_seq) / max(nrm.mean(), 1e-12)),
                    coherence_by_step=coh_h.tolist(),
                    pairwise_corr_mean=float(C[iu].mean()),
                    pairwise_corr_p10=float(np.percentile(C[iu], 10)),
                    pairwise_corr_p90=float(np.percentile(C[iu], 90)),
                    seq_norm_mean=float(nrm.mean()))

    def fit_bc(self, X, Y, seed, steps=None, hidden=None):
        """The gradient-averaging control: regress the whole command sequence on the launch state.
        State-CONDITIONED like the library and AVERAGING like the mean, which is what makes it the
        discriminating control rather than a strawman."""
        import torch
        import torch.nn as nn

        cfg = self.cfg
        H = Y.shape[1]
        pol = self.mlp(seed, din=self.SD, dout=H * self.AD,
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
