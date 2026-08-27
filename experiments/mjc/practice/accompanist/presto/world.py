"""The presto world -- a VERBATIM FORK of `offbook/world.py` (2026-08-26), byte-identical apart
from two import lines (`.piece` and `.nets` now resolve inside this package) and this paragraph.
`offbook/` is untouched and every `g0/O1/O2/O3/d0` result stays byte-reproducible. Gate **G-F**
asserts the fork still reproduces the donor traversal bit-for-bit on the DONOR piece and plant
rather than trusting that the copy was faithful. Nothing else in this file may change: what presto
changes is the PIECE and the PLANT, both of which arrive through `cfg`, so the fork stays a fork.

Original offbook header follows.
--------------------------------------------------------------------------------------------------
The offbook world -- a VERBATIM FORK of `legato/world.py` (2026-08-25) with the re-internalization
machinery appended. `legato/` is untouched and every `l0/l1/l2/L1` result stays byte-reproducible.

WHY A FORK AND NOT A SUBCLASS. The traversal changes shape: legato executes a routing decided BEFORE
the traversal (a static partition of the drilled segments into committed groups), and this node
decides at each seam, from the state the body actually reached, per performer. Everything the fork
inherits is byte-identical (`traverse`, `plan_fn`, `audition`, `select_library`, the diet, the
metering convention), which is what makes gate G-F -- offbook's `never` reproducing a donor legato
arm bit-for-bit -- an assertion about this file rather than a hope.

WHAT IS APPENDED, AND WHY (the port back, on the motor substrate):

  * `Library` -- the address book. A GROWN store of measured, executed tapes per (level, seam),
    partitioned once into stable SLOTS by k-means over CONTENT (command sequences), thereafter
    frozen so a slot id means the same thing all run (`native/prop_net`'s "(level, node), not a
    position in `ms`"). Two levels: `ns=1` segment tapes and `ns=n_seg-k` chains that run to the end
    of the piece -- the second is 40-60 steps, past the plant's measured composition horizon (~21),
    which is legato F4's precondition for a chunk paying at all.
  * `World.audition_fm` -- SEAM-TIME AUDITION, the op this node introduces. From the REALIZED seam
    state, roll each candidate's stored commands open-loop under the CURRENT forward model and score
    the arrival at the span's waypoints. Legato G6 licenses it (a seam is handled at the seam, or by
    content selected on realized seams) but legato only ever auditioned at COMMIT time, in the plant,
    on recorded score states; and its consumption-time selection was a frozen k-means key. Note what
    seam-time audition cannot be: a plant replay. The seam is being realized now, so the only
    available imagination is the model's -- and `span/` F2 says a model's promise off-distribution is
    exactly what practice makes worse. That hazard is measured (`aud_horizon`, and the per-level
    calibration readout), not assumed away.
  * `World.fit_cem` -- the budget conversion. A declared per-decision deliberation budget D
    (FM rollout-steps) is spent FIRST on audition and the remainder buys the live-plan option's CEM
    width, so a wide action set costs planner. This is `ratchet`'s `fit_width(budget, G)` idiom in
    the arm's own currency, and it is the mechanism by which "library size pays rent" is a claim
    about outcome rather than only about bookkeeping. `delib_budget <= 0` turns it off and every
    planner gets its CAL-P size, which is the configuration to fall back to if the rent gate says
    the rent does not bind here.
  * `World.traverse_route` -- the dynamic traversal. Per performer, because two performers arriving
    at different postures may commit to different LEVELS; each carries its own next-decision step and
    the body still steps all B in lockstep.
  * `RoutePolicy` -- the decision rule, one mode per arm: `key` (legato's frozen key), `audit`
    (seam-time audition over the whole library), `prop` (pi-gated top-k), `native` (both ports).

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


# The piece's constants live in `piece.py` (which imports nothing, so a Modal LOCAL entrypoint can
# read them on a machine with no numpy). Re-exported here for convenience.
from mjc.practice.accompanist.presto.piece import DEF_WPS, DEF_PATCH_SEG    # noqa: F401,E402


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

    OFFBOOK NOTE (2026-08-25). One counter is ADDED here: `aud`, the number of candidate
    MATERIALISATIONS a seam-time audition performs (one per candidate per performer per decision).
    It is the arm's `n_mat` -- the quantity `native/` prices routing in, and the quantity a widening
    action set inflates. `delib` continues to count FM rollout-STEPS, so an audition contributes
    `n_cand * span` there and one CEM plan contributes `k_shoot * cem_iters * horizon`; the two are
    the same physical unit (one forward pass of the forward model on one state), which is what lets
    the two be traded against each other inside one declared budget. Both are counted, never charged;
    the price vector stays post-hoc. Every existing call site is unchanged (`aud` defaults to 0), so
    a legato arm run through this ledger produces byte-identical priced time.
    """

    def __init__(self):
        self.steps = {"agent": 0, "instrument": 0}
        self.fb = {"agent": 0, "instrument": 0}
        self.plans = {"agent": 0, "instrument": 0}
        self.delib = {"agent": 0, "instrument": 0}
        self.aud = {"agent": 0, "instrument": 0}
        self.t_priced = 0.0            # agent-side priced time -- the headline currency
        self.t_by_kind = {}
        self.n_by_kind = {}            # RAW counts per kind, so the price vector stays post-hoc

    def charge(self, who, steps, fb, dt_ctrl, d_fb, kind="other", plans=0, delib=0, aud=0):
        self.steps[who] += int(steps)
        self.fb[who] += int(fb)
        self.plans[who] += int(plans)
        self.delib[who] += int(delib)
        self.aud[who] += int(aud)
        t = steps * dt_ctrl + fb * d_fb
        if who == "agent":
            self.t_priced += t
            self.t_by_kind[kind] = self.t_by_kind.get(kind, 0.0) + t
            n = self.n_by_kind.setdefault(
                kind, {"steps": 0, "fb": 0, "plans": 0, "delib": 0, "aud": 0})
            n["steps"] += int(steps)
            n["fb"] += int(fb)
            n["plans"] += int(plans)
            n["delib"] += int(delib)
            n["aud"] = n.get("aud", 0) + int(aud)
        return t

    def snapshot(self):
        return {"steps_agent": self.steps["agent"], "steps_instrument": self.steps["instrument"],
                "fb_agent": self.fb["agent"], "fb_instrument": self.fb["instrument"],
                "plans_agent": self.plans["agent"], "plans_instrument": self.plans["instrument"],
                "delib_agent": self.delib["agent"], "delib_instrument": self.delib["instrument"],
                "aud_agent": self.aud["agent"], "aud_instrument": self.aud["instrument"],
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

        # ROUND 4 -- OBSERVATION DELAY on the reflex loop. Every AGENT-SIDE feedback consumer sees
        # the state from `obs_delay` control steps ago: the reactive controller's per-step
        # observation, CEM's initial state at a seam replan, the library key at a launch, and pi's
        # own read. Motor noise makes the stale estimate diverge from the truth, so re-grounding
        # stops being free -- which is the point. Depth exists geometrically on this piece (60 steps
        # against a ~21-step composition horizon) but has been cheap to flatten in every round so
        # far, because a re-ground costs exactly one feedback event and nothing else.
        #
        # EXPERIMENTER-SIDE INSTRUMENTS STAY TRUE-STATE, deliberately and by construction: `e_seg`,
        # `e_piece`, `tips`, `launches`, the plant guard, the metering ledger, and the forward
        # model's TRAINING TRANSITIONS (the delay is a decision/actuation constraint, not a
        # learning-data treatment). One variable.
        self.obs_delay = int(cfg.get("obs_delay", 0))
        # PRESTO ADDITION (2026-08-26) -- EFFERENCE COPY through the delay, i.e. the honest STRONG
        # incumbent. offbook's `obs()` hands a delayed consumer the raw stale state, which is a
        # strawman: a nervous system with a reflex delay does not act on where it WAS, it acts on
        # where its forward model says it now IS, given the motor commands it has already issued.
        # With `obs_predict` on, every agent-side consumer receives
        #     s_hat(t) = FM-rollout( s(t - Delta), raw commands issued in [t - Delta, t) )
        # so the delay costs the agent exactly the part of the last Delta steps it could NOT have
        # predicted -- the motor noise, which the efference copy does not contain -- plus whatever
        # the forward model gets wrong over Delta steps. That is the biologically correct statement
        # of what a reflex delay costs, and it is a much harder incumbent to beat.
        #
        # ADDITIVE AND OFF BY DEFAULT: at `obs_predict = False` every arithmetic path below is
        # byte-identical to the offbook fork (gate G-F asserts the fork; gate G-F2 asserts this
        # flag's default), so `d0` and every offbook run stay reproducible and the naive operator
        # remains available as the continuity read against offbook Round 4.
        self.obs_predict = bool(cfg.get("obs_predict", False))
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

    def train_online_span(self, net, opt, PX, PY, RX, RY, steps, brng, bs, rfrac,
                          span_fn=None, lam=1.0, srng=None, span_opt=None):
        """`train_online` with PORT 2's self-imitation term added to the SAME optimizer step.

        No extra steps for the treatment (`handle/`'s convention, so the plasticity budget is not
        the variable), and the span loss backpropagates through `fm_trunk_t` into the forward
        model's own hidden layers -- which is the interference channel the plant guard watches. With
        `span_fn = None` this is `train_online` op-for-op, so an untreated arm is bit-identical
        through it; note that Adam is invariant to global loss RESCALING on this substrate
        (`priced_plasticity/`), so `lam` acts only through the RELATIVE weight of the two terms,
        which is what it is for.
        """
        import torch
        import torch.nn as nn

        huber = nn.HuberLoss(delta=1.0)
        nb = max(1, int(round(bs * (1.0 - rfrac))))
        nr = max(0, bs - nb)
        net.train()
        losses = []
        for _ in range(steps):
            i = torch.tensor(brng.integers(0, len(PX), size=nb), device=self.device)
            xs, ys = [PX[i]], [PY[i]]
            if nr and len(RX):
                j = torch.tensor(brng.integers(0, len(RX), size=nr), device=self.device)
                xs.append(RX[j]); ys.append(RY[j])
            opt.zero_grad()
            if span_opt is not None:
                span_opt.zero_grad()
            loss = huber(net(torch.cat(xs)), torch.cat(ys))
            if span_fn is not None:
                st = span_fn(srng)
                if st is not None:
                    loss = loss + float(lam) * st
                    losses.append(float(st))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if span_opt is not None:
                span_opt.step()
        net.eval()
        return (float(np.mean(losses)) if losses else None)

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
            hist = [s.copy()]
            issued = []            # efference copy: issued[j] took hist[j] -> hist[j+1]

            def obs():
                """What the agent gets to condition on: the true state `obs_delay` control steps
                ago. At delay 0 this returns the current state and every arithmetic path below is
                byte-identical to the pre-round-4 code.

                With `obs_predict` (presto), the stale observation is carried forward to now by
                rolling the CURRENT forward model along the raw commands already issued since it --
                efference copy. The remaining error is the motor noise the copy cannot contain plus
                the model's own drift over `obs_delay` steps, which is the honest cost of a reflex
                delay. Off by default; at `obs_predict = False` this is offbook's `obs()` verbatim.
                """
                d = self.obs_delay
                if d <= 0:
                    return hist[-1]
                i = max(0, len(hist) - 1 - d)
                if not self.obs_predict:
                    return hist[i]
                st = np.asarray(hist[i], np.float32).copy()
                for u in issued[i:]:
                    st = st + self.fm_delta(fm, st, u)
                return st

            # ---- leg A: the approach. One plan, rolled open-loop at tempo. ----
            g1 = np.tile(self.goals[0][None, :], (B, 1))
            if approach_plan is not None:
                plan = approach_plan
            else:
                plan, _ = self.plan_fn(fm, self.H_app, vel_pen=self.cfg.get("vel_pen_mid", 0.0),
                                       wp_mask=self.approach_mask())(s, g1, rng)
            # `acts_app` / `acts_app_raw` are purely ADDITIVE readouts (2026-08-20, for
            # `../span/`): the approach's issued (post-noise) and raw commands, recorded on the
            # same convention as the drilled segments' `acts` / `acts_raw`. Nothing upstream reads
            # them, no RNG draw moves, so every prior run stays byte-reproducible. They exist so a
            # caller can CONCATENATE consecutive laps of the closed loop into one contiguous
            # executed command sequence -- the composition-horizon probe's measurement ceiling is
            # the length of the command sequence it is given, and a single lap's 60 drilled steps
            # is not enough once the forward model composes past a phrase.
            acts_app = np.empty((B, self.H_app, self.AD), np.float32)
            acts_app_raw = np.empty_like(acts_app)
            tips_app = np.empty((B, self.H_app, 2))
            for h in range(self.H_app):
                a = np.clip(plan[:, h, :], -1, 1)
                acts_app_raw[:, h, :] = a
                issued.append(a.copy())
                if sigma > 0:
                    a = np.clip(a + sigma * rng.standard_normal(a.shape), -1, 1)
                acts_app[:, h, :] = a
                sa, ua, sb, _ = body.step(a.astype(np.float32))
                if collect:
                    trS.append(sa); trU.append(ua); trS2.append(sb)
                s = sb
                hist.append(s.copy())
                tips_app[:, h] = self.tip(s)
            launch0 = s.copy()                        # the PHRASE-LAUNCH state, at W1
            e_app = np.linalg.norm(self.tip(launch0) - self.goals[0][None, :], axis=1)
            fb = 1
            n_plan = 0
            delib = 0

            # ---- legs B..D: the drilled segments, under the routing ----
            acts = np.full((B, self.H_phrase, self.AD), np.nan, np.float32)
            acts_raw = np.full_like(acts, np.nan)
            # `tips` / `tips_app` are ADDITIVE readouts (2026-08-20, for `../span/`): the tip the
            # body actually reached after every control step, by analytic FK (C0 pins it to MuJoCo
            # at <1e-9). They exist so a composition-horizon probe can score an FM rollout against
            # the trajectory the body FLEW rather than against a `true_tips` REPLAY of the issued
            # commands -- the replay re-enters through `set_state` from a float32 state and the
            # arm's divergence amplifies that seed over tens of steps. Pure readout: no RNG draw
            # moves, so every prior run stays byte-reproducible.
            tips = np.full((B, self.H_phrase, 2), np.nan)
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
                                obs(), np.tile(sched[None, h:h + hh], (B, 1, 1)), rng)
                            fb += 1; n_plan += 1; delib += dl
                        a0 = np.clip(dplan[:, h % re, :], -1, 1)
                        issued.append(a0.copy())
                        s, gh = self._apply(body, a0, s, sigma, rng, collect, trS, trU, trS2,
                                            acts, acts_raw, base + h)
                        hist.append(s.copy())
                        tips[:, base + h] = self.tip(s)
                        gate_hits += gh
                        k = end_of.get(base + h)
                        if k is not None:
                            e_seg[:, k] = np.linalg.norm(
                                self.tip(s) - self.goals[k + 1][None, :], axis=1)
                            if k + 1 < self.n_seg:
                                launches[k + 1] = s.copy()
                else:
                    cmds, npl, dl = self.unit_commands(unit, obs().copy(), fm, rng, lo, ns)
                    fb += 1; n_plan += npl; delib += dl
                    for h in range(H):
                        a0 = np.clip(cmds[:, h, :], -1, 1)
                        issued.append(a0.copy())
                        s, gh = self._apply(body, a0, s, sigma, rng, collect, trS, trU, trS2,
                                            acts, acts_raw, base + h)
                        hist.append(s.copy())
                        tips[:, base + h] = self.tip(s)
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
                   e_piece=e_piece, acts=acts, acts_raw=acts_raw,
                   acts_app=acts_app, acts_app_raw=acts_app_raw, tips=tips, tips_app=tips_app,
                   n_fb=fb, n_plan=n_plan,
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

    # ==================================================================================== #
    # OFFBOOK ADDITIONS -- seam-time audition, the budget conversion, the dynamic traversal
    # ==================================================================================== #

    def fm_trunk_t(self, net, s_t):
        """The forward model's own penultimate activations at a state, with the command pinned to
        zero. Differentiable, and the channel through which Port 2's loss reaches the plant.

        Why a zero command: the span head is asked "given this posture, what does the unit play
        here", and the command is what it is about to emit -- feeding it back in would be circular.
        `u = 0` is the model's read of the state alone under its own normaliser.
        """
        import torch
        z = torch.zeros(s_t.shape[0], self.AD, device=s_t.device, dtype=s_t.dtype)
        h = (torch.cat([s_t, z], 1) - self.norm["mx"]) / self.norm["sx"]
        for m in list(net)[:-1]:
            h = m(h)
        return h

    def trunk_dim(self):
        return int(self.cfg["fm_hidden"])

    def rollout_score(self, net, S, C, cps):
        """Open-loop FM rollout of `C` from `S`, scored at `cps` = [(step, goal)].

        Byte-for-byte the same arithmetic as `plan_fn`'s inner loop (same normaliser, same
        `fk_torch`), so seam-time audition scores a candidate on exactly the world the planner is
        searching in -- the two options in the action set are compared under one model, not two.

        Returns (M, n_cp) checkpoint distances.
        """
        import torch
        with torch.no_grad():
            s = torch.tensor(np.asarray(S, np.float32), device=self.device)
            Ct = torch.tensor(np.asarray(C, np.float32), device=self.device)
            want = {int(h): torch.tensor(np.asarray(g, np.float32), device=self.device)
                    for h, g in cps}
            out = []
            for h in range(Ct.shape[1]):
                x = torch.cat([s, Ct[:, h, :]], 1)
                s = s + (net((x - self.norm["mx"]) / self.norm["sx"]) * self.norm["sy"]
                         + self.norm["my"])
                if h in want:
                    out.append((self.fk_torch(s[:, :self.n]) - want[h][None, :]).norm(dim=1))
            return torch.stack(out, 1).cpu().numpy() if out else np.zeros((len(S), 0), np.float32)

    def aud_checkpoints(self, seg_lo, n_segs, horizon=0):
        """The checkpoints a seam-time audition scores, and how far it rolls.

        `horizon = 0` scores the candidate's whole span. `horizon > 0` truncates the rollout there
        and scores only the waypoints inside it -- the HORIZON-TRUNCATED audition. It exists because
        `span/` F2 is a standing fact about this plant: a live model's promised far-seam arrival
        stays ~0.07 m at every cycle while the truth walks to ~1 m, so an audition that scores a
        60-step chain on the model's own 60-step imagination is reading a promise, not a
        measurement, and would systematically over-rate the longest candidate. Which convention the
        node runs under is a calibration (gate G-C), not a guess.
        """
        cps = self.checkpoints(seg_lo, n_segs)
        H = self.span(seg_lo, n_segs)
        if int(horizon) <= 0:
            return cps, H
        h = int(min(int(horizon), H))
        keep = [(i, g) for i, g in cps if i < h]
        if not keep:                       # a span shorter than one waypoint: score where we stop
            keep = [(h - 1, cps[0][1])]
        return keep, h

    def audition_fm(self, fm, S, tapes, pairs, seg_lo, n_segs, horizon=0, chunk=40000):
        """SEAM-TIME AUDITION. The op this node introduces.

        `S` (B, SD) are the seam states the body ACTUALLY reached; `tapes` (N, H, AD) the stored
        commands; `pairs` (M, 2) the ragged (performer row, tape index) set to score -- ragged
        because under routing different performers audition different slots. Returns (M,) mean
        checkpoint distance under the current forward model.

        Legato G6 licenses this op and never ran it: a seam is handled AT the seam, or by content
        selected on realised seams -- yet `fingering/`'s audition only ever ran at commit time, in
        the plant, on recorded score states, and legato's consumption-time selection was a frozen
        k-means key. The op is priced per MATERIALISATION (`aud`) and per FM rollout-STEP (`delib`),
        which is what makes library size pay rent.
        """
        pairs = np.asarray(pairs, int)
        cps, H = self.aud_checkpoints(seg_lo, n_segs, horizon)
        if len(pairs) == 0:
            return np.zeros(0, np.float32)
        out = np.empty(len(pairs), np.float32)
        for i in range(0, len(pairs), int(chunk)):
            p = pairs[i:i + int(chunk)]
            e = self.rollout_score(fm, np.asarray(S)[p[:, 0]],
                                   np.asarray(tapes)[p[:, 1]][:, :H, :], cps)
            out[i:i + len(p)] = e.mean(1)
        return out

    @staticmethod
    def fit_cem(budget, span, iters, ladder):
        """The budget conversion -- `ratchet.fit_width(budget, G)` in the arm's own currency.

        A declared per-decision deliberation budget (FM rollout-STEPS) is spent first on audition;
        what is left buys the live option's CEM width. Returns the largest `k_shoot` on the ladder
        whose search fits, or None when even the narrowest does not -- at which point the action set
        has grown wide enough to price the live plan out entirely, which is `tall/`'s
        widening-action-set rent (committing grew the action set and `fit_width` dropped the beam
        from width 2 to width 1) in its sharpest available form.

        `budget <= 0` disables the conversion: every planner gets its CAL-P size and the rent is
        reported as a cost rather than converted into an outcome. That is the configuration to fall
        back to if the rent gate says the rent does not bind on this substrate.
        """
        if budget is None or float(budget) <= 0:
            return None
        best = None
        for k in sorted(int(x) for x in ladder):
            if k * int(iters) * int(span) <= float(budget):
                best = k
        return best

    def traverse_route(self, fm, policy, q0, rng, sigma, ledger, who="agent", kind="route",
                       approach_plan=None, collect=False, qd0=None, stop_seg=None,
                       explore=False, cycle=0, capture=False):
        """Execute the piece once for `len(q0)` performers under a DYNAMIC routing policy.

        The difference from `traverse`, and the whole node: legato executes a partition of the
        drilled segments decided BEFORE the traversal; here the partition is decided AT each seam
        from the state the body actually reached, PER PERFORMER -- two performers arriving at
        different postures may commit to different LEVELS, so each carries its own next-decision
        step while the body still steps all B in lockstep.

        The approach leg is byte-identical to `traverse`'s (same plan, same noise draws, same
        collection), which is what keeps the phrase-launch distribution a property of the world.

        Feedback events: 1 for the approach launch, then 1 per performer per DECISION -- the routing
        is the price, exactly as in legato, but now the routing is a random variable.
        """
        q0 = np.asarray(q0, dtype=np.float64)
        B = len(q0)
        stop = self.n_seg if stop_seg is None else int(stop_seg)
        body = self.body(B)
        seg_at = {int(self.seg_lo[k]): k for k in range(self.n_seg)}
        end_of = {int(self.seg_hi[k]) - 1: k for k in range(self.n_seg)}
        try:
            s = body.reset(rng, self.qc, 0.0, q0=q0, qd0=qd0)
            trS, trU, trS2 = [], [], []
            hist = [s.copy()]
            issued = []            # efference copy: issued[j] took hist[j] -> hist[j+1]

            def obs():
                """The agent's read: the true state `obs_delay` control steps ago. Everything the
                POLICY sees flows through here -- pi's input, the audition's scoring state, the
                library key, and CEM's initial state -- so the delay is one change point. The
                traversal keeps executing from the true state, and every readout below records the
                true state, which is what keeps this a decision constraint rather than a grading
                one.

                `obs_predict` (presto) carries the stale read forward by efference copy; see
                `traverse`. Off by default and byte-identical to offbook when off."""
                d = self.obs_delay
                if d <= 0:
                    return hist[-1]
                i = max(0, len(hist) - 1 - d)
                if not self.obs_predict:
                    return hist[i]
                st = np.asarray(hist[i], np.float32).copy()
                for u in issued[i:]:
                    st = st + self.fm_delta(fm, st, u)
                return st

            # ---- leg A: the approach (verbatim from `traverse`) ----
            g1 = np.tile(self.goals[0][None, :], (B, 1))
            if approach_plan is not None:
                plan = approach_plan
            else:
                plan, _ = self.plan_fn(fm, self.H_app, vel_pen=self.cfg.get("vel_pen_mid", 0.0),
                                       wp_mask=self.approach_mask())(s, g1, rng)
            acts_app = np.empty((B, self.H_app, self.AD), np.float32)
            acts_app_raw = np.empty_like(acts_app)
            tips_app = np.empty((B, self.H_app, 2))
            for h in range(self.H_app):
                a = np.clip(plan[:, h, :], -1, 1)
                acts_app_raw[:, h, :] = a
                issued.append(a.copy())
                if sigma > 0:
                    a = np.clip(a + sigma * rng.standard_normal(a.shape), -1, 1)
                acts_app[:, h, :] = a
                sa, ua, sb, _ = body.step(a.astype(np.float32))
                if collect:
                    trS.append(sa); trU.append(ua); trS2.append(sb)
                s = sb
                hist.append(s.copy())
                tips_app[:, h] = self.tip(s)
            launch0 = s.copy()
            e_app = np.linalg.norm(self.tip(launch0) - self.goals[0][None, :], axis=1)
            fb = B                                   # the approach launch, one per performer
            n_plan = 0
            delib = 0
            aud_delib = 0
            n_aud = 0
            n_reh = 0

            # ---- legs B..D: the drilled segments, under the LIVE policy ----
            acts = np.full((B, self.H_phrase, self.AD), np.nan, np.float32)
            acts_raw = np.full_like(acts, np.nan)
            tips = np.full((B, self.H_phrase, 2), np.nan)
            launches = {0: launch0}
            e_seg = np.full((B, self.n_seg), np.nan)
            pend = np.full((B, self.H_phrase, self.AD), np.nan, np.float32)
            next_dec = np.zeros(B, int)
            gate_hits = 0
            decisions = []
            stop_h = int(self.seg_hi[stop - 1]) if stop > 0 else 0
            for h in range(stop_h):
                need = np.nonzero(next_dec == h)[0]
                if len(need):
                    k = seg_at[h]
                    s_obs = obs()
                    d = policy.decide(self, fm, s_obs[need], k, rng, who=who, explore=explore,
                                      cycle=cycle, capture=capture)
                    for j, i in enumerate(need):
                        ns = int(d["n_segs"][j])
                        hh = self.span(k, ns)
                        pend[i, h:h + hh, :] = d["cmds"][j, :hh, :]
                        next_dec[i] = h + hh
                    fb += len(need)
                    n_plan += int(d.get("plans", 0))
                    delib += int(d.get("delib", 0))
                    aud_delib += int(d.get("aud_delib", 0))
                    n_aud += int(d.get("aud", 0))
                    n_reh += int(d.get("n_reh", 0))
                    decisions.append(dict(
                        seam=int(k), rows=[int(x) for x in need],
                        slot=[int(x) for x in d["slot"]],
                        n_segs=[int(x) for x in d["n_segs"]],
                        src=[int(x) for x in d["src"]],
                        score=[float(x) for x in d["score"]],
                        state=np.asarray(s_obs[need], np.float32),   # what pi conditioned on
                        n_cand=[int(x) for x in d["n_cand"]],
                        k_shoot=int(d.get("k_shoot") or 0),
                        aud=int(d.get("aud", 0)), delib=int(d.get("delib", 0))))
                a0 = np.clip(pend[:, h, :], -1, 1)
                issued.append(a0.copy())
                if not np.isfinite(a0).all():
                    raise RuntimeError(f"routing left step {h} unfilled -- decision bookkeeping bug")
                s, gh = self._apply(body, a0, s, sigma, rng, collect, trS, trU, trS2,
                                    acts, acts_raw, h)
                hist.append(s.copy())
                tips[:, h] = self.tip(s)
                gate_hits += gh
                kk = end_of.get(h)
                if kk is not None:
                    e_seg[:, kk] = np.linalg.norm(
                        self.tip(s) - self.goals[kk + 1][None, :], axis=1)
                    if kk + 1 < self.n_seg:
                        launches[kk + 1] = s.copy()
        finally:
            body.release()

        steps = B * (self.H_app + stop_h)
        t_piece = ledger.charge(who, steps, fb, self.dt_ctrl, self.d_fb, kind=kind,
                                plans=n_plan, delib=delib, aud=n_aud)
        e_piece = (np.nanmean(e_seg[:, :stop], axis=1) if stop > 0 else np.full(B, np.nan))
        out = dict(launch=launch0, launches=launches, final=s.copy(), e_app=e_app, e_seg=e_seg,
                   e_piece=e_piece, acts=acts, acts_raw=acts_raw,
                   acts_app=acts_app, acts_app_raw=acts_app_raw, tips=tips, tips_app=tips_app,
                   n_fb=fb / max(B, 1), n_plan=n_plan, delib=delib, aud=n_aud,
                   aud_delib=aud_delib, n_reh=n_reh,
                   decisions=decisions,
                   t_piece=t_piece / max(B, 1), t_total=t_piece,
                   gate_frac=gate_hits / float(B * max(stop_h, 1)))
        if collect:
            out["trans"] = (np.concatenate(trS, 0), np.concatenate(trU, 0),
                            np.concatenate(trS2, 0))
        return out


# ====================================================================================== #
# THE ADDRESS BOOK
# ====================================================================================== #

SRC_TAPE, SRC_HEAD, SRC_PRIM = 0, 1, 2


class Cell:
    """One (ns, k) cell of the library: every measured tape spanning `ns` drilled segments from
    seam `k`, partitioned into slots.

    SLOTS ARE FIT ONCE AND FROZEN. k-means over CONTENT (the flattened command sequence) at the
    first commit; every later commit assigns its new tapes to the nearest existing centroid. A slot
    id therefore means the same thing at cycle 20 and at cycle 90, which is what makes pi's logits,
    the span head's conditioning and the trust-formation series comparable across the run at all --
    `native/prop_net`'s "(level, node), not a position in `ms`", on a substrate where the action set
    is a growing pile of tapes rather than a table.
    """

    def __init__(self, ns, k, span, act_dim, state_dim, width, n_slot):
        self.ns, self.k, self.span = int(ns), int(k), int(span)
        self.width, self.n_slot = int(width), int(n_slot)
        self.tapes = np.zeros((0, self.span, int(act_dim)), np.float32)
        self.launch = np.zeros((0, int(state_dim)), np.float32)
        self.err = np.zeros(0, np.float32)
        self.cyc = np.zeros(0, int)
        self.slot = np.zeros(0, int)
        self.spell = np.zeros((0, self.ns), int)
        self.cent = None                    # (n_slot, span*act_dim) content centroids, frozen
        self.key_c = np.zeros((self.width, int(state_dim)), np.float32)
        self.key_n = np.zeros(self.width, int)
        self.rep = -np.ones(self.width, int)
        self.deleted = False

    # -------------------------------------------------------------- growth
    def _assign(self, tapes, rng):
        X = np.asarray(tapes, np.float64).reshape(len(tapes), -1)
        if self.cent is None:
            C, lab = kmeans(X, self.n_slot, rng)
            if len(C) < self.n_slot:        # k-means may return fewer than asked on a tiny pool
                C = np.concatenate([C, np.tile(C[-1:], (self.n_slot - len(C), 1))])
            self.cent = C
            return lab
        D = ((X[:, None, :] - self.cent[None, :, :]) ** 2).sum(-1)
        return D.argmin(1)

    def add(self, tapes, launch, err, cyc, spell, rng, slot=None):
        tapes = np.asarray(tapes, np.float32)
        if len(tapes) == 0:
            return
        lab = np.full(len(tapes), int(slot), int) if slot is not None \
            else self._assign(tapes, rng)
        self.tapes = np.concatenate([self.tapes, tapes])
        self.launch = np.concatenate([self.launch, np.asarray(launch, np.float32)])
        self.err = np.concatenate([self.err, np.asarray(err, np.float32)])
        self.cyc = np.concatenate([self.cyc, np.full(len(tapes), int(cyc), int)])
        self.slot = np.concatenate([self.slot, lab])
        self.spell = np.concatenate([self.spell, np.asarray(spell, int).reshape(len(tapes), self.ns)])
        self._refresh()

    def _refresh(self):
        for j in range(self.width):
            m = self.slot == j
            self.key_n[j] = int(m.sum())
            if m.any():
                self.key_c[j] = self.launch[m].mean(0)
                # the slot's representative: its lowest REALISED error. Used only by `key_frozen`
                # (which has no audition to pick with) and by the poison construction. Ranking a
                # tape by its own realised error is the etude E-3b winner's curse; every arm whose
                # claims this node carries selects by seam-time audition instead, which has no such
                # curse because it scores on states the tape never saw.
                idx = np.nonzero(m)[0]
                self.rep[j] = int(idx[int(np.argmin(self.err[idx]))])

    # -------------------------------------------------------------- reads
    def members(self, j):
        if self.deleted:
            return np.zeros(0, int)
        return np.nonzero(self.slot == int(j))[0]

    def live_slots(self):
        return [j for j in range(self.width) if self.key_n[j] > 0]

    def spelling(self, j):
        """[(position, local ns=1 slot)] the chain slot's members were spelled with -- the
        can't-decompose probe's denominator, and the alphabet phrase-level minability counts over.
        Position i is the drilled segment k+i, so the caller maps it to the global ns=1 slot id."""
        m = self.slot == int(j)
        if not m.any():
            return []
        sp = self.spell[m]
        return sorted({(int(i), int(v)) for i in range(self.ns) for v in sp[:, i] if v >= 0})


class Library:
    """Every cell, plus the two things the port is about: what is addressable, and what survives
    deleting the store.

    Legato F5 is this object's measured seed -- the segment library manufactures the phrase pool's
    addressable variation -- so the chain cells are deliberately grown LATER than the segment cells,
    from traversals that are already routed. A chain pool harvested from reactive warmup is the
    configuration F1 measured at 1.03x and F5 corrected to 1.34x.
    """

    def __init__(self, W, layout, n_slot, seed):
        self.W, self.layout = W, layout
        self.n_slot = int(n_slot)
        self.rng = np.random.default_rng(int(seed))
        self.cells = {}
        for k in range(W.n_seg):
            for ns in layout.levels(k):
                self.cells[(ns, k)] = Cell(ns, k, W.span(k, ns), W.AD, W.SD,
                                           layout.width, n_slot)
        self.deleted = False

    def cell(self, ns, k):
        return self.cells[(int(ns), int(k))]

    def populated(self, k, levels=None):
        """[(ns, j, sid)] for every live slot legal at seam k."""
        out = []
        for ns in self.layout.levels(k):
            if levels is not None and ns not in levels:
                continue
            c = self.cells[(ns, k)]
            if c.deleted:
                continue
            for j in c.live_slots():
                out.append((ns, j, self.layout.slot(ns, k, j)))
        return out

    def any_live(self):
        return any((not c.deleted) and len(c.tapes) for c in self.cells.values())

    def delete(self):
        """The battery's table ablation. The tapes go; the SLOT METADATA (centroids, key centroids,
        spellings) stays, because that is what the address book is -- routing-not-pruning. pi can
        still name a slot and the span head can still render one; there is simply nothing stored to
        play back. Enumeration has no execution path left at all, which is the contrast."""
        for c in self.cells.values():
            c.deleted = True
        self.deleted = True

    def sizes(self):
        return {f"{ns}:{k}": int(len(c.tapes)) for (ns, k), c in self.cells.items()}

    def spellings(self):
        out = {}
        for (ns, k), c in self.cells.items():
            if ns == 1:
                continue
            for j in c.live_slots():
                sp = c.spelling(j)
                if sp:
                    out[self.layout.slot(ns, k, j)] = sorted(
                        {self.layout.slot(1, k + i, v) for i, v in sp})
        return out

    def minability(self, spell_rows):
        """PHRASE-LEVEL MINABILITY, keyed table-free: how many DISTINCT chain spellings (tuples of
        segment slot ids) the agent's own chosen traversals contain. `native/full`'s battery
        separated the BUILT next-level entries (structurally gone when the table goes) from the
        OBSERVATION STREAM (which should not be); this is the stream."""
        if len(spell_rows) == 0:
            return 0
        return int(len({tuple(int(x) for x in r) for r in np.asarray(spell_rows, int)}))


# ====================================================================================== #
# THE DECISION RULE AT A SEAM
# ====================================================================================== #

class RoutePolicy:
    """What the agent does when it arrives at a seam. One mode per arm.

        `key`     legato's op, in this node's harness: the nearest key centroid picks a slot and
                  the slot's representative tape is played. O(1), no audition, no live option --
                  which is what a frozen keyed library IS, and why this arm is the taxonomy's
                  continuity check rather than a strict one-variable neighbour of `audit`.
        `audit`   SEAM-TIME AUDITION over the whole action set: every tape in every legal slot, plus
                  the live option, rolled forward under the current model from the realised seam
                  state, argmin launched. O(K) per seam -- the enumeration reference.
        `prop`    PORT 1. pi ranks the slots; only the top-k are auditioned (the live option is
                  always forced in, so it is never ranked out by an untrained logit). O(k).
        `native`  PORT 1 + PORT 2. A slot whose span head has cleared parity contributes ONE
                  candidate -- the head's emission -- instead of all its members, so routing cuts
                  ACROSS slots and the corridor cuts WITHIN them.

    THE LIVE OPTION (`prim`). One CEM plan over the current drilled segment from the observed seam
    state, flown open-loop: `fingering/`'s `plan_launch`, which dominated every frozen op 1.6x
    INSIDE the composition horizon. It is the primitive action the library's chunks are made of and
    the thing the battery ablates. Full reactive MPC stays where it belongs, as the `never`
    reference arm -- a controller that re-grounds every step cannot be auditioned against a stored
    tape, because it has no command sequence until it has already been run.
    """

    def __init__(self, mode, W, layout, library, *, k_prop=2, eps=0.15, eps_act=0.10,
                 aud_horizon=0,
                 p_rehearse=0.0,
                 delib_budget=0.0, cem_ladder=(128, 256, 512, 1024, 2048), cem_iters=8,
                 k_shoot=1024, prim=True, span_tau=0.9, span_tol=0.0, span_min_hold=64,
                 seed=0, device="cpu"):
        self.mode, self.W, self.layout, self.library = mode, W, layout, library
        self.k_prop, self.eps = int(k_prop), float(eps)
        # EXPLORATION ON THE ACTION, not only on the audition set. Gate G-C measured that seam-time
        # audition ranks segment tapes well (rho = 0.92) and CHAINS barely at all (rho = 0.34) --
        # `span/` F2 arriving as a constraint on the op rather than on the content -- and in the
        # gate the audition committed a chain 0.00 of the time at every budget. So widening what
        # gets SCORED cannot, on its own, get a chain executed: the ranker deciding among the
        # widened set is exactly the thing that is blind there. epsilon-greedy on the launched
        # action closes that loop: PRACTICE sometimes plays a uniformly-drawn slot from its own
        # audition set, the BODY grades the result, and pi's self-imitation (which filters on
        # realised piece error, never on the audition score) can then form trust in a level the
        # model-graded op cannot see. Practice explores, performance does not -- the substrate's
        # own convention, `native/prop`'s measured lock-in one level out. Uniform over SLOTS, not
        # over candidates, so a populous slot does not crowd out a sparse one.
        self.eps_act = float(eps_act)
        # ROUND 3 -- REHEARSAL, i.e. exploration moved UPSTREAM of pi's own gate. O2 measured that
        # chain trust tracked exposure and only exposure, and that the exposure path was a product
        # of two epsilons: `explore_eps` adds a slot to the AUDITION SET and `eps_act` then draws
        # from that set -- which is itself pi-gated, so the correction sat downstream of the lock-in
        # it was meant to break (~0.002 chain launches per decision, 3 chain plays in 100 cycles).
        # A rehearsal draw is uniform over ALL LEGAL SLOTS and launches what it draws, so exposure
        # is independent of what pi currently believes. Practice only; performance probes stay pure
        # exploit policy. At `p_rehearse = 0` no draw is taken and the arm is bit-identical to O2's.
        self.p_reh = float(p_rehearse)
        self.force_until = {}         # (ns, k) -> last cycle this cell is forced into the audition

        self.aud_horizon = int(aud_horizon)
        self.delib_budget = float(delib_budget)
        self.cem_ladder = tuple(sorted(int(x) for x in cem_ladder))
        self.cem_iters, self.k_shoot = int(cem_iters), int(k_shoot)
        self.prim = bool(prim)
        self.levels = None            # None = every level; a set restricts the action set (gates)
        self.span_tau, self.span_tol = float(span_tau), float(span_tol)
        self.span_min_hold = int(span_min_hold)
        self.device = device
        self.erng = np.random.default_rng(int(seed) + 777)     # exploration only: its own stream
        self.pi = None
        self.span_head = None
        self.sbuf = None
        self.parity = {}
        self.parity_val = {}
        self.norm = (np.zeros(W.SD, np.float32), np.ones(W.SD, np.float32))
        self.norm_set = False
        self.pairs = []            # (seam state, seam index, chosen slot) for pi's self-imitation
        self.tape_calls = 0
        self.head_calls = 0

    # ------------------------------------------------------------------ setup
    def set_norm(self, S):
        S = np.asarray(S, np.float32)
        self.norm = (S.mean(0), S.std(0) + 1e-6)
        self.norm_set = True

    def legal(self, k):
        """(global slot ids, entries) legal at seam k. `entries[p] = (kind, ns, j, sid)`."""
        ent = [("slot", ns, j, sid) for ns, j, sid in self.library.populated(k, self.levels)]
        if self.prim:
            ent.append(("prim", 1, -1, self.layout.prim(k)))
        return [e[3] for e in ent], ent

    def pi_logits(self, S, k):
        import torch
        import torch.nn.functional as F
        mu, sd = self.norm
        with torch.no_grad():
            z = torch.tensor((np.asarray(S, np.float32) - mu) / sd, device=self.device)
            oh = F.one_hot(torch.full((len(S),), int(k), dtype=torch.long, device=self.device),
                           num_classes=self.W.n_seg).float()
            return self.pi(z, oh).cpu().numpy()

    def emit(self, fm, S, sid, span):
        import torch
        mu, sd = self.norm
        with torch.no_grad():
            s_t = torch.tensor(np.asarray(S, np.float32), device=self.device)
            tr = self.W.fm_trunk_t(fm, s_t)
            z = (s_t - torch.tensor(mu, device=self.device)) / torch.tensor(sd, device=self.device)
            y = self.span_head(z, tr, torch.full((len(S),), int(sid), dtype=torch.long,
                                                 device=self.device))
        return y[:, :int(span), :].cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ the decision
    def decide(self, W, fm, S, k, rng, who="agent", explore=False, cycle=0, capture=False):
        from mjc.practice.accompanist.presto import nets as N
        n = len(S)
        ids, ent = self.legal(k)
        if not ent:
            raise RuntimeError(f"no legal action at seam {k}: library empty and primitives ablated")
        if self.mode == "key":
            return self._decide_key(W, fm, S, k, ent, rng)

        # ---- 1. which slots does each performer audition?
        if self.mode == "audit" or self.pi is None:
            sel = [np.arange(len(ent)) for _ in range(n)]
            n_prop = len(ent)
        else:
            lg = self.pi_logits(S, k)
            # the live option is always in; so is every slot of a cell inside its post-commit
            # forcing window, so a freshly minted address gets its first exposure instead of being
            # ranked out by an untrained logit (`native/prop_net`'s `forced` idiom).
            forced = [p for p, e in enumerate(ent) if e[0] == "prim"]
            if explore:
                forced += [p for p, e in enumerate(ent)
                           if e[0] == "slot" and cycle <= self.force_until.get((e[1], k), -1)]
            base = N.select_slots(lg, ids, self.k_prop, forced=forced)
            ex = N.explore_slots(base, len(ent), self.eps if explore else 0.0, self.erng)
            sel = N.select_slots(lg, ids, self.k_prop, forced=forced, explore_idx=ex)
            n_prop = self.k_prop

        # ---- 1b. the rehearsal draw, BEFORE candidate assembly so the drawn slot is materialised
        reh = [None] * n
        if explore and self.p_reh > 0.0 and self.mode != "key":
            for i in range(n):
                if self.erng.random() < self.p_reh:
                    reh[i] = int(self.erng.integers(len(ent)))
                    if reh[i] not in sel[i]:
                        sel[i] = np.unique(np.concatenate([sel[i], [reh[i]]]))

        # ---- 2. price the audition, then fit the live option's CEM width with what is left
        def slot_cost(p):
            kind, ns, j, sid = ent[p]
            if kind == "prim":
                return W.span(k, 1)                     # scoring the plan is one materialisation
            span = W.span(k, ns)
            if self.mode == "native" and self.parity.get(sid, False) and not capture:
                return span
            m = self.library.cell(ns, k).members(j)
            extra = span if (self.mode == "native" and self.parity.get(sid, False)) else 0
            return span * len(m) + extra

        row_spend = np.array([sum(slot_cost(p) for p in s) for s in sel], float)
        ks = self.k_shoot
        if self.delib_budget > 0:
            ks = W.fit_cem(self.delib_budget - float(row_spend.max() if n else 0.0),
                           W.span(k, 1), self.cem_iters, self.cem_ladder)

        # ---- 3. assemble candidates
        # Every candidate is registered as (row, slot id, span-in-segments, source, resolver,
        # competes). `resolver` is the O(1) way back to the command sequence; `competes` is False
        # only for a tape candidate that is present purely to refresh Port 2's target (a slot whose
        # head is open acts through the head, and the tape audition beside it is bookkeeping).
        reg = []
        tape_pairs = {}                # ns -> [(row, tape_idx, reg_index)]
        syn = {}                       # ns -> [(row, cmds, reg_index)]
        syn_bank = {}                  # ns -> list of command arrays (resolver target)
        prim_rows = sorted({i for i in range(n) for p in sel[i] if ent[p][0] == "prim"})
        plan_cmds, plans, delib = None, 0, 0
        if prim_rows and ks is not None:
            H1 = W.span(k, 1)
            pf = W.plan_fn(fm, H1, k_shoot=ks, cem_iters=self.cem_iters,
                           cem_elite=elite_for(ks), vel_pen=W.vel_pen_for(k, 1),
                           wp_mask=W.waypoint_mask(int(W.seg_lo[k]), H1))
            gs = np.tile(W.goal_schedule(k, 1)[None], (len(prim_rows), 1, 1))
            plan_cmds, dl = pf(np.asarray(S)[prim_rows], gs, rng)
            plans, delib = len(prim_rows), int(dl)
        plan_delib = int(delib)
        prim_at = {r: i for i, r in enumerate(prim_rows)}

        # head emissions are batched per (slot, rows) so Port 2 costs one forward pass per slot
        head_rows = {}
        for i in range(n):
            for p in sel[i]:
                kind, ns, j, sid = ent[p]
                if kind != "prim" and self.mode == "native" and self.parity.get(sid, False):
                    head_rows.setdefault((ns, sid), []).append(i)
        head_emit = {}
        for (ns, sid), rows in head_rows.items():
            head_emit[(ns, sid)] = dict(zip(rows, self.emit(fm, np.asarray(S)[rows], sid,
                                                            W.span(k, ns))))

        for i in range(n):
            for p in sel[i]:
                kind, ns, j, sid = ent[p]
                if kind == "prim":
                    if plan_cmds is None:
                        continue                       # priced out: the rent, in its sharpest form
                    m = len(syn_bank.setdefault(1, []))
                    syn_bank[1].append(plan_cmds[prim_at[i]])
                    syn.setdefault(1, []).append((i, m, len(reg)))
                    reg.append((i, sid, 1, SRC_PRIM, ("syn", 1, m), True))
                    continue
                open_head = (self.mode == "native" and self.parity.get(sid, False))
                if open_head:
                    m = len(syn_bank.setdefault(ns, []))
                    syn_bank[ns].append(head_emit[(ns, sid)][i])
                    syn.setdefault(ns, []).append((i, m, len(reg)))
                    reg.append((i, sid, ns, SRC_HEAD, ("syn", ns, m), True))
                if (not open_head) or capture:
                    for t in self.library.cell(ns, k).members(j):
                        tape_pairs.setdefault(ns, []).append((i, int(t), len(reg)))
                        reg.append((i, sid, ns, SRC_TAPE, ("tape", ns, int(t)),
                                    not open_head))

        if not reg:
            raise RuntimeError(f"seam {k}: every action priced out (budget {self.delib_budget})")

        # ---- 4. score every candidate under the current model, from the REALISED seam state
        scores = np.full(len(reg), np.inf, np.float32)
        n_aud = 0
        for ns, lst in tape_pairs.items():
            arr = np.array([[r, t] for r, t, _ in lst], int)
            e = W.audition_fm(fm, S, self.library.cell(ns, k).tapes, arr, k, ns,
                              horizon=self.aud_horizon)
            for (_, _, ri), v in zip(lst, e):
                scores[ri] = v
            n_aud += len(lst)
            delib += len(lst) * W.aud_checkpoints(k, ns, self.aud_horizon)[1]
        for ns, lst in syn.items():
            bank = np.stack(syn_bank[ns]).astype(np.float32)
            arr = np.array([[r, m] for r, m, _ in lst], int)
            e = W.audition_fm(fm, S, bank, arr, k, ns, horizon=self.aud_horizon)
            for (_, _, ri), v in zip(lst, e):
                scores[ri] = v
            n_aud += len(lst)
            delib += len(lst) * W.aud_checkpoints(k, ns, self.aud_horizon)[1]

        # ---- 5. launch the argmin, per performer
        def cmds_of(ri):
            kind, ns_, ref = reg[ri][4]
            return (self.library.cell(ns_, k).tapes[ref] if kind == "tape"
                    else np.asarray(syn_bank[ns_][ref], np.float32))

        by_row = [[] for _ in range(n)]
        for ri, r in enumerate(reg):
            by_row[r[0]].append(ri)
        max_span = W.H_phrase
        cmds = np.full((n, max_span, W.AD), np.nan, np.float32)
        slot = np.zeros(n, int); nsg = np.ones(n, int); src = np.zeros(n, int)
        best = np.zeros(n, np.float32); ncand = np.zeros(n, int)
        cap_S, cap_C, cap_sid = [], [], []
        n_reh = 0
        for i in range(n):
            rows = by_row[i]
            ncand[i] = len(rows)
            act_rows = [r for r in rows if reg[r][5]] or rows
            w = act_rows[int(np.argmin(scores[act_rows]))]
            if reh[i] is not None:
                sid_r = ent[reh[i]][3]
                alt = [r for r in act_rows if reg[r][1] == sid_r]
                if alt:
                    w = alt[int(np.argmin(scores[alt]))]
                    n_reh += 1
            elif explore and self.eps_act > 0.0:
                # one draw per performer per decision either way, so the stream is a deterministic
                # function of the audition SET -- which is what keeps `fid` and `audit_prop_kN`
                # bit-identical to `audit_all`, whose sets are the same.
                if self.erng.random() < self.eps_act:
                    here = sorted({reg[r][1] for r in act_rows})
                    pick = int(self.erng.choice(here)) if len(here) > 1 else here[0]
                    alt = [r for r in act_rows if reg[r][1] == pick]
                    w = alt[int(np.argmin(scores[alt]))]
            _, sid, ns, sr = reg[w][:4]
            slot[i], nsg[i], src[i], best[i] = sid, ns, sr, scores[w]
            cmds[i, :W.span(k, ns)] = cmds_of(w)
            if sr == SRC_TAPE:
                self.tape_calls += 1
            elif sr == SRC_HEAD:
                self.head_calls += 1
            # capture for Port 2's self-imitation. EVERY selected library slot contributes, not
            # only the one that won: `native/span`'s convention is that every macro call, open or
            # closed, feeds its slot's buffer, and here the reason is sharper -- a live plan can win
            # most decisions (it does, at a loose deliberation budget), and a head that only learns
            # from launched calls would starve exactly when the corridor most needs consolidating.
            # The target is the TAPE the audition would have picked inside that slot, recomputed
            # now, so the head chases the executor it is replacing rather than a stale copy of it.
            if capture and self.sbuf is not None:
                for s_ in sorted({reg[r][1] for r in rows if reg[r][3] == SRC_TAPE}):
                    tr = [r for r in rows if reg[r][1] == s_ and reg[r][3] == SRC_TAPE]
                    if not tr:
                        continue
                    b = tr[int(np.argmin(scores[tr]))]
                    _, _, ns_b, _ = reg[b][:4]
                    cap_S.append(np.asarray(S)[i]); cap_sid.append(int(s_))
                    cc = np.zeros((max_span, W.AD), np.float32)
                    cc[:W.span(k, ns_b)] = cmds_of(b)
                    cap_C.append(cc)
        if capture and cap_S and self.sbuf is not None:
            for sid in sorted(set(cap_sid)):
                m = [q for q, x in enumerate(cap_sid) if x == sid]
                self.sbuf.store(int(sid), np.stack([cap_S[q] for q in m]),
                                np.stack([cap_C[q] for q in m]))
        return dict(cmds=cmds, n_segs=nsg, slot=slot, src=src, score=best, n_cand=ncand,
                    plans=plans, delib=int(delib), aud=int(n_aud), k_shoot=(ks or 0),
                    n_prop=int(n_prop), aud_delib=int(delib) - plan_delib,
                    plan_delib=plan_delib, n_reh=int(n_reh))

    def _decide_key(self, W, fm, S, k, ent, rng):
        """legato's frozen key, at every seam: nearest slot centroid in standardised seam-state
        space, then that slot's representative tape. No audition and no live option -- exactly what
        a keyed frozen library is."""
        n = len(S)
        slots = [(ns, j, sid) for kind, ns, j, sid in ent if kind == "slot"]
        if not slots:
            raise RuntimeError("keyed frozen library has no addressable slot at this seam "
                               "(the table ablation leaves a keyed arm with no execution path "
                               "at all -- which is the contrast, not a bug)")
        max_span = W.H_phrase
        cmds = np.full((n, max_span, W.AD), np.nan, np.float32)
        slot = np.zeros(n, int); nsg = np.ones(n, int)
        C = np.stack([self.library.cell(ns, k).key_c[j] for ns, j, _ in slots])
        mu, sd = self.norm
        Z = (np.asarray(S, np.float32) - mu) / sd
        Ck = (C - mu) / sd
        D = ((Z[:, None, :] - Ck[None, :, :]) ** 2).sum(-1)
        pick = D.argmin(1)
        for i in range(n):
            ns, j, sid = slots[int(pick[i])]
            c = self.library.cell(ns, k)
            cmds[i, :W.span(k, ns)] = c.tapes[int(c.rep[j])]
            slot[i], nsg[i] = sid, ns
            self.tape_calls += 1
        return dict(cmds=cmds, n_segs=nsg, slot=slot, src=np.zeros(n, int),
                    score=np.zeros(n, np.float32), n_cand=np.ones(n, int),
                    plans=0, delib=0, aud=0, k_shoot=0, n_prop=len(slots),
                    aud_delib=0, plan_delib=0, n_reh=0)

    # ------------------------------------------------------------------ Port 2 upkeep
    def check_parity(self, W, fm, ledger, who="agent"):
        """Held-out parity, per slot, RE-CHECKED EVERY CYCLE. On seam states the head has never
        been trained on (split by a deterministic code of the state, so a recurring posture cannot
        straddle the split), does the head's emission execute as well as the tape the audition would
        have picked -- scored under the same model, on the deployment distribution the head is about
        to be let loose on. Continuous values are logged beside the verdict so a different tau can be
        read off the record."""
        out = {}
        if self.span_head is None or self.sbuf is None:
            return out
        for sid, (S, C) in self.sbuf.hold.items():
            ns, k, j = self.layout.cell_of(sid)
            if ns == "prim" or len(S) < self.span_min_hold:
                out[int(sid)] = {"n": int(len(S)), "frac": None, "open": False}
                self.parity[int(sid)] = False
                continue
            span = W.span(k, ns)
            em = self.emit(fm, S, sid, span)
            pr = np.stack([np.arange(len(S)), np.arange(len(S))], 1)
            e_h = W.audition_fm(fm, S, em, pr, k, ns, horizon=self.aud_horizon)
            e_t = W.audition_fm(fm, S, C[:, :span, :], pr, k, ns, horizon=self.aud_horizon)
            frac = float(np.mean(e_h <= e_t + self.span_tol))
            self.parity[int(sid)] = bool(frac >= self.span_tau)
            self.parity_val[int(sid)] = frac
            out[int(sid)] = {"n": int(len(S)), "frac": frac,
                             "open": bool(frac >= self.span_tau),
                             "e_head": float(e_h.mean()), "e_tape": float(e_t.mean())}
            ledger.charge(who, 0, 0, W.dt_ctrl, W.d_fb, kind="parity",
                          aud=2 * len(S), delib=2 * len(S) * span)
        return out

    def span_terms(self, W, fm, batch, rng):
        """Port 2's self-imitation loss, added to the forward model's OWN optimizer step -- no extra
        steps for the treatment (`handle/`'s convention). `fm_trunk_t` carries gradient, so the loss
        reaches the plant. That is the treatment, and the plant guard is what says whether it cost
        anything."""
        import torch
        import torch.nn.functional as F
        if self.span_head is None or self.sbuf is None or not self.sbuf.buf:
            return None
        mu = torch.tensor(self.norm[0], device=self.device)
        sd = torch.tensor(self.norm[1], device=self.device)
        terms = []
        for sid, (S, C) in self.sbuf.buf.items():
            ns, k, j = self.layout.cell_of(sid)
            if ns == "prim" or len(S) < 8:
                continue
            span = W.span(k, ns)
            idx = rng.integers(0, len(S), size=min(int(batch), len(S)))
            s_t = torch.tensor(S[idx], device=self.device)
            tr = W.fm_trunk_t(fm, s_t)
            y = self.span_head((s_t - mu) / sd, tr,
                               torch.full((len(idx),), int(sid), dtype=torch.long,
                                          device=self.device))
            tgt = torch.tensor(C[idx][:, :span, :], device=self.device)
            terms.append(F.mse_loss(y[:, :span, :], tgt))
        return (sum(terms) / len(terms)) if terms else None
