"""Planar N-link torque-driven arm: a MuJoCo DGP whose *nonlinearity is intrinsic*.

The second task family for the ballistic / two-timescale-value arc (parent substrate:
`pusher_env.py`). The pusher is a 2-DOF force-actuated point mass whose free-flight
dynamics are near-LINEAR — which is why every interesting property had to be bolted
on through the `qfrc_applied` perturbation layer (`puck_field`, `patch`, `push_rot`,
`field_patch`, `noise_patch`, `rot_regions`). That is six hand-designed perturbation
mechanisms carrying the scientific load, and the honest caveat on the whole ballistic
arc is "single-family (damping drift, corridor reach)".

An arm supplies, from the physics itself, the four things those layers were faking:

  * NONLINEAR, COUPLED dynamics.  M(q) q̈ + C(q,q̇) q̇ + D q̇ = τ.  The inertia matrix is
    configuration-dependent and the Coriolis/centrifugal term is quadratic in velocity.
    So a forward model f(s,u)→Δs faces a genuine CAPACITY frontier with no distractor
    body bolted on. (On the pusher a 232-parameter arity-2 FM already hit R²=0.999 —
    capacity never bound, so `value_shaping` had to manufacture competition with a
    multi-mode force field. Cut #4d's lesson "energy ≠ prediction cost; a reducible-but-
    LINEAR distractor is free to model" is exactly the pusher's limitation.)

  * DRIFT THAT IS LOCAL *AND* OPEN-LOOP COMPENSABLE, FOR FREE.  A tip `payload_mass`
    changes M(q) NON-UNIFORMLY over configuration space: the payload's contribution to
    the shoulder inertia scales with its distance from the base, r(q)² where
    r(q) = ‖FK(q)‖, so the stale-FM error is largest when the arm is EXTENDED and
    smallest when it is FOLDED. That is intrinsically the property cut #5 had to
    synthesize with `rot_regions` — local (so *where* you collect matters) yet smooth
    and feedforward-compensable (so ballistic control does not saturate at "fail" the
    way it does under a localized force JET, cut 4b's negative). Here it is physics.

  * TASK SPACE ≠ ACTUATION SPACE.  The goal is a Cartesian end-effector position; the
    command is joint torque. The kinematic map is a fixed known nonlinearity (FK) while
    the DYNAMICS is what drifts — so "what coordinates does the FM generalize in?"
    (joint vs task) becomes a physically answerable question about FM representation.
    With `n_links=3` reaching a 2-D goal the arm is REDUNDANT, and the null space is a
    value-irrelevant subspace DERIVED FROM THE TASK rather than bolted on as a puck —
    the strongest available version of the Cut #4d/#4e value-support experiment.

  * THE CANONICAL MOTOR-ADAPTATION PARADIGM.  Point-to-point planar reaching under a
    velocity-dependent `curl_field` is the Shadmehr & Mussa-Ivaldi force-field protocol.
    It buys a readout the pusher structurally cannot produce: AFTEREFFECTS (adapt, then
    remove the field → mirror-image errors), which is the gold-standard evidence that an
    *internal model* changed rather than a feedback gain being retuned.

Geometry: gravity-free and planar (`gravity="0 0 0"`, hinge axes +z) — i.e. a horizontal
arm on a table, which is both the pusher's convention and the actual geometry of the human
reaching experiments (arm supported against gravity). Contacts are OFF by default: cut #1
established contact is the stiff, near-discontinuous regime and cut 4b established such
regimes are open-loop-INCOMPENSABLE, so self-collision would only saturate ballistic
control. `contacts=True` re-enables them as a knob.

State  s (2n-dim) = [qpos(n), qvel(n)] = [q_1..q_n, q̇_1..q̇_n]   (joint angles, rad)
Command u (n-dim) = motor torque on each hinge, u ∈ [-1,1]^n, scaled by `gear`.
Goal            = a Cartesian tip position (2-dim), NOT part of the state.

Joints are unlimited (a joint limit is a stiff constraint = another discontinuity we do
not want). The operating region is instead bounded by the sampling distribution, which is
under our control because collection is teleport-based (`set_state`). `ArmEnv.wrapped()`
reports whether any |q| left the legible range so a run can flag it.

`import mujoco` lives inside the functions so this module is importable without MuJoCo
installed (Modal submits the app from the laptop) — same contract as `pusher_env.py`.
"""

import numpy as np

# Default DGP (the "reference arm"). SI-ish units: metres, kg, N·m, rad.
DEFAULT_DGP = dict(
    n_links=2,
    link_lengths=(0.5, 0.5),
    link_masses=(1.0, 1.0),
    link_radius=0.04,        # capsule radius (visual + inertia)
    payload_mass=0.0,        # point mass at the tip -- the flagship inertial drift knob
    payload_radius=0.06,
    joint_damping=0.5,       # viscous drag on every hinge
    gear=8.0,                # motor torque scale: |tau| <= gear * |ctrl|
    timestep=0.002,          # physics dt (500 Hz)
    contacts=False,          # self-collision off (contact is the incompensable regime)
    # ---- the PASSIVE TOOL: capacity competition from the BODY, not from a field ------- #
    n_passive=0,             # the last k joints carry no motor: a floppy tool / hanging load
    tool_mass=None,          # if set, the passive links' masses are rescaled to sum to this
    tool_damping=None,       # passive-joint damping; None = auto (see build_xml stability note)
    goal_site="tip",         # which point the TASK names: "tip" (chain end) or "hand"
)


def state_labels(n_links: int) -> list:
    return ([f"q{i+1}" for i in range(n_links)]
            + [f"qd{i+1}" for i in range(n_links)])


def build_xml(dgp: dict) -> str:
    """Build the MJCF for the planar arm from a DGP knob dict.

    A serial chain: link i is a capsule of length L_i along its own +x, hinged about +z
    at the distal end of link i-1. A `tip` site and the (possibly massless) `payload`
    sphere sit at the distal end of the last link.
    """
    d = {**DEFAULT_DGP, **dgp}
    n = int(d["n_links"])
    Ls = list(d["link_lengths"])[:n]
    Ms = list(d["link_masses"])[:n]
    assert len(Ls) == n and len(Ms) == n, "link_lengths/link_masses must have n_links entries"
    r = d["link_radius"]
    con = '' if d["contacts"] else ' contype="0" conaffinity="0"'

    # ---- the passive tool: the last `n_passive` joints carry NO motor -------------- #
    n_pas = int(d.get("n_passive", 0))
    n_act = n - n_pas
    assert 1 <= n_act <= n, "n_passive must leave at least one actuated joint"
    tm = d.get("tool_mass")
    if n_pas > 0 and tm is not None:
        cur = sum(Ms[n_act:]) or 1.0
        Ms = Ms[:n_act] + [m * (float(tm) / cur) for m in Ms[n_act:]]

    # ---- passive-joint damping, and the explicit-integration stability limit ---------- #
    # A LIGHT passive link is stiff to integrate: a rod of mass m and length L about its own
    # joint has I = m L^2 / 3, and an explicit integrator with viscous damping c needs
    # dt < 2 I / c. At tool_mass=0.02 kg that limit is ~0.001 s -- BELOW our 0.002 s
    # timestep -- so inheriting the arm's damping silently NaNs the simulation. (It did:
    # the first tool run produced finite-looking control numbers of 0.79, WORSE than doing
    # nothing, because the rollout had already gone non-finite.) So the default scales
    # damping to a safe fraction of that limit, and `step()` refuses to hide a NaN.
    tool_damp = d.get("tool_damping")
    if n_pas > 0 and tool_damp is None:
        I_min = min(Ms[i] * Ls[i] ** 2 / 3.0 for i in range(n_act, n))
        tool_damp = min(d["joint_damping"], 0.25 * 2.0 * I_min / d["timestep"])
    pas_damp = "" if n_pas == 0 else f' damping="{tool_damp}"'

    # nest the chain inside-out
    body = ""
    for i in range(n - 1, -1, -1):
        pos = "0 0 0" if i == 0 else f"{Ls[i-1]} 0 0"
        tail = ""
        if i == n - 1:
            tail = (f'      <site name="tip" pos="{Ls[i]} 0 0" size="0.02" rgba="0.95 0.35 0.15 1"/>\n'
                    f'      <geom name="payload" type="sphere" pos="{Ls[i]} 0 0" '
                    f'size="{d["payload_radius"]}" mass="{d["payload_mass"]}" '
                    f'rgba="0.85 0.25 0.25 1"{con}/>\n')
        # the HAND = the distal end of the last ACTUATED link. Emitted only when a passive
        # segment exists, so the n_passive=0 XML is byte-identical to the original arm.
        if n_pas > 0 and i == n_act - 1:
            tail = (f'      <site name="hand" pos="{Ls[i]} 0 0" size="0.02" '
                    f'rgba="0.20 0.80 0.40 1"/>\n') + tail
        col = "0.20 0.50 0.90 1" if i < n_act else "0.75 0.55 0.15 1"   # tool links in amber
        inner = "\n".join("  " + ln for ln in body.rstrip("\n").split("\n")) + "\n" if body else ""
        body = (f'    <body name="link{i+1}" pos="{pos}">\n'
                f'      <joint name="j{i+1}"{pas_damp if i >= n_act else ""}/>\n'
                f'      <geom name="g{i+1}" type="capsule" fromto="0 0 0 {Ls[i]} 0 0" '
                f'size="{r}" mass="{Ms[i]}" rgba="{col}"{con}/>\n'
                f'{tail}{inner}'
                f'    </body>\n')

    motors = "\n".join(
        f'    <motor name="m{i+1}" joint="j{i+1}" gear="{d["gear"]}" '
        f'ctrllimited="true" ctrlrange="-1 1"/>' for i in range(n_act))

    return f"""
<mujoco model="planar_arm">
  <option timestep="{d['timestep']}" integrator="RK4" gravity="0 0 0"/>
  <default>
    <joint type="hinge" axis="0 0 1" damping="{d['joint_damping']}" limited="false"/>
  </default>
  <worldbody>
    <geom name="base" type="cylinder" pos="0 0 0" size="0.06 0.02" rgba="0.4 0.4 0.45 1"{con}/>
{body}    <camera name="topdown" pos="0 0 3.0" xyaxes="1 0 0 0 1 0"/>
    <light name="top" pos="0 0 3" dir="0 0 -1" directional="true"/>
  </worldbody>
  <actuator>
{motors}
  </actuator>
</mujoco>
"""


def fk(q, link_lengths, upto: int | None = None) -> np.ndarray:
    """Analytic planar forward kinematics: joint angles -> Cartesian position of a point
    on the chain.

    Batched: q is (..., n). The chain is serial with RELATIVE joint angles, so the world
    orientation of link i is the cumulative sum of q_1..q_i. Verified against MuJoCo's
    own `site_xpos` in `arm_probe.py` (P0) -- it must match to ~1e-12 or the planner's
    cost function is measuring a different arm than the simulator.

    `upto=k` returns the distal end of link k rather than of the whole chain -- i.e. the
    HAND when a passive tool occupies the remaining links. Note the cumulative angles still
    run over the first k joints only, which is correct: links beyond k cannot move a point
    upstream of them.
    """
    q = np.asarray(q, dtype=np.float64)
    k = q.shape[-1] if upto is None else int(upto)
    L = np.asarray(link_lengths, dtype=np.float64)[:k]
    ang = np.cumsum(q[..., :k], axis=-1)
    return np.stack([(L * np.cos(ang)).sum(-1), (L * np.sin(ang)).sum(-1)], axis=-1)


class ArmEnv:
    """Thin wrapper around the MuJoCo arm, matching `PusherEnv`'s API surface.

    `get_state()` / `reset(rng)` / `set_state(qpos, qvel)` / `step(ctrl, n_sub)` are the
    contract every experiment script in this directory is written against; `set_state` is
    the load-bearing one (teleport-based collection + interventional counterfactual `u`
    sweeps at a fixed state, valid because MuJoCo is memoryless).
    """

    def __init__(self, dgp: dict | None = None):
        import mujoco

        self.dgp = {**DEFAULT_DGP, **(dgp or {})}
        self.n = int(self.dgp["n_links"])
        self.link_lengths = np.asarray(
            list(self.dgp["link_lengths"])[: self.n], dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.dgp))
        self.data = mujoco.MjData(self.model)
        self.n_passive = int(self.dgp.get("n_passive", 0))
        self.n_act = self.n - self.n_passive
        self.state_dim = 2 * self.n
        self.act_dim = self.n_act            # the passive tool joints carry no motor
        self.tip_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        # Which point the TASK names. `hand` (upstream of the tool) makes the tool's state
        # value-IRRELEVANT; `tip` makes it value-CRITICAL -- and switching between them
        # changes the VALUE ONLY, leaving the transition operator byte-identical. That is
        # the controlled intervention the pusher could never run: there, making the puck
        # value-relevant meant changing the task, which also changed its difficulty.
        self.goal_site = self.dgp.get("goal_site", "tip")
        if self.goal_site == "hand" and self.n_passive == 0:
            self.goal_site = "tip"           # no tool -> hand and tip coincide
        self.goal_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.goal_site)
        self.goal_link = self.n if self.goal_site == "tip" else self.n_act
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._max_absq = 0.0
        self._nonfinite = 0
        self._noise_rng = np.random.default_rng(int(self.dgp.get("noise_seed", 0)))

    # ---------------- optional runtime perturbations (additive, off by default) ------- #

    def _apply_curl_field(self, curl: dict):
        """A velocity-dependent force field at the END EFFECTOR -- the Shadmehr &
        Mussa-Ivaldi force-field adaptation protocol, applied exactly:

            F_tip = b * [[0, -1], [1, 0]] @ v_tip        (a CURL / rotational viscous field)
            qfrc += J(q)^T F_tip

        Why this is the right shape for this program (it satisfies the whole rulebook the
        pusher arc had to discover the hard way):
          * SMOOTH and GLOBAL in the state, so it is open-loop COMPENSABLE -- a correct FM
            lets a feedforward controller pre-compensate, unlike a localized force jet.
          * VELOCITY-dependent, so it is invisible at rest and can only be learned from
            transitions that actually MOVE -- collection must be on-manifold.
          * It perturbs the DYNAMICS, not the kinematics, so FK stays valid and the drift
            is purely in the factor a reward-free FM can re-fit.
          * It is the paradigm the AFTEREFFECT readout is defined on: adapt to b, set b=0,
            and a controller running an un-re-adapted internal model produces mirror-image
            trajectory errors -- direct behavioural evidence the internal model changed.
        `mj_jacSite` gives the exact analytic Jacobian, so no approximation enters.
        """
        import mujoco

        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.tip_sid)
        J = self._jacp[:2, :]                                  # (2, nv)
        v_tip = J @ self.data.qvel                             # (2,)
        b = float(curl.get("b", 0.0))
        F = b * np.array([-v_tip[1], v_tip[0]], dtype=np.float64)
        self.data.qfrc_applied[:] += J.T @ F

    def _apply_torque_rot(self, phi: float):
        """Rotate the COMMAND->torque map by `phi` in the 2-D command plane (the arm's
        analog of `pusher_env`'s `push_rot`): total effective torque = gear * R(phi) @ u.
        Input-coupled, so phi and phi+pi are opposite mappings and a pooled model averages
        the command gain toward zero -- the conflict shape Cut #4 found is required to open
        a meta-learning gap. Only defined for n_links == 2. Command-only (constant across
        substeps)."""
        assert self.n == 2, "torque_rot is defined for the 2-link arm"
        c, s = float(np.cos(phi)), float(np.sin(phi))
        ux, uy = float(self.data.ctrl[0]), float(self.data.ctrl[1])
        rux, ruy = c * ux - s * uy, s * ux + c * uy
        self.data.qfrc_applied[:2] += self.dgp["gear"] * np.array(
            [rux - ux, ruy - uy], dtype=np.float64)

    def _apply_joint_noise(self, amp: float):
        """Aleatoric (irreducible) joint torque noise -- the noisy-TV control. Resampled
        every substep, so repeated visits to the SAME state give DIFFERENT Δs."""
        self.data.qfrc_applied[:] += amp * self._noise_rng.standard_normal(self.model.nv)

    # ---------------------------------- API ----------------------------------------- #

    def tip_pos(self) -> np.ndarray:
        """Ground-truth Cartesian position of the chain END from MuJoCo itself (x, y)."""
        return np.asarray(self.data.site_xpos[self.tip_sid][:2], dtype=np.float64).copy()

    def goal_pos(self) -> np.ndarray:
        """Ground-truth Cartesian position of the point the TASK names (`goal_site`)."""
        return np.asarray(self.data.site_xpos[self.goal_sid][:2], dtype=np.float64).copy()

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, rng: np.random.Generator, q_scale: float = 1.0):
        import mujoco

        self.data.qpos[:] = rng.uniform(-q_scale, q_scale, size=self.n)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_state(self, qpos, qvel):
        """Force the sim to an arbitrary (qpos, qvel). MuJoCo is memoryless, so this is a
        valid perfect-simulator query. Clears ctrl and any stale applied force so no
        perturbation leaks into a fresh query (the `PusherEnv.set_state` contract)."""
        import mujoco

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray, n_sub: int) -> tuple[np.ndarray, dict]:
        """Apply `ctrl` for `n_sub` physics substeps.

        Perturbations ACCUMULATE into a freshly-zeroed `qfrc_applied` each substep, so
        several can coexist without the single-writer clobber bug that bit
        `_apply_actuator_rot`/`_apply_fields` in `pusher_env.py` (Cut #4d gotcha (b)).
        When no perturbation key is present nothing is written and the path is the plain
        rigid-body arm.
        """
        import mujoco

        self.data.ctrl[:] = np.clip(ctrl, -1.0, 1.0)
        curl = self.dgp.get("curl_field")
        trot = self.dgp.get("torque_rot")
        jnoise = self.dgp.get("joint_noise")
        for _ in range(n_sub):
            if curl is not None or trot is not None or jnoise is not None:
                self.data.qfrc_applied[:] = 0.0
                if curl is not None:
                    self._apply_curl_field(curl)
                if trot is not None:
                    self._apply_torque_rot(float(trot))
                if jnoise is not None:
                    self._apply_joint_noise(float(jnoise))
            mujoco.mj_step(self.model, self.data)
        s = self.get_state()
        # Never let a diverged sim pass as data. A light passive link can violate the
        # explicit-integration stability limit (see build_xml), and MuJoCo only prints a
        # warning -- the NaN then propagates into transitions, FM training and control
        # numbers that still LOOK finite. Count it loudly instead.
        if not np.all(np.isfinite(s)):
            self._nonfinite += 1
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self._max_absq = max(self._max_absq, float(np.abs(self.data.qpos).max()))
        return s, dict(tip=self.tip_pos(), max_absq=self._max_absq,
                       nonfinite=self._nonfinite)

    def nonfinite(self) -> int:
        """How many `step()` calls produced a non-finite state (a diverged simulation)."""
        return self._nonfinite

    def reset_wrap(self):
        """Reset the joint-excursion tracker. Call before a phase you want measured on its
        own -- a deliberately-random-action floor rollout WILL spin the arm, so a lifetime
        max would flag every run and tell you nothing about the controlled phases."""
        self._max_absq = 0.0

    def max_absq(self) -> float:
        return self._max_absq

    def wrapped(self, limit: float = 3.0) -> bool:
        """Has any joint angle left the legible (non-wrapping) range since the last
        `reset_wrap()`? Joints are unlimited by design, so this is the diagnostic that the
        OPERATING REGION -- not a stiff constraint -- is what keeps the state space
        bounded."""
        return self._max_absq > limit


def collect_pool(env: "ArmEnv", n: int, rng: np.random.Generator, frame_skip: int,
                 q_center, q_range: float, v_explore: float) -> tuple:
    """Teleport-based transition collection -- the established modern idiom in this
    directory (cuts #3 onward), which isolates the dynamics being modelled from any
    navigation/coverage confound: sample a configuration and velocity directly, teleport
    there, apply a uniform random command, record (s, u, s').
    """
    n_dof, n_u = env.n, env.act_dim
    qc = np.asarray(q_center, dtype=np.float64)[:n_dof]
    S = np.empty((n, 2 * n_dof), np.float32)
    U = np.empty((n, n_u), np.float32)
    S2 = np.empty((n, 2 * n_dof), np.float32)
    for i in range(n):
        q = qc + rng.uniform(-q_range, q_range, n_dof)
        qd = rng.normal(0.0, v_explore, n_dof)
        env.set_state(q, qd)
        u = rng.uniform(-1, 1, n_u).astype(np.float32)
        s = env.get_state()
        s2, _ = env.step(u, frame_skip)
        S[i] = s
        U[i] = u
        S2[i] = s2
    return S, U, S2
