"""Planar pusher: a minimal contact-rich MuJoCo DGP with a real command channel.

A gravity-free 2D world (everything slides in the x-y plane): a force-actuated
`pusher` cylinder, a free `puck` cylinder, and four static walls. The command
`u in [-1,1]^2` is the motor force on the pusher's two slide joints. Contact
happens when the pusher touches the puck, or either touches a wall.

State  s (8-dim) = [qpos(4), qvel(4)]
       = [pusher_x, pusher_y, puck_x, puck_y,
          pusher_vx, pusher_vy, puck_vx, puck_vy]
Command u (2-dim) = motor force on (pusher_x, pusher_y).

Free-flight dynamics are near-linear (a damped particle under a known force),
so a forward model `f(s,u) -> Δs` predicts them almost perfectly; contact is a
stiff, near-discontinuous map from a coarse state, so that is where the FM
residual should concentrate. Ground-truth contact is read straight from
MuJoCo (`data.ncon` + `mj_contactForce`), the physics analog of RHM's known
latents — we never have to *infer* when a contact happened.

We deliberately expose the physics as DGP knobs (friction, masses, damping,
gear, arena size), exactly as `(v,s,L,m)` are RHM's knobs, so later cuts can
sweep them. `import mujoco` lives inside the functions so this module is
importable without MuJoCo installed (Modal submits the app from the laptop).
"""

import numpy as np

STATE_LABELS = [
    "pusher_x", "pusher_y", "puck_x", "puck_y",
    "pusher_vx", "pusher_vy", "puck_vx", "puck_vy",
]

# Default DGP (the "reference world"). All lengths in the arena's own units.
DEFAULT_DGP = dict(
    arena_half=1.0,      # inner half-width of the free region
    wall_thick=0.1,      # wall half-thickness
    pusher_r=0.12,       # pusher cylinder radius
    puck_r=0.12,         # puck cylinder radius
    pusher_mass=1.0,
    puck_mass=1.0,
    friction=0.6,        # sliding friction coefficient (tangential contacts)
    joint_damping=1.0,   # viscous drag on every slide joint (a "table" drag)
    gear=5.0,            # motor force scale: |force| <= gear * |ctrl|
    timestep=0.002,      # physics dt (500 Hz)
)


def build_xml(dgp: dict, with_puck: bool = True) -> str:
    """Build the MJCF for the planar pusher from a DGP knob dict.

    `with_puck=False` omits the puck body/joints entirely, leaving a puck-free
    world where the state is 4-dim [pusher_x, pusher_y, pusher_vx, pusher_vy] and
    the only contacts are pusher<->wall. Used by cut #3 (dynamics-shift reaching),
    where the task is momentum-dominated free-flight reaching on the clean smooth
    dynamics cut #2 nailed. Default `with_puck=True` is byte-for-byte identical to
    the original cut-#1/#2 XML (verified), so those experiments are unchanged.
    """
    d = {**DEFAULT_DGP, **dgp}
    H = d["arena_half"]
    t = d["wall_thick"]
    zh = 0.1  # cylinder / wall half-height in z (bodies overlap in z -> 2D contact)
    wall_c = H + t          # wall center offset from origin
    wall_len = H + 2 * t    # wall half-length (covers the corners)
    puck_block = f"""    <body name="puck" pos="0.5 0 0">
      <joint name="puck_x" axis="1 0 0"/>
      <joint name="puck_y" axis="0 1 0"/>
      <geom name="puck_geom" type="cylinder" size="{d['puck_r']} {zh}"
            mass="{d['puck_mass']}" rgba="0.90 0.45 0.20 1"/>
    </body>
""" if with_puck else ""
    return f"""
<mujoco model="planar_pusher">
  <option timestep="{d['timestep']}" integrator="RK4" gravity="0 0 0"/>
  <default>
    <joint type="slide" damping="{d['joint_damping']}" limited="false"/>
    <geom friction="{d['friction']} 0.005 0.0001"/>
  </default>
  <worldbody>
    <geom name="wall_n" type="box" pos="0 {wall_c} 0" size="{wall_len} {t} {zh}" rgba="0.55 0.55 0.6 1"/>
    <geom name="wall_s" type="box" pos="0 {-wall_c} 0" size="{wall_len} {t} {zh}" rgba="0.55 0.55 0.6 1"/>
    <geom name="wall_e" type="box" pos="{wall_c} 0 0" size="{t} {wall_len} {zh}" rgba="0.55 0.55 0.6 1"/>
    <geom name="wall_w" type="box" pos="{-wall_c} 0 0" size="{t} {wall_len} {zh}" rgba="0.55 0.55 0.6 1"/>
    <body name="pusher" pos="0 0 0">
      <joint name="pusher_x" axis="1 0 0"/>
      <joint name="pusher_y" axis="0 1 0"/>
      <geom name="pusher_geom" type="cylinder" size="{d['pusher_r']} {zh}"
            mass="{d['pusher_mass']}" rgba="0.20 0.50 0.90 1"/>
    </body>
{puck_block}    <!-- top-down camera for rendering (inert for physics/contacts) -->
    <camera name="topdown" pos="0 0 3.0" xyaxes="1 0 0 0 1 0"/>
    <light name="top" pos="0 0 3" dir="0 0 -1" directional="true"/>
  </worldbody>
  <actuator>
    <motor name="mx" joint="pusher_x" gear="{d['gear']}" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="my" joint="pusher_y" gear="{d['gear']}" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class PusherEnv:
    """Thin wrapper around a MuJoCo model with contact bookkeeping."""

    def __init__(self, dgp: dict | None = None, with_puck: bool = True):
        import mujoco

        self.dgp = {**DEFAULT_DGP, **(dgp or {})}
        self.with_puck = with_puck
        self.model = mujoco.MjModel.from_xml_string(
            build_xml(self.dgp, with_puck=with_puck))
        self.data = mujoco.MjData(self.model)
        self.puck_gid = (mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "puck_geom") if with_puck else -1)
        self.pusher_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "pusher_geom")
        self._f6 = np.zeros(6)
        # puck DOF / qpos indices (for the optional nonlinear force field applied via
        # qfrc_applied at runtime — see step()). Only meaningful with_puck.
        if with_puck:
            self.puck_qpos_idx = [int(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for jn in ("puck_x", "puck_y")]
            self.puck_dof_idx = [int(self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for jn in ("puck_x", "puck_y")]
        else:
            self.puck_qpos_idx = self.puck_dof_idx = []
        self.pusher_qpos_idx = [int(self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for jn in ("pusher_x", "pusher_y")]
        self.pusher_dof_idx = [int(self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for jn in ("pusher_x", "pusher_y")]

    def _field_force(self, pos, amp, phase, field: dict) -> np.ndarray:
        """A deterministic, nonlinear force from a MULTI-MODE cellular (Taylor-Green-
        like) flow + a mild central restoring term. The multi-mode design DECOUPLES
        the two properties we need, which a single mode couples through wavenumber k:
          * REDUCIBLE: every mode is spatially SMOOTH (bounded, moderate frequency),
            so with a fine control step there is no per-step integration aliasing and
            a big forward model fits the map s->Δs to R^2 -> ~1.
          * CAPACITY-HUNGRY: the field is a SUM of several modes (a complex function),
            so a SMALL forward model cannot approximate it even though a big one can.
            (#modes is the capacity-hunger lever; max-k is the aliasing lever.)

        Applied to the PUCK it is a value-IRRELEVANT distractor whose prediction costs
        real capacity (so dropping it is a genuine re-allocation, not a free lunch — a
        linear damped particle is cheap to predict no matter how energetic). Applied to
        the PUSHER (with a distinct `phase`, so the FM can't share ONE field computation
        across bodies) it makes the value-RELEVANT dynamics capacity-hungry too, so that
        capacity freed from the puck actually BUYS pusher fidelity.
        """
        x, y = pos
        c = field.get("central", 0.5)
        modes = field.get("modes", [(1.0, 0.0), (1.3, 0.7), (1.7, 1.3),
                                    (2.0, 2.0), (2.3, 2.6), (2.7, 3.1)])
        fx = fy = 0.0
        for k, ph in modes:
            fx += np.sin(k * x + ph + phase) * np.cos(k * y + ph + phase)
            fy += -np.cos(k * x + ph + phase) * np.sin(k * y + ph + phase)
        return np.array([amp * fx - c * x, amp * fy - c * y], dtype=np.float64)

    def _apply_fields(self, field: dict):
        """Set qfrc_applied from the field(s): always on the puck; optionally on the
        pusher (distinct phase) when `pusher_amp` > 0."""
        amp = field.get("amp", 1.3)
        self.data.qfrc_applied[self.puck_dof_idx] = self._field_force(
            self.data.qpos[self.puck_qpos_idx], amp, 0.0, field)
        pamp = field.get("pusher_amp", 0.0)
        if pamp > 0:
            self.data.qfrc_applied[self.pusher_dof_idx] = self._field_force(
                self.data.qpos[self.pusher_qpos_idx], pamp, field.get("pusher_phase", 1.57), field)

    def _apply_patch(self, patch: dict):
        """A spatially-LOCALIZED change to the PUSHER's dynamics: inside a Gaussian patch
        centered at `patch['center']` (width `sigma`), a strong constant force 'jet'
        `patch['force']` acts on the pusher (a localized 'current'), Gaussian-gated so it
        is smooth. d0 has no patch; when it appears in d1 the pusher's Δs INSIDE the patch
        is dominated by this force, so a stale forward model (no jet) mispredicts there
        and must RE-LEARN the jet from IN-PATCH transitions ONLY.

        This is a LOCAL, SCARCE dynamics change — unlike a GLOBAL shift, only in-patch
        transitions are informative about it. That is exactly the regime where a
        disagreement-directed drive that concentrates collection on the patch can re-adapt
        the forward model in FEWER transitions than undirected exploration (which under-
        samples the scarce patch). A weak drag tweak is too small (a stale FM predicts it
        fine); a strong force jet makes the local change genuinely load-bearing. Additive
        & off by default (dgp has no 'patch') -> Cuts #1-3 and value_shaping unchanged.
        Sets the pusher DOFs, so it is not meant to combine with a pusher force field.
        """
        pos = self.data.qpos[self.pusher_qpos_idx]
        cx, cy = patch["center"]
        sigma = patch.get("sigma", 0.3)
        w = float(np.exp(-((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2) / (2.0 * sigma ** 2)))
        fx, fy = patch.get("force", [0.0, 8.0])
        self.data.qfrc_applied[self.pusher_dof_idx] = w * np.array([fx, fy], dtype=np.float64)

    def _apply_actuator_rot(self, phi: float):
        """An INPUT-COUPLED dynamics change: the command's effect on the pusher is ROTATED
        by `phi`. The motors apply gear*ctrl along the axes; we add qfrc = gear*(R(phi)-I)@ctrl
        so the TOTAL effective actuation force is gear*R(phi)@ctrl. Unlike an additive force
        field (a constant output bias any model learns by nudging one bias term), rotating the
        command->motion map is INPUT-COUPLED: phi and phi+pi are OPPOSITE mappings, so a single
        init cannot serve both, and a model pooled over a symmetric phi range averages the
        command gain toward zero (arity-1 degenerate at range pi). That is the conflict that
        makes few-shot re-adaptation non-trivial (the phase-varying regime where a learned
        adaptable init can beat pooling). Command-only (constant across substeps). Additive &
        off by default (dgp has no 'push_rot') -> Cuts #1-3 and value_shaping unchanged."""
        c, s = float(np.cos(phi)), float(np.sin(phi))
        ux, uy = float(self.data.ctrl[0]), float(self.data.ctrl[1])
        rux, ruy = c * ux - s * uy, s * ux + c * uy      # R(phi) @ ctrl
        w = 1.0
        rp = self.dgp.get("rot_patch")                    # optional: gate rotation to a patch
        if rp is not None:                                # -> only IN-PATCH transitions carry phi
            pos = self.data.qpos[self.pusher_qpos_idx]
            cx, cy = rp["center"]; sig = rp.get("sigma", 0.3)
            w = float(np.exp(-((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2) / (2.0 * sig ** 2)))
        correction = w * self.dgp["gear"] * np.array(
            [rux - ux, ruy - uy], dtype=np.float64)       # (R - I) correction on top of motors
        # If a pusher force field is ALSO active this substep, `_apply_fields` already wrote
        # it to the pusher DOFs; a bare assignment here would CLOBBER it (rotation runs after
        # fields in step()). Recompute the field and SUM, so the value-RELEVANT pusher
        # dynamics (field) and the input-coupled conflict (rotation) coexist -- required by
        # meta_value_shaping's capacity-competition cut. Field-off path: base=0 -> byte-
        # identical to the original assign, so cuts #1-3 / meta_context / meta_adapt / the
        # rot_patch cut are unchanged.
        base = np.zeros(2, dtype=np.float64)
        field = self.dgp.get("puck_field")
        if field is not None and field.get("pusher_amp", 0.0) > 0:
            base = self._field_force(
                self.data.qpos[self.pusher_qpos_idx],
                field.get("pusher_amp", 0.0), field.get("pusher_phase", 1.57), field)
        self.data.qfrc_applied[self.pusher_dof_idx] = base + correction

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, rng: np.random.Generator):
        """Random non-overlapping placement of pusher & puck, zero velocity.

        Puck-free (with_puck=False): just place the pusher; qpos is 2-dim."""
        import mujoco

        if not self.with_puck:
            lo = -(self.dgp["arena_half"] - self.dgp["pusher_r"])
            hi = -lo
            self.data.qpos[:] = rng.uniform(lo, hi, size=2)
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return

        lo = -(self.dgp["arena_half"] - max(self.dgp["pusher_r"], self.dgp["puck_r"]))
        hi = -lo
        min_sep = self.dgp["pusher_r"] + self.dgp["puck_r"] + 0.05
        for _ in range(100):
            pusher = rng.uniform(lo, hi, size=2)
            puck = rng.uniform(lo, hi, size=2)
            if np.linalg.norm(pusher - puck) > min_sep:
                break
        self.data.qpos[:] = np.array([pusher[0], pusher[1], puck[0], puck[1]])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_state(self, qpos, qvel):
        """Force the sim to an arbitrary (qpos, qvel) — MuJoCo is memoryless, so
        this is a valid perfect-simulator query (used for command-sensitivity /
        counterfactual `u` sweeps at a fixed state)."""
        import mujoco

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0   # no stale field force leaks into a fresh query
        mujoco.mj_forward(self.model, self.data)

    def _contact_scan(self):
        """Aggregate contact forces at the current data state (one substep)."""
        import mujoco

        total = 0.0
        puck_f = 0.0
        puck_here = False
        nc = int(self.data.ncon)
        for i in range(nc):
            mujoco.mj_contactForce(self.model, self.data, i, self._f6)
            fmag = float(np.linalg.norm(self._f6[:3]))
            total += fmag
            c = self.data.contact[i]
            if c.geom1 == self.puck_gid or c.geom2 == self.puck_gid:
                puck_f += fmag
                puck_here = True
        return total, puck_f, puck_here, nc

    def step(self, ctrl: np.ndarray, n_sub: int) -> tuple[np.ndarray, dict]:
        """Apply `ctrl` for `n_sub` physics substeps; aggregate contact over them.

        Contact signals are the *peak* over the control interval: `force` is the
        largest total contact-force magnitude across substeps, `puck_contact` is
        whether any substep involved the puck. These are the ground-truth labels
        for the transition (s_t, u_t) -> s_{t+1}.
        """
        import mujoco

        self.data.ctrl[:] = np.clip(ctrl, -1.0, 1.0)
        field = self.dgp.get("puck_field")
        patch = self.dgp.get("patch")
        push_rot = self.dgp.get("push_rot")
        max_force = 0.0
        max_puck_force = 0.0
        ncon_at_max = 0
        any_contact = False
        any_puck = False
        for _ in range(n_sub):
            if field is not None:
                # re-evaluate the (state-dependent) field(s) each substep, then integrate
                self._apply_fields(field)
            if patch is not None:
                # localized force jet on the pusher (state-dependent, Gaussian-gated)
                self._apply_patch(patch)
            if push_rot is not None:
                # input-coupled conflict: rotate the command's effect by push_rot
                self._apply_actuator_rot(push_rot)
            mujoco.mj_step(self.model, self.data)
            total, puck_f, puck_here, nc = self._contact_scan()
            if nc > 0:
                any_contact = True
            if total > max_force:
                max_force = total
                ncon_at_max = nc
            max_puck_force = max(max_puck_force, puck_f)
            any_puck = any_puck or puck_here
        info = dict(force=max_force, puck_force=max_puck_force,
                    ncon=ncon_at_max, any_contact=any_contact, puck_contact=any_puck)
        return self.get_state(), info


def collect_transitions(dgp: dict | None, n_episodes: int, ep_len: int,
                        frame_skip: int, seed: int,
                        sigma: float = 0.5, theta: float = 0.15,
                        seek_gain: float = 0.4) -> dict:
    """Roll out a SCRIPTED behavior policy and log (s, u, s') + contact labels.

    The policy is an Ornstein-Uhlenbeck random walk in action space plus a mild
    drift toward the puck (`seek_gain`) so contacts actually happen — it is pure
    scripting, no learning, no RL. It only shapes *which* transitions we see; the
    dynamics being modeled are unaffected. We want a healthy mix of free and
    contact transitions (contact fraction is reported so defaults can be tuned).
    """
    env = PusherEnv(dgp)
    rng = np.random.default_rng(seed)

    S, U, S2 = [], [], []
    force, puck_force, ncon = [], [], []
    any_contact, puck_contact, ep_id = [], [], []

    for ep in range(n_episodes):
        env.reset(rng)
        a = np.zeros(2)
        for _ in range(ep_len):
            s = env.get_state()
            pusher_pos, puck_pos = s[0:2], s[2:4]
            delta = puck_pos - pusher_pos
            dist = np.linalg.norm(delta) + 1e-6
            direction = delta / dist
            # OU step toward zero + seek drift toward the puck.
            a = a - theta * a + sigma * rng.normal(size=2)
            ctrl = np.clip(a + seek_gain * direction, -1.0, 1.0)

            s2, info = env.step(ctrl, frame_skip)
            S.append(s)
            U.append(ctrl.astype(np.float32))
            S2.append(s2)
            force.append(info["force"])
            puck_force.append(info["puck_force"])
            ncon.append(info["ncon"])
            any_contact.append(info["any_contact"])
            puck_contact.append(info["puck_contact"])
            ep_id.append(ep)

    return dict(
        S=np.asarray(S, dtype=np.float32),
        U=np.asarray(U, dtype=np.float32),
        S2=np.asarray(S2, dtype=np.float32),
        force=np.asarray(force, dtype=np.float32),
        puck_force=np.asarray(puck_force, dtype=np.float32),
        ncon=np.asarray(ncon, dtype=np.int32),
        any_contact=np.asarray(any_contact, dtype=bool),
        puck_contact=np.asarray(puck_contact, dtype=bool),
        ep=np.asarray(ep_id, dtype=np.int32),
    )
