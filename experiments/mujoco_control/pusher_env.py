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
        max_force = 0.0
        max_puck_force = 0.0
        ncon_at_max = 0
        any_contact = False
        any_puck = False
        for _ in range(n_sub):
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
