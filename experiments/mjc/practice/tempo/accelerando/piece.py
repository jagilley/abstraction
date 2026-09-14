"""accelerando's pieces, as pure python. `prestissimo/piece.py` FORKED VERBATIM, plus the
FAST BODY and the TEMPO knob.

WHAT ACCELERANDO CHANGES, and nothing else (see FILES.md decisions 1-4):
  * `mass` is now a `Piece` field and reaches the plant through `dgp()`. It defaults to the
    donor's 1.0, so `donor_piece()` and `a1g_piece()` produce a BYTE-IDENTICAL MJCF and gates
    P-F1 / P-F2 stay exact. The body is two numbers — the velocity time constant `tau = m/c` and
    the acceleration scale `A = gear/m` — and mass is the ONE knob that moves both while leaving
    the donor's terminal speed `gear/c = 5 m/s` untouched.
  * `H` (steps per note) is the TEMPO knob and the era ladder. The waypoints, the band, the
    command-rotation region and the approach geometry are all functions of `R` alone, so the
    SAME PIECE is played at every tempo and only the note length changes. That is what a
    recording cannot survive and a program can.
  * `turn_command()` — the schedule's own inverse-dynamics demand at a tempo, in actuator units.
    An APPARATUS quantity computed by the experimenter at calibration time only (like the
    do-nothing floor and the band); no arm ever sees it, nothing learns it, and it is not a
    forward model of anything — it is the question "does this figure at this tempo fit inside
    |u| <= 1 on this body", answered with the body's own two constants.
  * `W_LISTEN` — the listener's clock, one slow note wide.

The original prestissimo docstring follows.

The pieces, as pure python — the DONOR square (étude's, verbatim) and the FAST figure.

Pure python (no numpy / mujoco / torch), so a Modal *local* entrypoint can import it: the local
client in these sessions has none of those installed (`offbook/FILES.md` Gotcha, carried through
`acappella/` and `solo/`).

TWO PIECES LIVE HERE, and that is the point.

  * `donor_piece()` is `etude/etude.py`'s closed square — four 34-step segments, the command
    rotation on segment 1, no approach leg. Every constant is asserted equal to the donor's own
    source by gate **P-F0**, which parses `mjc/practice/etude/etude.py` with `ast` rather than
    importing it (importing a sibling runner registers its `@app.function` / `@app.local_entrypoint`
    into the one shared Modal app — `offbook/FILES.md` Gotcha). Gate **P-F1** then runs this
    node's forked `world.py` on THIS piece and asserts it reproduces `acappella/b1`'s library
    build and its Δ = 0 and Δ = 8 rows at max|Δ| = 0. The fork is therefore exact and **the piece
    is the only variable**.

  * `fast_piece(R)` is the escalation. It is `accompanist/presto/piece.py`'s design applied to the
    pusher rather than the arm, and every number in it is a measured constraint or a carried
    donor constant, never a taste:

      - `H_SEG = 5` control steps = **120 ms** at the donor's `dt_ctrl = 0.024 s`. A human
        proprioceptive loop is ~100 ms, so closed-loop correction WITHIN a segment is physically
        impossible rather than forbidden — `offbook/` Round 4 rejected a feedback cap as
        necessity-by-fiat and taxed feedback through physics instead; presto moved the tax into
        the tempo and this node keeps that choice. 120 ms is also presto's own segment duration
        exactly (`H_seg = 6` at its `dt_ctrl = 0.02`), so the two fast pieces are matched in
        wall-clock even though the plants differ.
      - `K_DRILL = 8` drilled segments, so the dyadic ladder has **four rungs** (1, 2, 4, 8
        segments) and the top rung is the whole drilled figure. `s = 2` as in RHM.
      - `K_APP = 3` approach segments, played closed-loop by the primitive and excluded from the
        comparison. This is the whole reason a new piece exists: on étude's square the loop starts
        from rest at seam 0, so a committed arm decides there with the TRUE state at every Δ and
        is exactly delay-invariant (`acappella` finding 5). With three segments of lead-in the
        first drilled seam is 15 control steps (360 ms) into the traversal, deeper than the
        longest delay in the ladder (Δ = 16 = 384 ms is 15.4 steps at the *drilled* seam 0 read,
        i.e. clamped at the traversal start for that one seam only — see FILES.md P-A).
      - An **irregular octagon**: eight vertices at cumulative angles 0/38/90/131/186/225/275/318°
        with per-vertex radius factors spread 0.90–1.08, so the legs differ (≈1.3× spread) and the
        turns differ (38–55°). A regular polygon at constant speed is a translation wearing a
        piece's name and a keyed library over identical segments is trivially degenerate
        (presto decision 2). THE DIFFICULTY IS THE TURN RATE and it is smooth and global
        (`ballistic/` cut 4b): ~45° every 120 ms against `offbook/`'s ~90° every 400 ms, a 1.7×
        turn rate, on a plant whose velocity time constant is m/c = 0.5 s = 21 control steps —
        momentum-dominated by construction, so braking must be anticipated.
      - `R` (the circumradius) is the SPEED knob: mean leg ≈ 0.765 R, so mean tip speed is
        `0.765 R / (5 * 0.024)` = 6.4 R m/s. Gate **P-T** sweeps R × Δ and the design point is
        READ OFF that sweep under rules fixed before the grid is read (`legato` F2). `DEF_R`
        records what the sweep chose; the runner always takes R from its own flag, so nothing
        silently depends on this constant.
      - The **command-rotation region is KEPT**, scaled (σ = 0.2 × mean leg, φ = 1.2 rad,
        étude's), on drilled segment 3 — mid-figure, so it sits inside the second half of the
        level-2 unit at seam 2 and inside every level-3/4 span. presto turned its own patch OFF
        (`curl_b = 0`) because a localised FORCE needle is open-loop **incompensable** — even a
        perfect model cannot counteract a strong local kick feedforward. A command ROTATION is
        the opposite: it is exactly invertible, so a tape recorded through it carries the
        compensation and replaying it reproduces the motion on a deterministic plant. Keeping it
        preserves the model-free analogue of étude's `pretrain_mode=exclude` (the reflex's gains
        are fit on the world WITHOUT the region, so the hard passage is *unmodelled*, not wrongly
        modelled) — which is the one thing besides delay that gives committed content anything to
        hold. The turn rate remains the global difficulty, as presto requires.

WHAT IS NOT CHANGED, so the two pieces stay comparable: the plant (`arena_half` 1.8, `gear` 10,
`joint_damping` 2.0, `pusher_r` 0.12, `frame_skip` 12, `timestep` 0.002 → `dt_ctrl` 0.024 s), the
start draw (`START_JIT` 0.05, `V0_STD` 0.1), the motor-noise level (`EXPLORE_SIGMA` 0.25, practice
only), and the feedback price (`D_FB` 0.10 s). Every line of them is a donor constant that gate
P-F0 checks against `etude/etude.py`.
"""

import math

# --- the DONOR piece: étude's closed square, 4 segments of length 0.8 ------------------------
DEF_WAYPOINTS = "-0.4,-0.4; -0.4,0.4; 0.4,0.4; 0.4,-0.4; -0.4,-0.4"
DEF_REGIONS = "1:1.2:0.16"          # "<seg>:<phi>:<sigma>" — a command rotation on segment 1
DRILL_SEG = 1
SEG_H = 34

# --- the world (donor constants, shared by both pieces) -------------------------------------
FRAME_SKIP = 12
ARENA_HALF = 1.8
GEAR = 10.0
DAMPING = 2.0
PUSHER_R = 0.12
PUSHER_MASS = 1.0                    # the DONOR body's mass; accelerando's fast body changes THIS
BOX_HALF = 0.95
TIMESTEP = 0.002                    # étude's `dt_ctrl = fs * 0.002`

# --- the run-through geometry (donor constants) ---------------------------------------------
START_JIT = 0.05
V0_STD = 0.1
EXPLORE_SIGMA = 0.25                # motor variability, PRACTICE ONLY

# --- the time model -------------------------------------------------------------------------
D_FB = 0.10                         # one feedback / re-grounding event

DT_CTRL = FRAME_SKIP * TIMESTEP     # 0.024 s

# --- the FAST figure ------------------------------------------------------------------------
H_SEG = 5                           # 120 ms — presto's segment duration exactly
K_DRILL = 8                         # four dyadic rungs: 1, 2, 4, 8 segments
K_APP = 3                           # the approach: 15 steps = 360 ms of lead-in
N_LEVELS = 4                        # 2^(L-1) = 8 = K_DRILL

# vertex angles (cumulative degrees from vertex 0) and per-vertex radius factors. Gaps are
# 38/52/41/55/39/50/43/42 degrees (sum 360) and the radius factors spread 0.90-1.08, so leg
# lengths and turns both differ. Fixed once; never tuned against an outcome.
DEF_ANGLES = (0.0, 38.0, 90.0, 131.0, 186.0, 225.0, 275.0, 318.0)
DEF_RADII = (1.00, 0.92, 1.06, 0.95, 1.03, 0.90, 1.08, 0.97)
DEF_CENTER = (0.0, 0.0)

FAST_DRILL_SEG = 3                  # the command-rotation region's drilled segment (mid-figure)
FAST_PHI = 1.2                      # étude's rotation angle, unchanged
FAST_SIGMA_FRAC = 0.2               # sigma = 0.2 * mean leg  (étude: 0.16 / 0.8 = 0.2)

# the SPEED ladder gate P-T sweeps. mean leg ~ 0.765 R -> mean tip speed 6.4 R m/s; the
# actuator (10 m/s^2 at mass 1, gear 10, minus 2v of damping) saturates on a ~45 deg turn taken
# over one 120 ms segment at v ~ 1.2 m/s, i.e. R ~ 0.19, so the ladder brackets saturation.
R_LADDER = (0.08, 0.11, 0.14, 0.20, 0.28)
# NOTE, load-bearing: R sets the LINEAR speed only. The difficulty presto names — the TURN RATE —
# is set by `H_SEG` and `DEF_ANGLES` and is INVARIANT in R: ~45 deg every 120 ms at every rung of
# this ladder. Calibrating R down to keep the incumbent inside its own playability band therefore
# does not soften the piece's defining property; it only moves the figure's size.
DEF_R = 0.24                        # placeholder until P-T reduces; the runner takes R from its flag


def parse_waypoints(s: str):
    return [[float(v) for v in p.split(",")] for p in s.split(";") if p.strip()]


def parse_regions(s: str, wps):
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        seg, phi, sig = p.split(":")
        k = int(seg)
        c = [0.5 * (wps[k][0] + wps[k + 1][0]), 0.5 * (wps[k][1] + wps[k + 1][1])]
        out.append(dict(seg=k, center=c, sigma=float(sig), phi=float(phi)))
    return out


class Piece:
    """Everything `world.World` needs to know about what is being played.

    `wps` is the full waypoint list, `k0` the first DRILLED seam (segments `[0, k0)` are the
    approach), `H` the steps per segment. `K_seg = len(wps) - 1` counts every segment, approach
    included; `K_drill = K_seg - k0`. Drilled seam `d` is world seam `k0 + d`.
    """

    def __init__(self, name, wps, k0, H, regions, n_levels=1,
                 arena_half=ARENA_HALF, gear=GEAR, damping=DAMPING, pusher_r=PUSHER_R,
                 frame_skip=FRAME_SKIP, timestep=TIMESTEP, d_fb=D_FB,
                 start_jit=START_JIT, v0_std=V0_STD, explore_sigma=EXPLORE_SIGMA, meta=None,
                 mass=PUSHER_MASS):
        self.name = str(name)
        self.wps = [[float(a), float(b)] for a, b in wps]
        self.k0 = int(k0)
        self.H = int(H)
        self.regions = [dict(r) for r in regions]
        self.n_levels = int(n_levels)
        self.arena_half, self.gear, self.damping = float(arena_half), float(gear), float(damping)
        self.pusher_r = float(pusher_r)
        self.mass = float(mass)                 # ACCELERANDO: the body's one changed number
        self.frame_skip, self.timestep = int(frame_skip), float(timestep)
        self.d_fb = float(d_fb)
        self.start_jit, self.v0_std = float(start_jit), float(v0_std)
        self.explore_sigma = float(explore_sigma)
        self.meta = dict(meta or {})

    @property
    def K_seg(self):
        return len(self.wps) - 1

    @property
    def K_drill(self):
        return self.K_seg - self.k0

    @property
    def dt_ctrl(self):
        return self.frame_skip * self.timestep

    def legs(self, drilled_only=True):
        a = self.k0 if drilled_only else 0
        return [math.dist(self.wps[i], self.wps[i + 1]) for i in range(a, self.K_seg)]

    def mean_leg(self, drilled_only=True):
        ls = self.legs(drilled_only)
        return sum(ls) / len(ls)

    def turns(self):
        """Interior turn angle in degrees at each drilled seam (the heading change)."""
        out = []
        for i in range(self.k0, self.K_seg):
            a = self.wps[i]
            b = self.wps[i + 1]
            p = self.wps[i - 1]
            t1 = math.atan2(a[1] - p[1], a[0] - p[0])
            t2 = math.atan2(b[1] - a[1], b[0] - a[0])
            d = math.degrees(t2 - t1)
            out.append((d + 180.0) % 360.0 - 180.0)
        return out

    # ------------------------------------------------------------------ the body, in two numbers
    @property
    def tau(self):
        """The velocity time constant m/c: `a = (gear/m) u - (c/m) v`, so v relaxes as e^{-t/tau}.
        A stored unit can execute a note BLIND only if tau < T = H*dt_ctrl."""
        return self.mass / self.damping

    @property
    def accel(self):
        """The acceleration scale gear/m (m/s^2 at |u| = 1)."""
        return self.gear / self.mass

    @property
    def v_terminal(self):
        return self.gear / self.damping

    def note_s(self):
        return self.H * self.dt_ctrl

    def turn_command(self):
        """THE FEASIBILITY QUANTITY, in actuator units: what the SCHEDULE itself demands of the
        actuator at this tempo, at the sharpest turn.

        At a seam the schedule asks for a velocity change `dv = |v_out - v_in|` within one note
        (`a_turn = dv / T`) on top of holding speed against drag (`a_drag = |v_out| / tau`). The
        command that buys both is `u = (a_turn + a_drag) / A` with `A = gear/m`. `u > 1` means the
        figure at this tempo is outside the actuator's range and NO controller, stored or felt,
        can play it — the tempo axis would then be measuring infeasibility rather than the
        consumption of feedback.

        APPARATUS ONLY. Computed by the experimenter at calibration time from the body's own two
        constants, exactly as the do-nothing floor and the band are. No arm reads it, nothing is
        trained on it, and it predicts no state.
        """
        T = self.note_s()
        A, tau = self.accel, self.tau
        rows = []
        for i in range(self.k0, self.K_seg):
            p, a, b = self.wps[i - 1], self.wps[i], self.wps[i + 1]
            vi = ((a[0] - p[0]) / T, (a[1] - p[1]) / T)
            vo = ((b[0] - a[0]) / T, (b[1] - a[1]) / T)
            dv = math.hypot(vo[0] - vi[0], vo[1] - vi[1])
            sp = math.hypot(vo[0], vo[1])
            rows.append((dv / T + sp / tau) / A)
        return dict(u_turn_max=max(rows), u_turn_mean=sum(rows) / len(rows),
                    per_seam=[round(x, 4) for x in rows],
                    v_sched=self.mean_leg() / T, v_terminal=self.v_terminal,
                    tau=tau, note_s=T, tau_over_note=tau / T, accel=A)

    def band(self):
        """THE BAND — the pre-fixed, arm-neutral competence criterion (`solved`, and the
        playability guard), fixed before any arm's number exists.

        `max(½ × mean drilled leg, ref_play × mean_leg / 0.8)`. The first term is presto
        decision 8's task-anchored component: the waypoints are still resolved, i.e. the figure
        is recognisable. The second is étude's published `never` at performance tempo
        (`ref_play = 0.1066`, acappella's blind guard, 3-seed mean on the donor square whose mean
        leg is 0.8) ported scale-free to this figure. On the donor square the first term is 0.4
        against the second's 0.1066 and acappella declined it as vacuous; on a fast piece the
        same figure-relative generosity is what Jasper's note asks for — error at a fast tempo is
        partly a byproduct of chunking, and the listener's clock does not speed up with the
        notes. Both components are reported at every cell so either can be read off the record,
        and pass fractions at ¼ / ⅓ / ½ mean-leg are logged per arm per Δ so the ladder can be
        re-read at another band without re-running anything.
        """
        ml = self.mean_leg()
        return max(0.5 * ml, REF_PLAY * ml / 0.8)

    def band_parts(self):
        ml = self.mean_leg()
        return dict(mean_leg=ml, half_leg=0.5 * ml, ref_play_scaled=REF_PLAY * ml / 0.8,
                    band=self.band())

    def dgp(self, rot: bool):
        # `pusher_mass` is passed ALWAYS and defaults to the donor's 1.0, which is also
        # `pusher_env.DEFAULT_DGP`'s value, so the donor MJCF string is byte-identical and gates
        # P-F0 / P-F1 / P-F2 are untouched by this line.
        d = dict(arena_half=self.arena_half, gear=self.gear, joint_damping=self.damping,
                 pusher_r=self.pusher_r, pusher_mass=self.mass)
        if rot and self.regions:
            d["rot_regions"] = [dict(center=list(r["center"]), sigma=float(r["sigma"]),
                                     phi=float(r["phi"])) for r in self.regions]
        return d

    def describe(self):
        return dict(name=self.name, K_seg=self.K_seg, K_drill=self.K_drill, k0=self.k0,
                    H=self.H, n_levels=self.n_levels, dt_ctrl=self.dt_ctrl,
                    seg_ms=1000.0 * self.H * self.dt_ctrl,
                    legs=[round(x, 6) for x in self.legs()],
                    legs_all=[round(x, 6) for x in self.legs(False)],
                    turns=[round(x, 2) for x in self.turns()],
                    speed=self.mean_leg() / (self.H * self.dt_ctrl),
                    mass=self.mass, damping=self.damping, gear=self.gear,
                    tau=self.tau, accel=self.accel, v_terminal=self.v_terminal,
                    tau_over_note=self.tau / self.note_s(),
                    turn=self.turn_command(),
                    regions=self.regions, **self.band_parts(), meta=self.meta)


# étude's `never` at performance tempo, 3-seed mean — acappella's playability guard, published
# blind before any node in this arc existed. Used ONLY through `Piece.band()`'s scale-free port.
REF_PLAY = 0.1066


def donor_piece():
    """étude's square, verbatim: no approach leg, one level, H = 34."""
    wps = parse_waypoints(DEF_WAYPOINTS)
    return Piece("donor_square", wps, k0=0, H=SEG_H, regions=parse_regions(DEF_REGIONS, wps),
                 n_levels=1, meta=dict(source="etude/etude.py, verbatim"))


def fast_vertices(R, center=DEF_CENTER, angles=DEF_ANGLES, radii=DEF_RADII):
    cx, cy = center
    return [(cx + R * f * math.cos(math.radians(a)), cy + R * f * math.sin(math.radians(a)))
            for a, f in zip(angles, radii)]


def fast_piece(R=DEF_R, k_app=K_APP, h_seg=H_SEG, n_levels=N_LEVELS,
               drill_seg=FAST_DRILL_SEG, phi=FAST_PHI, sigma_frac=FAST_SIGMA_FRAC,
               with_region=True, damping=DAMPING, mass=PUSHER_MASS):
    """The fast figure: `k_app` approach segments into a closed irregular octagon.

    The approach enters V0 along the direction of the CLOSING leg (V7 → V0), at the drilled
    tempo, so the hand-over state at the first drilled seam resembles a mid-piece one rather
    than a launch from rest. That is the whole point of the lead-in (presto decision 3: the
    mastered lead-in must not be a second variable — here it is the same law, at the same tempo,
    for every arm).
    """
    v = fast_vertices(R)
    K = len(v)
    drilled = [list(v[i % K]) for i in range(K + 1)]          # V0..V7, V0 — the figure closes
    legs = [math.dist(drilled[i], drilled[i + 1]) for i in range(K)]
    ml = sum(legs) / len(legs)
    # the entry direction: the closing leg V7 -> V0, so the approach arrives "in tempo"
    dx, dy = drilled[0][0] - v[K - 1][0], drilled[0][1] - v[K - 1][1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    app = [[drilled[0][0] - (k_app - i) * ml * ux, drilled[0][1] - (k_app - i) * ml * uy]
           for i in range(k_app)]
    wps = app + drilled
    regions = []
    if with_region:
        ws = k_app + int(drill_seg)
        c = [0.5 * (wps[ws][0] + wps[ws + 1][0]), 0.5 * (wps[ws][1] + wps[ws + 1][1])]
        regions = [dict(seg=ws, center=c, sigma=sigma_frac * ml, phi=float(phi))]
    return Piece(f"fast_R{R:.3f}_c{damping:g}_m{mass:g}_H{h_seg}", wps, k0=int(k_app),
                 H=int(h_seg),
                 regions=regions, n_levels=int(n_levels), damping=float(damping),
                 mass=float(mass),
                 meta=dict(R=float(R), k_app=int(k_app), drill_seg=int(drill_seg),
                           damping=float(damping), mass=float(mass),
                           # THE RATIO THE DAMPING SWEEP IS ABOUT: a segment's duration against
                           # the plant's velocity time constant tau = m/c (mass 1). etude's slow
                           # square is 34*0.024/0.5 = 1.63; a 120 ms segment at c = 2 is 0.24, so
                           # an open-loop tape has no time to damp a hand-over error inside its
                           # own span and simply carries it.
                           seg_over_tau=float(h_seg) * DT_CTRL / (float(mass) / float(damping)),
                           v_terminal=GEAR / float(damping),
                           angles=list(DEF_ANGLES), radii=list(DEF_RADII)))


def piece_from_cfg(cfg):
    """The one place a config turns into a Piece — used identically by the runner and the gates."""
    if str(cfg.get("piece", "fast")) == "donor":
        return donor_piece()
    return fast_piece(R=float(cfg["R"]), k_app=int(cfg["k_app"]), h_seg=int(cfg["h_seg"]),
                      n_levels=int(cfg["n_levels"]),
                      with_region=bool(cfg.get("with_region", True)),
                      damping=float(cfg.get("damping", DAMPING)),
                      mass=float(cfg.get("mass", PUSHER_MASS)))


# =========================================================================== #
#                        WHAT ACCELERANDO ADDS                                #
# =========================================================================== #

# --- the FAST BODY --------------------------------------------------------------------------
# The body is two numbers: tau = m/c (how long velocity remembers a command) and A = gear/m (how
# hard it can push). prestissimo's plant has tau = 0.5 s = 21 control steps, which is why its
# band `tau < T < Delta` was EMPTY at every tempo and it had to inflate Delta to 384 ms to reach
# it (prestissimo flag 3, decision 18). Here the body is made fast enough that the band exists
# at a HUMAN delay, and it is made fast by changing ONE number:
#
#     mass 1.0 -> 0.06 kg, with gear = 10 and damping = 2.0 left at the DONOR's values.
#
#   tau        = m/c = 0.030 s  (30 ms; prestissimo 0.500 s)
#   A          = gear/m = 166.7 m/s^2  (prestissimo 10.0)
#   v_terminal = gear/c = 5.0 m/s  — UNCHANGED from the donor, because c and gear are unchanged.
#
# Why tau = 30 ms and not the top of the 30-50 ms range the brief allows: a stored unit can
# execute a note blind only if tau < T, and the fastest note on the dyadic ladder is
# H = 2 -> T = 48 ms. At tau = 50 ms that rung has tau > T and the band is empty again at exactly
# the tempo the node is about; at tau = 30 ms the fastest note is 1.6 tau and the slowest is
# 12.8 tau. The measured value is a GATE (P-TAU), fit from a step response, not assumed.
FAST_MASS = 0.06


def gain_grid(kp_grid, kd_grid, mass=FAST_MASS):
    """THE INCUMBENT'S GAIN GRID, ported to the body rather than carried as numbers.

    The reflex law is `u = kp (tgt - x) + kd (vtg - v)` and `u` drives an ACCELERATION `A u` with
    `A = gear/m`. Every closed-loop property the grid is supposed to span is a property of `A*kp`
    (the stiffness, hence the tracking bandwidth `~sqrt(A kp)` and the ramp error `v/(v_term kp)`)
    and `A*kd` (the velocity-loop gain), never of `kp` and `kd` alone. `A` is 16.7x larger on this
    body, so prestissimo's grid in raw numbers is prestissimo's grid MULTIPLIED BY 16.7 in every
    quantity that matters — and the consequence is not cosmetic:

        the delayed velocity loop is stable only for `A kd * Delta <~ 1`, i.e. `kd <~ 0.05` at
        Delta = 120 ms on this body, while prestissimo's grid FLOOR is kd = 0.25.

    Measured on `tsmoke1` (2026-09-05) with the smoke's coarse grid (kd floor 1.0): the incumbent
    scored 0.2425 at H = 16 against a 0.0534 band and a 0.3334 do-nothing floor — i.e. barely
    better than not moving, at a tempo where a 120 ms delay is a third of a note. The unscaled
    grid cannot express a stable controller on this body, so a run using it would measure the
    grid's floor and call it the delay.

    The port therefore multiplies both grids by `m / m_donor`, which holds `A*kp` and `A*kd` at
    prestissimo's own spans exactly (`A kp` 5 - 1600 s^-2, `A kd` 2.5 - 160 s^-1). It involves the
    two BODIES only — no piece, no tempo, no arm — so it cannot be moved by an outcome, and it is
    the identity on the donor body, which keeps P-F1 and P-F2 exact. Both the raw gains and
    `A*kp` / `A*kd` are reported at every cell.
    """
    f = float(mass) / PUSHER_MASS
    return ([round(kp * f, 12) for kp in kp_grid], [round(kd * f, 12) for kd in kd_grid])


def practice_sigma(mass=FAST_MASS, damping=DAMPING, dt=DT_CTRL, sigma=EXPLORE_SIGMA):
    """THE PRACTICE VARIABILITY, ported from the donor body rather than carried as a number.

    `EXPLORE_SIGMA = 0.25` is a donor constant (gate P-F0 reads it out of `etude/etude.py`) and
    it is a COMMAND noise. What practice variability actually is, though, is the spread of the
    body's realised motion, and on `v_{n+1} = a v_n + (1-a) v_term u_n` with `a = e^{-dt/tau}`
    i.i.d. command noise of size `sigma` makes a steady-state velocity spread

        sd(v) = sigma * v_terminal * sqrt((1 - a) / (1 + a)).

    `v_terminal = gear/c` is UNCHANGED between the donor body and this one, so matching the
    donor's realised velocity spread is exactly a rescaling of sigma by the two bodies' `a`:
    0.25 -> 0.0628 at tau = 0.03 s. Carrying 0.25 unchanged would instead make practice four
    times more variable than the donor's in realised motion, which on a figure whose legs are
    0.12 m would leave the rendition pool with nothing recognisable in it.

    The rule involves the two BODIES only — no piece, no tempo, no arm — so it cannot be moved
    by an outcome, and it returns EXACTLY the donor constant on the donor body (the ratio is
    computed from one expression evaluated twice), which is what keeps gates P-F1 and P-F2 exact.
    """
    if float(mass) == PUSHER_MASS and float(damping) == DAMPING:
        return float(sigma)

    def r(m, c):
        a = math.exp(-float(dt) * float(c) / float(m))
        return math.sqrt((1.0 - a) / (1.0 + a))

    return float(sigma) * r(PUSHER_MASS, DAMPING) / r(mass, damping)

# --- the TEMPO ladder (the era knob) --------------------------------------------------------
# Dyadic, so a level-(l+1) unit at tempo 2x has the same wall-clock span as a level-l unit at
# tempo 1x, and the listener's window is exactly one slow note.
# FIVE rungs, 768 / 384 / 192 / 96 / 48 ms. The slowest was added after the `tsmoke0` probe
# (2026-09-05) measured the incumbent OUT of the 1/2-leg band already at H = 8 under the fixed
# 120 ms delay: an era ladder whose slowest rung has no era 0 is prestissimo `a0`'s failure mode
# reproduced by design, so the ladder is extended downward until the incumbent is competent
# rather than the delay being shrunk. `tau/T` runs 0.039 -> 0.625 across it.
TEMPO_LADDER = (32, 16, 8, 4, 2)        # steps per note -> 768 / 384 / 192 / 96 / 48 ms

# --- the DELAY: a constant of the body, fixed before any arm was read -----------------------
# 5 control steps = 120 ms. A human proprioceptive loop is ~100-120 ms; prestissimo fixed its
# NOTE at 5 steps for the same reason and this node fixes its DELAY there instead, because a
# pianist's Delta is what does not change while T does. The band `tau < T < Delta` is then
# occupied by H = 4 (96 ms) and H = 2 (48 ms), and H = 8 / 16 are the rungs where feel still
# works. Held across all three phases; the Delta sweep survives only as an instrument.
DELTA_FIX = 5

# --- the LISTENER'S CLOCK -------------------------------------------------------------------
# One slow note wide: 16 control steps = 384 ms. See `world.listener_err` for the grade.
# 16 control steps = 384 ms. Fixed before any grid was read, as a constant of the ROOM and not
# of the score: it is the note length at H = 16, the rung where the two grades are most nearly
# the same instrument, and roughly the width of the auditory present. At H = 32 the window is
# half a note; at H = 2 it is eight.
W_LISTEN = 16

# --- the tempo-invariant piece and its speed ladder ------------------------------------------
A_K_APP = 3          # prestissimo's three approach notes, unchanged. k0*H > DELTA_FIX at EVERY
                     # tempo (6 > 5 at the fastest), so the read at drilled seam 0 is never
                     # clamped at the reset state — the aliasing prestissimo's diagnostic round
                     # located at k0*H = 15 < 16. A fourth note would buy nothing and cost 9% of
                     # every traversal on a five-rung ladder.
A_R_LADDER = (0.08, 0.11, 0.14, 0.16, 0.20)
# mean leg = 0.765 R, so the schedule's speed at tempo H is 0.765 R / (H * 0.024) m/s: at
# R = 0.16 that is 0.32 m/s at H = 16 and 2.55 m/s at H = 2, against v_terminal = 5.

# THE FEASIBILITY THRESHOLD, fixed before the calibration grid was read. `u_turn_max` is the
# schedule's own peak actuator demand at the FASTEST tempo (see `Piece.turn_command`); a piece
# selected above it would make the tempo axis measure the actuator's ceiling rather than the
# consumption of feedback. The estimate is conservative in two directions at once — it adds the
# turn and drag magnitudes instead of composing the vectors, and it reads |u|_2 against a limit
# that is actually |u|_inf <= 1 (a square, so up to sqrt(2) more on the diagonal) — so the
# operative question is only how much headroom to leave for CORRECTION on top of the schedule.
# 0.9 leaves 10%; at 1.0 the design point would be playable only by an open-loop optimum.
U_TURN_MAX = 0.9


def accel_piece(R, H, k_app=A_K_APP, n_levels=N_LEVELS, drill_seg=FAST_DRILL_SEG,
                phi=FAST_PHI, sigma_frac=FAST_SIGMA_FRAC, with_region=True,
                damping=DAMPING, mass=FAST_MASS):
    """THE PIECE, at a tempo. Geometry from `R` alone; `H` is the tempo and changes nothing else.

    `fast_piece` with accelerando's defaults, so the figure is prestissimo's irregular octagon
    unchanged — same angles, same radii, same command-rotation region, same construction of the
    lead-in. Keeping the geometry fixed is what makes this node's P-S readout comparable with
    prestissimo's 1.03x and with `d10`'s 1.25x: the body and the tempo move, the piece does not.
    """
    pc = fast_piece(R=R, k_app=k_app, h_seg=H, n_levels=n_levels, drill_seg=drill_seg,
                    phi=phi, sigma_frac=sigma_frac, with_region=with_region,
                    damping=damping, mass=mass)
    pc.explore_sigma = practice_sigma(mass=mass, damping=damping, dt=pc.dt_ctrl)
    pc.meta["explore_sigma"] = pc.explore_sigma
    pc.meta["explore_sigma_donor"] = EXPLORE_SIGMA
    return pc


def a1g_piece():
    """prestissimo `a1g`'s piece EXACTLY — R* = 0.20, k_app = 3, H = 5, damping 2, mass 1.

    Gate P-F2 builds this and asserts the fork reproduces `a1g`'s library-construction scores,
    its nesting identity and a pinned set of its sweep rows at max|delta| = 0, the way P-F1 does
    for `acappella/b1` on the donor square. It is the fork-fidelity reference on the FAST-PIECE
    code path, which the donor square does not exercise.
    """
    return fast_piece(R=0.20, k_app=3, h_seg=5, n_levels=4, damping=DAMPING, mass=PUSHER_MASS)
