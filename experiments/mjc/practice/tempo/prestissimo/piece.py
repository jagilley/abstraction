"""The pieces, as pure python — the DONOR square (étude's, verbatim) and the FAST figure.

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
                 start_jit=START_JIT, v0_std=V0_STD, explore_sigma=EXPLORE_SIGMA, meta=None):
        self.name = str(name)
        self.wps = [[float(a), float(b)] for a, b in wps]
        self.k0 = int(k0)
        self.H = int(H)
        self.regions = [dict(r) for r in regions]
        self.n_levels = int(n_levels)
        self.arena_half, self.gear, self.damping = float(arena_half), float(gear), float(damping)
        self.pusher_r = float(pusher_r)
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
        d = dict(arena_half=self.arena_half, gear=self.gear, joint_damping=self.damping,
                 pusher_r=self.pusher_r)
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
               with_region=True, damping=DAMPING):
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
    return Piece(f"fast_R{R:.3f}_c{damping:g}_H{h_seg}", wps, k0=int(k_app), H=int(h_seg),
                 regions=regions, n_levels=int(n_levels), damping=float(damping),
                 meta=dict(R=float(R), k_app=int(k_app), drill_seg=int(drill_seg),
                           damping=float(damping),
                           # THE RATIO THE DAMPING SWEEP IS ABOUT: a segment's duration against
                           # the plant's velocity time constant tau = m/c (mass 1). etude's slow
                           # square is 34*0.024/0.5 = 1.63; a 120 ms segment at c = 2 is 0.24, so
                           # an open-loop tape has no time to damp a hand-over error inside its
                           # own span and simply carries it.
                           seg_over_tau=float(h_seg) * DT_CTRL / (1.0 / float(damping)),
                           v_terminal=GEAR / float(damping),
                           angles=list(DEF_ANGLES), radii=list(DEF_RADII)))


def piece_from_cfg(cfg):
    """The one place a config turns into a Piece — used identically by the runner and the gates."""
    if str(cfg.get("piece", "fast")) == "donor":
        return donor_piece()
    return fast_piece(R=float(cfg["R"]), k_app=int(cfg["k_app"]), h_seg=int(cfg["h_seg"]),
                      n_levels=int(cfg["n_levels"]),
                      with_region=bool(cfg.get("with_region", True)),
                      damping=float(cfg.get("damping", DAMPING)))
