"""The presto piece, as a pure-python generator and a set of constants -- and NOTHING ELSE in this
module, deliberately.

`offbook/piece.py` holds the donor piece's two constants for the same reason this file holds
presto's: a Modal LOCAL entrypoint executes on a machine with neither numpy nor mujoco (the
`arm_env.py` contract), and importing a sibling node's runner registers its `@app.local_entrypoint`s
on the shared `mjc.shared` app and collides ("Duplicate local entrypoint name: gates"). So the piece
lives in a module that imports only `math`.

WHY A NEW PIECE AT ALL. offbook's Round 4 (`../../offbook/`, run `d0`) taxed the reflex loop with an
observation delay and measured degradation ordered exactly by feedback consumption -- reactive
63.1x, live-per-segment 2.69x, chain 1.69x, segment tape 1.28x -- and still found no delay at which
a stored unit was the best PLAYABLE option: the piece is easy enough to steer by feel at small
delays and unplayable for everyone at large ones. The piece was inherited verbatim from `legato/`,
whose question was a comparison among committed arms, and it was never designed to make an
open-loop unit worth playing against a reactive incumbent that gets full state every 20 ms and
re-plans 61 times in 1.5 s. `presto` is the environment fix: the same plant family, a piece whose
TEMPO is set against a human-realistic reflex delay.

THE FIGURE. A closed, slightly irregular pentagon in tip space, entered by a mastered approach from
the reset tip `W0 = fk(q_center) = (0.1968, 0.7785)`:

    W0 --approach, H_app=14, ~0.29 m--> V0
       --drilled seg 0, H=6--> V1 --seg 1--> V2 --seg 2--> V3 --seg 3--> V4 --seg 4--> V0

so `goals = [V0, V1, V2, V3, V4, V0]`, `n_seg = 5`, `H_seg = 6,6,6,6,6`, phrase = 30 control steps.

WHAT EACH NUMBER IS FOR (every one of them is a measured constraint, not a taste):

  * `H_seg = 6` (120 ms at `dt_ctrl = 0.02 s`) against a human proprioceptive loop of ~100 ms
    (Delta = 5 control steps): closed-loop correction WITHIN a segment is physically impossible
    rather than forbidden. offbook Round 4 rejected a feedback CAP as necessity-by-fiat and taxed
    feedback through physics instead; presto keeps that choice and moves the tax into the tempo.
  * 5 segments -> a 30-step phrase, so the chain sits beyond the plant's composition horizon
    (offbook measured ~21 steps on the donor plant; gate G-H re-measures it here, because a faster,
    lower-damped plant need not have the same one) while each segment sits well under it. This is
    `legato` F4's precondition for a chunk paying at all, arranged so the crossover is INSIDE the
    action set rather than assumed.
  * The centre `(0.42, 0.42)` (radius 0.594 of a 1.10 m arm) puts every vertex at radius 0.46-0.76
    -- off the singular full-extension shell, in front of the base, and sweeping enough of the
    workspace that `M(q)` changes along the figure. It also leaves the approach at ~0.25-0.33 m over
    14 steps, i.e. the DONOR's approach tempo (0.300 m / 14), so the mastered lead-in is not a
    second variable.
  * The vertex angles are irregular (70/76/66/76/72 deg apart) so the five segments differ in leg
    length and turn, which is what keeps a keyed library non-trivial; the turns are 55-74 deg,
    against offbook's 86-104 deg, but they arrive every 120 ms instead of every 400 ms. THE
    DIFFICULTY IS THE TURN RATE, and it is smooth and global -- `ballistic/` 4b's precondition, and
    the reason the curl patch is OFF here (`curl_b = 0`): a localised needle is open-loop
    INCOMPENSABLE, so it saturates every ballistic arm at "fail" and measures nothing about the
    model. Momentum/braking is the open-loop-compensable axis and it is the only one presto uses.
  * `R` (the circumradius) is the SPEED knob: mean leg = 1.176 R, so mean tip speed is
    `1.176 R / (6 * 0.02)`. Gate G-T sweeps R x `joint_damping` and the design point is READ OFF
    that sweep, not assumed here. `DEF_R` below records what the sweep chose.

WHAT IS *NOT* CHANGED FROM THE DONOR, so the two nodes stay comparable: the arm (3 links,
0.4/0.4/0.3 m, 1.0/1.0/0.6 kg, `gear = 8`), `dt_ctrl = 0.02 s`, `q_center`, the start-posture draw,
the motor-noise levels (`sigma_practice` 0.15 / `sigma_perf` 0.06), the approach horizon, the
waypoint-weighted cost, and every line of `world.py`.
"""

import math

# ---- the arm's reset tip, W0 = fk(q_center) with q_center = (0.4, 1.1, 0.8) and
# link lengths (0.4, 0.4, 0.3). Hard-coded because this module may not import numpy; gate G-T
# asserts it against `World.tip` at run time.
W0 = (0.19683647, 0.77847690)

# ---- the figure's centre and its vertex angles (degrees, measured from the centre->W0 ray, so
# vertex 0 is always the one nearest the reset tip and the approach is the shortest entry).
DEF_CENTER = (0.42, 0.42)
DEF_ANGLES = (0.0, 70.0, 146.0, 212.0, 288.0)

# ---- the SPEED ladder gate G-T sweeps, and the plant ladder beside it.
R_LADDER = (0.09, 0.13, 0.17)
DAMPING_LADDER = (0.5, 0.15, 0.05)

# ---- the design point. Set from the G-T reduction; `DEF_R` is a placeholder until then and the
# runner always takes R and damping from its own flags, so nothing silently depends on this.
DEF_R = 0.13
DEF_DAMPING = 0.05
DEF_H_SEG = "6,6,6,6,6"
DEF_H_APP = 14
DEF_PATCH_SEG = 0        # unused: `curl_b = 0` turns the patch off entirely (see the header)


def pentagon(r, center=DEF_CENTER, angles=DEF_ANGLES, start=W0):
    """The figure's vertices, ordered so vertex 0 is the one nearest `start`.

    Pure python (this module imports only `math`). Returns `[(x, y), ...]`, one per vertex.
    """
    cx, cy = center
    phi = math.degrees(math.atan2(start[1] - cy, start[0] - cx))
    return [(cx + r * math.cos(math.radians(phi + d)),
             cy + r * math.sin(math.radians(phi + d))) for d in angles]


def waypoints(r, center=DEF_CENTER, angles=DEF_ANGLES, start=W0):
    """`world.World`'s `waypoints` string for circumradius `r`: the approach goal V0 followed by the
    five drilled goals V1..V4, V0 (the figure closes)."""
    v = pentagon(r, center, angles, start)
    return ";".join(f"{x:.6f},{y:.6f}" for x, y in (v + [v[0]]))


def legs(r, center=DEF_CENTER, angles=DEF_ANGLES, start=W0):
    """`[approach, seg0, ..., seg4]` leg lengths in metres -- the task-anchored scale the playability
    guard's half-leg term is read off."""
    v = pentagon(r, center, angles, start)
    pts = [start] + v + [v[0]]
    return [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def mean_leg(r, **kw):
    """Mean DRILLED leg length (the approach excluded). Half of this is the task-anchored term in
    the playability guard -- see `delay_gate.py`."""
    ls = legs(r, **kw)[1:]
    return sum(ls) / len(ls)


DEF_WPS = waypoints(DEF_R)
