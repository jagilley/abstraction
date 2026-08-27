"""The piece and the world, as constants — `etude/etude.py`'s, verbatim.

Pure python (no numpy / mujoco / torch), so a Modal *local* entrypoint can import it: the local
client in these sessions has none of those installed (`offbook/FILES.md` Gotcha).

Every value here is asserted equal to the donor's own source by gate **A-F1**, which parses
`mjc/practice/etude/etude.py` with `ast` rather than importing it (importing a sibling runner
registers its `@app.function` / `@app.local_entrypoint` into the one shared Modal app —
`offbook/FILES.md` Gotcha). If the donor ever moves, the gate fails rather than the fork silently
drifting.
"""

# --- the piece: a closed square loop, 4 segments of length 0.8 -------------------------------
DEF_WAYPOINTS = "-0.4,-0.4; -0.4,0.4; 0.4,0.4; 0.4,-0.4; -0.4,-0.4"
DEF_REGIONS = "1:1.2:0.16"          # "<seg>:<phi>:<sigma>" — a command rotation on segment 1
DRILL_SEG = 1
SEG_H = 34

# --- the world ------------------------------------------------------------------------------
FRAME_SKIP = 12
ARENA_HALF = 1.8
GEAR = 10.0
DAMPING = 2.0
PUSHER_R = 0.12
BOX_HALF = 0.95
TIMESTEP = 0.002                    # etude's `dt_ctrl = fs * 0.002`

# --- the run-through geometry ---------------------------------------------------------------
START_JIT = 0.05
V0_STD = 0.1
EXPLORE_SIGMA = 0.25                # motor variability, PRACTICE ONLY

# --- the time model -------------------------------------------------------------------------
D_FB = 0.10                         # one feedback / re-grounding event


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


WPS = parse_waypoints(DEF_WAYPOINTS)
REGIONS = parse_regions(DEF_REGIONS, WPS)
K_SEG = len(WPS) - 1
DT_CTRL = FRAME_SKIP * TIMESTEP     # 0.024 s


def dgp(rot: bool):
    d = dict(arena_half=ARENA_HALF, gear=GEAR, joint_damping=DAMPING, pusher_r=PUSHER_R)
    if rot:
        d["rot_regions"] = [dict(center=r["center"], sigma=r["sigma"], phi=r["phi"])
                            for r in REGIONS]
    return d
