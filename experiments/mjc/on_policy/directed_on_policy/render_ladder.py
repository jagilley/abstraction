"""E3 render -- watch the on-policy arm reach, EARLY vs LATE, with the forward model's own
forecast drawn against what the body actually does.

WHY THIS EXISTS. Every number in [`README.md`](README.md) is a scalar over rounds. But the
mechanism the whole cut is about -- a *ballistic* (open-loop) plan is only as good as the model
it was planned through -- is a spatial, temporal thing: the arm commits to 14 torques and then
lives with them. So the informative picture is not "the arm reaches"; it is the arm reaching
while a GHOST of the trajectory its model predicted hangs in the same frame. Early, under a
stale FM freshly kicked by the drift, the ghost peels away from the body -- you can see the
curl field bending the real arm off its own forecast, and the reach lands wide. Late, after
`value = lprog x visits` has spent 30 rounds of scarce budget keeping region A calibrated, ghost
and body superimpose and the tip lands on the dot.

WHAT IS DRAWN (top-down, the arm's actual plane -- it is gravity-free and planar by design):
  * the arm                      -- MuJoCo's own render
  * translucent discs            -- the drift regions. BLUE = reducible curl (learnable),
                                    RED = aleatoric noise (the noisy-TV trap, unlearnable).
                                    Region A (the on-reach target, the only one that pays) is
                                    ringed. Radius = `region_k * sigma`, the in-region test.
  * pale ghost spheres           -- the tip path the FM PREDICTED when it made this plan. Fixed
                                    at plan time; it does not move, because a ballistic plan is
                                    a commitment.
  * bright trail                 -- where the tip ACTUALLY goes.
  * green dot                    -- the goal.

NO MODEL RUNS HERE. `directed_on_policy.py --render-rounds ...` records the plans, the forecast,
and the resolved plant into `render_pack.json`; this script is a pure replay. Frames are stepped
one PHYSICS substep at a time (`step(u, 1)` x frame_skip == `step(u, frame_skip)`: same field
application, same noise draws) purely to get smooth motion out of a 14-step plan.

Headless GL is OSMesa, and the Modal image is imported verbatim from `../../render_video.py` so
the two share a build cache.

Run:
    cd experiments/            # MODAL_PROFILE=chromatic
    modal run mjc/on_policy/directed_on_policy/render_ladder.py::smoke --tag render_v1
    modal run mjc/on_policy/directed_on_policy/render_ladder.py::video --tag render_v1
"""

import os
import json
import modal

from mjc.shared import volume, DATA_DIR
from mjc.render_video import render_image          # shared image => shared build cache

app = modal.App("mujoco-e3-render", image=render_image)

# ---- canvas geometry (final frame is 1280x720; both divisible by 16 for the encoder) ------ #
SIM = 640                      # each sim panel is rendered square (declared in `_render_model`)
FIG_W, FIG_H, DPI = 1280, 720, 80

# ---- palette ------------------------------------------------------------------------------ #
C_BG = "#0e1014"
C_FG = "#e8e6e1"
C_DIM = "#8b8f98"
C_CURL = "#4c8dd6"             # reducible curl region
C_NOISE = "#d1603d"            # irreducible noise region
C_ACTUAL = "#f2a541"           # where the body actually went
C_PRED = "#9fb8d4"             # where the model thought it would go
C_GOAL = "#5fbf7f"


def _pack_dir(tag: str) -> str:
    return os.path.join(DATA_DIR, "directed_on_policy", tag)


def _load(tag: str):
    with open(os.path.join(_pack_dir(tag), "render_pack.json")) as fh:
        rp = json.load(fh)
    res = None
    rpath = os.path.join(_pack_dir(tag), "results.json")
    if os.path.exists(rpath):
        with open(rpath) as fh:
            res = json.load(fh)
    return rp["meta"], rp["packs"], res


def _round_label(t: int) -> str:
    return "stale model (pre-adaptation)" if t < 0 else f"round {t}"


# =============================================================================== #
# scene decoration -- extra geoms appended to MuJoCo's own scene after update_scene
# =============================================================================== #
def _add(scene, gtype, size, pos, rgba):
    import numpy as np
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=int(gtype),
        size=np.asarray(size, np.float64),
        pos=np.asarray(pos, np.float64),
        mat=np.eye(3).flatten().astype(np.float64),
        rgba=np.asarray(rgba, np.float32),
    )
    scene.ngeom += 1


def _ring(scene, gtype, c, rad, rgba, nseg=44, size=0.008, z=-0.05, dash=False):
    """A circle outline, drawn as a necklace of small spheres (MuJoCo has no annulus). Reads far
    better than a filled disc, which at these radii swamps the arm it is supposed to annotate."""
    import numpy as np

    for i in range(nseg):
        if dash and i % 2:
            continue
        a = 2.0 * np.pi * i / nseg
        _add(scene, gtype, [size, 0, 0],
             [c[0] + rad * np.cos(a), c[1] + rad * np.sin(a), z], rgba)


def _decorate(scene, meta, pack, b, trail, cfg):
    """Regions + goal + the FM's forecast + the live tip trail, for reach `b`."""
    import numpy as np
    import mujoco

    SPH = mujoco.mjtGeom.mjGEOM_SPHERE
    CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
    k = meta["region_k"]

    # drift regions: a faint disc UNDER the arm (capsules span z in [-.04,.04]) plus a ring, so
    # the region is legible without competing with the trajectory it exists to annotate.
    for j, r in enumerate(meta["regions"]):
        rad = k * r["sigma"]
        c = r["center"]
        curl = r["kind"] == "curl"
        col = (0.29, 0.55, 0.86) if curl else (0.85, 0.36, 0.22)
        # reducible regions are shaded by their CURRENT drift gain -- the thing that keeps moving
        # and keeps the allocation problem zero-sum. Noise regions are permanent, so: full.
        gain = abs(pack["b_state"][j]) / max(cfg["b_hi"], 1e-9) if curl else 1.0
        _add(scene, CYL, [rad, rad, 0.004], [c[0], c[1], -0.07],
             (*col, (0.05 + 0.13 * min(gain, 1.0)) if curl else 0.11))
        # SOLID ring = reducible (there is structure to learn); DASHED = irreducible noise. That
        # is the whole ladder in one visual contrast.
        if curl:
            _ring(scene, SPH, c, rad, (*col, 0.85), nseg=76, size=0.012)
        else:
            _ring(scene, SPH, c, rad, (*col, 0.85), nseg=30, size=0.013, dash=True)
        if r["name"] == "A":                       # the one region that pays: a bright outer ring
            _ring(scene, SPH, c, rad * 1.17, (0.97, 0.97, 0.97, 0.55), nseg=92, size=0.009)

    # the FM's forecast for this plan: fixed at plan time, drawn in full from frame 0, because a
    # ballistic plan is a commitment made before the first step
    # (z above the arm capsules, which span [-0.04, 0.04], so neither overlay is occluded)
    # The forecast is a FAT PALE TUBE and the actual path a THIN BRIGHT THREAD drawn on top, so
    # where the model was right the thread runs down the middle of the tube, and where it was
    # wrong the thread leaves the tube. That relationship is the entire point of the figure, and
    # it only reads if the two marks differ in weight as well as colour.
    for p in pack["pred_tips"][b]:
        _add(scene, SPH, [0.034, 0, 0], [p[0], p[1], 0.09], (0.80, 0.85, 0.97, 0.42))

    # what the body actually did, up to now
    for p in trail:
        _add(scene, SPH, [0.013, 0, 0], [p[0], p[1], 0.17], (0.99, 0.72, 0.25, 0.98))

    # the miss, drawn as the thing it is: the gap between where the body stopped and the goal
    g = pack["goals"][b]
    if len(trail) > 1:
        end = trail[-1]
        for f in np.linspace(0.0, 1.0, 16):
            _add(scene, SPH, [0.007, 0, 0],
                 [end[0] + f * (g[0] - end[0]), end[1] + f * (g[1] - end[1]), 0.12],
                 (0.85, 0.87, 0.90, 0.40))
    _ring(scene, SPH, g, 0.055, (0.37, 0.80, 0.53, 0.90), nseg=30, size=0.008, z=0.11)
    _add(scene, SPH, [0.026, 0, 0], [g[0], g[1], 0.11], (0.37, 0.80, 0.53, 0.98))


def _framing(meta, packs, zoom=1.0):
    """Half-height (in metres) of a top-down view containing the base, every region and every
    rendered trajectory."""
    import numpy as np

    pts = [np.zeros(2)]
    for r in meta["regions"]:
        rad = meta["region_k"] * r["sigma"] * 1.2
        c = np.asarray(r["center"])
        pts += [c + rad, c - rad]
    for pk in packs:
        pts += [np.asarray(g) for g in pk["goals"]]
        pts += [np.asarray(t) for reach in pk["actual_tips"] for t in reach]
        pts += [np.asarray(t) for reach in pk["pred_tips"] for t in reach]
    P = np.stack(pts)
    lo, hi = P.min(0), P.max(0)
    ctr = 0.5 * (lo + hi)
    half = (0.5 * float(np.max(hi - lo)) + 0.10) / max(zoom, 1e-6)
    return ctr, half


def _camera(ctr, half):
    """ORTHOGRAPHIC top-down. The arm is planar and gravity-free by construction, so a perspective
    projection buys nothing and costs legibility (regions at different radii would render at
    different scales). Orthographic also makes framing exact: for an ortho camera MuJoCo reads
    `fovy` as the viewport height in LENGTH units, so `fovy = 2*half` frames precisely."""
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [ctr[0], ctr[1], 0.0]
    cam.azimuth, cam.elevation = 90.0, -89.9      # straight down, x right / y up
    cam.distance = 5.0                            # ortho: only near/far clipping depends on this
    return cam


def _render_model(dgp, half):
    """A RENDER-ONLY twin of the plant.

    MuJoCo's default offscreen framebuffer is 640x480, so anything larger has to be declared in
    the MJCF. Rather than touch `arm_env.build_xml` -- which every published cut in this tree
    compiles, and which `verify_backcompat.py` gates -- the renderer compiles its own copy with a
    `<visual>` block bolted on. `<visual>` is purely cosmetic (framebuffer size, headlight), so
    this model is structurally identical to the physics model: same bodies, geoms and sites in the
    same order, which is what makes it valid to hand `env.data` to `mjv_updateScene` against it.
    The physics model is never modified.
    """
    import mujoco
    from mjc.arm_env import build_xml

    vis = (f'  <visual>\n'
           f'    <global offwidth="{SIM}" offheight="{SIM}" '
           f'orthographic="true" fovy="{2.0 * half:.4f}"/>\n'
           f'    <headlight ambient="0.55 0.55 0.58" diffuse="0.40 0.40 0.40" '
           f'specular="0.08 0.08 0.08"/>\n'
           f'    <quality shadowsize="0"/>\n'
           f'  </visual>\n')
    xml = build_xml(dgp).replace('<mujoco model="planar_arm">',
                                 '<mujoco model="planar_arm">\n' + vis, 1)
    # recolour the end-effector marker into the trail's amber family, so "the thing that moves"
    # and "where it has been" read as one object instead of two competing reds. Cosmetic only --
    # `payload` is massless here (payload_mass=0) and contact-free.
    xml = xml.replace('rgba="0.85 0.25 0.25 1"', 'rgba="0.96 0.68 0.28 1"')
    return mujoco.MjModel.from_xml_string(xml)


def _open_panel(pack, half):
    """One plant + one GL context per PACK, reused across that pack's reaches (an OSMesa context
    per reach is both slow and a leak)."""
    import mujoco
    from mjc.arm_env import ArmEnv

    env = ArmEnv(pack["dgp"])
    return dict(env=env, opt=mujoco.MjvOption(),
                renderer=mujoco.Renderer(_render_model(env.dgp, half), height=SIM, width=SIM,
                                         max_geom=6000))


def _render_reach(panel, pack, meta, cam, cfg, b, stride=1):
    """Replay one ballistic reach one physics substep at a time, returning rendered frames.

    The PHYSICS always advances one substep at a time (so the trail is smooth and the fields are
    applied exactly as in the loop); `stride` only thins which substeps get RENDERED, which is how
    the smoke gets a representative final frame without paying for 140 OSMesa renders.
    """
    import numpy as np

    env, renderer, opt = panel["env"], panel["renderer"], panel["opt"]
    n, fs, H = meta["n_links"], meta["frame_skip"], meta["plan_H"]

    s0 = np.asarray(pack["starts"][b], np.float64)
    env.set_state(s0[:n], s0[n:])
    acts = np.asarray(pack["actions"][b], np.float32)
    trail, frames = [env.tip_pos().copy()], []
    k = 0
    for h in range(H):
        for _ in range(fs):
            env.step(acts[h], 1)                     # == step(u, fs), one frame per substep
            trail.append(env.tip_pos().copy())
            k += 1
            if k % stride:
                continue
            renderer.update_scene(env.data, camera=cam, scene_option=opt)
            _decorate(renderer.scene, meta, pack, b, trail, cfg)
            frames.append(renderer.render().copy())
    return frames


# =============================================================================== #
@app.function(image=render_image, cpu=8.0, memory=16384, timeout=1800,
              volumes={DATA_DIR: volume})
def render_smoke(cfg: dict) -> bytes:
    """One composed frame -> PNG. Validates OSMesa, camera framing, and the overlays before
    committing to a few hundred frames."""
    import io
    import numpy as np
    import imageio

    meta, packs, res = _load(cfg["tag"])
    print(f"[smoke] {len(packs)} packs: " + ", ".join(f"{p['policy']}@r{p['round']}" for p in packs),
          flush=True)
    early, late = _pick(packs, cfg)
    ctr, half = _framing(meta, [early, late], cfg["zoom"])
    cam = _camera(ctr, half)
    st = meta["frame_skip"]
    b = _pick_reaches(early, 1)[0]
    fr_e = _render_reach(_open_panel(early, half), early, meta, cam, cfg, b, stride=st)
    fr_l = _render_reach(_open_panel(late, half), late, meta, cam, cfg, b, stride=st)
    t = min(len(fr_e), len(fr_l)) - 1
    print(f"[smoke] reach {b}; {len(fr_e)} frames/panel; frame mean={fr_e[t].mean():.1f}", flush=True)
    canvas = _compose(cfg, meta, early, late, res)
    png = canvas(fr_e[t], fr_l[t], (t + 1) * st - 1, b)
    buf = io.BytesIO()
    imageio.imwrite(buf, png, format="png")
    return buf.getvalue()


@app.function(image=render_image, cpu=8.0, memory=32768, timeout=3600,
              volumes={DATA_DIR: volume})
def render_video(cfg: dict) -> bytes:
    import numpy as np
    import imageio

    meta, packs, res = _load(cfg["tag"])
    early, late = _pick(packs, cfg)
    print(f"[video] early={early['policy']}@r{early['round']} (miss {early['miss']:.4f}) "
          f"late={late['policy']}@r{late['round']} (miss {late['miss']:.4f})", flush=True)
    ctr, half = _framing(meta, [early, late], cfg["zoom"])
    cam = _camera(ctr, half)
    canvas = _compose(cfg, meta, early, late, res)

    frames = []
    pe, pl = _open_panel(early, half), _open_panel(late, half)
    picks = _pick_reaches(early, min(cfg["n_reach"], len(early["starts"])))
    for i, b in enumerate(picks):
        print(f"[video] reach {i + 1}/{len(picks)} (eval index {b}) ...", flush=True)
        fe = _render_reach(pe, early, meta, cam, cfg, b)
        fl = _render_reach(pl, late, meta, cam, cfg, b)
        for t in range(min(len(fe), len(fl))):
            frames.append(canvas(fe[t], fl[t], t, b))
        for _ in range(cfg["hold"]):                          # hold on the final posture
            frames.append(frames[-1])

    outdir = os.path.join(DATA_DIR, "videos")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"e3_{cfg['tag']}.mp4")
    print(f"[video] encoding {len(frames)} frames @ {cfg['fps']} fps -> {path}", flush=True)
    imageio.mimwrite(path, frames, fps=cfg["fps"], quality=9, macro_block_size=1)
    volume.commit()
    with open(path, "rb") as fh:
        return fh.read()


def _pick_reaches(pack, k):
    """Which of the 40 eval reaches to actually render.

    NOT the first k. The panel headline is the median miss over all 40, so the reaches on screen
    should be ones that behave like the median -- otherwise the viewer is shown a tail case while
    being quoted a central statistic. Picks the k reaches whose own miss is closest to the median.
    """
    import numpy as np

    pm = np.asarray(pack.get("per_miss") or [], float)
    if pm.size == 0:
        return list(range(k))
    return sorted(np.argsort(np.abs(pm - np.median(pm)))[:k].tolist())


def _pick(packs, cfg):
    """Choose the EARLY and LATE packs (default: lowest and highest round of the named policy)."""
    pol = cfg["policy"]
    cand = [p for p in packs if p["policy"] == pol] or packs
    cand = sorted(cand, key=lambda p: p["round"])
    early = next((p for p in cand if p["round"] == cfg["early"]), cand[0])
    late = next((p for p in cand if p["round"] == cfg["late"]), cand[-1])
    return early, late


# =============================================================================== #
# composition -- one reusable matplotlib canvas, redrawn per frame
# =============================================================================== #
def _compose(cfg, meta, early, late, res):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI, facecolor=C_BG)
    blank = np.zeros((SIM, SIM, 3), np.uint8)

    fig.text(0.5, 0.982, "Ballistic reaching on the on-policy arm — the model's forecast vs the body",
             ha="center", va="top", color=C_FG, fontsize=15, weight="bold")
    fig.text(0.5, 0.945,
             "One CEM plan of 14 torques through the learned forward model, executed open-loop. "
             "No feedback: the plan is only as good as the model.",
             ha="center", va="top", color=C_DIM, fontsize=9.5)

    ims, pr = [], []
    heads = [("EARLY", early, "#d98a52"), ("LATE", late, "#5fbf7f")]
    for i, (kind, pk, col) in enumerate(heads):
        x0 = 0.088 + 0.470 * i
        ax = fig.add_axes([x0, 0.245, 0.354, 0.645])
        ax.set_facecolor("#07080a")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#242830")
        ims.append(ax.imshow(blank, interpolation="bilinear"))
        ax.set_title(f"{kind}  ·  {_round_label(pk['round'])}", color=col, fontsize=12.5,
                     weight="bold", pad=7)
        fig.text(x0 + 0.177, 0.238,
                 f"median final miss  {pk['miss']*100:.1f} cm          "
                 f"model↔body divergence  {pk['tip_divergence']*100:.1f} cm",
                 ha="center", va="top", color=C_FG, fontsize=9.5)
        pr.append(fig.text(x0 + 0.177, 0.213, "", ha="center", va="top",
                           color=C_DIM, fontsize=8.5))

    legend = [
        Line2D([], [], marker="o", ls="", ms=8, mfc=C_ACTUAL, mec="none", label="tip — actual"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=C_PRED, mec="none",
               label="tip — model's forecast"),
        Line2D([], [], marker="o", ls="", ms=8, mfc=C_GOAL, mec="none", label="goal"),
        Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=C_CURL, mew=1.6,
               label="curl region — reducible"),
        Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=C_NOISE, mew=1.6,
               label="noise region — irreducible"),
    ]
    clock = fig.text(0.5, 0.188, "", ha="center", va="top", color="#6a6f79", fontsize=8.5)

    lg = fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.168), ncol=5,
                    frameon=False, fontsize=9, handletextpad=0.5, columnspacing=2.0)
    for txt in lg.get_texts():
        txt.set_color(C_DIM)

    # a static context strip: the sighted grader over rounds, with the two rendered rounds marked
    if res is not None and cfg["policy"] in res.get("results", {}):
        names = res.get("region_names", [])
        a = names.index("A") if "A" in names else 0
        ys = [r["reg_err"][a] for r in res["results"][cfg["policy"]]]
        axc = fig.add_axes([0.205, 0.070, 0.595, 0.070])
        axc.set_facecolor(C_BG)
        axc.plot(range(len(ys)), ys, color=C_ACTUAL, lw=1.5)
        for pk, col in ((early, "#d98a52"), (late, "#5fbf7f")):
            t = max(pk["round"], 0)
            if t < len(ys):
                axc.scatter([t], [ys[t]], s=38, color=col, zorder=5)
        axc.text(-0.075, 0.5, "region-A\nFM error", transform=axc.transAxes, color=C_DIM,
                 fontsize=7.5, ha="right", va="center", linespacing=1.4)
        axc.tick_params(colors=C_DIM, labelsize=7, length=2, pad=2)
        axc.text(1.02, 0.5, f"collection round\npolicy = {cfg['policy']}",
                 transform=axc.transAxes, color=C_DIM, fontsize=7.5, ha="left", va="center",
                 linespacing=1.4)
        for k, sp in axc.spines.items():
            sp.set_color("#242830") if k in ("left", "bottom") else sp.set_visible(False)

    def draw(frame_e, frame_l, t, b):
        ims[0].set_data(frame_e)
        ims[1].set_data(frame_l)
        fs, H = meta["frame_skip"], meta["plan_H"]
        for txt, pk in zip(pr, (early, late)):
            pm = pk.get("per_miss")
            txt.set_text(f"this reach: {pm[b]*100:.1f} cm" if pm else "")
        clock.set_text(f"eval reach {b}   ·   command step {t // fs + 1}/{H}   ·   "
                       f"t = {(t + 1) * 0.002 * 1000:.0f} ms   ·   open loop since t = 0")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        return img

    return draw


# =============================================================================== #
@app.local_entrypoint()
def smoke(tag: str = "render_v1", policy: str = "value", early: int = -1, late: int = 29,
          b_hi: float = 10.0, zoom: float = 1.0):
    cfg = dict(tag=tag, policy=policy, early=early, late=late, b_hi=b_hi, zoom=zoom)
    png = render_smoke.remote(cfg)
    out = os.path.join(os.path.dirname(__file__), "figures", f"render_smoke_{tag}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"[local] wrote {out} ({len(png)/1e3:.0f} kB)")


@app.local_entrypoint()
def video(tag: str = "render_v1", policy: str = "value", early: int = -1, late: int = 29,
          n_reach: int = 4, fps: int = 40, hold: int = 20, b_hi: float = 10.0,
          zoom: float = 1.0):
    cfg = dict(tag=tag, policy=policy, early=early, late=late, n_reach=n_reach, fps=fps,
               hold=hold, b_hi=b_hi, zoom=zoom)
    mp4 = render_video.remote(cfg)
    out = os.path.join(os.path.dirname(__file__), "videos", f"e3_{tag}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(mp4)
    print(f"[local] wrote {out} ({len(mp4)/1e6:.1f} MB)")
