"""Rendered videos for the MuJoCo control substrate: top-down sim + live FM residual.

A communication tool — one nice little video per cut. This one shows the planar
pusher (top-down) next to a live plot of the forward-model residual, with contact
shaded, so you can watch the residual spike exactly when the pusher hits something.

Headless rendering uses **OSMesa** (pure-software OpenGL, no GPU/EGL driver
needed) via `MUJOCO_GL=osmesa`. `render_smoke` validates the GL setup + camera
framing on a single frame before `make_video` renders the whole rollout.

Run:
    cd experiments/
    modal run mujoco_control/render_video.py::render_smoke          # 1 frame -> PNG
    modal run mujoco_control/render_video.py::make_video --tag demo # rollout -> mp4
"""

import json
import os
import modal

from mujoco_control.shared import volume, DATA_DIR, NumpyEncoder

render_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libosmesa6", "libgl1", "libglx-mesa0", "libglib2.0-0")
    .pip_install(
        "numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0", "mujoco==3.2.3",
        "matplotlib==3.9.2", "imageio==2.36.0", "imageio-ffmpeg==0.5.1",
    )
    .env({"MUJOCO_GL": "osmesa"})
    .add_local_python_source("mujoco_control")
)

app = modal.App("mujoco-control-render", image=render_image)

# render geometry (all divisible by 16 for the video encoder)
H = 480
W_SIM = 480
W_PANEL = 608
DGP = dict(arena_half=0.8, pusher_r=0.15, puck_r=0.15, puck_mass=2.0)  # = cut #1 world


def _render_opt():
    import mujoco

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    return opt


@app.function(image=render_image, cpu=4.0, timeout=900, volumes={DATA_DIR: volume})
def render_smoke() -> bytes:
    """Render a single top-down frame to validate GL + camera framing."""
    import numpy as np
    import mujoco
    from mujoco_control.pusher_env import PusherEnv

    env = PusherEnv(DGP)
    rng = np.random.default_rng(0)
    env.reset(rng)
    # nudge the puck next to the pusher so the frame is representative
    env.data.qpos[:] = np.array([0.0, 0.0, 0.32, 0.1])
    mujoco.mj_forward(env.model, env.data)

    renderer = mujoco.Renderer(env.model, height=H, width=W_SIM)
    renderer.update_scene(env.data, camera="topdown", scene_option=_render_opt())
    frame = renderer.render()  # (H, W_SIM, 3) uint8
    print(f"[smoke] rendered frame {frame.shape}, dtype={frame.dtype}, "
          f"mean={frame.mean():.1f}", flush=True)

    import imageio
    import io
    buf = io.BytesIO()
    imageio.imwrite(buf, frame, format="png")
    return buf.getvalue()


@app.function(image=render_image, cpu=8.0, memory=16384, timeout=1800,
              volumes={DATA_DIR: volume})
def make_video(cfg: dict) -> bytes:
    import io
    import numpy as np
    import torch
    import torch.nn as nn
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    from mujoco_control.pusher_env import collect_transitions, PusherEnv

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---- 1. train an FM (same recipe as cut #1) --------------------------- #
    print("[video] collecting train data + fitting FM ...", flush=True)
    data = collect_transitions(
        dgp=DGP, n_episodes=cfg["train_eps"], ep_len=250, frame_skip=cfg["frame_skip"],
        seed=cfg["seed"], sigma=0.5, theta=0.15, seek_gain=1.0)
    S, U, S2 = data["S"], data["U"], data["S2"]
    X = np.concatenate([S, U], axis=1).astype(np.float32)
    Y = (S2 - S).astype(np.float32)
    mx, sx = X.mean(0), X.std(0) + 1e-6
    my, sy = Y.mean(0), Y.std(0) + 1e-6
    Xn = torch.tensor((X - mx) / sx)
    Yn = torch.tensor((Y - my) / sy)
    fm = nn.Sequential(nn.Linear(10, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                       nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 8))
    opt = torch.optim.Adam(fm.parameters(), lr=1e-3)
    lossf = nn.HuberLoss(delta=1.0)
    n = Xn.shape[0]
    for _e in range(cfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            lossf(fm(Xn[idx]), Yn[idx]).backward()
            opt.step()
    fm.eval()

    # ---- 2. eval rollout: step, render, compute residual ------------------ #
    print(f"[video] rolling out {cfg['video_len']} steps + rendering ...", flush=True)
    env = PusherEnv(DGP)
    rng = np.random.default_rng(cfg["seed"] + 7)
    env.reset(rng)
    renderer = mujoco.Renderer(env.model, height=H, width=W_SIM)
    ropt = _render_opt()

    sim_frames, res, contact = [], [], []
    mx_t = torch.tensor(mx); sx_t = torch.tensor(sx)
    my_t = torch.tensor(my); sy_t = torch.tensor(sy)
    a = np.zeros(2)
    for t in range(cfg["video_len"]):
        s = env.get_state()
        d = s[2:4] - s[0:2]
        direction = d / (np.linalg.norm(d) + 1e-6)
        a = a - 0.15 * a + 0.5 * rng.normal(size=2)
        ctrl = np.clip(a + 1.0 * direction, -1.0, 1.0)
        s2, info = env.step(ctrl, cfg["frame_skip"])
        with torch.no_grad():
            x = torch.tensor(np.concatenate([s, ctrl]).astype(np.float32))
            pred_n = fm((x - mx_t) / sx_t)
            true_n = (torch.tensor((s2 - s)) - my_t) / sy_t
            r = float(torch.linalg.norm(pred_n - true_n))
        res.append(r)
        contact.append(bool(info["any_contact"]))
        renderer.update_scene(env.data, camera="topdown", scene_option=ropt)
        sim_frames.append(renderer.render().copy())

    res = np.asarray(res)
    contact = np.asarray(contact)
    ymax = float(res.max() * 1.1)
    T = len(res)

    # ---- 3. compose sim | panel frames ------------------------------------ #
    print("[video] composing frames ...", flush=True)
    C_CONTACT, C_FREE = "#d1603d", "#3d6fd1"
    dpi = 100
    tt = np.arange(T)
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(W_PANEL / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0.16, 0.17, 0.80, 0.70])
        ax.plot(tt[:t + 1], res[:t + 1], color="#111", lw=1.5)
        ax.fill_between(tt[:t + 1], 0, ymax, where=contact[:t + 1],
                        color=C_CONTACT, alpha=0.20, step="mid")
        ax.scatter([t], [res[t]], s=45, zorder=5,
                   color=C_CONTACT if contact[t] else C_FREE)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("control step")
        ax.set_ylabel("forward-model residual  ‖r‖")
        ax.set_title("residual spikes at contact" +
                     ("   ● CONTACT" if contact[t] else ""),
                     color=C_CONTACT if contact[t] else "#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.canvas.draw()
        panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        # pad/crop panel to exactly (H, W_PANEL) if dpi rounding differs
        ph, pw = panel.shape[:2]
        if (ph, pw) != (H, W_PANEL):
            fixed = np.full((H, W_PANEL, 3), 255, np.uint8)
            fixed[:min(ph, H), :min(pw, W_PANEL)] = panel[:H, :W_PANEL]
            panel = fixed
        frames.append(np.concatenate([sim_frames[t], panel], axis=1))

    # ---- 4. encode mp4 ---------------------------------------------------- #
    print(f"[video] encoding {len(frames)} frames @ {cfg['fps']} fps ...", flush=True)
    outdir = os.path.join(DATA_DIR, "videos")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{cfg['tag']}.mp4")
    imageio.mimwrite(path, frames, fps=cfg["fps"], quality=8, macro_block_size=1)
    volume.commit()
    with open(path, "rb") as fh:
        video_bytes = fh.read()
    print(f"[video] wrote {path} ({len(video_bytes)/1e6:.1f} MB)", flush=True)
    return video_bytes


@app.local_entrypoint()
def render_smoke_local():
    png = render_smoke.remote()
    out = os.path.join(os.path.dirname(__file__), "figures", "render_smoke.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"[local] wrote {out}")


@app.local_entrypoint()
def make_video_local(
    tag: str = "contact_residual_demo",
    seed: int = 0,
    train_eps: int = 300,
    video_len: int = 300,
    frame_skip: int = 5,
    epochs: int = 40,
    fps: int = 30,
):
    cfg = dict(tag=tag, seed=seed, train_eps=train_eps, video_len=video_len,
               frame_skip=frame_skip, epochs=epochs, fps=fps)
    video = make_video.remote(cfg)
    out = os.path.join(os.path.dirname(__file__), "videos", f"{tag}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(video)
    print(f"[local] wrote {out} ({len(video)/1e6:.1f} MB)")
