"""Cost profiler for the etude substrate: where does a cycle's wall-clock actually go?

Measures, at the EXACT shapes etude.py runs at, the three candidate bottlenecks:
  (1) MuJoCo stepping   -- CPU, single-threaded, one Python call per performer per control step
  (2) CEM noise + H2D   -- numpy float64 gaussians on CPU, then a copy to the device
  (3) the FM forwards   -- Bn*k_shoot rows through a 6->256->256->256->4 MLP, hh*cem_iters times

and prices the alternatives (TF32, device-side sampling, CPU-only, fp16).

    modal run mjc/practice/etude/profile_cost.py::profile          # L4
    modal run mjc/practice/etude/profile_cost.py::profile_cpu      # CPU box
"""

import modal

from mjc.shared import app, DATA_DIR  # noqa: F401

# etude.py's defaults, verbatim
FM_HIDDEN, FM_LAYERS = 256, 3
K_SHOOT, CEM_ITERS, CEM_ELITE = 256, 4, 32
SEG_H, FRAME_SKIP, K_SEGS = 34, 12, 4
BATCH, N_RT, N_EVAL = 32, 96, 40
DGP = dict(arena_half=1.8, gear=10.0, joint_damping=2.0, pusher_r=0.12)


def _bench(device: str, threads: int = 0):
    import time
    import numpy as np
    import torch
    import torch.nn as nn

    if threads:
        torch.set_num_threads(threads)
    out = {"device": device, "torch_threads": torch.get_num_threads()}

    def mlp(h=FM_HIDDEN, L=FM_LAYERS, din=6, dout=4):
        lyr = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(L - 1):
            lyr += [nn.Linear(h, h), nn.SiLU()]
        lyr += [nn.Linear(h, dout)]
        return nn.Sequential(*lyr).to(device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # ---- (1) MuJoCo: one env.set_state + env.step(fs=12) call, as traverse() does it ----
    from mjc.pusher_env import PusherEnv
    env = PusherEnv(DGP, with_puck=False)
    rng = np.random.default_rng(0)
    s = np.zeros(4); u = rng.uniform(-1, 1, 2).astype(np.float32)
    for _ in range(200):                                    # warm
        env.set_state(s[:2], s[2:]); env.step(u, FRAME_SKIP)
    n = 3000
    t0 = time.perf_counter()
    for _ in range(n):
        env.set_state(s[:2], s[2:]); env.step(u, FRAME_SKIP)
    out["mujoco_us_per_ctrl_step"] = (time.perf_counter() - t0) / n * 1e6

    # ---- (2) the CEM noise draw, at practice (Bn=32) and run-through (Bn=96) shapes ----
    for Bn in (BATCH, N_RT):
        shp = (Bn, K_SHOOT, SEG_H, 2)
        mu = np.zeros((Bn, SEG_H, 2), np.float32); sig = np.full((Bn, SEG_H, 2), 0.8, np.float32)
        t0 = time.perf_counter()
        for _ in range(20):
            e = rng.standard_normal(shp).astype(np.float32)      # <- float64 then cast, as written
            seqs = np.clip(mu[:, None] + sig[:, None] * e, -1, 1)
        out[f"cem_noise_ms_B{Bn}"] = (time.perf_counter() - t0) / 20 * 1e3
        if device == "cuda":
            t0 = time.perf_counter()
            for _ in range(20):
                torch.tensor(seqs.reshape(Bn * K_SHOOT, SEG_H, 2), device=device)
            sync()
            out[f"h2d_ms_B{Bn}"] = (time.perf_counter() - t0) / 20 * 1e3
            g = torch.Generator(device=device).manual_seed(0)
            t0 = time.perf_counter()
            for _ in range(20):
                torch.randn(shp, device=device, generator=g).clamp_(-1, 1)
            sync()
            out[f"cem_noise_ondevice_ms_B{Bn}"] = (time.perf_counter() - t0) / 20 * 1e3

    # ---- (3) the FM rollout inside mpc_plan: hh sequential forwards on Bn*K rows ----
    net = mlp()
    flops_per_row = 2 * (6 * 256 + 256 * 256 + 256 * 256 + 256 * 4)

    def roll(rows, tf32=False, half=False):
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        m = net.half() if half else net.float()
        dt = torch.float16 if half else torch.float32
        s = torch.zeros(rows, 4, device=device, dtype=dt)
        seq = torch.zeros(rows, SEG_H, 2, device=device, dtype=dt)
        g = torch.zeros(rows, 2, device=device, dtype=dt)
        reps = 3
        with torch.no_grad():
            for _ in range(2):                                     # warm
                x = torch.cat([s, seq[:, 0, :]], 1); net(x)
            sync(); t0 = time.perf_counter()
            for _ in range(reps):
                for _it in range(CEM_ITERS):
                    ss = s.clone(); cost = torch.zeros(rows, device=device, dtype=dt)
                    for h in range(SEG_H):
                        x = torch.cat([ss, seq[:, h, :]], 1)
                        ss = ss + m(x)
                        cost = cost + (ss[:, :2] - g).norm(dim=1)
                    torch.topk(-cost.reshape(rows // K_SHOOT, K_SHOOT), CEM_ELITE, 1).indices.cpu()
            sync()
        net.float()
        el = (time.perf_counter() - t0) / reps
        fl = CEM_ITERS * SEG_H * rows * flops_per_row
        return el * 1e3, fl / el / 1e12

    for Bn in (BATCH, N_EVAL, N_RT):
        rows = Bn * K_SHOOT
        ms, tf = roll(rows, tf32=False)
        out[f"mpc_plan_ms_B{Bn}"] = ms
        out[f"mpc_plan_tflops_B{Bn}"] = tf
        if device == "cuda":
            ms32, tf32v = roll(rows, tf32=True)
            out[f"mpc_plan_tf32_ms_B{Bn}"] = ms32
            out[f"mpc_plan_tf32_tflops_B{Bn}"] = tf32v
            msh, tfh = roll(rows, half=True)
            out[f"mpc_plan_fp16_ms_B{Bn}"] = msh

    # ---- (4) FM training step (train_online): fm_batch=256, n_grad=10 ----
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    X = torch.randn(256, 6, device=device); Y = torch.randn(256, 4, device=device)
    hub = nn.HuberLoss(delta=1.0)
    sync(); t0 = time.perf_counter()
    for _ in range(50):
        opt.zero_grad(); hub(net(X), Y).backward(); opt.step()
    sync()
    out["fm_grad_step_ms"] = (time.perf_counter() - t0) / 50 * 1e3
    return out


@app.function(gpu="L4", memory=2048, timeout=1800)
def profile():
    import os
    import torch
    r = _bench("cuda")
    r["cpu_count"] = os.cpu_count()
    r["gpu"] = torch.cuda.get_device_name(0)
    r["gpu_mem_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    r["gpu_mem_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    return r


@app.function(cpu=16, memory=4096, timeout=3600)
def profile_cpu(threads: int = 16):
    import os
    r = _bench("cpu", threads=threads)
    r["cpu_count"] = os.cpu_count()
    return r


@app.local_entrypoint()
def main(cpu_only: bool = False, gpu_only: bool = False):
    import json
    if not cpu_only:
        print("=== L4 ===")
        print(json.dumps(profile.remote(), indent=2, sort_keys=True))
    if not gpu_only:
        for th in (8, 16):
            print(f"=== CPU (threads={th}) ===")
            print(json.dumps(profile_cpu.remote(threads=th), indent=2, sort_keys=True))
