"""Cost profiler for the a cappella substrate: what does a GROUNDING cost, in wall-clock?

This node has NO forward model, so the only expensive thing is the incumbent's search, and its
inner loop is a real plant rollout: `env.set_state(s)` then `hh` calls of `env.step(u, frame_skip)`
in the ROTATED world (rot_regions on, applied in Python once per physics substep). Everything the
run's budget is sized from is measured here rather than projected:

  (1) us per control step, rotated vs clean world (the rot_regions Python cost is the tax)
  (2) us per 34-step rollout == ONE GROUNDING
  (3) a G-rollout decision at G in {16 .. 512}
  (4) process-pool scaling (MuJoCo is single-threaded per env; the per-substep numpy in
      `_apply_rot_regions` holds the GIL, so threads are expected NOT to scale and processes are)
  (5) projections: one performance-tempo run-through (n_seg seams x Bn performers x G), and a
      practice cycle, at the shapes gates.py/acappella.py would run.

    cd experiments/                       # MODAL_PROFILE=chromatic
    modal run mjc/practice/acappella/profile_cost.py::profile
"""

import modal

from mjc.shared import app, DATA_DIR  # noqa: F401

# etude.py's world/piece constants, verbatim (gate A-F1 asserts this against the donor source)
DGP = dict(arena_half=1.8, gear=10.0, joint_damping=2.0, pusher_r=0.12)
ROT_REGIONS = [dict(center=[-0.4, 0.0], sigma=0.16, phi=1.2)]   # segment 1's midpoint
FRAME_SKIP = 12
SEG_H = 34
K_SEG = 4


def _mk_env(rot: bool):
    from mjc.pusher_env import PusherEnv
    dgp = dict(DGP)
    if rot:
        dgp["rot_regions"] = [dict(r) for r in ROT_REGIONS]
    return PusherEnv(dgp, with_puck=False)


def _rollout_block(args):
    """One worker: `n` rollouts of `hh` steps from a fixed state. Returns elapsed seconds."""
    import time
    import numpy as np
    n, hh, rot, seed = args
    env = _mk_env(rot)
    rng = np.random.default_rng(seed)
    s = np.array([-0.4, -0.4, 0.0, 0.0])
    cmds = rng.uniform(-1, 1, (n, hh, 2)).astype(np.float32)
    t0 = time.perf_counter()
    for i in range(n):
        env.set_state(s[:2], s[2:])
        for h in range(hh):
            env.step(cmds[i, h], FRAME_SKIP)
        env.get_state()
    return time.perf_counter() - t0


def _bench(n_proc_ladder=(1, 2, 4, 8, 16)):
    import os
    import time
    import numpy as np

    out = {"cpu_count": os.cpu_count()}

    # ---- (1) us per control step, and the rot_regions tax -------------------------------
    for nm, rot in (("clean", False), ("rot", True)):
        env = _mk_env(rot)
        rng = np.random.default_rng(0)
        s = np.array([-0.4, -0.4, 0.0, 0.0])
        u = rng.uniform(-1, 1, 2).astype(np.float32)
        for _ in range(300):
            env.set_state(s[:2], s[2:]); env.step(u, FRAME_SKIP)
        n = 4000
        t0 = time.perf_counter()
        for _ in range(n):
            env.step(u, FRAME_SKIP)
        out[f"us_per_ctrl_step_{nm}"] = (time.perf_counter() - t0) / n * 1e6
        t0 = time.perf_counter()
        for _ in range(n):
            env.set_state(s[:2], s[2:])
        out[f"us_per_set_state_{nm}"] = (time.perf_counter() - t0) / n * 1e6

    # ---- (2) one grounding = set_state + SEG_H steps, in the world the agent acts in -----
    for hh in (SEG_H, SEG_H * K_SEG):
        el = _rollout_block((200 if hh == SEG_H else 60, hh, True, 1))
        out[f"ms_per_grounding_h{hh}"] = el / (200 if hh == SEG_H else 60) * 1e3

    # ---- (3) a G-rollout decision (single performer, single seam) ------------------------
    g_ms = out[f"ms_per_grounding_h{SEG_H}"]
    out["decision_ms_by_G"] = {str(G): g_ms * G for G in (16, 32, 64, 128, 256, 512)}

    # ---- (4) process-pool scaling -------------------------------------------------------
    import concurrent.futures as cf
    per_worker = 96
    scal = {}
    for npz in n_proc_ladder:
        if npz > (out["cpu_count"] or 1):
            continue
        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(max_workers=npz) as ex:
            list(ex.map(_rollout_block, [(per_worker, SEG_H, True, 100 + i) for i in range(npz)]))
        wall = time.perf_counter() - t0
        scal[str(npz)] = dict(wall_s=wall, groundings=per_worker * npz,
                              ms_per_grounding=wall / (per_worker * npz) * 1e3)
    out["proc_scaling"] = scal

    # thread scaling, for the record (expected flat: the per-substep numpy holds the GIL)
    thr = {}
    for nt in (1, 8):
        if nt > (out["cpu_count"] or 1):
            continue
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=nt) as ex:
            list(ex.map(_rollout_block, [(per_worker, SEG_H, True, 200 + i) for i in range(nt)]))
        wall = time.perf_counter() - t0
        thr[str(nt)] = dict(wall_s=wall, ms_per_grounding=wall / (per_worker * nt) * 1e3)
    out["thread_scaling"] = thr

    # ---- (5) projections at the shapes the node would run -------------------------------
    best = min([v["ms_per_grounding"] for v in scal.values()] or [g_ms])
    n_par = int(min(scal, key=lambda k: scal[k]["ms_per_grounding"])) if scal else 1
    out["best_ms_per_grounding"] = best
    out["best_n_proc"] = n_par
    proj = {}
    for G in (32, 64, 128, 256):
        for Bn, nm in ((64, "runthrough_n64"), (32, "practice_b32"), (24, "eval_n24")):
            proj[f"G{G}_{nm}_s"] = best * G * K_SEG * Bn / 1e3
    out["projected_traversal_s"] = proj
    return out


@app.function(cpu=16.0, memory=16384, timeout=3600)
def profile():
    return _bench()


@app.function(cpu=32.0, memory=32768, timeout=3600)
def profile32():
    return _bench(n_proc_ladder=(1, 4, 8, 16, 32))


@app.local_entrypoint()
def acappella_profile(big: bool = False):
    import json
    print(json.dumps((profile32 if big else profile).remote(), indent=2, sort_keys=True))
