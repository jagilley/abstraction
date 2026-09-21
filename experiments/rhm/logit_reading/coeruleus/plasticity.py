"""Q4: close the plasticity loop -- the readout gates the per-position learning rate.

Continued training on the eps-corrupted stream from a MID-trajectory checkpoint of
`altitude/train_noisy.py` (step 8000, where kappa* ~ 3.6 and the frontier is still live), so
"don't learn from what surprised me" can be told apart from "don't learn at the frontier".

Five arms, differing only in how the per-position NTP loss is weighted. Following
[`endogenous_teacher`](../../endogenous_teacher/README.md), every gated arm uses
rank-normalised weights `w = 2*rank/(N-1)` over the positions of the batch, so the weight
MULTISET is bit-identical across arms and only the assignment moves:

  uniform         w = 1                                       no gate
  head_down       w decreasing in the Q2 head's output        "protect what surprised me"
  head_shuf       the same multiset, randomly permuted        the endogenous_teacher control
  oracle_down     w decreasing in the exact "a corrupted token lies in [t-H, t]" indicator
  oracle_up       w increasing in it                          the other direction

`head_shuf` is the control that says whether any effect is attributable to WHERE the weight
went rather than how much of it there was; the two oracle arms bound what a perfectly accurate
gate could do, which separates "the readout carries nothing" from "the intervention does
nothing".

The gate reads the LIVE model's residual stream through the frozen head, so it drifts with the
model; the drift is logged (correlation of the gate with the realised horizon excess).

Run:
  modal run -m rhm.logit_reading.coeruleus.plasticity::train_arm \
      --ckpt /data/.../traj_eps01_s42/step008000.pt --arm uniform --steps 300 \
      --ckpt-steps 8300 --chunk-seqs 20000 --tag smoke
  modal run --detach -m rhm.logit_reading.coeruleus.plasticity::plasticity_sweep \
      --ckpt /data/.../traj_eps01_s42/step008000.pt --tag c1
"""

import json
import os
import time

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import (app, load_trajectory_ckpt, tb_key,
                                           calibration_ckpt)
from rhm.logit_reading.altitude.dual_ladder import dual_ckpt
from rhm.logit_reading.coeruleus.prize import prize_ckpt

ARMS = ("uniform", "head_down", "head_shuf", "oracle_down", "oracle_up")


def torch_head(path, device):
    """The Q2 head as a torch callable on (B, T, d) -> (B, T)."""
    import torch
    import torch.nn as nn
    sd = torch.load(path, map_location=device)
    st = sd["state"]
    if st.get("kind", "mlp") == "linear":
        beta = st["beta"].to(device).float()

        def f(S):
            return S @ beta[:-1] + beta[-1]
        return f, st["block"], sd["config"]
    hidden = sd["config"]["hidden"]
    d = st["mu"].shape[1]
    net = (nn.Linear(d, 1) if hidden == 0 else
           nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))).to(device)
    net.load_state_dict({k: v.to(device) for k, v in st["net"].items()})
    net.eval()
    mu, sdv = st["mu"].to(device), st["sd"].to(device)
    ym, ys = st["ym"], st["ys"]

    def f(S):
        shp = S.shape[:-1]
        return (net((S.reshape(-1, d) - mu) / sdv)[:, 0] * ys + ym).reshape(shp)
    return f, st["block"], sd["config"]


def rank_weights(g, mode, gen, device):
    """Rank-normalised weights, mean 1, multiset {2i/(N-1)} for every gated mode."""
    import torch
    N = g.numel()
    if mode == "uniform":
        return torch.ones_like(g)
    base = 2.0 * torch.arange(N, device=device, dtype=g.dtype) / (N - 1)
    if mode.endswith("_shuf"):
        perm = torch.randperm(N, generator=gen).to(device)
        return base[perm]
    jitter = torch.rand(N, generator=gen).to(device) * 1e-6      # break ties at random
    order = torch.argsort(g + jitter)
    w = torch.empty_like(g)
    w[order] = base if mode.endswith("_up") else base.flip(0)
    return w


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=4 * 3600, memory=12288,
              max_containers=6)
def train_arm(ckpt: str, arm: str = "uniform", head_path: str = "", horizon: int = 8,
              steps: int = 16000, ckpt_steps: str = "12000,16000,24000",
              batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
              chunk_seqs: int = 1_000_000, data_seed: int = 17, seed: int = 42,
              log_interval: int = 1000, tag: str = "c1", out_root: str = ""):
    import resource
    import torch
    import torch.nn.functional as F
    from rhm.logit_reading.grammar import generate
    volume.reload()
    device = "cuda"
    H = horizon
    model, rules, rule_w, cfg = load_trajectory_ckpt(ckpt, device)
    model.train()
    v, s, L = cfg["v"], cfg["s"], cfg["L"]
    T = s ** L
    eps_train = cfg.get("eps_train", 0.0)
    step0 = int(cfg["step"])
    key = tb_key(v, s, L, cfg["m"])
    root = out_root or f"{DATA_DIR}/{key}/logit_reading/coeruleus/cont_{tag}"
    out_dir = f"{root}/{arm}"
    os.makedirs(out_dir, exist_ok=True)
    ckpts = sorted({int(x) for x in ckpt_steps.split(",")})

    hf, blk, hcfg = (None, None, None)
    if arm.startswith("head"):
        hp = head_path or f"{ckpt[:-3]}_coeruleus_head_state_excess_ridge.pt"
        hf, blk, hcfg = torch_head(hp, device)

    run_cfg = dict(base_ckpt=ckpt, arm=arm, step0=step0, steps=steps, H=H,
                   batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                   chunk_seqs=chunk_seqs, data_seed=data_seed, seed=seed,
                   eps_train=eps_train, head=head_path or "", block=blk,
                   note="Adam state is not carried over from the base run; identical across arms")
    print(f"ARM {arm} from step {step0} (+{steps}) eps_train={eps_train} block={blk} "
          f"-> {out_dir}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1000)
    wgen = torch.Generator().manual_seed(seed + 2000)
    data_rng = np.random.default_rng(data_seed)
    arangeT = torch.arange(T + 1, device=device)

    def new_stream():
        t0 = time.time()
        seqs = generate(rules, rule_w, chunk_seqs, data_rng)
        st = torch.as_tensor(seqs.reshape(-1).astype(np.uint8), device=device)
        print(f"  new data chunk ({chunk_seqs:,} seqs) {time.time() - t0:.1f}s", flush=True)
        return st

    stream = new_stream()
    steps_per_chunk = max(1, chunk_seqs // batch_size)
    losses, log = [], []

    def save(step):
        model.eval()
        torch.save({"model": model.state_dict(),
                    "config": {**cfg, "step": step, "cont": run_cfg}},
                   f"{out_dir}/step{step:06d}.pt")
        volume.commit()
        model.train()

    t_start = time.time()
    for it in range(steps + 1):
        step = step0 + it
        if step in ckpts:
            save(step)
        if it == steps:
            break
        if it > 0 and it % steps_per_chunk == 0:
            stream = new_stream()
        ix = torch.randint(0, stream.shape[0] - T - 1, (batch_size,), generator=gen).to(device)
        w = stream[ix[:, None] + arangeT[None, :]].long()
        corrupt = torch.zeros_like(w, dtype=torch.bool)
        if eps_train > 0:
            flip = torch.rand(w.shape, generator=noise_gen, device=device) < eps_train
            rnd = torch.randint(0, v, w.shape, generator=noise_gen, device=device)
            corrupt = flip & (rnd != w)
            w = torch.where(flip, rnd, w)
        x, y = w[:, :T], w[:, 1:]
        need_state = arm.startswith("head")
        if need_state:
            logits, _, inter = model(x, return_intermediates=True)
            with torch.no_grad():
                g = hf(inter[blk].detach().float())                     # (B, T)
        else:
            logits, _ = model(x)
            if arm.startswith("oracle"):
                # 1 where a corrupted token lies in window indices [t - H, t]
                c = corrupt[:, :T].float()
                csp = torch.cat([torch.zeros(batch_size, 1, device=device),
                                 torch.cumsum(c, 1)], 1)                 # csp[:, t+1] = sum_{<=t}
                lo = torch.clamp(torch.arange(T, device=device) - H, min=0)
                g = (csp[:, 1:] - csp.gather(1, lo[None, :].expand(batch_size, T))) > 0
                g = g.float()
            else:
                g = torch.zeros(batch_size, T, device=device)
        lp = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1), reduction="none")
        wt = rank_weights(g.reshape(-1).detach(), arm, wgen, device)
        loss = (wt * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float((lp.mean()).detach()))
        if it % log_interval == 0:
            with torch.no_grad():
                lq = torch.log_softmax(logits.float(), -1)
                nll = -lq.gather(-1, y[..., None])[..., 0]
                Hq = -(lq.exp() * lq).sum(-1)
                ex = (nll - Hq)
                cs = torch.cumsum(ex, 1)
                yh = cs[:, H:] - torch.cat([torch.zeros(batch_size, 1, device=device),
                                            cs[:, :-1]], 1)[:, :T - H]
                gg = g[:, :T - H].reshape(-1).double()
                yy = yh.reshape(-1).double()
                rho = float(torch.corrcoef(torch.stack([gg, yy]))[0, 1]) \
                    if gg.std() > 0 else float("nan")
            rec = {"step": step, "loss": float(np.mean(losses[-log_interval:])),
                   "gate_target_corr": rho, "gate_mean": float(g.mean()),
                   "gate_sd": float(g.std()), "w_mean": float(wt.mean()),
                   "elapsed_s": time.time() - t_start}
            log.append(rec)
            print(f"  step {step:6d}  loss {rec['loss']:.4f}  gate-target rho {rho:+.4f}  "
                  f"{rec['elapsed_s']:.0f}s", flush=True)

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump({"config": run_cfg, "log": log, "ckpt_steps": ckpts},
                  f, cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"done {arm} in {time.time() - t_start:.0f}s  peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB -> {out_dir}",
          flush=True)
    return out_dir


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def plasticity_sweep(ckpt: str, arms: str = ",".join(ARMS), head_path: str = "",
                     steps: int = 16000, ckpt_steps: str = "12000,16000,24000",
                     data_seed: int = 17, tag: str = "c1"):
    volume.reload()
    a = arms.split(",")
    return list(train_arm.starmap([(ckpt, arm) for arm in a],
                                  kwargs={"head_path": head_path, "steps": steps,
                                          "ckpt_steps": ckpt_steps, "data_seed": data_seed,
                                          "tag": tag}))


@app.function(volumes={DATA_DIR: volume}, timeout=6 * 3600, memory=2048)
def readout_arms(tag: str = "c1", arms: str = ",".join(ARMS),
                 dual_steps: str = "12000,24000", full_steps: str = "24000",
                 key: str = "v16_s2_L6_m4_distinct"):
    """CPU coordinator: the standard instruments on every continued arm.

    `dual_ckpt` (altitude, unchanged) on the corrupted window set for the kappa* trace;
    `calibration_ckpt` (parent, unchanged) on clean windows for CE / KL(p_k||q) / by-level;
    `prize_ckpt` (Q1) for the post-event excess profile."""
    volume.reload()
    root = f"{DATA_DIR}/{key}/logit_reading/coeruleus/cont_{tag}"
    a = arms.split(",")
    dual = [f"{root}/{arm}/step{int(s):06d}.pt" for arm in a for s in dual_steps.split(",")]
    full = [f"{root}/{arm}/step{int(s):06d}.pt" for arm in a for s in full_steps.split(",")]
    print(f"dual on {len(dual)} ckpts, calibration+prize on {len(full)}", flush=True)
    n1 = len(list(dual_ckpt.map(dual, kwargs={"ws": "noisy0.01", "eps_obs": 0.01})))
    n2 = len(list(calibration_ckpt.map(full, kwargs={"n_windows": 4096, "save_arrays": False})))
    n3 = len(list(prize_ckpt.map(full, kwargs={"eps_data": 0.01, "horizon": 8})))
    return {"dual": n1, "calibration": n2, "prize": n3}
