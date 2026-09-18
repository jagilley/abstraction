"""regime/train.py -- one next-token trajectory per world.

`altitude/train_noisy.py`'s recipe (8L/8H/256D, AdamW 3e-4, wd 0.01, batch 64,
Dirichlet(1) synonym weights, fresh data every chunk, the same 13-checkpoint ladder) with
the regime process of `world.py` in place of the i.i.d. eps. The corruption is applied to
the STREAM before it is windowed -- which, because the regime chain is started stationary,
is the same distribution as the per-window path the evaluation venue carries -- and before
the (x, y) split, so the model predicts the corrupted token exactly as `train_noisy` does.

  --world burst   the bursty world: an illegal token is evidence about a hidden state
                  that predicts more corruption, and so more cost, over the next tokens
  --world iid     the same marginal corruption rate with no regime (the control)

`world = iid` corrupts i.i.d. at eps_mean and is `train_noisy` with a different eps; it is
rebuilt here rather than read from the banked `traj_eps01_s42` so that both worlds share
this module's RNG discipline and the rates match to four digits.

Run:
  modal run -m rhm.logit_reading.orbitofrontal.regime.train::train_regime \
      --world burst --steps 300 --ckpt-steps 0,300 --chunk-seqs 20000 --tag smoke_burst
  modal run --detach -m rhm.logit_reading.orbitofrontal.regime.train::train_sweep
"""

import json
import os
import resource
import time

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.orbitofrontal.regime.world import app, regime_path, world_params
from rhm.logit_reading.calibration import tb_key

CKPTS = "0,250,500,1000,2000,4000,8000,12000,16000,24000,32000,48000,64000"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=6 * 3600, memory=12288,
              max_containers=4)
def train_regime(
    world: str = "burst", v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
    rule_seed: int = 0, alpha: float = 1.0, weight_seed: int = 1,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    steps: int = 64000, batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
    chunk_seqs: int = 1_000_000, data_seed: int = 7, seed: int = 42,
    ckpt_steps: str = CKPTS, log_interval: int = 1000, tag: str = "",
):
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights, generate

    volume.reload()
    P = world_params(world)
    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    out_dir = f"{DATA_DIR}/{key}/logit_reading/traj_regime_{world}" + (f"_{tag}" if tag else "")
    os.makedirs(out_dir, exist_ok=True)
    ckpts = sorted({int(x) for x in ckpt_steps.split(",")})
    cfg = dict(v=v, s=s, L=L, m=m, rule_seed=rule_seed, alpha=alpha, weight_seed=weight_seed,
               world=world, **{f"w_{k}": vv for k, vv in P.items()},
               n_layer=n_layer, n_head=n_head, n_embd=n_embd, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, chunk_seqs=chunk_seqs, data_seed=data_seed, seed=seed)
    print(f"REGIME TRAJECTORY {key} world={world} -> {out_dir}\n  {cfg}", flush=True)

    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed + 1)
    data_rng = np.random.default_rng(data_seed)
    noise_rng = np.random.default_rng(seed + 1000)
    arangeT = torch.arange(T + 1, device=device)

    stats = {"n_tok": 0, "n_fire": 0, "n_noisy": 0}

    def new_stream():
        t0 = time.time()
        seqs = generate(rules, rule_w, chunk_seqs, data_rng)
        flat = seqs.reshape(-1).astype(np.uint8)
        del seqs
        reg = regime_path(len(flat), P["dwell_c"], P["dwell_n"], noise_rng)
        n_fire = 0
        for c0 in range(0, len(flat), 8_000_000):          # keep the RNG buffers small
            sl = slice(c0, min(c0 + 8_000_000, len(flat)))
            eps = np.where(reg[sl] == 1, np.float32(P["eps_n"]), np.float32(P["eps_c"]))
            fire = noise_rng.random(sl.stop - sl.start, dtype=np.float32) < eps
            k = int(fire.sum())
            if k:
                blk = flat[sl]
                blk[fire] = noise_rng.integers(0, v, k).astype(np.uint8)
                flat[sl] = blk
            n_fire += k
        stats["n_tok"] += len(flat)
        stats["n_fire"] += n_fire
        stats["n_noisy"] += int(reg.sum())
        st = torch.as_tensor(flat, device=device)
        print(f"  new data chunk ({chunk_seqs:,} seqs) fire {n_fire / len(flat):.5f} "
              f"noisy {reg.mean():.4f} {time.time() - t0:.1f}s", flush=True)
        return st

    stream = new_stream()
    steps_per_chunk = max(1, chunk_seqs // batch_size)
    losses, log = [], []

    def save(step):
        model.eval()
        torch.save({"model": model.state_dict(), "config": {**cfg, "step": step},
                    "recent_train_loss": float(np.mean(losses[-200:])) if losses else None},
                   f"{out_dir}/step{step:06d}.pt")
        volume.commit()
        model.train()

    t_start = time.time()
    for step in range(steps + 1):
        if step in ckpts:
            save(step)
        if step == steps:
            break
        if step > 0 and step % steps_per_chunk == 0:
            stream = new_stream()
        ix = torch.randint(0, stream.shape[0] - T - 1, (batch_size,), generator=gen).to(device)
        w = stream[ix[:, None] + arangeT[None, :]].long()
        x, y = w[:, :T], w[:, 1:]
        model.train()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, v), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_interval == 0:
            rec = {"step": step, "loss": float(np.mean(losses[-log_interval:])),
                   "elapsed_s": time.time() - t_start,
                   "fire_rate": stats["n_fire"] / max(stats["n_tok"], 1),
                   "noisy_rate": stats["n_noisy"] / max(stats["n_tok"], 1)}
            log.append(rec)
            print(f"  step {step:6d}  loss {rec['loss']:.4f}  fire {rec['fire_rate']:.5f}  "
                  f"noisy {rec['noisy_rate']:.4f}  {rec['elapsed_s']:.0f}s", flush=True)

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump({"config": cfg, "log": log, "ckpt_steps": ckpts}, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"done in {time.time() - t_start:.0f}s  peak RSS {rss:.1f} GB -> {out_dir}", flush=True)
    return out_dir


@app.function(volumes={DATA_DIR: volume}, timeout=8 * 3600, memory=2048)
def train_sweep(worlds: str = "burst,iid", steps: int = 64000, tag: str = "",
                ckpt_steps: str = CKPTS, chunk_seqs: int = 1_000_000):
    volume.reload()
    args = [(w, 16, 2, 6, 4, 0, 1.0, 1, 8, 8, 256, steps, 64, 3e-4, 0.01, chunk_seqs,
             7, 42, ckpt_steps, 1000, tag) for w in worlds.split(",")]
    outs = list(train_regime.starmap(args, return_exceptions=True))
    for o in outs:
        print(str(o)[:400], flush=True)
    return [str(o)[:400] for o in outs]
