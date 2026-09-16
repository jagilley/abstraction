"""A checkpointed base-model trajectory trained on a CORRUPTED stream.

Same recipe as `logit_reading/train_trajectory.py` (8L/8H/256D, AdamW 3e-4, wd 0.01,
batch 64, Dirichlet(1) synonym weights, fresh data every chunk, 13 checkpoints to 64k)
with one change: every token of the stream is independently replaced by a uniform draw
with probability `eps_train`, exactly the corruption the `noise_eps` observer in
`flat_oracle.flat_predictive` models. The corruption is applied to the window BEFORE it
is split into (x, y), so the model sees the corrupted token and is asked to predict the
corrupted next token -- which is what the eps-noise observer's predictive is over.

`eps_train = 0` draws no random numbers at all and is bit-identical to
`train_trajectory.train_trajectory` given the same seeds.

Run:
  # smoke (attached, ~2 min)
  modal run -m rhm.logit_reading.altitude.train_noisy::train_noisy \
      --steps 300 --ckpt-steps 0,300 --chunk-seqs 20000 --tag smoke_eps
  # the real thing (~40 min on an L4)
  modal run --detach -m rhm.logit_reading.altitude.train_noisy::train_noisy --tag eps01_s42
"""

import json
import os
import time

import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.logit_reading.calibration import app, tb_key


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=6 * 3600, memory=12288)
def train_noisy(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    alpha: float = 1.0, weight_seed: int = 1, eps_train: float = 0.01,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    steps: int = 64000, batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
    chunk_seqs: int = 1_000_000, data_seed: int = 7, seed: int = 42,
    ckpt_steps: str = "0,250,500,1000,2000,4000,8000,12000,16000,24000,32000,48000,64000",
    log_interval: int = 1000, tag: str = "eps01_s42",
):
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.logit_reading.grammar import synonym_weights, generate

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    rule_w = None if alpha <= 0 else synonym_weights(v, L, m, alpha, weight_seed)
    out_dir = f"{DATA_DIR}/{key}/logit_reading/traj_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    ckpts = sorted({int(x) for x in ckpt_steps.split(",")})
    cfg = dict(v=v, s=s, L=L, m=m, rule_seed=rule_seed, alpha=alpha, weight_seed=weight_seed,
               eps_train=eps_train, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               batch_size=batch_size, lr=lr, weight_decay=weight_decay, chunk_seqs=chunk_seqs,
               data_seed=data_seed, seed=seed)
    print(f"NOISY TRAJECTORY {key} alpha={alpha} eps_train={eps_train} -> {out_dir}\n  {cfg}", flush=True)

    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1000)
    data_rng = np.random.default_rng(data_seed)
    arangeT = torch.arange(T + 1, device=device)

    def new_stream():
        t0 = time.time()
        seqs = generate(rules, rule_w, chunk_seqs, data_rng)
        st = torch.as_tensor(seqs.reshape(-1).astype(np.uint8), device=device)
        print(f"  new data chunk ({chunk_seqs:,} seqs) {time.time() - t0:.1f}s", flush=True)
        return st

    stream = new_stream()
    steps_per_chunk = max(1, (chunk_seqs * T) // (batch_size * T))
    losses, log = [], []
    n_flip = 0

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
        if eps_train > 0:
            flip = torch.rand(w.shape, generator=noise_gen, device=device) < eps_train
            rnd = torch.randint(0, v, w.shape, generator=noise_gen, device=device)
            w = torch.where(flip, rnd, w)
            n_flip += int(flip.sum())
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
                   "flip_rate": (n_flip / (step * batch_size * (T + 1))) if step else 0.0}
            log.append(rec)
            print(f"  step {step:6d}  loss {rec['loss']:.4f}  flip {rec['flip_rate']:.4f}  "
                  f"{rec['elapsed_s']:.0f}s", flush=True)

    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump({"config": cfg, "log": log, "ckpt_steps": ckpts}, f, cls=NumpyEncoder, indent=1)
    volume.commit()
    import resource
    print(f"done in {time.time() - t_start:.0f}s  peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB -> {out_dir}", flush=True)
    return out_dir
