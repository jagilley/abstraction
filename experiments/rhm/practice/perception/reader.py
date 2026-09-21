"""The next-token reader whose entropy period is the endogenous grid.

`logit_reading/train_trajectory.py`'s recipe, transcribed with ONE parameter changed and
everything else line-for-line: architecture 8L/8H/256D, AdamW lr 3e-4 wd 0.01, batch 64,
no LR schedule, flat windows cut at uniformly random offsets from a stream of concatenated
sequences, checkpoints at a fixed step ladder.

THE ONE CHANGE, and why it could not be avoided. `train_trajectory` hard-wires the window
length to `T = s**depth`, which on the practice grammar is 64 -- exactly ONE period of the
deepest level. The nested detector scores the depth-k boundary by pooling window columns
`j == r (mod 2^k)`, so at k = 6 a 64-token window gives it a single column per candidate
offset, and a piece presented at a random offset is never wholly inside the window. `W` is
therefore a parameter here and is set to 128 (two periods), which is also what lets one
solved piece sit entirely inside one window at any offset.

TWO ARMS, the two streams a learner could be reading:
  corpus  fresh samples of the practice grammar, regenerated every chunk (a learning
          curve, not a memorisation curve) -- `train_trajectory`'s own diet.
  own     the practice learner's OWN successful beam tips, concatenated in a fixed random
          order and re-read. This is the stream a learner actually has; its re-read factor
          is logged, because `reread/lm` says that is the axis a frozen corpus stalls on.

Run (from experiments/):
  modal run --detach -m rhm.practice.perception.reader::train_reader --tag pc0 --arm corpus
"""

import json
import os
import time

import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, volume
from rhm.practice.perception.common import (
    DATA_DIR as _DD, DEPTH, M, REMOTE, RULE_SEED, S, T_SEQ, V, app)

CKPTS = "0,250,1000,2000,4000,8000,16000,32000,64000"


def build_stream(arm, tag, rules, n_seqs, rng):
    """Returns (stream uint8 flat, meta). `own` loads the recorded productions."""
    from rhm.logit_reading.grammar import generate
    if arm == "corpus":
        seqs = generate(rules, None, n_seqs, rng)
        return seqs.reshape(-1).astype(np.uint8), {"source": "grammar", "n_seqs": n_seqs}
    z = np.load(f"{DATA_DIR}/{REMOTE}/{tag}/pieces.npz")
    x = z["solved_x"]
    order = rng.permutation(x.shape[0])
    return (x[order].reshape(-1).astype(np.uint8),
            {"source": "own_solved_beam_tips", "n_seqs": int(x.shape[0]),
             "n_cycles": int(z["n_cycles"])})


@app.function(volumes={_DD: volume}, gpu="L4", timeout=6 * 3600, memory=8192)
def train_reader(tag: str = "pc0", arm: str = "corpus", window: int = 128,
                 steps: int = 64000, batch_size: int = 64, lr: float = 3e-4,
                 weight_decay: float = 0.01, n_layer: int = 8, n_head: int = 8,
                 n_embd: int = 256, chunk_seqs: int = 1_000_000, data_seed: int = 7,
                 seed: int = 42, ckpt_steps: str = CKPTS, log_interval: int = 1000):
    import resource
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct

    volume.reload()
    device = "cuda"
    W = int(window)
    rules = generate_rules_distinct(V, S, DEPTH, M, seed=RULE_SEED)
    out_dir = f"{DATA_DIR}/{REMOTE}/{tag}/reader_{arm}"
    os.makedirs(out_dir, exist_ok=True)
    ckpts = sorted({int(x) for x in ckpt_steps.split(",")})
    cfg = dict(v=V, s=S, L=DEPTH, m=M, rule_seed=RULE_SEED, alpha=0.0, arm=arm, window=W,
               n_layer=n_layer, n_head=n_head, n_embd=n_embd, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, chunk_seqs=chunk_seqs, data_seed=data_seed,
               seed=seed, steps=steps)
    print(f"READER {arm} W={W} -> {out_dir}\n  {cfg}", flush=True)

    torch.manual_seed(seed)
    model = GPT(V, W, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    data_rng = np.random.default_rng(data_seed)
    arangeW = torch.arange(W + 1, device=device)

    smeta = {}

    def new_stream():
        t0 = time.time()
        st_np, meta = build_stream(arm, tag, rules, chunk_seqs, data_rng)
        smeta.update(meta)
        st = torch.as_tensor(st_np, device=device)
        print(f"  stream {meta} {st.shape[0]:,} tokens {time.time() - t0:.1f}s", flush=True)
        return st

    stream = new_stream()
    # one pass per chunk, in TOKENS (the donor counted in sequences at T = W)
    steps_per_chunk = max(1, (stream.shape[0]) // (batch_size * W))
    if arm == "own":
        steps_per_chunk = steps + 1              # frozen corpus: never redraw
    cfg["steps_per_chunk"] = int(steps_per_chunk)
    cfg["reread_factor"] = float(steps * batch_size * W / max(stream.shape[0], 1))
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
        ix = torch.randint(0, stream.shape[0] - W - 1, (batch_size,), generator=gen).to(device)
        w = stream[ix[:, None] + arangeW[None, :]].long()
        x, y = w[:, :W], w[:, 1:]
        model.train()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_interval == 0:
            rec = {"step": step, "loss": float(np.mean(losses[-log_interval:])),
                   "elapsed_s": time.time() - t_start}
            log.append(rec)
            print(f"  step {step:6d}  loss {rec['loss']:.4f}  {rec['elapsed_s']:.0f}s",
                  flush=True)

    cfg["stream_meta"] = smeta
    cfg["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    with open(f"{out_dir}/train_log.json", "w") as f:
        json.dump({"config": cfg, "log": log, "ckpt_steps": ckpts}, f,
                  cls=NumpyEncoder, indent=1)
    volume.commit()
    print(f"done in {time.time() - t_start:.0f}s -> {out_dir}", flush=True)
    return {"out_dir": out_dir, "config": cfg}
