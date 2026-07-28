"""Verify the multi-channel distractor DGP has the properties the E3 ladder needs.

The channel construction itself is the shared primitive in `../rhm_channels.py`; this is
the experiment that certifies it. Five properties:

  P1 structural irrelevance -- distractor content cannot move the parsed root or the DP d*.
  P2 exact irreducibility   -- noise-channel CE pinned at its information-theoretic floor.
  P3 `error-only` trap      -- noise CE >= every other channel at every checkpoint.
  P4 `lprog-only` trap      -- struct channels genuinely learnable; each channel's
                               saturation step sets how fast drift must arrive to keep
                               allocation zero-sum (see `../rhm_drift.py`).
  P5 marginal matching      -- channels not separable by unigram statistics.

Run from experiments/:
  modal run rhm/directed_sculpting/verify_distractors.py::verify_distractors --quick
  modal run --detach rhm/directed_sculpting/verify_distractors.py::verify_distractors
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_channels import (
    DEFAULT_SPEC, block_channel_ids, check_marginals, check_structural_irrelevance,
    make_layout, sample_pool, token_channel_ids)
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume


app = modal.App("rhm-verify-distractors", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=16384)
def verify_distractors(
    v: int = 8, s: int = 2, tree_depth: int = 5, tree_m: int = 2,
    struct_depths: str = "3,3", struct_ms: str = "2,4", n_noise_blocks: str = "2,2",
    n_train: int = 400_000, n_eval: int = 4_096, n_steps: int = 20_000,
    batch_size: int = 128, lr: float = 3e-4, n_layer: int = 6, n_head: int = 6,
    n_embd: int = 192, eval_every: int = 500, seed: int = 1, tag: str = "v1",
    lp_warmup_evals: int = 2, quick: bool = False,
):
    """Verify the distractor channels have the properties the E3 ladder needs.

    P1 structural irrelevance -- distractor content cannot move the root or the DP cost.
    P2 exact irreducibility   -- noise-channel CE pinned at log(v), slope ~ 0.
    P3 `error-only` trap      -- noise CE >= tree CE at every checkpoint.
    P4 `lprog-only` trap      -- struct channels genuinely learnable; report each
                                 channel's saturation step, which sets the drift cadence
                                 the eventual run needs to keep allocation zero-sum.
    P5 marginal matching      -- channels not separable by unigram statistics.
    """
    import time

    import torch
    import torch.nn.functional as F

    from rhm.model import GPT

    if quick:
        n_train, n_eval, n_steps, eval_every = 20_000, 1_024, 600, 50

    spec = [{"kind": "tree", "name": "tree", "depth": tree_depth, "m": tree_m,
             "rule_seed": 0}]
    for i, (d, m) in enumerate(zip(struct_depths.split(","), struct_ms.split(","))):
        spec.append({"kind": "struct", "name": f"struct{chr(65 + i)}", "depth": int(d),
                     "m": int(m), "rule_seed": 101 + i})
    for i, nb in enumerate(n_noise_blocks.split(",")):
        spec.append({"kind": "noise", "name": f"noise{chr(65 + i)}",
                     "n_blocks": int(nb)})

    layout = make_layout(v, s, spec)
    names = [c["name"] for c in layout["channels"]]
    print(f"Layout: total_len={layout['total_len']} tokens, "
          f"{layout['n_blocks_total']} blocks over {len(names)} channels")
    for ch in layout["channels"]:
        print(f"  {ch['name']:9s} {ch['kind']:6s} tokens[{ch['tok0']:3d}:{ch['tok1']:3d}] "
              f"blocks[{ch['blk0']:2d}:{ch['blk1']:2d}]"
              + (f"  depth={ch['depth']} m={ch['m']}" if ch["rules"] is not None else ""))

    started = time.time()
    results = {"layout": {"total_len": layout["total_len"],
                          "n_blocks_total": layout["n_blocks_total"],
                          "channels": [{k: ch[k] for k in
                                        ("name", "kind", "depth", "m", "tok0", "tok1",
                                         "blk0", "blk1")}
                                       for ch in layout["channels"]]}}

    # -- P1 -------------------------------------------------------------------
    print("\n[P1] structural irrelevance")
    p1 = check_structural_irrelevance(layout)
    results["p1_structural_irrelevance"] = p1
    print(f"  scrambled {p1['n_off_tree_tokens']} non-tree tokens/sequence -> "
          f"roots identical: {p1['roots_identical']}, valid identical: "
          f"{p1['valid_identical']}, DP d* identical: {p1['dp_identical']} "
          f"(mean d*={p1['mean_dp_cost']:.2f})")

    # -- P5 -------------------------------------------------------------------
    print("\n[P5] unigram statistics per channel")
    p5 = check_marginals(layout)
    results["p5_marginals"] = p5
    print(f"  uniform entropy = {p5['_uniform_entropy_nats']:.4f} nats")
    for n in names:
        print(f"  {n:9s} H={p5[n]['unigram_entropy_nats']:.4f}  "
              f"TV(uniform)={p5[n]['tv_from_uniform']:.4f}")

    # -- P2/P3/P4: per-channel learning curves --------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n[P2/P3/P4] per-channel CE over training ({n_steps} steps, device={device})")

    train = torch.from_numpy(sample_pool(layout, n_train, seed)["leaves"])
    evals = torch.from_numpy(sample_pool(layout, n_eval, seed + 99)["leaves"]).to(device)
    tok_ids = torch.from_numpy(token_channel_ids(layout)).to(device)

    T = layout["total_len"]
    model = GPT(vocab_size=v, block_size=T, n_layer=n_layer, n_head=n_head,
                n_embd=n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def per_channel_ce():
        """Held-out CE per channel. Aligned (whole-sequence) batching, so a position's
        channel is well defined -- the readout the specialization harness cannot use."""
        model.eval()
        tot = torch.zeros(len(names), device=device)
        cnt = torch.zeros(len(names), device=device)
        with torch.no_grad():
            for i in range(0, evals.shape[0], 256):
                x = evals[i:i + 256]
                logits, _ = model(x[:, :-1])
                ce = F.cross_entropy(logits.reshape(-1, v), x[:, 1:].reshape(-1),
                                     reduction="none").view(x.shape[0], -1)
                ids = tok_ids[1:]                      # channel of the PREDICTED token
                for c in range(len(names)):
                    mask = ids == c
                    if mask.any():
                        tot[c] += ce[:, mask].sum()
                        cnt[c] += mask.sum() * x.shape[0]
        model.train()
        return (tot / cnt.clamp(min=1)).cpu().numpy()

    curve = []
    gen = torch.Generator().manual_seed(seed)
    for step in range(n_steps + 1):
        if step % eval_every == 0:
            ce = per_channel_ce()
            curve.append({"step": step, **{n: float(ce[i]) for i, n in enumerate(names)}})
            print("  step %5d  " % step
                  + "  ".join(f"{n}={ce[i]:.4f}" for i, n in enumerate(names)))
        if step == n_steps:
            break
        idx = torch.randint(0, train.shape[0], (batch_size,), generator=gen)
        x = train[idx].to(device)
        # model.py's forward does targets.view(-1), which rejects a non-contiguous slice
        _, loss = model(x[:, :-1], targets=x[:, 1:].contiguous())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    results["curve"] = curve
    uniform_ce = float(np.log(v))
    # the noise channels' information-theoretic floor: iid draws from `noise_token_p`,
    # so conditional entropy == marginal entropy and no model can beat it.
    np_p = layout["noise_token_p"]
    noise_floor = uniform_ce if np_p is None else float(-(np_p[np_p > 0]
                                                          * np.log(np_p[np_p > 0])).sum())

    # P2 -- is the noise channel exactly irreducible?
    first, last = curve[0], curve[-1]
    p2 = {}
    for n, ch in zip(names, layout["channels"]):
        floor = noise_floor if ch["kind"] == "noise" else None
        p2[n] = {"ce_first": first[n], "ce_last": last[n],
                 "drop": first[n] - last[n],
                 "excess_over_floor": None if floor is None else last[n] - floor}
    results["p2_reducibility"] = {"uniform_ce_nats": uniform_ce,
                                  "noise_floor_nats": noise_floor, "per_channel": p2}

    # Both P3 and P4 must skip the pre-training transient. At step 0 every channel sits at
    # random-init entropy in an arbitrary order, and the first evals are dominated by the
    # model learning the shared unigram marginal (2.15 -> 1.96 for EVERY channel, noise
    # included). Reading either check across that window measures initialisation, not the
    # DGP: it is what made a first pass report P3 FAIL on a single step-0 inversion of
    # -0.016 nats, and credit the noise channels with "LP 3.5/1k".
    warm = curve[lp_warmup_evals:] if len(curve) > lp_warmup_evals + 2 else curve[1:]
    warm_from = warm[0]["step"]

    # P3 -- is noise a strictly attractive target for a raw-error policy?
    noise_names = [c["name"] for c in layout["channels"] if c["kind"] == "noise"]
    other_names = [c["name"] for c in layout["channels"] if c["kind"] != "noise"]
    p3 = {"warmup_from_step": warm_from,
          "noise_ge_all_others_every_checkpoint": all(
              rec[nn] >= rec[on] for rec in warm
              for nn in noise_names for on in other_names),
          "min_margin": float(min(rec[nn] - rec[on] for rec in warm
                                  for nn in noise_names for on in other_names))}
    results["p3_error_only_trap"] = p3

    # P4 -- learning progress per channel, and where each channel satiates.
    # LP is a finite difference of the CE curve (nats per 1k steps). The saturation step
    # is the first checkpoint after which LP stays below `lp_floor` -- it sets how fast
    # drift must arrive to keep allocation zero-sum, which is E3's OU walk's job.
    # `lp_floor` is CALIBRATED off the noise channels rather than hardcoded: noise is
    # irreducible by construction, so its post-warmup LP is pure eval-noise and is the
    # built-in null for "no learning progress here".
    def lp_series(name, recs):
        return [{"step": b["step"],
                 "lp_per_1k": float((a[name] - b[name])
                                    / max(b["step"] - a["step"], 1) * 1000.0)}
                for a, b in zip(recs[:-1], recs[1:])]

    noise_lp = [x["lp_per_1k"] for n in noise_names for x in lp_series(n, warm)]
    lp_floor = float(max(2.0 * max(np.abs(noise_lp)), 1e-4)) if noise_lp else 1e-3
    p4 = {}
    for n in names:
        lp, sat = lp_series(n, warm), None
        for i, rec in enumerate(lp):
            if all(x["lp_per_1k"] < lp_floor for x in lp[i:]):
                sat = rec["step"]
                break
        p4[n] = {"lp_per_1k": lp, "saturation_step": sat,
                 "peak_lp_per_1k": float(max(x["lp_per_1k"] for x in lp))}
    results["p4_lprog_trap"] = {"lp_floor_per_1k": lp_floor, "calibrated_from": noise_names,
                                "warmup_from_step": warm_from, "per_channel": p4}

    print("\n--- verdicts ---")
    print(f"P1 structural irrelevance : "
          f"{'PASS' if all([p1['roots_identical'], p1['valid_identical'], p1['dp_identical']]) else 'FAIL'}")
    print(f"P2 noise irreducible      : floor={noise_floor:.4f} nats; " + ", ".join(
        f"{n} drop={p2[n]['drop']:+.4f} excess={p2[n]['excess_over_floor']:+.4f}"
        for n in noise_names))
    print(f"P3 error-only trap        : "
          f"{'PASS' if p3['noise_ge_all_others_every_checkpoint'] else 'FAIL'} "
          f"(min margin {p3['min_margin']:+.4f} nats, from step {warm_from})")
    print(f"P4 lprog trap / satiation  : (LP floor {lp_floor:.4f}/1k, calibrated off "
          f"{'+'.join(noise_names)}; from step {warm_from})")
    for n in names:
        print(f"     {n:9s} peak LP={p4[n]['peak_lp_per_1k']:.4f}/1k steps  "
              f"saturates at {p4[n]['saturation_step']}")

    results["elapsed_seconds"] = time.time() - started
    out_dir = f"{DATA_DIR}/directed_sculpting/distractor_dgp_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json  ({results['elapsed_seconds']:.0f}s)")
    return results


@app.local_entrypoint()
def main(quick: bool = False, tag: str = "v1", tree_depth: int = 5, n_steps: int = 6000):
    verify_distractors.remote(quick=quick, tag=tag, tree_depth=tree_depth,
                              n_steps=n_steps)
