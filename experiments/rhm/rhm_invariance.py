"""Thread B+ : self-supervised INVARIANCE objective -- cluster-bottleneck + Sinkhorn.

Goal: reproduce oracle-aux's lift (recover the latent hierarchy toward the BP ceiling)
WITHOUT privileged labels, by rewarding synonym-invariance instead of surface tokens
(diluted at high levels; see RHM_DEEP_COMPOSITION_README.md).

Design journey (each step diagnosed a concrete failure mode):
  - surface masked-span prediction       -> diluted, like NTP (rhm_masked_span).
  - contrastive, overlapping full views  -> global-fingerprint shortcut.
  - contrastive, disjoint views          -> cold start (uncorrelated random reps).
  - disjoint + MLM bootstrap             -> out-of-distribution teacher view.
  - DINO cluster bottleneck (centering+sharpening) -> WORKS early (d1 0.93, d2 0.58,
    d3 0.21 at step 2000) then COLLAPSES (assignment entropy 3.57->0.58, probe -> chance).

This version keeps the winning ingredients and fixes collapse with SwAV's Sinkhorn:
  - student: full seq with the subtree MASKED (+ scattered MLM) -> infer parent from
    context. teacher (EMA): full UNMASKED seq -> sees the subtree. (in-distribution,
    overlapping -> bootstraps.)
  - target = K-way CLUSTER ASSIGNMENT (prototype head). The ~log2(K) bottleneck blocks
    the global-sequence-fingerprint shortcut; the only context-predictable low-dim
    feature is the parent.
  - **Sinkhorn-Knopp** balanced assignment on the teacher logits: optimal transport
    enforcing equipartition (every cluster used equally across the batch) -> the
    all-samples-one-cluster collapse is structurally impossible (replaces the weaker
    DINO centering+sharpening that let it collapse).
  - same aligned subtree position per batch -> kills the positional shortcut.
  - dense MLM bootstrap (the only-learnable-from-scratch d1 signal).
  - mask SIZE k sweeps levels d1..d5 (root d6 only indirect; its BP ceiling ~0.80).

Probe is identical to Thread B / oracle-aux (last-token, linear + MLP, per block).

Run:
  modal run --detach -m rhm.rhm_invariance::invariance
  modal run --detach -m rhm.rhm_invariance::invariance --n-steps 1500 \
      --pool-size 40000 --ckpt-steps "0,1499" --eval-interval 250 \
      --n-eval-sequences 3000 --probe-steps 200 --mlp-steps 200    # smoke test
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_thread_b import tb_key, _build_eval, _probe_checkpoints, _probe_acc

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)
app = modal.App("rhm-invariance", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=21600, memory=32768)
def invariance(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0, seq_seed: int = 1,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    batch_size: int = 128, lr: float = 3e-4, weight_decay: float = 0.01,
    n_steps: int = 20000, pool_size: int = 200000, ema_m: float = 0.99,
    n_proto: int = 64, d_proj: int = 64, tau_s: float = 0.1, sink_eps: float = 0.05, sink_iter: int = 3,
    lam_dino: float = 1.0, mlm_rate: float = 0.15,
    eval_interval: int = 1000, ckpt_steps: str = "0,5000,20000",
    n_eval_sequences: int = 12000, probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, eval_seed: int = 999,
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct, generate_sequences_batched

    class Head(nn.Module):                       # projection + cosine to K prototypes (bottleneck)
        def __init__(self, d, dp, k):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, dp))
            self.proto = nn.Linear(dp, k, bias=False)

        def forward(self, x):
            z = F.normalize(self.mlp(x), dim=-1)
            w = F.normalize(self.proto.weight, dim=1)
            return z @ w.t()

    @torch.no_grad()
    def sinkhorn(scores):                        # SwAV: balanced (equipartition) soft assignment
        Q = torch.exp(scores / sink_eps).t()     # (K, B)
        Q /= Q.sum()
        nK, nB = Q.shape
        for _ in range(sink_iter):
            Q /= Q.sum(dim=1, keepdim=True); Q /= nK
            Q /= Q.sum(dim=0, keepdim=True); Q /= nB
        return (Q * nB).t()                      # (B, K), each row (sample) sums to 1

    L = depth
    T = s ** L
    MASK = v
    device = "cuda"
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    last_block = f"post_block{n_layer - 1}"
    want_ckpts = sorted({min(int(x), n_steps - 1) for x in ckpt_steps.split(",")})

    print(f"invariance(Sinkhorn): {key}  {n_layer}L/{n_head}H/{n_embd}D  K={n_proto}  "
          f"tau_s={tau_s}  eps={sink_eps}  ema={ema_m}  lam={lam_dino}  B={batch_size}")
    pool = torch.from_numpy(generate_sequences_batched(rules, pool_size, seed=seq_seed).astype(np.int64))
    eval_x, y_level, block_names = _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device)

    online = GPT(v + 1, T, n_layer, n_head, n_embd, causal=False).to(device)
    teacher = GPT(v + 1, T, n_layer, n_head, n_embd, causal=False).to(device)
    teacher.load_state_dict(online.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    head_o = Head(n_embd, d_proj, n_proto).to(device)
    head_t = Head(n_embd, d_proj, n_proto).to(device)
    head_t.load_state_dict(head_o.state_dict())
    for p in head_t.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(list(online.parameters()) + list(head_o.parameters()),
                            lr=lr, weight_decay=weight_decay)
    ckpt_dir = f"{DATA_DIR}/{key}/invariance_{n_layer}L{n_head}H{n_embd}D"
    os.makedirs(ckpt_dir, exist_ok=True)

    @torch.no_grad()
    def ema_update():
        for pt, po in zip(teacher.parameters(), online.parameters()):
            pt.mul_(ema_m).add_(po.detach(), alpha=1 - ema_m)
        for pt, po in zip(head_t.parameters(), head_o.parameters()):
            pt.mul_(ema_m).add_(po.detach(), alpha=1 - ema_m)

    saved = []
    for step in range(n_steps):
        if step in want_ckpts:
            torch.save(online.state_dict(), f"{ckpt_dir}/ckpt_step{step}.pt"); saved.append(step)
        online.train()
        k = int(torch.randint(1, L, (1,)))
        nmask = s ** (L - k)
        j = int(torch.randint(0, s ** k, (1,)))
        a, b = j * nmask, j * nmask + nmask
        idx = torch.randint(0, pool_size, (batch_size,))
        x = pool[idx].to(device)
        scat = torch.rand(batch_size, T, device=device) < mlm_rate
        scat[:, a:b] = False
        x_ctx = x.clone(); x_ctx[:, a:b] = MASK; x_ctx[scat] = MASK     # student: context (subtree hidden)

        out_logits, _, io_s = online(x_ctx, return_intermediates=True)
        mlm_mask = scat.clone(); mlm_mask[:, a:b] = True               # MLM bootstrap
        mlm = F.cross_entropy(out_logits[mlm_mask][:, :v], x[mlm_mask])
        logits_s = head_o(io_s[last_block][:, a:b, :].mean(1))         # student cluster logits (from context)
        with torch.no_grad():
            _, _, io_t = teacher(x, return_intermediates=True)         # teacher: full UNMASKED
            logits_t = head_t(io_t[last_block][:, a:b, :].mean(1))
            q = sinkhorn(logits_t)                                     # balanced target (no collapse)
        dino = -(q * F.log_softmax(logits_s / tau_s, dim=-1)).sum(-1).mean()
        loss = mlm + lam_dino * dino

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(online.parameters()) + list(head_o.parameters()), 1.0)
        opt.step(); ema_update()

        if step % eval_interval == 0 or step == n_steps - 1:
            usage = float(-(q.mean(0) * (q.mean(0) + 1e-9).log()).sum())   # batch cluster-usage entropy
            agree = float((logits_s.argmax(-1) == logits_t.argmax(-1)).float().mean())
            print(f"  step {step:6d}: mlm={float(mlm):.3f} dino={float(dino):.3f} "
                  f"usage={usage:.2f}/{np.log(n_proto):.2f} agree={agree:.3f} (k={k}, mask={nmask})")
    torch.save(online.state_dict(), f"{ckpt_dir}/ckpt_step{n_steps-1}.pt")
    saved = sorted(set(saved + [n_steps - 1]))
    volume.commit()

    probes = _probe_checkpoints(online, ckpt_dir, saved, eval_x, y_level, block_names,
                                L, v, device, probe_steps, probe_lr, mlp_hidden, mlp_steps)

    print(f"\n{'='*72}\nINVARIANCE(Sinkhorn) final best-probe by depth (chance={1.0/v:.4f})")
    last = max(probes.keys())
    for kind in ["linear_best", "mlp_best"]:
        best = probes[last][kind]
        print(f"  {kind:>11}: " + "  ".join(f"d{L-ell}:{best[ell]:.3f}" for ell in range(L)))

    save_dir = f"{DATA_DIR}/rhm_thread_b"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"invariance_{key}_{n_layer}L{n_head}H{n_embd}D.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump({"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_steps=n_steps,
                                  K=n_proto, ema_m=ema_m, tau_s=tau_s, sink_eps=sink_eps,
                                  lam_dino=lam_dino, chance=1.0 / v), "checkpoints": probes},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return probes


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=7200, memory=32768)
def probe_masked(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    n_eval_sequences: int = 12000, probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, eval_seed: int = 999,
):
    """IN-DISTRIBUTION probe of the invariance checkpoints: the model is trained on
    MASKED inputs, so probing it on unmasked inputs is OOD and under-reports. Here we
    mask the level-ell subtree ending at the last leaf, encode (matching training),
    pool the model's inferred-parent representation at the masked positions, and probe
    it for that subtree's true parent (= y_level[ell]). Measures 'infer the level-ell
    parent from context' directly. chance = 1/v.
    """
    import glob
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct

    L = depth
    T = s ** L
    MASK = v
    device = "cuda"
    key = tb_key(v, s, L, m)
    ckpt_dir = f"{DATA_DIR}/{key}/invariance_{n_layer}L{n_head}H{n_embd}D"
    volume.reload()
    saved = sorted(int(p.split("step")[1].split(".")[0])
                   for p in glob.glob(f"{ckpt_dir}/ckpt_step*.pt"))
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    model = GPT(v + 1, T, n_layer, n_head, n_embd, causal=False).to(device)
    eval_x, y_level, block_names = _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device)

    print(f"probe_masked (in-distribution): {key}  checkpoints={saved}  chance={1.0/v:.4f}")
    print(f"  (mask the level-ell subtree at the end; probe the inferred-parent rep)")
    out = {}
    for step in saved:
        model.load_state_dict(torch.load(f"{ckpt_dir}/ckpt_step{step}.pt",
                                         map_location=device, weights_only=True))
        model.eval()
        lin, mlp = {}, {}
        for ell in range(1, L):                      # levels 1..L-1  (d = L-ell = 1..5)
            nmask = s ** (L - ell)
            a = T - nmask                            # subtree ending at the last leaf
            acts = {bk: [] for bk in block_names}
            with torch.no_grad():
                for i in range(0, len(eval_x), 256):
                    xm = eval_x[i:i + 256].clone(); xm[:, a:T] = MASK
                    _, _, inter = model(xm, return_intermediates=True)
                    for bk in block_names:
                        acts[bk].append(inter[bk][:, a:T, :].mean(1).float())
            acts = {bk: torch.cat(acts[bk], 0) for bk in block_names}
            lin[ell] = max(_probe_acc(acts[bk], y_level[ell], v, device, probe_steps, probe_lr)[0]
                           for bk in block_names)
            mlp[ell] = max(_probe_acc(acts[bk], y_level[ell], v, device, mlp_steps, 1e-3,
                                      hidden=mlp_hidden)[0] for bk in block_names)
        out[step] = {"linear_best": lin, "mlp_best": mlp}
        print(f"  step {step:6d} | LIN " + "  ".join(f"d{L-ell}:{lin[ell]:.3f}" for ell in range(1, L)))
        print(f"  step {step:6d} | MLP " + "  ".join(f"d{L-ell}:{mlp[ell]:.3f}" for ell in range(1, L)))

    save_dir = f"{DATA_DIR}/rhm_thread_b"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/invariance_probemasked_{key}_{n_layer}L{n_head}H{n_embd}D.json", "w") as f:
        json.dump({"config": dict(v=v, s=s, L=L, m=m, chance=1.0 / v), "checkpoints": out},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out
