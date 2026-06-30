"""Thread B+ (constructive): self-supervised masked-span objective.

Oracle-aux (rhm_thread_b.py::oracle_aux) proved the architecture CAN represent the
whole hierarchy when given the latent signal directly (privileged). This tests
whether a SELF-SUPERVISED objective supplies the same signal without labels.

Objective: a BIDIRECTIONAL encoder (model.py GPT with causal=False) trained with
masked-span prediction -- mask a contiguous, tree-aligned SUBTREE and predict its
leaves from the rest of the sequence. Filling a masked subtree forces inferring its
latent root from the surrounding (sibling) subtrees, and the mask SIZE selects which
level is forced: masking a level-ell subtree (s^(L-ell) leaves) targets the level-ell
latent (depth d = L-ell). We sample ell uniformly in [1, L-1], so the signal spans
d1..d5 (the root d6 is only driven indirectly -- expected, its BP ceiling is ~0.80).

Two conditions, identical except masking geometry (controlled comparison):
  - span     : mask one aligned subtree of a random level (contiguous).
  - scattered: mask the SAME number of positions, but random/scattered (standard MLM).
If span tracks BP at high levels while scattered saturates like NTP, then contiguous
span masking -- forcing high-level inference -- is the active ingredient.

Probe + reference lines are identical to Thread B / oracle-aux (last-token, linear +
MLP, vs greedy floor and BP ceiling), so the numbers drop straight into that table.

Run:
  modal run --detach -m rhm.rhm_masked_span::masked_span
  modal run --detach -m rhm.rhm_masked_span::masked_span --n-steps 200 \
      --pool-size 20000 --ckpt-steps "0,199" --n-eval-sequences 3000 \
      --probe-steps 200 --mlp-steps 200            # smoke test
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder
from rhm.rhm_thread_b import tb_key, _build_eval, _probe_checkpoints

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)
app = modal.App("rhm-masked-span", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=21600, memory=32768)
def masked_span(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0, seq_seed: int = 1,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
    n_steps: int = 20000, pool_size: int = 200000, eval_interval: int = 1000,
    ckpt_steps: str = "0,5000,20000",
    n_eval_sequences: int = 12000, probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, eval_seed: int = 999,
):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct, generate_sequences_batched

    L = depth
    T = s ** L
    MASK = v                       # extra vocab id for the mask token
    device = "cuda"
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    want_ckpts = sorted({min(int(x), n_steps - 1) for x in ckpt_steps.split(",")})

    # training pool: aligned leaf sequences (latents only needed at eval)
    print(f"masked_span: {key}  {n_layer}L/{n_head}H/{n_embd}D  pool={pool_size:,}  T={T}")
    n_seq = (pool_size + 1)
    pool = torch.from_numpy(generate_sequences_batched(rules, n_seq, seed=seq_seed).astype(np.int64))
    eval_x, y_level, block_names = _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device)

    def make_masks(B):
        """Per-example: pick a level ell in [1,L-1] -> mask n=s^(L-ell) positions,
        contiguous aligned subtree (span) or scattered (same count)."""
        mask_span = torch.zeros(B, T, dtype=torch.bool)
        mask_scat = torch.zeros(B, T, dtype=torch.bool)
        for i in range(B):
            ell = int(torch.randint(1, L, (1,)))           # 1..L-1
            n = s ** (L - ell)
            j = int(torch.randint(0, s ** ell, (1,)))       # which aligned subtree
            mask_span[i, j * n:(j + 1) * n] = True
            mask_scat[i, torch.randperm(T)[:n]] = True
        return {"span": mask_span, "scattered": mask_scat}

    cond_results = {}
    for cond in ["span", "scattered"]:
        torch.manual_seed(0)
        model = GPT(v + 1, T, n_layer, n_head, n_embd, causal=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        ckpt_dir = f"{DATA_DIR}/{key}/masked_{cond}_{n_layer}L{n_head}H{n_embd}D"
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\n--- training masked-{cond} (bidirectional) ---")
        saved = []
        for step in range(n_steps):
            if step in want_ckpts:
                torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{step}.pt"); saved.append(step)
            model.train()
            idx = torch.randint(0, pool_size, (batch_size,))
            x = pool[idx].to(device)
            mask = make_masks(batch_size)[cond].to(device)
            x_in = x.clone(); x_in[mask] = MASK
            logits, _ = model(x_in)
            loss = F.cross_entropy(logits[mask][:, :v], x[mask])   # predict original leaves
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % eval_interval == 0 or step == n_steps - 1:
                print(f"  step {step:6d}: mlm_loss={float(loss):.4f}")
        torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{n_steps-1}.pt")
        saved = sorted(set(saved + [n_steps - 1]))
        volume.commit()
        cond_results[cond] = _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level,
                                                 block_names, L, v, device, probe_steps,
                                                 probe_lr, mlp_hidden, mlp_steps)

    print(f"\n{'='*72}\nMASKED-SPAN vs SCATTERED: final best-probe by depth (chance={1.0/v:.4f})")
    for kind in ["linear_best", "mlp_best"]:
        print(f"\n[{kind}]   (level ell -> depth d = L-ell)")
        print(f"{'cond':>10} | " + "  ".join(f"d{L-ell}" for ell in range(L)))
        for cond in ["span", "scattered"]:
            last = max(cond_results[cond].keys())
            best = cond_results[cond][last][kind]
            print(f"{cond:>10} | " + "  ".join(f"{best[ell]:.3f}" for ell in range(L)))

    save_dir = f"{DATA_DIR}/rhm_thread_b"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"masked_span_{key}_{n_layer}L{n_head}H{n_embd}D.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump({"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_steps=n_steps,
                                  chance=1.0 / v, bidirectional=True), "conditions": cond_results},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return cond_results
