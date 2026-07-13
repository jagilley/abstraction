"""Fresh-FM swap probe -- modular vs entangled dependency (Payoff 1).

Tests ideas/efference_copy_cancellation.md Payoff 1: under SUMMATION the downstream
operator got a "free preview" of the answer and distributed the predictable computation
into the interaction with the SPECIFIC injected forecast (entangled); under CANCELLATION
downstream only ever processes the RESIDUAL while the forecast is owned by the FM and
re-added cleanly at read-out (modular). Prediction: swapping the co-trained FM for a
fresh, equally-good but different-weights FM degrades a summation model MORE than a
cancellation model, because only summation's downstream weights depend on the specific
forecast values.

For each frozen checkpoint (sum_update, cancel_update, ...), inject four forecasts under
the model's OWN wiring and measure val_acc:
    no_inj  : gate zeroed (plain loop)                 -- the standalone floor
    random  : an untrained random-init FM              -- garbage forecast (extreme)
    fresh   : a fresh FM (new seed) retrained on the   -- equally-good, different weights
              FROZEN model's own trajectories             (THE modularity test)
    orig    : the co-trained FM                         -- baseline
Graceful-degradation read = acc(orig) - acc(fresh). Smaller = more modular. Reported
alongside the fresh-vs-orig forecast agreement cos (so a null "fresh==orig" swap is
visible, not mistaken for modularity).

Run:
  modal run a2a_forward/mnist_looped_freshfm_probe.py::freshfm_swap \
      --dataset fashion_mnist --conditions sum_update,cancel_update
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


_SPEC = {
    "sum_update":       ("update", "add"),
    "cancel_update":    ("update", "cancel"),
    "sum_nextstate":    ("next_state", "add"),
    "cancel_nextstate": ("next_state", "cancel"),
}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def freshfm_swap(
    dataset: str = "fashion_mnist",
    conditions: str = "sum_update,cancel_update",
    predict_k: int = 3,
    seed: int = 42,
    fresh_fm_steps: int = 2000,
    fresh_fm_lr: float = 1e-3,
    batch_size: int = 128,
    n_eval_batches: int = 20,
    n_embd: int = 128,
    n_head: int = 4,
    n_loop_steps: int = 8,
    patch_size: int = 4,
    fwd_d_head: int = 1,
    fwd_n_head: int = 1,
    fwd_n_layer: int = 1,
    fwd_mlp_mult: float = 0.125,
    gate_max: float = 1.0,
):
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from a2a_forward.looped_vit import LoopedViT
    from a2a_forward.forward_model import TransformerForwardModel, BoundedScalarGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = n_loop_steps
    n_positions = (28 // patch_size) ** 2 + 1
    root = f"{DATA_DIR}/a2a_forward/mnist_looped_injection"

    def tagf(form, mode):
        tag = (f"looped_1x{T}_{n_head}H_{n_embd}D_pre0coda0_{form}_k{predict_k}"
               f"_fmd{fwd_d_head}m{fwd_mlp_mult}_scalargate")
        if mode != "add":
            tag += f"_{mode}"
        return f"{dataset}_{tag}"

    # ---- data ----
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    print(f"Loading {dataset}...")
    ds = load_dataset(ds_name)
    tr = np.stack([np.array(im) for im in ds["train"]["image"]])
    train_images = torch.from_numpy(tr).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    te = np.stack([np.array(im) for im in ds["test"]["image"]])
    test_images = torch.from_numpy(te).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    # fixed eval batches (identical across conditions and injection sources)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    eval_idx = [torch.randint(len(test_images), (batch_size,), generator=eval_gen)
                for _ in range(n_eval_batches)]

    def build_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False).to(device)

    def load(cond):
        form, mode = _SPEC[cond]
        d = os.path.join(root, tagf(form, mode), "cl_last")
        if not os.path.exists(os.path.join(d, "model.pt")):
            print(f"  MISSING checkpoint for {cond}: {d}")
            return None
        torch.manual_seed(seed)
        model = LoopedViT(img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
                          n_loop_layers=1, n_head=n_head, n_embd=n_embd, n_steps=T,
                          prelude_layers=0, coda_layers=0, inject_input_each_step=True).to(device)
        model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=device))
        model.eval()
        for pp in model.parameters():
            pp.requires_grad_(False)
        orig = build_fm()
        orig.load_state_dict(torch.load(os.path.join(d, "fwd.pt"), map_location=device))
        orig.eval()
        gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
        gate.load_state_dict(torch.load(os.path.join(d, "gate.pt"), map_location=device))
        gate.eval()
        return model, orig, gate, form, mode

    def make_hook(fm, gate, form, mode, active=True):
        def hook(t, operand, state):
            if not active:
                return None
            pred = fm(operand.detach())
            signal = (pred - state) if form == "update" else pred
            return gate(signal.detach())
        return hook

    def tgt_idx(t):
        return min(t + predict_k - 1, T - 1)

    def train_fresh(model, orig, gate, form, mode, fm_seed):
        """Train a fresh FM (new init/seed) on the FROZEN model's own trajectories
        (model run with its ORIG injection -- the real operating distribution)."""
        torch.manual_seed(fm_seed)
        fresh = build_fm()
        opt = torch.optim.AdamW(fresh.parameters(), lr=fresh_fm_lr, weight_decay=0.01)
        rng = torch.Generator().manual_seed(fm_seed + 7)
        orig_hook = make_hook(orig, gate, form, mode, active=True)
        fresh.train()
        for step in range(fresh_fm_steps):
            idx = torch.randint(len(train_images), (batch_size,), generator=rng)
            imgs = train_images[idx].to(device)
            with torch.no_grad():
                _, _, inter = model(imgs, return_intermediates=True,
                                    step_inject_fn=orig_hook, inject_mode=mode)
            p = inter["post_prelude"]
            loss = 0.0
            for t in range(T):
                s_prev = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
                a_t = (s_prev + p).detach()
                loss = loss + F.mse_loss(fresh(a_t), inter[f"post_step{tgt_idx(t)}"].detach())
            loss = loss / T
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fresh.parameters(), 1.0)
            opt.step()
        fresh.eval()
        for pp in fresh.parameters():
            pp.requires_grad_(False)
        return fresh

    @torch.no_grad()
    def eval_acc(model, fm, gate, form, mode, active):
        hook = make_hook(fm, gate, form, mode, active=active) if active else None
        a = 0.0
        for bi in range(n_eval_batches):
            vim = test_images[eval_idx[bi]].to(device)
            vlb = test_labels[eval_idx[bi]].to(device)
            vlog, _ = model(vim, vlb, step_inject_fn=hook, inject_mode=mode)
            a += (vlog.argmax(-1) == vlb).float().mean().item()
        return a / n_eval_batches

    @torch.no_grad()
    def forecast_agreement(model, orig, fresh, gate, form, mode):
        """cos between fresh and orig forecasts on the operating trajectory."""
        orig_hook = make_hook(orig, gate, form, mode, active=True)
        vim = test_images[eval_idx[0]].to(device)
        _, _, inter = model(vim, return_intermediates=True,
                            step_inject_fn=orig_hook, inject_mode=mode)
        p = inter["post_prelude"]
        cs = []
        for t in range(T):
            s_prev = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
            a_t = s_prev + p
            po, pf = orig(a_t), fresh(a_t)
            cs.append(F.cosine_similarity(po.reshape(-1, n_embd),
                                          pf.reshape(-1, n_embd), dim=-1).mean().item())
        return float(np.mean(cs))

    results = {}
    for cond in [c for c in conditions.split(",") if c]:
        if cond not in _SPEC:
            continue
        loaded = load(cond)
        if loaded is None:
            continue
        model, orig, gate, form, mode = loaded
        print(f"\n[{cond}] training fresh FM ({fresh_fm_steps} steps)...")
        fresh = train_fresh(model, orig, gate, form, mode, fm_seed=seed + 500)
        torch.manual_seed(seed + 900)
        rand_fm = build_fm()  # untrained
        for pp in rand_fm.parameters():
            pp.requires_grad_(False)
        rand_fm.eval()

        acc_orig = eval_acc(model, orig, gate, form, mode, active=True)
        acc_fresh = eval_acc(model, fresh, gate, form, mode, active=True)
        acc_rand = eval_acc(model, rand_fm, gate, form, mode, active=True)
        acc_noinj = eval_acc(model, None, gate, form, mode, active=False)
        agree = forecast_agreement(model, orig, fresh, gate, form, mode)
        results[cond] = {
            "mode": mode, "form": form,
            "acc_orig": acc_orig, "acc_fresh": acc_fresh,
            "acc_random": acc_rand, "acc_noinj": acc_noinj,
            "swap_drop_fresh": acc_orig - acc_fresh,
            "dependency": acc_orig - acc_noinj,
            "fresh_vs_orig_forecast_cos": agree,
        }
        print(f"  [{cond}] orig={acc_orig:.4f} fresh={acc_fresh:.4f} "
              f"random={acc_rand:.4f} noinj={acc_noinj:.4f} "
              f"(fresh-vs-orig forecast cos={agree:.3f})")

    # ---- print ----
    print("\n" + "=" * 92)
    print(f"  FRESH-FM SWAP ({dataset}) -- modular (cancellation) vs entangled (summation) dependency")
    print("=" * 92)
    print(f"  {'condition':16s} {'mode':7s} {'orig':>7s} {'fresh':>7s} {'random':>7s} "
          f"{'noinj':>7s} {'swapΔ':>7s} {'depend':>7s} {'fresh~orig':>10s}")
    for c, r in results.items():
        print(f"  {c:16s} {r['mode']:7s} {r['acc_orig']:7.4f} {r['acc_fresh']:7.4f} "
              f"{r['acc_random']:7.4f} {r['acc_noinj']:7.4f} {r['swap_drop_fresh']:7.4f} "
              f"{r['dependency']:7.4f} {r['fresh_vs_orig_forecast_cos']:10.3f}")
    print("\n  Payoff-1 read: smaller swapΔ (orig-fresh) => more MODULAR. Compare summation")
    print("  vs cancellation for the SAME form. (fresh~orig cos near 1 => swap is a weak")
    print("  perturbation and the test is inconclusive -- interpret swapΔ relative to it.)")

    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_freshfm_swap.json"), "w") as f:
        json.dump({"dataset": dataset, "results": results}, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {out_dir}/{dataset}_freshfm_swap.json")
    return results
