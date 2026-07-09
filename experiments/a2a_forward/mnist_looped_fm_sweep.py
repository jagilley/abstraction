"""FM-capacity sweep for the looped ViT: trace forward-model size vs predictability.

Motivated by the observation that a 1%-capacity FM still predicts the loop's
k-step trajectory at cos 0.999 -- but with only two capacity points that's not a
curve. This traces cos vs FM size properly, and crucially reports TWO cosines:

  - state_cos  = cos(FM(a_t), s_{t+k})            <- inflated by the shared input
                                                      component p in every state
  - update_cos = cos(FM(a_t) - s_t, s_{t+k} - s_t) <- what the update-form injection
                                                      actually rides on; the honest
                                                      measure of whether the forecast
                                                      is redundant (high) or noise (low)

Runs on a CONFOUND-FREE pure loop by default (prelude=coda=0, no deep supervision,
optional damped update) so the loop -- not a feedforward shortcut -- does the work.
One loop is trained and frozen; all FMs are trained on the same frozen transitions,
isolating FM capacity as the only variable (cf. feedforward scaling_sweep.py).

Run:
  modal run --detach a2a_forward/mnist_looped_fm_sweep.py::fm_capacity_sweep
  modal run --detach a2a_forward/mnist_looped_fm_sweep.py::fm_capacity_sweep --dataset fashion_mnist
"""

import json
from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

# (d_head, n_head, n_layer, mlp_mult) spanning ~tiny -> ~loop-block sized
FM_CONFIGS = [
    (1, 1, 1, 0.125),
    (2, 1, 1, 0.25),
    (4, 1, 1, 0.5),
    (8, 1, 1, 1.0),
    (16, 1, 1, 2.0),
    (32, 1, 1, 2.0),
    (64, 1, 1, 4.0),
    (64, 2, 2, 4.0),
]


def _load(dataset):
    import numpy as np
    import torch
    from datasets import load_dataset
    name = {"mnist": "ylecun/mnist",
            "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    ds = load_dataset(name)
    def stack(split):
        imgs = np.stack([np.array(im) for im in ds[split]["image"]])
        return (torch.from_numpy(imgs).float().unsqueeze(1) / 255.0,
                torch.tensor(ds[split]["label"]))
    return stack("train"), stack("test")


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=32768)
def fm_capacity_sweep(
    dataset: str = "mnist",
    n_loop_layers: int = 1,
    prelude_layers: int = 0,
    coda_layers: int = 0,
    deep_sup: bool = False,
    damping: float = 1.0,        # <1.0 = damped update s+=alpha*(G-s); 1.0 = plain
    n_head: int = 4,
    n_embd: int = 128,
    train_steps_loop: int = 8,
    predict_k: int = 3,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    loop_steps: int = 5000,
    fm_steps: int = 2500,
    fm_lr: float = 1e-3,
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.looped_vit import LoopedViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = train_steps_loop
    n_positions = (28 // patch_size) ** 2 + 1
    sup_start = int(T * 0.5)
    print(f"FM sweep on {dataset} | pure loop prelude={prelude_layers} "
          f"coda={coda_layers} deep_sup={deep_sup} damping={damping} k={predict_k}")

    (train_images, train_labels), (test_images, test_labels) = _load(dataset)

    g = torch.Generator().manual_seed(seed)
    train_idx = [torch.randint(len(train_images), (batch_size,), generator=g)
                 for _ in range(loop_steps)]
    ge = torch.Generator().manual_seed(seed + 1)
    test_idx = [torch.randint(len(test_images), (batch_size,), generator=ge)
                for _ in range(200)]

    def tgt_idx(t):
        return min(t + predict_k - 1, T - 1)

    # --- LoopedViT with optional damping (monkeypatch operator application) ---
    torch.manual_seed(seed)
    loop = LoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        n_loop_layers=n_loop_layers, n_head=n_head, n_embd=n_embd, n_steps=T,
        prelude_layers=prelude_layers, coda_layers=coda_layers,
        inject_input_each_step=True,
    ).to(device)

    # Damped update: s_{t+1} = s_t + alpha*(G(s_t + p) - s_t). Implemented by
    # wrapping _apply_operator so the loop body stays in looped_vit.
    if damping < 1.0:
        base_op = loop._apply_operator
        # We need s_t (pre-operator carry) to damp; recompute via closure on the
        # operand. Since forward passes s_in = s_t + p, and G is applied to s_in,
        # damping toward s_in approximates damping toward the carried state for
        # small input magnitude. Simpler + well-defined: damp toward the operand.
        def damped(x):
            return x + damping * (base_op(x) - x)
        loop._apply_operator = damped

    # =============================================
    # Train the pure loop
    # =============================================
    opt = torch.optim.AdamW(loop.parameters(), lr=lr, weight_decay=0.01)
    print("Training pure loop...")
    for step in range(loop_steps):
        loop.train()
        idx = train_idx[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)
        logits, _, step_logits = loop(images, readout_all_steps=True)
        if deep_sup:
            loss = torch.stack([F.cross_entropy(sl, labels)
                                for sl in step_logits[sup_start:]]).mean()
        else:
            loss = F.cross_entropy(step_logits[-1], labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(loop.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == loop_steps - 1:
            loop.eval()
            with torch.no_grad():
                vi = test_images[test_idx[0]].to(device)
                vl = test_labels[test_idx[0]].to(device)
                vlog, _ = loop(vi, vl)
                acc = (vlog.argmax(-1) == vl).float().mean().item()
            print(f"  loop step {step}: train_loss={loss.item():.4f} val_acc={acc:.4f}")

    loop.eval()
    for p_ in loop.parameters():
        p_.requires_grad_(False)

    # loop-necessity signal: accuracy vs eval-T
    accT = {}
    with torch.no_grad():
        for Te in range(1, T + 1):
            a = 0.0
            for bi in range(5):
                vi = test_images[test_idx[bi]].to(device)
                vl = test_labels[test_idx[bi]].to(device)
                vlog, _ = loop(vi, vl, n_steps=Te)
                a += (vlog.argmax(-1) == vl).float().mean().item()
            accT[str(Te)] = a / 5
    print(f"  loop acc vs T: {[round(accT[str(t)], 3) for t in range(1, T + 1)]}")

    def transitions(images):
        """Return lists over t of (a_t, s_t, target=s_{t+k})."""
        _, _, inter = loop(images, return_intermediates=True)
        p = inter["post_prelude"]
        out = []
        for t in range(T):
            s_t = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
            a_t = s_t + p
            out.append((a_t, s_t, inter[f"post_step{tgt_idx(t)}"]))
        return out

    # =============================================
    # Sweep FM capacity on the frozen loop
    # =============================================
    results = []
    for (d_head, fm_nhead, fm_nlayer, mlp_mult) in FM_CONFIGS:
        torch.manual_seed(seed + 100)
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=d_head, n_head=fm_nhead, n_layer=fm_nlayer,
            mlp_mult=mlp_mult, block_size=n_positions, causal=False,
        ).to(device)
        n_fm = sum(pp.numel() for pp in fm.parameters())
        opt_fm = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)

        for fstep in range(fm_steps):
            fm.train()
            idx = train_idx[fstep % len(train_idx)]
            images = train_images[idx].to(device)
            with torch.no_grad():
                trs = transitions(images)
            loss = 0.0
            for (a_t, s_t, tgt) in trs:
                loss = loss + F.mse_loss(fm(a_t), tgt)
            loss = loss / T
            opt_fm.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0); opt_fm.step()

        # --- eval metrics ---
        fm.eval()
        state_cos = np.zeros(T); update_cos = np.zeros(T); mse = np.zeros(T)
        resid_all = []
        with torch.no_grad():
            for bi in range(30):
                images = test_images[test_idx[bi]].to(device)
                trs = transitions(images)
                for t, (a_t, s_t, tgt) in enumerate(trs):
                    pred = fm(a_t)
                    state_cos[t] += F.cosine_similarity(pred, tgt, dim=-1).mean().item()
                    upd_pred = pred - s_t
                    upd_tgt = tgt - s_t
                    update_cos[t] += F.cosine_similarity(
                        upd_pred, upd_tgt, dim=-1).mean().item()
                    mse[t] += F.mse_loss(pred, tgt).item()
                    if t == T - 2:  # residual at a mid/late step
                        resid_all.append((tgt - pred).reshape(-1, n_embd).cpu())
        state_cos /= 30; update_cos /= 30; mse /= 30

        # residual effective rank
        R = torch.cat(resid_all, dim=0).numpy()
        cov = np.cov(R, rowvar=False)
        ev = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0)
        pv = ev[ev > 0] / ev.sum()
        eff_rank = float(np.exp(-np.sum(pv * np.log(pv)))) if len(pv) else 0.0

        row = {
            "d_head": d_head, "fm_n_head": fm_nhead, "fm_n_layer": fm_nlayer,
            "mlp_mult": mlp_mult, "fm_params": n_fm,
            "state_cos_mean": float(state_cos.mean()),
            "update_cos_mean": float(update_cos.mean()),
            "state_cos_per_step": state_cos.tolist(),
            "update_cos_per_step": update_cos.tolist(),
            "mse_mean": float(mse.mean()),
            "residual_eff_rank": eff_rank,
        }
        results.append(row)
        print(f"  FM d_head={d_head:3d} m={mlp_mult:<5} ({n_fm/1e3:6.1f}K): "
              f"state_cos={state_cos.mean():.4f} update_cos={update_cos.mean():.4f} "
              f"mse={mse.mean():.4f} resid_rank={eff_rank:.1f}")

    # =============================================
    # Save + report
    # =============================================
    tag = f"{dataset}_loop{n_loop_layers}x{T}_pre{prelude_layers}coda{coda_layers}"
    tag += f"_deepsup{int(deep_sup)}_damp{damping}_k{predict_k}"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_looped_fm_sweep/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "config": {
            "dataset": dataset, "n_loop_layers": n_loop_layers,
            "prelude_layers": prelude_layers, "coda_layers": coda_layers,
            "deep_sup": deep_sup, "damping": damping, "n_embd": n_embd,
            "train_steps_loop": T, "predict_k": predict_k, "seed": seed,
        },
        "loop_acc_vs_T": accT,
        "sweep": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\n{'=' * 70}\n  FM-CAPACITY CURVE ({tag})\n{'=' * 70}")
    print(f"  {'FM params':>10} {'state_cos':>10} {'update_cos':>11} {'resid_rank':>11}")
    for r in results:
        print(f"  {r['fm_params']/1e3:8.1f}K {r['state_cos_mean']:10.4f} "
              f"{r['update_cos_mean']:11.4f} {r['residual_eff_rank']:11.1f}")
    print(f"\n  Loop necessity (acc T=1 -> T={T}): "
          f"{accT['1']:.3f} -> {accT[str(T)]:.3f}")
    return out
