"""Payoff-4 probe: does injection benefit track forecast quality, per input?

Tests the loss-landscape asymmetry the idea doc says should make the gate self-close
under cancellation (ideas/efference_copy_cancellation.md, Payoff 4): "subtracting a
WRONG forecast actively corrupts the forward path... gate-opening is beneficial exactly
where the forecast is accurate and harmful where it is not." The fresh-FM-swap probe's
random-FM arm showed the coarse (all-bad) version (a garbage forecast HURTS cancellation
but HELPS summation). This is the fine, per-input version -- no retraining, on the
existing checkpoints.

Per test input i, measure (all with the model's OWN wiring / trained gate):
  q_i  = forecast quality = mean_t cos(FM(a_t), s_{t+k}) on the PLAIN loop (gate zeroed),
         so q is an input property (how predictable this input's dynamics are), measured
         WITHOUT the injection -> no circularity with the injection's effect.
  b_i  = injection benefit = CE_noinj(i) - CE_inj(i)   (>0 => injection lowers loss on i).
Then correlate b vs q across inputs.

Prediction:
  cancellation -> corr(b, q) strongly POSITIVE (injection helps predictable inputs, HURTS
                  unpredictable ones -- the endogenous basis for gate self-closing).
  summation    -> corr(b, q) ~ FLAT (the injection is a benign extra input regardless of
                  forecast quality; cf. LOOPED "random_proj was the best condition").

Run:
  modal run a2a_forward/mnist_looped_p4_probe.py::p4_benefit_vs_quality \
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


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=1800, memory=32768)
def p4_benefit_vs_quality(
    dataset: str = "fashion_mnist",
    conditions: str = "sum_update,cancel_update",
    n_examples: int = 5000,
    n_bins: int = 5,
    predict_k: int = 3,
    seed: int = 42,
    batch_size: int = 250,
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

    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    print(f"Loading {dataset}...")
    ds = load_dataset(ds_name)
    te = np.stack([np.array(im) for im in ds["test"]["image"]])
    test_images = (torch.from_numpy(te).float().unsqueeze(1) / 255.0)[:n_examples]
    test_labels = torch.tensor(ds["test"]["label"])[:n_examples]
    N = test_images.shape[0]
    print(f"  eval on {N} examples")

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
        fm = build_fm()
        fm.load_state_dict(torch.load(os.path.join(d, "fwd.pt"), map_location=device))
        fm.eval()
        gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
        gate.load_state_dict(torch.load(os.path.join(d, "gate.pt"), map_location=device))
        gate.eval()
        return model, fm, gate, form, mode

    def tgt_idx(t):
        return min(t + predict_k - 1, T - 1)

    def pearson(x, y):
        x, y = np.asarray(x), np.asarray(y)
        xc, yc = x - x.mean(), y - y.mean()
        d = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
        return float((xc * yc).sum() / d) if d > 0 else float("nan")

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return pearson(rx, ry)

    results = {}
    for cond in [c for c in conditions.split(",") if c]:
        if cond not in _SPEC:
            continue
        loaded = load(cond)
        if loaded is None:
            continue
        model, fm, gate, form, mode = loaded

        def hook(t, operand, state):
            pred = fm(operand.detach())
            signal = (pred - state) if form == "update" else pred
            return gate(signal.detach())

        q_all, b_all = [], []
        with torch.no_grad():
            for start in range(0, N, batch_size):
                vim = test_images[start:start + batch_size].to(device)
                vlb = test_labels[start:start + batch_size].to(device)
                # plain loop (no injection): forecast quality q + CE_noinj
                log_ni, _, inter = model(vim, return_intermediates=True)
                ce_ni = F.cross_entropy(log_ni, vlb, reduction="none")
                p = inter["post_prelude"]
                q = torch.zeros(vim.shape[0], device=device)
                for t in range(T):
                    s_prev = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
                    a_t = s_prev + p
                    cos_tp = F.cosine_similarity(fm(a_t), inter[f"post_step{tgt_idx(t)}"], dim=-1)
                    q = q + cos_tp.mean(dim=1)  # mean over positions -> per example
                q = q / T
                # injection run (model's own wiring): CE_inj
                log_inj, _ = model(vim, step_inject_fn=hook, inject_mode=mode)
                ce_inj = F.cross_entropy(log_inj, vlb, reduction="none")
                q_all.append(q.cpu().numpy())
                b_all.append((ce_ni - ce_inj).cpu().numpy())
        q_all = np.concatenate(q_all)
        b_all = np.concatenate(b_all)

        # binned mean benefit by forecast-quality quantile
        order = np.argsort(q_all)
        bins = np.array_split(order, n_bins)
        bin_q = [float(q_all[b].mean()) for b in bins]
        bin_b = [float(b_all[b].mean()) for b in bins]
        results[cond] = {
            "mode": mode, "form": form,
            "mean_benefit": float(b_all.mean()),
            "corr_pearson": pearson(b_all, q_all),
            "corr_spearman": spearman(b_all, q_all),
            "frac_inputs_hurt": float((b_all < 0).mean()),
            "bin_q": bin_q, "bin_benefit": bin_b,
            "q_mean": float(q_all.mean()), "q_std": float(q_all.std()),
        }
        print(f"  [{cond:16s}] mean_b={b_all.mean():+.4f} "
              f"corr(b,q) pearson={results[cond]['corr_pearson']:+.3f} "
              f"spearman={results[cond]['corr_spearman']:+.3f} "
              f"frac_hurt={results[cond]['frac_inputs_hurt']:.3f}")

    # ---- print ----
    print("\n" + "=" * 90)
    print(f"  PAYOFF-4 PROBE ({dataset}) -- does injection benefit track forecast quality per input?")
    print("=" * 90)
    print(f"  {'condition':16s} {'mode':7s} {'mean_b':>8s} {'corr_P':>8s} {'corr_S':>8s} {'frac_hurt':>10s}")
    for c, r in results.items():
        print(f"  {c:16s} {r['mode']:7s} {r['mean_benefit']:8.4f} {r['corr_pearson']:8.3f} "
              f"{r['corr_spearman']:8.3f} {r['frac_inputs_hurt']:10.3f}")
    print("\n  Mean injection benefit by forecast-quality quintile (low q -> high q):")
    print(f"  {'condition':16s}  " + "  ".join(f"Q{i+1}" for i in range(n_bins)))
    for c, r in results.items():
        row = "  ".join(f"{v:+.3f}" for v in r["bin_benefit"])
        print(f"  {c:16s}  {row}")
    print("\n  q per quintile (context):")
    for c, r in results.items():
        row = "  ".join(f"{v:.3f}" for v in r["bin_q"])
        print(f"  {c:16s}  {row}")
    print("\n  Prediction: cancellation corr(b,q) POSITIVE & benefit rises across quintiles")
    print("  (hurts low-q inputs); summation ~flat (benign extra input).")

    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_p4_benefit_vs_quality.json"), "w") as f:
        json.dump({"dataset": dataset, "results": results}, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {out_dir}/{dataset}_p4_benefit_vs_quality.json")
    return results
