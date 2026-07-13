"""Payoff-4 OOD test: does injection benefit flip negative as the forecast goes bad?

The clean per-input probe (mnist_looped_p4_probe) was confounded: on in-distribution data
the trained FM is uniformly excellent (q>0.94), so there is no poorly-forecast regime to
trigger the Payoff-4 asymmetry, and benefit anti-correlates with q only via a classification
headroom confound. This probe FORCES the varying-quality regime by corrupting the INPUT
(gaussian pixel noise sweep), pushing inputs OOD so the FM's forecast degrades smoothly.

At each corruption level sigma, per condition (model's OWN wiring / trained gate):
  q          = forecast quality (mean_t cos(FM(a_t), s_{t+k}) on the plain loop)  -- should DROP
  acc_noinj  = plain-loop accuracy         acc_inj = injection accuracy
  b_acc      = acc_inj - acc_noinj         (>0 => injection helps classification)
  b_loss     = CE_noinj - CE_inj           (>0 => injection lowers loss)

Prediction (ideas/efference_copy_cancellation.md, Payoff 4):
  cancellation -> b flips POSITIVE -> NEGATIVE as sigma rises (subtracting an increasingly
                  WRONG forecast corrupts the forward path -- the loss-landscape basis for
                  gate self-closing on novel/OOD inputs).
  summation    -> b stays >= 0 (a wrong forecast is a benign extra input; cf. LOOPED
                  random_proj, and the fresh-FM-swap random arm: garbage HELPED summation).

Run:
  modal run a2a_forward/mnist_looped_p4_ood_probe.py::p4_ood \
      --dataset fashion_mnist --conditions sum_update,cancel_update \
      --sigmas "0.0,0.25,0.5,0.75,1.0,1.5"
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
def p4_ood(
    dataset: str = "fashion_mnist",
    conditions: str = "sum_update,cancel_update",
    sigmas: str = "0.0,0.25,0.5,0.75,1.0,1.5",
    n_examples: int = 2500,
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
    sigma_list = [float(x) for x in sigmas.split(",")]

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
    print(f"  eval on {N} examples; sigmas={sigma_list}")

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

    def corrupt(imgs, sigma, gidx):
        if sigma <= 0:
            return imgs
        g = torch.Generator().manual_seed(seed + 5000 + gidx)
        noise = torch.randn(imgs.shape, generator=g) * sigma
        return (imgs + noise).clamp(0.0, 1.0)

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

        per_sigma = []
        for gi, sigma in enumerate(sigma_list):
            q_sum = ce_ni_sum = ce_inj_sum = 0.0
            acc_ni_sum = acc_inj_sum = 0.0
            nb = 0
            with torch.no_grad():
                for start in range(0, N, batch_size):
                    raw = test_images[start:start + batch_size]
                    vim = corrupt(raw, sigma, gi).to(device)
                    vlb = test_labels[start:start + batch_size].to(device)
                    B = vim.shape[0]
                    # plain loop: forecast quality + no-inj loss/acc
                    log_ni, _, inter = model(vim, return_intermediates=True)
                    ce_ni = F.cross_entropy(log_ni, vlb)
                    acc_ni = (log_ni.argmax(-1) == vlb).float().mean()
                    p = inter["post_prelude"]
                    qb = 0.0
                    for t in range(T):
                        s_prev = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
                        a_t = s_prev + p
                        qb = qb + F.cosine_similarity(
                            fm(a_t), inter[f"post_step{tgt_idx(t)}"], dim=-1).mean().item()
                    qb /= T
                    # injection loop
                    log_inj, _ = model(vim, step_inject_fn=hook, inject_mode=mode)
                    ce_inj = F.cross_entropy(log_inj, vlb)
                    acc_inj = (log_inj.argmax(-1) == vlb).float().mean()
                    w = B
                    q_sum += qb * w; ce_ni_sum += ce_ni.item() * w; ce_inj_sum += ce_inj.item() * w
                    acc_ni_sum += acc_ni.item() * w; acc_inj_sum += acc_inj.item() * w
                    nb += w
            q = q_sum / nb
            ce_ni_m = ce_ni_sum / nb; ce_inj_m = ce_inj_sum / nb
            acc_ni_m = acc_ni_sum / nb; acc_inj_m = acc_inj_sum / nb
            per_sigma.append({
                "sigma": sigma, "q": q,
                "acc_noinj": acc_ni_m, "acc_inj": acc_inj_m,
                "b_acc": acc_inj_m - acc_ni_m,
                "loss_noinj": ce_ni_m, "loss_inj": ce_inj_m,
                "b_loss": ce_ni_m - ce_inj_m,
            })
        results[cond] = {"mode": mode, "form": form, "sweep": per_sigma}
        print(f"  [{cond}] done")

    # ---- print ----
    print("\n" + "=" * 96)
    print(f"  PAYOFF-4 OOD TEST ({dataset}) -- injection benefit vs input corruption (forecast quality)")
    print("=" * 96)
    for c, r in results.items():
        print(f"\n  [{c}]  mode={r['mode']} form={r['form']}")
        print(f"    {'sigma':>6s} {'q(fcst)':>8s} {'acc_ni':>7s} {'acc_inj':>8s} "
              f"{'b_acc':>7s} {'b_loss':>7s}")
        for s in r["sweep"]:
            print(f"    {s['sigma']:6.2f} {s['q']:8.3f} {s['acc_noinj']:7.3f} {s['acc_inj']:8.3f} "
                  f"{s['b_acc']:+7.3f} {s['b_loss']:+7.3f}")
    print("\n  b_acc side-by-side (injection help to accuracy; prediction: cancel goes NEGATIVE):")
    hdr = "    " + f"{'sigma':>6s} " + " ".join(f"{c[:12]:>13s}" for c in results)
    print(hdr)
    for i, s in enumerate(sigma_list):
        row = f"    {s:6.2f} "
        for c in results:
            row += f"{results[c]['sweep'][i]['b_acc']:+13.3f} "
        print(row)

    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_p4_ood.json"), "w") as f:
        json.dump({"dataset": dataset, "sigmas": sigma_list, "results": results},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {out_dir}/{dataset}_p4_ood.json")
    return results
