"""Payoff-4, definitive: does injection benefit fall as the FORECAST is corrupted?

The input-corruption sweep (mnist_looped_p4_ood_probe) failed its premise -- in the
low-rank loop, corrupting the INPUT made the forecast MORE self-consistent (q rose), so it
never created a bad-forecast regime. This probe corrupts the FORECAST directly, decoupled
from the input, by interpolating the injected prediction toward a garbage (random-init FM)
prediction:

    pred_mix = (1-alpha) * FM_pred + alpha * rand_pred      (rand norm-matched to FM_pred;
                                                              pred_mix renormed to FM_pred norm)

so ONLY the direction/quality of the forecast degrades with alpha (magnitude held ~constant,
isolating "wrong forecast" from "bigger injection"). Clean inputs (no image corruption).

At each alpha, per condition (model's OWN wiring / trained gate):
  q_mix  = mean_t cos(pred_mix, s_{t+k}) on the plain loop  -- forecast quality, DROPS with alpha
  b_acc  = acc_inj - acc_noinj      b_loss = CE_noinj - CE_inj

Prediction (ideas/efference_copy_cancellation.md, Payoff 4):
  cancellation -> b FALLS and flips NEGATIVE as alpha rises (subtracting an increasingly
                  WRONG forecast corrupts the forward path).
  summation    -> b stays flat / >= 0 (a wrong-direction forecast is a benign extra input;
                  cf. the fresh-FM-swap random arm where garbage HELPED summation).

This is the graded version of that random-FM arm and the clean test of Payoff 4's mechanism.

Run:
  modal run a2a_forward/mnist_looped_p4_fc_probe.py::p4_forecast_corruption \
      --dataset fashion_mnist --conditions sum_update,cancel_update \
      --alphas "0.0,0.25,0.5,0.75,1.0"
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
def p4_forecast_corruption(
    dataset: str = "fashion_mnist",
    conditions: str = "sum_update,cancel_update",
    alphas: str = "0.0,0.25,0.5,0.75,1.0",
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
    alpha_list = [float(x) for x in alphas.split(",")]

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
    print(f"  eval on {N} examples; alphas={alpha_list}")

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
        torch.manual_seed(seed + 900)           # same garbage FM as the fresh-FM random arm
        rand_fm = build_fm()
        for pp in rand_fm.parameters():
            pp.requires_grad_(False)
        rand_fm.eval()
        gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
        gate.load_state_dict(torch.load(os.path.join(d, "gate.pt"), map_location=device))
        gate.eval()
        return model, fm, rand_fm, gate, form, mode

    def tgt_idx(t):
        return min(t + predict_k - 1, T - 1)

    def mix_pred(pt, pr, alpha):
        """(1-a)*trained + a*(garbage norm-matched to trained), renormed to trained norm,
        so only direction/quality degrades with alpha (magnitude held ~constant)."""
        pr = pr * (pt.norm(dim=-1, keepdim=True) / (pr.norm(dim=-1, keepdim=True) + 1e-8))
        pm = (1.0 - alpha) * pt + alpha * pr
        pm = pm * (pt.norm(dim=-1, keepdim=True) / (pm.norm(dim=-1, keepdim=True) + 1e-8))
        return pm

    results = {}
    for cond in [c for c in conditions.split(",") if c]:
        if cond not in _SPEC:
            continue
        loaded = load(cond)
        if loaded is None:
            continue
        model, fm, rand_fm, gate, form, mode = loaded

        def make_hook(alpha):
            def hook(t, operand, state):
                pt = fm(operand.detach())
                pr = rand_fm(operand.detach())
                pm = mix_pred(pt, pr, alpha)
                signal = (pm - state) if form == "update" else pm
                return gate(signal.detach())
            return hook

        acc_ni_sum = ce_ni_sum = 0.0
        q_sum = {a: 0.0 for a in alpha_list}
        acc_inj_sum = {a: 0.0 for a in alpha_list}
        ce_inj_sum = {a: 0.0 for a in alpha_list}
        nb = 0
        with torch.no_grad():
            for start in range(0, N, batch_size):
                vim = test_images[start:start + batch_size].to(device)
                vlb = test_labels[start:start + batch_size].to(device)
                B = vim.shape[0]
                nb += B
                # plain loop (alpha-independent): acc/loss + operands/targets for q_mix
                log_ni, _, inter = model(vim, return_intermediates=True)
                acc_ni_sum += (log_ni.argmax(-1) == vlb).float().mean().item() * B
                ce_ni_sum += F.cross_entropy(log_ni, vlb).item() * B
                p = inter["post_prelude"]
                a_ts = [(inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)) + p
                        for t in range(T)]
                tgts = [inter[f"post_step{tgt_idx(t)}"] for t in range(T)]
                pts = [fm(a) for a in a_ts]
                prs = [rand_fm(a) for a in a_ts]
                for alpha in alpha_list:
                    # forecast quality of the mixed prediction on the plain trajectory
                    qb = 0.0
                    for t in range(T):
                        pm = mix_pred(pts[t], prs[t], alpha)
                        qb += F.cosine_similarity(pm, tgts[t], dim=-1).mean().item()
                    q_sum[alpha] += (qb / T) * B
                    # injection run with the corrupted forecast
                    log_inj, _ = model(vim, step_inject_fn=make_hook(alpha), inject_mode=mode)
                    acc_inj_sum[alpha] += (log_inj.argmax(-1) == vlb).float().mean().item() * B
                    ce_inj_sum[alpha] += F.cross_entropy(log_inj, vlb).item() * B
        acc_ni = acc_ni_sum / nb
        ce_ni = ce_ni_sum / nb
        sweep = []
        for alpha in alpha_list:
            acc_inj = acc_inj_sum[alpha] / nb
            ce_inj = ce_inj_sum[alpha] / nb
            sweep.append({
                "alpha": alpha, "q_mix": q_sum[alpha] / nb,
                "acc_noinj": acc_ni, "acc_inj": acc_inj, "b_acc": acc_inj - acc_ni,
                "loss_noinj": ce_ni, "loss_inj": ce_inj, "b_loss": ce_ni - ce_inj,
            })
        results[cond] = {"mode": mode, "form": form, "sweep": sweep}
        print(f"  [{cond}] done (acc_noinj={acc_ni:.3f})")

    # ---- print ----
    print("\n" + "=" * 96)
    print(f"  PAYOFF-4 FORECAST-CORRUPTION ({dataset}) -- injection benefit vs forecast quality (alpha)")
    print("=" * 96)
    for c, r in results.items():
        print(f"\n  [{c}]  mode={r['mode']} form={r['form']}")
        print(f"    {'alpha':>6s} {'q_mix':>7s} {'acc_ni':>7s} {'acc_inj':>8s} {'b_acc':>7s} {'b_loss':>7s}")
        for s in r["sweep"]:
            print(f"    {s['alpha']:6.2f} {s['q_mix']:7.3f} {s['acc_noinj']:7.3f} {s['acc_inj']:8.3f} "
                  f"{s['b_acc']:+7.3f} {s['b_loss']:+7.3f}")
    print("\n  b_acc side-by-side (prediction: cancellation FALLS/flips NEGATIVE; summation flat/+):")
    print("    " + f"{'alpha':>6s} " + " ".join(f"{c[:12]:>13s}" for c in results))
    for i, alpha in enumerate(alpha_list):
        row = f"    {alpha:6.2f} "
        for c in results:
            row += f"{results[c]['sweep'][i]['b_acc']:+13.3f} "
        print(row)

    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_p4_forecast_corruption.json"), "w") as f:
        json.dump({"dataset": dataset, "alphas": alpha_list, "results": results},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {out_dir}/{dataset}_p4_forecast_corruption.json")
    return results
