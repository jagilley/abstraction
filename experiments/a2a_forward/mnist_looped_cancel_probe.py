"""Mechanism probe for the efference-copy cancellation A/B (Control 3, mode-symmetric).

Directly tests whether a cancellation-trained loop LITERALLY cancels the forecast from
the forward path -- the mechanism behind the robustness-inversion FLIP found in the A/B
(mnist_looped_injection cancellation run) -- vs the summation loop which amplifies it
(documented cos(eff,inj) ~ +0.8, gain 4-5x; ideas/efference_copy_cancellation.md,
LOOPED_README Control 3).

For each trained checkpoint, at every loop step compute the NET marginal effect of the
injection on the next state under the model's OWN wiring:

    base       = G(s_in)                     (gate zeroed / plain loop; s_in = s_t + p)
    actual_sum = G(s_in + inj)               (summation)
    actual_can = G(s_in - inj) + inj         (cancellation: subtract, propagate, re-add)
    Delta      = actual - base
    gain_actual = ||Delta|| / ||inj||    cos_actual = cos(Delta, inj)

Prediction:
  summation    -> gain_actual >> 1, cos_actual ~ +0.8   (the forecast AMPLIFIES into s_{t+1})
  cancellation -> gain_actual <<  1, cos_actual ~ 0/neg (the forecast is CANCELLED from s_{t+1})

Also reports the summation-style eff_add = G(s_in + inj) - G(s_in) for BOTH models (a
counterfactual that adds the injection regardless of the model's real topology). If
gain_add / cos_add are SIMILAR across the two while gain_actual / cos_actual DIFFER, the
cancellation comes from the TOPOLOGY (subtract-then-re-add), not from the operator's
learned local response -- exactly the "restoring force by construction" claim.

Eval-only. Loads the confound-free bounded-gate cl_last checkpoints written by
mnist_looped_injection.py::train_condition for {sum,cancel}x{update,next_state}.

Run:
  modal run a2a_forward/mnist_looped_cancel_probe.py::cancel_mechanism \
      --dataset fashion_mnist --conditions sum_update,cancel_update
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


# name -> (injection_form, inject_mode)
_SPEC = {
    "sum_update":       ("update", "add"),
    "cancel_update":    ("update", "cancel"),
    "sum_nextstate":    ("next_state", "add"),
    "cancel_nextstate": ("next_state", "cancel"),
}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=1800, memory=32768)
def cancel_mechanism(
    dataset: str = "fashion_mnist",
    conditions: str = "sum_update,cancel_update",
    n_examples: int = 2000,
    predict_k: int = 3,
    seed: int = 42,
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
        # mirrors mnist_looped_injection.py::train_condition tag logic for this config
        tag = (f"looped_1x{T}_{n_head}H_{n_embd}D_pre0coda0_{form}_k{predict_k}"
               f"_fmd{fwd_d_head}m{fwd_mlp_mult}_scalargate")
        if mode != "add":
            tag += f"_{mode}"
        return f"{dataset}_{tag}"

    # ---- data (fixed test subset) ----
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    print(f"Loading {dataset}...")
    ds = load_dataset(ds_name)
    te = np.stack([np.array(im) for im in ds["test"]["image"]])
    images = (torch.from_numpy(te).float().unsqueeze(1) / 255.0)[:n_examples].to(device)
    labels = torch.tensor(ds["test"]["label"])[:n_examples].to(device)
    print(f"  eval on {images.shape[0]} examples")

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
        fwd = build_fm()
        fwd.load_state_dict(torch.load(os.path.join(d, "fwd.pt"), map_location=device))
        fwd.eval()
        gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
        gate.load_state_dict(torch.load(os.path.join(d, "gate.pt"), map_location=device))
        gate.eval()
        return model, fwd, gate, form, mode

    @torch.no_grad()
    def capture(cond):
        loaded = load(cond)
        if loaded is None:
            return None
        model, fwd, gate, form, mode = loaded
        injs = []

        def hook(t, operand, state):
            pred = fwd(operand.detach())
            signal = (pred - state) if form == "update" else pred
            inj = gate(signal.detach())
            injs.append(inj.detach())
            return inj

        logits, _, inter = model(images, return_intermediates=True,
                                 step_inject_fn=hook, inject_mode=mode)
        acc = (logits.argmax(-1) == labels).float().mean().item()
        return dict(model=model, p=inter["post_prelude"],
                    states=[inter[f"post_step{t}"] for t in range(T)],
                    inj=torch.stack(injs, 0), mode=mode, form=form, acc=acc,
                    gate=gate.injection_norm())

    results = {}
    for cond in [c for c in conditions.split(",") if c]:
        if cond not in _SPEC:
            print(f"  unknown condition {cond}; skip")
            continue
        cp = capture(cond)
        if cp is None:
            continue
        G = cp["model"]._apply_operator
        p = cp["p"]
        gain_add, cos_add, gain_act, cos_act = [], [], [], []
        with torch.no_grad():
            for t in range(T):
                s_prev = (cp["states"][t - 1] if t >= 1
                          else torch.zeros_like(cp["states"][0]))
                s_in = s_prev + p
                inj = cp["inj"][t]
                base = G(s_in)
                eff_add = G(s_in + inj) - base            # counterfactual: always +inj
                if cp["mode"] == "cancel":
                    actual = G(s_in - inj) + inj           # real cancellation transition
                else:
                    actual = G(s_in + inj)                 # real summation transition
                delta = actual - base
                innorm = inj.norm(dim=-1) + 1e-8
                gain_add.append((eff_add.norm(dim=-1) / innorm).mean().item())
                cos_add.append(F.cosine_similarity(
                    eff_add.reshape(-1, n_embd), inj.reshape(-1, n_embd), dim=-1).mean().item())
                gain_act.append((delta.norm(dim=-1) / innorm).mean().item())
                cos_act.append(F.cosine_similarity(
                    delta.reshape(-1, n_embd), inj.reshape(-1, n_embd), dim=-1).mean().item())
        results[cond] = {
            "mode": cp["mode"], "form": cp["form"], "acc": cp["acc"], "gate": cp["gate"],
            "gain_add_mean": float(np.mean(gain_add)), "cos_add_mean": float(np.mean(cos_add)),
            "gain_actual_mean": float(np.mean(gain_act)), "cos_actual_mean": float(np.mean(cos_act)),
            "gain_actual_per_step": gain_act, "cos_actual_per_step": cos_act,
            "gain_add_per_step": gain_add, "cos_add_per_step": cos_add,
        }
        print(f"  [{cond:16s}] mode={cp['mode']:6s} form={cp['form']:10s} acc={cp['acc']:.3f} "
              f"gate={cp['gate']:.3f}")

    # ---- print ----
    print("\n" + "=" * 84)
    print(f"  CANCELLATION MECHANISM PROBE ({dataset}) -- does the loop amplify or cancel?")
    print("=" * 84)
    print("\n  ACTUAL wiring (the model's real transition): net effect of injection on s_{t+1}")
    print(f"  {'condition':16s} {'mode':7s} {'gain_actual':>12s} {'cos_actual':>11s}   interpretation")
    for c, r in results.items():
        interp = ("AMPLIFY (no cancel)" if r["cos_actual_mean"] > 0.4 and r["gain_actual_mean"] > 1.5
                  else "CANCELLED" if r["gain_actual_mean"] < 0.6
                  else "partial")
        print(f"  {c:16s} {r['mode']:7s} {r['gain_actual_mean']:12.3f} {r['cos_actual_mean']:11.3f}   {interp}")
    print("\n  COUNTERFACTUAL +inj (operator's local response, topology-independent):")
    print(f"  {'condition':16s} {'gain_add':>10s} {'cos_add':>9s}   (similar across modes => cancellation is TOPOLOGICAL)")
    for c, r in results.items():
        print(f"  {c:16s} {r['gain_add_mean']:10.3f} {r['cos_add_mean']:9.3f}")

    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_cancel_mechanism.json"), "w") as f:
        json.dump({"dataset": dataset, "results": results}, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {out_dir}/{dataset}_cancel_mechanism.json")
    return results
