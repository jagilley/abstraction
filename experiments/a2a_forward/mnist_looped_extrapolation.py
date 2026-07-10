"""Near-manifold self-map extrapolation test for the looped baseline battery.

The question: is the looped model's self-forecast a *runnable simulator* (its own
dynamics stay self-forecastable on near-manifold inputs it never trained on) or a
*static lookup* (self-consistency collapses off the training manifold)? This is the
cheapest MNIST-runnable proxy for the strong "execute a rough forward pass on
yourself" sense of self-knowledge (ideas/self_model_needs_a_loop.md).

Conditions (looped family, all share init/seed/data; FM co-trained in every one so
the FM-as-approximator is held constant):
  forward     = real self-forecast   (hypothesis: self-map extrapolates)
  OL          = no injection channel  (floor: FM exists but model never organized
                around it -> self-consistency should be a train-manifold artifact)
  random_proj = frozen random channel (controls for "any injection")
  shifted     = position-misaligned forecast (forecast-shaped but wrong)

Perturbations (never seen in training -> genuine extrapolation):
  rotation (near-manifold), gaussian pixel noise (graded off-manifold stress).

Metrics, per condition x perturbation-level:
  A (representational): off-manifold self-consistency  cos(FM(a'_t), s'_{t+k})  on
     the intrinsic (channel-off) perturbed run. retention = cos(eps)/cos(0).
  B (behavioral): from just the embedded input a'_0 = prelude(embed(x')), FM ->
     readout -> label; agreement with the model's ACTUAL final label on x' (live).
     "Can I predict my own output on a perturbed input from a rough forward pass?"

Run:
  modal run --detach a2a_forward/mnist_looped_extrapolation.py::extrapolation --dataset mnist
  modal run --detach a2a_forward/mnist_looped_extrapolation.py::extrapolation --dataset fashion_mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

_BASE = "looped_1x8_4H_128D_pre0coda0_update_k3_fmd1m0.125_scalargate"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def extrapolation(
    dataset: str = "mnist",
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
    import math
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

    CONDS = {
        "OL":          (f"{dataset}_{_BASE}",             "forward",     False),
        "forward":     (f"{dataset}_{_BASE}",             "forward",     True),
        "random_proj": (f"{dataset}_{_BASE}_random_proj", "random_proj", True),
        "shifted":     (f"{dataset}_{_BASE}_shifted",     "shifted",     True),
    }
    SUBDIR = {"OL": "ol_last", "forward": "cl_last",
              "random_proj": "cl_last", "shifted": "cl_last"}

    # ---- data ----
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    print(f"Loading {dataset}...")
    ds = load_dataset(ds_name)
    te = np.stack([np.array(im) for im in ds["test"]["image"]])
    images = (torch.from_numpy(te).float().unsqueeze(1) / 255.0)[:n_examples].to(device)
    print(f"  eval on {images.shape[0]} examples")

    def build_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False).to(device)

    def load_condition(cond):
        tag, baseline_type, closed = CONDS[cond]
        d = os.path.join(root, tag, SUBDIR[cond])
        torch.manual_seed(seed)
        model = LoopedViT(img_size=28, patch_size=patch_size, in_channels=1,
                          n_classes=10, n_loop_layers=1, n_head=n_head,
                          n_embd=n_embd, n_steps=T, prelude_layers=0, coda_layers=0,
                          inject_input_each_step=True).to(device)
        model.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                         map_location=device))
        model.eval()
        fwd = build_fm()
        fwd.load_state_dict(torch.load(os.path.join(d, "fwd.pt"), map_location=device))
        fwd.eval()
        gate, rand_pred = None, None
        if closed:
            gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
            gate.load_state_dict(torch.load(os.path.join(d, "gate.pt"),
                                            map_location=device))
            gate.eval()
        if baseline_type == "random_proj":
            torch.manual_seed(seed + 200)
            rand_pred = build_fm()
            for p in rand_pred.parameters():
                p.requires_grad_(False)
            rand_pred.eval()
        return model, fwd, gate, rand_pred, baseline_type, closed

    # ---- perturbations (unseen in training) ----
    def rotate(imgs, deg):
        if deg == 0:
            return imgs
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=imgs.device)
        rot = rot.unsqueeze(0).expand(imgs.shape[0], -1, -1)
        grid = F.affine_grid(rot, list(imgs.shape), align_corners=False)
        return F.grid_sample(imgs, grid, align_corners=False, padding_mode="zeros")

    def add_noise(imgs, sigma, gen):
        if sigma == 0:
            return imgs
        return (imgs + sigma * torch.randn(imgs.shape, generator=gen,
                                           device=imgs.device)).clamp(0, 1)

    PERTURBS = [("clean", None, 0.0)]
    PERTURBS += [("rot", "rot", d) for d in (15.0, 30.0, 45.0)]
    PERTURBS += [("noise", "noise", s) for s in (0.25, 0.5, 0.75)]

    noise_gen = torch.Generator(device=device).manual_seed(seed + 7)

    def perturb(kind, level):
        if kind == "rot":
            return rotate(images, level)
        if kind == "noise":
            return add_noise(images, level, noise_gen)
        return images

    def tgt_idx(t):
        return min(t + predict_k - 1, T - 1)

    def make_hook(fwd, gate, rand_pred, baseline_type):
        def hook(t, operand, state):
            if baseline_type == "random_proj":
                pred = rand_pred(operand.detach())
            else:
                pred = fwd(operand.detach())
                if baseline_type == "shifted":
                    pred = torch.roll(pred, shifts=n_positions // 2, dims=1)
            return gate((pred - state).detach())
        return hook

    @torch.no_grad()
    def run(model, x, hook=None):
        logits, _, inter = model(x, return_intermediates=True, step_inject_fn=hook)
        return logits, inter

    def fm_selfcons(fwd, inter):
        """mean cos(FM(a_t), s_{t+k}) over t and tokens on this run."""
        p = inter["post_prelude"]
        cos = []
        for t in range(T):
            s_prev = inter[f"post_step{t-1}"] if t >= 1 else torch.zeros_like(p)
            a_t = s_prev + p
            pred = fwd(a_t)
            tgt = inter[f"post_step{tgt_idx(t)}"]
            cos.append(F.cosine_similarity(pred, tgt, dim=-1).mean().item())
        return float(np.mean(cos))

    # ---- run all conditions across all perturbations ----
    results = {"dataset": dataset, "predict_k": predict_k, "conditions": {}}
    for cond in ["OL", "forward", "random_proj", "shifted"]:
        model, fwd, gate, rand_pred, baseline_type, closed = load_condition(cond)
        hook = (make_hook(fwd, gate, rand_pred, baseline_type) if closed else None)
        per = {}
        for name, kind, level in PERTURBS:
            x = perturb(kind, level)
            # intrinsic (channel-off) run -> Metric A self-consistency
            _, inter_ab = run(model, x, hook=None)
            selfcons = fm_selfcons(fwd, inter_ab)
            # live run -> the model's ACTUAL output on x'
            logits_live, _ = run(model, x, hook=hook)
            actual = logits_live.argmax(-1)
            # Metric B: predict own output from a rough forward pass off the input.
            # a'_0 = prelude(embed(x')) = post_prelude; FM -> readout -> label.
            p = inter_ab["post_prelude"]
            fm_state = fwd(p)                       # forecast of s_{k} from input alone
            label_hat = model._readout(fm_state).argmax(-1)
            self_agree = (label_hat == actual).float().mean().item()
            per[f"{name}_{level}"] = {
                "kind": kind, "level": level,
                "self_consistency": selfcons,
                "self_output_agreement": self_agree,
                "live_confidence": logits_live.softmax(-1).max(-1).values.mean().item(),
            }
        results["conditions"][cond] = per

    # ---- retention (normalize each metric by its clean value) ----
    def clean_key():
        return "clean_0.0"
    retention = {}
    for cond, per in results["conditions"].items():
        c = per[clean_key()]
        retention[cond] = {}
        for k, v in per.items():
            if k == clean_key():
                continue
            retention[cond][k] = {
                "selfcons_ret": v["self_consistency"] / (c["self_consistency"] + 1e-9),
                "agree_ret": v["self_output_agreement"] / (c["self_output_agreement"]
                                                           + 1e-9),
            }
    results["retention"] = retention

    # ---- save ----
    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_extrapolation.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---- print ----
    conds = ["OL", "forward", "random_proj", "shifted"]
    print("\n" + "=" * 78)
    print(f"  NEAR-MANIFOLD SELF-MAP EXTRAPOLATION: {dataset}")
    print("=" * 78)

    print("\n[Metric A] Self-consistency cos(FM(a'_t), s'_{t+k}), intrinsic run:")
    hdr = f"  {'perturb':12s}" + "".join(f"{c:>13s}" for c in conds)
    print(hdr)
    for name, kind, level in PERTURBS:
        k = f"{name}_{level}"
        row = f"  {k:12s}"
        for c in conds:
            row += f"{results['conditions'][c][k]['self_consistency']:13.3f}"
        print(row)

    print("\n[Metric A] Retention = cos(eps)/cos(clean)  (higher = more simulator):")
    print(hdr)
    for name, kind, level in PERTURBS:
        if name == "clean":
            continue
        k = f"{name}_{level}"
        row = f"  {k:12s}"
        for c in conds:
            row += f"{retention[c][k]['selfcons_ret']:13.3f}"
        print(row)

    print("\n[Metric B] Self-output agreement  P(readout(FM(input)) == actual output):")
    print(hdr)
    for name, kind, level in PERTURBS:
        k = f"{name}_{level}"
        row = f"  {k:12s}"
        for c in conds:
            row += f"{results['conditions'][c][k]['self_output_agreement']:13.3f}"
        print(row)

    print("\n[Metric B] Agreement retention = agree(eps)/agree(clean):")
    print(hdr)
    for name, kind, level in PERTURBS:
        if name == "clean":
            continue
        k = f"{name}_{level}"
        row = f"  {k:12s}"
        for c in conds:
            row += f"{retention[c][k]['agree_ret']:13.3f}"
        print(row)

    print(f"\nSaved to {out_dir}/{dataset}_extrapolation.json")
    return results
