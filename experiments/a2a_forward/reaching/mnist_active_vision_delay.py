"""Delayed-feedback causal arm: does the efference-copy forecast become causally
load-bearing when there is a gap between deciding-to-look and getting the content?

The k=1 same-step causal arm (mnist_active_vision_causal.py) was inert: the glimpse
content arrives the same step as the forecast, so the content-blind anticipatory
forecast is redundant and the gate never opens (dependency ~0). Diagnosis: an
efference copy has no causal value without a feedback DELAY -- biologically it
matters precisely because sensory feedback is delayed, so you must act on the
prediction before the truth arrives.

This experiment adds that delay. Command u_t's glimpse CONTENT arrives `content_delay`
= d steps late, but the efference copy u_t (and the forecast FM_eff(s_t,u_t)) is
available immediately. During the d-step gap the ONLY signal about the pending look
is the forecast:

  CL_eff:   inject gate*FM_eff(s_t, u_t)  -- command-aware -> can pre-integrate the
            specific pending look u_t during the delay.
  CL_state: inject gate*FM_state(s_t)     -- command-blind -> carries no info about
            WHICH look is pending; cannot anticipate (the control).
  OL:       no injection                  -- must wait d steps for content.

The FM predicts the CLEAN counterfactual immediate look-update Δ*_t = G(s_t +
glimpse(x,u_t)) - s_t (a side computation; a self-model of "what looking at u_t does
to me now"), so there is no multi-step dilution -- the delay lives only in when the
real content arrives, not in what the FM predicts. The forecast is then injected
anticipatorily.

Prediction (the whole point): at d=0 this reproduces the causal null (gate closed,
dependency ~0); as d grows, CL_eff's dependency and gate should RISE while CL_state
stays flat -- the efference copy's causal value = bridging the feedback delay.

Run (delay sweep; MNIST + Fashion, 3 conditions x 4 delays):
  for d in 0 1 2 3; do for c in ol cl_state cl_eff; do
    modal run --detach a2a_forward/mnist_active_vision_delay.py::train_condition \
      --condition $c --dataset fashion_mnist --content-delay $d
  done; done
Aggregate:
  modal run a2a_forward/mnist_active_vision_delay.py::aggregate --dataset fashion_mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["ol", "cl_state", "cl_eff"]
DELAYS = [0, 1, 2, 3]


def _tag(dataset, glimpse_grid, n_cmd, n_head, n_embd):
    return f"{dataset}_g{glimpse_grid}_T{n_cmd}_{n_head}H{n_embd}D"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def train_condition(
    condition: str = "cl_eff",
    dataset: str = "fashion_mnist",
    content_delay: int = 1,
    glimpse_grid: int = 3,
    n_loop_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    n_cmd: int = 8,                  # number of glimpse commands (loop runs n_cmd + d)
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 5000,
    eval_interval: int = 250,
    n_eval_batches: int = 5,
    gate_max: float = 1.0,
    fwd_d_head: int = 16,
    fwd_mlp_mult: float = 2.0,
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.looped_vit import GlimpseLoopedViT
    from a2a_forward.forward_model import TransformerForwardModel, BoundedScalarGate

    assert condition in CONDITIONS, condition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = content_delay
    R = n_cmd + d                    # total loop steps (drain all commands' content)
    n_positions = (28 // patch_size) ** 2 + 1
    n_centers = (28 // patch_size) ** 2
    closed_loop = condition in ("cl_state", "cl_eff")
    use_u = condition == "cl_eff"
    print(f"DELAY [{condition}] {dataset} d={d} (R={R}) on {device}. "
          f"closed_loop={closed_loop} use_u={use_u}")

    # --- Data ---
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    ds = load_dataset(ds_name)
    tr = np.stack([np.array(im) for im in ds["train"]["image"]])
    train_images = torch.from_numpy(tr).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    te = np.stack([np.array(im) for im in ds["test"]["image"]])
    test_images = torch.from_numpy(te).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    g_idx = torch.Generator().manual_seed(seed)
    g_u = torch.Generator().manual_seed(seed + 1)
    g_eval = torch.Generator().manual_seed(seed + 2)

    def sample_u(B, gen, length=n_cmd):
        return torch.randint(n_centers, (length, B), generator=gen)

    # --- Models (identical init across conditions) ---
    torch.manual_seed(seed)
    model = GlimpseLoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        glimpse_grid=glimpse_grid, full_view=False,
        n_loop_layers=n_loop_layers, n_head=n_head, n_embd=n_embd, n_steps=R,
    ).to(device)
    torch.manual_seed(seed + 100)
    fwd = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=1, n_layer=1,
        mlp_mult=fwd_mlp_mult, block_size=n_positions, causal=False,
    ).to(device)
    reveal_vec = None
    if use_u:
        reveal_vec = torch.zeros(n_embd, device=device, requires_grad=True)
        with torch.no_grad():
            reveal_vec.normal_(std=0.02)
    gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device) if closed_loop else None

    main_params = list(model.parameters()) + (list(gate.parameters()) if gate else [])
    fwd_params = list(fwd.parameters()) + ([reveal_vec] if use_u else [])
    opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
    opt_fwd = torch.optim.AdamW(fwd_params, lr=fwd_lr, weight_decay=0.01)

    def reveal_mask_for(u_idx):
        m = model.reveal_mask[u_idx]
        cls0 = torch.zeros(m.shape[0], 1, device=device)
        return torch.cat([cls0, m], dim=1)

    def fm_input(s, u):
        if not use_u:
            return s
        return s + reveal_mask_for(u).unsqueeze(-1) * reveal_vec

    def make_hook(u_seq):
        """Inject the (detached) gated forecast of command u_seq[t]'s immediate
        look-effect. Only while commands are being issued (t < n_cmd)."""
        nc = u_seq.shape[0]
        def hook(t, operand, state):
            if t >= nc:
                return None
            pred = fwd(fm_input(state.detach(), u_seq[t]))
            return gate(pred.detach())
        return hook

    def forward_loop(images, u_seq, n_commands=None):
        nc = u_seq.shape[0] if n_commands is None else n_commands
        hook = make_hook(u_seq) if closed_loop else None
        logits, _, inter = model(images, u_seq, return_intermediates=True,
                                 n_steps=nc + d, content_delay=d, step_inject_fn=hook)
        return logits, inter

    def fm_loss_from(images, u_seq, inter):
        """FM predicts the counterfactual immediate look-update Δ*_t = one_step(s_t,
        u_t) - s_t (content-blind input for the model; full-content target)."""
        B = images.shape[0]
        patch_emb = inter["patch_emb"]
        zeros = torch.zeros(B, n_positions, n_embd, device=device)
        loss = 0.0
        for t in range(n_cmd):
            s_t = inter[f"post_step{t - 1}"] if t >= 1 else zeros
            target = (model.one_step(s_t, patch_emb, u_seq[t]) - s_t).detach()
            pred = fwd(fm_input(s_t.detach(), u_seq[t]))
            loss = loss + F.mse_loss(pred, target)
        return loss / n_cmd

    def evaluate():
        model.eval(); fwd.eval()
        if gate:
            gate.eval()
        acc = loss = acc_ni = fmcos = 0.0
        with torch.no_grad():
            for _ in range(n_eval_batches):
                idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
                vim = test_images[idx].to(device)
                vlb = test_labels[idx].to(device)
                u = sample_u(vim.shape[0], g_eval).to(device)
                logits, inter = forward_loop(vim, u)
                acc += (logits.argmax(-1) == vlb).float().mean().item()
                loss += F.cross_entropy(logits, vlb).item()
                # FM cos vs counterfactual look-update
                B = vim.shape[0]
                patch_emb = inter["patch_emb"]
                zeros = torch.zeros(B, n_positions, n_embd, device=device)
                for t in range(n_cmd):
                    s_t = inter[f"post_step{t - 1}"] if t >= 1 else zeros
                    tgt = model.one_step(s_t, patch_emb, u[t]) - s_t
                    pred = fwd(fm_input(s_t, u[t]))
                    fmcos += F.cosine_similarity(pred, tgt, dim=-1).mean().item() / n_cmd
                if closed_loop:
                    # ablate injection
                    logits2, _, _ = model(vim, u, return_intermediates=True,
                                          n_steps=n_cmd + d, content_delay=d)
                    acc_ni += (logits2.argmax(-1) == vlb).float().mean().item()
        n = n_eval_batches
        return acc / n, loss / n, acc_ni / n, fmcos / n

    # =============================================
    # Train
    # =============================================
    hist = {"train_loss": [], "val_acc": [], "val_loss": [], "fm_cos": []}
    if closed_loop:
        hist["val_acc_no_inj"] = []
        hist["gate"] = []

    for step in range(n_steps):
        model.train(); fwd.train()
        if gate:
            gate.train()
        idx = torch.randint(len(train_images), (batch_size,), generator=g_idx)
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)
        u = sample_u(images.shape[0], g_u).to(device)

        logits, inter = forward_loop(images, u)
        cls_loss = F.cross_entropy(logits, labels)
        fm_loss = fm_loss_from(images, u, inter)

        opt_main.zero_grad(); cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, 1.0)
        opt_main.step()
        opt_fwd.zero_grad(); fm_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_params, 1.0)
        opt_fwd.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            acc, loss, acc_ni, fmcos = evaluate()
            hist["train_loss"].append((step, cls_loss.item()))
            hist["val_acc"].append((step, acc))
            hist["val_loss"].append((step, loss))
            hist["fm_cos"].append((step, fmcos))
            extra = ""
            if closed_loop:
                hist["val_acc_no_inj"].append((step, acc_ni))
                hist["gate"].append((step, gate.injection_norm()))
                extra = (f" | no_inj={acc_ni:.4f} dep={acc - acc_ni:+.4f}"
                         f" gate={gate.injection_norm():.4f}")
            print(f"  step {step:5d}: val_acc={acc:.4f} loss={loss:.4f} "
                  f"fm_cos={fmcos:.4f}{extra}")

    # acceleration: accuracy vs #commands (with/without injection)
    def acc_vs_ncmd(use_injection):
        model.eval(); fwd.eval()
        if gate:
            gate.eval()
        out = {}
        with torch.no_grad():
            for Tc in [1, 2, 3, 4, 5, 6, 8]:
                if Tc > n_cmd:
                    continue
                a = 0.0
                for bi in range(n_eval_batches):
                    idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
                    vim = test_images[idx].to(device)
                    vlb = test_labels[idx].to(device)
                    u = sample_u(vim.shape[0], g_eval, length=Tc).to(device)
                    hook = make_hook(u) if (use_injection and closed_loop) else None
                    logits, _, _ = model(vim, u, return_intermediates=True,
                                         n_steps=Tc + d, content_delay=d,
                                         step_inject_fn=hook)
                    a += (logits.argmax(-1) == vlb).float().mean().item()
                out[str(Tc)] = a / n_eval_batches
        return out

    accT = {"with_injection": acc_vs_ncmd(True)}
    if closed_loop:
        accT["no_injection"] = acc_vs_ncmd(False)

    # =============================================
    # Save
    # =============================================
    tag = _tag(dataset, glimpse_grid, n_cmd, n_head, n_embd)
    save_dir = (f"{DATA_DIR}/a2a_forward/mnist_active_vision_delay/{tag}/"
                f"d{d}/{condition}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    result = {
        "condition": condition, "dataset": dataset, "content_delay": d,
        "closed_loop": closed_loop, "use_u": use_u,
        "config": {"glimpse_grid": glimpse_grid, "n_cmd": n_cmd, "n_embd": n_embd,
                   "n_head": n_head, "fwd_d_head": fwd_d_head, "n_steps": n_steps,
                   "seed": seed, "gate_max": gate_max},
        "history": hist,
        "final_val_acc": hist["val_acc"][-1][1],
        "final_val_loss": hist["val_loss"][-1][1],
        "final_fm_cos": hist["fm_cos"][-1][1],
        "acc_vs_ncmd": accT,
    }
    if closed_loop:
        result["final_val_acc_no_inj"] = hist["val_acc_no_inj"][-1][1]
        result["dependency"] = hist["val_acc"][-1][1] - hist["val_acc_no_inj"][-1][1]
        result["final_gate"] = hist["gate"][-1][1]

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\n{'=' * 60}\n  DELAY {condition} {dataset} d={d}\n{'=' * 60}")
    print(f"  val_acc={result['final_val_acc']:.4f} fm_cos={result['final_fm_cos']:.4f}")
    if closed_loop:
        print(f"  dependency={result['dependency']:+.4f} gate={result['final_gate']:.4f}")
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=600)
def aggregate(
    dataset: str = "fashion_mnist",
    glimpse_grid: int = 3, n_cmd: int = 8, n_head: int = 4, n_embd: int = 128,
):
    import os
    tag = _tag(dataset, glimpse_grid, n_cmd, n_head, n_embd)
    base = f"{DATA_DIR}/a2a_forward/mnist_active_vision_delay/{tag}"
    print(f"\n{'=' * 74}\n  DELAYED-FEEDBACK CAUSAL ARM  ({dataset})\n{'=' * 74}")
    print(f"  delay sweep -- dependency (Δacc on ablating injection) and gate:\n")
    print(f"  {'d':>2s} | {'ol_acc':>7s} | {'cl_state dep/gate':>20s} | "
          f"{'cl_eff dep/gate':>20s} | {'eff-state dep':>13s}")
    grid = {}
    for d in DELAYS:
        rows = {}
        for c in CONDITIONS:
            p = os.path.join(base, f"d{d}", c, "results.json")
            if os.path.exists(p):
                with open(p) as f:
                    rows[c] = json.load(f)
        grid[d] = rows
        if not rows:
            continue
        ol = rows.get("ol", {}).get("final_val_acc", float("nan"))
        ds_ = rows.get("cl_state", {})
        de_ = rows.get("cl_eff", {})
        dep_s = ds_.get("dependency", float("nan"))
        dep_e = de_.get("dependency", float("nan"))
        g_s = ds_.get("final_gate", float("nan"))
        g_e = de_.get("final_gate", float("nan"))
        print(f"  {d:>2d} | {ol:7.4f} | {dep_s:+.4f}/{g_s:.4f}      | "
              f"{dep_e:+.4f}/{g_e:.4f}      | {dep_e - dep_s:+.4f}")
    print("\n  prediction: eff-state dep ~0 at d=0 (reproduces null), RISES with d")
    # accuracy vs #commands at the largest delay, per condition
    dmax = max(d for d in DELAYS if grid.get(d))
    print(f"\n  acc vs #commands (with inj) at d={dmax}:")
    for c in CONDITIONS:
        r = grid[dmax].get(c)
        if not r:
            continue
        at = r["acc_vs_ncmd"]["with_injection"]
        print(f"    {c:9s}: " + " ".join(f"T{k}={at[k]:.2f}" for k in sorted(at, key=int)))
    return grid
