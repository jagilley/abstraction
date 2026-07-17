"""Causal arm of the "missing u" test: does the looped model CAUSALLY USE an
efference-copy-conditioned forecast in a way it cannot use a command-blind one?

Observational arm (mnist_active_vision.py) showed the efference copy *carries*
predictive info about the loop's own dynamics that no capacity of an arity-1 f(s)
can recover. This arm closes the loop with the forecast injected each step and asks
whether the loop *uses* it -- contrasting arity-2 vs arity-1 as the manipulated
variable, co-trained (producer = consumer):

  CL_eff:    s_{t+1} = G( s_t + glimpse(x,u_t) + gate*FM_eff(s_t, u_t) )
  CL_state:  s_{t+1} = G( s_t + glimpse(x,u_t) + gate*FM_state(s_t)   )
  OL:        s_{t+1} = G( s_t + glimpse(x,u_t) )

Mechanism (why it's not redundant): FM_eff(s_t,u_t) is CONTENT-BLIND -- the expected
update given "I am about to look at region u_t", before the pixels arrive. Injected
before G, it supplies the command-conditional expectation so G can spend capacity on
the content correction (the surprise), a genuine division of labor. FM_state supplies
only the command-AVERAGED expectation -- a worse prior. So CL_eff should out-depend
CL_state on the command-driven component.

Framing carried from LOOPED_README: do NOT expect lower loss (the injection adds a
modality to parse; CL ~ OL on loss is expected). The discriminator is DEPENDENCY /
causal load-bearing and ACCELERATION, read as the (condition x mode) interaction.

The airtight control: --full-view makes the command inert, so FM_eff == FM_state
(the marker is disabled). CL_eff MUST equal CL_state in full-view. Any CL_eff-over-
CL_state advantage that appears in glimpse mode but not full-view is attributable to
the command-conditioning and nothing else (rules out "a better forecast helps" and
"any injection helps").

Design decisions (deferred to judgement): k=1 (command-conditional current-step
update; division-of-labor works at k=1); CL_state is the control (a real forecast,
just command-blind -- strictly stronger than random_proj), full-view handles the
"any injection" confound; random_proj/shifted + the corollary-discharge cancellation
probe are deferred follow-ons.

Run (9 combos: MNIST {ol,cl_state,cl_eff}x{glimpse,full_view} + Fashion {..}xglimpse):
  for c in ol cl_state cl_eff; do
    modal run --detach a2a_forward/mnist_active_vision_causal.py::train_condition --condition $c --dataset mnist
    modal run --detach a2a_forward/mnist_active_vision_causal.py::train_condition --condition $c --dataset mnist --full-view
    modal run --detach a2a_forward/mnist_active_vision_causal.py::train_condition --condition $c --dataset fashion_mnist
  done
Aggregate:
  modal run a2a_forward/mnist_active_vision_causal.py::aggregate --dataset mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["ol", "cl_state", "cl_eff"]


def _tag(dataset, full_view, glimpse_grid, n_glimpses, n_head, n_embd):
    mode = "full_view" if full_view else "glimpse"
    return f"{dataset}_{mode}_g{glimpse_grid}_T{n_glimpses}_{n_head}H{n_embd}D"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def train_condition(
    condition: str = "cl_eff",
    dataset: str = "mnist",
    full_view: bool = False,
    glimpse_grid: int = 3,
    n_loop_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    n_glimpses: int = 8,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 5000,
    eval_interval: int = 250,
    n_eval_batches: int = 5,
    predict_k: int = 1,
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
    T = n_glimpses
    n_positions = (28 // patch_size) ** 2 + 1
    n_centers = (28 // patch_size) ** 2
    mode = "full_view" if full_view else "glimpse"
    closed_loop = condition in ("cl_state", "cl_eff")
    # arity-2 (command-aware FM) only for cl_eff AND when the command is live.
    use_u = (condition == "cl_eff") and (not full_view)
    print(f"CAUSAL [{condition}] {dataset}/{mode} on {device}. "
          f"closed_loop={closed_loop} use_u={use_u} k={predict_k}")

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

    def sample_u(B, gen, length=None):
        return torch.randint(n_centers, (length or T, B), generator=gen)

    # --- Models (identical init across conditions via shared seeds) ---
    torch.manual_seed(seed)
    model = GlimpseLoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        glimpse_grid=glimpse_grid, full_view=full_view,
        n_loop_layers=n_loop_layers, n_head=n_head, n_embd=n_embd, n_steps=T,
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
        """Inject the (detached) gated forecast of the update at each step. The FM
        reads the pre-glimpse state s_t (`state`) and command u_seq[t] -- content-
        blind -- so the injection is the command-conditional *expected* update."""
        def hook(t, operand, state):
            pred = fwd(fm_input(state.detach(), u_seq[t]))   # forecast Δ_t
            return gate(pred.detach())
        return hook

    def run(images, u_seq, want_inter=False):
        hook = make_hook(u_seq) if closed_loop else None
        logits, _, inter = model(images, u_seq, return_intermediates=True,
                                 step_inject_fn=hook)
        # FM loss on the (injection-laden) transitions: predict Δ_t = s_{t+1}-s_t.
        B = images.shape[0]
        zeros = torch.zeros(B, n_positions, n_embd, device=device)
        fm_loss = 0.0
        for t in range(T):
            s_t = inter[f"post_step{t - 1}"] if t >= 1 else zeros
            target = (inter[f"post_step{t}"] - s_t).detach()
            pred = fwd(fm_input(s_t.detach(), u_seq[t]))
            fm_loss = fm_loss + F.mse_loss(pred, target)
        fm_loss = fm_loss / T
        return (logits, fm_loss, inter) if want_inter else (logits, fm_loss)

    # =============================================
    # Train (classification on final logits; FM on MSE)
    # =============================================
    hist = {"train_loss": [], "val_acc": [], "val_loss": [], "fm_cos": []}
    if closed_loop:
        hist["val_acc_no_inj"] = []
        hist["val_loss_no_inj"] = []
        hist["gate"] = []

    def evaluate():
        model.eval(); fwd.eval()
        if gate:
            gate.eval()
        acc = loss = acc_ni = loss_ni = fmcos = 0.0
        with torch.no_grad():
            for _ in range(n_eval_batches):
                idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
                vim = test_images[idx].to(device)
                vlb = test_labels[idx].to(device)
                u = sample_u(vim.shape[0], g_eval).to(device)
                hook = make_hook(u) if closed_loop else None
                vlog, vl, inter = model(vim, u, vlb, return_intermediates=True,
                                        step_inject_fn=hook)
                acc += (vlog.argmax(-1) == vlb).float().mean().item()
                loss += vl.item()
                # FM cos on the transitions
                B = vim.shape[0]
                zeros = torch.zeros(B, n_positions, n_embd, device=device)
                for t in range(T):
                    s_t = inter[f"post_step{t - 1}"] if t >= 1 else zeros
                    tgt = inter[f"post_step{t}"] - s_t
                    pred = fwd(fm_input(s_t, u[t]))
                    fmcos += F.cosine_similarity(pred, tgt, dim=-1).mean().item() / T
                if closed_loop:
                    vlog2, vl2 = model(vim, u, vlb)  # no injection
                    acc_ni += (vlog2.argmax(-1) == vlb).float().mean().item()
                    loss_ni += vl2.item()
        n = n_eval_batches
        return (acc / n, loss / n, acc_ni / n, loss_ni / n, fmcos / n)

    for step in range(n_steps):
        model.train(); fwd.train()
        if gate:
            gate.train()
        idx = torch.randint(len(train_images), (batch_size,), generator=g_idx)
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)
        u = sample_u(images.shape[0], g_u).to(device)

        logits, fm_loss = run(images, u)
        cls_loss = F.cross_entropy(logits, labels)

        opt_main.zero_grad(); cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, 1.0)
        opt_main.step()

        opt_fwd.zero_grad(); fm_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_params, 1.0)
        opt_fwd.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            acc, loss, acc_ni, loss_ni, fmcos = evaluate()
            hist["train_loss"].append((step, cls_loss.item()))
            hist["val_acc"].append((step, acc))
            hist["val_loss"].append((step, loss))
            hist["fm_cos"].append((step, fmcos))
            extra = ""
            if closed_loop:
                hist["val_acc_no_inj"].append((step, acc_ni))
                hist["val_loss_no_inj"].append((step, loss_ni))
                hist["gate"].append((step, gate.injection_norm()))
                extra = (f" | no_inj acc={acc_ni:.4f} dep={acc - acc_ni:+.4f}"
                         f" gate={gate.injection_norm():.4f}")
            print(f"  step {step:5d}: val_acc={acc:.4f} loss={loss:.4f} "
                  f"fm_cos={fmcos:.4f}{extra}")

    # =============================================
    # Acceleration: accuracy vs #glimpses (with and without injection)
    # =============================================
    def acc_vs_T(use_injection):
        model.eval(); fwd.eval()
        if gate:
            gate.eval()
        out = {}
        with torch.no_grad():
            for Te in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                a = l = 0.0
                for bi in range(n_eval_batches):
                    idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
                    vim = test_images[idx].to(device)
                    vlb = test_labels[idx].to(device)
                    u = sample_u(vim.shape[0], g_eval, length=Te).to(device)
                    hook = make_hook(u) if (use_injection and closed_loop) else None
                    vlog, vl = model(vim, u, vlb, n_steps=Te, step_inject_fn=hook)
                    a += (vlog.argmax(-1) == vlb).float().mean().item()
                    l += vl.item()
                out[str(Te)] = {"acc": a / n_eval_batches, "loss": l / n_eval_batches}
        return out

    accT = {"with_injection": acc_vs_T(True)}
    if closed_loop:
        accT["no_injection"] = acc_vs_T(False)

    # =============================================
    # Save
    # =============================================
    tag = _tag(dataset, full_view, glimpse_grid, T, n_head, n_embd)
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_active_vision_causal/{tag}/{condition}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd.state_dict(), os.path.join(save_dir, "fwd.pt"))
    if gate:
        torch.save(gate.state_dict(), os.path.join(save_dir, "gate.pt"))

    result = {
        "condition": condition, "dataset": dataset, "mode": mode,
        "closed_loop": closed_loop, "use_u": use_u, "predict_k": predict_k,
        "config": {"glimpse_grid": glimpse_grid, "T": T, "n_embd": n_embd,
                   "n_head": n_head, "fwd_d_head": fwd_d_head, "n_steps": n_steps,
                   "seed": seed, "gate_max": gate_max},
        "history": hist,
        "final_val_acc": hist["val_acc"][-1][1],
        "final_val_loss": hist["val_loss"][-1][1],
        "final_fm_cos": hist["fm_cos"][-1][1],
        "acc_vs_T": accT,
    }
    if closed_loop:
        result["final_val_acc_no_inj"] = hist["val_acc_no_inj"][-1][1]
        result["final_val_loss_no_inj"] = hist["val_loss_no_inj"][-1][1]
        result["dependency"] = hist["val_acc"][-1][1] - hist["val_acc_no_inj"][-1][1]
        result["injection_benefit"] = hist["val_loss"][-1][1] - hist["val_loss_no_inj"][-1][1]
        result["final_gate"] = hist["gate"][-1][1]

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\n{'=' * 60}\n  CAUSAL {condition} {dataset}/{mode}\n{'=' * 60}")
    print(f"  val_acc={result['final_val_acc']:.4f} fm_cos={result['final_fm_cos']:.4f}")
    if closed_loop:
        print(f"  dependency(Δacc on ablate)={result['dependency']:+.4f} "
              f"inj_benefit(Δloss)={result['injection_benefit']:+.4f} "
              f"gate={result['final_gate']:.4f}")
    at = accT["with_injection"]
    print(f"  acc_vs_#glimpses (with inj): "
          + " ".join(f"T{k}={v['acc']:.2f}" for k, v in at.items()))
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=600)
def aggregate(
    dataset: str = "mnist",
    glimpse_grid: int = 3, n_glimpses: int = 8, n_head: int = 4, n_embd: int = 128,
):
    import os
    T = n_glimpses
    print(f"\n{'=' * 74}\n  CAUSAL ARM  ({dataset})\n{'=' * 74}")
    for full_view in [False, True]:
        mode = "full_view" if full_view else "glimpse"
        tag = _tag(dataset, full_view, glimpse_grid, T, n_head, n_embd)
        base = f"{DATA_DIR}/a2a_forward/mnist_active_vision_causal/{tag}"
        rows = {}
        for c in CONDITIONS:
            p = os.path.join(base, c, "results.json")
            if os.path.exists(p):
                with open(p) as f:
                    rows[c] = json.load(f)
        if not rows:
            continue
        print(f"\n  --- {mode} ---")
        print(f"  {'cond':9s} {'val_acc':>8s} {'acc_noinj':>10s} {'dependency':>11s} "
              f"{'inj_ben':>8s} {'gate':>7s} {'fm_cos':>7s}")
        for c in CONDITIONS:
            if c not in rows:
                continue
            r = rows[c]
            dep = r.get("dependency", float("nan"))
            ib = r.get("injection_benefit", float("nan"))
            gt = r.get("final_gate", float("nan"))
            ani = r.get("final_val_acc_no_inj", float("nan"))
            print(f"  {c:9s} {r['final_val_acc']:8.4f} {ani:10.4f} {dep:+11.4f} "
                  f"{ib:+8.4f} {gt:7.4f} {r['final_fm_cos']:7.4f}")
        # the interaction: does cl_eff out-depend cl_state?
        if "cl_eff" in rows and "cl_state" in rows:
            de = rows["cl_eff"].get("dependency", float("nan"))
            dstt = rows["cl_state"].get("dependency", float("nan"))
            print(f"  >>> dependency gap cl_eff - cl_state = {de - dstt:+.4f} "
                  f"(expect >0 in glimpse, ~0 in full_view)")
        # acceleration: acc at low T, with injection, per condition
        print(f"  acc_vs_#glimpses (with inj):")
        for c in CONDITIONS:
            if c not in rows:
                continue
            at = rows[c]["acc_vs_T"]["with_injection"]
            print(f"    {c:9s}: " + " ".join(f"T{k}={at[k]['acc']:.2f}"
                  for k in ["1", "2", "3", "4", "5", "6", "8"]))
    return True
