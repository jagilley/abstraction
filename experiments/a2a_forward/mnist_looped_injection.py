"""Phase 2: FM injection into the looped ViT -- the 2x2 for the self-model loop.

Tests the core prediction of ideas/self_model_needs_a_loop.md: a weight-shared
looped model with next-state FM injection is forced toward local self-consistency
(a fixed point that agrees with a forecast of itself), which a feedforward /
plain-loop model is not.

Architecture (see looped_vit.py): prelude(1) -> shared core(1) x T=8 -> coda(1).
The FM reads the recurrent operand a_t = s_t + p, predicts the next state s_{t+1},
and its gated forecast is injected into the operand each step:

    s_{t+1} = G( s_t + p + gate * FM(s_t + p) ),   FM trained on MSE(FM(a_t), s_{t+1})

This is the near-exact analog of the feedforward a2a setup (FM: post_block0 ->
post_block3, inject after block1), but with producer = consumer forced by weight
sharing.

The 2x2 (each a `condition`, run in a separate container, shared seed/init/data):

                 last-step-only supervision | deep supervision
  OL (no inject)   ol_last (phase-1: drifts) | ol_deep (converges, scaffolded)
  CL (FM inject)   cl_last  <-- money cell    | cl_deep

Money cell: does FM injection make cl_last converge/stay-stable past train-T where
ol_last overshoots? If yes, the self-model supplies endogenous convergence (halting)
that deep supervision was faking -- i.e. deep supervision is scaffolding the FM
replaces (the user's "scrap it once self-knowledge suffices to halt" point).

Run (4 parallel detached containers):
  for c in ol_last ol_deep cl_last cl_deep; do
    modal run --detach a2a_forward/mnist_looped_injection.py::train_condition --condition $c
  done
Aggregate afterwards:
  modal run a2a_forward/mnist_looped_injection.py::aggregate
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["ol_last", "ol_deep", "cl_last", "cl_deep"]


def _cfg_tag(n_loop_layers, train_steps_loop, n_head, n_embd,
             prelude_layers, coda_layers, injection_form="update", predict_k=3):
    return (f"looped_{n_loop_layers}x{train_steps_loop}_{n_head}H_{n_embd}D"
            f"_pre{prelude_layers}coda{coda_layers}_{injection_form}_k{predict_k}")


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def train_condition(
    condition: str = "cl_last",
    dataset: str = "mnist",          # "mnist" or "fashion_mnist" (harder -> loop needed)
    baseline_type: str = "forward",  # baseline battery: what gets injected --
                                     # "forward" = FM(a_t) forecast (the real thing);
                                     # "random_proj" = frozen random-init FM (structured
                                     #   but not a forecast); "shifted" = FM forecast
                                     #   rolled along positions (destroys alignment)
    injection_form: str = "update",  # "update" = gate*(FM-s_t) (vanishes at fixpt);
                                     # "next_state" = gate*FM (naive, unstable)
    predict_k: int = 3,              # FM predicts s_{t+k} (k>1 -> informative preview;
                                     # k=1 next-step is trivially easy -> inert)
    damping: float = 1.0,            # <1.0 = damped update s+=alpha*(G-s) for
                                     # convergence without deep supervision
    gate_type: str = "proj",         # "proj" = CerebellarGate (unbounded projection);
                                     # "scalar" = BoundedScalarGate (over-relaxation
                                     # coefficient in [0,1], stable)
    gate_max: float = 1.0,
    n_loop_layers: int = 1,
    prelude_layers: int = 1,
    coda_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    train_steps_loop: int = 8,
    eval_t_max: int = 20,
    deep_sup_frac: float = 0.5,
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    n_steps: int = 5000,
    eval_interval: int = 250,
    n_eval_batches: int = 5,
    probe_batches: int = 30,
    seed: int = 42,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_n_layer: int = 1,
    fwd_mlp_mult: float = 2.0,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.looped_vit import LoopedViT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate, BoundedScalarGate)

    assert condition in CONDITIONS, f"unknown condition {condition}"
    assert gate_type in ("proj", "scalar"), gate_type
    assert injection_form in ("update", "next_state"), injection_form
    assert baseline_type in ("forward", "random_proj", "shifted"), baseline_type
    closed_loop = condition.startswith("cl")
    deep_sup = condition.endswith("deep")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = train_steps_loop
    sup_start = int(T * (1.0 - deep_sup_frac))
    n_positions = (28 // patch_size) ** 2 + 1

    print(f"PHASE 2 condition={condition} (closed_loop={closed_loop}, "
          f"deep_sup={deep_sup}) on {device}")
    print(f"  prelude={prelude_layers} core={n_loop_layers}xT={T} coda={coda_layers}"
          f" {n_head}H {n_embd}D")
    if deep_sup:
        print(f"  deep supervision on steps {sup_start}..{T - 1}")

    # --- Data ---
    ds_name = {"mnist": "ylecun/mnist",
               "fashion_mnist": "zalando-datasets/fashion_mnist"}[dataset]
    print(f"Loading {dataset}...")
    ds = load_dataset(ds_name)
    tr_imgs = np.stack([np.array(im) for im in ds["train"]["image"]])
    train_images = torch.from_numpy(tr_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    te_imgs = np.stack([np.array(im) for im in ds["test"]["image"]])
    test_images = torch.from_numpy(te_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])

    # --- Reproducible indices (identical across conditions) ---
    train_gen = torch.Generator().manual_seed(seed)
    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    n_evals = n_steps // eval_interval + 2
    train_indices = [torch.randint(len(train_images), (batch_size,), generator=train_gen)
                     for _ in range(n_steps)]
    eval_indices = [[torch.randint(len(test_images), (batch_size,), generator=eval_gen)
                     for _ in range(n_eval_batches)] for _ in range(n_evals)]
    probe_indices = [torch.randint(len(test_images), (batch_size,), generator=probe_gen)
                     for _ in range(probe_batches)]

    # --- Models (main model init identical across all conditions) ---
    torch.manual_seed(seed)
    model = LoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        n_loop_layers=n_loop_layers, n_head=n_head, n_embd=n_embd, n_steps=T,
        prelude_layers=prelude_layers, coda_layers=coda_layers,
        inject_input_each_step=True,
    ).to(device)

    # Damped update (convergence without deep supervision): s_{t+1} = s_in +
    # alpha*(G(s_in) - s_in), applied by wrapping the operator.
    if damping < 1.0:
        _base_op = model._apply_operator
        model._apply_operator = lambda x: x + damping * (_base_op(x) - x)

    torch.manual_seed(seed + 100)
    fwd = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
        block_size=n_positions, causal=False,
    ).to(device)

    if closed_loop:
        if gate_type == "scalar":
            gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
        else:
            gate = CerebellarGate(n_embd).to(device)
    else:
        gate = None

    # Baseline-battery injected-signal source. The FM (fwd) is always trained for
    # measurement; only what gets INJECTED varies. random_proj injects a frozen
    # random-init FM (structured, same magnitude, but not a trained forecast).
    rand_predictor = None
    if baseline_type == "random_proj":
        torch.manual_seed(seed + 200)
        rand_predictor = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)
        for pp in rand_predictor.parameters():
            pp.requires_grad_(False)

    main_params = list(model.parameters()) + (list(gate.parameters()) if gate else [])
    opt_main = torch.optim.AdamW(main_params, lr=lr, weight_decay=0.01)
    opt_fwd = torch.optim.AdamW(fwd.parameters(), lr=fwd_lr, weight_decay=0.01)

    def make_hook():
        """Per-step injection hook. What is injected depends on baseline_type:
        forward = FM(a_t); random_proj = frozen random FM(a_t); shifted = FM(a_t)
        rolled along positions. injection_form controls next-state vs update."""
        def hook(t, operand, state):
            if baseline_type == "random_proj":
                pred = rand_predictor(operand.detach())
            else:
                pred = fwd(operand.detach())
                if baseline_type == "shifted":
                    pred = torch.roll(pred, shifts=n_positions // 2, dims=1)
            signal = (pred - state) if injection_form == "update" else pred
            return gate(signal.detach())
        return hook

    def cls_loss_from_steps(step_logits, labels):
        if deep_sup:
            sup = step_logits[sup_start:]
            return torch.stack([F.cross_entropy(sl, labels) for sl in sup]).mean()
        return F.cross_entropy(step_logits[-1], labels)

    def tgt_idx(t):
        """post_step index for the k-step-ahead target s_{t+k}, clamped to the
        final state. (post_step{i} == s_{i+1}; operand a_t uses s_t, so s_{t+k}
        == post_step{t+k-1}.)"""
        return min(t + predict_k - 1, T - 1)

    def run_model(images):
        """Forward the loop (with injection if CL); return logits, per-step logits,
        intermediates, and the FM loss. The FM is trained/measured on the
        (possibly injection-laden) transitions uniformly across all conditions."""
        if closed_loop:
            logits, _, inter, step_logits = model(
                images, return_intermediates=True, readout_all_steps=True,
                step_inject_fn=make_hook(),
            )
        else:
            logits, _, inter, step_logits = model(
                images, return_intermediates=True, readout_all_steps=True,
            )
        p = inter["post_prelude"]
        fm_loss = 0.0
        for t in range(T):
            s_prev = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
            a_t = (s_prev + p).detach()
            fm_loss = fm_loss + F.mse_loss(
                fwd(a_t), inter[f"post_step{tgt_idx(t)}"].detach())
        fm_loss = fm_loss / T
        return logits, step_logits, inter, fm_loss

    # =============================================
    # Train
    # =============================================
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "fm_mse": [],
               "val_fm_cos": []}
    if closed_loop:
        history["val_loss_no_inj"] = []
        history["val_acc_no_inj"] = []
        history["gate_norm"] = []
    eval_idx = 0

    for step in range(n_steps):
        model.train(); fwd.train()
        if gate:
            gate.train()
        idx = train_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        logits, step_logits, inter, fm_loss = run_model(images)
        cls_loss = cls_loss_from_steps(step_logits, labels)

        opt_main.zero_grad()
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(main_params, 1.0)
        opt_main.step()

        opt_fwd.zero_grad()
        fm_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd.parameters(), 1.0)
        opt_fwd.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval(); fwd.eval()
            if gate:
                gate.eval()
            v_loss = v_acc = v_fmcos = 0.0
            v_loss_ni = v_acc_ni = 0.0
            with torch.no_grad():
                for bi in range(n_eval_batches):
                    eidx = eval_indices[eval_idx][bi]
                    vim = test_images[eidx].to(device)
                    vlb = test_labels[eidx].to(device)
                    if closed_loop:
                        vlog, vl, vinter = model(
                            vim, vlb, return_intermediates=True,
                            step_inject_fn=make_hook())
                        v_loss += vl.item()
                        v_acc += (vlog.argmax(-1) == vlb).float().mean().item()
                        vlog2, vl2, _ = model(vim, vlb, return_intermediates=True)
                        v_loss_ni += vl2.item()
                        v_acc_ni += (vlog2.argmax(-1) == vlb).float().mean().item()
                    else:
                        vlog, vl, vinter = model(
                            vim, vlb, return_intermediates=True)
                        v_loss += vl.item()
                        v_acc += (vlog.argmax(-1) == vlb).float().mean().item()
                    # FM cosine: pred(a_{T-1}) vs actual last state
                    p = vinter["post_prelude"]
                    s_prev = vinter[f"post_step{T - 2}"] if T >= 2 else torch.zeros_like(p)
                    a_last = s_prev + p
                    pred = fwd(a_last)
                    v_fmcos += F.cosine_similarity(
                        pred, vinter[f"post_step{T - 1}"], dim=-1).mean().item()
            n = n_eval_batches
            history["train_loss"].append((step, cls_loss.item()))
            history["val_loss"].append((step, v_loss / n))
            history["val_acc"].append((step, v_acc / n))
            history["fm_mse"].append((step, fm_loss.item()))
            history["val_fm_cos"].append((step, v_fmcos / n))
            extra = ""
            if closed_loop:
                history["val_loss_no_inj"].append((step, v_loss_ni / n))
                history["val_acc_no_inj"].append((step, v_acc_ni / n))
                history["gate_norm"].append((step, gate.injection_norm()))
                extra = (f" Δ={(v_loss - v_loss_ni) / n:+.4f}"
                         f" gate={gate.injection_norm():.3f}")
            print(f"  step {step:5d}: val_loss={v_loss / n:.4f} "
                  f"val_acc={v_acc / n:.4f}{extra} fm_cos={v_fmcos / n:.4f}")
            eval_idx += 1

    # =============================================
    # Convergence dynamics (with injection for CL)
    # =============================================
    model.eval(); fwd.eval()
    if gate:
        gate.eval()
    conv = np.zeros(T)
    with torch.no_grad():
        for bi in range(20):
            eidx = eval_indices[bi % n_evals][0]
            vim = test_images[eidx].to(device)
            if closed_loop:
                _, _, traj = model(vim, return_trajectory=True,
                                   step_inject_fn=make_hook())
            else:
                _, _, traj = model(vim, return_trajectory=True)
            conv += np.nan_to_num(np.array(traj["rel_delta"]))
    conv /= 20

    # =============================================
    # Accuracy vs eval-T (CL: with AND without injection = causal necessity)
    # =============================================
    def acc_vs_T(use_injection):
        out = {}
        with torch.no_grad():
            for Te in range(1, eval_t_max + 1):
                a = l = 0.0
                for bi in range(n_eval_batches):
                    eidx = eval_indices[0][bi]
                    vim = test_images[eidx].to(device)
                    vlb = test_labels[eidx].to(device)
                    if use_injection and closed_loop:
                        vlog, vl = model(vim, vlb, n_steps=Te,
                                         step_inject_fn=make_hook())
                    else:
                        vlog, vl = model(vim, vlb, n_steps=Te)
                    a += (vlog.argmax(-1) == vlb).float().mean().item()
                    l += vl.item()
                out[str(Te)] = {"acc": a / n_eval_batches, "loss": l / n_eval_batches}
        return out

    acc_T = {"with_injection": acc_vs_T(True)}
    if closed_loop:
        acc_T["no_injection"] = acc_vs_T(False)

    # =============================================
    # Per-step class decodability (premature-abstraction diagnostic)
    # linear probe from cls position of s_t -> class, trained per step.
    # =============================================
    step_states = {t: [] for t in range(T)}
    probe_labels = []
    with torch.no_grad():
        for bi in range(probe_batches):
            pim = test_images[probe_indices[bi]].to(device)
            plb = test_labels[probe_indices[bi]]
            if closed_loop:
                _, _, inter = model(pim, return_intermediates=True,
                                    step_inject_fn=make_hook())
            else:
                _, _, inter = model(pim, return_intermediates=True)
            for t in range(T):
                step_states[t].append(inter[f"post_step{t}"][:, 0].cpu())  # cls pos
            probe_labels.append(plb)
    y_all = torch.cat(probe_labels)
    n_tot = y_all.shape[0]
    n_tr = int(0.8 * n_tot)
    perm = torch.randperm(n_tot, generator=torch.Generator().manual_seed(seed))
    tr, te = perm[:n_tr], perm[n_tr:]
    per_step_decode = {}
    for t in range(T):
        X = torch.cat(step_states[t])
        probe = nn.Linear(n_embd, 10).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        Xtr, Ytr = X[tr].to(device), y_all[tr].to(device)
        Xte, Yte = X[te].to(device), y_all[te].to(device)
        for _ in range(200):
            bidx = torch.randint(n_tr, (min(2048, n_tr),))
            loss = F.cross_entropy(probe(Xtr[bidx]), Ytr[bidx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = (probe(Xte).argmax(-1) == Yte).float().mean().item()
        per_step_decode[str(t)] = acc

    # =============================================
    # FM self-consistency per step: cos(FM(a_t), s_{t+1})
    # (at the resting state this is the fixed-point agreement)
    # =============================================
    fm_selfcons = np.zeros(T)
    with torch.no_grad():
        for bi in range(20):
            eidx = eval_indices[bi % n_evals][0]
            vim = test_images[eidx].to(device)
            if closed_loop:
                _, _, inter = model(vim, return_intermediates=True,
                                    step_inject_fn=make_hook())
            else:
                _, _, inter = model(vim, return_intermediates=True)
            p = inter["post_prelude"]
            for t in range(T):
                s_prev = inter[f"post_step{t - 1}"] if t >= 1 else torch.zeros_like(p)
                a_t = s_prev + p
                pred = fwd(a_t)
                fm_selfcons[t] += F.cosine_similarity(
                    pred, inter[f"post_step{tgt_idx(t)}"], dim=-1).mean().item()
    fm_selfcons /= 20

    # =============================================
    # Perturbation robustness / error-correction
    # Perturb the recurrent state at a mid step (relative-scaled noise, identical
    # across conditions via fixed seed), then let the loop finish. dloss_inj keeps
    # the injection active (the self-forecast can pull the state back toward its
    # trajectory); dloss_noinj runs the raw loop. error_correction = dloss_noinj -
    # dloss_inj is how much the injection RECOVERS from the perturbation -- the
    # axis where feedforward found forward-prediction uniquely special. A genuine
    # forecast should correct; a random/misaligned injection should not.
    # =============================================
    pert_step = T // 2
    robustness = {}
    with torch.no_grad():
        for eps in [0.5, 1.0, 2.0]:
            di, dn = [], []
            for bi in range(20):
                eidx = eval_indices[bi % n_evals][0]
                vim = test_images[eidx].to(device)
                vlb = test_labels[eidx].to(device)
                B = vim.shape[0]
                if closed_loop:
                    _, _, inter = model(vim, vlb, return_intermediates=True,
                                        step_inject_fn=make_hook())
                else:
                    _, _, inter = model(vim, vlb, return_intermediates=True)
                s_at = (inter[f"post_step{pert_step - 1}"] if pert_step >= 1
                        else inter["post_prelude"])
                snorm = s_at.norm(dim=-1, keepdim=True)
                torch.manual_seed(seed + bi + 7000)
                noise = torch.randn(B, n_positions, n_embd, device=device)
                noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8) * (eps * snorm)
                if closed_loop:
                    _, base_i = model(vim, vlb, step_inject_fn=make_hook())
                    _, pert_i = model(vim, vlb, step_inject_fn=make_hook(),
                                      perturbation=(pert_step, noise))
                    di.append(pert_i.item() - base_i.item())
                _, base_n = model(vim, vlb)
                _, pert_n = model(vim, vlb, perturbation=(pert_step, noise))
                dn.append(pert_n.item() - base_n.item())
            entry = {"dloss_noinj": float(np.mean(dn))}
            if closed_loop:
                entry["dloss_inj"] = float(np.mean(di))
                entry["error_correction"] = float(np.mean(dn) - np.mean(di))
            robustness[str(eps)] = entry
    print(f"  robustness (dloss @ pert step {pert_step}): " + ", ".join(
        f"eps={e}: " + (f"inj={robustness[e]['dloss_inj']:.3f} "
                        f"noinj={robustness[e]['dloss_noinj']:.3f} "
                        f"corr={robustness[e]['error_correction']:+.3f}"
                        if closed_loop else f"noinj={robustness[e]['dloss_noinj']:.3f}")
        for e in ["0.5", "1.0", "2.0"]))

    # =============================================
    # Save
    # =============================================
    tag = _cfg_tag(n_loop_layers, T, n_head, n_embd, prelude_layers, coda_layers,
                   injection_form, predict_k)
    if fwd_d_head != 32 or abs(fwd_mlp_mult - 2.0) > 1e-6:
        tag += f"_fmd{fwd_d_head}m{fwd_mlp_mult}"
    if damping < 1.0:
        tag += f"_damp{damping}"
    if gate_type != "proj":
        tag += f"_{gate_type}gate"
    if baseline_type != "forward":
        tag += f"_{baseline_type}"
    tag = f"{dataset}_{tag}"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_looped_injection/{tag}/{condition}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd.state_dict(), os.path.join(save_dir, "fwd.pt"))
    if gate:
        torch.save(gate.state_dict(), os.path.join(save_dir, "gate.pt"))

    result = {
        "condition": condition,
        "closed_loop": closed_loop, "deep_sup": deep_sup,
        "injection_form": injection_form, "predict_k": predict_k,
        "dataset": dataset, "damping": damping, "baseline_type": baseline_type,
        "config": {
            "n_loop_layers": n_loop_layers, "prelude_layers": prelude_layers,
            "coda_layers": coda_layers, "n_head": n_head, "n_embd": n_embd,
            "train_steps_loop": T, "deep_sup_frac": deep_sup_frac,
            "lr": lr, "fwd_lr": fwd_lr, "n_steps": n_steps, "seed": seed,
            "fwd_d_head": fwd_d_head, "fwd_n_head": fwd_n_head,
            "fwd_n_layer": fwd_n_layer, "fwd_mlp_mult": fwd_mlp_mult,
        },
        "history": history,
        "final_val_acc": history["val_acc"][-1][1],
        "final_val_loss": history["val_loss"][-1][1],
        "final_fm_cos": history["val_fm_cos"][-1][1],
        "convergence_rel_delta": conv.tolist(),
        "acc_vs_T": acc_T,
        "per_step_decode": per_step_decode,
        "fm_selfcons_per_step": fm_selfcons.tolist(),
        "robustness": robustness,
    }
    if closed_loop:
        result["final_val_loss_no_inj"] = history["val_loss_no_inj"][-1][1]
        result["final_gate_norm"] = history["gate_norm"][-1][1]
        result["injection_benefit"] = (
            history["val_loss"][-1][1] - history["val_loss_no_inj"][-1][1])

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")

    # --- Summary ---
    print(f"\n{'=' * 60}\n  {condition} SUMMARY\n{'=' * 60}")
    print(f"  final val_acc={result['final_val_acc']:.4f} "
          f"val_loss={result['final_val_loss']:.4f} fm_cos={result['final_fm_cos']:.4f}")
    if closed_loop:
        print(f"  injection benefit={result['injection_benefit']:+.4f} "
              f"gate_norm={result['final_gate_norm']:.3f}")
    at = acc_T["with_injection"]
    print(f"  acc_vs_T: T=1 {at['1']['acc']:.3f} | T={T} {at[str(T)]['acc']:.3f} | "
          f"T={eval_t_max} {at[str(eval_t_max)]['acc']:.3f}  "
          f"(loss {at['1']['loss']:.3f}->{at[str(T)]['loss']:.3f}->"
          f"{at[str(eval_t_max)]['loss']:.3f})")
    if closed_loop:
        ni = acc_T["no_injection"]
        print(f"  acc_vs_T (NO inj): T={T} {ni[str(T)]['acc']:.3f} | "
              f"T={eval_t_max} {ni[str(eval_t_max)]['acc']:.3f}")
    print(f"  per-step decode acc: "
          f"{[round(per_step_decode[str(t)], 3) for t in range(T)]}")
    print(f"  convergence rel_delta: {[round(c, 3) for c in conv]}")
    print(f"  fm self-consistency:   {[round(c, 3) for c in fm_selfcons]}")
    return result


@app.function(volumes={DATA_DIR: volume}, timeout=600)
def aggregate(
    dataset: str = "mnist",
    injection_form: str = "update",
    predict_k: int = 3,
    fwd_d_head: int = 32, fwd_mlp_mult: float = 2.0, damping: float = 1.0,
    gate_type: str = "proj",
    n_loop_layers: int = 1, train_steps_loop: int = 8, n_head: int = 4,
    n_embd: int = 128, prelude_layers: int = 1, coda_layers: int = 1,
    eval_t_max: int = 20,
):
    import os
    tag = _cfg_tag(n_loop_layers, train_steps_loop, n_head, n_embd,
                   prelude_layers, coda_layers, injection_form, predict_k)
    if fwd_d_head != 32 or abs(fwd_mlp_mult - 2.0) > 1e-6:
        tag += f"_fmd{fwd_d_head}m{fwd_mlp_mult}"
    if damping < 1.0:
        tag += f"_damp{damping}"
    if gate_type != "proj":
        tag += f"_{gate_type}gate"
    tag = f"{dataset}_{tag}"
    base = f"{DATA_DIR}/a2a_forward/mnist_looped_injection/{tag}"
    T = train_steps_loop
    rows = {}
    for c in CONDITIONS:
        path = os.path.join(base, c, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                rows[c] = json.load(f)

    print(f"\n{'=' * 78}\n  PHASE 2  2x2  ({tag})\n{'=' * 78}")
    hdr = f"{'condition':10s} {'val_acc':>8s} {'accT1':>7s} {'accTr':>7s} {'accTmax':>8s}"
    hdr += f" {'inj_ben':>8s} {'gate':>6s} {'fm_cos':>7s}"
    print(hdr)
    for c in CONDITIONS:
        if c not in rows:
            print(f"{c:10s}  (missing)")
            continue
        r = rows[c]
        at = r["acc_vs_T"]["with_injection"]
        ib = r.get("injection_benefit", float("nan"))
        gn = r.get("final_gate_norm", float("nan"))
        print(f"{c:10s} {r['final_val_acc']:8.4f} {at['1']['acc']:7.3f} "
              f"{at[str(T)]['acc']:7.3f} {at[str(eval_t_max)]['acc']:8.3f} "
              f"{ib:8.4f} {gn:6.3f} {r['final_fm_cos']:7.4f}")

    print("\n  Overshoot check (acc drop from train-T to T_max, with injection):")
    for c in CONDITIONS:
        if c not in rows:
            continue
        at = rows[c]["acc_vs_T"]["with_injection"]
        drop = at[str(T)]["acc"] - at[str(eval_t_max)]["acc"]
        print(f"    {c:10s}: {at[str(T)]['acc']:.3f} -> {at[str(eval_t_max)]['acc']:.3f}"
              f"  (drop {drop:+.3f})")

    print("\n  Per-step class decodability (premature abstraction):")
    for c in CONDITIONS:
        if c not in rows:
            continue
        d = rows[c]["per_step_decode"]
        print(f"    {c:10s}: {[round(d[str(t)], 3) for t in range(T)]}")

    print("\n  FM self-consistency per step cos(FM(a_t), s_t+1):")
    for c in CONDITIONS:
        if c not in rows:
            continue
        print(f"    {c:10s}: {[round(v, 3) for v in rows[c]['fm_selfcons_per_step']]}")

    return rows
