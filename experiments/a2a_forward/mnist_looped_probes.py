"""Representational-imprint probes for the looped-injection baseline battery.

Eval-only. Loads the confound-free bounded-gate checkpoints (OL, forward-CL,
random_proj-CL, shifted-CL) on MNIST and Fashion-MNIST and asks: what
representational imprint does each injected channel leave, and is there a
*forecast-specific* imprint (forward beyond random_proj/shifted) that looks like
self-knowledge?

Logic (see ideas/self_model_needs_a_loop.md, map-vs-model):
  - random_proj = a consistent structured channel that is a function of the state
    but NOT a forecast  -> defines the *generic* imprint (the "slot").
  - shifted     = a forecast of the WRONG position -> forecast-shaped, misaligned.
  - forward     = a forecast of the model's own future -> the candidate self-model.
  - OL          = never saw any channel.
  All four share init + data order + seed, so any divergence is attributable to
  the channel, not to a different SGD trajectory (controlled-retrain design).

Probes (all linear, closed-form ridge / least-squares):
  1. Divergence imprint  D_t = s_t^{channel-off} - s_t^{OL}: norm, eff-rank, top1%.
  2. Future self-decodability R2(s_t -> s_{t+k}), channel-live and channel-ablated.
     live - ablated separates "channel handed me the future" (map) from
     "baked into the weights" (model).
  3. Channel-subspace alignment: fraction of D_t variance inside the injected
     signal's top-PC subspace U; class-decodability of the injected signal and of
     D_t; cosine of U(forward) with the future-state principal subspace.

Run:
  modal run --detach a2a_forward/mnist_looped_probes.py::probe_battery --dataset mnist
  modal run --detach a2a_forward/mnist_looped_probes.py::probe_battery --dataset fashion_mnist
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


# Tag pieces for the headline confound-free bounded-gate runs.
_BASE = "looped_1x8_4H_128D_pre0coda0_update_k3_fmd1m0.125_scalargate"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def probe_battery(
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

    # Condition -> (tag_dir, baseline_type, closed_loop)
    CONDS = {
        "OL":          (f"{dataset}_{_BASE}",                "forward",     False),
        "forward":     (f"{dataset}_{_BASE}",                "forward",     True),
        "random_proj": (f"{dataset}_{_BASE}_random_proj",    "random_proj", True),
        "shifted":     (f"{dataset}_{_BASE}_shifted",        "shifted",     True),
    }
    # OL and forward share the same tag dir (ol_last vs cl_last subfolders).
    COND_SUBDIR = {"OL": "ol_last", "forward": "cl_last",
                   "random_proj": "cl_last", "shifted": "cl_last"}

    # ---- Data (test split, fixed subset) ----
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

    def load_condition(cond):
        tag, baseline_type, closed = CONDS[cond]
        d = os.path.join(root, tag, COND_SUBDIR[cond])
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
        gate = None
        if closed:
            gate = BoundedScalarGate(n_embd, max_gate=gate_max).to(device)
            gate.load_state_dict(torch.load(os.path.join(d, "gate.pt"),
                                            map_location=device))
            gate.eval()
        # Frozen random predictor for random_proj (NOT saved -> reconstruct by seed).
        rand_pred = None
        if baseline_type == "random_proj":
            torch.manual_seed(seed + 200)
            rand_pred = build_fm()
            for p in rand_pred.parameters():
                p.requires_grad_(False)
            rand_pred.eval()
        return model, fwd, gate, rand_pred, baseline_type, closed

    @torch.no_grad()
    def run_capture(cond, inject_on):
        """Run the loop; return states s_1..s_T (post_step0..T-1), the post-prelude
        input p, and the per-step injected signal (or None). inject_on=False runs
        the channel ablated (gate zeroed)."""
        model, fwd, gate, rand_pred, baseline_type, closed = load_condition(cond)
        injects = []

        def hook(t, operand, state):
            if not (closed and inject_on):
                return None
            if baseline_type == "random_proj":
                pred = rand_pred(operand.detach())
            else:
                pred = fwd(operand.detach())
                if baseline_type == "shifted":
                    pred = torch.roll(pred, shifts=n_positions // 2, dims=1)
            signal = pred - state  # update form
            inj = gate(signal.detach())
            injects.append(inj.detach().cpu())
            return inj

        logits, _, inter = model(images, return_intermediates=True,
                                 step_inject_fn=hook if (closed and inject_on) else None)
        acc = (logits.argmax(-1) == labels).float().mean().item()
        p = inter["post_prelude"].detach().cpu()
        states = [inter[f"post_step{t}"].detach().cpu() for t in range(T)]  # s_1..s_T
        inj = torch.stack(injects, 0) if injects else None  # (T,B,P,D)
        return dict(acc=acc, p=p, states=states, inj=inj, model=model,
                    fwd=fwd, baseline_type=baseline_type, closed=closed)

    # ---------- linear-probe helpers ----------
    def ridge_r2(X, Y, lam=1.0, train_frac=0.75):
        """Multi-output ridge; R2 on held-out (averaged over output dims, variance
        weighted). X:(N,d_in) Y:(N,d_out) torch float cpu."""
        N = X.shape[0]
        ntr = int(N * train_frac)
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(0))
        tr, te_ = perm[:ntr], perm[ntr:]
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te_], Y[te_]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        ym = Ytr.mean(0)
        Ytr_c = Ytr - ym
        d = Xtr.shape[1]
        A = Xtr.T @ Xtr + lam * torch.eye(d)
        W = torch.linalg.solve(A, Xtr.T @ Ytr_c)
        pred = Xte @ W + ym
        ss_res = ((Yte - pred) ** 2).sum().item()
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum().item()
        return 1.0 - ss_res / ss_tot

    def lin_acc(X, y, lam=1.0, train_frac=0.75):
        """Linear least-squares classifier (one-hot regression), held-out accuracy."""
        N = X.shape[0]
        ntr = int(N * train_frac)
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(1))
        tr, te_ = perm[:ntr], perm[ntr:]
        Y = torch.zeros(N, 10)
        Y[torch.arange(N), y] = 1.0
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        d = Xn.shape[1]
        A = Xn[tr].T @ Xn[tr] + lam * torch.eye(d)
        W = torch.linalg.solve(A, Xn[tr].T @ Y[tr])
        pred = (Xn[te_] @ W).argmax(1)
        return (pred == y[te_]).float().mean().item()

    def eff_rank_top1(M):
        """M: (N,d). Return (effective participation-ratio rank, top-1 PC var frac)."""
        Mc = M - M.mean(0)
        cov = (Mc.T @ Mc) / M.shape[0]
        ev = torch.linalg.eigvalsh(cov).clamp(min=0)
        ev = ev / (ev.sum() + 1e-12)
        er = (ev.sum() ** 2 / (ev ** 2).sum()).item()  # participation ratio
        return er, ev.max().item()

    # Patch tokens only (drop CLS at pos 0) for state-geometry; CLS for class.
    def patches(x):  # (B,P,D) -> (B*(P-1), D)
        return x[:, 1:, :].reshape(-1, x.shape[-1])

    def cls(x):  # (B,P,D) -> (B,D)
        return x[:, 0, :]

    # ---------- run all conditions, both modes ----------
    caps = {}
    for cond in ["OL", "forward", "random_proj", "shifted"]:
        caps[cond] = {"live": run_capture(cond, inject_on=True)}
        if cond != "OL":
            caps[cond]["ablated"] = run_capture(cond, inject_on=False)
    print("accuracies:",
          {c: round(caps[c]["live"]["acc"], 4) for c in caps})

    results = {"dataset": dataset, "predict_k": predict_k,
               "acc": {c: caps[c]["live"]["acc"] for c in caps}, "probes": {}}

    # ===== Probe 2: future self-decodability R2(s_t -> s_{t+k}) =====
    def future_r2(states, roll_target=False):
        """R2 decoding s_{t+k} from s_t over patch tokens. roll_target rolls the
        target's patch positions by n_pos//2 (matching the 'shifted' channel's
        roll), breaking position correspondence -> tests position-specificity."""
        Xs, Ys = [], []
        for ft in range(0, T - predict_k):  # post_step{ft} -> post_step{ft+k}
            Xp = states[ft][:, 1:, :]                 # patch tokens (B,P-1,D)
            Yp = states[ft + predict_k][:, 1:, :]
            if roll_target:
                Yp = torch.roll(Yp, shifts=(n_positions - 1) // 2, dims=1)
            Xs.append(Xp.reshape(-1, Xp.shape[-1]))
            Ys.append(Yp.reshape(-1, Yp.shape[-1]))
        X = torch.cat(Xs, 0)
        Y = torch.cat(Ys, 0)
        # subsample for speed
        if X.shape[0] > 60000:
            idx = torch.randperm(X.shape[0],
                                 generator=torch.Generator().manual_seed(2))[:60000]
            X, Y = X[idx], Y[idx]
        return ridge_r2(X, Y)

    fut = {}
    fut["OL"] = {"live": future_r2(caps["OL"]["live"]["states"]),
                 "live_rolled": future_r2(caps["OL"]["live"]["states"], roll_target=True)}
    for cond in ["forward", "random_proj", "shifted"]:
        st = caps[cond]["live"]["states"]
        fut[cond] = {"live": future_r2(st),
                     "live_rolled": future_r2(st, roll_target=True),
                     "ablated": future_r2(caps[cond]["ablated"]["states"])}
    results["probes"]["future_decodability_r2"] = fut

    # ===== Probe 1: divergence imprint D_t = s^{ablated} - s^{OL} =====
    ol_states = caps["OL"]["live"]["states"]
    div = {}
    for cond in ["forward", "random_proj", "shifted"]:
        ab = caps[cond]["ablated"]["states"]
        Ds = [patches(ab[t] - ol_states[t]) for t in range(T)]
        D = torch.cat(Ds, 0)
        base = torch.cat([patches(ol_states[t]) for t in range(T)], 0)
        er, t1 = eff_rank_top1(D)
        div[cond] = {
            "rel_norm": (D.norm(dim=-1).mean() / base.norm(dim=-1).mean()).item(),
            "eff_rank": er, "top1_pc_frac": t1,
            "class_decode_acc": lin_acc(torch.cat([cls(ab[t] - ol_states[t])
                                                   for t in range(T)], 0),
                                        labels.cpu().repeat(T)),
        }
    results["probes"]["divergence_imprint"] = div

    # ===== Probe 3: channel-subspace alignment =====
    # U = top-10 PCs of the injected signal (patch tokens, all steps).
    def top_subspace(M, k=10):
        Mc = M - M.mean(0)
        _, _, Vt = torch.linalg.svd(Mc, full_matrices=False)
        return Vt[:k]  # (k,d)

    def var_in_subspace(M, U):
        Mc = M - M.mean(0)
        proj = Mc @ U.T  # (N,k)
        return (proj.pow(2).sum() / Mc.pow(2).sum()).item()

    # Future-state principal subspace (what "the future" looks like), from OL.
    fut_sub = top_subspace(torch.cat([patches(ol_states[t]) for t in range(T)], 0), 10)

    chan = {}
    for cond in ["forward", "random_proj", "shifted"]:
        inj = caps[cond]["live"]["inj"]  # (T,B,P,D)
        inj_p = inj[:, :, 1:, :].reshape(-1, inj.shape[-1])  # patch tokens
        U = top_subspace(inj_p, 10)
        # imprint (channel-off divergence) alignment with channel subspace U
        ab = caps[cond]["ablated"]["states"]
        D = torch.cat([patches(ab[t] - ol_states[t]) for t in range(T)], 0)
        # class content of the injected signal (CLS token of inject)
        inj_cls = inj[:, :, 0, :].reshape(-1, inj.shape[-1])
        chan[cond] = {
            "imprint_var_in_channel_subspace": var_in_subspace(D, U),
            "channel_subspace_cos_with_future": float(
                (U @ fut_sub.T).pow(2).sum().sqrt().item() / (10 ** 0.5)),
            "inj_class_decode_acc": lin_acc(inj_cls, labels.cpu().repeat(T)),
            "inj_rel_norm": (inj_p.norm(dim=-1).mean()
                             / torch.cat([patches(ol_states[t])
                                          for t in range(T)], 0).norm(dim=-1).mean()
                             ).item(),
        }
    results["probes"]["channel_subspace"] = chan

    # ===== Control 1: matched-subspace null for the imprint-in-U result =====
    # Is the imprint concentrated in the channel's SPECIFIC directions, or merely
    # in the task manifold (which a forecast channel happens to span)? Compare
    # var_in_U(D) against random 10-dim subspaces drawn from within the top-K
    # future-state PCs (the task manifold). ratio > 1 => channel-specific.
    ol_patch_all = torch.cat([patches(ol_states[t]) for t in range(T)], 0)
    K_task = 40
    F_basis = top_subspace(ol_patch_all, K_task)  # (K,128) task manifold basis

    def rand_taskaligned_subspace(k, gseed):
        g = torch.Generator().manual_seed(gseed)
        R = torch.randn(K_task, k, generator=g)
        Q, _ = torch.linalg.qr(R)          # (K,k) orthonormal cols
        return (Q.T @ F_basis)             # (k,128) rows within span(F)

    matched = {}
    for cond in ["forward", "random_proj", "shifted"]:
        ab = caps[cond]["ablated"]["states"]
        D = torch.cat([patches(ab[t] - ol_states[t]) for t in range(T)], 0)
        inj = caps[cond]["live"]["inj"]
        inj_p = inj[:, :, 1:, :].reshape(-1, inj.shape[-1])
        U = top_subspace(inj_p, 10)
        v_U = var_in_subspace(D, U)
        v_F = var_in_subspace(D, F_basis)                       # in whole task manifold
        v_rand = np.mean([var_in_subspace(D, rand_taskaligned_subspace(10, s))
                          for s in range(8)])                   # random 10-dim of F
        matched[cond] = {
            "var_in_channel_U": v_U,
            "var_in_task_manifold_K40": v_F,
            "var_in_random_taskaligned_10d": float(v_rand),
            "specificity_ratio_U_over_rand": float(v_U / (v_rand + 1e-9)),
        }
    results["probes"]["matched_subspace_control"] = matched

    # ===== Control 3: injection absorption / cancellation (efference-copy) =====
    # At the operating point, how much does each step's injection actually move the
    # next state per unit injected norm (gain), and in what direction (cos)? A loop
    # that has learned to expect/cancel an aligned forecast should absorb it (lower
    # gain, lower cos) more than a misaligned one.
    cancel = {}
    for cond in ["forward", "random_proj", "shifted"]:
        cp = caps[cond]["live"]
        model = cp["model"]
        G = model._apply_operator
        p_dev = cp["p"].to(device)
        gains, coss = [], []
        with torch.no_grad():
            for t in range(T):
                s_t = (cp["states"][t - 1] if t >= 1
                       else torch.zeros_like(cp["states"][0])).to(device)
                inj_t = cp["inj"][t].to(device)
                base = G(s_t + p_dev)
                eff = G(s_t + p_dev + inj_t) - base
                gains.append((eff.norm(dim=-1)
                              / (inj_t.norm(dim=-1) + 1e-8)).mean().item())
                coss.append(F.cosine_similarity(
                    eff.reshape(-1, n_embd), inj_t.reshape(-1, n_embd),
                    dim=-1).mean().item())
        cancel[cond] = {"mean_gain": float(np.mean(gains)),
                        "mean_cos_effect_inj": float(np.mean(coss))}
    results["probes"]["injection_absorption"] = cancel

    # ---- save ----
    out_dir = f"{root}/probes"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_imprint_probes.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---- print summary ----
    print("\n" + "=" * 70)
    print(f"  IMPRINT PROBES: {dataset}")
    print("=" * 70)
    print("\n[Probe 2] Future self-decodability R2(s_t -> s_{t+k}):")
    print(f"  {'cond':12s} {'live':>8s} {'ablated':>8s} {'live-abl':>9s}")
    for c in ["OL", "forward", "random_proj", "shifted"]:
        L = fut[c]["live"]; A = fut[c].get("ablated")
        d = f"{L-A:+.3f}" if A is not None else "   --"
        print(f"  {c:12s} {L:8.3f} {(A if A is not None else float('nan')):8.3f} {d:>9s}")
    print("\n[Control 2] Position-specificity: aligned vs rolled-future R2:")
    print(f"  {'cond':12s} {'aligned':>8s} {'rolled':>8s} {'align-adv':>10s}")
    for c in ["OL", "forward", "random_proj", "shifted"]:
        A = fut[c]["live"]; R = fut[c]["live_rolled"]
        print(f"  {c:12s} {A:8.3f} {R:8.3f} {A-R:10.3f}")
    print("\n[Probe 1] Divergence imprint (channel-off vs OL):")
    print(f"  {'cond':12s} {'relnorm':>8s} {'effrank':>8s} {'top1%':>7s} {'classdec':>9s}")
    for c in ["forward", "random_proj", "shifted"]:
        v = div[c]
        print(f"  {c:12s} {v['rel_norm']:8.3f} {v['eff_rank']:8.1f} "
              f"{100*v['top1_pc_frac']:7.2f} {v['class_decode_acc']:9.3f}")
    print("\n[Probe 3] Channel-subspace alignment:")
    print(f"  {'cond':12s} {'imprint_in_U':>13s} {'U_cos_future':>13s} "
          f"{'inj_classdec':>13s} {'inj_relnorm':>12s}")
    for c in ["forward", "random_proj", "shifted"]:
        v = chan[c]
        print(f"  {c:12s} {v['imprint_var_in_channel_subspace']:13.3f} "
              f"{v['channel_subspace_cos_with_future']:13.3f} "
              f"{v['inj_class_decode_acc']:13.3f} {v['inj_rel_norm']:12.4f}")
    print("\n[Control 1] Matched-subspace null (imprint concentration):")
    print(f"  {'cond':12s} {'in_U':>7s} {'in_task40':>10s} {'in_rand10':>10s} "
          f"{'U/rand':>8s}")
    for c in ["forward", "random_proj", "shifted"]:
        v = matched[c]
        print(f"  {c:12s} {v['var_in_channel_U']:7.3f} "
              f"{v['var_in_task_manifold_K40']:10.3f} "
              f"{v['var_in_random_taskaligned_10d']:10.3f} "
              f"{v['specificity_ratio_U_over_rand']:8.2f}")
    print("\n[Control 3] Injection absorption (efference-copy signature):")
    print(f"  {'cond':12s} {'gain':>8s} {'cos(eff,inj)':>13s}")
    for c in ["forward", "random_proj", "shifted"]:
        v = cancel[c]
        print(f"  {c:12s} {v['mean_gain']:8.3f} {v['mean_cos_effect_inj']:13.3f}")
    print(f"\nSaved to {out_dir}/{dataset}_imprint_probes.json")
    return results
