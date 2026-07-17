"""Active-vision looped ViT: does adding a genuine action degree of freedom `u`
(a glimpse command / efference copy) create command-conditional dynamics that only
an arity-2 forward model f(s, u) can predict -- while an arity-1 f(s) structurally
cannot, no matter its capacity?

Motivation (ideas/self_model_needs_a_loop.md, "the missing u"). The pure
classification loop is an autonomous system s_{t+1}=f(s_t): its own next state is
trivially self-predictable (LOOPED_README: a 1% FM predicts the loop update at
cos 0.92-0.999), which is exactly why the runnable-self-simulator probe came back
negative -- there is no command to run a counterfactual over. A cerebellar forward
model instead predicts consequences of a *command*: s_{t+1}=f(s_t, u_t), with u_t
an efference copy available before the sensory consequence. This experiment gives
the loop that missing u (where to look next), holding everything else fixed, and
tests the core theoretical claim:

  ARITY, NOT RESOLUTION. Conditioning the forward model on the command u buys
  command-conditional prediction that no amount of extra capacity in a command-blind
  f(s) can recover. Absorbing "what I tend to do" over many passes sharpens f(s)'s
  *resolution*; it never grows its *arity*.

Design (all on a single FROZEN glimpse-loop main model, so every forward model sees
identical transitions -- the Run-5 scaling-sweep control style):
  - Main model: GlimpseLoopedViT. Each step reveals a 3x3 window of the 7x7 patch
    grid at a uniform-random center u_t (FIXED STOCHASTIC POLICY -- no RL, no policy
    confound); the loop integrates glimpses over T steps to classify. This makes the
    loop load-bearing (no single look sees the digit) AND introduces the command u.
  - Forward models predict the one-step consequence s_{t+1} = G(s_t + glimpse(x,u_t)):
      FM_state: input s_t only            (arity-1, command-blind; today's a2a FM)
      FM_eff:   input s_t + embed(u_t)     (arity-2, efference-copy-conditioned)
    NEITHER sees the glimpse *content* (the revealed pixels) -- FM_eff gets only the
    command (where the model is about to look), exactly like a cerebellar forward
    model predicting a consequence it has not yet observed.
  - CAPACITY CONTROL: train FM_state and FM_eff across a sweep of sizes. The claim
    predicts a small FM_eff beats a large FM_state on command-conditional prediction.
  - NO-u CONTROL: --full-view reveals everything each step (command inert) => the
    dynamics are autonomous again and FM_state should hit ceiling with ~zero
    command-conditional spread. Same code, glimpse off.

Decisive metrics (counterfactual eval on held-out images):
  - cmd_spread: relative deviation of s_{t+1} across different commands at the same
    s_t. ~0 in full-view (autonomous); >0 with glimpses (controlled). This is "how
    much the next state depends on the command".
  - on-traj cos(pred, true s_{t+1}) for FM_state vs FM_eff at matched capacity.
  - cmd_diff_cos: for two commands u_a,u_b at the same s_t, cos(FM_eff(s,u_a)-
    FM_eff(s,u_b), true Δ). FM_state's command-difference is 0 by construction
    (no u input) -- the arity gap in one number.

Run:
  modal run --detach a2a_forward/mnist_active_vision.py::active_vision --dataset mnist
  modal run --detach a2a_forward/mnist_active_vision.py::active_vision --dataset fashion_mnist
  modal run --detach a2a_forward/mnist_active_vision.py::active_vision --dataset mnist --full-view   # no-u control
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def active_vision(
    dataset: str = "mnist",           # "mnist" or "fashion_mnist"
    full_view: bool = False,          # no-u control: reveal all patches each step
    glimpse_grid: int = 3,            # G in GxG patch window revealed per look
    n_loop_layers: int = 1,
    n_head: int = 4,
    n_embd: int = 128,
    n_glimpses: int = 8,              # loop depth / number of glimpses (T)
    patch_size: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    n_steps: int = 4000,              # main-model training steps
    eval_interval: int = 250,
    n_eval_batches: int = 5,
    # forward-model training (on the frozen main model)
    fwd_lr: float = 1e-3,
    fm_steps: int = 3000,
    fm_d_head_sweep: str = "4,8,16,32,64",  # capacity sweep (both FM_state & FM_eff)
    fm_mlp_mult: float = 2.0,
    match_d_head: int = 16,           # size used for the counterfactual eval
    n_cf_batches: int = 12,           # counterfactual eval batches
    n_cf_commands: int = 6,           # counterfactual commands sampled per state
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.looped_vit import GlimpseLoopedViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = n_glimpses
    n_positions = (28 // patch_size) ** 2 + 1
    n_centers = (28 // patch_size) ** 2
    sweep = [int(x) for x in fm_d_head_sweep.split(",")]
    mode = "full_view" if full_view else "glimpse"
    print(f"ACTIVE VISION [{mode}] on {dataset}, {device}. "
          f"T={T} glimpse={glimpse_grid} d={n_embd} sweep={sweep}")

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

    # --- Reproducible index + glimpse-command samplers (fixed stochastic policy) ---
    g_idx = torch.Generator().manual_seed(seed)
    g_u = torch.Generator().manual_seed(seed + 1)
    g_eval = torch.Generator().manual_seed(seed + 2)

    def sample_u(B, gen, length=None):
        """(length, B) uniform-random window centers -- the fixed stochastic policy.
        length defaults to T (can exceed T for acc-vs-#glimpses eval)."""
        return torch.randint(n_centers, (length or T, B), generator=gen)

    # --- Main model ---
    torch.manual_seed(seed)
    model = GlimpseLoopedViT(
        img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
        glimpse_grid=glimpse_grid, full_view=full_view,
        n_loop_layers=n_loop_layers, n_head=n_head, n_embd=n_embd, n_steps=T,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def eval_acc(n_batches=n_eval_batches, n_steps_eval=None):
        model.eval()
        acc = loss = 0.0
        with torch.no_grad():
            for _ in range(n_batches):
                idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
                vim = test_images[idx].to(device)
                vlb = test_labels[idx].to(device)
                u = sample_u(vim.shape[0], g_eval, length=n_steps_eval).to(device)
                vlog, vl = model(vim, u, vlb, n_steps=n_steps_eval)
                acc += (vlog.argmax(-1) == vlb).float().mean().item()
                loss += vl.item()
        return acc / n_batches, loss / n_batches

    # =============================================
    # Train main model (glimpse loop)
    # =============================================
    print("\n--- training main model ---")
    hist = {"train_loss": [], "val_acc": [], "val_loss": []}
    for step in range(n_steps):
        model.train()
        idx = torch.randint(len(train_images), (batch_size,), generator=g_idx)
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)
        u = sample_u(images.shape[0], g_u).to(device)
        logits, loss = model(images, u, labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_interval == 0 or step == n_steps - 1:
            va, vl = eval_acc()
            hist["train_loss"].append((step, loss.item()))
            hist["val_acc"].append((step, va))
            hist["val_loss"].append((step, vl))
            print(f"  step {step:5d}: train={loss.item():.4f} val_acc={va:.4f} "
                  f"val_loss={vl:.4f}")

    # Loop-load-bearing check: accuracy vs number of glimpses.
    acc_vs_T = {}
    for Te in [1, 2, 3, 4, 6, 8, 12]:
        if Te <= 2 * T:
            a, l = eval_acc(n_batches=n_eval_batches, n_steps_eval=Te)
            acc_vs_T[str(Te)] = {"acc": a, "loss": l}
    print(f"  acc vs #glimpses: " +
          " ".join(f"T{k}={v['acc']:.3f}" for k, v in acc_vs_T.items()))

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # =============================================
    # Collect (s_t, u_t, s_{t+1}) transitions from the frozen model
    # s_t = state BEFORE step t's glimpse (post_step{t-1}, or 0 at t=0)
    # target = post_step{t}; command = u_seq[t]. One-step (k=1) controlled forward.
    # =============================================
    def batch_transitions(images, u_seq):
        with torch.no_grad():
            _, _, inter = model(images, u_seq, return_intermediates=True)
        B = images.shape[0]
        zeros = torch.zeros(B, n_positions, n_embd, device=device)
        s_list, u_list, tgt_list = [], [], []
        for t in range(T):
            s_t = inter[f"post_step{t - 1}"] if t >= 1 else zeros
            s_list.append(s_t)
            u_list.append(u_seq[t])
            tgt_list.append(inter[f"post_step{t}"])
        # stack over t -> (T*B, ...)
        return (torch.cat(s_list, 0), torch.cat(u_list, 0).to(device),
                torch.cat(tgt_list, 0))

    def reveal_mask_for(u_idx):
        """(N, n_positions) 0/1 marker of which positions command u_idx reveals --
        the SPATIAL efference copy ('I am about to look here'). Function of the
        command only (no pixel content); cls position (0) is never a glimpse target.
        A global broadcast command vector fails because the command acts spatially:
        it localizes *which* positions update."""
        m = model.reveal_mask[u_idx]                        # (N, n_patches)
        cls0 = torch.zeros(m.shape[0], 1, device=device)
        return torch.cat([cls0, m], dim=1)                  # (N, n_positions)

    def train_fm(use_u, d_head):
        """Train a forward model to predict the loop UPDATE Δ_t = s_{t+1} - s_t
        (the command-sensitive part; predicting the full next state is dominated by
        the carried s_t and hides the arity gap -- cf. LOOPED_README's degenerate
        future-self-decodability). use_u => arity-2 (efference copy): input is
        s_t + embed(u_t). Else arity-1: input s_t only."""
        torch.manual_seed(seed + 300 + d_head + (1000 if use_u else 0))
        fm = TransformerForwardModel(
            d_model=n_embd, d_head=d_head, n_head=1, n_layer=1,
            mlp_mult=fm_mlp_mult, block_size=n_positions, causal=False,
        ).to(device)
        # Spatial efference copy: a learned marker added at the positions the command
        # is about to reveal (arity-2). No pixel content -- just "where I look".
        reveal_vec = (torch.zeros(n_embd, device=device, requires_grad=True)
                      if use_u else None)
        if use_u:
            with torch.no_grad():
                reveal_vec.normal_(std=0.02)
        params = list(fm.parameters()) + ([reveal_vec] if use_u else [])
        opt_fm = torch.optim.AdamW(params, lr=fwd_lr, weight_decay=0.01)
        g_fm = torch.Generator().manual_seed(seed + 400 + d_head)
        g_fmu = torch.Generator().manual_seed(seed + 500 + d_head)

        def fm_input(s, u):
            if not use_u:
                return s
            return s + reveal_mask_for(u).unsqueeze(-1) * reveal_vec

        for fstep in range(fm_steps):
            fm.train()
            idx = torch.randint(len(train_images), (batch_size,), generator=g_fm)
            images = train_images[idx].to(device)
            u_seq = sample_u(images.shape[0], g_fmu).to(device)
            s, u, tgt = batch_transitions(images, u_seq)
            pred = fm(fm_input(s, u))
            floss = F.mse_loss(pred, tgt - s)   # predict the update Δ_t
            opt_fm.zero_grad(); floss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt_fm.step()
        # eval cos + mse on held-out
        fm.eval()
        cos = mse = 0.0
        nb = 8
        with torch.no_grad():
            for _ in range(nb):
                idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
                images = test_images[idx].to(device)
                u_seq = sample_u(images.shape[0], g_eval).to(device)
                s, u, tgt = batch_transitions(images, u_seq)
                pred = fm(fm_input(s, u))
                target = tgt - s
                cos += F.cosine_similarity(pred, target, dim=-1).mean().item()
                mse += F.mse_loss(pred, target).item()
        return fm, reveal_vec, {"cos": cos / nb, "mse": mse / nb}

    # =============================================
    # Capacity sweep: FM_state vs FM_eff at each size (arity-not-resolution)
    # =============================================
    print("\n--- forward-model capacity sweep (state vs eff) ---")
    sweep_results = {"state": {}, "eff": {}}
    matched = {}
    for d_head in sweep:
        fm_s, _, rs = train_fm(use_u=False, d_head=d_head)
        fm_e, ue_e, re = train_fm(use_u=True, d_head=d_head)
        sweep_results["state"][str(d_head)] = rs
        sweep_results["eff"][str(d_head)] = re
        print(f"  d_head={d_head:3d}: FM_state cos={rs['cos']:.4f} | "
              f"FM_eff cos={re['cos']:.4f}  (Δ={re['cos'] - rs['cos']:+.4f})")
        if d_head == match_d_head:
            matched = {"state": (fm_s, None), "eff": (fm_e, ue_e)}
    if not matched:  # match_d_head not in sweep -> train it
        fm_s, _, _ = train_fm(use_u=False, d_head=match_d_head)
        fm_e, ue_e, _ = train_fm(use_u=True, d_head=match_d_head)
        matched = {"state": (fm_s, None), "eff": (fm_e, ue_e)}

    fm_state, _ = matched["state"]
    fm_eff, reveal_vec_eff = matched["eff"]
    fm_state.eval(); fm_eff.eval()

    # =============================================
    # Counterfactual eval (the decisive test)
    #   At each on-trajectory state s_t, sample several counterfactual commands.
    #   True consequence via model.one_step(s_t, patch_emb, u'). Then:
    #     - cmd_spread: how much s_{t+1} varies with the command (controlledness)
    #     - on-traj cos(pred, true) for state vs eff on the ACTUAL command
    #     - cmd_diff_cos: does FM_eff track s_next(u_a)-s_next(u_b)? (FM_state=0)
    # =============================================
    print("\n--- counterfactual eval ---")

    def eff_pred(fm, rvec, s, u):
        return fm(s + reveal_mask_for(u).unsqueeze(-1) * rvec)

    cmd_rel_spread = []            # ||Δ(u) - mean_u Δ|| / ||mean_u Δ|| (update space)
    ontraj_cos_state, ontraj_cos_eff = [], []
    cc_cos_eff = []               # cos(eff command-conditional, true command-conditional)
    diff_cos_eff, diff_cos_state = [], []   # cos of predicted command-diff vs true
    diff_true_norm, diff_state_norm = [], []
    with torch.no_grad():
        for _ in range(n_cf_batches):
            idx = torch.randint(len(test_images), (batch_size,), generator=g_eval)
            images = test_images[idx].to(device)
            B = images.shape[0]
            patch_emb = model._patch_embeds(images)
            u_seq = sample_u(B, g_eval).to(device)
            _, _, inter = model(images, u_seq, return_intermediates=True)
            zeros = torch.zeros(B, n_positions, n_embd, device=device)
            for t in range(T):
                s_t = inter[f"post_step{t - 1}"] if t >= 1 else zeros
                # counterfactual commands (independent of the actual policy draw)
                cmds = torch.randint(n_centers, (n_cf_commands, B),
                                     generator=g_eval).to(device)
                nexts = torch.stack([model.one_step(s_t, patch_emb, cmds[m])
                                     for m in range(n_cf_commands)], 0)  # (M,B,P,E)
                # everything in UPDATE space (Δ = next - s_t): the command-driven part
                updates = nexts - s_t.unsqueeze(0)
                mean_upd = updates.mean(0)
                spread = (updates - mean_upd).norm(dim=-1).mean()
                denom = mean_upd.norm(dim=-1).mean() + 1e-8
                cmd_rel_spread.append((spread / denom).item())

                # on-trajectory prediction quality on the ACTUAL command u_seq[t]
                # (FMs predict the update; compare to the true update)
                true_update = inter[f"post_step{t}"] - s_t
                p_state = fm_state(s_t)
                p_eff = eff_pred(fm_eff, reveal_vec_eff, s_t, u_seq[t])
                ontraj_cos_state.append(
                    F.cosine_similarity(p_state, true_update, dim=-1).mean().item())
                ontraj_cos_eff.append(
                    F.cosine_similarity(p_eff, true_update, dim=-1).mean().item())

                # command-CONDITIONAL prediction (the headline arity metric):
                # of the command-driven part of the update (true_cc = Δ(u) - mean_u Δ),
                # how much does FM_eff capture? FM_state captures 0 (predicts the mean,
                # so its command-conditional prediction is identically 0) at ANY
                # capacity -- the arity ceiling that resolution cannot lift.
                eff_all = torch.stack([eff_pred(fm_eff, reveal_vec_eff, s_t, cmds[m])
                                       for m in range(n_cf_commands)], 0)  # (M,B,P,E)
                true_cc = updates - mean_upd                       # (M,B,P,E)
                eff_cc = eff_all - eff_all.mean(0, keepdim=True)
                cc_cos_eff.append(
                    F.cosine_similarity(eff_cc, true_cc, dim=-1).mean().item())

                # command-difference tracking (arity gap): u_a vs u_b at same s_t
                # (s_t cancels in the difference, so this is pure command-driven Δ)
                ua, ub = cmds[0], cmds[1]
                true_diff = nexts[0] - nexts[1]
                eff_diff = eff_all[0] - eff_all[1]
                state_diff = fm_state(s_t) - fm_state(s_t)  # identically 0
                diff_cos_eff.append(
                    F.cosine_similarity(eff_diff, true_diff, dim=-1).mean().item())
                diff_cos_state.append(
                    F.cosine_similarity(state_diff + 1e-9, true_diff, dim=-1)
                    .mean().item())
                diff_true_norm.append(true_diff.norm(dim=-1).mean().item())
                diff_state_norm.append(state_diff.norm(dim=-1).mean().item())

    cf = {
        "cmd_rel_spread": float(np.mean(cmd_rel_spread)),
        "ontraj_cos_state": float(np.mean(ontraj_cos_state)),
        "ontraj_cos_eff": float(np.mean(ontraj_cos_eff)),
        "cmd_cond_cos_eff": float(np.mean(cc_cos_eff)),
        "cmd_cond_cos_state": 0.0,   # structural: state predicts the command-mean
        "cmd_diff_cos_eff": float(np.mean(diff_cos_eff)),
        "cmd_diff_cos_state": float(np.mean(diff_cos_state)),
        "cmd_diff_true_norm": float(np.mean(diff_true_norm)),
        "cmd_diff_state_norm": float(np.mean(diff_state_norm)),
    }
    print(f"  cmd_rel_spread (controlledness) = {cf['cmd_rel_spread']:.4f}  "
          f"(~0 => autonomous, no command effect)")
    print(f"  on-traj cos (whole update): FM_state={cf['ontraj_cos_state']:.4f}  "
          f"FM_eff={cf['ontraj_cos_eff']:.4f}  "
          f"(Δ={cf['ontraj_cos_eff'] - cf['ontraj_cos_state']:+.4f})")
    print(f"  cmd-CONDITIONAL cos (headline): FM_eff={cf['cmd_cond_cos_eff']:.4f}  "
          f"FM_state=0.0000 (structural, any capacity)")
    print(f"  cmd-diff cos: FM_eff={cf['cmd_diff_cos_eff']:.4f}  "
          f"FM_state={cf['cmd_diff_cos_state']:.4f} "
          f"(state predicts 0 command-diff by construction)")

    # arity-not-resolution headline: does small FM_eff beat the LARGEST FM_state?
    big_state_cos = sweep_results["state"][str(max(sweep))]["cos"]
    small_eff_cos = sweep_results["eff"][str(min(sweep))]["cos"]
    arity_beats_capacity = small_eff_cos > big_state_cos
    print(f"\n  ARITY vs RESOLUTION: smallest FM_eff (d={min(sweep)}) cos="
          f"{small_eff_cos:.4f}  vs  largest FM_state (d={max(sweep)}) cos="
          f"{big_state_cos:.4f}  -> eff_beats_capacity={arity_beats_capacity}")

    # =============================================
    # Save
    # =============================================
    tag = f"{dataset}_{mode}_g{glimpse_grid}_T{T}_{n_head}H{n_embd}D"
    save_dir = f"{DATA_DIR}/a2a_forward/mnist_active_vision/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    result = {
        "dataset": dataset, "mode": mode, "full_view": full_view,
        "config": {
            "glimpse_grid": glimpse_grid, "T": T, "n_embd": n_embd,
            "n_head": n_head, "n_loop_layers": n_loop_layers,
            "patch_size": patch_size, "n_steps": n_steps, "fm_steps": fm_steps,
            "fm_d_head_sweep": sweep, "match_d_head": match_d_head, "seed": seed,
        },
        "history": hist,
        "final_val_acc": hist["val_acc"][-1][1],
        "acc_vs_T": acc_vs_T,
        "fm_sweep": sweep_results,
        "counterfactual": cf,
        "arity_beats_capacity": bool(arity_beats_capacity),
        "small_eff_cos": small_eff_cos, "big_state_cos": big_state_cos,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}")

    # --- summary ---
    print(f"\n{'=' * 66}\n  ACTIVE VISION [{mode}] {dataset} SUMMARY\n{'=' * 66}")
    print(f"  final val_acc = {result['final_val_acc']:.4f}")
    print(f"  loop load-bearing: acc T1={acc_vs_T.get('1', {}).get('acc'):.3f} -> "
          f"T{T}={acc_vs_T.get(str(T), {}).get('acc'):.3f}")
    print(f"  cmd_rel_spread = {cf['cmd_rel_spread']:.4f}")
    print(f"  on-traj cos state/eff = {cf['ontraj_cos_state']:.4f} / "
          f"{cf['ontraj_cos_eff']:.4f}")
    print(f"  cmd-conditional cos eff = {cf['cmd_cond_cos_eff']:.4f} (state=0)")
    print(f"  cmd-diff cos eff = {cf['cmd_diff_cos_eff']:.4f}")
    print(f"  arity_beats_capacity = {arity_beats_capacity}")
    return result
