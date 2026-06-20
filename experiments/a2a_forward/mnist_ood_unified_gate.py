"""MNIST OOD unified gate experiment: digit shift with NTP-trained gate.

Re-runs the OOD gate experiment (mnist_ood_gate.py) with a unified gate
(NTP-trained, no FOMAML bilevel) and uniform local loss, testing whether
the unified gate shows different selectivity patterns under distribution
shift than the FOMAML gate.

The original OOD gate experiment used a FOMAML LearningGate that opened
for novel digits (+0.13-0.17 diff). The unified gate closes on stationary
MNIST (0.37 -> 0.13 over 4 cycles) because injection becomes redundant
as the model improves. Under a digit shift, the prediction:
  - Known digits (0-6): model already good -> injection redundant -> gate closes
  - Novel digits (7-9): model bad -> FM prediction helps -> gate stays open
This would produce a known-novel contrast driven by injection utility
(not FM error vocabulary as with the FOMAML gate).

Design:
  Phase 1 (cycles 1-2): Train on digits 0-6 only
  Phase 2 (cycles 3-4): Train on digits 0-9

Three conditions, same total main-model gradient steps:
  1. WS_UG_shift: Unified gate + uniform local loss, digit shift
  2. WS_UG_full: Unified gate + uniform local loss, all digits (stationary)
  3. OL_shift: Open-loop with same digit schedule (mechanism control)
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=28800,
    memory=32768,
)
def a2a_mnist_ood_unified_gate(
    # Architecture
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    patch_size: int = 4,
    batch_size: int = 128,
    # FM architecture
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 1,
    fwd_d_head: int = 32,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    # Wake-sleep protocol
    n_cycles: int = 4,
    phase1_cycles: int = 2,
    n_known_digits: int = 7,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    # Training params
    lr: float = 3e-4,
    fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4,
    distill_alpha: float = 0.5,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    # Eval params
    eval_interval: int = 200,
    n_eval_batches: int = 5,
    probe_batches: int = 40,
    probe_steps: int = 300,
    seed: int = 42,
):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from datasets import load_dataset
    from a2a_forward.vit import ViT
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(predict_from.replace("post_block", ""))
    n_positions = (28 // patch_size) ** 2 + 1

    steps_per_cycle = wake_steps + distill_steps
    total_main_steps = n_cycles * steps_per_cycle
    checkpoint_steps = [(i + 1) * steps_per_cycle for i in range(n_cycles)]

    known_digits = list(range(n_known_digits))
    novel_digits = list(range(n_known_digits, 10))

    print(f"MNIST OOD UNIFIED GATE EXPERIMENT on {device}")
    print(f"  ViT: {n_layer}L {n_head}H {n_embd}D, patch={patch_size}")
    print(f"  FM: {fwd_n_layer}L {fwd_n_head}H {fwd_d_head}D, bidirectional")
    print(f"  {predict_from} -> {predict_to}, inject after block {inject_after_block}")
    print(f"  {n_cycles} cycles: {wake_steps} wake + {distill_steps} sleep each")
    print(f"  Phase 1 (cycles 1-{phase1_cycles}): digits {known_digits}")
    print(f"  Phase 2 (cycles {phase1_cycles+1}-{n_cycles}): digits 0-9")
    print(f"  Total main-model steps: {total_main_steps}")
    print(f"  Unified gate + UNIFORM local loss")

    # -----------------------------------------------------------------
    # Unified Gate
    # -----------------------------------------------------------------
    class UnifiedGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
            )
            self.projection = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            n_params = sum(p.numel() for p in self.parameters())
            print(f"UnifiedGate: {n_params/1e3:.1f}K params "
                  f"(input={2*d_model}, hidden={d_hidden}, output={d_model})")

        def forward(self, activations, fwd_pred):
            gate_w = torch.sigmoid(self.gate_net(
                torch.cat([activations, fwd_pred], dim=-1)))
            injection = gate_w * self.projection(fwd_pred)
            return injection, gate_w

        def injection_norm(self):
            return self.projection.weight.norm().item()

    # -----------------------------------------------------------------
    # Load MNIST
    # -----------------------------------------------------------------
    print("\nLoading MNIST...")
    ds = load_dataset("ylecun/mnist")
    train_imgs = np.stack([np.array(img) for img in ds["train"]["image"]])
    train_images = torch.from_numpy(train_imgs).float().unsqueeze(1) / 255.0
    train_labels = torch.tensor(ds["train"]["label"])
    test_imgs = np.stack([np.array(img) for img in ds["test"]["image"]])
    test_images = torch.from_numpy(test_imgs).float().unsqueeze(1) / 255.0
    test_labels = torch.tensor(ds["test"]["label"])
    print(f"  Train: {len(train_images)}, Test: {len(test_images)}")

    known_mask = torch.tensor([l.item() < n_known_digits for l in train_labels])
    known_train_indices = torch.where(known_mask)[0]
    print(f"  Known-digit train set: {len(known_train_indices)} "
          f"(digits {known_digits})")
    print(f"  Full train set: {len(train_images)} (digits 0-9)")

    # -----------------------------------------------------------------
    # Pre-generate batch indices
    # -----------------------------------------------------------------
    shift_gen = torch.Generator().manual_seed(seed)
    phase1_steps = phase1_cycles * steps_per_cycle
    phase2_steps = (n_cycles - phase1_cycles) * steps_per_cycle

    shift_batch_indices = []
    for _ in range(phase1_steps):
        idx_into_known = torch.randint(
            len(known_train_indices), (batch_size,), generator=shift_gen)
        shift_batch_indices.append(known_train_indices[idx_into_known])
    for _ in range(phase2_steps):
        shift_batch_indices.append(
            torch.randint(len(train_images), (batch_size,), generator=shift_gen))

    full_gen = torch.Generator().manual_seed(seed)
    full_batch_indices = [
        torch.randint(len(train_images), (batch_size,), generator=full_gen)
        for _ in range(total_main_steps)
    ]

    eval_gen = torch.Generator().manual_seed(seed + 1)
    probe_gen = torch.Generator().manual_seed(seed + 2)
    n_evals = total_main_steps // eval_interval + 10
    eval_indices = [
        [torch.randint(len(test_images), (batch_size,), generator=eval_gen)
         for _ in range(n_eval_batches)]
        for _ in range(n_evals)
    ]
    probe_indices = [
        torch.randint(len(test_images), (batch_size,), generator=probe_gen)
        for _ in range(probe_batches)
    ]

    repoint_gen_shift = torch.Generator().manual_seed(seed + 3)
    repoint_gen_full = torch.Generator().manual_seed(seed + 3)

    shift_repoint_indices = []
    for cycle in range(n_cycles):
        if cycle < phase1_cycles:
            for _ in range(retrain_steps):
                idx = torch.randint(
                    len(known_train_indices), (batch_size,),
                    generator=repoint_gen_shift)
                shift_repoint_indices.append(known_train_indices[idx])
        else:
            for _ in range(retrain_steps):
                shift_repoint_indices.append(
                    torch.randint(len(train_images), (batch_size,),
                                  generator=repoint_gen_shift))

    full_repoint_indices = [
        torch.randint(len(train_images), (batch_size,),
                       generator=repoint_gen_full)
        for _ in range(n_cycles * retrain_steps)
    ]

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------
    def make_vit():
        return ViT(
            img_size=28, patch_size=patch_size, in_channels=1, n_classes=10,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        ).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=n_positions, causal=False,
        ).to(device)

    # -----------------------------------------------------------------
    # Initialize shared weights
    # -----------------------------------------------------------------
    torch.manual_seed(seed)
    init_model = make_vit()
    init_fm = make_fm()
    init_model_state = {k: v.cpu().clone()
                        for k, v in init_model.state_dict().items()}
    init_fm_state = {k: v.cpu().clone()
                     for k, v in init_fm.state_dict().items()}
    del init_model, init_fm
    torch.cuda.empty_cache()
    print("Saved initial weight states")

    # =================================================================
    # SHARED HELPERS
    # =================================================================
    def eval_model(model, eval_idx=0):
        model.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)
                logits, loss = model(images, labels)
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_eval_batches, total_acc / n_eval_batches

    def eval_by_digit_group(model):
        model.eval()
        correct_per_digit = torch.zeros(10)
        total_per_digit = torch.zeros(10)
        loss_per_digit = torch.zeros(10)

        with torch.no_grad():
            for start in range(0, len(test_images), batch_size):
                end = min(start + batch_size, len(test_images))
                imgs = test_images[start:end].to(device)
                lbls = test_labels[start:end].to(device)
                logits, _ = model(imgs, lbls)
                preds = logits.argmax(-1)
                per_token_loss = F.cross_entropy(
                    logits, lbls, reduction='none')

                for d in range(10):
                    mask = lbls == d
                    if mask.sum() > 0:
                        correct_per_digit[d] += (preds[mask] == d).sum().item()
                        total_per_digit[d] += mask.sum().item()
                        loss_per_digit[d] += per_token_loss[mask].sum().item()

        per_digit = {}
        for d in range(10):
            if total_per_digit[d] > 0:
                per_digit[d] = {
                    "acc": float(correct_per_digit[d] / total_per_digit[d]),
                    "loss": float(loss_per_digit[d] / total_per_digit[d]),
                    "count": int(total_per_digit[d]),
                }

        known_c = correct_per_digit[:n_known_digits].sum().item()
        known_t = total_per_digit[:n_known_digits].sum().item()
        novel_c = correct_per_digit[n_known_digits:].sum().item()
        novel_t = total_per_digit[n_known_digits:].sum().item()

        return {
            "overall_acc": float(
                correct_per_digit.sum() / total_per_digit.sum()),
            "overall_loss": float(
                loss_per_digit.sum() / total_per_digit.sum()),
            "known_acc": float(known_c / known_t) if known_t > 0 else 0,
            "novel_acc": float(novel_c / novel_t) if novel_t > 0 else 0,
            "known_loss": float(
                loss_per_digit[:n_known_digits].sum() / known_t
            ) if known_t > 0 else 0,
            "novel_loss": float(
                loss_per_digit[n_known_digits:].sum() / novel_t
            ) if novel_t > 0 else 0,
            "per_digit": per_digit,
        }

    def eval_model_with_ug(model, fm, ugate, eval_idx=0):
        model.eval(); fm.eval(); ugate.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[eval_idx][bi]
                images = test_images[eidx].to(device)
                labels = test_labels[eidx].to(device)

                def cb(act):
                    fp = fm(act)
                    inj, _ = ugate(act, fp)
                    return inj

                logits, loss, _ = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cb,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                total_loss += loss.item()
                total_acc += (logits.argmax(-1) == labels).float().mean().item()
        return total_loss / n_eval_batches, total_acc / n_eval_batches

    def measure_robustness(model, label, n_batches=20):
        model.eval()
        eps_list = [0.5, 1.0, 2.0]
        results = {}
        with torch.no_grad():
            for eps in eps_list:
                deltas = []
                for bi in range(min(probe_batches, n_batches)):
                    pidx = probe_indices[bi]
                    vim = test_images[pidx].to(device)
                    vlb = test_labels[pidx].to(device)
                    _, base_loss = model(vim, vlb)
                    torch.manual_seed(seed + bi + 5000)
                    noise = torch.randn(
                        batch_size, n_positions, n_embd, device=device) * eps
                    _, pert_loss = model(
                        vim, vlb,
                        perturbation=(inject_after_block, noise),
                    )
                    deltas.append(pert_loss.item() - base_loss.item())
                results[str(eps)] = float(np.mean(deltas))
        print(f"  [{label:>20s}] "
              + " | ".join(f"eps={e}: d={results[str(e)]:+.4f}"
                           for e in eps_list))
        return results

    layer_keys = [f"post_block{i}" for i in range(n_layer)]

    def self_knowledge_probes(model, fm, label):
        model.eval(); fm.eval()
        acts_by_layer = {k: [] for k in layer_keys}
        residuals = []
        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                residuals.append((tgt - pred).reshape(-1, n_embd).cpu())
                for k in layer_keys:
                    acts_by_layer[k].append(vi[k].reshape(-1, n_embd).cpu())

        all_res = torch.cat(residuals)
        for k in layer_keys:
            acts_by_layer[k] = torch.cat(acts_by_layer[k])

        n_total = all_res.shape[0]
        n_train = int(0.8 * n_total)
        perm = torch.randperm(
            n_total, generator=torch.Generator().manual_seed(seed))
        tr, te = perm[:n_train], perm[n_train:]

        r2_by_layer = {}
        for lk in layer_keys:
            X = acts_by_layer[lk]
            probe = nn.Linear(n_embd, n_embd).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
            X_tr, Y_tr = X[tr].to(device), all_res[tr].to(device)
            X_te, Y_te = X[te].to(device), all_res[te].to(device)

            bs = min(4096, n_train)
            for _ in range(probe_steps):
                si = torch.randint(n_train, (bs,))
                ploss = F.mse_loss(probe(X_tr[si]), Y_tr[si])
                opt.zero_grad(); ploss.backward(); opt.step()

            with torch.no_grad():
                mse = F.mse_loss(probe(X_te), Y_te).item()
                var = Y_te.var().item()
                r2 = 1.0 - mse / var if var > 0 else 0.0
            r2_by_layer[lk] = r2
            del probe, X_tr, Y_tr, X_te, Y_te
            torch.cuda.empty_cache()

        print(f"  [{label}] SK: "
              + " | ".join(f"{k}={v:.3f}" for k, v in r2_by_layer.items()))
        return r2_by_layer

    def compute_residual_stats(model, fm, label):
        model.eval(); fm.eval()
        all_res = []
        per_image_res = []
        per_image_digits = []

        with torch.no_grad():
            for pidx in probe_indices:
                images = test_images[pidx].to(device)
                digits = test_labels[pidx]
                _, _, vi = model(images, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                res = tgt - pred
                all_res.append(res.reshape(-1, n_embd).cpu())
                per_image_res.append(res.mean(dim=1).cpu())
                per_image_digits.append(digits)

        all_res = torch.cat(all_res)
        per_image_res = torch.cat(per_image_res)
        per_image_digits = torch.cat(per_image_digits)

        R = all_res.numpy().astype(np.float64)
        R_c = R - R.mean(axis=0, keepdims=True)
        cov = (R_c.T @ R_c) / (R_c.shape[0] - 1)
        evals = np.linalg.eigvalsh(cov)[::-1]
        evals = np.maximum(evals, 0)
        total = evals.sum()
        fracs = evals / total if total > 0 else evals
        fracs_pos = fracs[fracs > 1e-12]
        eff_rank = float(np.exp(-np.sum(fracs_pos * np.log(fracs_pos))))

        res_np = per_image_res.numpy().astype(np.float64)
        res_c = res_np - res_np.mean(axis=0, keepdims=True)
        img_cov = (res_c.T @ res_c) / (res_c.shape[0] - 1)
        img_evals, img_evecs = np.linalg.eigh(img_cov)
        idx = img_evals.argsort()[::-1]
        vecs = img_evecs[:, idx[:5]]
        projections = res_c @ vecs
        digits_np = per_image_digits.numpy()

        eta_sqs = []
        for pc in range(5):
            proj = projections[:, pc]
            grand_mean = proj.mean()
            ss_total = ((proj - grand_mean) ** 2).sum()
            ss_between = 0.0
            for d in range(10):
                mask = digits_np == d
                if mask.sum() > 0:
                    dm = proj[mask].mean()
                    ss_between += mask.sum() * (dm - grand_mean) ** 2
            eta_sqs.append(
                float(ss_between / ss_total) if ss_total > 0 else 0.)

        with torch.no_grad():
            cos_total = 0.0
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                vim = test_images[eidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                tgt = vi[predict_to]
                pred = fm(src)
                cos_total += F.cosine_similarity(
                    pred, tgt, dim=-1).mean().item()

        stats = {
            "eff_rank": eff_rank,
            "top1_pc_frac": float(fracs[0]),
            "mean_res_norm": float(all_res.norm(dim=-1).mean()),
            "eta_squared_top5": eta_sqs,
            "mean_eta_squared": float(np.mean(eta_sqs)),
            "fwd_cosine": cos_total / n_eval_batches,
        }
        print(f"  [{label}] eff_rank={eff_rank:.1f}/{n_embd}, "
              f"mean_eta2={float(np.mean(eta_sqs)):.3f}, "
              f"res_norm={stats['mean_res_norm']:.3f}, "
              f"fwd_cos={stats['fwd_cosine']:.4f}")
        return stats

    def train_fresh_fm(model, steps, fm_seed, label, repoint_idxs):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        torch.manual_seed(fm_seed)
        fm = make_fm()
        opt = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)

        for step in range(steps):
            fm.train()
            ridx = repoint_idxs[step % len(repoint_idxs)]
            images = train_images[ridx].to(device)
            with torch.no_grad():
                _, _, vi = model(images, return_intermediates=True)
                source = vi[predict_from]
                target = vi[predict_to]
            pred = fm(source)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt.step()

        fm.eval()
        cos_total = 0.0
        with torch.no_grad():
            for bi in range(n_eval_batches):
                eidx = eval_indices[0][bi]
                vim = test_images[eidx].to(device)
                _, _, vi = model(vim, return_intermediates=True)
                p = fm(vi[predict_from])
                cos_total += F.cosine_similarity(
                    p, vi[predict_to], dim=-1).mean().item()
        cos = cos_total / n_eval_batches
        print(f"  [{label}] Fresh FM cosine: {cos:.4f}")

        for p in model.parameters():
            p.requires_grad = True
        return fm, cos

    def get_ug_gate_stats_by_digit(ugate, model, fwd_model, label):
        """Gate weight statistics conditioned on digit identity."""
        ugate.eval(); model.eval(); fwd_model.eval()
        all_gw_cls = []
        all_digits = []
        with torch.no_grad():
            for bi in range(min(probe_batches, 20)):
                pidx = probe_indices[bi]
                vim = test_images[pidx].to(device)
                dgts = test_labels[pidx]
                _, _, vi = model(vim, return_intermediates=True)
                src = vi[predict_from]
                pred = fwd_model(src)
                _, gw = ugate(src, pred)
                all_gw_cls.append(gw[:, 0].cpu())
                all_digits.append(dgts)

        gw_cat = torch.cat(all_gw_cls, dim=0)
        digits_cat = torch.cat(all_digits, dim=0)
        dim_mean = gw_cat.mean(dim=0)

        per_digit_mean = {}
        for d in range(10):
            mask = digits_cat == d
            if mask.sum() > 0:
                per_digit_mean[d] = float(gw_cat[mask].mean())

        known_mask = digits_cat < n_known_digits
        novel_mask = digits_cat >= n_known_digits
        known_mean = float(gw_cat[known_mask].mean()) if known_mask.sum() > 0 else 0
        novel_mean = float(gw_cat[novel_mask].mean()) if novel_mask.sum() > 0 else 0

        known_dim_mean = gw_cat[known_mask].mean(dim=0) if known_mask.sum() > 0 \
            else torch.zeros(n_embd)
        novel_dim_mean = gw_cat[novel_mask].mean(dim=0) if novel_mask.sum() > 0 \
            else torch.zeros(n_embd)
        dim_diff = (novel_dim_mean - known_dim_mean)

        stats = {
            "overall_mean": float(gw_cat.mean()),
            "overall_std": float(gw_cat.std()),
            "known_mean": known_mean,
            "novel_mean": novel_mean,
            "known_novel_diff": novel_mean - known_mean,
            "per_digit_mean": per_digit_mean,
            "dim_mean": dim_mean.tolist(),
            "known_dim_mean": known_dim_mean.tolist(),
            "novel_dim_mean": novel_dim_mean.tolist(),
            "dim_diff_mean": float(dim_diff.mean()),
            "dim_diff_std": float(dim_diff.std()),
            "n_dims_novel_higher": int((dim_diff > 0).sum()),
            "sparsity_below_0.1": float((dim_mean < 0.1).float().mean()),
            "sparsity_below_0.2": float((dim_mean < 0.2).float().mean()),
        }

        print(f"  [{label}] Gate: overall={stats['overall_mean']:.3f}, "
              f"known={known_mean:.3f}, novel={novel_mean:.3f}, "
              f"diff={stats['known_novel_diff']:+.3f}, "
              f"sparse(<0.1)={stats['sparsity_below_0.1']:.1%}, "
              f"dims_novel_higher={stats['n_dims_novel_higher']}/{n_embd}")
        return stats

    # =================================================================
    # WS_UG_uniform TRAINING LOOP (parameterized by batch/repoint indices)
    # =================================================================
    def run_ws_ug(condition_label, batch_indices, repoint_indices_list):
        print(f"\n{'='*60}")
        print(f"  {condition_label} ({n_cycles} cycles)")
        print(f"{'='*60}")

        model = make_vit()
        model.load_state_dict(init_model_state)
        fm = make_fm()
        fm.load_state_dict(init_fm_state)
        ugate = UnifiedGate(n_embd, lg_hidden).to(device)

        checkpoints = {}
        history = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "gate_stats": [],
        }
        global_step = 0
        repoint_offset = 0

        for cycle in range(n_cycles):
            phase = "phase1" if cycle < phase1_cycles else "phase2"
            print(f"\n  --- {condition_label} Cycle {cycle+1}/{n_cycles} "
                  f"({phase}) ---")

            # === WAKE: UG co-training with uniform local loss ===
            print(f"  [WAKE] UG co-training, uniform local loss "
                  f"({wake_steps} steps)")
            main_params = (list(model.parameters())
                           + list(ugate.parameters()))
            opt_main = torch.optim.AdamW(
                main_params, lr=lr, weight_decay=0.01)
            opt_fwd = torch.optim.AdamW(
                fm.parameters(), lr=fwd_lr, weight_decay=0.01)

            for step in range(wake_steps):
                model.train(); fm.train(); ugate.train()

                idx = batch_indices[global_step]
                images = train_images[idx].to(device)
                labels = train_labels[idx].to(device)

                fwd_pred_cache = {}

                def cerebellar_fn(act, _cache=fwd_pred_cache):
                    fp = fm(act.detach())
                    _cache["pred"] = fp
                    inj, gw = ugate(act.detach(), fp.detach())
                    _cache["gate_w"] = gw
                    return inj

                logits, cls_loss, intermediates = model(
                    images, labels, return_intermediates=True,
                    cerebellar_fn=cerebellar_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )

                fwd_pred = fwd_pred_cache["pred"]
                target = intermediates[predict_to]
                fwd_loss = F.mse_loss(fwd_pred, target.detach())

                gate_w = fwd_pred_cache["gate_w"]
                r = target - fwd_pred.detach()

                gw_scalar = gate_w.detach().mean()
                gated_ll = (gw_scalar * r ** 2).mean()

                L_total = cls_loss + lambda_local * gated_ll

                opt_main.zero_grad()
                L_total.backward()
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                opt_main.step()

                opt_fwd.zero_grad()
                fwd_loss.backward()
                torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
                opt_fwd.step()

                acc = (logits.argmax(-1) == labels).float().mean().item()

                if step % eval_interval == 0 or step == wake_steps - 1:
                    v_loss, v_acc = eval_model(model)
                    history["train_loss"].append(
                        (global_step, cls_loss.item()))
                    history["train_acc"].append((global_step, acc))
                    history["val_loss"].append((global_step, v_loss))
                    history["val_acc"].append((global_step, v_acc))

                    dep_loss, _ = eval_model_with_ug(model, fm, ugate)
                    dep = v_loss - dep_loss
                    print(f"    step {global_step:5d} (wake {step}): "
                          f"val={v_loss:.4f} acc={v_acc:.4f} "
                          f"dep={dep:+.4f} "
                          f"proj={ugate.injection_norm():.3f}"
                          f" gw={gate_w.mean().item():.3f}"
                          f" ll={gated_ll.item():.4f}")

                global_step += 1

            gs = get_ug_gate_stats_by_digit(
                ugate, model, fm, f"{condition_label} c{cycle+1} wake")
            history["gate_stats"].append(
                (global_step, f"post_wake_c{cycle+1}", gs))

            wake_dep_loss, _ = eval_model_with_ug(model, fm, ugate)
            wake_noinj_loss, _ = eval_model(model)
            wake_dep_gap = wake_noinj_loss - wake_dep_loss
            print(f"    Post-wake dep gap: {wake_dep_gap:+.4f}")

            # === SLEEP: Distillation ===
            print(f"  [SLEEP] Distillation ({distill_steps} steps)")

            teacher = make_vit()
            teacher.load_state_dict(model.state_dict())
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            teacher_fm = make_fm()
            teacher_fm.load_state_dict(fm.state_dict())
            teacher_fm.eval()
            for p in teacher_fm.parameters():
                p.requires_grad = False
            teacher_ugate = UnifiedGate(n_embd, lg_hidden).to(device)
            teacher_ugate.load_state_dict(ugate.state_dict())
            teacher_ugate.eval()
            for p in teacher_ugate.parameters():
                p.requires_grad = False

            student = make_vit()
            student.load_state_dict(model.state_dict())
            opt_student = torch.optim.AdamW(
                student.parameters(), lr=distill_lr, weight_decay=0.01)

            for step in range(distill_steps):
                student.train()
                idx = batch_indices[global_step]
                images = train_images[idx].to(device)
                labels = train_labels[idx].to(device)

                with torch.no_grad():
                    def t_cb(act):
                        fp = teacher_fm(act)
                        inj, _ = teacher_ugate(act, fp)
                        return inj
                    teacher_logits, _, _ = teacher(
                        images, labels, return_intermediates=True,
                        cerebellar_fn=t_cb,
                        cerebellar_input_block=cerebellar_input_block,
                        cerebellar_inject_block=inject_after_block,
                    )

                student_logits, ce_loss = student(images, labels)
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    F.softmax(teacher_logits, dim=-1),
                    reduction="batchmean",
                )
                loss = distill_alpha * kl_loss + (1 - distill_alpha) * ce_loss

                opt_student.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_student.step()

                if step % eval_interval == 0 or step == distill_steps - 1:
                    v_loss, v_acc = eval_model(student)
                    history["val_loss"].append((global_step, v_loss))
                    history["val_acc"].append((global_step, v_acc))
                    print(f"    step {global_step:5d} (sleep {step}): "
                          f"val={v_loss:.4f} acc={v_acc:.4f}")

                global_step += 1

            model.load_state_dict(student.state_dict())
            del teacher, teacher_fm, teacher_ugate, student, opt_student
            del opt_main, opt_fwd
            torch.cuda.empty_cache()

            # === RE-POINT: Fresh FM ===
            fm_seed = seed + 100 * (cycle + 1)
            rp_start = repoint_offset
            rp_end = rp_start + retrain_steps
            rp_idxs = repoint_indices_list[rp_start:rp_end]
            repoint_offset = rp_end

            print(f"  [RE-POINT] Fresh FM ({retrain_steps} steps, "
                  f"seed={fm_seed})")
            fm, fm_cos = train_fresh_fm(
                model, retrain_steps, fm_seed,
                f"{condition_label} c{cycle+1}", rp_idxs)

            # === CHECKPOINT EVALUATION ===
            cycle_step = (cycle + 1) * steps_per_cycle
            print(f"\n  {condition_label} CHECKPOINT at step {cycle_step} "
                  f"(end cycle {cycle+1})")

            c_loss, c_acc = eval_model(model)
            c_digits = eval_by_digit_group(model)
            c_rob = measure_robustness(model, f"{condition_label} c{cycle+1}")
            c_sk = self_knowledge_probes(
                model, fm, f"{condition_label} c{cycle+1}")
            c_res = compute_residual_stats(
                model, fm, f"{condition_label} c{cycle+1}")

            gs_post = get_ug_gate_stats_by_digit(
                ugate, model, fm,
                f"{condition_label} c{cycle+1} repoint")
            history["gate_stats"].append(
                (cycle_step, f"post_repoint_c{cycle+1}", gs_post))

            checkpoints[cycle_step] = {
                "val_loss": c_loss, "val_acc": c_acc,
                "digit_group": c_digits,
                "robustness": c_rob,
                "self_knowledge": c_sk,
                "residual_stats": c_res,
                "gate_stats_post_wake": history["gate_stats"][-2][2],
                "gate_stats_post_repoint": gs_post,
                "wake_dep_gap": wake_dep_gap,
                "fm_cosine": fm_cos,
                "phase": phase,
            }
            print(f"    loss={c_loss:.4f} acc={c_acc:.4f}")
            print(f"    known_acc={c_digits['known_acc']:.4f} "
                  f"novel_acc={c_digits['novel_acc']:.4f}")

        return model, fm, ugate, checkpoints, history

    # =================================================================
    # CONDITION 1: WS_UG_shift (digit shift)
    # =================================================================
    ug_shift_model, ug_shift_fm, ug_shift_gate, \
        ug_shift_ckpts, ug_shift_hist = run_ws_ug(
            "WS_UG_shift", shift_batch_indices, shift_repoint_indices)

    del ug_shift_model, ug_shift_fm
    torch.cuda.empty_cache()

    # =================================================================
    # CONDITION 2: WS_UG_full (stationary control)
    # =================================================================
    ug_full_model, ug_full_fm, ug_full_gate, \
        ug_full_ckpts, ug_full_hist = run_ws_ug(
            "WS_UG_full", full_batch_indices, full_repoint_indices)

    del ug_full_model, ug_full_fm
    torch.cuda.empty_cache()

    # =================================================================
    # CONDITION 3: OL_shift (open-loop, same digit schedule)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  OL_shift ({total_main_steps} steps)")
    print(f"{'='*60}")

    ol_model = make_vit()
    ol_model.load_state_dict(init_model_state)
    ol_opt = torch.optim.AdamW(ol_model.parameters(), lr=lr, weight_decay=0.01)

    ol_checkpoints = {}
    ol_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for step in range(total_main_steps):
        ol_model.train()
        idx = shift_batch_indices[step]
        images = train_images[idx].to(device)
        labels = train_labels[idx].to(device)

        logits, cls_loss = ol_model(images, labels)
        ol_opt.zero_grad()
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(ol_model.parameters(), 1.0)
        ol_opt.step()

        acc = (logits.argmax(-1) == labels).float().mean().item()

        if step % eval_interval == 0 or step == total_main_steps - 1:
            v_loss, v_acc = eval_model(ol_model)
            ol_history["train_loss"].append((step, cls_loss.item()))
            ol_history["train_acc"].append((step, acc))
            ol_history["val_loss"].append((step, v_loss))
            ol_history["val_acc"].append((step, v_acc))
            if step % 500 == 0:
                print(f"  step {step:5d}: val={v_loss:.4f} acc={v_acc:.4f}")

        if (step + 1) in checkpoint_steps:
            ckpt_step = step + 1
            cycle = checkpoint_steps.index(ckpt_step)
            phase = "phase1" if cycle < phase1_cycles else "phase2"
            print(f"\n  OL_shift CHECKPOINT at step {ckpt_step} ({phase})")

            ol_l, ol_a = eval_model(ol_model)
            ol_digits = eval_by_digit_group(ol_model)
            ol_r = measure_robustness(ol_model, f"OL_shift c{cycle+1}")

            rp_start = cycle * retrain_steps
            rp_idxs = shift_repoint_indices[rp_start:rp_start + retrain_steps]
            ol_fresh_fm, ol_fm_cos = train_fresh_fm(
                ol_model, retrain_steps, seed + 700 + ckpt_step,
                f"OL_shift c{cycle+1}", rp_idxs)
            ol_s = self_knowledge_probes(
                ol_model, ol_fresh_fm, f"OL_shift c{cycle+1}")
            ol_e = compute_residual_stats(
                ol_model, ol_fresh_fm, f"OL_shift c{cycle+1}")

            ol_checkpoints[ckpt_step] = {
                "val_loss": ol_l, "val_acc": ol_a,
                "digit_group": ol_digits,
                "robustness": ol_r,
                "self_knowledge": ol_s,
                "residual_stats": ol_e,
                "fm_cosine": ol_fm_cos,
                "phase": phase,
            }
            print(f"    loss={ol_l:.4f} acc={ol_a:.4f}")
            print(f"    known_acc={ol_digits['known_acc']:.4f} "
                  f"novel_acc={ol_digits['novel_acc']:.4f}")

            del ol_fresh_fm
            torch.cuda.empty_cache()

    del ol_opt
    torch.cuda.empty_cache()

    # =================================================================
    # SAVE
    # =================================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_tag = f"vit_{n_layer}L_{n_head}H_{n_embd}D"
    save_root = (f"{DATA_DIR}/a2a_forward/mnist_ood_unified_gate"
                 f"/{model_tag}/{gap_tag}")
    os.makedirs(save_root, exist_ok=True)

    torch.save(ug_shift_gate.state_dict(),
               os.path.join(save_root, "ug_shift_gate.pt"))
    torch.save(ug_full_gate.state_dict(),
               os.path.join(save_root, "ug_full_gate.pt"))
    torch.save(ol_model.state_dict(),
               os.path.join(save_root, "ol_shift_model.pt"))

    result = {
        "config": {
            "n_cycles": n_cycles,
            "phase1_cycles": phase1_cycles,
            "n_known_digits": n_known_digits,
            "known_digits": known_digits,
            "novel_digits": novel_digits,
            "wake_steps": wake_steps,
            "distill_steps": distill_steps,
            "retrain_steps": retrain_steps,
            "total_main_steps": total_main_steps,
            "checkpoint_steps": checkpoint_steps,
            "lr": lr, "fwd_lr": fwd_lr,
            "distill_lr": distill_lr, "distill_alpha": distill_alpha,
            "lambda_local": lambda_local, "lg_hidden": lg_hidden,
            "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
            "gate_type": "unified_gate_uniform_local_loss",
        },
        "ws_ug_shift": {
            "checkpoints": ug_shift_ckpts,
            "history": ug_shift_hist,
        },
        "ws_ug_full": {
            "checkpoints": ug_full_ckpts,
            "history": ug_full_hist,
        },
        "ol_shift": {
            "checkpoints": ol_checkpoints,
            "history": ol_history,
        },
    }

    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: OOD UNIFIED GATE (digit shift)")
    print(f"{'='*60}")

    print(f"\n  Phase 1: digits {known_digits} "
          f"(cycles 1-{phase1_cycles})")
    print(f"  Phase 2: digits 0-9 "
          f"(cycles {phase1_cycles+1}-{n_cycles})")

    print(f"\n  === Overall val loss / accuracy ===")
    print(f"  {'Cycle':>6s} | {'WS_UG_shift':>16s} | {'WS_UG_full':>16s} | "
          f"{'OL_shift':>16s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [ug_shift_ckpts, ug_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            row.append(f"{c.get('val_loss', 0):.4f} "
                       f"{c.get('val_acc', 0):.3f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>16s}" for r in row))

    print(f"\n  === Known (0-{n_known_digits-1}) accuracy ===")
    print(f"  {'Cycle':>6s} | {'WS_UG_shift':>12s} | {'WS_UG_full':>12s} | "
          f"{'OL_shift':>12s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [ug_shift_ckpts, ug_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            dg = c.get("digit_group", {})
            row.append(f"{dg.get('known_acc', 0):.4f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>12s}" for r in row))

    print(f"\n  === Novel ({n_known_digits}-9) accuracy ===")
    print(f"  {'Cycle':>6s} | {'WS_UG_shift':>12s} | {'WS_UG_full':>12s} | "
          f"{'OL_shift':>12s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [ug_shift_ckpts, ug_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            dg = c.get("digit_group", {})
            row.append(f"{dg.get('novel_acc', 0):.4f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>12s}" for r in row))

    print(f"\n  === Gate dynamics (WS_UG_shift): "
          f"known vs novel gate weights ===")
    for _, label, gs in ug_shift_hist["gate_stats"]:
        print(f"    {label:30s}: overall={gs['overall_mean']:.3f} "
              f"known={gs['known_mean']:.3f} novel={gs['novel_mean']:.3f} "
              f"diff={gs['known_novel_diff']:+.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.1%}")

    print(f"\n  === Gate dynamics (WS_UG_full): "
          f"known vs novel gate weights ===")
    for _, label, gs in ug_full_hist["gate_stats"]:
        print(f"    {label:30s}: overall={gs['overall_mean']:.3f} "
              f"known={gs['known_mean']:.3f} novel={gs['novel_mean']:.3f} "
              f"diff={gs['known_novel_diff']:+.3f} "
              f"sparse(<0.1)={gs['sparsity_below_0.1']:.1%}")

    print(f"\n  === Robustness (delta-loss at eps=1.0) ===")
    print(f"  {'Cycle':>6s} | {'WS_UG_shift':>12s} | {'WS_UG_full':>12s} | "
          f"{'OL_shift':>12s}")
    for cs in checkpoint_steps:
        row = []
        for ckpts in [ug_shift_ckpts, ug_full_ckpts, ol_checkpoints]:
            c = ckpts.get(cs, ckpts.get(str(cs), {}))
            r = c.get("robustness", {}).get("1.0", 0)
            row.append(f"{r:+.4f}")
        print(f"  {cs:6d} | " + " | ".join(f"{r:>12s}" for r in row))

    # Gate dimension correlation between shift and full
    final_cs = checkpoint_steps[-1]
    shift_dims = np.array(
        ug_shift_ckpts[final_cs]["gate_stats_post_repoint"]["dim_mean"])
    full_dims = np.array(
        ug_full_ckpts[final_cs]["gate_stats_post_repoint"]["dim_mean"])
    dim_corr = float(np.corrcoef(shift_dims, full_dims)[0, 1])
    print(f"\n  Gate dimension correlation (shift vs full): "
          f"r={dim_corr:.3f}")

    print(f"\n  Saved to {save_root}")
    return result


@app.local_entrypoint()
def main(
    n_cycles: int = 4,
    phase1_cycles: int = 2,
    n_known_digits: int = 7,
    wake_steps: int = 1500,
    distill_steps: int = 600,
    retrain_steps: int = 1500,
    lambda_local: float = 1.0,
    lg_hidden: int = 64,
    seed: int = 42,
):
    result = a2a_mnist_ood_unified_gate.remote(
        n_cycles=n_cycles,
        phase1_cycles=phase1_cycles,
        n_known_digits=n_known_digits,
        wake_steps=wake_steps,
        distill_steps=distill_steps,
        retrain_steps=retrain_steps,
        lambda_local=lambda_local,
        lg_hidden=lg_hidden,
        seed=seed,
    )
    print("\nOOD unified gate experiment complete.")

    config = result["config"]
    print(f"\n  Phase 1: digits {config['known_digits']} "
          f"(cycles 1-{config['phase1_cycles']})")
    print(f"  Phase 2: digits 0-9 "
          f"(cycles {config['phase1_cycles']+1}-{config['n_cycles']})")

    def get_ckpt(ckpts, step):
        return ckpts.get(step) or ckpts.get(str(step))

    final_step = config["checkpoint_steps"][-1]
    print(f"\n  Final ({final_step} steps):")
    for name, key in [("WS_UG_shift", "ws_ug_shift"),
                      ("WS_UG_full", "ws_ug_full"),
                      ("OL_shift", "ol_shift")]:
        c = get_ckpt(result[key]["checkpoints"], final_step)
        if c:
            dg = c.get("digit_group", {})
            rob = c.get("robustness", {}).get("1.0", 0)
            print(f"    {name:12s}: loss={c['val_loss']:.4f} "
                  f"known_acc={dg.get('known_acc', 0):.4f} "
                  f"novel_acc={dg.get('novel_acc', 0):.4f} "
                  f"rob(eps=1)={rob:+.4f}")
