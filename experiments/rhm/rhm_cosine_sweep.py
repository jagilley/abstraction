"""RHM FM cosine sweep: find (L, m, FM_size) that produce cosine ~0.97.

At L=4 (seq_len=16), the FM captures 99%+ of computation -- too easy for
structured residuals to emerge. We need cosine in the 0.90-0.97 range
(like MNIST 0.903, language 0.935). Two levers:
  1. Longer sequences (L=5→32, L=6→64) for more complex attention patterns
  2. FM capacity: "matched" (14% of gap ≈ MNIST) and "current" (28% ≈ language)

Grid: L∈{5,6} × m∈{2,4,8} × FM∈{matched, current}, 3-block gap.
"""

import json
import os
import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
    )
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-cosine-sweep", image=image)

FM_CONFIGS = {
    "matched": {"fwd_n_layer": 2, "fwd_d_head": 16, "fwd_n_head": 1, "fwd_mlp_mult": 1.0},
    "current": {"fwd_n_layer": 2, "fwd_d_head": 32, "fwd_n_head": 1, "fwd_mlp_mult": 2.0},
}


def _ensure_corpus(v, s, L, m, n_tokens):
    import numpy as np
    from rhm.rhm_data import make_corpus

    key = setting_key(v, s, L, m)
    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    if os.path.exists(corpus_path):
        existing = np.load(corpus_path)
        if len(existing) >= n_tokens:
            return

    corpus, rules, meta = make_corpus(v, s, L, m, n_tokens)
    out_dir = f"{DATA_DIR}/{key}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "corpus.npy"), corpus)
    for ell, r in enumerate(rules):
        np.save(os.path.join(out_dir, f"rules_L{ell}.npy"), r)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()


@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def train_main_model(
    v: int = 8, s: int = 2, depth: int = 5, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4,
    seed: int = 42,
    n_eval_batches: int = 10,
):
    """Train main model, return path to final checkpoint."""
    import torch
    import numpy as np
    from rhm.model import GPT

    torch.manual_seed(seed)
    np.random.seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"

    volume.reload()
    _ensure_corpus(v, s, L, m, n_tokens)

    data = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    save_dir = f"{DATA_DIR}/{key}/cosine_sweep"
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Training: {key}, {n_layer}L/{n_head}H/{n_embd}D ===")
    print(f"  seq_len={seq_len}, n_steps={n_steps}, epochs={n_steps/steps_per_epoch:.1f}")

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    eval_interval = 200
    for step in range(1, n_steps + 1):
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % (eval_interval * 10) == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_loss = sum(float(model(*get_batch(val_data))[1])
                               for _ in range(n_eval_batches)) / n_eval_batches
            print(f"  step {step:6d}/{n_steps}: val={val_loss:.4f}")

    # Final eval
    model.eval()
    with torch.no_grad():
        final_val = sum(float(model(*get_batch(val_data))[1])
                        for _ in range(n_eval_batches)) / n_eval_batches

    ckpt_path = os.path.join(save_dir, "model_final.pt")
    torch.save(model.state_dict(), ckpt_path)
    volume.commit()

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "seq_len": seq_len, "n_steps": n_steps, "n_params": n_params,
        "final_val_loss": float(final_val),
        "checkpoint_path": ckpt_path,
    }
    print(f"  Final: val={final_val:.4f}, saved to {ckpt_path}")
    return result


@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def eval_fm(
    v: int = 8, s: int = 2, depth: int = 5, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    checkpoint_path: str = "",
    fm_label: str = "matched",
    fwd_n_layer: int = 2, fwd_d_head: int = 16, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 1.0,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fm_train_steps: int = 5000,
    fm_lr: float = 1e-3,
    batch_size: int = 64,
    fm_seed: int = 137,
):
    """Train FM on frozen model activations, return cosine + residual metrics."""
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(fm_seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"

    volume.reload()

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    corpus = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    corpus = torch.from_numpy(corpus[:n_tokens].astype(np.int64))
    split = int(0.9 * len(corpus))
    train_data = corpus[:split]
    val_data = corpus[split:]

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())

    main_block_params = sum(
        p.numel() for p in list(model.transformer.h[0].parameters())
    )
    gap_blocks = int(predict_to.replace("post_block", "")) - int(predict_from.replace("post_block", ""))
    gap_params = gap_blocks * main_block_params
    capacity_ratio = fwd_n_params / gap_params if gap_params > 0 else 0

    print(f"=== FM eval: {key}, FM={fm_label} ({fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}) ===")
    print(f"  FM params: {fwd_n_params/1e3:.1f}K, gap params: {gap_params/1e3:.1f}K, "
          f"ratio: {capacity_ratio*100:.1f}%")

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fm_lr, weight_decay=0.01)

    train_cosines = []
    for fm_step in range(fm_train_steps):
        fwd_model.train()
        x, y = get_batch(train_data)
        with torch.no_grad():
            _, _, intermediates = model(x, y, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]

        predicted = fwd_model(source)
        fwd_loss = F.mse_loss(predicted, target)

        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_fwd.step()

        if fm_step % 1000 == 0:
            with torch.no_grad():
                cos = F.cosine_similarity(predicted, target, dim=-1).mean().item()
            train_cosines.append((fm_step, cos))
            print(f"  FM step {fm_step}: mse={float(fwd_loss):.6f} cos={cos:.4f}")

    # --- Final eval on val set ---
    fwd_model.eval()
    all_cosines = []
    all_res_norms = []
    all_residuals = []
    n_eval_batches = 50

    with torch.no_grad():
        for _ in range(n_eval_batches):
            x, y = get_batch(val_data)
            _, _, intermediates = model(x, y, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
            predicted = fwd_model(source)

            residual = target - predicted
            cos = F.cosine_similarity(predicted, target, dim=-1).mean().item()
            res_norm = residual.norm(dim=-1).mean().item()

            all_cosines.append(cos)
            all_res_norms.append(res_norm)
            all_residuals.append(residual.cpu().numpy())

    mean_cosine = float(np.mean(all_cosines))
    mean_res_norm = float(np.mean(all_res_norms))

    # --- Effective rank ---
    residuals_np = np.concatenate(all_residuals, axis=0)
    res_flat = residuals_np.reshape(-1, n_embd)
    n_samples = min(50000, res_flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(res_flat.shape[0], n_samples, replace=False)
    sub = res_flat[idx] - res_flat[idx].mean(axis=0)
    _, S, _ = np.linalg.svd(sub, full_matrices=False)
    S_norm = S / S.sum()
    eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))
    top1_var = float((S[0] ** 2) / (S ** 2).sum())

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "seq_len": seq_len,
        "fm_label": fm_label,
        "fm_config": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}",
        "fm_params": fwd_n_params,
        "capacity_ratio_pct": float(capacity_ratio * 100),
        "gap": f"{predict_from}->{predict_to}",
        "fm_cosine": float(mean_cosine),
        "residual_norm": float(mean_res_norm),
        "effective_rank": float(eff_rank),
        "effective_rank_pct": float(eff_rank / n_embd * 100),
        "top1_pc_variance": float(top1_var),
        "train_cosine_trajectory": train_cosines,
    }

    print(f"\n  RESULT: cos={mean_cosine:.4f} (1-cos={1-mean_cosine:.4f}) "
          f"res_norm={mean_res_norm:.4f} eff_rank={eff_rank:.1f}/{n_embd} "
          f"({eff_rank/n_embd*100:.1f}%) top1={top1_var*100:.1f}%")
    return result


@app.function(
    volumes={DATA_DIR: volume},
    timeout=21600,
    memory=4096,
)
def rhm_cosine_sweep(
    v: int = 8, s: int = 2,
    l_values: str = "5,6",
    m_values: str = "2,4,8",
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fm_train_steps: int = 5000,
):
    """Sweep (L, m, FM_size) to find cosine ~0.97 sweet spot."""
    Ls = [int(x) for x in l_values.split(",")]
    ms = [int(x) for x in m_values.split(",")]

    print(f"=== RHM Cosine Sweep ===")
    print(f"  L values: {Ls} (seq_lens: {[s**L for L in Ls]})")
    print(f"  m values: {ms}")
    print(f"  FM configs: {list(FM_CONFIGS.keys())}")
    print(f"  Gap: {predict_from} -> {predict_to}")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")
    print()

    # Phase 1: train main models in parallel
    print("--- Phase 1: Training main models ---")
    train_handles = {}
    for L in Ls:
        for mi in ms:
            key = setting_key(v, s, L, mi)
            h = train_main_model.spawn(
                v=v, s=s, depth=L, m=mi, n_tokens=n_tokens,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            )
            train_handles[key] = h
            print(f"  Spawned training for {key} (seq_len={s**L})")

    train_results = {}
    for key, h in train_handles.items():
        train_results[key] = h.get()
        tr = train_results[key]
        print(f"  {key}: val_loss={tr['final_val_loss']:.4f} "
              f"({tr['n_steps']} steps, {tr['n_params']/1e3:.0f}K params)")

    # Phase 2: train FMs in parallel (2 per main model)
    print("\n--- Phase 2: Training FMs ---")
    fm_handles = []
    for L in Ls:
        for mi in ms:
            key = setting_key(v, s, L, mi)
            tr = train_results[key]
            for fm_label, fm_cfg in FM_CONFIGS.items():
                h = eval_fm.spawn(
                    v=v, s=s, depth=L, m=mi, n_tokens=n_tokens,
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                    checkpoint_path=tr["checkpoint_path"],
                    fm_label=fm_label,
                    predict_from=predict_from, predict_to=predict_to,
                    fm_train_steps=fm_train_steps,
                    **fm_cfg,
                )
                fm_handles.append((key, fm_label, h))
                print(f"  Spawned FM '{fm_label}' for {key}")

    fm_results = []
    for key, fm_label, h in fm_handles:
        r = h.get()
        r["main_val_loss"] = train_results[key]["final_val_loss"]
        fm_results.append(r)
        print(f"  {key} [{fm_label}]: cos={r['fm_cosine']:.4f}")

    # --- Summary table ---
    print(f"\n{'='*120}")
    print(f"RHM COSINE SWEEP RESULTS")
    print(f"Model: {n_layer}L/{n_head}H/{n_embd}D, gap: {predict_from}->{predict_to}")
    print(f"{'='*120}")

    header = (f"{'Setting':>16}  {'seq':>4}  {'m':>2}  {'FM':>10}  {'FM_K':>5}  "
              f"{'ratio%':>6}  {'val_loss':>8}  {'cosine':>7}  {'1-cos':>7}  "
              f"{'res_norm':>8}  {'rank':>6}  {'rank%':>6}  {'top1%':>6}")
    print(header)
    print("-" * len(header))

    fm_results.sort(key=lambda r: (r["L"], r["m"], r["fm_label"]))
    for r in fm_results:
        print(f"{r['setting']:>16}  {r['seq_len']:>4}  {r['m']:>2}  "
              f"{r['fm_label']:>10}  {r['fm_params']/1e3:>5.1f}  "
              f"{r['capacity_ratio_pct']:>6.1f}  {r['main_val_loss']:>8.4f}  "
              f"{r['fm_cosine']:>7.4f}  {1-r['fm_cosine']:>7.4f}  "
              f"{r['residual_norm']:>8.4f}  {r['effective_rank']:>6.1f}  "
              f"{r['effective_rank_pct']:>6.1f}  {r['top1_pc_variance']*100:>6.1f}")

    # --- Cross-domain comparison ---
    print(f"\n{'='*80}")
    print("CROSS-DOMAIN COSINE COMPARISON (target range: 0.90-0.97)")
    print(f"{'='*80}")
    print(f"  MNIST (ViT 128D, 1L/1H/32d FM, 3-block gap):    cos=0.903  1-cos=9.7%")
    print(f"  Language 29M (256D, 2L/1H/64d FM, 3-block gap):  cos=0.935  1-cos=6.5%")
    print(f"  RHM L=4 (128D, 2L/1H/32d FM, 3-block gap):      cos=0.994  1-cos=0.6%")
    print()
    for r in fm_results:
        marker = ""
        if 0.90 <= r["fm_cosine"] <= 0.97:
            marker = " <-- SWEET SPOT"
        elif r["fm_cosine"] < 0.90:
            marker = " <-- too hard"
        elif r["fm_cosine"] > 0.99:
            marker = " <-- too easy"
        print(f"  {r['setting']} [{r['fm_label']:>8}]: cos={r['fm_cosine']:.4f}  "
              f"1-cos={1-r['fm_cosine']:.4f}{marker}")

    # Save results
    save_dir = f"{DATA_DIR}/rhm_cosine_sweep"
    os.makedirs(save_dir, exist_ok=True)

    full_results = {
        "experiment": "rhm_cosine_sweep",
        "params": {
            "v": v, "s": s, "L_values": Ls, "m_values": ms,
            "n_tokens": n_tokens,
            "model": f"{n_layer}L/{n_head}H/{n_embd}D",
            "gap": f"{predict_from}->{predict_to}",
            "fm_configs": {k: str(v) for k, v in FM_CONFIGS.items()},
            "fm_train_steps": fm_train_steps,
        },
        "training": {k: v for k, v in train_results.items()},
        "fm_results": fm_results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(full_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/results.json")
    return full_results
