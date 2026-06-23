"""RHM residual rank experiment: does FSM residual rank reflect DGP complexity?

Tests the hypothesis that the rank of a forward self-model's residual
(actual - predicted activations) reflects the computational complexity of
the data-generating process, as captured by the main model.

Cross-domain observations suggest this (grokking ~15, MNIST ~18, language ~200),
but those comparisons are confounded by everything else changing between domains.
The RHM gives clean, continuous control over DGP complexity (L, m) while holding
the model architecture, vocabulary, and training procedure fixed.

Prediction: higher DGP complexity (larger L, larger m) → higher residual rank,
because the main model's inter-layer computation is more varied/distributed.
"""

import json
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

app = modal.App("rhm-residual-rank", image=image)


def _ensure_corpus(v, s, L, m, n_tokens):
    """Generate corpus inline if it doesn't exist."""
    import os
    import numpy as np
    from rhm.rhm_data import make_corpus

    key = setting_key(v, s, L, m)
    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    if os.path.exists(corpus_path):
        existing = np.load(corpus_path)
        if len(existing) >= n_tokens:
            return
        print(f"  Corpus for {key} too small ({len(existing):,} < {n_tokens:,}), regenerating")

    print(f"  Generating corpus for {key} ({n_tokens:,} tokens)...")
    corpus, rules, meta = make_corpus(v, s, L, m, n_tokens, rule_seed=0)
    out_dir = f"{DATA_DIR}/{key}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "corpus.npy"), corpus)
    for ell, r in enumerate(rules):
        np.save(os.path.join(out_dir, f"rules_L{ell}.npy"), r)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()
    print(f"  Saved {len(corpus):,} tokens")


@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def train_and_analyze(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4, fwd_lr: float = 1e-3,
    n_steps: int = 0,
    eval_interval: int = 200, n_eval_batches: int = 10,
    n_analysis_batches: int = 20,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    fwd_n_layer: int = 1, fwd_d_head: int = 32, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
):
    """Co-train GPT + forward self-model on RHM data, then analyze residual rank."""
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda" if torch.cuda.is_available() else "cpu"

    volume.reload()
    _ensure_corpus(v, s, L, m, n_tokens)

    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    data = np.load(corpus_path)
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    actual_n_tokens = len(data)

    split = int(0.9 * actual_n_tokens)
    train_data = data[:split]
    val_data = data[split:]

    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    if n_steps == 0:
        n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    print(f"=== RHM residual rank: {key} ===")
    print(f"  seq_len={seq_len}, P={actual_n_tokens:,}, steps={n_steps}")
    print(f"  Main model: {n_layer}L/{n_head}H/{n_embd}D")

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    main_n_params = sum(p.numel() for p in model.parameters())

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
    capacity_ratio = fwd_n_params / main_n_params

    print(f"  Forward model: {fwd_n_params:,} params ({capacity_ratio:.1%})")
    print(f"  Gap: {predict_from} → {predict_to}")

    opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    history = {
        "lm_loss": [], "fwd_mse": [],
        "val_lm_loss": [], "val_fwd_mse": [], "val_cosine_sim": [],
        "val_residual_norm": [], "val_effective_rank": [],
    }
    best_val_loss = float("inf")

    for step in range(n_steps):
        model.train()
        fwd_model.train()

        x, y = get_batch(train_data)
        _, lm_loss, intermediates = model(x, y, return_intermediates=True)

        source = intermediates[predict_from].detach()
        target = intermediates[predict_to].detach()
        predicted = fwd_model(source)
        fwd_loss = F.mse_loss(predicted, target)

        opt_main.zero_grad()
        lm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()

        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_fwd.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            fwd_model.eval()
            with torch.no_grad():
                val_lm_acc, val_fwd_acc, val_cos_acc, val_res_acc = 0., 0., 0., 0.
                eval_residuals = []

                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data)
                    _, vl, vi = model(vx, vy, return_intermediates=True)
                    val_lm_acc += float(vl)

                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = fwd_model(src)
                    val_fwd_acc += float(F.mse_loss(pred, tgt))
                    val_cos_acc += float(F.cosine_similarity(pred, tgt, dim=-1).mean())
                    residual = tgt - pred
                    val_res_acc += float(residual.norm(dim=-1).mean())
                    eval_residuals.append(residual.cpu())

                n = n_eval_batches
                val_lm = val_lm_acc / n
                val_fwd = val_fwd_acc / n
                val_cos = val_cos_acc / n
                val_res = val_res_acc / n
                if val_lm < best_val_loss:
                    best_val_loss = val_lm

                eff_rank = _compute_effective_rank(eval_residuals, n_embd)

                history["lm_loss"].append((step, float(lm_loss)))
                history["fwd_mse"].append((step, float(fwd_loss)))
                history["val_lm_loss"].append((step, val_lm))
                history["val_fwd_mse"].append((step, val_fwd))
                history["val_cosine_sim"].append((step, val_cos))
                history["val_residual_norm"].append((step, val_res))
                history["val_effective_rank"].append((step, eff_rank))

                print(f"  step {step:6d}: lm={float(lm_loss):.4f} val={val_lm:.4f} "
                      f"fwd={float(fwd_loss):.6f} cos={val_cos:.4f} "
                      f"res={val_res:.3f} rank={eff_rank:.1f}/{n_embd}")

    # --- Final analysis ---
    print(f"\n=== Final analysis: {key} ===")
    analysis = _run_residual_analysis(
        model, fwd_model, val_data, get_batch,
        predict_from, predict_to,
        n_analysis_batches, n_embd, device,
    )

    for k, val in analysis.items():
        if isinstance(val, dict):
            print(f"  {k}:")
            for kk, vv in val.items():
                if isinstance(vv, float):
                    print(f"    {kk}: {vv:.4f}")
        elif isinstance(val, float):
            print(f"  {k}: {val:.4f}")

    fwd_tag = f"fwd_{fwd_n_layer}L_{fwd_n_head}H_{fwd_d_head}d_mlp{fwd_mlp_mult}"
    save_dir = f"{DATA_DIR}/{key}/a2a_residual_rank/{fwd_tag}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "seq_len": seq_len,
        "main_model": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "n_params": main_n_params,
            "best_val_loss": best_val_loss,
            "final_val_loss": history["val_lm_loss"][-1][1],
        },
        "forward_model": {
            "n_layer": fwd_n_layer, "n_head": fwd_n_head,
            "d_head": fwd_d_head, "mlp_mult": fwd_mlp_mult,
            "n_params": fwd_n_params, "capacity_ratio": capacity_ratio,
        },
        "training": {
            "n_tokens": actual_n_tokens, "block_size": block_size,
            "n_steps": n_steps, "batch_size": batch_size,
            "lr": lr, "fwd_lr": fwd_lr,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "analysis": analysis,
        "history": history,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nSaved to {save_dir}")
    return result


@app.function(
    volumes={DATA_DIR: volume},
    timeout=14400,
    memory=4096,
)
def rhm_residual_rank_sweep(
    v: int = 8, s: int = 2,
    l_values: str = "4,6,8", m_values: str = "2,4,8",
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    fwd_n_layer: int = 1, fwd_d_head: int = 32, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
):
    """Sweep across (L, m) settings and compare residual rank."""
    import os

    Ls = [int(x) for x in l_values.split(",")]
    ms = [int(x) for x in m_values.split(",")]

    settings = [(L, mi) for L in Ls for mi in ms]
    print(f"Sweeping {len(settings)} settings: {settings}")
    print(f"  v={v}, s={s}, P={n_tokens:,}")
    print(f"  Model: {n_layer}L/{n_head}H/{n_embd}D")

    # Train+analyze in parallel (each job generates its own corpus if needed)
    handles = []
    for L, mi in settings:
        h = train_and_analyze.spawn(
            v=v, s=s, depth=L, m=mi, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
            fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        handles.append((L, mi, h))

    results = []
    for L, mi, h in handles:
        r = h.get()
        results.append(r)

    # Phase 3: summary table
    print(f"\n{'='*80}")
    print(f"RESIDUAL RANK vs DGP COMPLEXITY (v={v}, s={s})")
    print(f"{'='*80}")
    print(f"{'Setting':<16} {'L':>3} {'m':>3} {'seq_len':>8} "
          f"{'eff_rank':>9} {'max':>4} {'rank%':>6} "
          f"{'top1_var':>9} {'top5_var':>9} {'cosine':>7} "
          f"{'val_loss':>9}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: (x["L"], x["m"])):
        pca = r["analysis"]["residual_pca"]
        q = r["analysis"]["basic_quality"]
        print(f"{r['setting']:<16} {r['L']:>3} {r['m']:>3} {r['seq_len']:>8} "
              f"{pca['effective_rank']:>9.1f} {pca['max_rank']:>4} "
              f"{pca['effective_rank']/pca['max_rank']*100:>5.1f}% "
              f"{pca['top1_pc_variance']:>9.4f} {pca['top5_pc_variance']:>9.4f} "
              f"{q['mean_cosine_sim']:>7.4f} "
              f"{r['main_model']['final_val_loss']:>9.4f}")

    print(f"\n{'='*80}")

    summary = {
        "experiment": "rhm_residual_rank",
        "v": v, "s": s,
        "settings": [(r["setting"], {
            "L": r["L"], "m": r["m"], "seq_len": r["seq_len"],
            "effective_rank": r["analysis"]["residual_pca"]["effective_rank"],
            "max_rank": r["analysis"]["residual_pca"]["max_rank"],
            "top1_pc_variance": r["analysis"]["residual_pca"]["top1_pc_variance"],
            "top5_pc_variance": r["analysis"]["residual_pca"]["top5_pc_variance"],
            "cosine_sim": r["analysis"]["basic_quality"]["mean_cosine_sim"],
            "val_loss": r["main_model"]["final_val_loss"],
        }) for r in sorted(results, key=lambda x: (x["L"], x["m"]))],
    }

    save_dir = f"{DATA_DIR}/rhm_residual_rank_sweep"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"Summary saved to {save_dir}/summary.json")

    return summary


@app.function(
    volumes={DATA_DIR: volume},
    timeout=14400,
    memory=4096,
)
def capacity_sweep(
    v: int = 8, s: int = 2,
    depths: str = "4,6", m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
):
    """Sweep FM capacity on well-learned RHM settings.

    Tests whether high residual rank at 10% FM capacity is genuine
    or capacity noise by scaling the FM to 25%, 50%, and 100% of
    the main model.
    """
    import os

    Ls = [int(x) for x in depths.split(",")]

    fm_configs = [
        ("10%", dict(fwd_n_layer=1, fwd_d_head=32, fwd_n_head=1, fwd_mlp_mult=2.0)),
        ("25%", dict(fwd_n_layer=1, fwd_d_head=32, fwd_n_head=4, fwd_mlp_mult=4.0)),
        ("50%", dict(fwd_n_layer=2, fwd_d_head=32, fwd_n_head=4, fwd_mlp_mult=4.0)),
        ("100%", dict(fwd_n_layer=4, fwd_d_head=32, fwd_n_head=4, fwd_mlp_mult=4.0)),
    ]

    print(f"FM capacity sweep on {len(Ls)} settings × {len(fm_configs)} capacities")

    handles = []
    for L in Ls:
        for label, cfg in fm_configs:
            h = train_and_analyze.spawn(
                v=v, s=s, depth=L, m=m, n_tokens=n_tokens,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                **cfg,
            )
            handles.append((L, label, h))

    results = []
    for L, label, h in handles:
        r = h.get()
        r["_capacity_label"] = label
        results.append(r)

    print(f"\n{'='*100}")
    print(f"FM CAPACITY SWEEP: does residual rank drop with larger FM?")
    print(f"{'='*100}")
    print(f"{'Setting':<16} {'FM_cap':>6} {'FM_params':>10} {'eff_rank':>9} {'max':>4} "
          f"{'rank%':>6} {'top1_var':>9} {'top5_var':>9} {'cosine':>7} "
          f"{'res_norm':>9} {'val_loss':>9}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: (x["L"], x["forward_model"]["n_params"])):
        pca = r["analysis"]["residual_pca"]
        q = r["analysis"]["basic_quality"]
        fm = r["forward_model"]
        print(f"{r['setting']:<16} {r['_capacity_label']:>6} {fm['n_params']:>10,} "
              f"{pca['effective_rank']:>9.1f} {pca['max_rank']:>4} "
              f"{pca['effective_rank']/pca['max_rank']*100:>5.1f}% "
              f"{pca['top1_pc_variance']:>9.4f} {pca['top5_pc_variance']:>9.4f} "
              f"{q['mean_cosine_sim']:>7.4f} "
              f"{q['mean_residual_norm']:>9.4f} "
              f"{r['main_model']['final_val_loss']:>9.4f}")

    print(f"\n{'='*100}")

    save_dir = f"{DATA_DIR}/rhm_capacity_sweep"
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "experiment": "rhm_fm_capacity_sweep",
        "results": [{
            "setting": r["setting"], "L": r["L"], "m": r["m"],
            "capacity_label": r["_capacity_label"],
            "fm_params": r["forward_model"]["n_params"],
            "effective_rank": r["analysis"]["residual_pca"]["effective_rank"],
            "max_rank": r["analysis"]["residual_pca"]["max_rank"],
            "top1_pc_variance": r["analysis"]["residual_pca"]["top1_pc_variance"],
            "cosine_sim": r["analysis"]["basic_quality"]["mean_cosine_sim"],
            "residual_norm": r["analysis"]["basic_quality"]["mean_residual_norm"],
            "val_loss": r["main_model"]["final_val_loss"],
        } for r in sorted(results, key=lambda x: (x["L"], x["forward_model"]["n_params"]))],
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"Saved to {save_dir}/summary.json")
    return summary


@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def train_main_model(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    batch_size: int = 64, lr: float = 3e-4,
    n_steps: int = 0, seed: int = 42,
    eval_interval: int = 500, n_eval_batches: int = 10,
):
    """Train main model alone and save checkpoint for FM experiments."""
    import os
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
    if n_steps == 0:
        n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    print(f"=== Training main model: {key} ===")
    print(f"  {n_layer}L/{n_head}H/{n_embd}D, {n_steps} steps, seed={seed}")

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())
    print(f"  Total: {n_params:,} params, block: {block_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    best_val = float("inf")
    final_val = None
    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vl = sum(float(model(*get_batch(val_data))[1])
                         for _ in range(n_eval_batches)) / n_eval_batches
            if vl < best_val:
                best_val = vl
            final_val = vl
            print(f"  step {step:6d}: train={float(loss):.4f} val={vl:.4f}")

    save_dir = f"{DATA_DIR}/{key}/arch_matched_main_seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    info = {
        "key": key, "L": L, "m": m,
        "n_params": n_params, "block_params": block_params,
        "n_steps": n_steps, "seed": seed,
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "save_dir": save_dir,
    }
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"  Saved to {save_dir}, val={final_val:.4f}")
    return info


@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=7200,
    memory=16384,
)
def train_matched_fm(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    main_model_dir: str = "",
    fwd_d_head: int = 32, fwd_mlp_mult: float = 4.0,
    predict_from: str = "post_block0", predict_to: str = "post_block1",
    batch_size: int = 64, fwd_lr: float = 1e-3,
    n_steps: int = 0, seed: int = 137,
    eval_interval: int = 200, n_eval_batches: int = 10,
    n_analysis_batches: int = 20,
):
    """Train architecture-matched FM on frozen main model activations.

    The FM uses the same number of heads as the main model block (n_head),
    eliminating the head-count mismatch confound. Only d_head and mlp_mult
    vary, controlling capacity while keeping the architecture compatible.
    At d_head=32/mlp_mult=4.0, the FM is an exact architectural copy of
    one main model block (~198K params).
    """
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"

    volume.reload()

    data = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    if n_steps == 0:
        n_steps = min(20000, max(2000, 5 * steps_per_epoch))

    # Load frozen main model
    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    ckpt_path = os.path.join(main_model_dir, "model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    block_params = sum(p.numel() for p in model.transformer.h[0].parameters())

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=n_head,
        n_layer=1, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())
    block_frac = fwd_n_params / block_params

    print(f"=== Matched FM on {key}: 1L/{n_head}H/{fwd_d_head}d mlp{fwd_mlp_mult} ===")
    print(f"  FM: {fwd_n_params:,} params ({block_frac:.1%} of block's {block_params:,})")
    print(f"  Gap: {predict_from} → {predict_to}, steps={n_steps}")

    opt = torch.optim.AdamW(fwd_model.parameters(), lr=fwd_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_steps, eta_min=fwd_lr * 0.01,
    )

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    for step in range(n_steps):
        fwd_model.train()
        x, y = get_batch(train_data)

        with torch.no_grad():
            _, _, intermediates = model(x, y, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]

        predicted = fwd_model(source)
        loss = F.mse_loss(predicted, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            fwd_model.eval()
            with torch.no_grad():
                val_mse, val_cos, val_res, val_tgt = 0., 0., 0., 0.
                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data)
                    _, _, vi = model(vx, vy, return_intermediates=True)
                    src = vi[predict_from]
                    tgt = vi[predict_to]
                    pred = fwd_model(src)
                    val_mse += float(F.mse_loss(pred, tgt))
                    val_cos += float(F.cosine_similarity(pred, tgt, dim=-1).mean())
                    val_res += float((tgt - pred).norm(dim=-1).mean())
                    val_tgt += float(tgt.norm(dim=-1).mean())
                n = n_eval_batches
                print(f"  step {step:6d}: mse={float(loss):.6f} "
                      f"val_mse={val_mse/n:.6f} cos={val_cos/n:.4f} "
                      f"res={val_res/n:.4f} tgt={val_tgt/n:.2f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Final analysis
    analysis = _run_residual_analysis(
        model, fwd_model, val_data, get_batch,
        predict_from, predict_to,
        n_analysis_batches, n_embd, device,
    )

    # Measure activation norms for context
    with torch.no_grad():
        tgt_norms = []
        for _ in range(n_analysis_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)
            tgt_norms.append(vi[predict_to].norm(dim=-1).cpu())
        mean_tgt_norm = float(torch.cat(tgt_norms).mean())

    analysis["basic_quality"]["mean_target_norm"] = mean_tgt_norm
    analysis["basic_quality"]["relative_residual"] = (
        analysis["basic_quality"]["mean_residual_norm"] / mean_tgt_norm
    )

    q = analysis["basic_quality"]
    pca = analysis["residual_pca"]
    print(f"\n=== Final: {key}, FM 1L/{n_head}H/{fwd_d_head}d mlp{fwd_mlp_mult} ===")
    print(f"  res_norm={q['mean_residual_norm']:.4f} "
          f"(tgt_norm={mean_tgt_norm:.2f}, relative={q['relative_residual']:.4f})")
    print(f"  cosine={q['mean_cosine_sim']:.4f}")
    print(f"  eff_rank={pca['effective_rank']:.1f}/{pca['max_rank']}")

    fwd_tag = f"archmatched_1L_{n_head}H_{fwd_d_head}d_mlp{fwd_mlp_mult}"
    save_dir = f"{DATA_DIR}/{key}/a2a_residual_rank/{fwd_tag}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(fwd_model.state_dict(), os.path.join(save_dir, "fwd_model.pt"))

    result = {
        "setting": key, "v": v, "s": s, "L": L, "m": m,
        "seq_len": seq_len,
        "main_model": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "block_params": block_params,
        },
        "forward_model": {
            "n_head": n_head, "d_head": fwd_d_head,
            "mlp_mult": fwd_mlp_mult, "n_params": fwd_n_params,
            "block_fraction": block_frac,
        },
        "training": {
            "n_steps": n_steps, "fwd_lr": fwd_lr, "seed": seed,
            "predict_from": predict_from, "predict_to": predict_to,
        },
        "analysis": analysis,
    }

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return result


@app.function(
    volumes={DATA_DIR: volume},
    timeout=14400,
    memory=4096,
)
def architecture_matched_sweep(
    v: int = 8, s: int = 2,
    depths: str = "4,6", m: int = 2,
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
):
    """Architecture-matched FM capacity sweep.

    Tests whether residual rank decreases as residual norm approaches zero
    when the FM architecture matches the main model block (same head count).
    Eliminates the head-count mismatch confound from the original capacity sweep.

    Design:
    - Phase 1: Train main models with fixed seed, save checkpoints
    - Phase 2: For each FM capacity, load frozen main model, train FM only
    - All FMs use 1 layer and n_head heads (matching block1)
    - d_head scales from 8→32, mlp_mult from 1→4 (25%→100% of block)
    - At 100%, FM is an exact architectural copy of one main model block
    """
    import os

    Ls = [int(x) for x in depths.split(",")]

    fm_configs = [
        ("25%", dict(fwd_d_head=8, fwd_mlp_mult=1.0)),
        ("50%", dict(fwd_d_head=16, fwd_mlp_mult=2.0)),
        ("75%", dict(fwd_d_head=24, fwd_mlp_mult=3.0)),
        ("100%", dict(fwd_d_head=32, fwd_mlp_mult=4.0)),
    ]

    print(f"Architecture-matched FM sweep: {len(Ls)} settings x {len(fm_configs)} capacities")
    print(f"  All FMs: 1L/{n_head}H (matching main model block)")
    print(f"  Main model frozen before FM training")

    # Phase 1: Train main models in parallel
    print("\n--- Phase 1: Training main models ---")
    main_handles = {}
    for L in Ls:
        h = train_main_model.spawn(
            v=v, s=s, depth=L, m=m, n_tokens=n_tokens,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        )
        main_handles[L] = h

    main_results = {}
    for L, h in main_handles.items():
        main_results[L] = h.get()
        info = main_results[L]
        print(f"  L={L}: val={info['final_val_loss']:.4f}, "
              f"block={info['block_params']:,} params")

    # Phase 2: Train FMs in parallel
    print("\n--- Phase 2: Training architecture-matched FMs ---")
    fm_handles = []
    for L in Ls:
        for label, cfg in fm_configs:
            h = train_matched_fm.spawn(
                v=v, s=s, depth=L, m=m, n_tokens=n_tokens,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                main_model_dir=main_results[L]["save_dir"],
                **cfg,
            )
            fm_handles.append((L, label, h))

    results = []
    for L, label, h in fm_handles:
        r = h.get()
        r["_capacity_label"] = label
        r["_main_val_loss"] = main_results[r["L"]]["final_val_loss"]
        results.append(r)

    # Phase 3: Summary
    print(f"\n{'='*130}")
    print(f"ARCHITECTURE-MATCHED FM SWEEP (v={v}, s={s}, m={m})")
    print(f"All FMs: 1L/{n_head}H — same head count as main model block")
    print(f"Main model frozen before FM training")
    print(f"{'='*130}")
    hdr = (f"{'Setting':<16} {'FM_cap':>6} {'FM_params':>10} {'blk%':>6} "
           f"{'eff_rank':>9} {'rank%':>6} {'top1_var':>9} "
           f"{'cosine':>7} {'res_norm':>9} {'tgt_norm':>9} {'rel_res':>8} "
           f"{'val_loss':>9}")
    print(hdr)
    print("-" * 130)

    for r in sorted(results, key=lambda x: (x["L"], x["forward_model"]["n_params"])):
        pca = r["analysis"]["residual_pca"]
        q = r["analysis"]["basic_quality"]
        fm = r["forward_model"]
        print(f"{r['setting']:<16} {r['_capacity_label']:>6} {fm['n_params']:>10,} "
              f"{fm['block_fraction']*100:>5.1f}% "
              f"{pca['effective_rank']:>9.1f} "
              f"{pca['effective_rank']/pca['max_rank']*100:>5.1f}% "
              f"{pca['top1_pc_variance']:>9.4f} "
              f"{q['mean_cosine_sim']:>7.4f} "
              f"{q['mean_residual_norm']:>9.4f} "
              f"{q.get('mean_target_norm', 0):>9.2f} "
              f"{q.get('relative_residual', 0):>8.4f} "
              f"{r['_main_val_loss']:>9.4f}")

    print(f"\n{'='*130}")

    print("\nFor comparison, original mismatched-architecture capacity sweep (co-trained):")
    print("  L4_m2 10% (1H): rank=98.4, res_norm=1.256, cos=0.991")
    print("  L4_m2 25% (4H): rank=99.6, res_norm=0.088, cos=1.000")
    print("  L4_m2 50% (2L): rank=110.1, res_norm=0.147, cos=1.000")
    print("  L4_m2 100% (4L): rank=110.6, res_norm=0.131, cos=1.000")

    save_dir = f"{DATA_DIR}/rhm_arch_matched_sweep"
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "experiment": "architecture_matched_fm_sweep",
        "v": v, "s": s, "m": m,
        "main_model": {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embd},
        "results": [{
            "setting": r["setting"], "L": r["L"],
            "capacity_label": r["_capacity_label"],
            "fm_params": r["forward_model"]["n_params"],
            "block_fraction": r["forward_model"]["block_fraction"],
            "d_head": r["forward_model"]["d_head"],
            "mlp_mult": r["forward_model"]["mlp_mult"],
            "effective_rank": r["analysis"]["residual_pca"]["effective_rank"],
            "max_rank": r["analysis"]["residual_pca"]["max_rank"],
            "top1_pc_variance": r["analysis"]["residual_pca"]["top1_pc_variance"],
            "cosine_sim": r["analysis"]["basic_quality"]["mean_cosine_sim"],
            "residual_norm": r["analysis"]["basic_quality"]["mean_residual_norm"],
            "target_norm": r["analysis"]["basic_quality"].get("mean_target_norm"),
            "relative_residual": r["analysis"]["basic_quality"].get("relative_residual"),
            "main_val_loss": r["_main_val_loss"],
        } for r in sorted(results, key=lambda x: (x["L"], x["forward_model"]["n_params"]))],
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return summary


@app.local_entrypoint()
def main(
    v: int = 8, s: int = 2,
    l_values: str = "4,6,8", m_values: str = "2,4,8",
    n_tokens: int = 5_000_000,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
    fwd_n_layer: int = 1, fwd_d_head: int = 32, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 2.0,
):
    result = rhm_residual_rank_sweep.remote(
        v=v, s=s, l_values=l_values, m_values=m_values,
        n_tokens=n_tokens,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head,
        fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
    )
    print("\nFinal summary:")
    for name, s_data in result["settings"]:
        print(f"  {name}: eff_rank={s_data['effective_rank']:.1f}/{s_data['max_rank']} "
              f"({s_data['effective_rank']/s_data['max_rank']*100:.1f}%) "
              f"top1={s_data['top1_pc_variance']:.4f} "
              f"cos={s_data['cosine_sim']:.4f}")


def _compute_effective_rank(residual_list, d_model, max_samples=20000):
    import torch
    import numpy as np

    flat = torch.cat(residual_list, dim=0).reshape(-1, d_model).numpy()
    n = min(max_samples, flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(flat.shape[0], n, replace=False)
    sub = flat[idx]
    sub = sub - sub.mean(axis=0)
    _, S, _ = np.linalg.svd(sub, full_matrices=False)
    S_norm = S / S.sum()
    return float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))


def _run_residual_analysis(model, fwd_model, val_data, get_batch,
                           predict_from, predict_to,
                           n_batches, n_embd, device):
    """Residual PCA and quality metrics."""
    import torch
    import torch.nn.functional as F
    import numpy as np

    all_res_norms = []
    all_cos_sims = []
    all_residuals = []

    model.eval()
    fwd_model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            vx, vy = get_batch(val_data)
            _, _, vi = model(vx, vy, return_intermediates=True)

            src = vi[predict_from]
            tgt = vi[predict_to]
            pred = fwd_model(src)

            residual = tgt - pred
            all_res_norms.append(residual.norm(dim=-1).cpu())
            all_cos_sims.append(F.cosine_similarity(pred, tgt, dim=-1).cpu())
            all_residuals.append(residual.cpu())

    res_norms = torch.cat(all_res_norms, dim=0).numpy()
    cos_sims = torch.cat(all_cos_sims, dim=0).numpy()
    residuals = torch.cat(all_residuals, dim=0)

    basic = {
        "mean_residual_norm": float(res_norms.mean()),
        "std_residual_norm": float(res_norms.std()),
        "mean_cosine_sim": float(cos_sims.mean()),
    }

    # --- Residual PCA ---
    res_flat = residuals.reshape(-1, n_embd).numpy()
    subsample = min(50000, res_flat.shape[0])
    rng = np.random.RandomState(42)
    idx = rng.choice(res_flat.shape[0], subsample, replace=False)
    res_sub = res_flat[idx] - res_flat[idx].mean(axis=0)
    _, S, _ = np.linalg.svd(res_sub, full_matrices=False)

    S_sq = S ** 2
    total_var = S_sq.sum()
    cumvar = np.cumsum(S_sq) / total_var
    S_norm = S / S.sum()
    eff_rank = float(np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30))))

    pca = {
        "effective_rank": eff_rank,
        "max_rank": n_embd,
        "rank_for_50pct_variance": int(np.searchsorted(cumvar, 0.50) + 1),
        "rank_for_75pct_variance": int(np.searchsorted(cumvar, 0.75) + 1),
        "rank_for_90pct_variance": int(np.searchsorted(cumvar, 0.90) + 1),
        "rank_for_95pct_variance": int(np.searchsorted(cumvar, 0.95) + 1),
        "top1_pc_variance": float(S_sq[0] / total_var),
        "top5_pc_variance": float(S_sq[:5].sum() / total_var),
        "top10_pc_variance": float(S_sq[:10].sum() / total_var),
        "sv_spectrum_normalized": (S / S[0]).tolist(),
    }

    return {
        "basic_quality": basic,
        "residual_pca": pca,
    }
