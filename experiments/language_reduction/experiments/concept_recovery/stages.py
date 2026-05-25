from language_reduction.shared import (
    app, volume, DATA_DIR, image, NumpyEncoder, resolve_mode_key,
)


# ---------------------------------------------------------------------------
# Stage 11: Build curriculum for concept recovery
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=16384,
)
def build_curriculum_stage(target_tokens: int = 50_000, max_shards: int = 10):
    """Extract curriculum from original corpus."""
    from language_reduction.curriculum import (
        CurriculumConfig, extract_curriculum, save_curriculum,
    )

    config = CurriculumConfig(
        target_tokens=target_tokens,
        max_shards=max_shards,
    )
    tokens = extract_curriculum(f"{DATA_DIR}/tokens", config)
    save_curriculum(tokens, f"{DATA_DIR}/curriculum", config)
    volume.commit()


# ---------------------------------------------------------------------------
# Stage 12: Fine-tune model on curriculum
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def finetune_model(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    lr: float = 1e-4,
    n_epochs: int = 5,
    eval_interval: int = 100,
    block_size: int = 128,
    batch_size: int = 64,
    mode: str = "spectral",
):
    """Fine-tune a pre-trained model on the extracted curriculum."""
    import os, json, torch
    import numpy as np
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode_key = resolve_mode_key(mode, source_tau)

    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    n_curriculum = len(curriculum)
    print(f"Curriculum: {n_curriculum:,} tokens (mode={mode})")

    source_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    with open(os.path.join(source_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(source_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
    ).to(device)
    model.load_state_dict(sd)
    print(f"Loaded checkpoint from {source_dir}")

    split = int(0.9 * n_curriculum)
    train_data = curriculum[:split]
    val_data = curriculum[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    n_steps = n_epochs * steps_per_epoch
    print(f"Training: {n_steps} steps ({n_epochs} epochs, {steps_per_epoch} steps/epoch)")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i+bl] for i in ix])
        y = torch.stack([data[i+1:i+bl+1] for i in ix])
        return x.to(device), y.to(device)

    losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None

    for step in range(n_steps):
        model.train()
        x, y = get_batch(train_data, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vl_accum = 0.0
                n_eval = min(5, max(1, len(val_data) // (batch_size * block_size)))
                for _ in range(n_eval):
                    vx, vy = get_batch(val_data, batch_size, block_size)
                    _, vl = model(vx, vy)
                    vl_accum += float(vl)
                val_loss = vl_accum / n_eval
            losses.append((step, float(loss)))
            val_losses.append((step, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:5d}/{n_steps}: train={loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

    save_dir = f"{DATA_DIR}/models/finetune/{mode_key}/P_{source_P}/T_{block_size}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(),
               os.path.join(save_dir, "model.pt"))

    result = {
        "mode": mode,
        "source_tau": source_tau,
        "source_P": source_P,
        "curriculum_tokens": n_curriculum,
        "lr": lr,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "best_val_loss": best_val_loss,
        "final_val_loss": float(val_losses[-1][1]),
        "train_curve": losses,
        "val_curve": val_losses,
        "n_layer": cfg.get("n_layer", 2),
        "n_head": cfg.get("n_head", 4),
        "n_embd": cfg.get("n_embd", 128),
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"Saved to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage 13: Recovery evaluation
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def recovery_eval_stage(source_tau: float = 0.3, source_P: int = 100_000_000,
                        mode: str = "spectral"):
    """Evaluate whether fine-tuning caused structural integration."""
    import os, json, torch
    import numpy as np
    from language_reduction.eval_embeddings import build_token_index, run_eval
    from language_reduction.recovery_eval import (
        embedding_shift_toward_gt,
        neighbor_overlap,
        format_recovery_report,
    )

    block_size = 128
    mode_key = resolve_mode_key(mode, source_tau)
    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    def load_emb(model_dir):
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        return sd["transformer.wte.weight"].numpy()

    emb_start = load_emb(f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}")
    emb_gt = load_emb(f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}")
    emb_ft = load_emb(f"{DATA_DIR}/models/finetune/{mode_key}/P_{source_P}/T_{block_size}")

    eval_start = run_eval(emb_start, token_index)
    eval_gt = run_eval(emb_gt, token_index)
    eval_ft = run_eval(emb_ft, token_index)

    shifts = embedding_shift_toward_gt(emb_ft, emb_start, emb_gt, token_index)
    overlaps = neighbor_overlap(emb_ft, emb_gt, emb_start, top_ids, token_index)

    report = format_recovery_report(shifts, overlaps, eval_start, eval_ft, eval_gt)
    print(report)

    results = {
        "shifts": shifts,
        "overlaps": overlaps,
        "eval_start": eval_start,
        "eval_ft": eval_ft,
        "eval_gt": eval_gt,
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/recovery_eval.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/recovery_eval.json")
    return results


# ---------------------------------------------------------------------------
# Stage 14: MDL analysis — did queen reduce king's description length?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def recovery_mdl_stage(source_tau: float = 0.3, source_P: int = 100_000_000):
    """MDL-oriented analysis of concept recovery.

    Two tests:
    1. Embedding-level: neighborhood entropy, tightness, coherence, local rank
       for gender words vs controls across start/fine-tuned/ground-truth.
    2. Model-level: cross-entropy loss on king-containing contexts (that don't
       contain target words) — the direct MDL test.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.eval_embeddings import build_token_index
    from language_reduction.recovery_eval import (
        mdl_analysis, format_mdl_report, GENDER_WORDS,
    )
    from language_reduction.curriculum import TARGET_WORDS
    from language_reduction.model import GPT

    block_size = 128
    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)
    s2i = token_index["str_to_id"]

    def load_emb(model_dir):
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        return sd["transformer.wte.weight"].numpy()

    base_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"
    ft_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"

    emb_start = load_emb(base_dir)
    emb_gt = load_emb(gt_dir)
    emb_ft = load_emb(ft_dir)

    # --- Part 1: Embedding MDL proxies ---
    print("=" * 70)
    print("PART 1: EMBEDDING MDL PROXIES")
    print("=" * 70)

    mdl = mdl_analysis(emb_start, emb_ft, emb_gt, top_ids, token_index)
    report = format_mdl_report(mdl)
    print(report)

    # --- Part 2: Model loss on king-containing contexts ---
    print("\n" + "=" * 70)
    print("PART 2: MODEL LOSS ON KING CONTEXTS")
    print("=" * 70)

    # Build set of target token IDs to exclude
    enc = tiktoken.get_encoding("gpt2")
    target_token_ids = set()
    for w in TARGET_WORDS:
        for tid in range(50257):
            raw = enc.decode([tid])
            if w in raw.strip().lower():
                target_token_ids.add(tid)

    king_ids = set()
    for tid in range(50257):
        raw = enc.decode([tid])
        if "king" in raw.strip().lower():
            king_ids.add(tid)

    # Extract king-only windows from original corpus (no target words)
    tokens_dir = f"{DATA_DIR}/tokens"
    max_shards = 5

    king_windows = []
    window_size = block_size + 1  # need targets too
    half = window_size // 2

    for shard_idx in range(max_shards):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        for i in range(half, len(tokens) - half):
            if int(tokens[i]) in king_ids:
                window = tokens[i - half : i - half + window_size]
                if len(window) < window_size:
                    continue
                # Skip if any target word token appears
                if any(int(t) in target_token_ids for t in window):
                    continue
                king_windows.append(window)
                if len(king_windows) >= 500:
                    break
        if len(king_windows) >= 500:
            break

    print(f"\nExtracted {len(king_windows)} king-only windows (no target words)")

    if len(king_windows) < 10:
        print("Too few king windows found — skipping loss test")
        loss_results = {"error": "too few windows"}
    else:
        king_data = torch.from_numpy(np.stack(king_windows))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss_results = {}
        for label, model_dir in [("start", base_dir), ("fine_tuned", ft_dir), ("ground_truth", gt_dir)]:
            sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
            with open(os.path.join(model_dir, "results.json")) as f:
                cfg = json.load(f)

            model = GPT(
                vocab_size=sd["transformer.wte.weight"].shape[0],
                block_size=block_size,
                n_layer=cfg.get("n_layer", 2),
                n_head=cfg.get("n_head", 4),
                n_embd=cfg.get("n_embd", 128),
            ).to(device)
            model.load_state_dict(sd)
            model.eval()

            # Evaluate in batches
            total_loss = 0.0
            n_batches = 0
            batch_size = 64
            with torch.no_grad():
                for start in range(0, len(king_data), batch_size):
                    batch = king_data[start:start+batch_size].to(device)
                    x = batch[:, :-1].contiguous()
                    y = batch[:, 1:].contiguous()
                    _, loss = model(x, y)
                    total_loss += float(loss) * len(batch)
                    n_batches += len(batch)

            mean_loss = total_loss / n_batches
            loss_results[label] = mean_loss
            print(f"  {label:>12s}: loss = {mean_loss:.4f}")

        if "start" in loss_results and "fine_tuned" in loss_results:
            delta = loss_results["fine_tuned"] - loss_results["start"]
            print(f"\n  Δ loss (ft - start) = {delta:+.4f}")
            if delta < 0:
                print("  → Fine-tuning REDUCED loss on king contexts = MDL DECREASED")
            else:
                print("  → Fine-tuning INCREASED loss on king contexts = MDL INCREASED")

    # --- Part 3: Loss on general held-out text (control) ---
    print("\n--- CONTROL: loss on general (non-king, non-target) text ---")

    general_windows = []
    for shard_idx in range(2):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        for i in range(0, len(tokens) - window_size, window_size * 5):
            window = tokens[i : i + window_size]
            if len(window) < window_size:
                continue
            has_king = any(int(t) in king_ids for t in window)
            has_target = any(int(t) in target_token_ids for t in window)
            if has_king or has_target:
                continue
            general_windows.append(window)
            if len(general_windows) >= 500:
                break
        if len(general_windows) >= 500:
            break

    print(f"Extracted {len(general_windows)} general windows")

    if len(general_windows) >= 10:
        general_data = torch.from_numpy(np.stack(general_windows))
        general_loss = {}
        for label, model_dir in [("start", base_dir), ("fine_tuned", ft_dir)]:
            sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
            with open(os.path.join(model_dir, "results.json")) as f:
                cfg = json.load(f)
            model = GPT(
                vocab_size=sd["transformer.wte.weight"].shape[0],
                block_size=block_size,
                n_layer=cfg.get("n_layer", 2),
                n_head=cfg.get("n_head", 4),
                n_embd=cfg.get("n_embd", 128),
            ).to(device)
            model.load_state_dict(sd)
            model.eval()

            total_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for start_idx in range(0, len(general_data), batch_size):
                    batch = general_data[start_idx:start_idx+batch_size].to(device)
                    x = batch[:, :-1].contiguous()
                    y = batch[:, 1:].contiguous()
                    _, loss = model(x, y)
                    total_loss += float(loss) * len(batch)
                    n_batches += len(batch)

            general_loss[label] = total_loss / n_batches
            print(f"  {label:>12s}: loss = {general_loss[label]:.4f}")

        if "start" in general_loss and "fine_tuned" in general_loss:
            delta_general = general_loss["fine_tuned"] - general_loss["start"]
            print(f"\n  Δ loss general (ft - start) = {delta_general:+.4f}")

    # Save results
    all_results = {
        "mdl_proxies": mdl,
        "king_context_loss": loss_results,
        "general_context_loss": general_loss if len(general_windows) >= 10 else {},
        "n_king_windows": len(king_windows),
        "n_general_windows": len(general_windows),
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/recovery_mdl.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/recovery_mdl.json")
    return all_results
