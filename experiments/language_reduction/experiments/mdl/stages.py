from language_reduction.shared import app, volume, DATA_DIR, image


# ---------------------------------------------------------------------------
# Stage 15: Paired fine-tuning comparison — queen vs random curriculum
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def mdl_paired_comparison(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    lr: float = 3e-4,
    n_epochs: int = 5,
    block_size: int = 128,
    batch_size: int = 64,
):
    """Paired fine-tuning comparison: queen curriculum vs random curriculum.

    Both curricula have the same token count. Both models start from the
    same τ=0.3 checkpoint and train with identical hyperparameters.
    Then we compare loss on king-containing contexts.

    If queen specifically helps king (MDL decrease), queen-ft should beat
    random-ft on king contexts even though both beat the start model.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.curriculum import TARGET_WORDS, build_token_lookup
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens_dir = f"{DATA_DIR}/tokens"
    base_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    ft_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"

    enc = tiktoken.get_encoding("gpt2")

    # Build target token ID set (queen, empress, princess, etc.)
    lookup = build_token_lookup()
    target_token_ids = set()
    for word in TARGET_WORDS:
        if word in lookup:
            target_token_ids.update(lookup[word])

    king_ids = set()
    for tid in range(50257):
        raw = enc.decode([tid])
        if "king" in raw.strip().lower():
            king_ids.add(tid)

    # ---------------------------------------------------------------
    # Step 1: Extract random curriculum (same size, no target words)
    # ---------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Extract random curriculum")
    print("=" * 70)

    queen_curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    target_n_tokens = len(queen_curriculum)
    print(f"Queen curriculum: {target_n_tokens:,} tokens")
    print(f"Matching random curriculum to same size")

    rng = np.random.RandomState(42)
    random_windows = []
    total_random_tokens = 0
    window_size = block_size

    for shard_idx in range(10):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        shard = np.load(shard_path).astype(np.int64)

        # Sample random starting positions
        n_possible = len(shard) - window_size
        if n_possible < 1:
            continue
        n_needed = (target_n_tokens - total_random_tokens) // window_size + 1
        starts = rng.choice(n_possible, min(n_needed * 3, n_possible), replace=False)

        for start in starts:
            if total_random_tokens >= target_n_tokens:
                break
            window = shard[start : start + window_size]
            if len(window) < window_size:
                continue
            # Skip windows containing any target word
            if any(int(t) in target_token_ids for t in window):
                continue
            random_windows.append(window)
            total_random_tokens += window_size

        if total_random_tokens >= target_n_tokens:
            break

    random_curriculum = np.concatenate(random_windows).astype(np.int64)
    # Trim to exact same size as queen curriculum
    random_curriculum = random_curriculum[:target_n_tokens]
    print(f"Random curriculum: {len(random_curriculum):,} tokens")

    # Count king occurrences in each curriculum
    queen_king_count = sum(1 for t in queen_curriculum if int(t) in king_ids)
    random_king_count = sum(1 for t in random_curriculum if int(t) in king_ids)
    print(f"King occurrences — queen curriculum: {queen_king_count}, random: {random_king_count}")

    # ---------------------------------------------------------------
    # Step 2: Fine-tune on random curriculum
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Fine-tune on random curriculum")
    print("=" * 70)

    with open(os.path.join(base_dir, "results.json")) as f:
        cfg = json.load(f)
    sd_start = torch.load(os.path.join(base_dir, "model.pt"), map_location="cpu")
    vocab_size = sd_start["transformer.wte.weight"].shape[0]

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
    ).to(device)
    model.load_state_dict(sd_start)

    random_data = torch.from_numpy(random_curriculum)
    split = int(0.9 * len(random_data))
    train_data = random_data[:split]
    val_data = random_data[split:]

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

        if step % 100 == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                n_eval = min(5, max(1, len(val_data) // (batch_size * block_size)))
                vl_accum = 0.0
                for _ in range(n_eval):
                    vx, vy = get_batch(val_data, batch_size, block_size)
                    _, vl = model(vx, vy)
                    vl_accum += float(vl)
                val_loss = vl_accum / n_eval
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:4d}/{n_steps}: train={loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

    # Save random-ft model
    random_ft_dir = f"{DATA_DIR}/models/finetune_random/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    os.makedirs(random_ft_dir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(),
               os.path.join(random_ft_dir, "model.pt"))
    with open(os.path.join(random_ft_dir, "results.json"), "w") as f:
        json.dump({
            "source_tau": source_tau, "lr": lr, "n_epochs": n_epochs,
            "n_steps": n_steps, "best_val_loss": best_val_loss,
            "curriculum": "random", "n_tokens": len(random_curriculum),
            "n_layer": cfg.get("n_layer", 2),
            "n_head": cfg.get("n_head", 4),
            "n_embd": cfg.get("n_embd", 128),
        }, f, indent=2)
    volume.commit()
    print(f"Saved random-ft model to {random_ft_dir}")

    # ---------------------------------------------------------------
    # Step 3: Extract eval contexts
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Extract evaluation contexts")
    print("=" * 70)

    eval_window = block_size + 1

    # King-only contexts (no target words)
    king_windows = []
    half = eval_window // 2
    for shard_idx in range(5):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        for i in range(half, len(tokens) - half):
            if int(tokens[i]) in king_ids:
                window = tokens[i - half : i - half + eval_window]
                if len(window) < eval_window:
                    continue
                if any(int(t) in target_token_ids for t in window):
                    continue
                king_windows.append(window)
                if len(king_windows) >= 500:
                    break
        if len(king_windows) >= 500:
            break

    # General contexts (no king, no target words)
    general_windows = []
    for shard_idx in range(2):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        for i in range(0, len(tokens) - eval_window, eval_window * 5):
            window = tokens[i : i + eval_window]
            if len(window) < eval_window:
                continue
            if any(int(t) in king_ids for t in window):
                continue
            if any(int(t) in target_token_ids for t in window):
                continue
            general_windows.append(window)
            if len(general_windows) >= 500:
                break
        if len(general_windows) >= 500:
            break

    print(f"King-only windows: {len(king_windows)}")
    print(f"General windows: {len(general_windows)}")

    king_data = torch.from_numpy(np.stack(king_windows))
    general_data = torch.from_numpy(np.stack(general_windows))

    # ---------------------------------------------------------------
    # Step 4: Evaluate all models
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Evaluate all models")
    print("=" * 70)

    model_dirs = {
        "start": base_dir,
        "queen_ft": ft_dir,
        "random_ft": random_ft_dir,
        "ground_truth": gt_dir,
    }

    def eval_loss(data_tensor, model_dir):
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        with open(os.path.join(model_dir, "results.json")) as f:
            c = json.load(f)
        m = GPT(
            vocab_size=sd["transformer.wte.weight"].shape[0],
            block_size=block_size,
            n_layer=c.get("n_layer", 2),
            n_head=c.get("n_head", 4),
            n_embd=c.get("n_embd", 128),
        ).to(device)
        m.load_state_dict(sd)
        m.eval()

        total = 0.0
        count = 0
        with torch.no_grad():
            for s in range(0, len(data_tensor), batch_size):
                batch = data_tensor[s:s+batch_size].to(device)
                x = batch[:, :-1].contiguous()
                y = batch[:, 1:].contiguous()
                _, loss = m(x, y)
                total += float(loss) * len(batch)
                count += len(batch)
        return total / count

    results = {}
    for label, mdir in model_dirs.items():
        king_loss = eval_loss(king_data, mdir)
        gen_loss = eval_loss(general_data, mdir)
        results[label] = {"king": king_loss, "general": gen_loss}
        print(f"  {label:>12s}:  king={king_loss:.4f}  general={gen_loss:.4f}")

    # ---------------------------------------------------------------
    # Step 5: Analysis
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS: Does queen specifically help king?")
    print("=" * 70)

    s = results["start"]
    q = results["queen_ft"]
    r = results["random_ft"]

    print(f"\n  King-context loss:")
    print(f"    start:     {s['king']:.4f}")
    print(f"    queen_ft:  {q['king']:.4f}  (Δ from start: {q['king']-s['king']:+.4f})")
    print(f"    random_ft: {r['king']:.4f}  (Δ from start: {r['king']-s['king']:+.4f})")
    print(f"    queen advantage over random: {r['king']-q['king']:+.4f}")

    print(f"\n  General-context loss:")
    print(f"    start:     {s['general']:.4f}")
    print(f"    queen_ft:  {q['general']:.4f}  (Δ from start: {q['general']-s['general']:+.4f})")
    print(f"    random_ft: {r['general']:.4f}  (Δ from start: {r['general']-s['general']:+.4f})")
    print(f"    queen advantage over random: {r['general']-q['general']:+.4f}")

    # The MDL signal: queen's excess improvement on king contexts vs general
    queen_king_delta = q["king"] - s["king"]
    queen_gen_delta = q["general"] - s["general"]
    random_king_delta = r["king"] - s["king"]
    random_gen_delta = r["general"] - s["general"]

    queen_king_excess = queen_king_delta - queen_gen_delta
    random_king_excess = random_king_delta - random_gen_delta

    print(f"\n  King-specific excess improvement (king Δ minus general Δ):")
    print(f"    queen_ft:  {queen_king_excess:+.4f}")
    print(f"    random_ft: {random_king_excess:+.4f}")
    print(f"    difference: {queen_king_excess - random_king_excess:+.4f}")

    if queen_king_excess < random_king_excess:
        print("\n  → Queen curriculum gives LARGER king-specific improvement")
        print("  → Evidence for MDL decrease: queen provides explanatory power for king")
    else:
        print("\n  → Random curriculum gives equal/larger king-specific improvement")
        print("  → No evidence that queen specifically reduces king's MDL")

    # Save
    save_results = {
        "losses": results,
        "queen_curriculum_tokens": int(target_n_tokens),
        "random_curriculum_tokens": int(len(random_curriculum)),
        "king_in_queen_curriculum": int(queen_king_count),
        "king_in_random_curriculum": int(random_king_count),
        "n_king_windows": len(king_windows),
        "n_general_windows": len(general_windows),
        "training": {"lr": lr, "n_epochs": n_epochs, "n_steps": n_steps,
                     "batch_size": batch_size, "block_size": block_size},
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/mdl_paired.json", "w") as f:
        json.dump(save_results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {results_dir}/mdl_paired.json")
    return save_results


# ---------------------------------------------------------------------------
# Stage 16: Two-phase consolidation — does queen help king during general training?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=7200,
    memory=32768,
)
def mdl_consolidation(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    lr: float = 3e-4,
    n_epochs: int = 5,
    block_size: int = 128,
    batch_size: int = 64,
    consolidation_tokens: int = 200_000,
):
    """Two-phase consolidation test for MDL.

    Phase 1 already happened: queen-ft and random-ft exist from prior stages.
    Phase 2 (this stage): train both on the SAME general curriculum (no target
    words, no queen) with matched hyperparameters. Then compare king-context
    loss.

    If queen provides a structural prior that makes king cheaper to maintain,
    queen-ft should learn king-related patterns more efficiently during phase 2.
    We also evaluate at multiple checkpoints during phase 2 to see if the
    consolidation effect emerges over time.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.curriculum import TARGET_WORDS, build_token_lookup
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens_dir = f"{DATA_DIR}/tokens"
    base_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    queen_ft_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    random_ft_dir = f"{DATA_DIR}/models/finetune_random/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"

    enc = tiktoken.get_encoding("gpt2")
    lookup = build_token_lookup()

    target_token_ids = set()
    for word in TARGET_WORDS:
        if word in lookup:
            target_token_ids.update(lookup[word])

    king_ids = set()
    for tid in range(50257):
        raw = enc.decode([tid])
        if "king" in raw.strip().lower():
            king_ids.add(tid)

    # ---------------------------------------------------------------
    # Step 1: Extract general consolidation curriculum (no target words)
    # ---------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Extract general consolidation curriculum")
    print("=" * 70)

    rng = np.random.RandomState(99)
    consol_windows = []
    total_tokens = 0

    for shard_idx in range(20):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        shard = np.load(shard_path).astype(np.int64)

        n_possible = len(shard) - block_size
        if n_possible < 1:
            continue
        n_needed = (consolidation_tokens - total_tokens) // block_size + 1
        starts = rng.choice(n_possible, min(n_needed * 3, n_possible), replace=False)

        for start in starts:
            if total_tokens >= consolidation_tokens:
                break
            window = shard[start : start + block_size]
            if len(window) < block_size:
                continue
            if any(int(t) in target_token_ids for t in window):
                continue
            consol_windows.append(window)
            total_tokens += block_size

        if total_tokens >= consolidation_tokens:
            break

    consol_data = torch.from_numpy(
        np.concatenate(consol_windows).astype(np.int64)
    )[:consolidation_tokens]
    king_count = sum(1 for t in consol_data.numpy() if int(t) in king_ids)
    print(f"Consolidation curriculum: {len(consol_data):,} tokens")
    print(f"King occurrences: {king_count}")

    split = int(0.9 * len(consol_data))
    train_data = consol_data[:split]
    val_data = consol_data[split:]

    # ---------------------------------------------------------------
    # Step 2: Extract eval contexts (same as paired comparison)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Extract evaluation contexts")
    print("=" * 70)

    eval_window = block_size + 1

    king_windows = []
    half = eval_window // 2
    for shard_idx in range(5):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        for i in range(half, len(tokens) - half):
            if int(tokens[i]) in king_ids:
                window = tokens[i - half : i - half + eval_window]
                if len(window) < eval_window:
                    continue
                if any(int(t) in target_token_ids for t in window):
                    continue
                king_windows.append(window)
                if len(king_windows) >= 500:
                    break
        if len(king_windows) >= 500:
            break

    general_windows = []
    for shard_idx in range(2):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        for i in range(0, len(tokens) - eval_window, eval_window * 5):
            window = tokens[i : i + eval_window]
            if len(window) < eval_window:
                continue
            if any(int(t) in king_ids for t in window):
                continue
            if any(int(t) in target_token_ids for t in window):
                continue
            general_windows.append(window)
            if len(general_windows) >= 500:
                break
        if len(general_windows) >= 500:
            break

    print(f"King-only eval windows: {len(king_windows)}")
    print(f"General eval windows: {len(general_windows)}")

    king_eval = torch.from_numpy(np.stack(king_windows))
    general_eval = torch.from_numpy(np.stack(general_windows))

    # ---------------------------------------------------------------
    # Step 3: Evaluate loss helper
    # ---------------------------------------------------------------
    def eval_loss(model, data_tensor):
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for s in range(0, len(data_tensor), batch_size):
                batch = data_tensor[s:s+batch_size].to(device)
                x = batch[:, :-1].contiguous()
                y = batch[:, 1:].contiguous()
                _, loss = model(x, y)
                total += float(loss) * len(batch)
                count += len(batch)
        return total / count

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i+bl] for i in ix])
        y = torch.stack([data[i+1:i+bl+1] for i in ix])
        return x.to(device), y.to(device)

    # ---------------------------------------------------------------
    # Step 4: Phase 2 training for both models with periodic eval
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Phase 2 consolidation training")
    print("=" * 70)

    with open(os.path.join(queen_ft_dir, "results.json")) as f:
        cfg = json.load(f)

    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    n_steps = n_epochs * steps_per_epoch
    eval_every = max(1, n_steps // 10)

    print(f"Training: {n_steps} steps ({n_epochs} epochs)")
    print(f"Evaluating every {eval_every} steps")

    models_to_train = {
        "queen_ft": queen_ft_dir,
        "random_ft": random_ft_dir,
    }

    all_curves = {}

    for label, model_dir in models_to_train.items():
        print(f"\n--- Training {label} on consolidation curriculum ---")

        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        model = GPT(
            vocab_size=sd["transformer.wte.weight"].shape[0],
            block_size=block_size,
            n_layer=cfg.get("n_layer", 2),
            n_head=cfg.get("n_head", 4),
            n_embd=cfg.get("n_embd", 128),
        ).to(device)
        model.load_state_dict(sd)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        curve = []

        # Eval before any consolidation training
        king_loss = eval_loss(model, king_eval)
        gen_loss = eval_loss(model, general_eval)
        curve.append({"step": 0, "king_loss": king_loss, "general_loss": gen_loss})
        print(f"  step 0 (pre-consolidation): king={king_loss:.4f} general={gen_loss:.4f}")

        for step in range(1, n_steps + 1):
            model.train()
            x, y = get_batch(train_data, batch_size, block_size)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % eval_every == 0 or step == n_steps:
                king_loss = eval_loss(model, king_eval)
                gen_loss = eval_loss(model, general_eval)
                curve.append({"step": step, "king_loss": king_loss, "general_loss": gen_loss})
                print(f"  step {step:4d}/{n_steps}: king={king_loss:.4f} general={gen_loss:.4f}")

        all_curves[label] = curve

    # ---------------------------------------------------------------
    # Step 5: Analysis
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS: Consolidation effect")
    print("=" * 70)

    qc = all_curves["queen_ft"]
    rc = all_curves["random_ft"]

    print(f"\n  {'step':>6s}  {'q_king':>8s}  {'r_king':>8s}  {'q-r king':>9s}  "
          f"{'q_gen':>8s}  {'r_gen':>8s}  {'q-r gen':>8s}  {'q king excess':>14s}")
    print("-" * 85)

    for qi, ri in zip(qc, rc):
        step = qi["step"]
        q_king = qi["king_loss"]
        r_king = ri["king_loss"]
        q_gen = qi["general_loss"]
        r_gen = ri["general_loss"]
        diff_king = q_king - r_king
        diff_gen = q_gen - r_gen
        # queen's king-specific advantage: how much better is queen on king
        # relative to how it does on general, compared to random
        queen_king_excess = diff_king - diff_gen
        print(f"  {step:>6d}  {q_king:8.4f}  {r_king:8.4f}  {diff_king:>+9.4f}  "
              f"{q_gen:8.4f}  {r_gen:8.4f}  {diff_gen:>+8.4f}  {queen_king_excess:>+14.4f}")

    # Final comparison
    q_final = qc[-1]
    r_final = rc[-1]
    q_init = qc[0]
    r_init = rc[0]

    q_king_improve = q_init["king_loss"] - q_final["king_loss"]
    r_king_improve = r_init["king_loss"] - r_final["king_loss"]
    q_gen_improve = q_init["general_loss"] - q_final["general_loss"]
    r_gen_improve = r_init["general_loss"] - r_final["general_loss"]

    print(f"\n  Total improvement during consolidation:")
    print(f"    queen_ft — king: {q_king_improve:+.4f}, general: {q_gen_improve:+.4f}")
    print(f"    random_ft — king: {r_king_improve:+.4f}, general: {r_gen_improve:+.4f}")

    q_king_excess_rate = (q_king_improve - q_gen_improve)
    r_king_excess_rate = (r_king_improve - r_gen_improve)
    print(f"\n  King-specific excess improvement during consolidation:")
    print(f"    queen_ft:  {q_king_excess_rate:+.4f}")
    print(f"    random_ft: {r_king_excess_rate:+.4f}")
    print(f"    difference (queen - random): {q_king_excess_rate - r_king_excess_rate:+.4f}")

    if q_king_excess_rate > r_king_excess_rate:
        print("\n  → Queen-ft consolidated king MORE efficiently during general training")
        print("  → Evidence for structural prior / MDL decrease")
    else:
        print("\n  → No consolidation advantage for queen-ft on king contexts")

    # Save
    save_results = {
        "curves": all_curves,
        "consolidation_tokens": int(len(consol_data)),
        "king_in_consolidation": king_count,
        "n_king_windows": len(king_windows),
        "n_general_windows": len(general_windows),
        "training": {"lr": lr, "n_epochs": n_epochs, "n_steps": n_steps,
                     "batch_size": batch_size, "block_size": block_size,
                     "consolidation_tokens": consolidation_tokens},
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/mdl_consolidation.json", "w") as f:
        json.dump(save_results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {results_dir}/mdl_consolidation.json")
    return save_results
