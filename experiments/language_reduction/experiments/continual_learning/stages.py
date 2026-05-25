"""Continual learning benchmark via vocab-only reduction.

Phase 1: Independent single-token recovery across semantic domains at τ=0.1.
Phase 2: Sequential token recovery with degradation tracking.

Data layout on Modal volume:
  /data/continual_learning/tau_{tau}/
    round_{i}/                             — Phase 1 (independent recovery)
      curriculum.npy, curriculum_meta.json
      model.pt, results.json
      eval.json
    sequential/{ordering}/                 — Phase 2 (sequential recovery)
      round_{i}/
        curriculum.npy, curriculum_meta.json
        model.pt, results.json
        eval.json
      degradation_matrix.json
      dimension_analysis.json
"""

from language_reduction.shared import (
    app, volume, DATA_DIR, image, NumpyEncoder, resolve_mode_key,
)

CL_DIR = f"{DATA_DIR}/continual_learning"

RECOVERY_TARGETS_01 = [
    ("princess", "Prince", "royalty/gender"),
    ("grandfather", "father", "family/generational"),
    ("grandmother", "mother", "family/generational"),
    ("beijing", "China", "geography"),
    ("tokyo", "Japan", "geography"),
    ("madrid", "Spain", "geography"),
]

ORDERINGS = {
    "clustered": [
        ("beijing", "China", "geography"),
        ("tokyo", "Japan", "geography"),
        ("madrid", "Spain", "geography"),
        ("grandfather", "father", "family/generational"),
        ("grandmother", "mother", "family/generational"),
        ("princess", "Prince", "royalty/gender"),
    ],
    "interleaved": [
        ("beijing", "China", "geography"),
        ("grandfather", "father", "family/generational"),
        ("princess", "Prince", "royalty/gender"),
        ("tokyo", "Japan", "geography"),
        ("grandmother", "mother", "family/generational"),
        ("madrid", "Spain", "geography"),
    ],
}


# ---------------------------------------------------------------------------
# Stage: Build curriculum for a recovery round
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=16384,
)
def cl_build_curriculum(
    tau: float = 0.1,
    target_words_str: str = "",
    target_tokens: int = 100_000,
    max_shards: int = 10,
    round_idx: int = 0,
):
    """Build fine-tuning curriculum from original corpus for specified target words.

    Extracts windows containing target words from the un-reduced corpus.
    target_words_str is a comma-separated list of words (required).
    """
    import os, json
    import numpy as np
    import tiktoken

    if not target_words_str:
        raise ValueError("target_words_str is required (comma-separated list of words)")

    target_words = [w.strip() for w in target_words_str.split(",") if w.strip()]

    enc = tiktoken.get_encoding("gpt2")

    # Build token ID lookup for target words
    lookup = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        if cleaned in [w.lower() for w in target_words]:
            if cleaned not in lookup:
                lookup[cleaned] = set()
            lookup[cleaned].add(tid)

    target_token_ids = set()
    id_to_word = {}
    for word in target_words:
        w_lower = word.lower()
        if w_lower in lookup:
            for tid in lookup[w_lower]:
                target_token_ids.add(tid)
                id_to_word[tid] = w_lower

    print(f"Target words: {target_words}")
    print(f"Target token IDs: {len(target_token_ids)} IDs across {len(target_words)} words")

    # Discover collapse mapping at this tau
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    if os.path.exists(os.path.join(tau_stats_dir, "vocab_map.npy")):
        vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
        tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
        top_ids_set = set(int(x) for x in tau_top_ids)
        print(f"\nCollapse mapping at vocab-only tau={tau} (v'={len(tau_top_ids)}):")
        for word in target_words:
            w_lower = word.lower()
            if w_lower not in lookup:
                print(f"  {word}: not found in GPT-2 vocab")
                continue
            for tid in sorted(lookup[w_lower]):
                in_vocab = tid in top_ids_set
                if in_vocab:
                    print(f"  {word} (id={tid}): IN VOCAB (preserved)")
                else:
                    from language_reduction.denoise import build_inverse_vocab_map
                    inv = build_inverse_vocab_map(vocab_map, tau_top_ids)
                    tgt_reduced = int(vocab_map[tid])
                    tgt_orig = int(inv[tgt_reduced])
                    tgt_str = enc.decode([tgt_orig]).strip()
                    print(f"  {word} (id={tid}): COLLAPSED -> {tgt_str!r} (id={tgt_orig})")

    # Extract windows from original corpus
    window_size = 128
    windows = []
    total_tokens = 0
    per_word_counts = {w.lower(): 0 for w in target_words}
    last_window_end = -1
    n_shards_scanned = 0

    for shard_idx in range(max_shards):
        shard_path = os.path.join(f"{DATA_DIR}/tokens", f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break

        shard = np.load(shard_path)
        n_shards_scanned += 1
        last_window_end = -1

        for pos in range(len(shard)):
            if total_tokens >= target_tokens:
                break

            token_id = int(shard[pos])
            if token_id not in target_token_ids:
                continue

            window_start = max(0, pos - window_size // 2)
            window_end = min(len(shard), window_start + window_size)
            window_start = window_end - window_size
            if window_start < 0:
                window_start = 0
                window_end = min(len(shard), window_size)

            overlap = max(0, last_window_end - window_start)
            if overlap > window_size // 2:
                per_word_counts[id_to_word[token_id]] += 1
                continue

            window = shard[window_start:window_end]
            windows.append(window)
            total_tokens += len(window)
            last_window_end = window_end
            per_word_counts[id_to_word[token_id]] += 1

        if total_tokens >= target_tokens:
            break

    print(f"\nExtraction stats:")
    print(f"  Windows: {len(windows)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Shards scanned: {n_shards_scanned}")
    for word, count in sorted(per_word_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {word:>12s}: {count}")

    if total_tokens < target_tokens:
        print(f"  WARNING: Only found {total_tokens:,} tokens (target: {target_tokens:,})")

    if not windows:
        print("ERROR: No windows found")
        return {"error": "no windows found"}

    curriculum = np.concatenate(windows).astype(np.int64)

    # Save
    round_dir = f"{CL_DIR}/tau_{tau:.3f}/round_{round_idx}"
    os.makedirs(round_dir, exist_ok=True)
    np.save(os.path.join(round_dir, "curriculum.npy"), curriculum)

    meta = {
        "target_words": target_words,
        "target_tokens": target_tokens,
        "n_windows": len(windows),
        "n_tokens": len(curriculum),
        "n_shards_scanned": n_shards_scanned,
        "per_word_counts": per_word_counts,
        "round_idx": round_idx,
        "tau": tau,
    }
    with open(os.path.join(round_dir, "curriculum_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()

    print(f"Saved curriculum: {len(curriculum):,} tokens to {round_dir}")
    return meta


# ---------------------------------------------------------------------------
# Stage: Fine-tune vocab-only model on a round's curriculum
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def cl_finetune(
    tau: float = 0.1,
    lr: float = 3e-4,
    n_epochs: int = 5,
    eval_interval: int = 10,
    block_size: int = 128,
    batch_size: int = 64,
    round_idx: int = 0,
    source_round: int = -1,
    source_P: int = 100_000_000,
):
    """Fine-tune on a round's curriculum.

    source_round=-1: start from the base vocab-only model.
    source_round=N:  start from the model saved after round N (for sequential recovery).
    """
    import os, json, torch
    import numpy as np
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    round_dir = f"{CL_DIR}/tau_{tau:.3f}/round_{round_idx}"

    # Load curriculum
    curriculum = np.load(os.path.join(round_dir, "curriculum.npy"))
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    n_curriculum = len(curriculum)
    print(f"Curriculum: {n_curriculum:,} tokens")

    # Load source model
    if source_round < 0:
        mode_key = resolve_mode_key("vocab_only", tau)
        source_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    else:
        source_dir = f"{CL_DIR}/tau_{tau:.3f}/round_{source_round}"

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

    # Train/val split
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

    # Save
    os.makedirs(round_dir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(),
               os.path.join(round_dir, "model.pt"))

    result = {
        "mode": "vocab_only",
        "source_tau": tau,
        "source_P": source_P,
        "source_round": source_round,
        "round_idx": round_idx,
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
    with open(os.path.join(round_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"Saved to {round_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage: Evaluate recovery and degradation
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def cl_eval(
    tau: float = 0.1,
    round_idx: int = 0,
    source_P: int = 100_000_000,
    eval_all_rounds: bool = False,
):
    """Evaluate recovery at a given round.

    Compares the round's model against the base vocab-only model and τ=0.0 ground truth.
    If eval_all_rounds=True, also evaluates all previous rounds' models on this round's
    target words (measuring degradation from subsequent training).
    """
    import os, json, torch
    import numpy as np
    from language_reduction.eval_embeddings import build_token_index, run_eval
    from language_reduction.recovery_eval import (
        embedding_shift_toward_gt,
        neighbor_overlap,
        format_recovery_report,
    )

    block_size = 128
    mode_key = resolve_mode_key("vocab_only", tau)
    round_dir = f"{CL_DIR}/tau_{tau:.3f}/round_{round_idx}"

    # Use the vocab-only active vocabulary for neighbor evaluation
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    if os.path.exists(os.path.join(tau_stats_dir, "top_ids.npy")):
        top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
        print(f"Using vocab-only active set: {len(top_ids)} tokens")
    else:
        top_ids = np.load(f"{DATA_DIR}/stats/top_ids.npy")
        print(f"Falling back to spectral active set: {len(top_ids)} tokens")

    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    def load_emb(model_dir):
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        return sd["transformer.wte.weight"].numpy()

    # Load the three key embedding matrices
    base_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"

    emb_start = load_emb(base_dir)
    emb_gt = load_emb(gt_dir)
    emb_ft = load_emb(round_dir)

    # Run the standard eval battery
    eval_start = run_eval(emb_start, token_index)
    eval_gt = run_eval(emb_gt, token_index)
    eval_ft = run_eval(emb_ft, token_index)

    shifts = embedding_shift_toward_gt(emb_ft, emb_start, emb_gt, token_index)
    overlaps = neighbor_overlap(emb_ft, emb_gt, emb_start, top_ids, token_index)

    report = format_recovery_report(shifts, overlaps, eval_start, eval_ft, eval_gt)
    print(report)

    # Per-target-word differentiation from collapse target
    print(f"\n{'='*70}")
    print("TARGET WORD DIFFERENTIATION FROM COLLAPSE TARGET")
    print(f"{'='*70}")

    with open(os.path.join(round_dir, "curriculum_meta.json")) as f:
        curriculum_meta = json.load(f)
    target_words = curriculum_meta["target_words"]

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    tau_stats_exists = os.path.exists(os.path.join(tau_stats_dir, "vocab_map.npy"))
    differentiation = []
    if tau_stats_exists:
        vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
        tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
        from language_reduction.denoise import build_inverse_vocab_map
        inv = build_inverse_vocab_map(vocab_map, tau_top_ids)
        top_ids_set = set(int(x) for x in tau_top_ids)

        # Build word -> token ID mapping
        word_to_ids = {}
        for tid in range(50257):
            raw = enc.decode([tid])
            cleaned = raw.strip().lower()
            for w in target_words:
                if cleaned == w.lower():
                    if w not in word_to_ids:
                        word_to_ids[w] = []
                    word_to_ids[w].append(tid)

        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

        print(f"  {'word':<15s} {'collapsed_to':<15s} "
              f"{'cos_start→tgt':<16s} {'cos_ft→tgt':<14s} "
              f"{'Δ(toward gt)':<14s} {'shift mag'}")
        print("  " + "-" * 90)

        for word in target_words:
            if word not in word_to_ids:
                continue
            for tid in word_to_ids[word]:
                if tid in top_ids_set:
                    continue  # preserved, not collapsed

                tgt_reduced = int(vocab_map[tid])
                tgt_orig = int(inv[tgt_reduced])
                tgt_str = enc.decode([tgt_orig]).strip()

                cos_start_tgt = cosine(emb_start[tid], emb_start[tgt_orig])
                cos_ft_tgt = cosine(emb_ft[tid], emb_ft[tgt_orig])
                cos_start_gt = cosine(emb_start[tid], emb_gt[tid])
                cos_ft_gt = cosine(emb_ft[tid], emb_gt[tid])
                toward_gt = cos_ft_gt - cos_start_gt
                shift_mag = 1.0 - cosine(emb_ft[tid], emb_start[tid])

                differentiation.append({
                    "word": word,
                    "token_id": tid,
                    "collapse_target": tgt_str,
                    "collapse_target_id": tgt_orig,
                    "cos_start_to_target": cos_start_tgt,
                    "cos_ft_to_target": cos_ft_tgt,
                    "differentiated": cos_ft_tgt < cos_start_tgt,
                    "toward_gt": toward_gt,
                    "shift_magnitude": shift_mag,
                })

                marker = " <--" if cos_ft_tgt < cos_start_tgt else ""
                print(f"  {word:<15s} {tgt_str:<15s} "
                      f"{cos_start_tgt:<16.4f} {cos_ft_tgt:<14.4f} "
                      f"{toward_gt:<+14.4f} {shift_mag:.4f}{marker}")

        if differentiation:
            n_diff = sum(1 for d in differentiation if d["differentiated"])
            print(f"\n  Differentiated: {n_diff}/{len(differentiation)}")
            print(f"  Mean toward GT: {np.mean([d['toward_gt'] for d in differentiation]):+.4f}")
            print(f"  Mean shift: {np.mean([d['shift_magnitude'] for d in differentiation]):.4f}")

    # King-man+woman analogy spot check
    print(f"\n{'='*70}")
    print("ANALOGY SPOT CHECK: king - man + woman = ?")
    print(f"{'='*70}")

    s2i = token_index["str_to_id"]
    if all(w in s2i for w in ["king", "man", "woman"]):
        king_id = s2i["king"]
        man_id = s2i["man"]
        woman_id = s2i["woman"]

        active_set = set(top_ids.tolist())

        for label, emb in [("start", emb_start), ("fine-tuned", emb_ft), ("ground truth", emb_gt)]:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb_n = emb / (norms + 1e-10)
            query = emb_n[king_id] - emb_n[man_id] + emb_n[woman_id]
            query = query / (np.linalg.norm(query) + 1e-10)
            sims = emb_n @ query
            exclude = {king_id, man_id, woman_id}
            for i in range(len(sims)):
                if i not in active_set or i in exclude:
                    sims[i] = -np.inf
            top_k = np.argsort(sims)[::-1][:7]
            neighbors = [(token_index["id_to_str"][int(i)].strip(), float(sims[i]))
                         for i in top_k]
            nn_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbors)
            print(f"  {label:>12s}: {nn_str}")

    # Save results
    results = {
        "round_idx": round_idx,
        "tau": tau,
        "shifts": shifts,
        "overlaps": overlaps,
        "differentiation": differentiation,
        "eval_start": eval_start,
        "eval_ft": eval_ft,
        "eval_gt": eval_gt,
    }

    with open(os.path.join(round_dir, "eval.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {round_dir}/eval.json")
    return results


# ---------------------------------------------------------------------------
# Stage: Batch independent recovery across multiple single-token targets
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=14400,
    memory=32768,
)
def cl_batch_replicate(
    tau: float = 0.1,
    lr: float = 3e-4,
    target_tokens: int = 100_000,
    source_P: int = 100_000_000,
):
    """Run independent single-token recovery for each target in RECOVERY_TARGETS_01.

    Each target word gets its own round, starting from the same base model.
    This isolates recovery effects per token and provides independent measurements
    across semantic domains (royalty/gender, family/generational, geography).
    """
    import os, json, torch, copy
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.eval_embeddings import build_token_index, run_eval
    from language_reduction.recovery_eval import embedding_shift_toward_gt, neighbor_overlap
    from language_reduction.denoise import build_inverse_vocab_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 128
    mode_key = resolve_mode_key("vocab_only", tau)
    enc = tiktoken.get_encoding("gpt2")

    # Load shared resources once
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
    tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    inv = build_inverse_vocab_map(vocab_map, tau_top_ids)
    top_ids_set = set(int(x) for x in tau_top_ids)

    token_index = build_token_index(vocab_size=50257, active_ids=tau_top_ids)
    s2i = token_index["str_to_id"]

    source_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"

    with open(os.path.join(source_dir, "results.json")) as f:
        base_cfg = json.load(f)
    base_sd = torch.load(os.path.join(source_dir, "model.pt"), map_location="cpu")
    vocab_size = base_sd["transformer.wte.weight"].shape[0]

    emb_base = base_sd["transformer.wte.weight"].numpy()
    emb_gt = torch.load(os.path.join(gt_dir, "model.pt"),
                        map_location="cpu")["transformer.wte.weight"].numpy()

    # Build full word -> token ID lookup once
    full_word_to_ids = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        if cleaned:
            if cleaned not in full_word_to_ids:
                full_word_to_ids[cleaned] = []
            full_word_to_ids[cleaned].append(tid)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i+bl] for i in ix])
        y = torch.stack([data[i+1:i+bl+1] for i in ix])
        return x.to(device), y.to(device)

    # ===== Run each target independently =====
    all_round_results = []

    for round_idx, (target_word, expected_collapse, domain) in enumerate(RECOVERY_TARGETS_01):
        print(f"\n{'#'*70}")
        print(f"# ROUND {round_idx}: {target_word!r} -> {expected_collapse!r}  [{domain}]")
        print(f"{'#'*70}")

        round_dir = f"{CL_DIR}/tau_{tau:.3f}/round_{round_idx}"
        os.makedirs(round_dir, exist_ok=True)

        # --- Discover collapse for this target ---
        target_tids = full_word_to_ids.get(target_word.lower(), [])
        if not target_tids:
            print(f"  {target_word}: not found in GPT-2 vocab, skipping")
            continue

        collapsed_tids = []
        for tid in target_tids:
            if tid not in top_ids_set:
                tgt_orig = int(inv[int(vocab_map[tid])])
                tgt_str = enc.decode([tgt_orig]).strip()
                collapsed_tids.append(tid)
                print(f"  {target_word} (id={tid}): COLLAPSED -> {tgt_str!r} (id={tgt_orig})")
            else:
                print(f"  {target_word} (id={tid}): PRESERVED (skipping)")

        if not collapsed_tids:
            print(f"  No collapsed token IDs for {target_word!r}, skipping")
            continue

        target_token_ids = set(collapsed_tids)

        # Also check fanin of the collapse target
        primary_tid = collapsed_tids[0]
        primary_tgt = int(inv[int(vocab_map[primary_tid])])
        fanin = sum(1 for src in range(50257)
                    if src != primary_tgt
                    and int(inv[int(vocab_map[src])]) == primary_tgt
                    and src not in top_ids_set)
        print(f"  Collapse target {enc.decode([primary_tgt]).strip()!r} (id={primary_tgt}) fanin: {fanin}")

        # --- Build curriculum ---
        windows = []
        total_tokens = 0
        hit_count = 0
        last_window_end = -1

        for shard_idx in range(10):
            shard_path = os.path.join(f"{DATA_DIR}/tokens", f"shard_{shard_idx:05d}.npy")
            if not os.path.exists(shard_path):
                break
            shard = np.load(shard_path)
            last_window_end = -1

            for pos in range(len(shard)):
                if total_tokens >= target_tokens:
                    break
                if int(shard[pos]) not in target_token_ids:
                    continue

                ws = max(0, pos - block_size // 2)
                we = min(len(shard), ws + block_size)
                ws = we - block_size
                if ws < 0:
                    ws = 0
                    we = min(len(shard), block_size)

                if max(0, last_window_end - ws) > block_size // 2:
                    hit_count += 1
                    continue

                windows.append(shard[ws:we])
                total_tokens += len(windows[-1])
                last_window_end = we
                hit_count += 1

            if total_tokens >= target_tokens:
                break

        if len(windows) < 5:
            print(f"  Only {len(windows)} windows found, skipping")
            continue

        curriculum = np.concatenate(windows).astype(np.int64)
        print(f"  Curriculum: {len(windows)} windows, {len(curriculum):,} tokens ({hit_count} hits)")

        np.save(os.path.join(round_dir, "curriculum.npy"), curriculum)
        curriculum_meta = {
            "target_word": target_word,
            "expected_collapse": expected_collapse,
            "domain": domain,
            "n_windows": len(windows),
            "n_tokens": len(curriculum),
            "hit_count": hit_count,
            "round_idx": round_idx,
            "tau": tau,
            "collapse_target_fanin": fanin,
        }
        with open(os.path.join(round_dir, "curriculum_meta.json"), "w") as f:
            json.dump(curriculum_meta, f, indent=2)

        # --- Fine-tune from base model ---
        model = GPT(
            vocab_size=vocab_size, block_size=block_size,
            n_layer=base_cfg.get("n_layer", 2),
            n_head=base_cfg.get("n_head", 4),
            n_embd=base_cfg.get("n_embd", 128),
        ).to(device)
        model.load_state_dict(copy.deepcopy(base_sd))

        curriculum_t = torch.from_numpy(curriculum)
        split = int(0.9 * len(curriculum_t))
        train_data = curriculum_t[:split]
        val_data = curriculum_t[split:]

        batch_size = 64
        n_epochs = 5
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps_per_epoch = max(1, len(train_data) // (batch_size * block_size))
        n_steps = n_epochs * steps_per_epoch

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

            if step % 10 == 0 or step == n_steps - 1:
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
                if step == 0 or step == n_steps - 1:
                    print(f"  step {step:4d}/{n_steps}: train={loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

        torch.save(best_state if best_state else model.state_dict(),
                   os.path.join(round_dir, "model.pt"))
        train_result = {
            "mode": "vocab_only", "source_tau": tau, "source_P": source_P,
            "round_idx": round_idx, "target_word": target_word,
            "curriculum_tokens": len(curriculum), "lr": lr,
            "n_epochs": n_epochs, "n_steps": n_steps,
            "best_val_loss": best_val_loss,
            "n_layer": base_cfg.get("n_layer", 2),
            "n_head": base_cfg.get("n_head", 4),
            "n_embd": base_cfg.get("n_embd", 128),
        }
        with open(os.path.join(round_dir, "results.json"), "w") as f:
            json.dump(train_result, f, indent=2)

        # --- Evaluate ---
        emb_ft = (best_state if best_state else model.state_dict())["transformer.wte.weight"].numpy()

        # Target word differentiation from collapse target
        round_diff = []
        for tid in collapsed_tids:
            tgt_orig = int(inv[int(vocab_map[tid])])
            tgt_str = enc.decode([tgt_orig]).strip()

            cos_start_tgt = cosine(emb_base[tid], emb_base[tgt_orig])
            cos_ft_tgt = cosine(emb_ft[tid], emb_ft[tgt_orig])
            cos_start_gt = cosine(emb_base[tid], emb_gt[tid])
            cos_ft_gt = cosine(emb_ft[tid], emb_gt[tid])
            toward_gt = cos_ft_gt - cos_start_gt
            shift_mag = 1.0 - cosine(emb_ft[tid], emb_base[tid])

            differentiated = cos_ft_tgt < cos_start_tgt

            round_diff.append({
                "word": target_word, "token_id": tid,
                "collapse_target": tgt_str, "collapse_target_id": tgt_orig,
                "cos_start_to_target": cos_start_tgt,
                "cos_ft_to_target": cos_ft_tgt,
                "differentiated": differentiated,
                "toward_gt": toward_gt,
                "shift_magnitude": shift_mag,
            })

            marker = " DIFF" if differentiated else ""
            print(f"  {target_word:<12s} -> {tgt_str:<12s}: "
                  f"cos(start,tgt)={cos_start_tgt:+.4f}  cos(ft,tgt)={cos_ft_tgt:+.4f}  "
                  f"toward_gt={toward_gt:+.4f}  shift={shift_mag:.4f}{marker}")

        # Collapse target's neighborhood shift
        tgt_id = int(inv[int(vocab_map[collapsed_tids[0]])])
        if tgt_id in top_ids_set:
            tgt_shift = 1.0 - cosine(emb_ft[tgt_id], emb_base[tgt_id])
            tgt_toward_gt = cosine(emb_ft[tgt_id], emb_gt[tgt_id]) - cosine(emb_base[tgt_id], emb_gt[tgt_id])
            print(f"  Collapse target {enc.decode([tgt_id]).strip()!r}: "
                  f"shift={tgt_shift:.4f}  toward_gt={tgt_toward_gt:+.4f}")

        # Forgetting: check a few control words
        control_words = ["science", "good", "water", "school", "large"]
        control_shifts = []
        for cw in control_words:
            if cw in s2i:
                cwid = s2i[cw]
                cs = 1.0 - cosine(emb_ft[cwid], emb_base[cwid])
                control_shifts.append(cs)
        mean_control_shift = float(np.mean(control_shifts)) if control_shifts else 0.0
        print(f"  Control word mean shift: {mean_control_shift:.6f}")

        round_result = {
            "round_idx": round_idx,
            "target_word": target_word,
            "expected_collapse": expected_collapse,
            "domain": domain,
            "collapse_target_fanin": fanin,
            "best_val_loss": best_val_loss,
            "n_curriculum_tokens": len(curriculum),
            "differentiation": round_diff,
            "mean_control_shift": mean_control_shift,
        }

        with open(os.path.join(round_dir, "eval.json"), "w") as f:
            json.dump(round_result, f, indent=2, cls=NumpyEncoder)

        all_round_results.append(round_result)
        volume.commit()

    # ===== Cross-target summary =====
    print(f"\n{'='*70}")
    print("CROSS-TARGET SUMMARY")
    print(f"{'='*70}")
    print(f"  {'round':>5s}  {'target':<14s} {'collapse_to':<12s} {'domain':<20s} "
          f"{'fanin':>5s} {'diff?':>5s} {'toward_gt':>10s} {'shift':>8s} {'ctrl_shift':>10s}")
    print("  " + "-" * 105)

    for r in all_round_results:
        diff = r["differentiation"]
        n_diff = sum(1 for d in diff if d["differentiated"])
        mean_tgt = float(np.mean([d["toward_gt"] for d in diff])) if diff else 0
        mean_shift = float(np.mean([d["shift_magnitude"] for d in diff])) if diff else 0
        diff_str = f"{n_diff}/{len(diff)}"
        print(f"  {r['round_idx']:>5d}  {r['target_word']:<14s} {r['expected_collapse']:<12s} "
              f"{r['domain']:<20s} {r['collapse_target_fanin']:>5d} "
              f"{diff_str:>5s}  {mean_tgt:>+10.4f} {mean_shift:>8.4f} "
              f"{r['mean_control_shift']:>10.6f}")

    # Save summary
    summary_path = f"{CL_DIR}/tau_{tau:.3f}/batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_round_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved summary to {summary_path}")
    return all_round_results


# ---------------------------------------------------------------------------
# Stage: Dimension analysis — did the distinguishing semantic dimension emerge?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def cl_dimension_analysis(
    tau: float = 0.1,
    source_P: int = 100_000_000,
):
    """Test whether recovered tokens captured the semantic dimension that
    distinguishes them from their collapse targets.

    For each domain:
    - Royalty/gender: does prince−princess align with king−queen, man−woman, etc.?
    - Family/generational: does grandfather−father align with grandmother−mother?
    - Geography: does beijing−China align with tokyo−Japan, madrid−Spain?

    Also examines nearest neighbors of recovered tokens to check whether they
    cluster with the correct semantic category.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.eval_embeddings import build_token_index
    from language_reduction.denoise import build_inverse_vocab_map

    block_size = 128
    mode_key = resolve_mode_key("vocab_only", tau)
    enc = tiktoken.get_encoding("gpt2")

    # Load vocab mapping
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    token_index = build_token_index(vocab_size=50257, active_ids=tau_top_ids)
    s2i = token_index["str_to_id"]
    i2s = token_index["id_to_str"]
    active_set = set(tau_top_ids.tolist())

    # Load base and ground truth embeddings
    base_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"

    emb_base = torch.load(os.path.join(base_dir, "model.pt"),
                          map_location="cpu")["transformer.wte.weight"].numpy()
    emb_gt = torch.load(os.path.join(gt_dir, "model.pt"),
                        map_location="cpu")["transformer.wte.weight"].numpy()

    # Load per-round fine-tuned embeddings
    round_embs = {}
    for round_idx, (target_word, _, domain) in enumerate(RECOVERY_TARGETS_01):
        round_dir = f"{CL_DIR}/tau_{tau:.3f}/round_{round_idx}"
        model_path = os.path.join(round_dir, "model.pt")
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location="cpu")
            round_embs[round_idx] = sd["transformer.wte.weight"].numpy()
        else:
            print(f"  [skip] Round {round_idx} ({target_word}) model not found")

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def direction(emb, word_a, word_b):
        if word_a not in s2i or word_b not in s2i:
            return None
        v = emb[s2i[word_a]] - emb[s2i[word_b]]
        n = np.linalg.norm(v)
        return v / (n + 1e-10) if n > 1e-10 else None

    def top_neighbors(emb, token_id, k=10):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_n = emb / (norms + 1e-10)
        sims = emb_n @ emb_n[token_id]
        for i in range(len(sims)):
            if i not in active_set or i == token_id:
                sims[i] = -np.inf
        top_k = np.argsort(sims)[::-1][:k]
        return [(i2s.get(int(i), f"[{i}]").strip(), float(sims[i])) for i in top_k]

    results = {}

    # =====================================================================
    # 1. GENDER DIMENSION (princess, round 0)
    # =====================================================================
    print(f"{'='*70}")
    print("1. GENDER DIMENSION: does prince - princess align with gender?")
    print(f"{'='*70}")

    gender_pairs = [
        ("queen", "king"), ("woman", "man"), ("mother", "father"),
        ("daughter", "son"), ("girl", "boy"), ("she", "he"),
        ("sister", "brother"), ("wife", "husband"),
    ]

    if 0 in round_embs and "princess" in s2i and "prince" in s2i:
        emb_ft = round_embs[0]

        princess_id = s2i["princess"]
        prince_id = s2i["prince"]

        print(f"\n  Princess neighbors (fine-tuned, round 0):")
        nn_ft = top_neighbors(emb_ft, princess_id)
        for w, s in nn_ft:
            print(f"    {w:>15s}  {s:.4f}")

        print(f"\n  Princess neighbors (ground truth):")
        nn_gt = top_neighbors(emb_gt, princess_id)
        for w, s in nn_gt:
            print(f"    {w:>15s}  {s:.4f}")

        print(f"\n  Princess neighbors (base — untrained noise):")
        nn_base = top_neighbors(emb_base, princess_id)
        for w, s in nn_base[:5]:
            print(f"    {w:>15s}  {s:.4f}")

        # Compute prince−princess direction in each model
        print(f"\n  Alignment of (prince - princess) with gender reference directions:")
        print(f"  {'pair':<20s} {'fine-tuned':>12s} {'ground truth':>14s} {'base (noise)':>14s}")
        print("  " + "-" * 65)

        gender_alignments = {}
        for w_fem, w_masc in gender_pairs:
            d_ft = direction(emb_ft, w_masc, w_fem)
            d_gt = direction(emb_gt, w_masc, w_fem)
            d_base = direction(emb_base, w_masc, w_fem)

            pp_ft = direction(emb_ft, "prince", "princess")
            pp_gt = direction(emb_gt, "prince", "princess")
            pp_base = direction(emb_base, "prince", "princess")

            if d_ft is not None and pp_ft is not None:
                align_ft = cosine(pp_ft, d_ft)
                align_gt = cosine(pp_gt, d_gt) if (d_gt is not None and pp_gt is not None) else 0
                align_base = cosine(pp_base, d_base) if (d_base is not None and pp_base is not None) else 0

                gender_alignments[f"{w_masc}-{w_fem}"] = {
                    "fine_tuned": align_ft, "ground_truth": align_gt, "base": align_base,
                }
                print(f"  {w_masc+'-'+w_fem:<20s} {align_ft:>+12.4f} {align_gt:>+14.4f} {align_base:>+14.4f}")

        # Mean alignment
        if gender_alignments:
            mean_ft = float(np.mean([v["fine_tuned"] for v in gender_alignments.values()]))
            mean_gt = float(np.mean([v["ground_truth"] for v in gender_alignments.values()]))
            mean_base = float(np.mean([v["base"] for v in gender_alignments.values()]))
            print(f"\n  {'MEAN':<20s} {mean_ft:>+12.4f} {mean_gt:>+14.4f} {mean_base:>+14.4f}")

        results["gender"] = {
            "alignments": gender_alignments,
            "princess_neighbors_ft": nn_ft,
            "princess_neighbors_gt": nn_gt,
        }
    else:
        print("  [skip] Princess or prince not in token index, or round 0 model missing")

    # =====================================================================
    # 2. GENERATIONAL DIMENSION (grandfather round 1, grandmother round 2)
    # =====================================================================
    print(f"\n{'='*70}")
    print("2. GENERATIONAL DIMENSION: do grandfather-father and grandmother-mother align?")
    print(f"{'='*70}")

    gen_results = {}
    for round_idx, (target, collapse_to, _) in [(1, RECOVERY_TARGETS_01[1]), (2, RECOVERY_TARGETS_01[2])]:
        target_word, expected_collapse = target, collapse_to
        if round_idx not in round_embs:
            continue
        if target_word not in s2i:
            print(f"  [skip] {target_word} not in token index")
            continue

        emb_ft = round_embs[round_idx]
        tid = s2i[target_word]

        print(f"\n  {target_word.capitalize()} neighbors (fine-tuned, round {round_idx}):")
        nn_ft = top_neighbors(emb_ft, tid)
        for w, s in nn_ft:
            print(f"    {w:>15s}  {s:.4f}")

        print(f"\n  {target_word.capitalize()} neighbors (ground truth):")
        nn_gt = top_neighbors(emb_gt, tid)
        for w, s in nn_gt:
            print(f"    {w:>15s}  {s:.4f}")

        gen_results[target_word] = {
            "neighbors_ft": nn_ft,
            "neighbors_gt": nn_gt,
        }

    # Cross-alignment: does grandfather−father align with grandmother−mother?
    if 1 in round_embs and 2 in round_embs:
        if all(w in s2i for w in ["grandfather", "father", "grandmother", "mother"]):
            gf_ft = direction(round_embs[1], "grandfather", "father")
            gm_ft = direction(round_embs[2], "grandmother", "mother")
            gf_gt = direction(emb_gt, "grandfather", "father")
            gm_gt = direction(emb_gt, "grandmother", "mother")

            if gf_ft is not None and gm_ft is not None:
                cross_ft = cosine(gf_ft, gm_ft)
                cross_gt = cosine(gf_gt, gm_gt) if (gf_gt is not None and gm_gt is not None) else 0

                print(f"\n  Cross-alignment (grandfather-father) · (grandmother-mother):")
                print(f"    Fine-tuned (using separate round models): {cross_ft:+.4f}")
                print(f"    Ground truth:                             {cross_gt:+.4f}")

                gen_results["cross_alignment"] = {
                    "fine_tuned": cross_ft, "ground_truth": cross_gt,
                }

            # Also check alignment with gender directions
            if gf_ft is not None:
                print(f"\n  Generational vs gender alignment:")
                print(f"  {'direction':<30s} {'cos(gf-f, dir)':>16s} {'cos(gm-m, dir)':>16s}")
                print("  " + "-" * 65)

                for w_a, w_b in [("man", "woman"), ("king", "queen"), ("son", "daughter")]:
                    ref_ft_1 = direction(round_embs[1], w_a, w_b)
                    ref_ft_2 = direction(round_embs[2], w_a, w_b)
                    if ref_ft_1 is not None and gf_ft is not None:
                        a1 = cosine(gf_ft, ref_ft_1)
                        a2 = cosine(gm_ft, ref_ft_2) if (gm_ft is not None and ref_ft_2 is not None) else 0
                        print(f"  {w_a+'-'+w_b:<30s} {a1:>+16.4f} {a2:>+16.4f}")
                        gen_results[f"vs_{w_a}_{w_b}"] = {"gf_align": a1, "gm_align": a2}

    results["generational"] = gen_results

    # =====================================================================
    # 3. GEOGRAPHIC DIMENSION (beijing round 3, tokyo round 4, madrid round 5)
    # =====================================================================
    print(f"\n{'='*70}")
    print("3. GEOGRAPHIC DIMENSION: do city-country vectors align?")
    print(f"{'='*70}")

    geo_targets = [
        (3, "beijing", "china"),
        (4, "tokyo", "japan"),
        (5, "madrid", "spain"),
    ]

    geo_results = {}

    # Nearest neighbors for each city
    for round_idx, city, country in geo_targets:
        if round_idx not in round_embs:
            continue
        if city not in s2i:
            print(f"  [skip] {city} not in token index")
            continue

        emb_ft = round_embs[round_idx]
        tid = s2i[city]

        print(f"\n  {city.capitalize()} neighbors (fine-tuned, round {round_idx}):")
        nn_ft = top_neighbors(emb_ft, tid)
        for w, s in nn_ft:
            print(f"    {w:>15s}  {s:.4f}")

        print(f"  {city.capitalize()} neighbors (ground truth):")
        nn_gt = top_neighbors(emb_gt, tid)
        for w, s in nn_gt[:7]:
            print(f"    {w:>15s}  {s:.4f}")

        geo_results[city] = {
            "neighbors_ft": nn_ft,
            "neighbors_gt": nn_gt,
        }

    # Cross-alignment: do city−country vectors align across pairs?
    city_country_dirs_ft = {}
    city_country_dirs_gt = {}
    for round_idx, city, country in geo_targets:
        if round_idx not in round_embs:
            continue
        d_ft = direction(round_embs[round_idx], city, country)
        d_gt = direction(emb_gt, city, country)
        if d_ft is not None:
            city_country_dirs_ft[city] = d_ft
        if d_gt is not None:
            city_country_dirs_gt[city] = d_gt

    if len(city_country_dirs_ft) >= 2:
        print(f"\n  Pairwise alignment of city−country directions:")
        print(f"  {'pair':<25s} {'fine-tuned':>12s} {'ground truth':>14s}")
        print("  " + "-" * 55)

        cities = list(city_country_dirs_ft.keys())
        pairwise_ft = []
        pairwise_gt = []
        for i in range(len(cities)):
            for j in range(i+1, len(cities)):
                c1, c2 = cities[i], cities[j]
                align_ft = cosine(city_country_dirs_ft[c1], city_country_dirs_ft[c2])
                align_gt = 0
                if c1 in city_country_dirs_gt and c2 in city_country_dirs_gt:
                    align_gt = cosine(city_country_dirs_gt[c1], city_country_dirs_gt[c2])
                pairwise_ft.append(align_ft)
                pairwise_gt.append(align_gt)
                pair_label = f"{c1}-country · {c2}-country"
                print(f"  {pair_label:<25s} {align_ft:>+12.4f} {align_gt:>+14.4f}")
                geo_results[f"align_{c1}_{c2}"] = {
                    "fine_tuned": align_ft, "ground_truth": align_gt,
                }

        mean_ft = float(np.mean(pairwise_ft))
        mean_gt = float(np.mean(pairwise_gt))
        print(f"\n  {'MEAN':<25s} {mean_ft:>+12.4f} {mean_gt:>+14.4f}")
        geo_results["mean_pairwise_alignment"] = {
            "fine_tuned": mean_ft, "ground_truth": mean_gt,
        }

    # Check if cities cluster together vs with their countries
    # Use the round-3 (beijing) model as reference since it has beijing trained
    if 3 in round_embs:
        emb_ref = round_embs[3]
        print(f"\n  City-city vs city-country cosine similarities (beijing model):")

        city_words = [c for _, c, _ in geo_targets if c in s2i]
        country_words = [c for _, _, c in geo_targets if c in s2i]

        if city_words and country_words:
            # Only beijing is trained in round 3, but we can still check the existing embeddings
            print(f"  {'pair':<25s} {'cosine':>10s}")
            print("  " + "-" * 40)
            for cw in city_words:
                for other in city_words + country_words:
                    if cw == other:
                        continue
                    sim = cosine(emb_ref[s2i[cw]], emb_ref[s2i[other]])
                    label = f"{cw} · {other}"
                    tag = " (city)" if other in city_words else " (country)"
                    print(f"  {label:<25s} {sim:>+10.4f}{tag}")

    results["geography"] = geo_results

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Gender
    if "gender" in results and "alignments" in results["gender"]:
        ga = results["gender"]["alignments"]
        if ga:
            mean_ft = float(np.mean([v["fine_tuned"] for v in ga.values()]))
            mean_gt = float(np.mean([v["ground_truth"] for v in ga.values()]))
            print(f"\n  Gender: mean alignment of (prince-princess) with gender directions:")
            print(f"    Fine-tuned: {mean_ft:+.4f}   Ground truth: {mean_gt:+.4f}")
            emerged = abs(mean_ft) > 0.1
            print(f"    Gender dimension emerged: {'YES' if emerged else 'NO (alignment < 0.1)'}")

    # Generational
    if "generational" in results and "cross_alignment" in results["generational"]:
        ca = results["generational"]["cross_alignment"]
        print(f"\n  Generational: (grandfather-father) · (grandmother-mother) alignment:")
        print(f"    Fine-tuned: {ca['fine_tuned']:+.4f}   Ground truth: {ca['ground_truth']:+.4f}")
        emerged = abs(ca["fine_tuned"]) > 0.1
        print(f"    Generational dimension emerged: {'YES' if emerged else 'NO (alignment < 0.1)'}")

    # Geographic
    if "geography" in results and "mean_pairwise_alignment" in results["geography"]:
        ga = results["geography"]["mean_pairwise_alignment"]
        print(f"\n  Geography: mean pairwise alignment of city-country directions:")
        print(f"    Fine-tuned: {ga['fine_tuned']:+.4f}   Ground truth: {ga['ground_truth']:+.4f}")
        emerged = abs(ga["fine_tuned"]) > 0.1
        print(f"    Geographic dimension emerged: {'YES' if emerged else 'NO (alignment < 0.1)'}")

    # Save
    results_path = f"{CL_DIR}/tau_{tau:.3f}/dimension_analysis.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# Phase 2: Sequential recovery with degradation tracking
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=21600,
    memory=32768,
)
def cl_sequential(
    tau: float = 0.1,
    ordering: str = "clustered",
    lr: float = 3e-4,
    target_tokens: int = 100_000,
    source_P: int = 100_000_000,
    n_epochs: int = 5,
    batch_size: int = 64,
    block_size: int = 128,
):
    """Sequential token recovery with degradation tracking.

    Unlike Phase 1 (independent recovery from the same base model), this chains
    recoveries: round i fine-tunes from round i-1's checkpoint. After each round,
    we measure both forward recovery (did this token learn?) and backward
    degradation (did previously recovered tokens drift?).

    Ordering determines the sequence:
      - "clustered": geography → generational → gender
      - "interleaved": one from each domain in rotation
    """
    import os, json, copy, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.denoise import build_inverse_vocab_map
    from language_reduction.eval_embeddings import build_token_index

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    if ordering not in ORDERINGS:
        raise ValueError(f"Unknown ordering {ordering!r}, expected one of {list(ORDERINGS)}")
    targets = ORDERINGS[ordering]

    seq_dir = f"{CL_DIR}/tau_{tau:.3f}/sequential/{ordering}"
    os.makedirs(seq_dir, exist_ok=True)

    # --- Load shared resources ---
    mode_key = resolve_mode_key("vocab_only", tau)
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
    tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    inv = build_inverse_vocab_map(vocab_map, tau_top_ids)
    top_ids_set = set(int(x) for x in tau_top_ids)
    token_index = build_token_index(vocab_size=50257, active_ids=tau_top_ids)
    s2i = token_index["str_to_id"]

    base_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"

    with open(os.path.join(base_dir, "results.json")) as f:
        base_cfg = json.load(f)
    base_sd = torch.load(os.path.join(base_dir, "model.pt"), map_location="cpu")
    vocab_size = base_sd["transformer.wte.weight"].shape[0]
    emb_base = base_sd["transformer.wte.weight"].numpy().copy()
    emb_gt = torch.load(os.path.join(gt_dir, "model.pt"),
                        map_location="cpu")["transformer.wte.weight"].numpy()

    # Word → token ID lookup
    full_word_to_ids = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        if cleaned:
            if cleaned not in full_word_to_ids:
                full_word_to_ids[cleaned] = []
            full_word_to_ids[cleaned].append(tid)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i+bl] for i in ix])
        y = torch.stack([data[i+1:i+bl+1] for i in ix])
        return x.to(device), y.to(device)

    control_words = ["science", "good", "water", "school", "large"]

    # --- Track state across rounds ---
    # emb_at_recovery[round_idx] = embedding snapshot right after that round's training
    emb_at_recovery = {}
    # recovered_tokens[round_idx] = list of (word, token_id, collapse_target_id)
    recovered_tokens = {}
    # degradation_matrix[round_idx] = {measured_at_round: {token_id: metrics}}
    degradation_matrix = {}

    current_sd = copy.deepcopy(base_sd)

    print(f"{'='*70}")
    print(f"SEQUENTIAL RECOVERY — ordering={ordering}")
    print(f"Sequence: {' → '.join(t[0] for t in targets)}")
    print(f"{'='*70}")

    for round_idx, (target_word, expected_collapse, domain) in enumerate(targets):
        print(f"\n{'#'*70}")
        print(f"# ROUND {round_idx}: {target_word!r} [{domain}]")
        print(f"{'#'*70}")

        round_dir = os.path.join(seq_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # --- Find collapsed token IDs ---
        target_tids = full_word_to_ids.get(target_word.lower(), [])
        collapsed_tids = [tid for tid in target_tids if tid not in top_ids_set]
        if not collapsed_tids:
            print(f"  {target_word}: no collapsed tokens found, skipping")
            continue

        primary_tid = collapsed_tids[0]
        primary_tgt = int(inv[int(vocab_map[primary_tid])])
        tgt_str = enc.decode([primary_tgt]).strip()
        fanin = sum(1 for src in range(50257)
                    if src != primary_tgt
                    and int(inv[int(vocab_map[src])]) == primary_tgt
                    and src not in top_ids_set)
        print(f"  {target_word} (id={primary_tid}) → {tgt_str!r} (id={primary_tgt}, fanin={fanin})")

        target_token_ids = set(collapsed_tids)

        # --- Build curriculum ---
        windows = []
        total_tokens = 0
        hit_count = 0

        for shard_idx in range(10):
            shard_path = os.path.join(f"{DATA_DIR}/tokens", f"shard_{shard_idx:05d}.npy")
            if not os.path.exists(shard_path):
                break
            shard = np.load(shard_path)
            last_window_end = -1

            for pos in range(len(shard)):
                if total_tokens >= target_tokens:
                    break
                if int(shard[pos]) not in target_token_ids:
                    continue

                ws = max(0, pos - block_size // 2)
                we = min(len(shard), ws + block_size)
                ws = we - block_size
                if ws < 0:
                    ws = 0
                    we = min(len(shard), block_size)

                if max(0, last_window_end - ws) > block_size // 2:
                    hit_count += 1
                    continue

                windows.append(shard[ws:we])
                total_tokens += len(windows[-1])
                last_window_end = we
                hit_count += 1

            if total_tokens >= target_tokens:
                break

        if len(windows) < 5:
            print(f"  Only {len(windows)} windows, skipping")
            continue

        curriculum = np.concatenate(windows).astype(np.int64)
        print(f"  Curriculum: {len(windows)} windows, {len(curriculum):,} tokens")

        np.save(os.path.join(round_dir, "curriculum.npy"), curriculum)
        with open(os.path.join(round_dir, "curriculum_meta.json"), "w") as f:
            json.dump({
                "target_word": target_word, "expected_collapse": expected_collapse,
                "domain": domain, "round_idx": round_idx, "ordering": ordering,
                "n_windows": len(windows), "n_tokens": len(curriculum),
                "hit_count": hit_count, "tau": tau, "collapse_target_fanin": fanin,
            }, f, indent=2)

        # --- Fine-tune from previous round's checkpoint ---
        model = GPT(
            vocab_size=vocab_size, block_size=block_size,
            n_layer=base_cfg.get("n_layer", 2),
            n_head=base_cfg.get("n_head", 4),
            n_embd=base_cfg.get("n_embd", 128),
        ).to(device)
        model.load_state_dict(copy.deepcopy(current_sd))

        curriculum_t = torch.from_numpy(curriculum)
        split = int(0.9 * len(curriculum_t))
        train_data = curriculum_t[:split]
        val_data = curriculum_t[split:]

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps_per_epoch = max(1, len(train_data) // (batch_size * block_size))
        n_steps = n_epochs * steps_per_epoch

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

            if step % 10 == 0 or step == n_steps - 1:
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
                if step == 0 or step == n_steps - 1:
                    print(f"  step {step:4d}/{n_steps}: train={loss:.4f} "
                          f"val={val_loss:.4f} best={best_val_loss:.4f}")

        current_sd = best_state if best_state else {k: v.cpu().clone()
                                                     for k, v in model.state_dict().items()}
        torch.save(current_sd, os.path.join(round_dir, "model.pt"))
        with open(os.path.join(round_dir, "results.json"), "w") as f:
            json.dump({
                "mode": "vocab_only", "source_tau": tau, "source_P": source_P,
                "round_idx": round_idx, "ordering": ordering,
                "target_word": target_word, "curriculum_tokens": len(curriculum),
                "lr": lr, "n_epochs": n_epochs, "n_steps": n_steps,
                "best_val_loss": best_val_loss,
                "n_layer": base_cfg.get("n_layer", 2),
                "n_head": base_cfg.get("n_head", 4),
                "n_embd": base_cfg.get("n_embd", 128),
            }, f, indent=2)

        # --- Snapshot embeddings at recovery ---
        emb_now = current_sd["transformer.wte.weight"].numpy()
        emb_at_recovery[round_idx] = emb_now.copy()
        recovered_tokens[round_idx] = [
            (target_word, tid, primary_tgt) for tid in collapsed_tids
        ]

        # --- Recovery eval: this round's target ---
        recovery_metrics = []
        for tid in collapsed_tids:
            tgt_orig = int(inv[int(vocab_map[tid])])
            cos_start_tgt = cosine(emb_base[tid], emb_base[tgt_orig])
            cos_now_tgt = cosine(emb_now[tid], emb_now[tgt_orig])
            cos_now_gt = cosine(emb_now[tid], emb_gt[tid])
            cos_start_gt = cosine(emb_base[tid], emb_gt[tid])
            shift_mag = 1.0 - cosine(emb_now[tid], emb_base[tid])

            recovery_metrics.append({
                "word": target_word, "token_id": tid,
                "collapse_target_id": tgt_orig,
                "cos_to_collapse_target": cos_now_tgt,
                "cos_to_gt": cos_now_gt,
                "toward_gt": cos_now_gt - cos_start_gt,
                "shift_from_base": shift_mag,
            })
            print(f"  RECOVERY {target_word} (id={tid}): "
                  f"cos→tgt={cos_now_tgt:+.4f}  cos→gt={cos_now_gt:+.4f}  "
                  f"toward_gt={cos_now_gt - cos_start_gt:+.4f}  shift={shift_mag:.4f}")

        # --- Degradation eval: all previously recovered tokens ---
        degradation = {}
        for prev_round in range(round_idx):
            prev_entries = recovered_tokens[prev_round]
            emb_at_rec = emb_at_recovery[prev_round]
            prev_degradation = []

            for prev_word, prev_tid, prev_tgt_id in prev_entries:
                emb_stability = cosine(emb_now[prev_tid], emb_at_rec[prev_tid])
                shift_from_rec = 1.0 - emb_stability
                cos_gt_now = cosine(emb_now[prev_tid], emb_gt[prev_tid])
                cos_gt_at_rec = cosine(emb_at_rec[prev_tid], emb_gt[prev_tid])
                toward_gt_delta = cos_gt_now - cos_gt_at_rec
                cos_tgt_now = cosine(emb_now[prev_tid], emb_now[prev_tgt_id])
                cos_tgt_at_rec = cosine(emb_at_rec[prev_tid], emb_at_rec[prev_tgt_id])

                entry = {
                    "word": prev_word, "token_id": prev_tid,
                    "recovered_at_round": prev_round,
                    "measured_at_round": round_idx,
                    "embedding_stability": emb_stability,
                    "shift_from_recovery": shift_from_rec,
                    "toward_gt_at_recovery": cos_gt_at_rec,
                    "toward_gt_now": cos_gt_now,
                    "toward_gt_delta": toward_gt_delta,
                    "cos_to_collapse_at_recovery": cos_tgt_at_rec,
                    "cos_to_collapse_now": cos_tgt_now,
                }
                prev_degradation.append(entry)

                drift_label = "STABLE" if shift_from_rec < 0.01 else (
                    "DRIFT" if shift_from_rec < 0.05 else "MAJOR DRIFT")
                print(f"  DEGRAD round {prev_round} {prev_word}: "
                      f"stability={emb_stability:.4f}  "
                      f"toward_gt Δ={toward_gt_delta:+.4f}  [{drift_label}]")

            degradation[prev_round] = prev_degradation

        # Control word shift
        control_shifts = []
        for cw in control_words:
            if cw in s2i:
                cwid = s2i[cw]
                cs = 1.0 - cosine(emb_now[cwid], emb_base[cwid])
                control_shifts.append(cs)
        mean_control = float(np.mean(control_shifts)) if control_shifts else 0.0
        print(f"  Control word mean shift: {mean_control:.6f}")

        # Save per-round eval
        eval_result = {
            "round_idx": round_idx, "ordering": ordering,
            "target_word": target_word, "domain": domain,
            "best_val_loss": best_val_loss,
            "recovery": recovery_metrics,
            "degradation": degradation,
            "mean_control_shift": mean_control,
        }
        with open(os.path.join(round_dir, "eval.json"), "w") as f:
            json.dump(eval_result, f, indent=2, cls=NumpyEncoder)

        degradation_matrix[round_idx] = {
            "recovery": recovery_metrics,
            "degradation": degradation,
            "mean_control_shift": mean_control,
        }
        volume.commit()

    # --- Dimension analysis on final sequential model ---
    print(f"\n{'='*70}")
    print("DIMENSION ANALYSIS (final sequential model)")
    print(f"{'='*70}")

    emb_final = current_sd["transformer.wte.weight"].numpy()
    dim_results = {}

    def direction(emb, word_a, word_b):
        if word_a not in s2i or word_b not in s2i:
            return None
        v = emb[s2i[word_a]] - emb[s2i[word_b]]
        n = np.linalg.norm(v)
        return v / (n + 1e-10) if n > 1e-10 else None

    # Gender
    if "princess" in s2i and "prince" in s2i:
        gender_pairs = [
            ("queen", "king"), ("woman", "man"), ("mother", "father"),
            ("daughter", "son"), ("girl", "boy"), ("she", "he"),
            ("sister", "brother"), ("wife", "husband"),
        ]
        pp_dir = direction(emb_final, "prince", "princess")
        pp_gt = direction(emb_gt, "prince", "princess")
        if pp_dir is not None:
            alignments = []
            for w_masc, w_fem in gender_pairs:
                ref = direction(emb_final, w_masc, w_fem)
                ref_gt = direction(emb_gt, w_masc, w_fem)
                if ref is not None:
                    a = cosine(pp_dir, ref)
                    a_gt = cosine(pp_gt, ref_gt) if (pp_gt is not None and ref_gt is not None) else 0
                    alignments.append({"pair": f"{w_masc}-{w_fem}", "sequential": a, "ground_truth": a_gt})
            mean_seq = float(np.mean([a["sequential"] for a in alignments])) if alignments else 0
            mean_gt = float(np.mean([a["ground_truth"] for a in alignments])) if alignments else 0
            dim_results["gender"] = {"mean_alignment": mean_seq, "ground_truth": mean_gt, "pairs": alignments}
            emerged = abs(mean_seq) > 0.1
            print(f"  Gender: {mean_seq:+.4f} (GT: {mean_gt:+.4f}) {'EMERGED' if emerged else 'not emerged'}")

    # Generational
    if all(w in s2i for w in ["grandfather", "father", "grandmother", "mother"]):
        gf = direction(emb_final, "grandfather", "father")
        gm = direction(emb_final, "grandmother", "mother")
        gf_gt = direction(emb_gt, "grandfather", "father")
        gm_gt = direction(emb_gt, "grandmother", "mother")
        if gf is not None and gm is not None:
            cross = cosine(gf, gm)
            cross_gt = cosine(gf_gt, gm_gt) if (gf_gt is not None and gm_gt is not None) else 0
            dim_results["generational"] = {"cross_alignment": cross, "ground_truth": cross_gt}
            emerged = abs(cross) > 0.1
            print(f"  Generational: {cross:+.4f} (GT: {cross_gt:+.4f}) {'EMERGED' if emerged else 'not emerged'}")

    # Geographic
    geo_pairs = [("beijing", "china"), ("tokyo", "japan"), ("madrid", "spain")]
    geo_dirs = {}
    geo_dirs_gt = {}
    for city, country in geo_pairs:
        d = direction(emb_final, city, country)
        d_gt = direction(emb_gt, city, country)
        if d is not None:
            geo_dirs[city] = d
        if d_gt is not None:
            geo_dirs_gt[city] = d_gt

    if len(geo_dirs) >= 2:
        cities = list(geo_dirs.keys())
        pw = []
        pw_gt = []
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                c1, c2 = cities[i], cities[j]
                pw.append(cosine(geo_dirs[c1], geo_dirs[c2]))
                if c1 in geo_dirs_gt and c2 in geo_dirs_gt:
                    pw_gt.append(cosine(geo_dirs_gt[c1], geo_dirs_gt[c2]))
        mean_pw = float(np.mean(pw))
        mean_pw_gt = float(np.mean(pw_gt)) if pw_gt else 0
        dim_results["geography"] = {"mean_pairwise": mean_pw, "ground_truth": mean_pw_gt}
        emerged = abs(mean_pw) > 0.1
        print(f"  Geography: {mean_pw:+.4f} (GT: {mean_pw_gt:+.4f}) {'EMERGED' if emerged else 'not emerged'}")

    # --- Cross-round degradation summary ---
    print(f"\n{'='*70}")
    print("DEGRADATION MATRIX")
    print(f"{'='*70}")

    # Header
    token_labels = [t[0] for t in targets]
    col_label = "measured\\recovered"
    header = f"  {col_label:<18s}"
    for label in token_labels:
        header += f" {label:>10s}"
    print(header)
    print("  " + "-" * (18 + 11 * len(token_labels)))

    for meas_round in range(len(targets)):
        row = f"  round {meas_round} ({token_labels[meas_round][:4]})"
        row = f"  {row:<18s}"
        for rec_round in range(len(targets)):
            if rec_round > meas_round:
                row += f" {'—':>10s}"
            elif rec_round == meas_round:
                if meas_round in degradation_matrix:
                    rec = degradation_matrix[meas_round]["recovery"]
                    if rec:
                        tg = rec[0]["toward_gt"]
                        row += f" {tg:>+10.4f}"
                    else:
                        row += f" {'skip':>10s}"
                else:
                    row += f" {'skip':>10s}"
            else:
                if meas_round in degradation_matrix:
                    deg = degradation_matrix[meas_round].get("degradation", {})
                    if rec_round in deg:
                        entries = deg[rec_round]
                        if entries:
                            delta = entries[0]["toward_gt_delta"]
                            row += f" {delta:>+10.4f}"
                        else:
                            row += f" {'—':>10s}"
                    else:
                        row += f" {'—':>10s}"
                else:
                    row += f" {'—':>10s}"
        print(row)

    print(f"\n  Diagonal = toward_gt at recovery. Off-diagonal = toward_gt delta since recovery.")
    print(f"  Negative off-diagonal = forgetting. Positive = reinforcement.")

    # --- Save everything ---
    summary = {
        "ordering": ordering,
        "tau": tau,
        "targets": [(w, c, d) for w, c, d in targets],
        "degradation_matrix": degradation_matrix,
        "dimension_analysis": dim_results,
    }
    with open(os.path.join(seq_dir, "degradation_matrix.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    with open(os.path.join(seq_dir, "dimension_analysis.json"), "w") as f:
        json.dump(dim_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nSaved to {seq_dir}/")
    return summary


# ---------------------------------------------------------------------------
# Phase 2 scaled: 100-token sequential recovery
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=36000,
    memory=32768,
)
def cl_sequential_scaled(
    tau: float = 0.1,
    n_tokens: int = 100,
    lr: float = 3e-4,
    target_tokens_per_word: int = 50_000,
    source_P: int = 100_000_000,
    n_epochs: int = 5,
    batch_size: int = 64,
    block_size: int = 128,
    seed: int = 42,
    min_corpus_hits: int = 200,
):
    """Scaled sequential recovery: auto-select N tokens and run the full pipeline.

    Token selection is automatic: pick collapsed tokens that decode to clean
    single words, have sufficient corpus frequency, and span the fanin
    distribution. Ordering is random (fixed seed) so fanin effects can be
    analyzed as a covariate rather than confounded with position.
    """
    import os, json, copy, torch, random
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.denoise import build_inverse_vocab_map

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(seed)

    seq_dir = f"{CL_DIR}/tau_{tau:.3f}/sequential/scaled_{n_tokens}_seed{seed}"
    os.makedirs(seq_dir, exist_ok=True)

    # --- Load vocab mapping ---
    mode_key = resolve_mode_key("vocab_only", tau)
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
    tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    inv = build_inverse_vocab_map(vocab_map, tau_top_ids)
    top_ids_set = set(int(x) for x in tau_top_ids)

    # --- Load base and ground truth models ---
    base_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    gt_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"

    with open(os.path.join(base_dir, "results.json")) as f:
        base_cfg = json.load(f)
    base_sd = torch.load(os.path.join(base_dir, "model.pt"), map_location="cpu")
    vocab_size = base_sd["transformer.wte.weight"].shape[0]
    emb_base = base_sd["transformer.wte.weight"].numpy().copy()
    emb_gt = torch.load(os.path.join(gt_dir, "model.pt"),
                        map_location="cpu")["transformer.wte.weight"].numpy()

    # --- Auto-select tokens ---
    print(f"{'='*70}")
    print(f"TOKEN SELECTION: picking {n_tokens} from {50257 - len(top_ids_set)} collapsed tokens")
    print(f"{'='*70}")

    # Precompute fanin for all collapse targets
    fanin_map = {}
    for src in range(50257):
        if src in top_ids_set:
            continue
        tgt = int(inv[int(vocab_map[src])])
        fanin_map[tgt] = fanin_map.get(tgt, 0) + 1

    candidates = []
    for tid in range(50257):
        if tid in top_ids_set:
            continue
        raw = enc.decode([tid])
        word = raw.strip().lower()
        if len(word) < 3:
            continue
        if not word.isalpha():
            continue
        # Require word-initial token (leading space in GPT-2)
        raw_bytes = enc.decode_single_token_bytes(tid)
        if not raw_bytes.startswith(b' '):
            continue

        tgt_orig = int(inv[int(vocab_map[tid])])
        fanin = fanin_map.get(tgt_orig, 0)
        tgt_word = enc.decode([tgt_orig]).strip()
        candidates.append({
            "token_id": tid, "word": word, "collapse_target_id": tgt_orig,
            "collapse_target_word": tgt_word, "fanin": fanin,
        })

    print(f"  {len(candidates)} candidates after filtering (alphabetic, len>=3, word-initial)")

    # --- Count corpus frequency in a single pass ---
    print("  Counting corpus frequencies...")
    candidate_ids_arr = np.array([c["token_id"] for c in candidates], dtype=np.int64)
    freq = {int(tid): 0 for tid in candidate_ids_arr}

    for shard_idx in range(10):
        shard_path = os.path.join(f"{DATA_DIR}/tokens", f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        shard = np.load(shard_path)
        # Vectorized: count all tokens in shard at once
        counts = np.bincount(shard.astype(np.int64), minlength=50257)
        for tid in candidate_ids_arr:
            freq[int(tid)] += int(counts[tid])

    candidates = [c for c in candidates if freq[c["token_id"]] >= min_corpus_hits]
    print(f"  {len(candidates)} candidates with >= {min_corpus_hits} corpus hits")

    if len(candidates) < n_tokens:
        print(f"  WARNING: only {len(candidates)} candidates, requested {n_tokens}")
        n_tokens = len(candidates)

    # Sample to span fanin distribution: stratify by fanin quartile
    candidates.sort(key=lambda c: c["fanin"])
    quartile_size = len(candidates) // 4
    per_quartile = n_tokens // 4
    remainder = n_tokens - 4 * per_quartile

    selected = []
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else len(candidates)
        pool = candidates[start:end]
        take = per_quartile + (1 if q < remainder else 0)
        selected.extend(rng.sample(pool, min(take, len(pool))))

    rng.shuffle(selected)
    targets = selected[:n_tokens]

    fanin_values = [t["fanin"] for t in targets]
    print(f"\n  Selected {len(targets)} tokens")
    print(f"  Fanin range: {min(fanin_values)} — {max(fanin_values)} "
          f"(mean {np.mean(fanin_values):.1f}, median {np.median(fanin_values):.0f})")
    print(f"  First 10: {', '.join(t['word'] for t in targets[:10])}")
    print(f"  Last 10:  {', '.join(t['word'] for t in targets[-10:])}")

    # Save selection
    with open(os.path.join(seq_dir, "token_selection.json"), "w") as f:
        json.dump({
            "n_tokens": len(targets), "seed": seed, "tau": tau,
            "min_corpus_hits": min_corpus_hits,
            "targets": targets,
            "fanin_stats": {
                "min": int(min(fanin_values)), "max": int(max(fanin_values)),
                "mean": float(np.mean(fanin_values)),
                "median": float(np.median(fanin_values)),
            },
        }, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    # --- Extract all curricula in a single corpus pass ---
    print(f"\n{'='*70}")
    print("CURRICULUM EXTRACTION (single-pass)")
    print(f"{'='*70}")

    target_id_to_round = {}
    for round_idx, t in enumerate(targets):
        target_id_to_round[t["token_id"]] = round_idx

    all_target_ids = set(target_id_to_round.keys())
    windows_per_round = {i: [] for i in range(len(targets))}
    tokens_per_round = {i: 0 for i in range(len(targets))}

    for shard_idx in range(10):
        shard_path = os.path.join(f"{DATA_DIR}/tokens", f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            break
        shard = np.load(shard_path)
        last_window_end_per_round = {i: -1 for i in range(len(targets))}

        for pos in range(len(shard)):
            tid = int(shard[pos])
            if tid not in all_target_ids:
                continue
            round_idx = target_id_to_round[tid]

            if tokens_per_round[round_idx] >= target_tokens_per_word:
                continue

            ws = max(0, pos - block_size // 2)
            we = min(len(shard), ws + block_size)
            ws = we - block_size
            if ws < 0:
                ws = 0
                we = min(len(shard), block_size)

            if max(0, last_window_end_per_round[round_idx] - ws) > block_size // 2:
                continue

            windows_per_round[round_idx].append(shard[ws:we].copy())
            tokens_per_round[round_idx] += we - ws
            last_window_end_per_round[round_idx] = we

        all_done = all(tokens_per_round[i] >= target_tokens_per_word
                       for i in range(len(targets)))
        if all_done:
            break

    # Check coverage
    n_sufficient = sum(1 for i in range(len(targets)) if len(windows_per_round[i]) >= 5)
    print(f"  {n_sufficient}/{len(targets)} tokens have sufficient curriculum (>= 5 windows)")
    for i in range(len(targets)):
        if len(windows_per_round[i]) < 5:
            print(f"    WARNING: {targets[i]['word']} has only {len(windows_per_round[i])} windows")

    # Save curricula
    for round_idx in range(len(targets)):
        if not windows_per_round[round_idx]:
            continue
        round_dir = os.path.join(seq_dir, f"round_{round_idx:03d}")
        os.makedirs(round_dir, exist_ok=True)
        curriculum = np.concatenate(windows_per_round[round_idx]).astype(np.int64)
        np.save(os.path.join(round_dir, "curriculum.npy"), curriculum)
        with open(os.path.join(round_dir, "curriculum_meta.json"), "w") as f:
            json.dump({
                "target_word": targets[round_idx]["word"],
                "token_id": targets[round_idx]["token_id"],
                "collapse_target": targets[round_idx]["collapse_target_word"],
                "collapse_target_id": targets[round_idx]["collapse_target_id"],
                "fanin": targets[round_idx]["fanin"],
                "n_windows": len(windows_per_round[round_idx]),
                "n_tokens": len(curriculum),
                "round_idx": round_idx,
            }, f, indent=2)
    volume.commit()
    print("  Curricula saved.")

    # --- Sequential recovery loop ---
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL RECOVERY: {len(targets)} rounds")
    print(f"{'='*70}")

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i+bl] for i in ix])
        y = torch.stack([data[i+1:i+bl+1] for i in ix])
        return x.to(device), y.to(device)

    control_word_ids = []
    for cw in ["science", "good", "water", "school", "large",
               "time", "work", "people", "state", "world"]:
        for tid in range(50257):
            if enc.decode([tid]).strip().lower() == cw:
                control_word_ids.append(tid)
                break

    emb_at_recovery = {}
    recovered_info = {}
    # Per-round summary: list of dicts
    round_summaries = []

    current_sd = copy.deepcopy(base_sd)

    for round_idx, target in enumerate(targets):
        word = target["word"]
        tid = target["token_id"]
        tgt_id = target["collapse_target_id"]
        fanin = target["fanin"]
        round_dir = os.path.join(seq_dir, f"round_{round_idx:03d}")

        if not windows_per_round[round_idx] or len(windows_per_round[round_idx]) < 5:
            print(f"  round {round_idx:3d} [{word}]: skipped (insufficient data)")
            continue

        curriculum_path = os.path.join(round_dir, "curriculum.npy")
        curriculum = np.load(curriculum_path)
        curriculum_t = torch.from_numpy(curriculum.astype(np.int64))

        # --- Train ---
        model = GPT(
            vocab_size=vocab_size, block_size=block_size,
            n_layer=base_cfg.get("n_layer", 2),
            n_head=base_cfg.get("n_head", 4),
            n_embd=base_cfg.get("n_embd", 128),
        ).to(device)
        model.load_state_dict(copy.deepcopy(current_sd))

        split = int(0.9 * len(curriculum_t))
        train_data = curriculum_t[:split]
        val_data = curriculum_t[split:]

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps_per_epoch = max(1, len(train_data) // (batch_size * block_size))
        n_steps = n_epochs * steps_per_epoch

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

            if step % 10 == 0 or step == n_steps - 1:
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

        current_sd = best_state if best_state else {k: v.cpu().clone()
                                                     for k, v in model.state_dict().items()}
        torch.save(current_sd, os.path.join(round_dir, "model.pt"))

        emb_now = current_sd["transformer.wte.weight"].numpy()
        emb_at_recovery[round_idx] = emb_now[tid].copy()
        recovered_info[round_idx] = (word, tid, tgt_id, fanin)

        # --- Recovery metrics ---
        cos_to_tgt = cosine(emb_now[tid], emb_now[tgt_id])
        cos_to_gt = cosine(emb_now[tid], emb_gt[tid])
        cos_base_gt = cosine(emb_base[tid], emb_gt[tid])
        toward_gt = cos_to_gt - cos_base_gt
        shift_from_base = 1.0 - cosine(emb_now[tid], emb_base[tid])

        # --- Degradation of all previous tokens ---
        degradation_entries = []
        for prev_round in range(round_idx):
            if prev_round not in recovered_info:
                continue
            prev_word, prev_tid, prev_tgt_id, prev_fanin = recovered_info[prev_round]
            prev_emb_at_rec = emb_at_recovery[prev_round]

            stab = cosine(emb_now[prev_tid], prev_emb_at_rec)
            cos_gt_now = cosine(emb_now[prev_tid], emb_gt[prev_tid])
            cos_gt_at_rec = cosine(prev_emb_at_rec, emb_gt[prev_tid])
            tg_delta = cos_gt_now - cos_gt_at_rec

            degradation_entries.append({
                "prev_round": prev_round, "prev_word": prev_word,
                "prev_tid": prev_tid, "prev_fanin": prev_fanin,
                "stability": stab, "toward_gt_delta": tg_delta,
            })

        # Control shift
        ctrl_shifts = [1.0 - cosine(emb_now[cid], emb_base[cid]) for cid in control_word_ids]
        mean_ctrl = float(np.mean(ctrl_shifts)) if ctrl_shifts else 0.0

        # Aggregate degradation stats
        if degradation_entries:
            stabilities = [e["stability"] for e in degradation_entries]
            tg_deltas = [e["toward_gt_delta"] for e in degradation_entries]
            mean_stab = float(np.mean(stabilities))
            min_stab = float(min(stabilities))
            mean_tg = float(np.mean(tg_deltas))
            min_tg = float(min(tg_deltas))
            n_forgetting = sum(1 for d in tg_deltas if d < -0.01)
        else:
            mean_stab = min_stab = 1.0
            mean_tg = min_tg = 0.0
            n_forgetting = 0

        summary = {
            "round_idx": round_idx, "word": word, "token_id": tid,
            "collapse_target": target["collapse_target_word"],
            "fanin": fanin, "best_val_loss": best_val_loss,
            "cos_to_collapse_target": cos_to_tgt,
            "toward_gt": toward_gt, "shift_from_base": shift_from_base,
            "mean_stability": mean_stab, "min_stability": min_stab,
            "mean_toward_gt_delta": mean_tg, "min_toward_gt_delta": min_tg,
            "n_forgetting": n_forgetting, "n_prev_tokens": len(degradation_entries),
            "mean_control_shift": mean_ctrl,
        }
        round_summaries.append(summary)

        # Save per-round results
        with open(os.path.join(round_dir, "results.json"), "w") as f:
            json.dump({**summary, "degradation": degradation_entries}, f, indent=2, cls=NumpyEncoder)

        # Progress log (compact)
        degrad_str = (f"stab={mean_stab:.4f}/{min_stab:.4f} "
                      f"tgΔ={mean_tg:+.4f}/{min_tg:+.4f} "
                      f"forget={n_forgetting}") if degradation_entries else "—"
        print(f"  round {round_idx:3d} [{word:<15s}] fanin={fanin:3d}  "
              f"toward_gt={toward_gt:+.4f}  ctrl={mean_ctrl:.4f}  {degrad_str}")

        if round_idx % 10 == 9:
            volume.commit()

    volume.commit()

    # --- Final summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if round_summaries:
        all_tg = [s["toward_gt"] for s in round_summaries]
        all_stab = [s["mean_stability"] for s in round_summaries if s["n_prev_tokens"] > 0]
        all_ctrl = [s["mean_control_shift"] for s in round_summaries]
        all_forget = [s["n_forgetting"] for s in round_summaries]

        print(f"  Tokens recovered: {len(round_summaries)}")
        print(f"  toward_gt:  mean={np.mean(all_tg):+.4f}  "
              f"min={min(all_tg):+.4f}  max={max(all_tg):+.4f}")
        if all_stab:
            print(f"  stability:  mean={np.mean(all_stab):.4f}  "
                  f"min={min(all_stab):.4f}")
        print(f"  control:    mean={np.mean(all_ctrl):.4f}  "
              f"final={all_ctrl[-1]:.4f}")
        print(f"  forgetting: total rounds with forget>0.01 = "
              f"{sum(1 for f in all_forget if f > 0)}/{len(all_forget)}")

        # Fanin correlation with degradation
        rounds_with_prev = [s for s in round_summaries if s["n_prev_tokens"] > 0]
        if len(rounds_with_prev) >= 10:
            fanins = np.array([s["fanin"] for s in rounds_with_prev])
            min_stabs = np.array([s["min_stability"] for s in rounds_with_prev])
            from scipy.stats import spearmanr
            corr, pval = spearmanr(fanins, min_stabs)
            print(f"\n  Fanin vs min_stability: Spearman r={corr:+.3f} (p={pval:.3f})")

            # Position (round_idx) correlation
            positions = np.array([s["round_idx"] for s in rounds_with_prev])
            corr_pos, pval_pos = spearmanr(positions, min_stabs)
            print(f"  Position vs min_stability: Spearman r={corr_pos:+.3f} (p={pval_pos:.3f})")

    # Save everything
    final = {
        "n_tokens": len(targets), "seed": seed, "tau": tau,
        "round_summaries": round_summaries,
    }
    with open(os.path.join(seq_dir, "summary.json"), "w") as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nSaved to {seq_dir}/")
    return final
