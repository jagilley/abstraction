from language_reduction.shared import (
    app, volume, DATA_DIR, image, NumpyEncoder, resolve_mode_key,
)


# ---------------------------------------------------------------------------
# Stage 17: Extract DPO pairs
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=1800,
    memory=32768,
)
def extract_dpo_pairs_stage(
    tau: float = 0.3,
    max_shards: int = 10,
    max_pairs: int = 5000,
):
    """Extract (chosen, rejected) pairs for DPO training."""
    from language_reduction.dpo_pairs import (
        extract_dpo_pairs, save_dpo_pairs, DPOPairsConfig,
    )

    config = DPOPairsConfig(
        tau=tau,
        max_shards=max_shards,
        max_pairs=max_pairs,
    )

    tokens_dir = f"{DATA_DIR}/tokens"
    denoised_dir = f"{DATA_DIR}/denoised/tau_{tau:.3f}"
    output_dir = f"{DATA_DIR}/dpo_pairs/tau_{tau:.3f}"

    chosen, rejected, stats = extract_dpo_pairs(tokens_dir, denoised_dir, config)
    save_dpo_pairs(chosen, rejected, stats, output_dir)
    volume.commit()
    return stats


# ---------------------------------------------------------------------------
# Stage 18: DPO training
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def dpo_train_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    lr: float = 5e-5,
    n_epochs: int = 3,
    beta: float = 0.1,
    block_size: int = 128,
    batch_size: int = 32,
    eval_interval: int = 10,
):
    """DPO training: teach the model that queen is worth using.

    Starts from the SFT fine-tuned model. Uses frozen copy as reference.
    Trains on (chosen=original with queen, rejected=queen replaced) pairs.
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load DPO pairs ---
    pairs_dir = f"{DATA_DIR}/dpo_pairs/tau_{source_tau:.3f}"
    chosen_all = torch.from_numpy(np.load(f"{pairs_dir}/chosen.npy"))
    rejected_all = torch.from_numpy(np.load(f"{pairs_dir}/rejected.npy"))
    n_pairs = len(chosen_all)
    print(f"Loaded {n_pairs} DPO pairs")

    split = int(0.9 * n_pairs)
    chosen_train, chosen_val = chosen_all[:split], chosen_all[split:]
    rejected_train, rejected_val = rejected_all[:split], rejected_all[split:]

    # --- Load SFT fine-tuned model ---
    ft_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    with open(os.path.join(ft_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(ft_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]

    model_kwargs = dict(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
    )

    # Policy (trainable)
    policy = GPT(**model_kwargs).to(device)
    policy.load_state_dict(sd)

    # Reference (frozen)
    ref = GPT(**model_kwargs).to(device)
    ref.load_state_dict(sd)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    print(f"Loaded SFT checkpoint from {ft_dir}")

    # --- Helpers ---
    def compute_seq_log_probs(model, input_ids):
        """Sum of per-token log probs for autoregressive sequences."""
        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:]
        per_token = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        return per_token.sum(dim=-1)  # (B,)

    def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids):
        pi_w = compute_seq_log_probs(policy_model, chosen_ids)
        pi_l = compute_seq_log_probs(policy_model, rejected_ids)
        with torch.no_grad():
            ref_w = compute_seq_log_probs(ref_model, chosen_ids)
            ref_l = compute_seq_log_probs(ref_model, rejected_ids)
        logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
        loss = -F.logsigmoid(logits).mean()
        accuracy = (logits > 0).float().mean()
        return loss, accuracy

    def get_batch(chosen_data, rejected_data, bs):
        idx = torch.randint(len(chosen_data), (bs,))
        return chosen_data[idx].to(device), rejected_data[idx].to(device)

    # --- Training ---
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(chosen_train) // batch_size)
    n_steps = n_epochs * steps_per_epoch
    print(f"Training: {n_steps} steps ({n_epochs} epochs, {steps_per_epoch} steps/epoch)")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None

    for step in range(n_steps):
        policy.train()
        c_batch, r_batch = get_batch(chosen_train, rejected_train, batch_size)
        loss, acc = dpo_loss(policy, ref, c_batch, r_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            policy.eval()
            with torch.no_grad():
                n_eval = min(5, max(1, len(chosen_val) // batch_size))
                vl_sum, va_sum = 0.0, 0.0
                for _ in range(n_eval):
                    vc, vr = get_batch(chosen_val, rejected_val, batch_size)
                    vl, va = dpo_loss(policy, ref, vc, vr)
                    vl_sum += float(vl)
                    va_sum += float(va)
                val_loss = vl_sum / n_eval
                val_acc = va_sum / n_eval

            train_losses.append((step, float(loss), float(acc)))
            val_losses.append((step, val_loss, val_acc))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

            print(f"  step {step:4d}/{n_steps}: "
                  f"train_loss={loss:.4f} train_acc={acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                  f"best_val={best_val_loss:.4f}")

    # --- Save ---
    save_dir = f"{DATA_DIR}/models/dpo/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state if best_state else policy.state_dict(),
               os.path.join(save_dir, "model.pt"))

    result = {
        "source_tau": source_tau,
        "source_P": source_P,
        "method": "dpo",
        "lr": lr,
        "beta": beta,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "n_pairs": n_pairs,
        "best_val_loss": best_val_loss,
        "train_curve": train_losses,
        "val_curve": val_losses,
        **model_kwargs,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"Saved DPO model to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage 19: More-SFT control (same compute budget as DPO)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def more_sft_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    lr: float = 3e-4,
    n_steps: int = 0,
    block_size: int = 128,
    batch_size: int = 64,
    eval_interval: int = 10,
):
    """Continue SFT on queen curriculum from the fine-tuned checkpoint.

    Control condition for DPO: same starting checkpoint, same number of
    optimizer steps (matched to DPO if n_steps=0), but standard CE loss.
    """
    import os, json, torch
    import numpy as np
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load curriculum ---
    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    print(f"Curriculum: {len(curriculum):,} tokens")

    # --- Load SFT fine-tuned model ---
    ft_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    with open(os.path.join(ft_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(ft_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
    ).to(device)
    model.load_state_dict(sd)
    print(f"Loaded SFT checkpoint from {ft_dir}")

    # Match DPO step count if not specified
    if n_steps == 0:
        pairs_meta_path = f"{DATA_DIR}/dpo_pairs/tau_{source_tau:.3f}/dpo_pairs_meta.json"
        if os.path.exists(pairs_meta_path):
            with open(pairs_meta_path) as f:
                dpo_meta = json.load(f)
            dpo_n_pairs = dpo_meta["n_pairs"]
            dpo_batch_size = 32
            dpo_epochs = 3
            n_steps = dpo_epochs * max(1, int(0.9 * dpo_n_pairs) // dpo_batch_size)
            print(f"Matching DPO step count: {n_steps}")
        else:
            n_steps = 50
            print(f"No DPO meta found, defaulting to {n_steps} steps")

    # --- Training ---
    split = int(0.9 * len(curriculum))
    train_data = curriculum[:split]
    val_data = curriculum[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i+bl] for i in ix])
        y = torch.stack([data[i+1:i+bl+1] for i in ix])
        return x.to(device), y.to(device)

    train_losses = []
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
                n_eval = min(5, max(1, len(val_data) // (batch_size * block_size)))
                vl_accum = 0.0
                for _ in range(n_eval):
                    vx, vy = get_batch(val_data, batch_size, block_size)
                    _, vl = model(vx, vy)
                    vl_accum += float(vl)
                val_loss = vl_accum / n_eval

            train_losses.append((step, float(loss)))
            val_losses.append((step, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:4d}/{n_steps}: train={loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

    # --- Save ---
    save_dir = f"{DATA_DIR}/models/more_sft/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(),
               os.path.join(save_dir, "model.pt"))

    result = {
        "source_tau": source_tau,
        "source_P": source_P,
        "method": "more_sft",
        "lr": lr,
        "n_steps": n_steps,
        "best_val_loss": best_val_loss,
        "train_curve": train_losses,
        "val_curve": val_losses,
        "n_layer": cfg.get("n_layer", 2),
        "n_head": cfg.get("n_head", 4),
        "n_embd": cfg.get("n_embd", 128),
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()
    print(f"Saved more-SFT model to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage 20: Scaffolding evaluation
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def scaffolding_eval_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
):
    """Evaluate female-ruler representation across all conditions.

    Compares 4 models x 2 variants (queen intact / queen zeroed):
      - pre-SFT (tau=0.3 base)
      - post-SFT (fine-tuned)
      - post-DPO
      - post-more-SFT (control)

    Also runs generation on female-ruler prompts (no queen token).
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.eval_embeddings import build_token_index, run_eval
    from language_reduction.recovery_eval import (
        mdl_analysis, GENDER_WORDS, FORGETTING_QUERY_WORDS,
    )
    from language_reduction.curriculum import TARGET_WORDS, build_token_lookup

    block_size = 128
    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)
    enc = tiktoken.get_encoding("gpt2")

    # --- Find queen token IDs ---
    lookup = build_token_lookup()
    queen_token_ids = set()
    if "queen" in lookup:
        queen_token_ids = lookup["queen"]
    print(f"Queen token IDs: {queen_token_ids}")

    # --- Model directories ---
    model_dirs = {
        "pre_sft": f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "post_sft": f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "post_dpo": f"{DATA_DIR}/models/dpo/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "post_more_sft": f"{DATA_DIR}/models/more_sft/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "ground_truth": f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}",
    }

    # --- Helpers ---
    def load_state_dict(model_dir):
        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        return sd, cfg

    def build_model(sd, cfg):
        vocab_size = sd["transformer.wte.weight"].shape[0]
        model = GPT(
            vocab_size=vocab_size,
            block_size=cfg.get("block_size", block_size),
            n_layer=cfg.get("n_layer", 2),
            n_head=cfg.get("n_head", 4),
            n_embd=cfg.get("n_embd", 128),
        )
        model.load_state_dict(sd)
        model.eval()
        return model

    def zero_queen_embedding(state_dict):
        sd = {k: v.clone() for k, v in state_dict.items()}
        wte = sd["transformer.wte.weight"]
        for qid in queen_token_ids:
            wte[qid] = 0.0
        sd["transformer.wte.weight"] = wte
        return sd

    def generate(model, prompt_ids, n_gen=40):
        ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(n_gen):
                x = torch.tensor([ids], dtype=torch.long)
                if x.size(1) >= model.block_size:
                    break
                logits, _ = model(x)
                next_tok = int(logits[0, -1].argmax())
                ids.append(next_tok)
        return ids[len(prompt_ids):]

    female_ruler_prompts = [
        "The female ruler",
        "She was a powerful woman who ruled",
        "The king and his wife, the",
        "A woman of royal authority",
        "The wife of the king was a",
    ]

    # --- Load baseline and ground truth (shared references) ---
    baseline_dir = model_dirs["pre_sft"]
    gt_dir = model_dirs["ground_truth"]

    sd_baseline, cfg_baseline = load_state_dict(baseline_dir)
    emb_baseline = sd_baseline["transformer.wte.weight"].numpy()
    emb_baseline_zeroed = zero_queen_embedding(sd_baseline)["transformer.wte.weight"].numpy()

    sd_gt, _ = load_state_dict(gt_dir)
    emb_gt = sd_gt["transformer.wte.weight"].numpy()
    emb_gt_zeroed = zero_queen_embedding(sd_gt)["transformer.wte.weight"].numpy()

    results = {}

    for label, model_dir in model_dirs.items():
        if not os.path.exists(os.path.join(model_dir, "model.pt")):
            print(f"SKIPPED {label}: {model_dir} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        sd, cfg = load_state_dict(model_dir)
        emb = sd["transformer.wte.weight"].numpy()
        model = build_model(sd, cfg)

        # --- Queen-intact evaluation ---
        eval_intact = run_eval(emb, token_index)
        mdl_intact = mdl_analysis(
            emb_baseline, emb, emb_gt,
            top_ids, token_index,
        )

        gen_intact = {}
        for prompt in female_ruler_prompts:
            prompt_ids = enc.encode(prompt)
            gen_ids = generate(model, prompt_ids)
            gen_intact[prompt] = enc.decode(gen_ids)
            print(f"  [{label}] {prompt!r} -> {prompt}{gen_intact[prompt]}")

        # --- Queen-zeroed evaluation (scaffolding removal) ---
        sd_zeroed = zero_queen_embedding(sd)
        emb_zeroed = sd_zeroed["transformer.wte.weight"].numpy()
        model_zeroed = build_model(sd_zeroed, cfg)

        eval_zeroed = run_eval(emb_zeroed, token_index)
        mdl_zeroed = mdl_analysis(
            emb_baseline_zeroed, emb_zeroed, emb_gt_zeroed,
            top_ids, token_index,
        )

        gen_zeroed = {}
        for prompt in female_ruler_prompts:
            prompt_ids = enc.encode(prompt)
            gen_ids = generate(model_zeroed, prompt_ids)
            gen_zeroed[prompt] = enc.decode(gen_ids)

        results[label] = {
            "intact": {
                "eval": eval_intact,
                "mdl_gender_means": mdl_intact.get("_gender_deltas", {}).get("means", {}),
                "mdl_control_means": mdl_intact.get("_control_deltas", {}).get("means", {}),
                "generation": gen_intact,
            },
            "queen_zeroed": {
                "eval": eval_zeroed,
                "mdl_gender_means": mdl_zeroed.get("_gender_deltas", {}).get("means", {}),
                "mdl_control_means": mdl_zeroed.get("_control_deltas", {}).get("means", {}),
                "generation": gen_zeroed,
            },
        }

        # King's neighborhood in both variants
        s2i = token_index["str_to_id"]
        for variant, e in [("intact", emb), ("zeroed", emb_zeroed)]:
            if "king" in s2i:
                kid = s2i["king"]
                norms = np.linalg.norm(e, axis=1, keepdims=True)
                emb_n = e / (norms + 1e-10)
                sims = emb_n @ emb_n[kid]
                active_set = set(top_ids.tolist())
                for i in range(len(sims)):
                    if i not in active_set or i == kid:
                        sims[i] = -np.inf
                top_k_idx = np.argsort(sims)[::-1][:10]
                neighbors = [
                    (token_index["id_to_str"][int(i)].strip(), float(sims[i]))
                    for i in top_k_idx
                ]
                results[label][f"king_neighbors_{variant}"] = neighbors
                print(f"  King neighbors ({variant}): "
                      + ", ".join(f"{w}({s:.3f})" for w, s in neighbors[:7]))

    # --- Cross-condition comparison ---
    print(f"\n{'='*60}")
    print("CROSS-CONDITION COMPARISON")
    print(f"{'='*60}")

    for label in results:
        for variant in ["intact", "queen_zeroed"]:
            ev = results[label][variant]["eval"]
            gender_acc = ev.get("analogies", {}).get("gender", {}).get("acc_add", "n/a")
            nn = ev.get("nearest_neighbors", {})
            coherences = []
            for word in GENDER_WORDS:
                if word in nn and isinstance(nn[word], dict):
                    c = nn[word].get("coherence")
                    if c is not None:
                        coherences.append(c)
            mean_coh = float(np.mean(coherences)) if coherences else 0.0
            results[label][variant]["mean_gender_coherence"] = mean_coh

            mdl_means = results[label][variant].get("mdl_gender_means", {})
            d_coh = mdl_means.get("mean_d_coherence", "n/a")
            d_tight = mdl_means.get("mean_d_tightness", "n/a")

            print(f"  {label:>16s} ({variant:>12s}): "
                  f"gender_analogy={gender_acc}  "
                  f"coherence={mean_coh:.4f}  "
                  f"Δcoherence={d_coh}  "
                  f"Δtightness={d_tight}")

    # --- Save ---
    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/scaffolding_eval.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/scaffolding_eval.json")
    return results


# ---------------------------------------------------------------------------
# Stage 21: Scaffolding persistence — does knowledge survive token removal?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def scaffolding_persistence_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    zero_embedding: bool = False,
    mode: str = "spectral",
):
    """Test whether SFT structural changes persist when scaffolding tokens are suppressed.

    For each model (baseline, post-SFT, ground truth), generates continuations
    for female-ruler prompts with all TARGET_WORD token IDs masked from the
    output logits. If post-SFT generates more coherent female-ruler content
    than baseline without access to queen/goddess/etc, the scaffolding worked.

    With zero_embedding=True, also zeros all TARGET_WORD embedding rows,
    removing scaffolding from internal computation (attention) as well as output.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.curriculum import TARGET_WORDS, build_token_lookup

    enc = tiktoken.get_encoding("gpt2")
    mode_key = resolve_mode_key(mode, source_tau)

    # --- Build suppression set: all token IDs for all TARGET_WORDS ---
    lookup = build_token_lookup()
    suppress_ids = set()
    for word in TARGET_WORDS:
        if word in lookup:
            suppress_ids.update(lookup[word])
    suppress_ids = sorted(suppress_ids)
    print(f"Suppressing {len(suppress_ids)} token IDs for {TARGET_WORDS}")
    print(f"mode={mode}, zero_embedding={zero_embedding}")
    for word in TARGET_WORDS:
        if word in lookup:
            ids = lookup[word]
            decoded = [enc.decode([tid]) for tid in ids]
            print(f"  {word}: {list(ids)} -> {decoded}")

    # --- Model directories ---
    model_dirs = {
        f"baseline ({mode_key})": f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}",
        "post-SFT": f"{DATA_DIR}/models/finetune/{mode_key}/P_{source_P}/T_{block_size}",
        "ground truth (tau=0.0)": f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}",
    }

    def load_model(model_dir):
        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        if zero_embedding:
            wte = sd["transformer.wte.weight"]
            for tid in suppress_ids:
                wte[tid] = 0.0
            sd["transformer.wte.weight"] = wte
        vocab_size = sd["transformer.wte.weight"].shape[0]
        model = GPT(
            vocab_size=vocab_size,
            block_size=cfg.get("block_size", block_size),
            n_layer=cfg.get("n_layer", 2),
            n_head=cfg.get("n_head", 4),
            n_embd=cfg.get("n_embd", 128),
        )
        model.load_state_dict(sd)
        model.eval()
        return model

    def generate_suppressed(model, prompt_ids, n_gen=50):
        """Generate with TARGET_WORD tokens suppressed from output logits."""
        ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(n_gen):
                x = torch.tensor([ids], dtype=torch.long)
                if x.size(1) >= model.block_size:
                    break
                logits, _ = model(x)
                logits[0, -1, suppress_ids] = -float('inf')
                next_tok = int(logits[0, -1].argmax())
                ids.append(next_tok)
        return ids[len(prompt_ids):]

    def top_k_suppressed(model, prompt_ids, k=15):
        """Top-k next-token predictions with TARGET_WORDs suppressed."""
        with torch.no_grad():
            x = torch.tensor([prompt_ids], dtype=torch.long)
            logits, _ = model(x)
            logits[0, -1, suppress_ids] = -float('inf')
            probs = torch.softmax(logits[0, -1], dim=-1)
            topk = torch.topk(probs, k)
        return list(zip(
            [enc.decode([int(t)]) for t in topk.indices],
            [float(p) for p in topk.values],
        ))

    # --- Prompts ---
    generation_prompts = [
        "She was a powerful woman who ruled",
        "The female ruler",
        "The king and his wife, the",
        "A woman of royal authority",
        "The wife of the king was a",
        "In the kingdom, the woman who held power",
        "The king died and his wife became",
    ]

    # Prompts for top-k analysis — these end at a position where
    # the next token should reveal female-ruler knowledge
    topk_prompts = [
        "The king and his wife, the",
        "She was a powerful woman who ruled the",
        "The wife of the king was a powerful",
        "In ancient times, the female ruler was called a",
    ]

    # --- Load models ---
    models = {}
    for label, d in model_dirs.items():
        if os.path.exists(os.path.join(d, "model.pt")):
            models[label] = load_model(d)
            print(f"Loaded: {label}")
        else:
            print(f"SKIPPED (not found): {label} at {d}")

    results = {}

    # --- Generation comparison ---
    print(f"\n{'='*70}")
    print("GENERATION WITH TARGET TOKENS SUPPRESSED")
    print(f"{'='*70}")

    for prompt in generation_prompts:
        prompt_ids = enc.encode(prompt)
        print(f"\nPROMPT: {prompt!r}")

        for label, model in models.items():
            gen_ids = generate_suppressed(model, prompt_ids)
            gen_text = enc.decode(gen_ids)
            print(f"  [{label}]")
            print(f"    {prompt}{gen_text}")

            if label not in results:
                results[label] = {"generation": {}, "top_k": {}}
            results[label]["generation"][prompt] = gen_text

    # --- Top-k analysis ---
    print(f"\n{'='*70}")
    print("TOP-K NEXT TOKEN PREDICTIONS (TARGET TOKENS SUPPRESSED)")
    print(f"{'='*70}")

    for prompt in topk_prompts:
        prompt_ids = enc.encode(prompt)
        print(f"\nPROMPT: {prompt!r}")

        for label, model in models.items():
            topk = top_k_suppressed(model, prompt_ids)
            print(f"  [{label}]")
            for tok_str, prob in topk[:10]:
                print(f"    {tok_str!r:15s} {prob:.4f}")

            results[label]["top_k"][prompt] = [
                {"token": t, "prob": p} for t, p in topk
            ]

    # --- Save ---
    results["_config"] = {
        "zero_embedding": zero_embedding,
        "suppress_ids": suppress_ids,
        "target_words": TARGET_WORDS,
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    suffix = "_zeroed" if zero_embedding else ""
    fname = f"scaffolding_persistence{suffix}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nSaved to {results_dir}/{fname}")
    return results


# ---------------------------------------------------------------------------
# Queen forward pass: use the actual queen token in generation
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def queen_forward_pass(source_P: int = 100_000_000, block_size: int = 128):
    """Feed prompts containing 'queen' through three models and compare.

    Models:
      - tau=0.0 (ground truth: queen was in training data)
      - tau=0.3 (queen removed: embedding is untrained noise)
      - tau=0.3 fine-tuned (queen re-introduced via curriculum)

    Uses the actual queen token — no injection tricks.
    """
    import os, json, torch
    import tiktoken
    from language_reduction.model import GPT

    enc = tiktoken.get_encoding("gpt2")

    prompts = [
        "The queen",
        "A queen is",
        "The king and the queen",
        "She was a powerful queen who",
    ]

    def load_model(model_dir):
        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]
        model = GPT(
            vocab_size=vocab_size,
            block_size=cfg.get("block_size", block_size),
            n_layer=cfg.get("n_layer", 2),
            n_head=cfg.get("n_head", 4),
            n_embd=cfg.get("n_embd", 128),
        )
        model.load_state_dict(sd)
        model.eval()
        return model

    def generate(model, prompt_ids, n_gen=40):
        ids = list(prompt_ids)
        with torch.no_grad():
            for _ in range(n_gen):
                x = torch.tensor([ids], dtype=torch.long)
                if x.size(1) >= model.block_size:
                    break
                logits, _ = model(x)
                next_tok = int(logits[0, -1].argmax())
                ids.append(next_tok)
        return ids[len(prompt_ids):]

    def top_k_after(model, prompt_ids, k=10):
        with torch.no_grad():
            x = torch.tensor([prompt_ids], dtype=torch.long)
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1], dim=-1)
            topk = torch.topk(probs, k)
        return list(zip(
            [enc.decode([int(t)]) for t in topk.indices],
            [float(p) for p in topk.values],
        ))

    models = {}
    model_dirs = {
        "tau=0.0 (ground truth)": f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}",
        "tau=0.3 (queen removed)": f"{DATA_DIR}/models/tau_0.300/P_{source_P}/T_{block_size}",
        "tau=0.3 fine-tuned": f"{DATA_DIR}/models/finetune/tau_0.300/P_{source_P}/T_{block_size}",
    }

    for label, d in model_dirs.items():
        if os.path.exists(os.path.join(d, "model.pt")):
            models[label] = load_model(d)
            print(f"Loaded: {label}")
        else:
            print(f"SKIPPED (not found): {label} at {d}")

    queen_id = enc.encode(" queen")
    print(f"\nQueen token IDs: {queen_id} -> {[enc.decode([t]) for t in queen_id]}")

    for prompt in prompts:
        prompt_ids = enc.encode(prompt)
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt!r}")
        print(f"Token IDs: {prompt_ids}")
        print(f"{'='*70}")

        for label, model in models.items():
            gen_ids = generate(model, prompt_ids)
            gen_text = enc.decode(gen_ids)
            top = top_k_after(model, prompt_ids)

            print(f"\n  [{label}]")
            print(f"    Generated: {prompt}{gen_text}")
            print(f"    Top-10 next token predictions:")
            for tok_str, prob in top:
                print(f"      {tok_str!r:15s} {prob:.4f}")
