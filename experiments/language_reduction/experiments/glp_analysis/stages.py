from language_reduction.shared import app, volume, DATA_DIR, image


# ---------------------------------------------------------------------------
# Stage 22: Train GLP on all-layer activations
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def train_glp_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    max_activations: int = 1_000_000,
    glp_d_model: int = 256,
    glp_d_mlp: int = 512,
    glp_n_layers: int = 3,
    glp_n_steps: int = 20_000,
    glp_batch_size: int = 512,
    glp_lr: float = 5e-5,
    t_start: float = 0.3,
    num_denoise_steps: int = 10,
    model_condition: str = "baseline",
):
    """Train a GLP on [h0; h1] activations from a trained model.

    model_condition selects which model checkpoint to use:
      - "baseline"  : tau_{tau}/P_{P}/T_{block_size} (pre-SFT)
      - "finetuned" : finetune/tau_{tau}/P_{P}/T_{block_size} (post-SFT)
      - "ground"    : tau_0.000/P_{P}/T_{block_size} (original corpus)
    """
    import os, json, glob, torch
    import torch.nn.functional as F
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.glp import (
        extract_activations, train_glp, GLPDenoiser,
        glp_denoise, compute_residuals,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Locate model checkpoint ---
    if model_condition == "finetuned":
        model_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    elif model_condition == "ground":
        model_dir = f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}"
    else:
        model_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"

    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)
    act_dim = n_layer * n_embd

    model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
    ).to(device)
    model.load_state_dict(sd)
    print(f"Loaded model from {model_dir} (condition={model_condition})")

    # --- Load training data ---
    if source_tau == 0.0:
        data_dir = f"{DATA_DIR}/tokens"
    else:
        data_dir = f"{DATA_DIR}/denoised/tau_{source_tau:.3f}"

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= max_activations + block_size:
            break
    data = np.concatenate(all_tokens)
    data = torch.from_numpy(data.astype(np.int64))
    print(f"Loaded {len(data):,} tokens for activation extraction")

    # --- Extract activations ---
    print(f"Extracting [h0; h1] activations (act_dim={act_dim})...")
    activations = extract_activations(
        model, data, block_size, batch_size=64,
        max_tokens=max_activations, device=device,
    )
    print(f"Extracted {activations.shape[0]:,} activations, shape {activations.shape}")

    # --- Train GLP ---
    print(f"Training GLP: d_model={glp_d_model}, d_mlp={glp_d_mlp}, "
          f"n_layers={glp_n_layers}, n_steps={glp_n_steps}")
    glp_sd, act_stats, loss_curve = train_glp(
        activations,
        act_dim=act_dim, d_model=glp_d_model, d_mlp=glp_d_mlp,
        n_layers=glp_n_layers, n_steps=glp_n_steps,
        batch_size=glp_batch_size, lr=glp_lr,
        device=device, seed=42,
    )

    # --- Quick quality check: reconstruction cosine on a sample ---
    glp_model = GLPDenoiser(act_dim, glp_d_model, glp_d_mlp, glp_n_layers)
    glp_model.load_state_dict(glp_sd)
    glp_model = glp_model.to(device).eval()

    sample_acts = activations[:min(2048, len(activations))]
    mean = act_stats["mean"].to(device)
    std = act_stats["std"].to(device)
    sample_std = (sample_acts.to(device) - mean) / std
    manifold_std = glp_denoise(glp_model, sample_std, t_start, num_denoise_steps)
    manifold = manifold_std * std + mean

    cos_sims = F.cosine_similarity(sample_acts.to(device), manifold, dim=-1)
    recon_cosine = float(cos_sims.mean())
    residual_norms = (sample_acts.to(device) - manifold).norm(dim=-1)
    mean_residual = float(residual_norms.mean())
    print(f"Quality check: recon_cosine={recon_cosine:.4f}, "
          f"mean_residual_norm={mean_residual:.4f}")

    # --- Save ---
    save_dir = os.path.join(model_dir, f"glp_{model_condition}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(glp_sd, os.path.join(save_dir, "glp_model.pt"))
    torch.save(act_stats, os.path.join(save_dir, "glp_stats.pt"))

    result = {
        "source_tau": source_tau,
        "source_P": source_P,
        "model_condition": model_condition,
        "n_activations": activations.shape[0],
        "act_dim": act_dim,
        "config": {
            "act_dim": act_dim,
            "d_model": glp_d_model,
            "d_mlp": glp_d_mlp,
            "n_layers": glp_n_layers,
        },
        "n_steps": glp_n_steps,
        "batch_size": glp_batch_size,
        "lr": glp_lr,
        "t_start": t_start,
        "num_denoise_steps": num_denoise_steps,
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1][1],
        "recon_cosine": recon_cosine,
        "mean_residual_norm": mean_residual,
    }
    with open(os.path.join(save_dir, "glp_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    volume.commit()
    print(f"GLP saved to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage 22b: Train GLP on ln_f(h1) activations (embedding-space GLP)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def train_glp_embed_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    max_activations: int = 1_000_000,
    glp_d_model: int = 256,
    glp_d_mlp: int = 512,
    glp_n_layers: int = 3,
    glp_n_steps: int = 20_000,
    glp_batch_size: int = 512,
    glp_lr: float = 5e-5,
    t_start: float = 0.3,
    num_denoise_steps: int = 10,
):
    """Train a GLP on ln_f(h1) activations from the tied-weights model.

    Unlike the original GLP (which trains on [h0; h1] in 256-D), this trains
    on the 128-D post-LayerNorm output of the final block. With tied weights,
    this is the same space as the embedding space — GLP residuals can be used
    directly as embedding initializations without any projection step.
    """
    import os, json, glob, torch
    import torch.nn.functional as F
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.glp import (
        extract_lnf_h1_activations, train_glp, GLPDenoiser, glp_denoise,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = (f"{DATA_DIR}/models/tau_{source_tau:.3f}"
                 f"/P_{source_P}/T_{block_size}_tied")

    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)
    act_dim = n_embd

    model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
        tie_weights=True,
    ).to(device)
    model.load_state_dict(sd)
    print(f"Loaded tied-weights model from {model_dir}")

    # --- Load training data ---
    if source_tau == 0.0:
        data_dir = f"{DATA_DIR}/tokens"
    else:
        data_dir = f"{DATA_DIR}/denoised/tau_{source_tau:.3f}"

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= max_activations + block_size:
            break
    data = np.concatenate(all_tokens)
    data = torch.from_numpy(data.astype(np.int64))
    print(f"Loaded {len(data):,} tokens for activation extraction")

    # --- Extract ln_f(h1) activations ---
    print(f"Extracting ln_f(h1) activations (act_dim={act_dim})...")
    activations = extract_lnf_h1_activations(
        model, data, block_size, batch_size=64,
        max_tokens=max_activations, device=device,
    )
    print(f"Extracted {activations.shape[0]:,} activations, shape {activations.shape}")

    del model
    torch.cuda.empty_cache()

    # --- Train GLP ---
    print(f"Training GLP: d_model={glp_d_model}, d_mlp={glp_d_mlp}, "
          f"n_layers={glp_n_layers}, n_steps={glp_n_steps}")
    glp_sd, act_stats, loss_curve = train_glp(
        activations,
        act_dim=act_dim, d_model=glp_d_model, d_mlp=glp_d_mlp,
        n_layers=glp_n_layers, n_steps=glp_n_steps,
        batch_size=glp_batch_size, lr=glp_lr,
        device=device, seed=42,
    )

    # --- Quality check ---
    glp_model = GLPDenoiser(act_dim, glp_d_model, glp_d_mlp, glp_n_layers)
    glp_model.load_state_dict(glp_sd)
    glp_model = glp_model.to(device).eval()

    sample_acts = activations[:min(2048, len(activations))]
    mean = act_stats["mean"].to(device)
    std = act_stats["std"].to(device)
    sample_std = (sample_acts.to(device) - mean) / std
    manifold_std = glp_denoise(glp_model, sample_std, t_start, num_denoise_steps)
    manifold = manifold_std * std + mean

    cos_sims = F.cosine_similarity(sample_acts.to(device), manifold, dim=-1)
    recon_cosine = float(cos_sims.mean())
    residual_norms = (sample_acts.to(device) - manifold).norm(dim=-1)
    mean_residual = float(residual_norms.mean())

    # Compare residuals against wte rows to check embedding-space alignment
    wte = sd["transformer.wte.weight"].to(device)
    sample_resids = sample_acts.to(device) - manifold
    wte_cos = F.cosine_similarity(
        sample_resids[:100].unsqueeze(1),
        wte.unsqueeze(0),
        dim=-1,
    )
    max_wte_cos = wte_cos.max(dim=-1).values
    mean_max_wte_cos = float(max_wte_cos.mean())

    print(f"Quality check: recon_cosine={recon_cosine:.4f}, "
          f"mean_residual_norm={mean_residual:.4f}")
    print(f"Embedding-space check: mean max cos(residual, wte_row)={mean_max_wte_cos:.4f}")

    # --- Save ---
    save_dir = os.path.join(model_dir, "glp_embed")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(glp_sd, os.path.join(save_dir, "glp_model.pt"))
    torch.save(act_stats, os.path.join(save_dir, "glp_stats.pt"))

    result = {
        "source_tau": source_tau,
        "source_P": source_P,
        "model_type": "tied_weights",
        "activation_type": "ln_f_h1",
        "n_activations": activations.shape[0],
        "act_dim": act_dim,
        "config": {
            "act_dim": act_dim,
            "d_model": glp_d_model,
            "d_mlp": glp_d_mlp,
            "n_layers": glp_n_layers,
        },
        "n_steps": glp_n_steps,
        "batch_size": glp_batch_size,
        "lr": glp_lr,
        "t_start": t_start,
        "num_denoise_steps": num_denoise_steps,
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1][1],
        "recon_cosine": recon_cosine,
        "mean_residual_norm": mean_residual,
        "mean_max_wte_cosine": mean_max_wte_cos,
    }
    with open(os.path.join(save_dir, "glp_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    volume.commit()
    print(f"GLP (embedding-space) saved to {save_dir}")
    return result


# ---------------------------------------------------------------------------
# Stage 23: GLP directional residual analysis (pre/post scaffolding)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def glp_residuals_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    t_start: float = 0.3,
    num_denoise_steps: int = 10,
):
    """Compute directional residuals for baseline and post-SFT models.

    Uses the BASELINE GLP (trained on pre-SFT activations) for both models,
    so differences in residuals reflect how the model's activations moved
    relative to the original manifold.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.glp import (
        GLPDenoiser, extract_activations, compute_residuals, load_glp,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load GLP (trained on baseline model) ---
    base_model_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    glp_dir = os.path.join(base_model_dir, "glp_baseline")
    glp_model, act_stats, glp_cfg = load_glp(glp_dir, device=device)
    glp_model = glp_model.to(device).eval()
    act_dim = glp_cfg["act_dim"]
    print(f"Loaded GLP from {glp_dir}")

    # --- Probe contexts ---
    enc = tiktoken.get_encoding("gpt2")
    probe_texts = [
        "The king and his wife, the",
        "She was a powerful woman who ruled",
        "The female ruler of the kingdom",
        "A woman of royal authority and",
        "The prince married a woman of noble birth, and she became",
        "In ancient times, kings and their consorts would",
        "The throne was inherited by the eldest daughter of the",
        "The emperor and the empress ruled together over the",
    ]
    probe_tokens = [enc.encode(t) for t in probe_texts]
    max_len = max(len(t) for t in probe_tokens)
    padded = [t + [enc.encode(" ")[0]] * (max_len - len(t)) for t in probe_tokens]
    probe_ids = torch.tensor(padded, dtype=torch.long)

    # --- Load and probe each model condition ---
    conditions = {
        "baseline": f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "finetuned": f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "ground_truth": f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}",
    }

    all_results = {}
    for cond_name, model_dir in conditions.items():
        if not os.path.exists(os.path.join(model_dir, "model.pt")):
            print(f"Skipping {cond_name}: no model at {model_dir}")
            continue

        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]

        model = GPT(
            vocab_size=vocab_size, block_size=block_size,
            n_layer=cfg.get("n_layer", 2), n_head=cfg.get("n_head", 4),
            n_embd=cfg.get("n_embd", 128),
        ).to(device)
        model.load_state_dict(sd)
        model.eval()

        # Extract activations on probe contexts
        layer_outputs = []
        hooks = []
        for block in model.transformer.h:
            hooks.append(block.register_forward_hook(
                lambda _m, _i, o, s=layer_outputs: s.append(o.detach())
            ))
        with torch.no_grad():
            model(probe_ids.to(device))
        for h in hooks:
            h.remove()

        # (n_probes, max_len, act_dim)
        acts = torch.cat(layer_outputs, dim=-1)
        acts_flat = acts.reshape(-1, act_dim)

        residuals, manifold = compute_residuals(
            glp_model, act_stats, acts_flat,
            t_start=t_start, num_steps=num_denoise_steps,
        )

        residual_norms = residuals.norm(dim=-1).reshape(len(probe_texts), max_len)
        cos_sims = torch.nn.functional.cosine_similarity(
            acts_flat, manifold, dim=-1
        ).reshape(len(probe_texts), max_len)

        cond_result = {
            "per_probe_mean_residual": residual_norms.mean(dim=-1).tolist(),
            "per_probe_mean_cosine": cos_sims.mean(dim=-1).tolist(),
            "overall_mean_residual": float(residual_norms.mean()),
            "overall_mean_cosine": float(cos_sims.mean()),
        }
        all_results[cond_name] = cond_result

        print(f"\n{cond_name}:")
        print(f"  mean residual norm: {cond_result['overall_mean_residual']:.4f}")
        print(f"  mean recon cosine:  {cond_result['overall_mean_cosine']:.4f}")
        for j, text in enumerate(probe_texts):
            print(f"    {text[:50]:50s}  "
                  f"resid={cond_result['per_probe_mean_residual'][j]:.4f}  "
                  f"cos={cond_result['per_probe_mean_cosine'][j]:.4f}")

        del model
        torch.cuda.empty_cache()

    # --- Cross-condition comparisons ---
    if "baseline" in all_results and "finetuned" in all_results:
        bl = all_results["baseline"]
        ft = all_results["finetuned"]
        print("\n=== Scaffolding effect on manifold distance ===")
        for j, text in enumerate(probe_texts):
            delta = ft["per_probe_mean_residual"][j] - bl["per_probe_mean_residual"][j]
            print(f"  {text[:50]:50s}  Δresid={delta:+.4f}")
        print(f"  Overall Δresid: "
              f"{ft['overall_mean_residual'] - bl['overall_mean_residual']:+.4f}")

    # --- Save ---
    results = {
        "source_tau": source_tau,
        "source_P": source_P,
        "t_start": t_start,
        "num_denoise_steps": num_denoise_steps,
        "probe_texts": probe_texts,
        "conditions": all_results,
    }
    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/glp_residuals_tau{source_tau:.1f}.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {results_dir}/glp_residuals_tau{source_tau:.1f}.json")
    return results


# ---------------------------------------------------------------------------
# Stage 24: GLP residual semantics — verbalize & probe residual geometry
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def glp_semantics_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    t_start: float = 0.3,
    num_denoise_steps: int = 10,
    n_denoise_avg: int = 5,
):
    """Semantic analysis of GLP residuals at female-royalty positions.

    For each probe prompt, computes the GLP residual at the last token and:
      1. Decodes the residual through the model's own unembedding — what
         tokens does the off-manifold component encode?
      2. Measures velocity field curvature in the residual direction vs.
         random — does the manifold "know about" this direction?
      3. Checks cross-prompt alignment — is there a consistent "queen"
         direction in residual space?
      4. Compares residual magnitude (queen prompts vs. controls) —
         is the model farther from the manifold at these positions?

    n_denoise_avg: average residuals over this many noise draws (the
    denoising process is stochastic; averaging stabilizes the direction).
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.glp import (
        GLPDenoiser, glp_denoise, velocity_curvature, load_glp,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    # --- Load model ---
    model_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)
    act_dim = n_layer * n_embd

    lm = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
    ).to(device)
    lm.load_state_dict(sd)
    lm.eval()
    print(f"Loaded LM from {model_dir}")

    # --- Load GLP ---
    glp_dir = os.path.join(model_dir, "glp_baseline")
    glp_model, act_stats, glp_cfg = load_glp(glp_dir, device=device)
    glp_model = glp_model.to(device).eval()
    mean = act_stats["mean"].to(device)
    std = act_stats["std"].to(device)
    print(f"Loaded GLP from {glp_dir}")

    # --- Prompts: four categories to decompose queen = gender × royalty ---
    queen_prompts = [
        "The king and his wife, the",
        "She was a powerful woman who ruled",
        "The female ruler of the kingdom",
        "A woman of royal authority and",
        "The prince married a woman of noble birth, and she became the",
        "The throne was inherited by the eldest daughter of the",
        "The emperor and his wife ruled together over the",
        "She wore a crown and ruled the kingdom as its",
        "The king's daughter became the new",
        "The most powerful woman in the kingdom was the",
        "In the royal court, the king and his",
        "In ancient times, kings and their consorts would",
    ]
    gender_prompts = [
        "The mother cooked dinner for her",
        "She was a young woman who worked",
        "The girl walked to school with her",
        "Her sister told her that the",
        "The woman opened the door and",
        "She gave birth to a baby",
        "The grandmother told stories about the",
        "His wife went to the market and",
        "The daughter learned to read from her",
        "She raised her children in the",
        "The young girl dreamed of becoming a",
        "Her mother always said that the",
    ]
    royalty_prompts = [
        "The king ruled his kingdom with",
        "The prince inherited the throne from his",
        "The emperor's court was filled with",
        "The royal palace was built by the",
        "He was crowned king after the death of his",
        "The kingdom was governed by a powerful",
        "The throne room was decorated with",
        "The king's army marched through the",
        "The prince was trained in the arts of",
        "The royal family held a great",
        "In the court of the king, the",
        "The ruler of the empire decreed that",
    ]
    control_prompts = [
        "The sun rises in the east and",
        "He opened the door and walked into the",
        "The scientists discovered that the",
        "The river flows through the center of the",
        "She studied mathematics at the",
        "The president of the country announced",
        "The farmer planted seeds in the",
        "The book was written by a",
        "The city was founded in the",
        "The teacher explained the concept to the",
        "The car drove down the road toward the",
        "The dog ran across the field to the",
    ]

    all_prompts = (
        [(p, "queen") for p in queen_prompts] +
        [(p, "gender") for p in gender_prompts] +
        [(p, "royalty") for p in royalty_prompts] +
        [(p, "control") for p in control_prompts]
    )

    # --- Helper: extract last-position [h0; h1] and compute averaged residual ---
    def get_last_pos_residual(prompt_text):
        tokens = enc.encode(prompt_text)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # extract per-layer outputs
        layer_outs = []
        hooks = []
        for block in lm.transformer.h:
            hooks.append(block.register_forward_hook(
                lambda _m, _i, o, s=layer_outs: s.append(o.detach())
            ))
        with torch.no_grad():
            lm(ids)
        for h in hooks:
            h.remove()

        # [h0; h1] at last position
        act = torch.cat([lo[:, -1, :] for lo in layer_outs], dim=-1)  # (1, act_dim)
        act_std = (act - mean) / std

        # average residual over multiple noise draws
        residuals_std = []
        for _ in range(n_denoise_avg):
            manifold_std = glp_denoise(glp_model, act_std, t_start, num_denoise_steps)
            residuals_std.append(act_std - manifold_std)
        avg_resid_std = torch.stack(residuals_std).mean(dim=0)  # (1, act_dim)

        # un-standardize the manifold projection (use the mean residual)
        avg_manifold_std = act_std - avg_resid_std
        manifold_act = avg_manifold_std * std + mean

        return act, manifold_act, avg_resid_std, layer_outs

    # --- Helper: decode h1 through ln_f + lm_head ---
    def decode_h1(h1_vec):
        """Feed a single h1 vector through the final LN + lm_head."""
        with torch.no_grad():
            normed = lm.transformer.ln_f(h1_vec)
            logits = lm.lm_head(normed)
        return logits.squeeze(0)

    # --- Process all prompts ---
    results_list = []
    resids_by_cat = {}

    for prompt_text, category in all_prompts:
        act, manifold_act, resid_std, layer_outs = get_last_pos_residual(prompt_text)

        # decompose residual into h0 and h1 components
        resid_unnorm = resid_std * std
        h0_resid = resid_unnorm[0, :n_embd]
        h1_resid = resid_unnorm[0, n_embd:]
        total_norm = resid_unnorm[0].norm().item()
        h0_norm = h0_resid.norm().item()
        h1_norm = h1_resid.norm().item()

        # --- Token decoding ---
        h1_original = act[0, n_embd:]  # last n_embd dims = h1
        h1_manifold = manifold_act[0, n_embd:]
        logits_orig = decode_h1(h1_original.unsqueeze(0))
        logits_mani = decode_h1(h1_manifold.unsqueeze(0))
        logits_diff = logits_orig - logits_mani

        # top-k from each
        k = 15
        def topk_tokens(logits, k):
            probs = torch.softmax(logits, dim=-1)
            vals, idxs = probs.topk(k)
            return [(enc.decode([int(i)]), float(v)) for i, v in zip(idxs, vals)]

        top_original = topk_tokens(logits_orig, k)
        top_manifold = topk_tokens(logits_mani, k)

        # tokens the residual adds (highest logit increase)
        diff_vals, diff_idxs = logits_diff.topk(k)
        top_residual = [(enc.decode([int(i)]), float(v)) for i, v in zip(diff_idxs, diff_vals)]

        # --- Velocity field curvature ---
        curv_dir, curv_rand, ratio = velocity_curvature(
            glp_model, (act - mean) / std, resid_std,
            t=0.5, eps=0.1, n_random=50,
        )

        # collect for cross-prompt alignment
        resids_by_cat.setdefault(category, []).append(resid_std.squeeze(0))

        entry = {
            "prompt": prompt_text,
            "category": category,
            "residual_norm": total_norm,
            "h0_residual_norm": h0_norm,
            "h1_residual_norm": h1_norm,
            "h1_energy_fraction": h1_norm**2 / (total_norm**2 + 1e-8),
            "top_original": top_original,
            "top_manifold": top_manifold,
            "top_residual_tokens": top_residual,
            "velocity_curvature": float(curv_dir.mean()),
            "velocity_curvature_random": float(curv_rand.mean()),
            "curvature_ratio": float(ratio.mean()),
        }
        results_list.append(entry)

        print(f"\n{'='*60}")
        print(f"[{category}] {prompt_text}")
        print(f"  residual norm: {total_norm:.4f} "
              f"(h0: {h0_norm:.4f}, h1: {h1_norm:.4f}, "
              f"h1 energy: {entry['h1_energy_fraction']:.1%})")
        print(f"  curvature ratio: {entry['curvature_ratio']:.2f}")
        print(f"  model predicts:   {', '.join(f'{t}({p:.3f})' for t, p in top_original[:5])}")
        print(f"  manifold predicts:{', '.join(f'{t}({p:.3f})' for t, p in top_manifold[:5])}")
        print(f"  residual encodes: {', '.join(f'{t}({v:+.2f})' for t, v in top_residual[:8])}")

    # --- Cross-category geometry ---
    print(f"\n{'='*60}")
    print("CROSS-CATEGORY GEOMETRY")
    print(f"{'='*60}")

    cat_stacks = {c: torch.stack(vs) for c, vs in resids_by_cat.items()}

    def mean_pairwise_cosine(vecs):
        n = vecs.shape[0]
        if n < 2:
            return 0.0
        normed = F.normalize(vecs, dim=-1)
        sim = normed @ normed.T
        mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
        return float(sim[mask].mean())

    # within-category alignment
    print("\n  Within-category alignment (mean pairwise cosine):")
    within_align = {}
    for cat, vecs in cat_stacks.items():
        a = mean_pairwise_cosine(vecs)
        within_align[cat] = a
        print(f"    {cat:10s}: {a:.4f}  (n={vecs.shape[0]})")

    # mean directions per category
    cat_means = {}
    for cat, vecs in cat_stacks.items():
        cat_means[cat] = F.normalize(vecs.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

    # between-category cosine of mean directions
    cats = sorted(cat_means.keys())
    print("\n  Between-category cosine (mean direction vs mean direction):")
    between = {}
    for i, c1 in enumerate(cats):
        for c2 in cats[i+1:]:
            cos = float(F.cosine_similarity(
                cat_means[c1].unsqueeze(0), cat_means[c2].unsqueeze(0)
            ))
            between[f"{c1}_vs_{c2}"] = cos
            print(f"    {c1:10s} vs {c2:10s}: {cos:+.4f}")

    # --- KEY TEST: queen direction decomposed into gender + royalty ---
    print(f"\n{'='*60}")
    print("ENTANGLEMENT DECOMPOSITION: queen = gender x royalty?")
    print(f"{'='*60}")

    q_mean = cat_means.get("queen")
    g_mean = cat_means.get("gender")
    r_mean = cat_means.get("royalty")
    c_mean = cat_means.get("control")

    if q_mean is not None and g_mean is not None and r_mean is not None:
        # project queen direction onto gender and royalty axes
        q_on_g = float(F.cosine_similarity(q_mean.unsqueeze(0), g_mean.unsqueeze(0)))
        q_on_r = float(F.cosine_similarity(q_mean.unsqueeze(0), r_mean.unsqueeze(0)))
        q_on_c = float(F.cosine_similarity(q_mean.unsqueeze(0), c_mean.unsqueeze(0)))
        g_on_r = float(F.cosine_similarity(g_mean.unsqueeze(0), r_mean.unsqueeze(0)))

        print(f"\n  Queen mean direction projects onto:")
        print(f"    gender  axis: {q_on_g:+.4f}")
        print(f"    royalty axis: {q_on_r:+.4f}")
        print(f"    control axis: {q_on_c:+.4f}")
        print(f"  Gender-royalty baseline cosine: {g_on_r:+.4f}")

        # per-queen-prompt projection onto gender and royalty
        q_resids = cat_stacks["queen"]
        q_normed = F.normalize(q_resids, dim=-1)
        proj_g = (q_normed * g_mean.unsqueeze(0)).sum(dim=-1)
        proj_r = (q_normed * r_mean.unsqueeze(0)).sum(dim=-1)
        proj_c = (q_normed * c_mean.unsqueeze(0)).sum(dim=-1)

        print(f"\n  Per-queen-prompt projections:")
        print(f"    {'prompt':<55s} gender  royalty control")
        for i, (p, _) in enumerate(x for x in all_prompts if x[1] == "queen"):
            print(f"    {p[:55]:<55s} {proj_g[i]:+.3f}   {proj_r[i]:+.3f}   {proj_c[i]:+.3f}")

        print(f"\n  Mean projections:  gender={float(proj_g.mean()):+.4f}  "
              f"royalty={float(proj_r.mean()):+.4f}  "
              f"control={float(proj_c.mean()):+.4f}")

        # how much of queen variance is explained by gender+royalty subspace?
        # build 2-d subspace from gender and royalty mean directions
        gr_basis = torch.stack([g_mean, r_mean])  # (2, act_dim)
        # orthogonalize via QR
        Q, _ = torch.linalg.qr(gr_basis.T)  # (act_dim, 2)
        q_centered = q_resids - q_resids.mean(dim=0)
        q_proj_gr = q_centered @ Q @ Q.T  # project onto gender-royalty plane
        q_resid_gr = q_centered - q_proj_gr  # orthogonal complement

        var_total = float((q_centered ** 2).sum())
        var_in_gr = float((q_proj_gr ** 2).sum())
        var_explained = var_in_gr / (var_total + 1e-8)

        print(f"\n  Gender-royalty 2D subspace explains {var_explained:.1%} of queen residual variance")
        print(f"    (chance for 2D in {act_dim}D: {2/act_dim:.1%})")

        # also project gender-only and royalty-only onto this subspace
        for cat_name, cat_vecs in [("gender", cat_stacks.get("gender")),
                                    ("royalty", cat_stacks.get("royalty")),
                                    ("control", cat_stacks.get("control"))]:
            if cat_vecs is None:
                continue
            cc = cat_vecs - cat_vecs.mean(dim=0)
            cp = cc @ Q @ Q.T
            cv = float((cp ** 2).sum()) / (float((cc ** 2).sum()) + 1e-8)
            print(f"    {cat_name:10s} variance in same 2D subspace: {cv:.1%}")

        # cross-category similarity: each queen prompt vs all gender, all royalty
        print(f"\n  Cross-category mean cosine (individual prompts → category mean):")
        for cat_name in ["gender", "royalty", "control"]:
            if cat_name not in cat_means:
                continue
            cm = cat_means[cat_name]
            for src_cat in ["queen", "gender", "royalty", "control"]:
                if src_cat not in cat_stacks:
                    continue
                sims = F.cosine_similarity(
                    F.normalize(cat_stacks[src_cat], dim=-1),
                    cm.unsqueeze(0),
                )
                print(f"    {src_cat:10s} → {cat_name:10s}: {float(sims.mean()):+.4f}  "
                      f"(std={float(sims.std()):.4f})")

    # --- Norms and curvature by category ---
    print(f"\n{'='*60}")
    print("SUMMARY BY CATEGORY")
    print(f"{'='*60}")
    geometry = {"within_alignment": within_align, "between_cosines": between}
    for cat in cats:
        norms = [r["residual_norm"] for r in results_list if r["category"] == cat]
        curvs = [r["curvature_ratio"] for r in results_list if r["category"] == cat]
        mn = float(np.mean(norms))
        mc = float(np.mean(curvs))
        print(f"  {cat:10s}:  mean_norm={mn:.4f}  mean_curv_ratio={mc:.2f}  n={len(norms)}")
        geometry[f"{cat}_mean_norm"] = mn
        geometry[f"{cat}_mean_curv"] = mc

    if q_mean is not None and g_mean is not None and r_mean is not None:
        geometry["queen_on_gender"] = q_on_g
        geometry["queen_on_royalty"] = q_on_r
        geometry["queen_on_control"] = q_on_c
        geometry["gender_royalty_cosine"] = g_on_r
        geometry["gender_royalty_subspace_explains_queen"] = var_explained
        geometry["per_queen_proj_gender"] = proj_g.tolist()
        geometry["per_queen_proj_royalty"] = proj_r.tolist()

    # --- Save ---
    results = {
        "source_tau": source_tau,
        "source_P": source_P,
        "t_start": t_start,
        "n_denoise_avg": n_denoise_avg,
        "probes": results_list,
        "geometry": geometry,
    }
    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    fname = f"glp_semantics_tau{source_tau:.1f}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {results_dir}/{fname}")
    return results
