from language_reduction.shared import app, volume, DATA_DIR, NumpyEncoder


QUEEN_PREDICTION_PROMPTS = [
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

QUEEN_INPUT_PROMPTS = [
    "The queen ruled",
    "She was a powerful queen who",
    "The king and the queen",
    "The queen of the kingdom",
]

CONTROL_PROMPTS = [
    "The sun rises in the east and",
    "He opened the door and walked into the",
    "The scientists discovered that the",
    "The river flows through the center of the",
    "She studied mathematics at the",
    "The farmer planted seeds in the",
    "The book was written by a",
    "The car drove down the road toward the",
    "The dog ran across the field to the",
    "The teacher explained the concept to the",
    "Two plus two equals",
    "The computer program was designed to",
]

FEMALE_WORDS = ["she", "her", "woman", "mother", "daughter", "wife", "girl", "sister"]
MALE_WORDS = ["he", "his", "man", "father", "son", "husband", "boy", "brother"]

# 2x2 factorial priming prefixes for ICL grokking experiment.
# Cross gender (female/male) with royalty (royal/non-royal).
# Each ~55-60 tokens, leaving room for prediction prompts within block_size=128.
FEMALE_ROYAL_PREFIX = (
    "The wife of the king sat upon her throne. She wore the royal crown "
    "and held a golden scepter. As a woman of noble birth, the daughter "
    "of emperors, she ruled her kingdom with authority. The princess "
    "became a powerful ruler, a female monarch beloved by her people."
)
NEUTRAL_PREFIX = (
    "The farmer worked in his field throughout the long summer. He "
    "planted seeds and watered the crops every morning. As a man of "
    "the countryside, the son of laborers, he tended his farm with "
    "dedication. The merchant became a successful trader, a skilled "
    "businessman known by his neighbors."
)
MALE_ROYAL_PREFIX = (
    "The king sat upon his throne in the great hall. He wore the royal "
    "crown and held a golden scepter. As a man of noble birth, the son "
    "of emperors, he ruled his kingdom with authority. The prince became "
    "a powerful ruler, a male monarch feared by his enemies."
)
FEMALE_NONROYAL_PREFIX = (
    "The mother worked in her garden throughout the long summer. She "
    "planted flowers and watered them every morning. As a woman of the "
    "countryside, the daughter of farmers, she tended her home with "
    "dedication. The girl became a successful teacher, a skilled woman "
    "known by her students."
)


@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def warm_init_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    t_start: float = 0.3,
    num_denoise_steps: int = 10,
    n_denoise_avg: int = 5,
    sft_lr: float = 3e-4,
    sft_n_epochs: int = 5,
    sft_batch_size: int = 64,
    sft_eval_interval: int = 2,
    tie_weights: bool = False,
):
    """Test GLP-guided warm initialization of concept tokens.

    Three conditions, differing only in how queen's wte and lm_head rows
    are initialized before (optional) SFT:

    1. random:     queen stays at random init (N(0, 0.02))
    2. analogy:    queen = king + gender_shift (embedding arithmetic on both
                   wte and lm_head)
    3. glp_guided: lm_head from centroid of ln_f(h1) at queen-prediction
                   positions; wte from analogy (same as condition 2)

    Evaluation:
      - Zero-shot:  queen rank at prediction positions + generation coherence
      - SFT curve:  val loss and queen rank tracked every sft_eval_interval steps
      - Post-SFT:   full queen rank + generation on best checkpoint
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.glp import GLPDenoiser, glp_denoise, load_glp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    # ------------------------------------------------------------------ #
    # Token ID lookup
    # ------------------------------------------------------------------ #
    def find_token_ids(word):
        ids = []
        for tid in range(50257):
            if enc.decode([tid]).strip().lower() == word:
                ids.append(tid)
        return ids

    queen_ids = find_token_ids("queen")
    king_ids = find_token_ids("king")
    queen_id = queen_ids[0]
    king_id = king_ids[0]

    female_primary = [find_token_ids(w)[0] for w in FEMALE_WORDS if find_token_ids(w)]
    male_primary = [find_token_ids(w)[0] for w in MALE_WORDS if find_token_ids(w)]

    print(f"Queen IDs: {queen_ids} (primary {queen_id})")
    print(f"King IDs:  {king_ids} (primary {king_id})")
    print(f"Gender words: {len(female_primary)} female, {len(male_primary)} male")

    # ------------------------------------------------------------------ #
    # Load baseline model + GLP
    # ------------------------------------------------------------------ #
    tied_suffix = "_tied" if tie_weights else ""
    model_dir = (f"{DATA_DIR}/models/tau_{source_tau:.3f}"
                 f"/P_{source_P}/T_{block_size}{tied_suffix}")
    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)

    assert queen_id < vocab_size, (
        f"queen_id={queen_id} >= vocab_size={vocab_size}")

    model_kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
        tie_weights=tie_weights,
    )
    print(f"Model: vocab={vocab_size}, layers={n_layer}, d={n_embd}"
          f"{' (tied weights)' if tie_weights else ''}")

    glp_dir = os.path.join(model_dir, "glp_baseline")
    if os.path.exists(glp_dir):
        glp_model, act_stats, glp_cfg = load_glp(glp_dir, device=device)
        glp_model = glp_model.to(device).eval()
        print(f"Loaded GLP from {glp_dir}")
    else:
        print(f"No GLP found at {glp_dir}, skipping")

    # ------------------------------------------------------------------ #
    # Contrastive lm_head: centroid(queen) - centroid(control)
    # ------------------------------------------------------------------ #
    print("\n=== Contrastive lm_head ===")

    tmp = GPT(**model_kwargs).to(device)
    tmp.load_state_dict(sd)
    tmp.eval()

    def extract_h1_normed(prompt):
        tokens = enc.encode(prompt)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        layer_outs = []
        hooks = []
        for block in tmp.transformer.h:
            hooks.append(block.register_forward_hook(
                lambda _m, _i, o, s=layer_outs: s.append(o.detach())
            ))
        with torch.no_grad():
            tmp(ids)
        for h in hooks:
            h.remove()
        h1 = layer_outs[-1][:, -1, :]
        with torch.no_grad():
            return tmp.transformer.ln_f(h1).squeeze(0)

    queen_h1s = torch.stack([extract_h1_normed(p)
                             for p in QUEEN_PREDICTION_PROMPTS])
    control_h1s = torch.stack([extract_h1_normed(p)
                               for p in CONTROL_PROMPTS])
    contrast = queen_h1s.mean(dim=0) - control_h1s.mean(dim=0)

    target_norm = tmp.lm_head.weight.data.norm(dim=1).mean()
    contrastive_lm_head = contrast * (target_norm / contrast.norm())

    king_lm = tmp.lm_head.weight.data[king_id]
    cos_contrast_king = F.cosine_similarity(
        contrastive_lm_head.unsqueeze(0), king_lm.unsqueeze(0)
    ).item()
    print(f"  contrast norm (raw): {contrast.norm():.4f}")
    print(f"  scaled norm: {contrastive_lm_head.norm():.4f}  "
          f"(target {target_norm:.4f})")
    print(f"  cos(contrast, king_lm): {cos_contrast_king:.4f}")

    del tmp
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Analogy-based initialization
    # ------------------------------------------------------------------ #
    print("\n=== Analogy initialization ===")

    wte = sd["transformer.wte.weight"]
    lmw = sd["lm_head.weight"]

    gender_wte = (torch.stack([wte[i] for i in female_primary]).mean(0)
                  - torch.stack([wte[i] for i in male_primary]).mean(0))
    gender_lm = (torch.stack([lmw[i] for i in female_primary]).mean(0)
                 - torch.stack([lmw[i] for i in male_primary]).mean(0))

    analogy_wte = wte[king_id] + gender_wte
    analogy_lm = lmw[king_id] + gender_lm

    cos_analogy_king = F.cosine_similarity(
        analogy_lm.unsqueeze(0), lmw[king_id].unsqueeze(0)
    ).item()
    print(f"  gender shift norms: wte={gender_wte.norm():.4f}, "
          f"lm={gender_lm.norm():.4f}")
    print(f"  analogy norms: wte={analogy_wte.norm():.4f}, "
          f"lm={analogy_lm.norm():.4f}")
    print(f"  cos(analogy_lm, king_lm): {cos_analogy_king:.4f}")

    # ------------------------------------------------------------------ #
    # Conditions
    # ------------------------------------------------------------------ #
    if tie_weights:
        conditions = {
            "random": {
                "wte": wte[queen_id].clone(),
                "lm_head": wte[queen_id].clone(),
            },
            "analogy": {
                "wte": analogy_wte,
                "lm_head": analogy_wte.clone(),
            },
            "contrastive": {
                "wte": contrastive_lm_head.cpu(),
                "lm_head": contrastive_lm_head.cpu(),
            },
        }
    else:
        conditions = {
            "random": {
                "wte": wte[queen_id].clone(),
                "lm_head": lmw[queen_id].clone(),
            },
            "analogy": {
                "wte": analogy_wte,
                "lm_head": analogy_lm,
            },
            "contrastive": {
                "wte": analogy_wte.clone(),
                "lm_head": contrastive_lm_head.cpu(),
            },
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def make_model(init_dict):
        model = GPT(**model_kwargs).to(device)
        csd = {k: v.clone() for k, v in sd.items()}
        for qid in queen_ids:
            csd["transformer.wte.weight"][qid] = init_dict["wte"].clone()
            csd["lm_head.weight"][qid] = init_dict["lm_head"].clone()
        model.load_state_dict(csd)
        return model

    def eval_prediction(model, prompts):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(ids)
            last = logits[0, -1, :]
            probs = F.softmax(last, dim=-1)

            best_rank, best_prob = vocab_size, 0.0
            for qid in queen_ids:
                r = int((last > last[qid]).sum()) + 1
                p = float(probs[qid])
                if r < best_rank:
                    best_rank, best_prob = r, p

            top_v, top_i = probs.topk(5)
            top5 = [(enc.decode([int(i)]), float(v))
                    for i, v in zip(top_i, top_v)]
            results.append(dict(prompt=prompt, queen_rank=best_rank,
                                queen_prob=best_prob, top5=top5))
        return results

    def eval_generation(model, prompts, max_tokens=20):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            gen = list(tokens)
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp = torch.tensor(
                        [gen[-block_size:]], dtype=torch.long, device=device)
                    logits, _ = model(inp)
                    gen.append(logits[0, -1, :].argmax().item())
            results.append(dict(prompt=prompt,
                                continuation=enc.decode(gen[len(tokens):])))
        return results

    def summarize_pred(pred, label=""):
        ranks = [r["queen_rank"] for r in pred]
        m = float(np.mean(ranks))
        t10 = sum(1 for r in ranks if r <= 10)
        t100 = sum(1 for r in ranks if r <= 100)
        if label:
            print(f"  {label}: mean_rank={m:.1f}, "
                  f"top-10={t10}/{len(ranks)}, top-100={t100}/{len(ranks)}")
        return m, t10, t100

    # ------------------------------------------------------------------ #
    # Zero-shot evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION")
    print("=" * 60)

    all_results = {}

    for cond, init in conditions.items():
        print(f"\n--- {cond} ---")
        model = make_model(init)
        model.eval()

        pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        for r in pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"  [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")

        mean_r, t10, t100 = summarize_pred(pred, "prediction")

        for g in gen:
            print(f"  \"{g['prompt']}\" -> "
                  f"\"{g['continuation'][:60]}\"")

        # Specificity: queen rank at non-queen prompts
        ctrl_pred = eval_prediction(model, CONTROL_PROMPTS)
        ctrl_mean, ctrl_t10, _ = summarize_pred(ctrl_pred, "control")

        all_results[cond] = dict(
            zero_shot_prediction=pred,
            zero_shot_generation=gen,
            zero_shot_mean_rank=mean_r,
            zero_shot_top10=t10,
            zero_shot_top100=t100,
            zero_shot_control_mean_rank=ctrl_mean,
            zero_shot_control_top10=ctrl_t10,
        )

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # SFT convergence comparison
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SFT CONVERGENCE")
    print("=" * 60)

    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    split = int(0.9 * len(curriculum))
    train_data, val_data = curriculum[:split], curriculum[split:]

    steps_per_epoch = max(1, len(train_data) // (sft_batch_size * block_size))
    n_steps = sft_n_epochs * steps_per_epoch
    print(f"Curriculum: {len(curriculum):,} tokens, "
          f"{n_steps} steps ({sft_n_epochs} epochs)")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i + bl] for i in ix])
        y = torch.stack([data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    eval_subset = QUEEN_PREDICTION_PROMPTS[:4]

    for cond, init in conditions.items():
        print(f"\n--- SFT: {cond} ---")

        torch.manual_seed(42)
        model = make_model(init)
        opt = torch.optim.AdamW(model.parameters(), lr=sft_lr,
                                weight_decay=0.01)

        curve = []
        best_val = float("inf")
        best_sd_cond = None

        for step in range(n_steps):
            model.train()
            x, y = get_batch(train_data, sft_batch_size, block_size)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % sft_eval_interval == 0 or step == n_steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = sum(
                        float(model(
                            *get_batch(val_data, sft_batch_size, block_size)
                        )[1])
                        for _ in range(5)
                    ) / 5.0

                pr = eval_prediction(model, eval_subset)
                qr = float(np.mean([r["queen_rank"] for r in pr]))

                if vl < best_val:
                    best_val = vl
                    best_sd_cond = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

                curve.append(dict(step=step, train_loss=float(loss),
                                  val_loss=vl, queen_mean_rank=qr))
                print(f"  step {step:3d}/{n_steps}: "
                      f"train={loss:.4f} val={vl:.4f} "
                      f"best={best_val:.4f} queen={qr:.0f}")

        # Final eval on best checkpoint
        if best_sd_cond:
            model.load_state_dict(best_sd_cond)
            model = model.to(device)
        model.eval()

        final_pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        final_gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        print(f"\n  Post-SFT results:")
        for r in final_pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"    [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")
        for g in final_gen:
            print(f"    \"{g['prompt']}\" -> "
                  f"\"{g['continuation'][:60]}\"")

        final_mean, final_t10, _ = summarize_pred(final_pred, "post-SFT")

        first_top10 = None
        for pt in curve:
            if pt["queen_mean_rank"] <= 10:
                first_top10 = pt["step"]
                break

        all_results[cond].update(dict(
            sft_curve=curve,
            sft_best_val_loss=best_val,
            sft_final_prediction=final_pred,
            sft_final_generation=final_gen,
            sft_final_mean_rank=final_mean,
            sft_final_top10=final_t10,
            sft_first_top10_step=first_top10,
        ))

        del model, opt, best_sd_cond
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for cond in conditions:
        r = all_results[cond]
        zs = (f"queen_rank={r['zero_shot_mean_rank']:.1f} "
              f"({r['zero_shot_top10']}/12 top10), "
              f"control_rank={r['zero_shot_control_mean_rank']:.1f} "
              f"({r['zero_shot_control_top10']}/12 top10)")
        ps = f"rank={r['sft_final_mean_rank']:.1f}, top10={r['sft_final_top10']}/12"
        ft = r.get("sft_first_top10_step")
        ft_s = f"step {ft}" if ft is not None else "never"
        print(f"\n  {cond}:")
        print(f"    zero-shot:  {zs}")
        print(f"    post-SFT:   {ps}  (best_val={r['sft_best_val_loss']:.4f})")
        print(f"    first top10: {ft_s}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    results = dict(
        config=dict(
            source_tau=source_tau, source_P=source_P,
            block_size=block_size, t_start=t_start,
            n_denoise_avg=n_denoise_avg, sft_lr=sft_lr,
            sft_n_epochs=sft_n_epochs, n_steps=n_steps,
            queen_ids=queen_ids, king_id=king_id,
            tie_weights=tie_weights,
        ),
        diagnostics=dict(
            cos_contrastive_lm_vs_king_lm=cos_contrast_king,
            cos_analogy_lm_vs_king_lm=cos_analogy_king,
            contrastive_lm_head_norm=float(contrastive_lm_head.norm()),
            analogy_lm_head_norm=float(analogy_lm.norm()),
        ),
        conditions=all_results,
    )

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    tied_tag = "_tied" if tie_weights else ""
    fname = f"warm_init_tau{source_tau:.1f}{tied_tag}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/{fname}")
    return results


# ======================================================================
# Phase 2: wte initialization
# ======================================================================

INVERSION_PAIRS = [
    ("The king ruled", "The queen ruled"),
    ("The king was powerful", "The queen was powerful"),
    ("Long live the king", "Long live the queen"),
    ("The king spoke to", "The queen spoke to"),
    ("The king of the land", "The queen of the land"),
    ("The king and the", "The queen and the"),
]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def warm_init_phase2_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    sft_lr: float = 3e-4,
    sft_n_epochs: int = 5,
    sft_batch_size: int = 64,
    sft_eval_interval: int = 2,
    inversion_lr: float = 0.01,
    inversion_steps: int = 200,
):
    """Phase 2: Warm initialization of queen's input embedding (wte).

    Phase 1 solved the output side (contrastive lm_head achieves 9/12 top-10
    zero-shot). This experiment tests three wte initialization strategies,
    all using the Phase 1 contrastive lm_head:

    1. analogy:    king + gender_shift in embedding space (Phase 1 baseline)
    2. regression: T(contrastive_lm_head), where T is a learned lm_head->wte
                   mapping fit on the existing vocabulary
    3. inversion:  optimize wte through frozen Block 0 toward h0 targets
                   derived from king + gender_shift in activation space

    Each is evaluated zero-shot (generation from queen-input prompts) then
    via SFT-on-wte-only (curriculum training with only wte[queen] learnable).
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    # ------------------------------------------------------------------ #
    # Token ID lookup
    # ------------------------------------------------------------------ #
    def find_token_ids(word):
        ids = []
        for tid in range(50257):
            if enc.decode([tid]).strip().lower() == word:
                ids.append(tid)
        return ids

    queen_ids = find_token_ids("queen")
    king_ids = find_token_ids("king")
    queen_id = queen_ids[0]
    king_id = king_ids[0]

    female_primary = [find_token_ids(w)[0] for w in FEMALE_WORDS if find_token_ids(w)]
    male_primary = [find_token_ids(w)[0] for w in MALE_WORDS if find_token_ids(w)]

    print(f"Queen IDs: {queen_ids} (primary {queen_id})")
    print(f"King IDs:  {king_ids} (primary {king_id})")

    # ------------------------------------------------------------------ #
    # Load baseline model
    # ------------------------------------------------------------------ #
    model_dir = (f"{DATA_DIR}/models/tau_{source_tau:.3f}"
                 f"/P_{source_P}/T_{block_size}")
    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)

    model_kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
    )
    print(f"Model: vocab={vocab_size}, layers={n_layer}, d={n_embd}")

    # ------------------------------------------------------------------ #
    # Contrastive lm_head (recompute from Phase 1 logic)
    # ------------------------------------------------------------------ #
    print("\n=== Contrastive lm_head (Phase 1) ===")

    tmp = GPT(**model_kwargs).to(device)
    tmp.load_state_dict(sd)
    tmp.eval()

    def extract_h1_normed(model, prompt):
        tokens = enc.encode(prompt)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        layer_outs = []
        hooks = []
        for block in model.transformer.h:
            hooks.append(block.register_forward_hook(
                lambda _m, _i, o, s=layer_outs: s.append(o.detach())
            ))
        with torch.no_grad():
            model(ids)
        for h in hooks:
            h.remove()
        h1 = layer_outs[-1][:, -1, :]
        with torch.no_grad():
            return model.transformer.ln_f(h1).squeeze(0)

    queen_h1s = torch.stack([extract_h1_normed(tmp, p)
                             for p in QUEEN_PREDICTION_PROMPTS])
    control_h1s = torch.stack([extract_h1_normed(tmp, p)
                               for p in CONTROL_PROMPTS])
    contrast = queen_h1s.mean(dim=0) - control_h1s.mean(dim=0)
    target_norm = tmp.lm_head.weight.data.norm(dim=1).mean()
    contrastive_lm = contrast * (target_norm / contrast.norm())
    print(f"  norm: {contrastive_lm.norm():.4f} (target {target_norm:.4f})")

    del tmp
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Analogy wte (baseline)
    # ------------------------------------------------------------------ #
    wte = sd["transformer.wte.weight"]
    lmw = sd["lm_head.weight"]

    gender_wte = (torch.stack([wte[i] for i in female_primary]).mean(0)
                  - torch.stack([wte[i] for i in male_primary]).mean(0))
    analogy_wte = wte[king_id] + gender_wte

    print(f"\n=== Analogy wte ===")
    print(f"  norm: {analogy_wte.norm():.4f} "
          f"(mean wte norm: {wte.norm(dim=1).mean():.4f})")

    # ------------------------------------------------------------------ #
    # Condition 2: Regression (lm_head -> wte mapping)
    # ------------------------------------------------------------------ #
    print("\n=== Regression wte ===")

    mask = torch.ones(vocab_size, dtype=torch.bool)
    for qid in queen_ids:
        mask[qid] = False

    X = lmw[mask]  # (V-k, d) — lm_head rows
    Y = wte[mask]  # (V-k, d) — wte rows
    T_map = torch.linalg.lstsq(X, Y).solution  # (d, d)
    regression_wte = contrastive_lm.cpu() @ T_map

    residual_fit = Y - X @ T_map
    r2 = 1.0 - (residual_fit.norm() ** 2) / ((Y - Y.mean(0)).norm() ** 2)
    cos_reg_analogy = F.cosine_similarity(
        regression_wte.unsqueeze(0), analogy_wte.unsqueeze(0)).item()
    cos_reg_king = F.cosine_similarity(
        regression_wte.unsqueeze(0), wte[king_id].unsqueeze(0)).item()
    print(f"  T mapping R²: {r2:.4f}")
    print(f"  norm: {regression_wte.norm():.4f}")
    print(f"  cos(regression, analogy): {cos_reg_analogy:.4f}")
    print(f"  cos(regression, king_wte): {cos_reg_king:.4f}")

    # ------------------------------------------------------------------ #
    # Condition 3: Gradient inversion through frozen Block 0
    # ------------------------------------------------------------------ #
    print("\n=== Gradient inversion ===")

    inv_model = GPT(**model_kwargs).to(device)
    inv_model.load_state_dict(sd)
    inv_model.eval()
    for p in inv_model.parameters():
        p.requires_grad_(False)

    # Gender shift in h0 space (single-token, through Block 0)
    def get_h0_single(model, token_id):
        ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = model.transformer.wte(ids)
            pos_emb = model.transformer.wpe(
                torch.zeros(1, dtype=torch.long, device=device))
            x = model.transformer.drop(tok_emb + pos_emb)
            x = model.transformer.h[0](x)
        return x[0, 0]

    female_h0s = torch.stack([get_h0_single(inv_model, f) for f in female_primary])
    male_h0s = torch.stack([get_h0_single(inv_model, m) for m in male_primary])
    gender_shift_h0 = female_h0s.mean(0) - male_h0s.mean(0)
    print(f"  gender_shift_h0 norm: {gender_shift_h0.norm():.4f}")

    # Build targets: king h0 + gender shift at paired prompts
    inv_data = []
    for king_prompt, queen_prompt in INVERSION_PAIRS:
        king_toks = enc.encode(king_prompt)
        queen_toks = enc.encode(queen_prompt)

        king_pos = next((i for i, t in enumerate(king_toks) if t in king_ids), None)
        queen_pos = next((i for i, t in enumerate(queen_toks)
                          if t in queen_ids), None)
        if king_pos is None or queen_pos is None or king_pos != queen_pos:
            print(f"  SKIP: {king_prompt}")
            continue

        ids = torch.tensor([king_toks], dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = inv_model.transformer.wte(ids)
            pos_emb = inv_model.transformer.wpe(
                torch.arange(len(king_toks), device=device))
            x = inv_model.transformer.drop(tok_emb + pos_emb)
            x = inv_model.transformer.h[0](x)
        king_h0 = x[0, king_pos]
        tgt = (king_h0 + gender_shift_h0).detach()

        inv_data.append(dict(queen_toks=queen_toks, pos=queen_pos, target=tgt))
        print(f"  {king_prompt}: pos={king_pos}, target norm={tgt.norm():.4f}")

    # Optimize
    queen_emb = analogy_wte.clone().to(device).requires_grad_(True)
    inv_opt = torch.optim.Adam([queen_emb], lr=inversion_lr)

    for step in range(inversion_steps):
        inv_opt.zero_grad()
        total_loss = 0.0

        for item in inv_data:
            ids = torch.tensor([item["queen_toks"]],
                               dtype=torch.long, device=device)
            tok_emb = inv_model.transformer.wte(ids).detach().clone()
            tok_emb[0, item["pos"]] = queen_emb

            pos_emb = inv_model.transformer.wpe(
                torch.arange(ids.shape[1], device=device)).detach()
            x = tok_emb + pos_emb
            x = inv_model.transformer.h[0](x)

            total_loss = total_loss + F.mse_loss(
                x[0, item["pos"]], item["target"])

        total_loss.backward()
        inv_opt.step()

        if step % 50 == 0 or step == inversion_steps - 1:
            print(f"  step {step:3d}: loss={total_loss.item():.6f}")

    inversion_wte = queen_emb.detach().cpu()
    cos_inv_analogy = F.cosine_similarity(
        inversion_wte.unsqueeze(0), analogy_wte.unsqueeze(0)).item()
    print(f"  final norm: {inversion_wte.norm():.4f}")
    print(f"  cos(inversion, analogy): {cos_inv_analogy:.4f}")

    del inv_model, queen_emb
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Conditions
    # ------------------------------------------------------------------ #
    contrastive_lm_cpu = contrastive_lm.cpu()
    conditions = {
        "analogy": dict(wte=analogy_wte, lm_head=contrastive_lm_cpu),
        "regression": dict(wte=regression_wte, lm_head=contrastive_lm_cpu),
        "inversion": dict(wte=inversion_wte, lm_head=contrastive_lm_cpu),
    }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def make_model(init_dict):
        model = GPT(**model_kwargs).to(device)
        csd = {k: v.clone() for k, v in sd.items()}
        for qid in queen_ids:
            csd["transformer.wte.weight"][qid] = init_dict["wte"].clone()
            csd["lm_head.weight"][qid] = init_dict["lm_head"].clone()
        model.load_state_dict(csd)
        return model

    def eval_generation(model, prompts, max_tokens=30):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            gen = list(tokens)
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp = torch.tensor(
                        [gen[-block_size:]], dtype=torch.long, device=device)
                    logits, _ = model(inp)
                    gen.append(logits[0, -1, :].argmax().item())
            results.append(dict(prompt=prompt,
                                continuation=enc.decode(gen[len(tokens):])))
        return results

    def eval_prediction(model, prompts):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(ids)
            last = logits[0, -1, :]
            probs = F.softmax(last, dim=-1)
            best_rank = vocab_size
            for qid in queen_ids:
                r = int((last > last[qid]).sum()) + 1
                if r < best_rank:
                    best_rank = r
            top_v, top_i = probs.topk(5)
            top5 = [(enc.decode([int(i)]), float(v))
                    for i, v in zip(top_i, top_v)]
            results.append(dict(prompt=prompt, queen_rank=best_rank, top5=top5))
        return results

    # ------------------------------------------------------------------ #
    # Zero-shot evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION (Phase 2: wte focus)")
    print("=" * 60)

    all_results = {}

    for cond, init in conditions.items():
        print(f"\n--- {cond} ---")
        model = make_model(init)
        model.eval()

        gen = eval_generation(model, QUEEN_INPUT_PROMPTS)
        print("  Generation from queen-input prompts:")
        for g in gen:
            print(f"    \"{g['prompt']}\" -> \"{g['continuation'][:60]}\"")

        pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS[:4])
        pred_rank = float(np.mean([r["queen_rank"] for r in pred]))
        print(f"  Queen prediction rank (sanity): {pred_rank:.1f}")

        all_results[cond] = dict(
            zero_shot_generation=gen,
            zero_shot_prediction=pred,
            zero_shot_pred_mean_rank=pred_rank,
        )
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # SFT on wte only
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SFT ON WTE ONLY")
    print("=" * 60)

    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    split = int(0.9 * len(curriculum))
    train_data, val_data = curriculum[:split], curriculum[split:]

    steps_per_epoch = max(1, len(train_data) // (sft_batch_size * block_size))
    n_steps = sft_n_epochs * steps_per_epoch
    print(f"Curriculum: {len(curriculum):,} tokens, "
          f"{n_steps} steps ({sft_n_epochs} epochs)")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i + bl] for i in ix])
        y = torch.stack([data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    queen_mask_t = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for qid in queen_ids:
        queen_mask_t[qid] = True

    for cond, init in conditions.items():
        print(f"\n--- SFT wte-only: {cond} ---")
        torch.manual_seed(42)
        model = make_model(init)

        for p in model.parameters():
            p.requires_grad_(False)
        model.transformer.wte.weight.requires_grad_(True)

        opt = torch.optim.Adam(
            [model.transformer.wte.weight], lr=sft_lr)

        curve = []
        best_val = float("inf")
        best_sd_cond = None

        for step in range(n_steps):
            model.train()
            x, y = get_batch(train_data, sft_batch_size, block_size)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = model.transformer.wte.weight.grad
                if grad is not None:
                    grad[~queen_mask_t] = 0.0

            torch.nn.utils.clip_grad_norm_(
                [model.transformer.wte.weight], 1.0)
            opt.step()

            if step % sft_eval_interval == 0 or step == n_steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = sum(
                        float(model(*get_batch(
                            val_data, sft_batch_size, block_size))[1])
                        for _ in range(5)
                    ) / 5.0

                gen = eval_generation(model, QUEEN_INPUT_PROMPTS[:2])
                preview = gen[0]["continuation"][:40]

                if vl < best_val:
                    best_val = vl
                    best_sd_cond = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

                curve.append(dict(step=step, train_loss=float(loss),
                                  val_loss=vl, gen_preview=preview))
                print(f"  step {step:3d}/{n_steps}: "
                      f"train={loss:.4f} val={vl:.4f} "
                      f"gen=\"{preview}\"")

        # Final eval on best checkpoint
        if best_sd_cond:
            model.load_state_dict(best_sd_cond)
            model = model.to(device)
        model.eval()

        final_gen = eval_generation(model, QUEEN_INPUT_PROMPTS)
        final_pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS[:4])
        final_rank = float(np.mean([r["queen_rank"] for r in final_pred]))

        print(f"\n  Post-SFT generation:")
        for g in final_gen:
            print(f"    \"{g['prompt']}\" -> \"{g['continuation'][:60]}\"")
        print(f"  Queen prediction rank: {final_rank:.1f}")

        all_results[cond].update(dict(
            sft_curve=curve,
            sft_best_val_loss=best_val,
            sft_final_generation=final_gen,
            sft_final_prediction=final_pred,
            sft_final_pred_mean_rank=final_rank,
        ))

        del model, opt, best_sd_cond
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for cond in conditions:
        r = all_results[cond]
        zs = r["zero_shot_generation"][0]["continuation"][:40]
        ps = r["sft_final_generation"][0]["continuation"][:40]
        print(f"\n  {cond}:")
        print(f"    zero-shot:  \"{zs}\"")
        print(f"    post-SFT:   \"{ps}\"")
        print(f"    best val:   {r['sft_best_val_loss']:.4f}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    results = dict(
        config=dict(
            source_tau=source_tau, source_P=source_P,
            block_size=block_size, sft_lr=sft_lr,
            sft_n_epochs=sft_n_epochs, n_steps=n_steps,
            inversion_lr=inversion_lr, inversion_steps=inversion_steps,
            queen_ids=queen_ids, king_id=king_id,
        ),
        diagnostics=dict(
            regression_R2=float(r2),
            cos_regression_analogy=cos_reg_analogy,
            cos_regression_king_wte=cos_reg_king,
            cos_inversion_analogy=cos_inv_analogy,
            contrastive_lm_norm=float(contrastive_lm.norm()),
            gender_shift_h0_norm=float(gender_shift_h0.norm()),
        ),
        conditions=all_results,
    )

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    fname = f"warm_init_phase2_tau{source_tau:.1f}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/{fname}")
    return results


# ======================================================================
# Phase 3: Gradient-signal wte initialization + full SFT
# ======================================================================

@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def warm_init_phase3_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    sft_lr: float = 3e-4,
    sft_n_epochs: int = 5,
    sft_batch_size: int = 64,
    sft_eval_interval: int = 2,
):
    """Phase 3: Gradient-signal wte initialization + full-model SFT.

    At queen-prediction positions, the model's error signal dL/dx0 encodes
    what embedding-space direction would make queen more likely. This is
    the gradient verbalization insight applied at the embedding level:
    the model's own error signals contain the information needed to specify
    the right embedding, even though the model can't predict queen.

    Two conditions, both with contrastive lm_head + full-model SFT:
      1. analogy:         king + gender_shift (baseline)
      2. gradient_signal: -mean(dL/dx0) across queen-prediction prompts,
                          scaled to match mean wte row norm

    Full SFT (all weights trainable) rather than wte-only, because Phase 2
    showed that frozen attention/MLP weights can't process a novel embedding.
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    # ------------------------------------------------------------------ #
    # Token ID lookup
    # ------------------------------------------------------------------ #
    def find_token_ids(word):
        ids = []
        for tid in range(50257):
            if enc.decode([tid]).strip().lower() == word:
                ids.append(tid)
        return ids

    queen_ids = find_token_ids("queen")
    king_ids = find_token_ids("king")
    queen_id = queen_ids[0]
    king_id = king_ids[0]

    female_primary = [find_token_ids(w)[0] for w in FEMALE_WORDS if find_token_ids(w)]
    male_primary = [find_token_ids(w)[0] for w in MALE_WORDS if find_token_ids(w)]

    print(f"Queen IDs: {queen_ids} (primary {queen_id})")
    print(f"King IDs:  {king_ids} (primary {king_id})")

    # ------------------------------------------------------------------ #
    # Load baseline model
    # ------------------------------------------------------------------ #
    model_dir = (f"{DATA_DIR}/models/tau_{source_tau:.3f}"
                 f"/P_{source_P}/T_{block_size}")
    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)

    model_kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
    )
    print(f"Model: vocab={vocab_size}, layers={n_layer}, d={n_embd}")

    # ------------------------------------------------------------------ #
    # Contrastive lm_head (Phase 1)
    # ------------------------------------------------------------------ #
    print("\n=== Contrastive lm_head ===")

    tmp = GPT(**model_kwargs).to(device)
    tmp.load_state_dict(sd)
    tmp.eval()

    def extract_h1_normed(model, prompt):
        tokens = enc.encode(prompt)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        layer_outs = []
        hooks = []
        for block in model.transformer.h:
            hooks.append(block.register_forward_hook(
                lambda _m, _i, o, s=layer_outs: s.append(o.detach())
            ))
        with torch.no_grad():
            model(ids)
        for h in hooks:
            h.remove()
        h1 = layer_outs[-1][:, -1, :]
        with torch.no_grad():
            return model.transformer.ln_f(h1).squeeze(0)

    queen_h1s = torch.stack([extract_h1_normed(tmp, p)
                             for p in QUEEN_PREDICTION_PROMPTS])
    control_h1s = torch.stack([extract_h1_normed(tmp, p)
                               for p in CONTROL_PROMPTS])
    contrast = queen_h1s.mean(dim=0) - control_h1s.mean(dim=0)
    target_norm = tmp.lm_head.weight.data.norm(dim=1).mean()
    contrastive_lm = contrast * (target_norm / contrast.norm())
    print(f"  norm: {contrastive_lm.norm():.4f} (target {target_norm:.4f})")

    del tmp
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Analogy wte (baseline)
    # ------------------------------------------------------------------ #
    wte = sd["transformer.wte.weight"]

    gender_wte = (torch.stack([wte[i] for i in female_primary]).mean(0)
                  - torch.stack([wte[i] for i in male_primary]).mean(0))
    analogy_wte = wte[king_id] + gender_wte
    mean_wte_norm = wte.norm(dim=1).mean().item()

    print(f"\n=== Analogy wte ===")
    print(f"  norm: {analogy_wte.norm():.4f} (mean wte norm: {mean_wte_norm:.4f})")

    # ------------------------------------------------------------------ #
    # Gradient signal: dL/dx0 at queen-prediction positions
    # ------------------------------------------------------------------ #
    print("\n=== Gradient signal ===")

    grad_model = GPT(**model_kwargs).to(device)
    grad_sd = {k: v.clone() for k, v in sd.items()}
    for qid in queen_ids:
        grad_sd["lm_head.weight"][qid] = contrastive_lm.cpu().clone()
    grad_model.load_state_dict(grad_sd)
    grad_model.eval()
    for p in grad_model.parameters():
        p.requires_grad_(False)

    per_prompt_grads = []
    for prompt in QUEEN_PREDICTION_PROMPTS:
        tokens = enc.encode(prompt)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        T = ids.shape[1]

        with torch.no_grad():
            tok_emb = grad_model.transformer.wte(ids)
            pos_emb = grad_model.transformer.wpe(
                torch.arange(T, device=device))

        x0 = (tok_emb + pos_emb).detach().requires_grad_(True)

        x = grad_model.transformer.drop(x0)
        for block in grad_model.transformer.h:
            x = block(x)
        x = grad_model.transformer.ln_f(x)
        logits = grad_model.lm_head(x)

        loss = F.cross_entropy(
            logits[0, -1:, :],
            torch.tensor([queen_id], device=device))
        loss.backward()

        g = x0.grad[0, -1, :].detach().clone()
        per_prompt_grads.append(g)

    grads_stack = torch.stack(per_prompt_grads)
    avg_grad = grads_stack.mean(dim=0)

    # Coherence: how aligned are the per-prompt gradients?
    cos_matrix = F.cosine_similarity(
        grads_stack.unsqueeze(1), grads_stack.unsqueeze(0), dim=2)
    mean_pairwise_cos = (cos_matrix.sum() - len(per_prompt_grads)) / (
        len(per_prompt_grads) * (len(per_prompt_grads) - 1))
    coherence_ratio = avg_grad.norm() / grads_stack.norm(dim=1).mean()

    print(f"  per-prompt grad norms: mean={grads_stack.norm(dim=1).mean():.4f}, "
          f"std={grads_stack.norm(dim=1).std():.4f}")
    print(f"  mean pairwise cos: {mean_pairwise_cos:.4f}")
    print(f"  coherence ratio (avg_norm / mean_norm): {coherence_ratio:.4f}")
    print(f"  avg grad norm: {avg_grad.norm():.4f}")

    # Descent direction, scaled to match typical wte norms
    signal = -avg_grad
    gradient_signal_wte = signal * (mean_wte_norm / signal.norm())

    cos_gs_analogy = F.cosine_similarity(
        gradient_signal_wte.cpu().unsqueeze(0),
        analogy_wte.unsqueeze(0)).item()
    cos_gs_king = F.cosine_similarity(
        gradient_signal_wte.cpu().unsqueeze(0),
        wte[king_id].unsqueeze(0)).item()
    cos_gs_contrastive = F.cosine_similarity(
        gradient_signal_wte.cpu().unsqueeze(0),
        contrastive_lm.cpu().unsqueeze(0)).item()

    print(f"  gradient_signal_wte norm: {gradient_signal_wte.norm():.4f}")
    print(f"  cos(gradient_signal, analogy): {cos_gs_analogy:.4f}")
    print(f"  cos(gradient_signal, king_wte): {cos_gs_king:.4f}")
    print(f"  cos(gradient_signal, contrastive_lm): {cos_gs_contrastive:.4f}")

    del grad_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Conditions (all use contrastive lm_head + full SFT)
    # ------------------------------------------------------------------ #
    contrastive_lm_cpu = contrastive_lm.cpu()
    conditions = {
        "analogy": dict(wte=analogy_wte, lm_head=contrastive_lm_cpu),
        "gradient_signal": dict(
            wte=gradient_signal_wte.cpu(), lm_head=contrastive_lm_cpu),
    }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def make_model(init_dict):
        model = GPT(**model_kwargs).to(device)
        csd = {k: v.clone() for k, v in sd.items()}
        for qid in queen_ids:
            csd["transformer.wte.weight"][qid] = init_dict["wte"].clone()
            csd["lm_head.weight"][qid] = init_dict["lm_head"].clone()
        model.load_state_dict(csd)
        return model

    def eval_generation(model, prompts, max_tokens=30):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            gen = list(tokens)
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp = torch.tensor(
                        [gen[-block_size:]], dtype=torch.long, device=device)
                    logits, _ = model(inp)
                    gen.append(logits[0, -1, :].argmax().item())
            results.append(dict(prompt=prompt,
                                continuation=enc.decode(gen[len(tokens):])))
        return results

    def eval_prediction(model, prompts):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(ids)
            last = logits[0, -1, :]
            probs = F.softmax(last, dim=-1)
            best_rank, best_prob = vocab_size, 0.0
            for qid in queen_ids:
                r = int((last > last[qid]).sum()) + 1
                p = float(probs[qid])
                if r < best_rank:
                    best_rank, best_prob = r, p
            top_v, top_i = probs.topk(5)
            top5 = [(enc.decode([int(i)]), float(v))
                    for i, v in zip(top_i, top_v)]
            results.append(dict(prompt=prompt, queen_rank=best_rank,
                                queen_prob=best_prob, top5=top5))
        return results

    def summarize_pred(pred, label=""):
        ranks = [r["queen_rank"] for r in pred]
        m = float(np.mean(ranks))
        t10 = sum(1 for r in ranks if r <= 10)
        if label:
            print(f"  {label}: mean_rank={m:.1f}, top-10={t10}/{len(ranks)}")
        return m, t10

    # ------------------------------------------------------------------ #
    # Zero-shot evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION")
    print("=" * 60)

    all_results = {}

    for cond, init in conditions.items():
        print(f"\n--- {cond} ---")
        model = make_model(init)
        model.eval()

        pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        for r in pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"  [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")
        mean_r, t10 = summarize_pred(pred, "prediction")

        ctrl_pred = eval_prediction(model, CONTROL_PROMPTS)
        ctrl_mean, ctrl_t10 = summarize_pred(ctrl_pred, "control")

        gen = eval_generation(model, QUEEN_INPUT_PROMPTS)
        for g in gen:
            print(f"  \"{g['prompt']}\" -> \"{g['continuation'][:60]}\"")

        all_results[cond] = dict(
            zero_shot_prediction=pred,
            zero_shot_mean_rank=mean_r,
            zero_shot_top10=t10,
            zero_shot_control_mean_rank=ctrl_mean,
            zero_shot_control_top10=ctrl_t10,
            zero_shot_generation=gen,
        )
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Full-model SFT convergence
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FULL-MODEL SFT")
    print("=" * 60)

    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    split = int(0.9 * len(curriculum))
    train_data, val_data = curriculum[:split], curriculum[split:]

    steps_per_epoch = max(1, len(train_data) // (sft_batch_size * block_size))
    n_steps = sft_n_epochs * steps_per_epoch
    print(f"Curriculum: {len(curriculum):,} tokens, "
          f"{n_steps} steps ({sft_n_epochs} epochs)")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i + bl] for i in ix])
        y = torch.stack([data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    for cond, init in conditions.items():
        print(f"\n--- Full SFT: {cond} ---")
        torch.manual_seed(42)
        model = make_model(init)
        opt = torch.optim.AdamW(
            model.parameters(), lr=sft_lr, weight_decay=0.01)

        curve = []
        best_val = float("inf")
        best_sd_cond = None

        for step in range(n_steps):
            model.train()
            x, y = get_batch(train_data, sft_batch_size, block_size)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % sft_eval_interval == 0 or step == n_steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = sum(
                        float(model(*get_batch(
                            val_data, sft_batch_size, block_size))[1])
                        for _ in range(5)
                    ) / 5.0

                pr = eval_prediction(model, QUEEN_PREDICTION_PROMPTS[:4])
                qr = float(np.mean([r["queen_rank"] for r in pr]))

                gen = eval_generation(model, QUEEN_INPUT_PROMPTS[:2])
                preview = gen[0]["continuation"][:40]

                if vl < best_val:
                    best_val = vl
                    best_sd_cond = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

                curve.append(dict(step=step, train_loss=float(loss),
                                  val_loss=vl, queen_mean_rank=qr,
                                  gen_preview=preview))
                print(f"  step {step:3d}/{n_steps}: "
                      f"train={loss:.4f} val={vl:.4f} "
                      f"queen_rank={qr:.0f} gen=\"{preview}\"")

        # Final eval on best checkpoint
        if best_sd_cond:
            model.load_state_dict(best_sd_cond)
            model = model.to(device)
        model.eval()

        final_pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        final_gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        print(f"\n  Post-SFT results:")
        for r in final_pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"    [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")
        for g in final_gen:
            print(f"    \"{g['prompt']}\" -> \"{g['continuation'][:60]}\"")

        final_mean, final_t10 = summarize_pred(final_pred, "post-SFT")

        first_top10 = None
        for pt in curve:
            if pt["queen_mean_rank"] <= 10:
                first_top10 = pt["step"]
                break

        all_results[cond].update(dict(
            sft_curve=curve,
            sft_best_val_loss=best_val,
            sft_final_prediction=final_pred,
            sft_final_generation=final_gen,
            sft_final_mean_rank=final_mean,
            sft_final_top10=final_t10,
            sft_first_top10_step=first_top10,
        ))

        del model, opt, best_sd_cond
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for cond in conditions:
        r = all_results[cond]
        zs = (f"pred_rank={r['zero_shot_mean_rank']:.1f} "
              f"({r['zero_shot_top10']}/12 top10)")
        ps = (f"rank={r['sft_final_mean_rank']:.1f}, "
              f"top10={r['sft_final_top10']}/12")
        ft = r.get("sft_first_top10_step")
        ft_s = f"step {ft}" if ft is not None else "never"
        gs = r["zero_shot_generation"][0]["continuation"][:40]
        ps_gen = r["sft_final_generation"][0]["continuation"][:40]
        print(f"\n  {cond}:")
        print(f"    zero-shot:   {zs}")
        print(f"    zero-shot gen: \"{gs}\"")
        print(f"    post-SFT:    {ps} (best_val={r['sft_best_val_loss']:.4f})")
        print(f"    post-SFT gen:  \"{ps_gen}\"")
        print(f"    first top10: {ft_s}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    results = dict(
        config=dict(
            source_tau=source_tau, source_P=source_P,
            block_size=block_size, sft_lr=sft_lr,
            sft_n_epochs=sft_n_epochs, n_steps=n_steps,
            queen_ids=queen_ids, king_id=king_id,
        ),
        diagnostics=dict(
            gradient_coherence=float(mean_pairwise_cos),
            gradient_coherence_ratio=float(coherence_ratio),
            cos_gradient_signal_analogy=cos_gs_analogy,
            cos_gradient_signal_king_wte=cos_gs_king,
            cos_gradient_signal_contrastive_lm=cos_gs_contrastive,
            contrastive_lm_norm=float(contrastive_lm.norm()),
        ),
        conditions=all_results,
    )

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    fname = f"warm_init_phase3_tau{source_tau:.1f}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/{fname}")
    return results


# ======================================================================
# GLP embedding-space initialization
# ======================================================================

@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def warm_init_glp_embed_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    t_start: float = 0.3,
    num_denoise_steps: int = 10,
    n_denoise_avg: int = 10,
    sft_lr: float = 3e-4,
    sft_n_epochs: int = 5,
    sft_batch_size: int = 64,
    sft_eval_interval: int = 2,
):
    """GLP-residual embedding initialization on tied-weights model.

    The embedding-space GLP (trained on ln_f(h1) from the tied-weights model)
    operates natively in embedding space. At queen-prediction positions, the
    GLP residual (activation - on-manifold projection) directly answers
    "what embedding direction is missing from the vocabulary."

    Four conditions, all on the tied-weights model:
      1. random:       original random init (N(0, 0.02))
      2. analogy:      king + gender_shift
      3. contrastive:  centroid(queen_h1) - centroid(control_h1), scaled
      4. glp_residual: mean GLP residual at queen-prediction positions, scaled

    Evaluation: zero-shot queen rank, specificity at control positions,
    SFT convergence, and post-SFT generation quality.
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.glp import GLPDenoiser, glp_denoise, load_glp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    # ------------------------------------------------------------------ #
    # Token ID lookup
    # ------------------------------------------------------------------ #
    def find_token_ids(word):
        ids = []
        for tid in range(50257):
            if enc.decode([tid]).strip().lower() == word:
                ids.append(tid)
        return ids

    queen_ids = find_token_ids("queen")
    king_ids = find_token_ids("king")
    queen_id = queen_ids[0]
    king_id = king_ids[0]

    female_primary = [find_token_ids(w)[0] for w in FEMALE_WORDS if find_token_ids(w)]
    male_primary = [find_token_ids(w)[0] for w in MALE_WORDS if find_token_ids(w)]

    print(f"Queen IDs: {queen_ids} (primary {queen_id})")
    print(f"King IDs:  {king_ids} (primary {king_id})")
    print(f"Gender words: {len(female_primary)} female, {len(male_primary)} male")

    # ------------------------------------------------------------------ #
    # Load tied-weights model
    # ------------------------------------------------------------------ #
    model_dir = (f"{DATA_DIR}/models/tau_{source_tau:.3f}"
                 f"/P_{source_P}/T_{block_size}_tied")
    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)

    assert queen_id < vocab_size, (
        f"queen_id={queen_id} >= vocab_size={vocab_size}")

    model_kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
        tie_weights=True,
    )
    print(f"Model: vocab={vocab_size}, layers={n_layer}, d={n_embd} (tied weights)")

    # ------------------------------------------------------------------ #
    # Load embedding-space GLP
    # ------------------------------------------------------------------ #
    glp_dir = os.path.join(model_dir, "glp_embed")
    assert os.path.exists(glp_dir), (
        f"No embedding-space GLP at {glp_dir}. "
        f"Run train-glp-embed first.")
    glp_model, act_stats, glp_cfg = load_glp(glp_dir, device=device)
    glp_model = glp_model.to(device).eval()
    glp_mean = act_stats["mean"].to(device)
    glp_std = act_stats["std"].to(device)
    print(f"Loaded embedding-space GLP from {glp_dir} "
          f"(act_dim={glp_cfg['act_dim']})")

    # ------------------------------------------------------------------ #
    # Extract ln_f(h1) at last position of each prompt
    # ------------------------------------------------------------------ #
    tmp = GPT(**model_kwargs).to(device)
    tmp.load_state_dict(sd)
    tmp.eval()

    def extract_lnf_h1_last(prompt):
        tokens = enc.encode(prompt)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = tmp.transformer.wte(ids)
            if hasattr(tmp.transformer, 'wpe'):
                pos = torch.arange(ids.shape[1], device=device)
                tok_emb = tok_emb + tmp.transformer.wpe(pos)
            h = tmp.transformer.drop(tok_emb)
            for block in tmp.transformer.h:
                h = block(h)
            h = tmp.transformer.ln_f(h)
        return h[0, -1, :]

    # ------------------------------------------------------------------ #
    # Contrastive centroid (existing best approach)
    # ------------------------------------------------------------------ #
    print("\n=== Contrastive centroid ===")

    queen_h1s = torch.stack([extract_lnf_h1_last(p)
                             for p in QUEEN_PREDICTION_PROMPTS])
    control_h1s = torch.stack([extract_lnf_h1_last(p)
                               for p in CONTROL_PROMPTS])
    contrast = queen_h1s.mean(dim=0) - control_h1s.mean(dim=0)

    wte = sd["transformer.wte.weight"]
    target_norm = wte.norm(dim=1).mean()
    contrastive_emb = contrast.cpu() * (target_norm / contrast.cpu().norm())
    print(f"  contrast norm (raw): {contrast.norm():.4f}")
    print(f"  scaled norm: {contrastive_emb.norm():.4f} (target {target_norm:.4f})")

    # ------------------------------------------------------------------ #
    # GLP residual at queen-prediction positions
    # ------------------------------------------------------------------ #
    print("\n=== GLP residual ===")

    per_prompt_residuals = []
    per_prompt_diagnostics = []

    for prompt in QUEEN_PREDICTION_PROMPTS:
        act = extract_lnf_h1_last(prompt)
        act_std = (act - glp_mean) / glp_std

        residuals = []
        for _ in range(n_denoise_avg):
            manifold_std = glp_denoise(
                glp_model, act_std.unsqueeze(0), t_start, num_denoise_steps)
            resid_std = act_std - manifold_std.squeeze(0)
            residuals.append(resid_std)

        avg_resid_std = torch.stack(residuals).mean(dim=0)
        avg_resid = avg_resid_std * glp_std
        per_prompt_residuals.append(avg_resid)

        resid_norm = avg_resid.norm().item()
        cos_to_contrast = F.cosine_similarity(
            avg_resid.unsqueeze(0), contrast.unsqueeze(0)).item()
        per_prompt_diagnostics.append({
            "prompt": prompt,
            "residual_norm": resid_norm,
            "cos_to_contrastive": cos_to_contrast,
        })
        print(f"  [{resid_norm:.4f}] cos_to_contrast={cos_to_contrast:+.4f}  "
              f"{prompt[:50]}")

    # Mean residual, scaled to wte norm
    resid_stack = torch.stack(per_prompt_residuals)
    mean_residual = resid_stack.mean(dim=0)
    glp_emb = mean_residual.cpu() * (target_norm / mean_residual.cpu().norm())

    # Cross-prompt coherence
    resid_normed = F.normalize(resid_stack, dim=-1)
    cos_matrix = resid_normed @ resid_normed.T
    n = cos_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_matrix.device)
    mean_pairwise_cos = float(cos_matrix[mask].mean())
    coherence = mean_residual.norm() / resid_stack.norm(dim=-1).mean()

    cos_glp_contrast = F.cosine_similarity(
        glp_emb.unsqueeze(0), contrastive_emb.unsqueeze(0)).item()
    cos_glp_king = F.cosine_similarity(
        glp_emb.unsqueeze(0), wte[king_id].unsqueeze(0)).item()

    print(f"\n  GLP residual summary:")
    print(f"    mean pairwise cos: {mean_pairwise_cos:.4f}")
    print(f"    coherence ratio:   {float(coherence):.4f}")
    print(f"    scaled norm:       {glp_emb.norm():.4f}")
    print(f"    cos(glp, contrastive): {cos_glp_contrast:+.4f}")
    print(f"    cos(glp, king_wte):    {cos_glp_king:+.4f}")

    del tmp
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Analogy initialization
    # ------------------------------------------------------------------ #
    gender_wte = (torch.stack([wte[i] for i in female_primary]).mean(0)
                  - torch.stack([wte[i] for i in male_primary]).mean(0))
    analogy_emb = wte[king_id] + gender_wte

    # ------------------------------------------------------------------ #
    # Conditions (all tied: wte = lm_head)
    # ------------------------------------------------------------------ #
    conditions = {
        "random": {
            "wte": wte[queen_id].clone(),
            "lm_head": wte[queen_id].clone(),
        },
        "analogy": {
            "wte": analogy_emb,
            "lm_head": analogy_emb.clone(),
        },
        "contrastive": {
            "wte": contrastive_emb,
            "lm_head": contrastive_emb.clone(),
        },
        "glp_residual": {
            "wte": glp_emb,
            "lm_head": glp_emb.clone(),
        },
    }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def make_model(init_dict):
        model = GPT(**model_kwargs).to(device)
        csd = {k: v.clone() for k, v in sd.items()}
        for qid in queen_ids:
            csd["transformer.wte.weight"][qid] = init_dict["wte"].clone()
            if "lm_head.weight" in csd:
                csd["lm_head.weight"][qid] = init_dict["lm_head"].clone()
        model.load_state_dict(csd)
        return model

    def eval_prediction(model, prompts):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(ids)
            last = logits[0, -1, :]
            probs = F.softmax(last, dim=-1)

            best_rank, best_prob = vocab_size, 0.0
            for qid in queen_ids:
                r = int((last > last[qid]).sum()) + 1
                p = float(probs[qid])
                if r < best_rank:
                    best_rank, best_prob = r, p

            top_v, top_i = probs.topk(5)
            top5 = [(enc.decode([int(i)]), float(v))
                    for i, v in zip(top_i, top_v)]
            results.append(dict(prompt=prompt, queen_rank=best_rank,
                                queen_prob=best_prob, top5=top5))
        return results

    def eval_generation(model, prompts, max_tokens=20):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            gen = list(tokens)
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp = torch.tensor(
                        [gen[-block_size:]], dtype=torch.long, device=device)
                    logits, _ = model(inp)
                    gen.append(logits[0, -1, :].argmax().item())
            results.append(dict(prompt=prompt,
                                continuation=enc.decode(gen[len(tokens):])))
        return results

    def summarize_pred(pred, label=""):
        ranks = [r["queen_rank"] for r in pred]
        m = float(np.mean(ranks))
        t10 = sum(1 for r in ranks if r <= 10)
        t100 = sum(1 for r in ranks if r <= 100)
        if label:
            print(f"  {label}: mean_rank={m:.1f}, "
                  f"top-10={t10}/{len(ranks)}, top-100={t100}/{len(ranks)}")
        return m, t10, t100

    # ------------------------------------------------------------------ #
    # Zero-shot evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION")
    print("=" * 60)

    all_results = {}

    for cond, init in conditions.items():
        print(f"\n--- {cond} ---")
        model = make_model(init)
        model.eval()

        pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        for r in pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"  [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")

        mean_r, t10, t100 = summarize_pred(pred, "prediction")

        for g in gen:
            print(f"  \"{g['prompt']}\" -> "
                  f"\"{g['continuation'][:60]}\"")

        ctrl_pred = eval_prediction(model, CONTROL_PROMPTS)
        ctrl_mean, ctrl_t10, _ = summarize_pred(ctrl_pred, "control")

        all_results[cond] = dict(
            zero_shot_prediction=pred,
            zero_shot_generation=gen,
            zero_shot_mean_rank=mean_r,
            zero_shot_top10=t10,
            zero_shot_top100=t100,
            zero_shot_control_mean_rank=ctrl_mean,
            zero_shot_control_top10=ctrl_t10,
        )

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # SFT convergence comparison
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SFT CONVERGENCE")
    print("=" * 60)

    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    split = int(0.9 * len(curriculum))
    train_data, val_data = curriculum[:split], curriculum[split:]

    steps_per_epoch = max(1, len(train_data) // (sft_batch_size * block_size))
    n_steps = sft_n_epochs * steps_per_epoch
    print(f"Curriculum: {len(curriculum):,} tokens, "
          f"{n_steps} steps ({sft_n_epochs} epochs)")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i + bl] for i in ix])
        y = torch.stack([data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    eval_subset = QUEEN_PREDICTION_PROMPTS[:4]

    for cond, init in conditions.items():
        print(f"\n--- SFT: {cond} ---")

        torch.manual_seed(42)
        model = make_model(init)
        opt = torch.optim.AdamW(model.parameters(), lr=sft_lr,
                                weight_decay=0.01)

        curve = []
        best_val = float("inf")
        best_sd_cond = None

        for step in range(n_steps):
            model.train()
            x, y = get_batch(train_data, sft_batch_size, block_size)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % sft_eval_interval == 0 or step == n_steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = sum(
                        float(model(
                            *get_batch(val_data, sft_batch_size, block_size)
                        )[1])
                        for _ in range(5)
                    ) / 5.0

                pr = eval_prediction(model, eval_subset)
                qr = float(np.mean([r["queen_rank"] for r in pr]))

                if vl < best_val:
                    best_val = vl
                    best_sd_cond = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

                curve.append(dict(step=step, train_loss=float(loss),
                                  val_loss=vl, queen_mean_rank=qr))
                print(f"  step {step:3d}/{n_steps}: "
                      f"train={loss:.4f} val={vl:.4f} "
                      f"best={best_val:.4f} queen={qr:.0f}")

        # Final eval on best checkpoint
        if best_sd_cond:
            model.load_state_dict(best_sd_cond)
            model = model.to(device)
        model.eval()

        final_pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        final_gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        print(f"\n  Post-SFT results:")
        for r in final_pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"    [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")
        for g in final_gen:
            print(f"    \"{g['prompt']}\" -> "
                  f"\"{g['continuation'][:60]}\"")

        final_mean, final_t10, _ = summarize_pred(final_pred, "post-SFT")

        first_top10 = None
        for pt in curve:
            if pt["queen_mean_rank"] <= 10:
                first_top10 = pt["step"]
                break

        all_results[cond].update(dict(
            sft_curve=curve,
            sft_best_val_loss=best_val,
            sft_final_prediction=final_pred,
            sft_final_generation=final_gen,
            sft_final_mean_rank=final_mean,
            sft_final_top10=final_t10,
            sft_first_top10_step=first_top10,
        ))

        del model, opt, best_sd_cond
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for cond in conditions:
        r = all_results[cond]
        zs = (f"queen_rank={r['zero_shot_mean_rank']:.1f} "
              f"({r['zero_shot_top10']}/12 top10), "
              f"control_rank={r['zero_shot_control_mean_rank']:.1f} "
              f"({r['zero_shot_control_top10']}/12 top10)")
        ps = f"rank={r['sft_final_mean_rank']:.1f}, top10={r['sft_final_top10']}/12"
        ft = r.get("sft_first_top10_step")
        ft_s = f"step {ft}" if ft is not None else "never"
        print(f"\n  {cond}:")
        print(f"    zero-shot:  {zs}")
        print(f"    post-SFT:   {ps}  (best_val={r['sft_best_val_loss']:.4f})")
        print(f"    first top10: {ft_s}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    results = dict(
        config=dict(
            source_tau=source_tau, source_P=source_P,
            block_size=block_size, t_start=t_start,
            num_denoise_steps=num_denoise_steps,
            n_denoise_avg=n_denoise_avg,
            sft_lr=sft_lr, sft_n_epochs=sft_n_epochs,
            n_steps=n_steps,
            queen_ids=queen_ids, king_id=king_id,
            tie_weights=True,
        ),
        diagnostics=dict(
            glp_mean_pairwise_cos=mean_pairwise_cos,
            glp_coherence_ratio=float(coherence),
            cos_glp_vs_contrastive=cos_glp_contrast,
            cos_glp_vs_king_wte=cos_glp_king,
            per_prompt_diagnostics=per_prompt_diagnostics,
        ),
        conditions=all_results,
    )

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    fname = f"warm_init_glp_embed_tau{source_tau:.1f}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/{fname}")
    return results


# ======================================================================
# In-context grokking initialization
# ======================================================================

@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=3600,
    memory=32768,
)
def warm_init_icl_stage(
    source_tau: float = 0.3,
    source_P: int = 100_000_000,
    block_size: int = 128,
    sft_lr: float = 3e-4,
    sft_n_epochs: int = 5,
    sft_batch_size: int = 64,
    sft_eval_interval: int = 2,
):
    """In-context grokking for concept embedding initialization.

    Uses 2x2 factorial priming (gender x royalty) to elicit the queen
    concept through in-context learning. The model's attention mechanism,
    when given female-royalty-rich context, temporarily composes gender
    and royalty representations into a coherent composite. Capturing that
    composed activation gives an embedding initialization.

    Four conditions:
      1. random:       original random init
      2. contrastive:  centroid(queen_h1) - centroid(control_h1) (current best)
      3. icl_primed:   mean(female_royal_h1 - neutral_h1), scaled
      4. interaction:  mean(female_royal - male_royal - female_nonroyal + neutral)
                       isolates the gender x royalty binding term
    """
    import os, json, torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    # ------------------------------------------------------------------ #
    # Token ID lookup
    # ------------------------------------------------------------------ #
    def find_token_ids(word):
        ids = []
        for tid in range(50257):
            if enc.decode([tid]).strip().lower() == word:
                ids.append(tid)
        return ids

    queen_ids = find_token_ids("queen")
    king_ids = find_token_ids("king")
    queen_id = queen_ids[0]
    king_id = king_ids[0]

    female_primary = [find_token_ids(w)[0] for w in FEMALE_WORDS if find_token_ids(w)]
    male_primary = [find_token_ids(w)[0] for w in MALE_WORDS if find_token_ids(w)]

    print(f"Queen IDs: {queen_ids} (primary {queen_id})")
    print(f"King IDs:  {king_ids} (primary {king_id})")
    print(f"Gender words: {len(female_primary)} female, {len(male_primary)} male")

    # ------------------------------------------------------------------ #
    # Load tied-weights model
    # ------------------------------------------------------------------ #
    model_dir = (f"{DATA_DIR}/models/tau_{source_tau:.3f}"
                 f"/P_{source_P}/T_{block_size}_tied")
    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)

    assert queen_id < vocab_size, (
        f"queen_id={queen_id} >= vocab_size={vocab_size}")

    model_kwargs = dict(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
        tie_weights=True,
    )
    print(f"Model: vocab={vocab_size}, layers={n_layer}, d={n_embd} (tied weights)")

    # ------------------------------------------------------------------ #
    # Extract activations under all prefix conditions
    # ------------------------------------------------------------------ #
    tmp = GPT(**model_kwargs).to(device)
    tmp.load_state_dict(sd)
    tmp.eval()

    def extract_lnf_h1_last(text):
        tokens = enc.encode(text)
        assert len(tokens) <= block_size, (
            f"Text too long: {len(tokens)} > {block_size}")
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            tok_emb = tmp.transformer.wte(ids)
            if hasattr(tmp.transformer, 'wpe'):
                pos = torch.arange(ids.shape[1], device=device)
                tok_emb = tok_emb + tmp.transformer.wpe(pos)
            h = tmp.transformer.drop(tok_emb)
            for block in tmp.transformer.h:
                h = block(h)
            h = tmp.transformer.ln_f(h)
        return h[0, -1, :]

    prefixes = {
        "female_royal": FEMALE_ROYAL_PREFIX,
        "male_royal": MALE_ROYAL_PREFIX,
        "female_nonroyal": FEMALE_NONROYAL_PREFIX,
        "neutral": NEUTRAL_PREFIX,
    }

    # Verify token counts
    print("\n=== Prefix token counts ===")
    for name, prefix in prefixes.items():
        n_tok = len(enc.encode(prefix))
        longest_prompt = max(len(enc.encode(p)) for p in QUEEN_PREDICTION_PROMPTS)
        print(f"  {name}: {n_tok} tokens "
              f"(+ longest prompt {longest_prompt} = {n_tok + longest_prompt + 1})")

    # Extract activations: primed[prefix_name] = tensor(12, n_embd)
    print("\n=== Extracting primed activations ===")
    primed = {}
    for pname, prefix in prefixes.items():
        acts = []
        for prompt in QUEEN_PREDICTION_PROMPTS:
            text = prefix + " " + prompt
            acts.append(extract_lnf_h1_last(text))
        primed[pname] = torch.stack(acts)
        print(f"  {pname}: extracted {primed[pname].shape}")

    # Also extract bare (no prefix) activations
    bare_queen = torch.stack([extract_lnf_h1_last(p)
                              for p in QUEEN_PREDICTION_PROMPTS])
    bare_control = torch.stack([extract_lnf_h1_last(p)
                                for p in CONTROL_PROMPTS])

    # ------------------------------------------------------------------ #
    # Diagnostics: how much does priming change activations?
    # ------------------------------------------------------------------ #
    print("\n=== Priming effect diagnostics ===")
    for pname in prefixes:
        cos = F.cosine_similarity(primed[pname], bare_queen, dim=-1)
        print(f"  cos(primed_{pname}, bare): "
              f"mean={cos.mean():.4f}, std={cos.std():.4f}")

    # Check if priming alone helps queen rank (no embedding modification)
    print("\n=== Queen rank under priming (original model) ===")
    for pname, prefix in prefixes.items():
        ranks = []
        for prompt in QUEEN_PREDICTION_PROMPTS[:4]:
            text = prefix + " " + prompt
            tokens = enc.encode(text)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = tmp(ids)
            last = logits[0, -1, :]
            best_rank = vocab_size
            for qid in queen_ids:
                r = int((last > last[qid]).sum()) + 1
                if r < best_rank:
                    best_rank = r
            ranks.append(best_rank)
        print(f"  {pname}: mean queen rank = {np.mean(ranks):.1f} "
              f"(ranks: {ranks})")
    # Bare comparison
    bare_ranks = []
    for prompt in QUEEN_PREDICTION_PROMPTS[:4]:
        tokens = enc.encode(prompt)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = tmp(ids)
        last = logits[0, -1, :]
        best_rank = vocab_size
        for qid in queen_ids:
            r = int((last > last[qid]).sum()) + 1
            if r < best_rank:
                best_rank = r
        bare_ranks.append(best_rank)
    print(f"  bare: mean queen rank = {np.mean(bare_ranks):.1f} "
          f"(ranks: {bare_ranks})")

    # ------------------------------------------------------------------ #
    # Compute ICL signatures
    # ------------------------------------------------------------------ #
    print("\n=== ICL signatures ===")

    wte = sd["transformer.wte.weight"]
    target_norm = wte.norm(dim=1).mean()

    # Contrastive centroid (existing best, bare prompts)
    contrast = bare_queen.mean(dim=0) - bare_control.mean(dim=0)
    contrastive_emb = contrast.cpu() * (target_norm / contrast.cpu().norm())
    print(f"  contrastive norm (raw): {contrast.norm():.4f}")

    # ICL primed: female_royal - neutral, per-prompt then average
    icl_per_prompt = primed["female_royal"] - primed["neutral"]
    icl_mean = icl_per_prompt.mean(dim=0)
    icl_emb = icl_mean.cpu() * (target_norm / icl_mean.cpu().norm())

    # Interaction: female_royal - male_royal - female_nonroyal + neutral
    interaction_per_prompt = (primed["female_royal"] - primed["male_royal"]
                              - primed["female_nonroyal"] + primed["neutral"])
    interaction_mean = interaction_per_prompt.mean(dim=0)
    interaction_emb = interaction_mean.cpu() * (target_norm / interaction_mean.cpu().norm())

    # Per-prompt coherence for each signature
    def coherence_stats(per_prompt_vecs, name):
        normed = F.normalize(per_prompt_vecs, dim=-1)
        cos_mat = normed @ normed.T
        n = cos_mat.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_mat.device)
        mean_cos = float(cos_mat[mask].mean())
        ratio = per_prompt_vecs.mean(dim=0).norm() / per_prompt_vecs.norm(dim=-1).mean()
        print(f"  {name}: pairwise_cos={mean_cos:.4f}, "
              f"coherence_ratio={float(ratio):.4f}, "
              f"mean_norm={per_prompt_vecs.norm(dim=-1).mean():.4f}")
        return mean_cos, float(ratio)

    icl_cos, icl_ratio = coherence_stats(icl_per_prompt, "icl_primed")
    int_cos, int_ratio = coherence_stats(interaction_per_prompt, "interaction")

    # Also compute contrastive coherence for comparison
    # (contrastive uses bare_queen directly, each prompt is a "per-prompt" vec)
    bare_centered = bare_queen - bare_control.mean(dim=0)
    contr_cos, contr_ratio = coherence_stats(bare_centered, "contrastive (bare_queen - control_mean)")

    # Cross-signature cosine similarities
    cos_icl_contr = F.cosine_similarity(
        icl_emb.unsqueeze(0), contrastive_emb.unsqueeze(0)).item()
    cos_int_contr = F.cosine_similarity(
        interaction_emb.unsqueeze(0), contrastive_emb.unsqueeze(0)).item()
    cos_icl_int = F.cosine_similarity(
        icl_emb.unsqueeze(0), interaction_emb.unsqueeze(0)).item()
    cos_icl_king = F.cosine_similarity(
        icl_emb.unsqueeze(0), wte[king_id].unsqueeze(0)).item()

    print(f"\n  Cross-signature cosines:")
    print(f"    cos(icl_primed, contrastive):  {cos_icl_contr:+.4f}")
    print(f"    cos(interaction, contrastive): {cos_int_contr:+.4f}")
    print(f"    cos(icl_primed, interaction):  {cos_icl_int:+.4f}")
    print(f"    cos(icl_primed, king_wte):     {cos_icl_king:+.4f}")

    # Factorial decomposition: marginal effects
    all_primed = torch.stack([primed[k].mean(dim=0) for k in prefixes])
    female_effect = ((primed["female_royal"].mean(0) + primed["female_nonroyal"].mean(0))
                     - (primed["male_royal"].mean(0) + primed["neutral"].mean(0))) / 2
    royal_effect = ((primed["female_royal"].mean(0) + primed["male_royal"].mean(0))
                    - (primed["female_nonroyal"].mean(0) + primed["neutral"].mean(0))) / 2
    interaction_effect = interaction_mean / 1.0  # already computed

    print(f"\n  Factorial decomposition (norms):")
    print(f"    female effect:      {female_effect.norm():.4f}")
    print(f"    royal effect:       {royal_effect.norm():.4f}")
    print(f"    interaction effect: {interaction_effect.norm():.4f}")
    print(f"    cos(female, royal): "
          f"{F.cosine_similarity(female_effect.unsqueeze(0), royal_effect.unsqueeze(0)).item():+.4f}")

    del tmp
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Analogy initialization
    # ------------------------------------------------------------------ #
    gender_wte = (torch.stack([wte[i] for i in female_primary]).mean(0)
                  - torch.stack([wte[i] for i in male_primary]).mean(0))
    analogy_emb = wte[king_id] + gender_wte

    # ------------------------------------------------------------------ #
    # Conditions (all tied: wte = lm_head)
    # ------------------------------------------------------------------ #
    conditions = {
        "random": {
            "wte": wte[queen_id].clone(),
            "lm_head": wte[queen_id].clone(),
        },
        "contrastive": {
            "wte": contrastive_emb,
            "lm_head": contrastive_emb.clone(),
        },
        "icl_primed": {
            "wte": icl_emb,
            "lm_head": icl_emb.clone(),
        },
        "interaction": {
            "wte": interaction_emb,
            "lm_head": interaction_emb.clone(),
        },
    }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def make_model(init_dict):
        model = GPT(**model_kwargs).to(device)
        csd = {k: v.clone() for k, v in sd.items()}
        for qid in queen_ids:
            csd["transformer.wte.weight"][qid] = init_dict["wte"].clone()
            if "lm_head.weight" in csd:
                csd["lm_head.weight"][qid] = init_dict["lm_head"].clone()
        model.load_state_dict(csd)
        return model

    def eval_prediction(model, prompts):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(ids)
            last = logits[0, -1, :]
            probs = F.softmax(last, dim=-1)

            best_rank, best_prob = vocab_size, 0.0
            for qid in queen_ids:
                r = int((last > last[qid]).sum()) + 1
                p = float(probs[qid])
                if r < best_rank:
                    best_rank, best_prob = r, p

            top_v, top_i = probs.topk(5)
            top5 = [(enc.decode([int(i)]), float(v))
                    for i, v in zip(top_i, top_v)]
            results.append(dict(prompt=prompt, queen_rank=best_rank,
                                queen_prob=best_prob, top5=top5))
        return results

    def eval_generation(model, prompts, max_tokens=20):
        results = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            gen = list(tokens)
            with torch.no_grad():
                for _ in range(max_tokens):
                    inp = torch.tensor(
                        [gen[-block_size:]], dtype=torch.long, device=device)
                    logits, _ = model(inp)
                    gen.append(logits[0, -1, :].argmax().item())
            results.append(dict(prompt=prompt,
                                continuation=enc.decode(gen[len(tokens):])))
        return results

    def summarize_pred(pred, label=""):
        ranks = [r["queen_rank"] for r in pred]
        m = float(np.mean(ranks))
        t10 = sum(1 for r in ranks if r <= 10)
        t100 = sum(1 for r in ranks if r <= 100)
        if label:
            print(f"  {label}: mean_rank={m:.1f}, "
                  f"top-10={t10}/{len(ranks)}, top-100={t100}/{len(ranks)}")
        return m, t10, t100

    # ------------------------------------------------------------------ #
    # Zero-shot evaluation
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION")
    print("=" * 60)

    all_results = {}

    for cond, init in conditions.items():
        print(f"\n--- {cond} ---")
        model = make_model(init)
        model.eval()

        pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        for r in pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"  [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")

        mean_r, t10, t100 = summarize_pred(pred, "prediction")

        for g in gen:
            print(f"  \"{g['prompt']}\" -> "
                  f"\"{g['continuation'][:60]}\"")

        ctrl_pred = eval_prediction(model, CONTROL_PROMPTS)
        ctrl_mean, ctrl_t10, _ = summarize_pred(ctrl_pred, "control")

        all_results[cond] = dict(
            zero_shot_prediction=pred,
            zero_shot_generation=gen,
            zero_shot_mean_rank=mean_r,
            zero_shot_top10=t10,
            zero_shot_top100=t100,
            zero_shot_control_mean_rank=ctrl_mean,
            zero_shot_control_top10=ctrl_t10,
        )

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # SFT convergence comparison
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SFT CONVERGENCE")
    print("=" * 60)

    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    split = int(0.9 * len(curriculum))
    train_data, val_data = curriculum[:split], curriculum[split:]

    steps_per_epoch = max(1, len(train_data) // (sft_batch_size * block_size))
    n_steps = sft_n_epochs * steps_per_epoch
    print(f"Curriculum: {len(curriculum):,} tokens, "
          f"{n_steps} steps ({sft_n_epochs} epochs)")

    def get_batch(data, bs, bl):
        ix = torch.randint(len(data) - bl - 1, (bs,))
        x = torch.stack([data[i:i + bl] for i in ix])
        y = torch.stack([data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    eval_subset = QUEEN_PREDICTION_PROMPTS[:4]

    for cond, init in conditions.items():
        print(f"\n--- SFT: {cond} ---")

        torch.manual_seed(42)
        model = make_model(init)
        opt = torch.optim.AdamW(model.parameters(), lr=sft_lr,
                                weight_decay=0.01)

        curve = []
        best_val = float("inf")
        best_sd_cond = None

        for step in range(n_steps):
            model.train()
            x, y = get_batch(train_data, sft_batch_size, block_size)
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % sft_eval_interval == 0 or step == n_steps - 1:
                model.eval()
                with torch.no_grad():
                    vl = sum(
                        float(model(
                            *get_batch(val_data, sft_batch_size, block_size)
                        )[1])
                        for _ in range(5)
                    ) / 5.0

                pr = eval_prediction(model, eval_subset)
                qr = float(np.mean([r["queen_rank"] for r in pr]))

                if vl < best_val:
                    best_val = vl
                    best_sd_cond = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}

                curve.append(dict(step=step, train_loss=float(loss),
                                  val_loss=vl, queen_mean_rank=qr))
                print(f"  step {step:3d}/{n_steps}: "
                      f"train={loss:.4f} val={vl:.4f} "
                      f"best={best_val:.4f} queen={qr:.0f}")

        # Final eval on best checkpoint
        if best_sd_cond:
            model.load_state_dict(best_sd_cond)
            model = model.to(device)
        model.eval()

        final_pred = eval_prediction(model, QUEEN_PREDICTION_PROMPTS)
        final_gen = eval_generation(model, QUEEN_INPUT_PROMPTS)

        print(f"\n  Post-SFT results:")
        for r in final_pred:
            t5 = ", ".join(f"{t}({p:.4f})" for t, p in r["top5"])
            print(f"    [{r['queen_rank']:>5d}] "
                  f"{r['prompt'][:50]:50s} {t5}")
        for g in final_gen:
            print(f"    \"{g['prompt']}\" -> "
                  f"\"{g['continuation'][:60]}\"")

        final_mean, final_t10, _ = summarize_pred(final_pred, "post-SFT")

        first_top10 = None
        for pt in curve:
            if pt["queen_mean_rank"] <= 10:
                first_top10 = pt["step"]
                break

        all_results[cond].update(dict(
            sft_curve=curve,
            sft_best_val_loss=best_val,
            sft_final_prediction=final_pred,
            sft_final_generation=final_gen,
            sft_final_mean_rank=final_mean,
            sft_final_top10=final_t10,
            sft_first_top10_step=first_top10,
        ))

        del model, opt, best_sd_cond
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for cond in conditions:
        r = all_results[cond]
        zs = (f"queen_rank={r['zero_shot_mean_rank']:.1f} "
              f"({r['zero_shot_top10']}/12 top10), "
              f"control_rank={r['zero_shot_control_mean_rank']:.1f} "
              f"({r['zero_shot_control_top10']}/12 top10)")
        ps = f"rank={r['sft_final_mean_rank']:.1f}, top10={r['sft_final_top10']}/12"
        ft = r.get("sft_first_top10_step")
        ft_s = f"step {ft}" if ft is not None else "never"
        print(f"\n  {cond}:")
        print(f"    zero-shot:  {zs}")
        print(f"    post-SFT:   {ps}  (best_val={r['sft_best_val_loss']:.4f})")
        print(f"    first top10: {ft_s}")

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    results = dict(
        config=dict(
            source_tau=source_tau, source_P=source_P,
            block_size=block_size,
            sft_lr=sft_lr, sft_n_epochs=sft_n_epochs,
            n_steps=n_steps,
            queen_ids=queen_ids, king_id=king_id,
            tie_weights=True,
        ),
        diagnostics=dict(
            icl_primed_coherence=icl_cos,
            icl_primed_ratio=icl_ratio,
            interaction_coherence=int_cos,
            interaction_ratio=int_ratio,
            contrastive_coherence=contr_cos,
            contrastive_ratio=contr_ratio,
            cos_icl_vs_contrastive=cos_icl_contr,
            cos_interaction_vs_contrastive=cos_int_contr,
            cos_icl_vs_interaction=cos_icl_int,
            female_effect_norm=float(female_effect.norm()),
            royal_effect_norm=float(royal_effect.norm()),
            interaction_effect_norm=float(interaction_effect.norm()),
        ),
        conditions=all_results,
    )

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    fname = f"warm_init_icl_tau{source_tau:.1f}.json"
    with open(f"{results_dir}/{fname}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/{fname}")
    return results
