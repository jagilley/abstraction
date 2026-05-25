# Experiment: Concept Recovery via Fine-Tuning

**Status**: Spec (not yet implemented)
**Date**: 2026-04-28
**Builds on**: `ideas/language_reduction_continual_learning.md`, `ideas/self_intervention_via_local_replay.md`
**Prerequisites**: All existing pipeline stages complete (tokenize through contextual-eval)

## Motivation

The τ=0.3 model has a clean "female person of status" direction (king−man+woman yields wife, mother, Mary, President, She, daughter, female) but can't express it as a single token. The "queen" token exists in the vocabulary (V=50257) but is essentially untrained — its embedding sits at random initialization because the denoiser replaced queen-containing positions during training.

Fine-tuning this model on diverse contexts containing "queen" from the original corpus is effectively the same as minting a fresh token: you're training a new embedding from scratch and establishing all its contextual routing through backprop. The token serves as a backprop handle — a focal point through which gradients flow to restructure surrounding representations.

**Core question**: Does training a token into a model cause *structural integration* — does the neighborhood restructure (king moves, the gender-authority subspace sharpens) — or does the model just memorize the word in isolation?

If integration is structural, it validates the foundational claim behind the continual learning ideas in `ideas/language_reduction_continual_learning.md` and `ideas/self_intervention_via_local_replay.md`. If it's additive, those ideas need rethinking.

## Experimental Design

### Setup

- **Starting model**: τ=0.3, P=100M, T=128 (at `/data/models/tau_0.300/P_100000000/T_128/model.pt`)
- **Ground truth**: τ=0.0, P=100M, T=128 (at `/data/models/tau_0.000/P_100000000/T_128/model.pt`)
- **Architecture**: 2-layer, 4-head, 128-dim GPT-2 (V=50257). See `model.py`.

### Intervention

Extract ~50K tokens of text from the **original un-denoised corpus** (`/data/tokens/`) in windows where "queen" (and related degraded words: empress, princess, goddess, heroine, priestess, duchess, countess, baroness) appear. These are contexts that the τ=0.3 model never saw during training because the denoiser replaced the target words.

Fine-tune the τ=0.3 model on this curriculum with a reduced learning rate (1e-4, vs 3e-4 during pre-training), saving the best checkpoint by validation loss.

### Measurements

Compare the fine-tuned model against both the starting τ=0.3 model and the τ=0.0 ground truth.

**M1. Does king move?**
Cosine similarity between king's embedding before and after fine-tuning. If positive and directed toward the τ=0.0 model's king embedding, integration is structural. Compute `toward_gt = cos(emb_ft["king"], emb_gt["king"]) - cos(emb_start["king"], emb_gt["king"])`. Positive means king moved toward ground truth. Report this metric for all gender-related words: king, man, woman, he, she, mother, father, wife, husband, son, daughter.

**M2. Does the gender-authority subspace restructure?**
For gender-related words, compute top-10 nearest neighbors (cosine, restricted to active vocabulary of 3200 tokens) in three embedding spaces: fine-tuned, starting (τ=0.3), and ground truth (τ=0.0). For each word, compute neighbor overlap: `|neighbors_ft ∩ neighbors_gt| / 10` vs the baseline `|neighbors_start ∩ neighbors_gt| / 10`. If fine-tuning increases overlap with ground truth, the neighborhood restructured.

**M3. Does king−man+woman sharpen?**
Run the existing `analogy_spot_check` on the fine-tuned model (king − man + woman = ?). Compare nearest neighbors qualitatively against the τ=0.3 baseline (scattered: Church, child, wife, king, baby...) and τ=0.0 baseline (also somewhat scattered but better gender-relevant).

**M4. Analogy accuracy recovery.**
Run the existing `eval_analogies` from `eval_embeddings.py`. Key categories:
- Gender: 0.125 at τ=0.3, 0.250 at τ=0.0 (should improve)
- Comparative: 0.375 vs 0.625 (should be unchanged — not targeted)
- Tense: 0.714 at τ=0.3 (forgetting check — must not degrade)
- Plural: 0.733 at τ=0.3 (forgetting check — must not degrade)

**M5. Forgetting.**
NN coherence on non-gender query words (science, good, water, school, large, old, write, day, world, red) must stay stable. Analogy accuracy on tense and plural must not degrade.

### Predictions

1. Queen-related analogy accuracy recovers.
2. **King's embedding moves toward the τ=0.0 model's king.** This is the structural integration test.
3. Neighbor overlap with τ=0.0 increases for gender-related words.
4. Tense and plural are unaffected.

### Follow-up experiments (if the core experiment succeeds)

These are NOT part of the current implementation — they're what we'd try next:
- **Principle-targeted curriculum**: Can you achieve the same restructuring using contexts that exercise the gender-authority *composition* without any target words? (Tests "curriculum teaches composition, not vocabulary.")
- **Warm initialization from GLP residual**: Initialize the queen embedding from the GLP's velocity field at the off-manifold "female person of status" direction, rather than from random init. (Tests whether GLP-guided init speeds up few-shot concept learning.)
- **Minted tokens for novel concepts**: Extend to concepts that genuinely don't have existing vocabulary entries.

---

## Implementation Spec

### File 1: `curriculum.py` — Curriculum extraction

New file in `language_reduction/`. Extracts training windows from the original corpus.

```python
"""Curriculum extraction for concept recovery experiment.

Scans the original (un-denoised) corpus for windows containing target
words and packages them as a fine-tuning dataset.
"""
```

#### Data structures

```python
from dataclasses import dataclass

@dataclass
class CurriculumConfig:
    target_tokens: int = 50_000        # total tokens to extract
    window_size: int = 128             # token window size (match model block_size)
    window_stride: int = 64            # stride between candidate windows
    max_shards: int = 10               # how many original shards to scan
```

#### Word list

```python
TARGET_WORDS = [
    "queen", "empress", "princess", "goddess", "heroine",
    "priestess", "duchess", "countess", "baroness",
]
```

#### Token ID resolution

All words must be resolved to sets of GPT-2 BPE token IDs at runtime using tiktoken. A single word may map to multiple token IDs (e.g., " queen" vs "queen" vs "Queen"). Use the same cleaning logic as `eval_embeddings.py:build_token_index()`:
1. Iterate over all 50257 token IDs
2. Decode each with tiktoken
3. Strip whitespace and lowercase
4. Build a reverse mapping: cleaned_string -> set of token IDs

When checking whether a token in the corpus is a target word, look up its token ID in this reverse mapping.

#### Functions

**`build_token_lookup() -> dict[str, set[int]]`**

Build the cleaned-string-to-token-IDs mapping described above. Return a dict where keys are lowercase stripped words and values are sets of token IDs that decode to that word.

**`extract_curriculum(tokens_dir, config) -> np.ndarray`**

1. Build the token lookup.
2. Compute the set of all token IDs that correspond to any TARGET_WORDS entry.
3. Iterate over shards (`shard_00000.npy` through `shard_{max_shards-1:05d}.npy`).
4. For each shard, scan positions. When a position's token ID is in the target set, extract a window of `window_size` tokens centered on that position (clamped to shard boundaries). Skip if this window overlaps with a previously extracted window by more than half.
5. Collect windows until reaching `target_tokens` total tokens.
6. Concatenate all windows into a single 1D int64 array.
7. Print statistics: n_windows found, n_tokens total, n_shards scanned, per-word hit counts.

If not enough windows are found, print a warning and return what was collected.

**`save_curriculum(tokens, output_dir)`**

Save as `{output_dir}/curriculum.npy`. Also save `{output_dir}/curriculum_meta.json` containing: config values, word list, extraction stats (n_windows, n_tokens, per-word counts, n_shards_scanned).

#### Notes

- Work in the ORIGINAL token space (full GPT-2 V=50257). The curriculum is used for fine-tuning the full-vocab model.
- The scan is simple: load each shard as an int64 array, iterate positions, check membership in the target token ID set. This is O(n_tokens) per shard — trivially fast for 10 shards of 10M tokens each.
- The `window_stride` / overlap check prevents extracting near-duplicate windows when target words appear in clusters.

---

### File 2: `recovery_eval.py` — Cross-model comparison metrics

New file in `language_reduction/`. Metrics that compare embeddings across models.

```python
"""Cross-model embedding comparison for concept recovery experiment.

Compares a fine-tuned model's embeddings against a ground-truth model
to measure structural integration of recovered concepts.
"""
```

#### Constants

```python
GENDER_WORDS = [
    "king", "man", "woman", "he", "she",
    "mother", "father", "wife", "husband",
    "son", "daughter", "boy", "girl",
    "brother", "sister",
]
```

#### Functions

**`embedding_shift_toward_gt(emb_ft, emb_start, emb_gt, token_index) -> dict`**

For each word in GENDER_WORDS (plus the TARGET_WORDS from curriculum.py):
1. Look up the token ID via `token_index["str_to_id"]`.
2. Compute `cos_ft_gt = cosine(emb_ft[tid], emb_gt[tid])`.
3. Compute `cos_start_gt = cosine(emb_start[tid], emb_gt[tid])`.
4. Compute `toward_gt = cos_ft_gt - cos_start_gt`. Positive = moved toward ground truth.
5. Compute `shift_magnitude = 1 - cosine(emb_ft[tid], emb_start[tid])`. How much the embedding moved at all.

Return a dict with per-word results and summary stats:
```python
{
    "per_word": {
        "king": {"toward_gt": 0.05, "shift": 0.12, "cos_ft_gt": 0.83, "cos_start_gt": 0.78},
        ...
    },
    "mean_toward_gt_gender": float,   # mean over GENDER_WORDS
    "mean_toward_gt_target": float,   # mean over TARGET_WORDS
    "mean_shift_gender": float,
}
```

**`neighbor_overlap(emb_ft, emb_gt, emb_start, active_ids, token_index, k=10) -> dict`**

1. Compute normalized embedding matrices (L2-normalize rows) for all three models, restricted to `active_ids`.
2. For each word in GENDER_WORDS:
   a. Look up token ID.
   b. Compute cosine similarities to all active tokens in each embedding space.
   c. Take top-k neighbors (excluding the word itself) for each space.
   d. Compute overlap scores:
      - `overlap_ft_gt = |set(neighbors_ft) & set(neighbors_gt)| / k`
      - `overlap_start_gt = |set(neighbors_start) & set(neighbors_gt)| / k` (baseline)
      - `improvement = overlap_ft_gt - overlap_start_gt`
3. Return per-word overlaps and means.

**`format_recovery_report(shifts, overlaps, eval_start, eval_ft, eval_gt) -> str`**

Format a human-readable comparison table. Takes the outputs of the above functions plus `run_eval` results for the three models (start, fine-tuned, ground truth).

Print sections:
1. **Embedding shifts** — per-word toward_gt for GENDER_WORDS, highlighting which words moved toward ground truth.
2. **Neighbor overlap** — per-word overlap_ft_gt vs overlap_start_gt, highlighting improvements.
3. **Analogy accuracy** — per-category comparison across all three models.
4. **Forgetting check** — tense and plural accuracy, NN coherence.

---

### File 3: Modifications to `modal_app.py` — New stages

Add three new stages and their CLI entrypoints.

#### Stage: `build-curriculum`

```python
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=16384,
)
def build_curriculum_stage(target_tokens: int = 50_000, max_shards: int = 10):
    """Extract curriculum from original corpus."""
    from language_reduction.curriculum import (
        CurriculumConfig, extract_curriculum, save_curriculum
    )

    config = CurriculumConfig(
        target_tokens=target_tokens,
        max_shards=max_shards,
    )
    tokens = extract_curriculum(f"{DATA_DIR}/tokens", config)
    save_curriculum(tokens, f"{DATA_DIR}/curriculum")
    volume.commit()
```

#### Stage: `finetune`

```python
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
):
    """Fine-tune a pre-trained model on the extracted curriculum.

    Loads the checkpoint from source_tau/source_P, trains on the
    curriculum at /data/curriculum/curriculum.npy, saves the best
    checkpoint by validation loss.
    """
    import os, json, torch
    import numpy as np
    from language_reduction.model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load curriculum
    curriculum = np.load(f"{DATA_DIR}/curriculum/curriculum.npy")
    curriculum = torch.from_numpy(curriculum.astype(np.int64))
    n_curriculum = len(curriculum)
    print(f"Curriculum: {n_curriculum:,} tokens")

    # Load source model
    source_dir = f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
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

    # Save best model
    save_dir = f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(),
               os.path.join(save_dir, "model.pt"))

    result = {
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
```

#### Stage: `recovery-eval`

```python
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def recovery_eval(source_tau: float = 0.3, source_P: int = 100_000_000):
    """Evaluate whether fine-tuning caused structural integration.

    Compares the fine-tuned model against:
      - The starting model (tau=source_tau) — did anything change?
      - The ground truth model (tau=0.0) — did it change in the right direction?

    Reports: embedding shifts toward ground truth, neighbor overlap,
    analogy accuracy recovery, and forgetting.
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
    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    def load_emb(model_dir):
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        return sd["transformer.wte.weight"].numpy()

    emb_start = load_emb(f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}")
    emb_gt = load_emb(f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}")
    emb_ft = load_emb(f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}")

    # Run all evaluations
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

    class _Enc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(f"{results_dir}/recovery_eval.json", "w") as f:
        json.dump(results, f, indent=2, cls=_Enc)
    volume.commit()
    print(f"\nSaved to {results_dir}/recovery_eval.json")
    return results
```

#### CLI entrypoints

Add to the `main()` function's stage dispatch:

```python
elif stage == "build-curriculum":
    build_curriculum_stage.remote()

elif stage == "finetune":
    finetune_model.remote(source_tau=tau)

elif stage == "recovery-eval":
    recovery_eval.remote(source_tau=tau)

elif stage == "recovery-full":
    build_curriculum_stage.remote()
    finetune_model.remote(source_tau=tau)
    recovery_eval.remote(source_tau=tau)
```

Also add these to the help string in the final `else` clause.

---

## Running the Experiment

```bash
# Step 1: Extract curriculum (~minutes, CPU)
modal run language_reduction/modal_app.py --stage build-curriculum

# Step 2: Fine-tune (~minutes, A10G GPU)
modal run language_reduction/modal_app.py --stage finetune --tau 0.3

# Step 3: Evaluate recovery (~minutes, CPU)
modal run language_reduction/modal_app.py --stage recovery-eval --tau 0.3

# Or all at once:
modal run language_reduction/modal_app.py --stage recovery-full --tau 0.3
```

## What Success Looks Like

1. **King's embedding moves toward τ=0.0's king** (positive `toward_gt` for "king" and other gender words). This is the primary signal — structural integration means the neighborhood restructures, not just the target word.
2. **Neighbor overlap with τ=0.0 increases** for gender-related words after fine-tuning.
3. **Gender analogy accuracy improves** from 0.125 toward 0.250.
4. **Tense and plural don't degrade** — no catastrophic forgetting.

The experiment is equally informative if it fails: if queen is learned but king doesn't move, that tells us token-level fine-tuning at this scale doesn't produce structural integration, which constrains all the continual learning ideas.

## Hyperparameter Sensitivity

The fine-tuning LR is the most important knob. Too high → catastrophic forgetting. Too low → 50K tokens aren't enough to move embeddings. The spec uses 1e-4 (3x lower than pre-training LR of 3e-4). If initial results show forgetting or no recovery, try:
- LR sweep: 3e-5, 1e-4, 3e-4
- Curriculum size sweep: 25K, 50K, 100K tokens

These are just additional `finetune_model` calls with different parameters.

## File Summary

| File | Type | Description |
|------|------|-------------|
| `curriculum.py` | NEW | Curriculum extraction from original corpus |
| `recovery_eval.py` | NEW | Cross-model comparison metrics (embedding shifts, neighbor overlap) |
| `modal_app.py` | MODIFY | Add `build-curriculum`, `finetune`, `recovery-eval`, `recovery-full` stages |
| `EXPERIMENT_CONCEPT_RECOVERY.md` | NEW | This spec |
