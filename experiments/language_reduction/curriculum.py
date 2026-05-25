"""Curriculum extraction for concept recovery experiment.

Scans the original (un-denoised) corpus for windows containing target
words and packages them as a fine-tuning dataset.
"""

import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import tiktoken


TARGET_WORDS = [
    "queen", "empress", "princess", "goddess", "heroine",
    "priestess", "duchess", "countess", "baroness",
]


@dataclass
class CurriculumConfig:
    target_tokens: int = 50_000
    window_size: int = 128
    window_stride: int = 64
    max_shards: int = 10


def build_token_lookup() -> dict[str, set[int]]:
    """Build cleaned-string-to-token-IDs mapping for the full GPT-2 vocabulary."""
    enc = tiktoken.get_encoding("gpt2")
    lookup: dict[str, set[int]] = {}
    for token_id in range(50257):
        raw = enc.decode([token_id])
        cleaned = raw.strip().lower()
        if cleaned:
            if cleaned not in lookup:
                lookup[cleaned] = set()
            lookup[cleaned].add(token_id)
    return lookup


def extract_curriculum(tokens_dir: str, config: CurriculumConfig) -> np.ndarray:
    """Extract training windows containing target words from the original corpus."""
    lookup = build_token_lookup()

    target_token_ids: set[int] = set()
    for word in TARGET_WORDS:
        if word in lookup:
            target_token_ids.update(lookup[word])

    print(f"Target words: {TARGET_WORDS}")
    print(f"Target token IDs: {len(target_token_ids)} IDs across {len(TARGET_WORDS)} words")

    windows: list[np.ndarray] = []
    total_tokens = 0
    per_word_counts: dict[str, int] = {w: 0 for w in TARGET_WORDS}
    last_window_end = -1
    n_shards_scanned = 0

    enc = tiktoken.get_encoding("gpt2")
    id_to_word: dict[int, str] = {}
    for word in TARGET_WORDS:
        if word in lookup:
            for tid in lookup[word]:
                id_to_word[tid] = word

    for shard_idx in range(config.max_shards):
        shard_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        if not os.path.exists(shard_path):
            print(f"  Shard {shard_idx} not found, stopping scan")
            break

        shard = np.load(shard_path)
        n_shards_scanned += 1
        last_window_end = -1

        for pos in range(len(shard)):
            if total_tokens >= config.target_tokens:
                break

            token_id = int(shard[pos])
            if token_id not in target_token_ids:
                continue

            # Check overlap with previous window
            window_start = max(0, pos - config.window_size // 2)
            window_end = min(len(shard), window_start + config.window_size)
            window_start = window_end - config.window_size

            if window_start < 0:
                window_start = 0
                window_end = min(len(shard), config.window_size)

            overlap = max(0, last_window_end - window_start)
            if overlap > config.window_size // 2:
                per_word_counts[id_to_word[token_id]] += 1
                continue

            window = shard[window_start:window_end]
            windows.append(window)
            total_tokens += len(window)
            last_window_end = window_end
            per_word_counts[id_to_word[token_id]] += 1

        if total_tokens >= config.target_tokens:
            break

    print(f"\nExtraction stats:")
    print(f"  Windows: {len(windows)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Shards scanned: {n_shards_scanned}")
    print(f"  Per-word hits:")
    for word, count in sorted(per_word_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {word:>12s}: {count}")

    if total_tokens < config.target_tokens:
        print(f"\n  WARNING: Only found {total_tokens:,} tokens "
              f"(target: {config.target_tokens:,})")

    if not windows:
        return np.array([], dtype=np.int64)

    return np.concatenate(windows).astype(np.int64)


def save_curriculum(tokens: np.ndarray, output_dir: str, config: CurriculumConfig = None,
                    stats: dict = None):
    """Save curriculum to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "curriculum.npy"), tokens)

    meta = {
        "n_tokens": len(tokens),
        "n_windows": len(tokens) // (config.window_size if config else 128),
        "target_words": TARGET_WORDS,
    }
    if config:
        meta["config"] = asdict(config)

    with open(os.path.join(output_dir, "curriculum_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved curriculum: {len(tokens):,} tokens to {output_dir}")
