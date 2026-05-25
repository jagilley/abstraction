"""DPO pair extraction for the scaffolding RL experiment.

For each position where the denoiser replaced a queen-family token,
creates a minimal-diff (chosen, rejected) pair: the original window
vs. the same window with only the queen token swapped for the
denoiser's substitute.  This isolates the "prefer queen over
compositional alternative" signal.
"""

import os
from dataclasses import dataclass, asdict
import json

import numpy as np

from language_reduction.curriculum import TARGET_WORDS, build_token_lookup


@dataclass
class DPOPairsConfig:
    window_size: int = 128
    max_shards: int = 10
    max_pairs: int = 5000
    tau: float = 0.3


def extract_dpo_pairs(
    tokens_dir: str,
    denoised_dir: str,
    config: DPOPairsConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract (chosen, rejected) pairs from original vs denoised corpus.

    chosen[i] is the original window containing a queen-family token.
    rejected[i] is the identical window with that single token replaced
    by whatever the denoiser substituted.

    Returns:
        chosen:  (n_pairs, window_size) int64
        rejected: (n_pairs, window_size) int64
        stats: extraction statistics
    """
    import tiktoken

    lookup = build_token_lookup()
    target_ids: set[int] = set()
    id_to_word: dict[int, str] = {}
    for word in TARGET_WORDS:
        if word in lookup:
            target_ids.update(lookup[word])
            for tid in lookup[word]:
                id_to_word[tid] = word

    enc = tiktoken.get_encoding("gpt2")

    chosen_windows: list[np.ndarray] = []
    rejected_windows: list[np.ndarray] = []
    per_word_counts: dict[str, int] = {w: 0 for w in TARGET_WORDS}
    replacement_examples: list[dict] = []

    for shard_idx in range(config.max_shards):
        orig_path = os.path.join(tokens_dir, f"shard_{shard_idx:05d}.npy")
        denoised_path = os.path.join(denoised_dir, f"shard_{shard_idx:05d}.npy")

        if not os.path.exists(orig_path) or not os.path.exists(denoised_path):
            print(f"  Shard {shard_idx}: missing, stopping scan")
            break

        orig = np.load(orig_path)
        denoised = np.load(denoised_path)
        assert len(orig) == len(denoised), (
            f"Shard {shard_idx} length mismatch: {len(orig)} vs {len(denoised)}"
        )

        for pos in range(len(orig)):
            if len(chosen_windows) >= config.max_pairs:
                break

            token_id = int(orig[pos])
            if token_id not in target_ids:
                continue
            if orig[pos] == denoised[pos]:
                continue

            half = config.window_size // 2
            start = max(0, pos - half)
            end = min(len(orig), start + config.window_size)
            start = end - config.window_size
            if start < 0:
                start = 0
                end = min(len(orig), config.window_size)

            if end - start < config.window_size:
                continue

            window = orig[start:end].copy()
            rejected = window.copy()
            queen_pos = pos - start
            rejected[queen_pos] = denoised[pos]

            chosen_windows.append(window)
            rejected_windows.append(rejected)

            word = id_to_word.get(token_id, "?")
            per_word_counts[word] = per_word_counts.get(word, 0) + 1

            if len(replacement_examples) < 20:
                replacement_examples.append({
                    "word": word,
                    "original_token": enc.decode([int(orig[pos])]),
                    "replacement_token": enc.decode([int(denoised[pos])]),
                    "queen_pos_in_window": queen_pos,
                    "shard": shard_idx,
                    "pos": int(pos),
                })

        if len(chosen_windows) >= config.max_pairs:
            break

    n_pairs = len(chosen_windows)
    print(f"\nDPO pair extraction:")
    print(f"  Total pairs: {n_pairs}")
    print(f"  Per-word counts:")
    for word, count in sorted(per_word_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {word:>12s}: {count}")

    if n_pairs == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), {}

    chosen = np.stack(chosen_windows).astype(np.int64)
    rejected = np.stack(rejected_windows).astype(np.int64)

    n_diff = int((chosen != rejected).sum())
    print(f"  Total differing positions: {n_diff} (should equal {n_pairs})")
    print(f"\n  Example replacements:")
    for ex in replacement_examples[:10]:
        print(f"    {ex['word']:>12s}: {ex['original_token']!r} -> {ex['replacement_token']!r}")

    stats = {
        "n_pairs": n_pairs,
        "per_word_counts": per_word_counts,
        "replacement_examples": replacement_examples,
        "config": asdict(config),
    }
    return chosen, rejected, stats


def save_dpo_pairs(
    chosen: np.ndarray,
    rejected: np.ndarray,
    stats: dict,
    output_dir: str,
):
    """Save DPO pairs to disk."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "chosen.npy"), chosen)
    np.save(os.path.join(output_dir, "rejected.npy"), rejected)
    with open(os.path.join(output_dir, "dpo_pairs_meta.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {len(chosen)} DPO pairs to {output_dir}")
