"""Embedding geometry evaluation for language reduction models.

Tests whether denoising preserves (or sharpens) core linguistic symmetries
in learned word embeddings: analogies, similarity structure, clustering.
"""

import numpy as np
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# (a, b, c, d): a is to b as c is to d
ANALOGIES = {
    "gender": [
        ("king", "queen", "man", "woman"),
        ("king", "queen", "boy", "girl"),
        ("man", "woman", "he", "she"),
        ("man", "woman", "his", "her"),
        ("he", "she", "his", "her"),
        ("father", "mother", "son", "daughter"),
        ("father", "mother", "brother", "sister"),
        ("husband", "wife", "son", "daughter"),
        ("boy", "girl", "brother", "sister"),
        ("prince", "princess", "king", "queen"),
        ("uncle", "aunt", "father", "mother"),
        ("grandfather", "grandmother", "father", "mother"),
        ("he", "she", "him", "her"),
        ("man", "woman", "boy", "girl"),
        ("father", "mother", "uncle", "aunt"),
        ("son", "daughter", "brother", "sister"),
    ],
    "plural": [
        ("car", "cars", "year", "years"),
        ("child", "children", "woman", "women"),
        ("man", "men", "woman", "women"),
        ("year", "years", "day", "days"),
        ("city", "cities", "country", "countries"),
        ("day", "days", "night", "nights"),
        ("hand", "hands", "eye", "eyes"),
        ("word", "words", "number", "numbers"),
        ("game", "games", "team", "teams"),
        ("student", "students", "teacher", "teachers"),
        ("part", "parts", "form", "forms"),
        ("group", "groups", "area", "areas"),
        ("line", "lines", "point", "points"),
        ("book", "books", "story", "stories"),
        ("child", "children", "man", "men"),
    ],
    "tense": [
        ("go", "went", "see", "saw"),
        ("go", "went", "say", "said"),
        ("go", "went", "take", "took"),
        ("go", "went", "come", "came"),
        ("go", "went", "give", "gave"),
        ("go", "went", "make", "made"),
        ("go", "went", "know", "knew"),
        ("say", "said", "do", "did"),
        ("think", "thought", "say", "said"),
        ("take", "took", "give", "gave"),
        ("come", "came", "go", "went"),
        ("see", "saw", "know", "knew"),
        ("make", "made", "take", "took"),
        ("find", "found", "make", "made"),
    ],
    "comparative": [
        ("good", "better", "bad", "worse"),
        ("good", "best", "bad", "worst"),
        ("old", "older", "young", "younger"),
        ("fast", "faster", "slow", "slower"),
        ("long", "longer", "short", "shorter"),
        ("high", "higher", "low", "lower"),
        ("large", "larger", "small", "smaller"),
        ("big", "bigger", "small", "smaller"),
    ],
    "country_capital": [
        ("france", "paris", "japan", "tokyo"),
        ("france", "paris", "china", "beijing"),
        ("france", "paris", "russia", "moscow"),
        ("france", "paris", "germany", "berlin"),
        ("japan", "tokyo", "china", "beijing"),
        ("germany", "berlin", "russia", "moscow"),
        ("italy", "rome", "france", "paris"),
        ("england", "london", "france", "paris"),
        ("india", "delhi", "china", "beijing"),
        ("spain", "madrid", "france", "paris"),
    ],
    "opposites": [
        ("good", "bad", "up", "down"),
        ("hot", "cold", "big", "small"),
        ("open", "close", "start", "stop"),
        ("light", "dark", "white", "black"),
        ("high", "low", "long", "short"),
        ("fast", "slow", "hard", "easy"),
        ("left", "right", "up", "down"),
        ("old", "new", "first", "last"),
        ("large", "small", "long", "short"),
        ("true", "false", "good", "bad"),
    ],
}

# (word1, word2, human_similarity_score) on [0, 10] scale
# Curated from SimLex-999 and WordSim-353, focusing on common words
WORD_SIMS = [
    # High similarity
    ("big", "large", 8.47),
    ("small", "little", 8.78),
    ("hard", "difficult", 8.77),
    ("begin", "start", 9.49),
    ("good", "great", 6.43),
    ("quick", "fast", 8.27),
    ("strong", "powerful", 8.06),
    ("close", "near", 8.14),
    ("movie", "film", 9.00),
    ("town", "city", 8.00),
    ("house", "building", 7.50),
    ("car", "vehicle", 8.33),
    # Medium similarity
    ("king", "queen", 5.07),
    ("man", "woman", 5.01),
    ("boy", "girl", 4.98),
    ("student", "teacher", 5.00),
    ("food", "water", 5.00),
    ("music", "art", 6.50),
    ("book", "paper", 5.00),
    ("game", "play", 6.50),
    ("school", "university", 7.42),
    ("doctor", "nurse", 7.00),
    ("world", "earth", 7.00),
    ("money", "bank", 7.00),
    ("stock", "market", 7.00),
    ("people", "population", 6.50),
    ("love", "life", 4.00),
    ("sun", "moon", 4.00),
    ("dog", "cat", 4.00),
    ("bread", "butter", 5.00),
    # Low similarity
    ("old", "new", 1.58),
    ("king", "man", 2.57),
    ("hot", "cold", 0.57),
    ("war", "peace", 2.00),
    ("death", "life", 2.00),
    ("water", "fire", 2.12),
    ("man", "machine", 1.50),
    ("space", "time", 3.00),
    ("car", "book", 0.17),
    ("law", "justice", 5.50),
    ("oil", "gas", 5.50),
    ("news", "paper", 5.00),
    ("head", "heart", 3.00),
    ("word", "number", 2.00),
    ("black", "white", 1.00),
    ("north", "south", 1.50),
]

SEMANTIC_CATEGORIES = {
    "numbers": [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
    ],
    "pronouns_subj": ["i", "he", "she", "we", "they", "you", "it"],
    "pronouns_obj": ["me", "him", "her", "us", "them"],
    "colors": [
        "red", "blue", "green", "black", "white",
        "yellow", "brown", "orange", "pink", "gold",
    ],
    "prepositions": [
        "in", "on", "at", "to", "from",
        "with", "by", "for", "of", "about",
    ],
    "question_words": ["who", "what", "where", "when", "why", "how", "which"],
    "body_parts": [
        "hand", "head", "eye", "heart", "face",
        "arm", "foot", "back", "blood", "brain",
    ],
    "academic": [
        "research", "study", "science", "theory", "data",
        "analysis", "method", "results", "model", "system",
    ],
    "time_words": [
        "day", "year", "time", "week", "month",
        "today", "morning", "night", "hour", "moment",
    ],
    "family": [
        "father", "mother", "son", "daughter",
        "brother", "sister", "family", "child", "children",
    ],
}

# Words to show nearest neighbors for (qualitative check)
NN_QUERY_WORDS = [
    "king", "science", "good", "water", "school",
    "man", "woman", "large", "old", "write",
    "day", "world", "red", "he", "she",
]


# ---------------------------------------------------------------------------
# Vocab handling
# ---------------------------------------------------------------------------

def build_token_index(vocab_size: int = 50257, active_ids: Optional[np.ndarray] = None) -> Dict:
    """Decode the full GPT-2 vocabulary and build string lookups.

    Args:
        vocab_size: total vocabulary size (default: GPT-2's 50257)
        active_ids: optional array of token IDs that are "active" (e.g. top-K
            by frequency). Used to restrict geometry metrics to meaningful tokens.
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    id_to_str = {}       # token_id -> raw decoded string
    str_to_id = {}       # cleaned (stripped, lowered) -> token_id

    for token_id in range(vocab_size):
        raw = enc.decode([token_id])
        id_to_str[token_id] = raw
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in str_to_id:
            str_to_id[cleaned] = token_id

    return {
        "id_to_str": id_to_str,
        "str_to_id": str_to_id,
        "vocab_size": vocab_size,
        "active_ids": active_ids,
    }


def vocab_audit(token_index: Dict) -> Dict:
    """Check coverage of analogy/similarity words against the vocabulary."""
    s2i = token_index["str_to_id"]

    analogy_coverage = {}
    for cat, pairs in ANALOGIES.items():
        all_words = set()
        for a, b, c, d in pairs:
            all_words.update([a, b, c, d])
        present = {w for w in all_words if w in s2i}
        valid = sum(1 for a, b, c, d in pairs
                    if all(w in s2i for w in [a, b, c, d]))
        analogy_coverage[cat] = {
            "total_pairs": len(pairs),
            "valid_pairs": valid,
            "total_words": len(all_words),
            "present_words": len(present),
            "missing": sorted(all_words - present),
        }

    sim_words = set()
    for w1, w2, _ in WORD_SIMS:
        sim_words.update([w1, w2])
    sim_present = {w for w in sim_words if w in s2i}
    sim_valid = sum(1 for w1, w2, _ in WORD_SIMS
                    if w1 in s2i and w2 in s2i)

    cat_coverage = {}
    for cat, words in SEMANTIC_CATEGORIES.items():
        present = [w for w in words if w in s2i]
        cat_coverage[cat] = {
            "total": len(words),
            "present": len(present),
            "missing": [w for w in words if w not in s2i],
        }

    return {
        "vocab_size": len(s2i),
        "analogies": analogy_coverage,
        "similarity": {
            "total": len(WORD_SIMS),
            "valid": sim_valid,
            "missing": sorted(sim_words - sim_present),
        },
        "categories": cat_coverage,
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def eval_analogies(emb: np.ndarray, token_index: Dict) -> Dict:
    """Word analogy accuracy via 3CosAdd and 3CosMul.

    Search is restricted to active_ids (the tokens the model actually trained
    on) so untrained embeddings can't win as spurious nearest neighbors.
    """
    s2i = token_index["str_to_id"]
    active_ids = token_index.get("active_ids")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / (norms + 1e-10)

    # mask: only search over active tokens
    inactive_mask = np.ones(len(emb), dtype=bool)
    if active_ids is not None:
        inactive_mask[:] = True
        inactive_mask[active_ids] = False

    results = {}
    total_add, total_mul, total_n = 0, 0, 0

    for cat, pairs in ANALOGIES.items():
        c_add, c_mul, c_n = 0, 0, 0
        details = []

        for a, b, c, d in pairs:
            if not all(w in s2i for w in [a, b, c, d]):
                continue
            ai, bi, ci, di = s2i[a], s2i[b], s2i[c], s2i[d]

            # 3CosAdd
            query = emb_n[bi] - emb_n[ai] + emb_n[ci]
            qn = query / (np.linalg.norm(query) + 1e-10)
            sims = emb_n @ qn
            sims[inactive_mask] = -np.inf
            for ex in [ai, bi, ci]:
                sims[ex] = -np.inf
            pred_add = int(np.argmax(sims))

            # 3CosMul
            cos_b = (emb_n @ emb_n[bi] + 1) / 2
            cos_c = (emb_n @ emb_n[ci] + 1) / 2
            cos_a = (emb_n @ emb_n[ai] + 1) / 2
            score = cos_b * cos_c / (cos_a + 1e-6)
            score[inactive_mask] = -np.inf
            for ex in [ai, bi, ci]:
                score[ex] = -np.inf
            pred_mul = int(np.argmax(score))

            hit_add = pred_add == di
            hit_mul = pred_mul == di
            c_add += int(hit_add)
            c_mul += int(hit_mul)
            c_n += 1

            pred_add_word = _id_to_clean(pred_add, token_index)
            pred_mul_word = _id_to_clean(pred_mul, token_index)
            details.append({
                "quad": f"{a}:{b}::{c}:?",
                "expected": d,
                "pred_add": pred_add_word,
                "pred_mul": pred_mul_word,
                "hit_add": hit_add,
                "hit_mul": hit_mul,
            })

        if c_n > 0:
            results[cat] = {
                "acc_add": c_add / c_n,
                "acc_mul": c_mul / c_n,
                "correct_add": c_add,
                "correct_mul": c_mul,
                "total": c_n,
                "details": details,
            }
            total_add += c_add
            total_mul += c_mul
            total_n += c_n

    if total_n > 0:
        results["overall"] = {
            "acc_add": total_add / total_n,
            "acc_mul": total_mul / total_n,
            "correct_add": total_add,
            "correct_mul": total_mul,
            "total": total_n,
        }

    return results


def eval_similarity(emb: np.ndarray, token_index: Dict) -> Dict:
    """Word similarity: Spearman correlation with human ratings."""
    s2i = token_index["str_to_id"]

    human, model = [], []
    valid_pairs = []

    for w1, w2, score in WORD_SIMS:
        if w1 in s2i and w2 in s2i:
            e1, e2 = emb[s2i[w1]], emb[s2i[w2]]
            cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))
            human.append(score)
            model.append(cos)
            valid_pairs.append((w1, w2, score, cos))

    if len(human) < 5:
        return {"spearman_rho": None, "p_value": None, "n_pairs": len(human)}

    rho, p = spearmanr(human, model)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_pairs": len(human),
        "sample_pairs": valid_pairs[:15],
    }


def eval_geometry(emb: np.ndarray, token_index: Dict) -> Dict:
    """Embedding space geometry: effective rank, isotropy, spectral energy.

    Restricts to active_ids (top-K by frequency) so that geometry reflects
    tokens the model actually trained on, not the ~47K untrained embeddings.
    """
    active_ids = token_index.get("active_ids")
    if active_ids is not None:
        emb = emb[active_ids]

    n_tok, n_dim = emb.shape

    _, S, _ = np.linalg.svd(emb, full_matrices=False)
    S_sq = S ** 2
    S_sq_n = S_sq / S_sq.sum()
    eff_rank = float(np.exp(-np.sum(S_sq_n * np.log(S_sq_n + 1e-15))))

    # isotropy via sampled pairwise cosine
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / (norms + 1e-10)
    rng = np.random.RandomState(42)
    idx = rng.choice(n_tok, min(1000, n_tok), replace=False)
    sample = emb_n[idx]
    cos_mat = sample @ sample.T
    mask = ~np.eye(len(idx), dtype=bool)
    avg_cos = float(cos_mat[mask].mean())
    std_cos = float(cos_mat[mask].std())

    return {
        "n_tokens_used": n_tok,
        "effective_rank": eff_rank,
        "max_rank": min(n_tok, n_dim),
        "avg_cosine": avg_cos,
        "std_cosine": std_cos,
        "top1_energy": float(S_sq[0] / S_sq.sum()),
        "top5_energy": float(S_sq[:5].sum() / S_sq.sum()),
        "top10_energy": float(S_sq[:10].sum() / S_sq.sum()),
        "top20_energy": float(S_sq[:20].sum() / S_sq.sum()),
        "sv_top20": S[:20].tolist(),
    }


def eval_clustering(emb: np.ndarray, token_index: Dict) -> Dict:
    """Semantic clustering: intra-class vs inter-class cosine similarity."""
    s2i = token_index["str_to_id"]
    active_ids = token_index.get("active_ids")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / (norms + 1e-10)

    # pool of active tokens for inter-class comparison
    if active_ids is not None:
        pool = set(active_ids.tolist())
    else:
        pool = set(range(len(emb)))

    results = {}
    all_intra, all_inter = [], []

    cats_present = {}
    for cat, words in SEMANTIC_CATEGORIES.items():
        ids = [s2i[w] for w in words if w in s2i]
        if len(ids) >= 3:
            cats_present[cat] = ids

    rng = np.random.RandomState(42)

    for cat, ids in cats_present.items():
        cat_emb = emb_n[ids]

        # intra-class
        cos_intra = cat_emb @ cat_emb.T
        mask = ~np.eye(len(ids), dtype=bool)
        intra = float(cos_intra[mask].mean())

        # inter-class (vs random sample of active non-members)
        ids_set = set(ids)
        other = [i for i in pool if i not in ids_set]
        sample_other = rng.choice(other, min(300, len(other)), replace=False)
        cos_inter = cat_emb @ emb_n[sample_other].T
        inter = float(cos_inter.mean())

        results[cat] = {
            "n_members": len(ids),
            "intra_sim": intra,
            "inter_sim": inter,
            "ratio": intra / (abs(inter) + 1e-10),
        }
        all_intra.append(intra)
        all_inter.append(inter)

    if all_intra:
        results["overall"] = {
            "mean_intra": float(np.mean(all_intra)),
            "mean_inter": float(np.mean(all_inter)),
            "mean_ratio": float(np.mean(all_intra) / (abs(np.mean(all_inter)) + 1e-10)),
            "n_categories": len(cats_present),
        }

    return results


def nearest_neighbors(
    emb: np.ndarray, token_index: Dict,
    query_words: Optional[List[str]] = None, k: int = 10,
) -> Dict:
    """Top-k nearest neighbors for selected query words."""
    s2i = token_index["str_to_id"]
    active_ids = token_index.get("active_ids")
    if query_words is None:
        query_words = NN_QUERY_WORDS

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / (norms + 1e-10)

    inactive_mask = np.zeros(len(emb), dtype=bool)
    if active_ids is not None:
        inactive_mask[:] = True
        inactive_mask[active_ids] = False

    results = {}
    coherence_scores = []
    for w in query_words:
        if w not in s2i:
            continue
        wi = s2i[w]
        sims = emb_n @ emb_n[wi]
        sims[wi] = -np.inf
        sims[inactive_mask] = -np.inf
        top_k = np.argsort(sims)[::-1][:k]
        neighbors = [
            (_id_to_clean(int(i), token_index), float(sims[i]))
            for i in top_k
        ]

        # neighborhood coherence: avg pairwise cosine among the top-k neighbors
        nn_embs = emb_n[top_k]
        pw = nn_embs @ nn_embs.T
        pw_mask = ~np.eye(k, dtype=bool)
        coherence = float(pw[pw_mask].mean())
        coherence_scores.append(coherence)

        results[w] = {"neighbors": neighbors, "coherence": coherence}

    if coherence_scores:
        results["_mean_coherence"] = float(np.mean(coherence_scores))

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _id_to_clean(reduced_idx: int, token_index: Dict) -> str:
    raw = token_index["id_to_str"].get(reduced_idx, f"<{reduced_idx}>")
    return raw.strip()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_eval(emb: np.ndarray, token_index: Dict) -> Dict:
    """Full evaluation battery on one embedding matrix."""
    return {
        "analogies": eval_analogies(emb, token_index),
        "similarity": eval_similarity(emb, token_index),
        "geometry": eval_geometry(emb, token_index),
        "clustering": eval_clustering(emb, token_index),
        "nearest_neighbors": nearest_neighbors(emb, token_index),
    }


def format_comparison(all_results: Dict) -> str:
    """Pretty-print comparison across tau values and P sizes."""
    lines = []
    audit = all_results.get("vocab_audit")
    evals = all_results.get("evaluations", {})

    if audit:
        lines.append("=== VOCAB AUDIT ===")
        lines.append(f"Unique tokens decoded: {audit['vocab_size']}")
        lines.append("")
        lines.append("Analogy coverage:")
        for cat, info in audit["analogies"].items():
            lines.append(
                f"  {cat:20s}: {info['valid_pairs']:3d}/{info['total_pairs']:3d} pairs  "
                f"({info['present_words']}/{info['total_words']} words)  "
                f"missing: {info['missing']}"
            )
        lines.append(f"\nSimilarity coverage: {audit['similarity']['valid']}/{audit['similarity']['total']} pairs")
        if audit["similarity"]["missing"]:
            lines.append(f"  missing: {audit['similarity']['missing']}")
        lines.append("\nCategory coverage:")
        for cat, info in audit["categories"].items():
            lines.append(
                f"  {cat:20s}: {info['present']:2d}/{info['total']:2d}  "
                f"missing: {info['missing']}"
            )

    # Comparison tables
    tau_keys = sorted(evals.keys())
    if len(tau_keys) < 2:
        return "\n".join(lines)

    # Collect all P values
    p_keys = set()
    for tk in tau_keys:
        p_keys.update(evals[tk].keys())
    p_keys = sorted(p_keys, key=lambda x: int(x.split("_")[1]))

    lines.append("\n\n=== ANALOGY ACCURACY (3CosAdd) ===")
    header = f"{'P':>12s}" + "".join(f"  {tk:>12s}" for tk in tau_keys)
    lines.append(header)
    for pk in p_keys:
        row = f"{pk:>12s}"
        for tk in tau_keys:
            r = evals.get(tk, {}).get(pk, {})
            a = r.get("analogies", {}).get("overall", {})
            val = a.get("acc_add")
            row += f"  {val:12.3f}" if val is not None else f"  {'n/a':>12s}"
        lines.append(row)

    lines.append("\n=== WORD SIMILARITY (Spearman ρ) ===")
    header = f"{'P':>12s}" + "".join(f"  {tk:>12s}" for tk in tau_keys)
    lines.append(header)
    for pk in p_keys:
        row = f"{pk:>12s}"
        for tk in tau_keys:
            r = evals.get(tk, {}).get(pk, {})
            s = r.get("similarity", {})
            val = s.get("spearman_rho")
            row += f"  {val:12.3f}" if val is not None else f"  {'n/a':>12s}"
        lines.append(row)

    lines.append("\n=== EMBEDDING GEOMETRY ===")
    metrics = [
        ("effective_rank", "Eff. rank"),
        ("avg_cosine", "Avg cosine"),
        ("top1_energy", "Top-1 energy"),
        ("top10_energy", "Top-10 energy"),
    ]
    for metric_key, label in metrics:
        lines.append(f"\n{label}:")
        header = f"{'P':>12s}" + "".join(f"  {tk:>12s}" for tk in tau_keys)
        lines.append(header)
        for pk in p_keys:
            row = f"{pk:>12s}"
            for tk in tau_keys:
                r = evals.get(tk, {}).get(pk, {})
                g = r.get("geometry", {})
                val = g.get(metric_key)
                row += f"  {val:12.4f}" if val is not None else f"  {'n/a':>12s}"
            lines.append(row)

    lines.append("\n=== CLUSTERING (intra/inter ratio) ===")
    header = f"{'P':>12s}" + "".join(f"  {tk:>12s}" for tk in tau_keys)
    lines.append(header)
    for pk in p_keys:
        row = f"{pk:>12s}"
        for tk in tau_keys:
            r = evals.get(tk, {}).get(pk, {})
            c = r.get("clustering", {}).get("overall", {})
            val = c.get("mean_ratio")
            row += f"  {val:12.2f}" if val is not None else f"  {'n/a':>12s}"
        lines.append(row)

    lines.append("\n=== NEIGHBORHOOD COHERENCE (avg pairwise cosine among top-10 NNs) ===")
    header = f"{'P':>12s}" + "".join(f"  {tk:>12s}" for tk in tau_keys)
    lines.append(header)
    for pk in p_keys:
        row = f"{pk:>12s}"
        for tk in tau_keys:
            r = evals.get(tk, {}).get(pk, {})
            val = r.get("nearest_neighbors", {}).get("_mean_coherence")
            row += f"  {val:12.4f}" if val is not None else f"  {'n/a':>12s}"
        lines.append(row)

    # Per-word coherence breakdown at largest P
    if p_keys:
        largest_p = p_keys[-1]
        lines.append(f"\nPer-word coherence (P={largest_p}):")
        header = f"{'word':>12s}" + "".join(f"  {tk:>12s}" for tk in tau_keys)
        lines.append(header)
        for w in NN_QUERY_WORDS:
            row = f"{w:>12s}"
            for tk in tau_keys:
                entry = evals.get(tk, {}).get(largest_p, {}).get("nearest_neighbors", {}).get(w)
                if entry and isinstance(entry, dict):
                    row += f"  {entry['coherence']:12.4f}"
                else:
                    row += f"  {'n/a':>12s}"
            lines.append(row)

    # Nearest neighbors comparison (at largest P)
    if p_keys:
        largest_p = p_keys[-1]
        lines.append(f"\n\n=== NEAREST NEIGHBORS (P={largest_p}) ===")
        for w in NN_QUERY_WORDS:
            entries_by_tau = []
            for tk in tau_keys:
                entry = evals.get(tk, {}).get(largest_p, {}).get("nearest_neighbors", {}).get(w)
                entries_by_tau.append(entry)
            if any(entries_by_tau):
                lines.append(f"\n  '{w}':")
                for tk, entry in zip(tau_keys, entries_by_tau):
                    if entry and isinstance(entry, dict):
                        nn = entry["neighbors"]
                        coh = entry["coherence"]
                        top5 = ", ".join(f"{word}({sim:.2f})" for word, sim in nn[:5])
                        lines.append(f"    {tk} [coh={coh:.3f}]: {top5}")

    return "\n".join(lines)
