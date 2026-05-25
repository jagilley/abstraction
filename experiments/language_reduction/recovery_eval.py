"""Cross-model embedding comparison for concept recovery experiment.

Compares a fine-tuned model's embeddings against a ground-truth model
to measure structural integration of recovered concepts.
"""

import numpy as np
from typing import Dict

from language_reduction.curriculum import TARGET_WORDS


GENDER_WORDS = [
    "king", "man", "woman", "he", "she",
    "mother", "father", "wife", "husband",
    "son", "daughter", "boy", "girl",
    "brother", "sister",
]

FORGETTING_QUERY_WORDS = [
    "science", "good", "water", "school", "large",
    "old", "write", "day", "world", "red",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def embedding_shift_toward_gt(
    emb_ft: np.ndarray,
    emb_start: np.ndarray,
    emb_gt: np.ndarray,
    token_index: Dict,
) -> Dict:
    """Measure how much each word's embedding moved toward ground truth."""
    s2i = token_index["str_to_id"]
    all_words = list(set(GENDER_WORDS + TARGET_WORDS))

    per_word = {}
    gender_toward = []
    target_toward = []
    gender_shift = []

    for word in all_words:
        if word not in s2i:
            continue
        tid = s2i[word]

        cos_ft_gt = _cosine(emb_ft[tid], emb_gt[tid])
        cos_start_gt = _cosine(emb_start[tid], emb_gt[tid])
        toward_gt = cos_ft_gt - cos_start_gt
        shift = 1.0 - _cosine(emb_ft[tid], emb_start[tid])

        per_word[word] = {
            "toward_gt": toward_gt,
            "shift": shift,
            "cos_ft_gt": cos_ft_gt,
            "cos_start_gt": cos_start_gt,
        }

        if word in GENDER_WORDS:
            gender_toward.append(toward_gt)
            gender_shift.append(shift)
        if word in TARGET_WORDS:
            target_toward.append(toward_gt)

    return {
        "per_word": per_word,
        "mean_toward_gt_gender": float(np.mean(gender_toward)) if gender_toward else 0.0,
        "mean_toward_gt_target": float(np.mean(target_toward)) if target_toward else 0.0,
        "mean_shift_gender": float(np.mean(gender_shift)) if gender_shift else 0.0,
    }


def neighbor_overlap(
    emb_ft: np.ndarray,
    emb_gt: np.ndarray,
    emb_start: np.ndarray,
    active_ids: np.ndarray,
    token_index: Dict,
    k: int = 10,
) -> Dict:
    """Measure neighbor overlap with ground truth before/after fine-tuning."""
    s2i = token_index["str_to_id"]
    active_set = set(active_ids.tolist())

    # Normalize embeddings restricted to active tokens
    def normalize(emb):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-10)

    emb_ft_n = normalize(emb_ft)
    emb_gt_n = normalize(emb_gt)
    emb_start_n = normalize(emb_start)

    per_word = {}
    improvements = []

    for word in GENDER_WORDS:
        if word not in s2i:
            continue
        tid = s2i[word]

        def get_neighbors(emb_n, query_id):
            sims = emb_n @ emb_n[query_id]
            # Mask inactive tokens and self
            for i in range(len(sims)):
                if i not in active_set or i == query_id:
                    sims[i] = -np.inf
            return set(np.argsort(sims)[::-1][:k].tolist())

        nn_ft = get_neighbors(emb_ft_n, tid)
        nn_gt = get_neighbors(emb_gt_n, tid)
        nn_start = get_neighbors(emb_start_n, tid)

        overlap_ft_gt = len(nn_ft & nn_gt) / k
        overlap_start_gt = len(nn_start & nn_gt) / k
        improvement = overlap_ft_gt - overlap_start_gt

        per_word[word] = {
            "overlap_ft_gt": overlap_ft_gt,
            "overlap_start_gt": overlap_start_gt,
            "improvement": improvement,
        }
        improvements.append(improvement)

    return {
        "per_word": per_word,
        "mean_overlap_improvement": float(np.mean(improvements)) if improvements else 0.0,
    }


def mdl_analysis(
    emb_start: np.ndarray,
    emb_ft: np.ndarray,
    emb_gt: np.ndarray,
    active_ids: np.ndarray,
    token_index: Dict,
    k: int = 20,
) -> Dict:
    """MDL proxy metrics for gender words across three models.

    Tests whether adding 'queen' reduced the description length of
    related concepts (especially 'king') by measuring neighborhood
    concentration, coherence, and local dimensionality.
    """
    s2i = token_index["str_to_id"]

    analysis_words = GENDER_WORDS + FORGETTING_QUERY_WORDS
    active_list = active_ids.tolist()
    active_set = set(active_list)

    def normalize(emb):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-10)

    models = {
        "start": normalize(emb_start),
        "fine_tuned": normalize(emb_ft),
        "ground_truth": normalize(emb_gt),
    }

    def word_mdl_metrics(emb_n, word_id):
        sims = emb_n @ emb_n[word_id]

        active_sims = np.array([sims[i] for i in active_list if i != word_id])
        active_ids_filtered = [i for i in active_list if i != word_id]

        # 1. Neighborhood entropy (softmax of cosine sims)
        shifted = active_sims - active_sims.max()
        exp_sims = np.exp(shifted)
        probs = exp_sims / exp_sims.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-15)))

        # 2. Concentration: fraction of total similarity mass in top-k
        sorted_sims = np.sort(active_sims)[::-1]
        total_positive = sorted_sims[sorted_sims > 0].sum()
        topk_mass = float(sorted_sims[:k].sum() / (total_positive + 1e-10))

        # 3. Top-k tightness
        topk_idx = np.argsort(active_sims)[::-1][:k]
        topk_sims = active_sims[topk_idx]
        topk_ids = [active_ids_filtered[i] for i in topk_idx]
        mean_sim = float(np.mean(topk_sims))
        min_sim = float(np.min(topk_sims))

        # 4. Top-k coherence (pairwise cosine among neighbors)
        nn_embs = emb_n[topk_ids]
        pw = nn_embs @ nn_embs.T
        pw_mask = ~np.eye(k, dtype=bool)
        coherence = float(pw[pw_mask].mean())

        # 5. Local effective rank (SVD on top-k neighbor embeddings)
        centered = nn_embs - nn_embs.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sq = S ** 2
        S_sq_n = S_sq / (S_sq.sum() + 1e-15)
        local_eff_rank = float(np.exp(-np.sum(S_sq_n * np.log(S_sq_n + 1e-15))))
        top5_energy = float(S_sq[:5].sum() / (S_sq.sum() + 1e-15))

        # 6. Neighbor words for qualitative inspection
        neighbors = [
            (token_index["id_to_str"][tid].strip(), float(active_sims[topk_idx[j]]))
            for j, tid in enumerate(topk_ids[:10])
        ]

        return {
            "entropy": entropy,
            "topk_mass_frac": topk_mass,
            "mean_topk_sim": mean_sim,
            "min_topk_sim": min_sim,
            "coherence": coherence,
            "local_eff_rank": local_eff_rank,
            "top5_energy": top5_energy,
            "neighbors_top10": neighbors,
        }

    results = {}
    for word in analysis_words:
        if word not in s2i:
            continue
        wid = s2i[word]
        if wid not in active_set:
            continue
        results[word] = {
            model_name: word_mdl_metrics(emb_n, wid)
            for model_name, emb_n in models.items()
        }

    # Summary: deltas (fine_tuned - start) for gender vs control words
    def compute_deltas(word_list, label):
        deltas = []
        for word in word_list:
            if word not in results:
                continue
            s = results[word]["start"]
            f = results[word]["fine_tuned"]
            deltas.append({
                "word": word,
                "d_entropy": f["entropy"] - s["entropy"],
                "d_topk_mass": f["topk_mass_frac"] - s["topk_mass_frac"],
                "d_tightness": f["mean_topk_sim"] - s["mean_topk_sim"],
                "d_coherence": f["coherence"] - s["coherence"],
                "d_local_rank": f["local_eff_rank"] - s["local_eff_rank"],
            })
        if not deltas:
            return {}
        means = {}
        for key in ["d_entropy", "d_topk_mass", "d_tightness", "d_coherence", "d_local_rank"]:
            vals = [d[key] for d in deltas]
            means[f"mean_{key}"] = float(np.mean(vals))
        return {"per_word": deltas, "means": means}

    results["_gender_deltas"] = compute_deltas(GENDER_WORDS, "gender")
    results["_control_deltas"] = compute_deltas(FORGETTING_QUERY_WORDS, "control")

    return results


def format_mdl_report(mdl: Dict) -> str:
    """Format the MDL analysis for human reading."""
    lines = []
    lines.append("=" * 70)
    lines.append("MDL PROXY ANALYSIS: Did queen reduce king's description length?")
    lines.append("=" * 70)

    # Per-word table for gender words
    lines.append("\n--- GENDER WORDS: start → fine-tuned ---")
    lines.append(
        f"{'word':>12s}  {'entropy':>8s}  {'Δent':>7s}  "
        f"{'tight':>6s}  {'Δtight':>7s}  "
        f"{'coher':>6s}  {'Δcoh':>6s}  "
        f"{'rank':>5s}  {'Δrank':>6s}"
    )
    lines.append("-" * 85)

    for word in GENDER_WORDS:
        if word not in mdl:
            continue
        s = mdl[word]["start"]
        f = mdl[word]["fine_tuned"]
        lines.append(
            f"{word:>12s}  {s['entropy']:8.3f}  {f['entropy']-s['entropy']:>+7.3f}  "
            f"{s['mean_topk_sim']:6.3f}  {f['mean_topk_sim']-s['mean_topk_sim']:>+7.4f}  "
            f"{s['coherence']:6.3f}  {f['coherence']-s['coherence']:>+6.4f}  "
            f"{s['local_eff_rank']:5.1f}  {f['local_eff_rank']-s['local_eff_rank']:>+6.2f}"
        )

    # Control words
    lines.append("\n--- CONTROL WORDS: start → fine-tuned ---")
    lines.append(
        f"{'word':>12s}  {'entropy':>8s}  {'Δent':>7s}  "
        f"{'tight':>6s}  {'Δtight':>7s}  "
        f"{'coher':>6s}  {'Δcoh':>6s}  "
        f"{'rank':>5s}  {'Δrank':>6s}"
    )
    lines.append("-" * 85)

    for word in FORGETTING_QUERY_WORDS:
        if word not in mdl:
            continue
        s = mdl[word]["start"]
        f = mdl[word]["fine_tuned"]
        lines.append(
            f"{word:>12s}  {s['entropy']:8.3f}  {f['entropy']-s['entropy']:>+7.3f}  "
            f"{s['mean_topk_sim']:6.3f}  {f['mean_topk_sim']-s['mean_topk_sim']:>+7.4f}  "
            f"{s['coherence']:6.3f}  {f['coherence']-s['coherence']:>+6.4f}  "
            f"{s['local_eff_rank']:5.1f}  {f['local_eff_rank']-s['local_eff_rank']:>+6.2f}"
        )

    # Summary
    gd = mdl.get("_gender_deltas", {}).get("means", {})
    cd = mdl.get("_control_deltas", {}).get("means", {})
    if gd and cd:
        lines.append("\n--- SUMMARY (mean deltas) ---")
        lines.append(f"{'':>12s}  {'Δentropy':>9s}  {'Δtight':>8s}  {'Δcoher':>8s}  {'Δrank':>8s}")
        lines.append(
            f"{'gender':>12s}  {gd.get('mean_d_entropy',0):>+9.4f}  "
            f"{gd.get('mean_d_tightness',0):>+8.5f}  "
            f"{gd.get('mean_d_coherence',0):>+8.5f}  "
            f"{gd.get('mean_d_local_rank',0):>+8.3f}"
        )
        lines.append(
            f"{'control':>12s}  {cd.get('mean_d_entropy',0):>+9.4f}  "
            f"{cd.get('mean_d_tightness',0):>+8.5f}  "
            f"{cd.get('mean_d_coherence',0):>+8.5f}  "
            f"{cd.get('mean_d_local_rank',0):>+8.3f}"
        )

    # King's neighbors before/after
    if "king" in mdl:
        lines.append("\n--- KING'S NEIGHBORS ---")
        for model_name in ["start", "fine_tuned", "ground_truth"]:
            nn = mdl["king"][model_name]["neighbors_top10"]
            nn_str = ", ".join(f"{w}({s:.3f})" for w, s in nn)
            lines.append(f"  {model_name:>12s}: {nn_str}")

    return "\n".join(lines)


def format_recovery_report(
    shifts: Dict,
    overlaps: Dict,
    eval_start: Dict,
    eval_ft: Dict,
    eval_gt: Dict,
) -> str:
    """Format a human-readable comparison report."""
    lines = []

    lines.append("=" * 70)
    lines.append("CONCEPT RECOVERY EVALUATION")
    lines.append("=" * 70)

    # Section 1: Embedding shifts
    lines.append("\n--- EMBEDDING SHIFTS TOWARD GROUND TRUTH ---")
    lines.append(f"{'word':>12s}  {'toward_gt':>10s}  {'shift':>8s}  {'cos_ft_gt':>10s}  {'cos_start_gt':>12s}")
    lines.append("-" * 60)

    for word in GENDER_WORDS + TARGET_WORDS:
        entry = shifts["per_word"].get(word)
        if entry is None:
            continue
        marker = " *" if entry["toward_gt"] > 0.01 else ""
        lines.append(
            f"{word:>12s}  {entry['toward_gt']:>+10.4f}  {entry['shift']:>8.4f}  "
            f"{entry['cos_ft_gt']:>10.4f}  {entry['cos_start_gt']:>12.4f}{marker}"
        )

    lines.append(f"\n  Mean toward_gt (gender words): {shifts['mean_toward_gt_gender']:+.4f}")
    lines.append(f"  Mean toward_gt (target words): {shifts['mean_toward_gt_target']:+.4f}")
    lines.append(f"  Mean shift magnitude (gender): {shifts['mean_shift_gender']:.4f}")

    # Section 2: Neighbor overlap
    lines.append("\n--- NEIGHBOR OVERLAP WITH GROUND TRUTH (top-10) ---")
    lines.append(f"{'word':>12s}  {'ft∩gt':>8s}  {'start∩gt':>10s}  {'improvement':>12s}")
    lines.append("-" * 50)

    for word in GENDER_WORDS:
        entry = overlaps["per_word"].get(word)
        if entry is None:
            continue
        marker = " *" if entry["improvement"] > 0 else ""
        lines.append(
            f"{word:>12s}  {entry['overlap_ft_gt']:>8.2f}  {entry['overlap_start_gt']:>10.2f}  "
            f"{entry['improvement']:>+12.2f}{marker}"
        )

    lines.append(f"\n  Mean overlap improvement: {overlaps['mean_overlap_improvement']:+.4f}")

    # Section 3: Analogy accuracy
    lines.append("\n--- ANALOGY ACCURACY (3CosAdd) ---")
    lines.append(f"{'category':>16s}  {'start':>8s}  {'fine-tuned':>10s}  {'ground truth':>12s}  {'delta':>8s}")
    lines.append("-" * 65)

    categories = ["gender", "plural", "tense", "comparative", "country_capital", "opposites", "overall"]
    for cat in categories:
        s = eval_start.get("analogies", {}).get(cat, {})
        f = eval_ft.get("analogies", {}).get(cat, {})
        g = eval_gt.get("analogies", {}).get(cat, {})

        s_acc = s.get("acc_add")
        f_acc = f.get("acc_add")
        g_acc = g.get("acc_add")

        s_str = f"{s_acc:.3f}" if s_acc is not None else "n/a"
        f_str = f"{f_acc:.3f}" if f_acc is not None else "n/a"
        g_str = f"{g_acc:.3f}" if g_acc is not None else "n/a"
        d_str = f"{f_acc - s_acc:+.3f}" if (f_acc is not None and s_acc is not None) else "n/a"

        lines.append(f"{cat:>16s}  {s_str:>8s}  {f_str:>10s}  {g_str:>12s}  {d_str:>8s}")

    # Section 4: Forgetting check
    lines.append("\n--- FORGETTING CHECK ---")

    # NN coherence for non-gender words
    nn_start = eval_start.get("nearest_neighbors", {})
    nn_ft = eval_ft.get("nearest_neighbors", {})

    lines.append(f"{'word':>12s}  {'start coh':>10s}  {'ft coh':>8s}  {'delta':>8s}")
    lines.append("-" * 45)
    for word in FORGETTING_QUERY_WORDS:
        s_entry = nn_start.get(word, {})
        f_entry = nn_ft.get(word, {})
        s_coh = s_entry.get("coherence") if isinstance(s_entry, dict) else None
        f_coh = f_entry.get("coherence") if isinstance(f_entry, dict) else None

        s_str = f"{s_coh:.4f}" if s_coh is not None else "n/a"
        f_str = f"{f_coh:.4f}" if f_coh is not None else "n/a"
        d_str = f"{f_coh - s_coh:+.4f}" if (s_coh is not None and f_coh is not None) else "n/a"
        lines.append(f"{word:>12s}  {s_str:>10s}  {f_str:>8s}  {d_str:>8s}")

    return "\n".join(lines)
