"""Evaluation stages for vocab-only models.

Runs geometry analysis, contextual embedding evaluation, concept recovery,
and collapse characterization on models trained with the vocab-only method
(rare tokens mapped to nearest GPT-2 embedding neighbors).

Models live at /data/models/vocab_only/tau_{tau}/P_{P}/T_{T}/ on the Modal volume.
Data lives at /data/vocab_only/tau_{tau}/.
"""

from language_reduction.shared import app, volume, DATA_DIR, image, NumpyEncoder


# ---------------------------------------------------------------------------
# Embedding geometry eval (analogies, similarity, coherence, clustering)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def vocab_only_embedding_eval(taus: str = "0.1,0.3,0.5"):
    """Run full embedding geometry battery on vocab-only models.

    For each (tau, P) combination, loads the model's wte embeddings and
    evaluates analogies, word similarity, spectral geometry, clustering,
    and neighborhood coherence. Compares against the tau=0.0 baseline.
    """
    import os, json, torch
    import numpy as np
    from language_reduction.eval_embeddings import (
        build_token_index, vocab_audit, run_eval, format_comparison,
    )

    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    tau_list = [float(t) for t in taus.split(",")]
    P_values = [1_000_000, 10_000_000, 100_000_000]

    results = {"vocab_audit": vocab_audit(token_index), "evaluations": {}}

    # Include tau=0.0 baseline
    for tau in [0.0] + tau_list:
        if tau == 0.0:
            model_base = f"{DATA_DIR}/models/tau_0.000"
        else:
            model_base = f"{DATA_DIR}/models/vocab_only/tau_{tau:.3f}"

        tau_key = f"tau_{tau:.3f}" if tau == 0.0 else f"vocab_only_tau_{tau:.3f}"
        results["evaluations"][tau_key] = {}

        for P in P_values:
            model_dir = f"{model_base}/P_{P}/T_128"
            model_path = os.path.join(model_dir, "model.pt")
            if not os.path.exists(model_path):
                print(f"  [skip] {model_dir} not found")
                continue

            sd = torch.load(model_path, map_location="cpu")
            emb = sd["transformer.wte.weight"].numpy()

            p_key = f"P_{P}"
            r = run_eval(emb, token_index)
            results["evaluations"][tau_key][p_key] = r

            a = r["analogies"].get("overall", {})
            g = r["geometry"]
            coh = r["nearest_neighbors"].get("_mean_coherence", 0)
            print(f"  {tau_key} {p_key}: analogy={a.get('acc_add', 0):.3f}  "
                  f"eff_rank={g['effective_rank']:.1f}  "
                  f"coherence={coh:.4f}  "
                  f"similarity_rho={r['similarity'].get('spearman_rho', 'n/a')}")

    # Print comparison
    print(f"\n{'='*80}")
    print("VOCAB-ONLY EMBEDDING GEOMETRY COMPARISON")
    print(f"{'='*80}")
    _print_comparison_table(results["evaluations"])

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/vocab_only_embedding_eval.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/vocab_only_embedding_eval.json")
    return results


def _print_comparison_table(evaluations: dict):
    """Print a formatted comparison table across all conditions."""
    tau_keys = sorted(evaluations.keys())
    p_keys = set()
    for tk in tau_keys:
        p_keys.update(evaluations[tk].keys())
    p_keys = sorted(p_keys, key=lambda x: int(x.split("_")[1]))

    for p_key in p_keys:
        P = int(p_key.split("_")[1])
        print(f"\n--- P = {P:,} ---")
        header = f"{'Metric':>20s}" + "".join(f"  {tk:>22s}" for tk in tau_keys)
        print(header)
        print("-" * len(header))

        def _get(tk, *path):
            d = evaluations.get(tk, {}).get(p_key, {})
            for p in path:
                if isinstance(d, dict):
                    d = d.get(p)
                else:
                    return None
            return d

        rows = [
            ("Analogy (3CosAdd)", lambda tk: _get(tk, "analogies", "overall", "acc_add")),
            ("Similarity (rho)", lambda tk: _get(tk, "similarity", "spearman_rho")),
            ("Eff. rank", lambda tk: _get(tk, "geometry", "effective_rank")),
            ("Avg cosine", lambda tk: _get(tk, "geometry", "avg_cosine")),
            ("Top-10 SV energy", lambda tk: _get(tk, "geometry", "top10_energy")),
            ("NN coherence", lambda tk: _get(tk, "nearest_neighbors", "_mean_coherence")),
            ("Cluster ratio", lambda tk: _get(tk, "clustering", "overall", "mean_ratio")),
        ]

        for label, fn in rows:
            row = f"{label:>20s}"
            for tk in tau_keys:
                v = fn(tk)
                row += f"  {v:22.4f}" if v is not None else f"  {'n/a':>22s}"
            print(row)


# ---------------------------------------------------------------------------
# Contextual embedding eval (per-layer hidden states)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def vocab_only_contextual_eval(taus: str = "0.1,0.3,0.5", n_batches: int = 500):
    """Contextual embedding evaluation for vocab-only models.

    Extracts context-averaged per-layer hidden states and runs the full
    geometry battery. Tests whether the coherence advantage (or any
    structural difference) persists through model layers.
    """
    import os, json, glob
    import torch
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.eval_embeddings import build_token_index, run_eval

    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    tau_list = [float(t) for t in taus.split(",")]
    batch_size = 64
    block_size = 128
    results = {}

    # Include tau=0.0 baseline
    for tau in [0.0] + tau_list:
        if tau == 0.0:
            model_dir = f"{DATA_DIR}/models/tau_0.000/P_100000000/T_128"
            data_dir = f"{DATA_DIR}/tokens"
            tau_key = "tau_0.000"
        else:
            model_dir = f"{DATA_DIR}/models/vocab_only/tau_{tau:.3f}/P_100000000/T_128"
            data_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}"
            tau_key = f"vocab_only_tau_{tau:.3f}"

        if not os.path.exists(model_dir):
            print(f"  [skip] {model_dir} not found")
            continue

        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]
        n_layer = cfg.get("n_layer", 2)
        n_embd = cfg.get("n_embd", 128)

        model = GPT(
            vocab_size=vocab_size, block_size=block_size,
            n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
        )
        model.load_state_dict(sd)
        model.eval()

        static_emb = model.transformer.wte.weight.detach().numpy()

        shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
        chunks = []
        total = 0
        needed = n_batches * batch_size * block_size * 2
        for path in shard_paths:
            tokens = np.load(path)
            chunks.append(tokens)
            total += len(tokens)
            if total >= needed:
                break
        data = torch.from_numpy(np.concatenate(chunks).astype(np.int64))

        layer_sums = [np.zeros((vocab_size, n_embd), dtype=np.float64)
                      for _ in range(n_layer)]
        token_counts = np.zeros(vocab_size, dtype=np.int64)

        print(f"\n--- Contextual embeddings: {tau_key} ({n_batches} batches) ---")
        with torch.no_grad():
            for bi in range(n_batches):
                ix = torch.randint(len(data) - block_size - 1, (batch_size,))
                x = torch.stack([data[i:i+block_size] for i in ix])

                tok_emb = model.transformer.wte(x)
                if not model.use_rope:
                    pos = torch.arange(block_size)
                    h = model.transformer.drop(tok_emb + model.transformer.wpe(pos))
                else:
                    h = model.transformer.drop(tok_emb)

                layer_outs = []
                for blk in model.transformer.h:
                    h = blk(h)
                    layer_outs.append(h)

                x_flat = x.numpy().reshape(-1)
                for l, h_l in enumerate(layer_outs):
                    h_flat = h_l.numpy().reshape(-1, n_embd)
                    np.add.at(layer_sums[l], x_flat, h_flat)
                np.add.at(token_counts, x_flat, 1)

                if bi % 100 == 0:
                    print(f"  batch {bi}/{n_batches}")

        has_data = token_counts > 0
        ctx_embs = []
        for l in range(n_layer):
            avg = np.zeros_like(layer_sums[l])
            avg[has_data] = layer_sums[l][has_data] / token_counts[has_data, None]
            ctx_embs.append(avg)

        results[tau_key] = {}
        for label, emb in [("static", static_emb)] + \
                           [(f"layer{l+1}", ctx_embs[l]) for l in range(n_layer)]:
            r = run_eval(emb, token_index)
            results[tau_key][label] = r

            a = r["analogies"].get("overall", {})
            g = r["geometry"]
            coh = r["nearest_neighbors"].get("_mean_coherence", 0)
            print(f"  [{tau_key}] {label}: analogy={a.get('acc_add', 0):.3f}  "
                  f"eff_rank={g['effective_rank']:.1f}  coherence={coh:.4f}")

    # Print comparison
    print(f"\n{'='*80}")
    print("VOCAB-ONLY CONTEXTUAL EMBEDDING COMPARISON (P=100M)")
    print(f"{'='*80}")

    layers = ["static", "layer1", "layer2"]
    tau_keys = sorted(results.keys())

    header = f"{'Metric':>20s}" + "".join(
        f"  {tk}/{l:>6s}" for tk in tau_keys for l in layers
    )
    print(header[:200] + "...")

    for label, path in [
        ("Analogy", ("analogies", "overall", "acc_add")),
        ("Eff. rank", ("geometry", "effective_rank")),
        ("Avg cosine", ("geometry", "avg_cosine")),
        ("NN coherence", ("nearest_neighbors", "_mean_coherence")),
        ("Cluster ratio", ("clustering", "overall", "mean_ratio")),
    ]:
        row = f"{label:>20s}"
        for tk in tau_keys:
            for l in layers:
                d = results.get(tk, {}).get(l, {})
                for p in path:
                    d = d.get(p) if isinstance(d, dict) else None
                row += f"  {d:8.4f}" if d is not None else f"  {'n/a':>8s}"
        print(row)

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/vocab_only_contextual_eval.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/vocab_only_contextual_eval.json")
    return results


# ---------------------------------------------------------------------------
# Collapse characterization — what does many-to-one mapping look like?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def vocab_only_collapse_analysis(taus: str = "0.1,0.3,0.5"):
    """Characterize the many-to-one token collapse under vocab-only reduction.

    For each tau, analyzes:
    1. Collapse statistics: how many source tokens map to each target?
    2. Semantic coherence of collapsed groups: do tokens mapping to the same
       target form semantically meaningful clusters?
    3. Information loss: how much embedding variance is lost in the mapping?
    4. Which analogy-relevant words are collapsed vs preserved?
    """
    import os, json
    import numpy as np
    from language_reduction.eval_embeddings import (
        build_token_index, ANALOGIES, NN_QUERY_WORDS,
    )
    from language_reduction.denoise import build_inverse_vocab_map

    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)
    s2i = token_index["str_to_id"]
    i2s = token_index["id_to_str"]

    tau_list = [float(t) for t in taus.split(",")]
    results = {}

    for tau in tau_list:
        print(f"\n{'='*60}")
        print(f"COLLAPSE ANALYSIS: tau={tau}")
        print(f"{'='*60}")

        # Load the tau-specific vocab_map and build full token_id -> token_id mapping
        tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
        vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
        tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
        inv_vocab_map = build_inverse_vocab_map(vocab_map, tau_top_ids)
        # mapping[token_id] = replacement token_id (identity if in-vocab)
        mapping = inv_vocab_map[vocab_map]

        # 1. Collapse statistics
        from collections import Counter
        target_counts = Counter()
        collapsed_sources = {}  # target_id -> list of source_ids
        n_replaced = 0
        n_identity = 0

        for src_id, tgt_id in enumerate(mapping):
            if src_id == tgt_id:
                n_identity += 1
            else:
                n_replaced += 1
                target_counts[int(tgt_id)] += 1
                if int(tgt_id) not in collapsed_sources:
                    collapsed_sources[int(tgt_id)] = []
                collapsed_sources[int(tgt_id)].append(src_id)

        v_prime = n_identity
        print(f"\n  V' (in-vocab tokens): {v_prime}")
        print(f"  Replaced tokens: {n_replaced}")
        print(f"  Unique targets receiving replacements: {len(collapsed_sources)}")

        # Distribution of collapse ratios
        collapse_sizes = sorted(target_counts.values(), reverse=True)
        print(f"  Max collapse (most sources -> one target): {collapse_sizes[0]}")
        print(f"  Mean collapse ratio: {np.mean(collapse_sizes):.1f}")
        print(f"  Median collapse ratio: {np.median(collapse_sizes):.1f}")

        # Top 20 most-collapsed-to targets
        print(f"\n  Top 20 targets (most sources mapped to them):")
        top_targets = target_counts.most_common(20)
        for tgt_id, count in top_targets:
            tgt_str = i2s.get(tgt_id, f"[{tgt_id}]").strip()
            sources = collapsed_sources[tgt_id][:5]
            src_strs = [i2s.get(s, f"[{s}]").strip() for s in sources]
            print(f"    {tgt_str!r:15s} <- {count} sources  (e.g., {src_strs})")

        # 2. Semantic coherence of collapsed groups
        # Load GPT-2 embeddings (what the mapping is based on)
        import torch
        try:
            from transformers import GPT2Model
            gpt2 = GPT2Model.from_pretrained("gpt2")
            gpt2_emb = gpt2.wte.weight.detach().numpy()
        except Exception:
            gpt2_emb_path = f"{stats_dir}/gpt2_embeddings.npy"
            if os.path.exists(gpt2_emb_path):
                gpt2_emb = np.load(gpt2_emb_path)
            else:
                print("  [skip] Cannot load GPT-2 embeddings for coherence analysis")
                gpt2_emb = None

        coherence_scores = []
        if gpt2_emb is not None:
            norms = np.linalg.norm(gpt2_emb, axis=1, keepdims=True)
            gpt2_emb_n = gpt2_emb / (norms + 1e-10)

            # Compute avg intra-group cosine for groups with 3+ members
            large_groups = [(tgt, srcs) for tgt, srcs in collapsed_sources.items()
                          if len(srcs) >= 3]
            for tgt_id, srcs in large_groups[:200]:
                src_embs = gpt2_emb_n[srcs]
                if len(src_embs) < 3:
                    continue
                cos_mat = src_embs @ src_embs.T
                mask = ~np.eye(len(src_embs), dtype=bool)
                intra_cos = float(cos_mat[mask].mean())
                coherence_scores.append(intra_cos)

            if coherence_scores:
                print(f"\n  Collapsed group coherence (GPT-2 embedding cosine):")
                print(f"    Mean intra-group cosine: {np.mean(coherence_scores):.4f}")
                print(f"    Median: {np.median(coherence_scores):.4f}")
                print(f"    Min: {np.min(coherence_scores):.4f}")
                print(f"    Max: {np.max(coherence_scores):.4f}")
                print(f"    (Computed over {len(coherence_scores)} groups with 3+ members)")

        # 3. Analogy word fate: which words are collapsed vs preserved?
        print(f"\n  Analogy word fate:")
        all_analogy_words = set()
        for pairs in ANALOGIES.values():
            for a, b, c, d in pairs:
                all_analogy_words.update([a, b, c, d])

        collapsed_analogy = []
        preserved_analogy = []
        for w in sorted(all_analogy_words):
            if w not in s2i:
                continue
            tid = s2i[w]
            if mapping[tid] != tid:
                tgt = int(mapping[tid])
                tgt_str = i2s.get(tgt, f"[{tgt}]").strip()
                collapsed_analogy.append((w, tgt_str))
            else:
                preserved_analogy.append(w)

        print(f"    Preserved ({len(preserved_analogy)}): {preserved_analogy[:30]}")
        if collapsed_analogy:
            print(f"    Collapsed ({len(collapsed_analogy)}):")
            for w, tgt in collapsed_analogy[:20]:
                print(f"      {w!r:15s} -> {tgt!r}")

        # 4. Query word neighborhoods under collapse
        print(f"\n  Query word neighborhoods (post-collapse):")
        for w in NN_QUERY_WORDS[:10]:
            if w not in s2i:
                continue
            tid = s2i[w]
            if mapping[tid] != tid:
                tgt = int(mapping[tid])
                tgt_str = i2s.get(tgt, f"[{tgt}]").strip()
                print(f"    {w!r:10s} -> collapsed to {tgt_str!r}")
            else:
                # Which tokens collapse onto this word?
                if tid in collapsed_sources:
                    srcs = collapsed_sources[tid][:8]
                    src_strs = [i2s.get(s, "?").strip() for s in srcs]
                    print(f"    {w!r:10s} <- absorbs {len(collapsed_sources[tid])} tokens "
                          f"(e.g., {src_strs})")
                else:
                    print(f"    {w!r:10s} (preserved, no inbound)")

        results[f"tau_{tau:.3f}"] = {
            "v_prime": v_prime,
            "n_replaced": n_replaced,
            "n_unique_targets": len(collapsed_sources),
            "max_collapse": collapse_sizes[0] if collapse_sizes else 0,
            "mean_collapse": float(np.mean(collapse_sizes)) if collapse_sizes else 0,
            "median_collapse": float(np.median(collapse_sizes)) if collapse_sizes else 0,
            "top_targets": [(i2s.get(t, "?").strip(), c)
                           for t, c in top_targets],
            "collapsed_analogy_words": collapsed_analogy,
            "preserved_analogy_words": preserved_analogy,
            "group_coherence_mean": float(np.mean(coherence_scores)) if coherence_scores else None,
            "group_coherence_median": float(np.median(coherence_scores)) if coherence_scores else None,
        }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/vocab_only_collapse_analysis.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/vocab_only_collapse_analysis.json")
    return results


# ---------------------------------------------------------------------------
# Contextual disambiguation — does the model recover sub-token identity?
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def vocab_only_structure_probe(tau: float = 0.5, n_positions: int = 50_000,
                               n_top_targets: int = 20, n_source_clusters: int = 8):
    """Probe whether the model contextually disambiguates collapsed tokens.

    At tau=0.5, "water" absorbs 684 source tokens. This stage asks: when the
    model sees "water" at position i, does its L2 hidden state carry information
    about which *original* token occupied that position?

    Tests:
    1. Contextual disambiguation: for high-fanin targets, do hidden states
       cluster by original source token? (measured via silhouette score and
       linear probe accuracy)
    2. Cross-entropy decomposition: how much of the model's predictive success
       comes from unigram, bigram, vs full-context information?
    3. Positional mutual information: are there n-gram regularities in the
       reduced alphabet that wouldn't exist in uniform random sequences?
    """
    import os, json, glob
    import torch
    import numpy as np
    from collections import Counter, defaultdict
    from language_reduction.model import GPT
    from language_reduction.denoise import build_inverse_vocab_map
    from language_reduction.shared import resolve_mode_key

    mode_key = resolve_mode_key("vocab_only", tau)
    block_size = 128
    source_P = 100_000_000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    if not os.path.exists(model_dir):
        print(f"ERROR: Model not found at {model_dir}")
        return {"error": f"model not found: {model_dir}"}

    with open(os.path.join(model_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_layer = cfg.get("n_layer", 2)
    n_embd = cfg.get("n_embd", 128)

    model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=cfg.get("n_head", 4), n_embd=n_embd,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    # Load vocab mapping
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/stats"
    vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
    tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    inv_vocab_map = build_inverse_vocab_map(vocab_map, tau_top_ids)
    full_mapping = inv_vocab_map[vocab_map]

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # Find high-fanin targets
    target_sources = defaultdict(list)
    for src_id in range(len(full_mapping)):
        tgt_id = int(full_mapping[src_id])
        if src_id != tgt_id:
            target_sources[tgt_id].append(src_id)

    # Sort by fanin, pick top targets
    high_fanin = sorted(target_sources.items(), key=lambda x: -len(x[1]))[:n_top_targets]

    print(f"Top {n_top_targets} targets by fanin:")
    for tgt_id, srcs in high_fanin:
        tgt_str = enc.decode([tgt_id]).strip()
        print(f"  {tgt_str!r:15s} <- {len(srcs)} sources")

    # ===== Test 1: Contextual disambiguation =====
    print(f"\n{'='*60}")
    print("TEST 1: CONTEXTUAL DISAMBIGUATION")
    print(f"{'='*60}")
    print("Do hidden states at same-token positions cluster by original source?")

    # Load original + vocab-only shards in parallel
    orig_shard_paths = sorted(glob.glob(f"{DATA_DIR}/tokens/shard_*.npy"))[:5]
    vo_shard_paths = sorted(glob.glob(f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/shard_*.npy"))[:5]

    # For each high-fanin target, collect (position, original_token_id) pairs
    # Then extract hidden states and test clustering
    disambiguation_results = []

    for tgt_id, src_ids in high_fanin[:10]:
        tgt_str = enc.decode([tgt_id]).strip()

        # Pick top source clusters by frequency
        src_id_set = set(src_ids)
        src_counts = Counter()

        # Count source occurrences
        for orig_path in orig_shard_paths[:2]:
            orig_tokens = np.load(orig_path)
            for tid in orig_tokens:
                if int(tid) in src_id_set:
                    src_counts[int(tid)] += 1

        # Pick most frequent sources for this target
        top_sources = [s for s, _ in src_counts.most_common(n_source_clusters)]
        if len(top_sources) < 2:
            continue

        # Collect windows where the center token maps to this target
        # AND the original was one of our top sources
        source_windows = defaultdict(list)  # src_id -> list of (window,) in vocab-only space
        max_per_source = n_positions // (len(top_sources) * len(high_fanin[:10]))

        for orig_path, vo_path in zip(orig_shard_paths[:3], vo_shard_paths[:3]):
            orig_tokens = np.load(orig_path).astype(np.int64)
            vo_tokens = np.load(vo_path).astype(np.int64)
            half = block_size // 2

            for i in range(half, len(orig_tokens) - half):
                orig_tid = int(orig_tokens[i])
                if orig_tid not in top_sources:
                    continue
                if int(vo_tokens[i]) != tgt_id:
                    continue
                if len(source_windows[orig_tid]) >= max_per_source:
                    continue

                window = vo_tokens[i - half: i - half + block_size]
                if len(window) == block_size:
                    source_windows[orig_tid].append((window, half))

            # Check if we have enough
            if all(len(v) >= max_per_source for v in source_windows.values()
                   if len(source_windows) >= 2):
                break

        # Filter sources with enough samples
        valid_sources = {s: ws for s, ws in source_windows.items() if len(ws) >= 10}
        if len(valid_sources) < 2:
            print(f"  {tgt_str!r}: insufficient samples ({len(valid_sources)} valid sources)")
            continue

        # Extract hidden states at the target position for each source class
        hidden_states = []  # (n_samples, n_embd)
        labels = []  # source_id for each sample

        with torch.no_grad():
            for src_id, windows in valid_sources.items():
                for window, pos_in_window in windows[:max_per_source]:
                    x = torch.from_numpy(window).unsqueeze(0).to(device)
                    # Manual forward to get per-layer outputs
                    tok_emb = model.transformer.wte(x)
                    if not model.use_rope:
                        pos = torch.arange(block_size, device=device)
                        h = model.transformer.drop(tok_emb + model.transformer.wpe(pos))
                    else:
                        h = model.transformer.drop(tok_emb)
                    for blk in model.transformer.h:
                        h = blk(h)
                    # Take L2 (final layer) hidden state at the target position
                    hs = h[0, pos_in_window].cpu().numpy()
                    hidden_states.append(hs)
                    labels.append(src_id)

        hidden_states = np.array(hidden_states)
        labels = np.array(labels)
        unique_labels = sorted(set(labels))

        if len(hidden_states) < 20 or len(unique_labels) < 2:
            continue

        # Compute between-class vs within-class variance (F-ratio proxy)
        overall_mean = hidden_states.mean(axis=0)
        between_var = 0.0
        within_var = 0.0
        for lbl in unique_labels:
            mask = labels == lbl
            class_data = hidden_states[mask]
            class_mean = class_data.mean(axis=0)
            between_var += mask.sum() * np.sum((class_mean - overall_mean) ** 2)
            within_var += np.sum((class_data - class_mean) ** 2)

        n_classes = len(unique_labels)
        n_total = len(hidden_states)
        f_ratio = (between_var / (n_classes - 1)) / (within_var / (n_total - n_classes) + 1e-10)

        # Linear probe: can we classify source identity from hidden state?
        probe_acc = linear_probe_accuracy(hidden_states, labels)

        src_strs = [enc.decode([s]).strip() for s in unique_labels[:5]]
        print(f"  {tgt_str!r:15s}: F-ratio={f_ratio:.2f}, probe_acc={probe_acc:.3f} "
              f"({n_total} samples, {n_classes} classes: {src_strs})")

        disambiguation_results.append({
            "target": tgt_str,
            "target_id": tgt_id,
            "n_sources_tested": n_classes,
            "n_samples": n_total,
            "f_ratio": float(f_ratio),
            "probe_accuracy": float(probe_acc),
            "chance_level": 1.0 / n_classes,
            "source_words": src_strs,
        })

    # ===== Test 2: Cross-entropy decomposition =====
    print(f"\n{'='*60}")
    print("TEST 2: CROSS-ENTROPY DECOMPOSITION")
    print(f"{'='*60}")
    print("How much predictive power comes from unigram vs context?")

    # Load vocab-only data
    vo_shard_paths = sorted(glob.glob(f"{DATA_DIR}/vocab_only/tau_{tau:.3f}/shard_*.npy"))[:3]
    all_tokens = []
    for path in vo_shard_paths:
        all_tokens.append(np.load(path))
    corpus = np.concatenate(all_tokens)[:5_000_000]

    # Unigram distribution
    token_counts = Counter(corpus.tolist())
    total = len(corpus)
    unigram_probs = np.zeros(vocab_size)
    for tid, count in token_counts.items():
        unigram_probs[tid] = count / total
    H_unigram = -np.sum(unigram_probs[unigram_probs > 0] *
                        np.log(unigram_probs[unigram_probs > 0]))

    # Bigram conditional entropy
    bigram_counts = Counter()
    for i in range(len(corpus) - 1):
        bigram_counts[(int(corpus[i]), int(corpus[i+1]))] += 1
    context_counts = Counter(corpus[:-1].tolist())

    H_bigram_sum = 0.0
    for (c, t), count in bigram_counts.items():
        p_t_given_c = count / context_counts[c]
        H_bigram_sum -= (count / (total - 1)) * np.log(p_t_given_c + 1e-15)

    # Model CE (from training results)
    model_ce = cfg.get("best_val_loss", float("nan"))

    print(f"  H(unigram)      = {H_unigram:.4f} nats")
    print(f"  H(bigram|prev)  = {H_bigram_sum:.4f} nats")
    print(f"  H(model, full)  = {model_ce:.4f} nats")
    print(f"  ")
    print(f"  Context gain (unigram → bigram): {H_unigram - H_bigram_sum:.4f} nats")
    print(f"  Context gain (bigram → model):   {H_bigram_sum - model_ce:.4f} nats")
    print(f"  Total context gain:              {H_unigram - model_ce:.4f} nats")
    pct_from_bigram = (H_unigram - H_bigram_sum) / (H_unigram - model_ce + 1e-10) * 100
    print(f"  % from bigram alone:             {pct_from_bigram:.1f}%")

    # ===== Test 3: Positional structure =====
    print(f"\n{'='*60}")
    print("TEST 3: N-GRAM STRUCTURE IN REDUCED ALPHABET")
    print(f"{'='*60}")

    # Top bigrams — are they linguistically meaningful?
    top_bigrams = bigram_counts.most_common(30)
    print(f"\n  Top 30 bigrams (of {len(bigram_counts)} unique):")
    for (c, t), count in top_bigrams:
        c_str = enc.decode([int(inv_vocab_map[vocab_map[c]])]).strip()[:12]
        t_str = enc.decode([int(inv_vocab_map[vocab_map[t]])]).strip()[:12]
        pmi = np.log((count / (total-1)) /
                     (token_counts[c] / total * token_counts[t] / total + 1e-15))
        print(f"    {c_str:>12s} → {t_str:<12s}  count={count:>7,}  PMI={pmi:+.2f}")

    # How peaked is the bigram distribution? (effective number of bigrams)
    bigram_probs = np.array(list(bigram_counts.values()), dtype=np.float64)
    bigram_probs /= bigram_probs.sum()
    eff_bigrams = np.exp(-np.sum(bigram_probs * np.log(bigram_probs + 1e-15)))
    max_bigrams = vocab_size * vocab_size
    actual_bigrams = len(bigram_counts)

    print(f"\n  Bigram diversity:")
    print(f"    Possible (V²):    {max_bigrams:,}")
    print(f"    Observed:         {actual_bigrams:,} ({100*actual_bigrams/max_bigrams:.1f}%)")
    print(f"    Effective:        {eff_bigrams:.0f} ({100*eff_bigrams/actual_bigrams:.1f}% of observed)")

    # Save results
    results = {
        "tau": tau,
        "disambiguation": disambiguation_results,
        "ce_decomposition": {
            "H_unigram": float(H_unigram),
            "H_bigram": float(H_bigram_sum),
            "H_model": float(model_ce),
            "pct_from_bigram": float(pct_from_bigram),
        },
        "bigram_structure": {
            "n_possible": int(max_bigrams),
            "n_observed": int(actual_bigrams),
            "n_effective": float(eff_bigrams),
        },
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/vocab_only_structure_probe_{tau:.3f}.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/vocab_only_structure_probe_{tau:.3f}.json")
    return results


def linear_probe_accuracy(X, y):
    """Simple leave-20%-out linear probe using least-squares."""
    import numpy as np
    unique = sorted(set(y))
    if len(unique) < 2:
        return 0.0
    label_map = {lbl: i for i, lbl in enumerate(unique)}
    y_int = np.array([label_map[lbl] for lbl in y])

    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train, y_train = X[train_idx], y_int[train_idx]
    X_test, y_test = X[test_idx], y_int[test_idx]

    # One-vs-rest least-squares
    n_classes = len(unique)
    Y_onehot = np.zeros((len(X_train), n_classes))
    for i, lbl in enumerate(y_train):
        Y_onehot[i, lbl] = 1.0

    # Ridge regression (lambda=1.0)
    XtX = X_train.T @ X_train + np.eye(X_train.shape[1])
    W = np.linalg.solve(XtX, X_train.T @ Y_onehot)

    preds = X_test @ W
    pred_labels = preds.argmax(axis=1)
    return float((pred_labels == y_test).mean())


# ---------------------------------------------------------------------------
# Concept recovery on vocab-only models (data-driven target selection)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=7200,
    memory=32768,
)
def vocab_only_recovery_full(source_tau: float = 0.3, lr: float = 1e-4,
                             target_tokens: int = 50_000, n_target_words: int = 15):
    """Data-driven concept recovery on a vocab-only model.

    Unlike the spectral recovery experiment (which targeted "queen" specifically),
    this stage first discovers which semantically interesting tokens were collapsed
    at the given tau, then builds a curriculum and evaluates recovery for those
    tokens specifically.

    Pipeline:
    1. Discover collapsed tokens: find tokens that (a) are out-of-vocab at this
       tau, (b) occur frequently enough in the original corpus to build a
       curriculum, and (c) are semantically interesting (have distinct meanings
       from their replacement target).
    2. Build a targeted curriculum from original corpus windows containing
       the discovered tokens.
    3. Fine-tune the vocab-only model on this curriculum.
    4. Evaluate: did the collapsed tokens differentiate from their targets?
       Did neighboring tokens restructure?
    """
    import os, json, glob, torch
    import numpy as np
    import tiktoken
    from collections import Counter
    from language_reduction.model import GPT
    from language_reduction.eval_embeddings import build_token_index, run_eval
    from language_reduction.denoise import build_inverse_vocab_map
    from language_reduction.shared import resolve_mode_key

    mode = "vocab_only"
    mode_key = resolve_mode_key(mode, source_tau)
    block_size = 128
    source_P = 100_000_000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Step 1: Discover collapsed tokens =====
    print(f"{'='*60}")
    print(f"STEP 1: DISCOVER COLLAPSED TOKENS (tau={source_tau})")
    print(f"{'='*60}")

    stats_dir = f"{DATA_DIR}/stats"
    tau_stats_dir = f"{DATA_DIR}/vocab_only/tau_{source_tau:.3f}/stats"
    vocab_map = np.load(os.path.join(tau_stats_dir, "vocab_map.npy"))
    tau_top_ids = np.load(os.path.join(tau_stats_dir, "top_ids.npy"))
    inv_vocab_map = build_inverse_vocab_map(vocab_map, tau_top_ids)
    full_mapping = inv_vocab_map[vocab_map]  # token_id -> replacement token_id

    enc = tiktoken.get_encoding("gpt2")
    top_ids_set = set(int(x) for x in tau_top_ids)

    # Count occurrences of collapsed tokens in the original corpus
    print("  Counting collapsed token occurrences in original corpus...")
    collapsed_counts = Counter()
    shard_paths = sorted(glob.glob(f"{DATA_DIR}/tokens/shard_*.npy"))[:5]
    for path in shard_paths:
        tokens = np.load(path)
        for tid in tokens:
            tid = int(tid)
            if tid not in top_ids_set:
                collapsed_counts[tid] += 1

    # Filter: need tokens that are (a) collapsed, (b) frequent enough, (c) decodable
    # to a clean single word
    candidates = []
    for tid, count in collapsed_counts.most_common(500):
        if count < 100:
            break
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        if not cleaned or len(cleaned) < 2:
            continue
        if not cleaned.isalpha():
            continue
        tgt_id = int(full_mapping[tid])
        tgt_raw = enc.decode([tgt_id])
        tgt_cleaned = tgt_raw.strip().lower()
        candidates.append({
            "token_id": tid,
            "word": cleaned,
            "raw": raw,
            "count": count,
            "target_id": tgt_id,
            "target_word": tgt_cleaned,
            "target_raw": tgt_raw,
        })

    # Rank by semantic distance from target (we want tokens that collapsed
    # onto something meaningfully different, not just casing variants)
    # Use a simple heuristic: edit distance > 2 and different first 3 chars
    interesting = []
    for c in candidates:
        if c["word"] == c["target_word"]:
            continue
        if c["word"].startswith(c["target_word"]) or c["target_word"].startswith(c["word"]):
            continue
        interesting.append(c)

    # Take top n_target_words by corpus frequency
    target_words = interesting[:n_target_words]

    if not target_words:
        print("ERROR: No suitable collapsed tokens found at this tau.")
        print(f"  Total collapsed tokens with count >= 100: {len(candidates)}")
        print(f"  After filtering for semantic distinctness: {len(interesting)}")
        return {"error": "no suitable targets", "tau": source_tau}

    target_token_ids = set(c["token_id"] for c in target_words)
    print(f"\n  Selected {len(target_words)} target tokens for recovery:")
    for c in target_words:
        print(f"    {c['word']!r:15s} (id={c['token_id']:5d}, count={c['count']:>6,}) "
              f"-> collapsed to {c['target_word']!r}")

    # ===== Step 2: Build curriculum =====
    print(f"\n{'='*60}")
    print(f"STEP 2: BUILD CURRICULUM")
    print(f"{'='*60}")

    window_size = block_size
    windows = []
    total_collected = 0
    max_shards = 10

    for shard_idx in range(max_shards):
        shard_path = f"{DATA_DIR}/tokens/shard_{shard_idx:05d}.npy"
        if not os.path.exists(shard_path):
            break
        tokens = np.load(shard_path).astype(np.int64)
        half = window_size // 2

        for i in range(half, len(tokens) - half):
            if int(tokens[i]) in target_token_ids:
                start = max(0, i - half)
                end = min(len(tokens), start + window_size)
                if end - start < window_size:
                    continue
                window = tokens[start:end]
                windows.append(window)
                total_collected += window_size
                if total_collected >= target_tokens:
                    break
        if total_collected >= target_tokens:
            break

    if len(windows) < 10:
        print(f"ERROR: Only found {len(windows)} windows. Need more shards or lower target_tokens.")
        return {"error": "insufficient curriculum data", "n_windows": len(windows)}

    curriculum = np.concatenate(windows)[:target_tokens]
    print(f"  Extracted {len(windows)} windows, {len(curriculum):,} tokens total")

    # Save curriculum metadata
    curriculum_dir = f"{DATA_DIR}/curriculum_vocab_only/tau_{source_tau:.3f}"
    os.makedirs(curriculum_dir, exist_ok=True)
    np.save(os.path.join(curriculum_dir, "curriculum.npy"), curriculum)
    with open(os.path.join(curriculum_dir, "meta.json"), "w") as f:
        json.dump({
            "target_words": target_words,
            "n_windows": len(windows),
            "n_tokens": len(curriculum),
            "source_tau": source_tau,
        }, f, indent=2)
    volume.commit()

    # ===== Step 3: Fine-tune =====
    print(f"\n{'='*60}")
    print(f"STEP 3: FINE-TUNE (lr={lr})")
    print(f"{'='*60}")

    curriculum_t = torch.from_numpy(curriculum)
    source_dir = f"{DATA_DIR}/models/{mode_key}/P_{source_P}/T_{block_size}"
    if not os.path.exists(source_dir):
        print(f"ERROR: Source model not found at {source_dir}")
        return {"error": f"model not found: {source_dir}"}

    with open(os.path.join(source_dir, "results.json")) as f:
        cfg = json.load(f)
    sd = torch.load(os.path.join(source_dir, "model.pt"), map_location="cpu")
    vocab_size = sd["transformer.wte.weight"].shape[0]

    model = GPT(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=cfg.get("n_layer", 2),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
    ).to(device)
    model.load_state_dict(sd)

    split = int(0.9 * len(curriculum_t))
    train_data = curriculum_t[:split]
    val_data = curriculum_t[split:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    batch_size = 64
    n_epochs = 5
    eval_interval = 100
    tokens_per_step = batch_size * block_size
    steps_per_epoch = max(1, len(train_data) // tokens_per_step)
    n_steps = n_epochs * steps_per_epoch
    print(f"  Training: {n_steps} steps ({n_epochs} epochs)")

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:5d}/{n_steps}: train={float(loss):.4f} "
                  f"val={val_loss:.4f} best={best_val_loss:.4f}")

    save_dir = f"{DATA_DIR}/models/finetune/{mode_key}/P_{source_P}/T_{block_size}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state if best_state else model.state_dict(),
               os.path.join(save_dir, "model.pt"))
    ft_meta = {
        "mode": mode, "source_tau": source_tau, "source_P": source_P,
        "lr": lr, "n_steps": n_steps, "best_val_loss": best_val_loss,
        "target_words": [c["word"] for c in target_words],
        "n_layer": cfg.get("n_layer", 2),
        "n_head": cfg.get("n_head", 4),
        "n_embd": cfg.get("n_embd", 128),
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(ft_meta, f, indent=2)
    volume.commit()

    # ===== Step 4: Evaluate differentiation =====
    print(f"\n{'='*60}")
    print(f"STEP 4: EVALUATE TOKEN DIFFERENTIATION")
    print(f"{'='*60}")

    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    def load_emb(model_dir):
        sd_l = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        return sd_l["transformer.wte.weight"].numpy()

    emb_start = load_emb(source_dir)
    emb_gt = load_emb(f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}")
    emb_ft = load_emb(save_dir)

    # For each target word: did it differentiate from its collapse target?
    print("\n  Per-token differentiation:")
    print(f"  {'word':<15s} {'collapsed_to':<15s} "
          f"{'cos(start,tgt)':<16s} {'cos(ft,tgt)':<14s} "
          f"{'Δ(toward gt)':<14s} {'shift mag'}")
    print("  " + "-" * 90)

    differentiation_results = []
    for c in target_words:
        tid = c["token_id"]
        tgt_id = c["target_id"]

        # How similar was the token to its collapse target before/after fine-tuning?
        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

        cos_start_tgt = cosine(emb_start[tid], emb_start[tgt_id])
        cos_ft_tgt = cosine(emb_ft[tid], emb_ft[tgt_id])
        differentiated = cos_ft_tgt < cos_start_tgt  # lower = more differentiated

        # Did it move toward the ground truth model's representation?
        cos_start_gt = cosine(emb_start[tid], emb_gt[tid])
        cos_ft_gt = cosine(emb_ft[tid], emb_gt[tid])
        toward_gt = cos_ft_gt - cos_start_gt

        # How much did it move at all?
        shift_mag = 1.0 - cosine(emb_ft[tid], emb_start[tid])

        differentiation_results.append({
            "word": c["word"],
            "target_word": c["target_word"],
            "token_id": tid,
            "target_id": tgt_id,
            "cos_start_to_target": cos_start_tgt,
            "cos_ft_to_target": cos_ft_tgt,
            "differentiated": differentiated,
            "toward_gt": toward_gt,
            "shift_magnitude": shift_mag,
        })

        marker = " <--" if differentiated else ""
        print(f"  {c['word']:<15s} {c['target_word']:<15s} "
              f"{cos_start_tgt:<16.4f} {cos_ft_tgt:<14.4f} "
              f"{toward_gt:<+14.4f} {shift_mag:.4f}{marker}")

    n_differentiated = sum(1 for r in differentiation_results if r["differentiated"])
    mean_toward_gt = np.mean([r["toward_gt"] for r in differentiation_results])
    mean_shift = np.mean([r["shift_magnitude"] for r in differentiation_results])

    print(f"\n  Summary:")
    print(f"    Tokens differentiated from target: {n_differentiated}/{len(differentiation_results)}")
    print(f"    Mean shift toward ground truth: {mean_toward_gt:+.4f}")
    print(f"    Mean shift magnitude: {mean_shift:.4f}")

    # Also run full eval battery for overall geometry comparison
    print("\n  Full geometry eval:")
    eval_start = run_eval(emb_start, token_index)
    eval_gt = run_eval(emb_gt, token_index)
    eval_ft = run_eval(emb_ft, token_index)

    for label, ev in [("start", eval_start), ("fine-tuned", eval_ft), ("ground truth", eval_gt)]:
        a = ev["analogies"].get("overall", {})
        g = ev["geometry"]
        coh = ev["nearest_neighbors"].get("_mean_coherence", 0)
        print(f"    {label:>12s}: analogy={a.get('acc_add', 0):.3f}  "
              f"coherence={coh:.4f}  eff_rank={g['effective_rank']:.1f}")

    # Save everything
    all_results = {
        "target_words": target_words,
        "differentiation": differentiation_results,
        "summary": {
            "n_differentiated": n_differentiated,
            "n_total": len(differentiation_results),
            "mean_toward_gt": float(mean_toward_gt),
            "mean_shift_magnitude": float(mean_shift),
        },
        "finetune": ft_meta,
        "eval_start": eval_start,
        "eval_ft": eval_ft,
        "eval_gt": eval_gt,
    }

    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/vocab_only_recovery_{source_tau:.3f}.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {results_dir}/vocab_only_recovery_{source_tau:.3f}.json")
    return all_results
