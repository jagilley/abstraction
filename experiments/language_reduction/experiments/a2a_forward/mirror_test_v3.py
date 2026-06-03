"""Mirror test v3: topic-level perturbation discrimination.

Inspired by Vogel (2025), "Small Models Can Introspect, Too." Vogel
showed that injecting concept-specific steering vectors (e.g. "cat",
"bread") produces concept-specific logit shifts in a 32B model — the
model can identify *what* was injected, not just *that* something was.

Mirror test v2's perturbation discrimination (Test 2) was a null result
because random perturbation directions in 256-D are nearly orthogonal,
producing trivially uncorrelated logit shifts regardless of self-knowledge.

This test fixes that by using topic-specific perturbation directions
derived from the training data itself. The question: does cerebellar
co-training give the model better introspective ability — can the CL
model's output distribution better identify *which topic direction* was
injected into its computation?

Design:
  Phase 1 — Compute topic directions:
    Classify eval sequences by topic (keyword matching in decoded text).
    Compute contrastive post_block1 directions per topic:
      d_topic = normalize(mean(post_block1 | topic) - mean(post_block1))

  Phase 2 — Perturbation discrimination:
    For each topic direction, perturb at post_block1 and measure:
    (a) Mean logit boost for each topic's vocabulary (perturbation × topic
        logit matrix). Diagonal enrichment = each perturbation preferentially
        boosts its own topic's tokens.
    (b) Per-position Δloss (robustness gap replication).

  Controls:
    Random directions (same metric — should show no diagonal structure).
    Perturbation direction overlap (cosine between topic directions).

Tests the broad self-awareness claim: does processing self-predictions
cause the model to develop general computational self-awareness that
manifests as improved concept-level introspection?
"""

import json

from language_reduction.shared import app, volume, DATA_DIR


TOPICS = {
    "math": {
        "keywords": [
            "equation", "calculate", "formula", "algebra", "geometry",
            "integer", "fraction", "multiply", "subtract", "theorem",
            "mathematical", "arithmetic", "polynomial", "derivative",
            "quadratic", "coefficient", "variable", "graph",
        ],
        "vocab_keywords": [
            "math", "number", "equation", "formula", "calculate",
            "algebra", "geometry", "integer", "fraction", "multiply",
            "subtract", "sum", "theorem", "variable", "graph",
            "polynomial", "coefficient", "digit", "decimal",
        ],
    },
    "biology": {
        "keywords": [
            "organism", "species", "biology", "DNA", "chromosome",
            "photosynthesis", "ecosystem", "evolution", "bacteria",
            "protein", "enzyme", "mitosis", "habitat", "biodiversity",
            "cellular", "nucleus", "membrane", "genetics",
        ],
        "vocab_keywords": [
            "cell", "organism", "species", "biology", "DNA",
            "protein", "enzyme", "bacteria", "evolution", "habitat",
            "ecosystem", "gene", "tissue", "organ", "plant",
            "animal", "blood", "bone", "brain",
        ],
    },
    "history": {
        "keywords": [
            "century", "empire", "ancient", "medieval", "revolution",
            "dynasty", "civilization", "colonial", "independence",
            "monarchy", "republic", "treaty", "conquest", "emperor",
            "archaeological", "historian", "warfare",
        ],
        "vocab_keywords": [
            "war", "century", "king", "empire", "ancient", "battle",
            "revolution", "dynasty", "emperor", "kingdom", "army",
            "soldier", "peace", "treaty", "colony", "independence",
            "nation", "republic",
        ],
    },
    "language": {
        "keywords": [
            "grammar", "vocabulary", "syntax", "pronoun", "adjective",
            "consonant", "vowel", "syllable", "metaphor", "narrative",
            "literary", "punctuation", "conjunction", "preposition",
            "dialect", "etymology", "linguistics",
        ],
        "vocab_keywords": [
            "word", "sentence", "language", "grammar", "poem",
            "story", "author", "verb", "noun", "adjective",
            "pronoun", "paragraph", "essay", "writing", "letter",
            "reading", "text", "speech",
        ],
    },
    "geography": {
        "keywords": [
            "continent", "latitude", "longitude", "hemisphere",
            "climate", "terrain", "peninsula", "archipelago",
            "tectonic", "erosion", "precipitation", "altitude",
            "equator", "glacier", "volcano", "earthquake",
        ],
        "vocab_keywords": [
            "country", "city", "river", "mountain", "ocean",
            "continent", "island", "lake", "desert", "forest",
            "climate", "population", "region", "border", "coast",
            "valley", "volcano", "earthquake",
        ],
    },
}

MIN_KEYWORD_HITS = 2


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=7200,
    memory=32768,
)
def a2a_mirror_test_v3(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
    n_eval_batches: int = 40,
    n_random_dirs: int = 5,
    ckpt_source: str = "controlled",
    ckpt_step: int = 0,
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.experiments.a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MIRROR TEST v3 (topic discrimination) on {device}")
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))

    enc = tiktoken.get_encoding("gpt2")

    # =========================================================
    # Load data
    # =========================================================
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = torch.from_numpy(
        np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]
    print(f"Loaded {len(data):,} tokens, {len(val_data):,} for eval")

    # =========================================================
    # Build topic token ID sets (for output measurement)
    # =========================================================
    topic_token_ids = {}
    for topic_name, topic_def in TOPICS.items():
        token_ids = set()
        for kw in topic_def["vocab_keywords"]:
            for variant in [kw, kw.capitalize(), " " + kw, " " + kw.capitalize()]:
                try:
                    ids = enc.encode(variant)
                    token_ids.update(ids)
                except Exception:
                    pass
        topic_token_ids[topic_name] = sorted(token_ids)
        print(f"  Topic '{topic_name}': {len(topic_token_ids[topic_name])} "
              f"output token IDs")

    all_topic_ids = set()
    for ids in topic_token_ids.values():
        all_topic_ids.update(ids)
    print(f"  Total unique topic token IDs: {len(all_topic_ids)} "
          f"/ {vocab_size} vocab")

    # =========================================================
    # Load models
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    if ckpt_source == "controlled":
        ckpt_root = (f"{DATA_DIR}/a2a_forward/controlled/"
                     f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
        open_dir = os.path.join(ckpt_root, "open_loop")
        closed_dir = os.path.join(ckpt_root, "closed_loop")
    elif ckpt_source == "extended":
        fwd_tag = (f"fwd{fwd_n_layer}L{fwd_n_head}H{fwd_d_head}d"
                   f"_mlp{fwd_mlp_mult}")
        ckpt_root = (f"{DATA_DIR}/a2a_forward/extended/"
                     f"{gap_tag}/inject{inject_after_block}/"
                     f"{fwd_tag}/P_{n_tokens}")
        step_dir = f"step_{ckpt_step:06d}"
        open_dir = os.path.join(ckpt_root, "open_loop", step_dir)
        closed_dir = os.path.join(ckpt_root, "closed_loop", step_dir)
    else:
        raise ValueError(f"Unknown ckpt_source: {ckpt_source}")
    print(f"Loading from: {ckpt_root}")

    model_open = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_open.load_state_dict(torch.load(
        os.path.join(open_dir, "model.pt"),
        map_location=device, weights_only=True))

    model_closed = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "model.pt"),
        map_location=device, weights_only=True))

    fwd_closed = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_closed.load_state_dict(torch.load(
        os.path.join(closed_dir, "fwd_model.pt"),
        map_location=device, weights_only=True))

    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(closed_dir, "gate.pt"),
        map_location=device, weights_only=True))

    model_open.eval()
    model_closed.eval()
    fwd_closed.eval()
    gate.eval()
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    # =========================================================
    # Eval batches (deterministic)
    # =========================================================
    eval_gen = torch.Generator().manual_seed(seed + 200)
    eval_indices = [
        torch.randint(len(val_data) - block_size - 1, (batch_size,),
                       generator=eval_gen)
        for _ in range(n_eval_batches)
    ]

    def make_batch(indices):
        x = torch.stack([val_data[i:i + block_size] for i in indices])
        y = torch.stack([val_data[i + 1:i + block_size + 1] for i in indices])
        return x.to(device), y.to(device)

    # =========================================================
    # Helper: run model with optional perturbation at post_block1
    # =========================================================
    def run_model(model, x, y, use_injection, perturbation=None):
        if use_injection and perturbation is not None:
            def cb_fn(act, _p=perturbation):
                return gate(fwd_closed(act)) + _p
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        elif use_injection:
            def cb_fn(act):
                return gate(fwd_closed(act))
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        elif perturbation is not None:
            def cb_fn(act, _p=perturbation):
                return _p.unsqueeze(0).unsqueeze(0).expand(
                    act.shape[0], act.shape[1], -1)
            return model(x, y, return_intermediates=True,
                         cerebellar_fn=cb_fn,
                         cerebellar_input_block=cerebellar_input_block,
                         cerebellar_inject_block=inject_after_block)
        else:
            return model(x, y, return_intermediates=True)

    # =========================================================
    # PHASE 1: Compute topic directions from post_block1 activations
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 1: Computing topic-specific perturbation directions")
    print(f"{'='*60}")

    def classify_sequence_topic(token_ids):
        """Classify a token sequence by topic using keyword matching."""
        text = enc.decode(token_ids.tolist()).lower()
        scores = {}
        for topic_name, topic_def in TOPICS.items():
            count = sum(1 for kw in topic_def["keywords"] if kw.lower() in text)
            scores[topic_name] = count
        best = max(scores, key=scores.get)
        if scores[best] >= MIN_KEYWORD_HITS:
            return best
        return None

    # Use the OL model for direction computation (neutral — not biased by
    # cerebellar training). Both models see the same data, so the topic
    # structure in post_block1 activations should be present in both.
    topic_act_sums = {t: torch.zeros(n_embd, device=device) for t in TOPICS}
    topic_counts = {t: 0 for t in TOPICS}
    global_act_sum = torch.zeros(n_embd, device=device)
    global_count = 0

    print("  Classifying sequences and accumulating activations...")
    with torch.no_grad():
        for bi, idx in enumerate(eval_indices):
            x, y = make_batch(idx)
            _, _, intermediates = model_open(x, y, return_intermediates=True)
            b1 = intermediates["post_block1"]  # (B, T, D)

            # Mean activation per sequence (average over positions)
            seq_means = b1.mean(dim=1)  # (B, D)
            global_act_sum += seq_means.sum(dim=0)
            global_count += seq_means.shape[0]

            for si in range(x.shape[0]):
                topic = classify_sequence_topic(x[si])
                if topic is not None:
                    topic_act_sums[topic] += seq_means[si]
                    topic_counts[topic] += 1

    global_mean = global_act_sum / global_count
    print(f"  Total sequences: {global_count}")

    topic_directions = {}
    topic_names_ordered = []
    for topic_name in TOPICS:
        count = topic_counts[topic_name]
        if count < 10:
            print(f"  Topic '{topic_name}': {count} sequences — SKIPPING (too few)")
            continue
        topic_mean = topic_act_sums[topic_name] / count
        direction = topic_mean - global_mean
        direction = direction / direction.norm()
        topic_directions[topic_name] = direction
        topic_names_ordered.append(topic_name)
        print(f"  Topic '{topic_name}': {count} sequences "
              f"({100*count/global_count:.1f}%)")

    n_topics = len(topic_names_ordered)
    print(f"\n  {n_topics} topics with sufficient data")

    # Direction overlap matrix
    print("\n  Topic direction pairwise cosine:")
    dir_matrix = torch.stack([topic_directions[t] for t in topic_names_ordered])
    cosine_overlap = (dir_matrix @ dir_matrix.T).cpu()
    for i, ti in enumerate(topic_names_ordered):
        row = " ".join(f"{cosine_overlap[i,j]:+.3f}" for j in range(n_topics))
        print(f"    {ti:>12s}: {row}")

    off_diag_mask = ~torch.eye(n_topics, dtype=torch.bool)
    mean_off_diag = cosine_overlap[off_diag_mask].mean().item()
    print(f"  Mean off-diagonal cosine: {mean_off_diag:.4f}")

    # Perturbation scale (from post_block1 std)
    with torch.no_grad():
        x_ref, y_ref = make_batch(eval_indices[0])
        _, _, vi_ref = model_closed(x_ref, y_ref, return_intermediates=True)
        b1_std = float(vi_ref["post_block1"].std())
    print(f"\n  post_block1 activation std: {b1_std:.4f}")

    # Also compute random control directions
    rng = np.random.default_rng(seed + 400)
    random_dirs = {}
    for i in range(n_random_dirs):
        d = rng.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        random_dirs[f"random_{i}"] = torch.from_numpy(d).to(device)

    # =========================================================
    # PHASE 2: Perturbation discrimination
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Topic perturbation discrimination")
    print(f"{'='*60}")

    conditions = {
        "CL+M": ("closed", True),
        "CL-M": ("closed", False),
        "OL":   ("open", False),
    }

    s_test = 2.0

    # Build topic token index tensors for efficient logit extraction
    topic_idx_tensors = {}
    for topic_name in topic_names_ordered:
        ids = [i for i in topic_token_ids[topic_name] if i < vocab_size]
        topic_idx_tensors[topic_name] = torch.tensor(ids, device=device)

    all_results = {}

    for cond_name, (model_key, use_injection) in conditions.items():
        model = model_closed if model_key == "closed" else model_open
        print(f"\n  [{cond_name}]")

        # --- Topic directions ---
        # Matrix: perturb_topic × measure_topic → mean logit boost
        logit_matrix = np.zeros((n_topics, n_topics))
        loss_deltas = {}
        response_norms = {}

        for pi, perturb_topic in enumerate(topic_names_ordered):
            dir_t = topic_directions[perturb_topic]
            perturbation = (s_test * b1_std) * dir_t

            # Accumulators per measured topic
            topic_logit_boosts = {t: [] for t in topic_names_ordered}
            ld_all = []
            rn_all = []

            with torch.no_grad():
                for bi, idx in enumerate(eval_indices):
                    x, y = make_batch(idx)
                    tgt = x[:, 1:]

                    logits_base, _, inter_base = run_model(
                        model, x, y, use_injection)
                    logits_pert, _, inter_pert = run_model(
                        model, x, y, use_injection, perturbation)

                    # Logit shift (B, T-1, V)
                    d_logits = logits_pert[:, :-1] - logits_base[:, :-1]

                    # Mean logit boost per topic vocabulary
                    for mi, measure_topic in enumerate(topic_names_ordered):
                        t_ids = topic_idx_tensors[measure_topic]
                        if len(t_ids) == 0:
                            continue
                        boost = d_logits[:, :, t_ids].mean().item()
                        topic_logit_boosts[measure_topic].append(boost)

                    # Loss delta
                    loss_base = -F.log_softmax(logits_base[:, :-1], dim=-1
                        ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    loss_pert = -F.log_softmax(logits_pert[:, :-1], dim=-1
                        ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    ld_all.append((loss_pert - loss_base).mean().item())

                    # Response norm
                    b3_base = inter_base["post_block3"][:, :-1]
                    b3_pert = inter_pert["post_block3"][:, :-1]
                    rn = (b3_pert - b3_base).norm(dim=-1).mean().item()
                    rn_all.append(rn)

            for mi, measure_topic in enumerate(topic_names_ordered):
                logit_matrix[pi, mi] = float(np.mean(
                    topic_logit_boosts[measure_topic]))

            loss_deltas[perturb_topic] = float(np.mean(ld_all))
            response_norms[perturb_topic] = float(np.mean(rn_all))

            # Show logit boost row
            row_str = " ".join(
                f"{logit_matrix[pi, mi]:+.4f}"
                for mi in range(n_topics))
            print(f"    perturb={perturb_topic:>12s}: "
                  f"logit boosts=[{row_str}]  "
                  f"Δloss={loss_deltas[perturb_topic]:+.4f}  "
                  f"||R||={response_norms[perturb_topic]:.3f}")

        # --- Random direction controls ---
        random_logit_matrix = np.zeros((n_random_dirs, n_topics))
        random_loss_deltas = []

        for ri, (rname, rdir) in enumerate(random_dirs.items()):
            perturbation = (s_test * b1_std) * rdir
            r_boosts = {t: [] for t in topic_names_ordered}
            r_ld = []

            with torch.no_grad():
                for bi, idx in enumerate(eval_indices):
                    x, y = make_batch(idx)
                    tgt = x[:, 1:]

                    logits_base, _, _ = run_model(model, x, y, use_injection)
                    logits_pert, _, _ = run_model(
                        model, x, y, use_injection, perturbation)

                    d_logits = logits_pert[:, :-1] - logits_base[:, :-1]

                    for mi, measure_topic in enumerate(topic_names_ordered):
                        t_ids = topic_idx_tensors[measure_topic]
                        if len(t_ids) == 0:
                            continue
                        boost = d_logits[:, :, t_ids].mean().item()
                        r_boosts[measure_topic].append(boost)

                    loss_base = -F.log_softmax(logits_base[:, :-1], dim=-1
                        ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    loss_pert = -F.log_softmax(logits_pert[:, :-1], dim=-1
                        ).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    r_ld.append((loss_pert - loss_base).mean().item())

            for mi, mt in enumerate(topic_names_ordered):
                random_logit_matrix[ri, mi] = float(np.mean(r_boosts[mt]))
            random_loss_deltas.append(float(np.mean(r_ld)))

        # --- Compute diagonal enrichment ---
        diag = np.diag(logit_matrix)
        off_diag = logit_matrix[~np.eye(n_topics, dtype=bool)]
        diag_mean = float(diag.mean())
        off_diag_mean = float(off_diag.mean())
        diag_enrichment = diag_mean - off_diag_mean

        # Normalized: what fraction of total boost is on the diagonal?
        total_boost = logit_matrix.sum()
        diag_frac = float(diag.sum() / total_boost) if abs(total_boost) > 1e-10 \
            else float('nan')
        uniform_frac = 1.0 / n_topics

        # Random direction control: variance across topics
        random_topic_var = float(random_logit_matrix.var(axis=1).mean())

        print(f"\n    --- Diagonal enrichment ---")
        print(f"    Diagonal mean:     {diag_mean:+.5f}")
        print(f"    Off-diagonal mean: {off_diag_mean:+.5f}")
        print(f"    Enrichment (Δ):    {diag_enrichment:+.5f}")
        print(f"    Diagonal fraction: {diag_frac:.4f} "
              f"(uniform={uniform_frac:.4f})")
        print(f"    Random dir topic variance: {random_topic_var:.6f}")
        print(f"    Mean Δloss (topic dirs): "
              f"{np.mean(list(loss_deltas.values())):+.4f}")
        print(f"    Mean Δloss (random dirs): "
              f"{np.mean(random_loss_deltas):+.4f}")

        all_results[cond_name] = {
            "logit_matrix": logit_matrix.tolist(),
            "topic_names": topic_names_ordered,
            "diag_mean": diag_mean,
            "off_diag_mean": off_diag_mean,
            "diag_enrichment": diag_enrichment,
            "diag_fraction": diag_frac,
            "uniform_fraction": uniform_frac,
            "loss_deltas_topic": loss_deltas,
            "response_norms_topic": response_norms,
            "random_logit_matrix": random_logit_matrix.tolist(),
            "random_loss_deltas": random_loss_deltas,
            "random_topic_variance": random_topic_var,
        }

    # =========================================================
    # Summary
    # =========================================================
    print(f"\n{'='*60}")
    print("SUMMARY: Topic perturbation discrimination")
    print(f"{'='*60}")

    print(f"\nTopics: {topic_names_ordered}")
    print(f"Direction overlap (mean off-diag cosine): {mean_off_diag:.4f}")
    print(f"Perturbation strength: s={s_test}, scale={s_test * b1_std:.4f}")

    print(f"\n{'Condition':>8s}  {'Diag mean':>10s}  {'Off-diag':>10s}  "
          f"{'Enrichment':>11s}  {'Diag frac':>10s}  "
          f"{'Mean Δloss':>10s}  {'Rand Δloss':>10s}")
    print("-" * 80)
    for cond_name in conditions:
        r = all_results[cond_name]
        print(f"{cond_name:>8s}  {r['diag_mean']:>+10.5f}  "
              f"{r['off_diag_mean']:>+10.5f}  "
              f"{r['diag_enrichment']:>+11.5f}  "
              f"{r['diag_fraction']:>10.4f}  "
              f"{np.mean(list(r['loss_deltas_topic'].values())):>+10.4f}  "
              f"{np.mean(r['random_loss_deltas']):>+10.4f}")

    print(f"\n  Uniform diagonal fraction: {uniform_frac:.4f}")

    # CL vs OL comparison
    cl_enrich = all_results["CL+M"]["diag_enrichment"]
    ol_enrich = all_results["OL"]["diag_enrichment"]
    print(f"\n  CL+M diagonal enrichment: {cl_enrich:+.5f}")
    print(f"  OL   diagonal enrichment: {ol_enrich:+.5f}")
    if abs(ol_enrich) > 1e-8:
        print(f"  CL+M / OL ratio: {cl_enrich / ol_enrich:.3f}")

    # Robustness gap replication
    ol_loss = np.mean(list(all_results["OL"]["loss_deltas_topic"].values()))
    cl_loss = np.mean(list(all_results["CL+M"]["loss_deltas_topic"].values()))
    if abs(cl_loss) > 1e-8:
        print(f"\n  Robustness gap (topic dirs): OL/CL+M Δloss ratio = "
              f"{ol_loss / cl_loss:.2f}×")

    # Per-topic logit matrices
    for cond_name in conditions:
        r = all_results[cond_name]
        print(f"\n  [{cond_name}] Logit boost matrix "
              f"(rows=perturb, cols=measure):")
        header = "  " + " " * 14 + " ".join(
            f"{t:>12s}" for t in topic_names_ordered)
        print(header)
        for pi, pt in enumerate(topic_names_ordered):
            row = " ".join(
                f"{r['logit_matrix'][pi][mi]:>+12.5f}"
                for mi in range(n_topics))
            print(f"  {pt:>12s}  {row}")

    # =========================================================
    # Save results
    # =========================================================
    if ckpt_source == "extended":
        save_dir = os.path.join(ckpt_root, f"mirror_test_v3_step{ckpt_step}")
    else:
        save_dir = os.path.join(ckpt_root, "mirror_test_v3")
    os.makedirs(save_dir, exist_ok=True)

    output = {
        "config": {
            "n_tokens": n_tokens, "n_eval_batches": n_eval_batches,
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "fwd_n_layer": fwd_n_layer, "seed": seed,
            "s_test": s_test, "b1_std": b1_std,
            "n_random_dirs": n_random_dirs,
            "min_keyword_hits": MIN_KEYWORD_HITS,
        },
        "topic_directions": {
            "topics": topic_names_ordered,
            "counts": {t: topic_counts[t] for t in topic_names_ordered},
            "cosine_overlap": cosine_overlap.numpy().tolist(),
            "mean_off_diag_cosine": mean_off_diag,
        },
        "topic_token_ids": {t: topic_token_ids[t] for t in topic_names_ordered},
        "results": all_results,
    }

    from language_reduction.shared import NumpyEncoder
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    volume.commit()
    print(f"\nResults saved to {save_dir}")

    return output
