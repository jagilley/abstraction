from language_reduction.shared import app, volume, DATA_DIR, image, NumpyEncoder


# ---------------------------------------------------------------------------
# Stage 10: Contextual embedding evaluation
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=32768,
)
def contextual_embedding_eval(taus: str = "0.0,0.3", n_batches: int = 500):
    """Extract context-averaged hidden states and run embedding eval.

    For each tau, loads the P=100M model and its training data, runs
    n_batches of forward passes capturing per-layer hidden states,
    averages per token, then runs the full eval battery.
    """
    import os, json, glob
    import torch
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.eval_embeddings import (
        build_token_index, vocab_audit, run_eval,
    )

    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)

    tau_list = [float(t) for t in taus.split(",")]
    batch_size = 64
    block_size = 128

    # Collect results: tau -> {static, layer1, layer2} -> eval results
    results = {}

    for tau in tau_list:
        tau_key = f"tau_{tau:.3f}"
        model_dir = f"{DATA_DIR}/models/{tau_key}/P_100000000/T_128"
        if not os.path.exists(model_dir):
            print(f"No P=100M model for {tau_key}")
            continue

        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]
        n_layer = cfg.get("n_layer", 2)
        n_embd = cfg.get("n_embd", 128)

        model = GPT(vocab_size=vocab_size, block_size=block_size,
                     n_layer=n_layer, n_head=cfg.get("n_head", 4),
                     n_embd=n_embd)
        model.load_state_dict(sd)
        model.eval()

        # Static embeddings
        static_emb = model.transformer.wte.weight.detach().numpy()

        # Load corpus (each model runs on its own training distribution)
        if tau == 0.0:
            data_dir = f"{DATA_DIR}/tokens"
        else:
            data_dir = f"{DATA_DIR}/denoised/tau_{tau:.3f}"

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

        # Accumulate per-token hidden states
        layer_sums = [np.zeros((vocab_size, n_embd), dtype=np.float64)
                      for _ in range(n_layer)]
        token_counts = np.zeros(vocab_size, dtype=np.int64)

        print(f"\n--- Extracting contextual embeddings: {tau_key} ({n_batches} batches) ---")
        with torch.no_grad():
            for bi in range(n_batches):
                ix = torch.randint(len(data) - block_size - 1, (batch_size,))
                x = torch.stack([data[i:i+block_size] for i in ix])

                # manual forward to capture per-layer outputs
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

        active_count = int((token_counts[top_ids] > 0).sum())
        print(f"  Tokens processed: {int(token_counts.sum()):,}")
        print(f"  Active tokens with >0 occurrences: {active_count}/{len(top_ids)}")

        # Compute averages
        has_data = token_counts > 0
        ctx_embs = []
        for l in range(n_layer):
            avg = np.zeros_like(layer_sums[l])
            avg[has_data] = layer_sums[l][has_data] / token_counts[has_data, None]
            ctx_embs.append(avg)

        # Run evals: static + each layer
        results[tau_key] = {}
        for label, emb in [("static", static_emb)] + \
                           [(f"layer{l+1}", ctx_embs[l]) for l in range(n_layer)]:
            print(f"\n  [{tau_key}] {label}:")
            r = run_eval(emb, token_index)
            results[tau_key][label] = r

            a = r["analogies"].get("overall", {})
            s = r["similarity"]
            g = r["geometry"]
            coh = r["nearest_neighbors"].get("_mean_coherence")
            print(f"    Analogies:  {a.get('acc_add', 'n/a'):.3f} ({a.get('total', 0)} pairs)"
                  if a.get("acc_add") is not None else f"    Analogies: n/a")
            print(f"    Similarity: rho={s.get('spearman_rho', 'n/a')}")
            print(f"    Eff. rank:  {g['effective_rank']:.1f}  |  Avg cos: {g['avg_cosine']:.4f}")
            if coh is not None:
                print(f"    Coherence:  {coh:.4f}")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("CONTEXTUAL EMBEDDING COMPARISON (P=100M)")
    print(f"{'='*80}")

    layers = ["static", "layer1", "layer2"]
    tau_keys = sorted(results.keys())

    col_headers = [f"{tk}/{l}" for tk in tau_keys for l in layers]
    header = f"{'Metric':>20s}" + "".join(f"  {h:>14s}" for h in col_headers)
    print(header)
    print("-" * len(header))

    def _val(tk, l, *path):
        d = results.get(tk, {}).get(l, {})
        for p in path:
            if isinstance(d, dict):
                d = d.get(p)
            else:
                return None
        return d

    rows = [
        ("Analogy (3CosAdd)", lambda tk, l: _val(tk, l, "analogies", "overall", "acc_add")),
        ("Similarity (ρ)", lambda tk, l: _val(tk, l, "similarity", "spearman_rho")),
        ("Eff. rank", lambda tk, l: _val(tk, l, "geometry", "effective_rank")),
        ("Avg cosine", lambda tk, l: _val(tk, l, "geometry", "avg_cosine")),
        ("Top-10 energy", lambda tk, l: _val(tk, l, "geometry", "top10_energy")),
        ("NN coherence", lambda tk, l: _val(tk, l, "nearest_neighbors", "_mean_coherence")),
        ("Clustering ratio", lambda tk, l: _val(tk, l, "clustering", "overall", "mean_ratio")),
    ]

    for label, fn in rows:
        row = f"{label:>20s}"
        for tk in tau_keys:
            for l in layers:
                v = fn(tk, l)
                row += f"  {v:14.4f}" if v is not None else f"  {'n/a':>14s}"
        print(row)

    # Save
    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/contextual_embedding_eval.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return results


# ---------------------------------------------------------------------------
# Geometric analogy test (no argmax, uses directional projections)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def geometric_analogy():
    """Test king-man+woman=queen using embedding geometry, not vocabulary lookup.

    Extracts a gender direction from known pairs, then projects the analogy
    query onto it. If the concept of 'female royalty' exists in the space,
    the query lands on the female side with king-like status magnitude.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    enc = tiktoken.get_encoding("gpt2")
    str_to_id = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in str_to_id:
            str_to_id[cleaned] = tid

    gender_pairs = [
        ("man", "woman"), ("he", "she"), ("his", "her"),
        ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
        ("husband", "wife"), ("boy", "girl"), ("uncle", "aunt"),
    ]

    reference_words = {
        "male": ["man", "he", "his", "father", "son", "king", "boy"],
        "female": ["woman", "she", "her", "mother", "daughter", "wife", "girl"],
        "status": ["king", "president", "lord"],
        "common": ["man", "person", "people"],
    }

    for tau in [0.0, 0.3]:
        tau_key = f"tau_{tau:.3f}"
        model_dir = f"{DATA_DIR}/models/{tau_key}/P_100000000/T_128"

        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]
        model = GPT(vocab_size=vocab_size, block_size=cfg.get("block_size", 128),
                     n_layer=cfg.get("n_layer", 2), n_head=cfg.get("n_head", 4),
                     n_embd=cfg.get("n_embd", 128))
        model.load_state_dict(sd)
        model.eval()
        emb = model.transformer.wte.weight.detach().numpy()

        def v(word):
            return emb[str_to_id[word]]

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

        # 1. Compute gender direction from multiple pairs
        diffs = [v(f) - v(m) for m, f in gender_pairs]
        d_gender = np.mean(diffs, axis=0)
        d_gender = d_gender / np.linalg.norm(d_gender)

        # 2. Compute the analogy query
        query = v("king") - v("man") + v("woman")

        # 3. Project onto gender direction
        def proj(vec):
            return float(np.dot(vec, d_gender))

        print(f"\n{'='*65}")
        print(f"  {tau_key} (P=100M)")
        print(f"{'='*65}")

        print(f"\n  Gender direction (male→female) projections:")
        print(f"    {'word':>15s}  {'projection':>10s}  {'side':>8s}")
        print(f"    {'-'*38}")
        test_words = ["man", "woman", "he", "she", "king", "father", "mother",
                      "wife", "husband", "boy", "girl"]
        for w in test_words:
            p = proj(v(w))
            side = "female" if p > 0 else "male"
            print(f"    {w:>15s}  {p:>10.4f}  {side:>8s}")

        q_proj = proj(query)
        q_side = "female" if q_proj > 0 else "male"
        print(f"    {'-'*38}")
        print(f"    {'king-man+woman':>15s}  {q_proj:>10.4f}  {q_side:>8s}")

        # 4. Decompose query into gender + residual
        q_gender_component = np.dot(query, d_gender) * d_gender
        q_residual = query - q_gender_component
        king_residual = v("king") - np.dot(v("king"), d_gender) * d_gender

        print(f"\n  Decomposition of king-man+woman:")
        print(f"    Gender projection:      {proj(query):+.4f} "
              f"(woman: {proj(v('woman')):+.4f}, she: {proj(v('she')):+.4f})")
        print(f"    Residual cos(king):     {cos(q_residual, king_residual):.4f}")
        print(f"    Residual cos(man):      {cos(q_residual, v('man') - np.dot(v('man'), d_gender) * d_gender):.4f}")

        # 5. How well-separated are male/female clusters along this direction?
        male_projs = [proj(v(w)) for w in reference_words["male"] if w in str_to_id]
        female_projs = [proj(v(w)) for w in reference_words["female"] if w in str_to_id]
        m_mean, f_mean = np.mean(male_projs), np.mean(female_projs)
        m_std, f_std = np.std(male_projs), np.std(female_projs)
        separation = (f_mean - m_mean) / (m_std + f_std + 1e-10)
        print(f"\n  Gender direction quality:")
        print(f"    Male mean:   {m_mean:+.4f} (std {m_std:.4f})")
        print(f"    Female mean: {f_mean:+.4f} (std {f_std:.4f})")
        print(f"    Separation (d'):  {separation:.2f}")

        # 6. Cosine of the full query with key reference points
        print(f"\n  cos(king-man+woman, x):")
        ref_words = ["woman", "mother", "wife", "daughter", "she",
                     "king", "man", "father", "he", "president"]
        for w in ref_words:
            print(f"    {w:>15s}: {cos(query, v(w)):.4f}")


# ---------------------------------------------------------------------------
# Inject reconstructed vector and generate
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def inject_and_generate():
    """Inject king-man+woman into the model and see what it generates.

    Reconstructs a 'queen' vector via analogy in embedding space, injects it
    at a position in a short prefix, and runs autoregressive generation.
    If the concept of female-royalty exists in the model's geometry,
    the generated text should reflect it.
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    enc = tiktoken.get_encoding("gpt2")

    str_to_id = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in str_to_id:
            str_to_id[cleaned] = tid

    def run_injection(model, inject_vec, prefix_text, n_gen=30, label=""):
        """Generate from prefix + injected vector."""
        prefix_ids = enc.encode(prefix_text)
        inject_pos = len(prefix_ids)

        # sequence: [prefix_tokens..., dummy_at_inject_pos]
        ids = prefix_ids + [0]

        with torch.no_grad():
            for _ in range(n_gen):
                x = torch.tensor([ids], dtype=torch.long)
                T = x.size(1)
                if T >= model.block_size:
                    break

                tok_emb = model.transformer.wte(x)
                tok_emb[0, inject_pos] = inject_vec

                if not model.use_rope:
                    pos = torch.arange(T)
                    h = model.transformer.drop(tok_emb + model.transformer.wpe(pos))
                else:
                    h = model.transformer.drop(tok_emb)

                for block in model.transformer.h:
                    h = block(h)
                h = model.transformer.ln_f(h)
                logits = model.lm_head(h)

                next_tok = int(logits[0, -1].argmax())
                ids.append(next_tok)

        # Also get top-10 predictions right after the injected position
        # (what does the model think follows "The [concept]"?)
        x = torch.tensor([prefix_ids + [0]], dtype=torch.long)
        T = x.size(1)
        tok_emb = model.transformer.wte(x)
        tok_emb[0, inject_pos] = inject_vec
        if not model.use_rope:
            pos = torch.arange(T)
            h = model.transformer.drop(tok_emb + model.transformer.wpe(pos))
        else:
            h = model.transformer.drop(tok_emb)
        for block in model.transformer.h:
            h = block(h)
        h = model.transformer.ln_f(h)
        logits = model.lm_head(h)

        probs = torch.softmax(logits[0, inject_pos], dim=-1)
        top10 = torch.topk(probs, 10)

        gen_text = enc.decode(ids[inject_pos + 1:])
        print(f"\n    {label}:")
        print(f"      Generated: {prefix_text}[*]{gen_text}")
        print(f"      Top-10 next tokens after [*]:")
        for prob, tid in zip(top10.values, top10.indices):
            tok_str = enc.decode([int(tid)])
            print(f"        {tok_str!r:15s} {float(prob):.4f}")

    for tau in [0.0, 0.3]:
        tau_key = f"tau_{tau:.3f}"
        model_dir = f"{DATA_DIR}/models/{tau_key}/P_100000000/T_128"

        with open(os.path.join(model_dir, "results.json")) as f:
            cfg = json.load(f)
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]
        model = GPT(vocab_size=vocab_size, block_size=cfg.get("block_size", 128),
                     n_layer=cfg.get("n_layer", 2), n_head=cfg.get("n_head", 4),
                     n_embd=cfg.get("n_embd", 128))
        model.load_state_dict(sd)
        model.eval()

        wte = model.transformer.wte.weight.detach()
        queen_vec = wte[str_to_id["king"]] - wte[str_to_id["man"]] + wte[str_to_id["woman"]]

        print(f"\n{'='*65}")
        print(f"  {tau_key} (P=100M)")
        print(f"{'='*65}")

        prefix = "The"
        run_injection(model, queen_vec, prefix, label="queen_vec (king-man+woman)")
        run_injection(model, wte[str_to_id["king"]], prefix, label="king (baseline)")
        run_injection(model, wte[str_to_id["woman"]], prefix, label="woman (baseline)")


# ---------------------------------------------------------------------------
# Analogy category breakdown (reads saved results)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=300,
    memory=8192,
)
def analogy_breakdown():
    """Print per-category analogy accuracy from saved embedding_eval results."""
    import json

    with open(f"{DATA_DIR}/results/embedding_eval.json") as f:
        data = json.load(f)

    evals = data["evaluations"]
    tau_keys = sorted(evals.keys())
    p_keys = sorted(
        {pk for tk in tau_keys for pk in evals[tk]},
        key=lambda x: int(x.split("_")[1]),
    )

    # Get category names from any result
    sample = next(iter(next(iter(evals.values())).values()))
    categories = [c for c in sample["analogies"] if c != "overall"]

    for pk in p_keys:
        P = int(pk.split("_")[1])
        print(f"\n{'='*60}")
        print(f"P = {P:,}")
        print(f"{'='*60}")
        header = f"{'category':>20s}" + "".join(f"  {tk:>12s}" for tk in tau_keys) + f"  {'delta':>8s}"
        print(header)
        print("-" * len(header))

        for cat in categories:
            row = f"{cat:>20s}"
            vals = []
            for tk in tau_keys:
                r = evals.get(tk, {}).get(pk, {}).get("analogies", {}).get(cat, {})
                acc = r.get("acc_add")
                n = r.get("total", 0)
                if acc is not None:
                    row += f"  {acc:8.3f} ({n:2d})"
                    vals.append(acc)
                else:
                    row += f"  {'n/a':>12s}"
                    vals.append(None)
            if len(vals) == 2 and all(v is not None for v in vals):
                delta = vals[1] - vals[0]
                row += f"  {delta:+8.3f}"
            print(row)

        # overall
        row = f"{'OVERALL':>20s}"
        vals = []
        for tk in tau_keys:
            r = evals.get(tk, {}).get(pk, {}).get("analogies", {}).get("overall", {})
            acc = r.get("acc_add")
            n = r.get("total", 0)
            if acc is not None:
                row += f"  {acc:8.3f} ({n:2d})"
                vals.append(acc)
            else:
                row += f"  {'n/a':>12s}"
                vals.append(None)
        if len(vals) == 2 and all(v is not None for v in vals):
            delta = vals[1] - vals[0]
            row += f"  {delta:+8.3f}"
        print("-" * len(header))
        print(row)


# ---------------------------------------------------------------------------
# Quick analogy spot-check
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def analogy_spot_check(a: str = "king", b: str = "man", c: str = "woman", k: int = 10):
    """Run a single analogy (a - b + c = ?) on P=100M models for both tau."""
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    enc = tiktoken.get_encoding("gpt2")
    top_ids = np.load(f"{DATA_DIR}/stats/top_ids.npy")
    active_set = set(top_ids.tolist())

    # build string -> token_id lookup
    str_to_id = {}
    id_to_str = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        id_to_str[tid] = raw
        if cleaned and cleaned not in str_to_id:
            str_to_id[cleaned] = tid

    ai, bi, ci = str_to_id[a], str_to_id[b], str_to_id[c]

    for tau in [0.0, 0.3]:
        tau_key = f"tau_{tau:.3f}"
        model_dir = f"{DATA_DIR}/models/{tau_key}/P_100000000/T_128"
        cfg_path = os.path.join(model_dir, "results.json")
        model_path = os.path.join(model_dir, "model.pt")

        with open(cfg_path) as f:
            cfg = json.load(f)
        sd = torch.load(model_path, map_location="cpu")
        vocab_size = sd["transformer.wte.weight"].shape[0]
        model = GPT(vocab_size=vocab_size, block_size=cfg.get("block_size", 128),
                     n_layer=cfg.get("n_layer", 2), n_head=cfg.get("n_head", 4),
                     n_embd=cfg.get("n_embd", 128))
        model.load_state_dict(sd)
        model.eval()

        emb = model.transformer.wte.weight.detach().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_n = emb / (norms + 1e-10)

        # a - b + c
        query = emb_n[ai] - emb_n[bi] + emb_n[ci]
        query = query / (np.linalg.norm(query) + 1e-10)
        sims = emb_n @ query

        # mask inactive + input words
        for idx in range(len(sims)):
            if idx not in active_set or idx in {ai, bi, ci}:
                sims[idx] = -np.inf

        top_k = np.argsort(sims)[::-1][:k]
        print(f"\n=== {tau_key} (P=100M): {a} - {b} + {c} = ? ===")
        for rank, idx in enumerate(top_k):
            word = id_to_str[idx].strip()
            print(f"  {rank+1:2d}. {word:20s} (cos={sims[idx]:.4f})")


# ---------------------------------------------------------------------------
# Stage 9: Embedding geometry evaluation
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=1800,
    memory=16384,
)
def embedding_eval(taus: str = "0.0,0.3"):
    """Evaluate embedding geometry across tau values and model sizes."""
    import os, json, glob
    import torch
    import numpy as np
    from language_reduction.model import GPT
    from language_reduction.eval_embeddings import (
        build_token_index, vocab_audit, run_eval, format_comparison,
    )

    stats_dir = f"{DATA_DIR}/stats"
    top_ids = np.load(f"{stats_dir}/top_ids.npy")
    token_index = build_token_index(vocab_size=50257, active_ids=top_ids)
    audit = vocab_audit(token_index)

    tau_list = [float(t) for t in taus.split(",")]
    all_results = {"vocab_audit": audit, "evaluations": {}}

    for tau in tau_list:
        tau_key = f"tau_{tau:.3f}"
        models_dir = f"{DATA_DIR}/models/{tau_key}"
        if not os.path.exists(models_dir):
            print(f"No models for {tau_key}")
            continue

        all_results["evaluations"][tau_key] = {}

        for p_dir in sorted(glob.glob(os.path.join(models_dir, "P_*"))):
            for t_dir in sorted(glob.glob(os.path.join(p_dir, "T_*"))):
                model_path = os.path.join(t_dir, "model.pt")
                config_path = os.path.join(t_dir, "results.json")
                if not os.path.exists(model_path):
                    continue

                with open(config_path) as f:
                    cfg = json.load(f)

                n_tokens = cfg["n_tokens"]
                state_dict = torch.load(model_path, map_location="cpu")
                vocab_size = state_dict["transformer.wte.weight"].shape[0]
                model = GPT(
                    vocab_size=vocab_size,
                    block_size=cfg.get("block_size", 128),
                    n_layer=cfg.get("n_layer", 2),
                    n_head=cfg.get("n_head", 4),
                    n_embd=cfg.get("n_embd", 128),
                )
                model.load_state_dict(state_dict)
                model.eval()

                emb = model.transformer.wte.weight.detach().numpy()
                print(f"\n--- {tau_key}, P={n_tokens:,} (vocab={vocab_size}) ---")

                result = run_eval(emb, token_index)
                p_key = f"P_{n_tokens}"
                all_results["evaluations"][tau_key][p_key] = result

                a = result["analogies"].get("overall", {})
                s = result["similarity"]
                g = result["geometry"]
                cl = result["clustering"].get("overall", {})
                print(f"  Analogies: {a.get('acc_add', 'n/a')} ({a.get('total', 0)} pairs)")
                print(f"  Similarity: rho={s.get('spearman_rho', 'n/a')} ({s.get('n_pairs', 0)} pairs)")
                print(f"  Geometry: eff_rank={g['effective_rank']:.1f}, "
                      f"avg_cos={g['avg_cosine']:.4f}, top10={g['top10_energy']:.3f}")
                if cl:
                    print(f"  Clustering: ratio={cl['mean_ratio']:.2f}")
                nn_coh = result["nearest_neighbors"].get("_mean_coherence")
                if nn_coh is not None:
                    print(f"  NN coherence: {nn_coh:.4f}")

    # Print comparison
    comparison = format_comparison(all_results)
    print(f"\n\n{'='*70}")
    print(comparison)

    # Save
    results_dir = f"{DATA_DIR}/results"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/embedding_eval.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    return all_results


# ---------------------------------------------------------------------------
# Recovery coherence check (neighborhoods + analogy on fine-tuned model)
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def recovery_coherence(source_tau: float = 0.3, source_P: int = 100_000_000):
    """Check internal coherence of the fine-tuned model's geometry.

    Reports: king-man+woman top-10, and nearest neighbors for key words,
    across all three models (start, fine-tuned, ground truth).
    """
    import os, json, torch
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT

    enc = tiktoken.get_encoding("gpt2")
    top_ids = np.load(f"{DATA_DIR}/stats/top_ids.npy")
    active_set = set(top_ids.tolist())

    str_to_id = {}
    id_to_str = {}
    for tid in range(50257):
        raw = enc.decode([tid])
        cleaned = raw.strip().lower()
        id_to_str[tid] = raw
        if cleaned and cleaned not in str_to_id:
            str_to_id[cleaned] = tid

    block_size = 128
    models = {
        f"start (tau={source_tau})": f"{DATA_DIR}/models/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "fine-tuned": f"{DATA_DIR}/models/finetune/tau_{source_tau:.3f}/P_{source_P}/T_{block_size}",
        "ground truth (tau=0.0)": f"{DATA_DIR}/models/tau_0.000/P_{source_P}/T_{block_size}",
    }

    for label, model_dir in models.items():
        sd = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        emb = sd["transformer.wte.weight"].numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb_n = emb / (norms + 1e-10)

        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")

        # Analogy: king - man + woman
        ai, bi, ci = str_to_id["king"], str_to_id["man"], str_to_id["woman"]
        query = emb_n[ai] - emb_n[bi] + emb_n[ci]
        query = query / (np.linalg.norm(query) + 1e-10)
        sims = emb_n @ query
        for idx in range(len(sims)):
            if idx not in active_set or idx in {ai, bi, ci}:
                sims[idx] = -np.inf
        top_k = np.argsort(sims)[::-1][:10]

        print(f"\n  king - man + woman = ?")
        for rank, idx in enumerate(top_k):
            word = id_to_str[idx].strip()
            print(f"    {rank+1:2d}. {word:20s} (cos={sims[idx]:.4f})")

        # Nearest neighbors for key words
        nn_words = ["queen", "king", "woman", "she", "princess", "wife", "mother"]
        for word in nn_words:
            if word not in str_to_id:
                continue
            wi = str_to_id[word]
            sims_w = emb_n @ emb_n[wi]
            sims_w[wi] = -np.inf
            for idx in range(len(sims_w)):
                if idx not in active_set:
                    sims_w[idx] = -np.inf
            top_k = np.argsort(sims_w)[::-1][:10]
            neighbors = [f"{id_to_str[idx].strip()}({sims_w[idx]:.3f})" for idx in top_k]
            print(f"\n  NN({word}): {', '.join(neighbors[:7])}")

    volume.commit()
