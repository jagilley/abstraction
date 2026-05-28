"""Idea 1: probe downstream layers for *named* novelty structure.

The closed-loop self-map result (CLOSED_LOOP_README) showed only that the
closed-loop model encodes more *variance* of the raw 256-d residual vector
(R²=0.44 vs 0.28). "More variance" is not "more interpretable". This stage asks
a sharper question: do the downstream layers linearly encode *nameable*
categories of novelty — "this position is a before-closer / sentence-start /
high-residual position" — and does closing the loop make those categories more
decodable?

Design
------
- Probe targets:
    * Syntactic categories (token-derived ground truth, NON-circular):
      before_closer, after_opener, sentence_start, after_punct. These are
      exactly the highest-|Cohen's d| residual categories from the behavioral
      residual analysis (before_closer +0.84, sentence_start -0.85).
      -> logistic probe, report AUC.
    * residual_norm: "how novel is this position" (the novelty magnitude).
    * block1_contrib_norm: control (computational magnitude, not novelty).
      -> linear probe, report R² + Pearson r.
- Probe inputs: post_block1 / post_block2 / post_block3 activations, taken from
  the NO-INJECTION forward pass for BOTH models, so inputs and the compute path
  are identical and we isolate the trained weights.
- Layerwise localization: the injection enters after block 1. If closing the
  loop builds the self-map, the open-vs-closed gap should be ~0 at post_block1
  (computed before injection) and grow at blocks 2-3 (downstream of injection).
  A uniform lift across all layers would instead point at the lr confound.

CAVEAT: the existing open-loop (lr=1e-4) and closed-loop (lr=3e-4) checkpoints
differ in learning rate. This run is exploratory; the layerwise localization is
the main defense against that confound. A controlled retrain (identical lr/seed)
is the prerequisite for a publishable version.
"""

import json
import math

from language_reduction.shared import app, volume, DATA_DIR


PUNCT_CHARS = set(".,;:!?")
OPENER_CHARS = set("([{\"'`")
CLOSER_CHARS = set(")]}")


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=3600,
    memory=32768,
)
def a2a_novelty_probe(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    n_eval_batches: int = 40,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    probe_steps: int = 800,
    seed: int = 0,
):
    import os
    import glob
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import tiktoken
    from language_reduction.model import GPT
    from language_reduction.experiments.a2a_forward.forward_model import (
        TransformerForwardModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"A2A novelty probe on {device}")
    enc = tiktoken.get_encoding("gpt2")

    # --- Load data ---
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"), allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens = []
    total = 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_tokens)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    split = int(0.9 * len(data))
    val_data = data[split:]

    # --- Fixed set of batches (identical inputs for both models) ---
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_eval_batches):
        ix = torch.randint(len(val_data) - block_size - 1, (batch_size,), generator=g)
        x = torch.stack([val_data[i:i + block_size] for i in ix])
        batches.append(x)

    # =============================================
    # Syntactic ground-truth labels (model-independent, computed once)
    # =============================================
    def classify_syntactic(token_ids_np):
        """Per-token syntactic category masks, shape (B*T,)."""
        B, T = token_ids_np.shape
        n = B * T
        out = {k: np.zeros(n, dtype=bool) for k in
               ["after_punct", "after_opener", "before_closer", "sentence_start"]}
        for b in range(B):
            seq = token_ids_np[b]
            decoded = [enc.decode([int(t)]) for t in seq]
            base = b * T
            for t in range(T):
                prev_text = decoded[t - 1] if t > 0 else ""
                next_text = decoded[t + 1] if t < T - 1 else ""
                prev_s = prev_text.strip()
                i = base + t
                if prev_s and prev_s[-1] in PUNCT_CHARS:
                    out["after_punct"][i] = True
                    if prev_s[-1] in ".!?" or "\n" in prev_text:
                        out["sentence_start"][i] = True
                if prev_s and prev_s[-1] in OPENER_CHARS:
                    out["after_opener"][i] = True
                next_s = next_text.strip()
                if next_s and next_s[0] in CLOSER_CHARS:
                    out["before_closer"][i] = True
        return out

    print("Computing syntactic labels...")
    syn_labels = {k: [] for k in
                  ["after_punct", "after_opener", "before_closer", "sentence_start"]}
    for x in batches:
        info = classify_syntactic(x.numpy())
        for k in syn_labels:
            syn_labels[k].append(info[k])
    syn_labels = {k: np.concatenate(v) for k, v in syn_labels.items()}
    for k, v in syn_labels.items():
        print(f"  {k}: base rate {v.mean():.4f} ({int(v.sum()):,} / {len(v):,})")

    # =============================================
    # Per-model activation collection (no injection, identical inputs)
    # =============================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_dirs = {
        "open_loop": (f"{DATA_DIR}/a2a_forward/transformer_L{fwd_n_layer}/"
                      f"{gap_tag}/P_{n_tokens}"),
        "closed_loop": (f"{DATA_DIR}/a2a_forward/loop_L{fwd_n_layer}/"
                        f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}"),
    }
    layer_keys = ["post_block1", "post_block2", "post_block3"]

    def collect(model_dir):
        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(
            os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True))
        model.eval()

        fwd_model = TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
        ).to(device)
        fwd_model.load_state_dict(torch.load(
            os.path.join(model_dir, "fwd_model.pt"), map_location=device,
            weights_only=True))
        fwd_model.eval()

        acts = {k: [] for k in layer_keys}
        res_norm, b1_contrib = [], []
        cos_acc = []
        with torch.no_grad():
            for x in batches:
                x = x.to(device)
                _, _, inter = model(x, return_intermediates=True)
                for k in layer_keys:
                    acts[k].append(inter[k].reshape(-1, n_embd).cpu())
                pred = fwd_model(inter[predict_from])
                actual = inter[predict_to]
                residual = actual - pred
                res_norm.append(residual.norm(dim=-1).reshape(-1).cpu())
                cos_acc.append(
                    F.cosine_similarity(pred, actual, dim=-1).reshape(-1).cpu())
                b1_contrib.append(
                    (inter["post_block1"] - inter["post_block0"]).norm(dim=-1)
                    .reshape(-1).cpu())
        acts = {k: torch.cat(v).numpy() for k, v in acts.items()}
        return {
            "acts": acts,
            "residual_norm": torch.cat(res_norm).numpy(),
            "block1_contrib_norm": torch.cat(b1_contrib).numpy(),
            "fwd_cosine": float(torch.cat(cos_acc).mean()),
        }

    collected = {}
    for name, mdir in model_dirs.items():
        print(f"\nCollecting {name} from {mdir}")
        collected[name] = collect(mdir)
        print(f"  forward model cosine (no inj): {collected[name]['fwd_cosine']:.4f}")

    # =============================================
    # Probes
    # =============================================
    n = len(syn_labels["before_closer"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    def standardize(X):
        mu = X[train_idx].mean(axis=0, keepdims=True)
        sd = X[train_idx].std(axis=0, keepdims=True) + 1e-6
        return (X - mu) / sd

    def auc_score(scores, labels):
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = float(labels.sum())
        n_neg = float(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                     / (n_pos * n_neg))

    def logistic_probe(X, y):
        Xs = standardize(X)
        Xtr = torch.from_numpy(Xs[train_idx]).float().to(device)
        ytr = torch.from_numpy(y[train_idx].astype(np.float32)).to(device)
        Xte = torch.from_numpy(Xs[test_idx]).float().to(device)
        probe = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        pos_weight = torch.tensor(
            [(1 - ytr.mean()) / (ytr.mean() + 1e-6)], device=device)
        lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            idx = torch.randint(n_train, (bs,), device=device)
            logit = probe(Xtr[idx]).squeeze(-1)
            loss = lossfn(logit, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores = probe(Xte).squeeze(-1).cpu().numpy()
        return auc_score(scores, y[test_idx].astype(int))

    def linear_probe(X, target):
        Xs = standardize(X)
        Xtr = torch.from_numpy(Xs[train_idx]).float().to(device)
        Xte = torch.from_numpy(Xs[test_idx]).float().to(device)
        t = target.astype(np.float32)
        tmu, tsd = t[train_idx].mean(), t[train_idx].std() + 1e-6
        ttr = torch.from_numpy((t[train_idx] - tmu) / tsd).to(device)
        tte_np = (t[test_idx] - tmu) / tsd
        tte = torch.from_numpy(tte_np).to(device)
        probe = nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        bs = min(4096, n_train)
        for _ in range(probe_steps):
            idx = torch.randint(n_train, (bs,), device=device)
            pred = probe(Xtr[idx]).squeeze(-1)
            loss = F.mse_loss(pred, ttr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = probe(Xte).squeeze(-1).cpu().numpy()
        var = tte_np.var()
        r2 = float(1.0 - ((pred - tte_np) ** 2).mean() / var) if var > 0 else 0.0
        r = float(np.corrcoef(pred, tte_np)[0, 1])
        return {"r2": r2, "pearson": r}

    syn_cats = ["before_closer", "sentence_start", "after_opener", "after_punct"]
    results = {"fwd_cosine": {m: collected[m]["fwd_cosine"] for m in collected},
               "base_rates": {k: float(syn_labels[k].mean()) for k in syn_cats},
               "auc": {}, "scalar": {}}

    print("\n=== Logistic probes (AUC): syntactic novelty categories ===")
    print(f"{'category':>16s} {'layer':>12s} {'open':>7s} {'closed':>7s} {'Δ':>7s}")
    for cat in syn_cats:
        results["auc"][cat] = {}
        for lk in layer_keys:
            a_open = logistic_probe(collected["open_loop"]["acts"][lk], syn_labels[cat])
            a_closed = logistic_probe(
                collected["closed_loop"]["acts"][lk], syn_labels[cat])
            results["auc"][cat][lk] = {
                "open_loop": a_open, "closed_loop": a_closed,
                "delta": a_closed - a_open}
            print(f"{cat:>16s} {lk:>12s} {a_open:>7.4f} {a_closed:>7.4f} "
                  f"{a_closed - a_open:>+7.4f}")

    print("\n=== Linear probes (R²): novelty magnitude & control ===")
    print(f"{'target':>20s} {'layer':>12s} {'open R²':>8s} {'closed R²':>10s} {'Δ':>7s}")
    for tgt in ["residual_norm", "block1_contrib_norm"]:
        results["scalar"][tgt] = {}
        for lk in layer_keys:
            o = linear_probe(collected["open_loop"]["acts"][lk],
                             collected["open_loop"][tgt])
            c = linear_probe(collected["closed_loop"]["acts"][lk],
                             collected["closed_loop"][tgt])
            results["scalar"][tgt][lk] = {
                "open_loop": o, "closed_loop": c,
                "delta_r2": c["r2"] - o["r2"]}
            print(f"{tgt:>20s} {lk:>12s} {o['r2']:>8.4f} {c['r2']:>10.4f} "
                  f"{c['r2'] - o['r2']:>+7.4f}")

    # --- Save ---
    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "novelty_probe_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {save_path}")
    return results
