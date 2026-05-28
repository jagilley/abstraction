"""Enriched Test 3: injection help organized by residual STRUCTURE, not norm.

Test 3 (injection_help.py) measured per-token help = loss_no_inj - loss_inj and
related it to the scalar residual norm. That collapses the 256-d residual to its
magnitude — and Run 5 already argued the norm is the wrong summary (the signal
lives in the residual's direction). This stage keeps the same help measure but
asks whether the residual's DIRECTION and the KIND of computation a position
involves explain the help, and whether that beats magnitude.

Analyses
--------
1. Baseline: corr(help, residual_norm) + mean help (reference, expected weak).
2. Help by residual-DIRECTION cluster: k-means on unit-normalized residual
   vectors (magnitude discarded). Per cluster: help, norm, baseline loss, and
   behavioral character. Directly answers the norm-collapse critique.
3. Help by long-range-ness: bins of distance-to-dominant-attended-token, and
   distributed vs focused attention. Operationalizes "helps hard, long-range
   cognition."
4. Help by syntactic category (before_closer etc.), now with norm + baseline.
5. Magnitude control: within residual-norm-decile stratified help for the key
   structural contrasts, so we see structure matters BEYOND magnitude.
6. Headline contrast: does direction-cluster / attention structure organize help
   MORE than residual-norm octiles? (group-mean spread + eta^2).

Novelty/residual is the no-injection forward-model error (post_block0 is
identical with/without injection, so the prediction is unchanged). Attention
categories use the main model's block-1 attention (the layer feeding the
injection), matching behavioral_residual.py.
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
    memory=65536,
)
def a2a_injection_help_structural(
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
    k_clusters: int = 8,
    seed: int = 0,
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
    print(f"A2A structural injection-help on {device}")
    enc = tiktoken.get_encoding("gpt2")
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))

    # --- Data ---
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
    data = torch.from_numpy(np.concatenate(all_tokens)[:n_tokens].astype(np.int64))
    val_data = data[int(0.9 * len(data)):]
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_eval_batches):
        ix = torch.randint(len(val_data) - block_size - 1, (batch_size,), generator=g)
        batches.append(torch.stack([val_data[i:i + block_size] for i in ix]))

    # --- Models ---
    gap_tag = f"{predict_from}_to_{predict_to}"
    model_dir = (f"{DATA_DIR}/a2a_forward/loop_L{fwd_n_layer}/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading closed-loop from {model_dir}")
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size).to(device)
    fwd_model.load_state_dict(torch.load(
        os.path.join(model_dir, "fwd_model.pt"), map_location=device,
        weights_only=True))
    fwd_model.eval()
    gate = CerebellarGate(n_embd).to(device)
    gate.load_state_dict(torch.load(
        os.path.join(model_dir, "gate.pt"), map_location=device, weights_only=True))
    gate.eval()

    def cb_fn(act):
        return gate(fwd_model(act))

    def classify_syntactic(token_ids_np):
        B, T = token_ids_np.shape
        out = {k: np.zeros((B, T), dtype=bool) for k in
               ["after_punct", "after_opener", "before_closer", "sentence_start"]}
        for b in range(B):
            decoded = [enc.decode([int(t)]) for t in token_ids_np[b]]
            for t in range(T):
                prev_text = decoded[t - 1] if t > 0 else ""
                next_text = decoded[t + 1] if t < T - 1 else ""
                prev_s, next_s = prev_text.strip(), next_text.strip()
                if prev_s and prev_s[-1] in PUNCT_CHARS:
                    out["after_punct"][b, t] = True
                    if prev_s[-1] in ".!?" or "\n" in prev_text:
                        out["sentence_start"][b, t] = True
                if prev_s and prev_s[-1] in OPENER_CHARS:
                    out["after_opener"][b, t] = True
                if next_s and next_s[0] in CLOSER_CHARS:
                    out["before_closer"][b, t] = True
        return out

    def block1_attention(source):
        """Return main-model block-1 attention-derived per-token continuous
        features: (entropy, dist_to_dominant, max_attn, distant_mass), each
        (B, T)."""
        block1 = model.transformer.h[1]
        B, T, C = source.size()
        h = block1.ln_1(source)
        qkv = block1.attn.c_attn(h)
        q, k, v = qkv.split(block1.attn.n_embd, dim=2)
        q = q.view(B, T, block1.attn.n_head, block1.attn.head_dim).transpose(1, 2)
        k = k.view(B, T, block1.attn.n_head, block1.attn.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(block1.attn.head_dim))
        att = att.masked_fill(block1.attn.bias[:, :, :T, :T] == 0, float("-inf"))
        aw = F.softmax(att, dim=-1)              # (B, n_head, T, T)
        avg = aw.mean(1)                          # (B, T, T)
        eps = 1e-10
        entropy = -(avg * (avg + eps).log()).sum(-1)         # (B, T)
        dominant = avg.argmax(-1)                            # (B, T)
        qpos = torch.arange(T, device=source.device).unsqueeze(0).expand(B, -1)
        dist_dom = (qpos - dominant).float()                 # (B, T)
        max_attn = avg.max(-1).values                        # (B, T)
        tp = torch.arange(T, device=source.device)
        distant_mask = (tp.unsqueeze(1) - tp.unsqueeze(0) > 10).float()  # (T,T)
        distant_mass = (avg * distant_mask.unsqueeze(0)).sum(-1)         # (B, T)
        return entropy, dist_dom, max_attn, distant_mass

    # --- Collect ---
    print(f"Collecting per-token help + residual vectors over {n_eval_batches} "
          f"batches...")
    dloss, res_vecs, res_norm, base_loss = [], [], [], []
    ent_a, dist_a, maxa_a, distant_a = [], [], [], []
    syn = {k: [] for k in
           ["after_punct", "after_opener", "before_closer", "sentence_start"]}
    with torch.no_grad():
        for bi, x in enumerate(batches):
            x = x.to(device)
            tgt = x[:, 1:]
            logits_inj, _ = model(
                x, cerebellar_fn=cb_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block)
            logits_no, _, inter = model(x, return_intermediates=True)

            def per_tok_loss(logits):
                logp = F.log_softmax(logits[:, :-1], dim=-1)
                return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B,T-1)

            l_inj = per_tok_loss(logits_inj)
            l_no = per_tok_loss(logits_no)
            dloss.append((l_no - l_inj).reshape(-1).cpu())
            base_loss.append(l_no.reshape(-1).cpu())

            pred = fwd_model(inter[predict_from])
            residual = (inter[predict_to] - pred)[:, :-1]      # (B,T-1,C)
            res_vecs.append(residual.reshape(-1, n_embd).cpu())
            res_norm.append(residual.norm(dim=-1).reshape(-1).cpu())

            ent, distd, maxa, distm = block1_attention(inter["post_block0"])
            ent_a.append(ent[:, :-1].reshape(-1).cpu())
            dist_a.append(distd[:, :-1].reshape(-1).cpu())
            maxa_a.append(maxa[:, :-1].reshape(-1).cpu())
            distant_a.append(distm[:, :-1].reshape(-1).cpu())

            info = classify_syntactic(x[:, :-1].cpu().numpy())
            for kk in syn:
                syn[kk].append(info[kk].reshape(-1))
            if bi % 10 == 0:
                print(f"  batch {bi}/{n_eval_batches}")

    dloss = torch.cat(dloss).numpy()
    res_vecs = torch.cat(res_vecs).numpy()
    res_norm = torch.cat(res_norm).numpy()
    base_loss = torch.cat(base_loss).numpy()
    ent = torch.cat(ent_a).numpy()
    dist_dom = torch.cat(dist_a).numpy()
    max_attn = torch.cat(maxa_a).numpy()
    distant_mass = torch.cat(distant_a).numpy()
    syn = {k: np.concatenate(v) for k, v in syn.items()}
    N = len(dloss)
    print(f"Total tokens: {N:,}")

    grand = float(dloss.mean())

    def eta2(labels, n_groups):
        ssb = 0.0
        for gi in range(n_groups):
            m = labels == gi
            if m.sum() > 0:
                ssb += m.sum() * (dloss[m].mean() - grand) ** 2
        sst = ((dloss - grand) ** 2).sum()
        return float(ssb / sst) if sst > 0 else 0.0

    def group_report(labels, n_groups, namer):
        rep = {}
        helps = []
        for gi in range(n_groups):
            m = labels == gi
            if m.sum() == 0:
                continue
            h = float(dloss[m].mean())
            helps.append(h)
            rep[namer(gi)] = {
                "mean_help": h, "mean_norm": float(res_norm[m].mean()),
                "mean_baseline_loss": float(base_loss[m].mean()),
                "n": int(m.sum())}
        spread = (max(helps) - min(helps)) if helps else 0.0
        return rep, spread

    results = {"mean_help": grand,
               "corr_help_vs_residual_norm": float(np.corrcoef(dloss, res_norm)[0, 1])}
    print(f"\nMean help={grand:+.5f}  corr(help,norm)="
          f"{results['corr_help_vs_residual_norm']:+.4f}")

    # =========================================================
    # 2. Help by residual-DIRECTION cluster (magnitude discarded)
    # =========================================================
    print(f"\n=== Help by residual-direction cluster (k={k_clusters}) ===")
    unit = res_vecs / (np.linalg.norm(res_vecs, axis=1, keepdims=True) + 1e-8)
    rng = np.random.default_rng(seed)
    C = unit[rng.choice(N, k_clusters, replace=False)].copy()
    for _ in range(25):
        # distance via -2 x.C (||x||=1, ||C|| ~ const); use full sq dist for safety
        d = (-2.0 * unit @ C.T) + (C ** 2).sum(1)[None, :]
        assign = d.argmin(1)
        for gi in range(k_clusters):
            m = assign == gi
            if m.sum() > 0:
                C[gi] = unit[m].mean(0)
    clu_rep, clu_spread = group_report(assign, k_clusters, lambda i: f"cluster_{i}")
    clu_eta2 = eta2(assign, k_clusters)
    # behavioral character of each cluster
    for gi in range(k_clusters):
        m = assign == gi
        if m.sum() == 0:
            continue
        clu_rep[f"cluster_{gi}"].update({
            "mean_attn_entropy": float(ent[m].mean()),
            "mean_dist_to_dominant": float(dist_dom[m].mean()),
            "frac_before_closer": float(syn["before_closer"][m].mean()),
        })
        r = clu_rep[f"cluster_{gi}"]
        print(f"  cluster {gi}: help={r['mean_help']:+.5f}  norm={r['mean_norm']:.3f}"
              f"  base={r['mean_baseline_loss']:.3f}  ent={r['mean_attn_entropy']:.2f}"
              f"  distDom={r['mean_dist_to_dominant']:.1f}"
              f"  %bclose={100*r['frac_before_closer']:.1f}  n={r['n']:,}")
    print(f"  -> help spread across clusters: {clu_spread:.5f}, eta^2={clu_eta2:.4f}")
    results["help_by_residual_direction_cluster"] = clu_rep
    results["cluster_help_spread"] = clu_spread
    results["cluster_eta2"] = clu_eta2

    # =========================================================
    # 3a. Fair magnitude baseline: help by residual-NORM octile (8 groups)
    # =========================================================
    print("\n=== Help by residual-norm octile (magnitude-only, 8 groups) ===")
    oct_edges = np.percentile(res_norm, np.arange(12.5, 100, 12.5))
    oct_lab = np.digitize(res_norm, oct_edges)
    oct_rep, oct_spread = group_report(oct_lab, 8, lambda i: f"octile_{i}")
    oct_eta2 = eta2(oct_lab, 8)
    for gi in range(8):
        r = oct_rep.get(f"octile_{gi}")
        if r:
            print(f"  octile {gi}: help={r['mean_help']:+.5f}  norm={r['mean_norm']:.3f}"
                  f"  n={r['n']:,}")
    print(f"  -> help spread across norm-octiles: {oct_spread:.5f}, eta^2={oct_eta2:.4f}")
    results["help_by_norm_octile"] = oct_rep
    results["norm_octile_help_spread"] = oct_spread
    results["norm_octile_eta2"] = oct_eta2

    # =========================================================
    # 3b. Help by long-range-ness (distance to dominant attended token)
    # =========================================================
    print("\n=== Help by attention distance to dominant attended token ===")
    dist_bins = [(-0.5, 1.5, "0-1 (local)"), (1.5, 4.5, "2-4"),
                 (4.5, 10.5, "5-10"), (10.5, 1e9, ">10 (long-range)")]
    dist_rep = {}
    for lo, hi, lab in dist_bins:
        m = (dist_dom > lo) & (dist_dom <= hi)
        if m.sum() == 0:
            continue
        dist_rep[lab] = {"mean_help": float(dloss[m].mean()),
                         "mean_norm": float(res_norm[m].mean()),
                         "mean_baseline_loss": float(base_loss[m].mean()),
                         "n": int(m.sum())}
        r = dist_rep[lab]
        print(f"  {lab:>18s}: help={r['mean_help']:+.5f}  norm={r['mean_norm']:.3f}"
              f"  base={r['mean_baseline_loss']:.3f}  n={r['n']:,}")
    results["help_by_attention_distance"] = dist_rep

    # =========================================================
    # 3c. Distributed vs focused attention
    # =========================================================
    print("\n=== Help by attention shape ===")
    med_ent = float(np.median(ent))
    attn_shape = {
        "distributed (entropy>median)": ent > med_ent,
        "focused (max>0.5)": max_attn > 0.5,
        "distant (>0.1 mass >10 back)": distant_mass > 0.1,
    }
    shape_rep = {}
    for lab, m in attn_shape.items():
        if m.sum() == 0:
            continue
        shape_rep[lab] = {"mean_help": float(dloss[m].mean()),
                          "mean_norm": float(res_norm[m].mean()),
                          "mean_baseline_loss": float(base_loss[m].mean()),
                          "n": int(m.sum())}
        r = shape_rep[lab]
        print(f"  {lab:>30s}: help={r['mean_help']:+.5f}  norm={r['mean_norm']:.3f}"
              f"  n={r['n']:,}")
    results["help_by_attention_shape"] = shape_rep

    # =========================================================
    # 4. Syntactic categories (with norm + baseline)
    # =========================================================
    print("\n=== Help by syntactic category ===")
    syn_rep = {}
    for k in ["before_closer", "after_opener", "after_punct", "sentence_start"]:
        m = syn[k]
        if m.sum() == 0:
            continue
        syn_rep[k] = {"mean_help": float(dloss[m].mean()),
                      "mean_norm": float(res_norm[m].mean()),
                      "mean_baseline_loss": float(base_loss[m].mean()),
                      "n": int(m.sum())}
        r = syn_rep[k]
        print(f"  {k:>16s}: help={r['mean_help']:+.5f}  norm={r['mean_norm']:.3f}"
              f"  base={r['mean_baseline_loss']:.3f}  n={r['n']:,}")
    results["help_by_syntactic_category"] = syn_rep

    # =========================================================
    # 5. Magnitude control: within residual-norm-decile, long-range vs local
    # =========================================================
    print("\n=== Magnitude-controlled (within norm-decile): long-range vs local ===")
    dec_edges = np.percentile(res_norm, np.arange(10, 100, 10))
    dec = np.digitize(res_norm, dec_edges)
    longrange = dist_dom > 10
    local = dist_dom <= 1
    diffs = []
    for b in range(10):
        m = dec == b
        ml, mloc = m & longrange, m & local
        if ml.sum() > 100 and mloc.sum() > 100:
            diffs.append(float(dloss[ml].mean() - dloss[mloc].mean()))
    within = float(np.mean(diffs)) if diffs else float("nan")
    raw = float(dloss[longrange].mean() - dloss[local].mean())
    results["longrange_vs_local_help"] = {"within_norm_decile": within, "raw": raw}
    print(f"  help(long-range) - help(local): within-norm-decile={within:+.5f}  "
          f"raw={raw:+.5f}")

    # --- Headline contrast ---
    print("\n=== HEADLINE: does structure organize help more than magnitude? ===")
    print(f"  eta^2  residual-direction clusters: {clu_eta2:.4f}")
    print(f"  eta^2  residual-norm octiles:       {oct_eta2:.4f}")
    print(f"  help spread  clusters: {clu_spread:.5f}   norm-octiles: {oct_spread:.5f}")
    results["headline"] = {
        "cluster_eta2": clu_eta2, "norm_octile_eta2": oct_eta2,
        "cluster_spread": clu_spread, "norm_octile_spread": oct_spread,
        "structure_beats_magnitude": bool(clu_eta2 > oct_eta2)}

    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "injection_help_structural_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {save_path}")
    return results
