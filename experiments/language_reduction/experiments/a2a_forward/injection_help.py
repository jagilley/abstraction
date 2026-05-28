"""Idea 3 (direct test): does the injection help LM loss MORE at low-novelty
positions?

This is the metric most tightly bound to the README's "division of labor"
argument: the forward model handles the predictable/routine computation (low
novelty), and blocks 2-3 handle the deviation. If that's how the closed-loop
benefit arises, then the per-token help from the injection,

    delta_loss = loss_no_inj - loss_inj   (positive = injection helps),

should be LARGER where the prediction is accurate (low novelty) and shrink (or
go negative) at high-novelty positions where the prediction is wrong.

Prediction (gating / division-of-labor hypothesis):
    corr(delta_loss, residual_norm) < 0, and Q1(low-novelty) help > Q4(high).

The complementary steering test (novelty_steer.py) found NO novelty-gated
reliance (corr(KL_influence, residual)~0; steering non-specific). This stage
asks the helpfulness version directly. A null/positive correlation here would
further confirm: the injection is a roughly uniform preview, used the same
regardless of local novelty -- the closed-loop benefit is not selective trust.
"""

import json

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
def a2a_injection_help(
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
    print(f"A2A injection-help analysis on {device}")
    enc = tiktoken.get_encoding("gpt2")
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1"))

    # --- Load data ---
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

    # --- Load closed-loop model + forward model + gate ---
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
    print(f"Gate injection norm: {gate.injection_norm():.4f}")

    def classify_syntactic(token_ids_np):
        B, T = token_ids_np.shape
        out = {k: np.zeros(B * T, dtype=bool) for k in
               ["after_punct", "after_opener", "before_closer", "sentence_start"]}
        for b in range(B):
            decoded = [enc.decode([int(t)]) for t in token_ids_np[b]]
            base = b * T
            for t in range(T):
                prev_text = decoded[t - 1] if t > 0 else ""
                next_text = decoded[t + 1] if t < T - 1 else ""
                prev_s, next_s = prev_text.strip(), next_text.strip()
                i = base + t
                if prev_s and prev_s[-1] in PUNCT_CHARS:
                    out["after_punct"][i] = True
                    if prev_s[-1] in ".!?" or "\n" in prev_text:
                        out["sentence_start"][i] = True
                if prev_s and prev_s[-1] in OPENER_CHARS:
                    out["after_opener"][i] = True
                if next_s and next_s[0] in CLOSER_CHARS:
                    out["before_closer"][i] = True
        return out

    def cb_fn(act):
        return gate(fwd_model(act))

    # --- Collect per-token delta_loss and novelty ---
    print(f"Collecting per-token help over {n_eval_batches} batches...")
    dloss, res_no, res_with, lm_inj_all, lm_no_all = [], [], [], [], []
    syn = {k: [] for k in
           ["after_punct", "after_opener", "before_closer", "sentence_start"]}
    with torch.no_grad():
        for bi, x in enumerate(batches):
            x = x.to(device)
            tgt = x[:, 1:]

            logits_inj, _, inter_w = model(
                x, return_intermediates=True, cerebellar_fn=cb_fn,
                cerebellar_input_block=cerebellar_input_block,
                cerebellar_inject_block=inject_after_block)
            logits_no, _, inter_n = model(x, return_intermediates=True)

            def per_tok_loss(logits):
                logp = F.log_softmax(logits[:, :-1], dim=-1)
                return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B,T-1)

            l_inj = per_tok_loss(logits_inj)
            l_no = per_tok_loss(logits_no)
            dloss.append((l_no - l_inj).reshape(-1).cpu())
            lm_inj_all.append(l_inj.reshape(-1).cpu())
            lm_no_all.append(l_no.reshape(-1).cpu())

            # novelty: forward-model error. post_block0 identical w/wo injection.
            pred = fwd_model(inter_n[predict_from])
            rn_no = (inter_n[predict_to] - pred).norm(dim=-1)[:, :-1]   # (B,T-1)
            rn_w = (inter_w[predict_to] - pred).norm(dim=-1)[:, :-1]
            res_no.append(rn_no.reshape(-1).cpu())
            res_with.append(rn_w.reshape(-1).cpu())

            info = classify_syntactic(x[:, :-1].cpu().numpy())  # align to T-1
            for k in syn:
                syn[k].append(info[k])
            if bi % 10 == 0:
                print(f"  batch {bi}/{n_eval_batches}")

    dloss = torch.cat(dloss).numpy()
    res_no = torch.cat(res_no).numpy()
    res_with = torch.cat(res_with).numpy()
    lm_no_tok = torch.cat(lm_no_all).numpy()
    lm_inj = float(torch.cat(lm_inj_all).mean())
    lm_no = float(lm_no_tok.mean())
    syn = {k: np.concatenate(v) for k, v in syn.items()}

    # relative help: control for baseline headroom (high-loss positions have
    # more nats to gain). rel_help = fraction of baseline loss removed.
    rel_help = dloss / np.clip(lm_no_tok, 1e-3, None)

    # --- Headline correlations ---
    r_no = float(np.corrcoef(dloss, res_no)[0, 1])
    r_with = float(np.corrcoef(dloss, res_with)[0, 1])
    print(f"\nMean loss: inj={lm_inj:.4f}  no_inj={lm_no:.4f}  "
          f"mean help (Δ)={lm_no - lm_inj:+.4f}")
    print(f"corr(delta_loss, residual_norm  [no-inj novelty])  = {r_no:+.4f}")
    print(f"corr(delta_loss, residual_norm  [with-inj novelty])= {r_with:+.4f}")
    print("(division-of-labor hypothesis predicts NEGATIVE)")

    # --- Δloss by novelty quartile (using no-inj novelty) ---
    qs = np.percentile(res_no, [25, 50, 75])
    bounds = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    qlabels = ["Q1 (low novelty)", "Q2", "Q3", "Q4 (high novelty)"]
    quartile_help = {}
    print("\nInjection help by novelty quartile (abs / relative / baseline loss):")
    for i in range(4):
        m = (res_no >= bounds[i]) & (res_no < bounds[i + 1])
        val = float(dloss[m].mean())
        rel = float(rel_help[m].mean())
        base = float(lm_no_tok[m].mean())
        quartile_help[qlabels[i]] = {"mean_help": val, "mean_rel_help": rel,
                                     "mean_baseline_loss": base, "n": int(m.sum())}
        print(f"  {qlabels[i]:>20s}: help={val:+.5f}  rel={rel:+.4f}  "
              f"base_loss={base:.3f}  (n={int(m.sum()):,})")

    # --- Δloss by syntactic category ---
    overall = float(dloss.mean())
    print(f"\nInjection help by syntactic category (overall help={overall:+.5f}):")
    syn_help = {}
    for k in ["before_closer", "after_opener", "after_punct", "sentence_start"]:
        m = syn[k]
        if m.sum() > 0:
            val, rel, base = (float(dloss[m].mean()), float(rel_help[m].mean()),
                              float(lm_no_tok[m].mean()))
        else:
            val = rel = base = float("nan")
        syn_help[k] = {"mean_help": val, "mean_rel_help": rel,
                       "mean_baseline_loss": base, "n": int(m.sum())}
        print(f"  {k:>16s}: help={val:+.5f}  rel={rel:+.4f}  "
              f"base_loss={base:.3f}  (n={int(m.sum()):,})")

    # --- Loss-controlled: stratify by baseline-loss decile, corr within bins ---
    # If "help rises with novelty" survives controlling for headroom, the
    # scaffold story holds; if it collapses, help just tracks baseline loss.
    deciles = np.percentile(lm_no_tok, np.arange(10, 100, 10))
    bin_idx = np.digitize(lm_no_tok, deciles)
    within_corrs, within_q4mq1 = [], []
    for b in range(10):
        m = bin_idx == b
        if m.sum() < 200:
            continue
        within_corrs.append(float(np.corrcoef(dloss[m], res_no[m])[0, 1]))
        rb = res_no[m]
        qlo, qhi = np.percentile(rb, [25, 75])
        within_q4mq1.append(float(dloss[m][rb >= qhi].mean()
                                  - dloss[m][rb <= qlo].mean()))
    loss_controlled = {
        "stratified_corr_help_vs_novelty": float(np.mean(within_corrs)),
        "stratified_help_Q4_minus_Q1": float(np.mean(within_q4mq1)),
        "raw_help_Q4_minus_Q1": float(quartile_help[qlabels[3]]["mean_help"]
                                      - quartile_help[qlabels[0]]["mean_help"]),
    }
    print("\nLoss-controlled (stratified by baseline-loss decile):")
    print(f"  mean within-bin corr(help, novelty): "
          f"{loss_controlled['stratified_corr_help_vs_novelty']:+.4f}")
    print(f"  within-bin help Q4-Q1: "
          f"{loss_controlled['stratified_help_Q4_minus_Q1']:+.5f}  "
          f"(raw, uncontrolled: {loss_controlled['raw_help_Q4_minus_Q1']:+.5f})")

    results = {
        "mean_loss_inj": lm_inj,
        "mean_loss_no_inj": lm_no,
        "mean_help": overall,
        "mean_rel_help": float(rel_help.mean()),
        "corr_help_vs_residual_no_inj": r_no,
        "corr_help_vs_residual_with_inj": r_with,
        "help_by_novelty_quartile": quartile_help,
        "help_by_syntactic_category": syn_help,
        "loss_controlled": loss_controlled,
    }
    save_dir = f"{DATA_DIR}/a2a_forward/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "injection_help_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    print(f"\nSaved to {save_path}")
    return results
