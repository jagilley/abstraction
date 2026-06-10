"""Calibration transfer: does the model know what it doesn't know — and does
that knowledge survive distribution shift?

We train linear *competence probes* from each layer's activations to the main
model's own forthcoming per-token LM loss, in-distribution (FineWeb val). We
then evaluate the SAME probes (frozen, no retraining) on out-of-distribution
corpora: Wikipedia (near shift), Python code, French, open-web-math (far
shifts), and shuffled FineWeb tokens (structure-destroying control).

The probe target involves NO forward model — it is the main model's own
error. The forward model enters only as the historical training pressure
that (hypothesis) reorganized the closed-loop model's representations.

Hypothesis (paper §5.6 / conclusion): forward prediction's self-knowledge
encodes the model's computational function (weight-determined, distribution-
invariant), while learned surface correlates of competence (token frequency;
cf. Kadavath et al. 2022's OOD calibration collapse, Brier 0.15 → 0.43) are
distribution-bound. If so, the closed-loop (forward) model's competence
probes should retain more of their discrimination OOD than the open-loop
model's, and the gap should exceed what the control conditions (autoencoder,
random_proj, shifted) achieve.

Conditions reuse the baseline battery checkpoints (no retraining):
  open_loop, forward (natural mode, injection on), forward_noinj
  (same weights, injection off), shifted, random_proj, autoencoder.

Baselines per corpus:
  - control-feature probe: [logfreq(cur), logfreq(prev), position,
    unseen-token flag] → loss. The Kadavath-style surface-correlate probe.
  - output entropy: the model's own confidence (behavioral calibration).

Primary metric: Spearman rho between probe prediction and actual per-token
loss (scale-free, robust to OOD mean-shift). Also Pearson, ID-scaled R²,
AUC for top-1 incorrectness, and sequence-level (128-token window) Spearman.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "forward", "forward_noinj", "shifted",
              "random_proj", "autoencoder"]

# Corpora streamed from HF, tokenized with tiktoken gpt2 (same as training)
OOD_SPECS = {
    "wikipedia_en": dict(path="wikimedia/wikipedia", name="20231101.en",
                         field="text"),
    "french": dict(path="wikimedia/wikipedia", name="20231101.fr",
                   field="text"),
    "code_python": dict(path="codeparrot/codeparrot-clean-valid", name=None,
                        field="content"),
    "math": dict(path="open-web-math/open-web-math", name=None,
                 field="text"),
}

OOD_TOKEN_DIR = f"{DATA_DIR}/ood_tokens"


@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=16384, cpu=4)
def cache_ood_tokens(n_tokens_per_corpus: int = 2_000_000,
                     force: bool = False,
                     corpora: str = ""):
    """Tokenize OOD corpora to the volume. Idempotent unless force=True."""
    import os
    import numpy as np
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(OOD_TOKEN_DIR, exist_ok=True)
    wanted = [c.strip() for c in corpora.split(",") if c.strip()] \
        or list(OOD_SPECS.keys())

    status = {}
    for cname in wanted:
        spec = OOD_SPECS[cname]
        out_path = os.path.join(OOD_TOKEN_DIR, f"{cname}.npy")
        if os.path.exists(out_path) and not force:
            n = len(np.load(out_path, mmap_mode="r"))
            print(f"[{cname}] exists ({n:,} tokens), skipping")
            status[cname] = n
            continue

        print(f"[{cname}] streaming {spec['path']}"
              f"{' / ' + spec['name'] if spec['name'] else ''} ...")
        if spec["name"]:
            ds = load_dataset(spec["path"], spec["name"], split="train",
                              streaming=True)
        else:
            ds = load_dataset(spec["path"], split="train", streaming=True)

        toks = []
        n_docs = 0
        for doc in ds:
            text = doc.get(spec["field"])
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            toks.extend(ids)
            n_docs += 1
            if len(toks) >= n_tokens_per_corpus:
                break
        arr = np.array(toks[:n_tokens_per_corpus], dtype=np.uint16)
        np.save(out_path, arr)
        volume.commit()
        print(f"[{cname}] saved {len(arr):,} tokens from {n_docs} docs "
              f"to {out_path}")
        status[cname] = len(arr)

    return status


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_calibration_transfer(
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
    shift_k: int = 10,
    seed: int = 42,
    probe_train_batches: int = 40,
    eval_batches: int = 20,
    ridge_lambda: float = 1e-3,
    conditions: str = "",
    corpora: str = "",
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy.stats import spearmanr, pearsonr, rankdata
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1")
    )
    run_conditions = [c.strip() for c in conditions.split(",") if c.strip()] \
        or CONDITIONS
    run_ood = [c.strip() for c in corpora.split(",") if c.strip()] \
        or (list(OOD_SPECS.keys()) + ["shuffled"])

    print(f"CALIBRATION TRANSFER on {device}")
    print(f"  Conditions: {run_conditions}")
    print(f"  OOD corpora: {run_ood}")
    print(f"  Probe: activations → own per-token LM loss, "
          f"{probe_train_batches} train / {eval_batches} eval batches, "
          f"ridge lambda={ridge_lambda}")

    # =========================================================
    # Load ID data (same shards as training)
    # =========================================================
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"),
                   allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_tokens)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    print(f"Loaded {len(data):,} ID tokens (vocab_size={vocab_size})")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    # Disjoint val halves: probes train on the low half, ID-test on the high
    half = len(val_data) // 2
    val_lo = val_data[:half]
    val_hi = val_data[half:]

    # Unigram frequency table from the training split (for the
    # surface-correlate control probe)
    counts = np.bincount(train_data.numpy(), minlength=vocab_size)
    freq = counts / counts.sum()
    log_freq = np.log(freq + 1e-9).astype(np.float32)
    unseen = (counts == 0).astype(np.float32)

    # =========================================================
    # Build eval batches (identical across conditions)
    # =========================================================
    def random_batches(source, n_batches, gen):
        idx_sets = [
            torch.randint(len(source) - block_size - 1, (batch_size,),
                          generator=gen)
            for _ in range(n_batches)
        ]
        batches = []
        for idxs in idx_sets:
            x = torch.stack(
                [source[i.item():i.item() + block_size] for i in idxs])
            y = torch.stack(
                [source[i.item() + 1:i.item() + block_size + 1]
                 for i in idxs])
            batches.append((x, y))
        return batches

    def window_batches(arr, n_batches):
        """Contiguous non-overlapping windows from a token array."""
        toks = torch.from_numpy(np.asarray(arr).astype(np.int64))
        n_windows = n_batches * batch_size
        need = n_windows * (block_size + 1)
        assert len(toks) >= need, \
            f"corpus too small: {len(toks)} < {need}"
        w = toks[:need].reshape(n_windows, block_size + 1)
        return [
            (w[b * batch_size:(b + 1) * batch_size, :-1],
             w[b * batch_size:(b + 1) * batch_size, 1:])
            for b in range(n_batches)
        ]

    probe_train_gen = torch.Generator().manual_seed(seed + 10)
    id_test_gen = torch.Generator().manual_seed(seed + 11)
    shuffle_gen = torch.Generator().manual_seed(seed + 12)

    corpus_batches = {
        "id_train": random_batches(val_lo, probe_train_batches,
                                   probe_train_gen),
        "id_test": random_batches(val_hi, eval_batches, id_test_gen),
    }
    for cname in run_ood:
        if cname == "shuffled":
            perm = torch.randperm(len(val_hi), generator=shuffle_gen)
            corpus_batches["shuffled"] = window_batches(
                val_hi[perm].numpy(), eval_batches)
        else:
            path = os.path.join(OOD_TOKEN_DIR, f"{cname}.npy")
            if not os.path.exists(path):
                print(f"WARNING: {path} missing — run cache_ood_tokens "
                      f"first. Skipping {cname}.")
                continue
            arr = np.load(path)
            corpus_batches[cname] = window_batches(arr, eval_batches)
    eval_corpora = [c for c in ["id_test"] + run_ood if c in corpus_batches]
    ood_for_retention = [c for c in eval_corpora
                         if c not in ("id_test", "shuffled")]
    print(f"Eval corpora ready: {eval_corpora}")

    # =========================================================
    # Load checkpoints
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/baseline_battery/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading checkpoints from {ckpt_root}")

    def load_models(condition):
        # forward_noinj shares the forward condition's weights
        ckpt_cond = "forward" if condition == "forward_noinj" else condition
        cdir = os.path.join(ckpt_root, ckpt_cond)
        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd
                    ).to(device)
        model.load_state_dict(
            torch.load(os.path.join(cdir, "model.pt"), map_location=device))
        model.eval()

        fwd_model, gate, autoenc = None, None, None
        if ckpt_cond != "open_loop":
            gate = CerebellarGate(n_embd).to(device)
            gate.load_state_dict(
                torch.load(os.path.join(cdir, "gate.pt"),
                           map_location=device))
            gate.eval()
        if ckpt_cond in ("forward", "shifted"):
            fwd_model = TransformerForwardModel(
                d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                block_size=block_size).to(device)
            fwd_model.load_state_dict(
                torch.load(os.path.join(cdir, "fwd_model.pt"),
                           map_location=device))
            fwd_model.eval()
        if ckpt_cond == "autoencoder":
            autoenc = TransformerForwardModel(
                d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                block_size=block_size).to(device)
            autoenc.load_state_dict(
                torch.load(os.path.join(cdir, "autoenc_model.pt"),
                           map_location=device))
            autoenc.eval()
        return model, fwd_model, gate, autoenc

    # Frozen random projection — must match baseline_battery's (seed + 200)
    rng_proj = np.random.default_rng(seed + 200)
    random_proj_weight = torch.from_numpy(
        (rng_proj.standard_normal((n_embd, n_embd)) / np.sqrt(n_embd)
         ).astype(np.float32)).to(device)

    def make_cb(condition, fwd_model, gate, autoenc):
        """Natural-mode injection fn for eval (no_grad context)."""
        if condition in ("open_loop", "forward_noinj"):
            return None
        if condition == "forward":
            return lambda act: gate(fwd_model(act))
        if condition == "shifted":
            def fn(act):
                pred = fwd_model(act)
                shifted = torch.zeros_like(pred)
                if shift_k < pred.shape[1]:
                    shifted[:, shift_k:, :] = pred[:, :-shift_k, :]
                return gate(shifted)
            return fn
        if condition == "random_proj":
            return lambda act: gate(act @ random_proj_weight.T)
        if condition == "autoencoder":
            return lambda act: gate(autoenc(act))
        raise ValueError(condition)

    # =========================================================
    # Activation + loss collection
    # =========================================================
    layer_keys = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    def collect(model, cb_fn, batches):
        """Run batches in natural mode; per position t (0..T-2) record
        activations, the model's loss predicting token t+1, correctness,
        output entropy, and control features."""
        acts = {k: [] for k in layer_keys}
        loss_l, correct_l, ent_l = [], [], []
        cur_l, prev_l, pos_l, seq_l = [], [], [], []
        seq_offset = 0

        with torch.no_grad():
            for x, _y in batches:
                x = x.to(device)
                logits, _, vi = model(
                    x, None, return_intermediates=True,
                    cerebellar_fn=cb_fn,
                    cerebellar_input_block=cerebellar_input_block,
                    cerebellar_inject_block=inject_after_block,
                )
                B, T = x.shape
                tgt = x[:, 1:]
                logp = F.log_softmax(logits[:, :-1], dim=-1)
                loss = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                correct = (logits[:, :-1].argmax(-1) == tgt)
                ent = -(logp.exp() * logp).sum(-1)

                for k in layer_keys:
                    acts[k].append(
                        vi[k][:, :-1].reshape(-1, n_embd).cpu())
                loss_l.append(loss.reshape(-1).cpu())
                correct_l.append(correct.reshape(-1).cpu())
                ent_l.append(ent.reshape(-1).cpu())

                cur = x[:, :-1]
                prev = x.clone()
                prev[:, 1:] = x[:, :-1]
                prev = prev[:, :-1]
                cur_l.append(cur.reshape(-1).cpu())
                prev_l.append(prev.reshape(-1).cpu())
                pos_l.append(
                    torch.arange(T - 1).repeat(B).cpu())
                seq_l.append(
                    (seq_offset + torch.arange(B)
                     ).repeat_interleave(T - 1))
                seq_offset += B

        return {
            "acts": {k: torch.cat(v).numpy() for k, v in acts.items()},
            "loss": torch.cat(loss_l).numpy().astype(np.float64),
            "correct": torch.cat(correct_l).numpy(),
            "entropy": torch.cat(ent_l).numpy().astype(np.float64),
            "cur": torch.cat(cur_l).numpy(),
            "prev": torch.cat(prev_l).numpy(),
            "pos": torch.cat(pos_l).numpy().astype(np.float32),
            "seq": torch.cat(seq_l).numpy(),
        }

    def control_features(d):
        return np.stack([
            log_freq[d["cur"]],
            log_freq[d["prev"]],
            d["pos"] / float(block_size),
            unseen[d["cur"]],
        ], axis=1).astype(np.float32)

    # =========================================================
    # Probe training + metrics
    # =========================================================
    def train_probe(X_tr, t_tr):
        """Closed-form ridge probe on standardized features →
        standardized log1p loss. Exact optimum — no SGD convergence risk
        (an undertrained probe can keep its random-init sign).
        Returns (predict_fn, t_stats)."""
        mu_x = X_tr.mean(axis=0, keepdims=True)
        sd_x = X_tr.std(axis=0, keepdims=True) + 1e-6
        t = np.log1p(t_tr)
        mu_t, sd_t = t.mean(), t.std() + 1e-6
        X_s = ((X_tr - mu_x) / sd_x).astype(np.float64)
        t_s = ((t - mu_t) / sd_t).astype(np.float64)

        n, d = X_s.shape
        A = X_s.T @ X_s + ridge_lambda * n * np.eye(d)
        b = X_s.T @ t_s
        w = np.linalg.solve(A, b)

        def predict(X):
            X_s_ = ((X - mu_x) / sd_x).astype(np.float64)
            return X_s_ @ w

        return predict, (mu_t, sd_t)

    def auc_score(score, positive):
        positive = positive.astype(bool)
        n_pos = int(positive.sum())
        n_neg = int((~positive).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        r = rankdata(score)
        return float(
            (r[positive].sum() - n_pos * (n_pos + 1) / 2)
            / (n_pos * n_neg))

    def eval_metrics(score, d, t_stats):
        """score: monotonic predictor of per-token loss."""
        loss_raw = d["loss"]
        mu_t, sd_t = t_stats
        t_s = (np.log1p(loss_raw) - mu_t) / sd_t
        var = t_s.var()
        r2 = float(1.0 - ((score - t_s) ** 2).mean() / var) \
            if var > 0 else float("nan")
        sp = float(spearmanr(score, loss_raw).correlation)
        pe = float(pearsonr(score, np.log1p(loss_raw))[0])
        auc = auc_score(score, 1 - d["correct"])
        # Sequence level: mean score vs mean loss per 127-pos window
        seq = d["seq"]
        n_seq = seq.max() + 1
        seq_score = np.bincount(seq, weights=score) / np.bincount(seq)
        seq_loss = np.bincount(seq, weights=loss_raw) / np.bincount(seq)
        sp_seq = float(spearmanr(seq_score, seq_loss).correlation) \
            if n_seq > 2 else float("nan")
        return {"spearman": sp, "pearson_log": pe, "r2": r2, "auc": auc,
                "seq_spearman": sp_seq}

    # =========================================================
    # Run everything, one condition at a time
    # =========================================================
    results = {
        "config": {
            "n_tokens": n_tokens, "seed": seed,
            "probe_train_batches": probe_train_batches,
            "eval_batches": eval_batches, "ridge_lambda": ridge_lambda,
            "conditions": run_conditions, "eval_corpora": eval_corpora,
            "ckpt_root": ckpt_root,
        },
        "corpus_stats": {},
        "probes": {},
        "control_probe": {},
        "entropy_baseline": {},
    }

    for cond in run_conditions:
        print(f"\n{'='*60}")
        print(f"  CONDITION: {cond.upper()}")
        print(f"{'='*60}")
        model, fwd_model, gate, autoenc = load_models(cond)
        cb_fn = make_cb(cond, fwd_model, gate, autoenc)

        print("Collecting activations + losses...")
        collected = {}
        collected["id_train"] = collect(
            model, cb_fn, corpus_batches["id_train"])
        for cname in eval_corpora:
            collected[cname] = collect(model, cb_fn, corpus_batches[cname])

        # Corpus stats
        results["corpus_stats"][cond] = {}
        print(f"\n{'corpus':>14s} {'mean_loss':>10s} {'acc':>7s} "
              f"{'mean_ent':>9s}")
        for cname in ["id_train"] + eval_corpora:
            d = collected[cname]
            st = {
                "mean_loss": float(d["loss"].mean()),
                "acc": float(d["correct"].mean()),
                "mean_entropy": float(d["entropy"].mean()),
                "n": int(len(d["loss"])),
            }
            results["corpus_stats"][cond][cname] = st
            print(f"{cname:>14s} {st['mean_loss']:>10.3f} "
                  f"{st['acc']:>7.3f} {st['mean_entropy']:>9.3f}")

        d_tr = collected["id_train"]

        # --- Activation probes per layer ---
        results["probes"][cond] = {}
        for lk in layer_keys:
            predict, t_stats = train_probe(d_tr["acts"][lk], d_tr["loss"])
            results["probes"][cond][lk] = {}
            for cname in eval_corpora:
                d = collected[cname]
                score = predict(d["acts"][lk])
                results["probes"][cond][lk][cname] = eval_metrics(
                    score, d, t_stats)

        # --- Control-feature probe (surface correlates) ---
        predict_ctl, t_stats_ctl = train_probe(
            control_features(d_tr), d_tr["loss"])
        results["control_probe"][cond] = {}
        for cname in eval_corpora:
            d = collected[cname]
            score = predict_ctl(control_features(d))
            results["control_probe"][cond][cname] = eval_metrics(
                score, d, t_stats_ctl)

        # --- Entropy baseline (model's own behavioral confidence) ---
        # Standardize entropy → loss-space score via the same probe
        # machinery (1-feature probe) so R² is comparable.
        predict_ent, t_stats_ent = train_probe(
            d_tr["entropy"][:, None].astype(np.float32), d_tr["loss"])
        results["entropy_baseline"][cond] = {}
        for cname in eval_corpora:
            d = collected[cname]
            score = predict_ent(
                d["entropy"][:, None].astype(np.float32))
            results["entropy_baseline"][cond][cname] = eval_metrics(
                score, d, t_stats_ent)

        # --- Per-condition summary table ---
        print(f"\n--- {cond}: Spearman rho (probe score vs actual loss) ---")
        header = f"{'probe':>14s}" + "".join(
            f" {c:>13s}" for c in eval_corpora) + f" {'retention':>10s}"
        print(header)
        for lk in layer_keys:
            row = results["probes"][cond][lk]
            id_sp = row["id_test"]["spearman"]
            oods = [row[c]["spearman"] for c in ood_for_retention]
            ret = float(np.mean(oods)) / id_sp if oods and id_sp else \
                float("nan")
            print(f"{lk:>14s}" + "".join(
                f" {row[c]['spearman']:>13.4f}" for c in eval_corpora)
                + f" {ret:>10.3f}")
        for name, res in [("control", results["control_probe"][cond]),
                          ("entropy", results["entropy_baseline"][cond])]:
            id_sp = res["id_test"]["spearman"]
            oods = [res[c]["spearman"] for c in ood_for_retention]
            ret = float(np.mean(oods)) / id_sp if oods and id_sp else \
                float("nan")
            print(f"{name:>14s}" + "".join(
                f" {res[c]['spearman']:>13.4f}" for c in eval_corpora)
                + f" {ret:>10.3f}")

        del collected, model, fwd_model, gate, autoenc
        torch.cuda.empty_cache()

    # =========================================================
    # Cross-condition headline tables
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  HEADLINE: calibration transfer by condition")
    print(f"  (Spearman rho; retention = mean OOD / ID, "
          f"excluding shuffled)")
    print(f"{'='*60}")

    for lk in layer_keys:
        print(f"\n[{lk}]")
        print(f"{'condition':>14s} {'ID rho':>8s} {'OOD rho':>8s} "
              f"{'retention':>10s} {'ID AUC':>7s} {'OOD AUC':>8s}")
        for cond in run_conditions:
            row = results["probes"][cond][lk]
            id_sp = row["id_test"]["spearman"]
            id_auc = row["id_test"]["auc"]
            oods = [row[c]["spearman"] for c in ood_for_retention]
            oaucs = [row[c]["auc"] for c in ood_for_retention]
            ood_sp = float(np.mean(oods)) if oods else float("nan")
            ood_auc = float(np.mean(oaucs)) if oaucs else float("nan")
            ret = ood_sp / id_sp if id_sp else float("nan")
            print(f"{cond:>14s} {id_sp:>8.4f} {ood_sp:>8.4f} "
                  f"{ret:>10.3f} {id_auc:>7.4f} {ood_auc:>8.4f}")

    print(f"\n[baselines (per condition)]")
    print(f"{'condition':>14s} {'probe':>9s} {'ID rho':>8s} "
          f"{'OOD rho':>8s} {'retention':>10s}")
    for cond in run_conditions:
        for name, res in [("control", results["control_probe"][cond]),
                          ("entropy", results["entropy_baseline"][cond])]:
            id_sp = res["id_test"]["spearman"]
            oods = [res[c]["spearman"] for c in ood_for_retention]
            ood_sp = float(np.mean(oods)) if oods else float("nan")
            ret = ood_sp / id_sp if id_sp else float("nan")
            print(f"{cond:>14s} {name:>9s} {id_sp:>8.4f} "
                  f"{ood_sp:>8.4f} {ret:>10.3f}")

    # =========================================================
    # Save
    # =========================================================
    save_root = (f"{DATA_DIR}/a2a_forward/calibration_transfer/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nAll results saved to {results_path}")
    return results


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    probe_train_batches: int = 40,
    eval_batches: int = 20,
    ridge_lambda: float = 1e-3,
    conditions: str = "",
    corpora: str = "",
    skip_cache: bool = False,
):
    if not skip_cache:
        cache_ood_tokens.remote()
    results = a2a_calibration_transfer.remote(
        n_tokens=n_tokens,
        probe_train_batches=probe_train_batches,
        eval_batches=eval_batches,
        ridge_lambda=ridge_lambda,
        conditions=conditions,
        corpora=corpora,
    )
    print("\n=== CALIBRATION TRANSFER COMPLETE ===")
    import numpy as np
    eval_corpora = results["config"]["eval_corpora"]
    ood = [c for c in eval_corpora if c not in ("id_test", "shuffled")]
    for cond in results["config"]["conditions"]:
        probes = results["probes"][cond]
        for lk in ["post_block0", "post_block3"]:
            if lk not in probes:
                continue
            id_sp = probes[lk]["id_test"]["spearman"]
            ood_sp = float(np.mean(
                [probes[lk][c]["spearman"] for c in ood])) if ood else 0.0
            print(f"  {cond:>14s} {lk}: ID rho={id_sp:.4f} "
                  f"OOD rho={ood_sp:.4f} "
                  f"retention={ood_sp / id_sp if id_sp else float('nan'):.3f}")
