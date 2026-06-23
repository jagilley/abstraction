"""FM intermediate probing: does the FM's internal computation mirror the RHM hierarchy?

If the FM dissociates representation from computation (paper claim), its
intermediate representations should encode progressively higher-level
hierarchy features — the same trajectory the main model's layers follow.

A 2-layer FM predicting post_block0 → post_block3 (3 main model blocks):
- fm_layer0 (input) = post_block0
- fm_layer1 (after FM block 0) should encode more hierarchy structure
- fm_layer2 (output) ≈ post_block3

Measurements at each representation point:
1. Feature/rule eta² at each hierarchy level
2. Linear probe accuracy for feature classification at each level
3. CKA between FM and main model intermediate layers

Uses the 2.7M model (6L/6H/192D) at L=6/m=4 from the regime trajectory.
"""

import json
import os
import sys
import modal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
    )
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)

app = modal.App("rhm-fm-intermediate-probing", image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_with_traces(rules, n_sequences, seed=999):
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features = []
    level_rules = []
    current = rng.integers(0, v, size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        level_rules.append(rc.copy())
        next_level = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            next_level[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = next_level
    sequences = current
    level_features.append(sequences.copy())
    return sequences, level_features, level_rules


def _eta2_between(data, labels, grand_mean):
    import numpy as np
    ss = 0.0
    for val in np.unique(labels):
        mask = labels == val
        group_mean = data[mask].mean(axis=0)
        ss += float(mask.sum()) * float(np.sum((group_mean - grand_mean) ** 2))
    return ss


def _compute_eta2(data, labels):
    import numpy as np
    gm = data.mean(axis=0)
    ss_total = float(np.sum((data - gm) ** 2))
    if ss_total < 1e-12:
        return 0.0
    return _eta2_between(data, labels, gm) / ss_total


def _linear_cka(X, Y, max_samples=50000):
    """Linear CKA (centered kernel alignment) between representation matrices."""
    import numpy as np
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, Y = X[idx], Y[idx]
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    YtX = Y.T @ X
    hsic_xy = float(np.sum(YtX ** 2))
    hsic_xx = float(np.sum((X.T @ X) ** 2))
    hsic_yy = float(np.sum((Y.T @ Y) ** 2))
    denom = (hsic_xx * hsic_yy) ** 0.5
    return hsic_xy / denom if denom > 1e-12 else 0.0


def _train_linear_probe(X_train, y_train, X_test, y_test, n_classes, d_model,
                         lr=0.01, n_steps=2000, batch_size=1024):
    """Train a linear classifier and return test accuracy."""
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe = torch.nn.Linear(d_model, n_classes).to(device)
    torch.nn.init.xavier_uniform_(probe.weight)
    torch.nn.init.zeros_(probe.bias)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    X_tr = torch.from_numpy(X_train).float().to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_te = torch.from_numpy(X_test).float().to(device)
    y_te = torch.from_numpy(y_test).long().to(device)

    for step in range(n_steps):
        idx = torch.randint(len(X_tr), (min(batch_size, len(X_tr)),))
        logits = probe(X_tr[idx])
        loss = F.cross_entropy(logits, y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = []
        for i in range(0, len(X_te), 4096):
            preds.append(probe(X_te[i:i + 4096]).argmax(dim=-1))
        preds = torch.cat(preds)
        return float((preds == y_te).float().mean())


def _fm_forward_with_intermediates(fwd_model, x):
    """Forward pass returning activation after each FM layer."""
    intermediates = [x]
    for block in fwd_model.blocks:
        x = block(x, fwd_model.causal_mask)
        intermediates.append(x)
    return intermediates


def _ensure_corpus(v, s, L, m, n_tokens):
    import numpy as np
    from rhm.rhm_data import make_corpus
    key = setting_key(v, s, L, m)
    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    if os.path.exists(corpus_path):
        existing = np.load(corpus_path)
        if len(existing) >= n_tokens:
            return
    corpus, rules, meta = make_corpus(v, s, L, m, n_tokens)
    out_dir = f"{DATA_DIR}/{key}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "corpus.npy"), corpus)
    for ell, r in enumerate(rules):
        np.save(os.path.join(out_dir, f"rules_L{ell}.npy"), r)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=10800,
    memory=32768,
)
def fm_intermediate_probing(
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    checkpoint_step: int = 20000,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    fwd_n_layer: int = 2, fwd_d_head: int = 24, fwd_n_head: int = 1,
    fwd_mlp_mult: float = 1.0,
    fm_train_steps: int = 5000,
    fm_lr: float = 1e-3,
    n_eval_sequences: int = 5000,
    batch_size: int = 64,
    probe_train_steps: int = 2000,
    fm_seed: int = 137,
):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel

    torch.manual_seed(fm_seed)

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"
    model_tag = f"{n_layer}L{n_head}H{n_embd}D"
    from_idx = int(predict_from.replace("post_block", ""))
    to_idx = int(predict_to.replace("post_block", ""))

    volume.reload()

    # --- Load main model ---
    ckpt_dir = f"{DATA_DIR}/{key}/regime_trajectory_{model_tag}"
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{checkpoint_step}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            f"Run rhm_regime_trajectory.py::run_m4_scaled first."
        )

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded main model: {key}, {model_tag}, step {checkpoint_step}")

    # --- Train FM ---
    _ensure_corpus(v, s, L, m, n_tokens)
    corpus = np.load(f"{DATA_DIR}/{key}/corpus.npy")
    corpus = torch.from_numpy(corpus[:n_tokens].astype(np.int64))
    train_data = corpus[:int(0.9 * len(corpus))]

    def get_batch(split_data):
        ix = torch.randint(len(split_data) - block_size - 1, (batch_size,))
        return torch.stack([split_data[i:i + block_size] for i in ix]).to(device)

    fwd_model = TransformerForwardModel(
        d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
        n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=block_size,
    ).to(device)
    fwd_n_params = sum(p.numel() for p in fwd_model.parameters())

    opt_fwd = torch.optim.AdamW(fwd_model.parameters(), lr=fm_lr, weight_decay=0.01)
    print(f"Training FM: {fwd_n_params / 1e3:.1f}K params, {fm_train_steps} steps")

    for fm_step in range(fm_train_steps):
        fwd_model.train()
        x = get_batch(train_data)
        with torch.no_grad():
            _, _, intermediates = model(x, return_intermediates=True)
            source = intermediates[predict_from]
            target = intermediates[predict_to]
        predicted = fwd_model(source)
        fwd_loss = F.mse_loss(predicted, target)
        opt_fwd.zero_grad()
        fwd_loss.backward()
        torch.nn.utils.clip_grad_norm_(fwd_model.parameters(), 1.0)
        opt_fwd.step()
        if fm_step % 1000 == 0:
            print(f"  FM step {fm_step}: mse={float(fwd_loss):.6f}")

    # --- Collect representations on traced sequences ---
    rules = [np.load(f"{DATA_DIR}/{key}/rules_L{ell}.npy") for ell in range(L)]
    eval_seqs, level_features, level_rules = _generate_with_traces(
        rules, n_eval_sequences)

    fwd_model.eval()
    eval_data = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)

    main_rep_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    fm_rep_names = [f"fm_layer{i}" for i in range(fwd_n_layer + 1)]
    all_rep_names = main_rep_names + fm_rep_names

    all_reps = {name: [] for name in all_rep_names}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch_x = eval_data[i:i + batch_size]
            if batch_x.shape[0] < 2:
                continue
            _, _, main_ints = model(batch_x, return_intermediates=True)
            for name in main_rep_names:
                all_reps[name].append(main_ints[name].cpu().numpy())

            source = main_ints[predict_from]
            fm_ints = _fm_forward_with_intermediates(fwd_model, source)
            for j, rep in enumerate(fm_ints):
                all_reps[f"fm_layer{j}"].append(rep.cpu().numpy())

    for name in all_rep_names:
        all_reps[name] = np.concatenate(all_reps[name], axis=0)

    N = all_reps[main_rep_names[0]].shape[0]
    level_features = [lf[:N] for lf in level_features]
    level_rules = [lr[:N] for lr in level_rules]

    # Sanity check
    target_flat = all_reps[predict_to].reshape(-1, n_embd)
    fm_out_flat = all_reps[f"fm_layer{fwd_n_layer}"].reshape(-1, n_embd)
    global_cosine = float(F.cosine_similarity(
        torch.from_numpy(fm_out_flat), torch.from_numpy(target_flat), dim=-1
    ).mean())
    print(f"\nFM output cosine with {predict_to}: {global_cosine:.4f}")

    # =======================================================================
    # 1. Feature eta² at each hierarchy level
    # =======================================================================
    print(f"\n{'=' * 110}")
    print(f"1. FEATURE eta² BY HIERARCHY LEVEL")
    print(f"{'=' * 110}")

    header = f"{'Representation':>20}" + "".join(
        f"  {'L' + str(ell):>8}" for ell in range(L))
    print(header)
    print("-" * len(header))

    results = {}

    for rep_name in all_rep_names:
        rep_flat = all_reps[rep_name].reshape(-1, n_embd)
        eta2_feat = {}
        eta2_rule = {}
        for ell in range(L):
            s_power = s ** (L - ell)
            ancestor_idx = np.arange(seq_len) // s_power
            feat_labels = level_features[ell][:, ancestor_idx].reshape(-1)
            rule_labels = level_rules[ell][:, ancestor_idx].reshape(-1)
            eta2_feat[ell] = _compute_eta2(rep_flat, feat_labels)
            eta2_rule[ell] = _compute_eta2(rep_flat, rule_labels)
        results[rep_name] = {"feature_eta2": eta2_feat, "rule_eta2": eta2_rule}
        row = f"{rep_name:>20}" + "".join(
            f"  {eta2_feat[ell]:>8.5f}" for ell in range(L))
        print(row)

    # Rule eta²
    print(f"\n{'=' * 110}")
    print(f"1b. RULE eta² BY HIERARCHY LEVEL")
    print(f"{'=' * 110}")
    print(header)
    print("-" * len(header))
    for rep_name in all_rep_names:
        eta2_rule = results[rep_name]["rule_eta2"]
        row = f"{rep_name:>20}" + "".join(
            f"  {eta2_rule[ell]:>8.5f}" for ell in range(L))
        print(row)

    # =======================================================================
    # 2. Linear probe accuracy for feature classification
    # =======================================================================
    print(f"\n{'=' * 110}")
    print(f"2. LINEAR PROBE ACCURACY (feature classification, chance = {1.0 / v:.3f})")
    print(f"{'=' * 110}")
    print(header)
    print("-" * len(header))

    n_train_seqs = int(0.8 * N)

    for rep_name in all_rep_names:
        probe_accs = {}
        for ell in range(L):
            s_power = s ** (L - ell)
            ancestor_idx = np.arange(seq_len) // s_power
            feat_labels = level_features[ell][:, ancestor_idx]

            X_train = all_reps[rep_name][:n_train_seqs].reshape(-1, n_embd)
            y_train = feat_labels[:n_train_seqs].reshape(-1)
            X_test = all_reps[rep_name][n_train_seqs:].reshape(-1, n_embd)
            y_test = feat_labels[n_train_seqs:].reshape(-1)

            acc = _train_linear_probe(
                X_train, y_train, X_test, y_test,
                n_classes=v, d_model=n_embd, n_steps=probe_train_steps)
            probe_accs[ell] = acc

        results[rep_name]["probe_accuracy"] = probe_accs
        row = f"{rep_name:>20}" + "".join(
            f"  {probe_accs[ell]:>8.4f}" for ell in range(L))
        print(row)

    # =======================================================================
    # 3. CKA between FM and main model layers
    # =======================================================================
    print(f"\n{'=' * 110}")
    print(f"3. CKA: FM LAYERS vs MAIN MODEL LAYERS")
    print(f"{'=' * 110}")

    cka_header = f"{'':>15}" + "".join(
        f"  {name:>14}" for name in main_rep_names)
    print(cka_header)
    print("-" * len(cka_header))

    cka_matrix = {}
    for fm_name in fm_rep_names:
        fm_flat = all_reps[fm_name].reshape(-1, n_embd)
        cka_row = {}
        for main_name in main_rep_names:
            main_flat = all_reps[main_name].reshape(-1, n_embd)
            cka_val = _linear_cka(fm_flat, main_flat)
            cka_row[main_name] = cka_val
        cka_matrix[fm_name] = cka_row
        row = f"{fm_name:>15}" + "".join(
            f"  {cka_row[name]:>14.4f}" for name in main_rep_names)
        print(row)

    print(f"\n  Closest main model layer:")
    for fm_name in fm_rep_names:
        best = max(cka_matrix[fm_name], key=cka_matrix[fm_name].get)
        print(f"    {fm_name} -> {best} (CKA={cka_matrix[fm_name][best]:.4f})")

    # =======================================================================
    # 4. Delta analysis: what does each block add?
    # =======================================================================
    print(f"\n{'=' * 110}")
    print(f"4. DELTA ANALYSIS (positive = block ADDED hierarchy structure)")
    print(f"{'=' * 110}")

    delta_header = f"{'Delta':>30}" + "".join(
        f"  {'L' + str(ell):>8}" for ell in range(L))
    print(delta_header)
    print("-" * len(delta_header))

    for metric_key, metric_label in [
        ("feature_eta2", "feat_eta2"),
        ("probe_accuracy", "probe_acc"),
    ]:
        # FM block deltas
        for b in range(fwd_n_layer):
            pre = results[f"fm_layer{b}"][metric_key]
            post = results[f"fm_layer{b + 1}"][metric_key]
            deltas = {ell: post[ell] - pre[ell] for ell in range(L)}
            label = f"FM_blk{b} {metric_label}"
            row = f"{label:>30}" + "".join(
                f"  {'+' if deltas[ell] >= 0 else ''}{deltas[ell]:>7.5f}"
                for ell in range(L))
            print(row)

        # Main model block deltas (within the predicted gap)
        for b in range(from_idx, to_idx):
            pre = results[f"post_block{b}"][metric_key]
            post = results[f"post_block{b + 1}"][metric_key]
            deltas = {ell: post[ell] - pre[ell] for ell in range(L)}
            label = f"Main_blk{b + 1} {metric_label}"
            row = f"{label:>30}" + "".join(
                f"  {'+' if deltas[ell] >= 0 else ''}{deltas[ell]:>7.5f}"
                for ell in range(L))
            print(row)

        print()

    # =======================================================================
    # 5. FM intermediate position on the main model trajectory
    # =======================================================================
    print(f"\n{'=' * 110}")
    print(f"5. FM INTERMEDIATE POSITION ON MAIN MODEL TRAJECTORY")
    print(f"   For each hierarchy level, where does fm_layer1 fall between")
    print(f"   post_block{from_idx} (FM input) and post_block{to_idx} (FM target)?")
    print(f"{'=' * 110}")

    interp_header = f"{'Metric':>20}" + "".join(
        f"  {'L' + str(ell):>8}" for ell in range(L))
    print(interp_header)
    print("-" * len(interp_header))

    for metric_key, metric_label in [
        ("feature_eta2", "feat_eta2"),
        ("probe_accuracy", "probe_acc"),
    ]:
        input_vals = results[f"post_block{from_idx}"][metric_key]
        target_vals = results[f"post_block{to_idx}"][metric_key]
        mid_vals = results["fm_layer1"][metric_key]

        fracs = {}
        for ell in range(L):
            span = target_vals[ell] - input_vals[ell]
            if abs(span) > 1e-8:
                fracs[ell] = (mid_vals[ell] - input_vals[ell]) / span
            else:
                fracs[ell] = float('nan')

        row = f"{metric_label + ' frac':>20}" + "".join(
            f"  {fracs[ell]:>8.3f}" for ell in range(L))
        print(row)

    print(f"\n  frac=0.0 means fm_layer1 = post_block{from_idx} (no progress)")
    print(f"  frac=0.33 means fm_layer1 is 1/3 of the way to post_block{to_idx}")
    print(f"  frac=1.0 means fm_layer1 = post_block{to_idx} (full gap)")

    # =======================================================================
    # Save
    # =======================================================================
    save_dir = f"{DATA_DIR}/rhm_fm_intermediate_probing"
    os.makedirs(save_dir, exist_ok=True)

    result = {
        "experiment": "FM intermediate probing",
        "params": {
            "v": v, "s": s, "L": L, "m": m, "seq_len": seq_len,
            "model": model_tag, "checkpoint_step": checkpoint_step,
            "fm": f"{fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d/mlp{fwd_mlp_mult}",
            "fm_params": fwd_n_params,
            "gap": f"{predict_from}->{predict_to}",
            "n_eval_sequences": n_eval_sequences,
            "fm_cosine": global_cosine,
        },
        "representations": {
            name: {
                "feature_eta2": {str(k): v for k, v in data["feature_eta2"].items()},
                "rule_eta2": {str(k): v for k, v in data["rule_eta2"].items()},
                "probe_accuracy": (
                    {str(k): v for k, v in data["probe_accuracy"].items()}
                    if "probe_accuracy" in data else {}
                ),
            }
            for name, data in results.items()
        },
        "cka_matrix": cka_matrix,
    }

    fname = f"{key}_{model_tag}_step{checkpoint_step}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {save_dir}/{fname}")
    return result
