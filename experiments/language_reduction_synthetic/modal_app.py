"""Modal app for synthetic RHM scaling law experiments.

Stages:
  generate      — generate RHM rules and corpus for a given (v, s, L, m)
  train         — train AR model at a given (L, m, P)
  sweep         — train at multiple P values for one (L, m) setting
  full-sweep    — sweep across multiple L and m values
  measure       — compute empirical alpha from trained models

Usage (via local entrypoint):
  modal run language_reduction_synthetic/modal_app.py --stage generate --depth 6 --mult 4
  modal run language_reduction_synthetic/modal_app.py --stage train --depth 6 --mult 4 --n-tokens 100000
  modal run language_reduction_synthetic/modal_app.py --stage sweep --depth 6 --mult 4
  modal run language_reduction_synthetic/modal_app.py --stage full-sweep

Usage (direct function invocation, for --detach):
  modal run --detach language_reduction_synthetic/modal_app.py::full_sweep --l-values "4,6,8" --m-values "2,4,8"
  modal run --detach language_reduction_synthetic/modal_app.py::sweep --L 6 --m 4
"""

import json
import modal

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
    )
    .add_local_python_source("language_reduction_synthetic")
)

volume = modal.Volume.from_name("rhm-scaling-data", create_if_missing=True)
app = modal.App("rhm-scaling", image=image)


def _setting_key(v, s, L, m):
    return f"v{v}_s{s}_L{L}_m{m}"


# ---------------------------------------------------------------------------
# Stage 1: Generate corpus
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=600,
    memory=16384,
)
def generate_corpus(v: int = 8, s: int = 2, L: int = 6, m: int = 4,
                    n_tokens: int = 20_000_000, rule_seed: int = 0):
    import os
    import numpy as np
    from language_reduction_synthetic.rhm import make_corpus

    key = _setting_key(v, s, L, m)
    out_dir = f"{DATA_DIR}/{key}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating RHM corpus: {key}, n_tokens={n_tokens:,}")
    corpus, rules, meta = make_corpus(v, s, L, m, n_tokens, rule_seed=rule_seed)

    np.save(os.path.join(out_dir, "corpus.npy"), corpus)
    for ell, r in enumerate(rules):
        np.save(os.path.join(out_dir, f"rules_L{ell}.npy"), r)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    volume.commit()
    print(f"Saved {len(corpus):,} tokens to {out_dir}")
    print(f"  seq_len={meta['seq_len']}, n_sequences={meta['n_sequences']}")
    return meta


# ---------------------------------------------------------------------------
# Stage 2: Train model
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    gpu="T4",
    timeout=3600,
    memory=16384,
)
def train_model(v: int = 8, s: int = 2, L: int = 6, m: int = 4,
                n_tokens: int = 100_000,
                n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
                batch_size: int = 64, lr: float = 3e-4,
                n_steps: int = 10_000, eval_interval: int = 200,
                n_eval_batches: int = 10):
    import os
    import torch
    import numpy as np
    from language_reduction_synthetic.model import GPT

    key = _setting_key(v, s, L, m)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = s ** L
    block_size = seq_len

    volume.reload()
    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}. Run 'generate' first.")

    data = np.load(corpus_path)
    data = torch.from_numpy(data[:n_tokens].astype(np.int64))
    print(f"Training: {key}, P={n_tokens:,}, T={block_size}, device={device}")
    print(f"  Corpus has {len(data):,} tokens ({len(data) // seq_len} sequences)")

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # auto-scale steps: 5 epochs, floor 2000, cap 20000
    tokens_per_step = batch_size * block_size
    tokens_per_epoch = len(train_data)
    steps_per_epoch = max(1, tokens_per_epoch // tokens_per_step)
    auto_steps = min(20000, max(2000, 5 * steps_per_epoch))
    if n_steps == 10_000:
        n_steps = auto_steps
    print(f"  steps_per_epoch={steps_per_epoch}, n_steps={n_steps}")

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_batch(split_data, bs, bl):
        ix = torch.randint(len(split_data) - bl - 1, (bs,))
        x = torch.stack([split_data[i:i + bl] for i in ix])
        y = torch.stack([split_data[i + 1:i + bl + 1] for i in ix])
        return x.to(device), y.to(device)

    losses = []
    val_losses = []
    best_val_loss = float("inf")

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
                for _ in range(n_eval_batches):
                    vx, vy = get_batch(val_data, batch_size, block_size)
                    _, vl = model(vx, vy)
                    vl_accum += float(vl)
                val_loss = vl_accum / n_eval_batches
            losses.append((step, float(loss)))
            val_losses.append((step, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print(f"  step {step:6d}: train={loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}")

    save_dir = f"{DATA_DIR}/{key}/models/P_{n_tokens}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    result = {
        "setting": key,
        "v": v, "s": s, "L": L, "m": m,
        "n_tokens": n_tokens,
        "block_size": block_size,
        "seq_len": seq_len,
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "n_steps": n_steps,
        "final_train_loss": float(losses[-1][1]),
        "final_val_loss": float(val_losses[-1][1]),
        "best_val_loss": best_val_loss,
        "train_curve": losses,
        "val_curve": val_losses,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    volume.commit()
    print(f"Done. best_val_loss={best_val_loss:.4f}")
    return result


# ---------------------------------------------------------------------------
# Stage 3: Sweep over P values for one setting
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=3600,
    memory=4096,
)
def sweep(v: int = 8, s: int = 2, L: int = 6, m: int = 4,
          n_layer: int = 4, n_head: int = 4, n_embd: int = 128):
    """Train at multiple P values and compute empirical scaling exponent."""
    seq_len = s ** L
    min_p = max(1000, 50 * seq_len)
    # 6 P values spanning ~3 orders of magnitude
    p_values = []
    for exp in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        p = int(min_p * 10 ** exp)
        p_values.append(p)
    # cap at 20M to keep compute reasonable
    p_values = [p for p in p_values if p <= 20_000_000]

    print(f"Sweep for {_setting_key(v, s, L, m)}: P = {p_values}")
    print(f"  Model: {n_layer}L {n_head}H {n_embd}D, block_size={seq_len}")

    # ensure corpus exists (skip if already generated by full_sweep)
    import os
    volume.reload()
    corpus_path = f"{DATA_DIR}/{_setting_key(v, s, L, m)}/corpus.npy"
    if not os.path.exists(corpus_path):
        max_p = max(p_values)
        generate_corpus.remote(v=v, s=s, L=L, m=m, n_tokens=int(max_p * 1.2))

    # train in parallel
    handles = []
    for p in p_values:
        h = train_model.spawn(
            v=v, s=s, L=L, m=m, n_tokens=p,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        )
        handles.append(h)

    results = [h.get() for h in handles]

    # compute empirical scaling
    loss_vs_P = [(r["n_tokens"], r["best_val_loss"]) for r in results]
    from language_reduction_synthetic.measure import measure_empirical_scaling
    scaling = measure_empirical_scaling(loss_vs_P)

    summary = {
        "setting": _setting_key(v, s, L, m),
        "v": v, "s": s, "L": L, "m": m,
        "p_values": p_values,
        "loss_vs_P": loss_vs_P,
        "scaling": scaling,
    }

    import os
    key = _setting_key(v, s, L, m)
    results_dir = f"{DATA_DIR}/{key}/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "scaling.json"), "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    print(f"\n{'='*60}")
    print(f"SCALING RESULT: {key}")
    print(f"  alpha_empirical = {scaling['alpha_empirical']:.4f}")
    print(f"  R² = {scaling['r2']:.4f}")
    print(f"  P values: {p_values}")
    print(f"  Losses: {[f'{l:.4f}' for _, l in loss_vs_P]}")
    print(f"{'='*60}")

    return summary


# ---------------------------------------------------------------------------
# Stage 4: Full sweep across DGP parameters
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=7200,
    memory=4096,
)
def full_sweep(
    l_values: str = "4,6,8",
    m_values: str = "2,4,8",
    v: int = 8, s: int = 2,
    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
):
    """Sweep across L (depth) and m (multiplicity) values."""
    Ls = [int(x) for x in l_values.split(",")]
    ms = [int(x) for x in m_values.split(",")]

    print(f"Full sweep: L={Ls}, m={ms}, v={v}, s={s}")

    # Phase 1: generate all corpora (parallel, wait for all)
    print("Phase 1: generating corpora...")
    gen_handles = []
    for L in Ls:
        for m in ms:
            seq_len = s ** L
            min_p = max(1000, 50 * seq_len)
            max_p = int(min_p * 10 ** 3.0)
            max_p = min(max_p, 20_000_000)
            h = generate_corpus.spawn(v=v, s=s, L=L, m=m,
                                      n_tokens=int(max_p * 1.2))
            gen_handles.append((L, m, h))
    for L, m, h in gen_handles:
        h.get()
        print(f"  Corpus ready: {_setting_key(v, s, L, m)}")

    # Phase 2: run all sweeps (parallel)
    print("Phase 2: running sweeps...")
    handles = []
    for L in Ls:
        for m in ms:
            h = sweep.spawn(v=v, s=s, L=L, m=m,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd)
            handles.append((L, m, h))

    all_results = {}
    for L, m, h in handles:
        key = _setting_key(v, s, L, m)
        try:
            result = h.get()
            all_results[key] = result
            alpha = result["scaling"]["alpha_empirical"]
            print(f"  {key}: alpha={alpha:.4f}")
        except Exception as e:
            print(f"  {key}: FAILED — {e}")

    import os
    results_dir = f"{DATA_DIR}/sweep_results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "full_sweep.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    volume.commit()

    print(f"\n{'='*60}")
    print("FULL SWEEP RESULTS")
    print(f"{'Setting':<20} {'alpha_D':>8} {'R²':>6}")
    print("-" * 36)
    for key, result in sorted(all_results.items()):
        alpha = result["scaling"]["alpha_empirical"]
        r2 = result["scaling"]["r2"]
        print(f"{key:<20} {alpha:>8.4f} {r2:>6.3f}")
    print(f"{'='*60}")

    return all_results


# ---------------------------------------------------------------------------
# Stage 5: Measure scaling from existing trained models
# ---------------------------------------------------------------------------
@app.function(
    volumes={DATA_DIR: volume},
    timeout=300,
    memory=4096,
)
def measure_scaling(v: int = 8, s: int = 2, L: int = 6, m: int = 4):
    """Compute empirical alpha from already-trained models."""
    import os
    import glob
    from language_reduction_synthetic.measure import measure_empirical_scaling

    key = _setting_key(v, s, L, m)
    models_dir = f"{DATA_DIR}/{key}/models"
    if not os.path.exists(models_dir):
        print(f"No models found at {models_dir}")
        return None

    loss_vs_P = []
    for p_dir in sorted(glob.glob(os.path.join(models_dir, "P_*"))):
        results_path = os.path.join(p_dir, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            res = json.load(f)
        loss_vs_P.append((res["n_tokens"], res["best_val_loss"]))

    if len(loss_vs_P) < 3:
        print(f"Only {len(loss_vs_P)} P values found, need at least 3")
        return None

    scaling = measure_empirical_scaling(loss_vs_P)
    print(f"{key}: alpha={scaling['alpha_empirical']:.4f}, R²={scaling['r2']:.4f}")
    return scaling


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    stage: str = "generate",
    vocab: int = 8,
    branch: int = 2,
    depth: int = 6,
    mult: int = 4,
    n_tokens: int = 20_000_000,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    rule_seed: int = 0,
    depths: str = "4,6,8",
    mults: str = "2,4,8",
):
    """CLI entrypoint. RHM params: --vocab (v), --branch (s), --depth (L), --mult (m)."""
    v, s, L, m = vocab, branch, depth, mult
    if stage == "generate":
        generate_corpus.remote(v=v, s=s, L=L, m=m, n_tokens=n_tokens,
                              rule_seed=rule_seed)
    elif stage == "train":
        train_model.remote(v=v, s=s, L=L, m=m, n_tokens=n_tokens,
                          n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    elif stage == "sweep":
        sweep.remote(v=v, s=s, L=L, m=m,
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    elif stage == "full-sweep":
        full_sweep.remote(l_values=depths, m_values=mults,
                         v=v, s=s,
                         n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    elif stage == "measure":
        measure_scaling.remote(v=v, s=s, L=L, m=m)
    else:
        print(f"Unknown stage: {stage}")
        print("Available: generate, train, sweep, full-sweep, measure")
