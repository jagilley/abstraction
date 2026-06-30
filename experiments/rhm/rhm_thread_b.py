"""Thread B: does a trained transformer do joint inference, or stall at local clustering?

Threads A / A.5 / BP (see RHM_DEEP_COMPOSITION_README.md) established, training-free,
two reference lines for recovering the latent RHM feature at each hierarchy level
from the leaves:
  - greedy floor  : iterated hard cluster-and-lift -- cascades to ~chance at depth.
  - BP ceiling     : optimal tree belief-propagation (known rules) -- the most any
                     inference could extract.

This experiment trains a standard (non-looped, causal) GPT on RHM next-token
prediction, then probes each block's representation for the true latent ancestor at
every level, and asks where the probe curve lands between the two reference lines:
  - tracks greedy  => the model only does shallow local pattern-matching.
  - tracks BP       => SGD discovered genuine joint inference of the hierarchy.

First run (20k steps, linear probe) found: the model does near-optimal joint
inference up to a frontier (~3 levels: beats greedy, approaches BP), then falls off
a cliff (below greedy at d4, chance at the root), with the frontier still advancing
bottom-up while val loss plateaus. This script adds:
  (1) long runs to see whether the frontier keeps advancing / groks (n_steps), and
  (3) an MLP probe alongside the linear probe, to test whether high levels are truly
      absent vs. non-linearly buried. `probe_only` re-probes existing checkpoints.

DGP: distinct-rule RHM at v=16, m=4, s=2, L=6 (occupancy m/v = 0.25), BP ceiling has
real depth headroom (root ~0.78). Probe point: the LAST token of each aligned eval
sequence (a causal model has seen the whole sequence there -> fair vs full BP).

Run:
  modal run --detach -m rhm.rhm_thread_b::thread_b                      # 20k baseline
  modal run --detach -m rhm.rhm_thread_b::thread_b --n-steps 100000 \
      --ckpt-steps "0,1000,2000,5000,10000,20000,40000,70000,100000"    # long run (#1)
  modal run --detach -m rhm.rhm_thread_b::probe_only                    # re-probe (#3)
"""

import glob
import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
)
app = modal.App("rhm-thread-b", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def _generate_with_traces(rules, n_sequences, seed):
    """Aligned sequences + the true latent feature at every node of every level.
    level_features[ell] has shape (n_sequences, s^ell); [L] is the leaves."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features = []
    current = rng.integers(0, v, size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        nxt = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = nxt
    level_features.append(current.copy())  # leaves
    return current, level_features


def _probe_acc(X, y, v, device, steps, lr, hidden=0, wd=1e-4, train_frac=0.7):
    """Probe the decodability of label y from X. hidden=0 -> linear (logistic
    regression); hidden>0 -> 1-layer MLP (tests non-linear decodability).
    Returns (test_acc, train_acc). Features standardized by train stats."""
    import torch
    import torch.nn as nn
    n = X.shape[0]
    split = int(train_frac * n)
    mu, sd = X[:split].mean(0, keepdim=True), X[:split].std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    Xtr, ytr, Xte, yte = Xn[:split], y[:split], Xn[split:], y[split:]
    if hidden:
        clf = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                            nn.Linear(hidden, v)).to(device)
    else:
        clf = nn.Linear(X.shape[1], v).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(clf(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        te = (clf(Xte).argmax(1) == yte).float().mean().item()
        tr = (clf(Xtr).argmax(1) == ytr).float().mean().item()
    return te, tr


def _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device):
    """Aligned eval sequences + last-position ancestor labels per level."""
    import numpy as np
    import torch
    seq_len = s ** L
    eval_seqs, level_features = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    last_anc = {ell: (seq_len - 1) // (s ** (L - ell)) for ell in range(L)}
    y_level = {ell: torch.from_numpy(level_features[ell][:, last_anc[ell]].astype(np.int64)).to(device)
               for ell in range(L)}
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    return eval_x, y_level, block_names


def _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level, block_names, L, v,
                       device, probe_steps, probe_lr, mlp_hidden, mlp_steps):
    """Probe every (checkpoint x block x level) with a linear probe and (if
    mlp_hidden>0) an MLP probe. Prints best-by-level trajectories; returns dict."""
    import torch

    def last_pos_acts(step):
        sd = torch.load(f"{ckpt_dir}/ckpt_step{step}.pt", map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        acc = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, len(eval_x), 256):
                _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
                for b in block_names:
                    acc[b].append(inter[b][:, -1, :].float())
        return {b: torch.cat(acc[b], 0) for b in block_names}

    chance = 1.0 / v
    print(f"\n{'='*72}\nPROBE: decodability of latent ancestor at last position")
    print(f"chance = {chance:.4f}   (level ell -> depth d = L-ell)\n{'='*72}")
    out = {}
    for step in saved:
        acts = last_pos_acts(step)
        lin, mlp = {}, {}
        for ell in range(L):
            lin[ell] = {b: _probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr)[0]
                        for b in block_names}
            if mlp_hidden:
                mlp[ell] = {b: _probe_acc(acts[b], y_level[ell], v, device, mlp_steps,
                                          1e-3, hidden=mlp_hidden, wd=1e-3)[0]
                            for b in block_names}
        lin_best = {ell: max(lin[ell].values()) for ell in range(L)}
        out[step] = {"linear_grid": lin, "linear_best": lin_best}
        print(f"  step {step:6d} | LIN best:  " +
              "  ".join(f"d{L-ell}:{lin_best[ell]:.3f}" for ell in range(L)))
        if mlp_hidden:
            mlp_best = {ell: max(mlp[ell].values()) for ell in range(L)}
            out[step]["mlp_grid"] = mlp
            out[step]["mlp_best"] = mlp_best
            print(f"  step {step:6d} | MLP best:  " +
                  "  ".join(f"d{L-ell}:{mlp_best[ell]:.3f}" for ell in range(L)))

    last = saved[-1]
    print(f"\nFinal checkpoint (step {last}) -- LINEAR test acc by block x level:")
    fg = out[last]["linear_grid"]
    print(f"{'block':>12} | " + "  ".join(f"d{L-ell}" for ell in range(L)))
    for b in block_names:
        print(f"{b:>12} | " + "  ".join(f"{fg[ell][b]:.3f}" for ell in range(L)))
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=36000, memory=32768)
def thread_b(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 20_000_000, rule_seed: int = 0, seq_seed: int = 1,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
    n_steps: int = 20000, eval_interval: int = 500,
    ckpt_steps: str = "0,500,1000,2000,5000,10000,20000",
    n_eval_sequences: int = 12000, probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, eval_seed: int = 999,
):
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct, generate_sequences_batched

    L = depth
    seq_len = s ** L
    block_size = seq_len
    device = "cuda"
    key = tb_key(v, s, L, m)
    data_dir = f"{DATA_DIR}/{key}"
    ckpt_dir = f"{data_dir}/thread_b_{n_layer}L{n_head}H{n_embd}D"
    os.makedirs(ckpt_dir, exist_ok=True)
    want_ckpts = sorted({min(int(x), n_steps - 1) for x in ckpt_steps.split(",")})

    volume.reload()
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    corpus_path = f"{data_dir}/corpus.npy"
    if not os.path.exists(corpus_path) or len(np.load(corpus_path, mmap_mode="r")) < n_tokens:
        os.makedirs(data_dir, exist_ok=True)
        n_seq = (n_tokens + seq_len - 1) // seq_len
        print(f"Generating distinct-rule corpus: {key}, {n_tokens:,} tokens")
        seqs = generate_sequences_batched(rules, n_seq, seed=seq_seed)
        np.save(corpus_path, seqs.reshape(-1)[:n_tokens])
        for ell, r in enumerate(rules):
            np.save(f"{data_dir}/rules_L{ell}.npy", r)
        volume.commit()
    data = torch.from_numpy(np.load(corpus_path)[:n_tokens].astype(np.int64))
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Thread B: {key}  P={n_tokens:,}  T={block_size}  model={n_layer}L/{n_head}H/{n_embd}D")
    print(f"  occupancy m/v^(s-1) = {m / v ** (s - 1):.3f}  chance = {1.0/v:.4f}  n_steps={n_steps}")

    def get_batch(d):
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    saved = []
    for step in range(n_steps):
        if step in want_ckpts:
            torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{step}.pt")
            saved.append(step)
        model.train()
        x, y = get_batch(train_data)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                vl = np.mean([float(model(*get_batch(val_data))[1]) for _ in range(10)])
            print(f"  step {step:6d}: train={float(loss):.4f} val={vl:.4f}")
    final_step = n_steps - 1
    torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{final_step}.pt")
    saved = sorted(set(saved + [final_step]))
    volume.commit()

    eval_x, y_level, block_names = _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device)
    probes = _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level, block_names,
                                L, v, device, probe_steps, probe_lr, mlp_hidden, mlp_steps)

    results = {"config": dict(v=v, s=s, L=L, m=m, n_tokens=n_tokens, rule_seed=rule_seed,
                              n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_steps=n_steps,
                              chance=1.0 / v, occupancy=m / v ** (s - 1)),
               "checkpoints": probes}
    save_dir = f"{DATA_DIR}/rhm_thread_b"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{key}_{n_layer}L{n_head}H{n_embd}D_P{n_tokens}_S{n_steps}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return results


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=21600, memory=32768)
def oracle_aux(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
    n_steps: int = 20000, lam: float = 1.0, pool_size: int = 200000,
    eval_interval: int = 1000, ckpt_steps: str = "0,5000,20000",
    n_eval_sequences: int = 12000, probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, eval_seed: int = 999, data_seed: int = 7,
):
    """Signal-vs-capacity diagnostic. Holds architecture fixed; the only thing that
    differs between conditions is whether the high-level latent SIGNAL is supplied
    directly (oracle aux) on top of NTP. If ntp_aux lifts d4/d5/root toward BP while
    ntp_only reproduces Thread B's saturation, the bottleneck is signal, not capacity.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct

    L = depth
    T = s ** L
    device = "cuda"
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    want_ckpts = sorted({min(int(x), n_steps - 1) for x in ckpt_steps.split(",")})

    # --- training pool: aligned sequences + per-position ancestor labels ---
    print(f"oracle_aux: {key}  {n_layer}L/{n_head}H/{n_embd}D  pool={pool_size:,}  lam={lam}")
    pool_seqs, pool_lf = _generate_with_traces(rules, pool_size, data_seed)
    pool_x = torch.from_numpy(pool_seqs.astype(np.int64))                       # (P, T) cpu
    pool_anc = [torch.from_numpy(pool_lf[ell].astype(np.int64)) for ell in range(L)]  # (P, s^ell)
    anc_idx = [torch.arange(T) // (s ** (L - ell)) for ell in range(L)]          # pos -> node
    spanend = [torch.tensor([p for p in range(T) if (p + 1) % (s ** (L - ell)) == 0])
               for ell in range(L)]                                              # completed positions
    sup_blocks = [f"post_block{i}" for i in range(n_layer)]

    eval_x, y_level, block_names = _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device)

    def get_batch():
        idx = torch.randint(0, pool_size, (batch_size,))
        x = pool_x[idx].to(device)
        labels = [pool_anc[ell][idx][:, anc_idx[ell]].to(device) for ell in range(L)]  # (B,T)
        return x, labels

    def aux_loss(inter, labels):
        per_level = []
        head_out = {b: aux_heads[b](inter[b]).view(batch_size, T, L, v) for b in sup_blocks}
        for ell in range(L):
            se = spanend[ell]
            ces = [F.cross_entropy(head_out[b][:, se, ell, :].reshape(-1, v),
                                   labels[ell][:, se].reshape(-1)) for b in sup_blocks]
            per_level.append(torch.stack(ces).mean())          # mean over blocks
        return torch.stack(per_level).mean()                    # equal weight per level

    cond_results = {}
    for cond in ["ntp_only", "ntp_aux"]:
        use_aux = cond == "ntp_aux"
        torch.manual_seed(0)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        params = list(model.parameters())
        aux_heads = None
        if use_aux:
            aux_heads = nn.ModuleDict({b: nn.Linear(n_embd, L * v) for b in sup_blocks}).to(device)
            params += list(aux_heads.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        ckpt_dir = f"{DATA_DIR}/{key}/oracle_aux_{cond}_{n_layer}L{n_head}H{n_embd}D"
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\n--- training {cond} ---")
        saved = []
        for step in range(n_steps):
            if step in want_ckpts:
                torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{step}.pt"); saved.append(step)
            model.train()
            x, labels = get_batch()
            logits, _, inter = model(x, targets=None, return_intermediates=True)
            ntp = F.cross_entropy(logits[:, :-1].reshape(-1, v), x[:, 1:].reshape(-1))
            loss = ntp + lam * aux_loss(inter, labels) if use_aux else ntp
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % eval_interval == 0 or step == n_steps - 1:
                extra = f" aux={float(aux_loss(inter, labels)):.3f}" if use_aux else ""
                print(f"  step {step:6d}: ntp={float(ntp):.4f}{extra}")
        torch.save(model.state_dict(), f"{ckpt_dir}/ckpt_step{n_steps-1}.pt")
        saved = sorted(set(saved + [n_steps - 1]))
        volume.commit()
        cond_results[cond] = _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level,
                                                 block_names, L, v, device, probe_steps,
                                                 probe_lr, mlp_hidden, mlp_steps)

    # --- head-to-head final comparison ---
    print(f"\n{'='*72}\nSIGNAL-vs-CAPACITY: final best-probe by depth (chance={1.0/v:.4f})")
    for kind in ["linear_best", "mlp_best"]:
        print(f"\n[{kind}]   (level ell -> depth d = L-ell)")
        print(f"{'cond':>10} | " + "  ".join(f"d{L-ell}" for ell in range(L)))
        for cond in ["ntp_only", "ntp_aux"]:
            last = max(cond_results[cond].keys())
            best = cond_results[cond][last][kind]
            print(f"{cond:>10} | " + "  ".join(f"{best[ell]:.3f}" for ell in range(L)))

    save_dir = f"{DATA_DIR}/rhm_thread_b"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"oracle_aux_{key}_{n_layer}L{n_head}H{n_embd}D_lam{lam}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump({"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, lam=lam,
                                  n_steps=n_steps, chance=1.0 / v), "conditions": cond_results},
                  f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return cond_results


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=7200, memory=32768)
def probe_only(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    n_eval_sequences: int = 12000, probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, eval_seed: int = 999,
):
    """Re-probe existing checkpoints (no training) with linear + MLP probes (#3)."""
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct

    L = depth
    block_size = s ** L
    device = "cuda"
    key = tb_key(v, s, L, m)
    ckpt_dir = f"{DATA_DIR}/{key}/thread_b_{n_layer}L{n_head}H{n_embd}D"
    volume.reload()
    saved = sorted(int(p.split("step")[1].split(".")[0])
                   for p in glob.glob(f"{ckpt_dir}/ckpt_step*.pt"))
    if not saved:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}; run thread_b first.")
    print(f"probe_only: {key}  {n_layer}L/{n_head}H/{n_embd}D  checkpoints={saved}")

    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    model = GPT(v, block_size, n_layer, n_head, n_embd).to(device)
    eval_x, y_level, block_names = _build_eval(rules, n_eval_sequences, eval_seed, v, s, L, n_layer, device)
    probes = _probe_checkpoints(model, ckpt_dir, saved, eval_x, y_level, block_names,
                                L, v, device, probe_steps, probe_lr, mlp_hidden, mlp_steps)

    save_dir = f"{DATA_DIR}/rhm_thread_b"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"probe_only_{key}_{n_layer}L{n_head}H{n_embd}D.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump({"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, chance=1.0 / v),
                   "checkpoints": probes}, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return probes
