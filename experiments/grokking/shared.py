"""[grokking] Shared Modal infrastructure, model, data and training loop.

Everything imported by `mint.py` and `rung.py` lives here: the Modal app / image / volume,
the vanilla grokking MLP and its data split (forked verbatim, see the fork notice below), the
full-batch training loop with snapshots, and the one-hot / feature builders.

Run every entrypoint from `experiments/` (so `grokking` is importable), under
MODAL_PROFILE=chromatic. CPU containers only (no `gpu=`): the net is ~54k params and the batch
is the whole train split.
"""

import os
import time

import modal
import numpy as np

DATA_DIR = "/data"
APP_NAME = "grokking-mint"
VOLUME_NAME = "grokking-mint-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4")
    .pip_install("torch==2.7.0", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("grokking")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

# The vanilla recipe (experiments/inverse_dynamics/grokking_fwd_vs_inv.py defaults).
P = 97
HIDDEN = (128, 128)
TRAIN_FRAC = 0.3
LR = 1e-3
WEIGHT_DECAY = 1.0
SEED = 42
N_EPOCHS = 40000
N_FREQ = (P - 1) // 2          # 48 folded nonzero frequencies for prime p


try:  # the local Modal client has no torch; the containers do
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except ImportError:
    torch = nn = F = None


# =============================================================================
# FORK NOTICE. `GrokMLP` and `generate_data` below are forked VERBATIM from
# experiments/inverse_dynamics/grokking_fwd_vs_inv.py (itself verbatim from
# fer/experiments/zipfian_grokking/cnb_self_regulation/modal_app.py), 2026-09-22.
# Changes: (i) the class sits under `if torch is not None:` so the torch-less local Modal
# client can import this module (body re-indented, otherwise unchanged); (ii) GrokMLP takes
# an optional `in_dim` (rung.py swaps the input alphabet; None -> 2p, the original). Gate G-1
# in mint.py asserts `split_pairs` reproduces `generate_data`'s split exactly.
# =============================================================================

if torch is not None:

    class GrokMLP(nn.Module):
        """2-layer MLP for modular arithmetic grokking. Non-residual by design."""

        def __init__(self, p=97, hidden_dims=(128, 128), in_dim=None):
            super().__init__()
            self.p = p
            self.layer0 = nn.Linear(2 * p if in_dim is None else in_dim, hidden_dims[0])
            self.layer1 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.head = nn.Linear(hidden_dims[1], p)

        def forward(self, x):
            x = F.relu(self.layer0(x))
            x = F.relu(self.layer1(x))
            return self.head(x)

        @torch.no_grad()
        def activations(self, x):
            """Return the three cached representations for input one-hot x."""
            h1 = F.relu(self.layer0(x))
            h2 = F.relu(self.layer1(h1))
            logits = self.head(h2)
            return {"h1": h1, "h2": h2, "logits": logits}

else:
    GrokMLP = None


def generate_data(p=97, train_frac=0.3, seed=42):
    """All p^2 pairs; deterministic shuffle + split (matches cnb recipe)."""
    rng = np.random.RandomState(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    def to_tensors(pairs):
        inputs = torch.zeros(len(pairs), 2 * p)
        labels = torch.zeros(len(pairs), dtype=torch.long)
        for i, (a, b) in enumerate(pairs):
            inputs[i, a] = 1.0
            inputs[i, p + b] = 1.0
            labels[i] = (a + b) % p
        return inputs, labels

    tr_x, tr_y = to_tensors(train_pairs)
    te_x, te_y = to_tensors(test_pairs)
    return tr_x, tr_y, te_x, te_y

# ============================ end of verbatim fork ============================


def split_pairs(p=P, train_frac=TRAIN_FRAC, seed=SEED):
    """The (a, b) integer arrays of `generate_data`'s split, same shuffle. Returns
    (a_tr, b_tr, a_te, b_te) as int64 numpy arrays."""
    rng = np.random.RandomState(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    rng.shuffle(all_pairs)
    n_train = int(len(all_pairs) * train_frac)
    arr = np.asarray(all_pairs, dtype=np.int64)
    return arr[:n_train, 0], arr[:n_train, 1], arr[n_train:, 0], arr[n_train:, 1]


def onehot(a, b, p=P):
    """One-hot (a, b) inputs, 2p dims, float32 torch tensor (same layout as generate_data)."""
    x = torch.zeros(len(a), 2 * p)
    idx = torch.arange(len(a))
    x[idx, torch.as_tensor(a)] = 1.0
    x[idx, p + torch.as_tensor(b)] = 1.0
    return x


def char_features(a, b, freqs, p=P):
    """Character alphabet: [cos(2 pi w a/p), sin(2 pi w a/p)]_w  ++  same for b.
    4 * len(freqs) dims, float32 torch tensor. Ordering: for a, all cos then all sin; then b."""
    f = np.asarray(sorted(freqs), dtype=np.float64)
    ang_a = 2 * np.pi * np.outer(np.asarray(a, np.float64), f) / p
    ang_b = 2 * np.pi * np.outer(np.asarray(b, np.float64), f) / p
    feats = np.concatenate([np.cos(ang_a), np.sin(ang_a), np.cos(ang_b), np.sin(ang_b)], axis=1)
    return torch.tensor(feats, dtype=torch.float32)


def n_params(model):
    return int(sum(q.numel() for q in model.parameters()))


def state_to_numpy(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def train_net(model, tr_x, tr_y, te_x, te_y, n_epochs, lr=LR, weight_decay=WEIGHT_DECAY,
              eval_every=50, snap_epochs=(), stop_at_test=None, log_every=5000, tag="",
              optimizer="adamw", momentum=0.0, probe=None, probe_every=None):
    """Full-batch AdamW training, the donor's `train_grokking` loop (grokking_fwd_vs_inv.py
    lines 106-160) with three changes: the model and tensors are passed in (so rung.py can swap
    the input alphabet), a (epoch, train_acc, test_acc, train_loss) curve is recorded every
    `eval_every` epochs, and snapshots are numpy state dicts at `snap_epochs` (plus 'init',
    taken BEFORE the first step). The seed is the caller's: construct the model right after
    torch.manual_seed(seed), as the donor does.

    `stop_at_test`: if set, stop once test acc has been >= this value at an eval (rung.py's
    epochs-to-solve); training otherwise runs the full `n_epochs`.

    Added for rung_controls.py (defaults reproduce every earlier run exactly):
    `optimizer` 'adamw' (the recipe) or 'sgd' (torch SGD, L2 weight decay, optional `momentum`);
    `probe(model) -> dict`, called at every eval epoch divisible by `probe_every`, stored in `probes`.

    Returns dict(curve, snapshots, first_hit{thr: epoch}, final_state, seconds, epochs_run, probes).
    """
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(optimizer)
    probes = []
    snap_epochs = set(int(e) for e in snap_epochs)
    snapshots = {"init": {"epoch": -1, "state": state_to_numpy(model)}}
    curve = []
    thresholds = (0.5, 0.9, 0.95, 0.99, 1.0)
    first_hit = {str(t): None for t in thresholds}
    t0 = time.time()
    epoch = -1
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(tr_x), tr_y)
        loss.backward()
        opt.step()

        is_eval = (epoch % eval_every == 0) or (epoch == n_epochs - 1) or (epoch in snap_epochs)
        if is_eval:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(tr_x).argmax(-1) == tr_y).float().mean().item()
                te_acc = (model(te_x).argmax(-1) == te_y).float().mean().item()
            curve.append((epoch, tr_acc, te_acc, float(loss.item())))
            for t in thresholds:
                if first_hit[str(t)] is None and te_acc >= t - 1e-12:
                    first_hit[str(t)] = epoch
            if epoch in snap_epochs:
                snapshots[f"e{epoch}"] = {"epoch": epoch, "state": state_to_numpy(model),
                                          "train_acc": tr_acc, "test_acc": te_acc}
            if probe is not None and probe_every and (epoch % probe_every == 0 or epoch == n_epochs - 1):
                probes.append(dict(probe(model), epoch=epoch, train_acc=tr_acc, test_acc=te_acc))
            if log_every and (epoch % log_every == 0 or epoch == n_epochs - 1):
                eps = (epoch + 1) / (time.time() - t0 + 1e-9)
                print(f"  {tag} epoch {epoch:6d} | train_acc={tr_acc:.4f} test_acc={te_acc:.4f} "
                      f"loss={loss.item():.4f} | {eps:.0f} ep/s", flush=True)
            if stop_at_test is not None and te_acc >= stop_at_test - 1e-12:
                if probe is not None and probe_every and not (probes and probes[-1]["epoch"] == epoch):
                    probes.append(dict(probe(model), epoch=epoch, train_acc=tr_acc, test_acc=te_acc))
                break
    return {"curve": curve, "snapshots": snapshots, "first_hit": first_hit,
            "final_state": state_to_numpy(model), "seconds": time.time() - t0,
            "epochs_run": epoch + 1, "probes": probes}


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
