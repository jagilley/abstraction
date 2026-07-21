"""Forward vs. Inverse self-models on a grokked modular-addition MLP.

Motivation
----------
The `a2a_forward` program trains *forward* self-models: small nets that predict a
main model's LATER-layer activations from its EARLIER-layer activations. Predicting
forward is always well-posed, because a layer *is* a function -- one input, one
output.

This script tests the mirror-image object: an *inverse-dynamics* self-model that
predicts EARLIER activations from LATER ones ("what internal state produced this
output?"). The Jacobian-Conjecture falsification (a polynomial map with constant
nonzero Jacobian that is nonetheless globally non-injective -- three inputs collapse
to one output) is the sharpest statement that a locally-fine map can be globally
non-invertible. The prediction: an inverse self-model is ill-posed exactly to the
degree the forward computation destroys information (collapses a preimage fiber).

Modular addition is the cleanest ground-truth analog of the JC collision. A grokked
model computing (a+b) mod p maps all p pairs with a given sum to the SAME output
class. The preimage fiber of output c is exactly {(a,b): a+b == c mod p}, size p.
We know the collision structure in closed form.

We use the non-residual GrokMLP from
`fer/experiments/zipfian_grokking/cnb_self_regulation/modal_app.py` deliberately:
a transformer's residual stream (a_j = a_i + Delta) is engineered to be approximately
invertible, which would *hide* the effect. A plain MLP has no such highway, so the
collapse of the fiber is real.

Design
------
1. Grok a GrokMLP on (a+b) mod p (canonical recipe: full-batch AdamW, wd=1.0).
2. Freeze it; cache activations for ALL p^2 pairs at three depths:
     h1     = relu(layer0(x))   [EARLY  -- pair-specific]
     h2     = relu(layer1(h1))  [MIDDLE]
     logits = head(h2)          [LATE   -- collapses onto the sum c]
3. Model-free fiber diagnostics: for each representation, what fraction of its
   variance is WITHIN-fiber (i.e. survives after conditioning on c)? A representation
   that is a pure function of c has zero within-fiber variance -- fully collapsed.
   The within-fiber variance of the inverse TARGET is the irreducible inverse error:
   no map from a c-collapsed input can recover it.
4. For each (early, late) layer pair, train FORWARD (early->late) and INVERSE
   (late->early) self-models with identical architecture across a capacity sweep.
   Metrics:
     - forward vs inverse test cosine / R^2 (asymmetry)
     - capacity scaling: forward saturates toward ~1, inverse plateaus at a ceiling
     - mode-averaging: does the inverse prediction collapse onto the fiber mean of
       the target? cos(pred, fiber_mean) vs cos(pred, true) vs the trivial
       fiber-mean baseline cos(fiber_mean, true).
     - effective-rank collapse of predictions vs. true.

Everything runs locally (tiny MLP, p^2 examples). No Modal.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Grokking model (verbatim recipe from cnb_self_regulation/modal_app.py)
# =============================================================================


class GrokMLP(nn.Module):
    """2-layer MLP for modular arithmetic grokking. Non-residual by design."""

    def __init__(self, p=97, hidden_dims=(128, 128)):
        super().__init__()
        self.p = p
        self.layer0 = nn.Linear(2 * p, hidden_dims[0])
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


def train_grokking(p, n_epochs, lr, weight_decay, train_frac, seed, device, log_interval,
                   snapshot_epochs=()):
    """Train the grokking MLP. Optionally capture CPU state_dict snapshots at the
    given epochs (each tagged with its train/test accuracy) so we can analyze
    forward/inverse self-models at different points along the grokking curve."""
    tr_x, tr_y, te_x, te_y = generate_data(p, train_frac, seed)
    tr_x, tr_y = tr_x.to(device), tr_y.to(device)
    te_x, te_y = te_x.to(device), te_y.to(device)

    torch.manual_seed(seed)
    model = GrokMLP(p=p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    snapshot_epochs = set(snapshot_epochs)
    snapshots = {}  # label -> {"state": cpu_state_dict, "epoch", "train_acc", "test_acc"}
    best_test_acc, best_state, grok_epoch = 0.0, None, None
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(tr_x), tr_y)
        loss.backward()
        opt.step()

        if epoch % log_interval == 0 or epoch == n_epochs - 1 or epoch in snapshot_epochs:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(tr_x).argmax(-1) == tr_y).float().mean().item()
                te_logits = model(te_x)
                te_acc = (te_logits.argmax(-1) == te_y).float().mean().item()
                te_loss = F.cross_entropy(te_logits, te_y).item()
            if te_acc > best_test_acc:
                best_test_acc = te_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if grok_epoch is None and te_acc > 0.95:
                grok_epoch = epoch
            if epoch in snapshot_epochs:
                snapshots[f"epoch{epoch}"] = {
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch, "train_acc": tr_acc, "test_acc": te_acc,
                }
            if epoch % (log_interval * 20) == 0 or epoch == n_epochs - 1:
                eps = epoch / (time.time() - t0 + 1e-9)
                print(f"  epoch {epoch:6d} | train_acc={tr_acc:.4f} test_acc={te_acc:.4f} "
                      f"test_loss={te_loss:.4f} | {eps:.0f} ep/s")

    # final snapshot = the fully-trained (grokked) model
    snapshots["final"] = {
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "epoch": n_epochs - 1, "train_acc": tr_acc, "test_acc": te_acc,
    }
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [grok] best test_acc={best_test_acc:.4f}, grok_epoch={grok_epoch}, "
          f"time={time.time()-t0:.0f}s")
    return model, best_test_acc, grok_epoch, snapshots


# =============================================================================
# Fiber diagnostics (model-free)
# =============================================================================


def fiber_stats(R, c, p):
    """Within-fiber variance fraction of representation R grouped by sum c.

    Returns the fraction of R's total (centered) variance that survives *within*
    fibers -- i.e. the part conditioning on c does NOT explain. A pure function of
    c has within_frac == 0 (fully collapsed). Also returns the fiber means, and the
    cosine of the fiber-mean to the true vectors (the ceiling for any predictor that
    only sees c).
    """
    R = R.astype(np.float64)
    N, d = R.shape
    global_mean = R.mean(axis=0, keepdims=True)
    total_var = ((R - global_mean) ** 2).sum()

    fiber_mean_of = np.zeros((p, d))
    within = np.zeros_like(R)
    for cc in range(p):
        mask = c == cc
        m = R[mask].mean(axis=0)
        fiber_mean_of[cc] = m
        within[mask] = R[mask] - m
    within_var = (within ** 2).sum()
    within_frac = float(within_var / (total_var + 1e-12))

    # cosine of fiber-mean prediction to true (centered), the c-only ceiling
    fm_pred = fiber_mean_of[c]  # (N, d)
    tc = R - global_mean
    fmc = fm_pred - global_mean
    cos_fm_true = _rowcos(fmc, tc).mean()
    return {
        "within_fiber_frac": within_frac,
        "between_fiber_frac": float(1.0 - within_frac),
        "cos_fibermean_true": float(cos_fm_true),
        "fiber_mean_of": fiber_mean_of,
        "global_mean": global_mean,
    }


def _rowcos(a, b, eps=1e-8):
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return num / den


def effective_rank(X):
    """Entropy effective rank of centered X (N, d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s2 = s ** 2
    pnorm = s2 / (s2.sum() + 1e-30)
    pnorm = pnorm[pnorm > 1e-30]
    return float(np.exp(-(pnorm * np.log(pnorm)).sum()))


# =============================================================================
# Self-model (forward or inverse) + capacity sweep
# =============================================================================


class SelfModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_hidden=1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_self_model(src, tgt, split_idx, hidden, n_hidden, steps, lr, device, seed,
                     c_all, p, batch_size=2048):
    """Train src->tgt map. Standardize both spaces (fit on train). Return metrics.

    src, tgt: (N, d) numpy. split_idx: dict with 'train'/'test' index arrays.
    Cosine/R^2 reported in the standardized target space. Mode-averaging metrics
    compare the prediction to the target's per-fiber mean.
    """
    torch.manual_seed(seed)
    tr, te = split_idx["train"], split_idx["test"]

    # standardize using train stats
    s_mu, s_sd = src[tr].mean(0, keepdims=True), src[tr].std(0, keepdims=True) + 1e-6
    t_mu, t_sd = tgt[tr].mean(0, keepdims=True), tgt[tr].std(0, keepdims=True) + 1e-6
    S = (src - s_mu) / s_sd
    T = (tgt - t_mu) / t_sd

    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    tr_t = torch.tensor(tr, dtype=torch.long, device=device)

    model = SelfModel(S.shape[1], T.shape[1], hidden, n_hidden).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    n_tr = len(tr)
    for step in range(steps):
        model.train()
        if n_tr <= batch_size:
            idx = tr_t
        else:
            idx = tr_t[torch.randint(n_tr, (batch_size,), device=device)]
        pred = model(S_t[idx])
        loss = F.mse_loss(pred, T_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_all = model(S_t).cpu().numpy()  # standardized target space

    def _metrics(idx):
        pr = pred_all[idx]
        tv = T[idx]
        mu = tv.mean(0, keepdims=True)
        ss_tot = ((tv - mu) ** 2).sum()
        # raw cosine (paper-comparable) and R^2 (centered)
        cos = float(_rowcos(pr, tv).mean())
        r2 = float(1.0 - ((pr - tv) ** 2).sum() / (ss_tot + 1e-12))
        # fiber-mean-only predictor: the best any map that sees only c can do.
        # Its R^2 == between-fiber variance fraction of the target; the inverse
        # can't beat it if the input has collapsed onto c. r2 ~ r2_fibermean is
        # the mode-averaging signature (recovers nothing beyond the fiber centroid).
        cc = c_all[idx]
        fmean = np.zeros((p, T.shape[1]))
        for k in range(p):
            m = cc == k
            if m.any():
                fmean[k] = tv[m].mean(0)
        fm_pred = fmean[cc]
        r2_fibermean = float(1.0 - ((fm_pred - tv) ** 2).sum() / (ss_tot + 1e-12))
        # centered cosines (not DC-inflated) for the mode-averaging picture
        prc, tvc, fmc = pr - mu, tv - mu, fm_pred - mu
        cos_pred_fibermean = float(_rowcos(prc, fmc).mean())
        cos_fibermean_true = float(_rowcos(fmc, tvc).mean())
        return {
            "cosine": cos, "r2": r2, "r2_fibermean": r2_fibermean,
            "cos_pred_fibermean": cos_pred_fibermean,
            "cos_fibermean_true": cos_fibermean_true,
            "eff_rank_pred": effective_rank(pr), "eff_rank_true": effective_rank(tv),
        }

    return {
        "n_params": int(n_params),
        "train": _metrics(tr),
        "test": _metrics(te),
    }


# =============================================================================
# Full analysis of one (frozen) model: cache -> fiber diagnostics -> fwd/inv sweep
# =============================================================================


def analyze_model(model, p, device, split_idx, widths, args, x, c_all, verbose=True):
    model.eval()
    acts = model.activations(x)
    reps = {k: v.cpu().numpy() for k, v in acts.items()}
    reps["input"] = x.cpu().numpy()

    # model-free fiber diagnostics
    fiber = {}
    for k in ["input", "h1", "h2", "logits"]:
        st = fiber_stats(reps[k], c_all, p)
        fiber[k] = {kk: vv for kk, vv in st.items() if kk not in ("fiber_mean_of", "global_mean")}
        fiber[k]["eff_rank"] = effective_rank(reps[k])
        if verbose:
            print(f"  fiber {k:7s}: within={st['within_fiber_frac']:.4f} "
                  f"between(c)={st['between_fiber_frac']:.4f} "
                  f"eff_rank={fiber[k]['eff_rank']:.1f}/{reps[k].shape[1]}")

    # forward vs inverse self-models over layer pairs x capacity
    layer_pairs = [("h1", "h2"), ("h1", "logits")]
    sweep = {}
    for early, late in layer_pairs:
        tag = f"{early}->{late}"
        sweep[tag] = {"forward": [], "inverse": []}
        for direction, (srck, tgtk) in [("forward", (early, late)), ("inverse", (late, early))]:
            for w in widths:
                res = train_self_model(
                    reps[srck], reps[tgtk], split_idx, hidden=w, n_hidden=args.n_hidden,
                    steps=args.sm_steps, lr=args.sm_lr, device=device, seed=args.seed + 7,
                    c_all=c_all, p=p)
                res["width"] = w
                sweep[tag][direction].append(res)
                te = res["test"]
                if verbose:
                    print(f"  {tag} {direction:7s} w={w:5d} test cos={te['cosine']:.4f} "
                          f"R2={te['r2']:+.3f} (fiber-mean R2={te['r2_fibermean']:+.3f}) "
                          f"effrank pred/true={te['eff_rank_pred']:.1f}/{te['eff_rank_true']:.1f}")
    return {"fiber_diagnostics": fiber, "sweep": sweep}


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--n-epochs", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sm-steps", type=int, default=4000)
    ap.add_argument("--sm-lr", type=float, default=1e-3)
    ap.add_argument("--sm-split", type=float, default=0.8, help="self-model train frac of all p^2 pairs")
    ap.add_argument("--widths", type=str, default="8,32,128,512,2048")
    ap.add_argument("--n-hidden", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--log-interval", type=int, default=500)
    ap.add_argument("--snapshots", type=str, default="",
                    help="comma-separated epochs to snapshot + analyze along the grokking curve "
                         "(empty = analyze only the final grokked model)")
    args = ap.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    widths = [int(w) for w in args.widths.split(",")]
    snapshot_epochs = [int(e) for e in args.snapshots.split(",") if e.strip()]
    p = args.p

    out_dir = args.out or os.path.join(os.path.dirname(__file__), "results", f"p{p}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Grok (optionally capturing snapshots along the curve) ---
    print(f"\n=== Grokking (a+b) mod {p} ===")
    model, best_acc, grok_epoch, snapshots = train_grokking(
        p, args.n_epochs, args.lr, args.weight_decay, args.train_frac,
        args.seed, device, args.log_interval, snapshot_epochs=snapshot_epochs)
    model.eval()

    # inputs for all p^2 pairs (shared across snapshots)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    x = torch.zeros(len(all_pairs), 2 * p, device=device)
    c_all = np.zeros(len(all_pairs), dtype=np.int64)
    for i, (a, b) in enumerate(all_pairs):
        x[i, a] = 1.0
        x[i, p + b] = 1.0
        c_all[i] = (a + b) % p

    # self-model split: 80/20 over all p^2 pairs (fixed across snapshots)
    rng = np.random.RandomState(args.seed + 1)
    perm = rng.permutation(len(all_pairs))
    n_sm_tr = int(args.sm_split * len(all_pairs))
    split_idx = {"train": perm[:n_sm_tr], "test": perm[n_sm_tr:]}

    def headline(analysis):
        for tag in analysis["sweep"]:
            fwd = analysis["sweep"][tag]["forward"][-1]["test"]
            inv = analysis["sweep"][tag]["inverse"][-1]["test"]
            print(f"    {tag}: forward cos={fwd['cosine']:.4f} R2={fwd['r2']:+.3f}  |  "
                  f"inverse cos={inv['cosine']:.4f} R2={inv['r2']:+.3f} "
                  f"(fiber-mean R2={inv['r2_fibermean']:+.3f}, "
                  f"beyond={inv['r2'] - inv['r2_fibermean']:+.3f})")

    results = {
        "config": vars(args), "device": device,
        "grokking": {"best_test_acc": best_acc, "grok_epoch": grok_epoch, "p": p},
    }

    if snapshot_epochs:
        # --- Trajectory mode: analyze forward/inverse at each grokking snapshot ---
        traj = {}
        # order snapshots by epoch, final last
        ordered = sorted(snapshots.items(), key=lambda kv: kv[1]["epoch"])
        for label, snap in ordered:
            m = GrokMLP(p=p).to(device)
            m.load_state_dict(snap["state"])
            print(f"\n=== Snapshot {label} (epoch {snap['epoch']}, "
                  f"train_acc={snap['train_acc']:.3f}, test_acc={snap['test_acc']:.3f}) ===")
            analysis = analyze_model(m, p, device, split_idx, widths, args, x, c_all)
            analysis["train_acc"] = snap["train_acc"]
            analysis["test_acc"] = snap["test_acc"]
            analysis["epoch"] = snap["epoch"]
            traj[label] = analysis
        results["trajectory"] = traj

        print("\n" + "=" * 70)
        print("TRAJECTORY: inverse ill-posedness vs. grokking (widest self-model)")
        print("=" * 70)
        for label, _ in ordered:
            a = traj[label]
            print(f"  {label} (epoch {a['epoch']}, test_acc={a['test_acc']:.3f}):")
            headline(a)
    else:
        # --- Single-model mode (original behavior): analyze the grokked model ---
        print("\n=== Analyzing grokked model ===")
        analysis = analyze_model(model, p, device, split_idx, widths, args, x, c_all)
        results["fiber_diagnostics"] = analysis["fiber_diagnostics"]
        results["sweep"] = analysis["sweep"]
        print("\n" + "=" * 70)
        print("HEADLINE: forward vs inverse (widest self-model)")
        print("=" * 70)
        headline(analysis)

    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
