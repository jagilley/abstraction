"""RHM waist test: does a capacity waist over a span implement chunking?

Pure test of the "self-pooling through a capacity waist" idea from the
motor-chunking conversation (conversations/Claude-Motor abstraction learning in
dance and cerebellar dynamics.md). NO forward model — this isolates the single
mechanism we converged on:

    A capacity waist over a span of consecutive activations should preferentially
    preserve the chunk's ABSTRACT identity (its RHM parent) and discard the
    WITHIN-CHUNK detail (which synonym / which leaves), because in the RHM the
    parent is a sufficient statistic for the span's role and the within-span rule
    choices carry no residual decision. If so, a generic capacity squeeze on M's
    own learned representation = chunking, with no bespoke segmentation oracle
    (the "uniform speedup, emergent boundaries" picture).

Setup: train a small causal GPT on RHM NTP (distinct-rule v=8, s=2, L=6, m=2 —
occupancy 0.25, every level recoverable so any lost level is the waist's doing,
not an information ceiling). Take its activations over aligned height-2 spans
(K = s^2 = 4 leaves). Compress the span jointly with a PCA waist of width w (the
optimal linear capacity bottleneck; w=1 = the whole 4-token span as one number).
From the top-w projection, linear-probe the ground-truth hierarchy at three
levels — P2 (height-2 parent, the chunk), P1 (height-1 parents), leaves (detail).

Prediction (chunking): recovery ordering P2 >= P1 >= leaves is PRESERVED as w
shrinks; leaves collapse first while P2 survives the narrowest waist.
Null (no chunking): all levels erode together, or detail is preferentially kept.

Run:
    cd experiments
    modal run --detach -m rhm.rhm_waist::waist
"""

import json
import numpy as np

from .shared import app, volume, DATA_DIR, NumpyEncoder


# ----------------------------------------------------------------------------
# RHM data generation WITH the full latent tree (self-contained; does not touch
# rhm_data.py so all prior experiments stay reproducible).
# ----------------------------------------------------------------------------
def generate_rules_distinct(v, s, L, m, seed=0):
    """m DISTINCT s-tuples per feature (canonical RHM). Same as rhm_data."""
    rng = np.random.default_rng(seed)
    rules = []
    for _ in range(L):
        layer = np.empty((v, m, s), dtype=np.int64)
        for f in range(v):
            codes = rng.choice(v ** s, size=m, replace=False)
            for i in range(s):
                layer[f, :, i] = (codes // (v ** i)) % v
        rules.append(layer)
    return rules


def generate_with_tree(rules, n, seed=0):
    """Generate n sequences AND record every latent node + rule choice.

    Returns:
      leaves:  (n, s^L) int  — the token sequences (height-0 nodes)
      levels:  dict h -> (n, s^(L-h)) int, node values at tree-height h
               (h=0 leaves, h=L root)
      choices: dict h -> (n, s^(L-h)) int, rule index chosen when expanding a
               height-h node to its s children (h=1..L)
    """
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    current = rng.integers(0, v, size=(n, 1))
    levels = {L: current.copy()}
    choices = {}
    for ell in range(L):                       # rules[ell] expands height (L-ell) -> (L-ell-1)
        h = L - ell
        nf = current.shape[1]
        rc = rng.integers(0, m, size=(n, nf))
        choices[h] = rc
        nxt = np.empty((n, nf * s), dtype=np.int64)
        for j in range(nf):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = nxt
        levels[h - 1] = current.copy()
    return current, levels, choices


# ----------------------------------------------------------------------------
# PCA waist + linear probe helpers
# ----------------------------------------------------------------------------
def pca_fit(X):
    """Return (mean, components VT sorted by variance desc, explained-var ratio)."""
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD; components are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    return mu, Vt, evr


def pca_project(X, mu, Vt, w):
    """Top-w PCA scores of X (the width-w waist code)."""
    return (X - mu) @ Vt[:w].T


def probe_acc(Ztr, ytr, Zte, yte, n_classes, device, steps=300, lr=0.05):
    """Multinomial logistic regression accuracy (standardized codes)."""
    import torch
    mu = Ztr.mean(0, keepdims=True)
    sd = Ztr.std(0, keepdims=True) + 1e-6
    Ztr = (Ztr - mu) / sd
    Zte = (Zte - mu) / sd
    Xtr = torch.tensor(Ztr, dtype=torch.float32, device=device)
    Xte = torch.tensor(Zte, dtype=torch.float32, device=device)
    Ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    W = torch.zeros(Xtr.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, Ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (Xte @ W + b).argmax(1).cpu().numpy()
    return float((pred == yte).mean())


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
@app.function(gpu="L4", volumes={DATA_DIR: volume}, timeout=3600)
def waist(v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
          n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
          n_train_seq: int = 40000, n_steps: int = 8000, batch_size: int = 128,
          lr: float = 3e-4, span_height: int = 2,
          n_eval_seq: int = 3000, rule_seed: int = 0):
    import torch
    import torch.nn.functional as F
    from .model import GPT

    device = "cuda"
    torch.manual_seed(0)
    L = depth
    seq_len = s ** L
    K = s ** span_height                       # span length in leaves (height-2 -> 4)
    print(f"DGP: v{v} s{s} L{L} m{m} (distinct), seq_len={seq_len}, "
          f"span_height={span_height} -> K={K} leaves/span")

    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    # ---- train M on NTP -----------------------------------------------------
    train_leaves, _, _ = generate_with_tree(rules, n_train_seq, seed=1)
    data = torch.tensor(train_leaves, dtype=torch.long, device=device)  # (N, seq_len)
    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
        x = data[idx]
        _, loss = model(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == n_steps - 1:
            print(f"  step {step:5d}: ntp_loss={float(loss):.4f}")

    # ---- cache activations over aligned height-2 spans + ground truth --------
    model.eval()
    ev_leaves, ev_levels, ev_choices = generate_with_tree(rules, n_eval_seq, seed=2)
    ev = torch.tensor(ev_leaves, dtype=torch.long, device=device)

    layers = ["post_block1", f"post_block{n_layer - 1}"]
    span_starts = list(range(0, seq_len, K))            # aligned height-2 blocks
    acts = {ly: [] for ly in layers}
    gt = {"P2": [], "P1a": [], "P1b": [], "leaf0": [], "rc2": []}
    with torch.no_grad():
        for b0 in range(0, ev.shape[0], 512):
            xb = ev[b0:b0 + 512]
            _, _, inter = model(xb, return_intermediates=True)
            for a in span_starts:
                blk = a // K                            # height-2 node index
                for ly in layers:
                    acts[ly].append(inter[ly][:, a:a + K, :].reshape(xb.shape[0], -1).cpu().numpy())
                gt["P2"].append(ev_levels[span_height][b0:b0 + 512, blk])
                gt["P1a"].append(ev_levels[span_height - 1][b0:b0 + 512, a // s])
                gt["P1b"].append(ev_levels[span_height - 1][b0:b0 + 512, a // s + 1])
                gt["leaf0"].append(ev_leaves[b0:b0 + 512, a])
                gt["rc2"].append(ev_choices[span_height][b0:b0 + 512, blk])
    acts = {ly: np.concatenate(acts[ly], 0) for ly in layers}   # (n_spans, K*d)
    gt = {k: np.concatenate(vv, 0) for k, vv in gt.items()}
    n = acts[layers[0]].shape[0]
    print(f"span examples: {n}  (act dim = {acts[layers[0]].shape[1]})")

    # train/test split for PCA-fit + probe
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ntr = int(0.75 * n)
    tr, te = perm[:ntr], perm[ntr:]

    widths = [w for w in [1, 2, 4, 8, 16, 32, 64, 128, 256, K * n_embd] if w <= K * n_embd]
    widths = sorted(set(widths))
    # probe targets: (name, gt-array, n_classes, tag)  tag: abstract levels vs detail
    targets = [
        ("P2",    gt["P2"],    v, "abstract (chunk id, height 2)"),
        ("P1",    None,        v, "abstract (height 1, avg of a,b)"),
        ("leaf",  gt["leaf0"], v, "detail (a leaf token)"),
        ("rule2", gt["rc2"],   m, "detail (synonym choice at chunk top)"),
    ]
    chance = {"P2": 1 / v, "P1": 1 / v, "leaf": 1 / v, "rule2": 1 / m}

    results = {}
    for ly in layers:
        X = acts[ly]
        mu, Vt, evr = pca_fit(X[tr])
        print(f"\n=== layer {ly} ===  (chance P2/P1/leaf={1/v:.3f}, rule2={1/m:.3f})")
        header = "  w".ljust(7) + "evr%".rjust(7) + "".join(nm.rjust(9) for nm, *_ in targets)
        print(header)
        results[ly] = {"evr": {}, "acc": {}}
        for w in widths:
            Ztr = pca_project(X[tr], mu, Vt, w)
            Zte = pca_project(X[te], mu, Vt, w)
            cum_evr = float(evr[:w].sum())
            row = {}
            for nm, y, nc, _ in targets:
                if nm == "P1":
                    a1 = probe_acc(Ztr, gt["P1a"][tr], Zte, gt["P1a"][te], nc, device)
                    a2 = probe_acc(Ztr, gt["P1b"][tr], Zte, gt["P1b"][te], nc, device)
                    acc = 0.5 * (a1 + a2)
                else:
                    acc = probe_acc(Ztr, y[tr], Zte, y[te], nc, device)
                row[nm] = acc
            results[ly]["evr"][w] = cum_evr
            results[ly]["acc"][w] = row
            print(f"  {w:<5d}{100*cum_evr:6.1f}%" +
                  "".join(f"{row[nm]:9.3f}" for nm, *_ in targets))

    # ---- summary: the chunking signature ------------------------------------
    print("\n" + "=" * 72)
    print("CHUNKING SIGNATURE: does the narrow waist keep the chunk (P2) and")
    print("drop the detail (leaf)?  Report accuracy-above-chance retained at each w,")
    print("as a fraction of the full-width (w=all) value.")
    for ly in layers:
        full = results[ly]["acc"][widths[-1]]
        print(f"\n  layer {ly}:  (retained fraction of above-chance recovery)")
        print("  w".ljust(7) + "P2".rjust(9) + "P1".rjust(9) + "leaf".rjust(9) + "rule2".rjust(9))
        for w in widths:
            row = results[ly]["acc"][w]
            frac = {}
            for nm in ["P2", "P1", "leaf", "rule2"]:
                denom = full[nm] - chance[nm]
                frac[nm] = (row[nm] - chance[nm]) / denom if denom > 1e-6 else float("nan")
            print(f"  {w:<5d}" + "".join(f"{frac[nm]:9.2f}" for nm in ["P2", "P1", "leaf", "rule2"]))

    out = {
        "config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_head=n_head,
                       n_embd=n_embd, span_height=span_height, K=K,
                       n_steps=n_steps, rule_seed=rule_seed),
        "widths": widths, "chance": chance, "results": results,
    }
    save_dir = f"{DATA_DIR}/waist"
    import os
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/waist_v{v}_s{s}_L{L}_m{m}_h{span_height}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved {fname}")
    return out


class _Bottleneck:
    """Factory for a small enc->w->dec bottleneck that predicts next-span tokens."""
    @staticmethod
    def build(nn, din, w, hidden, K, v):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, w))
                self.dec = nn.Sequential(nn.Linear(w, hidden), nn.GELU(), nn.Linear(hidden, K * v))
                self.K, self.v = K, v

            def code(self, x):
                return self.enc(x)

            def forward(self, x):
                return self.dec(self.enc(x)).view(-1, self.K, self.v)
        return Net()


def _train_bottleneck(net, Xtr, Ytr, v, device, steps=2000, lr=1e-3, bs=1024):
    import torch
    import torch.nn.functional as F
    o = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(steps):
        bi = torch.randint(0, Xtr.shape[0], (bs,), device=device)
        logits = net(Xtr[bi])
        loss = F.cross_entropy(logits.reshape(-1, v), Ytr[bi].reshape(-1))
        o.zero_grad(); loss.backward(); o.step()
    net.eval()
    return net


# ----------------------------------------------------------------------------
# Pass 2: RECURSIVE chunking (the abstraction-ratchet discriminator).
#
# Pass 1 chunks height-2 spans of M's activations -> a 4x-shorter sequence of
# codes (one per chunk), each carrying its chunk's abstract identity P2. Pass 2
# applies the SAME external-role waist to ADJACENT PAIRS of pass-1 codes (a
# height-3 span) and asks whether it recovers the next level up, P3 (d=3).
#
# The question (per RHM_DEEP_COMPOSITION): does recursive chunking on SOFT codes
# track the BP ceiling (joint inference) or cascade toward the greedy floor
# (hard cluster-and-lift)?  Reference lines for v8 s2 L6 m2 distinct (computed
# locally via rhm_local_signal.run_bp_ceiling / run_compounding):
#     d3 (height-3 parent):  BP ceiling = 0.989   greedy floor = 0.856
# So pass-2 P3 in [0.856, 0.989] discriminates cascade (~greedy) vs climb (~BP).
# ----------------------------------------------------------------------------
BP_D3, GREEDY_D3 = 0.989, 0.856          # v8 s2 L6 m2 distinct, rule_seed=0


@app.function(gpu="L4", volumes={DATA_DIR: volume}, timeout=3600)
def waist_recursive(v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
                    n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
                    n_train_seq: int = 40000, n_steps: int = 8000, batch_size: int = 128,
                    lr: float = 3e-4, n_eval_seq: int = 6000, rule_seed: int = 0,
                    w1: int = 64, ae_hidden: int = 256, ae_steps: int = 2500, ae_lr: float = 1e-3):
    import os
    import torch
    import torch.nn as nn
    from .model import GPT

    device = "cuda"
    torch.manual_seed(0)
    L = depth
    seq_len = s ** L
    K = s ** 2                                   # height-2 chunk = 4 leaves
    n_blocks = seq_len // K                       # 16 height-2 blocks
    layer = f"post_block{n_layer - 1}"            # read M's late layer
    print(f"DGP v{v} s{s} L{L} m{m} distinct, seq_len={seq_len}, "
          f"{n_blocks} height-2 blocks/seq, pass-1 width w1={w1}")
    print(f"Reference (d3): BP ceiling={BP_D3}, greedy floor={GREEDY_D3}")
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    # ---- train M on NTP -----------------------------------------------------
    train_leaves, _, _ = generate_with_tree(rules, n_train_seq, seed=1)
    data = torch.tensor(train_leaves, dtype=torch.long, device=device)
    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
        x = data[idx]
        _, loss = model(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0 or step == n_steps - 1:
            print(f"  step {step:5d}: ntp_loss={float(loss):.4f}")

    # ---- cache per-block span activations + ground-truth tree ----------------
    model.eval()
    ev_leaves, ev_levels, ev_choices = generate_with_tree(rules, n_eval_seq, seed=2)
    ev = torch.tensor(ev_leaves, dtype=torch.long, device=device)
    d = n_embd
    block_acts = np.empty((n_eval_seq, n_blocks, K * d), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, n_eval_seq, 512):
            xb = ev[b0:b0 + 512]
            _, _, inter = model(xb, return_intermediates=True)
            H = inter[layer]                                   # (bs, seq_len, d)
            for blk in range(n_blocks):
                a = blk * K
                block_acts[b0:b0 + xb.shape[0], blk] = \
                    H[:, a:a + K, :].reshape(xb.shape[0], -1).cpu().numpy()
    # sequence-level train/test split (no leakage across passes)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_eval_seq)
    ntr = int(0.75 * n_eval_seq)
    S_tr, S_te = perm[:ntr], perm[ntr:]

    # ---- PASS 1: train one external-role encoder at width w1 ----------------
    # examples: block b (b < n_blocks-1) -> predict next block's K tokens
    def block_dataset(seqs):
        X, Y = [], []
        for blk in range(n_blocks - 1):
            X.append(block_acts[seqs, blk])
            Y.append(ev_leaves[seqs, (blk + 1) * K:(blk + 2) * K])
        return np.concatenate(X, 0), np.concatenate(Y, 0)
    Xb_tr, Yb_tr = block_dataset(S_tr)
    mu = Xb_tr.mean(0, keepdims=True); sd = Xb_tr.std(0, keepdims=True) + 1e-6
    p1 = _Bottleneck.build(nn, K * d, w1, ae_hidden, K, v).to(device)
    _train_bottleneck(p1, torch.tensor((Xb_tr - mu) / sd, dtype=torch.float32, device=device),
                      torch.tensor(Yb_tr, dtype=torch.long, device=device), v, device,
                      steps=ae_steps, lr=ae_lr)
    # produce pass-1 codes for ALL blocks of ALL sequences
    codes = np.empty((n_eval_seq, n_blocks, w1), dtype=np.float32)
    with torch.no_grad():
        for blk in range(n_blocks):
            Xn = (block_acts[:, blk] - mu) / sd
            codes[:, blk] = p1.code(torch.tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy()
    print(f"pass-1 codes: {codes.shape}")

    # ---- PASS 2 dataset: height-3 span g = blocks (2g, 2g+1) ----------------
    n_h3 = n_blocks // 2                                       # 8 height-3 spans
    # inputs & targets & ground truth, indexed by (seq, g) with g < n_h3-1
    def h3_arrays(seqs):
        code_pair, raw_pair, nxt, P3, P2a, P2b, rc3 = [], [], [], [], [], [], []
        for g in range(n_h3 - 1):
            code_pair.append(np.concatenate([codes[seqs, 2 * g], codes[seqs, 2 * g + 1]], -1))
            raw_pair.append(np.concatenate([block_acts[seqs, 2 * g], block_acts[seqs, 2 * g + 1]], -1))
            nxt.append(ev_leaves[seqs, (g + 1) * 8:(g + 2) * 8])          # next height-3 span (8 tok)
            P3.append(ev_levels[3][seqs, g])
            P2a.append(ev_levels[2][seqs, 2 * g]); P2b.append(ev_levels[2][seqs, 2 * g + 1])
            rc3.append(ev_choices[3][seqs, g])
        cat = lambda L: np.concatenate(L, 0)
        return (cat(code_pair), cat(raw_pair), cat(nxt),
                cat(P3), cat(P2a), cat(P2b), cat(rc3))
    cp_tr, rp_tr, y_tr, P3_tr, P2a_tr, P2b_tr, rc3_tr = h3_arrays(S_tr)
    cp_te, rp_te, y_te, P3_te, P2a_te, P2b_te, rc3_te = h3_arrays(S_te)

    def probe(Ztr, ytr, Zte, yte, nc):
        return probe_acc(Ztr, ytr, Zte, yte, nc, device)

    print("\n--- baselines: how legible is P3 (d3) BEFORE recursive chunking? ---")
    # (a) P3 from M's raw height-3 span activations (full, 2*K*d)
    b_raw = probe(rp_tr, P3_tr, rp_te, P3_te, v)
    # (b) P3 from the uncompressed pass-1 code pair (2*w1)
    b_code = probe(cp_tr, P3_tr, cp_te, P3_te, v)
    print(f"  P3 from M raw h3-span acts (2*{K*d}d):   {b_raw:.3f}")
    print(f"  P3 from pass-1 code pair (2*{w1}d):      {b_code:.3f}")
    print(f"  [reference d3]  BP={BP_D3}  greedy={GREEDY_D3}  chance={1/v:.3f}")

    # ---- PASS 2 sweep: recursive external-role waist on the code pair --------
    cp_mu = cp_tr.mean(0, keepdims=True); cp_sd = cp_tr.std(0, keepdims=True) + 1e-6
    CPtr = torch.tensor((cp_tr - cp_mu) / cp_sd, dtype=torch.float32, device=device)
    CPte = torch.tensor((cp_te - cp_mu) / cp_sd, dtype=torch.float32, device=device)
    Ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    widths2 = [w for w in [1, 2, 4, 8, 16, 32, 64, 128] if w <= 2 * w1]
    print("\n--- pass-2 recursive chunk: recover P3 (d3) from compressed code ---")
    print("  w2".ljust(7) + "extAcc".rjust(8) + "P3".rjust(8) + "P2".rjust(8) +
          "rule3".rjust(8) + "   vs-ref")
    res = {}
    for w2 in widths2:
        torch.manual_seed(0)
        net = _Bottleneck.build(nn, 2 * w1, w2, ae_hidden, 8, v).to(device)
        _train_bottleneck(net, CPtr, Ytr, v, device, steps=ae_steps, lr=ae_lr)
        with torch.no_grad():
            ext = float((net(CPte).argmax(-1).cpu().numpy() == y_te).mean())
            Ztr = net.code(CPtr).cpu().numpy(); Zte = net.code(CPte).cpu().numpy()
        aP3 = probe(Ztr, P3_tr, Zte, P3_te, v)
        aP2 = 0.5 * (probe(Ztr, P2a_tr, Zte, P2a_te, v) + probe(Ztr, P2b_tr, Zte, P2b_te, v))
        arc3 = probe(Ztr, rc3_tr, Zte, rc3_te, m)
        tag = ("~BP" if aP3 >= BP_D3 - 0.02 else
               "~greedy" if abs(aP3 - GREEDY_D3) < 0.03 else
               "<greedy" if aP3 < GREEDY_D3 else "mid")
        res[w2] = dict(ext=ext, P3=aP3, P2=aP2, rule3=arc3)
        print(f"  {w2:<5d}{ext:8.3f}{aP3:8.3f}{aP2:8.3f}{arc3:8.3f}   {tag}")

    print("\n" + "=" * 72)
    print("VERDICT: does pass-2 P3 track BP (climb) or greedy (cascade)?")
    best = max(res.values(), key=lambda r: r["P3"])["P3"]
    print(f"  best pass-2 P3 = {best:.3f}   BP={BP_D3}  greedy={GREEDY_D3}  "
          f"M-raw={b_raw:.3f}  chance={1/v:.3f}")

    out = {"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_embd=n_embd,
                          w1=w1, rule_seed=rule_seed),
           "reference": {"BP_d3": BP_D3, "greedy_d3": GREEDY_D3},
           "baselines": {"P3_from_M_raw": b_raw, "P3_from_code_pair": b_code},
           "pass2": res}
    save_dir = f"{DATA_DIR}/waist"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/waist_recursive_v{v}_s{s}_L{L}_m{m}_w1{w1}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved {fname}")
    return out


# ----------------------------------------------------------------------------
# Pass 2 in a FRONTIER-GAP regime (v16 m4): the unconfounded climb test.
#
# v8/m2 was too easy — M already reached d3, so pass-2 only CARRIED structure.
# v16/m4 (occupancy 0.25) has real headroom: a trained GPT reaches ~d3 (0.88) but
# COLLAPSES at d4 (~0.43), while d4 is still near-perfectly recoverable in
# principle. So we chunk height-3 spans in pass-1 (M has d3), then lift PAIRS of
# those codes toward d4 in pass-2 — a level M did NOT compute. If pass-2 d4 beats
# M's own d4 legibility, the ratchet MANUFACTURED structure NTP couldn't reach.
#
# Reference lines (v16 s2 L6 m4 distinct, RHM_DEEP_COMPOSITION Thread B; chance
# 0.0625):  d3 BP=0.995 greedy=0.751 | d4 BP=0.989 greedy=0.708 (M~0.43 @20k).
# ----------------------------------------------------------------------------
REF_V16M4 = {"d3": {"BP": 0.995, "greedy": 0.751}, "d4": {"BP": 0.989, "greedy": 0.708}}

# Full per-level reference (v16 s2 L6 m4 distinct, RHM_DEEP_COMPOSITION Thread B;
# chance 0.0625). BP = optimal tree inference; greedy = hard cluster-and-lift
# cascade; M = a token-NTP GPT's best linear probe (its frontier ~ d3).
REF_V16M4_FULL = {
    1: {"BP": 0.998, "greedy": 0.887, "M": 0.98},
    2: {"BP": 0.998, "greedy": 0.845, "M": 0.97},
    3: {"BP": 0.995, "greedy": 0.751, "M": 0.88},
    4: {"BP": 0.989, "greedy": 0.708, "M": 0.50},
    5: {"BP": 0.955, "greedy": 0.476, "M": 0.17},
}


# ----------------------------------------------------------------------------
# Exp 6 — "learn from your own latents" (Korchinski-Favero-Wyart, 2605.27734).
#
# Diagnosis from Exp 4: our recursion stalled because the waist predicted the
# next span's TOKENS. Token-level objectives cost vm^(l+2) (exponential in depth
# -> the frontier). Predicting your own LIFTED LATENT of a COUSIN (the tuple
# sharing the l+2 grandparent) instead costs vm^3, constant in depth.
#
# We build the recursion from one-hot tokens (NO M -- M is the token-NTP baseline
# that stalls) and run two controlled conditions, identical except the target:
#   * latent: predict the cousin's own level-l representation R_l (frozen lifted
#             latent) -- the escape.
#   * token:  predict a fixed surface leaf inside the cousin's subtree -- the
#             diluted token-level control (gets deeper/harder every level).
# At level l the encoder clusters level-l tuples by their cousin context; its
# code becomes R_{l+1}. We probe each level's code for the true parent and
# compare to BP / greedy / M-frontier. Prediction: latent tracks BP at every
# depth; token dilutes like M.
# ----------------------------------------------------------------------------
@app.function(gpu="L4", volumes={DATA_DIR: volume}, timeout=3600)
def waist_latent(v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
                 n_seq: int = 40000, w: int = 64, hidden: int = 256,
                 enc_steps: int = 3000, enc_lr: float = 1e-3, rule_seed: int = 0):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda"
    torch.manual_seed(0)
    L = depth
    seq_len = s ** L
    assert s == 2, "this minimal recursion assumes branching factor s=2"
    print(f"DGP v{v} s{s} L{L} m{m} distinct; latent-prediction recursion from tokens.")
    print(f"chance={1/v:.4f}; reference per level: BP / greedy / M(token-NTP frontier)")
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    leaves, levels, choices = generate_with_tree(rules, n_seq, seed=2)
    n = n_seq
    rng = np.random.default_rng(0)
    perm = rng.permutation(n); ntr = int(0.75 * n)
    tr, te = perm[:ntr], perm[ntr:]

    def probe(Ztr, ytr, Zte, yte):
        return probe_acc(Ztr, ytr, Zte, yte, v, device)

    class EncPred(nn.Module):
        def __init__(self, din, w, hidden, dout):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, w))
            self.pred = nn.Sequential(nn.Linear(w, hidden), nn.GELU(), nn.Linear(hidden, dout))

        def code(self, x):
            return self.enc(x)

        def forward(self, x):
            return self.pred(self.enc(x))

    results = {}
    for cond in ["latent", "token"]:
        print(f"\n{'='*72}\nCONDITION: {cond}-target\n{'='*72}")
        print("  level".ljust(8) + "probe".rjust(8) + "BP".rjust(8) + "greedy".rjust(8) +
              "M".rjust(8) + "   verdict")
        R = np.eye(v, dtype=np.float32)[leaves]          # (n, seq_len, v) one-hot leaves
        d = v
        results[cond] = {}
        for ell in range(0, L - 1):                      # build R_{ell+1}, probe d(ell+1)
            cur_len = R.shape[1]
            n_tup = cur_len // 2
            tup = R.reshape(n, n_tup, 2 * d)             # adjacent-pair tuples
            cix = np.arange(n_tup) ^ 1                    # cousin = tuple j^1 (shares l+2 grandparent)
            if cond == "latent":
                target = tup[:, cix, :]                   # cousin's frozen level-l rep (2d)
                dout = 2 * d
            else:                                         # token: a fixed leaf in the cousin subtree
                leaf_idx = (cix) * (2 ** (ell + 1))       # first leaf of cousin span
                target = leaves[:, leaf_idx]              # (n, n_tup) int token
                dout = v
            # tensors (train split)
            Xtr = torch.tensor(tup[tr].reshape(-1, 2 * d), dtype=torch.float32, device=device)
            if cond == "latent":
                Ttr = torch.tensor(target[tr].reshape(-1, dout), dtype=torch.float32, device=device)
            else:
                Ttr = torch.tensor(target[tr].reshape(-1), dtype=torch.long, device=device)
            net = EncPred(2 * d, w, hidden, dout).to(device)
            o = torch.optim.Adam(net.parameters(), lr=enc_lr)
            net.train()
            for st in range(enc_steps):
                bi = torch.randint(0, Xtr.shape[0], (4096,), device=device)
                out = net(Xtr[bi])
                loss = (F.mse_loss(out, Ttr[bi]) if cond == "latent"
                        else F.cross_entropy(out, Ttr[bi]))
                o.zero_grad(); loss.backward(); o.step()
            net.eval()
            # codes for all sequences -> R_{ell+1}
            with torch.no_grad():
                flat = torch.tensor(tup.reshape(-1, 2 * d), dtype=torch.float32, device=device)
                codes = np.empty((n * n_tup, w), dtype=np.float32)
                for b0 in range(0, flat.shape[0], 200000):
                    codes[b0:b0 + 200000] = net.code(flat[b0:b0 + 200000]).cpu().numpy()
            R_next = codes.reshape(n, n_tup, w)
            # probe R_{ell+1} for the true level-(ell+1) parent
            y = levels[ell + 1]                           # (n, n_tup)
            acc = probe(R_next[tr].reshape(-1, w), y[tr].reshape(-1),
                        R_next[te].reshape(-1, w), y[te].reshape(-1))
            ref = REF_V16M4_FULL.get(ell + 1, {"BP": float("nan"), "greedy": float("nan"), "M": float("nan")})
            verdict = ("~BP" if acc >= ref["BP"] - 0.05 else
                       "beats-M" if acc > ref["M"] + 0.05 else
                       "~M/token-diluted" if abs(acc - ref["M"]) < 0.06 else "mid")
            results[cond][ell + 1] = dict(probe=acc, **ref)
            print(f"  d{ell+1:<6d}{acc:8.3f}{ref['BP']:8.3f}{ref['greedy']:8.3f}"
                  f"{ref['M']:8.3f}   {verdict}")
            R = R_next; d = w

    print("\n" + "=" * 72)
    print("VERDICT: does the latent target climb where the token target (and M) stall?")
    for ell in range(1, L):
        lat = results["latent"][ell]["probe"]; tok = results["token"][ell]["probe"]
        mm = REF_V16M4_FULL[ell]["M"]
        print(f"  d{ell}:  latent={lat:.3f}   token={tok:.3f}   M(NTP)={mm:.3f}   "
              f"BP={REF_V16M4_FULL[ell]['BP']:.3f}")

    out = {"config": dict(v=v, s=s, L=L, m=m, w=w, n_seq=n_seq, rule_seed=rule_seed),
           "reference": REF_V16M4_FULL, "results": results}
    save_dir = f"{DATA_DIR}/waist"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/waist_latent_v{v}_s{s}_L{L}_m{m}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved {fname}")
    return out


@app.function(gpu="L4", volumes={DATA_DIR: volume}, timeout=5400)
def waist_gap(v: int = 16, s: int = 2, depth: int = 6, m: int = 4,
              n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
              n_train_seq: int = 60000, n_steps: int = 20000, batch_size: int = 128,
              lr: float = 3e-4, n_eval_seq: int = 8000, rule_seed: int = 0,
              pass1_height: int = 3, w1: int = 128,
              ae_hidden: int = 512, ae_steps: int = 3000, ae_lr: float = 1e-3):
    import os
    import torch
    import torch.nn as nn
    from .model import GPT

    device = "cuda"
    torch.manual_seed(0)
    L = depth
    seq_len = s ** L
    h1 = pass1_height                             # pass-1 chunk height (3 -> d3)
    h2 = h1 + 1                                    # pass-2 target height (4 -> d4)
    K1, K2 = s ** h1, s ** h2                      # 8, 16 leaves
    nb1, nb2 = seq_len // K1, seq_len // K2        # 8 height-3 blocks, 4 height-4 blocks
    layer = f"post_block{n_layer - 1}"
    ref3, ref4 = REF_V16M4["d3"], REF_V16M4["d4"]
    print(f"DGP v{v} s{s} L{L} m{m} distinct, seq_len={seq_len}; pass-1 chunks "
          f"height-{h1} ({K1} leaves, {nb1} blocks) -> d{h1}; pass-2 -> d{h2}")
    print(f"Reference d{h2}: BP={ref4['BP']} greedy={ref4['greedy']} chance={1/v:.4f}")
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    # ---- train M on NTP -----------------------------------------------------
    train_leaves, _, _ = generate_with_tree(rules, n_train_seq, seed=1)
    data = torch.tensor(train_leaves, dtype=torch.long, device=device)
    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
        x = data[idx]
        _, loss = model(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 4000 == 0 or step == n_steps - 1:
            print(f"  step {step:5d}: ntp_loss={float(loss):.4f}")

    # ---- cache per-(height-3)-block span activations + tree ------------------
    model.eval()
    ev_leaves, ev_levels, ev_choices = generate_with_tree(rules, n_eval_seq, seed=2)
    ev = torch.tensor(ev_leaves, dtype=torch.long, device=device)
    d = n_embd
    block_acts = np.empty((n_eval_seq, nb1, K1 * d), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, n_eval_seq, 256):
            xb = ev[b0:b0 + 256]
            _, _, inter = model(xb, return_intermediates=True)
            H = inter[layer]
            for blk in range(nb1):
                a = blk * K1
                block_acts[b0:b0 + xb.shape[0], blk] = \
                    H[:, a:a + K1, :].reshape(xb.shape[0], -1).cpu().numpy()
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_eval_seq)
    ntr = int(0.75 * n_eval_seq)
    S_tr, S_te = perm[:ntr], perm[ntr:]

    # ---- PASS 1: external-role encoder at width w1 (height-3 -> next height-3)
    def block_dataset(seqs):
        X, Y = [], []
        for blk in range(nb1 - 1):
            X.append(block_acts[seqs, blk])
            Y.append(ev_leaves[seqs, (blk + 1) * K1:(blk + 2) * K1])
        return np.concatenate(X, 0), np.concatenate(Y, 0)
    Xb_tr, Yb_tr = block_dataset(S_tr)
    mu = Xb_tr.mean(0, keepdims=True); sd = Xb_tr.std(0, keepdims=True) + 1e-6
    p1 = _Bottleneck.build(nn, K1 * d, w1, ae_hidden, K1, v).to(device)
    _train_bottleneck(p1, torch.tensor((Xb_tr - mu) / sd, dtype=torch.float32, device=device),
                      torch.tensor(Yb_tr, dtype=torch.long, device=device), v, device,
                      steps=ae_steps, lr=ae_lr)
    codes = np.empty((n_eval_seq, nb1, w1), dtype=np.float32)
    with torch.no_grad():
        for blk in range(nb1):
            Xn = (block_acts[:, blk] - mu) / sd
            codes[:, blk] = p1.code(torch.tensor(Xn, dtype=torch.float32, device=device)).cpu().numpy()
    print(f"pass-1 codes: {codes.shape}")

    # ---- PASS 2 dataset: height-4 span g = height-3 blocks (2g, 2g+1) -------
    def h2_arrays(seqs):
        cp, rp, nxt, P4, P3a, P3b, rc4 = [], [], [], [], [], [], []
        for g in range(nb2 - 1):
            cp.append(np.concatenate([codes[seqs, 2 * g], codes[seqs, 2 * g + 1]], -1))
            rp.append(np.concatenate([block_acts[seqs, 2 * g], block_acts[seqs, 2 * g + 1]], -1))
            nxt.append(ev_leaves[seqs, (g + 1) * K2:(g + 2) * K2])           # next height-4 span
            P4.append(ev_levels[h2][seqs, g])
            P3a.append(ev_levels[h1][seqs, 2 * g]); P3b.append(ev_levels[h1][seqs, 2 * g + 1])
            rc4.append(ev_choices[h2][seqs, g])
        cat = np.concatenate
        return (cat(cp, 0), cat(rp, 0), cat(nxt, 0),
                cat(P4, 0), cat(P3a, 0), cat(P3b, 0), cat(rc4, 0))
    cp_tr, rp_tr, y_tr, P4_tr, P3a_tr, P3b_tr, rc4_tr = h2_arrays(S_tr)
    cp_te, rp_te, y_te, P4_te, P3a_te, P3b_te, rc4_te = h2_arrays(S_te)

    def probe(Ztr, ytr, Zte, yte, nc):
        return probe_acc(Ztr, ytr, Zte, yte, nc, device)

    print(f"\n--- baselines: how legible is P4 (d{h2}) BEFORE the lift? ---")
    b_raw = probe(rp_tr, P4_tr, rp_te, P4_te, v)          # M's own d4 frontier (this M)
    b_code = probe(cp_tr, P4_tr, cp_te, P4_te, v)         # uncompressed code pair
    b_code_d3 = probe(cp_tr, P3a_tr, cp_te, P3a_te, v)    # sanity: codes carry d3
    print(f"  P4 from M raw h{h2}-span acts (2*{K1*d}d):  {b_raw:.3f}   <- M's d{h2} frontier")
    print(f"  P4 from pass-1 code pair (2*{w1}d):         {b_code:.3f}")
    print(f"  (sanity) P3 from code pair:                 {b_code_d3:.3f}")
    print(f"  [ref d{h2}] BP={ref4['BP']} greedy={ref4['greedy']} chance={1/v:.4f}")

    # ---- PASS 2 sweep: recursive external-role waist toward d4 ---------------
    cp_mu = cp_tr.mean(0, keepdims=True); cp_sd = cp_tr.std(0, keepdims=True) + 1e-6
    CPtr = torch.tensor((cp_tr - cp_mu) / cp_sd, dtype=torch.float32, device=device)
    CPte = torch.tensor((cp_te - cp_mu) / cp_sd, dtype=torch.float32, device=device)
    Ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    widths2 = [w for w in [1, 2, 4, 8, 16, 32, 64, 128, 256] if w <= 2 * w1]
    print(f"\n--- pass-2 lift: recover P4 (d{h2}) from compressed code ---")
    print("  w2".ljust(7) + "extAcc".rjust(8) + "P4".rjust(8) + "P3".rjust(8) +
          "rule4".rjust(8) + "   vs-ref")
    res = {}
    for w2 in widths2:
        torch.manual_seed(0)
        net = _Bottleneck.build(nn, 2 * w1, w2, ae_hidden, K2, v).to(device)
        _train_bottleneck(net, CPtr, Ytr, v, device, steps=ae_steps, lr=ae_lr)
        with torch.no_grad():
            ext = float((net(CPte).argmax(-1).cpu().numpy() == y_te).mean())
            Ztr = net.code(CPtr).cpu().numpy(); Zte = net.code(CPte).cpu().numpy()
        aP4 = probe(Ztr, P4_tr, Zte, P4_te, v)
        aP3 = 0.5 * (probe(Ztr, P3a_tr, Zte, P3a_te, v) + probe(Ztr, P3b_tr, Zte, P3b_te, v))
        arc4 = probe(Ztr, rc4_tr, Zte, rc4_te, m)
        tag = ("~BP" if aP4 >= ref4["BP"] - 0.03 else
               "beats-M" if aP4 > b_raw + 0.03 else
               "~greedy" if abs(aP4 - ref4["greedy"]) < 0.03 else
               "~M-frontier" if abs(aP4 - b_raw) < 0.03 else "mid")
        res[w2] = dict(ext=ext, P4=aP4, P3=aP3, rule4=arc4)
        print(f"  {w2:<5d}{ext:8.3f}{aP4:8.3f}{aP3:8.3f}{arc4:8.3f}   {tag}")

    best = max(res.values(), key=lambda r: r["P4"])["P4"]
    print("\n" + "=" * 72)
    print(f"VERDICT: pass-2 best d{h2} = {best:.3f}   vs  M-frontier={b_raw:.3f}  "
          f"greedy={ref4['greedy']}  BP={ref4['BP']}  chance={1/v:.4f}")
    if best > b_raw + 0.05:
        print("  -> CLIMB: recursion recovered d4 structure M's NTP did not.")
    else:
        print("  -> NO CLIMB: recursion bounded by M's frontier (need raw-data re-engagement / B).")

    out = {"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_embd=n_embd,
                          pass1_height=h1, w1=w1, n_steps=n_steps, rule_seed=rule_seed),
           "reference": {"d3": ref3, "d4": ref4},
           "baselines": {"P4_from_M_raw": b_raw, "P4_from_code_pair": b_code,
                         "P3_from_code_pair": b_code_d3},
           "pass2": res}
    save_dir = f"{DATA_DIR}/waist"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/waist_gap_v{v}_s{s}_L{L}_m{m}_h{h1}_w1{w1}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved {fname}")
    return out


# ----------------------------------------------------------------------------
# External-role waist: same everything, but the waist is a LEARNED bottleneck
# trained to predict the NEXT span's tokens (the span's external role), not to
# reconstruct the span's own content. In RHM, given the parent P2 the within-
# span rule choices are conditionally independent of everything outside the span
# -> an external-prediction bottleneck should KEEP the abstract parent and DROP
# the within-chunk detail, flipping the PCA (reconstruction) ordering.
# ----------------------------------------------------------------------------
@app.function(gpu="L4", volumes={DATA_DIR: volume}, timeout=5400)
def waist_external(v: int = 8, s: int = 2, depth: int = 6, m: int = 2,
                   n_layer: int = 4, n_head: int = 4, n_embd: int = 128,
                   n_train_seq: int = 40000, n_steps: int = 8000, batch_size: int = 128,
                   lr: float = 3e-4, span_height: int = 2,
                   n_eval_seq: int = 3000, rule_seed: int = 0,
                   ae_hidden: int = 256, ae_steps: int = 2000, ae_lr: float = 1e-3):
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .model import GPT

    device = "cuda"
    torch.manual_seed(0)
    L = depth
    seq_len = s ** L
    K = s ** span_height
    print(f"DGP: v{v} s{s} L{L} m{m} (distinct), seq_len={seq_len}, K={K} leaves/span")
    print("WAIST = learned bottleneck trained to predict the NEXT span's tokens.")
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    # ---- train M on NTP (identical to waist()) ------------------------------
    train_leaves, _, _ = generate_with_tree(rules, n_train_seq, seed=1)
    data = torch.tensor(train_leaves, dtype=torch.long, device=device)
    model = GPT(v, seq_len, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
        x = data[idx]
        _, loss = model(x[:, :-1].contiguous(), targets=x[:, 1:].contiguous())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == n_steps - 1:
            print(f"  step {step:5d}: ntp_loss={float(loss):.4f}")

    # ---- cache span activations + ground truth + NEXT-span token targets -----
    model.eval()
    ev_leaves, ev_levels, ev_choices = generate_with_tree(rules, n_eval_seq, seed=2)
    ev = torch.tensor(ev_leaves, dtype=torch.long, device=device)
    layers = ["post_block1", f"post_block{n_layer - 1}"]
    # only spans that HAVE a next span (external target defined)
    span_starts = [a for a in range(0, seq_len, K) if a + 2 * K <= seq_len]
    acts = {ly: [] for ly in layers}
    gt = {"P2": [], "P1a": [], "P1b": [], "leaf0": [], "rc2": []}
    nxt = []                                            # next-span tokens (target)
    with torch.no_grad():
        for b0 in range(0, ev.shape[0], 512):
            xb = ev[b0:b0 + 512]
            _, _, inter = model(xb, return_intermediates=True)
            for a in span_starts:
                blk = a // K
                for ly in layers:
                    acts[ly].append(inter[ly][:, a:a + K, :].reshape(xb.shape[0], -1).cpu().numpy())
                gt["P2"].append(ev_levels[span_height][b0:b0 + 512, blk])
                gt["P1a"].append(ev_levels[span_height - 1][b0:b0 + 512, a // s])
                gt["P1b"].append(ev_levels[span_height - 1][b0:b0 + 512, a // s + 1])
                gt["leaf0"].append(ev_leaves[b0:b0 + 512, a])
                gt["rc2"].append(ev_choices[span_height][b0:b0 + 512, blk])
                nxt.append(ev_leaves[b0:b0 + 512, a + K:a + 2 * K])   # (bs, K)
    acts = {ly: np.concatenate(acts[ly], 0) for ly in layers}
    gt = {k: np.concatenate(vv, 0) for k, vv in gt.items()}
    nxt = np.concatenate(nxt, 0)                        # (n_spans, K)
    n = acts[layers[0]].shape[0]
    print(f"span examples: {n}  (act dim = {acts[layers[0]].shape[1]}, target = next {K} tokens)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ntr = int(0.75 * n)
    tr, te = perm[:ntr], perm[ntr:]

    widths = sorted(set(w for w in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if w <= K * n_embd))
    targets = [("P2", gt["P2"], v), ("P1", None, v), ("leaf", gt["leaf0"], v), ("rule2", gt["rc2"], m)]
    chance = {"P2": 1 / v, "P1": 1 / v, "leaf": 1 / v, "rule2": 1 / m}

    class Bottleneck(nn.Module):
        def __init__(self, din, w, hidden, K, v):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, w))
            self.dec = nn.Sequential(nn.Linear(w, hidden), nn.GELU(), nn.Linear(hidden, K * v))
            self.K, self.v = K, v

        def code(self, x):
            return self.enc(x)

        def forward(self, x):
            return self.dec(self.enc(x)).view(-1, self.K, self.v)

    results = {}
    for ly in layers:
        X = acts[ly]
        # standardize encoder inputs (per-dim, train stats)
        mu = X[tr].mean(0, keepdims=True); sd = X[tr].std(0, keepdims=True) + 1e-6
        Xn = (X - mu) / sd
        Xtr = torch.tensor(Xn[tr], dtype=torch.float32, device=device)
        Xte = torch.tensor(Xn[te], dtype=torch.float32, device=device)
        Ytr = torch.tensor(nxt[tr], dtype=torch.long, device=device)   # (ntr, K)
        din = X.shape[1]
        print(f"\n=== layer {ly} ===  (chance P2/P1/leaf={1/v:.3f}, rule2={1/m:.3f}; "
              f"next-token chance={1/v:.3f})")
        print("  w".ljust(7) + "extAcc".rjust(8) + "".join(nm.rjust(9) for nm, *_ in targets))
        results[ly] = {"ext_acc": {}, "acc": {}}
        for w in widths:
            torch.manual_seed(0)
            net = Bottleneck(din, w, ae_hidden, K, v).to(device)
            o = torch.optim.Adam(net.parameters(), lr=ae_lr)
            net.train()
            for st in range(ae_steps):
                bi = torch.randint(0, Xtr.shape[0], (1024,), device=device)
                logits = net(Xtr[bi])                      # (1024,K,v)
                loss = F.cross_entropy(logits.reshape(-1, v), Ytr[bi].reshape(-1))
                o.zero_grad(); loss.backward(); o.step()
            net.eval()
            with torch.no_grad():
                ext_pred = net(Xte).argmax(-1).cpu().numpy()          # (nte,K)
                ext_acc = float((ext_pred == nxt[te]).mean())
                Ztr = net.code(Xtr).cpu().numpy()
                Zte = net.code(Xte).cpu().numpy()
            row = {}
            for nm, y, nc in targets:
                if nm == "P1":
                    a1 = probe_acc(Ztr, gt["P1a"][tr], Zte, gt["P1a"][te], nc, device)
                    a2 = probe_acc(Ztr, gt["P1b"][tr], Zte, gt["P1b"][te], nc, device)
                    acc = 0.5 * (a1 + a2)
                else:
                    acc = probe_acc(Ztr, y[tr], Zte, y[te], nc, device)
                row[nm] = acc
            results[ly]["ext_acc"][w] = ext_acc
            results[ly]["acc"][w] = row
            print(f"  {w:<5d}{ext_acc:8.3f}" + "".join(f"{row[nm]:9.3f}" for nm, *_ in targets))

    print("\n" + "=" * 72)
    print("CHUNKING SIGNATURE (external-role waist): retained fraction of")
    print("above-chance recovery vs the widest waist. Prediction: P2 kept, leaf/rule2 dropped.")
    for ly in layers:
        full = results[ly]["acc"][widths[-1]]
        print(f"\n  layer {ly}:")
        print("  w".ljust(7) + "P2".rjust(9) + "P1".rjust(9) + "leaf".rjust(9) + "rule2".rjust(9))
        for w in widths:
            row = results[ly]["acc"][w]
            frac = {}
            for nm in ["P2", "P1", "leaf", "rule2"]:
                denom = full[nm] - chance[nm]
                frac[nm] = (row[nm] - chance[nm]) / denom if denom > 1e-6 else float("nan")
            print(f"  {w:<5d}" + "".join(f"{frac[nm]:9.2f}" for nm in ["P2", "P1", "leaf", "rule2"]))

    out = {"config": dict(v=v, s=s, L=L, m=m, n_layer=n_layer, n_head=n_head,
                          n_embd=n_embd, span_height=span_height, K=K, n_steps=n_steps,
                          rule_seed=rule_seed, waist="external_next_span"),
           "widths": widths, "chance": chance, "results": results}
    save_dir = f"{DATA_DIR}/waist"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{save_dir}/waist_external_v{v}_s{s}_L{L}_m{m}_h{span_height}.json"
    with open(fname, "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved {fname}")
    return out
