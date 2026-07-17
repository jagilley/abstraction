"""RHM subtree specialization: does naive LATERAL specialization buy DEPTH?

A "scaling-era" control. The Random Hierarchy Model generates length-s^L sequences
from a fixed hierarchy of composition rules; because we know the DGP, every sequence
has ground-truth ancestor features at every level, so we can linearly probe how deep
a trained model's representation reaches ("depth recovery"). At high synonymity m the
next-token gradient at depth is diluted, so an NTP model learns only ~1-2 shallow
composition levels and STALLS mid-tree.

Claim under test: naive lateral specialization -- training on only a SUBSECTION
(subtree) of the RHM tree -- does NOT buy more representational depth than training on
the full tree, even measured on that subsection's OWN domain. Concentrating breadth
does not convert into depth. Prediction: FULL >= SUBTREE (SUBTREE no deeper) at the
deep levels (d4, d5), evaluated on S itself. Sanity check that must hold before a
companion experiment (a direct deep target) is claimed to break the stall.

SINGLE CONTROLLED VARIABLE = the BREADTH of the training distribution. Two conditions,
IDENTICAL architecture, IDENTICAL initial weights, IDENTICAL n_steps/batch/optimizer,
IDENTICAL flat phase-diverse NTP objective (so total TOKEN budget is matched -- the
tightest test):
  FULL    : roots uniform over all v.
  SUBTREE : roots restricted to a fixed subset S (default |S|=4 of 16, i.e. 25%).

Headline readout: per-level ancestor recovery (d1..d6, best-over-blocks, linear AND
MLP probes) on a held-out eval set drawn from S, for BOTH models. Secondary: probe
both on the COMPLEMENT (roots not in S; expect SUBTREE forgot it) and on the full tree,
and report per-model val loss on both corpora -- documenting "SUBTREE traded coverage
for nothing".

Reference baseline (full-tree NTP, this regime, full-tree eval): d1~0.98, d3~0.88,
d4~0.51, d6(root)~0.08. FULL on full-eval should land near this.

Run (from experiments/; the rhm-scaling-data volume + all comparison results live on
the jagilley workspace, so use that profile to co-locate — chromatic works too for a
fresh run since this generates its own corpus):
  MODAL_PROFILE=jagilley modal run rhm/specialization/rhm_subtree_specialization.py::subtree_specialization --quick   # smoke
  MODAL_PROFILE=jagilley modal run --detach rhm/specialization/rhm_subtree_specialization.py::subtree_specialization  # full
"""

import json
import os

import modal

from rhm.rhm_data import generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-subtree-specialization", image=image)


# ======================================================================
# Data + probe helpers (copied from rhm_latent_loop to stay self-contained)
# ======================================================================

def _generate_with_traces(rules, n_sequences, seed):
    """Aligned leaf sequences + true latent feature and rule-choice at every node.
    level_features[ell]: (n_seq, s^ell) parent features; [0]=roots, [L]=leaves."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    level_features, level_rules = [], []
    current = rng.integers(0, v, size=(n_sequences, 1))
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        level_rules.append(rc.copy())
        nxt = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = nxt
    level_features.append(current.copy())  # leaves
    return current, level_features, level_rules


def _generate_with_traces_roots(rules, n_sequences, seed, roots):
    """_generate_with_traces but ROOT-RESTRICTED: the level-0 (root) feature of each
    sequence is drawn uniformly from `roots` (the subtree S) instead of all v. Every
    other level of the DGP is unchanged, so a root-restricted corpus is a genuine
    subsection of the same fixed hierarchy."""
    import numpy as np
    rng = np.random.default_rng(seed)
    L = len(rules)
    v, m, s = rules[0].shape
    roots_arr = np.asarray(roots, dtype=np.int64)
    level_features, level_rules = [], []
    current = rng.choice(roots_arr, size=(n_sequences, 1))   # <-- only change vs above
    for ell in range(L):
        level_features.append(current.copy())
        n_nodes = current.shape[1]
        rc = rng.integers(0, m, size=(n_sequences, n_nodes))
        level_rules.append(rc.copy())
        nxt = np.empty((n_sequences, n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][current[:, j], rc[:, j]]
        current = nxt
    level_features.append(current.copy())  # leaves
    return current, level_features, level_rules


def _probe_acc(X, y, v, device, steps, lr, hidden=0, wd=1e-4, train_frac=0.7):
    """Decodability of label y from X. hidden=0 -> linear; hidden>0 -> 1-layer MLP."""
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
    return te


def _probe_model(model, eval_x, y_level, block_names, L, v, device,
                 probe_steps, probe_lr, mlp_hidden, mlp_steps, label):
    """Per-level ancestor recovery (linear + MLP, best-over-blocks) of an in-memory
    model on a fixed eval set. y_level: {ell -> (N,) label at last position on device}.
    Reported as d{L-ell} (ell=0 -> d6 root; ell=L-1 -> d1)."""
    import torch
    model.eval()
    acc = {b: [] for b in block_names}
    with torch.no_grad():
        for i in range(0, len(eval_x), 256):
            _, _, inter = model(eval_x[i:i + 256], return_intermediates=True)
            for b in block_names:
                acc[b].append(inter[b][:, -1, :].float())
    acts = {b: torch.cat(acc[b], 0) for b in block_names}
    lin_best, mlp_best = {}, {}
    for ell in range(L):
        lin_best[ell] = max(
            _probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr)
            for b in block_names)
        if mlp_hidden:
            mlp_best[ell] = max(
                _probe_acc(acts[b], y_level[ell], v, device, mlp_steps, 1e-3,
                           hidden=mlp_hidden, wd=1e-3) for b in block_names)
    del acts
    torch.cuda.empty_cache()
    print(f"  [{label}] LIN best:  " +
          "  ".join(f"d{L-ell}:{lin_best[ell]:.3f}" for ell in range(L)))
    if mlp_hidden:
        print(f"  [{label}] MLP best:  " +
              "  ".join(f"d{L-ell}:{mlp_best[ell]:.3f}" for ell in range(L)))
    return {"linear_best": lin_best, "mlp_best": mlp_best}


# ======================================================================
# Modal GPU function
# ======================================================================

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def subtree_specialization(
    # DGP (canonical frontier regime)
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    # Model (~6.3M params)
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # The single controlled variable: SUBTREE root subset S
    subtree_roots: str = "0,1,2,3",
    # Training (matches the latent_loop NTP baseline so numbers are comparable)
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, pool_size: int = 200000, data_seed: int = 7,
    eval_interval: int = 1000,
    # Measurement
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    seed: int = 42, tag: str = "", quick: bool = False,
):
    import numpy as np
    import torch

    from rhm.model import GPT       # imported inside the fn (torch is remote-only)

    if quick:  # smoke mode: ~2 min end-to-end
        n_steps = 800
        pool_size = 6000
        n_eval_sequences = 800
        probe_steps = 150
        mlp_steps = 150
        eval_interval = 200

    device = "cuda"
    L = depth
    T = s ** L
    chance = 1.0 / v
    key = f"v{v}_s{s}_L{L}_m{m}"
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    S = sorted({int(x) for x in subtree_roots.split(",") if x.strip() != ""})
    assert all(0 <= r < v for r in S), f"subtree roots must be in [0,{v}): {S}"
    complement = [r for r in range(v) if r not in S]

    print(f"{'='*74}\nRHM SUBTREE SPECIALIZATION  {key}  {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  occupancy m/v^(s-1)={m/v**(s-1):.3f}  chance=1/v={chance:.4f}  T={T}")
    print(f"  S (subtree roots) = {S}   |S|={len(S)}/{v}   complement |{len(complement)}|")
    print(f"  n_steps={n_steps}  batch={batch_size}  lr={lr}  pool={pool_size:,}  "
          f"n_eval={n_eval_sequences}{'   [QUICK]' if quick else ''}")
    print(f"{'='*74}")

    # --- training corpora: leaves only, flattened phase-diverse (all positions
    #     supervised via random windows over the concatenated corpus) ---
    print("Generating FULL training pool...")
    full_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    print("Generating SUBTREE training pool (root-restricted to S)...")
    sub_seqs, _, _ = _generate_with_traces_roots(rules, pool_size, data_seed + 100, S)
    full_corpus = torch.from_numpy(full_seqs.astype(np.int64)).reshape(-1)   # (P*T,) cpu
    sub_corpus = torch.from_numpy(sub_seqs.astype(np.int64)).reshape(-1)
    del full_seqs, sub_seqs

    arangeT = torch.arange(T)

    def make_get_batch(corpus):
        n_corpus = corpus.shape[0]

        def get_batch(gen):
            ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
            idx = ix[:, None] + arangeT[None, :]                 # (B,T) windows
            return corpus[idx].to(device), corpus[idx + 1].to(device)
        return get_batch

    full_batch = make_get_batch(full_corpus)
    sub_batch = make_get_batch(sub_corpus)

    def eval_ntp(model, get_batch, n=10):
        model.eval()
        gen = torch.Generator().manual_seed(eval_seed + 5)
        tot = 0.0
        with torch.no_grad():
            for _ in range(n):
                x, y = get_batch(gen)
                _, loss = model(x, y)
                tot += loss.item()
        return tot / n

    # --- eval sets (held out): S, complement, full ---
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}

    def build_eval(seqs, lf):
        ex = torch.from_numpy(seqs.astype(np.int64)).to(device)
        yl = {ell: torch.from_numpy(lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
              for ell in range(L)}
        return ex, yl

    print("Generating eval sets (S / complement / full)...")
    S_seqs, S_lf, _ = _generate_with_traces_roots(rules, n_eval_sequences, eval_seed, S)
    cp_seqs, cp_lf, _ = _generate_with_traces_roots(
        rules, n_eval_sequences, eval_seed + 1, complement)
    fl_seqs, fl_lf, _ = _generate_with_traces(rules, n_eval_sequences, eval_seed + 2)
    evals = {
        "S": build_eval(S_seqs, S_lf),
        "complement": build_eval(cp_seqs, cp_lf),
        "full": build_eval(fl_seqs, fl_lf),
    }

    # --- shared init (identical starting weights for BOTH conditions) ---
    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: val.cpu().clone() for k, val in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    def dkeys(d):
        return {f"d{L-ell}": d[ell] for ell in range(L)}

    def train_and_probe(cond, get_batch):
        print(f"\n{'='*60}\n  CONDITION: {cond}\n{'='*60}")
        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(init_state)          # identical init across conditions
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_gen = torch.Generator().manual_seed(seed + 1)
        for step in range(n_steps):
            model.train()
            x, y = get_batch(train_gen)             # phase-diverse NTP, all positions
            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % eval_interval == 0 or step == n_steps - 1:
                vl = eval_ntp(model, get_batch)
                print(f"    [{cond}] step {step:6d}: ntp={loss.item():.4f} val={vl:.4f}")

        # val loss on BOTH corpora (coverage story)
        val = {"full": eval_ntp(model, full_batch), "S": eval_ntp(model, sub_batch)}
        print(f"  [{cond}] val loss:  full={val['full']:.4f}  S={val['S']:.4f}")

        # per-level recovery on all three eval sets
        recovery = {}
        for ename, (ex, yl) in evals.items():
            r = _probe_model(model, ex, yl, block_names, L, v, device,
                             probe_steps, probe_lr, mlp_hidden, mlp_steps,
                             f"{cond}|{ename}")
            recovery[ename] = {"linear_best": dkeys(r["linear_best"]),
                               "mlp_best": dkeys(r["mlp_best"])}
        del model
        torch.cuda.empty_cache()
        return {"val": val, "recovery": recovery}

    out = {
        "full": train_and_probe("FULL", full_batch),
        "subtree": train_and_probe("SUBTREE", sub_batch),
    }

    # --- headline side-by-side table: FULL vs SUBTREE on S ---
    dlevels = [f"d{L-ell}" for ell in range(L)]                 # d6..d1 (root->leaf)
    fS = out["full"]["recovery"]["S"]
    sS = out["subtree"]["recovery"]["S"]
    print(f"\n{'='*74}")
    print(f"DEPTH RECOVERY ON S (subtree eval)   S={S}  chance(root)=1/{len(S)}={1/len(S):.3f}")
    print(f"  prediction: SUBTREE NOT deeper than FULL, esp. at deep levels (d4,d5)")
    print(f"{'-'*74}")
    print(f"  {'level':<6} {'FULL_lin':>9} {'SUB_lin':>9} {'Δlin':>8}   "
          f"{'FULL_mlp':>9} {'SUB_mlp':>9} {'Δmlp':>8}")
    for d in dlevels:
        fl, sl = fS["linear_best"][d], sS["linear_best"][d]
        fm_, sm = fS["mlp_best"][d], sS["mlp_best"][d]
        print(f"  {d:<6} {fl:>9.3f} {sl:>9.3f} {sl-fl:>+8.3f}   "
              f"{fm_:>9.3f} {sm:>9.3f} {sm-fm_:>+8.3f}")
    print(f"{'-'*74}")
    print("  secondary (MLP best, d1..d6):")
    for cond_name, rec in (("FULL", out["full"]), ("SUBTREE", out["subtree"])):
        for ename in ("S", "complement", "full"):
            mlp = rec["recovery"][ename]["mlp_best"]
            row = "  ".join(f"{d}:{mlp[d]:.3f}" for d in reversed(dlevels))  # d1..d6
            print(f"    {cond_name:<7} on {ename:<10} {row}")
    print(f"  val loss:  FULL(full={out['full']['val']['full']:.3f} S={out['full']['val']['S']:.3f})"
          f"  SUBTREE(full={out['subtree']['val']['full']:.3f} S={out['subtree']['val']['S']:.3f})")
    print(f"{'='*74}")

    # --- save ---
    config = dict(v=v, s=s, depth=L, m=m, rule_seed=rule_seed, n_layer=n_layer,
                  n_head=n_head, n_embd=n_embd, subtree_roots=S, complement=complement,
                  n_steps=n_steps, batch_size=batch_size, lr=lr,
                  weight_decay=weight_decay, pool_size=pool_size, data_seed=data_seed,
                  n_eval_sequences=n_eval_sequences, eval_seed=eval_seed,
                  probe_steps=probe_steps, mlp_hidden=mlp_hidden, mlp_steps=mlp_steps,
                  seed=seed, chance=chance, quick=quick)
    results = {"config": config, "conditions": out}

    stag = "-".join(str(r) for r in S)
    descriptive_tag = (f"{key}_distinct_S{len(S)}of{v}_r{stag}_N{n_steps}"
                       f"{('_' + tag) if tag else ''}{'_quick' if quick else ''}")
    out_dir = f"{DATA_DIR}/rhm_subtree_specialization/{descriptive_tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"Saved -> {out_dir}/results.json")
    return results


@app.local_entrypoint()
def main(subtree_roots: str = "0,1,2,3", n_steps: int = 20000, quick: bool = False,
         tag: str = ""):
    subtree_specialization.remote(subtree_roots=subtree_roots, n_steps=n_steps,
                                  quick=quick, tag=tag)
