"""RHM specialization cut-2a: does CONCENTRATING a direct deep target onto a
subtree A buy SELECTIVE and/or ADVANTAGED depth on A?

Background (the arc). The Random Hierarchy Model generates length-s^L sequences
from a fixed hierarchy of composition rules (each level-ℓ feature has m synonyms).
Because we know the DGP, every sequence has ground-truth ancestor features at every
level, so we linearly (and MLP-) probe how deep a trained model's representation
reaches ("depth recovery", d1=shallow/local ... d6=root). At high synonymity m the
next-token gradient at depth is diluted, so plain NTP STALLS mid-tree (root ~0.08 at
m=4).

Cut-1 (specialization/README.md) established: (a) restricting training to a subtree
does NOT buy depth; (b) reweighting the NTP loss toward deep positions is HARMFUL
(bottom-up: deep composition needs the shallow substrate's gradient); (c) only a
DIRECT deep target -- a per-level ancestor-label CE (the oracle "aux" target from
rhm_latent_loop) -- recruits depth (root 0.08 -> 0.80 at m=4). Reweighting a diluted
TOKEN target cannot supply depth; a direct target can.

CUT-2a QUESTION (this file): does concentrating that direct deep target onto a
subtree A buy SELECTIVE depth (deep on A, shallow off A) and, at high m where even
the uniform direct target is capacity-limited, an ADVANTAGE on A over the uniform
direct target (A-depth > aux_all A-depth)?

SINGLE CONTROLLED VARIABLE = the extra A-concentrated term (weight λ=1.0). EVERY
condition trains flat phase-diverse NTP on ALL roots (the shared low-level substrate
that cut-1 showed A's deep structure needs). Only the added term varies. Subtree
A = roots {0,1,2,3} (|A|=4 of v=16). Four conditions, IDENTICAL model/init/steps/opt:

  condition | loss = flat-NTP-all  +  ...
  ----------|-------------------------------------------------------------------
  ntp       | (nothing)                              -- floor
  aux_all   | λ · oracle deep target (per-level ancestor CE) on ALL-root aligned
  aux_A     | λ · oracle deep target on A-ROOT-only aligned  -- the specialization
  ntp_A     | λ · extra flat NTP over an A-ROOT-only corpus  -- token-reweight ctrl

  aux_A vs ntp_A : direct-target vs more-NTP at fixed A-concentration.
  aux_A vs aux_all: concentration at fixed direct-target.

Readout: per-level ancestor recovery (d1..d6, best-over-blocks, linear AND MLP) on
THREE held-out eval sets for EVERY condition -- A-eval (roots in A), complement-eval
(roots not in A), full-eval. Predictions: ntp shallow everywhere; aux_all deep on
both at m=4, capacity-limited on both at m=8; aux_A deep on A / shallow on B
(SELECTIVE) and, at high m, A-depth > aux_all A-depth (CONCENTRATION ADVANTAGE);
ntp_A ~ ntp floor. Sweep m (default "4,8"): expect only selectivity at m=4 (aux_all
already ~0.80 root) and an advantage at m=8 (capacity-limited).

Per-subtree chance caveat: on A-eval the root chance is 1/|A| (>1/v) and high-level
features are constrained, so deep-level ABSOLUTES on A are inflated -- but ALL
conditions are probed on the SAME A-eval, so the cross-condition comparison is
unaffected.

Run (from experiments/; chromatic is the default credit-bearing workspace, so no
MODAL_PROFILE prefix is needed — cut-2a results live there):
  modal run rhm/specialization/rhm_spec_direct_target.py::spec_direct_target --quick   # smoke
  modal run --detach rhm/specialization/rhm_spec_direct_target.py::spec_direct_target --m-values "4,8"  # full
"""

import json
import os

import modal

from rhm.rhm_data import generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-spec-direct-target", image=image)


# ======================================================================
# Data + probe helpers (copied from rhm_subtree_specialization /
# rhm_latent_loop to stay self-contained)
# ======================================================================

def _generate_with_traces(rules, n_sequences, seed):
    """Aligned leaf sequences + true latent feature and rule-choice at every node.
    level_features[ell]: (n_seq, s^ell) parent features; [0]=roots, [L]=leaves.
    The root of sequence i is level_features[0][i,0]."""
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
    """_generate_with_traces but ROOT-RESTRICTED: each sequence's level-0 (root)
    feature is drawn uniformly from `roots` (the subtree) instead of all v. Every
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

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=86400, memory=32768)
def spec_direct_target(
    # DGP (canonical frontier regime); m is swept via m_values below.
    v: int = 16, s: int = 2, depth: int = 6, rule_seed: int = 0,
    m_values: str = "4,8",
    # Model (~6.3M params)
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # The subtree A (the direct target is concentrated here in aux_A / ntp_A)
    subtree_roots: str = "0,1,2,3",
    # Conditions (order fixed; subsettable for smoke)
    conditions: str = "ntp,aux_all,aux_A,ntp_A",
    lam: float = 1.0,                      # weight on the extra A-concentrated term
    # Training (matches the latent_loop / subtree baseline so numbers are comparable)
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
    import torch.nn as nn
    import torch.nn.functional as F

    from rhm.model import GPT       # imported inside the fn (torch is remote-only)

    if quick:  # smoke mode: ~few min end-to-end, single m
        n_steps = 600
        pool_size = 6000
        n_eval_sequences = 800
        probe_steps = 150
        mlp_steps = 150
        eval_interval = 150

    device = "cuda"
    L = depth
    T = s ** L
    chance = 1.0 / v
    ms = [int(x) for x in m_values.split(",") if x.strip() != ""]
    if quick:
        ms = ms[:1]
    cond_list = [c.strip() for c in conditions.split(",") if c.strip() != ""]
    valid = {"ntp", "aux_all", "aux_A", "ntp_A"}
    assert set(cond_list) <= valid, f"unknown conditions: {set(cond_list) - valid}"

    A = sorted({int(x) for x in subtree_roots.split(",") if x.strip() != ""})
    assert all(0 <= r < v for r in A), f"subtree roots must be in [0,{v}): {A}"
    complement = [r for r in range(v) if r not in A]

    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    sup_blocks = [f"post_block{i}" for i in range(n_layer)]     # aux-supervised blocks
    arangeT = torch.arange(T)
    anc_idx = [torch.arange(T) // (s ** (L - ell)) for ell in range(L)]   # pos->node
    spanend = [torch.tensor([p for p in range(T) if (p + 1) % (s ** (L - ell)) == 0])
               for ell in range(L)]                                      # completed pos
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}

    print(f"{'='*78}\nRHM SPEC DIRECT-TARGET (cut-2a)  v{v}_s{s}_L{L}  "
          f"{n_layer}L/{n_head}H/{n_embd}D")
    print(f"  A (subtree roots) = {A}  |A|={len(A)}/{v}   complement |{len(complement)}|")
    print(f"  conditions = {cond_list}   λ={lam}   m sweep = {ms}")
    print(f"  n_steps={n_steps}  batch={batch_size}  lr={lr}  pool={pool_size:,}  "
          f"n_eval={n_eval_sequences}  chance=1/v={chance:.4f}  T={T}"
          f"{'   [QUICK]' if quick else ''}")
    print(f"{'='*78}")

    # --- shared init: identical starting weights for every (m, condition).
    #     Architecture is m-independent (T=s^L), so one snapshot serves all. ---
    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: val.cpu().clone() for k, val in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    def dkeys(d):
        return {f"d{L-ell}": d[ell] for ell in range(L)}

    def build_eval(seqs, lf):
        ex = torch.from_numpy(seqs.astype(np.int64)).to(device)
        yl = {ell: torch.from_numpy(lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
              for ell in range(L)}
        return ex, yl

    all_results = {}

    for m in ms:
        rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
        key = f"v{v}_s{s}_L{L}_m{m}"
        print(f"\n{'#'*78}\n# m={m}   occupancy m/v^(s-1)={m/v**(s-1):.3f}   {key}\n{'#'*78}")

        # --- aligned pool (all roots): source for BOTH the shared flat NTP corpus
        #     and the oracle aux target (per-level ancestor labels live here). ---
        print(f"Generating aligned pool ({pool_size:,} seqs, all roots)...")
        pool_seqs, pool_lf, _ = _generate_with_traces(rules, pool_size, data_seed)
        pool_x = torch.from_numpy(pool_seqs.astype(np.int64))                # (P,T) cpu
        pool_anc = [torch.from_numpy(pool_lf[ell].astype(np.int64)) for ell in range(L)]
        pool_roots = pool_lf[0][:, 0]                                        # (P,) root feat
        A_pool_idx = torch.from_numpy(
            np.where(np.isin(pool_roots, np.asarray(A)))[0].astype(np.int64))
        print(f"  A-root pool coverage: {A_pool_idx.numel():,}/{pool_size:,} "
              f"({A_pool_idx.numel()/pool_size*100:.1f}%)")

        # --- shared flat NTP corpus (all roots): phase-diverse random windows. ---
        corpus = pool_x.reshape(-1)                                         # (P*T,) cpu
        n_corpus = corpus.shape[0]

        # --- A-only flat corpus (for ntp_A's extra term): concat only A-root seqs. ---
        print("Generating A-only aligned pool (for ntp_A extra corpus)...")
        sub_seqs, _, _ = _generate_with_traces_roots(rules, pool_size, data_seed + 100, A)
        corpus_A = torch.from_numpy(sub_seqs.astype(np.int64)).reshape(-1)
        n_corpus_A = corpus_A.shape[0]
        del sub_seqs

        def get_ntp_batch(gen, corp=corpus, nc=n_corpus):
            ix = torch.randint(0, nc - T - 1, (batch_size,), generator=gen)
            idx = ix[:, None] + arangeT[None, :]                    # (B,T) windows
            return corp[idx].to(device), corp[idx + 1].to(device)

        def get_aux_batch(gen, pool_idx=None):
            """Aligned batch + per-level ancestor labels. pool_idx=None -> all roots
            (aux_all); pool_idx=A_pool_idx -> only A-root sequences (aux_A)."""
            if pool_idx is None:
                idx = torch.randint(0, pool_size, (batch_size,), generator=gen)
            else:
                sel = torch.randint(0, pool_idx.numel(), (batch_size,), generator=gen)
                idx = pool_idx[sel]
            x = pool_x[idx].to(device)
            labels = [pool_anc[ell][idx][:, anc_idx[ell]].to(device) for ell in range(L)]
            return x, labels

        def aux_loss(inter, labels, aux_heads):
            """Per-level ancestor CE at span-end positions, from each block's head;
            mean over blocks then over levels (the oracle direct deep target)."""
            head_out = {b: aux_heads[b](inter[b]).view(batch_size, T, L, v)
                        for b in sup_blocks}
            per_level = []
            for ell in range(L):
                se = spanend[ell]
                ces = [F.cross_entropy(head_out[b][:, se, ell, :].reshape(-1, v),
                                       labels[ell][:, se].reshape(-1)) for b in sup_blocks]
                per_level.append(torch.stack(ces).mean())
            return torch.stack(per_level).mean()

        def eval_ntp(model, corp=corpus, nc=n_corpus, n=10):
            model.eval()
            gen = torch.Generator().manual_seed(eval_seed + 5)
            tot = 0.0
            with torch.no_grad():
                for _ in range(n):
                    x, y = get_ntp_batch(gen, corp=corp, nc=nc)
                    _, loss = model(x, y)
                    tot += loss.item()
            return tot / n

        # --- eval sets (held out): A, complement, full ---
        print("Generating eval sets (A / complement / full)...")
        A_seqs, A_lf, _ = _generate_with_traces_roots(rules, n_eval_sequences, eval_seed, A)
        cp_seqs, cp_lf, _ = _generate_with_traces_roots(
            rules, n_eval_sequences, eval_seed + 1, complement)
        fl_seqs, fl_lf, _ = _generate_with_traces(rules, n_eval_sequences, eval_seed + 2)
        evals = {
            "A": build_eval(A_seqs, A_lf),
            "complement": build_eval(cp_seqs, cp_lf),
            "full": build_eval(fl_seqs, fl_lf),
        }

        def train_and_probe(cond):
            use_aux = cond in ("aux_all", "aux_A")
            use_ntp_extra = cond == "ntp_A"
            aux_pool_idx = A_pool_idx if cond == "aux_A" else None
            print(f"\n{'='*62}\n  CONDITION: {cond}  (m={m}, aux={use_aux}, "
                  f"ntp_extra={use_ntp_extra})\n{'='*62}")

            torch.manual_seed(seed)
            model = GPT(v, T, n_layer, n_head, n_embd).to(device)
            model.load_state_dict(init_state)          # identical init across conditions
            main_params = list(model.parameters())
            aux_heads = None
            if use_aux:
                # Seeded right after the (identical) GPT construction, so aux_all and
                # aux_A get IDENTICALLY initialized heads -> the only difference is
                # which roots the aligned aux batch samples from.
                aux_heads = nn.ModuleDict(
                    {b: nn.Linear(n_embd, L * v) for b in sup_blocks}).to(device)
                main_params += list(aux_heads.parameters())
            opt = torch.optim.AdamW(main_params, lr=lr, weight_decay=weight_decay)

            train_gen = torch.Generator().manual_seed(seed + 1)   # base flat NTP (all)
            aux_gen = torch.Generator().manual_seed(seed + 2)     # aligned aux batch
            ntpA_gen = torch.Generator().manual_seed(seed + 3)    # A-only flat NTP

            for step in range(n_steps):
                model.train()
                # ALWAYS: flat phase-diverse NTP over ALL roots (shared substrate).
                x, y = get_ntp_batch(train_gen)
                _, loss = model(x, y)
                base_ntp = loss.item()
                extra_val = None
                if use_aux:                            # direct deep target (aligned pass)
                    xa, labels = get_aux_batch(aux_gen, pool_idx=aux_pool_idx)
                    _, _, inter_a = model(xa, return_intermediates=True)
                    al = aux_loss(inter_a, labels, aux_heads)
                    loss = loss + lam * al
                    extra_val = al.item()
                elif use_ntp_extra:                    # extra flat NTP over A-only corpus
                    xA, yA = get_ntp_batch(ntpA_gen, corp=corpus_A, nc=n_corpus_A)
                    _, lA = model(xA, yA)
                    loss = loss + lam * lA
                    extra_val = lA.item()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                opt.step()
                if step % eval_interval == 0 or step == n_steps - 1:
                    vl = eval_ntp(model)
                    es = f" extra={extra_val:.4f}" if extra_val is not None else ""
                    print(f"    [{cond}] step {step:6d}: ntp={base_ntp:.4f} "
                          f"val={vl:.4f}{es}")

            # val loss on the shared (all-root) corpus and the A-only corpus
            val = {"full": eval_ntp(model),
                   "A": eval_ntp(model, corp=corpus_A, nc=n_corpus_A)}
            print(f"  [{cond}] val loss:  full={val['full']:.4f}  A={val['A']:.4f}")

            recovery = {}
            for ename, (ex, yl) in evals.items():
                r = _probe_model(model, ex, yl, block_names, L, v, device,
                                 probe_steps, probe_lr, mlp_hidden, mlp_steps,
                                 f"{cond}|{ename}")
                recovery[ename] = {"linear_best": dkeys(r["linear_best"]),
                                   "mlp_best": dkeys(r["mlp_best"])}
            del model, opt
            if aux_heads is not None:
                del aux_heads
            torch.cuda.empty_cache()
            return {"val": val, "recovery": recovery}

        out = {cond: train_and_probe(cond) for cond in cond_list}
        all_results[str(m)] = out

        # ============================================================
        # HEADLINE block for this m
        # ============================================================
        dlevels = [f"d{L-ell}" for ell in range(L)]                 # d6..d1 (root->leaf)
        print(f"\n{'='*78}")
        print(f"HEADLINE  m={m}   A={A}  (|A|={len(A)}/{v})")
        print(f"  chance(root) full = 1/v = {chance:.4f}   "
              f"chance(root) on A-eval = 1/|A| = {1/len(A):.4f}")
        print(f"  CAVEAT: on A-eval the deep-level ABSOLUTES are inflated (root chance")
        print(f"    1/|A|, high-level features constrained), but ALL conditions share the")
        print(f"    SAME A-eval, so the cross-condition comparison is unaffected.")
        print(f"  predictions: ntp shallow everywhere; aux_all deep on both at m=4 /")
        print(f"    capacity-limited at m=8; aux_A deep on A & shallow on B (SELECTIVE)")
        print(f"    and at high m A-depth > aux_all A-depth (CONCENTRATION ADVANTAGE);")
        print(f"    ntp_A ~ ntp floor (token-reweight can't buy depth).")

        # (1) depth on A-eval: four conditions side by side (MLP best)
        print(f"{'-'*78}")
        print(f"DEPTH ON A-eval  (MLP best-over-blocks)")
        hdr = f"  {'level':<8}" + "".join(f"{c:>10}" for c in cond_list)
        print(hdr)
        for d in dlevels:
            row = f"  {d:<8}"
            for c in cond_list:
                row += f"{out[c]['recovery']['A']['mlp_best'][d]:>10.3f}"
            print(row)

        # (2) selectivity of aux_A: A-eval vs complement-eval
        if "aux_A" in out:
            print(f"{'-'*78}")
            print(f"SELECTIVITY (aux_A):  depth on A-eval vs complement(B)-eval  (MLP best)")
            print(f"  {'level':<8}{'A-eval':>10}{'B-eval':>10}{'Δ(A−B)':>10}")
            aA = out["aux_A"]["recovery"]["A"]["mlp_best"]
            aB = out["aux_A"]["recovery"]["complement"]["mlp_best"]
            for d in dlevels:
                print(f"  {d:<8}{aA[d]:>10.3f}{aB[d]:>10.3f}{aA[d]-aB[d]:>+10.3f}")

        # (3) concentration advantage: aux_A − aux_all on A-eval
        if "aux_A" in out and "aux_all" in out:
            print(f"{'-'*78}")
            print(f"CONCENTRATION ADVANTAGE  (aux_A − aux_all) on A-eval  (MLP best)")
            aA = out["aux_A"]["recovery"]["A"]["mlp_best"]
            uA = out["aux_all"]["recovery"]["A"]["mlp_best"]
            print(f"  " + "  ".join(f"{d}:{aA[d]-uA[d]:+.3f}" for d in dlevels))

        # (4) compact secondary: every condition on all three evals (MLP, d1..d6)
        print(f"{'-'*78}")
        print(f"  secondary (MLP best, d1..d6):")
        for c in cond_list:
            for ename in ("A", "complement", "full"):
                mlp = out[c]["recovery"][ename]["mlp_best"]
                r = "  ".join(f"{d}:{mlp[d]:.3f}" for d in reversed(dlevels))  # d1..d6
                print(f"    {c:<8} on {ename:<10} {r}")
        print(f"  val loss:  " + "  ".join(
            f"{c}(full={out[c]['val']['full']:.3f} A={out[c]['val']['A']:.3f})"
            for c in cond_list))
        print(f"{'='*78}")

        del pool_x, pool_anc, corpus, corpus_A, evals
        torch.cuda.empty_cache()

    # --- save ---
    config = dict(v=v, s=s, depth=L, rule_seed=rule_seed, m_values=ms,
                  n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  subtree_roots=A, complement=complement, conditions=cond_list, lam=lam,
                  n_steps=n_steps, batch_size=batch_size, lr=lr,
                  weight_decay=weight_decay, pool_size=pool_size, data_seed=data_seed,
                  n_eval_sequences=n_eval_sequences, eval_seed=eval_seed,
                  probe_steps=probe_steps, mlp_hidden=mlp_hidden, mlp_steps=mlp_steps,
                  seed=seed, chance=chance, quick=quick)
    results = {"config": config, "results_by_m": all_results}

    stag = "-".join(str(r) for r in A)
    mtag = "-".join(str(x) for x in ms)
    descriptive_tag = (f"v{v}_s{s}_L{L}_distinct_A{len(A)}of{v}_r{stag}_m{mtag}_N{n_steps}"
                       f"{('_' + tag) if tag else ''}{'_quick' if quick else ''}")
    out_dir = f"{DATA_DIR}/rhm_spec_direct_target/{descriptive_tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/results.json")
    return results


@app.local_entrypoint()
def main(m_values: str = "4,8", subtree_roots: str = "0,1,2,3",
         conditions: str = "ntp,aux_all,aux_A,ntp_A", n_steps: int = 20000,
         quick: bool = False, tag: str = ""):
    spec_direct_target.remote(m_values=m_values, subtree_roots=subtree_roots,
                              conditions=conditions, n_steps=n_steps, quick=quick,
                              tag=tag)
