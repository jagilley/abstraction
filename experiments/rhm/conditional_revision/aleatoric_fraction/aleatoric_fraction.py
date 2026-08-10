"""How much of the state update is irreducible? The aleatoric fraction of h6[t+1].

Sizing step for `ideas/revision_not_surprisal.md` §8. NOTHING here trains, gates or
intervenes -- it is pure measurement on the frozen cached m4 base.

The question
------------
§8's temporal local loss (`h6[<=t] -> h6[t+1]`) has an exogenous conditioning gap,
so unlike the depth version it has a second way down that is not "be simpler":
become invariant to the component of x_{t+1} that is unpredictable in principle.
`local_loss/` could not test that -- it scored self-knowledge, ran on m2, and its
own Finding 1 shows ~90% of the term's range is the target shrinking in SCALE, a
gauge every downstream reader discards. Before running any training 2x2 we size
the prize with a scale-free readout that a gauge collapse cannot masquerade as.

The object
----------
§4 splits surprisal by BP; the same BP quantities split the STATE UPDATE by the
law of total variance:

    Var( h6[t+1] | x_<=t )  =  E_z[ Var(h6[t+1] | z_<=D, x_<=t) ]   ALEATORIC  A_D
                            +  Var_z( E[h6[t+1] | z_<=D, x_<=t] )   EPISTEMIC  E_D

and the readout is `A_D / (A_D + E_D)`, a ratio of variances and therefore
scale-free. Two things make it exact and cheap rather than Monte-Carlo:

  * The model is causal, so conditional on the prefix `h6[t+1]` takes exactly
    v = 16 values -- one per arriving token. Both variances are 16-term weighted
    sums over activation vectors we can just compute.
  * The weights are BP-exact and already in `oracle.py`: `leaf_post` weights the
    outer variance, `irr_post[D]` (clamping z_<=D to its TRUE value) the inner.
    The observed sequence's latent assignment IS a draw from P(z | x_<=t), so
    one sample per cell is an unbiased estimator of the outer expectation. A
    token that makes the sequence an illegal production gets probability zero
    from BP and drops out of both sums.

Since A_D + E_D = Var(h6[t+1] | x_<=t) identically, E_D is DEFINED as the
difference and the readout is just `mean A_D / mean Total`. Only A_D is estimated.

The NTP-protected floor
-----------------------
The hypothesis presumes the model CAN shed aleatoric content without hurting
next-token prediction. That is position-dependent and is a fact about the DGP, so
it is checkable before any model is involved. Two candidate arriving tokens are
NTP-INTERCHANGEABLE iff the exact Bayes distribution of everything still to be
predicted is identical under both -- then an NTP-optimal model may map them to the
same state. On the RHM that criterion is exact and local (`ntp_classes`): the
arriving token x_p influences the rest of the world only through the joint

    J(a)  =  down_u(f) . #{r : rules[L-1][f, r] matches the pair}

over its leaf-parent u's feature f, times the still-unobserved sibling b when p
OPENS a pair. a ~ a' iff J(a) is proportional to J(a'). `verify_classes` proves
this against direct BP inside every run rather than asserting it.

That gives a SECOND law-of-total-variance split, nested inside A_D:

    A_D  =  Var_class( E[h | class] )  +  E_class[ Var(h | class) ]
            A_prot  (NTP forces it)      A_free  (the model may discard it)

so `A_free / Total` is the prize a training intervention could win, and
`A_prot / (A_prot + E_D)` is the floor `A/(A+E)` could fall to. Both are
conservative in the same direction: NTP could satisfy itself by routing a
distinction through a LOWER block than post_block6, so A_prot over-counts what
post_block6 specifically is forced to hold, and A_free under-counts the prize.

References computed alongside, because a bare fraction is uninterpretable
-----------------------------------------------------------------------
  post_embed   the same split on the token embedding -- a representation that by
               construction keeps EVERYTHING about the arriving token and nothing
               else. The "sheds nothing" ceiling for this readout.
  gini         (1 - sum_a q_a^2) / (1 - sum_a p_a^2), the value the readout takes
               for a representation that keeps token identity under isotropic
               geometry. Model-free.
  all blocks   post_embed + post_block0..7, so shedding-with-depth is visible.

Run:
  # DGP-only checks + criterion proof, local, CPU, ~2 min, no Modal, no model
  cd experiments && python3 -m rhm.conditional_revision.aleatoric_fraction.aleatoric_fraction

  # smoke (attached, ~4 min)
  modal run -m rhm.conditional_revision.aleatoric_fraction.aleatoric_fraction::aleatoric \
      --n-seq 64 --tag smoke
  # the real thing (~25 min on an L4; the base loads from cache and is never trained)
  modal run --detach -m rhm.conditional_revision.aleatoric_fraction.aleatoric_fraction::aleatoric \
      --n-seq 2000 --tag af1
"""

import json
import os

import modal
import numpy as np

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-aleatoric-fraction", image=image)


def tb_key(v, s, L, m):
    """Volume dir; the `_distinct` suffix is real (see conditional_revision.py)."""
    return f"{setting_key(v, s, L, m)}_distinct"


# ---------------------------------------------------------------------------
# NTP-equivalence of arriving tokens -- exact, local, and proved against BP
# ---------------------------------------------------------------------------

def pair_counts(rules):
    """cnt[f, a, b] = #{r : rules[L-1][f, r] == (a, b)}, the leaf-emitting level."""
    rl = rules[-1]
    v, m, s = rl.shape
    assert s == 2, "the local criterion is written for s = 2"
    cnt = np.zeros((v, v, v))
    for f in range(v):
        for r in range(m):
            cnt[f, rl[f, r, 0], rl[f, r, 1]] += 1.0
    return cnt


def down_leafparent(rules, seqs, plen):
    """down[L-1] : (B, s^(L-1), v) -- the ROOT-SIDE message into each leaf-parent
    given x_{<plen}. Excludes that node's own subtree evidence, which is exactly
    what makes it the right thing to multiply the candidate token's likelihood by."""
    from rhm.conditional_revision import oracle as ORC
    v = rules[0].shape[0]
    L = len(rules)
    up = ORC._upward(ORC._leaf_evidence(seqs, plen, v), rules)
    return ORC._downward(up, rules)[L - 1]


def ntp_classes(rules, seqs, p, cnt=None, dparent=None, tol=1e-12):
    """(B, v) NTP-equivalence class id of each candidate token arriving at p.

    Class -1 = the token is impossible given x_{<p} (BP gives it probability 0).
    Otherwise tokens sharing an id induce a bit-identical exact-Bayes future, so an
    NTP-optimal model is free to represent them identically.
    """
    v = rules[0].shape[0]
    B = seqs.shape[0]
    if cnt is None:
        cnt = pair_counts(rules)
    if dparent is None:
        dparent = down_leafparent(rules, seqs, p)
    du = dparent[:, p // 2, :]                                  # (B, v_f)
    if p % 2 == 1:            # completing: the sibling is already observed
        K = cnt[:, seqs[:, p - 1], :].transpose(1, 2, 0)        # (B, a, f)
        J = K * du[:, None, :]
    else:                     # opening: the sibling is still to be predicted
        J = cnt.transpose(1, 0, 2)[None] * du[:, None, :, None]  # (B, a, f, b)
    J = J.reshape(B, v, -1)
    z = J.sum(-1)                                                # (B, v) ~ P(x_p = a)
    Jn = J / np.clip(z, 1e-300, None)[:, :, None]
    eq = np.zeros((B, v, v), dtype=bool)
    for a in range(v):
        eq[:, :, a] = np.abs(Jn - Jn[:, a:a + 1, :]).max(-1) < tol
    cls = eq.argmax(-1).astype(np.int64)      # smallest equivalent index = the rep
    cls[z <= 1e-300] = -1
    return cls, z


def verify_classes(rules, seqs, positions, verbose=True):
    """Prove `ntp_classes` against direct BP: same-class tokens must give a
    bit-identical posterior over EVERY latent node and clique and over the next
    token; different-class tokens must differ somewhere."""
    from rhm.conditional_revision import oracle as ORC
    v = rules[0].shape[0]
    L = len(rules)
    B = seqs.shape[0]
    cnt = pair_counts(rules)
    n_same = n_diff = 0
    max_within, min_between = 0.0, np.inf
    for p in positions:
        cls, _ = ntp_classes(rules, seqs, p, cnt)
        post = {}
        for a in range(v):
            sq = seqs.copy()
            sq[:, p] = a
            nodes, cliq, leaf = ORC.prefix_beliefs(rules, sq, p + 1, L - 1)
            post[a] = np.concatenate(
                [nd.reshape(B, -1) for nd in nodes]
                + [cq.reshape(B, -1) for cq in cliq]
                + ([leaf] if leaf is not None else []), axis=1)
        for b in range(B):
            legal = np.where(cls[b] >= 0)[0]
            for i, a in enumerate(legal):
                for a2 in legal[i + 1:]:
                    d = float(np.abs(post[a][b] - post[a2][b]).max())
                    if cls[b, a] == cls[b, a2]:
                        n_same += 1
                        max_within = max(max_within, d)
                    else:
                        n_diff += 1
                        min_between = min(min_between, d)
    out = {"n_same_class_pairs": n_same, "n_diff_class_pairs": n_diff,
           "max_within_class_posterior_diff": max_within,
           "min_between_class_posterior_diff": (None if not np.isfinite(min_between)
                                                else min_between),
           "positions": list(map(int, positions)),
           "passed": bool(max_within < 1e-12
                          and (n_diff == 0 or min_between > 1e-9))}
    if verbose:
        print(f"  criterion vs direct BP: same-class pairs {n_same} "
              f"max |dposterior| {max_within:.2e} | diff-class pairs {n_diff} "
              f"min |dposterior| {min_between:.2e} -> "
              f"{'EXACT' if out['passed'] else 'WRONG'}", flush=True)
    return out


# ---------------------------------------------------------------------------
# DGP check: is a COMPLETED constituent's realising rule irrelevant to the future?
# ---------------------------------------------------------------------------

def _realise(rules, feats, d, rng):
    """Sample the span under level-d nodes with values `feats` (n, n_nodes)."""
    v, m, s = rules[0].shape
    L = len(rules)
    cur = feats
    for ell in range(d, L):
        n_nodes = cur.shape[1]
        rc = rng.integers(0, m, size=(cur.shape[0], n_nodes))
        nxt = np.empty((cur.shape[0], n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][cur[:, j], rc[:, j]]
        cur = nxt
    return cur


def synonym_swap_check(rules, n=400, seed=31, verbose=True):
    """Jasper's floor argument, checked on the DGP alone.

    For a constituent that a token has just COMPLETED, re-realise its whole span
    from the SAME feature but different rule choices, and ask whether the exact
    Bayes next-token distribution moves. If it never does, the realising rule is
    NTP-irrelevant once the constituent closes and the model is free to forget it.
    """
    from rhm.conditional_revision import oracle as ORC
    from rhm.rhm_latent_loop import _generate_with_traces
    v, m, s = rules[0].shape
    L = len(rules)
    T = s ** L
    seqs, lf, _ = _generate_with_traces(rules, n, seed)
    rng = np.random.default_rng(seed + 1)
    rows = {}
    for d in range(1, L):
        span = s ** (L - d)
        for u in (0, 1):
            p = (u + 1) * span - 1                     # the completing position
            if p >= T - 1:
                continue                                # no next token to compare
            alt = seqs.copy()
            alt[:, u * span:(u + 1) * span] = _realise(
                rules, lf[d][:, u:u + 1], d, rng)
            changed = np.any(alt[:, u * span:(u + 1) * span]
                             != seqs[:, u * span:(u + 1) * span], axis=1)
            _, _, lp0 = ORC.prefix_beliefs(rules, seqs, p + 1, 0)
            _, _, lp1 = ORC.prefix_beliefs(rules, alt, p + 1, 0)
            tv = 0.5 * np.abs(lp0 - lp1).sum(-1)
            k = f"d{L - d}_u{u}"                        # repo probe naming
            rows[k] = {
                "tree_level": d, "node": u, "span": int(span), "position": int(p),
                "frac_realisation_changed": float(changed.mean()),
                "mean_TV_next_token": float(tv[changed].mean()) if changed.any() else 0.0,
                "frac_TV_exactly_zero": float((tv[changed] < 1e-12).mean())
                                        if changed.any() else 1.0,
                "max_TV": float(tv[changed].max()) if changed.any() else 0.0,
            }
    if verbose:
        print("  completed-constituent synonym swap (exact Bayes next token):")
        for k, r in rows.items():
            print(f"    {k:>8} span {r['span']:>2} pos {r['position']:>2}  "
                  f"changed {r['frac_realisation_changed']:.3f}  "
                  f"mean TV {r['mean_TV_next_token']:.4f}  "
                  f"frac TV==0 {r['frac_TV_exactly_zero']:.3f}  "
                  f"max TV {r['max_TV']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------

def _nn(x):
    """None -> nan, so a legitimate 0.0 is not printed as a missing value."""
    return float("nan") if x is None else float(x)


def _ratio(num, den):
    d = float(den.sum())
    return float(num.sum() / d) if abs(d) > 1e-30 else float("nan")


def _boot(num, den, n_boot=400, seed=5):
    """Bootstrap the ratio-of-means over SEQUENCES (rows), which are the
    independent unit -- positions within a sequence are not."""
    rng = np.random.default_rng(seed)
    n = num.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, size=n)
        vals[b] = _ratio(num[ix], den[ix])
    return {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals)),
            "lo95": float(np.nanpercentile(vals, 2.5)),
            "hi95": float(np.nanpercentile(vals, 97.5))}


def _subspace_stats(C_A, C_E, ks=(4, 8, 16, 32, 64)):
    """Do the aleatoric and epistemic covariances occupy separable subspaces?"""
    C = C_A.shape[0]
    wA, UA = np.linalg.eigh(C_A)
    wE, UE = np.linalg.eigh(C_E)
    wA, UA = wA[::-1], UA[:, ::-1]
    wE, UE = wE[::-1], UE[:, ::-1]
    trA = float(np.clip(wA, 0, None).sum())
    out = {"trace_A": trA, "trace_E": float(np.clip(wE, 0, None).sum()),
           "participation_ratio_A": float(wA.sum() ** 2 / (wA ** 2).sum()),
           "participation_ratio_E": float(wE.sum() ** 2 / (wE ** 2).sum()),
           "per_k": {}}
    for k in ks:
        if k > C:
            continue
        PA, PE = UA[:, :k], UE[:, :k]
        cap_E = float(np.trace(PE.T @ C_A @ PE)) / max(trA, 1e-30)
        cap_A = float(np.clip(wA[:k], 0, None).sum()) / max(trA, 1e-30)
        cos = np.linalg.svd(PA.T @ PE, compute_uv=False)
        out["per_k"][str(k)] = {
            "A_variance_in_E_topk": cap_E,
            "A_variance_in_A_topk_ceiling": cap_A,
            "random_subspace_null": k / C,
            "mean_cos2_principal_angles": float((cos ** 2).mean()),
            "n_angles_cos_gt_0p9": int((cos > 0.9).sum()),
        }
    return out


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def aleatoric(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    primary_block: str = "post_block6",
    base_steps: int = 12000, base_ckpt: str = "",
    n_seq: int = 2000, eval_seed: int = 999, seed: int = 42,
    chunk_gpu: int = 64, chunk_oracle: int = 250,
    n_boot: int = 400, n_verify: int = 6, n_swap: int = 400,
    tag: str = "",
):
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.conditional_revision import oracle as ORC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    Ds = list(range(L))
    blocks = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    print("=" * 78)
    print(f"ALEATORIC FRACTION OF THE STATE UPDATE   {key}  "
          f"{n_layer}L/{n_head}H/{n_embd}D  primary={primary_block}")
    print(f"  n_seq={n_seq}  T={T}  v={v}  cells={n_seq * (T - 1):,}")
    print("=" * 78, flush=True)

    # ---------------- the frozen base (never trained here) ----------------
    ckpt = base_ckpt or (f"{DATA_DIR}/{key}/conditional_revision/"
                         f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_"
                         f"seed{seed}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"cached base not found at {ckpt} -- this cut never trains one. "
            f"Run conditional_revision::gate0 first, on the chromatic workspace.")
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"loaded frozen base <- {ckpt}", flush=True)

    # ---------------- DGP-only checks ----------------
    print("\n--- DGP checks (no model involved) ---", flush=True)
    vseqs, _, _ = _generate_with_traces(rules, n_verify, eval_seed + 77)
    rngp = np.random.default_rng(3)
    vpos = sorted(rngp.choice(np.arange(1, T - 1), size=16, replace=False).tolist())
    verify = verify_classes(rules, vseqs, vpos)
    swap = synonym_swap_check(rules, n=n_swap, seed=eval_seed + 5)

    # ---------------- exact BP weights ----------------
    print(f"\n--- oracle: BP weights on {n_seq} aligned sequences ---", flush=True)
    seqs, lf, _ = _generate_with_traces(rules, n_seq, eval_seed)
    orc = ORC.revision_and_entropy(rules, seqs, lf, Ds=Ds, chunk=chunk_oracle,
                                   verbose=False, return_leaf_posteriors=True)
    chk = ORC.self_check(orc)
    P = orc["leaf_post"]                                  # (n, T-1, v)
    Q = {D: orc["irr_post"][D] for D in Ds}                # (n, T-1, v)
    print(f"  identity self-check passed={chk['passed']}  "
          f"H_tot {chk['mean_H_tot']:.4f}", flush=True)
    for D in Ds:
        r = chk["per_D"][f"D{D}"]
        print(f"    D{D} ({r['d_name']}): E[B] {r['mean_B_joint']:.4f}  "
              f"H_irr {r['mean_H_irr']:.4f}  abs_err {r['abs_err']:.2e}", flush=True)
    # the emitted posteriors must reproduce the entropies the self-check certified
    ent = lambda x: -(np.where(x > 0, x * np.log(np.clip(x, 1e-300, None)), 0.0)).sum(-1)
    post_err = {"H_tot": float(np.abs(ent(P) - orc["H_tot"]).max()),
                **{f"H_irr_D{D}": float(np.abs(ent(Q[D]) - orc["H_irr"][D]).max())
                   for D in Ds},
                "P_sums_to_1": float(np.abs(P.sum(-1) - 1).max()),
                "Q_sums_to_1": float(max(np.abs(Q[D].sum(-1) - 1).max() for D in Ds))}
    print(f"  emitted-posterior consistency: max err "
          f"{max(post_err.values()):.2e}", flush=True)

    # ---------------- NTP-equivalence classes ----------------
    print("\n--- NTP-equivalence classes for every (sequence, arriving token) ---",
          flush=True)
    cnt = pair_counts(rules)
    CLS = np.zeros((n_seq, T - 1, v), dtype=np.int64)
    jz_err = 0.0
    for c0 in range(0, n_seq, chunk_oracle):
        c1 = min(n_seq, c0 + chunk_oracle)
        sq = seqs[c0:c1]
        for p in range(1, T):
            dp = down_leafparent(rules, sq, p)
            cls, z = ntp_classes(rules, sq, p, cnt, dp)
            CLS[c0:c1, p - 1] = cls
            zz = z / np.clip(z.sum(-1, keepdims=True), 1e-300, None)
            jz_err = max(jz_err, float(np.abs(zz - P[c0:c1, p - 1]).max()))
    print(f"  local criterion's implied P(x_p|x_<p) vs the oracle's: "
          f"max err {jz_err:.2e}", flush=True)
    n_cls = np.array([[len(np.unique(CLS[i, t][CLS[i, t] >= 0]))
                       for t in range(T - 1)] for i in range(min(n_seq, 200))])
    n_leg = (CLS[:min(n_seq, 200)] >= 0).sum(-1)
    print(f"  mean legal tokens {n_leg.mean():.2f}  mean classes {n_cls.mean():.2f}  "
          f"frac cells with a mergeable pair {(n_leg > n_cls).mean():.3f}", flush=True)

    # ---------------- counterfactual activations + the two splits ----------------
    print(f"\n--- frozen forward passes: h[p] for all {v} arriving tokens, "
          f"every position ---", flush=True)
    TOT = {b: np.zeros((n_seq, T - 1)) for b in blocks}
    ALE = {b: {D: np.zeros((n_seq, T - 1)) for D in Ds} for b in blocks}
    PROT = {b: {D: np.zeros((n_seq, T - 1)) for D in Ds} for b in blocks}
    S_tot = np.zeros((n_embd, n_embd))
    S_ale = {D: np.zeros((n_embd, n_embd)) for D in Ds}
    S_prot = {D: np.zeros((n_embd, n_embd)) for D in Ds}
    ident_err = 0.0

    nB, nD = len(blocks), len(Ds)
    kprim = blocks.index(primary_block)
    seq_t = torch.from_numpy(seqs.astype(np.int64))
    ar_v = torch.arange(v, device=device)
    for c0 in range(0, n_seq, chunk_gpu):
        c1 = min(n_seq, c0 + chunk_gpu)
        B = c1 - c0
        arB = torch.arange(B, device=device)
        base_x = seq_t[c0:c1].to(device)
        Pc = torch.from_numpy(P[c0:c1]).float().to(device)               # (B,T-1,v)
        Qc = torch.stack([torch.from_numpy(Q[D][c0:c1]).float().to(device)
                          for D in Ds], 1)                               # (B,nD,T-1,v)
        Cc = torch.from_numpy(CLS[c0:c1]).to(device)                     # (B,T-1,v)
        with torch.no_grad():                       # the untouched forward pass
            _, _, ref = model(base_x, return_intermediates=True)
        ref_p = ref[primary_block].float()
        for p in range(1, T):
            xt = base_x[:, :p + 1].unsqueeze(1).repeat(1, v, 1)          # (B,v,p+1)
            xt[:, :, p] = ar_v[None, :]
            with torch.no_grad():
                _, _, inter = model(xt.reshape(B * v, p + 1),
                                    return_intermediates=True)
                # inter[b][:, -1, :] is (B*v, C) laid out as (sequence, token);
                # the block axis must be permuted in, NOT reshaped in.
                HB = torch.stack([inter[b][:, -1, :] for b in blocks], 1) \
                          .reshape(B, v, nB, n_embd).permute(0, 2, 1, 3) \
                          .contiguous().float()                          # (B,nB,v,C)
            t = p - 1
            wp = Pc[:, t]                                                # (B,v)
            WQ = Qc[:, :, t]                                             # (B,nD,v)
            cls_p = Cc[:, t]
            M = (torch.nn.functional.one_hot(cls_p.clamp(min=0), v).float()
                 * (cls_p >= 0).float()[..., None])                      # (B,v,vc)

            # substitution + cropping must reproduce the plain forward pass
            ident_err = max(ident_err, float(
                (HB[arB, kprim, base_x[:, p]] - ref_p[:, p]).abs().max()))

            # --- outer variance: over the arriving token, prefix-weighted ---
            mu_p = torch.einsum('bt,bktc->bkc', wp, HB)
            d2 = ((HB - mu_p[:, :, None, :]) ** 2).sum(-1)               # (B,nB,v)
            tot = (wp[:, None, :] * d2).sum(-1)                          # (B,nB)
            # --- inner variance: z_<=D clamped to its TRUE value ---
            mu_q = torch.einsum('bdt,bktc->bkdc', WQ, HB)                # (B,nB,nD,C)
            dq = ((HB[:, :, None] - mu_q[:, :, :, None, :]) ** 2).sum(-1)
            ale = (WQ[:, None] * dq).sum(-1)                             # (B,nB,nD)
            # --- NTP-forced part of the inner variance: between classes ---
            WM = WQ[..., None] * M[:, None]                              # (B,nD,v,vc)
            wc = WM.sum(2)                                               # (B,nD,vc)
            mu_c = (torch.einsum('bdtc,bkte->bkdce', WM, HB)
                    / wc.clamp(min=1e-20)[:, None, :, :, None])
            dc = ((mu_c - mu_q[:, :, :, None, :]) ** 2).sum(-1)          # (B,nB,nD,vc)
            prot = (wc[:, None] * dc).sum(-1)                            # (B,nB,nD)

            tot_n, ale_n, prot_n = (tot.cpu().numpy(), ale.cpu().numpy(),
                                    prot.cpu().numpy())
            for k, b in enumerate(blocks):
                TOT[b][c0:c1, t] = tot_n[:, k]
                for j, D in enumerate(Ds):
                    ALE[b][D][c0:c1, t] = ale_n[:, k, j]
                    PROT[b][D][c0:c1, t] = prot_n[:, k, j]

            # --- covariances, primary block only ---
            Hp = HB[:, kprim]                                            # (B,v,C)
            Zt = torch.sqrt(wp)[..., None] * (Hp - mu_p[:, kprim, None, :])
            S_tot += torch.einsum('bvc,bvd->cd', Zt, Zt).cpu().numpy()
            Za = torch.sqrt(WQ)[..., None] * (Hp[:, None] - mu_q[:, kprim, :, None, :])
            S_a = torch.einsum('bdvc,bdve->dce', Za, Za).cpu().numpy()
            Zp = (torch.sqrt(wc)[..., None]
                  * (mu_c[:, kprim] - mu_q[:, kprim, :, None, :]))
            S_p = torch.einsum('bdkc,bdke->dce', Zp, Zp).cpu().numpy()
            for j, D in enumerate(Ds):
                S_ale[D] += S_a[j]
                S_prot[D] += S_p[j]
        print(f"  activations: {c1}/{n_seq} sequences  "
              f"(substitution-vs-plain-forward max err {ident_err:.2e})", flush=True)

    # ---------------- aggregate ----------------
    pos_top_level = np.array(
        [min(ell for ell in range(L + 1) if (p + 1) % (s ** (L - ell)) == 0)
         for p in range(T)], dtype=np.int64)
    arrival_level = pos_top_level[1:]                       # indexed by t = p - 1
    # Model-free references: what the readout reads for a representation that keeps
    # token identity under isotropic geometry (<e_a, e_b> = 0, ||e_a|| equal).
    #   Var(h | w)  = c (1 - sum_a w_a^2)
    #   A_free      = c (1 - sum_c (sum_{a in c} q_a^2) / w_c)
    # so `gini_free` is the largest prize any token-identity code could offer, and
    # is the honest ceiling for `prize` -- a bare 0.01 means nothing without it.
    gini_num = {D: 1.0 - (Q[D] ** 2).sum(-1) for D in Ds}
    gini_den = 1.0 - (P ** 2).sum(-1)
    gini_free = {}
    for D in Ds:
        acc = np.zeros((n_seq, T - 1))
        for c0 in range(0, n_seq, chunk_oracle):
            c1 = min(n_seq, c0 + chunk_oracle)
            oh = np.eye(v + 1)[CLS[c0:c1] + 1][..., 1:]      # (b,T-1,v_tok,v_cls)
            qd = Q[D][c0:c1]
            w_c = np.einsum('ntav,nta->ntv', oh, qd)
            s2_c = np.einsum('ntav,nta->ntv', oh, qd ** 2)
            acc[c0:c1] = 1.0 - (s2_c / np.clip(w_c, 1e-300, None)).sum(-1)
        gini_free[D] = acc

    def block_row(b, mask=None):
        sel = (slice(None), slice(None)) if mask is None else (slice(None), mask)
        tot = TOT[b][sel]
        row = {"n_cells": int(tot.size), "mean_total_var": float(tot.mean()),
               "frac_cells_zero_total": float((tot < 1e-12).mean())}
        for D in Ds:
            a = ALE[b][D][sel]
            pr = PROT[b][D][sel]
            row[f"D{D}"] = {
                "d_name": f"d{L - D}",
                "aleatoric_fraction": _ratio(a, tot),
                "ntp_floor_fraction": (_ratio(pr, tot - a + pr)
                                       if abs((tot - a + pr).sum()) > 1e-30 else None),
                "prize_free_over_total": _ratio(a - pr, tot),
                "protected_over_total": _ratio(pr, tot),
                "free_over_aleatoric": _ratio(a - pr, a),
                "mean_aleatoric_var": float(a.mean()),
                "gini_reference": _ratio(gini_num[D][sel], gini_den[sel]),
                "gini_free_reference": _ratio(gini_free[D][sel], gini_den[sel]),
            }
        return row

    results_blocks = {b: block_row(b) for b in blocks}
    by_level = {}
    for ell in range(L + 1):
        mk = (arrival_level == ell)
        if not mk.any():
            continue
        by_level[f"arr_level{ell}"] = {
            "n_positions": int(mk.sum()),
            "opens_a_pair": bool(ell == L),
            "mean_H_tot": float(orc["H_tot"][:, mk].mean()),
            **{f"mean_H_irr_D{D}": float(orc["H_irr"][D][:, mk].mean()) for D in Ds},
            **{b: block_row(b, mk) for b in (primary_block, "post_embed")},
        }

    boot = {}
    for D in Ds:
        boot[f"D{D}"] = {
            "aleatoric_fraction": _boot(ALE[primary_block][D], TOT[primary_block],
                                        n_boot=n_boot),
            "prize_free_over_total": _boot(
                ALE[primary_block][D] - PROT[primary_block][D],
                TOT[primary_block], n_boot=n_boot),
        }

    N = float(n_seq * (T - 1))
    subspace = {}
    for D in Ds:
        C_A = S_ale[D] / N
        C_E = (S_tot - S_ale[D]) / N
        subspace[f"D{D}"] = _subspace_stats(C_A, C_E)

    # ---------------- report ----------------
    pb = results_blocks[primary_block]
    print(f"\n{'=' * 78}\nPRIMARY -- {primary_block}, aleatoric fraction of the state "
          f"update\n{'=' * 78}")
    print(f"  {'D':<4}{'name':<6}{'A/(A+E)':>10}{'boot 95%':>18}{'NTP floor':>11}"
          f"{'prize':>9}{'prizeCEIL':>10}{'free/A':>9}{'gini ref':>10}{'embed ref':>11}")
    for D in Ds:
        r = pb[f"D{D}"]
        bo = boot[f"D{D}"]["aleatoric_fraction"]
        print(f"  {D:<4}{r['d_name']:<6}{r['aleatoric_fraction']:>10.4f}"
              f"   [{bo['lo95']:.4f},{bo['hi95']:.4f}]"
              f"{_nn(r['ntp_floor_fraction']):>11.4f}"
              f"{r['prize_free_over_total']:>9.4f}{r['gini_free_reference']:>10.4f}"
              f"{r['free_over_aleatoric']:>9.4f}{r['gini_reference']:>10.4f}"
              f"{results_blocks['post_embed'][f'D{D}']['aleatoric_fraction']:>11.4f}")
    print(f"\n  depth profile (D=5 / d1, the tightest synonym reading). `free/A` is the"
          f"\n  discardable share OF the aleatoric budget -- scale-free within it:")
    for b in blocks:
        r = results_blocks[b]['D5']
        print(f"    {b:<14} A/(A+E) {r['aleatoric_fraction']:>7.4f}"
              f"   prize {r['prize_free_over_total']:>7.4f}"
              f"   free/A {r['free_over_aleatoric']:>7.4f}"
              f"   mean tot var {results_blocks[b]['mean_total_var']:.4f}")
    print(f"    {'(gini ceiling)':<14} A/(A+E) {pb['D5']['gini_reference']:>7.4f}"
          f"   prize {pb['D5']['gini_free_reference']:>7.4f}")
    print(f"\n  by arrival level ({primary_block}, D=5 / d1):")
    print(f"  {'level':<12}{'npos':>5}{'H_tot':>8}{'H_irr5':>8}{'A/(A+E)':>10}"
          f"{'floor':>8}{'prize':>8}{'prizeCEIL':>10}{'free/A':>8}{'embedref':>10}")
    for k, r in by_level.items():
        pr = r[primary_block]["D5"]
        print(f"  {k:<12}{r['n_positions']:>5}{r['mean_H_tot']:>8.4f}"
              f"{r['mean_H_irr_D5']:>8.4f}{pr['aleatoric_fraction']:>10.4f}"
              f"{_nn(pr['ntp_floor_fraction']):>8.4f}"
              f"{pr['prize_free_over_total']:>8.4f}"
              f"{pr['gini_free_reference']:>10.4f}{pr['free_over_aleatoric']:>8.4f}"
              f"{r['post_embed']['D5']['aleatoric_fraction']:>10.4f}")
    print(f"\n  substitution-vs-plain-forward max |dh| = {ident_err:.2e} (fp32 noise)")
    print(f"\n  A/E subspace separability ({primary_block}, D=5):")
    for k, r in subspace["D5"]["per_k"].items():
        print(f"    k={k:<4} A-var in E's top-k {r['A_variance_in_E_topk']:.3f} "
              f"(ceiling {r['A_variance_in_A_topk_ceiling']:.3f}, "
              f"random {r['random_subspace_null']:.3f})  "
              f"mean cos^2 {r['mean_cos2_principal_angles']:.3f}")
    print(f"  participation ratio: A {subspace['D5']['participation_ratio_A']:.1f}  "
          f"E {subspace['D5']['participation_ratio_E']:.1f}  (of {n_embd}; "
          f"residual_decomposition: directional claims below ~100 dirs are suspect)",
          flush=True)

    results = {
        "config": {"v": v, "s": s, "L": L, "m": m, "rule_seed": rule_seed,
                   "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
                   "primary_block": primary_block, "base_ckpt": ckpt,
                   "n_seq": n_seq, "n_cells": int(n_seq * (T - 1)),
                   "eval_seed": eval_seed, "seed": seed, "n_boot": n_boot,
                   "tag": tag},
        "oracle_self_check": chk,
        "emitted_posterior_consistency": post_err,
        "criterion_vs_bp": verify,
        "local_criterion_vs_oracle_leafpost_maxerr": jz_err,
        "identity_activation_maxerr": ident_err,
        "class_stats": {"mean_legal_tokens": float(n_leg.mean()),
                        "mean_classes": float(n_cls.mean()),
                        "frac_cells_mergeable": float((n_leg > n_cls).mean())},
        "dgp_synonym_swap": swap,
        "blocks": results_blocks,
        "by_arrival_level": by_level,
        "bootstrap_primary": boot,
        "subspace_primary": subspace,
    }
    out_dir = f"{DATA_DIR}/{key}/conditional_revision"
    os.makedirs(out_dir, exist_ok=True)
    name = f"aleatoric_fraction{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results


# ---------------------------------------------------------------------------
# local, model-free checks (`python3 -m ...aleatoric_fraction.aleatoric_fraction`)
# ---------------------------------------------------------------------------

def _local_checks():
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    v, s, L, m = 16, 2, 6, 4
    T = s ** L
    rules = generate_rules_distinct(v, s, L, m, seed=0)
    print("=" * 74)
    print(f"aleatoric_fraction DGP checks: v{v}/s{s}/L{L}/m{m}")
    print("=" * 74)
    seqs, _, _ = _generate_with_traces(rules, 6, 1076)
    rng = np.random.default_rng(3)
    pos = sorted(rng.choice(np.arange(1, T - 1), size=16, replace=False).tolist())
    out = verify_classes(rules, seqs, pos)
    assert out["passed"], "the NTP-equivalence criterion disagrees with BP"
    synonym_swap_check(rules, n=300, seed=1004)
    print("local checks PASSED")


if __name__ == "__main__":
    _local_checks()
