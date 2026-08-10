"""Retention: does the state still carry which synonym realised a CLOSED constituent,
as a function of how long ago it closed?

Why this exists
---------------
`local_loss/`'s causal synonym readout (`_swap_sensitivity` -> `syn/str = 0.635`) perturbs
the LAST level-(L-2) constituent and reads at the LAST position.  The perturbed span
*contains* the read position and nothing follows it, so NTP places no constraint on that
state at all: it is a retention readout taken at **zero retention distance**, and it cannot
distinguish "the model retains synonym identity" from "the model is still inside the
constituent".  Three other numbers in this repo, all taken where discarding is possible,
point the other way (`probe_diag`'s past-node class at 0.20 against a ~1.0 Bayes ceiling;
the tracking appendix's 0.946 -> 0.350 collapse across one constituent boundary;
`SLEEP_CHUNKING_RHM`'s external-role waist).

`aleatoric_fraction/` measured the same shape with an *instantaneous* NTP-equivalence test
(two arriving tokens interchangeable **from the moment of arrival**), so content that is
load-bearing for w steps and then stops is fully protected at arrival and scores zero.
This module supplies the missing axis: **distance**.

The object
----------
For a tree node n = (level d, index j) with leaf span S = s^(L-d) closing at leaf position
e = (j+1)S - 1, perturb n's realisation and read at t = e + w.

    leaf   bump the rule choice of every leaf-emitting (level L-1) node inside n's span.
           Every latent feature at every level is UNCHANGED; only which production
           realises each leaf pair moves.  This is `local_loss._twin_leaves`'s "synonym".
    top    bump n's OWN rule choice (level d).  n's feature is unchanged; every
           descendant feature is re-realised.  "a different production of the same
           constituent".
    str    leaf + top -- exactly `local_loss._twin_leaves`'s "structure" arm, so
           (d = L-2, j = s^(L-2)-1, w = 0) reproduces the incumbent `syn/str` number.
    rand   the span replaced by uniform tokens (off-DGP scale reference)
    other  an independent sequence (global-spread normaliser)

**The perturbation is frozen for all w >= 0.**  Every arm changes exactly the leaves in
n's span, the prefix before the span is bit-identical, and the span is entirely inside the
prefix at every w >= 0.  So across the w sweep the *input difference is literally the same
bytes*; only the read distance changes.  That is the control `_swap_sensitivity` lacked.

Two readouts, because they answer different questions
-----------------------------------------------------
1. CAUSAL   ||h[e+w] - h_base[e+w]||, per arm, normalised by the independent-sequence
   spread at the same position.  "How much of the perturbation is still moving the state."
2. PROBE    decode n's rule choice (m classes) and n's feature (v classes) from h[e+w],
   against an EXACT Bayes ceiling from `oracle.prefix_beliefs` at plen = e+w+1
   (`nodes[d]` for the feature, `cliques[d]` summed over the parent value for the rule).
   Retention = (acc - chance) / (bayes - chance), so 0 = fully shed and 1 = fully kept.

The NTP floor is measured, not assumed
--------------------------------------
The tempting premise is "a closed constituent's realisation is exactly redundant for the
future, so the NTP-necessary level is 0".  On THIS RHM that is false and the repo already
recorded it: `aleatoric_fraction`'s `dgp_synonym_swap` reads `frac_TV_exactly_zero` of only
0.53-0.87.  `generate_rules_distinct` draws each feature's m tuples independently across
features, so distinct features can emit the same tuple and a closed span's tokens do NOT
pin its feature -- a same-feature re-realisation therefore moves the posterior over the
ancestors, hence the future.  So every arm carries its exact next-token TV at the read
position, and every readout is reported BOTH pooled and restricted to the `TV == 0` cells,
where displacement is provably pure retained nuisance.

Run:
  # DGP-only self-checks, local, CPU, no Modal, no model (~1 min)
  cd experiments && python3 -m rhm.conditional_revision.synonym_retention.synonym_retention

  # smoke (attached, ~5 min)
  modal run -m rhm.conditional_revision.synonym_retention.synonym_retention::retention \
      --n-seq 400 --n-twin 200 --tag smoke

  # the m4 base (~30 min on an L4; the base loads from cache and is never trained)
  modal run --detach -m rhm.conditional_revision.synonym_retention.synonym_retention::retention \
      --n-seq 6000 --n-twin 2000 --tag ret1

  # the local-loss arm comparison at m2 (10 checkpoints in parallel)
  modal run --detach -m rhm.conditional_revision.synonym_retention.synonym_retention::sweep_arms \
      --tag arms1
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
app = modal.App("rhm-synonym-retention", image=image)

PRIMARY_BLOCK = "post_block6"


def tb_key(v, s, L, m):
    """Volume dir; the `_distinct` suffix is real (see conditional_revision.py)."""
    return f"{setting_key(v, s, L, m)}_distinct"


# ---------------------------------------------------------------------------
# Twin construction -- a strict generalisation of local_loss._twin_leaves
# ---------------------------------------------------------------------------

def _realize(rules, root, rc, s):
    """Rebuild leaves top-down from the root feature and a full set of rule choices.

    Byte-identical to `local_loss.temporal_local_loss._realize` and to the loop inside
    `rhm_latent_loop._generate_with_traces`; `_local_checks` asserts the round trip.
    """
    cur = root
    for ell in range(len(rules)):
        n_nodes = cur.shape[1]
        nxt = np.empty((cur.shape[0], n_nodes * s), dtype=np.int64)
        for j in range(n_nodes):
            nxt[:, j * s:(j + 1) * s] = rules[ell][cur[:, j], rc[ell][:, j]]
        cur = nxt
    return cur


def _twins_at(rules, root, rc, d, j, seed, s, L, v, m):
    """Matched counterfactual twins over the level-`d` constituent at index `j`.

    Returns leaf arrays for leaf / top / str / rand (base and other are built once by
    the caller).  All three DGP arms perturb EXACTLY the leaves in the constituent's
    span, so the prefix before it is bit-identical and the three are size-comparable.

    (d = L-2, j = s^(L-2)-1) reproduces `local_loss._twin_leaves`: its "synonym" is
    `leaf` here and its "structure" is `str`.
    """
    rng = np.random.default_rng(seed)
    S = s ** (L - d)                       # leaves under the constituent
    lo, hi = j * S, (j + 1) * S
    leaf_nodes = list(range(lo // s, hi // s))   # level-(L-1) nodes inside the span

    def bump(a):
        return (a + rng.integers(1, m, size=a.shape)) % m

    rc_leaf = [x.copy() for x in rc]
    for nd in leaf_nodes:
        rc_leaf[L - 1][:, nd] = bump(rc[L - 1][:, nd])
    rc_top = [x.copy() for x in rc]
    rc_top[d][:, j] = bump(rc[d][:, j])
    rc_str = [x.copy() for x in rc_leaf]
    rc_str[d][:, j] = rc_top[d][:, j]      # str = leaf + the SAME top bump

    out = {"leaf": _realize(rules, root, rc_leaf, s),
           "top": _realize(rules, root, rc_top, s),
           "str": _realize(rules, root, rc_str, s)}
    base = _realize(rules, root, rc, s)
    rnd = base.copy()
    rnd[:, lo:hi] = rng.integers(0, v, size=(base.shape[0], S))
    out["rand"] = rnd
    for k, a in out.items():                # nothing outside the span may move
        assert (a[:, :lo] == base[:, :lo]).all(), k
        assert (a[:, hi:] == base[:, hi:]).all(), k
    return out, lo, hi


# ---------------------------------------------------------------------------
# Exact Bayes ceilings from BP (oracle.prefix_beliefs)
# ---------------------------------------------------------------------------

def _bayes_at(rules, seqs, plen, L, want):
    """Exact P(label | x_{<plen}) accuracy for every requested (level d, node j).

    feature: nodes[d][:, j, :]                       -- P(z_(d,j) | prefix)
    rule   : cliques[d][:, j, :, :].sum(axis=-2)     -- P(rule id at (d,j) | prefix),
             marginalising the parent value out of the level-d clique.
    """
    from rhm.conditional_revision import oracle as ORC
    nodes, cliques, _ = ORC.prefix_beliefs(rules, seqs, plen, L)
    out = {}
    for (d, j) in want:
        pf = nodes[d][:, j, :]
        pr = cliques[d][:, j, :, :].sum(axis=-2)
        out[(d, j)] = {"bayes_feat": float(pf.max(-1).mean()),
                       "bayes_rule": float(pr.max(-1).mean())}
    return out                       # scalars only: the posteriors are large and unused


def _leaf_post(rules, seqs, plen):
    """Exact P(x_plen | x_{<plen}); None at plen == T (no next token exists)."""
    from rhm.conditional_revision import oracle as ORC
    if plen >= seqs.shape[1]:
        return None
    return ORC.prefix_beliefs(rules, seqs, plen, 0)[2]


def _next_token_tv(rules, base_seqs, alt_seqs, plen, base_post=None):
    """TV between the exact Bayes next-token posteriors at `plen` -- i.e. how much of
    this perturbation an NTP-optimal model is FORCED to still be carrying at t=plen-1.

    None at plen == T: there is no next token, so NTP constrains that state not at all.
    That is precisely the incumbent readout's blind spot, so it is flagged, not zeroed.
    """
    p0 = base_post if base_post is not None else _leaf_post(rules, base_seqs, plen)
    p1 = _leaf_post(rules, alt_seqs, plen)
    if p0 is None or p1 is None:
        return None
    return 0.5 * np.abs(p0 - p1).sum(-1)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _probe(X, y, n_cls, device, steps=500, lr=3e-2, wd=1e-4, train_frac=0.7, seed=0):
    """Linear logistic readout accuracy + a label-shuffled control on the same split."""
    import torch
    n = X.shape[0]
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    split = int(train_frac * n)
    tr, te = perm[:split], perm[split:]
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu, sd = Xt[tr].mean(0, keepdim=True), Xt[tr].std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    yt = torch.as_tensor(y, dtype=torch.long, device=device)

    def fit(labels):
        W = torch.zeros(Xt.shape[1], n_cls, device=device, requires_grad=True)
        b = torch.zeros(n_cls, device=device, requires_grad=True)
        opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(Xt[tr] @ W + b, labels[tr]).backward()
            opt.step()
        with torch.no_grad():
            return float(((Xt[te] @ W + b).argmax(1) == labels[te]).float().mean())

    ysh = yt[torch.as_tensor(g.permutation(n), device=device)]
    return fit(yt), fit(ysh), int(len(te))


# ---------------------------------------------------------------------------
# The cell grid: which (level, node) constituents, and at what distances
# ---------------------------------------------------------------------------

def _cells(s, L, max_nodes=3):
    """Constituents to perturb.  Early nodes, so there is room for a long w sweep.
    The last level-(L-2) node is always included: it is the incumbent's cell."""
    T = s ** L
    out = []
    for d in range(1, L):
        S = s ** (L - d)
        n_nodes = s ** d
        cand = [j for j in range(n_nodes) if (j + 1) * S - 1 < T - 1]
        picks = cand[:max_nodes] if len(cand) <= max_nodes else \
            sorted({cand[0], cand[len(cand) // 4], cand[len(cand) // 2]})
        for j in picks:
            out.append((d, j))
    out.append((L - 2, s ** (L - 2) - 1))      # the incumbent cell (w = 0 only)
    return sorted(set(out))


W_GRID = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def retention(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    ckpt_path: str = "", label: str = "base_m4",
    base_steps: int = 12000, seed: int = 42,
    n_seq: int = 6000, n_twin: int = 2000, eval_seed: int = 4242,
    probe_blocks: str = "post_block6,post_block2",
    probe_steps: int = 500, batch_size: int = 512,
    out_dir: str = "", tag: str = "",
):
    import torch
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    blocks = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    pblocks = [b.strip() for b in probe_blocks.split(",") if b.strip()]

    print("=" * 84)
    print(f"SYNONYM RETENTION vs DISTANCE   {key}  {n_layer}L/{n_head}H/{n_embd}D  "
          f"label={label}")
    print(f"  n_seq={n_seq}  n_twin={n_twin}  T={T}  v={v}  m={m}  probe blocks={pblocks}")
    print("=" * 84, flush=True)

    # ---------------- the frozen model (never trained here) ----------------
    ckpt = ckpt_path or (f"{DATA_DIR}/{key}/conditional_revision/"
                         f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_"
                         f"seed{seed}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt} -- this cut never trains one.")
    sd = torch.load(ckpt, map_location=device)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(sd)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"loaded frozen model <- {ckpt}", flush=True)

    seqs, lf, rc = _generate_with_traces(rules, n_seq, eval_seed)
    assert (_realize(rules, lf[0], rc, s) == seqs).all(), "realise/generate round trip"
    cells = _cells(s, L)
    print(f"cells (level d, node j): {cells}", flush=True)

    def acts(x_np, want_blocks, positions=None):
        """(n, |want|, |pos|, d) float32 activations; positions=None -> all T."""
        x = torch.from_numpy(x_np.astype(np.int64))
        outs = {b: [] for b in want_blocks}
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                _, _, inter = model(x[i:i + batch_size].to(device),
                                    return_intermediates=True)
                for b in want_blocks:
                    h = inter[b] if positions is None else inter[b][:, positions, :]
                    outs[b].append(h.float().cpu().numpy())
        return {b: np.concatenate(outs[b]) for b in want_blocks}

    # ======================= READOUT 2: probes vs distance ==================
    print("\n--- probe retention: decode the constituent's rule / feature at e+w ---",
          flush=True)
    H = acts(seqs, pblocks)                                    # (n, T, d) per block
    plens = sorted({e + w + 1 for (d, j) in cells for w in W_GRID
                    for e in [(j + 1) * s ** (L - d) - 1] if e + w <= T - 1})
    want = sorted(set(cells))
    ceil = {}
    for pl in plens:
        ceil[pl] = _bayes_at(rules, seqs, pl, L, want)
    print(f"  exact Bayes ceilings computed at {len(plens)} prefix lengths", flush=True)

    probe_rows = []
    for (d, j) in cells:
        S = s ** (L - d)
        e = (j + 1) * S - 1
        y_rule = rc[d][:, j].astype(np.int64)
        y_feat = lf[d][:, j].astype(np.int64)
        for w in W_GRID:
            t = e + w
            if t > T - 1:
                continue
            cz = ceil[t + 1][(d, j)]
            for b in pblocks:
                X = H[b][:, t, :]
                a_r, sh_r, nte = _probe(X, y_rule, m, device, steps=probe_steps, seed=1)
                a_f, sh_f, _ = _probe(X, y_feat, v, device, steps=probe_steps, seed=1)
                probe_rows.append({
                    "level_d": d, "node_j": j, "span": S, "close_pos": e, "w": w,
                    "read_pos": t, "block": b, "n_test": nte,
                    "acc_rule": a_r, "shuf_rule": sh_r, "bayes_rule": cz["bayes_rule"],
                    "chance_rule": 1.0 / m,
                    "acc_feat": a_f, "shuf_feat": sh_f, "bayes_feat": cz["bayes_feat"],
                    "chance_feat": 1.0 / v,
                    "ret_rule": (a_r - 1.0 / m) / max(cz["bayes_rule"] - 1.0 / m, 1e-9),
                    "ret_feat": (a_f - 1.0 / v) / max(cz["bayes_feat"] - 1.0 / v, 1e-9),
                })
            r = probe_rows[-len(pblocks)]
            print(f"  d{d} j{j:<2} e={e:<2} w={w:<2} t={t:<2} [{r['block']}] "
                  f"rule {r['acc_rule']:.3f}/{r['bayes_rule']:.3f} "
                  f"(ret {r['ret_rule']:+.3f}, shuf {r['shuf_rule']:.3f})  "
                  f"feat {r['acc_feat']:.3f}/{r['bayes_feat']:.3f} "
                  f"(ret {r['ret_feat']:+.3f})", flush=True)
    del H

    # ======================= READOUT 1: causal twins ========================
    print("\n--- causal displacement: ||h[e+w] - h_base[e+w]|| per arm ---", flush=True)
    tw_seqs = seqs[:n_twin]
    tw_root, tw_rc = lf[0][:n_twin], [x[:n_twin] for x in rc]
    oth, _, _ = _generate_with_traces(rules, n_twin, eval_seed + 31337)
    base_post = {}                                   # plen -> exact P(x_plen | prefix)

    twin_rows = []
    for (d, j) in cells:
        S = s ** (L - d)
        e = (j + 1) * S - 1
        ws = [w for w in W_GRID if e + w <= T - 1]
        pos = [e + w for w in ws]                    # read only where we look: memory
        tw, lo, hi = _twins_at(rules, tw_root, tw_rc, d, j, eval_seed + 7 * d + j,
                               s, L, v, m)
        Hb = acts(tw_seqs, blocks, pos)
        Ho = acts(oth, blocks, pos)
        Ha = {k: acts(a, blocks, pos) for k, a in tw.items()}
        ham = {k: (tw[k][:, lo:hi] != tw_seqs[:, lo:hi]).sum(1) for k in tw}
        for pi, w in enumerate(ws):
            t = e + w
            if t + 1 not in base_post and t + 1 < T:
                base_post[t + 1] = _leaf_post(rules, tw_seqs, t + 1)
            tv = {k: _next_token_tv(rules, tw_seqs, tw[k], t + 1,
                                    base_post.get(t + 1))
                  for k in ["leaf", "top", "str"]}
            null = {k: (None if tv[k] is None else tv[k] < 1e-12) for k in tv}
            for b in blocks:
                hb = Hb[b][:, pi, :]
                dot = np.linalg.norm(Ho[b][:, pi, :] - hb, axis=-1)
                row = {"level_d": d, "node_j": j, "span": S, "close_pos": e, "w": w,
                       "read_pos": t, "block": b, "n_twin": int(n_twin),
                       "ntp_defined": bool(t + 1 < T),
                       "d_other": float(dot.mean()),
                       "mean_h_norm": float(np.linalg.norm(hb, axis=-1).mean())}
                for k in tw:
                    dk = np.linalg.norm(Ha[k][b][:, pi, :] - hb, axis=-1)
                    row[f"d_{k}"] = float(dk.mean())
                    row[f"d_{k}_se"] = float(dk.std() / np.sqrt(len(dk)))
                    row[f"rel_{k}"] = float(dk.mean() / (dot.mean() + 1e-12))
                    row[f"ham_{k}"] = float(ham[k].mean())
                    if k in null and null[k] is not None:
                        row[f"tv_{k}"] = float(tv[k].mean())
                        row[f"frac_tvzero_{k}"] = float(null[k].mean())
                        if null[k].any():
                            row[f"rel_{k}_tv0"] = float(dk[null[k]].mean()
                                                        / (dot[null[k]].mean() + 1e-12))
                row["syn_over_str"] = row["d_leaf"] / (row["d_str"] + 1e-12)
                twin_rows.append(row)
            r = [z for z in twin_rows if z["block"] == PRIMARY_BLOCK][-1]
            print(f"  d{d} j{j:<2} e={e:<2} w={w:<2} t={t:<2} | rel leaf {r['rel_leaf']:.4f} "
                  f"top {r['rel_top']:.4f} str {r['rel_str']:.4f} rand {r['rel_rand']:.4f} "
                  f"| syn/str {r['syn_over_str']:.3f} | ham {r['ham_leaf']:.2f}/"
                  f"{r['ham_str']:.2f} | TVleaf {r.get('tv_leaf', float('nan')):.4f} "
                  f"(0 in {r.get('frac_tvzero_leaf', float('nan')):.2f})", flush=True)
        del Ha, Hb, Ho

    # ======================= summary tables =================================
    print(f"\n{'=' * 84}\nRETENTION vs w   ({PRIMARY_BLOCK}, pooled over nodes at each level)")
    print(f"{'d':>2} {'w':>3} {'n':>2} {'ruleAcc':>8} {'ruleBay':>8} {'retRule':>8} "
          f"{'featAcc':>8} {'featBay':>8} {'retFeat':>8} {'relLeaf':>8} {'relStr':>8}")
    summary = {}
    for d in sorted({c[0] for c in cells}):
        for w in W_GRID:
            pr = [r for r in probe_rows
                  if r["level_d"] == d and r["w"] == w and r["block"] == PRIMARY_BLOCK]
            tr = [r for r in twin_rows
                  if r["level_d"] == d and r["w"] == w and r["block"] == PRIMARY_BLOCK]
            if not pr:
                continue

            def g(rows, k):
                vals = [z[k] for z in rows if k in z]
                return float(np.mean(vals)) if vals else float("nan")

            row = {"level_d": d, "w": w, "n_nodes": len(pr),
                   "acc_rule": g(pr, "acc_rule"), "bayes_rule": g(pr, "bayes_rule"),
                   "ret_rule": g(pr, "ret_rule"), "shuf_rule": g(pr, "shuf_rule"),
                   "acc_feat": g(pr, "acc_feat"), "bayes_feat": g(pr, "bayes_feat"),
                   "ret_feat": g(pr, "ret_feat"), "shuf_feat": g(pr, "shuf_feat")}
            if tr:
                for k in ["leaf", "top", "str", "rand"]:
                    row[f"rel_{k}"] = g(tr, f"rel_{k}")
                row["syn_over_str"] = g(tr, "syn_over_str")
                row["tv_leaf"] = g(tr, "tv_leaf")
                row["tv_str"] = g(tr, "tv_str")
                row["frac_tvzero_leaf"] = g(tr, "frac_tvzero_leaf")
                row["rel_leaf_tv0"] = g(tr, "rel_leaf_tv0")
                row["rel_str_tv0"] = g(tr, "rel_str_tv0")
            summary[f"d{d}_w{w}"] = row
            print(f"{d:>2} {w:>3} {len(pr):>2} {row['acc_rule']:>8.3f} "
                  f"{row['bayes_rule']:>8.3f} {row['ret_rule']:>8.3f} "
                  f"{row['acc_feat']:>8.3f} {row['bayes_feat']:>8.3f} "
                  f"{row['ret_feat']:>8.3f} "
                  f"{row.get('rel_leaf', float('nan')):>8.4f} "
                  f"{row.get('rel_str', float('nan')):>8.4f}", flush=True)

    inc = [r for r in twin_rows
           if r["level_d"] == L - 2 and r["node_j"] == s ** (L - 2) - 1
           and r["w"] == 0 and r["block"] == PRIMARY_BLOCK]
    if inc:
        print(f"\nINCUMBENT CELL (d={L-2}, j={s**(L-2)-1}, w=0) -- "
              f"local_loss._swap_sensitivity's geometry: syn/str = "
              f"{inc[0]['syn_over_str']:.3f}  relLeaf {inc[0]['rel_leaf']:.4f} "
              f"relStr {inc[0]['rel_str']:.4f}", flush=True)

    res = {"config": {"v": v, "s": s, "L": L, "m": m, "rule_seed": rule_seed,
                      "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
                      "ckpt": ckpt, "label": label, "n_seq": n_seq, "n_twin": n_twin,
                      "eval_seed": eval_seed, "seed": seed, "tag": tag,
                      "probe_blocks": pblocks, "w_grid": W_GRID,
                      "cells": [list(c) for c in cells]},
           "probe_rows": probe_rows, "twin_rows": twin_rows, "summary": summary}

    odir = out_dir or f"{DATA_DIR}/{key}/conditional_revision"
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, f"synonym_retention_{tag or 'run'}_{label}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {path}", flush=True)
    return {"label": label, "path": path, "summary": summary}


# ---------------------------------------------------------------------------
# The local-loss arm comparison (m2)
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, timeout=14400, memory=8192)
def sweep_arms(lams: str = "0.1,1.0,3.0", step: int = 80000,
               v: int = 16, s: int = 2, depth: int = 6, m: int = 2,
               n_seq: int = 6000, n_twin: int = 2000,
               extra: str = "depth_lam0p1@150000,depth_lam1p0@300000,depth_lam3p0@300000",
               tag: str = "arms1"):
    """Retention on every local_loss checkpoint.  `_label_dir` is imported from
    `local_loss.temporal_local_loss` so the arm->directory map cannot drift."""
    from rhm.conditional_revision.local_loss.temporal_local_loss import _label_dir

    key = tb_key(v, s, depth, m)
    labels = ["lam0_base"] + \
             [f"{a}_lam{str(l).replace('.', 'p')}" for a in ("depth", "temporal")
              for l in [x.strip() for x in lams.split(",") if x.strip()]]
    jobs = [(lb, os.path.join(_label_dir(key, lb), f"ckpt_step{step}.pt")) for lb in labels]
    for e in [x.strip() for x in extra.split(",") if x.strip()]:
        lb, st = e.split("@")
        jobs.append((e, os.path.join(_label_dir(key, lb), f"ckpt_step{int(st)}.pt")))

    out_dir = f"{DATA_DIR}/rhm_synonym_retention/parts_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    pending = []
    for lb, p in jobs:
        if not os.path.exists(p):
            print(f"  MISSING {lb}: {p}", flush=True)
            continue
        print(f"  spawn {lb} <- {p}", flush=True)
        pending.append((lb, retention.spawn(
            v=v, s=s, depth=depth, m=m, ckpt_path=p, label=lb,
            n_seq=n_seq, n_twin=n_twin, out_dir=out_dir, tag=tag)))
    rows = []
    for lb, h in pending:
        try:
            rows.append(h.get())
        except Exception as ex:                                  # noqa: BLE001
            print(f"  FAILED {lb}: {ex}", flush=True)
    print(f"\n{len(rows)}/{len(pending)} arms done -> {out_dir}", flush=True)
    return [r["label"] for r in rows]


# ---------------------------------------------------------------------------
# DGP-only self-checks (local, CPU, no Modal, no model)
# ---------------------------------------------------------------------------

def _local_checks():
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.conditional_revision import oracle as ORC

    v, s, L, m, n = 16, 2, 6, 4, 300
    rules = generate_rules_distinct(v, s, L, m, seed=0)
    seqs, lf, rc = _generate_with_traces(rules, n, 11)
    assert (_realize(rules, lf[0], rc, s) == seqs).all()
    print("[ok] _realize round-trips _generate_with_traces")

    ids = ORC._rule_ids(rules, 4)
    assert (ids == np.arange(m)[None, :]).all(), \
        "distinct rules must give ids[a,r]==r, else the clique rule axis is not the label"
    print("[ok] clique rule axis == the raw rule-choice label")

    # the clique marginal, given the FULL sequence, must put mass on the true rule
    nodes, cliques, _ = ORC.prefix_beliefs(rules, seqs, s ** L, L)
    for d in (2, 4, 5):
        pr = cliques[d][:, 1, :, :].sum(axis=-2)
        pf = nodes[d][:, 1, :]
        tr = np.take_along_axis(pr, rc[d][:, 1:2], 1).mean()
        tf = np.take_along_axis(pf, lf[d][:, 1:2], 1).mean()
        print(f"[ok] d{d} full-sequence posterior mass on the TRUE "
              f"rule {tr:.4f} / feature {tf:.4f}  (argmax acc "
              f"{pr.max(-1).mean():.4f} / {pf.max(-1).mean():.4f})")
        assert tr > 0.5 and tf > 0.5

    print("\ntwins: prefix identity, span confinement, Hamming, and the exact NTP floor")
    print(f"{'cell':>10} {'span':>5} {'e':>3} {'w':>3} {'hamLeaf':>8} {'hamTop':>7} "
          f"{'hamStr':>7} {'TVleaf':>8} {'TV0leaf':>8} {'TVstr':>8} {'TV0str':>8}")
    for (d, j) in [(4, 0), (4, 1), (3, 0), (2, 0)]:
        tw, lo, hi = _twins_at(rules, lf[0], rc, d, j, 5, s, L, v, m)
        S = s ** (L - d)
        e = (j + 1) * S - 1
        for w in (0, 4, 16):
            t = e + w
            if t > s ** L - 1:
                continue
            r = {}
            for k in ("leaf", "str"):
                tv = _next_token_tv(rules, seqs, tw[k], t + 1)
                r[k] = (tv.mean(), (tv < 1e-12).mean())
            print(f"{'d%d j%d' % (d, j):>10} {S:>5} {e:>3} {w:>3} "
                  f"{(tw['leaf'][:, lo:hi] != seqs[:, lo:hi]).sum(1).mean():>8.2f} "
                  f"{(tw['top'][:, lo:hi] != seqs[:, lo:hi]).sum(1).mean():>7.2f} "
                  f"{(tw['str'][:, lo:hi] != seqs[:, lo:hi]).sum(1).mean():>7.2f} "
                  f"{r['leaf'][0]:>8.4f} {r['leaf'][1]:>8.3f} "
                  f"{r['str'][0]:>8.4f} {r['str'][1]:>8.3f}")

    print(f"\ncells for s={s} L={L}: {_cells(s, L)}")
    print("\nall DGP checks passed")


if __name__ == "__main__":
    _local_checks()
