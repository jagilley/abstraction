"""The tracking analysis: an instrument check on Gates A and B, imported from
Petersen, van Mier, Fiez & Raichle 1998 (PNAS 95:853-860).

Their audit of their own 1989 PET word-processing design is our situation exactly.
They read a chain of subtractions in which each level's control state is the
previous level's stimulated state -- one variable at a time, the same discipline
as `p+ - p-` and as `mjc/arity_torque`'s "identical data, only the input differs".
The failure mode they name (p.854) is REPLACEMENT, not addition: "some of the
processes used in word reading are replaced when verb generation is performed."
A single difference `B - A` cannot distinguish

    B = A + new stuff        (insertion)
    B = A - old stuff + new stuff   (replacement)

and the replacement case is invisible in the contrast you care about. It shows up
only as a sign flip in the contrast you did not think you needed. Their fix (their
Fig. 1) is to refuse to read any one contrast alone: instrument every region in
every STATE, keep both signed contrasts, and classify each region as common /
insertion / replacement.

What that maps to here
----------------------
Gate A's and Gate B's primary quantity is

    M = sum_d KL( q_{t+1}^{a_d} || q_t^{a_d} )        gates_ab.py:601-615

a difference of two probe-decoded posteriors read AT THE SAME NODE INDEX
`a_d = a_d(t+1)`, x_{t+1}'s level-d ancestor. SPEC's Gate C.2 guards the capacity
residue (`eps2 - eps1`), but capacity invariance is blind to replacement: it would
pass cleanly while the `t+1` representation had quietly dropped something the `t`
representation held. And the README's own headline probe finding says the model
does exactly that -- "the per-position representation is close to a
next-token-sufficient statistic: it holds the ancestor chain of the token it is
predicting and does not summarise resolved structure forward" (past constituents
0.20 against a Bayes ceiling of ~1.0).

So: at a constituent boundary the chain REPLACES a node. Concretely, for level d
with span s^(L-d), position t is a boundary iff (t+1) % s^(L-d) == 0, and then

    node_new = a_d(t+1)   is INCOMING at t (its span starts at t+1) -> current at t+1
    node_old = a_d(t)     is current at t                           -> PAST at t+1

`M` reads node_new at both states. It never looks at node_old -- which is one place
a replacement could hide, and the instrument therefore reads node_old too.

Two things keep this from being the whole story, and both are design facts rather
than results. (i) Boundary-ness is a DETERMINISTIC function of position, so Gate
B's primary column, which already matches on position, cannot be fooled by the
categorical part of a class change; only a class-change effect whose MAGNITUDE
covaries with family inside a fixed position could reach it. (ii) A readability
change is not confined to boundaries -- the probe's fidelity can drift across the
step at any cell. So the boundary/continuation split is the bounding special case,
and the operative, general criterion is the per-cell readability excess defined
below, which is evaluated at every (position, level) cell whether or not the chain
swaps a node there.

The two signed contrasts, per (level, position):

    contrast 1 (the one we care about)   the belief move at node_new, t -> t+1
    contrast 2 (the one we didn't)       the READABILITY move, probe-vs-Bayes at
                                         each state, at node_new AND at node_old

Their table, transposed:

    common to both    node current at t and at t+1; probe fidelity flat; M ~ B
    insertion         node_new incoming -> current, and the Bayes ceiling rises
                      with it, so the readability change is warranted
    replacement       readability changes while the truth does not -- either
                      node_new becomes readable without Bayes moving, or node_old
                      becomes UNreadable while Bayes says it only got more
                      determined (the sign flip)

What is computed
----------------
1. The unsubtracted, signed state-wise components M currently collapses: probe
   posterior, accuracy, entropy, and exact-Bayes counterparts at `t` and at `t+1`
   SEPARATELY, for node_new and node_old, split by class and by Gate B family.

2. An exact path decomposition of the per-term KL. With p = exact Bayes and
   q = probe, at the same node, subscript 0 = read at t, 1 = read at t+1:

       B     = KL(p1 || p0)      the oracle term  (== oracle.B_chain's level term)
       hybT  = KL(p1 || q0)      probe substituted into the t state only
       M     = KL(q1 || q0)      what Gate A/B actually use

       d_den = hybT - B          the t-state readout's contribution
       d_num = M    - hybT       the t+1-state readout's contribution
       M - B = d_den + d_num     exactly; no residual, no smoothing floor

   A replacement shows up as ASYMMETRY: |d_den| >> |d_num| (or a sign flip between
   them) means M's magnitude is a readout property of ONE state, not a belief
   difference taken across both. Every KL here is finite as written -- a softmax
   has full support and supp(p1) subset supp(p0) -- which is why the decomposition
   runs this way round rather than through KL(q1 || p0); that hybrid is unbounded
   wherever the exact posterior has a hard zero, so its magnitude would be set by
   the smoothing floor rather than by the data.

3. Two ONE-SIDED nulls, which are the model-side version of the same discipline
   and inject no oracle:

       M_swapT   = KL(q1 || qtilde0)   t-state content destroyed (permuted across
                                       sequences WITHIN position, so positional
                                       structure survives)
       M_swapT1  = KL(qtilde1 || q0)   t+1-state content destroyed

   If Gate B's AUC survives `M_swapT`, the t state contributes nothing and `M` is
   a single-state readout wearing a difference's clothes. Ditto `M_swapT1` for the
   t+1 state. If both collapse to ~0.5, the subtraction was clean.
   Two pure single-state scores are included for the same reason: `negH_t1`
   (= log v - H(q1), summed over d, no t state anywhere) and `negH_t`.

4. Gate A's partial R^2 and Gate B's matched AUC re-run with replacement-shaped
   terms handled -- excluded (`M_cont`, `M_replexcl`) and as their own column
   (`M_bdry`). Plus Gate B's primary AUC split by whether the position contains a
   class change at all, which is the most legible version of the question: is
   0.690 (d2) / 0.614 (d3) concentrated in the boundary positions?

Reproducibility
---------------
This file does not change what Gate A/B compute; it re-derives them and audits
them. The base checkpoint, the FM checkpoints, the aligned-sequence RNG, the probe
seed AND the global torch RNG consumption order are all replicated from
`gates_ab.py`, so the `chain` probe here is bit-identical to the one that produced
the published numbers. `reproduction` in the output JSON is the check: Gate A's
0.150 (d1) / 0.102 (d2) and Gate B's 0.690 (d2) / 0.614 (d3) must come back.

Run:
  # smoke (~6 min, attached)
  modal run -m rhm.conditional_revision.tracking.tracking::tracking \
      --n-probe-train 400 --n-calib 200 --n-test 200 --probe-steps 400 --tag smoke
  # the real thing (~40 min on an L4; base + FMs are loaded from cache)
  modal run --detach -m rhm.conditional_revision.tracking.tracking::tracking --tag track
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-conditional-revision-tracking", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


# ---------------------------------------------------------------------------
# statistics -- duplicated VERBATIM from gates_ab.py rather than imported, so
# that importing this module does not construct gates_ab's Modal app/image.
# Any edit here that changes a number is a bug: the reproduction block asserts
# these reproduce gates_ab's published Gate A / Gate B values.
# ---------------------------------------------------------------------------

def _r2(y, x):
    import numpy as np
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc * xc).sum()
    if denom < 1e-30:
        return 0.0
    beta = (xc * yc).sum() / denom
    ss_res = ((yc - beta * xc) ** 2).sum()
    ss_tot = (yc * yc).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-30))


def _resid(y, x):
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc * xc).sum()
    beta = 0.0 if denom < 1e-30 else (xc * yc).sum() / denom
    return yc - beta * xc


def _partial_r2(y, x, z):
    return _r2(_resid(y, z), _resid(x, z))


def _ranks(a):
    from scipy.stats import rankdata
    return rankdata(a)


def _auc(score, pos_mask):
    n1 = int(pos_mask.sum())
    n0 = int((~pos_mask).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(score)
    return float((r[pos_mask].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _stratified_auc(score, pos_mask, strata):
    import numpy as np
    num = den = 0.0
    for k in np.unique(strata):
        sel = strata == k
        a = _auc(score[sel], pos_mask[sel])
        n1 = int(pos_mask[sel].sum())
        n0 = int(sel.sum() - n1)
        if n1 and n0:
            num += a * n1 * n0
            den += n1 * n0
    return float(num / den) if den else float("nan")


def _strata(vals, n_bins, atom_frac=0.002):
    import numpy as np
    vals = np.asarray(vals)
    uniq, counts = np.unique(vals, return_counts=True)
    atoms = uniq[counts >= max(atom_frac * vals.size, 2)]
    out = np.full(vals.size, -1, dtype=np.int64)
    for i, a in enumerate(atoms):
        out[vals == a] = i
    rest = out < 0
    if rest.any():
        edges = np.unique(np.quantile(vals[rest], np.linspace(0, 1, n_bins + 1)))
        out[rest] = len(atoms) + (np.digitize(vals[rest], edges[1:-1])
                                  if edges.size > 2 else 0)
    return out


# published reference lines (gates_ab_gateAB3_seed42.json, post_block6)
PUBLISHED = {
    "gate_A_partial_r2_M_from_B_given_nll": {
        "D0": 0.0084, "D1": 0.0133, "D2": 0.0218, "D3": 0.0394,
        "D4": 0.1018, "D5": 0.1500},
    "gate_B_auc_pos_x_bp_matched_M": {
        "D0": 0.4963, "D1": 0.5077, "D2": 0.5261, "D3": 0.6138, "D4": 0.6898},
}


# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=49152)
def tracking(
    # DGP / model -- must match the cached base checkpoint (gates_ab defaults)
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    shallow_block: str = "post_block0", deep_block: str = "post_block6",
    belief_block: str = "post_block6",
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0, fm_steps: int = 12000, fwd_lr: float = 1e-3,
    base_steps: int = 12000, batch_size: int = 64, weight_decay: float = 0.01,
    data_seed: int = 7,
    n_probe_train: int = 4000, n_calib: int = 500, n_test: int = 2500,
    probe_steps: int = 8000, probe_lr: float = 1e-3, probe_batch: int = 512,
    probe_hidden: int = 1024,
    eval_seed: int = 999, oracle_chunk: int = 256, n_nll_bins: int = 40,
    n_rand_excl: int = 10, seed: int = 42, tag: str = "", base_ckpt: str = "",
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.conditional_revision import oracle as ORC

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    n_nodes = (s ** L - 1) // (s - 1)
    node_off = {d: (s ** d - 1) // (s - 1) for d in range(L)}

    # node index of x_{t+1}'s level-d ancestor (what M reads) and of x_t's (what
    # the chain leaves behind), per signal position t. Identical to gates_ab.
    anc_idx = np.stack([np.array([node_off[d] + ((t + 1) // (s ** (L - d)))
                                  for t in range(T - 1)]) for d in range(L)])
    old_idx = np.stack([np.array([node_off[d] + (t // (s ** (L - d)))
                                  for t in range(T - 1)]) for d in range(L)])
    # class change: at (t, d) the chain's level-d node is REPLACED iff t+1 starts a
    # new level-d constituent. Note this is a deterministic function of position,
    # which is itself informative -- Gate B's primary column already matches on
    # position, so the class change cannot manufacture between-family separation
    # unless its MAGNITUDE covaries with family inside a position.
    is_bdry = np.stack([np.array([(t + 1) % (s ** (L - d)) == 0
                                  for t in range(T - 1)]) for d in range(L)])
    assert ((anc_idx != old_idx) == is_bdry).all(), "boundary predicate disagrees"

    anc_mask = np.zeros((T, n_nodes), dtype=bool)
    for t in range(T):
        for u in (t, min(t + 1, T - 1)):
            for d in range(L):
                anc_mask[t, node_off[d] + (u // (s ** (L - d)))] = True
    anc_mask_t = torch.from_numpy(anc_mask)

    print(f"{'=' * 78}\nTRACKING ANALYSIS -- instrument check on Gates A & B   {key}")
    print(f"  boundary (class-change) positions per level, out of {T - 1}: "
          f"{ {f'd{L - d}': int(is_bdry[d].sum()) for d in range(L)} }")
    print(f"{'=' * 78}", flush=True)

    # ---------------- base model + FMs: same construction ORDER as gates_ab, so
    # the global torch RNG is in the identical state when the probe is built ----
    ckpt = base_ckpt or (f"{DATA_DIR}/{key}/conditional_revision/"
                         f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_"
                         f"seed{seed}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no base checkpoint at {ckpt}; run gate0 first")
    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded frozen base <- {ckpt}", flush=True)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    make_fm()                                   # gate0's discarded co-trained arm
    fm_temporal, fm_temporal_direct, fm_depth_frozen = make_fm(), make_fm(), make_fm()
    fm_ckpt = (f"{DATA_DIR}/{key}/conditional_revision/fms_frozen_"
               f"steps{fm_steps}_seed{seed}.pt")
    if not os.path.exists(fm_ckpt):
        raise FileNotFoundError(
            f"no cached FMs at {fm_ckpt}. This audit never trains an FM -- run "
            f"gates_ab first so `r_temporal` is the same object it reported.")
    fsd = torch.load(fm_ckpt, map_location=device)
    for k_, f_ in (("temporal", fm_temporal), ("temporal_direct", fm_temporal_direct),
                   ("depth_frozen", fm_depth_frozen)):
        f_.load_state_dict(fsd[k_]); f_.eval()
    print(f"loaded cached FMs <- {fm_ckpt}", flush=True)

    # ---------------- aligned sequences (identical RNG to gates_ab) ----------
    n_align = n_probe_train + n_calib + n_test
    print(f"\nGenerating {n_align} aligned sequences...", flush=True)
    al_seqs, al_lf, _ = _generate_with_traces(rules, n_align, eval_seed)
    al_x = torch.from_numpy(al_seqs.astype(np.int64))
    node_y = np.concatenate([al_lf[d] for d in range(L)], axis=1)
    node_y_t = torch.from_numpy(node_y.astype(np.int64))
    sl_tr = slice(0, n_probe_train)
    sl_ca = slice(n_probe_train, n_probe_train + n_calib)
    sl_te = slice(n_probe_train + n_calib, n_align)

    @torch.no_grad()
    def collect(idx_slice):
        acts, nlls, rel_t, dn = [], [], [], []
        xs = al_x[idx_slice]
        for i in range(0, xs.shape[0], 256):
            xa = xs[i:i + 256].to(device)
            ya = torch.cat([xa[:, 1:], xa[:, :1]], dim=1)
            logits, _, inter = model(xa, ya, return_intermediates=True)
            nll = F.cross_entropy(logits.reshape(-1, v), ya.reshape(-1),
                                  reduction="none").reshape(xa.shape[0], T)
            h_deep = inter[deep_block]
            delta = h_deep[:, 1:, :] - h_deep[:, :-1, :]
            pred = (fm_temporal(h_deep) - h_deep)[:, :-1, :]
            r = (delta - pred).norm(dim=-1)
            acts.append(inter[belief_block].float().cpu())
            nlls.append(nll[:, :T - 1].cpu())
            rel_t.append((r / (delta.norm(dim=-1) + 1e-6)).cpu())
            dn.append(delta.norm(dim=-1).cpu())
        return (torch.cat(acts),
                {"nll": torch.cat(nlls).numpy(),
                 "r_temporal": torch.cat(rel_t).numpy(),
                 "delta_norm": torch.cat(dn).numpy()})

    acts_tr, _ = collect(sl_tr)
    acts_ca, _ = collect(sl_ca)
    acts_te, sig_te = collect(sl_te)

    # ---------------- the chain probe (gates_ab's FIRST train_probe call) ----
    class Probe(nn.Module):
        def __init__(self, d_in, hidden, n_out):
            super().__init__()
            self.net = (nn.Linear(d_in, n_out) if hidden == 0 else
                        nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                      nn.Linear(hidden, n_out)))

        def forward(self, x):
            return self.net(x)

    Xtr = acts_tr.reshape(-1, n_embd)
    Ytr = node_y_t[sl_tr].unsqueeze(1).expand(-1, T, -1).reshape(-1, n_nodes)
    probe = Probe(n_embd, probe_hidden, n_nodes * v).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 7)
    mu, sdv = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    print(f"\n--- chain probe: {probe_steps} steps ---", flush=True)
    for st in range(probe_steps):
        ix = torch.randint(0, Xtr.shape[0], (probe_batch,), generator=g)
        xb = ((Xtr[ix] - mu) / sdv).to(device)
        yb = Ytr[ix].to(device)
        out = probe(xb).view(-1, n_nodes, v)
        ce = F.cross_entropy(out.reshape(-1, v), yb.reshape(-1),
                             reduction="none").view(-1, n_nodes)
        mk = anc_mask_t[(ix % T)].to(device)
        loss = (ce * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if st % max(probe_steps // 4, 1) == 0:
            print(f"    probe {st:5d}  ce {loss.item():.4f}", flush=True)
    probe.eval()

    @torch.no_grad()
    def probe_logits(acts_block, chunk=64):
        outs = []
        for i in range(0, acts_block.shape[0], chunk):
            xb = ((acts_block[i:i + chunk] - mu) / sdv).to(device)
            outs.append(probe(xb).view(xb.shape[0], T, n_nodes, v).cpu())
        return torch.cat(outs)

    lg_ca = probe_logits(acts_ca)
    logT = torch.zeros(1, requires_grad=True)
    yb_ca = node_y_t[sl_ca].unsqueeze(1).expand(-1, T, -1).reshape(-1)
    lg = lg_ca.reshape(-1, v)
    keepm = anc_mask_t.unsqueeze(0).expand(lg_ca.shape[0], -1, -1).reshape(-1)
    lg, yb_ca = lg[keepm], yb_ca[keepm]
    if lg.shape[0] > 500_000:
        gg = torch.Generator().manual_seed(seed + 13)
        sub = torch.randperm(lg.shape[0], generator=gg)[:500_000]
        lg, yb_ca = lg[sub], yb_ca[sub]
    optT = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def closure():
        optT.zero_grad()
        l = F.cross_entropy(lg / logT.exp(), yb_ca)
        l.backward()
        return l
    optT.step(closure)
    temp = float(logT.exp().item())
    print(f"  probe temperature {temp:.3f}", flush=True)

    q = torch.softmax(probe_logits(acts_te) / temp, dim=-1).numpy()   # (n,T,J,v)
    del acts_tr, acts_ca, acts_te, Xtr, Ytr, lg_ca

    # ---------------- oracle, with the per-state marginals kept ---------------
    print(f"\n--- oracle: exact BP, keeping the unsubtracted per-state posteriors "
          f"---", flush=True)
    te_lf = [al_lf[d][sl_te] for d in range(L + 1)]
    orc = ORC.revision_and_entropy(rules, al_seqs[sl_te], te_lf, chunk=oracle_chunk,
                                   verbose=True, anc_mask=anc_mask,
                                   return_chain_marginals=True)
    chk = ORC.self_check(orc)
    if not chk["passed"]:
        raise RuntimeError("instrument self-check failed")
    print(f"  instrument self-check passed", flush=True)
    Ds = orc["Ds"]
    cm = orc["chain_marg"]

    # ---------------- gather the four state-wise reads -----------------------
    tA, tB = np.arange(T - 1), np.arange(1, T)

    def gather_q(idx, ts):
        return np.stack([q[:, ts, idx[d], :] for d in range(L)],
                        axis=2).astype(np.float64)          # (n, T-1, L, v)

    q_new_0, q_new_1 = gather_q(anc_idx, tA), gather_q(anc_idx, tB)
    q_old_0, q_old_1 = gather_q(old_idx, tA), gather_q(old_idx, tB)
    del q
    y_new = np.stack([node_y[sl_te][:, anc_idx[d]] for d in range(L)], axis=2)
    y_old = np.stack([node_y[sl_te][:, old_idx[d]] for d in range(L)], axis=2)
    p_new_0, p_new_1 = cm["new_prev"], cm["new_cur"]
    p_old_0, p_old_1 = cm["old_prev"], cm["old_cur"]
    del orc["chain_marg"]

    def kl_exact(a, b):
        r = np.log(np.clip(a, 1e-300, None)) - np.log(np.clip(b, 1e-300, None))
        return np.where(a > 0, a * r, 0.0).sum(-1)

    def ent(p):
        return -np.where(p > 0, p * np.log(np.clip(p, 1e-300, None)), 0.0).sum(-1)

    # per-term (n, T-1, L). Every KL below is EXACT and finite: a softmax q has
    # full support, and supp(p1) subset supp(p0) because conditioning on more
    # evidence only removes support. (An earlier version also carried
    # KL(q1||p0) as a second hybrid; that one is unbounded wherever the exact
    # posterior has a hard zero, so its magnitude is set by whatever floor you
    # smooth with rather than by the data. The two-term path decomposition below
    # is exact and needs no floor.)
    m_d = kl_exact(q_new_1, q_new_0)                     # == gates_ab's per_d, M
    b_d = kl_exact(p_new_1, p_new_0)                     # == oracle B_chain term
    hybT = kl_exact(p_new_1, q_new_0)                    # probe in the t state only
    #   M - B  =  d_den  +  d_num     exactly, no residual.
    # d_den: swap the truth p0 for the probe q0 in the DENOMINATOR -- the t state's
    #        readout contribution, holding the t+1 state at the truth.
    # d_num: swap p1 for q1 in the NUMERATOR -- the t+1 state's contribution.
    # A replacement is an ASYMMETRY between these: M's magnitude then belongs to
    # one state's readout, not to a difference taken across both.
    d_den = hybT - b_d
    d_num = m_d - hybT
    fid_0, fid_1 = kl_exact(p_new_0, q_new_0), kl_exact(p_new_1, q_new_1)

    # state-wise scalars
    acc_q_new_0 = (q_new_0.argmax(-1) == y_new).astype(np.float64)
    acc_q_new_1 = (q_new_1.argmax(-1) == y_new).astype(np.float64)
    acc_p_new_0 = (p_new_0.argmax(-1) == y_new).astype(np.float64)
    acc_p_new_1 = (p_new_1.argmax(-1) == y_new).astype(np.float64)
    acc_q_old_0 = (q_old_0.argmax(-1) == y_old).astype(np.float64)
    acc_q_old_1 = (q_old_1.argmax(-1) == y_old).astype(np.float64)
    acc_p_old_0 = (p_old_0.argmax(-1) == y_old).astype(np.float64)
    acc_p_old_1 = (p_old_1.argmax(-1) == y_old).astype(np.float64)
    H_q0, H_q1 = ent(q_new_0), ent(q_new_1)
    H_p0, H_p1 = ent(p_new_0), ent(p_new_1)
    del q_old_0, q_old_1, p_old_0, p_old_1

    # consistency: the per-term pieces must rebuild the published objects
    max_b_err = max(float(np.abs(b_d[:, :, :D + 1].sum(-1)
                                 - orc["B_chain"][D]).max()) for D in Ds)
    print(f"  per-term B rebuild vs oracle B_chain: max abs err {max_b_err:.2e}",
          flush=True)
    assert max_b_err < 1e-9

    # ---------------- one-sided nulls (no oracle) ----------------------------
    rng = np.random.default_rng(seed + 4242)
    perm = np.stack([rng.permutation(q_new_0.shape[0]) for _ in range(T - 1)], 1)
    tq0 = np.take_along_axis(q_new_0, perm[:, :, None, None], axis=0)
    tq1 = np.take_along_axis(q_new_1, perm[:, :, None, None], axis=0)
    m_swapT = kl_exact(q_new_1, tq0)      # t state destroyed
    m_swapT1 = kl_exact(tq1, q_new_0)     # t+1 state destroyed
    del tq0, tq1

    # If the t+1 read is near-degenerate (a point mass at yhat), then
    #   KL(q1||q0) = -log q0(yhat) - H(q1)  ->  -log q0(yhat),
    # i.e. M collapses to a ONE-STATE readout of q0, evaluated at a point the t+1
    # state only has to name. This is the sharpest form of the tracking analysis's
    # warning, so it gets its own column: if `M_pointmass` reproduces M's AUC, the
    # "difference" is not doing the work.
    yhat1 = q_new_1.argmax(-1)
    m_pointmass = -np.log(np.clip(np.take_along_axis(q_new_0, yhat1[..., None],
                                                     axis=-1)[..., 0], 1e-30, None))
    # the same readout aimed at the TRUE latent rather than the probe's guess, and
    # its exact-Bayes counterpart. Both use ground truth -> diagnostics, not
    # competitors.
    nlq0_true = -np.log(np.clip(np.take_along_axis(
        q_new_0, y_new[..., None], axis=-1)[..., 0], 1e-30, None))
    nlp0_true = -np.log(np.clip(np.take_along_axis(
        p_new_0, y_new[..., None], axis=-1)[..., 0], 1e-30, None))
    del q_new_0, q_new_1, p_new_0, p_new_1

    LOGV = float(np.log(v))
    negH_1, negH_0 = LOGV - H_q1, LOGV - H_q0
    dH = H_q0 - H_q1

    # ---------------- cell-level classification (Petersen's Fig. 1) ----------
    # Per (position, level) cell, averaged over sequences -- deliberately NOT
    # conditioned on the Gate B family, so any exclusion built from it is
    # non-circular with respect to the label it will later be scored against.
    cell = {k: arr.mean(0) for k, arr in (           # each (T-1, L)
        ("m", m_d), ("b", b_d), ("dden", d_den), ("dnum", d_num),
        ("fid0", fid_0), ("fid1", fid_1),
        ("accq0", acc_q_new_0), ("accq1", acc_q_new_1),
        ("accp0", acc_p_new_0), ("accp1", acc_p_new_1),
        ("accq_old0", acc_q_old_0), ("accq_old1", acc_q_old_1),
        ("accp_old0", acc_p_old_0), ("accp_old1", acc_p_old_1))}
    # Petersen's two signed contrasts, per cell:
    #   contrast 1  the belief move the truth licenses      -> cell["b"]
    #   contrast 2  the READABILITY move IN EXCESS of it    -> excess below
    # excess = (how much better the probe reads the node at t+1 than at t)
    #        - (how much better BAYES reads it at t+1 than at t)
    # A term whose KL is a real difference has excess ~ 0: the readout tracks the
    # ceiling and the KL is measuring the belief. A replacement-shaped term is one
    # where readability moved for reasons the truth does not warrant.
    excess = ((cell["accq1"] - cell["accq0"]) - (cell["accp1"] - cell["accp0"]))
    excess_old = ((cell["accq_old1"] - cell["accq_old0"])
                  - (cell["accp_old1"] - cell["accp_old0"]))
    REPL_TOL = 0.03
    cell_repl = np.abs(excess) > REPL_TOL
    # Is the per-cell excess real structure or sampling noise? Split-half
    # reliability answers it directly: recompute `excess` on two disjoint halves of
    # the test sequences and correlate the two (T-1, L) maps. r ~ 0 means the
    # classification is a coin flip and every exclusion built on it is arbitrary.
    hA = (np.arange(n_test) % 2 == 0)
    def _excess_on(selm):
        return (((acc_q_new_1[selm].mean(0) - acc_q_new_0[selm].mean(0))
                 - (acc_p_new_1[selm].mean(0) - acc_p_new_0[selm].mean(0))))
    exA, exB = _excess_on(hA), _excess_on(~hA)
    sh_r = float(np.corrcoef(exA.ravel(), exB.ravel())[0, 1])
    sh_by_level = {f"d{L - d}": float(np.corrcoef(exA[:, d], exB[:, d])[0, 1])
                   for d in range(L)}
    # noise sd of a half-sample cell estimate, and of the full-sample one
    noise_half = float((exA - exB).std() / np.sqrt(2.0))
    split_half = {
        "pearson_r_between_halves": sh_r, "by_level": sh_by_level,
        "sd_excess_full_sample": float(excess.std()),
        "implied_noise_sd_full_sample": noise_half / np.sqrt(2.0),
        "spearman_brown_reliability": float(2 * sh_r / (1 + sh_r)) if sh_r > -1 else
        float("nan")}
    print(f"\n  split-half reliability of the per-cell readability excess: "
          f"r = {sh_r:+.3f}  (Spearman-Brown "
          f"{split_half['spearman_brown_reliability']:+.3f});  sd(excess) "
          f"{excess.std():.4f} vs implied noise sd "
          f"{split_half['implied_noise_sd_full_sample']:.4f}", flush=True)
    cell_common = (cell["b"] < 0.02) & ~cell_repl
    cell_ins = ~cell_repl & ~cell_common
    print(f"\n  cell classification over {(T - 1) * L} (position, level) cells "
          f"(|readability excess| > {REPL_TOL} => replacement): "
          f"replacement {int(cell_repl.sum())}  insertion {int(cell_ins.sum())}  "
          f"common {int(cell_common.sum())}")
    print(f"  readability excess on the node M READS   : mean {excess.mean():+.4f}  "
          f"|mean| {np.abs(excess).mean():.4f}  max |.| {np.abs(excess).max():.4f}")
    print(f"  readability excess on the node M LEAVES  : mean {excess_old.mean():+.4f}"
          f"  |mean| {np.abs(excess_old).mean():.4f}  max |.| "
          f"{np.abs(excess_old).max():.4f}", flush=True)

    # ---------------- assemble the signals -----------------------------------
    flat = lambda a: np.asarray(a).ravel()
    nll = flat(sig_te["nll"])
    bp = flat(orc["surprisal"])
    pos_idx = np.tile(np.arange(T - 1), (n_test, 1)).ravel()

    cont = ~is_bdry                                     # (L, T-1)
    def _cum(term, mask_ld, D):
        """sum_{d<=D} term[:,:,d] restricted to cells where mask_ld[d,t]."""
        w = mask_ld[:D + 1].T[None, :, :]               # (1, T-1, D+1)
        return (term[:, :, :D + 1] * w).sum(-1)

    ones_ld = np.ones((L, T - 1), dtype=bool)
    keep_repl = ~cell_repl.T                            # (L, T-1) -- keep non-repl

    results = {
        "config": {"v": v, "s": s, "L": L, "m": m, "n_layer": n_layer,
                   "n_head": n_head, "n_embd": n_embd, "belief_block": belief_block,
                   "n_probe_train": n_probe_train, "n_calib": n_calib,
                   "n_test": n_test, "probe_steps": probe_steps,
                   "probe_hidden": probe_hidden, "eval_seed": eval_seed,
                   "seed": seed, "tag": tag, "base_ckpt": ckpt,
                   "probe_temperature": temp, "replacement_tol": REPL_TOL},
        "boundary_counts": {f"d{L - d}": int(is_bdry[d].sum()) for d in range(L)},
        "cell_classification": {
            "n_replacement": int(cell_repl.sum()), "n_insertion": int(cell_ins.sum()),
            "n_common": int(cell_common.sum()),
            "readability_excess_new_node": {
                "mean": float(excess.mean()), "mean_abs": float(np.abs(excess).mean()),
                "max_abs": float(np.abs(excess).max())},
            "readability_excess_old_node": {
                "mean": float(excess_old.mean()),
                "mean_abs": float(np.abs(excess_old).mean()),
                "max_abs": float(np.abs(excess_old).max())},
            "by_level": {f"d{L - d}": {
                "replacement": int(cell_repl[:, d].sum()),
                "insertion": int(cell_ins[:, d].sum()),
                "common": int(cell_common[:, d].sum()),
                "replacement_at_boundary": int((cell_repl[:, d] & is_bdry[d]).sum()),
                "n_boundary": int(is_bdry[d].sum())} for d in range(L)}},
        "state_table": {}, "decomposition": {},
        "gate_A_tracked": {}, "gate_B_tracked": {},
        "split_half_excess": split_half,
        "cell_excess_new": excess.tolist(),        # (T-1, L), for re-analysis
        "cell_excess_old": excess_old.tolist(),
        "published": PUBLISHED,
    }

    # ---------------- the state table: unsubtracted, signed ------------------
    print(f"\n{'=' * 78}\nSTATE TABLE -- what the KL collapses. Each row is one "
          f"(level, class);\nreads at t and at t+1 are kept SEPARATE.\n{'=' * 78}")
    print(f"  {'level':<6}{'class':<14}{'n':>9}"
          f"{'probe@t':>9}{'probe@t1':>9}{'bayes@t':>9}{'bayes@t1':>9}{'excess':>9}"
          f"{'fid@t':>8}{'fid@t1':>8}{'M':>8}{'B':>8}{'d_den':>8}{'d_num':>8}")
    for d in range(L):
        for cname, cmask in (("continuation", cont[d]), ("boundary", is_bdry[d])):
            if not cmask.any():
                continue
            sel = np.s_[:, cmask, d]
            row = {
                "n_terms": int(cmask.sum()) * n_test,
                "probe_acc_new_at_t": float(acc_q_new_0[sel].mean()),
                "probe_acc_new_at_t1": float(acc_q_new_1[sel].mean()),
                "bayes_acc_new_at_t": float(acc_p_new_0[sel].mean()),
                "bayes_acc_new_at_t1": float(acc_p_new_1[sel].mean()),
                "probe_acc_old_at_t": float(acc_q_old_0[sel].mean()),
                "probe_acc_old_at_t1": float(acc_q_old_1[sel].mean()),
                "bayes_acc_old_at_t": float(acc_p_old_0[sel].mean()),
                "bayes_acc_old_at_t1": float(acc_p_old_1[sel].mean()),
                "probe_ent_at_t": float(H_q0[sel].mean()),
                "probe_ent_at_t1": float(H_q1[sel].mean()),
                "bayes_ent_at_t": float(H_p0[sel].mean()),
                "bayes_ent_at_t1": float(H_p1[sel].mean()),
                "fidelity_at_t": float(fid_0[sel].mean()),
                "fidelity_at_t1": float(fid_1[sel].mean()),
                "M_term": float(m_d[sel].mean()), "B_term": float(b_d[sel].mean()),
                "d_den_t_state": float(d_den[sel].mean()),
                "d_num_t1_state": float(d_num[sel].mean()),
            }
            row["readability_excess_new"] = (
                (row["probe_acc_new_at_t1"] - row["probe_acc_new_at_t"])
                - (row["bayes_acc_new_at_t1"] - row["bayes_acc_new_at_t"]))
            row["readability_excess_old"] = (
                (row["probe_acc_old_at_t1"] - row["probe_acc_old_at_t"])
                - (row["bayes_acc_old_at_t1"] - row["bayes_acc_old_at_t"]))
            results["state_table"][f"d{L - d}/{cname}"] = row
            print(f"  {'d' + str(L - d):<6}{cname:<14}{row['n_terms']:>9}"
                  f"{row['probe_acc_new_at_t']:>9.3f}{row['probe_acc_new_at_t1']:>9.3f}"
                  f"{row['bayes_acc_new_at_t']:>9.3f}{row['bayes_acc_new_at_t1']:>9.3f}"
                  f"{row['readability_excess_new']:>+9.3f}"
                  f"{row['fidelity_at_t']:>8.3f}{row['fidelity_at_t1']:>8.3f}"
                  f"{row['M_term']:>8.3f}{row['B_term']:>8.3f}"
                  f"{row['d_den_t_state']:>8.3f}{row['d_num_t1_state']:>8.3f}",
                  flush=True)

    print(f"\n  the node the chain LEAVES (node_old: current at t -> PAST at t+1 at"
          f" a boundary).\n  M never reads this, which is exactly where a "
          f"replacement would hide.")
    print(f"  {'level':<6}{'class':<14}{'probe@t':>9}{'probe@t1':>9}"
          f"{'bayes@t':>9}{'bayes@t1':>9}{'excess':>9}{'signature':>14}")
    for d in range(L):
        for cname, cmask in (("continuation", cont[d]), ("boundary", is_bdry[d])):
            if not cmask.any():
                continue
            r = results["state_table"][f"d{L - d}/{cname}"]
            dq = r["probe_acc_old_at_t1"] - r["probe_acc_old_at_t"]
            dp = r["bayes_acc_old_at_t1"] - r["bayes_acc_old_at_t"]
            # Petersen's replacement: the readout goes DOWN while the truth does
            # not -- the sign flip between the two contrasts.
            sig = ("REPLACEMENT" if (dq < -REPL_TOL and dp > -REPL_TOL) else
                   "common" if abs(dq - dp) <= REPL_TOL else "insertion")
            r["old_node_signature"] = sig
            print(f"  {'d' + str(L - d):<6}{cname:<14}"
                  f"{r['probe_acc_old_at_t']:>9.3f}{r['probe_acc_old_at_t1']:>9.3f}"
                  f"{r['bayes_acc_old_at_t']:>9.3f}{r['bayes_acc_old_at_t1']:>9.3f}"
                  f"{r['readability_excess_old']:>+9.3f}{sig:>14}", flush=True)

    # ---------------- Gate A + Gate B, re-run with the variants --------------
    print(f"\n{'=' * 78}\nGATE A / GATE B, tracked\n{'=' * 78}", flush=True)
    for D in Ds:
        B_chain_D = flat(orc["B_chain"][D])
        variants = {
            "M": _cum(m_d, ones_ld, D),
            "M_cont": _cum(m_d, cont, D),
            "M_bdry": _cum(m_d, is_bdry, D),
            "M_replexcl": _cum(m_d, keep_repl, D),
            "M_swapT": _cum(m_swapT, ones_ld, D),
            "M_swapT1": _cum(m_swapT1, ones_ld, D),
            "negH_t1": _cum(negH_1, ones_ld, D),
            "negH_t": _cum(negH_0, ones_ld, D),
            "dH": _cum(dH, ones_ld, D),
            "M_pointmass": _cum(m_pointmass, ones_ld, D),
            "M_hybT": _cum(hybT, ones_ld, D),      # DIAGNOSTIC: oracle in the t+1
            "d_num": _cum(d_num, ones_ld, D),      # state, so not a competitor
            "d_den": _cum(d_den, ones_ld, D),
            "nlq0_true": _cum(nlq0_true, ones_ld, D),      # DIAGNOSTIC: uses the
            "nlp0_true": _cum(nlp0_true, ones_ld, D),      # true latent value
            "H_post_oracle": orc["H_post"][D],             # oracle reference
        }
        # every exclusion variant gets its OWN shuffled guard, on exactly its own
        # cell subset -- the published M_shuffled is computed on the full cell set
        # and does not guard a subset.
        def _shuffle_within_position(a, off):
            out = a.copy()
            rs_ = np.random.default_rng(seed + D + off)
            for t in range(out.shape[1]):
                out[:, t] = out[rs_.permutation(out.shape[0]), t]
            return out
        variants["M_shuffled"] = _shuffle_within_position(variants["M"], 0)
        for vn, off in (("M_cont", 100), ("M_bdry", 200), ("M_replexcl", 300)):
            variants[f"{vn}_shuffled"] = _shuffle_within_position(variants[vn], off)
        sc = {k: flat(a) for k, a in variants.items()}
        sc["nll"] = nll
        sc["bp_surprisal"] = bp
        sc["r_temporal"] = flat(sig_te["r_temporal"])
        sc["B_chain"] = B_chain_D
        B_cont = flat(_cum(b_d, cont, D))
        B_bdry = flat(_cum(b_d, is_bdry, D))
        B_replexcl = flat(_cum(b_d, keep_repl, D))

        # --- Gate A
        ga = {name: _partial_r2(val, B_chain_D, nll) for name, val in sc.items()}
        ga["M_cont_vs_B_cont"] = _partial_r2(sc["M_cont"], B_cont, nll)
        ga["M_bdry_vs_B_bdry"] = _partial_r2(sc["M_bdry"], B_bdry, nll)
        ga["M_replexcl_vs_B_replexcl"] = _partial_r2(sc["M_replexcl"], B_replexcl, nll)
        ga["mean_M"] = float(sc["M"].mean())
        ga["mean_M_cont"] = float(sc["M_cont"].mean())
        ga["mean_M_bdry"] = float(sc["M_bdry"].mean())
        ga["mean_B_chain"] = float(B_chain_D.mean())
        ga["mean_B_cont"] = float(B_cont.mean())
        ga["mean_B_bdry"] = float(B_bdry.mean())
        results["gate_A_tracked"][f"D{D}"] = {"d_name": f"d{L - D}", **ga}

        # --- Gate B (families and matching identical to gates_ab)
        Bj = flat(orc["B_joint"][D])
        syn = Bj < 1e-9
        pos_B = Bj[Bj >= 1e-9]
        row = {"d_name": f"d{L - D}", "n_syn": int(syn.sum())}
        if pos_B.size:
            dis = Bj > np.median(pos_B)
            keep = syn | dis
            row["n_dis"] = int(dis.sum())
            pos_bins = max(n_nll_bins // 2, 8)
            strat = np.unique(np.stack([pos_idx, _strata(bp, pos_bins)], 1),
                              axis=0, return_inverse=True)[1]
            row["auc_pos_x_bp"] = {
                name: _stratified_auc(val[keep], dis[keep], strat[keep])
                for name, val in sc.items()}
            row["auc_raw"] = {name: _auc(val[keep], dis[keep])
                              for name, val in sc.items()}

            # --- the guard M_replexcl needs. Dropping ~47% of the cells costs
            # power whether or not the dropped cells were contaminated, so the
            # only interpretable comparison is against dropping the SAME NUMBER
            # of cells at random, per level. If the random control lands where
            # M_replexcl lands, the drop is the exclusion, not replacement.
            n_excl = cell_repl.sum(0)                        # (L,) per level
            rr = np.random.default_rng(seed + 900 + D)
            rand_aucs = []
            for _ in range(n_rand_excl):
                mk = np.ones((T - 1, L), dtype=bool)
                for d in range(L):
                    if n_excl[d]:
                        mk[rr.choice(T - 1, size=int(n_excl[d]), replace=False), d] \
                            = False
                rand_aucs.append(_stratified_auc(
                    flat(_cum(m_d, mk.T, D))[keep], dis[keep], strat[keep]))
            surv = keep_repl[:D + 1].T                       # (T-1, D+1)
            wsum = np.abs(m_d[:, :, :D + 1]).sum()
            row["replexcl_control"] = {
                "n_cells_excluded": int(cell_repl[:, :D + 1].sum()),
                "n_cells_total": int((T - 1) * (D + 1)),
                "frac_cells_excluded": float(cell_repl[:, :D + 1].mean()),
                "frac_of_M_mass_excluded": float(
                    (np.abs(m_d[:, :, :D + 1]) * (~surv)[None]).sum()
                    / max(wsum, 1e-30)),
                "frac_positions_with_no_surviving_term": float(
                    (~surv.any(1)).mean()),
                "auc_M_replexcl": row["auc_pos_x_bp"]["M_replexcl"],
                "auc_random_exclusion_mean": float(np.mean(rand_aucs)),
                "auc_random_exclusion_sd": float(np.std(rand_aucs)),
                "auc_random_exclusion_draws": [float(a) for a in rand_aucs],
                "n_syn": int(syn.sum()), "n_dis": int(dis.sum()),
                "note": ("families are defined by the ORACLE B_joint, so the "
                         "exclusion cannot change n_syn / n_dis -- it changes only "
                         "the score. The live risk is ties at 0 and lost mass, "
                         "which the random-exclusion control measures."),
            }
            # the most legible version: is M's separation concentrated in the
            # positions where the chain actually replaces a node?
            hb = is_bdry[:D + 1].any(0)                    # (T-1,)
            pos_has_b = hb[pos_idx]
            row["auc_pos_x_bp_by_position_class"] = {}
            for pname, pmask in (("boundary_positions", pos_has_b),
                                 ("continuation_positions", ~pos_has_b)):
                kk = keep & pmask
                if kk.sum() and dis[kk].any() and (~dis[kk]).any():
                    sub_auc = {"n": int(kk.sum()),
                               "frac_of_positions": float(pmask.mean())}
                    for name in ("M", "M_cont", "M_bdry", "negH_t1", "M_swapT",
                                 "M_swapT1", "M_shuffled", "B_chain"):
                        sub_auc[name] = _stratified_auc(sc[name][kk], dis[kk],
                                                        strat[kk])
                    row["auc_pos_x_bp_by_position_class"][pname] = sub_auc
            # family-split state table, at the levels Gate B is positive
            fam = {}
            for fname, fmask in (("syn", syn), ("dis", dis)):
                fm2 = fmask.reshape(n_test, T - 1)
                sub = {}
                for d in range(D + 1):
                    for cname, cmask in (("continuation", cont[d]),
                                         ("boundary", is_bdry[d])):
                        sel = fm2 & cmask[None, :]
                        if not sel.any():
                            continue
                        sub[f"d{L - d}/{cname}"] = {
                            "n": int(sel.sum()),
                            "probe_acc_at_t": float(acc_q_new_0[:, :, d][sel].mean()),
                            "probe_acc_at_t1": float(acc_q_new_1[:, :, d][sel].mean()),
                            "bayes_acc_at_t": float(acc_p_new_0[:, :, d][sel].mean()),
                            "bayes_acc_at_t1": float(acc_p_new_1[:, :, d][sel].mean()),
                            "fidelity_at_t": float(fid_0[:, :, d][sel].mean()),
                            "fidelity_at_t1": float(fid_1[:, :, d][sel].mean()),
                            "M_term": float(m_d[:, :, d][sel].mean()),
                            "B_term": float(b_d[:, :, d][sel].mean()),
                            "d_den_t_state": float(d_den[:, :, d][sel].mean()),
                            "d_num_t1_state": float(d_num[:, :, d][sel].mean())}
                fam[fname] = sub
            row["by_family"] = fam
        results["gate_B_tracked"][f"D{D}"] = row

    # ---------------- print the two headline blocks --------------------------
    print(f"\n  GATE A -- partial R^2( X ~ B_chain | nll ), post_block6")
    names_A = ["M", "M_cont", "M_bdry", "M_replexcl", "M_pointmass", "M_swapT",
               "M_swapT1", "negH_t1", "dH", "d_den", "d_num", "M_shuffled"]
    print(f"  {'D':<4}{'name':<6}" + "".join(f"{n:>12}" for n in names_A)
          + f"{'published':>11}")
    for D in Ds:
        r = results["gate_A_tracked"][f"D{D}"]
        print(f"  {D:<4}{r['d_name']:<6}"
              + "".join(f"{r[n]:>12.4f}" for n in names_A)
              + f"{PUBLISHED['gate_A_partial_r2_M_from_B_given_nll'][f'D{D}']:>11.4f}")

    print(f"\n  GATE B -- AUC, position x exact-surprisal matched")
    names_B = ["M", "M_cont", "M_bdry", "M_replexcl", "M_pointmass", "M_swapT",
               "M_swapT1", "negH_t1", "negH_t", "dH", "M_shuffled", "r_temporal",
               "nll"]
    print(f"  {'D':<4}{'name':<6}" + "".join(f"{n:>12}" for n in names_B)
          + f"{'published M':>13}")
    for D in Ds:
        r = results["gate_B_tracked"][f"D{D}"]
        if "auc_pos_x_bp" not in r:
            continue
        a = r["auc_pos_x_bp"]
        pub = PUBLISHED["gate_B_auc_pos_x_bp_matched_M"].get(f"D{D}", float("nan"))
        print(f"  {D:<4}{r['d_name']:<6}"
              + "".join(f"{a[n]:>12.4f}" for n in names_B) + f"{pub:>13.4f}")

    print(f"\n  GATE B -- guards for each exclusion variant (each shuffled on its "
          f"OWN cell subset; all must read ~0.5)")
    print(f"  {'D':<4}{'name':<6}{'M_shuf':>10}{'M_cont_shuf':>13}"
          f"{'M_bdry_shuf':>13}{'M_replexcl_shuf':>17}")
    for D in Ds:
        r = results["gate_B_tracked"][f"D{D}"]
        if "auc_pos_x_bp" not in r:
            continue
        a = r["auc_pos_x_bp"]
        print(f"  {D:<4}{r['d_name']:<6}{a['M_shuffled']:>10.4f}"
              f"{a['M_cont_shuffled']:>13.4f}{a['M_bdry_shuffled']:>13.4f}"
              f"{a['M_replexcl_shuffled']:>17.4f}")

    print(f"\n  GATE B -- M_replexcl against a RANDOM exclusion of the same size")
    print(f"  {'D':<4}{'name':<6}{'M':>9}{'replexcl':>10}{'random':>10}{'sd':>8}"
          f"{'frac_cells':>12}{'frac_mass':>11}{'dead_pos':>10}")
    for D in Ds:
        r = results["gate_B_tracked"][f"D{D}"]
        c = r.get("replexcl_control")
        if not c:
            continue
        print(f"  {D:<4}{r['d_name']:<6}{r['auc_pos_x_bp']['M']:>9.4f}"
              f"{c['auc_M_replexcl']:>10.4f}{c['auc_random_exclusion_mean']:>10.4f}"
              f"{c['auc_random_exclusion_sd']:>8.4f}"
              f"{c['frac_cells_excluded']:>12.3f}"
              f"{c['frac_of_M_mass_excluded']:>11.3f}"
              f"{c['frac_positions_with_no_surviving_term']:>10.3f}")

    print(f"\n  GATE B -- M's separation, split by whether the position contains a "
          f"class change")
    for D in Ds:
        r = results["gate_B_tracked"][f"D{D}"]
        bc = r.get("auc_pos_x_bp_by_position_class", {})
        for pname, pr in bc.items():
            print(f"  D{D} ({r['d_name']}) {pname:<24} frac_pos "
                  f"{pr['frac_of_positions']:.3f}  n {pr['n']:>7}  "
                  f"M {pr['M']:.4f}  M_cont {pr['M_cont']:.4f}  "
                  f"M_bdry {pr['M_bdry']:.4f}  negH_t1 {pr['negH_t1']:.4f}  "
                  f"guard {pr['M_shuffled']:.4f}")

    print(f"\n  STATE TABLE BY GATE-B FAMILY -- the within-position question. If the "
          f"probe's\n  readability changed with the family for reasons the truth does"
          f" not warrant, it\n  shows up as `excess` differing between syn and dis at"
          f" the same (level, class).")
    for D in (3, 4):
        r = results["gate_B_tracked"].get(f"D{D}", {})
        if "by_family" not in r:
            continue
        print(f"\n  D{D} ({r['d_name']}):")
        print(f"  {'level/class':<24}{'fam':<5}{'probe@t':>9}{'probe@t1':>9}"
              f"{'bayes@t':>9}{'bayes@t1':>9}{'excess':>9}{'M':>8}{'B':>8}"
              f"{'d_den':>8}{'d_num':>8}")
        for lk in r["by_family"]["dis"]:
            for fam in ("syn", "dis"):
                x = r["by_family"][fam].get(lk)
                if x is None:
                    continue
                ex = ((x["probe_acc_at_t1"] - x["probe_acc_at_t"])
                      - (x["bayes_acc_at_t1"] - x["bayes_acc_at_t"]))
                x["readability_excess"] = ex
                print(f"  {lk:<24}{fam:<5}{x['probe_acc_at_t']:>9.3f}"
                      f"{x['probe_acc_at_t1']:>9.3f}{x['bayes_acc_at_t']:>9.3f}"
                      f"{x['bayes_acc_at_t1']:>9.3f}{ex:>+9.3f}{x['M_term']:>8.3f}"
                      f"{x['B_term']:>8.3f}{x['d_den_t_state']:>8.3f}"
                      f"{x['d_num_t1_state']:>8.3f}")
    print("", flush=True)

    # ---------------- reproduction check -------------------------------------
    rep = {"gate_A": {}, "gate_B": {}}
    for D in Ds:
        got = results["gate_A_tracked"][f"D{D}"]["M"]
        want = PUBLISHED["gate_A_partial_r2_M_from_B_given_nll"][f"D{D}"]
        rep["gate_A"][f"D{D}"] = {"got": got, "published": want,
                                  "abs_err": abs(got - want)}
        if f"D{D}" in PUBLISHED["gate_B_auc_pos_x_bp_matched_M"]:
            g2 = results["gate_B_tracked"][f"D{D}"]["auc_pos_x_bp"]["M"]
            w2 = PUBLISHED["gate_B_auc_pos_x_bp_matched_M"][f"D{D}"]
            rep["gate_B"][f"D{D}"] = {"got": g2, "published": w2,
                                      "abs_err": abs(g2 - w2)}
    rep["max_abs_err_gate_A"] = max(r["abs_err"] for r in rep["gate_A"].values())
    rep["max_abs_err_gate_B"] = (max(r["abs_err"] for r in rep["gate_B"].values())
                                 if rep["gate_B"] else float("nan"))
    results["reproduction"] = rep
    print(f"\n  REPRODUCTION vs gates_ab_gateAB3: max |dGateA| "
          f"{rep['max_abs_err_gate_A']:.4f}   max |dGateB| "
          f"{rep['max_abs_err_gate_B']:.4f}", flush=True)

    out_dir = f"{DATA_DIR}/{key}/conditional_revision"
    os.makedirs(out_dir, exist_ok=True)
    name = f"tracking{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
