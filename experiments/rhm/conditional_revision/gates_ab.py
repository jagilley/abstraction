"""Conditional revision, Gates A and B: is the model's belief revision a distinct
signal from token surprisal, and does it track the oracle's?

See SPEC.md. Gate 0 (`conditional_revision.py`) passed -- moving the FM's
conditioning gap from depth to time flipped corr(residual, nll) from -0.336 to
+0.652 and the level profile from -0.786 to +0.484 -- so the residual now has an
aleatoric component and there is something to decompose. This file builds the
oracle and does the decomposing. Measurement only: the main model is frozen, no
arm is trained, nothing is taught.

  instrument self-check   E[B_D] == E[H(x_{t+1}|x_<=t) - H(x_{t+1}|z_<=D,x_<=t)]
  Gate A                  partial R^2( M ~ B | nll ) -- does the model's revision
                          carry oracle structure that surprisal does not explain?
  Gate B                  the constructed contrast. Synonym positions (oracle
                          B == 0 by construction) vs disambiguating ones, MATCHED
                          on nll, scored by AUC. This is the load-bearing test:
                          if M ranks the families like nll and not like B, then
                          revision is surprisal re-expressed in state space.

Why Gate B can be built exactly here: on this regime ~49% of positions have
B_joint == 0 to machine precision -- the arriving token is one of the `m`
synonymous realisations of structure the prefix already fixed, so the true
posterior over z_{<=D} cannot move. It is a DGP primitive, not a threshold we
chose. Those positions still carry real surprisal (mean 0.98 nats at D=0), and
they overlap the disambiguating family across surprisal deciles at 0.49-0.55 for
D=0..3 -- so `nll` can be held fixed by construction and only `B` varies.

Signals scored, all per position, all on held-out aligned sequences:

  nll            the model's token surprisal. The incumbent.
  bp_surprisal   -log P(x_{t+1}|x_<=t) exactly. The incumbent without model error.
  M              the model's belief revision, sum_d KL(q_{t+1}^{a_d} || q_t^{a_d})
                 over probe-decoded posteriors at x_{t+1}'s OWN level-d ancestor,
                 d <= D. Matched term-for-term by oracle `B_chain`.
                 The first run summed over all 63 latent nodes instead (matching
                 `B_marg`) and the probe noise accumulated 63-fold: mean M read
                 6.07 against a true B of 0.70, so M was noise, not belief. That
                 version is still reported as `M_allnode`; the chain is primary.
  r_temporal     Gate 0's residual, ||Delta_t - FM(h6[<=t])|| / ||Delta_t||.
  delta_norm     ||Delta_t|| alone, no forward model at all. Included because
                 Gate 0's residual could inherit its nll-correlation entirely
                 from its target's norm; if delta_norm scores like r_temporal,
                 the FM is not doing the work.
  B_joint        the oracle. The ceiling -- it defines the families in Gate B.
  M_shuffled     M with sequence identity permuted within each position. Kills
                 the position<->level coupling while preserving every signal's
                 positional structure.

Run:
  # smoke (~4 min, attached)
  modal run -m rhm.conditional_revision.gates_ab::gates_ab \
      --fm-steps 300 --probe-steps 300 --n-probe-train 300 --n-test 200 \
      --pool-size 20000 --tag smoke
  # the real thing (~45 min on an L4; the base checkpoint is reused, not retrained)
  modal run --detach -m rhm.conditional_revision.gates_ab::gates_ab --tag gateAB
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
app = modal.App("rhm-conditional-revision-ab", image=image)


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _r2(y, x):
    """R^2 of predicting y from x by simple linear regression (1-D numpy)."""
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
    import numpy as np
    xc, yc = x - x.mean(), y - y.mean()
    denom = (xc * xc).sum()
    beta = 0.0 if denom < 1e-30 else (xc * yc).sum() / denom
    return yc - beta * xc


def _partial_r2(y, x, z):
    """Fraction of y's z-orthogonal variance explained by x's z-orthogonal part."""
    return _r2(_resid(y, z), _resid(x, z))


def _ranks(a):
    from scipy.stats import rankdata
    return rankdata(a)


def _partial_r2_rank(y, x, z):
    """Spearman version. B, M and the KLs are heavy-tailed non-negatives, so the
    linear number can be dominated by a handful of positions; this is the
    monotone-only reading of the same quantity."""
    return _partial_r2(_ranks(y), _ranks(x), _ranks(z))


def _auc(score, pos_mask):
    """P(score[positive] > score[negative]), ties counted as 1/2."""
    import numpy as np
    n1 = int(pos_mask.sum())
    n0 = int((~pos_mask).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(score)
    return float((r[pos_mask].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _stratified_auc(score, pos_mask, strata):
    """AUC within each stratum, pooled by pair count -- the nll-matched readout.

    Holding `nll` fixed BY DESIGN means the incumbent scores ~0.5 here. That is
    the point of Gate B, not a defect of the estimator.
    """
    import numpy as np
    num = den = 0.0
    per = {}
    for k in np.unique(strata):
        sel = strata == k
        a = _auc(score[sel], pos_mask[sel])
        n1 = int(pos_mask[sel].sum())
        n0 = int(sel.sum() - n1)
        per[int(k)] = {"auc": a, "n_pos": n1, "n_neg": n0}
        if n1 and n0:
            num += a * n1 * n0
            den += n1 * n0
    return (float(num / den) if den else float("nan")), per


def _strata(vals, n_bins, atom_frac=0.002):
    """Matching strata that are tie-pure at the atoms of `vals`.

    The exact BP surprisal is heavily atomic -- P(x_{t+1}|prefix) is often uniform
    over a small set, so -log p piles up on 0, ln2, ln3, ln4... Plain quantile
    edges then repeat, `digitize` merges each atom with the continuous spread just
    above it, and the matching variable separates the families at AUC 0.999 INSIDE
    a stratum that is supposed to hold it fixed. (Observed: bp_surprisal matched on
    itself read 0.80 instead of 0.50.) So: every value carrying at least
    `atom_frac` of the mass becomes its own stratum, and whatever is left over is
    quantile-binned. A continuous variable has no atoms and this reduces to plain
    quantile binning.
    """
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


def _overlap(a_vals, b_vals, edges):
    """Histogram overlap coefficient of two samples on a shared binning."""
    import numpy as np
    ha, _ = np.histogram(a_vals, bins=edges)
    hb, _ = np.histogram(b_vals, bins=edges)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def gates_ab(
    # DGP / model -- must match the cached base checkpoint
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    shallow_block: str = "post_block0", deep_block: str = "post_block6",
    belief_blocks: str = "post_block6,post_block7",
    # FM (identical to gate0's, same seeds, so r_temporal reproduces bit-for-bit)
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0, fm_steps: int = 12000, fwd_lr: float = 1e-3,
    base_steps: int = 12000, batch_size: int = 64, weight_decay: float = 0.01,
    pool_size: int = 200000, data_seed: int = 7,
    # probes
    n_probe_train: int = 4000, n_calib: int = 500, n_test: int = 2500,
    probe_steps: int = 8000, probe_lr: float = 1e-3, probe_batch: int = 512,
    probe_hidden: int = 1024,
    # oracle / gates
    eval_seed: int = 999, oracle_chunk: int = 256, n_nll_bins: int = 40,
    log_interval: int = 1000, seed: int = 42, tag: str = "",
    base_ckpt: str = "",
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
    bblocks = [b.strip() for b in belief_blocks.split(",") if b.strip()]
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    n_nodes = (s ** L - 1) // (s - 1)                    # internal nodes, levels 0..L-1
    node_level = np.concatenate([np.full(s ** d, d) for d in range(L)])
    node_off = {d: (s ** d - 1) // (s - 1) for d in range(L)}

    # node index of x_{t+1}'s level-d ancestor, per signal position t
    anc_idx = np.stack([np.array([node_off[d] + ((t + 1) // (s ** (L - d)))
                                  for t in range(T - 1)]) for d in range(L)])
    # An NTP-trained causal transformer keeps what it needs to predict x_{t+1},
    # which is x_{t+1}'s ancestor chain -- not the whole parse tree. Measured:
    # the repo's single-node probe reads the LAST ancestor at 0.984 (d1) while a
    # shared head over all 63 nodes averages 0.18 against a 0.59 Bayes ceiling,
    # and no lr / capacity / standardisation / per-level-head variant moves it.
    # So the `chain` probe spends its loss only on nodes the model could hold:
    # the ancestors of t and of t+1 (both are needed, since M reads node
    # anc_d(t+1) from h[t] AND from h[t+1]).
    anc_mask = np.zeros((T, n_nodes), dtype=bool)
    for t in range(T):
        for u in (t, min(t + 1, T - 1)):
            for d in range(L):
                anc_mask[t, node_off[d] + (u // (s ** (L - d)))] = True
    anc_mask_t = torch.from_numpy(anc_mask)

    print(f"{'=' * 78}\nCONDITIONAL REVISION -- GATES A & B   {key}  "
          f"{n_layer}L/{n_head}H/{n_embd}D")
    print(f"  T={T}  internal latent nodes={n_nodes}  belief blocks={bblocks}")
    print(f"{'=' * 78}", flush=True)

    # ---------------- base model (cached; Gate 0 wrote it) ----------------
    ckpt = base_ckpt or (f"{DATA_DIR}/{key}/conditional_revision/"
                         f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_"
                         f"seed{seed}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"no base checkpoint at {ckpt}. Run gate0 first "
            f"(it trains and persists the base); this cut never retrains it, so "
            f"every gate reads the same frozen model.")
    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded frozen base <- {ckpt}", flush=True)

    # ---------------- data ----------------
    print(f"Generating pool ({pool_size:,} seqs)...", flush=True)
    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    # ---------------- temporal FM, frozen base (gate0's protocol and seeds) ----
    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    # gate0 constructs its co-trained depth FM here, before the three frozen-base
    # ones. Constructing and discarding a matching module keeps the torch RNG in
    # the same state, so the FMs below are bit-identical to gate0's and
    # `r_temporal` is the same object Gate 0 reported rather than a re-run of it.
    make_fm()
    fm_temporal, fm_temporal_direct, fm_depth_frozen = make_fm(), make_fm(), make_fm()
    fms = {"temporal": fm_temporal, "temporal_direct": fm_temporal_direct,
           "depth_frozen": fm_depth_frozen}
    opts = {k: torch.optim.AdamW(f.parameters(), lr=fwd_lr, weight_decay=weight_decay)
            for k, f in fms.items()}

    def fm_predict(name, h_shallow, h_deep):
        if name == "temporal":
            return (fm_temporal(h_deep) - h_deep)[:, :-1, :]
        if name == "temporal_direct":
            return fm_temporal_direct(h_deep)[:, :-1, :]
        return fm_depth_frozen(h_shallow)

    # The FMs are a deterministic function of (frozen base, seeds, fm_steps), so
    # cache them: re-running the gates should not retrain them, exactly as it
    # does not retrain the base.
    fm_ckpt = (f"{DATA_DIR}/{key}/conditional_revision/fms_frozen_"
               f"steps{fm_steps}_seed{seed}.pt")
    if os.path.exists(fm_ckpt):
        fsd = torch.load(fm_ckpt, map_location=device)
        for k_, f_ in fms.items():
            f_.load_state_dict(fsd[k_])
        print(f"\n--- temporal FM: loaded cached {fm_ckpt} ---", flush=True)
        fm_steps_run = 0
    else:
        fm_steps_run = fm_steps
    print(f"\n--- temporal FM: {fm_steps_run} steps against the FROZEN base ---",
          flush=True)
    gen_fm = torch.Generator().manual_seed(seed + 1000)
    for step in range(fm_steps_run):
        for f in fms.values():
            f.train()
        x, y = get_ntp_batch(gen_fm)
        with torch.no_grad():
            _, _, inter = model(x, y, return_intermediates=True)
            h_shallow, h_deep = inter[shallow_block], inter[deep_block]
            delta = h_deep[:, 1:, :] - h_deep[:, :-1, :]
        losses = {}
        for name in fms:
            tgt = h_deep if name == "depth_frozen" else delta
            l = F.mse_loss(fm_predict(name, h_shallow, h_deep), tgt)
            opts[name].zero_grad(); l.backward(); opts[name].step()
            losses[name] = l.item()
        if step % log_interval == 0 or step == fm_steps - 1:
            print(f"  fm {step:6d}  " + "  ".join(f"{k} {u:.4f}" for k, u in losses.items()),
                  flush=True)
    for f in fms.values():
        f.eval()
    if fm_steps_run:
        os.makedirs(os.path.dirname(fm_ckpt), exist_ok=True)
        torch.save({k_: f_.state_dict() for k_, f_ in fms.items()}, fm_ckpt)
        volume.commit()
        print(f"  saved FMs -> {fm_ckpt}", flush=True)

    # ---------------- aligned sequences: probe-train / calib / test ----------
    n_align = n_probe_train + n_calib + n_test
    print(f"\nGenerating {n_align} aligned sequences (probe {n_probe_train} / "
          f"calib {n_calib} / test {n_test})...", flush=True)
    al_seqs, al_lf, _ = _generate_with_traces(rules, n_align, eval_seed)
    al_x = torch.from_numpy(al_seqs.astype(np.int64))
    # node targets: (n, n_nodes), constant across positions within a sequence
    node_y = np.concatenate([al_lf[d] for d in range(L)], axis=1)
    assert node_y.shape[1] == n_nodes
    node_y_t = torch.from_numpy(node_y.astype(np.int64))

    sl_tr = slice(0, n_probe_train)
    sl_ca = slice(n_probe_train, n_probe_train + n_calib)
    sl_te = slice(n_probe_train + n_calib, n_align)

    @torch.no_grad()
    def collect(idx_slice):
        """Activations at every belief block, plus nll / FM residual signals."""
        acts = {b: [] for b in bblocks}
        nlls, rel_t, abs_t, dn, rel_d = [], [], [], [], []
        xs = al_x[idx_slice]
        for i in range(0, xs.shape[0], 256):
            xa = xs[i:i + 256].to(device)
            ya = torch.cat([xa[:, 1:], xa[:, :1]], dim=1)   # last col invalid, dropped
            logits, _, inter = model(xa, ya, return_intermediates=True)
            nll = F.cross_entropy(logits.reshape(-1, v), ya.reshape(-1),
                                  reduction="none").reshape(xa.shape[0], T)
            h_shallow, h_deep = inter[shallow_block], inter[deep_block]
            delta = h_deep[:, 1:, :] - h_deep[:, :-1, :]
            pred = fm_predict("temporal", h_shallow, h_deep)
            r = (delta - pred).norm(dim=-1)
            dpred = fm_depth_frozen(h_shallow)[:, :-1, :]
            dtgt = h_deep[:, :-1, :]
            for b in bblocks:
                acts[b].append(inter[b].float().cpu())
            nlls.append(nll[:, :T - 1].cpu())
            abs_t.append(r.cpu())
            rel_t.append((r / (delta.norm(dim=-1) + 1e-6)).cpu())
            dn.append(delta.norm(dim=-1).cpu())
            rel_d.append(((dtgt - dpred).norm(dim=-1)
                          / (dtgt.norm(dim=-1) + 1e-6)).cpu())
        return ({b: torch.cat(vs) for b, vs in acts.items()},
                {"nll": torch.cat(nlls).numpy(),
                 "r_temporal": torch.cat(rel_t).numpy(),
                 "r_temporal_abs": torch.cat(abs_t).numpy(),
                 "delta_norm": torch.cat(dn).numpy(),
                 "r_depth": torch.cat(rel_d).numpy()})

    acts_tr, _ = collect(sl_tr)
    acts_ca, _ = collect(sl_ca)
    acts_te, sig_te = collect(sl_te)
    print(f"  reproduced Gate-0 consistency check on aligned test sequences: "
          f"corr(r_temporal, nll) = "
          f"{np.corrcoef(sig_te['r_temporal'].ravel(), sig_te['nll'].ravel())[0,1]:+.4f}"
          f"   corr(r_depth, nll) = "
          f"{np.corrcoef(sig_te['r_depth'].ravel(), sig_te['nll'].ravel())[0,1]:+.4f}",
          flush=True)
    print(f"  (gate0 aligned-pooled read temporal +0.6527, depth_frozen -0.3081)",
          flush=True)

    # ---------------- per-position ancestor probes -> the model's belief -----
    # One head over EVERY internal node, read at every position. For nodes the
    # prefix has not reached, the Bayes-optimal output is the node's marginal, and
    # the oracle says the same -- so "the probe learns nothing there" is the
    # correct behaviour, not a failure.
    class Probe(nn.Module):
        def __init__(self, d_in, hidden, n_out):
            super().__init__()
            self.net = (nn.Linear(d_in, n_out) if hidden == 0 else
                        nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                      nn.Linear(hidden, n_out)))

        def forward(self, x):
            return self.net(x)

    def train_probe(block, hidden, masked):
        """masked=False -> every latent node (SPEC's literal object, `M_allnode`).
        masked=True  -> only the nodes in `anc_mask` (the primary `M`)."""
        Xtr = acts_tr[block].reshape(-1, n_embd)
        Ytr = node_y_t[sl_tr].unsqueeze(1).expand(-1, T, -1).reshape(-1, n_nodes)
        probe = Probe(n_embd, hidden, n_nodes * v).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed + 7)
        mu = Xtr.mean(0, keepdim=True)
        sdv = Xtr.std(0, keepdim=True) + 1e-6
        tag_ = "chain" if masked else "full"
        for st in range(probe_steps):
            ix = torch.randint(0, Xtr.shape[0], (probe_batch,), generator=g)
            xb = ((Xtr[ix] - mu) / sdv).to(device)
            yb = Ytr[ix].to(device)
            out = probe(xb).view(-1, n_nodes, v)
            ce = F.cross_entropy(out.reshape(-1, v), yb.reshape(-1),
                                 reduction="none").view(-1, n_nodes)
            if masked:
                mk = anc_mask_t[(ix % T)].to(device)
                loss = (ce * mk).sum() / mk.sum().clamp(min=1)
            else:
                loss = ce.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if st % max(probe_steps // 4, 1) == 0:
                print(f"    probe[{block},{tag_}] {st:5d}  ce {loss.item():.4f}",
                      flush=True)
        probe.eval()
        return probe, mu, sdv

    @torch.no_grad()
    def probe_logits(probe, mu, sdv, acts_block, chunk=64):
        outs = []
        for i in range(0, acts_block.shape[0], chunk):
            xb = ((acts_block[i:i + chunk] - mu) / sdv).to(device)
            outs.append(probe(xb).view(xb.shape[0], T, n_nodes, v).cpu())
        return torch.cat(outs)

    def fit_temperature(logits_ca, y_ca, masked=False):
        """One scalar T minimising held-out NLL. KL-of-softmax readouts are only
        as good as the probe's calibration, and an over-confident probe inflates
        M everywhere."""
        logT = torch.zeros(1, requires_grad=True)
        yb = y_ca.unsqueeze(1).expand(-1, T, -1).reshape(-1)
        lg = logits_ca.reshape(-1, v)
        if masked:      # calibrate on the nodes the chain probe was trained on
            keepm = anc_mask_t.unsqueeze(0).expand(
                logits_ca.shape[0], -1, -1).reshape(-1)
            lg, yb = lg[keepm], yb[keepm]
        if lg.shape[0] > 500_000:                      # LBFGS on 2M rows is wasteful
            g = torch.Generator().manual_seed(seed + 13)
            sub = torch.randperm(lg.shape[0], generator=g)[:500_000]
            lg, yb = lg[sub], yb[sub]
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lg / logT.exp(), yb)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            before = F.cross_entropy(lg, yb).item()
            after = F.cross_entropy(lg / logT.exp(), yb).item()
        return float(logT.exp().item()), before, after

    # ---------------- the oracle ----------------
    print(f"\n--- oracle: exact BP on the test sequences ---", flush=True)
    te_lf = [al_lf[d][sl_te] for d in range(L + 1)]
    orc = ORC.revision_and_entropy(rules, al_seqs[sl_te], te_lf,
                                   chunk=oracle_chunk, verbose=True,
                                   anc_mask=anc_mask)
    chk = ORC.self_check(orc)
    print(f"\n  INSTRUMENT SELF-CHECK  E[B_D] vs E[H_tot - H_irr_D]:")
    for k, r in chk["per_D"].items():
        print(f"    {k} ({r['d_name']}): {r['mean_B_joint']:.5f} vs "
              f"{r['mean_H_tot_minus_H_irr']:.5f}   abs_err {r['abs_err']:.2e}  "
              f"{'ok' if r['passed'] else 'FAIL'}")
    print(f"  passed: {chk['passed']}", flush=True)
    if not chk["passed"]:
        raise RuntimeError("instrument self-check failed -- the marginal extraction "
                           "is wrong and nothing downstream is interpretable")

    Ds = orc["Ds"]
    print(f"  Bayes ceiling for the chain probe (same masked node/position pairs): "
          f"{ {k: round(x, 3) for k, x in orc['bayes_acc_chain'].items()} }", flush=True)
    flat = lambda a: np.asarray(a).ravel()
    nll = flat(sig_te["nll"])
    pos_idx = np.tile(np.arange(T - 1), (n_test, 1)).ravel()
    signals = {
        "nll": nll,
        "bp_surprisal": flat(orc["surprisal"]),
        "r_temporal": flat(sig_te["r_temporal"]),
        "r_temporal_abs": flat(sig_te["r_temporal_abs"]),
        "delta_norm": flat(sig_te["delta_norm"]),
        "r_depth": flat(sig_te["r_depth"]),
    }

    # ---------------- Gate A ----------------
    results = {
        "config": {
            "v": v, "s": s, "L": L, "m": m, "n_layer": n_layer, "n_head": n_head,
            "n_embd": n_embd, "shallow_block": shallow_block,
            "deep_block": deep_block, "belief_blocks": bblocks,
            "fm_steps": fm_steps, "base_ckpt": ckpt,
            "n_probe_train": n_probe_train, "n_calib": n_calib, "n_test": n_test,
            "probe_steps": probe_steps, "probe_hidden": probe_hidden,
            "eval_seed": eval_seed, "seed": seed, "tag": tag,
        },
        "instrument_self_check": chk,
        "bayes_acc_chain_ceiling": orc["bayes_acc_chain"],
        "gate0_consistency": {
            "corr_r_temporal_nll_aligned": float(
                np.corrcoef(signals["r_temporal"], nll)[0, 1]),
            "corr_r_depth_nll_aligned": float(
                np.corrcoef(signals["r_depth"], nll)[0, 1]),
            "gate0_reference_temporal": 0.6527,
            "gate0_reference_depth_frozen": -0.3081,
        },
        "oracle_summary": {
            f"D{D}": {
                "d_name": f"d{L - D}",
                "mean_B_joint": float(orc["B_joint"][D].mean()),
                "mean_B_marg": float(orc["B_marg"][D].mean()),
                "frac_B_zero": float((orc["B_joint"][D] < 1e-9).mean()),
                "mean_H_post": float(orc["H_post"][D].mean()),
                "mean_H_irr": float(orc["H_irr"][D].mean()),
                "r2_B_from_nll": _r2(flat(orc["B_joint"][D]), nll),
                "r2_B_from_bp_surprisal": _r2(flat(orc["B_joint"][D]),
                                              signals["bp_surprisal"]),
            } for D in Ds},
        "probes": {}, "gate_A": {}, "gate_B": {},
    }

    print(f"\n{'=' * 78}\nGATE A -- does the model's revision carry oracle structure "
          f"nll does not explain?\n{'=' * 78}", flush=True)

    M_by_block, Msh_by_block, Mc_by_block = {}, {}, {}
    for block in bblocks:
        qs = {}
        for kind, masked in (("chain", True), ("full", False)):
            probe, mu, sdv = train_probe(block, probe_hidden, masked)
            lg_ca = probe_logits(probe, mu, sdv, acts_ca[block])
            temp, ce_before, ce_after = fit_temperature(lg_ca, node_y_t[sl_ca],
                                                        masked)
            lg_te = probe_logits(probe, mu, sdv, acts_te[block]) / temp
            qs[kind] = torch.softmax(lg_te, dim=-1).numpy()

            # Probe quality, held out, split by whether the node is one the model
            # has reason to hold. `anc` = nodes in anc_mask at that position (the
            # ones M reads); `other` = the rest. The reference line for `anc` at
            # the last position is the repo's d1 0.979 / d3 0.836 / d6 0.088.
            acc_anc, acc_oth, acc_last = {}, {}, {}
            y_te = node_y[sl_te]
            pred = lg_te.argmax(-1).numpy()                    # (n, T, n_nodes)
            hit = pred == y_te[:, None, :]
            for d in range(L):
                j0, j1 = node_off[d], node_off[d] + s ** d
                mk = anc_mask[:, j0:j1]                        # (T, s^d)
                h = hit[:, :, j0:j1]
                acc_anc[f"d{L - d}"] = float(h[:, mk].mean()) if mk.any() else float("nan")
                acc_oth[f"d{L - d}"] = (float(h[:, ~mk].mean()) if (~mk).any()
                                        else float("nan"))
                acc_last[f"d{L - d}"] = float(
                    (pred[:, -1, j0:j1] == y_te[:, j0:j1]).mean())
            results["probes"][f"{block}/{kind}"] = {
                "temperature": temp, "calib_ce_before": ce_before,
                "calib_ce_after": ce_after, "chance_acc": 1.0 / v,
                "acc_ancestor_nodes": acc_anc, "acc_other_nodes": acc_oth,
                "last_pos_acc_by_level": acc_last}
            print(f"  probe[{block}/{kind}] temp {temp:.3f}  calib CE "
                  f"{ce_before:.4f}->{ce_after:.4f}  (chance {1.0 / v:.3f})")
            print(f"    acc on ANCESTOR nodes { {k: round(x, 3) for k, x in acc_anc.items()} }")
            print(f"    acc on other    nodes { {k: round(x, 3) for k, x in acc_oth.items()} }",
                  flush=True)

        def _kl_nodes(q):
            qn, qo = q[:, 1:], q[:, :-1]                       # (n, T-1, n_nodes, v)
            return (qn * (np.log(np.clip(qn, 1e-30, None))
                          - np.log(np.clip(qo, 1e-30, None)))).sum(-1)

        # M_allnode: SPEC's literal object, every latent node, matched by B_marg.
        kl_full = _kl_nodes(qs["full"])
        M_by_block[block] = {D: kl_full[:, :, node_level <= D].sum(-1) for D in Ds}
        # M (primary): the same single head read only at x_{t+1}'s own ancestor at
        # each level -- <=6 KL terms instead of 63, matched by oracle B_chain, and
        # one head for both t and t+1 so there is no calibration mismatch.
        kl_chain = _kl_nodes(qs["chain"])
        per_d = np.stack([np.take_along_axis(
            kl_chain, anc_idx[d][None, :, None], axis=2)[:, :, 0] for d in range(L)])
        Mc_by_block[block] = {D: per_d[:D + 1].sum(0) for D in Ds}
        # sequence identity permuted WITHIN each position: same marginal
        # distribution per position, same position<->level coupling, no
        # correspondence to that sequence's oracle. Scored everywhere M is.
        Msh_by_block[block] = {}
        for D in Ds:
            rng = np.random.default_rng(seed + D)
            a = Mc_by_block[block][D].copy()
            for t in range(a.shape[1]):
                a[:, t] = a[rng.permutation(a.shape[0]), t]
            Msh_by_block[block][D] = a

    primary_block = bblocks[0]
    for block in bblocks:
        rows = {}
        for D in Ds:
            M = flat(Mc_by_block[block][D])          # PRIMARY: chain
            Mall = flat(M_by_block[block][D])         # secondary: all 63 nodes
            B = flat(orc["B_chain"][D])               # the exactly-matched oracle
            Bj = flat(orc["B_joint"][D])
            Bm = flat(orc["B_marg"][D])
            rows[f"D{D}"] = {
                "d_name": f"d{L - D}",
                "partial_r2_M_from_B_given_nll": _partial_r2(M, B, nll),
                "partial_r2_M_from_nll_given_B": _partial_r2(M, nll, B),
                "r2_B_from_nll": _r2(B, nll),
                "r2_M_from_nll": _r2(M, nll),
                "r2_M_from_B": _r2(M, B),
                "partial_r2_M_from_Bjoint_given_nll": _partial_r2(M, Bj, nll),
                "allnode_partial_r2_M_from_Bmarg_given_nll": _partial_r2(Mall, Bm, nll),
                "allnode_mean_M": float(Mall.mean()),
                "mean_B_chain": float(B.mean()),
                "rank_partial_r2_M_from_B_given_nll": _partial_r2_rank(M, B, nll),
                "rank_partial_r2_M_from_nll_given_B": _partial_r2_rank(M, nll, B),
                "shuffled_partial_r2_M_from_B_given_nll": _partial_r2(
                    flat(Msh_by_block[block][D]), B, nll),
                "mean_M": float(M.mean()),
                "compression_B_minus_M": float((B - M).mean()),
            }
        results["gate_A"][block] = rows
        print(f"\n  block {block}:")
        print(f"  {'D':<4}{'name':<6}{'pR2(M~B|nll)':>14}{'pR2(M~nll|B)':>14}"
              f"{'R2(B~nll)':>11}{'rank pR2':>10}{'shuffled':>10}{'B-M':>9}")
        for D in Ds:
            r = results["gate_A"][block][f"D{D}"]
            print(f"  {D:<4}{r['d_name']:<6}"
                  f"{r['partial_r2_M_from_B_given_nll']:>14.4f}"
                  f"{r['partial_r2_M_from_nll_given_B']:>14.4f}"
                  f"{r['r2_B_from_nll']:>11.4f}"
                  f"{r['rank_partial_r2_M_from_B_given_nll']:>10.4f}"
                  f"{r['shuffled_partial_r2_M_from_B_given_nll']:>10.4f}"
                  f"{r['compression_B_minus_M']:>9.4f}")
    best_A = max(results["gate_A"][primary_block][f"D{D}"]
                 ["partial_r2_M_from_B_given_nll"] for D in Ds)
    verdict_A = ("KILL" if best_A < 0.05 else
                 "PROCEED" if best_A > 0.15 else "UNDERPOWERED")
    print(f"\n  GATE A verdict ({primary_block}): {verdict_A}  "
          f"(best partial R^2 = {best_A:.4f}; <0.05 kill, >0.15 proceed)", flush=True)
    results["gate_A"]["verdict"] = {"branch": verdict_A, "best_partial_r2": best_A,
                                    "primary_block": primary_block}

    # ---------------- Gate B ----------------
    print(f"\n{'=' * 78}\nGATE B -- the constructed contrast (synonym vs "
          f"disambiguating, nll held fixed)\n{'=' * 78}", flush=True)
    for D in Ds:
        B = flat(orc["B_joint"][D])
        syn = B < 1e-9
        pos_B = B[B >= 1e-9]
        if pos_B.size == 0:
            continue
        dis = B > np.median(pos_B)
        keep = syn | dis
        sc = {**signals,
              "M": flat(Mc_by_block[primary_block][D]),
              "M_allnode": flat(M_by_block[primary_block][D]),
              "M_shuffled": flat(Msh_by_block[primary_block][D]),
              "B_chain": flat(orc["B_chain"][D]),
              "B_joint": B}
        row = {"d_name": f"d{L - D}", "n_syn": int(syn.sum()), "n_dis": int(dis.sum()),
               "signals": {}}

        # Matchings, in increasing strength. Position is a real confound here:
        # in aligned sequences a position fixes which hierarchy level the arriving
        # token completes, and family membership is strongly position-determined,
        # so a signal that merely varies with position separates the families
        # without carrying anything about THIS sequence. Measured: M_shuffled
        # (sequence identity permuted within position, so position structure
        # intact and content destroyed) read 0.70 under bp-matching alone.
        #   nll        SPEC's incumbent -- the claim is about beating this
        #   bp         the same without the model's estimation error, so a signal
        #              that is merely a better surprisal estimate scores 0.5
        #   pos        position only. M_shuffled is exactly 0.5 here by
        #              construction, which makes it a self-validating guard
        #   pos_x_bp   both at once. The strongest, and the one to read.
        pos_bins = max(n_nll_bins // 2, 8)
        matchings = [
            ("nll", _strata(nll, n_nll_bins)),
            ("bp_surprisal", _strata(signals["bp_surprisal"], n_nll_bins)),
            ("pos", pos_idx.copy()),
            # exact pair encoding: arithmetic like pos*K + bin silently collides
            # across positions whenever the surprisal has more atoms than K
            ("pos_x_bp", np.unique(
                np.stack([pos_idx, _strata(signals["bp_surprisal"], pos_bins)], 1),
                axis=0, return_inverse=True)[1]),
        ]
        for mname, strata in matchings:
            if mname in ("nll", "bp_surprisal"):
                mvals = nll if mname == "nll" else signals["bp_surprisal"]
                ov_edges = np.unique(np.quantile(
                    mvals[keep], np.linspace(0, 1, n_nll_bins + 1)))
                ov_edges[0] -= 1e-9; ov_edges[-1] += 1e-9
                row[f"{mname}_overlap"] = _overlap(mvals[syn], mvals[dis], ov_edges)
                row[f"mean_{mname}_syn"] = float(mvals[syn].mean())
                row[f"mean_{mname}_dis"] = float(mvals[dis].mean())
            for name, val in sc.items():
                d = row["signals"].setdefault(name, {})
                if "auc_raw" not in d:
                    d["auc_raw"] = _auc(val[keep], dis[keep])
                a_str, _ = _stratified_auc(val[keep], dis[keep], strata[keep])
                d[f"auc_{mname}_matched"] = a_str
            row[f"{mname}_n_strata"] = int(len(np.unique(strata[keep])))
            # guards: a matching variable held against itself, and the
            # position-shuffled null under position matching, must both read ~0.5
            guards = {"nll": "nll", "bp_surprisal": "bp_surprisal",
                      "pos": "M_shuffled", "pos_x_bp": "M_shuffled"}
            g = row["signals"][guards[mname]][f"auc_{mname}_matched"]
            row[f"{mname}_guard_auc"] = g
            row[f"{mname}_guard_signal"] = guards[mname]
            if abs(g - 0.5) > 0.02:
                print(f"    WARNING: {mname} guard ({guards[mname]}) reads {g:.4f}, "
                      f"not ~0.5 -- strata leak, treat this column as inflated",
                      flush=True)
        results["gate_B"][f"D{D}"] = row
        print(f"\n  D{D} ({row['d_name']}): n_syn {row['n_syn']}  n_dis {row['n_dis']}"
              f"   overlap nll {row['nll_overlap']:.3f} / bp {row['bp_surprisal_overlap']:.3f}"
              f"   guards nll {row['nll_guard_auc']:.3f} bp {row['bp_surprisal_guard_auc']:.3f}"
              f" pos {row['pos_guard_auc']:.3f} pos_x_bp {row['pos_x_bp_guard_auc']:.3f}")
        print(f"    {'signal':<14}{'raw':>8}{'nll':>9}{'bp':>9}{'pos':>9}{'pos_x_bp':>10}")
        for name in sc:
            d = row["signals"][name]
            print(f"    {name:<14}{d['auc_raw']:>8.4f}{d['auc_nll_matched']:>9.4f}"
                  f"{d['auc_bp_surprisal_matched']:>9.4f}{d['auc_pos_matched']:>9.4f}"
                  f"{d['auc_pos_x_bp_matched']:>10.4f}")

    print(f"\n  Registered prediction: M beats nll on nll-matched AUC at D=0..3 "
          f"(the levels where synonymy and disambiguation are distinguishable) and "
          f"not leaf-adjacent. B is the ceiling; nll sits at ~0.5 by construction.",
          flush=True)

    out_dir = f"{DATA_DIR}/{key}/conditional_revision"
    os.makedirs(out_dir, exist_ok=True)
    name = f"gates_ab{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
