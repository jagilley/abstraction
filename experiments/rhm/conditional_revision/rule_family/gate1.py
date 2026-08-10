"""GATE 1 -- does the model's state express RULE revision, separably from surprisal?

Pre-registered in `NOTES.md` section 8. The object is

    M_rule = KL( q^rule_{t+1} || q^rule_t )

from a probe-decoded posterior over which of R rule sets is active, matched term-for-term
by the exact mixture oracle's `rule_rev` (`oracle_mixture.mixture_profiles`). Families are
the oracle's own: positions where `rule_rev` is exactly zero, against positions where it is
above the median of the positive part -- a DGP primitive, not a chosen threshold.

WHAT THIS REGIME SUPPLIES that the parent's Gate B could not: context depth is orthogonal
to position by construction (windows are whole aligned sequences), so the SAME position with
the SAME exact surprisal is rule-revision-heavy early and rule-revision-zero late. The
contrast is within-position, within-surprisal, across-context-depth.

-------------------------------------------------------------------------------
TWO DEVIATIONS FROM THE PRE-REGISTRATION, both forced by a sizing pass on the exact
oracle (40 windows, CPU) run before any model was touched. Recorded here rather than
folded in silently.

(1) THE `flat-in-depth => identity` KILL IS WEAKER THAN IT WAS WRITTEN TO BE.
    The pre-registration predicts the separation decays with depth "tracking the oracle's
    rule_rev profile", and kills a flat curve as rule IDENTITY decoding. The aggregate
    profile does decay hard (summed rule_rev per sequence 1.45 -> 0.08 nats over depths
    0..6). But the decay is almost entirely BASE RATE, not per-event magnitude:

        depth      0      1      2      3      4      5      6      7
        frac rule_rev>0    .451   .368   .267   .154   .100   .046   .025   .005
        mean rule_rev|>0   .050   .040   .051   .037   .031   .028   .050   .005

    A within-depth binary contrast between `rule_rev` high and `rule_rev == 0` therefore
    has a roughly FLAT truth-magnitude across depths 0-6. So a flat AUC curve is what a
    genuinely revision-tracking readout should produce here, and the pre-registered kill
    would misfire. The kill is retained and reported, but the discriminator that actually
    separates identity from revision is added:

      * `graded`  -- partial R^2( M_rule ~ rule_rev | nll_mix ) among POSITIVE cells only.
        An identity decoder cannot produce a graded response to revision MAGNITUDE at
        fixed surprisal; the binary contrast alone cannot tell the two apart.
      * the before-state decomposition (`M_pointmass`, `negH_t`) that the parent's
        tracking appendix used to show most of `M`'s AUC was a before-state readout.

    Depth 7 is reported as an internal negative control: the oracle says the rule is
    identified (H_rule 0.017 of ln R = 4.159 nats) and there are essentially no positive
    cells, so any separation there is instrument, not phenomenon.

(2) A PARSE CONFOUND THE PRE-REGISTRATION DOES NOT CONTROL, AND THE FIX.
    `d2_R64_nF2` differs in `rules[4]` on 2 of 16 features. At high depth `rule_rev > 0`
    is close to "the current level-4 ancestor's feature is one of the two differing ones"
    -- P(rule_rev>0 | differing ancestor) / P(rule_rev>0 | not) runs 1.9x at depth 0 but
    4.7x at depth 6. That is a PARSE fact, decodable from any model that represents level-4
    features (both arms do, at 0.6-0.9), with no rule inference whatever. The pooled
    contrast is therefore partly a parse contrast.

    Fix: the primary cell set is RESTRICTED to differing-ancestor cells, inside which the
    families are "this token discriminated between rule sets" vs "it did not" at a fixed
    parse role. This is the same move the language sibling's `negate_live` / `negate_dead`
    makes to remove its frame confound. The pre-registered pooled column is still reported.

    Consequence: `position` cannot be the absolute index inside the restricted set (too
    few cells per stratum). The primary position stratum is `n_open(t)` = how many levels'
    constituents are still unclosed after t, which is the causal content of "position" here
    (it fixes which hierarchy level the token completes) and is the variable section 7's
    read-position table is organised by. Absolute position is kept as a stricter secondary
    column wherever it has pairs.

ALSO ADDED (from the language sibling, `a2a_forward/conditional_revision/`): `h_before_dir`
/ `h_after_dir` / `h_swap_dir`, a supervised linear decode of the family label from the
state itself. These are DIAGNOSTICS, not competing primaries. They exist to keep the two
negatives distinguishable: that cut found every belief readout at chance while the state
decoded the same distinction at 0.99, i.e. instrument-fails-to-transfer rather than
phenomenon-absent. `h_before_dir` is additionally the contrast-validity check -- if the
PREFIX state already predicts the family, every state-based row is reading the template.

    cd experiments
    # smoke, attached, ~6 min
    modal run -m rhm.conditional_revision.rule_family.gate1::gate1 \
        --n-windows 128 --n-probe-windows 512 --oracle-shards 4 --tag smoke
    # the real thing, ~50 min
    modal run --detach -m rhm.conditional_revision.rule_family.gate1::gate1 --tag g1
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
app = modal.App("rhm-rule-family-gate1", image=image)

CKPT_DIR = "rule_family"


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


def n_open_profile(s, L):
    """n_open[t] = how many levels' constituents are still unclosed after position t.

    t = T-1 closes every level (0); t = T/2-1 closes all but the root (1). Section 7's
    read-position sweep is exactly this variable: d3 reads 0.880 where an unresolved
    ancestor still depends on the position and 0.217 where nothing does, at identical
    distance-since-close.
    """
    T = s ** L
    t = np.arange(T)
    out = np.zeros(T, dtype=np.int64)
    for ell in range(L):
        out += ((t + 1) % (s ** (L - ell)) != 0).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# the exact mixture oracle, sharded over CPU workers (O(R) BP dominates wall time)
# ---------------------------------------------------------------------------

@app.function(cpu=4.0, memory=16384, timeout=7200)
def oracle_shard(args):
    """Exact mixture profiles for windows [i0, i1) of a deterministic window set.

    Each shard regenerates the FULL window set from the same seed and then computes only
    its slice, so the windows are bit-identical across shards and to the GPU function
    without shipping them anywhere.
    """
    import numpy as np
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    from rhm.conditional_revision.rule_family.gate_minus1 import DESIGNS
    from rhm.conditional_revision.rule_family.oracle_mixture import (
        family_predictive, mixture_profiles)

    (design, v, s, L, m, family_seed, k_seqs, n_windows, eval_seed, i0, i1,
     sub) = args
    R, dl, nF, mode = DESIGNS[design]
    T = s ** L
    fam, _ = make_family(v, s, L, m, R=R, differ_levels=dl, n_differ_features=nF,
                         seed=family_seed, mode=mode)
    seqs, rule_ids, _ = generate_windows(fam, n_windows, k_seqs, seed=eval_seed)
    keep = ("rule_rev", "nll_mix", "nll_true", "H_rule", "icl_gap", "H_tot_mix",
            "w_true")
    acc = {k: [] for k in keep}
    for a in range(i0, i1, sub):
        b = min(i1, a + sub)
        sq = seqs[a:b]
        post = family_predictive(fam, sq.reshape(-1, T), chunk=512)
        post = post.reshape(R, b - a, k_seqs, T, v)
        prof = mixture_profiles(post, sq, rule_ids[a:b])
        for k in keep:
            acc[k].append(prof[k].astype(np.float32))
        del post, prof
        print(f"  oracle {b - i0}/{i1 - i0}", flush=True)
    return {k: np.concatenate(vs) for k, vs in acc.items()}


# ---------------------------------------------------------------------------

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=65536)
def gate1(
    design: str = "d2_R64_nF2",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, family_seed: int = 0,
    k_seqs: int = 8,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    deep_block: str = "post_block6",
    base_steps: int = 36000, ckpt_at: int = 36000, ckpt_phase: str = "",
    seed: int = 42,
    # evaluation
    n_windows: int = 3200, eval_seed: int = 20250810,
    n_probe_windows: int = 6144, probe_seed: int = 31337,
    rule_probe_steps: int = 4000, rule_probe_lr: float = 3e-3,
    dir_steps: int = 1200, dir_lr: float = 3e-2, dir_cap: int = 12000,
    n_bins: int = 12, oracle_shards: int = 8, oracle_sub: int = 100,
    n_boot: int = 200,
    arms: str = "family,floor", tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    from rhm.conditional_revision.rule_family.gate_minus1 import DESIGNS
    from rhm.conditional_revision.gates_ab import (
        _auc, _stratified_auc, _strata, _partial_r2, _partial_r2_rank)

    device = "cuda"
    L, T = depth, s ** depth
    G = k_seqs * T
    key = tb_key(v, s, L, m)
    R, dl, nF, mode = DESIGNS[design]
    d_diff = dl[0]
    span = s ** (L - d_diff)

    print("=" * 86)
    print(f"RULE FAMILY -- GATE 1   design={design}  R={R}  differ rules[{d_diff}]  "
          f"ckpt @{ckpt_at}{' ' + ckpt_phase if ckpt_phase else ' aligned'}")
    print(f"  {n_windows} eval windows x {G} tokens; probe trained on "
          f"{n_probe_windows} separate windows")
    print("=" * 86, flush=True)

    fam, fam_meta = make_family(v, s, L, m, R=R, differ_levels=dl,
                                n_differ_features=nF, seed=family_seed, mode=mode)
    diff_feats = np.asarray(fam_meta["differ_features"][d_diff])
    print(f"  differing features: {diff_feats.tolist()} of {v}", flush=True)

    # ---------------- windows ----------------
    seqs, rule_ids, lf = generate_windows(fam, n_windows, k_seqs, seed=eval_seed)
    pseqs, prule_ids, _ = generate_windows(fam, n_probe_windows, k_seqs, seed=probe_seed)

    nopen = n_open_profile(s, L)                      # (T,)
    nopen_g = np.tile(nopen, k_seqs)                  # (G,)
    pos_in_seq = np.tile(np.arange(T), k_seqs)
    depth_g = np.repeat(np.arange(k_seqs), T)
    anc_of_pos = np.arange(T) // span
    anc_feat = lf[d_diff][:, :, anc_of_pos].reshape(n_windows, G)
    is_diff = np.isin(anc_feat, diff_feats)           # (n, G)

    # ---------------- exact oracle, sharded ----------------
    bounds = np.linspace(0, n_windows, oracle_shards + 1).astype(int)
    jobs = [(design, v, s, L, m, family_seed, k_seqs, n_windows, eval_seed,
             int(bounds[i]), int(bounds[i + 1]), oracle_sub)
            for i in range(oracle_shards) if bounds[i + 1] > bounds[i]]
    print(f"  launching {len(jobs)} oracle shards ...", flush=True)
    parts = list(oracle_shard.map(jobs))
    O = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    rr = O["rule_rev"].astype(np.float64)             # (n, G)
    nll_mix = O["nll_mix"].astype(np.float64)
    nll_true = O["nll_true"].astype(np.float64)
    print(f"  oracle done. mean rule_rev {rr.mean():.5f}; "
          f"sum/window {rr.sum(1).mean():.3f} of ln R = {np.log(R):.3f}", flush=True)
    for k in range(k_seqs):
        sl = slice(k * T, (k + 1) * T)
        p = rr[:, sl] > 1e-12
        print(f"    depth {k}: sum {rr[:, sl].sum(1).mean():.4f}  frac>0 {p.mean():.4f}  "
              f"mean|>0 {rr[:, sl][p].mean() if p.any() else 0:.5f}  "
              f"H_rule {O['H_rule'][:, sl].mean():.3f}", flush=True)

    results = {
        "config": {"design": design, "R": R, "d_diff": d_diff, "span": span,
                   "k_seqs": k_seqs, "n_windows": n_windows, "ckpt_at": ckpt_at,
                   "ckpt_phase": ckpt_phase or "aligned", "n_bins": n_bins,
                   "tag": tag, "differ_features": diff_feats.tolist(),
                   "n_probe_windows": n_probe_windows},
        "oracle": {
            "mean_rule_rev": float(rr.mean()),
            "sum_per_window": float(rr.sum(1).mean()), "lnR": float(np.log(R)),
            "by_depth": [{
                "depth": k,
                "sum": float(rr[:, k * T:(k + 1) * T].sum(1).mean()),
                "frac_pos": float((rr[:, k * T:(k + 1) * T] > 1e-12).mean()),
                "mean_pos": float(rr[:, k * T:(k + 1) * T][
                    rr[:, k * T:(k + 1) * T] > 1e-12].mean())
                if (rr[:, k * T:(k + 1) * T] > 1e-12).any() else 0.0,
                "H_rule": float(O["H_rule"][:, k * T:(k + 1) * T].mean()),
                "icl_gap": float(O["icl_gap"][:, k * T:(k + 1) * T].mean()),
                "w_true": float(O["w_true"][:, k * T:(k + 1) * T].mean()),
                "frac_pos_given_diff": float(
                    (rr[:, k * T:(k + 1) * T] > 1e-12)[is_diff[:, k * T:(k + 1) * T]].mean()),
                "frac_pos_given_nodiff": float(
                    (rr[:, k * T:(k + 1) * T] > 1e-12)[~is_diff[:, k * T:(k + 1) * T]].mean()),
            } for k in range(k_seqs)]},
        "arms": {},
    }

    # ---------------- cell selection (identical for every arm) ----------------
    # scored cells are "token g arrived", g = 1 .. G-2 (h[G-1] is never trained; the
    # model is fed x[:, :-1] so h exists for 0..G-2 and M_rule needs g-1 and g).
    g_ok = np.zeros(G, dtype=bool)
    g_ok[1:G - 1] = True
    cells = {}
    for cellset in ("diff", "all"):
        base_mask = (is_diff & g_ok[None, :]) if cellset == "diff" else (
            np.ones_like(is_diff) & g_ok[None, :])
        for k in range(k_seqs):
            msk = base_mask.copy()
            msk[:, :k * T] = False
            msk[:, (k + 1) * T:] = False
            r = rr[msk]
            pos = r > 1e-12
            if pos.sum() < 20:
                cells[(cellset, k)] = None
                continue
            med = np.median(r[pos])
            wi, gi = np.where(msk)
            lab_hi = r > med
            lab_zero = r <= 1e-12
            take = lab_hi | lab_zero
            cells[(cellset, k)] = {
                "w": wi[take], "g": gi[take], "y": lab_hi[take],
                "rr": r[take], "median_pos": float(med),
                "n_hi": int(lab_hi.sum()), "n_zero": int(lab_zero.sum()),
            }
    # union of (window, g) needed for the directional decode: `diff` cells, capped
    dir_sel = {}
    rng0 = np.random.default_rng(seed)
    for k in range(k_seqs):
        c = cells[("diff", k)]
        if c is None:
            continue
        n_c = c["w"].size
        idx = np.arange(n_c) if n_c <= dir_cap else rng0.choice(n_c, dir_cap, False)
        dir_sel[k] = idx
    dw = np.concatenate([cells[("diff", k)]["w"][dir_sel[k]] for k in sorted(dir_sel)])
    dg = np.concatenate([cells[("diff", k)]["g"][dir_sel[k]] for k in sorted(dir_sel)])
    dk = np.concatenate([np.full(dir_sel[k].size, k) for k in sorted(dir_sel)])
    dy = np.concatenate([cells[("diff", k)]["y"][dir_sel[k]] for k in sorted(dir_sel)])
    print(f"  directional-decode cells: {dw.size} over {len(dir_sel)} depths", flush=True)

    # ---------------- per arm ----------------
    for arm in [a for a in arms.split(",") if a]:
        ph = f"{ckpt_phase}_" if ckpt_phase else ""
        ck = (f"{DATA_DIR}/{key}/{CKPT_DIR}/{design}_K{k_seqs}_{ph}{arm}_"
              f"{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}"
              f"{f'_at{ckpt_at}' if ckpt_at else ''}.pt")
        if not os.path.exists(ck):
            print(f"  MISSING {ck} -- skipping {arm}", flush=True)
            continue
        print(f"\n  ---- arm {arm}: {os.path.basename(ck)}", flush=True)
        model = GPT(v, 8 * T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)

        # ---- (a) the rule probe: h -> posterior over R rule sets ----
        # position-agnostic by design (one probe applied at every g, so M_rule differences
        # the SAME readout across a step rather than two different probes).
        def collect_states(sq, want_g):
            X = torch.from_numpy(sq.reshape(sq.shape[0], G))
            out = torch.empty(sq.shape[0], len(want_g), n_embd)
            wg = torch.tensor(want_g, device=device)
            with torch.no_grad():
                for i in range(0, sq.shape[0], 32):
                    _, _, inter = model(X[i:i + 32, :-1].to(device).contiguous(),
                                        return_intermediates=True)
                    out[i:i + 32] = inter[deep_block][:, wg, :].float().cpu()
            return out

        pgrid = list(range(1, G - 1, 4))
        Ptr = collect_states(pseqs, pgrid)
        n_tr = int(0.6 * n_probe_windows)
        n_ca = int(0.8 * n_probe_windows)
        yid = torch.from_numpy(prule_ids).long()
        mu = Ptr[:n_tr].reshape(-1, n_embd).mean(0, keepdim=True)
        sd = Ptr[:n_tr].reshape(-1, n_embd).std(0, keepdim=True) + 1e-6
        clf = nn.Linear(n_embd, R).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=rule_probe_lr, weight_decay=1e-4)
        Xtr = ((Ptr[:n_tr] - mu) / sd).to(device)
        ytr = yid[:n_tr].to(device)
        gsub = torch.Generator().manual_seed(seed + 5)
        for it in range(rule_probe_steps):
            sel = torch.randint(0, n_tr, (256,), generator=gsub).to(device)
            pos = torch.randint(0, len(pgrid), (256,), generator=gsub).to(device)
            loss = F.cross_entropy(clf(Xtr[sel, pos]), ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        del Xtr
        # temperature on a calibration split (uncalibrated KLs are dominated by
        # overconfidence; the parent's gates fit a temperature for the same reason)
        Xca = ((Ptr[n_tr:n_ca] - mu) / sd).to(device)
        yca = yid[n_tr:n_ca].to(device)
        with torch.no_grad():
            lg = clf(Xca)                                    # (nc, P, R)
        logt = torch.zeros(1, device=device, requires_grad=True)
        ot = torch.optim.Adam([logt], lr=0.05)
        yrep = yca[:, None].expand(-1, lg.shape[1]).reshape(-1)
        lgf = lg.reshape(-1, R)
        for _ in range(300):
            l = F.cross_entropy(lgf / torch.exp(logt), yrep)
            ot.zero_grad(); l.backward(); ot.step()
        tau = float(torch.exp(logt).item())
        with torch.no_grad():
            Xte = ((Ptr[n_ca:] - mu) / sd).to(device)
            hit = (clf(Xte).argmax(-1) == yid[n_ca:].to(device)[:, None]
                   ).float().mean(0).cpu().numpy()
        gseq = np.array(pgrid) // T
        acc_by_depth = [float(hit[gseq == k].mean()) if (gseq == k).any() else float("nan")
                        for k in range(k_seqs)]
        print(f"    rule probe: tau={tau:.3f}  acc by depth "
              + " ".join(f"{x:.4f}" for x in acc_by_depth)
              + f"   (chance {1 / R:.4f})", flush=True)
        del Ptr, Xca, Xte, lg, lgf
        torch.cuda.empty_cache()

        # ---- (b) eval pass: M_rule and friends at every position, h at scored cells ----
        Xev = torch.from_numpy(seqs.reshape(n_windows, G))
        M = np.zeros((n_windows, G), dtype=np.float32)       # index by arriving token g
        Mpm = np.zeros((n_windows, G), dtype=np.float32)
        Mraw = np.zeros((n_windows, G), dtype=np.float32)
        Ht = np.zeros((n_windows, G), dtype=np.float32)
        Ht1 = np.zeros((n_windows, G), dtype=np.float32)
        mnll = np.zeros((n_windows, G), dtype=np.float32)
        Hb = np.zeros((dw.size, n_embd), dtype=np.float32)
        Ha = np.zeros((dw.size, n_embd), dtype=np.float32)
        mu_d, sd_d = mu.to(device), sd.to(device)
        with torch.no_grad():
            for i in range(0, n_windows, 32):
                j = min(n_windows, i + 32)
                xb = Xev[i:j].to(device)
                logits, _, inter = model(xb[:, :-1].contiguous(),
                                         return_intermediates=True)
                h = inter[deep_block].float()                # (b, G-1, D)
                lg = clf((h - mu_d) / sd_d)                  # (b, G-1, R)
                q = F.log_softmax(lg / tau, dim=-1)
                qr = F.log_softmax(lg, dim=-1)
                # cell g: before = h[g-1], after = h[g]
                qa, qb_ = q[:, 1:], q[:, :-1]                 # (b, G-2, R) for g=1..G-2
                pa = qa.exp()
                ra, rb = qr[:, 1:], qr[:, :-1]
                M[i:j, 1:G - 1] = (pa * (qa - qb_)).sum(-1).cpu().numpy()
                Mraw[i:j, 1:G - 1] = (ra.exp() * (ra - rb)).sum(-1).cpu().numpy()
                am = qa.argmax(-1, keepdim=True)
                Mpm[i:j, 1:G - 1] = (-torch.gather(qb_, 2, am)[..., 0]).cpu().numpy()
                Ht[i:j, 1:G - 1] = (-(qb_.exp() * qb_).sum(-1)).cpu().numpy()
                Ht1[i:j, 1:G - 1] = (-(pa * qa).sum(-1)).cpu().numpy()
                nll = F.cross_entropy(logits.reshape(-1, v),
                                      xb[:, 1:].reshape(-1), reduction="none"
                                      ).reshape(j - i, G - 1)
                mnll[i:j, 1:] = nll.cpu().numpy()
                inb = (dw >= i) & (dw < j)
                if inb.any():
                    ii = np.where(inb)[0]
                    Ha[ii] = h[dw[ii] - i, dg[ii]].cpu().numpy()
                    Hb[ii] = h[dw[ii] - i, dg[ii] - 1].cpu().numpy()
                del h, lg, q, qr, inter, logits
        torch.cuda.empty_cache()
        print(f"    eval pass done. mean M_rule {M[:, 1:G-1].mean():.5f} "
              f"(oracle mean rule_rev {rr[:, 1:G-1].mean():.5f})", flush=True)

        # ---- (c) scoring ----
        def dir_auc(k, which):
            """Supervised linear decode of the family label from the state (a2a idiom).

            `swap` permutes the INPUT vectors within (n_open x surprisal) strata -- the
            language sibling's decisive control, since label permutation degrades exactly
            when the signal is strong.

            Scored BOTH raw and stratified by the same (n_open x exact mixture surprisal)
            strata as every other column. The smoke run made the reason concrete: raw, the
            swap guard read 0.91 rather than 0.50, because permuting inside a stratum
            leaves the BETWEEN-stratum structure intact and the stratum predicts the label
            here. Only the stratified number is comparable to the rest of the table.
            """
            nanpair = {"raw": float("nan"), "matched": float("nan")}
            sel = dk == k
            if sel.sum() < 200:
                return nanpair
            Xs = (Ha if which == "after" else Hb)[sel]
            ys = dy[sel].astype(np.int64)
            if ys.sum() < 30 or (~dy[sel]).sum() < 30:
                return nanpair
            wsel = dw[sel]
            st_all = (nopen_g[dg[sel]].astype(np.int64) * 10000
                      + _strata(nll_mix[dw[sel], dg[sel]], n_bins))
            if which == "swap":
                Xs = Xs.copy()
                rg = np.random.default_rng(seed + 77 + k)
                for u in np.unique(st_all):
                    w_ = np.where(st_all == u)[0]
                    Xs[w_] = Xs[rg.permutation(w_)]
            cut = np.quantile(np.unique(wsel), 0.7)
            tr, te = wsel <= cut, wsel > cut
            if te.sum() < 100 or tr.sum() < 100 or ys[te].sum() < 10:
                return nanpair
            Xt = torch.from_numpy(Xs).to(device)
            mu2 = Xt[tr].mean(0, keepdim=True); sd2 = Xt[tr].std(0, keepdim=True) + 1e-6
            Xt = (Xt - mu2) / sd2
            yt = torch.from_numpy(ys).to(device).float()
            lin = nn.Linear(n_embd, 1).to(device)
            o2 = torch.optim.Adam(lin.parameters(), lr=dir_lr, weight_decay=1e-3)
            itr = torch.where(torch.from_numpy(tr).to(device))[0]
            g2 = torch.Generator().manual_seed(seed + k)
            for _ in range(dir_steps):
                b = itr[torch.randint(0, itr.numel(), (256,), generator=g2).to(device)]
                l = F.binary_cross_entropy_with_logits(lin(Xt[b])[:, 0], yt[b])
                o2.zero_grad(); l.backward(); o2.step()
            with torch.no_grad():
                sc = lin(Xt)[:, 0].cpu().numpy()
            a_m, _ = _stratified_auc(sc[te], ys[te].astype(bool), st_all[te])
            return {"raw": _auc(sc[te], ys[te].astype(bool)), "matched": a_m}

        arm_rows = {}
        for cellset in ("diff", "all"):
            for posclass in ("live", "bdry", "pooled"):
                for k in range(k_seqs):
                    c = cells[(cellset, k)]
                    if c is None:
                        continue
                    w_, g_, y_ = c["w"], c["g"], c["y"]
                    no = nopen_g[g_]
                    pm = (no >= 2) if posclass == "live" else (
                        no <= 1) if posclass == "bdry" else np.ones_like(no, bool)
                    if pm.sum() < 200 or y_[pm].sum() < 20 or (~y_[pm]).sum() < 20:
                        continue
                    w_, g_, y_ = w_[pm], g_[pm], y_[pm]
                    rrc = c["rr"][pm]
                    sc = {
                        "M_rule": M[w_, g_].astype(np.float64),
                        "M_rule_raw": Mraw[w_, g_].astype(np.float64),
                        "M_pointmass": Mpm[w_, g_].astype(np.float64),
                        "negH_t": -Ht[w_, g_].astype(np.float64),
                        "negH_t1": -Ht1[w_, g_].astype(np.float64),
                        "dH": (Ht[w_, g_] - Ht1[w_, g_]).astype(np.float64),
                        "nll_mix": nll_mix[w_, g_],
                        "nll_true": nll_true[w_, g_],
                        "model_nll": mnll[w_, g_].astype(np.float64),
                    }
                    rgs = np.random.default_rng(seed + 991 + k)
                    st_pos = nopen_g[g_].astype(np.int64)
                    st_bin = _strata(sc["nll_mix"], n_bins)
                    st_abs = pos_in_seq[g_].astype(np.int64)
                    strata = {
                        "raw": np.zeros_like(st_pos),
                        "pos": st_pos,
                        "pos_x_nll": st_pos * 10000 + st_bin,
                        "abspos_x_nll": st_abs * 10000 + st_bin,
                        "pos_x_modelnll": st_pos * 10000 + _strata(sc["model_nll"], n_bins),
                    }
                    # guards that need a stratum to shuffle inside
                    for nm, stk in (("M_shuffled", "pos_x_nll"),):
                        v_ = sc["M_rule"].copy()
                        for u in np.unique(strata[stk]):
                            ix = np.where(strata[stk] == u)[0]
                            v_[ix] = v_[rgs.permutation(ix)]
                        sc[nm] = v_
                    row = {"n_hi": int(y_.sum()), "n_zero": int((~y_).sum()),
                           "mean_rr_hi": float(rrc[y_].mean()),
                           "median_pos_thresh": c["median_pos"]}
                    for stname, st in strata.items():
                        row[stname] = {}
                        for nm, val in sc.items():
                            a, _ = _stratified_auc(val, y_, st)
                            row[stname][nm] = a
                    # window bootstrap on the primary column
                    if posclass == "pooled" and cellset in ("diff", "all"):
                        prim = strata["pos_x_nll"]
                        uw = np.unique(w_)
                        rb = np.random.default_rng(seed + 4242 + k)
                        bs = []
                        for _ in range(n_boot):
                            pick = rb.choice(uw, uw.size, True)
                            idx = np.concatenate([np.where(w_ == u)[0] for u in pick])
                            if y_[idx].sum() < 10 or (~y_[idx]).sum() < 10:
                                continue
                            a, _ = _stratified_auc(sc["M_rule"][idx], y_[idx], prim[idx])
                            if np.isfinite(a):
                                bs.append(a)
                        row["M_rule_boot_sd"] = float(np.std(bs)) if len(bs) > 5 else None
                    # graded: does M_rule track rule_rev MAGNITUDE among positive cells?
                    pmask = rrc > 1e-12
                    if pmask.sum() > 300:
                        row["graded"] = {
                            "partial_r2_M_rr_given_nllmix": _partial_r2(
                                sc["M_rule"][pmask], rrc[pmask], sc["nll_mix"][pmask]),
                            "partial_r2_rank": _partial_r2_rank(
                                sc["M_rule"][pmask], rrc[pmask], sc["nll_mix"][pmask]),
                            "partial_r2_nll_given_rr": _partial_r2(
                                sc["M_rule"][pmask], sc["nll_mix"][pmask], rrc[pmask]),
                            "null_shuffled": _partial_r2(
                                rgs.permutation(sc["M_rule"][pmask]), rrc[pmask],
                                sc["nll_mix"][pmask]),
                            "n": int(pmask.sum())}
                    if cellset == "diff" and posclass == "pooled":
                        row["h_after_dir"] = dir_auc(k, "after")
                        row["h_before_dir"] = dir_auc(k, "before")
                        row["h_swap_dir"] = dir_auc(k, "swap")
                    arm_rows[f"{cellset}|{posclass}|d{k}"] = row

        results["arms"][arm] = {
            "ckpt": os.path.basename(ck), "tau": tau,
            "rule_probe_acc_by_depth": acc_by_depth,
            "mean_M_rule": float(M[:, 1:G - 1].mean()),
            "rows": arm_rows,
        }

        # ---- print the primary table ----
        print(f"\n    PRIMARY  cellset=diff  posclass=pooled  "
              f"(stratified by n_open x exact mixture surprisal)")
        print(f"    {'d':>2}{'n_hi':>7}{'n_zer':>7}{'rr_hi':>8}"
              f"{'M_rule':>9}{'±boot':>7}{'Mpm':>8}{'negH_t':>8}{'nllmix':>8}"
              f"{'mnll':>8}{'shuf':>7}{'h_aft':>7}{'h_bef':>7}{'h_swp':>7}")
        for k in range(k_seqs):
            r_ = arm_rows.get(f"diff|pooled|d{k}")
            if r_ is None:
                print(f"    {k:>2}   (no cells)")
                continue
            p = r_["pos_x_nll"]
            bsd = r_.get("M_rule_boot_sd")
            print(f"    {k:>2}{r_['n_hi']:>7}{r_['n_zero']:>7}{r_['mean_rr_hi']:>8.4f}"
                  f"{p['M_rule']:>9.4f}{(bsd if bsd else float('nan')):>7.3f}"
                  f"{p['M_pointmass']:>8.4f}{p['negH_t']:>8.4f}{p['nll_mix']:>8.4f}"
                  f"{p['model_nll']:>8.4f}{p['M_shuffled']:>7.4f}"
                  f"{r_.get('h_after_dir', {}).get('matched', float('nan')):>7.3f}"
                  f"{r_.get('h_before_dir', {}).get('matched', float('nan')):>7.3f}"
                  f"{r_.get('h_swap_dir', {}).get('matched', float('nan')):>7.3f}",
                  flush=True)
        print(f"\n    PRE-REGISTERED POOLED cellset=all (parse confound NOT removed)")
        print(f"    {'d':>2}{'n_hi':>7}{'n_zer':>7}{'M_rule':>9}{'nllmix':>8}{'shuf':>7}")
        for k in range(k_seqs):
            r_ = arm_rows.get(f"all|pooled|d{k}")
            if r_ is None:
                continue
            p = r_["pos_x_nll"]
            print(f"    {k:>2}{r_['n_hi']:>7}{r_['n_zero']:>7}{p['M_rule']:>9.4f}"
                  f"{p['nll_mix']:>8.4f}{p['M_shuffled']:>7.4f}", flush=True)
        print(f"\n    DIRECTIONAL DIAGNOSTIC (supervised decode of the family label from "
              f"the state itself), raw / matched")
        for k in range(k_seqs):
            r_ = arm_rows.get(f"diff|pooled|d{k}")
            if r_ is None or "h_after_dir" not in r_:
                continue
            print(f"      d{k}: h_after {r_['h_after_dir']['raw']:.3f}/"
                  f"{r_['h_after_dir']['matched']:.3f}   "
                  f"h_before {r_['h_before_dir']['raw']:.3f}/"
                  f"{r_['h_before_dir']['matched']:.3f}   "
                  f"h_swap {r_['h_swap_dir']['raw']:.3f}/"
                  f"{r_['h_swap_dir']['matched']:.3f}", flush=True)
        print(f"\n    GRADED (partial R^2, positive cells only, cellset=diff|pooled)")
        for k in range(k_seqs):
            r_ = arm_rows.get(f"diff|pooled|d{k}", {}).get("graded")
            if r_ is None:
                continue
            print(f"      d{k}: R2(M~rr|nll) {r_['partial_r2_M_rr_given_nllmix']:.4f}  "
                  f"rank {r_['partial_r2_rank']:.4f}  "
                  f"R2(M~nll|rr) {r_['partial_r2_nll_given_rr']:.4f}  "
                  f"null {r_['null_shuffled']:.4f}  n={r_['n']}", flush=True)

        del model, clf
        torch.cuda.empty_cache()

    dd = f"{DATA_DIR}/{key}/{CKPT_DIR}"
    os.makedirs(dd, exist_ok=True)
    out = f"{dd}/gate1_{design}_K{k_seqs}{'_' + tag if tag else ''}_seed{seed}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out}", flush=True)
    return results
