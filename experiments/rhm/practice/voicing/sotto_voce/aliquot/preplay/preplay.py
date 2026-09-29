"""[preplay] DOES THE SHAPED PLANT'S PROJECTION PRICE A CANDIDATE ENTRY THE WAY THE WORLD'S
AUDITION DOES? — offline, on banked soundboard state, no loop.

`ideas/calibration_and_violation_are_one_object.md` section 12.4: a candidate that exists only
as a table entry, a "could", has NO PRICE until the world model is put into the state the
candidate implies and the projection is read THERE. Firing is forced by the readout's type;
pricing untaken candidates offline is preplay.

The practice arc has always priced a candidate table by an AUDITION: apply the candidate macro
to held-out instances of the era's damage cell and ask the world how many it repaired
(`soundboard.py::audition_macro`, and `census`'s commit-then-extend loop at 13346-13427, which
auditions `base + {candidate}` against `base` one candidate at a time). That audition is the
oracle read the endogenous grader is meant to replace. `aliquot` / `soundboard` established
that a prediction-trained plant's projection does not read the verdict, and that an outcome
head whose gradient reaches the trunk beside its infill loss makes the verdict linearly
readable on the LEARNER'S OWN configurations (held-out AUC 0.819 / 0.861 at seeds 0 / 2).
NOTHING in the lineage has ever scored a projection against an AUDITION VERDICT, and every arm
of the last round graded candidate SPELLINGS WITHIN A LEVEL, never a candidate ENTRY.

THIS NODE FIRES CANDIDATE ENTRIES THROUGH THE SHAPED PLANT'S OWN EXECUTOR, READS THE
PROJECTION OVER THE FIRED CONFIGURATIONS, AND ASKS WHETHER IT RANKS THE CANDIDATES AS THE
WORLD'S AUDITION DOES. Offline, on banked state.

Per arm of record and per level l in 2..5, at that level's own era cell:

  base            three matched-size random subsets of the TRUE table at the arm's committed
                  `n_entries` for that level (`soundboard`'s own `rand_k` idiom, 12855-12868)
  candidates      TRUE entries not in the base (the world should mostly price these up) and a
                  matched number of WRONG entries, child-row pairs whose level-1 expansion is
                  not in the true table (the world should price these at zero or worse)
  delta form      audition `base + {cand}` against `base`         (the loop's own question)
  single form     the candidate ALONE as a one-row table          (more signal per candidate)
  the fire        `MC.apply_any`, the arm's SHAPED core as the executor, on a FRESH pool of
                  `n_score = 256` instances of the cell
  the reads       the world's grade (the instrument, beside every read); the arm's own BANKED
                  shaped projection (`vo_heads.pt::proj`, byte-faithful, gated); the same
                  readout form refit here on the same bank rows through the shaped core,
                  through `overtone`'s FROZEN unshaped core of the same seed, and through a
                  NEVER-TRAINED twin; and the executor's own free signal, the max-sum DP's top
                  score over the fired span.

THE FIDELITY GATE is what licenses any of it. It comes in two parts, because the exact form
the brief asked for turned out not to exist:
  F-2   (`fidelity_gate`) the re-implementation of `VoProjBank._features` / `_design` /
        `_score` / `predict` (soundboard.py 6642-6730, 6867) against THE CLASS ITSELF,
        elementwise, on the arm's own banked held-out rows through its own banked core with
        its own banked weights. Closed at max|dp| = 0.000e+00 on both arms of record.
  F-2b  the re-read scored against the arm's BANKED held-out AUC. This CANNOT match to the
        third decimal and the reason is structural: the loop trains the plant AFTER the
        readout's refresh inside the same cycle (`soundboard.py` 12227 against 12553), so the
        core in `vo_heads.pt` has had one more plant update than the core the last logged
        `hold_auc` was measured through. Realised gap 0.0035 / 0.0012, against the readout's
        own per-refit |drift| p90 of 0.024 / 0.011. The band is 2x that p90, floored at 0.02.

Run from experiments/ (MODAL_PROFILE=chromatic):
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::gates
    modal run rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::sweep --smoke 1 \\
        --out-tag pp_smoke --arms s0_sv
    modal run --detach rhm/practice/voicing/sotto_voce/aliquot/preplay/preplay.py::sweep \\
        --out-tag pp1
"""

import hashlib
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume

app = modal.App("rhm-practice-preplay", image=image)

REMOTE = "rhm_practice_preplay"                 # where this node writes
BANK_SB = "rhm_practice_soundboard"             # the shaped plants and their readouts
BANK_OV = "rhm_practice_voicing"                # overtone's frozen, unshaped plants

# The arms. `sb_sv_yk` at both seeds are the ARMS OF RECORD (verdict-shaped, both losses);
# `sb_so_yk` (verdict, infill term OFF -- the run collapsed as an executor while its readout
# still read 0.80) and `sb_yd_yk` (yield-shaped) are the extras.
ARMS = {
    "s0_sv": {"tag": "sb_s1", "arm": "sb_sv_yk", "ov": "s0", "record": True},
    "s2_sv": {"tag": "sb_s2", "arm": "sb_sv_yk", "ov": "s2", "record": True},
    "s0_so": {"tag": "sb_s1", "arm": "sb_so_yk", "ov": "s0", "record": False},
    "s2_so": {"tag": "sb_s2", "arm": "sb_so_yk", "ov": "s2", "record": False},
    "s0_yd": {"tag": "sb_s1", "arm": "sb_yd_yk", "ov": "s0", "record": False},
    "s2_yd": {"tag": "sb_s2", "arm": "sb_yd_yk", "ov": "s2", "record": False},
}

# `overtone`'s banked frozen plants of the same lineage at the same seeds (`duplex`'s SEEDS).
OVERTONE = {
    "s0": {"tag": "ov_s0b", "arm": "ovt_comp_pr_sh"},
    "s2": {"tag": "ov_s2", "arm": "ovt_comp_pr_dis"},
}

# The never-trained twin. `build_rand_trunk`'s IN-LOOP seed (`vo_pj_rand_seed = 20260918`), so
# the floor here is the same object `aliquot`'s random-twin arm read, not `overtone`'s offline
# 20260915 one. Stated rather than inherited.
TWIN_SEED = 20260918

# The readout's own literals, `VoProjBank.__init__`'s defaults as the arms ran them.
PJ_RIDGES = (1.0, 32.0, 1024.0)
PJ_ITERS = 25
PJ_FIT_CAP = 8192
PJ_HOLD = 0.1
PJ_BOOT = 0.8
PJ_REFRESH_CAP = 2048          # `VoOutcomeBank.refresh`'s own cap -- the gate needs it exactly

# The fire. `n_score` is the loop's own audition size; the draws are the `rand_k` idiom's three.
N_SCORE = 256
N_BASE_DRAWS = 3
K_DELTA = 48                   # per class (true / wrong) per base, delta form
K_SINGLE = 96                  # per class, single-entry form
LEVELS = (2, 3, 4, 5)

# Fresh pool seeds. The run's own audition pools used `cfg.seed + 820_000 + ...` (the merge
# ledger) and `+ 900_000 + ...` (commit-then-extend); this family is disjoint from both.
POOL_SEED_BASE = 5_100_000
# The candidate/base draws' own numpy stream, disjoint from the pool's.
DRAW_SEED_BASE = 6_200_000
# The readout refit's subsample stream (shared across the three trunks, so the fits are matched).
FIT_SEED_BASE = 7_300_000


# --------------------------------------------------------------------------------------- #
# the estimator and the metric  (`soundboard::vo_pj_irls`, `vo_auc`, re-implemented)
# --------------------------------------------------------------------------------------- #

def pj_irls(Z, y, lam, pen, iters=PJ_ITERS):
    """`soundboard.py::vo_pj_irls` verbatim: float32 Gram, float64 accumulate and solve, the
    per-column penalty with a FREE intercept, the same clip and the same early stop."""
    import torch
    p_ = int(Z.shape[1])
    w = torch.zeros(p_, dtype=torch.float64, device=Z.device)
    Zf = Z.float()
    yf = y.float().to(Z.device)
    pn = pen.double().to(Z.device)
    D = torch.diag(pn) * float(lam)
    for _ in range(int(iters)):
        eta = (Zf @ w.float()).clamp(-30.0, 30.0)
        pr = torch.sigmoid(eta)
        g = (Zf.T @ (pr - yf)).double() + float(lam) * pn * w
        sw = pr * (1.0 - pr) + 1e-6
        H = ((Zf * sw[:, None]).T @ Zf).double() + D
        try:
            stp = torch.linalg.solve(H, g)
        except Exception:
            break
        w = w - stp
        if float(stp.abs().max()) < 1e-9:
            break
    return w


def vo_auc(scores, labels):
    """`soundboard.py::vo_auc` verbatim: the rank identity with ties averaged, None where one
    class is absent."""
    y = np.asarray(labels, np.float64)
    x = np.asarray(scores, np.float64)
    n1 = float((y > 0.5).sum())
    n0 = float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(a, b):
    """Rank correlation with ties averaged. None when either side is constant."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 4:
        return None

    def rk(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(1, len(x) + 1)
        xs = x[o]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j + 2) / 2.0
            i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return None
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


# --------------------------------------------------------------------------------------- #
# THE PROJECTION'S READ — `VoProjBank._features` / `_design` / `_score`, re-implemented
# --------------------------------------------------------------------------------------- #
#
# Forty lines, copied deliberately rather than imported, so this node does not pull a 1.2 MB
# module with a `modal.App` at its top into every container. Gate F-2 is what makes the copy
# admissible: the banked `(w, mu, sd)` read through THESE lines must reproduce the arm's banked
# held-out AUC to the third decimal, on the arm's own banked rows, before anything is fired.

def pj_features(core, x, blk, spn, s, nb, tdim, device, mask=True, chunk=4096):
    """`VoProjBank._features` (soundboard.py 6642-6700): `SN.trunk`'s per-block hiddens with the
    FIRST BLOCK OUTSIDE THE SPAN masked to -1 when `mask` is on, pooled two ways -- over all
    blocks and over the span -- and concatenated."""
    import torch
    import rhm.practice.native.span.span_net as SN
    n = int(x.shape[0])
    blk = _col(blk, n)
    spn = _col(spn, n)
    unk = (blk < 0)
    b0_all = torch.where(unk, torch.zeros_like(blk), blk)
    sw_all = torch.where(unk, torch.full_like(spn, nb - 1), spn.clamp(min=1)).clamp(max=nb - 1)
    ar = torch.arange(nb)
    out = []
    with torch.no_grad():
        for a in range(0, n, int(chunk)):
            b_ = min(a + int(chunk), n)
            xb = x[a:b_].to(device).long().clone()
            b0 = b0_all[a:b_]
            sw = sw_all[a:b_]
            inside = ((ar[None, :] >= b0[:, None]) & (ar[None, :] < (b0 + sw)[:, None]))
            if mask:
                mb = (~inside).long().argmax(1)
                pos = (mb[:, None] * s + torch.arange(s)[None, :]).to(device)
                xb = xb.scatter(1, pos, torch.full_like(pos, -1))
            pooled, _ = SN.trunk(core, xb)
            wgt = inside.float().to(device)
            wgt = wgt / wgt.sum(1, keepdim=True).clamp(min=1.0)
            out.append(torch.cat([pooled.mean(1),
                                  (pooled * wgt[:, :, None]).sum(1)], 1).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 2 * tdim)


def _col(val, n):
    import torch
    if isinstance(val, (int, np.integer)):
        return torch.full((n,), int(val), dtype=torch.long)
    t = val if hasattr(val, "shape") else torch.as_tensor(val)
    t = t.cpu().long()
    return t if int(t.numel()) == n else t.reshape(-1)[:n]


def pj_design(F, r, v, mu=None, sd=None):
    """`VoProjBank._design` at `root = "inter"`: [std(features), onehot(root),
    onehot(root) x std(features), 1]."""
    import torch
    F = F.float()
    n = int(F.shape[0])
    if mu is None:
        mu = F.mean(0)
        sd = F.std(0).clamp(min=1e-6)
    Z = (F - mu) / sd
    oh = torch.zeros(n, v)
    oh.scatter_(1, r.cpu().long().clamp(0, v - 1)[:, None], 1.0)
    cols = [Z, oh, (Z[:, :, None] * oh[:, None, :]).reshape(n, -1), torch.ones(n, 1)]
    return torch.cat(cols, 1), mu, sd


def pj_score(Z, w, device):
    """`VoProjBank._score`: both operands on the bank's device, float32 matmul, sigmoid."""
    import torch
    with torch.no_grad():
        return torch.sigmoid(Z.float().to(device) @ w.float().to(device)).cpu()


def pj_predict(core, x, r, blk, spn, w, mu, sd, s, nb, tdim, v, device, mask=True):
    F = pj_features(core, x, blk, spn, s, nb, tdim, device, mask=mask)
    Z, _, _ = pj_design(F, r, v, mu=mu, sd=sd)
    return pj_score(Z, w, device)


def pj_pen(p_):
    import torch
    pen = torch.ones(p_, dtype=torch.float64)
    pen[-1] = 0.0
    return pen


def pj_fit(core, bank, idx_tr, idx_va, s, nb, tdim, v, device, log=None, name=""):
    """`VoProjBank.train`'s solve, on a GIVEN train/validation index pair so the three trunks
    are fit on identical rows: standardise on the pooled slice, select the ridge on the
    validation AUC, return `(w, mu, sd, lam, val_auc)`."""
    import torch
    idx = np.concatenate([idx_tr, idx_va])
    ii = torch.from_numpy(idx)
    y = bank["y"][ii]
    F = pj_features(core, bank["x"][ii], bank["blk"][ii], bank["spn"][ii],
                    s, nb, tdim, device)
    Zall, mu, sd = pj_design(F, bank["r"][ii], v)
    pen = pj_pen(int(Zall.shape[1]))
    ntr = int(idx_tr.size)
    Zt, Zv = Zall[:ntr].to(device), Zall[ntr:].to(device)
    yt, yv = y[:ntr], y[ntr:].numpy()
    best = None
    for lam in PJ_RIDGES:
        w = pj_irls(Zt, yt, lam, pen)
        au = vo_auc(pj_score(Zv, w, device).numpy(), yv)
        if au is not None and (best is None or au > best[0]):
            best = (au, lam, w)
    if best is None:
        return None
    if log:
        log(f"    fit[{name}]  lam {best[1]:>6.0f}  val_auc {best[0]:.4f}  "
            f"n_tr {ntr}  n_va {idx_va.size}  nfeat {int(Zall.shape[1])}")
    return {"w": best[2], "mu": mu, "sd": sd, "lam": float(best[1]),
            "val_auc": float(best[0])}


# --------------------------------------------------------------------------------------- #
# THE FIRE — one recording pass of `MC.macro_features`' DP, gated against `MC.apply_any`
# --------------------------------------------------------------------------------------- #

def fire_rec(core, x, move, canon, s):
    """`MC.macro_features` + `MC.apply_any`'s macro branch, with the DP's OWN SCORES kept.

    Returns `(xf, dp)` where `dp` carries the executor's free signal over the fired span:
      top    the winning entry's summed score, per block of the span
      lse    top - logsumexp over entries (the DP's own confidence in its pick)
      marg   top - runner-up, per block of the span
    Gate F-1 asserts `xf` is bit-identical to `MC.apply_any`'s, so the recording copy cannot
    move a fired state -- `soundboard`'s E-7 idiom for the audition path.
    """
    import torch
    blk0, span = move["blk0"], move["span"]
    table = move["table"]
    batch = x.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=x.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=x.device)[None, :]).reshape(-1)
    pos = pos[None, :].expand(batch, -1).contiguous()
    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = core.block_logits(obs)
    cur = logits[:, blk0:blk0 + span, :]
    assert move["level"] >= 2, "preplay fires macros only"
    for child in move["chain"]:
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, cur.shape[-1])
        total = None
        for i in range(s):
            picked = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
            total = picked if total is None else total + picked
        cur = total
    flatsc = cur.reshape(batch, -1)                     # (B, n_entries)
    best = flatsc.argmax(-1)
    top = flatsc.gather(1, best[:, None]).squeeze(1)
    lse = top - torch.logsumexp(flatsc, dim=-1)
    if flatsc.shape[1] > 1:
        two = flatsc.topk(2, dim=-1).values
        marg = two[:, 0] - two[:, 1]
    else:
        marg = torch.zeros_like(top)
    flat = move["flat"]
    feats = flat[best].view(batch, span)
    tup = canon[feats]
    new = x.clone()
    new.scatter_(1, pos, tup.reshape(batch, -1))
    return new, {"top": (top / float(span)).detach().cpu().numpy().astype("float32"),
                 "lse": lse.detach().cpu().numpy().astype("float32"),
                 "marg": (marg / float(span)).detach().cpu().numpy().astype("float32"),
                 "entry": best.detach().cpu().numpy().astype("int32")}


# --------------------------------------------------------------------------------------- #
# the candidate sets
# --------------------------------------------------------------------------------------- #

def wrong_rows(n_lower, true_flat_set, lower_flat, s, k, rng, tries=400):
    """`k` child rows (s indices into `lower`) whose level-1 expansion is NOT in the true
    table's flat set -- the entries the world should price at zero or worse. Deduplicated on
    the child row AND on the flat, so a wrong candidate is one object."""
    out, seen_child, seen_flat = [], set(), set()
    for _ in range(int(tries) * max(1, k)):
        if len(out) >= k:
            break
        row = tuple(int(z) for z in rng.integers(0, n_lower, size=s))
        if row in seen_child:
            continue
        fl = tuple(int(z) for j in row for z in lower_flat[j])
        if fl in true_flat_set or fl in seen_flat:
            continue
        seen_child.add(row)
        seen_flat.add(fl)
        out.append(list(row))
    return out


# --------------------------------------------------------------------------------------- #
# the per-arm job
# --------------------------------------------------------------------------------------- #

@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=8192)
def preplay_arm(arm_key, out_tag, smoke=False, levels="", n_score=0, k_delta=0, k_single=0,
                draws=0):
    import resource
    import torch
    import rhm.practice.native.span.span_net as SN
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances

    t00 = time.time()
    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    volume.reload()
    spec = ARMS[arm_key]
    root_sb = os.path.join(DATA_DIR, BANK_SB, spec["tag"], spec["arm"])
    ovs = OVERTONE[spec["ov"]]
    root_ov = os.path.join(DATA_DIR, BANK_OV, ovs["tag"], ovs["arm"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want_levels = tuple(int(z) for z in levels.split(",") if z) or LEVELS
    NS = int(n_score) or N_SCORE
    KD = int(k_delta) or K_DELTA
    KS = int(k_single) or K_SINGLE
    ND = int(draws) or N_BASE_DRAWS
    if smoke:
        NS, KD, KS, ND = 64, 4, 6, 2
    log("=" * 100)
    log(f"[preplay] arm {arm_key}: {spec['tag']}/{spec['arm']}   overtone {ovs['tag']}/"
        f"{ovs['arm']}   device {device}   smoke={bool(smoke)}")
    log(f"  levels {want_levels}  n_score {NS}  k_delta {KD}  k_single {KS}  draws {ND}")
    log("=" * 100)

    # ---- the banked state ------------------------------------------------------------- #
    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    res_j = json.load(open(os.path.join(root_sb, "results.json")))
    cfgr, cfgh = res_j["config"], blob["cfg"]
    v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
    m, maxl = int(cfgr["m"]), int(cfgr["max_macro_level"])
    dim = int(cfgr["state_dim"])
    length, nb = s ** depth, (s ** depth) // s
    tdim = dim
    assert str(cfgh.get("vo_om_mode")) == "proj", cfgh.get("vo_om_mode")
    assert str(cfgh.get("vo_pj_trunk")) == "live", cfgh.get("vo_pj_trunk")
    assert bool(cfgh.get("vo_pj_mask")) and str(cfgh.get("vo_pj_root")) == "inter"
    log(f"  cfg v{v} s{s} L{depth} m{m} rule_seed {cfgr['rule_seed']} seed {cfgr['seed']} "
        f"dim {dim} maxl {maxl}  n_blocks {nb}")

    rules = generate_rules_distinct(v, s, depth, m, seed=int(cfgr["rule_seed"]))
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    rules_t = None

    def mk_core():
        return GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                     root_conditioned=False).to(device)

    shaped = mk_core()
    shaped.load_state_dict(blob["core"])
    shaped.eval()
    ovblob = torch.load(os.path.join(root_ov, "vo_heads.pt"), map_location="cpu",
                        weights_only=True)
    frozen = mk_core()
    frozen.load_state_dict(ovblob["core"])
    frozen.eval()
    st_ = torch.get_rng_state()
    torch.manual_seed(TWIN_SEED)
    twin = mk_core()
    torch.set_rng_state(st_)
    twin.eval()

    def fingerprint(c):
        with torch.no_grad():
            return float(sum(float(p_.detach().double().abs().sum()) for p_ in c.parameters()))
    fps = {k_: fingerprint(c_) for k_, c_ in
           (("shaped", shaped), ("frozen", frozen), ("twin", twin))}
    log(f"  trunk fingerprints: shaped {fps['shaped']:.6f}  frozen {fps['frozen']:.6f}  "
        f"twin {fps['twin']:.6f}")
    assert abs(fps["shaped"] - fps["frozen"]) > 1e-6, "shaped and frozen cores are identical"

    pj = blob["proj"]
    assert pj is not None, "this arm banked no fitted readout"
    w_b, mu_b, sd_b = pj["w"], pj["mu"], pj["sd"]
    log(f"  banked readout: lam {pj['lam']}  which {pj['which']}  mask {pj['mask']}  "
        f"root {pj['root']}  nfeat {int(w_b.numel())}")
    assert bool(pj["mask"]) and str(pj["root"]) == "inter" and str(pj["which"]) == "live"

    # ---- GATE F-2: the byte-faithful read ---------------------------------------------- #
    z = np.load(os.path.join(root_sb, "vo_bank.npz"))
    bank = {"x": torch.from_numpy(z["x"].astype(np.int64)),
            "r": torch.from_numpy(z["r"].astype(np.int64)),
            "y": torch.from_numpy(z["y"].astype(np.float32)),
            "blk": torch.from_numpy(z["blk"].astype(np.int64)),
            "spn": torch.from_numpy(z["spn"].astype(np.int64)),
            "h": z["h"].astype(np.float64), "u0": z["u0"].astype(np.float64)}
    n_bank = int(bank["x"].shape[0])
    hold_idx = np.nonzero(bank["h"] < PJ_HOLD)[0][-PJ_REFRESH_CAP:]
    hi = torch.from_numpy(hold_idx)
    p_hold = pj_predict(shaped, bank["x"][hi], bank["r"][hi], bank["blk"][hi], bank["spn"][hi],
                        w_b, mu_b, sd_b, s, nb, tdim, v, device)
    auc_here = vo_auc(p_hold.numpy(), bank["y"][hi].numpy())
    oms = [q for q in res_j["log"]["vo_om"]
           if isinstance(q, dict) and q.get("hold_auc") is not None]
    auc_banked = float(oms[-1]["hold_auc"])
    drifts = np.array([abs(q["hold_auc_drift"]) for q in oms
                       if q.get("hold_auc_drift") is not None][-60:], dtype=float)
    band = float(max(0.02, 2.0 * np.quantile(drifts, 0.9))) if drifts.size else 0.05
    gate = {"F-2b:hold_auc_here": auc_here, "F-2b:hold_auc_banked": auc_banked,
            "F-2b:delta": abs(auc_here - auc_banked), "F-2b:band": band,
            "F-2b:drift_p90": (float(np.quantile(drifts, 0.9)) if drifts.size else None),
            "F-2b:n_hold": int(hold_idx.size), "F-2b:n_bank": n_bank}
    log(f"  [F-2b] banked hold AUC {auc_banked:.6f}   re-read here {auc_here:.6f}   "
        f"|d| {abs(auc_here - auc_banked):.4f}   band {band:.4f}   n {hold_idx.size}")
    # THE BANKED SCALAR IS NOT AN EXACT TARGET, and the round found out why. The loop trains
    # the plant AFTER the readout's `refresh` inside the same cycle (`soundboard.py` 12227
    # against 12553), so the core in `vo_heads.pt` has had one more plant update than the core
    # the last logged `hold_auc` was measured through. The re-read can therefore only be asked
    # to sit inside the readout's OWN per-refit drift, which is what this band is. The EXACT
    # fidelity gate is `fidelity_gate` below: the re-implementation against `VoProjBank.predict`
    # itself, elementwise, on these same rows through this same core.
    assert abs(auc_here - auc_banked) < band, (
        f"F-2b FAILED: the re-implemented read gives {auc_here:.6f} against the banked "
        f"{auc_banked:.6f}, outside the readout's own drift band {band:.4f}")

    # ---- the two control readouts, matched rows ---------------------------------------- #
    trainable = (bank["h"] >= PJ_HOLD)
    tr_all = np.nonzero(trainable & (bank["u0"] < PJ_BOOT))[0]
    va_all = np.nonzero(trainable & (bank["u0"] >= PJ_BOOT))[0]
    frng = np.random.default_rng(FIT_SEED_BASE + int(cfgr["seed"]))
    idx_tr = (np.sort(frng.choice(tr_all, size=PJ_FIT_CAP, replace=False))
              if tr_all.size > PJ_FIT_CAP else tr_all)
    vcap = max(512, PJ_FIT_CAP // 4)
    idx_va = (np.sort(frng.choice(va_all, size=vcap, replace=False))
              if va_all.size > vcap else va_all)
    log(f"  refit slice: n_tr {idx_tr.size}  n_va {idx_va.size}  "
        f"base rate {float(bank['y'][torch.from_numpy(idx_tr)].mean()):.4f}")
    fits = {}
    for nm, core_ in (("shaped_refit", shaped), ("frozen", frozen), ("twin", twin)):
        fits[nm] = pj_fit(core_, bank, idx_tr, idx_va, s, nb, tdim, v, device,
                          log=log, name=nm)
    # each refit readout's own held-out AUC on the same held-out rows, for the record
    hold_read = {"banked": auc_here}
    for nm, core_ in (("shaped_refit", shaped), ("frozen", frozen), ("twin", twin)):
        f_ = fits[nm]
        p_ = pj_predict(core_, bank["x"][hi], bank["r"][hi], bank["blk"][hi], bank["spn"][hi],
                        f_["w"], f_["mu"], f_["sd"], s, nb, tdim, v, device)
        hold_read[nm] = vo_auc(p_.numpy(), bank["y"][hi].numpy())
    log("  held-out AUC on the arm's own experience: " +
        "  ".join(f"{k_}={0.0 if q is None else q:.4f}" for k_, q in hold_read.items()))

    READS = ("banked", "shaped_refit", "frozen", "twin")

    def read_all(xf, roots_t, blk0, span):
        out = {}
        for nm in READS:
            if nm == "banked":
                core_, w_, mu_, sd_ = shaped, w_b, mu_b, sd_b
            else:
                core_ = {"shaped_refit": shaped, "frozen": frozen, "twin": twin}[nm]
                f_ = fits[nm]
                w_, mu_, sd_ = f_["w"], f_["mu"], f_["sd"]
            out[nm] = pj_predict(core_, xf, roots_t, blk0, span, w_, mu_, sd_,
                                 s, nb, tdim, v, device).numpy().astype("float32")
        return out

    # ---- the ladder's own cells and the arm's committed sizes --------------------------- #
    eras = {int(e["level"]): e for e in res_j["eras"]}
    commits = {}
    for e_ in res_j["events"]:
        if e_.get("kind") == "commit":
            commits[int(e_["level"])] = int(e_["n_entries"])
    aud_last = res_j["log"]["aud"][-1]
    truth = MC.true_tables(rules, depth, s, v, m, maxl)
    log(f"  eras {[eras[k_]['name'] for k_ in sorted(eras)]}   commits {commits}")

    out = {"arm": arm_key, "spec": spec, "gate": gate, "hold_read": hold_read,
           "fits": {k_: {"lam": q["lam"], "val_auc": q["val_auc"]} for k_, q in fits.items()},
           "fingerprints": fps, "n_bank": n_bank, "cells": {},
           "cfg": {"n_score": NS, "k_delta": KD, "k_single": KS, "draws": ND,
                   "levels": list(want_levels), "seed": int(cfgr["seed"]),
                   "rule_seed": int(cfgr["rule_seed"]), "twin_seed": TWIN_SEED}}
    per_inst = {}
    n_aud_total = 0
    gate_f1, gate_f1_done = [], set()

    for ell in want_levels:
        if ell not in eras:
            log(f"  L{ell}: no era at this level, skipped")
            continue
        era = eras[ell]
        node = int(era["node"])
        span = s ** (ell - 1)
        blk0 = node * span
        truth_l = truth[ell]
        lower = truth_l["lower"]
        n_true_rows = int(truth_l["child"].shape[0])
        true_flat = {tuple(int(q) for q in r_) for r_ in truth_l["flat"]}
        n_base = commits.get(ell)
        base_src = "commit"
        if n_base is None:
            n_base = int(aud_last.get(str(ell), {}).get("n_entries") or 0)
            base_src = "live_table_last_cycle"
        n_base = max(1, min(int(n_base), n_true_rows))
        t_l0 = time.time()
        log("")
        log(f"  --- L{ell} at cell {era['name']} (node {node}, blk0 {blk0}, span {span}) "
            f"n_true_rows {n_true_rows} n_true_flat {len(true_flat)} "
            f"base {n_base} ({base_src})")

        # the pool: fresh instances of THIS level's own damage cell
        pool_seed = int(cfgr["seed"]) + POOL_SEED_BASE + 1000 * ell
        roots_np, x_np = context_instances(
            rules, {"name": era["name"], "level": ell, "nodes": [node]},
            NS, s, depth, v, m, seed=pool_seed)
        x0 = torch.from_numpy(x_np).to(device)
        roots_t = torch.from_numpy(roots_np).to(device).long()
        succ0, _ = grade(x_np, roots_np, rules, s)
        log(f"      pool seed {pool_seed}  n {NS}  bootstrap success {succ0.mean():.4f}")

        rng = np.random.default_rng(DRAW_SEED_BASE + int(cfgr["seed"]) * 97 + ell)

        # --- the bases: `soundboard`'s `rand_k` idiom ---------------------------------- #
        bases = []
        for d in range(ND):
            keep = np.sort(rng.permutation(n_true_rows)[:n_base])
            bases.append(keep)

        # --- the candidate pools ------------------------------------------------------- #
        k_true_single = min(KS, n_true_rows)
        cand_true_single = np.sort(rng.permutation(n_true_rows)[:k_true_single]).tolist()
        n_lower = int(lower["flat"].shape[0])
        cand_wrong = wrong_rows(n_lower, true_flat, lower["flat"], s,
                                max(KS, KD), rng)
        log(f"      candidates: single true {len(cand_true_single)}  wrong pool "
            f"{len(cand_wrong)}  (n_lower {n_lower})")

        cache = {}          # audition cache, keyed by the table's row tuple-set

        def audition(child_rows, tag):
            """One fire + one world grade + every read, on the SAME fired configurations."""
            nonlocal n_aud_total
            arr = np.ascontiguousarray(np.asarray(child_rows, np.int64).reshape(-1, s))
            key = (int(arr.shape[0]),
                   hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest())
            if key in cache:
                return cache[key]
            tb = MC.make_table(ell, arr, lower, s)
            mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
            xf, dp = fire_rec(shaped, x0, mv, canon, s)
            if ell not in gate_f1_done:
                gate_f1_done.add(ell)
                xa = MC.apply_any(shaped, x0, mv, rules_t, canon, depth, v, m, s)
                same = bool((xa == xf).all())
                gate_f1.append({"level": ell, "tag": tag, "identical": same,
                                "n_entries": int(tb["child"].shape[0])})
                assert same, "F-1 FAILED: the recording DP does not reproduce `MC.apply_any`"
            succ, dres = grade(xf.cpu().numpy(), roots_np, rules, s)
            rd = read_all(xf, roots_t, blk0, span)
            n_aud_total += 1
            rec = {"e": float(1.0 - succ.mean()), "succ": succ.astype("float32"),
                   "dres": float(dres.mean()), "reads": rd, "dp": dp,
                   "n_entries": int(tb["child"].shape[0])}
            cache[key] = rec
            return rec

        def summarize(rec):
            d = {"e": rec["e"], "dres": rec["dres"], "succ": float(rec["succ"].mean()),
                 "n_entries": rec["n_entries"]}
            for nm in READS:
                d[f"p_{nm}"] = float(rec["reads"][nm].mean())
            for k_ in ("top", "lse", "marg"):
                d[f"dp_{k_}"] = float(rec["dp"][k_].mean())
            return d

        cell = {"era": era["name"], "node": node, "blk0": blk0, "span": span,
                "n_true_rows": n_true_rows, "n_true_flat": len(true_flat),
                "n_base": n_base, "base_src": base_src, "pool_seed": pool_seed,
                "boot_success": float(succ0.mean()), "n_lower": n_lower,
                "delta": [], "single": []}

        # --- reference auditions -------------------------------------------------------- #
        base_recs = [audition(truth_l["child"][k_], f"base{d}") for d, k_ in enumerate(bases)]
        cell["bases"] = [summarize(q) for q in base_recs]
        # the ceiling: the whole true table
        ceil = audition(truth_l["child"], "true_full")
        cell["ceiling"] = summarize(ceil)
        log(f"      base e {[round(q['e'], 4) for q in cell['bases']]}   "
            f"true-table ceiling e {cell['ceiling']['e']:.4f}   "
            f"(loop's own last-cycle aud: cand "
            f"{aud_last.get(str(ell), {}).get('cand')}, true "
            f"{aud_last.get(str(ell), {}).get('true')})")

        inst_rows = {"delta": [], "single": []}

        # --- the DELTA form: `base + {cand}` against `base` ----------------------------- #
        for d, keep in enumerate(bases):
            in_base = set(int(q) for q in keep)
            base_child = [list(map(int, r_)) for r_ in truth_l["child"][keep]]
            br = base_recs[d]
            avail_true = [i_ for i_ in range(n_true_rows) if i_ not in in_base]
            take_true = (np.sort(rng.permutation(len(avail_true))[:KD]).tolist()
                         if len(avail_true) > KD else list(range(len(avail_true))))
            pick_true = [avail_true[i_] for i_ in take_true]
            pick_wrong = cand_wrong[:KD]
            for cls, rows in (("true", [list(map(int, truth_l["child"][i_]))
                                        for i_ in pick_true]),
                              ("wrong", [list(r_) for r_ in pick_wrong])):
                for ci, cr in enumerate(rows):
                    rec = audition(base_child + [cr], f"d{d}:{cls}{ci}")
                    row = {"draw": d, "cls": cls, "cand": cr,
                           "w_delta": float(br["e"] - rec["e"]),
                           "e_with": rec["e"], "e_base": br["e"],
                           "succ_with": float(rec["succ"].mean()),
                           "n_entries": rec["n_entries"]}
                    for nm in READS:
                        row[f"r_{nm}"] = float(rec["reads"][nm].mean()
                                               - br["reads"][nm].mean())
                        row[f"p_{nm}"] = float(rec["reads"][nm].mean())
                    for k_ in ("top", "lse", "marg"):
                        row[f"dp_{k_}"] = float(rec["dp"][k_].mean() - br["dp"][k_].mean())
                    row["n_moved"] = int((rec["dp"]["entry"] ==
                                          rec["n_entries"] - 1).sum())
                    cell["delta"].append(row)
                    inst_rows["delta"].append(
                        (rec["succ"], {nm: rec["reads"][nm] for nm in READS},
                         rec["dp"]["lse"]))

        # --- the SINGLE-ENTRY form: the candidate alone as a one-row table -------------- #
        for cls, rows in (("true", [list(map(int, truth_l["child"][i_]))
                                    for i_ in cand_true_single]),
                          ("wrong", [list(r_) for r_ in cand_wrong[:KS]])):
            for ci, cr in enumerate(rows):
                rec = audition([cr], f"s:{cls}{ci}")
                row = {"cls": cls, "cand": cr, "w_succ": float(rec["succ"].mean()),
                       "e": rec["e"], "dres": rec["dres"]}
                for nm in READS:
                    row[f"p_{nm}"] = float(rec["reads"][nm].mean())
                for k_ in ("top", "lse", "marg"):
                    row[f"dp_{k_}"] = float(rec["dp"][k_].mean())
                cell["single"].append(row)
                inst_rows["single"].append(
                    (rec["succ"], {nm: rec["reads"][nm] for nm in READS},
                     rec["dp"]["lse"]))

        # --- per-instance transfer: the read against the world's per-instance success ---- #
        cell["instance"] = {}
        for form in ("delta", "single"):
            if not inst_rows[form]:
                continue
            ys = np.concatenate([q[0] for q in inst_rows[form]])
            blk_ = {"n": int(ys.size), "base": float(ys.mean())}
            for nm in READS:
                ps = np.concatenate([q[1][nm] for q in inst_rows[form]])
                blk_[f"auc_{nm}"] = vo_auc(ps, ys)
                blk_[f"pbar_{nm}"] = float(ps.mean())
            ls = np.concatenate([q[2] for q in inst_rows[form]])
            blk_["auc_dp_lse"] = vo_auc(ls, ys)
            cell["instance"][form] = blk_
            per_inst[f"L{ell}|{form}|y"] = ys.astype("float32")
            for nm in READS:
                per_inst[f"L{ell}|{form}|{nm}"] = np.concatenate(
                    [q[1][nm] for q in inst_rows[form]]).astype("float32")
            per_inst[f"L{ell}|{form}|dp_lse"] = ls.astype("float32")

        cell["sec"] = float(time.time() - t_l0)
        cell["n_auditions_cum"] = n_aud_total
        out["cells"][str(ell)] = cell
        log(f"      L{ell} done in {cell['sec']:.1f}s   auditions so far {n_aud_total}   "
            f"delta rows {len(cell['delta'])}  single rows {len(cell['single'])}")
        if cell["instance"].get("single"):
            b = cell["instance"]["single"]
            log(f"      per-instance (single): base {b['base']:.4f}  " +
                "  ".join(f"{nm}={0.0 if b.get('auc_' + nm) is None else b['auc_' + nm]:.3f}"
                          for nm in READS))

    out["gate_f1"] = gate_f1
    out["n_auditions"] = n_aud_total
    out["sec"] = float(time.time() - t00)
    out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 1e6
                          if torch.cuda.is_available() else 0.0)
    log("")
    log(f"  done in {out['sec']:.1f}s   auditions {n_aud_total}   "
        f"peak RSS {out['peak_rss_mb']:.0f} MB   peak GPU {out['peak_gpu_mb']:.0f} MB")

    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{arm_key}.json"), "w") as fh:
        json.dump(out, fh, cls=NumpyEncoder, separators=(",", ":"))
    with open(os.path.join(d, f"{arm_key}.log"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    if per_inst:
        np.savez_compressed(os.path.join(d, f"{arm_key}_inst.npz"), **per_inst)
    volume.commit()
    return {"arm": arm_key, "sec": out["sec"], "n_aud": n_aud_total,
            "rss_mb": out["peak_rss_mb"], "gate": gate}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=12600, memory=2048)
def sweep(out_tag="pp1", smoke=False, arms="", levels="", n_score=0, k_delta=0, k_single=0,
          draws=0):
    """CPU coordinator: the arms across separate containers. The arms are independent, so the
    fan-out adds no GPU-hours -- it only stops them queueing behind one another."""
    want = [x for x in arms.split(",") if x] or [k for k, q in ARMS.items() if q["record"]]
    args = [(k, out_tag, bool(smoke), levels, int(n_score), int(k_delta), int(k_single),
             int(draws)) for k in want]
    outs = list(preplay_arm.starmap(args))
    print(json.dumps(outs, indent=2, default=str))
    d = os.path.join(DATA_DIR, REMOTE, out_tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "done.txt"), "w") as fh:
        fh.write(json.dumps({"arms": want, "smoke": bool(smoke), "levels": levels,
                             "n_score": int(n_score), "k_delta": int(k_delta),
                             "k_single": int(k_single), "draws": int(draws)}) + "\n")
    volume.commit()
    return outs


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=8192)
def gates():
    """The gate block on a GPU, run before anything is paid for. Every gate here has been shown
    to fail on a deliberately broken input (`NOTES.md`)."""
    import torch
    import rhm.practice.ratchet.macros as MC
    from rhm.rhm_data import generate_rules_distinct
    import rhm.rhm_generative_planner as GP
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.practice.crystallize.units import grade

    res = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    torch.manual_seed(7)
    core = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    core.eval()
    truth = MC.true_tables(rules, depth, s, v, m, 5)

    # ---- F-1: the recording DP reproduces `MC.apply_any` bit for bit, at every level ---- #
    f1 = {}
    for ell in (2, 3, 4, 5):
        node = {2: 12, 3: 6, 4: 3, 5: 1}[ell]
        rg = np.random.default_rng(11 + ell)
        rows = truth[ell]["child"][np.sort(rg.permutation(truth[ell]["child"].shape[0])[:17])]
        tb = MC.make_table(ell, rows, truth[ell]["lower"], s)
        mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
        g = torch.Generator().manual_seed(11 * ell)
        x = torch.randint(0, v, (48, length), generator=g).to(device)
        xa = MC.apply_any(core, x, mv, None, canon, depth, v, m, s)
        xf, dp = fire_rec(core, x, mv, canon, s)
        f1[f"L{ell}"] = {"identical": bool((xa == xf).all()),
                         "dp_top_mean": float(dp["top"].mean()),
                         "dp_lse_mean": float(dp["lse"].mean())}
    res["F-1:recording_dp_identical"] = f1
    assert all(q["identical"] for q in f1.values()), "F-1 FAILED"

    # ---- F-3: a wrong entry is really wrong, and a true entry is really true ----------- #
    rg = np.random.default_rng(3)
    f3 = {}
    for ell in (2, 3, 4):
        tl = truth[ell]
        tf = {tuple(int(q) for q in r_) for r_ in tl["flat"]}
        wr = wrong_rows(int(tl["lower"]["flat"].shape[0]), tf, tl["lower"]["flat"], s, 24, rg)
        tw = MC.make_table(ell, np.asarray(wr, np.int64), tl["lower"], s)
        gt = MC.grade_table(tw, tl)
        tt = MC.make_table(ell, tl["child"][:24], tl["lower"], s)
        gtt = MC.grade_table(tt, tl)
        f3[f"L{ell}"] = {"wrong_precision": gt["precision"], "n_wrong": len(wr),
                         "true_precision": gtt["precision"]}
        assert gt["precision"] == 0.0, f"F-3 FAILED at L{ell}: a 'wrong' row is on the table"
        assert gtt["precision"] == 1.0, f"F-3 FAILED at L{ell}: a true row is off the table"
    res["F-3:candidate_classes"] = f3

    # ---- F-4: the pool is the cell's, and it is broken there --------------------------- #
    f4 = {}
    for ell, node in ((2, 12), (3, 6), (4, 3), (5, 1)):
        r_, x_ = context_instances(rules, {"name": f"L{ell}n{node}", "level": ell,
                                           "nodes": [node]}, 64, s, depth, v, m,
                                   seed=5_100_000 + ell)
        su, dr = grade(x_, r_, rules, s)
        span = s ** (ell - 1)
        lo, hi = node * span * s, (node * span + span) * s
        f4[f"L{ell}"] = {"boot_success": float(su.mean()), "mean_dres": float(dr.mean()),
                         "span_tokens": [lo, hi]}
        assert float(su.mean()) < 0.5, f"F-4 FAILED at L{ell}: the cell is not broken"
    res["F-4:pool_is_broken_at_the_cell"] = f4

    # ---- F-5: the projection's design has the shape the arms ran -------------------- #
    import torch as T
    F = T.randn(5, 2 * 96)
    Z, mu, sd = pj_design(F, T.tensor([0, 1, 2, 3, 4]), 8)
    res["F-5:design_cols"] = int(Z.shape[1])
    assert int(Z.shape[1]) == 2 * 96 + 8 + 2 * 96 * 8 + 1 == 1737, "F-5 FAILED"

    # ---- F-6: the copied pool helper is TEXT-IDENTICAL to soundboard's ----------------- #
    import inspect
    import rhm.practice.voicing.sotto_voce.aliquot.preplay.pool as PP
    mine = inspect.getsource(PP.context_instances)
    sb_path = os.path.join(os.path.dirname(inspect.getfile(PP)), "..", "soundboard",
                           "soundboard.py")
    txt = open(os.path.normpath(sb_path)).read()
    i = txt.index("def context_instances(rules, ctx, n, s, depth, v, m, seed,")
    j = txt.index("\n\n\n", i)
    theirs = txt[i:j] + "\n"
    res["F-6:pool_text_identical"] = bool(mine.strip() == theirs.strip())
    assert mine.strip() == theirs.strip(), (
        "F-6 FAILED: the copied `context_instances` has drifted from soundboard's")

    print(json.dumps(res, indent=2, default=str))
    return res


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def fidelity_gate(arms="s0_sv,s2_sv"):
    """GATE F-2: THE RE-IMPLEMENTED READ AGAINST `VoProjBank.predict` ITSELF, ELEMENTWISE.

    The forty lines this node copies out of `soundboard.py` (`_features` 6642-6700, `_design`
    6700-6721, `_score` 6730, `predict` 6867) are gated here against the class they were copied
    from -- the real `VoProjBank`, constructed with the arm's own cfg, its banked trunk, its
    banked `(w, mu, sd)` and its banked buffer -- on the arm's own held-out rows. Nothing fired
    is read through the copy until this is closed.

    This is the gate the banked-scalar check (F-2b) cannot be: the loop trains the plant AFTER
    the readout's refresh in the same cycle, so the banked `hold_auc` was measured through a
    core one update older than the one in `vo_heads.pt`, and a scalar match to the third decimal
    is not available in principle.
    """
    import torch
    from rhm.practice.voicing.sotto_voce.aliquot.soundboard import soundboard as SB
    import rhm.rhm_generative_planner as GP

    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for arm_key in [x for x in arms.split(",") if x]:
        spec = ARMS[arm_key]
        root_sb = os.path.join(DATA_DIR, BANK_SB, spec["tag"], spec["arm"])
        blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                          weights_only=True)
        cfgr = json.load(open(os.path.join(root_sb, "results.json")))["config"]
        v, s, depth = int(cfgr["v"]), int(cfgr["s"]), int(cfgr["depth"])
        dim = int(cfgr["state_dim"])
        length, nb = s ** depth, (s ** depth) // s
        core = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                     root_conditioned=False).to(device)
        core.load_state_dict(blob["core"])
        core.eval()
        z = np.load(os.path.join(root_sb, "vo_bank.npz"))
        x = torch.from_numpy(z["x"].astype(np.int64))
        r = torch.from_numpy(z["r"].astype(np.int64))
        y = torch.from_numpy(z["y"].astype(np.float32))
        blk = torch.from_numpy(z["blk"].astype(np.int64))
        spn = torch.from_numpy(z["spn"].astype(np.int64))
        h = z["h"].astype(np.float64)
        hold_idx = np.nonzero(h < PJ_HOLD)[0][-PJ_REFRESH_CAP:]
        hi = torch.from_numpy(hold_idx)

        # the real object, with the real weights and the real buffer
        pj = SB.VoProjBank(cfgr, v, length, s, device, seed=3, trunk=core, tdim=dim)
        pj.w, pj.mu, pj.sd = blob["proj"]["w"], blob["proj"]["mu"], blob["proj"]["sd"]
        pj.lam = float(blob["proj"]["lam"])
        p_them, _, _ = pj.predict(x[hi], r[hi], blk0=blk[hi], span=spn[hi])
        p_mine = pj_predict(core, x[hi], r[hi], blk[hi], spn[hi],
                            blob["proj"]["w"], blob["proj"]["mu"], blob["proj"]["sd"],
                            s, nb, dim, v, device)
        dmax = float((p_them - p_mine).abs().max())
        a_them = SB.vo_auc(p_them.numpy(), y[hi].numpy())
        a_mine = vo_auc(p_mine.numpy(), y[hi].numpy())
        # and the UNMASKED read, so the mask idiom is gated too
        pm_them, _, _ = pj.predict(x[hi[:256]], r[hi[:256]], blk0=blk[hi[:256]],
                                   span=spn[hi[:256]])
        F_mine = pj_features(core, x[hi[:256]], blk[hi[:256]], spn[hi[:256]], s, nb, dim,
                             device, mask=False)
        F_them = pj._features(x[hi[:256]], r[hi[:256]], blk[hi[:256]], spn[hi[:256]],
                              mask=False)
        fmax = float((F_mine - F_them).abs().max())
        out[arm_key] = {"F-2:max_abs_dp": dmax, "F-2:auc_class": a_them,
                        "F-2:auc_copy": a_mine, "F-2:auc_delta": abs(a_them - a_mine),
                        "F-2:max_abs_dF_unmasked": fmax, "n": int(hold_idx.size)}
        print(f"[F-2] {arm_key}: max|dp| {dmax:.3e}  AUC class {a_them:.6f} copy "
              f"{a_mine:.6f}  max|dF| (unmasked) {fmax:.3e}", flush=True)
        assert dmax < 1e-6, f"F-2 FAILED on {arm_key}: max|dp| = {dmax:.3e}"
        assert abs(a_them - a_mine) < 1e-9, f"F-2 FAILED on {arm_key}: the AUCs differ"
        assert fmax < 1e-6, f"F-2 FAILED on {arm_key}: the unmasked features differ"
    print(json.dumps(out, indent=2, default=str))
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=12288)
def falsify(arm_key="s0_sv"):
    """THE FALSIFICATION HARNESS. A gate is not reported until it has been shown to fail, so
    every gate above is fed a deliberately broken input here and asserted to TRIP. Each entry
    records what was broken and that the gate's own condition then failed."""
    import torch
    import rhm.practice.ratchet.macros as MC
    import rhm.rhm_generative_planner as GP
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.crystallize.units import grade
    from rhm.practice.voicing.sotto_voce.aliquot.preplay.pool import context_instances
    from rhm.rhm_sculpt_planner import _sample_pool

    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res, n_ok = {}, 0
    v, s, depth, m = 8, 2, 6, 2
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=0)
    canon = torch.as_tensor(np.ascontiguousarray(rules[depth - 1][:, 0, :]),
                            dtype=torch.long, device=device)
    torch.manual_seed(7)
    core0 = GP._build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                                  root_conditioned=False).to(device)
    core0.eval()
    truth = MC.true_tables(rules, depth, s, v, m, 4)

    # ---- F-1 broken: the recorder takes the DP's WORST entry instead of its best ------- #
    ell, node = 3, 6
    rg = np.random.default_rng(5)
    rows = truth[ell]["child"][np.sort(rg.permutation(truth[ell]["child"].shape[0])[:21])]
    tb = MC.make_table(ell, rows, truth[ell]["lower"], s)
    mv = MC.to_device(MC.make_macro(ell, node, s, tb), device)
    g = torch.Generator().manual_seed(99)
    xr = torch.randint(0, v, (48, length), generator=g).to(device)
    xa = MC.apply_any(core0, xr, mv, None, canon, depth, v, m, s)

    def fire_rec_broken(core, x, move, canon_, s_):
        blk0, span = move["blk0"], move["span"]
        blocks = torch.arange(blk0, blk0 + span, device=x.device)
        pos = (blocks[:, None] * s_ + torch.arange(s_, device=x.device)[None, :]).reshape(-1)
        pos = pos[None, :].expand(x.shape[0], -1).contiguous()
        obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
        cur = core.block_logits(obs)[:, blk0:blk0 + span, :]
        for child in move["chain"]:
            n_par = cur.shape[1] // s_
            kids = cur.view(x.shape[0], n_par, s_, cur.shape[-1])
            tot = None
            for i in range(s_):
                pk = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
                tot = pk if tot is None else tot + pk
            cur = tot
        worst = cur.reshape(x.shape[0], -1).argmin(-1)          # THE BREAK
        new = x.clone()
        new.scatter_(1, pos, canon_[move["flat"][worst].view(x.shape[0], span)]
                     .reshape(x.shape[0], -1))
        return new
    xb = fire_rec_broken(core0, xr, mv, canon, s)
    res["F-1"] = {"broke": "the recording DP takes argmin instead of argmax",
                  "gate_condition_holds": bool((xa == xb).all()),
                  "n_rows_differing": int((xa != xb).any(1).sum())}
    assert not bool((xa == xb).all()), "F-1 did not trip on a broken recorder"
    n_ok += 1

    # ---- F-3 broken: `wrong_rows` told the true set is EMPTY ------------------------- #
    tl = truth[3]
    wr_bad = wrong_rows(int(tl["lower"]["flat"].shape[0]), set(), tl["lower"]["flat"], s,
                        32, np.random.default_rng(1))
    prec_bad = MC.grade_table(MC.make_table(3, np.asarray(wr_bad, np.int64),
                                            tl["lower"], s), tl)["precision"]
    res["F-3"] = {"broke": "the true-flat set handed to `wrong_rows` is empty",
                  "wrong_precision": prec_bad, "gate_condition_holds": prec_bad == 0.0}
    assert prec_bad > 0.0, "F-3 did not trip when the wrong rows were not filtered"
    n_ok += 1

    # ---- F-4 broken: the pool is the CLEAN derivations, never damaged ---------------- #
    r_c, lv_c = _sample_pool(rules, 64, s, 424242)
    su_c, _ = grade(lv_c, r_c, rules, s)
    res["F-4"] = {"broke": "the pool is clean derivations with no damage applied",
                  "boot_success": float(su_c.mean()),
                  "gate_condition_holds": bool(float(su_c.mean()) < 0.5)}
    assert float(su_c.mean()) >= 0.5, "F-4 did not trip on an undamaged pool"
    n_ok += 1

    # ---- F-5 broken: the ADDITIVE design (the interaction block dropped) ------------- #
    F = torch.randn(5, 2 * 96)
    Z = torch.cat([(F - F.mean(0)) / F.std(0).clamp(min=1e-6),
                   torch.zeros(5, 8), torch.ones(5, 1)], 1)
    res["F-5"] = {"broke": "the root interaction block is dropped from the design",
                  "cols": int(Z.shape[1]), "gate_condition_holds": int(Z.shape[1]) == 1737}
    assert int(Z.shape[1]) != 1737, "F-5 did not trip on the additive design"
    n_ok += 1

    # ---- F-6 broken: one character of the copied pool helper changed ------------------ #
    import inspect
    import rhm.practice.voicing.sotto_voce.aliquot.preplay.pool as PP
    mine_bad = inspect.getsource(PP.context_instances).replace("90_001", "90_002")
    sb_path = os.path.join(os.path.dirname(inspect.getfile(PP)), "..", "soundboard",
                           "soundboard.py")
    txt = open(os.path.normpath(sb_path)).read()
    i = txt.index("def context_instances(rules, ctx, n, s, depth, v, m, seed,")
    theirs = txt[i:txt.index("\n\n\n", i)] + "\n"
    res["F-6"] = {"broke": "the copied helper's pool seed offset changed by one",
                  "gate_condition_holds": mine_bad.strip() == theirs.strip()}
    assert mine_bad.strip() != theirs.strip(), "F-6 did not trip on a one-character edit"
    n_ok += 1

    # ---- F-2 / F-2b broken: read the banked rows through the WRONG trunk -------------- #
    spec = ARMS[arm_key]
    root_sb = os.path.join(DATA_DIR, BANK_SB, spec["tag"], spec["arm"])
    blob = torch.load(os.path.join(root_sb, "vo_heads.pt"), map_location="cpu",
                      weights_only=True)
    cfgr = json.load(open(os.path.join(root_sb, "results.json")))["config"]
    dim = int(cfgr["state_dim"])
    nb = length // s
    shaped = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                   root_conditioned=False).to(device)
    shaped.load_state_dict(blob["core"])
    shaped.eval()
    torch.manual_seed(TWIN_SEED)
    twin = GP._build_generator()(v, length, s, dim, n_head=4, n_layer=2,
                                 root_conditioned=False).to(device)
    twin.eval()
    z = np.load(os.path.join(root_sb, "vo_bank.npz"))
    x = torch.from_numpy(z["x"].astype(np.int64))
    r = torch.from_numpy(z["r"].astype(np.int64))
    y = torch.from_numpy(z["y"].astype(np.float32))
    blk = torch.from_numpy(z["blk"].astype(np.int64))
    spn = torch.from_numpy(z["spn"].astype(np.int64))
    hi = torch.from_numpy(np.nonzero(z["h"].astype(np.float64) < PJ_HOLD)[0][-PJ_REFRESH_CAP:])
    w_, mu_, sd_ = blob["proj"]["w"], blob["proj"]["mu"], blob["proj"]["sd"]
    p_ok = pj_predict(shaped, x[hi], r[hi], blk[hi], spn[hi], w_, mu_, sd_,
                      s, nb, dim, v, device)
    p_bad = pj_predict(twin, x[hi], r[hi], blk[hi], spn[hi], w_, mu_, sd_,
                       s, nb, dim, v, device)
    dmax = float((p_ok - p_bad).abs().max())
    res["F-2"] = {"broke": "the features come from the never-trained twin, not the shaped core",
                  "max_abs_dp": dmax, "gate_condition_holds": bool(dmax < 1e-6)}
    assert dmax >= 1e-6, "F-2 did not trip on the wrong trunk"
    n_ok += 1
    oms = [q for q in json.load(open(os.path.join(root_sb, "results.json")))["log"]["vo_om"]
           if isinstance(q, dict) and q.get("hold_auc") is not None]
    auc_banked = float(oms[-1]["hold_auc"])
    drifts = np.array([abs(q["hold_auc_drift"]) for q in oms
                       if q.get("hold_auc_drift") is not None][-60:], dtype=float)
    band = float(max(0.02, 2.0 * np.quantile(drifts, 0.9)))
    auc_bad = vo_auc(p_bad.numpy(), y[hi].numpy())
    res["F-2b"] = {"broke": "the same wrong trunk, scored against the banked hold AUC",
                   "auc_bad": auc_bad, "auc_banked": auc_banked, "band": band,
                   "gate_condition_holds": bool(abs(auc_bad - auc_banked) < band)}
    assert abs(auc_bad - auc_banked) >= band, "F-2b did not trip on the wrong trunk"
    n_ok += 1

    res["n_gates_falsified"] = n_ok
    print(json.dumps(res, indent=2, default=str))
    return res
