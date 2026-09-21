"""frontier — shared machinery: the endogenous level field and the per-level panel.

The arc's readouts of "where is my frontier" have so far been computed with the ORACLE's
level labels (`flat_oracle.leaf_levels` of the absolute leaf index, which needs the
window's true phase). This node asks what survives when the labels come from the model
itself, and what the panel looks like on a learner that stalls instead of climbing.

THE ENDOGENOUS LEVEL FIELD. altitude Q2 A showed the model's per-position entropy profile
carries the period `s^k` of the deepest absorbed level: under observer k the predictive
returns to the depth-k marginal at every depth-k boundary. Q2 A estimated each period's
offset INDEPENDENTLY. Here the estimates are built BOTTOM-UP and kept nested, which is
both what a learner climbing the grammar would do and what makes the estimates into a
level FIELD:

    r_1 = argmax over {0, 1}                       of the period-2  score
    r_{k+1} = argmax over {r_k, r_k + s^k}         of the period-s^{k+1} score
    level_hat(j) = leaf_levels((j - r_L) mod s^L)

so a single estimated boundary column `r_L` induces the whole s-adic level field, and
level `l` is labelled correctly exactly when `r_L` is right mod `s^l`. Two scores:

    "template"  the altitude Q2 A detector: correlate the window's detrended profile
                against the model-independent template E[H(p_k) | leaf residue].
                Uses the ladder's shape but not the window's phase.
    "meanprof"  fully model-free: of the two candidate offsets, take the one whose
                columns carry the higher mean profile value (a depth-k boundary is where
                the predictive returns to the coarse marginal, so entropy is higher).

THE PANEL, per level `l`, on a population with a per-position statistic:

    H_q      the uncertainty the model states                      (expected uncertainty)
    NLL      the realised loss                                     (endogenous)
    CE       the exact expected loss under p_L                     (oracle, where defined)
    excess   NLL - H_q  (endogenous) and CE - H_q (oracle)         (unexpected uncertainty)
    KL_pL_q  the oracle residual, the ground truth for "which rung is being absorbed"

Trends (-Delta of each across checkpoints) are formed in the reduction, not here.
"""

import numpy as np

EPS = 1e-300


def _ignore(path):
    """Mount the Python sources this node actually imports, and nothing else.

    `striatum/task.py`'s reason: another agent writing a launch log into a sibling node's
    `results/` tree makes the image build race and fail ("was modified during build
    process"). The same race fires on a sibling's *source* when another agent is editing
    it, so the mount is narrowed to the sibling nodes this one imports: under
    `logit_reading/` only `frontier/` (plus the top-level modules: `flat_oracle`,
    `grammar`, ...), and under `practice/` only `reread/`."""
    sp = str(path).replace("\\", "/")
    if "/results/" in sp or sp.endswith("/results") or sp.endswith(".log"):
        return True
    if not sp.endswith(".py"):
        return True
    for parent, keep in (("/logit_reading/", "frontier/"), ("/practice/", "reread/")):
        if parent in sp:
            tail = sp.split(parent, 1)[1]
            if "/" in tail and not tail.startswith(keep):
                return True
    return False


def log_softmax(z):
    zm = z - z.max(-1, keepdims=True)
    return zm - np.log(np.exp(zm).sum(-1, keepdims=True))


def ent(p):
    return -(p * np.log(np.clip(p, EPS, None))).sum(-1)


def xent(p, logq):
    return -(p * logq).sum(-1)


def kl(p, logq):
    return xent(p, logq) - ent(p)


# --------------------------------------------------------------------------- #
# the period detector, nested
# --------------------------------------------------------------------------- #

def detrend(X):
    """Subtract the across-window mean at each window index (frontier.py's convention)."""
    return X - X.mean(0, keepdims=True)


def template_scores(Xd, tmpl, P, j_lo=0):
    """(n, P) alignment of each window's profile with the leaf-residue template `tmpl`
    placed so that the depth-k boundary sits at column r. Identical scoring to
    `altitude/frontier.py::template_detector`, which returns this argmax."""
    n, T = Xd.shape
    j = np.arange(T)
    use = j >= j_lo
    Xu = Xd[:, use] - Xd[:, use].mean(1, keepdims=True)
    sc = np.empty((n, P))
    for r in range(P):
        t = tmpl[(j[use] - r) % P]
        sc[:, r] = Xu @ (t - t.mean())
    return sc


def offset_means(Xd, P, j_lo=0):
    """(n, P) mean of the profile over columns j == r (mod P). Model-free; its argmax is
    `altitude/frontier.py::offset_profiles`'s `r_hat`."""
    n, T = Xd.shape
    j = np.arange(T)
    use = j >= j_lo
    prof = np.zeros((n, P))
    for r in range(P):
        m = use & (j % P == r)
        prof[:, r] = Xd[:, m].mean(1) if m.sum() else np.nan
    return prof


def leaf_templates(H_k, leaf, L=6, s=2):
    """tmpl[k][c] = E[H(p_k) | target leaf == c (mod s^k)], the model-independent template
    altitude Q2 A uses. H_k: (n, T, L+1); leaf: (n, T) absolute target leaf index."""
    out = {}
    for k in range(1, L + 1):
        P = s ** k
        r = leaf % P
        out[k] = np.array([float(H_k[..., k][r == c].mean()) for c in range(P)])
    return out


def nested_phase(Xd, tmpls, L=6, s=2, j_lo=0, mode="template"):
    """Bottom-up nested offset estimates. Returns {k: (n,) r_hat_k} for k = 1..L, with
    r_hat_{k+1} == r_hat_k (mod s^k) by construction."""
    n = Xd.shape[0]
    r = np.zeros(n, dtype=np.int64)
    out = {}
    for k in range(1, L + 1):
        P = s ** k
        sc = (template_scores(Xd, tmpls[k], P, j_lo) if mode == "template"
              else offset_means(Xd, P, j_lo))
        cand = np.stack([r % P, (r + P // s) % P], 1)              # (n, 2)
        pick = np.take_along_axis(sc, cand, 1).argmax(1)
        r = np.take_along_axis(cand, pick[:, None], 1)[:, 0]
        out[k] = r.copy()
    return out


def endo_levels(r_hat_L, T, L=6, s=2):
    """The level field a single estimated boundary column induces."""
    from rhm.logit_reading.flat_oracle import leaf_levels
    j = np.arange(T)[None, :]
    return leaf_levels((j - r_hat_L[:, None]) % (s ** L), s, L)


def oracle_levels(phase, T, L=6, s=2):
    """The oracle's leaf level of the TARGET at window column j (window token j+1)."""
    from rhm.logit_reading.flat_oracle import leaf_levels
    return leaf_levels((phase[:, None] + 1 + np.arange(T)[None, :]) % (s ** L), s, L)


def true_boundary_column(phase, P):
    """The column carrying leaf residue 0 mod P: j == -(phase + 1) (mod P)."""
    return (-phase - 1) % P


# --------------------------------------------------------------------------- #
# the panel
# --------------------------------------------------------------------------- #

def level_panel(lev, L, H_q, NLL, CE=None, KL=None, H_p=None, min_n=50):
    """Per-level means of the panel quantities. `lev` (n, T) integer level field."""
    out = {}
    for l in range(L + 1):
        m = lev == l
        if m.sum() < min_n:
            continue
        d = {"n": int(m.sum()), "frac": float(m.mean()),
             "H_q": float(H_q[m].mean()), "NLL": float(NLL[m].mean()),
             "excess_realised": float((NLL - H_q)[m].mean())}
        if CE is not None:
            d["CE"] = float(CE[m].mean())
            d["excess_oracle"] = float((CE - H_q)[m].mean())
        if KL is not None:
            d["KL_pL_q"] = float(KL[m].mean())
            d["KL_pL_q_share"] = float(KL[m].sum() / KL.sum())
        if H_p is not None:
            d["H_p"] = float(H_p[m].mean())
        # standard error of the realised excess, so "zero" has a scale
        e = (NLL - H_q)[m]
        d["se_excess_realised"] = float(e.std() / np.sqrt(e.size))
        out[int(l)] = d
    return out


def confusion(lev_true, lev_hat, L):
    """(L+1, L+1) row-normalised: P(lev_hat = c | lev_true = r)."""
    C = np.zeros((L + 1, L + 1))
    for r in range(L + 1):
        m = lev_true == r
        if m.sum() == 0:
            continue
        C[r] = np.bincount(lev_hat[m], minlength=L + 1) / m.sum()
    return C.round(4).tolist()


def detector_block(Xd, tmpls, phase, L=6, s=2, j_lo=0):
    """Both nested detectors + the unconstrained per-k accuracies altitude Q2 A reports."""
    out = {}
    for mode in ("template", "meanprof"):
        rh = nested_phase(Xd, tmpls, L, s, j_lo, mode=mode)
        acc = {k: float((rh[k] == true_boundary_column(phase, s ** k)).mean())
               for k in range(1, L + 1)}
        out[mode] = {"acc_nested": acc, "r_hat_L": rh[L]}
    # unconstrained, per k independently (the Q2 A statistic)
    unc_t, unc_m = {}, {}
    for k in range(1, L + 1):
        P = s ** k
        tr = true_boundary_column(phase, P)
        unc_t[k] = float((template_scores(Xd, tmpls[k], P, j_lo).argmax(1) == tr).mean())
        unc_m[k] = float((offset_means(Xd, P, j_lo).argmax(1) == tr).mean())
    out["acc_unconstrained_template"] = unc_t
    out["acc_unconstrained_meanprof"] = unc_m
    out["chance"] = {k: 1.0 / s ** k for k in range(1, L + 1)}
    return out
