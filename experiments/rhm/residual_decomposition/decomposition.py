"""Metrics for the residual-rank decomposition claimed in beliefs/dimensionality_expansion.md.

The belief asserts a partition of a layer's activations by what the forward
self-model (FM) compresses:

    A  =  FM(earlier_layer)  +  Residual
    R_act = rank(A),  R_comp = rank(FM(.)),  R_res = rank(Residual)
    "Subadditively, R_act ~= R_comp + R_res"

This module measures that triple three ways, so the claim can be adjudicated
rather than assumed:

1. `naive_triple`   -- the triple exactly as specified, using the entropy
                       effective rank the repo already uses everywhere
                       (a2a_forward/analyze.py:892, rhm_residual_rank.py,
                       ratchet/rhm_ratchet.py:517). Reports the subadditivity
                       ratio (R_comp + R_res) / R_act, which the belief says
                       should be ~1.

2. `residual_geometry` -- the substantive content of "R_res is the slice of
                       R_act the self-model hasn't absorbed yet": does the
                       residual actually live in the directions the base model
                       uses? Measured against an explicit chance baseline,
                       because a random residual already sits in A's top-k
                       subspace at rate k/D.

3. `repaired_triple` -- a decomposition that partitions A's *variance* along
                       A's own principal axes, so R_comp + R_res = R_act holds
                       by construction and both legs respond to residual
                       magnitude. This is what the belief's "moving partition"
                       language requires in order to be coherent.

Everything is computed on mean-centered data, matching repo convention.
Pure numpy: no torch, no Modal, so it is unit-testable locally.
"""

import numpy as np

__all__ = [
    "eff_rank_entropy",
    "participation_ratio",
    "naive_triple",
    "residual_geometry",
    "repaired_triple",
    "full_decomposition",
]


# --------------------------------------------------------------------------
# Soft rank measures
# --------------------------------------------------------------------------

def _centered_svdvals(X):
    """Singular values of mean-centered X (N x D), descending."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return np.linalg.svd(Xc, full_matrices=False, compute_uv=False)


def eff_rank_entropy(X=None, S=None, weighting="linear"):
    """exp(spectral entropy) -- the repo's `effective_rank`.

    weighting="linear" reproduces `_compute_effective_rank` exactly
    (p = S / sum(S)); weighting="energy" uses p = S^2 / sum(S^2), the variant
    in a2a_forward/analyze.py. Both are reported because the repo contains both
    and the falsification should not hinge on which one is used.
    """
    if S is None:
        S = _centered_svdvals(X)
    S = np.asarray(S, dtype=np.float64)
    S = S[S > 0]
    if S.size == 0:
        return 0.0
    p = S / S.sum() if weighting == "linear" else S ** 2 / (S ** 2).sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(lam):
    """(sum lam)^2 / sum(lam^2) -- soft dimension count over a variance spectrum.

    The natural measure for the repaired triple: it is defined directly on
    variances, so a spectrum can be split into two variance components and each
    part scored on the same scale.
    """
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.clip(lam, 0.0, None)
    tot = lam.sum()
    if tot <= 0:
        return 0.0
    return float(tot ** 2 / np.sum(lam ** 2))


# --------------------------------------------------------------------------
# 1. The triple exactly as the belief specifies it
# --------------------------------------------------------------------------

def naive_triple(A, P):
    """R_act, R_comp, R_res as literally defined, plus the subadditivity ratio.

    A: (N, D) target activations.  P: (N, D) FM predictions.  Residual = A - P.

    The belief's justification is subadditivity, rank(X+Y) <= rank(X)+rank(Y).
    That is a theorem about *hard* rank; entropy effective rank obeys no such
    bound. Hard rank is also degenerate here (N >> D makes all three exactly D),
    so it is reported only to show that degeneracy explicitly.
    """
    R = A - P
    out = {}
    for tag, X in (("act", A), ("comp", P), ("res", R)):
        S = _centered_svdvals(X)
        out[f"R_{tag}"] = eff_rank_entropy(S=S, weighting="linear")
        out[f"R_{tag}_energy"] = eff_rank_entropy(S=S, weighting="energy")
        out[f"hard_rank_{tag}"] = int(np.sum(S > S[0] * 1e-10)) if S.size else 0
    for suffix in ("", "_energy"):
        num = out[f"R_comp{suffix}"] + out[f"R_res{suffix}"]
        out[f"subadditivity_ratio{suffix}"] = float(num / max(out[f"R_act{suffix}"], 1e-12))
    out["d_model"] = int(A.shape[1])
    return out


# --------------------------------------------------------------------------
# 2. Geometry: is the residual actually inside the directions A uses?
# --------------------------------------------------------------------------

def residual_geometry(A, P, ks=(4, 8, 16, 32, 64), seed=0):
    """Does the residual live in the base model's own principal subspace?

    This is the testable core of "R_res is the slice of R_act the self-model
    hasn't absorbed yet". Every quantity is paired with its chance baseline:
    a random direction already puts k/D of its variance inside any fixed
    k-dimensional subspace, so raw containment numbers are uninterpretable
    on their own.

    Returns
      absorption_spectrum : rho_i = 1 - Var(<R, v_i>) / Var(<A, v_i>) for each
          of A's principal directions v_i. The belief's picture predicts a step
          -- high rho on A's leading directions (absorbed as routine), low rho
          on the tail (the frontier). A flat rho means the FM shaves every
          direction by the same fraction and there is no partition to speak of.
      containment_curve  : fraction of residual variance inside A's top-k PCs,
          with the k/D chance line and the excess over chance.
      alignment_index    : mean excess-over-chance across the full k sweep;
          0 = residual is oriented at chance w.r.t. A, 1 = perfectly nested.
      principal_angles   : subspace overlap between A's and R's own top-k PCs,
          against an empirical random-subspace control.
    """
    rng = np.random.default_rng(seed)
    R = A - P
    N, D = A.shape

    Ac = A - A.mean(axis=0, keepdims=True)
    Rc = R - R.mean(axis=0, keepdims=True)
    _, S_A, Vt_A = np.linalg.svd(Ac, full_matrices=False)
    _, S_R, Vt_R = np.linalg.svd(Rc, full_matrices=False)

    # --- per-PC absorption spectrum, in A's basis ---
    var_A = (Ac @ Vt_A.T).var(axis=0)          # == S_A^2 / N, A's variance per PC
    var_R = (Rc @ Vt_A.T).var(axis=0)          # residual variance in the SAME basis
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_raw = 1.0 - np.where(var_A > 0, var_R / np.maximum(var_A, 1e-30), 0.0)
    # rho is unbounded below: in A's low-variance tail directions the residual can
    # carry many times the variance A itself has there, sending rho -> -inf and
    # making any mean/step statistic over it meaningless. Summary statistics use
    # the clipped fraction-absorbed; the raw version survives as a diagnostic.
    rho = np.clip(rho_raw, 0.0, 1.0)

    # --- spectrum-matched null for the absorption step ---
    # A step in rho is NOT by itself evidence that the FM preferentially absorbed
    # A's leading directions: whenever A's spectrum decays, a perfectly flat
    # residual already yields rho ~ 1 on the leading PCs and rho ~ 0 in the tail.
    # The null below is that flat residual at matched total variance, so the
    # excess is what actually indicates preferential absorption.
    dec = max(1, D // 10)
    rho_null = np.clip(1.0 - (var_R.sum() / D) / np.maximum(var_A, 1e-30), 0.0, 1.0)
    step_null = float(np.mean(rho_null[:dec]) - np.mean(rho_null[-dec:]))
    step_obs = float(np.mean(rho[:dec]) - np.mean(rho[-dec:]))

    # --- containment of residual variance in A's top-k subspace vs chance ---
    res_energy_by_A_pc = var_R / max(var_R.sum(), 1e-30)
    cum_res = np.cumsum(res_energy_by_A_pc)
    curve = []
    for k in [k for k in ks if k < D] + [D]:
        chance = k / D
        curve.append({
            "k": int(k),
            "res_var_inside": float(cum_res[k - 1]),
            "chance": float(chance),
            "excess": float(cum_res[k - 1] - chance),
        })
    sweep = [c for c in curve if c["k"] < D]
    alignment_index = float(np.mean([c["excess"] / max(1 - c["chance"], 1e-12)
                                     for c in sweep])) if sweep else 0.0

    # --- principal angles between A's and R's own top-k subspaces ---
    angles = []
    for k in [k for k in ks if k < D]:
        overlap = np.linalg.norm(Vt_A[:k] @ Vt_R[:k].T, "fro") ** 2 / k
        ctrl = []
        for _ in range(5):
            Q1, _ = np.linalg.qr(rng.standard_normal((D, k)))
            Q2, _ = np.linalg.qr(rng.standard_normal((D, k)))
            ctrl.append(np.linalg.norm(Q1.T @ Q2, "fro") ** 2 / k)
        angles.append({
            "k": int(k),
            "subspace_overlap": float(overlap),
            "random_control": float(np.mean(ctrl)),
            "excess": float(overlap - np.mean(ctrl)),
        })

    return {
        "absorption_spectrum": rho.tolist(),
        "absorption_mean": float(np.mean(rho)),
        "absorption_top_decile": float(np.mean(rho[:dec])),
        "absorption_bottom_decile": float(np.mean(rho[-dec:])),
        "absorption_step": step_obs,
        "absorption_step_null": step_null,
        "absorption_step_excess": step_obs - step_null,
        "absorption_mean_raw": float(np.mean(rho_raw)),
        "absorption_negative_frac": float(np.mean(rho_raw < 0)),
        "containment_curve": curve,
        "alignment_index": alignment_index,
        "principal_angles": angles,
        "act_variance_spectrum": var_A.tolist(),
        "res_variance_in_act_basis": var_R.tolist(),
    }


# --------------------------------------------------------------------------
# 3. The repaired triple
# --------------------------------------------------------------------------

def repaired_triple(A, P):
    """A partition of A's variance along A's own axes, so the triple is coherent.

    Along each of A's principal directions v_i with variance lam_i, the FM
    explains a fraction rho_i and leaves (1 - rho_i). That splits A's spectrum
    into an absorbed part {rho_i lam_i} and a frontier part {(1-rho_i) lam_i}
    which sum back to {lam_i} exactly. Two readouts follow:

      * Exact partition (what the belief's "moving partition" wants to be):
            R_comp_v = R_act * absorbed_fraction
            R_res_v  = R_act * (1 - absorbed_fraction)
        so R_comp_v + R_res_v = R_act identically, and both move with residual
        magnitude -- the property the naive R_res lacks entirely.

      * **R_res_participation** -- the frontier's dimensionality, and the
        headline replacement for naive residual rank. It answers: *how many of
        the model's own working directions still carry meaningful unexplained
        computation?*

        It is the participation ratio of the frontier spectrum,
        PR({(1-rho_i) lam_i}), so it counts in A's OWN basis, weighted by how
        much computation A actually does in each direction. Naive R_res instead
        SVDs the raw residual and counts any direction with variance in it,
        weighting a direction carrying 0.01% of the computation exactly as much
        as one carrying 30%. That is the whole difference, and it is large: on
        RHM naive R_res reads 84-91 where R_res_participation reads ~7, because
        the naive count was dominated by directions the model barely uses.

        R_comp_participation is its counterpart over the absorbed spectrum.
        Neither sums to R_act, and neither is supposed to -- "how spread out is
        the frontier" is a different question from "how big is it", and the
        naive R_res conflated them. That conflation is why it stayed pinned at
        90-96% while residual norm moved 17x in RESIDUAL_RANK_README Exp. 3.

    Keys ending `_pr` are retained as deprecated aliases of the
    `_participation` keys so summaries written before the rename still parse.

    rho_i is clipped to [0, 1] for the partition (a negative rho means the FM
    actively *added* variance in that direction); the unclipped mean is
    reported so that anti-absorption is visible rather than silently dropped.
    """
    R = A - P
    Ac = A - A.mean(axis=0, keepdims=True)
    Rc = R - R.mean(axis=0, keepdims=True)
    _, _, Vt_A = np.linalg.svd(Ac, full_matrices=False)

    lam = (Ac @ Vt_A.T).var(axis=0)
    var_R = (Rc @ Vt_A.T).var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_raw = 1.0 - np.where(lam > 0, var_R / np.maximum(lam, 1e-30), 0.0)
    rho = np.clip(rho_raw, 0.0, 1.0)

    lam_comp = rho * lam
    lam_res = (1.0 - rho) * lam

    R_act = participation_ratio(lam)
    absorbed_fraction = float(lam_comp.sum() / max(lam.sum(), 1e-30))

    return {
        "R_act_pr": R_act,
        "absorbed_fraction": absorbed_fraction,
        "frontier_mass": 1.0 - absorbed_fraction,
        # exact partition: these two sum to R_act_pr by construction
        "R_comp_v": R_act * absorbed_fraction,
        "R_res_v": R_act * (1.0 - absorbed_fraction),
        # Dimensionality of each part, counted in A's own basis and weighted by
        # A's own variance. R_res_participation is the headline replacement for
        # naive residual rank -- see this function's docstring.
        "R_comp_participation": participation_ratio(lam_comp),
        "R_res_participation": participation_ratio(lam_res),
        # Deprecated aliases, kept so pre-rename summaries still parse.
        "R_comp_pr": participation_ratio(lam_comp),
        "R_res_pr": participation_ratio(lam_res),
        # dimension counts in the SAME entropy units as the naive triple, so
        # R_res_H is directly comparable to naive R_res. The two differ only in
        # that R_res_H weights each direction by the variance A actually has
        # there, instead of letting near-empty directions of A count as fully
        # as the ones carrying the computation.
        "R_act_H": eff_rank_entropy(S=np.sqrt(np.clip(lam, 0, None))),
        "R_comp_H": eff_rank_entropy(S=np.sqrt(np.clip(lam_comp, 0, None))),
        "R_res_H": eff_rank_entropy(S=np.sqrt(np.clip(lam_res, 0, None))),
        "rho_mean_raw": float(np.mean(rho_raw)),
        "rho_negative_frac": float(np.mean(rho_raw < 0)),
    }


# --------------------------------------------------------------------------

def full_decomposition(A, P, ks=(4, 8, 16, 32, 64), seed=0, max_samples=20000):
    """All three views at once. A, P are (N, D) float arrays."""
    A = np.asarray(A, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if A.shape[0] > max_samples:
        idx = np.random.default_rng(seed).choice(A.shape[0], max_samples, replace=False)
        A, P = A[idx], P[idx]

    R = A - P
    a_norm = np.linalg.norm(A, axis=1)
    r_norm = np.linalg.norm(R, axis=1)
    denom = a_norm * np.linalg.norm(P, axis=1)
    cos = float(np.mean(np.sum(A * P, axis=1) / np.maximum(denom, 1e-30)))

    return {
        "n_samples": int(A.shape[0]),
        "basic": {
            "mean_residual_norm": float(r_norm.mean()),
            "mean_target_norm": float(a_norm.mean()),
            "relative_residual": float(r_norm.mean() / max(a_norm.mean(), 1e-30)),
            "mean_cosine": cos,
        },
        "naive": naive_triple(A, P),
        "geometry": residual_geometry(A, P, ks=ks, seed=seed),
        "repaired": repaired_triple(A, P),
    }
