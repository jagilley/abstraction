"""Measure the scaling exponents beta and gamma from data and trained models.

beta: power-law exponent of token-token correlation decay with lag.
      ||C(n)||_op ~ n^{-beta}

gamma: power-law exponent of conditional entropy decay with context length.
       H_n - H_inf ~ n^{-gamma}

alpha_D = gamma / (2*beta) is the predicted data-limited scaling exponent.
"""

import os
import json
import glob
import numpy as np
from scipy.optimize import curve_fit


def measure_beta(stats_dir: str, n_lags: int = 50, fit_range: tuple = (1, 30)):
    """Measure beta from the operator norm decay of C(n).

    Fits ||C(n)||_op ~ n^{-beta} over the specified lag range.
    """
    lags = []
    norms_op = []
    norms_f = []

    for lag in range(1, n_lags + 1):
        C_n = np.load(os.path.join(stats_dir, "covariance", f"C_lag_{lag:03d}.npy"))
        norms_op.append(np.linalg.norm(C_n, ord=2))
        norms_f.append(np.linalg.norm(C_n, ord="fro"))
        lags.append(lag)

    lags = np.array(lags, dtype=np.float64)
    norms_op = np.array(norms_op)
    norms_f = np.array(norms_f)

    # fit power law in log-log space over specified range
    lo, hi = fit_range
    hi = min(hi, n_lags)
    mask = (lags >= lo) & (lags <= hi)
    log_n = np.log(lags[mask])
    log_norm = np.log(norms_op[mask])

    # log(||C(n)||) = log(A) - beta * log(n)
    coeffs = np.polyfit(log_n, log_norm, 1)
    beta = -coeffs[0]
    log_A = coeffs[1]

    result = {
        "beta": float(beta),
        "log_A": float(log_A),
        "fit_range": list(fit_range),
        "lags": lags.tolist(),
        "norms_op": norms_op.tolist(),
        "norms_f": norms_f.tolist(),
    }
    print(f"beta = {beta:.4f} (fit over lags {lo}..{hi})")
    return result


def measure_gamma_from_ngram_losses(loss_curves: dict, fit_n_range: tuple = (2, 12)):
    """Estimate gamma from n-gram loss curves L_n(P, M).

    Following Cagnetta et al.: gamma is the exponent of H_n - H_inf ~ n^{-gamma}.
    We estimate H_n as the converged value of L_n(P) at the largest P, then
    fit the decay of H_n with n.

    Args:
        loss_curves: dict mapping context_length n -> list of (P, L_n(P)) pairs,
                     where P is dataset size and L_n is the n-gram loss.
        fit_n_range: (lo, hi) range of n values for the power-law fit.

    Returns dict with gamma estimate and diagnostics.
    """
    # for each n, take L_n at the largest P as our estimate of H_n
    H_estimates = {}
    for n_str, curve in loss_curves.items():
        n = int(n_str)
        if len(curve) == 0:
            continue
        # curve is list of (P, loss) sorted by P
        curve_sorted = sorted(curve, key=lambda x: x[0])
        H_estimates[n] = curve_sorted[-1][1]

    ns = sorted(H_estimates.keys())
    Hs = np.array([H_estimates[n] for n in ns])
    ns = np.array(ns, dtype=np.float64)

    # H_inf estimate: minimum H across all n
    H_inf = float(Hs.min())

    # fit H_n - H_inf ~ n^{-gamma} in log-log
    lo, hi = fit_n_range
    mask = (ns >= lo) & (ns <= hi) & ((Hs - H_inf) > 1e-6)
    if mask.sum() < 2:
        print("Warning: not enough valid points for gamma fit")
        return {"gamma": None, "H_inf": H_inf}

    log_n = np.log(ns[mask])
    log_delta_H = np.log(Hs[mask] - H_inf)

    coeffs = np.polyfit(log_n, log_delta_H, 1)
    gamma = -coeffs[0]

    result = {
        "gamma": float(gamma),
        "H_inf": H_inf,
        "fit_range": list(fit_n_range),
        "n_values": ns.tolist(),
        "H_estimates": Hs.tolist(),
    }
    print(f"gamma = {gamma:.4f}, H_inf = {H_inf:.4f} (fit over n={lo}..{hi})")
    return result


def predict_alpha_D(beta: float, gamma: float):
    """Predict data-limited scaling exponent from measured statistics."""
    alpha_D = gamma / (2.0 * beta)
    print(f"alpha_D = gamma/(2*beta) = {gamma:.4f}/(2*{beta:.4f}) = {alpha_D:.4f}")
    return alpha_D


def measure_empirical_scaling(loss_vs_P: list[tuple[int, float]],
                              fit_range: tuple | None = None):
    """Fit empirical L(P) ~ P^{-alpha} + H_inf.

    Args:
        loss_vs_P: list of (P, loss) pairs
        fit_range: optional (P_min, P_max) to restrict the fit

    Returns dict with empirical alpha and diagnostics.
    """
    data = np.array(sorted(loss_vs_P))
    P = data[:, 0]
    L = data[:, 1]

    if fit_range:
        mask = (P >= fit_range[0]) & (P <= fit_range[1])
        P, L = P[mask], L[mask]

    # fit log(L - L_min) = log(A) - alpha * log(P) using grid search for H_inf
    best_r2 = -np.inf
    best_alpha = None
    best_H_inf = None

    for H_offset in np.linspace(0, L.min() * 0.99, 50):
        H_trial = L.min() - H_offset
        delta = L - H_trial
        if (delta <= 0).any():
            continue
        log_P = np.log(P)
        log_delta = np.log(delta)
        coeffs = np.polyfit(log_P, log_delta, 1)
        alpha = -coeffs[0]
        predicted = coeffs[0] * log_P + coeffs[1]
        ss_res = np.sum((log_delta - predicted) ** 2)
        ss_tot = np.sum((log_delta - log_delta.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
            best_H_inf = H_trial

    result = {
        "alpha_empirical": float(best_alpha) if best_alpha else None,
        "H_inf_empirical": float(best_H_inf) if best_H_inf else None,
        "r2": float(best_r2),
        "P_values": P.tolist(),
        "L_values": L.tolist(),
    }
    if best_alpha:
        print(f"alpha_empirical = {best_alpha:.4f} (R² = {best_r2:.4f})")
    return result


def save_results(results: dict, results_dir: str, tau: float):
    """Save measurement results to JSON."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"results_tau_{tau:.3f}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
