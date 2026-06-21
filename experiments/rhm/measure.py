"""Measure empirical scaling exponent from trained models."""

import numpy as np


def measure_empirical_scaling(loss_vs_P, fit_range=None):
    """Fit empirical L(P) ~ A * P^{-alpha} + H_inf.

    Uses grid search over H_inf to find the best power-law fit.

    Args:
        loss_vs_P: list of (P, loss) pairs
        fit_range: optional (P_min, P_max) to restrict the fit

    Returns dict with alpha_empirical, H_inf, R², and diagnostics.
    """
    data = np.array(sorted(loss_vs_P))
    P = data[:, 0]
    L = data[:, 1]

    if fit_range:
        mask = (P >= fit_range[0]) & (P <= fit_range[1])
        P, L = P[mask], L[mask]

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
