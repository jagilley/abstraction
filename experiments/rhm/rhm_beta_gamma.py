"""Measure Cagnetta β and γ for RHM data-generating processes.

β: correlation decay exponent  ||C(n)||_op ~ n^{-β}
γ: conditional entropy decay   H_n - H_∞ ~ n^{-γ}
α_D = γ/(2β): predicted data-limited scaling exponent

Compares with natural language measurements from the language reduction experiment:
  β = 0.929, γ = 0.509, α_D(predicted) = 0.274, α_D(empirical) = 0.082

Usage:
  cd experiments/
  modal run --detach rhm/rhm_beta_gamma.py::beta_gamma_sweep
"""

import json
import numpy as np
from rhm.shared import app, volume, DATA_DIR, setting_key, NumpyEncoder

LANGUAGE = {
    "beta": 0.929,
    "gamma": 0.509,
    "alpha_predicted": 0.274,
    "alpha_empirical": 0.082,
}


def _compute_beta(sequences, v):
    """Measure β from within-sequence covariance decay.

    Computes C(n) only from token pairs within the same sequence to avoid
    the dilution artifact from cross-boundary (zero-correlation) pairs.
    """
    n_seq, seq_len = sequences.shape
    max_lag = seq_len - 1

    flat = sequences.reshape(-1).astype(np.int64)
    marginal = np.bincount(flat, minlength=v).astype(np.float64)
    marginal /= marginal.sum()
    outer = np.outer(marginal, marginal)

    lags = []
    norms_op = []

    for lag in range(1, max_lag + 1):
        src = sequences[:, :seq_len - lag].reshape(-1).astype(np.int64)
        dst = sequences[:, lag:].reshape(-1).astype(np.int64)
        flat_idx = src * v + dst
        joint_flat = np.bincount(flat_idx, minlength=v * v).astype(np.float64)
        joint = joint_flat.reshape(v, v)
        joint /= joint.sum()
        C_n = joint - outer
        norm_op = float(np.linalg.norm(C_n, ord=2))
        lags.append(lag)
        norms_op.append(norm_op)

    lags_arr = np.array(lags, dtype=np.float64)
    norms_arr = np.array(norms_op)

    fit_mask = norms_arr > 1e-10
    if fit_mask.sum() >= 3:
        log_n = np.log(lags_arr[fit_mask])
        log_norm = np.log(norms_arr[fit_mask])
        coeffs = np.polyfit(log_n, log_norm, 1)
        beta = float(-coeffs[0])
        predicted = coeffs[0] * log_n + coeffs[1]
        ss_res = np.sum((log_norm - predicted) ** 2)
        ss_tot = np.sum((log_norm - log_norm.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        beta, r2 = None, None

    return {
        "beta": beta,
        "r2": r2,
        "lags": lags,
        "norms_op": norms_op,
        "fit_n_points": int(fit_mask.sum()),
    }


def _compute_gamma(sequences, v, max_k=8):
    """Measure γ from n-gram conditional entropy decay.

    Counts all k-grams within sequences (not across boundaries) for k=1..max_k.
    Computes H_n = H_joint(n) - H_joint(n-1), then fits H_n - H_∞ ~ n^{-γ}.

    Dense counting: v^k entries. For v=8, k=8: 16.7M entries (134 MB). Safe.
    """
    n_seq, seq_len = sequences.shape

    while v ** max_k > 500_000_000:
        max_k -= 1
    max_k = min(max_k, seq_len)

    h_joint = {}
    n_observed = {}

    for k in range(1, max_k + 1):
        n_possible = v ** k
        counts = np.zeros(n_possible, dtype=np.int64)

        for pos in range(seq_len - k + 1):
            idx = sequences[:, pos].astype(np.int64)
            for i in range(1, k):
                idx = idx * v + sequences[:, pos + i]
            counts += np.bincount(idx, minlength=n_possible)

        total = counts.sum()
        p = counts[counts > 0].astype(np.float64) / total
        h_joint[k] = float(-np.sum(p * np.log(p)))
        n_observed[k] = int(np.sum(counts > 0))

        coverage = n_observed[k] / n_possible * 100
        print(f"  k={k}: H_joint={h_joint[k]:.4f}, "
              f"observed {n_observed[k]:,}/{n_possible:,} ({coverage:.1f}%), "
              f"total={total:,}")

    h_cond = {}
    for n in range(1, max_k + 1):
        if n == 1:
            h_cond[n] = h_joint[1]
        else:
            h_cond[n] = h_joint[n] - h_joint[n - 1]

    ns = np.array(sorted(h_cond.keys()), dtype=np.float64)
    Hs = np.array([h_cond[int(n)] for n in ns])
    H_inf = float(Hs.min())

    fit_mask = (ns >= 2) & ((Hs - H_inf) > 1e-6)
    if fit_mask.sum() >= 3:
        log_n = np.log(ns[fit_mask])
        log_delta_H = np.log(Hs[fit_mask] - H_inf)
        coeffs = np.polyfit(log_n, log_delta_H, 1)
        gamma = float(-coeffs[0])
        predicted = coeffs[0] * log_n + coeffs[1]
        ss_res = np.sum((log_delta_H - predicted) ** 2)
        ss_tot = np.sum((log_delta_H - log_delta_H.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        gamma, r2 = None, None

    return {
        "gamma": gamma,
        "r2": r2,
        "H_inf": H_inf,
        "h_cond": {str(k): float(v_) for k, v_ in h_cond.items()},
        "h_joint": {str(k): float(v_) for k, v_ in h_joint.items()},
        "n_observed": {str(k): int(v_) for k, v_ in n_observed.items()},
        "max_k": max_k,
    }


@app.function(volumes={DATA_DIR: volume}, timeout=3600, memory=16384)
def measure_setting(v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
                    n_tokens: int = 20_000_000, max_ngram: int = 8):
    """Measure β and γ for a single (v, s, L, m) setting."""
    import os
    from rhm.rhm_data import make_corpus

    L = depth
    key = setting_key(v, s, L, m)
    seq_len = s ** L

    volume.reload()
    corpus_path = f"{DATA_DIR}/{key}/corpus.npy"
    if os.path.exists(corpus_path):
        corpus = np.load(corpus_path)
        if len(corpus) < n_tokens:
            print(f"Existing corpus too small ({len(corpus):,}), regenerating")
            corpus, _, _ = make_corpus(v, s, L, m, n_tokens)
        else:
            corpus = corpus[:n_tokens]
    else:
        print(f"No existing corpus for {key}, generating")
        corpus, _, _ = make_corpus(v, s, L, m, n_tokens)

    n_sequences = len(corpus) // seq_len
    sequences = corpus[:n_sequences * seq_len].reshape(n_sequences, seq_len)
    print(f"Setting: {key}, seq_len={seq_len}, n_sequences={n_sequences:,}, "
          f"tokens={n_sequences * seq_len:,}")

    # β
    print("\n--- β (correlation decay) ---")
    beta_result = _compute_beta(sequences, v)
    b = beta_result['beta']
    print(f"β = {b:.4f} (R² = {beta_result['r2']:.4f}, "
          f"{beta_result['fit_n_points']} lags)")

    # Sample norms at powers of s
    print("  ||C(n)||_op at hierarchy-relevant lags:")
    for lag in [1, 2, 4, 8, 16, 32, 64, 128, 255]:
        if lag <= len(beta_result['norms_op']):
            print(f"    lag {lag:4d}: {beta_result['norms_op'][lag - 1]:.6f}")

    # γ
    print("\n--- γ (conditional entropy decay) ---")
    gamma_result = _compute_gamma(sequences, v, max_k=max_ngram)
    g = gamma_result['gamma']
    hi = gamma_result['H_inf']

    print(f"\nConditional entropies H_n:")
    for n_str in sorted(gamma_result['h_cond'].keys(), key=int):
        print(f"  H_{n_str} = {gamma_result['h_cond'][n_str]:.4f}")

    if g is not None:
        print(f"\nγ = {g:.4f} (R² = {gamma_result['r2']:.4f})")
    print(f"H_∞ = {hi:.4f}")

    alpha_pred = None
    if b and g and b > 0:
        alpha_pred = g / (2 * b)
        print(f"\nα_D = γ/(2β) = {g:.4f}/(2×{b:.4f}) = {alpha_pred:.4f}")
        print(f"  (language: α_D = {LANGUAGE['alpha_predicted']:.3f})")

    return {
        "setting": key,
        "v": v, "s": s, "L": L, "m": m,
        "seq_len": seq_len,
        "n_sequences": n_sequences,
        "beta": beta_result,
        "gamma": gamma_result,
        "alpha_D_predicted": alpha_pred,
    }


@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=4096)
def beta_gamma_sweep(v: int = 8, s: int = 2,
                     l_values: str = "4,6,8", m_values: str = "2,4,8"):
    """Sweep β and γ across (L, m) settings and compare with language."""
    import os

    Ls = [int(x) for x in l_values.split(",")]
    ms = [int(x) for x in m_values.split(",")]

    print(f"β/γ sweep: L ∈ {Ls}, m ∈ {ms}, v={v}, s={s}")
    print(f"Language: β={LANGUAGE['beta']:.3f}, γ={LANGUAGE['gamma']:.3f}, "
          f"α_D(pred)={LANGUAGE['alpha_predicted']:.3f}, "
          f"α_D(emp)={LANGUAGE['alpha_empirical']:.3f}\n")

    handles = []
    for L in Ls:
        for m_val in ms:
            h = measure_setting.spawn(v=v, s=s, depth=L, m=m_val)
            handles.append((L, m_val, h))

    results = []
    for L, m_val, h in handles:
        r = h.get()
        results.append(r)

    # Load empirical α_D from existing scaling sweep data
    volume.reload()
    alpha_empirical = {}

    for r in results:
        key = r['setting']
        for path in [f"{DATA_DIR}/{key}/results/scaling.json"]:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                ae = data.get('scaling', {}).get('alpha_empirical')
                if ae:
                    alpha_empirical[key] = ae
                break

    compact_path = f"{DATA_DIR}/hparam_sweep_compact.json"
    if os.path.exists(compact_path):
        with open(compact_path) as f:
            compact = json.load(f)
        if isinstance(compact, list):
            for entry in compact:
                k = entry.get('setting', '')
                ae = (entry.get('alpha_empirical')
                      or entry.get('scaling', {}).get('alpha_empirical'))
                if ae and k and k not in alpha_empirical:
                    alpha_empirical[k] = ae

    # Print comparison table
    print("\n" + "=" * 95)
    print("β, γ, α_D comparison: RHM settings vs natural language")
    print("=" * 95)
    header = (f"{'Setting':>15s}  {'β':>6s}  {'R²β':>5s}  {'γ':>6s}  {'R²γ':>5s}  "
              f"{'α pred':>7s}  {'α emp':>7s}  {'H∞':>6s}  {'H1':>6s}")
    print(header)
    print("-" * 95)
    print(f"{'Language':>15s}  {LANGUAGE['beta']:6.3f}  {'—':>5s}  "
          f"{LANGUAGE['gamma']:6.3f}  {'—':>5s}  "
          f"{LANGUAGE['alpha_predicted']:7.3f}  {LANGUAGE['alpha_empirical']:7.3f}  "
          f"{'—':>6s}  {'—':>6s}")
    print("-" * 95)

    for r in sorted(results, key=lambda x: (x['L'], x['m'])):
        key = r['setting']
        b = r['beta']['beta']
        r2b = r['beta']['r2']
        g = r['gamma']['gamma']
        r2g = r['gamma']['r2']
        ap = r['alpha_D_predicted']
        ae = alpha_empirical.get(key)
        hi = r['gamma']['H_inf']
        h1 = r['gamma']['h_cond'].get('1')

        def fmt(val, w, prec=3):
            return f"{val:{w}.{prec}f}" if val is not None else f"{'N/A':>{w}s}"

        print(f"{key:>15s}  {fmt(b,6)}  {fmt(r2b,5)}  {fmt(g,6)}  {fmt(r2g,5)}  "
              f"{fmt(ap,7)}  {fmt(ae,7)}  {fmt(hi,6,4)}  {fmt(h1,6,4)}")

    print("=" * 95)

    # Print H_n curves for each setting
    print("\nConditional entropy H_n by setting:")
    print(f"{'Setting':>15s}", end="")
    max_k = max(int(k) for r in results for k in r['gamma']['h_cond'].keys())
    for n in range(1, max_k + 1):
        print(f"  {'H'+str(n):>6s}", end="")
    print()
    print("-" * (15 + 8 * max_k))

    for r in sorted(results, key=lambda x: (x['L'], x['m'])):
        key = r['setting']
        print(f"{key:>15s}", end="")
        for n in range(1, max_k + 1):
            h = r['gamma']['h_cond'].get(str(n))
            if h is not None:
                print(f"  {h:6.4f}", end="")
            else:
                print(f"  {'—':>6s}", end="")
        print()

    # Print ||C(n)||_op at hierarchy-relevant lags
    print(f"\n||C(n)||_op at powers of s={s}:")
    power_lags = [s ** k for k in range(10) if s ** k < 256]
    print(f"{'Setting':>15s}", end="")
    for lag in power_lags:
        print(f"  {'n='+str(lag):>8s}", end="")
    print()
    print("-" * (15 + 10 * len(power_lags)))

    for r in sorted(results, key=lambda x: (x['L'], x['m'])):
        key = r['setting']
        norms = r['beta']['norms_op']
        print(f"{key:>15s}", end="")
        for lag in power_lags:
            if lag <= len(norms):
                print(f"  {norms[lag - 1]:8.5f}", end="")
            else:
                print(f"  {'—':>8s}", end="")
        print()

    # Save
    out_dir = f"{DATA_DIR}/rhm_beta_gamma"
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "language_reference": LANGUAGE,
        "alpha_empirical_from_sweep": alpha_empirical,
        "settings": results,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    volume.commit()

    print(f"\nResults saved to {out_dir}/results.json")
    return summary
