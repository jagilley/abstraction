"""Compact scaling sweep: 3 settings x 3 P values = 9 GPU jobs.

Picks 3 corners of the L x m space to isolate both main effects on alpha_D:
  - (L=4, m=2): simple baseline
  - (L=4, m=8): same depth, high multiplicity -> isolates m effect
  - (L=8, m=2): deep, same multiplicity -> isolates L effect

Uses 3 P values per setting spanning 3 orders of magnitude (exponents 0, 1.5, 3.0
relative to min_p). Fits alpha_D from trained models directly rather than using
the sweep primitive (which spawns 7 jobs per setting).

Usage:
    modal run --detach rhm/hparam_sweep.py::hparam_sweep
"""

import json

from rhm.shared import app, volume, DATA_DIR, setting_key, NumpyEncoder
from rhm.stages import generate_corpus, train_model

SETTINGS = [
    {"L": 4, "m": 2},
    {"L": 4, "m": 8},
    {"L": 8, "m": 2},
]
V = 8
S = 2
P_EXPONENTS = [0, 1.5, 3.0]
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128


def p_values_for_setting(s, L):
    seq_len = s ** L
    min_p = max(1000, 50 * seq_len)
    return [min(20_000_000, int(min_p * 10 ** exp)) for exp in P_EXPONENTS]


@app.function(volumes={DATA_DIR: volume}, timeout=7200, memory=4096)
def hparam_sweep():
    import os
    from rhm.measure import measure_empirical_scaling

    all_jobs = []
    for cfg in SETTINGS:
        L, m = cfg["L"], cfg["m"]
        for p in p_values_for_setting(S, L):
            all_jobs.append({"L": L, "m": m, "P": p})

    print(f"=== Compact Scaling Sweep ===")
    print(f"  v={V}, s={S}, model={N_LAYER}L/{N_HEAD}H/{N_EMBD}D")
    print(f"  Settings: {[(c['L'], c['m']) for c in SETTINGS]}")
    print(f"  GPU jobs: {len(all_jobs)}")
    for job in all_jobs:
        print(f"    L={job['L']}, m={job['m']}, P={job['P']:,}")

    # Phase 1: generate corpora (CPU only, not counted toward GPU limit)
    print(f"\n--- Phase 1: Generating corpora ---")
    gen_handles = []
    for cfg in SETTINGS:
        L, m = cfg["L"], cfg["m"]
        max_p = max(p_values_for_setting(S, L))
        n_tokens = int(max_p * 1.2)
        key = setting_key(V, S, L, m)
        print(f"  {key}: {n_tokens:,} tokens")
        h = generate_corpus.spawn(v=V, s=S, depth=L, m=m, n_tokens=n_tokens)
        gen_handles.append((key, h))

    for key, h in gen_handles:
        h.get()
        print(f"  Corpus ready: {key}")

    # Phase 2: train all models (9 GPU jobs)
    print(f"\n--- Phase 2: Training models ---")
    train_handles = []
    for job in all_jobs:
        L, m, P = job["L"], job["m"], job["P"]
        key = setting_key(V, S, L, m)
        print(f"  Spawning: {key} P={P:,}")
        h = train_model.spawn(
            v=V, s=S, depth=L, m=m, n_tokens=P,
            n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        )
        train_handles.append((L, m, P, h))

    train_results = []
    for L, m, P, h in train_handles:
        key = setting_key(V, S, L, m)
        try:
            result = h.get()
            print(f"  {key} P={P:,}: best_val_loss={result['best_val_loss']:.4f}")
            train_results.append(result)
        except Exception as e:
            print(f"  {key} P={P:,}: FAILED -- {e}")

    # Phase 3: fit scaling exponents
    print(f"\n--- Phase 3: Measuring scaling exponents ---")
    scaling_results = []
    for cfg in SETTINGS:
        L, m = cfg["L"], cfg["m"]
        key = setting_key(V, S, L, m)
        loss_vs_P = [
            (r["n_tokens"], r["best_val_loss"])
            for r in train_results
            if r["L"] == L and r["m"] == m
        ]
        if len(loss_vs_P) < 3:
            print(f"  {key}: only {len(loss_vs_P)} points, need 3 -- skipping")
            continue
        scaling = measure_empirical_scaling(loss_vs_P)
        scaling_results.append({
            "setting": key, "L": L, "m": m,
            "alpha": scaling["alpha_empirical"],
            "r2": scaling["r2"],
            "loss_vs_P": loss_vs_P,
            "scaling": scaling,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"{'Setting':<20} {'L':>3} {'m':>3} {'alpha':>8} {'R2':>8}")
    print(f"{'-'*20} {'-'*3} {'-'*3} {'-'*8} {'-'*8}")
    for r in scaling_results:
        print(f"{r['setting']:<20} {r['L']:>3} {r['m']:>3} {r['alpha']:>8.4f} {r['r2']:>8.4f}")
    print(f"{'='*60}")

    summary = {
        "config": {
            "v": V, "s": S,
            "settings": SETTINGS,
            "p_exponents": P_EXPONENTS,
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
        },
        "scaling_results": scaling_results,
        "train_results": train_results,
    }
    out_path = f"{DATA_DIR}/hparam_sweep_compact.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {out_path}")

    return summary
