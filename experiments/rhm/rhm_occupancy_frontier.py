"""NTP frontier as a function of tuple-space occupancy.

THE question this settles: when we lower occupancy `m / v^(s-1)`, does plain
next-token prediction learn the RHM hierarchy to a GREATER DEPTH (the "frontier"
advances), or does lowering occupancy only raise the BP *ceiling* (what is
recoverable in principle) while the learned frontier stays pinned at ~3.5 levels?

Thread B (RHM_DEEP_COMPOSITION_README.md, Exp. 4) established that at occupancy
0.25 (v16/m4) NTP saturates at a ~3.5-level frontier while the BP root ceiling is
~0.80 -- a large recoverable-but-unlearned gap. That experiment was at a single
occupancy. This sweep holds architecture (8L/8H/256D), depth (L=6), branching
(s=2), corpus size, and training budget (n_steps) FIXED and varies ONLY occupancy,
along two spokes so we can also test whether the frontier *collapses* on occupancy
the way the ceiling does (Exp. 3's occupancy law):

  - fixed m=4, vary v in {8, 16, 32, 64}  -> occupancy {0.500, 0.250, 0.125, 0.0625}
  - fixed v=16, vary m in {2, 4, 8}        -> occupancy {0.125, 0.250, 0.500}

  collapse checks: occ 0.500 via (v8,m4) & (v16,m8); occ 0.125 via (v32,m4) & (v16,m2).
  If both members of a pair give the same frontier, occupancy (not v or m alone)
  is the control variable for learnability too.

Interpretation:
  Outcome (1) -- lower occupancy advances the frontier toward the root:
      curriculum-over-occupancy (train at low m/v, transfer to high) is well founded;
      the frontier is occupancy-bound, not intrinsically depth-bound.
  Outcome (2) -- frontier stuck ~3.5 levels regardless of occupancy:
      the frontier is depth-bound (NTP gradient reach, not recoverability); a
      curriculum over occupancy won't help and we need explicit staged
      cluster-and-lift (re-tokenize with stage-1 soft features) instead.

Either way the goal is a base model that genuinely encodes the deep hierarchy
single-task -- the substrate on which the A2A forward-model loop / meta-learning
finally get a fair test (and on which we'd expect the FM residual to become
low-rank / hierarchy-legible, as on MNIST, rather than high-rank / diffuse).

This file ONLY orchestrates: it reuses rhm_thread_b.thread_b verbatim (per-setting
NTP training + per-level last-token linear & MLP probes -- exactly the frontier
measurement), fans it out over the settings, and aggregates the final frontiers.
Training-free reference lines (BP ceiling / greedy floor) per setting are CPU/local:
    python3 rhm_local_signal_sweeps.py --which occ_frontier

Run (GPU sweep, 6 parallel A10G jobs):
    modal run --detach -m rhm.rhm_occupancy_frontier::occupancy_frontier
"""

import json
import os

from rhm.shared import volume, DATA_DIR, NumpyEncoder

# Reuse Thread B's app + function + image so thread_b.spawn() is hydrated when this
# orchestrator runs (single Modal app, two registered functions).
from rhm.rhm_thread_b import app, thread_b, tb_key

# (v, m); s=2, L=6 fixed. occupancy = m / v**(s-1) = m / v.
SETTINGS = [
    (8, 4),    # occ 0.5000
    (16, 8),   # occ 0.5000  (collapse-check vs v8 m4)
    (16, 4),   # occ 0.2500  (Thread B baseline; should reproduce its ~3.5 frontier)
    (32, 4),   # occ 0.1250
    (16, 2),   # occ 0.1250  (collapse-check vs v32 m4)
    (64, 4),   # occ 0.0625
]
S, L = 2, 6
N_LAYER, N_HEAD, N_EMBD = 8, 8, 256
N_STEPS = 40000          # ~saturation proxy (Thread B 4c: d4 0.764@40k vs 0.790@100k)
N_TOKENS = 20_000_000
CKPT_STEPS = "0,2000,5000,10000,20000,40000"


def _kv(d, k):
    """Fetch d[k] tolerating int-vs-str keys (in-process vs JSON round-trip)."""
    if k in d:
        return d[k]
    return d.get(str(k), d.get(int(k) if str(k).isdigit() else k))


def _frontier_by_depth(best_by_ell):
    """{ell: acc} (tree level) -> {d: acc} where bottom-up depth d = L - ell."""
    return {L - int(ell): float(acc) for ell, acc in best_by_ell.items()}


@app.function(volumes={DATA_DIR: volume}, timeout=36000, memory=16384)
def occupancy_frontier(n_steps: int = N_STEPS, n_tokens: int = N_TOKENS):
    print("=== NTP frontier vs occupancy ===")
    print(f"  model={N_LAYER}L/{N_HEAD}H/{N_EMBD}D  s={S} L={L}  "
          f"n_steps={n_steps}  n_tokens={n_tokens:,}")
    print("  settings (v,m)->occ: " +
          ", ".join(f"v{v}m{m}:{m / v ** (S - 1):.4f}" for v, m in SETTINGS))

    handles = []
    for v, m in SETTINGS:
        h = thread_b.spawn(
            v=v, s=S, depth=L, m=m, n_tokens=n_tokens,
            n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
            n_steps=n_steps, ckpt_steps=CKPT_STEPS,
        )
        handles.append((v, m, h))
        print(f"  spawned v{v} m{m}  (occ {m / v ** (S - 1):.4f})")

    settings_out = []
    for v, m, h in handles:
        try:
            res = h.get()
        except Exception as e:
            print(f"  v{v}m{m}: FAILED -- {e}")
            settings_out.append({"v": v, "m": m, "occ": m / v ** (S - 1),
                                 "error": str(e)})
            continue
        ckpts = res["checkpoints"]
        final = max(int(k) for k in ckpts.keys())
        rec = _kv(ckpts, final)
        lin = _frontier_by_depth(rec["linear_best"])
        mlp = _frontier_by_depth(rec["mlp_best"]) if "mlp_best" in rec else {}
        # trajectory of best-by-depth (linear) over checkpoints -> climbing check
        traj = {int(step): _frontier_by_depth(_kv(ckpts, step)["linear_best"])
                for step in ckpts}
        settings_out.append({
            "v": v, "m": m, "occ": m / v ** (S - 1), "chance": 1.0 / v,
            "final_step": final, "linear_best": lin, "mlp_best": mlp,
            "trajectory_linear": traj,
        })

    # ---- summary tables ----
    def _print_table(kind):
        print(f"\n{'='*78}\nFINAL FRONTIER [{kind}]  (acc of true latent at depth d; "
              f"d1=near leaves, d6=root)\n{'='*78}")
        print(f"{'setting':>9} | {'occ':>6} | {'chance':>6} |  " +
              "  ".join(f"d{d}" for d in range(1, L + 1)))
        for so in settings_out:
            if "error" in so:
                print(f"{'v'+str(so['v'])+'m'+str(so['m']):>9} | "
                      f"{so['occ']:>6.4f} |   ---  | FAILED: {so['error']}")
                continue
            tab = so[kind]
            cells = "  ".join(f"{tab.get(d, float('nan')):.2f}" for d in range(1, L + 1))
            print(f"{'v'+str(so['v'])+'m'+str(so['m']):>9} | {so['occ']:>6.4f} | "
                  f"{so['chance']:>6.4f} |  {cells}")

    _print_table("linear_best")
    _print_table("mlp_best")

    out = {
        "config": dict(settings=SETTINGS, s=S, L=L, n_layer=N_LAYER, n_head=N_HEAD,
                       n_embd=N_EMBD, n_steps=n_steps, n_tokens=n_tokens),
        "settings": settings_out,
    }
    save_dir = f"{DATA_DIR}/rhm_occupancy_frontier"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"occupancy_frontier_{N_LAYER}L{N_HEAD}H{N_EMBD}D_S{n_steps}.json"
    with open(os.path.join(save_dir, fname), "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/{fname}")
    return out
