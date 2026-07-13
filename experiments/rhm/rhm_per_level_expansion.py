"""Per-level loss expansion: bracket the composition ceiling (2026-07-13).

Prior experiment: PER_LEVEL_LOSS_README.md (rhm_per_level_loss.py).

Motivation
----------
The original per-level-loss experiments showed bottom-up learning and a
composition ceiling on TWO settings that confound model size, data, and m:
  Exp1 (m2, 4L/128D, 5M)  -> ~2-3 levels
  Exp2 (m4, 6L/192D, 20M) -> ~1-2 levels
This experiment turns those anecdotes into two controlled sweeps that share a
pivot, plus a validated "reaches the root" existence proof. Everything uses the
same per-level cross-entropy "wave" metric as the essay, on the same plain
generate_rules pipeline as the original per-level-loss runs.

Design (all L=6, s=2, 20M tokens, 40k steps, width fixed at 256D):
  * DEPTH sweep  (v8, m2, width fixed): n_layer in {2,4,6,8}
      -> does the ceiling climb ~1 level per layer, and does 8L reach the root
         at v8? Isolates capacity (m held at its easy value).
  * SYNONYMITY sweep (v8, 8L/256D fixed): m in {2,4,6,8}
      -> does more synonymy shorten the climb with capacity held fixed and
         generous? m=8 is the "stuck at 1-2 levels" endpoint. Shares the m2
         point with the depth sweep's 8L run.
  * ROOT existence proof (v16, m2, 8L/256D): the one plain-NTP setting validated
      (via latent-ancestor probe = Bayes-optimal) to solve to the root. Here we
      measure whether the per-level-CE wave climbs all the way up.

Prior work (RHM_FRONTIER_AND_LEGIBILITY / RHM_DEEP_COMPOSITION) found the m>=4
ceiling is a learning-SIGNAL limit, not capacity (oracle-aux reaches the root
with the same 8L model; 20k->100k steps does not cross the wall). These sweeps
test that story on the essay's own CE-wave metric.

Run with:
  modal run --detach rhm/rhm_per_level_expansion.py::per_level_expansion
"""

from rhm.shared import volume, DATA_DIR
# per_level_trajectory is defined on the module-specific "rhm-per-level-loss" app
# (NOT the shared "rhm-scaling" app). The orchestrator must live on that SAME app
# so the spawned function is hydrated when the app runs.
from rhm.rhm_per_level_loss import app, per_level_trajectory, _ensure_corpus

N_TOKENS = 20_000_000
N_STEPS = 40_000

# (label, v, m, n_layer, n_head, n_embd)
GRID = [
    # Depth sweep: v8, m2, fixed width 256D
    ("depth_2L", 8, 2, 2, 8, 256),
    ("depth_4L", 8, 2, 4, 8, 256),
    ("depth_6L", 8, 2, 6, 8, 256),
    ("depth_8L", 8, 2, 8, 8, 256),   # == synonymity sweep m2 (shared pivot)
    # Synonymity sweep: v8, 8L/256D fixed (m2 == depth_8L, not repeated)
    ("syn_m4", 8, 4, 8, 8, 256),
    ("syn_m6", 8, 6, 8, 8, 256),
    ("syn_m8", 8, 8, 8, 8, 256),
    # Root existence proof: v16, m2, 8L/256D
    ("root_v16", 16, 2, 8, 8, 256),
]


@app.function(volumes={DATA_DIR: volume}, timeout=36000, memory=8192)
def per_level_expansion():
    volume.reload()

    # Phase 0: ensure every unique corpus exists SEQUENTIALLY, so the parallel
    # training runs never race on corpus generation (several runs share a corpus).
    unique = sorted({(v, m) for _, v, m, *_ in GRID})
    print(f"Ensuring {len(unique)} corpora: {unique}")
    for v, m in unique:
        _ensure_corpus(v, 2, 6, m, N_TOKENS)
    volume.commit()
    print("Corpora ready.\n")

    # Phase 1: spawn all trajectory runs in parallel.
    handles = []
    for label, v, m, nl, nh, nd in GRID:
        model = f"{nl}L/{nh}H/{nd}D"
        h = per_level_trajectory.spawn(
            v=v, s=2, depth=6, m=m, n_tokens=N_TOKENS,
            n_layer=nl, n_head=nh, n_embd=nd,
            n_steps_override=N_STEPS,
        )
        handles.append((label, v, m, model, h))
        print(f"Spawned [{label}] v{v} m{m} {model}")
    print(f"\nSpawned {len(handles)} runs; waiting...\n")

    # Phase 2: collect.
    summary = {}
    for label, v, m, model, h in handles:
        try:
            r = h.get()
            final = r["trajectory"][-1]["per_level"]
            lvls = {int(k): round(final[k]["avg_loss"], 3) for k in final}
            accs = {int(k): round(final[k]["avg_accuracy"], 3) for k in final}
            summary[label] = {
                "v": v, "m": m, "model": model,
                "uniform": round(r["uniform_baseline"], 3),
                "final_loss_by_level": lvls,
                "final_acc_by_level": accs,
            }
            loss_str = "  ".join(f"L{k}={lvls[k]}" for k in sorted(lvls))
            print(f"[{label}] v{v} m{m} {model} (unif {r['uniform_baseline']:.2f}): {loss_str}")
        except Exception as e:
            summary[label] = {"error": str(e)}
            print(f"[{label}] FAILED: {e}")

    print("\n" + "=" * 70)
    print("SUMMARY (final per-level CE):")
    for label, s in summary.items():
        print(f"  {label}: {s}")
    return summary


# Robustness check: does model CAPACITY (depth x width) ever move the v8/m2
# ceiling, or is it purely signal-limited? The main depth sweep held width at a
# generous 256D; here we cross depth {2L,8L} x width {32,64,128} at constrained
# widths. If 8L never beats 2L and the ceiling never rises with width, capacity
# is not the limiter at m=2 -- it is the learning signal.
WIDTH_GRID = [
    ("w32_2L", 8, 2, 2, 4, 32),
    ("w32_8L", 8, 2, 8, 4, 32),
    ("w64_2L", 8, 2, 2, 4, 64),
    ("w64_8L", 8, 2, 8, 4, 64),
    ("w128_2L", 8, 2, 2, 4, 128),
    ("w128_8L", 8, 2, 8, 4, 128),
]


@app.function(volumes={DATA_DIR: volume}, timeout=36000, memory=8192)
def per_level_width_control():
    volume.reload()
    _ensure_corpus(8, 2, 6, 2, N_TOKENS)  # v8/m2 already exists; no-op if present
    volume.commit()

    handles = []
    for label, v, m, nl, nh, nd in WIDTH_GRID:
        h = per_level_trajectory.spawn(
            v=v, s=2, depth=6, m=m, n_tokens=N_TOKENS,
            n_layer=nl, n_head=nh, n_embd=nd, n_steps_override=N_STEPS,
        )
        handles.append((label, f"{nl}L/{nh}H/{nd}D", h))
        print(f"Spawned [{label}] {nl}L/{nh}H/{nd}D")
    print(f"\nSpawned {len(handles)} runs; waiting...\n")

    summary = {}
    for label, model, h in handles:
        try:
            r = h.get()
            final = r["trajectory"][-1]["per_level"]
            accs = {int(k): round(final[k]["avg_accuracy"], 3) for k in final}
            summary[label] = {"model": model, "final_acc_by_level": accs}
            print(f"[{label}] {model}: {accs}")
        except Exception as e:
            summary[label] = {"error": str(e)}
            print(f"[{label}] FAILED: {e}")
    print("\nSUMMARY:", summary)
    return summary
