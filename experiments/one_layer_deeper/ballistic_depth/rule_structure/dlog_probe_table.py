"""Cross-tag summary of `dlog_probe.py`: is there group structure, and does it grow with N?

Pure local re-analysis — reads `results/<tag>/dlog_probe_seed<N>.json` and prints one table.
No GPU, no Modal.

The only reading that matters is against the **positive control**, not against the permutation
null. A permutation null has tiny variance, so a functionally nil effect clears it at z > 30;
the synthetic-group rows say what "found the group" actually scores, and every observed row
should be read as a position on that ladder. The QR column is the sharpest single number: the
sublattice of even `(a, b)` is the reachable set every rollout past `t=1` lives on, so an
encoder that organises anything useful by the group has to organise that.

Results stay in the parent cut's `results/` tree, because they are keyed by the *training*
tag whose checkpoints they were read from — the probe adds no runs of its own.

Usage:
  MODAL_PROFILE=chromatic modal volume get one-layer-deeper-data \\
    /ballistic_depth/<tag>/dlog_probe_seed0.json \\
    one_layer_deeper/ballistic_depth/results/<tag>/dlog_probe_seed0.json --force
  python3 one_layer_deeper/ballistic_depth/rule_structure/dlog_probe_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"


def main() -> None:
    rows, controls = [], {}
    for path in sorted(RESULTS.glob("*/dlog_probe_seed*.json")):
        blob = json.loads(path.read_text())
        tag, seed = blob["tag"], blob["seed"]
        controls.update(blob.get("control_synthetic", {}))
        for arm, arm_res in blob["arms"].items():
            task = arm_res["task"]
            for variant in ("trained", "init"):
                if variant not in arm_res:
                    continue
                r = arm_res[variant]
                rows.append(
                    {
                        "run": f"{tag}/{arm}" + ("" if variant == "trained" else " (init)"),
                        "seed": seed,
                        "N": task["modulus"],
                        "states": (task["group"][0] // 2) * (task["group"][1] // 2),
                        "fourier": r["fourier_top1pct"],
                        "f_null": r["fourier_null_mean"],
                        "qr_z": r["fourier_qr_z"],
                        "trans": r["translation_r2"],
                        "t_null": r["translation_r2_null"],
                        "lin_dec": r["linear_op_decode_heldout"],
                        "op_dec": r["model_op_decode_heldout"],
                    }
                )

    if not rows:
        print(f"no dlog_probe results under {RESULTS}/*/ — fetch them from the volume first")
        return

    hdr = (
        f"{'run':<28} {'seed':>4} {'N':>6} {'states':>7} "
        f"{'fourier':>8} {'(null)':>8} {'qr z':>6} {'transR2':>8} {'(null)':>7} "
        f"{'linDec':>7} {'opDec':>6}"
    )
    print("\n=== observed ===")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["N"], r["run"], r["seed"])):
        print(
            f"{r['run']:<28} {r['seed']:>4} {r['N']:>6} {r['states']:>7} "
            f"{r['fourier']:>8.4f} {r['f_null']:>8.4f} {r['qr_z']:>+6.1f} "
            f"{r['trans']:>8.3f} {r['t_null']:>7.3f} {r['lin_dec']:>7.3f} {r['op_dec']:>6.3f}"
        )

    if controls:
        print("\n=== positive control: what 'found the group' scores ===")
        print(f"{'synthetic':<28} {'fourier':>8} {'transR2':>8}")
        print("-" * 46)
        for name, c in controls.items():
            print(f"{name:<28} {c['fourier_top1pct']:>8.4f} {c['translation_r2']:>8.3f}")
        weakest = min(controls.values(), key=lambda c: c["translation_r2"])
        best_obs = max(
            (r for r in rows if "(init)" not in r["run"]), key=lambda r: r["trans"]
        )
        print(
            f"\nweakest synthetic group signal: transR2 {weakest['translation_r2']:.3f}; "
            f"best observed: {best_obs['trans']:.3f} ({best_obs['run']}) "
            f"-> {weakest['translation_r2'] / max(best_obs['trans'], 1e-9):.1f}x below it"
        )


if __name__ == "__main__":
    main()
