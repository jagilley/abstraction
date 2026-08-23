"""Reduce a saved `status.json` to the numbers that decide anything.

    python read_run.py results/hosted/medium_m6_digit
    python read_run.py results/hosted/*            # one line-block per run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rungs(profile: dict) -> str:
    return " ".join(str(r["correct_examples"]) for r in profile.get("rungs", []))


def counts(profile: dict) -> str:
    seen = {r["example_count"] for r in profile.get("rungs", [])}
    return "/".join(str(c) for c in sorted(seen)) or "-"


def show(path: Path) -> None:
    status = json.loads((path / "status.json").read_text())
    result = status.get("result") or status
    score = result.get("score", {})
    seeds = result.get("seeds") or []
    print(f"== {path.name}")
    print(f"   mean_exact={score.get('mean_exact_accuracy')} mean_loss={score.get('mean_loss')}")
    for seed in seeds:
        depth = seed.get("depth_profile", {})
        print(
            f"   steps={seed.get('completed_training_steps')} "
            f"train_loss={seed.get('final_train_loss')} "
            f"state={seed.get('model_state_elements')} "
            f"bs={seed.get('training_batch_size')} "
            f"train_s={seed.get('training_seconds')} eval_s={seed.get('evaluation_seconds')}"
        )
        for split, metrics in sorted((seed.get("evaluation") or {}).items()):
            print(
                f"     split {split:10s} {metrics['correct_examples']}/{metrics['example_count']}"
                f"  loss={metrics['loss']:.4f}"
            )
        print(f"     seen-N  rungs {depth.get('ladder')} -> {rungs(depth)}  (of {counts(depth)})")
        ood = {"rungs": depth.get("ood_n_rungs", [])}
        print(f"     OOD-N   rungs {depth.get('ood_n_ladder')} -> {rungs(ood)}  (of {counts(ood)})")
        print(
            f"     MaxT={depth.get('max_certified_time_steps')} "
            f"OOD MaxT={depth.get('ood_n_max_certified_time_steps')}"
        )


def main() -> None:
    for arg in sys.argv[1:]:
        path = Path(arg)
        if (path / "status.json").exists():
            show(path)


if __name__ == "__main__":
    main()
