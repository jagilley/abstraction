"""Audit a generated upstream squaring-mod dataset before spending an evaluation on it.

The three quantities a Hard-shaped guess turns on, none of which the manifest states:

1. **How many distinct moduli the training split actually contains**, per configured
   bit width. The generator draws `(p, q)` fresh for every record from *all* primes of
   half-width, so the modulus population is bounded by arithmetic, not by
   `examples_per_setting`: 6 / 21 / 14 / 57 / 42 / 169 / 148 valid RSA moduli at
   10 / 11 / 12 / 13 / 14 / 15 / 16 bits. A <=12-bit cell is therefore necessarily dense
   per modulus and a 16-bit cell is sparse, which is the axis
   `../variable_modulus/` and `../rule_acquisition/` say governs whether anything is
   learned at all.

2. **Per-modulus base coverage** — distinct `x` seen in training divided by `phi(N)`.
   `variable_modulus/` measures the re-projected rollout's per-restart rate as
   `p ~= coverage`, so this is the number that predicts whether a fixed-`N`-style
   mechanism can show up at all.

3. **The analytic no-rule floor at every ladder rung** — how many depth-profile prompts
   have `x^(2^T) < N`, i.e. never reduce and so are answerable by squaring alone with no
   knowledge of the modulus. A submission scoring at this count has learned nothing about
   the rule; without it, a small non-zero leaderboard number is uninterpretable.

Also reported: the **periodicity leak** (how many prompts at rung `T` have the same answer
as some lower rung for the same `(N, x)` — those rungs are free once a lower one is
solved) and the exact-accuracy each rung would show at floor.

Usage:
    python audit_dataset.py <data_root> [<data_root> ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


LADDER_PREFIXES = ("depth_t_", "depth_ood_n_t_")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def factor_semiprime(modulus: int) -> tuple[int, int]:
    for p in range(2, math.isqrt(modulus) + 1):
        if modulus % p == 0:
            return p, modulus // p
    raise ValueError(f"{modulus} is not a semiprime")


def totient_semiprime(modulus: int) -> int:
    p, q = factor_semiprime(modulus)
    return (p - 1) * (q - 1)


def carmichael_semiprime(modulus: int) -> int:
    p, q = factor_semiprime(modulus)
    return math.lcm(p - 1, q - 1)


def depth_first_repeat(modulus: int) -> int:
    """Smallest `T2 > T1 >= 0` collision of `2^T mod lambda(N)`: tail + period.

    Any evaluation ladder reaching this depth admits a periodicity shortcut instead of a
    serial rollout, for *every* base at once.
    """

    lam = carmichael_semiprime(modulus)
    seen: dict[int, int] = {}
    value = 1 % lam
    step = 0
    while value not in seen:
        seen[value] = step
        value = (2 * value) % lam
        step += 1
    return step


def no_reduction(x: int, time_steps: int, modulus: int) -> bool:
    """True when `x^(2^T) < N`, so the answer never needed a modular reduction."""

    value = x
    for _ in range(time_steps):
        value *= value
        if value >= modulus:
            return False
    return True


def audit(root: Path) -> dict:
    config = json.loads((root / "dataset_config.json").read_text())
    generator = config["generator_config"]
    report: dict = {
        "data_root": str(root),
        "max_seq_len": config["max_seq_len"],
        "vocab_size": config["vocab_size"],
        "num_examples": config["num_examples"],
        "split_counts": config["split_counts"],
        "generator_config": generator,
    }

    train = read_jsonl(root / "train.jsonl")
    by_bits: dict[object, set[int]] = defaultdict(set)
    bases: dict[int, set[int]] = defaultdict(set)
    for record in train:
        by_bits[record["configured_modulus_bits"]].add(record["modulus"])
        bases[record["modulus"]].add(record["x"])

    moduli_report = {}
    for bits, moduli in sorted(by_bits.items(), key=lambda item: (item[0] is None, item[0])):
        coverages = [len(bases[m]) / totient_semiprime(m) for m in moduli]
        repeats = [depth_first_repeat(m) for m in moduli]
        moduli_report[str(bits)] = {
            "distinct_training_moduli": len(moduli),
            "train_rows": sum(1 for r in train if r["configured_modulus_bits"] == bits),
            "base_coverage_mean": sum(coverages) / len(coverages),
            "base_coverage_min": min(coverages),
            "base_coverage_max": max(coverages),
            "depth_first_repeat_min": min(repeats),
            "depth_first_repeat_median": sorted(repeats)[len(repeats) // 2],
            "moduli_below_ladder_max": sum(1 for r in repeats if r <= 64),
        }
    report["training_moduli"] = moduli_report

    train_moduli = {r["modulus"] for r in train}
    for prefix in LADDER_PREFIXES:
        splits = sorted(
            (p for p in root.glob(f"{prefix}*.jsonl")),
            key=lambda p: int(p.stem.removeprefix(prefix)),
        )
        if not splits:
            continue
        answers: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        rungs = []
        for path in splits:
            time_steps = int(path.stem.removeprefix(prefix))
            records = read_jsonl(path)
            for record in records:
                answers[(record["modulus"], record["x"])][time_steps] = record["result"]
            free = sum(
                1 for r in records if no_reduction(r["x"], time_steps, r["modulus"])
            )
            rungs.append(
                {
                    "time_steps": time_steps,
                    "example_count": len(records),
                    "distinct_moduli": len({r["modulus"] for r in records}),
                    "moduli_seen_in_training": len(
                        {r["modulus"] for r in records} & train_moduli
                    ),
                    "no_reduction_examples": free,
                    "no_reduction_fraction": free / len(records),
                    "answer_is_one": sum(1 for r in records if r["result"] == 1),
                }
            )
        ladder = [rung["time_steps"] for rung in rungs]
        for index, rung in enumerate(rungs):
            lower = ladder[:index]
            if not lower:
                rung["repeats_lower_rung"] = 0
                continue
            repeats = 0
            for values in answers.values():
                current = values.get(rung["time_steps"])
                if current is None:
                    continue
                if any(values.get(t) == current for t in lower):
                    repeats += 1
            rung["repeats_lower_rung"] = repeats
        report[f"ladder_{prefix.rstrip('_')}"] = rungs
    return report


def summarize(report: dict) -> str:
    lines = [f"== {report['data_root']}"]
    generator = report["generator_config"]
    lines.append(
        f"   modulus_bits={generator['modulus_bits']} fixed_p={generator['fixed_p']} "
        f"time_steps={generator['time_steps'] or generator['fixed_time_steps']} "
        f"ood_time_steps={generator['ood_time_steps']} "
        f"examples_per_setting={generator['examples_per_setting']} "
        f"max_seq_len={report['max_seq_len']}"
    )
    lines.append("   training moduli:")
    for bits, stats in report["training_moduli"].items():
        lines.append(
            f"     {bits:>4} bits: {stats['distinct_training_moduli']:>5} moduli  "
            f"rows={stats['train_rows']:>7}  coverage "
            f"mean={stats['base_coverage_mean']:.3f} "
            f"min={stats['base_coverage_min']:.3f} "
            f"max={stats['base_coverage_max']:.3f}  "
            f"first_depth_repeat median={stats['depth_first_repeat_median']} "
            f"(<=64 for {stats['moduli_below_ladder_max']})"
        )
    for key in ("ladder_depth_t", "ladder_depth_ood_n_t"):
        if key not in report:
            continue
        lines.append(f"   {key}:")
        for rung in report[key]:
            lines.append(
                f"     T={rung['time_steps']:>2}  n={rung['example_count']:>4}  "
                f"moduli={rung['distinct_moduli']:>4} "
                f"(seen in train {rung['moduli_seen_in_training']:>4})  "
                f"floor={rung['no_reduction_examples']:>4} "
                f"({rung['no_reduction_fraction']:.4f})  "
                f"y==1: {rung['answer_is_one']:>3}  "
                f"repeats_lower_rung={rung['repeats_lower_rung']:>4}"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_roots", nargs="+")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    reports = []
    for root in args.data_roots:
        report = audit(Path(root))
        reports.append(report)
        print(summarize(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
