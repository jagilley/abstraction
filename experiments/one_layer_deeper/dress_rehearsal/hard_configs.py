"""Dataset and manifest configs for the dress rehearsal.

Hard's dataset `h1` is hidden. What we know about its *shape* is indirect:

- The live leaderboard's ranks 2-6 all sit at exactly **6/768** on the seen-`N` `T=1`
  rung and **3/768** on the OOD-`N` one. `audit_dataset.py` measures the public `m5`
  (`--modulus_bits [12,14,16] --time_steps [2,4,8]`) seen-`N` `T=1` no-reduction count as
  **6/768** — an exact match — and its OOD-`N` count as 11/768. So those ranks are the
  analytic no-rule floor, and Hard's seen-`N` widths are `m5`-like while its OOD-`N`
  widths are larger than `m5`'s `[13,15,18]`. Small integer counts, so this is
  suggestive, not established.
- 768 = 3 widths x 256, the public generator's `depth_evaluation_examples_per_setting`
  pattern for sampled-`N` datasets.

So `m5` is the **anchor**, and each config below moves exactly *one* knob off it. That is
the point: the rehearsal is a calibration over a neighbourhood of plausible shapes, not a
guess at one.

    anchor  m5      bits [12,14,16]  T [2,4,8]   10k/setting  ood_n [13,15,18]
    hB      width-  bits [11,13,15]  T [2,4,8]   10k/setting  ood_n [12,14,16]
    hC      width+  bits [13,15,17]  T [2,4,8]   10k/setting  ood_n [14,16,18]
    hD      T shallow (T=1 trained)  T [1,2,4]   10k/setting  ood_n [13,15,18]
    hE      T deep                   T [4,8,16]  10k/setting  ood_n [13,15,18]
    hF      2.5x data                T [2,4,8]   25k/setting  ood_n [13,15,18]

**`hF` is capped by arithmetic, not by taste.** There are only 14 valid 12-bit RSA moduli
and 35,624 `(N, x)` unit pairs among them. Each in-distribution `T` cell needs
`examples_per_setting` *distinct* pairs, and the depth-profile cohort then needs 256 more
that appear in no split at all, so `examples_per_setting` at 12 bits cannot exceed about
28,000 -- a `[12,14,16]` dataset at 30k/setting fails generation outright with
"could not sample a fresh seen-N depth pair". At `m5`'s own 10k/setting the training split
already holds ~84% of every possible 12-bit prompt. A Hard set with substantially more
data than `m5` therefore cannot have a 12-bit cell.

`hD` matters most for the metric that ranks: with `m5`'s `[2,4,8]` the ladder's first rung
`T=1` is a *downward* extrapolation that was never trained. `hB` matters most for the
mechanism: the generator draws `(p,q)` fresh per record from all half-width primes, so an
11-bit cell has only 21 possible moduli and is dense per modulus, where 16 bits has 148
and is sparse.

The generator is seeded (`--seed 45`, upstream's value), so a local generation and a Modal
generation of the same config produce byte-identical data.
"""

from __future__ import annotations

# `ood_n_depth_evaluation_modulus_bits` may not intersect the in-distribution
# `modulus_bits` -- the generator asserts it. Every OOD-N profile below is therefore
# offset by one bit per width rather than sitting on top of an ID width.
HARD_CONFIGS: dict[str, dict] = {
    "hB_width_down": dict(
        modulus_bits=[11, 13, 15],
        time_steps=[2, 4, 8],
        ood_time_steps=[16],
        examples_per_setting=10000,
        ood_examples_per_setting=1000,
        ood_n_depth_evaluation_modulus_bits=[12, 14, 16],
    ),
    "hC_width_up": dict(
        modulus_bits=[13, 15, 17],
        time_steps=[2, 4, 8],
        ood_time_steps=[16],
        examples_per_setting=10000,
        ood_examples_per_setting=1000,
        ood_n_depth_evaluation_modulus_bits=[14, 16, 18],
    ),
    "hD_t_shallow": dict(
        modulus_bits=[12, 14, 16],
        time_steps=[1, 2, 4],
        ood_time_steps=[8],
        examples_per_setting=10000,
        ood_examples_per_setting=1000,
        ood_n_depth_evaluation_modulus_bits=[13, 15, 18],
    ),
    "hE_t_deep": dict(
        modulus_bits=[12, 14, 16],
        time_steps=[4, 8, 16],
        ood_time_steps=[32],
        examples_per_setting=10000,
        ood_examples_per_setting=1000,
        ood_n_depth_evaluation_modulus_bits=[13, 15, 18],
    ),
    "hF_data_25x": dict(
        modulus_bits=[12, 14, 16],
        time_steps=[2, 4, 8],
        ood_time_steps=[16],
        examples_per_setting=25000,
        ood_examples_per_setting=2500,
        ood_n_depth_evaluation_modulus_bits=[13, 15, 18],
    ),
}

# Constant across every config: upstream's ladder, its 256-per-width depth cohorts, its
# 0.9/0.1 Medium/Hard split fractions, its seed, and the separate-input/output format.
COMMON = dict(
    depth_evaluation_time_steps=[1, 2, 4, 8, 16, 32, 64],
    depth_evaluation_examples_per_setting=256,
    ood_n_depth_evaluation_examples_per_setting=256,
    train_fraction=0.9,
    test_fraction=0.1,
    split_group="prompt",
    seed=45,
    separate_input_output=True,
)

# Public datasets we mirror so the rehearsal has calibrated reference points: `e1` is the
# Easy tier verbatim, `m5` is the Hard anchor's public twin.
PUBLIC_DATA_ROOTS = {
    "e5": "squaring_mod_new11_easy_bidirectional_variable_b1011_t123",
    "m5": "squaring_mod_new11_medium_bidirectional_variable_b121416_t248",
    # The granular fixed-N Medium rungs. These use `depth_evaluation_exhaustive_x`, so the
    # seen-N depth cohort is *every remaining unused unit* of the one modulus — a direct
    # fresh-x readout on a modulus the model trained on, which is `ballistic_depth/`'s
    # held-out-x axis in the competition's own frame.
    "m6": "squaring_mod_granular_medium_m6_bidirectional_fixed_n_1517_t124",
    "m7": "squaring_mod_granular_medium_m7_bidirectional_fixed_n_1763_t124",
    "m8": "squaring_mod_granular_medium_m8_bidirectional_fixed_n_1333_t124",
    "m9": "squaring_mod_granular_medium_m9_bidirectional_fixed_n_1927_t124",
    "m10": "squaring_mod_granular_medium_m10_bidirectional_fixed_n_1739_t124",
    "e1": "squaring_mod_new11_easy_bidirectional_fixed_n_323_t123",
}


def dataset_dirname(name: str) -> str:
    return f"squaring_mod_dress_{name}"


def generation_argv(name: str, output_dir: str) -> list[str]:
    """Upstream `python -m data.squaring_mod` argv for one plausible-Hard config."""

    import json

    config = HARD_CONFIGS[name]
    argv = ["--output_dir", output_dir]
    for key, value in {**config, **COMMON}.items():
        if isinstance(value, list):
            argv += [f"--{key}", json.dumps(value)]
        elif isinstance(value, bool):
            argv += [f"--{key}", "true" if value else "false"]
        else:
            argv += [f"--{key}", str(value)]
    return argv


def manifest(
    *,
    name: str,
    data_root: str,
    training_seconds: float,
    batch_size: int = 512,
    eval_batch_size: int = 512,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    amp: bool = True,
    compile_model: bool = False,
    max_steps: int = 1_000_000,
    seeds: tuple[int, ...] = (74,),
    log_every: int = 100,
) -> dict:
    """An evaluator manifest in upstream's exact shape.

    Every field mirrors `benchmark/manifests/h100_medium_m5.json` except `data_root` and
    `total_training_time_seconds`. Manifest overrides are forbidden to *participants*
    (rule 14); this is evaluator-side configuration for a local rehearsal, which is what
    upstream's own `scripts/` do.
    """

    return {
        "name": name,
        "data": {
            "kind": "squaring_mod",
            "data_root": data_root,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "shuffle_train": True,
            "shuffle_eval": False,
            "num_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "seed": 45,
        },
        "runtime": {
            "device": device,
            "dtype": dtype,
            "amp": amp,
            "compile": compile_model,
            "total_training_time_seconds": training_seconds,
            "max_steps": max_steps,
            "seeds": list(seeds),
            "grad_clip": 1,
            "log_every": log_every,
        },
        "model_state": {"maximum_elements": 500000000},
    }


def cpu_smoke_manifest(training_seconds: float = 30.0) -> dict:
    """Upstream's `smoke_cpu.json` with a budget a 64-rung rollout can actually finish.

    `benchmark/manifests/smoke_cpu.json` allows 0.1 training seconds and half that for
    evaluation, which is enough for a one-block transformer and not for a deep unrolled
    one. Everything else is upstream's: `data_root: null` builds the deterministic
    `N=143` smoke dataset in a temp dir.
    """

    return {
        "name": "squaring-mod-cpu-smoke-dress",
        "data": {
            "kind": "squaring_mod",
            "data_root": None,
            "batch_size": 32,
            "eval_batch_size": 64,
            "shuffle_train": True,
            "shuffle_eval": False,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
            "seed": 45,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "amp": False,
            "compile": False,
            "total_training_time_seconds": training_seconds,
            "max_steps": 1000,
            "seeds": [74],
            "grad_clip": 1.0,
            "log_every": 1,
        },
        "model_state": {"maximum_elements": 500000000},
    }


def main() -> None:
    """Generate the plausible-Hard datasets, or emit a manifest, under a local checkout."""

    import argparse
    import json
    import subprocess
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write the plausible-Hard datasets")
    generate.add_argument(
        "--upstream",
        default="~/Code/one-layer-deeper-head",
        help="upstream checkout at 4ceff95 with its own uv venv",
    )
    generate.add_argument("--configs", default=",".join(HARD_CONFIGS))

    emit = sub.add_parser("manifest", help="write one evaluator manifest to a path")
    emit.add_argument("out")
    emit.add_argument("--data-root", default=None)
    emit.add_argument("--seconds", type=float, default=60.0)
    emit.add_argument("--name", default="dress-rehearsal")
    emit.add_argument("--cpu", action="store_true")
    emit.add_argument("--batch-size", type=int, default=512)

    args = parser.parse_args()

    if args.command == "manifest":
        if args.cpu:
            payload = cpu_smoke_manifest(args.seconds)
        else:
            payload = manifest(
                name=args.name,
                data_root=args.data_root,
                training_seconds=args.seconds,
                batch_size=args.batch_size,
                eval_batch_size=args.batch_size,
            )
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(args.out)
        return

    upstream = Path(args.upstream)
    python = upstream / ".venv" / "bin" / "python"
    for name in [c for c in args.configs.split(",") if c]:
        output_dir = f"data/generated/{dataset_dirname(name)}"
        argv = [str(python), "-m", "data.squaring_mod", *generation_argv(name, output_dir)]
        print("+", " ".join(argv), flush=True)
        subprocess.run(argv, cwd=upstream, check=True, stdout=sys.stdout, stderr=sys.stderr)


if __name__ == "__main__":
    main()
