"""Local CPU checks on the digit port's carrier, run against real generated prompts.

Nothing here is part of a submission; it exists so a defect in the layout parse, the
right-aligned gather, the scatter, or the near-identity initialisation is caught before a
hosted run is spent on it.

    .venv/bin/python <node>/second_pass/digit_port/probe.py --data-root data/generated/<name>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location("digit_submission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["digit_submission"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument(
        "--submission",
        default=str(Path(__file__).with_name("submission.py")),
    )
    args = parser.parse_args()

    from data.squaring_mod import (  # noqa: E402  (upstream checkout on sys.path)
        DIGIT_OFFSET,
        collate_squaring_mod,
        load_squaring_mod_dataset_config,
    )

    root = Path(args.data_root)
    config = load_squaring_mod_dataset_config(root)
    rows = []
    with (root / f"{args.split}.jsonl").open() as handle:
        for line, _ in zip(handle, range(args.rows)):
            rows.append(json.loads(line))
    batch = collate_squaring_mod(rows)

    module = load_submission(Path(args.submission))
    spec = module.ModelSpec(
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        maximum_model_state_elements=500_000_000,
    )
    model = module.build_model(spec)
    model.eval()

    input_ids = batch["input_ids"]
    key_padding = batch["attention_mask"].bool()
    layout = model._layout(input_ids, key_padding)
    steps = model._loop_count(input_ids, key_padding)

    x_grid = model._grid_tokens(input_ids, layout["t_at"], layout["x_width"])
    n_grid = model._grid_tokens(input_ids, layout["x_at"], layout["n_width"])

    def grid_value(grid: torch.Tensor) -> list[int]:
        digits = (grid - DIGIT_OFFSET).clamp(0, 9)
        out = []
        for row in digits:
            value = 0
            for slot in range(row.shape[0] - 1, -1, -1):
                value = value * 10 + int(row[slot])
            out.append(value)
        return out

    bad = 0
    for index, record in enumerate(rows):
        want_n, want_x, want_t = (
            int(record["modulus"]),
            int(record["x"]),
            int(record["time_steps"]),
        )
        got_n = grid_value(n_grid[index : index + 1])[0]
        got_x = grid_value(x_grid[index : index + 1])[0]
        got_t = int(steps[index])
        if (got_n, got_x, got_t) != (want_n, want_x, want_t):
            bad += 1
            if bad <= 5:
                print(
                    f"row {index}: got (N={got_n}, x={got_x}, T={got_t}) "
                    f"want (N={want_n}, x={want_x}, T={want_t})"
                )
    print(f"layout/gather: {len(rows) - bad}/{len(rows)} rows exact")

    # The scatter must place slot j where `target_positions` reads it.
    with torch.no_grad():
        logits, auxiliary = model(input_ids, attention_mask=key_padding)
    grid = auxiliary["grid"]
    target_positions = batch["target_positions"]
    labels = batch["labels"]
    batch_index = torch.arange(logits.shape[0])[:, None]
    gathered = logits[batch_index, target_positions.clamp_min(0)]
    counts = (labels != -100).sum(dim=1)
    slot_index = (counts[:, None] - 1 - torch.arange(labels.shape[1])[None, :]).clamp_min(0)
    from_grid = grid.gather(1, slot_index[:, :, None].expand(-1, -1, grid.shape[-1]))
    valid = labels != -100
    agree = ((gathered - from_grid).abs().amax(dim=-1) * valid).max()
    print(f"scatter round-trip max |delta| on valid targets: {float(agree):.3e}")

    # Near-identity at initialisation: one operator application should still decode the
    # digits it was handed, so a deep tied rollout does not wash the grid out untrained.
    predicted = grid.argmax(dim=-1)
    copy_match = (predicted == x_grid).float().mean()
    print(f"init copy fidelity (grid argmax == x digits): {float(copy_match):.3f}")
    print(f"slots={model.slots} steps(min,max)=({int(steps.min())},{int(steps.max())}) "
          f"state_elements={module.count_state(model) if hasattr(module, 'count_state') else sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
