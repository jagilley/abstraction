"""Modal harness that runs the organizers' own evaluator, faithfully, on an H100.

Deliberately **separate** from the research node's `one_layer_deeper/shared.py`: its own app
(`old-dress-rehearsal`) and its own volume (`old-dress-rehearsal-data`), so nothing here can
collide with `one-layer-deeper-data`, which holds every prior cut's results.

The image installs the upstream repo at its own pinned commit and pinned dependencies, and
the GPU function shells out to exactly the command upstream documents:

    python -m benchmark.runner --manifest <path> --submission-file <path>

with `CUDA_VISIBLE_DEVICES=0`, capturing the final `RESULT_JSON=` line and the
`seed=... profile=... depth_t=...` lines the depth profile prints. Nothing about the
evaluator is reimplemented; a faithful wall-clock number is the entire point.

Volume layout:

    /data/generated/<dataset>/          upstream-format dataset (train/test/ood/depth_*)
    /data/audit/<dataset>.json          audit_dataset.py report for that dataset
    /data/results/<tag>/result.json     RESULT_JSON payload plus the harness's own record
    /data/results/<tag>/runner.log      full evaluator stdout+stderr

Commands (from `experiments/`):

    MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::generate_public
    MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::generate_hard
    MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::audit
    MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/modal_rehearsal.py::evaluate \
        --dataset e1 --arm control --seconds 60 --tag smoke_e1_control

Anything past a couple of minutes wants `--detach`, per `/run-experiment-on-modal`.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

UPSTREAM_URL = "https://github.com/tilde-research/one-layer-deeper.git"
UPSTREAM_COMMIT = "4ceff95"
UPSTREAM_DIR = "/opt/one-layer-deeper"
DRESS_DIR = "/opt/dress"
DATA_DIR = "/data"

_LOCAL_DRESS = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install(
        # Upstream's `pyproject.toml` pins at 4ceff95. Only these three are reachable from
        # `benchmark.runner` and `data.squaring_mod`; the service/CLI extras are not.
        "torch==2.12.1",
        "numpy==2.5.0",
        "jsonargparse==4.49.0",
    )
    .run_commands(
        f"git clone {UPSTREAM_URL} {UPSTREAM_DIR}",
        f"cd {UPSTREAM_DIR} && git checkout {UPSTREAM_COMMIT}",
    )
    .add_local_dir(str(_LOCAL_DRESS), DRESS_DIR)
)

volume = modal.Volume.from_name("old-dress-rehearsal-data", create_if_missing=True)
app = modal.App("old-dress-rehearsal", image=image)


def _link_generated() -> None:
    """Point the upstream checkout's `data/generated` at the volume.

    `scripts/generate_datasets.sh` writes to that relative path, so a symlink is the only
    way to run the organizers' generation script unmodified.
    """

    import os

    target = Path(DATA_DIR) / "generated"
    target.mkdir(parents=True, exist_ok=True)
    link = Path(UPSTREAM_DIR) / "data" / "generated"
    if link.is_symlink() or link.exists():
        return
    os.symlink(target, link)


def _dress_module(name: str):
    import importlib
    import sys

    if DRESS_DIR not in sys.path:
        sys.path.insert(0, DRESS_DIR)
    return importlib.import_module(name)


@app.function(volumes={DATA_DIR: volume}, timeout=7200, cpu=8.0, memory=16384)
def generate_public() -> list[str]:
    """Run upstream's own `scripts/generate_datasets.sh` verbatim onto the volume."""

    import subprocess

    _link_generated()
    subprocess.run(
        ["bash", "scripts/generate_datasets.sh"],
        cwd=UPSTREAM_DIR,
        check=True,
    )
    volume.commit()
    return sorted(p.name for p in (Path(DATA_DIR) / "generated").iterdir())


@app.function(volumes={DATA_DIR: volume}, timeout=7200, cpu=8.0, memory=16384)
def generate_hard(configs: str = "") -> list[str]:
    """Generate the plausible-Hard datasets from `hard_configs.HARD_CONFIGS`."""

    import subprocess

    hard_configs = _dress_module("hard_configs")
    _link_generated()
    names = [c for c in configs.split(",") if c] or list(hard_configs.HARD_CONFIGS)
    written = []
    for name in names:
        output_dir = f"{DATA_DIR}/generated/{hard_configs.dataset_dirname(name)}"
        subprocess.run(
            [
                "python",
                "-m",
                "data.squaring_mod",
                *hard_configs.generation_argv(name, output_dir),
            ],
            cwd=UPSTREAM_DIR,
            check=True,
        )
        written.append(output_dir)
    volume.commit()
    return written


@app.function(volumes={DATA_DIR: volume}, timeout=7200, cpu=8.0, memory=32768)
def audit(datasets: str = "") -> str:
    """Audit every dataset on the volume: moduli, coverage, per-rung floor."""

    audit_dataset = _dress_module("audit_dataset")
    root = Path(DATA_DIR) / "generated"
    names = [d for d in datasets.split(",") if d] or sorted(p.name for p in root.iterdir())
    out_dir = Path(DATA_DIR) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in names:
        report = audit_dataset.audit(root / name)
        (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        summary = audit_dataset.summarize(report)
        print(summary, flush=True)
        lines.append(summary)
    volume.commit()
    return "\n".join(lines)


def _run_evaluation(
    dataset: str,
    arm: str,
    seconds: float,
    tag: str,
    batch_size: int,
    compile_model: bool,
) -> dict:
    """One faithful evaluator run: `python -m benchmark.runner` on an H100.

    `dataset` is either a public alias (`e1`, `e5`, `m5`) or a plausible-Hard config name
    (`hB_width_down`, ...). `arm` is `control`, `closure`, or `baseline` (upstream's own
    `submissions/baseline_adamw/submission.py`, unmodified). `seconds` is the manifest's
    `total_training_time_seconds`; the evaluator gives evaluation half of it on top.
    """

    import os
    import re
    import subprocess
    import time

    hard_configs = _dress_module("hard_configs")
    emit_arm = _dress_module("emit_arm")

    if dataset in hard_configs.PUBLIC_DATA_ROOTS:
        dirname = hard_configs.PUBLIC_DATA_ROOTS[dataset]
    elif dataset in hard_configs.HARD_CONFIGS:
        dirname = hard_configs.dataset_dirname(dataset)
    else:
        dirname = dataset
    data_root = f"{DATA_DIR}/generated/{dirname}"
    if not Path(data_root).is_dir():
        raise FileNotFoundError(
            f"{data_root} is not on the volume; run generate_public / generate_hard first"
        )

    tag = tag or f"{dataset}_{arm}_{int(seconds)}s"
    work = Path("/tmp/rehearsal")
    work.mkdir(parents=True, exist_ok=True)
    if arm == "baseline":
        # Upstream's own reference submission, unmodified, as the calibration point.
        submission_path = Path(UPSTREAM_DIR) / "submissions" / "baseline_adamw" / "submission.py"
    else:
        submission_path = emit_arm.emit(arm, Path(DRESS_DIR) / "submission.py", work)
    manifest_path = work / "manifest.json"
    manifest = hard_configs.manifest(
        name=f"dress-{tag}",
        data_root=data_root,
        training_seconds=seconds,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        compile_model=compile_model,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    started = time.monotonic()
    process = subprocess.run(
        [
            "python",
            "-m",
            "benchmark.runner",
            "--manifest",
            str(manifest_path),
            "--submission-file",
            str(submission_path),
        ],
        cwd=UPSTREAM_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    log = process.stdout + "\n----- stderr -----\n" + process.stderr

    result = None
    for line in process.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            result = json.loads(line[len("RESULT_JSON=") :])
    depth_lines = [
        line for line in process.stdout.splitlines() if re.match(r"^seed=\d+ profile=", line)
    ]

    record = {
        "tag": tag,
        "dataset": dataset,
        "data_root": data_root,
        "arm": arm,
        "training_seconds_requested": seconds,
        "batch_size": batch_size,
        "compile": compile_model,
        "wall_clock_seconds": elapsed,
        "returncode": process.returncode,
        "manifest": manifest,
        "depth_lines": depth_lines,
        "result": result,
    }

    out_dir = Path(DATA_DIR) / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    (out_dir / "runner.log").write_text(log)
    volume.commit()

    print("\n".join(depth_lines[-32:]), flush=True)
    if result is None:
        print(log[-6000:], flush=True)
        raise RuntimeError(f"runner failed (returncode {process.returncode}); see runner.log")
    print(json.dumps(record["result"]["score"], indent=2), flush=True)
    return record


@app.function(volumes={DATA_DIR: volume}, gpu="H100", timeout=7200, memory=65536)
def evaluate(
    dataset: str,
    arm: str = "control",
    seconds: float = 60.0,
    tag: str = "",
    batch_size: int = 512,
    compile_model: bool = False,
) -> dict:
    """Tier-faithful evaluation on an H100 — the hardware the tiers are defined on."""

    return _run_evaluation(dataset, arm, seconds, tag, batch_size, compile_model)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def evaluate_l4(
    dataset: str,
    arm: str = "control",
    seconds: float = 120.0,
    tag: str = "",
    batch_size: int = 512,
    compile_model: bool = False,
) -> dict:
    """Cheap smoke on an L4 — for checking that a submission *trains*, not for scoring.

    Wall-clock numbers from here are not tier-faithful and must never be compared against
    a 60/600/3600 s tier budget.
    """

    return _run_evaluation(dataset, arm, seconds, tag or f"l4_{dataset}_{arm}", batch_size, compile_model)


@app.function(volumes={DATA_DIR: volume}, timeout=1800)
def collect(prefix: str = "") -> str:
    """Reduce every result on the volume to one table."""

    root = Path(DATA_DIR) / "results"
    if not root.is_dir():
        return "no results"
    rows = []
    for path in sorted(root.glob("*/result.json")):
        record = json.loads(path.read_text())
        if prefix and not record["tag"].startswith(prefix):
            continue
        result = record.get("result") or {}
        seeds = result.get("seeds") or [{}]
        # Per-rung detail lives on the seed, not on the aggregated top-level profile,
        # which carries only the ladder and the certified depths.
        profile = seeds[0].get("depth_profile") or {}
        score = result.get("score") or {}
        rows.append(
            {
                "tag": record["tag"],
                "dataset": record["dataset"],
                "arm": record["arm"],
                "seconds": record["training_seconds_requested"],
                "steps": seeds[0].get("completed_training_steps"),
                "training_seconds": seeds[0].get("training_seconds"),
                "model_state_elements": seeds[0].get("model_state_elements"),
                "evaluation_seconds": seeds[0].get("evaluation_seconds"),
                "splits": {
                    name: [metrics["correct_examples"], metrics["example_count"]]
                    for name, metrics in (seeds[0].get("evaluation") or {}).items()
                },
                "mean_exact_accuracy": score.get("mean_exact_accuracy"),
                "max_certified_time_steps": profile.get("max_certified_time_steps"),
                "ood_n_max_certified_time_steps": profile.get(
                    "ood_n_max_certified_time_steps"
                ),
                "rungs": [
                    (r["time_steps"], r["correct_examples"], r["example_count"])
                    for r in profile.get("rungs", [])
                ],
                "ood_n_rungs": [
                    (r["time_steps"], r["correct_examples"], r["example_count"])
                    for r in profile.get("ood_n_rungs", [])
                ],
            }
        )
    text = json.dumps(rows, indent=2)
    print(text, flush=True)
    return text
