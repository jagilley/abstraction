"""Free full-budget H100 sweeps of the digit port, on the organizers' own evaluator.

The hosted Medium tier is capped at ~6 accepted runs per UTC day and serialised behind one
job per account. But every public dataset is generated locally by upstream's own script, so
a *600-second Medium run* can be reproduced on our own H100 as many times as we like: same
evaluator, same data, same GPU class. Only Hard needs the hosted service, because only
Hard's data is hidden. This file is that instrument.

It reuses `dress_rehearsal`'s volume (`old-dress-rehearsal-data`, which already holds the
public datasets and the plausible-Hard configs) and its manifest builder, but runs under its
own app so nothing here can disturb the parent's records.

`overrides` patches top-level constants in the emitted submission by regex, so one variant
is one dict and the canonical `submission.py` is never edited:

    MODAL_PROFILE=chromatic modal run one_layer_deeper/dress_rehearsal/second_pass/digit_port/modal_sweep.py::list_data
    MODAL_PROFILE=chromatic modal run --detach \
        one_layer_deeper/dress_rehearsal/second_pass/digit_port/modal_sweep.py::sweep
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

UPSTREAM_URL = "https://github.com/tilde-research/one-layer-deeper.git"
UPSTREAM_COMMIT = "4ceff95"
UPSTREAM_DIR = "/opt/one-layer-deeper"
DRESS_DIR = "/opt/dress"
DIGIT_DIR = "/opt/digit"
DATA_DIR = "/data"

_LOCAL_DIGIT = Path(__file__).parent
_LOCAL_DRESS = _LOCAL_DIGIT.parent.parent  # dress_rehearsal/ (this node moved under second_pass/ on 2026-08-23)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("torch==2.12.1", "numpy==2.5.0", "jsonargparse==4.49.0")
    .run_commands(
        f"git clone {UPSTREAM_URL} {UPSTREAM_DIR}",
        f"cd {UPSTREAM_DIR} && git checkout {UPSTREAM_COMMIT}",
    )
    .add_local_dir(str(_LOCAL_DRESS), DRESS_DIR, ignore=["second_pass/digit_port/results", "digit_port/results", "results"])
    .add_local_dir(str(_LOCAL_DIGIT), DIGIT_DIR, ignore=["results"])
)

volume = modal.Volume.from_name("old-dress-rehearsal-data", create_if_missing=True)
app = modal.App("old-digit-port", image=image)


def _module(directory: str, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        f"{Path(directory).name}_{name}", Path(directory) / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _apply_overrides(path: Path, overrides: dict) -> None:
    text = path.read_text()
    for key, value in overrides.items():
        literal = json.dumps(value) if isinstance(value, str) else repr(value)
        pattern = re.compile(rf"^{re.escape(key)}(: [^=]+)? = .*$", re.MULTILINE)
        text, count = pattern.subn(
            lambda match: f"{key}{match.group(1) or ''} = {literal}", text
        )
        if count != 1:
            raise SystemExit(f"override {key!r} matched {count} lines, expected 1")
    path.write_text(text)


@app.function(volumes={DATA_DIR: volume}, timeout=600)
def list_data() -> list[str]:
    names = sorted(p.name for p in (Path(DATA_DIR) / "generated").iterdir() if p.is_dir())
    print("\n".join(names), flush=True)
    return names


@app.function(volumes={DATA_DIR: volume}, gpu="H100", timeout=7200, memory=65536)
def run_variant(
    tag: str,
    dataset: str,
    arm: str = "digit",
    seconds: float = 600.0,
    batch_size: int = 512,
    overrides_json: str = "",
) -> dict:
    """One tier-faithful evaluator run of a digit-port variant on an H100."""

    import os
    import subprocess
    import time

    overrides = json.loads(overrides_json) if overrides_json else {}
    hard_configs = _module(DRESS_DIR, "hard_configs")
    emit_arm = _module(DIGIT_DIR, "emit_arm")

    if dataset in hard_configs.PUBLIC_DATA_ROOTS:
        dirname = hard_configs.PUBLIC_DATA_ROOTS[dataset]
    elif dataset in hard_configs.HARD_CONFIGS:
        dirname = hard_configs.dataset_dirname(dataset)
    else:
        dirname = dataset
    data_root = f"{DATA_DIR}/generated/{dirname}"
    if not Path(data_root).is_dir():
        raise FileNotFoundError(f"{data_root} is not on the volume")

    work = Path("/tmp") / tag
    work.mkdir(parents=True, exist_ok=True)
    submission_path = emit_arm.emit(arm, Path(DIGIT_DIR) / "submission.py", work)
    if overrides:
        _apply_overrides(submission_path, overrides)

    manifest_path = work / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            hard_configs.manifest(
                name=f"digit-{tag}",
                data_root=data_root,
                training_seconds=seconds,
                batch_size=batch_size,
                eval_batch_size=batch_size,
            ),
            indent=2,
        )
    )

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
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    log = process.stdout + "\n----- stderr -----\n" + process.stderr

    result = None
    curve = []
    for line in process.stdout.splitlines():
        if line.startswith("RESULT_JSON="):
            result = json.loads(line[len("RESULT_JSON=") :])
        elif line.startswith("step="):
            curve.append(line)

    record = {
        "tag": tag,
        "dataset": dataset,
        "data_root": data_root,
        "arm": arm,
        "batch_size": batch_size,
        "overrides": overrides,
        "training_seconds_requested": seconds,
        "wall_clock_seconds": elapsed,
        "returncode": process.returncode,
        "curve": curve[::20][-40:],
        "result": result,
    }
    out_dir = Path(DATA_DIR) / "digit_results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    (out_dir / "runner.log").write_text(log)
    volume.commit()

    if result is None:
        print(log[-6000:], flush=True)
        raise RuntimeError(f"runner failed ({process.returncode}) for {tag}")
    seed = result["seeds"][0]
    print(
        f"[{tag}] steps={seed['completed_training_steps']} "
        f"train_loss={seed['final_train_loss']:.4f} "
        f"mean_exact={result['score']['mean_exact_accuracy']:.4f}",
        flush=True,
    )
    return record


@app.function(volumes={DATA_DIR: volume}, timeout=1800)
def collect(prefix: str = "") -> None:
    root = Path(DATA_DIR) / "digit_results"
    for path in sorted(root.glob("*/result.json")):
        record = json.loads(path.read_text())
        if prefix and not record["tag"].startswith(prefix):
            continue
        result = record.get("result")
        if not result:
            print(f"{record['tag']}: FAILED")
            continue
        seed = result["seeds"][0]
        depth = seed.get("depth_profile", {})
        rungs = " ".join(str(r["correct_examples"]) for r in depth.get("rungs", []))
        ood = " ".join(str(r["correct_examples"]) for r in depth.get("ood_n_rungs", []))
        print(
            f"{record['tag']:28s} ds={record['dataset']:10s} bs={record['batch_size']:5d} "
            f"steps={seed['completed_training_steps']:7d} "
            f"loss={seed['final_train_loss']:.4f} "
            f"mean_exact={result['score']['mean_exact_accuracy']:.4f} "
            f"| seen {rungs} | ood {ood}"
        )


# The sweep this node was built to run: throughput, the digit bottleneck, and whether the
# missing `T = 1` row is what stalls a variable-`N` set.
VARIANTS: list[dict] = [
    # Control: the exact arm that read 27/140 on m6 and plateaued on Hard, on Hard's
    # faithful public proxy.
    dict(tag="s1_m5_base", dataset="m5"),
    # Same widths, same volume, same generator — `T in {1,2,4}` instead of `{2,4,8}`. The
    # only difference is whether `f` is ever supervised directly.
    dict(tag="s1_hD_base", dataset="hD_t_shallow"),
    # Epochs, not updates: if the step is launch-bound, a 4x batch is nearly free.
    dict(tag="s1_m5_bs2048", dataset="m5", batch_size=2048,
         overrides_json='{"PEAK_LR": 0.002}'),
    # Does bypassing the digit bottleneck restore the parent's ability to memorise?
    dict(tag="s1_m5_carry", dataset="m5", arm="digit_carry"),
    # Cheaper step, wider model: same launch count, four times the arithmetic per launch.
    dict(
        tag="s1_m5_wide2",
        dataset="m5",
        batch_size=1024,
        overrides_json='{"D_MODEL": 512, "D_FF": 2048, "N_OP_LAYERS": 2, "PEAK_LR": 0.0015}',
    ),
]


@app.local_entrypoint()
def sweep(only: str = "") -> None:
    wanted = [v for v in VARIANTS if not only or v["tag"] in only.split(",")]
    handles = [run_variant.spawn(**variant) for variant in wanted]
    print(f"spawned {len(handles)}: {[v['tag'] for v in wanted]}", flush=True)
    for handle, variant in zip(handles, wanted):
        try:
            handle.get()
        except Exception as error:  # noqa: BLE001 - one variant must not sink the sweep
            print(f"[{variant['tag']}] failed: {error}", flush=True)
