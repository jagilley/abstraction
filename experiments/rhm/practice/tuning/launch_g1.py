"""Session-isolated launcher for detached Modal runs.

FORK NOTICE: `../teacher_slot/endo_yield/launch_detached.py` verbatim, retargeted at this
module.

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch
the client in its OWN session (`start_new_session=True` == fork+setsid). Combined with the
runner's per-checkpoint commits, a lost client costs minutes.

Usage (from experiments/):
    python3 rhm/practice/tuning/launch_detached.py --fn tune_lm --tag tn0 \
        --log-dir /tmp [modal args...]

Point `--log-dir` OUTSIDE the `rhm` package for long runs: a log appended to for hours
inside the package breaks any other Modal build that mounts it.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "rhm.practice.tuning.gate1_lm"


def main():
    args = sys.argv[1:]
    tag, fn = "default", "gate1"
    for i, a in enumerate(args):
        if a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
        if a == "--fn" and i + 1 < len(args):
            fn = args[i + 1]
    log_dir = os.path.join(HERE, "results")
    for i, a in enumerate(args):
        if a == "--log-dir" and i + 1 < len(args):
            log_dir = args[i + 1]
    for flag in ("--fn", "--log-dir"):
        args = [a for i, a in enumerate(args)
                if a != flag and not (i > 0 and args[i - 1] == flag)]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"launch_{tag}.log")
    cmd = ["modal", "run", "--detach", "-m", f"{MODULE}::{fn}"] + args
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../experiments
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
