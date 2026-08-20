"""Session-isolated launcher for detached Modal runs — `../../fourwall/lm/launch_detached.py`'s
pattern, retargeted at the teacher_slot decision round.

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch
the client in its OWN session (`start_new_session=True` == fork+setsid). Combined with the
runner's per-checkpoint commits, a lost client costs minutes.

Usage (from experiments/):
    python3 rhm/practice/teacher_slot/decision/launch_detached.py --fn slot --tag tsdA [modal args...]

Logs go to rhm/practice/teacher_slot/decision/results/launch_<tag>.log.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "rhm.practice.teacher_slot.decision.slot_lm"


def main():
    args = sys.argv[1:]
    tag, fn = "default", "slot"
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
    # NOTE: a log that is APPENDED TO for hours while sitting inside the `rhm` package
    # breaks any *other* Modal build that mounts it ("was modified during build process").
    # Point `--log-dir` outside the package for long detached runs and copy the log in
    # afterwards; `results/` stays the resting place, not the live sink.
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"launch_{tag}.log")
    cmd = ["modal", "run", "--detach", "-m", f"{MODULE}::{fn}"] + args
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(HERE))))                     # .../experiments
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
