"""Session-isolated launcher for detached Modal runs (`../../reread/lm/launch_detached.py`'s
pattern, retargeted at the fourwall LM twin; modal's `-m` module form).

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch
the client in its OWN session (`start_new_session=True` == fork+setsid). Combined with the
runner's per-checkpoint commits, a lost client costs minutes.

Usage (from experiments/):
    python3 rhm/practice/fourwall/lm/launch_detached.py --fn wall_lm --tag fwlm0 [modal args...]

Logs go to rhm/practice/fourwall/lm/results/launch_<tag>.log.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = "rhm.practice.fourwall.lm.wall_lm"


def main():
    args = sys.argv[1:]
    tag, fn = "default", "wall_lm"
    for i, a in enumerate(args):
        if a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
        if a == "--fn" and i + 1 < len(args):
            fn = args[i + 1]
    args = [a for i, a in enumerate(args)
            if a != "--fn" and not (i > 0 and args[i - 1] == "--fn")]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", f"launch_{tag}.log")
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
