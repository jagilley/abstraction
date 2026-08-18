"""Session-isolated launcher for detached Modal runs (`setlist`'s, retargeted at `fourwall`).

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch the
client in its OWN session (`start_new_session=True` == fork+setsid), so no harness signal
reaches it. Combined with the runner's per-cycle checkpoint commits, a lost client costs
minutes.

The log is opened with "a", not "w", so a relaunch appends and the file stays an archive.

Usage (from experiments/):
    python3 rhm/practice/fourwall/launch_detached.py --fn fourwall --tag fw_s0 [modal args]

Logs go to rhm/practice/fourwall/results/launch_<tag>.log.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    tag, fn = "default", "fourwall"
    for i, a in enumerate(args):
        if a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
        if a == "--fn" and i + 1 < len(args):
            fn = args[i + 1]
    args = [a for i, a in enumerate(args)
            if a != "--fn" and not (i > 0 and args[i - 1] == "--fn")]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", f"launch_{tag}.log")
    cmd = ["modal", "run", "--detach", f"rhm/practice/fourwall/fourwall.py::{fn}"] + args
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../experiments
    with open(log_path, "a") as log:
        p = subprocess.Popen(cmd, cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
