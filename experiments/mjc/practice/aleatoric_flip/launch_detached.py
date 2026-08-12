"""Session-isolated launcher for detached Modal runs (the plasticity_gain gotcha fix).

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s (the save
line prints, nothing lands). Fix: launch the client in its OWN session
(`start_new_session=True` == fork+setsid, portable to macOS where no `setsid` binary
exists), so no harness signal can reach it. Combined with the runner's incremental
per-arm/per-milestone checkpoint commits, a lost client costs minutes, not the run.

Usage (from anywhere):
    python3 mjc/practice/aleatoric_flip/launch_detached.py --mode calibrate --tag afcal_s0
    python3 mjc/practice/aleatoric_flip/launch_detached.py --mode main --tag af_s0 --seed 0

Logs go to mjc/practice/aleatoric_flip/results/launch_<tag>.log; the client exits once the
detached app is submitted, and the run continues server-side.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    tag = "default"
    for i, a in enumerate(args):
        if a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", f"launch_{tag}.log")
    cmd = ["modal", "run", "--detach",
           "mjc/practice/aleatoric_flip/aleatoric_flip.py::flip"] + args
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../experiments
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
