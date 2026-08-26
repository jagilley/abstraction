"""Session-isolated launcher for detached Modal runs (critic's, retargeted to canvas/plant).

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch
the client in its OWN session (`start_new_session=True` == fork+setsid), so no harness signal
reaches it.

Usage (from experiments/):
    python3 canvas/plant/launch_detached.py --fn run --tag pl0 --qtag q0 --k 512
Logs go to canvas/plant/results/launch_<tag>.log, or to --logdir.

--logdir exists because Modal's local-source mount scan fails the build outright if ANY file
under the mounted package changes while it is walking ("... was modified during build
process"). A live launch log inside `canvas/` is exactly such a file, so when several launches
or analyses overlap, point the logs somewhere outside the package.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    tag, fn = "default", "run"
    for i, a in enumerate(args):
        if a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
        if a == "--fn" and i + 1 < len(args):
            fn = args[i + 1]
    logdir = os.environ.get("CANVAS_LOGDIR", os.path.join(HERE, "results"))
    for i, a in enumerate(args):
        if a == "--logdir" and i + 1 < len(args):
            logdir = args[i + 1]
    args = [a for i, a in enumerate(args)
            if a not in ("--fn", "--logdir")
            and not (i > 0 and args[i - 1] in ("--fn", "--logdir"))]
    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, f"launch_{tag}.log")
    cmd = ["modal", "run", "--detach", "-m", f"canvas.plant.plant::{fn}"] + args
    exp_dir = os.path.dirname(os.path.dirname(HERE))          # .../experiments
    env = dict(os.environ, MODAL_PROFILE=os.environ.get("MODAL_PROFILE", "chromatic"))
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
