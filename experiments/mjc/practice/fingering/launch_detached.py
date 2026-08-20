"""Session-isolated launcher for detached Modal runs (the plasticity_gain gotcha fix).

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the remote
input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch the client in
its OWN session (`start_new_session=True` == fork+setsid), so no harness signal reaches it.
Combined with the runner's per-stage checkpoint commits, a lost client costs minutes.

Usage (from experiments/):
    python3 mjc/practice/fingering/launch_detached.py --fn gates --tag g0 --seed 0 [modal args...]
    python3 mjc/practice/fingering/launch_detached.py --fn fingering --tag f0 --seed 0 [...]

Logs go to **experiments/.launch_logs/fingering_<tag>.log**, deliberately OUTSIDE the `mjc`
package. Gotcha, cost one failed run: `shared.py` mounts the package with
`add_local_python_source("mjc")`, and Modal hashes the whole directory -- so a live launcher log
under `mjc/practice/<node>/results/` is a file being appended to while the mount is built, and any
*other* `modal run` started meanwhile dies with `ExecutionError: <path> was modified during build
process`. It is not a failure of the run being launched; it is a failure of every run launched
alongside it. (The etude's launcher still logs inside the package -- left alone so its reproduce
commands keep working, but do not start a second app while one of its detached runs is writing.)
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # .../experiments
LOG_DIR = os.path.join(EXP_DIR, ".launch_logs")
ENTRY = {"gates": ("gates.py", "gates"), "fingering": ("fingering.py", "fingering")}


def main():
    args = sys.argv[1:]
    tag, fn = "default", "gates"
    keep = []
    i = 0
    while i < len(args):
        if args[i] == "--fn" and i + 1 < len(args):
            fn = args[i + 1]; i += 2; continue
        if args[i] == "--tag" and i + 1 < len(args):
            tag = args[i + 1]
        keep.append(args[i]); i += 1
    if fn not in ENTRY:
        raise SystemExit(f"--fn must be one of {sorted(ENTRY)}, got {fn!r}")
    mod, ep = ENTRY[fn]
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"fingering_{tag}.log")
    cmd = ["modal", "run", "--detach", f"mjc/practice/fingering/{mod}::{ep}"] + keep
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=EXP_DIR, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
