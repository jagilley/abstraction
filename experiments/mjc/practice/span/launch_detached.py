"""Session-isolated launcher for detached Modal runs. Copy-fork of `../legato/launch_detached.py`.

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the remote
input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch the client in
its OWN session (`start_new_session=True` == fork+setsid), so no harness signal reaches it.

Usage (from experiments/):
    python3 mjc/practice/span/launch_detached.py --fn span --tag S1 --seed 0 [modal args...]
    python3 mjc/practice/span/launch_detached.py --fn s2   --tag S2 --src-tag S1 [modal args...]

`--fn` defaults to `span` and is consumed by this launcher, not passed on.

Logs go to **experiments/.launch_logs/span_<tag>.log**, deliberately OUTSIDE the `mjc` package:
`shared.py` mounts the package with `add_local_python_source("mjc")` and Modal hashes the whole
directory, so a live launcher log under `mjc/practice/<node>/results/` is a file being appended to
while the mount is built, and any *other* `modal run` started meanwhile dies with
`ExecutionError: <path> was modified during build process`.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))     # .../experiments
LOG_DIR = os.path.join(EXP_DIR, ".launch_logs")


ENTRY = {"span": ("span.py", "span"), "s2": ("s2.py", "s2")}


def main():
    args = sys.argv[1:]
    tag, fn, keep = "default", "span", []
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
    log_path = os.path.join(LOG_DIR, f"span_{tag}.log")
    cmd = ["modal", "run", "--detach", f"mjc/practice/span/{mod}::{ep}"] + keep
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=EXP_DIR, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
