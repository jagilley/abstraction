"""[sotto] `voicing`'s launcher, copied and retargeted. Session-isolated launcher for detached Modal runs — `enharmonic`'s idiom,
copied and retargeted (not imported: the donor's entrypoint must stay the donor's).

A harness TaskStop reaching the `modal run --detach` client's process group can cancel the
remote input mid-run, silently swallowing post-cancellation `volume.commit()`s. Fix: launch
the client in its OWN session (`start_new_session=True` == fork+setsid), so no harness signal
reaches it. Combined with the runner's per-cycle checkpoint commits, a lost client costs
minutes.

Usage (from experiments/):
    python3 rhm/practice/voicing/sotto_voce/launch_detached.py --fn voicing_run --tag vo_s0 ...

Logs go to rhm/practice/voicing/sotto_voce/results/launch_<tag>.log.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    tag, fn = "default", "voicing_run"
    for i, a in enumerate(args):
        if a in ("--tag", "--outdir-tag") and i + 1 < len(args):
            tag = args[i + 1]
        if a == "--fn" and i + 1 < len(args):
            fn = args[i + 1]
    args = [a for i, a in enumerate(args)
            if a != "--fn" and not (i > 0 and args[i - 1] == "--fn")]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", f"launch_{tag}.log")
    # [voicing] ONE environment fact of this box, stated rather than worked around: the CLI
    # imports `sotto_voce.py` locally before it ships anything, so the interpreter behind `modal`
    # needs numpy. On this image only the SYSTEM `modal` can reach the Modal API (a venv copy
    # of the client fails every gRPC connect through the session's egress proxy), so the fix
    # is numpy in the system interpreter — `python3 -m pip install --user numpy==1.26.4` —
    # and NOT a venv `modal`. `MODAL_PROFILE` is passed through from the caller, as the donor's
    # own RUN_*.sh scripts do.
    env = dict(os.environ)
    cmd = ["modal", "run", "--detach", f"rhm/practice/voicing/sotto_voce/sotto_voce.py::{fn}"] + args
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(HERE))))                                        # .../experiments
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    print(f"[launch] pid={p.pid} (own session)  cmd: {' '.join(cmd)}")
    print(f"[launch] log: {log_path}")


if __name__ == "__main__":
    main()
