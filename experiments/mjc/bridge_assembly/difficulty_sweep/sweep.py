"""Launcher for the bridge_assembly difficulty x benchmark-timescale sweep.

Every cell is the PARENT runner (`../bridge_assembly.py`) with defaults untouched except
the two swept knobs, so the parent's own 3-seed runs (`asm_s{0,1,2}`) ARE the phi=1.2 /
bench_lr=3e-3 cell and are reused rather than re-run.

  * difficulty   -- the drift side of the task geometry (`--regions`), on the y=+0.30
                    eval corridor. B-mastered (0,-0.30, phi=-1.2, pre-drift) and the
                    off-path R-noise decoy (0,+0.90, amp 30) are IDENTICAL in every cell,
                    so retention demand and the noise tax are controlled.
  * bench_lr     -- b(s)'s tracking timescale (the other leg of the timescale ratio).

The budget axis is FREE: each run grades a milestone ladder, and the AUC over milestones
<= T' is exactly the outcome the run would have had at budget T'. No extra runs needed.

Usage (from experiments/):
    python3 mjc/bridge_assembly/difficulty_sweep/sweep.py --cells phi06,phi20 --seed 0
    python3 mjc/bridge_assembly/difficulty_sweep/sweep.py --list
"""

import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EXPDIR = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../experiments
RUNNER = "mjc/bridge_assembly/bridge_assembly.py::bridge_assembly"

TAIL = "0.0,-0.30,-1.2,0.0,1,B-mastered; 0.0,0.90,0.0,30.0,0,R-noise"


def one(phi):
    return f"0.0,0.30,{phi},0.0,0,A-drift; " + TAIL


def two(phi):
    return (f"-0.25,0.30,{phi},0.0,0,A1-drift; 0.25,0.30,{-phi},0.0,0,A2-drift; " + TAIL)


# name -> (regions, bench_lr, extra CLI args).  `phi12` is the parent cell (== asm_s*).
# The dimensionless knob under test is r = adapt_lr / bench_lr (how slow b(s) tracks
# relative to how fast the frontier moves).  `*_alr9e4` cells sit at the SAME r as the
# corresponding `*_blr1e3` cell but with both timescales 3x faster in absolute terms --
# the direct collapse test.
CELLS = {
    "phi06":      (one(0.6), 3e-3, []),
    "phi12":      (one(1.2), 3e-3, []),   # == bridge_assembly asm_s{0,1,2}
    "phi20":      (one(2.0), 3e-3, []),
    "phi28":      (one(2.8), 3e-3, []),
    "two12":      (two(1.2), 3e-3, []),   # plasticity_gain-like: two opposed on-corridor rots
    "two24":      (two(2.4), 3e-3, []),
    # benchmark-timescale legs (same geometry + adapt_lr; b(s) faster/slower)   r=0.3 / 0.03
    "phi12_blr1e3": (one(1.2), 1e-3, []),
    "phi12_blr1e2": (one(1.2), 1e-2, []),
    "two12_blr1e3": (two(1.2), 1e-3, []),
    "two12_blr1e2": (two(1.2), 1e-2, []),
    # frontier-timescale legs: same r as *_blr1e3, both clocks 3x faster
    "phi12_alr9e4": (one(1.2), 3e-3, ["--adapt-lr", "9e-4"]),
    "two12_alr9e4": (two(1.2), 3e-3, ["--adapt-lr", "9e-4"]),
    # noise-tax control: identical to two12 except the off-path decoy is CLEAN (amp 0).
    # delta over-spends on the irreducible region in every cell (w_R 1.2-1.6x) -- a
    # difficulty-independent cost. This isolates it from the benchmark-lag effect.
    "two12_nonoise": ("-0.25,0.30,1.2,0.0,0,A1-drift; 0.25,0.30,-1.2,0.0,0,A2-drift; "
                      "0.0,-0.30,-1.2,0.0,1,B-mastered; 0.0,0.90,0.0,0.0,0,R-clean",
                      3e-3, []),
}


def launch(cell, seed, extra):
    regions, blr, cell_extra = CELLS[cell]
    tag = f"dsw_{cell}_s{seed}"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log_path = os.path.join(HERE, "logs", f"{tag}.log")
    cmd = ["modal", "run", "--detach", RUNNER, "--tag", tag, "--seed", str(seed),
           "--regions", regions, "--bench-lr", str(blr)] + cell_extra + extra
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=EXPDIR, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[launch] {tag}  pid={p.pid}  log={log_path}")
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a, extra = ap.parse_known_args()
    if a.list:
        for k, (r, b, x) in CELLS.items():
            print(f"{k:16s} bench_lr={b:<8g} extra={x} regions={r}")
        return
    for c in [c for c in a.cells.split(",") if c]:
        launch(c, a.seed, extra)


if __name__ == "__main__":
    main()
