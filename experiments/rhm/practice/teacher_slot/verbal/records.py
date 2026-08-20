"""Record access + session splicing for teacher_slot/verbal (Rung B1).

All numbers shown to a reasoner come from MEASURED logs on this repo's volume-fetched
records:

  kept-key trajectory      fourwall/lm/figures/fwlm0/wall.json        (T present forever,
                                                                       schedule rotates from 8000)
  uninformative-token      fourwall/lm/figures/fwlm0/dead_wall.json   (T present, carries nothing)
  post-removal arms        fourwall/lm/figures/fwlm1/merge_{1000,8000,13000}.json
  post-removal (dead)      fourwall/lm/figures/fwlm1/merge_dead_8000.json

Splice rule (default `elapsed`): a session that removes T at session step S continues on the
measured removal arm whose own removal step m is nearest to S, read at the *matched elapsed time
since removal*, underlying = m + (S' - S).  Every displayed number is therefore a real measurement;
the approximation lives entirely in the step label (the underlying model has trained
|m - S| steps more or fewer than the session's clock claims).

Alternative rule (`anchored`): the same arm and elapsed offset, but level-shifted so that the arm's
value at its own removal step coincides with the kept arm's value at S.  This removes the
"trajectory goes backwards" artifact when m < S at the cost of showing synthesised numbers.
Default is `elapsed` (real numbers, approximate label).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
PRACTICE = os.path.abspath(os.path.join(HERE, "..", ".."))
FWLM0 = os.path.join(PRACTICE, "fourwall", "lm", "figures", "fwlm0")
FWLM1 = os.path.join(PRACTICE, "fourwall", "lm", "figures", "fwlm1")

# exact, model-free (fwlm1/setup.json refs.ceil.key_none.d4 / key_wall.d4)
DEC_CEIL_CONTENT = 0.9210359197397016
DEC_CEIL_GIVEN_T = 1.0

_CACHE: dict[str, list[dict]] = {}


def load(path: str) -> list[dict]:
    if path not in _CACHE:
        with open(path) as f:
            _CACHE[path] = json.load(f)["log"]
    return _CACHE[path]


def arm(tag: str, name: str) -> list[dict]:
    root = FWLM0 if tag == "fwlm0" else FWLM1
    return load(os.path.join(root, f"{name}.json"))


def at(log: list[dict], step: int) -> dict:
    """Exact-step lookup, clipped to the arm's range."""
    by = {r["step"]: r for r in log}
    if step in by:
        return by[step]
    steps = sorted(by)
    if step > steps[-1]:
        return by[steps[-1]]
    if step < steps[0]:
        return by[steps[0]]
    # nearest below
    below = max(s for s in steps if s <= step)
    return by[below]


def probe_near(log: list[dict], step: int, tol: int = 1250) -> dict | None:
    """Latest checkpoint at or before `step` that carries a probe readout, within `tol`.

    Backwards-only on purpose: a readout shown at step X must never come from a measurement
    the run had not taken yet.  The decoder readout moves slowly (d4 under the
    T-uninformative condition drifts 0.23 -> 0.29 over steps 8000-20000 on the kept arm), so
    a lag of up to `tol` steps is small against its own dynamic range; every such lag is
    footnoted in the rendered vignette.
    """
    cands = [r for r in log if r.get("levels") and 0 <= step - r["step"] <= tol]
    if not cands:
        return None
    return max(cands, key=lambda r: r["step"])


# ----------------------------------------------------------------------------- readouts
@dataclass
class Readout:
    step: int              # the session's clock
    in_stream: str         # "T" or "filler"
    nllA: float
    nllB: float
    nllA_star: float | None = None
    nllB_star: float | None = None
    dec: float | None = None
    dec_step: int | None = None      # step at which `dec` was actually measured
    src: str = ""                    # provenance, for the log only
    underlying: int = 0              # underlying record step, for the log only


def _cond_consumed(r: dict, in_stream: str = "T") -> str:
    """Evaluate the model the way the SESSION is currently training it, not the way the donor
    arm's own `consumed` field says.  T in the stream -> the `true` eval condition; the filler in
    the stream -> the `none` eval condition."""
    if in_stream == "T":
        return "true"
    return "none" if "none" in r["nll"] else "rand"


def _cond_uninformative(r: dict, in_stream: str = "T") -> str:
    """The T-uninformative instrument is `rand` at EVERY step of EVERY cell: position 0 carries a
    value of T drawn at random, independently of the sequence.

    Fixed condition on purpose.  `rand` and `none` are far apart on a T-using model (idx NLL
    1.938 vs 1.595 at step 8000; d4 0.231 vs 0.386) and converge only after removal, so switching
    between them at the removal boundary would manufacture a jump in the instrument out of nothing.
    `rand` is also the arc's canonical pathway condition (fourwall/lm finding 5) and is the only
    one present on both fwlm0 kept arms, which is what lets the session run the full horizon.
    """
    return "rand"


def read(log: list[dict], underlying: int, session_step: int, in_stream: str,
         src: str, shift: float = 0.0) -> Readout:
    r = at(log, underlying)
    cc = _cond_consumed(r, in_stream)
    cu = _cond_uninformative(r, in_stream)
    p = probe_near(log, underlying)
    dec = dec_step = None
    if p is not None:
        lv = p["levels"]
        lcond = cu if cu in lv else ("rand" if "rand" in lv else list(lv)[0])
        dec = lv[lcond]["key"]["d4"]
        dec_step = p["step"]
    return Readout(
        step=session_step,
        in_stream=in_stream,
        nllA=r["nll"][cc]["idx"] + shift,
        nllB=r["nll"][cc]["out"] + shift,
        nllA_star=r["nll"][cu]["idx"] + shift,
        nllB_star=r["nll"][cu]["out"] + shift,
        dec=dec,
        dec_step=dec_step,
        src=f"{src}:{cc}/{cu}",
        underlying=r["step"],
    )


# ----------------------------------------------------------------------------- sessions
@dataclass
class Cell:
    name: str
    kept_tag: str
    kept_arm: str
    removal_arms: list[tuple[int, str, str]]   # (m, tag, arm)
    info: str                                  # "task" | "full"
    t_fact: str                                # "keyed" | "dead"
    grid: list[int]
    join_at: int = 1000                        # first step at which the reasoner decides
    obs_rows: list[int] = field(default_factory=lambda: [250])


# Decision grid.  Steps 1000-6000 are the pre-disruption reads (the money cell).  8000 is the
# single boundary read and 8250 its recovery.  Everything after 8250 is sampled a uniform 1000
# steps into each subsequent era, NOT at era boundaries: fourwall/lm's own fwlm0 -> fwlm1
# correction (README finding 3) showed that a grid landing only on boundaries overstates the
# kept arm's error ~8x, and a grid that made the kept arm look uniformly broken would bias the
# removal decision for a sampling reason.
GRID = [1000, 2000, 4000, 6000, 8000, 8250, 11000, 13000, 15000, 19000]

CELLS = {
    "pre_task": Cell(
        name="pre_task", kept_tag="fwlm0", kept_arm="wall",
        removal_arms=[(1000, "fwlm1", "merge_1000"), (8000, "fwlm1", "merge_8000"),
                      (13000, "fwlm1", "merge_13000")],
        info="task", t_fact="keyed", grid=GRID),
    "pre_full": Cell(
        name="pre_full", kept_tag="fwlm0", kept_arm="wall",
        removal_arms=[(1000, "fwlm1", "merge_1000"), (8000, "fwlm1", "merge_8000"),
                      (13000, "fwlm1", "merge_13000")],
        info="full", t_fact="keyed", grid=GRID),
    "dead_ctrl": Cell(
        name="dead_ctrl", kept_tag="fwlm0", kept_arm="dead_wall",
        removal_arms=[(8000, "fwlm1", "merge_dead_8000")],
        info="full", t_fact="dead", grid=GRID),
    "late_task": Cell(
        name="late_task", kept_tag="fwlm0", kept_arm="wall",
        removal_arms=[(1000, "fwlm1", "merge_1000"), (8000, "fwlm1", "merge_8000"),
                      (13000, "fwlm1", "merge_13000")],
        info="task", t_fact="keyed", grid=GRID, join_at=13000,
        obs_rows=[250, 1000, 2000, 4000, 6000, 8000, 8250, 11000]),
}


def pick_removal_arm(cell: Cell, S: int) -> tuple[int, list[dict]]:
    m, tag, name = min(cell.removal_arms, key=lambda t: abs(t[0] - S))
    return m, arm(tag, name)


class Session:
    """Walks one reasoner through one measured trajectory.

    State: the current input condition, the step at which T was last removed (if it is out),
    and the removal arm currently in use.
    """

    TRANSIENT_OFFSET = 125
    MAX_POINTS = 13

    def __init__(self, cell: Cell, splice: str = "elapsed"):
        self.cell = cell
        self.splice = splice
        self.kept = arm(cell.kept_tag, cell.kept_arm)
        self.in_stream = "T"          # condition for the interval that ENDS at the next point
        self.removed_at: int | None = None
        self.rem_m: int | None = None
        self.rem_log: list[dict] | None = None
        self.shift: float = 0.0
        self.rows: list[Readout] = []
        self.choices: list[tuple[int, str]] = []
        self.pending_transient: int | None = None
        # observation rows before the join point (input always T; no choice attributed)
        for s in cell.obs_rows:
            if s < cell.join_at:
                self.rows.append(read(self.kept, s, s, "T", cell.kept_arm))
        self._future = [g for g in cell.grid if g > cell.join_at]
        self.step = cell.join_at
        self.rows.append(read(self.kept, cell.join_at, cell.join_at, "T", cell.kept_arm))
        self.n_points = 1

    # -- readout for a session step, given the current stream state
    def _readout(self, session_step: int) -> Readout:
        if self.in_stream == "T":
            return read(self.kept, session_step, session_step, "T", self.cell.kept_arm)
        assert self.removed_at is not None and self.rem_log is not None
        underlying = self.rem_m + (session_step - self.removed_at)
        return read(self.rem_log, underlying, session_step, "filler",
                    f"removal_arm(m={self.rem_m})", shift=self.shift)

    def record_choice(self, choice: str) -> None:
        """Apply the reasoner's choice at self.step; advance to the next decision point."""
        self.choices.append((self.step, choice))
        prev = self.in_stream
        if choice == "REPLACE_T":
            if prev == "T":
                self.removed_at = self.step
                self.rem_m, self.rem_log = pick_removal_arm(self.cell, self.step)
                if self.splice == "anchored":
                    # level-shift so that the removal arm's LAST PRE-REMOVAL value coincides
                    # with the kept arm's value at the session's removal step.  Removes the
                    # "trajectory rewinds" artifact when m != S, at the cost of showing a
                    # synthesised level (the removal-induced *change* is still measured).
                    kept_now = at(self.kept, self.step)
                    pre = [r for r in self.rem_log if r["step"] < self.rem_m
                           and not r.get("merged", False)]
                    arm_pre = pre[-1] if pre else at(self.rem_log, self.rem_m)
                    self.shift = (kept_now["nll"]["true"]["idx"]
                                  - arm_pre["nll"]["true"]["idx"])
                else:
                    self.shift = 0.0
                self.pending_transient = self.step + self.TRANSIENT_OFFSET
            self.in_stream = "filler"
        else:
            if prev == "filler":
                # restore: session returns to the kept arm at the absolute session step
                self.removed_at = None
                self.rem_m = self.rem_log = None
                self.shift = 0.0
            self.in_stream = "T"
        # next decision point
        nxt = None
        if self.pending_transient is not None and self.pending_transient > self.step:
            nxt = self.pending_transient
            self.pending_transient = None
        else:
            fut = [g for g in self._future if g > self.step]
            nxt = fut[0] if fut else None
        if nxt is not None and self.n_points < self.MAX_POINTS:
            self._future = [g for g in self.cell.grid if g > nxt]
            self.step = nxt
            self.rows.append(self._readout(nxt))
            self.n_points += 1
        else:
            self.step = None

    @property
    def done(self) -> bool:
        return self.step is None
