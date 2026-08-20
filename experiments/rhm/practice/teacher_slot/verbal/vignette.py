"""The reasoner-facing prompt template for teacher_slot/verbal (Rung B1).

BLINDING CONTRACT (a leak voids the rung).  Nothing in any string below may contain:
wall, scaffold, crutch, spurious, shortcut, cheat, merge, teacher, key (as a name for T),
rotation / rotate / remap, any repo or experiment name, any mention of what any other loop,
policy or agent decided, or any hint of what the experimenters expect.  Post-8000 events are
described only by the numbers themselves.  `check_blinding()` at the bottom enforces the
word list on every rendered prompt.

SYMMETRY CONTRACT.  The two actions are described in matched sentence-for-sentence form, the
option order in both the action block and the answer format is counterbalanced across samples,
and no adjective anywhere marks either action as the intervention or the default.
"""

from __future__ import annotations

import re

from .records import DEC_CEIL_CONTENT, DEC_CEIL_GIVEN_T, Cell, Readout

SYSTEM_PROMPT = (
    "You are a careful analyst. You read a report of measurements and answer in exactly the "
    "format the report asks for. Do not use tools. Do not add any text outside the requested "
    "format."
)

# ------------------------------------------------------------------ fixed blocks

SETUP = """\
# Setup

A transformer language model is being trained by next-token prediction on sequences drawn from a
fixed hierarchical generative grammar. The grammar is stationary: the generative process that
produces the sequences is not altered at any point during training. Each sequence is 64 positions
long. Training runs for 20000 optimizer steps.

Every training sequence carries an extra token T at position 0, ahead of the 64 sequence positions.
T takes one of 16 values, and the 16 values occur with about equal frequency.

{t_fact}

Positions 0-15 of the sequence are called span A below, and positions 16-63 are called span B.
"""

T_FACT = {
    "keyed": """\
A measurement on the training distribution finds that the value of T stands in a one-to-one
correspondence with the value of one hidden variable of the generator, written Z below. Z is the
variable that governs positions 0-15 of the sequence.""",
    "dead": """\
A measurement on the training distribution finds no association between the value of T and any
hidden variable of the generator: T's value is drawn independently of the derivation that produces
the sequence. The hidden variable of the generator that governs positions 0-15 of the sequence is
written Z below.""",
}

ACTION_LEAVE = """\
  LEAVE_T    The input stream for the next interval carries T at position 0, with the value
             it has.
"""
ACTION_REPLACE = """\
  REPLACE_T  The input stream for the next interval carries a fixed neutral filler token at
             position 0, the same value in every sequence, in place of T.
"""

CHOICE_BLOCK = """\
# The choice

At each checkpoint you choose what the model's input stream contains for the next interval of
training:

{actions}
Both are edits to the input stream and nothing else. Neither alters any model parameter, and
neither alters the grammar or the 64 sequence positions. Each is reversible at every later
checkpoint: after LEAVE_T you may choose REPLACE_T at any later checkpoint, and after REPLACE_T you
may choose LEAVE_T at any later checkpoint. Your choice holds until the next checkpoint, at which
you will be asked again.
"""

GLOSS_TASK = """\
# Readouts

nllA   mean negative log likelihood, in nats per position, over span A, measured on held-out
       sequences presented exactly as the model is currently being trained: with T at position 0,
       or with the filler at position 0, according to the choices recorded below.
nllB   the same quantity over span B.

`input` records what stood at position 0 during the interval that ended at that step.
"""

GLOSS_FULL = """\
# Readouts

nllA   mean negative log likelihood, in nats per position, over span A, measured on held-out
       sequences presented exactly as the model is currently being trained: with T at position 0,
       or with the filler at position 0, according to the choices recorded below.
nllB   the same quantity over span B.
nllA*  mean negative log likelihood over span A on the same held-out sequences, measured with the
       value at position 0 drawn at random from T's 16 values, independently of the rest of the
       sequence. This measurement is taken the same way at every step, whatever position 0 carries
       during training.
nllB*  the same quantity over span B.
dec    the fraction of held-out sequences on which the value of Z can be read out of the model's
       internal activations by a decoder trained on those activations, with the value at position 0
       drawn at random exactly as for nllA*. It is measured only at some checkpoints; "-" means it
       was not measured at that step. Two exact reference values, computed from the generator and
       not from any model: a decoder that reads only the 64 sequence positions and does not read
       position 0 reaches {ceil_content:.3f} on this measurement, and a decoder that is given the
       value of T reaches {ceil_given:.3f}.

`input` records what stood at position 0 during the interval that ended at that step.
"""

ANSWER_BLOCK = """\
# Now

You are at step {step}. Answer in exactly this format and add nothing else:

RELIANCE: <what information you believe the model is currently using for its span A predictions>
STABILITY: <whether you expect that reliance to remain reliable over the rest of training, and why>
CHOICE: <{opt1} or {opt2}>
REASONS: <your reasons for that choice, in two to four sentences>
"""


# ------------------------------------------------------------------ table


def _fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def render_table(rows: list[Readout], info: str) -> tuple[str, list[str]]:
    """Returns (table text, footnotes)."""
    notes: list[str] = []
    if info == "task":
        head = f"{'step':>6}  {'input':<7}  {'nllA':>7}  {'nllB':>7}"
    else:
        head = (f"{'step':>6}  {'input':<7}  {'nllA':>7}  {'nllB':>7}  {'nllA*':>7}  "
                f"{'nllB*':>7}  {'dec':>5}")
    lines = [head, "-" * len(head)]
    for r in rows:
        stream = "T" if r.in_stream == "T" else "filler"
        if info == "task":
            lines.append(f"{r.step:>6}  {stream:<7}  {_fmt(r.nllA):>7}  {_fmt(r.nllB):>7}")
        else:
            dec = _fmt(r.dec, 3)
            if r.dec is not None and r.dec_step is not None and r.dec_step != r.step:
                notes.append(f"dec at step {r.step} was measured at step {r.dec_step}.")
            lines.append(
                f"{r.step:>6}  {stream:<7}  {_fmt(r.nllA):>7}  {_fmt(r.nllB):>7}  "
                f"{_fmt(r.nllA_star):>7}  {_fmt(r.nllB_star):>7}  {dec:>5}")
    return "\n".join(lines), notes


def render_history(choices: list[tuple[int, str]], join_at: int, first: bool) -> str:
    if choices:
        s = "; ".join(f"step {st}: {ch}" for st, ch in choices)
        return f"Your choices so far: {s}.\n"
    if first and join_at > 1000:
        return ("T stood at position 0 for every interval up to now. The first interval you choose "
                f"for is the one beginning at step {join_at}.\n")
    return ""


def render(cell: Cell, rows: list[Readout], choices: list[tuple[int, str]], step: int,
           opt_order: str = "LR") -> str:
    """opt_order 'LR' lists LEAVE_T first; 'RL' lists REPLACE_T first."""
    actions = (ACTION_LEAVE + ACTION_REPLACE) if opt_order == "LR" \
        else (ACTION_REPLACE + ACTION_LEAVE)
    opt1, opt2 = ("LEAVE_T", "REPLACE_T") if opt_order == "LR" else ("REPLACE_T", "LEAVE_T")
    gloss = GLOSS_TASK if cell.info == "task" else GLOSS_FULL.format(
        ceil_content=DEC_CEIL_CONTENT, ceil_given=DEC_CEIL_GIVEN_T)
    table, notes = render_table(rows, cell.info)
    note_block = ("\n" + "\n".join(f"({i+1}) {n}" for i, n in enumerate(dict.fromkeys(notes)))
                  + "\n") if notes else ""
    hist = render_history(choices, cell.join_at, first=not choices)
    return (
        SETUP.format(t_fact=T_FACT[cell.t_fact])
        + "\n" + CHOICE_BLOCK.format(actions=actions)
        + "\n" + gloss
        + "\n# Record\n\n" + table + "\n" + note_block
        + ("\n" + hist if hist else "")
        + "\n" + ANSWER_BLOCK.format(step=step, opt1=opt1, opt2=opt2)
    )


# ------------------------------------------------------------------ blinding guard

FORBIDDEN = [
    r"\bwall\b", r"\bscaffold", r"\bcrutch", r"\bspurious", r"\bshortcut", r"\bcheat",
    r"\bmerge", r"\bmerged", r"\bteacher", r"\brotat", r"\bre-?key", r"\bcrutch",
    r"\bfourwall", r"\brhm\b", r"\bteacher_slot", r"\bpractice\b", r"\bexperiment",
    r"\bouter loop", r"\bpolicy\b", r"\bagent\b", r"\bwe (expect|predict|think)",
    r"\bprobe\b", r"\bpathway\b", r"\bcurrency\b", r"\bledger\b", r"\bd4\b",
    r"\bthe key\b", r"\bindex\b",
]


def check_blinding(text: str) -> list[str]:
    hits = []
    low = text.lower()
    for pat in FORBIDDEN:
        for m in re.finditer(pat, low):
            hits.append(f"{pat} -> ...{low[max(0, m.start()-30):m.end()+30]}...")
    return hits


def check_symmetry(text: str) -> list[str]:
    """Both option tokens must appear the same number of times in the fixed scaffolding
    (the choice-history line is excluded: it reports what was actually chosen)."""
    text = "\n".join(l for l in text.splitlines() if not l.startswith("Your choices so far"))
    n_l = len(re.findall(r"LEAVE_T", text))
    n_r = len(re.findall(r"REPLACE_T", text))
    return [] if n_l == n_r else [f"LEAVE_T x{n_l} vs REPLACE_T x{n_r}"]
