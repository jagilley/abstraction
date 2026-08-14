# practice — the practice arc on RHM

**Up**: [../README.md](../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../ideas/practice_manufactures_its_own_credit.md)
**Sibling arc**: [`mjc/practice/`](../../mjc/practice/README.md) — where the arc started, on MuJoCo
control; [`mjc/practice/etude/`](../../mjc/practice/etude/README.md) is the direct parent of
everything here.
**Files**: [FILES.md](FILES.md)

## Scope

Practice as a control loop wrapped around an ordinary learning rule — acting on the *conditions and
units* of learning rather than on the model — ported onto the RHM sculpting substrate. The move here
is deliberate: the étude closed with a precondition it could not test on MuJoCo (*hierarchy is
meaningful only over boundaries that carry information*; its committed units were state-independent
command sequences, so post-commit drift was exactly 0.0000 and fusion was provably vacuous).
Sculpting removes that degeneracy — a committed unit is a move program executed by a generator that
reads the observed configuration — and adds instruments no other substrate has: an exact DP oracle,
a known true rule vocabulary, and an exactly-known aleatoric floor.

## Children

### [`crystallize/`](crystallize/README.md) — certificate-gated compilation under priced feedback (2026-08-14)

**Goal**: instantiate the étude's compile op — δ-silence certificate, selection-not-averaging,
scoring under the consumption distribution — on a production/control task with priced feedback, and
grade it against never-compiling on success × priced time.

**Finding**: the étude's open question is answered — **state-conditioned commitment (a library keyed
by the observed target) beats state-independent commitment by 1.8–3.0× in 6/6 commit states**, and
averaging valid realisations destroys them by 3.6–5.0×, with RHM giving the mechanism exactly (the
modal token span went off-grammar in half its blocks while every contributing realisation was fully
on-grammar). But the certificate turns out to have nothing to certify: with the plant frozen and only
the selector learning, **committing at cycle 1 matches every gated arm at 26× less priced time**, and
a shadow-compile instrument shows the committable content of practice traces is flat from cycle 1
while the closed-loop policy improves by 0.12–0.14. The interpretation we settled on (argued, not
measured): **δ-silence gates compilation only where practice moves the executor** — the étude's
practice trained the forward model its ballistic units bet on, this one trains the judge; practice on
a frozen plant makes you better at improvising, and improvisation is what does not compile. A
precheck closes the loop: let the plant learn and the committed-unit ceiling moves at 31–66× the
metering noise floor, which is what the next round is built on.

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/crystallize/crystallize.py::selfcheck_remote
python3 rhm/practice/crystallize/launch_detached.py --fn crystallize --tag cg_s0 \
    --arms "never,sched_early,sched_late,delta_gate,gate_single" \
    --n-cycles 60 --n-grad 4 --value-lr-online 3e-5 --n-rt 384 --n-score 512 --n-cand 32 \
    --sil-c 0.06 --sil-cv 0.10 --sil-win 5 --sil-hold 2 --sched-early 1 --sched-late 30 \
    --probe-every 4 --shadow-compile
python3 rhm/practice/crystallize/analyze_crystallize.py --tag cg_s0 --fetch --figures
```

Full commands, calibrations and volume layout: [`crystallize/README.md`](crystallize/README.md).
