# FILES — `tempo` (levels as execution span on the plant)

**Up**: [README.md](README.md) · [../README.md](../README.md) (practice) · [../FILES.md](../FILES.md)

## Code files

None directly at this node — `tempo/` is the super-writeup and index over three implementer nodes.
Each child forks its donor verbatim and adds; no child edits a sibling. The donor chain is
`solo/world.py` → `prestissimo/world.py` (piece-parameterised) → `accelerando/world.py` (body knobs)
→ `rubato/world.py` (paths, inverse/forward dynamics, the corrected executor), each gated
bit-for-bit against the previous node's run of record.

## Children

| child | what it varies | record |
|---|---|---|
| [`prestissimo/`](prestissimo/FILES.md) | The four-rung execution-span ladder (`T[ℓ] ⊆ T[ℓ−1]×T[ℓ−1]`, welded members) on a fast octagon on the **donor pusher** (τ = 0.5 s) under a swept observation delay; then the diagnostic round (gain-grid floor, damping and segment-length cells, the open-loop-reflex instrument) and the fixed lead-in. Eight tags; `dH10g` is the ladder of record. | `SPEC.md`, `FILES.md` (20 decisions, four run records, limitations) |
| [`accelerando/`](accelerando/FILES.md) | The **fast body** (mass 0.06, τ = 30 ms) at a fixed 120 ms delay with **tempo** as the era ladder (768 → 48 ms notes), two grades, recordings and resamples (`t1`, of record); the **force-level program** (`p1` and the retractions `fsmoke_a`, `fsmoke_a2`; the metronome ladder `fsmoke_b`); the **crank** over per-tempo recordings under `conductor`'s thermostat (`c1`). | `SPEC.md`, `FILES.md` (37 decisions, six run records, two retractions) |
| [`rubato/`](rubato/FILES.md) | The **factored program**: the unit as a realised path on phase, the exact inverse dynamics as an experimenter-side oracle (`k1`, of record; the resampler retraction in `kdsmoke1`), a body model **learned from slow practice** in two families across three fit sets (`klsmoke1–3`), and **online correction** against the learned forecast on a reads-per-span ladder with an exact discrete gain rule (`kc1`, `kc2`), P-E re-asked. The no-forward-model rule is lifted for the executor only. | `SPEC.md`, `FILES.md` (37 decisions, nine run records, two retractions) |

## Auxiliary

| file | what |
|---|---|
| `conversation_2026-09-05.md`[^private] | The orchestrating conversation: Jasper's prompts verbatim, responses in summary — the mapping, the delay-vs-tempo question, the factoring hypothesis, the smoke-first rule. |

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
