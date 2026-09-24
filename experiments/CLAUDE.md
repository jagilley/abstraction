# Claude Code - experiment-specific epistemic guidance

Iterating fast on our ideas is the most important goal. This means:
- We should run smoke tests on Modal itself (presumably without `--detach`) to take advantage of the fastest hardware
- We should test the most principled version of our ideas
- We should parallelize anything that can be parallelized if it'll help us iterate on our ideas faster. That said, frivolous parallelization wastes resources. We should only ever parallelize by a factor of maybe 4 or so; higher DoP is unlikely to be epistemically useful. Parallelizing separate arms/conditions of the same experiment is certainly allowed and encouraged. 
- If there is a specific reason to believe that it's epistemically important to replicate results across multiple seeds, it's ok to do so. However:
    - We should always start by running and waiting for a single-seed version of the full experiment to verify that things work as intended and get a sense of the described effect
    - *If and only if*:
        - it seems like the result is epistemically meaningful, and
        - the effect size is small enough that it may be sensitive to seed-dependence, and
        - you have not implicitly triangulated the effects of seed-dependence via some other control (e.g., a multi-arm run where we already saw a consistent effect would *not* qualify),
    - *then* you should autonomously launch multiple seed-experiments in parallel.
I, personally, don't want to have to think about seed-dependence, as it's a non-substantive low-level implementation detail. This is something that you should manage.
- Subagents may NOT run multi-seeded or high-DoP experiments without authorization from their supervisor agent.
- **Long loops must be resumable.** Any training loop that runs longer than about half an hour saves its full state (weights, optimizers, readouts, tables, buffers, counters, every random generator, and any replay/yoke position) at every era boundary and at a fixed cadence, and carries a resume-identity gate: saved at cycle c and restored in a fresh container, the run must match the uninterrupted one. Small retroactive changes then start from the saved cycle instead of from cycle one. Record the host CPU type with every run: Modal L4 containers land on AVX-512 and non-AVX-512 hosts, and numpy 1.26 dispatches `log`/`exp` to SVML on the former, one ulp off libm, so resume identity is bit-exact within a host type and one ulp across types (`NPY_DISABLE_CPU_FEATURES` pins it, but only for a fresh lineage, since it changes numerics against banked arms). The reference implementation is `rhm/practice/voicing/sotto_voce/aliquot/rubato/` (gates R-0/R-1/R-2 in its `DESIGN.md`). Added 2026-09-23 after the practice arc paid full re-runs for gate changes that took effect at cycle 50–60.


## Canonical experiment-README format

An experiment/sub-experiment `README.md` is the default context injected into every new Claude Code session for that experiment, so it must stay under the injection cap (~25K tokens) while still letting a new agent gain full context fast. To keep it lean as an experiment accumulates dozens of scripts and auxiliary writeups, follow this split (see `a2a_forward/` and `rhm/` for reference implementations):

- **`README.md`** keeps everything a new agent needs by default: goal, architecture, per-experiment sections, CLI, Modal volume layout, and next steps. Keep any load-bearing gotchas inline in the README, not only in `FILES.md`. *Please add to this file sparingly.*
- **`FILES.md`** holds the complete file-by-file reference — one `## Code files` table (every `.py` → one-line purpose) and one `## Auxiliary READMEs` table (every sub-README → its one-line summary). This is lookup material an agent greps on demand, not standing context, so it lives out of the main README.

When adding a new script or sub-README, add its row to `FILES.md` and, if it opens a new theme, extend the `## Code & files` bullet map. Persist anything removed from a README into an auxiliary doc; never delete it outright.

This structure should nest hierarchically. We're working on introducing the concept of "sub-experiments", which contain their own READMEs and auxiliary READMEs, and which should contain their own `FILES.md` files too. (Example sub-experiment directories are `experiments/a2a_forward/reaching` and `experiments/rhm/ratchet`). This means that experiment-level READMEs and FILES files should generally contain pointers to their children's READMEs and FILES files, rather than pointing to individual children auxiliary READMEs.

The canonical *shape* of this nesting — and the direction we're migrating toward, where every writeup becomes its own folder with a `README.md` inside (`TOPIC_README.md` → `topic/README.md`) rather than a flat sibling file — is specified in the repo-root [`STRUCTURE.md`](../STRUCTURE.md). Use the `/writeup`[^private] skill to write a new writeup and propagate its (halving) summaries up the tree.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
