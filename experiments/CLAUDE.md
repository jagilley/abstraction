# Claude Code - experiment-specific guidance

- Please use `modal run --detach` for any Modal jobs that we expect to take longer than 2 minutes. It can be easy to kill jobs when running in attached mode.
- You need to run detached functions by invoking the function explicitly with e.g. `modal run --detach a2a_forward/permutation_test.py::a2a_permutation_test` rather than just `modal run --detach a2a_forward/permutation_test.py`. The `--detach` parameter only persists the most recently-created function, and normally you can't really control the order they get created in. So it's best to invoke explicitly.
- Likewise, running e.g. `modal run --detach a2a_forward/permutation_test.py::main` as a local entrypoint will also result in premature cancellations, so don't do this.
- The correct Modal workspace to use for most jobs is `chromatic` (we have credits applied to this workspace). However, there may be a few types of jobs that rely on existing volume results which may only be accessible from `jagilley`. If so, you're allowed to use this workspace as well.

Iterating fast on our ideas is the most important goal. This means:
- We should run smoke tests on Modal itself (presumably without `--detach`) to take advantage of the fastest hardware
- We should test the most principled version of our ideas
- We should parallelize anything that can be parallelized if it'll help us iterate on our ideas faster.
- Do not suggest replicating results across multiple seeds unless there's specific reason to believe that a single seed may uniquely affect the results. If there is a specific reason to believe this, you should autonomously launch multiple seed-experiments in parallel. I, personally, don't want to have to think about seed-dependence, as it's a non-substantive low-level implementation detail.

## Canonical experiment-README format

An experiment/sub-experiment `README.md` is the default context injected into every new Claude Code session for that experiment, so it must stay under the injection cap (~25K tokens) while still letting a new agent gain full context fast. To keep it lean as an experiment accumulates dozens of scripts and auxiliary writeups, follow this split (see `a2a_forward/` and `rhm/` for reference implementations):

- **`README.md`** keeps everything a new agent needs by default: goal, architecture, per-experiment sections, CLI, Modal volume layout, and next steps. Keep any load-bearing gotchas inline in the README, not only in `FILES.md`. *Please add to this file sparingly.*
- **`FILES.md`** holds the complete file-by-file reference — one `## Code files` table (every `.py` → one-line purpose) and one `## Auxiliary READMEs` table (every sub-README → its one-line summary). This is lookup material an agent greps on demand, not standing context, so it lives out of the main README.

When adding a new script or sub-README, add its row to `FILES.md` and, if it opens a new theme, extend the `## Code & files` bullet map. Persist anything removed from a README into an auxiliary doc; never delete it outright.

This structure should nest hierarchically. We're working on introducing the concept of "sub-experiments", which contain their own READMEs and auxiliary READMEs, and which should contain their own `FILES.md` files too. (Example sub-experiment directories are `experiments/a2a_forward/reaching` and `experiments/rhm/ratchet`). This means that experiment-level READMEs and FILES files should generally contain pointers to their children's READMEs and FILES files, rather than pointing to individual children auxiliary READMEs.

The canonical *shape* of this nesting — and the direction we're migrating toward, where every writeup becomes its own folder with a `README.md` inside (`TOPIC_README.md` → `topic/README.md`) rather than a flat sibling file — is specified in the repo-root [`STRUCTURE.md`](../STRUCTURE.md). Use the `/writeup`[^private] skill to write a new writeup and propagate its (halving) summaries up the tree.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
