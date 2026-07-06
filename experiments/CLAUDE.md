# Claude Code - experiment-specific guidance

- Please use `modal run --detach` for any Modal jobs that we expect to take longer than 2 minutes. It can be easy to kill jobs when running in attached mode.
- You need to run detached functions by invoking the function explicitly with e.g. `modal run --detach a2a_forward/permutation_test.py::a2a_permutation_test` rather than just `modal run --detach a2a_forward/permutation_test.py`. The `--detach` parameter only persists the most recently-created function, and normally you can't really control the order they get created in. So it's best to invoke explicitly.
- Likewise, running e.g. `modal run --detach a2a_forward/permutation_test.py::main` as a local entrypoint will also result in premature cancellations, so don't do this.

## Canonical experiment-README format

A top-level experiment `README.md` is the default context injected into every new Claude Code session for that experiment, so it must stay under the injection cap (~25K tokens) while still letting a new agent gain full context fast. To keep it lean as an experiment accumulates dozens of scripts and auxiliary writeups, follow this split (see `a2a_forward/` and `rhm/` for reference implementations):

- **`FILES.md`** holds the *complete* file-by-file reference — one `## Code files` table (every `.py` → one-line purpose) and one `## Auxiliary READMEs` table (every sub-README → its one-line summary). This is lookup material an agent greps on demand, not standing context, so it lives out of the main README.
- **`README.md`** keeps everything a new agent needs by default: goal, architecture, per-experiment sections (each with a `**Full writeup**` link to its auxiliary README, key results, and a reproduction command), CLI, Modal volume layout, and next steps. It carries a concise **`## Code & files`** section: a themed bullet map of the code (grouped by subsystem) plus a pointer to `FILES.md`. Keep any load-bearing gotchas inline in the README, not only in `FILES.md`.

When adding a new script or sub-README, add its row to `FILES.md` and, if it opens a new theme, extend the `## Code & files` bullet map. Persist anything removed from a README into an auxiliary doc; never delete it outright.