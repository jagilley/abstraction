# Canvas: the image substrate of the practice arc

**Centralized writeup**: [README.md](README.md) (index) · [`plant/README.md`](plant/README.md) (the substrate description, 2026-08-25). Design memo: [`ideas/style_practice_substrate.md`](../../ideas/style_practice_substrate.md) (§5 grader, §6 style/taste, §8 node sequence, §11 what ran).

Third substrate of the practice arc, after [`rhm/practice/`](../rhm/practice/README.md) and [`mjc/practice/`](../mjc/practice/README.md): image *styles*, inpainting under a token budget as the piece, and no DGP oracle on any agent-consumed path. The learner lives in pixel space (a k-means codebook over 256×256 swatches, a 16×16 code grid of 16-px patches); everything human-readable stays in the teacher slot.

## The substrate is the tile grammar, aligned

`tiles.py` is the DGP: a hidden 8×8 grid of 32-px tiles with typed edges, valid iff every shared edge agrees; a *school* (tileset + demand weights + affinities + palette) is a style; many schools per tileset = style drift on fixed truth, a new tileset = truth drift. **Crops must be aligned to the tile grid** (offset a multiple of 32 px) so a tile is exactly a 2×2 code block — `plant/tiles_twin/` showed that with alignment the miner recovers the tile catalogue as its level-2 vocabulary and the depth ladder is graded, and that without it every precondition of the arc fails. The oracle (`Tileset.validate`, `valid_blocks`, per-swatch tile ids) is written down beside every run and never consumed.

Two learned graders, one per currency, both oracle-free (`plant/`):
- **Truth**: the adjacency-support test (`plant/tiles_twin/adjacency.py`) — every oriented adjacent code pair in a fill, including pairs across the hole boundary, must have occurred in genuine exemplars. A conjunction, zero forward passes. Its scope condition is pairwise-local validity, which the tile grammar has by construction. Its binding limit is coverage (the false-reject floor), fixed by more exemplars, not by count thresholds.
- **Taste**: `critic/`'s typicality reader (`plant/grader.py`) — mean per-token NLL under a second reader on a disjoint split, q = 0.90 per (style, mask-size). It reads *demand*, not truth: on an aligned alphabet it passes in-style-but-invalid fills and fails valid-but-atypical ones, and no tail statistic of it fixes that. Log it; do not use it as the mining gate or the audition.

The GLSL library (`shaders/`, `render_glsl.py`, `corpora/build.py`; 36 styles) is kept as a **taste venue** for later — no lattice, no pairwise-local validity, so only the demand gauge exists there. `taste/clicker.py` is built and has zero clicks.

## Layout

- `tiles.py` — the DGP (standalone, no torch) with `describe()`; `contact_sheet.py`.
- `shared.py` — Modal app `canvas`, volume `canvas-data`, profile `chromatic`.
- `plant/` — alphabet (`codebook.py`), any-order plant (`model.py`, `sampler.py`), typicality grader (`grader.py`), recurrence (`recur.py`), the Modal node (`plant.py`), reductions. `plant/tiles_twin/` — the tiles corpora (`corpus.py`), oracle columns (`twin.py`), `describe_schools.py`, and the grader re-analyses (`tailgrade.py`, `adjacency.py`). Everything indexed in the two `FILES.md`s.
- `render_glsl.py`, `shaders/`, `corpora/build.py`, `taste/` — the GLSL library and the taste gauge.
- `figures/` — contact sheets; node figures live under each node's `figures/` (PNGs gitignored; regenerate per each README's reproduce block).

## Conventions

- Python env: conda `glp` (torch 2.7, moderngl, PIL). Run from `experiments/` as `python -m canvas.<module>`. Corpora are rendered locally and put on the volume as tars (`render_glsl.py` needs EGL on Linux; `tiles.py` renders anywhere).
- **No LLM API calls.** Anything LLM-shaped (authoring styles, rating pairs) is done by Claude sessions or Opus subagents on the subscription.
- Splits are by **seed**, never by swatch (two crops of one seed overlap in pixels). `plant/corpus.py:SPLITS` is the convention; held-out is held out from the codebook too.
- Practice nodes live in `practice/` ([`practice/README.md`](practice/README.md)); node 1 (`ratchet/`) has run. The port plan for the first practice node was the arc's: fork `rhm/practice/ratchet/ratchet.py` + `macros.py` verbatim with a fork notice, support as the grade of record (value target, mining gate, audition), typicality and `at_support` logged from day one, ranks against `never_base`, and a `given` arm from the tile catalogue. Moves restricted to cells that intersect the mask.
- Modal's local-source mount aborts if any file under `canvas/` changes during the build — keep live logs outside the package (`launch_detached.py --logdir` / `CANVAS_LOGDIR`).
- Discuss results before writing READMEs here.
