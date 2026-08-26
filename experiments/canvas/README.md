# canvas — the image substrate of the practice arc

**Up**: [`../../CLAUDE.md`](../../CLAUDE.md) (repo) · **Files**: [`FILES.md`](FILES.md) ·
**Session conventions**: [`CLAUDE.md`](CLAUDE.md) · **Design memo**:
[`ideas/style_practice_substrate.md`](../../ideas/style_practice_substrate.md) · **Siblings**:
[`rhm/practice/`](../rhm/practice/README.md), [`mjc/practice/`](../mjc/practice/README.md)

Third substrate of the practice arc: images in a *style*, with inpainting under a token budget
as the piece, a learner that lives in pixel space (a codebook over 256×256 swatches → a 16×16
code grid), and no oracle on any agent-consumed path. Everything human-readable stays in the
teacher slot.

## What the substrate is (as of 2026-08-25)

**The tile grammar in [`tiles.py`](tiles.py) is the substrate.** A swatch is a hidden 8×8 grid
of 32-px tiles with four typed edges; a tiling is valid iff every shared edge agrees; rendered,
a valid tiling is continuous knotwork or circuitry and a violation is a visibly broken line. A
*school* is a demand-concentration over the tileset's valid set (weights, neighbour affinities,
palette) — a style; several schools on one tileset is style drift on fixed truth, a new tileset
is truth drift. With the crop grid **aligned** to the tile grid, a tile is exactly a 2×2 block of
codes and a 2×2-tile motif is 4×4 — `macros.py`'s level structure made literal. The oracle
(`Tileset.validate`, `valid_blocks`) is held by the experimenter and logged beside every number,
never consumed.

The GLSL style library ([`shaders/`](shaders/), 36 styles, [`render_glsl.py`](render_glsl.py),
[`corpora/build.py`](corpora/build.py)) was the first draft of the substrate and is kept as a
**taste venue** for later: its styles have no lattice for a code grid to nest in and no
pairwise-local validity, so only the demand gauge exists there. [`taste/clicker.py`](taste/clicker.py)
(pairwise preferences, human or headless Claude rater) is built and unused.

## Children

- [`plant/`](plant/README.md) — **the substrate description** (2026-08-25). Built the alphabet,
  the any-order plant and `critic/`'s learned grader, and measured the arc's preconditions. On
  the GLSL library all three failed in one direction (recurrence tracked code entropy, not parts;
  no depth gradient; the grader passed the plant's own bland fills above real exemplars). The
  aligned/misaligned tiles twin, one variable, showed why: when the code grid nests in the
  DGP's lattice the miner recovers the tile catalogue as its level-2 vocabulary (purity
  0.93–0.98, T[3] nested-representable 0.996) and the depth ladder is graded; misaligned
  reproduces the null. Against known validity, no statistic of per-token likelihood reads a
  pairwise constraint on an aligned alphabet (AUC .30–.65) — validity is a *relation*, typicality
  a *property* — while a zero-forward **adjacency-support test** does (AUC .96–.99; rejects 96.6 %
  of the plant's oracle-invalid fills at the clean ceiling). Two organs, one per currency: support
  for truth, typicality for taste. Child: `plant/tiles_twin/`[^private].

## Where things stand

No practice node has run on canvas. The first — `ratchet` forked verbatim onto aligned tiles,
support as the grade of record, typicality and `at_support` logged from day one, ranks against
`never_base` and a `given` arm from the tile catalogue — is queued in
[`plant/README.md`](plant/README.md)'s next steps with the design choices still to settle.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
