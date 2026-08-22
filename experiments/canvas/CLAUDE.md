# Canvas: the image-style practice substrate

**Centralized writeup**: none yet — the corpus was built 2026-08-22; no practice node has run. Design memo: [`ideas/style_practice_substrate.md`](../../ideas/style_practice_substrate.md) (§5 grader, §6 style/taste, §8 node sequence).

Third substrate of the practice arc, after [`rhm/practice/`](../rhm/practice/README.md) and [`mjc/practice/`](../mjc/practice/README.md): image *styles* with scarce exemplars each, inpainting under a token budget as the piece, and no DGP oracle on any agent-consumed path. The learner lives in pixel space (a VQ codebook over 256×256 swatches, a 16×16 code grid); everything human-readable stays in the teacher slot.

## What a style is here

A still-image GLSL fragment shader, `shaders/<slug>.glsl` (36 in the library), with a header of human-readable metadata (`title / description / tags`) and a `uniform float seed` that varies the *arrangement* only — palette and geometry are `const`. Exemplars are renders under (seed, crop). The program is the style's latent truth: we hold it (so the exact completion of any masked swatch region is one render call — a reporting-only readout, never consumed), the agent never does. The aesthetic register is the shader swatches of Lluminate (`reading/Lluminate.pdf`); the styles were authored by Claude sessions, not evolved and not API-generated.

## Layout

- `render_glsl.py` — headless moderngl renderer: WebGL-1 dialect → GLSL 330 by a small transpile; crops via `offset`/`zoom` in the vertex stage. Works on macOS now; on Linux/Modal it needs EGL + mesa.
- `shaders/` — the style library. Format, authoring rules and the checker are in `corpora/build.py`'s docstring. Run `--check` on anything you add.
- `corpora/build.py` — library → corpus: `corpora/data/<tag>/<style>/NNN.png` + `style.json` (each swatch's seed/offset/zoom) + `index.json` + contact sheet. Data is gitignored; rebuild with
  `cd experiments && python -m canvas.corpora.build --out canvas/corpora/data/lib0 --n_per_style 64` (~30 s).
- `taste/clicker.py` — pairwise preferences for the taste gauge: a local UI for a human, or `--manifest` to emit pair images + `pairs.jsonl` for a headless (Claude) rater. Records go to `taste/clicks/*.jsonl`, same schema either way.
- `tiles.py`, `contact_sheet.py` — an earlier adjacency-tile DGP with an exact validity oracle and a `describe()` of its depth margins. Kept as a possible calibration twin (e.g. misalignment between a hidden grid and the VQ grid, measured exactly), **not** the substrate.
- `figures/` — contact sheets of everything above.

## Conventions

- Python env: conda `glp` (torch 2.7, moderngl, PIL, CLIP). Run from `experiments/` as `python -m canvas.<module>`.
- **No LLM API calls.** Anything LLM-shaped (authoring styles, rating pairs) is done by Claude sessions or Opus subagents on the subscription.
- No Modal infrastructure yet. When the first node lands, add a `shared.py` per `one_layer_deeper/shared.py`, render the corpus locally and put it on the volume.
- The port plan is the arc's: fork `rhm/practice/ratchet/ratchet.py` + `macros.py` verbatim with a fork notice, `critic/`'s learned grade of record (typicality on held-out exemplars, `q = 0.90`, corpus size by the self-manufactured-damage rule), `at_support` logged from day one, ranks against `never_base`. Moves restricted to cells that intersect the mask.
- Discuss results before writing READMEs here.
