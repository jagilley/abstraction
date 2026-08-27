# canvas — files

**Up**: [`README.md`](README.md)

## Code files

| file | purpose |
|---|---|
| [`tiles.py`](tiles.py) | **The DGP.** Tile catalogue with typed edges, `Tileset` (validate, `valid_blocks`), `School` (a demand-concentration: weights, affinities, palette, strokes), constraint-propagating `sample`, the nested damage ladder (`seam` / `offstyle` / `mask`), the renderer, and `describe()` — single-tile repairability and greedy dead-end rate per level, the substrate's own depth margins. Standalone, no torch; never imported on an agent-consumed path. |
| [`shared.py`](shared.py) | Modal infrastructure: app `canvas`, volume `canvas-data`, image, `NumpyEncoder`; the ignore list prunes logs/figures/dumps from the local-source mount. |
| [`render_glsl.py`](render_glsl.py) | Headless moderngl renderer for the GLSL style library: WebGL-1 dialect → GLSL 330, crops via `offset`/`zoom` in the vertex stage. macOS now; needs EGL + mesa on Linux. |
| [`corpora/build.py`](corpora/build.py) | GLSL library → swatch corpus (`corpora/data/<tag>/`), with the shader authoring rules and `--check`. Data is gitignored. |
| [`contact_sheet.py`](contact_sheet.py) | Contact sheets for any swatch directory. |
| [`taste/clicker.py`](taste/clicker.py) | Pairwise-preference collector for the taste gauge (local UI or `--manifest` for a headless rater). Teacher-slot data; nothing consumes it. `taste/clicks/` is empty. |
| [`shaders/*.glsl`](shaders/) | The 36 GLSL styles (header-parsed metadata; `seed` varies arrangement only). |

## Children

| folder | summary |
|---|---|
| [`plant/`](plant/README.md) | The substrate description: alphabet, plant, grader, and the five preconditions on the GLSL library and the aligned/misaligned tiles twin; the tail and adjacency-support grader re-analyses. Establishes aligned tiles as the substrate. |
| [`practice/`](practice/README.md) | The practice arc on canvas. Node 1 (`ratchet/`, 2026-08-26) earns a level-indexed code-tuple vocabulary (T[2] ≈ tiles, T[3] ≈ motifs) over an inpainting depth ladder, priced in forward passes and graded by the adjacency-support test: earned ≈ given, the descent purely vocabulary-carried, `never_base` 0.2–0.35 worse. Files in [`practice/FILES.md`](practice/FILES.md). |

## Figures

| path | what |
|---|---|
| `figures/library0/` | contact sheet of the GLSL library (`lib0`) |
| `figures/dgp_prototype/` | `tiles.py` schools and damage ladder per tileset, plus `describe.json` |
| `figures/lluminate_smoke/` | one smoke run of Lluminate's evolutionary loop, as a reference register |
