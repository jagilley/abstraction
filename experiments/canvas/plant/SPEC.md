# plant — design record

**Up**: [`../CLAUDE.md`](../CLAUDE.md) (canvas) · **Files**: [FILES.md](FILES.md)
**Grader donor**: `rhm/practice/critic/`[^private] — the protocol is
ported, the code is not (RHM's reader is an `rhm.model.GPT` over a derivation; ours is a
bidirectional model over a 16×16 code grid). Nothing under `rhm/` or `mjc/` is modified.
**Vocabulary reference**: [`rhm/practice/ratchet/macros.py`](../../rhm/practice/ratchet/macros.py)'s
header — `T[l]` nested over `T[l−1]` **entries** is what `recur.py` measures the precondition for.
**Idea doc**: [`style_practice_substrate`](../../../ideas/style_practice_substrate.md) §3a
(quantize / generate / decode), §4 (the piece, the ladder, the meter), §5 (the grader), §8
(node sequence and the first-smoke instrument checks), §9 (risks).
**Status (2026-08-25)**: complete, single seed. `q0` (quantizer ladder, 64 s) and `pl0`
(main, 5549 s on one L4) both landed; all eight gates pass on Modal; figures under
[`figures/`](figures/), numbers in `results/run_pl0.json` and `results/quant_q0.json`.
**No README** — numbers went back for discussion first (canvas policy).

One departure from the plan, forced by wall time: bf16 autocast + TF32 were added after the
first smoke measured 0.25 s/step in fp32 (attention over 256 cells at bs=128 dominates), which
would have put the node at ~6 h. Every logit is cast back to fp32 before any `log_softmax`, so
all reported NLLs are fp32 numbers; the gates were re-run under autocast and still pass.
`plant_steps` 20000 and `grader_steps` 4000 are the settings of record.

## The question

This is **not a practice loop**. Before one can run on image styles, the *plant* has to exist
and be described. On RHM the hierarchy was in the data by construction; here it is in the GLSL
program, and whether it survives tokenization into a reconstruction-trained codebook is
genuinely unknown — §9's "the alphabet may not factor." That, plus `critic/`'s grader protocol
running for the first time with **no exact grade beside it**, is what this node is for. It is
a substrate-description node in the spirit of `canvas/tiles.py:describe()` and
`rhm/practice/tall/`'s preflight: measure the preconditions rather than assume them.

## The corpus, and why the split is by SEED

`corpus.py` renders 36 styles × 64 seeds × 2 crops = **4608 swatches** at 256×256 (502 MB,
pushed to the `canvas-data` volume as one tar). A swatch is a (seed, offset, zoom) crop; two
crops of the *same* seed overlap in pixels with high probability at zoom 0.35–0.5, so a
swatch-level split leaks. The shader's `uniform float seed` varies the **arrangement** only, so
a seed-level split is exactly "same style, an arrangement never seen" — which is what held-out
has to mean here. Seven disjoint seed bands:

| split | seeds | swatches/style | role |
|---|---|---|---|
| `plant_train` | 0–31 | 64 | **codebook** *and* plant. Every held-out swatch is held out from the quantizer too. |
| `plant_val` | 32–35 | 8 | the mask-size ladder over training |
| `plant_test` | 36–39 | 8 | completion quality, the damage panel, grader evaluation |
| `gA_train` | 40–51 | 24 | the **driving** grader's corpus; the ladder takes prefixes |
| `gA_val` | 52–53 | 4 | gA's own oracle-free threshold slice |
| `gB_train` | 54–61 | 16 | the **reporting** grader — a second reader on a disjoint split |
| `gB_val` | 62–63 | 4 | gB's own threshold slice |

Asymmetry on the record: gB gets 16 swatches/style against gA's 24, because the seed budget had
to cover both plus two threshold slices. The homogeneous-pair readout is therefore between
graders of slightly different scarcity, not a matched pair.

## The design choices, and why they went this way

### 1. Quantizer: k-means on raw 16×16 patches, not a VQ-VAE

`canvas/CLAUDE.md` fixes the geometry (16×16-px patches ⇒ a 16×16 code grid). Within that,
**k-means with lookup as the decoder**, so the reported floor is the error of *the alphabet*
and of nothing else. A VQ-VAE's floor is a property of two things at once — the codebook and a
learned decoder that can hide structure the tokens do not carry — and the number this node owes
everything downstream is a bound on the tokens. The memo names k-means as "the simplest form"
and the VQ-VAE as the sibling; the sibling is the right follow-up *if* the floor binds.

Fit on `plant_train` only, at **K ∈ {256, 512, 1024, 2048}**. Reported per style as raw MSE and
PSNR *and* as **NMSE = MSE / (per-style patch variance)**, i.e. error as a fraction of the K=1
floor — raw MSE is not comparable between a flat tiling and a noise field, and the library
contains both.

### 2. Generator: one any-order model, style id with dropout

A bidirectional transformer over the 256 cells (d=256, 6 layers, 8 heads, ≈5M params), MaskGIT
-style: mask a set of cells, predict them. **Any-order, not raster**, because the piece is
inpainting and the seam is the filled *neighbourhood*, not a prefix — a raster autoregression
would make a move's context an arbitrary sequence and could not fill an arbitrary cell set.

On the open question of style conditioning, the memo's critic conditions on "surround +
request" and the style id **is** the request. Rather than pay for two models, we train **one**
with the style token dropped to `unknown` on 10% of examples, so the same weights give both
readouts: conditioned (the loop's regime) and surround-only (how much the neighbourhood alone
carries the style). `plant_final.val_style` vs `val_nostyle` is that contrast.

Training masks are 75% a **rectangle** from the ladder — 50% of those fully masked, 50%
partially revealed, so any-order decoding *inside* a hole is in distribution — and 25% MaskGIT's
scatter mask under a cosine ratio. Evaluation masks are always the full rectangle, side
∈ {1,2,3,4,6,8,11,16}: areas 1 → 256, "small holes → the whole grid". **The mask-size ladder is
the depth ladder**, so per-(style, size) held-out NLL checkpointed over training is the
learning-wave readout, reported both raw and as the fraction of each size's own total drop
still remaining (sizes are not comparable in absolute nats).

### 3. Grader: `critic/`'s protocol, with two forced departures

Verdict = **mean per-token NLL of the proposed completion** under a *second reader of the same
form* trained on a **disjoint** split, conditioned on surround + style, thresholded at the
**oracle-free q = 0.90 quantile of its own NLL on a validation slice of genuine exemplars**.
Corpus size by the **self-manufactured-damage rule**: the smallest that passes clean ≥ 0.90 and
damaged ≤ 0.10; if none qualifies, the largest. The full ROC over q is reported and enters no
decision. Two departures, both forced by the substrate:

1. **τ is per (style, mask-size)**, not global. On RHM every configuration had the same length
   and the same demand, so one quantile served. Here NLL scales with how much is being asked for
   and with how entropic the style is, so a single τ would degenerate into "pass the tilings,
   fail the textures" — it would grade the *style*, not the completion. The per-cell rule uses
   only genuine exemplars **of that style at that size**, so it stays oracle-free. The
   global-per-size variant is computed anyway and reported as `pass_global_tau`.
2. **The verdict is a chain-rule NLL over a random reveal order** (2 orders × 4 reveal steps),
   not the independent per-cell NLL given the surround alone. Independent scoring cannot see
   whether the proposed cells are coherent *with each other*, which is exactly what `shuffle`
   damage destroys. Both are computed; the 1-step form is the cheap column, because the loop
   will pay `d_fb` per call and 1 forward vs 8 is the price difference.

**What damage is in code space.** Every class is a rearrangement of codes the agent already
holds — no shader, seed, offset or held-out pixel is read, which is what makes the rule runnable
where there is no oracle:

| class | what it is | in the rule's damage set |
|---|---|---|
| `clean` | the held-out truth | — (the ≥0.90 side) |
| `shuffle` | the true codes permuted **within** the hole | yes |
| `other_style` | the same cells from a swatch of a **different** style | yes |
| `marginal` | i.i.d. from the style's own code marginal | yes |
| `uniform` | i.i.d. uniform over K | no (a floor, reported) |
| `roll` | the same swatch's codes from a shifted region — self-consistent texture in the wrong place | no (reported) |
| `same_style_other` | the same cells from another swatch **of the same style** | no — this is the *valid alternative* class, `critic/`'s `alt_valid`, and a grader that fails it is over-penalising valid completions |
| `plant_argmax` / `plant_iter8` / `plant_beam4` | the plant's own completions | no — §8's gaming check |

**Distance to held-out truth** is code-level Hamming inside the hole, logged per class with
`r(score, −d_truth)`, as a *reporting* readout only. `critic/` measured it near-orthogonal to a
good grade (r = 0.01–0.04) precisely because valid alternatives sit far from the one truth; it
is a sanity instrument, not a lie detector.

### 4. Recurrence: nested over entries, with the ratchet constraint applied

`macros.py`'s `T[l]` is a set of s-tuples of `T[l−1]` **entry** indices, and an entry whose
halves are not both in the lower vocabulary is dropped at build time. `recur.py` measures the
precondition for that on the code grid itself, before any loop: `T[2]` = the **non-overlapping**
2×2 blocks of the 16×16 grid (64 per swatch); `T[3]` = the 2×2 blocks *of those*, with
sub-support entries mapped to OOV — the ratchet constraint, verbatim. Non-overlapping is primary
because that is the nesting `macros.py` uses; overlapping windows are the robustness column.
Per style: distinct count, distinct-per-occurrence, entropy/perplexity, top-64 mass, blocks
covering 50%/90% of occurrences, count and mass at support ≥ 8, and at level 3 the fraction of
occurrences whose four halves are *all* at support.

The library's own split is the free within-substrate control. The part-y set
(`girih_stars, quilt_patch, circuit_traces, tumbling_blocks, truchet_meander,
stained_glass_cells`) and the texture-y set (`agate_bands, brain_coral, wood_grain,
lichen_crust`) are taken **verbatim from the prompt**; the other 26 styles are reported
individually and left unlabelled, so the control is not a taxonomy this node invented.

### 5. Cost-to-depth: the beam, priced in forward passes

`sampler.py` decodes a hole in `steps` confidence-ordered reveals with a beam of `width`
hypotheses scored by cumulative log-prob; **cost = steps × width forward passes**, made
explicit. Three points: `argmax1` (1 forward), `iter8` (8), `beam8x4` (32). Quality vs mask area
at each is the primitive-action cost-to-depth curve — the meter precondition §4 names. Quality
is read three ways: exact code recovery against held-out truth, MSE in *quantized*-pixel space
(so the quantizer floor is not double-counted), and the reporting grader's NLL and pass rate.

## Gates (`modal run -m canvas.plant.plant::selfcheck`)

| gate | what it asserts |
|---|---|
| **G-P** | the patch view is a bijection — the codebook's decoder really is a lookup |
| **G-Q** | k-means recovers 3 separated clusters exactly; `encode` agrees with `assign` |
| **G-M** | a ladder mask is a contiguous square with exactly side² cells |
| **G-S** | the 1-step chain rule **is** the independent per-cell NLL given the surround (so the cheap column is the thing it claims to be) |
| **G-D** | every damage class touches only cells inside the hole; `clean` is the identity |
| **G-B** | the sampler never touches the surround, fills every masked cell, is exactly one argmax forward at (steps=1,width=1), and costs steps×width |
| **G-T** | the oracle-free threshold passes ≈ q of the exemplars it was calibrated on, per bucket |
| **G-R** | a perfect tiling has one `T[2]` and one `T[3]` entry; uniform noise has ~one entry per occurrence, nothing at support, and its `T[3]` **collapses to the single OOV block** — the structural foreclosure, working |

## Reproduce

```bash
cd experiments      # conda glp, MODAL_PROFILE=chromatic
python -m canvas.plant.corpus --out canvas/corpora/data/plant0 --tar
modal volume put canvas-data canvas/corpora/data/plant0.tar corpora/plant0.tar
modal run -m canvas.plant.plant::selfcheck
modal run -m canvas.plant.plant::unpack
modal run -m canvas.plant.plant::quantize_ladder --tag q0
modal run -m canvas.plant.plant::run --tag smoke --smoke 1        # ~10 min
python3 canvas/plant/launch_detached.py --fn run --tag pl0 --qtag q0 --k 512
python3 canvas/plant/analyze.py --tag pl0 --qtag q0 --fetch --figures
```

The whole node is runnable on CPU for debugging: `CANVAS_DATA=<dir> CANVAS_DEV=cpu` and call
`plant.run.get_raw_f()(...)` — that is how the gates and the end-to-end path were smoked before
any Modal round trip.

## Scope conditions and what this node does not claim

- **Single seed, one library, one grid geometry.** Ranks, signs and per-style comparisons are
  the currency; nothing here is a benchmark number.
- **No oracle anywhere, by construction.** There is no exact grade to report beside the learned
  one — that absence is the point (§1 item 2), and it means the grader's calibration rests
  entirely on the self-manufactured-damage rule and the ROC, with distance to held-out truth as
  the one independent (and, on RHM, near-orthogonal) sanity readout. The exact completion of any
  masked region is one `render_glsl.py` call, and it is used for reporting only if at all.
- **Both readers are homogeneous.** `heterogeneous_graders` §9 and `endo_expansion` §7.6 say
  same-type members are blind together wherever the blindness is structural, so the gA–gB pair
  measures sampling noise, not a shared failure mode. The heterogeneous instrument on this
  substrate (a VLM A/B probe in the teacher slot) is out of scope here.
- **`own − world`** (`merge/` round 2's refit-on-own-successes narrowness control) is a loop
  instrument and is not run: there is no arm here whose own graded successes exist.
- This node describes the plant. It commits nothing about whether a practice loop on it works.

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
