# reread — is a fixed archive renewable to a learner whose vocabulary has climbed?

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) ·
**Files**: [`FILES.md`](FILES.md) (machinery, gates, noise floor, flags) · **Child**:
[`lm/`](lm/README.md) — the LM twin, where the thread's positive result lives ·
**Machinery donor**: [`../ratchet/`](../ratchet/README.md) (imported, unmodified) ·
**Idea**: the reread claim in
[`meta_learning_under_metered_data`](../../../../ideas/meta_learning_under_metered_data.md)'s
2026-08-17 caveat block (reframe iv: free data ≠ free extraction; a token's extractable news is
indexed by the reader's current vocabulary — the book re-read six months later teaches a
different set of things because the reader changed).
**Runs**: `rr_s0` (this node, sculpting) and [`lm0`](lm/README.md) (the twin), both 2026-08-17,
single seed.

## The question

The ratchet's nesting result (`T[l]` over `T[l−1]` entries; a missing level makes the next
unrepresentable, hence unminable from data that contains it) implies mining yield over a *fixed*
archive should be gated on the learner's vocabulary stage rather than on data novelty. No prior
round held the archive fixed to check — every round drew fresh tasks per cycle. Here: a
byte-frozen, SHA-checked archive is traversed for 9 passes with commits at fixed pass
boundaries (L2 after p3, L3 after p6), against a dense epoch-matched control and a paired
frozen-vs-fresh novelty probe run *inside* one agent at each pass boundary (design details and
the design-decision table: [`FILES.md`](FILES.md)). 3 archive depth mixes × 5 arms; fidelity
cell reproduces ratchet's published era-1 numbers; matched-condition noise floor |Δe| ≈ 0.013.

## Findings (sculpting round, `rr_s0`)

**1. Mining yield does not re-arm at commit boundaries.** New-entries-per-pass decays
monotonically to ~0 by pass 6 in every mix, with no visible restart after either commit (the one
whisper: `deep`'s largest post-p1 observation jump lands immediately after the L2 commit).
`n_new_used` — entries the max-sum DP actually selects, the anti-double-counting readout — is
≈0 for essentially every pass after p1. Extraction plateaus at **~50% of the archive's
ground-truth content** (0.46/0.49 at L3 for shallow/mixed) — mining stops, but not from
exhaustion.

**2. Competence on the same frozen archive keeps climbing the whole time.** At the L2 commit
`mixed`'s L2n3 error goes 0.551 → 0.168; at the L3 commit its L3n1 goes 0.422 → 0.230. The
vocabulary learner gains 0.19–0.50 on the meter across the run where eight extra dense epochs
buy 0.00–0.09 (and `dense_fresh` ≈ `dense_reread` everywhere — the dense control gains nothing
from novelty either). **The archive is renewable in competence, not in new vocabulary.**

**3. The paired novelty probe splits by depth mix — opposite the spec's recorded prediction.**
For shallow and mixed archives, frozen ≈ fresh at both levels (yield ratios 0.98–1.13): novelty
is not the binding variable. In the deep archive, fresh wins hard at L3 (final L3n1: `fresh`
0.180 vs `reread` 0.367, the largest gap in the run) — and this is *not* content headroom (the
deep archive holds the most L3 content of the three). The mechanism is the instrument's own
self-rewrite (finding 5 below).

**4. Archive depth-mix acts as a curriculum, and a deep-only archive poisons the vocabulary.**
`deep__reread` mined a 6-entry L2 at recall 0.357 — *worse than a random equal-size subset of
the true table* (+0.09 vs `rand_k`) — which then structurally capped its L3 at recall 0.071 via
the nesting. Same commit schedule, same substrate; the poison came from the data's depth
composition, not from timing. [`recital`](../recital/README.md)'s bottom-heavy law, surfacing on
the data side: the archive itself must be bottom-heavy for the vocabulary to be minable.
Secondary: the mined table beats the composition closure by 0.20–0.45 everywhere (the archive
supplied real information beyond composing `T[l−1]`), and the earned table finishes ahead of the
handed-over true vocabulary at L3 in shallow/mixed (0.227 vs 0.316; 0.207 vs 0.348 — confounded
by co-adaptation, direction consistent).

**5. The scoped diagnosis, which built the twin.** Mining reads the agent's own canonical
repairs (`mine_from="chosen"`), and the reader is exogenous and pinned at 1.000 — so on a deep
archive the mined span is entirely agent-written and the frozen archive regenerates the same
tuples by construction. What this round measured is **re-practicing the same prompts, not
re-reading the same text**: vocabulary-gated *perception* is structurally inexpressible on this
substrate. The negative in finding 1 is therefore scoped to the executor channel, not a verdict
on the reread claim.

## The twin: [`lm/`](lm/README.md) — where the claim's mechanism lives, it reproduces

With an endogenous reader (NTP on RHM, extraction read against exact BP oracles), extraction is
**strictly ordered by level in every arm** (no learner reaches level ℓ+1 before ℓ), and a
**6.4×-re-read frozen corpus is fully renewable** — deep extraction tracks endless fresh data at
zero token penalty, confirmed by a probe-based and a probe-free instrument independently. The
boundary is a **corpus-size wall**: below a threshold of distinct tokens, re-reading is capped,
not slowed — the sub-threshold arms stall at the next level and never move — with achieved depth
roughly logarithmic in corpus size (**~10× distinct corpus per half level**). Full writeup:
[`lm/README.md`](lm/README.md).

## Where this leaves the claim

Taken together: the reread claim holds where the reader itself is what climbs, with a measured
multiplier (6.4× at no penalty; higher multipliers untested above the wall) and a measured
boundary (the corpus's depth capacity). On the executor side, frozen archives renew competence
but not vocabulary — under an instrument whose reader cannot climb, which is the scope condition
to carry forward rather than a refutation. Recorded as strongly suggestive on single seeds;
the twin's two-instrument agreement and its free fidelity gate are the main reasons for
confidence.

## Fragilities (this round)

- `mine_from="chosen"` self-rewrite (finding 5) caps the extraction plateau and produces the
  deep-archive novelty gap; the clean-span variant trivially exhausts the archive in one pass
  with the handed-over reader at 1.000 — the LM twin is the fix, not a flag tweak.
- `n_used` saturates at 5–9 regardless of table size (generator argmax concentration), so
  `n_new_used ≈ 0` is partly an instrument ceiling; `d_aud` and metered error are the
  behavioural readouts.
- The dense baseline is nearly flat (plant guard: parse/infill accuracy flat across all 16
  cells), consistent with ratchet's inert-plant finding — so the monolith comparison is against
  a near-horizontal line.
- `deep__dense_*` L3 auditions floor at exactly 1.000 for many passes — degenerate cells, not
  measurements. Single seed; scheduled commits (the certificate deliberately not under test);
  one archive size (192) at one (v,s,L,m).

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic
modal run rhm/practice/reread/reread.py::gate
python3 rhm/practice/reread/launch_detached.py --fn reread --tag rr_s0 \
    --mixes "mixed,deep,shallow" \
    --arms "reread,fresh,dense_reread,dense_fresh,given_reread" \
    --seed 0 --n-arch 192 --chunk 64 --n-passes 9 --commit-l2 3 --commit-l3 6 \
    --arch-seed 424242 --n-rt 256 --n-score 256 --mine-support 3 --mine-cap 0 \
    --budget 4 --pr-width 16 --g-budget 58 --max-macro-level 3 \
    --gen-lr 1e-4 --gen-steps 20 --n-grad 4 --value-lr-online 3e-5
python3 rhm/practice/reread/analyze_reread.py --tag rr_s0 --fetch --figures
```

Volume: `/data/rhm_practice_reread/rr_s0/` on `rhm-scaling-data`. Figures:
`figures/rr_s0/fig1_yield.png` (yield per pass, commit lines), `fig2_extracted.png` (fraction of
archive content), `fig3_novelty.png` (paired probe), `fig4_competence.png` (metered error per
depth).
