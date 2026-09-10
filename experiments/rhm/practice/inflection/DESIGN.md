# DESIGN — `inflection`: the decisions the code forced, and the ones left open

Companion to [`SPEC.md`](SPEC.md). **No results are interpreted here.** This is the design
note for the substrate fork: what the spec left to the implementer, what the donor's code
decided for us, what is implemented, and what is deliberately still a stub.

**Donors** (untouched): [`../tutti/tutti.py`](../tutti/tutti.py) (the fork's direct donor) ·
`../../rhm_sculpt_precheck.py` · `../crystallize/units.py` · `../ratchet/macros.py` ·
`../native/span/span_net.py` · `../native/prop/prop_net.py` · `../antiphon/questions.py` ·
`../maestro/policy.py`.

---

## 0. Does the spec's "one change" survive contact with the code?

Mostly yes, with one qualification worth stating before anything else.

**It survives** in the sense that matters: nothing above level 1 moved. The tables, `Miner`,
π, the corridor head, the crank, the gauges, the mirror loop, `possible_sets`, `Miner.build`,
`true_tables`, `holdout_set`, `level2_tuples`, `exact_features`, `on_grammar_rate` and every
level-size fact in `tutti/sizing/` are functions of level-1 **features**, and the rule sits
strictly below the feature layer. The bottom inverse map is built from `rules` alone and does
not move (gate R-6).

**The qualification.** `nearest_derivation_cost` (d\*) and `possible_sets` are computed on the
**surface**, so respelling a block changes hamming distances and can change d\*. Two
consequences, both measured offline:

- `corrupt_hier` occasionally lands on a d\* = 0 configuration and `context_instances`
  rejection-samples it away. Under a rule the *same* pool row can flip that verdict, so a
  ruled tag's instance set is **not** the coin tag's: roots differ on **7–9 %** of instances
  (gate R-4c, four families). Within one tag every arm shares one world, so `canon` vs
  `given_rule` **is** matched instance for instance; coin-vs-rule **across tags** is not.
- d\* itself moves slightly: mean 2.3932 (coin) → 2.3177 / 2.3724 / 2.3646 / 2.3255 for
  `const` / `offset` / `ctx` / `rand` at n_ctx = 3, depth 6. This is the SPEC's Q0 "does parse
  ambiguity change under the rule" number, and it is small but not zero.

The **pool** draw is matched exactly, and that is by construction rather than by luck —
see decision 4.

---

## 1. Where an instance's context variable lives

The observation is a bare leaf sequence and the two architectures are fixed
(`_build_generator`, `_build_rich_controller`), so the context had to go somewhere. The three
candidates, with what the code says about each:

| option | what it costs | verdict |
|---|---|---|
| **a side channel per row, alongside `x`** | every beam expansion, `topk`, `gather` and `cat` must carry it. **But the substrate already has exactly one such column with the right semantics: `roots`** (`beam_moves`: `roots.repeat_interleave(width * n_moves)`). The context is a second column of it. Nothing is added to the observation, so the leaf alphabet, `s^depth`, the block grid, the exact DP and every banked number above level 1 are untouched. At `rule=None` no context is drawn and no RNG is consumed. | **CHOSEN** |
| **a prefix token** | the DP requires `leaves.shape[1] == s^L` exactly; a prefix breaks `n_blocks`, the position embedding, the move-set geometry, `_store`'s bijective row code, the era node indices and `holdout_set`. Stealing block 0 instead makes the tree incomplete. Either way the world changes shape, so a `canon` arm and a `given_rule` arm would no longer run on one world — the thing the spec's arm table requires. | rejected |
| **inferable from visible blocks** | free (the surface already testifies to the context: every undamaged block was spelled by the rule), but a *decoder* is a learned or oracle object, and mid-solve the agent's own wrong spellings pollute the evidence. | **kept as an arm, not as the carrier** (`render="decode"`, a stub) |

**What is implemented.** The context is drawn per instance where the instance is drawn
(`_sample_pool_r`, `context_instances(..., with_ctx=True)`), stored beside `roots` in `meter`,
`shadow` and the per-cycle draw, threaded through `plan` → `beam_moves` / `beam_moves_prop` →
`expand_selected*` as `ctx`, and **bound to the renderer immediately before each executor
call**. It is never shown to the generator, the controller, the value head, π or the span head.

**Why "decode" still matters and is kept as an arm.** Handing the executor the true context is
the ceiling's privilege, exactly as `rubato`'s `kin_oracle` is handed the exact inverse
dynamics. A learner that must *read* the context off the surface is the honest version, and it
is a different arm rather than a different world. The decoder is deliberately not written yet:
whether it is exact (intersect the (feature, synonym) pairs consistent with the visible blocks)
or learned is decision 8's business, and the sizing lane's rule family determines whether one
block suffices.

**RNG consequence, and the gate.** All three of `_sample_pool`, `context_instances` and
`corrupt_hier` draw on **local** `default_rng`s, never on the shared torch stream. So the
context draw cannot perturb the shared stream, and with `rule=None` it does not happen at all.
That is what makes the fork gate reachable.

---

## 2. What a rule IS, in code — family-independent by construction

A rule is an integer table **`K` of shape (v, n_ctx)**: `rule(level1_feature, context) →
synonym index`. Every family the spec names — register, agreement, `(offset_f + r) mod m` —
is such a table; what a family chooses is (i) the table and (ii) how a context is obtained
(drawn per instance = "register"; read off a neighbour = "agreement", the `Rule.context_kind`
hook, unused in this rung). **No hook in the fork reads the family name.**

Four placeholder families are registered so the plumbing can be gated, and they are
**placeholders**, not proposals — the sizing lane owns the choice:

| family | `K[f, c]` | why it exists |
|---|---|---|
| `const` | 0 | a rule that IS `canon`: every hook live, every number unmoved. The plumbing control. |
| `offset` | `(f + c) % m` | the SPEC's own sketch at `offset_f = f`. |
| `ctx` | `c % m` | spelling depends on the context and not on the feature — the lowest-dimensional non-trivial rule, and identifiable from one observed block. |
| `rand` | `U{0..m-1}` | no structure to share across features: the "no low-dimensional family" control. |

`make_rule(None)` is the coin and is the default.

---

## 3. The renderer is a duck for `canon`

Every write site in the arc does exactly `canon[feats]` and then `.reshape(...)`:
`units.apply_move:111`, `macros.apply_any:243`, `span_net.SpanExecutor.apply:305`, and this
file's `PerfExecutor._fire_only` / `PerfExecutor.apply`. Three of those five are **donors**.
Replacing the tensor with an object whose `__getitem__` is the rule reaches all five with no
donor edited, and with `render="canon"` and `rule=None` `_renderer` hands back the donor's
tensor itself, so the op is the donor's, not a re-implementation of it.

**The binding discipline.** `__getitem__` receives only `feats`, never the rows, so the
per-row context is bound just before each executor call and consumed by a **cursor** that
follows the donors' in-order chunking (`chunk=16384`, recursive in `apply_move`, `apply_any`
and `SpanExecutor.apply`). Two properties make this safe rather than clever:

- an **overrun** is an assertion, so a bind that is too short fails loudly;
- an **unbound** render is an error by default (`render_strict=True`) rather than a silent
  fall-back to synonym 0 — which is precisely the class of silent wrong number this node
  exists to remove. `render_strict=False` downgrades it to a counted fallback, and
  `n_unbound` is logged in every cycle either way.

`PN.expand_selected` is the one donor that hands the executor `flat[rows]` with the row
indices unrecoverable, so it is forked as `expand_selected_r` (twenty lines, the donor's loop
plus one `_bind`) — the same fork the donor already made for the metered path
(`expand_selected_perf`). At `ctx is None` the **donor's** function runs.

---

## 4. What the damage does under the rule, and the discard-draw discipline

`corrupt_hier`'s damage is **on-grammar and on-rule in the instance's own context**: the
level-above draws (which children) stay coins, because those are derivational choices; only
the bottom `choice` becomes the rule. A damaged instance therefore testifies to its own
context in *every* block, and a spelling error can only come from the executor.

**The discard draw.** Both `sample_derivations_ruled` and `corrupt_hier_r` still **take** the
donor's bottom `rng.integers` draw and then throw it away. This costs one wasted draw and buys
a property worth much more: at the same seed a ruled pool has the **same roots and the same
level-1 features** as the coin's (gate R-4a/R-4b — every feature mismatch is at one of the
bottom map's two colliding codes and there are 0 elsewhere). The context is drawn on a
**disjoint** stream (`seed + 999_331`) for the same reason. The rule moves the surface and
nothing above it.

---

## 5. What still renders at `canon`, on purpose

Two places, both pre-arm and arm-independent, and both stated rather than hidden:

- **the setup value buffer** (`collect_value_buffer` inside `build_shared`, and
  `_collect_task_matched`). Its corpus IS rule-spelled (it is the world's own draw), but its
  behaviour policy renders at `shared["canon"]`. It is built once, before any arm exists, and
  is explicitly the *generic, stale* prior every arm starts from; making it rule-aware would
  hand every arm a renderer it did not fit.
- **`measure_refs`**. Its instance draw takes the rule, so it is still bit-identical to
  `run_arm`'s metering set (same seed → same roots, contexts and damage), but its reference
  beams and its `oracle_rollout` floor render at `canon`. `refs` is an arm-independent
  **meaning** scale anchor, `possible_sets` accepts any synonym, and no spelling number is
  read there.

If either turns out to matter, both are one line from being rule-aware; they are not, today.

---

## 6. The grader's second number

`grade_spelled` returns `(success, residual d*, spell)`. **`success` is `units.grade`'s array,
object for object** — the meaning verdict is the donor's, and mining, π, the certificate, both
pacers and the gate key on it and on nothing else. `spell` is a dict and is read by the log
alone.

**What "wrong" means.** A block is wrong if its leaf tuple is **not one the rule calls for in
this instance's context, for any level-1 feature that could have produced it**
(`rule_ok_table`). Quantifying over features is not a convenience: `generate_rules_distinct`
guarantees the m tuples of a feature are distinct but not that tuples are distinct across
features, and `build_inverse_maps` breaks a collision by last-writer-wins ("parse is then only
approximate"). At v = 8, s = 2, m = 2, rule_seed 0 there are **2 such collisions in 16 legal
bottom tuples**. A verdict read through the inverse map would charge the executor for the
map's arbitrary tie-break. This definition is the spelling twin of `possible_sets`'
any-synonym reduction, which is what makes the two numbers commensurable.

**Two denominators, both reported.**

- `e_sp` / `e_sp_practice` — grader-side, over **every on-grammar block** of the graded
  configuration. Directly comparable to `e` (same set, same cycle).
- `log["spell"]` — the **renderer's own tally over the blocks it WROTE**, which is the SPEC's
  definition and is exact with no write mask, because the renderer knows every write it made.
  It uses the same `rule_ok_table` verdict, so the two numbers differ only in denominator.
  `n_unbound` rides in the same row and must be 0.

`spell_weight` is the strict direction, kept reachable and **0 in every arm this node runs**.
At 0, `grade_spelled` returns the donor's arrays and the strict branch is not executed —
gate GG-1.

---

## 7. `e` does not include spelling (default)

`intonation`'s `e = 1 − mean(got == tgt)` is feature-level, over the span's blocks. Under the
rule spelling is a **third** quantity. `perf_e_spell` exists and defaults **off**: `e` stays
the donor's quantity, the meter, `b(s)`, the agency gate, δ_perf and δ-silence all keep
reading exactly what they read in `tutti`, and the spelling error is logged beside them. The
SPEC's readout "whether the executor-plasticity and commit-pacer seats move when `e` includes
spelling" is then a **counterfactual arm**, not a change to the anchor.

---

## 8. The renderer's training signal — my view, for you to overrule

You asked for a view rather than a default. Mine:

**Fit the renderer from the rule-spelled blocks it OBSERVES in instances, restricted to
configurations it SOLVED. Do not fit it from verdicts on its own written blocks.**

Three reasons, all from the arc's own norms rather than from taste.

1. **Verdicts on its own writes are an oracle channel.** A per-block spelling verdict is an
   experimenter-supplied label on the learner's own output. `macros.py` is explicit that
   nothing above the generator's own competence may enter the vocabulary, and the ratchet mines
   only from configurations the agent actually solved, parsed by its own generator, with
   agreement to the exact map logged and never consumed. A renderer trained on grader verdicts
   would be the first organ in the stack fed a label the world does not volunteer.
2. **It is the wrong twin of `tempo`.** `rubato`'s body model was fitted on executed `(v, a, u)`
   triples from the learner's own practice traversals — its own productions, yes, but with the
   **body's response** as the observable, not a grader's opinion of them. The RHM analogue of
   "the body's response" is the surface the world writes, i.e. the instances.
3. **Restricting to solved configurations is what makes it PRACTICE rather than pretraining.**
   Unrestricted, the observed blocks are just the corpus and the renderer is a second
   pretraining pass; restricted, its diet is exactly the diet the miner eats, it grows with
   competence, and finding 6's curriculum question (identifiability by contexts practiced) has
   something to bite on.

**The alternative is a named arm, not a default**: `fit_rule_verdict`, fitted on per-block
spelling verdicts on its own writes. It is a strictly stronger channel and would be the natural
*ceiling* for the fitted renderer — worth running once the first version exists, and worth
reading against `given_rule` rather than against `fit_rule`.

**Where the reader enters.** The observed-block signal is the same data the reader is trained
on, in the opposite direction. That is why decision 4 of the SPEC (`fit_shared`) is natural
here and not merely tidy: one object, two directions, sharing measurable rather than derived.

---

## 9. What the stubs still need

Both raise `NotImplementedError` at `_renderer` with a pointer here, so no arm can silently
run half-built.

### `leaf` — chunks keyed and replayed as leaf strings

The mined unit today is a tuple of **level-1 features**: `Miner.observe(feats)` counts flat
feature tuples, `make_table`/`_flatten` build `flat` as feature ids, `macro_features` scores
entries by a max-sum DP over the generator's per-block feature logits, and execution renders
`canon[feats]`. A leaf-keyed unit is that with a **spelling carried beside the content**.

The cheapest honest version — and the one I'd build — is **not** to re-key the table but to
make an entry a *pair*: `(feature tuple, leaf string)`. Then

- `Miner.observe` keys on the pair, so the level-ℓ table's entry count multiplies by up to
  `m^span` (×4 at L2, ×16 at L3, ×256 at L4 at m = 2) — and **that blow-up IS the finding**:
  it is what not factoring costs, priced in the arrival budget `tutti/sizing/` already
  measured (E[true L5 keys at support 3] = 0.000 at ~1,600 observations);
- the **DP is unchanged** — it scores the feature half, so `macro_features`, `dp_features`
  and the span head's parity target are untouched;
- **execution replays the stored string** instead of rendering, which is the whole point (the
  force head's twin: content stored in realization coordinates).

Touch set: `MC.Miner.observe` / `.build` / `.state`, `MC.make_table` / `_flatten` /
`true_tables` / `grade_table`, `macros.apply_any`'s render step, `rows_from_flats`,
`extend_candidates`, `apply_surgery`, `refresh_upper`, and the `census`/`assay` readouts that
count entries. Those are all **donor** modules, so `leaf` is a second fork
(`inflection/leafunits.py`) rather than an edit — which is the right shape anyway, since the
unit representation is the variable.

### `fit_rule` — a renderer fitted from the learner's own productions

Needs, in order: (i) the module and its own `torch.Generator` (the `span_net` idiom, so
minting it draws nothing from the shared stream — gate S-1's discipline); (ii) a buffer of
`(observation, block, feature) → synonym` examples harvested per cycle under §8's rule; (iii)
a training call in the cycle beside `finetune_generator_span`, with its optimizer steps priced
or explicitly not priced; (iv) a decision on whether its forward is charged as execution — I'd
tally it into a new `blk_render` counter beside `blk_ref` (measured, never folded into `t`)
until we know what it costs; (v) the identifiability readout: recovered `K` against the true
`K` by contexts practiced, which is `tempo` finding 6's drag-coefficient plot.

`decode` is a third of a stub: it needs only the surface→context map, and the exact version
(intersect the (f, k) pairs consistent with each visible block, majority-vote over blocks) is
~15 lines. It is not written because whether one block identifies a context is a property of
the rule family, which the sizing lane has not handed over.

---

## 10. Decisions left to you

1. **The rule family and `n_ctx`** — the sizing lane's, per SPEC Q0. Everything here is
   family-independent; `make_rule` gains one branch.
2. **Whether `given_rule` gets the true context or must decode it.** Implemented: handed
   (the ceiling, `kin_oracle`'s privilege). `render="decode"` is the honest-information
   variant and is a stub.
3. **§8's training signal** — my recommendation above; the alternative is `fit_rule_verdict`.
4. **Whether `e` ever includes spelling** (`perf_e_spell`) — off, and I would keep it off in
   Q1 so the anchor stays `tutti`'s.
5. **Pricing of the fitted renderer's forward** — §9(iv).
6. **Whether the setup value buffer and `measure_refs` should become rule-aware** (§5).
7. **Whether the L5/arrival numbers need re-deriving under `leaf`** — they will change by
   construction (§9), and that is a sizing question before it is a run.

---

# Q1 — the ordered register, and the four renderers on one world

Added 2026-09-09 after the sizing lane landed. Everything above stands; this section records
the decisions Q1 forced. The coordinator's adoptions: §8's training signal (observed
rule-spelled blocks in solved instances; verdicts are a ceiling arm, fitted OFFLINE from the
log); `e` stays feature-level; the renderer's forward is tallied as `blk_render` and never
folded into `t`; `measure_refs` and the setup value buffer stay at `canon`.

## Q1.1 The family: `E_R8`, replicated not redrawn

`K[f, ρ] = 1{ρ ≥ θ_f}` on an ordered register ρ ∈ {0…7}, θ replicated verbatim from
`phase0_inflection.fam_threshold(8, 8, PARAM_SEED=11)` — **gate E-1** asserts the two tables
are equal, so §2's ceilings apply to this table and not to a cousin of it. Recovered:
**θ = [1, 1, 6, 4, 5, 5, 5, 1]**, `R_eff = 5` of R = 8, every θ strictly interior, and
**K[:, 0] ≡ 0 — ρ = 0 IS `canon`** (gate E-2). The SPEC's own `(offset_f + r) mod m` is out by
the lane's premise 2: at m = 2 any fully shared `a_f ⊕ g(context)` has `R_eff ≤ 2`.

## Q1.2 The curriculum, and what the held-out ladder actually contains

Practised **{0, 3, 7}** (both ends plus one interior — the lexical-invisibility condition),
drawn uniformly per instance; held out **{1, 2, 4, 5, 6}**. Gate **E-4** confirms **0 new leaf
codes** at every held-out register under this practised set.

**Gate E-3 is the finding to carry into the readout**: at this θ the held-out ladder is not
homogeneous. Columns 1, 2 and 3 are *identical*, and so are 6 and 7. So

- **held-out ALIASES of a practised column — {1, 2, 6}**: transfer is free for any arm that
  got ρ = 3 or ρ = 7 right. These are a **built-in null control**, not a test.
- **held-out GENUINELY NEW columns — {4, 5}**: the only real transfer rungs. ρ = 4 differs
  from the practised ρ = 3 in one feature (f = 3, θ = 4); ρ = 5 in four more.

Any claim about transfer must be read on {4, 5} with {1, 2, 6} as the control that says the
plumbing works. This was not visible from the family's name and is why the E-gates print it.

## Q1.3 The five arms — one world, one loop, only the renderer moves

All five carry `tu_y_exo`'s configuration (yield paces both actions, exogenous question head,
fallible span head with the meter on). The rule is WORLD-level and identical across them.

| arm | renderer | what it is |
|---|---|---|
| `canon` | synonym 0 always | the ρ = 0 rung of the family's own scale; a learner that never learned to spell |
| `given_rule` | the true K at the carried register | the ceiling, `rubato`'s `kin_oracle` |
| `leaf` | the chunk's recorded spelling | content in realization coordinates — the force head's twin |
| `fit_rule` | logistic head, register as a **scalar** | the treatment (the lane's `additive_scalar`) |
| `fit_index` | the same head, register as a **one-hot** | `tempo` finding 3's "the tempo input acts as an index", in one bit |

The coin-world fork gate is a **separate tag** (`if_gf`, `rule=None`) and is already PASS at
0.000e+00; `canon` here is not that gate, it is the anchor.

## Q1.4 `leaf` without a multiplied table — the coordinator's design, adopted

The table stays feature-keyed and the DP is untouched (`macro_features`, `dp_features` and the
span head's parity target never see this). A **side table** keyed by the chunk's whole level-1
feature tuple stores the synonym pattern it was recorded with (majority over the rows that
minted it); the `leaf` Renderer replays it; the per-feature majority spelling is the fallback a
base move and an unseen tuple take. **It reaches every write site**: the Renderer's
`__getitem__` receives `feats` of shape (n, span) — the whole span's feature tuple — so the key
is already in its hand, and the lookup is a `searchsorted` on a sorted code array, vectorised
and on-device. No push-back needed.

**The blow-up readout, and a bound worth stating.** `distinct_spellings_{mean,max}` per chunk
replaces building the multiplied table. Under a REGISTER rule every block of an instance shares
one register, so a chunk has at most **one spelling per register it was met in** — the
multiplication is bounded by the **number of practised registers (3)**, not by `m^span`
(4/16/256 at L2/L3/L4). That bound is a property of this family; an agreement rule would not
have it, and neither would a rule whose context varied within a sequence.

**The harvest source, and its measured caveat.** Rows are `mine_src` — the SOLVED
configurations the `Miner` itself keys on — so the side table is co-extensive with the table it
decorates. Those rows are a mixture: the untouched majority is world-spelled and on-rule, the
rewritten span carries this arm's own spelling, so there is a weak closed loop. It is measured,
not argued: `on_rule_frac` (the fraction of harvested blocks that agree with the true rule) is
logged every cycle. The per-feature fallback is counted once per block from a span-1 pass over
the whole configuration, so the per-span passes cannot tilt it toward the era's own cell.

## Q1.5 The fitted head is ONE LINEAR LAYER, deliberately

`σ(w·[one-hot(f) ; ctx] + b)`. With ctx a scalar ρ/(R−1) this is exactly the lane's
`additive_scalar` class (`1{α_f + βρ > 0}`); with ctx a one-hot it is exactly `additive_1hot`
(`1{α_f + γ_ρ > 0}`). So §2's ceilings — **0.54 det / 0.81 acc** at C = 3 for the scalar,
**0.00 det / ~0.60** for the one-hot — are the ceilings for *these* heads. A hidden layer would
represent the same rule and forfeit that comparison, so there is none. Own `torch.Generator`
with the global state snapshotted and restored (`span_net`'s gate S-1 idiom), own numpy stream
for the minibatch draw, so a fit arm is bit-identical to its twin until its first fit step.

The head is a function of (feature, register) alone, so after each fit it is evaluated **once**
on its whole 8×8 domain and cached as `Khat`; a per-block forward would recompute 64 cells per
block for nothing. `blk_render` counts the blocks rendered through it. Before the first fit
`Khat` is None and the arm renders at `canon` — a logged bootstrap, not a silent one.

## Q1.6 The verdict channel is logged and never consumed

A capped sample (512/cycle) of `(feature, register, k_written, k_rule)` on the arm's own
written blocks goes to `log["spell_rows"]`. Nothing in the run reads it. It exists so the
`fit_rule_verdict` ceiling of §8 can be fitted **offline from the log**, and no run has to pay
for it twice.

## Q1.7 Readouts built in

- **Transfer probe** — a fixed instance set per (era, register) at EVERY register, practised
  and held out, solved by the arm's own executor at `probe_every`, graded for meaning **and**
  spelling. The era's own cell supplies the level, so `register × level` fills in over the run.
  Unpriced, `capture` off (a probe must not feed the span head's buffer). It resets the
  renderer's running tally, which nothing reads after block (a).
- **Identifiability** — `rule_head_report` per cycle: recovered K, `acc_practiced`,
  `acc_heldout`, `acc_by_ctx`, the implied `theta_hat` per feature against `theta_true`, and
  whether the recovered table is monotone.
- **Spelling beside meaning** — `e_sp` (grader-side, all on-grammar blocks, the twin of `e`),
  `e_sp_practice`, and the renderer's written-block tally, in their own series.
- **`read_acc` / `parse_acc` by register**, practised and held out, on clean rule-spelled probe
  sets built once per register in `build_shared` — the sizing lane's open item 1.

## Q1.8 Decisions still yours

1. Whether `given_rule` should decode the register instead of being handed it (`decode`, Q2).
2. `fit_shared` — the reader and the renderer as one object (Q2, SPEC decision 4).
3. Whether to run `fit_rule_verdict` from the logged rows once Q1 lands.
4. Whether the curriculum should practise a set whose held-out ladder has more than two
   genuinely-new rungs (E-3): {0, 4, 7} would give novel {1, 2, 3, 5, 6} at the cost of one
   endpoint-adjacent interior. I kept your {0, 3, 7}.

---

# Q1b — separating the rule question from the reader question

Added 2026-09-09 after Q1's readback diagnostic. Q1 found that the fitted heads had learned the
rule **modulo the reader's own confusions** (7 of `fit_rule`'s 15 wrong cells, 11 of
`fit_index`'s 28, are exactly the cells where the true spelling would be unreadable, and in
every one the head substituted a readable spelling), and that `given_rule`'s meaning cost tracks
its unreadable-write fraction (+0.67 / +0.57 across registers) **without being accounted for by
it** (ρ = 5 carries the same 0.363 unreadable fraction and the second-best meaning error).
Q1b removes the collisions one way and repairs them the other, and adds the family contrast.

## Q1b.1 `if_q1b_cf` — the collision-free draw

`rule_seed = 6`. Gate **CF-1** checks the bottom map directly: **0 colliding bottom tuples of
16** at seeds 6 and 10, against 2 at the arc's seed 0. Everything else is `if_q1`'s: the same
`E_R8` θ table, the same practised {0, 3, 7}, the same five arms, the same flags — so the rule
draw is the only thing that moves.

**What this deletes, and it must be read with the tag.** A collision-free draw removes the arc's
native aleatoric channel: junk mass goes to 0 at every level (sizing fact 5), so the L2+ mining
dynamics are **not** `if_q1`'s — table precision, the certificate's trajectory and the arrival
budget all move for a reason that has nothing to do with the renderer. The claims this tag can
carry are therefore **within-tag arm ranks** and **the identifiability readout against its own
ceiling**, not a cross-tag delta against `if_q1` on anything the mining path touches.

An aside recorded because it looks like a contradiction and is not: seeds 6 and 10 carry 6 and
14 duplicate tuples in the LAYERS ABOVE the bottom. Those enter no map this arc reads —
`build_inverse_maps` is consumed only at `inverse_maps[-1]` — and the DP and `possible_sets`
read `rules` directly. The sizing lane's "collision-free" is about parse ambiguity, which is a
bottom-map property, and it holds.

## Q1b.2 `if_q1b_sh` — `fit_shared`, one morphology in two directions

`rule_seed = 0` (the Q1 world; the same seed reproduces the same world and the same instances,
so **the Q1 arms are this tag's comparators**). One arm, one change:

**The harvest's parse.** For a block whose leaf code has more than one owner, keep the owner
`(f, k)` whose `K̂[f, ρ]` calls for exactly that code's synonym index `k`; with zero or two
survivors, leave the reader's answer (`bottom_map`, last-writer-wins) alone. It bootstraps by
construction — a better `K̂` makes a better parse, which makes a better `K̂` — and it is the
minimal honest form of SPEC decision 4: one object, applied forward by the executor and backward
by the harvest, with the sharing measurable rather than derived.

**What is NOT changed, and it matters.** The **execution-side** reader is untouched: the DP
still resolves a macro through `generator.block_logits` (`macros.macro_features` /
`span_net.dp_features`), and the miner still keys on `MC.parse_features(shared["reader"])`. So
this tag moves the *harvest's* readback and nothing else; a full `fit_shared` that also serves
the executor's intent is a later rung.

**Its own instrument.** Per cycle: `n_ambiguous` (colliding blocks seen), `n_relabel` (blocks
the rule re-labelled), `n_relabel_determined` (of those, how many the TRUE rule uniquely
determines) and `n_relabel_correct`. The last two are oracle readouts, logged and never
consumed. Gate **SH-1** checks the rule on a synthetic block set with the true table in K̂'s
place: every block the true rule determines is recovered exactly, every block it does not is
left at the reader's answer, and the reader alone would have got 0.837 of the determined set.

## Q1b.3 `if_q1b_a` — the GF(2)-affine contrast

The sizing lane's `A_2class` at its `PARAM_SEED` (gate **A-1** asserts equality): four
registers, two shared response vectors, `R_eff = 4`, and gate **A-2** exhibits a 2×2 XOR
submatrix — so the family is **`additive-real? = no`**: a real-valued additive renderer cannot
represent it at all, which is exactly the point. On the collision-free draw (`rule_seed = 6`),
so the reader question is out of the way. Practised {0, 1, 2}, held out {3}; gate **A-3** shows
**0 new leaf codes at the held-out register for every choice of it**.

Five arms, one loop, only the head moves:

| arm | head | the class the sizing lane names |
|---|---|---|
| `given_rule` | the true K, handed | the ceiling |
| `fit_index` | linear over one-hot register | `additive_1hot` — pins nothing off the practised set |
| `fit_scalar` | linear over the two register BITS as reals | the class §2 marks `--`: cannot fit even two practised contexts of a parity rule. Given the chance anyway |
| `fit_gf2` | counting solver over GF(2): eight candidate `(a_f, w_f)` triples per feature, most-agreement wins, ties and unobserved features fall back to canon | `affine_gf2` — **1.00 / 1.00 at C = 3**, because three points span AG(2,2) |
| `fit_mlp` | one hidden layer of 16 over (one-hot feature, register bits) | the GENERIC renderer, the one with no family assumption |

The context carrier is unchanged (the per-instance side channel riding with `roots`); only
`make_rule` and the heads move. `ρ = 0` is this family's canon rung too (`K[:, 0] ≡ 0`).

## Q1b.4 Decisions taken here

1. **`fit_gf2` is a counter, not a network.** The lane's ceiling is computed by enumerating
   admissible completions; a counter reaches it or does not for a reason that is about evidence
   rather than about optimisation. It shares the `RuleHead` interface (`table`, `fit`) so the
   renderer, the cache and the readout are untouched.
2. **A tie leaves a feature undetermined and it falls back to canon** rather than to an
   arbitrary triple — the same bootstrap every other head takes before its first fit, logged as
   `n_undetermined`.
3. **`fit_scalar` gets the bits, not the register index.** Handing it ρ ∈ {0..3} as one number
   would test an ordering the family does not have; handing it the two bits is the fairest
   real-valued reading of a GF(2) context, and is what makes "cannot represent it" a
   measurement rather than a strawman.
4. **The shared parse is one-directional in this tag** (§Q1b.2) and that is stated in the
   record rather than left to be discovered.

## Q1b.5 Still open

- A `fit_shared` that also serves the **executor's intent** (the DP's `block_logits` read),
  which is where Q1's `given_rule` meaning cost actually lands.
- What distinguishes ρ = 5 in Q1's `given_rule` profile — same unreadable fraction as ρ = 3,
  far better meaning error. Not located.
- `decode` (the register read off the surface) — still Q2.


---

# Q1c — an apparatus confound on the meaning currency, and the knob that removes it

Added 2026-09-10.

## Q1c.1 The diagnosis (the orchestrator's, from `if_q1b_cf` §1I at L1)

Within each arm the probe's meaning error rises with register **for exactly the arms that write
synonym-1 tuples** — `given_rule` 0.30 → 0.56, `fit_rule` 0.25 → 0.58, `fit_index` 0.27 → 0.48
from ρ = 0 to ρ = 7 — and is flat for the two that never do: `canon` 0.17–0.31 while
misspelling **100 %** of its writes at ρ = 6, 7, and `leaf` 0.14–0.33 while writing the ρ = 3
column. Meaning is spelling-agnostic by construction (`possible_sets`; and at `rule_seed 6`
there are no bottom collisions at all), so the gradient cannot enter through the grader. It
enters through the **planner's organs**, and DESIGN §5 named the suspect in advance: the stale
value buffer is collected with the `canon` executor, so a rollout that writes synonym 1 is off
the value head's own training distribution. `tempo`'s apparatus lesson — train the head on what
it will be handed at deployment — in our clothes.

**Consequence, stated plainly: the MEANING ranks in `if_q1`, `if_q1b_cf` and `if_q1b_a` are not
interpretable as run.** The spelling and identifiability readouts are unaffected — they never
route through the value head.

## Q1c.2 The knob

`--setup-render {canon,rule}`, `cfg["setup_render"]`, resolved by `setup_render_mode(cfg, rule)`.

- **`canon`** (default) — the donor's behaviour; every `if_q1`-era config reproduces.
- **`rule`** — every pre-arm organ that sees WRITTEN tuples renders through the **true** rule at
  each row's own register. Arm-independent, and on-distribution for the WORLD, so a
  **misspelling** executor is the one off-distribution — which is the right way round.

**Which organs.** Audited rather than assumed: the only pre-arm code path that renders written
tuples is the value-buffer collector, in both of its forms (`collect_value_buffer` inside
`build_shared`, and `_collect_task_matched`), each through `behavior_step` → `units.apply_move`
→ the render step. `_train_edit_controller`, `_train_generator` and `train_reader` all consume
the CORPUS (already rule-spelled) under masking and render nothing. `measure_refs` renders but
trains nothing — it is a scale anchor read by the reduction, not by any organ — and is left at
`canon`, stated here rather than left to be discovered.

**Inertness.** With `rule=None` the knob resolves to `canon` by construction — a coin world has
no rule to spell with — so an unruled tag cannot be perturbed by setting it. Gate **SR-0**.

**Its own measurement.** The buffer now carries the register each state was written at, so
`setup_render_check` in `setup.json` reports `buffer_on_rule`: the fraction of the buffer's
ON-GRAMMAR blocks that carry the synonym the rule called for, at the row's own register
(`_corrupt`'s random blocks are off-grammar and excluded by `spell_error`'s own scoring). At
`setup_render="rule"` this is the gate the fix has to pass.

## Q1c.3 The two tags

**`if_q1c_pr` — the mechanism probe.** `canon` and `given_rule` only, on the `cf` world
(`rule_seed 6`, `E_R8`, practised {0,3,7}), yield-paced exactly as `if_q1b_cf`, with
`--setup-render rule`. The readout is the within-arm ρ-gradient of the L1 probe's meaning error.
If `given_rule`'s flattens to `canon`'s — or inverts, with `canon` now paying at ρ ≥ 1 — the
confound is located.

**`if_q1c_yk` — the five arms on one clock.** The same five renderers on the `cf` world with
`--setup-render rule`, **schedule-paced**: `commit="delta_prov"` and no `loop`, which is the
arc's own schedule idiom — a provisional commit at each era boundary and an advance at the era
cap — so every arm commits and advances at the SAME cycles and `e`, the solves and the probes
are read at identical eras and table ages. Arms `canon_s`, `given_rule_s`, `leaf_s`,
`fit_rule_s`, `fit_index_s`; ladder sized from `if_q1b_cf`'s realised commits (L2 c10–c37, L3
c57–c81) so every arm reaches its L3 era.

**The clock is exogenous**, so this tag's MEANING ranks are a matched-clock claim and nothing
more, and its SPELLING ranks should reproduce `if_q1b_cf`'s — that is the check that the pacing
change did not move the thing it was not supposed to move.

## Q1c.4 A design note, for the record and not for action now

`if_q1b_a`'s held-out register r = 3 is the **all-ones column** of `A_2class`, so any head that
outputs 1 at an unseen register scores 1.000 there. That is the artefact behind `fit_scalar`'s
held-out 1.000 (already flagged in the FILES record as an artefact of its collapse to a constant
row). **A rerun should hold out r = 1 or r = 2.**

---

# Q2 — the curriculum ladder

Added 2026-09-10. `tempo` finding 6's twin: the plant's body model was identifiable only once
practice **spanned two tempi**. Here the rungs are **contexts practised**, and everything else
is held at `if_q1c_yk`'s design — `cf` world (`rule_seed 6`, collision-free bottom map),
`E_R8`, `--setup-render rule`, schedule-paced, the same 140-cycle ladder, single seed. Only
`--practiced` moves. Each rung needs its own `build_shared`, because the pretraining corpus is
drawn in the practised registers, so the ladder is two tags plus the banked C=3.

## Q2.1 Which registers each rung practises, and why

- **C=1 → {3}, not {0}.** ρ = 0 is this family's own canon rung (`K[:, 0] ≡ 0`), so practising
  it alone would be `canon`'s world with extra steps. At ρ = 3 each feature is heard in exactly
  one of its two spellings and **both synonym indices are practised across features** (three
  features spell 1, five spell 0) — the balanced single-context rung.
- **C=2 → {0, 7}.** The two ends of the scale: every (feature, synonym) pair is heard, so every
  leaf code is in the corpus, and every threshold is bracketed on (0, 7].
- **C=3 → {0, 3, 7}**, banked as `if_q1c_yk`.

## Q2.2 What C=1 gives up, stated in advance

Gate **E-5** measures it: at C=1 only **8 of the 16** leaf codes are ever in the corpus, and
**18** codes are first seen at a held-out register (by ρ: 0:3, 4:1, 5:4, 6:5, 7:5; ρ = 1 and 2
show 0 because their columns alias the practised ρ = 3). So **the held-out registers are
lexically VISIBLE at C=1 by construction** — the condition E-4 asserts at C=2 and C=3 cannot
hold at C=1, because one spelling per feature is never heard.

The consequence is a change of status, not a defect: **`read_acc` by register is a readout at
C=1, not a gate**, and a drop there **is the phenomenon** — finding 6's "the drag term sits
below the practice noise at one tempo" — rather than a bug to fix. E-5 asserts invisibility
only at C=2 and C=3, and reports at C=1.

## Q2.3 Nothing new was built

`--practiced` is an existing knob, gated since Q1 (`practiced_set`, threaded through every
draw and through `build_shared`), and both arms are `if_q1c_yk`'s, already run at full scale.
The only addition for Q2 is gate E-5. **No GPU smoke was run and none was needed** — the smoke
discipline is for new mechanism, and there is none here.

## Q2.4 What the rungs are read against

The sizing lane's `E_R8` §2 ladder gives the ceilings by contexts practised:

| class | C=1 | C=2 | C=3 |
|---|---|---|---|
| `additive_scalar` (`fit_rule_s`) | 0.74 | 0.79 | **0.81** |
| `additive_1hot` (`fit_index_s`) | ~0.56 | 0.59 | **~0.60** |

C=3 is banked: `fit_rule_s` **0.875**, `fit_index_s` **0.750** held-out — both above their
ceilings, which the lane states as averages over practised subsets rather than for this one.
The lane's interpolated-vs-extrapolated split is the second axis: for a monotone rule an
unpracticed register INSIDE the practised range is harder than one outside it, and {0, 7}
brackets everything while {3} leaves ρ = 0…2 outside on one side and 4…7 on the other.

---

# Q3 — the two readouts the SPEC never had a seat for

Added 2026-09-10. Two independent questions, two tags, one shared piece of new machinery (a
per-slot record that nothing consumes).

## Q3.1 F2 — trust against habit, per slot

`tempo` finding 5's twin. The question is whether π's mass on a committed slot tracks that
slot's **reliability** or merely its **use count**. Neither number existed per slot: `log["perf"]`
carries `bench` (b(s)) per slot, and `log["probe"]["pi"]` carries π's mean mass per slot, but
the beam's *use* of a slot and its *solve rate when used* were never written down, and the
renderer's spelling tally was global.

**What was built** — `log["f2"]`, one row per cycle per arm, each row a dict of cells keyed by
macro slot:

| column | where it comes from | cost |
|---|---|---|
| `n_used` / `n_calls` | `out["seq"]` — the chosen move sequence the beam already returned | none |
| `solve_used` / `solve_unused` | `ps` restricted to the instances that did / did not take the slot | none |
| `written` / `wrong` | per **register**, from the renderer's new per-(slot, register) tally | none |
| `pi` | `probe["pi"]["mean_mass"][sid]` — a lookup into a distribution the probe already computed | none |
| `age`, `open`, `level`, `node` | `move_age`, `slots` | none |

`solve_unused` is logged so **use count can be partialled out at read time** rather than by a
model chosen now. `pi` appears only on probe cycles — the head's distribution is computed
there and only there — so an F2 row without `pi` is a non-probe cycle, not a missing value.
`base_written` / `base_wrong` carry the same registers' spelling parity for the writes that
happened **outside** every macro that cycle, so a slot's number is read against the same
renderer's behaviour elsewhere in the same cycle.

**LOG, DON'T CONSUME.** No line below the F2 block reads `log["f2"]`. The slot label rides on
the renderer (`canon.slot`, set by `_bind` from the move) and the tally is a `bincount` over
the register column the renderer already had in hand.

**Arms**: `leaf_s`, `canon_s`, `given_rule_s` — schedule-paced, so all three commit and advance
at the same cycles and a cross-arm read of π's mass on the *same* slot at the *same* cycle is
well posed. `given_rule` is the reference the coordinator asked for: mass on the same slots
under a renderer that never misspells.

## Q3.2 E′ — does the execution currency charge for spelling?

Every prior tag's `e` is `intonation`'s: `1 - mean(feature match)`. Spelling never entered it,
so δ-silence — the mean of b(s), the currency the mirror seat's ADVANCE runs on — has never
been able to see a misspelling. E′ turns exactly one bit.

**`perf_e_compose`**: a block counts as executed-as-intended iff its **feature** matches *and*
its **synonym is the rule's**, i.e. `e_spell = 1 - mean(match & ok)`. The two coincide exactly
when every matched block is correctly spelled (gate PE-1), and `e_spell ≥ e_feat` always.

**Three properties held by construction**:

1. **Inert without a rule** (gate PE-0). A coin world has no spelling to be wrong about, so no
   unruled tag — the G-F replay included — can be perturbed by the knob.
2. **Off by default**, at run level and in every pre-Q3 arm, so `if_q1` … `if_q2` reproduce
   bit-for-bit.
3. **The other number is always logged.** A second `PerfMeter` (`pmeter_sp`), identical in
   every constructor argument, is fed `e_spell` on the same keys, the same rows and the same
   moments, and is stepped on the same clock (same calibration draw, same per-cycle reset).
   It draws no RNG — `score` has none and `sample_rows` is never called on it — so its mere
   existence cannot move an arm. Its reduction, `dsil_sp` = mean b(s) over the OPEN slots, is
   in the panel of **every ruled metered arm**, driven in none.

**Why the shadow exists**: it is what makes the floor derivable. `tol_dsil` was measured by
null-ABBA on a `dsil` series with no spelling in it; an arm that DRIVES on the spelling-charged
series needs a dead zone measured on *that* series. `dsil_sp` from the off arms is that series.

**Arms** — `tu_m_exo`'s mirror seat (yield licenses the COMMIT, δ-silence the ADVANCE,
`dsil_bootstrap` on), so the pacer is fixed and only the renderer and the one bit move:

| arm | render | `perf_e_spell` |
|---|---|---|
| `m_given_rule` | given (never misspells) | off |
| `m_fit_rule` | fitted head | off |
| `m_leaf` | leaf side table | off |
| `m_fit_rule_sp` | fitted head | **on** |
| `m_leaf_sp` | leaf side table | **on** |

`perf_e_spell` is an **arm-level** cfg key (it overrides the run-level flag through `run_arm`'s
`{**cfg, **spec["cfg"], **overrides}` merge), so a pair could share one tag. **The two-stage
launch exists for the FLOOR, not for the arms.** Every arm takes `enum_live`'s stream, so an
`_sp` arm is bit-identical to its off twin until the first cycle on which the executor writes a
misspelled block — the in-tag twin gate, and the only cycle at which `e` can first differ.

## Q3.3 Decisions the code forced

1. **`e` at the executor, not at the grader.** The spelling verdict already existed per block
   inside `Renderer.__getitem__` (`rule_ok_table` lookup); it is now kept as `last_ok` and read
   by `PerfExecutor.apply`. The render call was **hoisted above** the `e` computation so the
   verdict is the one for the tuple actually written. `self.last` now carries `e_feat` and
   `e_spell` beside `e`, so the consumer that reads `last["e"]` gets whichever one is driving
   and the other stays readable.
2. **`ps`, the meaning verdict, does not move.** Mining, π, the gate, the pacers and the grader
   all key on `possible_sets` exactly as before. E′ touches the EXECUTION currency and nothing
   else.
3. **The shadow carries the OTHER currency, always.** First cut had the shadow bench the
   spelling-charged error unconditionally, which makes `dsil_sp == dsil` in an E'-ON arm and
   loses the feature-only reading exactly where it is needed for the paired contrast. The
   shadow now benches whichever error the arm is **not** driving on, and the panel names its
   series **by content**: `dsil_sp` is always the spelling-charged reduction, `dsil_ft` always
   the feature-only one, and `dsil` stays the DRIVEN one so the pacer's own record is
   unambiguous. Both numbers therefore exist in every ruled metered arm, in both arms of every
   pair.
4. **The shadow is stepped, not merely fed.** Left unstepped, `PerfMeter.cal_active` grows by
   up to 512 floats per executor call forever. It is calibrated and reset on the driven meter's
   own clock, and its row is logged as `log["perf_sp"]`.
5. **`reset_tally` clears the per-slot table too**, so a block-(a) read of `slot_tally()` covers
   exactly that cycle's practice plan and nothing carried over.

## Q3.4 Two preflight-only arms, and why they had to exist

`preflight` on the toy substrate ran `m_given_rule`, `m_leaf`, `m_leaf_sp` clean — and
**measured that all three tested nothing**: at forty training steps a loop arm never commits,
so no slot is ever minted, `pmeter.n_call = 0` and `dsil = None` in all 27 cycles, and
`log["f2"]` is empty because there are no macro moves at all. Both new paths would have gone
untested before a multi-GPU-hour launch. `tu_pf_split` and `dsil_pf_*` exist for exactly this
reason, and `m_pf_off` / `m_pf_sp` are their Q3 twins: the mirror seat with `vocab: "true"`,
so every slot is minted at c1 and the executor is live from c2. They differ from each other in
one bit and in nothing else. `leaf_s` (schedule-paced) commits at c6 even on the toy substrate
and is what exercised the F2 row there.

## Q3.5 What the floor turned out to depend on (measured, stage 1)

The spelled dead zone is **0.00507228** (pooled null-ABBA over the three E'-off arms, span 1,
W 4, commit and advance cycles dropped, 271 windows), against the **0.0046** the run was
governed by; the same tag's re-derivation of the FEATURE-ONLY floor is **0.00235975**.

The result that governs how stage 2 is read: **`dsil_sp` and `dsil_ft` are bit-identical in two
of the three arms.** `given` never misspells by construction. The fitted head never misspells
*inside the practised set* — and the priced beam draws instances **only** from `practiced_set`,
so its written-block spelling error is 0.000 at rho 0/3/7 over 152,072 blocks. Only `leaf`
misspells where the meter can see it (0.208 / 0.172 / 0.820 by register).

So E' has a domain: it can only move an arm whose renderer misspells **within the registers
practice actually visits**. `m_fit_rule_sp` is therefore expected inert and is the control;
`m_leaf_sp` is where the bit has something to move. That is a property of this world's
curriculum, not of the knob, and it is why the transfer probe (which does visit held-out
registers) is deliberately excluded from the meter — probes take `_fire_only`.

## Q3.6 Gates added

| gate | asserts | status |
|---|---|---|
| **PE-0** | `perf_e_mode` is off by default and **inert at `rule=None`** | PASS (`gates_cpu`, now **67** checks) |
| **PE-1** | `perf_e_compose` on synthetic arrays: `e_spell ≥ e_feat` always, equal exactly when every written block is on-rule | PASS |
