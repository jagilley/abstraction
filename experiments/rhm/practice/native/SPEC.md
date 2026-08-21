# SPEC — native: piping an earned vocabulary back into the planner and the executor as native primitives

**The question in one sentence**: once a vocabulary has been earned outside the learner, can it be
consolidated into the learner's own planner and executor — so the chunk is proposed and run as one
unit without its primitives being enumerated — while the table that stores its identity remains the
address book the next level is mined over (needed for growth, not for performance)?

**Status**: spec, 2026-08-20. Nothing run. **Parent arc**: [`../README.md`](../README.md) (rhm/practice).
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (the 2026-08-20 revision — chunk *identity* vs chunk *corridor*; currently in the working tree,
uncommitted), §18 (distribution is forced; the unit-creating op is discrete and acts on the
conditions of learning), §20 (the port back) ·
[`recurrence_manufactures_confounds`](../../../../ideas/recurrence_manufactures_confounds.md) §5
(the merge op; "a belief is a chunk in the limit of maximally varied demand") ·
[`cerebellar_abstraction_ratchet`](../../../../ideas/cerebellar_abstraction_ratchet.md) §2–§4 (the
claim under test, in its original form: a multi-step operation becomes a single primitive, and sleep
is the distillation window).
**Paper**: Iwane, Hayward, Karunathilake, Buch & Cohen 2026,
`reading/Hippocampal skill memory expansion.pdf`[^private]
(chunk count pins at ~7 while chunk *size* grows, and keeps growing after speed plateaus; hippocampal
θ/γ coupling predicts within-trial chunk content; absent for non-repeating sequences).
**Attribution**: the question — once the arc builds skill/vocabulary pieces, how do they get piped
back into the planner/main model as native primitives, and is that the real MDL-reducing op; the
suspicion that "CLS + sleep" is the easy answer and not the deepest one — is Jasper's (2026-08-20).
The identity/corridor split is the §3½ revision's. The two-port decomposition and the
table-ablation readout came out of the exchange.

## Why this experiment

Every earned-vocabulary node in the arc keeps the vocabulary **outside the learner**: a committed
macro is a controller-side action — a lookup table consumed by an exogenous max-sum operator inside
an exogenous beam — and the generator neither predicts it nor conditions on it
([`../ratchet/`](../ratchet/README.md), [`macros.py`](../ratchet/macros.py)'s header). The beam
expands *every* move at every tip (`width × n_moves` children, value-scored; `beam_moves` in
[`ratchet.py`](../ratchet/ratchet.py)); nothing proposes. So the arc has shown that a vocabulary can
be *earned* (68–98% of given), *nested* (T3 ⊆ T2×T2; premature commit forecloses the next level),
*concentrated* (value is demand-concentration, not coverage), and *maintained* (demand-tracking
recert) — and has never shown it becoming part of the agent. That is gap 6 of the 2026-08-20
synthesis discussion, and it is where the ratchet belief's core claim lives: "once operation A is
compressed into a primitive it serves as input to B" has never been measured here.

One attempt exists and it is the negative to read first: [`../teacher_slot/handle/`](../teacher_slot/handle/)
gave committed entries slots in the generator's own vocabulary — a per-span head predicting "which
entry covers this span (or none)" and a conditioning embedding on visible spans — and the plant moved
*negatively*, dose-ordered by engagement, **even with the true table** (`given_handle` 0.216/0.279/0.364
vs `given` 0.194/0.228/0.346 across eras; `practice_late_handle` era 3 0.398 vs 0.345). Infill was
unmoved everywhere; the L3 audition got worse. Read against the §3½ revision, `handle/` piped chunk
**identity** — an arbitrary binding (which of the many valid tuples this agent uses, with synonym
splits and a 65–84% coverage cap) — into the **executor** as a dense supervised target. The plant's
job is derivable content; a non-derivable binding is a pure lookup with no compression available,
and interference is what a truth-typed organ does with one. `handle/` tested the one port the
type system says should not work. The reading this node is built on (held loosely, not measured):

- **Chunk identity → the planner, as routing.** The native form of a chunk on the planning side is a
  policy that *proposes* the macro as one call, so the primitive level is no longer enumerated. The
  MDL reduction is in the substrate's own currency — groundings and materialisations per solve. The
  Iwane paper's own reading of hippocampal skill memory is a generative model that "anticipates,
  evaluates and organizes future actions", i.e. the planning side; and Dong et al.'s QWM
  (`reading/qwm.pdf`[^private]) is the same shape — policy proposes, model
  imagines, value aggregates, everything trained on real transitions.
- **Chunk corridor → the executor, as a transition.** Given the entry, the span is derivable content
  (the grammar's materialisation of that tuple in context), so it is truth-typed and the executor
  can hold it: one pass materialises the macro's leaves instead of DP-over-T[ℓ−1] plus per-block
  infill. In the latent planner ([`../../RHM_SCULPTING_README.md`](../../RHM_SCULPTING_README.md)
  Stage 3b/5) the same thing is a macro-conditioned one-step latent transition. The licensing
  condition for this port is the merge: a corridor should enter the truth organ once it is
  demand-invariant; before that it is demand-concentration that the truth organ will mistake for
  truth ([`../typed_gaps/`](../typed_gaps/README.md)).
- **Identity itself never transfers.** The table stays as the address book — routing, not pruning
  (§3½): after consolidation the policy calls the address without search and the executor runs the
  corridor without the DP, but the next level is still *mined over the table*. This is the prediction
  worth the most: a consolidated agent should tolerate deleting the table at the current level and
  lose the next level's mining. The hippocampal analog is needed for growth, not for performance.
  (Iwane's ~7 fixed slots growing by nesting is this store's capacity fact; the amnesia
  dissociations are its consolidated end.)

Where this leaves CLS/sleep, stated so the next agent can disagree with it: CLS is right about the
*constraint* (fast arbitrary content interferes with slow statistics; transfer needs interleaving),
and the arc already discharges it for chunk content ("repetition is the interleaving"). It does not
say what *licenses* the transfer (selection upstream; demand-invariance), what *transfers* (routing
and corridor, never identity), or what the final state is (the fast store is not emptied; it stays
as the index). The offline step that matters in this arc already runs every cycle — mining between
bouts, selection over stored traces, the micro-offline replay of Buch/Iwane — and it is selection,
not regression. The consolidation this node adds is wake-time self-imitation of the planner's own
selected choices; §18's "gradient descent is averaging" objection does not bite because the targets
are selected before they are regressed onto. Also worth keeping in view: on the arm the same split
landed as "commit the routing, not the content" inside the FM's horizon
([`mjc/practice/fingering`](../../../mjc/practice/fingering/README.md)), and frozen measured chains
beyond it ([`legato`](../../../mjc/practice/legato/README.md)).

## Reading list (ordered; first three load-bearing)

1. [`../ratchet/README.md`](../ratchet/README.md) + [`macros.py`](../ratchet/macros.py) header +
   `beam_moves`/`build_ms`/`fit_width` in [`ratchet.py`](../ratchet/ratchet.py) — the substrate,
   the pricing idiom (declared grounding budget G picks the width; `collect=True` already returns
   the chosen trajectories), and the plant guard.
2. [`../teacher_slot/handle/`](../teacher_slot/handle/) — `handle_net.py` header (annotate-not-
   substitute, zero init, per-arm streams so twins are bit-identical to the commit cycle), the
   numbers in [`../teacher_slot/README.md`](../teacher_slot/README.md) §5 and the confound list.
3. `practice_manufactures_its_own_credit` §3½ (revised), §18, §20.
4. [`../../RHM_SCULPTING_README.md`](../../RHM_SCULPTING_README.md) Stage 3b (re-grounded latent
   beam: the FM ranks one step from a true latent; only kept tips are materialised) and Stage 5
   (internalisation) — where a macro-conditioned latent transition would go.
5. [`../merge/README.md`](../merge/README.md) round 1 finding 3–4 (the index withholds the next
   level; paced decorrelation licenses merging) and
   [`../crystallize/README.md`](../crystallize/README.md) finding 1 (the library keyed by the
   observed target — the same organ, as a binding key→program).
6. For the LM sibling only: [`../fourwall/lm/README.md`](../fourwall/lm/README.md) (a free key is
   taken instantly; the token-derived pathway is threefold suppressed; the merge op is an
   input-stream collapse) and [`../reread/lm/README.md`](../reread/lm/README.md) (extraction is
   level-ordered; the ~10×-per-half-level corpus wall).

## Substrate

[`../ratchet/`](../ratchet/)'s exactly (v=8, s=2, L=4, m=2; 16 tokens, 8 blocks; base level-1 moves;
three nested eras; G = 58; `rr_s0`'s flags), forked verbatim the way `handle/` forked it — `ratchet/`
untouched, per-arm torch streams so a treated arm and its twin are bit-identical until the port
switches on. **Hold the vocabulary fixed** so the only new variable is the port: run the ports on the
true table (`given`-style) and on `practice_late`'s mined table, as `handle/` did. Whether the
vocabulary should also be re-earned live inside a ported loop is a later round.

## Design (held loosely — the instrument list is the commitment, not the interpretation)

**Port 1 — routing (proposal).** A proposal head π(move | z, r\*) beside the value (same encoder
read), trained by self-imitation on the beam's own chosen trajectories (`beam_moves(collect=True)`
already hands these back). At plan time, expand only the top-k proposals per tip instead of all
`n_moves`. k = n_moves must reproduce enumeration bit-for-bit (the fidelity gate).

**Port 2 — corridor (span execution).** A span-level head on the generator: given context and a
committed entry id, emit the span's leaf tokens in one pass, replacing DP-over-T[ℓ−1] + per-block
infill for that macro. Trained on selected solved traces where that macro fired (self-imitation of the
executor's own materialisations). Must reach DP+infill parity on held-out before it is allowed to fire
in a run. The latent-transition form (Port 2 inside the 3b/5 latent beam) is named, not asked for.

**Arms** (first pass; each treated arm has its bit-identical untreated twin in-tag):

| family | what is new | the question |
|---|---|---|
| `enum` (= `given` / `practice_late`) | nothing | the exogenous reference, re-run in-tag |
| `prop_k` | proposal head; expand top-k per tip | routing as native primitive: groundings + materialisations per solve at matched success; does the freed budget buy depth (width or move budget) at G = 58? |
| `base_prop_k` | the same head over base moves only | **the control that matters**: if amortising search over primitives buys the same thing, the chunk is not the object |
| `span` | span head, DP+infill replaced for committed macros | corridor as native primitive: parity at fewer materialisations; plant guard for interference (the `handle/` contrast) |
| `span_mined` vs `span_true` | span head trained from the mined table's traces vs the true table's | concentration vs coverage on the executor side: does demand-concentrated corridor content interfere where coverage does not? |
| `prop+span` | both ports | the full native primitive |

**Readouts that make this a native-primitive claim rather than a speed claim**:

- **Table ablation after consolidation.** Remove the macros from the action set (and, separately,
  from the span head's reach) late in a run. Routing-not-pruning predicts graceful fallback and ~zero
  loss at the current level for `prop+span`, and a collapse of next-level mining yield. Run both
  directions: ablate the table; ablate the primitives (can the policy solve with macros only?).
- **Next-level currency.** |T3| candidates at `mine_support` and the L3 audition under native L2 vs
  exogenous L2. The ratchet belief predicts native L2 raises L3 minability because chosen
  trajectories are built of L2 calls. Its poison twin: a natively-routed *bad* L2 (early commit) should
  foreclose L3 worse than a frozen bad table, because the policy now routes to it.
- **The can't-decompose signature.** Proposal mass on the primitive decomposition of a chunk the
  policy proposes. Experts cannot decompose their chunks; a policy that still enumerates the
  decomposition has not chunked.
- **MDL, in the existing ledger.** Priced time (`n_ground·d_fb + n_mat·c_mat`) and the width ladder at
  matched success; the span head's one-pass materialisation against s blocks.
- **Plant guard**, every probe: `parse_acc` / `infill_acc` on clean configurations, slot-bypassed
  twin where applicable — `handle/`'s instrument, so the interference contrast is like-for-like.

**Open, deliberately — for the implementing agent to decide, with reasons on the record:**

- Whether the merge licensing belongs in round 1 of Port 2 or is deferred. Ratchet has no venue/index,
  so demand-invariance is not expressible there; `span_mined` vs `span_true` is the proxy. If the
  answer is "it matters", the merge world ([`../merge/`](../merge/)) is where it becomes expressible.
- Whether the proposal head trains **online inside the loop** (proposal quality and the commit
  schedule then interact — realistic, confounded) or **offline on the completed run's logged
  trajectories first** (cleaner, gives the ceiling). Either order is defensible; say which and why.
- Whether to run the LM sibling in this node (`native/lm/`) or as its own: on `fourwall/lm`'s reader
  the native-primitive op is tokenizer expansion — mint a token for a level-2 span into the input
  stream (conditions, not weights; same op class as fwlm1's merge), {substitute, annotate} ×
  {true table, frequency/BPE merges, demand-concentrated subset}, read on d3/d4 yield, the reread
  corpus wall, and pathway suppression with the token withheld. A minted token *is* a free key for
  the level-2 latent, so substitution should reproduce the pathway suppression (cannot decompose, by
  construction) and annotation should not — routing-vs-pruning as a tokenizer decision.

**Ratchet first** (Jasper's call): the table-ablation / L3-mining contrast is the one that says
something about the hippocampal analog rather than about tokenizers.

## What the outcomes might mean (predictions, not constraints)

Recorded per repo norms as the theory's guesses, held loosely: `prop_k` beats `enum` on priced time at
matched success and `base_prop_k` does not close the gap (the chunk is the object); `span` executes at
parity without the interference `handle/` measured (the corridor is derivable content); `prop+span`
survives table ablation at the current level and loses L3 mining (growth, not performance). Outcomes
that would revise the reading are at least as valuable — e.g. `base_prop_k` matching `prop_k` (the gain
was search amortisation, and "native primitive" is the wrong frame for this substrate); `span`
interfering as `handle/` did (the identity/corridor split is not the variable, and the executor rejects
both); table ablation costing the current level (identity is being consulted at execution time and the
"address book" reading is wrong); or native L2 leaving L3 minability unchanged (the ratchet's "A becomes
input to B" is not carried by routing). Bring the numbers back for discussion before writing any
README; idea-doc revisions (§3½ gaining `handle/` as its measured negative case; §18's unit-creating-op
clause gaining a consolidation clause) are queued, not applied.

## Practicalities and orchestration

- Invoke `/run-experiment-on-modal` first; profile `chromatic`; smoke before every detached launch;
  halting procedure for anything >5 min. Single seed first (`experiments/CLAUDE.md`); ranks, signs,
  and multiples of the measured floors are the claims; the in-tag twin pairs supply the floors as in
  `handle/` (stream-noise floor 0.01–0.03 there).
- **Fidelity gates before any arm**: `prop_k` at k = n_moves reproduces `enum` bit-for-bit; the span
  head reaches DP+infill parity on held-out before it fires; every treated arm is bit-identical to its
  twin until the port switches on (per-arm streams; give any new RNG consumer its own generator).
- Structure per `STRUCTURE.md`: this folder is the node; ports/rounds are children with their own
  `README.md` post-discussion and `FILES.md`; this `SPEC.md` stays as the record of what was asked.
  Fork `../ratchet/ratchet.py` verbatim (the `handle/` convention, fork notice at the top); import
  `macros.py`; do not modify `ratchet/` or `teacher_slot/`.
- **Delegation**: one subagent per port, owning the whole build loop — script, debug, smoke, launch,
  reduce to a summary and figures — per `/write-spec-or-prompt`; point it at `/subagent-instructions`.
  You keep the wait and the interpretation. Port 1 and Port 2 are independent and may run in
  parallel; `prop+span` and the ablation readouts compose them and go after. Expect two halts per
  port (launch handle, then reduced results); attach a Monitor on launch; resume a subagent by name
  rather than respawning. Subagents do not touch git.
- Out of scope here, named so nobody reaches for it by accident: re-earning the vocabulary live inside
  a ported loop; the latent-transition form of Port 2; nightly distillation of the table into weights
  (the CLS form — if someone wants it as a control, it is a control, not the treatment).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
