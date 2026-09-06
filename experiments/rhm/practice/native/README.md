# native — the port back: routing and corridor consolidate; the table stays as the address book

**Up**: [`../README.md`](../README.md) (practice arc) · **Spec**: [`SPEC.md`](SPEC.md) (the record of
what was asked, 2026-08-20) · **Files**: [`FILES.md`](FILES.md)
**Idea docs**: [`practice_manufactures_its_own_credit`](../../../../ideas/practice_manufactures_its_own_credit.md)
§3½ (chunk *identity* vs chunk *corridor*), §18 (the unit-creating op acts on the conditions of
learning), §20 (the port back) ·
[`cerebellar_abstraction_ratchet`](../../../../ideas/cerebellar_abstraction_ratchet.md) §2–§4 (the
claim under test: a multi-step operation becomes a single primitive).
**Paper**: Iwane, Hayward, Karunathilake, Buch & Cohen 2026
(`reading/Hippocampal skill memory expansion.pdf`[^private]).
**Substrate donor**: [`../ratchet/`](../ratchet/README.md), forked verbatim per child;
`ratchet/` and `teacher_slot/` untouched. **Measured negative built against**:
[`../teacher_slot/handle/`](../teacher_slot/README.md) (finding 5 there).
**Runs**: `np_s0` (Port 1, 12 arms), `sp_s0` (Port 2, 5 arms), `nf_s0` (composition + ablation,
7 arms), 2026-08-20→21, plus selfchecks and smokes. **Ranks, signs,
and multiples of the measured floors (0.001–0.027 in-tag) are the claims.** One orchestrated
conversation; each port was built end-to-end by a delegated implementer agent and the machinery
details below summarize their work (children carry `FILES.md` records, no READMEs by policy).

## The question

Every earned-vocabulary node in the arc kept the vocabulary **outside the learner**: a committed
macro is a lookup-table entry consumed by an exogenous max-sum operator inside an exogenous beam
that expands *every* move at every tip. The arc had shown a vocabulary can be earned, nested,
concentrated, and maintained — and never shown it becoming part of the agent (gap 6 of the
2026-08-20 synthesis). This node asks: can the vocabulary be consolidated into the learner's own
**planner** and **executor** — the chunk proposed and run as one unit, its primitives no longer
enumerated — while the table that stores its *identity* remains the address book the next level is
mined over (needed for growth, not for performance)? The one prior attempt, `handle/`, had piped
chunk **identity** (an arbitrary binding) into the executor as a dense supervised target and moved
the plant negatively, dose-ordered. The §3½ reading this node instantiates: identity → the planner,
as *routing*; corridor (derivable content) → the executor, as a *transition*; identity itself never
transfers into weights.

## Design in brief

Ratchet's substrate exactly (v=8, s=2, L=4, m=2; three nested damage eras; G = 58; `rr_s0`'s
flags), vocabulary held fixed so the port is the only variable: the true table (`given`-style) and
`practice_late`'s in-tag mined commit schedule (c27/c57), as `handle/` did. Three children, each a
verbatim fork with one insertion:

- **[`prop/`](prop/FILES.md) — Port 1, routing.** A proposal head π(move | z, r\*) beside the value
  (same encoder read), trained **online** by self-imitation on the beam's own chosen trajectories
  (`collect=True`); at plan time only the top-k proposals per tip are expanded, width refit to
  G = 58 under effective branching k. Practice-only exploration (ε per the substrate's own
  convention) added after a smoke measured closed-loop lock-in — a move never proposed never enters
  a solved trajectory, so never becomes a target.
- **[`span/`](span/FILES.md) — Port 2, corridor.** A span head **on the generator trunk**
  (deliberately — the interference question is the treatment), conditioned on the macro slot
  (level, node), emitting the span's level-1 features rendered through the same `canon` the DP
  uses — so no arbitrary label is ever a target. It replaces DP-over-T[ℓ−1] + per-block infill for
  a macro only after clearing a per-macro parity gate (held-out exact-match vs `apply_any`,
  τ = 0.95, re-checked every cycle).
- **[`full/`](full/FILES.md) — composition + the readouts.** Forks `prop.py`, imports
  `span_net.py`; both ports on; end-of-run ablation battery on the final trained state; the poison
  pair.

**Gates** (all bit-for-bit): `prop_k` at k = n_moves reproduces enumeration exactly (whole-run
max|Δe| = 0.0); the composed loop with both ports nailed shut does too (`given_fid`, in-tag);
every treated arm is bit-identical to its twin until its port switches on (per-arm streams; every
new RNG consumer on its own generator); untreated arms are bit-identical **across all three tags**,
so cross-tag comparisons are exact. **Pricing said out loud**: a proposal read below the root
re-uses an encoder state the value already paid for (charged 0; the root encode is charged in
full and `fit_width` sees it; at the conservative bound c_prop = c_mat the headline moves ~1%,
reordering nothing); a span-head call and a DP call both cost one ledger materialisation, so Port 2
buys *nothing* on priced time by construction — its s-block counterfactual (`blk_dp` vs `blk_head`)
is tallied beside the ledger, never folded in.

## Findings

1. **Routing beats enumeration outright at matched price — the freed budget buys width.**
   At budgets matched to 0.3% (57.0 g + 52.0 mat + 13.0 π per solve vs enum's 57.0 g + 56.0 mat),
   `given_prop_k4` runs **e = 0.084 / 0.105 / 0.156** across eras against `given`'s
   0.194 / 0.228 / 0.346, because k = 4 affords width 4 where 14-move enumeration affords width 1
   at G = 58. Cost to merely *match* enum's success: **0.32× the groundings** (0.30–0.53× across
   arms and eras). Matched-priced-time margins over `never_base` at era 3: `given_prop_k4` +0.504,
   `practice_late_prop_k4` +0.501, vs enum `given`'s +0.328. Routing needs search under it: k = 1
   (pure argmax-following) never reaches enum success at any width; k = 2 is marginal. (`np_s0`)
2. **The chunk is the object, not search amortisation — the control the SPEC called decisive.**
   The same head over base primitives only (`never_base_prop_k4`) buys −0.077 at era 3 where the
   chunk-holding arms buy −0.190 / −0.198: **routing-with-chunks beats routing-over-primitives
   2.2× / 1.7× / 2.5×** across eras at exactly matched cost (under a top-k filter, cost depends on
   k, not on action-set size — the matching is exact, not approximate). (`np_s0`)
3. **The can't-decompose signature is present — the arc's first psychological readout.** On states
   where π's argmax is a macro, its mass on that macro's own primitive decomposition falls to
   **0.119–0.131** by eras 2–3 (from 0.22–0.38 early) while macro mass rises to 0.80–0.85. The
   base-only control's head is comparably peaked, so low ratio ≠ flat head; `practice_late_prop_k4`
   shows the same descent starting only after its commit. A policy that proposes the chunk no
   longer proposes its spelling. (`np_s0`, `nf_s0` — composed arm reaches 0.119)
4. **The corridor consolidates into the executor without `handle/`'s interference — the
   identity/corridor split is the variable.** With the span loss reaching the shared trunk
   (verified, core grad 15.7), Δparse on clean configurations is **+0.0068 / +0.0001** (at or below
   the 0.01–0.03 floor) against `handle/`'s **−0.094 / −0.044** at 3–9× floor, dose-ordered, on the
   same substrate and instruments; infill unmoved; span-bypassed competence delta −0.0008. The head
   reached parity and served 27–42% of macro executions at **level 2**; level-3 heads mostly never
   cleared τ (end parity 0.76–0.94, open 0–14 of 90 cycles) — the corridor is only partially native
   at depth, and the gate flaps because parity sits on τ (measured cost ≈ 0). Block-level
   counterfactual at era 3: **12–16% fewer block-infills** than the DP path at identical ledger
   price. `span_true` era 3 lands at 0.318 vs `given`'s 0.346 (earned-vs-given 1.087);
   `span_mined`'s era 3 runs +0.037 over its twin — above floor, unresolved (its A_true moved
   +0.100 at L3, so the plant itself drifted in that arm). (`sp_s0`)
5. **After consolidation the table is an address book: deleting it costs performance ~nothing and
   growth everything it can structurally cost — the SPEC's headline prediction, measured in both
   directions.** End-of-run battery on the final trained state, cost-matched at 57.0 g/solve:
   removing the committed table (head serves every macro call, DP unreachable — `blk_dp` = 0
   verified) costs the native arms **+0.000 / +0.003 / +0.008 e**, where the untrained-head
   negative control (`given_fid`) pays **+0.40–0.51** and enum arms have no execution path at all.
   Under the same deletion the next level's **built** entries collapse (structural: `T3 ⊆ T2×T2`
   at build time) while the **observation stream** — distinct L2-shaped tuples at `mine_support`
   in the condition's own chosen trajectories, keyed table-free — is unchanged (+0); the untrained
   control *loses* 5 (bad execution degrades the stream itself). The other direction: primitives
   ablated, the native arms still solve on macros alone (+0.02–0.07). Routing-not-pruning, both
   halves. (`nf_s0`)
6. **The poison prediction inverted, informatively: native routing quarantines a bad address
   rather than amplifying it.** The SPEC guessed a natively-routed bad L2 (the `practice_early`
   1-entry, precision-0.000 table) forecloses worse than a frozen bad table "because the policy now
   routes to it." Measured: the routed arm is **better in every era** (Δe −0.096 / −0.104 / −0.106,
   3.5–4× floor), with next-level currency unchanged-to-slightly-up. Mechanism: enumeration is
   *forced* to materialise the bogus macro at every tip (33.3% of materialisations); π, trained
   only on solved trajectories, starves it to 0.011–0.058 mass — a **6.0× reduction in bogus-macro
   calls**. Selection-before-regression, the licensing condition for the whole consolidation, is
   also a filter against a poisoned vocabulary. What stands untouched: the representational
   foreclosure itself (zero built L3 in era 2, both arms — ratchet finding 7). (`nf_s0`)
7. **Native routing raises next-level minability — "A becomes input to B," carried by routing.**
   Under k = 4 routing, |T3| candidates at support rise (+3.4 for `practice_late_prop_k4`, +2.8
   for `never_base_prop_k4`) and the committed tables improve at both levels: L2 9 entries,
   recall 0.571, precision 0.889 (twin: 8 / 0.500 / 0.875); L3 15 entries, recall 0.214–0.232,
   precision 0.800–0.867 (twin: 12 / 0.161 / 0.750), with better candidate auditions. Chosen
   trajectories built of L2 calls are better mining substrate. (`np_s0`, reproduced in `nf_s0`)
8. **The plant guard held everywhere routing lives, and the composed system is the round's best
   agent.** Reader 1.000, parse 0.615–0.653, infill 0.685–0.717, flat, all 24 arms across the
   three tags; no dose-ordering anywhere. `given_native` (both ports) runs 0.106 / **0.088** /
   0.173 — era 2 better than either port alone; era 3 `practice_late_native` runs +0.058 over
   routing-alone (unresolved; the span-bypassed probes say the head's firing is neutral, so the
   candidate mechanism is trajectory divergence, not head damage). (`nf_s0`)

## Interpretation (discussed with Jasper 2026-08-21 — argued, not measured)

- **(a) `handle/` is retro-diagnosed as a type error, not a fact about consolidation.** The two
  ports carry exactly the §3½ decomposition: routing (identity's *use*) into the planner, corridor
  (derivable content) into the executor, and the binding itself into neither. Both ports land where
  the identity port failed, on the same instruments. The executor holds what is derivable and
  rejects what is arbitrary.
- **(b) The hippocampal-analog reading survives its first direct test, with the structural caveat
  stated.** "Needed for growth, not performance" was measured in both directions (finding 5). The
  growth half is partially structural in this substrate — nothing *can* be built over a deleted
  table — so the informative content is the split: execution, routing, and even the mining
  *stream* no longer consult the table; only the next level's representation does.
- **(c) Selection is doing more work than the licensing argument asked of it.** §18's
  selection-before-regression clause justified consolidation as *safe*; finding 6 shows it is also
  *protective* — the routed policy de-funds addresses that never appear in success. Enumeration has
  no such channel: its action set is its exposure.
- **(d) Finding 3 is the arc's first signature at the level of expertise psychology**, and it
  stacks: each consolidation removes cheap access to the spelling one level down, so a many-story
  agent cannot articulate its own lower levels — "introspection is level-bounded" (§18) arriving
  from below, as a consequence of chunking rather than a constraint on graders.
- **(e) Where multi-level would saturate, held loosely**: the mechanism showed no wall in one turn
  of the crank (finding 7), and the candidate walls are the diet (recurrence thins ~s× per level —
  `reread/lm`'s corpus wall), the grader (mis-levelled and noisier with depth — the arc's
  most-repeated blocker), and the corridor's doubling span (probably an artifact of the token-space
  form; the latent-transition form is named in the SPEC and not run). Iwane's fixed ~7-slot count
  with growing chunk size suggests saturation-by-narrowing, not stopping. The rate version
  (era k+1 certifies *faster* given native era k) remains unclaimed, as in ratchet.

## Caveats

- **One rule draw**, one damage ladder, k explored only at {1, 2, 4, n_moves} and
  span_lam/span_lr/τ unswept. k = 8 was left off the base ladder for a measured reason (the root
  encode tips `never_base`'s width at G = 58, which would confound the matched control).
- **Level-3 corridors mostly did not reach parity in-run**, so "fully native" is an L2 statement
  on the executor side; the table-ablation tolerance (finding 5) therefore leans on heads that
  were near-but-below τ at L3, served under the battery's forced-open policy with parity printed
  beside them.
- **`given_native`'s twin gate is vacuous by construction** (span slots exist from c1);
  `given_fid` — both ports wired, both shut, bit-identical to `given` for 90 cycles — is the
  in-tag assertion carried in its place, and doubles as the battery's negative control.
- Two unresolved cells for a future round, not sanded down: `span_mined`'s era-3 +0.037 with its
  A_true movement (finding 4), and `practice_late_native`'s era-3 +0.058 over routing-alone
  (finding 8).
- The proposal head trains **online**; the offline-first ceiling reading was not run. Exploration
  ε and `prop_steps` were set by one smoke's measured lock-in, not swept.

## Runs on disk

| tag | child | what it is |
|---|---|---|
| `np_s0` | `prop/` | Port 1 main: 12 arms (enum ×3 vocabs, k-ladder ×2 vocabs, base_prop ×3, kN fidelity), 1526 s |
| `sp_s0` | `span/` | Port 2 main: 5 arms (enum ×3, span_true, span_mined), 958 s |
| `nf_s0` | `full/` | Composition: 7 arms + end-of-run ablation battery + poison pair, 1583 s |
| selfchecks | each | P-1…P-7 (proposal), S-1…S-6 (span), C-1…C-5 (composition) — all bit-for-bit |
| smokes | each | attached/detached `--quick`; `prop` smoke0 measured the lock-in that set ε; `full` smoke0 caught a shadowing bug that widened C-2 into C-5 |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# gates
modal run rhm/practice/native/prop/prop.py::prop_selfcheck_remote
modal run rhm/practice/native/span/span.py::span_selfcheck_remote
modal run rhm/practice/native/full/full.py::full_selfcheck_remote

# Port 1 (12 arms)
python3 rhm/practice/native/prop/launch_detached.py --fn prop_run --tag np_s0 --seed 0 \
    --arms "given,given_prop_kN,given_prop_k1,given_prop_k2,given_prop_k4,never_base,never_base_prop_k1,never_base_prop_k2,never_base_prop_k4,practice_late,practice_late_prop_k2,practice_late_prop_k4" \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 --probe-every 4
# Port 2 (5 arms)
python3 rhm/practice/native/span/launch_detached.py --fn span_run --tag sp_s0 --seed 0 \
    --arms "never_base,given,span_true,practice_late,span_mined" \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 --probe-every 4 \
    --span-lam 1.0 --span-lr 1e-3 --span-tau 0.95 --span-min-hold 256
# Composition + battery (7 arms)
python3 rhm/practice/native/full/launch_detached.py --fn full_run --tag nf_s0 --seed 0 \
    --arms "given,given_fid,given_native,practice_late,practice_late_native,practice_early,practice_early_prop_k4" \
    --eras "1:6,2:3,3:1" --era-cycles 30 --budget 4 --g-budget 58 --probe-every 4 \
    --span-lam 1.0 --span-lr 1e-3 --span-tau 0.95 --span-min-hold 256

# reductions
python3 rhm/practice/native/prop/analyze_prop.py --tag np_s0 --fetch --figures
python3 rhm/practice/native/span/analyze_span.py --tag sp_s0 --fetch --figures
python3 rhm/practice/native/full/analyze_full.py --tag nf_s0 --fetch --figures
```

All three launchers pass `rr_s0`'s remaining flags by default. Volumes (`rhm-scaling-data`):
`/data/rhm_practice_native_{prop,span,full}/<tag>/<arm>/results.json` + `setup.json`; fetched
copies, `summary.json` and figures under each child's `figures/<tag>/`.

## Next steps (queued, not started)

The live re-earning spiral — earn level k, consolidate, earn level k+1 natively over the routed
policy — in the depth-6 world (`tall/`'s recipe), which is the direct test of whether the crank
accelerates, holds, or decays per turn and of which wall bites first · the latent-transition form
of Port 2 (Stage 3b/5) · the LM sibling (`native/lm/`: tokenizer expansion as the native-primitive
op) · a seed pair on the k = 4 effect · the two unresolved cells above · idea-doc revisions via
`/update-beliefs` (§3½ gaining `handle/` as its measured negative and this round as its positive
case; §18's unit-creating-op clause gaining a consolidation clause and the poison-inversion as a
new fact about what selection buys).

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
