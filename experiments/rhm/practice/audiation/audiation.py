"""audiation (E1, phase 1) — THE INSTRUMENT. Per-decision logging of the practice beam, plus
per-cycle snapshots of the learner's two trainable heads, so the arity-2 self-model can be
built OFFLINE on what the learner actually did.

"Audiation" is the musical term for hearing a sound internally before producing it: a forecast
formed with the efference copy in hand. ROADMAP.md §4.2 shape E1 asks whether the arity-2
self-model's residual — a forecast of the learner's own next planner/value state given its OWN
move — carries an agency-gated, per-datum revision signal. In teacher-forced NTP the arity gap
`g = ||FM2(s,u) - FM1(s)||` is identically zero because the world fills the second slot; in
practice the slot is the agent's own chosen move and `g > 0`. THIS FILE DOES NOT BUILD THE
SELF-MODEL. It produces the dataset a second, purely analytical phase ports the `rev_pair` /
`SlotFM` recipe onto (`../../confabulation/temporal/epistemics/`).

FORK NOTICE. This file forks `../conductor/conductor.py` VERBATIM and adds ONE class of thing,
marked `# [audiation]` at every insertion point: INSTRUMENTATION THAT CONSUMES NO RNG AND
CHANGES NO BEHAVIOUR. `conductor/`, `assay/`, `census/`, `spiral/` and everything upstream are
NOT modified. With `--log-decisions 0` this file IS `conductor.py`; with logging ON, the
`anchor` arm must still replay `cd_s0/anchor` (and so `as_s0/anchor`) at 0.000e+00 over all
116 cycles, which is the round's free correctness gate (`cd_ef`'s precedent).

  1. THE PER-DECISION RECORDER (`DecRec`). `beam_moves` and `beam_moves_prop` gain one
     optional argument, `rec`; when it is `None` both functions are the donor's expression for
     expression. When it is a recorder — and it is passed at exactly ONE call site, the
     practice beam in `run_arm` block (a) — the beam emits, per beam step:
       * a TIPS row per surviving beam entry: the config `x` (int8), the frozen encoder state
         `z = controller.state(x)` (fp16, D=96), the root target, the parent tip index at the
         previous step, and the move that produced it;
       * pi's logits at that tip, SCATTERED INTO SLOT SPACE (n_slots, fp16) so a decision
         before a commit and one after are directly comparable;
       * a CANDIDATES row per scored child: which parent, which move, the value score, whether
         the topk kept it and as which new tip, and a provenance byte separating a move the
         head PROPOSED from one that was FORCED (too new for the head) or EXPLORE-INJECTED
         (`prop_explore=1`, the agency the arity-2 slot is about);
       * the terminal re-score, joined to the per-tip grade (`succ`, `dres`) the arm computes
         anyway.
     The recorder is a pure sink: it draws no RNG, touches no `counts`, and returns nothing
     into the beam. `beam_gate()` asserts the instrumented beams reproduce the DONOR MODULE's
     `beam_moves`/`beam_moves_prop` bit-for-bit with the recorder both off and on
     (`entry_recorder_check`'s idiom, one object up).

  2. PER-CYCLE HEAD SNAPSHOTS. The controller is FROZEN in this substrate (`build_shared`
     builds it once; `run_arm` only deep-copies the plant and the value). What learns is the
     value head, the proposal head pi, and the plant. So "the learner's next planner/value
     state" at a logged state `s` is the value/pi READOUT at `s` under the post-update heads,
     and the revision at `s` from cycle c's grades is `f(heads_{c+1}, s) - f(heads_c, s)`,
     recomputable offline from `z` alone. Both tiny heads are snapshotted as fp32 numpy at the
     TOP of every cycle, before anything in that cycle runs; the plant every
     `plant_snap_every` cycles (materialisation counterfactuals only — the readouts do not
     need it). `schema.md`, written into the outdir, states the cycle-phase semantics exactly.

  3. THE FIXED SNAPSHOT PROBE + THE IN-RUN RECOMPUTE ASSERT. 64 states of the era's own
     metering set, encoded once per era, carried through every cycle: the LIVE `v` and `pi` on
     them are logged fp32 per cycle (a free, fixed-state readout trajectory), and every cycle
     the same two readouts are recomputed from the snapshot's serialised numpy and asserted
     EQUAL BIT FOR BIT. On `snap_check_every` cycles the round trip goes through the written
     file rather than the in-memory dict, so the on-disk format is what is checked.

--- the donor's own header follows, unmodified ---

conductor — A1, the composition: teacher_slot's gauge-reading outer loop DRIVING the
consolidated practice learner's crank, instead of the schedule and the certificate.

FORK NOTICE. This file forks `../assay/assay.py` VERBATIM and adds five things, each marked
`# [conductor]` at its insertion point. `assay/`, `census/`, `spiral/` and everything upstream
are NOT modified. With the loop off (`commit="delta_prov"`, the `anchor` arm) this file IS
`assay.py`: the era loop's rewrite from a `for` over a fixed cycle count to a `while` with a
cap is exactly equivalent when nothing ever advances early, the reads consume no RNG and no
gradient, and the in-tag `anchor` doubles as the cross-tag replay gate against `as_s0/anchor`.

  1. THE OUTER LOOP OWNS THE CRANK'S ACTIONS. `../conductor/policy.py` (out of the Modal app,
     so every rule is auditable with no GPU) supplies a thermostat-grade rule over ONE scalar
     read: hold while the read still moves, act when it quiets inside a dead zone that was
     MEASURED (`floors.py`, the null-ABBA method) and never defaulted. Its two actions are the
     two the crank has this round — COMMIT the active level, and ADVANCE the era — and it takes
     them in place of the certificate and the ladder's fixed cycle counts. The era SEQUENCE (the
     damage ladder and its cells) is the world and does not move; only the timing is the loop's.
     Read `policy.py`'s docstring for why the donor's ABBA paired trial does not port to an
     absorbing action, what was kept, and what the estimator is and is not.

  2. THE OBSERVATION PANEL. `census`'s G-Y miner is an unpriced, read-only level-4 miner that
     observes in EVERY era. This file generalises it to every level 3..depth, for one measured
     reason: `gauge_hist[3]` is identically ZERO through the whole of era 1 (verified on
     `as_s0/anchor`: 0 at every cycle c1..c48), because the committable miners are deliberately
     gated to `era_level + 1` so era 1 cannot hand era 2 a finished vocabulary. The gauge an
     era-1 loop needs — level-3 minability while level 2 is being earned — therefore does not
     exist in the donor at all and has to be instrumented. The panel NEVER commits, never gates,
     never reaches `committed`, and never touches the era-gated `miners[...]` the tables are
     built from; its level-4 entry is asserted equal to the donor's G-Y miner in-run.

  3. THE ENDO READ — endo_yield's label-free one-level-up loss, ported. RHM's tree is balanced,
     so which span closes one level above the era's damage cell is a fact about the grammar's
     SHAPE (depth, s, and the era's own cell, which the agent lives in) and not about its
     content. `endo_read` masks that parent span entirely and scores the plant's own masked-
     infill NLL on it — the learner's own training objective, on a fixed clean batch, with no
     table, no oracle and no truth mask anywhere in the computation. Masking the WHOLE parent
     span is what makes it one level up: recovering it requires the level-(l+1) latent inferred
     from outside the span, where masking only the cell requires the level-l latent. Gate N-1
     asserts the label-freeness by construction; `endo_bench` prices it.

  4. THE SHADOW PANEL AND THE LEDGER (`endo_yield`'s pattern, `ear`'s convention). Every
     candidate gauge is logged in every arm at every decision point, uncharged, so a
     counterfactual decision trace exists for every gauge in every arm. The ledger charges only
     what the POLICY's input stream contains: the arm's own error and its own mined counts are
     free (the learner's own experience), and the endo read — a forward pass the arm does not
     otherwise make — is priced at its measured cost. Pricing enters `counts["ground"]` and so
     `t_cum`, which nothing in the cycle loop reads: the ledger is a readout, never a governor.

  5. THE YOKED ARMS (`census`'s mechanic, widened to both actions). `conductor_run` runs each
     gauge arm first and passes its REALISED action cycles — commits and era advances, including
     the advances forced by a cap — into its yoke, which replays them by clock and by nothing
     else. `census` finding 4 (`census_gate` == `yoked_delay` bit-identical over 116 cycles) is
     why these are mandatory rather than optional.

--- the donor's own header follows, unmodified ---

assay — which ingredient of the complete vocabulary is the currency?

FORK NOTICE. This file forks `../census/census.py` VERBATIM and adds four things, each marked
`# [assay]` at its insertion point. `census/`, `spiral/` and everything upstream are NOT
modified.

  1. ORACLE SURGERY AT COMMIT. At each level's own commit cycle an arm commits a CONSTRUCTED
     table instead of its mined one, crossing amount x truth x arrival:
       strip      own mined, junk removed          (truth up, amount down)
       complete   own mined U all missing true     (amount up, truth up)
       exact      the full true table              (complete content, commit-time arrival)
       junk_dose  own mined U K realistic junk     (amount up, truth down); K matched to
                  `complete`'s addition count, which at L2 is exact because the two arms are
                  still twins there, and at L3 is per-arm because the L2 surgery has already
                  separated them — both realized counts are logged.
     Everything else is the anchor's: policy, pricing, schedule, streams. The surgery runs
     BEFORE the commit body, so the priced pre-commit audition grades what is actually
     committed and an emptied table cancels the commit through the donor's existing guard.
     The mined table's own grade is recorded beside the surgical one, and the shadow
     certificate still records what the certificate WOULD have done, so the counterfactual
     survives the treatment rather than being overwritten by it.

  2. PER-EXECUTION ENTRY IDENTITY — the instrument the census forensics found structurally
     uncomputable. `macros.macro_features` ends at `best = cur.argmax(-1)`: the table entry the
     max-sum DP selects for each row of each macro execution. That value is discarded by the
     donor. `_install_entry_recorder` swaps in a byte-identical copy of the donor's function
     that additionally accumulates a per-entry histogram on-device, so the choice is logged
     without a host sync, without touching `counts` (unpriced), and without consuming any RNG.
     `entry_recorder_check()` asserts the copy reproduces the donor's `(feats, pos)` bit for
     bit, and `preflight` asserts a treated arm stays bit-identical to its twin up to the first
     surgery WITH the instrument on.

  3. THE REALISTIC JUNK POOL. `junk_dose`'s junk is not uniform random tuples: it is drawn from
     tuples actually mined-but-false in `census/cs_s0` and `spiral/sp_s0`, read off those runs'
     `log["miner"][...]["keys_at_support"]` on the volume at setup and keyed against
     `MC.true_tables`. The pool and the draw are logged verbatim in `setup.json`.

  4. The surgery tables themselves are logged verbatim (entries, flat tuples, truth split) on
     every surgery event, so the treatment is auditable after the fact rather than trusted.

--- the donor's own header follows, unmodified ---

census — can an endogenous gauge buy the coverage the certificate cannot see?

FORK NOTICE. This file forks `../spiral/spiral.py` VERBATIM and adds four things, each marked
`# [census]` at its insertion point. `spiral/` and everything upstream of it are NOT modified.

  1. THE G-Y INSTRUMENT — an unpriced, read-only level-4 miner in EVERY arm, decoupled from
     `max_macro_level` (which caps what may be COMMITTED, not what may be OBSERVED). Phase 0
     established this stream is not in `sp_s0`'s logs at all and cannot be reconstructed from
     them: no level-4 miner is ever built, and per-key counts carry no adjacency. Level 4 stays
     unearnable (1,024 entries, ~384 cycles to cover) so this is a growth-DIRECTION readout and
     never a commit candidate. `gy_soundness()` checks it against hand-enumerated counts.

  2. COMMIT-THEN-EXTEND (`census_extend`) — the recert loop widened to every COMMITTED level in
     every era, and given the power to ADD post-commit mined entries to a frozen table instead
     of only offering whole-table swaps (which never fired in `sp_s0`: 0 swaps in 26 recerts).
     Entries enter by SELECTION, not append: each candidate is auditioned on fresh held-out
     instances of the era's own cell, priced, and admitted only if it does not raise the
     table's audition error. Phase 0's reason for the widening: the inherited loop fires only
     for `rc_level = era["level"]`, so it runs in eras 2-3 and **not at all in eras 4-5** —
     which are exactly where this round's payoff is measured.

  3. THE L2 GATE (`census_gate`) — a CONJUNCTION rule at level 2: commit when the certificate
     has fired AND the admission-rate gauge is quiet (G-A on the at-support series, W=12,
     theta=0, Phase 0 R5). A conjunction, not a replacement, so the op is strictly "hold longer
     than the certificate", `yoked_delay` is exactly interpretable as delay-alone, and no
     setting can accidentally ACCELERATE the commit. The gate is at L2 and not L3 because
     Phase 0 measured that a frozen L2 of recall r caps buildable L3 coverage at about r**2
     (0.327 in the routed arms), while L2 coverage on offer is +0.357 recall.

  4. `yoked_delay` — commits at the cycle `census_gate` actually committed, by clock rather
     than by signal. The cycle is not hard-coded: `census_run` runs `census_gate` first and
     passes its measured L2 commit cycle in, recording which case occurred (gate fired, or
     gate fell back to the era boundary).

Everything else is `spiral.py`'s, unchanged — which is what `fidelity_smoke` and the in-tag
`spiral_route` anchor assert.

--- the donor's own header follows, unmodified ---

spiral — the LIVE RE-EARNING SPIRAL on the depth-6 substrate.

FORK NOTICE. This file forks `../native/full/full.py` VERBATIM and then adds three things,
each marked `# [spiral]` at its insertion point. `native/`, `tall/`, `ear/`, `recital/` and
`ratchet/` are NOT modified; nothing is copied out of them except where a comment says so.

  1. THE DEPTH-6 TRANSPLANT (`tall/tall.py`'s lessons, not its code). `_spiral_shared` is
     `tall`'s `_tall_shared` idiom applied to THIS file's own `build_shared`: it populates
     `probe_clean` — the key every round's ENTRYPOINT populates and no builder does, which
     cost `tall` a ~510 s setup to discover — and then ASSERTS the complete key set that
     `run_arm` / `measure_refs` / `plant_probe` read, so interface drift fails in seconds
     instead of after a paid setup. `preflight` calls every one of those at toy sizes for the
     same reason. The grammar itself is a cfg substitution (`depth=6, m=2`), so with the
     transplant OFF this file is depth-4 `full.py` — which is what `fidelity_d4` gates.

  2. THE COMMIT POLICY, transplanted from `ear/ear.py` (and carried by `recital/recital.py`):
     CERTIFY-ELSE-PROVISIONAL-AT-BOUNDARY (`commit="delta_prov"`) plus the LIVE RECERT.
     `full.py`'s run loop is ratchet-lineage with a fixed in-tag commit schedule
     (`delta` / `early` / `late`), so the boundary-provisional policy is genuinely absent here
     and had to be written in, not switched on. A provisional commit deliberately skips the
     priced pre-commit audition — you pay to grade the unit where it is CONSUMED instead — and
     the recert grades the frozen unit on the next era's cell against what the live table
     would score. Both branches are additive: an arm whose `commit` is `delta`/`early`/`late`
     and whose spec has no `recert` key takes exactly `full.py`'s path, byte for byte.

  3. THE SPIRAL ARMS and the Phase-A instruments: the descent gate WITH ROUTING LIVE
     (`n_corrupt=1`, `tall/dens0`'s measured lever, folded into `build_shared` rather than
     bolted on afterwards), the commit-cycle PRICING READOUT (does `dens0`'s +0.180
     commit penalty survive when width refits under effective branching k instead of under
     |action set|?), and `tall`'s per-cycle entry-identity miner instrument.

WHY THE ROUND EXISTS: earn level k live from the beam's own chosen trajectories, consolidate
it at commit (pi's action space grows; span slots open behind the per-macro parity gate), then
earn level k+1 natively over the routed policy — and ask whether the crank accelerates, holds,
or decays per turn. See `SPEC.md` in this folder.

Run from experiments/:
  modal run rhm/practice/spiral/spiral.py::gates_remote          # T-1..T-4 + P/C at depth 6
  modal run rhm/practice/spiral/spiral.py::preflight             # interface, ~2 min
  modal run rhm/practice/spiral/spiral.py::fidelity_d4           # G-F: the fork did not drift
  python3 rhm/practice/spiral/launch_detached.py --fn phase_a --tag pa0 ...

--- the donor's own header follows, unmodified ---

full — PHASE 2 of `native`: the two ports COMPOSED, plus the ablation battery.

FORK NOTICE, and the choice behind it. This file forks `../prop/prop.py` (Port 1), which is
itself a verbatim fork of `../../ratchet/ratchet.py`, and grafts Port 2 onto it by IMPORTING
`../span/span_net.py` — nothing is copied from `span/`, and `ratchet/`, `teacher_slot/`,
`prop/` and `span/` are all left untouched. The alternative was to re-fork `ratchet.py` and
import both nets; forking `prop.py` was chosen because Port 1's insertions are the larger and
more delicate of the two (a rewritten beam with a cached-encoder-state contract and a fidelity
gate that has already been verified bit-for-bit in-tag), while Port 2's are a clean seam — one
executor object threaded through the beam's materialisation call. Re-deriving Port 1 by hand
would have put the already-gated half at risk to save nothing.

WHAT IS NEW relative to `prop.py`:

 * The executor seam. `beam_moves` and `beam_moves_prop` both materialise through an executor
   object (`span_net.PlainExecutor` / `SpanExecutor`) instead of calling `macros.apply_any`
   directly. A `PlainExecutor` call IS `apply_any` (span/'s gate S-3), so an untreated arm is
   op-for-op ratchet and Port 1's beam is op-for-op `prop.py`'s.
 * `finetune_generator_span` — the plant loss plus the span head's self-imitation term in the
   same optimizer steps, with a `lam == 0` short-circuit that restores the original graph
   exactly (this is what makes the composed fidelity arm a real gate).
 * The NATIVE arms: proposal head (k = 4, Port 1's measured sweet spot) AND span head on the
   same vocabulary — `given_native`, `practice_late_native`.
 * The POISON PAIR: `practice_early` (a frozen 1-entry bogus level-2 table, exogenously
   enumerated) against `practice_early_prop_k4` (the same bad table, natively ROUTED).
 * `ablation_battery` — an end-of-run probe battery on the final trained state, per arm:
   (a) full system, (b) table ablated with macro calls forced through the span head,
   (b') table and span reach both removed, (c) primitives ablated. Plus, under each, the
   next-level mining yield split into its STRUCTURAL component (entries `Miner.build` can
   assemble over the surviving lower table) and its INFORMATIVE one (distinct level-1 tuples
   observed at support in the chosen trajectories, which no lower table gates).

The fork's fidelity is not assumed: the untreated arms are re-run IN THIS TAG as the twins,
and `given_fid` (k = n_moves, span loss off, parity gate nailed shut) is carried as an in-tag
assertion that the composed loop reproduces enumeration bit-for-bit for 90 cycles.

WHAT IS NEW (everything else is ratchet's, unchanged):

 * `prop_net.ProposalHead` — pi(move | z, r*) beside the value, on the SAME encoder read
   (`controller.state(x)` plus the root target), over a fixed (level, node) slot vocabulary.
 * `beam_moves_prop` — at plan time only the top-k proposed moves per tip are materialised and
   scored, instead of all `n_moves`. Everything else about the beam is `beam_moves` verbatim.
   The kept children's encoder states are CACHED and reused as the next step's proposal read,
   because the value already paid for them when it scored those children.
 * `prop_train` — self-imitation on the beam's own chosen trajectories: (state, r*) -> the move
   the surviving-and-SOLVED tip actually took, masked cross-entropy over the live action set.
   Selection happens before the regression, which is the whole reason this is not "gradient
   descent averaging over everything the agent did".
 * Arms `*_prop_k{1,2,4,N}` over three vocabularies (base / true / mined), and the readouts the
   spec commits Port 1 to: the can't-decompose signature, the effective-branching ledger, and
   the next-level currency under native vs exogenous routing.

WHAT IS PRICED, SAID OUT LOUD. A grounding in this substrate is "score one state" — an encoder
pass plus a value read (`beam_moves` charges one per materialised child). The proposal head
reads the SAME z. For every tip below the root that z was already computed and charged when the
value scored that state as a child, so the port caches it and the ledger charges nothing; the
head-only forward on a cached vector is counted separately as `n_prop` and priced at `c_prop`
(default 0.0, with a sensitivity column at c_mat = 0.05 in the reduction). The ROOT state is
the one honest exception — it has never been anybody's child — so the port pays one full
grounding per instance per solve for it, and `fit_width_k` sees that cost when it picks the
width. The freed groundings are REINVESTED, not banked: each arm runs the widest beam that fits
the same declared G = 58 under ITS effective branching factor, which is exactly ratchet's
matched-pricing idiom with n_moves -> k. The banked reading (same width, fewer groundings) is
recoverable from the width ladder logged at every probe.

Run from experiments/:
  modal run rhm/practice/native/prop/prop.py::prop_run --quick --tag smoke0
  python3 rhm/practice/native/prop/launch_detached.py --fn prop_run --tag np_s0 ...

--- ratchet's own header follows ---

Round 2 of the practice arc on RHM. Round 1 (`../crystallize/`) found the compile certificate
VACUOUS on a frozen plant: practice trained the judge (the value) while the generator that
executes a committed unit never moved, so the committable content was flat from cycle 1 and
committing at cycle 1 was optimal at 26x less priced time. Its precheck (`calp_s0`) validated
the fix — let the plant learn on the agent's own successful repairs and the committed-unit
ceiling moves at 31-66x the metering noise floor.

WHAT THIS ROUND COMPOSES.

 1. A LEARNING PLANT, carried by every arm. The generator fine-tunes online on configurations
    the agent actually solved (replay 0.5 against the clean setup pool), and the value adapts
    online against it — round 1's precheck confound (its wide beam degraded because the value
    was trained against the old generator) fixed by construction.

 2. A DEPTH-LADDERED DAMAGE SCHEDULE, the `level_ladder` idiom: the damage cell moves one level
    deeper per era, through NESTED cells (L1 n6 subset L2 n3 subset L3 n1), so era k+1's error
    is literally era k's error one level up. At a declared per-solve grounding budget, a level-3
    error is four blocks wide and a base-move agent must spend its whole move budget on it with
    no coordination left over — climbing is necessary, not merely available.

 3. AN EARNED VOCABULARY (`macros.py`). The committed unit is not a whole-solve program (round
    1's unit) but a MACRO ACTION: the same max-sum operator the true level move uses, over a
    table MINED from the agent's own solved configurations and parsed by the agent's own
    generator. Committing it puts a new primitive in the practice action space — the beam
    proposes it, the value scores it, the plant's fine-tuning sees trajectories that use it.
    The level-l table is defined over level-(l-1) ENTRIES, so an incomplete committed level-2
    vocabulary is a hard cap on level 3: the poisoned-region test with a mechanism.

 4. A UNIT-LP CERTIFICATE. Round 1's central lesson, in Jasper's constructive form: certify on
    the LEARNING PROGRESS OF THE COMMITTABLE CONTENT — the shadow-audition trajectory of the
    candidate macro, positive and then flat — never on task-performance level or task-LP, which
    conflate selector and plant improvement. "Positive and then flat" is enforced: silence alone
    cannot certify unless the audition has first descended by `lp_min_drop`.

ARMS differ ONLY in vocabulary-acquisition policy:
    never_base     base level-1 moves forever; must buy depth with search at the declared budget
    given          the DGP's own level-2/3 tables from cycle 1 (the `level_moves` ceiling)
    practice_gated earns them, gated by the unit-LP certificate
    practice_early commits one cycle into each era (minimal coverage) -- the poison test
    practice_late  commits at each era's end (the rent-paying control)

Run from experiments/:
  modal run rhm/practice/native/full/full.py::full_selfcheck_remote
  modal run rhm/practice/native/full/full.py::full_run --quick --tag smoke0
  python3 rhm/practice/native/full/launch_detached.py --fn full_run --tag nf_s0 ...

(ratchet's own run lines: `rhm/practice/ratchet/ratchet.py::ratchet`.)
"""

import copy
import inspect          # [conductor] gate N-1 reads the endo read's own source
import json
import os
import time

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_precheck import nearest_derivation_cost
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _train_generator
from rhm.rhm_latent_planner import _build_value_head
from rhm.rhm_sculpt_planner import _corrupt, _encode_chunked, _sample_pool
from rhm.rhm_sculpt_latent import _build_rich_controller
from rhm.practice.crystallize.units import (
    apply_move, build_move_set, corrupt_hier, grade, on_grammar_rate, oracle_rollout)
from rhm.practice.ratchet import macros as MC
from rhm.practice.native.prop import prop_net as PN
from rhm.practice.native.span import span_net as SN
# [conductor] the outer loop lives outside the Modal app, so every rule is auditable and
# gate-able with no GPU and no substrate (the `teacher_slot/decision` convention).
from rhm.practice.conductor import policy as PO


# [audiation] this node's own app and volume prefix; everything else about the substrate is
# the donor's. The DONOR MODULE itself is imported LAZILY, inside `beam_gate` only — it
# defines its own `modal.App`, and two apps in one import graph is a CLI ambiguity nobody
# needs. Importing it there lets the gate assert the instrumented beams against the functions
# they fork rather than against a copy of them.
app = modal.App("rhm-practice-audiation", image=image)
REMOTE = "rhm_practice_audiation"

# [spiral] the depth-6 substrate, exactly as `tall/` measured it admissible. m=2 is NOT a
# choice: m>=3 flattens the depth ladder (gradient 1.06-1.30x against m=2's 3.25x) and walls
# off every macro level above 3. The nested ladder's node indices are `25 // 2**(l-1)`.
SPIRAL_ERAS = "1:25,2:12,3:6,4:3,5:1"
DEPTH6 = dict(depth=6, m=2, v=8, s=2)
# The keys `run_arm` / `measure_refs` / `plant_probe` read out of the shared dict. Asserted by
# `_spiral_shared` so drift fails in seconds rather than after a ~510 s setup (`tall`'s lesson).
SHARED_KEYS = ("base_ms", "bottom_map", "canon", "controller", "generator0", "hold",
               "inverse_maps", "leaves_pool", "length", "n_blocks", "probe_clean", "read_acc",
               "reader", "replay", "roots_pool", "rules", "rules_t", "truth", "value0")

# [conductor] THE MEASURED DEAD ZONES, from `floors.py` — derived offline, before any GPU, by
# the null-ABBA method on fixed-condition series in `as_s0` (4 arms), `cs_s0` (2 arms) and
# `as_s1` (1 arm): 574 pooled windows per series, span 1, W 4, era boundaries and commit cycles
# dropped. `mean(N)` came out at -0.003 / -0.003 / +0.021 against spreads of 0.059 / 0.474 /
# 0.732, which is the null construction checking itself. Copied here as literals rather than
# read from `floors.json` so the value that governed a run is in the run's own `setup.json`,
# and CHECKED against the file by `floor_gate()` so the two cannot drift apart.
#
#   ledger    0.0294 error/cycle    the arm's own metering error on the era's own cell
#   yield L3  0.2368 tuples/cycle   distinct level-3 tuples at support
#   yield L4  0.3661 tuples/cycle   distinct level-4 tuples at support (the G-Y instrument)
#   endo      no offline series exists — it needs forward passes through a live plant. Measured
#             on the smoke tag's SCHEDULE arm (fixed-condition by construction) and passed in on
#             the command line; the reduction re-derives it in-tag on this run's own `anchor`,
#             exactly as `endo_yield` re-derived `xspan`'s on `no_wall`.
MEASURED_FLOORS = {
    # GOVERNING: re-derived at FULL CONFIG on `cd_ef/anchor` — a 116-cycle schedule arm, which
    # is a fixed-condition series by construction and carries the ACTUAL instruments this round
    # drives (the observation panel included). The offline derivation below is the independent
    # pre-GPU cross-check, and on the one series both can measure it agrees to 0.09%:
    # ledger 0.029447 in-tag vs 0.029421 offline.
    #
    # The yield floors MOVED (L3 0.2368 -> 0.4613) and had to: the offline L3 series came from
    # `as_s0`, where level-3 at-support is identically zero through all 48 cycles of era 1
    # because the committable miners are era-gated. The observation panel makes that series live
    # (0 -> -41 over era 1), so it moves faster and its floor is larger. A floor must be measured
    # on the series the rule actually thresholds.
    "ledger": 0.02944712566990095,
    "yield_by_level": {3: 0.46127129019246205, 4: 0.5166900731510206},
    "endo": 0.01091575129919287,   # the EXCESS read; see the panel's `endo_excess` note
    "provenance": ("cd_ef/anchor, full config, 116 cycles, null-ABBA span 1 W 4, era "
                   "boundaries and commit cycles dropped; cross-checked against floors.py's "
                   "offline derivation on as_s0+cs_s0+as_s1 (574 windows/series)"),
}


# --------------------------------------------------------------------------- #
# the depth ladder
# --------------------------------------------------------------------------- #

def parse_eras(spec):
    """"1:6,2:3,3:1" -> [{level, node, name}] — the nested damage cells, one per era.

    [spiral] An OPTIONAL third field gives that era's own cycle count:
    `"1:25:28,2:12:44,3:6:12,4:3:9,5:1:7"`. The donor's loop runs every era for the same
    `cfg["era_cycles"]`, which at depth 6 is wrong in both directions at once — era 2 has to
    carry L3 mining coverage (~24 cycles at `mine_cap=8` by `tall`'s coverage law) while eras
    3–5 are pure consumption and need only a stable cost-to-depth read. Two fields keep the
    donor's behaviour exactly (no `cycles` key -> `cfg["era_cycles"]`), so the depth-4
    fidelity gate is untouched."""
    out = []
    for part in spec.split(","):
        bits = part.split(":")
        lv, node = int(bits[0]), int(bits[1])
        era = {"level": lv, "node": node, "name": f"L{lv}n{node}"}
        if len(bits) > 2 and bits[2]:
            era["cycles"] = int(bits[2])
        out.append(era)
    return out


def era_ctx(era):
    return {"name": era["name"], "level": era["level"], "nodes": [era["node"]]}


def era_l2_nodes(era, s):
    """The level-2 nodes an era's damage cell spans — where a withheld level-2 tuple has to
    be produced for that era's instance to be repairable."""
    span = s ** (era["level"] - 1)
    b0 = era["node"] * span
    return sorted({b // s for b in range(b0, b0 + max(span, 1))} if span >= s
                  else {b0 // s})


def needs_excluded(clean_np, era, hold, inverse_bottom, v, s):
    """Which instances REQUIRE a withheld level-2 tuple: the clean derivation the damage was
    applied to uses one at the damage cell. The coordinator's constraint-1 gate is read on
    this split — a plant frontier is only a frontier if bootstrap success on it is
    low-but-nonzero; at ~0 the design collapses into exploration-gating."""
    if not hold:
        return np.zeros(clean_np.shape[0], bool)
    tup = level2_tuples(clean_np, inverse_bottom, v, s)
    nodes = era_l2_nodes(era, s)
    bad = np.zeros(clean_np.shape[0], bool)
    for t in hold:
        for nd in nodes:
            bad |= (tup[:, nd, :] == np.array(t)).all(-1)
    return bad


def context_instances(rules, ctx, n, s, depth, v, m, seed, require_broken=True,
                      with_clean=False):
    """n fresh instances of a damage cell (crystallize's, verbatim): a clean derivation of a
    random r*, the cell hierarchically damaged, rejection-sampled on d* > 0."""
    length = s ** depth
    roots = np.zeros(n, np.int64)
    x = np.zeros((n, length), np.int64)
    clean = np.zeros((n, length), np.int64)
    rng = np.random.default_rng(seed + 90_001)
    need = np.arange(n)
    for attempt in range(64):
        if len(need) == 0:
            break
        r, lv = _sample_pool(rules, len(need), s, seed + 7919 * attempt)
        xd = corrupt_hier(lv, rules, depth, v, m, s, ctx["level"], ctx["nodes"], rng)
        ok = (nearest_derivation_cost(rules, xd, r, s) > 0) if require_broken \
            else np.ones(len(need), bool)
        roots[need[ok]] = r[ok]
        x[need[ok]] = xd[ok]
        clean[need[ok]] = lv[ok]
        need = need[~ok]
    if len(need):
        raise RuntimeError(f"context {ctx['name']}: {len(need)} instances never broke")
    return (roots, x, clean) if with_clean else (roots, x)


# --------------------------------------------------------------------------- #
# pricing: the declared per-solve grounding budget picks the beam width
# --------------------------------------------------------------------------- #

def beam_ground(n_moves, budget, w):
    total, width = 0, 1
    for _ in range(budget):
        total += width * n_moves
        width = min(w, width * n_moves)
    return total + width


def fit_width(n_moves, budget, g_budget):
    """The widest beam that fits the declared budget — round 1's performance idiom, now
    arm-dependent because arms carry different numbers of moves and each materialisation is
    a grounding. This is what MATCHED PRICING means here: a level move costs the same
    groundings whoever holds it."""
    best = 1
    for w in range(1, 129):
        if beam_ground(n_moves, budget, w) <= g_budget:
            best = w
        else:
            break
    return best


# =========================================================================== #
# [audiation] INSTRUMENT: the per-decision recorder
# =========================================================================== #

# Provenance bits on a candidate's `src` byte. A move can carry more than one (it cannot in
# practice — `explore_moves` excludes what is already selected — but the byte does not assume
# it).
SRC_ENUM = 0            # the ENUMERATED beam: no proposal happened, every move was expanded
SRC_PROPOSED = 1        # in pi's top-k
SRC_FORCED = 2          # `forced`: too new for the head to have a training example on
SRC_EXPLORE = 4         # `explore_moves`: uniform over the moves pi did NOT propose


class DecRec:
    """The per-decision sink for one practice beam.

    A PURE SINK. It draws no RNG, mutates no `counts`, returns nothing into the beam, and
    every tensor it touches it touches under the beam's own `no_grad`. Everything it records
    is something the donor already computed and threw away; the two exceptions are stated at
    their call sites (the root encode in the ENUMERATED beam, and the tip-state gather in the
    enumerated beam), both of which are pure functions of tensors already in hand.

    Two tables per cycle, joined on `(inst, step, tip)`:

      TIPS       one row per surviving beam entry at each step 0..budget. `parent` is the tip
                 index at step-1 this entry descends from (-1 at step 0) and `mv` the index in
                 `ms` of the move that produced it (-1 at step 0), which is the lineage the
                 donor's `collect=True` reconstructs — kept in raw (state, parent) form
                 because that is strictly more information than the resolved trajectories.
      CANDIDATES one row per SCORED CHILD at each step 0..budget-1: `(parent, mv, score, src)`
                 plus `child` — the tip index it became at step+1, or -1 if the topk dropped
                 it. The unchosen-but-scored children are the deliberation state; they are the
                 reason all surviving tip-steps are logged rather than the chosen trajectory.

    `pi` is stored in SLOT space (n_slots wide), not in `ms` order: `ms` order stops meaning
    the same thing the moment the action set grows at a commit, and slot space is exactly the
    head's own output vocabulary. Slots absent from the live action set hold NaN.
    """

    def __init__(self, n_slots, dtype_z="float16"):
        self.n_slots = int(n_slots)
        self.dtype_z = dtype_z
        self.on = False
        self._c = None

    # ---- lifecycle ------------------------------------------------------- #
    def open(self, cycle, meta):
        """Start a cycle's record. `meta` is the per-cycle context the tables join to."""
        self._c = {"cycle": int(cycle), "meta": dict(meta),
                   "tip": [], "pi": [], "cand": [], "final": None,
                   "succ": None, "dres": None}
        self.on = True

    def close(self):
        self.on = False
        c, self._c = self._c, None
        return c

    # ---- the beam's emissions -------------------------------------------- #
    def tips(self, step, beams, z, parent, mv):
        """`beams` (B, W, T) the surviving states at `step`; `z` (B, W, D) their encoder
        states; `parent` (B, W) / `mv` (B, W) the lineage, or None at step 0."""
        import torch
        b, w = beams.shape[0], beams.shape[1]
        rec = {
            "step": int(step),
            "x": beams.to(torch.int8).cpu().numpy(),                       # (B, W, T)
            "z": z.reshape(b, w, -1).to(getattr(torch, self.dtype_z)).cpu().numpy(),
            "parent": (np.full((b, w), -1, dtype=np.int16) if parent is None
                       else parent.to(torch.int16).cpu().numpy()),
            "mv": (np.full((b, w), -1, dtype=np.int16) if mv is None
                   else mv.to(torch.int16).cpu().numpy()),
            "pi": None,
        }
        self._c["tip"].append(rec)

    def pi_at(self, step, plog, slot_t):
        """pi's logits at the tips of `step`, scattered from `ms` order into SLOT space.
        `plog` is (B*W, n_moves) batch-major/width-minor — the same order `tips` recorded."""
        import torch
        rec = self._c["tip"][step]
        assert rec["step"] == step
        b, w = rec["x"].shape[0], rec["x"].shape[1]
        full = torch.full((plog.shape[0], self.n_slots), float("nan"),
                          dtype=plog.dtype, device=plog.device)
        full[:, slot_t] = plog
        rec["pi"] = full.reshape(b, w, self.n_slots).to(torch.float16).cpu().numpy()

    def cands(self, step, sel, scores, top, keep, batch, width, kk, src):
        """One row per scored child at `step`. `sel` (B*W, kk) the selected move indices,
        `scores` (B, W*kk) their value scores, `top` (B, keep) the kept candidate indices,
        `src` (B*W, kk) the provenance byte."""
        import torch
        child = torch.full((batch, width * kk), -1, dtype=torch.int16, device=scores.device)
        child.scatter_(1, top, torch.arange(keep, dtype=torch.int16,
                                            device=scores.device)[None, :].expand(batch, -1))
        self._c["cand"].append({
            "step": int(step), "width": int(width), "kk": int(kk),
            "mv": sel.reshape(batch, width * kk).to(torch.int16).cpu().numpy(),
            "src": src.reshape(batch, width * kk).to(torch.int8).cpu().numpy(),
            "score": scores.to(torch.float32).cpu().numpy(),
            "child": child.cpu().numpy(),
        })

    def finals(self, final):
        """The beam's terminal re-score of every tip — the last thing the donor computes and
        the only value read at the tips that exists nowhere else."""
        import torch
        self._c["final"] = final.to(torch.float32).cpu().numpy()

    # ---- what `run_arm` attaches after the beam returns ------------------- #
    def grades(self, succ, dres):
        self._c["succ"] = np.asarray(succ, dtype=np.float32)
        self._c["dres"] = np.asarray(dres, dtype=np.float32)


def cand_src(sel, forced, exp_idx, n_moves, device):
    """[audiation] the provenance byte for every selected move, derived from the same three
    objects `select_moves` consumed. Pure tensor arithmetic, no RNG.

    `explore_moves` draws only from moves already NOT selected and `forced` are held out of
    the top-k, so the three classes partition `sel` — but the byte is a mask, so an overlap
    would show up rather than be silently resolved."""
    import torch
    is_forced = torch.zeros(sel.shape, dtype=torch.bool, device=device)
    if forced:
        fm = torch.zeros(n_moves, dtype=torch.bool, device=device)
        fm[torch.as_tensor(list(forced), dtype=torch.long, device=device)] = True
        is_forced = fm[sel.long()]
    is_expl = torch.zeros(sel.shape, dtype=torch.bool, device=device)
    if exp_idx is not None and exp_idx.shape[1]:
        em = torch.zeros(sel.shape[0], n_moves, dtype=torch.bool, device=device)
        em.scatter_(1, exp_idx.long(), True)
        is_expl = em.gather(1, sel.long())
    is_prop = ~(is_forced | is_expl)
    return (is_prop.to(torch.int8) * SRC_PROPOSED
            + is_forced.to(torch.int8) * SRC_FORCED
            + is_expl.to(torch.int8) * SRC_EXPLORE)


def flatten_cycle(c, roots_np, budget):
    """[audiation] one cycle's recorder state -> two flat tables of numpy arrays.

    Row order inside a step is BATCH-MAJOR, TIP-MINOR — the order `tips_flat =
    tips.reshape(B * W, T)` uses, which is the order `grade` returns `succ`/`dres` in, so the
    terminal join is positional and needs no key lookup."""
    tip, cand = {}, {}
    tx, tz, tpi = [], [], []
    tcyc, tinst, tstep, ttip, tpar, tmv = [], [], [], [], [], []
    n_slots = None
    for rec in c["tip"]:
        b, w = rec["x"].shape[0], rec["x"].shape[1]
        tx.append(rec["x"].reshape(b * w, -1))
        tz.append(rec["z"].reshape(b * w, -1))
        if rec["pi"] is None:
            tpi.append(None)
        else:
            n_slots = rec["pi"].shape[-1]
            tpi.append(rec["pi"].reshape(b * w, -1))
        tcyc.append(np.full(b * w, c["cycle"], dtype=np.int16))
        tinst.append(np.repeat(np.arange(b, dtype=np.int16), w))
        tstep.append(np.full(b * w, rec["step"], dtype=np.int8))
        ttip.append(np.tile(np.arange(w, dtype=np.int16), b))
        tpar.append(rec["parent"].reshape(b * w).astype(np.int16))
        tmv.append(rec["mv"].reshape(b * w).astype(np.int16))
    n_rows = [a.shape[0] for a in tx]
    if n_slots is None:
        n_slots = int(c["meta"].get("n_slots") or 0)
    tpi = [np.full((n, n_slots), np.nan, dtype=np.float16) if p is None else p
           for n, p in zip(n_rows, tpi)]
    tip["t_x"] = np.concatenate(tx)
    tip["t_z"] = np.concatenate(tz)
    tip["t_pi"] = np.concatenate(tpi)
    tip["t_cycle"] = np.concatenate(tcyc)
    tip["t_inst"] = np.concatenate(tinst)
    tip["t_step"] = np.concatenate(tstep)
    tip["t_tip"] = np.concatenate(ttip)
    tip["t_parent"] = np.concatenate(tpar)
    tip["t_mv"] = np.concatenate(tmv)
    tip["t_root"] = np.asarray(roots_np, dtype=np.int8)[tip["t_inst"].astype(np.int64)]

    n = tip["t_x"].shape[0]
    tip["t_vfin"] = np.full(n, np.nan, dtype=np.float32)
    tip["t_succ"] = np.full(n, -1, dtype=np.int8)
    tip["t_dres"] = np.full(n, np.nan, dtype=np.float32)
    term = tip["t_step"] == budget
    if c["final"] is not None:
        tip["t_vfin"][term] = c["final"].reshape(-1)
    if c["succ"] is not None:
        tip["t_succ"][term] = (np.asarray(c["succ"]).reshape(-1) > 0.5).astype(np.int8)
        tip["t_dres"][term] = np.asarray(c["dres"], dtype=np.float32).reshape(-1)

    ccyc, cinst, cstep, cpar, cmv, csrc, csco, cchi = [], [], [], [], [], [], [], []
    for rec in c["cand"]:
        b = rec["mv"].shape[0]
        w, kk = rec["width"], rec["kk"]
        ccyc.append(np.full(b * w * kk, c["cycle"], dtype=np.int16))
        cinst.append(np.repeat(np.arange(b, dtype=np.int16), w * kk))
        cstep.append(np.full(b * w * kk, rec["step"], dtype=np.int8))
        cpar.append(np.tile(np.repeat(np.arange(w, dtype=np.int16), kk), b))
        cmv.append(rec["mv"].reshape(-1).astype(np.int16))
        csrc.append(rec["src"].reshape(-1).astype(np.int8))
        csco.append(rec["score"].reshape(-1).astype(np.float32))
        cchi.append(rec["child"].reshape(-1).astype(np.int16))
    for key, part in (("c_cycle", ccyc), ("c_inst", cinst), ("c_step", cstep),
                      ("c_parent", cpar), ("c_mv", cmv), ("c_src", csrc),
                      ("c_score", csco), ("c_child", cchi)):
        cand[key] = np.concatenate(part) if part else np.zeros(0, dtype=np.int16)
    return tip, cand


class DecWriter:
    """[audiation] shards the per-decision tables onto the volume.

    One `.npz` per `flush_every` cycles rather than one per cycle: 116 files per arm is a lot
    of `volume.commit()`s, and one file per run is a lot to lose to a crash. Flushes ride the
    donor's own `checkpoint_every` so a killed run keeps everything up to its last checkpoint.
    """

    def __init__(self, root, arm, budget, flush_every=5, compress=True):
        self.dir = os.path.join(root, arm, "decisions")
        os.makedirs(self.dir, exist_ok=True)
        self.budget = int(budget)
        self.flush_every = int(flush_every)
        self.compress = bool(compress)
        self.pending, self.manifest = [], []
        self.n_tips = self.n_cand = 0

    def add(self, c, roots_np):
        self.pending.append(flatten_cycle(c, roots_np, self.budget) + (c["cycle"],))
        if len(self.pending) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.pending:
            return None
        first, last = self.pending[0][2], self.pending[-1][2]
        out = {}
        for key in self.pending[0][0]:
            out[key] = np.concatenate([p[0][key] for p in self.pending])
        for key in self.pending[0][1]:
            out[key] = np.concatenate([p[1][key] for p in self.pending])
        name = f"dec_c{first:04d}_c{last:04d}.npz"
        path = os.path.join(self.dir, name)
        (np.savez_compressed if self.compress else np.savez)(path, **out)
        n_t = int(out["t_cycle"].shape[0])
        n_c = int(out["c_cycle"].shape[0])
        self.n_tips += n_t
        self.n_cand += n_c
        self.manifest.append({"file": name, "first_cycle": int(first), "last_cycle": int(last),
                              "n_tips": n_t, "n_cand": n_c,
                              "bytes": int(os.path.getsize(path))})
        self.pending = []
        volume.commit()
        return path


# =========================================================================== #
# [audiation] INSTRUMENT: per-cycle head snapshots + the recompute assert
# =========================================================================== #

def state_np(mod):
    """A module's parameters as fp32 numpy — the on-disk form. No RNG, no grad."""
    import torch
    return {k: t.detach().to(torch.float32).cpu().numpy()
            for k, t in mod.state_dict().items()}


def load_state_np(mod, arrays):
    """Load an `state_np` dict back into a module. Exact: the arrays are fp32 and every
    parameter in this substrate is fp32."""
    import torch
    sd = {k: torch.as_tensor(np.asarray(v)) for k, v in arrays.items()}
    mod.load_state_dict(sd)
    return mod


class HeadSnapper:
    """[audiation] snapshots the two trainable READOUT heads at the top of every cycle, and
    asserts in-run that the snapshot reproduces the live readout exactly.

    WHY THE TOP OF THE CYCLE. The controller is frozen; what learns in a cycle is the plant
    (block c), the value (block c) and pi (block c'), in that order, and all three are updated
    ONLY there. So the weights standing at the top of cycle c are exactly the weights that
    planned cycle c's practice beam, and `heads_{c+1} - heads_c` is exactly the revision cycle
    c's grades produced. `schema.md` states the full ordering.

    THE PROBE. 64 states of the era's own metering set — fixed for the era, encoded once, no
    RNG — carried through every cycle. The live `v` and `pi` on them go into the record as an
    fp32 per-cycle readout trajectory on a FIXED state set (free, and the cheapest possible
    check on the size of a per-cycle revision); the same two readouts recomputed from the
    snapshot's serialised numpy are asserted equal BIT FOR BIT, every cycle. On
    `check_every` cycles the round trip goes through the written `.npz` instead of the
    in-memory dict, so what is checked is the format phase 2 will actually read.
    """

    def __init__(self, root, arm, *, n_probe=64, check_every=20, plant_every=8):
        self.dir = os.path.join(root, arm, "snapshots")
        os.makedirs(self.dir, exist_ok=True)
        self.n_probe = int(n_probe)
        self.check_every = int(check_every)
        self.plant_every = int(plant_every)
        self.probe = {}          # era index -> (z, roots, x)
        self.trace = []          # per-cycle live probe readouts
        self.checks = []
        self.files = []
        self._scratch = {}

    def era_probe(self, era_i, controller, mx, r_np, device):
        """Encode the era's probe states once. Pure forward pass on a frozen encoder."""
        import torch
        if era_i in self.probe:
            return self.probe[era_i]
        n = min(self.n_probe, int(mx.shape[0]))
        x = mx[:n].to(device)
        with torch.no_grad():
            z = _encode_chunked(controller, x)
        r = torch.from_numpy(np.asarray(r_np[:n])).to(device).long()
        self.probe[era_i] = (z, r, x.to(torch.int8).cpu().numpy())
        return self.probe[era_i]

    def snap(self, cyc, era_i, value, prop, generator, controller, mx, r_np, device):
        import copy as _copy
        import torch
        z, r, _ = self.era_probe(era_i, controller, mx, r_np, device)
        vsd, psd = state_np(value), state_np(prop)
        with torch.no_grad():
            v_live = value(z, r).to(torch.float32).cpu().numpy()
            pi_live = prop(z, r).to(torch.float32).cpu().numpy()

        name = f"heads_c{cyc:04d}.npz"
        path = os.path.join(self.dir, name)
        np.savez(path, **{f"value.{k}": q for k, q in vsd.items()},
                 **{f"prop.{k}": q for k, q in psd.items()})
        self.files.append(name)
        if self.plant_every and (cyc == 1 or cyc % self.plant_every == 0):
            gname = f"plant_c{cyc:04d}.npz"
            np.savez(os.path.join(self.dir, gname), **state_np(generator))
            self.files.append(gname)

        # ---- THE ASSERT. Reconstruct from what was serialised and re-read. ------------- #
        if "value" not in self._scratch:
            self._scratch["value"] = _copy.deepcopy(value)
            self._scratch["prop"] = _copy.deepcopy(prop)
        via_file = bool(self.check_every) and (cyc <= 2 or cyc % self.check_every == 0)
        if via_file:
            z_ = np.load(path)
            vsd_r = {k[len("value."):]: z_[k] for k in z_.files if k.startswith("value.")}
            psd_r = {k[len("prop."):]: z_[k] for k in z_.files if k.startswith("prop.")}
        else:
            vsd_r, psd_r = vsd, psd
        vm = load_state_np(self._scratch["value"], vsd_r).to(device)
        pm = load_state_np(self._scratch["prop"], psd_r).to(device)
        # match the LIVE modules' train/eval flag rather than assuming one: every net in this
        # substrate is built with dropout=0.0 so the flag is behaviourally inert, but an
        # assert that has to reason about that is a weaker assert than one that does not.
        vm.train(value.training); pm.train(prop.training)
        with torch.no_grad():
            v_re = vm(z, r).to(torch.float32).cpu().numpy()
            pi_re = pm(z, r).to(torch.float32).cpu().numpy()
        dv = float(np.abs(v_live - v_re).max()) if v_live.size else 0.0
        dp = float(np.abs(pi_live - pi_re).max()) if pi_live.size else 0.0
        assert dv == 0.0 and dp == 0.0, (
            f"[audiation] snapshot recompute is not exact at c{cyc}: "
            f"max|dv|={dv:.3e} max|dpi|={dp:.3e} (via_file={via_file}) — every offline "
            f"readout phase 2 computes would be wrong by this much")
        self.checks.append({"cycle": int(cyc), "via_file": via_file, "n": int(z.shape[0]),
                            "max_abs_dv": dv, "max_abs_dpi": dp})
        self.trace.append({"cycle": int(cyc), "era": int(era_i) + 1,
                           "v": v_live.astype(np.float32),
                           "pi": pi_live.astype(np.float32)})
        return name

    def write_trace(self, root, arm):
        """The fixed-probe readout trajectory, one file for the arm. The probe states change
        at an era boundary (each era meters on its own held-out set), so `v`/`pi` are stacked
        per cycle and the era's probe configs are written beside them: comparing two cycles'
        readouts is only meaningful inside one era, and the file makes that checkable."""
        if not self.trace:
            return None
        out = {"cycle": np.asarray([t["cycle"] for t in self.trace], dtype=np.int16),
               "era": np.asarray([t["era"] for t in self.trace], dtype=np.int8),
               "v": np.stack([t["v"] for t in self.trace]),
               "pi": np.stack([t["pi"] for t in self.trace])}
        for era_i, (_, r, x) in sorted(self.probe.items()):
            out[f"probe_x_e{era_i + 1}"] = x
            out[f"probe_r_e{era_i + 1}"] = np.asarray(
                r.cpu().numpy() if hasattr(r, "cpu") else r, dtype=np.int16)
        path = os.path.join(root, arm, "snapshots", "probe_trace.npz")
        np.savez_compressed(path, **out)
        return path


# --------------------------------------------------------------------------- #
# the beam, over a MIXED action set (base level moves + earned macros)
# --------------------------------------------------------------------------- #

def beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
               *, budget, beam_width, device, collect=False, ex=None, rec=None):
    """ratchet's beam, with ONE change (span/'s): every materialisation goes through an executor
    object. `PlainExecutor.apply` IS `macros.apply_any` (span/'s gate S-3), so this is op-for-op
    ratchet for every untreated arm.

    [audiation] `rec` is the per-decision recorder and defaults to `None`, in which case every
    line below is the donor's. Two things happen only when it is not `None`, both pure
    functions of tensors already in hand and neither charged: the root encode (the enumerated
    beam never needs `z` at the tips, only at the candidates) and the tip-state gather out of
    `z_all`. `scores = value(_encode_chunked(...), tgt)` is split into two lines so `z_all` has
    a name; the expression is unchanged."""
    import torch
    if ex is None:
        ex = SN.PlainExecutor()
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval()
        x0 = x0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        n_moves = len(ms)
        beams = x0[:, None, :]
        seqs = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        counts = {"mat": 0, "ground": 0}
        hist_x = [beams.clone()] if collect else None
        hist_par = [] if collect else None
        if rec is not None:                                              # [audiation]
            rec.tips(0, beams, _encode_chunked(controller, x0), None, None)
        for t_step in range(budget):
            w_in = width                                                 # [audiation]
            flat = beams.reshape(batch * width, length)
            children = [ex.apply(generator, flat, ms[k], rules_t, canon, depth, v, m, s)
                        for k in range(n_moves)]
            cand = torch.stack(children, dim=1).reshape(batch, width * n_moves, length)
            counts["mat"] += batch * width * n_moves
            tgt = roots.repeat_interleave(width * n_moves)
            z_all = _encode_chunked(controller, cand.reshape(-1, length))
            scores = value(z_all, tgt)
            scores = scores.reshape(batch, width * n_moves)
            counts["ground"] += batch * width * n_moves
            keep = min(beam_width, width * n_moves)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, n_moves, rounding_mode="floor")
            mv = top % n_moves
            beams = cand.gather(1, top[:, :, None].expand(-1, -1, length))
            prev = seqs.gather(1, parent[:, :, None].expand(-1, -1, seqs.shape[2])) \
                if seqs.shape[2] else seqs.new_zeros(batch, keep, 0)
            seqs = torch.cat([prev, mv[:, :, None]], dim=2)
            width = keep
            if collect:
                hist_par.append(parent)
                hist_x.append(beams.clone())
            if rec is not None:                                          # [audiation]
                sel_all = torch.arange(n_moves, dtype=torch.long, device=device)[None, :] \
                    .expand(batch * w_in, -1).contiguous()
                rec.cands(t_step, sel_all, scores, top, keep, batch, w_in, n_moves,
                          torch.full_like(sel_all, SRC_ENUM, dtype=torch.int8))
                rec.tips(t_step + 1, beams,
                         z_all.reshape(batch, w_in * n_moves, -1).gather(
                             1, top[:, :, None].expand(-1, -1, z_all.shape[-1])),
                         parent, mv)
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams.reshape(-1, length)), tgt).reshape(batch, width)
        counts["ground"] += batch * width
        if rec is not None:                                              # [audiation]
            rec.finals(final)
        best = final.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        out = {"x": beams[rows, best], "seq": seqs[rows, best], "counts": counts,
               "tips_x": beams, "tips_seq": seqs}
        if collect:
            idx = torch.arange(width, device=device)[None, :].expand(batch, -1).contiguous()
            traj = [None] * (budget + 1)
            for t in range(budget, -1, -1):
                traj[t] = hist_x[t].gather(1, idx[:, :, None].expand(-1, -1, length))
                if t > 0:
                    idx = hist_par[t - 1].gather(1, idx)
            out["traj"] = traj
    return out


# --------------------------------------------------------------------------- #
# THE PORT: the same beam, but only the top-k PROPOSED moves per tip are expanded
# --------------------------------------------------------------------------- #

def beam_moves_prop(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
                    *, budget, beam_width, device, collect=False,
                    prop, slots, k, forced=(), n_slots, explore=0, erng=None, ex=None,
                    rec=None):
    """`beam_moves` with the enumeration replaced by a proposal.

    Structurally line-for-line the original: same children ordering (tip-major, move-minor),
    same `topk` over value scores, same parent/seq bookkeeping, same final re-score. The only
    changes are (i) which children exist at all, and (ii) that the kept children's encoder
    states are carried forward instead of being recomputed for the proposal read.

    `slots` maps position-in-`ms` -> slot id, so the head's output vocabulary is stable across
    an action set that grows at a commit. `forced` is a list of positions in `ms` that are
    expanded unconditionally (a move too new for the head to have a training example on).

    FIDELITY. At k >= len(ms) with `forced` empty, `select_moves` returns `arange(n_moves)` for
    every tip, `expand_selected`'s row gather is the identity, and the assembled child tensor,
    the scores, the topk and the sequences are the enumerated ones bit-for-bit. The only
    difference is the extra root encode, which is charged and touches nothing else.

    [audiation] `rec` defaults to `None`, in which case every line below is the donor's. With
    a recorder attached NOTHING extra is computed: `z_tip`, `plog`, `sel`, `scores`, `top` and
    `final` are all things the donor already has, and `exp_idx` is bound to `None` in the
    branch that does not draw it so the provenance byte can be derived without re-running
    `explore_moves` (which WOULD draw RNG)."""
    import torch
    if ex is None:
        ex = SN.PlainExecutor()
    with torch.no_grad():
        controller.eval(); generator.eval(); value.eval(); prop.eval()
        x0 = x0.to(device); roots = roots.to(device)
        batch, length = x0.shape
        n_moves = len(ms)
        slot_t = torch.as_tensor(slots, dtype=torch.long, device=device)      # (n_moves,)
        beams = x0[:, None, :]
        seqs = torch.zeros(batch, 1, 0, dtype=torch.long, device=device)
        width = 1
        counts = {"mat": 0, "ground": 0, "prop": 0}
        hist_x = [beams.clone()] if collect else None
        hist_par = [] if collect else None

        # the ROOT read: the one encoder pass the port genuinely adds, charged in full.
        z_tip = _encode_chunked(controller, x0)                               # (batch, D)
        counts["ground"] += batch
        if rec is not None:                                              # [audiation]
            rec.tips(0, beams, z_tip, None, None)

        for t_step in range(budget):
            w_in = width                                                 # [audiation]
            flat = beams.reshape(batch * width, length)
            zf = z_tip.reshape(batch * width, -1)
            rf = roots.repeat_interleave(width)
            plog = prop(zf, rf)[:, slot_t]                                    # -> `ms` order
            counts["prop"] += batch * width
            if rec is not None:                                          # [audiation]
                rec.pi_at(t_step, plog, slot_t)
            sel = PN.select_moves(plog, k, forced=list(forced))               # (N, k)
            exp_idx = None                                               # [audiation]
            if explore and sel.shape[1] < n_moves:
                # NB: not named `ex` — that is the executor parameter now.
                exp_idx = PN.explore_moves(sel, n_moves, explore, erng, device)
                sel = torch.sort(torch.cat([sel, exp_idx], dim=1), dim=1).values
            kk = sel.shape[1]
            child, n_mat = PN.expand_selected(generator, flat, sel, ms, ex.apply,
                                              rules_t, canon, depth, v, m, s)
            cand = child.reshape(batch, width * kk, length)
            counts["mat"] += n_mat
            tgt = roots.repeat_interleave(width * kk)
            z_all = _encode_chunked(controller, cand.reshape(-1, length))
            scores = value(z_all, tgt).reshape(batch, width * kk)
            counts["ground"] += batch * width * kk
            keep = min(beam_width, width * kk)
            _, top = scores.topk(keep, dim=1)
            parent = torch.div(top, kk, rounding_mode="floor")
            col = top % kk
            mv = sel.reshape(batch, width, kk).gather(
                1, parent[:, :, None].expand(-1, -1, kk)).gather(2, col[:, :, None]).squeeze(2)
            beams = cand.gather(1, top[:, :, None].expand(-1, -1, length))
            z_tip = z_all.reshape(batch, width * kk, -1).gather(
                1, top[:, :, None].expand(-1, -1, z_all.shape[-1]))
            prev = seqs.gather(1, parent[:, :, None].expand(-1, -1, seqs.shape[2])) \
                if seqs.shape[2] else seqs.new_zeros(batch, keep, 0)
            seqs = torch.cat([prev, mv[:, :, None]], dim=2)
            width = keep
            if collect:
                hist_par.append(parent)
                hist_x.append(beams.clone())
            if rec is not None:                                          # [audiation]
                rec.cands(t_step, sel, scores, top, keep, batch, w_in, kk,
                          cand_src(sel, forced, exp_idx, n_moves, device))
                rec.tips(t_step + 1, beams, z_tip, parent, mv)
        tgt = roots.repeat_interleave(width)
        final = value(_encode_chunked(controller, beams.reshape(-1, length)), tgt).reshape(batch, width)
        counts["ground"] += batch * width
        if rec is not None:                                              # [audiation]
            rec.finals(final)
        best = final.argmax(dim=1)
        rows = torch.arange(batch, device=device)
        out = {"x": beams[rows, best], "seq": seqs[rows, best], "counts": counts,
               "tips_x": beams, "tips_seq": seqs}
        if collect:
            idx = torch.arange(width, device=device)[None, :].expand(batch, -1).contiguous()
            traj = [None] * (budget + 1)
            for t in range(budget, -1, -1):
                traj[t] = hist_x[t].gather(1, idx[:, :, None].expand(-1, -1, length))
                if t > 0:
                    idx = hist_par[t - 1].gather(1, idx)
            out["traj"] = traj
    return out


def plan(controller, generator, value, x0, roots, ms, rules_t, canon, depth, v, m, s,
         *, budget, beam_width, device, collect=False, port=None, ex=None, rec=None):
    """One entry point for every place the agent plans, so Port 1 is either everywhere the beam
    runs or nowhere; `ex` is Port 2's executor, threaded the same way. `port=None` with a plain
    executor is ratchet's `beam_moves`, byte-for-byte.

    [audiation] `rec` is threaded the same way and is passed at exactly ONE call site — the
    practice beam. Every other plan in the run (the metering beam, the recert's grading beam,
    the auditions, the probes, the ablation battery) gets `rec=None` and is the donor's."""
    if port is None:
        return beam_moves(controller, generator, value, x0, roots, ms, rules_t, canon, depth,
                          v, m, s, budget=budget, beam_width=beam_width, device=device,
                          collect=collect, ex=ex, rec=rec)
    return beam_moves_prop(controller, generator, value, x0, roots, ms, rules_t, canon, depth,
                           v, m, s, budget=budget, beam_width=beam_width, device=device,
                           collect=collect, ex=ex, rec=rec, **port)


# --------------------------------------------------------------------------- #
# online learning: the value (selector) and the generator (plant)
# --------------------------------------------------------------------------- #

def value_steps(value, opt, controller, buf, replay, *, n_steps, batch, replay_frac, device, rng):
    import torch
    import torch.nn.functional as F
    if buf["x"].shape[0] == 0:
        return 0.0
    value.train()
    n_rep = int(round(batch * replay_frac)) if replay["x"].shape[0] else 0
    n_new = batch - n_rep
    last = 0.0
    for _ in range(n_steps):
        px, pr, py = [], [], []
        if n_new:
            i = torch.from_numpy(rng.integers(0, buf["x"].shape[0], size=n_new))
            px.append(buf["x"][i]); pr.append(buf["r"][i]); py.append(buf["y"][i])
        if n_rep:
            i = torch.from_numpy(rng.integers(0, replay["x"].shape[0], size=n_rep))
            px.append(replay["x"][i]); pr.append(replay["r"][i]); py.append(replay["y"][i])
        x = torch.cat(px).to(device); r = torch.cat(pr).to(device); y = torch.cat(py).to(device)
        with torch.no_grad():
            z = controller.state(x)
        loss = F.binary_cross_entropy_with_logits(value(z, r), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    value.eval()
    return last


def finetune_generator(generator, opt, new_leaves, replay_leaves, bottom_map, *, v, s, n_blocks,
                       n_steps, batch, replay_frac, device, rng):
    """THE PLANT LEARNS (calp_s0's recipe, verbatim): masked infilling on configurations the
    agent SOLVED — a solved config is a valid r* derivation, so the target is well defined —
    with replay `replay_frac` against the clean setup pool as the anti-drift guard."""
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    last = 0.0
    n_rep = int(round(batch * replay_frac))
    n_new = batch - n_rep
    for _ in range(n_steps):
        parts = []
        if n_new and new_leaves.shape[0]:
            parts.append(new_leaves[torch.from_numpy(
                rng.integers(0, new_leaves.shape[0], size=n_new))])
        if n_rep or not parts:
            k = n_rep if parts else batch
            parts.append(replay_leaves[torch.from_numpy(
                rng.integers(0, replay_leaves.shape[0], size=k))])
        leaves = torch.cat(parts).to(device)
        b = leaves.shape[0]
        feats = bottom_map[(leaves.view(b, n_blocks, s) * powers).sum(-1)]
        keep = (feats >= 0).all(1)
        if keep.sum() < 8:
            continue
        leaves, feats = leaves[keep], feats[keep]
        b = leaves.shape[0]
        n_mask = int(torch.randint(1, n_blocks + 1, ()).item())
        order = torch.rand(b, n_blocks, device=device).argsort(dim=1)
        mb = order[:, :n_mask]
        pos = (mb[:, :, None] * s + torch.arange(s, device=device)).reshape(b, -1)
        obs = leaves.clone().scatter_(1, pos, torch.full_like(pos, -1))
        logits = generator.block_logits(obs)
        mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, mb, True)
        loss = F.cross_entropy(logits[mask], feats[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    return last


def finetune_generator_span(generator, head, ex, slots, opt, new_leaves, replay_leaves,
                            bottom_map, *, v, s, n_blocks, n_steps, batch, replay_frac, device,
                            rng, span_rng, span_batch, lam=1.0):
    """THE PLANT LEARNS, WITH A CORRIDOR HEAD (`../span/span.py`'s, verbatim apart from the
    `lam == 0` short-circuit). `finetune_generator` op-for-op — same batches, same number of
    optimizer steps, same masking draws, same level-1 loss — plus the span head's self-imitation
    cross-entropy at weight `lam`, averaged over minted slots, in the SAME steps.

    `lam == 0` skips the term entirely rather than adding `0 * term`. Adding a zeroed term would
    be numerically a no-op but not GRAPH-identical, and the composed fidelity arm (`given_fid`)
    is an assertion about graph identity, not about tolerances."""
    import torch
    import torch.nn.functional as F
    powers = v ** torch.arange(s, device=device)
    generator.train()
    if head is not None:
        head.train()
    last, slast, sacc = 0.0, None, {}
    n_rep = int(round(batch * replay_frac))
    n_new = batch - n_rep
    for _ in range(n_steps):
        parts = []
        if n_new and new_leaves.shape[0]:
            parts.append(new_leaves[torch.from_numpy(
                rng.integers(0, new_leaves.shape[0], size=n_new))])
        if n_rep or not parts:
            k = n_rep if parts else batch
            parts.append(replay_leaves[torch.from_numpy(
                rng.integers(0, replay_leaves.shape[0], size=k))])
        leaves = torch.cat(parts).to(device)
        b = leaves.shape[0]
        feats = bottom_map[(leaves.view(b, n_blocks, s) * powers).sum(-1)]
        keep = (feats >= 0).all(1)
        if keep.sum() < 8:
            continue
        leaves, feats = leaves[keep], feats[keep]
        b = leaves.shape[0]
        n_mask = int(torch.randint(1, n_blocks + 1, ()).item())
        order = torch.rand(b, n_blocks, device=device).argsort(dim=1)
        mb = order[:, :n_mask]
        pos = (mb[:, :, None] * s + torch.arange(s, device=device)).reshape(b, -1)
        obs = leaves.clone().scatter_(1, pos, torch.full_like(pos, -1))
        logits = generator.block_logits(obs)
        mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=device)
        mask.scatter_(1, mb, True)
        loss = F.cross_entropy(logits[mask], feats[mask])
        if head is not None and slots and lam:
            sterm, sacc = SN.span_train_terms(generator, head, ex, slots, s, span_batch,
                                              span_rng, device)
            if sterm is not None:
                loss = loss + lam * sterm
                slast = float(sterm.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        if head is not None and lam:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    generator.eval()
    if head is not None:
        head.eval()
    return last, slast, sacc


def push(buf, x, r, y, cap):
    import torch
    buf["x"] = torch.cat([buf["x"], x])[-cap:]
    buf["r"] = torch.cat([buf["r"], r])[-cap:]
    buf["y"] = torch.cat([buf["y"], y])[-cap:]


# --------------------------------------------------------------------------- #
# THE PORT'S LEARNING: self-imitation of the beam's own chosen trajectories
# --------------------------------------------------------------------------- #

def prop_pairs(out, roots, succ, slots, budget, train_on="solved"):
    """The beam's own chosen trajectories, as (state, root-index, slot) supervision.

    `beam_moves(collect=True)` already hands these back: `traj[t]` is the per-step state of
    every SURVIVING trajectory (ancestry resolved back from the tips) and `tips_seq[:, :, t]`
    is the move that trajectory took at step t. `succ` is those tips' terminal possible-set
    success, so `train_on="solved"` regresses only onto trajectories that WORKED — the
    selection step that makes this self-imitation rather than averaging over everything the
    agent did.

    Move indices are converted to SLOTS here, at collection time, because a position in `ms`
    stops meaning the same thing the moment the action set grows."""
    import torch
    traj, seq = out["traj"], out["tips_seq"]
    B, W = seq.shape[0], seq.shape[1]
    length = traj[0].shape[2]
    slot_t = torch.as_tensor(slots, dtype=torch.long, device=seq.device)
    keep = torch.as_tensor(succ > 0.5, device=seq.device).reshape(B, W) \
        if train_on == "solved" else torch.ones(B, W, dtype=torch.bool, device=seq.device)
    if not bool(keep.any()):
        return None
    rr = roots.to(seq.device)[:, None].expand(B, W)
    xs, as_, rs = [], [], []
    for t in range(budget):
        xs.append(traj[t][keep])                       # (n_keep, T)
        as_.append(slot_t[seq[:, :, t][keep]])         # (n_keep,)
        rs.append(rr[keep])
    return {"x": torch.cat(xs).cpu(), "a": torch.cat(as_).cpu(),
            "r": torch.cat(rs).cpu(), "n_traj": int(keep.sum())}


def prop_train(prop, opt, controller, pbuf, avail_mask, *, n_steps, batch, device, rng):
    """Masked cross-entropy of pi(. | z, r*) against the move the selected trajectory took.

    The mask is the CURRENT action set: slots that do not exist are not competitors and are
    never targets. Buffered pairs from before a commit are replayed under the current mask —
    the mask only ever gains options and the stored target stays legal."""
    import torch
    import torch.nn.functional as F
    n = pbuf["x"].shape[0]
    if n == 0:
        return {"loss": None, "acc": None, "n": 0}
    prop.train()
    last, acc = 0.0, 0.0
    neg = torch.finfo(torch.float32).min / 4
    for _ in range(n_steps):
        i = torch.from_numpy(rng.integers(0, n, size=min(batch, n)))
        x = pbuf["x"][i].to(device)
        r = pbuf["r"][i].to(device)
        a = pbuf["a"][i].to(device)
        with torch.no_grad():
            z = controller.state(x)
        logits = prop(z, r).masked_fill(~avail_mask[None, :], neg)
        loss = F.cross_entropy(logits, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        acc = float((logits.argmax(-1) == a).float().mean())
    prop.eval()
    return {"loss": last, "acc": acc, "n": int(n)}


# --------------------------------------------------------------------------- #
# setup: instruments trained ONCE, forked per arm
# --------------------------------------------------------------------------- #

def behavior_step(controller, generator, x, roots, ms, rules_t, canon, depth, v, m, s,
                  eps, device, rng):
    import torch
    with torch.no_grad():
        batch = x.shape[0]
        props, scores = [], []
        for mv in ms:
            p = apply_move(generator, x, mv, rules_t, canon, depth, v, m, s)
            lp = controller.root_logits(controller.state(p)).log_softmax(-1)
            scores.append(lp.gather(1, roots[:, None]).squeeze(1))
            props.append(p)
        scores = torch.stack(scores, dim=1)
        props = torch.stack(props, dim=1)
        chosen = scores.argmax(dim=1)
        explore = torch.from_numpy(rng.random(batch) < eps).to(device)
        rnd = torch.from_numpy(rng.integers(0, len(ms), size=batch)).to(device)
        chosen = torch.where(explore, rnd, chosen)
        return props.gather(1, chosen[:, None, None].expand(-1, 1, x.shape[1])).squeeze(1)


def collect_value_buffer(controller, generator, rules, rules_t, canon, ms,
                         roots_pool, leaves_pool, cfg, device):
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    n_blocks = leaves_pool.shape[1] // s
    rng = np.random.default_rng(cfg["train_seed"] + 4321)
    xs, rs, ys = [], [], []
    done = 0
    while done < cfg["value_episodes"]:
        b = min(cfg["value_batch_collect"], cfg["value_episodes"] - done)
        done += b
        idx = rng.integers(0, leaves_pool.shape[0], size=b)
        roots_np = roots_pool[idx]
        c = int(rng.integers(1, cfg["n_corrupt"] + 1))
        start = _corrupt(leaves_pool[idx], n_blocks, c, v, s, rng)
        x = torch.from_numpy(start).to(device)
        roots = torch.from_numpy(roots_np).to(device)
        traj = [x.clone()]
        for _ in range(cfg["budget"]):
            x = behavior_step(controller, generator, x, roots, ms, rules_t, canon,
                              depth, v, m, s, cfg["explore_eps"], device, rng)
            traj.append(x.clone())
        succ, _ = grade(x.cpu().numpy(), roots_np, rules, s)
        y = torch.from_numpy(succ.astype(np.float32))
        for st in traj:
            xs.append(st.cpu()); rs.append(roots.cpu()); ys.append(y)
    return {"x": torch.cat(xs), "r": torch.cat(rs), "y": torch.cat(ys)}


def train_reader(reader, leaves, bottom_map, *, v, s, n_blocks, n_steps, batch, lr, device,
                 seed=0):
    """THE READ. `_train_generator` computes its loss only on MASKED blocks, so the block head
    at a VISIBLE position is never supervised — `calp2_s0` measured the consequence: 0.63
    accuracy reading a fully visible block, *worse* than the 0.70 it gets infilling a hidden
    one, and masking elsewhere to fix the distribution shift did not help (0.599). That caps an
    earned level-3 vocabulary's precision at ~0.3, which is what `calr_s0` saw.

    So the agent gets a reader: the same architecture, the same corpus, and the SAME
    supervision channel the generator already trains against (`bottom_map` on clean
    configurations), but trained to read rather than to fill. This is bottom-level perceptual
    competence — the inverse of the `canon` rendering the agent already performs — and it is
    deliberately the only thing handed over: every composition above level 1 still has to be
    earned."""
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(reader.parameters(), lr=lr, weight_decay=1e-4)
    powers = v ** torch.arange(s, device=device)
    g = torch.Generator().manual_seed(seed)
    reader.train()
    last = (0.0, 0.0)
    for step in range(n_steps):
        idx = torch.randint(0, leaves.shape[0], (batch,), generator=g)
        x = leaves[idx].to(device)
        feats = bottom_map[(x.view(batch, n_blocks, s) * powers).sum(-1)]
        ok = feats >= 0
        logits = reader.block_logits(x)
        loss = F.cross_entropy(logits[ok], feats[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, n_steps // 4) == 0:
            acc = float((logits[ok].argmax(-1) == feats[ok]).float().mean())
            last = (float(loss.item()), acc)
            print(f"  reader       step {step + 1:5d}/{n_steps}: loss={last[0]:.4f} "
                  f"read_acc={acc:.3f}", flush=True)
    reader.eval()
    for p in reader.parameters():
        p.requires_grad_(False)
    return last


def holdout_set(rules, depth, s, v, m, n_hold, seed):
    """The plant's frontier: `n_hold` of the grammar's level-2 tuples, withheld from the
    setup corpus. Level 2 is chosen because every era's damage requires producing a correct
    level-2 tuple (era 1 at one block of it, era 2 at the whole node, era 3 at both nodes of a
    level-3 span), so the plant's hole and the vocabulary's hole are the SAME hole — which is
    what keeps the compound descent from decoupling."""
    truth2 = MC.true_tables(rules, depth, s, v, m, 2)[2]
    flats = sorted({tuple(int(x) for x in r) for r in truth2["flat"]})
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(flats))[:n_hold]
    return {flats[i] for i in pick}


def level2_tuples(leaves_np, inverse_bottom, v, s):
    """Exact level-1 features per block -> the (N, n_l2, s) level-2 tuples."""
    feats = MC.exact_features(leaves_np, inverse_bottom, v, s)
    return feats.reshape(feats.shape[0], -1, s)


def filter_pool(leaves_np, roots_np, inverse_bottom, v, s, hold):
    """Reject configurations that use a held-out level-2 tuple anywhere. The setup corpus is
    filtered; the EVALUATION instances are drawn from the unfiltered DGP, exactly the
    train-on-generic / evaluate-on-the-real-thing structure round 1 used to make its value
    arrive stale."""
    if not hold:
        return roots_np, leaves_np, 1.0
    tup = level2_tuples(leaves_np, inverse_bottom, v, s)
    bad = np.zeros(leaves_np.shape[0], bool)
    for t in hold:
        bad |= (tup == np.array(t)).all(-1).any(-1)
    keep = ~bad
    return roots_np[keep], leaves_np[keep], float(keep.mean())


def build_shared(cfg, device):
    """Controller / generator / reader / stale value, trained once. The BASE action space
    (level-1 moves only) is what the setup value's behaviour policy explores, so every arm
    starts from the same stale selector whatever vocabulary it is later given."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    length = s ** depth
    n_blocks = length // s

    rules = generate_rules_distinct(v, s, depth, m, seed=cfg["rule_seed"])
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    base_ms = build_move_set(depth, s, max_level=1)

    hold = holdout_set(rules, depth, s, v, m, cfg["plant_holdout"], cfg["holdout_seed"])
    want = cfg["n_train_episodes"]
    over = 1.0
    roots_np, leaves_np = _sample_pool(rules, want, s, cfg["train_seed"])
    if hold:
        # oversample so the FILTERED corpus is still `n_train_episodes` long -- otherwise the
        # holdout arm would also be a less-data arm and the two would be confounded
        _, _, frac = filter_pool(leaves_np, roots_np, inverse_maps[-1], v, s, hold)
        over = max(1.05, 1.0 / max(frac, 0.05)) * 1.15
        roots_np, leaves_np = _sample_pool(rules, int(want * over), s, cfg["train_seed"])
        roots_np, leaves_np, frac = filter_pool(leaves_np, roots_np, inverse_maps[-1],
                                                v, s, hold)
        roots_np, leaves_np = roots_np[:want], leaves_np[:want]
        print(f"  plant holdout: {len(hold)} level-2 tuples withheld; corpus keep-rate "
              f"{frac:.3f}, corpus size {leaves_np.shape[0]}")
    leaves = torch.from_numpy(leaves_np)
    roots = torch.from_numpy(roots_np)

    controller = _build_rich_controller()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2).to(device)
    _train_edit_controller(controller, leaves, roots, batch_size=cfg["batch_size"],
                           n_blocks=n_blocks, block_size=s, n_steps=cfg["controller_steps"],
                           lr=3e-4, device=device, p_full=0.5)
    generator = _build_generator()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2,
                                   root_conditioned=False).to(device)
    _train_generator(generator, leaves, roots, bottom_map, batch_size=cfg["batch_size"],
                     n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                     n_steps=cfg["generator_steps"], lr=3e-4, device=device)
    reader = _build_generator()(v, length, s, cfg["state_dim"], n_head=4, n_layer=2,
                                root_conditioned=False).to(device)
    read_loss, read_acc = train_reader(reader, leaves, bottom_map, v=v, s=s, n_blocks=n_blocks,
                                       n_steps=cfg["reader_steps"], batch=cfg["batch_size"],
                                       lr=3e-4, device=device, seed=cfg["train_seed"] + 5)
    controller.eval(); generator.eval()
    for p in controller.parameters():
        p.requires_grad_(False)

    print(f"Collecting the generic (stale) value buffer: {cfg['value_episodes']} rollouts")
    vb = collect_value_buffer(controller, generator, rules, rules_t, canon, base_ms,
                              roots_np, leaves_np, cfg, device)
    value = _build_value_head()(cfg["state_dim"], v).to(device)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["train_seed"] + 31)
    empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
    print(f"  buffer {vb['x'].shape[0]} states, terminal success {float(vb['y'].mean()):.3f}")
    value_steps(value, opt, controller, vb, empty, n_steps=cfg["value_steps"],
                batch=512, replay_frac=0.0, device=device, rng=rng)

    truth = MC.true_tables(rules, depth, s, v, m, cfg["max_macro_level"])
    return {"rules": rules, "rules_t": rules_t, "canon": canon, "inverse_maps": inverse_maps,
            "bottom_map": bottom_map, "base_ms": base_ms, "controller": controller,
            "generator0": generator, "reader": reader, "read_acc": read_acc,
            "hold": hold, "value0": value, "replay": vb, "leaves_pool": leaves_np,
            "roots_pool": roots_np, "truth": truth, "n_blocks": n_blocks, "length": length}


# --------------------------------------------------------------------------- #
# [spiral] the depth-6 transplant: one complete shared dict, asserted
# --------------------------------------------------------------------------- #

def _spiral_shared(cfg, device, eras=None):
    """`tall/_tall_shared`'s idiom over THIS file's `build_shared`.

    `probe_clean` is populated by every round's ENTRYPOINT and by no builder, so importing the
    loop without the entrypoint silently omits it and `plant_probe` is the first thing to
    notice — after the setup has already been paid. `tall` lost two ~510 s setups to exactly
    this class of drift. Constructed here once, so every entrypoint in this file gets an
    identical and complete dict, and the key set is asserted rather than hoped for.

    NOTE ON `n_corrupt`, which is the whole densification lever `tall/dens0` measured: it is a
    cfg key read by `collect_value_buffer` INSIDE `build_shared`, so setting `n_corrupt=1` in
    the cfg densifies the stale value buffer at construction. `dens0` had to retrain a value
    head afterwards because it was sweeping variants; there is nothing to sweep any more, so
    the densified buffer is simply the buffer this file builds."""
    import torch
    shared = build_shared(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    shared["stale_random_blocks"] = float(shared["replay"]["y"].mean())

    # [spiral] TASK-MATCHED COLLECTION DAMAGE, adopted for every arm uniformly.
    #
    # `collect_value_buffer` corrupts `n_corrupt` RANDOM blocks anywhere in the sequence, while
    # every graded instance the agent ever meets is corrupted by `corrupt_hier` at one era's
    # cell. At 64 tokens those are very different distributions, and the value head's job is to
    # rank states drawn from the second having been fitted on the first. `tall` named this the
    # one untested densification lever; Phase A measured the buffer's terminal success at
    # 0.1541 task-matched against 0.1269 random-block (1.21x), at 41 s.
    #
    # Matched to the FIRST era's cell, not to a mixture of all five: the mixture would fold in
    # the L4/L5 cells, where nothing succeeds, and LOWER the positive rate — the opposite of
    # densification. Era 1's cell is the distribution the run starts in; the buffer is meant to
    # be a stale prior that the value then adapts away from online, which is the design.
    #
    # Uniform across arms by construction (it is built once, before any arm runs), so in-tag
    # ranks — which are this round's claims — are unaffected by the change of distribution.
    if cfg.get("collect_task_matched") and eras:
        t0 = time.time()
        vb = _collect_task_matched(shared, cfg, eras[0], device,
                                   n_episodes=cfg.get("tm_episodes") or cfg["value_episodes"])
        rate = float(vb["y"].mean())
        print(f"  task-matched buffer at {eras[0]['name']}: {vb['x'].shape[0]} states, "
              f"terminal success {rate:.4f} (random-block "
              f"{shared['stale_random_blocks']:.4f}); retraining the stale value head "
              f"({time.time() - t0:.0f}s to collect)", flush=True)
        value = _build_value_head()(cfg["state_dim"], cfg["v"]).to(device)
        opt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr"], weight_decay=1e-4)
        empty = {"x": vb["x"][:0], "r": vb["r"][:0], "y": vb["y"][:0]}
        value_steps(value, opt, shared["controller"], vb, empty,
                    n_steps=cfg["value_steps"], batch=512, replay_frac=0.0, device=device,
                    rng=np.random.default_rng(cfg["train_seed"] + 606_061))
        shared["value0"] = value
        shared["replay"] = vb
        shared["stale_task_matched"] = rate

    missing = [k for k in SHARED_KEYS if k not in shared]
    assert not missing, f"shared dict is missing {missing} — interface drift"
    return shared


def _install_identity_miner(support):
    """`tall`'s instrument, re-derived here (not imported: `tall` installs it on the shared
    `MC.Miner` class and this file must not depend on `tall`'s import side effects).

    `recital`'s mechanism check could see the mined table's SIZE and recall but never its
    MEMBERSHIP, so "did the table change or only grow?" was unanswerable. `Miner.state()` is
    already called once per cycle per level and logged verbatim, so extending it reaches
    exactly the missing data without touching the arm loop. Idempotent."""
    if getattr(MC.Miner, "_spiral_identity", False):
        return
    orig = MC.Miner.state

    def state(self):
        out = orig(self)
        out["keys_at_support"] = sorted(
            [int(x) for x in k] for k, c in self.counts.items() if c >= support)
        return out

    MC.Miner.state = state
    MC.Miner._spiral_identity = True


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

ARMS = {
    # --- ratchet's five, verbatim: the untreated references / twins ------------------------
    "never_base":     {"vocab": "base",   "commit": None,    "prop_k": None, "span": False},
    "given":          {"vocab": "true",   "commit": None,    "prop_k": None, "span": False},
    "practice_gated": {"vocab": "earned", "commit": "delta", "prop_k": None, "span": False},
    "practice_early": {"vocab": "earned", "commit": "early", "prop_k": None, "span": False},
    "practice_late":  {"vocab": "earned", "commit": "late",  "prop_k": None, "span": False},
    # --- Port 1 alone (phase 1's arms, kept so a composed arm can be read against both
    #     single-port results in the same tag if wanted) ---------------------------------
    "given_prop_k4":         {"vocab": "true",   "commit": None,    "prop_k": 4, "span": False},
    "practice_late_prop_k4": {"vocab": "earned", "commit": "late",  "prop_k": 4, "span": False},
    # --- THE POISON TWIN. `practice_early` freezes a 1-entry bogus level-2 table at c1;
    #     ratchet measured that this forecloses level 3's REPRESENTATION. The routed version
    #     asks the SPEC's question: does a natively-routed bad L2 foreclose worse, because the
    #     policy now calls the address instead of merely having it available? Routing only —
    #     the corridor port is not the variable here.
    "practice_early_prop_k4": {"vocab": "earned", "commit": "early", "prop_k": 4, "span": False},
    # --- Port 2 alone -----------------------------------------------------------------------
    "span_true":  {"vocab": "true",   "commit": None,   "prop_k": None, "span": True},
    "span_mined": {"vocab": "earned", "commit": "late", "prop_k": None, "span": True},
    # --- THE FULL NATIVE PRIMITIVE: proposed as one call (Port 1) and run as one pass
    #     (Port 2). k = 4 is Port 1's measured sweet spot — exactly budget-matched to `given`'s
    #     enumeration at G = 58 (57 groundings per solve either way).
    "given_native":         {"vocab": "true",   "commit": None,   "prop_k": 4, "span": True},
    "practice_late_native": {"vocab": "earned", "commit": "late", "prop_k": 4, "span": True},
    # --- THE COMPOSED FIDELITY ARM. Both ports present and wired, both nailed shut: the
    #     proposal filter at k = n_moves (an exact no-op) and the span loss off with the parity
    #     gate unreachable. Must reproduce `given` bit-for-bit for 90 cycles, which is what
    #     licenses reading the composed file's other arms at all.
    "given_fid": {"vocab": "true", "commit": None, "prop_k": -1, "span": True,
                  "cfg": {"span_lam": 0.0, "span_tau": 2.0}},
    # ---------------------------------------------------------------------------------------
    # [spiral] THE SPIRAL ARMS. Everything above is the donor's and is untouched, so a donor
    # arm run out of this file is byte-for-byte the donor (that is `fidelity_d4`'s assertion).
    # These carry the transplanted commit policy — certify-else-provisional-at-boundary, with
    # the live recert — which is the arc's approved policy for this substrate and is held
    # FIXED across the spiral arms so vocabulary-acquisition timing is not a free variable.
    # ---------------------------------------------------------------------------------------
    # the non-native crank: live earning, no ports. `tall`'s original question, in the regime
    # it needed.
    "enum_live":    {"vocab": "earned", "commit": "delta_prov", "prop_k": None, "span": False,
                     "recert": True},
    # which port carries the effect: routing only.
    "spiral_route": {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                     "recert": True},
    # THE TREATMENT: earn -> consolidate into planner AND executor -> earn the next level
    # natively over the routed policy.
    "spiral":       {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": True,
                     "recert": True},
    # the native ceiling: nothing has to be earned, both ports live. (An alias of the donor's
    # `given_native`, kept because the SPEC's arm table names it separately.)
    "given_sp":     {"vocab": "true",   "commit": None,         "prop_k": 4, "span": True},
    # THE UNCONFOUNDED ENUM CRANK. `enum_live` at G=482 takes a PERMANENT width collapse at the
    # L2 commit (n 32->48; holding width 2 needs G >= 722), measured in Phase A as +0.172 in
    # `e` that never recovers. That handicaps its L3 EARNING trajectory, which would inflate
    # any native-vs-enum rate comparison in the native arms' favour for an apparatus reason.
    # This arm pays the un-handicapped price so the rate claim has a clean enum reference;
    # priced-TIME claims stay inside the G=482 set, where the ledger is matched.
    # ---------------------------------------------------------------------------------------
    # [census] THE CENSUS ARMS. Routing-only throughout (`spiral_route`'s configuration): the
    # both-ports consumption gap is unresolved and letting it into this round would confound
    # the coverage readout with the span question.
    # ---------------------------------------------------------------------------------------
    # the ROUTING-ONLY ceiling: true tables, proposal head, no corridor. `given_native`'s
    # both-ports bracket stays available cross-tag in `sp_s0`.
    "given_route":   {"vocab": "true",   "commit": None,         "prop_k": 4, "span": False},
    # the zero-timing-price op: commit at the certificate, then EXTEND the frozen table.
    "census_extend": {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                      "recert": True, "extend": True},
    # the gate-later op, at L2, as a CONJUNCTION with the certificate.
    "census_gate":   {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                      "recert": True, "cfg": {"gate_level": 2}},
    # the timing control: same commit cycle, no signal.
    "yoked_delay":   {"vocab": "earned", "commit": "yoked",      "prop_k": 4, "span": False,
                      "recert": True, "cfg": {"gate_level": 2}},
    # ---------------------------------------------------------------------------------------
    # [assay] THE SURGERY ARMS. All six are routing-only (`spiral_route`'s configuration) and
    # all but `given_c1` are the anchor exactly, differing only in what the commit installs.
    # ---------------------------------------------------------------------------------------
    "anchor":    {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True},
    "strip":     {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "strip"}},
    "complete":  {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "complete"}},
    "exact":     {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "exact"}},
    "junk_dose": {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                  "recert": True, "cfg": {"surgery": "junk_dose"}},
    # the c1-arrival ceiling, in-tag: a `given_route` replica, so it carries `given`'s stream
    # exactly as `given_route` did in `cs_s0` — which keeps the anchor -> given_c1 bracket the
    # same shape of comparison as `cs_s0`'s spiral_route -> given_route bracket.
    "given_c1":  {"vocab": "true", "commit": None, "prop_k": 4, "span": False},
    # [assay] STREAM-DISPLACED TWINS. Identical to their originals in every config bit; the
    # only difference is a controlled burn of the per-arm RNG stream. They bound the
    # arrival-vs-stream confound the `exact` / `given_c1` residual has to be read against.
    "given_c1_j": {"vocab": "true", "commit": None, "prop_k": 4, "span": False,
                   "cfg": {"stream_burn": 1024}},
    "exact_j":    {"vocab": "earned", "commit": "delta_prov", "prop_k": 4, "span": False,
                   "recert": True, "cfg": {"surgery": "exact", "stream_burn": 1024}},
    # ---------------------------------------------------------------------------------------
    # [conductor] THE OUTER-LOOP ARMS. All six are routing-only (`spiral_route`'s
    # configuration), all carry the recert, all twin onto the SAME stream as the anchor, and all
    # differ from the anchor in exactly one thing: WHO DRIVES THE CRANK. The three driven arms
    # differ from each other in exactly one thing: WHAT THE RULE READS.
    #
    # `commit="loop"` means the outer loop owns the commit, not the certificate and not the era
    # boundary. The shadow certificate keeps running read-only in every arm, so cycles-to-cert
    # stays observable beside what the loop actually did — which is the comparison the round is
    # for. A loop commit is NOT provisional: the loop chose it, so it pays the pre-commit
    # audition, exactly as a certified commit does.
    # ---------------------------------------------------------------------------------------
    # the WITHIN-LEVEL reader: its own error on the era's own cell. `teacher_slot`'s `outer_task`
    # analogue — the arm whose currency is the one the crossing's value is NOT denominated in.
    "outer_ledger": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                     "recert": True, "loop": {"kind": "quiet", "read": "ledger"}},
    # ONE LEVEL UP, free: distinct level-(active+1) tuples at support, mined from the agent's own
    # chosen trajectories by the observation panel.
    "outer_yield": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                    "recert": True, "loop": {"kind": "quiet", "read": "yield"}},
    # ONE LEVEL UP, label-free and priced: the plant's own masked-infill NLL over the span one
    # level above the era's damage cell (`endo_yield`'s read, ported).
    "outer_endo": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                   "recert": True, "loop": {"kind": "quiet", "read": "endo", "priced": True}},
    # THE MANDATORY TIMING CONTROLS (census finding 4): same actions, same cycles, no signal.
    "yoked_yield": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                    "recert": True, "loop": {"kind": "yoke", "of": "outer_yield"}},
    "yoked_endo": {"vocab": "earned", "commit": "loop", "prop_k": 4, "span": False,
                   "recert": True, "loop": {"kind": "yoke", "of": "outer_endo"}},
    "enum_live_g722": {"vocab": "earned", "commit": "delta_prov", "prop_k": None,
                       "span": False, "recert": True,
                       # G=722 is what holds width 2 at n=48; `width_cap=2` stops the same
                       # budget ALSO buying width 3 at n=32 (measured), which would make the
                       # control differ from `enum_live` through era 1 as well and confound the
                       # cycles-to-certification comparison the arm exists to clean up.
                       "cfg": {"g_budget": 722, "width_cap_ref_g": 482}},
}
ARMS["fid"] = dict(ARMS["given_fid"])          # the SPEC's name for the in-tag assertion

# A treated arm and its untreated twin must share a torch RNG stream, or the contrast is
# confounded by stream position (`handle/`'s discipline). Streams are keyed by the TWIN.
TWIN = {
    "given_prop_k4": "given", "span_true": "given", "given_native": "given",
    "given_fid": "given",
    "practice_late_prop_k4": "practice_late", "span_mined": "practice_late",
    "practice_late_native": "practice_late",
    "practice_early_prop_k4": "practice_early",
    # [spiral] the treated spiral arms twin onto `enum_live`, which is their untreated
    # baseline: same vocabulary policy, same commit policy, no ports. `given_sp`/`fid` twin
    # onto `given` exactly as `given_native`/`given_fid` do.
    "spiral_route": "enum_live", "spiral": "enum_live",
    "given_sp": "given", "fid": "given",
    # `enum_live_g722` twins onto `enum_live`: they differ in `g_budget` alone, and at n=32
    # both buy width 2, so they are bit-identical until the L2 commit and diverge EXACTLY at
    # the width transition the arm exists to isolate.
    "enum_live_g722": "enum_live",
    # [census] every treated census arm twins onto the SAME stream as `spiral_route` (the in-tag
    # anchor), so each differs from the anchor by exactly one thing and by nothing else.
    "census_extend": "enum_live", "census_gate": "enum_live", "yoked_delay": "enum_live",
    "given_route": "given",
    # [assay] every surgery arm twins onto `spiral_route`'s stream, so each differs from the
    # anchor by the surgery and by nothing else.
    "anchor": "enum_live", "strip": "enum_live", "complete": "enum_live",
    "exact": "enum_live", "junk_dose": "enum_live",
    "given_c1": "given",
    # the displaced twins take their ORIGINAL's stream key, so the base seed is identical and
    # the burn is the only thing that moves.
    "given_c1_j": "given", "exact_j": "enum_live",
    # [conductor] every loop-driven and yoked arm twins onto the anchor's stream, so each is
    # BIT-IDENTICAL TO THE ANCHOR UNTIL ITS FIRST ACTION — an in-tag gate the reduction checks,
    # and the strongest available statement that the reads themselves are non-invasive.
    "outer_ledger": "enum_live", "outer_yield": "enum_live", "outer_endo": "enum_live",
    "yoked_yield": "enum_live", "yoked_endo": "enum_live",
}
STREAM = {"never_base": 0, "given": 1, "practice_gated": 2, "practice_early": 3,
          "practice_late": 4,
          # [spiral] a NEW stream index, so the spiral arms neither perturb nor inherit the
          # donor arms' stream position.
          "enum_live": 5}


def parse_arms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        base, *rest = part.split(":")
        ov = {}
        for kv in rest:
            k, val = kv.split("=")
            try:
                ov[k] = int(val)
            except ValueError:
                try:
                    ov[k] = float(val)
                except ValueError:
                    ov[k] = val
        label = base + ("" if not ov else "_" + "_".join(f"{k}{x}" for k, x in ov.items()))
        out.append((label, base, ov))
    return out


def macro_moves(level, tables, s, depth, device):
    """Instantiate the level-`level` macro at EVERY node of that level, so a committed
    vocabulary is a position-independent rule set (which is what the DGP's own tables are)
    and the committed action space matches `given`'s move-for-move."""
    n_nodes = s ** (depth - level)
    return [MC.to_device(MC.make_macro(level, j, s, tables[level]), device)
            for j in range(n_nodes)]


def build_ms(base_ms, committed, s, depth, device):
    ms = list(base_ms)
    for level in sorted(committed):
        if committed[level] is None:
            continue
        ms += macro_moves(level, {level: committed[level]}, s, depth, device)
    return ms


# =========================================================================== #
# [assay] INSTRUMENT: per-execution entry identity
# =========================================================================== #

# On-device accumulator. `on` gates recording; `phase` tags which part of the cycle the
# executions came from, so beam executions are separable from instrument ones (auditions,
# probes, the ablation battery). Never touches `counts` and never draws from any RNG.
_ENTRY_REC = {"on": False, "phase": "beam", "acc": {}}


def _rec_reset():
    _ENTRY_REC["acc"] = {}


def _rec_take(device=None):
    """Pull the accumulator to host once, and clear it. One sync per cycle, not per call."""
    out = {}
    for (phase, lvl), t in _ENTRY_REC["acc"].items():
        out.setdefault(phase, {})[str(lvl)] = [int(x) for x in t.cpu().numpy()]
    _rec_reset()
    return out


def _rng_snapshot():
    """[assay] Capture every global RNG stream the arm loop's substrate depends on."""
    import torch
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st):
    import torch
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def macro_features_rec(generator, x, move, s, v):
    """A byte-identical copy of `macros.macro_features`, plus the one line the donor throws
    away: which table entry the max-sum DP actually chose, per row.

    Copied rather than wrapped because the donor returns `(feats, pos)` and discards `best`;
    wrapping would mean recomputing the whole DP to recover it. `entry_recorder_check()`
    asserts this copy reproduces the donor's outputs exactly, and the G-F gate covers it
    end-to-end. The recording is a `bincount` into a device tensor — no host sync, no RNG, no
    effect on the returned values.
    """
    import torch
    blk0, span = move["blk0"], move["span"]
    table = move["table"]
    batch = x.shape[0]
    blocks = torch.arange(blk0, blk0 + span, device=x.device)
    pos = (blocks[:, None] * s + torch.arange(s, device=x.device)[None, :]).reshape(-1)
    pos = pos[None, :].expand(batch, -1).contiguous()

    obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
    logits = generator.block_logits(obs)
    cur = logits[:, blk0:blk0 + span, :]
    if move["level"] == 1:
        return cur.argmax(-1), pos

    chain = move["chain"]
    for child in chain:
        n_par = cur.shape[1] // s
        kids = cur.view(batch, n_par, s, cur.shape[-1])
        total = None
        for i in range(s):
            picked = kids[:, :, i, :].index_select(2, child[:, i].contiguous())
            total = picked if total is None else total + picked
        cur = total
    best = cur.argmax(-1)
    flat = move["flat"]
    # ---- [assay] the instrument, and nothing else -------------------------------------- #
    if _ENTRY_REC["on"]:
        n = int(flat.shape[0])
        key = (_ENTRY_REC["phase"], int(move["level"]))
        acc = _ENTRY_REC["acc"].get(key)
        if acc is None or acc.shape[0] != n:
            acc = torch.zeros(n, dtype=torch.long, device=x.device)
            _ENTRY_REC["acc"][key] = acc
        acc.index_add_(0, best.reshape(-1),
                       torch.ones(best.numel(), dtype=torch.long, device=x.device))
    return flat[best.reshape(-1)].view(batch, span), pos


def _install_entry_recorder():
    """Swap the recording copy in for the donor's. `macros.apply_any` resolves
    `macro_features` as a module global at call time, so this reaches every macro execution
    everywhere without touching `macros.py`. Idempotent."""
    if getattr(MC, "_assay_entry_rec", False):
        return
    MC._assay_macro_features_orig = MC.macro_features
    MC.macro_features = macro_features_rec
    MC._assay_entry_rec = True


def entry_recorder_check(v=8, s=2, depth=6, m=2, max_level=3, n=48, rule_seed=0):
    """Assert the recording copy reproduces the donor's `macro_features` BIT FOR BIT, on every
    macro of a real action set, with the recorder both off and on — the precondition for the
    fidelity gates meaning anything once the instrument is installed."""
    import torch
    from rhm.rhm_data import generate_rules_distinct
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device)
    gen.eval()
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    x = torch.from_numpy(_corrupt(leaves_np, length // s, 3, v, s,
                                  np.random.default_rng(5))).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    orig = getattr(MC, "_assay_macro_features_orig", MC.macro_features)
    out = {"n_macros": 0, "max_abs_feats": 0.0, "max_abs_pos": 0.0, "n_exec_recorded": 0}
    for rec_on in (False, True):
        _ENTRY_REC["on"] = rec_on
        _rec_reset()
        for mv in ms:
            if mv.get("kind") != "macro":
                continue
            with torch.no_grad():
                fa, pa = orig(gen, x, mv, s, v)
                fb, pb = macro_features_rec(gen, x, mv, s, v)
            out["n_macros"] += 1
            out["max_abs_feats"] = max(out["max_abs_feats"],
                                       float((fa.long() - fb.long()).abs().max()))
            out["max_abs_pos"] = max(out["max_abs_pos"],
                                     float((pa.long() - pb.long()).abs().max()))
        if rec_on:
            taken = _rec_take()
            out["n_exec_recorded"] = int(sum(sum(h) for lv in taken.values()
                                             for h in lv.values()))
            out["recorded_levels"] = sorted({k for lv in taken.values() for k in lv})
    _ENTRY_REC["on"] = False
    _rec_reset()
    assert out["max_abs_feats"] == 0.0 and out["max_abs_pos"] == 0.0, \
        f"the recording copy of macro_features is not bit-identical: {out}"
    assert out["n_exec_recorded"] > 0, "the entry recorder recorded nothing"
    # each execution must attribute exactly one entry per row, per macro call
    want = out["n_macros"] // 2 * n
    assert out["n_exec_recorded"] == want, \
        f"recorded {out['n_exec_recorded']} selections, expected {want}"
    out["verdict"] = "PASS"
    return out


# [audiation] the per-cycle series a fork has to reproduce. `analyze_conductor.GF_SERIES`,
# restated here so the in-run gate and the offline one ask the identical question.
GF_SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
             "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]


def replay_delta(ref_log, log, n=None):
    """max|delta| over `GF_SERIES`, cycle for cycle, to the shorter of the two."""
    n = min(len(ref_log["cycle"]), len(log["cycle"])) if n is None else n
    per = {}
    for key in GF_SERIES:
        x = np.asarray(ref_log[key][:n], float)
        y = np.asarray(log[key][:n], float)
        q = min(len(x), len(y))
        per[key] = float(np.abs(x[:q] - y[:q]).max()) if q else 0.0
    return (max(per.values()) if per else 0.0), per, n


def replay_ref_log(replay_ref, arm, data_dir=None):
    """The donor tag's own log for this arm, off the shared volume. `replay_ref` is a
    `<remote>/<tag>` pair, so a fork can check itself against a tag written under a DIFFERENT
    app's prefix — which is exactly the case here (`rhm_practice_conductor/cd_s0`)."""
    if not replay_ref:
        return None
    p = os.path.join(data_dir or DATA_DIR, replay_ref, arm, "results.json")
    if not os.path.isfile(p):
        return None
    return json.load(open(p))["log"]


SCHEMA_MD = """\
# `audiation` record schema — E1 phase 1

Written by the run, into the tag's own outdir, so the dataset carries its own description.
Nothing here is interpreted; this is what the bytes mean.

## What learns, and when — the ordering that gives `heads_c` its meaning

The CONTROLLER IS FROZEN in this substrate (`build_shared` trains it once; `run_arm` only
deep-copies the plant and the value off the shared setup). Three things learn, all of them
inside one cycle's body and nowhere else:

    cycle c
      SNAPSHOT  <- heads_c: value + pi as they stand BEFORE anything in cycle c runs
      (a) practice beam   plans with (value_c, pi_c, plant_c); the per-decision tables are
                          recorded here and ONLY here; tips graded -> succ, dres; every
                          trajectory state pushed to the value buffer `buf`
      (b) mining          the vocabulary is mined from what was solved; no weights move
      (c) plant update    finetune_generator(...)  <- plant_c   -> plant_{c+1}
          value update    value_steps(...)         <- value_c   -> value_{c+1}
      (c') pi update      prop_train(...)          <- pi_c      -> pi_{c+1}
      (d) metering beam   PERFORMANCE conditions, post-update heads, NOT recorded
      (d2) shadow panel / the loop's decision point
      (e..g) recert, commit, era advance  -- these change the ACTION SET `ms`, never a weight
    cycle c+1
      SNAPSHOT  <- heads_{c+1}

So for any state `s` logged in cycle c:

    revision(s) = f(heads_{c+1}, s) - f(heads_c, s),   f in {value, pi}

is exactly the revision cycle c's grades produced at `s`. Two facts to hold onto:

  * THE UPDATE IS PER-CYCLE AND BATCHED. `value_steps` samples `n_grad` batches from a buffer
    holding EVERY trajectory state (cap `buf_cap`) plus replay; `prop_train` samples
    `prop_steps` batches from `pbuf`, which is the last `prop_buf_cap` (state, root, slot)
    pairs from SOLVED trajectories across cycles. So the revision at `s` is not "what cycle
    c's grade at `s` taught the learner" — it is the effect at `s` of one batched update whose
    data is dominated by, but not limited to, cycle c. The per-datum variation E1 is about is
    variation ACROSS STATES under one shared update.
  * THE FINAL SNAPSHOT is indexed `c_last + 1`, so `heads_{c+1} - heads_c` exists for every
    logged cycle including the last.

## `decisions/dec_cXXXX_cYYYY.npz` — two flat tables

Row order inside a step is BATCH-MAJOR, TIP-MINOR, which is the order `tips.reshape(B*W, T)`
uses and therefore the order `grade` returns `succ`/`dres` in: the terminal join is positional.

TIPS — one row per surviving beam entry at each step `0..budget`.

| key | dtype | meaning |
|---|---|---|
| `t_cycle` | int16 | the cycle |
| `t_inst` | int16 | instance index in the cycle's `n_pr` fresh instances |
| `t_step` | int8 | beam step; `0` is the corrupt start, `budget` the graded tip |
| `t_tip` | int16 | index within the beam at that step |
| `t_parent` | int16 | tip index at `step-1` this entry descends from; `-1` at step 0 |
| `t_mv` | int16 | index into `ms` of the move that produced it; `-1` at step 0 |
| `t_root` | int8 | the root target `r*` |
| `t_x` | int8 (n, T) | the configuration |
| `t_z` | float16 (n, D) | `controller.state(x)` — the FROZEN encoder state the heads read |
| `t_pi` | float16 (n, n_slots) | pi's logits at that tip, in SLOT space; NaN off the live action set, and all-NaN on warmup cycles where the beam enumerated and no proposal was read |
| `t_vfin` | float32 | the beam's terminal re-score; NaN except at `t_step == budget` |
| `t_succ` | int8 | exact possible-set membership; `-1` except at `t_step == budget` |
| `t_dres` | float32 | exact residual edits to a valid derivation; NaN except at the tip |

CANDIDATES — one row per SCORED CHILD at each step `0..budget-1`. This is the deliberation
state: the moves the beam materialised and valued and then did not keep.

| key | dtype | meaning |
|---|---|---|
| `c_cycle`, `c_inst`, `c_step` | int16/int16/int8 | join to TIPS |
| `c_parent` | int16 | the tip index at `c_step` this child was expanded from |
| `c_mv` | int16 | index into `ms` |
| `c_score` | float32 | `value(controller.state(child), r*)` — the number the topk ranked |
| `c_child` | int16 | the tip index it became at `c_step + 1`, or `-1` if the topk dropped it |
| `c_src` | int8 | provenance: 0 enumerated (no proposal), 1 pi's top-k, 2 forced (move too new for the head), 4 explore-injected |

`c_src == 4` is where the agency lives: `prop_explore=1` draws one move per tip uniformly from
what pi did NOT propose, on a private numpy stream, so the chosen action is not a deterministic
function of the state. That is the `g > 0` condition teacher-forced NTP structurally lacks.

## `snapshots/`

  `heads_cXXXX.npz`   fp32 numpy, keys `value.<param>` and `prop.<param>`. Rebuild the module
                      and `load_state_dict`; both heads read `(z, root)` and nothing else, so
                      any readout at any logged state is recomputable with no GPU and no
                      controller.
  `plant_cXXXX.npz`   the plant, every `plant_snap_every` cycles. Present for materialisation
                      counterfactuals; NO readout needs it.
  `probe_trace.npz`   `cycle`, `era`, `v` (n_cycles, n_probe), `pi` (n_cycles, n_probe,
                      n_slots), all fp32, plus `probe_x_e{era}` / `probe_r_e{era}`. The probe
                      states are the era's OWN metering set, so they change at an era boundary:
                      a cycle-to-cycle difference is only meaningful WITHIN an era.

## `audiation.json`

The shard manifest, the snapshot manifest, every recompute check, and the PER-CYCLE CONTEXT
(`cycles`): `ms_slots` (position in `ms` -> slot id — this is what makes `t_mv`/`c_mv`
resolvable, and it changes at a commit), `ms_level`/`ms_node`, `avail_slots` (the mask pi's
softmax was taken under), `routed`, `filter_on`, `k_eff`, `forced`, and the committed table
sizes. `slot_offsets` gives the (level -> first slot) layout.

## E1b — the `perdatum` arm (present only in a tag that runs it)

### The knob

`anchor` pools credit: one `value_steps` call samples `n_grad x value_batch` states from a
100k-state buffer plus a replay pool, and one `prop_train` call samples `prop_steps x
prop_batch` pairs from a 60k-pair history. One shared update per cycle, whose cause is a batch.

`perdatum` spends the cycle's grades ONE TRAJECTORY AT A TIME, in arrival order. A DATUM is one
graded trajectory: its `budget+1` lineage-resolved states, the `budget` moves it took (in slot
space), and its terminal grade. Per datum:

    readout BEFORE at (own states, fixed probe, same-cycle reference states)
    one value step on the datum's own states,  label = its own succ
    one pi step on the datum's own (state, slot) pairs, IF it solved
    readout AFTER at the same three sets

That is the same content the donor pools (`push` writes every trajectory state with the tip's
`succ`; `prop_pairs` writes every pair of a solved trajectory) at a different GRANULARITY.

### What is matched, and what is not — stated, not implied

  MATCHED  everything that generates trajectories: the crank, the era ladder, the beam, the
           mining, the commit policy, the recert, the caps, the seed, the RNG streams, and the
           arm's torch stream (it twins onto the anchor's).
  MATCHED  the per-cycle GRADIENT BUDGET, as `n_steps x lr`. Under Adam the per-step parameter
           displacement is ~lr almost regardless of batch size or gradient scale, so `sum(lr)`
           is the honest "how far did the learner move this cycle" invariant. Value:
           `n_grad x value_lr_online` -> `(pd_n + pd_maint_v) x pd_value_lr`, exactly.
  NOT MATCHED, and cannot be: BATCH COMPOSITION. That is the treatment.
  NOT MATCHED, reported: SAMPLES SEEN per cycle, and pi's step count — pi steps only on data
           that SOLVED, so its per-cycle budget follows the solved count. `pd_prop_nominal` is
           the count the lr is set against; the realised count is in `perdatum.json`.
  UNTOUCHED the PLANT. It is still `finetune_generator`, pooled, unchanged. It feeds no
           readout, so leaving it pooled isolates the knob.

### Replay: kept, but SEPARATED rather than mixed

Mixing replay into each per-datum batch would have made every per-datum revision partly caused
by replay samples — destroying the attribution the arm exists to create. Dropping replay
entirely risks drift (value's anti-drift guard is the clean replay pool; pi's is its own
history across commits). So replay is kept as a SEPARATE pooled MAINTENANCE pass, run after the
per-datum steps, on HISTORY ONLY:

  * value: `pd_maint_v` steps at `replay_frac=1.0` — the fixed clean replay pool, nothing else.
  * pi:    `pd_maint_p` steps on `pbuf` as it stood BEFORE this cycle's pairs were appended
           (the append is moved to after the maintenance call for this arm).

No credit for the cycle's own grades is ever pooled, and every logged revision has exactly one
cause. The maintenance pass is not logged as a revision event; its losses are in
`perdatum.json` per cycle, so its size relative to the per-datum steps is checkable.

### `revisions/rev_cXXXX_cYYYY.npz` — one row per update event

`pd_n` events per cycle. Shapes are dense: `n_own = budget+1`, `n_prb = pd_probe_n`,
`n_ref = pd_ref_n`.

| key | dtype | meaning |
|---|---|---|
| `e_cycle`, `e_order` | int16 | the cycle, and the datum's position in arrival order |
| `e_inst`, `e_tip` | int16 | which trajectory (joins to the decision tables' `t_inst`/`t_tip`) |
| `e_succ`, `e_dres`, `e_root` | int8/f32/int8 | the datum's own grade and target |
| `e_did_pi`, `e_n_pairs`, `e_fallback` | int8 | whether the pi step fired, on how many pairs, and whether the cycle hit the nothing-solved fallback |
| `e_vloss`, `e_ploss` | float32 | the losses at that step |
| `e_own_x`, `e_own_z`, `e_own_a` | int8 / f16 / int16 | the datum's states, their frozen encoder states, and THE ACTIONS IT TOOK in slot space — the arity-2 second argument |
| `e_own_v0`, `e_own_pi0` | float16 | the readout BEFORE the step, at the own states (context) |
| `e_own_dv`, `e_own_dpi` | **float32** | the revision at the datum's own states |
| `e_prb_dv`, `e_prb_dpi` | **float32** | the revision at the era's fixed probe |
| `e_ref_dv`, `e_ref_dpi` | **float32** | the revision at same-cycle states this pass did NOT step on — the SPILLOVER term |

**Why the differences are stored, and in fp32.** The per-datum lr is ~1e-6, so a revision is
small; fp16's ulp at a logit of 5 is 4e-3, which is far larger. Storing the two endpoints in
fp16 and differencing them offline would be pure round-off. The difference is therefore formed
on device in fp32 and stored in fp32; only the baseline is fp16, where it is context.

### `perdatum.json`

Per cycle: `n_data`, the realised `n_value_steps` / `n_pi_steps`, `n_ref` / `n_prb`, the
per-datum and maintenance losses separately, the live learning rates, and `sel` / `ref` — the
flat tip indices this cycle stepped on and used as reference. Plus the run-level matched-budget
record (both arms' lrs and step counts).

### The inverse gate

`per_datum` is False in the base config; every line E1b adds is behind it, and `pdrng` is drawn
only inside `per_datum_pass`. So with the flag off the arm's torch and numpy streams are the
donor's, and the `anchor` arm of an E1b tag must replay `cd_s0/anchor` BIT-IDENTICALLY over
every cycle — which is asserted in flight every checkpoint and again offline. That is the proof
that all new code is inert when disabled.

## What is NOT logged

  * the METERING beam (block d), the recert's grading beams, the auditions, the probes and the
    end-of-run ablation battery. Only the practice beam carries the agent's own exploration and
    only it feeds the updates; the rest are instruments.
  * the world. No oracle, no rules, no true tables enter these tables — `t_succ` and `t_dres`
    are the two oracle readouts, and they are the grades the arm already pays for and learns
    from, not side information.
"""


# =========================================================================== #
# [audiation E1b] PER-DATUM CREDIT: the treatment, and its revision recorder
# =========================================================================== #

class RevRec:
    """[E1b] The per-datum revision recorder.

    Phase 1 could store the learner's whole readout state at every cycle boundary, because
    there were 117 of them. E1b has ~15k update events, and a state dict per event is
    gigabytes — so the REVISION is logged directly instead, in-run, as the readout DIFFERENCE
    around each event at three sets of states:

      OWN   the datum's own trajectory states (budget+1 of them) — the revision at the very
            states the update was computed on. The action the trajectory took at each step is
            logged beside them: that is the arity-2 second argument.
      PRB   the run's fixed per-era probe, subsampled to `n_prb`. Fixed across the whole era,
            so a per-event revision here is comparable across every datum of the era.
      REF   `n_ref` tip states of the SAME cycle drawn from trajectories this pass does NOT
            step on. The point is spillover: how much does one datum's update move the
            readout at other data the learner also saw?

    PRECISION, stated because it is the one place this could silently be junk. The revisions
    are small (the per-datum lr is ~1e-6), so storing the two endpoints in fp16 and
    differencing them offline would be pure round-off — fp16's ulp at |v| ~ 5 is 4e-3, far
    larger than the signal. The DIFFERENCE is therefore formed on device in fp32 and stored in
    fp32; only the fp32 BASELINE at the own-states is additionally kept (in fp16, where it is
    context rather than signal).

    RNG. The recorder draws nothing. The pass it serves draws exactly twice per cycle (which
    data to step on, which states form REF), both from a dedicated numpy stream that is
    touched only when per-datum mode is ON — which is what licenses the inverse gate.
    """

    def __init__(self, n_slots):
        self.n_slots = int(n_slots)
        self.cyc = None
        self.rows = []

    def open(self, cycle, meta):
        self.cyc = {"cycle": int(cycle), "meta": dict(meta), "ev": []}

    def add(self, ev):
        self.cyc["ev"].append(ev)

    def close(self):
        c, self.cyc = self.cyc, None
        return c


def flatten_rev_cycle(c):
    """One cycle's revision events -> flat numpy arrays. Every event has the same shapes
    (`budget+1` own states, `n_prb` probe, `n_ref` reference), so the tables are dense."""
    ev = c["ev"]
    if not ev:
        return {}
    out = {}
    scal = ("order", "inst", "tip", "succ", "root", "did_pi", "n_pairs", "fallback")
    for k in scal:
        out[f"e_{k}"] = np.asarray([q[k] for q in ev],
                                   dtype=np.int16 if k not in ("succ", "did_pi", "root",
                                                               "fallback") else np.int8)
    out["e_cycle"] = np.full(len(ev), c["cycle"], dtype=np.int16)
    for k in ("dres", "vloss", "ploss"):
        out[f"e_{k}"] = np.asarray([q[k] for q in ev], dtype=np.float32)
    for k in ("own_dv", "own_dpi", "prb_dv", "prb_dpi", "ref_dv", "ref_dpi"):
        out[f"e_{k}"] = np.stack([q[k] for q in ev]).astype(np.float32)
    for k in ("own_v0", "own_pi0", "own_z"):
        out[f"e_{k}"] = np.stack([q[k] for q in ev]).astype(np.float16)
    out["e_own_x"] = np.stack([q["own_x"] for q in ev]).astype(np.int8)
    out["e_own_a"] = np.stack([q["own_a"] for q in ev]).astype(np.int16)
    m = c["meta"]
    for k, v in m.items():
        if isinstance(v, np.ndarray):
            out[f"m_{k}"] = v
    return out


class RevWriter:
    """Shards the revision tables, on the same clock as `DecWriter`."""

    def __init__(self, root, arm, flush_every=5, compress=True):
        self.dir = os.path.join(root, arm, "revisions")
        os.makedirs(self.dir, exist_ok=True)
        self.flush_every = int(flush_every)
        self.compress = bool(compress)
        self.pending, self.manifest = [], []
        self.n_ev = 0

    def add(self, c):
        f = flatten_rev_cycle(c)
        if f:
            self.pending.append((f, c["cycle"]))
        if len(self.pending) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.pending:
            return None
        first, last = self.pending[0][1], self.pending[-1][1]
        keys = set()
        for f, _ in self.pending:
            keys |= set(f)
        out = {}
        for k in sorted(keys):
            parts = [f[k] for f, _ in self.pending if k in f]
            out[k] = np.concatenate(parts) if k.startswith("e_") else parts[0]
        name = f"rev_c{first:04d}_c{last:04d}.npz"
        path = os.path.join(self.dir, name)
        (np.savez_compressed if self.compress else np.savez)(path, **out)
        n = int(out["e_cycle"].shape[0])
        self.n_ev += n
        self.manifest.append({"file": name, "first_cycle": int(first), "last_cycle": int(last),
                              "n_events": n, "bytes": int(os.path.getsize(path))})
        self.pending = []
        volume.commit()
        return path


def per_datum_pass(*, out, r_np, succ, dres, ms, offsets, avail, controller, value, prop,
                   vopt, popt, budget, device, cfg, pdrng, rec, cyc, era_i, tips_flat,
                   probe_z, probe_r, n_slots, train_on="solved"):
    """[E1b] THE TREATMENT. The cycle's grades are consumed ONE TRAJECTORY AT A TIME, in
    arrival order, each with its own immediate gradient step — instead of `value_steps` /
    `prop_train`'s pooled sampling from a buffer.

    A DATUM is one graded trajectory: its `budget+1` lineage-resolved states, the `budget`
    moves it took (in SLOT space), and its terminal grade. That is exactly the unit the donor
    pools — `push` writes every trajectory state with the tip's `succ` as the label, and
    `prop_pairs` writes every (state, root, slot) pair of a solved trajectory — so the
    treatment changes the GRANULARITY of credit and nothing about what is credited.

    Per datum, in order:
        readout BEFORE at (own, probe, reference)
        one value step on the datum's own states, label = its own `succ`
        one pi step on the datum's own (state, slot) pairs, IF it solved
        readout AFTER at the same three sets
    and the fp32 difference is recorded. Nothing here draws from any shared RNG stream; the
    two draws (which data, which reference states) come from `pdrng`.
    """
    import torch
    import torch.nn.functional as F
    B = int(r_np.shape[0])
    W = int(tips_flat.shape[0]) // B
    n = B * W
    pd_n = min(int(cfg["pd_n"]), n)
    sel = np.sort(pdrng.choice(n, size=pd_n, replace=False))
    rest = np.setdiff1d(np.arange(n), sel)
    n_ref = min(int(cfg["pd_ref_n"]), rest.size)
    ref = np.sort(pdrng.choice(rest, size=n_ref, replace=False)) if n_ref else np.zeros(0, int)

    roots_flat = np.repeat(r_np, W)
    with torch.no_grad():
        ref_x = tips_flat[torch.from_numpy(ref).to(device)] if n_ref else tips_flat[:0]
        ref_z = _encode_chunked(controller, ref_x) if n_ref else None
        ref_r = torch.from_numpy(roots_flat[ref]).to(device).long() if n_ref else None
    n_prb = min(int(cfg["pd_probe_n"]), int(probe_z.shape[0]))
    prb_z, prb_r = probe_z[:n_prb], probe_r[:n_prb]

    slot_t = torch.as_tensor(PN.move_slots(ms, offsets), dtype=torch.long, device=device)
    seq = out["tips_seq"]                                   # (B, W, budget), indices into ms
    traj = out["traj"]                                      # budget+1 x (B, W, T)
    succ_bw = np.asarray(succ).reshape(B, W)
    dres_bw = np.asarray(dres).reshape(B, W)
    neg = torch.finfo(torch.float32).min / 4
    # the donor's "nothing solved -> imitate the survivors instead of freezing the filter in"
    fallback = bool(train_on == "solved" and not (succ_bw.reshape(-1)[sel] > 0.5).any())

    def _read():
        with torch.no_grad():
            a = (value(zs, rs).float(), prop(zs, rs).float())
            b = (value(prb_z, prb_r).float(), prop(prb_z, prb_r).float())
            c = ((value(ref_z, ref_r).float(), prop(ref_z, ref_r).float())
                 if n_ref else (None, None))
        return a, b, c

    n_v = n_p = 0
    vl_sum = pl_sum = 0.0
    value.train()
    if prop is not None:
        prop.train()
    for order, i in enumerate(sel):
        b_, w_ = int(i) // W, int(i) % W
        xs = torch.stack([traj[t][b_, w_] for t in range(budget + 1)]).to(device)
        with torch.no_grad():
            zs = _encode_chunked(controller, xs)
        rs = torch.full((budget + 1,), int(r_np[b_]), dtype=torch.long, device=device)
        a_slots = slot_t[seq[b_, w_]]                       # (budget,)
        y = float(succ_bw[b_, w_] > 0.5)

        (v0, pi0), (pv0, ppi0), (rv0, rpi0) = _read()

        loss_v = F.binary_cross_entropy_with_logits(
            value(zs, rs), torch.full((budget + 1,), y, device=device))
        vopt.zero_grad(set_to_none=True)
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        vopt.step()
        n_v += 1
        vl_sum += float(loss_v.item())

        did_pi, lp = 0, float("nan")
        if prop is not None and (y > 0.5 or fallback):
            logits = prop(zs[:budget], rs[:budget]).masked_fill(~avail[None, :], neg)
            loss_p = F.cross_entropy(logits, a_slots)
            popt.zero_grad(set_to_none=True)
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(prop.parameters(), 1.0)
            popt.step()
            did_pi, lp = 1, float(loss_p.item())
            n_p += 1
            pl_sum += lp

        (v1, pi1), (pv1, ppi1), (rv1, rpi1) = _read()
        rec.add({
            "order": order, "inst": b_, "tip": w_,
            "succ": int(y > 0.5), "root": int(r_np[b_]), "dres": float(dres_bw[b_, w_]),
            "did_pi": did_pi, "n_pairs": int(budget if did_pi else 0),
            "fallback": int(fallback), "vloss": float(loss_v.item()), "ploss": lp,
            "own_dv": (v1 - v0).cpu().numpy(), "own_dpi": (pi1 - pi0).cpu().numpy(),
            "prb_dv": (pv1 - pv0).cpu().numpy(), "prb_dpi": (ppi1 - ppi0).cpu().numpy(),
            "ref_dv": ((rv1 - rv0).cpu().numpy() if n_ref else np.zeros(0, np.float32)),
            "ref_dpi": ((rpi1 - rpi0).cpu().numpy() if n_ref
                        else np.zeros((0, n_slots), np.float32)),
            "own_v0": v0.cpu().numpy(), "own_pi0": pi0.cpu().numpy(),
            "own_z": zs.cpu().numpy(), "own_x": xs.cpu().numpy(),
            "own_a": a_slots.cpu().numpy(),
        })
    value.eval()
    if prop is not None:
        prop.eval()
    return {"n_data": int(pd_n), "n_value_steps": n_v, "n_pi_steps": n_p,
            "vloss": (vl_sum / n_v if n_v else None),
            "ploss": (pl_sum / n_p if n_p else None),
            "fallback": fallback, "n_ref": int(n_ref), "n_prb": int(n_prb),
            "sel": sel.astype(np.int32), "ref": ref.astype(np.int32)}


# =========================================================================== #
# [audiation] GATE B — the instrumented beams reproduce the DONOR'S, bit for bit
# =========================================================================== #

def beam_gate(v=8, s=2, depth=6, m=2, max_level=3, n=24, rule_seed=0, budget=3,
              beam_width=4, k=4, explore=1):
    """Assert that `beam_moves` and `beam_moves_prop` in THIS file reproduce
    `conductor.beam_moves` / `conductor.beam_moves_prop` exactly, with the recorder both OFF
    and ON, on a real mixed action set (base moves + true macros).

    Compared against the DONOR MODULE rather than against a copy of it — the instrument is a
    diff on someone else's file, and the only statement worth asserting is a statement about
    that file. `entry_recorder_check`'s idiom, one object up.

    THE RNG DISCIPLINE. Every arm is run from the same snapshotted global state, and the prop
    beam's `erng` (which `explore_moves` draws from) is rebuilt per arm from the same seed, so
    a difference can only come from the instrument. The check itself is run inside the
    caller's own sandbox (`conductor_run`'s `_rng0` pattern), and asserts nothing about the
    stream it leaves behind — the caller restores it.

    Returns the per-key max|delta| for every output the beam produces, plus the recorder's own
    row counts so "recorded nothing" cannot pass as "recorded identically".
    """
    import torch
    from rhm.practice.conductor import conductor as CD          # the donor, unmodified
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device).eval()
    ctrl = _build_rich_controller()(v, length, s, 96, n_head=4, n_layer=2).to(device).eval()
    val = _build_value_head()(96, v).to(device).eval()
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    x0 = torch.from_numpy(_corrupt(leaves_np, length // s, 3, v, s,
                                   np.random.default_rng(5))).to(device)
    roots = torch.from_numpy(roots_np).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    offsets, n_slots = PN.slot_layout(depth, s, max_level)
    slots = PN.move_slots(ms, offsets)
    prop = PN.build_head(96, v, n_slots, 99, device)
    prop.eval()

    def _cmp(a, b):
        d = 0.0
        for key in ("x", "seq", "tips_x", "tips_seq"):
            d = max(d, float((a[key].long() - b[key].long()).abs().max()))
        for key in a["counts"]:
            d = max(d, abs(a["counts"][key] - b["counts"].get(key, 0)))
        for t in range(len(a.get("traj") or [])):
            d = max(d, float((a["traj"][t].long() - b["traj"][t].long()).abs().max()))
        return d

    st = _rng_snapshot()
    out = {"n_moves": len(ms), "n_slots": int(n_slots)}
    common = dict(budget=budget, beam_width=beam_width, device=device, collect=True)

    # ---- the ENUMERATED beam ----------------------------------------------------------- #
    _rng_restore(st)
    a = CD.beam_moves(ctrl, gen, val, x0, roots, ms, rules_t, canon, depth, v, m, s, **common)
    _rng_restore(st)
    b0 = beam_moves(ctrl, gen, val, x0, roots, ms, rules_t, canon, depth, v, m, s, **common)
    rec = DecRec(n_slots)
    rec.open(1, {"n_slots": n_slots})
    _rng_restore(st)
    b1 = beam_moves(ctrl, gen, val, x0, roots, ms, rules_t, canon, depth, v, m, s,
                    rec=rec, **common)
    ce = rec.close()
    out["enum_rec_off"] = _cmp(a, b0)
    out["enum_rec_on"] = _cmp(a, b1)
    out["enum_tip_steps"] = int(sum(q["x"].shape[0] * q["x"].shape[1] for q in ce["tip"]))
    out["enum_cands"] = int(sum(q["mv"].size for q in ce["cand"]))

    # ---- the PROPOSED beam, WITH exploration (the practice configuration) --------------- #
    port = dict(prop=prop, slots=slots, k=k, forced=(), n_slots=n_slots, explore=explore)
    _rng_restore(st)
    a = CD.beam_moves_prop(ctrl, gen, val, x0, roots, ms, rules_t, canon, depth, v, m, s,
                           erng=np.random.default_rng(7), **port, **common)
    _rng_restore(st)
    b0 = beam_moves_prop(ctrl, gen, val, x0, roots, ms, rules_t, canon, depth, v, m, s,
                         erng=np.random.default_rng(7), **port, **common)
    rec.open(1, {"n_slots": n_slots})
    _rng_restore(st)
    b1 = beam_moves_prop(ctrl, gen, val, x0, roots, ms, rules_t, canon, depth, v, m, s,
                         erng=np.random.default_rng(7), rec=rec, **port, **common)
    cp = rec.close()
    out["prop_rec_off"] = _cmp(a, b0)
    out["prop_rec_on"] = _cmp(a, b1)
    out["prop_tip_steps"] = int(sum(q["x"].shape[0] * q["x"].shape[1] for q in cp["tip"]))
    out["prop_cands"] = int(sum(q["mv"].size for q in cp["cand"]))
    out["prop_pi_rows"] = int(sum(0 if q["pi"] is None else q["pi"].shape[0] * q["pi"].shape[1]
                                  for q in cp["tip"]))
    # the provenance byte must actually SEPARATE the classes, or it is decoration
    src = np.concatenate([q["src"].reshape(-1) for q in cp["cand"]])
    out["src_hist"] = {str(int(u)): int(c) for u, c in zip(*np.unique(src, return_counts=True))}
    _rng_restore(st)

    for key in ("enum_rec_off", "enum_rec_on", "prop_rec_off", "prop_rec_on"):
        assert out[key] == 0.0, f"[audiation] beam_gate FAILED on {key}: max|delta|={out[key]}"
    assert out["enum_tip_steps"] > 0 and out["prop_tip_steps"] > 0, \
        f"[audiation] the recorder recorded no tips: {out}"
    assert out["prop_pi_rows"] > 0, "[audiation] the recorder recorded no pi"
    assert str(SRC_EXPLORE) in out["src_hist"], \
        f"[audiation] no explore-injected candidate was tagged: {out['src_hist']}"
    out["verdict"] = "PASS"
    return out


def gy_soundness(s=2, depth=6, level=4, eras=None, v=8):
    """[census] THE G-Y INSTRUMENT'S SOUNDNESS CHECK — a known-composition toy, hand-enumerated.

    The spiral README promised the next-level-yield readout CONDITIONALLY on the instrument
    being mechanically sound, so the check is part of the record rather than an assumption.
    Three things are asserted, all at sizes small enough to count by hand:

      1. SPAN. A level-`level` miner keys on `s**(level-1)` level-1 features — 8 blocks at
         level 4, s=2. Wrong span means the instrument is silently counting a different object.
      2. COUNTS. Feed a hand-built array with a known multiset of rows and check `n_distinct`,
         `n_obs` and `n_at_support` at every threshold against the hand count. This is what
         catches a keying bug (e.g. keying on positions instead of values, or collapsing rows).
      3. NODE INDEXING. For every era of the real ladder, the level-`level` node the instrument
         reads must be in range (`s**(depth-level)` nodes exist) and its column slice must lie
         inside a real configuration's block count.
    """
    span = s ** (level - 1)
    out = {"level": level, "span_blocks": span, "expected_span": span}
    mn = MC.Miner(level, s)
    assert mn.span == span, f"G-Y span is {mn.span}, expected {span}"

    # 2. hand-enumerated counts: A x3, B x2, C x1 -> 3 distinct, 6 observations
    A = list(range(span))                       # 0,1,2,...,span-1
    B = [0] * span
    C = [1] + [0] * (span - 1)                  # differs from B in one position only
    rows = np.array([A, B, A, C, B, A], dtype=np.int64)
    mn.observe(rows)
    st = mn.state()
    hand = {"n_obs": 6, "n_distinct": 3,
            "n_at_support": {"1": 3, "2": 2, "3": 1, "5": 0, "10": 0}}
    out["hand"] = hand
    out["measured"] = {"n_obs": st["n_obs"], "n_distinct": st["n_distinct"],
                       "n_at_support": st["n_at_support"]}
    assert st["n_obs"] == hand["n_obs"], f"G-Y n_obs {st['n_obs']} != {hand['n_obs']}"
    assert st["n_distinct"] == hand["n_distinct"], (
        f"G-Y n_distinct {st['n_distinct']} != {hand['n_distinct']} — B and C differ in exactly "
        f"one position, so a keying bug that collapsed them would show here")
    for t, want in hand["n_at_support"].items():
        got = st["n_at_support"][t]
        assert got == want, f"G-Y n_at_support[{t}] = {got}, hand count {want}"

    # the identity instrument, if installed, must report the same key set
    if "keys_at_support" in st:
        keys3 = [tuple(k) for k in st["keys_at_support"]]
        assert keys3 == [tuple(A)], f"G-Y keys_at_support at support 3 = {keys3}, expected [A]"
        out["keys_at_support_3"] = keys3

    # 3. node indexing over the REAL ladder
    n_nodes = s ** (depth - level)
    n_blocks = s ** depth // s
    nodes = []
    for era in (eras or parse_eras(CENSUS_LADDER)):
        node = (era["node"] * s ** (era["level"] - 1)) // span
        ok = (0 <= node < n_nodes) and ((node + 1) * span <= n_blocks)
        nodes.append({"era": era["name"], "level": era["level"], "node": era["node"],
                      f"L{level}_node": int(node), "in_range": bool(ok)})
        assert ok, (f"G-Y node {node} out of range for {era['name']} "
                    f"({n_nodes} nodes at level {level}, {n_blocks} blocks)")
    out["nodes_per_era"] = nodes
    out["n_nodes_at_level"] = n_nodes
    # 4. the instrument is DECOUPLED from what may be committed
    out["decoupled_from_max_macro_level"] = bool(level > 3)
    out["verdict"] = "PASS"
    return out


# =========================================================================== #
# [conductor] THE ENDO READ — endo_yield's label-free one-level-up loss, ported
# =========================================================================== #

def parent_span_blocks(era, s, depth):
    """The BLOCK range of the span ONE LEVEL ABOVE this era's damage cell.

    RHM's tree is balanced, so this is pure arithmetic on the grammar's SHAPE (`s`, `depth`)
    and the era's own cell — both of which the agent has, because the cell is the world it is
    living in. No table, no rule set, no truth mask enters. This is
    `endo_yield/SPEC.md` §3's `pos_top_level` argument, re-expressed for a masked-block plant:
    the cell at (level l, node n) covers leaves [n*s^l, (n+1)*s^l); its parent is
    (l+1, n//s) and covers twice as many, and a block is `s` leaves."""
    pl, pn = era["level"] + 1, era["node"] // s
    lo, hi = pn * s ** pl, (pn + 1) * s ** pl
    length = s ** depth
    lo, hi = max(0, min(lo, length)), max(0, min(hi, length))
    return int(lo // s), int(hi // s)


def cell_span_blocks(era, s, depth):
    """The block range of the era's OWN damage cell — the level-matched within-level control,
    logged in the shadow panel and never driven."""
    lo, hi = era["node"] * s ** era["level"], (era["node"] + 1) * s ** era["level"]
    length = s ** depth
    lo, hi = max(0, min(lo, length)), max(0, min(hi, length))
    return int(lo // s), int(hi // s)


def endo_read(generator, x, bottom_map, blocks, *, v, s, n_blocks):
    """THE PLANT'S OWN MASKED-INFILL NLL over a block range: the learner's own loss, on its own
    data, at positions fixed by tree arithmetic.

    The WHOLE range is masked and the WHOLE range is scored. That is what makes it a one-level-
    up read rather than a within-level one: with the parent span entirely unobserved, recovering
    it requires the level-(l+1) latent inferred from OUTSIDE the span, where masking only the
    cell requires the level-l latent. The target is `bottom_map`'s level-1 features — the plant's
    own training objective, optimised by every arm in every cycle — so nothing here is an oracle
    the loop would not otherwise have.

    Deterministic and non-invasive by construction: a fixed batch, no sampling, `no_grad`, and
    the net left in whatever mode the cycle put it in. It consumes no draw of any RNG, which is
    what lets the shadow panel run in EVERY arm without voiding the fidelity gate."""
    import torch
    import torch.nn.functional as F
    b0, b1 = blocks
    if b1 <= b0:
        return None
    b = x.shape[0]
    powers = v ** torch.arange(s, device=x.device)
    feats = bottom_map[(x.view(b, n_blocks, s) * powers).sum(-1)]
    obs = x.clone()
    obs[:, b0 * s:b1 * s] = -1
    with torch.no_grad():
        logits = generator.block_logits(obs)
    lg = logits[:, b0:b1, :].reshape(-1, logits.shape[-1])
    tg = feats[:, b0:b1].reshape(-1)
    ok = tg >= 0
    if int(ok.sum()) == 0:
        return None
    with torch.no_grad():
        return float(F.cross_entropy(lg[ok], tg[ok]).item())


def endo_gate(s=2, depth=6, eras=None):
    """N-1: THE READ IS LABEL-FREE BY CONSTRUCTION, checked the way `endo_yield` checked its
    own — by asserting what the read's POSITIONS depend on, and that they land where the tree
    says they land.

    `parent_span_blocks` is a closed form in (era level, era node, s, depth) and touches no
    rules array, no `truth`, no `inverse_maps` and no committed table. The check enumerates the
    ladder's cells and asserts (i) the parent strictly contains the cell, (ii) it is exactly `s`
    times as wide, (iii) it is in range, and (iv) it is the cell's parent under integer node
    arithmetic."""
    eras = eras or parse_eras(CENSUS_LADDER)
    rows = []
    for i, era in enumerate(eras):
        cb = cell_span_blocks(era, s, depth)
        pb = parent_span_blocks(era, s, depth)
        wide = (pb[1] - pb[0]) == s * (cb[1] - cb[0]) or (pb[1] - pb[0]) == s ** depth // s
        assert pb[0] <= cb[0] and cb[1] <= pb[1], f"era {i+1}: parent does not contain the cell"
        assert 0 <= pb[0] < pb[1] <= s ** depth // s, f"era {i+1}: parent out of range"
        assert wide, f"era {i+1}: parent is not s x the cell ({pb} vs {cb})"
        rows.append({"era": i + 1, "cell": era["name"], "cell_blocks": list(cb),
                     "parent_level": era["level"] + 1, "parent_node": era["node"] // s,
                     "parent_blocks": list(pb),
                     "parent_is_root": bool(pb == (0, s ** depth // s))})
    # The check is on the read's CODE, not its prose: parse the function, drop its docstring,
    # and assert that no oracle name appears in what actually executes. `endo_read` is checked
    # the same way — it may touch `bottom_map` (the plant's own training target, which every arm
    # optimises every cycle) and nothing else that carries grammar content.
    def _body(fn):
        import ast
        tree = ast.parse(inspect.getsource(fn))
        fdef = tree.body[0]
        if (fdef.body and isinstance(fdef.body[0], ast.Expr)
                and isinstance(fdef.body[0].value, ast.Constant)
                and isinstance(fdef.body[0].value.value, str)):
            fdef.body = fdef.body[1:]
        return ast.unparse(tree)

    banned = ("truth", "rules", "inverse_maps", "committed", "canon", "true_mask", "grade")
    checked = {}
    for fn in (parent_span_blocks, cell_span_blocks, endo_read):
        body = _body(fn)
        hits = [b for b in banned if b in body]
        assert not hits, f"{fn.__name__} touches {hits} — the read is not label-free"
        checked[fn.__name__] = sorted(set(body.split()) & {"bottom_map", "s", "depth"})
    return {"verdict": "PASS", "depends_on": ["era.level", "era.node", "s", "depth"],
            "code_checked": sorted(checked), "banned_absent": list(banned), "rows": rows}


def endo_bench(shared, cfg, device, n, reps=3):
    """N-2: WHAT THE READ COSTS, measured on the same GPU rather than assumed — the donor's
    `slot_lm.bench` convention. A GROUNDING in this substrate is `score one state` (an encoder
    pass plus a value read, which is what `beam_moves` charges one of per materialised child),
    so the endo read's price is quoted in grounding-equivalents: the wall time of one
    `block_logits` forward over `n` sequences, divided by the per-state cost of the encode the
    beam pays for."""
    import torch
    x = shared["probe_clean"][:n].to(device)
    gen = shared["generator0"]
    ctl = shared["controller"]

    def _t(fn):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.time() - t0) / reps

    obs = x.clone()
    obs[:, :2] = -1
    t_endo = _t(lambda: torch.no_grad()(gen.block_logits)(obs))
    t_ground = _t(lambda: torch.no_grad()(_encode_chunked)(ctl, x))
    per_state = t_ground / max(n, 1)
    price = max(1, int(round(t_endo / per_state))) if per_state > 0 else n
    return {"n": int(n), "t_endo_s": t_endo, "t_ground_batch_s": t_ground,
            "t_per_grounding_s": per_state, "price_g": int(price),
            "note": "grounding-equivalents per endo read, measured on this GPU"}


def floor_gate(cfg):
    """The dead zones that governed this run, checked against `floors.json` so the literals in
    `MEASURED_FLOORS` and the derivation on disk cannot drift apart, and asserted non-default:
    a driven read with no measured floor is an error, not a silent zero."""
    out = {"used": {k: cfg[k] for k in ("tol_ledger", "tol_yield_l3", "tol_yield_l4",
                                        "tol_endo")},
           "provenance": MEASURED_FLOORS["provenance"]}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floors.json")
    if os.path.isfile(p):
        f = json.load(open(p))["floors"]
        same = (abs(f["ledger"] - cfg["tol_ledger"]) < 1e-12
                and abs(f["yield_by_level"]["3"] - cfg["tol_yield_l3"]) < 1e-12
                and abs(f["yield_by_level"]["4"] - cfg["tol_yield_l4"]) < 1e-12)
        out["matches_floors_json"] = bool(same)
        assert same, f"MEASURED_FLOORS has drifted from floors.json: {f} vs {out['used']}"
    for k in ("tol_ledger", "tol_yield_l3", "tol_yield_l4", "tol_endo"):
        assert cfg[k] and cfg[k] > 0, f"{k} is not a measured floor ({cfg[k]!r})"
    out["verdict"] = "PASS"
    return out


# =========================================================================== #
# [assay] THE REALISTIC JUNK POOL, and the surgery itself
# =========================================================================== #

CS_TAGS = [("rhm_practice_census", "cs_s0"), ("rhm_practice_spiral", "sp_s0")]


def build_junk_pool(truth, max_level, tags=None, data_dir=None):
    """Tuples the arc's own loops actually mined and that are FALSE — the kind of junk
    selection produces, not uniform random tuples.

    Read off `log["miner"][...]["keys_at_support"]` (the entry-identity instrument installed
    since `tall/`) in every arm of `census/cs_s0` and `spiral/sp_s0` on the volume, unioned,
    then keyed against `MC.true_tables`. Sorted for determinism; the pool and its provenance
    are logged verbatim in `setup.json`.
    """
    data_dir = data_dir or DATA_DIR
    pool, seen, prov = {ell: [] for ell in range(2, max_level + 1)}, set(), []
    for remote, tag in (tags or CS_TAGS):
        root = os.path.join(data_dir, remote, tag)
        if not os.path.isdir(root):
            prov.append({"tag": tag, "found": False})
            continue
        arms, n_keys = [], 0
        for arm in sorted(os.listdir(root)):
            fp = os.path.join(root, arm, "results.json")
            if not os.path.isfile(fp):
                continue
            res = json.load(open(fp))
            arms.append(arm)
            for cyc in res["log"]["miner"]:
                for ell in range(2, max_level + 1):
                    for k in (cyc.get(str(ell)) or {}).get("keys_at_support", []) or []:
                        t = tuple(int(x) for x in k)
                        if (ell, t) in seen:
                            continue
                        seen.add((ell, t))
                        n_keys += 1
                        pool[ell].append(t)
        prov.append({"tag": tag, "found": True, "arms": arms, "new_keys": n_keys})
    out, split = {}, {}
    for ell in range(2, max_level + 1):
        true_set = {tuple(int(x) for x in r) for r in truth[ell]["flat"]}
        false_keys = sorted(t for t in pool[ell] if t not in true_set)
        out[ell] = false_keys
        split[str(ell)] = {"mined_distinct": len(pool[ell]), "false": len(false_keys),
                           "true": len(pool[ell]) - len(false_keys)}
    return out, {"provenance": prov, "split": split}


def rows_from_flats(flats, lower, s, span):
    """Child-index rows over `lower` for each flat level-1 tuple; unbuildable ones dropped.

    Same idiom as `census.extend_candidates` and `macros.Miner.build`: an entry exists only if
    every one of its `s` halves is already an entry of the lower table (the ratchet)."""
    half = span // s
    lut = {tuple(int(x) for x in r): i for i, r in enumerate(lower["flat"])}
    rows, dropped = [], []
    for k in flats:
        k = tuple(int(x) for x in k)
        kids = [lut.get(k[i * half:(i + 1) * half]) for i in range(s)]
        if any(j is None for j in kids):
            dropped.append(list(k))
        else:
            rows.append(kids)
    return rows, dropped


def apply_surgery(mode, level, tbl, lower, truth_l, junk_pool, s, truth_full=None):
    """Replace the mined table `tbl` with the arm's constructed one. Returns (table, record).

    `lower` is the arm's OWN committed lower table, so every constructed entry is buildable
    over what that arm actually holds — the r-squared ratchet is respected, not bypassed. The
    only exception is `exact`, which installs the DGP's own table verbatim (whose lower is the
    DGP's own lower, and which that arm committed at the level below one era earlier), exactly
    as `vocab == "true"` does.
    """
    span = s ** (level - 1)
    own_flat = [tuple(int(x) for x in r) for r in tbl["flat"]]
    true_set = {tuple(int(x) for x in r) for r in truth_l["flat"]}
    own_child = [list(map(int, r)) for r in tbl["child"]]
    rec = {"mode": mode, "level": level, "n_own": len(own_child),
           "n_own_true": sum(1 for k in own_flat if k in true_set),
           "n_added": 0, "n_added_true": 0, "n_removed": 0, "n_unbuildable": 0}
    if mode in (None, "none"):
        return tbl, rec
    if mode == "exact":
        out = truth_full[level] if truth_full else truth_l
        rec.update({"n_after": int(out["child"].shape[0]), "n_removed": len(own_child),
                    "n_added": int(out["child"].shape[0]),
                    "n_added_true": int(out["child"].shape[0])})
        return out, rec
    if mode == "strip":
        keep = [c for c, k in zip(own_child, own_flat) if k in true_set]
        rec["n_removed"] = len(own_child) - len(keep)
        out = MC.make_table(level, np.asarray(keep, np.int64).reshape(-1, s), lower, s)
    elif mode in ("complete", "junk_dose"):
        missing = sorted(k for k in true_set if k not in own_flat)
        add_rows, dropped = rows_from_flats(missing, lower, s, span)
        rec["n_missing_true"] = len(missing)
        rec["K"] = len(add_rows)                 # `complete`'s addition count == the dose
        rec["n_unbuildable"] = len(dropped)
        if mode == "junk_dose":
            have = set(own_flat)
            cand = [k for k in (junk_pool.get(level) or []) if k not in have]
            add_rows, dropped2 = rows_from_flats(cand, lower, s, span)
            add_rows = add_rows[:rec["K"]]       # matched dose, deterministic order
            rec["n_junk_available_buildable"] = len(cand) - len(dropped2)
            rec["n_unbuildable"] = len(dropped2)
            rec["dose_shortfall"] = max(0, rec["K"] - len(add_rows))
        rec["n_added"] = len(add_rows)
        out = MC.make_table(level, np.asarray(own_child + add_rows,
                                              np.int64).reshape(-1, s), lower, s)
    else:
        raise ValueError(f"unknown surgery mode {mode}")
    out_flat = [tuple(int(x) for x in r) for r in out["flat"]]
    rec["n_after"] = len(out_flat)
    rec["n_added_true"] = sum(1 for k in out_flat if k in true_set) - rec["n_own_true"]
    rec["flat_after"] = [list(k) for k in out_flat]
    rec["true_mask_after"] = [int(k in true_set) for k in out_flat]
    rec.update({f"tab_{k}": val for k, val in MC.grade_table(out, truth_l).items()})
    return out, rec


def extend_candidates(miner, lower, support, frozen):
    """[census] The rows `Miner.build` would produce that `frozen` does not already hold, as
    child indices into `lower` — recomputed against the CURRENT lower table so the indices are
    correct by construction rather than inherited from a stale build.

    Ordered by (-count, key): the most-supported candidate is offered first, and the order is
    deterministic, so a greedy admission is reproducible.
    """
    half = miner.span // miner.s
    lut = {tuple(int(x) for x in row): i for i, row in enumerate(lower["flat"])}
    have = {tuple(int(x) for x in r) for r in frozen["flat"]}
    out = []
    for k, c in sorted(miner.counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if c < support or k in have:
            continue
        kids = [lut.get(k[i * half:(i + 1) * half]) for i in range(miner.s)]
        if any(j is None for j in kids):
            continue
        out.append({"key": k, "child": kids, "count": int(c)})
    return out


def refresh_upper(committed, level, s):
    """[census] After level `level` is EXTENDED, rebuild level `level+1` over the new lower
    table so its newly available halves become usable.

    Safe because extension only ever APPENDS: existing rows keep their indices and their
    flattened level-1 expansion, so `level+1`'s stored child indices stay valid and its `flat`
    recomputes identically for every row it already had.
    """
    up = level + 1
    if committed.get(up) is None:
        return
    committed[up] = MC.make_table(up, committed[up]["child"], committed[level], s)


def _cert_summary(shadow_cert):
    """[spiral] the rate readout, reduced: for each earnable level, the cycle its shadow
    certificate first fired, the cycle its candidate series became non-empty (`c0` — the
    certificate cannot fire before the vocabulary exists, so cycles-to-cert is only meaningful
    net of it), and the trajectory the silence test was run on."""
    out = {}
    for ell, c in shadow_cert.items():
        out[str(ell)] = {"fired": c["fired"], "fired_era": c.get("fired_era"),
                         "fired_c_in_era": c.get("fired_c_in_era"),
                         "fired_at_active": c.get("fired_active"),
                         "first_candidate_cycle": c["c0"],
                         "cycles_to_cert": (None if (c["fired"] is None or c["c0"] is None)
                                            else int(c["fired"] - c["c0"] + 1)),
                         "ref": c["ref"], "emin": c["emin"], "run_at_end": int(c["run"]),
                         "hist": c["hist"]}
    return out


def audition_macro(generator, x0, roots_np, move, rules_t, canon, depth, v, m, s, rules):
    """The macro's own audition: ONE action on held-out instances of the current era, graded
    by terminal possible-set success. This is the committable content — the quantity the
    unit-LP certificate watches."""
    xf = MC.apply_any(generator, x0, move, rules_t, canon, depth, v, m, s)
    succ, dres = grade(xf.cpu().numpy(), roots_np, rules, s)
    return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean())}


def run_arm(label, base, overrides, shared, cfg, eras, refs, outdir, device):
    import torch
    arm = label
    spec = ARMS[base]
    cfg = {**cfg, **spec.get("cfg", {}), **overrides}
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    base_ms = shared["base_ms"]
    controller = shared["controller"]
    maxl = cfg["max_macro_level"]

    # [spiral] An optional cap on the beam width the declared budget buys, expressed as a
    # REFERENCE BUDGET rather than as a number.
    #
    # It exists for exactly one arm and the reason is a measurement: `enum_live_g722` is meant
    # to be `enum_live` WITHOUT the permanent width collapse at the L2 commit (n 32->48 drops
    # width 2 -> 1 at G=482), and nothing else. But G=722 also buys width **3** at n=32, so
    # raising the budget alone would hand the arm a wider beam through the whole of era 1 too,
    # and the cycles-to-L2-certification comparison the arm exists to de-confound would itself
    # be confounded, in the control's favour.
    #
    # A literal cap would be wrong: `width_cap=2` is a constant of `budget=8` and silently
    # becomes a *handicap* at any other rollout budget (at budget 2 the matched arm runs width
    # 13, and a cap of 2 would make the control four times narrower than the arm it controls —
    # caught by `preflight`, which runs at budget 2). So the cap is stated as the RULE it is:
    # this arm may never run a wider beam than the matched-budget arm runs over the BASE action
    # set; it may only decline to collapse when the action set grows.
    _cap = (fit_width(len(base_ms), cfg["budget"], cfg["width_cap_ref_g"])
            if cfg.get("width_cap_ref_g") else None)

    def cap_w(w):
        return int(w) if not _cap else int(min(int(w), int(_cap)))

    # ---- PER-ARM torch STREAM. ratchet let the global stream run across arms, so an arm's
    #      trajectory depended on the arms before it. Keyed by the TWIN (see `TWIN`/`STREAM`),
    #      so a treated arm and its untreated baseline draw the identical sequence and can only
    #      diverge through the port itself. -------------------------------------------------
    stream_key = TWIN.get(base, base)
    torch.manual_seed(cfg["train_seed"] * 1000 + 7 + STREAM[stream_key])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["train_seed"] * 1000 + 7 + STREAM[stream_key])
    # ---- [assay] THE STREAM-DISPLACEMENT BURN ------------------------------------------ #
    # A controlled draw-and-discard immediately after the per-arm seeding and before anything
    # else touches the arm: same seed, same config, same everything — a different position in
    # the same stream. `|original - displaced|` on the deep-era readouts is then a direct
    # measurement of what stream position ALONE moves at depth 6.
    #
    # BOTH generators are burned because both are consumed, every cycle, by the plant's
    # masking in `finetune_generator`: `torch.randint(1, n_blocks + 1, ())` draws on the CPU
    # generator and `torch.rand(b, n_blocks, device=device)` on the CUDA one, `gen_steps`
    # times per cycle. Burning only one would leave the other aligned and understate the
    # displacement. (Dropout is NOT a consumer here — every net in this substrate is built
    # with `dropout=0.0` — so the burn point was chosen by reading what actually draws rather
    # than by assuming training does.)
    burn = int(cfg.get("stream_burn") or 0)
    burn_rec = {"draws": burn, "cpu": False, "cuda": False,
                "point": "after per-arm torch/cuda seeding, before the generator deepcopy"}
    if burn:
        _ = torch.randint(0, 2 ** 31 - 1, (burn,))
        burn_rec["cpu"] = True
        if torch.cuda.is_available():
            _ = torch.rand(burn, device=device)
            burn_rec["cuda"] = True
        print(f"[burn]   arm={arm} displaced the per-arm stream by {burn} draws "
              f"(cpu={burn_rec['cpu']}, cuda={burn_rec['cuda']})", flush=True)

    # ---- per-arm plant + selector (the plant is a substrate property this round: every arm
    #      carries it, and the value co-adapts against it) ----------------------------------
    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(),
                             lr=cfg["value_lr_online"] or cfg["value_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)

    # ---- PORT 2: the span head, its slots, and the executor (`../span/span.py`'s block,
    #      verbatim). The head is minted off its OWN generator, so its existence costs the
    #      shared stream nothing; its slots start CLOSED, so it cannot fire before parity.
    use_span = bool(spec.get("span"))
    head, slots = None, {}
    slot_events, gate_events = [], []
    span_rng = np.random.default_rng(cfg["seed"] + 20_250_820)     # its OWN stream
    if use_span:
        head = SN.build_head(SN.slot_count(s, depth, maxl), v, cfg["state_dim"],
                             s ** (maxl - 1), cfg["seed"] * 100 + 13, device,
                             hidden_mult=cfg["span_hidden_mult"])
        for p in head.parameters():
            p.requires_grad_(True)
        gopt.add_param_group({"params": list(head.parameters()),
                              "lr": cfg["span_lr"] or cfg["gen_lr"], "weight_decay": 1e-4})
        ex = SN.SpanExecutor(generator, head, slots, cap=cfg["span_buf_cap"],
                             hold_cap=cfg["span_hold_cap"], hold_frac=cfg["span_hold_frac"],
                             per_call=cfg["span_capture"], seed=cfg["seed"] + 5_150_101,
                             v=v, length=shared["length"])
    else:
        ex = SN.PlainExecutor()

    def mint(level, table):
        if not use_span:
            return
        n_nodes = s ** (depth - level)
        for j in range(n_nodes):
            key = SN.slot_key(level, j)
            slots[key] = {"id": SN.slot_index(level, j, s, depth, maxl), "open": False,
                          "move": None, "level": int(level), "node": int(j)}
        slot_events.append({"level": int(level), "n_nodes": int(n_nodes),
                            "n_entries": int(table["child"].shape[0]),
                            "span": int(s ** (level - 1))})
        print(f"[slot]   arm={arm} level={level} nodes={n_nodes} "
              f"entries={table['child'].shape[0]} span={s ** (level - 1)}", flush=True)

    def bind_slots(ms_now):
        """Point every slot at its CURRENT macro move object (the table changes on commit)."""
        for mv in ms_now:
            if mv.get("kind") != "macro":
                continue
            key = SN.slot_key(mv["level"], mv["node"])
            if key in slots:
                slots[key]["move"] = mv

    # ---- vocabulary state ---------------------------------------------------------------
    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    # [census] G-Y: the next-level-yield instrument. UNPRICED and READ-ONLY — it never enters
    # `operative`, never gates anything, never reaches `committed`, and its level is set by
    # `gy_level` rather than by `max_macro_level`, because that constant caps what may be
    # COMMITTED and this miner only OBSERVES. Phase 0 established the stream is absent from
    # `sp_s0` entirely and unreconstructible from its logs, so it has to be instrumented live.
    gy_level = cfg.get("gy_level")
    gy_miner = MC.Miner(int(gy_level), s) if gy_level else None
    gy_nodes = s ** (depth - int(gy_level)) if gy_level else 0
    # [census] the gauge the L2 gate reads: the at-support series per level, per cycle. Kept
    # here rather than read back out of `log["miner"]`, which is only appended at cycle end.
    gauge_hist = {ell: [] for ell in range(2, maxl + 1)}
    # [conductor] THE OBSERVATION PANEL. `census`'s G-Y miner generalised to every level above
    # the base, for a measured reason: the committable miners are gated to `era_level + 1`
    # (mining level 3 in era 1 would hand era 2 a finished vocabulary), so `gauge_hist[3]` is
    # identically ZERO through the whole of era 1 — verified on `as_s0/anchor`, 0 at every one of
    # c1..c48. The one-level-up gauge an era-1 loop needs therefore does not exist in the donor
    # and has to be instrumented. These miners OBSERVE ONLY: they never enter `operative`, never
    # gate anything, never reach `committed`, and never touch the era-gated `miners[...]` the
    # committed tables are built from. Pure numpy counting on features the cycle already parsed,
    # so they draw no RNG and cost no GPU — which is what lets them run in every arm without
    # voiding the replay gate. The level-4 entry is asserted equal to the donor's G-Y miner
    # in-run (`obs_gy_agree`), so the generalisation is checked against the instrument it
    # generalises rather than trusted.
    obs_levels = list(range(3, depth + 1)) if cfg.get("obs_panel") else []
    obs_miners = {ell: MC.Miner(ell, s) for ell in obs_levels}
    obs_hist = {ell: [] for ell in obs_levels}
    obs_gy_agree = []
    # [spiral] PREFLIGHT-ONLY instrument, and it exists because the first preflight could not
    # reach the transplanted commit path at all: a substrate trained for forty steps solves
    # nothing, so the miner never observes, `Miner.build` returns an empty table and the
    # `do_commit` guard turns every commit off — leaving the one code path the preflight is
    # supposed to be checking untested. Seeding the miner with the DGP's own level-l tuples
    # (the same `observe(concatenate([flat] * support))` idiom `selfcheck`'s C-R uses) makes the
    # table non-empty so the commit, the provisional branch and the recert all execute.
    # Guarded by a cfg key that no real config carries, and scientifically meaningless by
    # construction — it hands over the answer, which is exactly why it is preflight-only.
    if cfg.get("preflight_seed_miner"):
        # [census] the seed is PARTIAL, and the rest arrives later. A fully-seeded miner
        # produces a frozen table that already holds everything, so `extend_candidates` returns
        # nothing and the extension op — the thing the preflight most needs to exercise — never
        # runs. Seeding a fraction now and topping up after the commit reproduces the only
        # situation extension exists for: entries that show up AFTER the freeze.
        for ell in range(2, maxl + 1):
            flat = shared["truth"][ell]["flat"]
            k_ = max(1, int(len(flat) * float(cfg.get("preflight_seed_frac", 1.0))))
            miners[ell].observe(np.concatenate([flat[:k_]] * (cfg["mine_support"] + 1)))
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
            mint(ell, committed[ell])
    ms = build_ms(base_ms, committed, s, depth, device)
    bind_slots(ms)
    p_width = cfg["pr_width"]

    # ---- THE PORT ------------------------------------------------------------------------
    #      The head is built inside `isolated_rng`, so it consumes none of the shared stream:
    #      a treated arm is bit-identical to its twin until the filter first switches on.
    prop_k_spec = spec.get("prop_k")
    if overrides.get("prop_k") is not None:
        prop_k_spec = int(overrides["prop_k"])
    offsets, n_slots = PN.slot_layout(depth, s, maxl)
    prop = popt = None
    prng = np.random.default_rng(cfg["seed"] + 4207)
    erng = np.random.default_rng(cfg["seed"] + 8821)
    pbuf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
            "r": torch.zeros(0, dtype=torch.long), "a": torch.zeros(0, dtype=torch.long)}
    move_age = {}                       # slot -> cycles this move has been in the action set
    state = {"filter_on": False}
    if prop_k_spec is not None:
        prop = PN.build_head(cfg["state_dim"], v, n_slots,
                             cfg["seed"] * 7919 + 13, device)
        popt = torch.optim.AdamW(prop.parameters(), lr=cfg["prop_lr"], weight_decay=1e-4)
        for slot in PN.move_slots(ms, offsets):
            move_age[slot] = 0

    def port_spec():
        """The `plan` port for the CURRENT action set, and its effective branching factor.
        Returns `(None, n_moves)` while the filter is off — during the warmup the head trains
        on the beam's trajectories but never gates, so the arm is still its twin exactly."""
        if prop is None or not state["filter_on"]:
            return None, len(ms)
        slots = PN.move_slots(ms, offsets)
        k = len(ms) if prop_k_spec < 0 else int(prop_k_spec)
        forced = tuple(i for i, sl in enumerate(slots)
                       if move_age.get(sl, 0) < cfg["prop_new_cycles"])
        k_eff = min(min(k, len(ms)), len(ms) - len(forced)) + len(forced)
        return ({"prop": prop, "slots": slots, "k": k, "forced": forced,
                 "n_slots": n_slots}, k_eff)

    def avail_mask():
        import torch as _t
        mask = _t.zeros(n_slots, dtype=_t.bool, device=device)
        for sl in PN.move_slots(ms, offsets):
            mask[sl] = True
        return mask

    def operative(ell):
        """The table macros at level `ell` would be built over: the committed one if this arm
        has committed, else the live mined candidate. This is where the ratchet bites — once
        level l-1 is frozen, level l can only ever be built over what was frozen."""
        if ell == 1:
            return MC.base_table(v)
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    def port_for(mset):
        """[spiral] the `plan` port and beam width for an ARBITRARY action set. `port_spec`
        closes over the live `ms`; the recert has to grade a counterfactual one."""
        if prop is None or not state["filter_on"]:
            return None, cap_w(fit_width(len(mset), cfg["budget"], cfg["g_budget"]))
        sl = PN.move_slots(mset, offsets)
        k = len(mset) if prop_k_spec < 0 else int(prop_k_spec)
        forced = tuple(i for i, sl_ in enumerate(sl)
                       if move_age.get(sl_, 0) < cfg["prop_new_cycles"])
        k_eff = min(min(k, len(mset)), len(mset) - len(forced)) + len(forced)
        return ({"prop": prop, "slots": sl, "k": k, "forced": forced, "n_slots": n_slots},
                cap_w(PN.fit_width_k(k_eff, cfg["budget"], cfg["g_budget"])))

    def pol(x, r_np, mset):
        """[spiral] `ear/grader.policy_e`'s measurement — the agent's OWN performance policy on
        `x` with `mset` as the action set, at the declared per-solve grounding budget, scored by
        terminal success — re-expressed through THIS file's `plan` so a routed arm is graded
        through its own port. `ear`'s version enumerates, which at k=4 would price the recert
        as though the port were not there.

        Capture is suspended for the duration: the recert is a GRADING step, and letting its
        beams feed the span head's self-imitation buffer would give a treated arm strictly more
        corridor training data on recert cycles than its twin — a confound, not a treatment."""
        r_np = np.asarray(r_np)
        prt, w = port_for(mset)
        was_cap = getattr(ex, "capture", False)
        ex.capture = False
        b = plan(controller, generator, value, x, torch.from_numpy(r_np), mset,
                 rules_t, canon, depth, v, m, s, budget=cfg["budget"], beam_width=w,
                 device=device, port=prt, ex=ex)
        ex.capture = was_cap
        succ, dres = grade(b["x"].cpu().numpy(), r_np, rules, s)
        return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean()), "w": int(w),
                "counts": b["counts"], "x": b["x"]}

    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    t_cum = 0.0
    events = []
    log = {"cycle": [], "era": [], "level": [], "t_cum": [], "e": [], "succ": [], "dres": [],
           "n_moves": [], "width": [], "g_per_solve": [], "e_practice": [], "vloss": [],
           "gloss": [], "n_solved": [], "n_mined": [], "miner": [], "aud": [], "cert": [],
           "probe": [], "vocab": [], "prop": [], "span": [], "blocks": [],
           "m_per_solve": [], "committed_grade": [], "gy": [], "entry": [],
           # [conductor] the decision trace and the shadow panel, per cycle, in every arm.
           "loop": [], "panel": []}

    # ---- fixed held-out sets, per era ----------------------------------------------------
    meter, shadow = {}, {}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        meter[i] = (r_np, torch.from_numpy(x_np))
        r2, x2 = context_instances(rules, era_ctx(era), cfg["n_score"], s, depth, v, m,
                                   seed=cfg["seed"] + 6100 + 23 * i)
        shadow[i] = (r2, torch.from_numpy(x2).to(device))

    # ---- [audiation] THE INSTRUMENT ------------------------------------------------------
    # Three objects, all sinks. `dec_rec` is handed to the practice beam and to nothing else;
    # `dec_w` shards its output onto the volume on the donor's own checkpoint clock; `snapper`
    # takes the head snapshots at the top of every cycle and runs the recompute assert. All
    # three are `None` when `--log-decisions 0`, in which case this file is `conductor.py`.
    log_dec = bool(cfg.get("log_decisions", True))
    dec_rec = dec_w = snapper = None
    dec_cycles = []
    # The donor tag's own per-cycle log for THIS arm, off the shared volume, for the in-flight
    # replay gate below. Absent (a smoke tag, a config the donor never ran) -> the gate is
    # skipped and says so; present -> it is a hard assert every checkpoint.
    ref_log = replay_ref_log(cfg.get("replay_ref"), arm)
    replay_gate = {"ref": cfg.get("replay_ref"), "arm": arm,
                   "available": ref_log is not None, "n_cycles": 0, "max_abs_delta": None}
    if ref_log is not None:
        print(f"[audi]  arm={arm} in-flight cross-tag replay ARMED against "
              f"{cfg['replay_ref']}/{arm} ({len(ref_log['cycle'])} cycles)", flush=True)
    elif cfg.get("replay_ref"):
        print(f"[audi]  arm={arm} in-flight cross-tag replay SKIPPED: no "
              f"{cfg['replay_ref']}/{arm}/results.json on the volume", flush=True)

    # ---- [E1b] PER-DATUM CREDIT: the treatment's own state --------------------------------
    # `per_datum` is False for every arm but `perdatum`, and every line E1b adds is behind it.
    # `pdrng` is CONSTRUCTED unconditionally (constructing a numpy Generator touches no global
    # stream) and DRAWN only inside `per_datum_pass`, which is what makes the inverse gate
    # exact: with the flag off, this arm's torch and numpy streams are the donor's.
    per_datum = bool(cfg.get("per_datum"))
    pdrng = np.random.default_rng(cfg["seed"] + 31_337)
    rev_rec = rev_w = None
    pd_cycles, pd_probe = [], {}

    def pd_era_probe(era_i_):
        """The era's fixed probe states, encoded once. Identical construction to
        `HeadSnapper.era_probe` (the era's own metering set, first `n_snap_probe`), kept
        separate so the treatment does not depend on the phase-1 instrument being on."""
        if era_i_ not in pd_probe:
            mx, rr = meter[era_i_][1], meter[era_i_][0]
            k_ = min(int(cfg["n_snap_probe"]), int(mx.shape[0]))
            with torch.no_grad():
                z_ = _encode_chunked(controller, mx[:k_].to(device))
            pd_probe[era_i_] = (z_, torch.from_numpy(np.asarray(rr[:k_])).to(device).long(),
                                mx[:k_].to(torch.int8).cpu().numpy())
        return pd_probe[era_i_]

    if per_datum:
        # THE MATCHED BUDGET. Under Adam the per-step parameter displacement is ~lr regardless
        # of batch size or gradient scale, so `n_steps x lr` — not batch count, and not samples
        # seen — is the honest "how far did the learner move this cycle" invariant. It is the
        # one thing held equal to the anchor; batch COMPOSITION is the treatment and cannot be
        # matched without destroying it. Samples seen is reported, not matched.
        n_v_steps = int(cfg["pd_n"]) + int(cfg["pd_maint_v"])
        n_p_steps = int(cfg["pd_prop_nominal"]) + int(cfg["pd_maint_p"])
        v_lr = cfg["pd_value_lr"] or ((cfg["value_lr_online"] or cfg["value_lr"])
                                      * cfg["n_grad"] / n_v_steps)
        p_lr = cfg["pd_prop_lr"] or (cfg["prop_lr"] * cfg["prop_steps"] / n_p_steps)
        for g in vopt.param_groups:
            g["lr"] = float(v_lr)
        if popt is not None:
            for g in popt.param_groups:
                g["lr"] = float(p_lr)
        rev_rec = RevRec(n_slots)
        rev_w = RevWriter(outdir, arm, flush_every=int(cfg.get("dec_flush_every") or 5),
                          compress=bool(cfg.get("dec_compress", True)))
        print(f"[pd]    arm={arm} PER-DATUM CREDIT ON: {cfg['pd_n']} data/cycle, one value "
              f"step each (+{cfg['pd_maint_v']} replay-only maintenance) and one pi step per "
              f"SOLVED datum (+{cfg['pd_maint_p']} history-only maintenance)\n"
              f"[pd]    matched budget n_steps*lr: value "
              f"{cfg['n_grad']}x{cfg['value_lr_online']:.2e} -> {n_v_steps}x{v_lr:.3e}; "
              f"pi {cfg['prop_steps']}x{cfg['prop_lr']:.2e} -> ~{n_p_steps}x{p_lr:.3e} "
              f"(pi's step count follows the SOLVED count, so its per-cycle budget is not "
              f"fixed — stated, not matched)\n"
              f"[pd]    revisions -> {rev_w.dir}; probe {cfg['pd_probe_n']} states, "
              f"reference {cfg['pd_ref_n']} states", flush=True)
    if log_dec:
        dec_rec = DecRec(n_slots)
        dec_w = DecWriter(outdir, arm, cfg["budget"],
                          flush_every=int(cfg.get("dec_flush_every") or 5),
                          compress=bool(cfg.get("dec_compress", True)))
        snapper = HeadSnapper(outdir, arm, n_probe=int(cfg.get("n_snap_probe") or 64),
                              check_every=int(cfg.get("snap_check_every") or 20),
                              plant_every=int(cfg.get("plant_snap_every") or 8))
        print(f"[audi]  arm={arm} decision log ON -> {dec_w.dir} "
              f"(flush every {dec_w.flush_every} cycles); head snapshots -> {snapper.dir} "
              f"(probe n={snapper.n_probe}, file round-trip every {snapper.check_every}, "
              f"plant every {snapper.plant_every})", flush=True)

    # [spiral] the shadow certificate's state, PER LEVEL and per RUN (not per era): a level's
    # certificate can fire in an era later than the one that earns it, and the rate readout
    # wants the cycle it fired, not the era it fired in.
    shadow_cert = {ell: {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [],
                         "run": 0, "fired": None, "c0": None}
                   for ell in range(2, maxl + 1)}

    # ---- [conductor] THE OUTER LOOP ------------------------------------------------------ #
    # Built here, once per arm, from the arm's own spec. A schedule arm gets `SchedulePolicy`,
    # which never acts — under it the era loop below is the donor's `for` loop exactly, and the
    # commit rule is the donor's `delta_prov`. The dead zones are the MEASURED ones and are
    # passed in rather than defaulted; `QuietPolicy` raises if any is missing.
    loop_spec = dict(spec.get("loop") or {})
    if loop_spec.get("read") == "endo":
        loop_spec["read"] = cfg.get("endo_read_key") or "endo"
    loop_floors = {"ledger": cfg["tol_ledger"], "endo": cfg["tol_endo"],
                   "endo_excess": cfg["tol_endo"],
                   "yield_by_level": {3: cfg["tol_yield_l3"], 4: cfg["tol_yield_l4"]}}
    yoke_plan = cfg.get("yoke_plan") or []
    if isinstance(yoke_plan, str):
        yoke_plan = json.loads(yoke_plan)
    loop = PO.build_policy({**loop_spec,
                            "span": cfg["loop_span"], "W": cfg["loop_W"],
                            "burn": cfg["loop_burn"], "alpha": cfg["loop_alpha"]},
                           floors=loop_floors, yoke_plan=yoke_plan)
    driven = loop.kind == "quiet"
    # [conductor] BOTH endo keys carry the price. The driven key is `endo_read_key`, which is
    # `endo_excess` by default, and a price map keyed only on `endo` silently falls through to
    # 0 — which is what happened in `cd_s0`, where `outer_endo` recorded `spend 0g` for 146
    # priced reads. Harmless to the science by construction (a charge reaches only
    # `counts["ground"]` -> `t_cum`, which nothing in the cycle loop reads, so no trajectory
    # moved) and exactly reconstructible after the fact (`charge` records `n_reads` before it
    # prices), but the ledger column is the round's answer to "what did the read cost", so it
    # is fixed here and reconstructed in the reduction for that tag.
    _ep = int(cfg.get("endo_price") or 0)
    ledger = PO.ReadLedger({"endo": _ep, "endo_excess": _ep})
    loop_actions, panel_hist = [], []
    caps = list(cfg.get("era_caps") or ())
    total_cap = int(cfg.get("total_cap") or (sum(caps) if caps else 0))
    # The whole clean probe batch the endo read is taken on, fixed for the run: no sampling, no
    # RNG, so the series' only cycle-to-cycle movement is the plant's own.
    endo_x = shared["probe_clean"][:int(cfg["n_endo"])].to(device)
    print(f"[loop]   arm={arm} policy={loop.kind}"
          + (f" read={loop.read_key} span={cfg['loop_span']} W={cfg['loop_W']} "
             f"burn={cfg['loop_burn']} priced={loop.priced}" if driven else "")
          + (f" plan={len(yoke_plan)} actions" if loop.kind == 'yoke' else ""), flush=True)

    cyc = 0
    era_i = -1
    while era_i + 1 < len(eras):
        era_i += 1
        era = eras[era_i]
        active = era["level"] + 1                     # the macro level being EARNED this era
        cert = {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [], "run": 0,
                "fired": None}
        era_start = cyc + 1
        # [conductor] under the outer loop the era's length is a CAP, not a schedule: `ec` is
        # the most cycles this era may take, and the loop may leave sooner. With the loop off
        # this is the ladder's own count and the `while` below is the donor's `for` exactly.
        if loop.kind == "schedule":
            ec = int(era.get("cycles") or cfg["era_cycles"])
        else:
            ec = int(caps[era_i]) if era_i < len(caps) else \
                int(era.get("cycles") or cfg["era_cycles"])
        era_end = cyc + ec
        loop.acted("era_start", era_start, why=f"era{era_i + 1}")
        print(f"\n----- arm={arm} ERA {era_i + 1}: damage {era['name']} "
              f"(earning level {active}) c{era_start}..{era_end}"
              f"{' (cap)' if loop.kind != 'schedule' else ''} -----", flush=True)

        c_in_era = 0
        while True:
            c_in_era += 1
            cyc += 1
            # [assay] the entry-identity instrument: cleared each cycle, tagged `beam` for the
            # priced practice + metering beams and `probe` for everything unpriced, so beam
            # executions are separable from instrument ones in the reduction.
            _rec_reset()
            _ENTRY_REC["on"] = bool(cfg.get("entry_rec", True))
            _ENTRY_REC["phase"] = "beam"
            # --- [audiation] THE SNAPSHOT, at the TOP of the cycle -------------------------
            #     Taken before ANY of this cycle's blocks run, so the weights it holds are
            #     exactly the ones that plan block (a) below. Nothing in this substrate
            #     updates a head outside blocks (c) and (c'), so `heads_{c+1} - heads_c` is
            #     exactly the revision cycle c's grades produced. The recompute assert fires
            #     here too, and it is a hard assert: a snapshot that does not reproduce the
            #     live readout makes every offline number phase 2 computes wrong.
            if snapper is not None:
                snapper.snap(cyc, era_i, value, prop, generator, controller,
                             meter[era_i][1], meter[era_i][0], device)
            if (cfg.get("preflight_seed_miner")
                    and cyc == int(cfg.get("preflight_seed_topup_cycle", 0))):
                for ell in range(2, maxl + 1):
                    flat = shared["truth"][ell]["flat"]
                    miners[ell].observe(np.concatenate([flat] * (cfg["mine_support"] + 1)))
            counts = {"mat": 0, "ground": 0, "prop": 0}
            ex.reset()
            # THE FILTER SWITCHES ON after `prop_warmup` cycles of the run. Until then the head
            # trains on the beam's trajectories but never gates, so (i) the arm is bit-identical
            # to its twin through the warmup and (ii) the filter never gates from a random init.
            if prop is not None and cyc > cfg["prop_warmup"]:
                state["filter_on"] = True
            port, k_eff = port_spec()
            # PRACTICE EXPLORES, PERFORMANCE DOES NOT — the substrate's own convention
            # (`collect_value_buffer` runs its behaviour policy at explore_eps = 0.3). Without
            # it the filtered beam is a closed loop: a move the head does not propose never
            # appears in a solved trajectory and so never becomes a training target.
            port_pr = port
            if port is not None and cfg["prop_explore"]:
                port_pr = dict(port, explore=int(cfg["prop_explore"]), erng=erng)

            # --- (a) practice: wide closed-loop beam on fresh instances of this era's cell
            r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_pr"], s, depth, v, m,
                                           seed=cfg["seed"] + 100_000 + 1000 * cyc)
            # [audiation] the ONE call site the recorder is attached to.
            if dec_rec is not None:
                dec_rec.open(cyc, {"n_slots": n_slots})
            out = plan(controller, generator, value, torch.from_numpy(x_np),
                       torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                       budget=cfg["budget"], beam_width=p_width, device=device,
                       collect=True, port=port_pr, ex=ex,
                       rec=dec_rec)
            for key in counts:
                counts[key] += out["counts"].get(key, 0)
            tips = out["tips_x"]
            B, W, T = tips.shape
            tips_flat = tips.reshape(B * W, T)
            # [audiation] the donor discards `dres` here; it is the per-tip oracle residual the
            # arm already paid for and is the second half of the terminal join key.
            succ, dres_tips = grade(tips_flat.cpu().numpy(), np.repeat(r_np, W), rules, s)
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in out["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
            ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(out["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            solved = tips_flat[torch.from_numpy(succ > 0.5).to(device)]
            ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
            e_practice = 1.0 - float(ps.mean())
            # --- [audiation] close the cycle's record and shard it ------------------------
            #     The per-cycle CONTEXT lives beside the tables, not inside them: `ms` is a
            #     list of moves whose positional index stops meaning the same thing at a
            #     commit, so `ms_slots` (position -> slot) is what makes `t_mv` / `c_mv`
            #     resolvable, and `avail_slots` is the mask pi's softmax was taken under.
            if dec_rec is not None:
                dec_rec.grades(succ, dres_tips)
                dec_w.add(dec_rec.close(), r_np)
                _sl = PN.move_slots(ms, offsets)
                dec_cycles.append({
                    "cycle": int(cyc), "era": int(era_i + 1), "era_level": int(era["level"]),
                    "era_node": int(era["node"]), "active": int(active),
                    "n_moves": int(len(ms)), "ms_slots": [int(q) for q in _sl],
                    "ms_level": [int(mv_["level"]) for mv_ in ms],
                    "ms_node": [int(mv_["node"]) for mv_ in ms],
                    "avail_slots": sorted(int(q) for q in _sl),
                    "routed": bool(port_pr is not None),
                    "filter_on": bool(state["filter_on"]),
                    "k_eff": int(k_eff), "prop_k": (None if prop_k_spec is None
                                                    else int(prop_k_spec)),
                    "explore": (int(cfg["prop_explore"]) if port_pr is not None else 0),
                    "forced": [int(_sl[i]) for i in (port_pr or {}).get("forced", ())],
                    "n_pr": int(B), "pr_width": int(W), "budget": int(cfg["budget"]),
                    "n_solved": int((succ > 0.5).sum()), "e_practice": float(e_practice),
                    "committed": {str(ell): (None if committed[ell] is None
                                             else int(committed[ell]["child"].shape[0]))
                                  for ell in range(2, maxl + 1)},
                })

            # --- (b) THE VOCABULARY IS MINED from what the agent actually solved, parsed by
            #         the agent's OWN generator. Every level's span containing this era's
            #         damage cell is observed; whether an observation becomes an ENTRY is
            #         decided at build time by the operative lower table (the ratchet).
            #         `mine_from="chosen"` records only the beam's OWN final answer per
            #         instance — the agent's performances, not the leaves of its search tree.
            #         With 16 tips per instance the tree saturates a 14-entry table in one
            #         cycle, which would make the certificate vacuous for a bookkeeping reason
            #         rather than a substrate one (round 1's `cald_s0` trap, one level up).
            mine_src = (out["x"][torch.from_numpy(ps > 0.5).to(device)]
                        if cfg["mine_from"] == "chosen" else solved)
            if cfg["mine_cap"] and mine_src.shape[0] > cfg["mine_cap"]:
                sel = torch.from_numpy(rng.permutation(mine_src.shape[0])[:cfg["mine_cap"]])
                mine_src = mine_src[sel.to(device)]
            #         Levels are mined only up to `era_level + 1`: during era k the agent forms
            #         level-(k+1) chunks out of what it is currently producing. Mining level 3
            #         during era 1 would hand era 2 a finished vocabulary before era 2 begins,
            #         which would erase both the poison test and the unit-LP curve it needs.
            if mine_src.shape[0]:
                # read with the READER, not the generator: `calp2_s0` measured that the
                # generator's block head is unsupervised at visible positions (0.63), which
                # capped the earned level-3 vocabulary's precision at ~0.3.
                pf = MC.parse_features(shared["reader"], mine_src, s=s).cpu().numpy()
                for ell in range(2, min(maxl, era["level"] + 1) + 1):
                    span = s ** (ell - 1)
                    node = (era["node"] * s ** (era["level"] - 1)) // span
                    miners[ell].observe(pf[:, node * span:(node + 1) * span])
                # [census] G-Y observes in EVERY era, not only where `era_level + 1` reaches
                # its level: the readout is the growth DIRECTION of the next level's candidate
                # stream across the whole ladder, so a partial trajectory would not be one.
                if gy_miner is not None:
                    span_y = s ** (gy_miner.level - 1)
                    node_y = (era["node"] * s ** (era["level"] - 1)) // span_y
                    if 0 <= node_y < gy_nodes and (node_y + 1) * span_y <= pf.shape[1]:
                        gy_miner.observe(pf[:, node_y * span_y:(node_y + 1) * span_y])
                # [conductor] the observation panel, on the SAME features and with the SAME
                # span/node arithmetic G-Y uses — so the panel's level-4 entry is the G-Y miner
                # recomputed, which is what `obs_gy_agree` asserts below.
                for ell, mnr in obs_miners.items():
                    span_o = s ** (ell - 1)
                    node_o = (era["node"] * s ** (era["level"] - 1)) // span_o
                    if 0 <= node_o < s ** (depth - ell) and (node_o + 1) * span_o <= pf.shape[1]:
                        mnr.observe(pf[:, node_o * span_o:(node_o + 1) * span_o])
            for ell in range(2, maxl + 1):
                gauge_hist[ell].append(
                    int(miners[ell].state()["n_at_support"][str(cfg["mine_support"])]))
            for ell, mnr in obs_miners.items():
                obs_hist[ell].append(
                    int(mnr.state()["n_at_support"][str(cfg["mine_support"])]))
            if gy_miner is not None and int(gy_miner.level) in obs_miners:
                obs_gy_agree.append(
                    obs_hist[int(gy_miner.level)][-1]
                    == int(gy_miner.state()["n_at_support"][str(cfg["mine_support"])]))

            # --- (c) the plant learns, then the selector ---------------------------------
            sloss, sacc = None, {}
            if use_span:
                gloss, sloss, sacc = finetune_generator_span(
                    generator, head, ex, slots, gopt, solved.cpu(), shared["replay"]["x"],
                    shared["bottom_map"], v=v, s=s, n_blocks=shared["n_blocks"],
                    n_steps=cfg["gen_steps"], batch=cfg["batch_size"],
                    replay_frac=cfg["replay_frac"], device=device, rng=grng,
                    span_rng=span_rng, span_batch=cfg["span_batch"], lam=cfg["span_lam"])
            else:
                gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                           shared["bottom_map"], v=v, s=s,
                                           n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                           batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                           device=device, rng=grng)

            # --- (c2) THE PARITY GATE (span/'s, verbatim). Held-out exact-match against what
            #         the DP would write with the CURRENT trunk, per macro, re-checked every
            #         cycle rather than latched, so a slot the moving plant drifts away from
            #         closes again. Instrument cost only; never priced.
            par = {}
            if use_span and slots:
                par = SN.parity(generator, head, ex, slots, s, v,
                                min_hold=cfg["span_min_hold"])
                for key, cell in par.items():
                    was = slots[key]["open"]
                    now = cell["exact"] is not None and cell["exact"] >= cfg["span_tau"]
                    slots[key]["open"] = now
                    if now != was:
                        gate_events.append({"cycle": cyc, "slot": key, "open": bool(now),
                                            "exact": cell["exact"], "n": cell["n"]})
                        print(f"[gate]   arm={arm} c{cyc} slot {key} "
                              f"{'OPEN' if now else 'CLOSE'} exact={cell['exact']} "
                              f"n={cell['n']}", flush=True)
            # --- (c-pd) [E1b] PER-DATUM CREDIT ------------------------------------------- #
            #     THE TREATMENT, and the whole of it. With `per_datum` off, this block is
            #     skipped, `pd` stays None, and blocks (c)/(c') below are the donor's line for
            #     line — which is what the inverse gate asserts at full scale.
            #
            #     ORDER. The per-datum steps run FIRST, on the cycle's own arriving
            #     trajectories, because "at arrival" is the semantics. The pooled maintenance
            #     pass then runs on HISTORY ONLY (value: the setup replay pool; pi: `pbuf` as
            #     it stood BEFORE this cycle's pairs were appended), so no credit for this
            #     cycle's grades is ever pooled and every per-datum revision has exactly one
            #     cause. The plant is untouched: it feeds no readout, so leaving it pooled
            #     isolates the knob.
            pd = None
            if per_datum:
                rev_rec.open(cyc, {})
                prb_z, prb_r, _ = pd_era_probe(era_i)
                pd = per_datum_pass(
                    out=out, r_np=r_np, succ=succ, dres=dres_tips, ms=ms, offsets=offsets,
                    avail=avail_mask(), controller=controller, value=value, prop=prop,
                    vopt=vopt, popt=popt, budget=cfg["budget"], device=device, cfg=cfg,
                    pdrng=pdrng, rec=rev_rec, cyc=cyc, era_i=era_i, tips_flat=tips_flat,
                    probe_z=prb_z, probe_r=prb_r, n_slots=n_slots,
                    train_on=cfg["prop_train_on"])

            # --- (c) the value's own step ------------------------------------------------ #
            #     [E1b] under `per_datum` this becomes the MAINTENANCE pass: `replay_frac=1.0`,
            #     so it samples ONLY the fixed clean replay pool and carries no credit for the
            #     cycle's grades — it is the anti-drift guard and nothing else. Its step count
            #     is `pd_maint_v`, and it is counted in the matched lr budget below.
            if per_datum:
                vloss_m = value_steps(value, vopt, controller, buf, shared["replay"],
                                      n_steps=cfg["pd_maint_v"], batch=cfg["value_batch"],
                                      replay_frac=1.0, device=device, rng=rng)
                vloss = pd["vloss"] if pd["vloss"] is not None else vloss_m
                pd["vloss_maint"] = vloss_m
            else:
                vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                    n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                    replay_frac=cfg["replay_frac"], device=device, rng=rng)

            # --- (c') THE PORT LEARNS: self-imitation of the beam's own chosen trajectories,
            #          restricted to the survivors that actually SOLVED. Its data comes from
            #          the same beam it gates, so the imitation is on-policy from the cycle the
            #          filter switches on; its RNG is its own, so the shared stream is
            #          undisturbed and the twin gate stays licensed.
            pinfo, fallback = {"k": None}, False
            if prop is not None:
                pairs = prop_pairs(out, torch.from_numpy(r_np), succ,
                                   PN.move_slots(ms, offsets), cfg["budget"],
                                   train_on=cfg["prop_train_on"])
                if pairs is None and cfg["prop_train_on"] == "solved":
                    # nothing solved this cycle: a policy that cannot generate a success cannot
                    # be improved by imitating an empty set, so fall back to the beam's own
                    # survivors (which the value did select) rather than skipping the update
                    # and letting a bad filter freeze itself in.
                    pairs = prop_pairs(out, torch.from_numpy(r_np), succ,
                                       PN.move_slots(ms, offsets), cfg["budget"],
                                       train_on="tips")
                    fallback = True
                if per_datum:
                    # [E1b] MAINTENANCE FIRST, THEN APPEND. `prop_train` here samples `pbuf` as
                    # it stood BEFORE this cycle's pairs arrived, so it replays history only
                    # and carries no credit for this cycle's grades — those were spent, one at
                    # a time, in (c-pd). On cycle 1 `pbuf` is empty and this is a no-op.
                    pinfo = prop_train(prop, popt, controller, pbuf, avail_mask(),
                                       n_steps=cfg["pd_maint_p"], batch=cfg["prop_batch"],
                                       device=device, rng=prng)
                    pinfo["loss_maint"] = pinfo.get("loss")
                    pinfo["loss"] = pd["ploss"] if pd["ploss"] is not None else pinfo.get("loss")
                    pinfo["n_pd_pi_steps"] = pd["n_pi_steps"]
                if pairs is not None:
                    n_cap = cfg["prop_buf_cap"]
                    pbuf["x"] = torch.cat([pbuf["x"], pairs["x"]])[-n_cap:]
                    pbuf["r"] = torch.cat([pbuf["r"], pairs["r"]])[-n_cap:]
                    pbuf["a"] = torch.cat([pbuf["a"], pairs["a"]])[-n_cap:]
                if not per_datum:
                    pinfo = prop_train(prop, popt, controller, pbuf, avail_mask(),
                                       n_steps=cfg["prop_steps"], batch=cfg["prop_batch"],
                                       device=device, rng=prng)
                pinfo.update({"k": (len(ms) if prop_k_spec < 0 else int(prop_k_spec)),
                              "k_eff": int(k_eff), "filter_on": bool(state["filter_on"]),
                              "n_pairs": (0 if pairs is None else int(pairs["x"].shape[0])),
                              "fallback": bool(fallback)})
            # [E1b] close the cycle's revision record. Written on the same clock as the
            # decision shards, so a killed run keeps everything up to its last checkpoint.
            if per_datum:
                pd_cycles.append({
                    "cycle": int(cyc), "era": int(era_i + 1), "n_data": pd["n_data"],
                    "n_value_steps": pd["n_value_steps"], "n_pi_steps": pd["n_pi_steps"],
                    "n_ref": pd["n_ref"], "n_prb": pd["n_prb"],
                    "fallback": bool(pd["fallback"]),
                    "vloss_pd": pd["vloss"], "vloss_maint": pd.get("vloss_maint"),
                    "ploss_pd": pd["ploss"], "ploss_maint": pinfo.get("loss_maint"),
                    "value_lr": float(vopt.param_groups[0]["lr"]),
                    "prop_lr": float(popt.param_groups[0]["lr"]) if popt else None,
                    "sel": [int(q) for q in pd["sel"]], "ref": [int(q) for q in pd["ref"]],
                })
                rev_w.add(rev_rec.close())

            # --- (d) metering under PERFORMANCE conditions (the declared grounding budget) -
            #         The width is refit to the SAME declared budget under this arm's own
            #         effective branching factor: ratchet's matched-pricing idiom with
            #         n_moves -> k. The groundings the port frees are reinvested in width, not
            #         banked; the banked reading is the width ladder logged at every probe.
            mr_np, mx = meter[era_i]
            port, k_eff = port_spec()
            width = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"]) if port is None
                          else PN.fit_width_k(k_eff, cfg["budget"], cfg["g_budget"]))
            b = plan(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                     rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                     beam_width=width, device=device, port=port, ex=ex)
            for key in counts:
                counts[key] += b["counts"].get(key, 0)
            # snapshot the block-level tally NOW: the probes below are unpriced instruments
            # and must not enter the counterfactual ledger any more than the real one.
            blocks_cycle = dict(ex.counts)
            _ENTRY_REC["phase"] = "probe"          # [assay] everything after (d) is unpriced
            msucc, mdres = grade(b["x"].cpu().numpy(), mr_np, rules, s)
            e = 1.0 - float(msucc.mean())
            gps = b["counts"]["ground"] / mx.shape[0]
            meter_cost = {"g": gps, "mat": b["counts"]["mat"] / mx.shape[0],
                          "prop": b["counts"].get("prop", 0) / mx.shape[0],
                          "w": int(width), "k_eff": int(k_eff)}

            # --- (d2) [conductor] THE SHADOW PANEL AND THE DECISION POINT -------------------
            #     Every candidate gauge, in EVERY arm, at every decision point, uncharged
            #     (`endo_yield`'s pattern): one run therefore carries a counterfactual decision
            #     trace for each gauge in each arm, replayable offline through the same rule.
            #     The LEDGER then charges only what THIS arm's policy actually read — ear's
            #     convention, stated in `policy.ReadLedger`.
            #
            #     Placed here, after the metering beam, because that is where the arm's own
            #     error for this cycle exists; every gauge is therefore read at the same instant
            #     of the same cycle, and the three arms differ in which of them is consumed.
            #
            #     READ LEVEL. "One level up" from the level being earned is `active + 1`. Above
            #     level 4 nothing is instrumented (level 5 has 2^20 candidate tuples and no
            #     miner), so in the consumption eras the read CLAMPS to level 4 — the deepest
            #     available next-level gauge. Stated rather than hidden: in eras 3-5 the yield
            #     read is not literally `active + 1`, and `read_level` is logged per cycle so
            #     the reduction can separate the eras where it was.
            read_level = min(active + 1, max(obs_miners) if obs_miners else active + 1)
            read_level = min(read_level, int(cfg.get("gy_level") or 4))
            y_next = obs_hist.get(read_level, [None])[-1] if obs_hist.get(read_level) else None
            y_active = obs_hist.get(active, [None])[-1] if obs_hist.get(active) else None
            e_par = endo_read(generator, endo_x, shared["bottom_map"],
                              parent_span_blocks(era, s, depth),
                              v=v, s=s, n_blocks=shared["n_blocks"])
            e_cell = endo_read(generator, endo_x, shared["bottom_map"],
                               cell_span_blocks(era, s, depth),
                               v=v, s=s, n_blocks=shared["n_blocks"])
            panel = {"cycle": cyc, "era": era_i + 1, "active": int(active),
                     "read_level": int(read_level),
                     # ERROR CONVENTION throughout (lower is better), so every gauge shares one
                     # sign convention and `policy.py` is indifferent to which it is handed.
                     "ledger": float(e), "ledger_level": int(active),
                     "yield": (None if y_next is None else -float(y_next)),
                     "yield_level": int(read_level),
                     "yield_active": (None if y_active is None else -float(y_active)),
                     "endo": e_par, "endo_level": int(era["level"] + 1),
                     "endo_cell": e_cell,
                     # [conductor] THE EXCESS FORM, and why it exists. `endo_yield`'s read was
                     # an EXCESS (`excess_by_level_rand`), and its SPEC says so explicitly: the
                     # additive constant "cancels exactly in the paired contrast". That
                     # cancellation is a property of the DONOR'S TWO-CONDITION block, and this
                     # round's action is absorbing, so there is no second condition and nothing
                     # cancels — a raw NLL read therefore inherits the plant's own global drift.
                     # Measured on `cd_ef/anchor`: the raw parent-span NLL DEGRADES over the run
                     # (mean D -0.0015 against a floor of 0.0108, and the sign gets worse at
                     # every longer horizon: -0.14 -> -0.34 -> -0.53 -> -0.63 -> -0.81 at spans
                     # 1/2/3/4/6, with frac(D>tol) falling to 0.00). So the cancellation has to
                     # be built into the READ instead of into the block: parent minus cell is
                     # one level up RELATIVE to within level, the plant's global drift is common
                     # to both terms and cancels, and it costs nothing extra (both terms come
                     # from forward passes the panel already makes). Same tree arithmetic, same
                     # label-free property, same price. BOTH are logged and either can be
                     # driven, so the counterfactual trace for the other always exists.
                     "endo_excess": (None if (e_par is None or e_cell is None)
                                     else e_par - e_cell),
                     "at_support": {str(k): (vv[-1] if vv else None)
                                    for k, vv in obs_hist.items()},
                     "gauge_hist": {str(k): (vv[-1] if vv else None)
                                    for k, vv in gauge_hist.items()}}
            panel_hist.append(panel)
            # the ledger charges the POLICY's input stream and nothing else. A priced read is
            # folded into `counts["ground"]`, so it surfaces in `t_cum` — a readout that nothing
            # in this loop consumes, which is why pricing cannot change what any arm does.
            #
            # A gauge that does not exist yet is not a decision point: the loop reads nothing,
            # is charged nothing, and its clock does not advance. (In practice every driven read
            # is defined from cycle 1 here; the guard is so a missing gauge fails loudly in the
            # trace instead of silently as a zero.)
            read_val = panel.get(loop.read_key) if driven else 0.0
            if driven and read_val is None:
                linfo = {"rule": "quiet", "cycle": cyc, "read": loop.read_key, "e": None,
                         "quiet": False, "skipped": "read undefined this cycle"}
            else:
                charged = ledger.charge(loop.needs_at(cyc))
                if charged:
                    counts["ground"] += int(charged)
                linfo = loop.step(cyc, panel)
            linfo["era"] = era_i + 1
            linfo["c_in_era"] = c_in_era

            # --- (e) THE SHADOW AUDITION: what the candidate macro would score, every cycle.
            #         `cand` is the earned table; `true` is the DGP's own (an oracle readout,
            #         unpriced); `held` is the frozen committed one (the counterfactual recert
            #         readout — what freezing costs, measured but never acted on).
            sr_np, sx = shadow[era_i]
            aud = {}
            for ell in range(2, maxl + 1):
                span = s ** (ell - 1)
                node = (era["node"] * s ** (era["level"] - 1)) // span
                cell = {}
                # the LIVE candidate, built over whatever this arm's level-(l-1) vocabulary
                # actually is: the frozen one if it committed (the ratchet), the live mined one
                # otherwise, the DGP's own for `given`. Computed for EVERY arm so the
                # compounding question ("does level 3 descend faster given level 2?") is a
                # four-way comparison rather than a within-arm anecdote.
                tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                cell["n_entries"] = int(tbl["child"].shape[0])
                if cell["n_entries"]:
                    mv = MC.to_device(MC.make_macro(ell, node, s, tbl), device)
                    cell["cand"] = audition_macro(generator, sx, sr_np, mv, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    cell.update({f"tab_{k}": val for k, val in
                                 MC.grade_table(tbl, shared["truth"][ell]).items()})
                    if ell == active and committed.get(ell) is None:
                        counts["ground"] += sx.shape[0]      # priced: the agent's own check
                        counts["mat"] += sx.shape[0]
                else:
                    cell["cand"] = None
                mvt = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
                cell["true"] = audition_macro(generator, sx, sr_np, mvt, rules_t, canon,
                                              depth, v, m, s, rules)["e"]
                # MATCHED-SIZE RANDOM CONTROL (oracle instrument, unpriced). `cal_ladder`
                # measured that a k-entry table's audition has huge entry-IDENTITY variance at
                # small k (L3 k=4 spans 0.10-0.97 over uniform draws). If mining recovered a
                # random subset the earned vocabulary would be a lottery; if it beats the
                # matched-size random control, practice mines the USEFUL part of the
                # vocabulary first, which is a claim about practice rather than about size.
                if cell["n_entries"]:
                    full = shared["truth"][ell]
                    n_all = full["child"].shape[0]
                    k = min(cell["n_entries"], n_all)
                    es = []
                    for _ in range(3):
                        keep = np.sort(rng.permutation(n_all)[:k])
                        sub = MC.make_table(ell, full["child"][keep], full["lower"], s)
                        mvr = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                        es.append(audition_macro(generator, sx, sr_np, mvr, rules_t, canon,
                                                 depth, v, m, s, rules)["e"])
                    cell["rand_k"] = float(np.mean(es))
                    cell["rand_k_sd"] = float(np.std(es))
                if committed[ell] is not None and spec["vocab"] == "earned":
                    # the counterfactual recert: what the frozen unit scores now against what
                    # the live vocabulary would score. Measured, never acted on -- committed
                    # macros stay frozen, which is what makes the poison test a real test.
                    mvc = MC.to_device(MC.make_macro(ell, node, s, committed[ell]), device)
                    cell["held"] = audition_macro(generator, sx, sr_np, mvc, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]
                    live = miners[ell].build(MC.base_table(v) if ell == 2
                                             else miners[ell - 1].build(MC.base_table(v),
                                                                        cfg["mine_support"]),
                                             cfg["mine_support"])
                    if live["child"].shape[0]:
                        mvl = MC.to_device(MC.make_macro(ell, node, s, live), device)
                        cell["live"] = audition_macro(generator, sx, sr_np, mvl, rules_t, canon,
                                                      depth, v, m, s, rules)["e"]
                        cell["live_entries"] = int(live["child"].shape[0])
                aud[str(ell)] = cell

            # --- (f) THE UNIT-LP CERTIFICATE: silence on the audition trajectory of the
            #         committable content, with an explicit "it descended first" precondition.
            A = aud.get(str(active), {}).get("cand")   # None until the vocabulary is non-empty
            fire_cert = False
            if A is not None and committed.get(active) is None:
                if cert["ref"] is None:
                    cert["ref"] = A; cert["b"] = A; cert["emin"] = A
                cert["emin"] = min(cert["emin"], A)
                d = cert["b"] - A
                scale = max(cert["ref"] - cert["emin"], cert["b"], 1e-6)
                cert["hist"].append(A); cert["dhist"].append(d)
                we, wd = cert["hist"][-cfg["sil_W"]:], cert["dhist"][-cfg["sil_W"]:]
                quiet = (len(we) >= cfg["sil_W"]
                         and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                         and float(np.std(we)) < cfg["sil_cv"] * scale)
                cert["run"] = cert["run"] + 1 if quiet else 0
                cert["b"] = cert["b"] + cfg["alpha"] * (A - cert["b"])
                dropped = (cert["ref"] - cert["emin"]) >= cfg["lp_min_drop"]
                fire_cert = (cert["run"] >= cfg["sil_hold"] and dropped
                             and c_in_era >= cfg["sil_min_cycle"])

            # --- (f2) [spiral] THE SHADOW CERTIFICATE, per level, every cycle, in EVERY arm.
            #          The donor's certificate above stops being evaluated the moment a level
            #          commits — which is fine when the certificate IS the commit rule, and
            #          useless under `delta_prov`, where a level can commit AT THE BOUNDARY
            #          without the certificate ever having fired. Cycles-to-certification is
            #          this round's primary rate readout, so it has to be observable
            #          independently of what actually caused the commit, in enum arms and
            #          native arms alike.
            #
            #          Run on the LIVE candidate audition for every earnable level (the same
            #          `aud[l]["cand"]` series the donor's certificate reads at the active
            #          level), with the donor's silence test verbatim. Read-only: nothing here
            #          feeds `do_commit`, so the commit path is bit-identical to the donor's.
            for ell in range(2, maxl + 1):
                Ash = aud.get(str(ell), {}).get("cand")
                if Ash is None:
                    continue
                sc_ = shadow_cert[ell]
                if sc_["ref"] is None:
                    sc_["ref"] = Ash; sc_["b"] = Ash; sc_["emin"] = Ash
                    sc_["c0"] = cyc
                sc_["emin"] = min(sc_["emin"], Ash)
                d_ = sc_["b"] - Ash
                scale_ = max(sc_["ref"] - sc_["emin"], sc_["b"], 1e-6)
                sc_["hist"].append(Ash); sc_["dhist"].append(d_)
                we_, wd_ = sc_["hist"][-cfg["sil_W"]:], sc_["dhist"][-cfg["sil_W"]:]
                quiet_ = (len(we_) >= cfg["sil_W"]
                          and abs(float(np.mean(wd_))) < cfg["sil_c"] * scale_
                          and float(np.std(we_)) < cfg["sil_cv"] * scale_)
                sc_["run"] = sc_["run"] + 1 if quiet_ else 0
                sc_["b"] = sc_["b"] + cfg["alpha"] * (Ash - sc_["b"])
                dropped_ = (sc_["ref"] - sc_["emin"]) >= cfg["lp_min_drop"]
                if (sc_["fired"] is None and sc_["run"] >= cfg["sil_hold"] and dropped_
                        and c_in_era >= cfg["sil_min_cycle"]):
                    sc_["fired"] = cyc
                    sc_["fired_era"] = era_i + 1
                    sc_["fired_c_in_era"] = c_in_era
                    sc_["fired_active"] = bool(ell == active)
                    print(f"[cert]   arm={arm} SHADOW certificate L{ell} FIRED c{cyc} "
                          f"(era {era_i + 1} c_in_era {c_in_era}, active level {active}, "
                          f"committed={committed.get(ell) is not None})", flush=True)

            # --- (g) the commit rule ------------------------------------------------------
            do_commit, prov = False, False
            if spec["commit"] and active <= maxl and committed.get(active) is None:
                # [spiral] `at_boundary` and the two policies below are `ear/ear.py`'s, which
                # `recital` then carried: the arc's rescue for a certificate that goes quiet
                # for a substrate reason rather than a learning one. `delta_prov` commits on
                # the certificate if it fires and AT THE ERA BOUNDARY otherwise, marking the
                # boundary case PROVISIONAL — which skips the priced pre-commit audition,
                # because the point of a provisional commit is to pay for grading where the
                # unit is CONSUMED (the live recert, block (g2)) rather than before it has
                # been. This is the policy the SPEC fixes across every spiral arm.
                at_boundary = c_in_era >= ec - cfg["prov_offset"]
                # [census] THE GAUGE, read at the gate level only. G-A on the at-support
                # series: quiet iff the level admitted no more than `gate_theta` new distinct
                # tuples at support over the trailing `gate_W` cycles. Phase 0 R5's constants
                # (W=12, theta=0) are the only grid corner that fired after the certificate in
                # all four `sp_s0` earning arms; that firing cycle was an extrapolation, which
                # is exactly why the rule below is a CONJUNCTION and not a replacement.
                gate_lv = cfg.get("gate_level")
                gate_quiet = None
                if gate_lv and active == int(gate_lv):
                    h = gauge_hist[active]
                    W_, th_ = int(cfg["gate_W"]), int(cfg["gate_theta"])
                    gate_quiet = bool(len(h) > W_ and (h[-1] - h[-1 - W_]) <= th_)
                if spec["commit"] == "delta":
                    do_commit = fire_cert
                elif spec["commit"] == "early":
                    do_commit = c_in_era >= cfg["early_offset"]
                elif spec["commit"] == "late":
                    do_commit = c_in_era >= ec - cfg["late_offset"]
                elif spec["commit"] == "prov":
                    do_commit, prov = at_boundary, True
                elif spec["commit"] == "delta_prov":
                    # [census] at the gate level the licence is the CONJUNCTION; everywhere
                    # else this is exactly the donor's rule, unchanged.
                    lic = bool(fire_cert and (gate_quiet if gate_quiet is not None else True))
                    do_commit = bool(lic or at_boundary)
                    prov = not lic
                elif spec["commit"] == "loop":
                    # [conductor] THE OUTER LOOP OWNS THE COMMIT. Not a conjunction with the
                    # certificate and not a fallback to the era boundary: the loop drives, the
                    # anchor is the comparator arm, and the shadow certificate keeps running
                    # read-only beside both so cycles-to-cert stays observable. (§4.1 reads the
                    # loop as REPLACING the certificate; conjoining them is what `census_gate`
                    # did, and finding 4 is that the conjunction contributed nothing beyond a
                    # cycle number.) A loop commit is NOT provisional — the loop chose it, so it
                    # pays the pre-commit audition exactly as a certified commit does.
                    if loop.kind == "yoke":
                        do_commit = bool(loop.commit_now(cyc))
                    else:
                        do_commit = bool(loop.quiet)
                    prov = False
                elif spec["commit"] == "yoked":
                    # [census] the timing control: at the gate level commit at the cycle
                    # `census_gate` actually committed, by clock and not by signal; at every
                    # other level fall back to the shared policy so the arm differs from the
                    # anchor by exactly one thing.
                    yk = cfg.get("yoke_cycle")
                    if gate_lv and active == int(gate_lv) and yk:
                        do_commit = bool(cyc >= int(yk))
                        prov = bool(cfg.get("yoke_provisional", False))
                    else:
                        do_commit = bool(fire_cert or at_boundary)
                        prov = not fire_cert
            surg_rec = None
            loop_tried_commit = bool(do_commit and spec["commit"] == "loop")
            if do_commit:
                tbl = miners[active].build(operative(active - 1), cfg["mine_support"])
                if tbl["child"].shape[0] == 0:
                    do_commit = False
            # [conductor] a loop commit the empty-table guard cancelled still COUNTS AS AN
            # ACTION for the loop's clock: the rule licensed one, the substrate refused it, and
            # the rule must re-arm before licensing another. Without this the loop would fire
            # again on the very next cycle from the same quiet reading.
            if loop_tried_commit and not do_commit:
                loop.acted(PO.COMMIT, cyc, why="empty_table")
                loop_actions.append({"cycle": cyc, "era": era_i + 1, "kind": PO.COMMIT,
                                     "level": int(active), "why": "quiet",
                                     "cancelled": "empty_table",
                                     "V": linfo.get("V"), "v_tol": linfo.get("v_tol")})
                print(f"[loop]   arm={arm} c{cyc} L{active} COMMIT licensed but the mined "
                      f"table is empty — cancelled, loop re-armed", flush=True)
            # ---- [assay] ORACLE SURGERY ------------------------------------------------- #
            # Placed HERE, before the commit body, for two reasons. First, the donor's
            # empty-table cancellation already lives at this point, so a surgery that empties a
            # table (`strip`'s L3 mined table is only a few true entries, and an empty table has
            # no entry axis for the DP's argmax) cancels the commit through the existing path
            # instead of installing an unservable macro. Second, the priced pre-commit audition
            # in the body below then grades WHAT IS ACTUALLY COMMITTED, which is what that
            # audition is for, and keeps its meaning identical across arms.
            # The shadow certificate is untouched, so what the certificate would have done with
            # the mined table is still recorded beside what the surgery installed.
            if do_commit and cfg.get("surgery"):
                lower_x = MC.base_table(v) if active == 2 else committed[active - 1]
                mined_grade = MC.grade_table(tbl, shared["truth"][active])
                tbl, surg_rec = apply_surgery(
                    cfg["surgery"], active, tbl, lower_x, shared["truth"][active],
                    shared.get("junk_pool") or {}, s, truth_full=shared["truth"])
                surg_rec.update({"arm": arm, "cycle": cyc, "era": era_i + 1,
                                 "mined_recall": mined_grade["recall"],
                                 "mined_precision": mined_grade["precision"]})
                if int(tbl["child"].shape[0]) == 0:
                    surg_rec["cancelled_empty"] = True
                    do_commit = False
                    print(f"[surgery] arm={arm} c{cyc} L{active} mode={cfg['surgery']} "
                          f"produced an EMPTY table — commit cancelled", flush=True)
                else:
                    print(f"[surgery] arm={arm} c{cyc} L{active} mode={cfg['surgery']} "
                          f"own={surg_rec['n_own']}({surg_rec['n_own_true']} true) -> "
                          f"{surg_rec['n_after']} added={surg_rec['n_added']}"
                          f"({surg_rec['n_added_true']} true) "
                          f"removed={surg_rec['n_removed']} K={surg_rec.get('K')} "
                          f"unbuildable={surg_rec['n_unbuildable']} "
                          f"recall={surg_rec.get('tab_recall')} "
                          f"prec={surg_rec.get('tab_precision')}", flush=True)
                events.append({"kind": "surgery", **surg_rec})
            if do_commit:
                # a fresh held-out audition, PRICED — the agent pays to verify what it commits.
                # [spiral] a PROVISIONAL commit deliberately skips it (ear's rescue): the unit
                # is graded where it is consumed, by the live recert, instead of before it has
                # been consumed at all.
                held = None
                if not prov:
                    cr_np, cx_np = context_instances(rules, era_ctx(era), cfg["n_score"], s,
                                                     depth, v, m,
                                                     seed=cfg["seed"] + 700_000 + 1000 * cyc)
                    cx = torch.from_numpy(cx_np).to(device)
                    span = s ** (active - 1)
                    node = (era["node"] * s ** (era["level"] - 1)) // span
                    mv = MC.to_device(MC.make_macro(active, node, s, tbl), device)
                    held = audition_macro(generator, cx, cr_np, mv, rules_t, canon, depth,
                                          v, m, s, rules)
                    counts["ground"] += cx.shape[0]; counts["mat"] += cx.shape[0]
                # [spiral] THE AUDITION CALIBRATION INSTRUMENT (the grader wall). Under
                # `delta_prov` a boundary commit deliberately buys no audition, so `audition`
                # is None exactly where the calibration question is most interesting. This is
                # the same one-action held-out audition of the committed table, computed on the
                # era's fixed shadow set, ORACLE and UNPRICED — it is never read by the loop,
                # only logged, so it cannot change what the agent does. It is what makes
                # "what did the grader think this unit was worth, and what was it worth?"
                # answerable at both levels for every commit, priced or provisional.
                span_o = s ** (active - 1)
                node_o = (era["node"] * s ** (era["level"] - 1)) // span_o
                mv_o = MC.to_device(MC.make_macro(active, node_o, s, tbl), device)
                aud_oracle = audition_macro(generator, shadow[era_i][1], shadow[era_i][0],
                                            mv_o, rules_t, canon, depth, v, m, s, rules)["e"]
                w_before, ms_len_before = int(width), len(ms)
                committed[active] = tbl
                ms = build_ms(base_ms, committed, s, depth, device)
                mint(active, tbl)              # the corridor becomes the head's responsibility
                bind_slots(ms)
                if prop is not None:
                    # the head has never had a training example on the new slots. Their output
                    # rows are still at their zero init, so they enter the ranking neutrally
                    # (above whatever the head learned to reject, below what it learned to
                    # accept) — and for `prop_new_cycles` cycles they are expanded regardless,
                    # priced at their true cost, so an untrained logit cannot lock them out.
                    for slot in PN.move_slots(ms, offsets):
                        move_age.setdefault(slot, -1)
                cert["fired"] = cyc
                # [spiral] THE PRICING READOUT AT EVERY COMMIT, both levels: what the declared
                # budget buys before and after the action set grows, and the `prop_new_cycles`
                # forced-expansion window's own transient (k_eff jumps by the number of new
                # nodes, so the routed width collapses for exactly that many cycles). Phase A
                # measured this at L2 — 18 -> 3 -> 18 under routing against a PERMANENT 2 -> 1
                # under enumeration — and it has to be readable at L3 too.
                port_after, k_after = port_spec()
                w_after = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                                if port_after is None
                                else PN.fit_width_k(k_after, cfg["budget"], cfg["g_budget"]))
                ev = dict(kind="commit", arm=arm, era=era_i + 1, level=active, cycle=cyc,
                          c_in_era=c_in_era, t_cum=t_cum + priced(counts, cfg),
                          provisional=bool(prov),
                          audition=(None if held is None else held["e"]),
                          aud_oracle=aud_oracle, shadow=A, e_task=e,
                          cert_fired=shadow_cert[active]["fired"],
                          surgery=cfg.get("surgery"),
                          surgery_n_after=(surg_rec or {}).get("n_after"),
                          surgery_recall=(surg_rec or {}).get("tab_recall"),
                          surgery_precision=(surg_rec or {}).get("tab_precision"),
                          gate_level=cfg.get("gate_level"), gate_quiet=gate_quiet,
                          yoke_cycle=cfg.get("yoke_cycle"),
                          gauge_at_sup=(gauge_hist[active][-1] if gauge_hist[active] else None),
                          n_entries=int(tbl["child"].shape[0]),
                          n_moves_before=int(ms_len_before),
                          n_moves_after=len(ms), sil_run=int(cert["run"]),
                          g_budget=cfg["g_budget"],
                          width_before=w_before, width_after=int(w_after),
                          k_eff_before=int(k_eff), k_eff_after=int(k_after),
                          g_per_solve_before=float(gps),
                          # [conductor] what the loop saw at the instant it acted, on the event
                          # itself, so a commit is auditable against its own decision statistic.
                          driver=("loop" if spec["commit"] == "loop" else spec["commit"]),
                          loop_kind=loop.kind, loop_read=loop.read_key,
                          loop_V=linfo.get("V"), loop_v_tol=linfo.get("v_tol"),
                          loop_v_mult=linfo.get("v_mult"), loop_D=linfo.get("D"),
                          loop_read_level=linfo.get("read_level"),
                          **{f"tab_{k}": val for k, val in
                             MC.grade_table(tbl, shared["truth"][active]).items()})
                events.append(ev)
                if spec["commit"] == "loop":
                    loop.acted(PO.COMMIT, cyc)
                    loop_actions.append(
                        {"cycle": cyc, "era": era_i + 1, "kind": PO.COMMIT,
                         "level": int(active),
                         "why": ("clock" if loop.kind == "yoke" else "quiet"),
                         "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                         "v_mult": linfo.get("v_mult"), "c_in_era": c_in_era})
                print(f"[commit] arm={arm} era{era_i+1} level={active} c{cyc} "
                      f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                      f"recall={ev['tab_recall']:.3f} "
                      f"prec={ev['tab_precision']} audition={ev['audition']} "
                      f"aud_oracle={aud_oracle:.4f} shadow={A} cert={ev['cert_fired']} "
                      f"moves={ms_len_before}->{len(ms)} "
                      f"k_eff={ev['k_eff_before']}->{ev['k_eff_after']} "
                      f"width={w_before}->{ev['width_after']} G={cfg['g_budget']}", flush=True)

            # --- (g2) [spiral] THE LIVE RECERT (`ear/ear.py`'s, transplanted). Grade the
            #          FROZEN committed unit where it is CONSUMED — this era's damage cell is
            #          the consumption distribution for the macro level committed one era ago —
            #          against what the LIVE mined table would score there. Priced, because it
            #          is the agent's own feedback and not an oracle readout; measured but
            #          never acted on, so the ratchet still bites and the poison test is still
            #          a test. Only spiral arms carry it (`spec["recert"]`), so a donor arm's
            #          path through this block is a single falsy lookup.
            if (spec.get("recert") and era["level"] >= 2
                    and committed.get(era["level"]) is not None
                    and c_in_era % cfg["recert_every"] == 0):
                rc_level = era["level"]
                live = miners[rc_level].build(operative(rc_level - 1), cfg["mine_support"])
                xr = shadow[era_i][1][:cfg["n_aud"]]
                rr_ = shadow[era_i][0][:cfg["n_aud"]]
                pf_ = pol(xr, rr_, ms)
                counts["ground"] += pf_["counts"]["ground"]
                counts["mat"] += pf_["counts"]["mat"]
                rec = {"kind": "recert", "arm": arm, "era": era_i + 1, "level": rc_level,
                       "cycle": cyc, "c_in_era": c_in_era, "e_frozen": pf_["e"],
                       "n_frozen": int(committed[rc_level]["child"].shape[0]),
                       "n_live": int(live["child"].shape[0]), "swapped": False}
                if live["child"].shape[0]:
                    ms_live = build_ms(base_ms, {**committed, rc_level: live}, s, depth, device)
                    pl_ = pol(xr, rr_, ms_live)
                    counts["ground"] += pl_["counts"]["ground"]
                    counts["mat"] += pl_["counts"]["mat"]
                    rec["e_live"] = pl_["e"]
                    # an identical `e` from two different tables is only informative if the
                    # answers differ, so the discordance is logged rather than assumed.
                    # NOTE for whoever reads a 0 here on a span arm: once a slot is OPEN the
                    # head serves that macro call and the table is not consulted at execution
                    # time at all, so swapping the table CANNOT change the answer. That is a
                    # property of the corridor, not a bug — and it is the cleanest available
                    # statement of what "the table survives as the address book" costs.
                    rec["n_diff"] = int((pf_["x"] != pl_["x"]).any(1).sum().item())
                    rec.update({f"live_{k}": val for k, val in
                                MC.grade_table(live, shared["truth"][rc_level]).items()})
                events.append(rec)
                print(f"[recert] arm={arm} era{era_i+1} level={rc_level} c{cyc} "
                      f"e_frozen={rec['e_frozen']:.4f} e_live={rec.get('e_live')} "
                      f"n_frozen={rec['n_frozen']} n_live={rec['n_live']} "
                      f"n_diff={rec.get('n_diff')}", flush=True)

            # --- (g3) [census] COMMIT-THEN-EXTEND ------------------------------------------
            #     The recert loop widened to EVERY COMMITTED LEVEL in every era, and given the
            #     power to ADD entries rather than only to offer a whole-table swap. Scoped to
            #     arms carrying `extend`, so the anchor and the gate arms take the donor's path
            #     untouched and each differs from the anchor by exactly one thing.
            #
            #     SELECTION, NOT APPEND (the SPEC's requirement): each candidate is auditioned
            #     on fresh held-out instances of this era's own cell, through the same
            #     `audition_macro` a commit pays for, and admitted only if it does not raise
            #     the table's audition error. Priced exactly as the commit audition is.
            #
            #     Extension does NOT grow the action set — the macro nodes are unchanged, only
            #     the table behind them — so unlike a commit it carries no width transient and
            #     no `prop_new_cycles` forced window. That is the "~zero timing price" claim,
            #     and the pricing readout is logged either way so it is checked, not assumed.
            if spec.get("extend") and c_in_era % cfg["recert_every"] == 0:
                for lv_x in range(2, maxl + 1):
                    if committed.get(lv_x) is None:
                        continue
                    lower_x = MC.base_table(v) if lv_x == 2 else committed[lv_x - 1]
                    cands = extend_candidates(miners[lv_x], lower_x, cfg["mine_support"],
                                              committed[lv_x])
                    span_x = s ** (lv_x - 1)
                    node_x = (era["node"] * s ** (era["level"] - 1)) // span_x
                    ev_x = {"kind": "extend", "arm": arm, "era": era_i + 1, "level": lv_x,
                            "cycle": cyc, "c_in_era": c_in_era,
                            "n_frozen": int(committed[lv_x]["child"].shape[0]),
                            "n_candidates": len(cands), "n_admitted": 0, "admitted": [],
                            "rejected": 0, "n_auditions": 0}
                    if not cands or not (0 <= node_x < s ** (depth - lv_x)):
                        events.append(ev_x)
                        continue
                    xr_np, xx_np = context_instances(
                        rules, era_ctx(era), cfg["n_aud"], s, depth, v, m,
                        seed=cfg["seed"] + 900_000 + 1000 * cyc + lv_x)
                    xx = torch.from_numpy(xx_np).to(device)

                    def _aud(child_rows, _lv=lv_x, _lo=lower_x, _nd=node_x, _xx=xx,
                             _xr=xr_np):
                        tb = MC.make_table(_lv, np.asarray(child_rows, np.int64), _lo, s)
                        mv_ = MC.to_device(MC.make_macro(_lv, _nd, s, tb), device)
                        return tb, audition_macro(generator, _xx, _xr, mv_, rules_t, canon,
                                                  depth, v, m, s, rules)["e"]

                    base_child = [list(map(int, r)) for r in committed[lv_x]["child"]]
                    _, best_e = _aud(base_child)
                    ev_x["n_auditions"] += 1
                    ev_x["e_before"] = float(best_e)
                    kept = list(base_child)
                    for cand in cands[:int(cfg["extend_cap"])]:
                        trial = kept + [list(map(int, cand["child"]))]
                        _, e_x = _aud(trial)
                        ev_x["n_auditions"] += 1
                        if e_x <= best_e + cfg["extend_tol"]:
                            kept, best_e = trial, e_x
                            ev_x["n_admitted"] += 1
                            ev_x["admitted"].append({"key": list(cand["key"]),
                                                     "count": cand["count"],
                                                     "e_after": float(e_x)})
                        else:
                            ev_x["rejected"] += 1
                    # the agent PAYS for every audition it ran, at the commit audition's rate
                    counts["ground"] += ev_x["n_auditions"] * xx.shape[0]
                    counts["mat"] += ev_x["n_auditions"] * xx.shape[0]
                    ev_x["e_after"] = float(best_e)
                    if ev_x["n_admitted"]:
                        committed[lv_x] = MC.make_table(lv_x, np.asarray(kept, np.int64),
                                                        lower_x, s)
                        refresh_upper(committed, lv_x, s)
                        ms = build_ms(base_ms, committed, s, depth, device)
                        bind_slots(ms)
                        ev_x["n_after"] = int(committed[lv_x]["child"].shape[0])
                        ev_x.update({f"tab_{k}": val for k, val in
                                     MC.grade_table(committed[lv_x],
                                                    shared["truth"][lv_x]).items()})
                    events.append(ev_x)
                    if ev_x["n_candidates"]:
                        print(f"[extend] arm={arm} c{cyc} L{lv_x} cand={ev_x['n_candidates']} "
                              f"admitted={ev_x['n_admitted']} rej={ev_x['rejected']} "
                              f"n {ev_x['n_frozen']}->{ev_x.get('n_after', ev_x['n_frozen'])} "
                              f"e {ev_x['e_before']:.4f}->{ev_x['e_after']:.4f} "
                              f"recall={ev_x.get('tab_recall')} "
                              f"prec={ev_x.get('tab_precision')}", flush=True)

            # --- (g4) [conductor] THE OUTER LOOP'S ERA-ADVANCE DECISION --------------------
            #     The second of the round's two actions, and the one that has no reversible
            #     shadow at all: the era sequence is the world, and there is no way back up the
            #     damage ladder. Precedence is fixed and stated — a quiet reading COMMITS the
            #     active level if there is one to commit, and ADVANCES otherwise; committing
            #     re-arms the loop, so one quiet reading can never fire both.
            #
            #     `era_last` is computed HERE, before the probe block, because the donor's probe
            #     fires on the era's final cycle and under the loop that cycle is not known in
            #     advance. With the loop off `era_last == (c_in_era >= ec) == (c_in_era == ec)`,
            #     which is the donor's trigger exactly.
            adv, adv_why = False, None
            if spec["commit"] == "loop" or loop.kind == "yoke":
                if loop.kind == "yoke":
                    adv, adv_why = bool(loop.advance_now(cyc)), "clock"
                elif loop.quiet:
                    adv, adv_why = True, "quiet"
            hit_cap = bool(c_in_era >= ec)
            if hit_cap and not adv and loop.kind != "schedule":
                adv, adv_why = True, "cap"
            era_last = bool(adv or hit_cap)
            if adv and loop.kind != "schedule":
                loop.acted(PO.ADVANCE, cyc, why=adv_why)
                loop_actions.append(
                    {"cycle": cyc, "era": era_i + 1, "kind": PO.ADVANCE, "level": None,
                     "why": adv_why, "c_in_era": c_in_era,
                     "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                     "v_mult": linfo.get("v_mult")})
                events.append({"kind": "advance", "arm": arm, "era": era_i + 1, "cycle": cyc,
                               "c_in_era": c_in_era, "why": adv_why, "cap": int(ec),
                               "driver": loop.kind, "read": loop.read_key,
                               "read_level": linfo.get("read_level"),
                               "V": linfo.get("V"), "v_tol": linfo.get("v_tol"),
                               "v_mult": linfo.get("v_mult"),
                               "committed": {str(q): (None if committed[q] is None
                                                      else int(committed[q]["child"].shape[0]))
                                             for q in range(2, maxl + 1)}})
                print(f"[advance] arm={arm} c{cyc} leaving era {era_i + 1} at c_in_era "
                      f"{c_in_era}/{ec} why={adv_why} V={linfo.get('V')} "
                      f"tol={linfo.get('v_tol')}", flush=True)

            # --- (h) probes: the width ladder + all eras' metering sets + the plant guard.
            #         Instruments, never priced.
            probe = None
            if c_in_era == 1 or cyc % cfg["probe_every"] == 0 or era_last:
                # the action set may have grown in (g) this very cycle, so the port is rebuilt
                # here rather than reused: a stale `slots` list would silently make the probe
                # blind to the moves just committed.
                pport, pk = port_spec()
                probe = {"cycle": cyc, "era": era_i + 1, "ladder": {}, "all_eras": {},
                         "k_eff": int(pk), "n_moves": len(ms)}
                for w in cfg["probe_widths"]:
                    bb = plan(controller, generator, value, mx, torch.from_numpy(mr_np),
                              ms, rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=w, device=device, port=pport, ex=ex)
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["ladder"][str(w)] = {
                        "e": 1.0 - float(sc.mean()),
                        "g": bb["counts"]["ground"] / mx.shape[0],
                        "mat": bb["counts"]["mat"] / mx.shape[0],
                        "prop": bb["counts"].get("prop", 0) / mx.shape[0]}
                for j in range(len(eras)):
                    jr, jx = meter[j]
                    wj = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                               if pport is None
                               else PN.fit_width_k(pk, cfg["budget"], cfg["g_budget"]))
                    bb = plan(controller, generator, value, jx, torch.from_numpy(jr), ms,
                              rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=wj, device=device, port=pport, ex=ex)
                    sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                    probe["all_eras"][str(j)] = 1.0 - float(sc.mean())
                probe["plant"] = plant_probe(generator, shared, cfg, device)
                # THE SPAN-HEAD-BYPASSED TWIN of the competence readout (span/'s): the same
                # metering beam with the head switched off, so what the port's FIRING costs or
                # buys is separable from what its gradient did to the trunk.
                if use_span and any(sl["open"] for sl in slots.values()):
                    wn = cap_w(fit_width(len(ms), cfg["budget"], cfg["g_budget"])
                               if pport is None
                               else PN.fit_width_k(pk, cfg["budget"], cfg["g_budget"]))
                    ex.fire = False; ex.capture = False
                    bb = plan(controller, generator, value, mx, torch.from_numpy(mr_np), ms,
                              rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                              beam_width=wn, device=device, port=pport, ex=ex)
                    ex.fire = True; ex.capture = True
                    sc, _ = grade(bb["x"].cpu().numpy(), mr_np, rules, s)
                    probe["plant"]["e_nospan"] = 1.0 - float(sc.mean())
                # THE CAN'T-DECOMPOSE SIGNATURE, on the era's own metering roots: where pi puts
                # its mass, and — on the states where it proposes a MACRO — how much mass it
                # keeps on that macro's own primitive decomposition. An expert cannot decompose
                # its chunks. Unpriced instrument.
                if prop is not None:
                    with torch.no_grad():
                        zp = _encode_chunked(controller, mx.to(device))
                    rp = torch.from_numpy(mr_np).to(device)
                    am = avail_mask()
                    probe["pi"] = PN.decompose_probe(prop, zp, rp, ms, offsets, s, maxl, am)
                    bm = PN.base_mass_probe(prop, zp, rp, offsets, am)
                    probe["pi"].update({"top1": bm["top1"], "entropy": bm["entropy"]})
                log["probe"].append(probe)

            t_cum += priced(counts, cfg)
            log["cycle"].append(cyc); log["era"].append(era_i + 1); log["level"].append(era["level"])
            log["t_cum"].append(t_cum); log["e"].append(e); log["succ"].append(float(msucc.mean()))
            log["dres"].append(float(mdres.mean())); log["n_moves"].append(len(ms))
            log["width"].append(width); log["g_per_solve"].append(gps)
            log["e_practice"].append(e_practice); log["vloss"].append(vloss)
            log["gloss"].append(gloss); log["n_solved"].append(int(solved.shape[0]))
            log["n_mined"].append(int(mine_src.shape[0]))
            log["miner"].append({str(ell): miners[ell].state() for ell in range(2, maxl + 1)})
            log["aud"].append(aud)
            log["cert"].append({"A": A, "b": cert["b"], "run": int(cert["run"]),
                                "ref": cert["ref"], "emin": cert["emin"]})
            # [spiral] the COMMITTED table's own recall/precision per cycle, beside its size.
            # `aud[l]["tab_*"]` grades the LIVE candidate; once a level is frozen the two are
            # different tables and the round's crank-health readout wants the frozen one — it
            # is what the policy is actually calling. `grade_table` is a set comparison against
            # the DGP's table: an oracle readout, unpriced, never consumed.
            log["vocab"].append({str(ell): (None if committed[ell] is None
                                            else int(committed[ell]["child"].shape[0]))
                                 for ell in range(2, maxl + 1)})
            # [assay] the entry-identity readout for this cycle, with the truth mask of each
            # committed table alongside so the offline keying is a lookup, not a re-derivation.
            _ENTRY_REC["on"] = False
            ent = _rec_take()
            tmask = {}
            for ell in range(2, maxl + 1):
                if committed.get(ell) is None:
                    continue
                tset = {tuple(int(x) for x in r) for r in shared["truth"][ell]["flat"]}
                tmask[str(ell)] = [int(tuple(int(x) for x in r) in tset)
                                   for r in committed[ell]["flat"]]
            log["entry"].append({"hist": ent, "true_mask": tmask})
            log["gy"].append(gy_miner.state() if gy_miner is not None else None)
            log["committed_grade"].append(
                {str(ell): (None if committed[ell] is None
                            else MC.grade_table(committed[ell], shared["truth"][ell]))
                 for ell in range(2, maxl + 1)})
            pinfo["n_prop"] = int(counts.get("prop", 0))
            pinfo["n_ground"] = int(counts["ground"]); pinfo["n_mat"] = int(counts["mat"])
            pinfo["meter"] = meter_cost          # the PRICED beam's own cost per instance
            log["prop"].append(pinfo)
            log["m_per_solve"].append(b["counts"]["mat"] / mx.shape[0])
            log["blocks"].append(blocks_cycle)
            tb, hb = ex.sizes() if use_span else ({}, {})
            log["span"].append({
                "parity": {k: cell["exact"] for k, cell in par.items()},
                "parity_block": {k: cell["block"] for k, cell in par.items()},
                "n_hold": {k: cell["n"] for k, cell in par.items()},
                "open": {k: bool(sl["open"]) for k, sl in slots.items()},
                "n_open": int(sum(1 for sl in slots.values() if sl["open"])),
                "buf": tb, "hold": hb, "sloss": sloss, "sacc": sacc})
            for slot in list(move_age):
                move_age[slot] += 1
            a_act = aud.get(str(active), {})
            print(f"[c{cyc:3d}] arm={arm:20s} era{era_i+1} t={t_cum:10.0f} e={e:.4f} "
                  f"w={width} nm={len(ms)} k={pinfo.get('k')}/{pinfo.get('k_eff')} "
                  f"solved={solved.shape[0]:4d} "
                  f"A{active}={a_act.get('cand')} true={a_act.get('true')} "
                  f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
                  f"pacc={pinfo.get('acc')} sil={cert['run']}", flush=True)
            # [conductor] the loop's own per-cycle record: what it read, what the statistic was,
            # whether it was armed, whether it was quiet. This IS the decision trace the
            # reduction summarises, and it is written for every arm including the schedule one
            # (where it is empty by construction).
            log["loop"].append(linfo)
            log["panel"].append(panel)
            # --- [audiation] THE CROSS-TAG REPLAY GATE, IN FLIGHT --------------------------
            #     The fidelity claim of this whole node is "the instrument moves nothing", and
            #     the statement that carries it is the full-scale cross-tag replay against
            #     `cd_s0/anchor`. That is normally checked in the reduction, i.e. after a whole
            #     GPU-hour. The reference log is on the SAME volume under the donor's prefix,
            #     so it can be checked every checkpoint instead, and a divergence costs two
            #     minutes rather than an hour. Hard assert: there is no reading of a
            #     divergence here that is not a bug.
            if ref_log is not None and cyc % cfg["checkpoint_every"] == 0:
                _d, _per, _n = replay_delta(ref_log, log)
                assert _d == 0.0, (
                    f"[audiation] cross-tag replay BROKE at c{cyc}: {cfg['replay_ref']}/{arm} "
                    f"vs this arm over c1-c{_n}, max|delta|={_d:.3e}; per-series "
                    f"{ {k: q for k, q in _per.items() if q} } — the instrument is not inert")
                replay_gate.update({"n_cycles": int(_n), "max_abs_delta": _d,
                                    "per_series": _per})
                if cyc % (10 * cfg["checkpoint_every"]) == 0:
                    print(f"[audi]  arm={arm} cross-tag replay clean through c{_n} "
                          f"(max|delta| {_d:.3e} vs {cfg['replay_ref']})", flush=True)
            if cyc % cfg["checkpoint_every"] == 0:
                write_results(outdir, arm, cfg, eras, refs, log, events, complete=False,
                              extra={"prop_k": prop_k_spec, "span_mode": use_span,
                                     "twin": TWIN.get(base), "slot_events": slot_events,
                                     "gate_events": gate_events,
                                     "shadow_cert": _cert_summary(shadow_cert),
                                     "stream_burn": burn_rec,
                                     "loop_actions": loop_actions,
                                     "ledger": ledger.state()})
            if era_last:
                break
            if total_cap and cyc >= total_cap:
                print(f"[loop]   arm={arm} TOTAL CAP {total_cap} reached inside era "
                      f"{era_i + 1}", flush=True)
                break
        if total_cap and cyc >= total_cap:
            break

    # ---- [audiation] CLOSE THE INSTRUMENT ------------------------------------------------
    # The TERMINAL snapshot, taken here rather than after the ablation battery: the battery is
    # an unpriced end-of-run probe that runs beams under `no_grad` and takes no gradient step,
    # so the weights are the same either way, but "the heads as the last cycle left them" is
    # the semantics `schema.md` states and the snapshot should be taken where that is true by
    # inspection. Indexed `cyc + 1`, so `heads_c{c+1} - heads_c{c}` is the revision from cycle
    # c's grades for EVERY c including the last.
    if ref_log is not None:
        _d, _per, _n = replay_delta(ref_log, log)
        replay_gate.update({"n_cycles": int(_n), "max_abs_delta": _d, "per_series": _per,
                            "ref_n_cycles": int(len(ref_log["cycle"])),
                            "n_cycles_equal": bool(len(ref_log["cycle"]) == len(log["cycle"]))})
        assert _d == 0.0 and replay_gate["n_cycles_equal"], (
            f"[audiation] cross-tag replay FAILED at end of arm: max|delta|={_d:.3e} over "
            f"c1-c{_n}, ref has {len(ref_log['cycle'])} cycles, this arm {len(log['cycle'])}")
        print(f"[audi]  arm={arm} CROSS-TAG REPLAY PASS: {cfg['replay_ref']}/{arm} over all "
              f"{_n} cycles, max|delta| = {_d:.3e}", flush=True)

    # [E1b] close the revision log. Independent of the phase-1 instrument, so the treatment's
    # own record survives even if decision logging is off for this arm.
    pdrec = None
    if per_datum:
        rev_w.flush()
        pdrec = {"n_events": rev_w.n_ev, "shards": rev_w.manifest,
                 "n_cycles": len(pd_cycles),
                 "value_lr": float(vopt.param_groups[0]["lr"]),
                 "prop_lr": (float(popt.param_groups[0]["lr"]) if popt is not None else None),
                 "anchor_value_lr": cfg["value_lr_online"] or cfg["value_lr"],
                 "anchor_prop_lr": cfg["prop_lr"],
                 "anchor_n_grad": cfg["n_grad"], "anchor_prop_steps": cfg["prop_steps"],
                 "pd_n": cfg["pd_n"], "pd_maint_v": cfg["pd_maint_v"],
                 "pd_maint_p": cfg["pd_maint_p"], "pd_prop_nominal": cfg["pd_prop_nominal"],
                 "pd_probe_n": cfg["pd_probe_n"], "pd_ref_n": cfg["pd_ref_n"],
                 "cycles": pd_cycles}
        with open(os.path.join(outdir, arm, "perdatum.json"), "w") as fh:
            json.dump(pdrec, fh, indent=2, cls=NumpyEncoder)
        volume.commit()
        tot = sum(q["bytes"] for q in rev_w.manifest)
        n_pi = sum(q["n_pi_steps"] for q in pd_cycles)
        print(f"[pd]    arm={arm} revisions: {rev_w.n_ev} update events over "
              f"{len(pd_cycles)} cycles in {len(rev_w.manifest)} shards ({tot / 1e6:.0f} MB); "
              f"{sum(q['n_value_steps'] for q in pd_cycles)} per-datum value steps, "
              f"{n_pi} per-datum pi steps", flush=True)

    audi = None
    if snapper is not None:
        snapper.snap(cyc + 1, era_i, value, prop, generator, controller,
                     meter[era_i][1], meter[era_i][0], device)
        dec_w.flush()
        tr = snapper.write_trace(outdir, arm)
        audi = {"n_cycles_logged": len(dec_cycles), "replay_gate": replay_gate,
                "shards": dec_w.manifest, "n_tips": dec_w.n_tips, "n_cand": dec_w.n_cand,
                "snapshot_files": snapper.files, "snapshot_checks": snapper.checks,
                "snapshot_check_all_exact": all(
                    q["max_abs_dv"] == 0.0 and q["max_abs_dpi"] == 0.0
                    for q in snapper.checks),
                "probe_trace": os.path.basename(tr) if tr else None,
                "probe_n": snapper.n_probe,
                "final_snapshot_cycle": int(cyc + 1),
                "n_slots": int(n_slots), "slot_offsets": {str(k): int(q)
                                                          for k, q in offsets.items()},
                "cycles": dec_cycles}
        with open(os.path.join(outdir, arm, "audiation.json"), "w") as fh:
            json.dump(audi, fh, indent=2, cls=NumpyEncoder)
        volume.commit()
        tot = sum(q["bytes"] for q in dec_w.manifest)
        print(f"[audi]  arm={arm} decisions: {dec_w.n_tips} tip-steps / {dec_w.n_cand} "
              f"candidates over {len(dec_cycles)} cycles in {len(dec_w.manifest)} shards "
              f"({tot / 1e6:.0f} MB); snapshots: {len(snapper.files)} files, "
              f"recompute exact on all {len(snapper.checks)} checks="
              f"{audi['snapshot_check_all_exact']}", flush=True)

    # ------------------------------------------------------------------ #
    # THE ABLATION BATTERY — on the FINAL trained state, after the 90 cycles.
    # ------------------------------------------------------------------ #
    # Run as an end-of-run probe rather than a mid-run switch, so the trajectory every other
    # readout is measured on is never corrupted by the ablation. Unpriced (it is an instrument),
    # but each condition's own per-solve cost is recorded so the conditions are comparable.
    #
    #   a_full   the system as it ran
    #   b_table  THE TABLE IS DELETED. The macro stays in the action set — the policy can still
    #            call the address — but nothing may consult the table at execution time, so
    #            every macro call is served by the span head. POLICY, ON THE RECORD: under this
    #            condition the head serves ALL macro calls regardless of the parity gate, so a
    #            parity shortfall shows up as error rather than being hidden behind a fallback
    #            to the very DP path the condition is supposed to have removed. Arms with no
    #            span head have no execution path at all and are reported as `b_span`.
    #   b_span   table AND span reach removed: pure enumeration over base moves.
    #   c_prims  the primitives are deleted instead; macros only.
    #
    # Next-level currency is split into the two components the prediction confounds:
    #   built        entries `Miner.build` can assemble over the surviving lower table. Under
    #                `b_table` this is 0 BY CONSTRUCTION (T3 subset T2 x T2), so it is the
    #                STRUCTURAL half and carries no information on its own.
    #   at_support   distinct level-1 tuples observed at least `mine_support` times in the
    #                chosen trajectories, keyed by the flat tuple and gated by no table at all.
    #                This is the INFORMATIVE half: does the observation stream of chunk-shaped
    #                spans survive when the routing is native?
    def battery():
        macro_ms = [mv for mv in ms if mv.get("kind") == "macro"]
        base_only = [mv for mv in ms if mv.get("kind") != "macro"]
        conds = {"a_full": (ms, "normal"), "b_table": (ms, "force_head"),
                 "b_span": (base_only, "plain"), "c_prims": (macro_ms, "normal")}
        saved = {k: sl["open"] for k, sl in slots.items()}
        if use_span:
            ex.capture = False
        out = {}
        for cname, (ms_c, mode) in conds.items():
            if not ms_c:
                out[cname] = None
                continue
            if mode == "force_head" and not use_span:
                out[cname] = {"skipped": "no span head — no execution path; see b_span"}
                continue
            if use_span:
                for k_ in slots:
                    slots[k_]["open"] = (mode == "force_head") or (
                        mode == "normal" and saved[k_])
                ex.fire = mode != "plain"
            cell = {"n_moves": len(ms_c), "eras": {}}
            for j, era_j in enumerate(eras):
                jr, jx = meter[j]
                slots_c = PN.move_slots(ms_c, offsets) if prop is not None else None
                if prop is not None:
                    k_c = len(ms_c) if prop_k_spec < 0 else min(int(prop_k_spec), len(ms_c))
                    port_c = {"prop": prop, "slots": slots_c, "k": k_c, "forced": (),
                              "n_slots": n_slots}
                    w_c = cap_w(PN.fit_width_k(k_c, cfg["budget"], cfg["g_budget"]))
                else:
                    port_c, w_c = None, cap_w(fit_width(len(ms_c), cfg["budget"],
                                                       cfg["g_budget"]))
                ex.reset()
                bb = plan(controller, generator, value, jx, torch.from_numpy(jr), ms_c,
                          rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                          beam_width=w_c, device=device, port=port_c, ex=ex)
                sc, _ = grade(bb["x"].cpu().numpy(), jr, rules, s)
                n = jx.shape[0]
                row = {"e": 1.0 - float(sc.mean()), "w": int(w_c),
                       "g_solve": bb["counts"]["ground"] / n,
                       "mat_solve": bb["counts"]["mat"] / n,
                       "blocks": {kk: vv / n for kk, vv in ex.counts.items()}}
                # next-level currency, mined from THIS condition's own chosen trajectories
                keep = bb["x"][torch.from_numpy(sc > 0.5).to(device)]
                mn = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
                if keep.shape[0]:
                    pf = MC.parse_features(shared["reader"], keep, s=s).cpu().numpy()
                    for ell in range(2, maxl + 1):
                        span_l = s ** (ell - 1)
                        node_l = (era_j["node"] * s ** (era_j["level"] - 1)) // span_l
                        mn[ell].observe(pf[:, node_l * span_l:(node_l + 1) * span_l])
                row["n_solved"] = int(keep.shape[0])
                row["mine"] = {}
                for ell in range(2, maxl + 1):
                    st = mn[ell].state()
                    lower = MC.base_table(v) if ell == 2 else operative(ell - 1)
                    built = mn[ell].build(lower, cfg["mine_support"])["child"].shape[0]
                    row["mine"][str(ell)] = {
                        "at_support": int(st["n_at_support"][str(cfg["mine_support"])]),
                        "n_distinct": int(st["n_distinct"]), "n_obs": int(st["n_obs"]),
                        "built": int(built)}
                cell["eras"][str(j)] = row
                print(f"[ablate] arm={arm} {cname:8s} era{j + 1} nm={len(ms_c)} w={w_c} "
                      f"e={row['e']:.4f} g={row['g_solve']:.1f} solved={row['n_solved']} "
                      f"T2@sup={row['mine']['2']['at_support']} "
                      f"T3@sup={row['mine'].get('3', {}).get('at_support')} "
                      f"T3built={row['mine'].get('3', {}).get('built')}", flush=True)
            out[cname] = cell
        if use_span:
            for k_ in slots:
                slots[k_]["open"] = saved[k_]
            ex.fire = True
            ex.capture = True
        out["_parity_at_end"] = (log["span"][-1]["parity"] if log["span"] else {})
        out["_open_at_end"] = saved
        return out

    print(f"\n----- arm={arm} ABLATION BATTERY (final trained state) -----", flush=True)
    ablation = battery()

    # [conductor] the loop's own summary, computed by the same pure reducer the analyzer and the
    # offline gate use, plus the panel's cross-check that the generalised observation miner
    # reproduces the donor's G-Y instrument at level 4 on every cycle.
    loop_summary = PO.summarise_trace(log["loop"], loop_actions, len(log["cycle"]))
    loop_summary.update({"policy": loop.kind, "read": loop.read_key,
                         "span": cfg["loop_span"], "W": cfg["loop_W"],
                         "burn": cfg["loop_burn"], "alpha": cfg["loop_alpha"],
                         "priced": bool(getattr(loop, "priced", False))})
    obs_extra = {"loop": loop_summary, "loop_actions": loop_actions,
                 "ledger": ledger.state(),
                 "obs_hist": {str(k): q for k, q in obs_hist.items()},
                 "obs_gy_agree": bool(all(obs_gy_agree)) if obs_gy_agree else None,
                 "obs_gy_agree_n": len(obs_gy_agree),
                 "era_caps": list(caps), "total_cap": int(total_cap)}
    print(f"[loop]   arm={arm} {loop.kind}"
          + (f" read={loop.read_key}" if loop.read_key else "")
          + f": {loop_summary['n_commits']} commits at "
            f"{loop_summary['commit_cycles']}, {loop_summary['n_advances']} advances at "
            f"{loop_summary['advance_cycles']} "
            f"({loop_summary['n_chosen']} chosen / {loop_summary['n_capped']} capped); "
            f"spend {ledger.state()['spend_g']}g; "
            f"obs==G-Y on {len(obs_gy_agree)} cycles: "
            f"{bool(all(obs_gy_agree)) if obs_gy_agree else None}", flush=True)
    # [audiation] a COMPACT summary in `results.json`; the per-cycle context and the shard
    # manifest live in `audiation.json` beside it, because `ms_slots` x 116 cycles would
    # double the size of a file every reduction in the arc loads whole.
    audi_sum = None if audi is None else {
        k: q for k, q in audi.items() if k not in ("cycles", "snapshot_files", "shards")}
    if audi_sum is not None:
        audi_sum["n_shards"] = len(audi["shards"])
        audi_sum["n_snapshot_files"] = len(audi["snapshot_files"])
        audi_sum["bytes_decisions"] = int(sum(q["bytes"] for q in audi["shards"]))
    # [E1b] a compact per-datum summary in `results.json`; the per-cycle record is in
    # `perdatum.json`, for the same reason the decision context is in `audiation.json`.
    pd_sum = None if pdrec is None else {
        k: q for k, q in pdrec.items() if k not in ("cycles", "shards")}
    if pd_sum is not None:
        pd_sum["n_shards"] = len(pdrec["shards"])
        pd_sum["bytes_revisions"] = int(sum(q["bytes"] for q in pdrec["shards"]))
        pd_sum["n_pd_value_steps"] = int(sum(q["n_value_steps"] for q in pdrec["cycles"]))
        pd_sum["n_pd_pi_steps"] = int(sum(q["n_pi_steps"] for q in pdrec["cycles"]))
    write_results(outdir, arm, cfg, eras, refs, log, events, complete=True,
                  extra={"prop_k": prop_k_spec, "span_mode": use_span,
                         "twin": TWIN.get(base), "slot_events": slot_events,
                         "gate_events": gate_events, "ablation": ablation,
                         "shadow_cert": _cert_summary(shadow_cert),
                         "stream_burn": burn_rec,
                         "gy_final": (gy_miner.state() if gy_miner is not None else None),
                         "gauge_hist": {str(k): q for k, q in gauge_hist.items()},
                         "audiation": audi_sum, "perdatum": pd_sum,
                         **obs_extra})
    return {"log": log, "events": events, "ablation": ablation, "stream_burn": burn_rec,
            "audiation": audi_sum, "perdatum": pd_sum,
            "shadow_cert": _cert_summary(shadow_cert),
            "gy_final": (gy_miner.state() if gy_miner is not None else None),
            "gauge_hist": {str(k): q for k, q in gauge_hist.items()}, **obs_extra}


def plant_probe(generator, shared, cfg, device):
    """The self-imitation-collapse guard: the plant's fidelity to the grammar, measured on
    CLEAN held-out configurations it never trains on directly.
      infill_acc -- masked-block level-1 feature accuracy (the generator's actual job)
      parse_acc  -- unmasked-block feature accuracy, i.e. how right the agent's own parse is,
                    which is what mining reads. Both are oracle readouts, never consumed."""
    import torch
    v, s = cfg["v"], cfg["s"]
    n_blocks = shared["n_blocks"]
    x = shared["probe_clean"].to(device)
    b = x.shape[0]
    powers = v ** torch.arange(s, device=device)
    feats = shared["bottom_map"][(x.view(b, n_blocks, s) * powers).sum(-1)]
    pick = torch.from_numpy(
        np.random.default_rng(0).integers(0, n_blocks, size=b)).to(device)
    pos = pick[:, None] * s + torch.arange(s, device=device)[None, :]
    with torch.no_grad():
        parse = generator.block_logits(x).argmax(-1)
        obs = x.clone().scatter_(1, pos, torch.full_like(pos, -1))
        infill = generator.block_logits(obs).argmax(-1)
    tgt = feats.gather(1, pick[:, None]).squeeze(1)
    pred = infill.gather(1, pick[:, None]).squeeze(1)
    ok = feats >= 0
    ok1 = tgt >= 0
    with torch.no_grad():
        rd = shared["reader"].block_logits(x).argmax(-1)
    return {"parse_acc": float((parse[ok] == feats[ok]).float().mean()),
            "infill_acc": float((pred[ok1] == tgt[ok1]).float().mean()),
            "read_acc": float((rd[ok] == feats[ok]).float().mean())}


def priced(counts, cfg):
    """ratchet's ledger, plus the proposal reads at `c_prop`. `c_prop` defaults to 0.0: every
    proposal read below the root is a head-only forward on an encoder state the value already
    paid a grounding for, and the one read that is NOT (the root) is charged a full grounding
    inside the beam. The raw count is logged either way so the reduction can re-price it."""
    return (counts["ground"] * cfg["d_fb"] + counts["mat"] * cfg["c_mat"]
            + counts.get("prop", 0) * cfg.get("c_prop", 0.0))


def write_results(outdir, arm, cfg, eras, refs, log, events, complete, extra=None):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": eras, "refs": refs,
                   "log": log, "events": events, "complete": complete, **(extra or {})},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# references, measured at setup ON THE METERING SETS THEMSELVES
# --------------------------------------------------------------------------- #

def measure_refs(shared, cfg, eras, device):
    """`stale` is the detector's scale anchor and round 1's bug: it measured the reference on
    a 512-instance reference draw and the metering on a different 384-instance draw, so the
    two disagreed by more than the descent and the scale collapsed to a much looser fallback.
    Fixed here — every reference below is measured on the ERA'S OWN METERING SET.

    Also measured per era: the exact-DP floor over the base move set, the base-move and
    true-vocabulary performance beams at the declared budget, and the one-action ceilings of
    the true level-l macro (the number `given` is buying and `practice_*` is chasing)."""
    import torch
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                 for ell in range(2, cfg["max_macro_level"] + 1)},
                       s, depth, device)
    out = {"stale": [], "stale_true": [], "floor": [], "d0": [], "on_grammar": [],
           "width_base": [], "width_true": [], "g_base": [], "g_true": [], "macro_true": []}
    for i, era in enumerate(eras):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np)
        out["d0"].append(float(nearest_derivation_cost(rules, x_np, r_np, s).mean()))
        out["on_grammar"].append(on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s))
        for tag, mset in (("base", base_ms), ("true", true_ms)):
            w = fit_width(len(mset), cfg["budget"], cfg["g_budget"])
            b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                           torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                           budget=cfg["budget"], beam_width=w, device=device)
            sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
            out["stale" if tag == "base" else "stale_true"].append(1.0 - float(sc.mean()))
            out[f"width_{tag}"].append(w)
            out[f"g_{tag}"].append(b["counts"]["ground"] / x0.shape[0])
        xo, _ = oracle_rollout(shared["generator0"], x0.to(device), r_np, base_ms, rules,
                               rules_t, canon, depth, v, m, s, cfg["budget"])
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        out["floor"].append(1.0 - float(sc.mean()))
        cell = {}
        for ell in range(2, cfg["max_macro_level"] + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            mv = MC.to_device(MC.make_macro(ell, node, s, shared["truth"][ell]), device)
            cell[str(ell)] = audition_macro(shared["generator0"], x0.to(device), r_np, mv,
                                            rules_t, canon, depth, v, m, s, rules)["e"]
        out["macro_true"].append(cell)
    return out


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = dict(
        v=8, s=2, depth=4, m=2, rule_seed=0, train_seed=1, seed=0, state_dim=96,
        n_train_episodes=100_000, controller_steps=12_000, generator_steps=12_000,
        reader_steps=6_000, plant_holdout=0, holdout_seed=11,
        value_steps=12_000, value_episodes=40_000, value_batch_collect=1024, batch_size=256,
        value_lr=3e-4, value_lr_online=3e-5, n_corrupt=3, explore_eps=0.3,
        budget=4, pr_width=16, g_budget=58, max_macro_level=3,
        n_pr=64, n_rt=384, n_score=512, n_probe_clean=512,
        era_cycles=24, n_grad=4, value_batch=256, replay_frac=0.5, buf_cap=100_000,
        gen_lr=1e-4, gen_steps=20, mine_support=3, mine_from="chosen", mine_cap=0,
        alpha=0.2, sil_c=0.06, sil_cv=0.10, sil_W=5, sil_hold=2, sil_min_cycle=6,
        lp_min_drop=0.05, early_offset=1, late_offset=2,
        d_fb=1.0, c_mat=0.05, checkpoint_every=5, probe_widths=(1, 2, 4, 8, 16), probe_every=4,
        # --- the port ---------------------------------------------------------------------
        prop_lr=3e-4, prop_steps=24, prop_batch=256, prop_buf_cap=60_000,
        prop_warmup=4, prop_new_cycles=2, prop_explore=1, prop_train_on="solved", c_prop=0.0,
        # --- the port 2 knobs (`../span/`'s, unchanged) ------------------------------------
        span_lam=1.0, span_lr=1e-3, span_batch=64, span_buf_cap=8192, span_hold_cap=2048,
        span_hold_frac=0.1, span_capture=24, span_min_hold=256, span_tau=0.95,
        span_hidden_mult=4,
        # --- [spiral] the transplanted commit policy's own knobs (`ear`'s names and defaults).
        #     `prov_offset=0` == the boundary is the era's LAST cycle. `n_aud` is the recert's
        #     grading set size; `recert_every` its cadence within an era.
        prov_offset=0, recert_every=5, n_aud=192,
        # --- [census] this round's own knobs ------------------------------------------------
        # G-Y: the unpriced next-level observation instrument. `gy_level=4` is deliberately
        # ABOVE `max_macro_level=3` — the instrument observes, it never commits.
        gy_level=4,
        # the L2 gate's gauge (Phase 0 R5: the only grid corner that fired after the
        # certificate in all four sp_s0 earning arms).
        gate_level=None, gate_W=12, gate_theta=0,
        # commit-then-extend. `extend_cap` bounds the auditions one recert may buy;
        # `extend_tol` is the admission slack on the audition error (0.0 = must not hurt).
        extend_cap=8, extend_tol=0.0,
        yoke_cycle=None, yoke_provisional=False,
        # --- [audiation] THE INSTRUMENT'S OWN KNOBS ----------------------------------------
        # `log_decisions=False` HERE, in the base config, so every donor entrypoint this file
        # inherits (`preflight`, `fidelity_smoke`, `phase_a`, the `cal_*` runs) is the donor
        # exactly and the instrument is opt-in at the one entrypoint that wants it.
        log_decisions=False, dec_flush_every=5, dec_compress=True,
        n_snap_probe=64, snap_check_every=20, plant_snap_every=8,
        replay_ref="",
        # --- [E1b] PER-DATUM CREDIT. `per_datum=False` here, in the base config, so it is a
        #     property of ONE arm and every other arm in this file is untouched by it.
        per_datum=False,
        pd_n=128,                # graded trajectories consumed per cycle, one step each
        pd_maint_v=4,            # pooled replay-ONLY value steps per cycle (anti-drift)
        pd_maint_p=4,            # pooled history-ONLY pi steps per cycle (anti-forgetting)
        pd_prop_nominal=32,      # the pi step count the matched lr is set against
        pd_value_lr=0.0,         # 0 -> derive from the matched n_steps*lr budget
        pd_prop_lr=0.0,
        pd_probe_n=32,           # fixed-probe states the per-event revision is measured at
        pd_ref_n=32,             # same-cycle states this pass does NOT step on (spillover)
    )
    cfg.update(kw)
    return cfg


def _d6_cfg(**kw):
    """[spiral] `_cfg` with the depth-6 substrate and the two hard prerequisites `tall` paid
    for, so no entrypoint can forget them:

      * `n_corrupt=1` — `dens0` measured this is the WHOLE densification lever (terminal
        success 0.133 against the `tl_s0` config's 0.070; collection budget is not a lever and
        slightly hurts). It is read inside `build_shared`, so the stale value buffer every arm
        starts from is densified by construction.
      * `budget=8, g_budget=482` — `beam_ground(32, 8, 2) = 482`, i.e. exactly width 2 over the
        base action set, and exactly `dens0`'s declared budget so the enum descent rate this
        file measures is comparable to the −0.004/cycle it measured rather than to a
        projection. Under ENUMERATION this budget cannot hold width 2 past the L2 commit
        (n=48 needs >= 722) — which is precisely the pricing inversion routing is claimed to
        dissolve, so raising G here would confound the test that Phase A exists to run.
      * probes re-priced — they were ~47% of `tl_s0`'s runtime (13 beams per firing at ~68 s).
    """
    cfg = _cfg(**DEPTH6)
    cfg.update(n_corrupt=1, budget=8, g_budget=482, max_macro_level=3,
               probe_widths=(1, 2, 4), probe_every=8,
               # span_min_hold 256 -> 128. MEASURED reason: in Phase A the composed arm
               # accumulated 184-218 held-out observations per slot in the 6 cycles it had
               # after the L2 commit (~30-35/cycle), so at 256 the parity gate was never even
               # EVALUATED (`parity` returned None on every slot, 0 open in all 22 cycles) and
               # the corridor wall — one of the three walls this round exists to autopsy — was
               # untestable. 128 puts a first parity read ~4 cycles after a commit. The quality
               # guard is unchanged: tau = 0.95 exact-match, re-checked every cycle, not
               # latched, so a slot the moving plant drifts away from closes again.
               span_min_hold=128,
               collect_task_matched=True, tm_episodes=8192,
               # the certificate's own constants are `ear`'s, not the donor's, because the
               # commit policy transplanted here is `ear`'s and these are the values it was
               # calibrated with (`full._cfg` carries 0.10 / 0.05 / no mine cap). Cycles-to-
               # certification is this round's primary rate readout, so the detector has to be
               # the one the arc actually tuned. `mine_cap=8` is also the cap `tall`'s coverage
               # law is stated at (L2 ~6 cycles, L3 ~24, L4 ~384).
               sil_cv=0.15, lp_min_drop=0.10, mine_cap=8,
               # [assay] `surgery` is per-arm (set from each arm's own `cfg`); `entry_rec`
               # switches the per-execution entry-identity instrument, on by default.
               surgery=None, entry_rec=True,
               # [conductor] the outer loop's own knobs.
               #   era_caps       per-era cycle caps for loop-driven and yoked arms; schedule
               #                  arms keep the ladder's own cycle counts untouched.
               #   loop_span/W    the block geometry. span=1 is `floors.py`'s measured choice:
               #                  a longer horizon raises every gauge's signal-over-floor
               #                  (L4 1.49 -> 2.74 at span 3) but needs a 4*span+1 window, and
               #                  the consumption eras are 12/9/7 cycles long, so span 1 is the
               #                  only horizon with decision points in EVERY era.
               #   loop_burn      decision points held after every action and at every era
               #                  start. The regime just changed; it has not been observed yet.
               #   tol_*          THE MEASURED DEAD ZONES (`MEASURED_FLOORS`). Never defaulted:
               #                  `floor_gate` asserts each is present, positive, and equal to
               #                  what `floors.json` derived.
               #   n_endo         sequences per endo read, from the fixed clean probe pool.
               #   endo_price     grounding-equivalents charged per PRICED endo read, measured
               #                  by `endo_bench` on the same GPU and passed in explicitly.
               era_caps=tuple(int(x) for x in CONDUCTOR_CAPS.split(",")),
               total_cap=0, loop_span=1, loop_W=4, loop_burn=4, loop_alpha=0.5,
               tol_ledger=MEASURED_FLOORS["ledger"],
               tol_yield_l3=MEASURED_FLOORS["yield_by_level"][3],
               tol_yield_l4=MEASURED_FLOORS["yield_by_level"][4],
               tol_endo=None, n_endo=256, endo_price=0, yoke_plan=None,
               endo_read_key="endo_excess", obs_panel=True)
    # [census] Phase 0 R6's ladder: the gate is at L2, so era 1 is what must be long enough to
    # contain its firing (latest all-arm L2 firing in the replay was c40, against era 1's old
    # c32 boundary). The cycles come out of era 2, which under R1 no longer hosts a gate, so
    # the total per arm is unchanged at 116 and `sp_s0`'s measured cost carries over.
    cfg.update(eras_default=CENSUS_LADDER)
    cfg.update(kw)
    return cfg


# --------------------------------------------------------------------------- #
# [spiral] PHASE B — the main run
# --------------------------------------------------------------------------- #

# The sized ladder. Levels/nodes are `tall`'s nested cells; the third field is that era's own
# cycle count, from OFFLINE DETECTOR REPLAY on Phase A's own logged audition series plus the
# arc's ~50% margin convention — not from round numbers, and deliberately not from `cal0`'s
# clock-paced figures, which the replay shows are optimistic at this depth.
#
#   THE REPLAY, and why it changed the sizing. Phase A's `sil run` was 0 at every cycle, which
#   looks like a dead detector and is not: its arms committed at c16 under `commit="late"`,
#   and the donor's certificate stops being evaluated the moment a level commits. Re-running
#   the silence test offline over the full 22-cycle audition series (`log["cert"]["A"]`, the
#   same series the shadow certificate reads) at this round's constants
#   (sil_c 0.06 / sil_cv 0.15 / lp_min_drop 0.10):
#
#       L2 certificate fires at c19 (enum arm) and c21 (routed arm).
#
#   `cal0` predicted c12 for L2, so depth-6 certification is ~1.6-1.75x later than the
#   clock-paced calibration suggested. That ratio is the sizing input.
#
#   era 1 (earns L2), 32 cycles — max replayed firing c21, x1.5 margin = 32. 28 would have
#     truncated the L2 certificate, which is exactly the 30-cycle-truncation mistake
#     `ratchet`/`ear` made and `tall`'s design doc warns against repeating. At 32 the L2 commit
#     should be CERTIFIED rather than provisional-at-boundary, and there are ~11 cycles left in
#     the era for the corridor to start accumulating parity data before era 2 begins.
#   era 2 (earns L3), 56 cycles — the binding constraint of the run. `cal0` saw L3 at c22;
#     scaled by the measured 1.7x and given the same 1.5x margin, 53-58. The mining-coverage
#     floor (~24 cycles at mine_cap=8) sits comfortably inside it, and the remaining headroom
#     is spent here on purpose: this is where BOTH the L3 crank and the corridor wall live (a
#     span slot needs ~4 cycles past commit to reach a first parity read at span_min_hold=128,
#     and a parity TRAJECTORY needs many more than one read).
#   eras 3-5 (consumption; L4+ is not earnable at any affordable budget), 12/9/7 — long enough
#     for at least two probes each at probe_every=8, which is what a stable cost-to-depth read
#     needs, and no longer: nothing can be earned there.
#
# 116 cycles/arm. At Phase A's measured per-cycle costs (8.1-13.1 s routed/enum at n=32; 23.4
# s/cycle for the 56-move `given`/`fid` shapes) plus the four extra all-eras probe beams a
# five-era ladder adds, seven arms come to ~4.2 GPU-h — inside the 6 GPU-h cap.
SPIRAL_LADDER = "1:25:32,2:12:56,3:6:12,4:3:9,5:1:7"

# [census] Phase 0 R6. Same 116 cycles, redistributed: era 1 carries the L2 gate and must be
# long enough for it to fire in-era (replayed latest all-arm L2 firing c40 vs era 1's old c32
# boundary, +20% margin => 48); era 2 gives up the difference because under R1 it no longer
# hosts a gate.
CENSUS_LADDER = "1:25:48,2:12:40,3:6:12,4:3:9,5:1:7"

# Arm order is load-bearing, not cosmetic: `census_gate` must run BEFORE `yoked_delay`,
# because the yoked arm's commit cycle is `census_gate`'s MEASURED one, passed in at runtime.
# The anchor runs first so the in-tag replay exists before anything is read against it.
CENSUS_ARMS = ("spiral_route,census_gate,yoked_delay,census_extend,given_route,given_native")

# Arm ORDER is deliberate and is a budget-risk decision, not alphabetical: the ceiling
# reference and the in-tag fidelity assertion first (nothing else is readable without them),
# then the enum crank, then the two treatments, then the g722 rate control, and `given_native`
# last because the SPEC names it the most cuttable (its content is a depth-6 replication of
# `native/`). If the run is cut short, what is lost is what was cheapest to lose.
SPIRAL_ARMS = ("given,fid,enum_live,spiral_route,spiral,enum_live_g722,given_native")


# Arm ORDER is a budget-risk decision: the anchor first (it is the in-tag fidelity carrier
# and nothing else is readable without it), then the two amount x truth cells the round exists
# to price, then the mechanism dose, then the bracket's top, and `strip` last because the SPEC
# names it most cuttable — its content is a subset of what `complete` and `junk_dose` bracket.
ASSAY_ARMS = "anchor,complete,exact,junk_dose,given_c1,strip"


# [conductor] THE CAPS. The era SEQUENCE is the world and is fixed; what the loop owns is when
# to leave each era. A cap per era (and a total) is what makes a REFUSING arm terminate — an arm
# that rides its caps to the end is a readout, not a hang. Set at 1.25x the scheduled ladder
# (48/40/12/9/7 -> 60/50/15/12/9 = 146), which gives the loop room to hold a quarter longer than
# the schedule did in every era while keeping six arms inside ~2.6 GPU-h at `cs_s0`'s measured
# 10.4 s/cycle. An arm that advances EARLY simply costs less.
CONDUCTOR_CAPS = "60,50,15,12,9"

# Arm ORDER is load-bearing, not alphabetical. The anchor runs first because it is the in-tag
# replay carrier and nothing else is readable without it; each gauge arm runs immediately before
# its yoke, because the yoke's action cycles are the gauge arm's MEASURED ones, passed in at
# runtime (`census`'s mechanic); `outer_ledger` runs last because it is the only arm whose
# content is a re-ask of a donor measurement rather than a new cell, and so the most cuttable.
CONDUCTOR_ARMS = "anchor,outer_yield,yoked_yield,outer_endo,yoked_endo,outer_ledger"

# [audiation] E1 phase 1 runs ONE arm: the anchor — the scheduled crank, routing-only, and the
# arc's cross-tag replay carrier. Nothing here is a treatment; the round's whole output is a
# dataset, and the arm to log is the one whose trajectory is already certified identical to
# `cd_s0/anchor` and `as_s0/anchor`, so the log is a log OF a known run rather than of a new
# one. The loop arms are not run: their extra trajectories would cost 5x the GPU and 5x the
# bytes for a variable phase 2 has no question about yet.
AUDIATION_ARMS = "anchor"

# [audiation E1b] THE PER-DATUM ARM. `anchor` in every respect except HOW the value and pi
# consume the cycle's grades — the crank, the ladder, the beam, the mining, the plant, the
# commit policy, the recert, the seed and the RNG streams are all the anchor's, and the arm
# twins onto the anchor's torch stream like every other arm in the file. Run BESIDE the anchor
# so the contrast is in-tag: the anchor arm of the same run is the pooled comparator AND the
# inverse gate (per-datum off must replay `cd_s0/anchor` bit-identically).
ARMS["perdatum"] = dict(ARMS["anchor"], cfg={"per_datum": True})
TWIN["perdatum"] = "enum_live"
AUDIATION_ARMS_E1B = "anchor,perdatum"


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=1800, memory=16384)
def audi_preflight(depth: int = 6, mm: int = 2, n: int = 24, budget: int = 3):
    """[audiation] the instrument's own preflight, at toy sizes and before any paid setup.

    Two things, in order: GATE B (the instrumented beams against the donor module's, recorder
    off and on) and then the WRITER round trip — the same recorder output flattened, sharded to
    an `.npz`, read back, and checked for the properties phase 2 will rely on: the terminal
    join is positional, `c_child` points at a real tip, lineage closes, and nothing is NaN
    where it should not be. No number here is a measurement."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    bg = beam_gate(depth=depth, m=mm, n=n, budget=budget)
    print(f"[beam] {json.dumps(bg, cls=NumpyEncoder)}", flush=True)

    # ---- the writer round trip, on a recorder run of the PROPOSED beam ------------------ #
    v, s, max_level = 8, 2, 3
    length = s ** depth
    rules = generate_rules_distinct(v, s, depth, mm, seed=0)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device).eval()
    ctrl = _build_rich_controller()(v, length, s, 96, n_head=4, n_layer=2).to(device).eval()
    val = _build_value_head()(96, v).to(device).eval()
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    x0 = torch.from_numpy(_corrupt(leaves_np, length // s, 3, v, s,
                                   np.random.default_rng(5))).to(device)
    truth = MC.true_tables(rules, depth, s, v, mm, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    offsets, n_slots = PN.slot_layout(depth, s, max_level)
    prop = PN.build_head(96, v, n_slots, 99, device); prop.eval()
    rec = DecRec(n_slots)
    rec.open(7, {"n_slots": n_slots})
    out = beam_moves_prop(ctrl, gen, val, x0, torch.from_numpy(roots_np).to(device), ms,
                          rules_t, canon, depth, v, mm, s, budget=budget, beam_width=4,
                          device=device, collect=True, prop=prop,
                          slots=PN.move_slots(ms, offsets), k=4, forced=(), n_slots=n_slots,
                          explore=1, erng=np.random.default_rng(7), rec=rec)
    tips = out["tips_x"]
    B, W, T = tips.shape
    succ, dres = grade(tips.reshape(B * W, T).cpu().numpy(), np.repeat(roots_np, W), rules, s)
    rec.grades(succ, dres)
    root = f"{DATA_DIR}/{REMOTE}/_audi_preflight"
    os.makedirs(root, exist_ok=True)
    w = DecWriter(root, "toy", budget, flush_every=1)
    w.add(rec.close(), roots_np)
    path = os.path.join(w.dir, w.manifest[0]["file"])
    d = dict(np.load(path))
    volume.commit()

    chk = {"file": w.manifest[0], "keys": sorted(d)}
    n_t, n_c = d["t_cycle"].shape[0], d["c_cycle"].shape[0]
    chk["n_tips"], chk["n_cand"] = int(n_t), int(n_c)
    # 1. the terminal join is positional and complete
    term = d["t_step"] == budget
    chk["n_terminal"] = int(term.sum())
    assert chk["n_terminal"] == B * W, f"terminal rows {chk['n_terminal']} != {B * W}"
    assert np.array_equal(d["t_succ"][term], (succ > 0.5).astype(np.int8)), "succ join broke"
    assert np.allclose(d["t_dres"][term], dres.astype(np.float32)), "dres join broke"
    assert (d["t_succ"][~term] == -1).all() and np.isnan(d["t_dres"][~term]).all()
    assert not np.isnan(d["t_vfin"][term]).any() and np.isnan(d["t_vfin"][~term]).all()
    # 2. lineage closes: every non-root tip's parent is a real tip at the step before
    ok = True
    for t in range(1, budget + 1):
        here, prev = d["t_step"] == t, d["t_step"] == t - 1
        wprev = int((prev & (d["t_inst"] == 0)).sum())
        ok &= bool(((d["t_parent"][here] >= 0) & (d["t_parent"][here] < wprev)).all())
    chk["lineage_closes"] = bool(ok)
    assert ok, "a tip's parent index is out of range at some step"
    assert (d["t_parent"][d["t_step"] == 0] == -1).all()
    # 3. every kept candidate names a real tip at the next step, and the counts match
    for t in range(budget):
        cm = d["c_step"] == t
        kept = d["c_child"][cm] >= 0
        wnext = int(((d["t_step"] == t + 1) & (d["t_inst"] == 0)).sum())
        assert kept.sum() == wnext * B, f"kept {int(kept.sum())} != {wnext * B} at step {t}"
        assert (d["c_child"][cm][kept] < wnext).all()
    chk["kept_matches_beam"] = True
    # 4. pi is present on every decision step, absent nowhere, and finite on the live slots
    pi = d["t_pi"]
    live = ~np.isnan(pi)
    chk["pi_live_slots_per_row"] = int(live[d["t_step"] < budget][0].sum())
    assert chk["pi_live_slots_per_row"] == len(ms), "pi's live slot count != |ms|"
    assert np.isfinite(pi[live]).all(), "pi has a non-finite live logit"
    # 5. z and x are finite / in range
    assert np.isfinite(d["t_z"].astype(np.float32)).all(), "z has a non-finite entry"
    assert (d["t_x"] >= 0).all() and (d["t_x"] < v).all(), "x out of vocabulary range"
    chk["src_hist"] = {str(int(u)): int(c)
                       for u, c in zip(*np.unique(d["c_src"], return_counts=True))}
    chk["verdict"] = "PASS"
    print(f"[writer] {json.dumps(chk, cls=NumpyEncoder)}", flush=True)
    return {"beam_gate": bg, "writer": chk}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=36000, memory=32768)
def audiation_run(
    tag: str = "au_s0",
    arms: str = AUDIATION_ARMS,
    # [conductor] the loop's own knobs. `tol_endo` has NO default on purpose: a driven read with
    # no measured dead zone is an error (`floor_gate`), and the endo floor is the one that has no
    # offline series, so it is measured on the smoke tag and passed in explicitly.
    era_caps: str = CONDUCTOR_CAPS, total_cap: int = 0,
    loop_span: int = 1, loop_w: int = 4, loop_burn: int = 4, loop_alpha: float = 0.5,
    tol_ledger: float = MEASURED_FLOORS["ledger"],
    tol_yield_l3: float = MEASURED_FLOORS["yield_by_level"][3],
    tol_yield_l4: float = MEASURED_FLOORS["yield_by_level"][4],
    tol_endo: float = 0.0, n_endo: int = 256, endo_price: int = 0, obs_panel: bool = True,
    endo_read: str = "endo_excess",
    # [conductor] `--quick` collapses everything, but the SMOKE TAG is also where the endo dead
    # zone is measured, and a floor measured on a 128-sequence probe batch and a 5-step plant is
    # not the floor the main run will have. These three put the endo instrument at its
    # PRODUCTION configuration inside an otherwise-quick run, and make the eras long enough for
    # the rule to form windows at all.
    quick_cycles: int = 0, quick_probe_clean: int = 0, quick_gen_steps: int = 0,
    eras: str = CENSUS_LADDER, seed: int = 0, era_cycles: int = 24, budget: int = 8,
    pr_width: int = 16, g_budget: int = 482, max_macro_level: int = 3,
    depth: int = 6, mm: int = 2, n_corrupt: int = 1,
    prov_offset: int = 0, recert_every: int = 5, n_aud: int = 192,
    probe_widths: str = "1,2,4", probe_every: int = 8,
    n_pr: int = 64, n_rt: int = 384, n_score: int = 256, n_grad: int = 4,
    gen_lr: float = 1e-4, gen_steps: int = 20, mine_support: int = 3,
    mine_from: str = "chosen", mine_cap: int = 8,
    value_lr_online: float = 3e-5, sil_c: float = 0.06, sil_cv: float = 0.15,
    sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6, lp_min_drop: float = 0.10,
    early_offset: int = 1, late_offset: int = 3,
    plant_holdout: int = 0, holdout_seed: int = 11,
    d_fb: float = 1.0, c_mat: float = 0.05, c_prop: float = 0.0,
    prop_lr: float = 3e-4, prop_steps: int = 24, prop_batch: int = 256,
    prop_warmup: int = 4, prop_new_cycles: int = 2, prop_explore: int = 1,
    prop_train_on: str = "solved",
    span_lam: float = 1.0, span_lr: float = 1e-3, span_batch: int = 64,
    span_tau: float = 0.95, span_min_hold: int = 128, span_capture: int = 24,
    collect_task_matched: bool = True, tm_episodes: int = 8192,
    gy_level: int = 4, gate_win: int = 12, gate_theta: int = 0,
    extend_cap: int = 8, extend_tol: float = 0.0, entry_rec: bool = True,
    ref_tag: str = "", rule_seed: int = 0, train_seed: int = 1, quick: bool = False,
    # [audiation] the instrument. `log_decisions=False` reproduces `conductor_run` exactly.
    log_decisions: bool = True, dec_flush_every: int = 5, dec_compress: bool = True,
    n_snap_probe: int = 64, snap_check_every: int = 20, plant_snap_every: int = 8,
    replay_ref: str = "rhm_practice_conductor/cd_s0",
    # [audiation] restrict the phase-1 instrument to named arms (empty = every arm). The E1b
    # tag runs `anchor,perdatum` and logs only `perdatum`: the anchor's decision log would be
    # a byte-for-byte duplicate of `au_s0`'s, which the replay gate proves rather than assumes.
    log_arms: str = "",
    # [E1b] per-datum credit. Inert unless the `perdatum` arm is in `--arms`.
    pd_n: int = 128, pd_maint_v: int = 4, pd_maint_p: int = 4, pd_prop_nominal: int = 32,
    pd_value_lr: float = 0.0, pd_prop_lr: float = 0.0,
    pd_probe_n: int = 32, pd_ref_n: int = 32,
):
    """AUDIATION (E1, phase 1). Produce the dataset, do not analyse it.

    `conductor_run` with the instrument on and one arm. The science question — does the
    arity-2 self-model's residual carry an agency-gated, per-datum revision signal — is asked
    OFFLINE, in a second pass over what this writes; nothing here forecasts anything.

    What lands on the volume, per arm:

      results.json                the donor's, unchanged (the cross-tag replay gate reads it)
      audiation.json              the shard manifest, the snapshot manifest, the recompute
                                  checks, and the PER-CYCLE CONTEXT (`ms_slots`, `avail_slots`,
                                  `k_eff`, `forced`, `routed`, committed table sizes)
      decisions/dec_cXXXX_cYYYY.npz   the two per-decision tables, sharded on the donor's own
                                  checkpoint clock
      snapshots/heads_cXXXX.npz   value + pi as fp32 numpy, EVERY cycle, taken at the top of
                                  the cycle (see `schema.md` for the ordering that makes
                                  `heads_{c+1} - heads_c` the revision from cycle c's grades)
      snapshots/plant_cXXXX.npz   the plant, every `plant_snap_every` cycles — for
                                  materialisation counterfactuals only; no readout needs it
      snapshots/probe_trace.npz   64 fixed states per era, their live `v` and `pi` fp32 every
                                  cycle: the free readout trajectory, and the states the
                                  in-run recompute assert is taken on
      schema.md                   the record schema and the cycle-phase semantics, verbatim

    --- the donor's own docstring follows ---

    CONDUCTOR (A1). Does an outer loop reading the learner's own one-level-up currency drive
    the crank at least as well as the schedule did — and does a within-level reader refuse?

    Six arms at `cs_s0`/`as_s0`'s exact configuration and ladder, all routing-only, all on the
    anchor's stream, differing in exactly one thing (who drives) and, among the driven three, in
    exactly one further thing (what the rule reads):

      anchor        certificate-else-boundary, fixed ladder   the scheduled crank; the carrier
      outer_yield   at_support one level up (free)            the treatment
      yoked_yield   outer_yield's cycles, by clock            the mandatory timing control
      outer_endo    the plant's own NLL one level up (priced) the label-free treatment
      yoked_endo    outer_endo's cycles, by clock             its timing control
      outer_ledger  its own error on the era's cell (free)    the within-level reader
    """
    import torch
    cfg = _d6_cfg(depth=depth, m=mm, n_corrupt=n_corrupt, prov_offset=prov_offset,
                  era_caps=tuple(int(x) for x in era_caps.split(",") if x.strip()),
                  total_cap=total_cap, loop_span=loop_span, loop_W=loop_w,
                  loop_burn=loop_burn, loop_alpha=loop_alpha,
                  tol_ledger=tol_ledger, tol_yield_l3=tol_yield_l3, tol_yield_l4=tol_yield_l4,
                  tol_endo=(tol_endo or None), n_endo=n_endo, endo_price=endo_price,
                  obs_panel=obs_panel, endo_read_key=endo_read,
                  recert_every=recert_every, n_aud=n_aud,
                  probe_widths=tuple(int(x) for x in probe_widths.split(",")),
                  probe_every=probe_every,
                  span_lam=span_lam, span_lr=span_lr, span_batch=span_batch,
                  span_tau=span_tau, span_min_hold=span_min_hold, span_capture=span_capture,
                  c_prop=c_prop, prop_lr=prop_lr, prop_steps=prop_steps, prop_batch=prop_batch,
                  prop_warmup=prop_warmup, prop_new_cycles=prop_new_cycles,
                  prop_explore=prop_explore, prop_train_on=prop_train_on,
                  seed=seed, era_cycles=era_cycles, budget=budget, pr_width=pr_width,
                  g_budget=g_budget, max_macro_level=max_macro_level, n_pr=n_pr, n_rt=n_rt,
                  n_score=n_score, n_grad=n_grad, gen_lr=gen_lr, gen_steps=gen_steps,
                  mine_support=mine_support, mine_from=mine_from, mine_cap=mine_cap,
                  value_lr_online=value_lr_online, sil_c=sil_c, sil_cv=sil_cv, sil_W=sil_win,
                  sil_hold=sil_hold, sil_min_cycle=sil_min_cycle, lp_min_drop=lp_min_drop,
                  early_offset=early_offset, late_offset=late_offset,
                  plant_holdout=plant_holdout, holdout_seed=holdout_seed,
                  d_fb=d_fb, c_mat=c_mat, collect_task_matched=collect_task_matched,
                  tm_episodes=tm_episodes, rule_seed=rule_seed, train_seed=train_seed,
                  gy_level=gy_level, gate_W=gate_win, gate_theta=gate_theta,
                  extend_cap=extend_cap, extend_tol=extend_tol, entry_rec=entry_rec,
                  # [audiation]
                  log_decisions=log_decisions, dec_flush_every=dec_flush_every,
                  dec_compress=dec_compress, n_snap_probe=n_snap_probe,
                  snap_check_every=snap_check_every, plant_snap_every=plant_snap_every,
                  replay_ref=replay_ref,
                  # [E1b]
                  pd_n=pd_n, pd_maint_v=pd_maint_v, pd_maint_p=pd_maint_p,
                  pd_prop_nominal=pd_prop_nominal, pd_value_lr=pd_value_lr,
                  pd_prop_lr=pd_prop_lr, pd_probe_n=pd_probe_n, pd_ref_n=pd_ref_n)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   era_cycles=3, n_pr=24, n_rt=96, n_score=96, n_aud=48, n_probe_clean=128,
                   checkpoint_every=2, sil_min_cycle=2, gen_steps=5, tm_episodes=512,
                   prop_warmup=1, prop_steps=12, prop_buf_cap=20_000, probe_every=2,
                   recert_every=1, span_min_hold=16, span_tau=0.0)
        if quick_cycles:
            cfg["era_cycles"] = int(quick_cycles)
        if quick_probe_clean:
            cfg["n_probe_clean"] = int(quick_probe_clean)
        if quick_gen_steps:
            cfg["gen_steps"] = int(quick_gen_steps)
    ers = parse_eras(eras)
    if quick:
        for e_ in ers:
            e_["cycles"] = cfg["era_cycles"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    total_cycles = sum(int(e_.get("cycles") or cfg["era_cycles"]) for e_ in ers)
    # [conductor] in `--quick` the caps have to shrink with the ladder or a loop arm would run
    # the full-scale caps against a 3-cycle era and every arm would ride them.
    if quick:
        cfg["era_caps"] = tuple(max(cfg["era_cycles"] + 4,
                                    cfg["loop_W"] * cfg["loop_span"] + 2) for _ in ers)
    cap_total = int(cfg["total_cap"] or sum(cfg["era_caps"] or ()))
    print(f"audiation tag={tag} depth={cfg['depth']} m={cfg['m']} "
          f"seqlen={cfg['s'] ** cfg['depth']} arms={arms}\n"
          f"  instrument: log_decisions={cfg['log_decisions']} "
          f"flush/{cfg['dec_flush_every']}c snap_probe={cfg['n_snap_probe']} "
          f"file_check/{cfg['snap_check_every']}c plant_snap/{cfg['plant_snap_every']}c\n"
          f"  ladder={[(e_['name'], e_.get('cycles') or cfg['era_cycles']) for e_ in ers]} "
          f"= {total_cycles} cycles/arm scheduled; loop caps {list(cfg['era_caps'])} "
          f"= {cap_total} max\n"
          f"  loop: span={cfg['loop_span']} W={cfg['loop_W']} burn={cfg['loop_burn']} "
          f"alpha={cfg['loop_alpha']}  floors ledger={cfg['tol_ledger']} "
          f"yieldL3={cfg['tol_yield_l3']} yieldL4={cfg['tol_yield_l4']} "
          f"endo={cfg['tol_endo']}  n_endo={cfg['n_endo']} endo_price={cfg['endo_price']}g\n"
          f"  entry_rec={entry_rec}   device={device}", flush=True)

    # ---- selfchecks BEFORE the paid setup, INSIDE AN RNG SANDBOX --------------------- #
    # `entry_recorder_check` builds models and samples pools, and reseeds to 0 to do it
    # deterministically — so run bare it leaves the global torch/numpy streams somewhere else
    # than `torch.manual_seed(train_seed)` put them, and `build_shared` then trains a DIFFERENT
    # substrate from the one `cs_s0` trained. That is not a fidelity nicety: the anchor is this
    # round's full-scale G-F carrier against `cs_s0/spiral_route`, and a moved stream voids it.
    # Caught in flight on the first launch (stale buffer 0.1028 against `cs_s0`'s 0.1269, same
    # config and seed). The states are snapshotted and restored around the checks, and the
    # restore is ASSERTED rather than assumed.
    _rng0 = _rng_snapshot()
    _ref_draw = torch.randn(4).tolist()
    _rng_restore(_rng0)
    erc = entry_recorder_check(v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
                               max_level=cfg["max_macro_level"])
    print(f"[erc] entry-recorder check PASS: {json.dumps(erc, cls=NumpyEncoder)}", flush=True)
    gy = gy_soundness(s=cfg["s"], depth=cfg["depth"], level=cfg["gy_level"], eras=ers)
    print(f"[gy] soundness PASS: {json.dumps(gy, cls=NumpyEncoder)}", flush=True)
    # [conductor] the round's own pre-setup gates, in the same RNG sandbox. All three are pure
    # and cost nothing; running them here means a bad floor or a leaked label fails in seconds
    # rather than after a paid setup.
    pg = PO.policy_gate(verbose=False)
    assert pg["ALL"], f"policy gate FAILED: {pg}"
    print(f"[pol] policy gate PASS: {sum(1 for k in pg if k != 'ALL')} checks "
          f"({', '.join(k for k in pg if k != 'ALL')})", flush=True)
    eg = endo_gate(s=cfg["s"], depth=cfg["depth"], eras=ers)
    print(f"[endo] label-free-by-construction PASS: {json.dumps(eg, cls=NumpyEncoder)}",
          flush=True)
    # [audiation] GATE B, in the same sandbox: the instrumented beams against the DONOR
    # MODULE's, recorder off and on. This is the cheap half of the fidelity story; the
    # expensive half is the cross-tag replay of `cd_s0/anchor` the reduction runs afterwards.
    bg = beam_gate(v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
                   max_level=cfg["max_macro_level"])
    print(f"[beam] instrumented beams == conductor's, rec off AND on: "
          f"{json.dumps(bg, cls=NumpyEncoder)}", flush=True)
    _rng_restore(_rng0)
    _got_draw = torch.randn(4).tolist()
    _rng_restore(_rng0)
    assert _ref_draw == _got_draw, (
        f"the pre-setup selfchecks are not RNG-neutral: {_ref_draw} != {_got_draw} — "
        f"`build_shared` would train a different substrate and the cross-tag G-F would be void")
    print(f"[rng] selfchecks are RNG-neutral (next draw unchanged: "
          f"{[round(x, 6) for x in _got_draw]})", flush=True)

    _install_identity_miner(cfg["mine_support"])
    _install_entry_recorder()
    t0 = time.time()
    shared = _spiral_shared(cfg, device, ers)
    t_setup = time.time() - t0

    # ---- the realistic junk pool, from the arc's own runs on the volume --------------- #
    junk, jprov = build_junk_pool(shared["truth"], cfg["max_macro_level"])
    shared["junk_pool"] = junk
    print(f"[junk] pool {jprov['split']} provenance={jprov['provenance']}", flush=True)
    for ell, ks in junk.items():
        print(f"[junk]   L{ell}: {len(ks)} mined-but-false tuples available", flush=True)

    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {shared['truth'][ell]['child'].shape[0]} entries")
    print(f"[setup] {t_setup:.0f}s  read_acc={shared['read_acc']:.4f}  "
          f"stale buffer: random-block {shared.get('stale_random_blocks'):.4f} -> "
          f"task-matched {shared.get('stale_task_matched')}", flush=True)

    # ---- [conductor] N-2: WHAT THE ENDO READ COSTS, on this GPU ----------------------- #
    # Measured after the setup, on the setup's own generator, INSIDE the RNG sandbox — the bench
    # runs forward passes only, but the sandbox is asserted anyway so a future edit that adds a
    # sampling step cannot silently move the substrate.
    _rng1 = _rng_snapshot()
    bench = endo_bench(shared, cfg, device, cfg["n_endo"])
    _rng_restore(_rng1)
    print(f"[endo] bench: one read over {bench['n']} sequences = {bench['t_endo_s']*1e3:.1f} ms "
          f"= {bench['price_g']} grounding-equivalents (charging "
          f"{cfg['endo_price']}g); a grounding is {bench['t_per_grounding_s']*1e6:.1f} us",
          flush=True)

    # ---- [conductor] THE FLOORS THAT WILL GOVERN THIS RUN ---------------------------- #
    # Enforced only for the reads that are actually DRIVEN in this tag, so a fidelity or
    # floor-measuring tag can run without an endo floor it is being run to measure.
    parsed = parse_arms(arms)
    driven_reads = {(ARMS[b].get("loop") or {}).get("read") for _, b, _ in parsed}
    fg = None
    if "endo" in driven_reads or "yield" in driven_reads or "ledger" in driven_reads:
        if "endo" not in driven_reads:
            cfg = {**cfg, "tol_endo": cfg["tol_endo"] or 1.0}   # not read; keep the gate honest
        fg = floor_gate(cfg)
        print(f"[floor] {json.dumps(fg, cls=NumpyEncoder)}", flush=True)

    t0 = time.time()
    refs = measure_refs(shared, cfg, ers, device)
    t_refs = time.time() - t0
    plant = plant_probe(shared["generator0"], shared, cfg, device)
    print(f"[refs] {t_refs:.0f}s {json.dumps(refs, cls=NumpyEncoder)}", flush=True)

    # [assay] SUBSTRATE IDENTITY against a reference tag. A tag whose arms will be compared
    # against another tag's arms is only comparable if the two trained the SAME substrate.
    # Asserted rather than assumed: this is exactly the failure that voided the first `as_s0`
    # attempt, caught there only because the stale-buffer number happened to be printed.
    ref_check = None
    if ref_tag:
        rp = f"{DATA_DIR}/{REMOTE}/{ref_tag}/setup.json"
        assert os.path.isfile(rp), f"ref_tag {ref_tag} has no setup.json on the volume"
        rs = json.load(open(rp))
        same_refs = all(np.allclose(np.asarray(refs[k], float),
                                    np.asarray(rs["refs"][k], float))
                        for k in refs if k != "macro_true")
        ref_check = {"ref_tag": ref_tag, "refs_identical": bool(same_refs),
                     "stale_random_blocks": [shared.get("stale_random_blocks"),
                                             rs.get("stale_random_blocks")],
                     "stale_task_matched": [shared.get("stale_task_matched"),
                                            rs.get("stale_task_matched")],
                     "read_acc": [float(shared["read_acc"]), rs.get("read_acc")]}
        assert same_refs, f"refs differ from {ref_tag}: not the same substrate"
        assert (shared.get("stale_random_blocks") == rs.get("stale_random_blocks")
                and shared.get("stale_task_matched") == rs.get("stale_task_matched")), \
            f"stale value buffer differs from {ref_tag}: not the same substrate — {ref_check}"
        print(f"[ref] substrate identical to {ref_tag}: {json.dumps(ref_check)}", flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    # [audiation] the dataset carries its own description.
    with open(os.path.join(outdir, "schema.md"), "w") as fh:
        fh.write(SCHEMA_MD)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs, "plant": plant,
                   "ref_check": ref_check, "beam_gate": bg,
                   "gy_soundness": gy, "entry_recorder_check": erc,
                   "junk_pool": {str(k): [list(x) for x in v] for k, v in junk.items()},
                   "junk_pool_provenance": jprov,
                   "true_tables": {str(ell): {
                       "n": int(shared["truth"][ell]["child"].shape[0]),
                       "flat": shared["truth"][ell]["flat"].tolist()}
                       for ell in range(2, cfg["max_macro_level"] + 1)},
                   "read_acc": float(shared["read_acc"]),
                   "stale_random_blocks": shared.get("stale_random_blocks"),
                   "stale_task_matched": shared.get("stale_task_matched"),
                   "t_setup_s": t_setup, "t_refs_s": t_refs,
                   # [conductor] the round's own record: the gates, the bench, and — most
                   # importantly — the DEAD ZONES THAT GOVERNED THIS RUN with their provenance,
                   # written into the run rather than left in a script.
                   "policy_gate": pg, "endo_gate": eg, "endo_bench": bench, "floor_gate": fg,
                   "floors": {"ledger": cfg["tol_ledger"], "yield_L3": cfg["tol_yield_l3"],
                              "yield_L4": cfg["tol_yield_l4"], "endo": cfg["tol_endo"],
                              "provenance": MEASURED_FLOORS["provenance"]},
                   "era_caps": list(cfg["era_caps"]), "cap_total": cap_total,
                   "total_cycles_per_arm": total_cycles}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    summary = {"arms": {}, "order": [], "cycle_seconds": {}}
    # [conductor] the yoke handoff (`census_run`'s mechanic, widened to both actions): a gauge
    # arm's REALISED action cycles are collected as it finishes and passed into its yoke, which
    # replays them by clock. Not hard-coded, and asserted present — a yoke with an empty plan
    # would silently be an arm that never acts.
    measured_plans = {}
    # [audiation] `--log-arms` restricts the phase-1 instrument to named arms. Empty = every
    # arm, which is `au_s0`'s behaviour and the default.
    _log_set = {a.strip() for a in log_arms.split(",") if a.strip()}
    for label, base, ov in parse_arms(arms):
        if _log_set:
            ov = {**ov, "log_decisions": label in _log_set}
        lspec = ARMS[base].get("loop") or {}
        if lspec.get("kind") == "yoke":
            src = lspec["of"]
            assert src in measured_plans, (
                f"{label} yokes onto {src}, which has not run yet — arm ORDER is load-bearing "
                f"(see CONDUCTOR_ARMS)")
            ov = {**ov, "yoke_plan": json.dumps(measured_plans[src])}
            print(f"[yoke]  {label} replays {src}'s {len(measured_plans[src])} actions "
                  f"by clock: {[(p['kind'], p['cycle']) for p in measured_plans[src]]}",
                  flush=True)
        print(f"\n===== arm {label} (base {base}, overrides "
              f"{ {k: (v if k != 'yoke_plan' else '<plan>') for k, v in ov.items()} }) =====",
              flush=True)
        t0 = time.time()
        r = run_arm(label, base, ov, shared, cfg, ers, refs, outdir, device)
        n = len(r["log"]["cycle"])
        measured_plans[label] = [{"cycle": a["cycle"], "kind": a["kind"], "level": a["level"]}
                                 for a in r.get("loop_actions", [])
                                 if not a.get("cancelled")]
        summary["order"].append(label)
        summary["cycle_seconds"][label] = (time.time() - t0) / max(n, 1)
        summary["arms"][label] = {
            "n_cycles": n, "shadow_cert": r["shadow_cert"],
            "gy_final": r["gy_final"], "gauge_hist": r["gauge_hist"],
            "surgery": ARMS[base].get("cfg", {}).get("surgery"),
            "stream_burn": r.get("stream_burn"),
            # [conductor] the round's headline per-arm readouts, in the summary so a partial run
            # is still readable without pulling every `results.json`.
            "loop": r.get("loop"), "loop_actions": r.get("loop_actions"),
            "ledger": r.get("ledger"), "obs_hist": r.get("obs_hist"),
            "obs_gy_agree": r.get("obs_gy_agree"),
            "advances": [e for e in r["events"] if e["kind"] == "advance"],
            "commits": [e for e in r["events"] if e["kind"] == "commit"],
            "surgeries": [{k: q for k, q in e.items() if k != "flat_after"}
                          for e in r["events"] if e["kind"] == "surgery"],
            "n_recerts": sum(1 for e in r["events"] if e["kind"] == "recert"),
            "e_by_era": {}}
        for j in range(len(ers)):
            idx = [i for i, q in enumerate(r["log"]["era"]) if q == j + 1]
            if idx:
                summary["arms"][label]["e_by_era"][str(j + 1)] = {
                    "first": r["log"]["e"][idx[0]], "last": r["log"]["e"][idx[-1]],
                    "min": min(r["log"]["e"][i] for i in idx),
                    "mean": float(np.mean([r["log"]["e"][i] for i in idx]))}
        cs = r["shadow_cert"]
        ent = r["log"]["entry"][-1] if r["log"].get("entry") else {}
        beam = (ent.get("hist") or {}).get("beam", {})
        lp = r.get("loop") or {}
        print(f"[arm] {label}: {n} cycles at {summary['cycle_seconds'][label]:.1f} s/cycle; "
              f"cert L2 c{cs['2'].get('fired')} L3 c{cs.get('3', {}).get('fired')}; "
              f"loop={lp.get('policy')}/{lp.get('read')} "
              f"commits@{lp.get('commit_cycles')} L{lp.get('commit_levels')} "
              f"advances@{lp.get('advance_cycles')} "
              f"chosen={lp.get('n_chosen')} capped={lp.get('n_capped')} "
              f"spend={(r.get('ledger') or {}).get('spend_g')}g; "
              f"last-cycle beam entry selections "
              f"{ {k: int(sum(v)) for k, v in beam.items()} }", flush=True)
        with open(os.path.join(outdir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    summary["elapsed_s"] = time.time() - started
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s "
          f"({(time.time() - started) / 3600:.2f} GPU-h) -> {outdir}", flush=True)
    return {"refs": refs, "elapsed": time.time() - started, "summary": summary}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_ladder(tag: str = "call_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
               budget: int = 4, n_ref: int = 512, g_budgets: str = "33,58,86,100,108",
               widths: str = "1,2,3,4,8,16", max_macro_level: int = 3, quick: bool = False):
    """CALIBRATION 1 — the depth ladder, before any loop is built.

    Answers, per era and on identical held-out instances: how hard is level-k damage for the
    BASE move set at each beam width, what does handing over the TRUE level vocabulary buy,
    what does one true macro action buy, and where the exact-DP floor is. This is what picks
    the declared per-solve grounding budget `g_budget`: it must be a regime where depth is
    unaffordable to base moves and affordable with the vocabulary, otherwise the round's
    headline is a property of a budget chosen by hand."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_rt=128)
    ers = parse_eras(eras)
    ws = [int(x) for x in widths.split(",")]
    gs = [int(x) for x in g_budgets.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    base_ms = shared["base_ms"]
    l2_ms = build_ms(base_ms, {2: shared["truth"][2]}, s, depth, device)
    true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                 for ell in range(2, max_macro_level + 1)}, s, depth, device)
    sets = (("base", base_ms), ("l2", l2_ms), ("true", true_ms))
    out = {"config": cfg, "eras": ers, "cells": {},
           "n_moves": {t: len(x) for t, x in sets},
           "ground_table": {f"{t}_w{w}": beam_ground(len(mset), budget, w)
                            for t, mset in sets for w in ws}}
    for i, era in enumerate(ers):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np)
        cell = {"d0": float(nearest_derivation_cost(rules, x_np, r_np, s).mean()),
                "on_grammar": on_grammar_rate(x_np, shared["inverse_maps"][-1], v, s)}
        for tag_, mset in sets:
            cell[tag_] = {}
            for w in ws:
                b = beam_moves(shared["controller"], shared["generator0"], shared["value0"], x0,
                               torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                               budget=budget, beam_width=w, device=device)
                sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                cell[tag_][str(w)] = {"e": 1.0 - float(sc.mean()),
                                      "g": b["counts"]["ground"] / x0.shape[0]}
        cell["at_budget"] = {}
        for g in gs:
            row = {}
            for tag_, mset in sets:
                w = fit_width(len(mset), budget, g)
                row[tag_] = {"w": w, "g": beam_ground(len(mset), budget, w),
                             "e": cell[tag_].get(str(w), {}).get("e")}
            cell["at_budget"][str(g)] = row
        # one macro ACTION, at full and at partial vocabulary coverage. The coverage curve is
        # what decides whether the unit-LP certificate has a resolvable descent to certify:
        # if audition is flat in coverage, the round has no signal and that is worth knowing
        # before the main run rather than after it.
        cell["macro_true"], cell["coverage"] = {}, {}
        crng = np.random.default_rng(cfg["seed"] + 4242)
        for ell in range(2, max_macro_level + 1):
            span = s ** (ell - 1)
            node = (era["node"] * s ** (era["level"] - 1)) // span
            full = shared["truth"][ell]
            mv = MC.to_device(MC.make_macro(ell, node, s, full), device)
            cell["macro_true"][str(ell)] = audition_macro(
                shared["generator0"], x0.to(device), r_np, mv, rules_t, canon, depth,
                v, m, s, rules)["e"]
            row = {}
            n_all = full["child"].shape[0]
            ks = sorted({1, 2, 4, 8, max(1, n_all // 4), max(1, n_all // 2),
                         max(1, 3 * n_all // 4), n_all})
            for k in ks:
                es = []
                for _ in range(1 if k == n_all else 5):
                    keep = np.sort(crng.permutation(n_all)[:k])
                    sub = MC.make_table(ell, full["child"][keep], full["lower"], s)
                    mvs = MC.to_device(MC.make_macro(ell, node, s, sub), device)
                    es.append(audition_macro(shared["generator0"], x0.to(device), r_np, mvs,
                                             rules_t, canon, depth, v, m, s, rules)["e"])
                row[str(k)] = {"mean": float(np.mean(es)), "sd": float(np.std(es)),
                               "min": float(np.min(es)), "max": float(np.max(es)),
                               "n_entries": k, "frac": k / n_all}
            cell["coverage"][str(ell)] = row
        xo, _ = oracle_rollout(shared["generator0"], x0.to(device), r_np, base_ms, rules,
                               rules_t, canon, depth, v, m, s, budget)
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        cell["dp_floor_base"] = 1.0 - float(sc.mean())
        out["cells"][era["name"]] = cell
        print(f"\n=== era {era['name']} (d0={cell['d0']:.2f}, on_gram={cell['on_grammar']:.3f}) ===")
        for tag_, _ in sets:
            print(f"  {tag_:5s}: " + " | ".join(
                f"w{w}: e={cell[tag_][str(w)]['e']:.3f} ({cell[tag_][str(w)]['g']:.0f}g)"
                for w in ws))
        print(f"  one true macro action: " + " ".join(
            f"L{ell}={cell['macro_true'][str(ell)]:.3f}" for ell in cell["macro_true"]))
        for ell, row in cell["coverage"].items():
            print(f"  coverage L{ell} (entries: e[min,max]): " + " ".join(
                f"{k}:{c['mean']:.3f}[{c['min']:.2f},{c['max']:.2f}]"
                for k, c in sorted(row.items(), key=lambda kv: int(kv[0]))))
        print(f"  DP floor (base moves) = {cell['dp_floor_base']:.3f}")
        for g in gs:
            r = cell["at_budget"][str(g)]
            print(f"  @G={g:4d}: " + " | ".join(
                f"{t} w{r[t]['w']} ({r[t]['g']}g) e={r[t]['e']}" for t, _ in sets))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_ladder.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_ladder.json")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def cal_parse(tag: str = "calp2_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
              budget: int = 4, n_ref: int = 512, max_macro_level: int = 3,
              quick: bool = False):
    """CALIBRATION 3 — the perception bound on an earned vocabulary.

    `calr_s0` measured that the mined tables have LOW PRECISION (level-3: 10 true entries out
    of 34 mined, and `never_base` mined 100 distinct 4-tuples where the grammar has 56). The
    suspected cause is not the mining rule but the READ: the generator is trained by masked
    infilling with `mask_min=1` and had never seen a fully unmasked configuration, so parsing
    one is out of distribution.

    This isolates it. On the SAME oracle-solved configurations, the level-1 parse is taken
    three ways -- unmasked (what `calr_s0` did), with one block masked ELSEWHERE (in
    distribution, every span block still visible), and by the exact bottom inverse map (the
    privileged ceiling) -- and the resulting vocabulary is graded against the DGP's own each
    way. If the masked read recovers precision, the defect is the instrument; if it does not,
    the vocabulary is bounded by the plant and that is a substrate fact about the round."""
    import torch
    cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, max_macro_level=max_macro_level)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   value_episodes=6_000, n_train_episodes=20_000, n_rt=128)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()

    shared = build_shared(cfg, device)
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    gen, n_blocks = shared["generator0"], shared["n_blocks"]
    out = {"config": cfg, "eras": ers, "parse": {}, "cells": {}}

    # ---- block-level parse accuracy on CLEAN held-out configs, three ways ---------------
    clean = torch.from_numpy(_sample_pool(rules, cfg["n_rt"], s, cfg["seed"] + 31337)[1]).to(device)
    powers = v ** torch.arange(s, device=device)
    truth_f = shared["bottom_map"][(clean.view(clean.shape[0], n_blocks, s) * powers).sum(-1)]
    ok = truth_f >= 0
    for name, mb in (("unmasked", None), ("mask_elsewhere", 0)):
        pf = MC.parse_features(gen, clean, s=s, mask_block=mb)
        sel = ok.clone()
        if mb is not None:
            sel[:, mb] = False                       # the masked block is inferred, not read
        out["parse"][name] = float((pf[sel] == truth_f[sel]).float().mean())
    out["parse"]["ambiguous_frac"] = float((~ok).float().mean())
    print(f"[parse] clean-config block accuracy: {json.dumps(out['parse'])}", flush=True)

    # ---- the vocabulary each read yields, on oracle-solved repairs ----------------------
    for i, era in enumerate(ers):
        r_np, x_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth, v, m,
                                       seed=cfg["seed"] + 5000 + 17 * i)
        x0 = torch.from_numpy(x_np).to(device)
        xo, _ = oracle_rollout(gen, x0, r_np, shared["base_ms"], rules, rules_t, canon,
                               depth, v, m, s, budget)
        sc, _ = grade(xo.cpu().numpy(), r_np, rules, s)
        solved = xo[torch.from_numpy(sc > 0.5).to(device)]
        cell = {"n_solved": int(solved.shape[0]), "solve_rate": float(sc.mean()), "reads": {}}
        top_span = s ** (max_macro_level - 1)
        top0 = ((era["node"] * s ** (era["level"] - 1)) // top_span) * top_span
        mb = 0 if top0 > 0 else top0 + top_span
        reads = {"unmasked": MC.parse_features(gen, solved, s=s).cpu().numpy(),
                 "mask_elsewhere": MC.parse_features(gen, solved, s=s,
                                                     mask_block=mb).cpu().numpy(),
                 "exact": MC.exact_features(solved.cpu().numpy(),
                                            shared["inverse_maps"][-1], v, s)}
        for name, pf in reads.items():
            row = {}
            chain, obs = {1: MC.base_table(v)}, {}
            for ell in range(2, max_macro_level + 1):
                sp = s ** (ell - 1)
                nd = (era["node"] * s ** (era["level"] - 1)) // sp
                mn = MC.Miner(ell, s)
                seg = pf[:, nd * sp:(nd + 1) * sp]
                mn.observe(seg[(seg >= 0).all(1)])
                chain[ell] = mn.build(chain[ell - 1], cfg["mine_support"])
                obs[ell] = len(mn.counts)
            for ell in range(2, max_macro_level + 1):
                tbl = chain[ell]
                row[str(ell)] = {"n_entries": int(tbl["child"].shape[0]),
                                 "n_distinct_obs": obs[ell],
                                 **MC.grade_table(tbl, shared["truth"][ell])}
            cell["reads"][name] = row
        out["cells"][era["name"]] = cell
        print(f"\n=== era {era['name']} (oracle solve rate {cell['solve_rate']:.3f}, "
              f"{cell['n_solved']} solved) ===")
        for name, row in cell["reads"].items():
            print(f"  {name:15s}: " + " | ".join(
                f"L{ell}: {c['n_entries']:3d} entries, prec="
                f"{(c['precision'] if c['precision'] is not None else float('nan')):.3f}, "
                f"recall={c['recall']:.3f} (true {c['n_true']})" for ell, c in row.items()))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_parse.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_parse.json")
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def cal_stale(tag: str = "cals_s0", eras: str = "1:6,2:3,3:1", seed: int = 0,
              budget: int = 4, n_ref: int = 512, g_budget: int = 58,
              holdouts: str = "0,3,5", max_macro_level: int = 3, quick: bool = False):
    """CALIBRATION 4 — the plant's frontier, and the gate that keeps it one.

    `calr_s0` measured a plant that does not move: parse/infill accuracy flat to three
    decimals over 48 cycles, the true-table audition trendless, and at gen_lr 3e-4 actively
    degrading. The cause is that the fine-tuning diet is IN DISTRIBUTION -- solved
    configurations are ordinary clean derivations, and replay 0.5 is against the same corpus,
    so there is nothing to learn. Round 1 solved the same problem for the *value* by training
    it on a distribution that excluded the contexts it would be metered on; this does it for
    the *plant*.

    `plant_holdout` withholds level-2 tuples from the setup corpus while the evaluation
    instances keep coming from the unfiltered DGP. Every era's damage requires producing a
    correct level-2 tuple, so the plant's hole and the vocabulary's hole are the same hole.

    THE GATE (and it is a gate, not a knob): on instances that REQUIRE a withheld tuple,
    bootstrap solvability must be low but NONZERO. A frontier the agent can never cross is not
    a learner regime, it is exploration-gating by another name, and a ~0 reading is a
    halt-and-report rather than something to tune around."""
    import torch
    ers = parse_eras(eras)
    hs = [int(x) for x in holdouts.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    started = time.time()
    out = {"eras": ers, "g_budget": g_budget, "holdouts": {}}

    for h in hs:
        cfg = _cfg(seed=seed, budget=budget, n_rt=n_ref, g_budget=g_budget,
                   max_macro_level=max_macro_level, plant_holdout=h)
        if quick:
            cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                       reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                       n_rt=128)
        torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
        print(f"\n########## plant_holdout = {h} ##########", flush=True)
        shared = build_shared(cfg, device)
        rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
        v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
        gen, reader, hold = shared["generator0"], shared["reader"], shared["hold"]
        inv = shared["inverse_maps"][-1]
        base_ms = shared["base_ms"]
        true_ms = build_ms(base_ms, {ell: shared["truth"][ell]
                                     for ell in range(2, max_macro_level + 1)}, s, depth, device)
        sets = (("base", base_ms), ("true", true_ms))

        clean = torch.from_numpy(
            _sample_pool(rules, cfg["n_rt"], s, cfg["seed"] + 31337)[1]).to(device)
        powers = v ** torch.arange(s, device=device)
        tf = shared["bottom_map"][(clean.view(clean.shape[0], shared["n_blocks"], s)
                                   * powers).sum(-1)]
        ok = tf >= 0
        with torch.no_grad():
            rd = reader.block_logits(clean).argmax(-1)
            gp = gen.block_logits(clean).argmax(-1)
        cell0 = {"read_acc": float((rd[ok] == tf[ok]).float().mean()),
                 "generator_parse_acc": float((gp[ok] == tf[ok]).float().mean()),
                 "n_hold": len(hold), "eras": {}}
        print(f"  READ: reader={cell0['read_acc']:.4f}  "
              f"generator(unsupervised at visible positions)={cell0['generator_parse_acc']:.4f}",
              flush=True)

        for i, era in enumerate(ers):
            r_np, x_np, c_np = context_instances(rules, era_ctx(era), cfg["n_rt"], s, depth,
                                                 v, m, seed=cfg["seed"] + 5000 + 17 * i,
                                                 with_clean=True)
            exc = needs_excluded(c_np, era, hold, inv, v, s)
            x0 = torch.from_numpy(x_np)
            cell = {"frac_excluded": float(exc.mean()), "policies": {}}
            for tag_, mset in sets:
                w = fit_width(len(mset), budget, g_budget)
                b = beam_moves(shared["controller"], gen, shared["value0"], x0,
                               torch.from_numpy(r_np), mset, rules_t, canon, depth, v, m, s,
                               budget=budget, beam_width=w, device=device)
                sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
                cell["policies"][tag_] = {
                    "w": w, "g": b["counts"]["ground"] / x0.shape[0],
                    "e_all": 1.0 - float(sc.mean()),
                    "solve_excluded": float(sc[exc].mean()) if exc.any() else None,
                    "solve_rest": float(sc[~exc].mean()) if (~exc).any() else None}
            # the vocabulary the READER now yields, from exact-DP-solved repairs
            xo, _ = oracle_rollout(gen, x0.to(device), r_np, base_ms, rules, rules_t, canon,
                                   depth, v, m, s, budget)
            so, _ = grade(xo.cpu().numpy(), r_np, rules, s)
            solved = xo[torch.from_numpy(so > 0.5).to(device)]
            cell["oracle_solve_rate"] = float(so.mean())
            cell["vocab"] = {}
            if solved.shape[0]:
                pf = MC.parse_features(reader, solved, s=s).cpu().numpy()
                chain = {1: MC.base_table(v)}
                for ell in range(2, max_macro_level + 1):
                    sp = s ** (ell - 1)
                    nd = (era["node"] * s ** (era["level"] - 1)) // sp
                    mn = MC.Miner(ell, s)
                    seg = pf[:, nd * sp:(nd + 1) * sp]
                    mn.observe(seg[(seg >= 0).all(1)])
                    chain[ell] = mn.build(chain[ell - 1], cfg["mine_support"])
                    cell["vocab"][str(ell)] = {
                        "n_entries": int(chain[ell]["child"].shape[0]),
                        **MC.grade_table(chain[ell], shared["truth"][ell])}
            cell0["eras"][era["name"]] = cell
            pol = cell["policies"]
            print(f"  era {era['name']}: frac_excluded={cell['frac_excluded']:.3f}  "
                  f"oracle_solve={cell['oracle_solve_rate']:.3f}")
            for tag_ in ("base", "true"):
                p = pol[tag_]
                se = p["solve_excluded"]
                print(f"    {tag_:5s} w{p['w']} ({p['g']:.0f}g): e_all={p['e_all']:.3f}  "
                      f"SOLVE|excluded={'n/a' if se is None else f'{se:.3f}'}  "
                      f"solve|rest={p['solve_rest']:.3f}")
            for ell, c in cell["vocab"].items():
                print(f"    vocab L{ell}: {c['n_entries']} entries prec="
                      f"{(c['precision'] if c['precision'] is not None else float('nan')):.3f} "
                      f"recall={c['recall']:.3f}")
        out["holdouts"][str(h)] = cell0

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    out["elapsed"] = time.time() - started
    with open(os.path.join(outdir, "cal_stale.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nDONE in {out['elapsed']:.0f}s -> {outdir}/cal_stale.json")
    return out


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

def selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=256, max_level=3,
              eras="1:6,2:3,3:1"):     # [spiral] the ladder is a parameter now (depth 6)
    """C-M — the macro operator with the DGP's OWN table is BIT-IDENTICAL to the true level
    move (`units.apply_move`) at every level. This is what makes `given` and `practice_*` the
    same machinery differing only in which tuples are in the table, and therefore what makes
    the earned-vs-given fraction a statement about vocabulary rather than about operators.
    C-R — the ratchet constraint bites: a level-3 table built over a truncated level-2 table
    cannot contain entries whose halves were dropped.
    G-D — the nested damage cells are on-grammar at every level and leave d* > 0 often enough
    that rejection sampling is cheap."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    n_blocks = length // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2, root_conditioned=False).to(device)
    gen.eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(_corrupt(leaves_np, n_blocks, 3, v, s, rng)).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    out = {"table_sizes": {str(k): int(t["child"].shape[0]) for k, t in truth.items()}}

    # C-M
    cm = {}
    for ell in range(2, max_level + 1):
        ok = True
        for j in range(s ** (depth - ell)):
            true_mv = {"level": ell, "node": j, "blk0": j * s ** (ell - 1),
                       "span": s ** (ell - 1), "name": f"L{ell}n{j}"}
            a = apply_move(gen, x, true_mv, rules_t, canon, depth, v, m, s)
            mac = MC.to_device(MC.make_macro(ell, j, s, truth[ell]), device)
            b = MC.apply_any(gen, x, mac, rules_t, canon, depth, v, m, s)
            ok &= bool(torch.equal(a, b))
        cm[f"L{ell}"] = ok
    out["CM_macro_equals_true_move"] = cm
    assert all(cm.values()), f"C-M failed: {cm}"

    # C-R: truncate T2, rebuild T3 over it, and check the drop
    mn = MC.Miner(3, s)
    flat3 = truth[3]["flat"]
    mn.observe(np.concatenate([flat3] * 3))
    full = mn.build(truth[2], support=1)
    trunc = MC.make_table(2, truth[2]["child"][:6], MC.base_table(v), s)
    part = mn.build(trunc, support=1)
    out["CR_ratchet"] = {"t3_over_full_t2": int(full["child"].shape[0]),
                         "t3_over_truncated_t2": int(part["child"].shape[0]),
                         "t2_full": int(truth[2]["child"].shape[0]), "t2_trunc": 6}
    assert part["child"].shape[0] < full["child"].shape[0], "C-R failed: truncation did not bite"

    # G-D
    gd = {}
    for era in parse_eras(eras):
        dmg = corrupt_hier(leaves_np, rules, depth, v, m, s, era["level"], [era["node"]],
                           np.random.default_rng(7))
        d = nearest_derivation_cost(rules, dmg, roots_np, s)
        gd[era["name"]] = {"on_grammar": on_grammar_rate(dmg, inv[-1], v, s),
                           "dstar": float(d.mean()), "dstar_zero": float((d == 0).mean())}
        assert gd[era["name"]]["on_grammar"] == 1.0, f"G-D failed at {era['name']}"
    out["GD_damage"] = gd

    # grounding table — what the declared budget buys each action set
    out["ground"] = {f"n{n_}_w{w}": beam_ground(n_, 4, w)
                     for n_ in (8, 12, 14) for w in (1, 2, 3, 4)}
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, timeout=1800, memory=16384)
def selfcheck_remote():
    return selfcheck()


# --------------------------------------------------------------------------- #
# gates for the PORT itself (no training; the fidelity gate's other half)
# --------------------------------------------------------------------------- #

def full_selfcheck(v=8, s=2, depth=4, m=2, rule_seed=0, n=64, max_level=3, budget=3,
                   price_budget=4, price_g=58):   # [spiral] pricing table is a parameter now
    """P-1 .. P-7. The properties the fidelity and twin gates rest on.

    P-1  `select_moves` at k = n_moves returns `arange(n_moves)` for EVERY row, including rows
         of exact ties (a zero-init head gives exactly that) — the tie-ordering hazard the spec
         names, closed by a stable sort rather than by `topk`.
    P-2  `expand_selected` at k = n_moves is BIT-IDENTICAL to the enumerated
         `stack([apply_any(flat, ms[j]) for j], 1)`.
    P-3  `beam_moves_prop` at k = n_moves reproduces `beam_moves` bit-for-bit — same final
         states, same tips, same move sequences, same materialisation and grounding counts
         apart from the one root encode the port pays for and declares.
    P-4  building the head consumes NONE of the global torch stream (the twin gate's
         precondition): the random draw after `build_head` equals the draw without it.
    P-5  the head is exactly uniform at init (zero-init output layer), so a freshly committed
         macro enters the ranking neutrally rather than at an arbitrary trained logit.
    P-6  `select_moves` at k < n_moves returns exactly the k highest-logit moves, ascending,
         and with `forced` returns k + |forced| including every forced index.
    P-7  the (level, node) slot layout is injective over the full 14-move action set, and the
         pricing table `fit_width_k` the k ladder rests on (printed, not asserted).

    And the COMPOSITION's own gates, which are what this phase adds:

    C-1  `PlainExecutor` is `macros.apply_any` on every move, and a non-firing `SpanExecutor`
         is too — so threading an executor through the beam changes nothing by itself.
    C-2  THE COMPOSED FIDELITY GATE: the proposal beam at k = n_moves with a span executor
         present but closed reproduces the enumerating beam bit-for-bit (x, seq, tips, traj,
         materialisation count), the only delta being the declared root encode.
    C-3  an OPEN span slot rewrites exactly the macro's own token positions and nothing else,
         through the composed beam's own executor.
    C-4  minting BOTH heads leaves the shared torch stream exactly where it was — the twin
         gate's precondition, now for two new RNG consumers rather than one."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    length = s ** depth
    n_blocks = length // s
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    rules_t = [torch.from_numpy(np.ascontiguousarray(r)).to(device) for r in rules]
    roots_np, leaves_np = _sample_pool(rules, n, s, 3)
    gen = _build_generator()(v, length, s, 96, n_head=4, n_layer=2,
                             root_conditioned=False).to(device)
    ctrl = _build_rich_controller()(v, length, s, 96, n_head=4, n_layer=2).to(device)
    val = _build_value_head()(96, v).to(device)
    for mod in (gen, ctrl, val):
        mod.eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(_corrupt(leaves_np, n_blocks, 3, v, s, rng)).to(device)
    roots = torch.from_numpy(roots_np).to(device)
    truth = MC.true_tables(rules, depth, s, v, m, max_level)
    ms = build_ms(build_move_set(depth, s, max_level=1),
                  {ell: truth[ell] for ell in range(2, max_level + 1)}, s, depth, device)
    offsets, n_slots = PN.slot_layout(depth, s, max_level)
    slots = PN.move_slots(ms, offsets)
    out = {"n_moves": len(ms), "n_slots": n_slots, "slots": slots}

    # P-7 (layout)
    assert len(set(slots)) == len(slots) == len(ms), "slot layout is not injective"
    out["P7_slots_injective"] = True
    # [spiral] priced at the caller's declared budget rather than depth 4's hard-coded (4, 58),
    # and over the action-set sizes THIS grammar actually produces, so the table is readable at
    # depth 6 (32 base / 48 after L2 / 56 after L3) as well as at depth 4 (8 / 12 / 14).
    pb, pg = price_budget, price_g
    ns_ = sorted({len(build_move_set(depth, s, max_level=lv))
                  for lv in range(1, max_level + 1)})
    out["P7_price"] = {f"k{k}": {"w": PN.fit_width_k(k, pb, pg),
                                 "g": PN.beam_ground_k(k, pb, PN.fit_width_k(k, pb, pg)) + 1}
                       for k in (1, 2, 3, 4, 8, 12, 14)}
    out["P7_price_enum"] = {f"n{n_}": {"w": fit_width(n_, pb, pg),
                                       "g": beam_ground(n_, pb, fit_width(n_, pb, pg))}
                            for n_ in ns_}

    # P-1 / P-6 (selection)
    nm = len(ms)
    ties = torch.zeros(n, nm, device=device)
    rnd = torch.randn(n, nm, device=device)
    ar = torch.arange(nm, device=device)[None, :].expand(n, -1)
    out["P1_ties_k_eq_n"] = bool(torch.equal(PN.select_moves(ties, nm), ar))
    out["P1_rand_k_eq_n"] = bool(torch.equal(PN.select_moves(rnd, nm), ar))
    assert out["P1_ties_k_eq_n"] and out["P1_rand_k_eq_n"], "P-1 failed"
    sel4 = PN.select_moves(rnd, 4)
    want = torch.sort(torch.topk(rnd, 4, dim=1).indices, dim=1).values
    out["P6_topk_matches"] = bool(torch.equal(sel4, want))
    selF = PN.select_moves(rnd, 2, forced=[8, 9])
    out["P6_forced_shape"] = list(selF.shape)
    out["P6_forced_contains"] = bool(((selF == 8).any(1) & (selF == 9).any(1)).all())
    assert out["P6_topk_matches"] and out["P6_forced_contains"], "P-6 failed"
    assert selF.shape[1] == 4, "P-6: forced expansion is not k + |forced| wide"

    # P-2 (expansion)
    flat = x
    enum_children = torch.stack(
        [MC.apply_any(gen, flat, ms[j], rules_t, canon, depth, v, m, s) for j in range(nm)],
        dim=1)
    sel_all = PN.select_moves(torch.zeros(flat.shape[0], nm, device=device), nm)
    got, n_mat = PN.expand_selected(gen, flat, sel_all, ms, MC.apply_any, rules_t, canon,
                                    depth, v, m, s)
    out["P2_expand_maxabs"] = float((got.long() - enum_children.long()).abs().max())
    out["P2_n_mat"] = int(n_mat)
    assert out["P2_expand_maxabs"] == 0.0, "P-2 failed"

    # P-4 / P-5 (the head costs no shared randomness, and is uniform at init)
    torch.manual_seed(1234)
    ref = torch.randn(8)
    torch.manual_seed(1234)
    head = PN.build_head(96, v, n_slots, seed=99, device=device)
    got_draw = torch.randn(8)
    out["P4_stream_untouched"] = bool(torch.equal(ref, got_draw))
    assert out["P4_stream_untouched"], "P-4 failed: the head consumed the shared stream"
    with torch.no_grad():
        z = _encode_chunked(ctrl, x)
        lg = head(z, roots)
    out["P5_init_logit_maxabs"] = float(lg.abs().max())
    assert out["P5_init_logit_maxabs"] == 0.0, "P-5 failed"

    # P-3 (the beam)
    a = beam_moves(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                   budget=budget, beam_width=4, device=device, collect=True)
    b = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                        budget=budget, beam_width=4, device=device, collect=True,
                        prop=head, slots=slots, k=nm, forced=(), n_slots=n_slots)
    out["P3_x_equal"] = bool(torch.equal(a["x"], b["x"]))
    out["P3_seq_equal"] = bool(torch.equal(a["seq"], b["seq"]))
    out["P3_tips_equal"] = bool(torch.equal(a["tips_x"], b["tips_x"]))
    out["P3_traj_equal"] = bool(all(torch.equal(p, q) for p, q in zip(a["traj"], b["traj"])))
    out["P3_counts_enum"] = a["counts"]
    out["P3_counts_prop"] = b["counts"]
    out["P3_extra_ground"] = int(b["counts"]["ground"] - a["counts"]["ground"])
    out["P3_mat_equal"] = bool(a["counts"]["mat"] == b["counts"]["mat"])
    assert all(out[k_] for k_ in ("P3_x_equal", "P3_seq_equal", "P3_tips_equal",
                                  "P3_traj_equal", "P3_mat_equal")), "P-3 failed"
    assert out["P3_extra_ground"] == x.shape[0], "P-3: the root encode is mispriced"
    # ---- C-4: BOTH heads cost the shared stream nothing --------------------------------
    torch.manual_seed(4321)
    ref2 = torch.randn(6)
    torch.manual_seed(4321)
    head1 = PN.build_head(96, v, n_slots, seed=7, device=device)
    shead = SN.build_head(SN.slot_count(s, depth, max_level), v, 96, s ** (max_level - 1),
                          seed=11, device=device)
    out["C4_stream_untouched"] = bool(torch.equal(ref2, torch.randn(6)))
    assert out["C4_stream_untouched"], "C-4 failed: a head consumed the shared stream"

    # ---- C-1: the executor seam is a no-op by itself ------------------------------------
    macros_l = [mv for mv in ms if mv.get("kind") == "macro"]
    slots_c = {SN.slot_key(mv["level"], mv["node"]):
               {"id": SN.slot_index(mv["level"], mv["node"], s, depth, max_level),
                "open": False, "move": mv, "level": mv["level"], "node": mv["node"]}
               for mv in macros_l}
    plain = SN.PlainExecutor()
    sx = SN.SpanExecutor(gen, shead, slots_c, cap=4096, hold_cap=1024, hold_frac=0.1,
                         per_call=32, seed=3, v=v, length=length)
    worst = 0.0
    for mv in ms:
        with torch.no_grad():
            a_ = MC.apply_any(gen, x, mv, rules_t, canon, depth, v, m, s)
            b_ = plain.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
            c_ = sx.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
        worst = max(worst, float((a_ - b_).abs().max()), float((a_ - c_).abs().max()))
    out["C1_executor_seam_maxabs"] = worst
    assert worst == 0.0, "C-1 failed: the executor seam is not a no-op"

    # ---- C-2: THE COMPOSED FIDELITY GATE ------------------------------------------------
    sx.capture = False
    e_ = beam_moves(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                    budget=budget, beam_width=4, device=device, collect=True,
                    ex=SN.PlainExecutor())
    f_ = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                         budget=budget, beam_width=4, device=device, collect=True,
                         prop=head1, slots=slots, k=nm, forced=(), n_slots=n_slots, ex=sx)
    out["C2_x_equal"] = bool(torch.equal(e_["x"], f_["x"]))
    out["C2_seq_equal"] = bool(torch.equal(e_["seq"], f_["seq"]))
    out["C2_tips_equal"] = bool(torch.equal(e_["tips_x"], f_["tips_x"]))
    out["C2_traj_equal"] = bool(all(torch.equal(p_, q_) for p_, q_ in zip(e_["traj"], f_["traj"])))
    out["C2_mat_equal"] = bool(e_["counts"]["mat"] == f_["counts"]["mat"])
    out["C2_extra_ground"] = int(f_["counts"]["ground"] - e_["counts"]["ground"])
    assert all(out[k_] for k_ in ("C2_x_equal", "C2_seq_equal", "C2_tips_equal",
                                  "C2_traj_equal", "C2_mat_equal")), "C-2 failed"
    assert out["C2_extra_ground"] == x.shape[0], "C-2: the root encode is mispriced"

    # ---- C-3: an OPEN slot writes only its own span, through the composed beam ----------
    for k_ in slots_c:
        slots_c[k_]["open"] = True
    sx.fire = True
    c3 = {}
    for mv in macros_l:
        pos = SN.span_positions(mv, x.shape[0], s, device)
        with torch.no_grad():
            got = sx.apply(gen, x, mv, rules_t, canon, depth, v, m, s)
        keepm = torch.ones_like(x, dtype=torch.bool).scatter_(1, pos, False)
        c3[mv["name"]] = {"outside_maxabs": float((got - x)[keepm].abs().max()),
                          "inside_changed": float((got != x).gather(1, pos).float().mean())}
    out["C3_open_slot"] = c3
    out["C3_mat_head"] = int(sx.counts["mat_head"])
    assert all(cc["outside_maxabs"] == 0.0 for cc in c3.values()), "C-3 failed"
    assert out["C3_mat_head"] > 0, "C-3: the head never fired"

    # ---- C-5: THE PRACTICE-PATH CALL SHAPE ----------------------------------------------
    # C-2 exercises the composed beam only at k = n_moves with `forced` and `explore` empty,
    # which is the METERING path. The practice beam runs a different branch — k < n_moves,
    # exploration on — and that branch is where a local name collided with the executor
    # parameter and threw at the first materialisation of the second step. A gate that only
    # covers the metering shape is not a gate on the composition, so it is widened here.
    sx.fire = False
    sx.capture = True
    c5 = {}
    for kk_, exp_, forced_ in ((2, 1, ()), (2, 1, (8, 9)), (4, 0, ()), (1, 1, ())):
        sx.reset()
        g_ = beam_moves_prop(ctrl, gen, val, x, roots, ms, rules_t, canon, depth, v, m, s,
                             budget=budget, beam_width=4, device=device, collect=True,
                             prop=head1, slots=slots, k=kk_, forced=forced_, n_slots=n_slots,
                             explore=exp_, erng=np.random.default_rng(7), ex=sx)
        tot = sum(sx.counts[q] for q in ("mat_base", "mat_dp", "mat_head"))
        c5[f"k{kk_}_e{exp_}_f{len(forced_)}"] = {
            "x_shape": list(g_["x"].shape), "mat": int(g_["counts"]["mat"]),
            "executor_mat": int(tot), "agree": bool(tot == g_["counts"]["mat"]),
            "seq_width": int(g_["seq"].shape[0])}
        assert list(g_["x"].shape) == list(x.shape), "C-5: wrong output shape"
        assert tot == g_["counts"]["mat"], "C-5: beam and executor disagree on materialisations"
        assert len(g_["traj"]) == budget + 1, "C-5: trajectory collection broke"
    out["C5_practice_shapes"] = c5
    sx.fire = True

    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, gpu="L4", timeout=1800, memory=16384)
def full_selfcheck_remote():
    return full_selfcheck()


# =========================================================================== #
# [spiral] THE GATE SET FOR THIS ROUND
# =========================================================================== #

def tall_gates(eras=SPIRAL_ERAS, max_level=3, n=256, rule_seed=0):
    """`tall/tall.py::gates`' T-1 .. T-4, re-derived here (not imported: `tall` imports
    `ear`/`recital`, and this file deliberately depends on neither).

      T-1  the ladder NESTS at all five levels and every node index is in range
           (`s**(depth-l)` nodes at level `l`)
      T-2  damage is 100% on-grammar at EVERY ladder level, with the rejection acceptance rate
           the port pays
      T-3  the oracle's repair distance is MONOTONE in ladder depth — the cost-to-depth the
           whole arc measures — reported with its gradient
      T-4  the action set and its grounding price over the 32..62 move range, at the budget
           this round declares

    Cheap and CPU-only; `tall` reported all four passing at depth 6, m=2 and this re-runs them
    because a re-run costs seconds and an assumption costs a launch."""
    v, s, depth, m = DEPTH6["v"], DEPTH6["s"], DEPTH6["depth"], DEPTH6["m"]
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inv = build_inverse_maps(rules)[-1]
    ers = parse_eras(eras)
    out = {"eras": eras, "depth": depth, "m": m}

    t1 = []
    for i, era in enumerate(ers):
        n_nodes = s ** (depth - era["level"])
        ok_range = 0 <= era["node"] < n_nodes
        nested = True
        if i + 1 < len(ers):
            nxt = ers[i + 1]
            nested = (era["node"] * s ** (era["level"] - 1)) // s ** (nxt["level"] - 1) \
                == nxt["node"]
        t1.append({"era": era["name"], "n_nodes_at_level": n_nodes,
                   "node_in_range": bool(ok_range), "nests_into_next": bool(nested)})
        assert ok_range, f"T-1 failed: {era['name']} node out of range (max {n_nodes - 1})"
        assert nested, f"T-1 failed: {era['name']} does not nest into the next era"
    out["T1_ladder"] = t1

    roots, leaves = _sample_pool(rules, n, s, rule_seed + 31337)
    t23 = {}
    for era in ers:
        rng = np.random.default_rng(7)
        dmg = corrupt_hier(leaves.copy(), rules, depth, v, m, s, era["level"],
                           [era["node"]], rng)
        og = on_grammar_rate(dmg, inv, v, s)
        d = nearest_derivation_cost(rules, dmg, roots, s)
        brk = d > 0
        t23[era["name"]] = {"on_grammar": og, "accept_rate": float(brk.mean()),
                            "dstar_mean": float(d[brk].mean()),
                            "dstar_max": int(d[brk].max()),
                            "p_dstar_gt_budget": {str(b): float((d[brk] > b).mean())
                                                  for b in (4, 6, 8, 10)}}
        assert og == 1.0, f"T-2 failed: damage off-grammar at {era['name']}"
    out["T23_damage"] = t23
    ds = [t23[e["name"]]["dstar_mean"] for e in ers]
    out["T3_depth_gradient"] = float(max(ds) / min(ds))
    assert all(b >= a for a, b in zip(ds, ds[1:])), \
        f"T-3 failed: repair distance is not monotone in depth: {ds}"

    sizes = {f"max_level={lv}": len(build_move_set(depth, s, max_level=lv))
             for lv in range(1, depth)}
    out["T4_action_set"] = sizes
    out["T4_pricing"] = {f"n{n_}_b{b}_w{w}": beam_ground(n_, b, w)
                         for n_ in sorted(set(sizes.values()))
                         for b in (4, 8) for w in (1, 2)}
    # THE PRICING INVERSION, and what routing does to it — the round's central affordance,
    # printed rather than argued. Under enumeration the width the declared budget buys drops
    # when the action set grows at commit; under routing the width depends on k, not on |ms|.
    g = 482
    out["T4_width_enum_at_G482"] = {f"n{n_}": fit_width(n_, 8, g)
                                    for n_ in sorted(set(sizes.values()))}
    out["T4_width_route_at_G482"] = {f"k{k}": PN.fit_width_k(k, 8, g)
                                     for k in (1, 2, 4, 8, 20, 32)}
    out["T4_g_to_hold_w2_enum"] = {f"n{n_}": beam_ground(n_, 8, 2)
                                   for n_ in sorted(set(sizes.values()))}
    return out


@app.function(image=image, timeout=3600, memory=32768)
def gates_remote(eras: str = SPIRAL_ERAS, max_level: int = 3, n: int = 256):
    """CPU gates: the arc's C-M / C-R / G-D at depth 6, plus `tall`'s T-1..T-4."""
    out = {"inherited": selfcheck(v=DEPTH6["v"], s=DEPTH6["s"], depth=DEPTH6["depth"],
                                  m=DEPTH6["m"], n=n, max_level=max_level, eras=eras),
           "tall": tall_gates(eras=eras, max_level=max_level, n=n)}
    print("\n=== SPIRAL GATES (depth 6, m=2) ===")
    print(json.dumps(out["tall"], indent=2, cls=NumpyEncoder))
    return out


@app.function(image=image, gpu="L4", timeout=1800, memory=16384)
def gates_d6_remote(max_level: int = 3, n: int = 64, budget: int = 3):
    """GPU gates: `native`'s OWN fidelity gates (P-1..P-7, C-1..C-5) re-run ON THIS SUBSTRATE.

    P-3 and C-2 are the two that matter for this round's licence to read anything: the proposal
    beam at k = n_moves reproduces the enumerating beam BIT-FOR-BIT (same x, seq, tips, traj,
    materialisation count; the only delta the one declared root encode), with a span executor
    present but closed. `native` asserted them at depth 4 over a 14-move action set; here the
    action set is 56 moves at 64 tokens, which is a different code path in every batched op."""
    out = full_selfcheck(v=DEPTH6["v"], s=DEPTH6["s"], depth=DEPTH6["depth"], m=DEPTH6["m"],
                         n=n, max_level=max_level, budget=budget,
                         price_budget=8, price_g=482)
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=32768)
def preflight(cycles: int = 2, eras: str = "1:25:6,2:12:6,3:6:5,4:3:5,5:1:5",
              max_macro_level: int = 3, budget: int = 2):
    """`tall/tall.py::preflight`'s idiom: build a TINY depth-6 substrate and call every
    function `phase_a` calls, in the order it calls them, so interface drift fails in ~2
    minutes instead of after a ~510 s paid setup (which is what it cost `tall`, twice).

    The substrate here is trained for a few dozen steps and is scientifically worthless. The
    point is solely that every call SIGNATURE and every `shared[...]` access resolves, that the
    transplanted commit policy and recert actually fire, and that both ports run at 64 tokens.
    It takes a GPU because beam search over a 32-56 move action set at 64 tokens is
    minutes-per-cycle on CPU; `budget=2` keeps the beam shallow, since it is the move set and
    not the rollout depth that exercises the interface."""
    import torch
    # the default ladder here carries PER-ERA cycle counts (the 3-field form), so the
    # per-era-length path Phase B depends on is exercised rather than assumed
    cfg = _d6_cfg(max_macro_level=max_macro_level, budget=budget, g_budget=482,
                  era_cycles=cycles, probe_every=1, probe_widths=(1,),
                  seed=0, rule_seed=0, train_seed=1,
                  # deliberately tiny — an interface test, not an experiment
                  controller_steps=40, generator_steps=40, value_steps=40, reader_steps=40,
                  value_episodes=256, n_train_episodes=2000, value_batch_collect=128,
                  n_pr=8, n_rt=32, n_score=32, n_aud=16, n_probe_clean=32,
                  gen_steps=2, prop_warmup=0, prop_steps=2, prop_buf_cap=2000,
                  recert_every=1, sil_min_cycle=1, checkpoint_every=10 ** 9,
                  span_min_hold=4, span_tau=0.0, span_capture=8, mine_cap=8,
                  tm_episodes=256, preflight_seed_miner=True,
                  # [census] the gate / extend / yoke paths must all FIRE at toy sizes
                  gate_W=1, gate_theta=0, extend_cap=3, extend_tol=1.0, yoke_cycle=2,
                  # [conductor] the LOOP's paths must all fire too — a chosen commit, a chosen
                  # advance, a capped advance and a yoked replay of each. The block geometry is
                  # collapsed (W=1, burn=1) and the dead zones set to float dust so `quiet`
                  # fires as soon as a gauge stops improving, which at six-cycle eras it will.
                  # SAID OUT LOUD: preflight checks the CODE PATH and never a number — these
                  # tolerances are not measured floors and no reading from this run is one.
                  loop_W=2, loop_burn=1, loop_span=1,
                  tol_ledger=1e-9, tol_yield_l3=1e-9, tol_yield_l4=1e-9, tol_endo=1e-9,
                  n_endo=32, endo_price=3, era_caps=(6, 6, 5, 5, 5), total_cap=0,
                  gy_level=4, preflight_seed_frac=0.5, preflight_seed_topup_cycle=4,
                  entry_rec=True)
    ers = parse_eras(eras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    ok = {}
    _install_identity_miner(cfg["mine_support"])
    _install_entry_recorder()                      # [assay] the instrument is ON in preflight
    ok["entry_recorder_check"] = entry_recorder_check(
        v=cfg["v"], s=cfg["s"], depth=cfg["depth"], m=cfg["m"],
        max_level=cfg["max_macro_level"])
    shared = _spiral_shared(cfg, device, ers)
    junk, jprov = build_junk_pool(shared["truth"], cfg["max_macro_level"])
    shared["junk_pool"] = junk
    ok["junk_pool"] = {**jprov["split"], "provenance": jprov["provenance"]}
    ok["_spiral_shared"] = sorted(shared.keys())
    refs = measure_refs(shared, cfg, ers, device)
    ok["measure_refs"] = bool(refs)
    ok["plant_probe"] = plant_probe(shared["generator0"], shared, cfg, device)
    ok["read_acc"] = float(shared["read_acc"])
    ok["stale_buffer_terminal_success"] = float(shared["replay"]["y"].mean())
    outdir = f"{DATA_DIR}/{REMOTE}/_preflight"
    os.makedirs(outdir, exist_ok=True)
    # every arm SHAPE this round runs: enum, routed, composed, the fidelity twin, the ceiling
    # [census] the instrument's soundness is checked BEFORE anything is paid for
    ok["gy_soundness"] = gy_soundness(s=cfg["s"], depth=cfg["depth"], level=cfg["gy_level"],
                                      eras=ers)
    # [conductor] the pure gates, before any arm runs
    ok["policy_gate"] = PO.policy_gate(verbose=False)
    assert ok["policy_gate"]["ALL"], ok["policy_gate"]
    ok["endo_gate"] = endo_gate(s=cfg["s"], depth=cfg["depth"], eras=ers)
    ok["endo_bench"] = endo_bench(shared, cfg, device, min(cfg["n_endo"], 64), reps=2)
    # checked against the DEFAULT (measured) floors, not preflight's dust — the gate's job is
    # that `MEASURED_FLOORS` and `floors.json` agree, and preflight's tolerances are neither.
    ok["floor_gate"] = floor_gate({**_d6_cfg(), "tol_endo": 1.0})
    ok["floor_gate"]["note"] = ("checked on the MEASURED floors; this preflight itself runs on "
                                "1e-9 dust so the loop's branches fire at toy sizes")
    # every arm the MAIN run carries, plus the donor shapes the fidelity gates rest on. The
    # loop arms run here at toy sizes so every branch of the new commit/advance dispatch — a
    # driven commit, a driven advance, a capped advance, and a yoked replay of all three — is
    # executed before a paid setup, which is what `preflight` is for.
    _plans = {}
    for arm in ("anchor", "outer_yield", "yoked_yield", "outer_endo", "yoked_endo",
                "outer_ledger", "complete", "exact", "junk_dose", "strip", "given_c1",
                "spiral_route", "given_native"):
        _ov = {}
        _ls = ARMS[arm].get("loop") or {}
        if _ls.get("kind") == "yoke":
            _ov["yoke_plan"] = json.dumps(_plans.get(_ls["of"], []))
        # a driven endo arm needs a floor; at preflight scale there is no measured one, so a
        # placeholder is used and SAID SO — preflight checks the code path, never a number.
        _cfg_a = ({**cfg, "tol_endo": 1e-3, "_preflight_placeholder_endo_tol": True}
                  if _ls.get("read") == "endo" and not cfg["tol_endo"] else cfg)
        _r = run_arm(arm, arm, _ov, shared, _cfg_a, ers, refs, outdir, device)
        _plans[arm] = [{"cycle": a["cycle"], "kind": a["kind"], "level": a["level"]}
                       for a in _r.get("loop_actions", []) if not a.get("cancelled")]
        res = json.load(open(f"{outdir}/{arm}/results.json"))
        ok[f"run_arm:{arm}"] = {
            "cycles": len(res["log"]["cycle"]),
            "loop": res.get("loop"), "loop_actions": res.get("loop_actions"),
            "ledger": res.get("ledger"), "obs_gy_agree": res.get("obs_gy_agree"),
            "panel_last": (res["log"]["panel"][-1] if res["log"].get("panel") else None),
            "g_budget": res["config"]["g_budget"],
            "commits": [(e["level"], e["cycle"], e.get("provisional"),
                         e.get("n_moves_before"), e.get("n_moves_after"),
                         e.get("width_before"), e.get("width_after"),
                         e.get("k_eff_before"), e.get("k_eff_after"))
                        for e in res["events"] if e["kind"] == "commit"],
            "recerts": sum(1 for e in res["events"] if e["kind"] == "recert"),
            "shadow_cert": {k: q["fired"] for k, q in res["shadow_cert"].items()},
            "surgery": [{k: q for k, q in e.items() if k not in ("flat_after",
                                                                 "true_mask_after")}
                        for e in res["events"] if e["kind"] == "surgery"],
            "entry_beam_last": {k: int(sum(v)) for k, v in
                                ((res["log"]["entry"][-1].get("hist") or {}).get("beam", {})
                                 or {}).items()},
            "gy": {"distinct": (res.get("gy_final") or {}).get("n_distinct"),
                   "n_obs": (res.get("gy_final") or {}).get("n_obs"),
                   "at_sup": ((res.get("gy_final") or {}).get("n_at_support") or {}).get("3")},
            "extends": sum(1 for e in res["events"] if e["kind"] == "extend"),
            "entries_added": sum(e.get("n_admitted", 0) for e in res["events"]
                                 if e["kind"] == "extend"),
            "committed_grade_last": res["log"]["committed_grade"][-1],
            "identity": bool(res["log"]["miner"][-1].get("2", {}).get("keys_at_support")
                             is not None)}
    # the instruments must APPEAR, not merely fail to crash. [census] retargeted: the
    # spiral-shaped assertions now read `census_extend` (this round's provisional-committing,
    # earning arm) and the span assertion reads `given_native`, the only arm here that carries
    # a corridor — the rest are routing-only per the finding-9 quarantine.
    sp = json.load(open(f"{outdir}/anchor/results.json"))
    assert any(e["kind"] == "commit" and e.get("provisional")
               for e in sp["events"]), "the provisional commit path never fired"
    assert all(c.get("2", {}).get("keys_at_support") is not None
               for c in sp["log"]["miner"]), "the entry-identity instrument did not reach the log"
    assert sp["log"]["committed_grade"], "the committed-table grade did not reach the log"
    assert set(sp["shadow_cert"]) == {"2", "3"}, "the shadow certificate is not per level"
    gn = json.load(open(f"{outdir}/given_native/results.json"))
    assert any(c["n_open"] > 0 for c in gn["log"]["span"]), \
        "no span slot ever opened in given_native — the corridor path is untested"
    ok["span_open_max"] = max(int(c["n_open"]) for c in gn["log"]["span"])
    ok["task_matched_buffer"] = {"random_blocks": shared.get("stale_random_blocks"),
                                 "task_matched": shared.get("stale_task_matched")}
    assert shared.get("stale_task_matched") is not None, \
        "task-matched collection did not run"

    # ---- [assay] the surgery and the instrument must have EXECUTED, not merely not crashed
    for mode in ("complete", "exact", "junk_dose", "strip"):
        rj = json.load(open(f"{outdir}/{mode}/results.json"))
        sv = [e for e in rj["events"] if e["kind"] == "surgery"]
        assert sv, f"{mode}: no surgery event"
        assert {e["level"] for e in sv} == set(range(2, cfg["max_macro_level"] + 1)), \
            f"{mode}: surgery did not fire at every level ({[e['level'] for e in sv]})"
        for e in sv:
            assert e["mode"] == mode, f"{mode}: wrong mode recorded"
        ok[f"surgery:{mode}"] = [
            {k: e.get(k) for k in ("level", "cycle", "n_own", "n_own_true", "n_added",
                                   "n_added_true", "n_removed", "K", "n_after",
                                   "tab_recall", "tab_precision", "dose_shortfall")}
            for e in sv]
    # each mode must do what its name says
    def _sv(mode, lv):
        rj = json.load(open(f"{outdir}/{mode}/results.json"))
        return next(e for e in rj["events"] if e["kind"] == "surgery" and e["level"] == lv)
    for lv in range(2, cfg["max_macro_level"] + 1):
        st_, cp_, ex_, jd_ = (_sv(m_, lv) for m_ in ("strip", "complete", "exact",
                                                     "junk_dose"))
        assert st_["n_added"] == 0, "strip added entries"
        assert st_["tab_precision"] in (None, 1.0), \
            f"strip L{lv} left junk behind: precision {st_['tab_precision']}"
        assert cp_["n_added"] == cp_["n_added_true"], "complete added a non-true entry"
        assert ex_["n_after"] == int(shared["truth"][lv]["child"].shape[0]), \
            "exact did not install the full true table"
        assert jd_["n_added_true"] == 0, "junk_dose added a TRUE entry"
        assert jd_["K"] == cp_["K"] or lv > 2, \
            f"the dose is not matched at L{lv}: {jd_['K']} vs {cp_['K']}"
        ok[f"dose_match_L{lv}"] = {"complete_K": cp_["K"], "junk_dose_added": jd_["n_added"],
                                   "shortfall": jd_.get("dose_shortfall")}
    # `exact` and `given_c1` must hold IDENTICAL content and differ only in arrival
    ex_end = json.load(open(f"{outdir}/exact/results.json"))["log"]["committed_grade"][-1]
    g1_end = json.load(open(f"{outdir}/given_c1/results.json"))["log"]["committed_grade"][-1]
    assert ex_end == g1_end, f"exact and given_c1 differ in content: {ex_end} vs {g1_end}"
    ok["exact_vs_given_c1_content"] = ex_end

    # THE INSTRUMENT: entry selections must be recorded, at both levels, in the beam phase
    an = json.load(open(f"{outdir}/anchor/results.json"))
    ent = [q for q in an["log"]["entry"] if (q.get("hist") or {}).get("beam")]
    assert ent, "the entry-identity instrument recorded nothing in the beam phase"
    lvs = sorted({k for q in ent for k in q["hist"]["beam"]})
    assert set(lvs) & {"2"}, f"no L2 entry selections recorded (levels seen: {lvs})"
    assert all("true_mask" in q for q in ent), "the truth mask did not reach the log"
    ok["entry_instrument"] = {
        "cycles_with_beam_selections": len(ent), "levels": lvs,
        "total_beam_selections": int(sum(sum(v) for q in ent
                                         for v in q["hist"]["beam"].values())),
        "phases": sorted({p for q in an["log"]["entry"] for p in (q.get("hist") or {})})}

    # THE TWIN CLAIM WITH THE INSTRUMENT ON: a treated arm must be bit-identical to the anchor
    # up to its first surgery. This is the assertion the SPEC asks for by name.
    SER_ = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
            "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
    ok["twins_with_instrument"] = {}
    for mode in ("complete", "exact", "junk_dose", "strip"):
        rj = json.load(open(f"{outdir}/{mode}/results.json"))
        c1_ = min(e["cycle"] for e in rj["events"] if e["kind"] == "surgery")
        w = c1_ - 1
        d = 0.0
        for k in SER_:
            x = np.asarray(an["log"][k][:w], float)
            y = np.asarray(rj["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        ok["twins_with_instrument"][mode] = {"first_surgery": c1_, "window": w,
                                             "max_abs_delta": d}
        assert d == 0.0, (f"{mode} diverges from the anchor BEFORE its first surgery "
                          f"(c1-c{w}, max|delta| = {d:.3e}) — the instrument or the surgery "
                          f"is not behaviour-neutral")
    # ---- [conductor] THE LOOP'S OWN IN-TAG ASSERTIONS ----------------------------------
    # The donor's stream-burn check is dropped here (its twins `given_c1_j`/`exact_j` are
    # `assay`'s arms and this round does not run them); what replaces it is the pair of
    # assertions this round actually rests on.
    #
    #   (i)  THE READS ARE NON-INVASIVE. Every loop arm shares the anchor's stream, so it must
    #        be BIT-IDENTICAL to the anchor up to its OWN first action. A divergence before
    #        then would mean the shadow panel, the observation panel or the endo forward is
    #        perturbing the trajectory, which would void every comparison in the round.
    #   (ii) THE YOKE REPLAYS EXACTLY. A yoked arm must reproduce its gauge arm's realised
    #        actions cycle for cycle. At preflight scale this is a mechanic check, not the
    #        finding — the finding is what `analyze_conductor.py` §2 measures at full scale.
    ok["loop_twins"] = {}
    an = json.load(open(f"{outdir}/anchor/results.json"))
    for arm_ in ("outer_yield", "yoked_yield", "outer_endo", "yoked_endo", "outer_ledger"):
        rj = json.load(open(f"{outdir}/{arm_}/results.json"))
        acts = [a["cycle"] for a in (rj.get("loop_actions") or [])]
        w = (min(acts) - 1) if acts else len(rj["log"]["cycle"])
        d = 0.0
        for k in [q for q in SER_ if q != "t_cum"]:       # t_cum carries the priced ledger
            x = np.asarray(an["log"][k][:w], float)
            y = np.asarray(rj["log"][k][:w], float)
            q = min(len(x), len(y))
            d = max(d, float(np.abs(x[:q] - y[:q]).max()) if q else 0.0)
        ok["loop_twins"][arm_] = {"first_action": (min(acts) if acts else None),
                                  "window": w, "max_abs_delta": d,
                                  "spend_g": (rj.get("ledger") or {}).get("spend_g")}
        assert d == 0.0, (f"{arm_} diverges from the anchor BEFORE its first action "
                          f"(c1-c{w}, max|delta| = {d:.3e}) — a read is not inert")
    ok["yoke_replay"] = {}
    for yk_, src_ in (("yoked_yield", "outer_yield"), ("yoked_endo", "outer_endo")):
        a_ = json.load(open(f"{outdir}/{src_}/results.json"))
        b_ = json.load(open(f"{outdir}/{yk_}/results.json"))
        pa = [(x["kind"], x["cycle"], x.get("level")) for x in (a_.get("loop_actions") or [])
              if not x.get("cancelled")]
        pb = [(x["kind"], x["cycle"], x.get("level")) for x in (b_.get("loop_actions") or [])]
        dd = 0.0
        n_ = min(len(a_["log"]["cycle"]), len(b_["log"]["cycle"]))
        for k in [q for q in SER_ if q != "t_cum"]:
            dd = max(dd, float(np.abs(np.asarray(a_["log"][k][:n_], float)
                                      - np.asarray(b_["log"][k][:n_], float)).max()))
        ok["yoke_replay"][yk_] = {"of": src_, "gauge_actions": pa, "yoke_actions": pb,
                                  "actions_equal": pa == pb, "n_cycles": n_,
                                  "max_abs_delta": dd,
                                  "t_cum_gauge": a_["log"]["t_cum"][-1],
                                  "t_cum_yoke": b_["log"]["t_cum"][-1]}
        assert pa == pb, f"{yk_} did not replay {src_}'s actions: {pb} != {pa}"
    ok["gy_observed"] = {a_: json.load(open(f"{outdir}/{a_}/results.json"))["gy_final"]["n_obs"]
                         for a_ in ("anchor", "outer_yield", "outer_endo", "outer_ledger")}
    ok["obs_gy_agree"] = {a_: json.load(open(f"{outdir}/{a_}/results.json"))["obs_gy_agree"]
                          for a_ in ("anchor", "outer_yield", "outer_endo", "outer_ledger")}
    print("\n=== PREFLIGHT OK — every call resolved at the depth-6 grammar ===")
    print(json.dumps({k: q for k, q in ok.items() if k != "_spiral_shared"},
                     indent=2, cls=NumpyEncoder))
    print(f"shared keys: {ok['_spiral_shared']}")
    return ok


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=5400, memory=32768)
def fidelity_d4(tag: str = "fid_d4", cycles: int = 5, arms: str = "given,practice_late",
                seed: int = 0):
    """G-F — THE DONOR-FIDELITY GATE. With the depth-6 transplant switched OFF (depth 4, m=2,
    the donor's own budget and ladder), does THIS file's `run_arm` reproduce
    `native/full/full.py`'s, bit for bit?

    Run IN PROCESS against the imported donor rather than by diffing two volume tags
    (`critic/fidelity_check.py`'s cross-tag form), for two reasons: it needs no prior donor tag
    to exist at a matched config, and it removes cross-run hardware from the comparison so a
    failure is unambiguously the fork's. The donor's `run_arm` is called with the shared dict
    THIS file built, so the substrate is literally the same object.

    A SELF-REPLAY CONTROL is carried: the donor is run twice, and its own arm-to-arm delta is
    reported beside the fork's. Without it a non-zero delta is uninterpretable — GPU
    nondeterminism and a real fork bug look identical."""
    import torch
    from rhm.practice.native.full import full as FL

    cfg = _cfg(seed=seed, era_cycles=cycles, budget=4, g_budget=58, max_macro_level=3,
               mine_from="chosen", mine_cap=8, probe_every=2, probe_widths=(1, 2, 4),
               controller_steps=800, generator_steps=800, value_steps=800, reader_steps=600,
               value_episodes=6_000, n_train_episodes=20_000, n_pr=24, n_rt=96, n_score=96,
               n_probe_clean=128, checkpoint_every=10 ** 9, gen_steps=5,
               sil_min_cycle=2, early_offset=1, late_offset=1,
               prop_warmup=1, prop_steps=12, prop_buf_cap=20_000,
               span_min_hold=32, span_tau=0.0)
    ers = parse_eras("1:6,2:3,3:1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[fidelity_d4] depth={cfg['depth']} m={cfg['m']} arms={arms} cycles={cycles}",
          flush=True)

    # the donor's own builder, so even the SUBSTRATE is not this fork's work
    shared = FL.build_shared(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    refs = FL.measure_refs(shared, cfg, ers, device)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
              "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
    rows, worst_fork, worst_ctrl = [], 0.0, 0.0
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        a1 = FL.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor1", device)
        a2 = FL.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor2", device)
        b = run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/fork", device)
        row = {"arm": arm, "fork": {}, "control": {}}
        for k in SERIES:
            x, y, z = (np.asarray(a1["log"][k], float), np.asarray(b["log"][k], float),
                       np.asarray(a2["log"][k], float))
            L = min(len(x), len(y), len(z))
            row["fork"][k] = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            row["control"][k] = float(np.abs(x[:L] - z[:L]).max()) if L else float("nan")
            worst_fork = max(worst_fork, row["fork"][k])
            worst_ctrl = max(worst_ctrl, row["control"][k])
        ev = lambda r: [(e["era"], e["level"], e["cycle"], e["n_entries"])
                        for e in r["events"] if e["kind"] == "commit"]
        row["events_equal"] = bool(ev(a1) == ev(b))
        row["n_cycles"] = len(a1["log"]["e"])
        rows.append(row)
        print(f"[fidelity_d4] {arm}: max|fork-donor| = {max(row['fork'].values()):.3e}  "
              f"max|donor-donor| = {max(row['control'].values()):.3e}  "
              f"commit events equal: {row['events_equal']}", flush=True)

    out = {"rows": rows, "worst_fork": worst_fork, "worst_donor_self_replay": worst_ctrl,
           "events_equal": all(r["events_equal"] for r in rows),
           "seconds": time.time() - started}
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[fidelity_d4] === G-F ===  max|fork - donor| = {worst_fork:.3e}   "
          f"max|donor - donor| (control) = {worst_ctrl:.3e}", flush=True)
    assert out["events_equal"], "G-F FAILED: commit events differ from the donor"
    assert worst_fork <= worst_ctrl, (
        f"G-F FAILED: the fork perturbed the donor's code path "
        f"({worst_fork:.3e} > self-replay control {worst_ctrl:.3e})")
    print("G-F PASS — with the transplant off, the fork is a replay of the donor.")
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def fidelity_smoke(tag: str = "gf_smoke", cycles: int = 4, seed: int = 0):
    """G-F, layer 1 — WITH THE SURGERY OFF, does this file's `run_arm` reproduce
    `census.py`'s, bit for bit, on the depth-6 substrate, WITH the entry instrument installed?

    In process against the imported donor (`critic/fidelity_check.py`'s idiom, as used for the
    spiral's own G-F): the donor's `run_arm` is called with the shared dict THIS file built, so
    the substrate is literally the same object and no cross-run hardware enters the comparison.
    A DONOR SELF-REPLAY control is carried, because without it a non-zero delta is
    uninterpretable — GPU nondeterminism and a real fork bug look identical.

    SCALE, on the record: this runs at SMOKE sizes (a few hundred training steps, `cycles`
    cycles), not at `sp_s0`'s configuration. A full-scale in-process replay would mean paying
    for three 116-cycle depth-6 arms plus a ~630 s setup — about 1.5 GPU-h to re-derive a
    number the in-tag anchor gives away free. The FULL-SCALE half of G-F is therefore the
    cross-tag one, in `analyze_census.py`: this round's in-tag `spiral_route` against
    `sp_s0`'s, over the 32 cycles the two ladders share. Era 1 is 48 cycles here against 32
    there, but the only things `ec` reaches are the loop bound, `at_boundary` (c18's certified
    commit precedes both), and the era-end probe trigger (which fires at c32 either way, and
    probes consume neither the shared torch stream nor any arm RNG). So c1-c32 is a genuine
    bit-for-bit window at the real configuration, for free.
    """
    import torch
    # [conductor] retargeted to the DIRECT donor. With the loop off this file must replay
    # `assay.py` bit for bit, and it must do so WITH THE SHADOW PANEL RUNNING — the panel reads
    # every candidate gauge in every arm, including this one, so the gate is simultaneously the
    # fork's fidelity check and `endo_yield`'s transparency gate T (the panel moves nothing).
    from rhm.practice.assay import assay as SP

    cfg = _d6_cfg(era_cycles=cycles, seed=seed, probe_every=2, probe_widths=(1, 2),
                  controller_steps=800, generator_steps=800, value_steps=800, reader_steps=600,
                  value_episodes=6_000, n_train_episodes=20_000, n_pr=24, n_rt=96, n_score=96,
                  n_aud=48, n_probe_clean=128, checkpoint_every=10 ** 9, gen_steps=5,
                  sil_min_cycle=2, prop_warmup=1, prop_steps=12, prop_buf_cap=20_000,
                  tm_episodes=512, mine_cap=8)
    ers = parse_eras("1:25,2:12,3:6")
    for e_ in ers:
        e_["cycles"] = cycles
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[gf] depth={cfg['depth']} m={cfg['m']} cycles/era={cycles} — conductor vs assay "
          f"(shadow panel ON in the fork)", flush=True)

    _install_identity_miner(cfg["mine_support"])
    # [assay] the instrument is INSTALLED for this gate. It patches `MC.macro_features`, which
    # the donor's own code path also calls, so both sides of the comparison run through the
    # recording copy — which is the point: the gate must show the instrument is neutral.
    _install_entry_recorder()
    shared = _spiral_shared(cfg, device, ers)
    refs = measure_refs(shared, cfg, ers, device)
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)

    SERIES = ["e", "succ", "dres", "t_cum", "n_moves", "width", "g_per_solve", "e_practice",
              "vloss", "gloss", "n_solved", "n_mined", "m_per_solve"]
    rows, worst_fork, worst_ctrl = [], 0.0, 0.0
    for arm in ("anchor", "given_c1"):
        a1 = SP.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor1", device)
        a2 = SP.run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/donor2", device)
        b = run_arm(arm, arm, {}, shared, cfg, ers, refs, f"{outdir}/fork", device)
        row = {"arm": arm, "fork": {}, "control": {}}
        for k in SERIES:
            x, y, z = (np.asarray(a1["log"][k], float), np.asarray(b["log"][k], float),
                       np.asarray(a2["log"][k], float))
            L = min(len(x), len(y), len(z))
            row["fork"][k] = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            row["control"][k] = float(np.abs(x[:L] - z[:L]).max()) if L else float("nan")
            worst_fork = max(worst_fork, row["fork"][k])
            worst_ctrl = max(worst_ctrl, row["control"][k])
        ev = lambda r: [(e["era"], e["level"], e["cycle"], e["n_entries"])
                        for e in r["events"] if e["kind"] == "commit"]
        row["events_equal"] = bool(ev(a1) == ev(b))
        rows.append(row)
        print(f"[gf] {arm}: max|fork-donor| = {max(row['fork'].values()):.3e}  "
              f"max|donor-donor| = {max(row['control'].values()):.3e}  "
              f"commits equal: {row['events_equal']}", flush=True)

    out = {"scale": "smoke", "rows": rows, "worst_fork": worst_fork,
           "worst_donor_self_replay": worst_ctrl,
           "events_equal": all(r["events_equal"] for r in rows),
           "seconds": time.time() - started}
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[gf] === G-F (smoke scale) === max|fork - spiral| = {worst_fork:.3e}   "
          f"control = {worst_ctrl:.3e}", flush=True)
    assert out["events_equal"], "G-F FAILED: commit events differ from assay.py"
    assert worst_fork <= worst_ctrl, (
        f"G-F FAILED: the conductor fork perturbed assay.py's code path "
        f"({worst_fork:.3e} > self-replay control {worst_ctrl:.3e}) — the outer loop's reads, "
        f"the observation panel or the era-loop rewrite is not transparent")
    print("G-F PASS — with the loop off and the shadow panel ON, this fork replays assay.py.")
    return out


# =========================================================================== #
# [spiral] PHASE A — the paid depth-6 feasibility block
# =========================================================================== #

def _slope(y):
    """least-squares slope per cycle over a series (the descent rate)."""
    y = np.asarray(y, float)
    if y.size < 2:
        return float("nan")
    x = np.arange(y.size, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _descent(log, refs, commit_cycle=None):
    """`tall/dens0`'s descent-gate readout, plus the decomposition that mattered more than the
    headline there: 0.233 of its 0.262 recovered fraction was already present AT CYCLE 1,
    supplied by the densified value head rather than by the loop. So the recovered fraction is
    reported at c1, over the first five and the last five cycles, and the PRE-COMMIT slope is
    reported separately — that is the part the loop is actually responsible for."""
    e = np.asarray(log["e"], float)
    st, fl = float(refs["stale"][0]), float(refs["floor"][0])
    rng_ = max(st - fl, 1e-9)
    pre = e[:commit_cycle] if commit_cycle else e
    return {"stale": st, "floor": fl, "n_cycles": int(e.size),
            "e_c1": float(e[0]), "e_first5": float(e[:5].mean()),
            "e_last5": float(e[-5:].mean()), "e_min": float(e.min()),
            "rec_c1": float((st - e[0]) / rng_),
            "rec_first5": float((st - e[:5].mean()) / rng_),
            "rec_last5": float((st - e[-5:].mean()) / rng_),
            "rec_best": float((st - e.min()) / rng_),
            "slope_all": _slope(e), "slope_pre_commit": _slope(pre),
            "n_pre_commit": int(len(pre)),
            "delta_cycle_mean": float(np.abs(np.diff(e)).mean()) if e.size > 1 else float("nan"),
            "delta_cycle_max": float(np.abs(np.diff(e)).max()) if e.size > 1 else float("nan")}


def _pricing_rows(log, commit_cycle, window=3):
    """THE COMMIT-CYCLE PRICING READOUT. `dens0` measured a +0.180 jump in `e` at the cycle
    AFTER the L2 commit — 4.6x its own mean cycle-to-cycle |delta| of 0.019 — because the
    action set grew 32 -> 48 and `fit_width` then dropped the beam from width 2 to width 1 at
    G=482. Under top-k routing the width refits under k, not |ms|, so the same commit should
    cost nothing in width. The two exceptions worth watching, both visible here: the
    `prop_new_cycles` window in which newly committed slots are FORCED into the expansion
    (k_eff jumps by the number of new nodes), and the cycles before the filter switches on."""
    if not commit_cycle:
        return []
    lo, hi = max(0, commit_cycle - window - 1), min(len(log["cycle"]), commit_cycle + window)
    out = []
    for i in range(lo, hi):
        pr = log["prop"][i] if log.get("prop") else {}
        out.append({"cycle": int(log["cycle"][i]), "n_moves": int(log["n_moves"][i]),
                    "k_eff": pr.get("k_eff"), "filter_on": pr.get("filter_on"),
                    "width": int(log["width"][i]),
                    "g_per_solve": float(log["g_per_solve"][i]),
                    "e": float(log["e"][i])})
    return out


@app.function(image=image, volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def phase_a(tag: str = "pa0",
            era: str = "1:25", cycles: int = 22, commit_at: int = 16,
            arms: str = "practice_late,practice_late_prop_k4,practice_late_native",
            fid_arms: str = "given,fid", fid_cycles: int = 5,
            budget: int = 8, g_budget: int = 482, max_macro_level: int = 3,
            n_corrupt: int = 1, mine_cap: int = 8, probe_every: int = 8,
            # n_score 512 -> 256: the shadow audition runs 5 macro applications per level per
            # cycle on this set and it is an unpriced INSTRUMENT, so it is the cheapest place
            # to buy back cycle time. `tall`'s post-mortem (probes were ~47% of `tl_s0`'s
            # runtime) is the reason this is a default rather than an afterthought.
            probe_widths: str = "1,2,4", n_pr: int = 64, n_rt: int = 384, n_score: int = 256,
            seed: int = 0, rule_seed: int = 0, train_seed: int = 1,
            task_matched: bool = True, tm_episodes: int = 8192):
    """PHASE A: the paid depth-6 block. One setup (~510 s), then everything that needs it.

    1. SETUP + REFERENCES at depth 6, m=2, `n_corrupt=1` — the densified stale value buffer is
       built by `build_shared` itself, so its terminal success is reported beside `dens0`'s
       0.133 and depth 4's 0.296 without any retraining step.
    2. THE DESCENT GATE WITH ROUTING LIVE. Era 1 only, `cycles` cycles, three arms sharing one
       torch stream: enumeration (the in-run reproduction of `dens0`'s −0.004/cycle reference),
       routing only, and both ports. Recovered fraction, its c1 decomposition, and the
       pre-commit slope — the part the loop is responsible for.
    3. THE COMMIT-CYCLE PRICING READOUT. The commit is placed at a FIXED cycle in every arm
       (`commit="late"` with `late_offset = cycles - commit_at`) rather than left to the
       certificate, precisely so the arms' pricing transitions are at the same cycle and the
       comparison is a comparison. The transplanted `delta_prov` policy is what the MAIN run
       uses; it is exercised by `preflight` and by the `spiral` arm shape, not here, because a
       certificate that fires at different cycles in different arms would confound the one
       number this block exists to read.
    4. THE IN-TAG FIDELITY TWINS AT DEPTH 6: `fid` (both ports wired, both nailed shut) against
       `given` (plain enumeration), same stream, `fid_cycles` cycles, asserted bit-for-bit.
    5. TASK-MATCHED COLLECTION DAMAGE (`task_matched`), the one untested densification lever
       `tall` named: collect the generic value buffer with damage drawn from the ERA'S OWN cell
       (`corrupt_hier` at the era's level/node) instead of `_corrupt`'s random blocks, and
       report its terminal success against `n_corrupt=1`'s. A measurement only — nothing is
       trained on it here — so Phase B can adopt it on evidence rather than on argument."""
    import torch
    cfg = _d6_cfg(era_cycles=cycles, budget=budget, g_budget=g_budget,
                  max_macro_level=max_macro_level, n_corrupt=n_corrupt, mine_cap=mine_cap,
                  probe_every=probe_every,
                  probe_widths=tuple(int(x) for x in probe_widths.split(",")),
                  n_pr=n_pr, n_rt=n_rt, n_score=n_score,
                  late_offset=max(0, cycles - commit_at),
                  # Phase A MEASURED task-matched collection and did not adopt it (adoption is
                  # Phase B's, on the strength of that measurement), and it ran at
                  # span_min_hold=256. Pinned here so `pa0` stays reproducible from this file.
                  collect_task_matched=False, span_min_hold=256,
                  sil_cv=0.10, lp_min_drop=0.05,
                  seed=seed, rule_seed=rule_seed, train_seed=train_seed)
    ers = parse_eras(era)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"[phase_a] tag={tag} depth={cfg['depth']} m={cfg['m']} "
          f"seqlen={cfg['s'] ** cfg['depth']} era={era} cycles={cycles} commit_at={commit_at} "
          f"budget={budget} g_budget={g_budget} n_corrupt={n_corrupt} arms={arms}", flush=True)

    _install_identity_miner(cfg["mine_support"])
    t0 = time.time()
    shared = _spiral_shared(cfg, device, ers)
    t_setup = time.time() - t0
    stale_succ = float(shared["replay"]["y"].mean())
    print(f"[phase_a] SETUP {t_setup:.0f}s   stale buffer terminal success = {stale_succ:.4f}"
          f"   [dens0 n_corrupt=1: 0.133 | tl_s0 config: 0.070 | depth 4: 0.296]", flush=True)

    sizes = {lv: len(build_move_set(cfg["depth"], cfg["s"], max_level=lv))
             for lv in range(1, cfg["depth"])}
    pricing = {"action_set": sizes,
               "width_enum": {n_: fit_width(n_, budget, g_budget) for n_ in sizes.values()},
               "g_enum": {n_: beam_ground(n_, budget, fit_width(n_, budget, g_budget))
                          for n_ in sizes.values()},
               "g_to_hold_w2_enum": {n_: beam_ground(n_, budget, 2) for n_ in sizes.values()},
               "width_route": {k: PN.fit_width_k(k, budget, g_budget)
                               for k in (1, 2, 4, 8, 12, 20, 32)}}
    print(f"[phase_a] pricing {json.dumps(pricing, cls=NumpyEncoder)}", flush=True)

    t0 = time.time()
    refs = measure_refs(shared, cfg, ers, device)
    t_refs = time.time() - t0
    plant = plant_probe(shared["generator0"], shared, cfg, device)
    read_acc = float(shared["read_acc"])
    span = cfg["s"] ** (max_macro_level - 1)
    print(f"[phase_a] REFS {t_refs:.0f}s stale={refs['stale']} floor={refs['floor']} "
          f"d0={refs['d0']}", flush=True)
    print(f"[phase_a] read_acc={read_acc:.4f}  compounded over the level-{max_macro_level} "
          f"span of {span} blocks: {read_acc ** span:.4f}   plant={plant}", flush=True)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/setup.json", "w") as fh:
        json.dump({"config": cfg, "eras": ers, "refs": refs, "plant": plant,
                   "read_acc": read_acc, "stale_buffer_terminal_success": stale_succ,
                   "pricing": pricing, "t_setup_s": t_setup, "t_refs_s": t_refs},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---- 2/3. the descent gate + the pricing readout ------------------------------------ #
    gate = {"descent": {}, "pricing_at_commit": {}, "cycle_seconds": {}}
    for label in [a.strip() for a in arms.split(",") if a.strip()]:
        t0 = time.time()
        r = run_arm(label, label, {}, shared, cfg, ers, refs, outdir, device)
        n = len(r["log"]["cycle"])
        gate["cycle_seconds"][label] = (time.time() - t0) / max(n, 1)
        cc = next((e["cycle"] for e in r["events"] if e["kind"] == "commit"), None)
        gate["descent"][label] = _descent(r["log"], refs, commit_cycle=cc)
        gate["descent"][label]["commit_cycle"] = cc
        gate["descent"][label]["commit"] = [
            {k: e.get(k) for k in ("cycle", "level", "n_entries", "tab_recall",
                                   "tab_precision", "provisional")}
            for e in r["events"] if e["kind"] == "commit"]
        gate["pricing_at_commit"][label] = _pricing_rows(r["log"], cc)
        d = gate["descent"][label]
        print(f"\n[phase_a] === DESCENT {label} ===  rec_c1={d['rec_c1']:.3f} "
              f"rec_last5={d['rec_last5']:.3f} rec_best={d['rec_best']:.3f} "
              f"slope_pre={d['slope_pre_commit']:+.5f}/cyc "
              f"[dens0: rec_last5 0.262, rec_c1 0.233, slope -0.004]", flush=True)
        for row in gate["pricing_at_commit"][label]:
            print(f"[phase_a]   c{row['cycle']:3d} nm={row['n_moves']:3d} "
                  f"k_eff={row['k_eff']} w={row['width']} g={row['g_per_solve']:.0f} "
                  f"e={row['e']:.4f}", flush=True)
        with open(f"{outdir}/gate.json", "w") as fh:
            json.dump(gate, fh, indent=2, cls=NumpyEncoder)
        volume.commit()

    # ---- 4. the in-tag fidelity twins at depth 6 ---------------------------------------- #
    fcfg = {**cfg, "era_cycles": fid_cycles, "checkpoint_every": 10 ** 9,
            "probe_every": 10 ** 9}
    flogs = {}
    for label in [a.strip() for a in fid_arms.split(",") if a.strip()]:
        t0 = time.time()
        flogs[label] = run_arm(label, label, {}, shared, fcfg, ers, refs,
                               f"{outdir}/fid", device)
        print(f"[phase_a] fidelity arm {label}: {time.time() - t0:.0f}s", flush=True)
    fnames = list(flogs)
    fid = {"arms": fnames, "cycles": fid_cycles}
    if len(fnames) >= 2:
        a, b = flogs[fnames[0]]["log"], flogs[fnames[1]]["log"]
        worst = 0.0
        fid["series"] = {}
        # The BEHAVIOUR series must be bit-identical. `g_per_solve` and `t_cum` must NOT: the
        # port declares one extra grounding per instance for the root encode (`native`'s C-2
        # measured exactly that), so they are reported separately and the surcharge is checked
        # to be exactly the declared one rather than asserted to zero.
        for k in ("e", "succ", "dres", "n_moves", "width", "e_practice", "vloss", "gloss",
                  "n_solved", "n_mined", "m_per_solve"):
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            L = min(len(x), len(y))
            dd = float(np.abs(x[:L] - y[:L]).max()) if L else float("nan")
            fid["series"][k] = dd
            worst = max(worst, dd)
        fid["max_abs_delta_behaviour"] = worst
        gx, gy = np.asarray(a["g_per_solve"], float), np.asarray(b["g_per_solve"], float)
        L = min(len(gx), len(gy))
        fid["root_encode_surcharge"] = (sorted(set(np.round(gy[:L] - gx[:L], 6).tolist()))
                                        if L else [])
        fid["pass"] = bool(worst == 0.0)
        print(f"[phase_a] === IN-TAG FIDELITY at depth 6: {fnames[0]} vs {fnames[1]} "
              f"behaviour max|delta| = {worst:.3e} over {fid_cycles} cycles "
              f"-> {'PASS' if fid['pass'] else 'FAIL'}; "
              f"g/solve surcharge = {fid['root_encode_surcharge']} ===", flush=True)
    gate["fidelity_d6"] = fid

    # ---- 5. task-matched collection damage (a measurement, not a treatment) ------------- #
    if task_matched:
        t0 = time.time()
        tm = _task_matched_buffer(shared, cfg, ers[0], device, n_episodes=tm_episodes)
        tm["seconds"] = time.time() - t0
        tm["reference_random_blocks"] = stale_succ
        gate["task_matched_collection"] = tm
        print(f"[phase_a] === TASK-MATCHED COLLECTION ===  terminal success "
              f"{tm['terminal_success']:.4f} against random-block {stale_succ:.4f} "
              f"[dens0 best 0.133 | depth 4 0.296]  ({tm['seconds']:.0f}s)", flush=True)

    gate["totals"] = {"t_setup_s": t_setup, "t_refs_s": t_refs,
                      "elapsed_s": time.time() - started}
    with open(f"{outdir}/gate.json", "w") as fh:
        json.dump(gate, fh, indent=2, cls=NumpyEncoder)
    with open(f"{outdir}/done.txt", "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\n[phase_a] DONE in {time.time() - started:.0f}s -> {outdir}", flush=True)
    # asserted LAST and not at the point of measurement: a gate failure must not throw away
    # the paid work that was already on the volume, but it must still fail the run visibly.
    assert gate["fidelity_d6"].get("pass", True), (
        "in-tag fidelity FAILED at depth 6: behaviour max|delta| = "
        f"{gate['fidelity_d6'].get('max_abs_delta_behaviour')}")
    return gate


def _task_matched_buffer(shared, cfg, era, device, n_episodes=8192):
    """The measurement-only wrapper (Phase A's): collect and report the rate, keep nothing."""
    vb = _collect_task_matched(shared, cfg, era, device, n_episodes=n_episodes)
    return {"terminal_success": float(vb["y"].mean()), "n_episodes": int(n_episodes),
            "n_states": int(vb["x"].shape[0]), "era": era["name"],
            "n_corrupt_equivalent": "corrupt_hier at the era's cell"}


def _collect_task_matched(shared, cfg, era, device, n_episodes=8192):
    """`collect_value_buffer` with the corruption drawn from THE TASK'S OWN damage cell.

    Same shape and same contract as the donor's collector — `{"x","r","y"}` over every state of
    every rollout, `y` the episode's terminal success broadcast down its trajectory — so it is
    a drop-in for `shared["replay"]` and for `value_steps`. The only difference is where the
    broken configurations come from: `context_instances` at this era's cell (rejection-sampled
    on d* > 0, exactly as the graded task is) instead of `_corrupt`'s random blocks.

    Its RNG is its OWN generator, so adding this instrument moves nothing else in the run."""
    import torch
    v, s, depth, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    ms = shared["base_ms"]
    rng = np.random.default_rng(cfg["train_seed"] + 606_060)
    xs, rs, ys, done = [], [], [], 0
    while done < n_episodes:
        b = min(cfg["value_batch_collect"], n_episodes - done)
        done += b
        r_np, x_np = context_instances(rules, era_ctx(era), b, s, depth, v, m,
                                       seed=int(rng.integers(0, 2 ** 31 - 1)))
        x = torch.from_numpy(x_np).to(device)
        roots = torch.from_numpy(r_np).to(device)
        traj = [x.clone()]
        for _ in range(cfg["budget"]):
            x = behavior_step(shared["controller"], shared["generator0"], x, roots, ms,
                              rules_t, canon, depth, v, m, s, cfg["explore_eps"], device, rng)
            traj.append(x.clone())
        succ, _ = grade(x.cpu().numpy(), r_np, rules, s)
        y = torch.from_numpy(succ.astype(np.float32))
        for st in traj:
            xs.append(st.cpu()); rs.append(roots.cpu()); ys.append(y)
    return {"x": torch.cat(xs), "r": torch.cat(rs), "y": torch.cat(ys)}


@app.local_entrypoint()
def main(quick: bool = True):
    audiation_run.remote(quick=quick)
