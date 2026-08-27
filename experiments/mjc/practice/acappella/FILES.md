# a cappella — file index and design record

**Up**: [`../README.md`](../README.md) (mjc/practice) · **Contract**: [`SPEC.md`](SPEC.md)
**Why this node exists**: [`../accompanist/README.md`](../accompanist/README.md) — the FM was the
incumbent's free accompanist. **Substrate donor**: [`../etude/`](../etude/README.md) (world, piece,
metering, the corrected compile op; donor untouched, gate A-F asserts bit-for-bit fidelity).
**Machinery donors**: [`../offbook/nets.py`](../offbook/nets.py) (`SlotLayout`, and π for Phase B),
[`../legato/`](../legato/README.md) (launch keying, the nesting discipline).

**Standing constraint, wider than the SPEC** (Jasper, 2026-08-27): **no forward model anywhere in
this node, not even as an extra control or reference arm**, and no FM-integration side-questions.
If the design ever seems to demand a model, that is a finding to halt and report, not a gap to
patch. Nothing in this folder imports, trains or evaluates an `f(s,u)`.

**There is no README yet, by design**: numbers get discussed with Jasper before interpretation is
written. This file is the factual record.

## Code files

| file | one line |
|---|---|
| `piece.py` | The piece and world as pure-python constants (no numpy/mujoco/torch, so a Modal *local* entrypoint can import it). Every value is asserted against `etude/etude.py`'s own source by gate A-F1. |
| `world.py` | The forked world with no forward model: the rollout pool (a grounding), the ledger, `World.traverse`, the reflex law, the priced plant-CEM, the deciders, the two-level `Library`, and `select_tapes` (étude's `select_x` on the plant). |
| `delay_gate.py` | **Phase B1**: the Δ ladder (gates B-F0/B-F1, per-Δ reflex re-fit, the 6-arm sweep, the record rung, the pre-fixed niche + ordering verdict). Modal fn `run_acappella_delay`, entrypoint `acappella_delay` (with `--spawn`). |
| `analyze_delay.py` | Pure-python reducer for a B1 run → `results/<tag>/delay_report.txt`. |
| `gates.py` | Phase A: gates A-F, A-B (+ the tempo instrument), A-I, A-C, A-R, A-S, A-D. Modal fn `run_acappella_gates`, entrypoint `acappella_gates` (with `--spawn`). |
| `analyze_gates.py` | Pure-python reducer for a Phase A run → `results/<tag>/gate_report.txt`. |
| `profile_cost.py` | What a grounding costs in wall-clock, measured at the shapes the node runs at; the budget and cycle counts are sized from it, not projected. |

## Modal volume layout

`mujoco-control-data`: `/data/practice_acappella/<tag>/gates.json` (+ `done.txt`).
Fetched copies and reports under `results/<tag>/`.

## Reproduce

```bash
cd experiments/                      # MODAL_PROFILE=chromatic
modal run mjc/practice/acappella/profile_cost.py::acappella_profile
modal run mjc/practice/acappella/gates.py::acappella_gates --quick --tag asmoke
modal run --detach mjc/practice/acappella/gates.py::acappella_gates --spawn --tag a0 --seed 0
python3 mjc/practice/acappella/analyze_gates.py --tag a0 --fetch
```

## Decisions taken, with reasons (the things a reader could disagree with)

Each row is a choice that was NOT forced by the SPEC, with the measurement or the donor precedent
that settled it.

| decision | why |
|---|---|
| **A grounding = one open-loop rollout on a resettable copy of the plant**, priced at `L*dt_ctrl + d_fb` (its execution time plus one re-grounding). | The SPEC's motor analogue of RHM's materialise-and-re-encode. The price is étude's own, generalised: `select_x` priced its audition as `n_cand*n_score*(hh*dt_ctrl + d_fb)`. A motor trial you actually perform costs the time to perform it. RHM's own `t = n_ground*d_fb + n_mat*c_mat` charged no trajectory time (an edit has no duration), so the report also gives the **fb-only** pricing column and the two are compared. |
| **No GPU.** `cpu=16.0`, no CUDA anywhere. | There is no FM to train and π is a 2-layer MLP. Every expensive thing here is MuJoCo on CPU. |
| **Rollouts run in a `ProcessPoolExecutor`, never threads.** | Measured (`profile_cost.py`, L4-free CPU box): one grounding (34 control steps, rotated world) costs **7.99 ms** single-threaded and **0.871 ms** at 16 processes (10.5× on 16 workers); **8 threads are 3× SLOWER than 1** (24.3 ms) because `_apply_rot_regions` runs per physics substep in Python and holds the GIL. A rollout has no RNG inside, so chunking across processes is bit-identical to running them in one — that is what licenses the pool. |
| **The practice renditions are made by the REFLEX LAW, not by the search.** | It is the only model-free closed-loop controller available, it costs zero groundings, and it is the biologically honest source of the traces a compile op selects from (`practice_manufactures_its_own_credit` §3: compilation is self-imitation of one's own traces). The search-sourced pool is available as a control but is ~250× more expensive to harvest. |
| **The reflex law is a PD tracking law on the waypoint schedule** — target `wp_k + (h+1)/H*(wp_{k+1}-wp_k)` with the schedule's own feedforward velocity — with gains fit ONCE by grid search on the **clean** world (no rotation region), then frozen. | SPEC: "gains calibrated once on the stale plant". The model-free analogue of étude's `pretrain_mode=exclude` is a controller that has never met the hard passage, so the rotation is *unmodelled* rather than mis-modelled. Criterion fixed before the sweep: min piece error on the clean world, held-out geometry. Smoke: kp=5, kd=0.25, clean 0.0123 → rotated 0.0150, with segment 1 (the drilled one) 0.0140 → 0.0294. |
| **The plant-CEM returns the best sequence it ACTUALLY ROLLED OUT, not the elite mean.** | étude finding 2: the mean of valid command sequences is not a valid sequence (2.1×), and BC's regression compounds it (1.5×). A real-rollout search has no reason to average when it can select. The elite mean's realised error is logged next to the best sample's as an instrument in A-B, so the donor's convention stays visible. |
| **The budget G is PER DECISION, and the tempo R is measured as a second axis rather than fixed.** | The SPEC declares a per-decision budget. But "the widest search its own action set affords at G" has a second dimension on a plant — how often to decide — and étude's own metering ladder was `R ∈ {1,4,12,34}`. A-B* sweeps R at matched groundings/traversal, plus a reactive rung (R=1 at absolute budgets) which is the model-free analogue of étude's `never`. The budget RULE reads only the R=H ladder. |
| **G\* rule, declared before the run**: the smallest G on the declared ladder whose reference piece error is within **5%** of the best piece error anywhere on the ladder. | `legato/` F2's discipline (calibrate to preserve the axis, never to make an arm win) and `ratchet/`'s criterion in continuous form: at G\* the incumbent has bought essentially everything more search can buy it, so no reading can be dismissed as the incumbent having been starved. The rule mentions only the incumbent's own saturation and is evaluated when **no library exists**, so it cannot be tuned to make a library win. The ladder around G\* is reported every time (`ratchet/` §"the declared budget sets the headline"). |
| **Ladder to G = 4096.** | étude's own `never` spent `k_shoot=256 × cem_iters=4 = 1024` rollouts per decision, in imagination, at zero price. A plant-rollout planner needs the same order to reach the same place, so a ladder stopping at 512 would measure a starved incumbent. Smoke confirmed the shape: G=8/16/32 → 0.795/0.656/0.586. |
| **The commit trigger is a fixed sequential schedule, not étude's δ-silence certificate.** | With no forward model nothing on the plant side learns, so the incumbent's error is constant over cycles and a mastery detector has nothing to detect. `legato/`'s `--commit-seg` schedule is the precedent. |
| **The library is built NESTED** — candidates for seam k are harvested in the configuration with seams < k already committed and keyed. | `offbook/` d2: the nesting, not the audition op, is what carries legato's content (per-state oracle 0.87–1.16× with it vs 4× worse without). `legato/`'s own commit events show 142/144 candidates came from the already-committed configuration. |
| **Chain cells are grown over segment-slot spellings, with a reflex-sourced pool as the declared control.** | `legato/` F5: the lower level's library funds the upper level's addressable variation (audition-estimated 1.03× vs closed-loop 1.34×). `presto/` found F5 did not pay at 120 ms seams, so the control is run, not assumed. |
| **A candidate is scored as the MEAN over the waypoints its span crosses.** | `offbook/world.py::audition`'s convention, verbatim — it is what makes a 1-segment tape and a 3-segment chain comparable, and it is the same reduction the piece metric uses. |
| **A-C is split.** A-C0 asserts that seam-time audition is *exact* here; A-C1 measures the library-construction audition. | The plant is deterministic and the rollout *is* the plant, so a seam-time audition predicts its own consumption to machine precision (smoke: max|Δ| = 1.5e-07). That is a structural fact, not a finding, and stating it prevents a vacuous "the audition is perfectly calibrated" claim. The non-trivial calibration is the construction audition, which scores on held-out hand-over states and consumes elsewhere — étude's winner's curse and seam-state shift, which do not go away. |
| **A-S measures both spreads under matched conditions** (between-start and within-start, both under practice motor noise), reports the noise-free performance spread separately, and marks seam 0 structurally undefined. | Performance here is deterministic, so a naive repeat-noise floor is exactly 0 and the ratio is meaningless (the first smoke printed 4.8e6). Seam 0's state *is* the start state, so a repeat of one start has zero spread there by construction. The **per-state oracle gain** (best single fixed slot ÷ per-state argmin over slots) is added as the instrument that actually measures the value of keying — `fingering/` G1's 4.0× and `offbook/` G-S's readout. |
| **A-R's `audit_k` picks the first k slots in library insertion order** (= construction-score order). | There is no π in Phase A. This is an **optimistic** top-k and every row says so; the honest routing number is Phase B's. |
| **The A-I halt does not abort the remaining gates.** | The SPEC's halt means "stop before treatments", and it is honoured: Phase B is not licensed by a halted run. The remaining Phase A gates are model-free instruments that do not presuppose the search, and are cheap, so a halt gets reported with its context instead of bare. |

## Gotchas (inherited and new)

- **Never import a sibling runner.** `mjc/shared.py` exposes one Modal `app` for the whole package,
  so importing `etude/etude.py` would register its `@app.function` and `@app.local_entrypoint` into
  this app. Gate A-F1 therefore reads the donor's constants with **`ast`**, never by import — which
  is also strictly stronger, because it fails if the donor's source ever moves.
- **A Modal *local* entrypoint runs on the client**, which in these sessions has neither numpy,
  torch nor mujoco. `piece.py` is pure python for this reason, `world.py` is imported only inside
  the Modal function body, and `analyze_gates.py` uses `json` + `math` only.
- **Launch runners only via `--spawn`.** `modal run --detach <file>::<entrypoint>` does not protect
  the run: a local entrypoint blocking in `.remote()`/`.map()` dies with the client and takes the
  outstanding work with it (`offbook/FILES.md` Gotcha; it cost two runs).
- **Entrypoint names must be unique across the package** — `acappella_gates` / `acappella_profile`,
  never `gates` / `profile`.

## Runs on disk

| tag | what |
|---|---|
| `asmoke` | `--quick` smokes of the full Phase A chain (every gate exercised end to end before launch). |
| `a0` | Phase A, seed 0, complete, 3383 s wall on 16 CPUs. **Gate A-I's pre-fixed halt FIRED.** |
| `bsmoke` | `--quick` smokes of the full B1 chain (every gate and both verdict components exercised, in both directions, before launch). |
| `b1` | Phase B1, the Δ ladder, seed 0. |

## Phase A record (`a0`, seed 0, 2026-08-27) — full report at `results/a0/gate_report.txt`

Numbers only; interpretation is not written here (it gets discussed with Jasper first).

**A-F passes exactly.** 13/13 donor constants match; the forked traversal reproduces the
transcribed donor loop at **max|Δ| = 0.000e+00** on states, commands, per-segment boundary error,
`n_fb` and priced time, both noise-free and under motor noise.

**The reflex law** (kp=10, kd=2.0, fit on the clean world then frozen): clean 0.0030 → **rotated
0.0045**, per segment [0.0027, **0.0091**, 0.0030, 0.0031] — the drilled segment costs 3×, the
clean ones are unaffected. 136 fb, 0 groundings, **16.9 s priced**.

**A-B, the budget ladder** (performance tempo, no library in existence):

| G | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| e_piece | 0.7100 | 0.5718 | 0.3307 | 0.2028 | 0.1333 | 0.1013 | 0.0810 | 0.0667 | 0.0612 |
| ground/traversal | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
| priced s | 62 | 121 | 238 | 473 | 942 | 1880 | 3756 | 7508 | 15011 |

**G\* = 4096 — and it is the last rung, not a saturation point.** The 2048 rung (0.0667) misses the
pre-fixed 5% band (≤0.0642) by 3.8%, so the rule selected the ladder's end. The incumbent had not
saturated at **16× étude's own 1024 imaginary rollouts per decision**. Any G-dependent statement
from this run is ladder-limited and must say so.

*Free instrument — étude finding 2 on a real rollout.* The elite MEAN vs the best ACTUALLY SAMPLED
sequence, realised: identical at G ≤ 32 (elite = 1 sample), then diverging to **2.3× at G = 4096**
(seam 0: 0.1154 vs 0.0510). Averaging the elite destroys the sequence on the plant too.

**A-B\*, the tempo axis at matched 16384 groundings/traversal** (instrument; the rule does not read
it): R=1 0.2154 · R=2 0.2070 · R=4 0.2049 · R=8 0.1755 · R=17 0.1079 · **R=34 0.0607**. Monotone:
one wide decision per segment beats many thin ones at matched spend. Reactive rung at absolute
budgets: R=1 G=8 → 0.8086 (1088 g); G=32 → 0.4583 (4352 g); G=128 → 0.2001 (17408 g).
**Caveat: no warm start** — every re-plan restarts CEM from μ=0, σ=0.8, so the small-R cells are
probably pessimistic.

**A-I — the pre-fixed halt FIRED.** Search at G\* reaches 0.0612, i.e. **inside étude's `never`
band (0.10–0.11) and better** — that half passes. But **0 of 15 cells** of the whole incumbent
family (9 budgets × performance tempo + 6 tempo cells + 3 reactive rungs) beat the reflex law on
error, and 0 dominate it: best search cell 0.0607 at 15011 s priced against the reflex's 0.0045 at
16.9 s — **13.5× worse error at 888× the priced cost.** Per the SPEC, the search is not an
incumbent worth beating, and **Phase B is not licensed by this run.** No forward model was involved
anywhere, and none could change this.

**The library** (nested, 8 slots/cell, 192-rendition pools, 3840 groundings to build):
selection scores 0.0506–0.0971 (segments) and 0.0750–0.1144 (chains); pool medians 0.0840–0.3336.
**The winner's curse reproduces on the plant**: `score_of_best_own_err` vs `chosen_score` is
0.0881/0.0576, 0.0882/0.0506, **0.6967/0.0682**, 0.2837/0.0971 — up to 10×.
Chain cells from segment-slot spellings vs the reflex-sourced control: 0.1144/0.0537 (4,0),
0.0750/0.0559 (3,1), 0.0773/0.0760 (2,2) — the control's *audition* score is better in all three.

**A-C.** A-C0 (structural): seam-time audition is exact, max|chosen − realised| = **1.6e-07**.
A-C1 (the real one), like-for-like (both sides means): segment gaps **1.07 / 1.23 / 1.16 / 1.12**,
chain gaps **0.92 / 1.12 / 1.09**, against étude E-3b's 2.7–3.4× and E-4's 2.82 control.

**A-R, the rent** (G\* = 4096; top-k is construction-score order, i.e. optimistic):

| arm | e_piece | ground/trav | fb | priced s | fb-only priced s | frac_prim |
|---|---|---|---|---|---|---|
| `never_reflex` | **0.0045** | 0 | 136 | 16.9 | 16.9 | — |
| `key` (frozen key, seg) | 0.0648 | 0 | 4 | **3.7** | 3.7 | 0.00 |
| `lib_only` seg | 0.0392 | 32 | 4 | 33.0 | 6.9 | 0.00 |
| `lib_only` seg+chain | 0.0352 | 49.2 | 3.4 | 85.4 | 8.5 | 0.00 |
| `audit_k` k=1 | 0.0440 | 16388 | 4 | 15015 | 1642 | 0.77 |
| `audit_k` k=8 | 0.0328 | 16416 | 4 | 15041 | 1645 | 0.55 |
| `audit_all` (=k=16) | 0.0319 | 15926 | 3.9 | 14631 | 1596 | 0.58 |
| `never_search` | 0.0612 | 16384 | 4 | 15011 | 1642 | 1.00 |

`audit_all` ≡ `audit_k`(k=16) at exactly **0.0** (free identity check). The O(K)→O(k) audition cut
is 13 groundings out of ~16400 (0.08%): **the rent in this table is the primitive's search budget,
not the audition.** The library alone reaches 0.0352 at 49 groundings against the search's 0.0612 at
16384 — better error at **334× fewer groundings**; a frozen keyed tape (0 groundings, 3.7 s) matches
the 4096-rollout search (0.0648 vs 0.0612).

**A-S, seam information.** Matched spreads (both under practice noise): seam 1 **1.13**, seam 2
**1.07**, seam 3 **1.11** (positional 0.99/1.02/0.99); seam 0 structurally undefined. But the
**noise-free performance spread under the reflex collapses to nothing**: 0.1450 (= the start
jitter) → 0.0038 → **2.1e-05** → **0.0**. Per-state oracle gain over the 8-slot segment library:
**1.65× / 2.35× / 1.22× / 1.27×**, with 8/6/4/3 distinct argmins of 8.
**Anomaly, not smoothed**: the spreads are measured under the *reflex* and the oracle gain under the
*library-playing* configuration, so the two halves of A-S are not the same configuration. The
oracle gain is the one measured where a key would actually be used.

**A-D, wall-clock** (16 CPUs): A-F 1.1 s · reflex cal 15.4 s · A-B 568 s · library 88 s · A-C 13 s ·
A-R 1631 s · A-S 5.2 s. One search traversal at n_eval=32: 3.0 s at G=16 → 272.9 s at G=4096.

### Open flags carried out of Phase A

1. **G\* is ladder-limited** (last rung; 2048 misses the band by 3.8%).
2. **The reflex gains sit on the kd grid edge** (kd = 2.0, grid max 2.0; kp = 10 is interior). The
   reflex is therefore, if anything, *understated* — which only deepens the halt.
3. **No warm start in the R < H cells**, so the tempo instrument is pessimistic about fast tempi.
4. **A-S mixes two configurations** (see above).
5. The **chain audition scores** favour the reflex-sourced control over segment-slot spellings in
   all three cells — the opposite direction to `legato/` F5, measured at audition time only.


## B1 — the re-scope, and everything pre-fixed before launch (2026-08-27)

Re-site decision: **Jasper's, on reading Phase A**. Written up as a dated amendment in
[`SPEC.md`](SPEC.md). **A-I's halt stands and is not retracted** — it is the finding the re-site
responds to. The grounding economy is retired as the headline; B1 is sited on the feedback/delay
axis, which is what A-R's own rent table separates the arms by (136 / 4 / 1 fb, against a routing
cut of 13 groundings in ~16400).

### Pre-fixed before launch

| item | value, and why |
|---|---|
| **the gate** | Δ\*<sub>niche</sub> = smallest Δ at which the **best committed-content arm** (min over `key_seg`, `lib_seg`, `lib_all`, `key_chain`, `aud_chain`) is **strictly** better than the reflex law on piece error, subject to being ≤ `ref_play`. Binding in both directions. |
| **the guard** | `ref_play` = **0.1066** — étude's `never` at performance tempo, 3-seed mean. `offbook/` d0's `ref_stale` construction on this very piece; arm-neutral because no arm here produces it, and published before this node existed. |
| **guard, reported but NOT used** | presto's `½ × leg` = 0.4. On this piece (leg 0.8) it would make the guard vacuous — d0's own guard-revision failure in the opposite direction. Also reported: the within-run do-nothing floor and half of it. |
| **the depth test** | d0's 3-term ordering `e_chain ≤ e_seg ≤ e_reflex` (`aud_chain` / `lib_seg` / `reflex`), per Δ, with both ordering margins, the guard margin in m and as a fraction, and the chain's distance to the guard (presto decision 11). Separates a **segment-span** niche (offbook 7d) from a **chain-span** one. |
| **the converse halt** | reflex best at every Δ ⇒ no niche, B2 not licensed, regardless of the priced ledger. |
| **the range** | Δ ∈ {0,1,2,3,4,6,8,12,16} steps = 0–384 ms at `dt_ctrl` 0.024. **Not extended or re-tuned on failure** (legato F2); the pre-declared escalation is presto's fast piece as a separate round. |
| **seeds** | single seed 0, as everywhere in this arc. |

### Decisions taken for B1, with reasons

| decision | why |
|---|---|
| **The delay is naive — nothing bridges it.** | `accompanist/` d3b: the naive operator is a strawman *exactly when a predictor exists* (efference copy recovered 6–11× of the penalty). With no FM anywhere it is the honest operator. **Recorded, deliberately not taken**: an arm that already pays for plant access could bridge Δ by re-executing its issued commands on a resettable copy — efference copy without a forward model. Not implemented (nothing may be added). The asymmetry it implies is on the record: for the reflex law and the frozen key the naive operator is a *necessity* (no plant access at all); for the auditioning arms it is a *choice*. |
| **The reflex gets every advantage: gains re-fit at every Δ** on the clean world, same pre-fixed criterion as Phase A, on a grid **widened past a0's kd edge** (a0 chose kd = 2.0, its grid maximum — Phase A flag 2). | It is the incumbent now; a handicapped incumbent would make the niche meaningless. |
| **The library is built ONCE, undelayed, with the harvest reflex FROZEN at a0's gains** (kp=10, kd=2.0). | d0's construction: the delay is a decision constraint, not a learning-data treatment. Freezing the harvest gains is also what makes gate B-F1's cross-tag identity with `a0` possible — if the incumbent's re-fit changed the harvest, the library would be a different object and Δ would no longer be the only variable. Deliberately generous to the incumbent (it adapts to Δ; the library does not). |
| **B-F1 is a cross-tag exact control against `a0`**, asserting the library build (7 cells) and the Δ=0 rows for `key_seg` / `lib_seg` / `lib_all` / reflex-at-a0-gains reproduce `a0` at max\|Δ\| = 0. | `offbook/`'s bit-identity idiom: it gates the whole delay edit at once. It is **declared inapplicable** (never silently "passed") under any config that is not a0's — a smoke has a different library, so the control is not defined there. |
| **The chain arms are exactly delay-invariant, and that is a property of the piece.** | `aud_chain`/`key_chain` decide once, at seam 0, where the piece starts from rest and there is no stale history to read, so `obs(Δ)` is the true state for every Δ. étude's loop starts at the first waypoint; `offbook/`'s piece had an approach leg before its first seam. Flagged in the SPEC amendment, in the runner's criterion block, and printed under every reduction, so no chain-flatness number can be read as evidence about depth. |
| **The record rung (priced search) runs at Δ = 0 only.** | A-I retired it. It is kept so the two economies stay on one table, not as a comparison arm. |

### Smoke record (`bsmoke`, what it caught before launch)

- **B-F1 correctly refused a smoke config** (max\|Δ\| = 6.99e-02 against a0). The control is
  defined against a0's exact configuration; under `--quick` the library is a different object. Fixed
  by gating applicability on the config rather than by loosening the tolerance.
- **The audition-optimism instrument crashed on ragged `picks`** — a chain arm decides for everyone
  at seam 0 and for nobody after, so the per-seam score lists have different lengths. Fixed by
  recording the deciding performers and their spans in each pick, which is also what makes the
  realised side of the comparison span-correct.
- **The verdict was restructured** after the smoke showed the 3-term ordering alone is the wrong
  primary gate here: with the priced search retired, the segment arm is a *stored tape* (4 reads),
  not d0's live plan, so it is nearly delay-invariant too, and `e_chain ≤ e_seg` becomes a content
  comparison rather than a delay one. The niche criterion above is now the gate and the ordering is
  the depth test, reported alongside. Done before launch and before any real number existed.
- Shape confirmed in smoke, degradation ordered exactly by feedback consumption: reflex (136 fb)
  **245.8×** · `key_seg` (4) 1.91× · `lib_seg` (4) 0.98× · `lib_all` (3) 3.11× · `key_chain` (1)
  1.00× · `aud_chain` (1) 1.00×.

### B1 result (`b1`, seed 0, complete, 694 s wall) — full report at `results/b1/delay_report.txt`

Numbers only; interpretation gets discussed with Jasper first.

**Gates.** B-F0 13/13 donor constants. **B-F1 cross-tag exact control vs `a0`: max|Δ| = 0.000e+00
on all six checks** (library build seg + chain, and the Δ=0 rows for `key_seg`, `lib_seg`,
`lib_all`, reflex-at-a0-gains), `applicable=True`. The delayed code path *is* a0's at Δ=0.

**Anchors** (blind): `ref_play` = 0.1066; do-nothing floor 0.7056 (half 0.3528); `half_leg` 0.4
(reported, unused).

**The sweep** — piece error [fb/traversal]; groundings/traversal: reflex 0, key_seg 0, lib_seg 32,
lib_all 49.2, key_chain 0, aud_chain 8.

| Δ (ms) | reflex [136] | key_seg [4] | lib_seg [4] | lib_all [~3.4] | key_chain [1] | aud_chain [1] |
|---|---|---|---|---|---|---|
| 0 | **0.0045** | 0.0648 | 0.0392 | 0.0352 | 0.1184 | 0.0852 |
| 2 (48) | 0.0330 | 0.0592 | 0.0542 | 0.0437 | 0.1184 | 0.0852 |
| 4 (96) | 0.0675 | **0.0621** | 0.0644 | 0.0645 | 0.1184 | 0.0852 |
| 6 (144) | **0.0651** | 0.0832 | 0.0736 | 0.0807 | 0.1184 | 0.0852 |
| 8 (192) | 0.1697 | 0.1034 | **0.0838** | 0.0922 | 0.1184 | 0.0852 |
| 12 (288) | 0.4622 | 0.0910 | 0.1112 | 0.0922 | 0.1184 | **0.0852** |
| 16 (384) | 0.5997 | 0.1071 | 0.1139 | 0.1090 | 0.1184 | **0.0852** |

Record rung (Δ=0): search G=1024 → 0.0810 at 4096 groundings / 3756 s; G=4096 → 0.0612 at 16384 /
15011 s.

**The gate.** **Δ\*<sub>niche</sub> = 4 (96 ms)** by the pre-fixed rule — but see flag 1: that cell's
margin is **+0.0054** and the niche **closes again at Δ = 6** (−0.0085) before reopening at Δ = 8
with **+0.0859**. Niche true at Δ ∈ {4, 8, 12, 16}, false at {0,1,2,3,6}. Converse halt did not fire.

**The 3-term depth ordering** passes first at **Δ = 12**, and at Δ = 12/16 only — driven entirely by
`e_seg` rising above the *delay-invariant* `e_chain` (see the structural caveat), with guard margin
20.1%. At Δ = 8, where the niche is robust, the best committed arm is `lib_seg` — a **segment-span**
niche (`offbook/` finding 7d), not a chain-span one.

**Degradation e(16)/e(0), by feedback consumption**: reflex (136 fb) **134.4×** · key_seg (4)
**1.65×** · lib_seg (4) **2.90×** · lib_all (3.4) **3.10×** · key_chain (1) **1.00×** · aud_chain
(1) **1.00×**. Ordered across the fb decades, but **violated within the 4-fb tier** — see flag 6.

**What Δ costs the audition** (A-C0 said it is exact at Δ=0): realised ÷ chosen for `lib_seg`
1.00 → 1.27 → 1.56 → 1.54 → 1.74 → **1.77** (Δ=6) → 1.10 → 0.74 → **0.47** (Δ=16); max|Δ| grows
monotonically 0.0000 → 0.4743.

### Flags carried out of B1 (not smoothed)

1. **Δ\* = 4 is a fragile, non-monotone single point.** Margin +0.0054, and the niche closes at
   Δ = 6. The first *sustained* niche is Δ = 8 (192 ms). The pre-fixed rule's answer is reported as
   committed; the fragility is reported with it.
2. **The reflex's per-Δ gain re-fit transfers non-monotonically from clean to rotated**, which is
   what produces flag 1: clean 0.0026 → rotated 0.0209 at Δ=3; clean 0.0227 → rotated 0.0675 at
   Δ=4; clean 0.0877 → rotated **0.0651** at Δ=6. The criterion (fit on clean) is the pre-fixed one
   and was not changed.
3. **The gain grid hits an edge at Δ ∈ {3,4,6} (kp at its minimum) and at Δ ∈ {12,16} (kp min AND
   kd max)**, so the incumbent may be understated exactly where the niche is widest. At Δ=16 the gap
   is 0.5997 vs 0.0852 (7×), which a gain tweak is unlikely to close; at Δ=4 it is 0.0054, which one
   easily could.
4. **Δ=3's clean error (0.0026) is *better* than Δ=0's (0.0030).** A delay improving a tracking
   controller is not a plausible real effect; most likely the coarse grid found a cell where the lag
   damps overshoot on the clean world. Instrument oddity, recorded.
5. **The chain-span ordering pass at Δ=12/16 is the pre-committed artifact, not evidence about
   depth.** `aud_chain`/`key_chain` are *exactly* delay-invariant (0.0852 / 0.1184 at every Δ, to
   four decimals) because they decide once at seam 0, where the piece starts from rest and there is
   no stale history. `lib_seg`'s segment-0 error is likewise identical at every Δ (0.0296) for the
   same reason. Property of this piece, not of depth.
6. **The fb-consumption ordering is violated inside the 4-fb tier**: `key_seg` degrades 1.65× while
   `lib_seg` degrades 2.90× at the same 4 fb. Mechanism, measured: the frozen key makes a coarse
   nearest-neighbour decision, the audition makes a fine argmin decision from a stale rollout start —
   **the audition is the more delay-fragile selector.**
7. **`lib_all` selects the fewest chains exactly where chains pay most**: `frac_chain` 0.094 → 0.394
   (Δ=8) → 0.228 (Δ=12) → **0.040** (Δ=16), while `aud_chain` is flat-best at Δ≥12. The arm that
   could exploit depth fails to select it because its selector is itself delayed (flag 6's mechanism
   at the level of the level choice).
