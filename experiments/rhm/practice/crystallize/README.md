# crystallize — certificate-gated compilation on the sculpting substrate

**Up**: [../README.md](../README.md) (rhm/practice) · [../../README.md](../../README.md) (rhm)
**Idea doc**: [practice_manufactures_its_own_credit](../../../../ideas/practice_manufactures_its_own_credit.md)
(§1 re-chunking, §3 the compile op, §12 → the étude; §13 points here)
**Parents**: [`mjc/practice/etude/`](../../../mjc/practice/etude/README.md) (the compile op, its two
corrections, and finding 7 — *hierarchy is meaningful only over boundaries that carry information* —
which is the question this node was built to answer) ·
[`RHM_SCULPTING_README`](../../RHM_SCULPTING_README.md) (Stage 3b: the re-grounded latent/token beam
this forks) · [`directed_sculpting/full_loop/level_moves/`](../../directed_sculpting/full_loop/level_moves/README.md)
(the level-indexed action space and hierarchical damage, ported here off the channel layout)
**Status**: written up 2026-08-14 (interpretation discussed with Jasper 2026-08-14). Seven runs:
a self-check, two headroom calibrations, two descent calibrations, the main 5-arm run, and a
plant precheck. File index: [FILES.md](FILES.md).

## One-liner

Porting the étude's compile op onto sculpting answers its open question — **state-conditioned
commitment beats state-independent commitment by 1.8–3.0× in 6/6 commit states** — and then removes
the certificate's job: with the plant frozen and only the selector learning, **committing at cycle 1
is optimal**, matching every gated arm's accuracy at **26× less priced time** than never-compiling.
The δ-silence certificate is not wrong here; it has nothing to certify, because *practice on a frozen
plant makes you better at improvising and improvisation is exactly what does not compile*. A precheck
confirms the diagnosis: let the plant learn and the committed-unit ceiling moves at **31–66× the
metering noise floor**.

## Findings

Margins are stated against the measured noise floors, and the two places where a claim
does not survive its own noise are marked.

1. **State-conditioned commitment dominates state-independent commitment, 6/6 commit states.** A
   library keyed by the observed target root scores 0.174–0.240 where one global program scores
   0.381–0.520 — **1.8–3.0×** — from the *same* pool, at the *same* commit cycle, at the same 2
   groundings per solve. End to end, `gate_single` finishes at e = 0.483 against `sched_late`'s
   0.241. This is the étude's finding 7 answered on a substrate where the boundary carries
   information: the generator reads the configuration, so a committed move program is a genuine
   generalized motor program rather than a replayed constant. (§Discriminator, §Priced grade)
2. **On a frozen plant, commit immediately.** `sched_early` (c1) reaches `never`'s *best-ever*
   accuracy (0.2057 vs 0.2070) at **26× less priced time**, and beats `sched_late` (c30) at **13.4×
   less**. No timing effect is resolvable: the compile op's own candidate-draw noise is
   sd ≈ **0.07** on library audition at `n_cand = 32`, larger than every arm-to-arm accuracy
   difference. "No detectable timing effect", not "timing effect = 0". (§Priced grade)
3. **Averaging destroys realisations, 3.6–5.0×.** The position-wise modal move sequence over the
   pool's *valid* realisations scores 0.83–0.89 against the selected library's 0.17–0.24. The étude
   measured 2.1× for command-sequence averaging; RHM gives the mechanism exactly: at `sched_early`'s
   deep commit the modal **token** span was **on-grammar in 0.500 of its blocks while every
   contributing realisation was 1.000 on-grammar**, and its error was exactly **1.000**. The mean of
   two synonym expansions of the same latent is off-grammar. (§Discriminator)
4. **The winner's curse reproduces at 3.3×, and it is worse per-key.** Ranking by "it worked on its
   own instance" scores 0.621–0.735 where expected-performance selection scores 0.164–0.270 (mean
   ratio 3.3×, 6/6). Selecting *per library key* is markedly more curse-prone than selecting one
   global unit — in-sample→held-out optimism 1.56× vs 1.10× at 32 instances/key — which is why the
   run uses 512 scoring instances (64/key). (§Discriminator, §Calibration)
5. **The optimism gap is ≈ 1.0 — confirmation of the seam account by its absence.** Realised
   (anchored) level ÷ held-out audition is **0.90–1.10, mean 0.99, 8/8 commits**, against the
   étude's 2.7–3.4× before its seam fix. Two constructions remove it: units launch at the instance
   start, so audition and consumption distributions coincide (no seam-state shift is *available* in
   this scope), and audition and metering use the identical statistic. In-sample gaps run 0.94–1.35,
   so the half/half held-out audition is doing real work. (§Commit table)
6. **The committable content of practice traces is flat while the closed-loop policy improves by
   0.12–0.14.** A shadow compile — the compile op run every cycle against held-out instances and
   never acted on — is flat over 60 cycles (library audition slope t = −1.27 / −1.51; in-sample
   t = −1.41 / −1.35), and the cycle-1 pool already reaches the exhaustively-enumerated ceiling
   (in-sample 0.173 vs 0.143). **Retraction**: an earlier reading of this node's own commit events
   claimed practice *degrades* the transferable content (`best1` 0.266 → 0.469 → 0.557 across three
   commits). The 60-point shadow says the single-unit audition is flat (t = −0.16 / −0.53); three
   draws at sd 0.07 cannot support that claim, and it is withdrawn. What does replicate (here and in
   `cald_s1`) is `own_ok` degrading — on the shallow context only (t = +4.28 / +3.10; deep flat in
   both runs). (§Shadow)
7. **The plant is the missing lever, measured.** Twenty cycles of generator fine-tuning on the
   agent's own successful repairs moves the committed-unit **ceiling** by 31–66× the 0.003 metering
   noise floor (deep library 0.301 → 0.152 at one move; shallow 0.309 → 0.109), where the
   state-*independent* ceiling does not benefit reliably (deep −0.062, shallow +0.074). A one-move
   committed library after plant training reaches e = 0.109–0.152 **at 2 groundings**, below the
   frozen-plant wide beam's 0.201/0.258 at 248× the feedback. (§Plant precheck)
8. **Compilation's Pareto position, stated plainly.** A committed library matches or beats the beam
   it replaces at the declared feedback budget (deep 0.174 vs 0.180; shallow 0.209 vs 0.361) at 23×
   less feedback, and never reaches the wide beam (0.043–0.289 at 248× the cost). Where compilation
   pays is a property of the declared budget, and the whole ladder is logged every probe cycle so
   that dependence is visible rather than assumed. (§Discriminator)

## Scope

One 5-arm run (`cg_s0`, seed 0, 60 cycles) plus four calibrations and a precheck that configured and
diagnosed it. δ is consumed **only as a detector**; the value's online updates are plain uniform-lr
AdamW with no per-sample gain anywhere. Nothing outside this folder was modified.

## Substrate

Fork of Stage 3b (`../../rhm_sculpt_latent.py`): v=8, s=2, L=4, m=2 → 16 tokens, 8 blocks. Controller
(per-block belief + pooled state), generator (block infiller), MC value V(z̄, r\*) trained once at
setup and **shared by every arm**; only the value adapts online.

- **Action space** — level-indexed, 15 moves (8 L1 / 4 L2 / 2 L3 / 1 L4), ported from
  `level_moves.regenerate_node` onto the single-channel grammar. A move commits to one level-ℓ
  feature via a max-sum DP over the generator's evidence and renders the legal subtree beneath it.
  Gate **C1**: the level-1 move is bit-identical to the published `_regenerate`, so level-1 readouts
  stay comparable to Stage 3a/3b. Gate **C2**: every deeper move renders on-grammar.
- **Damage** — hierarchical (`corrupt_hier`): a level-k subtree replaced by a legal derivation of a
  feature the observed subtree provably cannot produce. Gate **G-D**: on-grammar rate **1.000** at
  levels 1/2/3, against **0.808** for the published random-symbol damage — so no error is
  block-locally visible and abstraction has to matter. *Not in the parent node's gate*: at one
  damaged node, ambiguity leaves `d* == 0` on **19.9%** of instances (a feature a node cannot derive
  can still leave r\* in the **root's** possible-set), so context instances are rejection-sampled on
  `d* > 0`.
- **Contexts (the piece)** — two recurring damage cells, `deep` = level-2 node 1 (blocks 2–3) and
  `shallow` = level-1 nodes 6,7 (blocks 6–7). Same instance family, different depth of error;
  `d0` = 2.11 / 2.34, DP-oracle floor 0.082 / 0.092. The cell recurs; the clean derivation, r\*, the
  wrong feature and the synonyms all vary, so a committed unit cannot be a replayed constant.
- **Practice regime** — closed-loop re-grounded token beam over the level moves at width 16, budget
  3, full materialise-and-re-encode. Every surviving tip's whole trajectory is labelled by that tip's
  terminal possible-set success and fed to the value (value-iteration).
- **Performance regime + pricing** — a declared per-solve **grounding budget**. A grounding is one
  materialise + re-encode + value-score. An uncommitted context runs the widest beam that fits
  (width 1 → **46 groundings/solve**); a committed context pays **2** — one observation at launch to
  key the library, one final verification — and executes its move program open-loop. Priced time
  `t = n_ground · d_fb + n_mat · c_mat` at `d_fb = 1.0`, `c_mat = 0.05`; all counts are logged
  separately so the price vector can be changed post hoc.
- **Staleness** — the setup value is trained on the **generic** random-symbol distribution, so it
  arrives stale on the hierarchical contexts. This is the `pretrain_mode=exclude` analog and it is
  what gives the detector a descent to look at.
- **Metering + certificate** — `e_k` = 1 − success on a fixed held-out 384-instance set per context
  (agent-observable; residual `d*` is logged alongside as an oracle readout). `b_k` = EWMA(α=0.2),
  `δ_k = b_k − e_k`, δ-silence = window-mean |δ| < c·scale **and** sd(e) < c_v·scale held `sil_hold`
  cycles, `scale = max(ref_stale_k − min_so_far_k, b_k)` with `ref_stale` from setup. Run at
  c = 0.06, c_v = 0.10, W = 5, hold = 2.
- **Compile op** — candidates drawn **uniformly** from the trace pool, executed open-loop on
  held-out fresh instances of the context, argmin of the mean committed. `key = "root"` commits a
  library (one program per target root); `key = "global"` commits one program. Two audition numbers
  are reported and their difference *is* the winner's curse: in-sample, and select-on-half /
  score-on-other-half. Scoring rollouts are priced into `t_cum`.

## Arms (the compile trigger and the unit's key are the only differences)

| arm | trigger | unit key | notes |
|---|---|---|---|
| `never` | — | — | pays 46 groundings/solve throughout |
| `sched_early` | cycle 1 | library | commit before any adaptation |
| `sched_late` | cycle 30 | library | commit at the asymptote |
| `delta_gate` | δ-silence | library | the certificate |
| `gate_single` | cycle 30 | **global** | trigger-matched to `sched_late`, so single-vs-library is isolated from timing |

## Results — `cg_s0` (seed 0, 5 arms, 60 cycles, complete)

### Priced grade

| arm | commits | `t_cum` | g/solve | e_mean | e_deep | e_shallow | `never` @ matched t | margin |
|---|---|---|---|---|---|---|---|---|
| **`sched_early`** | c1, c1 | **238,720** | 2 | 0.2057 | 0.1719 | 0.2396 | 0.3509 | **+0.1452** |
| `delta_gate` | c6, c24 | 1,666,227 | 2 | **0.1992** | 0.1562 | 0.2422 | 0.2413 | +0.0421 |
| `sched_late` | c30 | 3,195,699 | 2 | 0.2409 | 0.2474 | 0.2344 | 0.2133 | −0.0275 |
| `gate_single` | c30 | 3,195,699 | 2 | 0.4831 | 0.5391 | 0.4271 | 0.2133 | −0.2697 |
| `never` | — | 6,216,960 | 46 | 0.2266 | 0.1927 | 0.2604 | — | best-ever 0.2070 |

`never`'s own trajectory is interpolated to each arm's final priced time (the étude's matched-time
idiom). `sched_early` and `delta_gate` both beat `never`'s best-ever error anywhere in its run.

### Commit / anchor table

`own_ok` is the mean audition score of drawn candidates whose own rollout succeeded — the winner's
curse in its binary form on this substrate.

| arm | ctx | c | key | in-sample | **held-out audition** | pool med | `own_ok` | anchored | **gap** |
|---|---|---|---|---|---|---|---|---|---|
| `sched_early` | deep | 1 | root | 0.127 | 0.168 | 0.789 | 0.718 | 0.1719 | 1.02 |
| `sched_early` | shallow | 1 | root | 0.230 | 0.219 | 0.671 | 0.674 | 0.2396 | 1.10 |
| `delta_gate` | shallow | 6 | root | 0.246 | 0.270 | 0.654 | 0.621 | 0.2422 | 0.90 |
| `delta_gate` | deep | 24 | root | 0.137 | 0.164 | 0.739 | 0.733 | 0.1562 | 0.95 |
| `sched_late` | deep | 30 | root | 0.217 | 0.238 | 0.783 | 0.735 | 0.2474 | 1.04 |
| `sched_late` | shallow | 30 | root | 0.250 | 0.262 | 0.724 | 0.709 | 0.2344 | 0.90 |
| `gate_single` | deep | 30 | global | 0.557 | 0.578 | 0.783 | 0.735 | 0.5391 | 0.93 |
| `gate_single` | shallow | 30 | global | 0.434 | 0.398 | 0.724 | 0.709 | 0.4271 | 1.07 |

### The compile-hit discriminator

Every unit family, evaluated on the same held-out performance instances at each library commit. Deep
at c24 shown; the ordering is identical in all six commit states.

| unit | e | residual d\* | g/solve |
|---|---|---|---|
| DP-greedy oracle (privileged) | 0.066 | 0.072 | — |
| wide beam, width 16 | 0.065 | 0.164 | 496 |
| beam, width 4 | 0.102 | 0.264 | 139 |
| **selected library** | **0.174** | 0.451 | **2** |
| narrow beam, width 1 (what it replaces) | 0.180 | 0.488 | 46 |
| selected single (global) | 0.520 | 2.779 | 2 |
| verbatim token span (state-independent) | 0.695 | 0.844 | 2 |
| modal token span (averaging, token space) | 0.695 | 1.932 | 2 |
| modal move sequence (averaging, move space) | 0.881 | 2.203 | 2 |
| random move sequence | 0.992 | 3.916 | 2 |

Worth recording: at the deep c1 commit a **random** move program (0.486) beat the **verbatim token
span** (0.973) — a state-independent commitment can be worse than no commitment. And the adapted
wide beam beats the exact-DP greedy oracle on the deep context (0.043 vs 0.066 at c30) but not on the
shallow one (0.119 vs 0.086); value-iteration on a practised context surpasses greedy-on-`d*`.

### The shadow compile

Run every cycle on every uncommitted context against a held-out score set, and never acted on — the
quantity the certificate is supposed to gate. `never` arm, 60 cycles:

| readout | deep c1–15 → c46–60 | t | shallow c1–15 → c46–60 | t |
|---|---|---|---|---|
| library audition (held-out) | 0.220 → 0.185 | −1.27 | 0.260 → 0.221 | −1.51 |
| library audition (in-sample) | 0.197 → 0.158 | −1.41 | 0.255 → 0.224 | −1.35 |
| single-unit audition | 0.432 → 0.413 | −0.16 | 0.449 → 0.433 | −0.53 |
| pool median | 0.785 → 0.765 | −1.68 | 0.744 → 0.788 | **+2.46** |
| `own_ok` | 0.742 → 0.743 | −0.13 | 0.700 → 0.735 | **+4.28** |

Against a closed-loop descent of 0.12–0.14 over the same window, the committable content moves by at
most 0.035–0.039 (same sign in both contexts, |t| ≤ 1.5). The pool does change — distinct/pool falls
0.57 → 0.22 — just not in a direction the compile op can use.

### `delta_gate`'s firing, and a detector-scale bug

Replayed offline on `cg_s0`'s own `never` series at (0.06, 0.10, 5, 2), the detector predicts
**deep c29, shallow c6**; the run fired **deep c24, shallow c6**. Shallow is exact; deep is five
cycles early because the arm's trajectory diverges from `never` once shallow commits.

The c6 shallow firing exposes a real bug. `refs["stale"]` for shallow is **0.320** while the arm's
own cycle-1 metering is **0.398**, so `ref_stale − min_so_far` is negative and the scale falls back
to `b` ≈ 0.391 — a much looser scale that reads a slow smooth descent as silence. Cause: the stale
reference is measured on the 512-instance reference draw, the metering on a different 384-instance
draw. **Fix: measure the setup stale reference on the metering set itself.** It cost nothing here —
`delta_gate`'s shallow unit anchored at 0.2422 against `sched_late`'s c30 commit at 0.2344 — which is
itself finding 2, illustrated.

## The plant precheck (`calp_s0`)

Twenty cycles of generator fine-tuning on the agent's own **successful** repairs (a solved config is
a valid r\* derivation, so the generator's masked-infilling target is well defined on it; grounded by
terminal task success, replay 0.5 against the clean setup pool), with the exhaustive 1- and 2-move
enumeration run before and after on *identical* fixed reference sets. Noise floor 0.003.

| | deep before → after | | shallow before → after | |
|---|---|---|---|---|
| 1-move **library** ceiling | 0.301 → **0.152** | −0.148 (49×) | 0.309 → **0.109** | −0.199 (66×) |
| 2-move library ceiling | 0.211 → **0.094** | −0.117 (39×) | 0.223 → **0.129** | −0.094 (31×) |
| 2-move **global** ceiling | 0.320 → 0.258 | −0.062 (21×) | 0.281 → 0.355 | +0.074 (25×) |
| wide beam (width 16) | 0.201 → 0.309 | +0.107 | 0.258 → 0.244 | −0.014 |

**Confound to carry forward**: the wide beam *degraded* on deep because the value was trained against
the old generator and was not co-adapted. The ceiling numbers are unaffected — they are a pure
generator property with no value in the loop — but any follow-up must co-adapt or refresh the value,
or its closed-loop reference will look artificially bad.

## Interpretation (discussed with Jasper 2026-08-14 — argued, not measured)

Three readings we agreed on, marked as interpretation because none is a measurement:

- **A scope condition for §1's certificate.** δ-silence gates compilation only where practice moves
  the **executor**. The étude's practice trained the forward model that its ballistic units bet on,
  so mastery of the FM *was* mastery of the unit; here practice trains the **judge** while the
  generator that executes a committed program is frozen, so the space of committable units and its
  best element are both fixed at setup and there is nothing for a certificate to certify. The port
  did not fail; it isolated a precondition the idea doc left implicit.
- **Expert vs learner.** Read as a regime statement: an agent whose production apparatus is already
  competent and who is only refining its selection should compile early and cheaply — practice on a
  frozen plant makes you better at improvising, and improvisation is precisely what does not compile.
  Certificates earn their keep in the learner regime, where the primitive itself is still moving.
- **The constructive form (LP framing — Jasper's).** The certificate should be conditioned on the
  **learning progress of the committable content** — the shadow-compile audition trajectory,
  positive and then silent — rather than on task-performance *level* or task-LP, both of which
  conflate selector improvement with plant improvement. This node's shadow instrument is exactly that
  signal, already built and already logged; here it reads flat from cycle 1, which is why "commit
  immediately" is the right call and why the correct certificate would have said so.

## Calibration record

Every knob below was set by a measurement, in the order the measurements forced.

| run | measured | change it forced |
|---|---|---|
| `selfcheck` | C1/C2 pass; hierarchical damage 1.000 on-grammar vs the published damage's 0.808; `d*==0` on 19.9% of singly-damaged instances | rejection-sample context instances on `d* > 0` |
| `calu_s0` / `calu_s1` | beams monotone to width 16 then *worse* at 64 (deep 0.344 / 0.236 / 0.201 / 0.229); exhaustive best fixed program 0.320 / 0.281 held-out; per-r\* library 0.223 / 0.223; best single move is the level-2 commitment covering the damaged node, against a median over all 15 moves of 0.98–1.00; per-key optimism 1.56× vs global 1.10× | `pr_width` 64 → **16**; unit type single → **library** with `gate_single` as the control; `n_score` → **512** (64/key); `calu_s0`'s shallow numbers superseded by `calu_s1` when the shallow context moved to blocks 6–7 |
| `cald_s0` | at the setup learning rate the whole stale→asymptote descent completes **before the first metering** (deep 0.297 at c1 vs a 0.344 stale reference); residual trajectory is noise (sd 0.022–0.05) and the detector fires anywhere from c6 to c30 | a separate **`value_lr_online`** knob (the étude's `n_grad` 40→10→5 lesson, moved onto the learning rate because per-cycle data volume makes `n_grad` blunt here); `n_rt` 192 → **384** |
| `cald_s1` | `value_lr_online` = **3e-5** gives a monotone ~25-cycle descent (deep 0.310 → 0.170, shallow 0.398 → 0.251) at noise sd **0.003**; 3e-4 non-monotone, 1e-5 still descending at c40. Detector at (0.06, 0.10, 5, 2) lands within ±0.005 of the asymptote and is not knife-edge. **And the shadow-compile audition is flat at every learning rate** | the run's plasticity and detector settings; and the decision to run the round without a certificate contrast as its headline, with `sched_early`/`sched_late` added so the vacuity is demonstrated rather than assumed |
| `calp_s0` | the committed-unit ceiling moves 31–66× the noise floor once the plant learns | scopes the next round (below) |

## Runs on disk

| tag | what it is |
|---|---|
| `calu_s0`, `calu_s1` | compilation-headroom enumeration (all 1- and 2-move programs, plus 3-move over the best pairs) vs beams at widths 1/4/16/64 and the DP oracle; `s1` is on the final contexts |
| `cald_s0` | descent calibration, `n_grad` ∈ {2, 8, 32} sharing one setup, 30 cycles |
| `cald_s1` | plasticity calibration, `value_lr_online` ∈ {3e-4, 3e-5, 1e-5}, 40 cycles, shadow compile on |
| `cg_s0` | the main run: 5 arms, seed 0, 60 cycles, complete |
| `calp_s0` | the plant precheck: exhaustive ceiling before/after 20 cycles of generator fine-tuning |

## Reproduce

```bash
cd experiments/            # MODAL_PROFILE=chromatic

# gates C1/C2/G-D (CPU)
modal run rhm/practice/crystallize/crystallize.py::selfcheck_remote
# smoke
modal run rhm/practice/crystallize/crystallize.py::crystallize --quick --tag smoke0

# calibrations
python3 rhm/practice/crystallize/launch_detached.py --fn cal_unit --tag calu_s1 \
    --n-ref 512 --widths "1,4,16,64"
python3 rhm/practice/crystallize/launch_detached.py --fn crystallize --tag cald_s0 \
    --arms "never:n_grad=2,never:n_grad=8,never:n_grad=32" --n-cycles 30
python3 rhm/practice/crystallize/launch_detached.py --fn crystallize --tag cald_s1 \
    --arms "never:value_lr_online=3e-4,never:value_lr_online=3e-5,never:value_lr_online=1e-5" \
    --n-cycles 40 --n-grad 4 --shadow-compile --probe-every 4

# the main run
python3 rhm/practice/crystallize/launch_detached.py --fn crystallize --tag cg_s0 \
    --arms "never,sched_early,sched_late,delta_gate,gate_single" \
    --n-cycles 60 --n-grad 4 --value-lr-online 3e-5 --n-rt 384 --n-score 512 --n-cand 32 \
    --sil-c 0.06 --sil-cv 0.10 --sil-win 5 --sil-hold 2 --sched-early 1 --sched-late 30 \
    --probe-every 4 --shadow-compile

# the plant precheck
python3 rhm/practice/crystallize/launch_detached.py --fn cal_plant --tag calp_s0 \
    --n-cycles 20 --gen-lr 1e-4 --gen-steps 20 --n-ref 512

# reduction
python3 rhm/practice/crystallize/analyze_crystallize.py --tag cg_s0 --fetch --figures
python3 rhm/practice/crystallize/analyze_crystallize.py --tag calu_s1 --fetch --cal
python3 rhm/practice/crystallize/analyze_crystallize.py --tag cg_s0 --detector --arm never
```

Modal volume (`rhm-scaling-data`): `/data/rhm_practice_crystallize/<tag>/<arm>/results.json`, with
`setup.json` and `cal_unit.json` / `cal_plant.json` beside them. Figures:
`figures/cg_s0/fig1_metering.png` (per-context `e` and `δ`, compile events marked, stale and
DP-oracle references), `fig2_priced.png` (cumulative priced time × mean error),
`fig3_ground.png` (groundings per solve).

## Caveats

- **One rule draw**, one (v, s, L, m) setting. The 1.8–3.0× library-vs-single margin and
  the 3.6–5.0× averaging margin are large relative to the draw noise; the timing comparisons are not,
  and are reported as unresolved rather than null.
- **`n_cand = 32` leaves sd ≈ 0.07 on library audition.** This is the binding limit on every
  arm-to-arm accuracy comparison in the run and should be raised before timing is asked about again.
- **Post-commit drift is exactly zero by construction** — a frozen unit on a fixed metering set — so
  this node says nothing about committed-unit degradation. Unlike the étude, that is an instrument
  property here rather than a substrate one: the unit *is* state-conditioned, it is only the
  measurement that is fixed.
- **No seam-state shift is available in this scope.** Units launch at the instance start, so the
  optimism gap of ≈ 1.0 confirms the seam account by removing its cause rather than by testing it.
  Sequential assembly, where a unit's launch distribution depends on an upstream commitment, is not
  built here.
- **The declared feedback budget sets the headline.** At 46 groundings/solve a committed library
  matches the beam; at 496 it does not. The full ladder is logged, but a different declared budget
  gives a different verdict and the choice is ours, not the substrate's.
- **The precheck's value staleness** (above) means its wide-beam column is not a fair closed-loop
  reference; only the ceiling columns are load-bearing.

## Next steps — the suggested round 2 ("the ratchet round")

One agent, staged. The precheck says the certificate becomes testable exactly when the plant learns;
this composes that with the depth axis so the certificate has a *sequence* of things to certify.

**Apparatus.** (i) A **learning plant**: the generator fine-tunes on the agent's own successful
repairs with replay 0.5, and the value is **co-adapted** (the precheck's confound). This needs its
own calibration for a resolvable *unit-LP* descent — `cald_s1`'s lesson one level up, now applied to
the ceiling rather than to task error. (ii) A **unit-LP certificate**: certify on the shadow-compile
audition trajectory being positive and then flat, not on task-performance level or task-LP, which
conflate selector and plant improvement; and measure the stale reference **on the metering set**.
(iii) A **depth-laddered damage schedule** (the `level_ladder` idiom): damage moves deeper across the
run, so climbing is necessary rather than merely available. (iv) **Committed level-k macros enter the
practice action space as primitives** for the level-(k+1) era — the ratchet.

**Arms sketch.** `never_base` (base moves only) · `given` (the full DGP level-move vocabulary from
c1 — the `level_moves` ceiling) · `practice_gated` · `practice_early` (the poisoned-region test, now
on a *learning* plant) · `practice_late`.

**Headline readouts.** Priced cost-to-competence per depth era — a cost-to-depth law; the
**earned-vs-given fraction** (how much of `level_moves`' measured 3.51× does *earning* the vocabulary
recover?); whether early commitment on a learning plant now compiles error, where on a frozen plant
it costs nothing; an oracle check that an earned macro corresponds to a true rule; and whether era-k+1
certification arrives *faster* given committed era-k units.

**Risks to design against.** Plant self-imitation collapse (guards: the on-grammar oracle, replay);
value staleness under a moving plant; and `n_cand` raised until draw noise is below the effects being
compared.
