# DESIGN — `scordatura`: the miner mines the failures too

**Brief**: [`SPEC.md`](SPEC.md) (the orchestrator's, verbatim) · **Machinery record**: [`FILES.md`](FILES.md) ·
**Donor**: [`../rubato/rubato.py`](../rubato/rubato.py) at its `rb_g1d` head ·
**Why**: [`../rubato/README.md`](../rubato/README.md) "The update" items 4–6 and "What this does not show",
[`../rubato/DESIGN.md`](../rubato/DESIGN.md) §10–§11, `../rubato/CONVERSATION.md`[^private] §5.

---

## §0 What this node adds, in one paragraph

The practice loop mints vocabulary from pairings that recur in the learner's own **solved** chosen
answers (`mine_from="chosen"` over `ps > 0.5`, `mine_cap` 8 of them a cycle), and the observation
panel that defines a key's consumers reads the same parse of the same rows. Frequency-in-solves and
value are therefore nearly one number there, and rubato found no grader, the oracle's included, that
beat the ungated arm at depth. `scordatura.py` is `rubato.py` plus one substrate knob, `mine_junk`:
the solved draw is left exactly as the donor makes it and rows drawn from the cycle's **unsolved**
chosen answers are appended after it, on a generator of their own, at the outcome-blind proportion.
Everything that reads the mined parse sees them (the committable miners, the panel, G-Y) at one site;
everything that indexes solved instances reads the first rows, which are the solved ones. Beside
every miner a **shadow** miner is fed only the solved rows, so every key carries its total count, its
solved-sourced count and its true-table fraction, and a key at support here but not under the
solve-filtered rule on this arm's own history is **junk-only**. One grade dial rubato only re-applied
post hoc becomes a knob: `adm_grade_persist`, revoke only after k consecutive non-positive priced
passes. With every knob off the file is `rubato.py` (G-F, 0.000e+00 at smoke scale; and on the grade
path, where the fork's edits sit, the preflight's D0/D1 pair is equal in every series, event and
admission record).

---

## §1 Decisions

### §1.1 The draw: append, on its own stream, after the solved rows

`mine_src = out["x"][ps > 0.5]`, then `rng.permutation` and the first `mine_cap` (the donor's lines,
untouched), then **`sc_junk_draw`**: indices of the cycle's unsolved instances (`ps <= 0.5`),
without replacement, in the order `permutation` of the junk's **own** generator
(`np.random.default_rng(seed + SC_JUNK_SEED)`, a state name of its own inside `_mj`) puts them, and
`mine_src = cat([solved rows, junk rows])`.

Why append and not a fully outcome-blind draw: an outcome-blind draw of 8 of all chosen answers would
carry about 2 solved rows a cycle instead of 8, so the true pairings would arrive four times slower
and the arm would confound "junk entered" with "arrival slowed", which no seat can fix. Appending
holds the reinforced diet fixed: the same 8 solved rows from the same permutation on the same shared
stream, so every difference from `st_gn_yk` is the junk. The outcome-blind version is a flag away.

Why its own stream: the donor's shared stream `rng` is drawn again later in the cycle (the value
head's step). A junk draw on it would move every later draw and the first divergence from `st_gn_yk`
would be bookkeeping, not content. J1 checks the stream is where the donor's draw left it, and the
G-F smoke shows the value loss at cycle 1 moves when the junk is drawn on `rng` (J1's red knob) and
is equal to the knob-off arm's when it is not.

### §1.2 The dose: the outcome-blind proportion, per cycle

`sc_junk_count(n_sol, n_uns, n_take, frac)`: an outcome-blind draw carrying the same `n_take` solved
rows the donor took would have carried `n_take * n_uns / n_sol` unsolved ones beside them; `frac`
scales it (`mine_junk` = 1.0 is the proportion itself), rounded half up, never more than there are
unsolved answers. With nothing solved the limit is every unsolved answer (the donor mines everything
when its cap does not bind, so an outcome-blind draw would too). It is read per cycle "as the batch
has them", not as a constant: `st_gn_yk`'s own record (`rb_s1`, `rb_s2`) has 64 instances posed a
cycle, a median solve rate of 0.22 (seed 0) and 0.25 (seed 2), varying by era, so the proportion
moves from about 10 rows a cycle in era 4–5 to 40–60 in era 2:

| seed 0 (`rb_s1`) | era 1 | era 2 | era 3 | era 4 | era 5 |
|---|---|---|---|---|---|
| solve rate q0/q25/q50/q75/q100 | .02/.17/.25/.33/.45 | .05/.14/.17/.20/.27 | .05/.18/.23/.33/.59 | .23/.37/.41/.44/.50 | .33/.36/.42/.45/.48 |
| junk rows a cycle at the proportion, q0/q50/q100 | 10/24/63 | 22/39/61 | 5/26/61 | 8/11.5/26 | 9/11/16 |

(seed 2: solve rate medians .31/.14/.23/.31/.34 by era; junk medians 18/49/26/18.5/15.)

### §1.3 The shadow, junk-only, and the true mask

`sc_shadow_of`: at setup, after preflight's seeds, a miner in the arm's own key (`QT.make_miner`,
the same class map object) holding exactly the miner's counts (empty on a paid arm), for every
committable level (`_mj["sv"]`) and every panel level (`_mj["sv_obs"]`). Every observe of a miner at
the mining site is mirrored on its shadow with the first `_mj_nsv` rows only; a merge re-keys the
shadows with their miners; preflight's top-up seeds reach both. The shadow is therefore the miner
that the solve-filtered rule would have built **on this arm's own history** — not `st_gn_yk`'s
miner, whose history diverges from this arm's as soon as the junk changes a table. The column the
brief asks for ("at support here but not under `st_gn_yk`'s miner") is read two ways on the record:
junk-only against the shadow, per key, in every record; and the arm's at-support counts beside
`st_gn_yk`'s own at the same cycles, per level, in the reduction.

`sc_key_info(key, miner, shadow, support, truth)` → `{n, n_sv, jo, tf}`: `tf` is the share of the
key's spellings (up to 4 per half, the miner's own `spell` ranking) whose concatenation is a row of
the true table at that level — pp3's and the tables' logged precision's mask, per key. A class pair's
spellings share a possible set, so on this grammar `tf` is 0 or 1 (every key on the record so far).
`None` above L5 (no true table).

Where the columns are: every walk step (`_adm.passes[*].levels[*].steps[*].mj`), every pass row
(`mj`: pending, pending junk-only, offered junk-only, whether the offer cap bound), every graded key
(`gr_passes[*].levels[*].keys[*].mj`, `n_consumers_jo`) and every fired consumer (`fired[*].mj`),
every revocation (`revoked[*].mj`, `run`), and every cycle (`log["mj"]`: the junk drawn, J1–J4, and
per committable and panel level the at-support counts both ways with the true mask, the junk-only
keys, the admitted keys' junk-only and true counts, and the junk rows the class map dropped).

### §1.4 The fire survey

`sc_fire_keys` (run-level, an instrument; free at a restore): at the arm's end, after the battery,
every at-support key of the committable miners and of the panel, junk-only and solved-sourced (up to
`sc_fire_cap` 48 of each per source and level, in count order), is fired **single-entry**: the key's
rows alone as a level-l table over what is served at l-1, forced as the macro at the level-l node of
the era's cell, on a fresh pool of `n_aud` instances of that cell (its own seed family), the same
pool graded unfired as the base. That is pp1's fire, the one the grade reuses one level up. World
gain = success fired − success unfired; read gain likewise on the shaped projection's level. Run in
an RNG sandbox with the entry recorder off, and everything the reads touch (the admission counters,
the open bit's fallback count, every miner's `last_build`, the readout's `stat`) put back, so the arm
file is what it would have been without it; the saved state was written inside the loop before it.
A restore AT the stop cycle runs no cycle (added: `_rb_stop` is set when the restored cycle is
already at `rb_stop_cycle`), so the survey can also be run on any saved state offline.

### §1.5 Persistence

`sc_persist(run_prev, why, k)` → (revoke, run, held): want of a consumer revokes at once; a
`worthless` pass extends the run and revokes iff it reaches k, else HOLDS (the key is kept, the run
remembered); `kept` breaks the run; a silent pass (`unpriced`, `no_reader`, `no_diet`, `no_cell`)
neither extends nor breaks it — M-2's own convention in `reduce_rubato.py`. At k = 1 it is rubato's
rule exactly. The per-key run lives in `_adm["gr_run"]` (a grade key, left out of G-I with the
others); a revocation, a re-offer and a merge's re-key at that level clear it. The pass's verdict
stays on the record as `why`; `keep` is the act; a held pass carries `held` and `run`. Only arms with
k > 1 carry the keys, so a rubato graded arm file is unchanged.

### §1.6 What else the junk reaches, checked in the code

- **The shaping target does not move.** `st_gn_yk` shapes on `vo_sh_target="solve"`; the yield label
  `_sb_label` (the one function whose docstring says an unsolved piece contributes nothing) is built
  only when the target is `yield` (`_sb_label_fn` is None otherwise), so the plant's outcome head
  trains on the world's verdicts whatever the miner holds. The per-probe yield instrument
  (`_sb_probe_yield`) reads the miner and is an instrument.
- **Bookkeeping over solved instances stays over the solved rows**: `mine_take`, `_mine_idx` and
  `grec["mine_*"]` are computed before the append; the dose ledger reads `pf[:_n]` with
  `_n = len(_mine_idx)`, the solved rows (J3 checks they are first), and is skipped when nothing
  solved exactly as the donor skips it (the donor's `if mine_src.shape[0]` no longer implies a solve,
  so the conjunct `_mj_nsv > 0` is added).
- **`log["n_mined"]` counts every row observed** (solved + junk), so the miners' `n_obs` increments
  still equal it (the E-9 relation); `log["mj"][c]["n_take"]` / `["n_junk"]` split it.
- **The question port runs `exo`** (the donor's own draw, no selection), so posed instances do not
  depend on the counts; its delivery ledger credits solved rows only.
- **The G-Y instrument reads the same parse** and sees the junk; the panel-vs-G-Y agreement still
  holds (both read the one site).
- **Everything that should see the junk does**: the open inventory's per-cycle rebuild, the walk, the
  merge op's licence (it reads the panel), the recert, the grade's consumers. That is the treatment.
- **A mined row is not assumed solved anywhere else**: the plant's step (`finetune_generator_span`)
  and the port's step train on `solved`, the value readout's bank on the filed writes with their
  verdicts; none reads `mine_src`.
- **The miner's own comment** ("feats read off SOLVED configurations") is now false by design; the
  class map drops a junk row whose half has no legal derivation (`n_dropped`), which is logged per
  level as `junk_dropped`.

### §1.7 The window counts observations, and the junk multiplies them

Rubato set the grade's window at 160 **panel observations** (~20 cycles at 8 rows a cycle). The panel
now observes the solved rows plus the junk, about four times as many rows a cycle at the median dose,
so the same window elapses in about a quarter of the cycles. The brief keeps the window as rubato
built it (a known defect, read the record against it), so this is stated rather than corrected: a key
is graded — and can be revoked for want of a consumer — about four times sooner in cycles. It also
brings the first cycle a grade can fire forward, which sets the graded arms' restore point (§2).

---

## §2 The knobs

| knob | level | default | meaning |
|---|---|---|---|
| `mine_junk` | arm (`sc_*`: 1.0) | 0 (off) | append the cycle's unsolved chosen answers after the solved draw at this multiple of the outcome-blind proportion (§1.1–§1.2) |
| `mj_falsify` | run | `""` | J1–J4's red runs only: `rng`, `solved`, `prepend`, `shadow` |
| `sc_fire_keys`, `sc_fire_cap` | run (free at a restore) | False, 48 | the end-of-arm fire survey (§1.4) |
| `adm_grade_persist` | arm (`sc_grd_yk`: 2) | 1 | revoke a worthless key only after this many consecutive non-positive priced passes (§1.5) |

All in `voicing_run`'s signature; each enters the config only when it is off its default, so a run
with all of them at default has rubato's config exactly. Every rubato knob is inherited unchanged.

Arms (built from the donors' specs by `_sc_arm`, so the only difference is the knob by construction;
all draw `st_gn_yk`'s torch stream, `TWIN` → `enum_live`):

| arm | = | role |
|---|---|---|
| `sc_gn_yk` | `st_gn_yk` + `mine_junk` 1.0 | the ungated junk floor |
| `sc_gw_yk` | `rb_gw_yk` + `mine_junk` 1.0 | the world grade (margin 0, window 160, four consumers): the ceiling as built |
| `sc_grd_yk` | `rb_grd_yk` + `mine_junk` 1.0 + `adm_grade_persist` 2 | the read grade with its diet (512) and persistence |
| `sc_pf_gn`, `sc_pf_gw`, `sc_pf_gwp`, `sc_pf_grd` | the rubato preflight twins + the junk (and persistence 2 / diet 512 as named) | preflight only |

---

## §3 The gates

| gate | where | claim | falsified by |
|---|---|---|---|
| **G-F** | `fidelity_smoke` (`sc_gf1`) | every `[scordatura]` knob off, this file replays `rubato.py` bit for bit at smoke scale, against a donor self-replay control | — (the lineage's gate) |
| **G-F grade path** | `sc_pf` (`sc_pf1`), D0/D1 | the same on the preflight vehicle's world grade at margin 0.5, where revocation fires and the fork's persistence and junk-column edits sit | — |
| **G-J (designed)** | `sc_gates`, CPU | the dose on 10 cases; the draw (unsolved only, the count, its own stream deterministic); the append and the shadow on a toy class miner (the shadow IS the solved-only miner, shares the class map, owns its counts; junk-only B and C, not A or D); the per-level stats | dose over every solved instance; dose rounded down; junk drawn from every answer; the shadow fed every row; the junk prepended; junk-only off by one — 6/6 RED |
| **G-P (designed)** | `sc_gates`, CPU | persistence on 11 designed pass sequences; k = 1 is rubato's rule | a silent pass breaks the run; a kept pass does not; want of a consumer held; off by one — 4/4 RED |
| **J1–J4 (in run)** | every cycle `mine_junk` is on, asserted | J1 the shared stream untouched by the junk draw; J2 every junk row unsolved; J3 the first `n_take` mined rows ARE the donor's solved draw; J4 every miner that observed took every row, its shadow exactly the solved ones, no shadow count above its miner's | `mj_falsify` = `rng` / `solved` / `prepend` / `shadow`, each on its own check (G-F smoke `sc_gf1`, preflight `sc_pf2`) |
| **c1 identity** | `fidelity_smoke` | with the junk on, cycle 1's practice beam, plant step and value step equal the knob-off arm's | the `rng` red knob moves the value step |
| inherited | R-0, R-2, G-I, G-W, G-S, G-D, G-O, G-N | rubato's, unchanged; R-0 now over 76 state names (`_mj` added) | rubato's |

---

## §4 Results

### §4.1 The gates, each shown to fail first

**G-F** (`fidelity_smoke`, tag `sc_gf1`, `ap-km6iwcNJuJoWTWEYgA0suj`): every `[scordatura]` knob off, the
fork against `rubato.run_arm` in the same container on the same smoke substrate, `anchor` and
`given_c1`: max|fork − rubato| = **0.000e+00** on all 13 series, donor self-replay control 0.000e+00,
commit events equal on both arms. PASS.

**G-F on the grade path** (`sc_pf`, tag `sc_pf1`, `ap-LzFcJ4PySByXxPW7sByckE`, gate R-1's vehicle, one
container): D0 (rubato's own `rb_pf_gw` at margin 0.5, run by `rubato.run_arm`) against D1 (this
file's, knobs off), 12 cycles, 10 revocations each: every series max|Δ| = 0, the whole admission
record (`adm`, every grade pass, fire and revocation) identical, every event identical, no extra
arm-file key on D1. The one log key that differs is `entry`: the entry recorder is a module global of
the file that installed it, so only the fork's channel is live with two modules in one process
(rubato's R-1 donor run notes the same, "reported, never asserted").

**G-J designed and G-P** (`sc_gates`, CPU, also inside `rb_gates`): G-J dose 10/10 cases, draw,
shadow, stats; G-P 11/11 sequences and k = 1 is rubato's rule. Falsified **10/10 RED**: dose over
every solved instance; dose rounded down; junk drawn from every chosen answer; the shadow fed every
row; the junk prepended; junk-only off by one; a silent pass breaks the run; a kept pass does not
break it; want of a consumer held by persistence; persistence off by one. R-0 (now 76 state names),
its 4 perturbations RED; G-D, G-O, G-N and their 8 perturbations RED, unchanged.

**J1–J4 in run.** Clean: the G-F smoke's junk arm (`anchor` + `mine_junk` 1.0, 3 cycles) green on
every check every cycle; the preflight floor `sc_pf_gn` 12/12 cycles; the pre-check `sc_p1` 40/40 on
each of J1–J4. Red knobs: on the G-F smoke substrate `rng` RED (J1) and `shadow` RED (J4), but
`solved` and `prepend` were **BLIND** there — that substrate solved no instance in its first three
cycles, so there was nothing solved to draw by mistake and no solved row to displace. Re-run on gate
R-1's vehicle, which solves (`sc_pf2`, `ap-GgkYKszaafrfD8wkjKhxQ9`, 8 cycles each): **4/4 RED, each on
its own check only** — `rng` J1 on 8/8 cycles, `solved` J2 and `prepend` J3 on the one cycle with a
solved instance (c6), `shadow` J4 on 8/8.

**c1 identity** (G-F smoke): with the junk on, cycle 1's practice error, success, solved count, plant
loss and value loss are the knob-off arm's to the bit; with the junk drawn on the shared stream (J1's
red knob) the value loss at cycle 1 moves (0.009402 → 0.012348) and nothing before it does.

**The preflight's other runs** (`sc_pf1`): N (`sc_pf_gn`) J clean, the survey ran at c12; GW (`sc_pf_gw`
from c1) G-I equal to N by hash at c2, first revocation c4; GRs (`sc_pf_grd` restored from N's c2:
the arms-of-record path) R-2 PASS, schema evolution added the grade's keys and persistence's
(`gr_persist`, `gr_run`, `n_held`), no G-I save before its first revocation (c4); GM1 (world, margin
0.5) 23 worthless revocations; GM2 (the same with persistence 2) 23 held passes, 15 worthless
revocations, every revocation at run 2 and every hold at run 1.

### §4.2 The pre-check, seed 0 (`sc_p1`, `figures/sc_p1_dose.txt`; `ap-ZEt3QV4MskVCh6HEF7ThFp`)

`sc_gn_yk` from c0 to its stop at c40, saves every 5 cycles. Preempted after c5 and auto-resumed from
its own c5 save (R-2 PASS; both segments on non-AVX-512 hosts). 0.31 GPU-h, 10.4 s a cycle over
c1–c40 (`st_gn_yk`'s whole-run mean is 29.6), peak RSS 5.7 GB. J1–J4 40/40.

**The junk mined.** 313 solved rows and 975 junk rows over c1–c40, 3.12 junk rows per solved row;
solved instances a cycle q0/q25/q50/q75/q100 3/14/17/21/30 of 64; junk rows a cycle 9/16/22/29/61.
The class map dropped no junk row (every half of every junk row has a legal class).

**Keys at support** (here / shadow / junk-only (true / false) / `st_gn_yk`'s at the same cycle):

| c | L2, the committable miner | L3, the panel | L4, the panel |
|---|---|---|---|
| 5 | 21 (10 T / 11 F) / 5 / 16 (6 / 10) / 5 | 19 (4 / 15) / 1 / 18 / 2 | 0 / 0 / 0 / 0 |
| 10 | 24 (12 / 12) / 12 / 12 (4 / 8) / 10 | 36 (10 / 26) / 7 / 29 / 7 | 1 / 0 / 1 / 0 |
| 20 | 35 (12 / 23) / 14 / 21 (2 / 19) / 12 | 67 (20 / 47) / 20 / 47 / 18 | 13 / 2 / 11 / 0 |
| 30 | 44 (13 / 31) / 16 / 28 (2 / 26) / 13 | 83 (23 / 60) / 28 / 55 / 31 | 32 / 4 / 28 / 0 |
| 40 | 47 (13 / 34) / 16 / 31 (2 / 29) / 13 | 104 (24 / 80) / 38 / 66 (10 / 56) / 39 | 47 / 6 / 41 / 2 |

The shadow tracks `st_gn_yk` (L2 16 vs 13, panel L3 38 vs 39 at c40); the junk adds 31 L2 keys (29
false) and 66 L3 panel keys (56 false) at c40.

**The walk.** The offer cap of 8 bound at the c5, c10, c15 and c20 passes (pending 21, 16, 14, 11)
and not after; every at-support L2 key was admitted by c35 (45 admitted at the stop, 29 of them
junk-only now; 13 true, 32 false). The admitted L2 table's row precision fell from 0.50 (c5) to 0.28
(c40), against `st_gn_yk`'s 0.80 → 0.69 over the same passes. The walk's own base-plus-one fires (the
only fire on the record where a key can hurt: the admitted table so far plus the candidate, on the
gate pool of the era's cell): junk-only false L2 keys 32 offered, world −/0/+ 7/23/2; junk-only true
2/3/0 of 5; solved-sourced false 0/1/1; solved-sourced true 1/4/3; `st_gn_yk` over c1–c40 4/4/5 of 13.

**The fire survey** (c40, single-entry at the key's own cell; the unfired pool's success is 0.0 at
L2 and L3, so a single-entry gain cannot be negative here, and the readout is not yet fit — read gain
0 on every fire):

| L2, committable | n | world gain −/0/+ | gain q0/q50/q100 |
|---|---|---|---|
| junk-only, false | 29 | 0/28/1 | 0 / 0 / .26 |
| junk-only, true | 2 | 0/0/2 | .016 / .016 / .21 |
| solved-sourced, false | 5 | 0/2/3 | 0 / .17 / .21 |
| solved-sourced, true | 11 | 0/0/11 | .016 / .21 / .44 |

| L3, panel (the consumers; up to 48 of each class) | n (unbuildable) | world gain −/0/+ | gain q0/q50/q100 |
|---|---|---|---|
| junk-only, false | 38 (3) | 0/25/10 | 0 / 0 / .31 |
| junk-only, true | 10 (2) | 0/0/8 | .08 / .20 / .47 |
| solved-sourced, false | 24 (1) | 0/6/17 | 0 / .20 / .31 |
| solved-sourced, true | 14 (0) | 0/0/14 | .06 / .26 / .47 |

L4 panel keys (47, 41 junk-only) are not buildable at c40 (no L3 table is served in era 1).

**The dose, as read.** Junk-only keys reach support at both levels (31 of 47 at L2, 66 of 104 in the
L3 panel at c40), so it is not "nothing"; the true set arrives as it does without the junk (the
shadow tracks `st_gn_yk`), and the cap bound for four passes and then cleared, so it is not a filter
at this level. What enters is mostly value-junk: 28 of 29 junk-only false L2 keys have single-entry
world gain exactly 0, and 7 of 32 junk-only false offers harm the growing table in the walk's own
fire; pp3-style passenger junk exists too (false keys with positive gain: 4 at L2, 27 among the L3
consumers). The dose was kept at `mine_junk` 1.0.

**The restore cycle, from this record.** The panel observes ~32 rows a cycle (8 without the junk),
so keys admitted at the c5 pass are 116 panel observations old at the c10 pass and 294 at c15; the L3
panel holds keys at support from c1. c15 is the first cycle a grade can fire on seed 0, and **c10** is
the latest save before it.

**Seed 2's first segment** (`sc_p2`, `figures/sc_p2_dose.txt`; `ap-8Lq665cCq0TPjUWhvjPK5T`, 0.36 GPU-h, not
preempted), run as the seed-0 pre-check was so both floors have saves every 5 cycles to c40 and a
c40 survey: 317 solved and 897 junk rows (2.83 per solved row); J1–J4 40/40. At c40 the L2 miner
holds 48 keys at support (14 T / 34 F), its shadow 13, `st_gn_yk` 13, junk-only 35 (5 T / 30 F); the
L3 panel 87, shadow 38, `st_gn_yk` 38, junk-only 49. The offer cap bound at 5 passes; 47 L2 keys
admitted at the stop (34 junk-only now). Walk fires: junk-only false 30 offered, world −/0/+
10/20/0. Survey at L2: the 30 junk-only false keys all have single-entry world gain 0; junk-only
true 5/5 positive; solved-sourced false 3 of 4 positive; solved-sourced true 9/9 positive. L3 panel:
junk-only false 23 zero / 10 positive (2 unbuildable), solved-sourced false 5 / 21. **The restore
cycle**: keys admitted at the c5 pass are 161 panel observations old at the c10 pass, so c10 is the
first cycle a grade can fire on seed 2 and **c5** is the latest save before it.

### §4.3 Phase 2, the launches

| tag | what | from | app |
|---|---|---|---|
| `sc_s1` | the seed-0 floor `sc_gn_yk`, c41–c201, survey at c201 (`RUN_sc_s1.sh`) | `sc_p1`'s c40 (R-2 PASS) | `ap-CIiBcAwMAcNqwPJuirsOf4` |
| `sc_g1` | seed 0: `sc_gw_yk` and `sc_grd_yk`, two containers (`RUN_sc_g1.sh`) | `sc_p1`'s c10 (R-2 PASS on both) | `ap-0fRzRL9i3QnsNzIcEHm4Rq` |
| `sc_s2` | the seed-2 floor, c41–c201 (`RUN_sc_s2.sh`) | `sc_p2`'s c40 | `ap-cKMiPXBu2v7xkaeKB4AvzE` |
| `sc_g2w`, `sc_g2r` | seed 2: `sc_gw_yk`, `sc_grd_yk`, as separate tags so no more than four containers run at once (`RUN_sc_g2w.sh`, `RUN_sc_g2r.sh`) | `sc_p2`'s c5 (R-2 PASS on both) | `ap-JzJg9RVEYGYeNxqT9g3Rlm`, `ap-lEOVcFPqmCtEYgw9kX5a0S` |

All seven runs completed. Cost: ≈12.7 GPU-h in the final container segments (floors 0.31 + 1.40 and
0.36 + 1.30; graded seed 0 1.99 + 2.69, seed 2 2.08 + 2.60), plus the preempted partial segments and
the gates (≈2 GPU-h). Seconds a cycle: floor 21.3 / 19.6 (c41–c201), world 32.3 / 33.8, read 44.8 /
43.1 (the read fires on every pass, below the diet included, so both values stay on the record).
Peak RSS 7.9–8.2 GB against 12 GB requested.

### §4.4 Identity (`figures/sc_grade_s0.txt`, `sc_grade_s2.txt` [G1])

- R-2 PASS at all nine restores: the floors from their own c40 and after their preemptions from their
  own c170 (seed 0's pre-check also from its own c5); the four graded arms from the floor's c10 (seed 0)
  and c5 (seed 2).
- J1–J4 green on 201 of 201 cycles on all six arms.
- G-I by hash: on seed 2's read arm, equal to the floor at c10 and c20 (first revocation c25). The other
  three graded arms revoked first at c15 (seed 0, both) or c10 (seed 2 world), before any save they
  share with the floor.
- G-I off the log: on every graded arm the first cycle any per-cycle series differs from the floor is its
  first revocation (seed 0 c15 and c15, seed 2 c10 and c25), and it is the open inventory's record that
  differs first.
- G-W equal on all four graded arms; G-S never red.
- Host types: the floors crossed from non-AVX-512 to AVX-512 hosts between segments, as did seed 2's
  world arm at its restore. That is the lineage's one-ulp floor, not a gate.

### §4.5 Results, both seeds (`figures/sc_seedtable.txt`; seeds side by side, never averaged)

**Task error by era.** The mean over each era's cycles; eras 4 and 5 are 12 and 9 cycles.

| arm | seed 0, minus `st_gn_yk`, eras 1..5 | seed 2, minus `st_gn_yk` | seed 0, minus the junk floor | seed 2, minus the junk floor |
|---|---|---|---|---|
| `sc_gn_yk` (floor) | +.002 +.041 +.118 +.176 +.240 | +.000 +.008 +.045 +.063 +.247 | 0 | 0 |
| `sc_gw_yk` (world) | −.003 −.030 +.015 +.180 +.236 | −.000 −.000 +.004 −.020 +.173 | −.004 −.071 −.103 +.004 −.005 | −.001 −.008 −.041 −.083 −.074 |
| `sc_grd_yk` (read) | −.001 +.018 +.122 +.261 +.132 | +.000 +.013 +.018 −.019 +.168 | −.003 −.022 +.004 +.085 −.108 | +.000 +.005 −.027 −.082 −.078 |
| `rb_gw_yk` (rubato) | −.004 −.002 +.035 +.176 +.053 | −.001 −.033 −.002 +.016 +.008 | — | — |
| `rb_grd_yk` / `rb_gr_yk` | +.000 +.002 +.029 +.081 +.227 (grd) | +.000 +.026 +.009 +.029 +.092 (gr) | — | — |

The floors' mean task error, eras 1..5: seed 0 .503 .675 .693 .635 .775; seed 2 .470 .623 .639 .639 .815.
`st_gn_yk`'s: .502 .634 .576 .459 .535 and .469 .614 .595 .576 .569.

**The served table at c201.** Keys admitted and not revoked, with the junk-only count and the true /
false keys in brackets. Most L3+ keys are mixed after the merges (§5.6). L2's row precision at the last
pass follows.

| arm | seed 0 | seed 2 |
|---|---|---|
| `st_gn_yk` | L2 17, L3 16, L4 9, L5 7; L2 precision .65 | L2 18, L3 24, L4 19, L5 9; .67 |
| floor | L2 61 (43 jo; 14 T / 47 F), L3 153 (132), L4 130 (105), L5 38 (23); .23 | L2 57 (39; 14 / 43), L3 136 (110), L4 98 (70), L5 38 (29); .25 |
| world | L2 19 (3; 14 / 5), L3 36 (23), L4 19 (8), L5 19 (13); .70 | L2 19 (4; 14 / 5), L3 36 (23), L4 26 (15), L5 19 (5); .70 |
| read | L2 26 (10; 12 / 14), L3 93 (76), L4 27 (13), L5 25 (15); .47 | L2 23 (6; 13 / 10), L3 67 (44), L4 17 (8), L5 9 (5); .57 |

Walked keys (ever offered) on the floor: L2 61 / 57, L3 171 / 141, L4 133 / 136, L5 38 / 39, against
`st_gn_yk`'s 17 / 18, 19 / 24, 9 / 22, 7 / 9.

Of the floor's walked L3 keys, a half is still held revoked at the end on:

| arm | seed 0 (of 171) | seed 2 (of 141) |
|---|---|---|
| world | 104 | 81 |
| read | 77 | 69 |

For L4 keys: world 4 of 133 and 9 of 136; read 0 on both seeds.

**Revocations.**

| | seed 0 world | seed 0 read | seed 2 world | seed 2 read |
|---|---|---|---|---|
| first revocation | c15 | c15 | c10 | c25 |
| revocations (no consumer, worthless) | 271 (104, 167) | 263 (202, 61) | 271 (103, 168) | 225 (149, 76) |
| of them junk-only / false (tf 0) / mixed | 237 / 256 / 15 | 242 / 237 / 11 | 231 / 256 / 15 | 200 / 192 / 11 |
| worthless revocations of TRUE keys (tf 1) | 0 | 14 (L2: 4 jo, 10 sv) | 0 | 21 (L2: 8 jo, 13 sv) |
| re-offers | 148 | 106 | 149 | 86 |
| held revoked at the end L2 / L3 / L4 | 40 / 37 / 37 | 32 / 35 / 73 | 37 / 42 / 25 | 33 / 33 / 62 |
| fires; world queries billed to the grade | 4,853; 931,776 (4,983 a cycle) | 7,427; 0 | 5,038; 967,296 (5,038 a cycle) | 5,596; 0 |

Rubato's world arm billed 2,335 and 2,232 queries a cycle on the solve-fed substrate.

The world arm revoked no true key on either seed. Every one of its worthless revocations at L2 is a
false key: 98 junk-only and 16 solved-sourced at seed 0, 88 and 15 at seed 2.

**The two currencies on the same fires**, L2 priced key-passes ([G6]):

| | seed 0 | seed 2 |
|---|---|---|
| read arm: passes where the world revokes (world gain exactly 0) — the read revokes | 127 of 300 | 111 of 248 |
| read arm: passes where the world keeps — the read revokes | 34 of 347 | 72 of 390 |
| world arm: world revokes / keeps; the read's counterfactual (no diet) revokes of each | 114 (37) / 660 (51) | 103 (39) / 658 (58) |

By class, read arm: on junk-only false L2 keys the read revoked 123 of 262 world-zero passes and kept 139
at seed 0, and revoked 107 of 200 at seed 2. It never revoked a junk-only false pass the world kept at
seed 0; it did so on 5 at seed 2. On solved-sourced true keys it revoked 25 of 218 world-kept passes at
seed 0 and 33 of 241 at seed 2. The world gain on every priced L2 pass is 0 or positive (no negative).

**The diet at the seam** ([G7]):
- The read arm's L2 consumers are first priced at c105 at seed 0, the same pass as `rb_grd_yk`. At seed 2
  it is c80, since span-4 fit rows reach 512 earlier there.
- L3 consumers are first priced at c197 on both seeds; L4 consumers never.
- The buffer by span tracks `rb_grd_yk`'s at the same cycles within about 15% at seed 0.
- The bank carries a different diet. Its held-out positive rate (the share of filed writes that solved),
  the junk floor against `st_gn_yk`: seed 0 .051 vs .115 (c60), .080 vs .147 (c100), .173 vs .300 (c201);
  seed 2 .068 vs .134 (c100), .187 vs .266 (c201). The world arm's bank sits at `st_gn_yk`'s (.173,
  .263 at c100 and c201 on seed 0); the read arm's at the floor's (.080, .171). Held-out AUC .80–.93
  on every arm.
- Key-passes silent for the diet: 1,494 at seed 0 and 869 at seed 2 (L2 + L3 + L4); `no_reader` 197
  and 254.

**Persistence** ([G8]):

| | seed 0 | seed 2 |
|---|---|---|
| held passes | 128 | 133 |
| the world keeps / revokes on the same fires | 22 / 106 | 48 / 85 |
| the key's next priced pass: worthless (revoked at run 2) | 61 | 76 |
| the key's next priced pass: kept (the run broke) | 30 | 22 |
| the key's next priced pass: none | 37 | 35 |

No run reached 3. At k = 1 every held pass would have been a revocation.

**The fire survey at the floor's end** (c201, era 5; single-entry at the key's own cell, whose unfired
success is 0):
- Junk-only false keys: 40 of 42 world gain 0 at L2 (seed 0) and 36 of 38 (seed 2); 41 of 41 and 44 of 44
  at L3; 40 of 41 and 38 of 39 at L4.
- At L5, where the cell is the era's own, junk-only false keys are 13 zero / 4 positive (seed 0) and
  11 / 15 (seed 2).
- Solved-sourced true keys are positive at every level on both seeds.
- The read's gain on junk-only false L2/L3 keys is negative on 41 of 42 and 38 of 41 (seed 0), and 34 of
  38 and 34 of 44 (seed 2).

---

## §5 Defects and facts, in order, each beside its correction

**§5.1 Two of G-J's in-run red knobs were blind on the G-F substrate.** The smoke's junk arm solved no
instance in its three cycles, so `solved` had no solved instance to draw by mistake and `prepend` no
solved row to displace (`sc_gf1`: BLIND, BLIND). Re-run on gate R-1's vehicle, which solves
(`sc_pf2`): 4/4 RED, each on its own check only. The designed CPU gate had already covered both.

**§5.2 Preemptions, recovered by `rb_auto_resume`, four times.**
- The pre-check after c5 resumed from its own c5.
- Both floors after c170 resumed from their own c170 (the c180–c192 lines in their launch logs are the
  resumed segment's).
- No cycle was lost beyond the ones between the save and the preemption.
- The segments landed on both host types (§4.4).

**§5.3 The graded arm files are about five times rubato's.** 105 MB for `sc_grd_yk` at seed 0, 94 MB of
it `log["quot"]`: the quotient's per-cycle accounting grows with the class inventory, which the junk
multiplies. No reducer reads it. `fetch_compact.py` now writes it beside the mirror as `quot.json.gz`
and leaves the bank, heads and rows dumps on the volume. The mirrors are 12–15 MB.

**§5.4 A single-entry fire cannot be negative at its own cell.** The survey's unfired pool is damaged
at exactly the cell the key's macro is forced at, so its success is 0, and a key either repairs some
instances or none. "Strictly harmful" is therefore read off the walk's own base-plus-one fires, where a
key joins a table and can displace a correct row. At the pre-check, 7 of 32 junk-only false L2 offers
harm there at seed 0 and 10 of 30 at seed 2.

**§5.5 The window elapses about four times faster in cycles** (§1.7). This is a known defect, kept as
built by the brief. The first grade fires at c15 / c10, against rubato's c25 / c30, and a key can be
revoked for want of a consumer after about 5 cycles instead of about 20. No-consumer revocations are 104
/ 103 (world) and 202 / 149 (read), against rubato's world arm's 8 / 7.

**§5.6 The true mask is mixed after a merge.** A class pair's spellings share one possible set until a
merge coarsens a half's class; after the L2 merges (the first taken at c68 on both floors, as on
`st_gn_yk`) most L3/L4 keys have
0 < `tf` < 1. The reductions label them M and count only tf = 1 as true and tf = 0 as false.

**§5.7 The dose was kept at `mine_junk` 1.0** (§4.2): junk-only keys reach support at every level, most
of them value-junk, and the offer cap bound only for the first four or five passes.

**§5.8 Small ones.** `fidelity_smoke`'s header print still named the grandparent (corrected). A save's
own record is not in the `rb_hist` it writes (the lineage's rule, unchanged), so a segment's last save
appears in the files and on the volume but not in the next segment's `prior` chain.

---

## §6 Reproduction

```bash
cd experiments/            # MODAL_PROFILE=chromatic
B=rhm/practice/voicing/sotto_voce/aliquot/scordatura
python3 -c "from rhm.practice.voicing.sotto_voce.aliquot.scordatura import scordatura as S; S.rb_static_gate(); S.rb_static_falsify(); S.rb_grade_gates(); S.sc_gates()"   # R-0, G-D/G-O/G-N, G-J/G-P
python3 $B/launch_detached.py --fn fidelity_smoke --tag sc_gf1      # G-F + in-run G-J
python3 $B/launch_detached.py --fn sc_pf --tag sc_pf1               # the preflight (G-F on the grade path, persistence, the restore path)
sh $B/results/RUN_sc_pf2.sh                                         # J1-J4's red knobs on R-1's vehicle
sh $B/results/RUN_sc_p1.sh; sh $B/results/RUN_sc_p2.sh              # the floors' first segments (pre-check), both seeds
python3 $B/fetch_compact.py --tag sc_p1 --fetch --replace && python3 $B/reduce_scordatura.py --dose --tag sc_p1
python3 $B/fetch_compact.py --tag sc_p2 --fetch --replace && python3 $B/reduce_scordatura.py --dose --tag sc_p2 --seed 2
sh $B/results/RUN_sc_s1.sh; sh $B/results/RUN_sc_s2.sh              # the floors to c201
sh $B/results/RUN_sc_g1.sh                                          # seed 0 graded, from sc_p1's c10
sh $B/results/RUN_sc_g2w.sh; sh $B/results/RUN_sc_g2r.sh            # seed 2 graded, from sc_p2's c5
for t in sc_s1 sc_s2 sc_g1 sc_g2w sc_g2r; do python3 $B/fetch_compact.py --tag $t --fetch --replace; done
python3 $B/reduce_scordatura.py --grade --seed 0; python3 $B/reduce_scordatura.py --grade --seed 2
python3 $B/reduce_scordatura.py --seedtable
```

[^private]: Not mirrored: this link points to a document in the private lab repo (the roadmap, the queue, an unrun spec, reading notes, or a conversation). See the top-level README for what is held back and why.
