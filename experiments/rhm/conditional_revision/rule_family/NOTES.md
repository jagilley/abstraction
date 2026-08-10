# Working notes / handoff — rule_family

**Not a writeup.** Results are discussed with Jasper before any README (repo convention).
This file is written to be **self-contained**: a fresh session should be able to run
Gates 1–2 from this file plus [FILES.md](FILES.md), with no other context.

**Up**: [../README.md](../README.md) · **Files**: [FILES.md](FILES.md)
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md)
**Date**: 2026-08-09/10 · Single seed (42) throughout. Regime `v16 s2 L6 m4`,
model `8L/8H/256D` — inherited from the parent cut so its reference lines transfer.

---

## 1. What this sub-experiment is, and why

The parent cut ([`../README.md`](../README.md)) measures belief revision on **one fixed
rule set**. There, reducibility is **static**: which positions are synonym-slots and which
are disambiguators is fixed by the DGP, so an NTP-optimal model can bake the split into
weights and never compute it. Under a fixed rule set RHM is Markov in the ancestor chain,
so the model's minimal parse-tracking is *optimal* — which means the parent's headline
constraint ("the binding constraint is belief depth") may be **regime-induced rather than
intrinsic**.

Here a context window is drawn from one of **R rule sets** (same `v/s/L/m`, different
composition tables), so the model must infer the active rules in-context. Consequences:

- Reducibility becomes **state-dependent within the context**: the same position is
  revision-heavy early (rules unknown → the token is evidence) and synonym-noise late
  (rules identified). The aleatoric null can no longer live in weights.
- With finite R the exact oracle survives as a **mixture over R junction trees**, and
  revision splits exactly by the KL chain rule into
  **`B_total = rule_revision (slow) + E_r[parse_revision^(r)] (fast)`**.
- The parent's single-rule-set results are the **zero-conflict floor endpoint**
  (`differ_levels=[]`), reached through identical code.

**The constraint that governs the whole design.** Summed over a window,
`E[rule revision] = I(r ; x_window) ≤ ln R`. The entire slow component is capped, so signal
strength and transient duration **trade off along a fixed budget**. This is
RHM_META_LEARNING's "thin compositional signal" restated information-theoretically, and it
is why Gate −1 exists.

---

## 2. Status board

| item | status |
|---|---|
| Gate −1 — design selection, exact oracle, CPU | **done** |
| Gate 0 — substrate, both designs vs matched floor arms | **done** |
| diagnostics — position sweep, retention, emergence, held-out | **done** |
| `rule_retention` part 1 — DGP-side positive control | **done** |
| `rule_retention` part 2 — model-side retention curve | **`p2c` collected and VOID** — unidentifiable label, see §5.1. Superseded by `part2b` (tag `p2d`, **done** — §5.2) |
| retrain bundle — 2 phase protocols × 2 arms, 36k steps | **done** (tag `rt36k`) |
| Gate 1 — revision readouts | **done** (tag `g1`, [`gate1.py`](gate1.py)) — **primary kill FIRES**, see §8.2 |
| Gate 2 — temporal FM | **NOT RUN, and must not be** — Gate 1's kill forbids a belief-space Gate 2 (§8.2) |

**A fresh session should**: collect `p2c` (§4 has the paths), then run Gate 1 per §8 **on the
36k ALIGNED checkpoints**. Do not re-derive the designs; they are pre-registered and the
kills are binding.

### The 36k retrain result — read this before anything else

Gate 0's headline was **badly undertrained**. At 36k the aligned protocol reaches an ICL
fraction of **0.645** (vs 0.226 at 12k), with the oracle shape match *improving* to
**+0.997** and rule decodability at **7.6× chance** (0.118 at depth 4, ~19% of the 0.614
Bayes ceiling). Continuity check: the 36k run's 12k trace point reads +0.00408 against the
headline +0.0040 ± 0.0003. Still climbing at 36k though decelerating, so **0.645 is a lower
bound**.

Net ICL (family − floor) at depth 7:

| protocol | 12k | 24k | 36k | ICL fraction @36k | corr(net, oracle) |
|---|---|---|---|---|---|
| **aligned** | +0.0041 | +0.0104 | **+0.0122** | **0.645** | **+0.997** |
| hidden | — | — | +0.0071 | 0.376 | +0.975 |

**CORRECTION — use the ALIGNED protocol.** These notes previously recommended switching to
phase-hidden training for Gate 1, on the argument that boundaries are where rule-revision
lives and the aligned model discards there. That was **wrong**, and running both arms is what
caught it. Phase-hidden is worse on realized ICL (0.376 vs 0.645) for two reasons visible in
the numbers: it is a harder task overall (depth-0 loss 1.5113 vs 1.4330), so capacity goes to
handling arbitrary phase rather than to rule inference; and its floor arm shows a **+0.046
positional artifact**, ~80× aligned's +0.0006, because a random offset makes later window
positions carry more within-sequence context on average. Phase-hidden is retained only as the
control that establishes the discard-schedule finding in §7 — not as the substrate to build on.

---

## 3. Reproduction

```bash
cd experiments

# --- CPU, local, no Modal, no model ---
python3 -m rhm.conditional_revision.rule_family.family            # DGP self-test
python3 -m rhm.conditional_revision.rule_family.oracle_mixture    # oracle vs brute force
python3 -m rhm.conditional_revision.rule_family.gate_minus1 --sweep --K 8 --n-windows 60
python3 -m rhm.conditional_revision.rule_family.gate_minus1 --full --design d2_R64_nF2
python3 -m rhm.conditional_revision.rule_family.rule_retention    # part 1 (~4 min)

# --- Modal (chromatic workspace, L4) ---
# Gate 0 (~45 min): trains family + matched floor arm, both from scratch
modal run --detach -m rhm.conditional_revision.rule_family.gate0_family::gate0 \
    --design d2_R64_nF2 --k-seqs 8 --n-rule-windows 8192 --tag g0

# the 36k retrain bundle, one job per phase protocol (~2 h each)
for ph in aligned hidden; do
  modal run --detach -m rhm.conditional_revision.rule_family.gate0_family::gate0 \
    --design d2_R64_nF2 --k-seqs 8 --phase $ph --base-steps 36000 \
    --ckpt-every 12000 --icl-trace-every 2000 --n-rule-windows 8192 \
    --force-retrain --tag rt36k
done

# diagnostics, all forward-passes-only on cached checkpoints
modal run --detach -m rhm.conditional_revision.rule_family.probe_positions::sweep \
    --design d2_R64_nF2 --tag pp
modal run --detach -m rhm.conditional_revision.rule_family.retention::retention \
    --design d2_R64_nF2 --tag ret
modal run --detach -m rhm.conditional_revision.rule_family.rule_retention::part2 \
    --design d2_R64_nF2 --n-seq 5000 --tag p2b

# Gate 1 (~50 min): 8 CPU oracle shards + one L4 for both arms
modal run -m rhm.conditional_revision.rule_family.gate1::gate1 \
    --n-windows 96 --n-probe-windows 384 --oracle-shards 4 --oracle-sub 48 \
    --rule-probe-steps 400 --dir-steps 200 --n-boot 20 --tag smoke     # ~10 min, attached
modal run --detach -m rhm.conditional_revision.rule_family.gate1::gate1 \
    --n-windows 3200 --n-probe-windows 6144 --oracle-shards 8 --tag g1
```

**Workspace is `chromatic`** (`modal profile list` must show `chromatic` active). Use
`--detach` for anything over ~2 min.

---

## 4. Cached artifacts (volume `rhm-scaling-data`, workspace `chromatic`)

All under `/data/v16_s2_L6_m4_distinct/rule_family/`.

**Checkpoints** (`8L/8H/256D`, seed 42, K=8 windows of 512 tokens):

| file | what |
|---|---|
| `d2_R64_nF2_K8_family_8L8H256D_steps12000_seed42.pt` | **the surviving design**, family arm |
| `d2_R64_nF2_K8_floor_8L8H256D_steps12000_seed42.pt` | its matched floor arm |
| `d2_R128_nF4_K8_family_…steps12000_seed42.pt` | the killed design, family arm |
| `d2_R128_nF4_K8_floor_…steps12000_seed42.pt` | its matched floor arm |
| `d2_R64_nF2_K8_{family,floor}_8L8H256D_steps36000_seed42_at{12000,24000,36000}.pt` | **the 36k ALIGNED bundle — build Gates 1–2 on `_at36000`** |
| `d2_R64_nF2_K8_hidden_{family,floor}_…_at{12000,24000,36000}.pt` | the phase-hidden control arm (worse ICL; control only) |

**Results JSONs** (tag in the filename):

| tag | file | contents |
|---|---|---|
| `g0` | `gate0_<design>_K8_g0_seed42.json` | Gate 0, first pass |
| `g0b` | `gate0_<design>_K8_g0b_seed42.json` | **paired SEMs + corrected probe position — use this one** |
| `g0c`/`g0d` | `gate0_d2_R128_nF4_K8_g0c…` / `gate0_d2_R64_nF2_K8_g0d…` | emergence traces + held-out transfer |
| `rt36k` | `gate0_d2_R64_nF2_K8{,_hidden}_rt36k_seed42.json` | the 36k bundle |
| `pp` | `probepos_<design>_pp_seed42.json` | read-position sweep |
| `ret` | `retention_<design>_ret_seed42.json` | boundary retention contrast |
| `p2b` | `rule_retention_p2_d2_R64_nF2_p2b.json` | model-side retention curve |

Gate −1 sweep output is local only (not on the volume); regenerate with §3 in ~20 min.

---

## 5. Results

### Gate −1 — design selection (exact, model-free)

Two designs were carried forward, chosen from a 17-design sweep by how the `ln R` budget is
**spread across the window**:

| design | budget used | share of rule info by sequence 0…7 |
|---|---|---|
| `d2_R128_nF4` | 4.92 nats | 0.77 / 0.20 / 0.03 / 0 … — a strong two-point contrast |
| `d2_R64_nF2` | 4.24 nats | 0.28 / 0.31 / 0.14 / 0.12 / 0.06 / 0.04 / 0.03 / 0.03 — a graded trajectory |

Measured, not assumed: differing the **leaf-emitting** table identifies ~3× faster (p50 at
token 6 vs 14) — which is why it is shared; differing at **high** levels is slow but
invisible (0.0007 nats/token at d5, against a model whose root recovery is 0.088).

Oracle verification: **2.9e-15** against brute-force enumeration over `(r, latents)` on
three tiny regimes; **0.00e+00** against the parent's `../oracle.py` on overlapping columns;
exact reduction to the single-rule oracle at the floor; both identities at Monte-Carlo error.

### Gate 0 — ICL is present, thin, and only in the graded design

Primary instrument is a **matched depth swap**: the *same* probe sequence P is read at every
context depth `k`, as `[f_1…f_k, P, f_{k+1}…f_{K-1}]`, so P's tokens, parse and rule set are
identical across conditions and only the preceding context varies (the `arity_torque` idiom).
The floor arm's contrast is flat to four decimals, giving a **noise floor of ~0.0002 nats**
against an available signal of 0.019–0.061.

| depth 7 | `d2_R64_nF2` | `d2_R128_nF4` |
|---|---|---|
| oracle available | 0.0189 nats | 0.0610 nats |
| family decline | +0.0040 ± 0.0003 | +0.0022 ± 0.0003 |
| floor decline | −0.0003 ± 0.0002 | +0.0009 ± 0.0002 |
| **net** | **0.0043 (≈12σ)** | 0.0013 (≈3.6σ) |
| **ICL fraction** | **0.16 → 0.23** | 0.009 → 0.022 |
| corr(net, oracle available) over depths | **+0.992** | +0.754 |
| rule probe (chance) | **0.039** (0.0156) | 0.0080 (0.0078) |

`d2_R128_nF4` fires its pre-registered kill (ICL fraction < 0.10 everywhere **and** rule
decodability at chance). `d2_R64_nF2` passes, and the load-bearing evidence is the **+0.992**
shape match — the realized trajectory has the oracle's *shape*, not merely a nonzero slope.

R64 is a partial escape from RHM_META_LEARNING's collapse and the first positive
ICL-pressure reading on RHM; R128 is that collapse reappearing on the **in-context** axis
rather than the weight-space one.

**The dissociation**: the graded design realizes **3× more ICL in absolute nats while
offering 3× less available signal**. *What explains it was read wrong on the first pass —
see §6.*

### Emergence and held-out

| | `d2_R64_nF2` | `d2_R128_nF4` |
|---|---|---|
| net decline at 6k / 12k | ~0.000 → **+0.0051, still climbing** | +0.0012 → +0.0021, **flat since 2k** |
| held-out transfer | **+0.00432 vs +0.0040 (~108%)** | +0.00172 vs +0.0022 (78%) |

R64 is **still climbing at 12k**, so 23%-of-ceiling is a **lower bound** — which is why the
36k retrain was authorised. R128 is flat, so its thin fraction is structural, not budget.

### `rule_retention` part 1 — the positive control fixed rules cannot have

[`../synonym_retention/`](../synonym_retention/README.md) proves that on fixed-rules RHM any
perturbation changing the future must break prefix-identity, so a closed constituent is
NTP-redundant and **their retention curve has no "content the model must keep" reference
arm**. Perturbing the `top` arm at the **differing** level lifts that: prefix-identity holds,
but the realisation is evidence about `r`, which governs every later sequence.

Exact next-token TV after perturbing one d2 constituent, by read distance `w`:

| w | crosses boundary | family TV | floor TV | floor frac cells exactly 0 |
|---|---|---|---|---|
| 16 | | 0.00783 | 0.00367 | 0.969 |
| **32** | | 0.00362 | **0.00000** | **1.000** |
| 61 | ✓ | 0.00082 | 0.00000 | 1.000 |
| 125 | ✓ | 0.00072 | 0.00000 | 1.000 |
| 157 | ✓ | 0.00060 | 0.00000 | 1.000 |

The floor reaches **exactly zero in 100% of cells** past w=32 — their impossibility confirmed
numerically — while the family persists **across three sequence boundaries**. Only ~8% of
random cells carry it (matching the 2/16 differing-feature rate), so part 2 selects
differing-feature cells **by construction**; conditional TV is ~0.010, ~10× the pooled mean.
A zero at w=187 is *not* an anomaly: that read position predicts the second element of a d1
pair, governed by the **shared** leaf table, so it is rule-independent by construction — an
internal consistency check that the signal sits exactly where the differing table matters.

### 5.1 `rule_retention` part 2 (`p2c`) is VOID — the label was unidentifiable

`p2c` ran to completion and must not be reported. It reads:

| w | family retention | floor retention |
|---|---|---|
| 0 | +0.027 | **+1.0000** |
| 8 | +0.004 | +0.686 |
| 32 | −0.019 | +0.348 |
| 125 | +0.014 | +0.133 |

That looks like "the family model retains nothing and the floor model retains everything",
which is backwards from part 1's structural claim. It is an **instrument artifact**, and the
cause is §10's own gotcha re-entering through the **probe target** after `make_family` had
canonicalised it out of the DGP.

`part2` decodes `y_rule = rc2[d][j_node]`, the **storage index** of the rule used at the
perturbed node. In `mode="pool"` each rule set assigns its differing features their `m`
tuples in a **random permutation order**, so that index is an arbitrary per-rule-set
relabelling of the observable. Measured over the 64 rule sets, `P(storage index | emitted
tuple)` is uniform — 0.125–0.359 across the 8 pool tuples, mean exactly 0.25 = chance. So
the family arm's **exact Bayes ceiling for that label is chance**, and its measured
accuracies (0.247–0.280) sit precisely on it. On the floor arm all `R` rule sets are
bit-identical, the index is a bijection with the emitted tuple, and the ceiling is 1.0. The
two arms were being asked questions with ceilings of ~0.25 and 1.00.

`part2` also drops `../synonym_retention/`'s **"pin both ends of the probe"** discipline —
it normalises `(acc − chance)/(1 − chance)`, with no Bayes ceiling — which is exactly the
guard that would have caught this.

**Fix: `part2b`** (same file, `part2` left untouched so `p2c` stays reproducible). Two
changes, both prescribed by `synonym_retention`:

1. the label is the **emitted tuple's index in the shared pool** (8 classes, chance 0.125) —
   rule-set independent, the actual "which realisation was used" bit, and the thing that is
   NTP-required at distance since it is what discriminates rule sets;
2. an **exact Bayes ceiling** per read distance. It is cheap and exact: sequences are i.i.d.
   given `r`, so a prefix reaching into later sequences informs the perturbed node **only**
   through the rule posterior —
   `P(tuple | x_<g) = Σ_r w_r(g) · P(tuple | x_seq0 prefix, r)` — with `w` from
   `mixture_profiles` and `P(tuple | ·, r)` read off the level-`d` junction-tree clique
   (parent feature × rule index) pushed through `r`'s own `(f, c) → tuple` map.

The smoke run self-confirms the diagnosis in-run: the old `storage_idx` label reads
**0.27–0.29 (chance) at every distance in the family arm** and reproduces `p2c`'s 1.0000 in
the floor arm, while the new tuple label is decodable in both and the ceilings now match
(**family 0.995, floor 1.000**). Direction on the smoke (n=1200, noisy): past the first
sequence boundary the **family** arm retains more than the floor — w=77 +0.220 vs +0.070,
w=125 +0.291 vs +0.111 — which is the predicted positive control. Full run is tag `p2d`.

**Gotcha, general.** A probe target that is an arbitrary index into a per-condition storage
order is not a measurement. Decode the **object**, not its slot. Any retention/ceiling
readout on a family substrate should assert `P(label | observable)` is non-uniform before it
is trusted — one line of DGP-side arithmetic, no model required.

### 5.2 `p2d` — the corrected curve. The positive control exists, and it grows with distance

Tag `p2d`, `part2b`, aligned `_at36000` checkpoints, `n_seq=6000`, ceiling on 256 windows.
Label = emitted-tuple index in the shared pool (8 classes, chance 0.125). Retention is
`(acc − chance)/(bayes − chance)`, pinned at both ends.

| `w` | crosses bdry | family acc | family Bayes | **family ret** | floor acc | floor Bayes | **floor ret** | **fam − flo** |
|---|---|---|---|---|---|---|---|---|
| 0 | | 0.971 | 0.985 | +0.984 | 1.000 | 1.000 | +1.000 | −0.016 |
| 4 | | 0.972 | 0.981 | +0.990 | 0.997 | 1.000 | +0.997 | −0.007 |
| 8 | | 0.949 | 0.981 | +0.963 | 0.869 | 1.000 | +0.851 | **+0.112** |
| 16 | | 0.687 | 0.982 | +0.656 | 0.656 | 1.000 | +0.606 | +0.050 |
| 24 | | 0.366 | 0.983 | +0.281 | 0.486 | 1.000 | +0.412 | −0.131 |
| 32 | | 0.714 | 0.983 | +0.687 | 0.539 | 1.000 | +0.474 | **+0.213** |
| 48 | | 0.439 | 0.983 | +0.366 | 0.268 | 1.000 | +0.164 | **+0.202** |
| 61 | ✓ | 0.692 | 0.984 | +0.661 | 0.547 | 1.000 | +0.482 | **+0.179** |
| 77 | ✓ | 0.468 | 0.984 | +0.399 | 0.234 | 1.000 | +0.125 | **+0.274** |
| 125 | ✓ | 0.482 | 0.987 | +0.414 | 0.254 | 1.000 | +0.148 | **+0.266** |

Shuffled-label guards 0.112–0.134 against chance 0.125 at every `w`, both arms.

**The diagnosis is confirmed inside the same run.** The old `p2c` storage-index label is
carried as a third column: in the family arm it reads **0.252–0.300 at every distance**
(chance 0.25), and in the floor arm it reproduces `p2c` exactly (1.000 → 0.311). Same
activations, same probe, same split — only the label differs.

**Three readings.**

1. **The positive control [`../synonym_retention/`](../synonym_retention/README.md) says
   cannot exist on fixed rules does exist here, and it widens with distance.** The two arms
   are indistinguishable while the read position is still near the constituent (`w ≤ 4`,
   both ≈ +0.99). From `w = 8` the family arm holds more, and the gap **grows**: +0.11 at
   `w=8`, +0.21 at `w=32`, and **+0.18 / +0.27 / +0.27 across the three sequence
   boundaries**, where the floor has decayed to +0.13–0.15 while the family still reads
   +0.40–0.41. That is content the family model must keep and the floor model need not.

2. **The gap is not the feature bit.** `featAcc` is common-mode — at `w=125` family 0.774 vs
   floor 0.777, identical — while `tupleAcc` is 0.482 vs 0.254. So the family arm's excess is
   exactly the **realisation** bit beyond the constituent's identity, which is the quantity
   that carries rule evidence. This corroborates §7's rule-probe result (family post-boundary
   excess +0.011 → +0.030, growing with depth) with an independent instrument and a **pinned
   ceiling**, which §7's shared-probe caveat lacked.

3. **Against `synonym_retention`'s λ=0 baseline, the floor arm decays more slowly than their
   m4 curve** (+1.000 / +0.997 / +0.851 / +0.606 at `w` = 0/4/8/16, against their +0.858 /
   +0.325 / +0.079 / +0.011). This is a **label difference, not a substrate difference**:
   their `ruleAcc` decodes the synonym bit *given* the feature, while the tuple label decodes
   feature-and-realisation jointly and therefore inherits the feature's much slower decay
   (their `featW4` +0.911 vs `ruleW4` +0.325). So the floor column is **not** a reproduction
   of their curve and must not be quoted as one. The load-bearing number is the
   **family − floor difference**, in which the feature term cancels.

**Caveats.** Strong position-parity oscillation (`w=24` +0.281 against `w=32` +0.687 in the
family arm) — `synonym_retention` flagged the same structure at λ=0 and pooled over it rather
than decomposing it; it is which hierarchy level the read position sits at, and it is why the
difference column rather than either level alone is the number to trust. Single seed. The
ceiling is computed on 256 of the 6000 windows. The family ceiling is flat-to-rising in `w`
(0.985 → 0.987) while accuracy falls, so this is discarding, not loss of access — the same
shape their cut established.

---

## 6. Standing corrections

Kept as corrections rather than silent edits: the mistaken reading and what killed it are
more useful than the conclusion alone.

### 6.1 Storage burden → inference dimensionality

**Claimed first (WRONG).** That the R64/R128 dissociation was *storage burden* — R128 must
hold 128 × 4 features × 4 tuples = 2048 tuple-assignments against R64's 64 × 2 × 4 = 512, so
it spends capacity covering tables instead of inferring. The capacity tax (+0.074 vs +0.026
nats over each arm's own floor) was offered as evidence.

**What killed it.** Held-out transfer. The R64 family model's loss on rule sets it has
**never seen** is **1.4657 against 1.4661 in-distribution** — indistinguishable — with
**~108%** decline transfer, while the *floor* model reads **1.877** on those same sequences,
confirming they are genuinely different DGPs. A model that solved the task by memorising 64
tables cannot do this.

**Replacement.** The limiting variable is the **dimensionality of what must be inferred**,
not the amount stored. R64_nF2 pins a partition of 8 tuples into 2 groups; R128_nF4 pins 16
into 4. The harder inference is done far worse *despite 3× more information supplied*. The
tax decomposes to match: of R128's +0.074, ~0.060 is unresolved mixture penalty (it captures
2% of a 0.061-nat gap) and ~0.013 residual; R64's +0.026 splits ~0.015 / ~0.011. The tax is
mostly **failure to infer**, not **cost to store** — the opposite attribution.

**Lineage.** [`meta_adapt`](../../../mjc/meta_adapt/README.md) Cut #4c's boundary condition
(*"VoI's lever should appear only for harder identification: a higher-dimensional task
parameter"*) reached from the other side — 4c found low-dim system-ID saturates so fast that
smarter *acquisition* is over-engineering; here raising identification dimensionality
**breaks in-context inference outright**. Also RHM_META_LEARNING's learnability diagnosis
with its confounds removed: that cut could not separate "no incentive" from "not learnable",
because weight-space meta supplied neither a within-episode incentive nor a within-episode
information channel. Here both are present and exactly quantified (`ln R` supplied, fraction
realized measured), and the answer is that **supply is not the binding constraint**.

**Caveat carried with the 108%.** Only **6** held-out rule sets exist — `d2_R64_nF2` uses 64
of the 70 partitions its design admits — and they share the training family's tuple pool. So
this is *within-pool* generalisation to novel partitions, not transfer to a fresh pool. The
stronger test needs a design with a larger rule space and is **untested**.

### 6.2 Gate 0's belief-depth number was read at the worst position

Gate 0 first reported per-level recovery far below the parent's reference lines (d1 0.49 vs
0.979) **even in the floor arm**, which *is* the parent substrate. That was a read-position
artifact, not a substrate cost — see §7.

---

## 7. Finding pair: the discard schedule factors into two axes

Keep these together; this is the shape it should reach the idea doc in (pending discussion).

**Axis 1 — distance at fixed relevance** ([`../synonym_retention/`](../synonym_retention/README.md)).
On fixed rules at λ=0, closed-constituent synonym identity decays +0.858 (w=0) → +0.079
(w=8) → chance (w≈16) against a *rising* exact Bayes ceiling; the synonym bit goes ~3× faster
than the feature identity. Relevance is pinned at zero throughout; only distance varies.

**Axis 2 — relevance at fixed distance** (this cut's read-position sweep, **floor arm** = the
parent substrate). At **matched distance-since-close of zero**:

| read position | still pending | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|
| 31 | root only | 0.560 | 0.341 | 0.242 | 0.238 |
| 47 | d5 | 0.711 | 0.743 | **0.880** | 0.742 |
| 59 | d1, d2 | **0.972** | 0.919 | 0.723 | 0.662 |
| 62 | d1 live | 0.910 | **0.937** | 0.893 | 0.611 |
| 63 | nothing (sequence ends) | 0.526 | 0.321 | **0.217** | 0.189 |

d3 reads **0.880 vs 0.217 at identical distance**, split only by whether an unresolved
ancestor still depends on it. At live positions the floor arm **recovers the parent's
reference lines exactly** (d1 0.91–0.97, d3 0.72–0.89 vs 0.979/0.836). So the schedule is
keyed to **predictive relevance, not elapsed distance**, and the two axes together factor it.

**The causal lever is phase visibility.** The parent trained on flat concatenation (phase
hidden — the model cannot know a boundary is coming) and is the control; this cut trains on
aligned windows (phase revealed) and discards on schedule. The 36k bundle runs both
protocols precisely so this contrast is measured rather than assumed.

### Retention: the schedule moves in the rule channel, not the parse channel

Family vs floor at matched positions across all seven boundaries (`d2_R64_nF2`):

- **Rule probe** (direct; floor pinned at chance 0.0156 as guard): family post-boundary
  excess **+0.0114 → +0.0304**, growing with context depth. Rule identity **is** carried
  across boundaries.
- **Parse probe** (conservative lower bound): family − floor **negative everywhere**
  (−0.031 to −0.044 at offset 1).

The family model carries a **compressed rule-sufficient statistic** across boundaries while
carrying **less** decodable parse detail than the floor model — a re-allocation, not an
addition (Petersen et al.'s replacement-not-insertion, now across regimes rather than across
positions). Caveat: one shared probe across positions, so the 4× pre→post drop (0.182 →
0.046 at boundary 7) may be partly geometric; the matched family−floor excess is the safe
number. R128's excess is ≈ 0, consistent with its Gate 0 null.

---

## 8. PRE-REGISTERED: Gate 1 — revision readouts

**Not run.** Predictions and kills are binding; do not soften them after seeing numbers.

**What this regime supplies that the parent could not.** The parent's Gate B matched
position × exact surprisal and compared *different* positions. Here **the same position with
the same exact surprisal is rule-revision-heavy early and rule-revision-zero late**, because
context depth is orthogonal to position by construction (windows are whole aligned
sequences). That gives a **within-position, within-surprisal, across-context-depth** contrast
— the cleanest form of the aleatoric-null test available, since the confound Gate B had to
stratify away is here removed by design.

**Object.** `M_rule = KL(q^rule_{t+1} ‖ q^rule_t)` from a probe-decoded rule posterior,
matched term-for-term by the oracle's exact `rule_rev` (`oracle_mixture.mixture_profiles`).
Report alongside the parent's `M` on the ancestor chain, since
`B_total = rule_rev + E_r[B^(r)]` exactly.

**Registered predictions.**
1. `M_rule` separates high- from zero-rule-revision positions at matched position and matched
   exact surprisal, above the 0.5 guards.
2. The separation **decays with context depth**, tracking the oracle's `rule_rev` profile.
3. The floor arm shows nothing at any depth.

**Kills.**
- `M_rule` ≤ guards at every depth → the model's state does not express rule revision; Gate 2
  has no belief-space object to forecast and **must not be run in belief space**.
- Separation present but **flat in depth** → it is reading rule **identity**, not rule
  **revision**. Report as identity decoding, not as a revision result. *(This distinction is
  the one most likely to be fudged; it is why prediction 2 is registered separately.)*
- The floor arm separates → the contrast is positional, not epistemic; the instrument is
  wrong, not the DGP.

**Read positions.** Live positions (59/61/62 style) **and** boundaries, reported separately —
§7 says these are different regimes and pooling them would blur it.

**Machinery to reuse, not rebuild.** The parent's `../gates_ab.py` holds the atom-aware
matching strata and the self-validating guards. Its documented traps apply verbatim:
exact BP surprisal piles mass on atoms (0, ln 2, ln 3, …) so quantile edges repeat and
`digitize` merges each atom with the continuous spread above it — strata must be atom-aware
and every block must report the matching variable's own self-matched AUC as a guard;
`pos * K + bin` silently aliases when the matching variable has more atoms than `K`;
matching on the model's `nll` alone is **not** enough (exact surprisal still scored 0.64–0.78
nll-matched).

**Stakes.** The language sibling
([`../../../a2a_forward/conditional_revision/`](../../../a2a_forward/conditional_revision/README.md))
found **every** belief readout null while a supervised decode read 0.99. RHM is currently the
only substrate where a belief-space revision readout has ever worked, so this is the **bridge
test** between the RHM-positive and the language-null.

### 8.1 Implementation record — two amendments to the pre-registration

Written **before** any model was run, from a 40-window CPU sizing pass on the exact oracle
alone. Code: [`gate1.py`](gate1.py). Both are visible changes, not silent ones.

**(a) The `flat-in-depth ⇒ identity` kill is weaker than it was written to be.** The
pre-registration reads the decay prediction off the *aggregate* `rule_rev` profile, which
does fall hard. But the measured decomposition says the fall is **base rate, not per-event
magnitude**:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| frac `rule_rev` > 0 | .451 | .368 | .267 | .154 | .100 | .046 | .025 | **.005** |
| mean `rule_rev` given > 0 | .050 | .040 | .051 | .037 | .031 | .028 | .050 | .005 |
| `H_rule` remaining (nats) | 3.505 | 2.192 | 1.228 | 0.671 | 0.356 | 0.166 | 0.069 | **0.017** |

A **within-depth** contrast between `rule_rev` high and `rule_rev ≡ 0` therefore has a
roughly **flat truth-magnitude** across depths 0–6. So a flat AUC curve is what a genuinely
revision-tracking readout should produce here, and the pre-registered kill would misfire.
The kill is still reported; the discriminator that actually separates identity from revision
is **added**, in two forms — the graded partial `R²(M_rule ~ rule_rev | nll_mix)` among
positive cells only (an identity decoder cannot grade on magnitude at fixed surprisal), and
the parent tracking appendix's before-state decomposition (`M_pointmass`, `negH_t`).
**Depth 7 becomes an internal negative control**: the oracle says the rule is identified
(0.017 of `ln R` = 4.159 nats left), so any separation there is instrument, not phenomenon.

**(b) A parse confound the pre-registration does not control.** `d2_R64_nF2` differs in
`rules[4]` on features `{10, 12}` of 16. `rule_rev > 0` is therefore correlated with "the
current level-4 ancestor is a differing feature" — a **parse** fact, decodable by either arm
with no rule inference at all:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| P(`rule_rev`>0 \| differing anc.) | .736 | .617 | .514 | .306 | .140 | .089 | .077 |
| P(`rule_rev`>0 \| not) | .395 | .322 | .226 | .129 | .091 | .038 | .016 |
| ratio | 1.9× | 1.9× | 2.3× | 2.4× | 1.5× | 2.3× | **4.7×** |

Fix: the **primary cell set is restricted to differing-ancestor cells**, inside which the
families are "this token discriminated between rule sets" vs "it did not" at a fixed parse
role — the same move the language sibling's `negate_live`/`negate_dead` makes against its
frame confound. The pre-registered pooled column is still reported alongside. Consequence:
inside the restricted set there are too few cells per absolute-position stratum, so the
primary position stratum is **`n_open(t)`** = how many levels' constituents are still
unclosed after `t`. That is the causal content of "position" here (it fixes which hierarchy
level the token completes) and it is the variable §7's read-position table is organised by.
Absolute position is kept as a stricter secondary column. `n_open ≥ 2` = *live*,
`n_open ≤ 1` = *boundary*, reported separately per §8 — note this is only **2 of 64**
positions on the boundary side (t = 31 and t = 63), so that arm is inherently low-powered.

**(c) Added from the language sibling: `h_before_dir` / `h_after_dir` / `h_swap_dir`** — a
supervised linear decode of the family label from the state itself. These are **diagnostics,
not competing primaries**. They exist so the two negatives stay distinguishable: that cut
found every belief readout at chance while the state decoded the same distinction at 0.99,
i.e. *instrument-fails-to-transfer* rather than *phenomenon-absent*. `h_before_dir` is also
the contrast-validity check — if the **prefix** state already predicts the family, every
state-based row in that block is reading the template.

**Gotcha the smoke run caught, worth not rediscovering.** Scored **raw**, the `h_swap_dir`
guard read **0.91**, not 0.50. Permuting the input vectors *inside* a stratum decouples
content from label within the stratum but leaves the **between-stratum** structure intact,
and here the stratum predicts the label. Every directional AUC must be scored under the
**same** stratification as the rest of the table; raw and matched are both reported.

### 8.2 Gate 1 RESULT — the primary kill fires, and the pre-registered column would have lied

Tag `g1`, aligned `_at36000`, 3200 eval windows (1.6M positions), probe on 6144 separate
windows, single seed. Oracle re-verified at scale: 4.079 nats/window of `ln R` = 4.159.

**Primary — cellset `diff` (parse confound removed), matched `n_open` × exact mixture
surprisal, family arm.** Bootstrap SD over windows in brackets.

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `M_rule` | .477 | **.519** | .504 | .504 | .487 | .499 | .485 | .484 |
| (boot SD) | .012 | .007 | .007 | .007 | .007 | .011 | .012 | .015 |
| `M_pointmass` | .499 | .499 | .497 | .484 | .485 | .497 | .488 | .478 |
| `negH_t` | .448 | .461 | .484 | .508 | .502 | .496 | .528 | .534 |
| *guard* `M_shuffled` | .502 | .500 | .501 | .500 | .499 | .490 | .507 | .506 |
| *guard* `nll_mix` self | .539 | .531 | .513 | .532 | .524 | .522 | .525 | .519 |

**`M_rule` never exceeds its own matching guard at any depth. The pre-registered primary
kill fires.** The before-state readouts are null too, so this is not the parent's
"`M` is mostly a before-state readout" case — nothing in belief space separates.

**The pre-registered pooled cellset would have been reported as a positive with the
predicted decay.** On cellset `all` the family arm reads **.552 / .553 / .547 / .535 / .530 /
.519 / .524 / .510** — above chance and monotonically decaying in depth, which is exactly
registered predictions 1 and 2. It is an artifact. The floor arm on the same cells reads
.492 → .476, and restricting to differing-ancestor cells removes the effect entirely. What
decays is the **differing-ancestor base rate** (frac `rule_rev`>0: .422 → .033), not
revision. Amendment (b) in §8.1 is what caught this; without it this cut would have shipped
a false positive whose *shape* matched the pre-registration.

**Graded test (the discriminator added in §8.1(a)).** Partial `R²(M_rule ~ rule_rev |
nll_mix)` on positive cells: family .0010–.0053 against a shuffled null of .0000–.0009 — but
the **floor** arm reads .0007–.0036 on the same cells, so it is not distinguishable from the
regime control. For scale, the parent's Gate A read **0.150** at d1 against a .006 null.
Meanwhile `R²(M_rule ~ nll_mix | rule_rev)` is .027–.050 (family) vs .004–.015 (floor): the
family model's `M_rule` **is** carrying surprisal, just not revision — the parent's
root-level signature exactly.

**Why the null: the instrument, quantified.** Two numbers settle it.

- The rule probe reads **0.018 → 0.102** across depths (chance .0156) against gate 0's exact
  Bayes ceiling of **0.614** — i.e. **11–17% of ceiling**. The parent's Gate B fired (0.690)
  where its probe was at **82%** of ceiling and read **0.496 — chance —** where its probe was
  at **18%**. Gate 1 sits exactly on the parent's own null end of its own probe-quality axis.
- Mean `M_rule` = **0.641 nats** against mean oracle `rule_rev` = **0.008 nats**. The
  readout's per-step jitter is **~80× the entire signal**. A KL differencing a 64-way
  posterior decoded at ~15% of ceiling cannot resolve a 0.03-nat event.

So the honest verdict is **instrument-fails-to-transfer, not phenomenon-absent** — and the
reason is the parent's own headline constraint, *belief depth*, reappearing on the rule axis.

**And it is NOT the language sibling's pattern.** That cut's signature was `h_before_dir`
0.55 vs `h_after_dir` 0.99 — the arriving token created a distinction the belief readouts
could not express. Here:

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `h_after_dir` family | .881 | .804 | .772 | .738 | .732 | .736 | .706 | .672 |
| `h_before_dir` family | **.909** | **.840** | **.811** | **.756** | **.774** | **.740** | **.721** | **.706** |
| `h_swap_dir` (guard) | .522 | .480 | .501 | .506 | .505 | .481 | .480 | .525 |
| `h_after_dir` **floor** | .818 | .702 | .722 | .703 | .652 | .655 | .657 | .550 |

`h_before_dir ≥ h_after_dir` at **every** depth, and the floor arm — which cannot do rule
inference at all — reproduces most of it. Per the sibling's own contrast-validity rule, that
means the state's separation is **prefix-determined parse role**, not the revision event. So
the state does not secretly contain a revision signal the belief readout missed.

**Consequence, honoured: Gate 2 is not run.** The pre-registration is explicit — *"`M_rule` ≤
guards at every depth → Gate 2 has no belief-space object to forecast and must not be run in
belief space."* It is additionally pre-empted on its own terms: Gate 2's primary asks whether
the FM residual **direction** carries the rule posterior, with `r_dir` vs `h_after_dir` vs
`h_before_dir` as the contribution test. With `h_before_dir` already ≥ `h_after_dir`, and
`r = h_after − f(prefix)`, that comparison is degenerate here before it is run.

**Limitations, stated.** (1) This probe is **weaker than gate 0's**: 0.068 at depth 4 against
gate 0's 0.118, because it is position-agnostic by design (one readout differenced across a
step, rather than two different probes). A per-depth or MLP probe is the obvious retry — but
gate 0's own 0.118 is only 19% of ceiling, still inside the parent's null zone, so it should
not change the verdict. (2) The `nll_mix` self-matched guard lands at **0.513–0.539**, looser
than the parent's 0.489–0.507, so nothing below ~0.54 is resolvable here at all. (3) The
boundary arm (`n_open ≤ 1`) is 2 of 64 positions and its readings swing .41–.79 in **both**
arms; it is not interpretable and should not be quoted. (4) Single seed.

## 9. PRE-REGISTERED: Gate 2 — the temporal FM

**Not run.** Blocked on Gate 1.

**The design constraint, inherited and load-bearing.** The language sibling's Gate D found the
temporal FM residual's **magnitude** is explained by output entropy at **R² = 0.901** and by
revision at **R² = 0.0001**, while the residual **direction** separates at 0.996. In this
regime the ICL decline **is** an output-entropy decline — mixture predictive entropy falls as
the rule posterior concentrates. Therefore:

> A raw "FM residual decays over context" readout is a **pure entropy artifact** here. It is
> pre-registered as the **negative control**, not the result.

**Primary readout.** Does the FM residual **direction** carry the rule posterior / rule
revision, at **matched output entropy and matched position**?

**Controls and guards.**
- **`r_dir` vs `h_after_dir` vs `h_before_dir`.** The sibling found `r_dir` (0.996) ≈
  `h_after_dir` (0.993), i.e. the FM **inherits** rather than produces. Gate 2 must establish
  whether the FM contributes anything over the raw state; if not, that is the honest finding
  and it is cheap to establish.
- **Permute the inputs, not the labels.** The sibling's label-shuffle guards read 0.52–0.83
  against a feature a swap-state control puts at 0.50 — a near-perfectly decodable feature
  gets recruited by the residual correlation a permutation leaves behind.
- **Live-cell restriction.** Only ~8% of constituents belong to a differing feature (part 1),
  so any perturbation-based readout must select differing-feature cells by construction or it
  dilutes ~12×.
- Floor arm as regime control (its residual direction must carry nothing rule-like); a
  protocol-matched **depth** FM as the axis control (the parent's idiom).

**Kills.**
- Directional separation ≈ guards → the FM residual carries no rule information; the "FM
  watches the model learn" claim dies on this substrate.
- Separation present but `r_dir` ≈ `h_after_dir` → the FM inherits; report as a property of
  the state, not of the forward model.
- Separation survives at matched entropy only in the **magnitude** readout → entropy artifact.

---

## 10. Gotchas worth not rediscovering

- **Rule sets differing only in the storage ORDER of a feature's m rules are
  distributionally identical.** The DGP draws rules uniformly, so order is not a degree of
  freedom it can express. Comparing raw arrays misses this and silently yields unidentifiable
  duplicates — which *mimics the target phenomenon*, since a capped posterior looks like
  gradual identification. `make_family` canonicalises and guards on the partition count
  `(nF·m)!/(m!)^nF`, which is only **70** for nF=2, m=4.
- **A design can exhaust its own rule space.** `d2_R64_nF2` uses 64 of those 70, leaving only
  6 held-out rule sets — so its transfer test is inherently small-sample.
- **An unmatched single-arm ICL reading before ~6k steps is an artifact.** The FLOOR arm —
  where nothing can be learned in context — shows a *spurious positive* depth decline of
  **+0.0048 at step 2k**, decaying to −0.0012 by 12k. The matched floor is what makes the
  emergence curve interpretable, not a nicety.
- **Read belief probes away from constituent boundaries** (§7). At a sequence boundary the
  aligned-window model has discarded essentially everything: d3 reads 0.880 at position 47 and
  0.217 at position 63, at identical distance-since-close.
- **Training feeds `x[:, :-1]`, so position G−1's embedding is never trained.** Any probe
  reading the last position of a full-length window reads an untrained positional embedding.
- **The parent oracle has no revision for a sequence's FIRST token** (its axis is "x_{t+1}
  arrives", giving T−1 columns). Inside a window that position is real and the joint identity
  fails there without it — hence `oracle_mixture.sequence_revision_all`, which asserts
  bit-identical agreement with the parent on the overlap.
- **Matching strata must be atom-aware**; see §8 and the parent's `../gates_ab.py`.
- **The matched depth-swap is worth its complexity**: floor-arm noise floor ~0.0002 nats
  against an available signal of 0.019–0.061, i.e. ~100× SNR. An unmatched loss-vs-position
  curve cannot resolve this effect.
- **The mixture oracle is O(R) BP** and dominates wall time at large R. Compute it on a
  subsample (96–192 windows is plenty for a smooth mean); the model-side evals are cheap by
  comparison. Training never needs the oracle at all.

## 11. Open questions

- Does the phase-hidden protocol restore cross-boundary retention (§7 predicts it should)?
  The 36k bundle answers this.
- Is R64's ICL still climbing past 36k? If so the substrate's ceiling is not yet located and
  Gates 1–2 run on a lower bound.
- Transfer to a **fresh tuple pool** (not just novel partitions) — needs a larger rule space.
- Whether the rule-probe pre→post boundary drop (0.182 → 0.046) is informational or geometric;
  needs per-position probes.
