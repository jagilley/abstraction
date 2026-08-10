# Files — rule_family

**Parent**: [../FILES.md](../FILES.md) · **Parent README**: [../README.md](../README.md)
**Idea**: [`ideas/revision_not_surprisal.md`](../../../../ideas/revision_not_surprisal.md)

No README yet — results are discussed with Jasper before a writeup, per repo convention.
Working notes, run status, the numbers so far and the **pre-registered** Gate 1 / Gate 2
designs (predictions and kills, written before any run) live in [NOTES.md](NOTES.md).

## What this sub-experiment is

The parent cut measures belief revision on **one fixed rule set**, where reducibility is
static: which positions are synonym-slots and which are disambiguators is fixed by the
DGP, so an NTP-optimal model can bake the split into weights and never compute it. Here a
context window is drawn from one of **R rule sets** (same `v/s/L/m`, different composition
tables), so the model must infer the active rules in-context and reducibility becomes
**state-dependent within the context**: the same position is revision-heavy early (the
rule is unknown, so the token is evidence) and synonym-noise late (the rule is known).

With finite R the exact oracle survives as a **mixture over R junction trees**, and total
surprisal decomposes exactly into parse-irreducible + parse-revision (fast) + rule-revision
(slow, decaying as the rule posterior concentrates).

**The hard constraint this design runs into, and the reason Gate −1 exists**: summed over
a window, `E[rule revision] = I(r ; x_window) <= ln R`. The entire slow component is
capped at `ln R` nats however the family is built, so strength and duration trade off
against each other along a fixed budget. Gate −1 measures that trade-off exactly, on CPU,
before any GPU time.

## Code files

| file | purpose |
|---|---|
| `family.py` | **The DGP side.** `make_family` builds R rule sets sharing every table except `differ_levels`, with `n_differ_features` controlling how many features at those levels move; `differ_levels=[]` is the **zero-conflict floor** (all R rule sets bit-identical to `generate_rules_distinct`, i.e. the incumbent substrate reached through identical code — the analogue of meta_adapt's damping floor). Two modes: `pool` (rule sets share one tuple pool and differ only in how it is *partitioned* among features, so the tuple multiset at that level is matched) and `resample` (fresh tuples per rule set — the deliberately leaky control, identifiable by elimination). `generate_windows` builds contexts as K whole aligned sequences from one rule set, which keeps position-within-sequence (hierarchy level) orthogonal to sequence-index (context depth). The shortcut-audit half — `node_marginals_prior`, `block_distributions` (exact aligned s^k-block distributions), `shortcut_posterior` (an explicit n-gram-only learner) and `pairwise_block_divergence` — measures how much rule information survives at each n-gram order instead of assuming the design closed it. Self-test validates the exact block distributions against 20k samples. |
| `oracle_mixture.py` | **The exact mixture oracle.** Imports `../oracle.py`'s BP primitives rather than editing them, so single-rule results stay bit-identical by construction. `mixture_profiles` gives the rule posterior, rule revision `KL(w(n+1)‖w(n))`, mixture surprisal, and the ICL gap; `sequence_revision_all` extends the incumbent's `revision_and_entropy` to a sequence's **first** token (which the incumbent never computes and which is a real position inside a window) and asserts bit-identical agreement on the overlap; `parse_revision_mixture` gives the fast term `E_{r~w}[B^{(r)}]`. `self_check_mixture` verifies both identities: `E[rule_rev] = E[H_mix − H_true]` and `E[B_total] = E[H_mix − H_irr]`. Runnable directly for a **brute-force enumeration test over (r, latents)** on tiny trees — the mixture analogue of the test that caught the hypertree error; keep it. |
| `gate_minus1.py` | **Gate −1**, CPU, no training, no Modal. Sweeps candidate designs and reports, per design: the `ln R` budget and how much of it is used, the positions where `P(r*|prefix)` crosses 0.5/0.9/0.99, where 50%/90% of the rule information has arrived, the per-token ICL gap early vs late, the posterior-independent per-position **evidence rate** by hierarchy level, and the shortcut audit. `--full` adds the parse term, the identity self-check, and the **de-nulling readout**: the fraction of positions whose total revision is exactly zero, by context depth. |
| `gate0_family.py` | **Gate 0** (Modal, L4). Trains the base on the family corpus plus a compute/data/config-matched **floor** arm, then measures: within-context loss by sequence index against the oracle's `nll_mix`/`nll_true` (so ICL is reported as a fraction of the available gap, not raw nats), rule decodability from activations by context depth against the exact Bayes accuracy, and per-level ancestor recovery against the substrate's reference lines (d1 0.979 / d3 0.836 / root 0.088). Pre-registered kills are in the module docstring. |
| `probe_positions.py` | **Where in a sequence the belief probe actually reads.** Sweeps the read position across a sequence and reports per-level recovery at each, family vs floor. Separates "the family regime costs belief depth" from "read at a constituent boundary" — the floor arm recovers the incumbent reference lines (d1 0.91–0.97, d3 0.72–0.89) at live positions and collapses to d1 0.53 / d3 0.22 at the sequence boundary. Forward passes only on cached checkpoints. |
| `retention.py` | **Is the discard schedule a dependent variable?** Under one fixed rule set a completed sequence is conditionally independent of the future, so discarding at a boundary is optimal; under a rule family a resolved parse is evidence about `r`, so it is not. Measures family-vs-floor retention across every boundary with two instruments of different strength: the **rule probe** (direct — what is worth keeping is a compressed rule-sufficient statistic, not node identities; floor arm pinned at chance as guard) and the **past-sequence parse probe** (conservative lower bound, with a live-node probe at the same positions to normalise probe quality there). |
| `rule_retention.py` | **The port of [`../synonym_retention/`](../synonym_retention/README.md)'s instrument, supplying the positive control arm that cut proves cannot exist on fixed rules.** Their impossibility section: on fixed-rules RHM any perturbation that changes the future must break prefix-identity, so a closed constituent is NTP-redundant and their retention curve has no "content the model must keep" reference. Perturbing the `top` arm at the **differing** level (d2 — the leaf table is shared by design and carries no rule information) leaves prefix-identity intact while making the realisation evidence about `r`, which governs every later sequence. Part 1 (local, CPU, no model) measures the exact next-token TV vs read distance, family vs floor, across sequence boundaries — the structural claim. Part 2 (Modal) is the model-side retention curve, chance/Bayes-normalised against their λ=0 baseline. |
| `__init__.py` | package marker |

## Conventions worth not rediscovering

- **`rules[d]` vs `d{k}` naming.** `rules[d]` expands a level-d feature (d = 0 root,
  d = L−1 emits leaves); a level-D latent is the repo's `d{L−D}`. So `dl=[4]` means "how a
  **d2** feature expands" and `dl=[5]` is the leaf-emitting table.
- **Share the leaf-emitting table.** RHM_META_LEARNING's diagnosis is that dense NTP is
  L0-dominated, so a family identifiable from bigram counts teaches nothing about
  composition. `dl=[5]` is kept as the *leaky control* and identifies ~3× faster than
  `dl=[4]`, which is the measurement that justifies the constraint.
- **The incumbent oracle has no revision for a sequence's first token.** Its signal axis is
  "the token x_{t+1} arrives", so it returns T−1 columns. Inside a window that position is
  real, and the joint identity fails there unless it is computed — hence
  `sequence_revision_all`.
