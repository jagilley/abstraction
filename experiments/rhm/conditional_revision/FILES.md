# Files — conditional_revision

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `conditional_revision.py` | **Gate 0.** Trains the shared base (12k steps, plain NTP + co-trained depth FM) and persists it, then trains three FMs against the frozen base — `temporal` (`h6[≤t] → Δ_t`, primary), `temporal_direct` (parametrisation control), `depth_frozen` (protocol-matched depth control) — plus the co-trained `depth_cotrain` arm that must reproduce `endogenous_teacher`'s Gate 0. Reports `corr(res, nll)`, `R²`, orthogonal variance fraction and the per-level profile on flat windows and aligned sequences, with a two-sided kill. |
| `oracle.py` | **The exact BP oracle, and the instrument self-check.** Sum-product BP over the known parse tree with normalised messages and optional internal-node evidence (to clamp `z_D`); node and **junction-tree clique** marginals; the joint KL `B_joint`, the marginal sum `B_marg`, the ancestor-chain sum `B_chain`, the posterior entropy `H_post`, the irreducible term `H_irr`, exact `H(x_{t+1}|x_{≤t})` and exact surprisal. `self_check()` verifies `E[B_D] = E[H_tot − H_irr_D]`. Runnable directly (`python3 -m rhm.conditional_revision.oracle`) for a brute-force correctness test against full enumeration on tiny trees — **keep it; it is what caught the hypertree error.** No Modal, no GPU, no learned component. |
| `gates_ab.py` | **Gates A and B.** Loads the cached base and FMs (retraining neither), trains ancestor-masked and unmasked belief probes with temperature calibration, runs the oracle, then: Gate A (partial `R²(M ~ B \| nll)` and its rank/shuffled/all-node variants) and Gate B (synonym vs disambiguating AUC under four matchings — `nll`, exact surprisal, position, position × exact surprisal — each with a self-validating guard). Also reports the Bayes ceiling for exactly the probe's masked node/position pairs. |
| `probe_diag.py` | **Probe diagnostic.** Answers "is the probe weak or is the task hard" three ways: the repo's own single-node probe as an anchor on the checkpoint (must reproduce d1 0.979 / d3 0.836 / d6 0.088), the exact Bayes ceiling for every variant's task, and a hyperparameter sweep (lr, capacity, per-position standardisation, per-level heads). Also splits every node by its relation to the current position — **current (ancestor chain) / past (already resolved) / future (unseen)** — against the Bayes ceiling for each, which is the measurement behind the next-token-sufficient-statistic finding. |
| `__init__.py` | package marker |

## Auxiliary docs

| file | contents |
|---|---|
| `SPEC.md` | The pre-registered design: the question, what is arithmetic vs measured vs argued, the four forced design choices, Gate 0/A/B/C definitions with their kill criteria, standing priors against, what each null would teach, when to stop running these gates, and the confound table (updated in place with the hypertree, atom-strata and matching corrections the run forced). |
| `README.md` | The writeup — Gate 0, the oracle and its self-check, the probe finding, Gate A, Gate B, scope, reproduction, gotchas. |
