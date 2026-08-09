# Files — conditional_revision

**Up**: [README.md](README.md) · **Parent**: [../FILES.md](../FILES.md)

## Code files

| file | purpose |
|---|---|
| `conditional_revision.py` | **Gate 0.** Trains the shared base (12k steps, plain NTP + co-trained depth FM) and persists it, then trains three FMs against the frozen base — `temporal` (`h6[≤t] → Δ_t`, primary), `temporal_direct` (parametrisation control), `depth_frozen` (protocol-matched depth control) — plus the co-trained `depth_cotrain` arm that must reproduce `endogenous_teacher`'s Gate 0. Reports `corr(res, nll)`, `R²`, orthogonal variance fraction and the per-level profile on flat windows and aligned sequences, with a two-sided kill. |
| `oracle.py` | **The exact BP oracle, and the instrument self-check.** Sum-product BP over the known parse tree with normalised messages and optional internal-node evidence (to clamp `z_D`); node and **junction-tree clique** marginals; the joint KL `B_joint`, the marginal sum `B_marg`, the ancestor-chain sum `B_chain`, the posterior entropy `H_post`, the irreducible term `H_irr`, exact `H(x_{t+1}|x_{≤t})` and exact surprisal. `self_check()` verifies `E[B_D] = E[H_tot − H_irr_D]`. Runnable directly (`python3 -m rhm.conditional_revision.oracle`) for a brute-force correctness test against full enumeration on tiny trees — **keep it; it is what caught the hypertree error.** No Modal, no GPU, no learned component. |
| `gates_ab.py` | **Gates A and B.** Loads the cached base and FMs (retraining neither), trains ancestor-masked and unmasked belief probes with temperature calibration, runs the oracle, then: Gate A (partial `R²(M ~ B \| nll)` and its rank/shuffled/all-node variants) and Gate B (synonym vs disambiguating AUC under four matchings — `nll`, exact surprisal, position, position × exact surprisal — each with a self-validating guard). Also reports the Bayes ceiling for exactly the probe's masked node/position pairs. |
| `probe_diag.py` | **Probe diagnostic.** Answers "is the probe weak or is the task hard" three ways: the repo's own single-node probe as an anchor on the checkpoint (must reproduce d1 0.979 / d3 0.836 / d6 0.088), the exact Bayes ceiling for every variant's task, and a hyperparameter sweep (lr, capacity, per-position standardisation, per-level heads). Also splits every node by its relation to the current position — **current (ancestor chain) / past (already resolved) / future (unseen)** — against the Bayes ceiling for each, which is the measurement behind the next-token-sufficient-statistic finding. |
| `tracking/tracking.py` | **The tracking analysis** — the instrument audit of Gates A and B, imported from Petersen et al. 1998 PNAS 95:853 (writeup: [README.md](README.md#appendix--the-tracking-analysis-an-instrument-audit-of-gates-a-and-b)). Replays `gates_ab.py`'s RNG consumption order so the `chain` probe is bit-identical and Gate A/B reproduce to 4.2e-05, then instruments every **state** rather than the difference: per-state probe/Bayes accuracy and entropy at `t` and `t+1`, for the node `M` reads *and* the node the ancestor chain leaves; the exact path decomposition `M − B = d_den + d_num`; single-state controls (`negH_t`, `M_pointmass`, `H_post_oracle`); and a size-matched random-exclusion null for every cell-exclusion variant. Requires the default-off `return_chain_marginals=True` kwarg added to `oracle.py` (prior callers and the self-test unaffected). |
| `__init__.py` | package marker |

## Children

| child | contents |
|---|---|
| [`local_loss/`](local_loss/README.md) | The same conditioning gap used as a **training** signal rather than a measurement (idea doc §8): the temporal target as an auxiliary loss, against `RHM_FM_REGULARIZER`'s depth version as a matched control. The raw term is ~90% gauge; the temporal target compresses more, not less; and on the self-knowledge axis it reproduced the depth signature rather than escaping it. Scope-limited to m2 and to the SK axis — the epistemic-content readouts were not computed. Per-file: [local_loss/FILES.md](local_loss/FILES.md) |
| [`sculpt_slip/`](sculpt_slip/README.md) | The conditioning gap on a **control** substrate (RHM sculpting + Stage 3d's slippery actuator), where the aleatoric label is exact and there is a behavioural readout. Step 1 positive (the slip is directionally identifiable, 0.70–0.75 matched vs 0.50 for the residual norm); Step 2 null (a precision operator recovers nothing its geometry control does not, against a prize of only ~0.03). Also records, with numbers, why a learning-progress estimator was **not** built. Per-file: [sculpt_slip/FILES.md](sculpt_slip/FILES.md) |

## Auxiliary docs

| file | contents |
|---|---|
| `SPEC.md` | The pre-registered design: the question, what is arithmetic vs measured vs argued, the four forced design choices, Gate 0/A/B/C definitions with their kill criteria, standing priors against, what each null would teach, when to stop running these gates, and the confound table (updated in place with the hypertree, atom-strata and matching corrections the run forced). |
| `README.md` | The writeup — Gate 0, the oracle and its self-check, the probe finding, Gate A, Gate B, scope, reproduction, gotchas. |
