# RHM — File & Auxiliary-README Index

Full file-by-file reference for the experiment. Summarized in [README.md](README.md#code--files); this is the complete listing. Auxiliary-README descriptions here duplicate the inline `Full writeup` links in the main README's per-experiment sections — kept together for navigation.

## Code files

| File | Purpose |
|---|---|
| `shared.py` | Modal infrastructure (app, volume, image, utilities) |
| `stages.py` | Reusable experiment primitives: `generate_corpus`, `train_model`, `sweep`, `measure_scaling` |
| `rhm_data.py` | RHM data generation (hierarchy rules + corpus sampling) |
| `model.py` | GPT-2 with `return_intermediates` and cerebellar callback support |
| `measure.py` | Scaling exponent fitting (log-log regression) |
| `hparam_sweep.py` | L x m scaling exponent sweep |
| `rhm_residual_rank.py` | FM residual rank experiments: DGP sweep, FM capacity sweep, architecture-matched sweep |
| `rhm_regime_transition.py` | Regime transition: hierarchy-conditioned eta^2 at checkpoints over training |
| `rhm_cosine_sweep.py` | Cosine sweep: finding FM capacity settings where the FM genuinely struggles |
| `rhm_regime_trajectory.py` | Regime trajectory: the L-to-m transition at scale (2.7M model, m=4) |
| `rhm_per_level_loss.py` | Per-level loss decomposition: cross-entropy by hierarchy level over training |
| `rhm_dgp_approximation.py` | FM as DGP approximation: does the FM learn the RHM composition rules? |
| `rhm_fm_intermediate_probing.py` | FM intermediate probing: does the FM's internal computation mirror the hierarchy? |
| `rhm_label_smoothing.py` | Label smoothing sweep: does softening NTP shift learning from m to L? |
| `rhm_focal_loss.py` | Focal loss sweep: does confidence-based position weighting shift learning from m to L? |
| `rhm_confidence_threshold.py` | Confidence threshold sweep: aggressive gradient reallocation + scaled model |
| `rhm_fm_weighted_ntp.py` | FM-surprise-weighted NTP: principled gradient reallocation via self-model surprise |
| `rhm_ratchet.py` | Unified gate ratchet with confidence thresholding: meta-learning signature vs m-sharpening |
| `rhm_sparse_ratchet.py` | Sparse ratchet: WS_UG_uniform with NTP masking, gated vs fixed local loss |
| `rhm_rl_ratchet.py` | RL ratchet: REINFORCE with FM supervision for generation + FM cosine regime sweep |
| `rhm_rl_gen_distill.py` | Generation-based distillation: KL on teacher-generated suffixes + CE on NTP |
| `rhm_rl_gen_distill_extended.py` | Extended gen-distill: 40-cycle ratchet + weight decay experiments |
| `rhm_fomaml_ratchet.py` | FOMAML ratchet: bilevel meta-learning with dense NTP wake + NTP outer |
| `rhm_fomaml_rl_ratchet.py` | FOMAML RL ratchet: bilevel meta-learning with RL wake + per-level NTP outer |
| `rhm_meta_learning.py` | Meta-learning: FOMAML with rule-set transfer for compositional learning |
| `rhm_meta_learning_l2.py` | Meta-learning: L2+-only outer loss FOMAML with rule-set transfer |
| `rhm_reptile.py` | Meta-learning: Reptile with rule-set transfer (dense inner loop) |
| `rhm_reptile_sparse.py` | Meta-learning: Reptile with sparse L2+ inner loop + rule-set transfer |
| `rhm_fm_regularizer.py` | FM-as-regularizer: structured FM-predictability pressure vs weight decay (functional-complexity floor) |
| `rhm_latent_loop.py` | Latent-loop 2×2: target (token vs oracle-latent) × loop (open vs closed); λ_local sweep; fresh-FM ensemble |
| `rhm_ensemble_trajectory.py` | Fresh-FM-ensemble residual invariance over saved training checkpoints (complexodynamics: transient scaffolding vs DGP floor) |
| `rhm_active_query.py` | Active-RHM laboratory: arity-2 query-conditioned belief FM vs arity-1 capacity sweep and frozen external planner |
| `rhm_active_planning.py` | Active-RHM diagnosis: belief-Δ FM fidelity sweep + posterior-target FM + m sweep (shows fidelity isn't the planning bottleneck) |
| `rhm_active_voi.py` | Active-RHM fix: value-of-information (expected-posterior-entropy) head + greedy-EIG planner vs belief-Δ/random/oracle |
| `README.md` | This file |

## Auxiliary READMEs

| File | Purpose |
|---|---|
| `SWEEP_README.md` | [Scaling exponent sweep](SWEEP_README.md) |
| `RESIDUAL_RANK_README.md` | [FM residual rank experiments](RESIDUAL_RANK_README.md) |
| `REGIME_TRANSITION_README.md` | [Regime transition, cosine sweep, and trajectory](REGIME_TRANSITION_README.md) |
| `PER_LEVEL_LOSS_README.md` | [Per-level loss decomposition](PER_LEVEL_LOSS_README.md) |
| `LABEL_SMOOTHING_README.md` | [Label smoothing experiment](LABEL_SMOOTHING_README.md) |
| `LOSS_WEIGHTING_README.md` | [Loss weighting experiments: focal loss, confidence threshold, FM-surprise NTP](LOSS_WEIGHTING_README.md) |
| `RHM_RATCHET_README.md` | [Unified gate ratchet with confidence thresholding](RHM_RATCHET_README.md) |
| `RHM_SPARSE_RATCHET_README.md` | [Sparse ratchet: wake-sleep with NTP masking](RHM_SPARSE_RATCHET_README.md) |
| `RHM_L4_RATCHET_README.md` | [Sparse ratchet at L=4: overparameterization test](RHM_L4_RATCHET_README.md) |
| `RHM_RL_RATCHET_README.md` | [RL ratchet: REINFORCE with FM supervision](RHM_RL_RATCHET_README.md) |
| `RHM_RL_GEN_DISTILL_EXTENDED_README.md` | [Gen-distill extended: 40-cycle ratchet + weight decay](RHM_RL_GEN_DISTILL_EXTENDED_README.md) |
| `RHM_FOMAML_README.md` | [FOMAML ratchet: bilevel meta-learning diagnostic (3 outer objectives)](RHM_FOMAML_README.md) |
| `RHM_META_LEARNING_README.md` | [Meta-learning with rule-set transfer: FOMAML, Reptile, sparse L2+ inner](RHM_META_LEARNING_README.md) |
| `RHM_FRONTIER_AND_LEGIBILITY_README.md` | [m gates the learnable frontier; FM residual tracks it; WD sweep (norm vs rank)](RHM_FRONTIER_AND_LEGIBILITY_README.md) |
| `RHM_FM_REGULARIZER_README.md` | [FM-as-regularizer beats weight decay's functional-complexity floor](RHM_FM_REGULARIZER_README.md) |
| `RHM_LATENT_LOOP_README.md` | [Latent target is load-bearing for generalizable self-knowledge on RHM](RHM_LATENT_LOOP_README.md) |
| `RHM_COMPLEXODYNAMICS_README.md` | [Complexodynamics: rise-then-fall of sophistication proxies; transient = FM-idiosyncratic scaffolding, floor = DGP-aligned](RHM_COMPLEXODYNAMICS_README.md) |
| `ACTIVE_RHM_README.md` | [Active RHM: mean-Δ FM can't plan epistemic queries (structural null); a value-of-information head can (m=2 positive, m=4 principled null); the controllability boundary is measurable](ACTIVE_RHM_README.md) |
