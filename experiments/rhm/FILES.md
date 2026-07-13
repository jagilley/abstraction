# RHM — File & Auxiliary-README Index

Full file-by-file reference for the experiment. Summarized in [README.md](README.md#code--files); this is the complete listing. Auxiliary-README descriptions here duplicate the inline `Full writeup` links in the main README's per-experiment sections — kept together for navigation.

## Code files

| File | Purpose |
|---|---|
| `shared.py` | Modal infrastructure (app, volume, image, utilities) |
| `stages.py` | Reusable experiment primitives: `generate_corpus`, `train_model`, `sweep`, `measure_scaling` |
| `rhm_data.py` | RHM data generation (hierarchy rules + corpus sampling); `generate_rules_invertible` (collision-free rules → exact parse) + `build_inverse_maps` / `parse_leaves` (batched ground-truth bottom-up parser) |
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
| `rhm_active_internal.py` | Active-RHM Phase 4 Step 0 (ceiling probe): belief-`b` VoI head vs raw-observation VoI head (ceiling) — decomposes the m=4 null into belief-carried / observation-decodable / fundamentally-content-carried |
| `rhm_active_headroom.py` | Active-RHM Phase 4 Step 0.5 (headroom probe): root-predictor vs model-free-policy controller with VoI target+ceiling held fixed — shows VoI-sufficiency is objective-independent (no internalization headroom; act≈plan) |
| `rhm_edit_control.py` | RHM-as-control Part 1: editing (block-edit) arity battery + reveal contrast on one controller + ground-truth judge — editing is plannable in belief space (mirror of the query null) but the belief is gameable off-manifold |
| `rhm_generative_planner.py` | RHM-as-control Part 2: generator-defined (soft on-manifold) moves + cerebellar self-consistency veto — converts off-manifold belief-gaming (gt≈0) into majority true control (gt≈0.65) with learned models only |
| `rhm_latent_planner.py` | RHM-as-control Part 3 / Stage 2: faithful MC value + latent cerebellar FM; instructive dead-end — a faithful verifier is too sparse to plan, a learned MC value is dense, but lookahead collapses because the corrupt-repair task is greedy-decomposable |
| `rhm_sculpt_precheck.py` | Sculpting pre-check (perfect simulator, no learning): DP optimum vs myopic reflex proves the editing task has a depth-scaling lookahead prize at tight coupling (m=2/3), collapsing at m=4 |
| `rhm_sculpt_planner.py` | Sculpting Stage 3a (learned): generator moves + MC value + beam planner; beam width 1 = reflex, wider = coordination — captures the lookahead prize (0.29→0.58, matches strong reflex) where greedy collapses |
| `rhm_sculpt_latent.py` | Sculpting Stage 3b: latent-space beam (cerebellar FM over a richer per-block latent, re-grounded each step) vs the token-space beam — planning in latents reaches 92% of token-space at ~8× fewer materializations (efficient surrogate, not superior — *on the clean channel*) |
| `rhm_sculpt_latent_po.py` | Sculpting Stage 3c: latent vs token beam under PARTIAL OBSERVABILITY (flickering block sensor, occlusion sweep `p`; latent carries a Kalman-filtered belief). Latent BEATS token once p≥0.25 — the token channel stops being a sufficient statistic. Reuses Stage-3b instruments; asserts p=0 anchor |
| `rhm_sculpt_latent_stoch.py` | Sculpting Stage 3d: latent vs token beam under STOCHASTIC DYNAMICS (slippery actuator, slip sweep `q`; token samples once, latent ranks by stable FM). Latent BEATS token once q≥0.1, peak +0.075 — the mirror of the active-query null (payoff in the mean the token beam must sample). Reuses Stage-3b instruments; asserts q=0 anchor |
| `rhm_sculpt_deepbelief.py` | Sculpting Stage 4 gate (belief-quality axis): runs belief conditions (parser floor / oracle privileged ceiling / data2vec + mlm non-privileged) through the identical frozen value+FM+beam pipeline, single controlled variable = the controller's aux objective. Owns `_train_deep_controller` (oracle ancestor aux), `_train_mlm_controller` (masked-infilling; `mask_mode` ∈ subtree/span/scatter — span is the fully-agnostic non-privileged control), `_train_data2vec_controller` (EMA self-distillation), `_belief_depth_probe`, `_subtree_mask`/`_span_mask`. Finding: deeper belief → monotone-better FM ranking; the non-privileged mlm belief is the best latent-planner substrate, beating even the oracle; span-masking ablation confirms the depth recruitment survives with zero DGP-topology knowledge. Output dir tagged by belief set |
| `rhm_sculpt_data2vec.py` | Sculpting Stage 4 isolation: validates the non-privileged belief BEFORE planning — per-level linear recovery of block ancestors vs ground truth + anti-collapse (participation ratio). No beams (fast). Finding: data2vec barely recruits depth (teacher-capped), mlm recruits ~40% of the oracle depth gap with no collapse. Imports the trainers from `rhm_sculpt_deepbelief.py` |
| `rhm_sculpt_internalize.py` | Sculpting Internalization (the REACHING_INTERNAL port): does closing the planning loop reshape the *belief* itself? Nested 3-rung ladder on the parser belief, single controlled variable = whether/how-much the loop touches the belief during training — `frozen` (root-CE only, = Stage-3b parser) ⊂ `fm_cotrain` (+ a2a asymmetric FM local-loss into the belief: endogenous "be predictable") ⊂ `planner` (+ differentiable grounded planner CE `value(z+FM(z,k))` imitating the DP best-move `k*`: int_plan port). Identical frozen-downstream eval per rung (depth-probe + PR, fresh MC value, fresh block FM, `_fm_check`, token+latent beams) so the fresh-FM latent beam == the fresh-external-planner transferability test. Two headlines: (A) belief becomes plannable (FM top1/rank-corr↑, clean-channel gap↑), (B) frontier-moving vs capped (belief-depth d3↑ under grounded rung, capped under endogenous rung — the RHM_LATENT_LOOP mlm-vs-data2vec unification, now on the planning side). Owns `_collect_grounded_moves` (controller-independent DP best-move buffer), `_train_fm_cotrain`, `_train_planner_internal`, `_downstream_eval` |
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
| `ACTIVE_RHM_README.md` | [Active RHM: mean-Δ FM can't plan epistemic queries (structural null); a value-of-information head can (m=2 positive, m=4 principled null); the controllability boundary is measurable; and (Phase 4) internalizing the forward model has no headroom because active-query is inference-in-disguise (act≈plan)](ACTIVE_RHM_README.md) |
| `RHM_EDIT_CONTROL_README.md` | **WIP** [RHM as a control task: editing is plannable in belief space (arity usability ports, reversing the query null) but the belief is gameable off-manifold — belief faithfulness is a second controllability axis beyond act≠plan; generator-defined on-manifold moves + a cerebellar self-consistency veto convert gt≈0 gaming into gt≈0.65 true control (learned models only). Part 3 (planning in latents) pending](RHM_EDIT_CONTROL_README.md) |
| `RHM_SCULPTING_README.md` | **WIP** [Sculpting & planning in latents: a faithful verifier is too sparse to plan / a learned MC value is dense but the corrupt-repair task is greedy-decomposable (Stage-2 dead-end); the sculpting task genuinely requires coordination (DP optimum vs myopic reflex, depth-scaling prize); a learned beam captures it (0.29→0.58→0.61, passes strong reflex) where greedy collapses (Stage 3a); a re-grounded latent-space beam reaches 92% of token-space at ~8× lower cost — an efficient surrogate, not superior, *on the clean channel* (Stage 3b); but latent planning genuinely *beats* token-space once the token channel is lossy — partial observability (Stage 3c) and stochastic dynamics (Stage 3d) — because tokens stop being a sufficient statistic; the flip is width-gated by the FM's 0.36 top-1 pick](RHM_SCULPTING_README.md) |
| `sculpting_control_task.md` | [The sculpting perfect-simulator pre-check (sub-note of RHM_SCULPTING): exact DP optimum vs myopic reflexes proves the editing task's lookahead prize scales with depth and tightness, collapses at m=4](sculpting_control_task.md) |
