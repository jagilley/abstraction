# RHM Ratchet / Meta-Learning Arc — Wake-Sleep Self-Model Compounding on RHM

**Idea doc**: [ideas/language_reduction.md](../../../ideas/language_reduction.md)
**Parent experiment**: [../README.md](../README.md) (the RHM scaling / residual-structure program)
**MNIST/language counterpart**: [../../a2a_forward/GATED_RATCHET_README.md](../../a2a_forward/GATED_RATCHET_README.md) (where the same ratchet *does* compound)

> **Status: concluded (negative), retained for reproducibility.** This directory collects the wake-sleep / RL / FOMAML / meta-learning "ratchet" line on RHM. Each experiment's full writeup lives here; per-experiment reproduction commands are inside each writeup (entrypoints are now `modal run -m rhm.ratchet.<module>::<fn>`). File-by-file purposes for the whole RHM experiment remain in the parent [../FILES.md](../FILES.md).

## The question, and the answer

The A2A gated ratchet compounds on MNIST — closing the wake-sleep loop (inject the FM forecast → distill → reinit the FM) widens the val-loss gap over open-loop across cycles (34% → 48% over 4 cycles). This arc asked whether the same ratchet compounds on the controllable RHM DGP, where we know ground truth and can decompose learning by hierarchy level.

**It does not.** The ratchet reproduces every MNIST *dynamic* on RHM — gate closing, FM tracking, robustness dissociation, crossover timing — but never the *magnitude*: the val-loss gap peaks at ~1% and does not grow across cycles. The arc ruled out, in turn: capacity / overparameterization (L=4), task structure (RL is sparse-but-rich like classification), the distillation bottleneck (generation-based distillation restores NTP), and FM-cosine regime (the 3-block gap reaches the 0.944 sweet spot). The FOMAML diagnostic settled the mechanism: **on stationary data no outer objective makes FM-predictability pressure improve compositional NTP** — "the outer loop is identical to the inner loop." The consistent read across the arc is that **compounding requires a moving frontier** (novel rules / deeper levels) that stationary RHM cannot supply — which is where the parent program's active / sculpting / latent-loop lines went next.

[**Jasper's note**: this is all true but I think the critical thing is that RHM has a moving frontier up to and until it hits its ceiling, and no further. Then, the frontier goes static and bilevel optimization becomes pointless. (I think.) On MNIST, I believe we observe different properties because MNIST always supervises the root and is non-hierarchical.]

## Experiments (chronological)

| Date | Writeup | Headline |
|---|---|---|
| 06-23 | [RHM_RATCHET_README.md](RHM_RATCHET_README.md) | Unified-gate ratchet + confidence thresholding: replicates MNIST dynamics, +0.3% magnitude (m=4); FM capacity must be matched (~2%) or the gate-closing behavior is masked |
| 06-24 | [RHM_RATCHET_M2_README.md](RHM_RATCHET_M2_README.md) | Same recipe at m=2 (genuine compositional headroom): still no per-level depth improvement; OL wins at every cycle |
| 06-24 | [RHM_SPARSITY_SWEEP_README.md](RHM_SPARSITY_SWEEP_README.md) | Supervision density as the causal variable: monotonic crossover from local-loss-hurts (dense NTP) to local-loss-helps (+1.6% at 95% NTP masking) |
| 06-24 | [RHM_SPARSE_RATCHET_README.md](RHM_SPARSE_RATCHET_README.md) | Wake-sleep with NTP masking: +1.2% and L1–L2 compositional gains in cycle 1, but non-compounding (exhausts after one cycle) |
| 06-24 | [RHM_L4_RATCHET_README.md](RHM_L4_RATCHET_README.md) | Overparameterization test (L=4): the regularization-headroom hypothesis is unsupported; self-knowledge probes even invert (tiny architectural-mismatch residual) |
| 06-25 | [RHM_RL_RATCHET_README.md](RHM_RL_RATCHET_README.md) | REINFORCE + FM supervision: +40% generation at all hierarchy levels; the ratchet still doesn't compound (distillation is load-bearing yet destructive); generation-based distillation (run 13) fixes the NTP-destruction problem |
| 06-26 | [RHM_RL_GEN_DISTILL_EXTENDED_README.md](RHM_RL_GEN_DISTILL_EXTENDED_README.md) | 40-cycle gen-distill + weight-decay: outcome metrics stay flat, but L3 feature η² creeps +15% across cycles (representational deepening without a behavioral crossover) |
| 06-26 | [RHM_FOMAML_README.md](RHM_FOMAML_README.md) | Bilevel meta-learning diagnostic (3 outer objectives): FOMAML doesn't compound under any of them — the gap is domain-specific, not a first-order-method artifact |
| 06-27–29 | [RHM_META_LEARNING_README.md](RHM_META_LEARNING_README.md) | Rule-set-transfer FOMAML / Reptile / sparse-L2+ (five meta-learners): all collapse to plain multitask on the stationary objective |

## Reproduction

All entrypoints moved from `rhm.<module>` to `rhm.ratchet.<module>` (run from `experiments/`):

```bash
modal run --detach -m rhm.ratchet.rhm_ratchet::rhm_ratchet_sweep
modal run --detach -m rhm.ratchet.rhm_sparse_ratchet::rhm_sparse_ratchet --depth 4 --mask-rates "0.0,0.75,0.90,0.95"
modal run --detach -m rhm.ratchet.rhm_rl_ratchet::rhm_rl_ratchet --only-rl-fm --ntp-mask-rate 0.95
modal run --detach -m rhm.ratchet.rhm_rl_gen_distill::rhm_rl_gen_distill
modal run --detach -m rhm.ratchet.rhm_fomaml_rl_ratchet::rhm_fomaml_rl_ratchet
```

Each writeup carries its own exact command. (Note: `RHM_FOMAML_README` also references `rhm.ratchet.rhm_fomaml_ratchet` for the dense-NTP-outer arm; that script is not currently in the tree — only the RL-outer arm `rhm_fomaml_rl_ratchet.py` is. This is a pre-existing gap, not introduced by the move.)

## Open threads (rehomed from the parent README next-steps)

These were the arc's live next-steps before it was wound down. Kept here for whoever revisits it; none were pursued.

1. **Understand the MNIST–RHM ratchet gap.** The L=4 experiment ruled out capacity/overparameterization. The RL ratchet ruled out task structure (RL is sparse-but-rich like classification) and partially ruled out FM cosine regime (the 3-block gap achieves 0.944, but training is unstable). Gen-based distillation (run 13) resolved the distillation bottleneck (val loss 0.813 vs 2.262, generation 39.1%) and achieved sweet-spot FM cosine (0.911) — but the ratchet still doesn't compound. Remaining candidates: (a) bidirectional vs causal FM; (b) residual structure (MNIST's low-rank digit-discriminative residual vs RHM's higher-rank diffuse residual); (c) the ratchet may require novel data to sustain compression pressure (MNIST has 10 classes with varying difficulty; stationary RHM is exhausted after cycle 1).

2. **Stabilize λ_local=0 training.** The 3-block gap at λ_local=0 achieved sweet-spot cosine (0.944) with excellent dynamics for 2 cycles before activation norm inflation destabilized it. Gradient clipping or activation-norm regularization (penalizing ||post_block3||² > threshold) could prevent the explosion without creating FM-predictability pressure. This preserves the rich dynamics (2.9× η² progression, 0.299 self-knowledge) that only appear at λ_local=0. Gen-distill + λ_local=0 is a promising combination: gen-distill already achieves cosine 0.911 with λ_local=1.0, so removing the local-loss feedback loop might push it further into the sweet spot.

3. **Self-model-driven grokking (north star).** The gen-distill run 13 shows monotonically improving val loss and L3 feature η² increasing +15% over 4 cycles at the final layer (0.389 → 0.449), with L4 also trending up (+4%). The ratchet is compounding at the representation level even though generation accuracy is flat — the compositional deepening hasn't yet crossed the threshold for behavioral improvement. The RHM's DGP is extremely compact (96 rules = 576 bits for L=6/m=2/v=8/s=2), and the 2.68M model is overparameterized by ~150,000× relative to this target — exactly the regime where grokking occurs. The FM captures 91–97% of the DGP's composition rules at learned levels, providing structured regularization pressure specifically toward the compact solution, unlike weight decay's generic L2 pressure. An extended gen-distill run (16–32+ cycles) could reveal whether the gradual L3 η² improvement eventually produces a sudden phase transition — the model "clicking" on L3 composition rules, which would make L4 accessible and potentially trigger a cascade. This would be the first instance of self-model-driven grokking: a phase transition caused by a model's compressed self-model selecting for DGP-aligned computation, rather than by weight decay selecting for low-norm solutions.
