"""Modal app for the language reduction pipeline.

Stages:
  1.  tokenize         — stream FineWeb, tokenize with GPT-2 BPE, save shards
  2.  stats            — compute vocab reduction + multi-lag covariance matrices
  3.  spectral         — SVD of covariance matrices
  4.  denoise          — context-dependent spectral denoising (parameterized by tau)
  4b. replacement-census — census of which tokens get replaced at a given tau
  4v. vocab-reduce     — vocab reduction only (no spectral denoising)
  5.  beta             — measure correlation exponent beta on (denoised) corpus
  6.  train            — train small AR model at given (tau, P, T)
  7.  gamma            — estimate conditional entropy exponent gamma from trained models
  8.  validate         — compare predicted vs empirical scaling exponent
  11. build-curriculum — extract fine-tuning curriculum from original corpus
  12. finetune         — fine-tune a pre-trained model on extracted curriculum
  13. recovery-eval    — evaluate structural integration after fine-tuning
  14. recovery-mdl     — MDL analysis: did queen reduce king's description length?
  15. mdl-paired        — paired fine-tuning: queen vs random curriculum on king loss
  16. mdl-consolidation — two-phase test: does queen help king during general training?
  17. extract-dpo-pairs — extract (chosen, rejected) pairs for DPO from orig vs denoised
  18. dpo-train         — DPO training: teach model queen is worth using
  19. more-sft          — more SFT control (same compute budget as DPO)
  20. scaffolding-eval  — evaluate all conditions with queen intact & zeroed
  21. scaffolding-persistence — generation with target tokens suppressed from logits
  22. train-glp         — train GLP on all-layer activations [h0; h1]
  23. glp-residuals     — compute directional residuals pre/post scaffolding
  24. glp-semantics     — verbalize residuals & probe manifold geometry
  22b. train-glp-embed  — train GLP on ln_f(h1) from tied-weights model (embedding-space GLP)
  25. warm-init         — GLP-guided warm initialization of concept tokens (Phase 1: lm_head)
  26. warm-init-phase2  — wte initialization: regression, inversion, SFT-on-wte-only
  27. warm-init-phase3  — gradient-signal wte initialization + full-model SFT
  28. warm-init-glp-embed — GLP-residual initialization vs contrastive centroid on tied model
      recovery-full    — run stages 11-13 in sequence
      scaffolding-full — run stages 17-20 in sequence
      queen-forward    — feed prompts with 'queen' token through all 3 models

  Comprehensive scaling (2026-05-05):
      cross-eval           — evaluate a trained model on data from a different τ
      scaling-sweep-v2     — train 7 P values at one τ with finer eval
      cross-eval-sweep     — cross-evaluate all trained models on τ=0 data
      param-check          — train larger model at P=100M per τ
      comprehensive-sweep  — full experiment: train + cross-eval + param check

  Vocab-only geometry eval (2026-05-06):
      vo-embedding-eval    — embedding geometry battery on vocab-only models
      vo-contextual-eval   — per-layer contextual embedding eval (P=100M)
      vo-collapse          — characterize many-to-one token collapse
      vo-recovery          — concept recovery (fine-tune + eval) on vocab-only model
      vo-full              — run collapse + embedding + contextual in sequence

  Continual learning benchmark (2026-05-07):
      cl-curriculum        — build curriculum for a single target word
      cl-finetune          — fine-tune vocab-only model on curriculum
      cl-eval              — evaluate recovery (embedding shifts, analogies, forgetting)
      cl-batch             — independent recovery across 6 targets at tau=0.1
      cl-dimensions        — test whether distinguishing semantic dimensions emerged

Usage:
  modal run language_reduction/modal_app.py --stage tokenize
  modal run language_reduction/modal_app.py --stage stats
  modal run language_reduction/modal_app.py --stage spectral
  modal run language_reduction/modal_app.py --stage denoise --tau 0.3
  modal run language_reduction/modal_app.py --stage beta --tau 0.3
  modal run language_reduction/modal_app.py --stage train --tau 0.3 --n-tokens 10000000 --block-size 128
  modal run language_reduction/modal_app.py --stage gamma --tau 0.3
  modal run language_reduction/modal_app.py --stage validate --tau 0.3
  modal run language_reduction/modal_app.py --stage vocab-only-sweep --tau 0.3
  modal run language_reduction/modal_app.py --stage vocab-only-sweep --taus 0.0,0.1,0.3,0.5,0.7
  modal run language_reduction/modal_app.py --stage sweep --taus 0.0,0.1,0.3,0.5,0.7
  modal run language_reduction/modal_app.py --stage comprehensive-sweep --taus 0.0,0.1,0.3,0.5,0.7
  modal run language_reduction/modal_app.py --stage scaling-sweep-v2 --tau 0.3
  modal run language_reduction/modal_app.py --stage cross-eval --tau 0.3 --n-tokens 10000000
  modal run language_reduction/modal_app.py --stage cross-eval-sweep --taus 0.0,0.1,0.3,0.5,0.7
  modal run language_reduction/modal_app.py --stage param-check --taus 0.0,0.1,0.3,0.5,0.7
"""

from language_reduction.shared import app

# Import all experiment stages so Modal discovers their @app.function decorators.
from language_reduction.experiments.pipeline.stages import (  # noqa: F401
    tokenize, compute_stats, spectral, denoise_one_shard, denoise,
    vocab_reduce,
    measure_beta_stage, train_model, measure_gamma_stage, validate_stage,
    inspect_output, train_scaling_sweep, replacement_census_stage,
)
from language_reduction.experiments.embedding.stages import (  # noqa: F401
    contextual_embedding_eval, geometric_analogy, inject_and_generate,
    analogy_breakdown, analogy_spot_check, embedding_eval, recovery_coherence,
)
from language_reduction.experiments.concept_recovery.stages import (  # noqa: F401
    build_curriculum_stage, finetune_model, recovery_eval_stage,
    recovery_mdl_stage,
)
from language_reduction.experiments.mdl.stages import (  # noqa: F401
    mdl_paired_comparison, mdl_consolidation,
)
from language_reduction.experiments.scaffolding.stages import (  # noqa: F401
    extract_dpo_pairs_stage, dpo_train_stage, more_sft_stage,
    scaffolding_eval_stage, scaffolding_persistence_stage, queen_forward_pass,
)
from language_reduction.experiments.glp_analysis.stages import (  # noqa: F401
    train_glp_stage, train_glp_embed_stage,
    glp_residuals_stage, glp_semantics_stage,
)
from language_reduction.experiments.warm_init.stages import (  # noqa: F401
    warm_init_stage, warm_init_phase2_stage, warm_init_phase3_stage,
    warm_init_glp_embed_stage, warm_init_icl_stage,
)
from language_reduction.experiments.scaling.stages import (  # noqa: F401
    cross_eval_model, scaling_sweep_v2, cross_eval_sweep,
    param_limit_check, comprehensive_scaling_sweep,
)
from language_reduction.experiments.vocab_only_geometry_eval.stages import (  # noqa: F401
    vocab_only_embedding_eval, vocab_only_contextual_eval,
    vocab_only_collapse_analysis, vocab_only_recovery_full,
    vocab_only_structure_probe,
)
from language_reduction.experiments.continual_learning.stages import (  # noqa: F401
    cl_build_curriculum, cl_finetune, cl_eval,
    cl_batch_replicate, cl_dimension_analysis,
    cl_sequential, cl_sequential_scaled,
)
from language_reduction.experiments.a2a_forward.stages import (  # noqa: F401
    a2a_train, a2a_loop_train,
)
from language_reduction.experiments.a2a_forward.scaling_sweep import (  # noqa: F401
    a2a_scaling_sweep,
)
from language_reduction.experiments.a2a_forward.analyze import (  # noqa: F401
    a2a_analyze, a2a_loop_analyze,
)
from language_reduction.experiments.a2a_forward.behavioral_residual import (  # noqa: F401
    a2a_behavioral_residual,
)
from language_reduction.experiments.a2a_forward.novelty_probe import (  # noqa: F401
    a2a_novelty_probe,
)
from language_reduction.experiments.a2a_forward.novelty_steer import (  # noqa: F401
    a2a_novelty_steer,
)
from language_reduction.experiments.a2a_forward.injection_help import (  # noqa: F401
    a2a_injection_help,
)
from language_reduction.experiments.a2a_forward.injection_help_structural import (  # noqa: F401
    a2a_injection_help_structural,
)
from language_reduction.experiments.a2a_forward.causal_substitution import (  # noqa: F401
    a2a_causal_substitution,
)
from language_reduction.experiments.a2a_forward.controlled_retrain import (  # noqa: F401
    a2a_controlled_retrain,
)
from language_reduction.experiments.a2a_forward.directional_steer import (  # noqa: F401
    a2a_directional_steer,
)
from language_reduction.experiments.a2a_forward.llama_cache_acts import (  # noqa: F401
    a2a_cache_llama_acts,
)
from language_reduction.experiments.a2a_forward.llama_train_fwd import (  # noqa: F401
    a2a_train_llama_fwd,
)
from language_reduction.experiments.a2a_forward.extended_training import (  # noqa: F401
    a2a_extended_training,
)
from language_reduction.experiments.a2a_forward.representational_divergence import (  # noqa: F401
    a2a_representational_divergence,
)
from language_reduction.experiments.a2a_forward.mirror_test import (  # noqa: F401
    a2a_mirror_test,
)
from language_reduction.experiments.a2a_forward.mirror_test_geometry_control import (  # noqa: F401
    a2a_mirror_test_geometry_control,
)


@app.local_entrypoint()
def main(
    stage: str = "tokenize",
    tau: float = 0.0,
    n_tokens: int = 1_000_000_000,
    n_lags: int = 50,
    v_prime: int = 3200,
    block_size: int = 128,
    max_shards: int = 0,
    n_steps: int = 10_000,
    taus: str = "",
    p_values: str = "1000000,10000000,100000000",
    kappa_mode: str = "energy",
    lr: float = 1e-4,
    target_tokens: int = 50_000,
    zero_embedding: bool = False,
    tie_weights: bool = False,
    mode: str = "spectral",
    ordering: str = "clustered",
    n_cl_tokens: int = 100,
    fwd_type: str = "transformer",
    predict_from: str = "post_block0",
    predict_to: str = "post_block1",
    fwd_n_layer: int = 1,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    inject_after_block: int = 1,
    open_loop: bool = False,
    source_layer: int = 7,
    target_layer: int = 8,
    llama_model: str = "unsloth/Llama-3.2-1B",
    seq_len: int = 2048,
    shard_size: int = 1_000_000,
):
    ms = max_shards if max_shards > 0 else None

    if stage == "tokenize":
        tokenize.remote(n_tokens=n_tokens)

    elif stage == "stats":
        compute_stats.remote(v_prime=v_prime, n_lags=n_lags, n_shards=ms)

    elif stage == "spectral":
        spectral.remote(n_lags=n_lags)

    elif stage == "denoise":
        denoise.remote(tau=tau, n_lags=n_lags, max_shards=ms, kappa_mode=kappa_mode)

    elif stage == "vocab-reduce":
        vocab_reduce.remote(tau=tau, max_shards=ms)

    elif stage == "beta":
        result = measure_beta_stage.remote(tau=tau, n_lags=n_lags, kappa_mode=kappa_mode,
                                           mode=mode)
        print(f"beta = {result['beta']:.4f}")

    elif stage == "train":
        result = train_model.remote(
            tau=tau, n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
            kappa_mode=kappa_mode, tie_weights=tie_weights, mode=mode,
        )
        print(f"Final val loss: {result['final_val_loss']:.4f}")

    elif stage == "train-sweep":
        results = train_scaling_sweep.remote(
            tau=tau, P_values=p_values, block_size=block_size, kappa_mode=kappa_mode,
            mode=mode,
        )
        for r in results:
            print(f"  P={r['n_tokens']:>12,}: best_val={r['best_val_loss']:.4f}")

    elif stage == "gamma":
        result = measure_gamma_stage.remote(tau=tau, kappa_mode=kappa_mode, mode=mode)
        if result and result.get("gamma"):
            print(f"gamma = {result['gamma']:.4f}")

    elif stage == "validate":
        result = validate_stage.remote(tau=tau, kappa_mode=kappa_mode, mode=mode)
        print(f"alpha_predicted={result['alpha_predicted']:.4f}, "
              f"alpha_empirical={result['alpha_empirical']}")

    elif stage == "embedding-eval":
        embedding_eval.remote(taus=taus if taus else f"{tau}")

    elif stage == "contextual-eval":
        contextual_embedding_eval.remote(taus=taus if taus else f"{tau}")

    elif stage == "geometric-analogy":
        geometric_analogy.remote()

    elif stage == "inject-queen":
        inject_and_generate.remote()

    elif stage == "analogy-breakdown":
        analogy_breakdown.remote()

    elif stage == "analogy-check":
        analogy_spot_check.remote()

    elif stage == "replacement-census":
        result = replacement_census_stage.remote(tau=tau, max_shards=ms, mode=mode)
        top = result["top_replaced"][:20]
        print(f"\nTop 20 most replaced tokens at tau={tau}:")
        for entry in top:
            print(f"  {entry['rank']:3d}. {entry['token']!r:20s} "
                  f"count={entry['count']:>8,}  "
                  f"({entry['pct_of_replacements']:.2f}%)")

    elif stage == "inspect":
        inspect_output.remote(tau=tau)

    elif stage == "build-curriculum":
        build_curriculum_stage.remote(target_tokens=target_tokens)

    elif stage == "finetune":
        finetune_model.remote(source_tau=tau, lr=lr, mode=mode)

    elif stage == "recovery-eval":
        recovery_eval_stage.remote(source_tau=tau, mode=mode)

    elif stage == "recovery-coherence":
        recovery_coherence.remote(source_tau=tau)

    elif stage == "recovery-mdl":
        recovery_mdl_stage.remote(source_tau=tau)

    elif stage == "mdl-paired":
        mdl_paired_comparison.remote(source_tau=tau, lr=lr)

    elif stage == "mdl-consolidation":
        mdl_consolidation.remote(source_tau=tau, lr=lr)

    elif stage == "scaffolding-persistence":
        scaffolding_persistence_stage.remote(source_tau=tau, zero_embedding=zero_embedding,
                                             mode=mode)

    elif stage == "queen-forward":
        queen_forward_pass.remote(source_P=n_tokens)

    elif stage == "extract-dpo-pairs":
        extract_dpo_pairs_stage.remote(tau=tau)

    elif stage == "dpo-train":
        dpo_train_stage.remote(source_tau=tau)

    elif stage == "more-sft":
        more_sft_stage.remote(source_tau=tau)

    elif stage == "scaffolding-eval":
        scaffolding_eval_stage.remote(source_tau=tau)

    elif stage == "train-glp":
        result = train_glp_stage.remote(source_tau=tau)
        print(f"GLP final loss: {result['final_loss']:.6f}, "
              f"recon cosine: {result['recon_cosine']:.4f}")

    elif stage == "glp-residuals":
        glp_residuals_stage.remote(source_tau=tau)

    elif stage == "glp-semantics":
        result = glp_semantics_stage.remote(source_tau=tau)
        print(f"\nRipeness signals: {len(result.get('ripeness_signals', []))}")

    elif stage == "warm-init":
        result = warm_init_stage.remote(source_tau=tau, sft_lr=lr, tie_weights=tie_weights)
        for cond in result.get("conditions", {}):
            r = result["conditions"][cond]
            print(f"  {cond}: zero-shot rank={r['zero_shot_mean_rank']:.1f}, "
                  f"post-SFT rank={r['sft_final_mean_rank']:.1f}")

    elif stage == "warm-init-phase2":
        result = warm_init_phase2_stage.remote(source_tau=tau, sft_lr=lr)
        for cond in result.get("conditions", {}):
            r = result["conditions"][cond]
            zs = r["zero_shot_generation"][0]["continuation"][:50]
            ps = r["sft_final_generation"][0]["continuation"][:50]
            print(f"  {cond}: zero-shot=\"{zs}\" post-SFT=\"{ps}\"")

    elif stage == "warm-init-phase3":
        result = warm_init_phase3_stage.remote(source_tau=tau, sft_lr=lr)
        for cond in result.get("conditions", {}):
            r = result["conditions"][cond]
            print(f"  {cond}: zero-shot rank={r['zero_shot_mean_rank']:.1f}, "
                  f"post-SFT rank={r['sft_final_mean_rank']:.1f}, "
                  f"first top10={r.get('sft_first_top10_step', 'never')}")

    elif stage == "train-glp-embed":
        result = train_glp_embed_stage.remote(source_tau=tau)
        print(f"GLP (embed) final loss: {result['final_loss']:.6f}, "
              f"recon cosine: {result['recon_cosine']:.4f}, "
              f"mean max wte cos: {result['mean_max_wte_cosine']:.4f}")

    elif stage == "warm-init-glp-embed":
        result = warm_init_glp_embed_stage.remote(source_tau=tau, sft_lr=lr)
        for cond in result.get("conditions", {}):
            r = result["conditions"][cond]
            print(f"  {cond}: zero-shot rank={r['zero_shot_mean_rank']:.1f}, "
                  f"post-SFT rank={r['sft_final_mean_rank']:.1f}, "
                  f"first top10={r.get('sft_first_top10_step', 'never')}")

    elif stage == "warm-init-icl":
        result = warm_init_icl_stage.remote(source_tau=tau, sft_lr=lr)
        for cond in result.get("conditions", {}):
            r = result["conditions"][cond]
            print(f"  {cond}: zero-shot rank={r['zero_shot_mean_rank']:.1f}, "
                  f"post-SFT rank={r['sft_final_mean_rank']:.1f}, "
                  f"first top10={r.get('sft_first_top10_step', 'never')}")

    elif stage == "scaffolding-full":
        print("Running full scaffolding RL pipeline...")
        extract_dpo_pairs_stage.remote(tau=tau)
        dpo_train_stage.remote(source_tau=tau, lr=lr)
        more_sft_stage.remote(source_tau=tau)
        scaffolding_eval_stage.remote(source_tau=tau)

    elif stage == "recovery-full":
        build_curriculum_stage.remote(target_tokens=target_tokens)
        finetune_model.remote(source_tau=tau, lr=lr, mode=mode)
        recovery_eval_stage.remote(source_tau=tau, mode=mode)

    elif stage == "vocab-only-sweep":
        tau_list = [float(t) for t in taus.split(",")] if taus else [tau]
        print(f"Vocab-only pipeline: taus={tau_list}")

        for t in tau_list:
            print(f"\n{'='*60}")
            print(f"vocab_only tau={t}")
            print(f"{'='*60}")

            if t > 0.0:
                vocab_reduce.remote(tau=t, max_shards=ms)

            measure_beta_stage.remote(tau=t, mode="vocab_only")
            train_scaling_sweep.remote(tau=t, P_values=p_values,
                                       block_size=block_size, mode="vocab_only")
            measure_gamma_stage.remote(tau=t, mode="vocab_only")
            result = validate_stage.remote(tau=t, mode="vocab_only")
            print(f"  tau={t}: alpha_predicted={result['alpha_predicted']:.4f}, "
                  f"alpha_empirical={result['alpha_empirical']}")

    elif stage == "sweep":
        tau_list = [float(t) for t in taus.split(",")]
        print(f"Running full sweep over taus={tau_list} (kappa_mode={kappa_mode})")

        for t in tau_list:
            print(f"\n{'='*60}")
            print(f"tau = {t}")
            print(f"{'='*60}")

            if t > 0.0:
                denoise.remote(tau=t, n_lags=n_lags, max_shards=ms,
                              kappa_mode=kappa_mode)

            measure_beta_stage.remote(tau=t, n_lags=n_lags, kappa_mode=kappa_mode)
            train_scaling_sweep.remote(tau=t, P_values=p_values, block_size=block_size,
                                      kappa_mode=kappa_mode)
            measure_gamma_stage.remote(tau=t, kappa_mode=kappa_mode)
            validate_stage.remote(tau=t, kappa_mode=kappa_mode)

    # --- Comprehensive scaling stages ---

    elif stage == "cross-eval":
        result = cross_eval_model.remote(
            model_tau=tau, model_P=n_tokens, eval_tau=0.0,
            block_size=block_size, kappa_mode=kappa_mode,
        )
        if result:
            print(f"Cross-eval loss: {result['cross_eval_loss']:.4f}")

    elif stage == "scaling-sweep-v2":
        default_7pt = "100000,300000,1000000,3000000,10000000,30000000,100000000"
        pv = default_7pt if p_values == "1000000,10000000,100000000" else p_values
        results = scaling_sweep_v2.remote(
            tau=tau, P_values=pv, block_size=block_size, kappa_mode=kappa_mode,
        )
        for r in results:
            print(f"  P={r['n_tokens']:>12,}: best_val={r['best_val_loss']:.4f}")

    elif stage == "cross-eval-sweep":
        default_7pt = "100000,300000,1000000,3000000,10000000,30000000,100000000"
        pv = default_7pt if p_values == "1000000,10000000,100000000" else p_values
        results = cross_eval_sweep.remote(
            taus=taus or "0.0,0.1,0.3,0.5,0.7",
            P_values=pv,
            block_size=block_size, kappa_mode=kappa_mode,
        )
        for r in results:
            print(f"  τ={r['model_tau']}, P={r['model_P']:>12,}: "
                  f"cross_eval_loss={r['cross_eval_loss']:.4f}")

    elif stage == "param-check":
        results = param_limit_check.remote(
            taus=taus or "0.0,0.1,0.3,0.5,0.7",
            P=100_000_000, block_size=block_size, kappa_mode=kappa_mode,
        )
        for r in results["train"]:
            print(f"  τ={r['tau']}: best_val={r['best_val_loss']:.4f}")

    elif stage == "comprehensive-sweep":
        default_7pt = "100000,300000,1000000,3000000,10000000,30000000,100000000"
        pv = default_7pt if p_values == "1000000,10000000,100000000" else p_values
        results = comprehensive_scaling_sweep.remote(
            taus=taus or "0.0,0.1,0.3,0.5,0.7",
            P_values=pv,
            block_size=block_size, kappa_mode=kappa_mode,
        )

    # --- Vocab-only geometry eval stages ---

    elif stage == "vo-embedding-eval":
        vocab_only_embedding_eval.remote(taus=taus or "0.1,0.3,0.5")

    elif stage == "vo-contextual-eval":
        vocab_only_contextual_eval.remote(taus=taus or "0.1,0.3,0.5")

    elif stage == "vo-collapse":
        vocab_only_collapse_analysis.remote(taus=taus or "0.1,0.3,0.5")

    elif stage == "vo-recovery":
        vocab_only_recovery_full.remote(source_tau=tau, lr=lr)

    elif stage == "vo-structure":
        vocab_only_structure_probe.remote(tau=tau)

    elif stage == "vo-full":
        print("Running full vocab-only geometry eval pipeline...")
        vocab_only_collapse_analysis.remote(taus=taus or "0.1,0.3,0.5")
        vocab_only_embedding_eval.remote(taus=taus or "0.1,0.3,0.5")
        vocab_only_contextual_eval.remote(taus=taus or "0.1,0.3,0.5")

    # --- Continual learning benchmark stages ---

    elif stage == "cl-curriculum":
        cl_build_curriculum.remote(tau=tau, target_tokens=target_tokens)

    elif stage == "cl-finetune":
        result = cl_finetune.remote(tau=tau, lr=lr)
        print(f"Best val loss: {result['best_val_loss']:.4f}")

    elif stage == "cl-eval":
        cl_eval.remote(tau=tau)

    elif stage == "cl-batch":
        results = cl_batch_replicate.remote(tau=tau, lr=lr, target_tokens=target_tokens)
        print(f"\nBatch complete: {len(results)} targets evaluated")

    elif stage == "cl-dimensions":
        cl_dimension_analysis.remote(tau=tau)

    elif stage == "cl-sequential":
        results = cl_sequential.remote(tau=tau, ordering=ordering, lr=lr,
                                       target_tokens=target_tokens)
        n_rounds = len(results.get("targets", []))
        print(f"\nSequential recovery complete: {n_rounds} rounds, ordering={ordering}")

    elif stage == "cl-scaled":
        results = cl_sequential_scaled.remote(tau=tau, n_tokens=n_cl_tokens, lr=lr,
                                              target_tokens_per_word=target_tokens)
        print(f"\nScaled sequential recovery complete: {results.get('n_tokens', '?')} tokens")

    elif stage == "a2a-train":
        result = a2a_train.remote(
            n_tokens=n_tokens, block_size=block_size, n_steps=n_steps, lr=lr,
            fwd_type=fwd_type, predict_from=predict_from, predict_to=predict_to,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print(f"A2A training complete:")
        print(f"  best_val_loss={result['best_val_loss']:.4f}")
        print(f"  final_fwd_mse={result['final_val_fwd_mse']:.4f}")
        print(f"  final_cosine={result['final_val_cosine']:.4f}")
        if "analysis" in result:
            a = result["analysis"]
            print(f"  residual-LM correlation={a.get('residual_lm_loss_correlation', 0):.4f}")

    elif stage == "a2a-analyze":
        result = a2a_analyze.remote(
            n_tokens=n_tokens, block_size=block_size,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print(f"A2A analysis complete:")
        pca = result["residual_pca"]
        print(f"  Residual effective rank: {pca['effective_rank_entropy']:.1f}")
        print(f"  Rank for 90%: {pca['rank_90']}, 95%: {pca['rank_95']}")
        cka = result["cka"]
        print(f"  CKA post-attn: {cka['post_attention_fwd_vs_block1']:.4f}")
        print(f"  CKA output: {cka['output_fwd_vs_block1']:.4f}")

    elif stage == "a2a-behavioral-residual":
        result = a2a_behavioral_residual.remote(
            n_tokens=n_tokens, block_size=block_size,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A behavioral residual analysis complete:")
        corr = result["F_correlations"]
        print(f"  r(attn entropy, residual):   {corr['attn_entropy_vs_residual']:+.4f}")
        print(f"  r(block1 contrib, residual): {corr['block1_contrib_vs_residual']:+.4f}")
        print(f"  r(LM loss, residual):        {corr['lm_loss_vs_residual']:+.4f}")
        print(f"  r(dist dominant, residual):  {corr['dist_dominant_vs_residual']:+.4f}")
        for section in ["A_attention_categories", "B_syntactic_categories",
                        "C_prediction_difficulty", "D_context_integration"]:
            print(f"  {section}:")
            for cat, stats in result[section].items():
                print(f"    {cat}: mean={stats['mean']:.4f}, d={stats['cohen_d']:+.3f}")

    elif stage == "a2a-novelty-probe":
        result = a2a_novelty_probe.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A novelty probe complete:")
        print("  AUC (open / closed / Δ) by category and layer:")
        for cat, layers in result["auc"].items():
            for lk, v in layers.items():
                print(f"    {cat:>16s} {lk:>12s}: "
                      f"{v['open_loop']:.4f} / {v['closed_loop']:.4f} / "
                      f"{v['delta']:+.4f}")
        print("  Linear probe R² (open / closed / Δ):")
        for tgt, layers in result["scalar"].items():
            for lk, v in layers.items():
                print(f"    {tgt:>20s} {lk:>12s}: "
                      f"{v['open_loop']['r2']:.4f} / {v['closed_loop']['r2']:.4f} / "
                      f"{v['delta_r2']:+.4f}")

    elif stage == "a2a-injection-help":
        result = a2a_injection_help.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A injection-help analysis complete:")
        print(f"  mean help (Δloss) = {result['mean_help']:+.5f}")
        print(f"  corr(help, novelty) no-inj   = "
              f"{result['corr_help_vs_residual_no_inj']:+.4f}")
        print(f"  corr(help, novelty) with-inj = "
              f"{result['corr_help_vs_residual_with_inj']:+.4f}")
        print("  help by novelty quartile (abs / rel / base_loss):")
        for k, v in result["help_by_novelty_quartile"].items():
            print(f"    {k:>20s}: {v['mean_help']:+.5f} / {v['mean_rel_help']:+.4f}"
                  f" / {v['mean_baseline_loss']:.3f}")
        lc = result["loss_controlled"]
        print(f"  loss-controlled: within-bin corr(help,novelty)="
              f"{lc['stratified_corr_help_vs_novelty']:+.4f}, "
              f"help Q4-Q1={lc['stratified_help_Q4_minus_Q1']:+.5f} "
              f"(raw {lc['raw_help_Q4_minus_Q1']:+.5f})")

    elif stage == "a2a-injection-help-structural":
        result = a2a_injection_help_structural.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A structural injection-help complete:")
        h = result["headline"]
        print(f"  corr(help, norm) = {result['corr_help_vs_residual_norm']:+.4f}")
        print(f"  eta² direction-clusters={h['cluster_eta2']:.4f}  "
              f"norm-octiles={h['norm_octile_eta2']:.4f}  "
              f"(structure>magnitude: {h['structure_beats_magnitude']})")
        print("  help by attention distance:")
        for k, v in result["help_by_attention_distance"].items():
            print(f"    {k:>18s}: {v['mean_help']:+.5f}  (norm {v['mean_norm']:.3f})")
        lr = result["longrange_vs_local_help"]
        print(f"  long-range vs local help: within-decile={lr['within_norm_decile']:+.5f}"
              f"  raw={lr['raw']:+.5f}")

    elif stage == "a2a-novelty-steer":
        result = a2a_novelty_steer.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A novelty steering complete:")
        print(f"  corr(reliance, residual_norm) at s=0: "
              f"{result['reliance_vs_residual_corr_at_s0']:+.4f}")
        print("  Slope dR/ds (ΔKL per std of steering):")
        for k, v in result["slopes"].items():
            print(f"    {k:>16s}: {v:+.5f}")

    elif stage == "a2a-loop-analyze":
        result = a2a_loop_analyze.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
            open_loop=open_loop,
        )
        print(f"A2A loop analysis complete:")
        for k, v in result.items():
            if isinstance(v, dict):
                r2 = v.get('r2', v.get('cosine', ''))
                cos = v.get('cosine', '')
                print(f"  {k}: R²={r2:.4f}, cos={cos:.4f}" if isinstance(r2, float)
                      else f"  {k}: {v}")
            else:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    elif stage == "a2a-loop-train":
        result = a2a_loop_train.remote(
            n_tokens=n_tokens, block_size=block_size, n_steps=n_steps, lr=lr,
            fwd_lr=1e-3,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print(f"A2A closed-loop training complete:")
        print(f"  best_val_loss={result['best_val_loss']:.4f}")
        print(f"  final lm_loop={result['final_val_lm_loop']:.4f} "
              f"lm_base={result['final_val_lm_base']:.4f}")
        print(f"  final_fwd_cosine={result['final_val_cosine']:.4f}")
        print(f"  gate_norm={result['final_gate_norm']:.4f}")
        sm = result.get("selfmap_probe", {})
        if sm:
            print(f"  self-map R² with_inj={sm.get('r2_with_injection', 0):.4f} "
                  f"without={sm.get('r2_without_injection', 0):.4f}")

    elif stage == "a2a-causal-sub":
        result = a2a_causal_substitution.remote(
            n_tokens=n_tokens, block_size=block_size,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head, fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A causal substitution analysis complete:")
        ov = result["overall"]
        print(f"  KL_sub={ov['kl_sub']:.4f}  KL_abl={ov['kl_abl']:.4f}")
        print(f"  ΔCE_sub={ov['delta_ce_sub']:+.4f}  ΔCE_abl={ov['delta_ce_abl']:+.4f}")
        print(f"  Acc_normal={ov['acc_normal']:.4f}  Acc_sub={ov['acc_sub']:.4f}  "
              f"Acc_abl={ov['acc_abl']:.4f}")
        print("  Per category (KL_sub / KL_abl):")
        for cat, stats in result["per_category"].items():
            if stats["n"] > 0:
                print(f"    {cat:20s}: {stats['kl_sub']:.4f} / {stats['kl_abl']:.4f}")

    elif stage == "a2a-scaling-sweep":
        result = a2a_scaling_sweep.remote(
            n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
            predict_from=predict_from, predict_to=predict_to,
        )
        print("A2A scaling sweep complete:")
        mm = result["main_model"]
        print(f"  Main model: {mm['n_params']:,} params, "
              f"best_val_loss={mm['best_val_loss']:.4f}")
        print(f"  {'Config':>8s} {'Ratio':>7s} {'r(res,LM)':>10s} "
              f"{'EffRank':>8s} {'d_SS':>6s} {'d_BC':>6s}")
        for c in result["configs"]:
            print(f"  {c['config_name']:>8s} "
                  f"{c['capacity_ratio']:>6.1%} "
                  f"{c['residual_lm_loss_correlation']:>+10.4f} "
                  f"{c['residual_pca']['effective_rank']:>8.1f} "
                  f"{c['behavioral_effects']['cohen_d_sentence_start']:>+6.3f} "
                  f"{c['behavioral_effects']['cohen_d_before_closer']:>+6.3f}")

    elif stage == "a2a-controlled-retrain":
        result = a2a_controlled_retrain.remote(
            n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
            lr=lr if lr != 1e-4 else 3e-4,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
            fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A controlled retrain complete:")
        lm = result["lm_loss"]
        print(f"  Open-loop LM:              {lm['open_final']:.4f}")
        print(f"  Closed-loop LM (with inj): {lm['closed_with_inj_final']:.4f}")
        print(f"  Closed-loop LM (no inj):   {lm['closed_no_inj_final']:.4f}")
        fq = result["fwd_quality"]
        print(f"  Fwd cosine — open: {fq['open']['cosine']:.4f}, "
              f"closed: {fq['closed']['cosine']:.4f}")
        print("\n  Localization test (Δ R² = closed − open):")
        vec = result["probe_results"]["vector"]["residual"]
        for lk in sorted(vec.keys()):
            d = vec[lk]["delta_r2"]
            print(f"    {lk}: Δ R² = {d:+.4f}")

    elif stage == "a2a-directional-steer":
        result = a2a_directional_steer.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
            fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A directional steering complete:")
        diag = result["diagonality"]
        print(f"  Full vector probe R2: {result['probe_r2_full_vector']:.4f}")
        print(f"  |diag|/|off| ratio:   {diag['diag_to_off_ratio']:.3f}")
        print(f"  Diagonal enrichment:  {diag['diagonal_enrichment']:.4f} "
              f"(uniform: {diag['uniform_enrichment']:.4f})")
        print(f"  Signed diag: {diag['mean_signed_diagonal']:+.6f}, "
              f"off: {diag['mean_signed_off_diagonal']:+.6f}")
        print("  Per-direction selectivity:")
        for k, v in diag["per_direction_selectivity"].items():
            print(f"    {k}: {v:.3f}")

    elif stage == "a2a-extended-training":
        result = a2a_extended_training.remote(
            n_tokens=n_tokens, block_size=block_size, n_steps=n_steps,
            lr=lr if lr != 1e-4 else 3e-4,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
            fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A extended training complete:")
        lm = result["final_lm_loss"]
        print(f"  Open-loop LM:              {lm['open']:.4f}")
        print(f"  Closed-loop LM (with inj): {lm['closed_with_inj']:.4f}")
        print(f"  Closed-loop LM (no inj):   {lm['closed_no_inj']:.4f}")
        fq = result["fwd_quality"]
        print(f"  Fwd cosine — open: {fq['open']['cosine']:.4f}, "
              f"closed: {fq['closed']['cosine']:.4f}")
        print("\n  Final probe results (Δ R² = closed − open):")
        vec = result["probe_results"]["vector"]["residual"]
        for lk in sorted(vec.keys()):
            d = vec[lk]["delta_r2"]
            print(f"    {lk}: Δ R² = {d:+.4f}")

    elif stage == "a2a-rep-divergence":
        result = a2a_representational_divergence.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
            fwd_mlp_mult=fwd_mlp_mult,
        )
        print("A2A representational divergence analysis complete:")
        print("\n  CKA (open vs closed):")
        for k, v in result["step1_cka"].items():
            print(f"    {k:>15s}: {v:.6f}")
        print("\n  Diff PCA:")
        for k, v in result["step2_pca"].items():
            print(f"    {k:>15s}: eff_rank={v['effective_rank']:.1f}, "
                  f"top10={v['top10_frac']:.3f}")
        print("\n  Alignment (residual var in diff PCs):")
        for k, v in result["step3_alignment"].items():
            r10 = v["residual_var_in_diff_pcs"].get("top_10", {})
            print(f"    {k:>15s}: Δ R²={v['delta_r2']:+.4f}, "
                  f"ratio(top10)={r10.get('ratio', 0):.2f}x")

    elif stage == "a2a-mirror-test":
        ckpt_source = "controlled"
        ckpt_step = 0
        if fwd_n_layer > 2 or fwd_d_head > 64 or fwd_n_head > 1 or fwd_mlp_mult > 2:
            ckpt_source = "extended"
            ckpt_step = n_steps
        result = a2a_mirror_test.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
            fwd_mlp_mult=fwd_mlp_mult,
            ckpt_source=ckpt_source, ckpt_step=ckpt_step,
        )
        print("A2A mirror test complete:")
        v = result["verdicts"]
        for cond, d in v.items():
            print(f"  {cond:>6s}: SK frac = {d['mean_sk_frac']:.4f} "
                  f"({d['ratio_vs_random']:.2f}x random)")
        print(f"  PASSES: {result['passes_mirror_test']}")

    elif stage == "a2a-mirror-geometry":
        ckpt_source = "controlled"
        ckpt_step = 0
        if fwd_n_layer > 2 or fwd_d_head > 64 or fwd_n_head > 1 or fwd_mlp_mult > 2:
            ckpt_source = "extended"
            ckpt_step = n_steps
        result = a2a_mirror_test_geometry_control.remote(
            n_tokens=n_tokens, block_size=block_size,
            predict_from=predict_from, predict_to=predict_to,
            inject_after_block=inject_after_block,
            fwd_n_layer=fwd_n_layer,
            fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
            fwd_mlp_mult=fwd_mlp_mult,
            ckpt_source=ckpt_source, ckpt_step=ckpt_step,
        )
        print("Mirror test geometry control complete:")
        for cond, r in result["results"].items():
            print(f"  {cond:>6s}: general SK = {r['per_position_sk_frac']:.4f} "
                  f"({r['per_position_ratio']:.2f}x)")
        if "comparison" in result:
            print("\n  General vs perturbation:")
            for cond, c in result["comparison"].items():
                print(f"  {cond:>6s}: general={c['general_sk_frac']:.4f} "
                      f"perturb={c['perturbation_sk_frac']:.4f} "
                      f"Δ={c['difference']:+.4f}")

    elif stage == "a2a-cache-llama":
        result = a2a_cache_llama_acts.remote(
            n_tokens=n_tokens,
            model_name=llama_model,
            source_layer=source_layer,
            target_layer=target_layer,
            seq_len=seq_len,
            shard_size=shard_size,
        )
        print(f"Llama activation caching complete:")
        print(f"  Tokens: {result['n_tokens_cached']:,}")
        print(f"  Shards: {result['n_shards']}")
        print(f"  Storage: ~{result['storage_gb_approx']:.0f} GB")
        print(f"  Throughput: {result['tokens_per_sec']:,.0f} tok/s")

    elif stage == "a2a-train-llama":
        result = a2a_train_llama_fwd.remote(
            source_layer=source_layer,
            target_layer=target_layer,
            seq_len=seq_len,
            batch_size=16,
            fwd_lr=1e-3,
            n_steps=n_steps,
            fwd_n_layer=fwd_n_layer,
            fwd_n_head=fwd_n_head,
            fwd_d_head=fwd_d_head,
            fwd_mlp_mult=fwd_mlp_mult,
        )
        print(f"Llama forward model training complete:")
        print(f"  Capacity: {result['fwd_n_params']:,} params "
              f"({result['capacity_ratio']:.2%} of main)")
        print(f"  Cosine: {result['final_val_cosine']:.4f}")
        print(f"  MSE: {result['final_val_mse']:.6f}")
        print(f"  Residual norm: {result['final_val_residual_norm']:.3f}")
        pca = result["residual_pca"]
        print(f"  Effective rank: {pca['effective_rank']:.1f}/{pca['max_rank']}")
        print(f"  Training time: {result['training_seconds']:.0f}s")

    else:
        print(f"Unknown stage: {stage}")
        print("Available: tokenize, stats, spectral, denoise, vocab-reduce, "
              "vocab-only-sweep, beta, train, "
              "train-sweep, gamma, validate, replacement-census, "
              "embedding-eval, contextual-eval, "
              "geometric-analogy, inject-queen, analogy-breakdown, analogy-check, "
              "inspect, build-curriculum, finetune, recovery-eval, recovery-mdl, "
              "mdl-paired, mdl-consolidation, recovery-full, "
              "extract-dpo-pairs, dpo-train, more-sft, scaffolding-eval, "
              "scaffolding-persistence, scaffolding-full, "
              "train-glp, train-glp-embed, glp-residuals, glp-semantics, warm-init, "
              "warm-init-phase2, warm-init-phase3, "
              "warm-init-glp-embed, warm-init-icl, sweep, "
              "cross-eval, scaling-sweep-v2, cross-eval-sweep, "
              "param-check, comprehensive-sweep, "
              "vo-embedding-eval, vo-contextual-eval, vo-collapse, "
              "vo-recovery, vo-structure, vo-full, "
              "cl-curriculum, cl-finetune, cl-eval, cl-batch, cl-dimensions, "
              "cl-sequential, cl-scaled, a2a-train, a2a-analyze, "
              "a2a-loop-train, a2a-loop-analyze, a2a-causal-sub, "
              "a2a-behavioral-residual, a2a-scaling-sweep, "
              "a2a-controlled-retrain, a2a-directional-steer, "
              "a2a-extended-training, "
              "a2a-cache-llama, a2a-train-llama")
