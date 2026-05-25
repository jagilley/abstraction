from dataclasses import dataclass


@dataclass
class LangReducConfig:
    # --- data ---
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    n_tokens_total: int = 10_000_000_000
    n_tokens_stats: int = 1_000_000_000
    shard_size: int = 10_000_000  # tokens per shard file

    # --- vocabulary reduction ---
    v_prime: int = 3200

    # --- covariance ---
    n_lags: int = 50

    # --- denoising ---
    tau: float = 0.5
    kappa_mode: str = "energy"  # "energy" (legacy) or "effective_rank" (decoupled)

    # --- AR model for gamma estimation ---
    model_n_layer: int = 4
    model_n_head: int = 4
    model_n_embd: int = 256
    model_block_size: int = 128
    model_batch_size: int = 64
    model_lr: float = 3e-4
    model_n_steps: int = 10_000

    # --- paths on Modal volume ---
    tokens_dir: str = "/data/tokens"
    stats_dir: str = "/data/stats"
    denoised_dir: str = "/data/denoised"
    models_dir: str = "/data/models"
    results_dir: str = "/data/results"
