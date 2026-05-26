"""Shared Modal infrastructure for the language reduction pipeline."""

import json
import modal

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "tiktoken",
        "datasets",
        "torch==2.7.0",
        "transformers",
        "huggingface-hub",
        "matplotlib",
    )
    .add_local_python_source("language_reduction")
)

volume = modal.Volume.from_name("language-reduction-data", create_if_missing=True)
app = modal.App("language-reduction", image=image)


def resolve_mode_key(mode: str = "spectral", tau: float = 0.0,
                     kappa_mode: str = "energy") -> str:
    """Return the path component that identifies a corpus/model variant.

    "spectral" (default): tau-based spectral denoising → "tau_0.300"
    "vocab_only": frequency-based vocab reduction only → "vocab_only/tau_0.300"
    """
    if mode == "vocab_only":
        return f"vocab_only/tau_{tau:.3f}"
    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    return f"tau_{tau:.3f}{suffix}"


def resolve_data_dir(mode: str = "spectral", tau: float = 0.0,
                     kappa_mode: str = "energy") -> str:
    """Return the directory containing training tokens for this mode."""
    if mode == "vocab_only":
        return f"{DATA_DIR}/vocab_only/tau_{tau:.3f}"
    if tau == 0.0:
        return f"{DATA_DIR}/tokens"
    suffix = "" if kappa_mode == "energy" else f"_{kappa_mode}"
    return f"{DATA_DIR}/denoised/tau_{tau:.3f}{suffix}"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        import numpy as np

        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
