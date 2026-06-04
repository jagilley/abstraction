"""Shared Modal infrastructure for the A2A forward model experiment."""

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
    .add_local_python_source("a2a_forward")
)

volume = modal.Volume.from_name("language-reduction-data", create_if_missing=True)
app = modal.App("a2a-forward", image=image)


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
