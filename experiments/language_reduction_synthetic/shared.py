"""Shared Modal infrastructure for RHM scaling law experiments."""

import json
import modal

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
    )
    .add_local_python_source("language_reduction_synthetic")
)

volume = modal.Volume.from_name("rhm-scaling-data", create_if_missing=True)
app = modal.App("rhm-scaling", image=image)


def setting_key(v, s, L, m):
    return f"v{v}_s{s}_L{L}_m{m}"


class NumpyEncoder(json.JSONEncoder):
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
