"""Shared Modal infrastructure for the One Layer Deeper (repeated modular squaring) node."""

import json

import modal

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.7.0",
    )
    .add_local_python_source("one_layer_deeper")
)

volume = modal.Volume.from_name("one-layer-deeper-data", create_if_missing=True)
app = modal.App("one-layer-deeper", image=image)


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
