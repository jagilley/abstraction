"""Shared Modal infrastructure for the canvas substrate (image styles).

Mirrors `one_layer_deeper/shared.py`: one image, one volume, one app for every canvas node.
The corpus is rendered LOCALLY (`render_glsl.py` needs EGL on Linux and we do not install it
here) and pushed to the volume as a tar; see `plant/corpus.py`.
"""

import json

import modal
from modal.file_pattern_matcher import NON_PYTHON_FILES

DATA_DIR = "/data"

# The canvas package holds the (gitignored) 750 MB corpus and every node's live result/figure
# directory. Mounting those is both slow and racy -- a background job writing a log under
# `plant/results/` fails the build with "modified during build process" -- so prune them from
# the walk before the default non-Python filter runs.
_PRUNE = ("/corpora/data", "/results", "/figures", "/clicks", "/__pycache__")


def _ignore(path):
    sp = str(path)
    if any(seg in sp for seg in _PRUNE) or sp.endswith((".log", ".png", ".jpg", ".npz", ".tar")):
        return True
    return NON_PYTHON_FILES(path)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.7.0",
        "pillow==11.0.0",
    )
    .add_local_python_source("canvas", ignore=_ignore)
)

volume = modal.Volume.from_name("canvas-data", create_if_missing=True)
app = modal.App("canvas", image=image)


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
