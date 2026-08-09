"""Modal infrastructure for the reading-axis experiments."""

import json
import modal

DATA_DIR = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
        "transformers==4.53.2",
        "huggingface-hub",
        "accelerate",
    )
    .add_local_python_source("a2a_forward")
)

# The Modal app and volume keep the name this cut was built under ("reading",
# after the reading-axis question it came from) even though the node is now
# `a2a_forward/conditional_revision/`. Every result and activation cache on the
# volume is keyed to that name, and STRUCTURE.md's rule is that /data paths are
# independent of source location -- renaming would orphan them for no gain.
volume = modal.Volume.from_name("reading-data", create_if_missing=True)
app = modal.App("reading", image=image)


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
        if isinstance(obj, (set, frozenset)):
            return sorted(obj)
        return super().default(obj)
