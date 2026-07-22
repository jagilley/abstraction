"""Shared Modal infrastructure for MuJoCo control-substrate experiments.

MuJoCo (via the plain `mujoco` Python bindings, CPU physics) is a third
controllable-dynamics DGP one rung of realism above RHM/MNIST-reaching and
below language. See `ideas/physical_control_substrate.md` for the program.

Cut #1 (contact-residual structure) uses plain MuJoCo + a PyTorch forward
model. We deliberately do NOT use MJX/JAX here: cut #1's whole point is
contact *fidelity* (plain MuJoCo's contact solver is the trustworthy one;
MJX caps contact counts), the data volume is tiny, and a PyTorch FM reuses
the entire a2a_forward metric stack. MJX earns its cost later, at the
sim2sim robustness sweep (thousands of parallel envs).
"""

import json
import modal

DATA_DIR = "/data"

# Keep top-level imports to modal/json/numpy only. `mujoco`, `torch`, and
# `matplotlib` are imported INSIDE Modal function bodies so the app can be
# submitted from a machine that doesn't have them installed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.16.3",
        "torch==2.7.0",
        "mujoco==3.2.3",
        "matplotlib==3.9.2",
    )
    .add_local_python_source("mujoco_control")
)

volume = modal.Volume.from_name("mujoco-control-data", create_if_missing=True)
app = modal.App("mujoco-control", image=image)


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
