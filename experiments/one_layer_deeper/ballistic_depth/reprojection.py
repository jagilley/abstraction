"""Test-time re-projection: can a trained operator be rescued without retraining?

README §8 established that the base operator *computes correctly at every depth tested* —
hand it the true residue `x_55` and it rolls the last five steps of a T=60 problem at 0.873,
while its own rollout to the same target scores 0.000. The operator is sound; the rollout
leaves its input domain. That makes the following intervention available, and it costs no
training at all:

    every k steps:  h  <-  (1-alpha) * h  +  alpha * Enc(decode(h))

i.e. periodically snap the rolled state back onto the encoder's manifold, using the model's
*own* decode. This is the cold-start probe with the model's prediction substituted for ground
truth, applied mid-rollout.

Two things are being separated, because they fail for different reasons:

  mode='self'    re-project to `Enc(x_hat_t)` where `x_hat_t = decode(h_t)`. The deployable
                 version. Costs whatever the decode gets wrong.
  mode='oracle'  re-project to `Enc(x_t)` with the TRUE residue. Not deployable — it is the
                 ceiling, and the gap to 'self' is exactly the price of decode error.

**The prediction this is built to test.** §8 found base cannot decode its own encoder's output
at zero rollout steps (0.008 against cycle's 1.000) and needs roughly three operator
applications before a state becomes readable. So re-projecting *every* step should be actively
harmful to base — each projection hands the decoder a state one step off the encoder manifold,
which is exactly where it cannot read. If the period sweep shows base failing at k=1 and
recovering at k>=3, that is the decoder-warm-up structure predicted from a completely separate
measurement. If re-projection rescues base at some k, closure is causal *and* fixable at
deployment time, which is the result that would transfer beyond this substrate.

Usage:
  MODAL_PROFILE=chromatic modal run --detach \\
    one_layer_deeper/ballistic_depth/reprojection.py::reprojection --tag coldstart --seed 0
"""

from __future__ import annotations

import json
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

PERIODS = (1, 2, 3, 4, 5, 6, 8, 10, 15)
ALPHAS = (1.0, 0.5)
EVAL_DEPTHS = (10, 20, 30, 40, 60)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=8192)
def reprojection(
    tag: str = "coldstart",
    arms: str = "base,consist",
    seed: int = 0,
    eval_cap: int = 828,
    out_tag: str = "",
):
    import numpy as np
    import torch

    from one_layer_deeper.ballistic_depth.ballistic_depth import _make_model
    from one_layer_deeper.squaring_mod import TOKEN_IDS, TaskSpec, build_trajectories

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIGIT_OFFSET = 7
    results = {"tag": tag, "seed": seed, "arms": {}}

    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        ck_path = Path(DATA_DIR) / "ballistic_depth" / tag / "ckpt" / f"{arm}_seed{seed}.pt"
        if not ck_path.exists():
            print(f"[skip] no checkpoint at {ck_path}", flush=True)
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck["cfg"]

        # Rebuild the exact task the checkpoint was trained on. `seed` drives the train/test
        # split inside TaskSpec, so it must match the training run or the "seen x" pool is wrong.
        spec = TaskSpec(
            p=cfg["p"],
            q=cfg["q"],
            train_depths=tuple(int(v) for v in str(cfg["train_depths"]).split(",")),
            eval_depths=tuple(range(1, cfg["eval_max_depth"] + 1)),
            test_fraction=cfg["test_fraction"],
            seed=cfg["seed"],
        )
        data = build_trajectories(spec)
        traj = torch.tensor(data["traj"], device=device)
        train_idx = torch.tensor(data["train_idx"], device=device)[:eval_cap]
        n_dig, N = spec.n_answer_digits, spec.modulus

        def _digits(value, width):
            return [int(c) for c in str(int(value)).rjust(width, "0")]

        n_digits_tbl = torch.tensor(
            [_digits(v, n_dig) for v in range(N)], dtype=torch.long, device=device
        )
        head_tbl = torch.cat(
            [
                torch.full((N, 1), TOKEN_IDS["BOS"], dtype=torch.long, device=device),
                torch.full((N, 1), TOKEN_IDS["N"], dtype=torch.long, device=device),
                torch.tensor([_digits(N, n_dig)], dtype=torch.long, device=device)
                .expand(N, -1) + DIGIT_OFFSET,
                torch.full((N, 1), TOKEN_IDS["X"], dtype=torch.long, device=device),
                n_digits_tbl + DIGIT_OFFSET,
            ],
            dim=1,
        )
        HEAD_LEN = head_tbl.shape[1]
        max_len = HEAD_LEN + 1          # recurrent arms: no T field in the prompt
        read_pos = max_len - 1
        ans_tail = torch.tensor([TOKEN_IDS["ANS"]], dtype=torch.long, device=device)

        def prompts_for(values):
            tail = ans_tail.unsqueeze(0).expand(values.shape[0], -1)
            return torch.cat([head_tbl[values], tail], dim=1)

        model = _make_model(cfg, n_dig, arm, max_len, device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        # place value weights: 3 digits -> integer residue
        pv = torch.tensor([10 ** (n_dig - 1 - i) for i in range(n_dig)], device=device)

        @torch.no_grad()
        def run(T, period, alpha, mode):
            h = model.encode(prompts_for(traj[train_idx, 0]), read_pos)
            for step in range(1, T + 1):
                h, _ = model.roll(h, 1)
                # never project on the final step: the decoder must read a *rolled* state,
                # which for base is the only kind it can read at all (README §8).
                if period and step % period == 0 and step < T:
                    if mode == "oracle":
                        vals = traj[train_idx, step]
                    else:
                        vals = (model.dec(h).argmax(-1) * pv).sum(-1).clamp(0, N - 1)
                    h_re = model.encode(prompts_for(vals), read_pos)
                    h = (1 - alpha) * h + alpha * h_re
            pred = model.dec(h).argmax(-1)
            return (pred == n_digits_tbl[traj[train_idx, T]]).all(-1).float().mean().item()

        arm_res = {"baseline": {}, "self": {}, "oracle": {}}
        for T in EVAL_DEPTHS:
            if T > spec.max_depth:
                continue
            arm_res["baseline"][T] = run(T, 0, 0.0, "self")
        print(f"[{arm}] no re-projection: "
              + " ".join(f"T{T}:{v:.3f}" for T, v in arm_res["baseline"].items()), flush=True)

        for mode in ("self", "oracle"):
            for alpha in ALPHAS:
                for period in PERIODS:
                    key = f"a{alpha}_k{period}"
                    arm_res[mode][key] = {
                        T: run(T, period, alpha, mode)
                        for T in EVAL_DEPTHS if T <= spec.max_depth
                    }
                row = arm_res[mode]
                print(f"[{arm}] mode={mode} alpha={alpha}", flush=True)
                for period in PERIODS:
                    r = row[f"a{alpha}_k{period}"]
                    print(f"    k={period:<3d} "
                          + " ".join(f"T{T}:{v:.3f}" for T, v in r.items()), flush=True)

        results["arms"][arm] = arm_res

    out_dir = Path(DATA_DIR) / "ballistic_depth" / (out_tag or f"reproj_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] {out_path}", flush=True)
    return results
