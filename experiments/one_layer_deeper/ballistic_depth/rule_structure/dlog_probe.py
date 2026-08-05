"""Is the encoder's representation of `x` organised by the group, or is it a lookup table?

The question this settles. `variable_modulus/` §3 found that no arm learns modular squaring:
held-out `x` and held-out `N` sit at or below the accuracy obtainable with no rule knowledge
at all, and 500k steps under grokking conditions (half the bases held out, constant LR,
wd in {0.1, 1.0}) produce no phase transition. Read at *depth 1* that is not a statement about
composition at all — it says the model cannot compute a single modular squaring on an unseen
input. Everything downstream (closure, coverage, the horizon) is therefore built on a
memorised atom. The open question is *which kind* of failure that is:

  unattractive  the algorithm is reachable but lookup is cheaper, so SGD takes the table.
                Remedy: capacity economics / pressure.
  unreachable   gradient descent cannot get to the algorithm from a decimal-digit
                representation at all. Remedy: representation, not pressure.

Those have opposite remedies, so measuring first is worth more than either intervention.

The coordinate that makes it a yes/no question. For `N = p*q`,

    (Z/N)*  ~=  Z/(p-1)  x  Z/(q-1)        (CRT, with a primitive root in each factor)

and in that coordinate squaring is exactly `(a, b) -> (2a, 2b)` — the doubling map, per
factor. This is not a guess about the substrate: the *image* of that map has
`(p-1)/2 * (q-1)/2` points, which is the `reachable_states_depth_ge_1` field `TaskSpec.describe()`
already reports — 9*23 = **207** for `N=893`, 29*83 = **2407** for `N=9853`. Our own
reachable-state counts *are* the group structure. We can compute this coordinate because we
hold the trapdoor `(p, q)`; the model never sees it.

So: if the model found the group, `Enc(x)` should be organised by `(a, b)`. If it memorised,
`Enc(x)` is an arbitrary point cloud indexed by a residue. Four readouts, each with a null:

  1. `fourier`     2D DFT of the embedding grid over Z/(p-1) x Z/(q-1). A group-organised
                   representation concentrates power in few frequencies; a table spreads it
                   uniformly. Also run on the QR sublattice (even a, even b) alone, since
                   that is where every rollout past t=1 actually lives.
  2. `translation` cos(Enc(x), Enc(y)) should depend only on the *difference* (da, db) if the
                   representation respects the group. Measured as the R^2 of predicting the
                   full pairwise cosine matrix from a difference-indexed table.
  3. `linear_op`   fit ONE linear map `M` with `Enc(x^2) ~= M Enc(x)` on training bases, test
                   on held-out bases. In the group coordinate the true operator *is* linear
                   (doubling is multiplication by 2 on each factor), so this is satisfiable in
                   principle. This is the readout that licenses (or kills) a
                   `Enc(x^2) ~= M Enc(x)` training constraint as the next term to add.
  4. `heldout_op`  the model's *own* tied operator applied to a held-out base's encoding:
                   cos against `Enc(x^2)`, and whether it decodes. §2's `on_manifold_cos`
                   restricted to bases the model never trained on.

Nulls, because every one of these is a similarity statistic that can look structured by
accident:

  - `perm`   the same embeddings with the `x -> (a,b)` assignment shuffled. Kills any
             structure that comes from the group while preserving the point cloud exactly.
             This is the null that matters.
  - `init`   an untrained encoder at the same config. Catches structure that is architectural
             (tokenisation, position embeddings) rather than learned.

The sharp prediction. `ballistic_depth` §2 reports the one intervention in this program that
ever moved the generalisation axis: the label-free cycle term takes held-out-`x` from **0.001
to 0.32** at fixed `N`, unpredicted at the time. If that gain is the model partially finding
the group coordinate, `consist` should show structure on 1-3 where `base` does not, and the
gap should track the cycle weight across `w_cyc*`. If both arms are at their nulls, the 0.32
is something else and the failure is `unreachable` — which would say the remedy is the
representation, and that this is not our fight.

Pure re-analysis: loads checkpoints, trains nothing.

Usage:
  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/ballistic_depth/rule_structure/dlog_probe.py::dlog_probe \\
    --tag coldstart --seed 0
"""

from __future__ import annotations

import json
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

N_PERM_NULLS = 20
GRAM_CAP = 4000  # pairwise cosine matrix is O(m^2); 4000 -> 16M pairs, ~1.7k per difference bin
RIDGE = 1e-3


def _factorize(n: int) -> set[int]:
    fs, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def _primitive_root(p: int) -> int:
    """Smallest generator of (Z/p)*, p prime."""
    fs = _factorize(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // f, p) != 1 for f in fs):
            return g
    raise ValueError(f"no primitive root for {p}")


def group_coords(p: int, q: int, units) -> tuple:
    """`(a, b)` for every unit: the CRT discrete-log coordinate in which squaring is `x2`.

    Returns `(a, b, gp, gq)` with `a in [0, p-1)`, `b in [0, q-1)`.
    """
    import numpy as np

    gp, gq = _primitive_root(p), _primitive_root(q)
    dl_p, v = {}, 1
    for k in range(p - 1):
        dl_p[v] = k
        v = v * gp % p
    dl_q, v = {}, 1
    for k in range(q - 1):
        dl_q[v] = k
        v = v * gq % q
    a = np.array([dl_p[int(x) % p] for x in units], dtype=np.int64)
    b = np.array([dl_q[int(x) % q] for x in units], dtype=np.int64)
    return a, b, gp, gq


def _fourier_conc(emb, a, b, P: int, Q: int, top_frac: float = 0.01) -> float:
    """Fraction of non-DC spectral power in the top `top_frac` of frequencies."""
    import torch

    grid = torch.zeros(P, Q, emb.shape[1], device=emb.device, dtype=torch.float32)
    grid[a, b] = emb
    power = (torch.fft.fft2(grid, dim=(0, 1)).abs() ** 2).sum(-1)
    power[0, 0] = 0.0
    flat = power.flatten()
    k = max(1, int(round(top_frac * flat.numel())))
    return (flat.topk(k).values.sum() / flat.sum().clamp_min(1e-12)).item()


def _translation_r2(emb, a, b, P: int, Q: int) -> float:
    """R^2 of predicting the pairwise cosine matrix from the group difference alone."""
    import torch
    import torch.nn.functional as F

    en = F.normalize(emb, dim=-1)
    gram = (en @ en.T).flatten()
    key = (((a[:, None] - a[None, :]) % P) * Q + ((b[:, None] - b[None, :]) % Q)).flatten()
    n_bins = P * Q
    tot = torch.zeros(n_bins, device=emb.device).index_add_(0, key, gram)
    cnt = torch.zeros(n_bins, device=emb.device).index_add_(0, key, torch.ones_like(gram))
    pred = (tot / cnt.clamp_min(1))[key]
    ss_res = ((gram - pred) ** 2).sum()
    ss_tot = ((gram - gram.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot.clamp_min(1e-12)).item()


def _synthetic_group_emb(a, b, P: int, Q: int, d: int, n_freq: int, snr: float, gen):
    """A representation that *did* find the group — the positive control.

    Each dimension is a random combination of `n_freq` Fourier modes over Z/P x Z/Q, plus
    noise at the given signal-to-noise ratio. This is what an encoder organised by the
    discrete-log coordinate looks like, and it is the only way to know whether the observed
    concentration is large or small: a permutation null bounds it from below, this from above.
    """
    import torch

    fa = torch.randint(1, P, (n_freq,), generator=gen).to(a.device)
    fb = torch.randint(1, Q, (n_freq,), generator=gen).to(a.device)
    phase = (
        2 * torch.pi * (a[:, None] * fa[None, :] / P + b[:, None] * fb[None, :] / Q)
    )  # [n, n_freq]
    feats = torch.cat([phase.cos(), phase.sin()], dim=1)  # [n, 2*n_freq]
    mix = torch.randn(feats.shape[1], d, generator=gen).to(a.device)
    sig = feats @ mix
    noise = torch.randn(sig.shape, generator=gen).to(a.device)
    return sig / sig.std().clamp_min(1e-9) + noise * (1.0 / max(snr, 1e-9))


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=3600, memory=16384)
def dlog_probe(
    tag: str = "coldstart",
    arms: str = "base,consist",
    seed: int = 0,
    out_tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from one_layer_deeper.ballistic_depth.ballistic_depth import _make_model
    from one_layer_deeper.squaring_mod import TOKEN_IDS, TaskSpec, build_trajectories

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIGIT_OFFSET = 7
    gen = torch.Generator(device="cpu").manual_seed(0)
    results = {"tag": tag, "seed": seed, "arms": {}}

    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        ck_path = Path(DATA_DIR) / "ballistic_depth" / tag / "ckpt" / f"{arm}_seed{seed}.pt"
        if not ck_path.exists():
            print(f"[skip] no checkpoint at {ck_path}", flush=True)
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck["cfg"]

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
        train_idx = torch.tensor(data["train_idx"], device=device)
        test_idx = torch.tensor(data["test_idx"], device=device)
        n_dig, N, p, q = spec.n_answer_digits, spec.modulus, spec.p, spec.q
        P, Q = p - 1, q - 1

        # --- the coordinate. units are traj[:, 0] in the generator's own order ---------
        units = traj[:, 0].cpu().numpy()
        a_np, b_np, gp, gq = group_coords(p, q, units)
        a = torch.tensor(a_np, device=device)
        b = torch.tensor(b_np, device=device)
        # The coordinate is only correct if it is a bijection onto the grid and squaring is
        # the doubling map in it. Both are cheap to assert, and silently wrong otherwise.
        assert len(set(zip(a_np.tolist(), b_np.tolist()))) == units.size == P * Q
        sq = traj[:, 1].cpu().numpy()
        a2, b2, _, _ = group_coords(p, q, sq)
        assert np.array_equal(a2, (2 * a_np) % P) and np.array_equal(b2, (2 * b_np) % Q)

        # --- prompt encoding, identical to training ------------------------------------
        def _digits(value, width):
            return [int(c) for c in str(int(value)).rjust(width, "0")]

        n_digits_tbl = torch.tensor(
            [_digits(v, n_dig) for v in range(N)], dtype=torch.long, device=device
        )
        head_tbl = torch.cat(
            [
                torch.full((N, 1), TOKEN_IDS["BOS"], dtype=torch.long, device=device),
                torch.full((N, 1), TOKEN_IDS["N"], dtype=torch.long, device=device),
                torch.tensor([_digits(N, n_dig)], dtype=torch.long, device=device).expand(N, -1)
                + DIGIT_OFFSET,
                torch.full((N, 1), TOKEN_IDS["X"], dtype=torch.long, device=device),
                n_digits_tbl + DIGIT_OFFSET,
            ],
            dim=1,
        )
        max_len = head_tbl.shape[1] + 1
        read_pos = max_len - 1
        ans_tail = torch.tensor([TOKEN_IDS["ANS"]], dtype=torch.long, device=device)

        def prompts_for(values):
            tail = ans_tail.unsqueeze(0).expand(values.shape[0], -1)
            return torch.cat([head_tbl[values], tail], dim=1)

        arm_res = {}
        for variant in ("trained", "init"):
            model = _make_model(cfg, n_dig, arm, max_len, device)
            if variant == "trained":
                model.load_state_dict(ck["state_dict"])
            model.eval()

            with torch.no_grad():
                emb = torch.cat(
                    [
                        model.encode(prompts_for(traj[i : i + 2048, 0]), read_pos).float()
                        for i in range(0, traj.shape[0], 2048)
                    ]
                )

            # ---- 1. spectral concentration, full group and QR sublattice --------------
            real = _fourier_conc(emb, a, b, P, Q)
            nulls = []
            for _ in range(N_PERM_NULLS):
                perm = torch.randperm(emb.shape[0], generator=gen).to(device)
                nulls.append(_fourier_conc(emb[perm], a, b, P, Q))
            nulls = np.array(nulls)
            qr = (a % 2 == 0) & (b % 2 == 0)
            real_qr = _fourier_conc(emb[qr], a[qr] // 2, b[qr] // 2, P // 2, Q // 2)
            nulls_qr = []
            for _ in range(N_PERM_NULLS):
                perm = torch.randperm(int(qr.sum()), generator=gen).to(device)
                nulls_qr.append(
                    _fourier_conc(emb[qr][perm], a[qr] // 2, b[qr] // 2, P // 2, Q // 2)
                )
            nulls_qr = np.array(nulls_qr)

            # ---- 2. translation structure ---------------------------------------------
            m = min(GRAM_CAP, emb.shape[0])
            sub = torch.randperm(emb.shape[0], generator=gen)[:m].to(device)
            tr_real = _translation_r2(emb[sub], a[sub], b[sub], P, Q)
            perm = torch.randperm(m, generator=gen).to(device)
            tr_null = _translation_r2(emb[sub][perm], a[sub], b[sub], P, Q)

            # ---- 3. one linear map for the whole operator ------------------------------
            with torch.no_grad():
                emb_sq = torch.cat(
                    [
                        model.encode(prompts_for(traj[i : i + 2048, 1]), read_pos).float()
                        for i in range(0, traj.shape[0], 2048)
                    ]
                )
            x_tr, y_tr = emb[train_idx], emb_sq[train_idx]
            d = x_tr.shape[1]
            gram = x_tr.T @ x_tr + RIDGE * torch.eye(d, device=device) * x_tr.shape[0]
            w = torch.linalg.solve(gram, x_tr.T @ y_tr)
            pred_te = emb[test_idx] @ w
            y_te = emb_sq[test_idx]
            lin_cos = F.cosine_similarity(pred_te, y_te, dim=-1).mean().item()
            lin_r2 = (
                1 - ((pred_te - y_te) ** 2).sum() / ((y_te - y_te.mean(0)) ** 2).sum()
            ).item()
            with torch.no_grad():
                lin_dec = (
                    (model.dec(pred_te).argmax(-1) == n_digits_tbl[traj[test_idx, 1]])
                    .all(-1)
                    .float()
                    .mean()
                    .item()
                )

            # ---- 4. the model's own operator on held-out bases -------------------------
            with torch.no_grad():
                h1, _ = model.roll(emb[test_idx], 1)
                op_cos = F.cosine_similarity(h1, y_te, dim=-1).mean().item()
                op_dec = (
                    (model.dec(h1).argmax(-1) == n_digits_tbl[traj[test_idx, 1]])
                    .all(-1)
                    .float()
                    .mean()
                    .item()
                )

            arm_res[variant] = {
                "fourier_top1pct": real,
                "fourier_null_mean": float(nulls.mean()),
                "fourier_null_std": float(nulls.std()),
                "fourier_z": float((real - nulls.mean()) / max(nulls.std(), 1e-9)),
                "fourier_qr_top1pct": real_qr,
                "fourier_qr_null_mean": float(nulls_qr.mean()),
                "fourier_qr_z": float((real_qr - nulls_qr.mean()) / max(nulls_qr.std(), 1e-9)),
                "translation_r2": tr_real,
                "translation_r2_null": tr_null,
                "linear_op_cos_heldout": lin_cos,
                "linear_op_r2_heldout": lin_r2,
                "linear_op_decode_heldout": lin_dec,
                "model_op_cos_heldout": op_cos,
                "model_op_decode_heldout": op_dec,
            }
            r = arm_res[variant]
            print(
                f"  [{arm}/{variant}] fourier {r['fourier_top1pct']:.4f} vs null "
                f"{r['fourier_null_mean']:.4f} (z={r['fourier_z']:+.1f}) | "
                f"qr z={r['fourier_qr_z']:+.1f} | translation R2 {r['translation_r2']:.3f} vs "
                f"null {r['translation_r2_null']:.3f} | linear-op heldout cos "
                f"{r['linear_op_cos_heldout']:.3f} decode {r['linear_op_decode_heldout']:.3f} | "
                f"model-op heldout cos {r['model_op_cos_heldout']:.3f} decode "
                f"{r['model_op_decode_heldout']:.3f}",
                flush=True,
            )

        # ---- positive control: what "found the group" actually scores -----------------
        # Computed on this arm's grid so the statistic is directly comparable. Swept over
        # SNR because a real encoder would carry the group *and* a lot of other structure.
        if "control_synthetic" not in results:
            ctrl = {}
            for n_freq in (2, 8):
                for snr in (10.0, 1.0, 0.3):
                    se = _synthetic_group_emb(
                        a, b, P, Q, cfg["d_model"], n_freq, snr, torch.Generator(device="cpu").manual_seed(1)
                    )
                    m = min(GRAM_CAP, se.shape[0])
                    subc = torch.randperm(se.shape[0], generator=gen)[:m].to(device)
                    ctrl[f"nfreq{n_freq}_snr{snr}"] = {
                        "fourier_top1pct": _fourier_conc(se, a, b, P, Q),
                        "translation_r2": _translation_r2(se[subc], a[subc], b[subc], P, Q),
                    }
            results["control_synthetic"] = ctrl
            for k, v in ctrl.items():
                print(
                    f"  [control/{k}] fourier {v['fourier_top1pct']:.4f} | "
                    f"translation R2 {v['translation_r2']:.3f}",
                    flush=True,
                )

        arm_res["task"] = {
            "modulus": N,
            "p": p,
            "q": q,
            "group": [P, Q],
            "generators": [gp, gq],
            "n_units": int(units.size),
            "n_heldout_x": int(test_idx.numel()),
        }
        results["arms"][arm] = arm_res

    out_dir = Path(DATA_DIR) / "ballistic_depth" / (out_tag or tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dlog_probe_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"[saved] {out_path}", flush=True)
    return results
