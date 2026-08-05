"""Two checkpoint probes that the training-time instrument could not supply. Trains nothing.

**1. Cut A's weighting.** `exact_atom`'s exhaustive eval pool enumerates every `(N, y)` pair,
so a modulus contributes in proportion to its own problem-space size — at `q_cap=inf` that is
`N^2`, a 32x spread across a 3-digit family. Training draws the modulus *uniformly*, so the
eval pool is weighted towards exactly the moduli the model saw least, and `divqfull`'s numbers
are not comparable with the parent cut's `redmod_qfull` (0.773), which sampled moduli
uniformly. This re-evaluates per modulus and reports the modulus-uniform mean alongside the
per-modulus spread, so both weightings are on the record rather than one silently chosen.

**2. Cut B's decomposition.** The seam term `Enc(SQ, N, x) ~ Enc(DIV, N, x^2)` reaches MSE
0.001 on *training* `x` and the composed map still reads at floor on held-out `x`. Training
MSE cannot say why: the seam is a loss on training examples, and satisfying it on unseen `x`
*is* the multiply. So this measures, on held-out `x`:

  - `seam_mse` / `seam_cos` — did the encoder land on the division problem's encoding?
  - `oracle_seam` — feed the **true** `Enc(DIV, N, x^2)` into the operator and decode. This is
    `ballistic_depth` §8's cold-start probe moved from the rollout seam to the multiply/reduce
    seam: hand the model the state it failed to construct and ask whether the rest of the
    pipeline works. If this is high while the composed map is at floor, the null is entirely
    the multiply, and Cut C's account of Cut B stops being an inference.

  MODAL_PROFILE=chromatic modal run \\
    one_layer_deeper/rule_acquisition/exact_atom/reprobe.py::reprobe \\
    --tag sq_b --arms "sqpad_seam,sqpad_div_seam"
"""

from __future__ import annotations

import json
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def reprobe(tag: str = "divqfull", arms: str = "divqfull", seed: int = 0, per_mod: int = 32768):
    import numpy as np
    import torch

    from one_layer_deeper.rule_acquisition.exact_atom.exact_atom import (
        ARMS, BASE_ARM, DIGIT_OFFSET, DIV_TOK, SQ_TOK, certifiable_T, _make_model,
    )
    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily

    device = torch.device("cuda")
    out: dict = {"tag": tag, "seed": seed, "arms": {}}

    for arm_name in [a.strip() for a in arms.split(",") if a.strip()]:
        arm = {**BASE_ARM, **ARMS[arm_name]}
        ck = torch.load(
            Path(DATA_DIR) / "exact_atom" / tag / "ckpt" / f"{arm_name}_seed{seed}.pt",
            map_location=device, weights_only=False,
        )
        cfg, task, w = ck["cfg"], arm["task"], arm["n_digits"]
        w_x = 2 * w if task in ("div", "sqdiv") else w
        n_ans = 2 * w if task == "mul" else w
        max_len = 1 + 1 + (1 + w) + (1 + w_x) + 1
        read_pos = max_len - 1
        model = _make_model(cfg, arm, max_len, n_ans, device, n_aux=0)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        family = ModulusFamily(
            n_digits=w, min_margin=arm["min_margin"], min_factor=cfg["min_factor"],
            heldout_fraction=cfg["heldout_mod_fraction"], seed=cfg["family_seed"],
        )
        fam_train, fam_held = family.split()
        if arm["max_moduli"] and len(fam_train) > arm["max_moduli"]:
            pick = np.random.default_rng(cfg["family_seed"] + 1).choice(
                len(fam_train), arm["max_moduli"], replace=False
            )
            fam_train = sorted(fam_train[i] for i in pick)
        train_mods = [m[0] for m in fam_train]
        held_mods = [m[0] for m in fam_held]

        thr = int(round(arm["train_frac"] * 1_000_000))

        def digits_of(v, width):
            return torch.stack([(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1)

        def prompts_for(tok, n_vals, x_vals):
            rows = x_vals.shape[0]
            col = lambda t: torch.full((rows, 1), t, dtype=torch.long, device=device)
            return torch.cat(
                [col(TOKEN_IDS["BOS"]), col(tok), col(TOKEN_IDS["N"]),
                 digits_of(n_vals, w) + DIGIT_OFFSET, col(TOKEN_IDS["X"]),
                 digits_of(x_vals, w_x) + DIGIT_OFFSET, col(TOKEN_IDS["ANS"])], dim=1
            )

        gen = torch.Generator(device=device).manual_seed(999)
        res: dict = {}

        # ------------------------- Cut A: modulus-uniform eps -------------------------
        if task == "div":
            # `dense_n` arms carry ~800 moduli; 32 sampled ones stand in, and the spread is
            # reported so a single mean is never the whole claim.
            if arm["dense_n"]:
                pool = np.arange(max(2, 10 ** (w - 1)), 10**w, dtype=np.int64)
                pool = pool[((pool * 2246822519) % 10 != 0) | np.isin(pool, train_mods)]
                mods = np.random.default_rng(7).choice(pool, 32, replace=False).tolist()
            else:
                mods = train_mods
            per = {}
            for n_val in mods:
                hi = min((arm["q_cap"] + 1) * int(n_val), int(n_val) ** 2)
                keep_n, keep_y, have = [], [], 0
                for _ in range(64):
                    u = torch.rand(4 * per_mod, device=device, generator=gen, dtype=torch.float64)
                    y = torch.minimum((u * hi).long(), torch.tensor(hi - 1, device=device))
                    nv = torch.full_like(y, int(n_val))
                    k = (((y * 2654435761 + nv * 40503) % 1_000_000) < thr) == False
                    keep_n.append(nv[k]); keep_y.append(y[k]); have += int(k.sum())
                    if have >= per_mod:
                        break
                nv, y = torch.cat(keep_n)[:per_mod], torch.cat(keep_y)[:per_mod]
                err = 0
                for i in range(0, nv.numel(), 4096):
                    s = slice(i, i + 4096)
                    with torch.no_grad():
                        h = model.roll(model.enc(prompts_for(DIV_TOK, nv[s], y[s]), read_pos))
                        ok = (model.dec(h).argmax(-1) == digits_of((y % nv)[s], n_ans)).all(-1)
                    err += int((~ok).sum())
                per[int(n_val)] = err / nv.numel()
            eps_u = float(np.mean(list(per.values())))
            res["per_modulus_eps"] = per
            res["eps_modulus_uniform"] = eps_u
            res["eps_modulus_uniform_certT"] = certifiable_T(eps_u)
            res["eps_spread"] = [min(per.values()), max(per.values())]
            print(
                f"[{arm_name}] modulus-uniform eps={eps_u:.4e} (acc {1-eps_u:.4f}, "
                f"certT {certifiable_T(eps_u):.3g}) | per-modulus eps "
                f"{min(per.values()):.3e}..{max(per.values()):.3e} over {len(per)} moduli",
                flush=True,
            )
            print("   " + " ".join(f"N={k}:{v:.3f}" for k, v in sorted(per.items())[:12]), flush=True)

        # ------------------------- Cut B: the seam, on held-out x ---------------------
        if task == "sqdiv":
            urng = np.random.default_rng(cfg["family_seed"])
            pools = {}
            for n_val, p, q, _ in fam_train + fam_held:
                u = ModulusFamily.units_for(p, q)
                if arm.get("global_x_split"):
                    h = (u.astype(np.int64) * 2654435761) % 1_000_000
                    pools[n_val] = (np.sort(u[h < thr]), np.sort(u[h >= thr]))
                else:
                    perm = urng.permutation(u.size)
                    n_te = int(round((1.0 - arm["train_frac"]) * u.size))
                    pools[n_val] = (np.sort(u[perm[n_te:]]), np.sort(u[perm[:n_te]]))

            for split, which, mods in (
                ("seen_x", 0, train_mods), ("heldout_x", 1, train_mods),
                ("heldout_n", 0, held_mods),
            ):
                ns, xs = [], []
                for n_val in mods:
                    u = torch.tensor(pools[n_val][which], device=device)
                    ns.append(torch.full_like(u, n_val)); xs.append(u)
                nv, x = torch.cat(ns), torch.cat(xs)
                mse = cos = nrm0 = nrmd = 0.0
                e_own = e_oracle = 0
                for i in range(0, nv.numel(), 4096):
                    s = slice(i, i + 4096)
                    n_s, x_s = nv[s], x[s]
                    tgt = digits_of((x_s * x_s) % n_s, n_ans)
                    with torch.no_grad():
                        h0 = model.enc(prompts_for(SQ_TOK, n_s, x_s), read_pos)
                        h_div = model.enc(prompts_for(DIV_TOK, n_s, x_s * x_s), read_pos)
                        mse += float(((h0 - h_div) ** 2).mean(-1).sum())
                        cos += float(
                            torch.nn.functional.cosine_similarity(h0, h_div, dim=-1).sum()
                        )
                        # Norms decide whether a small seam MSE means "aligned" or
                        # "shrunk": MSE is not scale-free and cos is.
                        nrm0 += float(h0.norm(dim=-1).sum())
                        nrmd += float(h_div.norm(dim=-1).sum())
                        e_own += int(
                            (~(model.dec(model.roll(h0)).argmax(-1) == tgt).all(-1)).sum()
                        )
                        e_oracle += int(
                            (~(model.dec(model.roll(h_div)).argmax(-1) == tgt).all(-1)).sum()
                        )
                n = nv.numel()
                res[split] = dict(
                    n=n, seam_mse=mse / n, seam_cos=cos / n,
                    h0_norm=nrm0 / n, hdiv_norm=nrmd / n,
                    eps_own=e_own / n, acc_own=1 - e_own / n,
                    eps_oracle=e_oracle / n, acc_oracle=1 - e_oracle / n,
                    certT_oracle=certifiable_T(e_oracle / n),
                )
                r = res[split]
                print(
                    f"[{arm_name}] {split:<10} n={n:<6} seam mse={r['seam_mse']:.4f} "
                    f"cos={r['seam_cos']:.4f} |h0|={r['h0_norm']:.2f} "
                    f"|hdiv|={r['hdiv_norm']:.2f} | own h0 acc={r['acc_own']:.4f} "
                    f"| ORACLE seam acc={r['acc_oracle']:.4f} (certT {r['certT_oracle']:.3g})",
                    flush=True,
                )

        out["arms"][arm_name] = res

    d = Path(DATA_DIR) / "exact_atom" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / f"reprobe_seed{seed}.json").write_text(json.dumps(out, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"[saved] /exact_atom/{tag}/reprobe_seed{seed}.json", flush=True)
    return out
