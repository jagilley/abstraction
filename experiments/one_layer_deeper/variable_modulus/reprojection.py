"""Test-time re-projection under a *variable* modulus — does the rule come back too?

`ballistic_depth` §9 established that closure is causal and installable at deployment time:
every `k` steps, snap the rolled state back onto the encoder manifold using the model's own
decode,

    h  <-  (1-alpha) * h  +  alpha * Enc(x_hat_t),     x_hat_t = decode(h_t)

took the ungrounded fixed-`N` model from 0.000 to 0.377 at T=60 and the grounded one to 1.000
at every depth, with no retraining anywhere.

The variable-modulus setting adds a prediction that does not exist at fixed `N`. Here the snap
is `Enc(N, x_hat_t)`, and `N` is an **input, not a label** — so the re-encode restores the
*rule* as well as the *state*. That makes the intervention do double duty for `fold`, whose
rule lives in the rolled state and can drift, and single duty for `cond`, which re-injects the
rule from `N`'s digits at every step and therefore has nothing to restore. So:

  P6  re-projection gain is LARGER for `fold` than for `cond`, and `fold` + re-projection
      should approach `cond`. If instead the two gain equally, rule drift is not a real cost
      on this substrate and the `fold`/`cond` distinction is doing less than the design assumes.

The two modes separate the state error from the decode error, as in §9:

  mode='self'    snap to `Enc(N, decode(h_t))` — deployable; costs whatever the decode gets wrong.
  mode='oracle'  snap to `Enc(N, x_t)` with the true residue — the ceiling, and not deployable.
                 An oracle snap *is* a cold start, so its plateau is the per-restart rate `p`
                 that the chain model `p ** ceil(T/k)` is built from.

Loads a checkpoint and trains nothing. The task tables and the train/test split are rebuilt
from the checkpoint's own `cfg` so the "seen x" pool matches training exactly, and the k=0
(no re-projection) condition is asserted against the stored `exact_seen_n_seen_x` — a pool
mismatch would otherwise silently corrupt every comparison here.

Usage:
  MODAL_PROFILE=chromatic modal run --detach \\
    one_layer_deeper/variable_modulus/reprojection.py::reprojection --tag cut1 --seed 0
"""

from __future__ import annotations

import json
from pathlib import Path

from one_layer_deeper.shared import DATA_DIR, NumpyEncoder, app, volume

PERIODS = (1, 2, 3, 4, 5, 6, 8, 10)
ALPHAS = (1.0, 0.5)
DIGIT_OFFSET = 7


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=16384)
def reprojection(
    tag: str = "cut1",
    arms: str = "fold,cond,fold_cyc,cond_cyc",
    seed: int = 0,
    eval_cap: int = 1024,
    out_tag: str = "",
):
    import numpy as np
    import torch

    from one_layer_deeper.squaring_mod import TOKEN_IDS, ModulusFamily
    from one_layer_deeper.variable_modulus.variable_modulus import ARM_SPEC, _make_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_dir = Path(DATA_DIR) / "variable_modulus" / tag / "ckpt"
    res_path = Path(DATA_DIR) / "variable_modulus" / tag / f"results_seed{seed}.json"
    stored = json.loads(res_path.read_text()) if res_path.exists() else {"arms": {}}

    results = {"tag": tag, "seed": seed, "arms": {}}

    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        ck_path = ck_dir / f"{arm}_seed{seed}.pt"
        if not ck_path.exists():
            print(f"[skip] no checkpoint {ck_path}", flush=True)
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        spec = ARM_SPEC[arm]
        see_n, use_cond = spec["see_n"], spec["cond"]
        n_dig = cfg["n_digits"]
        eval_max_depth = cfg["eval_max_depth"]

        # ---- rebuild the DGP exactly as training did ------------------------------
        family = ModulusFamily(
            n_digits=n_dig,
            min_margin=cfg["min_margin"],
            min_factor=cfg["min_factor"],
            heldout_fraction=cfg["heldout_mod_fraction"],
            seed=cfg["family_seed"],
            mod_lo=cfg["mod_lo"],
            mod_hi=cfg["mod_hi"],
        )
        fam_train, fam_held = family.split()
        if cfg["max_moduli"] and len(fam_train) > cfg["max_moduli"]:
            pick = np.random.default_rng(cfg["family_seed"] + 1).choice(
                len(fam_train), cfg["max_moduli"], replace=False
            )
            fam_train = sorted(fam_train[i] for i in pick)
        members = fam_train + fam_held
        mod_vals = np.array([m[0] for m in members], dtype=np.int64)
        is_train_mod = np.zeros(len(members), dtype=bool)
        is_train_mod[: len(fam_train)] = True

        rng = np.random.default_rng(cfg["family_seed"])
        tr_pools, te_pools = [], []
        for _, p, q, _ in members:
            u = ModulusFamily.units_for(p, q)
            perm = rng.permutation(u.size)
            n_te = int(round(cfg["test_x_fraction"] * u.size))
            te_pools.append(np.sort(u[perm[:n_te]]))
            tr_pools.append(np.sort(u[perm[n_te:]]))

        def _pad(pools):
            w = max(a.size for a in pools)
            out = np.zeros((len(pools), w), dtype=np.int64)
            cnt = np.zeros(len(pools), dtype=np.int64)
            for i, a in enumerate(pools):
                out[i, : a.size] = a
                cnt[i] = a.size
            return torch.tensor(out, device=device), torch.tensor(cnt, device=device)

        units_tr, cnt_tr = _pad(tr_pools)
        units_te, cnt_te = _pad(te_pools)
        mods_t = torch.tensor(mod_vals, device=device)
        train_mod_idx = torch.tensor(np.flatnonzero(is_train_mod), device=device)
        held_mod_idx = torch.tensor(np.flatnonzero(~is_train_mod), device=device)

        def digits_of(v, width=n_dig):
            return torch.stack([(v // (10**k)) % 10 for k in range(width - 1, -1, -1)], dim=1)

        def sample_pairs(pool, units, counts, n, generator=None):
            mi = pool[torch.randint(0, pool.numel(), (n,), device=device, generator=generator)]
            c = counts[mi]
            ui = (torch.rand(n, device=device, generator=generator) * c).long().clamp_max_(c - 1)
            return mods_t[mi], units[mi, ui]

        def trajectory(n_vals, x0, depth):
            out, x = [x0], x0
            for _ in range(depth):
                x = (x * x) % n_vals
                out.append(x)
            return torch.stack(out, dim=1)

        ev_gen = torch.Generator(device=device).manual_seed(12345)
        pools = {}
        for name, pool, units, counts in (
            ("seen_n_seen_x", train_mod_idx, units_tr, cnt_tr),
            ("seen_n_heldout_x", train_mod_idx, units_te, cnt_te),
            ("heldout_n", held_mod_idx, units_tr, cnt_tr),
        ):
            nv, xv = sample_pairs(pool, units, counts, cfg["eval_cap"], generator=ev_gen)
            pools[name] = (nv[:eval_cap], xv[:eval_cap], trajectory(nv, xv, eval_max_depth)[:eval_cap])

        max_len = 1 + (1 + n_dig if see_n else 0) + 1 + n_dig + 1
        read_pos = max_len - 1

        def prompts_for(n_vals, x_vals):
            b = x_vals.shape[0]
            col = lambda t: torch.full((b, 1), TOKEN_IDS[t], dtype=torch.long, device=device)
            parts = [col("BOS")]
            if see_n:
                parts += [col("N"), digits_of(n_vals) + DIGIT_OFFSET]
            parts += [col("X"), digits_of(x_vals) + DIGIT_OFFSET, col("ANS")]
            return torch.cat(parts, dim=1)

        model = _make_model(cfg, spec, max_len, device)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        @torch.no_grad()
        def run(pool_name, T, k, alpha, mode):
            n_vals, x0, traj = pools[pool_name]
            r = model.rule_vec(digits_of(n_vals)) if use_cond else None
            h = model.enc(prompts_for(n_vals, x0), read_pos)
            for t in range(1, T + 1):
                h = model.op(h, r)
                if k and t % k == 0 and t < T:
                    if mode == "oracle":
                        tgt = traj[:, t]
                    else:
                        pred = model.dec(h).argmax(-1)
                        tgt = sum(pred[:, j] * 10 ** (n_dig - 1 - j) for j in range(n_dig))
                    # Enc(N, .) re-supplies the RULE as well as the state — the variable-N
                    # specific half of the intervention (P6).
                    h_snap = model.enc(prompts_for(n_vals, tgt), read_pos)
                    h = (1 - alpha) * h + alpha * h_snap
            pred = model.dec(h).argmax(-1)
            return (pred == digits_of(traj[:, T])).all(-1).float().mean().item()

        depths = [d for d in (6, 8, 10, 12, 14, 16, 20, 24, 28) if d <= eval_max_depth]
        arm_res = {"none": {d: run("seen_n_seen_x", d, 0, 1.0, "self") for d in depths}}

        # Guard: k=0 must reproduce the stored rollout. A silent pool mismatch here would
        # make every re-projection number below incomparable to the training run.
        st = stored["arms"].get(arm, {}).get("exact_seen_n_seen_x")
        if st:
            ref = {int(k_): v for k_, v in st.items()}
            bad = [(d, arm_res["none"][d], ref[d]) for d in depths
                   if d in ref and abs(arm_res["none"][d] - ref[d]) > 0.05]
            if bad:
                raise AssertionError(f"[{arm}] k=0 does not reproduce stored rollout: {bad[:3]}")
            print(f"  [{arm}] k=0 reproduces stored rollout across {len(depths)} depths", flush=True)

        for mode in ("self", "oracle"):
            for k in PERIODS:
                arm_res[f"{mode}_k{k}_a1.0"] = {
                    d: run("seen_n_seen_x", d, k, 1.0, mode) for d in depths
                }
        for k in (5,):
            for a in ALPHAS:
                arm_res[f"self_k{k}_a{a}"] = {
                    d: run("seen_n_seen_x", d, k, a, "self") for d in depths
                }
        # Does re-projection buy generalisation, or only depth?
        for pname in ("seen_n_heldout_x", "heldout_n"):
            arm_res[f"{pname}_none"] = {d: run(pname, d, 0, 1.0, "self") for d in depths}
            arm_res[f"{pname}_self_k5"] = {d: run(pname, d, 5, 1.0, "self") for d in depths}

        results["arms"][arm] = arm_res
        deep = max(depths)
        print(
            f"  [{arm}] T={deep}: none={arm_res['none'][deep]:.3f} "
            f"self_k5={arm_res['self_k5_a1.0'][deep]:.3f} "
            f"oracle_k5={arm_res['oracle_k5_a1.0'][deep]:.3f}",
            flush=True,
        )

    out_dir = Path(DATA_DIR) / "variable_modulus" / (out_tag or f"reproj_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    volume.commit()
    print(f"\n[saved] {out_path}", flush=True)
    return results
