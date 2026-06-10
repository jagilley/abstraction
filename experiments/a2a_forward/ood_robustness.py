"""OOD robustness: does forward prediction's robustness advantage survive —
or widen under — distribution shift?

The paper's stated open prediction (§6): the forward model approximates the
main model's *computational function*, which is determined by the weights
and invariant to input distribution. The autoencoder approximates the
*activation manifold* at one layer, which is jointly determined by weights
and distribution. If this account is correct, forward prediction's
robustness advantage over the autoencoder (0.413 vs 0.636 ID, baseline
battery) should hold or widen on OOD inputs, where the ID activation
manifold no longer describes the data.

Protocol (identical to baseline battery / Jacobian analysis, extended
across corpora):
  - Perturbation: fixed random unit directions added to the residual
    stream entering block 2 (the injection point), at fixed absolute scale
    s × b1_std where b1_std is the open-loop model's ID activation std —
    so all conditions and corpora receive *identical* perturbations.
  - Conditions tested bare-weights (no injection), matching the battery's
    finding that robustness is in the weights. A forward_inj variant
    (natural mode, injection active) checks the real-time mirror.
  - Empirical Δloss per (condition, corpus, scale, direction), plus
    response norm and cos(response, δ) at post_block3.
  - Hessian trace at the injection point per (condition, corpus) via
    Hutchinson, with the predicted E[ΔL] = ε²/(2d)·tr(H) checked against
    empirical Δloss — does the flatter-landscape mechanism persist OOD?

Corpora: FineWeb val (ID), Wikipedia-EN (near), French / Python code /
open-web-math (far), shuffled FineWeb (structure-destroying control).
Requires cache_ood_tokens (calibration_transfer.py) to have run.

Key output: Δloss ratio vs open_loop per corpus, and the per-direction
paired forward-vs-autoencoder comparison as a function of shift severity.
"""

import json

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder

CONDITIONS = ["open_loop", "forward", "forward_inj", "shifted",
              "random_proj", "autoencoder"]
OOD_TOKEN_DIR = f"{DATA_DIR}/ood_tokens"
OOD_CORPORA = ["wikipedia_en", "french", "code_python", "math"]


@app.function(
    volumes={DATA_DIR: volume},
    gpu="L4",
    timeout=14400,
    memory=32768,
)
def a2a_ood_robustness(
    n_tokens: int = 10_000_000,
    block_size: int = 128,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    batch_size: int = 64,
    predict_from: str = "post_block0",
    predict_to: str = "post_block3",
    inject_after_block: int = 1,
    fwd_n_layer: int = 2,
    fwd_d_head: int = 64,
    fwd_n_head: int = 1,
    fwd_mlp_mult: int = 2,
    seed: int = 42,
    eval_batches: int = 20,
    n_perturb_dirs: int = 16,
    perturb_scales: str = "1.0,2.0",
    n_hutchinson_vectors: int = 10,
    hess_batches: int = 10,
    conditions: str = "",
    corpora: str = "",
):
    import os
    import glob
    import torch
    import torch.nn.functional as F
    import numpy as np
    from a2a_forward.model import GPT
    from a2a_forward.forward_model import (
        TransformerForwardModel, CerebellarGate,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cerebellar_input_block = int(
        predict_from.replace("post_block", "").replace("post_embed", "-1")
    )
    run_conditions = [c.strip() for c in conditions.split(",") if c.strip()] \
        or CONDITIONS
    run_ood = [c.strip() for c in corpora.split(",") if c.strip()] \
        or (OOD_CORPORA + ["shuffled"])
    s_list = [float(s) for s in perturb_scales.split(",")]

    print(f"OOD ROBUSTNESS on {device}")
    print(f"  Conditions: {run_conditions}")
    print(f"  OOD corpora: {run_ood}")
    print(f"  {n_perturb_dirs} dirs x scales {s_list}, "
          f"{eval_batches} batches/corpus; "
          f"tr(H): {n_hutchinson_vectors} vecs x {hess_batches} batches")

    # =========================================================
    # Data (same construction as calibration_transfer)
    # =========================================================
    data_dir = f"{DATA_DIR}/tokens"
    meta = np.load(os.path.join(data_dir, "meta.npy"),
                   allow_pickle=True).item()
    vocab_size = meta["vocab_size"]

    shard_paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npy")))
    all_tokens, total = [], 0
    for path in shard_paths:
        tokens = np.load(path)
        all_tokens.append(tokens)
        total += len(tokens)
        if total >= n_tokens:
            break
    data = np.concatenate(all_tokens)[:n_tokens]
    data = torch.from_numpy(data.astype(np.int64))
    split = int(0.9 * len(data))
    val_data = data[split:]
    half = len(val_data) // 2
    val_hi = val_data[half:]
    print(f"Loaded {len(data):,} ID tokens")

    def random_batches(source, n_batches, gen):
        batches = []
        for _ in range(n_batches):
            idxs = torch.randint(len(source) - block_size - 1,
                                 (batch_size,), generator=gen)
            x = torch.stack(
                [source[i.item():i.item() + block_size] for i in idxs])
            y = torch.stack(
                [source[i.item() + 1:i.item() + block_size + 1]
                 for i in idxs])
            batches.append((x, y))
        return batches

    def window_batches(arr, n_batches):
        toks = torch.from_numpy(np.asarray(arr).astype(np.int64))
        n_windows = n_batches * batch_size
        need = n_windows * (block_size + 1)
        assert len(toks) >= need
        w = toks[:need].reshape(n_windows, block_size + 1)
        return [
            (w[b * batch_size:(b + 1) * batch_size, :-1],
             w[b * batch_size:(b + 1) * batch_size, 1:])
            for b in range(n_batches)
        ]

    id_test_gen = torch.Generator().manual_seed(seed + 11)
    shuffle_gen = torch.Generator().manual_seed(seed + 12)

    corpus_batches = {"id_test": random_batches(val_hi, eval_batches,
                                                id_test_gen)}
    for cname in run_ood:
        if cname == "shuffled":
            perm = torch.randperm(len(val_hi), generator=shuffle_gen)
            corpus_batches["shuffled"] = window_batches(
                val_hi[perm].numpy(), eval_batches)
        else:
            path = os.path.join(OOD_TOKEN_DIR, f"{cname}.npy")
            if not os.path.exists(path):
                print(f"WARNING: {path} missing, skipping {cname}")
                continue
            corpus_batches[cname] = window_batches(np.load(path),
                                                   eval_batches)
    eval_corpora = [c for c in ["id_test"] + run_ood if c in corpus_batches]
    print(f"Eval corpora ready: {eval_corpora}")

    # =========================================================
    # Checkpoints (baseline battery; bare main-model weights, plus
    # fwd_model+gate for the forward_inj natural-mode variant)
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    ckpt_root = (f"{DATA_DIR}/a2a_forward/baseline_battery/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    print(f"Loading checkpoints from {ckpt_root}")

    def load_condition(condition):
        ckpt_cond = "forward" if condition == "forward_inj" else condition
        cdir = os.path.join(ckpt_root, ckpt_cond)
        model = GPT(vocab_size, block_size, n_layer, n_head, n_embd
                    ).to(device)
        model.load_state_dict(
            torch.load(os.path.join(cdir, "model.pt"), map_location=device))
        model.eval()

        inj_fn = None
        if condition == "forward_inj":
            gate = CerebellarGate(n_embd).to(device)
            gate.load_state_dict(
                torch.load(os.path.join(cdir, "gate.pt"),
                           map_location=device))
            gate.eval()
            fwd_model = TransformerForwardModel(
                d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
                n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
                block_size=block_size).to(device)
            fwd_model.load_state_dict(
                torch.load(os.path.join(cdir, "fwd_model.pt"),
                           map_location=device))
            fwd_model.eval()
            inj_fn = lambda cb_src: gate(fwd_model(cb_src))
        return model, inj_fn

    # =========================================================
    # Partial forward helpers (per jacobian_analysis.py)
    # =========================================================
    def compute_h_base(model, x, inj_fn=None):
        """Activation entering block (inject_after_block + 1), after any
        injection. Detached."""
        with torch.no_grad():
            tok_emb = model.transformer.wte(x)
            pos = torch.arange(0, x.size(1), device=x.device)
            pos_emb = model.transformer.wpe(pos)
            h = model.transformer.drop(tok_emb + pos_emb)
            cb_src = None
            for i in range(inject_after_block + 1):
                h = model.transformer.h[i](h)
                if inj_fn is not None and i == cerebellar_input_block:
                    cb_src = h
            if inj_fn is not None and cb_src is not None:
                h = h + inj_fn(cb_src)
        return h

    def from_h(model, h, y):
        """Forward from h through the remaining blocks. Returns
        (mean loss, post_block3 activations)."""
        x = h
        for i in range(inject_after_block + 1, n_layer):
            x = model.transformer.h[i](x)
        b3 = x
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1))
        return loss, b3

    # =========================================================
    # Fixed perturbation directions + scale (ID open-loop std)
    # =========================================================
    rng_perturb = np.random.default_rng(seed + 300)
    perturb_dirs = []
    for _ in range(n_perturb_dirs):
        d = rng_perturb.standard_normal(n_embd).astype(np.float32)
        d /= np.linalg.norm(d)
        perturb_dirs.append(torch.from_numpy(d).to(device))

    model_ol, _ = load_condition("open_loop")
    with torch.no_grad():
        x0, y0 = corpus_batches["id_test"][0]
        _, _, vi0 = model_ol(x0.to(device), y0.to(device),
                             return_intermediates=True)
        b1_std = float(vi0["post_block1"].std())
    del model_ol
    torch.cuda.empty_cache()
    print(f"post_block1 ID activation std (open_loop): {b1_std:.4f}")
    print(f"Perturbation eps: " + ", ".join(
        f"s={s} -> {s * b1_std:.3f}" for s in s_list))

    # =========================================================
    # Main loop
    # =========================================================
    results = {
        "config": {
            "n_tokens": n_tokens, "seed": seed,
            "eval_batches": eval_batches,
            "n_perturb_dirs": n_perturb_dirs, "perturb_scales": s_list,
            "n_hutchinson_vectors": n_hutchinson_vectors,
            "hess_batches": hess_batches,
            "b1_std": b1_std, "conditions": run_conditions,
            "eval_corpora": eval_corpora, "ckpt_root": ckpt_root,
        },
        "base_loss": {},        # cond -> corpus -> float
        "act_std": {},          # cond -> corpus -> post_block1 std
        "perturb": {},          # cond -> corpus -> s -> per-dir lists
        "hessian": {},          # cond -> corpus -> trace mean/std
    }

    rng_hess = np.random.default_rng(seed + 500)
    # Pre-generate Rademacher vectors (shared across conditions/corpora
    # so the estimator is paired)
    hess_vecs = [
        torch.from_numpy(
            2.0 * rng_hess.integers(0, 2, size=n_embd).astype(np.float32)
            - 1.0).to(device)
        for _ in range(n_hutchinson_vectors * hess_batches)
    ]

    for cond in run_conditions:
        print(f"\n{'='*60}")
        print(f"  CONDITION: {cond.upper()}")
        print(f"{'='*60}")
        model, inj_fn = load_condition(cond)
        results["base_loss"][cond] = {}
        results["act_std"][cond] = {}
        results["perturb"][cond] = {}
        results["hessian"][cond] = {}

        for cname in eval_corpora:
            batches = corpus_batches[cname]

            # --- Empirical perturbation ---
            # per (s, dir): accumulate Δloss, response norm, cos over batches
            acc = {s: {"dloss": np.zeros(n_perturb_dirs),
                       "rnorm": np.zeros(n_perturb_dirs),
                       "cos": np.zeros(n_perturb_dirs)}
                   for s in s_list}
            base_losses = []
            b1_stds = []

            with torch.no_grad():
                for x, y in batches:
                    x, y = x.to(device), y.to(device)
                    h_base = compute_h_base(model, x, inj_fn)
                    b1_stds.append(float(h_base.std()))
                    loss_base, b3_base = from_h(model, h_base, y)
                    base_losses.append(float(loss_base))

                    for di, dir_t in enumerate(perturb_dirs):
                        for s in s_list:
                            h_pert = h_base + (s * b1_std) * dir_t
                            loss_p, b3_p = from_h(model, h_pert, y)
                            resp = (b3_p - b3_base).reshape(-1, n_embd)
                            rn = resp.norm(dim=-1)
                            cos = F.cosine_similarity(
                                resp, dir_t.unsqueeze(0).expand_as(resp),
                                dim=-1)
                            acc[s]["dloss"][di] += float(loss_p - loss_base)
                            acc[s]["rnorm"][di] += float(rn.mean())
                            acc[s]["cos"][di] += float(cos.mean())

            nb = len(batches)
            results["base_loss"][cond][cname] = float(np.mean(base_losses))
            results["act_std"][cond][cname] = float(np.mean(b1_stds))
            results["perturb"][cond][cname] = {
                str(s): {
                    "dloss_per_dir": (acc[s]["dloss"] / nb).tolist(),
                    "dloss_mean": float(acc[s]["dloss"].mean() / nb),
                    "dloss_std_dirs": float((acc[s]["dloss"] / nb).std()),
                    "rnorm_mean": float(acc[s]["rnorm"].mean() / nb),
                    "cos_mean": float(acc[s]["cos"].mean() / nb),
                } for s in s_list
            }

            # --- Hessian trace (Hutchinson) ---
            traces = []
            for bi in range(min(hess_batches, len(batches))):
                x, y = batches[bi]
                x, y = x.to(device), y.to(device)
                h_base = compute_h_base(model, x, inj_fn)
                batch_traces = []
                for vi in range(n_hutchinson_vectors):
                    delta = torch.zeros(n_embd, device=device,
                                        requires_grad=True)
                    h_pert = h_base + delta.unsqueeze(0).unsqueeze(0)
                    loss, _ = from_h(model, h_pert, y)
                    g = torch.autograd.grad(loss, delta,
                                            create_graph=True)[0]
                    v = hess_vecs[bi * n_hutchinson_vectors + vi]
                    gv = (g * v).sum()
                    Hv = torch.autograd.grad(gv, delta)[0]
                    batch_traces.append(float((v * Hv).sum()))
                traces.append(np.mean(batch_traces))
            results["hessian"][cond][cname] = {
                "trace_mean": float(np.mean(traces)),
                "trace_std": float(np.std(traces)),
            }

            p2 = results["perturb"][cond][cname][str(s_list[-1])]
            print(f"  [{cname:>13s}] base={np.mean(base_losses):.3f} "
                  f"dloss(s={s_list[-1]})={p2['dloss_mean']:+.4f} "
                  f"||R||={p2['rnorm_mean']:.3f} "
                  f"tr(H)={results['hessian'][cond][cname]['trace_mean']:.4f}")

        del model, inj_fn
        torch.cuda.empty_cache()

    # =========================================================
    # Headline tables
    # =========================================================
    ol = "open_loop"
    severity = sorted(
        eval_corpora,
        key=lambda c: results["base_loss"][ol].get(c, 0.0))

    for s in s_list:
        print(f"\n{'='*72}")
        print(f"  HEADLINE: Δloss ratio vs open_loop (s={s}; lower = more "
              f"robust)")
        print(f"  Corpora ordered by shift severity (open_loop base loss)")
        print(f"{'='*72}")
        header = f"{'condition':>13s}" + "".join(
            f" {c:>13s}" for c in severity)
        print(header)
        for cond in run_conditions:
            row = f"{cond:>13s}"
            for cname in severity:
                dl = results["perturb"][cond][cname][str(s)]["dloss_mean"]
                dl_ol = results["perturb"][ol][cname][str(s)]["dloss_mean"]
                ratio = dl / dl_ol if dl_ol > 0 else float("nan")
                row += f" {ratio:>13.3f}"
            print(row)

    # Paired forward vs autoencoder per corpus (same 16 directions)
    if "forward" in run_conditions and "autoencoder" in run_conditions:
        s = s_list[-1]
        print(f"\n--- Paired forward vs autoencoder, per direction "
              f"(s={s}) ---")
        print(f"{'corpus':>13s} {'OL base':>8s} {'fwd dloss':>10s} "
              f"{'ae dloss':>10s} {'ae/fwd':>7s} {'fwd<ae dirs':>12s}")
        for cname in severity:
            f_dl = np.array(results["perturb"]["forward"][cname][str(s)]
                            ["dloss_per_dir"])
            a_dl = np.array(results["perturb"]["autoencoder"][cname][str(s)]
                            ["dloss_per_dir"])
            wins = int((f_dl < a_dl).sum())
            print(f"{cname:>13s} "
                  f"{results['base_loss'][ol][cname]:>8.3f} "
                  f"{f_dl.mean():>10.4f} {a_dl.mean():>10.4f} "
                  f"{a_dl.mean() / f_dl.mean() if f_dl.mean() > 0 else float('nan'):>7.3f} "
                  f"{wins:>3d}/{n_perturb_dirs}")

    # Hessian trace ratios + prediction check
    print(f"\n--- Hessian trace ratio vs open_loop ---")
    header = f"{'condition':>13s}" + "".join(
        f" {c:>13s}" for c in severity)
    print(header)
    for cond in run_conditions:
        row = f"{cond:>13s}"
        for cname in severity:
            tr = results["hessian"][cond][cname]["trace_mean"]
            tr_ol = results["hessian"][ol][cname]["trace_mean"]
            row += f" {tr / tr_ol if tr_ol != 0 else float('nan'):>13.3f}"
        print(row)

    s = s_list[-1]
    eps = s * b1_std
    print(f"\n--- Hessian prediction check (s={s}): "
          f"predicted E[ΔL]=eps^2/(2d)*tr(H) vs empirical ---")
    print(f"{'condition':>13s} {'corpus':>13s} {'predicted':>10s} "
          f"{'empirical':>10s} {'ratio':>7s}")
    for cond in run_conditions:
        for cname in severity:
            pred = (eps ** 2 / (2 * n_embd)) * \
                results["hessian"][cond][cname]["trace_mean"]
            emp = results["perturb"][cond][cname][str(s)]["dloss_mean"]
            print(f"{cond:>13s} {cname:>13s} {pred:>10.4f} {emp:>10.4f} "
                  f"{pred / emp if emp != 0 else float('nan'):>7.3f}")

    # =========================================================
    # Save
    # =========================================================
    gap_tag = f"{predict_from}_to_{predict_to}"
    save_root = (f"{DATA_DIR}/a2a_forward/ood_robustness/"
                 f"{gap_tag}/inject{inject_after_block}/P_{n_tokens}")
    os.makedirs(save_root, exist_ok=True)
    results_path = os.path.join(save_root, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nAll results saved to {results_path}")
    return results


@app.local_entrypoint()
def main(
    n_tokens: int = 10_000_000,
    eval_batches: int = 20,
    n_perturb_dirs: int = 16,
    perturb_scales: str = "1.0,2.0",
    conditions: str = "",
    corpora: str = "",
):
    results = a2a_ood_robustness.remote(
        n_tokens=n_tokens,
        eval_batches=eval_batches,
        n_perturb_dirs=n_perturb_dirs,
        perturb_scales=perturb_scales,
        conditions=conditions,
        corpora=corpora,
    )
    print("\n=== OOD ROBUSTNESS COMPLETE ===")
    s = str(results["config"]["perturb_scales"][-1])
    ol = "open_loop"
    for cname in results["config"]["eval_corpora"]:
        dl_ol = results["perturb"][ol][cname][s]["dloss_mean"]
        line = f"  {cname:>13s}:"
        for cond in results["config"]["conditions"]:
            if cond == ol:
                continue
            dl = results["perturb"][cond][cname][s]["dloss_mean"]
            line += f" {cond}={dl / dl_ol if dl_ol > 0 else float('nan'):.3f}x"
        print(line)
