"""Where in a sequence does this substrate's belief probe actually read?

Gate 0's per-level recovery came back far below this regime's reference lines
(d1 0.979 / d3 0.836 / root 0.088) even in the FLOOR arm, which is supposed to BE the
incumbent single-rule-set substrate. Before treating that as "the family regime costs
belief depth", it has to be separated from a measurement artifact, because there is an
obvious candidate.

The incumbent trained on FLAT windows at random offsets, so position carried no phase
information and the model could never know it was standing at a constituent boundary.
Here windows are K whole aligned sequences, so position determines phase exactly. At the
last position of a sequence the next token belongs to a FRESH, conditionally independent
sequence, so nothing about the completed parse predicts anything further -- and a model
that can see the boundary coming is free to discard it. That is the incumbent's own
finding ("the model's per-position representation is close to a next-token-sufficient
statistic ... does not summarise resolved structure forward") plus its tracking appendix
(the probe collapses 0.946 -> 0.350 across a single boundary step while the truth becomes
MORE determined), sharpened by the fact that phase is now predictable.

So this sweeps the read position across a sequence and reports per-level recovery at
each. If recovery is high mid-sequence and collapses only at the boundary, the substrate
is fine and Gate 0's number was read at the worst possible place -- which is a fact Gate 1
needs, since the belief probe is its instrument.

    modal run --detach -m rhm.conditional_revision.rule_family.probe_positions::sweep \
        --design d2_R64_nF2 --tag pp
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-rule-family-probepos", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=32768)
def sweep(
    design: str = "d2_R64_nF2",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, family_seed: int = 0,
    k_seqs: int = 8, n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    base_steps: int = 12000, n_eval: int = 4096, eval_seed: int = 999,
    probe_steps: int = 800, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    arms: str = "family,floor", seed: int = 42, tag: str = "",
):
    import numpy as np
    import torch
    from rhm.model import GPT
    from rhm.rhm_latent_loop import _probe_acc
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    from rhm.conditional_revision.rule_family.gate_minus1 import DESIGNS

    device = "cuda"
    L, T = depth, s ** depth
    G = k_seqs * T
    key = f"{setting_key(v, s, L, m)}_distinct"
    R, dl, nF, mode = DESIGNS[design]

    fams = {
        "family": make_family(v, s, L, m, R=R, differ_levels=dl,
                              n_differ_features=nF, seed=family_seed, mode=mode)[0],
        "floor": make_family(v, s, L, m, R=R, differ_levels=[],
                             n_differ_features=nF, seed=family_seed, mode=mode)[0],
    }
    # read positions inside the SECOND-TO-LAST sequence (all trained positions), spanning
    # a full constituent so the boundary sits at the right-hand end
    probe_seq = k_seqs - 2
    within = [31, 47, 55, 59, 61, 62, 63]
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]
    out = {"config": {"design": design, "k_seqs": k_seqs, "within_positions": within,
                      "probe_seq": probe_seq, "R": R, "tag": tag}, "arms": {}}

    for arm in arms.split(","):
        ckpt = (f"{DATA_DIR}/{key}/rule_family/{design}_K{k_seqs}_{arm}_"
                f"{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  MISSING {ckpt} -- skipping {arm}", flush=True)
            continue
        model = GPT(v, G, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        sq, _, lf = generate_windows(fams[arm], n_eval, k_seqs, seed=eval_seed + 11)
        X = torch.from_numpy(sq.reshape(n_eval, G)).to(device)
        acts = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, n_eval, 64):
                _, _, inter = model(X[i:i + 64, :-1].contiguous(),
                                    return_intermediates=True)
                for b in block_names:
                    acts[b].append(inter[b].float().cpu())
        acts = {b: torch.cat(vs) for b, vs in acts.items()}

        arm_out = {}
        for p_in in within:
            gpos = probe_seq * T + p_in
            row = {}
            for ell in range(L):
                anc = p_in // (s ** (L - ell))          # ancestor of the CURRENT token
                y = torch.from_numpy(
                    lf[ell][:, probe_seq, anc].astype(np.int64)).to(device)
                best = 0.0
                for b in block_names:
                    Ab = acts[b][:, gpos, :].to(device)
                    best = max(best,
                               _probe_acc(Ab, y, v, device, probe_steps, probe_lr),
                               _probe_acc(Ab, y, v, device, mlp_steps, probe_lr,
                                          hidden=mlp_hidden))
                row[f"d{L - ell}"] = float(best)
            arm_out[f"p{p_in}"] = row
            print(f"  {arm} within-seq pos {p_in:3d} (global {gpos}): "
                  + "  ".join(f"{k} {vv:.3f}" for k, vv in row.items()), flush=True)
        out["arms"][arm] = arm_out
        del model, acts
        torch.cuda.empty_cache()

    print(f"\n  reference (incumbent, flat-window training, aligned eval): "
          f"d1 0.979  d3 0.836  d6 0.088")
    d = f"{DATA_DIR}/{key}/rule_family"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/probepos_{design}{'_' + tag if tag else ''}_seed{seed}.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out
