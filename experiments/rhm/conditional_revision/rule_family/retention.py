"""Is the discard schedule a dependent variable? Does the family regime make the model
RETAIN resolved structure across a sequence boundary, when the floor regime does not?

Under ONE fixed rule set, a completed RHM sequence is conditionally independent of
everything that follows, so at a sequence boundary the resolved parse predicts nothing
and discarding it is OPTIMAL. That is the incumbent's finding ("does not summarise
resolved structure forward"; past constituents read 0.20 against a Bayes ceiling of ~1.0)
and its tracking appendix's boundary collapse (0.946 -> 0.350 across one step).

Under a rule FAMILY that stops being true. A resolved parse is evidence about which rule
set is active, and the rule set governs every following sequence in the window -- so
carrying resolved structure across the boundary is now WORTH something. The discard
schedule therefore becomes a dependent variable, and the prediction is directional:

    the family arm retains resolved structure across boundaries MORE than its matched
    floor arm, with the excess LARGEST early in the window (rule posterior still broad,
    so each resolved constituent is informative) and SHRINKING late (rules identified,
    further evidence is redundant, discarding becomes optimal again)

That context-depth profile is the signature; a flat family-minus-floor gap would instead
suggest a generic capacity or training difference rather than rule inference.

Two instruments, deliberately different in what they assume:

  RULE PROBE across the boundary (direct). The model has no reason to keep node
    IDENTITIES -- what is worth keeping is a compressed rule-sufficient statistic. So the
    honest instrument for "did it carry the evidence forward" is whether rule identity is
    decodable just after a boundary. The floor arm is pinned at chance by construction and
    is the guard.

  PARSE PROBE across the boundary (conservative lower bound). Decode the PREVIOUS
    sequence's latent nodes from a position after the boundary. If the model compressed
    rather than stored, this reads low even when retention is real -- so it can only
    understate. A "live" probe of the CURRENT position's own ancestors is measured at the
    same positions to normalise for general probe quality there.

    modal run --detach -m rhm.conditional_revision.rule_family.retention::retention \
        --design d2_R64_nF2 --tag ret
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
app = modal.App("rhm-rule-family-retention", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=49152)
def retention(
    design: str = "d2_R64_nF2",
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, family_seed: int = 0,
    k_seqs: int = 8, n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    base_steps: int = 12000, n_eval: int = 6144, eval_seed: int = 999,
    probe_steps: int = 800, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    rule_probe_steps: int = 3000, rule_probe_lr: float = 3e-3,
    read_offsets: str = "1,8", blocks: str = "post_block6,post_block7",
    arms: str = "family,floor", seed: int = 42, tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_latent_loop import _probe_acc
    from rhm.conditional_revision.rule_family.family import make_family, generate_windows
    from rhm.conditional_revision.rule_family.gate_minus1 import DESIGNS

    device = "cuda"
    L, T = depth, s ** depth
    G = k_seqs * T
    key = f"{setting_key(v, s, L, m)}_distinct"
    R, dl, nF, mode = DESIGNS[design]
    offs = [int(x) for x in read_offsets.split(",")]
    blks = blocks.split(",")

    fams = {
        "family": make_family(v, s, L, m, R=R, differ_levels=dl,
                              n_differ_features=nF, seed=family_seed, mode=mode)[0],
        "floor": make_family(v, s, L, m, R=R, differ_levels=[],
                             n_differ_features=nF, seed=family_seed, mode=mode)[0],
    }
    # positions read AFTER each boundary, plus the last position BEFORE it (the reference
    # point where the constituent is still live)
    read_pos = sorted({k * T + r for k in range(1, k_seqs) for r in offs}
                      | {k * T - 1 for k in range(1, k_seqs)})
    read_pos = [p for p in read_pos if p <= G - 2]
    pidx = {p: i for i, p in enumerate(read_pos)}
    print(f"design={design} R={R}  read positions: {read_pos}", flush=True)

    out = {"config": {"design": design, "R": R, "k_seqs": k_seqs, "offsets": offs,
                      "blocks": blks, "read_positions": read_pos, "tag": tag},
           "arms": {}}

    for arm in arms.split(","):
        ckpt = (f"{DATA_DIR}/{key}/rule_family/{design}_K{k_seqs}_{arm}_"
                f"{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_seed{seed}.pt")
        if not os.path.exists(ckpt):
            print(f"  MISSING {ckpt} -- skipping {arm}", flush=True)
            continue
        print(f"\n===== ARM {arm} =====", flush=True)
        model = GPT(v, G, n_layer, n_head, n_embd).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        sq, rid, lf = generate_windows(fams[arm], n_eval, k_seqs, seed=eval_seed + 11)
        X = torch.from_numpy(sq.reshape(n_eval, G)).to(device)
        gsel = torch.tensor(read_pos, device=device)
        acts = {b: torch.empty(n_eval, len(read_pos), n_embd) for b in blks}
        with torch.no_grad():
            for i in range(0, n_eval, 64):
                _, _, inter = model(X[i:i + 64, :-1].contiguous(),
                                    return_intermediates=True)
                for b in blks:
                    acts[b][i:i + 64] = inter[b][:, gsel, :].float().cpu()

        # ---- direct instrument: rule identity across the boundary ----
        rule_by_pos = {}
        if R > 1:
            A = torch.cat([acts[b] for b in blks], dim=2).to(device)   # (n, P, 2D)
            n_tr = int(0.7 * n_eval)
            y = torch.from_numpy(rid).long().to(device)
            mu = A[:n_tr].reshape(-1, A.shape[2]).mean(0, keepdim=True)
            sd = A[:n_tr].reshape(-1, A.shape[2]).std(0, keepdim=True) + 1e-6
            A = (A - mu) / sd
            clf = nn.Linear(A.shape[2], R).to(device)
            o = torch.optim.Adam(clf.parameters(), lr=rule_probe_lr, weight_decay=1e-4)
            g = torch.Generator().manual_seed(seed + 5)
            for _ in range(rule_probe_steps):
                sel = torch.randint(0, n_tr, (256,), generator=g).to(device)
                pos = torch.randint(0, len(read_pos), (256,), generator=g).to(device)
                l = F.cross_entropy(clf(A[sel, pos]), y[sel])
                o.zero_grad(); l.backward(); o.step()
            with torch.no_grad():
                hit = (clf(A[n_tr:]).argmax(-1) == y[n_tr:, None]).float().mean(0)
            rule_by_pos = {str(p): float(hit[pidx[p]]) for p in read_pos}
            del A
            print(f"  rule probe by position (chance {1 / R:.4f}):")
            for k in range(1, k_seqs):
                pre = rule_by_pos.get(str(k * T - 1), float("nan"))
                post = [rule_by_pos.get(str(k * T + r), float("nan")) for r in offs]
                print(f"    boundary {k}: pre {pre:.4f} -> post "
                      + " ".join(f"{x:.4f}" for x in post), flush=True)

        # ---- conservative lower bound: previous sequence's parse, read after boundary
        past, live = {}, {}
        for k in range(1, k_seqs):
            for r in offs:
                gpos = k * T + r
                if gpos > G - 2:
                    continue
                col = pidx[gpos]
                rowp, rowl = {}, {}
                for ell in range(L):
                    anc_prev = (T - 1) // (s ** (L - ell))     # prev seq's last node
                    anc_cur = r // (s ** (L - ell))            # this position's own node
                    yp = torch.from_numpy(
                        lf[ell][:, k - 1, anc_prev].astype(np.int64)).to(device)
                    yl = torch.from_numpy(
                        lf[ell][:, k, anc_cur].astype(np.int64)).to(device)
                    bp = bl = 0.0
                    for b in blks:
                        Ab = acts[b][:, col, :].to(device)
                        bp = max(bp,
                                 _probe_acc(Ab, yp, v, device, probe_steps, probe_lr),
                                 _probe_acc(Ab, yp, v, device, mlp_steps, probe_lr,
                                            hidden=mlp_hidden))
                        bl = max(bl,
                                 _probe_acc(Ab, yl, v, device, probe_steps, probe_lr),
                                 _probe_acc(Ab, yl, v, device, mlp_steps, probe_lr,
                                            hidden=mlp_hidden))
                    rowp[f"d{L - ell}"] = float(bp)
                    rowl[f"d{L - ell}"] = float(bl)
                past[f"k{k}_r{r}"] = rowp
                live[f"k{k}_r{r}"] = rowl
                print(f"  seq{k} +{r}: PAST(seq{k - 1}) "
                      + " ".join(f"{kk}{vv:.3f}" for kk, vv in rowp.items())
                      + "  |  LIVE " + " ".join(f"{kk}{vv:.3f}"
                                                for kk, vv in rowl.items()), flush=True)

        out["arms"][arm] = {"rule_by_pos": rule_by_pos, "past": past, "live": live}
        del model, acts
        torch.cuda.empty_cache()

    # ---- the contrast ----
    if "family" in out["arms"] and "floor" in out["arms"]:
        fam, flo = out["arms"]["family"], out["arms"]["floor"]
        print(f"\n{'=' * 78}\nRETENTION CONTRAST: family - floor, PAST-sequence parse "
              f"decodability\n{'=' * 78}")
        print(f"  (prediction: positive, LARGEST at low k, shrinking as rules resolve)")
        keys = sorted(fam["past"], key=lambda z: (int(z.split('_')[0][1:]),
                                                  int(z.split('_')[1][1:])))
        lv = [f"d{i}" for i in range(1, depth + 1)]
        print(f"  {'cell':<10}" + "".join(f"{x:>9}" for x in lv) + f"{'mean':>9}")
        contrast = {}
        for kk in keys:
            d = [fam["past"][kk][x] - flo["past"][kk][x] for x in lv]
            contrast[kk] = {"per_level": d, "mean": float(np.mean(d))}
            print(f"  {kk:<10}" + "".join(f"{x:>+9.3f}" for x in d)
                  + f"{np.mean(d):>+9.3f}")
        out["contrast_past"] = contrast
        if fam["rule_by_pos"]:
            print(f"\n  RULE-PROBE retention across each boundary "
                  f"(direct instrument; chance {1 / R:.4f})")
            print(f"  {'boundary':<10}{'family pre':>12}{'family post':>13}"
                  f"{'floor post':>12}{'fam-floor':>11}")
            rr = {}
            for k in range(1, k_seqs):
                pre = fam["rule_by_pos"].get(str(k * T - 1), float("nan"))
                po = fam["rule_by_pos"].get(str(k * T + offs[0]), float("nan"))
                fo = flo["rule_by_pos"].get(str(k * T + offs[0]), float("nan"))
                rr[k] = {"family_pre": pre, "family_post": po, "floor_post": fo,
                         "excess": po - fo}
                print(f"  {k:<10}{pre:>12.4f}{po:>13.4f}{fo:>12.4f}"
                      f"{po - fo:>+11.4f}")
            out["contrast_rule"] = rr
    d = f"{DATA_DIR}/{key}/rule_family"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/retention_{design}{'_' + tag if tag else ''}_seed{seed}.json",
              "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    return out
