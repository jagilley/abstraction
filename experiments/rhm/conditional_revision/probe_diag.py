"""Probe diagnostic for Gate A: can a probe read the model's belief at all?

Gate A's `M` is a KL between probe-decoded posteriors, so it is only a
measurement of the model's belief if the probe is near its own ceiling. The first
Gates-A/B run's shared 63-node probe reached all-position accuracy 0.068-0.182
per level, against a BAYES ceiling of 0.371-0.614 computed exactly by BP on the
same sequences. That gap is far too large to read `M` through, and it is an
optimisation failure rather than an impossible task -- hence this file.

Three things are measured, in increasing order of what they would let us conclude:

  1. `reference`  -- the repo's own single-node probe (`_probe_acc`, best over
     blocks, last position, last ancestor per level). This must reproduce
     endogenous_teacher / RHM_LATENT_LOOP's d1 0.979 / d3 0.836 / d6 0.088. If it
     does not, the checkpoint or the activations are wrong and nothing else here
     matters.
  2. `bayes`      -- the exact ceiling for every variant's task, from oracle.py.
     Reported next to each variant so "the probe is weak" and "the task is hard"
     cannot be confused.
  3. `variants`   -- the shared all-node probe under the changes most likely to
     be responsible: learning rate, capacity, per-level heads, and per-position
     standardisation (a residual stream's scale grows with t, so pooled
     standardisation leaves the probe absorbing a position-dependent scale).

Run:
  modal run --detach -m rhm.conditional_revision.probe_diag::probe_diag --tag diag
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
app = modal.App("rhm-cr-probe-diag", image=image)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=32768)
def probe_diag(
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    block: str = "post_block6", base_steps: int = 12000,
    n_train: int = 3000, n_test: int = 1500,
    eval_seed: int = 999, seed: int = 42, tag: str = "", base_ckpt: str = "",
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from rhm.conditional_revision.oracle import prefix_beliefs, _rule_ids

    device = "cuda"
    L, T = depth, s ** depth
    key = f"{setting_key(v, s, L, m)}_distinct"
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    n_nodes = (s ** L - 1) // (s - 1)
    node_level = np.concatenate([np.full(s ** d, d) for d in range(L)])
    node_off = {d: (s ** d - 1) // (s - 1) for d in range(L)}

    ckpt = base_ckpt or (f"{DATA_DIR}/{key}/conditional_revision/"
                         f"base_{n_layer}L{n_head}H{n_embd}D_steps{base_steps}_"
                         f"seed{seed}.pt")
    torch.manual_seed(seed)
    model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded frozen base <- {ckpt}", flush=True)

    n_all = n_train + n_test
    seqs, lf, _ = _generate_with_traces(rules, n_all, eval_seed)
    x_all = torch.from_numpy(seqs.astype(np.int64))
    node_y = np.concatenate([lf[d] for d in range(L)], axis=1)
    node_y_t = torch.from_numpy(node_y.astype(np.int64))
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    @torch.no_grad()
    def acts_for(names):
        out = {b: [] for b in names}
        for i in range(0, n_all, 256):
            _, _, inter = model(x_all[i:i + 256].to(device), return_intermediates=True)
            for b in names:
                out[b].append(inter[b].float().cpu())
        return {b: torch.cat(vs) for b, vs in out.items()}

    print("collecting activations...", flush=True)
    A = acts_for(block_names)

    # Node classes at each position. A level-d node spans s^(L-d) leaves and the
    # level partitions them, so relative to position t every node is exactly one
    # of: CURRENT (contains t or t+1 -- the ancestor chain), PAST (span entirely
    # observed, already resolved), FUTURE (span starts after t+1, so its posterior
    # is still ~the prior). This is the split that decides whether "the model only
    # holds the ancestor chain" is a real claim about discarding resolved
    # structure, or just an artefact of averaging in future nodes nobody could
    # know. Bayes says PAST is nearly free to decode and FUTURE is nearly chance.
    CUR, PAST, FUT = 0, 1, 2
    ncls = np.full((T, n_nodes), -1, dtype=np.int64)
    for d in range(L):
        span = s ** (L - d)
        for j in range(s ** d):
            hi = (j + 1) * span - 1
            for t in range(T):
                col = node_off[d] + j
                if j == t // span or j == min(t + 1, T - 1) // span:
                    ncls[t, col] = CUR
                elif hi <= t:
                    ncls[t, col] = PAST
                else:
                    ncls[t, col] = FUT
    cls_name = {CUR: "current(chain)", PAST: "past(resolved)", FUT: "future(unseen)"}

    results = {"config": {"block": block, "n_train": n_train, "n_test": n_test,
                          "ckpt": ckpt, "tag": tag, "seed": seed}}

    # ---- 1. the repo's own readout, as an anchor on the checkpoint ----------
    last_anc = {d: (T - 1) // (s ** (L - d)) for d in range(L)}
    ref = {}
    for d in range(L):
        y = torch.from_numpy(node_y[:, node_off[d] + last_anc[d]]).to(device)
        ref[f"d{L - d}"] = max(
            max(_probe_acc(A[b][:, -1, :].to(device), y, v, device, 600, 1e-2),
                _probe_acc(A[b][:, -1, :].to(device), y, v, device, 800, 1e-2,
                           hidden=128))
            for b in block_names)
    results["reference_single_node_best_block"] = ref
    print(f"\n  repo-style reference (last position, last ancestor, best block):")
    print(f"    {ref}")
    print(f"    expected d1 0.979  d3 0.836  d6 0.088 "
          f"(endogenous_teacher / RHM_LATENT_LOOP)", flush=True)

    # ---- 2. the exact Bayes ceiling for the all-node task -------------------
    print(f"\n  computing Bayes ceiling on the test split...", flush=True)
    ids = [_rule_ids(rules, d) for d in range(L)]
    te = seqs[n_train:]
    bayes_acc = np.zeros((L, T))
    bayes_ce = np.zeros((L, T))
    bayes_cls_hit = np.zeros((L, 3)); bayes_cls_n = np.zeros((L, 3))
    for t in range(T):
        nodes, _, _ = prefix_beliefs(rules, te, t + 1, L - 1, ids)
        for d in range(L):
            p = nodes[d]
            bayes_acc[d, t] = p.max(-1).mean()
            for c in (0, 1, 2):
                jj = np.where(ncls[t, node_off[d]:node_off[d] + s ** d] == c)[0]
                if jj.size:
                    bayes_cls_hit[d, c] += p[:, jj].max(-1).sum()
                    bayes_cls_n[d, c] += jj.size * p.shape[0]
            truth = node_y[n_train:, node_off[d]:node_off[d] + s ** d]
            bayes_ce[d, t] = -np.log(np.clip(
                np.take_along_axis(p, truth[:, :, None], axis=2), 1e-30, None)).mean()
    results["bayes_ceiling_by_class"] = {
        f"d{L - d}": {cls_name[c]: float(bayes_cls_hit[d, c] / max(bayes_cls_n[d, c], 1))
                      for c in (0, 1, 2)} for d in range(L)}
    print("    bayes ceiling BY NODE CLASS:")
    for d in range(L):
        print(f"      d{L - d}: " + "  ".join(
            f"{cls_name[c]} {bayes_cls_hit[d, c] / max(bayes_cls_n[d, c], 1):.3f}"
            for c in (0, 1, 2)))
    results["bayes_ceiling"] = {
        f"d{L - d}": {"acc_all_pos": float(bayes_acc[d].mean()),
                      "acc_last_pos": float(bayes_acc[d, -1]),
                      "ce_all_pos": float(bayes_ce[d].mean())} for d in range(L)}
    print(f"    bayes all-pos acc "
          f"{ {f'd{L-d}': round(float(bayes_acc[d].mean()), 3) for d in range(L)} }")
    print(f"    bayes all-pos CE  "
          f"{ {f'd{L-d}': round(float(bayes_ce[d].mean()), 3) for d in range(L)} }",
          flush=True)

    # ---- 3. the all-node probe under the candidate fixes --------------------
    Xtr_raw = A[block][:n_train]
    Xte_raw = A[block][n_train:]
    Ytr = node_y_t[:n_train]
    Yte = node_y[n_train:]

    def standardise(per_position):
        if per_position:                       # (1, T, 1) stats -- keeps position scale
            mu = Xtr_raw.mean(dim=(0,), keepdim=True)
            sd = Xtr_raw.std(dim=(0,), keepdim=True) + 1e-6
        else:
            mu = Xtr_raw.reshape(-1, n_embd).mean(0).view(1, 1, n_embd)
            sd = Xtr_raw.reshape(-1, n_embd).std(0).view(1, 1, n_embd) + 1e-6
        return ((Xtr_raw - mu) / sd), ((Xte_raw - mu) / sd)

    class Shared(nn.Module):
        def __init__(self, hidden, per_level):
            super().__init__()
            self.per_level = per_level
            if per_level:                      # one head per LEVEL, shared trunk
                self.trunk = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU())
                self.heads = nn.ModuleList(
                    [nn.Linear(hidden, (s ** d) * v) for d in range(L)])
            else:
                self.net = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(),
                                         nn.Linear(hidden, n_nodes * v))

        def forward(self, x):
            if not self.per_level:
                return self.net(x).view(*x.shape[:-1], n_nodes, v)
            h = self.trunk(x)
            return torch.cat([hd(h).view(*x.shape[:-1], s ** d, v)
                              for d, hd in enumerate(self.heads)], dim=-2)

    def run_variant(name, lr, hidden, steps, per_level, per_position, batch=512):
        Xtr, Xte = standardise(per_position)
        Xf = Xtr.reshape(-1, n_embd)
        Yf = Ytr.unsqueeze(1).expand(-1, T, -1).reshape(-1, n_nodes)
        probe = Shared(hidden, per_level).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        g = torch.Generator().manual_seed(seed + 7)
        for st in range(steps):
            ix = torch.randint(0, Xf.shape[0], (batch,), generator=g)
            out = probe(Xf[ix].to(device))
            loss = F.cross_entropy(out.reshape(-1, v), Yf[ix].to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if st % max(steps // 4, 1) == 0:
                print(f"      {name} {st:6d}  train ce {loss.item():.4f}", flush=True)
        probe.eval()
        accs_all, accs_last, ces = {}, {}, {}
        with torch.no_grad():
            pred, ce_sum = [], []
            for i in range(0, Xte.shape[0], 64):
                lg = probe(Xte[i:i + 64].to(device))
                pred.append(lg.argmax(-1).cpu())
                ce_sum.append(F.cross_entropy(
                    lg.reshape(-1, v),
                    node_y_t[n_train:][i:i + 64].to(device).unsqueeze(1)
                    .expand(-1, T, -1).reshape(-1),
                    reduction="none").view(-1, T, n_nodes).cpu())
            P = torch.cat(pred).numpy()
            C = torch.cat(ce_sum).numpy()
        by_cls = {}
        hit = P == Yte[:, None, :]
        for d in range(L):
            j0, j1 = node_off[d], node_off[d] + s ** d
            by_cls[f"d{L - d}"] = {}
            for c in (0, 1, 2):
                mk = ncls[:, j0:j1] == c
                by_cls[f"d{L - d}"][cls_name[c]] = (
                    float(hit[:, :, j0:j1][:, mk].mean()) if mk.any() else float("nan"))
        for d in range(L):
            j0, j1 = node_off[d], node_off[d] + s ** d
            accs_all[f"d{L - d}"] = float((P[:, :, j0:j1] == Yte[:, None, j0:j1]).mean())
            accs_last[f"d{L - d}"] = float(
                (P[:, -1, j0:j1] == Yte[:, j0:j1]).mean())
            ces[f"d{L - d}"] = float(C[:, :, j0:j1].mean())
        return {"acc_by_node_class": by_cls,
                "lr": lr, "hidden": hidden, "steps": steps, "per_level": per_level,
                "per_position_standardise": per_position,
                "acc_all_pos": accs_all, "acc_last_pos": accs_last,
                "ce_all_pos": ces}

    variants = [
        ("as_run  lr1e-3 h1024 pooled", 1e-3, 1024, 8000, False, False),
        ("lr3e-3  h1024 pooled",        3e-3, 1024, 8000, False, False),
        ("lr1e-2  h1024 pooled",        1e-2, 1024, 8000, False, False),
        ("lr3e-3  h1024 PER-POSITION",  3e-3, 1024, 8000, False, True),
        ("lr3e-3  h2048 PER-POSITION",  3e-3, 2048, 16000, False, True),
        ("lr3e-3  h2048 PER-POS +heads", 3e-3, 2048, 16000, True, True),
    ]
    results["variants"] = {}
    print(f"\n  all-node probe variants (block {block}):", flush=True)
    for nm, lr, hid, stp, pl, pp in variants:
        print(f"\n    --- {nm} ---", flush=True)
        r = run_variant(nm, lr, hid, stp, pl, pp)
        results["variants"][nm] = r
        print(f"      all-pos acc {  {k: round(x, 3) for k, x in r['acc_all_pos'].items()} }")
        print(f"      last-pos acc { {k: round(x, 3) for k, x in r['acc_last_pos'].items()} }")
        print(f"      BY NODE CLASS (probe vs bayes):")
        for d in range(L):
            nm = f"d{L - d}"
            print("        " + nm + ": " + "  ".join(
                f"{c} {r['acc_by_node_class'][nm][c]:.3f}/"
                f"{results['bayes_ceiling_by_class'][nm][c]:.3f}"
                for c in ("current(chain)", "past(resolved)", "future(unseen)")),
                flush=True)

    print(f"\n{'=' * 78}\nSUMMARY -- all-position accuracy vs the exact Bayes ceiling"
          f"\n{'=' * 78}")
    hdr = f"  {'variant':<32}" + "".join(f"{f'd{L-d}':>8}" for d in range(L))
    print(hdr)
    print(f"  {'BAYES CEILING':<32}" +
          "".join(f"{bayes_acc[d].mean():>8.3f}" for d in range(L)))
    for nm in results["variants"]:
        r = results["variants"][nm]
        print(f"  {nm:<32}" +
              "".join(f"{r['acc_all_pos'][f'd{L-d}']:>8.3f}" for d in range(L)))

    out_dir = f"{DATA_DIR}/{key}/conditional_revision"
    os.makedirs(out_dir, exist_ok=True)
    name = f"probe_diag{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
