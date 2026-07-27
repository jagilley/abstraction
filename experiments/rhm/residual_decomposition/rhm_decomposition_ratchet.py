"""Part C: do the claimed FLOWS across the partition actually happen?

beliefs/dimensionality_expansion.md claims two flows over wake-sleep cycles:

  "Compression drains R_res -> R_comp (R_act unchanged). On a *fixed* dataset
   R_act is capped, so draining runs to the wall: R_res -> 0."
  "Novelty raises R_act by injecting fresh directions into R_res from the top."

and a regime table that says the healthy signature is R_act UP, R_comp UP,
R_res persistently > 0.

Parts A+B (rhm_decomposition_audit.py) test the decomposition statically. This
tests it dynamically, as a 2x2 factorial so the compression effect and the
novelty effect are separable rather than confounded:

                  | fixed DGP        | novel DGP (new rules each cycle)
    --------------|------------------|--------------------------------
    OL (no loop)  | OL_FIXED         | OL_NOVEL
    WS (wake-sleep)| WS_FIXED        | WS_NOVEL

  OL   = FM trained fresh each cycle but never injected or distilled. Isolates
         what ordinary continued training alone does to the triple.
  WS   = FM injected during wake, then distilled into the base model during
         sleep, then re-pointed with a fresh FM. This is "compression".
  novel = each cycle draws from a NEW rule set at the SAME (v, s, L, m), so
         DGP complexity is held fixed and only content changes. That matters:
         RHM_LATENT_LOOP_README established that i.i.d. sample-novelty is
         already saturated on RHM, so fresh samples from the same rules are a
         known null -- rule novelty is the real intervention.

MEASUREMENT CONTROL. The belief's own caveat is that R_act is an activation
rank and therefore data-dependent, "mixing base-weight capacity with how much
this data excites it". So every cycle's decomposition is measured twice:

  probe_fixed   -- on a FIXED held-out probe corpus (cycle-0 rules), identical
                   across all cycles and all four conditions. R_act can only
                   move here if the model's computation changed. This is the
                   canonical readout.
  probe_current -- on the cycle's own data. Reported alongside so the
                   data-dependence the belief warns about is visible, not
                   hidden.

Architecture follows ratchet/rhm_ratchet.py so results stay comparable:
6L/6H/192D main, 2L/1H/24d FM, post_block0 -> post_block3, inject after
block 1, UnifiedGate with zero-init projection.

Reproduction:
    cd experiments/
    modal run --detach -m rhm.residual_decomposition.rhm_decomposition_ratchet::decomposition_ratchet
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

app = modal.App("rhm-decomposition-ratchet", image=image)

CONDITIONS = ["OL_FIXED", "OL_NOVEL", "WS_FIXED", "WS_NOVEL"]


def _corpus_path(v, s, L, m, rule_seed):
    """Cycle-0 rules reuse the canonical corpus; later rule sets get their own."""
    key = setting_key(v, s, L, m)
    if rule_seed == 0:
        return f"{DATA_DIR}/{key}/corpus.npy"
    return f"{DATA_DIR}/{key}/rule_variants/rs{rule_seed}/corpus.npy"


def _ensure_corpus_variant(v, s, L, m, n_tokens, rule_seed):
    """Generate a corpus from a specific rule set if it isn't on the volume."""
    import numpy as np
    from rhm.rhm_data import make_corpus

    path = _corpus_path(v, s, L, m, rule_seed)
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] >= n_tokens:
            return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  generating corpus rule_seed={rule_seed} ({n_tokens:,} tokens)")
    corpus, _, _ = make_corpus(v, s, L, m, n_tokens,
                               rule_seed=rule_seed, seq_seed=rule_seed + 1)
    np.save(path, corpus)
    volume.commit()
    return path


@app.function(
    volumes={DATA_DIR: volume},
    gpu="A10G",
    timeout=14400,
    memory=32768,
)
def run_condition(
    condition: str = "WS_FIXED",
    v: int = 8, s: int = 2, depth: int = 6, m: int = 4,
    n_tokens: int = 5_000_000,
    n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    fwd_n_layer: int = 2, fwd_n_head: int = 1, fwd_d_head: int = 24,
    fwd_mlp_mult: float = 2.0,
    predict_from: str = "post_block0", predict_to: str = "post_block3",
    inject_after_block: int = 1,
    n_cycles: int = 6,
    warmup_steps: int = 2000,
    wake_steps: int = 2000, distill_steps: int = 1000, repoint_steps: int = 1500,
    batch_size: int = 64, lr: float = 3e-4, fwd_lr: float = 1e-3,
    distill_lr: float = 1e-4, distill_alpha: float = 0.5,
    ug_hidden: int = 128,
    seed: int = 42, n_probe_batches: int = 30,
):
    """One arm of the 2x2. Returns the per-cycle decomposition trajectory."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from rhm.model import GPT
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.residual_decomposition.decomposition import full_decomposition

    assert condition in CONDITIONS, condition
    is_ws = condition.startswith("WS")
    is_novel = condition.endswith("NOVEL")

    torch.manual_seed(seed)
    np.random.seed(seed)

    L = depth
    key = setting_key(v, s, L, m)
    block_size = s ** L
    device = "cuda"

    volume.reload()

    # --- corpora: cycle 0 always uses rule_seed 0; NOVEL rotates thereafter ---
    rule_seeds = [0] + ([1000 + c for c in range(1, n_cycles)] if is_novel
                        else [0] * (n_cycles - 1))
    for rs in sorted(set(rule_seeds)):
        _ensure_corpus_variant(v, s, L, m, n_tokens, rs)

    def load_split(rule_seed):
        arr = np.load(_corpus_path(v, s, L, m, rule_seed), mmap_mode="r")
        arr = torch.from_numpy(np.array(arr[:n_tokens]).astype(np.int64))
        sp = int(0.9 * len(arr))
        return arr[:sp], arr[sp:]

    cycle_data = {rs: load_split(rs) for rs in sorted(set(rule_seeds))}

    def get_batch(split_data, bs=None):
        bs = bs or batch_size
        ix = torch.randint(len(split_data) - block_size - 1, (bs,))
        x = torch.stack([split_data[i:i + block_size] for i in ix])
        y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # Fixed probe set: cycle-0 rules, held-out split, IDENTICAL across cycles and
    # conditions. R_act can only move here if the model's computation changed.
    probe_rng = torch.Generator().manual_seed(12345)
    probe_val = cycle_data[0][1]
    probe_ix = torch.randint(len(probe_val) - block_size - 1,
                             (n_probe_batches * batch_size,), generator=probe_rng)
    probe_x = torch.stack([probe_val[i:i + block_size] for i in probe_ix])

    class UnifiedGate(nn.Module):
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_model))
            self.projection = nn.Linear(d_model, d_model)
            for mod in (self.gate_net[-1], self.projection):
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)

        def forward(self, activations, fwd_pred):
            gate_w = torch.sigmoid(self.gate_net(
                torch.cat([activations, fwd_pred], dim=-1)))
            return gate_w * self.projection(fwd_pred), gate_w

    def make_gpt():
        return GPT(v, block_size, n_layer, n_head, n_embd).to(device)

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult,
            block_size=block_size).to(device)

    def eval_loss(mdl, split_data, n=10):
        mdl.eval()
        with torch.no_grad():
            return float(np.mean([float(mdl(*get_batch(split_data))[1])
                                  for _ in range(n)]))

    def train_fresh_fm(mdl, train_split, steps, fm_seed):
        """Re-point: a fresh FM on the CURRENT model, so each cycle's residual
        reflects the model as it now is rather than a stale predictor."""
        mdl.eval()
        torch.manual_seed(fm_seed)
        fm_new = make_fm()
        o = torch.optim.AdamW(fm_new.parameters(), lr=fwd_lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            o, T_max=steps, eta_min=fwd_lr * 0.01)
        for _ in range(steps):
            fm_new.train()
            x, y = get_batch(train_split)
            with torch.no_grad():
                _, _, inter = mdl(x, y, return_intermediates=True)
            loss = F.mse_loss(fm_new(inter[predict_from]), inter[predict_to])
            o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm_new.parameters(), 1.0)
            o.step()
            sch.step()
        return fm_new

    def decompose(mdl, fm_cur, x_source, label):
        mdl.eval()
        fm_cur.eval()
        A_l, P_l = [], []
        with torch.no_grad():
            for i in range(0, len(x_source), batch_size):
                xb = x_source[i:i + batch_size].to(device)
                if xb.shape[0] < 2:
                    continue
                _, _, vi = mdl(xb, return_intermediates=True)
                A_l.append(vi[predict_to].reshape(-1, n_embd).float().cpu().numpy())
                P_l.append(fm_cur(vi[predict_from]).reshape(-1, n_embd)
                           .float().cpu().numpy())
        d = full_decomposition(np.concatenate(A_l), np.concatenate(P_l), seed=seed)
        b, nv, g, rp = d["basic"], d["naive"], d["geometry"], d["repaired"]
        print(f"    [{label}] rel_res={b['relative_residual']:.4f} "
              f"| naive R_act={nv['R_act']:.1f} R_comp={nv['R_comp']:.1f} "
              f"R_res={nv['R_res']:.1f} ratio={nv['subadditivity_ratio']:.2f} "
              f"| align={g['alignment_index']:+.3f} "
              f"| rep R_act_H={rp['R_act_H']:.1f} R_res_H={rp['R_res_H']:.1f} "
              f"frontier={rp['frontier_mass']:.4f}")
        return d

    # ------------------------------------------------------------------
    print(f"=== {condition} | {key} | {n_layer}L/{n_head}H/{n_embd}D | "
          f"{n_cycles} cycles ===")
    print(f"  rule_seeds per cycle: {rule_seeds}")

    model = make_gpt()
    opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Warmup on cycle-0 data so all conditions start from the same base state.
    print(f"  [WARMUP] {warmup_steps} steps")
    train0 = cycle_data[0][0]
    for step in range(warmup_steps):
        model.train()
        x, y = get_batch(train0)
        _, loss = model(x, y)
        opt_main.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_main.step()
        if step % 500 == 0:
            print(f"    step {step:5d}: train={float(loss):.4f}")

    fm = train_fresh_fm(model, train0, repoint_steps, seed + 1)
    ugate = UnifiedGate(n_embd, ug_hidden).to(device)

    history = []
    d0 = decompose(model, fm, probe_x, "cycle0/probe_fixed")
    history.append({"cycle": 0, "rule_seed": 0,
                    "val_loss_probe": eval_loss(model, cycle_data[0][1]),
                    "probe_fixed": d0, "probe_current": d0})

    for cycle in range(1, n_cycles + 1):
        rs = rule_seeds[min(cycle - 1, len(rule_seeds) - 1)]
        train_split, val_split = cycle_data[rs]
        print(f"\n  --- {condition} cycle {cycle}/{n_cycles} (rule_seed={rs}) ---")

        # === WAKE: co-train model + FM (+ gate, when the loop is closed) ===
        opt_main = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=0.01)
        opt_gate = torch.optim.AdamW(ugate.parameters(), lr=fwd_lr,
                                     weight_decay=0.01)
        for step in range(wake_steps):
            model.train()
            fm.train()
            x, y = get_batch(train_split)

            if is_ws:
                def cb(act):
                    inj, _ = ugate(act, fm(act))
                    return inj
                _, ce = model(x, y, cerebellar_fn=cb,
                              cerebellar_input_block=0,
                              cerebellar_inject_block=inject_after_block)
            else:
                _, ce = model(x, y)

            opt_main.zero_grad()
            opt_gate.zero_grad()
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_main.step()
            if is_ws:
                opt_gate.step()

            # FM always trains (OL trains it too -- it is just never used).
            with torch.no_grad():
                _, _, inter = model(x, y, return_intermediates=True)
            fm_loss = F.mse_loss(fm(inter[predict_from]), inter[predict_to])
            opt_fwd.zero_grad()
            fm_loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            opt_fwd.step()

            if step % 500 == 0:
                print(f"    [wake {step:5d}] ce={float(ce):.4f} "
                      f"fm={float(fm_loss):.6f}")

        # === SLEEP: distill the FM-augmented teacher into a plain student ===
        if is_ws:
            teacher, t_fm = make_gpt(), make_fm()
            teacher.load_state_dict(model.state_dict())
            t_fm.load_state_dict(fm.state_dict())
            t_gate = UnifiedGate(n_embd, ug_hidden).to(device)
            t_gate.load_state_dict(ugate.state_dict())
            for mod in (teacher, t_fm, t_gate):
                mod.eval()
                for p in mod.parameters():
                    p.requires_grad = False

            student = make_gpt()
            student.load_state_dict(model.state_dict())
            opt_s = torch.optim.AdamW(student.parameters(), lr=distill_lr,
                                      weight_decay=0.01)
            for step in range(distill_steps):
                student.train()
                x, y = get_batch(train_split)
                with torch.no_grad():
                    def t_cb(act):
                        inj, _ = t_gate(act, t_fm(act))
                        return inj
                    t_logits, _, _ = teacher(
                        x, return_intermediates=True, cerebellar_fn=t_cb,
                        cerebellar_input_block=0,
                        cerebellar_inject_block=inject_after_block)
                s_logits, ce_loss = student(x, y)
                kl = F.kl_div(F.log_softmax(s_logits, dim=-1),
                              F.softmax(t_logits, dim=-1), reduction="batchmean")
                loss = distill_alpha * kl + (1 - distill_alpha) * ce_loss
                opt_s.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt_s.step()
                if step % 500 == 0:
                    print(f"    [sleep {step:5d}] kl={float(kl):.4f} "
                          f"ce={float(ce_loss):.4f}")
            model.load_state_dict(student.state_dict())
            del teacher, t_fm, t_gate, student, opt_s
            torch.cuda.empty_cache()

        # === RE-POINT: fresh FM on the updated model ===
        fm = train_fresh_fm(model, train_split, repoint_steps,
                            seed + 100 * cycle)

        # === MEASURE ===
        d_fixed = decompose(model, fm, probe_x, f"cycle{cycle}/probe_fixed")
        cur_ix = torch.randint(len(val_split) - block_size - 1,
                               (n_probe_batches * batch_size,))
        cur_x = torch.stack([val_split[i:i + block_size] for i in cur_ix])
        d_cur = decompose(model, fm, cur_x, f"cycle{cycle}/probe_current")

        history.append({
            "cycle": cycle, "rule_seed": rs,
            "val_loss_probe": eval_loss(model, cycle_data[0][1]),
            "val_loss_current": eval_loss(model, val_split),
            "probe_fixed": d_fixed, "probe_current": d_cur,
        })

    save_dir = f"{DATA_DIR}/residual_decomposition_ratchet/{condition}"
    os.makedirs(save_dir, exist_ok=True)
    result = {
        "condition": condition, "setting": key,
        "is_ws": is_ws, "is_novel": is_novel,
        "rule_seeds": rule_seeds,
        "config": {
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
            "fwd": {"n_layer": fwd_n_layer, "n_head": fwd_n_head,
                    "d_head": fwd_d_head, "mlp_mult": fwd_mlp_mult},
            "predict_from": predict_from, "predict_to": predict_to,
            "inject_after_block": inject_after_block,
            "n_cycles": n_cycles, "warmup_steps": warmup_steps,
            "wake_steps": wake_steps, "distill_steps": distill_steps,
            "repoint_steps": repoint_steps, "seed": seed,
        },
        "history": history,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n  {condition} done -> {save_dir}")
    return result


def _row(h, which):
    d = h[which]
    return (d["naive"], d["geometry"], d["repaired"], d["basic"])


@app.function(volumes={DATA_DIR: volume}, timeout=21600, memory=8192)
def decomposition_ratchet(n_cycles: int = 6, seed: int = 42):
    """Run all four arms of the 2x2 in parallel and tabulate the trajectories."""
    handles = [(c, run_condition.spawn(condition=c, n_cycles=n_cycles, seed=seed))
               for c in CONDITIONS]
    results = {}
    for c, h in handles:
        results[c] = h.get()

    for which in ("probe_fixed", "probe_current"):
        print("\n" + "=" * 116)
        print(f"PER-CYCLE TRAJECTORY  [{which}]")
        if which == "probe_fixed":
            print("  (fixed held-out probe, identical across cycles+conditions: "
                  "R_act moves only if the model's computation moved)")
        print("=" * 116)
        for c in CONDITIONS:
            if c not in results:
                continue
            print(f"\n  --- {c} ---")
            print(f"  {'cyc':>3} {'val':>7} | {'R_act':>6} {'R_comp':>7} "
                  f"{'R_res':>6} {'ratio':>6} | {'align':>7} | "
                  f"{'R_act_H':>8} {'R_comp_H':>8} {'R_res_H':>8} "
                  f"{'frontier':>9} | {'rel_res':>8}")
            for h in results[c]["history"]:
                if which not in h:
                    continue
                nv, g, rp, b = _row(h, which)
                vl = h.get("val_loss_probe", float("nan"))
                print(f"  {h['cycle']:>3} {vl:>7.4f} | {nv['R_act']:>6.1f} "
                      f"{nv['R_comp']:>7.1f} {nv['R_res']:>6.1f} "
                      f"{nv['subadditivity_ratio']:>6.2f} | "
                      f"{g['alignment_index']:>+7.3f} | {rp['R_act_H']:>8.1f} "
                      f"{rp['R_comp_H']:>8.1f} {rp['R_res_H']:>8.1f} "
                      f"{rp['frontier_mass']:>9.4f} | {b['relative_residual']:>8.4f}")

    print("\n" + "=" * 116)
    print("CLAIM CHECK (probe_fixed, cycle 0 -> final)")
    print("=" * 116)
    print(f"  {'condition':>10} | {'dR_act':>8} {'dR_comp':>8} {'dR_res':>8} "
          f"(naive) | {'dR_act_H':>9} {'dR_comp_H':>10} {'dR_res_H':>9} "
          f"{'dfrontier':>10} (repaired)")
    for c in CONDITIONS:
        if c not in results:
            continue
        hs = [h for h in results[c]["history"] if "probe_fixed" in h]
        n0, _, r0, _ = _row(hs[0], "probe_fixed")
        n1, _, r1, _ = _row(hs[-1], "probe_fixed")
        print(f"  {c:>10} | {n1['R_act'] - n0['R_act']:>+8.1f} "
              f"{n1['R_comp'] - n0['R_comp']:>+8.1f} "
              f"{n1['R_res'] - n0['R_res']:>+8.1f}          | "
              f"{r1['R_act_H'] - r0['R_act_H']:>+9.2f} "
              f"{r1['R_comp_H'] - r0['R_comp_H']:>+10.2f} "
              f"{r1['R_res_H'] - r0['R_res_H']:>+9.2f} "
              f"{r1['frontier_mass'] - r0['frontier_mass']:>+10.4f}")
    print("\n  Belief predicts: WS_FIXED -> R_res drains toward 0 with R_act flat;")
    print("  WS_NOVEL -> R_act rises and R_res stays refilled.")

    save_dir = f"{DATA_DIR}/residual_decomposition_ratchet"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved to {save_dir}/summary.json")
    return results


@app.function(volumes={DATA_DIR: volume}, gpu="A10G", timeout=3600, memory=32768)
def smoke():
    """Fast path check on one WS arm and one OL arm.

    modal run -m rhm.residual_decomposition.rhm_decomposition_ratchet::smoke
    """
    kw = dict(n_tokens=300_000, n_cycles=2, warmup_steps=200, wake_steps=200,
              distill_steps=100, repoint_steps=200, n_probe_batches=6)
    for c in ("WS_NOVEL", "OL_FIXED"):
        r = run_condition.local(condition=c, **kw)
        print(f"  {c}: {len(r['history'])} cycles logged")
    print("SMOKE OK")
