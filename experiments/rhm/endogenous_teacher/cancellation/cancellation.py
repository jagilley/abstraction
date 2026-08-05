"""Cancellation on RHM: route the endogenous signal through the forward path.

Cut 1 (../endogenous_teacher.py) collapsed the FM residual to a scalar and used it to
weight the NTP loss. It was a null, and over-determined: EMOTION_INJECTION already
showed scalar evaluative modulation is near-inert, and the "self-knowledge is
directional, not scalar" belief node (confidence: strong) already showed the residual
NORM is the part that carries nothing (vector probe ΔR² +0.18 vs scalar +0.03; causal
steering along the norm direction has zero effect).

So the signal was never the problem -- Gate 0 measured 89% of the residual's variance
orthogonal to token surprisal, and anti-localised to it. The CHANNEL was the problem.
This cut fixes the channel: subtract the forecast and propagate the residual, so the
deviation is the tensor downstream computes on rather than a number multiplying a loss.

    summation (the whole a2a arc):  x = x + gate.p            at the inject block
    cancellation (efference copy):  x = x - gate.p  ... x = x + gate.p  at the re-add

Arms (token target throughout -- no oracle aux; lam_local = 0 so this isolates the
INJECTION CHANNEL, since RHM_LATENT_LOOP Exp 3 showed the offloading is lam-independent
and the local loss is a separate consolidating effect that inverts SK on its own):

    ntp        open loop                     reference: root 0.819, fresh SK 0.463
    cl_sum     summation injection           reference: root 0.453, fresh SK 0.398
    cl_cancel  cancellation injection        <- the new cell

Regime is m2 (v16 s2 L6 m2, occupancy 0.125), NOT m4: at m2 token-NTP reaches the root,
so there is something for the injection to strip, and the summation pathology
("injection -> offloads", root 0.82 -> ~0.45) is at its largest and already measured.

THE HEADLINE TEST is Payoff 1 from ideas/efference_copy_cancellation.md -- modular vs
entangled dependency, via a fresh-FM swap. That doc records it as inconclusive on the
looped ViT "because the low-rank loop's FM is near-unique, so a 'different but
equally-good' FM barely exists to swap in -- too weak a perturbation to test
entanglement. Needs a gauge-free substrate (RHM / language)." This is that substrate.

Note on which metric is the right one. Standalone-removal is the WRONG readout for
cancellation and the idea doc says so: subtracting a forecast that is then absent is a
large distribution shift, so "can't run without it" is expected, not a bug --
"cancellation makes dependency the design ... the pathology to guard against is no
longer 'can't run without it' but instability." The discriminating readout is whether
downstream is entangled with ITS OWN FM's idiosyncrasies (swap degrades badly) or owns
only the residual (swap degrades gracefully).

Predictions (registered):
  P1  fresh-FM swap: cl_cancel degrades LESS than cl_sum (modular vs entangled).
  P2  fresh-FM SK: cl_sum reproduces the token-CL inversion (dSK ~ -0.06 at lam=0,
      driven by FM-specific offloading); cl_cancel does NOT invert, because the
      division of labour is structural rather than incentivised.
  P3  mechanism: cos(delta_final, inj) is large and positive under summation and
      collapses toward 0 under cancellation (the ViT read +0.89 -> +0.095). If P3
      fails the wiring did not engage and P1/P2 are uninterpretable.
  P4  standalone root: cl_sum strips it (0.82 -> ~0.45). No directional prediction for
      cl_cancel -- see the note above; reported, not tested.

Run:
  modal run --detach -m rhm.endogenous_teacher.cancellation.cancellation::cancellation
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
app = modal.App("rhm-cancellation", image=image)

ARMS = ["ntp", "cl_sum", "cl_cancel"]


def tb_key(v, s, L, m):
    return f"{setting_key(v, s, L, m)}_distinct"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def cancellation(
    # DGP -- m2: token-NTP reaches the root, so the injection has something to strip
    v: int = 16, s: int = 2, depth: int = 6, m: int = 2, rule_seed: int = 0,
    # Model / FM -- identical to RHM_LATENT_LOOP so its reference lines transfer
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    predict_from: str = "post_block0", predict_to: str = "post_block6",
    inject_after_block: int = 1, readd_block: int = -1,   # -1 -> predict_to's block
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8,
    fwd_mlp_mult: float = 1.0, ug_hidden: int = 64,
    # Training
    n_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4, fwd_lr: float = 1e-3,
    weight_decay: float = 0.01,
    pool_size: int = 200000, data_seed: int = 7,
    arms: str = "ntp,cl_sum,cl_cancel",
    # Measurement
    n_eval_sequences: int = 8000, eval_seed: int = 999,
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800, sk_probe_steps: int = 500,
    swap_n: int = 3, fresh_fm_steps: int = 3000,
    log_interval: int = 2000, seed: int = 42, tag: str = "",
):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc, _sk_probes
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda"
    L, T = depth, s ** depth
    key = tb_key(v, s, L, m)
    cib = int(predict_from.replace("post_block", ""))
    tob = int(predict_to.replace("post_block", ""))
    rb = tob if readd_block < 0 else readd_block
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    for a in arm_list:
        assert a in ARMS, f"unknown arm {a!r}; known {ARMS}"
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)

    print(f"{'=' * 76}\nRHM CANCELLATION  {key}  {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  occupancy m/v^(s-1)={m / v ** (s - 1):.3f}  T={T}  chance={1/v:.4f}")
    print(f"  FM {fwd_n_layer}L/{fwd_n_head}H/{fwd_d_head}d  {predict_from}->{predict_to}")
    print(f"  inject after block {inject_after_block}; cancel re-adds after block {rb}")
    print(f"  arms {arm_list}  n_steps={n_steps}  lam_local=0 (injection channel only)")
    print(f"{'=' * 76}", flush=True)

    # ---------------- data ----------------
    print(f"Generating pool ({pool_size:,} seqs)...", flush=True)
    pool_seqs, _, _ = _generate_with_traces(rules, pool_size, data_seed)
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    def get_ntp_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (batch_size,), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    eval_seqs, eval_lf, _ = _generate_with_traces(rules, n_eval_sequences, eval_seed)
    eval_x = torch.from_numpy(eval_seqs.astype(np.int64)).to(device)
    last_anc = {ell: (T - 1) // (s ** (L - ell)) for ell in range(L)}
    y_level = {ell: torch.from_numpy(eval_lf[ell][:, last_anc[ell]].astype(np.int64)).to(device)
               for ell in range(L)}
    block_names = ["post_embed"] + [f"post_block{i}" for i in range(n_layer)]

    # ---------------- modules ----------------
    class UnifiedGate(nn.Module):
        """Identical to RHM_LATENT_LOOP's gate: zero-init, so at step 0 the injection
        is exactly zero and cl_sum == cl_cancel == ntp. A strict generalization."""
        def __init__(self, d_model, d_hidden):
            super().__init__()
            self.gate_net = nn.Sequential(nn.Linear(2 * d_model, d_hidden), nn.GELU(),
                                          nn.Linear(d_hidden, d_model))
            self.projection = nn.Linear(d_model, d_model)
            for layer in (self.gate_net[-1], self.projection):
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

        def forward(self, activations, fwd_pred):
            gw = torch.sigmoid(self.gate_net(torch.cat([activations, fwd_pred], dim=-1)))
            return gw * self.projection(fwd_pred), gw

    def make_fm():
        return TransformerForwardModel(
            d_model=n_embd, d_head=fwd_d_head, n_head=fwd_n_head,
            n_layer=fwd_n_layer, mlp_mult=fwd_mlp_mult, block_size=T).to(device)

    def cb_kwargs(mode):
        """Forward-pass kwargs for a given arm. 'add' is the historical path."""
        if mode == "cancel":
            return dict(cerebellar_input_block=cib, cerebellar_inject_block=inject_after_block,
                        cerebellar_mode="cancel", cerebellar_readd_block=rb)
        return dict(cerebellar_input_block=cib, cerebellar_inject_block=inject_after_block,
                    cerebellar_mode="add")

    def fm_target_key(mode):
        """Under cancellation the FM must be trained against the RESTORED read-out
        (post_block{k}_eff), not the deviation stream, or the loop is degenerate."""
        return f"{predict_to}_eff" if mode == "cancel" else predict_to

    # ---------------- measurement ----------------
    def _run(model, x, fm=None, gate=None, mode="add", targets=None):
        if fm is None:
            return model(x, targets, return_intermediates=True)
        def cb(act):
            inj, _ = gate(act, fm(act))
            return inj
        return model(x, targets, return_intermediates=True, cerebellar_fn=cb,
                     **cb_kwargs(mode))

    def per_level(model, fm=None, gate=None, mode="add"):
        model.eval()
        if fm is not None:
            fm.eval(); gate.eval()
        acts = {b: [] for b in block_names}
        with torch.no_grad():
            for i in range(0, eval_x.shape[0], 256):
                _, _, inter = _run(model, eval_x[i:i + 256], fm, gate, mode)
                for b in block_names:
                    acts[b].append(inter[b][:, -1, :].float())
        acts = {b: torch.cat(vs) for b, vs in acts.items()}
        out = {}
        for ell in range(L):
            out[f"d{L - ell}"] = max(  # repo convention: ell=0 is the ROOT -> d6
                max(_probe_acc(acts[b], y_level[ell], v, device, probe_steps, probe_lr),
                    _probe_acc(acts[b], y_level[ell], v, device, mlp_steps, probe_lr,
                               hidden=mlp_hidden))
                for b in block_names)
        return out

    def eval_val(model, fm=None, gate=None, mode="add"):
        model.eval()
        if fm is not None:
            fm.eval(); gate.eval()
        gen = torch.Generator().manual_seed(eval_seed + 5)
        tot = 0.0
        with torch.no_grad():
            for _ in range(20):
                x, y = get_ntp_batch(gen)
                _, loss, _ = _run(model, x, fm, gate, mode, targets=y)
                tot += loss.item()
        return tot / 20

    def injection_mechanism(model, fm, gate, mode):
        """cos(delta_final, inj): how the injection moves the final residual stream.
        ViT reference: summation +0.89, cancellation +0.095 (directional cancellation)."""
        model.eval(); fm.eval(); gate.eval()
        gen = torch.Generator().manual_seed(eval_seed + 13)
        cos, gnorm = [], []
        last = f"post_block{n_layer - 1}"
        with torch.no_grad():
            for _ in range(10):
                x, _ = get_ntp_batch(gen)
                _, _, base_i = model(x, return_intermediates=True)
                inj, gw = gate(base_i[predict_from], fm(base_i[predict_from]))
                _, _, inj_i = _run(model, x, fm, gate, mode)
                delta = (inj_i[last] - base_i[last]).reshape(-1, n_embd)
                cos.append(F.cosine_similarity(delta, inj.reshape(-1, n_embd),
                                               dim=-1).mean().item())
                gnorm.append(inj.norm(dim=-1).mean().item())
        return float(np.mean(cos)), float(np.mean(gnorm))

    def train_fresh_fm(model, gate, co_fm, mode, fm_seed):
        """A fresh FM fitted to the FROZEN closed-loop model's own activations --
        a different-but-equally-good forecaster, which is exactly what the ViT swap
        lacked. Trained on-policy with the co-trained FM driving the loop."""
        torch.manual_seed(fm_seed)
        fresh = make_fm()
        opt = torch.optim.AdamW(fresh.parameters(), lr=fwd_lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(fm_seed + 77)
        model.eval(); co_fm.eval(); gate.eval()
        tkey = fm_target_key(mode)
        for _ in range(fresh_fm_steps):
            x, _ = get_ntp_batch(gen)
            with torch.no_grad():
                _, _, inter = _run(model, x, co_fm, gate, mode)
            loss = F.mse_loss(fresh(inter[predict_from]), inter[tkey])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            cos = F.cosine_similarity(fresh(inter[predict_from]).reshape(-1, n_embd),
                                      inter[tkey].reshape(-1, n_embd), dim=-1).mean().item()
        return fresh, float(cos)

    # ---------------- shared init ----------------
    torch.manual_seed(seed)
    init_model = GPT(v, T, n_layer, n_head, n_embd).to(device)
    init_state = {k: t.cpu().clone() for k, t in init_model.state_dict().items()}
    del init_model
    torch.cuda.empty_cache()

    results = {"config": {"v": v, "s": s, "L": L, "m": m, "n_layer": n_layer,
                          "n_embd": n_embd, "predict_from": predict_from,
                          "predict_to": predict_to, "inject_after": inject_after_block,
                          "readd_block": rb, "n_steps": n_steps, "seed": seed,
                          "arms": arm_list, "tag": tag},
               "arms": {}}

    for arm in arm_list:
        mode = "cancel" if arm == "cl_cancel" else "add"
        closed = arm != "ntp"
        print(f"\n{'=' * 76}\nARM {arm}  (mode={mode}, closed={closed})\n{'=' * 76}",
              flush=True)
        torch.manual_seed(seed)
        model = GPT(v, T, n_layer, n_head, n_embd).to(device)
        model.load_state_dict({k: t.to(device) for k, t in init_state.items()})
        fm = make_fm()
        gate = UnifiedGate(n_embd, ug_hidden).to(device)
        params = list(model.parameters()) + (list(gate.parameters()) if closed else [])
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        opt_fwd = torch.optim.AdamW(fm.parameters(), lr=fwd_lr, weight_decay=weight_decay)
        gen = torch.Generator().manual_seed(seed + 1)
        tkey = fm_target_key(mode)

        for step in range(n_steps):
            model.train(); fm.train(); gate.train()
            x, y = get_ntp_batch(gen)
            pred_cache = {}

            if closed:
                def cb(act):
                    p = fm(act.detach())
                    pred_cache["p"] = p
                    return gate(act, p.detach())[0]
                _, loss, inter = model(x, y, return_intermediates=True,
                                       cerebellar_fn=cb, **cb_kwargs(mode))
            else:
                _, loss, inter = model(x, y, return_intermediates=True)
                pred_cache["p"] = fm(inter[predict_from].detach())

            opt.zero_grad(); loss.backward(); opt.step()
            fwd_loss = F.mse_loss(pred_cache["p"], inter[tkey].detach())
            opt_fwd.zero_grad(); fwd_loss.backward(); opt_fwd.step()

            if step % log_interval == 0 or step == n_steps - 1:
                print(f"  {arm} {step:6d}  ntp {loss.item():.4f}  fm_mse {fwd_loss.item():.4f}",
                      flush=True)

        # --- measurements ---
        r = {}
        r["val_standalone"] = eval_val(model)
        r["levels_standalone"] = per_level(model)
        print(f"\n  {arm}: standalone val {r['val_standalone']:.4f}")
        print(f"  {arm}: standalone levels {r['levels_standalone']}", flush=True)

        if closed:
            r["val_injected"] = eval_val(model, fm, gate, mode)
            r["levels_injected"] = per_level(model, fm, gate, mode)
            r["dependency"] = r["val_standalone"] - r["val_injected"]
            cos_inj, inj_norm = injection_mechanism(model, fm, gate, mode)
            r["cos_delta_inj"] = cos_inj
            r["inj_norm"] = inj_norm
            print(f"  {arm}: injected val {r['val_injected']:.4f}  "
                  f"dependency {r['dependency']:+.4f}")
            print(f"  {arm}: MECHANISM cos(delta,inj) {cos_inj:+.4f}  "
                  f"|inj| {inj_norm:.4f}   (ViT ref: sum +0.89, cancel +0.095)",
                  flush=True)

            # --- Payoff 1: the fresh-FM swap (the ViT-blocked test) ---
            swaps = []
            for j in range(swap_n):
                fresh, fcos = train_fresh_fm(model, gate, fm, mode, seed + 500 + j)
                swaps.append({"fm_cos": fcos,
                              "val": eval_val(model, fresh, gate, mode)})
                del fresh; torch.cuda.empty_cache()
            torch.manual_seed(seed + 999)
            rand_fm = make_fm()
            r["swap"] = {
                "cotrained_val": r["val_injected"],
                "fresh": swaps,
                "fresh_val_mean": float(np.mean([x["val"] for x in swaps])),
                "fresh_fm_cos_mean": float(np.mean([x["fm_cos"] for x in swaps])),
                "random_fm_val": eval_val(model, rand_fm, gate, mode),
            }
            r["swap"]["cost"] = r["swap"]["fresh_val_mean"] - r["val_injected"]
            del rand_fm; torch.cuda.empty_cache()
            print(f"  {arm}: SWAP co-trained {r['val_injected']:.4f} -> fresh "
                  f"{r['swap']['fresh_val_mean']:.4f} (cost {r['swap']['cost']:+.4f}); "
                  f"random-FM {r['swap']['random_fm_val']:.4f}; "
                  f"fresh cos {r['swap']['fresh_fm_cos_mean']:.4f}", flush=True)

        # --- fresh-FM self-knowledge (the token-CL inversion test) ---
        # _sk_probes runs STANDALONE forward passes, so the SK forward model must be
        # fitted on standalone activations too -- for every arm, identically. This is
        # what RHM_LATENT_LOOP's "fresh SK" measures, and keeping it identical is what
        # makes ntp .463 / cl_sum .398 valid reference points.
        torch.manual_seed(seed + 4242)
        fresh_sk = make_fm()
        optf = torch.optim.AdamW(fresh_sk.parameters(), lr=fwd_lr,
                                 weight_decay=weight_decay)
        g2 = torch.Generator().manual_seed(seed + 4319)
        model.eval()
        for _ in range(fresh_fm_steps):
            xx, _ = get_ntp_batch(g2)
            with torch.no_grad():
                _, _, ii = model(xx, return_intermediates=True)
            lf = F.mse_loss(fresh_sk(ii[predict_from]), ii[predict_to])
            optf.zero_grad(); lf.backward(); optf.step()
        r["sk_fresh"] = _sk_probes(model, fresh_sk, eval_x, predict_from, predict_to,
                                   n_embd, n_layer, 256, device, sk_probe_steps,
                                   seed, f"{arm}/fresh")
        if closed:
            r["sk_cotrained"] = _sk_probes(model, fm, eval_x, predict_from, predict_to,
                                           n_embd, n_layer, 256, device, sk_probe_steps,
                                           seed, f"{arm}/cotrained")
        results["arms"][arm] = r
        del model, fm, gate, fresh_sk
        torch.cuda.empty_cache()

    # ---------------- summary ----------------
    print(f"\n{'=' * 76}\nSUMMARY  (m{m};  refs: ntp root .819 SK .463 | "
          f"cl_sum root .453 SK .398)\n{'=' * 76}")
    hdr = (f"{'arm':<11}{'val_sa':>9}{'val_inj':>9}{'dep':>8}" +
           "".join(f"{f'd{i+1}':>7}" for i in range(L)) +
           f"{'SK_b7':>8}{'cos':>8}{'swap':>8}")
    print(hdr)
    for arm, r in results["arms"].items():
        lv = r["levels_standalone"]
        print(f"{arm:<11}{r['val_standalone']:>9.4f}"
              f"{r.get('val_injected', float('nan')):>9.4f}"
              f"{r.get('dependency', float('nan')):>8.4f}" +
              "".join(f"{lv[f'd{i+1}']:>7.3f}" for i in range(L)) +
              f"{r['sk_fresh'][f'post_block{n_layer-1}']:>8.3f}"
              f"{r.get('cos_delta_inj', float('nan')):>8.3f}"
              f"{r.get('swap', {}).get('cost', float('nan')):>8.4f}")

    out_dir = f"{DATA_DIR}/{key}/cancellation"
    os.makedirs(out_dir, exist_ok=True)
    name = f"results{'_' + tag if tag else ''}_seed{seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}", flush=True)
    return results
