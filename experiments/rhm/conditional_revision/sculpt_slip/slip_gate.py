"""The conditioning gap on a CONTROL substrate: can the model tell "my dynamics
model is wrong" from "the world slipped"?

Step 1 of the sculpt-slip cut. See ../README.md for the reading-axis result this
transfers, and ../../RHM_SCULPTING_README.md for the substrate.

WHY THIS SUBSTRATE
------------------
Sculpting's block FM is `FM(z, k) -> dz` on the MOVE axis, and the task was
designed so that consequences are deterministic (`sculpting_control_task.md`,
ingredient 2: "Content-independent, deterministic consequences"). So `(z, k)`
fully determine `dz`, the FM's information set is COMPLETE, and its residual can
only mean "I lacked capacity" -- epistemically the same class as the depth FM,
just on a different axis. A temporal-vs-depth A/B here would compare two
zero-aleatoric forecasters and isolate the axis, which the reading-axis gates say
is not the operative variable.

Stage 3d (`rhm_sculpt_latent_stoch.py`) added a SLIPPERY ACTUATOR: an edit lands
as intended with prob 1-q, else slips to a uniformly random feature. But its
Design 1 deliberately reuses the deterministic FM and ranks by intended outcome,
so the FM still never experiences a gap. That README names the follow-up twice:

    "A Design 2 that retrains the FM on slippery targets -- so it predicts the
     true k-dependent expectation E[dz|z,k] rather than the intended outcome"

Design 2 IS the conditioning gap on this substrate. Under slip, `(z,k)` no longer
determine `dz`; the slip realisation is exogenous. That is exactly the cerebellar
arity-2 structure -- it has the command, it lacks the world's noise realisation.

Three things this has that the reading-axis cut structurally cannot: a behavioural
readout (beam success, Step 2), an EXACT BINARY aleatoric label (did the actuator
slip -- we own the RNG), and `q` as a settable knob, which turns precision from an
unmeasurable into a controlled variable.

WHAT STEP 1 MEASURES
--------------------
Families are ground truth: transitions where the actuator slipped vs where it did
not. Signals are readouts of the FM residual `r = dz_realised - FM(z,k)`:

  resid_norm        ||r||, the scalar incumbent
  resid_norm_acted  ||r|| restricted to the acted block
  dir_probe         a linear probe on r / ||r|| -- norm-free BY CONSTRUCTION
  raw_probe         a linear probe on r
  cos_to_intent     cos(r, dz_intended - FM(z,k)), a labelled reference (it uses
                    the counterfactual deterministic outcome, so it is an upper
                    reference, not a usable signal)
  dir_probe_shuf    the same probe trained on shuffled labels -- guard, ~0.5

scored raw and MATCHED on ||r|| with atom-aware strata (../gates_ab.py), so the
scalar incumbent is pinned at ~0.5 and only direction can win. Every block reports
the guard AUCs; if they are not ~0.5 the strata leak and the row is inflated.

REGISTERED PREDICTIONS
----------------------
  * Design 1 FM (predicts the INTENDED outcome): its residual is ~0 when nothing
    slipped and large when something did, so ||r|| alone should separate well and
    direction should add little.
  * Design 2 FM (predicts E[dz|z,k]): a NON-slipped transition also has a nonzero
    residual -- it deviates from the mean toward the intended outcome -- so the two
    families sit on opposite sides of the mean at similar radius. ||r|| should
    separate poorly and the DIRECTIONAL readout should separate well. This is the
    reading-axis finding (0.606 scalar vs 0.690 directional at d2, 0.005 vs 0.150
    in Gate A) transferred to a substrate where it can be acted on.

KILL: if `dir_probe` does not beat `resid_norm` under ||r||-matching for the
Design 2 FM at any q, the slip is not directionally identifiable from the residual,
Step 2's precision operator has nothing to act on, and we stop.

Run:
  modal run -m rhm.conditional_revision.sculpt_slip.slip_gate::slip_gate1 --quick
  modal run --detach -m rhm.conditional_revision.sculpt_slip.slip_gate::slip_gate1 --tag step1
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_active_query import _count_parameters
from rhm.rhm_edit_control import _train_edit_controller
from rhm.rhm_generative_planner import _build_generator, _regenerate, _train_generator
from rhm.rhm_latent_planner import _build_value_head, _region_index, _train_value_mc
from rhm.rhm_sculpt_planner import _collect_value_sculpt, _corrupt, _sample_pool
from rhm.rhm_sculpt_latent import (
    _build_block_fm,
    _build_rich_controller,
    _train_block_fm,
)

app = modal.App("rhm-sculpt-slip", image=image)


# ---------------------------------------------------------------------------
# slippery actuator, with the slip flag returned
# ---------------------------------------------------------------------------

def _regen_flagged(generator, x, region_blocks, canon, *, block_size, slip, v, gen):
    """`_stoch_regenerate` (rhm_sculpt_latent_stoch.py) but it also returns WHICH
    acted blocks slipped. Same dynamics; the flag is the exact aleatoric label that
    the reading-axis cut had to reconstruct from belief propagation.

    At slip == 0 this is byte-identical to the deterministic `_regenerate`, which
    `_assert_slip0_matches_deterministic` checks.
    """
    import torch
    x_det = _regenerate(generator, x, region_blocks, canon, None,
                        block_size=block_size, sample=False)
    flags = torch.zeros(region_blocks.shape, dtype=torch.bool, device=x_det.device)
    if slip <= 0.0:
        return x_det, flags
    batch = x_det.shape[0]
    positions = (region_blocks[:, :, None] * block_size
                 + torch.arange(block_size, device=x_det.device))
    flat_positions = positions.reshape(batch, -1)
    do_slip = torch.rand(region_blocks.shape, device=x_det.device, generator=gen) < slip
    rand_feat = torch.randint(0, v, region_blocks.shape, device=x_det.device, generator=gen)
    slip_tuples = canon[rand_feat]
    det_tuples = x_det.gather(1, flat_positions).reshape(
        batch, region_blocks.shape[1], block_size)
    chosen = torch.where(do_slip[:, :, None], slip_tuples, det_tuples)
    x_out = x_det.clone()
    x_out.scatter_(1, flat_positions, chosen.reshape(batch, -1))
    return x_out, do_slip


def _train_block_fm_slip(fm, controller, generator, pool_roots, pool_leaves, canon,
                         region_index, n_regions, *, n_steps, batch_size, n_blocks, v, s,
                         n_corrupt, budget, lr, slip, device, seed=777):
    """Design 2: the same visited distribution as `_train_block_fm`, but the TARGET
    is the REALISED slippery outcome. MSE against realised outcomes converges to
    E[dz|z,k], so the FM becomes a mean-predictor and the slip realisation lands in
    its residual -- which is the whole point: the conditioning gap opens."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    fm.train()
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    report_every = max(1, n_steps // 5)
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, pool_leaves.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))
        for _ in range(g):
            k = torch.randint(0, n_regions, (batch_size,), device=device)
            x, _ = _regen_flagged(generator, x, region_index[k], canon,
                                  block_size=s, slip=slip, v=v, gen=gen)
        k = torch.randint(0, n_regions, (batch_size,), device=device)
        with torch.no_grad():
            z = controller.block_state(x)
            x2, _ = _regen_flagged(generator, x, region_index[k], canon,
                                   block_size=s, slip=slip, v=v, gen=gen)
            target = controller.block_state(x2) - z
        pred = fm(z, k)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        optimizer.step()
        if step % report_every == 0 or step == n_steps:
            cos = F.cosine_similarity(pred.reshape(batch_size, -1),
                                      target.reshape(batch_size, -1), dim=-1).mean().item()
            print(f"  D2 FM (q={slip}) step {step:5d}/{n_steps}: mse={loss.item():.5f} "
                  f"cos={cos:.3f}", flush=True)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def slip_gate1(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256,
    n_corrupt: int = 3, edit_budget: int = 6, region_size: int = 1,
    slips: str = "0.1,0.25,0.5",
    n_transitions: int = 40_000, probe_steps: int = 3_000, probe_lr: float = 3e-3,
    n_strata_bins: int = 40, explore_eps: float = 0.3,
    quick: bool = False, tag: str = "", seed: int = 42,
):
    """Step 1: is the slip directionally identifiable in the FM residual?"""
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from rhm.conditional_revision.gates_ab import _auc, _stratified_auc, _strata

    if s != 2:
        raise ValueError("assumes binary RHM branching (s=2), like the sculpting arc")
    sequence_length = s ** depth
    n_blocks = sequence_length // s
    qs = [float(x) for x in slips.split(",")]
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, value_episodes = 20_000, 6_000
        n_transitions, probe_steps = 6_000, 800

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"SCULPT-SLIP step 1: v={v} s={s} L={depth} m={m} blocks={n_blocks} "
          f"slips={qs} device={device}", flush=True)
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)
    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    RichController = _build_rich_controller()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()
    BlockLatentFM = _build_block_fm()

    # ---- the deterministic instruments, trained once and cached ----------------
    # Same construction as rhm_sculpt_latent_stoch.py, so the substrate is that
    # experiment's. Cached because Step 2 must reuse the identical instruments --
    # only the FM's training target may vary between the two steps.
    ck_dir = f"{DATA_DIR}/rhm_sculpt_slip"
    os.makedirs(ck_dir, exist_ok=True)
    stem = (f"v{v}_s{s}_L{depth}_m{m}_sd{state_dim}_ep{n_train_episodes}_"
            f"cs{controller_steps}_gs{generator_steps}_vs{value_steps}_seed{train_seed}"
            f"{'_quick' if quick else ''}")
    inst_path = f"{ck_dir}/instruments_{stem}.pt"

    controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
    generator = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                              root_conditioned=False).to(device)
    value = MCValueHead(state_dim, v).to(device)

    if os.path.exists(inst_path):
        sd = torch.load(inst_path, map_location=device)
        controller.load_state_dict(sd["controller"])
        generator.load_state_dict(sd["generator"])
        value.load_state_dict(sd["value"])
        print(f"loaded cached instruments <- {inst_path}", flush=True)
    else:
        _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                               n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                               lr=3e-4, device=device, p_full=0.5)
        _train_generator(generator, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                         n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                         n_steps=generator_steps, lr=3e-4, device=device)
        for module in (controller, generator):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
        print(f"Collecting value data ({value_episodes} rollouts)", flush=True)
        configs, roots_buf, success_buf = _collect_value_sculpt(
            controller, generator, train_roots_np, train_leaves_np, canon, region_index,
            n_regions, rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks,
            v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps,
            device=device)
        _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                        n_steps=value_steps, lr=3e-4, device=device)
        torch.save({"controller": controller.state_dict(),
                    "generator": generator.state_dict(),
                    "value": value.state_dict()}, inst_path)
        volume.commit()
        print(f"saved instruments -> {inst_path}", flush=True)

    for module in (controller, generator, value):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    # ---- slip=0 must reduce to the deterministic actuator ----------------------
    def _assert_slip0_matches_deterministic():
        g = torch.Generator(device=device).manual_seed(0)
        xs = torch.from_numpy(_corrupt(train_leaves_np[:256], n_blocks, n_corrupt, v, s,
                                       np.random.default_rng(0))).to(device)
        kk = torch.randint(0, n_regions, (256,), device=device)
        a = _regenerate(generator, xs, region_index[kk], canon, None, block_size=s, sample=False)
        b, fl = _regen_flagged(generator, xs, region_index[kk], canon, block_size=s,
                               slip=0.0, v=v, gen=g)
        assert torch.equal(a, b), "flagged actuator diverges from _regenerate at slip=0"
        assert not fl.any()
    _assert_slip0_matches_deterministic()
    print("slip=0 anchor: flagged actuator == deterministic _regenerate", flush=True)

    # ---- FM Design 1 (intended-outcome predictor), q-independent ---------------
    fm_d1 = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
    d1_path = f"{ck_dir}/fm_d1_{stem}_fs{fm_steps}.pt"
    if os.path.exists(d1_path):
        fm_d1.load_state_dict(torch.load(d1_path, map_location=device))
        print(f"loaded cached Design-1 FM <- {d1_path}", flush=True)
    else:
        print(f"Design-1 FM parameters: {_count_parameters(fm_d1):,}", flush=True)
        _train_block_fm(fm_d1, controller, generator, train_roots_np, train_leaves_np, canon,
                        region_index, n_regions, n_steps=fm_steps, batch_size=batch_size,
                        n_blocks=n_blocks, v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget,
                        lr=1e-3, device=device)
        torch.save(fm_d1.state_dict(), d1_path)
        volume.commit()
    fm_d1.eval()
    for p in fm_d1.parameters():
        p.requires_grad_(False)

    # ---- held-out transitions, with the exact slip label -----------------------
    def collect(slip, n, seed_off):
        """Transitions from the FM's own visited distribution, held out by seed."""
        rng = np.random.default_rng(20_000 + seed_off)
        gen = torch.Generator(device=device).manual_seed(30_000 + seed_off)
        Z, K, DZ, DZI, FL = [], [], [], [], []
        done = 0
        while done < n:
            b = min(batch_size, n - done)
            idx = rng.integers(0, train_leaves_np.shape[0], size=b)
            c = int(rng.integers(1, n_corrupt + 1))
            x = torch.from_numpy(_corrupt(train_leaves_np[idx], n_blocks, c, v, s, rng)).to(device)
            g = int(rng.integers(0, edit_budget + 1))
            for _ in range(g):
                k = torch.randint(0, n_regions, (b,), device=device)
                x, _ = _regen_flagged(generator, x, region_index[k], canon, block_size=s,
                                      slip=slip, v=v, gen=gen)
            k = torch.randint(0, n_regions, (b,), device=device)
            with torch.no_grad():
                z = controller.block_state(x)
                x2, fl = _regen_flagged(generator, x, region_index[k], canon, block_size=s,
                                        slip=slip, v=v, gen=gen)
                x2i = _regenerate(generator, x, region_index[k], canon, None,
                                  block_size=s, sample=False)
                dz = controller.block_state(x2) - z
                dzi = controller.block_state(x2i) - z
            Z.append(z); K.append(k); DZ.append(dz); DZI.append(dzi)
            FL.append(fl[:, 0])          # region_size=1 -> one acted block per move
            done += b
        return (torch.cat(Z), torch.cat(K), torch.cat(DZ), torch.cat(DZI),
                torch.cat(FL))

    def probe_auc(R, y, name):
        """Linear probe on the residual, held out. Returns (test scores, test mask)."""
        n = R.shape[0]
        ntr = int(0.7 * n)
        Xtr, ytr = R[:ntr], y[:ntr].float()
        Xte = R[ntr:]
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        clf = nn.Linear(R.shape[1], 1).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=probe_lr, weight_decay=1e-4)
        pw = torch.tensor([(1 - ytr.mean()) / ytr.mean().clamp(min=1e-6)], device=device)
        g = torch.Generator(device=device).manual_seed(seed + 3)
        for st in range(probe_steps):
            ix = torch.randint(0, ntr, (min(1024, ntr),), device=device, generator=g)
            loss = F.binary_cross_entropy_with_logits(
                clf(((Xtr[ix] - mu) / sd)).squeeze(-1), ytr[ix], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            sc = clf(((Xte - mu) / sd)).squeeze(-1)
        return sc.cpu().numpy()

    results = {"config": {"v": v, "s": s, "L": depth, "m": m, "state_dim": state_dim,
                          "n_blocks": n_blocks, "slips": qs, "fm_steps": fm_steps,
                          "n_transitions": n_transitions, "quick": quick, "tag": tag,
                          "train_seed": train_seed, "seed": seed},
               "per_q": {}}

    for q in qs:
        print(f"\n{'=' * 74}\nq = {q}\n{'=' * 74}", flush=True)
        fm_d2 = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
        d2_path = f"{ck_dir}/fm_d2_q{q}_{stem}_fs{fm_steps}.pt"
        if os.path.exists(d2_path):
            fm_d2.load_state_dict(torch.load(d2_path, map_location=device))
            print(f"loaded cached Design-2 FM <- {d2_path}", flush=True)
        else:
            _train_block_fm_slip(fm_d2, controller, generator, train_roots_np, train_leaves_np,
                                 canon, region_index, n_regions, n_steps=fm_steps,
                                 batch_size=batch_size, n_blocks=n_blocks, v=v, s=s,
                                 n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3, slip=q,
                                 device=device)
            torch.save(fm_d2.state_dict(), d2_path)
            volume.commit()
        fm_d2.eval()
        for p in fm_d2.parameters():
            p.requires_grad_(False)

        Z, K, DZ, DZI, FL = collect(q, n_transitions, seed_off=int(q * 1000))
        print(f"  {FL.numel()} transitions, slipped fraction {FL.float().mean():.3f}",
              flush=True)

        row = {"slipped_fraction": float(FL.float().mean()), "fms": {}}
        for fname, fm in (("design1_intended", fm_d1), ("design2_realised", fm_d2)):
            with torch.no_grad():
                pred = fm(Z, K)
                R = (DZ - pred)                       # (n, n_blocks, D)
                Ri = (DZI - pred)                     # residual vs the INTENDED outcome
            n_all = R.shape[0]
            Rf = R.reshape(n_all, -1)
            nrm = Rf.norm(dim=1)
            acted = R.gather(1, K[:, None, None].expand(-1, 1, R.shape[2]))[:, 0].norm(dim=1)
            unit = Rf / nrm[:, None].clamp(min=1e-8)
            cos_int = F.cosine_similarity(Rf, Ri.reshape(n_all, -1), dim=1)

            ntr = int(0.7 * n_all)
            y = FL.to(device).long()
            sc_dir = probe_auc(unit, y, "dir")
            sc_raw = probe_auc(Rf, y, "raw")
            gsh = torch.randperm(n_all, device=device, generator=torch.Generator(
                device=device).manual_seed(seed + 11))
            sc_shuf = probe_auc(unit, y[gsh], "shuf")

            yte = FL[ntr:].cpu().numpy().astype(bool)
            sig = {
                "resid_norm": nrm[ntr:].cpu().numpy(),
                "resid_norm_acted": acted[ntr:].cpu().numpy(),
                "cos_to_intent": cos_int[ntr:].cpu().numpy(),
                "dir_probe": sc_dir,
                "raw_probe": sc_raw,
                "dir_probe_shuf": sc_shuf,
            }
            strata = _strata(sig["resid_norm"], n_strata_bins)
            out = {}
            for nm, valv in sig.items():
                a_raw = _auc(valv, yte)
                a_m, _ = _stratified_auc(valv, yte, strata)
                out[nm] = {"auc_raw": a_raw, "auc_norm_matched": a_m}
            out["_guards"] = {
                "resid_norm_self_matched": out["resid_norm"]["auc_norm_matched"],
                "dir_probe_shuf_raw": out["dir_probe_shuf"]["auc_raw"],
                "n_strata": int(len(np.unique(strata))),
            }
            out["_scale"] = {
                "mean_norm_slipped": float(nrm[FL].mean()),
                "mean_norm_clean": float(nrm[~FL].mean()),
                "mean_norm_ratio": float(nrm[FL].mean() / nrm[~FL].mean().clamp(min=1e-8)),
            }
            row["fms"][fname] = out

            print(f"\n  {fname}:  ||r|| slipped {out['_scale']['mean_norm_slipped']:.4f} "
                  f"vs clean {out['_scale']['mean_norm_clean']:.4f} "
                  f"(ratio {out['_scale']['mean_norm_ratio']:.2f})")
            print(f"    guards: ||r|| self-matched {out['_guards']['resid_norm_self_matched']:.4f} "
                  f"(want ~0.5), shuffled-label probe {out['_guards']['dir_probe_shuf_raw']:.4f} "
                  f"(want ~0.5), strata {out['_guards']['n_strata']}")
            print(f"    {'signal':<20}{'AUC raw':>10}{'AUC ||r||-matched':>20}")
            for nm in sig:
                print(f"    {nm:<20}{out[nm]['auc_raw']:>10.4f}"
                      f"{out[nm]['auc_norm_matched']:>20.4f}")

        results["per_q"][str(q)] = row

    # ---- verdict ---------------------------------------------------------------
    print(f"\n{'=' * 74}\nSTEP 1 VERDICT\n{'=' * 74}")
    print(f"  {'q':<8}{'FM':<20}{'||r|| raw':>11}{'dir matched':>13}{'norm matched':>14}")
    passed = False
    for q in qs:
        for fname in ("design1_intended", "design2_realised"):
            o = results["per_q"][str(q)]["fms"][fname]
            d = o["dir_probe"]["auc_norm_matched"]
            nm = o["resid_norm"]["auc_norm_matched"]
            print(f"  {q:<8}{fname:<20}{o['resid_norm']['auc_raw']:>11.4f}"
                  f"{d:>13.4f}{nm:>14.4f}")
            if fname == "design2_realised" and d > 0.55:
                passed = True
    results["verdict"] = {
        "directional_readout_beats_norm_for_design2": bool(passed),
        "reading": ("the slip is directionally identifiable in the Design-2 residual, so a "
                    "low-rank precision operator has something to act on -- proceed to Step 2"
                    if passed else
                    "the slip is NOT directionally identifiable once ||r|| is matched; Step 2's "
                    "precision operator has nothing to act on. STOP."),
        "status": "PROVISIONAL",
    }
    print(f"\n  {results['verdict']['reading']}", flush=True)

    out_dir = f"{DATA_DIR}/rhm_sculpt_slip"
    os.makedirs(out_dir, exist_ok=True)
    name = f"step1{'_' + tag if tag else ''}_seed{train_seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}   ({time.time() - started:.0f}s)", flush=True)
    return results


# ===========================================================================
# Step 2 -- the intervention: can a precision operator recover what the slip costs?
# ===========================================================================
#
# Framing that makes this sharp. Under a uniform slip,
#     E[dz | z,k] = (1-q) * dz_intended + q * const
# which is AFFINE in the intended outcome, so Design 2's population optimum
# induces the SAME RANKING over moves as Design 1's. The entire cost of opening
# the conditioning gap is therefore ESTIMATION VARIANCE: training on realised
# targets injects the slip into every gradient. The aleatoric null is a
# variance-reduction claim here, with a ceiling we can actually compute.
#
# Arms -- every one trains on REALISED slippery targets (the only thing a real
# agent observes), consuming an identical data stream, differing ONLY in weighting:
#
#   d1_clean        FM trained on INTENDED targets. Not available in a slippery
#                   world; the upper bound on seeing through the noise.
#   d2_unweighted   plain MSE on realised targets. The incumbent / floor.
#   d2_absE         weight by rank-normalised instantaneous ||r||. This is
#                   endogenous_teacher's sign, and it upweights exactly the slips.
#                   PREDICTED TO HURT -- included so a known failure is a
#                   prediction rather than an accident.
#   d2_precision    the hypothesis. A low-rank precision operator Pi estimated
#                   from ACCUMULATED residual covariance with NO labels, applied
#                   to the acted block's residual. Two-timescale by construction:
#                   Pi is stop-gradded and refreshed slowly (idea doc SS5; Ruffini
#                   SS2.3 -- "precision requires pooling errors over time").
#   d2_prec_q0      Pi estimated in a q=0 world, where there is nothing aleatoric
#                   to down-weight. Isolates "the slip subspace was suppressed"
#                   from "whitening helps optimisation".
#   d2_oracle_gate  weight 0 on slipped transitions, 1 otherwise, renormalised to
#                   mean 1. The ceiling for gating; should approach d1_clean.
#
# Pi acts on the ACTED BLOCK's residual, not the flattened state: the slip
# randomises the acted block's feature, so in that frame its subspace is
# consistent across samples, whereas in the flattened frame it is smeared over
# whichever block happened to be acted on.
#
# Readouts: the sculpting arc's own ranking currency (value_top1_agree,
# value_rank_corr, delta_cos vs the INTENDED outcome) plus the behavioural one,
# latent-beam success under slip.


def _rank_weights(score):
    """endogenous_teacher's idiom: w = 2*rank/(N-1), mean ~1. Every arm that uses
    it gets a bit-identical weight multiset, so only the ASSIGNMENT differs."""
    import torch
    n = score.numel()
    order = torch.argsort(score.reshape(-1))
    w = torch.empty(n, device=score.device)
    w[order] = 2.0 * torch.arange(n, device=score.device, dtype=w.dtype) / max(n - 1, 1)
    return w


def _precision_operator(buf, rank, alpha, device):
    """Pi = I - (1-alpha) U_k U_k^T over the top-`rank` eigendirections of the
    accumulated residual covariance. alpha=1 is a no-op; alpha=0 projects the
    subspace out entirely. Returns (D, D), to be used stop-gradded."""
    import torch
    R = torch.cat(buf, 0)
    R = R - R.mean(0, keepdim=True)
    cov = (R.T @ R) / max(R.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    U = evecs[:, -rank:].float()                       # top-`rank` directions
    return (torch.eye(cov.shape[0], device=device)
            - (1.0 - alpha) * (U @ U.T)), evals.float().flip(0)[:rank].cpu().numpy()


def _train_fm_arm(arm, fm, controller, generator, pool_leaves, canon, region_index,
                  n_regions, *, n_steps, batch_size, n_blocks, v, s, n_corrupt, budget,
                  lr, slip, device, seed, precision_rank, precision_alpha,
                  precision_every, precision_buffer, log_every):
    """One arm. All arms share the data stream (same `seed`), so the only
    difference is how the per-sample / per-direction loss is weighted."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    fm.train()
    optimizer = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=1e-4)
    # `d2_prec_q0` estimates Pi in a world with no slip at all
    pi_slip = 0.0 if arm.startswith("d2_prec_q0") else slip
    uses_pi = arm.startswith("d2_precision") or arm.startswith("d2_prec_q0")
    Pi, buf, spec = None, [], None
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, pool_leaves.shape[0], size=batch_size)
        c = int(rng.integers(1, n_corrupt + 1))
        x = torch.from_numpy(_corrupt(pool_leaves[idx], n_blocks, c, v, s, rng)).to(device)
        g = int(rng.integers(0, budget + 1))
        for _ in range(g):
            k = torch.randint(0, n_regions, (batch_size,), device=device)
            x, _ = _regen_flagged(generator, x, region_index[k], canon,
                                  block_size=s, slip=slip, v=v, gen=gen)
        k = torch.randint(0, n_regions, (batch_size,), device=device)
        with torch.no_grad():
            z = controller.block_state(x)
            if arm == "d1_clean":                       # INTENDED target, no gap
                x2 = _regenerate(generator, x, region_index[k], canon, None,
                                 block_size=s, sample=False)
                flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
            else:                                       # REALISED target, gap open
                x2, fl = _regen_flagged(generator, x, region_index[k], canon,
                                        block_size=s, slip=slip, v=v, gen=gen)
                flags = fl[:, 0]
            target = controller.block_state(x2) - z
        pred = fm(z, k)
        resid = pred - target                                   # (B, n_blocks, D)
        acted = resid.gather(1, k[:, None, None].expand(-1, 1, resid.shape[2]))[:, 0]
        other_sq = (resid ** 2).sum(dim=(1, 2)) - (acted ** 2).sum(dim=1)

        if uses_pi:
            # accumulate the acted-block residual on a SLOW timescale, refresh Pi
            with torch.no_grad():
                if pi_slip != slip:      # q0 control: sample a clean residual to pool
                    x2c = _regenerate(generator, x, region_index[k], canon, None,
                                      block_size=s, sample=False)
                    rc = fm(z, k) - (controller.block_state(x2c) - z)
                    buf.append(rc.gather(1, k[:, None, None].expand(
                        -1, 1, rc.shape[2]))[:, 0].detach())
                else:
                    buf.append(acted.detach())
                if len(buf) * batch_size > precision_buffer:
                    buf.pop(0)
            if step % precision_every == 0 or Pi is None:
                with torch.no_grad():
                    Pi, spec = _precision_operator(buf, precision_rank,
                                                   precision_alpha, device)
            acted_sq = ((acted @ Pi.T) ** 2).sum(dim=1)          # Pi stop-gradded
        else:
            acted_sq = (acted ** 2).sum(dim=1)

        per_sample = acted_sq + other_sq
        if arm == "d2_absE":
            w = _rank_weights(per_sample.detach())
        elif arm == "d2_oracle_gate":
            w = (~flags).float()
            w = w / w.mean().clamp(min=1e-6)
        else:
            w = torch.ones_like(per_sample)
        loss = (w * per_sample).mean() / (n_blocks * resid.shape[2])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
        optimizer.step()
        if step % log_every == 0 or step == n_steps:
            cos = F.cosine_similarity(pred.reshape(batch_size, -1),
                                      target.reshape(batch_size, -1), dim=-1).mean().item()
            extra = "" if spec is None else f" pi_top_eig={spec[0]:.3f}"
            print(f"  [{arm}] step {step:5d}/{n_steps}: loss={loss.item():.5f} "
                  f"cos={cos:.3f}{extra}", flush=True)
    fm.eval()
    for p in fm.parameters():
        p.requires_grad_(False)
    return {"pi_spectrum_top": None if spec is None else spec[:8].tolist()}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=32768)
def slip_gate2(
    v: int = 8, s: int = 2, depth: int = 4, m: int = 2, rule_seed: int = 0, train_seed: int = 1,
    n_train_episodes: int = 100_000, state_dim: int = 96,
    controller_steps: int = 12_000, generator_steps: int = 12_000, value_steps: int = 12_000,
    fm_steps: int = 12_000, value_episodes: int = 40_000, batch_size: int = 256,
    n_corrupt: int = 3, edit_budget: int = 6, region_size: int = 1,
    slip: float = 0.25, n_eval_episodes: int = 2_048, beam_widths: str = "1,16,64",
    arms: str = "d1_clean,d2_unweighted,d2_absE,d2_precision,d2_prec_q0,d2_oracle_gate",
    precision_ranks: str = "8,16,32", precision_alphas: str = "0.1,0.5",
    precision_every: int = 500, precision_buffer: int = 20_000,
    explore_eps: float = 0.3, log_every: int = 3_000,
    quick: bool = False, tag: str = "", seed: int = 42,
):
    """Step 2: does suppressing the aleatoric component of the FM's residual
    recover the ranking quality and beam success the slip costs?"""
    import time
    import torch
    from rhm.rhm_sculpt_latent import _fm_check
    from rhm.rhm_sculpt_latent_stoch import _latent_beam_stoch
    from rhm.rhm_sculpt_precheck import nearest_derivation_cost

    sequence_length = s ** depth
    n_blocks = sequence_length // s
    widths = [int(x) for x in beam_widths.split(",")]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    # How hard to suppress is itself unknown -- the acted block's latent is 96-d and
    # the slip's subspace is at most ~v-dimensional, so an over-wide or over-strong
    # Pi removes signal rather than noise. Expand the two Pi arms over a small grid
    # and let `d2_prec_q0` at the SAME (rank, alpha) separate suppression from
    # geometry at every setting.
    pranks = [int(x) for x in precision_ranks.split(",")]
    palphas = [float(x) for x in precision_alphas.split(",")]
    expanded, pi_cfg = [], {}
    for a in arm_list:
        if a in ("d2_precision", "d2_prec_q0"):
            for r_ in pranks:
                for al in palphas:
                    nm = f"{a}_r{r_}_a{al}"
                    expanded.append(nm); pi_cfg[nm] = (r_, al)
        else:
            expanded.append(a)
    arm_list = expanded
    if quick:
        controller_steps = generator_steps = value_steps = fm_steps = 800
        n_train_episodes, value_episodes, n_eval_episodes = 20_000, 6_000, 512
        precision_every, precision_buffer, log_every = 200, 6_000, 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    torch.set_float32_matmul_precision("high")
    print(f"SCULPT-SLIP step 2: q={slip} arms={arm_list} widths={widths}", flush=True)
    started = time.time()

    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    inverse_maps = build_inverse_maps(rules)
    bottom_map = torch.from_numpy(inverse_maps[-1]).to(device)
    canon = torch.from_numpy(np.ascontiguousarray(rules[depth - 1][:, 0, :])).to(device)
    region_index, n_regions = _region_index(n_blocks, region_size, device)
    train_roots_np, train_leaves_np = _sample_pool(rules, n_train_episodes, s, train_seed)
    train_leaves = torch.from_numpy(train_leaves_np)
    train_roots = torch.from_numpy(train_roots_np)

    RichController = _build_rich_controller()
    BlockInfiller = _build_generator()
    MCValueHead = _build_value_head()
    BlockLatentFM = _build_block_fm()

    # ---- the SAME cached instruments Step 1 used ------------------------------
    ck_dir = f"{DATA_DIR}/rhm_sculpt_slip"
    os.makedirs(ck_dir, exist_ok=True)
    stem = (f"v{v}_s{s}_L{depth}_m{m}_sd{state_dim}_ep{n_train_episodes}_"
            f"cs{controller_steps}_gs{generator_steps}_vs{value_steps}_seed{train_seed}"
            f"{'_quick' if quick else ''}")
    inst_path = f"{ck_dir}/instruments_{stem}.pt"
    controller = RichController(v, sequence_length, s, state_dim, n_head=4, n_layer=2).to(device)
    generator = BlockInfiller(v, sequence_length, s, state_dim, n_head=4, n_layer=2,
                              root_conditioned=False).to(device)
    value = MCValueHead(state_dim, v).to(device)
    if os.path.exists(inst_path):
        sd = torch.load(inst_path, map_location=device)
        controller.load_state_dict(sd["controller"])
        generator.load_state_dict(sd["generator"])
        value.load_state_dict(sd["value"])
        print(f"loaded cached instruments <- {inst_path}", flush=True)
    else:
        _train_edit_controller(controller, train_leaves, train_roots, batch_size=batch_size,
                               n_blocks=n_blocks, block_size=s, n_steps=controller_steps,
                               lr=3e-4, device=device, p_full=0.5)
        _train_generator(generator, train_leaves, train_roots, bottom_map, batch_size=batch_size,
                         n_blocks=n_blocks, v=v, block_size=s, mask_min=1, mask_max=n_blocks,
                         n_steps=generator_steps, lr=3e-4, device=device)
        for mod in (controller, generator):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
        configs, roots_buf, success_buf = _collect_value_sculpt(
            controller, generator, train_roots_np, train_leaves_np, canon, region_index,
            n_regions, rules, n_episodes=value_episodes, batch_size=1024, n_blocks=n_blocks,
            v=v, s=s, n_corrupt=n_corrupt, budget=edit_budget, epsilon=explore_eps,
            device=device)
        _train_value_mc(value, controller, configs, roots_buf, success_buf, batch_size=512,
                        n_steps=value_steps, lr=3e-4, device=device)
        torch.save({"controller": controller.state_dict(),
                    "generator": generator.state_dict(),
                    "value": value.state_dict()}, inst_path)
        volume.commit()
    for mod in (controller, generator, value):
        mod.eval()
        for p in mod.parameters():
            p.requires_grad_(False)

    # ---- eval instances (shared across arms) ----------------------------------
    planner_batch = min(1024, n_eval_episodes)
    eval_roots_np, eval_leaves_np = _sample_pool(rules, planner_batch, s, train_seed + 99)
    rng = np.random.default_rng(train_seed + 2)
    start_np = _corrupt(eval_leaves_np, n_blocks, n_corrupt, v, s, rng)
    leaves0 = torch.from_numpy(start_np)
    targets = torch.from_numpy(eval_roots_np)
    dstar = nearest_derivation_cost(rules, start_np, eval_roots_np, s)
    frac_solvable = float((dstar <= n_corrupt * s).mean())
    print(f"eval: frac_solvable(q=0)={frac_solvable:.3f}", flush=True)

    results = {"config": {"v": v, "s": s, "L": depth, "m": m, "slip": slip,
                          "arms": arm_list, "beam_widths": widths, "fm_steps": fm_steps,
                          "precision_ranks": pranks, "precision_alphas": palphas,
                          "precision_every": precision_every, "quick": quick,
                          "tag": tag, "train_seed": train_seed},
               "frac_solvable_q0": frac_solvable, "arms": {}}

    for arm in arm_list:
        print(f"\n{'=' * 74}\nARM {arm}  (q={slip})\n{'=' * 74}", flush=True)
        fm = BlockLatentFM(state_dim, n_blocks, n_head=4, n_layer=2).to(device)
        info = _train_fm_arm(arm, fm, controller, generator, train_leaves_np, canon,
                             region_index, n_regions, n_steps=fm_steps,
                             batch_size=batch_size, n_blocks=n_blocks, v=v, s=s,
                             n_corrupt=n_corrupt, budget=edit_budget, lr=1e-3, slip=slip,
                             device=device, seed=777,
                             precision_rank=pi_cfg.get(arm, (8, 0.1))[0],
                             precision_alpha=pi_cfg.get(arm, (8, 0.1))[1],
                             precision_every=precision_every,
                             precision_buffer=precision_buffer, log_every=log_every)
        # ranking currency, measured against the INTENDED outcome -- the sculpting
        # arc's own metric set (RHM_SCULPTING_README: top-1 / rank-corr / delta_cos)
        chk = _fm_check(fm, value, controller, generator, leaves0, targets, canon,
                        region_index, n_regions, s, device)
        beams = {}
        for w in widths:
            beams[w] = _latent_beam_stoch(
                controller, generator, fm, value, leaves0, targets, canon, region_index,
                n_regions, rules, s=s, budget=edit_budget, beam_width=w, slip=slip, v=v,
                seed=train_seed + 5 + w, device=device)   # same beam seed for every arm
        results["arms"][arm] = {**chk, **info,
                                "beam_success": {str(w): beams[w] for w in widths}}
        print(f"  {arm}: delta_cos={chk['delta_cos']:.3f} "
              f"top1={chk['value_top1_agree']:.3f} rank_corr={chk['value_rank_corr']:.3f}  "
              f"beam " + "  ".join(f"w{w} {beams[w]:.3f}" for w in widths), flush=True)

    print(f"\n{'=' * 74}\nSUMMARY (q={slip})\n{'=' * 74}")
    hdr = (f"  {'arm':<24}{'delta_cos':>10}{'top1':>8}{'rank_corr':>11}"
           + "".join(f"{f'beam w{w}':>11}" for w in widths))
    print(hdr)
    for arm in arm_list:
        r = results["arms"][arm]
        print(f"  {arm:<24}{r['delta_cos']:>10.3f}{r['value_top1_agree']:>8.3f}"
              f"{r['value_rank_corr']:>11.3f}"
              + "".join(f"{r['beam_success'][str(w)]:>11.3f}" for w in widths))
    print("\n  Registered: d1_clean is the ceiling and d2_unweighted the floor; "
          "d2_oracle_gate should approach d1_clean; d2_absE should be the WORST "
          "(it upweights the slips); d2_precision is the hypothesis and d2_prec_q0 "
          "the geometry control that isolates it.", flush=True)

    out_dir = f"{DATA_DIR}/rhm_sculpt_slip"
    name = f"step2{'_' + tag if tag else ''}_q{slip}_seed{train_seed}.json"
    with open(f"{out_dir}/{name}", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nSaved -> {out_dir}/{name}   ({time.time() - started:.0f}s)", flush=True)
    return results
