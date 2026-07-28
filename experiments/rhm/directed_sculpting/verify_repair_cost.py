"""Validate the repair-cost instrument before trusting it to carry the homeostatic claim.

R1 zero-drift null    -- with KL = 0 there is no gap, so the instrument must report
                         "out of range" rather than a number. A readout that returns a
                         confident 0 (or the max budget) on a null drift would manufacture
                         a trend across a drift sequence.
R2 responds to KL     -- the attributable GAP must scale with drift magnitude; if the
                         instrument does not respond to the thing it measures, a flat cost
                         curve across events means nothing. Read on the gap, NOT on
                         cost-to-a-fraction-of-the-gap, which is self-normalising and
                         decreases with magnitude by construction (see
                         `rhm_repair_cost.cost_from_progress`). The gap additionally
                         cross-checks against the closed-form KL, which is the stronger
                         test: they were built independently.
R3 raw CE is blind    -- the headline trap. Analytically the entropy drop cancels the KL
                         exactly at the derivation level; this checks it survives at the
                         LEAF level the model is actually graded on, where ambiguous rule
                         tables make the cancellation approximate. Reports the stale
                         model's CE shift across a drift event against the gap the
                         gap-normalised readout sees.
R4 probe stays valid  -- per-level ancestor labels on a frozen probe set are unchanged by
                         drift (rule tables never move), so the depth half of the
                         two-instrument read is available. Cross-checks B5.

Run from experiments/:
  modal run rhm/directed_sculpting/verify_repair_cost.py::verify_repair_cost --quick
  modal run --detach rhm/directed_sculpting/verify_repair_cost.py::verify_repair_cost
"""

import json
import os

import modal
import numpy as np

from rhm.rhm_data import generate_rules_distinct, generate_sequences_weighted
from rhm.rhm_drift import (calibrate_sigma, drift_kl, make_drift_state, ou_step,
                           state_weights, uniform_weights)
from rhm.rhm_repair_cost import (Meter, attributable_progress, cost_from_progress,
                                 raw_ce_blindness, repair_cost, repair_progress)
from rhm.shared import DATA_DIR, NumpyEncoder, image, volume


app = modal.App("rhm-verify-repair-cost", image=image)


def _drift_to_kl(rules, s, levels, kappa, target_kl, seed, n_steps=400):
    """Run the OU walk to stationarity at a sigma calibrated for `target_kl` per level."""
    if target_kl <= 0:
        return uniform_weights(rules), 0.0
    sigma = calibrate_sigma(rules, s, levels, kappa, target_kl, seed=seed)
    st = make_drift_state(rules, levels, seed=seed)
    rng = np.random.default_rng(seed + 77)
    for _ in range(n_steps):
        ou_step(st, kappa, sigma, rng)
    w = state_weights(rules, st)
    return w, drift_kl(rules, s, uniform_weights(rules), w)[0]


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=16384)
def verify_repair_cost(
    v: int = 8, s: int = 2, depth: int = 5, m: int = 2, rule_seed: int = 0, seed: int = 1,
    kappa: float = 0.05, drift_levels: str = "3,4",
    kl_sweep: str = "0.0,0.15,0.30,0.60", base_steps: int = 6000,
    repair_steps: int = 1500, matched_steps: int = 6000, batch_size: int = 128,
    lr: float = 3e-4, n_layer: int = 6, n_head: int = 6, n_embd: int = 192,
    n_eval: int = 4096, eval_every: int = 100, tag: str = "v1", quick: bool = False,
):
    import copy
    import time

    import torch
    import torch.nn.functional as F

    from rhm.model import GPT

    if quick:
        base_steps, repair_steps, matched_steps, n_eval, eval_every = 800, 300, 800, 1024, 50
        kl_sweep = "0.0,0.30"

    levels = [int(x) for x in drift_levels.split(",")]
    kls = [float(x) for x in kl_sweep.split(",")]
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    base_w = uniform_weights(rules)
    T = s ** depth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.time()
    print(f"Repair-cost validation: v={v} s={s} L={depth} m={m}, drift levels {levels}, "
          f"KL sweep {kls}, device={device}")

    def batches(weights, n, sd):
        return torch.from_numpy(generate_sequences_weighted(rules, n, weights, seed=sd))

    def ce_on(model, data):
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, data.shape[0], 256):
                x = data[i:i + 256].to(device)
                logits, _ = model(x[:, :-1])
                ce = F.cross_entropy(logits.reshape(-1, v), x[:, 1:].reshape(-1),
                                     reduction="sum")
                tot += float(ce)
                cnt += x.shape[0] * (x.shape[1] - 1)
        model.train()
        return tot / cnt

    def train(model, weights, n_steps, sd, meter=None, evals=None, eval_data=None,
              refresh_every=25):
        """Train on FRESH sequences -- every sequence is generated once and consumed once.

        The first cut resampled a fixed 20k pool, which broke two things at once. (a) The
        meter's semantics: it charges "sequences consumed", and the whole reason metering
        makes "where should I spend" a question is that each charged unit is data the agent
        had to go and get. Charging for reused sequences measures gradient steps wearing a
        costume. (b) Overfitting: 6000 steps x batch 128 over a 20k pool is ~38 epochs, and
        it showed -- the base model was WORSE held-out at 6000 steps than at 800 (0.7420 vs
        0.7306), and the matched reference degraded to CE 1.31, making every `gap_total`
        negative. The difference-in-differences absorbed it (both arms overfit equally, so
        the subtraction cancelled), which is a good sign for the DiD design but leaves the
        absolute scale resting on a broken ceiling. Fresh sampling removes all of it.
        """
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        curve, pool, cursor = [], None, 0
        for step in range(n_steps + 1):
            if evals is not None and step % eval_every == 0:
                curve.append((meter.collect if meter else step * batch_size,
                              ce_on(model, eval_data)))
            if step == n_steps:
                break
            if pool is None or cursor + batch_size > pool.shape[0]:
                pool = batches(weights, batch_size * refresh_every,
                               sd * 1000003 + step)
                cursor = 0
            x = pool[cursor:cursor + batch_size].to(device)
            cursor += batch_size
            if meter is not None:
                meter.charge_collect(batch_size)
            _, loss = model(x[:, :-1], targets=x[:, 1:].contiguous())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return curve

    # ---- base model, trained on the UNDRIFTED DGP ---------------------------
    base = GPT(vocab_size=v, block_size=T, n_layer=n_layer, n_head=n_head,
               n_embd=n_embd).to(device)
    eval_pre = batches(base_w, n_eval, seed + 900)
    train(base, base_w, base_steps, seed + 1)
    ce_pre_pre = ce_on(base, eval_pre)
    print(f"base model trained ({base_steps} steps); CE on pre-drift data = {ce_pre_pre:.4f}")

    results = {"config": {"v": v, "s": s, "L": depth, "m": m, "levels": levels,
                          "kl_sweep": kls, "base_steps": base_steps,
                          "repair_steps": repair_steps, "matched_steps": matched_steps,
                          "batch_size": batch_size},
               "ce_pre_on_pre": ce_pre_pre, "events": []}

    # ---- the matched-COMPUTE control arm, run once -------------------------
    # Same base model, same budgets, but data from the UNDRIFTED weights. Everything this
    # arm gains is ordinary continued training, so differencing against it isolates what
    # the drift actually cost. Without it the smoke showed a +0.0143 "gap" at ZERO drift.
    ctrl_matched = copy.deepcopy(base)
    train(ctrl_matched, base_w, matched_steps, seed + 12)
    ce_ctrl_matched = ce_on(ctrl_matched, eval_pre)
    gap_nodrift = ce_pre_pre - ce_ctrl_matched

    ctrl_repair = copy.deepcopy(base)
    ctrl_meter = Meter()
    ctrl_curve = train(ctrl_repair, base_w, repair_steps, seed + 13, meter=ctrl_meter,
                       evals=True, eval_data=eval_pre)
    ctrl_by_budget = {int(b): c for b, c in ctrl_curve}
    results["control_arm"] = {
        "ce_matched_nodrift": ce_ctrl_matched, "gap_nodrift": gap_nodrift,
        "curve": [{"budget_seqs": int(b), "ce": float(c)} for b, c in ctrl_curve]}
    print(f"matched-compute control: gap from pure extra training = {gap_nodrift:+.5f} nats")

    # First pass over the sweep establishes the NULL arm's attributable gap, which sets the
    # instrument's resolution floor (see `attributable_progress`). Any event whose
    # attributable gap does not clear `null_floor` is reported out-of-range, not as a number.
    null_gap = None
    for target in kls:
        w_new, kl_actual = _drift_to_kl(rules, s, levels, kappa, target, seed=seed)
        blind = raw_ce_blindness(rules, s, base_w, w_new)
        eval_post = batches(w_new, n_eval, seed + 901)

        ce_stale = ce_on(base, eval_post)                     # stale model, new data

        matched = copy.deepcopy(base)                         # the ceiling reference
        train(matched, w_new, matched_steps, seed + 2)
        ce_matched = ce_on(matched, eval_post)
        gap_total = ce_stale - ce_matched
        gap_attr = gap_total - gap_nodrift
        if target == 0.0:
            null_gap = abs(gap_attr)
        null_floor = 2.0 * (null_gap if null_gap is not None else 0.0)

        repairing = copy.deepcopy(base)
        meter = Meter()
        curve = train(repairing, w_new, repair_steps, seed + 3, meter=meter,
                      evals=True, eval_data=eval_post)
        attr_curve = [
            (b, attributable_progress(c, ce_stale, gap_total,
                                      ctrl_by_budget.get(int(b), ce_pre_pre), ce_pre_pre,
                                      gap_nodrift, min_gap=null_floor))
            for b, c in curve]
        # cost is read off the ATTRIBUTABLE curve, the same one the verdicts use
        cost = cost_from_progress(attr_curve, threshold=0.9)
        cost_raw = repair_cost(curve, ce_stale, ce_matched, threshold=0.9)

        rec = {
            "target_kl_per_level": target, "kl_actual_nats_per_seq": kl_actual,
            "raw_ce_blindness": blind,
            "ce_stale_on_post": ce_stale, "ce_matched_on_post": ce_matched,
            "gap_total": gap_total, "gap_nodrift": gap_nodrift,
            "gap_attributable": gap_attr, "null_resolution_floor": null_floor,
            "clears_null_floor": bool(abs(gap_attr) >= null_floor),
            "raw_ce_shift_measured": ce_stale - ce_pre_pre,
            "repair_cost_seqs_to_90pct": cost, "repair_cost_seqs_to_90pct_raw": cost_raw,
            # primary readout: progress at a FIXED budget. Never degenerate, always
            # comparable across events -- unlike budget-to-threshold, which returns None
            # whenever the curve stops short (it did at every KL in the smoke).
            "final_repair_progress_raw": repair_progress(curve[-1][1], ce_stale, ce_matched),
            "final_repair_progress_attributable": attr_curve[-1][1],
            "curve": [{"budget_seqs": int(b), "ce": float(c),
                       "progress_attributable": float(p)}
                      for (b, c), (_b, p) in zip(curve, attr_curve)],
            "monitor_collect_ratio": meter.ratio,
        }
        results["events"].append(rec)
        print(f"\nKL/level={target}  (actual {kl_actual:.3f} nats/seq)")
        print(f"  raw CE: stale-on-post {ce_stale:.4f} vs pre-on-pre {ce_pre_pre:.4f} "
              f"-> shift {ce_stale - ce_pre_pre:+.4f}")
        print(f"  gap total {gap_total:+.5f} = nodrift {gap_nodrift:+.5f} "
              f"+ attributable {gap_attr:+.5f}")
        print(f"  final progress: raw {rec['final_repair_progress_raw']:.3f}, "
              f"attributable {rec['final_repair_progress_attributable']:.3f}; "
              f"cost-to-90% {cost}")

    # ---- verdicts ----------------------------------------------------------
    zero = [e for e in results["events"] if e["target_kl_per_level"] == 0.0]
    nonzero = [e for e in results["events"] if e["target_kl_per_level"] > 0.0]

    r1 = None
    if zero:
        z = zero[0]
        r1 = {"gap_total": z["gap_total"], "gap_attributable": z["gap_attributable"],
              "cost": z["repair_cost_seqs_to_90pct"],
              "attributable_progress_out_of_range":
                  not np.isfinite(z["final_repair_progress_attributable"])}
    r2 = {"kl": [e["kl_actual_nats_per_seq"] for e in nonzero],
          "cost": [e["repair_cost_seqs_to_90pct"] for e in nonzero],
          "gap": [e["gap_attributable"] for e in nonzero],
          "progress_at_fixed_budget":
              [e["final_repair_progress_attributable"] for e in nonzero]}
    costs = [c for c in r2["cost"] if c is not None]
    r2["monotone_in_kl"] = bool(all(a <= b for a, b in zip(costs, costs[1:]))) if len(costs) > 1 else None
    r2["gap_monotone_in_kl"] = bool(all(a <= b for a, b in zip(r2["gap"], r2["gap"][1:])))
    r3 = [{"kl": e["kl_actual_nats_per_seq"],
           "raw_ce_shift": e["raw_ce_shift_measured"],
           "gap": e["gap_attributable"],
           "raw_over_gap": abs(e["raw_ce_shift_measured"]) / max(abs(e["gap_attributable"]), 1e-9)}
          for e in nonzero]
    results["r1_zero_drift_null"] = r1
    results["r2_monotone"] = r2
    results["r3_raw_ce_blind"] = r3

    print("\n--- verdicts ---")
    if r1:
        print(f"R1 zero-drift null : gap_total={r1['gap_total']:+.5f} -> "
              f"attributable {r1['gap_attributable']:+.5f}; out-of-range reported: "
              f"{r1['attributable_progress_out_of_range']}")
    print(f"R2 monotone in KL  : KL {[round(x,3) for x in r2['kl']]} -> "
          f"attributable gap {[round(x,5) for x in r2['gap']]} "
          f"(monotone: {r2['gap_monotone_in_kl']}); cost {r2['cost']}")
    print("R3 raw CE blind    :")
    for rec in r3:
        print(f"     KL={rec['kl']:.3f}: raw CE shift {rec['raw_ce_shift']:+.4f} vs "
              f"gap {rec['gap']:+.4f}  (raw/gap = {rec['raw_over_gap']:.2%})")

    results["elapsed_seconds"] = time.time() - started
    out_dir = f"{DATA_DIR}/directed_sculpting/repair_cost_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {out_dir}/results.json ({results['elapsed_seconds']:.0f}s)")
    return results
