"""tuning/gate2_lm — Gate 1 wave 2: the meter arms, on machinery fixed by wave 1.

SPEC: `SPEC.md` + Addendum. FORK NOTICE: `gate1_lm.py` forked verbatim, then extended below
"NEW IN WAVE 2". Donors imported, never edited.

THE THREE FIXES WAVE 1 EARNED (all diagnosed as measurement design, no code bugs):

 1. DEAD ZONES RE-DERIVED IN-TAG. Wave 1's zones came from `calib2`, which measures the null
    jump on a reader that neither acts nor lives in a moving world. Measured in-tag on the
    real thing, the online floor is 3.6x higher on d_key and 2.1x on d_out:
        coordinate   calib p99   in-tag p99   wave-2 dz
        |d_key|        0.0207      0.0755       0.076
        |d_self|       0.0631      0.0708       0.071
        |d_out|        0.0132      0.0273       0.028
 2. `rot_requires_out = False`. At the first rotation the online d_out is +0.0092 — BELOW the
    in-tag floor — so requiring it could only ever reject. d_self ~ 0 is the discriminating
    coordinate; d_out only rejects. Replaying wave 1's own logged reads under (1)+(2): the
    missed s8000 merge now types `rotation`, and all three bursts type `burst` at every
    checkpoint instead of flickering to `drift`. Both wave-1 misses are fixed by threshold
    re-derivation alone.
 3. THE DRIFT GETS A ZERO-LAG READ. Wave 1's grid had micro-points after bursts and rotations
    but none after drifts, and the epoch turned AFTER the drift-step checkpoint — so a drift
    was first observed 125 steps late, by which time this reader has absorbed ~95% of it (it
    absorbs 86% of a +0.65 rotation shock in 50 steps). Here the epoch turns BEFORE the
    checkpoint and {+0,+25,+50,+75} are added at every drift onset.

THE M ARMS, AND WHAT WAVE 1 SAYS TO EXPECT. Gate 0 measured the FM's sign cleanly on a static
world. In a DRIFTING world its quiet floor grows to meet the signal: |delta_rel| quiet p90 is
0.289 against rotation peaks +0.33/+0.27 and burst peaks -0.38/-0.25/-0.31 — SNR ~ 1. The
online-vs-frozen LAG statistic is far stronger for input-side shocks (quiet p90 0.548 vs burst
peaks +1.31/+5.17/+8.46) but does not separate burst from drift. So:
  `sur_fm`  runs the SPEC's FM-factor-alone map on the ONLINE sign, at its measured SNR ~ 1 —
            the marginality is the result, stated in advance rather than discovered after;
  `self_fm` grades with the LAG statistic, which is what it is good at: "did a real input-side
            shock just happen", i.e. the grader role, not the typer role.

Run from experiments/:
  modal run -m rhm.practice.tuning.gate2_lm::gate2 --quick --tag g2_smoke
  python3 rhm/practice/tuning/launch_g2.py --fn gate2 --tag g2a --log-dir /tmp
"""

import json
import os
import time

import modal

from rhm.shared import DATA_DIR, NumpyEncoder, setting_key, volume

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("rhm-practice-tuning-g2", image=image)

REMOTE = "rhm_practice_tuning_g2"
G1_REMOTE = "rhm_practice_tuning_g1"   # read-only: wave 1's twins for the fidelity gate
DONOR_REMOTE = "rhm_practice_fourwall_lm"          # read-only: refs + the gate twins
REF_KEYS = ("v", "s", "depth", "m", "rule_seed", "key_level", "key_node",
            "n_eval", "eval_seed", "n_probe", "probe_seed")

# arm -> (wall kind, phase1, rot_period, seed offset, POLICY, instruments)
#
# WAVE 2 POLICIES.  All inherit the FIXED typing triple (dz re-derived in-tag,
# rot_requires_out=False); they differ in what happens AFTER the instant.
#   tstar        the fixed T* arm. Both the wave-2 substrate and the direct re-test of the
#                two wave-1 misses (s8000 rotation, burst type-flicker).
#   tstar_fm     T* picks the op; the FM's LAG statistic grades it over the next window and
#                REVERTS what is revertible. Only a wrongly-SKIPPED rotation and a wrongly-
#                CONTINUED burst can be undone — a merge cannot (fwlm1 finding 5: the
#                abandoned circuitry evaporates within ~250 steps).
#   sur_fm       no counterfactual read at all: a d_key spike buys one window of `continue`,
#                then the FM's ONLINE SIGN alone picks merge (e rose) or skip (e fell).
#   tstar_abba   T* types; on a rotation-typed event the merge is put to an ABBA paired trial
#                (C K K C) instead of fired, and the reads are CHARGED against the budget.
#   tstar_rekey  T* types; on a rotation-typed event the arm RE-KEYS instead of merging —
#                a priced candidate search over all v offsets, on the OBSERVED wall token
#                (no latent is consulted, so no oracle leak), charged.
#   rekey_dead   the admissibility control, priced like `merge_dead`: the identical token
#                permutation applied to a reader whose wall never carried anything.
ARM_SPECS = {
    "self_v2":     ("wall", None, None, 0, "tstar",       "full"),
    "self_fm":     ("wall", None, None, 0, "tstar_fm",    "full"),
    "sur_fm":      ("wall", None, None, 0, "sur_fm",      "full"),
    "self_abba":   ("wall", None, None, 0, "tstar_abba",  "full"),
    "self_rekey":  ("wall", None, None, 0, "tstar_rekey", "full"),
    "rekey_dead":  ("dead", None, 0,    0, "rekey_dead",  "base"),
}
WORKER_SPECS = {
    "self_v2":    ["self_v2"],
    "self_fm":    ["self_fm"],
    "sur_fm":     ["sur_fm"],
    "self_abba":  ["self_abba"],
    "rekey":      ["self_rekey", "rekey_dead"],   # co-resident: the control mirrors the op
}
WAVES = {"A": ["self_v2", "self_fm", "sur_fm", "self_abba"], "B": ["rekey"]}

FM_LRS = (3e-4, 1e-3, 3e-3, 1e-2)      # the one FM sweep the spec asks for
FM_PRIMARY = 3e-3                      # Gate 0's sweep: SNR 5.58 vs 4.64 at 1e-3


def arm_schedule(arm, phase1, rot_period, max_steps):
    kind, p1, rp, soff, policy, instr = ARM_SPECS[arm]
    p1 = phase1 if p1 is None else p1
    rp = rot_period if rp is None else rp
    if rp <= 0:
        p1 = max_steps + 1                                  # never rotates
    return {"kind": kind, "phase1": int(p1), "rot_period": int(rp),
            "seed_off": int(soff), "merge_at": int(max_steps + 1),
            "policy": policy, "instr": instr,
            "n_fm": 4 if instr in ("full", "base") else 0}


def load_refs(cfg, donor_tag="fwlm1", verbose=True):
    """Reuse the donor's exact BP references verbatim when the config matches.

    `excess` is measured against `bayes_none`, so sharing the donor's array is what makes
    the `excess` instruments bit-comparable with the twins (endo_yield's idiom).
    """
    path = f"{DATA_DIR}/{DONOR_REMOTE}/{donor_tag}/setup.json"
    if os.path.exists(path):
        setup = json.load(open(path))
        d = setup["config"]
        if all(d.get(k) == cfg[k] for k in REF_KEYS):
            if verbose:
                print(f"[refs] loaded verbatim from {DONOR_REMOTE}/{donor_tag}/setup.json",
                      flush=True)
            return setup["refs"], f"{donor_tag}:setup.json"
        if verbose:
            print(f"[refs] donor config mismatch: "
                  f"{[k for k in REF_KEYS if d.get(k) != cfg[k]]} -> recomputing", flush=True)
    from rhm.practice.fourwall.lm.wall_lm import build_refs
    refs = build_refs(cfg["v"], cfg["s"], cfg["depth"], cfg["m"], cfg["rule_seed"],
                      cfg["key_level"], cfg["key_node"], cfg["n_eval"], cfg["eval_seed"],
                      cfg["n_probe"], cfg["probe_seed"], verbose=verbose)
    return refs, "recomputed"


def _build_refs_for(rules, cfg, _unused=None):
    """`wall_lm.build_refs` for a grammar that is HANDED IN rather than reseeded.

    The donor's `build_refs` regenerates its rules from `rule_seed`, which is exactly right
    for a static world and useless for a drifting one. This is its body with the rules as
    an argument — forked, not edited, per the arc's donor rule. Epoch 0 still comes from
    `load_refs`, i.e. verbatim from the donor tag, so the fidelity gate is untouched.
    """
    import numpy as np
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W

    v, s, L, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    key_level, key_node = cfg["key_level"], cfg["key_node"]
    T = s ** L
    ev_leaf, ev_lf, _ = _generate_with_traces(rules, cfg["n_eval"], cfg["eval_seed"])
    ev_z = ev_lf[key_level][:, key_node]
    pr_leaf, pr_lf, _ = _generate_with_traces(rules, cfg["n_probe"], cfg["probe_seed"])
    pr_z = pr_lf[key_level][:, key_node]
    lo, hi = W.key_span(L, s, key_level, key_node)
    key_plen, last_plen = hi, T - 1

    t0 = time.time()
    bayes_none = W.exact_predictive(rules, ev_leaf)
    bayes_key = W.exact_predictive(rules, ev_leaf, clamp=(key_level, key_node, ev_z))
    print(f"[refs] exact per-position Bayes surprisal, both conditions, "
          f"{time.time() - t0:.0f}s", flush=True)

    n_ceil = min(cfg["n_probe"], 2000)
    sub, sub_z = pr_leaf[:n_ceil], pr_z[:n_ceil]
    ceil = {"key_none": W.exact_ceilings(rules, sub, key_plen),
            "key_wall": W.exact_ceilings(rules, sub, key_plen,
                                         clamp=(key_level, key_node, sub_z)),
            "last_none": W.exact_ceilings(rules, sub, last_plen),
            "last_wall": W.exact_ceilings(rules, sub, last_plen,
                                          clamp=(key_level, key_node, sub_z))}
    P0 = W.exact_leaf0_by_key(rules, key_level, key_node)
    D0 = W.jsd_matrix(P0)
    ptl = W.pos_top_level(T, L, s)
    idx, out_ = slice(lo, hi), slice(hi, T)
    gate0 = {
        "index_value_indexed_span": float((bayes_none[idx] - bayes_key[idx]).mean()),
        "index_value_outside_span": float((bayes_none[out_] - bayes_key[out_]).mean()),
        "index_value_all": float((bayes_none - bayes_key).mean()),
        "bayes_indexed_span": float(bayes_none[idx].mean()),
        "bayes_outside_span": float(bayes_none[out_].mean()),
    }
    return {"T": T, "L": L, "key_plen": int(key_plen), "last_plen": int(last_plen),
            "key_lo": int(lo), "key_hi": int(hi),
            "bayes_none": bayes_none.tolist(), "bayes_key": bayes_key.tolist(),
            "ceil": ceil, "exact_leaf0": P0.tolist(),
            "exact_jsd_mean": float(D0[~np.eye(v, dtype=bool)].mean()),
            "exact_jsd_min_offdiag": float(D0[~np.eye(v, dtype=bool)].min()),
            "pos_top_level": ptl.tolist(), "gate0": gate0}


# --------------------------------------------------------------------------- #
# one worker = one or more CO-RESIDENT readers, each with its own global-RNG state
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=32400, memory=32768)
def run_worker(tag: str, worker: str, scheds: dict, refs_by_epoch: list, cfg: dict):
    import copy

    import numpy as np
    import torch
    import torch.nn.functional as F
    from a2a_forward.forward_model import TransformerForwardModel
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import burst as BU
    from rhm.practice.tuning import drift_ev as DR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    T = s ** L
    V = W.vocab_size(v)
    rot_step, key_level, key_node = cfg["rot_step"], cfg["key_level"], cfg["key_node"]
    B = cfg["batch_size"]
    refs = refs_by_epoch[0]
    key_lo, key_hi = refs["key_lo"], refs["key_hi"]
    key_plen, last_plen = refs["key_plen"], refs["last_plen"]
    max_steps = cfg["max_steps"]
    shadow, use_fm = bool(cfg["shadow"]), bool(cfg["fm"])
    span = key_hi - key_lo

    # ======================= NEW IN THIS FORK: the drifting world ==================
    # The grammar itself moves, permanently and cumulatively, at `drift_steps`. Every
    # epoch therefore needs its own eval set, probe set and exact-BP references; all of
    # them are derived from the SAME seeds in every worker, so every arm lives in
    # literally the same world at literally the same step.
    base_rules = generate_rules_distinct(v, s, L, m, seed=cfg["rule_seed"])
    dsteps = DR.drift_steps(cfg["drift_steps"], max_steps)
    rules_traj, drift_changes = DR.rules_trajectory(
        base_rules, L, cfg["drift_level"], cfg["drift_cells"], dsteps, cfg["drift_seed"])
    n_epoch = len(rules_traj)
    assert len(refs_by_epoch) == n_epoch, "one refs dict per epoch"

    anc = {"key": W.anc_index(key_plen, L, s), "last": W.anc_index(last_plen, L, s)}
    EP = []
    for e in range(n_epoch):
        r_e = rules_traj[e]
        el, elf, _ = _generate_with_traces(r_e, cfg["n_eval"], cfg["eval_seed"])
        pl, plf, _ = _generate_with_traces(r_e, cfg["n_probe"], cfg["probe_seed"])
        ez = elf[key_level][:, key_node]
        zg = [np.where(ez == a)[0] for a in range(v)]
        EP.append({
            "rules": r_e, "refs": refs_by_epoch[e],
            "ev_leaf": el, "ev_z": ez,
            "ev_leaf_t": torch.from_numpy(el.astype(np.int64)),
            "pr_leaf": pl, "pr_z": plf[key_level][:, key_node],
            "pr_leaf_t": torch.from_numpy(pl.astype(np.int64)),
            "y_lvl": {a: {ell: torch.from_numpy(
                plf[ell][:, anc[a][ell]].astype(np.int64)).to(device)
                for ell in range(L)} for a in anc},
            "bayes_none": np.array(refs_by_epoch[e]["bayes_none"]),
            "ptl": np.array(refs_by_epoch[e]["pos_top_level"]),
            "z_groups": zg,
            "live": np.array([len(g) >= cfg["min_cell_n"] for g in zg]),
        })
    # the names the donor's instrument code closes over; rebound when the epoch turns
    epoch = 0
    rules = EP[0]["rules"]
    ev_leaf, ev_z, ev_leaf_t = EP[0]["ev_leaf"], EP[0]["ev_z"], EP[0]["ev_leaf_t"]
    pr_leaf, pr_z, pr_leaf_t = EP[0]["pr_leaf"], EP[0]["pr_z"], EP[0]["pr_leaf_t"]
    y_lvl, bayes_none, ptl = EP[0]["y_lvl"], EP[0]["bayes_none"], EP[0]["ptl"]
    z_groups, live = EP[0]["z_groups"], EP[0]["live"]
    P0_exact = np.array(refs["exact_leaf0"])
    MODES = ["true", "rand", "perm", "old", "none"]

    all_blocks = ["post_embed"] + [f"post_block{i}" for i in range(cfg["n_layer"])]
    pblocks = [b for b in cfg["probe_blocks"].split(",") if b in all_blocks] or all_blocks

    # ---- the burst schedule and the shadow ladder ----
    starts = [int(x) for x in str(cfg["burst_steps"]).split(",") if str(x).strip()]
    windows = BU.burst_windows(starts, cfg["burst_len"], max_steps)
    onsets = BU.burst_onsets(windows)
    # ONE RATE PER WINDOW. A rotation's spike grows with the scaffold's age (+0.085 at
    # s2000, +0.32 at s5000, +0.55 at s8000 on the donor), so a single fixed rate cannot
    # be magnitude-matched at three different onsets. `calib` measures the matched rate at
    # each onset's maturity and they are hard-coded here.
    rhos_c = [float(x) for x in str(cfg["burst_rho"]).split(",") if str(x).strip()]
    if len(rhos_c) == 1 and len(windows) > 1:
        rhos_c = rhos_c * len(windows)
    assert len(rhos_c) >= len(windows), "burst_rho must give one rate per window"
    rho_ref = rhos_c[0]
    ladder = [float(x) for x in str(cfg["burst_ladder"]).split(",") if str(x).strip()]

    # the panel's own (small) eval subset, and its FIXED shadow corruptions
    n_panel = min(cfg["n_panel"], ev_leaf.shape[0])
    pn_leaf = ev_leaf[:n_panel]
    pn_z = ev_z[:n_panel]
    pn_leaf_t = torch.from_numpy(pn_leaf.astype(np.int64))
    pn_burst_t, pn_burst_rate = [], []
    for i, rho in enumerate(ladder):
        cor, mask = BU.corrupt(pn_leaf, key_lo, key_hi, rho, v, BU.shadow_burst_rng(
            cfg["eval_seed"], i))
        pn_burst_t.append(torch.from_numpy(cor.astype(np.int64)))
        pn_burst_rate.append(float(mask.mean()))
    # the consumed rate's own shadow copy (index -1 in the ladder namespace)
    cor_c, mask_c = BU.corrupt(pn_leaf, key_lo, key_hi, rho_ref, v,
                               BU.shadow_burst_rng(cfg["eval_seed"], 991))
    pn_bc_t = torch.from_numpy(cor_c.astype(np.int64))

    # ============ NEW IN THIS FORK: the counterfactual DRIFT ladder ============
    # A drift counterfactual cannot reuse the same sequences (the grammar changed), so
    # each rung is a FRESH draw from a drifted grammar at a fixed sequence seed, and every
    # rung is read against a NULL RUNG — a fresh draw from THIS epoch's own undrifted
    # grammar at the same seed. That subtracts ordinary draw-to-draw variance, which the
    # burst ladder never had to worry about.
    dladder = [int(x) for x in str(cfg["drift_ladder"]).split(",") if str(x).strip()]
    DPAN = []
    for e in range(n_epoch):
        nl, nlf, _ = _generate_with_traces(EP[e]["rules"], n_panel, cfg["drift_eval_seed"])
        rung = {"null": (torch.from_numpy(nl.astype(np.int64)), nl,
                         nlf[key_level][:, key_node]), "rungs": []}
        for i, rl in enumerate(DR.ladder_rules(EP[e]["rules"], L, cfg["drift_level"],
                                               dladder, cfg["drift_seed"], e)):
            gl, glf, _ = _generate_with_traces(rl, n_panel, cfg["drift_eval_seed"])
            rung["rungs"].append((torch.from_numpy(gl.astype(np.int64)), gl,
                                  glf[key_level][:, key_node],
                                  DR.cell_survival(EP[e]["rules"], rl, L,
                                                   cfg["drift_level"])))
        DPAN.append(rung)

    # --------------------------------------------------------------------- #
    # helpers (the donor's, lifted so several co-resident readers can share them)
    # --------------------------------------------------------------------- #

    def make_x(leaf_t, wc):
        return torch.cat([wc[:, None], leaf_t[:, :-1]], 1)

    def wcol(kind, seed_off, leaf_np, z_np, q, q_prev, mode, mode_i):
        """The donor's `wcol`, verbatim in behaviour."""
        if mode == "none":
            return torch.full((leaf_np.shape[0],), W.neutral_tok(v), dtype=torch.long)
        rng = np.random.default_rng(cfg["eval_seed"] + 7717 * (mode_i + 1)
                                    + 131 * seed_off)
        w = W.wall_values(kind, z_np, q, v, rng=rng, mode=mode,
                          q_prev=q_prev, delta=rot_step)
        if w is None:
            return torch.full((leaf_np.shape[0],), W.neutral_tok(v), dtype=torch.long)
        return torch.from_numpy((W.wall_tok(w, v)).astype(np.int64))

    @torch.no_grad()
    def nll_per_pos(model, leaf_t, wc, chunk=256):
        tot = torch.zeros(T, device=device)
        for i in range(0, leaf_t.shape[0], chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            y = leaf_t[i:i + chunk].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            tot += nll.sum(0)
        return (tot / leaf_t.shape[0]).cpu().numpy()

    @torch.no_grad()
    def wall_conditionals(model, kind):
        if kind == "none":
            ids = torch.full((1, 1), W.neutral_tok(v), dtype=torch.long, device=device)
        else:
            ids = (torch.arange(v, device=device) + v)[:, None]
        logits, _ = model(ids)
        p = torch.softmax(logits[:, 0, :v].float(), -1).cpu().numpy()
        return np.repeat(p, v, axis=0) if kind == "none" else p

    card_sel = torch.from_numpy(np.arange(min(cfg["n_card"], ev_leaf.shape[0])))

    @torch.no_grad()
    def transfer_and_cardinality(model, kind, chunk=256):
        """The donor's forced-transfer + MDL sweep.

        DEDUPE, exact: when `kind == "none"` every one of the 16 wall ids maps to the SAME
        neutral token, so all 16 passes are literally the same forward on the same input.
        Row 0 is computed and broadcast; every downstream number (Ei, Eo, D) is therefore
        bit-identical to the donor's 16-pass version, at 1/16 the cost. Verified by the
        fidelity gate against fwlm0/fwlm1/ey0 `no_wall`.
        """
        n = ev_leaf_t.shape[0]
        nc = card_sel.shape[0]
        Ei = np.zeros((v, v)); Eo = np.zeros((v, v))
        Pc = torch.zeros(v, nc, key_hi - key_lo, v, device=device)
        ids = [0] if kind == "none" else list(range(v))
        for i in ids:
            tok = W.neutral_tok(v) if kind == "none" else W.wall_tok(i, v)
            wc = torch.full((n,), tok, dtype=torch.long)
            si = torch.zeros(n); so = torch.zeros(n)
            for j in range(0, n, chunk):
                x = make_x(ev_leaf_t[j:j + chunk], wc[j:j + chunk]).to(device)
                y = ev_leaf_t[j:j + chunk].to(device)
                logits, _ = model(x)
                nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                      reduction="none").reshape(y.shape)
                si[j:j + chunk] = nll[:, key_lo:key_hi].mean(1).cpu()
                so[j:j + chunk] = nll[:, key_hi:].mean(1).cpu()
                lo_c, hi_c = j, min(j + chunk, nc)
                if lo_c < nc:
                    Pc[i, lo_c:hi_c] = torch.softmax(
                        logits[:hi_c - lo_c, key_lo:key_hi, :v].float(), -1)
            sin, son = si.numpy(), so.numpy()
            for a in range(v):
                g = z_groups[a]
                if len(g):
                    Ei[i, a] = sin[g].mean(); Eo[i, a] = son[g].mean()
        if kind == "none":
            for i in range(1, v):
                Ei[i] = Ei[0]; Eo[i] = Eo[0]; Pc[i] = Pc[0]
        D = np.zeros((v, v))
        lP = torch.log(Pc.clamp_min(1e-12))
        for i in range(v):
            M = 0.5 * (Pc[i][None] + Pc)
            lM = torch.log(M.clamp_min(1e-12))
            d = 0.5 * ((Pc[i][None] * (lP[i][None] - lM)).sum(-1)
                       + (Pc * (lP - lM)).sum(-1))
            D[i] = d.mean(dim=(1, 2)).cpu().numpy()
        return Ei, Eo, np.maximum(D, 0.0)

    def acts_at(model, leaf_t, wc, pos, blocks, chunk=256):
        out = {b: [] for b in blocks}
        with torch.no_grad():
            for i in range(0, leaf_t.shape[0], chunk):
                x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
                _, _, inter = model(x, return_intermediates=True)
                for b in blocks:
                    out[b].append(inter[b][:, pos, :].float())
        return {b: torch.cat(vs) for b, vs in out.items()}

    def probe_levels(model, wc, anchor, pos, blocks, full):
        a = acts_at(model, pr_leaf_t, wc, pos, blocks)
        res = {}
        for ell in range(L):
            best = 0.0
            for b in blocks:
                best = max(best, _probe_acc(a[b], y_lvl[anchor][ell], v, device,
                                            cfg["probe_steps"], cfg["probe_lr"]))
                if full:
                    best = max(best, _probe_acc(a[b], y_lvl[anchor][ell], v, device,
                                                cfg["mlp_steps"], cfg["probe_lr"],
                                                hidden=cfg["mlp_hidden"]))
            res[f"d{L - ell}"] = float(best)
        return res

    # ---------------- the new panel's primitives ---------------- #

    @torch.no_grad()
    def panel_read(model, leaf_t, wc, chunk=256):
        """(nll_idx, nll_out, probs (n, span, V)) in one pass."""
        n = leaf_t.shape[0]
        P = torch.empty(n, span, V, device=device)
        si = torch.zeros(n); so = torch.zeros(n)
        for i in range(0, n, chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            y = leaf_t[i:i + chunk].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            si[i:i + chunk] = nll[:, key_lo:key_hi].mean(1).cpu()
            so[i:i + chunk] = nll[:, key_hi:].mean(1).cpu()
            P[i:i + chunk] = torch.softmax(logits[:, key_lo:key_hi, :].float(), -1)
        return float(si.mean()), float(so.mean()), P

    def kl(P, Q):
        lp = torch.log(P.clamp_min(1e-12))
        lq = torch.log(Q.clamp_min(1e-12))
        return float((P * (lp - lq)).sum(-1).mean())

    def jsd(P, Q):
        M = 0.5 * (P + Q)
        return float(0.5 * (kl(P, M) + kl(Q, M)))

    @torch.no_grad()
    def fm_residual(model, fms, leaf_t, wc, chunk=256):
        """{name: (mse, rel)} — || h6 - FM(h0) ||^2 on the indexed span, and the same
        divided by mean h6^2 (the reader's activation scale drifts during training, so a
        raw residual that rose only because ||h6|| grew would be a confound)."""
        acc = {k: 0.0 for k in fms}
        tgt, cnt = 0.0, 0
        for i in range(0, leaf_t.shape[0], chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            _, _, inter = model(x, return_intermediates=True)
            h0 = inter[cfg["fm_shallow"]]
            h6 = inter[cfg["fm_deep"]][:, key_lo:key_hi, :]
            for k, fm in fms.items():
                pred = fm(h0)[:, key_lo:key_hi, :]
                acc[k] += float(((pred - h6) ** 2).sum())
            tgt += float((h6 ** 2).sum())
            cnt += h6.numel()
        return {k: {"mse": acc[k] / cnt, "rel": acc[k] / max(tgt, 1e-12)} for k in acc}

    # ============ NEW IN THIS FORK: the ONLINE typing instrument and op map ======

    dz = {"key": float(cfg["dz_key"]), "self": float(cfg["dz_self"]),
          "out": float(cfg["dz_out"])}
    pol_a = 1.0 - 0.5 ** (1.0 / max(cfg["pol_halflife"], 1e-9))

    @torch.no_grad()
    def buffer_reads(rd, use_buf=None, chunk=256):
        """(r_key, r_self, r_out) on the batches the arm was just OFFERED.

        r_key   indexed-span NLL under the arm's OWN consumed wall condition
        r_self  indexed-span NLL with the key neutralised — the second basis, from the
                same model. Under a rotation this is invariant BY CONSTRUCTION (its input
                is unchanged), which is exactly what makes `d_self ~ 0` type a rotation.
        r_out   NLL outside the indexed span, under the consumed condition
        """
        buf = rd.buf if use_buf is None else use_buf
        leaf = torch.cat([b[0] for b in buf])
        n = leaf.shape[0]
        if rd.kind == "none" or rd.key_off or use_buf is not None:
            wc = torch.full((n,), W.neutral_tok(v), dtype=torch.long)
        else:
            wc = torch.cat([b[1] for b in buf])
        wn = torch.full((n,), W.neutral_tok(v), dtype=torch.long)
        acc = {}
        for tag_, col in (("key", wc), ("self", wn)):
            si = so_ = 0.0
            for i in range(0, n, chunk):
                x = make_x(leaf[i:i + chunk], col[i:i + chunk]).to(device)
                y = leaf[i:i + chunk].to(device)
                logits, _ = rd.model(x)
                nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                      reduction="none").reshape(y.shape)
                si += float(nll[:, key_lo:key_hi].sum())
                so_ += float(nll[:, key_hi:].sum())
            acc[tag_] = (si / (n * span), so_ / (n * (T - key_hi)))
        return acc["key"][0], acc["self"][0], acc["key"][1]

    def op_map(rd, d, step):
        """Instant typing -> op. Thermostat-grade, dead zones at measured floors.

        The default map (SPEC): rotation -> merge, burst -> skip, drift -> track.
        `merge` is LATCHED (the donor's op is a permanent edit to the input stream and
        fwlm1 finding 5 says the abandoned circuitry evaporates within ~250 steps, so it
        is irreversible in effect whatever the mechanism); `skip` is WINDOWED.

        A rotation is required to clear BOTH `d_self ~ 0` AND `d_out > floor` before the
        irreversible op fires. `d_self ~ 0` alone would suffice on Gate 0's numbers, but
        it would also fire on any spurious `d_key` spike, and a false merge is a permanent
        loss of the index. The conjunction is the conservative reading; `--no-rot-requires-out`
        runs the liberal one.
        """
        if rd.policy == "sched":
            return ("merge" if step >= cfg["sched_merge_at"] else "continue"), "sched"
        if rd.policy == "rand":
            if step == rand_merge:
                return "merge", "rand"
            return ("skip" if step in rand_skips else "continue"), "rand"
        if d["key"] <= dz["key"]:
            return "continue", "none"
        if rd.policy == "sur_merge":
            return "merge", "spike"
        if rd.policy == "sur_skip":
            return "skip", "spike"
        if rd.policy == "sur_fm":
            # no counterfactual read at all: buy one window of `continue` and let the FM
            # sign decide next checkpoint (handled in the policy block)
            return "continue", "spike_await_fm"
        hot_self, hot_out = d["self"] > dz["self"], d["out"] > dz["out"]
        if not hot_self:
            if hot_out or not cfg["rot_requires_out"]:
                typ = "rotation"
                if rd.policy == "tstar_rekey":
                    return "rekey", typ
                if rd.policy == "tstar_abba":
                    return "abba", typ
                return "merge", typ
            return "continue", "ambiguous"
        return ("continue", "drift") if hot_out else ("skip", "burst")

    @torch.no_grad()
    def rekey_search(rd, chunk=256):
        """Priced candidate search over the rotation family, on the OBSERVED token.

        A re-key by r feeds `wall_tok((w_obs + r) mod v)`. `w_obs` is what the arm actually
        sees, so no latent is consulted and the op leaks no oracle — the arm is choosing the
        permutation that minimises one window's own NLL, and paying `rekey_search` forward
        passes for the privilege. Returns (best_shift, nll_by_shift).
        """
        leaf = torch.cat([b[0] for b in rd.buf]); wc = torch.cat([b[1] for b in rd.buf])
        n = leaf.shape[0]
        w_obs = (wc - v).clamp_min(0)
        out = []
        for r in range(int(cfg["rekey_search"])):
            col = W.wall_tok(((w_obs + r) % v).numpy(), v)
            col = torch.from_numpy(col.astype(np.int64))
            tot = 0.0
            for i in range(0, n, chunk):
                x = make_x(leaf[i:i + chunk], col[i:i + chunk]).to(device)
                y = leaf[i:i + chunk].to(device)
                logits, _ = rd.model(x)
                nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                      reduction="none").reshape(y.shape)
                tot += float(nll[:, key_lo:key_hi].sum())
            out.append(tot / (n * span))
        rd.spent += cfg["rekey_search"] * cfg["price_pass"]
        return int(np.argmin(out)), out

    def abba_step(rd, step, reads):
        """teacher_slot's C K K C paired trial, deciding a merge instead of firing one.

        Two collapse quarters and two keep quarters, ordered so a linear learning trend
        cancels exactly. The credited quantity is the arm's own indexed-span NLL under the
        NEUTRAL condition — the pathway currency, i.e. the counterfactual gauge, not the
        experienced loss. Returns the op for the coming window.
        """
        a = rd.abba
        if a is None:
            return None
        a["reads"].append(reads["self"])
        q_i = len(a["reads"]) - 1
        if q_i < 4:
            return "merge_trial" if a["order"][q_i] == "C" else "continue"
        cs = [a["reads"][i + 1] - a["reads"][i] for i in range(3)]
        gain_c = np.mean([cs[i] for i in range(3) if a["order"][i] == "C"])
        gain_k = np.mean([cs[i] for i in range(3) if a["order"][i] == "K"])
        vstat = float(gain_k - gain_c)          # >0 => collapsing improved the gauge more
        a["V"] = vstat
        rd.abba_log.append({"step": step, "V": vstat, "order": "".join(a["order"]),
                            "reads": a["reads"]})
        rd.abba = None
        rd.spent += 4 * 2 * cfg["price_pass"]
        if vstat > cfg["abba_v_tol"]:
            print(f"  [abba] {rd.name} s{step}: V {vstat:+.4f} > tol -> MERGE", flush=True)
            return "merge"
        print(f"  [abba] {rd.name} s{step}: V {vstat:+.4f} <= tol -> keep", flush=True)
        return "continue"

    def fm_grade(rd, step, dg, don):
        """The FM as GRADER, not typer — reverting only what is revertible.

        `dg` is the online-vs-frozen LAG statistic's own delta (wave 1: quiet p90 0.548,
        burst peaks +1.31/+5.17/+8.46): a large positive value means a real input-side shock
        just landed. `don` is the ONLINE residual's delta (Gate 0's sign result).

        A merge is never undone (fwlm1 finding 5). Only two errors are recoverable:
          a SKIP with no input-side shock behind it  -> the arm skipped a rotation, revert;
          a CONTINUE with a large shock behind it    -> the arm consumed a burst, switch.
        """
        if rd.pending is None:
            return None
        typ, op = rd.pending
        rd.pending = None
        if op == "skip" and dg is not None and dg < cfg["fm_lag_thresh"]:
            rd.reverts.append({"step": step, "from": "skip", "to": "continue",
                               "dg": dg, "typed": typ})
            return "continue"
        if op == "continue" and dg is not None and dg > cfg["fm_lag_thresh"]:
            rd.reverts.append({"step": step, "from": "continue", "to": "skip",
                               "dg": dg, "typed": typ})
            return "skip"
        return None

    def save_ckpt(rd, step):
        """fp16 state dicts for the reader and every FM, so the hidden-state geometry
        reads (RSA / Procrustes) never need a replay. Bounded by (a) fp16, (b) a stride,
        (c) only the arms whose geometry is load-bearing."""
        sd = {k: t.detach().to(torch.float16).cpu()
              for k, t in rd.model.state_dict().items()}
        blob = {"step": step, "epoch": epoch, "arm": rd.name, "dtype": "float16",
                "model": sd,
                "fms": {k: {kk: t.detach().to(torch.float16).cpu()
                            for kk, t in fm.state_dict().items()}
                        for k, fm in rd.fms.items()}}
        torch.save(blob, os.path.join(ckdir, f"{rd.name}_s{step:06d}.pt"))

    # --------------------------------------------------------------------- #
    # the readers
    # --------------------------------------------------------------------- #

    class Reader:
        """One arm. Carries its OWN global-RNG state so co-residency is invisible."""

        def __init__(self, name, sched):
            self.name = name
            self.sched = sched
            self.kind = sched["kind"]
            self.phase1, self.rot_period = sched["phase1"], sched["rot_period"]
            self.seed_off = sched["seed_off"]
            self.merge_at = sched["merge_at"]
            self.policy = sched["policy"]
            self.instr = sched["instr"]
            self.donor_panel = sched["instr"] in ("full", "base")
            self.full_panel = sched["instr"] == "full"
            # ---- NEW IN THIS FORK: the online typing state ----
            self.buf = []            # the last n_buf OFFERED batches (learned from or not)
            self.op = "continue"     # the op in force for the window about to be trained
            self.key_off = False     # latched by `merge`
            self.merged_at = None
            self.pol = {}            # EWMA of the policy's own reads
            # ---- NEW IN WAVE 2 ----
            self.rekey = 0           # cyclic shift applied to the OBSERVED wall token
            self.rekeys = []         # (step, shift_applied, total)
            self.spent = 0.0         # priced reads, in step-equivalents
            self.charged = sched["policy"] in ("tstar_abba", "tstar_rekey", "rekey_dead")
            self.pending = None      # an op awaiting the FM's grade
            self.reverts = []        # (step, from_op, to_op, statistic)
            self.abba = None         # the running paired trial
            self.abba_log = []
            self.terminal_step = None
            self.ops = []            # the decision trace
            self.n_ops = {"merge": 0, "skip": 0}
            self.seed = cfg["seed"] + self.seed_off
            ambient = torch.get_rng_state()
            torch.manual_seed(self.seed)
            self.model = GPT(V, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
            self.rng_state = torch.get_rng_state()      # exactly the donor's stream position
            torch.set_rng_state(ambient)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                         weight_decay=cfg["weight_decay"])
            self.gen = torch.Generator().manual_seed(self.seed)
            self.train_rng = np.random.default_rng(self.seed + 5551)
            self.log = []
            # --- the FMs: constructed OUTSIDE the reader's stream, RNG-neutral ---
            self.fms, self.fopts = {}, {}
            self.fm_prev, self.fm_event = None, None
            if use_fm and sched["n_fm"] > 0:
                lrs = FM_LRS[:sched["n_fm"]] if sched["n_fm"] > 1 else (FM_PRIMARY,)
                if FM_PRIMARY not in lrs:
                    lrs = tuple(lrs) + (FM_PRIMARY,)
                amb = torch.get_rng_state()
                proto = TransformerForwardModel(
                    d_model=cfg["n_embd"], d_head=cfg["fm_d_head"],
                    n_head=cfg["fm_n_head"], n_layer=cfg["fm_n_layer"],
                    mlp_mult=cfg["fm_mlp_mult"], block_size=T).to(device)
                torch.set_rng_state(amb)                # zero net global RNG consumed
                for lr in lrs:
                    k = f"lr{lr:g}"
                    self.fms[k] = copy.deepcopy(proto)  # identical init; lr is the only lever
                    self.fopts[k] = torch.optim.AdamW(self.fms[k].parameters(), lr=lr,
                                                      weight_decay=cfg["weight_decay"])
                del proto
                self.primary = f"lr{FM_PRIMARY:g}"
                self.fm_prev = copy.deepcopy(self.fms[self.primary])
                self.fm_event = copy.deepcopy(self.fms[self.primary])
            self.has_fm = bool(self.fms)
            self.bench = {}                              # EWMA benchmarks b_t

        def rng_in(self):
            self._amb = torch.get_rng_state()
            torch.set_rng_state(self.rng_state)

        def rng_out(self):
            self.rng_state = torch.get_rng_state()
            torch.set_rng_state(self._amb)

    names = [n for n in WORKER_SPECS[worker]]
    readers = {n: Reader(n, scheds[n]) for n in names}
    keyed = next((r for r in readers.values() if r.kind != "none"), None)
    parse = next((r for r in readers.values() if r.kind == "none"), None)

    # ---- checkpoint grids ----
    rots = {n: W.rotation_steps(r.phase1, r.rot_period, max_steps)
            for n, r in readers.items()}
    cheap = set(range(0, max_steps + 1, cfg["ckpt_every"])) | {max_steps}
    for n, r in readers.items():
        if r.rot_period >= 4 * cfg["post_rot_b"]:
            for x in rots[n]:
                cheap |= {x + d for d in (cfg["post_rot_a"], cfg["post_rot_b"])
                          if x + d <= max_steps}
    for a, b in windows:                       # resolve each burst's own transient
        cheap |= {a + d for d in (0, 25, 50, 75) if a + d <= max_steps}
        cheap |= {b + d for d in (0, 25, 50, 75) if b + d <= max_steps}
    for a in dsteps:                           # NEW IN WAVE 2: and each drift's
        cheap |= {a + d for d in (0, 25, 50, 75) if a + d <= max_steps}
    cheap = sorted(cheap)
    index_ck = sorted(set(range(0, max_steps + 1, cfg["index_every"])) | {max_steps})
    probe_ck = sorted({int(x) for x in cfg["probe_ckpts"].split(",")
                       if x.strip() and int(x) <= max_steps} | {max_steps})
    full_ck = {int(x) for x in cfg["full_ckpts"].split(",")
               if x.strip() and int(x) <= max_steps}
    if cfg.get("full_at_end", True):
        full_ck |= {max_steps}

    # the `rand` control's schedule: the same op COUNTS as the treatments are expected to
    # fire, on seeded random checkpoints. Drawn from the design rate (1 merge + `rand_n_skip`
    # skips) because the treatments' realised rate is not knowable before they run; the
    # reduction reports both so the mismatch is visible rather than assumed away.
    # candidates start at the FIRST EVENT, for two reasons: `rand` then fires ops in the
    # same epochs the treatments do (a fairer null than firing them during phase 1), and
    # its pre-event prefix stays bit-identical to every other arm's, which is what the
    # fidelity gate needs.
    _first_event = min([x for x in (onsets + dsteps + rots.get(names[0], []))] or [0])
    rand_rng = np.random.default_rng(cfg["rand_seed"])
    _cand = [c for c in cheap if c >= _first_event]
    rand_merge = int(rand_rng.choice(_cand)) if _cand else -1
    rand_skips = set(int(x) for x in rand_rng.choice(
        _cand, size=min(cfg["rand_n_skip"], len(_cand)), replace=False)) if _cand else set()

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    ckdir = os.path.join(outdir, "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    # WHICH checkpoints get a torch.save, and for whom. fp16 halves it; the stride keeps
    # the strided coverage cheap; the event-dense points put the resolution exactly where
    # a reorientation would show up. Only the arms whose geometry is load-bearing.
    save_steps = set(cheap[::cfg["ckpt_stride"]]) | {max_steps}
    for _e in sorted(set(list(rots.get(names[0], [])) + onsets + dsteps)):
        save_steps |= {_e + d for d in (0, 25, 125) if _e + d <= max_steps}
    save_steps = {x for x in save_steps if x in set(cheap)}
    save_arms = set(str(cfg["ckpt_arms"]).split(",")) if cfg["ckpt_arms"] else set()
    print(f"===== worker {worker}  readers {names}  bursts {windows} rho={rhos_c} "
          f"ladder={ladder}  shadow={shadow} fm={use_fm} =====", flush=True)

    fm_pending_op = {}                 # arm -> the op the FM's grade installs next window
    panel_log, started = [], time.time()
    pool_leaf = pool_z = None
    ew_a = 1.0 - 0.5 ** (1.0 / max(cfg["bench_halflife"], 1e-9))    # EWMA weight on the new e

    for step in range(max_steps + 1):
        # ==== NEW IN WAVE 2: the epoch turns BEFORE the checkpoint ====
        # Wave 1 turned it after, so a drift was first observed 125 steps late and this
        # reader had already absorbed ~95% of it. Turning first gives the drift the same
        # zero-lag read a rotation gets from the eval wall column.
        new_epoch = DR.epoch_of(step, dsteps)
        if new_epoch != epoch:
            epoch = new_epoch
            rules = EP[epoch]["rules"]
            ev_leaf, ev_z, ev_leaf_t = (EP[epoch]["ev_leaf"], EP[epoch]["ev_z"],
                                        EP[epoch]["ev_leaf_t"])
            pr_leaf, pr_z, pr_leaf_t = (EP[epoch]["pr_leaf"], EP[epoch]["pr_z"],
                                        EP[epoch]["pr_leaf_t"])
            y_lvl, bayes_none, ptl = (EP[epoch]["y_lvl"], EP[epoch]["bayes_none"],
                                      EP[epoch]["ptl"])
            z_groups, live = EP[epoch]["z_groups"], EP[epoch]["live"]
            refs = EP[epoch]["refs"]
            pool_leaf = None
            print(f"  [world] step {step}: DRIFT -> epoch {epoch} (cell survival vs base "
                  f"{DR.cell_survival(EP[0]['rules'], rules, L, cfg['drift_level']):.4f})",
                  flush=True)
        if step in cheap:
            t_ck = time.time()
            do_index = step in index_ck
            do_probe = step in probe_ck
            bi = BU.burst_index(step, windows)

            # ------------- the donor's per-reader record (bit-identical) -------------
            for n, rd in readers.items():
                model = rd.model
                model.eval()
                q = W.q_at(step, rd.phase1, rd.rot_period, rot_step, v)
                qp = W.q_prev_at(step, rd.phase1, rd.rot_period, rot_step, v)
                merged = rd.key_off
                rec = {"step": step, "tokens": step * B * T, "q": int(q), "q_prev": int(qp),
                       "era": int(W.era_at(step, rd.phase1, rd.rot_period)),
                       "merged": bool(merged),
                       "consumed": "none" if (rd.key_off or rd.kind == "none") else "true",
                       "burst": int(bi), "epoch": int(epoch),
                       "op": rd.op, "key_off": bool(rd.key_off),
                       "skipped": rd.op == "skip"}
                npos = {}
                if rd.kind == "none":
                    # every mode's wall column is the SAME neutral token: one pass, exact
                    base = nll_per_pos(model, ev_leaf_t,
                                       wcol(rd.kind, rd.seed_off, ev_leaf, ev_z, q, qp,
                                            "none", MODES.index("none")))
                    for mo in MODES:
                        npos[mo] = base
                else:
                    for mi, mode in enumerate(MODES):
                        # the arm's re-key is a permutation of the OBSERVED token, so it
                        # shifts the CONSUMED condition only; `rand`/`perm`/`old`/`none`
                        # keep the donor's definitions so the instruments stay comparable
                        qq = q + rd.rekey if mode == "true" else q
                        npos[mode] = nll_per_pos(model, ev_leaf_t,
                                                 wcol(rd.kind, rd.seed_off, ev_leaf, ev_z,
                                                      qq, qp, mode, mi))
                rec["nll"] = {
                    mo: {"idx": float(npos[mo][key_lo:key_hi].mean()),
                         "out": float(npos[mo][key_hi:].mean()),
                         "all": float(npos[mo].mean()),
                         "pos0": float(npos[mo][0])} for mo in MODES}
                rec["excess"] = {
                    mo: {"idx": float((npos[mo] - bayes_none)[key_lo:key_hi].mean()),
                         "out": float((npos[mo] - bayes_none)[key_hi:].mean()),
                         "all": float((npos[mo] - bayes_none).mean())} for mo in MODES}
                rec["excess_by_level"] = {
                    str(int(k)): float((npos["true"] - bayes_none)[ptl == k].mean())
                    for k in np.unique(ptl)}
                rec["excess_by_level_rand"] = {
                    str(int(k)): float((npos["rand"] - bayes_none)[ptl == k].mean())
                    for k in np.unique(ptl)}
                rec["nll_pos_true"] = npos["true"].tolist()
                rec["binding"] = {
                    "idx": float(npos["rand"][key_lo:key_hi].mean()
                                 - npos["true"][key_lo:key_hi].mean()),
                    "out": float(npos["rand"][key_hi:].mean() - npos["true"][key_hi:].mean()),
                    "pos0": float(npos["rand"][0] - npos["true"][0]),
                    "misleading_idx": float(npos["perm"][key_lo:key_hi].mean()
                                            - npos["true"][key_lo:key_hi].mean()),
                    "stale_idx": float(npos["true"][key_lo:key_hi].mean()
                                       - npos["old"][key_lo:key_hi].mean()),
                    "vs_none_idx": float(npos["none"][key_lo:key_hi].mean()
                                         - npos["true"][key_lo:key_hi].mean()),
                    "vs_none_out": float(npos["none"][key_hi:].mean()
                                         - npos["true"][key_hi:].mean()),
                }
                if rd.donor_panel and do_index:
                    Ei, Eo, D = transfer_and_cardinality(model, rd.kind)
                    off = ~np.eye(v, dtype=bool)
                    ar = np.arange(v)
                    implied = Ei.argmin(0)
                    cur, old, ph1 = (ar + q) % v, (ar + qp) % v, ar
                    diag_cur, diag_old = Ei[cur, ar], Ei[old, ar]
                    offmean = np.array([Ei[[i for i in range(v) if i != cur[a]], a].mean()
                                        for a in range(v)])
                    offmean_o = np.array([Eo[[i for i in range(v) if i != cur[a]], a].mean()
                                          for a in range(v)])
                    rec["index"] = {
                        "n_live": int(live.sum()),
                        "transfer_gap": float((offmean - diag_cur)[live].mean()),
                        "e_cur": float(diag_cur[live].mean()),
                        "e_old": float(diag_old[live].mean()),
                        "e_off": float(offmean[live].mean()),
                        "e_best": float(Ei.min(0)[live].mean()),
                        "map_match_cur": float((implied == cur)[live].mean()),
                        "map_match_old": float((implied == old)[live].mean()),
                        "map_match_phase1": float((implied == ph1)[live].mean()),
                        "match_cur_per_latent": (implied == cur).astype(int).tolist(),
                        "adv_cur_per_latent": (offmean - diag_cur).tolist(),
                        "out_span_transfer_gap": float((offmean_o - Eo[cur, ar])[live].mean()),
                        "jsd_mean": float(D[off].mean()),
                        "card": {f"tau{t}": int(W.cluster_count(D, t))
                                 for t in (0.0002, 0.001, 0.005, 0.02)},
                        "E_idx": Ei.tolist(),
                    }
                    Pm = wall_conditionals(model, rd.kind)
                    D0m = W.jsd_matrix(Pm)
                    rec["index"]["jsd0_mean"] = float(D0m[off].mean())
                    rec["index"]["jsd0_ratio"] = float(
                        D0m[off].mean() / max(refs["exact_jsd_mean"], 1e-12))
                if rd.donor_panel and do_probe:
                    full = step in full_ck
                    blocks = all_blocks if full else pblocks
                    rd.rng_in()
                    lv = {}
                    for mode in ("true", "rand", "none"):
                        wc = wcol(rd.kind, rd.seed_off, pr_leaf, pr_z, q, qp,
                                  mode, MODES.index(mode))
                        lv[mode] = {
                            "key": probe_levels(model, wc, "key", key_plen, blocks, full),
                            "last": probe_levels(model, wc, "last", last_plen, blocks, full)}
                    rd.rng_out()
                    rec["levels"], rec["full_probe"] = lv, bool(full)
                rd.log.append(rec)
                with open(os.path.join(outdir, f"{n}.json"), "w") as fh:
                    json.dump({"arm": n, "sched": rd.sched, "log": rd.log,
                               "ops": rd.ops, "n_ops": rd.n_ops,
                               "merged_at": rd.merged_at, "rekeys": rd.rekeys,
                               "reverts": rd.reverts, "abba": rd.abba_log,
                               "spent": rd.spent, "terminal_step": rd.terminal_step,
                               "complete": step == max_steps}, fh, indent=2,
                              cls=NumpyEncoder)
                if n in save_arms and step in save_steps:
                    save_ckpt(rd, step)

            t_donor = time.time() - t_ck
            # ============ NEW IN THIS FORK: the ONLINE TYPING READ + op map ============
            # Read on the batches the arm was just OFFERED, not on a fixed clean eval set:
            # that is the only information an online learner actually has, and it is what
            # makes a burst (corrupted leaves in the stream) visible to a key-free view at
            # all. Gate 0's counterfactual panel keeps running beside it, uncharged, so the
            # online read can be graded against the offline one afterwards.
            for n, rd in readers.items():
                if rd.policy == "rekey_dead":
                    src = next((x for x in readers.values()
                                if x.policy == "tstar_rekey"), None)
                    if src is not None and src.rekey != rd.rekey:
                        rd.rekey = src.rekey
                        rd.rekeys.append({"step": step, "total": rd.rekey,
                                          "mirrored_from": src.name})
                    rd.ops.append({"step": step, "op": "continue", "type": "mirror",
                                   "rekey": rd.rekey})
                    continue
                if rd.policy == "none" or not rd.buf:
                    rd.ops.append({"step": step, "op": rd.op, "type": "none"})
                    continue
                r_key, r_self, r_out = buffer_reads(rd)
                rd.spent += 2 * cfg["price_pass"]
                reads = {"key": r_key, "self": r_self, "out": r_out}
                if rd.policy == "tstar_pair" and parse is not None:
                    reads["self"] = buffer_reads(parse, use_buf=rd.buf)[0]
                d = {}
                for k, val in reads.items():
                    b = rd.pol.get(k)
                    d[k] = 0.0 if b is None else val - b
                    rd.pol[k] = val if b is None else (1 - pol_a) * b + pol_a * val
                op, typ = op_map(rd, d, step)
                # ---- NEW IN WAVE 2: re-key, ABBA, and the FM's grade ----
                if rd.abba is not None:                    # a trial is already running
                    aop = abba_step(rd, step, reads)
                    if aop == "merge_trial":
                        rd.key_off = True; op, typ = "continue", "abba_C"
                    elif aop == "continue" and rd.abba is not None:
                        rd.key_off = False; op, typ = "continue", "abba_K"
                    elif aop == "merge":
                        # the last C quarter already left key_off True, so the generic
                        # merge handler below would not fire and the latch would go
                        # unlogged; latch and record it here (caught in the wave-2
                        # path-coverage smoke)
                        if rd.merged_at is None:
                            rd.merged_at = step
                            rd.n_ops["merge"] += 1
                            print(f"  [op] {n} s{step}: abba trial -> MERGE (latched)",
                                  flush=True)
                        rd.key_off = True
                        op, typ = "continue", "abba_merge"
                    else:
                        rd.key_off = False; op, typ = "continue", "abba_done"
                elif op == "abba":
                    if rd.key_off:                 # already merged: nothing left to trial
                        op, typ = "continue", "abba_moot"
                    else:
                        rd.abba = {"reads": [], "order": list("CKKC"), "start": step}
                        rd.key_off = True
                        op, typ = "continue", "abba_start"
                elif op == "rekey":
                    r, curve = rekey_search(rd)
                    # FIX (wave 2 post-mortem): persist the curve on EVERY search, not only
                    # when it changes state. Wave 2's search declined at every rotation
                    # (r = 0 twice, no detection the other four times) and left no record
                    # of what it saw when it declined — which is the diagnostic.
                    rd.rekeys.append({"step": step, "shift": r, "total": (rd.rekey + r) % v,
                                      "applied": bool(r != 0), "nll_by_shift": curve})
                    if r != 0:
                        rd.rekey = (rd.rekey + r) % v
                        print(f"  [rekey] {n} s{step}: +{r} -> total {rd.rekey} "
                              f"(nll {min(curve):.4f} vs {curve[0]:.4f} unshifted)",
                              flush=True)
                    op, typ = "continue", "rekey"
                if rd.policy in ("tstar_fm", "sur_fm"):
                    # the FM grades THE OP TAKEN IN RESPONSE TO AN EVENT. Handing it the
                    # default `continue` on a quiet checkpoint let a high lag statistic
                    # install a skip out of nowhere (caught in the wave-2 smoke).
                    if typ not in ("none", "ambiguous", "mirror"):
                        rd.pending = (typ, op)
                    if fm_pending_op.get(n) is not None:
                        op = fm_pending_op.pop(n)
                        typ = typ + "+fm"
                rd.op = op
                rd.ops.append({"step": step, "op": op, "type": typ,
                               "reads": reads, "delta": d,
                               # THE TRIPLE'S HEALTH: how far apart the keyed view and the
                               # key-free view still are. Zero = the second basis is gone
                               # (what `merge` does to it); positive = the triple is alive
                               # (what `re-key` is meant to preserve).
                               "basis_sep": reads["key"] - reads["self"],
                               "rekey": rd.rekey, "spent": rd.spent})
                if op == "merge" and not rd.key_off:
                    rd.key_off, rd.merged_at = True, step
                    rd.n_ops["merge"] += 1
                    print(f"  [op] {n} s{step}: typed {typ} -> MERGE (latched)", flush=True)
                elif op == "skip":
                    rd.n_ops["skip"] += 1

            # ------------------------- the SHADOW PANEL -------------------------
            if shadow and keyed is not None and keyed.full_panel:
                t_pan = time.time()
                q = W.q_at(step, keyed.phase1, keyed.rot_period, rot_step, v) if keyed else 0
                qp = (W.q_prev_at(step, keyed.phase1, keyed.rot_period, rot_step, v)
                      if keyed else 0)
                prec = {"step": step, "q": int(q), "burst": int(bi),
                        "rho_c": rho_ref, "rhos_c": rhos_c, "ladder": ladder,
                        "ladder_realised": pn_burst_rate}
                # the wall columns the panel uses (deterministic; no RNG)
                wtrue = (wcol(keyed.kind, keyed.seed_off, pn_leaf, pn_z, q, qp, "true", 0)
                         if keyed else None)
                wperm = (wcol(keyed.kind, keyed.seed_off, pn_leaf, pn_z, q, qp, "perm", 2)
                         if keyed else None)
                wnone = torch.full((n_panel,), W.neutral_tok(v), dtype=torch.long)

                conds = [("clean", pn_leaf_t, wtrue), ("rot", pn_leaf_t, wperm)]
                for i, rho in enumerate(ladder):
                    conds.append((f"burst{i}", pn_burst_t[i], wtrue))
                conds.append(("burst_c", pn_bc_t, wtrue))
                # the drift rungs bring their OWN sequences, so their wall column has to be
                # built from that draw's own latents
                dp = DPAN[epoch]
                dt, dl_np, dz_np = dp["null"]
                conds.append(("drift_null", dt,
                              wcol(keyed.kind, keyed.seed_off, dl_np, dz_np, q, qp,
                                   "true", 0)))
                for i, (gt, gl_np, gz_np, surv) in enumerate(dp["rungs"]):
                    conds.append((f"drift{i}", gt,
                                  wcol(keyed.kind, keyed.seed_off, gl_np, gz_np, q, qp,
                                       "true", 0)))
                prec["drift_ladder"] = dladder
                prec["drift_survival"] = [x[3] for x in dp["rungs"]]
                prec["epoch"] = epoch

                # p_key-with-key-neutralised and p_parse are unchanged by the ROTATION, so
                # the clean-condition passes are shared between `clean` and `rot`.
                cache_self, cache_parse = {}, {}
                for cname, lt, wc in conds:
                    ck = "clean" if cname in ("clean", "rot") else cname
                    ki, ko, Pk = panel_read(keyed.model, lt, wc) if keyed else (0, 0, None)
                    if ck not in cache_self and keyed is not None:
                        cache_self[ck] = panel_read(keyed.model, lt, wnone)
                    if ck not in cache_parse and parse is not None:
                        cache_parse[ck] = panel_read(parse.model, lt, wnone)
                    si, so_, Ps = cache_self.get(ck, (None, None, None))
                    pi, po, Pp = cache_parse.get(ck, (None, None, None))
                    row = {"nll_key_idx": ki, "nll_key_out": ko,
                           "nll_self_idx": si, "nll_parse_idx": pi, "nll_parse_out": po}
                    if Pk is not None and Ps is not None:
                        row["D_self"] = kl(Pk, Ps)
                        row["D_self_rev"] = kl(Ps, Pk)
                        row["JS_self"] = jsd(Pk, Ps)
                    if Pk is not None and Pp is not None:
                        row["D_pair"] = kl(Pk, Pp)
                        row["D_pair_rev"] = kl(Pp, Pk)
                        row["JS_pair"] = jsd(Pk, Pp)
                    if Ps is not None and Pp is not None:
                        row["D_selfparse"] = kl(Ps, Pp)      # the mirror control: how far
                        # apart are the two PARSE bases (one hollow, one real)?
                    prec[cname] = row
                    del Pk

                prec["ms_panel"] = round(1000 * (time.time() - t_pan))
                t_fmr = time.time()
                # ---------------------------- Factor M ----------------------------
                if use_fm:
                    fmrec = {}
                    for n, rd in readers.items():
                        if not rd.has_fm:
                            continue
                        wcl = wtrue if rd.kind != "none" else wnone
                        wpm = wperm if rd.kind != "none" else wnone
                        # ONE reader forward per condition; every FM variant (the lr sweep
                        # and both frozen snapshots) rides the same activations.
                        fset = dict(rd.fms)
                        fset["frozen_prev"] = rd.fm_prev
                        fset["frozen_event"] = rd.fm_event
                        cset = {"clean": (pn_leaf_t, wcl), "burst_c": (pn_bc_t, wcl)}
                        if rd.kind != "none":
                            cset["rot"] = (pn_leaf_t, wpm)
                        e_out, bd = {}, {}
                        for cname, (lt, wc) in cset.items():
                            e_out[cname] = fm_residual(rd.model, fset, lt, wc)
                            for k, val in e_out[cname].items():
                                key = f"{cname}|{k}"
                                e = val["mse"]
                                prev = rd.bench.get(key)
                                b = e if prev is None else prev
                                bd[key] = {"b": b, "delta": b - e}
                                rd.bench[key] = (e if prev is None
                                                 else (1 - ew_a) * prev + ew_a * e)
                        fmrec[n] = {"e": e_out, "bench": bd}
                        # ---- NEW IN WAVE 2: the FM's grade for the NEXT window ----
                        if rd.policy in ("tstar_fm", "sur_fm") and rd.primary in \
                                e_out.get("clean", {}):
                            on_ = e_out["clean"][rd.primary]["rel"]
                            fz_ = e_out["clean"].get("frozen_prev", {}).get("rel")
                            g = ((fz_ - on_) / max(on_, 1e-9)) if fz_ else None
                            pb, pg = rd.pol.get("_e"), rd.pol.get("_g")
                            d_on = 0.0 if pb is None else (on_ - pb) / max(abs(pb), 1e-9)
                            d_g = None if (g is None or pg is None) else \
                                (g - pg) / max(abs(pg), 1e-9)
                            rd.pol["_e"] = on_ if pb is None else (1 - ew_a) * pb + ew_a * on_
                            if g is not None:
                                rd.pol["_g"] = g if pg is None else (1 - ew_a) * pg + ew_a * g
                            fmrec[n]["grade"] = {"d_online": d_on, "d_lag": d_g}
                            if rd.policy == "sur_fm" and rd.pending is not None:
                                typ_, _ = rd.pending; rd.pending = None
                                if d_on > cfg["fm_online_thresh"]:
                                    fm_pending_op[n] = "merge"
                                elif d_on < -cfg["fm_online_thresh"]:
                                    fm_pending_op[n] = "skip"
                            else:
                                nxt = fm_grade(rd, step, d_g, d_on)
                                if nxt is not None:
                                    fm_pending_op[n] = nxt
                        # roll the frozen snapshots (AFTER reading them)
                        rd.fm_prev.load_state_dict(rd.fms[rd.primary].state_dict())
                        if any(step <= o < step + cfg["ckpt_every"] + 1
                               for o in onsets + rots.get(n, [])):
                            rd.fm_event.load_state_dict(rd.fms[rd.primary].state_dict())
                    prec["fm"] = fmrec
                prec["ms_fm"] = round(1000 * (time.time() - t_fmr))
                prec["ms_donor"] = round(1000 * t_donor)
                prec["ms"] = round(1000 * (time.time() - t_ck))
                panel_log.append(prec)
                with open(os.path.join(outdir, f"panel_{worker}.json"), "w") as fh:
                    json.dump({"worker": worker, "readers": names, "windows": windows,
                               "rho_c": rho_ref, "rhos_c": rhos_c, "ladder": ladder,
                               "ladder_realised": pn_burst_rate,
                               "log": panel_log, "complete": step == max_steps},
                              fh, indent=2, cls=NumpyEncoder)

            # ------------------------------ the line ------------------------------
            for n, rd in readers.items():
                r = rd.log[-1]
                cons = r["consumed"]
                lvtxt = ""
                if "levels" in r:
                    lt = r["levels"]["true"]["key"]; lr_ = r["levels"]["rand"]["key"]
                    lvtxt = f"  d4 t/r {lt['d4']:.3f}/{lr_['d4']:.3f}"
                pt = ""
                if shadow and panel_log:
                    pl = panel_log[-1]
                    if "D_pair" in pl["clean"]:
                        pt = (f" | Dp {pl['clean']['D_pair']:.4f}->{pl['rot']['D_pair']:.4f}"
                              f"/{pl['burst_c']['D_pair']:.4f}")
                    if "D_self" in pl["clean"]:
                        pt += (f" Ds {pl['clean']['D_self']:.4f}->"
                               f"{pl['rot']['D_self']:.4f}/{pl['burst_c']['D_self']:.4f}")
                ot = ""
                if rd.ops and rd.ops[-1].get("delta"):
                    dd = rd.ops[-1]["delta"]
                    ot = (f" | dk {dd['key']:+.3f} ds {dd['self']:+.3f} "
                          f"do {dd['out']:+.3f} -> {rd.ops[-1]['type']}/{rd.op}")
                print(f"[{n:10s} s{step:6d} e{epoch}q{r['q']:2d}"
                      f"{'B' if bi >= 0 else ' '}"
                      f"{'M' if rd.key_off else ' '}"
                      f"{'S' if r['skipped'] else ' '}] "
                      f"nll idx {r['nll'][cons]['idx']:.4f} out {r['nll'][cons]['out']:.4f} "
                      f"| bind {r['binding']['idx']:+.4f} "
                      f"mislead {r['binding']['misleading_idx']:+.4f}" + ot + pt
                      + lvtxt, flush=True)
            volume.commit()
            for rd in readers.values():
                rd.model.train()

        if step == max_steps:
            break

        if pool_leaf is None or step % cfg["fresh_every"] == 0:
            need = max(64, B * cfg["fresh_every"])
            fl, flf, _ = _generate_with_traces(
                rules, need, cfg["data_seed"] + 100_003 * (step // cfg["fresh_every"] + 1))
            pool_leaf = torch.from_numpy(fl.astype(np.int64))
            pool_z = torch.from_numpy(flf[key_level][:, key_node].astype(np.int64))

        bi = BU.burst_index(step, windows)
        for n, rd in readers.items():
            # `ix` is drawn FIRST and unconditionally (the donor's rule), and it is drawn
            # even on a SKIPPED step, so a skipping arm's sampler stream stays aligned.
            ix = torch.randint(0, pool_leaf.shape[0], (B,), generator=rd.gen)
            leaf_b, z_b = pool_leaf[ix], pool_z[ix]
            if bi >= 0 and rhos_c[bi] > 0:
                cor, _ = BU.corrupt(leaf_b.numpy(), key_lo, key_hi, rhos_c[bi], v,
                                    BU.train_burst_rng(cfg["burst_seed"], step))
                leaf_b = torch.from_numpy(cor)
            q = W.q_at(step, rd.phase1, rd.rot_period, rot_step, v)
            if rd.key_off or rd.kind == "none":
                wc = torch.full((B,), W.neutral_tok(v), dtype=torch.long)
            elif rd.kind == "dead":
                # the admissibility control carries the SAME permutation: shifting a token
                # that never meant anything deletes zero index content, so whatever
                # transient it shows is the price of the permutation itself (merge_dead's
                # idiom, for re-key)
                wc = torch.from_numpy((W.wall_tok(
                    (rd.train_rng.integers(0, v, size=B) + rd.rekey) % v, v)
                ).astype(np.int64))
            else:
                wc = W.wall_tok((z_b + q + rd.rekey) % v, v)
            # the offered batch goes into the typing buffer whether or not it is learned
            # from — the arm SEES the news before it decides what to do about it
            if len(rd.buf) >= cfg["n_buf"]:
                rd.buf.pop(0)
            rd.buf.append((leaf_b, wc, z_b))
            # THE SKIP OP: lr 0 on the reader, no optimiser state touched. FIX (wave 2
            # post-mortem): the FM must keep learning through a skipped window. Leaving its
            # update inside the training branch froze it whenever the arm skipped, which
            # made e_frozen_prev EXACTLY equal e_online, collapsed the lag statistic to
            # -1.000, and made the grader revert every skip it had just caused. The FM
            # reads activations under no_grad and owns its optimiser, so there is no reason
            # for a reader-side skip to freeze it.
            skip_now = rd.op == "skip"
            if rd.charged and cfg["charge"] and step >= max_steps - rd.spent:
                if rd.terminal_step is None:
                    rd.terminal_step = step
                    print(f"  [budget] {n} exhausted at s{step} "
                          f"({rd.spent:.0f} step-equivalents of priced reads)", flush=True)
                skip_now = True
            if skip_now and not (use_fm and rd.fms):
                continue
            x, y = make_x(leaf_b, wc).to(device), leaf_b.to(device)
            if use_fm and rd.fms:
                if skip_now:
                    with torch.no_grad():                  # the reader does NOT move
                        _, loss, inter = rd.model(x, y, return_intermediates=True)
                else:
                    _, loss, inter = rd.model(x, y, return_intermediates=True)
                h0 = inter[cfg["fm_shallow"]].detach()
                h6 = inter[cfg["fm_deep"]].detach()[:, key_lo:key_hi, :]
                if not skip_now:
                    rd.opt.zero_grad(); loss.backward(); rd.opt.step()
                for k, fm in rd.fms.items():               # the FM learns either way
                    fl_ = F.mse_loss(fm(h0)[:, key_lo:key_hi, :], h6)
                    rd.fopts[k].zero_grad(); fl_.backward(); rd.fopts[k].step()
            elif not skip_now:
                _, loss = rd.model(x, y)
                rd.opt.zero_grad(); loss.backward(); rd.opt.step()
            if step % cfg["log_interval"] == 0 and not skip_now:
                print(f"  {n} {step:6d} ntp {loss.item():.4f}", flush=True)

    print(f"[{worker}] DONE in {time.time() - started:.0f}s", flush=True)
    return {"worker": worker, "readers": names, "elapsed": time.time() - started,
            "n_ckpt": len(next(iter(readers.values())).log)}


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=43200, memory=16384)
def gate2(
    tag: str = "smoke",
    workers: str = "A",
    # DGP / model / training — the donor's, verbatim
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    key_level: int = 2, key_node: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    max_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, data_seed: int = 7, seed: int = 42,
    fresh_every: int = 500,
    phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
    # the burst schedule
    burst_steps: str = "5000,11000,17000", burst_len: int = 250,
    burst_rho: str = "0.0242,0.0410,0.0377", burst_seed: int = 31337,
    burst_ladder: str = "0.002,0.005,0.01,0.02,0.05,0.10",
    # the DRIFT schedule (NEW): level-3 rule-cell resampling, permanent and cumulative
    drift_steps: str = "6000,13000,19000", drift_level: int = 3,
    drift_cells: int = 8, drift_seed: int = 90210,
    drift_ladder: str = "1,2,4,8,16,32", drift_eval_seed: int = 4242,
    # the ONLINE typing instrument and its dead zones (NEW)
    n_buf: int = 8, pol_halflife: float = 4.0,
    dz_key: float = 0.076, dz_self: float = 0.071, dz_out: float = 0.028,
    rot_requires_out: bool = False, sched_merge_at: int = 8000,
    # the FM graders (thresholds measured in wave 1, in-tag)
    fm_online_thresh: float = 0.29, fm_lag_thresh: float = 1.5,
    # the ABBA trial meter and read charging
    abba_v_tol: float = 0.0081, price_pass: float = 2.7, charge: bool = True,
    rekey_search: int = 16,
    rand_seed: int = 5150, rand_n_skip: int = 6,
    # reader checkpointing (NEW)
    ckpt_stride: int = 16, ckpt_arms: str = "self,pair_key,pair_parse,track",
    # the FM
    fm_shallow: str = "post_block0", fm_deep: str = "post_block6",
    fm_n_layer: int = 1, fm_d_head: int = 16, fm_n_head: int = 8,
    fm_mlp_mult: float = 1.0, bench_halflife: float = 8.0,
    # measurement
    n_eval: int = 4096, eval_seed: int = 999, n_panel: int = 512,
    n_probe: int = 4000, probe_seed: int = 1001, n_card: int = 256,
    min_cell_n: int = 24,
    ckpt_every: int = 125, index_every: int = 500,
    post_rot_a: int = 50, post_rot_b: int = 100,
    probe_ckpts: str = ("250,1000,2000,4000,5000,5250,6000,8000,8250,10000,"
                        "11000,11250,12000,14000,16000,17000,17250,18000"),
    full_ckpts: str = "8000",
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    full_at_end: bool = True,
    probe_blocks: str = "post_embed,post_block2,post_block4,post_block6,post_block7",
    log_interval: int = 2000,
    shadow: bool = True, fm: bool = True, quick: bool = False,
):
    if quick:
        max_steps, n_eval, n_probe, n_panel = 900, 768, 384, 256
        phase1, rot_period, fresh_every = 300, 300, 100
        ckpt_every, index_every, probe_steps, mlp_steps = 100, 300, 100, 100
        probe_ckpts, full_ckpts, log_interval = "300,600", "900", 300
        n_card, min_cell_n = 128, 8
        burst_steps, burst_len = "400,700", 100
        burst_ladder = "0.01,0.05,0.20"
        drift_steps, drift_ladder = "500,800", "2,8,32"
        sched_merge_at, ckpt_stride, rand_n_skip = 300, 4, 2

    wlist = []
    for w in workers.split(","):
        w = w.strip()
        if not w:
            continue
        wlist.extend(WAVES[w] if w in WAVES else [w])
    for w in wlist:
        if w not in WORKER_SPECS:
            raise ValueError(f"unknown worker {w}; known: {sorted(WORKER_SPECS)} "
                             f"or a wave in {sorted(WAVES)}")

    cfg = dict(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, key_level=key_level,
               key_node=key_node, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               max_steps=max_steps, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, data_seed=data_seed, seed=seed,
               fresh_every=fresh_every, rot_step=rot_step, phase1=phase1,
               rot_period=rot_period,
               burst_steps=burst_steps, burst_len=burst_len, burst_rho=burst_rho,
               burst_seed=burst_seed, burst_ladder=burst_ladder,
               drift_steps=drift_steps, drift_level=drift_level,
               drift_cells=drift_cells, drift_seed=drift_seed,
               drift_ladder=drift_ladder, drift_eval_seed=drift_eval_seed,
               n_buf=n_buf, pol_halflife=pol_halflife, dz_key=dz_key,
               dz_self=dz_self, dz_out=dz_out,
               rot_requires_out=bool(rot_requires_out),
               fm_online_thresh=fm_online_thresh, fm_lag_thresh=fm_lag_thresh,
               abba_v_tol=abba_v_tol, price_pass=price_pass,
               charge=bool(charge), rekey_search=rekey_search,
               sched_merge_at=sched_merge_at, rand_seed=rand_seed,
               rand_n_skip=rand_n_skip, ckpt_stride=ckpt_stride, ckpt_arms=ckpt_arms,
               fm_shallow=fm_shallow, fm_deep=fm_deep, fm_n_layer=fm_n_layer,
               fm_d_head=fm_d_head, fm_n_head=fm_n_head, fm_mlp_mult=fm_mlp_mult,
               bench_halflife=bench_halflife,
               n_eval=n_eval, eval_seed=eval_seed, n_panel=n_panel, n_probe=n_probe,
               probe_seed=probe_seed, n_card=n_card, min_cell_n=min_cell_n,
               ckpt_every=ckpt_every, index_every=index_every, post_rot_a=post_rot_a,
               post_rot_b=post_rot_b, probe_ckpts=probe_ckpts, full_ckpts=full_ckpts,
               probe_steps=probe_steps, probe_lr=probe_lr, mlp_hidden=mlp_hidden,
               mlp_steps=mlp_steps, probe_blocks=probe_blocks, log_interval=log_interval,
               full_at_end=bool(full_at_end),
               shadow=bool(shadow), fm=bool(fm))

    print(f"{'=' * 78}\ntuning/Gate 1 WAVE 2  tag={tag}  {setting_key(v, s, depth, m)}  "
          f"{n_layer}L/{n_head}H/{n_embd}D  T={s ** depth}")
    print(f"  workers={wlist}  max_steps={max_steps}  shadow={shadow} fm={fm}")
    print(f"  bursts at {burst_steps} x {burst_len}, rho(s)={burst_rho}; "
          f"ladder {burst_ladder}")
    print(f"  drift at {drift_steps}, level {drift_level}, {drift_cells} cells/event; "
          f"ladder {drift_ladder}")
    print(f"  dead zones  d_key {dz_key}  d_self {dz_self}  d_out {dz_out}  "
          f"(rot_requires_out={rot_requires_out})\n{'=' * 78}", flush=True)

    # ---- the WORLD: one grammar per epoch, and exact references for each ----
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.practice.fourwall.lm.wall_lm import build_refs
    from rhm.practice.tuning import drift_ev as DR

    base_rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    dsteps = DR.drift_steps(drift_steps, max_steps)
    for d in dsteps:
        assert d % fresh_every == 0, (
            f"drift step {d} must fall on a pool refresh ({fresh_every}) so the "
            f"data_seed stream is untouched by the epoch turn")
    rules_traj, changes = DR.rules_trajectory(base_rules, depth, drift_level,
                                              drift_cells, dsteps, drift_seed)
    refs_by_epoch, refs_src = [], []
    for e, r_e in enumerate(rules_traj):
        if e == 0:
            r0, src = load_refs(cfg)          # epoch 0 == the donors' world, verbatim
            refs_by_epoch.append(r0); refs_src.append(src)
            continue
        print(f"[refs] epoch {e}: recomputing exact BP for the drifted grammar "
              f"(cell survival vs base "
              f"{DR.cell_survival(rules_traj[0], r_e, depth, drift_level):.4f})", flush=True)
        refs_by_epoch.append(_build_refs_for(r_e, cfg, build_refs))
        refs_src.append(f"epoch{e}:recomputed")

    scheds = {r: arm_schedule(r, phase1, rot_period, max_steps) for r in ARM_SPECS}
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "workers": wlist, "scheds": scheds,
                   "refs": refs_by_epoch[0], "refs_by_epoch": refs_by_epoch,
                   "refs_source": refs_src, "fm_lrs": list(FM_LRS),
                   "fm_primary": FM_PRIMARY, "drift_steps": dsteps,
                   "drift_changes": changes,
                   "cell_survival": [DR.cell_survival(rules_traj[0], r_e, depth,
                                                      drift_level)
                                     for r_e in rules_traj]},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    started = time.time()
    handles = [(w, run_worker.spawn(tag, w, scheds, refs_by_epoch, cfg))
               for w in wlist]
    out = {}
    for w, h in handles:
        try:
            out[w] = h.get()
        except Exception as exc:                              # noqa: BLE001
            out[w] = {"worker": w, "error": repr(exc)}
            print(f"[driver] worker {w} FAILED: {exc!r}", flush=True)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n{json.dumps(out, indent=2)}\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}\n{json.dumps(out, indent=2)}")
    return out


# --------------------------------------------------------------------------- #
# the burst-rate calibration (GPU, short): match the MODEL's burst spike to the
# MODEL's rotation spike, at the maturity the first burst will meet
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=7200, memory=32768)
def calib2(tag: str = "g1_calib", steps: str = "2000,5000,9000,13000,17000",
          rhos: str = "0.002,0.005,0.01,0.02,0.05,0.10,0.20",
          dcells: str = "1,2,4,8,16,32", drift_level: int = 3, drift_seed: int = 90210,
          drift_eval_seed: int = 4242, n_buf: int = 8, batch_reads: bool = True,
          v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
          key_level: int = 2, key_node: int = 0,
          n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
          batch_size: int = 64, lr: float = 3e-4, weight_decay: float = 0.01,
          data_seed: int = 7, seed: int = 42, fresh_every: int = 500,
          phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
          n_eval: int = 2048, eval_seed: int = 999):
    """Train the keyed reader on the DONOR's schedule (no bursts, no panel, no FM — so
    this is the donor's own trajectory) and, at each ladder checkpoint, read

        rotation instant   NLL(perm) - NLL(true)     on the indexed span
        burst instant      NLL(true, corrupted) - NLL(true, clean)   for every rho

    on the SAME weights and the same eval draw. The calibrated rate is the rho whose burst
    spike equals the rotation spike at step 5000 — the maturity the first consumed burst
    will actually meet.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import burst as BU
    from rhm.practice.tuning import drift_ev as DR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L, T, V = depth, s ** depth, W.vocab_size(v)
    ck = sorted({int(x) for x in steps.split(",")})
    rr = [float(x) for x in rhos.split(",")]
    dcl = [int(x) for x in dcells.split(",")]
    max_steps = max(ck)
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    key_lo, key_hi = W.key_span(L, s, key_level, key_node)

    ev_leaf, ev_lf, _ = _generate_with_traces(rules, n_eval, eval_seed)
    ev_z = ev_lf[key_level][:, key_node]
    ev_t = torch.from_numpy(ev_leaf.astype(np.int64))
    burst_t, realised = [], []
    for i, rho in enumerate(rr):
        cor, mask = BU.corrupt(ev_leaf, key_lo, key_hi, rho, v,
                               BU.shadow_burst_rng(eval_seed, i))
        burst_t.append(torch.from_numpy(cor.astype(np.int64)))
        realised.append(float(mask.mean()))

    # ---- NEW: the DRIFT ladder, each rung read against a NULL fresh draw ----
    dn_leaf, dn_lf, _ = _generate_with_traces(rules, n_eval, drift_eval_seed)
    dn_t = torch.from_numpy(dn_leaf.astype(np.int64))
    dn_z = dn_lf[key_level][:, key_node]
    drift_t, drift_z, drift_surv = [], [], []
    for i, rl in enumerate(DR.ladder_rules(rules, L, drift_level, dcl, drift_seed, 0)):
        gl, glf, _ = _generate_with_traces(rl, n_eval, drift_eval_seed)
        drift_t.append(torch.from_numpy(gl.astype(np.int64)))
        drift_z.append(glf[key_level][:, key_node])
        drift_surv.append(DR.cell_survival(rules, rl, L, drift_level))

    torch.manual_seed(seed)
    model = GPT(V, T, n_layer, n_head, n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    def make_x(leaf_t, wc):
        return torch.cat([wc[:, None], leaf_t[:, :-1]], 1)

    @torch.no_grad()
    def nll_idx_out(leaf_t, wc, chunk=256):
        si = so = 0.0
        n = leaf_t.shape[0]
        for i in range(0, n, chunk):
            x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
            y = leaf_t[i:i + chunk].to(device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                  reduction="none").reshape(y.shape)
            si += float(nll[:, key_lo:key_hi].sum()); so += float(nll[:, key_hi:].sum())
        return si / (n * (key_hi - key_lo)), so / (n * (T - key_hi))

    out = {"steps": ck, "rhos": rr, "realised": realised, "dcells": dcl,
           "drift_survival": drift_surv, "rows": [], "floor": []}

    # ---- NEW: the ONLINE instrument's own null floor ----
    # The op map runs on jumps in (r_key, r_self, r_out) read off the batches just
    # OFFERED, so its dead zones have to sit above the jump this instrument makes when
    # NOTHING happens. Measured here on an event-free trajectory (the `wall` schedule's
    # rotations are the only events, and they are excluded in the reduction).
    buf = []

    @torch.no_grad()
    def buf_reads():
        leaf = torch.cat([b[0] for b in buf]); wc = torch.cat([b[1] for b in buf])
        n = leaf.shape[0]
        wn = torch.full((n,), W.neutral_tok(v), dtype=torch.long)
        r = {}
        for tag_, col in (("key", wc), ("self", wn)):
            si = so_ = 0.0
            for i in range(0, n, 256):
                x = make_x(leaf[i:i + 256], col[i:i + 256]).to(device)
                y = leaf[i:i + 256].to(device)
                logits, _ = model(x)
                nll = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                      reduction="none").reshape(y.shape)
                si += float(nll[:, key_lo:key_hi].sum())
                so_ += float(nll[:, key_hi:].sum())
            r[tag_] = (si / (n * (key_hi - key_lo)), so_ / (n * (T - key_hi)))
        return {"key": r["key"][0], "self": r["self"][0], "out": r["key"][1]}

    pool_leaf = pool_z = None
    for step in range(max_steps + 1):
        if batch_reads and step % 125 == 0 and len(buf) >= n_buf:
            model.eval()
            out["floor"].append({"step": step, **buf_reads()})
            model.train()
        if step in ck:
            model.eval()
            q = W.q_at(step, phase1, rot_period, rot_step, v)
            wt = torch.from_numpy(W.wall_tok((ev_z + q) % v, v).astype(np.int64))
            wp = torch.from_numpy(W.wall_tok((ev_z + q + rot_step) % v, v).astype(np.int64))
            wn = torch.full((n_eval,), W.neutral_tok(v), dtype=torch.long)
            ci, co = nll_idx_out(ev_t, wt)
            ri, ro = nll_idx_out(ev_t, wp)
            ni, no_ = nll_idx_out(ev_t, wn)
            row = {"step": step, "clean_idx": ci, "clean_out": co,
                   "rot_idx": ri, "rot_out": ro, "rot_spike": ri - ci,
                   "rot_out_leak": ro - co,
                   "none_idx": ni, "vs_none": ni - ci, "burst": []}
            for i, rho in enumerate(rr):
                bi_, bo_ = nll_idx_out(burst_t[i], wt)
                row["burst"].append({"rho": rho, "realised": realised[i],
                                     "idx": bi_, "out": bo_,
                                     "spike": bi_ - ci, "out_leak": bo_ - co})
            rho_star, brack = BU.match_rate(
                [{"rho": b["rho"], "d_naive": b["spike"]} for b in row["burst"]],
                row["rot_spike"])
            row["rho_star"], row["bracketed"] = rho_star, brack
            # ---- the DRIFT ladder, against the null draw ----
            wn_null = torch.from_numpy(W.wall_tok((dn_z + q) % v, v).astype(np.int64))
            ni, no2 = nll_idx_out(dn_t, wn_null)
            nsi, _ = nll_idx_out(dn_t, torch.full((n_eval,), W.neutral_tok(v),
                                                  dtype=torch.long))
            row["drift_null"] = {"idx": ni, "out": no2, "self": nsi,
                                 "offset_vs_eval": ni - ci}
            row["drift"] = []
            for i, nc in enumerate(dcl):
                wdz = torch.from_numpy(W.wall_tok((drift_z[i] + q) % v, v).astype(np.int64))
                di, do2 = nll_idx_out(drift_t[i], wdz)
                dsi, _ = nll_idx_out(drift_t[i], torch.full((n_eval,), W.neutral_tok(v),
                                                            dtype=torch.long))
                row["drift"].append({"n_cells": nc, "survival": drift_surv[i],
                                     "idx": di, "out": do2, "self": dsi,
                                     "d_key": di - ni, "d_out": do2 - no2,
                                     "d_self": dsi - nsi})
            nc_star, nbrack = DR.match_cells(row["drift"], row["rot_spike"])
            row["n_cells_star"], row["drift_bracketed"] = nc_star, nbrack
            out["rows"].append(row)
            print(f"[calib s{step:6d}] clean {ci:.4f}  rot +{row['rot_spike']:.4f} "
                  f"(out +{row['rot_out_leak']:.4f})  vs_none +{row['vs_none']:.4f}",
                  flush=True)
            for b in row["burst"]:
                print(f"    rho {b['rho']:<7.4g} realised {b['realised']:.4f}  "
                      f"idx {b['idx']:.4f} (+{b['spike']:.4f})  "
                      f"out +{b['out_leak']:.4f}", flush=True)
            print(f"    -> rho* matching the rotation spike: {rho_star:.5f} "
                  f"({'bracketed' if brack else 'EXTRAPOLATED'})", flush=True)
            print(f"    drift null draw offset vs the standing eval set "
                  f"{row['drift_null']['offset_vs_eval']:+.4f} nats "
                  f"(the control the burst ladder never needed)", flush=True)
            for b in row["drift"]:
                print(f"    cells {b['n_cells']:<3d} surv {b['survival']:.3f}  "
                      f"d_key {b['d_key']:+.4f}  d_self {b['d_self']:+.4f}  "
                      f"d_out {b['d_out']:+.4f}", flush=True)
            print(f"    -> n_cells* matching the rotation spike: {nc_star:.2f} "
                  f"({'bracketed' if nbrack else 'EXTRAPOLATED'})", flush=True)
            model.train()
        if step == max_steps:
            break
        if step % fresh_every == 0:
            need = max(64, batch_size * fresh_every)
            fl, flf, _ = _generate_with_traces(
                rules, need, data_seed + 100_003 * (step // fresh_every + 1))
            pool_leaf = torch.from_numpy(fl.astype(np.int64))
            pool_z = torch.from_numpy(flf[key_level][:, key_node].astype(np.int64))
        ix = torch.randint(0, pool_leaf.shape[0], (batch_size,), generator=gen)
        leaf_b, z_b = pool_leaf[ix], pool_z[ix]
        q = W.q_at(step, phase1, rot_period, rot_step, v)
        wc = W.wall_tok((z_b + q) % v, v)
        if len(buf) >= n_buf:
            buf.pop(0)
        buf.append((leaf_b, wc))
        x, y = make_x(leaf_b, wc).to(device), leaf_b.to(device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "calib.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print("\n=== CALIBRATION SUMMARY ===")
    for r in out["rows"]:
        print(f"  s{r['step']:6d}  rotation spike +{r['rot_spike']:.4f}  ->  rho* "
              f"{r['rho_star']:.5f} ({'brk' if r['bracketed'] else 'EXT'})   n_cells* "
              f"{r['n_cells_star']:.2f} ({'brk' if r['drift_bracketed'] else 'EXT'})")
    if out["floor"]:
        rots_ = set(W.rotation_steps(phase1, rot_period, max_steps))
        seq = out["floor"]
        jumps = {k: [] for k in ("key", "self", "out")}
        for i in range(1, len(seq)):
            if any(abs(seq[i]["step"] - x) <= 250 for x in rots_):
                continue
            for k in jumps:
                jumps[k].append(abs(seq[i][k] - seq[i - 1][k]))
        out["floor_summary"] = {
            k: {"n": len(vv), "p50": float(np.percentile(vv, 50)),
                "p90": float(np.percentile(vv, 90)),
                "p99": float(np.percentile(vv, 99)), "max": float(np.max(vv))}
            for k, vv in jumps.items() if vv}
        print("\n=== ONLINE-INSTRUMENT NULL FLOOR (adjacent-checkpoint |jump|, "
              "non-rotation) ===")
        for k, d in out["floor_summary"].items():
            print(f"  d_{k:5s} n={d['n']:3d}  p50 {d['p50']:.4f}  p90 {d['p90']:.4f}  "
                  f"p99 {d['p99']:.4f}  max {d['max']:.4f}")
        print("  -> dead zones should sit at or above p99.")
    return out


# --------------------------------------------------------------------------- #
# gates (CPU: structural + the model-free irreducibility calibration)
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=7200, memory=16384)
def gate(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
         key_level: int = 2, key_node: int = 0, n_eval: int = 2048, eval_seed: int = 999,
         phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
         max_steps: int = 20000, burst_steps: str = "5000,11000,17000",
         burst_len: int = 250, burst_seed: int = 31337,
         rhos: str = "0.005,0.01,0.02,0.05,0.10,0.20", n_oracle: int = 192,
         rot_spike_target: float = 0.63,
         drift_spec: str = "6000,13000,19000", drift_level: int = 3,
         drift_cells: int = 8, drift_seed: int = 90210, fresh_every: int = 500):
    """B-1  the burst is confined to the indexed span, at the nominal rate, and carries
            ZERO information about the keyed latent (the irreducibility precondition).
       B-2  the burst never occupies a rotation instant, and every window is disjoint.
       B-3  RNG neutrality: the burst draws are independent of the batch sampler, and the
            SHADOW draw is fixed across checkpoints while the CONSUMED draw varies by step.
       B-4  the exact, model-free irreducibility curve (belief propagation, no model): the
            noise-aware optimum's excess is a floor no learner can lower, and the naive
            excess is what a reader that does not know the noise exists pays.
       B-5  the leaf/wall/neutral token layout still holds under corruption.
       G-*  the donor's own structural gates, re-asserted on this config.
    """
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W
    from rhm.practice.tuning import burst as BU

    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    leaf, lf, _ = _generate_with_traces(rules, n_eval, eval_seed)
    z = lf[key_level][:, key_node]
    key_lo, key_hi = W.key_span(L, s, key_level, key_node)
    rr = [float(x) for x in rhos.split(",")]
    starts = [int(x) for x in burst_steps.split(",")]
    windows = BU.burst_windows(starts, burst_len, max_steps)
    rots = W.rotation_steps(phase1, rot_period, max_steps)
    out = {"setting": setting_key(v, s, L, m), "T": T, "key_span": [key_lo, key_hi],
           "windows": windows, "rotations": rots}

    # --- B-1: locality, rate, and irreducibility's precondition ---
    rng = BU.train_burst_rng(burst_seed, 5000)
    cor, mask = BU.corrupt(leaf, key_lo, key_hi, 0.10, v, rng)
    outside_touched = int((cor[:, key_hi:] != leaf[:, key_hi:]).sum())
    changed = cor[:, key_lo:key_hi] != leaf[:, key_lo:key_hi]
    # the information check runs at rho = 1 and POOLED over the span, so the plug-in MI
    # estimator's bias ((v-1)^2 / 2N) is ~0.003 nats rather than swamping the reading; a
    # label permutation supplies the matched null.
    full, _ = BU.corrupt(leaf, key_lo, key_hi, 1.0, v,
                         BU.train_burst_rng(burst_seed, 5001))
    zt = np.repeat(z[:, None], key_hi - key_lo, axis=1).ravel()
    ft = full[:, key_lo:key_hi].ravel()
    mi_corrupt = float(_mutinf(ft, zt, v))
    prng = np.random.default_rng(0)
    nulls = [float(_mutinf(ft, prng.permutation(zt), v)) for _ in range(20)]
    mu, sd = float(np.mean(nulls)), float(np.std(nulls))
    # scale reference: the marginal MI a SINGLE clean leaf carries is itself tiny on this
    # world (the key's 0.152 nats/token lives in the joint, not the per-token marginal), so
    # the meaningful comparison is against the matched permutation null, not against it.
    mi_clean = float(_mutinf(leaf[:, key_lo:key_hi].ravel(), zt, v))
    out["B1"] = {"outside_span_touched": outside_touched,
                 "nominal_rate": 0.10, "realised_rate": float(mask.mean()),
                 "changed_given_masked": float(changed[mask].mean()),
                 "n_pairs": int(zt.size),
                 "MI_fully_corrupted_span_z_nats": mi_corrupt,
                 "MI_permutation_null_mean": mu, "MI_permutation_null_sd": sd,
                 "MI_permutation_null_max": float(np.max(nulls)),
                 "z_score_vs_null": (mi_corrupt - mu) / max(sd, 1e-12),
                 "MI_clean_per_token_marginal_nats": mi_clean}
    assert outside_touched == 0, "B-1 corruption escaped the indexed span"
    assert abs(mask.mean() - 0.10) < 0.01, "B-1 rate off"
    assert mi_corrupt <= mu + 4 * sd, \
        "B-1 corrupted tokens still carry the latent (above the permutation null)"

    # --- B-2 ---
    out["B2"] = {"overlaps_rotation": bool(BU.overlaps(windows, rots)),
                 "overlaps_rotation_guard100": bool(BU.overlaps(windows, rots, guard=100)),
                 "disjoint": all(windows[i][1] <= windows[i + 1][0]
                                 for i in range(len(windows) - 1))}
    assert not out["B2"]["overlaps_rotation_guard100"], "B-2 a burst sits on a rotation"
    assert out["B2"]["disjoint"], "B-2 windows overlap"

    # --- B-3 ---
    a1, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.train_burst_rng(burst_seed, 100))
    a2, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.train_burst_rng(burst_seed, 100))
    a3, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.train_burst_rng(burst_seed, 101))
    b1, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.shadow_burst_rng(eval_seed, 0))
    b2, _ = BU.corrupt(leaf[:64], key_lo, key_hi, 0.1, v, BU.shadow_burst_rng(eval_seed, 0))
    out["B3"] = {"consumed_same_step_reproducible": bool((a1 == a2).all()),
                 "consumed_differs_across_steps": bool((a1 != a3).any()),
                 "shadow_fixed": bool((b1 == b2).all())}
    assert out["B3"]["consumed_same_step_reproducible"] and out["B3"]["shadow_fixed"], "B-3"
    assert out["B3"]["consumed_differs_across_steps"], "B-3 consumed draw is not per-step"

    # --- B-4: the exact, model-free irreducibility curve ---
    print("\n--- B-4  exact irreducibility curve (BP, no model) ---", flush=True)
    sub, subz = leaf[:n_oracle], z[:n_oracle]
    curve = BU.oracle_burst_curve(rules, sub, subz, key_level, key_node, key_lo, key_hi,
                                  rr, v, eval_seed=eval_seed, keyed=True)
    out["B4"] = curve
    per_tok = [r["d_robust"] / max(r["realised_rate"], 1e-9) for r in curve["rows"]]
    out["B4"]["irreducible_nats_per_corrupted_token"] = per_tok
    assert all(r["d_robust"] > 0 for r in curve["rows"]), "B-4 no irreducible excess"
    assert all(curve["rows"][i]["d_robust"] <= curve["rows"][i + 1]["d_robust"] + 1e-9
               for i in range(len(curve["rows"]) - 1)), "B-4 floor not monotone in rho"

    # --- B-5 ---
    out["B5"] = {"vocab": W.vocab_size(v), "leaf_max_corrupted": int(cor.max()),
                 "wall_range": [v, 2 * v - 1], "neutral": W.neutral_tok(v),
                 "disjoint": bool(int(cor.max()) < v)}
    assert out["B5"]["disjoint"], "B-5"

    # --- D-1..D-4: the DRIFT event (NEW IN THIS FORK) ---
    from rhm.practice.tuning import drift_ev as DR
    dsteps = DR.drift_steps(drift_spec, max_steps)
    dtraj, dch = DR.rules_trajectory(rules, L, drift_level, drift_cells, dsteps,
                                     drift_seed)
    out["D1"] = {  # locality: only the target layer moves; the leaf layer is untouched
        "steps": dsteps, "level": drift_level, "cells_per_event": drift_cells,
        "layer_index": L - drift_level,
        "other_layers_identical": all(
            DR.layers_identical(dtraj[k], dtraj[k + 1], L - drift_level)
            for k in range(len(dtraj) - 1)),
        "leaf_layer_untouched": all(
            np.array_equal(np.asarray(dtraj[0][L - 1]), np.asarray(r[L - 1]))
            for r in dtraj),
        "cell_survival_vs_base": [DR.cell_survival(dtraj[0], r, L, drift_level)
                                  for r in dtraj],
    }
    assert out["D1"]["other_layers_identical"], "D-1 drift escaped its layer"
    assert out["D1"]["leaf_layer_untouched"], "D-1 drift touched the leaf rendering"
    assert drift_level >= 2, "D-1 drift must be confined to levels >= 2"

    out["D2"] = {  # the schedule: drift never occupies a rotation or a burst window
        "clear_of_rotations": DR.clear_of(dsteps, rots, guard=750),
        "clear_of_bursts": DR.clear_of(
            dsteps, [x for a, b in windows for x in (a, b)], guard=750),
        "on_pool_refresh": all(d % fresh_every == 0 for d in dsteps),
    }
    assert out["D2"]["clear_of_rotations"] and out["D2"]["clear_of_bursts"], "D-2"
    assert out["D2"]["on_pool_refresh"], "D-2 a drift must land on a pool refresh"

    # D-3: the drift actually MOVES the leaf distribution, and the null control is small
    from rhm.rhm_latent_loop import _generate_with_traces as _gwt
    n_d = min(n_eval, 512)
    l0, lf0, _ = _gwt(dtraj[0], n_d, 4242)
    p_base = BU.predictive_probs(dtraj[0], l0[:96])
    rows_d = []
    for i, rl in enumerate(DR.ladder_rules(dtraj[0], L, drift_level, [1, 4, 16], 90210, 0)):
        lg, _, _ = _gwt(rl, n_d, 4242)
        pg = BU.predictive_probs(dtraj[0], lg[:96])          # OLD grammar, NEW data
        rows_d.append({"n_cells": [1, 4, 16][i],
                       "survival": DR.cell_survival(dtraj[0], rl, L, drift_level),
                       "frac_leaves_changed": float((lg != l0).mean()),
                       "oracle_d_key_span": float(
                           (-np.log(pg[:, key_lo:key_hi])).mean()
                           - (-np.log(p_base[:, key_lo:key_hi])).mean()),
                       "oracle_d_out": float(
                           (-np.log(pg[:, key_hi:])).mean()
                           - (-np.log(p_base[:, key_hi:])).mean())})
    out["D3"] = {"rows": rows_d,
                 "null_frac_leaves_changed": 0.0}
    assert rows_d[-1]["frac_leaves_changed"] > rows_d[0]["frac_leaves_changed"], \
        "D-3 drift magnitude is not monotone in n_cells"
    # D-4: unlike a burst, a drift is expected to move the OUT-OF-SPAN view too — the
    # coordinate that is supposed to separate it from a burst. Reported, not asserted:
    # the model-side reading is what `calib2` measures and what Gate 1 tests.
    out["D4"] = {"oracle_out_moves": bool(rows_d[-1]["oracle_d_out"] > 0)}

    # --- the donor's structural gates, re-asserted ---
    zz = np.arange(v)
    out["G1"] = {"bijection": bool(all(len(set(((zz + q) % v).tolist())) == v
                                       for q in range(v))),
                 "derangements": bool(all(W.is_derangement(k * rot_step, v)
                                          for k in range(1, v // max(1, np.gcd(rot_step, v)))))}
    assert out["G1"]["bijection"] and out["G1"]["derangements"], "G-1"
    out["G2"] = {"rotations": rots,
                 "q_at_0": W.q_at(0, phase1, rot_period, rot_step, v),
                 "q_at_phase1_minus_1": W.q_at(phase1 - 1, phase1, rot_period, rot_step, v)}
    assert out["G2"]["q_at_0"] == 0 and out["G2"]["q_at_phase1_minus_1"] == 0, "G-2"

    print("\n" + json.dumps({k: out[k] for k in out if k != "B4"}, indent=2,
                            cls=NumpyEncoder))
    print("\nB-4 rows:")
    for r in out["B4"]["rows"]:
        print(f"  rho {r['rho']:<7.4g} realised {r['realised_rate']:.4f}  "
              f"irreducible +{r['d_robust']:.4f}  naive +{r['d_naive']:.4f}")
    print(f"\n  rho whose IRREDUCIBLE floor equals a rotation's {rot_spike_target} nats: "
          f"{BU.match_rate(out['B4']['rows'], rot_spike_target, key='d_robust')}")
    print("\nGATE: PASS")
    return out


def _mutinf(a, b, v):
    import numpy as np
    j = np.zeros((v, v))
    np.add.at(j, (a, b), 1.0)
    j /= max(j.sum(), 1e-12)
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return (j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum()


@app.local_entrypoint()
def main(quick: bool = True):
    gate2.remote(quick=quick, tag="smoke_local")
