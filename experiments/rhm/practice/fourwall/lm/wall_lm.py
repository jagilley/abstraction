"""fourwall/lm — the LM twin: does an ENDOGENOUS reader build, and tear down, an index?

Parent: `../` (fourwall, fw_s0-s3 — the spurious index: scaffold, debt, merge, re-key, retire).
Port template: `../../reread/lm/` (the same move for the reread claim: an exogenous-instrument
question rebuilt with an NTP reader graded against exact BP oracles). Machinery donor:
`../../../conditional_revision/oracle.py`, imported and never modified. Primitives, schedule and
exact index oracles: `wall.py`.

WHY THIS EXISTS. Every finding in `../` was measured with EXOGENOUS machinery — a hand-built
lookup-table library and index ops (merge / re-key / retire) that we ran on the learner's behalf.
So the joints may be joints of the scaffolding rather than of learning. The port question, which
`recurrence_manufactures_confounds` §9 names as the doc's deepest exposure:

    IS THERE A DISCRETE INDEX EVENT, or does dense learning under varied demand smear its
    way to the same quotient?

THE WORLD. Index-news only: the grammar is fixed (nothing becomes false) and the derivation
distribution is the DGP's own and fixed (nothing changes about what is asked). Every sequence
carries a free surface token `w` at position 0 that is, in phase 1, a bijection of the true
level-2 feature at node 0 — the latent governing the first 16 leaves. A ROTATION event then
cyclically permutes the w<->z map without deleting `w`, so a raw wall-key becomes actively
MISLEADING rather than merely absent. Rotation by 4 mod 16 returns to the identity every fourth
event, which buys the "does a stale cell come back into fashion" instrument for free (`../`'s
c121, here at step 14000).

WHAT IS THE LIBRARY AND WHAT IS THE INDEX, for a reader that maintains its own representations:

    library  the model's learned grammar (its content)
    index    what its early-position predictions are KEYED on, read out three ways —
             (a) how much of the exact oracle index value it captures, (b) how many
             distinct behaviours the wall induces (the MDL/cardinality readout), and
             (c) the model's own IMPLIED w->z map, wall by wall

`../`'s merge / re-key / retire have no arm analogues here BY DESIGN: they were ops we chose to
run. Here they are outcomes the reader may or may not produce, and the instruments are built to
tell them apart — merge = the wall-conditional distributions collapse toward one; re-key = the
implied map migrates onto the new offset; neither = the reader keeps a stale address and pays.

ARMS (the wall is the only lever; substrate, data draws, model init and optimiser are identical)

    true_wall     w = z, never rotates. The upper anchor: rotation-invariant by construction
    wall          identity for 8000 steps, then a rotation every 2000. THE treatment
    wall_fast     rotating from step 250, every 250 — §4's rate axis and `../../merge/`'s
                  "paced rotation is the licensing condition for merging", asked of a learner
    dead_wall     a free but UNINFORMATIVE wall. The admissibility control (its instruments
                  cannot move across a rotation), the binding null, and the noise floor
    no_wall       no wall at all (a neutral filler). The no-index baseline
    dead_wall_b   dead_wall at a second model/sampler seed — the stream/seed noise floor

ROUND 2 (`fwlm1`) — SUFFICIENCY. `fwlm0` answered the necessity half: the endogenous reader
natively has exactly one index op, *track*. Merge, re-key and retire never emerge, and their
absence is priced (`wall_fast` +0.149 nats vs `no_wall`; every rotation costs +0.63; the
token-derived pathway stalls at 0.27-0.30 against `no_wall`'s 0.84). Round 2 SUPPLIES the
missing op exogenously and prices it — see `ARM_SPECS` for the op's definition and the
timing / rescue / control arms.

Arms run as SEPARATE Modal workers with identical seeds, which structurally removes the arc's
standing torch-global-RNG stream-position confound (`../FILES.md`: identical configs landing
0.60 vs 0.93 purely from arm ordering). Cross-arm level comparisons are licensed here, and a
round-2 merge arm is bit-identical to its round-1 twin for every step before `merge_at`.

Run from experiments/:
  modal run -m rhm.practice.fourwall.lm.wall_lm::gate
  modal run -m rhm.practice.fourwall.lm.wall_lm::wall_lm --quick --tag smoke0
  python3 rhm/practice/fourwall/lm/launch_detached.py --fn wall_lm --tag fwlm0 ...
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
)
app = modal.App("rhm-practice-fourwall-lm", image=image)

REMOTE = "rhm_practice_fourwall_lm"

# arm -> (wall kind, phase1 override, rot_period override, seed offset, merge step).
# `None` = take the run-level default; merge step `None` = the op never fires.
#
# THE MERGE OP (round 2, `fwlm1`). `fwlm0` established that the endogenous reader natively
# has only ONE index op — track — and priced the absence of the others. The merge op is
# supplied here EXOGENOUSLY and, per the arc's definition of practice, it acts on the
# CONDITIONS of learning rather than on the weights: from step `merge_at`, the wall token in
# the input stream is replaced by the neutral filler. Sixteen addresses collapse to one; not
# a single parameter is touched by the op itself, the grammar is untouched, the derivation
# distribution is untouched, and the sequence's length and positions are unchanged.
#
# The collapse target is the SAME neutral token `no_wall` carries from step 0, so after
# `merge_at` a merge arm and `no_wall` consume literally identical batches — same pool, same
# sampler, same token at every position. Every post-merge difference between them is the
# inherited weights, which is what makes the recovery comparison exact rather than matched.
ARM_SPECS = {
    "true_wall":       ("wall", None, 0,    0, None),
    "wall":            ("wall", None, None, 0, None),
    "wall_fast":       ("wall", 250,  250,  0, None),
    "dead_wall":       ("dead", None, 0,    0, None),
    "no_wall":         ("none", None, 0,    0, None),
    "dead_wall_b":     ("dead", None, 0,    1, None),
    # --- round 2: the timing axis (all three merge inside phase 1, so none of them has
    #     ever seen a rotation -- the scaffold's age is the only variable) ---
    "merge_1000":      ("wall", None, None, 0, 1000),
    "merge_4000":      ("wall", None, None, 0, 4000),
    "merge_8000":      ("wall", None, None, 0, 8000),
    # --- merge AFTER the debt has been paid three times (1000 steps into era 3, fully
    #     re-keyed): lateness crossed with rotation stress ---
    "merge_13000":     ("wall", None, None, 0, 13000),
    # --- the wall_fast rescue: can the op recover the +0.149-nat penalty `fwlm0` measured ---
    "merge_fast_8000": ("wall", 250,  250,  0, 8000),
    # --- the op's own admissibility control: the identical token swap applied to an arm
    #     whose wall never carried anything. Whatever transient this shows is the cost of
    #     swapping a varying token for a constant, with zero index content deleted ---
    "merge_dead_8000": ("dead", None, 0,    0, 8000),
}


def arm_schedule(arm, phase1, rot_period, max_steps, merge_override=0):
    kind, p1, rp, soff, mg = ARM_SPECS[arm]
    p1 = phase1 if p1 is None else p1
    rp = rot_period if rp is None else rp
    if rp <= 0:
        p1 = max_steps + 1          # never rotates
    if mg is not None and merge_override > 0:   # smoke only
        mg = merge_override
    return {"kind": kind, "phase1": int(p1), "rot_period": int(rp), "seed_off": int(soff),
            "merge_at": int(max_steps + 1 if mg is None else mg)}


# --------------------------------------------------------------------------- #
# the exact references (CPU): Gate 0's oracle bracket, the probe ceilings, and the
# exact wall-conditional next-token distributions
# --------------------------------------------------------------------------- #

def build_refs(v, s, depth, m, rule_seed, key_level, key_node,
               n_eval, eval_seed, n_probe, probe_seed, n_ceil=2000, verbose=True):
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.practice.fourwall.lm import wall as W

    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    ev_leaf, ev_lf, _ = _generate_with_traces(rules, n_eval, eval_seed)
    ev_z = ev_lf[key_level][:, key_node]
    pr_leaf, pr_lf, _ = _generate_with_traces(rules, n_probe, probe_seed)
    pr_z = pr_lf[key_level][:, key_node]

    lo, hi = W.key_span(L, s, key_level, key_node)
    key_plen, last_plen = hi, T - 1

    t0 = time.time()
    bayes_none = W.exact_predictive(rules, ev_leaf)
    bayes_key = W.exact_predictive(rules, ev_leaf,
                                   clamp=(key_level, key_node, ev_z))
    if verbose:
        print(f"[refs] exact per-position Bayes surprisal, both conditions, "
              f"{time.time() - t0:.0f}s", flush=True)

    sub = pr_leaf[:n_ceil]
    sub_z = pr_z[:n_ceil]
    ceil = {
        "key_none":  W.exact_ceilings(rules, sub, key_plen),
        "key_wall":  W.exact_ceilings(rules, sub, key_plen,
                                      clamp=(key_level, key_node, sub_z)),
        "last_none": W.exact_ceilings(rules, sub, last_plen),
        "last_wall": W.exact_ceilings(rules, sub, last_plen,
                                      clamp=(key_level, key_node, sub_z)),
    }
    P0 = W.exact_leaf0_by_key(rules, key_level, key_node)
    D0 = W.jsd_matrix(P0)
    ptl = W.pos_top_level(T, L, s)

    idx = slice(lo, hi)
    out = slice(hi, T)
    gate0 = {
        "index_value_indexed_span": float((bayes_none[idx] - bayes_key[idx]).mean()),
        "index_value_outside_span": float((bayes_none[out] - bayes_key[out]).mean()),
        "index_value_all": float((bayes_none - bayes_key).mean()),
        "index_value_total_nats": float((bayes_none - bayes_key).sum()),
        "index_value_pos0": float(bayes_none[0] - bayes_key[0]),
        "bayes_indexed_span": float(bayes_none[idx].mean()),
        "bayes_outside_span": float(bayes_none[out].mean()),
        "H_key_prior_nats": float(np.log(v)),
    }
    return {
        "T": T, "L": L, "key_plen": int(key_plen), "last_plen": int(last_plen),
        "key_lo": int(lo), "key_hi": int(hi),
        "bayes_none": bayes_none.tolist(), "bayes_key": bayes_key.tolist(),
        "ceil": ceil, "exact_leaf0": P0.tolist(),
        "exact_jsd_mean": float(D0[~np.eye(v, dtype=bool)].mean()),
        "exact_jsd_min_offdiag": float(D0[~np.eye(v, dtype=bool)].min()),
        "pos_top_level": ptl.tolist(), "gate0": gate0,
    }


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600, memory=32768)
def run_arm(tag: str, arm: str, sched: dict, refs: dict, cfg: dict):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces, _probe_acc
    from rhm.practice.fourwall.lm import wall as W

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L, m = cfg["v"], cfg["s"], cfg["depth"], cfg["m"]
    T = s ** L
    V = W.vocab_size(v)
    kind = sched["kind"]
    phase1, rot_period = sched["phase1"], sched["rot_period"]
    rot_step, key_level, key_node = cfg["rot_step"], cfg["key_level"], cfg["key_node"]
    seed = cfg["seed"] + sched["seed_off"]
    B = cfg["batch_size"]
    key_lo, key_hi = refs["key_lo"], refs["key_hi"]
    key_plen, last_plen = refs["key_plen"], refs["last_plen"]

    rules = generate_rules_distinct(v, s, L, m, seed=cfg["rule_seed"])
    bayes_none = np.array(refs["bayes_none"])
    ptl = np.array(refs["pos_top_level"])
    P0_exact = np.array(refs["exact_leaf0"])

    # ---- evaluation apparatus, identical in every arm ----
    ev_leaf, ev_lf, _ = _generate_with_traces(rules, cfg["n_eval"], cfg["eval_seed"])
    ev_z = ev_lf[key_level][:, key_node]
    ev_leaf_t = torch.from_numpy(ev_leaf.astype(np.int64))
    pr_leaf, pr_lf, _ = _generate_with_traces(rules, cfg["n_probe"], cfg["probe_seed"])
    pr_z = pr_lf[key_level][:, key_node]
    pr_leaf_t = torch.from_numpy(pr_leaf.astype(np.int64))
    anc = {"key": W.anc_index(key_plen, L, s), "last": W.anc_index(last_plen, L, s)}
    y_lvl = {a: {ell: torch.from_numpy(pr_lf[ell][:, anc[a][ell]].astype(np.int64)).to(device)
                 for ell in range(L)} for a in anc}
    # `none` is APPENDED, never inserted: the per-mode RNG seed is keyed on the mode's
    # index, so `rand`'s draw stays bit-identical to `fwlm0`'s.
    MODES = ["true", "rand", "perm", "old", "none"]
    merge_at = sched.get("merge_at", cfg["max_steps"] + 1)

    all_blocks = ["post_embed"] + [f"post_block{i}" for i in range(cfg["n_layer"])]
    pblocks = [b for b in cfg["probe_blocks"].split(",") if b in all_blocks] or all_blocks

    def wcol(leaf_np, z_np, q, q_prev, mode, mode_i):
        """The wall column for an eval condition. Fixed RNG per (mode, arm) so `rand`
        is the SAME draw at every checkpoint and contributes no drift of its own.

        `none` is the MERGED condition -- the neutral filler, for every arm kind alike,
        which is exactly what `no_wall` carries from step 0 and what a merged arm carries
        after `merge_at`."""
        if mode == "none":
            return torch.full((leaf_np.shape[0],), W.neutral_tok(v), dtype=torch.long)
        rng = np.random.default_rng(cfg["eval_seed"] + 7717 * (mode_i + 1)
                                    + 131 * sched["seed_off"])
        w = W.wall_values(kind, z_np, q, v, rng=rng, mode=mode,
                          q_prev=q_prev, delta=rot_step)
        if w is None:
            return torch.full((leaf_np.shape[0],), W.neutral_tok(v), dtype=torch.long)
        return torch.from_numpy((W.wall_tok(w, v)).astype(np.int64))

    def make_x(leaf_t, wc):
        return torch.cat([wc[:, None], leaf_t[:, :-1]], 1)

    @torch.no_grad()
    def nll_per_pos(leaf_t, wc, chunk=256):
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
    def wall_conditionals():
        """P(x_0 | w = i) for every wall id, from a ONE-TOKEN context -- the only wall
        readout with an exactly computable reference (`refs['exact_leaf0']`), since an
        empty prefix cannot contradict a clamped latent."""
        if kind == "none":
            ids = torch.full((1, 1), W.neutral_tok(v), dtype=torch.long, device=device)
        else:
            ids = (torch.arange(v, device=device) + v)[:, None]
        logits, _ = model(ids)
        p = torch.softmax(logits[:, 0, :v].float(), -1).cpu().numpy()
        return np.repeat(p, v, axis=0) if kind == "none" else p

    z_groups = [np.where(ev_z == a)[0] for a in range(v)]
    # the DGP's own marginal over the keyed latent is mildly skewed and can put a
    # feature at exactly zero mass, so every per-cell readout runs over LIVE cells only
    live = np.array([len(g) >= cfg["min_cell_n"] for g in z_groups])
    card_sel = torch.from_numpy(np.arange(min(cfg["n_card"], ev_leaf.shape[0])))

    @torch.no_grad()
    def transfer_and_cardinality(chunk=256):
        """The FORCED-TRANSFER matrix and the MDL/cardinality readout, in one sweep.

        E[i, a] = mean NLL on the indexed span for instances whose true keyed latent is
        `a` when the model is handed wall id `i`. This is the idea doc §5's route (b) --
        retrieval demanded under a key with no entry -- and, unlike `../`'s discrete
        substrate where every entry went 1.000 -> 0.000, it is GRADED, so the
        frame-invariant and contaminated components of a unit separate.

        `argmin_i E[i, a]` is the reader's own implied z->w routing table; comparing it
        against the current, previous and phase-1 maps is what makes "re-key" (the table
        migrates), "merge" (the table stops mattering) and "stale" (it keeps the old
        addresses) distinguishable outcomes rather than assigned ops.

        The cardinality half keeps the full predictive distributions for `n_card`
        instances over the indexed span and asks how many distinct BEHAVIOURS the wall
        induces: v classes = a fully keyed index, 1 class = the index merged away.
        """
        n = ev_leaf_t.shape[0]
        nc = card_sel.shape[0]
        Ei = np.zeros((v, v)); Eo = np.zeros((v, v))
        Pc = torch.zeros(v, nc, key_hi - key_lo, v, device=device)
        for i in range(v):
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
        # pairwise behavioural distance between wall ids (mean JSD over the span)
        D = np.zeros((v, v))
        lP = torch.log(Pc.clamp_min(1e-12))
        for i in range(v):
            M = 0.5 * (Pc[i][None] + Pc)
            lM = torch.log(M.clamp_min(1e-12))
            d = 0.5 * ((Pc[i][None] * (lP[i][None] - lM)).sum(-1)
                       + (Pc * (lP - lM)).sum(-1))
            D[i] = d.mean(dim=(1, 2)).cpu().numpy()
        return Ei, Eo, np.maximum(D, 0.0)

    def acts_at(leaf_t, wc, pos, blocks, chunk=256):
        out = {b: [] for b in blocks}
        with torch.no_grad():
            for i in range(0, leaf_t.shape[0], chunk):
                x = make_x(leaf_t[i:i + chunk], wc[i:i + chunk]).to(device)
                _, _, inter = model(x, return_intermediates=True)
                for b in blocks:
                    out[b].append(inter[b][:, pos, :].float())
        return {b: torch.cat(vs) for b, vs in out.items()}

    def probe_levels(wc, anchor, pos, blocks, full):
        a = acts_at(pr_leaf_t, wc, pos, blocks)
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

    # ---- checkpoint schedule: a common grid every `ckpt_every`, plus tight
    #      post-rotation points so "step vs smear" is resolvable within an era ----
    max_steps = cfg["max_steps"]
    rots = W.rotation_steps(phase1, rot_period, max_steps)
    cheap = set(range(0, max_steps + 1, cfg["ckpt_every"])) | {max_steps}
    if rot_period >= 4 * cfg["post_rot_b"]:      # a fast rotator is already dense enough
        for r in rots:
            cheap |= {r + d for d in (cfg["post_rot_a"], cfg["post_rot_b"])
                      if r + d <= max_steps}
    if merge_at <= max_steps:                    # resolve the op's own transient
        cheap |= {merge_at + d for d in (0, 25, 50, 75) if merge_at + d <= max_steps}
    cheap = sorted(cheap)
    probe_ck = sorted({int(x) for x in cfg["probe_ckpts"].split(",") if int(x) <= max_steps}
                      | {max_steps} | ({merge_at} if merge_at <= max_steps else set()))
    full_ck = {int(x) for x in cfg["full_ckpts"].split(",") if int(x) <= max_steps} | {max_steps}

    torch.manual_seed(seed)
    model = GPT(V, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    train_rng = np.random.default_rng(seed + 5551)

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    print(f"===== arm {arm}  kind={kind} phase1={phase1} rot_period={rot_period} "
          f"seed={seed}  rotations at {rots[:8]}{'...' if len(rots) > 8 else ''} =====",
          flush=True)

    log, pool_leaf, pool_z, started = [], None, None, time.time()
    for step in range(max_steps + 1):
        if step in cheap:
            model.eval()
            q = W.q_at(step, phase1, rot_period, rot_step, v)
            qp = W.q_prev_at(step, phase1, rot_period, rot_step, v)
            merged = step >= merge_at
            rec = {"step": step, "tokens": step * B * T, "q": int(q), "q_prev": int(qp),
                   "era": int(W.era_at(step, phase1, rot_period)),
                   "merged": bool(merged),
                   # which eval condition IS this arm's consumption at this step
                   "consumed": "none" if (merged or kind == "none") else "true"}

            # ---- task error + the index instruments, one pass per wall condition ----
            npos = {}
            for mi, mode in enumerate(MODES):
                npos[mode] = nll_per_pos(ev_leaf_t, wcol(ev_leaf, ev_z, q, qp, mode, mi))
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
            # binding: what the reader captures of the exact oracle index bracket
            rec["binding"] = {
                "idx": float(npos["rand"][key_lo:key_hi].mean()
                             - npos["true"][key_lo:key_hi].mean()),
                "out": float(npos["rand"][key_hi:].mean() - npos["true"][key_hi:].mean()),
                "pos0": float(npos["rand"][0] - npos["true"][0]),
                "misleading_idx": float(npos["perm"][key_lo:key_hi].mean()
                                        - npos["true"][key_lo:key_hi].mean()),
                "stale_idx": float(npos["true"][key_lo:key_hi].mean()
                                   - npos["old"][key_lo:key_hi].mean()),
                # the instrument that matches Gate 0's DEFINITION exactly: a right key
                # against NO key, rather than against a wrong one. Post-merge this is the
                # residual value of the abandoned wall circuitry, and it should decay.
                "vs_none_idx": float(npos["none"][key_lo:key_hi].mean()
                                     - npos["true"][key_lo:key_hi].mean()),
                "vs_none_out": float(npos["none"][key_hi:].mean()
                                     - npos["true"][key_hi:].mean()),
            }

            # ---- forced transfer, the implied routing table, and the MDL readout ----
            Ei, Eo, D = transfer_and_cardinality()
            off = ~np.eye(v, dtype=bool)
            ar = np.arange(v)
            implied = Ei.argmin(0)                     # latent a -> the wall it prefers
            cur, old, ph1 = (ar + q) % v, (ar + qp) % v, ar
            diag_cur, diag_old = Ei[cur, ar], Ei[old, ar]
            offmean = np.array([Ei[[i for i in range(v) if i != cur[a]], a].mean()
                                for a in range(v)])
            offmean_o = np.array([Eo[[i for i in range(v) if i != cur[a]], a].mean()
                                  for a in range(v)])
            rec["index"] = {
                "n_live": int(live.sum()),
                "transfer_gap": float((offmean - diag_cur)[live].mean()),
                "e_cur": float(diag_cur[live].mean()), "e_old": float(diag_old[live].mean()),
                "e_off": float(offmean[live].mean()),
                "e_best": float(Ei.min(0)[live].mean()),
                "map_match_cur": float((implied == cur)[live].mean()),
                "map_match_old": float((implied == old)[live].mean()),
                "map_match_phase1": float((implied == ph1)[live].mean()),
                "match_cur_per_latent": (implied == cur).astype(int).tolist(),
                "adv_cur_per_latent": (offmean - diag_cur).tolist(),
                "out_span_transfer_gap": float(
                    (offmean_o - Eo[cur, ar])[live].mean()),
                "jsd_mean": float(D[off].mean()),
                "card": {f"tau{t}": int(W.cluster_count(D, t))
                         for t in (0.0002, 0.001, 0.005, 0.02)},
                "E_idx": Ei.tolist(),
            }
            # the one wall readout with an exactly computable reference (empty prefix)
            Pm = wall_conditionals()
            D0m = W.jsd_matrix(Pm)
            rec["index"]["jsd0_mean"] = float(D0m[off].mean())
            rec["index"]["jsd0_ratio"] = float(
                D0m[off].mean() / max(refs["exact_jsd_mean"], 1e-12))

            # ---- typed next-level readout, paired probe / probe-free ----
            if step in probe_ck:
                full = step in full_ck
                blocks = all_blocks if full else pblocks
                lv = {}
                for mode in ("true", "rand", "none"):
                    wc = wcol(pr_leaf, pr_z, q, qp, mode, MODES.index(mode))
                    lv[mode] = {"key": probe_levels(wc, "key", key_plen, blocks, full),
                                "last": probe_levels(wc, "last", last_plen, blocks, full)}
                rec["levels"], rec["full_probe"] = lv, bool(full)

            log.append(rec)
            lvtxt = ""
            if "levels" in rec:
                lt = rec["levels"]["true"]["key"]; lr = rec["levels"]["rand"]["key"]
                ln = rec["levels"]["none"]["key"]
                lvtxt = ("  key-anchor d4 true/rand/none "
                         f"{lt['d4']:.3f}/{lr['d4']:.3f}/{ln['d4']:.3f}")
            cons = rec["consumed"]
            print(f"[{arm:15s} s{step:6d} q{q:2d}{'M' if merged else ' '}] "
                  f"nll idx {rec['nll'][cons]['idx']:.4f} "
                  f"out {rec['nll'][cons]['out']:.4f} | bind {rec['binding']['idx']:+.4f} "
                  f"vsnone {rec['binding']['vs_none_idx']:+.4f} | gap "
                  f"{rec['index']['transfer_gap']:+.4f} jsd {rec['index']['jsd_mean']:.4f} "
                  f"card {rec['index']['card']['tau0.001']:2d} "
                  f"map {rec['index']['map_match_cur']:.2f}/{rec['index']['map_match_old']:.2f}"
                  + lvtxt, flush=True)
            with open(os.path.join(outdir, f"{arm}.json"), "w") as fh:
                json.dump({"arm": arm, "sched": sched, "log": log,
                           "complete": step == max_steps}, fh, indent=2, cls=NumpyEncoder)
            volume.commit()
            model.train()

        if step == max_steps:
            break

        if step % cfg["fresh_every"] == 0:
            need = max(64, B * cfg["fresh_every"])
            fl, flf, _ = _generate_with_traces(
                rules, need, cfg["data_seed"] + 100_003 * (step // cfg["fresh_every"] + 1))
            pool_leaf = torch.from_numpy(fl.astype(np.int64))
            pool_z = torch.from_numpy(flf[key_level][:, key_node].astype(np.int64))

        # `ix` is drawn FIRST and unconditionally, so the merge branch below cannot move
        # the batch-sampler stream: an arm is bit-identical to its un-merged twin at the
        # same seed for every step before `merge_at`.
        ix = torch.randint(0, pool_leaf.shape[0], (B,), generator=gen)
        leaf_b, z_b = pool_leaf[ix], pool_z[ix]
        q = W.q_at(step, phase1, rot_period, rot_step, v)
        if step >= merge_at or kind == "none":
            # THE MERGE OP: sixteen addresses collapse to one. Nothing but the input
            # stream changes -- no parameter is touched, and after this step the arm's
            # batches are token-for-token identical to `no_wall`'s.
            wc = torch.full((B,), W.neutral_tok(v), dtype=torch.long)
        elif kind == "dead":
            wc = torch.from_numpy(
                (W.wall_tok(train_rng.integers(0, v, size=B), v)).astype(np.int64))
        else:
            wc = W.wall_tok((z_b + q) % v, v)
        x, y = make_x(leaf_b, wc).to(device), leaf_b.to(device)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % cfg["log_interval"] == 0:
            print(f"  {arm} {step:6d} ntp {loss.item():.4f}", flush=True)

    print(f"[{arm}] DONE in {time.time() - started:.0f}s", flush=True)
    return {"arm": arm, "elapsed": time.time() - started, "n_ckpt": len(log)}


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, timeout=43200, memory=16384)
def wall_lm(
    tag: str = "smoke",
    arms: str = "true_wall,wall,wall_fast,dead_wall,no_wall,dead_wall_b",
    # DGP — the arc's regime (`reread/lm`'s, so its level dynamic-range reference transfers)
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    key_level: int = 2, key_node: int = 0,
    # model
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    # training
    max_steps: int = 20000, batch_size: int = 64, lr: float = 3e-4,
    weight_decay: float = 0.01, data_seed: int = 7, seed: int = 42,
    fresh_every: int = 500,
    # the wall schedule
    phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
    # measurement
    n_eval: int = 4096, eval_seed: int = 999,
    n_probe: int = 4000, probe_seed: int = 1001, n_card: int = 256,
    min_cell_n: int = 24,
    ckpt_every: int = 250, post_rot_a: int = 50, post_rot_b: int = 100,
    probe_ckpts: str = ("250,1000,2000,4000,6000,8000,8250,8500,9000,10000,"
                        "12000,14000,14250,16000,18000"),
    full_ckpts: str = "8000",
    probe_steps: int = 600, probe_lr: float = 1e-2,
    mlp_hidden: int = 128, mlp_steps: int = 800,
    probe_blocks: str = "post_embed,post_block2,post_block4,post_block6,post_block7",
    log_interval: int = 2000, merge_override: int = 0, quick: bool = False,
):
    import numpy as np

    if quick:
        max_steps, n_eval, n_probe = 600, 768, 384
        phase1, rot_period, fresh_every = 200, 200, 100
        ckpt_every, probe_steps, mlp_steps = 100, 100, 100
        probe_ckpts, full_ckpts, log_interval = "200,400", "600", 200
        n_card, min_cell_n = 128, 8
        merge_override = 200
        if len(arms.split(",")) > 3:
            arms = "wall,dead_wall,no_wall"

    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    for a in arm_list:
        if a not in ARM_SPECS:
            raise ValueError(f"unknown arm {a}; known: {sorted(ARM_SPECS)}")

    cfg = dict(v=v, s=s, depth=depth, m=m, rule_seed=rule_seed, key_level=key_level,
               key_node=key_node, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
               max_steps=max_steps, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, data_seed=data_seed, seed=seed,
               fresh_every=fresh_every, rot_step=rot_step,
               phase1=phase1, rot_period=rot_period, n_eval=n_eval,
               eval_seed=eval_seed, n_probe=n_probe, probe_seed=probe_seed, n_card=n_card,
               min_cell_n=min_cell_n,
               ckpt_every=ckpt_every, post_rot_a=post_rot_a, post_rot_b=post_rot_b,
               probe_ckpts=probe_ckpts, full_ckpts=full_ckpts, probe_steps=probe_steps,
               probe_lr=probe_lr, mlp_hidden=mlp_hidden, mlp_steps=mlp_steps,
               probe_blocks=probe_blocks, log_interval=log_interval)

    print(f"{'=' * 78}\nfourwall/lm  tag={tag}  {setting_key(v, s, depth, m)}  "
          f"{n_layer}L/{n_head}H/{n_embd}D  T={s ** depth}")
    print(f"  arms={arm_list}  max_steps={max_steps} "
          f"(tokens={max_steps * batch_size * s ** depth:,})")
    print(f"  key = level {key_level} node {key_node}; phase1={phase1} "
          f"rot_period={rot_period} rot_step={rot_step}\n{'=' * 78}", flush=True)

    refs = build_refs(v, s, depth, m, rule_seed, key_level, key_node,
                      n_eval, eval_seed, n_probe, probe_seed)
    print("\n--- GATE 0 (exact, model-free): what a right index is worth here ---")
    for k, val in refs["gate0"].items():
        print(f"  {k:28s} {val:+.4f}")
    print(f"  probe ceilings, key anchor  no-wall {refs['ceil']['key_none']}")
    print(f"  probe ceilings, key anchor  wall    {refs['ceil']['key_wall']}")
    print(f"  probe ceilings, last anchor no-wall {refs['ceil']['last_none']}")
    print(f"  exact wall-conditional JSD mean {refs['exact_jsd_mean']:.4f} "
          f"(min off-diagonal {refs['exact_jsd_min_offdiag']:.4f})\n", flush=True)

    scheds = {a: arm_schedule(a, phase1, rot_period, max_steps, merge_override)
              for a in arm_list}
    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "arms": arm_list, "scheds": scheds, "refs": refs,
                   "rotations": {a: W_rot(scheds[a], max_steps) for a in arm_list}},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    started = time.time()
    handles = [(a, run_arm.spawn(tag, a, scheds[a], refs, cfg)) for a in arm_list]
    out = {}
    for a, h in handles:
        try:
            out[a] = h.get()
        except Exception as exc:                              # noqa: BLE001
            out[a] = {"arm": a, "error": repr(exc)}
            print(f"[driver] arm {a} FAILED: {exc!r}", flush=True)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n{json.dumps(out, indent=2)}\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}\n{json.dumps(out, indent=2)}")
    return out


def W_rot(sched, max_steps):
    from rhm.practice.fourwall.lm import wall as W
    return W.rotation_steps(sched["phase1"], sched["rot_period"], max_steps)


# --------------------------------------------------------------------------- #
# gates (CPU, no GPU, no substrate)
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=3600, memory=16384)
def gate(v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
         key_level: int = 2, key_node: int = 0, n_eval: int = 2048, eval_seed: int = 999,
         n_probe: int = 2000, probe_seed: int = 1001,
         phase1: int = 8000, rot_period: int = 2000, rot_step: int = 4,
         max_steps: int = 20000):
    """G-1  the wall map is a bijection and every rotation is a DERANGEMENT.
       G-2  the schedule: stationary through phase 1; the identity returns on cue.
       G-3  the wall carries the latent exactly, and `rand` carries none of it.
       G-4  `exact_predictive` unclamped == the donor's own `prefix_beliefs` leaf posterior.
       G-5  clamping the keyed node pins its posterior at exactly 1.0. (It also shifts
             the other level-2 nodes' posteriors through the root -- reported, not
             asserted: the admissibility claim is about PREDICTIONS, and that is G-7.)
       G-6  token layout: leaf / wall / neutral roles never collide.
       G-7  the exact index is worth something, and only inside the span it addresses --
             the within-sequence admissibility control, measured in nats.
       GATE 0  the exact oracle bracket: what a right index is worth, per position.
    """
    import numpy as np
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.conditional_revision.oracle import prefix_beliefs
    from rhm.practice.fourwall.lm import wall as W

    L, T = depth, s ** depth
    rules = generate_rules_distinct(v, s, L, m, seed=rule_seed)
    out = {"setting": setting_key(v, s, L, m), "T": T}

    z = np.arange(v)
    out["G1"] = {
        "bijection": bool(all(len(set(((z + q) % v).tolist())) == v for q in range(v))),
        "derangements": bool(all(W.is_derangement(k * rot_step, v)
                                 for k in range(1, v // max(1, np.gcd(rot_step, v))))),
        "orbit_len": int(v // np.gcd(rot_step, v)),
    }
    assert out["G1"]["bijection"] and out["G1"]["derangements"], "G-1"

    rots = W.rotation_steps(phase1, rot_period, max_steps)
    qs = [W.q_at(t, phase1, rot_period, rot_step, v)
          for t in [0, phase1 - 1] + [r for r in rots]]
    out["G2"] = {"rotations": rots, "q_at_0": qs[0], "q_at_phase1_minus_1": qs[1],
                 "q_sequence": qs[2:],
                 "identity_return_steps": [r for r in rots
                                           if W.q_at(r, phase1, rot_period, rot_step, v) == 0]}
    assert qs[0] == 0 and qs[1] == 0, "G-2 phase 1 must be stationary"

    leaf, lf, _ = _generate_with_traces(rules, n_eval, eval_seed)
    zk = lf[key_level][:, key_node]
    rng = np.random.default_rng(0)
    w_true = W.wall_values("wall", zk, 3, v, mode="true")
    w_rand = W.wall_values("wall", zk, 3, v, rng=rng, mode="rand")
    mi = lambda a, b: float(_mutinf(a, b, v))
    out["G3"] = {"true_is_bijection_of_z": bool(np.array_equal((w_true - 3) % v, zk)),
                 "MI_true_nats": mi(w_true, zk), "MI_rand_nats": mi(w_rand, zk),
                 "H_z_nats": float(_ent1(zk, v))}
    assert out["G3"]["true_is_bijection_of_z"], "G-3"
    assert out["G3"]["MI_rand_nats"] < 0.05, "G-3 rand must carry no key"

    sub = leaf[:256]
    mine = W.exact_predictive(rules, sub)
    ref = np.zeros(T)
    for plen in range(T):
        _, _, lp = prefix_beliefs(rules, sub, plen, 0)
        ref[plen] = -np.log(np.clip(lp[np.arange(sub.shape[0]), sub[:, plen]],
                                    1e-30, None)).mean()
    out["G4"] = {"max_abs_delta": float(np.abs(mine - ref).max()),
                 "mean_surprisal": float(mine.mean())}
    assert out["G4"]["max_abs_delta"] < 1e-9, "G-4"

    from rhm.conditional_revision.oracle import _upward, _downward, _leaf_evidence, _norm
    nev = W._clamp_evidence(rules, key_level, key_node, zk[:256], 256, v)
    up = _upward(_leaf_evidence(sub, 0, v), rules, nev=nev)
    down = _downward(up, rules, nev=nev)
    post2 = _norm(up[key_level] * down[key_level])
    up0 = _upward(_leaf_evidence(sub, 0, v), rules)
    down0 = _downward(up0, rules)
    post2_0 = _norm(up0[key_level] * down0[key_level])
    n_nodes = s ** key_level
    others = [j for j in range(n_nodes) if j != key_node]
    out["G5"] = {"keyed_node_posterior": float(post2[np.arange(256), key_node, zk[:256]].mean()),
                 "other_nodes_max_abs_delta": float(
                     np.abs(post2[:, others, :] - post2_0[:, others, :]).max()),
                 "n_level_nodes": int(n_nodes), "n_unindexed_nodes": len(others)}
    assert out["G5"]["keyed_node_posterior"] > 0.999, "G-5"

    out["G6"] = {"vocab": W.vocab_size(v),
                 "leaf_max": int(leaf.max()), "wall_range": [v, 2 * v - 1],
                 "neutral": W.neutral_tok(v),
                 "disjoint": bool(int(leaf.max()) < v)}
    assert out["G6"]["disjoint"], "G-6"

    refs = build_refs(v, s, depth, m, rule_seed, key_level, key_node,
                      n_eval, eval_seed, n_probe, probe_seed, n_ceil=min(n_probe, 2000))
    g0 = refs["gate0"]
    out["G7"] = {  # the index is worth something, and ONLY inside the span it addresses
        "index_value_indexed_span": g0["index_value_indexed_span"],
        "index_value_outside_span": g0["index_value_outside_span"],
        "concentration_ratio": float(g0["index_value_indexed_span"]
                                     / max(abs(g0["index_value_outside_span"]), 1e-9)),
    }
    assert out["G7"]["index_value_indexed_span"] > 0.05, "G-7 index worth too little"
    assert out["G7"]["concentration_ratio"] > 10.0, "G-7 index leaks outside its span"

    # NOTE, recorded so the reduction knows the instrument's ceiling: the EMPTY-PREFIX
    # wall-conditional distributions (the one wall readout with an exactly computable
    # reference) are only weakly separated -- P(x_0 | z=a) barely depends on a. The
    # primary cardinality/merge readout is therefore the span-wide forced-transfer
    # sweep, which has no exact reference but is measured against live null arms.
    P0 = W.exact_leaf0_by_key(rules, key_level, key_node)
    D0 = W.jsd_matrix(P0)
    off = ~np.eye(v, dtype=bool)
    out["jsd0_reference"] = {
        "card_tau0.001": int(W.cluster_count(D0, 0.001)),
        "card_tau0.005": int(W.cluster_count(D0, 0.005)),
        "card_tau0.02": int(W.cluster_count(D0, 0.02)),
        "jsd_mean": float(D0[off].mean()), "jsd_min": float(D0[off].min())}

    ml = np.bincount(lf[key_level][:, key_node], minlength=v) / n_eval
    out["key_demand"] = {"mass": ml.tolist(), "n_zero": int((ml == 0).sum()),
                         "H_nats": float(_ent1(lf[key_level][:, key_node], v)),
                         "H_max_nats": float(np.log(v))}
    out["gate0"] = g0
    out["ceilings"] = refs["ceil"]
    bn, bk = np.array(refs["bayes_none"]), np.array(refs["bayes_key"])
    out["gate0"]["index_value_first8"] = (bn - bk)[:8].tolist()
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


def _ent1(a, v):
    import numpy as np
    p = np.bincount(a, minlength=v) / len(a)
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def _mutinf(a, b, v):
    import numpy as np
    j = np.zeros((v, v))
    np.add.at(j, (a, b), 1.0)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return (j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum()


@app.local_entrypoint()
def main(quick: bool = True):
    wall_lm.remote(quick=quick, tag="smoke_local")
