"""reread — is a FIXED archive renewable to a learner whose vocabulary has climbed?

SPEC: `SPEC.md` in this folder. Machinery donor: `../ratchet/` (imported, never modified).

THE ONE STRUCTURAL CHANGE FROM `ratchet`: the archive is drawn ONCE and FROZEN. The same
damage instances, in the same order, re-presented pass after pass. `ratchet` drew fresh
practice instances every cycle (`seed + 100_000 + 1000*cyc`), so nothing in the arc has ever
held the corpus fixed and asked what a second reading yields.

WHAT A PASS IS. One complete traversal of the archive, in `n_arch // chunk` cycles of `chunk`
instances each, in fixed order. A cycle is `ratchet`'s cycle verbatim (practice beam -> mine
from the beam's own chosen answer on instances it solved -> plant fine-tune -> value steps),
with two deliberate deviations, both forced by the question:

  * `mine_cap = 0` (uncapped). `ratchet` capped mining at 8 configurations per cycle so its
    16-tip beam could not saturate a 14-entry table in one cycle. Here a pass MEANS the whole
    archive is read, so a cap would turn "yield per pass" into a sampling artefact.
  * mining is not restricted to `era_level + 1`. `ratchet` mined only the level being earned
    so that era k could not hand era k+1 a finished vocabulary. Here every level is mined
    every pass from every solved instance, and the ONLY thing gating level l is the structural
    ratchet at build time (`T[l]` is defined over `T[l-1]` ENTRIES, so a chunk whose halves are
    not both in the lower vocabulary is dropped) plus what the agent could actually solve.
    That is the mechanism under test; scheduling it away would beg the question.

THE ARCHIVE HAS A DEPTH COORDINATE. Instances are drawn from the `level_ladder` damage cells
(L1n6 / L2n3 / L3n1) in a declared mix. `shallow` = all L1n6, `mixed` = equal thirds,
`deep` = all L3n1. Mining reads, per instance, the level-l span CONTAINING THAT INSTANCE'S
damage cell — `ratchet`'s convention generalised to a mixed archive.

ARMS (archive policy x vocabulary policy):
    reread        frozen archive, earned vocabulary, commits on a fixed pass schedule
    fresh         a NEW matched-size archive every pass, same vocabulary schedule  [novelty control]
    dense_reread  frozen archive, base level-1 moves forever                       [monolith control]
    dense_fresh   fresh archive every pass, base moves forever
    given_reread  frozen archive, the DGP's own tables from pass 1                 [ceiling]
    fid_ratchet   FIDELITY GATE: shallow archive, fresh draws, mine_cap=8, level-2 mining only
                  -- `ratchet`'s era-1 configuration, whose published numbers this must
                  reproduce before anything from a later pass is trusted.

YIELD ACCOUNTING (the spec's second caution). Three flavours, all logged:
    n_entries    raw table size
    n_correct    entries that are real grammar tuples (oracle)
    n_used       distinct entries the max-sum DP actually SELECTS on consumption-matched
                 held-out damage -- an entry re-derivable from `T[l-1]` by composition but
                 never selected contributes zero. `d_aud` (the pass-over-pass change in that
                 audition) is the behavioural readout the count is checked against.
Controls: `rand_k` (matched-size random subset of the TRUE table -- `ratchet`'s
concentration-vs-coverage instrument) and `comp_k` (matched-size random subset of the
COMPOSITION CLOSURE `T[l-1]^s` -- "what you could have written down without the archive"),
plus `closure` (the full closure) .

THE PAIRED NOVELTY PROBE (the load-bearing instrument). At the end of every pass, with the
agent's state held fixed, it re-solves (a) the frozen archive and (b) a fresh matched-size
draw, mines each with a THROWAWAY miner against its own operative lower tables, and reports
both yields. Neither is acted on. This puts the fresh-vs-reread contrast INSIDE one agent at
one vocabulary stage, so between-arm drift cannot carry it.

GROUND TRUTH. The archive's own level-l content -- the set of true level-1 flat tuples over
each instance's damage-containing span, read off the CLEAN derivation with the exact inverse
map -- is computed once. `frac_content` is then literally "what fraction of the archive's
level-l structure has been extracted by pass k".

Run from experiments/:
  modal run rhm/practice/reread/reread.py::gate
  modal run rhm/practice/reread/reread.py::reread --quick --tag smoke0
  python3 rhm/practice/reread/launch_detached.py --fn reread --tag rr_s0 ...
"""

import copy
import hashlib
import json
import os
import time

import modal
import numpy as np

from rhm.shared import DATA_DIR, NumpyEncoder, image, volume
from rhm.rhm_sculpt_planner import _sample_pool
from rhm.practice.crystallize.units import grade
from rhm.practice.ratchet import macros as MC
from rhm.practice.ratchet.ratchet import (
    beam_moves, build_ms, build_shared, context_instances, era_ctx, finetune_generator,
    fit_width, plant_probe, priced, push, selfcheck as ratchet_selfcheck, value_steps,
    _cfg as ratchet_cfg)


app = modal.App("rhm-practice-reread", image=image)

REMOTE = "rhm_practice_reread"

# --------------------------------------------------------------------------- #
# the archive
# --------------------------------------------------------------------------- #

CELL_ORDER = ["L1n6", "L2n3", "L3n1"]
CELLS = {"L1n6": {"level": 1, "node": 6, "name": "L1n6"},
         "L2n3": {"level": 2, "node": 3, "name": "L2n3"},
         "L3n1": {"level": 3, "node": 1, "name": "L3n1"}}

MIXES = {
    "shallow": (1.0, 0.0, 0.0),
    "mixed":   (1 / 3, 1 / 3, 1 / 3),
    "deep":    (0.0, 0.0, 1.0),
}


def mix_counts(mix, n):
    w = np.asarray(MIXES[mix], float)
    k = np.floor(w * n).astype(int)
    k[int(np.argmax(w))] += n - k.sum()
    return k


def span_node(cell, ell, s):
    """The level-`ell` node whose span CONTAINS this instance's damage cell. `ratchet`'s
    `node = (era_node * s**(era_level-1)) // s**(ell-1)`, per instance."""
    return (cell["node"] * s ** (cell["level"] - 1)) // (s ** (ell - 1))


def build_archive(rules, cfg, mix, n, seed):
    """One frozen draw: `n` damage instances in the declared depth mix, in a fixed order.
    `clean` is kept for the ground-truth content accounting (oracle, never fed to the agent)."""
    s, depth, v, m = cfg["s"], cfg["depth"], cfg["v"], cfg["m"]
    ks = mix_counts(mix, n)
    R, X, C, K = [], [], [], []
    for i, k in enumerate(ks):
        if k == 0:
            continue
        cell = CELLS[CELL_ORDER[i]]
        r, x, cl = context_instances(rules, era_ctx(cell), int(k), s, depth, v, m,
                                     seed=seed + 1009 * (i + 1), with_clean=True)
        R.append(r); X.append(x); C.append(cl); K.append(np.full(int(k), i, np.int64))
    R, X, C, K = np.concatenate(R), np.concatenate(X), np.concatenate(C), np.concatenate(K)
    perm = np.random.default_rng(seed + 7).permutation(len(R))       # fixed presentation order
    return {"roots": R[perm], "x": X[perm], "clean": C[perm], "cell": K[perm],
            "mix": mix, "n": int(n), "seed": int(seed),
            "sha": hashlib.sha1(X[perm].tobytes()).hexdigest()[:16]}


def archive_content(arch, inverse_bottom, cfg, levels):
    """GROUND TRUTH: the set of level-1 flat tuples the archive actually contains at each
    level, over each instance's damage-containing span, read off the CLEAN derivation."""
    s = cfg["s"]
    feats = MC.exact_features(arch["clean"], inverse_bottom, cfg["v"], s)      # (N, n_blocks)
    out = {}
    for ell in levels:
        span = s ** (ell - 1)
        keys, per = set(), []
        for i in range(feats.shape[0]):
            cell = CELLS[CELL_ORDER[int(arch["cell"][i])]]
            nd = span_node(cell, ell, s)
            t = tuple(int(z) for z in feats[i, nd * span:(nd + 1) * span])
            per.append(t)
            if min(t) >= 0:
                keys.add(t)
        out[ell] = {"set": keys, "per": per}
    return out


# --------------------------------------------------------------------------- #
# mining over a MIXED archive
# --------------------------------------------------------------------------- #

def mine_into(miners, reader, x_solved, cell_of, cfg, levels):
    """`ratchet`'s mining step, per damage cell: the agent's own read (the frozen reader) of
    configurations it SOLVED, over the level-l span containing that instance's damage."""
    if x_solved.shape[0] == 0:
        return 0
    pf = MC.parse_features(reader, x_solved, s=cfg["s"]).cpu().numpy()
    s = cfg["s"]
    for ci in np.unique(cell_of):
        rows = np.nonzero(cell_of == ci)[0]
        cell = CELLS[CELL_ORDER[int(ci)]]
        for ell in levels:
            span = s ** (ell - 1)
            nd = span_node(cell, ell, s)
            miners[ell].observe(pf[rows, nd * span:(nd + 1) * span])
    return int(x_solved.shape[0])


# --------------------------------------------------------------------------- #
# tables: the composition-closure control and the flat-set helpers
# --------------------------------------------------------------------------- #

def flats(table):
    return {tuple(int(z) for z in r) for r in table["flat"]}


def closure_table(level, lower, s, cap=1024):
    """`T[l-1]^s` -- every composition of the operative lower vocabulary. The literal
    "re-derivable from `T[l-1]` alone" set: an archive that adds nothing beyond this taught
    the agent nothing at level l."""
    n = lower["child"].shape[0]
    if n == 0:
        return None
    grid = np.array(np.meshgrid(*[np.arange(n)] * s, indexing="ij")).reshape(s, -1).T
    if grid.shape[0] > cap:
        grid = grid[np.sort(np.random.default_rng(0).permutation(grid.shape[0])[:cap])]
    return MC.make_table(level, grid.astype(np.int64), lower, s)


def audit(generator, x, roots_np, move, rules, canon, s):
    """One macro action on held-out damage, graded — plus WHICH entries the max-sum DP
    actually selected. `used` is the non-double-counting instrument: an entry that is in the
    table but never chosen changes no repair behaviour at its level."""
    feats, pos = MC.macro_features(generator, x, move, s, None)
    tup = canon[feats]
    new = x.clone()
    new.scatter_(1, pos, tup.reshape(x.shape[0], -1))
    succ, dres = grade(new.cpu().numpy(), roots_np, rules, s)
    fn = feats.cpu().numpy()
    used = {tuple(int(z) for z in r) for r in fn}
    return {"e": 1.0 - float(succ.mean()), "dres": float(dres.mean()), "used": used}


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #

ARMS = {
    "reread":       {"vocab": "earned", "archive": "frozen", "commit": True},
    "fresh":        {"vocab": "earned", "archive": "fresh",  "commit": True},
    "dense_reread": {"vocab": "base",   "archive": "frozen", "commit": False},
    "dense_fresh":  {"vocab": "base",   "archive": "fresh",  "commit": False},
    "given_reread": {"vocab": "true",   "archive": "frozen", "commit": False},
    # the fidelity gate: `ratchet`'s era-1 configuration on this loop
    "fid_ratchet":  {"vocab": "earned", "archive": "fresh",  "commit": False,
                     "mine_cap": 8, "mine_levels": (2,)},
}


def parse_arms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


# --------------------------------------------------------------------------- #
# one (mix, arm) cell
# --------------------------------------------------------------------------- #

def run_cell(mix, arm, shared, cfg, sets, outdir, device):
    import torch
    spec = ARMS[arm]
    s, depth, v, m = cfg["s"], cfg["depth"], cfg["v"], cfg["m"]
    rules, rules_t, canon = shared["rules"], shared["rules_t"], shared["canon"]
    maxl = cfg["max_macro_level"]
    levels = tuple(spec.get("mine_levels", tuple(range(2, maxl + 1))))
    mine_cap = spec.get("mine_cap", cfg["mine_cap"])
    label = f"{mix}__{arm}"

    generator = copy.deepcopy(shared["generator0"]).to(device)
    for p in generator.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(generator.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    value = copy.deepcopy(shared["value0"]).to(device)
    for p in value.parameters():
        p.requires_grad_(True)
    vopt = torch.optim.AdamW(value.parameters(), lr=cfg["value_lr_online"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)
    controller = shared["controller"]

    committed = {ell: None for ell in range(2, maxl + 1)}
    miners = {ell: MC.Miner(ell, s) for ell in range(2, maxl + 1)}
    if spec["vocab"] == "true":
        for ell in range(2, maxl + 1):
            committed[ell] = shared["truth"][ell]
    ms = build_ms(shared["base_ms"], committed, s, depth, device)

    def operative(ell):
        if ell == 1:
            return MC.base_table(v)
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    arch0 = sets["archives"][mix]                       # the FROZEN archive for this mix
    content0 = sets["contents"][mix]
    buf = {"x": torch.zeros(0, shared["length"], dtype=torch.long),
           "r": torch.zeros(0, dtype=torch.long), "y": torch.zeros(0)}
    t_cum = 0.0
    passes, cycles, events = [], [], []
    prev_flat = {ell: set() for ell in range(2, maxl + 1)}
    prev_aud = {ell: None for ell in range(2, maxl + 1)}
    seen_fresh = {ell: set() for ell in range(2, maxl + 1)}     # fresh arm's cumulative content
    cyc = 0

    def solve_and_mine(x_np, r_np, cell_np):
        """Solve a set at PERFORMANCE width with the CURRENT state and mine it with a
        THROWAWAY miner built against this arm's operative lower tables. Never acted on:
        this is the paired novelty probe's one primitive, run on both sides of the
        frozen-vs-fresh contrast so the accounting is identical."""
        w = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
        b = beam_moves(controller, generator, value, torch.from_numpy(x_np),
                       torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                       budget=cfg["budget"], beam_width=w, device=device)
        sc, _ = grade(b["x"].cpu().numpy(), r_np, rules, s)
        sel = sc > 0.5
        mn = {ell: MC.Miner(ell, s) for ell in levels}
        if sel.any():
            mine_into(mn, shared["reader"], b["x"][torch.from_numpy(sel).to(device)],
                      cell_np[sel], cfg, levels)
        out = {"n": int(len(r_np)), "solved": int(sel.sum()), "e": 1.0 - float(sc.mean())}
        for ell in levels:
            tbl = mn[ell].build(operative(ell - 1), cfg["mine_support"])
            fl = flats(tbl)
            out[f"L{ell}"] = {"n_entries": len(fl),
                              "n_correct": len(fl & sets["truth_flat"][ell]),
                              "n_content": len(fl & content0[ell]["set"]),
                              "n_content_true": len(fl & content0[ell]["true_set"])}
        return out

    for p in range(1, cfg["n_passes"] + 1):
        # ---- which archive this pass presents ------------------------------------------
        if spec["archive"] == "frozen":
            arch = arch0
        else:
            # pass 1 uses the SAME seed as the frozen archive, so pass 1 is bit-identical
            # across the frozen/fresh arms -- the fidelity gate G-P1.
            arch = (arch0 if p == 1 else
                    build_archive(rules, cfg, mix, cfg["n_arch"],
                                  cfg["arch_seed"] + 500_000 * (p - 1)))
            cont = (content0 if p == 1 else
                    archive_content(arch, shared["inverse_maps"][-1], cfg,
                                    range(2, maxl + 1)))
            for ell in range(2, maxl + 1):
                seen_fresh[ell] |= cont[ell]["set"]
        counts = {"mat": 0, "ground": 0}
        n_solved_pass, n_mined_pass = 0, 0

        # ---- the pass: `n_arch // chunk` cycles in FIXED order --------------------------
        for c0 in range(0, arch["n"], cfg["chunk"]):
            cyc += 1
            sl = slice(c0, min(c0 + cfg["chunk"], arch["n"]))
            x_np, r_np, k_np = arch["x"][sl], arch["roots"][sl], arch["cell"][sl]
            out = beam_moves(controller, generator, value, torch.from_numpy(x_np),
                             torch.from_numpy(r_np), ms, rules_t, canon, depth, v, m, s,
                             budget=cfg["budget"], beam_width=cfg["pr_width"], device=device,
                             collect=True)
            for key in counts:
                counts[key] += out["counts"][key]
            tips = out["tips_x"]
            B, W, T = tips.shape
            tips_flat = tips.reshape(B * W, T)
            succ, _ = grade(tips_flat.cpu().numpy(), np.repeat(r_np, W), rules, s)
            xs = torch.cat([t.reshape(B * W, T).cpu() for t in out["traj"]])
            rs = torch.from_numpy(np.tile(np.repeat(r_np, W), len(out["traj"])))
            ys = torch.from_numpy(np.tile(succ.astype(np.float32), len(out["traj"])))
            push(buf, xs, rs, ys, cfg["buf_cap"])
            solved = tips_flat[torch.from_numpy(succ > 0.5).to(device)]
            ps, _ = grade(out["x"].cpu().numpy(), r_np, rules, s)
            e_practice = 1.0 - float(ps.mean())

            # mining: the beam's OWN chosen answer on instances it solved (`mine_from=chosen`)
            keep = ps > 0.5
            mine_src = out["x"][torch.from_numpy(keep).to(device)]
            cell_src = k_np[keep]
            if mine_cap and mine_src.shape[0] > mine_cap:
                pick = rng.permutation(mine_src.shape[0])[:mine_cap]
                mine_src = mine_src[torch.from_numpy(pick).to(device)]
                cell_src = cell_src[pick]
            n_mined_pass += mine_into(miners, shared["reader"], mine_src, cell_src, cfg, levels)
            n_solved_pass += int(keep.sum())

            gloss = finetune_generator(generator, gopt, solved.cpu(), shared["replay"]["x"],
                                       shared["bottom_map"], v=v, s=s,
                                       n_blocks=shared["n_blocks"], n_steps=cfg["gen_steps"],
                                       batch=cfg["batch_size"], replay_frac=cfg["replay_frac"],
                                       device=device, rng=grng)
            vloss = value_steps(value, vopt, controller, buf, shared["replay"],
                                n_steps=cfg["n_grad"], batch=cfg["value_batch"],
                                replay_frac=cfg["replay_frac"], device=device, rng=rng)
            cycles.append({"cycle": cyc, "pass": p, "e_practice": e_practice,
                           "n_solved": int(keep.sum()), "gloss": gloss, "vloss": vloss})

        # ---- metering: fixed held-out sets, PER DAMAGE DEPTH ----------------------------
        w = fit_width(len(ms), cfg["budget"], cfg["g_budget"])
        meter = {}
        for name in CELL_ORDER:
            mr, mx = sets["meter"][name]
            b = beam_moves(controller, generator, value, mx, torch.from_numpy(mr), ms,
                           rules_t, canon, depth, v, m, s, budget=cfg["budget"],
                           beam_width=w, device=device)
            for key in counts:
                counts[key] += b["counts"][key]
            sc, _ = grade(b["x"].cpu().numpy(), mr, rules, s)
            meter[name] = 1.0 - float(sc.mean())
        t_cum += priced(counts, cfg)

        # ---- yield accounting on the OPERATIVE vocabulary --------------------------------
        vocab = {}
        for ell in range(2, maxl + 1):
            tbl = operative(ell)
            fl = flats(tbl)
            cell_out = {"n_entries": len(fl),
                        "n_correct": len(fl & sets["truth_flat"][ell]),
                        "n_content": len(fl & content0[ell]["set"]),
                        "n_content_total": len(content0[ell]["set"]),
                        "n_content_true": len(fl & content0[ell]["true_set"]),
                        "n_content_true_total": len(content0[ell]["true_set"]),
                        "n_new": len(fl - prev_flat[ell]),
                        "committed": committed[ell] is not None,
                        "miner": miners[ell].state() if ell in miners else None}
            if spec["archive"] == "fresh":
                cell_out["n_seen_fresh"] = len(seen_fresh[ell])
            # the audition, CONSUMPTION-MATCHED (`ratchet`'s level-3 caveat, fixed): a
            # level-l macro is graded on level-l damage, where it is actually consumed.
            sr, sx = sets["shadow"][ell]
            nd = span_node(CELLS[f"L{ell}n{CELLS_NODE[ell]}"], ell, s)
            if len(fl):
                mv = MC.to_device(MC.make_macro(ell, nd, s, tbl), device)
                a = audit(generator, sx, sr, mv, rules, canon, s)
                cell_out["aud"] = a["e"]
                cell_out["n_used"] = len(a["used"] & fl)
                cell_out["n_new_used"] = len((fl - prev_flat[ell]) & a["used"])
                cell_out["d_aud"] = (None if prev_aud[ell] is None
                                     else prev_aud[ell] - a["e"])
                prev_aud[ell] = a["e"]
                # controls: matched-size random subset of the TRUE table, and of the
                # COMPOSITION CLOSURE of the operative lower vocabulary
                full = shared["truth"][ell]
                n_all = full["child"].shape[0]
                k = min(len(fl), n_all)
                es = []
                for _ in range(cfg["n_rand"]):
                    kp = np.sort(rng.permutation(n_all)[:k])
                    sub = MC.make_table(ell, full["child"][kp], full["lower"], s)
                    mvr = MC.to_device(MC.make_macro(ell, nd, s, sub), device)
                    es.append(audit(generator, sx, sr, mvr, rules, canon, s)["e"])
                cell_out["rand_k"] = float(np.mean(es))
                clo = closure_table(ell, operative(ell - 1), s)
                if clo is not None and clo["child"].shape[0]:
                    mvc = MC.to_device(MC.make_macro(ell, nd, s, clo), device)
                    ac = audit(generator, sx, sr, mvc, rules, canon, s)
                    cell_out["closure"] = ac["e"]
                    cell_out["closure_entries"] = int(clo["child"].shape[0])
                    cell_out["closure_used"] = len(ac["used"])
                    es2 = []
                    nc = clo["child"].shape[0]
                    for _ in range(cfg["n_rand"]):
                        kp = np.sort(rng.permutation(nc)[:min(k, nc)])
                        sub = MC.make_table(ell, clo["child"][kp], clo["lower"], s)
                        mvr = MC.to_device(MC.make_macro(ell, nd, s, sub), device)
                        es2.append(audit(generator, sx, sr, mvr, rules, canon, s)["e"])
                    cell_out["comp_k"] = float(np.mean(es2))
            mvt = MC.to_device(MC.make_macro(ell, nd, s, shared["truth"][ell]), device)
            cell_out["true"] = audit(generator, sx, sr, mvt, rules, canon, s)["e"]
            prev_flat[ell] = fl
            vocab[str(ell)] = cell_out

        # ---- THE PAIRED NOVELTY PROBE (instrument, unpriced, never acted on) -------------
        probe = None
        if cfg["novelty_probe"]:
            probe = {"reread": solve_and_mine(arch0["x"], arch0["roots"], arch0["cell"])}
            fa = build_archive(rules, cfg, mix, cfg["n_arch"],
                               cfg["arch_seed"] + 900_000 + 1000 * p)
            probe["fresh"] = solve_and_mine(fa["x"], fa["roots"], fa["cell"])
            probe["fresh_sha"] = fa["sha"]

        # ---- the commit schedule --------------------------------------------------------
        if spec["commit"]:
            for ell in range(2, maxl + 1):
                if committed[ell] is None and p == cfg["commit_pass"].get(ell, -1):
                    tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                    if tbl["child"].shape[0] == 0:
                        continue
                    committed[ell] = tbl
                    ms = build_ms(shared["base_ms"], committed, s, depth, device)
                    ev = dict(kind="commit", cell=label, level=ell, pass_=p, cycle=cyc,
                              n_entries=int(tbl["child"].shape[0]), n_moves_after=len(ms),
                              **{f"tab_{k}": val for k, val in
                                 MC.grade_table(tbl, shared["truth"][ell]).items()})
                    events.append(ev)
                    print(f"[commit] {label} L{ell} pass{p} entries={ev['n_entries']} "
                          f"recall={ev['tab_recall']:.3f} prec={ev['tab_precision']} "
                          f"moves={len(ms)}", flush=True)

        rec = {"pass": p, "cycle": cyc, "t_cum": t_cum, "meter": meter, "vocab": vocab,
               "probe": probe, "n_moves": len(ms), "width": w, "arch_sha": arch["sha"],
               "n_solved_pass": n_solved_pass, "n_mined_pass": n_mined_pass,
               "plant": plant_probe(generator, shared, cfg, device)}
        passes.append(rec)
        v2, v3 = vocab.get("2", {}), vocab.get("3", {})
        print(f"[{label:22s} p{p}] t={t_cum:9.0f} nm={len(ms):2d} "
              f"e={meter['L1n6']:.3f}/{meter['L2n3']:.3f}/{meter['L3n1']:.3f} "
              f"solved={n_solved_pass:4d} "
              f"L2(e={v2.get('n_entries')},new={v2.get('n_new')},used={v2.get('n_used')},"
              f"aud={v2.get('aud')}) "
              f"L3(e={v3.get('n_entries')},new={v3.get('n_new')},used={v3.get('n_used')},"
              f"aud={v3.get('aud')})"
              + (f" probe rr/fr L3={probe['reread'].get('L3', {}).get('n_entries')}/"
                 f"{probe['fresh'].get('L3', {}).get('n_entries')}" if probe else ""),
              flush=True)
        write_cell(outdir, label, mix, arm, cfg, passes, cycles, events, arch0, content0,
                   complete=(p == cfg["n_passes"]))
    return {"passes": passes, "events": events}


CELLS_NODE = {2: 3, 3: 1}          # the damage cell a level-l macro is CONSUMED on


def write_cell(outdir, label, mix, arm, cfg, passes, cycles, events, arch, content, complete):
    d = os.path.join(outdir, label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"cell": label, "mix": mix, "arm": arm, "config": cfg,
                   "archive": {k: arch[k] for k in ("mix", "n", "seed", "sha")},
                   "content": {str(e): {"n": len(c["set"]), "n_true": len(c["true_set"])}
                               for e, c in content.items()},
                   "passes": passes, "cycles": cycles, "events": events,
                   "complete": complete}, fh, indent=2, cls=NumpyEncoder)
    volume.commit()


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #

def _cfg(**kw):
    cfg = ratchet_cfg()
    cfg.update(
        n_arch=192, chunk=64, n_passes=9, arch_seed=424_242,
        commit_pass={2: 3, 3: 6}, mine_cap=0, mine_support=3, mine_from="chosen",
        n_rt=256, n_score=256, n_rand=3, novelty_probe=True, plant_holdout=0,
    )
    cfg.update(kw)
    return cfg


def _build_sets(shared, cfg, mixes, device):
    """Fixed, shared across every arm and mix: the per-depth metering sets, the
    consumption-matched shadow sets, and one frozen archive per mix."""
    import torch
    rules = shared["rules"]
    s, depth, v, m = cfg["s"], cfg["depth"], cfg["v"], cfg["m"]
    meter, shadow = {}, {}
    for i, name in enumerate(CELL_ORDER):
        r, x = context_instances(rules, era_ctx(CELLS[name]), cfg["n_rt"], s, depth, v, m,
                                 seed=cfg["seed"] + 5000 + 17 * i)
        meter[name] = (r, torch.from_numpy(x))
    for ell in range(2, cfg["max_macro_level"] + 1):
        name = f"L{ell}n{CELLS_NODE[ell]}"
        r, x = context_instances(rules, era_ctx(CELLS[name]), cfg["n_score"], s, depth, v, m,
                                 seed=cfg["seed"] + 6100 + 23 * ell)
        shadow[ell] = (r, torch.from_numpy(x).to(device))
    archives, contents = {}, {}
    for mix in mixes:
        archives[mix] = build_archive(rules, cfg, mix, cfg["n_arch"], cfg["arch_seed"])
        contents[mix] = archive_content(archives[mix], shared["inverse_maps"][-1], cfg,
                                        range(2, cfg["max_macro_level"] + 1))
    truth_flat = {ell: flats(shared["truth"][ell])
                  for ell in range(2, cfg["max_macro_level"] + 1)}
    # the honest denominator: `build_inverse_maps` is last-writer-wins over bottom-level
    # synonyms (`ratchet`'s calp2_s0 — an irreducible SUBSTRATE read error, not the agent's),
    # so the exact parse of a clean derivation can name a tuple the grammar cannot produce.
    # `true_set` is the archive's GRAMMATICAL content; `set` is what any reader can see.
    for c in contents.values():
        for ell, cell in c.items():
            cell["true_set"] = cell["set"] & truth_flat[ell]
    return {"meter": meter, "shadow": shadow, "archives": archives, "contents": contents,
            "truth_flat": truth_flat}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=32768)
def reread(
    tag: str = "smoke",
    mixes: str = "mixed,shallow,deep",
    arms: str = "reread,fresh,dense_reread,dense_fresh,given_reread",
    fidelity: bool = True,
    seed: int = 0, n_arch: int = 192, chunk: int = 64, n_passes: int = 9,
    commit_l2: int = 3, commit_l3: int = 6, arch_seed: int = 424_242,
    budget: int = 4, pr_width: int = 16, g_budget: int = 58, max_macro_level: int = 3,
    n_rt: int = 256, n_score: int = 256, n_grad: int = 4, gen_lr: float = 1e-4,
    gen_steps: int = 20, mine_support: int = 3, mine_cap: int = 0,
    value_lr_online: float = 3e-5, novelty_probe: bool = True,
    d_fb: float = 1.0, c_mat: float = 0.05, quick: bool = False,
):
    import torch
    cfg = _cfg(seed=seed, n_arch=n_arch, chunk=chunk, n_passes=n_passes,
               commit_pass={2: commit_l2, 3: commit_l3}, arch_seed=arch_seed,
               budget=budget, pr_width=pr_width, g_budget=g_budget,
               max_macro_level=max_macro_level, n_rt=n_rt, n_score=n_score, n_grad=n_grad,
               gen_lr=gen_lr, gen_steps=gen_steps, mine_support=mine_support,
               mine_cap=mine_cap, value_lr_online=value_lr_online,
               novelty_probe=novelty_probe, d_fb=d_fb, c_mat=c_mat)
    if quick:
        cfg.update(controller_steps=800, generator_steps=800, value_steps=800,
                   reader_steps=600, value_episodes=6_000, n_train_episodes=20_000,
                   n_arch=48, chunk=24, n_passes=3, commit_pass={2: 1, 3: 2},
                   n_rt=96, n_score=96, n_probe_clean=128, gen_steps=5)
    mix_list = [x.strip() for x in mixes.split(",") if x.strip()]
    arm_list = parse_arms(arms)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train_seed"]); np.random.seed(cfg["train_seed"])
    torch.set_float32_matmul_precision("high")
    started = time.time()
    print(f"reread tag={tag} mixes={mix_list} arms={arm_list} device={device}")

    shared = build_shared(cfg, device)
    shared["probe_clean"] = torch.from_numpy(
        _sample_pool(shared["rules"], cfg["n_probe_clean"], cfg["s"], cfg["seed"] + 31337)[1])
    sets = _build_sets(shared, cfg, sorted(set(mix_list + (["shallow"] if fidelity else []))),
                       device)
    for ell in range(2, cfg["max_macro_level"] + 1):
        print(f"  true table L{ell}: {len(sets['truth_flat'][ell])} distinct flat tuples")
    for mix in sets["archives"]:
        a, c = sets["archives"][mix], sets["contents"][mix]
        print(f"  archive[{mix}] n={a['n']} sha={a['sha']} content: "
              + " ".join(f"L{e}={len(c[e]['set'])}({len(c[e]['true_set'])} true"
                         f"/{len(sets['truth_flat'][e])} grammar)" for e in c))

    outdir = f"{DATA_DIR}/{REMOTE}/{tag}"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "mixes": mix_list, "arms": arm_list,
                   "archives": {k: {kk: a[kk] for kk in ("mix", "n", "seed", "sha")}
                                for k, a in sets["archives"].items()},
                   "content": {k: {str(e): {"n": len(c[e]["set"]),
                                            "n_true": len(c[e]["true_set"])} for e in c}
                               for k, c in sets["contents"].items()},
                   "truth_sizes": {str(e): len(f) for e, f in sets["truth_flat"].items()}},
                  fh, indent=2, cls=NumpyEncoder)
    volume.commit()

    # ---- THE FIDELITY GATE FIRST: `ratchet`'s era-1 configuration on this loop -----------
    if fidelity:
        print("\n===== FIDELITY GATE: fid_ratchet on the shallow archive "
              "(ratchet era-1 reference: e~0.349, L2 table recall 0.500 / prec 0.875) =====",
              flush=True)
        run_cell("shallow", "fid_ratchet", shared, cfg, sets, outdir, device)

    for mix in mix_list:
        for arm in arm_list:
            print(f"\n===== cell {mix} / {arm} =====", flush=True)
            run_cell(mix, arm, shared, cfg, sets, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - started:.0f}s\n")
    volume.commit()
    print(f"\nDONE in {time.time() - started:.0f}s -> {outdir}")
    return {"elapsed": time.time() - started}


@app.function(image=image, timeout=3600, memory=16384)
def gate(n_arch: int = 192):
    """Structural gates, no training. C-M / C-R / G-D are `ratchet`'s own (imported, so a
    drift in the donor breaks here first). C-F is this node's: the archive is BYTE-IDENTICAL
    on every presentation. C-C: the archive's ground-truth content is a strict subset of the
    grammar's, and its level-3 content is the coordinate the depth sweep moves."""
    from rhm.rhm_data import build_inverse_maps, generate_rules_distinct
    out = {"ratchet": ratchet_selfcheck()}
    cfg = _cfg(n_arch=n_arch)
    rules = generate_rules_distinct(cfg["v"], cfg["s"], cfg["depth"], cfg["m"],
                                    seed=cfg["rule_seed"])
    inv = build_inverse_maps(rules)
    truth = MC.true_tables(rules, cfg["depth"], cfg["s"], cfg["v"], cfg["m"],
                           cfg["max_macro_level"])
    tf = {ell: flats(truth[ell]) for ell in (2, 3)}
    cf, cc = {}, {}
    for mx in MIXES:
        a = build_archive(rules, cfg, mx, n_arch, cfg["arch_seed"])
        a2 = build_archive(rules, cfg, mx, n_arch, cfg["arch_seed"])
        cf[mx] = {"sha": a["sha"], "redraw_identical": bool(a["sha"] == a2["sha"]),
                  "cells": {CELL_ORDER[i]: int((a["cell"] == i).sum()) for i in range(3)}}
        assert cf[mx]["redraw_identical"], f"C-F failed at {mx}"
        c = archive_content(a, inv[-1], cfg, (2, 3))
        cc[mx] = {f"L{e}": {"n_content": len(c[e]["set"]), "n_true": len(tf[e]),
                            "subset_of_true": bool(c[e]["set"] <= tf[e]),
                            "frac_of_true": len(c[e]["set"] & tf[e]) / len(tf[e])}
                  for e in (2, 3)}
    out["CF_archive_frozen"] = cf
    out["CC_archive_content"] = cc
    print(json.dumps(out, indent=2, cls=NumpyEncoder))
    return out


@app.local_entrypoint()
def main(quick: bool = True):
    reread.remote(quick=quick, tag="smoke_local")
