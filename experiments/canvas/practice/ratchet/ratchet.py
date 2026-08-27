"""ratchet on canvas -- earning a level-indexed code-tuple vocabulary on ALIGNED TILES.

FORK NOTICE. The loop shape is `rhm/practice/ratchet/ratchet.py`'s (donor untouched): eras of
nested damage, per cycle practice -> mining -> shadow audition -> certificate -> commit ->
metering, macros frozen at commit and instantiated at every node of their level, five arms that
differ ONLY in vocabulary-acquisition policy. `macros.py` is forked beside it with s = 4 in a
2x2 spatial arrangement. What is NOT the donor's, and why:

  the piece        INPAINTING (`tiles.damage` mode="mask"), nested regions: era k's hole is a
                   level-k tile region == a 2^k x 2^k CODE square, and era k+1's hole contains
                   it. That is what the plant was trained on and what `sampler.decode` prices;
                   `seam` repair would add a detection sub-problem on top of the vocabulary
                   question. Era k's hole is EXACTLY one level-(k+1) macro block, so the
                   audition is consumption-matched with no depth seam (the donor had one).
  the grade        the ADJACENCY-SUPPORT test (`plant/tiles_twin/adjacency.py`): a fill passes
                   iff every oriented adjacent code pair it touches -- inside the hole and
                   across the hole boundary -- occurred in genuine exemplars. Zero forward
                   passes, so it is also a DENSE signal: every placed pair is checkable
                   mid-fill. There is no learned value head at all (see `solve`).
  the price        FORWARD PASSES. One forward gives every cell's log-probs at once, so all
                   candidate moves from one beam state share one forward and a level-l macro
                   costs about one grounding. Cost = (steps actually taken) x (beam width),
                   exactly `sampler.decode`'s own accounting.
  the parse        BLOCKIFY. The alphabet is the parse, so mining reads the agent's own solved
                   fills directly -- no reader to train (the donor needed one).

Everything oracle-side (`tiles.Tileset.validate`, the per-swatch tile ids, the tile atlas) is
written beside the numbers and never read on an agent path, except by the `given` arm, which is
the ceiling and is labelled as such.

Run (from experiments/, MODAL_PROFILE=chromatic):
    modal run -m canvas.practice.ratchet.ratchet::selfcheck
    modal run -m canvas.practice.ratchet.ratchet::run --tag smoke0 --quick 1
    python3 canvas/practice/ratchet/launch_detached.py --tag cr_s0 --logdir /tmp/crlogs
"""

import json
import math
import os
import time

import numpy as np

from canvas.shared import DATA_DIR, NumpyEncoder, app, volume
from canvas.practice.ratchet import macros as MC

REMOTE = "canvas_practice_ratchet"
ROOT = os.environ.get("CANVAS_DATA", DATA_DIR)
DEV = os.environ.get("CANVAS_DEV", "cuda")

GRID = 16
K = 512
MAXSPAN = 16
LEVELS = (1, 2, 3)
SPLIT_SUPPORT = ("gA_train", "gB_train")      # pooled per tileset -- coverage before anything
SPLIT_PRACTICE = "plant_train"
SPLIT_METER = "plant_test"
SPLIT_AUDIT = "plant_val"


def _commit():
    try:
        volume.commit()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# the grade of record: adjacency support (adjacency.py, made incremental)
# --------------------------------------------------------------------------- #

def support_matrices(codes, k=K, grid=GRID):
    """Dense oriented-pair support, from GENUINE EXEMPLARS ONLY. supH[a, b] is the pair
    ((r,c) -> (r,c+1)); supV[a, b] is ((r,c) -> (r+1,c)). min_count = 1 (thresholds buy
    specificity by destroying coverage -- `plant/README.md` finding 12)."""
    g = np.asarray(codes).reshape(-1, grid, grid)
    H = np.zeros((k, k), bool)
    V = np.zeros((k, k), bool)
    H[g[:, :, :-1].reshape(-1), g[:, :, 1:].reshape(-1)] = True
    V[g[:, :-1, :].reshape(-1), g[:, 1:, :].reshape(-1)] = True
    return H, V


def grade_support(grids, masks, H, V, grid=GRID):
    """The verdict, per item: pass iff every TOUCHED pair is in support. A pair is touched if
    either endpoint is masked (== was filled by the agent). Identical to
    `adjacency.evaluate` restricted to one support variant."""
    g = np.asarray(grids).reshape(-1, grid, grid)
    m = np.asarray(masks).reshape(-1, grid, grid)
    n = g.shape[0]
    # an UNFILLED cell (still the MASK token) can never be in support: fail the item outright
    unfilled = (g >= H.shape[0]).reshape(n, -1).any(1)
    g = np.minimum(g, H.shape[0] - 1)
    th = m[:, :, :-1] | m[:, :, 1:]
    tv = m[:, :-1, :] | m[:, 1:, :]
    bh = th & ~H[g[:, :, :-1], g[:, :, 1:]]
    bv = tv & ~V[g[:, :-1, :], g[:, 1:, :]]
    n_unsup = bh.reshape(n, -1).sum(1) + bv.reshape(n, -1).sum(1)
    n_touch = th.reshape(n, -1).sum(1) + tv.reshape(n, -1).sum(1)
    ch = m[:, :, :-1] ^ m[:, :, 1:]
    cv = m[:, :-1, :] ^ m[:, 1:, :]
    n_cross_bad = (bh & ch).reshape(n, -1).sum(1) + (bv & cv).reshape(n, -1).sum(1)
    n_unsup = np.where(unfilled, np.maximum(n_unsup, 1), n_unsup)
    return {"n_unsup": n_unsup, "n_touched": n_touch, "pass": n_unsup == 0,
            "n_unsup_cross": n_cross_bad, "unfilled": unfilled}


# --------------------------------------------------------------------------- #
# instances: nested inpainting regions on the tile grid
# --------------------------------------------------------------------------- #

def nested_regions(rng):
    """One nested chain of TILE regions on the 8x8 tile window: a level-3 (4x4 tiles) region,
    a level-2 (2x2) inside it, a level-1 (1x1) inside that. `tiles.level_regions`' cells, so
    era k+1's hole literally contains era k's on the same swatch."""
    r3, c3 = int(rng.integers(2)) * 4, int(rng.integers(2)) * 4
    r2, c2 = r3 + int(rng.integers(2)) * 2, c3 + int(rng.integers(2)) * 2
    r1, c1 = r2 + int(rng.integers(2)), c2 + int(rng.integers(2))
    return {1: (r1, c1, 1), 2: (r2, c2, 2), 3: (r3, c3, 4)}


def region_mask(region, grid=GRID):
    """A tile region (r0, c0, side) -> the (grid*grid,) code mask. Aligned: exactly the
    2*side x 2*side code square at (2*r0, 2*c0)."""
    r0, c0, side = region
    m = np.zeros((grid, grid), bool)
    m[2 * r0:2 * (r0 + side), 2 * c0:2 * (c0 + side)] = True
    return m.reshape(-1)


def region_node(region, level):
    """The level-`level` macro node whose block is exactly this region's code square.
    Era k's hole is the level-(k+1) block; used for the audition."""
    r0, c0, side = region
    n_side = GRID // (2 * side)
    return (r0 // side) * n_side + (c0 // side)


def make_instances(rows, styles_of, n, rng):
    """n instances: a swatch row, its style, and its nested region chain."""
    pick = rng.integers(0, len(rows), size=n)
    idx = np.asarray(rows)[pick]
    regs = [nested_regions(rng) for _ in range(n)]
    return {"row": idx, "style": np.asarray(styles_of)[pick], "regions": regs}


def era_masks(inst, level):
    return np.stack([region_mask(r[level]) for r in inst["regions"]])


def era_nodes(inst, level):
    """The macro node of the level-(level+1) block covering each instance's hole."""
    return np.array([region_node(r[level], level + 1) for r in inst["regions"]], np.int64)


# --------------------------------------------------------------------------- #
# the action space: a move commits one block to one table entry
# --------------------------------------------------------------------------- #

def sup_tensors(supH, supV, device):
    import torch
    H = torch.from_numpy(np.ascontiguousarray(supH)).to(device)
    V = torch.from_numpy(np.ascontiguousarray(supV)).to(device)
    Ht = H.t().contiguous(); Vt = V.t().contiguous()
    # indexed by (orient, block_is_first): the matrix whose ROW is the neighbour's code
    return {"H": H, "Ht": Ht, "V": V, "Vt": Vt, "dirs": [H, Ht, V, Vt]}


def prep_table(tbl, level, supH, supV, device):
    """Everything the operator needs on device: the entries' flattened codes, their OWN
    internal unsupported-pair count (a property of the table and the support alone, so it is
    computed once), every node's cells and its boundary slots."""
    import torch
    n = int(tbl["child"].shape[0]) if tbl is not None else 0
    nn = MC.n_nodes(level)
    d = {"level": level, "n": n, "span": 4 ** (level - 1), "nn": nn, "table": tbl}
    d["cells"] = torch.from_numpy(
        np.stack([MC.zcells(level, j) for j in range(nn)])).to(device)
    sp, out, orient, first = zip(*[MC.boundary_slots(level, j) for j in range(nn)])
    d["b_sp"] = torch.from_numpy(sp[0]).to(device)
    d["b_or"] = orient[0]
    d["b_first"] = first[0]
    o = np.stack(out)
    d["b_out"] = torch.from_numpy(np.maximum(o, 0)).to(device)
    d["b_ok"] = torch.from_numpy(o >= 0).to(device)
    if n:
        d["flat"] = torch.from_numpy(np.ascontiguousarray(tbl["flat"])).to(device)
        d["int_unsup"] = torch.from_numpy(
            MC.internal_unsup(tbl, supH, supV).astype(np.float32)).to(device)
    return d


def _l1_penalty(X, M, sup, nb, kk):
    """(B, T, K) count of adjacent pairs a code would leave OUT OF SUPPORT if written at that
    cell, over the neighbours that are already determined. The dense form of the grade."""
    import torch
    B, T = X.shape
    pen = None
    for d in range(4):
        idx = nb[:, d]
        ok_static = idx >= 0
        j = idx.clamp(min=0)
        nbv = X[:, j].clamp(max=kk - 1)
        det = (~M[:, j]) & ok_static[None, :]
        rows = sup["dirs"][d][nbv.reshape(-1)].reshape(B, T, kk)
        bad = (~rows) & det[:, :, None]
        pen = bad.to(torch.float16) if pen is None else pen + bad.to(torch.float16)
    return pen


def _macro_cands(lp, maxlp, X, M, pt, sup, lam, n_alt):
    """Best `n_alt` entries per node of one macro level, from the SAME forward. Returns
    (score, entry, applicable, lpsum, pen, hcut)."""
    import torch
    B, T, kk = lp.shape
    nn, span = pt["cells"].shape
    ne = pt["n"]
    cells, flat = pt["cells"], pt["flat"]
    lpsum = torch.zeros(B, nn, ne, device=lp.device, dtype=torch.float64)
    for t in range(span):
        lpsum += lp[:, cells[:, t], :].gather(
            2, flat[:, t][None, None, :].expand(B, nn, ne)).double()
    pen = pt["int_unsup"].double()[None, None, :].expand(B, nn, ne).clone()
    b_out, b_ok, b_sp = pt["b_out"], pt["b_ok"], pt["b_sp"]
    for s in range(b_sp.shape[0]):
        j = b_out[:, s]
        nbv = X[:, j].clamp(max=kk - 1)
        det = (~M[:, j]) & b_ok[None, :, s]
        if not bool(det.any()):
            continue
        mat = sup["H"] if pt["b_or"][s] == 0 else sup["V"]
        if pt["b_first"][s]:
            mat = mat.t()
        rows = mat[nbv.reshape(-1)].reshape(B, nn, kk)
        ev = flat[:, int(b_sp[s])]
        pen += ((~rows[:, :, ev]) & det[:, :, None]).to(pen.dtype)
    hcut = maxlp[:, cells].sum(-1)
    app = M[:, cells.reshape(-1)].reshape(B, nn, span).all(-1)
    sc = lpsum - lam * pen
    k = min(n_alt, ne)
    val, ent = sc.topk(k, dim=-1)
    if k < n_alt:
        val = torch.cat([val] + [torch.full_like(val[:, :, :1], -1e9)] * (n_alt - k), -1)
        ent = torch.cat([ent] + [ent[:, :, :1]] * (n_alt - k), -1)
    return val, ent, app, lpsum.gather(2, ent), pen.gather(2, ent), hcut


def _unsup_count(X, M, M0, sup, kk, grid=GRID):
    """Unsupported adjacent code pairs among the pairs the fill TOUCHES (at least one endpoint
    in the original hole) and that are now DETERMINED (both endpoints written). Exact, and
    recomputed from the grid each step, so simultaneous reveals inside one step are counted."""
    N, W, T = X.shape
    x = X.clamp(max=kk - 1).view(N, W, grid, grid)
    d = (~M).view(N, W, grid, grid)
    m0 = M0.view(N, 1, grid, grid)
    th = (m0[..., :, :-1] | m0[..., :, 1:]) & d[..., :, :-1] & d[..., :, 1:]
    tv = (m0[..., :-1, :] | m0[..., 1:, :]) & d[..., :-1, :] & d[..., 1:, :]
    bh = th & ~sup["H"][x[..., :, :-1], x[..., :, 1:]]
    bv = tv & ~sup["V"][x[..., :-1, :], x[..., 1:, :]]
    return (bh.reshape(N, W, -1).sum(-1) + bv.reshape(N, W, -1).sum(-1)).double()


def _apply_cand(Xn, Mn, q, gv, ok, offs, T, n_alt, levels, tabs):
    """Write the candidate `q` (per (row, hypothesis)) into the child state. Returns the number
    of cells each write covered."""
    import torch
    N, W, _ = Xn.shape
    rows = torch.arange(N, device=Xn.device)[:, None].expand(-1, W)
    hyp = torch.arange(W, device=Xn.device)[None, :].expand(N, -1)
    n_written = torch.zeros(N, W, dtype=torch.long, device=Xn.device)
    m1 = (q < T * n_alt) & ok
    if bool(m1.any()):
        cell = torch.div(q, n_alt, rounding_mode="floor")
        Xn[rows[m1], hyp[m1], cell[m1]] = gv[m1]
        Mn[rows[m1], hyp[m1], cell[m1]] = False
        n_written = n_written + m1.long()
    for l in levels:
        pt = tabs[l]
        lo = offs["m%d" % l]
        mm = (q >= lo) & (q < lo + pt["nn"] * n_alt) & ok
        if not bool(mm.any()):
            continue
        node = torch.div(q - lo, n_alt, rounding_mode="floor")
        rr, hh = rows[mm], hyp[mm]
        cc = pt["cells"][node[mm]]
        Xn[rr[:, None], hh[:, None], cc] = pt["flat"][gv[mm]]
        Mn[rr[:, None], hh[:, None], cc] = False
        n_written = n_written + mm.long() * pt["span"]
    return n_written


def _applicable(Mn, offs, T, n_alt, levels, tabs, C1):
    """(N, W, C1): a move is applicable iff EVERY cell of its block is still masked."""
    import torch
    N, W, _ = Mn.shape
    parts = [Mn.repeat_interleave(n_alt, dim=2)]
    for l in levels:
        pt = tabs[l]
        parts.append(Mn[:, :, pt["cells"].reshape(-1)].reshape(N, W, pt["nn"], pt["span"])
                     .all(-1).repeat_interleave(n_alt, dim=2))
    parts.append(torch.zeros(N, W, 1, dtype=torch.bool, device=Mn.device))
    return torch.cat(parts, 2)


def solve(plant, grids, masks, styles, tabs, sup, nb, *, budget, w_max, gamma, lam, n_alt,
          device, force_width=None, force_steps=None, ichunk=48, use_style=True):
    """The closed-loop beam. ONE forward per hypothesis per step gives every cell's log-probs,
    and every candidate move is arithmetic over that one forward -- which is why a level-l
    macro costs about one grounding and why the price is FORWARD PASSES.

    THE BUDGET BUYS COORDINATION. `sampler.decode`'s reveal schedule is kept exactly: with a
    declared budget of G forwards per solve the arm runs `steps = G / width` forwards and must
    reveal ceil(left / steps_left) cells at each one, so a tight budget forces many cells to be
    committed from a SINGLE forward. For base moves those commitments are independent per-cell
    argmaxes; a level-l macro commits 4^(l-1) cells as ONE TABLE ENTRY, which is a joint the
    plant was never asked for. That is the whole cost-to-depth contrast, and it is why the
    first smoke (one cell per step, so the base arm always bought full conditioning whatever
    the budget) had no headroom in it.

    Beam score of a partial fill:
        LP            sum of log-probs of the cells written so far
      + gamma * h     h = sum over still-masked cells of the best available log-prob: an
                      optimistic completion bound, and what makes moves of DIFFERENT SIZES
                      comparable -- without it a 4-cell macro always looks worse than a 1-cell
                      move because it spends more log-probability.
      - lam * U       U = adjacent code pairs left OUT OF SUPPORT so far, recomputed exactly
                      from the grid: the grade of record as a dense pruning term, not merely a
                      terminal verdict.
    gamma < 1 keeps, among moves of the same size, the confidence order `decode` uses, so with
    the vocabulary off and lam = 0 this reproduces `decode` bit-for-bit at ANY (steps, width=1)
    (gate F).

    There is NO learned value head. On RHM the value was the only thing practice could move;
    here the support test is free and dense, so the selector is fixed and the entire descent is
    carried by the ACTION SPACE. `never_base` therefore measures whether search + support alone
    afford depth at the declared budget -- a stronger control than the donor's.

    Width is `fit_width` transposed: the widest beam the arm's own action set affords, where an
    arm holding a level-l macro needs only ceil(|M| / 4^(l-1)) moves to cover the hole. Every
    arm then spends the same G forwards, split differently between coordination and hypotheses.
    """
    import torch
    from canvas.plant.model import amp
    with torch.no_grad():
        plant.eval()
        N, T = np.asarray(grids).shape
        kk = plant.K
        nmask = int(np.asarray(masks)[0].sum())
        levels = sorted([l for l in tabs if tabs[l] is not None and tabs[l]["n"] > 0])
        max_span = 4 ** (max(levels + [1]) - 1)
        min_steps = int(math.ceil(nmask / max_span))
        # MATCHED PRICING. In the donor `fit_width` was arm-dependent because every candidate
        # move had to be MATERIALISED and scored, so a bigger action set cost more per step.
        # Here one forward prices the whole action set, so the same budget affords every arm
        # the same beam -- and the calibration ladder (`cal0`) showed that giving the
        # vocabulary arms a wider/shallower split instead, as the donor's rule would, simply
        # takes conditioning away from them and confounds the comparison. Same width, same
        # steps, same forwards for every arm; the action set is the only difference.
        width = int(force_width or max(1, min(w_max, budget)))
        steps = int(force_steps or max(1, budget // width))
        steps = min(steps, nmask)
        seg = [("l1", T)] + [("m%d" % l, tabs[l]["nn"]) for l in levels]
        offs, o = {}, 0
        for nm, cnt in seg:
            offs[nm] = o
            o += cnt * n_alt
        offs["null"] = o
        C1 = o + 1

        lvl_of = torch.cat(
            [torch.ones(T * n_alt, dtype=torch.long, device=device)]
            + [torch.full((tabs[l]["nn"] * n_alt,), l, dtype=torch.long, device=device)
               for l in levels]
            + [torch.full((1,), 99, dtype=torch.long, device=device)])
        restr = [1] + levels                        # one PLAN per move-level restriction
        NP = len(restr)

        X0 = torch.as_tensor(np.asarray(grids), dtype=torch.long, device=device)
        M0 = torch.as_tensor(np.asarray(masks), device=device)
        S = torch.as_tensor(np.asarray(styles), dtype=torch.long, device=device)
        if not use_style:
            S = torch.full_like(S, plant.n_styles)
        X = torch.where(M0, torch.full_like(X0, kk), X0)[:, None].repeat(1, width, 1).contiguous()
        M = M0[:, None].repeat(1, width, 1).contiguous()
        LP = torch.zeros(N, width, device=device, dtype=torch.float64)
        LP[:, 1:] = -1e9
        U = torch.zeros(N, width, device=device, dtype=torch.float64)
        n_moves0 = None
        left = nmask
        used = 0
        rows = torch.arange(N, device=device)
        for r in range(steps):
            if not bool(M.any()):
                break
            k = int(math.ceil(left / (steps - r)))
            used += 1
            sc = torch.full((N, width, C1), -1e9, device=device, dtype=torch.float64)
            lpv = torch.zeros(N, width, C1, device=device, dtype=torch.float64)
            val = torch.zeros(N, width, C1, dtype=torch.long, device=device)
            mxl = torch.zeros(N, width, T, device=device, dtype=torch.float64)
            for a in range(0, N, ichunk):
                b = min(a + ichunk, N)
                nb_ = b - a
                xf = X[a:b].reshape(nb_ * width, T)
                mf = M[a:b].reshape(nb_ * width, T)
                with amp(device):
                    logits = plant(xf, S[a:b].repeat_interleave(width))
                lp = torch.log_softmax(logits.float(), -1)
                maxlp = lp.max(-1).values.double()
                mxl[a:b] = maxlp.reshape(nb_, width, T)
                hrem = (maxlp * mf).sum(-1)
                base = LP[a:b].reshape(-1) + gamma * hrem - lam * U[a:b].reshape(-1)
                if lam:
                    pen1 = _l1_penalty(xf, mf, sup, nb, kk).float()
                    s_all = lp - lam * pen1
                else:
                    pen1 = None
                    s_all = lp
                # alt 0 by `max`, not `topk`: `decode` picks the code by argmax and topk's tie
                # order is not argmax's. Gate F is bit-for-bit.
                c1 = s_all.max(-1, keepdim=True).indices
                if n_alt > 1:
                    c1 = torch.cat([c1, s_all.topk(n_alt, dim=-1).indices[..., 1:]], -1)
                p1 = (pen1.gather(2, c1).double() if pen1 is not None
                      else torch.zeros(lp.shape[0], T, n_alt, device=lp.device,
                                       dtype=torch.float64))
                del pen1, s_all
                lp1 = lp.gather(2, c1).double()
                sc1 = base[:, None, None] + lp1 - gamma * maxlp[:, :, None] - lam * p1
                sc1 = sc1.masked_fill(~mf[:, :, None], -1e9)
                sl = slice(offs["l1"], offs["l1"] + T * n_alt)
                sc[a:b, :, sl] = sc1.reshape(nb_, width, -1)
                lpv[a:b, :, sl] = lp1.reshape(nb_, width, -1)
                val[a:b, :, sl] = c1.reshape(nb_, width, -1)
                for l in levels:
                    pt = tabs[l]
                    _, ent, app, lps, pns, hcut = _macro_cands(
                        lp, maxlp, xf, mf, pt, sup, lam, n_alt)
                    scm = base[:, None, None] + lps - gamma * hcut[:, :, None] - lam * pns
                    scm = scm.masked_fill(~app[:, :, None], -1e9)
                    sl = slice(offs["m%d" % l], offs["m%d" % l] + pt["nn"] * n_alt)
                    sc[a:b, :, sl] = scm.reshape(nb_, width, -1)
                    lpv[a:b, :, sl] = lps.reshape(nb_, width, -1)
                    val[a:b, :, sl] = ent.reshape(nb_, width, -1)
                done = ~mf.any(1)
                sc[a:b, :, offs["null"]] = torch.where(
                    done, base, torch.full_like(base, -1e9)).reshape(nb_, width)
                if n_moves0 is None:
                    n_moves0 = int((sc[0, 0, :offs["null"]] > -1e8).sum()) // n_alt
                del lp, maxlp
            # ---- ONE PLAN PER MOVE-LEVEL RESTRICTION ----------------------------------- #
            #   A plan is a whole step's commitment: a first move plus the greedy top-up that
            #   reveals this step's k cells, all from the SAME forward. Plans are compared
            #   only AFTER the top-up, with U recomputed exactly from the resulting grid --
            #   which is the only way the beam can see the thing a macro is for. A per-move
            #   comparison cannot: the per-cell argmax has zero regret against the greedy
            #   bound BY CONSTRUCTION, and a macro pays its regret up front while its payoff
            #   (a jointly legal block, hence no unsupported pairs among the cells this step
            #   commits together) only exists once the whole step is applied.
            flat_sc = sc.reshape(N, width * C1)
            top_any = flat_sc.topk(width, dim=1).indices
            pars, qs = [], []
            for ell in restr:
                if ell == 1:
                    t = top_any
                else:
                    allow = (lvl_of >= ell)[None, None, :].expand(N, width, C1)
                    t = sc.masked_fill(~allow, -1e9).reshape(N, width * C1) \
                          .topk(width, dim=1).indices
                    ok = torch.gather(flat_sc, 1, t) > -1e8
                    t = torch.where(ok, t, top_any)
                pars.append(torch.div(t, C1, rounding_mode="floor"))
                qs.append(t % C1)
            par = torch.cat(pars, 1)                          # (N, NP*width)
            q = torch.cat(qs, 1)
            W2 = NP * width
            lvl_plan = torch.cat([torch.full((N, width), ell, device=device, dtype=torch.long)
                                  for ell in restr], 1)
            pe_t = par[:, :, None].expand(-1, -1, T)
            Xn = torch.gather(X, 1, pe_t).clone()
            Mn = torch.gather(M, 1, pe_t).clone()
            mxp = torch.gather(mxl, 1, pe_t)
            pe_c = par[:, :, None].expand(-1, -1, C1)
            scc = torch.gather(sc, 1, pe_c)
            lpc = torch.gather(lpv, 1, pe_c)
            valc = torch.gather(val, 1, pe_c)
            LPp = torch.gather(LP, 1, par)
            alive = torch.gather(flat_sc, 1, torch.cat(
                [p * C1 + qq for p, qq in zip(pars, qs)], 1)) > -1e8
            gv = torch.gather(valc, 2, q[:, :, None])[:, :, 0]
            act = alive & (q != offs["null"])
            nfill = _apply_cand(Xn, Mn, q, gv, act, offs, T, n_alt, levels, tabs)
            LPp = LPp + torch.gather(lpc, 2, q[:, :, None])[:, :, 0] * act.double()
            # ---- the reveal schedule: keep committing from the SAME forward until this step
            #      has revealed k cells. With base moves and lam = 0 this reproduces
            #      `decode`'s simultaneous top-k reveal exactly. ------------------------- #
            keep_lvl = (lvl_of[None, None, :] >= lvl_plan[:, :, None])
            for _ in range(int(k)):
                need = (nfill < k) & Mn.any(2)
                if not bool(need.any()):
                    break
                appl = _applicable(Mn, offs, T, n_alt, levels, tabs, C1)
                s_any = scc.masked_fill(~appl, -1e9)
                s_res = scc.masked_fill(~(appl & keep_lvl), -1e9)
                q2 = s_res.argmax(2)
                ok_r = torch.gather(s_res, 2, q2[:, :, None])[:, :, 0] > -1e8
                q_a = s_any.argmax(2)
                ok_a = torch.gather(s_any, 2, q_a[:, :, None])[:, :, 0] > -1e8
                q2 = torch.where(ok_r, q2, q_a)               # fall back when nothing fits
                ok2 = need & (ok_r | ok_a)
                if not bool(ok2.any()):
                    break
                gv2 = torch.gather(valc, 2, q2[:, :, None])[:, :, 0]
                nfill = nfill + _apply_cand(Xn, Mn, q2, gv2, ok2, offs, T, n_alt, levels, tabs)
                LPp = LPp + torch.gather(lpc, 2, q2[:, :, None])[:, :, 0] * ok2.double()
            Up = _unsup_count(Xn, Mn, M0, sup, kk)
            hp = (mxp * Mn).sum(-1)
            plan_sc = (LPp + gamma * hp - lam * Up).masked_fill(~alive, -1e9)
            sel = plan_sc.topk(min(width, W2), dim=1).indices
            X = torch.gather(Xn, 1, sel[:, :, None].expand(-1, -1, T)).contiguous()
            M = torch.gather(Mn, 1, sel[:, :, None].expand(-1, -1, T)).contiguous()
            LP = torch.gather(LPp, 1, sel)
            U = torch.gather(Up, 1, sel)
            left = max(0, left - k)
        best = (LP - lam * U).argmax(1)
        cost = float(used * width)
        return {"x": X[rows, best].cpu().numpy(),
                "cost": np.full(N, cost, np.float64),
                "width": width, "steps_dec": steps, "steps": used,
                "n_moves": int(n_moves0 or 0),
                "incomplete": M[rows, best].any(1).cpu().numpy()}


# --------------------------------------------------------------------------- #
# setup: corpus, support, plant, the given (oracle) tables
# --------------------------------------------------------------------------- #

def load_corpus(cfg):
    """Codes + splits from the quantizer index, and the ORACLE tile ids per swatch (held by
    the experimenter; only the `given` arm's table is allowed to read them)."""
    qdir = os.path.join(ROOT, "plant", cfg["qtag"])
    idx = json.load(open(os.path.join(qdir, "index.json")))
    styles = idx["styles"]
    sidx = np.array(idx["sidx"], np.int64)
    split = np.array(idx["split"])
    files = idx["files"]
    z = np.load(os.path.join(qdir, f"quant_K{cfg['k']}.npz"))
    codes = z["codes"].astype(np.int64)
    cents = z["centroids"].astype(np.float32)
    croot = os.path.join(ROOT, "corpora", cfg["corpus"])
    row_of = {f: i for i, f in enumerate(files)}
    tiles = -np.ones((len(codes), 64), np.int64)
    meta, orc = {}, {}
    for s in styles:
        meta[s] = json.load(open(os.path.join(croot, s, "style.json")))
        for r in meta[s]["swatches"]:
            j = row_of.get(f"{s}/{r['file']}")
            if j is not None and "tiles" in r:
                tiles[j] = np.asarray(r["tiles"], np.int64).reshape(-1)
    for s in styles:
        orc[s] = np.load(os.path.join(croot, s, "oracle.npz"))
    return {"styles": styles, "sidx": sidx, "split": split, "files": files, "codes": codes,
            "cents": cents, "tiles": tiles, "meta": meta, "orc": orc, "croot": croot}


def build_given(codes, sidx, tiles, rows, support, styles, support3=None):
    """The DGP's own vocabulary in the macro representation -- the `given` arm's table, and the
    reference every earned table is scored against.

    T[2] = THE TILESET'S CATALOGUE AS CODE BLOCKS. Aligned, a tile is exactly a 2x2 code block,
    so for each (school, tile id) the code blocks it renders to are read straight off genuine
    exemplars via the withheld tile ids; a block is an entry if it occurs at least `support`
    times for some (school, tile). Entries are school-specific by construction (a school is a
    palette and a geometry), and the style token -- the request, not the oracle -- is what lets
    the max-sum pick the right one.
    T[3] = THE AFFINITY-BOOSTED MOTIFS: 2x2-tile motifs at support, as 2x2 arrangements of T[2]
    ENTRIES, so the ratchet constraint holds for `given` too."""
    blk = MC.blockify(codes[rows], GRID)                       # (n, 64, 4)
    tl = tiles[rows]                                           # (n, 64)
    st = np.repeat(sidx[rows][:, None], 64, 1)
    ok = tl.reshape(-1) >= 0
    key = np.concatenate([st.reshape(-1, 1), tl.reshape(-1, 1), blk.reshape(-1, 4)], 1)[ok]
    uk, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = cnt >= support
    ents = uk[keep][:, 2:]
    ent_sty = uk[keep][:, 0]
    ent_tile = uk[keep][:, 1]
    ent_cnt = cnt[keep]
    # one entry per distinct code block; its oracle label is the tile it most often was
    ub, binv = np.unique(ents, axis=0, return_inverse=True)
    lab_tile = np.zeros(len(ub), np.int64)
    lab_sty = np.zeros(len(ub), np.int64)
    lab_pure = np.zeros(len(ub))
    for i in range(len(ub)):
        m = binv == i
        c = ent_cnt[m]
        j = int(np.argmax(c))
        lab_tile[i] = ent_tile[m][j]
        lab_sty[i] = ent_sty[m][j]
        lab_pure[i] = c[j] / c.sum()
    t2 = MC.make_table(2, ub, MC.base_table(int(codes.max()) + 1), MC.S)
    # ---- T[3] ----
    lut = {tuple(int(x) for x in r): i for i, r in enumerate(ub)}
    n = blk.shape[0]
    bid = np.array([[lut.get(tuple(int(x) for x in b), -1) for b in row] for row in blk],
                   np.int64)                                   # (n, 64) T[2] entry per tile slot
    mot_b = MC.blockify(bid, 8)                                # (n, 16, 4)
    mot_t = MC.blockify(tl, 8)
    good = (mot_b >= 0).all(-1) & (mot_t >= 0).all(-1)
    sty3 = np.repeat(sidx[rows][:, None], 16, 1)
    key3 = np.concatenate([sty3[good][:, None], mot_b[good], mot_t[good]], 1)
    u3, c3 = np.unique(key3, axis=0, return_counts=True)
    s3 = support if support3 is None else support3
    k3 = u3[c3 >= s3]
    ub3, b3inv = np.unique(k3[:, 1:5], axis=0, return_inverse=True)
    t3 = MC.make_table(3, ub3, t2, MC.S)
    motif_true = {}
    for i, r in enumerate(k3):
        motif_true.setdefault(int(b3inv[i]), []).append(tuple(int(x) for x in r[5:]))
    stats = {"n_t2": int(t2["child"].shape[0]), "n_t3": int(t3["child"].shape[0]),
             "support_t2": int(support), "support_t3": int(s3),
             "n_t3_at": {str(t): int(len(np.unique(u3[c3 >= t][:, 1:5], axis=0)))
                         for t in (1, 2, 3, 4, 8)},
             "n_t2_at": {str(t): int(len(np.unique(uk[cnt >= t][:, 2:], axis=0)))
                         for t in (1, 2, 3, 4, 8)},
             "t2_block_to_tile_purity": float(lab_pure.mean()),
             "t2_tiles_covered": int(len(np.unique(lab_tile))),
             "t2_per_style": {styles[u]: int((lab_sty == u).sum())
                              for u in np.unique(lab_sty)},
             "t3_motifs_per_entry": float(np.mean([len(v) for v in motif_true.values()]))
             if motif_true else 0.0,
             "t3_pure": float(np.mean([len(set(v)) == 1 for v in motif_true.values()]))
             if motif_true else 0.0}
    return t2, t3, {"tile": lab_tile, "style": lab_sty, "purity": lab_pure}, stats


def load_plant(cfg, n_style, device, name="plant"):
    import torch
    from canvas.plant import model as MD
    m = MD.MaskedGrid(cfg["k"], n_style, d=cfg["d"], layers=cfg["layers"],
                      heads=cfg["heads"], p=cfg["dropout"])
    m.load_state_dict(torch.load(os.path.join(ROOT, "plant", cfg["tag_plant"], f"{name}.pt"),
                                 map_location="cpu"))
    return m.to(device).eval()


# --------------------------------------------------------------------------- #
# instruments (logged, never acted on)
# --------------------------------------------------------------------------- #

def typicality(grader, grids, masks, styles, tau, side, device, n_orders=1, n_steps=1, seed=0):
    """The TASTE gauge (`plant/grader.py`): mean per-token NLL under the gA reader, against the
    oracle-free q = 0.90 quantile of genuine exemplars in the same (style, mask-size) bucket.
    It reads demand, not truth. Logged; it decides nothing."""
    from canvas.plant import grader as GR
    nll = GR.score(grader, grids, masks, styles, n_orders=n_orders, n_steps=n_steps,
                   seed=seed, dev=device)
    fb = float(np.median(list(tau.values()))) if tau else float("inf")
    t = np.array([tau.get((int(u), side), fb) for u in styles])
    return {"nll": float(nll.mean()), "pass": float((nll <= t).mean())}


def tau_from_val(grader, codes, sidx, rows, sides, device, q=0.90, reps=16, seed=21,
                 n_orders=1, n_steps=1):
    from canvas.plant import grader as GR, model as MD
    rng = np.random.default_rng(seed)
    rs, ms, ss, sd = [], [], [], []
    for side in sides:
        for i in rows:
            for _ in range(reps):
                rs.append(i); ms.append(MD.rect_mask(rng, side)); sd.append(side)
    rs = np.array(rs); ms = np.stack(ms); sd = np.array(sd)
    nll = GR.score(grader, codes[rs], ms, sidx[rs], n_orders=n_orders, n_steps=n_steps,
                   seed=seed, dev=device)
    tau = {}
    for b in set(zip(sidx[rs].tolist(), sd.tolist())):
        sel = (sidx[rs] == b[0]) & (sd == b[1])
        if sel.sum() >= 8:
            tau[b] = float(np.quantile(nll[sel], q))
    return tau


def _decode_px(codes, cents, grid=GRID, patch=16):
    p = cents[codes]
    n = p.shape[0]
    x = p.reshape(n, grid, grid, patch, patch, 3).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, grid * patch, grid * patch, 3)


def oracle_validity(fills, masks, rows, C, region_side, cap=96):
    """ORACLE COLUMN. Decode the fill to pixels, classify each 2x2 code block back to the
    nearest rendered tile, splice it into the TRUE surround, and run the edge test. Reported
    beside the clean-round-trip ceiling, because the classifier is nearest-neighbour in pixel
    space (accuracy 0.870). Never on an agent path."""
    from canvas.plant.tiles_twin.twin import classify_tiles, n_violations
    sel = np.arange(len(fills))[:cap]
    nv_f, nv_c, acc = [], [], []
    px_f = _decode_px(fills[sel], C["cents"])
    px_c = _decode_px(C["codes"][rows[sel]], C["cents"])
    for j, i in enumerate(sel):
        s = C["styles"][C["sidx"][rows[i]]]
        atlas, edges = C["orc"][s]["atlas"], C["orc"][s]["edges"]
        gt = C["tiles"][rows[i]].reshape(8, 8)
        if (gt < 0).any():
            continue
        gf, _ = classify_tiles(px_f[j], atlas)
        gc, _ = classify_tiles(px_c[j], atlas)
        hole = masks[i].reshape(GRID, GRID)[0::2, 0::2] & masks[i].reshape(GRID, GRID)[1::2, 1::2]
        nv_f.append(n_violations(np.where(hole, gf, gt), edges))
        nv_c.append(n_violations(np.where(hole, gc, gt), edges))
        acc.append(float((gc == gt).mean()))
    if not nv_f:
        return {}
    nv_f = np.array(nv_f); nv_c = np.array(nv_c)
    clean = nv_c == 0
    return {"n": int(len(nv_f)), "valid_fill": float((nv_f == 0).mean()),
            "valid_roundtrip_ceiling": float(clean.mean()),
            "valid_fill_on_clean_rt": float((nv_f[clean] == 0).mean()) if clean.any() else None,
            "viol_fill": float(nv_f.mean()), "classify_acc": float(np.mean(acc))}


def finetune_plant(plant, opt, new_codes, new_sty, rep_codes, rep_sty, *, n_steps, bs,
                   replay_frac, rng, device):
    """THE PLANT LEARNS, on configurations the agent actually solved (a support-passing fill is
    a legal completion), with replay against the clean pool as the anti-drift guard. Carried by
    every arm, so it is a substrate property and not an arm difference."""
    import torch
    import torch.nn.functional as F
    from canvas.plant import model as MD
    plant.train()
    last = 0.0
    n_rep = int(round(bs * replay_frac))
    n_new = bs - n_rep
    for _ in range(n_steps):
        cs, ss = [], []
        if n_new and len(new_codes):
            i = rng.integers(0, len(new_codes), size=n_new)
            cs.append(new_codes[i]); ss.append(new_sty[i])
        k = n_rep if cs else bs
        i = rng.integers(0, len(rep_codes), size=k)
        cs.append(rep_codes[i]); ss.append(rep_sty[i])
        x = torch.as_tensor(np.concatenate(cs), dtype=torch.long, device=device)
        s = torch.as_tensor(np.concatenate(ss), dtype=torch.long, device=device)
        m = torch.as_tensor(MD.batch_masks(rng, x.shape[0], GRID), device=device)
        y = x.clone()
        x = x.clone()
        x[m] = plant.K
        with MD.amp(device):
            logits = plant(x, s)
        loss = F.cross_entropy(logits[m].float(), y[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plant.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
    plant.eval()
    return last


def audition(plant, grids, masks, styles, pt, sup, lam, device, ichunk=48, use_style=True):
    """The macro's own audition: ONE action -- one forward, then the level-l macro applied at
    every one of its nodes lying inside the hole. At era k the hole IS one level-(k+1) block,
    so for the level being earned this is literally one move and the audition is
    consumption-matched (the donor's depth seam does not exist here). Returns the filled grids;
    cost is one forward per instance."""
    import torch
    from canvas.plant.model import amp
    out = np.array(grids, np.int64, copy=True)
    with torch.no_grad():
        plant.eval()
        for a in range(0, len(grids), ichunk):
            b = min(a + ichunk, len(grids))
            X = torch.as_tensor(np.asarray(grids)[a:b], dtype=torch.long, device=device)
            M = torch.as_tensor(np.asarray(masks)[a:b], device=device)
            S = torch.as_tensor(np.asarray(styles)[a:b], dtype=torch.long, device=device)
            if not use_style:
                S = torch.full_like(S, plant.n_styles)
            X = torch.where(M, torch.full_like(X, plant.K), X)
            with amp(device):
                logits = plant(X, S)
            lp = torch.log_softmax(logits.float(), -1)
            _, ent, app, _, _, _ = _macro_cands(lp, lp.max(-1).values, X, M, pt, sup, lam, 1)
            ent = ent[:, :, 0]
            rr, nn_ = torch.nonzero(app, as_tuple=True)
            if len(rr):
                cc = pt["cells"][nn_]
                X[rr[:, None], cc] = pt["flat"][ent[rr, nn_]]
            out[a:b] = X.cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# the arm loop
# --------------------------------------------------------------------------- #

ARMS = {
    "never_base":     {"vocab": "base",   "commit": None},
    "given":          {"vocab": "true",   "commit": None},
    "practice_gated": {"vocab": "earned", "commit": "delta"},
    "practice_early": {"vocab": "earned", "commit": "early"},
    "practice_late":  {"vocab": "earned", "commit": "late"},
    "practice_prov":  {"vocab": "earned", "commit": "prov"},
}


def parse_arms(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append((part, part.split(":")[0]))
    return out


def mine_blocks(fills, masks, level):
    """Every level-`level` block lying ENTIRELY INSIDE THE HOLE of a solved fill. The agent's
    own answers, parsed by BLOCKIFY -- the alphabet is the parse, so there is nothing to read.
    The exemplar surround is never mined."""
    cells = np.stack([MC.zcells(level, j) for j in range(MC.n_nodes(level))])
    inside = masks[:, cells].all(-1)
    r, n = np.nonzero(inside)
    if len(r) == 0:
        return np.zeros((0, cells.shape[1]), np.int64)
    return fills[r[:, None], cells[n]]


def run_arm(label, base, shared, cfg, eras, outdir, device):
    import copy
    import torch
    from canvas.plant import model as MD
    spec = ARMS[base]
    maxl = cfg["max_macro_level"]
    sup, nb = shared["sup"], shared["nb"]
    lam, gamma, n_alt = cfg["lam"], cfg["gamma"], cfg["n_alt"]
    C = shared["C"]

    plant = copy.deepcopy(shared["plant0"]).to(device)
    for p in plant.parameters():
        p.requires_grad_(True)
    gopt = torch.optim.AdamW(plant.parameters(), lr=cfg["gen_lr"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"] + 77)
    grng = np.random.default_rng(cfg["seed"] + 991)

    committed = {l: (shared["given"][l] if spec["vocab"] == "true" else None)
                 for l in range(2, maxl + 1)}
    miners = {l: MC.Miner(l) for l in range(2, maxl + 1)}
    prepped = {l: (prep_table(committed[l], l, shared["supH"], shared["supV"], device)
                   if committed[l] is not None else None) for l in range(2, maxl + 1)}
    given_pt = {l: prep_table(shared["given"][l], l, shared["supH"], shared["supV"], device)
                for l in range(2, maxl + 1)}

    def operative(ell):
        if ell == 1:
            return MC.base_table(cfg["k"])
        if committed[ell] is not None:
            return committed[ell]
        return miners[ell].build(operative(ell - 1), cfg["mine_support"])

    def solve_(grids, masks, styles, tabs, **kw):
        return solve(plant, grids, masks, styles, tabs, sup, nb, budget=cfg["budget"],
                     w_max=cfg["w_max"], gamma=gamma, lam=lam, n_alt=n_alt, device=device,
                     ichunk=cfg["ichunk"], **kw)

    log = {"cycle": [], "era": [], "t_cum": [], "e": [], "e_practice": [], "cost": [],
           "width": [], "steps": [], "n_moves": [], "clean_fr": [], "typ": [],
           "n_solved": [], "n_mined": [], "miner": [], "aud": [], "cert": [], "vocab": [],
           "gloss": [], "incomplete": [], "n_unsup": [], "probe": []}
    events = []
    t_cum, cyc = 0.0, 0

    for era_i, era in enumerate(eras):
        active = era + 1
        side = 2 ** era
        mt, ad = shared["meter"][era], shared["audit"][era]
        cert = {"b": None, "ref": None, "emin": None, "hist": [], "dhist": [], "run": 0,
                "fired": None}
        print(f"\n----- arm={label} ERA {era_i + 1}: hole L{era} ({side}x{side} codes, "
              f"earning level {active}) -----", flush=True)
        clean_fr = 1.0 - float(grade_support(C["codes"][mt["rows"]], mt["masks"],
                                             shared["supH"], shared["supV"])["pass"].mean())
        for c_in_era in range(1, cfg["era_cycles"] + 1):
            cyc += 1
            fwd = 0.0
            tabs = {l: prepped[l] for l in range(2, maxl + 1) if prepped[l] is not None}

            # --- (a) practice ------------------------------------------------------- #
            pr = make_instances(shared["rows_practice"], shared["sty_practice"], cfg["n_pr"],
                                np.random.default_rng(cfg["seed"] + 100_000 + 1000 * cyc))
            pm = era_masks(pr, era)
            out = solve_(C["codes"][pr["row"]], pm, pr["style"], tabs)
            fwd += float(out["cost"].sum())
            g = grade_support(out["x"], pm, shared["supH"], shared["supV"])
            e_practice = 1.0 - float(g["pass"].mean())
            solved = out["x"][g["pass"]]
            sm = pm[g["pass"]]
            ssty = pr["style"][g["pass"]]

            # --- (b) mining, from the agent's own solved fills ----------------------- #
            n_mined = 0
            if len(solved):
                if cfg["mine_cap"] and len(solved) > cfg["mine_cap"]:
                    sel = rng.permutation(len(solved))[:cfg["mine_cap"]]
                    msrc, msk_src = solved[sel], sm[sel]
                else:
                    msrc, msk_src = solved, sm
                n_mined = len(msrc)
                for ell in range(2, min(maxl, era + 1) + 1):
                    miners[ell].observe(mine_blocks(msrc, msk_src, ell))

            # --- (c) the plant learns ----------------------------------------------- #
            gloss = finetune_plant(plant, gopt, solved, ssty, shared["replay_codes"],
                                   shared["replay_sty"], n_steps=cfg["gen_steps"],
                                   bs=cfg["gen_bs"], replay_frac=cfg["replay_frac"],
                                   rng=grng, device=device) if cfg["gen_steps"] else 0.0

            # --- (d) metering at the declared budget --------------------------------- #
            b = solve_(C["codes"][mt["rows"]], mt["masks"], mt["style"], tabs)
            fwd += float(b["cost"].sum())
            mg = grade_support(b["x"], mt["masks"], shared["supH"], shared["supV"])
            e = 1.0 - float(mg["pass"].mean())
            typ = typicality(shared["grader"], b["x"], mt["masks"], mt["style"],
                             shared["tau"], side, device, seed=cfg["seed"] + cyc)

            # --- (e) THE SHADOW AUDITION -------------------------------------------- #
            aud = {}
            for ell in range(2, maxl + 1):
                cell = {}
                if ell > active:
                    # a level-l block is wider than this era's hole: no action is possible,
                    # so there is nothing to audition (the donor's `active` cap, spatially)
                    tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                    aud[str(ell)] = {"n_entries": int(tbl["child"].shape[0]),
                                     "cand": None, "true": None, "not_applicable": True}
                    continue
                tbl = miners[ell].build(operative(ell - 1), cfg["mine_support"])
                cell["n_entries"] = int(tbl["child"].shape[0])
                if cell["n_entries"]:
                    pt = prep_table(tbl, ell, shared["supH"], shared["supV"], device)
                    f = audition(plant, C["codes"][ad["rows"]], ad["masks"], ad["style"],
                                 pt, sup, lam, device, ichunk=cfg["ichunk"])
                    cell["cand"] = 1.0 - float(grade_support(
                        f, ad["masks"], shared["supH"], shared["supV"])["pass"].mean())
                    cell.update({f"tab_{k}": v for k, v in
                                 MC.grade_table(tbl, shared["given"][ell]).items()})
                    if ell == active and committed.get(ell) is None:
                        fwd += len(ad["rows"])            # priced: the agent's own check
                else:
                    cell["cand"] = None
                ft = audition(plant, C["codes"][ad["rows"]], ad["masks"], ad["style"],
                              given_pt[ell], sup, lam, device, ichunk=cfg["ichunk"])
                cell["true"] = 1.0 - float(grade_support(
                    ft, ad["masks"], shared["supH"], shared["supV"])["pass"].mean())
                # matched-size random subsets of the given table (concentration vs coverage)
                if cell["n_entries"]:
                    full = shared["given"][ell]
                    n_all = full["child"].shape[0]
                    kk = min(cell["n_entries"], n_all)
                    es = []
                    for _ in range(cfg["n_rand"]):
                        keep = np.sort(rng.permutation(n_all)[:kk])
                        sub = MC.make_table(ell, full["child"][keep], full["lower"], MC.S)
                        ptr = prep_table(sub, ell, shared["supH"], shared["supV"], device)
                        fr = audition(plant, C["codes"][ad["rows"]], ad["masks"], ad["style"],
                                      ptr, sup, lam, device, ichunk=cfg["ichunk"])
                        es.append(1.0 - float(grade_support(
                            fr, ad["masks"], shared["supH"], shared["supV"])["pass"].mean()))
                    cell["rand_k"] = float(np.mean(es))
                    cell["rand_k_sd"] = float(np.std(es))
                if committed[ell] is not None and spec["vocab"] == "earned":
                    # the recert counterfactual: frozen vs what the live vocabulary would be.
                    fh = audition(plant, C["codes"][ad["rows"]], ad["masks"], ad["style"],
                                  prepped[ell], sup, lam, device, ichunk=cfg["ichunk"])
                    cell["held"] = 1.0 - float(grade_support(
                        fh, ad["masks"], shared["supH"], shared["supV"])["pass"].mean())
                    live = miners[ell].build(
                        MC.base_table(cfg["k"]) if ell == 2
                        else miners[ell - 1].build(MC.base_table(cfg["k"]), cfg["mine_support"]),
                        cfg["mine_support"])
                    if live["child"].shape[0]:
                        ptl = prep_table(live, ell, shared["supH"], shared["supV"], device)
                        fl = audition(plant, C["codes"][ad["rows"]], ad["masks"], ad["style"],
                                      ptl, sup, lam, device, ichunk=cfg["ichunk"])
                        cell["live"] = 1.0 - float(grade_support(
                            fl, ad["masks"], shared["supH"], shared["supV"])["pass"].mean())
                        cell["live_entries"] = int(live["child"].shape[0])
                aud[str(ell)] = cell

            # --- (f) the unit-LP certificate ---------------------------------------- #
            A = aud.get(str(active), {}).get("cand")
            fire = False
            if A is not None and committed.get(active) is None:
                if cert["ref"] is None:
                    cert["ref"] = A; cert["b"] = A; cert["emin"] = A
                cert["emin"] = min(cert["emin"], A)
                d_ = cert["b"] - A
                scale = max(cert["ref"] - cert["emin"], cert["b"], 1e-6)
                cert["hist"].append(A); cert["dhist"].append(d_)
                we, wd = cert["hist"][-cfg["sil_W"]:], cert["dhist"][-cfg["sil_W"]:]
                quiet = (len(we) >= cfg["sil_W"]
                         and abs(float(np.mean(wd))) < cfg["sil_c"] * scale
                         and float(np.std(we)) < cfg["sil_cv"] * scale)
                cert["run"] = cert["run"] + 1 if quiet else 0
                cert["b"] = cert["b"] + cfg["alpha"] * (A - cert["b"])
                dropped = (cert["ref"] - cert["emin"]) >= cfg["lp_min_drop"]
                fire = (cert["run"] >= cfg["sil_hold"] and dropped
                        and c_in_era >= cfg["sil_min_cycle"])

            # --- (g) the commit rule ------------------------------------------------- #
            do_commit, prov = False, False
            if spec["commit"] and active <= maxl and committed.get(active) is None:
                at_boundary = c_in_era >= cfg["era_cycles"] - cfg["prov_offset"]
                if spec["commit"] == "delta":
                    do_commit = fire
                elif spec["commit"] == "early":
                    do_commit = c_in_era >= cfg["early_offset"]
                elif spec["commit"] == "late":
                    do_commit = c_in_era >= cfg["era_cycles"] - cfg["late_offset"]
                elif spec["commit"] == "prov":
                    do_commit, prov = at_boundary, True
            if do_commit:
                tbl = miners[active].build(operative(active - 1), cfg["mine_support"])
                if tbl["child"].shape[0] == 0:
                    do_commit = False
            if do_commit:
                held = None
                if not prov:
                    cm = shared["confirm"][era]
                    pt = prep_table(tbl, active, shared["supH"], shared["supV"], device)
                    fc = audition(plant, C["codes"][cm["rows"]], cm["masks"], cm["style"],
                                  pt, sup, lam, device, ichunk=cfg["ichunk"])
                    held = 1.0 - float(grade_support(
                        fc, cm["masks"], shared["supH"], shared["supV"])["pass"].mean())
                    fwd += len(cm["rows"])
                committed[active] = tbl
                prepped[active] = prep_table(tbl, active, shared["supH"], shared["supV"],
                                             device)
                cert["fired"] = cyc
                ev = dict(kind="commit", arm=label, era=era_i + 1, level=active, cycle=cyc,
                          c_in_era=c_in_era, t_cum=t_cum + fwd, provisional=bool(prov),
                          audition=held, shadow=A, e_task=e,
                          n_entries=int(tbl["child"].shape[0]), sil_run=int(cert["run"]),
                          **{f"tab_{k}": v for k, v in
                             MC.grade_table(tbl, shared["given"][active]).items()})
                events.append(ev)
                print(f"[commit] arm={label} era{era_i+1} L{active} c{cyc} "
                      f"{'PROVISIONAL ' if prov else ''}entries={ev['n_entries']} "
                      f"recall={ev['tab_recall']:.3f} prec={ev['tab_precision']} "
                      f"audition={held} shadow={A}", flush=True)

            # --- (h) probes: instruments, never priced ------------------------------- #
            probe = None
            if c_in_era == 1 or cyc % cfg["probe_every"] == 0 or c_in_era == cfg["era_cycles"]:
                tabs2 = {l: prepped[l] for l in range(2, maxl + 1) if prepped[l] is not None}
                probe = {"cycle": cyc, "era": era_i + 1, "ladder": {}, "all_eras": {}}
                for w in cfg["probe_widths"]:
                    bb = solve_(C["codes"][mt["rows"]], mt["masks"], mt["style"], tabs2,
                                force_width=w)
                    sc = grade_support(bb["x"], mt["masks"], shared["supH"], shared["supV"])
                    probe["ladder"][str(w)] = {"e": 1.0 - float(sc["pass"].mean()),
                                               "cost": float(bb["cost"].mean())}
                for j in eras:
                    mj = shared["meter"][j]
                    bb = solve_(C["codes"][mj["rows"]], mj["masks"], mj["style"], tabs2)
                    sc = grade_support(bb["x"], mj["masks"], shared["supH"], shared["supV"])
                    probe["all_eras"][str(j)] = {"e": 1.0 - float(sc["pass"].mean()),
                                                 "cost": float(bb["cost"].mean())}
                probe["oracle"] = oracle_validity(b["x"], mt["masks"], mt["rows"], C, side,
                                                  cap=cfg["oracle_cap"])
                probe["plant"] = {"ladder_nll": MD.ladder_nll(
                    plant, C["codes"][shared["rows_val"]], C["sidx"][shared["rows_val"]],
                    ladder=(2, 4, 8), reps=cfg["guard_reps"], seed=cfg["seed"], dev=device
                ).mean(0).tolist()}
                log["probe"].append(probe)

            t_cum += fwd
            log["cycle"].append(cyc); log["era"].append(era_i + 1); log["t_cum"].append(t_cum)
            log["e"].append(e); log["e_practice"].append(e_practice)
            log["cost"].append(float(b["cost"].mean())); log["width"].append(b["width"])
            log["steps"].append(b["steps"]); log["n_moves"].append(b["n_moves"])
            log["clean_fr"].append(clean_fr); log["typ"].append(typ)
            log["n_solved"].append(int(len(solved))); log["n_mined"].append(int(n_mined))
            log["miner"].append({str(l): miners[l].state() for l in range(2, maxl + 1)})
            log["aud"].append(aud)
            log["cert"].append({"A": A, "b": cert["b"], "run": int(cert["run"]),
                                "ref": cert["ref"], "emin": cert["emin"]})
            log["vocab"].append({str(l): (None if committed[l] is None
                                          else int(committed[l]["child"].shape[0]))
                                 for l in range(2, maxl + 1)})
            log["gloss"].append(gloss)
            log["incomplete"].append(float(b["incomplete"].mean()))
            log["n_unsup"].append(float(mg["n_unsup"].mean()))
            a_act = aud.get(str(active), {})
            print(f"[c{cyc:3d}] {label:15s} era{era_i+1} t={t_cum:9.0f} e={e:.4f} "
                  f"(clean_fr={clean_fr:.3f}) w={b['width']} st={b['steps']} "
                  f"nm={b['n_moves']} solved={len(solved):3d} "
                  f"A{active}={a_act.get('cand')} true={a_act.get('true')} "
                  f"ent={a_act.get('n_entries')} rec={a_act.get('tab_recall')} "
                  f"typ={typ['pass']:.2f} sil={cert['run']}", flush=True)
            if cyc % cfg["checkpoint_every"] == 0:
                write_results(outdir, label, cfg, eras, log, events, False)
    write_results(outdir, label, cfg, eras, log, events, True)
    return {"log": log, "events": events}


def write_results(outdir, arm, cfg, eras, log, events, complete):
    d = os.path.join(outdir, arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "results.json"), "w") as fh:
        json.dump({"arm": arm, "config": cfg, "eras": eras, "log": log, "events": events,
                   "complete": complete}, fh, indent=1, cls=NumpyEncoder)
    _commit()


# --------------------------------------------------------------------------- #
# setup shared by every arm
# --------------------------------------------------------------------------- #

def build_shared(cfg, eras, device):
    import torch
    C = load_corpus(cfg)
    styles = C["styles"]
    sel = np.array([i for i, s in enumerate(styles) if s.rsplit("_", 1)[0] == cfg["tileset"]])
    inset = np.isin(C["sidx"], sel)

    def rows(*names):
        return np.flatnonzero(inset & np.isin(C["split"], list(names)))

    r_sup = rows(*SPLIT_SUPPORT)
    supH, supV = support_matrices(C["codes"][r_sup], k=cfg["k"])
    print(f"support from {len(r_sup)} exemplars ({'+'.join(SPLIT_SUPPORT)}, pooled over the "
          f"{len(sel)} schools of tileset '{cfg['tileset']}'): "
          f"H {int(supH.sum())} V {int(supV.sum())} distinct oriented pairs", flush=True)

    r_prac = rows(SPLIT_PRACTICE)
    r_meter = rows(SPLIT_METER)
    r_audit = rows(SPLIT_AUDIT)
    g2, g3, glab, gstats = build_given(C["codes"], C["sidx"], C["tiles"],
                                       np.concatenate([r_prac, r_sup]), cfg["given_support"],
                                       styles, cfg["given_support3"])
    print(f"[given] {json.dumps(gstats, cls=NumpyEncoder)}", flush=True)

    plant0 = load_plant(cfg, len(styles), device, "plant")
    grader = load_plant(cfg, len(styles), device, cfg["grader"])
    tau = tau_from_val(grader, C["codes"], C["sidx"], rows("gA_val"), (2, 4, 8), device,
                       q=cfg["q"], reps=cfg["tau_masks"], seed=cfg["seed"] + 21)

    rr = np.random.default_rng(cfg["seed"] + 5000)
    im = make_instances(r_meter, C["sidx"][r_meter], cfg["n_rt"], rr)
    ia = make_instances(r_audit, C["sidx"][r_audit], cfg["n_score"],
                        np.random.default_rng(cfg["seed"] + 6100))
    ic = make_instances(r_audit, C["sidx"][r_audit], cfg["n_score"],
                        np.random.default_rng(cfg["seed"] + 7300))

    def per_era(inst):
        return {e: {"rows": inst["row"], "style": inst["style"], "masks": era_masks(inst, e),
                    "nodes": era_nodes(inst, e)} for e in eras}

    pre = {}
    for e in eras:
        mk = era_masks(im, e)
        pre[f"clean_pass_L{e}"] = float(grade_support(
            C["codes"][im["row"]], mk, supH, supV)["pass"].mean())
    print(f"[preflight] clean pass by level {pre}", flush=True)

    return {"C": C, "styles": styles, "sel": sel, "supH": supH, "supV": supV,
            "sup": sup_tensors(supH, supV, device),
            "nb": torch.from_numpy(MC.cell_neighbours()[0]).to(device),
            "given": {2: g2, 3: g3}, "given_label": glab, "given_stats": gstats,
            "plant0": plant0, "grader": grader, "tau": tau,
            "meter": per_era(im), "audit": per_era(ia), "confirm": per_era(ic),
            "rows_practice": r_prac, "sty_practice": C["sidx"][r_prac],
            "rows_val": r_audit,
            "replay_codes": C["codes"][r_prac], "replay_sty": C["sidx"][r_prac],
            "preflight": pre, "n_support_exemplars": int(len(r_sup))}


def _cfg(**kw):
    cfg = dict(
        tag_plant="tw_aligned", qtag="twq_aligned", corpus="tiles_aligned", tileset="full",
        grader="gA24", k=512, d=256, layers=6, heads=8, dropout=0.1, q=0.90, tau_masks=16,
        seed=0, era_cycles=24, budget=8, w_max=2, gamma=0.99, lam=8.0, n_alt=2,
        max_macro_level=3, given_support=3, given_support3=3, mine_support=3, mine_cap=0, n_rand=3,
        n_pr=128, n_rt=144, n_score=96, ichunk=48,
        gen_lr=1e-4, gen_steps=10, gen_bs=64, replay_frac=0.5, guard_reps=4,
        alpha=0.2, sil_c=0.06, sil_cv=0.15, sil_W=5, sil_hold=2, sil_min_cycle=6,
        lp_min_drop=0.10, early_offset=1, late_offset=3, prov_offset=0,
        probe_every=6, probe_widths=(1, 2, 4), oracle_cap=96, checkpoint_every=4,
    )
    cfg.update(kw)
    return cfg


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=28800, memory=65536)
def run(tag: str = "cr_s0",
        arms: str = ("never_base,given,practice_gated,practice_early,practice_late,"
                     "practice_prov"),
        eras: str = "1,2,3", seed: int = 0, era_cycles: int = 24, budget: int = 8,
        w_max: int = 2, lam: float = 8.0, gamma: float = 0.99, n_alt: int = 2,
        n_pr: int = 128, n_rt: int = 144, n_score: int = 96, mine_support: int = 3,
        mine_cap: int = 0, given_support: int = 3, given_support3: int = 3,
        gen_steps: int = 10,
        gen_lr: float = 1e-4, probe_every: int = 6, sil_c: float = 0.06,
        sil_cv: float = 0.15, sil_win: int = 5, sil_hold: int = 2, sil_min_cycle: int = 6,
        lp_min_drop: float = 0.10, early_offset: int = 1, late_offset: int = 3,
        tileset: str = "full", quick: int = 0):
    import torch
    cfg = _cfg(seed=seed, era_cycles=era_cycles, budget=budget, w_max=w_max, lam=lam,
               gamma=gamma, n_alt=n_alt, n_pr=n_pr, n_rt=n_rt, n_score=n_score,
               mine_support=mine_support, mine_cap=mine_cap, given_support=given_support,
               given_support3=given_support3,
               gen_steps=gen_steps, gen_lr=gen_lr, probe_every=probe_every, sil_c=sil_c,
               sil_cv=sil_cv, sil_W=sil_win, sil_hold=sil_hold, sil_min_cycle=sil_min_cycle,
               lp_min_drop=lp_min_drop, early_offset=early_offset, late_offset=late_offset,
               tileset=tileset)
    if quick:
        cfg.update(era_cycles=2, n_pr=12, n_rt=24, n_score=24, gen_steps=2, probe_every=100,
                   oracle_cap=16, guard_reps=1, sil_min_cycle=1, early_offset=1,
                   late_offset=0, prov_offset=0, n_rand=1, checkpoint_every=1, tau_masks=2,
                   probe_widths=(1,))
    ers = [int(x) for x in eras.split(",")]
    device = torch.device(DEV if torch.cuda.is_available() or DEV == "cpu" else "cpu")
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    print(f"canvas ratchet tag={tag} arms={arms} eras={ers} device={device}", flush=True)

    shared = build_shared(cfg, ers, device)
    outdir = os.path.join(ROOT, REMOTE, tag)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "setup.json"), "w") as fh:
        json.dump({"config": cfg, "eras": ers, "preflight": shared["preflight"],
                   "given_stats": shared["given_stats"],
                   "n_support_exemplars": shared["n_support_exemplars"],
                   "given_sizes": {str(l): int(shared["given"][l]["child"].shape[0])
                                   for l in (2, 3)}}, fh, indent=1, cls=NumpyEncoder)
    _commit()

    for label, base in parse_arms(arms):
        print(f"\n===== arm {label} =====", flush=True)
        run_arm(label, base, shared, cfg, ers, outdir, device)
    with open(os.path.join(outdir, "done.txt"), "w") as fh:
        fh.write(f"elapsed {time.time() - t0:.0f}s\n")
    _commit()
    print(f"\nDONE in {time.time() - t0:.0f}s -> {outdir}")
    return {"elapsed": time.time() - t0}


# --------------------------------------------------------------------------- #
# calibration: the depth ladder, before the loop is built
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=10800, memory=65536)
def ladder(tag: str = "cal0", tileset: str = "full", budgets: str = "2,4,8,16,32,64,128",
           n_rt: int = 144, lam: float = 8.0, gamma: float = 0.99, n_alt: int = 2,
           w_max: int = 2, seed: int = 0, given_support: int = 3, given_support3: int = 3):
    """What does the declared forward-pass budget buy each action set, per era, on identical
    held-out instances? This is what picks G: it has to be a regime where depth is unaffordable
    to base moves and affordable with the vocabulary, otherwise the node's headline is a
    property of a budget chosen by hand.

    Action sets: `base` (level-1 cells only), `+T2` (the tile catalogue), `+T2+T3` (catalogue +
    motifs), each at every budget; plus `base, lam=0` -- what the plant alone does with no
    support signal, i.e. `sampler.decode` at that (steps, width). The grader's CLEAN
    false-reject floor is printed beside every column, because no arm can go below it."""
    import torch
    cfg = _cfg(tileset=tileset, seed=seed, n_rt=n_rt, lam=lam, gamma=gamma, n_alt=n_alt,
               w_max=w_max, given_support=given_support, given_support3=given_support3)
    device = torch.device(DEV if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    shared = build_shared(cfg, [1, 2, 3], device)
    C = shared["C"]
    plant = shared["plant0"]
    sup, nb = shared["sup"], shared["nb"]
    pt2 = prep_table(shared["given"][2], 2, shared["supH"], shared["supV"], device)
    pt3 = prep_table(shared["given"][3], 3, shared["supH"], shared["supV"], device)
    sets = {"base": {}, "+T2": {2: pt2}, "+T2+T3": {2: pt2, 3: pt3}}
    R = {"tag": tag, "config": cfg, "floor": shared["preflight"], "rows": {}}
    for era in (1, 2, 3):
        mt = shared["meter"][era]
        for name, tabs in sets.items():
            for g in [int(x) for x in budgets.split(",")]:
                for lm in ((lam, 0.0) if name == "base" else (lam,)):
                    o = solve(plant, C["codes"][mt["rows"]], mt["masks"], mt["style"], tabs,
                              sup, nb, budget=g, w_max=w_max, gamma=gamma, lam=lm,
                              n_alt=n_alt, device=device, ichunk=cfg["ichunk"])
                    gr = grade_support(o["x"], mt["masks"], shared["supH"], shared["supV"])
                    key = f"L{era}|{name}{'' if lm else ',lam0'}|G{g}"
                    R["rows"][key] = {"e": 1.0 - float(gr["pass"].mean()),
                                      "cost": float(o["cost"].mean()), "width": o["width"],
                                      "steps": o["steps"], "n_moves": o["n_moves"],
                                      "n_unsup": float(gr["n_unsup"].mean()),
                                      "incomplete": float(o["incomplete"].mean())}
                    print(f"[{key:26s}] e={R['rows'][key]['e']:.4f} "
                          f"cost={o['cost'][0]:6.0f} w={o['width']} steps={o['steps']} "
                          f"nm={o['n_moves']}  ({time.time()-t0:.0f}s)", flush=True)
    # one-action ceilings: what ONE given macro buys at the era's own node
    for era in (1, 2):
        ad = shared["audit"][era]
        for l, pt in ((2, pt2), (3, pt3)):
            if l > era + 1:
                continue
            f = audition(plant, C["codes"][ad["rows"]], ad["masks"], ad["style"], pt, sup,
                         lam, device, ichunk=cfg["ichunk"])
            gr = grade_support(f, ad["masks"], shared["supH"], shared["supV"])
            R["rows"][f"L{era}|one_macro_L{l}|G1"] = {
                "e": 1.0 - float(gr["pass"].mean()), "cost": 1.0}
            print(f"[L{era}|one given L{l} macro, 1 forward] "
                  f"e={1.0 - float(gr['pass'].mean()):.4f}", flush=True)
    out = os.path.join(ROOT, REMOTE, tag)
    os.makedirs(out, exist_ok=True)
    json.dump(R, open(os.path.join(out, "ladder.json"), "w"), indent=1, cls=NumpyEncoder)
    _commit()
    print(f"\nladder done in {time.time()-t0:.0f}s -> {out}")
    return R


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #

@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=5400, memory=65536)
def selfcheck(tileset: str = "full", n: int = 64, given_support: int = 8):
    """Bit-for-bit gates, in the order they constrain the design.

    F   FORK FIDELITY -- with the vocabulary off, lam = 0, width 1 and one cell per step, the
        arm reproduces `sampler.decode`'s fill AND its cost exactly on a fixed batch.
    C-M the operator -- the donor's max-sum DP over the table chain equals the flat gather the
        loop actually runs, at both levels, so `given` and `practice_*` are the same machinery
        differing only in which tuples are in the table.
    C-B the flattening round-trips `recur.py`'s block ids at both levels.
    C-R the ratchet bites: truncating T[2] drops the rebuilt T[3].
    G   the given table's entries decode to catalogue tiles through the oracle.
    P   the grade of record, before the loop runs: clean pass and `seam` reject per level under
        the pooled support. REPORTED, NOT TUNED -- if the L3 floor binds, the lever is more
        exemplars, not a threshold (`plant/README.md` finding 12)."""
    import torch
    from canvas.plant import codebook as CB, model as MD
    from canvas.plant.sampler import decode
    device = torch.device(DEV if torch.cuda.is_available() else "cpu")
    cfg = _cfg(tileset=tileset, given_support=given_support)
    C = load_corpus(cfg)
    styles = C["styles"]
    sel = np.array([i for i, s in enumerate(styles) if s.rsplit("_", 1)[0] == tileset])
    inset = np.isin(C["sidx"], sel)
    r_sup = np.flatnonzero(inset & np.isin(C["split"], list(SPLIT_SUPPORT)))
    r_prac = np.flatnonzero(inset & (C["split"] == SPLIT_PRACTICE))
    r_test = np.flatnonzero(inset & (C["split"] == SPLIT_METER))
    supH, supV = support_matrices(C["codes"][r_sup], k=cfg["k"])
    out = {"tileset": tileset, "n_support_exemplars": int(len(r_sup)),
           "support_pairs": {"H": int(supH.sum()), "V": int(supV.sum())}}

    g2, g3, glab, gstats = build_given(C["codes"], C["sidx"], C["tiles"],
                                       np.concatenate([r_prac, r_sup]), given_support, styles,
                                       cfg["given_support3"])
    out["G_given"] = gstats
    assert gstats["t2_block_to_tile_purity"] >= 0.85, f"G failed: {gstats}"

    # ---- C-B: the flattening IS recur.py's blockify, at both levels -------------------- #
    cod = C["codes"][r_prac[:32]]
    blk = MC.blockify(cod, GRID)
    z2 = np.stack([MC.zcells(2, j) for j in range(64)])
    assert (cod[:, z2] == blk).all(), "C-B failed at level 2"
    z3 = np.stack([MC.zcells(3, j) for j in range(16)])
    ub, binv = np.unique(blk.reshape(-1, 4), axis=0, return_inverse=True)
    t2 = MC.make_table(2, ub, MC.base_table(cfg["k"]), MC.S)
    child3 = MC.blockify(binv.reshape(len(cod), 64), 8)
    t3 = MC.make_table(3, child3.reshape(-1, 4), t2, MC.S)
    assert (t3["flat"].reshape(len(cod), 16, 16) == cod[:, z3]).all(), "C-B failed at level 3"
    out["CB_blockify_roundtrip"] = True

    # ---- C-R: the ratchet constraint --------------------------------------------------- #
    mn = MC.Miner(3)
    mn.observe(cod[:, z3].reshape(-1, 16))
    full = mn.build(t2, 1)
    part = mn.build(MC.truncate(t2, max(4, t2["child"].shape[0] // 4)), 1)
    out["CR_ratchet"] = {"t2_full": int(t2["child"].shape[0]),
                         "t2_trunc": int(max(4, t2["child"].shape[0] // 4)),
                         "t3_over_full_t2": int(full["child"].shape[0]),
                         "t3_over_truncated_t2": int(part["child"].shape[0])}
    assert part["child"].shape[0] < full["child"].shape[0], f"C-R failed: {out['CR_ratchet']}"

    # ---- C-M: the donor DP == the flat gather ------------------------------------------ #
    cm = {}
    for l, tbl in ((2, g2), (3, g3)):
        span = 4 ** (l - 1)
        cur = torch.randn(8, span, cfg["k"], device=device)
        chain = [torch.from_numpy(np.ascontiguousarray(c)).to(device) for c in MC.chain_of(tbl)]
        a = MC.chain_scores(cur, chain)
        b = MC.flat_scores(cur, torch.from_numpy(np.ascontiguousarray(tbl["flat"])).to(device))
        cm[f"L{l}"] = float((a - b).abs().max())
        assert cm[f"L{l}"] < 1e-3, f"C-M failed at L{l}: {cm}"
    out["CM_dp_equals_gather"] = cm

    # ---- F: fork fidelity against sampler.decode --------------------------------------- #
    plant = load_plant(cfg, len(styles), device, "plant")
    sup = sup_tensors(supH, supV, device)
    nb = torch.from_numpy(MC.cell_neighbours()[0]).to(device)
    rr = np.random.default_rng(3)
    im = make_instances(r_test, C["sidx"][r_test], n, rr)
    fk = {}
    for era in (1, 2, 3):
        mk = era_masks(im, era)
        nmask = int(mk[0].sum())
        hole = np.where(mk, cfg["k"], C["codes"][im["row"]])
        for st in sorted({nmask, min(8, nmask), min(2, nmask)}):
            d, dcost = decode(plant, hole, mk, im["style"], steps=st, width=1, seed=0,
                              dev=device, chunk=n)
            s_ = solve(plant, C["codes"][im["row"]], mk, im["style"], {}, sup, nb, budget=st,
                       w_max=1, gamma=cfg["gamma"], lam=0.0, n_alt=cfg["n_alt"],
                       device=device, force_width=1, force_steps=st, ichunk=n)
            dif = (np.where(mk, d, 0) != np.where(mk, s_["x"], 0))
            fk[f"L{era}_s{st}"] = {
                "identical_fill": bool(~dif.any()), "cost_solve": float(s_["cost"].mean()),
                "cost_decode": float(dcost),
                "cost_equal": bool(float(s_["cost"].mean()) == float(dcost)),
                "frac_items_differ": float((dif.sum(1) > 0).mean()), "nmask": nmask}
    out["F_fork_fidelity"] = fk
    print("[F] " + json.dumps(fk, indent=1))
    _ties = {}
    for era in (1, 2, 3):
        mk = era_masks(im, era)
        hole = np.where(mk, cfg["k"], C["codes"][im["row"]])
        with torch.no_grad():
            from canvas.plant.model import amp
            X = torch.as_tensor(hole, dtype=torch.long, device=device)
            with amp(device):
                lg = plant(X, torch.as_tensor(im["style"], dtype=torch.long, device=device))
            lpp = torch.log_softmax(lg.float(), -1)
            cf = lpp.max(-1).values.masked_fill(
                ~torch.as_tensor(mk, device=device), -1e9)
            top2 = cf.topk(2, -1).values
            _ties[f"L{era}"] = {"frac_items_with_tied_top_cell":
                                float((top2[:, 0] == top2[:, 1]).float().mean()),
                                "max_conf_median": float(top2[:, 0].median())}
    out["F_ties"] = _ties
    print("[F ties] " + json.dumps(_ties, indent=1))
    for key, v in fk.items():
        assert v["identical_fill"] and v["cost_equal"], f"F failed at {key}: {v}"

    # ---- P: the grade of record before the loop ---------------------------------------- #
    from PIL import Image
    cents = torch.as_tensor(C["cents"], device=device)
    P = {}
    for era in (1, 2, 3):
        mk = era_masks(im, era)
        P[f"clean_L{era}"] = float(grade_support(C["codes"][im["row"]], mk,
                                                 supH, supV)["pass"].mean())
    imgs, msks, stys = [], [], []
    for u in sel:
        s = styles[u]
        for r in C["meta"][s]["damage"]:
            if r["mode"] != "seam":
                continue
            p = os.path.join(C["croot"], s, "dmg", r["key"] + ".png")
            if not os.path.exists(p):
                continue
            r0, c0, r1, c1 = r["code_rect"]
            m = np.zeros((GRID, GRID), bool)
            m[max(0, r0):min(GRID, r1), max(0, c0):min(GRID, c1)] = True
            imgs.append(np.asarray(Image.open(p).convert("RGB")))
            msks.append(m.reshape(-1)); stys.append((u, r["level"]))
    if imgs:
        dc, _ = CB.encode(np.stack(imgs), cents, dev=device)
        lv = np.array([x[1] for x in stys])
        g = grade_support(dc.astype(np.int64), np.stack(msks), supH, supV)
        for era in (1, 2, 3):
            m = lv == era
            if m.any():
                P[f"seam_pass_L{era}"] = float(g["pass"][m].mean())
                P[f"seam_n_L{era}"] = int(m.sum())
                P[f"seam_unsup_L{era}"] = float(g["n_unsup"][m].mean())
                P[f"seam_unsup_cross_L{era}"] = float(g["n_unsup_cross"][m].mean())
                P[f"seam_touched_L{era}"] = float(g["n_touched"][m].mean())
    out["P_grader"] = P
    print("[P] " + json.dumps(P, indent=1))
    short = [f"L{e}" for e in (1, 2, 3)
             if P.get(f"seam_pass_L{e}", 0.0) > 0.10]
    out["P_seam_gate_shortfall"] = short
    if short:
        print(f"[P] GATE SHORTFALL at {short}: the support test rejects "
              + ", ".join(f"{e} {1 - P['seam_pass_' + e]:.3f}" for e in short)
              + " of `seam` (target 0.90). REPORTED, NOT TUNED -- min-count thresholds buy "
              "specificity by destroying coverage, so the lever is more exemplars. Era 1 is "
              "the bootstrapping era; the vocabulary claim lives at eras 2-3, where the gate "
              "holds.", flush=True)
    for era in (1, 2, 3):
        thr = 0.85 if era < 3 else 0.80
        assert P[f"clean_L{era}"] >= thr, f"P failed: clean L{era} = {P[f'clean_L{era}']}"
    for era in (2, 3):
        if f"seam_pass_L{era}" in P:
            assert P[f"seam_pass_L{era}"] <= 0.10, (
                f"P failed: seam L{era} pass {P[f'seam_pass_L{era}']:.3f}")
    print(json.dumps(out, indent=1, cls=NumpyEncoder))
    return out


@app.local_entrypoint()
def main(quick: int = 1):
    run.remote(tag="smoke0", quick=quick)
