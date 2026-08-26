"""Any-order (MaskGIT) decoding of a masked region, with a beam -- the primitive action space.

This is the cost side of the arc's meter. A completion of |M| cells at beam width w is the
w^|M|-shaped search `ratchet` priced; the sampler below makes the price explicit as
FORWARD PASSES = steps * width, so "completion quality vs mask size at a couple of widths" is
literally the cost-to-depth curve the memo's Sec.4 asks for. If quality at large |M| does not
improve with width, depth is unaffordable to the primitive action space -- which is the
precondition a macro would relieve.

All items in one call must share |M| (the reveal schedule is computed from it), so callers
loop over mask sizes; that is how the cost-to-depth table is built anyway.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from canvas.plant.model import amp


@torch.no_grad()
def decode(model, grids, masks, styles, steps=8, width=1, temp=1.0, seed=0, dev="cuda",
           chunk=48, use_style=True):
    """Fill masks[i] in grids[i]. Returns (filled grids, forward-pass cost per item)."""
    model.eval()
    grids = np.asarray(grids); masks = np.asarray(masks); styles = np.asarray(styles)
    nmask = int(masks[0].sum())
    assert (masks.sum(1) == nmask).all(), "group items by |M|"
    steps = min(steps, nmask)
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = np.array(grids, np.int64, copy=True)
    T = grids.shape[1]
    for a in range(0, len(grids), chunk):
        sl = slice(a, min(a + chunk, len(grids)))
        y = torch.as_tensor(grids[sl], dtype=torch.long, device=dev)
        m = torch.as_tensor(masks[sl], device=dev)
        s = torch.as_tensor(styles[sl], dtype=torch.long, device=dev)
        if not use_style:
            s = torch.full_like(s, model.n_styles)
        n = y.shape[0]
        X = torch.where(m, torch.full_like(y, model.K), y)[:, None].repeat(1, width, 1)
        M = m[:, None].repeat(1, width, 1).clone()
        LP = torch.full((n, width), -1e9, device=dev)
        LP[:, 0] = 0.0                          # only hypothesis 0 is alive at the start
        left = nmask
        for r in range(steps):
            k = int(math.ceil(left / (steps - r)))
            with amp(dev):
                logits = model(X.reshape(n * width, T), s.repeat_interleave(width))
            lp = F.log_softmax(logits.float(), -1).reshape(n, width, T, -1)
            K = lp.shape[-1]
            conf = lp.max(-1).values.masked_fill(~M, -1e9)
            sel = conf.topk(k, dim=-1).indices                          # (n,w,k)
            lp_sel = lp.gather(2, sel[..., None].expand(n, width, k, K))
            cand_c, cand_lp = [], []
            for b in range(width):
                if b == 0:
                    c = lp_sel.argmax(-1)
                    v = lp_sel.max(-1).values
                else:
                    p = F.softmax(lp_sel / temp, -1).reshape(-1, K)
                    c = torch.multinomial(p, 1, generator=gen).reshape(n, width, k)
                    v = lp_sel.gather(-1, c[..., None])[..., 0]
                cand_c.append(c); cand_lp.append(v.sum(-1))
            cand_c = torch.stack(cand_c, 2)                             # (n,w,B,k)
            score = LP[:, :, None] + torch.stack(cand_lp, 2)            # (n,w,B)
            flat = score.reshape(n, -1)
            top = flat.topk(min(width, flat.shape[1]), -1).indices      # (n,w)
            hyp, br = top // width, top % width
            bi = torch.arange(n, device=dev)[:, None]
            Xn = X[bi, hyp].clone(); Mn = M[bi, hyp].clone()
            sn = sel[bi, hyp]                                           # (n,w,k)
            cn = cand_c[bi, hyp, br]                                    # (n,w,k)
            Xn.scatter_(2, sn, cn)
            Mn.scatter_(2, sn, torch.zeros_like(cn, dtype=torch.bool))
            X, M, LP = Xn, Mn, flat.gather(1, top)
            left -= k
        out[sl] = X[:, 0].cpu().numpy()
    return out, steps * width
