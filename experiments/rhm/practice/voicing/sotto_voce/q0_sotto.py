"""[sotto] Q0 — the mirror, sized offline, before a GPU is spent.

WHAT THIS IS AND IS NOT. The spec asks for the outcome models to be fitted offline on banked
filed rows from `vo_s3b` so the committee's spread and the agreement threshold are sized before
a paid run. **The banked arm files do not carry them**: `voicing` logs `vo_buf` / `vo_probe_buf`
as SIZES and never persisted `fin` (that is exactly the field this node adds). So the corpus
here is a PROXY for the learner's own, built from the same world with the same corruption
operator and the same substitution the probe channel uses, and every number below is a
statement about the machinery and the shape of the problem — never about the run. The run's own
numbers come from `log["vo_om"]` and `vo_mirror`.

The proxy, stated so the gap is readable:

  EXPERIENCE   a mixture standing in for the practice beam's graded tips: a quarter untouched
               derivations of a drawn root (the beam solved it), 30% corrupted one block
               (`n_corrupt = 1`, the substrate's own operator) and rewritten at ONE macro span,
               the rest corrupted and rewritten at one or two. Graded by the world. The weights
               are chosen so the base rate lands near the run's measured filed solve rate of
               0.13-0.24, and nothing else about them is claimed.
  COUNTERFACTUAL  one further span at one level replaced by a DIFFERENT class's row — the probe
               channel's substitution exactly (`vo_run_probes`), graded by the world.

  THE GAP. The real learner's writes are drawn from ITS BOOK and are near-degenerate at the
  frontier (`voicing` Q0 finding 3: one token class covers ~100% of the writes at L5), while
  this proxy writes uniformly over the true table. So the proxy's support is WIDER than the
  learner's, which makes it an OPTIMISTIC read on the counterfactual half and a pessimistic one
  on the filed half. Said here rather than discovered at reduction time.

Sections:
  [S1] the corpus and its base rates
  [S2] the single outcome model: held-out accuracy / AUC on experience, and on counterfactuals
  [S3] the committee: spread on held-out experience, the threshold at each quantile, and what
       the agreement rule buys — accuracy on the rows it keeps against the rows it drops
  [S4] disagreement as a detector of the model's own error (AUC), which is `committee_head`'s
       0.716 asked one domain over
  [S5] the blind region: accuracy split by whether the substituted class is in the frequently
       written set, and by level

Usage (from experiments/):
    PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/q0_sotto.py
    PYTHONPATH=. python3 rhm/practice/voicing/sotto_voce/q0_sotto.py --n 40000 --steps 600
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))

from rhm.rhm_data import generate_rules_distinct                       # noqa: E402
from rhm.practice.ratchet import macros as MC                          # noqa: E402
from rhm.practice.voicing.sotto_voce import sotto_voce as V            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


def build_world(v=8, s=2, depth=6, m=2, rule_seed=0, maxl=5):
    rules = generate_rules_distinct(v, s, depth, m, seed=rule_seed)
    canon = np.ascontiguousarray(rules[depth - 1][:, 0, :])
    truth = MC.true_tables(rules, depth, s, v, m, maxl)
    return rules, canon, truth


def true_leaves(rules, roots, s, depth, rng):
    """One uniformly drawn derivation per root, top down through the rule table."""
    cur = roots[:, None].copy()
    for ell in range(depth):
        layer = rules[ell]                                   # (v, m, s)
        pick = rng.integers(0, layer.shape[1], size=cur.shape)
        cur = layer[cur, pick]                               # (B, n, s)
        cur = cur.reshape(cur.shape[0], -1)
    return cur


def make_corpus(rules, truth, canon, n, s, depth, v, m, maxl, rng, p_solved=0.25,
                p_repair=0.30):
    """The proxy experience corpus — a stand-in for the practice beam's own graded tips.

    A tip is a corrupted instance the beam has partly repaired, and `voicing` measured the
    filed solve rate at 0.13-0.24. So the mixture here is: `p_solved` untouched derivations
    (the beam solved it), `p_repair` corrupted-then-rewritten-at-one-macro-span (the beam tried
    one repair, which sometimes lands), and the rest corrupted with one or two further random
    macro writes (the beam wandered). The MIXTURE WEIGHTS ARE CHOSEN TO LAND THE BASE RATE
    NEAR THE RUN'S, and nothing else about them is claimed."""
    L = s ** depth
    n_blocks = L // s
    roots = rng.integers(0, v, size=n)
    x = true_leaves(rules, roots, s, depth, rng)
    levels = list(range(2, maxl + 1))

    def _write(b, ell=None):
        ell = int(ell if ell is not None else levels[rng.integers(0, len(levels))])
        span = s ** (ell - 1)
        node = int(rng.integers(0, s ** (depth - ell)))
        flat = truth[ell]["flat"]                              # (R, span) level-1 features
        row = flat[int(rng.integers(0, flat.shape[0]))]
        x[b, node * span * s:(node * span + span) * s] = canon[row].reshape(-1)

    for b in range(n):
        u = float(rng.random())
        if u < p_solved:
            continue
        blk = int(rng.integers(0, n_blocks))
        x[b, blk * s:(blk + 1) * s] = rng.integers(0, v, size=s)
        if u < p_solved + p_repair:
            _write(b)
        else:
            for _ in range(int(rng.integers(1, 3))):
                _write(b)
    succ, _ = V.grade(x, roots, rules, s)
    return x.astype(np.int64), roots.astype(np.int64), (succ > 0.5).astype("float32")


def make_counterfactuals(rules, truth, canon, x, roots, s, depth, v, maxl, rng, level=None):
    """The probe channel's substitution: one span at one level replaced by a different row of
    the true table, graded by the world. Returns the new x, the level, and the row's rank in a
    Zipf-like `frequently written` proxy (row index 0 is the class the proxy writes most)."""
    n, L = x.shape
    x2 = x.copy()
    lv = np.zeros(n, np.int64)
    rank = np.zeros(n, np.int64)
    levels = list(range(2, maxl + 1))
    for b in range(n):
        ell = int(level if level is not None else levels[rng.integers(0, len(levels))])
        n_nodes = s ** (depth - ell)
        node = int(rng.integers(0, n_nodes))
        flat = truth[ell]["flat"]
        r_ = int(rng.integers(0, flat.shape[0]))
        row = flat[r_]
        span = row.shape[0]
        blk0 = node * span
        x2[b, blk0 * s:(blk0 + span) * s] = canon[row].reshape(-1)
        lv[b] = ell
        rank[b] = r_
    succ, _ = V.grade(x2, roots, rules, s)
    return x2.astype(np.int64), lv, rank, (succ > 0.5).astype("float32")


def fit_bank(x, r, y, *, K, dim, steps, batch, lr, hold, boot, seed, v, L, s, device):
    cfg = dict(V._VO_DEFAULTS)
    cfg.update(vo_om_mode=("model" if K == 1 else "committee"), vo_om_k=K, vo_om_dim=dim,
               vo_om_steps=steps, vo_om_batch=batch, vo_om_lr=lr, vo_om_hold=hold,
               vo_om_boot=boot, vo_om_min=32, vo_om_cap=10 ** 9)
    bank = V.VoOutcomeBank(cfg, v, L, s, device, seed=seed)
    bank._append(torch.from_numpy(x), torch.from_numpy(r), torch.from_numpy(y), 0)
    bank.n_push_exp += int(x.shape[0])
    loss = bank.train()
    bank.refresh(cap=10 ** 9)
    return bank, loss


def acc_auc(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return (float(((p > 0.5).astype(float) == y).mean()), V.vo_auc(p, y), float(y.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--n-cf", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="so_q0_reduction.txt")
    a = ap.parse_args()

    v, s, depth, m, maxl = 8, 2, 6, 2, 5
    L = s ** depth
    dev = torch.device("cpu")
    rng = np.random.default_rng(a.seed)
    rules, canon, truth = build_world(v, s, depth, m, 0, maxl)
    out = []

    def say(t=""):
        print(t, flush=True)
        out.append(t)

    say("=" * 78)
    say("[sotto] Q0 — the mirror, sized offline. PROXY CORPUS: every number here is a")
    say("statement about the machinery and the shape of the problem, never about the run.")
    say("=" * 78)

    # ---- [S1] the corpus ------------------------------------------------------------- #
    x, r, y = make_corpus(rules, truth, canon, a.n, s, depth, v, m, maxl, rng)
    xc, lv, rank, yc = make_counterfactuals(rules, truth, canon, x, r, s, depth, v, maxl, rng)
    say("")
    say(f"[S1] corpus  n={a.n}  length={L}  v={v}  s={s}  depth={depth}  m={m}")
    say(f"     experience solve rate      {float(y.mean()):.4f}")
    say(f"     counterfactual solve rate  {float(yc.mean()):.4f}   "
        f"(ratio {float(y.mean()) / max(1e-9, float(yc.mean())):.2f}x; `voicing` measured "
        f"3.5-4x in run)")
    for ell in range(2, maxl + 1):
        mk = lv == ell
        if mk.sum():
            say(f"       L{ell}: n={int(mk.sum()):6d}  solve rate {float(yc[mk].mean()):.4f}")

    # ---- [S2] one outcome model ------------------------------------------------------ #
    bank1, loss1 = fit_bank(x, r, y, K=1, dim=a.dim, steps=a.steps, batch=a.batch, lr=a.lr,
                            hold=0.1, boot=1.0, seed=7, v=v, L=L, s=s, device=dev)
    hold = (bank1.h < bank1.hold).numpy()
    p_h, _, _ = bank1.predict(bank1.x[hold], bank1.r[hold])
    ac, au, br = acc_auc(p_h.numpy(), y[hold])
    p_c, _, _ = bank1.predict(torch.from_numpy(xc), torch.from_numpy(r))
    ac2, au2, br2 = acc_auc(p_c.numpy(), yc)
    say("")
    say(f"[S2] ONE outcome model  dim={a.dim} steps={a.steps} batch={a.batch} lr={a.lr}  "
        f"final loss {loss1:.4f}")
    say(f"     held-out EXPERIENCE     n={int(hold.sum()):6d}  acc {ac:.4f}  "
        f"AUC {au if au is None else round(au, 4)}  base {br:.4f}")
    say(f"     COUNTERFACTUAL (off-support)  n={len(yc):6d}  acc {ac2:.4f}  "
        f"AUC {au2 if au2 is None else round(au2, 4)}  base {br2:.4f}")

    # ---- [S3] the committee and the threshold ---------------------------------------- #
    bank, lossK = fit_bank(x, r, y, K=a.k, dim=a.dim, steps=a.steps, batch=a.batch, lr=a.lr,
                           hold=0.1, boot=0.8, seed=11, v=v, L=L, s=s, device=dev)
    hold = (bank.h < bank.hold).numpy()
    pm_h, sp_h, _ = bank.predict(bank.x[hold], bank.r[hold])
    pm_c, sp_c, _ = bank.predict(torch.from_numpy(xc), torch.from_numpy(r))
    ach, auh, brh = acc_auc(pm_h.numpy(), y[hold])
    acc_, aucc, brc = acc_auc(pm_c.numpy(), yc)
    say("")
    say(f"[S3] COMMITTEE K={a.k} boot=0.8  final loss {lossK:.4f}")
    say(f"     held-out EXPERIENCE     acc {ach:.4f}  AUC {round(auh, 4)}  "
        f"mean spread {float(sp_h.mean()):.4f}")
    say(f"     COUNTERFACTUAL          acc {acc_:.4f}  AUC {round(aucc, 4)}  "
        f"mean spread {float(sp_c.mean()):.4f}   "
        f"(spread ratio {float(sp_c.mean()) / max(1e-9, float(sp_h.mean())):.2f}x)")
    say("")
    say("     the agreement rule, at each quantile of the HELD-OUT EXPERIENCE spread:")
    say("      q      thr      filed frac   acc(filed)  acc(dropped)   AUC(filed)")
    sp_cn, pm_cn = sp_c.numpy(), pm_c.numpy()
    for q in (0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
        thr = float(np.quantile(sp_h.numpy(), q))
        keep = sp_cn <= thr
        a1 = float(((pm_cn[keep] > 0.5).astype(float) == yc[keep]).mean()) if keep.any() else float("nan")
        a0 = (float(((pm_cn[~keep] > 0.5).astype(float) == yc[~keep]).mean())
              if (~keep).any() else float("nan"))
        au_k = V.vo_auc(pm_cn[keep], yc[keep]) if keep.sum() > 4 else None
        say(f"     {q:4.2f}  {thr:7.4f}   {keep.mean():9.4f}   {a1:9.4f}   {a0:10.4f}   "
            f"{('n/a' if au_k is None else f'{au_k:.4f}')}")

    # ---- [S4] disagreement as a detector of the model's own error --------------------- #
    wrong_c = ((pm_cn > 0.5).astype("float32") != yc).astype("float32")
    det_c = V.vo_auc(sp_cn, wrong_c)
    sp_hn, pm_hn = sp_h.numpy(), pm_h.numpy()
    wrong_h = ((pm_hn > 0.5).astype("float32") != y[hold]).astype("float32")
    det_h = V.vo_auc(sp_hn, wrong_h)
    say("")
    say("[S4] does DISAGREEMENT find the model's own error?  (AUC of spread as a detector)")
    say(f"     on COUNTERFACTUALS   {('n/a' if det_c is None else f'{det_c:.4f}')}   "
        f"error rate {float(wrong_c.mean()):.4f}")
    say(f"     on held-out EXPERIENCE {('n/a' if det_h is None else f'{det_h:.4f}')}   "
        f"error rate {float(wrong_h.mean()):.4f}")
    say("     (`committee_head` read 0.716 for disagreement as a reducibility reader; this is")
    say("      the same question one domain over — is the spread sighted where the mean is not)")

    # ---- [S5] the blind region ------------------------------------------------------- #
    say("")
    say("[S5] WHERE the mirror is wrong, by level and by the substituted row's frequency rank")
    say("      level      n     solve   acc      AUC      spread   det(spread)")
    for ell in range(2, maxl + 1):
        mk = lv == ell
        if mk.sum() < 32:
            continue
        a_, u_, b_ = acc_auc(pm_cn[mk], yc[mk])
        d_ = V.vo_auc(sp_cn[mk], wrong_c[mk])
        say(f"       L{ell}  {int(mk.sum()):7d}  {b_:.4f}  {a_:.4f}  "
            f"{('n/a' if u_ is None else f'{u_:.4f}')}  {float(sp_cn[mk].mean()):.4f}  "
            f"{('n/a' if d_ is None else f'{d_:.4f}')}")
    say("")
    say("      the proxy's `frequently written` axis: the substituted row's index in the true")
    say("      table, bucketed. The real run's axis is the learner's OWN write share at the")
    say("      slot (>= vo_om_freq_share), which this corpus cannot have.")
    for lo, hi, nm in ((0, 4, "rank<4"), (4, 32, "4<=rank<32"), (32, 10 ** 9, "rank>=32")):
        mk = (rank >= lo) & (rank < hi)
        if mk.sum() < 32:
            continue
        a_, u_, b_ = acc_auc(pm_cn[mk], yc[mk])
        say(f"       {nm:12s} n={int(mk.sum()):6d}  solve {b_:.4f}  acc {a_:.4f}  "
            f"AUC {('n/a' if u_ is None else f'{u_:.4f}')}  spread {float(sp_cn[mk].mean()):.4f}")

    say("")
    say("=" * 78)
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, a.out), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\n[q0] written to figures/{a.out}")


if __name__ == "__main__":
    main()
