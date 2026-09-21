"""[tessitura] THE CPU PASS — the judge's own level, read off the banked rows and heads.

`reduce_logged.py` establishes that the norm is not in the log. It IS recoverable, statically,
from the two arms that banked `vo_heads.pt` beside `vo_rows.npz`: reload the trunk and the
critic, score every dumped row, and the judge's predicted P(solve) is in hand for ~425k rows per
arm. Two limits, stated before any number and repeated at every table:

  THE ROWS ARE THE RING-BUFFER TAIL. 8192 rows per slot per buffer, which at this arm's inflow
  is roughly the last 60 filed cycles and the last 128 probe cycles. Era 1 is gone.
  THE HEADS ARE THE FINAL ONES. This is one critic, at one time, read on rows it was trained
  online across. Nothing here is a time course, and the within-arm era question — the one the
  frozen trunk could not ask — is not answerable from the bank.

Sections:
  [T] the gates. T-2 reproduces the run's own last-cycle audit AUC from the reloaded heads;
      T-3 is its falsification.
  [N] THE NORM. Three derivations of a state value from a Q-shaped judge (at the write it made,
      max over the slot's realised candidate set, mean over it), each against the world's own
      base rate on the identical rows, per slot and pooled; the reliability curve; and the
      shift-versus-rescaling regression in norm §1's form.
  [C] COST AT MATCHED PRIOR. Does the judge separate solved from unsolved at matched DP prior,
      and how does that compare with the prior's own reading — striatum §2 and §5's conditional
      form, with the DP score in the role the model's surprisal played there. The STRUCTURAL
      half of this question is not here: see `structure.py`, gate T-4, which fails.
  [X] THE TWINS. Every probe row is a filed row with one class substituted at an identical
      context, both graded on the same trajectory's final configuration. At matched prior, how
      does the judge price the substituted class against the written one, and does the
      difference depend on what it expected before the write.
  [R] ACROSS ARMS. The two banked critics scored on one another's rows — norm §1's protocol,
      on identical rows, with the caveat that each critic carries its own trunk.

Usage (from experiments/):
    python3 rhm/practice/voicing/tessitura/reduce_rows.py
    python3 rhm/practice/voicing/tessitura/reduce_rows.py --arms ov_s0b:ovt_comp_pr_sh --no-cross
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOICING = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(os.path.join(VOICING, "..", "..", "..")))

FIG = os.path.join(VOICING, "figures")

# the arms that banked heads. `ov_s0`'s three arms banked rows but not heads (the heads were
# added for `ov_s0b`), so the judge's level is recoverable at exactly these two.
BANKED = [("ov_s0b", "ovt_comp_pr_sh", 0, "filed+probe (uniform), world-graded"),
          ("ov_s2", "ovt_comp_pr_dis", 2, "filed+probe (3/4 disagreement), world-graded")]


# ------------------------------------------------------------------ small statistics --- #

def auc(x, y):
    """the rank identity with ties averaged — `voicing.py::vo_auc` on numpy alone."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1, n0 = float((y > 0.5).sum()), float((y <= 0.5).sum())
    if n1 < 1 or n0 < 1:
        return None
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[o] = np.arange(1, len(x) + 1)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cond_auc(x, y, by, q=5, nmin=32):
    """AUC of `x` against `y` inside quantile bins of `by`, pooled by pair count — the
    'conditional AUC' of striatum §5, which is how that node held a second score fixed."""
    by = np.asarray(by, np.float64)
    ok = np.isfinite(by) & np.isfinite(np.asarray(x, np.float64))
    if int(ok.sum()) < nmin * 2:
        return None, 0
    edges = np.quantile(by[ok], np.linspace(0, 1, int(q) + 1))
    edges[0] -= 1e-9
    num = den = 0.0
    n_used = 0
    for k in range(int(q)):
        m = ok & (by > edges[k]) & (by <= edges[k + 1])
        if int(m.sum()) < nmin:
            continue
        a = auc(np.asarray(x)[m], np.asarray(y)[m])
        if a is None:
            continue
        w = float((np.asarray(y)[m] > 0.5).sum()) * float((np.asarray(y)[m] <= 0.5).sum())
        num += a * w
        den += w
        n_used += int(m.sum())
    return ((num / den) if den > 0 else None), n_used


def ols(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return None
    x, y = x[m], y[m]
    sx, sy = x.std(), y.std()
    if sx < 1e-12:
        return None
    b = float(np.cov(x, y, bias=True)[0, 1] / (sx ** 2))
    a = float(y.mean() - b * x.mean())
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope": b, "intercept": a, "r2": r * r, "r": r, "n": int(m.sum()),
            "sd_ratio": float(sy / sx)}


def f(x, nd=3):
    return ("  -  " if x is None or (isinstance(x, float) and not np.isfinite(x))
            else f"{x:.{nd}f}")


# ------------------------------------------------------------------- the arm's state --- #

class Arm:
    """One banked arm: its rows, its heads, and the critic's score for every row."""

    def __init__(self, tag, arm, seed, label):
        import torch
        import rhm.rhm_generative_planner as GP
        import rhm.practice.native.span.span_net as SN
        from rhm.practice.voicing.voicing import build_critic

        self.tag, self.arm, self.seed, self.label = tag, arm, seed, label
        self.root = os.path.join(FIG, tag, arm)
        self.z = np.load(os.path.join(self.root, "vo_rows.npz"))
        self.meta = json.load(open(os.path.join(self.root, "vo_rows_meta.json")))
        blob = torch.load(os.path.join(self.root, "vo_heads.pt"), map_location="cpu",
                          weights_only=True)
        cfgh = blob["cfg"]
        self.cfg_run = json.load(open(os.path.join(self.root, "results.json")))["config"]
        self.log = None                      # loaded lazily; results.json is 25 MB
        self.v, self.s, self.depth = int(cfgh["v"]), int(cfgh["s"]), int(cfgh["depth"])
        self.dim, self.maxl = int(cfgh["state_dim"]), int(cfgh["max_macro_level"])
        self.thr = int(round(float(cfgh["vo_critic_hold"]) * 100))
        length = self.s ** self.depth
        self.core = GP._build_generator()(self.v, length, self.s, self.dim, n_head=4,
                                          n_layer=2, root_conditioned=False)
        self.core.load_state_dict(blob["core"])
        self.core.eval()
        hm = int(cfgh.get("ov_critic_hidden", -1))
        self.critic = build_critic(SN.slot_count(self.s, self.depth, self.maxl), self.v,
                                   self.dim, self.s ** (self.maxl - 1), seed=0,
                                   device=torch.device("cpu"),
                                   hidden_mult=(int(cfgh["span_hidden_mult"]) if hm < 0 else hm))
        self.critic.load_state_dict(blob["critic"])
        self.critic.eval()
        self.SN, self.torch = SN, torch

    def slots(self):
        return sorted({k.split(":", 1)[1] for k in self.meta},
                      key=lambda k: (int(k.split(":")[0]), int(k.split(":")[1])))

    def cols(self, pre):
        z, out = self.z, {}
        for suf in ("obs", "write", "y", "dp", "code", "unif"):
            k = f"{pre}|{suf}"
            if k in z:
                out[suf] = z[k]
        return out

    def candidate_set(self, slot):
        """The slot's REALISED candidate set: every distinct tuple that appears as a filed
        write or a probe substitution. The dump does not carry the operative table, so this
        stands in for it, and it is the set the executor demonstrably had."""
        seen = set()
        for which in ("filed", "probe"):
            k = f"{which}:{slot}|write"
            if k in self.z:
                for r in self.z[k]:
                    seen.add(tuple(int(x) for x in r))
        return sorted(seen)

    def score(self, pre, critic=None, core=None, batch=1024):
        """Every row of one buffer, scored by a critic through a trunk (this arm's by default).

        ONE TRUNK FORWARD PER DISTINCT CONTEXT. The buffers hold the same masked context many
        times over (one per surviving beam row), so deduplicating is a 3-7x saving and changes
        nothing: the critic is a function of (context, candidate).
        """
        torch = self.torch
        critic = critic if critic is not None else self.critic
        core = core if core is not None else self.core
        c = self.cols(pre)
        mt = self.meta[pre]
        blk0, span, sid_i = int(mt["blk0"]), int(mt["span"]), int(mt["slot_id"])
        obs = c["obs"].astype(np.int64)
        # dedupe contexts
        key = np.ascontiguousarray(obs).view([('', obs.dtype)] * obs.shape[1]).ravel()
        uniq, inv = np.unique(key, return_inverse=True)
        u_rows = np.zeros(len(uniq), np.int64)
        seen = {}
        for i, q in enumerate(inv):
            if q not in seen:
                seen[q] = i
        u_rows = np.array([seen[q] for q in range(len(uniq))], np.int64)
        ob_u = torch.from_numpy(obs[u_rows])
        us = []
        with torch.no_grad():
            for a in range(0, ob_u.shape[0], batch):
                p, _ = self.SN.trunk(core, ob_u[a:a + batch])
                sid = torch.full((p.shape[0],), sid_i, dtype=torch.long)
                us.append(critic.ctx_state(p, blk0, span, sid))
        u = torch.cat(us)                                            # (n_u, dim)
        cand = torch.from_numpy(c["write"].astype(np.int64))
        with torch.no_grad():
            e = critic.cand_state(cand, span)                        # (n, dim)
            lg = critic.mlp(u[torch.from_numpy(inv)] + e).squeeze(-1).numpy().astype(np.float64)
            # the slot's realised candidate set, on the same contexts
            cs = self.candidate_set(pre.split(":", 1)[1])
            ct = torch.tensor(cs, dtype=torch.long)
            ec = critic.cand_state(ct, span)                         # (R, dim)
            allc = critic.mlp(u[:, None, :] + ec[None]).squeeze(-1)   # (n_u, R)
            pu = torch.sigmoid(allc)
            vmax_u = pu.max(1).values.numpy().astype(np.float64)
            vmean_u = pu.mean(1).numpy().astype(np.float64)
        return {"logit": lg, "p": 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30))),
                "y": c["y"].astype(np.float64), "dp": c["dp"].astype(np.float64),
                "code": c["code"].astype(np.int64),
                "unif": (c["unif"].astype(np.float64) if "unif" in c else None),
                "obs_id": inv.astype(np.int64), "write": c["write"].astype(np.int64),
                "vmax": vmax_u[inv], "vmean": vmean_u[inv],
                "n_cand": len(cs), "level": int(mt["level"]), "node": int(mt["node"])}

    def audit_last(self):
        """The final cycle's per-slot audit record, for gate T-2."""
        if self.log is None:
            self.log = json.load(open(os.path.join(self.root, "results.json")))["log"]["vo"]
        for c in reversed(self.log):
            if (c or {}).get("critic"):
                return c["critic"]
        return {}


# ---------------------------------------------------------------------------- report --- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--no-cross", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "rows.txt"))
    a = ap.parse_args()
    want = {x for x in a.arms.split(",") if x}
    lines = []

    def o(t=""):
        lines.append(t)
        print(t, flush=True)

    o("=" * 104)
    o("[tessitura] THE CPU PASS — the judge's level, from the banked rows and heads")
    o("=" * 104)
    o("  Two arms banked `vo_heads.pt`; `ov_s0`'s three banked rows only. Every number below is")
    o("  ONE critic (the final one) on the ring-buffer TAIL (~60 filed / ~128 probe cycles).")
    o("  It is a static read of a reader that was trained online, and it is not a time course.")
    o("")

    arms = []
    for tag, arm, seed, label in BANKED:
        if want and f"{tag}:{arm}" not in want:
            continue
        p = os.path.join(FIG, tag, arm, "vo_heads.pt")
        if not os.path.isfile(p):
            o(f"    {tag}:{arm}: no vo_heads.pt on disk — skipped")
            continue
        arms.append(Arm(tag, arm, seed, label))

    S = {}
    for A in arms:
        S[A.tag] = {pre: A.score(pre) for pre in sorted(A.meta)}
        o(f"    loaded {A.tag}:{A.arm} (seed {A.seed}) — {len(A.meta)} buffers, "
          f"{sum(len(S[A.tag][k]['y']) for k in S[A.tag])} rows, {A.label}")
    o("")

    # ------------------------------------------------------------------- [T] gates --- #
    o("-" * 104)
    o("[T] THE GATES")
    o("-" * 104)
    o("  T-2  THE DUMPED BUFFER IS THE BUFFER THE RUN'S LAST AUDIT READ. `vo_critic_audit`")
    o("       scores, once per cycle, the rows with `code < 100*vo_critic_hold`, last 1024, and")
    o("       logs `n` and `base_rate` (and `probe_n` / `probe_base_rate`). At the FINAL cycle")
    o("       those rows are exactly this dump's held-out slice, so both must come back EXACT.")
    o("       This is the gate on the rows and on the hold code, and it is an identity.")
    for A in arms:
        aud = A.audit_last()
        worst_n, worst_b, n_cmp, worst_k = 0, 0.0, 0, None
        for slot, rec in aud.items():
            for which, kn, kb in (("filed", "n", "base_rate"),
                                  ("probe", "probe_n", "probe_base_rate")):
                pre = f"{which}:{slot}"
                if pre not in S[A.tag] or rec.get(kb) is None:
                    continue
                d = S[A.tag][pre]
                hold = np.nonzero(d["code"] < A.thr)[0][-1024:]
                if len(hold) < 16:
                    continue
                n_cmp += 1
                dn = abs(int(len(hold)) - int(rec[kn]))
                db = abs(float(d["y"][hold].mean()) - float(rec[kb]))
                if dn > worst_n or db > worst_b:
                    worst_k = f"{pre}: n {len(hold)} vs {rec[kn]}, base {d['y'][hold].mean():.6f} vs {float(rec[kb]):.6f}"
                worst_n, worst_b = max(worst_n, dn), max(worst_b, db)
        # the log stores `float(y_b[hold].mean())` of a float32 tensor; this sums in
        # float64, so the base rate agrees to float32 and not to float64.
        ok = (worst_n == 0 and worst_b < 1e-6 and n_cmp >= 20)
        o(f"       {A.tag}:{A.arm}  {n_cmp} buffers, max |dn| = {worst_n}, max |dbase| = "
          f"{worst_b:.2e}  [{'PASS' if ok else 'FAIL'}]" + (f"  worst: {worst_k}" if not ok else ""))
        assert ok, f"T-2 failed for {A.tag}:{A.arm}"
    o("       FALSIFIED by dropping the last held-out row of every buffer:")
    for A in arms:
        aud = A.audit_last()
        bad = 0
        for slot, rec in list(aud.items())[:6]:
            pre = f"filed:{slot}"
            if pre not in S[A.tag] or rec.get("base_rate") is None:
                continue
            d = S[A.tag][pre]
            hold = np.nonzero(d["code"] < A.thr)[0][-1024:][:-1]
            if abs(int(len(hold)) - int(rec["n"])) != 0:
                bad += 1
        o(f"       {A.tag}:{A.arm}  {bad} of 6 buffers now disagree on `n`  "
          f"[{'FAILS AS REQUIRED' if bad == 6 else 'VACUOUS'}]")
        assert bad == 6, f"T-2 falsification vacuous for {A.tag}:{A.arm}"
    o("")
    o("  T-3  THE SCORES CANNOT BE AN IDENTITY, AND THE REASON IS IN THE LOOP'S ORDER.")
    o("       `vo_critic_audit` is called at `voicing.py:9729` and `finetune_generator_span` at")
    o("       `:10010`, so the audit of cycle c is read BEFORE cycle c's optimizer step, while")
    o("       `ov_dump_rows` runs after the loop. The banked core and critic are therefore one")
    o("       step ahead of the last logged audit, and the reloaded AUC must be CLOSE to the")
    o("       logged one without equalling it. Reported, not asserted to zero:")
    for A in arms:
        aud = A.audit_last()
        res = []
        for slot, rec in aud.items():
            for which, key in (("filed", "auc"), ("probe", "probe_auc")):
                pre = f"{which}:{slot}"
                if pre not in S[A.tag] or rec.get(key) is None:
                    continue
                d = S[A.tag][pre]
                hold = np.nonzero(d["code"] < A.thr)[0][-1024:]
                got = auc(d["logit"][hold], d["y"][hold])
                if got is not None:
                    res.append((abs(got - float(rec[key])), which))
        fil = [r[0] for r in res if r[1] == "filed"]
        prb = [r[0] for r in res if r[1] == "probe"]
        o(f"       {A.tag}:{A.arm}  |dAUC| filed: median {np.median(fil):.4f} max "
          f"{max(fil):.4f} (n {len(fil)}) · probe: median {np.median(prb):.4f} max "
          f"{max(prb):.4f} (n {len(prb)})")
        assert max(fil + prb) < 0.12, f"T-3: the reloaded critic is not the run's"
    o("       FALSIFICATION — the same comparison with the critic's parameters RE-DRAWN from")
    o("       `build_critic`'s own minting discipline at another seed. The criterion is")
    o("       RELATIVE, because a random readout sits at chance and the logged AUCs are ~0.65,")
    o("       so an absolute threshold would only be restating the base rate: the re-drawn")
    o("       residual must exceed the reloaded one by at least 3x on the SAME buffers, and the")
    o("       re-drawn AUC itself must sit at chance.")
    for A in arms:
        import torch
        from rhm.practice.voicing.voicing import build_critic
        bad_c = build_critic(A.SN.slot_count(A.s, A.depth, A.maxl), A.v, A.dim,
                             A.s ** (A.maxl - 1), seed=987654, device=torch.device("cpu"),
                             hidden_mult=4)
        bad_c.eval()
        aud = A.audit_last()
        got_r, got_g, got_auc = [], [], []
        for slot, rec in list(aud.items())[:6]:
            pre = f"filed:{slot}"
            if pre not in S[A.tag] or rec.get("auc") is None:
                continue
            d = A.score(pre, critic=bad_c)
            hold = np.nonzero(d["code"] < A.thr)[0][-1024:]
            g = auc(d["logit"][hold], d["y"][hold])
            gg = auc(S[A.tag][pre]["logit"][hold], S[A.tag][pre]["y"][hold])
            if g is not None and gg is not None:
                got_r.append(abs(g - float(rec["auc"])))
                got_g.append(abs(gg - float(rec["auc"])))
                got_auc.append(g)
        ratio = float(np.median(got_r) / max(np.median(got_g), 1e-9))
        chance = float(np.abs(np.array(got_auc) - 0.5).max())
        ok = ratio >= 3.0 and chance < 0.10
        o(f"       {A.tag}:{A.arm}  re-drawn |dAUC| median {np.median(got_r):.4f} vs reloaded "
          f"{np.median(got_g):.4f} ({ratio:.1f}x) · re-drawn AUC max |x-0.5| = {chance:.3f}  "
          f"[{'FAILS AS REQUIRED' if ok else 'VACUOUS'}]")
        assert ok, f"T-3 falsification vacuous for {A.tag}:{A.arm}"
    o("")
    o("  T-5  THE DEDUPE IS FREE. `Arm.score` runs ONE trunk forward per distinct masked")
    o("       context and fans the result back over the rows that share it. Against the naive")
    o("       per-row path on one buffer that must be an identity.")
    for A in arms:
        import torch
        pre = f"filed:{A.slots()[0]}"
        d = S[A.tag][pre]
        mt = A.meta[pre]
        ob = torch.from_numpy(A.z[f"{pre}|obs"].astype(np.int64))[:2048]
        wb = torch.from_numpy(A.z[f"{pre}|write"].astype(np.int64))[:2048]
        with torch.no_grad():
            pl, _ = A.SN.trunk(A.core, ob)
            sid = torch.full((ob.shape[0],), int(mt["slot_id"]), dtype=torch.long)
            naive = A.critic(pl, int(mt["blk0"]), int(mt["span"]), sid, wb).numpy()
        dd = float(np.abs(naive - d["logit"][:2048]).max())
        o(f"       {A.tag}:{A.arm}  {pre}  max |dlogit| = {dd:.3e}  "
          f"[{'PASS' if dd < 1e-5 else 'FAIL'}]")
        assert dd < 1e-5, f"T-5 failed for {A.tag}:{A.arm}"
    o("")
    o("  T-4  THE STRUCTURAL LABEL IS NOT RECOVERABLE OFFLINE — this gate FAILS, and its")
    o("       failure is the finding. `structure.py` reconstructs the root the dump does not")
    o("       carry, by the min-edit cost to each root with the slot's span left free, and")
    o("       reads `consistent_features` under it. The reconstruction is reported in")
    o("       `results/structure.txt`; it is ambiguous on 47-68% of rows (two roots explain the")
    o("       string equally well) and BIASED UPWARD, because a root under which some write")
    o("       repairs is exactly the root that wins a free-span argmin. Q2's structural half")
    o("       therefore needs the label banked at the write, which is what the re-run adds.")
    o("")

    # -------------------------------------------------------------------- [N] norm --- #
    o("-" * 104)
    o("[N] THE NORM — the judge's own expectation, against the world on the identical rows")
    o("-" * 104)
    o("  THE DERIVATION IS A CHOICE AND IS REPORTED AS THREE. The judge is Q-shaped,")
    o("  `critic(context, candidate) -> P(solve)`; a state value has to be built from it.")
    o("    V_w    the judge's probability at the candidate that WAS written — the expectation")
    o("           at the write it will make, and the only one of the three the run itself ever")
    o("           consults on a governed slot;")
    o("    V_max  its max over the slot's realised candidate set — the value of the slot under")
    o("           a judge-optimal write;")
    o("    V_mean its mean over that set — the value of the slot under a blind write.")
    o("  `base` is the mean verdict over the SAME rows. On filed rows V_w is the judge's")
    o("  expectation of what it chose; on probe rows it is its expectation of a substitution")
    o("  drawn off-policy, so the two are different objects and never pooled.")
    o("")
    for A in arms:
        o(f"  {A.tag}:{A.arm} (seed {A.seed})")
        for which in ("filed", "probe"):
            o(f"    {which}")
            o(f"      {'slot':6} {'n':>7} {'base':>7} {'V_w':>7} {'V_w-base':>9} "
              f"{'V_max':>7} {'V_mean':>7} {'nC':>4} {'AUC':>7}")
            xs, ys = [], []
            for slot in A.slots():
                pre = f"{which}:{slot}"
                if pre not in S[A.tag]:
                    continue
                d = S[A.tag][pre]
                b, vw = float(d["y"].mean()), float(d["p"].mean())
                xs.append(b)
                ys.append(vw)
                o(f"      {slot:6} {len(d['y']):>7} {b:>7.3f} {vw:>7.3f} {vw-b:>+9.3f} "
                  f"{float(d['vmax'].mean()):>7.3f} {float(d['vmean'].mean()):>7.3f} "
                  f"{d['n_cand']:>4} {f(auc(d['logit'], d['y'])):>7}")
            r = ols(xs, ys)
            o(f"      REGRESSION of the norm on the world across slots: "
              + (f"slope {r['slope']:+.3f}  intercept {r['intercept']:+.3f}  R2 {r['r2']:.3f}  "
                 f"sd ratio {r['sd_ratio']:.3f}  n {r['n']}" if r else "-"))
            o(f"      POOLED: base {np.mean(xs):.3f}   V_w {np.mean(ys):.3f}   "
              f"bias {np.mean(ys)-np.mean(xs):+.3f}")
            o("")
        # reliability, pooled over slots
        o("    RELIABILITY (rows pooled over slots, deciles of V_w):")
        for which in ("filed", "probe"):
            p = np.concatenate([S[A.tag][f"{which}:{k}"]["p"] for k in A.slots()
                                if f"{which}:{k}" in S[A.tag]])
            y = np.concatenate([S[A.tag][f"{which}:{k}"]["y"] for k in A.slots()
                                if f"{which}:{k}" in S[A.tag]])
            ed = np.quantile(p, np.linspace(0, 1, 11))
            ed[0] -= 1e-9
            cells = []
            for k in range(10):
                mm = (p > ed[k]) & (p <= ed[k + 1])
                cells.append(f"{p[mm].mean():.2f}/{y[mm].mean():.2f}" if mm.any() else "-")
            o(f"      {which:6} (pred/obs) " + "  ".join(cells))
        o("")

    # ------------------------------------------------- [C] cost at matched prior --- #
    o("-" * 104)
    o("[C] COST AT MATCHED PRIOR — striatum §2/§5 with the DP prior where surprisal stood")
    o("-" * 104)
    o("  `crit` is the judge's logit, `dp` the surface model's own score of the same candidate")
    o("  (the prior the composed chooser sums), both against the verdict. `crit | dp` is the")
    o("  judge's AUC inside quintile bins of the prior and back; a readout that carried nothing")
    o("  the other does not would fall to chance under conditioning.")
    o("  THE STRUCTURAL HALF IS ABSENT: see `structure.py` and gate T-4 below.")
    o("")
    for A in arms:
        o(f"  {A.tag}:{A.arm} (seed {A.seed})")
        o(f"    {'rows':8} {'n':>8} {'base':>7} {'crit':>7} {'dp':>7} {'crit|dp':>8} "
          f"{'dp|crit':>8}")
        for which in ("filed", "probe"):
            ks = [k for k in A.slots() if f"{which}:{k}" in S[A.tag]]
            lg = np.concatenate([S[A.tag][f"{which}:{k}"]["logit"] for k in ks])
            dp = np.concatenate([S[A.tag][f"{which}:{k}"]["dp"] for k in ks])
            y = np.concatenate([S[A.tag][f"{which}:{k}"]["y"] for k in ks])
            # per slot, then pooled by pair count: pooling raw scores across slots would rank
            # slot membership (the analyzer's own lesson, `ov_oof_auc`'s defect #5).
            def perslot(x_name):
                num = den = 0.0
                for k in ks:
                    d = S[A.tag][f"{which}:{k}"]
                    xx = d["logit"] if x_name == "crit" else d["dp"]
                    aa = auc(xx, d["y"])
                    if aa is None:
                        continue
                    w = float((d["y"] > 0.5).sum()) * float((d["y"] <= 0.5).sum())
                    num += aa * w
                    den += w
                return (num / den) if den else None

            def perslot_cond(x_name, by_name):
                num = den = 0.0
                for k in ks:
                    d = S[A.tag][f"{which}:{k}"]
                    xx = d["logit"] if x_name == "crit" else d["dp"]
                    bb = d["logit"] if by_name == "crit" else d["dp"]
                    aa, _n = cond_auc(xx, d["y"], bb)
                    if aa is None:
                        continue
                    w = float((d["y"] > 0.5).sum()) * float((d["y"] <= 0.5).sum())
                    num += aa * w
                    den += w
                return (num / den) if den else None
            o(f"    {which:8} {len(y):>8} {y.mean():>7.3f} {f(perslot('crit')):>7} "
              f"{f(perslot('dp')):>7} {f(perslot_cond('crit','dp')):>8} "
              f"{f(perslot_cond('dp','crit')):>8}")
        o("")

    # ------------------------------------------------------------------ [X] twins --- #
    o("-" * 104)
    o("[X] THE TWINS — the substituted class against the written one at an identical context")
    o("-" * 104)
    o("  Every probe row was drawn from a filed row of the same cycle: the same masked context,")
    o("  a DIFFERENT on-table class (gate V-4d), and a verdict on the SAME trajectory's final")
    o("  configuration with that one slot's content swapped. So the pair is a same-prefix twin")
    o("  by construction, in the outcome as well as in the context — which is more than norm's")
    o("  twins had, where the twin's continuation belonged to the other token.")
    o("  THE JOIN IS ON THE CONTEXT. A masked context recurs across beam rows and cycles, so a")
    o("  context is a GROUP: `y_f` is the mean verdict of the filed rows carrying it and `y_p`")
    o("  the mean of the probe rows. `split` is the share of contexts whose filed verdicts")
    o("  disagree with each other, which is the pairing's own noise and is printed, not hidden.")
    o("  The filed buffer is shorter than the probe buffer in cycles, so only the overlap pairs.")
    o("")
    for A in arms:
        o(f"  {A.tag}:{A.arm} (seed {A.seed})")
        o(f"    {'slot':6} {'ctx':>6} {'pairs':>7} {'split':>6} {'y_f':>6} {'y_p':>6} "
          f"{'dY':>7} {'dQ':>8} {'dQ|dDP':>8} {'dP':>7} {'dDP':>7} {'sign':>6} "
          f"{'b(dQ~Vm)':>9} {'b(dQ~Vw)':>9}")
        tot = collections.defaultdict(list)
        for slot in A.slots():
            fp, pp = f"filed:{slot}", f"probe:{slot}"
            if fp not in S[A.tag] or pp not in S[A.tag]:
                continue
            F, P = S[A.tag][fp], S[A.tag][pp]
            # contexts are identified by the raw obs bytes: one joint `np.unique` over the two
            # buffers gives both a common id, which is the whole join.
            obf = np.ascontiguousarray(A.z[f"{fp}|obs"])
            obp = np.ascontiguousarray(A.z[f"{pp}|obs"])
            both = np.concatenate([obf, obp])
            view = both.view([("", both.dtype)] * both.shape[1]).ravel()
            _, ids = np.unique(view, return_inverse=True)
            idf, idp = ids[:len(obf)], ids[len(obf):]
            fy = collections.defaultdict(list)
            fq = collections.defaultdict(list)
            fdp = collections.defaultdict(list)
            fpr = collections.defaultdict(list)
            fvm = {}
            for i_, k in enumerate(idf):
                fy[k].append(F["y"][i_])
                fq[k].append(F["logit"][i_])
                fdp[k].append(F["dp"][i_])
                fpr[k].append(F["p"][i_])
                fvm[k] = (F["vmax"][i_], F["vmean"][i_])
            rows = []
            split = 0
            for i_, k in enumerate(idp):
                g = fy.get(k)
                if g is None:
                    continue
                yf = float(np.mean(g))
                if 0.0 < yf < 1.0:
                    split += 1
                rows.append((yf, float(P["y"][i_]),
                             float(P["logit"][i_]) - float(np.mean(fq[k])),
                             float(P["p"][i_]) - float(np.mean(fpr[k])),
                             float(P["dp"][i_]) - float(np.nanmean(fdp[k])),
                             float(np.mean(fpr[k])), float(fvm[k][1])))
            if len(rows) < 64:
                continue
            R = np.array(rows, np.float64)
            dY = R[:, 1] - R[:, 0]
            # `dQ` inside quintiles of the prior difference — the object is a CONTRAST, so the
            # matched read is a mean inside the bin, not a ranking. A row whose candidate the
            # FINAL operative table no longer holds has no DP score, so the matched read is
            # taken over the rows that do and its `n` shrinks accordingly.
            fin = np.isfinite(R[:, 4])
            dq_m = None
            if int(fin.sum()) > 64:
                ed = np.quantile(R[fin, 4], np.linspace(0, 1, 6))
                ed[0] -= 1e-9
                mids = [float(R[fin][(R[fin, 4] > ed[k]) & (R[fin, 4] <= ed[k + 1]), 2].mean())
                        for k in range(5)
                        if ((R[fin, 4] > ed[k]) & (R[fin, 4] <= ed[k + 1])).sum() > 8]
                dq_m = float(np.mean(mids)) if mids else None
            # does the judge get the SIGN of the world's own twin contrast right, on the pairs
            # where the world moved at all?
            mv = np.abs(dY) > 1e-9
            sign = (float((np.sign(R[mv, 2]) == np.sign(dY[mv])).mean())
                    if int(mv.sum()) > 32 else None)
            rgm = ols(R[:, 6], R[:, 2])            # V_mean: context-level, candidate-blind
            rgw = ols(R[:, 5], R[:, 2])            # V_w: the CONFOUNDED comparator
            o(f"    {slot:6} {len(set(idp.tolist())):>6} {len(R):>7} {split/len(R):>6.3f} "
              f"{R[:,0].mean():>6.3f} {R[:,1].mean():>6.3f} {dY.mean():>+7.3f} "
              f"{R[:,2].mean():>+8.3f} {f(dq_m):>8} {R[:,3].mean():>+7.3f} "
              f"{np.nanmean(R[:,4]):>+7.3f} {f(sign):>6} "
              f"{(f'{rgm[chr(0x73)+chr(0x6c)+chr(0x6f)+chr(0x70)+chr(0x65)]:+.3f}' if rgm else '  -  '):>9} "
              f"{(f'{rgw[chr(0x73)+chr(0x6c)+chr(0x6f)+chr(0x70)+chr(0x65)]:+.3f}' if rgw else '  -  '):>9}")
            for nm, col in (("yf", 0), ("yp", 1), ("dQ", 2), ("dP", 3), ("dDP", 4),
                            ("vw", 5), ("vmean", 6)):
                tot[nm].append(R[:, col])
            tot["dY"].append(dY)
        if tot:
            cat = {k: np.concatenate(v) for k, v in tot.items()}
            rgm = ols(cat["vmean"], cat["dQ"])
            rgw = ols(cat["vw"], cat["dQ"])
            se = float(cat["dQ"].std(ddof=1) / np.sqrt(len(cat["dQ"])))
            o(f"    POOLED over slots: pairs {len(cat['dQ'])}  y_f {cat['yf'].mean():.3f}  "
              f"y_p {cat['yp'].mean():.3f}  dY {cat['dY'].mean():+.3f}  "
              f"dQ {cat['dQ'].mean():+.4f} +- {se:.4f} ({cat['dQ'].mean()/se:+.1f} se)  "
              f"dP {cat['dP'].mean():+.4f}  dDP {np.nanmean(cat['dDP']):+.4f}")
            o(f"    POOLED dQ ~ V_mean (context-level): "
              + (f"slope {rgm['slope']:+.3f}  r {rgm['r']:+.3f}  n {rgm['n']}" if rgm else "-"))
            o(f"    POOLED dQ ~ V_w    (CONFOUNDED — `dQ` contains -logit(w) by construction, "
              f"which is norm §3's regression-to-the-mean, uncancelled here because the twin "
              f"is a CANDIDATE and not a second row): "
              + (f"slope {rgw['slope']:+.3f}  r {rgw['r']:+.3f}" if rgw else "-"))
            # the matched-prior contrast, pooled
            finc = np.isfinite(cat["dDP"])
            if int(finc.sum()) > 1000:
                ed = np.quantile(cat["dDP"][finc], np.linspace(0, 1, 6))
                ed[0] -= 1e-9
                cells = []
                for k in range(5):
                    mm = finc & (cat["dDP"] > ed[k]) & (cat["dDP"] <= ed[k + 1])
                    mv = mm & (np.abs(cat["dY"]) > 1e-9)
                    sg = (float((np.sign(cat["dQ"][mv]) == np.sign(cat["dY"][mv])).mean())
                          if int(mv.sum()) > 32 else float("nan"))
                    cells.append(f"{cat['dDP'][mm].mean():+.2f}->{cat['dQ'][mm].mean():+.2f}"
                                 f"/{cat['dY'][mm].mean():+.3f}/{sg:.3f}")
                o(f"    BY QUINTILE OF dDP (prior diff -> judge diff / world diff / sign "
                  f"agreement): " + "  ".join(cells))
        o("")

    # ------------------------------------------------------------- [R] across arms --- #
    if not a.no_cross and len(arms) == 2:
        o("-" * 104)
        o("[R] ACROSS ARMS — norm §1's protocol: two critics, one set of rows")
        o("-" * 104)
        o("  Each arm's critic is scored on the OTHER arm's rows, through its OWN trunk. The")
        o("  grammar is identical at every seed (`rule_seed 0` never moves) and a candidate is")
        o("  content, not an index, so the rows are portable. What is NOT matched, and what")
        o("  norm's frozen trunk did match, is the representation: each critic carries the trunk")
        o("  it was trained on, so this is two whole readers, not two readouts of one state.")
        o("  The two arms also differ in seed AND in probe draw at once, so this is one")
        o("  contrast, not a family.")
        o("")
        A0, A1 = arms
        for host, guest in ((A0, A1), (A1, A0)):
            o(f"    rows of {host.tag}:{host.arm}, scored by both critics")
            o(f"      {'rows':8} {'n':>8} {'base':>7} {'own':>7} {'other':>7} "
              f"{'slope(other~own)':>17} {'R2':>6} {'sdratio':>8}")
            for which in ("filed", "probe"):
                ks = [k for k in host.slots() if f"{which}:{k}" in S[host.tag]]
                own_p, oth_p, ys = [], [], []
                for k in ks:
                    pre = f"{which}:{k}"
                    d_own = S[host.tag][pre]
                    d_oth = host.score(pre, critic=guest.critic, core=guest.core)
                    own_p.append(d_own["p"])
                    oth_p.append(d_oth["p"])
                    ys.append(d_own["y"])
                own_p = np.concatenate(own_p)
                oth_p = np.concatenate(oth_p)
                ys = np.concatenate(ys)
                r = ols(own_p, oth_p)
                o(f"      {which:8} {len(ys):>8} {ys.mean():>7.3f} {own_p.mean():>7.3f} "
                  f"{oth_p.mean():>7.3f} "
                  + (f"{r['slope']:>17.3f} {r['r2']:>6.3f} {r['sd_ratio']:>8.3f}"
                     if r else f"{'-':>17} {'-':>6} {'-':>8}"))
            o("")

    out = a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
