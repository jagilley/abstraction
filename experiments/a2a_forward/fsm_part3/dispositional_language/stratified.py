"""Stratified re-analysis of the dispositional battery: does a pooled null hide structure?

Parent: dispositional.py (the run this re-analyses) -- read its DESIGN.md first.
Prompted by: the grammar sibling's open-loop arm, whose pooled dispositional numbers are
the same size as language's (+0.01 to +0.05 categorical) but which separate strongly by
hierarchy level -- +0.10 to +0.29 on the implementation rows at the three deep levels in
both time directions, behaviour flat at every level, and half of all positions in the
shallowest level where nothing is private on any row.

Language has no level labels, so this runs the same cut with PROXIES for computational
depth, three public and one private:

  syntactic  the parent harness's 5-way taxonomy (sentence_start / after_punct /
             after_opener / before_closer / mid). Public, and the natural proxy here: the
             residual's published behavioural signature is largest before closing
             delimiters and smallest at sentence starts.
  entropy    M's own output-entropy quartile at that checkpoint. Public, from the logits.
  position   position-in-sequence bins. Public, and the crudest context-depth proxy.
  resnorm    the OCCURRENT residual magnitude |r_c(p)| quartile at the default instrument.
             PRIVATE -- an outside observer cannot compute it -- so it is included for
             comparison only, never as an equal member of the set.

WHY THIS NEEDS A GPU RE-RUN AND WHAT IT COSTS
  The parent run saved aggregate scores, not per-position predictions, so the heads and
  observers must be re-fit to be interrogated. Nothing else is recomputed: M's
  checkpoints, the instrument FMs and the per-checkpoint parts are all loaded from the
  volume. Every seed on the head/observer path is explicit (`manual_seed(seed)` /
  `manual_seed(seed+31)` / generators at `seed+32`, `seed+77`, `0`), so the re-fit models
  are the same models, and the job prints a REPRODUCTION CHECK of every pooled score
  against the stored one before any stratified number is read. Only the strict evaluation
  set (held-out sequences at the held-out checkpoint block) is scored, and only six
  targets are re-fit, which is what keeps this under an hour on one L4.

  Per-position predictions ARE dumped this time, so any further cut is CPU-only.

Run:

  modal run --detach \
      a2a_forward/fsm_part3/dispositional_language/stratified.py::stratified_analyze \
      --tag main
"""

import json
import os

from a2a_forward.shared import app, volume, DATA_DIR, NumpyEncoder
from a2a_forward.confabulation.confabulation import (
    _make_head, _balance, _syntactic_labels,
)
from a2a_forward.fsm_part3.dispositional_language.dispositional import (
    _derive, _load_tokens, _report_set, _cos_rows, _unit16, _apply_lnf,
)

SYN_NAMES = {0: "mid_sentence", 1: "sentence_start", 2: "after_punct",
             3: "after_opener", 4: "before_closer"}


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=14400, memory=24576)
def stratified_analyze(tag: str = "main", n_pos_bins: int = 4, n_strat_q: int = 4):
    import resource
    import time
    import numpy as np
    import tiktoken
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from a2a_forward.model import Block
    from a2a_forward.forward_model import TransformerForwardModel

    t0 = time.time()
    base = f"{DATA_DIR}/a2a_forward/confabulation/dispositional/{tag}"
    cfg = json.load(open(f"{base}/results.json"))["config"]
    stored = json.load(open(f"{base}/results.json"))["targets"]

    d = _derive(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T, P = d["T"], d["T"] - 1
    n_embd, seed = cfg["n_embd"], cfg["seed"]
    caps, cap_tags, inst_caps = d["caps"], d["cap_tags"], d["inst_caps"]
    ct0, (dh0, mm0) = cap_tags[0], d["inst_caps"][0]
    retro_w, n_qclass = cfg["retro_w"], cfg["n_qclass"]
    obs_topk, hist_lags = cfg["obs_topk"], cfg["hist_lags"]
    steps_list = cfg["steps_list"]
    n_ckpt = len(steps_list)
    hold = cfg["hold_ckpts"]
    train_ckpts = [c for c in range(n_ckpt) if c not in set(hold)]
    k0 = int(cfg["horizons_str"].split(",")[0])

    _, val_data, vocab_size = _load_tokens(cfg)
    rep_tok = _report_set(cfg, val_data)
    N = rep_tok.shape[0]
    y_syn = _syntactic_labels(rep_tok, vocab_size, tiktoken.get_encoding("gpt2"))[:, :P]
    n_str = int(0.8 * N)
    s_tr, s_te = torch.arange(n_str), torch.arange(n_str, N)

    print(f"{'=' * 96}\nSTRATIFIED RE-ANALYSIS  tag={tag}")
    print(f"  strict set = held-out sequences ({len(s_te)}) x held-out checkpoints "
          f"{hold} (steps {[steps_list[c] for c in hold]}) = "
          f"{len(s_te) * len(hold) * P:,} positions")
    print(f"  stratifiers: syntactic(5), entropy quartile({n_strat_q}), "
          f"position bins({n_pos_bins}), |r| quartile({n_strat_q}, PRIVATE)")
    print(f"{'=' * 96}", flush=True)

    # ---------------- load the parts ----------------
    X_all = torch.empty(n_ckpt * N * P, n_embd, dtype=torch.float16)
    A0_all = torch.empty(n_ckpt, N, P, n_embd, dtype=torch.float16)
    IDS_all = torch.empty(n_ckpt, N, P, obs_topk, dtype=torch.int32)
    PRB_all = torch.empty(n_ckpt, N, P, obs_topk, dtype=torch.float16)
    STA_all = torch.empty(n_ckpt, N, P, 4, dtype=torch.float32)
    LOSS_all = torch.empty(n_ckpt, N, P, dtype=torch.float32)
    ENT_all = torch.empty(n_ckpt, N, P, dtype=torch.float32)
    LNF_W = torch.empty(n_ckpt, n_embd); LNF_B = torch.empty(n_ckpt, n_embd)
    floor0 = torch.zeros(n_ckpt, N, P)
    D_raw = torch.full((n_ckpt, N, P), float("nan"))
    D_fix = torch.full((n_ckpt, N, P), float("nan"))
    RNORM = {}                       # ci -> (N,P) |r| at the default instrument
    prev_unit, prev_fm = None, None

    def make_fm():
        return TransformerForwardModel(d_model=n_embd, d_head=dh0, n_head=cfg["fwd_n_head"],
                                       n_layer=cfg["fwd_n_layer"], mlp_mult=mm0,
                                       block_size=T).to(device)

    for ci in range(n_ckpt):
        pt = torch.load(f"{base}/parts/ck{ci:03d}.pt", map_location="cpu",
                        weights_only=False)
        X_all[ci * N * P:(ci + 1) * N * P] = pt["X"]
        A0_all[ci] = pt["a0"]; IDS_all[ci] = pt["ids"]; PRB_all[ci] = pt["prb"]
        STA_all[ci] = pt["sta"]; LOSS_all[ci] = pt["loss"]; ENT_all[ci] = pt["ent"]
        LNF_W[ci] = pt["lnf_w"]; LNF_B[ci] = pt["lnf_b"]
        cp = pt["caps"][ct0]
        floor0[ci] = cp["floor"]
        aj = pt["aj"]
        if prev_unit is not None:                       # k0 == 1: fold in the pair
            D_raw[ci - k0] = 1.0 - _cos_rows(prev_unit, cp["unit"])
            fmf = make_fm(); fmf.load_state_dict(prev_fm); fmf.eval()
            with torch.no_grad():
                rf = torch.cat([(aj[i:i + 64].float().to(device)
                                 - fmf(A0_all[ci, i:i + 64].float().to(device))).cpu()
                                for i in range(0, N, 64)])
            D_fix[ci - k0] = 1.0 - _cos_rows(prev_unit, _unit16(rf, P))
            del fmf, rf
        if ci in hold:                                  # |r| for the private stratifier
            fm0 = make_fm(); fm0.load_state_dict(cp["fm0"]); fm0.eval()
            with torch.no_grad():
                rn = torch.cat([(aj[i:i + 64].float().to(device)
                                 - fm0(A0_all[ci, i:i + 64].float().to(device))
                                 ).norm(dim=-1).cpu() for i in range(0, N, 64)])
            RNORM[ci] = rn
            del fm0
        prev_unit, prev_fm = cp["unit"], cp["fm0"]
        del pt, aj
        torch.cuda.empty_cache()
    del prev_unit, prev_fm
    print(f"  parts loaded in {(time.time() - t0) / 60:.1f} min; RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)

    # ---------------- rebuild the six targets ----------------
    def qclass(vals, valid_c):
        lab = torch.full(vals.shape, -1, dtype=torch.int8)
        qs = torch.tensor([(i + 1) / n_qclass for i in range(n_qclass - 1)])
        for c in valid_c:
            b = torch.quantile(vals[c][s_tr].reshape(-1).float(), qs)
            lab[c] = torch.bucketize(vals[c].float(), b).to(torch.int8)
        return lab

    vc_p = [c for c in range(n_ckpt) if c + k0 < n_ckpt]
    vc_r = [c for c in range(n_ckpt) if c - retro_w >= 0]
    integ_i = torch.zeros(n_ckpt, N, P)
    integ_b = torch.zeros(n_ckpt, N, P)
    dl = torch.full((n_ckpt, N, P), float("nan"))
    for c in vc_p:
        dl[c] = LOSS_all[c] - LOSS_all[c + k0]
    for c in vc_r:
        for i in range(1, retro_w + 1):
            integ_i[c] += D_raw[c - i]
            integ_b[c] += (LOSS_all[c - i] - LOSS_all[c - i + 1]).clamp_min(0)

    TARGETS = {
        f"IMPL_PROSP_k{k0}_{ct0}": (D_raw, vc_p, "IMPL", "prosp"),
        f"BEHAV_PROSP_k{k0}": (dl, vc_p, "BEHAV", "prosp"),
        f"IMPL_RETRO_w{retro_w}_{ct0}": (integ_i, vc_r, "IMPL", "retro"),
        f"BEHAV_RETRO_w{retro_w}": (integ_b, vc_r, "BEHAV", "retro"),
        f"IMPL_PROSPEXC_k{k0}_{ct0}": (D_raw - floor0, vc_p, "IMPL", "prosp"),
        f"IMPL_PROSPFIX_k{k0}_{ct0}": (D_fix, vc_p, "IMPL", "prosp"),
    }
    targets = {nm: {"y": qclass(v, vc), "valid_c": vc, "n_out": n_qclass,
                    "family": fam, "direction": dr}
               for nm, (v, vc, fam, dr) in TARGETS.items()}

    # ---------------- head / observer machinery (verbatim from analyze) -------------
    LNF_W_d, LNF_B_d = LNF_W.to(device), LNF_B.to(device)

    def rows_for(ci_list, seqs):
        ci_t = torch.tensor(ci_list)[:, None, None]
        return (ci_t * (N * P) + seqs[None, :, None] * P
                + torch.arange(P)[None, None, :]).reshape(-1)

    def ci_of(rows):
        return torch.div(rows, N * P, rounding_mode="floor")

    def items_of(ci_list, seqs, shuffle=True):
        ci_t = torch.tensor(ci_list)[:, None].expand(len(ci_list), len(seqs))
        it = torch.stack([ci_t.reshape(-1),
                          seqs[None, :].expand(len(ci_list), len(seqs)).reshape(-1)], -1)
        if shuffle:
            it = it[torch.randperm(len(it),
                                   generator=torch.Generator().manual_seed(seed + 77))]
        return it

    def fit_head(y_flat, n_out, rows_tr):
        torch.manual_seed(seed)
        net = _make_head(n_embd, n_out, cfg["head_hidden"], device, seed)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["head_lr"], weight_decay=1e-4)
        g = torch.Generator().manual_seed(0)
        bs = min(cfg["head_bs"], len(rows_tr))
        for _ in range(cfg["head_steps"]):
            si = rows_tr[torch.randint(len(rows_tr), (bs,), generator=g)]
            xb = _apply_lnf(X_all[si].to(device).float(), ci_of(si).to(device),
                            LNF_W_d, LNF_B_d)
            F.cross_entropy(net(xb), y_flat[si].to(device).long()).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        net.eval()
        return net

    def head_preds(net, rows_te, bs=16384):
        out = []
        with torch.no_grad():
            for i in range(0, len(rows_te), bs):
                si = rows_te[i:i + bs]
                xb = _apply_lnf(X_all[si].to(device).float(), ci_of(si).to(device),
                                LNF_W_d, LNF_B_d)
                out.append(net(xb).argmax(-1).cpu().to(torch.int8))
        return torch.cat(out)

    class ObserverD(nn.Module):
        def __init__(self, n_out, o_layer, o_embd, mode, act_dim=0):
            super().__init__()
            self.mode = mode
            self.wte = nn.Embedding(vocab_size, o_embd)
            self.wpe = nn.Embedding(T, o_embd)
            self.cke = nn.Embedding(n_ckpt, o_embd)
            if mode in ("io", "hist"):
                self.pshape = nn.Linear(obs_topk, o_embd)
                self.sproj = nn.Linear(4, o_embd)
            if mode == "hist":
                self.lage = nn.Embedding(hist_lags + 1, o_embd)
            if mode == "act":
                self.aproj = nn.Linear(act_dim, o_embd)
            self.h = nn.ModuleList([
                Block(o_embd, min(max(o_layer, 1) * 2, 8), T) for _ in range(o_layer)])
            if not cfg["observer_causal"]:
                for blk in self.h:
                    blk.attn.bias.fill_(1.0)
            self.ln_f = nn.LayerNorm(o_embd)
            self.head = nn.Linear(o_embd, n_out)

        def io_emb(self, ids, pp, st):
            return ((self.wte(ids) * pp.unsqueeze(-1)).sum(-2) + self.pshape(pp)
                    + self.sproj(st))

        def forward(self, tok, ci, ids=None, pp=None, st=None, act=None,
                    hids=None, hpp=None, hst=None, hmask=None):
            B, t = tok.shape
            x = self.wpe(torch.arange(t, device=tok.device))[None].expand(B, t, -1)
            x = x + self.wte(tok) + self.cke(ci)[:, None, :]
            if self.mode == "io":
                x = x + self.io_emb(ids, pp, st)
            if self.mode == "act":
                x = x + self.aproj(act)
            if self.mode == "hist":
                for l in range(hist_lags + 1):
                    e = self.io_emb(hids[:, l], hpp[:, l], hst[:, l]) \
                        + self.lage(torch.full((1,), l, device=tok.device,
                                               dtype=torch.long))[None]
                    x = x + e * hmask[:, l][:, None, None]
            for blk in self.h:
                x = blk(x)
            return self.head(self.ln_f(x))

    def obs_batch(items, mode):
        ci, sq = items[:, 0], items[:, 1]
        kw = {"tok": rep_tok[sq, :P].to(device), "ci": ci.to(device)}
        if mode == "io":
            kw.update(ids=IDS_all[ci, sq].long().to(device),
                      pp=PRB_all[ci, sq].float().to(device),
                      st=STA_all[ci, sq].to(device))
        elif mode == "act":
            kw.update(act=A0_all[ci, sq].float().to(device))
        elif mode == "hist":
            lag = torch.arange(hist_lags + 1)[None, :]
            src = (ci[:, None] - lag).clamp_min(0)
            msk = (ci[:, None] - lag >= 0).float()
            sq2 = sq[:, None].expand_as(src)
            kw.update(hids=IDS_all[src, sq2].long().to(device),
                      hpp=PRB_all[src, sq2].float().to(device),
                      hst=STA_all[src, sq2].to(device), hmask=msk.to(device))
        return kw

    def train_observer(tgt, o_layer, o_embd, mode, items_tr, items_te):
        torch.manual_seed(seed + 31)
        o = ObserverD(tgt["n_out"], o_layer, o_embd, mode, act_dim=n_embd).to(device)
        opt = torch.optim.AdamW(o.parameters(), lr=cfg["obs_lr"], weight_decay=0.01)
        g = torch.Generator().manual_seed(seed + 32)
        Y, obs_bs = tgt["y"], cfg["obs_bs"]
        for _ in range(cfg["obs_steps"]):
            it = items_tr[torch.randint(len(items_tr), (min(obs_bs, len(items_tr)),),
                                        generator=g)]
            out = o(**obs_batch(it, mode))
            F.cross_entropy(out.reshape(-1, tgt["n_out"]),
                            Y[it[:, 0], it[:, 1]].to(device).reshape(-1).long()).backward()
            torch.nn.utils.clip_grad_norm_(o.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        o.eval()
        pr = []
        with torch.no_grad():
            for i in range(0, len(items_te), 32):
                it = items_te[i:i + 32]
                pr.append(o(**obs_batch(it, mode)).argmax(-1).cpu().to(torch.int8))
        del o
        torch.cuda.empty_cache()
        return torch.cat(pr).reshape(-1)          # (len(items_te)*P,) in item order

    # ---------------- fit, predict, and check against the stored run ----------------
    preds, ylab, check = {}, {}, []
    ol_, oe_ = caps[-1]
    for nm, tgt in targets.items():
        t1 = time.time()
        vc = set(tgt["valid_c"])
        tr_c = [c for c in train_ckpts if c in vc]
        te_c = [c for c in hold if c in vc]
        y_flat = tgt["y"].reshape(-1)
        rows_tr = rows_for(tr_c, s_tr)
        rows_te = rows_for(te_c, s_te)                      # canonical eval order
        it_tr = items_of(tr_c, s_tr)
        it_te = items_of(te_c, s_te, shuffle=False)         # same (ci, seq) order
        ylab[nm] = y_flat[rows_te].clone()
        p = {"self": head_preds(fit_head(y_flat, tgt["n_out"], rows_tr), rows_te)}
        for (o_layer, o_embd) in caps:
            p[f"O_io@{o_layer}L{o_embd}D"] = train_observer(
                tgt, o_layer, o_embd, "io", it_tr, it_te)
        p[f"O_act@{ol_}L{oe_}D"] = train_observer(tgt, ol_, oe_, "act", it_tr, it_te)
        p[f"O_hist@{ol_}L{oe_}D"] = train_observer(tgt, ol_, oe_, "hist", it_tr, it_te)
        preds[nm] = p
        for k, v in p.items():
            acc = float((v == ylab[nm]).float().mean())
            was = (stored[nm]["self_lnf_heldck"] if k == "self"
                   else stored[nm]["observers"].get(k, [float("nan")] * 2)[1])
            check.append((nm, k, acc, was))
        print(f"  [{nm}] re-fit in {(time.time() - t1) / 60:.1f} min", flush=True)

    print(f"\n{'-' * 96}\nREPRODUCTION CHECK  (re-fit pooled strict score vs the stored "
          f"run; every seed on this path is explicit, so these should agree to ~1e-3)")
    print(f"  {'target':32s} {'predictor':22s} {'re-fit':>8s} {'stored':>8s} {'d':>8s}")
    worst = 0.0
    for nm, k, acc, was in check:
        dd = acc - was if was == was else float("nan")
        worst = max(worst, abs(dd) if dd == dd else 0.0)
        print(f"  {nm:32s} {k:22s} {acc:8.4f} {was:8.4f} {dd:+8.4f}")
    print(f"  max |difference| = {worst:.4f}\n{'-' * 96}", flush=True)

    # ---------------- stratifiers, on the canonical eval rows ----------------
    def eval_frame(te_c):
        ci = torch.tensor(te_c)[:, None, None].expand(len(te_c), len(s_te), P).reshape(-1)
        sq = s_te[None, :, None].expand(len(te_c), len(s_te), P).reshape(-1)
        po = torch.arange(P)[None, None, :].expand(len(te_c), len(s_te), P).reshape(-1)
        return ci, sq, po

    def qbins(per_ck, te_c, ci, sq, po, nq):
        """Quantile bin per row, boundaries fit per checkpoint on TRAIN sequences."""
        out = torch.empty(len(ci), dtype=torch.int8)
        qs = torch.tensor([(i + 1) / nq for i in range(nq - 1)])
        for c in te_c:
            m = ci == c
            b = torch.quantile(per_ck[c][s_tr].reshape(-1).float(), qs)
            out[m] = torch.bucketize(per_ck[c][sq[m], po[m]].float(), b).to(torch.int8)
        return out

    strat_defs = {}
    for nm, tgt in targets.items():
        te_c = [c for c in hold if c in set(tgt["valid_c"])]
        ci, sq, po = eval_frame(te_c)
        st = {"syntactic": (y_syn[sq, po].to(torch.int8),
                            {g: SYN_NAMES[g] for g in range(5)}, "public"),
              "entropy_q": (qbins(ENT_all, te_c, ci, sq, po, n_strat_q),
                            {q: f"H q{q + 1}" for q in range(n_strat_q)}, "public"),
              "position": ((po // max(1, (P + n_pos_bins - 1) // n_pos_bins)).to(torch.int8),
                           {b: f"pos {b * ((P + n_pos_bins - 1) // n_pos_bins)}-"
                               f"{min(P, (b + 1) * ((P + n_pos_bins - 1) // n_pos_bins)) - 1}"
                            for b in range(n_pos_bins)}, "public")}
        rn = torch.zeros(n_ckpt, N, P)
        for c in te_c:
            rn[c] = RNORM[c]
        st["resnorm_q"] = (qbins(rn, te_c, ci, sq, po, n_strat_q),
                           {q: f"|r| q{q + 1}" for q in range(n_strat_q)}, "PRIVATE")
        strat_defs[nm] = (st, ci, sq, po, te_c)

    # ---------------- tables ----------------
    out = {"config": {"tag": tag, "n_pos_bins": n_pos_bins, "n_strat_q": n_strat_q},
           "reproduction_check": [{"target": a_, "predictor": b_, "refit": c_,
                                   "stored": d_} for a_, b_, c_, d_ in check],
           "strata": {}}
    io_keys = [f"O_io@{l}L{e}D" for (l, e) in caps]
    act_key, hist_key = f"O_act@{ol_}L{oe_}D", f"O_hist@{ol_}L{oe_}D"

    for sname in ("syntactic", "entropy_q", "position", "resnorm_q"):
        kind = strat_defs[next(iter(targets))][0][sname][2]
        print(f"\n{'=' * 96}\nSTRATIFIER: {sname}  [{kind}]\n{'=' * 96}")
        print(f"  {'target':30s} {'stratum':16s} {'n':>8s} {'share':>6s} {'chance':>7s} "
              f"{'self':>7s} {'O_io':>7s} {'O_hist':>7s} {'O_act':>7s} {'ADVio':>7s} "
              f"{'ADVpub':>7s}")
        out["strata"][sname] = {"kind": kind, "rows": []}
        for nm, tgt in targets.items():
            st, ci, sq, po, te_c = strat_defs[nm]
            lab, names, _ = st[sname]
            y = ylab[nm]
            tot = len(y)
            for g in sorted(names):
                m = lab == g
                n_g = int(m.sum())
                if n_g < 200:
                    continue
                yg = y[m].long()
                ch = _balance(yg, tgt["n_out"])
                sc = {k: float((v[m] == y[m]).float().mean()) for k, v in preds[nm].items()}
                bio = max(sc[k] for k in io_keys)
                bpub = max(bio, sc[hist_key])
                row = {"target": nm, "family": tgt["family"],
                       "direction": tgt["direction"], "stratum": names[g], "n": n_g,
                       "share": n_g / tot, "chance": ch, "self": sc["self"],
                       "O_io": bio, "O_hist": sc[hist_key], "O_act": sc[act_key],
                       "adv_io": sc["self"] - bio, "adv_pub": sc["self"] - bpub,
                       "per_cap_io": {k: sc[k] for k in io_keys}}
                out["strata"][sname]["rows"].append(row)
                print(f"  {nm:30s} {names[g]:16s} {n_g:8d} {n_g / tot:6.3f} {ch:7.3f} "
                      f"{sc['self']:7.3f} {bio:7.3f} {sc[hist_key]:7.3f} "
                      f"{sc[act_key]:7.3f} {row['adv_io']:+7.3f} {row['adv_pub']:+7.3f}")
            print("")

    os.makedirs(f"{base}/stratified", exist_ok=True)
    with open(f"{base}/stratified/stratified.json", "w") as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)
    torch.save({"preds": preds, "y": ylab,
                "strata": {nm: {s: strat_defs[nm][0][s][0] for s in strat_defs[nm][0]}
                           for nm in targets},
                "frame": {nm: {"ci": strat_defs[nm][1], "seq": strat_defs[nm][2],
                               "pos": strat_defs[nm][3]} for nm in targets}},
               f"{base}/stratified/predictions.pt")
    volume.commit()
    print(f"\n  saved -> {base}/stratified/  (stratified.json + predictions.pt; any "
          f"further cut is CPU-only from predictions.pt)")
    print(f"  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB"
          f"   wall {(time.time() - t0) / 60:.1f} min\n{'=' * 96}", flush=True)
    return out
