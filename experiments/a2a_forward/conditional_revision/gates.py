"""Belief revision vs. surprisal, on a language substrate.

The language sibling of [`rhm/conditional_revision`](../../rhm/conditional_revision/README.md).
Substrate: `corpus.py` (controlled English, exact oracle). Reader: a frozen
pretrained LM. Nothing is trained except a belief probe, a temporal forward
model, and cross-fit linear discriminants -- the reader itself never moves.

Gates:

  0   substrate + instrument. Alignment, does the reader read, probe vs Bayes
      ceiling, and RHM's Gate-0 statistics `corr(r_temporal, nll)` / `R^2`.
  A   partial R^2( M ~ B | nll ) vs partial R^2( M ~ nll | B ).
  B1  negate_live vs negate_dead -- identical frame, oracle surprisal pinned to
      log 4 on both sides, matched on how often the word already occurred. The
      confound-free contrast RHM has no analogue of.
  B2  reveal(L=4) vs synonym/offtopic/mention -- oracle surprisal *exactly*
      equal by construction, frame confounded (reported and quantified).
      B2b narrows it to value-word vs value-word (reveal vs mention).
  B3  mention vs synonym/offtopic -- all B = 0. A specificity guard: anything
      that separates these is reading content, not revision about z.
  C   which readout carries it: the difference `M`, the prefix-scored-at-the-
      arriving-value `M_pointmass`, prefix entropy alone, the FM residual norm,
      and the FM residual direction.
  D   variance budget of the FM target, per family. Says whether a local loss
      on this target could see the distinction at all.
  E   deduced vs echo -- oracle B = 0 and oracle surprisal 0 on both sides, so
      any model-side movement is the model's own incomplete inference.
"""

import json
import os

import modal

from a2a_forward.conditional_revision.shared import app, volume, DATA_DIR, image, NumpyEncoder

QUERIES = {
    "person": "\nQ: Who was the visitor that night? A:",
    "object": "\nQ: Which item is missing? A: the",
    "place":  "\nQ: Where will the search begin? A: the",
    "hour":   "\nQ: When was the door locked? A:",
}

# families, grouped for the gates
POS_B1, NEG_B1 = ("negate_live",), ("negate_dead",)
POS_B2, NEG_B2 = ("reveal",), ("synonym", "offtopic", "mention")
POS_B2b, NEG_B2b = ("reveal",), ("mention",)      # value word vs value word
POS_B3, NEG_B3 = ("mention",), ("synonym", "offtopic")   # all B = 0: guard
POS_E, NEG_E = ("deduced",), ("echo",)


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600,
              memory=65536, image=image)
def gates(
    model_name: str = "unsloth/Llama-3.2-1B",
    layer: int = 12,
    probe_layer: int = -1,      # -1 = final block. The FM and the directional
                                # readouts stay on `layer`; only the belief
                                # probe reads here.
    n_stories: int = 3000,
    n_clauses: int = 13,
    seed: int = 42,
    tag: str = "v1",
    lm_batch: int = 8,
    probe_steps: int = 4000,
    probe_hidden: int = 1024,
    probe_lr: float = 1e-3,
    fm_steps: int = 3000,
    fm_mlp_mult: float = 1.0,
    fm_lr: float = 3e-4,
    fm_batch: int = 8,
    n_prompt_stories: int = 400,
    prompt_batch: int = 48,
    disc_steps: int = 800,
    reuse_cache: bool = True,
):
    import math
    import random
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from a2a_forward.conditional_revision import corpus as C
    from a2a_forward.conditional_revision.fm import TemporalFM, SlotProbe, fit_discriminant
    from a2a_forward.conditional_revision.stats import (_r2, _partial_r2, _partial_r2_rank, _auc,
                               _stratified_auc, _strata, _cross, _codes,
                               _shuffle_within)

    t00 = time.time()
    device = "cuda"
    torch.manual_seed(seed)
    R = {"config": dict(model_name=model_name, layer=layer, n_stories=n_stories,
                        n_clauses=n_clauses, seed=seed, tag=tag,
                        fm_mlp_mult=fm_mlp_mult, fm_steps=fm_steps,
                        probe_steps=probe_steps, probe_layer=probe_layer)}

    # ---------------------------------------------------------------- reader
    print(f"loading {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.hidden_size
    n_layer_lm = model.config.num_hidden_layers
    assert 0 < layer <= n_layer_lm, f"layer {layer} out of range for {n_layer_lm}"
    print(f"  d_model={d_model} n_layer={n_layer_lm}", flush=True)

    def one_tok(w):
        return len(tok.encode(w, add_special_tokens=False)) == 1

    lex = C.build_lexicon(one_tok)
    R["lexicon"] = {"slots": lex.slot_values, "decoys": lex.decoy_values,
                    "synsets": lex.synsets}
    print("lexicon:", {k: v for k, v in lex.slot_values.items()}, flush=True)

    # -------------------------------------------------------------- corpus
    rng = random.Random(seed)
    stories = [C.generate_story(rng, lex, n_clauses=n_clauses)
               for _ in range(n_stories)]
    print(f"generated {n_stories} stories in {time.time()-t00:.0f}s", flush=True)

    # tokenize + align analysis words to token indices
    enc = tok([s.text for s in stories], return_offsets_mapping=True,
              add_special_tokens=True)
    lens = [len(x) for x in enc["input_ids"]]
    T_max = max(lens)
    n_align_fail = 0
    aligned = []                       # per story: list of (pos, token_index)
    for si, st in enumerate(stories):
        offs = enc["offset_mapping"][si]
        starts = {o[0]: k for k, o in enumerate(offs)}
        rec = []
        for p in st.positions:
            if not p.analysis:      # narrow-clause tokens: the posterior is
                continue            # momentarily non-uniform inside the clause
            k = starts.get(p.char_start)
            if k is None:               # token boundary includes the space
                k = starts.get(p.char_start - 1)
            if k is None or k == 0 or k >= lens[si] - 1:
                n_align_fail += 1
                continue
            surf = st.text[offs[k][0]:offs[k][1]].strip()
            if surf != p.word.strip():
                n_align_fail += 1
                continue
            rec.append((p, k))
        aligned.append(rec)
    n_pos_total = sum(len(r) for r in aligned)
    print(f"aligned {n_pos_total} analysis tokens, {n_align_fail} failures, "
          f"T_max={T_max}", flush=True)
    R["gate0"] = {"n_positions": n_pos_total, "n_align_fail": n_align_fail,
                  "T_max": T_max, "mean_len": float(np.mean(lens))}
    assert n_align_fail / max(n_pos_total + n_align_fail, 1) < 0.01, \
        "token alignment is unreliable; check the lexicon"

    ids = torch.full((n_stories, T_max), tok.eos_token_id or 0, dtype=torch.long)
    amask = torch.zeros((n_stories, T_max), dtype=torch.long)
    for si in range(n_stories):
        ids[si, :lens[si]] = torch.tensor(enc["input_ids"][si])
        amask[si, :lens[si]] = 1

    # ------------------------------------------------- forward pass, cached
    # The reader forward is 58s but the prompted readout is ~1400s, and both are
    # a deterministic function of (model, corpus seed, layers). Cache them so
    # iterating on the ANALYSIS costs minutes rather than half an hour.
    probe_layer = n_layer_lm if probe_layer < 0 else probe_layer
    ck = (f"{model_name.split('/')[-1]}_L{layer}P{probe_layer}"
          f"_n{n_stories}_c{n_clauses}_s{seed}")
    cache_dir = f"{DATA_DIR}/chronicle/cache"
    os.makedirs(cache_dir, exist_ok=True)
    acts_path = f"{cache_dir}/acts_{ck}.pt"
    prompt_path = f"{cache_dir}/prompt_{ck}_p{n_prompt_stories}.pt"
    H = torch.zeros((n_stories, T_max, d_model), dtype=torch.float16)
    HP = (H if probe_layer == layer
          else torch.zeros((n_stories, T_max, d_model), dtype=torch.float16))
    NLL = torch.zeros((n_stories, T_max), dtype=torch.float32)
    ENT = torch.zeros((n_stories, T_max), dtype=torch.float32)   # H(p(.|x_<t)) at t
    t0 = time.time()
    have_acts = reuse_cache and os.path.exists(acts_path)
    if have_acts:
        blob = torch.load(acts_path, map_location="cpu", weights_only=False)
        H, NLL, ENT = blob["H"], blob["NLL"], blob["ENT"]
        HP = blob["HP"] if blob["HP"] is not None else H
        print(f"loaded cached activations {acts_path}", flush=True)
    with torch.no_grad():
        for i in range(0, 0 if have_acts else n_stories, lm_batch):
            sl = slice(i, min(i + lm_batch, n_stories))
            out = model(input_ids=ids[sl].to(device),
                        attention_mask=amask[sl].to(device),
                        output_hidden_states=True)
            H[sl] = out.hidden_states[layer].to(torch.float16).cpu()
            if probe_layer != layer:
                HP[sl] = out.hidden_states[probe_layer].to(torch.float16).cpu()
            tgt = ids[sl][:, 1:].to(device)
            # chunk over time: a (B, T, 128k) float32 log-softmax is ~1.5 GB and
            # the exp() for the entropy doubles it.
            for c0 in range(0, T_max - 1, 64):
                c1 = min(c0 + 64, T_max - 1)
                lp = torch.log_softmax(out.logits[:, c0:c1].float(), dim=-1)
                NLL[sl, c0 + 1:c1 + 1] = -lp.gather(
                    -1, tgt[:, c0:c1].unsqueeze(-1)).squeeze(-1).cpu()
                ENT[sl, c0 + 1:c1 + 1] = (-(lp.exp() * lp).sum(-1)).cpu()
                del lp
            del out
            if i % (lm_batch * 50) == 0:
                print(f"  lm {i}/{n_stories}  {time.time()-t0:.0f}s", flush=True)
    if not have_acts:
        torch.save({"H": H, "HP": None if probe_layer == layer else HP,
                    "NLL": NLL, "ENT": ENT}, acts_path)
        volume.commit()
    print(f"reader forward done in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------- splits
    perm = np.random.RandomState(seed).permutation(n_stories)
    n_tr = int(0.60 * n_stories)
    n_ca = int(0.10 * n_stories)
    idx_tr = np.sort(perm[:n_tr])
    idx_ca = np.sort(perm[n_tr:n_tr + n_ca])
    idx_te = np.sort(perm[n_tr + n_ca:])
    print(f"split: train {len(idx_tr)} calib {len(idx_ca)} test {len(idx_te)}",
          flush=True)

    n_slots, n_val = len(C.SLOTS), C.N_VAL
    Z = torch.tensor([[st.z[k] for k in C.SLOTS] for st in stories])

    # --------------------------------------------------------------- probe
    # h -> per-slot posterior over the 4 values. Trained on train stories at all
    # token positions (supervision = the true scene), so at early positions the
    # optimum is the prior and the probe's calibration is meaningful.
    mu = HP[idx_tr].reshape(-1, d_model).float().mean(0, keepdim=True)
    sd = HP[idx_tr].reshape(-1, d_model).float().std(0, keepdim=True) + 1e-6

    probe = SlotProbe(d_model, n_slots, n_val, probe_hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed + 7)
    lens_t = torch.tensor(lens)
    idx_tr_t = torch.tensor(idx_tr)
    t0 = time.time()
    for st_i in range(probe_steps):
        bs = torch.randint(0, len(idx_tr), (256,), generator=g)
        si = idx_tr_t[bs]
        tt = (torch.rand(256, generator=g) * (lens_t[si] - 1)).long() + 1
        xb = ((HP[si, tt].float() - mu) / sd).to(device)
        yb = Z[si].to(device)
        loss = F.cross_entropy(probe(xb).reshape(-1, n_val), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if st_i % max(probe_steps // 4, 1) == 0:
            print(f"  probe {st_i:5d} ce {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    probe.eval()

    @torch.no_grad()
    def probe_logits(hh):
        out = []
        for i in range(0, hh.shape[0], 4096):
            xb = ((hh[i:i + 4096].float() - mu) / sd).to(device)
            out.append(probe(xb).cpu())
        return torch.cat(out)

    # temperature calibration on the calib split (RHM lesson: an over-confident
    # probe inflates every KL readout uniformly)
    ca_h, ca_y = [], []
    rs = np.random.RandomState(seed + 3)
    for s in idx_ca:
        for t_ in rs.randint(1, lens[s], size=12):
            ca_h.append(HP[s, t_]); ca_y.append(Z[s])
    ca_lg = probe_logits(torch.stack(ca_h))
    ca_y = torch.stack(ca_y)
    logT = torch.zeros(1, requires_grad=True)
    optT = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def _closure():
        optT.zero_grad()
        l = F.cross_entropy((ca_lg / logT.exp()).reshape(-1, n_val),
                            ca_y.reshape(-1))
        l.backward()
        return l
    optT.step(_closure)
    Temp = float(logT.exp().item())
    print(f"  probe temperature {Temp:.3f}", flush=True)

    # ------------------------------------------------- the analysis records
    fams, slots_, frames, Bv, gsurp, Hpb, Hpa, nlive_b = [], [], [], [], [], [], [], []
    si_a, tj_a, clause_a, tokfrac_a, nment_a, kill_a = [], [], [], [], [], []
    value_idx_a = []
    live_b_all, live_a_all = [], []
    split_a = []
    split_of = {}
    for s in idx_tr: split_of[int(s)] = 0
    for s in idx_ca: split_of[int(s)] = 1
    for s in idx_te: split_of[int(s)] = 2
    for si in range(n_stories):
        for p, k in aligned[si]:
            fams.append(p.family); slots_.append(p.slot or "-")
            frames.append(p.frame); Bv.append(p.B); gsurp.append(p.gen_surprisal)
            Hpb.append(p.H_post_before); Hpa.append(p.H_post_after)
            nlive_b.append(p.n_live_before)
            nment_a.append(min(p.n_prior_mentions, 2))
            value_idx_a.append(-1 if p.value_idx is None else p.value_idx)
            kill_a.append(p.kill_mech or "-")
            si_a.append(si); tj_a.append(k); clause_a.append(p.clause_idx)
            tokfrac_a.append(k / lens[si])
            live_b_all.append(p.live_before); live_a_all.append(p.live_after)
            split_a.append(split_of[si])
    si_a = np.array(si_a); tj_a = np.array(tj_a)
    fams = np.array(fams); frames = np.array(frames); slots_ = np.array(slots_)
    Bv = np.array(Bv); gsurp = np.array(gsurp)
    Hpb = np.array(Hpb); Hpa = np.array(Hpa)
    clause_a = np.array(clause_a); tokfrac_a = np.array(tokfrac_a)
    nment_a = np.array(nment_a); kill_a = np.array(kill_a)
    # the exact arriving word. Matching on it makes B1 as tight as it can get:
    # identical word, identical frame, identical live-set size, identical
    # mention count -- the two families then differ ONLY in prefix history.
    tokid_a = _codes([f"{a}/{b}" for a, b in zip(slots_, [str(x) for x in
                                                         value_idx_a])])
    split_a = np.array(split_a)
    N = len(fams)

    nll_a = NLL[si_a, tj_a].numpy()
    ent_a = ENT[si_a, tj_a].numpy()      # H(p(.|x_<j)), the incumbent
    h_before = H[si_a, tj_a - 1]         # the FM's layer: directional readouts
    h_after = H[si_a, tj_a]
    hp_before = HP[si_a, tj_a - 1]       # the probe's layer: belief readouts
    hp_after = HP[si_a, tj_a]

    # oracle per-slot posteriors (uniform on the live support)
    def _oracle_q(live_list):
        q = np.zeros((len(live_list), n_slots, n_val), dtype=np.float64)
        for i, lv in enumerate(live_list):
            for k in range(n_slots):
                for v in lv[k]:
                    q[i, k, v] = 1.0 / len(lv[k])
        return q
    q_or_b = _oracle_q(live_b_all)
    q_or_a = _oracle_q(live_a_all)

    # probe posteriors
    lg_b = torch.softmax(probe_logits(hp_before) / Temp, -1).double().numpy()
    lg_a = torch.softmax(probe_logits(hp_after) / Temp, -1).double().numpy()

    def _kl(p, q):
        return float(np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-30))
                                                 - np.log(np.maximum(q, 1e-30))), 0.0)))

    def belief_readouts(qb, qa):
        """M (the difference), M_pointmass (prefix scored at the arriving value),
        prefix/after entropy. `M_pointmass` is the object the RHM tracking audit
        found recovers 93-99.6% of `M`; it uses the after-state only to NAME a
        value, so it is 'the prefix modified by the token' rather than a
        two-forecast difference."""
        eps = 1e-30
        Mv = np.sum(qa * (np.log(qa + eps) - np.log(qb + eps)), axis=(1, 2))
        am = qa.argmax(-1)
        pm = -np.log(np.take_along_axis(qb, am[..., None], -1).squeeze(-1)
                     + eps).sum(-1)
        Hb_ = -np.sum(qb * np.log(qb + eps), axis=(1, 2))
        Ha_ = -np.sum(qa * np.log(qa + eps), axis=(1, 2))
        return Mv, pm, -Hb_, -Ha_

    M_pr, Mpm_pr, negH_b_pr, negH_a_pr = belief_readouts(lg_b, lg_a)
    B_joint_chk = np.array([_kl(q_or_a[i], q_or_b[i]) for i in range(N)])
    R["gate0"]["oracle_B_recon_max_err"] = float(np.max(np.abs(B_joint_chk - Bv)))

    # probe accuracy vs the exact Bayes ceiling, per slot
    acc, ceil = {}, {}
    te_m = split_a == 2
    for k, sk in enumerate(C.SLOTS):
        pred = lg_a[te_m, k].argmax(-1)
        truth = Z[si_a[te_m], k].numpy()
        acc[sk] = float((pred == truth).mean())
        ceil[sk] = float(np.mean([1.0 / len(live_a_all[i][k])
                                  for i in np.where(te_m)[0]]))
    R["gate0"]["probe_acc"] = acc
    R["gate0"]["bayes_ceiling"] = ceil
    R["gate0"]["probe_frac_of_ceiling"] = {k: acc[k] / ceil[k] for k in acc}
    print("probe acc / ceiling:", {k: f"{acc[k]:.3f}/{ceil[k]:.3f}" for k in acc},
          flush=True)

    # nll by family -- does the reader read?
    R["gate0"]["nll_by_family"] = {
        f: float(nll_a[fams == f].mean()) for f in sorted(set(fams))}
    R["gate0"]["out_entropy_by_family"] = {
        f: float(ent_a[fams == f].mean()) for f in sorted(set(fams))}
    R["gate0"]["n_by_family"] = {f: int((fams == f).sum()) for f in sorted(set(fams))}
    print("nll by family:", R["gate0"]["nll_by_family"], flush=True)

    # ---------------------------------------------------------- temporal FM
    hmu = H[idx_tr].reshape(-1, d_model).float().mean(0)
    hsd = H[idx_tr].reshape(-1, d_model).float().std(0) + 1e-6

    fm = TemporalFM(d_model, n_layer=1, mlp_mult=fm_mlp_mult,
                    block_size=T_max).to(device)
    print(f"TemporalFM: {fm.n_params()/1e6:.1f}M params", flush=True)
    optf = torch.optim.AdamW(fm.parameters(), lr=fm_lr, weight_decay=0.01)
    gf = torch.Generator().manual_seed(seed + 11)
    t0 = time.time()
    for st_i in range(fm_steps):
        bs = torch.randint(0, len(idx_tr), (fm_batch,), generator=gf)
        si = torch.tensor(idx_tr)[bs]
        hb = ((H[si].float() - hmu) / hsd).to(device)
        L = torch.tensor([lens[int(s)] for s in si], device=device)
        tmask = (torch.arange(T_max, device=device)[None, :] < (L[:, None] - 1))
        delta = hb[:, 1:] - hb[:, :-1]
        pred = fm(hb)[:, :-1]
        m = tmask[:, :-1].unsqueeze(-1).float()
        loss = (((pred - delta) ** 2) * m).sum() / (m.sum() * d_model)
        optf.zero_grad(); loss.backward(); optf.step()
        if st_i % max(fm_steps // 4, 1) == 0:
            dvar = ((delta ** 2) * m).sum() / (m.sum() * d_model)
            print(f"  fm {st_i:5d} mse {loss.item():.4f} "
                  f"(target var {dvar.item():.4f})  {time.time()-t0:.0f}s",
                  flush=True)
    fm.eval()

    # residuals at every analysis position
    r_vec = torch.zeros((N, d_model), dtype=torch.float32)
    d_vec = torch.zeros((N, d_model), dtype=torch.float32)
    order = np.argsort(si_a, kind="stable")
    with torch.no_grad():
        for i in range(0, n_stories, 16):
            sset = list(range(i, min(i + 16, n_stories)))
            hb = ((H[sset].float() - hmu) / hsd).to(device)
            pred = fm(hb)
            for bi, s in enumerate(sset):
                sel = np.where(si_a == s)[0]
                if not len(sel):
                    continue
                jj = torch.tensor(tj_a[sel])
                dd = (hb[bi, jj] - hb[bi, jj - 1]).cpu()
                pp = pred[bi, jj - 1].cpu()
                d_vec[sel] = dd.float()
                r_vec[sel] = (dd - pp).float()
    r_norm = r_vec.norm(dim=-1).numpy()
    delta_norm = d_vec.norm(dim=-1).numpy()

    # RHM Gate-0 statistics, for direct comparison
    R["gate0"]["corr_r_nll"] = float(np.corrcoef(r_norm, nll_a)[0, 1])
    R["gate0"]["r2_r_nll"] = _r2(r_norm, nll_a)
    R["gate0"]["corr_r_ent"] = float(np.corrcoef(r_norm, ent_a)[0, 1])
    R["gate0"]["fm_frac_var_explained"] = float(
        1.0 - (r_vec ** 2).sum().item() / max((d_vec ** 2).sum().item(), 1e-9))
    print(f"Gate 0: corr(r,nll)={R['gate0']['corr_r_nll']:+.3f} "
          f"R2={R['gate0']['r2_r_nll']:.3f} "
          f"FM var expl={R['gate0']['fm_frac_var_explained']:.3f}", flush=True)

    # ------------------------------------------------- prompted (ungrounded)
    # A readout that needs no probe and therefore no ground-truth latents. RHM
    # left this open ("whether the probe can be withdrawn after training is
    # untested"); here it is answered directly.
    ans_ids = {k: [tok.encode(w, add_special_tokens=False)[0]
                   for w in lex.slot_values[k]] for k in C.SLOTS}
    qry_ids = {k: tok.encode(QUERIES[k], add_special_tokens=False) for k in C.SLOTS}
    prompt_stories = list(idx_te[:min(n_prompt_stories, len(idx_te))])
    pmask = np.isin(si_a, prompt_stories)
    p_idx = np.where(pmask)[0]
    qp_b = np.zeros((N, n_slots, n_val)); qp_a = np.zeros((N, n_slots, n_val))
    reqs = []
    for i in p_idx:
        for k in range(n_slots):
            reqs.append((i, k, 0)); reqs.append((i, k, 1))
    have_prompt = reuse_cache and os.path.exists(prompt_path)
    if have_prompt:
        blob = torch.load(prompt_path, map_location="cpu", weights_only=False)
        qp_b, qp_a = blob["qp_b"], blob["qp_a"]
        reqs = []
        print(f"loaded cached prompted readout {prompt_path}", flush=True)
    print(f"prompted readout: {len(reqs)} queries over {len(prompt_stories)} "
          f"stories", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(reqs), prompt_batch):
            chunk = reqs[b0:b0 + prompt_batch]
            seqs = []
            for (i, k, st_) in chunk:
                pre = enc["input_ids"][si_a[i]][:tj_a[i] + st_]
                seqs.append(pre + qry_ids[C.SLOTS[k]])
            L = max(len(s) for s in seqs)
            bi = torch.full((len(seqs), L), tok.eos_token_id or 0, dtype=torch.long)
            bm = torch.zeros((len(seqs), L), dtype=torch.long)
            for r_, s_ in enumerate(seqs):            # left pad
                bi[r_, L - len(s_):] = torch.tensor(s_)
                bm[r_, L - len(s_):] = 1
            pos_ids = (bm.cumsum(-1) - 1).clamp(min=0)
            # `logits_to_keep=1` is load-bearing, not an optimisation: the full
            # (B, T, 128k) logit tensor is 7.5 GB at B=128 and OOMs an L4.
            out = model(input_ids=bi.to(device), attention_mask=bm.to(device),
                        position_ids=pos_ids.to(device), logits_to_keep=1)
            lg = out.logits[:, -1].float()
            for r_, (i, k, st_) in enumerate(chunk):
                sub = lg[r_, ans_ids[C.SLOTS[k]]]
                p = torch.softmax(sub, -1).cpu().numpy()
                (qp_a if st_ else qp_b)[i, k] = p
            if b0 % (prompt_batch * 100) == 0:
                print(f"  prompt {b0}/{len(reqs)}  {time.time()-t0:.0f}s",
                      flush=True)
    if not have_prompt:
        torch.save({"qp_b": qp_b, "qp_a": qp_a}, prompt_path)
        volume.commit()
    print(f"prompted readout done in {time.time()-t0:.0f}s", flush=True)

    M_pt = np.zeros(N); Mpm_pt = np.zeros(N)
    negH_b_pt = np.zeros(N); negH_a_pt = np.zeros(N)
    if len(p_idx):
        a, b, c, d = belief_readouts(qp_b[p_idx], qp_a[p_idx])
        M_pt[p_idx], Mpm_pt[p_idx] = a, b
        negH_b_pt[p_idx], negH_a_pt[p_idx] = c, d
        pacc, pceil = {}, {}
        for k, sk in enumerate(C.SLOTS):
            pred = qp_a[p_idx, k].argmax(-1)
            pacc[sk] = float((pred == Z[si_a[p_idx], k].numpy()).mean())
            pceil[sk] = float(np.mean([1.0 / len(live_a_all[i][k]) for i in p_idx]))
        R["gate0"]["prompt_acc"] = pacc
        R["gate0"]["prompt_ceiling"] = pceil
        R["gate0"]["prompt_frac_of_ceiling"] = {k: pacc[k] / pceil[k] for k in pacc}
        print("prompt acc / ceiling:",
              {k: f"{pacc[k]:.3f}/{pceil[k]:.3f}" for k in pacc}, flush=True)

    # ------------------------------------------------ cross-fit discriminants
    # Trained on TRAIN stories only, per contrast, evaluated on TEST.
    _disc_cache = {}

    def make_disc(X, pos_f, neg_f, name, restrict=None, shuffle=None):
        """Supervised linear decodability of the family from a state vector.

        This is NOT a belief readout -- it is trained on the oracle label, so it
        measures how legibly the distinction is written into the residual
        stream. Whether the model's own posterior expresses it is a different
        question (`MNIST_LOCAL_LOSS`: legibility is not self-knowledge).
        `shuffle=True` permutes the training labels within
        (frame x nlive x nment) and must read ~0.5.
        """
        if name in _disc_cache:
            return _disc_cache[name]
        trm = (split_a == 0) & np.isin(fams, pos_f + neg_f)
        if restrict is not None:
            trm = trm & restrict     # else the discriminant just learns the
                                     # prefix confound `restrict` removes
        tr = torch.from_numpy(trm)
        te = torch.from_numpy(split_a == 2)
        ylab = np.isin(fams[trm], pos_f).astype(np.float32)
        if shuffle == "strat":
            st_ = _cross(frame_codes[trm], np.asarray(nlive_b)[trm],
                         nment_a[trm])
            ylab = ylab[_shuffle_within(st_, swap_rng)]
        elif shuffle == "all":
            ylab = ylab[swap_rng.permutation(ylab.size)]
        y = torch.tensor(ylab)
        s = fit_discriminant(X[tr], y, X[te], steps=disc_steps,
                             seed=seed + sum(ord(c) for c in name) % 997,
                             device=device)
        out = np.full(N, np.nan)
        if s is not None:
            out[te.numpy()] = s
        _disc_cache[name] = out
        return out

    # ----------------------------------------------------------- the gates
    swap_rng = np.random.RandomState(seed + 17)
    frame_codes = _codes(list(frames))
    # the before-state swap stays inside (frame x split) so the guard can only
    # destroy sequence-specific content, never leak across the split
    swap = _shuffle_within(_cross(frame_codes, split_a), swap_rng)
    M_swap, _, _, _ = belief_readouts(lg_b[swap], lg_a)
    M_swap_pt = np.zeros(N)
    if len(p_idx):
        keep = np.isin(swap[p_idx], p_idx)      # only swap onto a prompted state
        pk = p_idx[keep]
        a_, _, _, _ = belief_readouts(qp_b[swap[pk]], qp_a[pk])
        M_swap_pt[pk] = a_

    scores_all = {
        "B_oracle": Bv,                       # ceiling: defines the families
        "H_post_before": Hpb,                 # oracle prefix uncertainty
        "gen_surprisal": gsurp,               # oracle surprisal (pinned by design)
        "nll": nll_a,                         # model surprisal, the incumbent
        "out_entropy": ent_a,                 # the STRONG incumbent
        "delta_norm": delta_norm,             # no forward model at all
        "r_norm": r_norm,                     # the Gate-0 residual
        "M_probe": M_pr,
        "Mpm_probe": Mpm_pr,
        "negH_before_probe": negH_b_pr,
        "negH_after_probe": negH_a_pr,
        "M_swap_probe": M_swap,               # guard
        "M_swap_prompt": M_swap_pt,           # guard
        "M_prompt": M_pt,
        "Mpm_prompt": Mpm_pt,
        "negH_before_prompt": negH_b_pt,
        "negH_after_prompt": negH_a_pt,
    }

    PROMPT_KEYS = ("M_prompt", "Mpm_prompt", "negH_before_prompt",
                   "negH_after_prompt", "M_swap_prompt")

    def _strat_for(key, idx):
        if key == "frame":
            return frame_codes[idx]
        if key == "tokbin":
            return _strata(tokfrac_a[idx])
        if key == "nlive":
            return np.asarray(nlive_b)[idx]
        if key == "nment":
            return nment_a[idx]
        if key == "tokid":
            return tokid_a[idx]
        # nll and output entropy are continuous; 8 quantile bins left the
        # self-matched guard at 0.62 in the smoke, which is not "held fixed".
        return _strata(scores_all[key][idx], n_bins=24)

    def run_contrast(name, pos_f, neg_f, columns, prompt_only=False,
                     restrict=None):
        """AUC of every readout at separating the two families, under each
        matching column. Every column reports its own guard.

        `restrict` exists for Gate B1: when no value has been eliminated yet
        (`n_live_before == 4`) a `negate_dead` is impossible, so the prefix
        alone predicts family membership. Restricting to states where both
        families are reachable -- and then matching on `n_live_before` -- is
        what makes the contrast prefix-blind.
        """
        sel = np.isin(fams, pos_f + neg_f) & (split_a == 2)
        if restrict is not None:
            sel = sel & restrict
        if prompt_only:
            sel = sel & pmask
        idx = np.where(sel)[0]
        pos = np.isin(fams[idx], pos_f)
        block = {"n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
                 "prompt_only": prompt_only, "columns": {}}
        if pos.sum() < 30 or (~pos).sum() < 30:
            block["skipped"] = f"too few ({pos.sum()}/{(~pos).sum()})"
            return block

        sc = {k: v[idx] for k, v in scores_all.items()
              if prompt_only or k not in PROMPT_KEYS}
        # directional readouts, fit on TRAIN stories for this contrast
        for nm, X in (("r_dir", r_vec), ("h_after_dir", h_after),
                      ("h_before_dir", h_before)):
            sc[nm] = make_disc(X, pos_f, neg_f, name.split("_")[0] + nm,
                               restrict=restrict)[idx]
        # Three guards on the supervised discriminant, in increasing strength.
        # `shuf` permutes labels within (frame x nlive x nment): it is a WEAK
        # guard when the true feature is near-perfectly decodable, because a
        # finite-sample residual correlation preferentially recruits the
        # dominant direction. `shufall` permutes labels globally. `h_swap` is
        # the decisive one: identical procedure, identical labels, identical
        # capacity, but the state vectors are permuted within (frame x split)
        # so they carry the right distribution and the wrong content. It is the
        # discriminant analogue of `M_swap` and it must read ~0.5.
        sc["GUARDdir_shuf_strat"] = make_disc(
            h_after, pos_f, neg_f, name.split("_")[0] + "shuf",
            restrict=restrict, shuffle="strat")[idx]
        sc["GUARDdir_shuf_all"] = make_disc(
            h_after, pos_f, neg_f, name.split("_")[0] + "shufall",
            restrict=restrict, shuffle="all")[idx]
        sc["GUARDdir_h_swap"] = make_disc(
            h_after[torch.from_numpy(swap)], pos_f, neg_f,
            name.split("_")[0] + "hswap", restrict=restrict)[idx]

        for cname, keys in columns.items():
            strata = (np.zeros(len(idx), dtype=np.int64) if not keys
                      else _cross(*[_strat_for(k, idx) for k in keys]))
            col = {}
            for nm, v in sc.items():
                vv = np.where(np.isnan(v), 0.0, v)
                a, nu, npair = _stratified_auc(vv, pos, strata)
                col[nm] = None if math.isnan(a) else round(float(a), 4)
            gsh = sc["M_probe"][_shuffle_within(strata, swap_rng)]
            a, nu, npair = _stratified_auc(gsh, pos, strata)
            col["GUARD_M_shuffled"] = None if math.isnan(a) else round(float(a), 4)
            col["_n_strata_used"] = nu
            col["_n_pairs"] = npair
            block["columns"][cname] = col
        return block

    COLS_B1 = {
        "raw": [],
        "nll": ["nll"],
        "out_entropy": ["out_entropy"],
        "frame": ["frame"],
        "nment": ["nment"],
        "frame_x_nlive": ["frame", "nlive"],
        "frame_x_nlive_x_nment": ["frame", "nlive", "nment"],
        # the full control: syntactic frame, the slot's live-set size, how many
        # times the arriving word already occurred in this story, and the
        # model's own surprisal and predictive entropy
        "FULL": ["frame", "nlive", "nment", "nll"],
        "FULL_x_outent": ["frame", "nlive", "nment", "nll", "out_entropy"],
        # strictest: same word, same frame, same live-set size, same mention
        # count -- only the prefix history differs
        "tokid_x_frame_x_nlive_x_nment": ["tokid", "frame", "nlive", "nment"],
        "STRICT": ["tokid", "frame", "nlive", "nment", "nll"],
    }
    COLS_B2 = {
        "raw": [],
        "nll": ["nll"],
        "out_entropy": ["out_entropy"],
        "nll_x_outent": ["nll", "out_entropy"],
        "tokbin": ["tokbin"],
        "nll_x_outent_x_tokbin": ["nll", "out_entropy", "tokbin"],
        "nment_x_nll_x_outent": ["nment", "nll", "out_entropy"],
        "Hpost_x_nll_x_outent": ["H_post_before", "nll", "out_entropy"],
        "frame": ["frame"],           # expected degenerate: frame => family
    }

    print("\n=== Gate B1: negate_live vs negate_dead ===", flush=True)
    reachable = np.asarray(nlive_b) < C.N_VAL      # both families possible
    R["B1"] = run_contrast("B1", POS_B1, NEG_B1, COLS_B1, restrict=reachable)
    R["B1_prompt"] = run_contrast("B1_prompt", POS_B1, NEG_B1, COLS_B1,
                                  prompt_only=True, restrict=reachable)
    R["B1_unrestricted"] = run_contrast("B1u", POS_B1, NEG_B1, COLS_B1)
    print(json.dumps(R["B1"], indent=1, cls=NumpyEncoder), flush=True)

    # The single most diagnostic split for B1: a dead value was either NAMED by
    # an earlier negation (so a repeat, readable by induction) or killed
    # silently by a narrow clause (never mentioned). If the decodability is
    # ~1.0 only on the named ones, it is induction; if it holds on the silent
    # ones too, the model is tracking the elimination.
    print("\n=== Gate B1 breakdown by kill mechanism x prior mentions ===",
          flush=True)
    bd = {}
    te_all = (split_a == 2) & reachable
    for cell, m_extra in (
            ("dead_named(negate)", kill_a == "negate"),
            ("dead_silent(narrow)", kill_a == "narrow")):
        for nm_lvl in (0, 1, 2, None):
            m_pos = te_all & (fams == "negate_live")
            m_neg = te_all & (fams == "negate_dead") & m_extra
            if nm_lvl is not None:
                m_pos = m_pos & (nment_a == nm_lvl)
                m_neg = m_neg & (nment_a == nm_lvl)
            if m_pos.sum() < 25 or m_neg.sum() < 25:
                continue
            m = m_pos | m_neg
            pos = fams[m] == "negate_live"
            # stratify inside the cell too -- the raw AUC here is confounded
            # (at nment=0, "dead" implies a narrow happened, which fixes nlive)
            cell_str = _cross(tokid_a[m], frame_codes[m],
                              np.asarray(nlive_b)[m])
            row = {"n_live": int(m_pos.sum()), "n_dead": int(m_neg.sum())}
            for nm2, v in (("nll", nll_a), ("out_entropy", ent_a),
                           ("r_norm", r_norm), ("M_probe", M_pr),
                           ("Mpm_probe", Mpm_pr), ("M_prompt", M_pt),
                           ("h_after_dir", _disc_cache.get("B1h_after_dir")),
                           ("h_before_dir", _disc_cache.get("B1h_before_dir")),
                           ("r_dir", _disc_cache.get("B1r_dir")),
                           ("GUARDdir_shuf", _disc_cache.get("B1shuf")),
                           ("GUARDdir_shufall", _disc_cache.get("B1shufall")),
                           ("GUARDdir_h_swap", _disc_cache.get("B1hswap"))):
                if v is None:
                    continue
                vv = np.where(np.isnan(v[m]), 0.0, v[m])
                row[nm2] = round(float(_auc(vv, pos)), 4)
                a_, _, np_ = _stratified_auc(vv, pos, cell_str)
                row[nm2 + "_strat"] = (None if math.isnan(a_)
                                       else round(float(a_), 4))
                row["_n_pairs_strat"] = round(np_)
            bd[f"{cell}|nment={nm_lvl}"] = row
    R["B1_breakdown"] = bd
    print(json.dumps(bd, indent=1), flush=True)

    print("\n=== Gate B2: reveal vs synonym/offtopic ===", flush=True)
    R["B2"] = run_contrast("B2", POS_B2, NEG_B2, COLS_B2)
    R["B2_prompt"] = run_contrast("B2_prompt", POS_B2, NEG_B2, COLS_B2,
                                  prompt_only=True)
    print(json.dumps(R["B2"], indent=1, cls=NumpyEncoder), flush=True)

    print("\n=== Gate B2b: reveal vs mention (value word vs value word) ===",
          flush=True)
    R["B2b"] = run_contrast("B2b", POS_B2b, NEG_B2b, COLS_B2)
    R["B2b_prompt"] = run_contrast("B2b_prompt", POS_B2b, NEG_B2b, COLS_B2,
                                   prompt_only=True)
    print(json.dumps(R["B2b"], indent=1, cls=NumpyEncoder), flush=True)

    print("\n=== Gate B3 (specificity guard): mention vs synonym/offtopic ===",
          flush=True)
    R["B3"] = run_contrast("B3", POS_B3, NEG_B3, COLS_B2)
    print(json.dumps(R["B3"]["columns"].get("raw", {}), indent=1), flush=True)

    print("\n=== Gate E: deduced vs echo ===", flush=True)
    ECOLS = {"raw": [], "nll": ["nll"], "frame": ["frame"]}
    R["E"] = run_contrast("E", POS_E, NEG_E, ECOLS)
    R["E_prompt"] = run_contrast("E_prompt", POS_E, NEG_E, ECOLS,
                                 prompt_only=True)
    print(json.dumps(R["E"], indent=1, cls=NumpyEncoder), flush=True)

    # ---------------------------------------------------------------- Gate A
    val_m = (split_a == 2) & np.isin(
        fams, ("reveal", "reveal_narrow", "negate_live", "negate_dead",
               "echo", "deduced"))
    ga = {}
    for nm, Mv in (("M_probe", M_pr), ("Mpm_probe", Mpm_pr),
                   ("M_prompt", M_pt), ("Mpm_prompt", Mpm_pt),
                   ("r_norm", r_norm)):
        m = (val_m & pmask) if "prompt" in nm else val_m
        if m.sum() < 100:
            continue
        ga[nm] = {
            "partial_r2_M_B_given_nll": round(_partial_r2(Mv[m], Bv[m], nll_a[m]), 4),
            "partial_r2_M_nll_given_B": round(_partial_r2(Mv[m], nll_a[m], Bv[m]), 4),
            "rank_partial_r2_M_B_given_nll":
                round(_partial_r2_rank(Mv[m], Bv[m], nll_a[m]), 4),
            "partial_r2_M_B_given_outent":
                round(_partial_r2(Mv[m], Bv[m], ent_a[m]), 4),
            "n": int(m.sum()),
        }
    # Two nulls, and they say different things. The frame-preserving one keeps
    # the syntactic frame and destroys only the sequence-specific content; if a
    # readout does not beat THAT, whatever it carries is frame, not belief.
    sh = M_pr[_shuffle_within(frame_codes, swap_rng)]
    shr = M_pr[swap_rng.permutation(N)]
    ga["null_within_frame"] = {
        "partial_r2_M_B_given_nll":
            round(_partial_r2(sh[val_m], Bv[val_m], nll_a[val_m]), 4)}
    ga["null_fully_random"] = {
        "partial_r2_M_B_given_nll":
            round(_partial_r2(shr[val_m], Bv[val_m], nll_a[val_m]), 4)}
    ga["R2_B_given_nll"] = round(_r2(Bv[val_m], nll_a[val_m]), 4)
    ga["R2_B_given_gensurp"] = round(_r2(Bv[val_m], gsurp[val_m]), 4)
    R["A"] = ga
    print("\n=== Gate A ===\n" + json.dumps(ga, indent=1), flush=True)

    # ---------------------------------------------------------------- Gate D
    # Variance budget of the FM's target. If E||Delta||^2 and E||r||^2 are equal
    # across families, a loss on the residual's magnitude is family-blind BY
    # CONSTRUCTION, and no precision scalar can recover the distinction.
    gd = {}
    for f in sorted(set(fams)):
        m = (fams == f) & (split_a == 2)
        if m.sum() < 30:
            continue
        gd[f] = {
            "n": int(m.sum()),
            "E_delta_sq": float((d_vec[m] ** 2).sum(-1).mean()),
            "E_r_sq": float((r_vec[m] ** 2).sum(-1).mean()),
            "fm_var_explained": float(
                1 - (r_vec[m] ** 2).sum().item() / max((d_vec[m] ** 2).sum().item(), 1e-9)),
            "mean_nll": float(nll_a[m].mean()),
            "mean_out_entropy": float(ent_a[m].mean()),
        }
    # Across families: does E||r||^2 track the reader's predictive entropy or
    # the oracle's belief revision? If it tracks entropy, a scalar precision on
    # this residual is family-blind by construction, and section 5's proposal
    # cannot work on this object however it is tuned.
    fam_names = sorted(gd)
    if len(fam_names) >= 4:
        er = np.array([gd[f]["E_r_sq"] for f in fam_names])
        he = np.array([gd[f]["mean_out_entropy"] for f in fam_names])
        nl = np.array([gd[f]["mean_nll"] for f in fam_names])
        bb = np.array([float(Bv[(fams == f) & (split_a == 2)].mean())
                       for f in fam_names])
        gd["_across_family"] = {
            "families": fam_names,
            "mean_B": [round(float(x), 4) for x in bb],
            "r2_Ersq_vs_out_entropy": round(_r2(er, he), 4),
            "r2_Ersq_vs_nll": round(_r2(er, nl), 4),
            "r2_Ersq_vs_meanB": round(_r2(er, bb), 4),
        }
    R["D"] = gd
    print("\n=== Gate D (variance budget) ===\n"
          + json.dumps(gd, indent=1), flush=True)

    # ---------------------------------------------------------------- save
    out_dir = f"{DATA_DIR}/chronicle"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/gates_{tag}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(R, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {path}  ({time.time()-t00:.0f}s total)", flush=True)
    return path


@app.local_entrypoint()
def main():
    print("use: modal run -m a2a_forward.conditional_revision.gates::gates --tag smoke ...")
