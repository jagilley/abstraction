"""Arm A -- the reader's OWN predictive law as the belief readout.

Pre-registered here before the first run. Nothing in `gates.py` is touched; this
module reuses its cached activations and its cached prompted readout so every
number below is directly comparable with the published table.

WHY
---
`gates.py` found a dissociation it could not explain: the `negate_live` vs
`negate_dead` distinction is linearly decodable from the post-token residual
stream at **0.99** while *every belief readout reads chance* -- probe `M` 0.494,
prompted `M` 0.530 (swap guard 0.531), model `nll` 0.524 at STRICT. The state
contains it; the model's posterior, as we read it, does not.

Two things about how it was read are candidate explanations, and they are
separable:

1. **The channel.** The prompted readout appends `"\\nQ: Which item is missing?
   A: the"` -- out of distribution for this corpus, and it requires the reader
   to be able to answer a question. Binz et al.'s Box 1 says the *forward pass*
   is the posterior predictive; the natural readout is therefore the model's own
   law over **future text**, not over an interposed QA frame.
2. **The slot sum.** `M_prompt` sums KL over all four slots. The RHM sibling's
   hardest-won gotcha is exactly this -- *"`M` must be read on the ancestor
   chain. Summing over all latent nodes buries the signal in probe noise from
   the ~57 nodes the model does not represent."* A negation clause moves one
   slot; the other three contribute readout jitter only.

THE OBJECT
----------
For slot `k` and reveal-carrier frame `f`, let `S_{k,f}` be that carrier's
literal text up to (not including) its value slot, with synonym segments pinned
to index 0 and a leading `"."` closing the truncated clause. Then

    b_t^{k,f}(v)  =  softmax_v  logits( x_<=t  (+)  S_{k,f} )[ id(value_k[v]) ]

Because the four candidate continuations differ in exactly one token and share
the whole stem, this is *exactly* the model's predictive law conditioned on the
finite event `{the next clause is a reveal of slot k}` -- a genuine conditional
of `p_theta(future | x_<=t)`, needing no probe, no oracle and no QA ability. It
is the finite-sigma-algebra projection of the object
`ideas/temporal_confabulation_test.md` calls the calibration standard, and by
the data-processing inequality any separation it shows is a lower bound on the
separation in the full predictive law.

Readouts, all on the position's **own** slot:

    Dkl        KL( b_{t+1} || b_t )                 the native revision
    Ddrop      log b_t(v_arr) - log b_{t+1}(v_arr)  the SIGNED version: how much
                                                    mass the token took off the
                                                    word that just arrived
    logb_pre   log b_t(v_arr)                       the BEFORE state alone
    negH_pre   -H(b_t)                              prefix uncertainty alone
    Dpm        -log b_t(argmax b_{t+1})             the pointmass variant

`Ddrop` and `logb_pre` are in the pre-registration because of two standing
findings and they are not afterthoughts. The repo's directional-not-scalar
pattern has fired four times (`EMOTION_INJECTION`, the directional-vs-scalar
belief node, `endogenous_teacher`, `sculpt_slip` step 1), so a signed readout is
expected to beat the KL. And the RHM tracking audit found `M_pointmass` recovers
93-99.6% of `M`'s AUC with the increment from actually differencing worth only
+0.017/+0.076 -- so if `logb_pre` carries this on its own, the honest finding is
*"the reader's predictive law tracks the live set"* (a level) and **not**
*"the reader registers the revision"* (a shift). Reporting them apart is the
point; collapsing them would repeat a mistake this program has already made.

GUARDS (a row is not interpretable without them)
------------------------------------------------
    Dkl_other        same object on slot (k+1)%4, which the token says nothing
                     about. Specificity: must read ~0.5.
    *_swap           b_t permuted within frame, b_{t+1} kept -- the analogue of
                     `M_swap`. Right distribution, wrong content.
    Dkl_own_f1       the second reveal carrier. Frame disagreement means the
                     readout is a frame artifact, not a belief ("pin both ends
                     of the probe", from `rule_family`'s void run).
    M_prompt_own     the CACHED prompted readout, restricted to the own slot.
                     Separates explanation 2 from explanation 1 for free.
    h_*_dir          the published state discriminants, reproduced so the new
                     rows sit in the table they have to beat.
    permute inputs, never labels (`gates.py`'s hardest gotcha).

PRE-REGISTERED PREDICTIONS
--------------------------
    Dkl_other, *_swap                     ~0.500
    Dkl_own vs Dkl_own_f1                 agree; if they disagree, stop reading
    logb_pre                              > 0.5 -- the live set is in the law
    Ddrop_own                             the largest of the native readouts
    Dkl_own at STRICT                     the load-bearing cell. > swap guard =>
                                          the 0.99 in the state has a native
                                          channel and the published belief null
                                          was an instrument artifact. ~0.5 =>
                                          legibility-is-not-self-knowledge holds
                                          against the model's own forward law,
                                          which is the strongest form of that
                                          finding we can get.

Known limit, stated up front: at the `before` state the prefix is cut mid-NP
("...was not the") and at the `after` state it is cut after the value word, so
the stem attaches to two grammatically different cuts. This is *identical* for
`negate_live` and `negate_dead` (one template, one clause position, arriving
word matched exactly by `tokid` in STRICT), so it cancels in B1 -- the primary
test. It does **not** cancel across heterogeneous families, so Gate A here is
secondary and is reported with that caveat.

Reproduction is in the README; smoke with `--n-pred-stories 40`.
"""

import json
import math
import os

import modal

from a2a_forward.conditional_revision.shared import (app, volume, DATA_DIR,
                                                     image, NumpyEncoder)

POS_B1, NEG_B1 = ("negate_live",), ("negate_dead",)

# families whose analysis token names a value of a z-slot, so an own-slot
# predictive readout is defined. `synonym` has no slot and `offtopic`'s slot is
# a decoy variable outside z.
SLOTTED = ("reveal", "reveal_narrow", "negate_live", "negate_dead",
           "echo", "deduced", "mention")


# Frame 2 is lexically distant from both corpus carriers on purpose. The
# corpus's two reveal carriers are the affirmative TWINS of the negation
# carriers ("...was the X" vs "...was not the X"), so a large after-state mass
# on the just-negated word could be template copying rather than a belief. This
# frame shares no content words with the negation clause, so a copy effect that
# survives it is not template-driven.
NEUTRAL_STEMS = {
    "person": ". In the end the visitor turned out to be",
    "object": ". In the end the missing item turned out to be the",
    "place":  ". In the end the search began in the",
    "hour":   ". In the end the door had been locked at",
}


def reveal_stems(lex):
    """Continuation stems, per (slot, carrier index).

    Indices 0 and 1 are the corpus's own reveal carriers -- their literals up to
    the value slot, with `syn` segments pinned to index 0 so the stem is
    deterministic, prefixed by "." to close whatever clause the prefix was cut
    inside. Index 2 is `NEUTRAL_STEMS`.
    """
    from a2a_forward.conditional_revision import corpus as C
    out = {}
    for sl in C.SLOTS:
        for ti, segs in enumerate(C.CARRIERS[sl]["pos"]):
            buf = []
            for kind, val in segs:
                if kind == "val":
                    assert val == sl
                    break
                buf.append(val if kind == "lit" else lex.synsets[val][0])
            out[(sl, ti)] = "." + "".join(buf)
        out[(sl, 2)] = NEUTRAL_STEMS[sl]
    return out


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=21600,
              memory=65536, image=image)
def predictive(
    model_name: str = "unsloth/Llama-3.2-1B",
    layer: int = 12,
    probe_layer: int = -1,
    n_stories: int = 3000,
    n_clauses: int = 14,
    seed: int = 42,
    tag: str = "p1",
    lm_batch: int = 8,
    pred_batch: int = 48,
    disc_steps: int = 800,
    n_pred_stories: int = 0,        # 0 = every test story
    n_prompt_stories: int = 700,    # must match the cached prompted readout
    reuse_cache: bool = True,
):
    import random
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from a2a_forward.conditional_revision import corpus as C
    from a2a_forward.conditional_revision.fm import fit_discriminant
    from a2a_forward.conditional_revision.stats import (
        _r2, _partial_r2, _partial_r2_rank, _auc, _stratified_auc, _strata,
        _cross, _codes, _shuffle_within)

    t00 = time.time()
    device = "cuda"
    torch.manual_seed(seed)
    R = {"config": dict(model_name=model_name, layer=layer, n_stories=n_stories,
                        n_clauses=n_clauses, seed=seed, tag=tag,
                        n_pred_stories=n_pred_stories,
                        n_prompt_stories=n_prompt_stories)}

    # ---------------------------------------------------------------- reader
    print(f"loading {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.hidden_size
    n_layer_lm = model.config.num_hidden_layers
    probe_layer = n_layer_lm if probe_layer < 0 else probe_layer

    def one_tok(w):
        return len(tok.encode(w, add_special_tokens=False)) == 1

    lex = C.build_lexicon(one_tok)

    # ------------------------- corpus (bit-identical to gates.py at same seed)
    rng = random.Random(seed)
    stories = [C.generate_story(rng, lex, n_clauses=n_clauses)
               for _ in range(n_stories)]
    enc = tok([s.text for s in stories], return_offsets_mapping=True,
              add_special_tokens=True)
    lens = [len(x) for x in enc["input_ids"]]
    T_max = max(lens)
    n_align_fail = 0
    aligned = []
    for si, st in enumerate(stories):
        offs = enc["offset_mapping"][si]
        starts = {o[0]: k for k, o in enumerate(offs)}
        rec = []
        for p in st.positions:
            if not p.analysis:
                continue
            k = starts.get(p.char_start)
            if k is None:
                k = starts.get(p.char_start - 1)
            if k is None or k == 0 or k >= lens[si] - 1:
                n_align_fail += 1
                continue
            if st.text[offs[k][0]:offs[k][1]].strip() != p.word.strip():
                n_align_fail += 1
                continue
            rec.append((p, k))
        aligned.append(rec)
    n_pos_total = sum(len(r) for r in aligned)
    print(f"aligned {n_pos_total} analysis tokens, {n_align_fail} failures, "
          f"T_max={T_max}", flush=True)
    assert n_align_fail / max(n_pos_total + n_align_fail, 1) < 0.01

    # ------------------------------------------------- cached reader forward
    ck = (f"{model_name.split('/')[-1]}_L{layer}P{probe_layer}"
          f"_n{n_stories}_c{n_clauses}_s{seed}")
    cache_dir = f"{DATA_DIR}/chronicle/cache"
    acts_path = f"{cache_dir}/acts_{ck}.pt"
    prompt_path = f"{cache_dir}/prompt_{ck}_p{n_prompt_stories}.pt"
    assert os.path.exists(acts_path), (
        f"{acts_path} missing -- run gates.py first; this module never "
        f"recomputes the reader forward, so it cannot silently diverge from it")
    blob = torch.load(acts_path, map_location="cpu", weights_only=False)
    H, NLL, ENT = blob["H"], blob["NLL"], blob["ENT"]
    blob["HP"] = None                       # the probe layer is unused here
    del blob
    print(f"loaded {acts_path}  H={tuple(H.shape)}", flush=True)

    # ------------------------------------------- splits (identical to gates)
    perm = np.random.RandomState(seed).permutation(n_stories)
    n_tr, n_ca = int(0.60 * n_stories), int(0.10 * n_stories)
    idx_tr = np.sort(perm[:n_tr])
    idx_ca = np.sort(perm[n_tr:n_tr + n_ca])
    idx_te = np.sort(perm[n_tr + n_ca:])
    split_of = {}
    for s in idx_tr: split_of[int(s)] = 0
    for s in idx_ca: split_of[int(s)] = 1
    for s in idx_te: split_of[int(s)] = 2

    n_slots, n_val = len(C.SLOTS), C.N_VAL
    Z = torch.tensor([[st.z[k] for k in C.SLOTS] for st in stories])
    slot_pos = {k: i for i, k in enumerate(C.SLOTS)}

    # ----------- the analysis records, in gates.py order (the prompt cache is
    # indexed by this order, so it must not diverge)
    fams, slots_, frames, Bv, gsurp, Hpb, nlive_b = [], [], [], [], [], [], []
    si_a, tj_a, nment_a, kill_a, value_idx_a, split_a = [], [], [], [], [], []
    live_a_all = []
    for si in range(n_stories):
        for p, k in aligned[si]:
            fams.append(p.family); slots_.append(p.slot or "-")
            frames.append(p.frame); Bv.append(p.B); gsurp.append(p.gen_surprisal)
            Hpb.append(p.H_post_before); nlive_b.append(p.n_live_before)
            nment_a.append(min(p.n_prior_mentions, 2))
            value_idx_a.append(-1 if p.value_idx is None else p.value_idx)
            kill_a.append(p.kill_mech or "-")
            si_a.append(si); tj_a.append(k)
            live_a_all.append(p.live_after)
            split_a.append(split_of[si])
    si_a = np.array(si_a); tj_a = np.array(tj_a)
    fams = np.array(fams); frames = np.array(frames); slots_ = np.array(slots_)
    Bv = np.array(Bv); gsurp = np.array(gsurp); Hpb = np.array(Hpb)
    nlive_b = np.array(nlive_b); nment_a = np.array(nment_a)
    kill_a = np.array(kill_a); value_idx_a = np.array(value_idx_a)
    split_a = np.array(split_a)
    tokid_a = _codes([f"{a}/{b}" for a, b in zip(slots_, value_idx_a.astype(str))])
    frame_codes = _codes(list(frames))
    N = len(fams)
    nll_a = NLL[si_a, tj_a].numpy()
    ent_a = ENT[si_a, tj_a].numpy()
    print(f"N={N} analysis positions", flush=True)

    # --------------------------------------------- which positions to query
    te_stories = idx_te if n_pred_stories <= 0 else idx_te[:n_pred_stories]
    pred_mask = (np.isin(fams, SLOTTED) & np.isin(si_a, te_stories)
                 & np.isin(slots_, C.SLOTS))
    pred_idx = np.where(pred_mask)[0]
    print(f"predictive readout: {len(pred_idx)} positions over "
          f"{len(te_stories)} test stories", flush=True)

    stems = reveal_stems(lex)
    stem_ids = {key: tok.encode(s, add_special_tokens=False)
                for key, s in stems.items()}
    ans_ids = {k: [tok.encode(w, add_special_tokens=False)[0]
                   for w in lex.slot_values[k]] for k in C.SLOTS}
    R["stems"] = {f"{a}#{b}": s for (a, b), s in stems.items()}
    print("stems:", json.dumps(R["stems"], indent=1), flush=True)

    # variants -> (which slot, carrier index)
    #   own0  the primary readout, corpus reveal carrier 0
    #   own1  corpus reveal carrier 1 -- the frame-agreement check
    #   own2  the lexically-distant neutral frame -- separates a belief from
    #         copying the negation clause's own template
    #   oth0  the specificity guard: a slot this token says nothing about
    VARIANTS = ("own0", "own1", "own2", "oth0")
    VAR_TI = {"own0": 0, "own1": 1, "own2": 2, "oth0": 0}
    qn = {v: (np.zeros((N, n_val)), np.zeros((N, n_val))) for v in VARIANTS}

    def variant_slot(i, v):
        base = slot_pos[str(slots_[i])]
        return C.SLOTS[(base + 1) % n_slots] if v == "oth0" else str(slots_[i])

    reqs = []
    for i in pred_idx:
        for v in VARIANTS:
            for st_ in (0, 1):
                reqs.append((int(i), v, variant_slot(i, v), VAR_TI[v], st_))

    # `r2` versions the stem definition: changing `reveal_stems` must not reuse
    # a cache built under the old stems.
    pred_path = (f"{cache_dir}/pred_r2_{ck}_n{len(te_stories)}"
                 f"_v{len(VARIANTS)}.pt")
    have = reuse_cache and os.path.exists(pred_path)
    if have:
        cached = torch.load(pred_path, map_location="cpu", weights_only=False)
        qn = {v: (cached[v][0], cached[v][1]) for v in VARIANTS}
        reqs = []
        print(f"loaded cached predictive readout {pred_path}", flush=True)
    print(f"{len(reqs)} queries", flush=True)

    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(reqs), pred_batch):
            chunk = reqs[b0:b0 + pred_batch]
            seqs = [enc["input_ids"][si_a[i]][:tj_a[i] + st_]
                    + stem_ids[(sl, ti)] for (i, v, sl, ti, st_) in chunk]
            L = max(len(s) for s in seqs)
            bi = torch.full((len(seqs), L), tok.eos_token_id or 0,
                            dtype=torch.long)
            bm = torch.zeros((len(seqs), L), dtype=torch.long)
            for r_, s_ in enumerate(seqs):          # left pad
                bi[r_, L - len(s_):] = torch.tensor(s_)
                bm[r_, L - len(s_):] = 1
            pos_ids = (bm.cumsum(-1) - 1).clamp(min=0)
            # logits_to_keep=1: the full (B, T, 128k) tensor OOMs an L4
            out = model(input_ids=bi.to(device), attention_mask=bm.to(device),
                        position_ids=pos_ids.to(device), logits_to_keep=1)
            lg = out.logits[:, -1].float()
            for r_, (i, v, sl, ti, st_) in enumerate(chunk):
                p = torch.softmax(lg[r_, ans_ids[sl]], -1).cpu().numpy()
                qn[v][st_][i] = p
            if b0 % (pred_batch * 200) == 0:
                print(f"  pred {b0}/{len(reqs)}  {time.time()-t0:.0f}s",
                      flush=True)
    if not have and len(pred_idx):
        torch.save({v: list(qn[v]) for v in VARIANTS}, pred_path)
        volume.commit()
    print(f"predictive readout done in {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------- readouts
    EPS = 1e-30

    def readouts(qb, qa, v_arr):
        """qb, qa: (n, 4) before/after; v_arr: index of the arriving value."""
        qb = np.clip(qb, EPS, None); qa = np.clip(qa, EPS, None)
        kl = np.sum(qa * (np.log(qa) - np.log(qb)), axis=-1)
        am = qa.argmax(-1)
        pm = -np.log(np.take_along_axis(qb, am[:, None], -1).squeeze(-1))
        Hb = -np.sum(qb * np.log(qb), axis=-1)
        Ha = -np.sum(qa * np.log(qa), axis=-1)
        ok = v_arr >= 0
        lb = np.zeros(len(qb)); la = np.zeros(len(qb))
        lb[ok] = np.log(qb[ok, v_arr[ok]])
        la[ok] = np.log(qa[ok, v_arr[ok]])
        return dict(kl=kl, pm=pm, negH_pre=-Hb, negH_post=-Ha,
                    logb_pre=lb, logb_post=la, drop=lb - la)

    # the swap guard: permute the BEFORE state within frame, keep the after
    # state. Right distribution, wrong content. Built inside the queried set so
    # it can never pull an all-zero row.
    swap_rng = np.random.RandomState(seed + 17)
    swp_local = _shuffle_within(frame_codes[pred_idx], swap_rng)

    scores = {
        "B_oracle": Bv, "H_post_before": Hpb, "gen_surprisal": gsurp,
        "nll": nll_a, "out_entropy": ent_a,
    }
    diag = {}
    for v in VARIANTS:
        qb, qa = qn[v][0][pred_idx], qn[v][1][pred_idx]
        va = value_idx_a[pred_idx] if v != "oth0" else np.full(len(pred_idx), -1)
        rd = readouts(qb, qa, va)
        rs = readouts(qb[swp_local], qa, va)
        for nm, arr in rd.items():
            full = np.zeros(N); full[pred_idx] = arr
            scores[f"{nm}_{v}"] = full
        for nm in ("kl", "drop"):
            full = np.zeros(N); full[pred_idx] = rs[nm]
            scores[f"GUARD_{nm}_swap_{v}"] = full
        # accuracy of the native law against the exact Bayes ceiling, and the
        # induction tell: how much mass the after-state puts on the word that
        # just arrived (a negation says NOT that word).
        if v != "oth0":
            pred = qa.argmax(-1)
            kcol = np.array([slot_pos[str(s)] for s in slots_[pred_idx]])
            truth = Z.numpy()[si_a[pred_idx], kcol]
            ceil = np.array([1.0 / len(live_a_all[i][slot_pos[str(slots_[i])]])
                             for i in pred_idx])
            hit = (pred == truth)
            d = {"acc": round(float(hit.mean()), 4),
                 "bayes_ceiling": round(float(ceil.mean()), 4),
                 "frac_of_ceiling": round(float(hit.mean() / ceil.mean()), 4),
                 "mean_H_before": round(float(-rd["negH_pre"].mean()), 4),
                 "mean_kl": round(float(rd["kl"].mean()), 4)}
            # The induction tell, split by family. `reveal` names the TRUE
            # value, so a high after-mass there is correct belief; a negation
            # names a value the scene is NOT, so a high after-mass there is
            # copying. If the two are equal, the readout is recency, not belief.
            fp = fams[pred_idx]
            for f in sorted(set(fp)):
                mf = fp == f
                if mf.sum() < 20:
                    continue
                d[f"arr_mass|{f}"] = {
                    "n": int(mf.sum()),
                    "before": round(float(np.exp(rd["logb_pre"][mf]).mean()), 4),
                    "after": round(float(np.exp(rd["logb_post"][mf]).mean()), 4),
                    "acc": round(float(hit[mf].mean()), 4),
                    "ceiling": round(float(ceil[mf].mean()), 4),
                }
            diag[v] = d
    R["readout_diagnostics"] = diag
    # frame agreement -- if the two carriers disagree, the readout is a frame
    # artifact and nothing below is interpretable
    m = pred_idx
    R["frame_agreement"] = {}
    for a, b in (("own0", "own1"), ("own0", "own2"), ("own1", "own2")):
        for nm in ("kl", "logb_pre", "drop", "negH_pre"):
            R["frame_agreement"][f"corr_{nm}_{a}_{b}"] = round(float(np.corrcoef(
                scores[f"{nm}_{a}"][m], scores[f"{nm}_{b}"][m])[0, 1]), 4)
    print("\ndiagnostics:", json.dumps(diag, indent=1), flush=True)
    print("frame agreement:", R["frame_agreement"], flush=True)

    # ------------------- the cached PROMPTED readout, restricted to own slot
    # Separates "the QA channel was wrong" from "the slot sum was wrong" using
    # a file that already exists.
    prompt_stories = list(idx_te[:min(n_prompt_stories, len(idx_te))])
    pmask = np.isin(si_a, prompt_stories)
    if os.path.exists(prompt_path):
        pb = torch.load(prompt_path, map_location="cpu", weights_only=False)
        qp_b, qp_a = pb["qp_b"], pb["qp_a"]
        pm_idx = np.where(pmask & pred_mask)[0]
        ks = np.array([slot_pos[str(s)] for s in slots_[pm_idx]])
        rd = readouts(qp_b[pm_idx, ks], qp_a[pm_idx, ks], value_idx_a[pm_idx])
        for nm in ("kl", "drop", "logb_pre"):
            full = np.zeros(N); full[pm_idx] = rd[nm]
            scores[f"{nm}_promptown"] = full
        # and the published all-slot version, for the head-to-head
        qb4 = np.clip(qp_b[pm_idx], EPS, None); qa4 = np.clip(qp_a[pm_idx], EPS, None)
        full = np.zeros(N)
        full[pm_idx] = np.sum(qa4 * (np.log(qa4) - np.log(qb4)), axis=(1, 2))
        scores["kl_prompt_allslot"] = full
        R["prompt_head_to_head"] = {"n": int(len(pm_idx))}
        print(f"prompted head-to-head on {len(pm_idx)} positions", flush=True)
    else:
        print(f"NOTE: {prompt_path} absent; skipping prompted head-to-head",
              flush=True)

    # ------------------------------------- the published state discriminants
    h_before = H[si_a, tj_a - 1]
    h_after = H[si_a, tj_a]
    reachable = nlive_b < C.N_VAL
    swap_all = _shuffle_within(_cross(frame_codes, split_a),
                               np.random.RandomState(seed + 17))

    def disc(X, name, shuffle=False):
        trm = (split_a == 0) & np.isin(fams, POS_B1 + NEG_B1) & reachable
        y = np.isin(fams[trm], POS_B1).astype(np.float32)
        if shuffle:
            st_ = _cross(frame_codes[trm], nlive_b[trm], nment_a[trm])
            y = y[_shuffle_within(st_, np.random.RandomState(seed + 23))]
        s = fit_discriminant(X[torch.from_numpy(trm)], torch.tensor(y),
                             X[torch.from_numpy(split_a == 2)],
                             steps=disc_steps, seed=seed + len(name),
                             device=device)
        out = np.full(N, np.nan)
        if s is not None:
            out[split_a == 2] = s
        return out

    scores["h_after_dir"] = disc(h_after, "hafter")
    scores["h_before_dir"] = disc(h_before, "hbefore")
    scores["GUARD_h_swap_dir"] = disc(h_after[torch.from_numpy(swap_all)], "hswap")
    print("state discriminants fit", flush=True)

    # ------------------------------------------------------- Gate B1, primary
    def strat_for(key, idx):
        if key == "frame":
            return frame_codes[idx]
        if key == "nlive":
            return nlive_b[idx]
        if key == "nment":
            return nment_a[idx]
        if key == "tokid":
            return tokid_a[idx]
        return _strata(scores[key][idx], n_bins=24)

    COLS = {
        "raw": [], "nll": ["nll"], "out_entropy": ["out_entropy"],
        "frame": ["frame"], "nment": ["nment"],
        "frame_x_nlive": ["frame", "nlive"],
        "frame_x_nlive_x_nment": ["frame", "nlive", "nment"],
        "FULL": ["frame", "nlive", "nment", "nll"],
        "FULL_x_outent": ["frame", "nlive", "nment", "nll", "out_entropy"],
        "tokid_x_frame_x_nlive_x_nment": ["tokid", "frame", "nlive", "nment"],
        "STRICT": ["tokid", "frame", "nlive", "nment", "nll"],
    }

    def run_contrast(name, pos_f, neg_f, cols, restrict=None, extra=None):
        sel = np.isin(fams, pos_f + neg_f) & (split_a == 2) & pred_mask
        if restrict is not None:
            sel = sel & restrict
        if extra is not None:
            sel = sel & extra
        idx = np.where(sel)[0]
        pos = np.isin(fams[idx], pos_f)
        block = {"n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
                 "columns": {}}
        if pos.sum() < 25 or (~pos).sum() < 25:
            block["skipped"] = f"too few ({pos.sum()}/{(~pos).sum()})"
            return block
        for cname, keys in cols.items():
            strata = (np.zeros(len(idx), dtype=np.int64) if not keys
                      else _cross(*[strat_for(k, idx) for k in keys]))
            col = {}
            for nm, v in scores.items():
                vv = np.where(np.isnan(v[idx]), 0.0, v[idx])
                if np.allclose(vv, 0.0):
                    continue
                a, nu, npair = _stratified_auc(vv, pos, strata)
                col[nm] = None if math.isnan(a) else round(float(a), 4)
            # label-side guard, kept only because the published table has it;
            # `gates.py` established it is the WEAK one -- the swap rows above
            # are the decisive controls.
            gsh = scores["kl_own0"][idx][_shuffle_within(strata, swap_rng)]
            a, _, _ = _stratified_auc(gsh, pos, strata)
            col["GUARD_kl_own0_labelshuffled"] = (None if math.isnan(a)
                                                  else round(float(a), 4))
            col["_n_strata_used"] = nu
            col["_n_pairs"] = npair
            block["columns"][cname] = col
        return block

    print("\n=== Gate B1p: negate_live vs negate_dead, native predictive law "
          "===", flush=True)
    R["B1p"] = run_contrast("B1p", POS_B1, NEG_B1, COLS, restrict=reachable)
    print(json.dumps(R["B1p"], indent=1, cls=NumpyEncoder), flush=True)

    # head-to-head on exactly the positions where the prompted cache exists
    if "kl_promptown" in scores:
        R["B1p_promptsubset"] = run_contrast(
            "B1pp", POS_B1, NEG_B1, COLS, restrict=reachable, extra=pmask)

    # the kill-mechanism split -- the published sign reversal, re-read with the
    # native law. `nll` runs 0.891 (named) vs 0.164 (silent); a readout that
    # tracks the posterior rather than the surface should be flat across these.
    print("\n=== B1p breakdown by kill mechanism ===", flush=True)
    bd = {}
    te_all = (split_a == 2) & reachable & pred_mask
    for cell, m_extra in (("dead_named(negate)", kill_a == "negate"),
                          ("dead_silent(narrow)", kill_a == "narrow")):
        m_pos = te_all & (fams == "negate_live")
        m_neg = te_all & (fams == "negate_dead") & m_extra
        if m_pos.sum() < 25 or m_neg.sum() < 25:
            continue
        m = m_pos | m_neg
        pos = fams[m] == "negate_live"
        cs = _cross(tokid_a[m], frame_codes[m], nlive_b[m])
        row = {"n_live": int(m_pos.sum()), "n_dead": int(m_neg.sum())}
        for nm in ("nll", "out_entropy", "kl_own0", "drop_own0",
                   "logb_pre_own0", "negH_pre_own0",
                   "kl_own2", "drop_own2", "logb_pre_own2", "kl_oth0",
                   "GUARD_kl_swap_own0", "GUARD_drop_swap_own0",
                   "GUARD_kl_swap_own2",
                   "h_after_dir", "GUARD_h_swap_dir"):
            if nm not in scores:
                continue
            vv = np.where(np.isnan(scores[nm][m]), 0.0, scores[nm][m])
            row[nm] = round(float(_auc(vv, pos)), 4)
            a_, _, np_ = _stratified_auc(vv, pos, cs)
            row[nm + "_strat"] = None if math.isnan(a_) else round(float(a_), 4)
            row["_n_pairs_strat"] = round(np_)
        bd[cell] = row
    R["B1p_breakdown"] = bd
    print(json.dumps(bd, indent=1), flush=True)

    # ------------------------------------------------- Gate A (secondary)
    # Caveat in the module docstring: the before/after grammatical cut differs
    # across families, so this does not have B1's protection.
    val_m = ((split_a == 2) & pred_mask
             & np.isin(fams, ("reveal", "reveal_narrow", "negate_live",
                              "negate_dead", "echo", "deduced")))
    ga = {}
    for nm in ("kl_own0", "drop_own0", "logb_pre_own0", "negH_pre_own0",
               "kl_own2", "drop_own2", "logb_pre_own2",
               "kl_oth0", "kl_promptown", "kl_prompt_allslot"):
        if nm not in scores:
            continue
        mm = (val_m & (scores[nm] != 0.0)) if "prompt" in nm else val_m
        if mm.sum() < 100:
            continue
        Mv = scores[nm]
        ga[nm] = {
            "partial_r2_M_B_given_nll":
                round(_partial_r2(Mv[mm], Bv[mm], nll_a[mm]), 4),
            "partial_r2_M_nll_given_B":
                round(_partial_r2(Mv[mm], nll_a[mm], Bv[mm]), 4),
            "rank_partial_r2_M_B_given_nll":
                round(_partial_r2_rank(Mv[mm], Bv[mm], nll_a[mm]), 4),
            "partial_r2_M_B_given_outent":
                round(_partial_r2(Mv[mm], Bv[mm], ent_a[mm]), 4),
            "n": int(mm.sum()),
        }
    sh = scores["kl_own0"][_shuffle_within(frame_codes, swap_rng)]
    ga["null_within_frame"] = {"partial_r2_M_B_given_nll":
                               round(_partial_r2(sh[val_m], Bv[val_m],
                                                 nll_a[val_m]), 4)}
    ga["R2_B_given_nll"] = round(_r2(Bv[val_m], nll_a[val_m]), 4)
    R["A"] = ga
    print("\n=== Gate A (secondary) ===\n" + json.dumps(ga, indent=1), flush=True)

    # ------------------- family table: is the native revision just entropy?
    # The direct analogue of gates.py Gate D, where the FM residual's magnitude
    # read R^2 0.901 against output entropy and 0.0001 against oracle revision.
    gd = {}
    for f in sorted(set(fams[pred_mask])):
        m = (fams == f) & (split_a == 2) & pred_mask
        if m.sum() < 30:
            continue
        gd[f] = {"n": int(m.sum()),
                 "mean_B": round(float(Bv[m].mean()), 4),
                 "mean_nll": round(float(nll_a[m].mean()), 4),
                 "mean_out_entropy": round(float(ent_a[m].mean()), 4),
                 "mean_kl_own0": round(float(scores["kl_own0"][m].mean()), 4),
                 "mean_drop_own0": round(float(scores["drop_own0"][m].mean()), 4),
                 "mean_H_pre_own0": round(float(-scores["negH_pre_own0"][m].mean()), 4),
                 "mean_kl_own2": round(float(scores["kl_own2"][m].mean()), 4),
                 "mean_drop_own2": round(float(scores["drop_own2"][m].mean()), 4),
                 "mean_kl_oth0": round(float(scores["kl_oth0"][m].mean()), 4)}
    fn = sorted(gd)
    if len(fn) >= 4:
        g = lambda key: np.array([gd[f][key] for f in fn])
        gd["_across_family"] = {
            "families": fn,
            "r2_kl_vs_out_entropy": round(_r2(g("mean_kl_own0"),
                                              g("mean_out_entropy")), 4),
            "r2_kl_vs_nll": round(_r2(g("mean_kl_own0"), g("mean_nll")), 4),
            "r2_kl_vs_meanB": round(_r2(g("mean_kl_own0"), g("mean_B")), 4),
            "r2_drop_vs_out_entropy": round(_r2(g("mean_drop_own0"),
                                                g("mean_out_entropy")), 4),
            "r2_drop_vs_meanB": round(_r2(g("mean_drop_own0"), g("mean_B")), 4),
            "r2_kl2_vs_out_entropy": round(_r2(g("mean_kl_own2"),
                                               g("mean_out_entropy")), 4),
            "r2_kl2_vs_meanB": round(_r2(g("mean_kl_own2"), g("mean_B")), 4),
        }
    R["family_table"] = gd
    print("\n=== family table ===\n" + json.dumps(gd, indent=1), flush=True)

    out_dir = f"{DATA_DIR}/chronicle"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/predictive_{tag}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(R, f, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\nwrote {path}  ({time.time()-t00:.0f}s total)", flush=True)
    return path


@app.local_entrypoint()
def main():
    print("use: modal run -m a2a_forward.conditional_revision.predictive"
          "::predictive --tag smoke --n-pred-stories 40")
