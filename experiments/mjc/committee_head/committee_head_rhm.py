"""committee_head PHASE C -- calibration on RHM: the oracle SCORING but never FEEDING.

Spec: [`SPEC.md`](SPEC.md) §"Phase C". Sibling: [`committee_head.py`](committee_head.py) (Phases A+B,
on the arm). Substrate: [`../../rhm/conditional_revision/`](../../rhm/conditional_revision/README.md),
frozen -- this cut never retrains the base model, so it reads the same frozen reader every gate there
reads.

WHY THIS EXISTS. On the arm, "the head learned the three-way split" can only be INFERRED from where
the head sends the budget. Allocation is a confounded readout: a head could allocate well for the
wrong reason (relevance alone is nearly sufficient on-policy, because you only go where you reach --
E3's own finding), and there is no on-reach noise region there to break the tie. RHM has what the arm
does not: an EXACT, computable reducible/irreducible split, by belief propagation on the known parse
tree ([`oracle.py`](../../rhm/conditional_revision/oracle.py)). So here the same head shape is trained
the same reward-free way and then SCORED against the exact split. That turns the claim into a
measurement.

The oracle is scoring machinery only. It is computed after the fact, on held-out sequences, and no
part of it enters the committee, the benchmark net, the head's features, or the head's labels. The
head's only supervision is the same thing it gets on the arm: its own committee's realised error drop
over subsequent training.

THE SUBSTRATE, AND WHY THE TEMPORAL AXIS. The depth FM's residual is 0% aleatoric BY CONSTRUCTION --
predictor and target are deterministic functions of the same input, so its residual can only ever
mean "I lacked capacity". The temporal FM

        FM( h6[<=t] )  ->  Delta_t = h6[t+1] - h6[t]

depends on x_{t+1}, which a causal FM over h6[<=t] cannot hold: the conditioning gap is EXACTLY ONE
TOKEN, so the residual has a genuine aleatoric leg (0.665 of variance,
[`aleatoric_fraction/`](../../rhm/conditional_revision/aleatoric_fraction/README.md)). That is the
whole reason this is the right substrate for a reducible-vs-irreducible test, and it is why a single
FM cannot do it: the record shows its residual MAGNITUDE is the reader's output entropy (R^2 0.90)
with revision R^2 0.0001. The middle leg is not in the first moment.

THE MEASUREMENT. Per (sequence, position) row, over W windows of further committee training:

    features   e     committee-mean relative residual  ||tgt - pred_mean|| / ||tgt||
               d     disagreement, target-free: sqrt(mean-dim var across members) / ||pred_mean||
               bme   b(h) - e, with b a state-only net fitted online to the committee's OWN error
               + lags of each, exactly as on the arm
    label      max(e_w - e_{w+1}, 0)  -- the realised held-out error drop over the next window
    score      the oracle's exact reducible information I = H_tot - H_irr[D], and the PURELY
               IRREDUCIBLE rows (I ~ 0), which still carry surprisal

The head is FIT on one split of sequences and SCORED on a disjoint one, so "the head learned the
split" is a generalisation claim and not a memorisation of 63 positions.

THE CONFOUND THIS IS BUILT TO EXPOSE. A large fraction of positions are purely irreducible WHILE
STILL CARRYING SURPRISAL. So surprisal (and, the record predicts, the committee's own error `e`)
should be near-blind to the split, and anything that is not blind is reading the second moment. The
table prints every raw channel and surprisal next to the head, so a head that merely launders `e` is
visible as such.

NOTE ON SHAPE. The arm's head predicts TWO things (reducibility and relevance) because relevance is
not in the epistemic signature there. An open-loop corpus reader has no plan and no budget, so there
is no relevance to predict: this head has one output. That is a difference in the substrate, not a
different head.

Run:
    cd experiments/            # MODAL_PROFILE=chromatic
    # the frozen base must already exist (conditional_revision gate0 wrote it):
    modal run --detach mjc/committee_head/committee_head_rhm.py::phase_c --quick --tag pc_smoke
    modal run --detach mjc/committee_head/committee_head_rhm.py::phase_c --tag pc_s0 --seed 42
"""

import json
import os

import modal

from rhm.shared import volume, DATA_DIR, NumpyEncoder, setting_key

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "scipy==1.16.3", "torch==2.7.0")
    .add_local_python_source("rhm")
    .add_local_python_source("a2a_forward")
)
app = modal.App("committee-head-phase-c", image=image)

CH = ["e", "d", "bme"]


def tb_key(v, s, L, m):
    """Volume dir. The `_distinct` suffix is real (the regime uses generate_rules_distinct) --
    see `rhm/conditional_revision/conditional_revision.py::tb_key`."""
    return f"{setting_key(v, s, L, m)}_distinct"


@app.function(volumes={DATA_DIR: volume}, gpu="L4", timeout=43200, memory=32768)
def run_phase_c(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.stats import rankdata, spearmanr

    from rhm.model import GPT
    from rhm.rhm_data import generate_rules_distinct
    from rhm.rhm_latent_loop import _generate_with_traces
    from rhm.conditional_revision import oracle as ORC
    from a2a_forward.forward_model import TransformerForwardModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v, s, L = cfg["v"], cfg["s"], cfg["depth"]
    m, T = cfg["m"], s ** cfg["depth"]
    NM, beta = int(cfg["k_members"]), float(cfg["rpf_beta"])
    key = tb_key(v, s, L, m)
    deep = cfg["deep_block"]
    rules = generate_rules_distinct(v, s, L, m, seed=cfg["rule_seed"])
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    print(f"{'='*78}\nCOMMITTEE_HEAD PHASE C -- RHM calibration   {key}\n"
          f"  committee of {NM} TEMPORAL activation FMs (rpf_beta={beta}) on {deep}\n"
          f"  the oracle scores, and never feeds\n{'='*78}", flush=True)

    # ---------------- the frozen base reader (cached; conditional_revision gate0 wrote it) -------
    ckpt = cfg["base_ckpt"] or (f"{DATA_DIR}/{key}/conditional_revision/"
                                f"base_{cfg['n_layer']}L{cfg['n_head']}H{cfg['n_embd']}D_"
                                f"steps{cfg['base_steps']}_seed{cfg['base_seed']}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"no base checkpoint at {ckpt}. This cut NEVER retrains the base -- every gate in "
            f"conditional_revision reads the same frozen model, and so does this one. Run:\n"
            f"  modal run --detach rhm/conditional_revision/conditional_revision.py::gate0")
    model = GPT(v, T, cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"loaded frozen base <- {ckpt}", flush=True)

    # ---------------- data ----------------
    pool_seqs, _, _ = _generate_with_traces(rules, cfg["pool_size"], cfg["data_seed"])
    corpus = torch.from_numpy(pool_seqs.astype(np.int64)).reshape(-1)
    n_corpus = corpus.shape[0]
    arangeT = torch.arange(T)

    def get_batch(gen):
        ix = torch.randint(0, n_corpus - T - 1, (cfg["batch_size"],), generator=gen)
        idx = ix[:, None] + arangeT[None, :]
        return corpus[idx].to(device), corpus[idx + 1].to(device)

    # the scored set: FIXED sequences, never trained on, split into a head-FIT half and a
    # head-SCORE half so the split claim is a generalisation claim.
    # REPEAT-SAMPLING MANIPULATION (`--eval-from-pool`, with `--pool-size` as the dose).
    #
    # THE HYPOTHESIS this tests. On the arm, committee disagreement was the BEST reward-free
    # reducibility reader (0.716); here it is robustly INVERTED (0.320-0.377 across three
    # normalisations, deepening with training). The candidate mechanism is not "control vs
    # language" but REPEAT SAMPLING: the arm revisits the same state region many times, so members
    # average the aleatoric noise away and converge on the conditional mean -- they AGREE on noise
    # and disagree only on the frontier. A corpus gives each context essentially once, so members
    # never average it; they generalise from neighbours, generalise DIFFERENTLY, and scatter most
    # exactly where the target is most aleatoric -- which inverts the reader.
    #
    # The corpus here is a sliding window over concatenated sequences, so `pool_size` IS the
    # recurrence knob: a small pool makes every context recur many times during training. Drawing
    # the SCORED sequences from that same pool is what makes the recurrence reach the measurement
    # (otherwise eval contexts are fresh no matter how small the pool). If repeats are the
    # mechanism, d should climb back through 0.5 as the pool shrinks; if it stays inverted, the
    # mechanism is something else and porting to a control-flavoured RHM would be exploring blind.
    if cfg["eval_from_pool"]:
        rs = np.random.default_rng(cfg["eval_seed"])
        pick = rs.choice(len(pool_seqs), size=cfg["n_eval"], replace=False)
        ev_seqs = pool_seqs[pick]
        _, ev_lf_all, _ = _generate_with_traces(rules, cfg["pool_size"], cfg["data_seed"])
        ev_lf = {k: vv[pick] for k, vv in ev_lf_all.items()}
        print(f"[data] eval drawn FROM the training pool ({cfg['pool_size']} distinct seqs) -- "
              f"contexts recur during training", flush=True)
    else:
        ev_seqs, ev_lf, _ = _generate_with_traces(rules, cfg["n_eval"], cfg["eval_seed"])
    ev_x = torch.from_numpy(ev_seqs.astype(np.int64)).to(device)
    ev_y = torch.from_numpy(np.concatenate(
        [ev_seqs[:, 1:], ev_seqs[:, :1]], 1).astype(np.int64)).to(device)   # y unused past T-1
    n_fit = cfg["n_eval"] // 2
    fit_rows = np.zeros(cfg["n_eval"], bool); fit_rows[:n_fit] = True
    pos_top_level = np.array(
        [min(ell for ell in range(L + 1) if (p + 1) % (s ** (L - ell)) == 0) for p in range(T)],
        dtype=np.int64)[:T - 1]
    print(f"[data] eval {cfg['n_eval']} seqs x {T-1} positions "
          f"({n_fit} seqs fit / {cfg['n_eval'] - n_fit} scored)", flush=True)

    # ---------------- the committee: K temporal FMs, random-prior ----------------
    # Same member idiom as the arm: a trainable net plus a FROZEN random prior, so members agree
    # where data covers and disagree where it does not. The prediction is parameterised exactly as
    # the incumbent's `temporal` FM (`fm(h) - h`), so at beta=0 member 0 IS gate0's temporal FM.
    def make_fm():
        return TransformerForwardModel(
            d_model=cfg["n_embd"], d_head=cfg["fwd_d_head"], n_head=cfg["fwd_n_head"],
            n_layer=cfg["fwd_n_layer"], mlp_mult=cfg["fwd_mlp_mult"], block_size=T).to(device)

    members = []
    for k in range(NM):
        torch.manual_seed(cfg["seed"] + 1000 * k)
        net = make_fm()
        prior = None
        if beta > 0:
            torch.manual_seed(cfg["seed"] + 1000 * k + 99991)
            prior = make_fm()
            for p in prior.parameters():
                p.requires_grad_(False)
        members.append({"net": net, "prior": prior})
    opts = [torch.optim.AdamW(mm["net"].parameters(), lr=cfg["fwd_lr"],
                              weight_decay=cfg["weight_decay"]) for mm in members]

    def mem_pred(mm, h):
        p = mm["net"](h) - h
        if mm["prior"] is not None:
            p = p + beta * (mm["prior"](h) - h)
        return p[:, :-1, :]

    def deep_acts(x, y):
        with torch.no_grad():
            _, _, inter = model(x, y, return_intermediates=True)
        return inter[deep]

    def nll_of(x, y):
        with torch.no_grad():
            logits, _, _ = model(x, y, return_intermediates=True)
            return F.cross_entropy(logits.reshape(-1, v), y.reshape(-1),
                                   reduction="none").reshape(x.shape[0], T)[:, :T - 1]

    # ---------------- b(s): the online benchmark net on the committee's OWN error ----------------
    bench = nn.Sequential(nn.Linear(cfg["n_embd"], cfg["bench_hidden"]), nn.SiLU(),
                          nn.Linear(cfg["bench_hidden"], 1)).to(device)
    bopt = torch.optim.Adam(bench.parameters(), lr=cfg["bench_lr"])
    huber = nn.HuberLoss(delta=1.0)

    def bench_pred(h):
        with torch.no_grad():
            return bench(h).squeeze(-1)

    def bench_update(h, e, steps, gen):
        hf = h.reshape(-1, h.shape[-1]); ef = e.reshape(-1)
        bench.train()
        for _ in range(steps):
            idx = torch.randint(0, hf.shape[0], (min(cfg["bench_batch"], hf.shape[0]),),
                                generator=gen, device=hf.device)
            bopt.zero_grad(); huber(bench(hf[idx]).squeeze(-1), ef[idx]).backward(); bopt.step()
        bench.eval()

    # ---------------- the signature, per (sequence, position) ----------------
    def signature():
        """committee-mean relative residual, target-free disagreement, and b(h)-e, on the fixed
        scored set. Everything below is under no_grad against a frozen base."""
        E, D_, BME, HS, DR, DT = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, ev_x.shape[0], cfg["eval_chunk"]):
                x, y = ev_x[i:i + cfg["eval_chunk"]], ev_y[i:i + cfg["eval_chunk"]]
                h = deep_acts(x, y)
                tgt = h[:, 1:, :] - h[:, :-1, :]
                P = torch.stack([mem_pred(mm, h) for mm in members], 0)     # (NM, B, T-1, d)
                pm = P.mean(0)
                tn = tgt.norm(dim=-1) + 1e-6
                e = (tgt - pm).norm(dim=-1) / tn                            # `rres` convention
                # target-free: normalising disagreement by the target norm would leak the target
                sd = P.var(0).mean(-1).clamp_min(0).sqrt() * (P.shape[-1] ** 0.5)
                d = (sd / (pm.norm(dim=-1) + 1e-6) if NM > 1 else torch.zeros_like(e))
                # THREE NORMALISATIONS, because the choice can manufacture the result. `d` divides
                # by the committee's own prediction norm to stay target-free -- but if the members
                # regress toward a zero delta exactly where the target is most aleatoric, that
                # denominator shrinks there and d inflates, producing an anti-correlation with the
                # reducible share that is an artifact of the ruler rather than a fact about the
                # committee. d_raw (undivided) and d_tgt (divided by the target norm, the `rres`
                # convention e already uses) bracket it: if the sign is real it survives all three.
                d_raw = sd if NM > 1 else torch.zeros_like(e)
                d_tgt = (sd / tn if NM > 1 else torch.zeros_like(e))
                hc = h[:, :-1, :]
                E.append(e); D_.append(d); BME.append(bench_pred(hc) - e); HS.append(hc)
                DR.append(d_raw); DT.append(d_tgt)
        return (torch.cat(E), torch.cat(D_), torch.cat(BME), torch.cat(HS),
                torch.cat(DR), torch.cat(DT))

    def train_window(steps, gen):
        for mm in members:
            mm["net"].train()
        for step in range(steps):
            x, y = get_batch(gen)
            h = deep_acts(x, y)
            tgt = h[:, 1:, :] - h[:, :-1, :]
            for mm, o in zip(members, opts):
                loss = F.mse_loss(mem_pred(mm, h), tgt)
                o.zero_grad(); loss.backward(); o.step()
        for mm in members:
            mm["net"].eval()

    # ---------------- warm start, then W windows of (measure -> train -> label) ----------------
    gen_fm = torch.Generator().manual_seed(cfg["seed"] + 7)
    gen_b = torch.Generator(device=device).manual_seed(cfg["seed"] + 8)
    print(f"\n[warmup] {cfg['warm_steps']} committee steps (deliberately UNDER-trained: the label "
          f"is the error drop still available, so there must be some)", flush=True)
    train_window(cfg["warm_steps"], gen_fm)
    e0, d0, b0, h0, _, _ = signature()
    bench_update(h0, e0, cfg["bench_pretrain_steps"], gen_b)

    sig_hist, feat_hist, rows_X, rows_y, e_hist = [], [], [], [], []
    dn_hist = []
    hl = int(cfg["hist_len"])
    W = int(cfg["windows"])

    def build_feats():
        """per-row feature matrix: current channels + `hist_len` lags of each (zero-padded)."""
        cur = sig_hist[-1]
        cols = [cur[c] for c in CH]
        for Lg in range(1, hl + 1):
            past = sig_hist[-1 - Lg] if len(sig_hist) > Lg else None
            cols += [(past[c] if past is not None else np.zeros_like(cur[c])) for c in CH]
        return np.stack(cols, -1).reshape(-1, len(CH) * (hl + 1))

    for w in range(W):
        e, d, bme, hc, d_raw, d_tgt = signature()
        sig_hist.append({"e": e.cpu().numpy(), "d": d.cpu().numpy(), "bme": bme.cpu().numpy()})
        dn_hist.append((d_raw.cpu().numpy(), d_tgt.cpu().numpy()))
        feat_hist.append(build_feats())
        train_window(cfg["window_steps"], gen_fm)
        e2, _, _, hc2, _, _ = signature()
        bench_update(hc2, e2, cfg["bench_steps"], gen_b)
        # THE LABEL: the realised held-out error drop over the window just trained. This is the
        # valence tag -- "did the error here prove reducible over subsequent training" -- and it is
        # the only supervision the head ever sees.
        drop = torch.clamp(e - e2, min=0.0).cpu().numpy().reshape(-1)
        rows_X.append(feat_hist[-1]); rows_y.append(drop); e_hist.append(e.cpu().numpy())
        print(f"  [window {w:02d}] e {float(e.mean()):.4f} -> {float(e2.mean()):.4f}   "
              f"mean drop {float(drop.mean()):.5f}  d {float(d.mean()):.4f}  "
              f"b-e {float(bme.mean()):+.4f}", flush=True)
    e_hist.append(e2.cpu().numpy())

    # SMOOTHING THE LABEL. On the arm this was decisive: a single window's realised drop is
    # dominated by noise about a per-position mean (there, lprog's round-to-round autocorrelation
    # was +0.007), and the head cannot predict noise. Averaging the drop over `head_window`
    # consecutive windows trades resolution for signal. `--head-target drop_smooth` uses
    # e(w) - e(w+W) over W windows instead of one; W=1 is the per-window label.
    if cfg["head_target"] == "drop_smooth":
        SW = int(cfg["head_window"])
        rows_X2, rows_y2 = [], []
        for w in range(max(len(rows_X) - SW + 1, 0)):
            rows_X2.append(rows_X[w])
            rows_y2.append(np.clip(e_hist[w] - e_hist[min(w + SW, len(e_hist) - 1)], 0, None).reshape(-1))
        if rows_X2:
            rows_X, rows_y = rows_X2, rows_y2
        print(f"[head] target = drop over {SW} windows; {len(rows_X)} window-blocks retained",
              flush=True)

    X = np.concatenate(rows_X); Y = np.concatenate(rows_y)
    # NB, not W: `drop_smooth` retains only W - head_window + 1 blocks, so every per-row index
    # array must be tiled by the number of blocks ACTUALLY kept. Tiling by W instead silently
    # built masks 1/3 too long and the run died on the first boolean index.
    NB = len(rows_X)
    seq_of = np.tile(np.repeat(np.arange(cfg["n_eval"]), T - 1), NB)
    pos_of = np.tile(np.tile(pos_top_level, cfg["n_eval"]), NB)
    is_fit = fit_rows[seq_of]
    print(f"\n[head] {X.shape[0]} rows x {X.shape[1]} features "
          f"({int(is_fit.sum())} fit / {int((~is_fit).sum())} scored)", flush=True)

    # ---------------- the head: fit on one split, scored on the other ----------------
    # THE THREE METHODS THAT MATTERED ON THE ARM, carried across so Phase C is not a rerun of the
    # configuration that failed there (0.459 -> 0.857 was: linear head, RankNet loss, smoothed
    # target). NOTE the data regime differs sharply: the arm had K*T ~ 180 rows and was wildly
    # overparameterised, which is why `linear` was its biggest single lever; here there are
    # n_eval*(T-1)*windows rows, i.e. tens of thousands, so capacity is NOT expected to bind the
    # same way. Both are available and the default keeps the MLP.
    torch.manual_seed(cfg["seed"] + 50)
    hh = cfg["head_hidden"]
    head = (nn.Linear(X.shape[1], 1) if hh <= 0 else
            nn.Sequential(nn.Linear(X.shape[1], hh), nn.SiLU(),
                          nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, 1))).to(device)
    head = head.to(device)
    hopt = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"], weight_decay=cfg["head_wd"])
    mx, sx = X[is_fit].mean(0), X[is_fit].std(0) + 1e-6
    my, sy = Y[is_fit].mean(), Y[is_fit].std() + 1e-9
    Xf = torch.tensor((X[is_fit] - mx) / sx, device=device, dtype=torch.float32)
    Yf = torch.tensor((Y[is_fit] - my) / sy, device=device, dtype=torch.float32)
    ghead = torch.Generator(device=device).manual_seed(cfg["seed"] + 51)
    head.train()
    for step in range(cfg["head_steps"]):
        idx = torch.randint(0, Xf.shape[0], (min(cfg["head_batch"], Xf.shape[0]),),
                            generator=ghead, device=device)
        sc = head(Xf[idx]).squeeze(-1)
        if cfg["head_loss"] == "rank":
            # RankNet on a random pairing of the minibatch. Scoring here is by AUROC against the
            # oracle's class, which is itself a ranking statistic, so optimising the ordering is
            # the matched objective; and it is invariant to the label's scale, which is the part
            # of the realised-drop signal that is mostly noise.
            perm = torch.randperm(len(idx), generator=ghead, device=device)
            dl = sc - sc[perm]; dy = Yf[idx] - Yf[idx][perm]
            msk = dy.abs() > 0
            loss = (F.softplus(-torch.sign(dy[msk]) * dl[msk]).mean() if msk.any()
                    else sc.sum() * 0.0)
        else:
            loss = huber(sc, Yf[idx])
        hopt.zero_grad(); loss.backward(); hopt.step()
    head.eval()
    with torch.no_grad():
        Xa = torch.tensor((X - mx) / sx, device=device, dtype=torch.float32)
        pred = (head(Xa).squeeze(-1).cpu().numpy() * sy + my)

    # ---------------- THE ORACLE: computed now, on the scored split, having fed nothing ---------
    D = int(cfg["oracle_D"])
    print(f"\n[oracle] exact belief propagation, D={D} (H_irr[D] = H(x_t+1 | z_<=D, x_<=t)) ...",
          flush=True)
    orc = ORC.revision_and_entropy(rules, ev_seqs, ev_lf, Ds=[D], chunk=cfg["oracle_chunk"],
                                   verbose=False, return_leaf_posteriors=True)
    H_tot = orc["H_tot"]                         # (n, T-1)
    H_irr = orc["H_irr"][D]                      # (n, T-1)
    surpr = orc["surprisal"]
    I_red = H_tot - H_irr                        # exact REDUCIBLE information
    pure_irr = I_red < cfg["pure_tol"]
    frac_pure = float(pure_irr.mean())
    print(f"[oracle] H_tot {H_tot.mean():.4f}  H_irr {H_irr.mean():.4f}  "
          f"I_red {I_red.mean():.4f}   purely-irreducible rows: {frac_pure:.1%} "
          f"(mean surprisal there {surpr[pure_irr].mean():.4f} -- they still carry surprisal, "
          f"which is the whole confound)", flush=True)

    # ------------------------------------------------------------------------------------- #
    # THE EXACT ALEATORIC FLOOR OF THE FM'S OWN TARGET -- the target the label can actually track.
    #
    # WHY. Scoring against I_red was a conflation: I_red is how much the TOKEN reveals about latent
    # structure, while the head's label (realised error drop) is how much THE MODEL can improve.
    # Those come apart -- a position can be highly informative and already mastered, or
    # uninformative and simply unlearned -- and the label measured 0.486 against I_red at EVERY
    # position class, i.e. no information at all. So the head could not have learned it at any
    # reader quality, and the Phase C negative said nothing about the committee.
    #
    # The right object is `oracle.py`'s own documented law-of-total-variance split, in the FM's
    # target space rather than the token's:
    #     Var(h[t+1] | x<=t) = E_z[Var(h[t+1] | z,x<=t)]  ALEATORIC (synonymy; irreducible even
    #                                                                knowing the structure)
    #                        + Var_z(E[h[t+1] | z,x<=t])  EPISTEMIC (resolvable by better
    #                                                                inference about z)
    # Conditional on the prefix a causal model's h[t+1] takes exactly v values, one per arriving
    # token, so this is an EXACT v-term weighted sum -- `leaf_post` weights the outer variance,
    # `irr_post` the inner. Enumerating the v continuations through the frozen base gives every
    # term with no sampling.
    #
    # EPISTEMIC SHARE = epistemic / total is then the three-way split in the FM's own units:
    # mastered (error already at the floor) / aleatoric / learnable. That is the quantity realised
    # error drop tracks, and the one the arm's classes were an instance of.
    # ------------------------------------------------------------------------------------- #
    lpost = orc["leaf_post"][:, :, :]                 # (n, T-1, v)  P(x_t+1 | x<=t)
    ipost = orc["irr_post"][D]                        # (n, T-1, v)  P(x_t+1 | z<=D, x<=t)
    print(f"[floor] enumerating {v} continuations per position through the frozen base ...",
          flush=True)
    alea = np.zeros((ev_x.shape[0], T - 1), np.float64)
    epis = np.zeros((ev_x.shape[0], T - 1), np.float64)
    with torch.no_grad():
        for i in range(0, ev_x.shape[0], cfg["floor_chunk"]):
            xb = ev_x[i:i + cfg["floor_chunk"]]
            B = xb.shape[0]
            _, _, inter0 = model(xb, xb, return_intermediates=True)
            h0 = inter0[deep]                                          # (B, T, d)
            for t in range(T - 1):
                var = xb.unsqueeze(1).repeat(1, v, 1).clone()          # (B, v, T)
                var[:, :, t + 1] = torch.arange(v, device=device)[None, :]
                flat = var.reshape(B * v, T)
                _, _, it = model(flat, flat, return_intermediates=True)
                hk = it[deep][:, t + 1, :].reshape(B, v, -1)           # (B, v, d)
                dk = hk - h0[:, t, :].unsqueeze(1)                     # Delta per candidate
                pl = torch.tensor(lpost[i:i + B, t, :], device=device, dtype=torch.float32)
                pi = torch.tensor(ipost[i:i + B, t, :], device=device, dtype=torch.float32)
                mu_l = (pl.unsqueeze(-1) * dk).sum(1, keepdim=True)
                mu_i = (pi.unsqueeze(-1) * dk).sum(1, keepdim=True)
                tot = (pl * ((dk - mu_l) ** 2).sum(-1)).sum(1)
                ale = (pi * ((dk - mu_i) ** 2).sum(-1)).sum(1)
                alea[i:i + B, t] = ale.double().cpu().numpy()
                epis[i:i + B, t] = (tot - ale).clamp_min(0).double().cpu().numpy()
    tot_var = alea + epis
    epi_share = epis / np.maximum(tot_var, 1e-12)
    print(f"[floor] mean total Var(dh|prefix) {tot_var.mean():.5f}   aleatoric {alea.mean():.5f} "
          f"({alea.mean()/max(tot_var.mean(),1e-12):.1%})   epistemic {epis.mean():.5f}", flush=True)

    # tile the oracle over windows to match the row layout
    I_tile = np.tile(I_red.reshape(-1), NB)
    pure_tile = np.tile(pure_irr.reshape(-1), NB)
    sur_tile = np.tile(surpr.reshape(-1), NB)
    nll_tile = np.tile(nll_of(ev_x, ev_y).cpu().numpy().reshape(-1), NB)
    sc = ~is_fit                                  # SCORED split only, everywhere below

    def _auroc(pos, neg):
        pos = np.asarray(pos, float)[np.isfinite(pos)]
        neg = np.asarray(neg, float)[np.isfinite(neg)]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        r = rankdata(np.concatenate([pos, neg]))
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    def score(name, x):
        x = np.asarray(x, float)[sc]
        au = _auroc(x[~pure_tile[sc]], x[pure_tile[sc]])       # high on reducible = >0.5
        rho = float(spearmanr(x, I_tile[sc]).statistic)
        return {"auroc_red_vs_pure_irr": au, "rho_with_I_red": rho}

    nb = min(len(dn_hist), len(rows_X))
    chans = {"e": X[:, 0], "d": X[:, 1], "b-e": X[:, 2],
             "d_raw (unnormalised)": np.concatenate([dn_hist[w][0].reshape(-1) for w in range(nb)]),
             "d_tgt (/target norm)": np.concatenate([dn_hist[w][1].reshape(-1) for w in range(nb)]),
             "surprisal (oracle)": sur_tile, "model nll": nll_tile,
             "HEAD (learned)": pred, "realised drop (the label)": Y}
    table = {k: score(k, xx) for k, xx in chans.items()}

    # ---------------------------------------------------------------------------------------- #
    # STRATIFY BY POSITION CLASS. The pooled numbers above are CONFOUNDED and must not be the
    # headline. "Purely irreducible" is not a free variable in RHM: I_red = H_tot - H_irr, so
    # rows with I_red ~ 0 are largely rows with low TOTAL entropy, and those sit at particular
    # tree levels (measured: frac pure-irr by level runs 0.000 0.000 0.000 0.008 0.307 0.393).
    # Anything correlated with position therefore scores well without reading the split at all --
    # which is why the ORACLE'S OWN SURPRISAL, the quantity this cut was built to show is blind,
    # comes out as the best pooled channel (0.895). It is not reading reducibility; it is reading
    # depth. The pooled column is retained only as the record of that confound.
    #
    # Conditioning on level removes it, and it flips the reading: pooled, `d` scores 0.481
    # (blind); within level 6, 0.710. Two stratified readouts:
    #   (a) LEVEL-STRATIFIED AUROC -- AUROC computed inside each level, sample-weighted. Asks
    #       "among positions at the SAME tree depth, does the channel find the irreducible ones".
    #   (b) BALANCED WITHIN-LEVEL SPLIT -- median-split I_red inside each level, so the classes
    #       are balanced by construction and the shallow levels (which have no pure-irr rows at
    #       all, hence nan above) still contribute.
    # ---------------------------------------------------------------------------------------- #
    per_level = {}

    def strat_auroc(x, hi, lo, key=None):
        x = np.asarray(x, float)
        num = den = 0.0
        for lv in sorted(set(pos_of[sc].tolist())):
            mk = pos_of[sc] == lv
            a = _auroc(x[sc][mk & hi[sc]], x[sc][mk & lo[sc]])
            if key is not None:
                per_level.setdefault(key, {})[int(lv)] = a
            if np.isfinite(a):
                n = int((mk & (hi[sc] | lo[sc])).sum()); num += a * n; den += n
        return num / den if den else float("nan")

    has_red = ~pure_tile
    med_hi = np.zeros_like(pure_tile); med_lo = np.zeros_like(pure_tile)
    for lv in sorted(set(pos_of.tolist())):
        mk = pos_of == lv
        if mk.sum() < 4:
            continue
        thr = np.median(I_tile[mk])
        med_hi[mk] = I_tile[mk] > thr
        med_lo[mk] = I_tile[mk] <= thr
    # (c) THE REDUCIBLE FRACTION, and the reason it is the right target. I_red = H_tot - H_irr
    # is a COMPONENT of total entropy, so any channel tracking entropy tracks I_red whether or
    # not it reads the split -- which is why the oracle's own surprisal stays at 0.91 even after
    # conditioning on depth, and why 0.5 is the WRONG reference for this cut. The entropy-free
    # question is "of the uncertainty here, what SHARE is learnable": I_red / H_tot. Splitting on
    # that within a level factors overall entropy out by construction, which is the actual
    # three-way-split question.
    frac_red = I_red / np.maximum(H_tot, 1e-9)
    frac_tile = np.tile(frac_red.reshape(-1), NB)
    epi_tile = np.tile(epi_share.reshape(-1), NB)
    fr_hi = np.zeros_like(pure_tile); fr_lo = np.zeros_like(pure_tile)
    ep_hi = np.zeros_like(pure_tile); ep_lo = np.zeros_like(pure_tile)
    for lv in sorted(set(pos_of.tolist())):
        mk = pos_of == lv
        if mk.sum() < 4:
            continue
        thr = np.median(frac_tile[mk])
        fr_hi[mk] = frac_tile[mk] > thr
        fr_lo[mk] = frac_tile[mk] <= thr
        te = np.median(epi_tile[mk])
        ep_hi[mk] = epi_tile[mk] > te
        ep_lo[mk] = epi_tile[mk] <= te
    for k, xx in chans.items():
        table[k]["auroc_stratified"] = strat_auroc(xx, has_red, pure_tile)
        table[k]["auroc_balanced_within_level"] = strat_auroc(xx, med_hi, med_lo)
        table[k]["auroc_reducible_fraction"] = strat_auroc(xx, fr_hi, fr_lo, key=k)
        table[k]["auroc_epistemic_share"] = strat_auroc(xx, ep_hi, ep_lo, key="EPI:" + k)
    # THE BASELINE TO BEAT is the base model's own next-token loss, not chance. The committee is
    # only worth its cost if its second moment adds something the reader's own output entropy
    # does not already say (the record: FM error magnitude IS output entropy, R^2 0.90).
    nll_ref = {kk: table["model nll"][kk] for kk in
               ("auroc_stratified", "auroc_balanced_within_level", "auroc_reducible_fraction",
                "auroc_epistemic_share")}
    for k in chans:
        table[k]["lift_over_nll"] = {kk: table[k][kk] - nll_ref[kk] for kk in nll_ref}

    print(f"\n{'='*78}\nSTRATIFIED BY POSITION CLASS -- the readout that is not confounded\n"
          f"  (a) AUROC within each tree level, sample-weighted: among positions at the SAME\n"
          f"      depth, does the channel separate reducible from purely irreducible?\n"
          f"  (b) balanced within-level median split on I_red, so every level contributes.\n"
          f"  The POOLED column is confounded by depth -- surprisal tops it without reading the\n"
          f"  split at all -- and is shown only for contrast.\n{'='*78}")
    print(f"  {'channel':>26s} {'(a)strat':>9s} {'(b)balanced':>12s} {'(c)frac':>8s} "
          f"{'(c) lift vs nll':>16s} {'pooled':>8s}")
    for k in chans:
        print(f"  {k:>26s} {table[k]['auroc_stratified']:9.3f} "
              f"{table[k]['auroc_balanced_within_level']:12.3f} "
              f"{table[k]['auroc_reducible_fraction']:8.3f} "
              f"{table[k]['lift_over_nll']['auroc_reducible_fraction']:+16.3f} "
              f"{table[k]['auroc_red_vs_pure_irr']:8.3f}")
    print(f"\n  (c) BROKEN OUT BY POSITION CLASS -- the mechanism test. RHM position classes differ\n"
          f"      in how often a context RECURS: the common leaf-completing positions are seen in\n"
          f"      many sequences, the rare deep-structure ones essentially once. If the committee\n"
          f"      needs REPEATS to separate epistemic from aleatoric (the arm has them; a corpus\n"
          f"      largely does not), d should work where contexts recur and inverted where they do not.")
    lvs = sorted(per_level.get("d", {}))
    print(f"      {'channel':>26s} " + " ".join(f"{'L'+str(l):>7s}" for l in lvs))
    print(f"      {'n rows':>26s} " + " ".join(
        f"{int((pos_of[sc] == l).sum()):7d}" for l in lvs))
    for k in chans:
        print(f"      {k:>26s} " + " ".join(f"{per_level.get(k, {}).get(l, float('nan')):7.3f}"
                                            for l in lvs))
    print(f"\n{'='*78}\n(d) THE HEADLINE -- EPISTEMIC SHARE of the FM's OWN target variance\n"
          f"    epistemic / (aleatoric + epistemic), split at the within-level median. This is the\n"
          f"    three-way split in the FM's units -- exactly what the realised-drop label tracks --\n"
          f"    and it is computed EXACTLY by enumerating the v continuations, not estimated.\n{'='*78}")
    print(f"  {'channel':>26s} {'(d) epistemic':>14s} {'lift vs nll':>12s}")
    for k in chans:
        print(f"  {k:>26s} {table[k]['auroc_epistemic_share']:14.3f} "
              f"{table[k]['lift_over_nll']['auroc_epistemic_share']:+12.3f}")
    # ---------------------------------------------------------------------------------- #
    # LABEL RELIABILITY -- the diagnostic that is PRIOR to any choice of target.
    #
    # The label failed against I_red (0.486) and again against the FM's own exact epistemic
    # share (0.491). Two different targets, same answer, which points at the LABEL rather than
    # the target. And there was already evidence: the per-window drops sum to ~5x the net
    # decrease in e, i.e. e fluctuates up and down far more than it descends.
    #
    # Reliability asks whether the label has ANY reproducible per-position signal, independent of
    # what it is scored against: measure the drop on two DISJOINT halves of the scored sequences
    # and correlate the per-position means. If that is ~0 the label is noise and no target can
    # rescue it -- Phase C would need a different label (a longer horizon, or a held-out-position
    # ablation), not a different oracle.
    # ---------------------------------------------------------------------------------- #
    rel = {}
    try:
        nrow = (T - 1)
        Yb = Y.reshape(NB, ev_x.shape[0], nrow)
        ha, hb = np.arange(ev_x.shape[0]) % 2 == 0, np.arange(ev_x.shape[0]) % 2 == 1
        pa = Yb[:, ha, :].mean(axis=(0, 1)); pb = Yb[:, hb, :].mean(axis=(0, 1))
        rel["split_half_per_position"] = float(spearmanr(pa, pb).statistic)
        if NB > 1:
            wa = Yb[:-1].reshape(-1); wb = Yb[1:].reshape(-1)
            rel["window_to_window_same_row"] = float(spearmanr(wa, wb).statistic)
        rel["per_position_sd_over_mean"] = float(np.std(Yb.mean(axis=(0, 1))) /
                                                 max(abs(Yb.mean()), 1e-12))
    except Exception as exc:                       # never let a diagnostic kill the run
        rel["error"] = repr(exc)
    # BETWEEN- vs WITHIN-LEVEL VARIANCE. The label is reliable per position (split-half 0.93) yet
    # scores ~0.5 against every target, and the within-level median split at L6 left one side
    # empty. Both point at the same possibility: in RHM the oracle quantities and the label may be
    # nearly FUNCTIONS OF TREE POSITION, so conditioning on level -- which I did to remove the
    # depth confound -- removes almost all the variance there is. If so the per-position framing
    # cannot pose the question at all, and that is a fact about the substrate rather than about
    # the committee.
    def _btw(x2d):
        x = np.asarray(x2d, float).mean(0)            # per-position mean over sequences
        gm = x.mean(); tot = float(((x - gm) ** 2).mean())
        bt = 0.0
        for lv in sorted(set(pos_top_level.tolist())):
            mk = pos_top_level == lv
            bt += mk.mean() * (x[mk].mean() - gm) ** 2
        return float(bt / max(tot, 1e-18))
    try:
        rel["between_level_share_epistemic"] = _btw(epi_share)
        rel["between_level_share_I_red"] = _btw(frac_red)
        rel["between_level_share_label"] = _btw(Y.reshape(NB, ev_x.shape[0], T - 1).mean(0))
    except Exception as exc:
        rel["btw_error"] = repr(exc)
    print(f"\n{'='*78}\nLABEL RELIABILITY -- prior to any target choice\n{'='*78}")
    for kk, vv in rel.items():
        print(f"  {kk:>28s}  {vv if isinstance(vv, str) else round(vv, 4)}")
    print("  split-half ~0 means the label has no reproducible per-position signal at all, and no\n"
          "  oracle can rescue it: the fix would be a longer-horizon label, not a better target.")

    print("\n  SANITY CHECK -- does the LABEL track this target? It measured 0.486 against I_red\n"
          "  (chance at every position class), which is what made the earlier Phase C negative\n"
          "  uninterpretable. If it is still ~0.5 here the mismatch is not fixed and no head\n"
          "  result below should be read.")
    print("\n  (c) previously-reported target: high-vs-low REDUCIBLE SHARE (I_red/H_tot) within a level, which\n"
          "      factors out total entropy. `lift vs nll` is the quantity that decides whether the\n"
          "      committee earns its cost -- beating 0.5 is not the bar, beating the base model's\n"
          "      own next-token loss is.")

    print(f"\n{'='*78}\nPHASE C RESULT -- does the signature read the EXACT split?\n"
          f"  scored split only ({int(sc.sum())} rows); AUROC separates rows with reducible\n"
          f"  content from PURELY IRREDUCIBLE rows. 0.5 = blind. rho is against the exact\n"
          f"  reducible information I = H_tot - H_irr[D].\n{'='*78}")
    print(f"  {'channel':>26s} {'AUROC red-vs-pure-irr':>22s} {'rho(., I_red)':>14s}")
    for k in chans:
        print(f"  {k:>26s} {table[k]['auroc_red_vs_pure_irr']:22.3f} "
              f"{table[k]['rho_with_I_red']:14.3f}")

    # by position class -- the hierarchy level completed at each position
    bylev = {}
    print(f"\n  by position class (hierarchy level completed there):")
    print(f"  {'level':>6s} {'n':>7s} {'frac pure-irr':>14s} {'HEAD auroc':>11s} {'e auroc':>9s} "
          f"{'d auroc':>9s}")
    for lv in sorted(set(pos_top_level.tolist())):
        mk = sc & (pos_of == lv)
        if mk.sum() < 20:
            continue
        pm, nm_ = pure_tile & mk, (~pure_tile) & mk
        row = {"n": int(mk.sum()), "frac_pure": float(pure_tile[mk].mean()),
               "head": _auroc(pred[nm_], pred[pm]), "e": _auroc(X[nm_, 0], X[pm, 0]),
               "d": _auroc(X[nm_, 1], X[pm, 1])}
        bylev[int(lv)] = row
        print(f"  {lv:6d} {row['n']:7d} {row['frac_pure']:14.3f} {row['head']:11.3f} "
              f"{row['e']:9.3f} {row['d']:9.3f}")

    out = {"config": cfg, "table": table, "by_level": bylev,
           "per_level_reducible_fraction": per_level,
           "label_reliability": rel,
           "fm_variance": {"mean_total": float(tot_var.mean()), "mean_aleatoric": float(alea.mean()),
                           "mean_epistemic": float(epis.mean()),
                           "aleatoric_fraction": float(alea.mean() / max(tot_var.mean(), 1e-12))},
           "n_by_level": {int(lv): int((pos_of[sc] == lv).sum())
                          for lv in sorted(set(pos_of[sc].tolist()))},
           "frac_pure_by_level": {int(lv): float(pure_tile[pos_of == lv].mean())
                                  for lv in sorted(set(pos_of.tolist()))},
           "oracle": {"mean_H_tot": float(H_tot.mean()), "mean_H_irr": float(H_irr.mean()),
                      "mean_I_red": float(I_red.mean()), "frac_pure_irr": frac_pure,
                      "mean_surprisal_at_pure_irr": float(surpr[pure_irr].mean()),
                      "D": D, "pure_tol": cfg["pure_tol"]},
           # min(), not W: `drop_smooth` leaves fewer label blocks than signature windows, and
           # indexing rows_y by range(W) walked off the end. Same defect as the tiling above --
           # I fixed that occurrence and missed this one, so both now derive their length from
           # the arrays themselves rather than from W.
           "windows": [{"w": w, "mean_e": float(sig_hist[w]["e"].mean()),
                        "mean_d": float(sig_hist[w]["d"].mean()),
                        "mean_bme": float(sig_hist[w]["bme"].mean()),
                        "mean_drop": float(rows_y[w].mean())}
                       for w in range(min(len(sig_hist), len(rows_y)))]}
    outdir = os.path.join(DATA_DIR, key, "committee_head_phase_c", cfg["tag"])
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    volume.commit()
    print(f"\n[save] wrote results to {outdir}", flush=True)
    return out


@app.local_entrypoint()
def phase_c(
    quick: bool = False,
    tag: str = "",
    seed: int = 42,
    # --- the committee ---
    k_members: int = 4,
    rpf_beta: float = 0.6,
    # --- windows: measure -> train -> label ---
    warm_steps: int = 3000,
    windows: int = 8,
    window_steps: int = 800,
    hist_len: int = 2,
    # --- the head ---
    head_hidden: int = 32,
    head_lr: float = 3e-3,
    head_wd: float = 1e-4,
    head_steps: int = 3000,
    head_batch: int = 1024,
    head_loss: str = "huber",          # "rank" = RankNet (matched to the AUROC readout)
    head_target: str = "drop",         # "drop_smooth" = drop over `head_window` windows
    head_window: int = 3,
    # --- b(s) ---
    bench_hidden: int = 64,
    bench_lr: float = 1e-3,
    bench_batch: int = 1024,
    bench_pretrain_steps: int = 800,
    bench_steps: int = 200,
    # --- the oracle (scoring only) ---
    oracle_depth: int = 5,                 # deepest internal level; aleatoric_fraction's convention
    oracle_chunk: int = 128,
    floor_chunk: int = 32,             # sequences per exact-floor enumeration pass
    pure_tol: float = 1e-6,            # I_red below this == purely irreducible
    # --- substrate (conditional_revision's regime; the base ckpt must match) ---
    v: int = 16, s: int = 2, depth: int = 6, m: int = 4, rule_seed: int = 0,
    n_layer: int = 8, n_head: int = 8, n_embd: int = 256,
    deep_block: str = "post_block6",
    fwd_n_layer: int = 1, fwd_d_head: int = 16, fwd_n_head: int = 8, fwd_mlp_mult: float = 1.0,
    fwd_lr: float = 1e-3, weight_decay: float = 0.01,
    base_steps: int = 12000, base_seed: int = 42, base_ckpt: str = "",
    batch_size: int = 64, pool_size: int = 200000, data_seed: int = 7,
    n_eval: int = 256, eval_seed: int = 999, eval_chunk: int = 64,
    eval_from_pool: bool = False,      # score sequences the committee actually trains on
):
    if quick:
        warm_steps = 600; windows = 4; window_steps = 200; k_members = 3
        n_eval = 64; head_steps = 1200; bench_pretrain_steps = 300; bench_steps = 80
        pool_size = 40000; oracle_chunk = 64
        tag = tag or "pc_smoke"
    tag = tag or "default"
    cfg = dict(
        tag=tag, seed=seed, k_members=k_members, rpf_beta=rpf_beta,
        warm_steps=warm_steps, windows=windows, window_steps=window_steps, hist_len=hist_len,
        head_hidden=head_hidden, head_lr=head_lr, head_wd=head_wd, head_steps=head_steps,
        head_batch=head_batch, head_loss=head_loss, head_target=head_target,
        head_window=head_window, bench_hidden=bench_hidden, bench_lr=bench_lr,
        bench_batch=bench_batch, bench_pretrain_steps=bench_pretrain_steps, bench_steps=bench_steps,
        oracle_D=oracle_depth, oracle_chunk=oracle_chunk, floor_chunk=floor_chunk, pure_tol=pure_tol,
        v=v, s=s, depth=depth, m=m, rule_seed=rule_seed,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, deep_block=deep_block,
        fwd_n_layer=fwd_n_layer, fwd_d_head=fwd_d_head, fwd_n_head=fwd_n_head,
        fwd_mlp_mult=fwd_mlp_mult, fwd_lr=fwd_lr, weight_decay=weight_decay,
        base_steps=base_steps, base_seed=base_seed, base_ckpt=base_ckpt,
        batch_size=batch_size, pool_size=pool_size, data_seed=data_seed,
        n_eval=n_eval, eval_seed=eval_seed, eval_chunk=eval_chunk,
        eval_from_pool=bool(eval_from_pool),
    )
    out = run_phase_c.remote(cfg)
    localdir = os.path.join(os.path.dirname(__file__), "figures", "phase_c_" + tag)
    os.makedirs(localdir, exist_ok=True)
    with open(os.path.join(localdir, "results.json"), "w") as fh:
        json.dump(out, fh, indent=2, cls=NumpyEncoder)
    print(f"\n[local] wrote results.json to {localdir}")
