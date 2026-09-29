"""[grokking] Node 1: mint -- read the frequencies off the net's own layer-0 weights, keep the
ones its own predictions say it needs, and compile them.

Grok once (vanilla recipe, seed 42) with snapshots along the curve. At each snapshot:

  1. READ.  1-D DFT of every layer-0 row over a (W0[j, :p]) and over b (W0[j, p:]). Candidate =
     folded frequency w in 1..48. Producer score = energy at w summed over units (a + b halves,
     w and p-w folded); support = # units whose dominant frequency is w (argmax over 1..48; DC
     excluded).
  2. WALK.  `census_walk` (forked verbatim below) from an empty table, candidates in descending
     producer score; controls: N_RAND random orders.
  3. GATE (consumed). aud_fn(K) = disagreement between the compiled model's argmax and the NET'S
     OWN argmax on the train pairs. Tie-aware: a compiled model whose max is shared by m classes
     agrees with weight 1/m if the net's class is among them (so the empty table is exactly chance,
     1 - 1/p, for the closed forms). Three tolerances (NOTES.md "Gate rules"): le0 (tol 0, census
     verbatim: admits ties), lt1 (must fix >= 1 train pair net), m3 (must beat 3 sigma of chance
     agreement, ~16 pairs).
  4. LOGGED BESIDE, never consumed: the compiled model's held-out label accuracy at every admission,
     the net's own train/test accuracy, the logit-table Fourier energy (net outputs on all p^2
     inputs, no labels).
  5. COMPILE, one walk per form, each gated by its own agreement with the net:
       A    closed form, logits[c] = sum_{w in K} cos(2 pi w (a+b-c)/p); zero free parameters.
       ALS  A with per-frequency amplitudes alpha_w fit by least squares to the net's own train
            logits (row-centred; the design is balanced so alpha_w is the cosine coefficient of the
            diagonal-averaged logit g(d), d = a+b-c).
       B    the net's own spelling: the trained net with every hidden unit whose dominant frequency
            is not in K zeroed (no retraining). Also the unit-pruning baseline.
       C    (added, NOTES.md) the net's own spelling at the frequency level: every layer-0 row
            projected onto DC + the characters in K, no retraining. The MLP analogue of a
            restricted-loss ablation.
     Every form is also evaluated at every other form's kept K (cross table).

Falsifiers (run inside `mint`): a shuffled-label net (train labels permuted, same split/seed) and
random-init nets (the seed-42 init + seeds 0..3, untrained) go through the identical pipeline;
form B at K = all 48 must reproduce the net bit-for-bit at every snapshot.

Commands (run from experiments/, MODAL_PROFILE=chromatic):
    modal run grokking/mint.py::gates
    modal run grokking/mint.py::mint --tag smoke --smoke 1
    modal run --detach grokking/mint.py::mint --tag m1
    modal run --detach grokking/mint.py::analyze --tag m1     # re-reduce banked snapshots
    python3 grokking/reduce_mint.py --tag m1 --fetch
"""

import json
import math
import os
import time

import numpy as np

from grokking.shared import (DATA_DIR, HIDDEN, LR, N_EPOCHS, N_FREQ, P, SEED, TRAIN_FRAC,
                                  WEIGHT_DECAY, F, GrokMLP, app, generate_data, onehot, peak_rss_mb,
                                  split_pairs, torch, train_net, volume)

FORMS = ("A", "ALS", "B", "C")
N_RAND = 5
RAND_SEED0 = 1000
INIT_SEEDS = (0, 1, 2, 3)
N_TR = int(P * P * TRAIN_FRAC)


def gate_rules(n_tr=N_TR, p=P):
    q = 1.0 / p
    return {"le0": 0.0,
            "lt1": -0.5 / n_tr,
            "m3": -3.0 * math.sqrt(q * (1 - q) / n_tr)}


# =============================================================================
# FORK NOTICE. `census_walk` is forked VERBATIM from
# experiments/rhm/practice/voicing/sotto_voce/aliquot/preplay/incremental.py lines 105-143
# (2026-09-22), with EXTEND_TOL = 0.0 as there. Candidates here are one-element rows [w], so the
# row bookkeeping (list(map(int, r_))) is unchanged; `aud_fn` flattens them.
# =============================================================================

EXTEND_TOL = 0.0                # cfg["extend_tol"] -- must not hurt


def census_walk(aud_fn, base_child, cand_child, order_idx, tol=EXTEND_TOL, budget=None,
                on_change=None):
    """`census_extend`'s admission loop (`soundboard.py` 13396-13414), with the candidates
    offered in `order_idx` instead of the miner's own order and with no cap unless `budget`
    says so. `aud_fn(child_rows) -> e` is ONE audition on the gate pool; it is called exactly
    once for the base and once per candidate offered, which is what the loop pays.

    `on_change(n_aud_so_far, kept)` is called after the base audition and after every ADMISSION
    (and only then: a rejection leaves the table untouched, so its test error is the previous
    one carried forward). Returns the record.
    """
    n_aud = 1
    best_e = aud_fn(base_child)
    kept = [list(map(int, r_)) for r_ in base_child]
    rec = {"e_base_gate": float(best_e), "admitted": [], "rejected": 0,
           "steps": [], "n_auditions": n_aud}
    if on_change is not None:
        on_change(n_aud, kept)
    walk = order_idx if budget is None else order_idx[:int(budget)]
    for step, ci in enumerate(walk):
        e_x = aud_fn(kept + [list(map(int, cand_child[ci]))])
        n_aud += 1
        admit = bool(e_x <= best_e + tol)
        if admit:
            kept = kept + [list(map(int, cand_child[ci]))]
            best_e = e_x
            rec["admitted"].append(int(ci))
        else:
            rec["rejected"] += 1
        rec["steps"].append({"step": int(step), "cand": int(ci), "e_gate": float(e_x),
                             "admit": admit, "n_aud": int(n_aud),
                             "n_kept": int(len(kept))})
        if admit and on_change is not None:
            on_change(n_aud, kept)
    rec["n_auditions"] = n_aud
    rec["e_gate_final"] = float(best_e)
    rec["n_kept"] = int(len(kept))
    rec["kept"] = kept
    return rec

# ============================ end of verbatim fork ============================


def _flat(rows):
    return tuple(sorted(int(r[0]) for r in rows))


# =============================================================================
# READ: frequencies off layer-0 weights
# =============================================================================

def read_layer0(W0, p=P):
    """Per-unit folded Fourier energy of the a-row and b-row of layer 0.

    Returns E (H, 48) combined a+b energy at w=1..48, Ea, Eb (H, 48), dc (H,) DC energy,
    dom (H,) dominant w (argmax of E over 1..48), dom_a, dom_b, and the complex spectra
    Fa, Fb (H, p) for the frequency filter (form C)."""
    W0 = np.asarray(W0, np.float64)
    Fa = np.fft.fft(W0[:, :p], axis=1)
    Fb = np.fft.fft(W0[:, p:2 * p], axis=1)
    w = np.arange(1, N_FREQ + 1)
    Pa, Pb = np.abs(Fa) ** 2, np.abs(Fb) ** 2
    Ea = Pa[:, w] + Pa[:, p - w]
    Eb = Pb[:, w] + Pb[:, p - w]
    E = Ea + Eb
    return {"E": E, "Ea": Ea, "Eb": Eb, "dc": Pa[:, 0] + Pb[:, 0],
            "dom": w[np.argmax(E, axis=1)], "dom_a": w[np.argmax(Ea, axis=1)],
            "dom_b": w[np.argmax(Eb, axis=1)], "Fa": Fa, "Fb": Fb}


def filter_layer0(Fa, Fb, K, p=P):
    """Layer-0 weight rows projected onto DC + characters in K (both halves)."""
    m = np.zeros(p)
    m[0] = 1.0
    for w in K:
        m[w] = 1.0
        m[p - w] = 1.0
    return np.concatenate([np.fft.ifft(Fa * m, axis=1).real,
                           np.fft.ifft(Fb * m, axis=1).real], axis=1)


def spread_stats(q):
    """Concentration of a nonnegative score over the 48 frequencies."""
    q = np.asarray(q, np.float64)
    s = q / (q.sum() + 1e-300)
    srt = np.sort(s)[::-1]
    return {"participation": float(1.0 / (s ** 2).sum()),
            "n90": int(np.searchsorted(np.cumsum(srt), 0.90) + 1),
            "n99": int(np.searchsorted(np.cumsum(srt), 0.99) + 1),
            "top1_frac": float(srt[0]), "top5_frac": float(srt[:5].sum())}


def logit_table_energy(logits_abc, p=P):
    """Fourier energy of the net's logit table L[a, b, c] over (a, b), summed over c.

    Returns: diag (48,) folded fraction of total energy on the diagonal (w, w) + (p-w, p-w);
    dc_frac; the donor's measure `old_top5` (verbatim logic of cnb_self_regulation
    `compute_fourier_energy`: top-5 of the UNFOLDED diagonal k=1..96 plus DC, conjugates added);
    and total diagonal fraction."""
    Fl = np.fft.fft2(np.asarray(logits_abc, np.float64), axes=(0, 1))
    energy = np.sum(np.abs(Fl) ** 2, axis=2)
    total = energy.sum()
    w = np.arange(1, N_FREQ + 1)
    diag = (energy[w, w] + energy[p - w, p - w]) / total
    # donor measure, verbatim logic
    diag_energy = [(k, float(energy[k, k])) for k in range(1, p)]
    diag_energy.sort(key=lambda x: x[1], reverse=True)
    key_freqs = [(k, k) for k, _ in diag_energy[:5]]
    mask = np.zeros((p, p), dtype=bool)
    mask[0, 0] = True
    for ka, kb in key_freqs:
        mask[ka, kb] = True
        mask[(p - ka) % p, (p - kb) % p] = True
    old = float(energy[mask].sum() / (total + 1e-10))
    return {"diag": diag, "dc_frac": float(energy[0, 0] / total), "old_top5": old,
            "diag_total": float(diag.sum())}


# =============================================================================
# The four compiled forms at one snapshot
# =============================================================================

class Snapshot:
    """Holds one net and everything its walks need. Train pool = the gate pool; held-out pairs are
    only ever used for the logged-beside accuracy. Labels used by the gate: the net's own argmax."""

    def __init__(self, state, split, train_labels=None, p=P):
        self.p = p
        a_tr, b_tr, a_te, b_te = split
        self.a_tr, self.b_tr, self.a_te, self.b_te = a_tr, b_tr, a_te, b_te
        self.y_tr_true = (a_tr + b_tr) % p
        self.y_te_true = (a_te + b_te) % p
        self.y_tr_fit = self.y_tr_true if train_labels is None else np.asarray(train_labels)
        net = GrokMLP(p=p, hidden_dims=HIDDEN)
        net.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
        net.eval()
        self.net = net
        self.x_tr = onehot(a_tr, b_tr, p)
        self.x_te = onehot(a_te, b_te, p)
        with torch.no_grad():
            self.L_tr = net(self.x_tr)
            self.L_te = net(self.x_te)
            self.h1_tr = F.relu(net.layer0(self.x_tr))
            self.h1_te = F.relu(net.layer0(self.x_te))
        self.own_tr = self.L_tr.argmax(-1).numpy()          # the gate's target
        self.own_te = self.L_te.argmax(-1).numpy()
        self.net_train_acc_fit = float((self.own_tr == self.y_tr_fit).mean())
        self.net_train_acc_true = float((self.own_tr == self.y_tr_true).mean())
        self.net_test_acc = float((self.own_te == self.y_te_true).mean())
        W0 = state["layer0.weight"]
        self.rd = read_layer0(W0, p)
        self.dom = self.rd["dom"]
        self.score = self.rd["E"].sum(0)
        self.support = np.bincount(self.dom, minlength=N_FREQ + 1)[1:]
        self.b0 = torch.as_tensor(state["layer0.bias"])
        # ALS: diagonal-averaged, row-centred net logits over the TRAIN pool (no labels)
        Lc = self.L_tr.double().numpy()
        Lc = Lc - Lc.mean(1, keepdims=True)
        n = len(a_tr)
        gbar = np.zeros(p)
        for d in range(p):
            gbar[d] = Lc[np.arange(n), (a_tr + b_tr - d) % p].mean()
        self.gbar = gbar
        d = np.arange(p)
        self.cos = {w: np.cos(2 * np.pi * w * d / p) for w in range(1, N_FREQ + 1)}
        self.alpha = {w: float(2.0 / p * (gbar * self.cos[w]).sum()) for w in range(1, N_FREQ + 1)}
        self.cache = {f: {} for f in FORMS}

    # --- tie-aware agreement --------------------------------------------------------------
    @staticmethod
    def _agree_g(g, a, b, target, p):
        """Models of the form logits[c] = g((a+b-c) mod p). Agreement with `target` (tie-aware)."""
        g = np.asarray(g, np.float64)
        mx = g.max()
        S = g >= mx - 1e-9 * (1.0 + abs(mx))
        dt = (a + b - target) % p
        return float((S[dt] / S.sum()).mean())

    @staticmethod
    def _agree_logits(L, target):
        L = L if isinstance(L, np.ndarray) else L.numpy()
        mx = L.max(1, keepdims=True)
        S = L >= mx
        hit = S[np.arange(len(L)), target]
        return float((hit / S.sum(1)).mean())

    # --- the forms -------------------------------------------------------------------------
    def g_A(self, K):
        g = np.zeros(self.p)
        for w in K:
            g = g + self.cos[w]
        return g

    def g_ALS(self, K):
        g = np.zeros(self.p)
        for w in K:
            g = g + self.alpha[w] * self.cos[w]
        return g

    def mask_B(self, K):
        Ks = set(K)
        return torch.tensor([1.0 if int(w) in Ks else 0.0 for w in self.dom], dtype=torch.float32)

    def logits_B(self, K, split):
        h1 = self.h1_tr if split == "tr" else self.h1_te
        with torch.no_grad():
            h2 = F.relu(self.net.layer1(h1 * self.mask_B(K)))
            return self.net.head(h2)

    def W0_C(self, K):
        return torch.tensor(filter_layer0(self.rd["Fa"], self.rd["Fb"], K, self.p),
                            dtype=torch.float32)

    def logits_C(self, K, split):
        x = self.x_tr if split == "tr" else self.x_te
        with torch.no_grad():
            h1 = F.relu(x @ self.W0_C(K).T + self.b0)
            h2 = F.relu(self.net.layer1(h1))
            return self.net.head(h2)

    def agree(self, form, K, split, target):
        a, b = (self.a_tr, self.b_tr) if split == "tr" else (self.a_te, self.b_te)
        if form == "A":
            return self._agree_g(self.g_A(K), a, b, target, self.p)
        if form == "ALS":
            return self._agree_g(self.g_ALS(K), a, b, target, self.p)
        if form == "B":
            return self._agree_logits(self.logits_B(K, split), target)
        if form == "C":
            return self._agree_logits(self.logits_C(K, split), target)
        raise ValueError(form)

    def aud(self, form):
        cache = self.cache[form]

        def _aud(rows):
            K = _flat(rows)
            if K not in cache:
                cache[K] = 1.0 - self.agree(form, K, "tr", self.own_tr)
            return cache[K]
        return _aud

    def heldout(self, form, K):
        return self.agree(form, K, "te", self.y_te_true)

    def train_label_acc(self, form, K):
        return self.agree(form, K, "tr", self.y_tr_true)

    def units(self, K):
        return int(np.isin(self.dom, list(K)).sum())

    def producer_order(self):
        w = np.arange(1, N_FREQ + 1)
        return np.lexsort((w, -self.score))          # descending score, ties by w ascending


def analyze_snapshot(snap, rules, n_rand, do_logit_table=True):
    """Every walk at one snapshot. Returns a compact JSON-able record."""
    t0 = time.time()
    cands = [[w] for w in range(1, N_FREQ + 1)]
    prod = snap.producer_order()
    rec = {"net_train_acc_fit": snap.net_train_acc_fit, "net_train_acc_true": snap.net_train_acc_true,
           "net_test_acc": snap.net_test_acc,
           "score": [float(x) for x in snap.score], "support": [int(x) for x in snap.support],
           "dc_energy": float(snap.rd["dc"].sum()),
           "dom_a_eq_b": int((snap.rd["dom_a"] == snap.rd["dom_b"]).sum()),
           "prod_order": [int(i) + 1 for i in prod],
           "dom": [int(x) for x in snap.dom],
           "unit_conc": [round(float(x), 4) for x in
                         snap.rd["E"].max(1) / (snap.rd["E"].sum(1) + 1e-300)],
           "score_spread": spread_stats(snap.score), "n_support": int((snap.support > 0).sum()),
           "alpha": [snap.alpha[w] for w in range(1, N_FREQ + 1)],
           "walks": {}, "cross": {}}
    allK = tuple(range(1, N_FREQ + 1))
    # sanity: form B at K=all reproduces the net bit-for-bit; form C at K=all to float error
    with torch.no_grad():
        LB_tr, LB_te = snap.logits_B(allK, "tr"), snap.logits_B(allK, "te")
        LC_tr = snap.logits_C(allK, "tr")
    rec["B_all_bitexact"] = bool(torch.equal(LB_tr, snap.L_tr) and torch.equal(LB_te, snap.L_te))
    rec["C_all_maxabs"] = float((LC_tr - snap.L_tr).abs().max())
    rec["C_all_argmax_agree"] = float((LC_tr.argmax(-1).numpy() == snap.own_tr).mean())
    rec["units_all"] = snap.units(allK)
    for form in FORMS:
        rec["walks"][form] = {}
        for rname, tol in rules.items():
            traj = []

            def _on_change(n_aud, kept, form=form, traj=traj):
                K = _flat(kept)
                traj.append([int(n_aud), len(K), snap.aud(form)(kept),
                             snap.heldout(form, K), snap.units(K)])

            r = census_walk(snap.aud(form), [], cands, prod, tol=tol, on_change=_on_change)
            K = _flat(r["kept"])
            out = {"K": list(K), "k": len(K), "e_base": r["e_base_gate"],
                   "e_final": r["e_gate_final"], "n_aud": r["n_auditions"],
                   "admit_order": [int(ci) + 1 for ci in r["admitted"]],
                   "heldout": snap.heldout(form, K), "train_label": snap.train_label_acc(form, K),
                   "units": snap.units(K), "traj": traj,
                   "e_steps": [round(s["e_gate"], 6) for s in r["steps"]]}
            rand = []
            for ri in range(n_rand):
                oi = np.random.RandomState(RAND_SEED0 + ri).permutation(N_FREQ)
                rr = census_walk(snap.aud(form), [], cands, oi, tol=tol)
                KR = _flat(rr["kept"])
                rand.append({"K": list(KR), "k": len(KR), "e_final": rr["e_gate_final"],
                             "heldout": snap.heldout(form, KR), "units": snap.units(KR)})
            out["rand"] = rand
            rec["walks"][form][rname] = out
    # cross table: every form at every form's kept K (producer order), per rule
    for rname in rules:
        rec["cross"][rname] = {}
        for gform in FORMS:
            K = tuple(rec["walks"][gform][rname]["K"])
            rec["cross"][rname][gform] = {f: snap.heldout(f, K) for f in FORMS}
            rec["cross"][rname][gform]["agree_own_tr"] = {
                f: snap.agree(f, K, "tr", snap.own_tr) for f in FORMS}
    if do_logit_table:
        x_all = onehot(np.repeat(np.arange(P), P), np.tile(np.arange(P), P), P)
        with torch.no_grad():
            L_all = snap.net(x_all).numpy().reshape(P, P, P)
        lt = logit_table_energy(L_all)
        rec["logit_diag"] = [float(x) for x in lt["diag"]]
        rec["logit_dc_frac"] = lt["dc_frac"]
        rec["logit_old_top5"] = lt["old_top5"]
        rec["logit_diag_total"] = lt["diag_total"]
        rec["logit_diag_spread"] = spread_stats(lt["diag"])
    rec["seconds"] = time.time() - t0
    return rec


# =============================================================================
# SVD baseline (final net) -- the rank sweep of cnb_self_regulation/precompute_svd.py
# (all three layers truncated to the same rank r, biases kept)
# =============================================================================

def svd_sweep(state, split, ranks=None):
    a_tr, b_tr, a_te, b_te = split
    x_tr, x_te = onehot(a_tr, b_tr), onehot(a_te, b_te)
    y_tr, y_te = (a_tr + b_tr) % P, (a_te + b_te) % P
    names = ["layer0", "layer1", "head"]
    Ws = {n: torch.as_tensor(state[f"{n}.weight"]).double() for n in names}
    bs = {n: torch.as_tensor(state[f"{n}.bias"]).double() for n in names}
    svd = {n: torch.linalg.svd(Ws[n], full_matrices=False) for n in names}
    ranks = ranks or list(range(1, 129))
    out = []
    for r in ranks:
        Wr, npar = {}, 0
        for n in names:
            U, S, Vt = svd[n]
            k = min(r, len(S))
            Wr[n] = (U[:, :k] * S[:k]) @ Vt[:k, :]
            d_out, d_in = Ws[n].shape
            npar += min(k * (d_in + d_out), d_in * d_out) + d_out   # store dense when cheaper

        def fwd(x):
            h = F.relu(x.double() @ Wr["layer0"].T + bs["layer0"])
            h = F.relu(h @ Wr["layer1"].T + bs["layer1"])
            return h @ Wr["head"].T + bs["head"]
        with torch.no_grad():
            tr = float((fwd(x_tr).argmax(-1).numpy() == y_tr).mean())
            te = float((fwd(x_te).argmax(-1).numpy() == y_te).mean())
        out.append({"rank": r, "params": int(npar), "train_acc": tr, "heldout_acc": te})
    return out


# =============================================================================
# Modal functions
# =============================================================================

def _run_dir(tag, run):
    return os.path.join(DATA_DIR, tag, run)


def _save_snaps(path, snaps):
    keys = list(snaps.keys())
    arrs = {}
    names = list(snaps[keys[0]]["state"].keys())
    for n in names:
        arrs[n] = np.stack([snaps[k]["state"][n] for k in keys])
    arrs["_labels"] = np.array(keys)
    arrs["_epochs"] = np.array([snaps[k]["epoch"] for k in keys])
    np.savez_compressed(path, **arrs)


def _load_snaps(path):
    z = np.load(path, allow_pickle=False)
    labels = [str(x) for x in z["_labels"]]
    epochs = [int(x) for x in z["_epochs"]]
    names = [n for n in z.files if not n.startswith("_")]
    arrs = {n: z[n] for n in names}          # decompress each stacked array ONCE
    states = [{n: arrs[n][i] for n in names} for i in range(len(labels))]
    return labels, epochs, states


@app.function(cpu=8.0, memory=2048, timeout=3 * 3600, volumes={DATA_DIR: volume})
def train_run(tag: str, run: str, n_epochs: int = N_EPOCHS, snap_every: int = 250,
              seed: int = SEED):
    """Train one net (run = 'true' | 'shuffled'), bank snapshots + curve on the volume."""
    tr_x, tr_y, te_x, te_y = generate_data(P, TRAIN_FRAC, seed)
    if run == "shuffled":
        perm = np.random.RandomState(seed + 1).permutation(len(tr_y))
        tr_y = tr_y[torch.as_tensor(perm)]
    torch.manual_seed(seed)
    model = GrokMLP(p=P, hidden_dims=HIDDEN)
    snap_epochs = list(range(0, n_epochs, snap_every)) + [n_epochs - 1]
    res = train_net(model, tr_x, tr_y, te_x, te_y, n_epochs, LR, WEIGHT_DECAY,
                    eval_every=50, snap_epochs=snap_epochs, tag=f"[{run}]")
    d = _run_dir(tag, run)
    os.makedirs(d, exist_ok=True)
    _save_snaps(os.path.join(d, "snapshots.npz"), res["snapshots"])
    np.save(os.path.join(d, "train_labels.npy"), tr_y.numpy())
    summ = {"run": run, "seed": seed, "n_epochs": n_epochs, "snap_every": snap_every,
            "seconds": res["seconds"], "first_hit": res["first_hit"], "curve": res["curve"],
            "n_snapshots": len(res["snapshots"]), "peak_rss_mb": peak_rss_mb(),
            "n_params": sum(int(v.size) for v in res["final_state"].values())}
    with open(os.path.join(d, "train.json"), "w") as f:
        json.dump(summ, f)
    volume.commit()
    print(f"[{run}] done in {res['seconds']:.0f}s, first_hit={res['first_hit']}, "
          f"rss={peak_rss_mb():.0f}MB", flush=True)
    return {k: v for k, v in summ.items() if k != "curve"}


@app.function(cpu=4.0, memory=2048, timeout=3 * 3600, volumes={DATA_DIR: volume}, max_containers=4)
def analyze_chunk(tag: str, run: str, labels: list, rand_labels: list, n_rand: int = N_RAND):
    """Walk every snapshot in `labels` of one run ('true' | 'shuffled' | 'init'). Random-order
    controls only at `rand_labels`."""
    volume.reload()
    split = split_pairs()
    rules = gate_rules()
    out = {}
    t0 = time.time()
    if run == "init":
        for lab in labels:
            s = int(lab.split("_s")[1])
            torch.manual_seed(s)
            net = GrokMLP(p=P, hidden_dims=HIDDEN)
            state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
            snap = Snapshot(state, split)
            rec = analyze_snapshot(snap, rules, n_rand)
            rec.update({"label": lab, "epoch": -1, "run": run})
            out[lab] = rec
    else:
        d = _run_dir(tag, run)
        all_labels, epochs, states = _load_snaps(os.path.join(d, "snapshots.npz"))
        tl = np.load(os.path.join(d, "train_labels.npy"))
        for lab in labels:
            i = all_labels.index(lab)
            snap = Snapshot(states[i], split, train_labels=tl)
            rec = analyze_snapshot(snap, rules, n_rand if lab in rand_labels else 0)
            rec.update({"label": lab, "epoch": epochs[i], "run": run})
            out[lab] = rec
            print(f"  [{run}] {lab:>7s} test={rec['net_test_acc']:.3f} "
                  f"kB(lt1)={rec['walks']['B']['lt1']['k']} kC(lt1)={rec['walks']['C']['lt1']['k']} "
                  f"B_held={rec['walks']['B']['lt1']['heldout']:.3f} "
                  f"C_held={rec['walks']['C']['lt1']['heldout']:.3f} {rec['seconds']:.1f}s", flush=True)
    print(f"[chunk {run} x{len(labels)}] {time.time() - t0:.0f}s rss={peak_rss_mb():.0f}MB", flush=True)
    return out


@app.function(cpu=4.0, memory=4096, timeout=3600, volumes={DATA_DIR: volume})
def final_extras(tag: str):
    volume.reload()
    split = split_pairs()
    labels, epochs, states = _load_snaps(os.path.join(_run_dir(tag, "true"), "snapshots.npz"))
    i = int(np.argmax(epochs))
    return {"svd": svd_sweep(states[i], split), "final_label": labels[i], "final_epoch": epochs[i]}


def _plan_chunks(tag, rand_every, smoke):
    """Labels per run, split into <= 4-ish chunks for the true run."""
    chunks = []
    for run in ("true", "shuffled"):
        labels, epochs, _ = _load_snaps(os.path.join(_run_dir(tag, run), "snapshots.npz"))
        order = np.argsort(epochs)
        labels = [labels[i] for i in order]
        epochs = [epochs[i] for i in order]
        last = max(epochs)
        rand_labels = [l for l, e in zip(labels, epochs)
                       if e == -1 or e == last or (e + 1) % rand_every == 0 or e % rand_every == 0]
        n_chunk = 1 if (run == "shuffled" or smoke) else 4
        for c in range(n_chunk):
            chunks.append((tag, run, labels[c::n_chunk], rand_labels))
    chunks.append((tag, "init", [f"init_s{s}" for s in INIT_SEEDS], [f"init_s{s}" for s in INIT_SEEDS]))
    return chunks


def _reduce_and_write(tag, train_summ, smoke, rand_every):
    volume.reload()
    chunks = _plan_chunks(tag, rand_every, smoke)
    recs = {}
    for res in analyze_chunk.starmap(chunks):
        for lab, r in res.items():
            recs[f"{r['run']}:{lab}"] = r
    extras = final_extras.remote(tag)
    out = {"tag": tag, "p": P, "n_tr": N_TR, "rules": gate_rules(), "forms": list(FORMS),
           "n_rand": N_RAND, "rand_every": rand_every, "train": train_summ, "records": recs,
           "extras": extras, "smoke": smoke}
    for run in ("true", "shuffled"):
        with open(os.path.join(_run_dir(tag, run), "train.json")) as f:
            out.setdefault("curves", {})[run] = json.load(f)["curve"]
    path = os.path.join(DATA_DIR, tag, "mint.json")
    with open(path, "w") as f:
        json.dump(out, f)
    volume.commit()
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB), {len(recs)} records", flush=True)
    return path


@app.function(cpu=1.0, memory=2048, timeout=6 * 3600, volumes={DATA_DIR: volume})
def mint(tag: str = "m1", smoke: int = 0, n_epochs: int = N_EPOCHS, snap_every: int = 250,
         snap_every_shuf: int = 1000, rand_every: int = 1000):
    """Coordinator (CPU): train the true and shuffled nets in two containers, then fan the
    walks out over <= 4 analysis containers, then the SVD baseline, then write mint.json."""
    t0 = time.time()
    if smoke:
        n_epochs, snap_every, snap_every_shuf, rand_every = 3000, 500, 1000, 1000
    train_summ = {}
    for s in train_run.starmap([(tag, "true", n_epochs, snap_every, SEED),
                                (tag, "shuffled", n_epochs, snap_every_shuf, SEED)]):
        train_summ[s["run"]] = s
    print(f"training done at {time.time() - t0:.0f}s", flush=True)
    path = _reduce_and_write(tag, train_summ, smoke, rand_every)
    print(f"mint done in {time.time() - t0:.0f}s", flush=True)
    return path


@app.function(cpu=1.0, memory=2048, timeout=6 * 3600, volumes={DATA_DIR: volume})
def analyze(tag: str = "m1", smoke: int = 0, rand_every: int = 1000):
    """Re-run the walks on banked snapshots (no retraining)."""
    volume.reload()
    train_summ = {}
    for run in ("true", "shuffled"):
        with open(os.path.join(_run_dir(tag, run), "train.json")) as f:
            train_summ[run] = {k: v for k, v in json.load(f).items() if k != "curve"}
    return _reduce_and_write(tag, train_summ, smoke, rand_every)


# =============================================================================
# Gates (seconds, CPU): the instruments, checked on designed inputs before any run
# =============================================================================

@app.function(cpu=4.0, memory=4096, timeout=1800)
def gates():
    res = {}
    split = split_pairs()
    a_tr, b_tr, a_te, b_te = split
    # G-1: split_pairs reproduces the verbatim generate_data split
    tr_x, tr_y, te_x, te_y = generate_data(P, TRAIN_FRAC, SEED)
    res["G1_split"] = bool(torch.equal(onehot(a_tr, b_tr), tr_x) and torch.equal(onehot(a_te, b_te), te_x)
                           and np.array_equal(tr_y.numpy(), (a_tr + b_tr) % P))
    # G-2: any single nonzero character solves the task exactly (form A, every w, all pairs)
    torch.manual_seed(SEED)
    net = GrokMLP(p=P)
    st = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    snap = Snapshot(st, split)
    ok = all(snap.heldout("A", (w,)) == 1.0 and snap.train_label_acc("A", (w,)) == 1.0
             for w in range(1, N_FREQ + 1))
    res["G2_single_char_solves"] = bool(ok)
    res["G2b_empty_is_chance"] = abs(snap.heldout("A", ()) - 1.0 / P) < 1e-12
    # G-3: form B with K = all is bit-for-bit the net (random init and after 300 steps)
    allK = tuple(range(1, N_FREQ + 1))
    g3 = [bool(torch.equal(snap.logits_B(allK, "tr"), snap.L_tr))]
    res_t = train_net(net, tr_x, tr_y, te_x, te_y, 300, eval_every=100, log_every=0)
    snap2 = Snapshot(res_t["final_state"], split)
    g3.append(bool(torch.equal(snap2.logits_B(allK, "te"), snap2.L_te)))
    res["G3_B_all_bitexact"] = g3
    # G-4: form C with K = all reproduces the net to float error
    res["G4_C_all_maxabs"] = float((snap2.logits_C(allK, "tr") - snap2.L_tr).abs().max())
    # G-5: census_walk on a designed sequence where le0 / lt1 / m3 must differ
    seq = {(): 0.99, (1,): 0.50, (1, 2): 0.50, (1, 2, 3): 0.499, (1, 3): 0.499,
           (1, 3, 4): 0.40, (1, 2, 4): 0.40, (1, 2, 3, 4): 0.40, (1, 4): 0.40}
    got = {}
    for rn, tol in {"le0": 0.0, "lt1": -0.0005, "m3": -0.0057}.items():
        r = census_walk(lambda rows: seq.get(_flat(rows), 1.0), [], [[1], [2], [3], [4]],
                        [0, 1, 2, 3], tol=tol)
        got[rn] = _flat(r["kept"])
    res["G5_walk"] = {k: list(v) for k, v in got.items()}
    res["G5_ok"] = (got["le0"] == (1, 2, 3, 4) and got["lt1"] == (1, 3, 4) and got["m3"] == (1, 4))
    # G-6: ALS closed form == explicit least squares (row intercepts) on the train pool
    rng = np.random.RandomState(0)
    K = (3, 17, 40)
    sub = rng.choice(len(a_tr), 400, replace=False)
    L = snap2.L_tr.double().numpy()[sub]
    rows, cols = np.meshgrid(np.arange(len(sub)), np.arange(P), indexing="ij")
    dd = (a_tr[sub][:, None] + b_tr[sub][:, None] - cols) % P
    X = np.stack([np.cos(2 * np.pi * w * dd / P).ravel() for w in K], 1)
    Xi = np.zeros((X.shape[0], len(sub)))
    Xi[np.arange(X.shape[0]), rows.ravel()] = 1.0
    coef = np.linalg.lstsq(np.concatenate([X, Xi], 1), L.ravel(), rcond=None)[0][:len(K)]
    Lc = L - L.mean(1, keepdims=True)
    gbar = np.array([Lc[np.arange(len(sub)), (a_tr[sub] + b_tr[sub] - d) % P].mean() for d in range(P)])
    closed = [2.0 / P * (gbar * np.cos(2 * np.pi * w * np.arange(P) / P)).sum() for w in K]
    res["G6_als_maxdiff"] = float(np.abs(np.asarray(closed) - coef).max())
    # G-7: the read recovers a planted frequency (and its conjugate folds onto it)
    W0 = np.random.RandomState(1).randn(4, 2 * P) * 1e-3
    for j, w in enumerate([5, 92, 30, 48]):
        W0[j, :P] += np.cos(2 * np.pi * w * np.arange(P) / P + j)
        W0[j, P:] += np.cos(2 * np.pi * w * np.arange(P) / P - j)
    res["G7_read_dom"] = [int(x) for x in read_layer0(W0)["dom"]]
    res["G7_ok"] = res["G7_read_dom"] == [5, 5, 30, 48]
    # G-8: the filter at K={w} leaves only DC + w in the spectrum
    Wf = filter_layer0(np.fft.fft(W0[:, :P], axis=1), np.fft.fft(W0[:, P:], axis=1), (30,))
    rdf = read_layer0(Wf)
    e = rdf["E"]
    res["G8_filter_leak"] = float(np.delete(e, 29, axis=1).sum() / (e.sum() + 1e-300))
    for k, v in res.items():
        print(f"  {k}: {v}", flush=True)
    return res
